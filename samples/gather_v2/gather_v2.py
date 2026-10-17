# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Experimental GatherV2 Python DSL with packed-row and singleton paths.

The general kernel maps one SIMT thread to each output element. Wide, 32-bit
rows use a specialized GM-to-UB-to-GM DMA path. Short byte rows are copied as
32-bit packed words distributed over contiguous SIMT thread blocks. Input and
output use same-width signed integer words to copy data bits without conversion.
"""

from functools import lru_cache
from types import SimpleNamespace
from math import prod
from operator import index
import threading

import torch
import torch_npu  # noqa: F401 -- registers the NPU backend

from cannbotdsl import (
    Channel,
    MemLoc,
    Tensor,
    TensorSpec,
    compile as dsl_compile,
    const_expr,
    dim3,
    dtypes,
    jit,
    simt,
    vf,
)
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops import reg
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.tensor import tile_slice

__all__ = [
    "gather_v2",
    "GatherV2Tiling",
    "GatherV2Kernel",
    "GatherRowsDmaKernel",
    "GatherPackedRowsKernel",
    "GatherScalarRowsKernel",
    "GatherOneIndexKernel",
]

# Host Python integers have arbitrary precision; C/C++ 1U literals are not Python syntax.
INT64_MAX = 9_223_372_036_854_775_807
INT32_MAX = 2_147_483_647
BYTE_ELEMENT_BYTES = 1
WORD_ELEMENT_BYTES = 4
SIMD_VECTOR_BYTES = 256
SIMD_WORD_BITS = WORD_ELEMENT_BYTES * 8
SIMD_WORD_ELEMENTS = SIMD_VECTOR_BYTES // WORD_ELEMENT_BYTES
DEFAULT_THREAD_NUM = 256
PURE_SIMT_THREAD_CAP = 1024
PACKED_MIN_ROW_BYTES = 32
PACKED_MAX_ROW_BYTES = 128
PACKED_MIN_GATHER_ROWS = 8
SINGLE_INDEX_MIN_OUTPUT_ELEMENTS = 1024
DMA_INNER_ELEMENTS = 1024
DMA_MIN_GATHER_ROWS = 1024
DMA_MAX_TILE_ROWS = 28
DMA_CHANNEL_DEPTH = 2
INVALID_AXIS_INDEX = -1
GENERAL_COMPILE_CACHE_SIZE = 128
SPECIAL_COMPILE_CACHE_SIZE = 32
DMA_COMPILE_CACHE_SIZE = 16
_COMPILE_LOCK = threading.Lock()
MAX_CORE_NUM = 65535
WARP_SIZE = 32
PACKED_THREAD_CAP = 256
# The native public API permits 2048.  Pure SIMT launches in this wheel permit
# at most 1024, so that legacy request is accepted but capped at launch time.
SUPPORTED_THREAD_NUMS = (32, 64, 128, 256, 512, 1024, 2048)
_RAW_DTYPE = {
    1: torch.int8,
    2: torch.int16,
    4: torch.int32,
    8: torch.int64,
}
_RAW_DSL_DTYPE = {
    1: dtypes.int8,
    2: dtypes.int16,
    4: dtypes.int32,
    8: dtypes.int64,
}
_SUPPORTED_X_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.uint16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
)


def _integer(value, name):
    """Accept integer-index objects without silently truncating floats or bools."""
    if isinstance(value, bool):
        raise TypeError(f"The parameter {name} must be an integer, not bool.")
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(f"The parameter {name} must be an integer.") from exc


def _shape(value, name):
    """Validate nonnegative dimensions and signed 64-bit element counts."""
    result = tuple(_integer(dim, name) for dim in value)
    if any(dim < 0 for dim in result):
        raise ValueError(f"The parameter {name} must contain nonnegative dimensions.")
    if any(dim > INT64_MAX for dim in result) or prod(result) > INT64_MAX:
        raise OverflowError(f"The parameter {name} exceeds signed 64-bit addressing.")
    return result


class GatherV2Tiling:
    """Normalize [B, O, A, I] x [B, G] and divide flat output among cores."""

    def __init__(
        self,
        x_shape,
        indices_shape,
        axis,
        *,
        batch_dims=0,
        negative_index_support=False,
        core_num=1,
        thread_num=None,
    ):
        """Store compile-time geometry and the launch or buffer configuration."""
        self.x_shape = _shape(x_shape, "x_shape")
        self.indices_shape = _shape(indices_shape, "indices_shape")
        rank = len(self.x_shape)
        indices_rank = len(self.indices_shape)
        axis = _integer(axis, "axis")
        batch_dims = _integer(batch_dims, "batch_dims")
        core_num = _integer(core_num, "core_num")
        if thread_num is not None:
            thread_num = _integer(thread_num, "thread_num")
        if not isinstance(negative_index_support, bool):
            raise TypeError("The negative_index_support parameter must be bool.")
        if rank == 0 or not -rank <= axis < rank:
            raise ValueError(
                "The axis must be in [-x.ndim, x.ndim); x must not be scalar."
            )
        axis %= rank
        if not -indices_rank <= batch_dims <= indices_rank:
            raise ValueError("The batch_dims is outside the indices rank.")
        if batch_dims < 0:
            batch_dims += indices_rank
        if batch_dims > axis:
            raise ValueError("The batch_dims must not exceed the normalized axis.")
        if self.x_shape[:batch_dims] != self.indices_shape[:batch_dims]:
            raise ValueError("The x and indices must have equal batch dimensions.")
        if not 1 <= core_num <= MAX_CORE_NUM:
            raise ValueError("The core_num must be in [1, 65535].")
        if thread_num is not None and thread_num not in SUPPORTED_THREAD_NUMS:
            raise ValueError(
                "The thread_num must be one of 32/64/128/256/512/1024/2048."
            )

        self._prepare_geometry(axis, batch_dims, negative_index_support)
        self._prepare_launch(core_num, thread_num)

    def _prepare_geometry(self, axis, batch_dims, negative_index_support):
        """Compute B/O/A/I/G sizes and validate flattened address ranges."""
        self.axis = axis
        self.batch_dims = batch_dims
        self.negative_index_support = negative_index_support
        # x=[B,O,A,I], indices=[B,G], output=[B,O,G,I].
        # B: shared batch-prefix elements; empty prefixes have product one.
        self.batch_size = prod(self.x_shape[:batch_dims])
        # O: elements between the batch prefix and gather axis.
        self.outer_size = prod(self.x_shape[batch_dims:axis])
        # A: input gather-axis length; legal indices address this dimension.
        self.gather_dim_size = self.x_shape[axis]
        # I: contiguous elements after the gather axis, copied per selected row.
        inner_axis_start = axis + 1
        self.inner_size = prod(self.x_shape[inner_axis_start:])
        # G: number of selected rows per batch, including all indices suffix axes.
        self.gather_size = prod(self.indices_shape[batch_dims:])
        self.output_shape = (
            self.x_shape[:axis]
            + self.indices_shape[batch_dims:]
            + self.x_shape[inner_axis_start:]
        )
        self.output_numel = prod(self.output_shape)
        if (
            max(
                self.batch_size,
                self.outer_size,
                self.gather_size,
                self.inner_size,
                self.output_numel,
            )
            > INT64_MAX
        ):
            raise OverflowError("The flattened shape exceeds signed 64-bit addressing.")

        # Mirror Ascend C's 32/64-bit coordinate split.  Leave room for the
        # final warp's padded lanes before narrowing the SIMT offsets.
        self.use_32bit_offsets = (
            max(prod(self.x_shape), prod(self.indices_shape), self.output_numel)
            <= INT32_MAX - WARP_SIZE
        )

    def _prepare_launch(self, core_num, thread_num):
        """Assign warp-aligned output work and choose supported launch widths."""
        warps = (self.output_numel + WARP_SIZE - 1) // WARP_SIZE
        self.core_num = min(core_num, warps)
        self.per_core_elements = (
            ((warps + self.core_num - 1) // self.core_num) * WARP_SIZE if warps else 0
        )
        if warps:
            self.core_num = (
                self.output_numel + self.per_core_elements - 1
            ) // self.per_core_elements
        self.requested_thread_num = thread_num
        self.thread_num = (
            thread_num
            if thread_num is not None
            else (
                WARP_SIZE if self.per_core_elements <= WARP_SIZE else DEFAULT_THREAD_NUM
            )
        )
        self.thread_num = min(self.thread_num, PURE_SIMT_THREAD_CAP)


@kernel
class GatherV2Kernel:
    """General SIMT: map output [B,O,G,I] coordinates to input [B,O,A,I].

    Supports batch prefixes and arbitrary axis/inner geometry. Invalid indices
    write zero; the host proves whether address arithmetic fits in int32.
    """

    def __init__(self, tiling):
        """Store compile-time geometry and the launch or buffer configuration."""
        self.tiling = tiling

    def __call__(self, gm_x: Tensor, gm_indices: Tensor, gm_y: Tensor):
        # A dim3 [block, thread] launch requires a runtime-thread SIMT VF.
        # vf(mode="simt", thread=N) is instead a mixed-kernel launch form.
        """Execute this kernel using the precomputed geometry and memory views."""
        address_dtype = dtypes.int32 if self.tiling.use_32bit_offsets else dtypes.int64
        with vf(mode="simt"):
            # Coordinate queries already return int32. Widen only for large tensors.
            core_index = simt.block_idx()[0]
            lane = simt.thread_idx()[0]
            threads_per_block = simt.thread_dim()[0]
            if const_expr(not self.tiling.use_32bit_offsets):
                core_index = simt.cast(core_index, dtype=dtypes.int64)
                lane = simt.cast(lane, dtype=dtypes.int64)
                threads_per_block = simt.cast(threads_per_block, dtype=dtypes.int64)
            output_flat_start = core_index * self.tiling.per_core_elements
            for core_local_offset in range(
                lane, self.tiling.per_core_elements, threads_per_block
            ):
                output_flat_offset = output_flat_start + core_local_offset
                if output_flat_offset < self.tiling.output_numel:
                    output_row_index = output_flat_offset // self.tiling.inner_size
                    inner_element_offset = (
                        output_flat_offset - output_row_index * self.tiling.inner_size
                    )
                    batch_outer_index = output_row_index // self.tiling.gather_size
                    batch_index = batch_outer_index // self.tiling.outer_size
                    gather_axis_position = (
                        output_row_index - batch_outer_index * self.tiling.gather_size
                    )
                    indices_flat_offset = (
                        batch_index * self.tiling.gather_size + gather_axis_position
                    )
                    selected_axis_index = gm_indices[indices_flat_offset]
                    # Keep int32-to-int64 widening for negative-index and large-axis safety.
                    if const_expr(gm_indices.dtype != dtypes.int64):
                        selected_axis_index = simt.cast(
                            selected_axis_index, dtype=dtypes.int64
                        )
                    if self.tiling.negative_index_support:
                        if selected_axis_index < 0:
                            selected_axis_index += self.tiling.gather_dim_size
                    if selected_axis_index >= 0:
                        if selected_axis_index < self.tiling.gather_dim_size:
                            selected_axis_offset = selected_axis_index
                            if const_expr(address_dtype == dtypes.int32):
                                selected_axis_offset = simt.cast(
                                    selected_axis_index, dtype=dtypes.int32
                                )
                            input_flat_offset = (
                                batch_outer_index * self.tiling.gather_dim_size
                                + selected_axis_offset
                            ) * self.tiling.inner_size + inner_element_offset
                            gm_y[output_flat_offset] = gm_x[input_flat_offset]
                        else:
                            gm_y[output_flat_offset] = 0
                    else:
                        gm_y[output_flat_offset] = 0


@kernel
class GatherPackedRowsKernel:
    """Packed SIMT for B=O=1: combine four byte elements into one raw word.

    inner_words counts int32 words per row, not original byte elements. Threads
    cover contiguous output words instead of assigning one whole warp per row.
    """

    def __init__(self, axis_size, gather_size, inner_words, negative_indices):
        """Store compile-time geometry and the launch or buffer configuration."""
        self.axis_size = axis_size
        self.gather_size = gather_size
        self.inner_words = inner_words
        self.negative_indices = negative_indices
        output_words = gather_size * inner_words
        self.thread_num = min(PACKED_THREAD_CAP, 2 ** (output_words - 1).bit_length())

    def __call__(self, gm_x: Tensor, gm_indices: Tensor, gm_y: Tensor):
        """Execute this kernel using the precomputed geometry and memory views."""
        address_dtype = (
            dtypes.int32
            if max(self.axis_size, self.gather_size) * self.inner_words
            <= INT32_MAX - self.thread_num
            else dtypes.int64
        )
        # Prepare the core offset on the AIV scalar side, then enter SIMT.
        output_flat_start = get_block_idx() * self.thread_num
        with vf(mode="simt", thread=self.thread_num):
            lane = simt.thread_idx()[0]
            if const_expr(address_dtype == dtypes.int64):
                lane = simt.cast(lane, dtype=dtypes.int64)
            output_flat_offset = output_flat_start + lane
            if output_flat_offset < self.gather_size * self.inner_words:
                output_row_index = output_flat_offset // self.inner_words
                inner_word_offset = (
                    output_flat_offset - output_row_index * self.inner_words
                )
                selected_axis_index = gm_indices[output_row_index]
                # Keep int32-to-int64 widening for negative-index and large-axis safety.
                if const_expr(gm_indices.dtype != dtypes.int64):
                    selected_axis_index = simt.cast(
                        selected_axis_index, dtype=dtypes.int64
                    )
                if self.negative_indices:
                    if selected_axis_index < 0:
                        selected_axis_index += self.axis_size
                if selected_axis_index >= 0 and selected_axis_index < self.axis_size:
                    selected_axis_offset = selected_axis_index
                    if const_expr(address_dtype == dtypes.int32):
                        selected_axis_offset = simt.cast(
                            selected_axis_index, dtype=dtypes.int32
                        )
                    gm_y[output_row_index * self.inner_words + inner_word_offset] = (
                        gm_x[
                            selected_axis_offset * self.inner_words + inner_word_offset
                        ]
                    )
                else:
                    gm_y[output_row_index * self.inner_words + inner_word_offset] = 0


@kernel
class GatherScalarRowsKernel:
    """Scalar-row SIMT for B=O=I=1: each index selects one input element.

    gather_size is the number of output elements; axis_size is the input length.
    AIV prepares each core start, avoiding general B/O/G/I coordinate divisions.
    """

    def __init__(
        self,
        axis_size,
        gather_size,
        core_num,
        per_core_elements,
        thread_num,
        negative_indices,
    ):
        """Store compile-time geometry and the launch or buffer configuration."""
        self.axis_size = axis_size
        self.gather_size = gather_size
        self.core_num = core_num
        self.per_core_elements = per_core_elements
        self.thread_num = thread_num
        self.negative_indices = negative_indices

    def __call__(self, gm_x: Tensor, gm_indices: Tensor, gm_y: Tensor):
        """Execute this kernel using the precomputed geometry and memory views."""
        output_flat_start = get_block_idx() * self.per_core_elements
        with vf(mode="simt", thread=self.thread_num):
            lane = simt.cast(simt.thread_idx()[0], dtype=dtypes.int64)
            for core_local_offset in range(
                lane, self.per_core_elements, self.thread_num
            ):
                output_row_index = output_flat_start + core_local_offset
                if output_row_index < self.gather_size:
                    selected_axis_index = gm_indices[output_row_index]
                    # Keep int32-to-int64 widening for negative-index and large-axis safety.
                    if const_expr(gm_indices.dtype != dtypes.int64):
                        selected_axis_index = simt.cast(
                            selected_axis_index, dtype=dtypes.int64
                        )
                    if self.negative_indices:
                        if selected_axis_index < 0:
                            selected_axis_index += self.axis_size
                    if (
                        selected_axis_index >= 0
                        and selected_axis_index < self.axis_size
                    ):
                        gm_y[output_row_index] = gm_x[selected_axis_index]
                    else:
                        gm_y[output_row_index] = 0


@kernel
class GatherOneIndexKernel:
    """One-index SIMT for B=G=I=1: reuse a scalar index across outer rows.

    output_numel counts outer rows. A valid input address is output row times
    axis_size plus the shared selected index; no output coordinate division is needed.
    """

    def __init__(
        self,
        axis_size,
        output_numel,
        core_num,
        per_core_elements,
        thread_num,
        negative_indices,
    ):
        """Store compile-time geometry and the launch or buffer configuration."""
        self.axis_size = axis_size
        self.output_numel = output_numel
        self.core_num = core_num
        self.per_core_elements = per_core_elements
        self.thread_num = thread_num
        self.negative_indices = negative_indices

    def __call__(self, gm_x: Tensor, gm_indices: Tensor, gm_y: Tensor):
        """Execute this kernel using the precomputed geometry and memory views."""
        with vf(mode="simt"):
            core_index = simt.cast(simt.block_idx()[0], dtype=dtypes.int64)
            lane = simt.cast(simt.thread_idx()[0], dtype=dtypes.int64)
            threads_per_block = simt.cast(simt.thread_dim()[0], dtype=dtypes.int64)
            selected_axis_index = gm_indices[0]
            # Keep int32-to-int64 widening for negative-index and large-axis safety.
            if const_expr(gm_indices.dtype != dtypes.int64):
                selected_axis_index = simt.cast(selected_axis_index, dtype=dtypes.int64)
            if self.negative_indices:
                if selected_axis_index < 0:
                    selected_axis_index += self.axis_size
            output_flat_start = core_index * self.per_core_elements
            if selected_axis_index >= 0 and selected_axis_index < self.axis_size:
                for core_local_offset in range(
                    lane, self.per_core_elements, threads_per_block
                ):
                    output_flat_offset = output_flat_start + core_local_offset
                    if output_flat_offset < self.output_numel:
                        gm_y[output_flat_offset] = gm_x[
                            output_flat_offset * self.axis_size + selected_axis_index
                        ]
            else:
                for core_local_offset in range(
                    lane, self.per_core_elements, threads_per_block
                ):
                    output_flat_offset = output_flat_start + core_local_offset
                    if output_flat_offset < self.output_numel:
                        gm_y[output_flat_offset] = 0


@kernel
class GatherRowsDmaKernel:
    """Wide-row DMA/SIMD for B=O=1 and fixed-size 32-bit inner rows.

    axis_size bounds input row indices; gather_size counts output rows;
    inner_elements is the number of 32-bit elements per row. A three-entry
    tile-local FIFO tracks the latest DMA cache misses. SIMD materializes
    cache-hit rows and invalid zero rows before MTE3 stores the tile.
    """

    def __init__(self, axis_size, gather_size, inner_elements, core_num):
        """Store compile-time geometry and the launch or buffer configuration."""
        self.axis_size = axis_size
        self.gather_size = gather_size
        self.inner_elements = inner_elements
        self.core_num = core_num
        # Leave UB headroom for the two buffers and compiler bookkeeping.
        # Shape-adaptive at compile time: small per-core workloads avoid a
        # mostly empty 28-row tile. Large workloads retain the measured tile.
        # UB bytes <= DMA_MAX_TILE_ROWS * inner_elements * 4 * depth = 224 KiB.
        rows_per_core_target = (gather_size + core_num - 1) // core_num
        self.tile_rows = min(DMA_MAX_TILE_ROWS, max(1, rows_per_core_target))
        tiles = (gather_size + self.tile_rows - 1) // self.tile_rows
        self.rows_per_core = ((tiles + core_num - 1) // core_num) * self.tile_rows

    def __call__(self, gm_x: Tensor, gm_indices: Tensor, gm_y: Tensor):
        """Execute this kernel using the precomputed geometry and memory views."""
        row_slots = Channel(
            MemLoc.UB,
            (self.tile_rows, self.inner_elements),
            dtypes.int32,
            depth=DMA_CHANNEL_DEPTH,
        )
        output_row_start = get_block_idx() * self.rows_per_core
        output_row_stop = min(output_row_start + self.rows_per_core, self.gather_size)
        if output_row_start < output_row_stop:
            slot = row_slots.produce()
            self._fill_slot(
                slot,
                gm_x,
                gm_indices,
                output_row_start,
                min(self.tile_rows, output_row_stop - output_row_start),
            )
            for first_output_row in range(
                output_row_start, output_row_stop, self.tile_rows
            ):
                next_output_row = first_output_row + self.tile_rows
                if next_output_row < output_row_stop:
                    slot = row_slots.produce()
                    self._fill_slot(
                        slot,
                        gm_x,
                        gm_indices,
                        next_output_row,
                        min(self.tile_rows, output_row_stop - next_output_row),
                    )
                ready = row_slots.consume()
                valid_row_count = min(
                    self.tile_rows, output_row_stop - first_output_row
                )
                self._finish_slot(ready, gm_indices, first_output_row, valid_row_count)
                self._store_slot(ready, gm_y, first_output_row, valid_row_count)

    @jit
    def _fill_slot(self, slot, gm_x, gm_indices, first_output_row, valid_row_count):
        # A three-entry FIFO cache is local to this tile. Cache hits point to
        # rows filled by MTE2; they are materialized after Channel.consume().
        """DMA-fill cache-miss input rows; defer duplicate and invalid rows."""
        cached_axis_newest = INVALID_AXIS_INDEX
        cached_axis_previous = INVALID_AXIS_INDEX
        cached_axis_oldest = INVALID_AXIS_INDEX
        for local_row in range(valid_row_count):
            selected_axis_index = gm_indices[first_output_row + local_row]
            if selected_axis_index >= 0 and selected_axis_index < self.axis_size:
                if (
                    selected_axis_index != cached_axis_newest
                    and selected_axis_index != cached_axis_previous
                    and selected_axis_index != cached_axis_oldest
                ):
                    mem_copy(
                        tile_slice(slot, (1, self.inner_elements), (local_row, 0)),
                        tile_slice(
                            gm_x, (1, self.inner_elements), (selected_axis_index, 0)
                        ),
                    )
                    cached_axis_oldest = cached_axis_previous
                    cached_axis_previous = cached_axis_newest
                    cached_axis_newest = selected_axis_index

    @jit
    def _finish_slot(self, ready, gm_indices, first_output_row, valid_row_count):
        # All MTE2 fills are complete at this consumer. SIMD copies only read
        # the original cache-miss rows, so there are no V-store/V-load chains.
        """Materialize FIFO cache hits and invalid zero rows with UB SIMD."""
        cached_axis_newest = INVALID_AXIS_INDEX
        cached_axis_previous = INVALID_AXIS_INDEX
        cached_axis_oldest = INVALID_AXIS_INDEX
        cached_ub_row_newest = 0
        cached_ub_row_previous = 0
        cached_ub_row_oldest = 0
        for local_row in range(valid_row_count):
            selected_axis_index = gm_indices[first_output_row + local_row]
            if selected_axis_index >= 0 and selected_axis_index < self.axis_size:
                cached_source_ub_row = INVALID_AXIS_INDEX
                if selected_axis_index == cached_axis_newest:
                    cached_source_ub_row = cached_ub_row_newest
                elif selected_axis_index == cached_axis_previous:
                    cached_source_ub_row = cached_ub_row_previous
                elif selected_axis_index == cached_axis_oldest:
                    cached_source_ub_row = cached_ub_row_oldest
                else:
                    cached_axis_oldest = cached_axis_previous
                    cached_axis_previous = cached_axis_newest
                    cached_axis_newest = selected_axis_index
                    cached_ub_row_oldest = cached_ub_row_previous
                    cached_ub_row_previous = cached_ub_row_newest
                    cached_ub_row_newest = local_row
                if cached_source_ub_row >= 0:
                    with vf(mode="simd"):
                        mask = reg.create_mask("all", elem_bits=SIMD_WORD_BITS)
                        for inner_element_offset in range(
                            0, self.inner_elements, SIMD_WORD_ELEMENTS
                        ):
                            value = reg.vload(
                                ready,
                                cached_source_ub_row * self.inner_elements
                                + inner_element_offset,
                            )
                            reg.vstore(
                                ready,
                                local_row * self.inner_elements + inner_element_offset,
                                value,
                                mask,
                            )
            else:
                with vf(mode="simd"):
                    mask = reg.create_mask("all", elem_bits=SIMD_WORD_BITS)
                    value = reg.vdups(0, dtypes.int32)
                    for inner_element_offset in range(
                        0, self.inner_elements, SIMD_WORD_ELEMENTS
                    ):
                        reg.vstore(
                            ready,
                            local_row * self.inner_elements + inner_element_offset,
                            value,
                            mask,
                        )

    @jit
    def _store_slot(self, ready, gm_y, first_output_row, valid_row_count):
        """DMA-store a complete tile or individual valid tail rows."""
        if valid_row_count == self.tile_rows:
            mem_copy(
                tile_slice(
                    gm_y,
                    (self.tile_rows, self.inner_elements),
                    (first_output_row // self.tile_rows, 0),
                ),
                tile_slice(ready, (self.tile_rows, self.inner_elements), (0, 0)),
            )
        else:
            for local_row in range(valid_row_count):
                mem_copy(
                    tile_slice(
                        gm_y,
                        (1, self.inner_elements),
                        (first_output_row + local_row, 0),
                    ),
                    tile_slice(ready, (1, self.inner_elements), (local_row, 0)),
                )


class _GatherV2Launcher:
    """Bind compile-time metadata to the corresponding device kernel."""

    def __init__(self, tiling):
        """Store compile-time geometry and the launch or buffer configuration."""
        self.tiling = tiling

    @host
    def run(self, gm_x: Tensor, gm_indices: Tensor, gm_y: Tensor):
        """Launch the selected device implementation with its compile-time geometry."""
        op = GatherV2Kernel(self.tiling)
        op[dim3(self.tiling.core_num), dim3(self.tiling.thread_num)](
            gm_x, gm_indices, gm_y
        )


class _GatherPackedRowsLauncher:
    """Bind compile-time metadata to the corresponding device kernel."""

    def __init__(self, axis_size, gather_size, inner_words, negative_indices):
        """Store compile-time geometry and the launch or buffer configuration."""
        self.axis_size = axis_size
        self.gather_size = gather_size
        self.inner_words = inner_words
        self.negative_indices = negative_indices
        output_words = gather_size * inner_words
        self.thread_num = min(PACKED_THREAD_CAP, 2 ** (output_words - 1).bit_length())
        self.core_num = (output_words + self.thread_num - 1) // self.thread_num

    @host
    def run(self, gm_x: Tensor, gm_indices: Tensor, gm_y: Tensor):
        """Launch the selected device implementation with its compile-time geometry."""
        op = GatherPackedRowsKernel(
            self.axis_size,
            self.gather_size,
            self.inner_words,
            self.negative_indices,
        )
        op[self.core_num](gm_x, gm_indices, gm_y)


class _GatherScalarRowsLauncher:
    """Bind compile-time metadata to the corresponding device kernel."""

    def __init__(
        self,
        axis_size,
        gather_size,
        core_num,
        per_core_elements,
        thread_num,
        negative_indices,
    ):
        """Store compile-time geometry and the launch or buffer configuration."""
        self.axis_size = axis_size
        self.gather_size = gather_size
        self.core_num = core_num
        self.per_core_elements = per_core_elements
        self.thread_num = thread_num
        self.negative_indices = negative_indices

    @host
    def run(self, gm_x: Tensor, gm_indices: Tensor, gm_y: Tensor):
        """Launch the selected device implementation with its compile-time geometry."""
        op = GatherScalarRowsKernel(
            self.axis_size,
            self.gather_size,
            self.core_num,
            self.per_core_elements,
            self.thread_num,
            self.negative_indices,
        )
        op[self.core_num](gm_x, gm_indices, gm_y)


class _GatherOneIndexLauncher:
    """Bind compile-time metadata to the corresponding device kernel."""

    def __init__(
        self,
        axis_size,
        output_numel,
        core_num,
        per_core_elements,
        thread_num,
        negative_indices,
    ):
        """Store compile-time geometry and the launch or buffer configuration."""
        self.axis_size = axis_size
        self.output_numel = output_numel
        self.core_num = core_num
        self.per_core_elements = per_core_elements
        self.thread_num = thread_num
        self.negative_indices = negative_indices

    @host
    def run(self, gm_x: Tensor, gm_indices: Tensor, gm_y: Tensor):
        """Launch the selected device implementation with its compile-time geometry."""
        op = GatherOneIndexKernel(
            self.axis_size,
            self.output_numel,
            self.core_num,
            self.per_core_elements,
            self.thread_num,
            self.negative_indices,
        )
        op[dim3(self.core_num), dim3(self.thread_num)](gm_x, gm_indices, gm_y)


class _GatherRowsDmaLauncher:
    """Bind compile-time metadata to the corresponding device kernel."""

    def __init__(self, axis_size, gather_size, inner_elements, core_num):
        """Store compile-time geometry and the launch or buffer configuration."""
        self.axis_size = axis_size
        self.gather_size = gather_size
        self.inner_elements = inner_elements
        self.core_num = core_num

    @host
    def run(self, gm_x: Tensor, gm_indices: Tensor, gm_y: Tensor):
        """Launch the selected device implementation with its compile-time geometry."""
        op = GatherRowsDmaKernel(
            self.axis_size, self.gather_size, self.inner_elements, self.core_num
        )
        op[self.core_num](gm_x, gm_indices, gm_y)


@lru_cache(maxsize=GENERAL_COMPILE_CACHE_SIZE)
def _compile_gather(
    x_shape,
    indices_shape,
    axis,
    batch_dims,
    negative_index_support,
    core_num,
    thread_num,
    element_bytes,
    index_bytes,
):
    """Compile once per tiling and storage signature; retain the callable."""
    tiling = GatherV2Tiling(
        x_shape,
        indices_shape,
        axis,
        batch_dims=batch_dims,
        negative_index_support=negative_index_support,
        core_num=core_num,
        thread_num=thread_num,
    )
    word_dtype = _RAW_DSL_DTYPE[element_bytes]
    index_dtype = dtypes.int32 if index_bytes == WORD_ELEMENT_BYTES else dtypes.int64
    return dsl_compile(
        _GatherV2Launcher(tiling).run,
        TensorSpec((prod(x_shape),), word_dtype),
        TensorSpec((prod(indices_shape),), index_dtype),
        TensorSpec((tiling.output_numel,), word_dtype),
    )


@lru_cache(maxsize=DMA_COMPILE_CACHE_SIZE)
def _compile_rows_dma(axis_size, gather_size, inner_elements, core_num, index_bytes):
    """Compile and retain an AOT callable for this shape, dtype and launch plan."""
    index_dtype = dtypes.int32 if index_bytes == WORD_ELEMENT_BYTES else dtypes.int64
    return dsl_compile(
        _GatherRowsDmaLauncher(axis_size, gather_size, inner_elements, core_num).run,
        TensorSpec((axis_size, inner_elements), dtypes.int32),
        TensorSpec((gather_size,), index_dtype),
        TensorSpec((gather_size, inner_elements), dtypes.int32),
    )


@lru_cache(maxsize=SPECIAL_COMPILE_CACHE_SIZE)
def _compile_packed_rows(
    axis_size, gather_size, inner_words, negative_indices, index_bytes
):
    """Compile and retain an AOT callable for this shape, dtype and launch plan."""
    index_dtype = dtypes.int32 if index_bytes == WORD_ELEMENT_BYTES else dtypes.int64
    return dsl_compile(
        _GatherPackedRowsLauncher(
            axis_size, gather_size, inner_words, negative_indices
        ).run,
        TensorSpec((axis_size * inner_words,), dtypes.int32),
        TensorSpec((gather_size,), index_dtype),
        TensorSpec((gather_size * inner_words,), dtypes.int32),
    )


@lru_cache(maxsize=SPECIAL_COMPILE_CACHE_SIZE)
def _compile_scalar_rows(
    axis_size,
    gather_size,
    core_num,
    per_core_elements,
    thread_num,
    negative_indices,
    element_bytes,
    index_bytes,
):
    """Compile and retain an AOT callable for this shape, dtype and launch plan."""
    word_dtype = _RAW_DSL_DTYPE[element_bytes]
    index_dtype = dtypes.int32 if index_bytes == WORD_ELEMENT_BYTES else dtypes.int64
    return dsl_compile(
        _GatherScalarRowsLauncher(
            axis_size,
            gather_size,
            core_num,
            per_core_elements,
            thread_num,
            negative_indices,
        ).run,
        TensorSpec((axis_size,), word_dtype),
        TensorSpec((gather_size,), index_dtype),
        TensorSpec((gather_size,), word_dtype),
    )


@lru_cache(maxsize=SPECIAL_COMPILE_CACHE_SIZE)
def _compile_one_index(
    axis_size,
    output_numel,
    core_num,
    per_core_elements,
    thread_num,
    negative_indices,
    element_bytes,
    index_bytes,
):
    """Compile and retain an AOT callable for this shape, dtype and launch plan."""
    word_dtype = _RAW_DSL_DTYPE[element_bytes]
    index_dtype = dtypes.int32 if index_bytes == WORD_ELEMENT_BYTES else dtypes.int64
    return dsl_compile(
        _GatherOneIndexLauncher(
            axis_size,
            output_numel,
            core_num,
            per_core_elements,
            thread_num,
            negative_indices,
        ).run,
        TensorSpec((output_numel * axis_size,), word_dtype),
        TensorSpec((1,), index_dtype),
        TensorSpec((output_numel,), word_dtype),
    )


def _resolve_core_num(x, requested):
    """Resolve and validate the requested number of Vector Cores."""
    device_index = x.device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    available = int(
        getattr(torch.npu.get_device_properties(device_index), "vector_core_num", 0)
    )
    if available <= 0:
        available = 1
    if requested is None:
        return available
    requested = _integer(requested, "core_num")
    if not 1 <= requested <= MAX_CORE_NUM:
        raise ValueError("The core_num must be in [1, 65535].")
    if requested > available:
        raise ValueError(
            f"The core_num={requested} exceeds the device limit {available}."
        )
    return requested


def _validate_gather_inputs(state):
    """Validate tensor storage, dtype and forward-only constraints on the host."""
    if not isinstance(state.x, torch.Tensor) or not isinstance(
        state.indices, torch.Tensor
    ):
        raise TypeError("The x and indices must be torch.Tensor objects.")
    if (
        state.x.device.type not in ("npu", "privateuseone")
        or state.x.device != state.indices.device
    ):
        raise ValueError("The x and indices must be on the same NPU.")
    if not state.x.is_contiguous() or not state.indices.is_contiguous():
        raise ValueError("The x and indices must be contiguous.")
    if state.x.dtype not in _SUPPORTED_X_DTYPES:
        raise TypeError(f"The x dtype {state.x.dtype} is not supported.")
    if state.indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("The indices must have int32 or int64 dtype.")
    if state.x.requires_grad:
        raise ValueError("The gather_v2 is forward-only; detach x before calling.")


def _prepare_gather_views(state):
    """Create same-width integer storage views without converting any payload bits."""
    raw_dtype = _RAW_DTYPE[state.x.element_size()]
    state.raw_x = state.x.view(raw_dtype).view(-1)
    state.raw_output = state.output.view(raw_dtype).view(-1)
    state.flat_indices = state.indices.view(-1)


def _launch_gather_packed(state):
    """Compile or reuse the existing packed launch plan, then execute it."""
    inner_words = state.tiling.inner_size // WORD_ELEMENT_BYTES
    with _COMPILE_LOCK:
        compiled = _compile_packed_rows(
            state.tiling.gather_dim_size,
            state.tiling.gather_size,
            inner_words,
            state.negative_index_support,
            state.indices.element_size(),
        )
    compiled(
        state.raw_x.view(torch.int32).view(-1),
        state.flat_indices,
        state.raw_output.view(torch.int32).view(-1),
    )
    return state.output


def _launch_gather_single_selection(state, compiler, selected_extent):
    """Reuse identical host argument preparation for two distinct SIMT kernels."""
    with _COMPILE_LOCK:
        compiled = compiler(
            state.tiling.gather_dim_size,
            selected_extent,
            state.tiling.core_num,
            state.tiling.per_core_elements,
            state.tiling.thread_num,
            state.negative_index_support,
            state.x.element_size(),
            state.indices.element_size(),
        )
    compiled(state.raw_x, state.flat_indices, state.raw_output)
    return state.output


def _launch_gather_scalar(state):
    """Execute the existing gather_size SIMT launch plan."""
    return _launch_gather_single_selection(
        state, _compile_scalar_rows, state.tiling.gather_size
    )


def _launch_gather_one_index(state):
    """Execute the existing output_numel SIMT launch plan."""
    return _launch_gather_single_selection(
        state, _compile_one_index, state.tiling.output_numel
    )


def _launch_gather_dma(state):
    """Compile or reuse the existing dma launch plan, then execute it."""
    dma_cores = min(state.tiling.core_num, state.tiling.gather_size)
    with _COMPILE_LOCK:
        compiled = _compile_rows_dma(
            state.tiling.gather_dim_size,
            state.tiling.gather_size,
            state.tiling.inner_size,
            dma_cores,
            state.indices.element_size(),
        )
    compiled(
        state.raw_x.view(state.tiling.gather_dim_size, state.tiling.inner_size),
        state.flat_indices,
        state.raw_output.view(state.tiling.gather_size, state.tiling.inner_size),
    )
    return state.output


def _launch_gather_general(state):
    """Execute the general SIMT path with the original compile-cache arguments."""
    with _COMPILE_LOCK:
        compiled = _compile_gather(
            tuple(state.x.shape),
            tuple(state.indices.shape),
            state.axis,
            state.batch_dims,
            state.negative_index_support,
            state.tiling.core_num,
            state.thread_num,
            state.x.element_size(),
            state.indices.element_size(),
        )
    compiled(state.raw_x, state.flat_indices, state.raw_output)
    return state.output


def _packed_rows_has_batchless_byte_storage(state):
    """Check one group of host dispatch constraints, preserving short-circuit order."""
    return (
        state.tiling.batch_size == 1
        and state.tiling.outer_size == 1
        and (state.x.element_size() == BYTE_ELEMENT_BYTES)
    )


def _packed_rows_has_aligned_supported_geometry(state):
    """Check one group of host dispatch constraints, preserving short-circuit order."""
    return (
        PACKED_MIN_ROW_BYTES <= state.tiling.inner_size <= PACKED_MAX_ROW_BYTES
        and state.tiling.inner_size % WORD_ELEMENT_BYTES == 0
        and (
            PACKED_MIN_GATHER_ROWS <= state.tiling.gather_size <= state.tiling.core_num
        )
    )


def _packed_rows_uses_default_threads(state):
    """Check one group of host dispatch constraints, preserving short-circuit order."""
    return state.thread_num is None


def _scalar_rows_has_batchless_unit_inner(state):
    """Check one group of host dispatch constraints, preserving short-circuit order."""
    return (
        state.tiling.batch_size == 1
        and state.tiling.outer_size == 1
        and (state.tiling.inner_size == 1)
    )


def _scalar_rows_uses_default_threads(state):
    """Check one group of host dispatch constraints, preserving short-circuit order."""
    return state.thread_num is None


def _one_index_has_unit_geometry(state):
    """Check one group of host dispatch constraints, preserving short-circuit order."""
    return (
        state.tiling.batch_size == 1
        and state.tiling.gather_size == 1
        and (state.tiling.inner_size == 1)
    )


def _one_index_has_supported_workload(state):
    """Check one group of host dispatch constraints, preserving short-circuit order."""
    return (
        state.tiling.output_numel >= SINGLE_INDEX_MIN_OUTPUT_ELEMENTS
        and state.thread_num is None
    )


def _dma_rows_has_batchless_wide_geometry(state):
    """Check one group of host dispatch constraints, preserving short-circuit order."""
    return (
        state.tiling.batch_size == 1
        and state.tiling.outer_size == 1
        and (state.tiling.inner_size == DMA_INNER_ELEMENTS)
    )


def _dma_rows_has_supported_storage_and_indices(state):
    """Check one group of host dispatch constraints, preserving short-circuit order."""
    return (
        state.x.element_size() == WORD_ELEMENT_BYTES
        and state.tiling.gather_size >= DMA_MIN_GATHER_ROWS
        and (not state.negative_index_support)
    )


def _dma_rows_uses_default_threads(state):
    """Check one group of host dispatch constraints, preserving short-circuit order."""
    return state.thread_num is None


def _prepare_gather_v2_output(state):
    """Allocate output, preserve empty-input behavior and prepare the existing geometry."""
    state.tiling = GatherV2Tiling(
        state.x.shape,
        state.indices.shape,
        state.axis,
        batch_dims=state.batch_dims,
        negative_index_support=state.negative_index_support,
        core_num=_resolve_core_num(state.x, state.core_num),
        thread_num=state.thread_num,
    )
    state.output = torch.empty(
        state.tiling.output_shape, dtype=state.x.dtype, device=state.x.device
    )
    if not state.tiling.output_numel:
        return False
    if not state.tiling.gather_dim_size:
        state.output.zero_()
        return False
    return True


def gather_v2(
    x,
    indices,
    axis,
    *,
    batch_dims=0,
    negative_index_support=False,
    core_num=None,
    thread_num=None,
):
    """Gather along an axis; invalid indices write zero, with optional negatives."""
    state = SimpleNamespace(
        x=x,
        indices=indices,
        axis=axis,
        batch_dims=batch_dims,
        negative_index_support=negative_index_support,
        core_num=core_num,
        thread_num=thread_num,
    )
    _validate_gather_inputs(state)
    if not _prepare_gather_v2_output(state):
        return state.output
    _prepare_gather_views(state)
    if (
        _packed_rows_has_batchless_byte_storage(state)
        and _packed_rows_has_aligned_supported_geometry(state)
        and _packed_rows_uses_default_threads(state)
    ):
        return _launch_gather_packed(state)
    if _scalar_rows_has_batchless_unit_inner(
        state
    ) and _scalar_rows_uses_default_threads(state):
        return _launch_gather_scalar(state)
    if _one_index_has_unit_geometry(state) and _one_index_has_supported_workload(state):
        return _launch_gather_one_index(state)
    if (
        _dma_rows_has_batchless_wide_geometry(state)
        and _dma_rows_has_supported_storage_and_indices(state)
        and _dma_rows_uses_default_threads(state)
    ):
        return _launch_gather_dma(state)
    return _launch_gather_general(state)
