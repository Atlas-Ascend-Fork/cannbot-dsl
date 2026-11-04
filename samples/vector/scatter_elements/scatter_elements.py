# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ScatterElements with DMA initialization and Python DSL SIMT updates/reductions."""

from math import prod
from operator import index
from functools import lru_cache
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401 -- registers the NPU backend

import cannbotdsl as cb
from cannbotdsl import Tensor, dtypes
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.tensor import tile_slice


__all__ = ["scatter_elements"]

# Host Python integers are unbounded; unsigned C/C++ 1U literals do not apply.
INT64_MAX = 9_223_372_036_854_775_807
INT32_MAX = 2_147_483_647
UINT32_MODULUS = 4_294_967_296
UINT32_SIGN_BIT = 2_147_483_648
INT64_WORD_COUNT = 2
BYTE_ELEMENT_BYTES = 1
HALF_WORD_ELEMENT_BYTES = 2
WORD_ELEMENT_BYTES = 4
DOUBLE_WORD_ELEMENT_BYTES = 8
NATIVE_THREAD_COUNT = 512
NATIVE_TILE_BYTES = 4096
DMA_COPY_TILE_BYTES = 32768
DMA_COPY_MIN_BYTES = 1048576
DIRECT_MIN_DATA_ELEMENTS = 4096
SHORT_ROW_MIN_ELEMENTS = 16
SHORT_ROW_MAX_ELEMENTS = 64
SHORT_ROW_SMALL_THREADS = 32
SHORT_ROW_LARGE_THREADS = 64
AXIS0_OWNER_AXIS_ELEMENTS = 10
COMPILE_CACHE_SIZE = 128
PARAM_CACHE_SIZE = 256
# Only the MTE3 initialization writes need to precede the SIMT GM accesses.
COPY_READY_FLAG_ID = 1
MAX_RANK = 8
MAX_CORE_NUM = 65535
THREAD_COUNT = 256
REDUCTION_NONE = 0
REDUCTION_ADD = 1
REDUCTION_MUL = 2
REDUCTIONS = {"none": REDUCTION_NONE, "add": REDUCTION_ADD, "mul": REDUCTION_MUL}
INDEX_STRIDES_BASE = MAX_RANK
UPDATE_STRIDES_BASE = 2 * MAX_RANK
INDEX_SHAPE_BASE = 3 * MAX_RANK
RANK_PARAM = 4 * MAX_RANK
AXIS_PARAM = RANK_PARAM + 1
AXIS_SIZE_PARAM = RANK_PARAM + 2
REDUCTION_PARAM = RANK_PARAM + 3
BOOL_REDUCTION_PARAM = RANK_PARAM + 4
DEVICE_PARAM_COUNT = RANK_PARAM + 5
RAW_STORAGE_DTYPES = {
    BYTE_ELEMENT_BYTES: torch.uint8,
    HALF_WORD_ELEMENT_BYTES: torch.int16,
    WORD_ELEMENT_BYTES: torch.int32,
    DOUBLE_WORD_ELEMENT_BYTES: torch.int64,
}
TORCH_TO_DSL_DTYPE = {
    torch.uint8: dtypes.uint8,
    torch.int8: dtypes.int8,
    torch.int16: dtypes.int16,
    torch.int32: dtypes.int32,
    torch.int64: dtypes.int64,
    torch.float16: dtypes.float16,
    torch.bfloat16: dtypes.bfloat16,
    torch.float32: dtypes.float32,
}
SUPPORTED_DATA_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
)


@kernel
class ScatterDmaCopyKernel:
    """Move large contiguous raw-value tensors through UB before scatter.

    Full tiles use MTE2/MTE3; a partial final tile uses SIMT scalar copies to
    avoid DMA touching bytes beyond either tensor.  The caller passes a raw
    integer view, preserving NaN payload and signed-zero bits in copied data.
    """

    def __init__(self, numel, dtype, element_bytes, core_num, tile_bytes):
        """Store compile-time geometry and the launch or buffer configuration."""
        self.dtype = dtype
        self.tile_elements = tile_bytes // element_bytes
        self.full_tiles, self.tail = divmod(numel, self.tile_elements)
        self.copy_cores = min(core_num, max(1, self.full_tiles))
        self.tiles_per_core = (self.full_tiles + self.copy_cores - 1) // self.copy_cores

    def __call__(self, data: Tensor, output: Tensor):
        """Execute this kernel using the precomputed geometry and memory views."""
        block = get_block_idx()
        tiles = RingBuffer(
            cb.MemLoc.UB, self.dtype, cb.make_layout((self.tile_elements,)), depth=1
        )
        copy_tile_start = block * self.tiles_per_core
        copy_tile_stop = min(copy_tile_start + self.tiles_per_core, self.full_tiles)
        for copy_tile_index in range(copy_tile_start, copy_tile_stop):
            slot = tiles.next_write()
            mem_copy(slot, tile_slice(data, (self.tile_elements,), (copy_tile_index,)))
            ready = tiles.next_read()
            mem_copy(
                tile_slice(output, (self.tile_elements,), (copy_tile_index,)), ready
            )

        if self.tail:
            if block == 0:
                with cb.vf(mode="simt", thread=THREAD_COUNT):
                    thread_index = cb.simt.thread_idx()[0]
                    threads_per_block = cb.simt.thread_dim()[0]
                    for tail_element_offset in range(
                        thread_index, self.tail, threads_per_block
                    ):
                        copy_element_offset = (
                            self.full_tiles * self.tile_elements + tail_element_offset
                        )
                        output[copy_element_offset] = data[copy_element_offset]


class ScatterDmaCopyLauncher:
    """Compile and launch the DMA copy kernel with its static tile plan."""

    def __init__(self, numel, dtype, element_bytes, core_num, tile_bytes):
        """Store compile-time geometry and the launch or buffer configuration."""
        self.numel = numel
        self.dtype = dtype
        self.element_bytes = element_bytes
        self.core_num = core_num
        self.tile_bytes = tile_bytes
        self.copy_cores = min(core_num, max(1, numel // (tile_bytes // element_bytes)))

    @host
    def run(self, data: Tensor, output: Tensor) -> None:
        """Launch the selected device implementation with its compile-time geometry."""
        copier = ScatterDmaCopyKernel(
            self.numel,
            self.dtype,
            self.element_bytes,
            self.core_num,
            self.tile_bytes,
        )
        copier[self.copy_cores](data, output)


def _integer(value, name):
    """Accept integer-index objects and reject bools and nonintegral values."""
    if isinstance(value, bool):
        raise TypeError(f"The parameter {name} must be an integer, not bool.")
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(f"The parameter {name} must be an integer.") from exc


def _shape(value, name):
    """Normalize shape dimensions within signed 64-bit element addressing."""
    result = tuple(_integer(dim, name) for dim in value)
    if any(dim < 0 for dim in result):
        raise ValueError(f"The parameter {name} must contain nonnegative dimensions.")
    if any(dim > INT64_MAX for dim in result) or prod(result) > INT64_MAX:
        raise OverflowError(f"The parameter {name} exceeds signed 64-bit addressing.")
    return result


def _contiguous_strides(shape):
    """Compute contiguous strides in elements, from the last axis backwards."""
    strides = [0] * len(shape)
    running = 1
    for dim in range(len(shape) - 1, -1, -1):
        strides[dim] = running
        running *= shape[dim]
    return tuple(strides)


class ScatterElementsTiling:
    """Validate shapes and prepare element strides for the device kernel."""

    def __init__(
        self, data_shape, indices_shape, updates_shape, axis, reduction="none"
    ):
        """Validate the contract, then derive element counts and contiguous strides."""
        self.data_shape = _shape(data_shape, "data_shape")
        self.indices_shape = _shape(indices_shape, "indices_shape")
        self.updates_shape = _shape(updates_shape, "updates_shape")
        self._validate_ranks()
        self.axis = self._normalize_axis(axis)
        if reduction not in REDUCTIONS:
            raise ValueError("The reduction must be 'none', 'add', or 'mul'.")
        self._validate_shape_relationships()
        self.reduction = reduction
        self.reduction_code = REDUCTIONS[reduction]
        # Counts are logical elements, not bytes or reinterpreted int32 words.
        self.data_numel = prod(self.data_shape)
        self.indices_numel = prod(self.indices_shape)
        # axis_size bounds selected target coordinates; it is not the update count.
        self.axis_size = self.data_shape[self.axis]
        if self.indices_numel and self.axis_size == 0:
            raise ValueError(
                "The scatter axis cannot be empty when indices is nonempty."
            )
        self._prepare_strides()

    @property
    def stride_parameters(self):
        """Return stride and index-shape tables consumed from device memory."""

        # Fixed ABI: three MAX_RANK stride tables, one shape table, then scalars.
        tables = []
        for strides in (self.data_strides, self.index_strides, self.update_strides):
            tables.extend(strides)
            tables.extend([0] * (MAX_RANK - len(strides)))
        tables.extend(self.indices_shape)
        tables.extend([0] * (MAX_RANK - len(self.indices_shape)))
        tables.extend(
            (len(self.data_shape), self.axis, self.axis_size, self.reduction_code)
        )
        return tuple(tables)

    @property
    def has_flat_axis_coordinates(self):
        """A flattened index has the same non-axis coordinates in all tensors."""

        if self.data_numel > INT32_MAX or self.indices_numel > INT32_MAX:
            return False
        if self.indices_shape != self.updates_shape:
            return False
        for dim, sizes in enumerate(zip(self.indices_shape, self.data_shape)):
            index_size, data_size = sizes
            if dim != self.axis and index_size != data_size:
                return False
        return True

    def _validate_ranks(self):
        """Require equally ranked tensors within the fixed metadata table capacity."""
        rank = len(self.data_shape)
        if not 1 <= rank <= MAX_RANK:
            raise ValueError(f"The data rank must be in [1, {MAX_RANK}].")
        if len(self.indices_shape) != rank or len(self.updates_shape) != rank:
            raise ValueError("The data, indices, and updates must have the same rank.")

    def _normalize_axis(self, axis):
        """Validate the signed axis and convert it to a nonnegative dimension."""
        axis = _integer(axis, "axis")
        rank = len(self.data_shape)
        if not -rank <= axis < rank:
            raise ValueError("The axis must be in [-data.ndim, data.ndim).")
        return axis % rank

    def _validate_shape_relationships(self):
        """Every index needs an update; unchanged coordinates must fit the output."""
        for dim, (index_size, update_size) in enumerate(
            zip(self.indices_shape, self.updates_shape)
        ):
            if index_size > update_size:
                raise ValueError(
                    f"The indices.shape[{dim}] must not exceed updates.shape[{dim}]."
                )
            if dim != self.axis and index_size > self.data_shape[dim]:
                raise ValueError(
                    f"The indices.shape[{dim}] must not exceed data.shape[{dim}]."
                )

    def _prepare_strides(self):
        """Compute separate element strides because the three shapes may differ."""
        self.data_strides = _contiguous_strides(self.data_shape)
        self.index_strides = _contiguous_strides(self.indices_shape)
        self.update_strides = _contiguous_strides(self.updates_shape)


def _resolve_core_num(data, requested, data_numel):
    """Select active Vector Cores without launching more blocks than useful."""

    device_index = data.device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    properties = torch.npu.get_device_properties(device_index)
    available = int(getattr(properties, "vector_core_num", 0))
    if available <= 0:
        available = 1
    if requested is None:
        requested = available
    else:
        requested = _integer(requested, "core_num")
        if not 1 <= requested <= MAX_CORE_NUM:
            raise ValueError("The core_num must be in [1, 65535].")
        if requested > available:
            raise ValueError(
                f"The core_num={requested} exceeds the device limit {available}."
            )
    useful_cores = max(1, (data_numel + THREAD_COUNT - 1) // THREAD_COUNT)
    return min(requested, useful_cores)


@kernel
def _scatter_simt(
    data: Tensor, indices: Tensor, updates: Tensor, output: Tensor, params: Tensor
):
    """Each output owner scans updates in source order, avoiding races."""
    with cb.vf(mode="simt"):
        thread_index = dtypes.int64(cb.simt.thread_idx()[0])
        block = dtypes.int64(cb.simt.block_idx()[0])
        threads_per_block = dtypes.int64(cb.simt.thread_dim()[0])
        block_count = dtypes.int64(cb.simt.grid_dim()[0])
        copy_tile_start = block * threads_per_block + thread_index
        step = block_count * threads_per_block
        rank = params[RANK_PARAM]
        axis = params[AXIS_PARAM]
        axis_size = params[AXIS_SIZE_PARAM]
        reduction = params[REDUCTION_PARAM]
        bool_reduction = params[BOOL_REDUCTION_PARAM]
        for output_index in range(copy_tile_start, data.shape[0], step):
            output[output_index] = data[output_index]
            remaining_flat_offset = output_index
            index_non_axis_base = dtypes.int64(0)
            update_non_axis_base = dtypes.int64(0)
            output_axis_coord = dtypes.int64(0)
            inside_indices = True
            for dim in range(rank):
                data_stride = params[dim]
                dimension_coord = remaining_flat_offset // data_stride
                remaining_flat_offset -= dimension_coord * data_stride
                if dim == axis:
                    output_axis_coord = dimension_coord
                else:
                    if dimension_coord >= params[INDEX_SHAPE_BASE + dim]:
                        inside_indices = False
                    index_non_axis_base += (
                        dimension_coord * params[INDEX_STRIDES_BASE + dim]
                    )
                    update_non_axis_base += (
                        dimension_coord * params[UPDATE_STRIDES_BASE + dim]
                    )
            if inside_indices:
                index_axis_size = params[INDEX_SHAPE_BASE + axis]
                index_axis_stride = params[INDEX_STRIDES_BASE + axis]
                update_axis_stride = params[UPDATE_STRIDES_BASE + axis]
                for update_coord in range(index_axis_size):
                    selected_axis_index = indices[
                        index_non_axis_base + update_coord * index_axis_stride
                    ]
                    if (
                        selected_axis_index == output_axis_coord
                        and selected_axis_index >= 0
                        and (selected_axis_index < axis_size)
                    ):
                        update = updates[
                            update_non_axis_base + update_coord * update_axis_stride
                        ]
                        if reduction == REDUCTION_NONE:
                            output[output_index] = update
                        elif reduction == REDUCTION_ADD:
                            # Boolean addition is OR: normalize a nonzero sum to true.
                            if bool_reduction != 0:
                                output[output_index] = data.dtype(
                                    output[output_index] + update != 0
                                )
                            else:
                                output[output_index] = output[output_index] + update
                        else:
                            output[output_index] = output[output_index] * update


@kernel
def _scatter_none_simt(
    data: Tensor, indices: Tensor, updates: Tensor, output: Tensor, params: Tensor
):
    """Ordered replacement without routing payload bits through a loop phi."""

    with cb.vf(mode="simt"):
        thread_index = dtypes.int64(cb.simt.thread_idx()[0])
        block = dtypes.int64(cb.simt.block_idx()[0])
        threads_per_block = dtypes.int64(cb.simt.thread_dim()[0])
        block_count = dtypes.int64(cb.simt.grid_dim()[0])
        rank = params[RANK_PARAM]
        axis = params[AXIS_PARAM]
        axis_size = params[AXIS_SIZE_PARAM]

        for output_index in range(
            block * threads_per_block + thread_index,
            data.shape[0],
            block_count * threads_per_block,
        ):
            remaining_flat_offset = output_index
            index_non_axis_base = dtypes.int64(0)
            update_non_axis_base = dtypes.int64(0)
            output_axis_coord = dtypes.int64(0)
            inside_indices = True
            for dim in range(rank):
                data_stride = params[dim]
                dimension_coord = remaining_flat_offset // data_stride
                remaining_flat_offset -= dimension_coord * data_stride
                if dim == axis:
                    output_axis_coord = dimension_coord
                else:
                    if dimension_coord >= params[INDEX_SHAPE_BASE + dim]:
                        inside_indices = False
                    index_non_axis_base += (
                        dimension_coord * params[INDEX_STRIDES_BASE + dim]
                    )
                    update_non_axis_base += (
                        dimension_coord * params[UPDATE_STRIDES_BASE + dim]
                    )

            found = False
            last_update_offset = dtypes.int64(0)
            if inside_indices:
                index_axis_size = params[INDEX_SHAPE_BASE + axis]
                index_axis_stride = params[INDEX_STRIDES_BASE + axis]
                update_axis_stride = params[UPDATE_STRIDES_BASE + axis]
                for update_coord in range(index_axis_size):
                    selected_axis_index = indices[
                        index_non_axis_base + update_coord * index_axis_stride
                    ]
                    if (
                        selected_axis_index == output_axis_coord
                        and selected_axis_index >= 0
                        and selected_axis_index < axis_size
                    ):
                        found = True
                        last_update_offset = (
                            update_non_axis_base + update_coord * update_axis_stride
                        )
            if found:
                output[output_index] = updates[last_update_offset]
            else:
                output[output_index] = data[output_index]


@kernel
def _scatter_int64_reduce_simt(
    data: Tensor, indices: Tensor, updates: Tensor, output: Tensor, params: Tensor
):
    """Ordered int64 add/mul via two 32-bit words per logical element."""

    with cb.vf(mode="simt"):
        thread_index = dtypes.int64(cb.simt.thread_idx()[0])
        block = dtypes.int64(cb.simt.block_idx()[0])
        threads_per_block = dtypes.int64(cb.simt.thread_dim()[0])
        block_count = dtypes.int64(cb.simt.grid_dim()[0])
        rank = params[RANK_PARAM]
        axis = params[AXIS_PARAM]
        axis_size = params[AXIS_SIZE_PARAM]
        reduction = params[REDUCTION_PARAM]

        for output_index in range(
            block * threads_per_block + thread_index,
            data.shape[0] // INT64_WORD_COUNT,
            block_count * threads_per_block,
        ):
            output_low_word_offset = output_index * INT64_WORD_COUNT
            low = data[output_low_word_offset]
            high = data[output_low_word_offset + 1]
            remaining_flat_offset = output_index
            index_non_axis_base = dtypes.int64(0)
            update_non_axis_base = dtypes.int64(0)
            output_axis_coord = dtypes.int64(0)
            inside_indices = True
            for dim in range(rank):
                data_stride = params[dim]
                dimension_coord = remaining_flat_offset // data_stride
                remaining_flat_offset -= dimension_coord * data_stride
                if dim == axis:
                    output_axis_coord = dimension_coord
                else:
                    if dimension_coord >= params[INDEX_SHAPE_BASE + dim]:
                        inside_indices = False
                    index_non_axis_base += (
                        dimension_coord * params[INDEX_STRIDES_BASE + dim]
                    )
                    update_non_axis_base += (
                        dimension_coord * params[UPDATE_STRIDES_BASE + dim]
                    )

            if inside_indices:
                index_axis_size = params[INDEX_SHAPE_BASE + axis]
                index_axis_stride = params[INDEX_STRIDES_BASE + axis]
                update_axis_stride = params[UPDATE_STRIDES_BASE + axis]
                for update_coord in range(index_axis_size):
                    selected_axis_index = indices[
                        index_non_axis_base + update_coord * index_axis_stride
                    ]
                    if (
                        selected_axis_index == output_axis_coord
                        and selected_axis_index >= 0
                        and selected_axis_index < axis_size
                    ):
                        current_low = cb.simt.bitcast(low, dtype=dtypes.int32)
                        current_high = cb.simt.bitcast(high, dtype=dtypes.int32)
                        update_word = (
                            update_non_axis_base + update_coord * update_axis_stride
                        ) * INT64_WORD_COUNT
                        update_low = updates[update_word]
                        update_high = updates[update_word + 1]
                        current_low_unsigned = dtypes.int64(dtypes.uint32(current_low))
                        update_low_unsigned = dtypes.int64(dtypes.uint32(update_low))
                        # Add carries from the unsigned low word into the high word.
                        if reduction == REDUCTION_ADD:
                            new_low = current_low + update_low
                            carry = dtypes.int32(
                                current_low_unsigned + update_low_unsigned
                                >= UINT32_MODULUS
                            )
                            high = current_high + update_high + carry
                            low = new_low
                        else:
                            new_low = current_low * update_low
                            # Mul combines the low product high word and cross products.
                            new_high = cb.simt.mulhi(current_low, update_low)
                            if current_low_unsigned >= UINT32_SIGN_BIT:
                                new_high = new_high + update_low
                            if update_low_unsigned >= UINT32_SIGN_BIT:
                                new_high = new_high + current_low
                            high = (
                                new_high
                                + current_low * update_high
                                + current_high * update_low
                            )
                            low = new_low
            output[output_low_word_offset] = low
            output[output_low_word_offset + 1] = high


@kernel
def _copy_simt(data: Tensor, output: Tensor):
    """Copy raw one-dimensional storage using a SIMT grid-stride loop."""
    with cb.vf(mode="simt"):
        thread_index = dtypes.int64(cb.simt.thread_idx()[0])
        block = dtypes.int64(cb.simt.block_idx()[0])
        threads_per_block = dtypes.int64(cb.simt.thread_dim()[0])
        block_count = dtypes.int64(cb.simt.grid_dim()[0])
        for output_index in range(
            block * threads_per_block + thread_index,
            data.shape[0],
            block_count * threads_per_block,
        ):
            output[output_index] = data[output_index]


@kernel
def _scatter_direct_none_simt(
    indices: Tensor, updates: Tensor, output: Tensor, params: Tensor
):
    """One writer per update; repeated destinations follow device scheduling."""

    with cb.vf(mode="simt"):
        thread_index = dtypes.int64(cb.simt.thread_idx()[0])
        block = dtypes.int64(cb.simt.block_idx()[0])
        threads_per_block = dtypes.int64(cb.simt.thread_dim()[0])
        block_count = dtypes.int64(cb.simt.grid_dim()[0])
        axis = params[AXIS_PARAM]
        axis_size = params[AXIS_SIZE_PARAM]
        rank = params[RANK_PARAM]
        for index_flat_offset in range(
            block * threads_per_block + thread_index,
            indices.shape[0],
            block_count * threads_per_block,
        ):
            remaining_flat_offset = index_flat_offset
            data_offset = 0
            update_flat_offset = 0
            for dim in range(rank):
                index_stride = params[INDEX_STRIDES_BASE + dim]
                dimension_coord = remaining_flat_offset // index_stride
                remaining_flat_offset -= dimension_coord * index_stride
                update_flat_offset += (
                    dimension_coord * params[UPDATE_STRIDES_BASE + dim]
                )
                if dim != axis:
                    data_offset += dimension_coord * params[dim]
            selected_axis_index = indices[index_flat_offset]
            if selected_axis_index >= 0 and selected_axis_index < axis_size:
                output_flat_offset = data_offset + selected_axis_index * params[axis]
                output[output_flat_offset] = updates[update_flat_offset]


@kernel
def _scatter_direct_add_simt(
    indices: Tensor, updates: Tensor, output: Tensor, params: Tensor
):
    """Atomic add for float32; host restricts use to supported types."""

    with cb.vf(mode="simt"):
        thread_index = dtypes.int64(cb.simt.thread_idx()[0])
        block = dtypes.int64(cb.simt.block_idx()[0])
        threads_per_block = dtypes.int64(cb.simt.thread_dim()[0])
        block_count = dtypes.int64(cb.simt.grid_dim()[0])
        axis = params[AXIS_PARAM]
        axis_size = params[AXIS_SIZE_PARAM]
        rank = params[RANK_PARAM]
        for index_flat_offset in range(
            block * threads_per_block + thread_index,
            indices.shape[0],
            block_count * threads_per_block,
        ):
            remaining_flat_offset = index_flat_offset
            data_offset = 0
            update_flat_offset = 0
            for dim in range(rank):
                index_stride = params[INDEX_STRIDES_BASE + dim]
                dimension_coord = remaining_flat_offset // index_stride
                remaining_flat_offset -= dimension_coord * index_stride
                update_flat_offset += (
                    dimension_coord * params[UPDATE_STRIDES_BASE + dim]
                )
                if dim != axis:
                    data_offset += dimension_coord * params[dim]
            selected_axis_index = indices[index_flat_offset]
            if selected_axis_index >= 0 and selected_axis_index < axis_size:
                output_flat_offset = data_offset + selected_axis_index * params[axis]
                cb.simt.atomic_add(
                    output.ptr(output_flat_offset), updates[update_flat_offset]
                )


@kernel
def _scatter_unique_add_simt(
    indices: Tensor, updates: Tensor, output: Tensor, params: Tensor
):
    """Read-modify-write each distinct float32 destination without an atomic."""

    with cb.vf(mode="simt"):
        thread_index = dtypes.int64(cb.simt.thread_idx()[0])
        block = dtypes.int64(cb.simt.block_idx()[0])
        threads_per_block = dtypes.int64(cb.simt.thread_dim()[0])
        block_count = dtypes.int64(cb.simt.grid_dim()[0])
        axis = params[AXIS_PARAM]
        axis_size = params[AXIS_SIZE_PARAM]
        rank = params[RANK_PARAM]
        for index_flat_offset in range(
            block * threads_per_block + thread_index,
            indices.shape[0],
            block_count * threads_per_block,
        ):
            remaining_flat_offset = index_flat_offset
            data_offset = 0
            update_flat_offset = 0
            for dim in range(rank):
                index_stride = params[INDEX_STRIDES_BASE + dim]
                dimension_coord = remaining_flat_offset // index_stride
                remaining_flat_offset -= dimension_coord * index_stride
                update_flat_offset += (
                    dimension_coord * params[UPDATE_STRIDES_BASE + dim]
                )
                if dim != axis:
                    data_offset += dimension_coord * params[dim]
            selected_axis_index = indices[index_flat_offset]
            if selected_axis_index >= 0 and selected_axis_index < axis_size:
                output_flat_offset = data_offset + selected_axis_index * params[axis]
                output[output_flat_offset] = (
                    output[output_flat_offset] + updates[update_flat_offset]
                )


@kernel
def _scatter_compact_unique_add_simt(
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    axis_stride: cb.Constexpr[int],
    axis_size: cb.Constexpr[int],
):
    """Indices occupy a prefix-one, contiguous last-axis segment."""

    with cb.vf(mode="simt"):
        thread_index = dtypes.int64(cb.simt.thread_idx()[0])
        block = dtypes.int64(cb.simt.block_idx()[0])
        threads_per_block = dtypes.int64(cb.simt.thread_dim()[0])
        block_count = dtypes.int64(cb.simt.grid_dim()[0])
        for index_flat_offset in range(
            block * threads_per_block + thread_index,
            indices.shape[0],
            block_count * threads_per_block,
        ):
            selected_axis_index = indices[index_flat_offset]
            if selected_axis_index >= 0 and selected_axis_index < axis_size:
                selected64 = selected_axis_index
                # Preserve int32 sign extension; int64 loads already have the target type.
                if cb.const_expr(indices.dtype != dtypes.int64):
                    selected64 = dtypes.int64(selected_axis_index)
                output_flat_offset = selected64 * axis_stride + index_flat_offset
                output[output_flat_offset] = (
                    output[output_flat_offset] + updates[index_flat_offset]
                )


@kernel
def _scatter_direct_flat_none_simt(
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    axis: cb.Constexpr[int],
    axis_size: cb.Constexpr[int],
    index_axis_size: cb.Constexpr[int],
    axis_inner_elements: cb.Constexpr[int],
):
    """Map a flat update index to the output with one flattened axis split."""

    with cb.vf(mode="simt"):
        thread_index = dtypes.int64(cb.simt.thread_idx()[0])
        block = dtypes.int64(cb.simt.block_idx()[0])
        threads_per_block = dtypes.int64(cb.simt.thread_dim()[0])
        block_count = dtypes.int64(cb.simt.grid_dim()[0])
        index_outer_span = index_axis_size * axis_inner_elements
        data_outer_span = axis_size * axis_inner_elements

        for index_flat_offset in range(
            block * threads_per_block + thread_index,
            indices.shape[0],
            block_count * threads_per_block,
        ):
            selected_axis_index = indices[index_flat_offset]
            if selected_axis_index >= 0 and selected_axis_index < axis_size:
                linear32 = dtypes.int32(index_flat_offset)
                selected_axis_index32 = selected_axis_index
                # Bounds were checked above; only int64 inputs need narrowing.
                if cb.const_expr(indices.dtype != dtypes.int32):
                    selected_axis_index32 = dtypes.int32(selected_axis_index)
                output_flat_offset = 0
                # Leading axis: keep the inner coordinate and replace the outer axis.
                if axis == 0:
                    output_flat_offset = (
                        selected_axis_index32 * axis_inner_elements
                        + linear32 % axis_inner_elements
                    )
                # Unit inner stride: preserve the outer group and replace its axis position.
                elif axis_inner_elements == 1:
                    output_flat_offset = (
                        linear32 // index_axis_size
                    ) * axis_size + selected_axis_index32
                else:
                    # Middle axis: keep both outer-group and trailing coordinates.
                    outer_group_index = linear32 // index_outer_span
                    inner_coord = linear32 % axis_inner_elements
                    output_flat_offset = (
                        outer_group_index * data_outer_span
                        + selected_axis_index32 * axis_inner_elements
                        + inner_coord
                    )
                output[output_flat_offset] = updates[index_flat_offset]


@kernel
def _scatter_direct_flat_add_simt(
    indices: Tensor, updates: Tensor, output: Tensor, params: Tensor
):
    """Atomic float32 add using a flattened axis coordinate transform."""

    with cb.vf(mode="simt"):
        thread_index = dtypes.int64(cb.simt.thread_idx()[0])
        block = dtypes.int64(cb.simt.block_idx()[0])
        threads_per_block = dtypes.int64(cb.simt.thread_dim()[0])
        block_count = dtypes.int64(cb.simt.grid_dim()[0])
        axis = params[AXIS_PARAM]
        data_axis_size_64 = params[AXIS_SIZE_PARAM]
        data_axis_size = dtypes.int32(data_axis_size_64)
        index_axis_size = dtypes.int32(params[INDEX_SHAPE_BASE + axis])
        axis_inner_elements = dtypes.int32(params[INDEX_STRIDES_BASE + axis])
        index_outer_span = index_axis_size * axis_inner_elements
        data_outer_span = data_axis_size * axis_inner_elements

        for index_flat_offset in range(
            block * threads_per_block + thread_index,
            indices.shape[0],
            block_count * threads_per_block,
        ):
            selected_axis_index = indices[index_flat_offset]
            if selected_axis_index >= 0 and selected_axis_index < data_axis_size_64:
                linear32 = dtypes.int32(index_flat_offset)
                selected_axis_index32 = selected_axis_index
                # Bounds were checked above; only int64 inputs need narrowing.
                if cb.const_expr(indices.dtype != dtypes.int32):
                    selected_axis_index32 = dtypes.int32(selected_axis_index)
                output_flat_offset = 0
                # Leading axis: keep the inner coordinate and replace the outer axis.
                if axis == 0:
                    output_flat_offset = (
                        selected_axis_index32 * axis_inner_elements
                        + linear32 % axis_inner_elements
                    )
                # Unit inner stride: preserve the outer group and replace its axis position.
                elif axis_inner_elements == 1:
                    output_flat_offset = (
                        linear32 // index_axis_size
                    ) * data_axis_size + selected_axis_index32
                else:
                    # Middle axis: keep both outer-group and trailing coordinates.
                    outer_group_index = linear32 // index_outer_span
                    inner_coord = linear32 % axis_inner_elements
                    output_flat_offset = (
                        outer_group_index * data_outer_span
                        + selected_axis_index32 * axis_inner_elements
                        + inner_coord
                    )
                cb.simt.atomic_add(
                    output.ptr(output_flat_offset), updates[index_flat_offset]
                )


@host
def _launch_scatter(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    params: Tensor,
    core_num: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _scatter_simt[cb.dim3(core_num), cb.dim3(THREAD_COUNT)](
        data, indices, updates, output, params
    )


@host
def _launch_copy_only(
    data: Tensor, output: Tensor, core_num: cb.Constexpr[int]
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _copy_simt[cb.dim3(core_num), cb.dim3(THREAD_COUNT)](data, output)


@host
def _launch_scatter_none(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    params: Tensor,
    core_num: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _scatter_none_simt[cb.dim3(core_num), cb.dim3(THREAD_COUNT)](
        data, indices, updates, output, params
    )


@host
def _launch_scatter_int64_reduce(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    params: Tensor,
    core_num: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _scatter_int64_reduce_simt[cb.dim3(core_num), cb.dim3(THREAD_COUNT)](
        data, indices, updates, output, params
    )


@host
def _launch_scatter_direct_none(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    params: Tensor,
    copy_cores: cb.Constexpr[int],
    scatter_cores: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _copy_simt[cb.dim3(copy_cores), cb.dim3(THREAD_COUNT)](data, output)
    _scatter_direct_none_simt[cb.dim3(scatter_cores), cb.dim3(THREAD_COUNT)](
        indices, updates, output, params
    )


@host
def _launch_scatter_direct_add(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    params: Tensor,
    copy_cores: cb.Constexpr[int],
    scatter_cores: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _copy_simt[cb.dim3(copy_cores), cb.dim3(THREAD_COUNT)](data, output)
    _scatter_direct_add_simt[cb.dim3(scatter_cores), cb.dim3(THREAD_COUNT)](
        indices, updates, output, params
    )


@host
def _launch_scatter_unique_add(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    params: Tensor,
    copy_cores: cb.Constexpr[int],
    scatter_cores: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _copy_simt[cb.dim3(copy_cores), cb.dim3(THREAD_COUNT)](data, output)
    _scatter_unique_add_simt[cb.dim3(scatter_cores), cb.dim3(THREAD_COUNT)](
        indices, updates, output, params
    )


@host
def _launch_scatter_compact_unique_add(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    params: Tensor,
    copy_cores: cb.Constexpr[int],
    scatter_cores: cb.Constexpr[int],
    axis_stride: cb.Constexpr[int],
    axis_size: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _copy_simt[cb.dim3(copy_cores), cb.dim3(THREAD_COUNT)](data, output)
    _scatter_compact_unique_add_simt[cb.dim3(scatter_cores), cb.dim3(THREAD_COUNT)](
        indices, updates, output, axis_stride, axis_size
    )


@host
def _launch_scatter_direct_flat_none(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    params: Tensor,
    copy_cores: cb.Constexpr[int],
    scatter_cores: cb.Constexpr[int],
    axis: cb.Constexpr[int],
    axis_size: cb.Constexpr[int],
    index_axis_size: cb.Constexpr[int],
    axis_inner_elements: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _copy_simt[cb.dim3(copy_cores), cb.dim3(THREAD_COUNT)](data, output)
    _scatter_direct_flat_none_simt[cb.dim3(scatter_cores), cb.dim3(THREAD_COUNT)](
        indices, updates, output, axis, axis_size, index_axis_size, axis_inner_elements
    )


@host
def _launch_scatter_direct_flat_add(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    params: Tensor,
    copy_cores: cb.Constexpr[int],
    scatter_cores: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _copy_simt[cb.dim3(copy_cores), cb.dim3(THREAD_COUNT)](data, output)
    _scatter_direct_flat_add_simt[cb.dim3(scatter_cores), cb.dim3(THREAD_COUNT)](
        indices, updates, output, params
    )


@host
def _launch_scatter_direct_none_only(
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    params: Tensor,
    scatter_cores: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _scatter_direct_none_simt[cb.dim3(scatter_cores), cb.dim3(THREAD_COUNT)](
        indices, updates, output, params
    )


@host
def _launch_scatter_direct_flat_none_only(
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    params: Tensor,
    scatter_cores: cb.Constexpr[int],
    axis: cb.Constexpr[int],
    axis_size: cb.Constexpr[int],
    index_axis_size: cb.Constexpr[int],
    axis_inner_elements: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _scatter_direct_flat_none_simt[cb.dim3(scatter_cores), cb.dim3(THREAD_COUNT)](
        indices, updates, output, axis, axis_size, index_axis_size, axis_inner_elements
    )


@host
def _launch_scatter_unique_add_only(
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    params: Tensor,
    scatter_cores: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _scatter_unique_add_simt[cb.dim3(scatter_cores), cb.dim3(THREAD_COUNT)](
        indices, updates, output, params
    )


@host
def _launch_scatter_compact_unique_add_only(
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    params: Tensor,
    scatter_cores: cb.Constexpr[int],
    axis_stride: cb.Constexpr[int],
    axis_size: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _scatter_compact_unique_add_simt[cb.dim3(scatter_cores), cb.dim3(THREAD_COUNT)](
        indices, updates, output, axis_stride, axis_size
    )


@host
def _launch_scatter_direct_add_only(
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    params: Tensor,
    scatter_cores: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _scatter_direct_add_simt[cb.dim3(scatter_cores), cb.dim3(THREAD_COUNT)](
        indices, updates, output, params
    )


@kernel
def _scatter_axis0_owner_add_simt(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    axis_inner_elements: cb.Constexpr[int],
    index_axis_size: cb.Constexpr[int],
):
    """One output owner sums axis-0 updates in source order, without atomics."""

    with cb.vf(mode="simt"):
        thread_index = dtypes.int64(cb.simt.thread_idx()[0])
        block = dtypes.int64(cb.simt.block_idx()[0])
        threads_per_block = dtypes.int64(cb.simt.thread_dim()[0])
        block_count = dtypes.int64(cb.simt.grid_dim()[0])
        for output_index in range(
            block * threads_per_block + thread_index,
            data.shape[0],
            block_count * threads_per_block,
        ):
            output_flat_index32 = dtypes.int32(output_index)
            output_axis_coord = output_flat_index32 // axis_inner_elements
            inner_coord = output_flat_index32 % axis_inner_elements
            accumulated_value = data[output_index]
            for source_axis in cb.range_constexpr(index_axis_size):
                update_flat_index = source_axis * axis_inner_elements + inner_coord
                if indices[update_flat_index] == output_axis_coord:
                    accumulated_value = accumulated_value + updates[update_flat_index]
            output[output_index] = accumulated_value


@host
def _launch_axis0_owner_add(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    core_num: cb.Constexpr[int],
    axis_inner_elements: cb.Constexpr[int],
    index_axis_size: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _scatter_axis0_owner_add_simt[cb.dim3(core_num), cb.dim3(NATIVE_THREAD_COUNT)](
        data, indices, updates, output, axis_inner_elements, index_axis_size
    )


@lru_cache(maxsize=COMPILE_CACHE_SIZE)
def _compiled_axis0_owner_add(
    data_numel,
    indices_numel,
    value_dtype,
    index_dtype,
    core_num,
    axis_inner_elements,
    index_axis_size,
):
    """Compile and retain an AOT callable for this shape, dtype and launch plan."""
    descriptors = (
        cb.TensorSpec((data_numel,), value_dtype),
        cb.TensorSpec((indices_numel,), index_dtype),
        cb.TensorSpec((indices_numel,), value_dtype),
        cb.TensorSpec((data_numel,), value_dtype),
    )
    return cb.compile(
        _launch_axis0_owner_add,
        *descriptors,
        core_num,
        axis_inner_elements,
        index_axis_size,
    )


@kernel
def _scatter_short_row_reduce_simt(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    axis_size: cb.Constexpr[int],
    index_axis_size: cb.Constexpr[int],
    reduction_code: cb.Constexpr[int],
):
    """One block per short axis row, with ordered owners for every output.

    This handles repeated indices without atomics.  Flat non-axis dimensions
    and a unit axis stride make each row a contiguous segment in all tensors.
    """
    with cb.vf(mode="simt"):
        thread_index = cb.simt.thread_idx()[0]
        block = cb.simt.block_idx()[0]
        block_count = cb.simt.grid_dim()[0]
        if thread_index < axis_size:
            for row in range(block, data.shape[0] // axis_size, block_count):
                output_index = row * axis_size + thread_index
                index_non_axis_base = row * index_axis_size
                output[output_index] = data[output_index]
                for source_axis in cb.range_constexpr(index_axis_size):
                    update_flat_index = index_non_axis_base + source_axis
                    if indices[update_flat_index] == thread_index:
                        update = updates[update_flat_index]
                        if reduction_code == REDUCTION_ADD:
                            output[output_index] = output[output_index] + update
                        else:
                            output[output_index] = output[output_index] * update


@host
def _launch_short_row_reduce(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    core_num: cb.Constexpr[int],
    thread_num: cb.Constexpr[int],
    axis_size: cb.Constexpr[int],
    index_axis_size: cb.Constexpr[int],
    reduction_code: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _scatter_short_row_reduce_simt[cb.dim3(core_num), cb.dim3(thread_num)](
        data,
        indices,
        updates,
        output,
        axis_size,
        index_axis_size,
        reduction_code,
    )


@lru_cache(maxsize=COMPILE_CACHE_SIZE)
def _compiled_short_row_reduce(
    data_numel,
    indices_numel,
    value_dtype,
    index_dtype,
    core_num,
    thread_num,
    axis_size,
    index_axis_size,
    reduction_code,
):
    """Compile and retain an AOT callable for this shape, dtype and launch plan."""
    descriptors = (
        cb.TensorSpec((data_numel,), value_dtype),
        cb.TensorSpec((indices_numel,), index_dtype),
        cb.TensorSpec((indices_numel,), value_dtype),
        cb.TensorSpec((data_numel,), value_dtype),
    )
    return cb.compile(
        _launch_short_row_reduce,
        *descriptors,
        core_num,
        thread_num,
        axis_size,
        index_axis_size,
        reduction_code,
    )


@lru_cache(maxsize=PARAM_CACHE_SIZE)
def _compiled_scatter(
    path,
    data_numel,
    indices_numel,
    update_numel,
    value_dtype,
    index_dtype,
    copy_cores,
    scatter_cores,
    axis_stride=0,
    axis_size=0,
    axis=0,
    index_axis_size=0,
):
    """Keep a reusable provider program for each shape/dtype/launch plan."""

    launch = {
        "generic": _launch_scatter,
        "generic_none": _launch_scatter_none,
        "wide_reduce": _launch_scatter_int64_reduce,
        "direct_none": _launch_scatter_direct_none,
        "direct_add": _launch_scatter_direct_add,
        "direct_flat_none": _launch_scatter_direct_flat_none,
        "direct_flat_add": _launch_scatter_direct_flat_add,
        "unique_add": _launch_scatter_unique_add,
        "compact_unique_add": _launch_scatter_compact_unique_add,
    }[path]
    descriptors = (
        cb.TensorSpec((data_numel,), value_dtype),
        cb.TensorSpec((indices_numel,), index_dtype),
        cb.TensorSpec((update_numel,), value_dtype),
        cb.TensorSpec((data_numel,), value_dtype),
        cb.TensorSpec((DEVICE_PARAM_COUNT,), dtypes.int64),
    )
    if path in ("generic", "generic_none", "wide_reduce"):
        return cb.compile(launch, *descriptors, copy_cores)
    if path == "compact_unique_add":
        return cb.compile(
            launch,
            *descriptors,
            copy_cores,
            scatter_cores,
            axis_stride,
            axis_size,
        )
    if path == "direct_flat_none":
        return cb.compile(
            launch,
            *descriptors,
            copy_cores,
            scatter_cores,
            axis,
            axis_size,
            index_axis_size,
            axis_stride,
        )
    return cb.compile(launch, *descriptors, copy_cores, scatter_cores)


@lru_cache(maxsize=COMPILE_CACHE_SIZE)
def _compiled_dma_copy(data_numel, value_dtype, element_bytes, core_num):
    """Compile and retain an AOT callable for this shape, dtype and launch plan."""
    descriptor = cb.TensorSpec((data_numel,), value_dtype)
    launcher = ScatterDmaCopyLauncher(
        data_numel, value_dtype, element_bytes, core_num, DMA_COPY_TILE_BYTES
    )
    return cb.compile(launcher.run, descriptor, descriptor)


@lru_cache(maxsize=COMPILE_CACHE_SIZE)
def _compiled_direct_only(
    path,
    indices_numel,
    update_numel,
    data_numel,
    value_dtype,
    index_dtype,
    scatter_cores,
    axis_stride=0,
    axis_size=0,
    axis=0,
    index_axis_size=0,
):
    """Compile and retain an AOT callable for this shape, dtype and launch plan."""
    launch = {
        "direct_none": _launch_scatter_direct_none_only,
        "direct_flat_none": _launch_scatter_direct_flat_none_only,
        "direct_add": _launch_scatter_direct_add_only,
        "unique_add": _launch_scatter_unique_add_only,
        "compact_unique_add": _launch_scatter_compact_unique_add_only,
    }[path]
    descriptors = (
        cb.TensorSpec((indices_numel,), index_dtype),
        cb.TensorSpec((update_numel,), value_dtype),
        cb.TensorSpec((data_numel,), value_dtype),
        cb.TensorSpec((DEVICE_PARAM_COUNT,), dtypes.int64),
    )
    if path == "compact_unique_add":
        return cb.compile(
            launch,
            *descriptors,
            scatter_cores,
            axis_stride,
            axis_size,
        )
    if path == "direct_flat_none":
        return cb.compile(
            launch,
            *descriptors,
            scatter_cores,
            axis,
            axis_size,
            index_axis_size,
            axis_stride,
        )
    return cb.compile(
        launch,
        *descriptors,
        scatter_cores,
    )


@lru_cache(maxsize=COMPILE_CACHE_SIZE)
def _compiled_copy(data_numel, value_dtype, core_num):
    """Compile and retain an AOT callable for this shape, dtype and launch plan."""
    descriptor = cb.TensorSpec((data_numel,), value_dtype)
    return cb.compile(_launch_copy_only, descriptor, descriptor, core_num)


@lru_cache(maxsize=PARAM_CACHE_SIZE)
def _cached_params(device, stride_parameters, bool_reduction):
    """Reuse immutable device metadata after its initialization has completed."""
    params = torch.tensor(
        (*stride_parameters, bool_reduction), dtype=torch.int64, device=device
    )
    # The cache may be reused by a different stream.  Complete the one-time
    # host-to-device initialization before publishing the immutable tensor.
    torch.npu.synchronize(torch.device(device))
    return params


@kernel
class ScatterNativeKernel:
    """Ascend C style DMA initialization followed by a specialized SIMT VF.

    All copying is raw storage; only float32 add uses floating arithmetic.
    The launch-wide barrier orders every copied output before scatter writes.
    Geometry is compile-time metadata, never inferred from indices values.
    """

    def __init__(
        self,
        tiling,
        value_dtype,
        index_dtype,
        copy_dtype,
        copy_numel,
        copy_element_bytes,
        core_num,
        tile_bytes=NATIVE_TILE_BYTES,
    ):
        """Store compile-time geometry and the launch or buffer configuration."""
        # rank: number of logical axes; axis_size: legal destination-axis range.
        self.rank = len(tiling.data_shape)
        self.axis = tiling.axis
        self.axis_size = tiling.axis_size
        self.data_strides = tiling.data_strides
        self.index_strides = tiling.index_strides
        self.update_strides = tiling.update_strides
        self.flat = tiling.has_flat_axis_coordinates
        # index_axis_size: candidate updates along axis, independent of axis_size.
        self.index_axis_size = tiling.indices_shape[tiling.axis]
        # axis_inner_elements: product of trailing dimensions, in logical elements.
        self.axis_inner_elements = tiling.data_strides[tiling.axis]
        self.indices_numel = tiling.indices_numel
        # copy_numel uses the storage view; int64 payloads occupy two int32 words.
        self.copy_numel = copy_numel
        self.copy_dtype = copy_dtype
        self.reduction_code = tiling.reduction_code
        self.core_num = core_num
        self.tile_elements = tile_bytes // copy_element_bytes
        self.full_tiles, self.tail = divmod(copy_numel, self.tile_elements)

    def __call__(
        self,
        copy_data: Tensor,
        copy_output: Tensor,
        indices: Tensor,
        updates: Tensor,
        output: Tensor,
    ):
        """Execute this kernel using the precomputed geometry and memory views."""
        block = get_block_idx()
        queue = RingBuffer(
            cb.MemLoc.UB,
            self.copy_dtype,
            cb.make_layout((self.tile_elements,)),
            depth=1,
        )
        for copy_tile_index in range(block, self.full_tiles, self.core_num):
            slot = queue.next_write()
            mem_copy(
                slot, tile_slice(copy_data, (self.tile_elements,), (copy_tile_index,))
            )
            ready = queue.next_read()
            mem_copy(
                tile_slice(copy_output, (self.tile_elements,), (copy_tile_index,)),
                ready,
            )
        if self.tail:
            if block == 0:
                slot = queue.next_write()
                mem_copy(
                    slot[: self.tail,],
                    copy_data[self.full_tiles * self.tile_elements :,],
                )
                ready = queue.next_read()
                mem_copy(
                    copy_output[self.full_tiles * self.tile_elements :,],
                    ready[: self.tail,],
                )
        # This kernel is pure AIV. Drain only DMA output writes, then let all
        # Vector Cores rendezvous on MTE3 -> S before any SIMT GM read/write.
        # A local pipeline event alone cannot protect random cross-core targets.
        # core_num is bounded by resident Vector Cores by the public entry.
        cb.vec_sync_pipe(cb.PIPE.MTE3)
        cb.vec_sync_block_arrive(cb.PIPE.MTE3, COPY_READY_FLAG_ID, mode=0)
        cb.vec_sync_block_wait(cb.PIPE.S, COPY_READY_FLAG_ID, mode=0)
        with cb.vf(mode="simt", thread=NATIVE_THREAD_COUNT):
            thread_index = cb.simt.thread_idx()[0]
            simt_block = cb.simt.block_idx()[0]
            for index_flat_offset in range(
                simt_block * NATIVE_THREAD_COUNT + thread_index,
                self.indices_numel,
                self.core_num * NATIVE_THREAD_COUNT,
            ):
                selected_axis_index = indices[index_flat_offset]
                if selected_axis_index >= 0 and selected_axis_index < self.axis_size:
                    selected_axis_index32 = selected_axis_index
                    # Identity casts add no information; int64 indices still narrow after validation.
                    if cb.const_expr(indices.dtype != dtypes.int32):
                        selected_axis_index32 = dtypes.int32(selected_axis_index)
                    # Flat geometry shares source offsets; only the axis length changes.
                    if cb.const_expr(self.flat):
                        # Leading axis: preserve just the trailing coordinate.
                        if cb.const_expr(self.axis == 0):
                            output_flat_offset = (
                                selected_axis_index32 * self.axis_inner_elements
                                + index_flat_offset % self.axis_inner_elements
                            )
                        # Unit inner stride: retain the outer row and replace its column.
                        elif cb.const_expr(self.axis_inner_elements == 1):
                            output_flat_offset = (
                                index_flat_offset // self.index_axis_size
                            ) * self.axis_size + selected_axis_index32
                        else:
                            # Middle axis: split the offset into outer and inner coordinates.
                            outer_group_index = index_flat_offset // (
                                self.index_axis_size * self.axis_inner_elements
                            )
                            output_flat_offset = (
                                outer_group_index
                                * (self.axis_size * self.axis_inner_elements)
                                + selected_axis_index32 * self.axis_inner_elements
                                + index_flat_offset % self.axis_inner_elements
                            )
                        update_flat_offset = index_flat_offset
                    else:
                        # General shapes: independently accumulate data and update strides.
                        remaining_flat_offset = index_flat_offset
                        output_flat_offset = 0
                        update_flat_offset = 0
                        for dim in cb.range_constexpr(self.rank):
                            dimension_coord = (
                                remaining_flat_offset // self.index_strides[dim]
                            )
                            remaining_flat_offset -= (
                                dimension_coord * self.index_strides[dim]
                            )
                            update_flat_offset += (
                                dimension_coord * self.update_strides[dim]
                            )
                            if cb.const_expr(dim != self.axis):
                                output_flat_offset += (
                                    dimension_coord * self.data_strides[dim]
                                )
                        output_flat_offset += (
                            selected_axis_index32 * self.axis_inner_elements
                        )
                    if cb.const_expr(self.reduction_code == REDUCTION_NONE):
                        output[output_flat_offset] = updates[update_flat_offset]
                    else:
                        cb.simt.atomic_add(
                            output.ptr(output_flat_offset), updates[update_flat_offset]
                        )


class ScatterNativeLauncher:
    """Bind compile-time metadata to the corresponding device kernel."""

    def __init__(
        self,
        tiling,
        value_dtype,
        index_dtype,
        copy_dtype,
        copy_numel,
        copy_element_bytes,
        core_num,
    ):
        """Store compile-time geometry and the launch or buffer configuration."""
        self.operator = ScatterNativeKernel(
            tiling,
            value_dtype,
            index_dtype,
            copy_dtype,
            copy_numel,
            copy_element_bytes,
            core_num,
        )
        self.core_num = core_num

    @host
    def run(
        self,
        copy_data: Tensor,
        copy_output: Tensor,
        indices: Tensor,
        updates: Tensor,
        output: Tensor,
    ) -> None:
        """Launch the selected device implementation with its compile-time geometry."""
        self.operator[self.core_num](copy_data, copy_output, indices, updates, output)


@lru_cache(maxsize=COMPILE_CACHE_SIZE)
def _compiled_native(
    data_shape,
    index_shape,
    update_shape,
    axis,
    reduction,
    value_dtype,
    index_dtype,
    copy_dtype,
    copy_numel,
    copy_element_bytes,
    core_num,
):
    """Compile and retain an AOT callable for this shape, dtype and launch plan."""
    tiling = ScatterElementsTiling(
        data_shape, index_shape, update_shape, axis, reduction
    )
    launcher = ScatterNativeLauncher(
        tiling,
        value_dtype,
        index_dtype,
        copy_dtype,
        copy_numel,
        copy_element_bytes,
        core_num,
    )
    copy = cb.TensorSpec((copy_numel,), copy_dtype)
    return cb.compile(
        launcher.run,
        copy,
        copy,
        cb.TensorSpec((tiling.indices_numel,), index_dtype),
        cb.TensorSpec((prod(update_shape),), value_dtype),
        cb.TensorSpec((tiling.data_numel,), value_dtype),
    )


def _validate_scatter_inputs(state):
    """Validate the public tensor, dtype and forward-only contract on the host."""
    if not all(
        (
            isinstance(tensor, torch.Tensor)
            for tensor in (state.data, state.indices, state.updates)
        )
    ):
        raise TypeError("The data, indices, and updates must be torch.Tensor objects.")
    if state.data.device.type not in ("npu", "privateuseone"):
        raise ValueError("The data must be on an NPU.")
    if (
        state.indices.device != state.data.device
        or state.updates.device != state.data.device
    ):
        raise ValueError("The data, indices, and updates must be on the same NPU.")
    if (
        not state.data.is_contiguous()
        or not state.indices.is_contiguous()
        or (not state.updates.is_contiguous())
    ):
        raise ValueError("The data, indices, and updates must be contiguous.")
    if state.data.dtype != state.updates.dtype:
        raise TypeError("The data and updates must have the same dtype.")
    if state.indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("The indices must have int32 or int64 dtype.")
    if state.data.dtype not in SUPPORTED_DATA_DTYPES:
        raise TypeError(f"The data dtype {state.data.dtype} is not supported.")
    if state.reduction != "none" and state.data.dtype in (
        torch.float64,
        torch.complex64,
    ):
        raise TypeError("The add and mul do not support float64 or complex64.")
    if state.data.requires_grad or state.updates.requires_grad:
        raise ValueError(
            "The scatter_elements is forward-only; detach inputs before calling."
        )
    if not isinstance(state.deterministic, bool):
        raise TypeError("The deterministic parameter must be bool.")


def _prepare_scatter_views(state):
    """Preserve replacement bits and choose the existing reduction storage views."""
    if state.reduction == "none":
        state.carrier = RAW_STORAGE_DTYPES[state.data.element_size()]
        state.data_view = state.data.view(state.carrier).view(-1)
        state.update_view = state.updates.view(state.carrier).view(-1)
        state.output_view = state.output.view(state.carrier).view(-1)
    elif state.data.dtype == torch.bool:
        state.data_view = state.data.view(torch.uint8).view(-1)
        state.update_view = state.updates.view(torch.uint8).view(-1)
        state.output_view = state.output.view(torch.uint8).view(-1)
    elif state.data.dtype == torch.int64:
        state.data_view = state.data.view(torch.int32).view(-1)
        state.update_view = state.updates.view(torch.int32).view(-1)
        state.output_view = state.output.view(torch.int32).view(-1)
    else:
        state.data_view = state.data.view(-1)
        state.update_view = state.updates.view(-1)
        state.output_view = state.output.view(-1)
    state.index_view = state.indices.view(-1)
    state.value_dtype = TORCH_TO_DSL_DTYPE[state.data_view.dtype]


def _execute_scatter_native(state):
    """Execute the existing native branch without changing its device launch arguments."""
    if state.data.element_size() == DOUBLE_WORD_ELEMENT_BYTES:
        copy_data = state.data.view(torch.int32).view(-1)
        copy_output = state.output.view(torch.int32).view(-1)
        copy_dtype = dtypes.int32
    else:
        copy_data = state.data.view(
            state.carrier if state.reduction == "none" else torch.int32
        ).view(-1)
        copy_output = state.output.view(copy_data.dtype).view(-1)
        copy_dtype = TORCH_TO_DSL_DTYPE[copy_data.dtype]
    native_cores = min(
        state.active_cores,
        max(
            1,
            (state.data.numel() * state.data.element_size() + NATIVE_TILE_BYTES - 1)
            // NATIVE_TILE_BYTES,
            (state.tiling.indices_numel + NATIVE_THREAD_COUNT - 1)
            // NATIVE_THREAD_COUNT,
        ),
    )
    _compiled_native(
        state.tiling.data_shape,
        state.tiling.indices_shape,
        state.tiling.updates_shape,
        state.tiling.axis,
        state.reduction,
        state.value_dtype,
        TORCH_TO_DSL_DTYPE[state.index_view.dtype],
        copy_dtype,
        copy_data.numel(),
        copy_data.element_size(),
        native_cores,
    )(copy_data, copy_output, state.index_view, state.update_view, state.output_view)
    return state.output


def _execute_scatter_short_rows(state):
    """Execute the existing short_rows branch without changing its device launch arguments."""
    row_count = state.tiling.data_numel // state.tiling.axis_size
    row_cores = _resolve_core_num(state.data, state.core_num, row_count * THREAD_COUNT)
    thread_num = (
        SHORT_ROW_SMALL_THREADS
        if state.tiling.axis_size <= SHORT_ROW_SMALL_THREADS
        else SHORT_ROW_LARGE_THREADS
    )
    _compiled_short_row_reduce(
        state.data_view.numel(),
        state.tiling.indices_numel,
        state.value_dtype,
        TORCH_TO_DSL_DTYPE[state.index_view.dtype],
        row_cores,
        thread_num,
        state.tiling.axis_size,
        state.tiling.indices_shape[state.tiling.axis],
        REDUCTION_ADD if state.reduction == "add" else REDUCTION_MUL,
    )(state.data_view, state.index_view, state.update_view, state.output_view)
    return state.output


def _execute_scatter_axis0_owner(state):
    """Execute the existing axis0_owner branch without changing its device launch arguments."""
    _compiled_axis0_owner_add(
        state.data_view.numel(),
        state.tiling.indices_numel,
        state.value_dtype,
        TORCH_TO_DSL_DTYPE[state.index_view.dtype],
        state.active_cores,
        state.tiling.data_strides[0],
        state.tiling.indices_shape[0],
    )(state.data_view, state.index_view, state.update_view, state.output_view)
    return state.output


def _select_float32_add_fallback(state):
    """Distinguish sparse single writers from flat or general atomic updates."""
    if state.tiling.indices_shape[state.tiling.axis] != 1:
        return (
            "direct_flat_add"
            if state.tiling.has_flat_axis_coordinates
            else "direct_add"
        )
    compact_shape = all(size == 1 for size in state.tiling.indices_shape[:-1])
    if state.tiling.axis < len(state.tiling.data_shape) - 1 and compact_shape:
        return "compact_unique_add"
    return "unique_add"


def _prepare_scatter_fallback(state):
    """Prepare metadata and select the same fallback branches in the original order."""
    state.params = _cached_params(
        str(state.data.device),
        state.tiling.stride_parameters,
        int(state.data.dtype == torch.bool and state.reduction != "none"),
    )
    state.index_dtype = TORCH_TO_DSL_DTYPE[state.index_view.dtype]
    state.scatter_cores = min(
        state.active_cores,
        max(1, (state.tiling.indices_numel + THREAD_COUNT - 1) // THREAD_COUNT),
    )
    state.path = "generic_none" if state.reduction == "none" else "generic"
    if state.reduction != "none" and state.data.dtype == torch.int64:
        state.path = "wide_reduce"
        return
    if (
        state.tiling.data_numel <= DIRECT_MIN_DATA_ELEMENTS
        or not state.tiling.indices_numel
    ):
        return
    if state.reduction == "none" and not state.deterministic:
        state.path = (
            "direct_flat_none"
            if state.tiling.has_flat_axis_coordinates
            else "direct_none"
        )
    elif state.reduction == "add" and state.data.dtype == torch.float32:
        state.path = _select_float32_add_fallback(state)


def _execute_scatter_dma_fallback(state):
    """Initialize output using the original DMA copy and then execute the direct update kernel."""
    if state.path in ("direct_add", "unique_add", "compact_unique_add"):
        copy_data = state.data.view(torch.int32).view(-1)
        copy_output = state.output.view(torch.int32).view(-1)
        copy_dtype = dtypes.int32
    elif state.data.element_size() == DOUBLE_WORD_ELEMENT_BYTES:
        copy_data = state.data.view(torch.int32).view(-1)
        copy_output = state.output.view(torch.int32).view(-1)
        copy_dtype = dtypes.int32
    else:
        copy_data = state.data_view
        copy_output = state.output_view
        copy_dtype = state.value_dtype
    _compiled_dma_copy(
        copy_data.numel(), copy_dtype, copy_data.element_size(), state.active_cores
    )(copy_data, copy_output)
    _compiled_direct_only(
        state.path,
        state.tiling.indices_numel,
        state.update_view.numel(),
        state.data_view.numel(),
        state.value_dtype,
        state.index_dtype,
        state.scatter_cores,
        state.tiling.data_strides[state.tiling.axis],
        state.tiling.axis_size,
        state.tiling.axis,
        state.tiling.indices_shape[state.tiling.axis],
    )(state.index_view, state.update_view, state.output_view, state.params)
    return state.output


def _execute_scatter_cached_fallback(state):
    """Execute the existing cached fallback provider with unchanged tensor descriptors."""
    program = _compiled_scatter(
        state.path,
        state.data_view.numel(),
        state.tiling.indices_numel,
        state.update_view.numel(),
        state.value_dtype,
        state.index_dtype,
        state.active_cores,
        state.scatter_cores,
        state.tiling.data_strides[state.tiling.axis],
        state.tiling.axis_size,
        state.tiling.axis,
        state.tiling.indices_shape[state.tiling.axis],
    )
    program(
        state.data_view,
        state.index_view,
        state.update_view,
        state.output_view,
        state.params,
    )
    return state.output


def _execute_scatter_fallback(state):
    """Select DMA or cached fallback after preparing the same geometry and metadata."""
    _prepare_scatter_fallback(state)
    if (
        state.path
        in (
            "direct_none",
            "direct_flat_none",
            "direct_add",
            "unique_add",
            "compact_unique_add",
        )
        and state.data.numel() * state.data.element_size() >= DMA_COPY_MIN_BYTES
    ):
        return _execute_scatter_dma_fallback(state)
    return _execute_scatter_cached_fallback(state)


def _short_rows_has_flat_unit_data_stride(state):
    """Check one group of host dispatch constraints, preserving short-circuit order."""
    return (
        state.index_view.dtype == torch.int32
        and state.tiling.has_flat_axis_coordinates
        and (state.tiling.data_strides[state.tiling.axis] == 1)
    )


def _short_rows_has_supported_axis_extent(state):
    """Check one group of host dispatch constraints, preserving short-circuit order."""
    return (
        state.tiling.index_strides[state.tiling.axis] == 1
        and SHORT_ROW_MIN_ELEMENTS <= state.tiling.axis_size <= SHORT_ROW_MAX_ELEMENTS
        and (state.tiling.indices_shape[state.tiling.axis] == state.tiling.axis_size)
    )


def _axis0_owner_has_float32_add(state):
    """Check one group of host dispatch constraints, preserving short-circuit order."""
    return (
        state.reduction == "add"
        and state.data.dtype == torch.float32
        and (state.tiling.axis == 0)
    )


def _axis0_owner_has_full_axis_geometry(state):
    """Check one group of host dispatch constraints, preserving short-circuit order."""
    return (
        state.tiling.has_flat_axis_coordinates
        and state.tiling.indices_shape[0] == state.tiling.axis_size
        and (state.tiling.axis_size == AXIS0_OWNER_AXIS_ELEMENTS)
    )


def _axis0_owner_has_large_workload(state):
    """Check one group of host dispatch constraints, preserving short-circuit order."""
    return state.tiling.data_numel > DIRECT_MIN_DATA_ELEMENTS


def _uses_native_scatter(state):
    """Use native DMA/SIMT only for valid 32-bit extents and unordered writes/adds."""
    if (
        max(
            state.tiling.data_numel,
            state.tiling.indices_numel,
            prod(state.tiling.updates_shape),
        )
        > INT32_MAX
    ):
        return False
    if state.deterministic:
        return False
    if state.reduction == "none":
        return True
    return state.reduction == "add" and state.data.dtype == torch.float32


def _uses_short_row_scatter(state):
    """Use ordered short-row SIMT only for its existing dtype and geometry pairings."""
    float32_mul = state.reduction == "mul" and state.data.dtype == torch.float32
    int8_add = state.reduction == "add" and state.data.dtype == torch.int8
    if not (float32_mul or int8_add):
        return False
    return _short_rows_has_flat_unit_data_stride(
        state
    ) and _short_rows_has_supported_axis_extent(state)


def _prepare_scatter_elements_output(state):
    """Allocate output, preserve empty-input behavior and prepare the existing geometry."""
    state.tiling = ScatterElementsTiling(
        state.data.shape,
        state.indices.shape,
        state.updates.shape,
        state.axis,
        state.reduction,
    )
    state.output = torch.empty_like(state.data)
    if state.tiling.data_numel == 0:
        return False
    state.active_cores = _resolve_core_num(
        state.data, state.core_num, state.tiling.data_numel
    )
    return True


def scatter_elements(
    data,
    indices,
    updates,
    axis,
    *,
    reduction="none",
    core_num=None,
    deterministic=False,
):
    """Scatter updates into a copy of data along one axis.

    Indices must satisfy ``0 <= indices < data.shape[axis]``.  Replacement
    preserves flattened index order when ``deterministic=True``; the default
    large-tensor path permits nondeterministic ordering of duplicate writes.
    This sample is forward-only.
    """
    state = SimpleNamespace(
        data=data,
        indices=indices,
        updates=updates,
        axis=axis,
        reduction=reduction,
        core_num=core_num,
        deterministic=deterministic,
    )
    _validate_scatter_inputs(state)
    if not _prepare_scatter_elements_output(state):
        return state.output
    _prepare_scatter_views(state)
    if state.tiling.indices_numel == 0:
        _compiled_copy(state.data_view.numel(), state.value_dtype, state.active_cores)(
            state.data_view, state.output_view
        )
        return state.output
    if _uses_native_scatter(state):
        return _execute_scatter_native(state)
    if _uses_short_row_scatter(state):
        return _execute_scatter_short_rows(state)
    if (
        _axis0_owner_has_float32_add(state)
        and _axis0_owner_has_full_axis_geometry(state)
        and _axis0_owner_has_large_workload(state)
    ):
        return _execute_scatter_axis0_owner(state)
    return _execute_scatter_fallback(state)
