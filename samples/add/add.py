# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Ascend950 SIMT elementwise Add for equal-shaped tensors."""

__all__ = ["add"]

from dataclasses import dataclass
from functools import lru_cache
from operator import index
import threading

import torch
import torch_npu  # noqa: F401
from cannbotdsl import (
    Channel, MemLoc, Tensor, TensorSpec, compile as dsl_compile,
    dtypes, get_platform_info, jit, simt, vf,
)
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.info import get_mem_size
from cannbotdsl.ops.memcpy import mem_copy

# Public input types; map them to DSL types only when building compilation specs.
_SUPPORTED = (torch.float32, torch.int32)
_DSL_DTYPES = {torch.float32: dtypes.float32, torch.int32: dtypes.int32}
# Default threads per SIMT VF; supported overrides are multiples of the 32-thread warp size.
_DEFAULT_THREADS = 1024
_THREAD_NUMS = (32, 64, 128, 256, 512, _DEFAULT_THREADS)
# Cache up to 128 compiled variants keyed by (AddTiling, element_dtype).
# Reuse avoids recompilation; LRU eviction bounds retained cache entries.
_COMPILE_CACHE_CAPACITY = 128

# DMA core intervals and buffer capacities are aligned to a 32-byte block.
_DMA_ALIGNMENT_BYTES = 32
# Two slots per channel allow input prefetch and output writeback to overlap computation.
_DMA_BUFFER_DEPTH = 2
_DMA_ROLE_COUNT = 3  # input1, input2, output
# Total UB slots: two for each input and two for the output.
_DMA_SLOT_COUNT = _DMA_ROLE_COUNT * _DMA_BUFFER_DEPTH
# Ascend950 SIMT UB budget: 256 KiB total minus at least 32 KiB for Data Cache.
_SIMT_MAX_SHARED_BYTES = 229376  # 224 KiB, including the system reservation below.
_SIMT_SHARED_OVERHEAD_BYTES = 8192  # 8 KiB reserved for the compiler and Ascend C runtime.

# Serialize cache lookup and compilation so concurrent misses do not compile the same variant twice.
_COMPILE_LOCK = threading.Lock()


def _require_integer(value, parameter_name):
    if isinstance(value, bool):
        raise TypeError(f"{parameter_name} must be an integer, not bool")
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(f"{parameter_name} must be an integer") from exc


def _validate_inputs(input1, input2):
    """Reject unsupported tensor contracts before preparing compute views."""
    if not isinstance(input1, torch.Tensor) or not isinstance(input2, torch.Tensor):
        raise TypeError("x1 and x2 must be torch.Tensor objects")
    if tuple(input1.shape) != tuple(input2.shape):
        raise ValueError("Add requires inputs with the same shape")
    for input_tensor in (input1, input2):
        if input_tensor.layout != torch.strided or not input_tensor.is_contiguous():
            raise ValueError("Add requires contiguous strided tensors")
        if input_tensor.is_neg():
            raise ValueError("Resolve negative view flags before calling Add")
        if input_tensor.numel() == 0:
            raise ValueError("Add requires nonempty tensors")
        if input_tensor.requires_grad:
            raise ValueError("Add is forward-only; detach inputs before calling")
    if input1.device.type not in ("npu", "privateuseone") or input1.device != input2.device:
        raise ValueError("x1 and x2 must be on the same NPU")
    if input1.dtype != input2.dtype or input1.dtype not in _SUPPORTED:
        raise TypeError("Add requires matching float32 or int32 inputs")


def _validate_launch_options(core_num, thread_num, available_cores):
    """Validate user overrides; leave the automatic thread choice to tiling."""
    if thread_num is not None:
        thread_num = _require_integer(thread_num, "thread_num")
        if thread_num not in _THREAD_NUMS:
            raise ValueError(f"thread_num must be one of {_THREAD_NUMS}")
    if core_num is None:
        core_num = available_cores
    else:
        core_num = _require_integer(core_num, "core_num")
        if not 1 <= core_num <= available_cores:
            raise ValueError(f"core_num must be in [1, {available_cores}]")
    return core_num, thread_num


@dataclass(frozen=True)
class AddTiling:
    """Use DMA for every nonempty input, including scalars and offset views.

    Launch min(core_limit, ceil(N/T)) cores; T defaults to 1024. Distribute
    ceil(N/A) units across cores, where A=32/element_bytes. The first remainder
    cores receive one extra unit, and the final core is clipped to N.
    Tile elements are the minimum of the UB capacity aligned down to A and
    ceil(N/core_count) aligned up to A. UB holds three depth-two channels.
    Every tile carries its valid length through load/compute/store, including
    tensors shorter than 32 bytes. Global intervals use int64; UB indices int32.
    """
    element_count: int  # Total scalar elements in each flattened GM tensor.
    core_count: int  # Number of vector blocks launched for this input.
    thread_count: int  # SIMT threads per block, reused for every tile.
    tile_elements: int  # Capacity of one UB slot; the final tile may contain fewer valid elements.
    dma_alignment_elements: int  # Elements per 32-byte alignment unit: eight for both supported types.

    @property
    def core_alignment_units(self):
        """Return (units per core, number of cores receiving one extra unit)."""
        # Round up so the last, possibly partial alignment unit is assigned to a core.
        total_units = (self.element_count + self.dma_alignment_elements - 1) // self.dma_alignment_elements
        # The quotient is the base allocation; the first remainder cores receive one extra unit.
        return divmod(total_units, self.core_count)

    @classmethod
    def build(cls, output_view, *, core_num=None, thread_num=None):
        stream = torch.npu.current_stream(output_view.device)
        available_cores = int(get_platform_info(stream=stream).vector_core_num)
        core_limit, requested_threads = _validate_launch_options(core_num, thread_num, available_cores)
        thread_count = _DEFAULT_THREADS if requested_threads is None else requested_threads
        element_count = output_view.numel()
        # Cap the block count by available cores and ceil(N / threads); each block may process several tiles.
        core_count = min(core_limit, (element_count + thread_count - 1) // thread_count)
        # Both supported dtypes occupy four bytes per element.
        element_bytes = output_view.element_size()
        # Alignment granularity, not a limit on the number of elements copied per DMA operation.
        alignment_elements = _DMA_ALIGNMENT_BYTES // element_bytes
        # Cap all six explicit UB slots at 216 KiB, preserving Data Cache and the 8 KiB system reservation.
        usable_ub_bytes = min(get_mem_size("ub"), _SIMT_MAX_SHARED_BYTES - _SIMT_SHARED_OVERHEAD_BYTES)
        # Per-slot capacity: divide the shared budget among six slots, then round down to 32 bytes.
        tile_capacity_elements = (usable_ub_bytes // (_DMA_SLOT_COUNT * element_bytes)
                                  // alignment_elements * alignment_elements)
        if tile_capacity_elements < alignment_elements:
            raise RuntimeError("Insufficient SIMT UB capacity for one aligned DMA block")
        # Ceiling of the average core workload, before distributing aligned units.
        elements_per_core = (element_count + core_count - 1) // core_count
        # Round up to cover the largest core allocation; avoid oversized slots for small inputs.
        aligned_elements_per_core = ((elements_per_core + alignment_elements - 1)
                                     // alignment_elements * alignment_elements)
        return cls(element_count, core_count, thread_count,
                   tile_elements=min(tile_capacity_elements, aligned_elements_per_core),
                   dma_alignment_elements=alignment_elements)


class _AddDmaStages:
    """Own depth-two UB channels connecting DMA input, SIMT compute and DMA output stages."""
    def __init__(self, element_dtype, tiling):
        self.tile_elements = tiling.tile_elements
        self.thread_count = tiling.thread_count
        self.input1_channel = Channel(MemLoc.UB, (1, self.tile_elements), element_dtype, depth=_DMA_BUFFER_DEPTH)
        self.input2_channel = Channel(MemLoc.UB, (1, self.tile_elements), element_dtype, depth=_DMA_BUFFER_DEPTH)
        self.output_channel = Channel(MemLoc.UB, (1, self.tile_elements), element_dtype, depth=_DMA_BUFFER_DEPTH)

    @jit
    def load(self, input1_gm: Tensor, input2_gm: Tensor,
             core_element_offset: int, core_element_count: int, tile_index: int):
        element_offset, valid_element_count = self._tile_range(core_element_offset, core_element_count, tile_index)
        # Select writable input slots; channel dependencies protect reuse across tiles.
        input1_ub = self.input1_channel.produce()
        input2_ub = self.input2_channel.produce()
        # Match the two-dimensional UB layout without copying or reshaping GM storage.
        mem_copy(input1_ub[0:1, 0:valid_element_count],
                 input1_gm.view(1, input1_gm.shape[0])[0:1, element_offset:element_offset + valid_element_count])
        mem_copy(input2_ub[0:1, 0:valid_element_count],
                 input2_gm.view(1, input2_gm.shape[0])[0:1, element_offset:element_offset + valid_element_count])

    @jit
    def compute(self, core_element_count: int, tile_index: int):
        valid_element_count = self._tile_length(core_element_count, tile_index)
        input1_ub = self.input1_channel.consume()
        input2_ub = self.input2_channel.consume()
        output_ub = self.output_channel.produce()
        with vf(mode="simt", thread=self.thread_count):
            thread_index = simt.thread_idx()[0]
            # UB-local counts fit int32; global GM offsets remain int64 outside the VF.
            simt_element_count = simt.cast(valid_element_count, dtype=dtypes.int32)
            # Threads cover disjoint indices with a stride equal to the thread count.
            for tile_element_index in range(thread_index, simt_element_count, self.thread_count):
                output_ub[0, tile_element_index] = input1_ub[0, tile_element_index] + input2_ub[0, tile_element_index]

    @jit
    def store(self, output_gm: Tensor, core_element_offset: int, core_element_count: int, tile_index: int):
        element_offset, valid_element_count = self._tile_range(core_element_offset, core_element_count, tile_index)
        output_ub = self.output_channel.consume()
        mem_copy(output_gm.view(1, output_gm.shape[0])[0:1, element_offset:element_offset + valid_element_count],
                 output_ub[0:1, 0:valid_element_count])

    @jit
    def _tile_length(self, core_element_count, tile_index):
        return min(self.tile_elements, core_element_count - tile_index * self.tile_elements)

    @jit
    def _tile_range(self, core_element_offset, core_element_count, tile_index):
        # GM offsets are global element indices; every tile starts at offset zero in its UB slot.
        element_offset = core_element_offset + tile_index * self.tile_elements
        return element_offset, self._tile_length(core_element_count, tile_index)


@kernel
class AddDmaKernel:
    """MTE2 -> SIMT on UB -> MTE3, using six independently tracked UB slots."""
    def __init__(self, tiling):
        self.tiling = tiling
        # Base units per core and the number of leading cores assigned one additional unit.
        self.alignment_units_per_core, self.extra_alignment_units = tiling.core_alignment_units

    def __call__(self, input1_gm: Tensor, input2_gm: Tensor, output_gm: Tensor):
        core_element_offset, core_element_count, tile_count = self._core_range()
        dma_stages = _AddDmaStages(output_gm.dtype, self.tiling)
        # Three stages: load(k), compute(k - 1), store(k - 2).
        # Compute trails load by one step; store trails load by two steps.
        # Two extra steps drain the pipeline; index guards also handle a single tile.
        for pipeline_step in range(tile_count + 2):
            load_tile_index = pipeline_step
            compute_tile_index = pipeline_step - 1
            store_tile_index = pipeline_step - 2
            if load_tile_index < tile_count:
                dma_stages.load(input1_gm, input2_gm, core_element_offset, core_element_count, load_tile_index)
            if 0 <= compute_tile_index < tile_count:
                dma_stages.compute(core_element_count, compute_tile_index)
            if 0 <= store_tile_index < tile_count:
                dma_stages.store(output_gm, core_element_offset, core_element_count, store_tile_index)

    @jit
    def _core_range(self):
        """Resolve this core's aligned interval and tile count outside the VF."""
        block_index = get_block_idx()  # Global GM interval arithmetic stays int64.
        # Count all units assigned to preceding blocks, including their extra units.
        core_unit_offset = block_index * self.alignment_units_per_core + min(block_index, self.extra_alignment_units)
        core_element_offset = core_unit_offset * self.tiling.dma_alignment_elements
        alignment_unit_count = self.alignment_units_per_core
        if block_index < self.extra_alignment_units:
            alignment_unit_count = alignment_unit_count + 1
        # Clip the last alignment unit to the actual tensor length to prevent out-of-bounds accesses.
        core_element_count = min(
            alignment_unit_count * self.tiling.dma_alignment_elements, self.tiling.element_count - core_element_offset)
        tile_count = (core_element_count + self.tiling.tile_elements - 1) // self.tiling.tile_elements
        return core_element_offset, core_element_count, tile_count


class _AddLauncher:
    def __init__(self, tiling):
        self.tiling = tiling

    @host
    def run(self, input1_gm: Tensor, input2_gm: Tensor, output_gm: Tensor):
        kernel_instance = AddDmaKernel(self.tiling)
        kernel_instance[self.tiling.core_count](input1_gm, input2_gm, output_gm)


@lru_cache(maxsize=_COMPILE_CACHE_CAPACITY)
def _compile_add(tiling, element_dtype):
    tensor_spec = TensorSpec((tiling.element_count,), _DSL_DTYPES[element_dtype], stride=(1,))
    return dsl_compile(_AddLauncher(tiling).run, tensor_spec, tensor_spec, tensor_spec)


def add(x1, x2, *, core_num=None, thread_num=None):
    """Add matching float32/int32 contiguous NPU tensors with identical shapes."""
    _validate_inputs(x1, x2)
    with torch.npu.device(x1.device):
        output_tensor = torch.empty(tuple(x1.shape), dtype=x1.dtype, device=x1.device)
        input1_view, input2_view, output_view = (tensor.view(-1) for tensor in (x1, x2, output_tensor))
        tiling = AddTiling.build(output_view, core_num=core_num, thread_num=thread_num)
        with _COMPILE_LOCK:
            compiled_add = _compile_add(tiling, output_view.dtype)
        compiled_add(input1_view, input2_view, output_view)
    return output_tensor
