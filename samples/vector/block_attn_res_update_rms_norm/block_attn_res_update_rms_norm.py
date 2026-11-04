# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Block AttnRes Update + RMS Norm, independent CANNBotDSL implementation.

Public API:
    block_attn_res_update_rms_norm(partial_block, delta, pseudo_query,
                                  numerator, logit_max, exp_sum, gamma,
                                  score_eps=1e-6, norm_eps=1e-6) -> y

Updates partial_block in FP32, merges the online-softmax state, then normalizes
the rounded BF16 h in UB. Its RMS square sum is accumulated while h is
generated. A single kernel runs three source-level vector stages and writes only updated
partial and normalized y to GM. The two epsilon values are independent.

This module is self-contained: it does not import or register the unfused
operator. The Update arithmetic intentionally matches that implementation.
"""

__all__ = [
    "BlockAttnResUpdateRMSNorm",
    "UpdateKernel",
    "clear_caches",
    "block_attn_res_update_rms_norm",
]

import math
import threading
from dataclasses import dataclass

import cannbotdsl
import torch
from torch import _check
from torch._dynamo import substitute_in_graph

from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.host import host
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.types.dtypes import (
    bfloat16 as BFloat16,
    float32 as Float32,
    int64 as Int64,
)
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.reg import (
    PackMode,
    RoundingMode,
    UnpackMode,
    full_mask,
    update_mask,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdups,
    vexp_sub,
    vload,
    vload_broadcast,
    vload_unpack,
    vmax,
    vmem_bar,
    vmadd,
    vmul,
    vmuls,
    vreduce_sum,
    vsqrt,
    vstore,
    vstore_first,
    vstore_pack,
)
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.tensor import MemLoc, Tensor, make_layout, tile_slice
from cannbotdsl.lang.vf import vf
from cannbotdsl.aot import export


VL = 64
DEFAULT_BLOCK_NUM = 64
HIDDEN_SIZE = 7168
FP32_BITS = 32
UB_BYTES = 240 * 1024
EXECUTABLE_CACHE_CAPACITY = 128

_TORCH_TO_DSL = {
    torch.bfloat16: BFloat16,
}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalize_eps(eps: float | int) -> float:
    _require(
        isinstance(eps, (float, int))
        and math.isfinite(float(eps))
        and float(eps) > 0.0,
        "eps 必须为有限的正 Python float",
    )
    return float(eps)


def _ub_footprint(d: int, tile_t: int, depth: int) -> int:
    """32-byte-aligned UB storage, including query, gamma and tile buffers."""
    fp32_stride = _ceil_div(d, 8) * 8
    b16_stride = _ceil_div(d, 16) * 16
    stats_stride = _ceil_div(tile_t, 8) * 8
    slot_bytes = tile_t * (2 * fp32_stride * 4 + b16_stride * 2) + 3 * stats_stride * 4
    return fp32_stride * 4 + depth * slot_bytes + b16_stride * 2


def _get_tiling(tokens: int, d: int, vector_cores: int, ub_bytes: int):
    """Return block count, rows per core, tile rows and buffer depth."""
    tokens = max(1, tokens)
    rows_per_core = _ceil_div(tokens, vector_cores)
    block_num = _ceil_div(tokens, rows_per_core)
    tile_t = min(_ceil_div(rows_per_core, 2), 4095)
    fp32_stride = _ceil_div(d, 8) * 8
    b16_stride = _ceil_div(d, 16) * 16
    gamma_bytes = b16_stride * 2
    row_bytes = 2 * fp32_stride * 4 + b16_stride * 2 + 3 * 4
    tile_t = min(tile_t, (ub_bytes - fp32_stride * 4 - gamma_bytes) // (2 * row_bytes))
    if d % 16:
        tile_t = min(tile_t, 1)
    while tile_t > 0 and _ub_footprint(d, tile_t, 2) > ub_bytes:
        tile_t -= 1
    _require(tile_t > 0, "One complete D row does not fit in double-buffered UB")
    tile_count = _ceil_div(rows_per_core, tile_t)
    tile_t = _ceil_div(rows_per_core, tile_count)
    return block_num, rows_per_core, tile_t, 1 if tile_count == 1 else 2


@dataclass(frozen=True)
class _VectorConfig:
    d: int
    fp_stride: int
    b16_stride: int
    stats_stride: int
    delta_dtype: object
    norm_eps: float


@jit
def _snapshot_vector_config(config):
    # Read the owner's compile-time fields before entering a VF.
    return _VectorConfig(
        d=config.d,
        fp_stride=config.fp_stride,
        b16_stride=config.b16_stride,
        stats_stride=config.stats_stride,
        delta_dtype=config.delta_dtype,
        norm_eps=config.norm_eps,
    )


@dataclass(frozen=True)
class _UpdateBuffers:
    partial: object
    delta_h: object
    score: object
    query: object = None
    numerator: object = None
    stats: object = None
    gamma: object = None


@dataclass(frozen=True)
class _VectorMasks:
    full: object
    scalar: object
    tail: object


@dataclass(frozen=True)
class _RowOffsets:
    fp: object
    b16: object


@dataclass(frozen=True)
class _MergeRow:
    offsets: _RowOffsets
    alpha: object
    beta: object
    full: object


@dataclass(frozen=True)
class _VectorChunk:
    offset: object
    mask: object


@jit
def _make_vector_masks(d):
    full = full_mask()
    scalar, _ = update_mask(1, elem_bits=FP32_BITS)
    tail_size = d % VL
    if const_expr(tail_size == 0):
        tail_size = VL
    tail, _ = update_mask(tail_size, elem_bits=FP32_BITS)
    return _VectorMasks(full, scalar, tail)


@jit
def _load_updated_partial(buffers, offsets, off, mask):
    return vadd(
        vload(buffers.partial, offsets.fp + off),
        vcast(
            vload_unpack(
                buffers.delta_h, offsets.b16 + off, unpack_mode=UnpackMode.B16_TO_B32
            ),
            Float32,
            mask=mask,
        ),
        mask=mask,
    )


@jit
def _phase1_single(buffers, offsets, masks, query0):
    p0 = _load_updated_partial(buffers, offsets, 0, masks.tail)
    vstore(buffers.partial, offsets.fp, p0, masks.tail)
    dot = vmul(p0, query0, mask=masks.full)
    ssq = vmul(p0, p0, mask=masks.full)
    return dot, ssq


@jit
def _phase1_double(buffers, offsets, masks, queries):
    p0 = _load_updated_partial(buffers, offsets, 0, masks.full)
    p1 = _load_updated_partial(buffers, offsets, VL, masks.tail)
    vstore(buffers.partial, offsets.fp, p0, masks.full)
    vstore(buffers.partial, offsets.fp + VL, p1, masks.tail)
    dot = vadd(
        vmul(p1, queries[1], mask=masks.full),
        vmul(p0, queries[0], mask=masks.full),
        mask=masks.full,
    )
    ssq = vadd(
        vmul(p1, p1, mask=masks.full),
        vmul(p0, p0, mask=masks.full),
        mask=masks.full,
    )
    return dot, ssq


@jit
def _phase1_pair(buffers, offsets, off, masks, sums):
    dot0, dot1, ssq0, ssq1 = sums
    p0 = _load_updated_partial(buffers, offsets, off, masks.full)
    p1 = _load_updated_partial(buffers, offsets, off + VL, masks.full)
    q0 = vload(buffers.query, off)
    q1 = vload(buffers.query, off + VL)
    vstore(buffers.partial, offsets.fp + off, p0, masks.full)
    vstore(buffers.partial, offsets.fp + off + VL, p1, masks.full)
    dot0 = vadd(vmul(p0, q0, mask=masks.full), dot0, mask=masks.full)
    dot1 = vadd(vmul(p1, q1, mask=masks.full), dot1, mask=masks.full)
    ssq0 = vadd(vmul(p0, p0, mask=masks.full), ssq0, mask=masks.full)
    ssq1 = vadd(vmul(p1, p1, mask=masks.full), ssq1, mask=masks.full)
    return dot0, dot1, ssq0, ssq1


@jit
def _phase1_odd(buffers, offsets, off, masks, sums):
    dot0, dot1, ssq0, ssq1 = sums
    p0 = _load_updated_partial(buffers, offsets, off, masks.full)
    q0 = vload(buffers.query, off)
    vstore(buffers.partial, offsets.fp + off, p0, masks.full)
    dot0 = vadd(vmul(p0, q0, mask=masks.full), dot0, mask=masks.full)
    ssq0 = vadd(vmul(p0, p0, mask=masks.full), ssq0, mask=masks.full)
    return dot0, dot1, ssq0, ssq1


@jit
def _phase1_tail(config, buffers, offsets, masks, sums):
    dot0, dot1, ssq0, ssq1 = sums
    off1 = (config.d // VL) * VL
    p1 = _load_updated_partial(buffers, offsets, off1, masks.tail)
    # Clear tail operands, then accumulate under a full mask to retain sums.
    q1 = vmuls(vload(buffers.query, off1), 1.0, mask=masks.tail)
    vstore(buffers.partial, offsets.fp + off1, p1, masks.tail)
    if const_expr((config.d // VL) % 2 != 0):
        dot1 = vadd(vmul(p1, q1, mask=masks.full), dot1, mask=masks.full)
        ssq1 = vadd(vmul(p1, p1, mask=masks.full), ssq1, mask=masks.full)
    else:
        dot0 = vadd(vmul(p1, q1, mask=masks.full), dot0, mask=masks.full)
        ssq0 = vadd(vmul(p1, p1, mask=masks.full), ssq0, mask=masks.full)
    return dot0, dot1, ssq0, ssq1


@jit
def _phase1_large(config, buffers, offsets, masks):
    dot0 = vdups(0.0, Float32, mask=masks.full)
    dot1 = vdups(0.0, Float32, mask=masks.full)
    ssq0 = vdups(0.0, Float32, mask=masks.full)
    ssq1 = vdups(0.0, Float32, mask=masks.full)
    for pair in dsl_range(0, config.d // (2 * VL), 1):
        dot0, dot1, ssq0, ssq1 = _phase1_pair(
            buffers, offsets, pair * (2 * VL), masks, (dot0, dot1, ssq0, ssq1)
        )
    if const_expr((config.d // VL) % 2 != 0):
        dot0, dot1, ssq0, ssq1 = _phase1_odd(
            buffers,
            offsets,
            (config.d // (2 * VL)) * (2 * VL),
            masks,
            (dot0, dot1, ssq0, ssq1),
        )
    if const_expr(config.d % VL != 0):
        dot0, dot1, ssq0, ssq1 = _phase1_tail(
            config, buffers, offsets, masks, (dot0, dot1, ssq0, ssq1)
        )
    dot = vadd(dot0, dot1, mask=masks.full)
    ssq = vadd(ssq0, ssq1, mask=masks.full)
    return dot, ssq


@jit
def _store_score(config, destination, masks, sums, eps):
    score, row = destination
    dot, ssq = sums
    dot_sum = vreduce_sum(dot, mask=masks.full)
    square_sum = vreduce_sum(ssq, mask=masks.full)
    rms = vsqrt(
        vadds(
            vmuls(square_sum, 1.0 / config.d, mask=masks.scalar), eps, mask=masks.scalar
        ),
        mask=masks.scalar,
    )
    vstore_first(score, row, vdiv(dot_sum, rms, mask=masks.scalar, precision_mode=True))


# vmadd(a, b, c) computes a * b + c and can overwrite its first register.
# Phase 1 keeps separate multiply/add operations: p feeds dot and square sums.
# Phase 2 only uses vmadd on dead-after-use operands. Precision-mode divisions
# retain AscendC's 0-ULP, FTZ-true mode. Helpers inline inside the original VFs.
@jit
def _update_phase1(config, buffers, t_size, eps):
    config = _snapshot_vector_config(config)
    with vf(mode="simd"):
        masks = _make_vector_masks(config.d)
        query0 = None
        query1 = None
        if const_expr(config.d <= VL):
            query0 = vmuls(vload(buffers.query, 0), 1.0, mask=masks.tail)
        elif const_expr(config.d <= 2 * VL):
            query0 = vload(buffers.query, 0)
            query1 = vmuls(vload(buffers.query, VL), 1.0, mask=masks.tail)
        for row in dsl_range(0, t_size, 1):
            offsets = _RowOffsets(row * config.fp_stride, row * config.b16_stride)
            if const_expr(config.d <= VL):
                dot, ssq = _phase1_single(buffers, offsets, masks, query0)
            elif const_expr(config.d <= 2 * VL):
                dot, ssq = _phase1_double(buffers, offsets, masks, (query0, query1))
            else:
                dot, ssq = _phase1_large(config, buffers, offsets, masks)
            _store_score(config, (buffers.score, row), masks, (dot, ssq), eps)
        vmem_bar(mode="vst_vld")


@jit
def _merge_weights(config, buffers, row, full):
    maximum = vload_broadcast(buffers.stats, row)
    ell = vload_broadcast(buffers.stats, config.stats_stride + row)
    current_score = vload_broadcast(buffers.score, row)
    merged_max = vmax(maximum, current_score, mask=full)
    alpha = vexp_sub(maximum, merged_max, mask=full)
    beta = vexp_sub(current_score, merged_max, mask=full)
    merged_sum = vmadd(ell, alpha, beta, mask=full)
    inv_denominator = vdiv(
        vdups(1.0, Float32, mask=full), merged_sum, mask=full, precision_mode=True
    )
    alpha = vmul(alpha, inv_denominator, mask=full)
    beta = vmul(beta, inv_denominator, mask=full)
    return alpha, beta


@jit
def _merge_single(config, buffers, row, off, mask):
    p = vload(buffers.partial, row.offsets.fp + off)
    n = vload(buffers.numerator, row.offsets.fp + off)
    out = vmadd(n, row.alpha, vmul(p, row.beta, mask=mask), mask=mask)
    h = vcast(out, config.delta_dtype, mask=mask, rounding=RoundingMode.RN)
    vstore_pack(
        buffers.delta_h, row.offsets.b16 + off, h, mask, pack_mode=PackMode.B32_TO_B16
    )
    return vcast(h, Float32, mask=mask)


@jit
def _merge_pair(config, buffers, row, off, last_mask):
    p0 = vload(buffers.partial, row.offsets.fp + off)
    p1 = vload(buffers.partial, row.offsets.fp + off + VL)
    n0 = vload(buffers.numerator, row.offsets.fp + off)
    n1 = vload(buffers.numerator, row.offsets.fp + off + VL)
    out0 = vmadd(n0, row.alpha, vmul(p0, row.beta, mask=row.full), mask=row.full)
    out1 = vmadd(n1, row.alpha, vmul(p1, row.beta, mask=last_mask), mask=last_mask)
    h0 = vcast(out0, config.delta_dtype, mask=row.full, rounding=RoundingMode.RN)
    h1 = vcast(out1, config.delta_dtype, mask=last_mask, rounding=RoundingMode.RN)
    vstore_pack(
        buffers.delta_h,
        row.offsets.b16 + off,
        h0,
        row.full,
        pack_mode=PackMode.B32_TO_B16,
    )
    vstore_pack(
        buffers.delta_h,
        row.offsets.b16 + off + VL,
        h1,
        last_mask,
        pack_mode=PackMode.B32_TO_B16,
    )
    rounded0 = vcast(h0, Float32, mask=row.full)
    rounded1 = vcast(h1, Float32, mask=last_mask)
    return rounded0, rounded1


@jit
def _phase2_large(config, buffers, row, masks, sums):
    ssq0, ssq1 = sums
    for pair in dsl_range(0, config.d // (2 * VL), 1):
        rounded0, rounded1 = _merge_pair(
            config, buffers, row, pair * (2 * VL), masks.full
        )
        ssq0 = vadd(ssq0, vmul(rounded0, rounded0, mask=masks.full), mask=masks.full)
        ssq1 = vadd(ssq1, vmul(rounded1, rounded1, mask=masks.full), mask=masks.full)
    if const_expr((config.d // VL) % 2 != 0):
        rounded0 = _merge_single(
            config, buffers, row, (config.d // (2 * VL)) * (2 * VL), masks.full
        )
        ssq0 = vadd(ssq0, vmul(rounded0, rounded0, mask=masks.full), mask=masks.full)
    if const_expr(config.d % VL != 0):
        rounded1 = _merge_single(
            config, buffers, row, (config.d // VL) * VL, masks.tail
        )
        ssq1 = vadd(ssq1, vmul(rounded1, rounded1, mask=masks.full), mask=masks.full)
    return ssq0, ssq1


@jit
def _phase2_square_sum(config, buffers, row, masks):
    ssq0 = vdups(0.0, Float32, mask=masks.full)
    ssq1 = vdups(0.0, Float32, mask=masks.full)
    if const_expr(config.d <= VL):
        rounded = _merge_single(config, buffers, row, 0, masks.tail)
        # Keep the prior RMS reduction's full-segment/tail accumulator lanes.
        if const_expr(config.d == VL):
            ssq0 = vadd(ssq0, vmul(rounded, rounded, mask=masks.full), mask=masks.full)
        else:
            ssq1 = vadd(ssq1, vmul(rounded, rounded, mask=masks.full), mask=masks.full)
    elif const_expr(config.d <= 2 * VL):
        rounded0, rounded1 = _merge_pair(config, buffers, row, 0, masks.tail)
        ssq0 = vadd(ssq0, vmul(rounded0, rounded0, mask=masks.full), mask=masks.full)
        ssq1 = vadd(ssq1, vmul(rounded1, rounded1, mask=masks.full), mask=masks.full)
    else:
        ssq0, ssq1 = _phase2_large(config, buffers, row, masks, (ssq0, ssq1))
    return ssq0, ssq1


@jit
def _store_rstd(config, destination, masks, sums):
    score, row = destination
    ssq0, ssq1 = sums
    total = vreduce_sum(vadd(ssq0, ssq1, mask=masks.full), mask=masks.full)
    rms = vsqrt(
        vadds(
            vmuls(total, 1.0 / config.d, mask=masks.scalar),
            config.norm_eps,
            mask=masks.scalar,
        ),
        mask=masks.scalar,
    )
    one = vdups(1.0, Float32, mask=masks.scalar)
    # The score was consumed above; reuse its plane for this row's rstd.
    vstore_first(score, row, vdiv(one, rms, mask=masks.scalar, precision_mode=True))


@jit
def _update_phase2(config, buffers, t_size):
    """Generate rounded h and its RMS square sum in the same D pass."""
    config = _snapshot_vector_config(config)
    with vf(mode="simd"):
        masks = _make_vector_masks(config.d)
        for row in dsl_range(0, t_size, 1):
            alpha, beta = _merge_weights(config, buffers, row, masks.full)
            offsets = _RowOffsets(row * config.fp_stride, row * config.b16_stride)
            merged_row = _MergeRow(offsets, alpha, beta, masks.full)
            sums = _phase2_square_sum(config, buffers, merged_row, masks)
            _store_rstd(config, (buffers.score, row), masks, sums)
        vmem_bar(mode="vst_vld")


@jit
def _scale_chunk(config, buffers, base, chunk, rstd):
    h = vcast(
        vload_unpack(
            buffers.delta_h, base + chunk.offset, unpack_mode=UnpackMode.B16_TO_B32
        ),
        Float32,
        mask=chunk.mask,
    )
    gain = vcast(
        vload_unpack(buffers.gamma, chunk.offset, unpack_mode=UnpackMode.B16_TO_B32),
        Float32,
        mask=chunk.mask,
    )
    y = vmul(vmul(h, rstd, mask=chunk.mask), gain, mask=chunk.mask)
    vstore_pack(
        buffers.delta_h,
        base + chunk.offset,
        vcast(y, config.delta_dtype, mask=chunk.mask, rounding=RoundingMode.RN),
        chunk.mask,
        pack_mode=PackMode.B32_TO_B16,
    )


@jit
def _update_phase3_rms(config, buffers, t_size):
    """Scale rounded h using rstd computed while generating h."""
    config = _snapshot_vector_config(config)
    with vf(mode="simd"):
        full = full_mask()
        tail_size = config.d % VL
        if const_expr(tail_size == 0):
            tail_size = VL
        tail, _ = update_mask(tail_size, elem_bits=FP32_BITS)
        for row in dsl_range(0, t_size, 1):
            base = row * config.b16_stride
            rstd = vload_broadcast(buffers.score, row)
            for chunk in dsl_range(0, config.d // VL, 1):
                segment = _VectorChunk(chunk * VL, full)
                _scale_chunk(config, buffers, base, segment, rstd)
            if const_expr(config.d % VL != 0):
                segment = _VectorChunk((config.d // VL) * VL, tail)
                _scale_chunk(config, buffers, base, segment, rstd)
        vmem_bar(mode="vst_vld")


@dataclass(frozen=True)
class _GlobalBuffers:
    partial: Tensor
    delta: Tensor
    numerator: Tensor
    maximum: Tensor
    exp_sum: Tensor
    output: Tensor


class UpdateVector:
    """Select one slot per tile and retain its aliases through the final write."""

    def __init__(
        self, d: int, tile_t: int, depth: int, delta_dtype, norm_eps: float = 1e-6
    ):
        self.d = d
        self.tile_t = tile_t
        self.buffer_depth = depth
        self.fp_stride = _ceil_div(d, 8) * 8
        self.b16_stride = _ceil_div(d, 16) * 16
        self.stats_stride = _ceil_div(tile_t, 8) * 8
        self.delta_dtype = delta_dtype
        self.norm_eps = norm_eps
        self.partial = RingBuffer(
            MemLoc.UB, Float32, make_layout((tile_t, self.fp_stride)), depth=depth
        )
        self.delta = RingBuffer(
            MemLoc.UB, delta_dtype, make_layout((tile_t, self.b16_stride)), depth=depth
        )
        self.numerator = RingBuffer(
            MemLoc.UB, Float32, make_layout((tile_t, self.fp_stride)), depth=depth
        )
        self.query = RingBuffer(
            MemLoc.UB, Float32, make_layout((1, self.fp_stride)), depth=1
        ).next_write()
        self.stats = RingBuffer(
            MemLoc.UB, Float32, make_layout((2, self.stats_stride)), depth=depth
        )
        self.score = RingBuffer(
            MemLoc.UB, Float32, make_layout((1, self.stats_stride)), depth=depth
        )
        self.gamma = RingBuffer(
            MemLoc.UB, delta_dtype, make_layout((1, self.b16_stride)), depth=1
        ).next_write()

    def load_query(self, gm_query):
        slot = self.query
        mem_copy(tile_slice(slot, (1, self.d), (0, 0)), gm_query)

    @jit
    def row_tile(self, tensor, row, t_size):
        row_end = row + t_size
        d = self.d
        return tile_slice(tensor[row:row_end, 0:d], (self.tile_t, d), (0, 0))

    @jit
    def load_partial_delta(self, buffers, row, t_size):
        partial_in = self.partial.next_write()
        delta_in = self.delta.next_write()
        mem_copy(
            tile_slice(partial_in, (self.tile_t, self.d), (0, 0)),
            self.row_tile(buffers.partial, row, t_size),
        )
        mem_copy(
            tile_slice(delta_in, (self.tile_t, self.d), (0, 0)),
            self.row_tile(buffers.delta, row, t_size),
        )
        # Advance each cursor once per tile. Keep these aliases for all V/MTE3
        # accesses; RingBuffer selection itself neither copies data nor waits.
        partial = self.partial.next_read()
        delta_h = self.delta.next_read()
        score_out = self.score.next_write()
        return _UpdateBuffers(partial, delta_h, score_out, query=self.query)

    @jit
    def load_softmax_state(self, buffers, row, t_size):
        numerator_in = self.numerator.next_write()
        stats_in = self.stats.next_write()
        mem_copy(
            tile_slice(numerator_in, (self.tile_t, self.d), (0, 0)),
            self.row_tile(buffers.numerator, row, t_size),
        )
        row_end = row + t_size
        mem_copy(
            tile_slice(stats_in, (1, self.tile_t), (0, 0)),
            tile_slice(buffers.maximum[0:1, row:row_end], (1, self.tile_t), (0, 0)),
        )
        mem_copy(
            tile_slice(stats_in, (1, self.tile_t), (1, 0)),
            tile_slice(buffers.exp_sum[0:1, row:row_end], (1, self.tile_t), (0, 0)),
        )
        numerator = self.numerator.next_read()
        stats = self.stats.next_read()
        score = self.score.next_read()
        return numerator, stats, score

    @jit
    def process_tile(self, buffers, row, t_size, eps):
        update = self.load_partial_delta(buffers, row, t_size)
        _update_phase1(self, update, t_size, eps)
        # Partial is complete after phase 1. MTE3 and phase 2 both only read it.
        mem_copy(
            self.row_tile(buffers.partial, row, t_size),
            tile_slice(update.partial, (self.tile_t, self.d), (0, 0)),
        )
        numerator, stats, score = self.load_softmax_state(buffers, row, t_size)
        merged = _UpdateBuffers(
            update.partial,
            update.delta_h,
            score,
            numerator=numerator,
            stats=stats,
            gamma=self.gamma,
        )
        _update_phase2(self, merged, t_size)
        _update_phase3_rms(self, merged, t_size)
        mem_copy(
            self.row_tile(buffers.output, row, t_size),
            tile_slice(merged.delta_h, (self.tile_t, self.d), (0, 0)),
        )

    @jit
    def process_rows(self, buffers, row_start, row_end, eps):
        if const_expr(self.buffer_depth == 1):
            self.process_tile(buffers, row_start, row_end - row_start, eps)
        else:
            for row in dsl_range(row_start, row_end, self.tile_t):
                t_size = row_end - row
                if t_size > self.tile_t:
                    t_size = self.tile_t
                self.process_tile(buffers, row, t_size, eps)


@kernel(
    profile=cannbotdsl.ProfileSpec(
        name="block_attn_res_update_rms_norm",
        op_type="BlockAttnResUpdateRMSNorm",
    )
)
class UpdateKernel:
    """Full-D, multi-row tiles with a single-tile or ping-pong execution path."""

    def __init__(
        self,
        d: int,
        epsilon: float,
        delta_dtype,
        tile_t: int,
        buffer_depth: int,
        norm_eps: float = 1e-6,
    ):
        self.d = d
        self.epsilon = epsilon
        self.delta_dtype = delta_dtype
        self.tile_t = tile_t
        self.buffer_depth = buffer_depth
        self.norm_eps = norm_eps

    # Keep the flat kernel signature aligned with the host/export tensor ABI.
    def __call__(
        self,
        gm_partial: Tensor,
        gm_delta: Tensor,
        gm_query: Tensor,
        gm_numerator: Tensor,
        gm_max: Tensor,
        gm_sum: Tensor,
        gm_h: Tensor,
        gm_gamma: Tensor,
        rows_per_core: int,
    ):
        tokens = gm_partial.shape[0]
        row_start = get_block_idx() * rows_per_core
        row_end = row_start + rows_per_core
        if row_end > tokens:
            row_end = tokens
        # Host uses ceil(T / rows_per_core); retain a guard before any GM loads.
        if row_start < tokens:
            vector = UpdateVector(
                self.d, self.tile_t, self.buffer_depth, self.delta_dtype, self.norm_eps
            )
            vector.load_query(gm_query)
            gamma = vector.gamma
            mem_copy(tile_slice(gamma, (1, self.d), (0, 0)), gm_gamma)
            buffers = _GlobalBuffers(
                gm_partial, gm_delta, gm_numerator, gm_max, gm_sum, gm_h
            )
            vector.process_rows(buffers, row_start, row_end, self.epsilon)


class BlockAttnResUpdateRMSNorm:
    """Compiled entry used by :func:`block_attn_res_update_rms_norm`."""

    def __init__(self, d, delta_dtype, score_eps, norm_eps, tile_t, buffer_depth):
        _require(d == HIDDEN_SIZE, f"D 必须固定为 {HIDDEN_SIZE}")
        _require(delta_dtype == torch.bfloat16, "delta 必须为 bfloat16")
        self.d = d
        self.delta_dtype = _TORCH_TO_DSL[delta_dtype]
        self.epsilon = score_eps
        self.norm_eps = norm_eps
        self.tile_t = tile_t
        self.buffer_depth = buffer_depth

    # AOT TensorSpec binding requires individual tensor/scalar arguments; the
    # tensor declarations intentionally match the kernel's flat export ABI.
    @host
    def run(
        self,
        gm_partial: Tensor,
        gm_delta: Tensor,
        gm_query: Tensor,
        gm_numerator: Tensor,
        gm_max: Tensor,
        gm_sum: Tensor,
        gm_h: Tensor,
        gm_gamma: Tensor,
        rows_per_core: int,
        block_num: int,
    ):
        op = UpdateKernel(
            self.d,
            self.epsilon,
            self.delta_dtype,
            self.tile_t,
            self.buffer_depth,
            self.norm_eps,
        )
        op[block_num](
            gm_partial,
            gm_delta,
            gm_query,
            gm_numerator,
            gm_max,
            gm_sum,
            gm_h,
            gm_gamma,
            rows_per_core,
        )


@dataclass(frozen=True)
class _KernelConfig:
    """Static specialization fields shared by compilation and the cache key."""

    d: int
    delta_dtype: torch.dtype
    score_eps: float
    norm_eps: float
    tile_t: int
    buffer_depth: int


_COMPILED_KERNELS = {}
_COMPILED_KERNEL_LOCK = threading.Lock()


def _get_compiled_kernel(config, vector_cores=DEFAULT_BLOCK_NUM, ub_bytes=UB_BYTES):
    # T is dynamic; only static specialization and platform limits enter the key.
    key = (
        config.d,
        config.delta_dtype,
        config.score_eps,
        config.norm_eps,
        config.tile_t,
        config.buffer_depth,
        vector_cores,
        ub_bytes,
    )
    with _COMPILED_KERNEL_LOCK:
        if key in _COMPILED_KERNELS:
            return _COMPILED_KERNELS[key]

        # The public entry handles T=0 without launching; this kernel is nonempty-only.
        tokens = cannbotdsl.Dim("T", min=1)
        fake = cannbotdsl.TensorSpec
        dtype = _TORCH_TO_DSL[config.delta_dtype]
        row_shape = (tokens, config.d)
        stats_shape = (1, tokens)
        op = BlockAttnResUpdateRMSNorm(
            config.d,
            config.delta_dtype,
            config.score_eps,
            config.norm_eps,
            config.tile_t,
            config.buffer_depth,
        )
        compiled = cannbotdsl.compile(
            op.run,
            fake(row_shape, Float32),
            fake(row_shape, dtype),
            fake((1, config.d), Float32),
            fake(row_shape, Float32),
            fake(stats_shape, Float32),
            fake(stats_shape, Float32),
            fake(row_shape, dtype),
            fake((1, config.d), dtype),
            Int64,
            Int64,
        )
        _COMPILED_KERNELS[key] = compiled
        if len(_COMPILED_KERNELS) > EXECUTABLE_CACHE_CAPACITY:
            # Active callers retain their callable; its destructor releases resources.
            del _COMPILED_KERNELS[next(iter(_COMPILED_KERNELS))]
        return compiled


def _validate_input_shapes(inputs):
    partial_block, delta, pseudo_query, numerator, logit_max, exp_sum, gamma = inputs
    _require(partial_block.dim() == 2, "partial_block 必须为 [tokens,D]")
    tokens, d = map(int, partial_block.shape)
    _require(tokens >= 0, "T 必须 >= 0")
    _require(d == HIDDEN_SIZE, f"D 必须固定为 {HIDDEN_SIZE}")
    _require(delta.dtype == torch.bfloat16, "delta 必须为 bfloat16")
    expected_inputs = (
        ("partial_block", partial_block, (tokens, d), torch.float32),
        ("delta", delta, (tokens, d), delta.dtype),
        ("pseudo_query", pseudo_query, (d,), torch.float32),
        ("numerator", numerator, (tokens, d), torch.float32),
        ("logit_max", logit_max, (tokens,), torch.float32),
        ("exp_sum", exp_sum, (tokens,), torch.float32),
        ("gamma", gamma, (d,), delta.dtype),
    )
    for name, tensor, shape, dtype in expected_inputs:
        _require(
            tuple(tensor.shape) == shape and tensor.dtype == dtype,
            f"{name} 必须为 {shape} {dtype}",
        )
        _require(
            tensor.is_contiguous() and tensor.device == partial_block.device,
            f"{name} 必须 contiguous 且与 partial_block 同 device",
        )
    return tokens, d


def _launch_update_kernel(inputs, output, score_eps, norm_eps):
    partial_block, delta, pseudo_query, numerator, logit_max, exp_sum, gamma = inputs
    tokens, d = map(int, output.shape)
    with torch.npu.device(partial_block.device):
        device_index = partial_block.device.index
        stream = torch.npu.current_stream(device_index)
        effective_cores = int(
            cannbotdsl.get_platform_info(stream=stream).vector_core_num
        )
        ub_bytes = int(cannbotdsl.get_mem_size("ub")) - 8 * 1024
        _require(
            effective_cores > 0 and ub_bytes > 0, "Cannot query NPU Vector/UB limits"
        )
        block_num, rows_per_core, tile_t, buffer_depth = _get_tiling(
            tokens, d, effective_cores, ub_bytes
        )
        config = _KernelConfig(
            d, delta.dtype, score_eps, norm_eps, tile_t, buffer_depth
        )
        fn = _get_compiled_kernel(config, effective_cores, ub_bytes)
        fn(
            partial_block,
            delta,
            pseudo_query.reshape(1, d),
            numerator,
            logit_max.reshape(1, tokens),
            exp_sum.reshape(1, tokens),
            output,
            gamma.reshape(1, d),
            rows_per_core,
            block_num,
        )


# Keep the public tensor/epsilon ABI used by callers and the dispatcher schema.
def block_attn_res_update_rms_norm(
    partial_block: torch.Tensor,
    delta: torch.Tensor,
    pseudo_query: torch.Tensor,
    numerator: torch.Tensor,
    logit_max: torch.Tensor,
    exp_sum: torch.Tensor,
    gamma: torch.Tensor,
    score_eps: float = 1e-6,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Update partial in FP32 and return RMSNorm of the rounded BF16 h.

    partial_block/numerator: FP32 [T,D]; delta: BF16 [T,D].
    pseudo_query: FP32 [D]; logit_max/exp_sum: FP32 [T].
    gamma: [D], with the same dtype as delta. All inputs must be contiguous
    and on the same NPU. T >= 0, D == 7168; both epsilons must be positive.
    Only partial_block is mutated. The intermediate h stays in UB.
    """
    inputs = partial_block, delta, pseudo_query, numerator, logit_max, exp_sum, gamma
    tokens, d = _validate_input_shapes(inputs)
    score_eps = _normalize_eps(score_eps)
    norm_eps = _normalize_eps(norm_eps)
    _require(
        partial_block.device.type == "npu",
        "block_attn_res_update_rms_norm 仅支持 NPU Tensor",
    )
    output = torch.empty((tokens, d), dtype=delta.dtype, device=partial_block.device)
    if tokens == 0:
        return output
    _launch_update_kernel(inputs, output, score_eps, norm_eps)
    return output


def clear_caches():
    """Release compiled functions when there are no concurrent callers."""
    with _COMPILED_KERNEL_LOCK:
        for compiled in _COMPILED_KERNELS.values():
            close = getattr(compiled, "close", None)
            if callable(close):
                close()
        _COMPILED_KERNELS.clear()


@export("block_attn_res_update_rms_norm")
def export_block_attn_res_update_rms_norm():
    """Export dynamic-T BF16/D=7168 variants for dav-3510, with epsilons 1e-6.

    Other epsilon specializations remain available through Python JIT,
    but are not included in this default network export.
    """
    clear_caches()
    try:
        for tokens in (1, DEFAULT_BLOCK_NUM + 1):
            _, _, tile_t, buffer_depth = _get_tiling(
                tokens, HIDDEN_SIZE, DEFAULT_BLOCK_NUM, UB_BYTES
            )
            config = _KernelConfig(
                HIDDEN_SIZE, torch.bfloat16, 1e-6, 1e-6, tile_t, buffer_depth
            )
            _get_compiled_kernel(config)
    finally:
        clear_caches()


_GRAPH_LIBRARY = torch.library.Library(
    "cannbotdsl_block_attn_res_update_rms_norm", "DEF"
)
_GRAPH_LIBRARY.define(
    "block_attn_res_update_rms_norm("
    "Tensor(a!) partial_block, Tensor delta, Tensor pseudo_query, "
    "Tensor numerator, Tensor logit_max, Tensor exp_sum, Tensor gamma, "
    "float score_eps, float norm_eps"
    ") -> Tensor"
)


@torch.library.impl(_GRAPH_LIBRARY, "block_attn_res_update_rms_norm", "Meta")
# Match the exact positional ABI declared in the dispatcher schema.
def _block_attn_res_update_rms_norm_meta(
    partial_block,
    delta,
    pseudo_query,
    numerator,
    logit_max,
    exp_sum,
    gamma,
    score_eps,
    norm_eps,
):
    del pseudo_query, numerator, logit_max, exp_sum, gamma
    del score_eps, norm_eps
    _check(partial_block.dim() == 2, lambda: "partial_block 必须为 [T,D]")
    _check(delta.dim() == 2, lambda: "delta 必须为 [T,D]")
    _check(partial_block.shape[0] >= 0, lambda: "T 必须 >= 0")
    _check(partial_block.shape[1] == HIDDEN_SIZE, lambda: "D 必须固定为 7168")
    _check(
        delta.shape[0] == partial_block.shape[0],
        lambda: "delta 的 T 必须与 partial_block 相同",
    )
    _check(delta.shape[1] == HIDDEN_SIZE, lambda: "D 必须固定为 7168")
    _require(delta.dtype == torch.bfloat16, "delta 必须为 bfloat16")
    return torch.empty_like(delta, device="meta")


torch.library.impl(_GRAPH_LIBRARY, "block_attn_res_update_rms_norm", "PrivateUse1")(
    block_attn_res_update_rms_norm
)


block_attn_res_update_rms_norm_op = (
    torch.ops.cannbotdsl_block_attn_res_update_rms_norm.block_attn_res_update_rms_norm
)


# Old network code calls the public host function directly. Under Dynamo
# (torch.compile, fullgraph) that host body is untraceable, so during
# tracing the call is substituted with the registered dispatcher op;
# eager calls keep taking the host path unchanged.
substitute_in_graph(block_attn_res_update_rms_norm, can_constant_fold_through=False)(
    lambda *args, **kwargs: block_attn_res_update_rms_norm_op(*args, **kwargs)
)
