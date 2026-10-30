# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Ascend950 MX ``npu_quant_matmul`` sample.

Structure:
  1. Host-side ASWT tiling
  2. MX kernel with GM->L1->L0->L0C->GM data flow
  3. Torch-facing ``npu_quant_matmul`` wrapper

Formula: C[M,N] = dequant(A[M,K]) @ dequant(B[K,N])
MX scale ABI: ScaleA uses ScaleAND[M,G,2], ScaleB uses ScaleBND[G,N,2],
where G=ceil(K/64). Transpose is represented by stride metadata.
"""

__all__ = ["npu_quant_matmul"]

import functools
from math import gcd
from typing import NamedTuple

import cannbotdsl
import torch

from cannbotdsl import make_layout, dtypes, TensorSpec, get_mem_size, get_platform_info
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.sync import cube_fill_l1_zero
from cannbotdsl.tensor import MemLoc
from cannbotdsl.tensor import Tensor
from cannbotdsl.tensor import tile_slice
from cannbotdsl.ops.memcpy import copy_options, mem_copy


L1_CAPACITY_BYTES = get_mem_size("l1")
L0A_CAPACITY_BYTES = get_mem_size("l0a")
L0B_CAPACITY_BYTES = get_mem_size("l0b")
L0C_CAPACITY_BYTES = get_mem_size("l0c")
L0_NZ_OUTER_ALIGN = (
    64  # FP8 nz physical layout pads the L0A/L0B outer extent up to this multiple.
)
MX_GROUP_SIZE = 32  # One E8M0 scale is shared by every 32 K elements.
MX_K_ALIGN = 64  # Every 64 K elements map to one paired-scale group.
MX_SCALE_PAIR = 2
WINDOW_LEN = 4  # Number of output M tiles in one ASW window.
ESTIMATED_SCALE_K = 4096  # Initial data K span covered by one L1 scale window.
MX_L0C_PINGPONG_SCALE_K_L1_TARGET = 2048
MX_L0C_PINGPONG_OUTPUT_SIZE_LIMIT = 128 * 1024 * 1024
L2_CACHE_THRESHOLD_BYTES = 128 * 1024 * 1024 * 80 // 100
MMAD_BLOCK_SIZE = 256
CUBE_BLOCK = 16
AIC_NUM = get_platform_info().cube_core_num
BASIC_BLOCK_SIZE_128 = 128
LOAD_BALANCE_BASE_N_128_ALIGN_K_THRESHOLD = 2560
LOAD_BALANCE_THRESHOLD = 1792
MTE2_ADDRESS_ALIGN_SIZE = 128
# Floating-point tolerance used by tiling score comparisons.
SCORE_COMPARE_EPS = 1e-12
LOAD_BALANCE_ROOFLINE_MARGIN = 0.05
BASE_K_LIMIT = 4095

# Host transfer/compute cost model.
_MAX_HBM_BW_TBPS = 4.0
_MAX_L2_BW_TBPS = 5.2
_MTE2_BW_UTILIZATION = 0.9
_BYTES_PER_US_PER_TBPS = 1_000_000.0
_MX_CUBE_MACS_PER_CYCLE = 16 * 32 * 16
_CUBE_FREQ_MHZ = 1650.0


# ============================================================================
# 1. Host-side Tiling
# ============================================================================


class _TilingProfile(NamedTuple):
    """Per-dtype tiling knobs shared by every host-side sizing helper.

    ``element_bytes`` scales all L1/L0 byte models; ``l1_align_size`` and
    ``l2_align_size`` specify the inner-axis alignment for the dtype
    (32/128 elements for FP8, 64/256 for packed FP4 where one element is
    half a byte); ``cube_throughput``/``hbm_bandwidth`` feed the
    cube-vs-transfer bound estimates.
    """

    element_bytes: float
    l1_align_size: int
    l2_align_size: int
    cube_throughput: float
    hbm_bandwidth: float
    l2_bandwidth: float


FP8_TILING_PROFILE = _TilingProfile(
    element_bytes=1.0,
    l1_align_size=32,
    l2_align_size=128,
    cube_throughput=864.0,
    hbm_bandwidth=1.4,
    l2_bandwidth=5.2,
)
FP4_TILING_PROFILE = _TilingProfile(
    element_bytes=0.5,  # fp4x2 packs two E2M1 elements per byte.
    l1_align_size=64,
    l2_align_size=256,
    cube_throughput=1730.0,
    hbm_bandwidth=1.6,
    l2_bandwidth=5.2,
)


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def ceil_align(value, divisor):
    return ceil_div(value, divisor) * divisor


def floor_align(value, divisor):
    return value // divisor * divisor


def get_scale_k_len(k):
    """Return the contiguous L1 scale length for a data K extent."""
    return ceil_div(k, MX_K_ALIGN) * MX_SCALE_PAIR


def _optimize_base_block_for_load_balance(
    m,
    n,
    k,
    base_m,
    base_n,
    base_m_align=CUBE_BLOCK,
    base_n_align=CUBE_BLOCK,
    element_bytes=1.0,
):
    """Search ``baseM/baseN`` without adding a scheduling round.

    FP4 also searches larger grids, retaining aligned writeback and rejecting
    candidates that cross from compute-bound to MTE2-bound. Return (baseM, baseN).
    """
    is_fp4 = element_bytes == 0.5
    l1_align_size = MX_K_ALIGN if is_fp4 else MX_GROUP_SIZE
    origin_base_m, origin_base_n = base_m, base_n
    round_limit = ceil_div(ceil_div(m, base_m) * ceil_div(n, base_n), AIC_NUM)
    if round_limit == 1 or (round_limit > 3 and not is_fp4):
        return base_m, base_n

    origin_blocks = ceil_div(m, base_m) * ceil_div(n, base_n)
    origin_tail_blocks = origin_blocks % AIC_NUM or AIC_NUM
    origin_tail_used_cores = origin_tail_blocks * (AIC_NUM // origin_tail_blocks)
    origin_memory_score = (base_m + base_n) / (base_m * base_n)

    best_m, best_n = base_m, base_n
    best_balance = 0.0
    best_memory_score = float("inf")
    if is_fp4:
        best_balance = m * n / AIC_NUM / (round_limit * base_m * base_n)
        best_memory_score = origin_memory_score
        roofline_intensity = (
            _MX_CUBE_MACS_PER_CYCLE
            / element_bytes
            * _CUBE_FREQ_MHZ
            * AIC_NUM
            / (_MAX_L2_BW_TBPS * _MTE2_BW_UTILIZATION * _BYTES_PER_US_PER_TBPS)
        )
        origin_intensity = 1 / (origin_memory_score * element_bytes)
        require_base_n_128_align = (
            k < LOAD_BALANCE_BASE_N_128_ALIGN_K_THRESHOLD / element_bytes
            or round_limit > 3
        )
        keep_base_m = (
            round_limit > 3
            and (base_m_align == CUBE_BLOCK or base_n_align == CUBE_BLOCK)
            and k * element_bytes % MTE2_ADDRESS_ALIGN_SIZE != 0
        )
    m_tile_align = CUBE_BLOCK if base_m_align == CUBE_BLOCK else l1_align_size
    n_tile_align = CUBE_BLOCK if base_n_align == CUBE_BLOCK else l1_align_size
    found = False
    for search_m in range(ceil_align(base_m, base_m_align), 0, -base_m_align):
        for search_n in range(ceil_align(base_n, base_n_align), 0, -base_n_align):
            candidate_m, candidate_n = search_m, search_n
            if is_fp4:
                # Score the effective block used by the kernel, not an oversized search point.
                candidate_m = min(candidate_m, ceil_align(m, m_tile_align))
                candidate_n = min(candidate_n, ceil_align(n, n_tile_align))
            blocks = ceil_div(m, candidate_m) * ceil_div(n, candidate_n)
            rounds = ceil_div(blocks, AIC_NUM)
            if rounds > round_limit:
                break
            if candidate_m == base_m and candidate_n == base_n:
                continue
            tail_blocks = blocks % AIC_NUM or AIC_NUM
            memory_score = (candidate_m + candidate_n) / (candidate_m * candidate_n)
            if is_fp4:
                if require_base_n_128_align and candidate_n % BASIC_BLOCK_SIZE_128:
                    continue
                # Keep baseM for unaligned packed K to avoid extra B reloads.
                if keep_base_m and candidate_m != base_m:
                    continue
                candidate_intensity = 1 / (memory_score * element_bytes)
                if (
                    rounds == round_limit
                    and blocks > origin_blocks
                    and origin_intensity
                    > roofline_intensity * (1 + LOAD_BALANCE_ROOFLINE_MARGIN)
                    and candidate_intensity
                    < roofline_intensity * (1 - LOAD_BALANCE_ROOFLINE_MARGIN)
                ):
                    continue
            if tail_blocks < origin_tail_used_cores:
                continue
            if (
                tail_blocks == origin_tail_used_cores
                and memory_score > origin_memory_score + SCORE_COMPARE_EPS
            ):
                continue
            # Fraction of scheduled tile capacity that contains valid output.
            balance = m * n / AIC_NUM / (rounds * candidate_m * candidate_n)
            tie_break = (
                abs(balance - best_balance) <= SCORE_COMPARE_EPS
                and memory_score + (SCORE_COMPARE_EPS if is_fp4 else 0)
                < best_memory_score
            )
            if (
                (not found and not is_fp4)
                or balance > best_balance + SCORE_COMPARE_EPS
                or tie_break
            ):
                best_m, best_n = candidate_m, candidate_n
                best_balance = balance
                best_memory_score = memory_score
                found = True
    if not found:
        return base_m, base_n
    if (
        best_n % BASIC_BLOCK_SIZE_128 != 0
        and k < LOAD_BALANCE_BASE_N_128_ALIGN_K_THRESHOLD
    ):
        return origin_base_m, origin_base_n
    if best_m > m:
        best_m = ceil_align(m, m_tile_align if is_fp4 else base_m_align)
    if best_n > n:
        best_n = ceil_align(n, n_tile_align if is_fp4 else base_n_align)
    return best_m, best_n


def _get_base_block(
    m, n, k, transpose_a=False, transpose_b=True, profile=FP8_TILING_PROFILE
):
    """Select the per-AIC ``baseM/baseN/baseK`` compute block.

    Start from at most 256x256 with a 128-byte K block. For a
    multi-round launch, optimize final-round balance. For a single-round
    launch, redistribute the core grid and improve the tile aspect ratio.
    Finally, limit baseK by the double-buffered L0A/L0B capacity.
    """
    base_m_align = profile.l1_align_size if transpose_a else CUBE_BLOCK
    base_n_align = CUBE_BLOCK if transpose_b else profile.l1_align_size
    base_k_align = (
        MX_K_ALIGN if (transpose_a or not transpose_b) else profile.l2_align_size
    )
    adjust_base_m_align = profile.l2_align_size if transpose_a else CUBE_BLOCK
    adjust_base_n_align = CUBE_BLOCK if transpose_b else profile.l2_align_size
    # Tune the final round once all AICs are used.
    base_m = ceil_align(min(m, 256), base_m_align)
    base_n = ceil_align(min(n, 256), base_n_align)
    # The default K block covers 128 bytes: 256 packed FP4 or 128 FP8 values.
    # Retain a conservative K block for the packed transpose-A copy footprint.
    default_base_k = 256 if profile.element_bytes == 0.5 and not transpose_a else 128
    base_k = ceil_align(min(k, default_base_k), MX_K_ALIGN)
    if ceil_div(m, base_m) * ceil_div(n, base_n) >= AIC_NUM:
        base_m, base_n = _optimize_base_block_for_load_balance(
            m,
            n,
            k,
            base_m,
            base_n,
            adjust_base_m_align,
            adjust_base_n_align,
            element_bytes=profile.element_bytes,
        )
        return base_m, base_n, base_k

    # Single-round case: keep the less divisible axis and assign the remaining
    # cores to the other axis so that as many available AICs run as possible.
    m_max_tile = ceil_div(m, adjust_base_m_align)
    n_max_tile = ceil_div(n, adjust_base_n_align)
    # Leave underfilled FP4 inner-axis grids for single-round normalization.
    if (
        profile.element_bytes == 0.5
        and m_max_tile * n_max_tile < AIC_NUM
        and (transpose_a or not transpose_b)
    ):
        return base_m, base_n, base_k
    m_core = ceil_div(m, base_m)
    n_core = ceil_div(n, base_n)
    if m_max_tile < n_max_tile or (
        m_max_tile == n_max_tile and adjust_base_n_align == CUBE_BLOCK
    ):
        base_m = ceil_align(ceil_div(m, m_core), adjust_base_m_align)
        m_core = ceil_div(m, base_m)
        n_core = AIC_NUM // m_core
        base_n = ceil_align(ceil_div(n, n_core), adjust_base_n_align)
    else:
        base_n = ceil_align(ceil_div(n, n_core), adjust_base_n_align)
        n_core = ceil_div(n, base_n)
        m_core = AIC_NUM // n_core
        base_m = ceil_align(ceil_div(m, m_core), adjust_base_m_align)

    # Move baseM/baseN toward a square tile. This lowers the A/B transfer ratio
    # and may allow a larger baseK under the double-buffered L0 capacity limit.
    # When baseN >= 2*baseM, double n_core and derive m_core from BLOCK_DIM.
    while (
        base_n >= base_m * 2 and n_core < AIC_NUM // 2 and base_n != adjust_base_n_align
    ):
        n_core *= 2
        m_core = AIC_NUM // n_core
        base_m = ceil_align(ceil_div(m, m_core), adjust_base_m_align)
        base_n = ceil_align(ceil_div(n, n_core), adjust_base_n_align)
        m_core = ceil_div(m, base_m)
        n_core = ceil_div(n, base_n)
    # Apply the symmetric adjustment when the M direction is too long.
    while (
        base_m >= base_n * 2 and m_core < AIC_NUM // 2 and base_m != adjust_base_m_align
    ):
        m_core *= 2
        n_core = AIC_NUM // m_core
        base_m = ceil_align(ceil_div(m, m_core), adjust_base_m_align)
        base_n = ceil_align(ceil_div(n, n_core), adjust_base_n_align)
        m_core = ceil_div(m, base_m)
        n_core = ceil_div(n, base_n)

    if not transpose_a and transpose_b and base_n % BASIC_BLOCK_SIZE_128:
        # Compare both orientations by GM/L2 traffic and active-core utilization.
        # The common per-row K byte count cancels out of the load score.
        k_bytes = ceil_div(k, 2) if profile.element_bytes == 0.5 else k
        best_load_score = float("inf")
        best_grid_score = (float("inf"), True)
        for candidate_m, candidate_n in ((base_m, base_n), (base_n, base_m)):
            m_tiles = ceil_div(m, candidate_m)
            n_tiles = ceil_div(n, candidate_n)
            blocks = m_tiles * n_tiles
            if blocks > AIC_NUM:
                continue
            # Non-allocating loads come from GM; cached repeats can hit L2.
            a_gm_blocks = (
                blocks
                if n_tiles <= 2
                and m_tiles > n_tiles
                and k_bytes % MTE2_ADDRESS_ALIGN_SIZE == 0
                else m_tiles
            )
            b_gm_blocks = (
                blocks
                if m_tiles <= 2
                and n_tiles > m_tiles
                and k_bytes % MTE2_ADDRESS_ALIGN_SIZE == 0
                else n_tiles
            )
            gm_rows = a_gm_blocks * candidate_m + b_gm_blocks * candidate_n
            l2_rows = (blocks - a_gm_blocks) * candidate_m + (
                blocks - b_gm_blocks
            ) * candidate_n
            load_score = (
                AIC_NUM
                / blocks
                * (gm_rows / _MAX_HBM_BW_TBPS + l2_rows / _MAX_L2_BW_TBPS)
            )
            grid_score = (
                max(m_tiles, n_tiles) / min(m_tiles, n_tiles),
                candidate_n % BASIC_BLOCK_SIZE_128 != 0,
            )
            if load_score < best_load_score - SCORE_COMPARE_EPS or (
                abs(load_score - best_load_score) <= SCORE_COMPARE_EPS
                and grid_score < best_grid_score
            ):
                base_m, base_n = candidate_m, candidate_n
                best_load_score, best_grid_score = load_score, grid_score
    # L0A and L0B each hold two baseK tiles. Derive the baseK limit from their
    # 64 KiB capacity, align it down, and keep it within K and BASE_K_LIMIT.
    elem = profile.element_bytes
    # Keep the conservative L0 capacity estimate for transposed packed FP4 A.
    l0a_m = base_m * (2 if (elem != 1.0 and transpose_a) else 1)
    l0b_n = base_n
    # FP8 NZ outer extents use 64-element alignment in the physical L0 layout.
    if elem == 1.0:
        l0a_m = ceil_align(l0a_m, L0_NZ_OUTER_ALIGN)
        l0b_n = ceil_align(l0b_n, L0_NZ_OUTER_ALIGN)
    max_base_k = int(
        min(
            L0A_CAPACITY_BYTES // 2 / l0a_m / elem,
            L0B_CAPACITY_BYTES // 2 / l0b_n / elem,
        )
    )
    max_base_k = floor_align(max_base_k, base_k_align)
    if max_base_k >= base_k_align:
        base_k = min(ceil_align(k, base_k_align), max_base_k)
        if base_k > BASE_K_LIMIT:
            base_k = ceil_align(base_k // 2, base_k_align)
    base_m, base_n = _optimize_base_block_for_load_balance(
        m,
        n,
        k,
        base_m,
        base_n,
        adjust_base_m_align,
        adjust_base_n_align,
        element_bytes=profile.element_bytes,
    )
    return base_m, base_n, base_k


def _get_l1_used_bytes(
    base_m,
    base_n,
    k,
    k_l1,
    scale_k_l1,
    buffers,
    full_load,
    has_bias=False,
    profile=FP8_TILING_PROFILE,
):
    """Return the L1 bytes used by A/B data, their scales, and optional bias."""
    elem = profile.element_bytes
    a_k_extent = ceil_align(k, MX_K_ALIGN) if full_load else k_l1
    a_scale_k_extent = k if full_load else scale_k_l1
    a_buffers = 1 if full_load else buffers
    a_scale_buffers = 1 if full_load else 2
    a_bytes = base_m * a_k_extent * a_buffers * elem
    a_scale_bytes = base_m * get_scale_k_len(a_scale_k_extent) * a_scale_buffers
    b_bytes = base_n * k_l1 * buffers * elem
    b_scale_bytes = base_n * get_scale_k_len(scale_k_l1) * 2
    bias_bytes = base_n * 4 * 2 if has_bias else 0
    return a_bytes + b_bytes + a_scale_bytes + b_scale_bytes + bias_bytes


def _cal_normal_l1_tiling(
    base_m, base_n, k, base_k, has_bias=False, profile=FP8_TILING_PROFILE
):
    """Calculate ``kL1`` and ``scaleKL1`` for NORMAL_MODE."""
    elem = profile.element_bytes
    bias_l1 = base_n * 4 * 2 if has_bias else 0
    available_l1_size = L1_CAPACITY_BYTES - bias_l1
    base_ab_size = (base_m + base_n) * base_k * elem
    scale_base_size = base_m + base_n
    scale_k_l1 = min(k, ESTIMATED_SCALE_K)
    scale_size = scale_base_size * ceil_div(scale_k_l1, MX_K_ALIGN) * MX_SCALE_PAIR * 2

    # Estimate the largest stepK that fits double-buffered data and scales.
    depth = 1
    while depth * base_ab_size + scale_size <= available_l1_size:
        depth *= 2
        k_l1 = depth // 2 * base_k
        if k_l1 > scale_k_l1:
            scale_k_l1 = k_l1
            scale_size = (
                scale_base_size * ceil_div(scale_k_l1, MX_K_ALIGN) * MX_SCALE_PAIR * 2
            )
    depth = depth if depth == 1 else depth // 2

    # kL1 = stepK * baseK; NORMAL_MODE limits stepK to four.
    step_k = min(max(1, depth // 2), ceil_div(k, base_k), 4)
    k_l1 = step_k * base_k
    # Use the remaining L1 for scales and keep scaleKL1 a multiple of kL1.
    used_data_size = 2 * k_l1 * (base_m + base_n) * elem
    left_l1_size = max(0, available_l1_size - used_data_size)
    scale_group_size = (base_m + base_n) * MX_SCALE_PAIR * 2
    max_scale_k_l1 = left_l1_size // scale_group_size * MX_K_ALIGN
    scale_k_l1 = min(max_scale_k_l1, ceil_align(k, k_l1))
    scale_k_l1 = scale_k_l1 // k_l1 * k_l1
    return int(k_l1), int(max(k_l1, scale_k_l1))


def _cal_a_full_l1_tiling(
    base_m, base_n, k, base_k, has_bias=False, profile=FP8_TILING_PROFILE
):
    """Calculate K windows for AL1_FULL_LOAD with resident A and ScaleA."""
    elem = profile.element_bytes
    bias_l1 = base_n * 4 * 2 if has_bias else 0
    # Reserve resident A/ScaleA storage; B and ScaleB use the remaining L1.
    a_full_size = base_m * ceil_align(k, MX_K_ALIGN) * elem
    a_scale_size = base_m * get_scale_k_len(k)
    left_l1_size = max(
        0,
        L1_CAPACITY_BYTES - a_full_size - a_scale_size - bias_l1,
    )

    # Meet the 128-byte B transfer granularity before expanding stepK.
    step_k_base = max(1, ceil_div(128, base_k))
    base_b_with_scale = base_n * (base_k * elem + get_scale_k_len(base_k)) * step_k_base
    if left_l1_size >= 64 * 1024:
        step_k_scale = ceil_div(32 * 1024, base_b_with_scale)
    else:
        step_k_scale = ceil_div(left_l1_size // 2, base_b_with_scale)
    step_k = step_k_base * max(1, int(step_k_scale))
    # Merge at least two baseK blocks when another K window remains and fits.
    if step_k == 1 and k > base_k and left_l1_size > base_b_with_scale * 2:
        step_k = 2
    k_l1 = step_k * base_k
    # Use the space left by double-buffered B to increase ScaleB reuse.
    b_data_size = 2 * base_n * k_l1 * elem
    left_scale_size = max(0, left_l1_size - b_data_size)
    base_scale_b_size = base_n * get_scale_k_len(base_k)
    max_factor_by_k = max(1, min(127, k // k_l1))
    max_factor_by_l1 = min(64 * 1024, left_scale_size) // max(
        1, base_scale_b_size * step_k * 2
    )

    # Group ScaleB transfers to 128 bytes; do not reuse across kL1 if it cannot fit.
    scale_factor_base = max(
        1,
        ceil_div(128, get_scale_k_len(base_k)),
    )
    if scale_factor_base <= max_factor_by_k and max_factor_by_l1 >= scale_factor_base:
        scale_factor = min(
            max_factor_by_l1 // scale_factor_base * scale_factor_base,
            max_factor_by_k,
        )
    else:
        scale_factor = 1
    return int(k_l1), int(scale_factor * k_l1)


def _mx_full_k_load_bytes(outer_size, k, dtype_bytes):
    """Return full-K data and paired-scale bytes for ``outer_size`` rows."""
    k_aligned = ceil_align(k, MX_K_ALIGN)
    scale_k = ceil_div(k, MX_K_ALIGN) * MX_SCALE_PAIR
    return outer_size * k_aligned * dtype_bytes + outer_size * scale_k


def _is_mx_mte2_bound(
    m,
    n,
    k,
    base_m,
    base_n,
    transpose_a,
    transpose_b,
    a_dtype_bytes,
    b_dtype_bytes,
    has_bias,
    used_core_num,
):
    """Estimate whether GM->L1 traffic (MTE2) dominates the cube time.

    Mirrors IsMxMte2Bound: unaligned inner axes force the bound, otherwise
    compare the estimated MTE2 time against the estimated cube time
    (batch=1, non-full-load GM/L2 traffic split).
    """
    if used_core_num == 0 or base_m == 0 or base_n == 0:
        return False
    a_inner = m if transpose_a else k
    b_inner = k if transpose_b else n
    inner_aligned = (
        a_inner * a_dtype_bytes % MTE2_ADDRESS_ALIGN_SIZE == 0
        and b_inner * b_dtype_bytes % MTE2_ADDRESS_ALIGN_SIZE == 0
    )
    if not inner_aligned:
        return True

    a_bytes = _mx_full_k_load_bytes(m, k, a_dtype_bytes)
    b_bytes = _mx_full_k_load_bytes(n, k, b_dtype_bytes)
    bias_bytes = n * 4 if has_bias else 0
    m_blocks = ceil_div(m, base_m)
    n_blocks = ceil_div(n, base_n)
    a_load_count = n_blocks
    b_load_count = m_blocks
    bias_load_count = m_blocks if has_bias else 0
    # One GM pass per operand, the remaining rounds are expected to hit L2.
    a_gm_count = min(a_load_count, 1)
    b_gm_count = min(b_load_count, 1)
    bias_gm_count = min(bias_load_count, 1)
    gm_bytes = a_bytes * a_gm_count + b_bytes * b_gm_count + bias_bytes * bias_gm_count
    l2_bytes = (
        a_bytes * (a_load_count - a_gm_count)
        + b_bytes * (b_load_count - b_gm_count)
        + bias_bytes * (bias_load_count - bias_gm_count)
    )
    mte2_us = gm_bytes / (
        _MAX_HBM_BW_TBPS * _MTE2_BW_UTILIZATION * _BYTES_PER_US_PER_TBPS
    ) + l2_bytes / (_MAX_L2_BW_TBPS * _MTE2_BW_UTILIZATION * _BYTES_PER_US_PER_TBPS)

    aligned_m = ceil_align(m, CUBE_BLOCK)
    aligned_n = ceil_align(n, CUBE_BLOCK)
    aligned_k = ceil_align(k, MX_K_ALIGN)
    cube_us = (
        aligned_m
        * aligned_n
        * aligned_k
        / used_core_num
        / _MX_CUBE_MACS_PER_CYCLE
        / _CUBE_FREQ_MHZ
    )
    return mte2_us > cube_us


def _adjust_scale_k_l1_for_four_buffer(k, k_l1, scale_k_l1):
    """Shrink scaleKL1 for four buffers without adding a scale copy round."""
    if scale_k_l1 % ESTIMATED_SCALE_K == 0:
        return scale_k_l1
    half_k = ceil_div(k, 2)
    if half_k < scale_k_l1 < k:
        return min(scale_k_l1, ceil_align(half_k, k_l1))
    return scale_k_l1


def _cal_l1_tiling(
    m,
    n,
    k,
    base_m,
    base_n,
    base_k,
    full_load,
    transpose_a=False,
    transpose_b=True,
    has_bias=False,
    used_core_num=None,
    enable_l0c_pingpong=False,
    profile=FP8_TILING_PROFILE,
):
    """Return ``(kL1, scaleKL1, l1BufferNum)`` for the selected L1 mode.

    Select normal/full-load K windows, then try four A/B buffers, three
    when the double-buffer K coverage falls short and MTE2 is expected to
    dominate, and finally fall back to two.
    """
    cal_k_l1 = _cal_a_full_l1_tiling if full_load else _cal_normal_l1_tiling
    k_l1, scale_k_l1 = cal_k_l1(base_m, base_n, k, base_k, has_bias, profile=profile)
    if enable_l0c_pingpong:
        scale_k_l1 = _adjust_scale_k_l1_for_l0c_pingpong(m, n, k_l1, scale_k_l1)
    scale_k_l1 = _adjust_scale_k_l1_for_four_buffer(k, k_l1, scale_k_l1)
    candidates = [(k_l1, scale_k_l1)]
    # Try stepK=2 only when the GM-contiguous K axis meets its byte alignment.
    step_k_two_k_l1 = 2 * base_k
    step_two_aligned = (transpose_a and not transpose_b) or (
        k * profile.element_bytes % MTE2_ADDRESS_ALIGN_SIZE == 0
        and step_k_two_k_l1 * profile.element_bytes % 256 == 0
    )
    if k_l1 // base_k in (3, 4) and k_l1 * 2 < k and step_two_aligned:
        step_k_two_scale_k_l1 = _adjust_scale_k_l1_for_four_buffer(
            k,
            step_k_two_k_l1,
            scale_k_l1,
        )
        candidates.append((step_k_two_k_l1, step_k_two_scale_k_l1))

    # Preserve priority: current/reduced K with four buffers, then with three.
    for buffers in (4, 3):
        if buffers == 3:
            if profile.element_bytes == 0.5:
                break
            if not full_load and (
                k_l1 * 2 >= k
                or not _is_mx_mte2_bound(
                    m,
                    n,
                    k,
                    base_m,
                    base_n,
                    transpose_a,
                    transpose_b,
                    profile.element_bytes,
                    profile.element_bytes,
                    has_bias,
                    used_core_num,
                )
            ):
                break
        for candidate_k_l1, candidate_scale_k_l1 in candidates:
            if (
                _get_l1_used_bytes(
                    base_m,
                    base_n,
                    k,
                    candidate_k_l1,
                    candidate_scale_k_l1,
                    buffers,
                    full_load,
                    has_bias,
                    profile=profile,
                )
                > L1_CAPACITY_BYTES
            ):
                continue
            if buffers == 4:
                full_cover_scale_k_l1 = ceil_align(k, candidate_k_l1)
                if (
                    full_cover_scale_k_l1 > candidate_scale_k_l1
                    and _get_l1_used_bytes(
                        base_m,
                        base_n,
                        k,
                        candidate_k_l1,
                        full_cover_scale_k_l1,
                        buffers,
                        full_load,
                        has_bias,
                        profile=profile,
                    )
                    <= L1_CAPACITY_BYTES
                ):
                    candidate_scale_k_l1 = full_cover_scale_k_l1
            return candidate_k_l1, candidate_scale_k_l1, buffers

    return k_l1, scale_k_l1, 2


def _adjust_scale_k_l1_for_l0c_pingpong(
    m,
    n,
    k_l1,
    scale_k_l1,
):
    """Limit scaleKL1 to about 2048 K elements for L0C ping-pong.

    Keep the original window when the output is too large, L0C is not
    double-buffered, or scaleKL1 is already within the target.
    """
    output_size = m * n * 2
    if (
        output_size > MX_L0C_PINGPONG_OUTPUT_SIZE_LIMIT
        or scale_k_l1 <= MX_L0C_PINGPONG_SCALE_K_L1_TARGET
    ):
        return scale_k_l1
    scale_factor = max(
        1,
        MX_L0C_PINGPONG_SCALE_K_L1_TARGET // k_l1,
    )
    return max(k_l1, min(scale_k_l1, scale_factor * k_l1))


def _is_cube_bound(m, n, k, output_element_bytes, profile=FP8_TILING_PROFILE):
    """Estimate whether Cube compute time exceeds data transfer time."""
    elem = profile.element_bytes
    a_input_cost = m * k * elem
    b_input_cost = n * k * elem
    a_scale_cost = ceil_div(a_input_cost, MX_GROUP_SIZE)
    b_scale_cost = ceil_div(b_input_cost, MX_GROUP_SIZE)
    copy_out_bytes = m * n * output_element_bytes
    hbm_bytes = a_input_cost + a_scale_cost + b_input_cost + b_scale_cost
    a_l2_cost = m * k * elem * (ceil_div(n, MMAD_BLOCK_SIZE) - 1)
    b_l2_cost = n * k * elem * (ceil_div(m, MMAD_BLOCK_SIZE) - 1)
    l2_bytes = (
        a_l2_cost
        + ceil_div(a_l2_cost, MX_GROUP_SIZE)
        + b_l2_cost
        + ceil_div(b_l2_cost, MX_GROUP_SIZE)
    )
    l2_footprint = hbm_bytes + copy_out_bytes
    copy_out_bandwidth = (
        profile.l2_bandwidth
        if l2_footprint < L2_CACHE_THRESHOLD_BYTES
        else profile.hbm_bandwidth
    )
    transfer_cost = (
        hbm_bytes / profile.hbm_bandwidth
        + l2_bytes / profile.l2_bandwidth
        + copy_out_bytes / copy_out_bandwidth
    )
    compute_cost = 2.0 * m * n * k / profile.cube_throughput
    return compute_cost > transfer_cost


def _can_use_a_full_load(
    m,
    n,
    k,
    base_m,
    base_n,
    base_k,
    transpose_a=False,
    transpose_b=True,
    has_bias=False,
    profile=FP8_TILING_PROFILE,
):
    """Return whether the shape should use ``AL1_FULL_LOAD``.

    A and ScaleA must fit in half of L1, use fewer than four M tiles, and be
    reusable across N tiles. Enable full load directly with four buffers;
    otherwise require two buffers in NORMAL_MODE or more than 20 percent
    repeated A traffic.
    """
    elem = profile.element_bytes
    m_tiles = ceil_div(m, base_m)
    n_tiles = ceil_div(n, base_n)
    candidate = (
        base_m * ceil_align(k, MX_K_ALIGN) * elem <= L1_CAPACITY_BYTES // 2
        and m_tiles < WINDOW_LEN
        and AIC_NUM % m_tiles == 0
        and m_tiles * n_tiles > AIC_NUM
    )
    if not candidate:
        return False

    used_core_num = min(AIC_NUM, m_tiles * n_tiles)
    _, _, full_buffers = _cal_l1_tiling(
        m,
        n,
        k,
        base_m,
        base_n,
        base_k,
        True,
        transpose_a,
        transpose_b,
        has_bias,
        used_core_num,
        profile=profile,
    )
    if full_buffers > 2:
        return True

    _, _, normal_buffers = _cal_l1_tiling(
        m,
        n,
        k,
        base_m,
        base_n,
        base_k,
        False,
        transpose_a,
        transpose_b,
        has_bias,
        used_core_num,
        profile=profile,
    )
    repeated_a = (
        m * (ceil_align(k, MX_K_ALIGN) * elem + get_scale_k_len(k)) * (n_tiles - 1)
    )
    normal_bytes = (
        m * (ceil_align(k, MX_K_ALIGN) * elem + get_scale_k_len(k)) * n_tiles
        + n * (ceil_align(k, MX_K_ALIGN) * elem + get_scale_k_len(k)) * m_tiles
    )
    return normal_buffers == 2 or repeated_a / normal_bytes > 0.20


def _optimize_edge_basic_block(
    m,
    n,
    k,
    base_m,
    base_n,
    transpose_a=False,
    transpose_b=True,
):
    """Rebalance the final M/N blocks across adjacent outer-axis blocks."""
    m_tiles = ceil_div(m, base_m)
    n_tiles = ceil_div(n, base_n)
    m_base_tail_split_count = n_base_tail_split_count = 1
    m_tail_main = n_tail_main = 0
    if m_tiles == 1 or n_tiles == 1:
        return 1, 1, 0, 0

    inner_axis_aligned = k % MTE2_ADDRESS_ALIGN_SIZE == 0
    m_tail = m % base_m
    if (
        m_tail
        and not transpose_a
        and (inner_axis_aligned or m >= LOAD_BALANCE_THRESHOLD)
    ):
        merge_count_max = min((base_m - m_tail) // CUBE_BLOCK, m_tiles)
        window_size = min(WINDOW_LEN, m_tiles)
        main_window_count = m_tiles // window_size - 1
        tail_window_size = m_tiles - main_window_count * window_size
        best_cost = (main_window_count + 1) * base_m
        merged_window_count = 1
        for merge_len in range(tail_window_size - 1, merge_count_max, window_size):
            candidate = ceil_align(
                ceil_div(merge_len * base_m + m_tail, merge_len + 1),
                CUBE_BLOCK,
            )
            cost = (
                main_window_count + 1 - merged_window_count
            ) * base_m + merged_window_count * candidate
            if cost <= best_cost:
                best_cost = cost
                m_tail_main = candidate
                m_base_tail_split_count = merge_len + 1
            merged_window_count += 1

    n_tail = n % base_n
    balance_after_fixpipe = k < 1024 or (k == 1024 and n_tiles >= 8)
    is_n_inner_balanced = inner_axis_aligned or n >= LOAD_BALANCE_THRESHOLD
    if n_tail and transpose_b and not balance_after_fixpipe and is_n_inner_balanced:
        total_blocks = m_tiles * n_tiles
        total_rounds = ceil_div(total_blocks, AIC_NUM)
        main_rounds = ceil_div(
            (n_tiles - 1) * m_tiles + m_tiles % AIC_NUM,
            AIC_NUM,
        )
        is_aligned_window = m_tiles % AIC_NUM == 0 and (
            n_tiles % WINDOW_LEN == 0 or WINDOW_LEN % n_tiles == 0
        )
        if total_blocks <= AIC_NUM or is_aligned_window:
            main_rounds = total_rounds
        best_cost = main_rounds * base_n + (total_rounds - main_rounds) * n_tail
        merge_count_max = min((base_n - n_tail) // CUBE_BLOCK, n_tiles)
        for merge_len in range(1, merge_count_max):
            candidate = ceil_align(
                ceil_div(merge_len * base_n + n_tail, merge_len + 1),
                CUBE_BLOCK,
            )
            last = merge_len * (base_n - candidate) + n_tail
            new_main_rounds = 0
            new_tail_rounds = 0
            if merge_len < n_tiles - 1:
                new_main_rounds = ceil_div(
                    (n_tiles - 1 - merge_len) * m_tiles
                    + (merge_len + 1) * m_tiles % AIC_NUM,
                    AIC_NUM,
                )
            if merge_len:
                new_tail_rounds = min(
                    ceil_div(
                        merge_len * m_tiles + m_tiles % AIC_NUM,
                        AIC_NUM,
                    ),
                    total_rounds - new_main_rounds,
                )
            cost = (
                new_main_rounds * base_n
                + new_tail_rounds * candidate
                + (total_rounds - new_main_rounds - new_tail_rounds) * last
            )
            if cost < best_cost:
                best_cost = cost
                n_tail_main = candidate
                n_base_tail_split_count = merge_len + 1

    return (
        m_base_tail_split_count,
        n_base_tail_split_count,
        m_tail_main,
        n_tail_main,
    )


def _calc_tail_basic_block_split(
    m,
    n,
    base_m,
    base_n,
    aic_num,
    edge_tiling,
    transpose_a=False,
    transpose_b=True,
    profile=FP8_TILING_PROFILE,
):
    """Split each last-round block so idle AICs process its M/N sub-blocks."""
    m_tiles = ceil_div(m, base_m)
    n_tiles = ceil_div(n, base_n)
    tail_round = m_tiles * n_tiles % aic_num
    if tail_round == 0:
        return 1, 1

    m_tail = m - (m_tiles - 1) * base_m
    n_tail = n - (n_tiles - 1) * base_n
    m_base_tail_split_count, _, m_tail_main, _ = edge_tiling
    split_first_m = m_tail >= n_tail
    tile_max = aic_num // tail_round
    m_align = profile.l1_align_size if transpose_a else CUBE_BLOCK
    n_align = CUBE_BLOCK if transpose_b else profile.l1_align_size
    tail_base_m = m_tail_main if m_base_tail_split_count != 1 else base_m
    m_split_max = min(tile_max, ceil_div(min(m, tail_base_m), m_align))
    n_split_max = min(tile_max, ceil_div(min(n, base_n), n_align))
    first_max, second_max = (
        (m_split_max, n_split_max) if split_first_m else (n_split_max, m_split_max)
    )
    first_size, second_size = (
        (min(m, tail_base_m), min(n, base_n))
        if split_first_m
        else (min(n, base_n), min(m, tail_base_m))
    )
    first_inner_axis, second_inner_axis = (
        (transpose_a, not transpose_b)
        if split_first_m
        else (not transpose_b, transpose_a)
    )
    first_tail_size, second_tail_size = (
        (m_tail, n_tail) if split_first_m else (n_tail, m_tail)
    )
    first_packed_axis = profile.element_bytes == 0.5 and first_inner_axis
    second_packed_axis = profile.element_bytes == 0.5 and second_inner_axis
    single_window = m_tiles * n_tiles <= aic_num
    # Single-window splits become aligned base blocks before GM staging.
    # Only multi-window sub-blocks need byte-aligned raw split extents.
    first_split = second_split = 1
    first_valid = second_valid = 1
    first_aligned = second_aligned = 1
    while (
        first_split < first_max
        and (first_split + 1) * second_split * tail_round <= aic_num
    ) or (
        second_split < second_max
        and first_split * (second_split + 1) * tail_round <= aic_num
    ):
        if (
            first_split < first_max
            and (first_split + 1) * second_split * tail_round <= aic_num
        ):
            first_split += 1
            first_byte_aligned = (
                not first_packed_axis
                or single_window
                or (
                    ceil_div(first_size, first_split) % 2 == 0
                    and ceil_div(first_tail_size, first_split) % 2 == 0
                )
            )
            if first_byte_aligned:
                first_valid = first_split
                if (
                    first_inner_axis
                    and single_window
                    and ceil_div(first_size, first_split) % profile.l1_align_size == 0
                ):
                    first_aligned = first_split
        if (
            second_split < second_max
            and first_split * (second_split + 1) * tail_round <= aic_num
        ):
            second_split += 1
            second_byte_aligned = (
                not second_packed_axis
                or single_window
                or (
                    ceil_div(second_size, second_split) % 2 == 0
                    and ceil_div(second_tail_size, second_split) % 2 == 0
                )
            )
            if second_byte_aligned:
                second_valid = second_split
                if (
                    second_inner_axis
                    and single_window
                    and ceil_div(second_size, second_split) % profile.l1_align_size == 0
                ):
                    second_aligned = second_split
    first_split = first_aligned if first_aligned != 1 else first_valid
    second_split = second_aligned if second_aligned != 1 else second_valid
    return (first_split, second_split) if split_first_m else (second_split, first_split)


class QbmmMxTiling(NamedTuple):
    """Host tiling data consumed by the block scheduler and kernel."""

    base_m: int
    base_n: int
    base_k: int
    k_l1: int
    scale_k_l1: int
    l1_buffers: int
    l0c_buffers: int
    used_core_num: int
    full_load: bool
    m_tail_tile: int
    n_tail_tile: int
    m_base_tail_split_count: int
    n_base_tail_split_count: int
    m_tail_main: int
    n_tail_main: int
    l0c_tile_n: int = 0  # Zero keeps baseN; otherwise each L0C slot holds one N panel.
    scale_b_full_load: bool = False
    split_n_final: bool = False


def get_qbmm_mx_tile_config(
    m,
    n,
    k,
    transpose_a=False,
    transpose_b=True,
    has_bias=False,
    output_dtype=None,
    profile=FP8_TILING_PROFILE,
):
    """Generate the complete MX host tiling for one shape."""
    is_fp8 = profile.element_bytes == 1.0
    is_fp8_false_true = is_fp8 and not transpose_a and transpose_b
    # Both GM-contiguous inner extents must satisfy the dtype's byte alignment.
    inner_axis_align_bytes = 256 if profile.element_bytes == 0.5 else 128
    pingpong_aligned = (
        m if transpose_a else k
    ) * profile.element_bytes % inner_axis_align_bytes == 0 and (
        k if transpose_b else n
    ) * profile.element_bytes % inner_axis_align_bytes == 0
    # L0C ping-pong doubles the accumulator footprint, so it needs 16-bit
    # FixPipe output, aligned tiles, and a cube-bound shape where overlapping
    # the next tile's MMAD with FixPipe actually hides latency.
    enable_l0c_pingpong = (
        output_dtype in (torch.float16, torch.bfloat16)
        and pingpong_aligned
        and _is_cube_bound(m, n, k, 2, profile=profile)
    )
    base_m, base_n, base_k = _get_base_block(
        m, n, k, transpose_a, transpose_b, profile=profile
    )
    full_load = _can_use_a_full_load(
        m,
        n,
        k,
        base_m,
        base_n,
        base_k,
        transpose_a,
        transpose_b,
        has_bias,
        profile=profile,
    )
    total_tiles = ceil_div(m, base_m) * ceil_div(n, base_n)
    if not is_fp8:
        # Packed FP4: whole-tile GM staging requires tile-aligned outer
        # blocks, so keep the plain grid (no edge-basic-block merging).
        optimized_edge_tiling = (1, 1, 0, 0)
    else:
        optimized_edge_tiling = _optimize_edge_basic_block(
            m,
            n,
            k,
            base_m,
            base_n,
            transpose_a,
            transpose_b,
        )
    m_tail_tile, n_tail_tile = _calc_tail_basic_block_split(
        m,
        n,
        base_m,
        base_n,
        AIC_NUM,
        optimized_edge_tiling,
        transpose_a,
        transpose_b,
        profile=profile,
    )
    # Normalize a single-round tail split into real base blocks before L1 tiling.
    if (
        not is_fp8
        and not full_load
        and total_tiles < AIC_NUM
        and (m_tail_tile > 1 or n_tail_tile > 1)
    ):
        m_align = profile.l1_align_size if transpose_a else CUBE_BLOCK
        n_align = CUBE_BLOCK if transpose_b else profile.l1_align_size
        new_base_m = ceil_align(ceil_div(min(m, base_m), m_tail_tile), m_align)
        new_base_n = ceil_align(ceil_div(min(n, base_n), n_tail_tile), n_align)
        if (
            new_base_m <= base_m
            and new_base_n <= base_n
            and (new_base_m != base_m or new_base_n != base_n)
        ):
            base_m, base_n = new_base_m, new_base_n
            m_tail_tile, n_tail_tile = 1, 1
            total_tiles = ceil_div(m, base_m) * ceil_div(n, base_n)
    l0c_buffers = 2 if base_m * base_n * 4 * 2 <= L0C_CAPACITY_BYTES else 1
    k_l1, scale_k_l1, l1_buffers = _cal_l1_tiling(
        m,
        n,
        k,
        base_m,
        base_n,
        base_k,
        full_load,
        transpose_a,
        transpose_b,
        has_bias,
        min(AIC_NUM, total_tiles),
        enable_l0c_pingpong,
        profile=profile,
    )
    l0c_tile_n = 0
    normal_fp8_false_true = is_fp8_false_true and not has_bias and not full_load
    # Short single-wave K: smaller two-MAD L1 windows shorten the initial fill.
    if (
        normal_fp8_false_true
        and ceil_div(m, base_m) == 1
        and total_tiles <= AIC_NUM
        and k_l1 < k < 2 * k_l1
        and k_l1 // base_k >= 3
        and k % BASIC_BLOCK_SIZE_128 == 0
        and output_dtype in (torch.float16, torch.bfloat16)
    ):
        short_base_k = min(base_k, BASIC_BLOCK_SIZE_128)
        short_k_l1 = 2 * short_base_k
        short_scale_k_l1 = ceil_align(k, short_k_l1)
        if (
            _get_l1_used_bytes(
                base_m,
                base_n,
                k,
                short_k_l1,
                short_scale_k_l1,
                4,
                False,
                profile=profile,
            )
            <= L1_CAPACITY_BYTES
        ):
            base_k, k_l1, scale_k_l1 = short_base_k, short_k_l1, short_scale_k_l1
            l1_buffers = 4
    # Single-M multi-wave loads can stage two N panels together and retain A in L0.
    # Fit data/scales in L1, then compare the busiest-core load volume.
    if (
        normal_fp8_false_true
        and ceil_div(m, base_m) == 1
        and total_tiles >= 2 * AIC_NUM
        and k % MTE2_ADDRESS_ALIGN_SIZE == 0
        and output_dtype in (torch.float16, torch.bfloat16)
    ):
        candidates = [(base_n, k_l1, scale_k_l1, l1_buffers, 0)]
        max_l0c_n = floor_align(
            min(
                L0B_CAPACITY_BYTES // (2 * base_k),
                L0C_CAPACITY_BYTES // (8 * base_m),
            ),
            L0_NZ_OUTER_ALIGN,
        )
        wide_k_l1 = 2 * base_k
        for wide_base_n in range(
            min(2 * max_l0c_n, 2 * MMAD_BLOCK_SIZE), base_n, -CUBE_BLOCK
        ):
            data_bytes = (base_m + wide_base_n) * wide_k_l1
            scale_bytes = (base_m + wide_base_n) * get_scale_k_len(wide_k_l1) * 2
            k_windows = ceil_div(k, wide_k_l1)
            # Reserve Scale space for at most two GM loads before sizing data stages.
            initial_scale_factor = min(k_windows, max(4, ceil_div(k_windows, 2)))
            candidate_buffers = min(
                4,
                (L1_CAPACITY_BYTES - scale_bytes * initial_scale_factor) // data_bytes,
            )
            if candidate_buffers < 3:
                continue
            scale_factor = min(
                k_windows,
                (L1_CAPACITY_BYTES - candidate_buffers * data_bytes) // scale_bytes,
            )
            # Avoid fragmenting the public K-contiguous Scale into many DMA requests.
            if ceil_div(k_windows, scale_factor) > 2:
                continue
            candidates.append(
                (
                    wide_base_n,
                    wide_k_l1,
                    scale_factor * wide_k_l1,
                    candidate_buffers,
                    ceil_align(ceil_div(wide_base_n, 2), CUBE_BLOCK),
                )
            )

        best_score = float("inf")
        for (
            candidate_n,
            candidate_k_l1,
            candidate_scale_k_l1,
            candidate_buffers,
            candidate_l0c_n,
        ) in candidates:
            blocks = ceil_div(n, candidate_n)
            rounds = ceil_div(blocks, AIC_NUM)
            candidate_edge_tiling = (
                optimized_edge_tiling
                if candidate_l0c_n == 0
                else _optimize_edge_basic_block(
                    m,
                    n,
                    k,
                    base_m,
                    candidate_n,
                    transpose_a,
                    transpose_b,
                )
            )
            m_split, n_split = _calc_tail_basic_block_split(
                m,
                n,
                base_m,
                candidate_n,
                AIC_NUM,
                candidate_edge_tiling,
                transpose_a,
                transpose_b,
                profile=profile,
            )
            tail_blocks = blocks % AIC_NUM
            split_tail = tail_blocks > 0 and tail_blocks * m_split * n_split <= AIC_NUM
            m_load_rows = (rounds - 1) * m + (ceil_div(m, m_split) if split_tail else m)
            n_load_rows = (rounds - 1) * candidate_n + (
                ceil_div(candidate_n, n_split) if split_tail else candidate_n
            )
            # Public paired scales are K-contiguous; count covered 128B sectors.
            scale_row_bytes = get_scale_k_len(k)
            scale_window_bytes = get_scale_k_len(candidate_scale_k_l1)
            scale_stride_gcd = gcd(scale_row_bytes, MTE2_ADDRESS_ALIGN_SIZE)
            scale_window_period = scale_stride_gcd // gcd(
                scale_window_bytes, scale_stride_gcd
            )
            scale_window_count = ceil_div(scale_row_bytes, scale_window_bytes)
            scale_load_bytes = (
                ceil_align(scale_row_bytes, scale_stride_gcd)
                + scale_window_count * MTE2_ADDRESS_ALIGN_SIZE
                - scale_stride_gcd
                * (1 + (scale_window_count - 1) // scale_window_period)
            )
            cold_a_rows = m / AIC_NUM
            score = (k + scale_load_bytes) * (
                (cold_a_rows + n_load_rows) / _MAX_HBM_BW_TBPS
                + (m_load_rows - cold_a_rows) / _MAX_L2_BW_TBPS
            )
            if score < best_score - SCORE_COMPARE_EPS:
                best_score = score
                best_tiling = (
                    candidate_n,
                    candidate_k_l1,
                    candidate_scale_k_l1,
                    candidate_buffers,
                    candidate_l0c_n,
                )
                best_edge_tiling, best_tail_split = (
                    candidate_edge_tiling,
                    (m_split, n_split),
                )
        base_n, k_l1, scale_k_l1, l1_buffers, l0c_tile_n = best_tiling
        optimized_edge_tiling = best_edge_tiling
        m_tail_tile, n_tail_tile = best_tail_split
        total_tiles = ceil_div(m, base_m) * ceil_div(n, base_n)
        if l0c_tile_n:
            l0c_buffers = 2
            if base_n >= 2 * BASIC_BLOCK_SIZE_128 and l0c_tile_n < MMAD_BLOCK_SIZE:
                l0c_tile_n = MMAD_BLOCK_SIZE
    # Resident A: combine two B panels while preserving at least two N tasks/core.
    if (
        full_load
        and is_fp8_false_true
        and not has_bias
        and total_tiles >= 4 * AIC_NUM
        and k >= 3 * k_l1
        and output_dtype in (torch.float16, torch.bfloat16)
    ):
        wide_base_n = 2 * base_n
        full_scale_k_l1 = ceil_align(k, k_l1)
        if (
            base_m * base_n * 4 * 2 <= L0C_CAPACITY_BYTES
            and _get_l1_used_bytes(
                base_m, wide_base_n, k, k_l1, full_scale_k_l1, 2, True, profile=profile
            )
            <= L1_CAPACITY_BYTES
        ):
            l0c_tile_n, base_n = base_n, wide_base_n
            scale_k_l1, l1_buffers, l0c_buffers = full_scale_k_l1, 2, 2
            total_tiles = ceil_div(m, base_m) * ceil_div(n, base_n)
    # Two N accumulators let FixPipe read the first while Cube finishes the second.
    # Full-K scales load once per tile; preserve the fixed-N ScaleB reuse path.
    if (
        output_dtype in (torch.float16, torch.bfloat16)
        and pingpong_aligned
        and l0c_buffers == 1
        and l0c_tile_n == 0
        and normal_fp8_false_true
        and ceil_div(m, base_m) > 1
        and total_tiles >= 2 * AIC_NUM
        and base_n % (2 * CUBE_BLOCK) == 0
        and (
            enable_l0c_pingpong
            or (scale_k_l1 >= k and ceil_div(n, base_n) > WINDOW_LEN)
        )
    ):
        l0c_tile_n, l0c_buffers = base_n // 2, 2
    # Three data stages may shrink a full-K Scale window to at most two loads.
    if normal_fp8_false_true and l0c_tile_n > 0 and l1_buffers == 2:
        scale_capacity = L1_CAPACITY_BYTES - 3 * (base_m + base_n) * k_l1
        scale_window = floor_align(
            scale_capacity // ((base_m + base_n) * MX_SCALE_PAIR * 2) * MX_K_ALIGN,
            k_l1,
        )
        if (scale_k_l1 < k and scale_window >= scale_k_l1) or (
            scale_window < scale_k_l1 and k <= 2 * scale_window
        ):
            scale_k_l1 = min(ceil_align(k, k_l1), scale_window)
            l1_buffers = 3
    edge_tiling = (1, 1, 0, 0) if full_load else optimized_edge_tiling
    if full_load:
        used_core_num = min(AIC_NUM, total_tiles)
    elif total_tiles > AIC_NUM or total_tiles % AIC_NUM == 0:
        used_core_num = AIC_NUM
    else:
        used_core_num = total_tiles * m_tail_tile * n_tail_tile
    # An even number of whole ASW windows per core stride preserves its N tile.
    # Retain ScaleB in one L1 slot only when this also enables three data stages.
    m_tiles, n_tiles = ceil_div(m, base_m), ceil_div(n, base_n)
    scale_b_full_load = (
        is_fp8
        and not full_load
        and not has_bias
        and l1_buffers == 2
        and n_tiles <= WINDOW_LEN
        and m_tiles % WINDOW_LEN == 0
        and used_core_num % (2 * WINDOW_LEN * n_tiles) == 0
        and m_tail_tile == n_tail_tile == 1
        and edge_tiling[1] == 1
        and total_tiles > used_core_num
        and k % base_k == 0
        # Replace the two streaming ScaleB slots with one full-K slot.
        and (
            _get_l1_used_bytes(
                base_m, base_n, k, k_l1, scale_k_l1, 3, False, profile=profile
            )
            - base_n * get_scale_k_len(scale_k_l1) * 2
            + base_n * get_scale_k_len(k)
        )
        <= L1_CAPACITY_BYTES
    )
    if scale_b_full_load:
        l1_buffers = 3
    # Accumulate full N until the final baseK, then overlap the two panel stores.
    split_n_final = (
        l0c_tile_n > 0
        and l0c_buffers == 2
        and normal_fp8_false_true
        and not scale_b_full_load
        and k > scale_k_l1
        and m % base_m == n % base_n == k % k_l1 == 0
        and m_tiles % WINDOW_LEN == 0
        and n_tiles >= 8
        and total_tiles % used_core_num == 0
        and m_tail_tile == n_tail_tile == 1
        and base_m * base_n * 4 <= L0C_CAPACITY_BYTES
        and base_n * base_k * 2 <= L0B_CAPACITY_BYTES
    )
    if split_n_final:
        l0c_tile_n, l0c_buffers = 0, 1
    return QbmmMxTiling(
        base_m=base_m,
        base_n=base_n,
        base_k=base_k,
        k_l1=k_l1,
        scale_k_l1=scale_k_l1,
        l1_buffers=l1_buffers,
        l0c_buffers=l0c_buffers,
        used_core_num=used_core_num,
        full_load=full_load,
        m_tail_tile=m_tail_tile,
        n_tail_tile=n_tail_tile,
        m_base_tail_split_count=edge_tiling[0],
        n_base_tail_split_count=edge_tiling[1],
        m_tail_main=edge_tiling[2],
        n_tail_main=edge_tiling[3],
        l0c_tile_n=l0c_tile_n,
        scale_b_full_load=scale_b_full_load,
        split_n_final=split_n_final,
    )


# ============================================================================
# 2. Block Scheduler and Kernel
# ============================================================================


class QbmmMxBlockScheduler:
    """Map each AIC round to one output block and its valid M/N region."""

    def __init__(self, m, n, tiling):
        base_m, base_n = tiling.base_m, tiling.base_n
        self.base_m, self.base_n = base_m, base_n
        self.used_core_num = tiling.used_core_num
        self.m_tiles = ceil_div(m, base_m)
        self.n_tiles = ceil_div(n, base_n)
        self.total_tiles = self.m_tiles * self.n_tiles

        self.m_base_tail_split_count = tiling.m_base_tail_split_count
        self.n_base_tail_split_count = tiling.n_base_tail_split_count
        self.m_base_normal_count = self.m_tiles - tiling.m_base_tail_split_count
        self.n_base_normal_count = self.n_tiles - tiling.n_base_tail_split_count
        merged_m = m - self.m_base_normal_count * base_m
        merged_n = n - self.n_base_normal_count * base_n
        self.m_base_tail_main = (
            merged_m if tiling.m_base_tail_split_count == 1 else tiling.m_tail_main
        )
        self.n_base_tail_main = (
            merged_n if tiling.n_base_tail_split_count == 1 else tiling.n_tail_main
        )
        self.m_base_tail_last = (
            merged_m - (tiling.m_base_tail_split_count - 1) * self.m_base_tail_main
        )
        self.n_base_tail_last = (
            merged_n - (tiling.n_base_tail_split_count - 1) * self.n_base_tail_main
        )

        self.main_window = min(WINDOW_LEN, self.m_tiles)
        self.main_row = ceil_div(self.m_tiles, self.main_window) - 1
        self.tail_window = self.m_tiles - self.main_window * self.main_row
        self.total_main_tiles = self.main_row * self.main_window * self.n_tiles

        self.end_block_idx = (self.total_tiles - 1) % tiling.used_core_num
        total_tail_tile = tiling.m_tail_tile * tiling.n_tail_tile
        self.use_tail_split = (
            total_tail_tile > 1
            and (self.end_block_idx + 1) * total_tail_tile <= tiling.used_core_num
        )
        self.m_tail_tile = tiling.m_tail_tile if self.use_tail_split else 1
        self.n_tail_tile = tiling.n_tail_tile if self.use_tail_split else 1
        self.total_tail_tile = self.m_tail_tile * self.n_tail_tile
        self.tail_win_block_cnt = min(self.total_tiles, self.end_block_idx + 1)
        self.new_end_block_idx = self.end_block_idx + self.tail_win_block_cnt * (
            self.total_tail_tile - 1
        )

        # Swizzle each 4x8 tile group while retaining the ASW/tail paths.
        self.use_xor_n = (
            (tiling.l0c_tile_n > 0 or tiling.split_n_final)
            and m % base_m == 0
            and n % base_n == 0
            and self.m_tiles % WINDOW_LEN == 0
            and self.n_tiles >= 8
            and not self.use_tail_split
        )

    @jit
    def get_round_count(self, block_idx):
        """Return the number of output-block rounds assigned to this AIC."""
        round_count = ceil_div(self.total_tiles, self.used_core_num)
        if block_idx > self.end_block_idx:
            round_count -= 1
        if (
            const_expr(self.use_tail_split)
            and self.end_block_idx < block_idx <= self.new_end_block_idx
        ):
            round_count += 1
        return round_count

    @jit
    def get_tile_idx(self, block_idx, round_idx, round_count):
        """Return the ASW M/N index and final-round split index for this AIC."""
        tail_split = False
        logical_idx = round_idx * self.used_core_num + block_idx
        sub_m_idx = 0
        sub_n_idx = 0
        if const_expr(self.use_tail_split):
            tail_split = (
                round_idx == round_count - 1 and block_idx <= self.new_end_block_idx
            )
            if tail_split:
                logical_idx = (
                    round_idx * self.used_core_num + block_idx // self.total_tail_tile
                )
                split_idx = block_idx % self.total_tail_tile
                sub_m_idx = split_idx % self.m_tail_tile
                sub_n_idx = split_idx // self.m_tail_tile

        if const_expr(self.use_xor_n):
            group_idx = logical_idx // (WINDOW_LEN * self.n_tiles)
            within = logical_idx % (WINDOW_LEN * self.n_tiles)
            m_local = within % WINDOW_LEN
            m_idx = group_idx * WINDOW_LEN + m_local
            n_groups = self.n_tiles // 8
            group_size = WINDOW_LEN * 8
            full_blocks = n_groups * group_size
            n_idx = 0
            if within < full_blocks:
                n_group = within // group_size
                n_local = ((within % group_size) // WINDOW_LEN) ^ m_local
                n_group = n_group + (group_idx % 2) * (n_groups - 1 - 2 * n_group)
                n_idx = n_group * 8 + n_local
            else:
                n_idx = n_groups * 8 + (within - full_blocks) // WINDOW_LEN
        else:
            m_idx = 0
            n_forward = 0
            row_idx = 0
            if logical_idx < self.total_main_tiles:
                row_idx = logical_idx // (self.n_tiles * self.main_window)
                m_idx = row_idx * self.main_window + logical_idx % self.main_window
                n_forward = (logical_idx // self.main_window) % self.n_tiles
            else:
                tail_idx = logical_idx - self.total_main_tiles
                row_idx = self.main_row
                m_idx = self.main_row * self.main_window + tail_idx % self.tail_window
                n_forward = (tail_idx // self.tail_window) % self.n_tiles
            reverse = row_idx % 2
            n_idx = n_forward + reverse * ((self.n_tiles - 1) - 2 * n_forward)
        return m_idx, n_idx, tail_split, sub_m_idx, sub_n_idx

    @jit
    def get_block(self, m_idx, n_idx, tail_split, sub_m_idx, sub_n_idx):
        """Return the GM start and valid M/N shape of one scheduled block."""
        current_m = self.base_m
        if m_idx >= self.m_base_normal_count:
            current_m = (
                self.m_base_tail_last
                if m_idx == self.m_tiles - 1
                else self.m_base_tail_main
            )
        current_n = self.base_n
        if n_idx >= self.n_base_normal_count:
            current_n = (
                self.n_base_tail_last
                if n_idx == self.n_tiles - 1
                else self.n_base_tail_main
            )

        m_offset = 0
        n_offset = 0
        if tail_split:
            sub_m = ceil_div(current_m, self.m_tail_tile)
            sub_n = ceil_div(current_n, self.n_tail_tile)
            m_offset = sub_m_idx * sub_m
            n_offset = sub_n_idx * sub_n
            current_m = min(current_m - m_offset, sub_m)
            current_n = min(current_n - n_offset, sub_n)
        m_start = m_idx * self.base_m
        if m_idx > self.m_base_normal_count:
            m_start -= (m_idx - self.m_base_normal_count) * (
                self.base_m - self.m_base_tail_main
            )
        n_start = n_idx * self.base_n
        if n_idx > self.n_base_normal_count:
            n_start -= (n_idx - self.n_base_normal_count) * (
                self.base_n - self.n_base_tail_main
            )
        return (
            m_start + m_offset,
            n_start + n_offset,
            current_m,
            current_n,
        )


class QbmmMxKernel:
    def __init__(
        self,
        m,
        n,
        k,
        tiling,
        a_dtype=dtypes.float8_e4m3fn,
        b_dtype=dtypes.float8_e4m3fn,
        transpose_a=False,
        transpose_b=True,
        has_bias=False,
    ):
        """Create channels and copy engines from the host tiling."""
        self.k = k
        self.base_m, self.base_n = tiling.base_m, tiling.base_n
        self.l0c_tile_n = tiling.l0c_tile_n or self.base_n
        self.split_n = self.l0c_tile_n != self.base_n
        self.split_n_final = tiling.split_n_final
        self.base_k, self.k_l1 = tiling.base_k, tiling.k_l1
        self.l1_buffers = tiling.l1_buffers
        self.l0c_buffers = tiling.l0c_buffers
        self.a_dtype, self.b_dtype = a_dtype, b_dtype
        self.full_load = tiling.full_load
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.used_core_num = tiling.used_core_num
        self.has_bias = has_bias
        self.block_scheduler = QbmmMxBlockScheduler(m, n, tiling)
        self.k_l1_tiles = ceil_div(k, tiling.k_l1)
        self.scale_l1_tiles = ceil_div(k, tiling.scale_k_l1)
        self.scale_k_l1_factor = tiling.scale_k_l1 // tiling.k_l1
        self.step_k = tiling.k_l1 // tiling.base_k
        self.k_l0_tiles = ceil_div(k, tiling.base_k)
        # Contiguous L1 scale lengths for baseK, scaleKL1, and full K.
        self.scale_k_l0_len = self.base_k // MX_GROUP_SIZE
        self.scale_k_l1_len = get_scale_k_len(tiling.scale_k_l1)
        self.scale_k_len = get_scale_k_len(k)
        self.scale_b_full_load = tiling.scale_b_full_load
        # One/two tiles on one axis use NOTALLOC_KEEP for the less-reused operand.
        # Preserve GM/cache-line alignment; load modes are 0=NORMAL_FIRST_VICTIM, 4=NOTALLOC_KEEP.
        b_is_fp4 = b_dtype == dtypes.fp4x2_e2m1
        b_inner_axis = k if transpose_b else n
        b_inner_axis_bytes = ceil_div(b_inner_axis, 2) if b_is_fp4 else b_inner_axis
        b_inner_axis_aligned = (
            b_inner_axis_bytes != 0
            and b_inner_axis_bytes % MTE2_ADDRESS_ALIGN_SIZE == 0
        )
        b_cache_line_align = (
            FP4_TILING_PROFILE.l2_align_size
            if b_is_fp4
            else FP8_TILING_PROFILE.l2_align_size
        )
        b_cache_line_aligned = transpose_b or tiling.base_n % b_cache_line_align == 0
        b_use_notalloc_keep = (
            self.block_scheduler.m_tiles <= 2
            and self.block_scheduler.n_tiles > self.block_scheduler.m_tiles
            and b_inner_axis_aligned
            and b_cache_line_aligned
        )
        a_is_fp4 = a_dtype == dtypes.fp4x2_e2m1
        a_inner_axis = m if transpose_a else k
        a_inner_axis_bytes = ceil_div(a_inner_axis, 2) if a_is_fp4 else a_inner_axis
        a_cache_line_align = (
            FP4_TILING_PROFILE.l2_align_size
            if a_is_fp4
            else FP8_TILING_PROFILE.l2_align_size
        )
        a_use_notalloc_keep = (
            self.block_scheduler.n_tiles <= 2
            and self.block_scheduler.m_tiles > self.block_scheduler.n_tiles
            and a_inner_axis_bytes != 0
            and a_inner_axis_bytes % MTE2_ADDRESS_ALIGN_SIZE == 0
            and (not transpose_a or tiling.base_m % a_cache_line_align == 0)
        )
        self.l2_cache_ctl_a = 4 if a_use_notalloc_keep else 0
        self.l2_cache_ctl_b = 4 if b_use_notalloc_keep else 0
        # Stream long ScaleB windows together with their non-allocating B loads.
        if (
            self.split_n
            and not self.full_load
            and self.scale_k_l1_len >= MTE2_ADDRESS_ALIGN_SIZE
            and self.l2_cache_ctl_b == 4
        ):
            self.l2_cache_ctl_scale_b = 4
        elif self.block_scheduler.use_xor_n:
            self.l2_cache_ctl_scale_b = 0
        else:
            self.l2_cache_ctl_scale_b = 1
        # NOTALLOC_CLEAN avoids allocating streamed output in L2.
        self.l2_cache_ctl_c = (
            4
            if (
                n % L0_NZ_OUTER_ALIGN == 0
                and self.l0c_tile_n % L0_NZ_OUTER_ALIGN == 0
                and not (self.split_n and self.base_n < 2 * self.l0c_tile_n)
            )
            else 1
        )

    @jit
    def _zero_transposed_k_tail(self, slot, valid_k, physical_k, outer, c0):
        padded_k = ceil_align(valid_k, MX_K_ALIGN)
        missing_rows = padded_k - valid_k
        cube_fill_l1_zero(
            slot,
            offset_elems=valid_k * c0,
            repeat=ceil_div(outer, c0),
            blk_num=missing_rows,
            dst_gap=physical_k - missing_rows,
        )

    @jit
    def _compute_output_tile(
        self,
        out_gm,
        a_gm,
        b_gm,
        scale_a_gm,
        scale_b_gm,
        l1_a,
        l1_b,
        l1_scale_a,
        l1_scale_b,
        l0a,
        l0b,
        l0c,
        l1_bias,
        l1_bt,
        m_idx,
        n_idx,
        bias_gm,
        tail_split=False,
        sub_m_idx=0,
        sub_n_idx=0,
    ):
        """Compute one output tile with the shared L0C and FixPipe path."""
        m_start, n_start, current_m, current_n = self.block_scheduler.get_block(
            m_idx,
            n_idx,
            tail_split,
            sub_m_idx,
            sub_n_idx,
        )
        if current_m > 0 and current_n > 0:
            m_end = m_start + current_m
            n_end = n_start + current_n
            out_tile = out_gm[m_start:m_end, n_start:n_end]
            # One L0C slot per output tile: the whole K-accumulation and the
            # FixPipe read below share it; the produce cursor advances once per
            # tile so depth>1 rotates slots across tiles (FixPipe overlap).
            first_out = out_tile
            if const_expr(self.split_n):
                first_out = out_gm[
                    m_start:m_end, n_start : min(n_start + self.l0c_tile_n, n_end)
                ]
            l0c_slot = l0c.next_write(bare=True).view(
                make_layout(first_out.shape), dtype=dtypes.float32, data_format="nz"
            )
            l0c_second_slot = l0c_slot
            if const_expr(self.split_n):
                second_out = out_gm[
                    m_start:m_end, min(n_start + self.l0c_tile_n, n_end) : n_end
                ]
                l0c_second_slot = l0c.next_write(bare=True).view(
                    make_layout(second_out.shape),
                    dtype=dtypes.float32,
                    data_format="nz",
                )

            bias = None
            if const_expr(self.has_bias):
                bias_tile = bias_gm[n_start:n_end,]
                mem_copy(l1_bias.next_write(), bias_tile)
                mem_copy(l1_bt.next_write(), l1_bias.next_read())
                bias = l1_bt.next_read()

            # Each scaleKL1 window covers scale_k_l1_factor data kL1 windows.
            scale_group_count = self.scale_k_l1_len // MX_SCALE_PAIR
            # Fold window bounds and slot selection without expanding long or tail-K paths.
            for scale_l1_idx in dsl_range(
                self.scale_l1_tiles,
                unroll_full=const_expr(
                    self.block_scheduler.use_xor_n
                    and self.scale_l1_tiles <= 4
                    and self.k % self.k_l1 == 0
                ),
            ):
                scale_group_start = scale_l1_idx * scale_group_count
                scale_group_end = min(
                    scale_group_start + scale_group_count,
                    ceil_div(self.k, MX_K_ALIGN),
                )
                # Copy ScaleA from GM to L1 unless it is resident in full-load mode.
                if const_expr(not self.full_load):
                    if const_expr(self.transpose_a):
                        scale_a_gm_tile = scale_a_gm[
                            scale_group_start:scale_group_end,
                            m_start:m_end,
                            0:MX_SCALE_PAIR,
                        ]
                    else:
                        scale_a_gm_tile = scale_a_gm[
                            m_start:m_end,
                            scale_group_start:scale_group_end,
                            0:MX_SCALE_PAIR,
                        ]
                # Select each Scale read slot once per scaleKL1 window.
                scale_a_l1_rd = l1_scale_a.next_read().permute(1, 0)
                # ScaleB GM→L1
                if const_expr(self.transpose_b):
                    scale_b_gm_tile = scale_b_gm[
                        n_start:n_end,
                        scale_group_start:scale_group_end,
                        0:MX_SCALE_PAIR,
                    ]
                else:
                    scale_b_gm_tile = scale_b_gm[
                        scale_group_start:scale_group_end,
                        n_start:n_end,
                        0:MX_SCALE_PAIR,
                    ]
                scale_b_l1_rd = l1_scale_b.next_read()

                # Process only valid data windows in the final scaleKL1 window.
                current_scale_k_l1_factor = min(
                    self.scale_k_l1_factor,
                    self.k_l1_tiles - scale_l1_idx * self.scale_k_l1_factor,
                )
                for k_l1_idx_in_scale in range(current_scale_k_l1_factor):
                    global_k_l1_idx = (
                        scale_l1_idx * self.scale_k_l1_factor + k_l1_idx_in_scale
                    )
                    k_start = global_k_l1_idx * self.k_l1
                    k_end = (
                        k_start + self.k_l1
                        if const_expr(
                            self.k % self.k_l1 == 0 and self.block_scheduler.m_tiles > 1
                        )
                        else min(k_start + self.k_l1, self.k)
                    )
                    # Copy A/B from GM to L1; full-load mode copies only B here.
                    if const_expr(not self.full_load):
                        if const_expr(self.transpose_a):
                            a_gm_tile = a_gm[k_start:k_end, m_start:m_end]
                        else:
                            a_gm_tile = a_gm[m_start:m_end, k_start:k_end]
                        a_write = l1_a.next_write()
                        mem_copy(
                            a_write,
                            a_gm_tile,
                            options=copy_options(l2_cache_ctl=self.l2_cache_ctl_a),
                        )
                        if const_expr(self.transpose_a and self.k % MX_K_ALIGN != 0):
                            if k_end == self.k:
                                self._zero_transposed_k_tail(
                                    a_write,
                                    k_end - k_start,
                                    self.k_l1,
                                    self.base_m,
                                    self.a_dtype.c0_elems,
                                )
                    if const_expr(self.transpose_b):
                        b_gm_tile = b_gm[n_start:n_end, k_start:k_end]
                    else:
                        b_gm_tile = b_gm[k_start:k_end, n_start:n_end]
                    b_write = l1_b.next_write()
                    mem_copy(
                        b_write,
                        b_gm_tile,
                        options=copy_options(l2_cache_ctl=self.l2_cache_ctl_b),
                    )
                    if const_expr(not self.transpose_b and self.k % MX_K_ALIGN != 0):
                        if k_end == self.k:
                            self._zero_transposed_k_tail(
                                b_write,
                                k_end - k_start,
                                self.k_l1,
                                self.base_n,
                                self.b_dtype.c0_elems,
                            )
                    # Stage A/B before the first Scale load in each scaleKL1 window.
                    if k_l1_idx_in_scale == 0:
                        if const_expr(not self.full_load):
                            mem_copy(
                                l1_scale_a.next_write().permute(1, 0),
                                scale_a_gm_tile.permute((1, 0, 2)).flatten(-2, -1)
                                if self.transpose_a
                                else scale_a_gm_tile.flatten(-2, -1),
                                options=copy_options(l2_cache_ctl=0),
                            )
                        if const_expr(not self.scale_b_full_load):
                            mem_copy(
                                l1_scale_b.next_write(),
                                scale_b_gm_tile.flatten(-2, -1).permute(1, 0)
                                if self.transpose_b
                                else scale_b_gm_tile.permute((0, 2, 1)).flatten(0, 1),
                                options=copy_options(
                                    l2_cache_ctl=self.l2_cache_ctl_scale_b
                                ),
                            )
                    # Same kL1 window for the L1→L0A/L0B reads below: select
                    # once per window (full_load keeps A resident in slot 0).
                    a_l1_rd = l1_a.next_read()
                    b_l1_rd = l1_b.next_read()

                    # The final kL1 window contains only its valid baseK blocks.
                    current_k_l1 = (
                        self.k_l1
                        if const_expr(
                            self.k % self.k_l1 == 0 and self.block_scheduler.m_tiles > 1
                        )
                        else min(self.k_l1, self.k - k_start)
                    )
                    current_step_k = (
                        self.step_k
                        if const_expr(
                            self.k % self.k_l1 == 0 and self.block_scheduler.m_tiles > 1
                        )
                        else ceil_div(current_k_l1, self.base_k)
                    )
                    # Move each baseK block to L0A/L0B and accumulate with MX MMAD.
                    for k_l0_idx in dsl_range(current_step_k, unroll=2):
                        global_k_l0_idx = global_k_l1_idx * self.step_k + k_l0_idx
                        valid_k = (
                            self.base_k
                            if const_expr(self.k % self.base_k == 0)
                            else min(
                                self.base_k, self.k - global_k_l0_idx * self.base_k
                            )
                        )
                        valid_scale_k = ceil_div(valid_k, MX_K_ALIGN) * MX_SCALE_PAIR
                        scale_l0_idx_in_l1 = k_l1_idx_in_scale * self.step_k + k_l0_idx
                        # Full-load A uses a global K index; streamed A uses a local one.
                        a_k_l0_idx = (
                            global_k_l0_idx if const_expr(self.full_load) else k_l0_idx
                        )
                        scale_a_k_l0_idx = (
                            global_k_l0_idx
                            if const_expr(self.full_load)
                            else scale_l0_idx_in_l1
                        )
                        scale_b_k_l0_idx = (
                            global_k_l0_idx
                            if const_expr(self.scale_b_full_load)
                            else scale_l0_idx_in_l1
                        )
                        # L1→L0A + MX ScaleA
                        a_l1_tile = tile_slice(
                            a_l1_rd,
                            (self.base_k, self.base_m)
                            if const_expr(self.transpose_a)
                            else (self.base_m, self.base_k),
                            (a_k_l0_idx, 0)
                            if const_expr(self.transpose_a)
                            else (0, a_k_l0_idx),
                        )
                        scale_a_l1_tile = tile_slice(
                            scale_a_l1_rd,
                            (self.base_m, self.scale_k_l0_len),
                            (0, scale_a_k_l0_idx),
                        )
                        a_source = (
                            a_l1_tile.permute(1, 0) if self.transpose_a else a_l1_tile
                        )
                        a_source = a_source[: first_out.shape[0], :valid_k]
                        a_ready = l0a.next_write(bare=True)
                        a_ready = a_ready.view(make_layout(a_source.shape))
                        mem_copy(
                            a_ready,
                            a_source,
                            options=copy_options(
                                mx_scale=scale_a_l1_tile[
                                    : a_source.shape[0], :valid_scale_k
                                ]
                            ),
                        )
                        # Use two N panels only for the final baseK accumulation.
                        if (
                            const_expr(self.split_n_final)
                            and global_k_l0_idx + 1 == self.k_l0_tiles
                        ):
                            for n_panel in dsl_range(2, unroll_full=True):
                                n_panel_size = self.base_n // 2
                                b_l1_tile = tile_slice(
                                    b_l1_rd,
                                    (n_panel_size, self.base_k),
                                    (n_panel, k_l0_idx),
                                )
                                scale_b_l1_tile = tile_slice(
                                    scale_b_l1_rd,
                                    (self.scale_k_l0_len, n_panel_size),
                                    (scale_b_k_l0_idx, n_panel),
                                )
                                b_ready = l0b.next_write(bare=True)
                                b_ready = b_ready.view(make_layout(b_l1_tile.shape))
                                mem_copy(
                                    b_ready,
                                    b_l1_tile,
                                    options=copy_options(mx_scale=scale_b_l1_tile),
                                )
                                c_panel_tile = tile_slice(
                                    l0c_slot, (self.base_m, n_panel_size), (0, n_panel)
                                )
                                matmul(
                                    c_panel_tile,
                                    a_ready,
                                    b_ready,
                                    init=False,
                                    unit_flag=3,
                                    mx_mode=True,
                                )
                                panel_start = n_start + n_panel * n_panel_size
                                mem_copy(
                                    out_gm[
                                        m_start:m_end,
                                        panel_start : panel_start + n_panel_size,
                                    ],
                                    c_panel_tile,
                                    options=copy_options(
                                        unit_flag=3, l2_cache_ctl=self.l2_cache_ctl_c
                                    ),
                                )
                        else:
                            # L1→L0B + MX ScaleB
                            b_l1_tile = tile_slice(
                                b_l1_rd,
                                (self.l0c_tile_n, self.base_k)
                                if const_expr(self.transpose_b)
                                else (self.base_k, self.l0c_tile_n),
                                (0, k_l0_idx)
                                if const_expr(self.transpose_b)
                                else (k_l0_idx, 0),
                            )
                            scale_b_l1_tile = tile_slice(
                                scale_b_l1_rd,
                                (self.scale_k_l0_len, self.l0c_tile_n),
                                (scale_b_k_l0_idx, 0),
                            )
                            b_source = (
                                b_l1_tile.permute(1, 0)
                                if not self.transpose_b
                                else b_l1_tile
                            )
                            b_source = b_source[: first_out.shape[1], :valid_k]
                            b_ready = l0b.next_write(bare=True)
                            b_ready = b_ready.view(make_layout(b_source.shape))
                            mem_copy(
                                b_ready,
                                b_source,
                                options=copy_options(
                                    mx_scale=scale_b_l1_tile[
                                        :valid_scale_k, : b_source.shape[0]
                                    ]
                                ),
                            )
                            # MX MMAD
                            is_final_acc = global_k_l0_idx + 1 == self.k_l0_tiles
                            matmul(
                                l0c_slot,
                                a_ready,
                                b_ready,
                                init=(global_k_l0_idx == 0),
                                bias=bias,
                                unit_flag=3 if is_final_acc else 2,
                                mx_mode=True,
                            )
                            if const_expr(self.split_n):
                                if is_final_acc:
                                    first_end = min(n_start + self.l0c_tile_n, n_end)
                                    mem_copy(
                                        out_gm[m_start:m_end, n_start:first_end],
                                        l0c_slot,
                                        options=copy_options(
                                            unit_flag=3,
                                            l2_cache_ctl=self.l2_cache_ctl_c,
                                        ),
                                    )
                                if current_n > self.l0c_tile_n:
                                    b_second_l1_tile = tile_slice(
                                        b_l1_rd,
                                        (self.l0c_tile_n, self.base_k),
                                        (1, k_l0_idx),
                                    )
                                    scale_b_second_l1_tile = tile_slice(
                                        scale_b_l1_rd,
                                        (self.scale_k_l0_len, self.l0c_tile_n),
                                        (scale_b_k_l0_idx, 1),
                                    )
                                    second_n = second_out.shape[1]
                                    b_second_ready = l0b.next_write(bare=True)
                                    b_second_ready_source = b_second_l1_tile[
                                        :second_n, :valid_k
                                    ]
                                    b_second_ready = b_second_ready.view(
                                        make_layout(b_second_ready_source.shape)
                                    )
                                    mem_copy(
                                        b_second_ready,
                                        b_second_ready_source,
                                        options=copy_options(
                                            mx_scale=scale_b_second_l1_tile[
                                                :valid_scale_k, :second_n
                                            ]
                                        ),
                                    )
                                    matmul(
                                        l0c_second_slot,
                                        a_ready,
                                        b_second_ready,
                                        init=(global_k_l0_idx == 0),
                                        unit_flag=3 if is_final_acc else 2,
                                        mx_mode=True,
                                    )
                                    if is_final_acc:
                                        second_start = n_start + self.l0c_tile_n
                                        mem_copy(
                                            out_gm[m_start:m_end, second_start:n_end],
                                            l0c_second_slot,
                                            options=copy_options(
                                                unit_flag=3,
                                                l2_cache_ctl=self.l2_cache_ctl_c,
                                            ),
                                        )

            # FixPipe: L0C→GM
            if const_expr(not self.split_n and not self.split_n_final):
                mem_copy(
                    out_tile,
                    l0c_slot,
                    options=copy_options(unit_flag=3, l2_cache_ctl=self.l2_cache_ctl_c),
                )

    # Device kernel entry. It maps output tiles to AIC cores, stages A/B/Scale
    # from GM to L1 and then L0A/L0B, performs K-sliced MX MMAD accumulation
    # in L0C, and writes the completed tile back to GM via FixPipe.
    @kernel
    def qbmm_kernel(
        self,
        out_gm: Tensor,
        a_gm: Tensor,
        b_gm: Tensor,
        scale_a_gm: Tensor,
        scale_b_gm: Tensor,
        bias_gm: Tensor,
    ):
        """Assign output tiles to AICs with AL1_FULL_LOAD or ASW snake order."""
        full_load = self.full_load
        transpose_a, transpose_b = self.transpose_a, self.transpose_b
        base_m, base_n, base_k = self.base_m, self.base_n, self.base_k
        k_l1, l1_buffers, l0c_buffers = self.k_l1, self.l1_buffers, self.l0c_buffers
        a_dtype, b_dtype = self.a_dtype, self.b_dtype

        # L1/L0 RingBuffers and copy engines — created inside kernel context and
        # kept as locals: SSA values must not ride the `self` receiver across
        # the @jit helper boundary (ST101), so they are passed as explicit
        # arguments to _compute_output_tile instead.
        a_k = ceil_align(self.k, MX_K_ALIGN) if full_load else k_l1
        a_depth = 1 if full_load else l1_buffers
        scale_a_l1_len = self.scale_k_len if full_load else self.scale_k_l1_len
        a_scale_depth = 1 if full_load else 2
        l1_a = RingBuffer(
            MemLoc.L1,
            a_dtype,
            make_layout((a_k, base_m) if transpose_a else (base_m, a_k)),
            depth=a_depth,
        )
        l1_b = RingBuffer(
            MemLoc.L1,
            b_dtype,
            make_layout((base_n, k_l1) if transpose_b else (k_l1, base_n)),
            depth=l1_buffers,
        )
        l1_scale_a = RingBuffer(
            MemLoc.L1,
            dtypes.float8_e8m0,
            make_layout((scale_a_l1_len, base_m)),
            depth=a_scale_depth,
        )
        l1_scale_b = RingBuffer(
            MemLoc.L1,
            dtypes.float8_e8m0,
            make_layout(
                (
                    self.scale_k_len if self.scale_b_full_load else self.scale_k_l1_len,
                    base_n,
                )
            ),
            depth=1 if self.scale_b_full_load else 2,
        )
        l0a = RingBuffer(
            MemLoc.L0A,
            a_dtype,
            make_layout((base_m, base_k)),
            depth=2,
        )
        l0b = RingBuffer(
            MemLoc.L0B,
            b_dtype,
            make_layout((self.l0c_tile_n, base_k)),
            depth=2,
        )
        l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((base_m, self.l0c_tile_n)),
            depth=l0c_buffers,
        )
        l1_bias = None
        l1_bt = None
        if const_expr(self.has_bias):
            l1_bias = RingBuffer(
                MemLoc.L1,
                dtypes.float32,
                make_layout((base_n,)),
                depth=2,
                data_format="nd",
            )
            l1_bt = RingBuffer(
                MemLoc.BIAS,
                dtypes.float32,
                make_layout((base_n,)),
                depth=2,
            )

        # Shared tile context (GM tensors, channels, engines): SSA values
        # must not ride `self` across the @jit boundary (ST101), so they are
        # bundled here and passed to _compute_output_tile explicitly.
        tile_ctx = (
            out_gm,
            a_gm,
            b_gm,
            scale_a_gm,
            scale_b_gm,
            l1_a,
            l1_b,
            l1_scale_a,
            l1_scale_b,
            l0a,
            l0b,
            l0c,
            l1_bias,
            l1_bt,
        )

        # Multi-core scheduling: full_load assigns fixed M tiles across cores;
        # normal mode uses ASW snake ordering with 4-row sliding window.
        block_idx = get_block_idx()

        if const_expr(self.full_load):
            # Keep A/ScaleA for one M tile resident and reuse them across N tiles.
            full_m_idx = block_idx % self.block_scheduler.m_tiles
            n_lane = block_idx // self.block_scheduler.m_tiles
            n_stride = AIC_NUM // self.block_scheduler.m_tiles
            a_k_len = ceil_align(self.k, MX_K_ALIGN)
            a_tile = tile_slice(
                a_gm,
                (a_k_len, self.base_m)
                if const_expr(self.transpose_a)
                else (self.base_m, a_k_len),
                (0, full_m_idx) if const_expr(self.transpose_a) else (full_m_idx, 0),
            )
            a_write = l1_a.next_write()
            mem_copy(
                a_write,
                a_tile,
                options=copy_options(l2_cache_ctl=self.l2_cache_ctl_a),
            )
            if const_expr(self.transpose_a and self.k % MX_K_ALIGN != 0):
                self._zero_transposed_k_tail(
                    a_write, self.k, a_k_len, self.base_m, self.a_dtype.c0_elems
                )
            scale_group_len = self.scale_k_len // MX_SCALE_PAIR
            scale_a_tile = tile_slice(
                scale_a_gm,
                (scale_group_len, self.base_m, MX_SCALE_PAIR)
                if const_expr(self.transpose_a)
                else (self.base_m, scale_group_len, MX_SCALE_PAIR),
                (0, full_m_idx, 0)
                if const_expr(self.transpose_a)
                else (full_m_idx, 0, 0),
            )
            mem_copy(
                l1_scale_a.next_write().permute(1, 0),
                scale_a_tile.permute((1, 0, 2)).flatten(-2, -1)
                if self.transpose_a
                else scale_a_tile.flatten(-2, -1),
                options=copy_options(l2_cache_ctl=1),
            )
            for full_n_idx in range(n_lane, self.block_scheduler.n_tiles, n_stride):
                self._compute_output_tile(
                    *tile_ctx,
                    full_m_idx,
                    full_n_idx,
                    bias_gm,
                )
        else:
            round_count = self.block_scheduler.get_round_count(block_idx)
            if const_expr(self.scale_b_full_load):
                _, scale_n_idx, _, _, _ = self.block_scheduler.get_tile_idx(
                    block_idx, 0, round_count
                )
                scale_group_len = self.scale_k_len // MX_SCALE_PAIR
                scale_b_tile = tile_slice(
                    scale_b_gm,
                    (base_n, scale_group_len, MX_SCALE_PAIR)
                    if const_expr(transpose_b)
                    else (scale_group_len, base_n, MX_SCALE_PAIR),
                    (scale_n_idx, 0, 0)
                    if const_expr(transpose_b)
                    else (0, scale_n_idx, 0),
                )
                mem_copy(
                    l1_scale_b.next_write(),
                    scale_b_tile.flatten(-2, -1).permute(1, 0)
                    if transpose_b
                    else scale_b_tile.permute((0, 2, 1)).flatten(0, 1),
                    options=copy_options(l2_cache_ctl=self.l2_cache_ctl_scale_b),
                )
            for round_idx in range(round_count):
                m_idx, n_idx, tail_split, sub_m_idx, sub_n_idx = (
                    self.block_scheduler.get_tile_idx(block_idx, round_idx, round_count)
                )
                self._compute_output_tile(
                    *tile_ctx,
                    m_idx,
                    n_idx,
                    bias_gm,
                    tail_split,
                    sub_m_idx,
                    sub_n_idx,
                )

    @host
    def run(
        self,
        out_gm: Tensor,
        a_gm: Tensor,
        b_gm: Tensor,
        a_scale_gm: Tensor,
        b_scale_gm: Tensor,
        bias_gm: Tensor,
    ):
        self.qbmm_kernel[self.used_core_num](
            out_gm,
            a_gm,
            b_gm,
            a_scale_gm,
            b_scale_gm,
            bias_gm,
        )


# ============================================================================
# 3. Torch Interface
# ============================================================================

_TORCH_FP8_INPUT_TO_DSL = {
    torch.float8_e4m3fn: dtypes.float8_e4m3fn,
    torch.float8_e5m2: dtypes.float8_e5m2,
}
_TORCH_FP4_INPUT = torch.float4_e2m1fn_x2
_TORCH_OUTPUT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_TORCH_OUTPUT_TO_DSL = {
    torch.float32: dtypes.float32,
    torch.float16: dtypes.float16,
    torch.bfloat16: dtypes.bfloat16,
}


def _infer_qbmm_mx_layout(a, b, scale_a, scale_b, is_mxfp4=False):
    """Infer M/N/K and transpose flags from X1/X2/Scale view shapes.

    Do not rewrite the input tensors. Return
    ``(M, N, K, transposeX1, transposeX2)``. MXFP4 stores two logical
    elements in each byte of the physical innermost axis.
    """
    if scale_a.stride(-1) != 1 or scale_b.stride(-1) != 1:
        raise ValueError("scale paired lane must have unit stride")

    matches = []
    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)
    scale_a_shape = tuple(scale_a.shape)
    scale_b_shape = tuple(scale_b.shape)
    packed = 2 if is_mxfp4 else 1  # fp4x2 stores two nibbles per byte
    for transpose_a in (False, True):
        m = a_shape[1] * packed if transpose_a else a_shape[0]
        k = a_shape[0] if transpose_a else a_shape[1] * packed
        group_count = ceil_div(k, MX_K_ALIGN)
        expected_scale_a = (
            (group_count, m, MX_SCALE_PAIR)
            if transpose_a
            else (m, group_count, MX_SCALE_PAIR)
        )
        if scale_a_shape != expected_scale_a:
            continue
        for transpose_b in (False, True):
            # Packing follows the innermost dimension: transposeB packs the
            # (N, K) view along K; !transposeB packs the (K, N) view along
            # N, so K stays logical and N doubles.
            if transpose_b:
                n = b_shape[0]
                b_k = b_shape[1] * packed
            else:
                b_k = b_shape[0]
                n = b_shape[1] * packed
            if b_k != k:
                continue
            expected_scale_b = (
                (n, group_count, MX_SCALE_PAIR)
                if transpose_b
                else (group_count, n, MX_SCALE_PAIR)
            )
            if scale_b_shape == expected_scale_b:
                matches.append((m, n, k, transpose_a, transpose_b))

    if len(matches) != 1:
        if not matches:
            raise ValueError(
                "cannot infer MX layout from view shape: "
                f"a={a_shape}, b={b_shape}, "
                f"pertoken_scale={scale_a_shape}, scale={scale_b_shape}"
            )
        raise ValueError(
            "ambiguous MX view shape; please use non-ambiguous "
            f"a/b/scale shapes, candidates={matches}"
        )
    m, n, k, transpose_a, transpose_b = matches[0]
    if is_mxfp4:
        # Sub-byte packing follows the innermost dimension, so the packed
        # extent must be even: M for transposeA, N for !transposeB, K when
        # either operand packs along K.
        if transpose_a and m % 2 != 0:
            raise ValueError(
                f"transposeA=True packs 2 fp4 elements per byte along M; M={m} must be even"
            )
        if not transpose_b and n % 2 != 0:
            raise ValueError(
                f"transposeB=False packs 2 fp4 elements per byte along N; N={n} must be even"
            )
        if (not transpose_a or transpose_b) and k % 2 != 0:
            raise ValueError(
                f"K={k} must be even when either operand is packed along K"
            )
    return m, n, k, transpose_a, transpose_b


# AOT compilation is expensive (tiling + kernel build + run.compile), so the
# builder is memoized per (shape config, dtypes, profile) with lru_cache.
@functools.cache
def _build_mx_aot_callable(
    m,
    n,
    k,
    has_bias,
    output_dtype,
    transpose_a,
    transpose_b,
    a_dtype,
    b_dtype,
    profile,
):
    tiling = get_qbmm_mx_tile_config(
        m,
        n,
        k,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        has_bias=has_bias,
        output_dtype=output_dtype,
        profile=profile,
    )
    kernel_obj = QbmmMxKernel(
        m,
        n,
        k,
        tiling,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        has_bias=has_bias,
    )
    group_count = ceil_div(k, MX_K_ALIGN)
    a_shape = (k, m) if transpose_a else (m, k)
    b_shape = (k, n) if not transpose_b else (n, k)
    scale_a_shape = (
        (group_count, m, MX_SCALE_PAIR)
        if transpose_a
        else (m, group_count, MX_SCALE_PAIR)
    )
    scale_b_shape = (
        (n, group_count, MX_SCALE_PAIR)
        if transpose_b
        else (group_count, n, MX_SCALE_PAIR)
    )
    specs = (
        TensorSpec(shape=(m, n), dtype=_TORCH_OUTPUT_TO_DSL[output_dtype]),
        TensorSpec(shape=a_shape, dtype=a_dtype),
        TensorSpec(shape=b_shape, dtype=b_dtype),
        TensorSpec(shape=scale_a_shape, dtype=dtypes.float8_e8m0),
        TensorSpec(shape=scale_b_shape, dtype=dtypes.float8_e8m0),
        TensorSpec(shape=(n if has_bias else 0,), dtype=dtypes.float32),
    )
    return cannbotdsl.compile(kernel_obj.run, *specs)


def _validate_mx_quant_matmul_input(a, b, scale_a, scale_b, bias, output_dtype):
    """Validate npu_quant_matmul inputs; return (is_mxfp4, has_bias)."""
    is_mxfp4 = a.dtype == _TORCH_FP4_INPUT and b.dtype == _TORCH_FP4_INPUT
    if a.dtype not in _TORCH_FP8_INPUT_TO_DSL and not is_mxfp4:
        raise TypeError(
            f"a/b must be FP8 or packed FP4 tensors, got {a.dtype} and {b.dtype}"
        )
    if is_mxfp4 != (b.dtype == _TORCH_FP4_INPUT):
        raise TypeError("a and b must use the same MX data dtype")
    has_bias = bias is not None
    if output_dtype not in _TORCH_OUTPUT_DTYPES:
        raise ValueError(
            f"MX quant matmul requires a supported output_dtype, got {output_dtype}"
        )
    if a.dim() != 2 or b.dim() != 2:
        raise NotImplementedError(
            f"MX batch dims not implemented, got rank {a.dim()} and {b.dim()}"
        )
    if scale_a.dim() != 3 or scale_b.dim() != 3:
        raise ValueError("MX scales must use rank-3 paired layouts")
    if any(t.device != a.device for t in (a, b, scale_a, scale_b)):
        raise ValueError("a, b and both MX scales must be on the same device")
    if scale_a.dtype != torch.float8_e8m0fnu or scale_b.dtype != torch.float8_e8m0fnu:
        raise TypeError("scale_a and scale_b must be FLOAT8_E8M0 tensors")
    if has_bias and bias.dtype != torch.float32:
        raise TypeError(f"bias must be float32, got {bias.dtype}")
    return is_mxfp4, has_bias


def _prepare_bias_arg(bias, n, device):
    """Normalize bias for the kernel call: validate the length, or return a
    length-0 placeholder when no bias is used (six-tensor ABI contract)."""
    if bias is None:
        return torch.empty(0, dtype=torch.float32, device=device)
    if bias.shape[0] != n:
        raise ValueError(f"bias length {bias.shape[0]} != N={n}")
    return bias


def npu_quant_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    bias: torch.Tensor = None,
    output_dtype=None,
) -> torch.Tensor:
    """Execute a two-dimensional, fully quantized MX matrix multiply.

    Supports both MXFP8 (``float8_e4m3fn``/``float8_e5m2`` data) and packed
    MXFP4 (``float4_e2m1fn_x2`` data) with E8M0 block scales: ``scale_a``
    is the paired ScaleA of ``a`` and ``scale_b`` is the paired ScaleB of
    ``b``.
    """
    is_mxfp4, has_bias = _validate_mx_quant_matmul_input(
        a,
        b,
        scale_a,
        scale_b,
        bias,
        output_dtype,
    )

    m, n, k, transpose_a, transpose_b = _infer_qbmm_mx_layout(
        a, b, scale_a, scale_b, is_mxfp4=is_mxfp4
    )
    bias = _prepare_bias_arg(bias, n, a.device)
    output = torch.empty((m, n), dtype=output_dtype, device=a.device)

    if is_mxfp4:
        a_dtype = b_dtype = dtypes.fp4x2_e2m1
        profile = FP4_TILING_PROFILE
        a_input, b_input = a.view(torch.int8), b.view(torch.int8)
    else:
        a_dtype = _TORCH_FP8_INPUT_TO_DSL[a.dtype]
        b_dtype = _TORCH_FP8_INPUT_TO_DSL[b.dtype]
        profile = FP8_TILING_PROFILE
        a_input, b_input = a, b
    compiled_kernel = _build_mx_aot_callable(
        m,
        n,
        k,
        has_bias,
        output_dtype,
        transpose_a,
        transpose_b,
        a_dtype,
        b_dtype,
        profile,
    )
    compiled_kernel(output, a_input, b_input, scale_a, scale_b, bias)
    return output
