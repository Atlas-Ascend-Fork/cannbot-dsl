# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Ascend950 TT (per-tensor) quantized matmul ``npu_quant_matmul`` sample.

Structure:
  1. Host-side tiling (``QbmmTtTiling``)
  2. Device kernel (``QbmmTtKernel``): GM->L1->L0->L0C->FIXPIPE deq_scale->GM
  3. Torch-facing ``npu_quant_matmul`` wrapper

Formula:
  C[M,N] = (scaleA * scaleB) * (A[M,K] @ B[K,N])

``TT`` quantization uses one scalar scale per tensor for *both* operands:
``scale_a`` and ``scale_b`` are the per-tensor scales for A and B. Both scales
are runtime GM inputs. Each AIC loads them, computes
``scaleA * scaleB``, and passes the result to FIXPIPE; no per-group/per-channel
scale staging is needed.

Supported quant paths (mainline-consistent: the combined scaleA*scaleB is
loaded from GM at runtime and passed to FIXPIPE as ``deq_scale``, mirroring
how QuantBatchMatmulV3 passes ``deqScalar`` to AscendC Fixpipe):
  * ``INT8``: int8 operands -> int32 L0C -> scalar FIXPIPE dequant -> fp16/bf16.
  * ``FP8`` (e4m3/e5m2/hifloat8): float8 operands -> fp32 L0C -> QF322*_PRE
    FIXPIPE dequant -> fp16/bf16/fp32 output. For microscaling FP8 use
    ``quant_batch_matmul_mx``.
"""

__all__ = ["npu_quant_matmul"]

import math

import torch
import torch_npu

from cannbotdsl import make_layout, dtypes, get_mem_size, get_platform_info
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.matmul import matmul
from cannbotdsl import MemLoc, Tensor
from cannbotdsl.tensor import tile_slice
from cannbotdsl.ops.memcpy import copy_options, mem_copy, convert


# ============================================================================
# 1. Host-side Tiling
# ============================================================================


def ceil_div(a, b):
    if b == 0:
        return a
    return (a + b - 1) // b


def ceil_align(a, b):
    if b == 0:
        return a
    return (a + b - 1) // b * b


def floor_align(a, b):
    if b == 0:
        return a
    return a // b * b


# L2 cache mode
L2_CACHE_DEFAULT = 0
A_L2_CACHE_DISABLE = 1
B_L2_CACHE_DISABLE = 2
ALL_L2_CACHE_DISABLE = 3


class QbmmTtTiling:
    """Host-side tiling for TT quantized matmul.

    The plan mirrors the non-quantized matmul cost model, but separates the
    one-byte input element size from the four-byte L0C accumulator size.

    On-chip memory capacities and the AIC count are queried from the current
    platform instead of assuming a fixed Ascend 950 configuration.
    """

    L1_SIZE = get_mem_size("l1")
    L0A_SIZE = get_mem_size("l0a")
    L0C_SIZE = get_mem_size("l0c")
    L2_SIZE = 128 * 1024 * 1024
    AIC_NUM = get_platform_info().cube_core_num
    L1_ALIGN_SIZE = 32  # zn-layout L1 fractal alignment (outer dim)

    BASIC_BLOCK_16 = 16
    BASIC_BLOCK_64 = 64
    BASIC_BLOCK_128 = 128
    BASIC_BLOCK_256 = 256
    DB_SIZE = 2
    BASIC_L1_BUFFER_NUM = 4
    INPUT_ELEMENT_BYTES = 1
    L0C_ELEMENT_BYTES = 4
    BASIC_BLOCK_K_128B = 128
    BASIC_BLOCK_K_256B = 256
    BASIC_BLOCK_K_512B = 512
    L1_SINGLE_SIZE_LIMIT = 48 * 1024
    MIN_TAIL_BLOCK_SIZE = 4096
    CUBE_BOUND_RATIO = 0.85
    BALANCE_RATE_EDGE = 0.9
    EPSILON = 1e-9
    MMAD_UNITS_PER_CORE = 8
    MAX_STEP_K = 8
    DEFAULT_CORE_FREQ = 1.65
    DEFAULT_DDR_RATE = 31
    DEFAULT_L2_RATE = 100
    ALIGN_128 = 128

    def __init__(
        self,
        m,
        n,
        k,
        *,
        a_dtype,
        b_dtype,
        out_dtype_size,
        transpose_a=False,
        transpose_b=True,
    ):
        """Build the complete host tiling plan from the matrix metadata."""
        self.m = m
        self.n = n
        self.k = k
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.out_dtype_size = out_dtype_size
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self._compute()

    def _compute(self):
        """Run base-block, K-window, buffering, core, and cache selection."""
        self._reset_base()
        self._rebalance_block()
        # zn-layout fractal constraint (mirrors quant_batch_matmul_mx): K-major operands
        # need their outer dim 32-aligned in L1, so transpose_a forces base_m
        # (and transpose_b=False forces base_n) up to a 32 multiple.
        if self.transpose_a:
            self.base_m = ceil_align(self.base_m, self.L1_ALIGN_SIZE)
        if not self.transpose_b:
            self.base_n = ceil_align(self.base_n, self.L1_ALIGN_SIZE)
        # L0C ping-pong: when one tile would fill L0C (single buffer), halve
        # base_n so two slots fit. The depth-2 L0C channel then overlaps the
        # FixPipe drain of tile i with the MMAD chain of tile i+1. Narrower
        # tiles double the A re-reads per N column and the per-tile
        # format-conversion count, which measurably hurts transposed layouts
        # and K-tail shapes, so the halving is limited to aligned F x F cases
        # where the FixPipe overlap dominates.
        is_ff_layout = not self.transpose_a and not self.transpose_b
        if (
            self.base_m * self.base_n * self.L0C_ELEMENT_BYTES * self.DB_SIZE
            > self.L0C_SIZE
            and is_ff_layout
            and self.k % self.base_k == 0
        ):
            self.base_n = max(
                self.L1_ALIGN_SIZE,
                min(
                    self.BASIC_BLOCK_256,
                    self.L0C_SIZE
                    // (self.DB_SIZE * self.L0C_ELEMENT_BYTES * self.base_m)
                    // self.L1_ALIGN_SIZE
                    * self.L1_ALIGN_SIZE,
                ),
            )
        self._cal_l1_tiling()
        self._finalize()

    def _reset_base(self):
        """Start from large aligned Cube blocks to maximize data reuse per AIC."""
        self.base_m = min(self.BASIC_BLOCK_256, ceil_align(self.m, self.BASIC_BLOCK_16))
        self.base_n = min(self.BASIC_BLOCK_256, ceil_align(self.n, self.BASIC_BLOCK_16))
        self.base_k = self.BASIC_BLOCK_K_128B // self.INPUT_ELEMENT_BYTES

    def _rebalance_block(self):
        """Balance M/N work across AICs, then maximize the legal Cube K block.

        Better M/N tail occupancy reduces idle AICs; a larger base_k reduces the
        number of L1-to-L0 copies and MMAD iterations.
        """
        base_m_align_unit, base_n_align_unit, max_base_m, max_base_n = (
            self._cal_cube_bound_edge()
        )
        self._search_optimal_base_mn(
            base_m_align_unit,
            base_n_align_unit,
            max_base_m,
            max_base_n,
        )
        self.base_m = min(ceil_align(self.m, self.BASIC_BLOCK_16), self.base_m)
        self.base_n = min(ceil_align(self.n, self.BASIC_BLOCK_16), self.base_n)
        self._get_base_k()

    def _cal_cube_bound_edge(self):
        """Choose M/N alignment and search limits from the bandwidth model.

        Memory-bound shapes use wider aligned blocks to reduce data movement;
        Cube-bound shapes allow smaller blocks to expose more AIC parallelism.
        """
        hbm_bw = self._get_hbm_bw()
        l2_bw = self._get_l2_bw()
        single_core_compute_power = self.DEFAULT_CORE_FREQ * self.MMAD_UNITS_PER_CORE
        compute_power = single_core_compute_power * self.AIC_NUM
        l2_cache_usage = max(
            (self.m + self.n) * self.k * self.INPUT_ELEMENT_BYTES / self.L2_SIZE, 1.0
        )
        cmr = (self.m + self.n) / (self.m * self.n)
        self.cube_bound_edge = (
            (l2_bw / compute_power)
            + l2_cache_usage * (1 - l2_bw / hbm_bw) * cmr
            - (1 + l2_bw / hbm_bw) / self.k
        )
        base_m_best = min(ceil_align(self.m, self.BASIC_BLOCK_16), self.BASIC_BLOCK_256)
        base_n_best = max(
            self.BASIC_BLOCK_16,
            min(
                ceil_align(self.n, self.BASIC_BLOCK_16),
                floor_align(
                    self.L0C_SIZE // self.L0C_ELEMENT_BYTES // base_m_best,
                    self.BASIC_BLOCK_16,
                ),
            ),
        )
        cube_bound_param_best = (1.0 / base_m_best) + (1.0 / base_n_best)
        is_memory_bound = cube_bound_param_best > self.cube_bound_edge
        inner_align_unit = (
            self.BASIC_BLOCK_128 if is_memory_bound else self.BASIC_BLOCK_64
        )
        fixp_bound_edge = (self.m * self.n * hbm_bw) / ((self.m + self.n) * l2_bw)
        base_m_align_unit = (
            inner_align_unit // self.INPUT_ELEMENT_BYTES
            if self.transpose_a
            else self.BASIC_BLOCK_16
        )
        base_n_align_unit = (
            self.BASIC_BLOCK_K_256B // self.INPUT_ELEMENT_BYTES
            if self.k < fixp_bound_edge
            else (
                self.BASIC_BLOCK_16
                if self.transpose_b
                else inner_align_unit // self.INPUT_ELEMENT_BYTES
            )
        )
        max_base_m = self._get_max_base_with_limit(
            base_m_align_unit,
            is_right_matrix=False,
            is_memory_bound=is_memory_bound,
        )
        max_base_n = self._get_max_base_with_limit(
            base_n_align_unit,
            is_right_matrix=True,
            is_memory_bound=is_memory_bound,
        )
        self.base_m = max(self.BASIC_BLOCK_16, min(max_base_m, self.BASIC_BLOCK_256))
        self.base_n = max(
            self.BASIC_BLOCK_16,
            min(
                max_base_n,
                floor_align(
                    self.L0C_SIZE // self.L0C_ELEMENT_BYTES // self.base_m,
                    base_n_align_unit,
                ),
            ),
        )
        self.cube_bound_param = (1.0 / self.base_m) + (1.0 / self.base_n)
        self.cube_bound_edge *= self.CUBE_BOUND_RATIO
        return base_m_align_unit, base_n_align_unit, max_base_m, max_base_n

    def _search_optimal_base_mn(
        self,
        base_m_align_unit,
        base_n_align_unit,
        max_base_m,
        max_base_n,
    ):
        """Select aligned base_m/base_n without trading reuse for idle-tail AICs.

        The search improves final-round AIC occupancy while rejecting smaller
        blocks whose extra A/B traffic would move the shape past Cube bound.
        """
        balance_rate = self._get_balance_rate_with_tail(self.base_m, self.base_n)
        cur_base_m = max_base_m
        while cur_base_m >= 1:
            cur_max_base_n = min(
                max_base_n,
                floor_align(
                    self.L0C_SIZE // self.L0C_ELEMENT_BYTES // cur_base_m,
                    base_n_align_unit,
                ),
            )
            cur_base_n = cur_max_base_n
            while cur_base_n >= 1:
                cur_cube_bound_param = (1.0 / cur_base_m) + (1.0 / cur_base_n)
                cur_balance_rate = self._get_balance_rate_with_tail(
                    cur_base_m, cur_base_n
                )
                skip_cond = (
                    balance_rate >= self.BALANCE_RATE_EDGE
                    and cur_cube_bound_param > self.cube_bound_param
                    and cur_cube_bound_param > self.cube_bound_edge
                    and self.cube_bound_edge > 0
                )
                if not skip_cond:
                    cube_bound_cond = (
                        cur_cube_bound_param <= self.cube_bound_edge
                        and cur_balance_rate > balance_rate
                    )
                    balance_cond = (cur_cube_bound_param / cur_balance_rate) < (
                        self.cube_bound_param / balance_rate
                    ) or (
                        abs(
                            cur_cube_bound_param / cur_balance_rate
                            - self.cube_bound_param / balance_rate
                        )
                        < self.EPSILON
                        and cur_balance_rate > balance_rate
                    )
                    if cube_bound_cond or balance_cond:
                        if cube_bound_cond:
                            self.cube_bound_edge = cur_cube_bound_param
                        self.base_m = cur_base_m
                        self.base_n = cur_base_n
                        self.cube_bound_param = cur_cube_bound_param
                        balance_rate = cur_balance_rate
                cur_base_n -= base_n_align_unit
            cur_base_m -= base_m_align_unit

    def _get_base_k(self):
        """Use base_k=128 when alignment and L0 capacity allow, otherwise 64.

        The larger block halves K-loop copy/MMAD iterations. Arbitrary K stays
        legal because the final GM-to-L1 tile carries actual K and is padded.
        """
        max_base_k = (
            self.L0A_SIZE
            // self.DB_SIZE
            // self.INPUT_ELEMENT_BYTES
            // max(self.base_m, self.base_n)
        )
        if self.k % 128 == 0 and max_base_k >= 128:
            self.base_k = 128
        else:
            self.base_k = 64

    def _cal_l1_tiling(self):
        """Pack multiple base_k blocks into the largest divisor-based L1 window.

        A larger k_l1 amortizes GM-to-L1 copy and synchronization overhead while
        retaining double-buffer capacity; non-divisible K uses an actual-K tail.
        """
        residual = ceil_div(self.k, self.base_k)
        max_step_k = min(residual, self.MAX_STEP_K)
        res_step = 1
        for step_k in range(1, max_step_k + 1):
            cur_kl1 = self.base_k * step_k
            a_l1 = self.base_m * cur_kl1 * self.INPUT_ELEMENT_BYTES
            b_l1 = self.base_n * cur_kl1 * self.INPUT_ELEMENT_BYTES
            if (a_l1 + b_l1) * self.DB_SIZE > self.L1_SIZE:
                break
            if max(a_l1, b_l1) * self.DB_SIZE * 2 > self.L1_SIZE:
                break
            if self.k % cur_kl1 == 0:
                res_step = step_k
        self.step_k = res_step

    def _set_disable_l2cache(self):
        """Bypass L2 for aligned one-pass operands that have no cross-tile reuse.

        This avoids cache pollution while keeping reusable A/B data cached.
        """
        inner_a = self.m if self.transpose_a else self.k
        inner_b = self.k if self.transpose_b else self.n
        k_l1_aligned = self.k_l1 * self.INPUT_ELEMENT_BYTES % self.ALIGN_128 == 0
        total_size = (
            self.m * self.n * self.out_dtype_size
            + (self.m * self.k + self.k * self.n) * self.INPUT_ELEMENT_BYTES
        )
        if total_size < self.L2_SIZE:
            return L2_CACHE_DEFAULT
        m_cnt = ceil_div(self.m, self.base_m)
        n_cnt = ceil_div(self.n, self.base_n)
        left_not_l2_cache = (
            self.base_n >= self.n
            and n_cnt <= 1
            and inner_a * self.INPUT_ELEMENT_BYTES % self.ALIGN_128 == 0
            and k_l1_aligned
        )
        right_not_l2_cache = (
            self.base_m >= self.m
            and m_cnt <= 1
            and inner_b * self.INPUT_ELEMENT_BYTES % self.ALIGN_128 == 0
            and k_l1_aligned
        )
        if left_not_l2_cache and right_not_l2_cache:
            return ALL_L2_CACHE_DISABLE
        if left_not_l2_cache:
            return A_L2_CACHE_DISABLE
        if right_not_l2_cache:
            return B_L2_CACHE_DISABLE
        return L2_CACHE_DEFAULT

    def _finalize(self):
        """Finalize pipeline depth, active AIC count, and L2 policy.

        Four L1 buffers overlap GM copies with Cube consumption when capacity
        permits; L0C double buffering overlaps MMAD output with FixPipe.
        """
        self.m_l1 = min(ceil_align(self.m, self.BASIC_BLOCK_16), self.base_m)
        self.n_l1 = min(ceil_align(self.n, self.BASIC_BLOCK_16), self.base_n)
        self.step_k = min(self.step_k, self.BASIC_L1_BUFFER_NUM)
        self.k_l1 = self.base_k * self.step_k
        m_core = ceil_div(self.m, self.base_m)
        n_core = ceil_div(self.n, self.base_n)
        self.used_core_num = min(m_core * n_core, self.AIC_NUM)
        a_l1_4buf = (
            self.k_l1
            * self.base_m
            * self.INPUT_ELEMENT_BYTES
            * self.BASIC_L1_BUFFER_NUM
        )
        b_l1_4buf = (
            self.k_l1
            * self.base_n
            * self.INPUT_ELEMENT_BYTES
            * self.BASIC_L1_BUFFER_NUM
        )
        # Keep headroom for on-chip slot-allocation alignment overhead (a
        # full L1 budget overflows once slot alignment is applied): == falls
        # back to 2 too.
        self.l1_buffer_num = (
            self.DB_SIZE
            if (a_l1_4buf + b_l1_4buf) >= self.L1_SIZE - 32 * 1024
            else self.BASIC_L1_BUFFER_NUM
        )
        self.l0c_db = (
            self.DB_SIZE
            if self.base_m * self.base_n * self.L0C_ELEMENT_BYTES * self.DB_SIZE
            <= self.L0C_SIZE
            else 1
        )
        self.l2_cache_disable = self._set_disable_l2cache()

    def _get_max_base_with_limit(
        self,
        base_align_unit,
        is_right_matrix,
        is_memory_bound,
    ):
        """Maximize one M/N block within L0 accumulator and L1 K-slice limits.

        The limit prevents buffer overflow while retaining the largest legal
        block for A/B reuse and fewer output tiles.
        """
        shape_value = self.n if is_right_matrix else self.m
        k_align_value = ceil_align(self.k, self.BASIC_BLOCK_16)
        k_limit_value = (
            self.BASIC_BLOCK_16
            if is_memory_bound
            else self.BASIC_BLOCK_K_128B // self.INPUT_ELEMENT_BYTES
        )
        min_kl0 = min(k_limit_value, k_align_value) * self.INPUT_ELEMENT_BYTES
        max_base_mn_with_buffer = (
            self.L0C_SIZE // self.L0C_ELEMENT_BYTES // self.BASIC_BLOCK_16
        )
        max_base_block = min(
            self.L0A_SIZE // self.DB_SIZE // min_kl0,
            max_base_mn_with_buffer,
        )
        k_align_unit = (
            (self.BASIC_BLOCK_K_256B if is_memory_bound else self.BASIC_BLOCK_K_512B)
            // self.INPUT_ELEMENT_BYTES
            if (not self.transpose_a or self.transpose_b)
            else self.BASIC_BLOCK_16
        )
        max_base_mn_with_k_inner = self.L1_SIZE // (
            2
            * self.DB_SIZE
            * self.INPUT_ELEMENT_BYTES
            * min(k_align_unit, k_align_value)
        )
        max_base_block = min(max_base_block, max_base_mn_with_k_inner)
        max_base_block = min(
            ceil_align(shape_value, base_align_unit),
            floor_align(max_base_block, base_align_unit),
        )
        if shape_value < base_align_unit:
            max_base_block = min(
                max_base_block, ceil_align(shape_value, self.BASIC_BLOCK_16)
            )
        return max_base_block

    def _get_balance_rate_with_tail(self, base_m, base_n):
        """Score effective AIC work in full rounds and the final partial round.

        Candidate blocks with fewer idle tail AICs receive a higher score.
        """
        total_round = ceil_div(self.m, base_m) * ceil_div(self.n, base_n)
        main_round = ceil_div(total_round, self.AIC_NUM) - 1
        tail_round_blocks = total_round - self.AIC_NUM * main_round
        total_tail_split = self.AIC_NUM // tail_round_blocks if tail_round_blocks else 1
        eff_base_m = self.m if self.m <= self.BASIC_BLOCK_16 else base_m
        eff_base_n = self.n if self.n <= self.BASIC_BLOCK_16 else base_n
        if (
            main_round == 0
            or (eff_base_m * eff_base_n) // total_tail_split < self.MIN_TAIL_BLOCK_SIZE
        ):
            return (self.m * self.n / self.AIC_NUM) / (
                (main_round + 1) * eff_base_m * eff_base_n
            )
        tail_split_sqrt = int(math.sqrt(total_tail_split))
        offset = (
            total_tail_split - tail_split_sqrt * tail_split_sqrt
        ) // tail_split_sqrt + 1
        tail_round = 1.0 / (tail_split_sqrt * (tail_split_sqrt + offset - 1))
        return (self.m * self.n / self.AIC_NUM) / (
            (main_round + tail_round) * eff_base_m * eff_base_n
        )

    def _get_hbm_bw(self):
        """Return the aggregate HBM bandwidth used by the tiling cost model."""
        return self.DEFAULT_CORE_FREQ * self.AIC_NUM * self.DEFAULT_DDR_RATE / 1024

    def _get_l2_bw(self):
        """Return the aggregate L2 bandwidth used by the tiling cost model."""
        return self.DEFAULT_CORE_FREQ * self.AIC_NUM * self.DEFAULT_L2_RATE / 1024


# ============================================================================
# 2. Kernel
# ============================================================================

WINDOW_LEN = 4


class QbmmTtKernel:
    """TT quantized matmul with slide-window scheduling and an L1/L0 pipeline.

    A/B GM -> L1 -> L0A/L0B -> MMAD -> L0C -> FIXPIPE(deqScale) -> C GM.
    """

    def __init__(self, tiling: QbmmTtTiling, *, out_dtype):
        self.tiling = tiling
        self.out_dtype = out_dtype
        self.m_tiles = ceil_div(tiling.m, tiling.m_l1)
        self.n_tiles = ceil_div(tiling.n, tiling.n_l1)
        self.k_l1_tiles = ceil_div(tiling.k, tiling.k_l1)
        self.k_l0_per_l1 = ceil_div(tiling.k_l1, tiling.base_k)
        self.main_window = min(WINDOW_LEN, self.m_tiles)
        self.main_row = self.m_tiles // self.main_window - 1
        self.tail_window = self.m_tiles - self.main_row * self.main_window
        self.total_tiles = self.m_tiles * self.n_tiles
        disable_a = tiling.l2_cache_disable in (
            A_L2_CACHE_DISABLE,
            ALL_L2_CACHE_DISABLE,
        )
        disable_b = tiling.l2_cache_disable in (
            B_L2_CACHE_DISABLE,
            ALL_L2_CACHE_DISABLE,
        )
        self.l2_cache_ctl_a = 0 if disable_a else 1
        self.l2_cache_ctl_b = 0 if disable_b else 1

    @kernel
    def qbmm_kernel(
        self,
        gm_a: Tensor,
        gm_b: Tensor,
        gm_scale_b: Tensor,
        gm_scale_a: Tensor,
        gm_c: Tensor,
    ):
        tiling = self.tiling
        deq_scale = gm_scale_a[0] * gm_scale_b[0]
        # Transpose-aware L1/L0 layouts (mirrors quant_batch_matmul_mx):
        #   A (M,K) ND        -> L1 (base_m, k_l1) "nz"  (nd2nz engine)
        #   A (K,M) transposed -> L1 (k_l1, base_m) "zn" (dn2nz engine)
        #   B (N,K) ND        -> L1 (base_n, k_l1) "nz"  (nd2nz engine)
        #   B (K,N) transposed -> L1 (k_l1, base_n) "zn" (dn2nz engine)
        # L0A is always (base_m, base_k) nz; L0B is always (base_n, base_k) nz —
        # the DSL matmul computes lhs @ rhs^T with rhs in (N, K).
        l1_a = RingBuffer(
            MemLoc.L1,
            tiling.a_dtype,
            make_layout((tiling.base_m, tiling.k_l1)),
            depth=tiling.l1_buffer_num,
        )
        l1_b = RingBuffer(
            MemLoc.L1,
            tiling.b_dtype,
            make_layout((tiling.base_n, tiling.k_l1)),
            depth=tiling.l1_buffer_num,
        )
        l0a = RingBuffer(
            MemLoc.L0A,
            tiling.a_dtype,
            make_layout((tiling.base_m, tiling.base_k)),
            depth=2,
        )
        l0b = RingBuffer(
            MemLoc.L0B,
            tiling.b_dtype,
            make_layout((tiling.base_n, tiling.base_k)),
            depth=2,
        )
        # L0C is the cube accumulator (int32 for int8, fp32 for fp8/hif8); the
        # FIXPIPE engine dequants it to the float output dtype using the
        # runtime scalar deq_scale.
        l0c_dtype = dtypes.int32 if tiling.a_dtype == dtypes.int8 else dtypes.float32
        l0c = RingBuffer(
            MemLoc.L0C,
            l0c_dtype,
            make_layout((tiling.base_m, tiling.base_n)),
            depth=tiling.l0c_db,
        )
        # GM->L1 engines: nd2nz for ND operands, dn2nz for K-major operands.

        n_tiles = self.n_tiles
        main_window = self.main_window
        main_row = self.main_row
        tail_window = self.tail_window
        total_tiles = self.total_tiles
        k_l1_tiles = self.k_l1_tiles
        k_l0_per_l1 = self.k_l0_per_l1
        total_k_l0 = ceil_div(tiling.k, tiling.base_k)

        block_idx = get_block_idx()
        block_num = get_block_num()

        for tile_idx in range(block_idx, total_tiles, block_num):
            m_idx = 0
            n_idx = 0
            row_idx = tile_idx // n_tiles // main_window
            if row_idx < main_row:
                m_idx = row_idx * main_window + tile_idx % main_window
                n_idx = (tile_idx // main_window) % n_tiles
            else:
                row_idx = main_row
                tail_index = tile_idx - main_row * main_window * n_tiles
                m_idx = main_row * main_window + tail_index % tail_window
                n_idx = (tail_index // tail_window) % n_tiles
            n_idx = (n_tiles - 1 - n_idx) if (row_idx % 2 != 0) else n_idx

            m_start = m_idx * tiling.base_m
            n_start = n_idx * tiling.base_n
            m_count = min(tiling.base_m, tiling.m - m_start)
            n_count = min(tiling.base_n, tiling.n - n_start)
            gm_c_tile = gm_c[m_start : m_start + m_count, n_start : n_start + n_count]
            if const_expr(tiling.transpose_a):
                a_panel = gm_a[None, m_start : m_start + m_count]
            else:
                a_panel = gm_a[m_start : m_start + m_count, None]
            if const_expr(tiling.transpose_b):
                b_panel = gm_b[n_start : n_start + n_count, None]
            else:
                b_panel = gm_b[None, n_start : n_start + n_count]
            l0c_acc = (
                l0c.next_write()
                .bare()
                .view(make_layout(gm_c_tile.shape), dtype=l0c_dtype, data_format="nz")
            )
            for k_l1_idx in range(k_l1_tiles):
                k_start = k_l1_idx * tiling.k_l1
                k_count = (
                    tiling.k_l1
                    if const_expr(tiling.k % tiling.k_l1 == 0)
                    else min(tiling.k_l1, tiling.k - k_start)
                )
                k_end = k_start + k_count
                current_step_k = ceil_div(k_count, tiling.base_k)
                if const_expr(tiling.transpose_a):
                    gm_a_tile = a_panel[k_start:k_end, None]
                else:
                    gm_a_tile = a_panel[None, k_start:k_end]
                if const_expr(tiling.transpose_b):
                    gm_b_tile = b_panel[None, k_start:k_end]
                else:
                    gm_b_tile = b_panel[k_start:k_end, None]
                a_slot = l1_a.next_write()
                b_slot = l1_b.next_write()
                l1_a_tensor = a_slot.permute(1, 0) if tiling.transpose_a else a_slot
                l1_a_tensor = l1_a_tensor[tuple(map(slice, gm_a_tile.shape))]
                mem_copy(
                    l1_a_tensor,
                    gm_a_tile,
                    options=copy_options(l2_cache_ctl=self.l2_cache_ctl_a),
                )
                l1_b_tensor = b_slot if tiling.transpose_b else b_slot.permute(1, 0)
                l1_b_tensor = l1_b_tensor[tuple(map(slice, gm_b_tile.shape))]
                mem_copy(
                    l1_b_tensor,
                    gm_b_tile,
                    options=copy_options(l2_cache_ctl=self.l2_cache_ctl_b),
                )
                for k_l0_idx in range(current_step_k):
                    if const_expr(tiling.transpose_a):
                        l1_a_slice = tile_slice(
                            l1_a_tensor, (tiling.base_k, tiling.base_m), (k_l0_idx, 0)
                        )
                    else:
                        l1_a_slice = tile_slice(
                            l1_a_tensor, (tiling.base_m, tiling.base_k), (0, k_l0_idx)
                        )
                    if const_expr(tiling.transpose_b):
                        l1_b_slice = tile_slice(
                            l1_b_tensor, (tiling.base_n, tiling.base_k), (0, k_l0_idx)
                        )
                    else:
                        l1_b_slice = tile_slice(
                            l1_b_tensor, (tiling.base_k, tiling.base_n), (k_l0_idx, 0)
                        )
                    a_source = (
                        l1_a_slice.permute(1, 0) if tiling.transpose_a else l1_a_slice
                    )
                    b_source = (
                        l1_b_slice if tiling.transpose_b else l1_b_slice.permute(1, 0)
                    )
                    l0a_tensor = l0a.next_write(bare=True)
                    l0a_tensor = l0a_tensor.view(make_layout(a_source.shape))
                    mem_copy(l0a_tensor, a_source)
                    l0b_tensor = l0b.next_write(bare=True)
                    l0b_tensor = l0b_tensor.view(make_layout(b_source.shape))
                    mem_copy(l0b_tensor, b_source)
                    c_view = l0c_acc.bare().view(
                        make_layout((l0a_tensor.shape[0], l0b_tensor.shape[0])),
                        dtype=l0c_dtype,
                        data_format="nz",
                    )
                    global_k = k_l1_idx * k_l0_per_l1 + k_l0_idx
                    is_final_acc = global_k + 1 == total_k_l0
                    matmul(
                        c_view,
                        l0a_tensor,
                        l0b_tensor,
                        init=(global_k == 0),
                        unit_flag=3 if is_final_acc else 2,
                    )
            l0c_tensor = l0c_acc

            mem_copy(
                gm_c_tile,
                convert(l0c_tensor, gm_c_tile.dtype, scale=deq_scale),
                options=copy_options(unit_flag=3),
            )

    @host
    def run(
        self,
        gm_a: Tensor,
        gm_b: Tensor,
        gm_scale_b: Tensor,
        gm_scale_a: Tensor,
        gm_c: Tensor,
    ):
        self.qbmm_kernel[self.tiling.used_core_num](
            gm_a,
            gm_b,
            gm_scale_b,
            gm_scale_a,
            gm_c,
        )


# ============================================================================
# 3. Torch Interface
# ============================================================================


_TORCH_INPUT_TO_DSL = {
    torch.int8: dtypes.int8,
    torch.float8_e4m3fn: dtypes.float8_e4m3fn,
    torch.float8_e5m2: dtypes.float8_e5m2,
}
_FP8_INPUT_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})
_TORCH_OUT_TO_DSL = {
    torch.float16: (dtypes.float16, 2),
    torch.bfloat16: (dtypes.bfloat16, 2),
    torch.float32: (dtypes.float32, 4),
}
_INT8_OUT = frozenset({torch.float16, torch.bfloat16})
_FP8_OUT = frozenset({torch.float16, torch.bfloat16, torch.float32})


def _resolve_input_dtypes(a_torch_dtype, b_torch_dtype):
    """Validate the input quantization path and return its DSL dtypes."""
    a_dsl = _TORCH_INPUT_TO_DSL.get(a_torch_dtype)
    b_dsl = _TORCH_INPUT_TO_DSL.get(b_torch_dtype)
    if a_dsl is None or b_dsl is None:
        raise TypeError(
            f"a/b must be int8 or float8_e4m3fn/float8_e5m2, "
            f"got {a_torch_dtype} and {b_torch_dtype}"
        )
    a_is_fp8 = a_torch_dtype in _FP8_INPUT_DTYPES
    b_is_fp8 = b_torch_dtype in _FP8_INPUT_DTYPES
    if a_is_fp8 != b_is_fp8:
        raise TypeError(
            "a and b must both be int8 or both be FP8, "
            f"got {a_torch_dtype} and {b_torch_dtype}"
        )
    return a_is_fp8, a_dsl, b_dsl


def _infer_transpose_from_stride(tensor, name):
    """Distinguish a dense 2-D tensor from its simple transpose view."""
    rows, cols = tensor.shape
    stride = tuple(tensor.stride())
    if stride == (cols, 1):
        return False
    if stride == (1, rows):
        return True
    raise ValueError(
        f"{name} must be contiguous or a simple 2-D transpose view, "
        f"got shape={tuple(tensor.shape)}, stride={stride}"
    )


def _validate_scalar_scale(scale: torch.Tensor, name: str) -> None:
    """Validate one runtime FP32 per-tensor scale."""
    if scale is None:
        raise ValueError(f"{name} is required for TT (per-tensor) quantization")
    if scale.dtype != torch.float32:
        raise TypeError(f"{name} must be a float32 scalar tensor, got {scale.dtype}")
    if scale.numel() != 1:
        raise ValueError(
            f"{name} must be a per-tensor scalar (1 element), got {tuple(scale.shape)}"
        )


def _validate_inputs(a, b, scale_a, scale_b, output_dtype, a_dtype, b_dtype):
    """Validate inputs and resolve shape, layout, and DSL input dtypes."""
    a_is_hif8 = a_dtype == torch_npu.hifloat8
    b_is_hif8 = b_dtype == torch_npu.hifloat8
    if a_is_hif8 or b_is_hif8:
        if not (a_is_hif8 and b_is_hif8):
            raise TypeError("a_dtype and b_dtype must both be torch_npu.hifloat8")
        if a.dtype != torch.int8 or b.dtype != torch.int8:
            raise TypeError("HiFloat8 a/b storage tensors must use torch.int8")
        is_fp8 = True
        a_dsl = b_dsl = dtypes.hifloat8
        out_options = _FP8_OUT
    else:
        if a_dtype is not None or b_dtype is not None:
            raise NotImplementedError("a_dtype/b_dtype only support torch_npu.hifloat8")
        is_fp8, a_dsl, b_dsl = _resolve_input_dtypes(a.dtype, b.dtype)
        out_options = _FP8_OUT if is_fp8 else _INT8_OUT

    if a.dim() != 2 or b.dim() != 2:
        raise NotImplementedError(
            f"TT quantization requires 2-D inputs, got rank {a.dim()} and {b.dim()}"
        )
    if output_dtype is None or output_dtype not in out_options:
        allowed = ", ".join(str(dtype) for dtype in sorted(out_options, key=str))
        raise ValueError(
            f"output_dtype={output_dtype} not supported for "
            f"{'fp8' if is_fp8 else 'int8'}; allowed: {allowed}"
        )
    _validate_scalar_scale(scale_a, "scale_a")
    _validate_scalar_scale(scale_b, "scale_b")
    if any(t.device != a.device for t in (b, scale_a, scale_b)):
        raise ValueError("a, b and both scales must be on the same device")

    m, k = a.shape
    b_k, n = b.shape
    if b_k != k:
        raise ValueError(f"K mismatch: A K={k}, B K={b_k}")
    transpose_a = _infer_transpose_from_stride(a, "a")
    transpose_b = _infer_transpose_from_stride(b, "b")
    return m, n, k, is_fp8, a_dsl, b_dsl, transpose_a, transpose_b


def npu_quant_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor = None,
    scale_b: torch.Tensor = None,
    *,
    output_dtype=None,
    a_dtype=None,
    b_dtype=None,
) -> torch.Tensor:
    """Execute a two-dimensional TT (per-tensor) quantized matrix multiply.

    ``scale_a`` and ``scale_b`` are the per-tensor scales for A and B, in that
    order. Both must be scalar float32 tensors (one value per operand).

    INT8 path: int8 operands, int32 L0C, fused scalar FIXPIPE dequant -> fp16/bf16.
    FP8 path (e4m3/e5m2/hifloat8): float8 operands, fp32 L0C, fused
    QF322*_PRE FIXPIPE dequant (narrowing included) -> fp16/bf16/fp32 output.
    HiFloat8 inputs use int8 storage tensors with
    ``a_dtype=b_dtype=torch_npu.hifloat8``. The storage bytes are passed to
    the kernel unchanged, with ``dtypes.hifloat8`` selecting Cube decoding.

    ``a_dtype`` and ``b_dtype`` only accept ``torch_npu.hifloat8``.
    """
    (
        m,
        n,
        k,
        is_fp8,
        a_dsl,
        b_dsl,
        transpose_a,
        transpose_b,
    ) = _validate_inputs(
        a,
        b,
        scale_a,
        scale_b,
        output_dtype,
        a_dtype,
        b_dtype,
    )

    # GM->L1 requires unit innermost stride. Restore the physical axis order
    # with a metadata-only view; Tensor.t() does not copy input data.
    a_kernel_input = a.t() if transpose_a else a
    b_kernel_input = b.t() if transpose_b else b
    if is_fp8:
        a_kernel_input = a_kernel_input.view(torch.int8)
        b_kernel_input = b_kernel_input.view(torch.int8)
    out_dsl, out_size = _TORCH_OUT_TO_DSL[output_dtype]
    tiling = QbmmTtTiling(
        m,
        n,
        k,
        a_dtype=a_dsl,
        b_dtype=b_dsl,
        out_dtype_size=out_size,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )
    output = torch.empty((m, n), dtype=output_dtype, device=a.device)
    op = QbmmTtKernel(tiling, out_dtype=out_dsl)
    op.run(
        a_kernel_input,
        b_kernel_input,
        scale_b.reshape(1),
        scale_a.reshape(1),
        output,
    )
    return output
