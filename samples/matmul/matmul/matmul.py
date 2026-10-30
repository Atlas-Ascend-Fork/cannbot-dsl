# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Non-quantized matmul operator with host-side tiling.

Structure:
  1. MatmulTiling       — host-side tiling
  2. MatmulKernel  — @kernel with slide window scheduling + L1/L0 ping-pong pipeline
  3. matmul()     — torch interface

Formula:  C[M,N] = A[M,K] @ B[N,K]^T   (fp16/bf16 inputs, fp32 accumulator)
"""

from cannbotdsl.lang.host import host

__all__ = ["matmul"]

import math
import torch
from cannbotdsl import make_layout, dtypes, get_mem_size, get_platform_info
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ring_buffer import RingBuffer


from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.matmul import matmul as dsl_matmul
from cannbotdsl.tensor import tile_slice
from cannbotdsl import MemLoc, Tensor
from cannbotdsl.ops.memcpy import copy_options, mem_copy


# Integer alignment helpers shared by MatmulTiling and MatmulKernel
def ceil_div(a: int, b: int) -> int:
    """Integer ceil division used for tile-count calculations."""
    if b == 0:
        return a
    return (a + b - 1) // b


def ceil_align(a: int, b: int) -> int:
    """Round a value up to the nearest multiple of the alignment."""
    if b == 0:
        return a
    return (a + b - 1) // b * b


def floor_align(a: int, b: int) -> int:
    """Round a value down to the nearest multiple of the alignment."""
    if b == 0:
        return a
    return a // b * b


# ============================================================================
# 1. Host-side Tiling
# ============================================================================

# L2 cache mode
L2_CACHE_DEFAULT = 0
A_L2_CACHE_DISABLE = 1
B_L2_CACHE_DISABLE = 2
ALL_L2_CACHE_DISABLE = 3


class MatmulTiling:
    """Host-side tiling computation.

    tiling flow:
      ResetBase → GetRebalanceBlock → GetBaseK → CalL1Tiling

    Key outputs consumed by the kernel:
      base_m, base_n, base_k  — L0 tile sizes (per-MMAD)
      m_l1, n_l1, k_l1        — L1 tile sizes (per-block)
      l1_buffer_num           — L1 ping-pong depth (2 or 4)
      l0c_db                  — L0C double buffer (1 or 2)
      used_core_num           — number of AIC cores to launch
    """

    # ---- hardware properties ----
    L1_SIZE = get_mem_size("l1")
    L0A_SIZE = get_mem_size("l0a")
    L0C_SIZE = get_mem_size("l0c")
    L2_SIZE = 128 * 1024 * 1024
    AIC_NUM = get_platform_info().cube_core_num

    # ---- tiling constants ----
    BASIC_BLOCK_16 = 16
    BASIC_BLOCK_64 = 64
    BASIC_BLOCK_128 = 128
    BASIC_BLOCK_256 = 256
    DB_SIZE = 2
    BASIC_L1_BUFFER_NUM = 4
    DATA_SIZE_FP32 = 4
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
    _TORCH_DTYPE_TO_DSL = {
        torch.float16: (dtypes.float16, 2),
        torch.bfloat16: (dtypes.bfloat16, 2),
    }
    _DSL_DTYPE_TO_SIZE = {
        dtypes.float16: (dtypes.float16, 2),
        dtypes.bfloat16: (dtypes.bfloat16, 2),
    }

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        a_dtype=dtypes.float16,
        b_dtype=dtypes.float16,
        c_dtype=dtypes.float16,
        transpose_a: bool = False,
        transpose_b: bool = True,
        *,
        has_bias: bool = False,
    ):
        """Store shape/type metadata and immediately derive all tiling fields.

        Args:
          m/n/k: logical matmul dimensions for C[M, N] = A[M, K] @ B[N, K]^T.
          a_dtype/b_dtype/c_dtype: input/output dtypes. They may be torch dtypes
            or cannbotdsl scalar types; tiling resolves both to cannbotdsl dtype and
            element size internally.
          transpose_a/transpose_b: logical transpose flags used by alignment and
            cache heuristics. The current kernel path is optimized for
            transpose_a=False and transpose_b=True.
          has_bias: reserved keyword-only flag for future bias tiling support.
        """
        self.m = m
        self.n = n
        self.k = k
        self.a_dtype, self.a_dtype_size = self._resolve_dtype(a_dtype, "a_dtype")
        self.b_dtype, self.b_dtype_size = self._resolve_dtype(b_dtype, "b_dtype")
        self.c_dtype, self.c_dtype_size = self._resolve_dtype(c_dtype, "c_dtype")
        self.has_bias = has_bias
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        # Tiling outputs; all of them are derived by _compute() below.
        self.base_m = None
        self.base_n = None
        self.base_k = None
        self.cube_bound_edge = None
        self.cube_bound_param = None
        self.step_ka = None
        self.step_kb = None
        self.m_l1 = None
        self.n_l1 = None
        self.k_l1 = None
        self.used_core_num = None
        self.l1_buffer_num = None
        self.l0c_db = None
        self.l2_cache_disable = None
        self._compute()

    @classmethod
    def _resolve_dtype(cls, dtype, name: str) -> tuple:
        """Resolve a torch/cannbotdsl dtype into the cannbotdsl dtype and byte size.

        Keeping this conversion in tiling prevents callers from passing a DSL
        dtype that disagrees with the element size used by the tiling model.
        """
        props = cls._TORCH_DTYPE_TO_DSL.get(dtype)
        if props is None:
            props = cls._DSL_DTYPE_TO_SIZE.get(dtype)
        if props is None:
            supported = ", ".join(sorted(d.__name__ for d in cls._DSL_DTYPE_TO_SIZE))
            raise TypeError(f"{name} only supports {supported}; got {dtype}")
        return props

    def _compute(self):
        """Run the host-side tiling pipeline in dependency order.

        Later stages depend on fields produced by earlier stages: base tile
        sizes are picked first, then K/L1 tiling and buffering decisions are
        derived from those base sizes.
        """
        self._reset_base()
        self._rebalance_block()
        self._cal_l1_tiling()
        self._finalize()

    def _reset_base(self):
        """Reset the default base tile sizes along M, N and K."""
        self.base_m = min(self.BASIC_BLOCK_256, ceil_align(self.m, self.BASIC_BLOCK_16))
        self.base_n = min(self.BASIC_BLOCK_256, ceil_align(self.n, self.BASIC_BLOCK_16))
        self.base_k = self.BASIC_BLOCK_K_128B // self.a_dtype_size

    def _rebalance_block(self):
        """pick base_m/base_n via cube-bound + balance-rate search.

        Flow: cal cube-bound edge → init candidate bounds → search → finalize.
        """
        base_mn_buffer_limit = self.L0C_SIZE
        hbm_bw = self._get_hbm_bw()
        l2_bw = self._get_l2_bw()

        base_m_align_unit, base_n_align_unit, max_base_m, max_base_n = (
            self._cal_cube_bound_edge(hbm_bw, l2_bw, base_mn_buffer_limit)
        )

        self._search_optimal_base_mn(
            base_mn_buffer_limit,
            base_m_align_unit,
            base_n_align_unit,
            max_base_m,
            max_base_n,
        )

        self.base_m = min(ceil_align(self.m, self.BASIC_BLOCK_16), self.base_m)
        self.base_n = min(ceil_align(self.n, self.BASIC_BLOCK_16), self.base_n)
        self._get_base_k()

    def _cal_cube_bound_edge(self, hbm_bw, l2_bw, base_mn_buffer_limit):
        """Compute the cube-bound edge, alignment units, and candidate bounds.

        Returns (base_m_align_unit, base_n_align_unit, max_base_m, max_base_n).
        Sets self.cube_bound_edge, self.base_m, self.base_n, self.cube_bound_param.
        """
        single_core_compute_power = self.DEFAULT_CORE_FREQ * self.MMAD_UNITS_PER_CORE
        compute_power = single_core_compute_power * self.AIC_NUM

        l2_cache_usage = max(
            (self.m + self.n) * self.k * self.a_dtype_size / self.L2_SIZE,
            1.0,
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
                    base_mn_buffer_limit // self.DATA_SIZE_FP32 // base_m_best,
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
            inner_align_unit // self.a_dtype_size
            if self.transpose_a
            else self.BASIC_BLOCK_16
        )
        base_n_align_unit = (
            self.BASIC_BLOCK_K_256B // self.b_dtype_size
            if self.k < fixp_bound_edge
            else (
                self.BASIC_BLOCK_16
                if self.transpose_b
                else inner_align_unit // self.b_dtype_size
            )
        )

        max_base_m = self._get_max_base_with_limit(
            base_mn_buffer_limit,
            base_m_align_unit,
            is_right_matrix=False,
            is_memory_bound=is_memory_bound,
        )
        max_base_n = self._get_max_base_with_limit(
            base_mn_buffer_limit,
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
                    base_mn_buffer_limit // self.DATA_SIZE_FP32 // self.base_m,
                    base_n_align_unit,
                ),
            ),
        )
        self.cube_bound_param = (1.0 / self.base_m) + (1.0 / self.base_n)
        self.cube_bound_edge *= self.CUBE_BOUND_RATIO

        return base_m_align_unit, base_n_align_unit, max_base_m, max_base_n

    def _search_optimal_base_mn(
        self,
        base_mn_buffer_limit,
        base_m_align_unit,
        base_n_align_unit,
        max_base_m,
        max_base_n,
    ):
        """Exhaustive search over base_m × base_n candidates.

        Picks the (base_m, base_n) pair that best balances cube-bound and
        load-balancing rate. Updates self.base_m, self.base_n,
        self.cube_bound_param, and self.cube_bound_edge in place.
        """
        balance_rate = self._get_balance_rate_with_tail(self.base_m, self.base_n)

        cur_base_m = max_base_m
        while cur_base_m >= 1:
            cur_max_base_n = min(
                max_base_n,
                floor_align(
                    base_mn_buffer_limit // self.DATA_SIZE_FP32 // cur_base_m,
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
        """base_k constrained by L0A size."""
        k_align = ceil_align(self.k, self.BASIC_BLOCK_16)
        max_base_k = (
            self.L0A_SIZE
            // self.DB_SIZE
            // self.a_dtype_size
            // max(self.base_m, self.base_n)
        )
        if k_align <= max_base_k:
            self.base_k = k_align
        elif self.transpose_a and not self.transpose_b:
            self.base_k = floor_align(max_base_k, self.BASIC_BLOCK_16)
        elif max_base_k * self.a_dtype_size >= self.BASIC_BLOCK_K_256B:
            k_256_elems = self.BASIC_BLOCK_K_256B // self.a_dtype_size
            self.base_k = floor_align(max_base_k, k_256_elems)
        else:
            for cand in [128, 64, 32, 16]:
                if max_base_k >= cand:
                    self.base_k = cand
                    break

    def _cal_l1_tiling(self):
        """Compute step_ka/step_kb from L1 budget."""
        bias_l1 = (
            self.base_n * self.DB_SIZE * self.DATA_SIZE_FP32 if self.has_bias else 0
        )
        total_l1 = self.L1_SIZE - bias_l1
        max_step_k = min(ceil_div(self.k, self.base_k), self.MAX_STEP_K)

        k_256_elems = self.BASIC_BLOCK_K_256B // self.a_dtype_size
        k_align_unit = (
            self.BASIC_BLOCK_K_512B // self.a_dtype_size
            if (not self.transpose_a or self.transpose_b)
            else self.BASIC_BLOCK_16
        )
        res_kl1 = 0
        single_mte_size = 0
        for step_k in range(1, max_step_k + 1):
            cur_kl1 = self.base_k * step_k
            a_l1 = self.base_m * cur_kl1 * self.a_dtype_size
            b_l1 = self.base_n * cur_kl1 * self.b_dtype_size
            if (a_l1 + b_l1) * self.DB_SIZE > total_l1:
                break
            if max(a_l1, b_l1) * self.DB_SIZE * 2 > self.L1_SIZE:
                break
            cond_no_res = res_kl1 == 0
            cond_k_align_256b = cur_kl1 % k_256_elems == 0
            cond_k_align = res_kl1 % k_align_unit != 0 and (
                cond_k_align_256b
                or (
                    not cond_k_align_256b
                    and single_mte_size < self.L1_SINGLE_SIZE_LIMIT
                )
            )
            cond_mte_size = (
                res_kl1 % k_align_unit == 0
                and cur_kl1 % k_align_unit == 0
                and single_mte_size < self.L1_SINGLE_SIZE_LIMIT
            )
            if cond_no_res or cond_k_align or cond_mte_size:
                res_kl1 = cur_kl1
                single_mte_size = max(a_l1, b_l1)

        self.step_ka = max(1, res_kl1 // self.base_k)
        self.step_kb = max(1, res_kl1 // self.base_k)

    def _set_disable_l2cache(self, m_l1, ka_l1, kb_l1, n_l1):
        """decide L2 cache mode per matrix.

        When total data fits in L2, keep cache enabled (DEFAULT).
        Otherwise, disable L2 cache for a matrix that is NOT reused across
        tiles (i.e., each core loads it only once), freeing L2 capacity for
        the reused matrix.

        Left matrix (A) is not reused when base_n covers the entire N dimension.
        Right matrix (B) is not reused when base_m covers the entire M dimension.
        """
        inner_a = self.m if self.transpose_a else self.k
        inner_b = self.k if self.transpose_b else self.n
        flag_a = (
            m_l1 if self.transpose_a else ka_l1
        ) * self.a_dtype_size % self.ALIGN_128 == 0
        flag_b = (
            kb_l1 if self.transpose_b else n_l1
        ) * self.b_dtype_size % self.ALIGN_128 == 0

        total_size = (
            self.m * self.n * self.c_dtype_size
            + self.m * self.k * self.a_dtype_size
            + self.k * self.n * self.b_dtype_size
        )
        if total_size < self.L2_SIZE:
            return L2_CACHE_DEFAULT

        m_cnt = ceil_div(self.m, self.base_m)
        n_cnt = ceil_div(self.n, self.base_n)

        left_not_l2_cache = (
            self.base_n >= self.n
            and n_cnt <= 1
            and inner_a * self.a_dtype_size % self.ALIGN_128 == 0
            and flag_a
        )
        right_not_l2_cache = (
            self.base_m >= self.m
            and m_cnt <= 1
            and inner_b * self.b_dtype_size % self.ALIGN_128 == 0
            and flag_b
        )

        if left_not_l2_cache and right_not_l2_cache:
            return ALL_L2_CACHE_DISABLE
        elif left_not_l2_cache:
            return A_L2_CACHE_DISABLE
        elif right_not_l2_cache:
            return B_L2_CACHE_DISABLE
        return L2_CACHE_DEFAULT

    def _finalize(self):
        """Compute final m_l1/n_l1/k_l1, used_core_num, l1_buffer_num, l0c_db."""
        self.m_l1 = min(ceil_align(self.m, self.BASIC_BLOCK_16), self.base_m)
        self.n_l1 = min(ceil_align(self.n, self.BASIC_BLOCK_16), self.base_n)
        step_ka = min(self.step_ka, self.step_kb, self.BASIC_L1_BUFFER_NUM)
        self.k_l1 = self.base_k * step_ka

        m_core = ceil_div(self.m, self.base_m)
        n_core = ceil_div(self.n, self.base_n)
        self.used_core_num = min(m_core * n_core, self.AIC_NUM)

        a_l1_4buf = (
            self.k_l1 * self.base_m * self.a_dtype_size * self.BASIC_L1_BUFFER_NUM
        )
        b_l1_4buf = (
            self.k_l1 * self.base_n * self.b_dtype_size * self.BASIC_L1_BUFFER_NUM
        )
        bias_4buf = (
            self.base_n * self.DATA_SIZE_FP32 * self.BASIC_L1_BUFFER_NUM
            if self.has_bias
            else 0
        )
        self.l1_buffer_num = (
            self.DB_SIZE
            if (a_l1_4buf + b_l1_4buf + bias_4buf) > self.L1_SIZE
            else self.BASIC_L1_BUFFER_NUM
        )

        self.l0c_db = (
            self.DB_SIZE
            if self.base_m * self.base_n * self.DATA_SIZE_FP32 * self.DB_SIZE
            <= self.L0C_SIZE
            else 1
        )

        self.l2_cache_disable = self._set_disable_l2cache(
            self.m_l1, self.k_l1, self.k_l1, self.n_l1
        )

    def _get_max_base_with_limit(
        self, base_mn_buffer_limit, base_align_unit, is_right_matrix, is_memory_bound
    ):
        """Compute the largest legal base_m/base_n candidate under buffer limits.

        This helper is shared by M and N tile selection. It combines L0A/L0C
        capacity, L1 capacity, K alignment requirements, transpose layout, and
        the actual matrix shape to return an aligned upper bound for one output
        tile dimension.
        """
        shape_value = self.n if is_right_matrix else self.m
        k_align_value = ceil_align(self.k, self.BASIC_BLOCK_16)
        k_limit_value = (
            self.BASIC_BLOCK_16
            if is_memory_bound
            else self.BASIC_BLOCK_K_128B // self.a_dtype_size
        )
        min_kl0 = min(k_limit_value, k_align_value) * self.a_dtype_size

        max_base_mn_with_buffer = (
            base_mn_buffer_limit // self.DATA_SIZE_FP32 // self.BASIC_BLOCK_16
        )
        max_base_block = min(
            self.L0A_SIZE // self.DB_SIZE // min_kl0,
            max_base_mn_with_buffer,
        )

        k_align_unit = (
            (self.BASIC_BLOCK_K_256B if is_memory_bound else self.BASIC_BLOCK_K_512B)
            // self.a_dtype_size
            if (not self.transpose_a or self.transpose_b)
            else self.BASIC_BLOCK_16
        )
        max_base_mn_with_k_inner = self.L1_SIZE // (
            2 * self.DB_SIZE * self.a_dtype_size * min(k_align_unit, k_align_value)
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
        """get balance rate with tail for non-batch basic matmul."""
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
        """Estimate per-chip HBM bandwidth in the tiling cost model."""
        return self.DEFAULT_CORE_FREQ * self.AIC_NUM * self.DEFAULT_DDR_RATE / 1024

    def _get_l2_bw(self):
        """Estimate per-chip L2 bandwidth in the tiling cost model."""
        return self.DEFAULT_CORE_FREQ * self.AIC_NUM * self.DEFAULT_L2_RATE / 1024


# ============================================================================
# 2. Kernel
# ============================================================================

WINDOW_LEN = 4


class MatmulKernel:
    """C = A @ B^T  with slide window multi-core scheduling + L1/L0 pipeline.

    GM → L1 (MTE2, ping-pong) → L0A/L0B (MTE1, ping-pong) → MMAD (M) → L0C → GM (FIXPIPE)
    """

    def __init__(self, tiling: MatmulTiling):
        """Pre-compute loop bounds and scheduling metadata for the kernel.

        The tiling object contains hardware-aware tile sizes. This constructor
        converts them into tile counts, K-loop nesting factors, slide window
        parameters, and L2 cache control bits consumed by the device kernel.
        """
        self.t = tiling
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

    # Device kernel entry. It maps output tiles to AIC cores, stages A/B from
    # GM to L1 and then L0A/L0B, performs K-sliced MMAD accumulation in L0C,
    # and writes the completed C tile back to GM.
    @kernel
    def matmul_kernel(self, gm_a: Tensor, gm_b: Tensor, gm_c: Tensor):
        t = self.t
        l1_a = RingBuffer(
            MemLoc.L1,
            t.a_dtype,
            make_layout((t.base_m, t.k_l1)),
            depth=t.l1_buffer_num,
        )
        l1_b = RingBuffer(
            MemLoc.L1,
            t.b_dtype,
            make_layout((t.base_n, t.k_l1)),
            depth=t.l1_buffer_num,
        )
        l0a = RingBuffer(
            MemLoc.L0A,
            t.a_dtype,
            make_layout((t.base_m, t.base_k)),
            depth=2,
        )
        l0b = RingBuffer(
            MemLoc.L0B,
            t.b_dtype,
            make_layout((t.base_n, t.base_k)),
            depth=2,
        )
        l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((t.base_m, t.base_n)),
            depth=t.l0c_db,
        )

        n_tiles = self.n_tiles
        main_window = self.main_window
        main_row = self.main_row
        tail_window = self.tail_window
        total_tiles = self.total_tiles
        k_l1_tiles = self.k_l1_tiles
        k_l0_per_l1 = self.k_l0_per_l1

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

            gm_c_tile = tile_slice(gm_c, (t.base_m, t.base_n), (m_idx, n_idx))
            l0c_acc = (
                l0c.next_write()
                .bare()
                .view(
                    make_layout(gm_c_tile.shape), dtype=dtypes.float32, data_format="nz"
                )
            )
            for k_l1_idx in range(k_l1_tiles):
                gm_a_tile = tile_slice(gm_a, (t.base_m, t.k_l1), (m_idx, k_l1_idx))
                gm_b_tile = tile_slice(gm_b, (t.base_n, t.k_l1), (n_idx, k_l1_idx))
                l1_a_tensor = l1_a.next_write()
                l1_a_tensor = l1_a_tensor[tuple(map(slice, gm_a_tile.shape))]
                mem_copy(
                    l1_a_tensor,
                    gm_a_tile,
                    options=copy_options(l2_cache_ctl=self.l2_cache_ctl_a),
                )
                l1_b_tensor = l1_b.next_write()
                l1_b_tensor = l1_b_tensor[tuple(map(slice, gm_b_tile.shape))]
                mem_copy(
                    l1_b_tensor,
                    gm_b_tile,
                    options=copy_options(l2_cache_ctl=self.l2_cache_ctl_b),
                )
                for k_l0_idx in range(ceil_div(l1_a_tensor.shape[1], t.base_k)):
                    l1_a_slice = tile_slice(
                        l1_a_tensor, (t.base_m, t.base_k), (0, k_l0_idx)
                    )
                    l1_b_slice = tile_slice(
                        l1_b_tensor, (t.base_n, t.base_k), (0, k_l0_idx)
                    )
                    l0a_tensor = l0a.next_write(bare=True)
                    l0a_tensor = l0a_tensor.view(make_layout(l1_a_slice.shape))
                    mem_copy(l0a_tensor, l1_a_slice)
                    l0b_tensor = l0b.next_write(bare=True)
                    l0b_tensor = l0b_tensor.view(make_layout(l1_b_slice.shape))
                    mem_copy(l0b_tensor, l1_b_slice)
                    global_k = k_l1_idx * k_l0_per_l1 + k_l0_idx
                    c_view = l0c_acc.bare().view(
                        make_layout((l0a_tensor.shape[0], l0b_tensor.shape[0])),
                        dtype=dtypes.float32,
                        data_format="nz",
                    )
                    dsl_matmul(c_view, l0a_tensor, l0b_tensor, init=(global_k == 0))
            l0c_tensor = l0c_acc

            mem_copy(gm_c_tile, l0c_tensor)

    # JIT wrapper used by the Python interface. The bracket syntax launches the
    # kernel with the number of AIC cores selected by host-side tiling.
    @host
    def run(self, gm_a: Tensor, gm_b: Tensor, gm_c: Tensor):
        self.matmul_kernel[self.t.used_core_num](gm_a, gm_b, gm_c)


# ============================================================================
# 3. Torch Interface
# ============================================================================


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    transpose_a: bool = False,
    transpose_b: bool = True,
) -> torch.Tensor:
    """Torch-facing wrapper for the cannbotdsl matmul kernel.

    The kernel computes C[M, N] = A[M, K] @ B[N, K]^T. This wrapper normalizes
    the input layout according to transpose flags, delegates dtype resolution to
    host-side tiling, converts torch NPU tensors to cannbotdsl tensors, and
    synchronously launches the JIT-compiled kernel.
    """
    if a.dtype != b.dtype:
        raise TypeError(f"a and b must have the same dtype, got {a.dtype} vs {b.dtype}")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(
            f"inputs must be 2-D, got a.dim()={a.dim()}, b.dim()={b.dim()}"
        )

    a_kern = a.t().contiguous() if transpose_a else a
    b_kern = b if transpose_b else b.t().contiguous()

    m, k = a_kern.shape
    n = b_kern.shape[0]

    tiling = MatmulTiling(
        m,
        n,
        k,
        a_dtype=a.dtype,
        b_dtype=b.dtype,
        c_dtype=a.dtype,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )
    c = torch.zeros(m, n, dtype=a.dtype, device=a_kern.device)
    op = MatmulKernel(tiling)

    gm_a = a_kern
    gm_b = b_kern
    gm_c = c

    op.run(gm_a, gm_b, gm_c)

    return c
