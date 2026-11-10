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

from cannbotdsl import from_tensor

from cannbotdsl.lang.host import host

__all__ = ["matmul"]

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
    CUBE_BOUND_RATIO = 0.85
    MIN_COMPUTE_BOUND_TILE_AREA = 32768  # half of L0C (fp32 elems)
    L0C_DB_MAX_AREA = 32768  # base_m*base_n <= this enables l0c_db=2
    L0C_DB_MIN_DIM = 128  # relaxed min dim when targeting l0c_db=2
    MIN_COMPUTE_BOUND_TILE_DIM = 192  # thinner tiles regress
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
        stream = torch.npu.current_stream()
        self.AIC_NUM = int(get_platform_info(stream=stream).cube_core_num)
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
        self._refine_balance_compute_bound()
        self._finalize()

    def _refine_balance_compute_bound(self):
        """Re-pick the best discrete-balance tile for compute-bound shapes.

        The main search accepts on a cube-bound/balance joint ratio, which
        prefers large tiles even when the shape is compute-bound, and its
        BALANCE_RATE_EDGE early-stop halts exploration once the rate reaches
        0.9. For compute-bound shapes the discrete wall (max tiles per core)
        dominates the cube-bound proxy. Two guards keep the refinement safe:
          * tile-area floor — the balance rate does not price per-tile
            overhead, so degenerate fine tiles must be excluded;
          * operand-reload gate — a thinner tile multiplies the per-tile
            reload count of one operand (n_t up => A volume, m_t up => B
            volume); candidates whose total reload volume exceeds 1.15x the
            searched config are rejected.
        Memory-bound shapes keep the searched choice. Ties break toward the
        larger tile (fewer per-tile overheads), then wider base_m.
        """
        if getattr(self, "is_memory_bound", False):
            return
        l0c_area = self.L0C_SIZE // self.DATA_SIZE_FP32

        def reload_volume(bm, bn):
            return (
                ceil_div(self.n, bn) * self.m + ceil_div(self.m, bm) * self.n
            ) * self.k

        cur_reload = reload_volume(self.base_m, self.base_n)
        cur_rate = self._get_balance_rate_with_tail(self.base_m, self.base_n)
        best = (cur_rate, self.base_m * self.base_n, self.base_m, self.base_n)
        m_cap = min(self.BASIC_BLOCK_256, ceil_align(self.m, self.BASIC_BLOCK_16))
        n_cap = min(self.BASIC_BLOCK_256, ceil_align(self.n, self.BASIC_BLOCK_16))
        for bm in range(self.BASIC_BLOCK_16, m_cap + 1, self.BASIC_BLOCK_16):
            for bn in range(
                self.BASIC_BLOCK_16, min(n_cap, l0c_area // bm) + 1, self.BASIC_BLOCK_16
            ):
                if bm * bn < self.MIN_COMPUTE_BOUND_TILE_AREA:
                    continue
                if (
                    bm < self.MIN_COMPUTE_BOUND_TILE_DIM
                    or bn < self.MIN_COMPUTE_BOUND_TILE_DIM
                ):
                    continue
                if reload_volume(bm, bn) > cur_reload * 1.15:
                    continue
                rate = self._get_balance_rate_with_tail(bm, bn)
                key = (rate, bm * bn, bm)
                if key > (best[0], best[1], best[2]):
                    best = (rate, bm * bn, bm, bn)
        # L0C double-buffer search: if the best tile area exceeds L0C capacity
        # for double buffering, search for a smaller tile (area <= 32768,
        # dims >= 128) with a balance rate within 15% of the best. The
        # l0c_db=2 fixpipe/MMAD overlap (drain slot 0 while accumulating
        # slot 1) recovers the smaller-tile cost.
        if best[1] > self.L0C_DB_MAX_AREA:
            small_best = None
            for bm in range(self.L0C_DB_MIN_DIM, m_cap + 1, self.BASIC_BLOCK_16):
                for bn in range(
                    self.L0C_DB_MIN_DIM,
                    min(n_cap, self.L0C_DB_MAX_AREA // bm) + 1,
                    self.BASIC_BLOCK_16,
                ):
                    if bm * bn < self.MIN_COMPUTE_BOUND_TILE_AREA:
                        continue
                    if reload_volume(bm, bn) > cur_reload * 1.15:
                        continue
                    rate = self._get_balance_rate_with_tail(bm, bn)
                    if rate >= best[0] * 0.85:
                        key = (rate, bm * bn, bm)
                        if small_best is None or key > (
                            small_best[0],
                            small_best[1],
                            small_best[2],
                        ):
                            small_best = (rate, bm * bn, bm, bn)
            if small_best is not None:
                best = small_best

        if (best[2], best[3]) != (self.base_m, self.base_n):
            self.base_m, self.base_n = best[2], best[3]
            self._get_base_k()
            self._cal_l1_tiling()

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
        self.is_memory_bound = is_memory_bound
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
        """Get balance rate for non-batch basic matmul.

        Matches the kernel's discrete round-robin dispatch: a core processes
        at most ceil(total_tiles / AIC_NUM) tiles, and tail blocks are NOT
        split across cores.
        """
        total_round = ceil_div(self.m, base_m) * ceil_div(self.n, base_n)
        rounds = max(1, ceil_div(total_round, self.AIC_NUM))

        eff_base_m = self.m if self.m <= self.BASIC_BLOCK_16 else base_m
        eff_base_n = self.n if self.n <= self.BASIC_BLOCK_16 else base_n
        return (self.m * self.n / self.AIC_NUM) / (rounds * eff_base_m * eff_base_n)

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
        self.m_total = tiling.m
        self.n_total = tiling.n
        self.k_total = tiling.k
        self.m_tiles = ceil_div(tiling.m, tiling.base_m)
        self.n_tiles = ceil_div(tiling.n, tiling.base_n)
        self.k_l1_tiles = ceil_div(tiling.k, tiling.k_l1)
        self.k_l0_per_l1 = ceil_div(tiling.k_l1, tiling.base_k)
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
        self._build_tile_segments()
        self._decide_tail_split()
        tiling.used_core_num = min(self.total_tiles, tiling.AIC_NUM)

    def _build_tile_segments(self):
        """Window-enumeration metadata over the FULL base tile grid.

        The mainline-equivalent schedule enumerates every base tile (full
        and natural-tail) in slide-window snake order; the flat index space
        is [0, a_cnt) unsplit blocks followed by the last scheduling round's
        leftover blocks expanded into m_tc x n_tc pieces.
        """
        hm = self.m_tiles
        hn = self.n_tiles
        self.seg_a_rows = hm
        self.seg_a_cols = hn if hn > 0 else 1
        self.a_window = min(WINDOW_LEN, hm) if hm > 0 else 1
        self.a_main_row = max(0, hm // self.a_window - 1) if hm > 0 else 0
        self.a_tail_window = hm - self.a_main_row * self.a_window if hm > 0 else 1

    # Per-core rates for the split cost model: cube ~7.5e12 MAC/s, nd2nz
    # GM->L1 ~170e9 B/s. Each core is scored as max(mac_time, mte2_time):
    # the stages pipeline, so the slower one bounds the wall clock.
    # Splitting multiplies operand re-reads (B is re-read once per m row,
    # A once per n column), which grows mte2_time; on MTE2-bound shapes
    # (deep K) that outweighs the MAC balance a work-only model would chase.
    MAC_RATE = 7.5e12
    MTE2_BW = 1.7e11
    TILE_RESTART_FACTOR = 2

    def _window_decode(self, win_idx):
        """Host mirror of the kernel's slide-window snake decode."""
        a_cols = self.seg_a_cols
        row_idx = win_idx // a_cols // self.a_window
        if row_idx < self.a_main_row:
            m_row = row_idx * self.a_window + win_idx % self.a_window
            n_col = (win_idx // self.a_window) % a_cols
        else:
            row_idx = self.a_main_row
            tail_index = win_idx - self.a_main_row * self.a_window * a_cols
            m_row = self.a_main_row * self.a_window + tail_index % self.a_tail_window
            n_col = (tail_index // self.a_tail_window) % a_cols
        if row_idx % 2 != 0:
            n_col = a_cols - 1 - n_col
        return m_row, n_col

    def _block_extent(self, m_row, n_col):
        t = self.t
        m_ext = t.base_m if m_row < self.m_tiles - 1 else self.m_tail
        n_ext = t.base_n if n_col < self.n_tiles - 1 else self.n_tail
        return m_ext, n_ext

    def _sim_round_loads(self, m_tc, n_tc):
        """Mirror of the round-split enumeration + grid-stride assignment;
        returns the max per-core max(mac, mte2) workload.

        Each tile additionally costs TILE_RESTART_FACTOR x one k_l1
        window's MAC time: the L1/MTE2 prefetch pipeline drains at every
        tile boundary and the first window of the next tile cannot hide it.
        """
        t = self.t
        tail_cnt = m_tc * n_tc
        total = self.a_cnt + self.tail_blocks * tail_cnt
        launched = max(1, min(total, t.AIC_NUM))
        restart = self.TILE_RESTART_FACTOR * (
            t.base_m * t.base_n * t.k_l1 / self.MAC_RATE
        )
        mac_loads = [0.0] * launched
        mte2_loads = [0.0] * launched
        for tile_idx in range(total):
            if tile_idx < self.a_cnt:
                win_idx, piece = tile_idx, -1
            else:
                j = tile_idx - self.a_cnt
                win_idx, piece = (
                    self.a_cnt + j // tail_cnt,
                    j - (j // tail_cnt) * tail_cnt,
                )
            m_row, n_col = self._window_decode(win_idx)
            m_ext, n_ext = self._block_extent(m_row, n_col)
            if piece >= 0:
                s_m = (m_ext + m_tc - 1) // m_tc
                s_n = (n_ext + n_tc - 1) // n_tc
                m_si, n_si = piece % m_tc, piece // m_tc
                m_ext = min(m_ext - m_si * s_m, s_m)
                n_ext = min(n_ext - n_si * s_n, s_n)
            core = tile_idx % launched
            mac_loads[core] += m_ext * n_ext * self.k_total / self.MAC_RATE
            mte2_loads[core] += (m_ext + n_ext) * self.k_total * 2 / self.MTE2_BW
            mte2_loads[core] += restart
        return max(max(a, b) for a, b in zip(mac_loads, mte2_loads))

    def _sim_mac_tail(self, m_tc, n_tc):
        """Cube-path view of one split candidate: returns the max per-core
        mac workload, the max per-core tile count and the max per-core
        mte2 data volume (restart excluded).

        Second-stage criterion for _decide_tail_split: the primary model
        charges the tile-boundary restart inside the mte2 path, so on
        cube-bound grids a restart-inflated memory estimate can mask the
        mac-path gain of a beneficial last-round split and reject it --
        leaving the busiest core a full leftover block while the mainline
        scheduler halves it. The stage-2 comparison is only consulted
        when the primary model declines; accepting requires the maximum
        per-core tile count to stay at or below the baseline (every
        core's restart charge can only shrink), the memory path to keep
        up with the cube on both the baseline and the candidate (the
        mac gain is only real when the cube binds; a memory-bound grid
        would regress on the split's operand re-reads), and a >=3% mac
        gain. Over-split candidates that add per-core tiles stay
        rejected.
        """
        t = self.t
        tail_cnt = m_tc * n_tc
        total = self.a_cnt + self.tail_blocks * tail_cnt
        launched = max(1, min(total, t.AIC_NUM))
        mac_loads = [0.0] * launched
        mte2_loads = [0.0] * launched
        tile_counts = [0] * launched
        for tile_idx in range(total):
            if tile_idx < self.a_cnt:
                win_idx, piece = tile_idx, -1
            else:
                j = tile_idx - self.a_cnt
                win_idx, piece = (
                    self.a_cnt + j // tail_cnt,
                    j - (j // tail_cnt) * tail_cnt,
                )
            m_row, n_col = self._window_decode(win_idx)
            m_ext, n_ext = self._block_extent(m_row, n_col)
            if piece >= 0:
                s_m = (m_ext + m_tc - 1) // m_tc
                s_n = (n_ext + n_tc - 1) // n_tc
                m_si, n_si = piece % m_tc, piece // m_tc
                m_ext = min(m_ext - m_si * s_m, s_m)
                n_ext = min(n_ext - n_si * s_n, s_n)
            core = tile_idx % launched
            mac_loads[core] += m_ext * n_ext * self.k_total / self.MAC_RATE
            mte2_loads[core] += (m_ext + n_ext) * self.k_total * 2 / self.MTE2_BW
            tile_counts[core] += 1
        return max(mac_loads), max(tile_counts), max(mte2_loads)

    def _decide_tail_split(self):
        """Mainline-equivalent schedule: last-round leftover block split.

        Base grid = ceil(m/base_m) x ceil(n/base_n) tiles enumerated in
        slide-window snake order. Tiles that complete full grid-stride
        rounds run unsplit; only the blocks of the LAST scheduling round
        (base_blocks %% cores leftovers, or all blocks when the grid is
        smaller than the core count) are each split into m_tc x n_tc
        sub-pieces so the final round spreads lighter pieces across idle
        cores -- exactly mainline MatMulV3's mTailCnt x nTailCnt mechanism
        (Blaze BlockSchedulerMatmulBasic). Pieces partition their parent
        block, so operand re-read volume is unchanged and the MTE2 command
        mix keeps full blocks for the unsplit rounds. Candidates are scored
        with the grid-stride load simulation; a split needs a >=5%
        predicted win over the unsplit baseline.
        """
        t = self.t
        m_tail = self.m_total - (self.m_tiles - 1) * t.base_m
        n_tail = self.n_total - (self.n_tiles - 1) * t.base_n
        self.m_tail, self.n_tail = m_tail, n_tail
        self.m_sched_full, self.n_sched_full = self.m_tiles - 1, self.n_tiles - 1
        self.m_split = self.n_split = 1
        self.m_split_base, self.m_split_rem = m_tail, 0
        self.n_split_base, self.n_split_rem = n_tail, 0
        self.m_rows, self.n_cols = self.m_tiles, self.n_tiles

        base_blocks = self.m_tiles * self.n_tiles
        tail_blocks = base_blocks % t.AIC_NUM
        self.tail_blocks = tail_blocks
        self.a_cnt = base_blocks - tail_blocks

        # Split-factor caps from the smallest last-round block extents.
        min_m, min_n = t.base_m, t.base_n
        for j in range(tail_blocks):
            m_row, n_col = self._window_decode(self.a_cnt + j)
            m_ext, n_ext = self._block_extent(m_row, n_col)
            min_m = min(min_m, m_ext)
            min_n = min(min_n, n_ext)
        m_cap = max(1, min(4, min_m // 16))
        n_cap = max(1, min(4, min_n // 16))

        base_load = self._sim_round_loads(1, 1)
        best = (1, 1, base_load)
        for m_tc in range(1, m_cap + 1):
            for n_tc in range(1, n_cap + 1):
                if m_tc == 1 and n_tc == 1:
                    continue
                load = self._sim_round_loads(m_tc, n_tc)
                if load < base_load * 0.95 and load < best[2]:
                    best = (m_tc, n_tc, load)
        if best[0] != 1 or best[1] != 1:
            # Direction tie-break: equal-mac split shapes (equal m_tc x
            # n_tc products) can metric-tie while the loop's first-found
            # pick favored the N-major evaluation order; resolve near-ties
            # (<=0.5%) toward the most balanced factor pair, then M-major.
            near = [
                (m, n)
                for m in range(1, m_cap + 1)
                for n in range(1, n_cap + 1)
                if (m, n) != (1, 1) and self._sim_round_loads(m, n) <= best[2] * 1.005
            ]
            if near:
                pick = min(near, key=lambda mn: (abs(mn[0] - mn[1]), -mn[0]))
                if pick != (best[0], best[1]):
                    best = (pick[0], pick[1], self._sim_round_loads(*pick))
        if best[0] == 1 and best[1] == 1 and tail_blocks > 0:
            # Stage 2 -- cube-path criterion (see _sim_mac_tail): the primary
            # metric declined, but on cube-bound grids its restart-inflated
            # mte2 estimate can mask a real mac-path gain. Both directions
            # are searched (the M cap can be 1 when the last round holds
            # sub-32-row blocks, leaving N the only splittable axis);
            # candidates resolve on mac gain, then the balanced M-major
            # mainline convention.
            base_mac, base_cnt, base_mte2 = self._sim_mac_tail(1, 1)
            if base_mte2 < base_mac:
                stage2 = None
                for m_tc in range(1, m_cap + 1):
                    for n_tc in range(1, n_cap + 1):
                        if m_tc == 1 and n_tc == 1:
                            continue
                        cand_mac, cand_cnt, cand_mte2 = self._sim_mac_tail(m_tc, n_tc)
                        if cand_cnt > base_cnt or cand_mte2 >= cand_mac:
                            continue
                        if cand_mac < base_mac * 0.97:
                            key = (cand_mac, abs(m_tc - n_tc), -m_tc)
                            if stage2 is None or key < stage2[0]:
                                stage2 = (key, m_tc, n_tc)
                if stage2 is not None:
                    best = (stage2[1], stage2[2], base_load)
        self.m_tc, self.n_tc = best[0], best[1]
        self.tail_cnt = self.m_tc * self.n_tc
        self.total_tiles = self.a_cnt + tail_blocks * self.tail_cnt

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

        total_tiles = self.total_tiles
        k_l1_tiles = self.k_l1_tiles
        k_l0_per_l1 = self.k_l0_per_l1
        m_full_rows = self.m_sched_full
        n_full_cols = self.n_sched_full
        m_split_base = self.m_split_base
        m_split_rem = self.m_split_rem
        n_split_base = self.n_split_base
        n_split_rem = self.n_split_rem
        m_tail_start = m_full_rows * t.base_m
        n_tail_start = n_full_cols * t.base_n

        # LPT tile enumeration: windowed heavy grid first, light segments
        # (tail rows/cols, split pieces) last so grid-stride rounds pair
        # heavy and light tiles on the same core.
        a_cnt = self.a_cnt
        a_cols = self.seg_a_cols
        a_window = self.a_window
        a_main_row = self.a_main_row
        a_tail_window = self.a_tail_window
        tail_cnt = self.tail_cnt
        m_tc = self.m_tc
        n_tc = self.n_tc

        block_idx = get_block_idx()
        block_num = get_block_num()

        for tile_idx in range(block_idx, total_tiles, block_num):
            last_round = tile_idx >= a_cnt
            j = tile_idx - a_cnt if last_round else 0
            win_idx = a_cnt + j // tail_cnt if last_round else tile_idx
            piece = j - (j // tail_cnt) * tail_cnt if last_round else -1
            m_row = 0
            n_col = 0
            row_idx = win_idx // a_cols // a_window
            if row_idx < a_main_row:
                m_row = row_idx * a_window + win_idx % a_window
                n_col = (win_idx // a_window) % a_cols
            else:
                row_idx = a_main_row
                tail_index = win_idx - a_main_row * a_window * a_cols
                m_row = a_main_row * a_window + tail_index % a_tail_window
                n_col = (tail_index // a_tail_window) % a_cols
            n_col = (a_cols - 1 - n_col) if (row_idx % 2 != 0) else n_col

            # Runtime tile window: full rows/cols start at tile-multiple
            # origins; tail pieces start at even-distribution split origins.
            m_origin = m_row * t.base_m
            m_extent = t.base_m
            if m_row >= m_full_rows:
                m_piece = m_row - m_full_rows
                m_pre = m_piece if m_piece < m_split_rem else m_split_rem
                m_origin = m_tail_start + m_piece * m_split_base + m_pre
                m_extent = m_split_base + (1 if m_piece < m_split_rem else 0)

            n_origin = n_col * t.base_n
            n_extent = t.base_n
            if n_col >= n_full_cols:
                n_piece = n_col - n_full_cols
                n_pre = n_piece if n_piece < n_split_rem else n_split_rem
                n_origin = n_tail_start + n_piece * n_split_base + n_pre
                n_extent = n_split_base + (1 if n_piece < n_split_rem else 0)

            # Last-round leftover block piece (identity when piece < 0):
            # subdivide the parent block's window in both dims.
            act = piece >= 0
            s_m = (m_extent + m_tc - 1) // m_tc
            s_n = (n_extent + n_tc - 1) // n_tc
            m_si = piece % m_tc if act else 0
            n_si = piece // m_tc if act else 0
            m_end = m_origin + m_extent
            n_end = n_origin + n_extent
            m_org2 = m_origin + m_si * s_m
            n_org2 = n_origin + n_si * s_n
            m_origin = m_org2
            m_extent = min(m_end - m_org2, s_m) if act else m_extent
            n_origin = n_org2
            n_extent = min(n_end - n_org2, s_n) if act else n_extent

            # Runtime tile window on GM: split pieces can start at arbitrary
            # (non tile-multiple) row/column origins, so take per-tile
            # windows as runtime interval slices of the GM tensors.
            gm_c_tile = gm_c[
                m_origin : m_origin + m_extent, n_origin : n_origin + n_extent
            ]
            l0c_acc = from_tensor(
                l0c.next_write(),
                layout=make_layout(gm_c_tile.shape),
                dtype=dtypes.float32,
                data_format="nz",
            )
            for k_l1_idx in range(k_l1_tiles):
                k_org = k_l1_idx * t.k_l1
                k_rem = self.k_total - k_org
                k_act = k_rem if k_rem < t.k_l1 else t.k_l1
                a_piece = gm_a[m_origin : m_origin + m_extent, k_org : k_org + k_act]
                b_piece = gm_b[n_origin : n_origin + n_extent, k_org : k_org + k_act]
                # Declare the tile's TRUE extents ONCE per k_l1 chunk by
                # slicing the slot view, so one runtime shape drives every
                # L1->L0 copy, the MMAD and (through the operand chain)
                # avoids the per-instruction clamp-chain re-materialization
                # the channel transaction extent would otherwise emit inside
                # the k_l0 loop (documented tail-block pattern, ops.matmul
                # docstring).
                l1_a_tensor = l1_a.next_write()
                l1_a_tensor = l1_a_tensor[tuple(map(slice, a_piece.shape))]
                mem_copy(
                    l1_a_tensor,
                    a_piece,
                    options=copy_options(l2_cache_ctl=self.l2_cache_ctl_a),
                )
                l1_b_tensor = l1_b.next_write()
                l1_b_tensor = l1_b_tensor[tuple(map(slice, b_piece.shape))]
                mem_copy(
                    l1_b_tensor,
                    b_piece,
                    options=copy_options(l2_cache_ctl=self.l2_cache_ctl_b),
                )
                k_l0_total = ceil_div(k_act, t.base_k)
                for k_l0_idx in range(k_l0_total):
                    l1_a_slice = tile_slice(
                        l1_a_tensor, (t.base_m, t.base_k), (0, k_l0_idx)
                    )
                    l1_b_slice = tile_slice(
                        l1_b_tensor, (t.base_n, t.base_k), (0, k_l0_idx)
                    )
                    l0a_tensor = l0a.next_write()
                    l0a_tensor = from_tensor(
                        l0a_tensor, layout=make_layout(l1_a_slice.shape)
                    )
                    mem_copy(l0a_tensor, l1_a_slice)
                    l0b_tensor = l0b.next_write()
                    l0b_tensor = from_tensor(
                        l0b_tensor, layout=make_layout(l1_b_slice.shape)
                    )
                    mem_copy(l0b_tensor, l1_b_slice)
                    global_k = k_l1_idx * k_l0_per_l1 + k_l0_idx
                    is_final = k_l1_idx == k_l1_tiles - 1 and k_l0_idx == k_l0_total - 1
                    c_view = from_tensor(
                        l0c_acc,
                        layout=make_layout((l0a_tensor.shape[0], l0b_tensor.shape[0])),
                        dtype=dtypes.float32,
                        data_format="nz",
                    )
                    dsl_matmul(
                        c_view,
                        l0a_tensor,
                        l0b_tensor,
                        init=(global_k == 0),
                        unit_flag=(3 if is_final else 2),
                    )
            l0c_tensor = l0c_acc

            mem_copy(gm_c_tile, l0c_tensor, options=copy_options(unit_flag=3))

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
