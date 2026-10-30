# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Non-quantized GroupedMatmul (GMM).

Layers:
  1. GmmTiling      — host-side tiling (tile sizes, L1 partition, core count,
                      L2 policy)
  2. BlockMmad      — per-tile pipeline: GM→L1→L0A/L0B→MMAD→L0C→GM
  3. GmmKernel      — unified kernel: per-group iteration + tile dispatch,
                      tensorlist in / tensorlist out
  4. group_matmul() — torch interface: validate → resolve scenario →
                      tiling → allocate output → launch

Supported scenarios:
  | scenario | gt | x         | weight         | y       | splitItem | groupList |
  |----------|----|-----------|----------------|---------|-----------|-----------|
  | S1 mmm   | -1 | multi [m,k] | multi 2-D    | multi   | 0/1       | must be None |
  | S2 sss   | 0  | single [M,K] | single 3-D [G,K,N] | single | 2/3 | required |
  | S3 sms   | 0  | single [M,K] | multi 2-D    | single  | 2/3       | required |
  | S4 mms   | 0  | multi [m,K]  | multi 2-D    | single  | 2         | optional |
  | S5 sss   | 2  | single transposed | single [K,N] | single 3-D | 2/3 | required |
  | S6 smm   | 2  | single transposed | multi 2-D | multi  | 0/1       | optional |

Key properties:
  - Zero materialization: tensorlists are passed to the kernel as-is (no
    cat/stack/contiguous); the kernel reads per-group elements and shapes
    dynamically, so M/N/K may differ across groups.
  - Padding free: inputs are never padded on the host; groups are sliced at
    their real extents, tail tiles are clipped by tile_slice, and the ND2NZ
    copy engine zero-pads tails.
"""

__all__ = ["group_matmul"]

import dataclasses

import torch
import cannbotdsl
from cannbotdsl import make_layout
from cannbotdsl import dtypes, MemLoc, get_mem_size, get_platform_info
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_ctrl, set_ctrl
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.tensor import (
    tile_slice,
    permute,
    ceil_div,
)


# ============================================================================
# Constants
# ============================================================================

DB_SIZE = 2
DATA_SIZE_FP32 = 4
BASIC_BLOCK_16 = 16
BASE_M_DEFAULT = 256
BASE_N_DEFAULT = 256
BASE_K_DEFAULT = 64
ALIGN_128 = 128
ALIGN_DOWN_16 = 15
NUM_TWO = 2

L1_SIZE = get_mem_size("l1")
L0A_SIZE = get_mem_size("l0a")
L0B_SIZE = get_mem_size("l0b")
L0C_SIZE = get_mem_size("l0c")
PARTA_L1_SIZE = L1_SIZE // 2
L2_SIZE = 128 * 1024 * 1024
AIC_NUM = get_platform_info().cube_core_num
MAX_TENSOR = 1024

WINDOW_LEN = 4
EVEN_ROWS = 2

NO_SPLIT = -1
SPLIT_M = 0
SPLIT_K = 2

GROUPLIST_TYPE_CUMSUM = 0
GROUPLIST_TYPE_COUNT = 1

SPLIT_ITEM_SEPARATED = 0  # 0/1: multi-tensor output
SPLIT_ITEM_NO_SEPARATED = 2  # 2/3: single-tensor output

_TORCH_DTYPE_TO_DSL = {
    torch.float16: (dtypes.float16, 2),
    torch.bfloat16: (dtypes.bfloat16, 2),
}


def _ceil_div(a, b):
    if b == 0:
        return a
    return (a + b - 1) // b


def _ceil_align(a, b):
    if b == 0:
        return a
    return (a + b - 1) // b * b


def _floor_align(a, b):
    if b == 0:
        return a
    return a // b * b


def _is_canonical_last_two_transpose(tensor, name):
    """Return whether ``tensor`` is a compact ``transpose(-1, -2)`` view.

    Non-transposed tensors must be contiguous; arbitrary strided/sliced
    views are rejected.
    """
    if tensor.dim() < 2:
        raise ValueError(f"{name} must have rank >= 2, got {tensor.dim()}")

    is_transposed = tensor.stride(-2) == 1 and tensor.stride(-1) == tensor.size(-2)
    if not is_transposed and not tensor.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous or a canonical transpose(-1, -2) "
            f"view; got shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}"
        )

    # Verify leading dims stay compact, rejecting sliced views that only
    # happen to satisfy the two trailing stride checks.
    if is_transposed:
        expected_stride = tensor.size(-2) * tensor.size(-1)
        for dim in range(tensor.dim() - 3, -1, -1):
            if tensor.stride(dim) != expected_stride:
                raise ValueError(
                    f"{name} must be a compact transpose(-1, -2) view; "
                    f"got shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}"
                )
            expected_stride *= tensor.size(dim)
    return is_transposed


# ============================================================================
# 1. Host-side tiling
# ============================================================================


class GmmTiling:
    """Host-side tiling.

    Inputs are max-statistics m/n/k (aggregated by the wrapper for
    multi-tensor scenarios).  Outputs:
      base_m/base_n/base_k — L0 tile sizes; k_l1 — K staged per L1 pass;
      step_ka/step_kb, depth_a1/depth_b1 — L1 buffer depths; l1_buffer_num —
      L1 buffer count; used_core_num — launch core count; weight_no_l2_cache —
      weight DMA L2 policy.
    A single group takes the full FBB/AswtL1 tuning path; multi-group
    SPLIT_K rebalances with remain_core_num as aic_num // group_num.
    """

    def __init__(
        self,
        m,
        n,
        k,
        group_num,
        a_dtype=dtypes.float16,
        b_dtype=dtypes.float16,
        c_dtype=dtypes.float16,
        *,
        trans_a=False,
        trans_b=True,
        group_type=SPLIT_M,
        group_list_type=GROUPLIST_TYPE_CUMSUM,
        l2_disable_seed=False,
    ):
        self.m = m
        self.n = n
        self.k = k
        self.group_num = group_num
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.group_type = group_type
        self.is_split_k = group_type == SPLIT_K
        # Scenario seed for the weight L2 disable policy (see
        # _set_disable_l2_cache).
        self.l2_disable_seed = l2_disable_seed

        self.a_dtype, self.a_dtype_size = self._resolve_dtype(a_dtype)
        self.b_dtype, self.b_dtype_size = self._resolve_dtype(b_dtype)
        self.c_dtype, self.c_dtype_size = self._resolve_dtype(c_dtype)

        self.base_m = BASE_M_DEFAULT
        self.base_n = BASE_N_DEFAULT
        self.base_k = BASE_K_DEFAULT
        self.step_ka = 4
        self.step_kb = 4
        self.depth_a1 = 8
        self.depth_b1 = 8
        self.used_core_num = AIC_NUM
        self.weight_no_l2_cache = False
        self.group_list_type = group_list_type
        # Finalized by _finalize() during _compute() below.
        self.k_l1 = 0
        self.l1_buffer_num = DB_SIZE
        self.l0c_db = 1

        self._compute()
        self.a_full_load = False

    def _optimize_base_m_and_a_full_load(self, group_m_sizes):
        """Reduce padded M rows and reuse full-K A tiles for equal-sized groups."""
        if not group_m_sizes or len(set(group_m_sizes)) != 1:
            return
        group_m = group_m_sizes[0]
        if group_m and self.base_m >= 64:
            # Keep the M tile count unchanged, avoiding extra B loads.
            m_tile_count = _ceil_div(group_m, self.base_m)
            candidate_base_m = max(
                64, _ceil_align(_ceil_div(group_m, m_tile_count), BASIC_BLOCK_16)
            )
            while (
                candidate_base_m < self.base_m
                and _ceil_div(group_m, candidate_base_m) > m_tile_count
            ):
                candidate_base_m += BASIC_BLOCK_16
            if (
                candidate_base_m < self.base_m
                and _ceil_div(group_m, candidate_base_m) == m_tile_count
            ):
                self.base_m = candidate_base_m
                self._recalc_base_k()
                self._cal_l1_tiling()
        if (
            group_m
            and self.k > 0
            and self.group_type == SPLIT_M
            and not self.trans_a
            and not self.trans_b
            and _ceil_div(group_m, self.base_m) * self.group_num >= AIC_NUM
            and self.n > self.base_n
        ):
            # Keep one full-K A tile while double-buffering streamed B tiles.
            a_full_load_bytes = (
                self.base_m * _ceil_align(self.k, BASIC_BLOCK_16) * self.a_dtype_size
            )
            max_b_k_l1 = _floor_align(
                max(0, L1_SIZE - a_full_load_bytes)
                // (DB_SIZE * self.base_n * self.b_dtype_size),
                self.base_k,
            )
            m_tiles = _ceil_div(group_m, self.base_m)
            n_tiles = _ceil_div(self.n, self.base_n)
            # Estimate A+B row traffic per core; K and element size are common factors.
            streaming_load_rows = _ceil_div(
                m_tiles * n_tiles * self.group_num, AIC_NUM
            ) * (self.base_m + self.base_n)
            a_full_load_rows = _ceil_div(m_tiles * self.group_num, AIC_NUM) * (
                self.base_m + n_tiles * self.base_n
            )
            # Avoid extra B K windows when multiple M tiles reuse the weights.
            if (
                max_b_k_l1 >= 2 * self.base_k
                and (
                    m_tiles == 1
                    or max_b_k_l1 >= self.k_l1
                    or (
                        self.n >= group_m
                        and _ceil_div(self.k, max_b_k_l1)
                        <= _ceil_div(self.k, self.k_l1)
                    )
                )
                and a_full_load_rows < streaming_load_rows
            ):
                self.a_full_load = True
                self.k_l1 = min(self.k_l1, max_b_k_l1)
                self.l1_buffer_num = DB_SIZE

    @staticmethod
    def _resolve_dtype(dtype):
        if isinstance(dtype, torch.dtype):
            props = _TORCH_DTYPE_TO_DSL.get(dtype)
        else:
            props = None
            for _, (dd, ds) in _TORCH_DTYPE_TO_DSL.items():
                if dd == dtype:
                    props = (dd, ds)
                    break
        if props is None:
            raise TypeError(f"unsupported dtype: {dtype}")
        return props

    def _optimize_base_m(self):
        """Reduce M padding and increase tile count for L0C buffer reuse.

        Clamp base_m to the aligned maximum group M when it is smaller.
        When the tile count is between half and all AICs, split M further
        if each group spans multiple M tiles, creating more work for the buffers.
        """
        if 0 < self.m < self.base_m:
            self.base_m = _ceil_align(self.m, BASIC_BLOCK_16)
            self._recalc_base_k()
            self._cal_l1_tiling()
        m_tile_count = _ceil_div(self.m, self.base_m)
        n_tile_count = _ceil_div(self.n, self.base_n)
        total_tile_count = m_tile_count * n_tile_count * self.group_num
        if AIC_NUM // 2 < total_tile_count <= AIC_NUM and m_tile_count >= 2:
            self.base_m = _ceil_align(
                _ceil_div(self.m, m_tile_count * 2), BASIC_BLOCK_16
            )
            self._recalc_base_k()
            self._cal_l1_tiling()

    def _compute(self):
        self._cal_base_mm_tiling()
        if self.group_num == 1:
            self._formulate_basic_block(AIC_NUM)
            self._optimize_base_m()
            self._cal_aswt_l1_tiling()
            self._set_disable_l2_cache()
        else:
            if self.group_type in (SPLIT_M, NO_SPLIT):
                self._formulate_basic_block(max(1, AIC_NUM // self.group_num))
                self._optimize_base_m()
                self._set_disable_l2_cache()
            elif self.group_type == SPLIT_K:
                remain_core = max(1, AIC_NUM // self.group_num)
                self._formulate_basic_block(remain_core)
        self._finalize()

    def _cal_base_mm_tiling(self):
        """Derive initial base_k/base_m from L0B/L0A/L0C capacity."""
        self.base_k = (L0B_SIZE // DB_SIZE) // (self.base_n * self.a_dtype_size)
        self.base_k = self.base_k & ~ALIGN_DOWN_16
        if self.base_k == 0:
            self.base_k = BASIC_BLOCK_16

        max_base_m = L0C_SIZE // (self.base_n * DATA_SIZE_FP32)
        self.base_m = min(
            (L0A_SIZE // DB_SIZE) // (self.base_k * self.a_dtype_size),
            max_base_m,
        )
        if self.base_m > BASE_M_DEFAULT:
            self.base_m = BASE_M_DEFAULT
        if self.base_m == 0:
            self.base_m = BASIC_BLOCK_16

        self._cal_l1_tiling()

    def _cal_l1_tiling(self):
        """Split L1 between A/B and compute the step_k/depth values."""
        total_l1 = L1_SIZE
        l1_a_size = (
            PARTA_L1_SIZE if self.base_m > self.base_n else (total_l1 - PARTA_L1_SIZE)
        )
        l1_b_size = total_l1 - l1_a_size

        self.step_ka = (
            l1_a_size // DB_SIZE // self.base_m // self.base_k // self.a_dtype_size
        )
        self.step_kb = (
            l1_b_size // DB_SIZE // self.base_n // self.base_k // self.b_dtype_size
        )

        if self.step_ka == 0:
            self.step_ka = 1
        if self.step_kb == 0:
            self.step_kb = 1

        if self.step_ka >= self.step_kb:
            self.step_ka = self.step_ka // self.step_kb * self.step_kb
        else:
            self.step_kb = self.step_kb // self.step_ka * self.step_ka

        self.depth_a1 = self.step_ka * DB_SIZE
        self.depth_b1 = self.step_kb * DB_SIZE

    def _formulate_basic_block(self, remain_core_num):
        """Rebalance base_m/base_n for core utilization when tiles are scarce."""
        m_cnt = _ceil_div(self.m, self.base_m)
        n_cnt = _ceil_div(self.n, self.base_n)

        if m_cnt * n_cnt >= remain_core_num:
            return

        if m_cnt <= n_cnt:
            self.base_m = _ceil_align(_ceil_div(self.m, m_cnt), BASIC_BLOCK_16)
            m_cnt = _ceil_div(self.m, self.base_m)
            n_cnt = remain_core_num // m_cnt if m_cnt > 0 else 1
            self.base_n = _ceil_align(_ceil_div(self.n, n_cnt), BASIC_BLOCK_16)
        else:
            self.base_n = _ceil_align(_ceil_div(self.n, n_cnt), BASIC_BLOCK_16)
            n_cnt = _ceil_div(self.n, self.base_n)
            m_cnt = remain_core_num // n_cnt if n_cnt > 0 else 1
            self.base_m = _ceil_align(_ceil_div(self.m, m_cnt), BASIC_BLOCK_16)

        while (
            self.base_n >= self.base_m * NUM_TWO and n_cnt < remain_core_num // NUM_TWO
        ):
            n_cnt = n_cnt * NUM_TWO
            m_cnt = remain_core_num // n_cnt if n_cnt > 0 else 1
            self.base_m = _ceil_align(_ceil_div(self.m, m_cnt), BASIC_BLOCK_16)
            self.base_n = _ceil_align(_ceil_div(self.n, n_cnt), BASIC_BLOCK_16)
            m_cnt = _ceil_div(self.m, self.base_m)
            n_cnt = _ceil_div(self.n, self.base_n)

        while (
            self.base_m >= self.base_n * NUM_TWO and m_cnt < remain_core_num // NUM_TWO
        ):
            m_cnt = m_cnt * NUM_TWO
            n_cnt = remain_core_num // m_cnt if m_cnt > 0 else 1
            self.base_m = _ceil_align(_ceil_div(self.m, m_cnt), BASIC_BLOCK_16)
            self.base_n = _ceil_align(_ceil_div(self.n, n_cnt), BASIC_BLOCK_16)
            m_cnt = _ceil_div(self.m, self.base_m)
            n_cnt = _ceil_div(self.n, self.base_n)

        m_cnt = _ceil_div(self.m, self.base_m)
        n_cnt = _ceil_div(self.n, self.base_n)
        self.used_core_num = min(m_cnt * n_cnt * self.group_num, AIC_NUM)
        self._recalc_base_k()

    def _recalc_base_k(self):
        """Recompute the base_k cap after base_m/base_n rebalancing."""
        k_align = _ceil_align(self.k, BASIC_BLOCK_16)
        k_max = _floor_align(
            L0A_SIZE // DB_SIZE // self.a_dtype_size // max(self.base_m, self.base_n),
            BASIC_BLOCK_16,
        )
        self.base_k = min(k_align, k_max)
        if self.base_k == 0:
            self.base_k = BASIC_BLOCK_16

    def _cal_aswt_l1_tiling(self):
        """Single-group L1 tiling using the full L1 budget plus a small reserve."""
        total_l1 = L1_SIZE + 256

        depth_a1 = (
            total_l1 // NUM_TWO // self.base_m // self.base_k // self.a_dtype_size
        )
        depth_b1 = (
            total_l1 // NUM_TWO // self.base_n // self.base_k // self.b_dtype_size
        )

        depth_a_size = depth_a1 * self.base_m * self.base_k * self.a_dtype_size
        depth_b_size = depth_b1 * self.base_n * self.base_k * self.b_dtype_size

        if depth_a_size + depth_b_size > total_l1:
            if self.base_m <= self.base_n:
                depth_a1 = max(depth_a1 // NUM_TWO, 1)
            else:
                depth_b1 = max(depth_b1 // NUM_TWO, 1)

        self.step_ka = max(depth_a1 // DB_SIZE, 1)
        self.step_kb = max(depth_b1 // DB_SIZE, 1)

        if self.step_ka >= self.step_kb:
            self.step_ka = self.step_ka // self.step_kb * self.step_kb
        else:
            self.step_kb = self.step_kb // self.step_ka * self.step_ka

        self.depth_a1 = self.step_ka * DB_SIZE
        self.depth_b1 = self.step_kb * DB_SIZE

    def _set_disable_l2_cache(self):
        """Weight DMA L2 disable: data exceeds L2, 128B-aligned, seed true."""
        inner_b = self.k if self.trans_b else self.n
        dtype_size = self.a_dtype_size
        flag = (
            (self.base_k * self.step_kb * dtype_size % ALIGN_128 == 0)
            if self.trans_b
            else (self.base_n * dtype_size % ALIGN_128 == 0)
        )
        total_size = (
            self.m * self.k * dtype_size
            + self.group_num * self.k * self.n * dtype_size
            + self.m * self.n * dtype_size
        )
        if total_size < L2_SIZE:
            self.weight_no_l2_cache = False
            return
        self.weight_no_l2_cache = (
            (inner_b * dtype_size % ALIGN_128 == 0) and flag and self.l2_disable_seed
        )
        # Keep B caching enabled when multiple M tiles can reuse the weights.
        if self.weight_no_l2_cache and _ceil_div(self.m, self.base_m) > 1:
            self.weight_no_l2_cache = False

    def _finalize(self):
        """Finalize k_l1/used_core_num/l1_buffer_num."""
        self.k_l1 = min(
            self.base_k * min(self.step_ka, self.step_kb),
            _ceil_align(self.k, BASIC_BLOCK_16),
        )

        if self.group_num == 1:
            m_cnt = _ceil_div(self.m, self.base_m)
            n_cnt = _ceil_div(self.n, self.base_n)
            self.used_core_num = min(m_cnt * n_cnt, AIC_NUM)

        a_l1_4buf = self.k_l1 * self.base_m * self.a_dtype_size * self.depth_a1
        b_l1_4buf = self.k_l1 * self.base_n * self.b_dtype_size * self.depth_b1

        if a_l1_4buf + b_l1_4buf > L1_SIZE:
            self.l1_buffer_num = DB_SIZE
        else:
            self.l1_buffer_num = min(self.depth_a1, self.depth_b1, 4)
            if self.l1_buffer_num < DB_SIZE:
                self.l1_buffer_num = DB_SIZE

        # Use two L0C slots when they fit, allowing output/computation overlap.
        self.l0c_db = (
            2 if self.base_m * self.base_n * DATA_SIZE_FP32 * 2 <= L0C_SIZE else 1
        )


# ============================================================================
# 2. BlockMmad
# ============================================================================


class BlockMmad:
    """Per-tile pipeline: GM → L1 → L0A/L0B → MMAD → L0C → GM.

    Four layout paths selected at compile time by trans_a/trans_b:

      trans_a off: L1_A (base_m, k_l1)  <- GM [M,K]
      trans_a on:  L1_A (k_l1, base_m)  <- GM [K,M]
      trans_b off: L1_B (k_l1, base_n)  <- GM [K,N]
      trans_b on:  L1_B (base_n, k_l1)  <- GM [N,K]
    """

    def __init__(self, tiling, *, fixed_k=False):
        self.t = tiling
        # A single X split along M has the same host-known K in every group.
        # Preserve full K panels through copy results instead of turning their
        # size into a runtime min/subtraction chain. Split-K and tensor lists
        # can have different K per group and retain the bounded tail path.
        self.full_k_panels = fixed_k and tiling.k % tiling.k_l1 == 0
        a_l1_k = (
            _ceil_align(tiling.k, BASIC_BLOCK_16) if tiling.a_full_load else tiling.k_l1
        )

        self.l1_a = RingBuffer(
            MemLoc.L1,
            tiling.a_dtype,
            make_layout(
                (a_l1_k, tiling.base_m) if tiling.trans_a else (tiling.base_m, a_l1_k)
            ),
            depth=1 if tiling.a_full_load else tiling.l1_buffer_num,
        )

        self.l1_b = RingBuffer(
            MemLoc.L1,
            tiling.b_dtype,
            make_layout(
                (tiling.base_n, tiling.k_l1)
                if tiling.trans_b
                else (tiling.k_l1, tiling.base_n)
            ),
            depth=tiling.l1_buffer_num,
        )

        self.l0a = RingBuffer(
            MemLoc.L0A,
            tiling.a_dtype,
            make_layout((tiling.base_m, tiling.base_k)),
            depth=2,
        )
        self.l0b = RingBuffer(
            MemLoc.L0B,
            tiling.b_dtype,
            make_layout((tiling.base_n, tiling.base_k)),
            depth=2,
        )
        self.l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((tiling.base_m, tiling.base_n)),
            # UnitFlag mainline: the FIXPIPE is auto-triggered by the final
            # MMAD, so L0C stays single-buffered.
            depth=1,
        )
        # Keep output stores cached in L2; unit_flag=3 pairs the FIXPIPE
        # with the final MMAD's auto-trigger.
        self.fixpipe_options = copy_options(unit_flag=3, l2_cache_ctl=1)

    @jit
    def compute_tile(
        self,
        a_src,
        b_src,
        c_src,
        m_tile_idx,
        n_tile_idx,
        k_l1_tiles,
        k_total,
        disable_l2,
    ):
        """K reduction (GM→L1→L0→MMAD) + FIXPIPE for one output tile.

        a_src/b_src are this group's row-major aliases; c_src is the group's
        output slice (M/N/K tails are clipped once per enclosing loop). disable_l2 is a
        runtime per-group flag: groups with M at most base_m skip caching
        weight lines in L2; the copy accepts a runtime cache policy.
        """
        t = self.t
        base_m = t.base_m
        base_n = t.base_n
        base_k = t.base_k
        k_l1 = t.k_l1

        # Select M/N once. The reduction only changes K, so keep the live
        # row/column domains instead of repeatedly clipping a two-axis tile.
        m_start = m_tile_idx * base_m
        n_start = n_tile_idx * base_n
        m_extent = min(base_m, a_src.shape[-1 if t.trans_a else -2] - m_start)
        n_extent = min(base_n, b_src.shape[-2 if t.trans_b else -1] - n_start)
        if const_expr(t.trans_a):
            a_panel = a_src[None, m_start : m_start + m_extent]
        else:
            a_panel = a_src[m_start : m_start + m_extent, None]
        if const_expr(t.trans_b):
            b_panel = b_src[n_start : n_start + n_extent, None]
        else:
            b_panel = b_src[None, n_start : n_start + n_extent]
        # M/N and the physical accumulator pitch stay fixed for this entire
        # reduction. Bind them once; only the operand views advance along K.
        accumulator = self.l0c.next_write(bare=True).view(
            make_layout((m_extent, n_extent)), dtype=dtypes.float32, data_format="nz"
        )
        for k_l1_idx in range(k_l1_tiles):
            k_start = k_l1_idx * k_l1
            k_extent = (
                k_l1 if const_expr(self.full_k_panels) else min(k_l1, k_total - k_start)
            )
            if const_expr(t.trans_a):
                gm_a_tile = a_panel[k_start : k_start + k_extent, None]
            else:
                gm_a_tile = a_panel[None, k_start : k_start + k_extent]
            if const_expr(not t.a_full_load):
                l1_a_tensor = self.l1_a.next_write()
                l1_a_tensor = l1_a_tensor[tuple(map(slice, gm_a_tile.shape))]
                mem_copy(l1_a_tensor, gm_a_tile)
            else:
                l1_a_tensor = self.l1_a.next_read()[:m_extent, None]

            if const_expr(t.trans_b):
                gm_b_tile = b_panel[None, k_start : k_start + k_extent]
            else:
                gm_b_tile = b_panel[k_start : k_start + k_extent, None]

            # The current copy API accepts a per-call runtime cache policy.
            l1_b_tensor = self.l1_b.next_write()
            l1_b_tensor = l1_b_tensor[tuple(map(slice, gm_b_tile.shape))]
            mem_copy(
                l1_b_tensor,
                gm_b_tile,
                options=copy_options(l2_cache_ctl=4 if disable_l2 else 1),
            )

            k_l0_per_l1 = _ceil_div(k_extent, base_k)
            for k_l0_idx in range(k_l0_per_l1):
                a_k_l0_idx = (
                    k_l1_idx * (k_l1 // base_k) + k_l0_idx
                    if const_expr(t.a_full_load)
                    else k_l0_idx
                )
                # Keep the nominal K tile explicit for NZ alignment proof.
                # Interval slicing would hide that the origin is a multiple
                # of base_k and leave a composed map for transposed operands.
                if const_expr(t.trans_a):
                    l1_a_slice = tile_slice(
                        l1_a_tensor, (base_k, base_m), (a_k_l0_idx, 0)
                    )
                else:
                    l1_a_slice = tile_slice(
                        l1_a_tensor, (base_m, base_k), (0, a_k_l0_idx)
                    )
                if const_expr(t.trans_b):
                    l1_b_slice = tile_slice(
                        l1_b_tensor, (base_n, base_k), (0, k_l0_idx)
                    )
                else:
                    l1_b_slice = tile_slice(
                        l1_b_tensor, (base_k, base_n), (k_l0_idx, 0)
                    )
                is_first_k_block = k_l1_idx == 0 and k_l0_idx == 0
                l0a_tensor = self._load_l0a_block(l1_a_slice)
                l0b_tensor = self._load_l0b_block(l1_b_slice)
                is_final_k = k_l1_idx == k_l1_tiles - 1 and k_l0_idx == k_l0_per_l1 - 1
                matmul(
                    accumulator.bare(),
                    l0a_tensor,
                    l0b_tensor,
                    init=is_first_k_block,
                    unit_flag=3 if is_final_k else 2,
                )
        gm_c_tile = c_src[m_start : m_start + m_extent, n_start : n_start + n_extent]
        mem_copy(gm_c_tile, accumulator, options=self.fixpipe_options)

    @jit
    def _load_l0a_block(self, l1_a_slice):
        """Load one base_k A block from L1 to L0A.

        K-major A staging (trans_a) is un-transposed here by the permuted
        source view; the L0A slot is bound to the source's actual extent.
        """
        source = l1_a_slice.permute(1, 0) if self.t.trans_a else l1_a_slice
        l0a_tensor = self.l0a.next_write(bare=True)
        l0a_tensor = l0a_tensor.view(make_layout(source.shape))
        mem_copy(l0a_tensor, source)
        return l0a_tensor

    @jit
    def _load_l0b_block(self, l1_b_slice):
        """Load one base_k B block from L1 to L0B.

        (K, N) B staging (not trans_b) is transposed here by the permuted
        source view; the L0B slot is bound to the source's actual extent.
        """
        source = l1_b_slice if self.t.trans_b else l1_b_slice.permute(1, 0)
        l0b_tensor = self.l0b.next_write(bare=True)
        l0b_tensor = l0b_tensor.view(make_layout(source.shape))
        mem_copy(l0b_tensor, source)
        return l0b_tensor


# ============================================================================
# 3. GmmKernel
# ============================================================================


class GmmKernel:
    """Unified kernel: per-group iteration + tile dispatch, tensorlist in/out.

    Split of duties:
      - gmm_kernel (@kernel): fetches the tensorlist element per group and
        reads its shape (DSL restricts element access to @kernel bodies);
      - _iterate_group (@jit): reads group_list for this group's extent,
        builds the group's A/B/C aliases, schedules tiles (sliding window
        WINDOW_LEN=4 with even-row N reversal) and runs BlockMmad per tile.
        Zero-size groups are skipped; the tile counter carries across groups
        for load balancing.
    """

    def __init__(
        self,
        tiling,
        single_x=True,
        single_w=True,
        single_y=True,
        has_group_list=True,
        w3d=False,
    ):
        self.t = tiling
        self.is_split_k = tiling.is_split_k
        self.is_count = tiling.group_list_type == GROUPLIST_TYPE_COUNT
        self.single_x = single_x
        self.single_w = single_w
        self.single_y = single_y
        self.has_group_list = has_group_list
        self.w3d = w3d
        self.weight_no_l2_cache = tiling.weight_no_l2_cache

    @kernel
    def gmm_kernel(
        self,
        a_list: cannbotdsl.TensorList,
        b_list: cannbotdsl.TensorList,
        group_list_gm,
        c_list: cannbotdsl.TensorList,
        group_num,
    ):
        t = self.t
        single_x = self.single_x
        single_w = self.single_w
        single_y = self.single_y

        cur_block_idx = get_block_idx()
        block_num = get_block_num()

        # Loop-carried scheduler state: count is the cross-group tile
        # counter, pre_offset the accumulated group-axis offset.
        count = 0
        pre_offset = 0
        # Preserve ring cursors across groups, including odd K-window counts.
        block_mmad = BlockMmad(t, fixed_k=self.single_x and not self.is_split_k)
        saved_ctrl = get_ctrl()
        self._enable_mmad_row_major(saved_ctrl)

        for group_idx in range(group_num):
            # Element access must stay inside the @kernel body (DSL
            # restriction); element choice depends only on single/multi
            # tensor, not on layout.
            x_elem = a_list[0] if const_expr(single_x) else a_list[group_idx]
            w_elem = b_list[0] if const_expr(single_w) else b_list[group_idx]
            y_elem = c_list[0] if const_expr(single_y) else c_list[group_idx]

            # Layout normalization, driven purely by the transpose flags:
            # after this block both operands are in their logical layouts
            # (x: [M,K]; weight: [..,K,N])
            a_elem = permute(x_elem, (1, 0)) if const_expr(t.trans_a) else x_elem
            x_m = x_elem.shape[-2]
            x_k = x_elem.shape[-1]
            if const_expr(t.trans_b):
                b_k = w_elem.shape[-1]
                b_n = w_elem.shape[-2]
            else:
                b_k = w_elem.shape[-2]
                b_n = w_elem.shape[-1]

            # Everything else lives in the @jit per-group body.
            count, pre_offset = self._iterate_group(
                block_mmad,
                a_elem,
                w_elem,
                y_elem,
                group_list_gm,
                group_idx,
                x_m,
                x_k,
                b_k,
                b_n,
                count,
                pre_offset,
                cur_block_idx,
                block_num,
            )

        self._restore_ctrl(saved_ctrl)

    @jit
    def _enable_mmad_row_major(self, saved_ctrl):
        """Set CTRL.MM_LAYOUT_MODE_BIT for the UnitFlag MMAD mainline.

        AscendC SetMMRowMajor: the layout-mode bit is 51; set it only when
        the saved CTRL does not already carry it.
        """
        mm_layout_mode_mask = 1 << 51
        set_ctrl(
            saved_ctrl
            + (1 - saved_ctrl // mm_layout_mode_mask % 2) * mm_layout_mode_mask
        )

    @jit
    def _restore_ctrl(self, saved_ctrl):
        """Restore the CTRL captured before the row-major mainline."""
        set_ctrl(saved_ctrl)

    @jit
    def _iterate_group(
        self,
        block_mmad,
        a_elem,
        w_elem,
        y_elem,
        group_list_gm,
        group_idx,
        x_m,
        x_k,
        b_k,
        b_n,
        count,
        pre_offset,
        cur_block_idx,
        block_num,
    ):
        """Per-group iteration for elements already fetched by the kernel.

        Returns (count, pre_offset): the cross-group tile counter and the
        accumulated group-axis offset.
        """
        t = self.t
        is_split_k = self.is_split_k
        is_count = self.is_count
        single_x = self.single_x
        single_w = self.single_w
        single_y = self.single_y
        has_group_list = self.has_group_list

        # This group's extent on the split axis.
        if const_expr(has_group_list):
            group_list_val = group_list_gm[group_idx]
            split_value = group_list_val if is_count else group_list_val - pre_offset
        else:
            # group_list absent: the extent comes from element shapes —
            # SPLIT_K reads it from the weight element, otherwise from x.
            split_value = b_k if is_split_k else x_m

        group_start = pre_offset
        group_end = pre_offset + split_value

        # Effective M/N/K for this group.
        m_val = split_value if not is_split_k else x_m
        k_val = split_value if is_split_k else x_k
        n_val = b_n

        # Accumulate the group-axis offset.
        new_pre_offset = group_end

        # Skip zero-size groups.
        if m_val > 0 and k_val > 0 and n_val > 0:
            # FE202 compliance: views assigned in if-else branches are
            # pre-initialized with full-extent keep views (zero-copy).
            a_group = a_elem[None]
            b_src = w_elem[None]
            c_group = y_elem[None]

            # x: always owns the split axis (M or K per the group type);
            # multi-x carries one element per group.
            if const_expr(single_x):
                a_group = a_elem[group_start:group_end, None]
            else:
                a_group = a_elem

            # weight: multi -> one element per group; 3-D -> dim 0 is the
            # group axis; 2-D owns the split axis only under SPLIT_K (K
            # rows), otherwise it is shared by all groups as-is.
            if const_expr(not single_w):
                b_src = w_elem
            elif const_expr(self.w3d):
                b_src = w_elem[group_idx, None, None]
            elif is_split_k:
                b_src = w_elem[group_start:group_end, None]
            else:
                b_src = w_elem

            # y: multi -> one element per group; single under SPLIT_K is
            # 3-D [G,M,N] with dim 0 as the group axis; single under
            # SPLIT_M/NO_SPLIT owns the split axis (M rows).
            if const_expr(not single_y):
                c_group = y_elem
            elif is_split_k:
                c_group = y_elem[group_idx, None, None]
            else:
                c_group = y_elem[group_start:group_end, None]

            # Selecting a group keeps its leading singleton axis. Collapse
            # that axis so the tile body sees the same two-dimensional planes
            # for tensor lists and grouped three-dimensional storage.
            b_src = b_src.flatten(0, -2)
            c_group = c_group.flatten(0, -2)

            k_total = k_val
            k_l1_tiles = ceil_div(k_val, t.k_l1)
            m_tile_num = ceil_div(m_val, t.base_m)
            n_tile_num = ceil_div(n_val, t.base_n)
            total_tile_num = m_tile_num * n_tile_num
            if const_expr(t.a_full_load):
                total_tile_num = m_tile_num
            cur_count = count + total_tile_num

            cur_block = (
                cur_block_idx if cur_block_idx >= count else cur_block_idx + block_num
            )

            # Sliding-window scheduler parameters.
            main_window = WINDOW_LEN if WINDOW_LEN < m_tile_num else m_tile_num
            main_row = m_tile_num // main_window - 1
            tail_window = m_tile_num - main_window * main_row

            # Tile loop for this group.
            for cur_block_inner in range(cur_block, cur_count, block_num):
                index = cur_block_inner - count

                row_idx = index // n_tile_num // main_window

                m_tile_idx = 0
                n_tile_idx = 0

                if row_idx < main_row:
                    m_tile_idx = row_idx * main_window + index % main_window
                    n_tile_idx = (index // main_window) % n_tile_num
                else:
                    row_idx = main_row
                    tail_index = index - main_row * main_window * n_tile_num
                    m_tile_idx = main_row * main_window + tail_index % tail_window
                    n_tile_idx = (tail_index // tail_window) % n_tile_num

                # Even-row N reversal.
                if row_idx % EVEN_ROWS != 0:
                    n_tile_idx = n_tile_num - 1 - n_tile_idx

                # NOTALLOC_KEEP for B streamed once by a single M tile.
                disable_l2 = self.weight_no_l2_cache and t.base_m >= m_val
                if const_expr(t.a_full_load):
                    # One task owns an M tile and reuses its resident A across N.
                    m_tile_idx = index
                    gm_a_full = tile_slice(
                        a_group,
                        (t.base_m, _ceil_align(t.k, BASIC_BLOCK_16)),
                        (m_tile_idx, 0),
                    )
                    mem_copy(block_mmad.l1_a.next_write(), gm_a_full)
                    for full_n_idx in range(n_tile_num):
                        block_mmad.compute_tile(
                            a_group,
                            b_src,
                            c_group,
                            m_tile_idx,
                            full_n_idx,
                            k_l1_tiles,
                            k_total,
                            disable_l2,
                        )
                else:
                    block_mmad.compute_tile(
                        a_group,
                        b_src,
                        c_group,
                        m_tile_idx,
                        n_tile_idx,
                        k_l1_tiles,
                        k_total,
                        disable_l2,
                    )

            count = cur_count % block_num
        return count, new_pre_offset

    @host
    def run(
        self,
        a_list: cannbotdsl.TensorList,
        b_list: cannbotdsl.TensorList,
        group_list_gm,
        c_list: cannbotdsl.TensorList,
        group_num,
    ):
        self.gmm_kernel[self.t.used_core_num](
            a_list, b_list, group_list_gm, c_list, group_num
        )


# ============================================================================
# 4. Torch Interface
# ============================================================================


@dataclasses.dataclass
class GmmPlan:
    """Resolved scenario facts, produced by _resolve_scenario.

    Consumed by tiling, output allocation and kernel launch; the
    group_matmul body only executes the plan.
    """

    group_num: int  # group count
    m: int  # max M (tiling input)
    n: int  # max N
    k: int  # max K (k_total for SPLIT_K)
    single_m: int  # single-x M (None for multi-x)
    m_list: list = None  # per-group M for multi-x
    n_list: list = None  # per-group N for multi-weight
    trans_a: bool = False
    trans_b: bool = False
    w3d: bool = False  # S2 single 3-D weight
    single_x: bool = True
    single_w: bool = True
    single_y: bool = True
    l2_seed: bool = False  # scenario seed of the L2 disable policy


def _validate_basic(x, weight, group_type, group_list_type, split_item, output_dtype):
    """Common contract checks (types, dtype, device, enum values).

    Returns (x0, w0, dev, single_x, single_w, single_y); single_y is the
    output form derived from split_item (0/1 multi, 2/3 single).
    """
    if not isinstance(x, (list, tuple)) or not isinstance(weight, (list, tuple)):
        raise TypeError("x and weight must be lists of Tensors")
    if len(x) == 0 or len(weight) == 0:
        raise ValueError("x and weight must be non-empty lists")
    if not all(isinstance(t, torch.Tensor) for t in x):
        raise TypeError("x must be a list of Tensors")
    if not all(isinstance(t, torch.Tensor) for t in weight):
        raise TypeError("weight must be a list of Tensors")

    x0, w0 = x[0], weight[0]
    if x0.dtype not in _TORCH_DTYPE_TO_DSL:
        raise TypeError("only torch.float16 and torch.bfloat16 are supported")
    if any(t.dtype != x0.dtype for t in x):
        raise TypeError("all x tensors must share the same dtype")
    if any(t.dtype != x0.dtype for t in weight):
        raise TypeError("weight dtype must match x dtype")
    dev = x0.device
    if any(t.device != dev for t in x) or any(t.device != dev for t in weight):
        raise ValueError("all tensors must be on the same device")
    if output_dtype is not None and output_dtype != x0.dtype:
        raise ValueError("output_dtype must be None or x dtype")
    if group_type not in (NO_SPLIT, SPLIT_M, SPLIT_K):
        raise ValueError("group_type must be -1, 0 (SPLIT_M) or 2 (SPLIT_K)")
    if len(x) > MAX_TENSOR or len(weight) > MAX_TENSOR:
        raise ValueError(f"x/weight tensor count must be within [1, {MAX_TENSOR}]")
    if group_list_type not in (GROUPLIST_TYPE_CUMSUM, GROUPLIST_TYPE_COUNT):
        raise ValueError("group_list_type must be 0 (cumsum) or 1 (count)")

    single_y = split_item in (SPLIT_ITEM_NO_SEPARATED, 3)
    if split_item not in (SPLIT_ITEM_SEPARATED, 1, SPLIT_ITEM_NO_SEPARATED, 3):
        raise ValueError("split_item must be 0/1 (multi output) or 2/3 (single output)")

    return x0, w0, dev, len(x) == 1, len(weight) == 1, single_y


def _validate_group_list(group_list, dev):
    """group_list contract checks.  Returns True when present."""
    if group_list is None:
        return False
    if not isinstance(group_list, torch.Tensor):
        raise ValueError("group_list must be a Tensor")
    if group_list.dtype != torch.int64:
        raise TypeError("group_list must be torch.int64")
    if group_list.device != dev:
        raise ValueError("group_list must be on the same device as x")
    if group_list.dim() != 1 or group_list.numel() == 0:
        raise ValueError("group_list must be a non-empty 1-D int64 tensor")
    return True


def _resolve_transpose_flags(x, weight, single_x, is_split_k):
    """Transpose flags, derived uniformly from the input views.

    trans_a / trans_b simply report whether each operand is a canonical
    ``transpose(-1, -2)`` view — independent of the group-type scenario.
    Scenario contracts are only validated: SPLIT_K requires a transposed
    x and non-transposed weights; other scenarios require contiguous x.
    Weights must share a uniform transpose state.

    Returns (trans_a, trans_b).
    """
    # x: uniform view detection; scenario contract validated alongside.
    trans_a = _is_canonical_last_two_transpose(x[0], "x[0]")
    for g_idx, xg in enumerate(x):
        if xg.dim() != 2:
            suffix = " for SPLIT_K" if is_split_k else f", got {xg.dim()}-D"
            raise ValueError(f"x[{g_idx}] must be 2-D" + suffix)
        xg_transposed = _is_canonical_last_two_transpose(xg, f"x[{g_idx}]")
        if xg_transposed != trans_a:
            raise ValueError("x tensors must share a uniform transpose state")
        if is_split_k:
            if not xg_transposed:
                raise ValueError(
                    "SPLIT_K x must be a canonical transpose(-1,-2) view (npu contract)"
                )
        elif xg_transposed or not xg.is_contiguous():
            raise ValueError(f"x[{g_idx}] must be contiguous")
    if is_split_k and not single_x:
        raise ValueError("SPLIT_K requires single x tensor")

    # weight: uniform view detection; SPLIT_K requires non-transposed.
    trans_b = _is_canonical_last_two_transpose(weight[0], "weight[0]")
    for g_idx, wg in enumerate(weight):
        wg_transposed = _is_canonical_last_two_transpose(wg, f"weight[{g_idx}]")
        if wg_transposed != trans_b:
            raise ValueError("weight tensors must share a uniform transpose state")
        if is_split_k and wg_transposed:
            raise ValueError("SPLIT_K weight must be contiguous (non-transposed)")
    return trans_a, trans_b


def _resolve_scenario(
    x,
    weight,
    x0,
    w0,
    group_list,
    group_list_type,
    group_type,
    single_x,
    single_w,
    single_y,
    is_split_k,
    w3d,
    has_group_list,
    trans_a,
    trans_b,
):
    """Scenario dispatch and shape contracts (S1-S6).

    Returns a GmmPlan with the max-statistics m/n/k for tiling and the
    facts needed for output allocation (including the weight-L2 disable
    seed, derived at the end of this function).
    """

    def gl_len_ok(g):
        if has_group_list and len(group_list) != g:
            raise ValueError(
                f"group_list length ({len(group_list)}) != group count ({g})"
            )

    def gl_total_ok(total, axis):
        if has_group_list:
            values = [int(v) for v in group_list.tolist()]
            if group_list_type == GROUPLIST_TYPE_CUMSUM:
                # cumsum must be non-negative, non-decreasing, and end at the
                # tensor extent (each difference is a group size >= 0).
                prev = 0
                for v in values:
                    if v < prev:
                        raise ValueError(
                            f"group_list (cumsum) must be non-decreasing; got {values}"
                        )
                    prev = v
                gl_total = values[-1] if values else 0
            else:
                if any(v < 0 for v in values):
                    raise ValueError(
                        f"group_list (count) must be non-negative; got {values}"
                    )
                gl_total = sum(values)
            if gl_total != total:
                raise ValueError(
                    f"group_list total {axis} ({gl_total}) != tensor {axis} ({total})"
                )

    def gl_per_group_ok(sizes, axis):
        """Per-group check: group_list must match the per-group tensor sizes
        exactly (count mode) or reproduce them via cumsum differences."""
        if not has_group_list:
            return
        values = [int(v) for v in group_list.tolist()]
        gl_sizes = (
            values
            if group_list_type == GROUPLIST_TYPE_COUNT
            else [b - a for a, b in zip([0] + values[:-1], values)]
        )
        if gl_sizes != list(sizes):
            raise ValueError(
                f"group_list per-group {axis} ({gl_sizes}) != tensor sizes ({list(sizes)})"
            )

    group_num = len(group_list) if has_group_list else None
    m_list = n_list = k_list = None
    single_m = k_total = None

    if is_split_k:
        single_m = x0.shape[-2]
        k_total = x0.shape[-1]
        if not single_w:
            # S6: multi output; the weight elements carry the K split.
            group_num = len(weight)
            if single_y:
                raise ValueError("SPLIT_K + separated weight requires split_item 0/1")
            gl_len_ok(group_num)
            n_list = [wg.shape[-1] for wg in weight]
            k_list = [wg.shape[0] for wg in weight]
            if sum(k_list) != k_total:
                raise ValueError(f"sum(weight K) {sum(k_list)} != x K ({k_total})")
            gl_per_group_ok(k_list, "K")
            m_, n_, k_ = single_m, max(n_list), k_total
        else:
            # S5: single 3-D output; group_list carries the K split.
            n_dim = w0.shape[-1]
            if w0.shape[-2] != k_total:
                raise ValueError(f"weight K {w0.shape[-2]} != x K ({k_total})")
            if group_num is None:
                raise ValueError("SPLIT_K single-weight requires group_list")
            if (
                group_list_type == GROUPLIST_TYPE_CUMSUM
                and int(group_list[-1].item()) != k_total
            ):
                raise ValueError(
                    f"group_list total K ({int(group_list[-1].item())}) != x K ({k_total})"
                )
            m_, n_, k_ = single_m, n_dim, k_total
    elif single_x:
        single_m = x0.shape[0]
        k_dim = x[0].shape[-1]
        if w3d:
            # S2: the group count comes from the weight's leading dim, and
            # group_list is required.
            if not has_group_list:
                raise ValueError("single 3-D weight requires group_list")
            group_num = w0.shape[0]
            n_dim = w0.shape[-1]
            if w0.shape[-2] != k_dim:
                raise ValueError(f"weight K {w0.shape[-2]} != x K {k_dim}")
            gl_len_ok(group_num)
            gl_total_ok(single_m, "M")
        else:
            # S3: the group count equals the number of weight tensors, and
            # group_list is required.
            group_num = len(weight)
            if not has_group_list:
                raise ValueError("single-x multi-weight requires group_list")
            gl_len_ok(group_num)
            n_list = [wg.shape[-1] for wg in weight]
            if any(wg.shape[-2] != k_dim for wg in weight):
                raise ValueError(f"weight K must match x K {k_dim}")
            if any(n != n_list[0] for n in n_list):
                raise ValueError("weight tensors must share the same N")
            n_dim = n_list[0]
            gl_total_ok(single_m, "M")
        m_, n_, k_ = single_m, n_dim, k_dim
        if group_type == SPLIT_M and has_group_list:
            # Tiling plans per-group tiles: feed the max per-group M (not
            # the total M) so core-count rebalancing sizes blocks against
            # per-group work, matching the multi-x path's max(m_list).
            group_list_values = [int(v) for v in group_list.tolist()]
            if group_list_type == GROUPLIST_TYPE_CUMSUM:
                group_m_sizes = [group_list_values[0]] + [
                    group_list_values[i] - group_list_values[i - 1]
                    for i in range(1, len(group_list_values))
                ]
            else:
                group_m_sizes = group_list_values
            if group_m_sizes:
                m_ = max(group_m_sizes)
    else:
        # S1 / S4: group count = len(x) == len(weight).
        if len(x) != len(weight):
            raise ValueError(
                f"separated x requires len(x)==len(weight), got {len(x)} vs {len(weight)}"
            )
        group_num = len(x)
        gl_len_ok(group_num)
        m_list = [xg.shape[0] for xg in x]
        n_list = [wg.shape[-1] for wg in weight]
        k_list = [wg.shape[-2] for wg in weight]
        if group_type == SPLIT_M:
            if len(set(n_list)) > 1:
                raise ValueError("SPLIT_M separated tensors require uniform weight N")
            if single_y and len(set(k_list)) > 1:
                raise ValueError("SPLIT_M single-Y requires uniform weight K")
        if any(xg.shape[1] != k_list[g] for g, xg in enumerate(x)):
            raise ValueError("x[g] K must match weight[g] K")
        gl_total_ok(sum(m_list), "M")
        gl_per_group_ok(m_list, "M")
        m_, n_, k_ = max(m_list), max(n_list), max(k_list)

    # Weight-L2 disable seed, mirroring CANN's per-scenario assignment
    # (SplitMSingleXSingleWeightSingleY → true,
    # SplitMSingleXSeparatedWeight → isSingleY_, all other scenarios keep
    # the false default):
    if is_split_k:
        l2_seed = False  # SPLIT_K: never disable weight L2
    elif w3d and single_y:
        l2_seed = True  # S2 s-s-s: single weight, high reuse
    elif single_x and not single_w and single_y:
        l2_seed = True  # S3 s-m-s: per-group weights, single y
    else:
        l2_seed = False  # S1/S4 multi-x: default

    return GmmPlan(
        group_num=group_num,
        m=m_,
        n=n_,
        k=k_,
        single_m=single_m,
        m_list=m_list,
        n_list=n_list,
        trans_a=trans_a,
        trans_b=trans_b,
        w3d=w3d,
        single_x=single_x,
        single_w=single_w,
        single_y=single_y,
        l2_seed=l2_seed,
    )


def group_matmul(
    x,
    weight,
    *,
    group_list=None,
    group_list_type=0,
    group_type=0,
    split_item=2,
    output_dtype=None,
):
    """Non-quantized GroupedMatmul, tensorlist interface.

    Args:
        x (List[Tensor]): input matrices, passed to the kernel as-is.
        weight (List[Tensor]): weight matrices, passed to the kernel as-is.
        group_list (Tensor | None): (G,) int64, cumsum (type 0) / count
            (type 1); None infers per-group extents from tensor shapes
            (legal for S1/S4/S6 only).
        group_list_type: 0 (cumsum) or 1 (count).
        group_type: -1 (no split) / 0 (M axis) / 2 (K axis).
        split_item: 0/1 (multi output) / 2/3 (single output).
        output_dtype: None or x.dtype.

    Returns:
        List[Tensor]: G outputs for split_item 0/1; a single-element list
        for 2/3 (the element is 3-D [G,M,N] for SPLIT_K) — matching the
        always-list return of torch_npu.npu_grouped_matmul.
    """
    # ---- validation & scenario resolution ------------------------------
    x0, w0, dev, single_x, single_w, single_y = _validate_basic(
        x, weight, group_type, group_list_type, split_item, output_dtype
    )
    has_group_list = _validate_group_list(group_list, dev)
    is_split_k = group_type == SPLIT_K
    w3d = single_w and w0.dim() == 3 and not is_split_k
    # Rank contract: weight is exactly 3-D only for the single-3-D-weight
    # scenario (S2); every other scenario takes 2-D weights.
    expected_rank = 3 if w3d else 2
    for g_idx, wg in enumerate(weight):
        if wg.dim() != expected_rank:
            raise ValueError(
                f"weight[{g_idx}] must be {expected_rank}-D, got {wg.dim()}-D"
            )
    trans_a, trans_b = _resolve_transpose_flags(x, weight, single_x, is_split_k)
    plan = _resolve_scenario(
        x,
        weight,
        x0,
        w0,
        group_list,
        group_list_type,
        group_type,
        single_x,
        single_w,
        single_y,
        is_split_k,
        w3d,
        has_group_list,
        trans_a,
        trans_b,
    )
    x_dtype = x0.dtype

    # ---- output allocation ----------------------------------------------
    if is_split_k:
        if plan.single_w:
            y = [
                torch.zeros(
                    plan.group_num, plan.single_m, plan.n, dtype=x_dtype, device=dev
                )
            ]
        else:
            y = [
                torch.zeros(plan.single_m, wg.shape[-1], dtype=x_dtype, device=dev)
                for wg in weight
            ]
    elif plan.single_y:
        if plan.single_x:
            # Positive-K M-split groups overwrite the entire output.
            allocate = (
                torch.empty if group_type == SPLIT_M and plan.k > 0 else torch.zeros
            )
            y = [allocate(plan.single_m, plan.n, dtype=x_dtype, device=dev)]
        else:
            y = [
                torch.zeros(sum(plan.m_list), plan.n_list[0], dtype=x_dtype, device=dev)
            ]
    else:
        y = [
            torch.zeros(xg.shape[0], wg.shape[-1], dtype=x_dtype, device=dev)
            for xg, wg in zip(x, weight)
        ]

    if plan.m == 0 or plan.n == 0 or plan.k == 0:
        return y

    # ---- per-group M extents for host tiling ----
    if group_type == SPLIT_K:
        group_m_sizes = [plan.m]  # every group computes the full M
    elif plan.m_list is not None:
        group_m_sizes = plan.m_list
    elif group_list is not None and group_list.numel() > 0:
        group_list_values = [int(v) for v in group_list.tolist()]
        if group_list_type == 1:
            group_m_sizes = group_list_values
        else:
            group_m_sizes = [group_list_values[0]] + [
                group_list_values[i] - group_list_values[i - 1]
                for i in range(1, len(group_list_values))
            ]
    else:
        group_m_sizes = [plan.m] if plan.group_num == 1 else []

    # ---- tiling ----
    tiling = GmmTiling(
        plan.m,
        plan.n,
        plan.k,
        plan.group_num,
        a_dtype=x_dtype,
        b_dtype=w0.dtype,
        c_dtype=x_dtype,
        trans_a=plan.trans_a,
        trans_b=plan.trans_b,
        group_type=group_type,
        group_list_type=group_list_type,
        l2_disable_seed=plan.l2_seed,
    )
    tiling._optimize_base_m_and_a_full_load(group_m_sizes)

    # ---- kernel launch ----------------------------------------------------
    op = GmmKernel(
        tiling,
        single_x=plan.single_x,
        single_w=plan.single_w,
        single_y=plan.single_y,
        has_group_list=has_group_list,
        w3d=plan.w3d,
    )
    gl_arg = (
        group_list if has_group_list else torch.zeros(1, dtype=torch.int64, device=dev)
    )

    if plan.trans_b:
        kern_w = [wg.transpose(-1, -2) for wg in weight]
    else:
        kern_w = weight
    op.run(x, kern_w, gl_arg, y, plan.group_num)

    return y
