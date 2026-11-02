# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Dual-level MXFP4 matmul with 32-K E8M0 and 512-K FP32 scale groups.

Requires ND A and FRACTAL_NZ B in FT layout; supports FP16/BF16 output and Bias.
"""

import torch
import torch_npu

import cannbotdsl

from cannbotdsl import TensorSpec, dtypes, get_mem_size, get_platform_info, make_layout
from cannbotdsl.buffer import Buffer
from cannbotdsl.ring_buffer import RingBuffer, make_ring_buffer
from cannbotdsl.lang import const_expr
from cannbotdsl.lang.vf import vf
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import (
    cv_valid_extent,
    copy_options,
    mem_copy,
)
from cannbotdsl.ops import reg
from cannbotdsl.tensor import (
    MemLoc,
    Tensor,
    reinterpret,
    tile_slice,
)

__all__ = ["npu_dual_level_quant_matmul"]


L1_CAPACITY_BYTES = get_mem_size("l1")
L0A_CAPACITY_BYTES = get_mem_size("l0a")
L0B_CAPACITY_BYTES = get_mem_size("l0b")
L0C_CAPACITY_BYTES = get_mem_size("l0c")
LEVEL0_GROUP_SIZE = 512
SCALE_K_L1_SIZE = 4096
LEVEL1_GROUP_SIZE = 32
MX_K_ALIGN = 64
MX_SCALE_PAIR = 2
CUBE_BLOCK = 16
L1_ALIGN_ELEMENTS = 64  # FP4: 32B inner-axis alignment = 64 elements.
L2_ALIGN_ELEMENTS = 256  # FP4: 128B L2 alignment = 256 elements.
BASIC_BLOCK_SIZE_256 = 256
BASIC_BLOCK_SIZE_512 = 512
BASIC_BLOCK_SIZE_1024 = 1024
BASEM_BASEN_RATIO = 2
BASEK_LIMIT = 4095
WINDOW_LEN = 4
LOAD_BALANCE_THRESHOLD = 1792
MTE2_CACHELINE_BYTES = 256
L0_BASE_M = 128
L0_BASE_N = 128
L0_BASE_K = 512
VECTOR_LENGTH = 64
LEVEL0_SCALE_CACHE_GROUPS = 32

# Stagger data and scales across eight equal L1 regions; offsets are bytes.
L1_REGION_BYTES = L1_CAPACITY_BYTES // 8
L1_A_OFFSETS_BYTES = (2 * L1_REGION_BYTES, 4 * L1_REGION_BYTES)
L1_B_OFFSETS_BYTES = (0, 7 * L1_REGION_BYTES)
L1_SCALE_A_OFFSETS_BYTES = (L1_REGION_BYTES, 13 * L1_REGION_BYTES // 2)
L1_SCALE_B_OFFSETS_BYTES = (3 * L1_REGION_BYTES // 2, 6 * L1_REGION_BYTES)


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def ceil_align(value, divisor):
    return ceil_div(value, divisor) * divisor


def floor_align(value, divisor):
    return value // divisor * divisor


def get_scale_k_len(k):
    return ceil_div(k, MX_K_ALIGN) * MX_SCALE_PAIR


# ============================================================================
# 1. Host-side tiling (mainline DualLevelQuantBatchMatmul ASW port)
# ============================================================================


def _get_base_block(m, n, k, *, aic_num):
    """Keep the adjusted basic block within the double-buffered L0A capacity."""
    base_m = ceil_align(min(m, BASIC_BLOCK_SIZE_256), L1_ALIGN_ELEMENTS)
    base_n = ceil_align(min(n, BASIC_BLOCK_SIZE_256), CUBE_BLOCK)
    base_k = ceil_align(min(k, BASIC_BLOCK_SIZE_512), MX_K_ALIGN)
    tail_win_logic = True
    if ceil_div(m, base_m) * ceil_div(n, base_n) >= aic_num:
        return base_m, base_n, base_k, tail_win_logic

    base_m_align = CUBE_BLOCK
    base_n_align = CUBE_BLOCK
    base_k_align = L2_ALIGN_ELEMENTS
    m_max_tile = ceil_div(m, base_m_align)
    n_max_tile = ceil_div(n, base_n_align)

    m_core = ceil_div(m, base_m)
    n_core = ceil_div(n, base_n)
    if m_max_tile < n_max_tile or (
        m_max_tile == n_max_tile and base_n_align == CUBE_BLOCK
    ):
        base_m = ceil_align(ceil_div(m, m_core), base_m_align)
        m_core = ceil_div(m, base_m)
        n_core = aic_num // m_core
        base_n = ceil_align(ceil_div(n, n_core), base_n_align)
    else:
        base_n = ceil_align(ceil_div(n, n_core), base_n_align)
        n_core = ceil_div(n, base_n)
        m_core = aic_num // n_core
        base_m = ceil_align(ceil_div(m, m_core), base_m_align)

    def update_cores():
        nonlocal base_m, base_n, m_core, n_core
        base_m = ceil_align(ceil_div(m, m_core), base_m_align)
        base_n = ceil_align(ceil_div(n, n_core), base_n_align)
        m_core = ceil_div(m, base_m)
        n_core = ceil_div(n, base_n)

    while (
        base_n >= base_m * BASEM_BASEN_RATIO
        and n_core < aic_num // 2
        and base_n != base_n_align
    ):
        n_core *= 2
        m_core = aic_num // n_core
        update_cores()
    while (
        base_m >= base_n * BASEM_BASEN_RATIO
        and m_core < aic_num // 2
        and base_m != base_m_align
    ):
        m_core *= 2
        n_core = aic_num // m_core
        update_cores()

    k_value_align = ceil_align(k, base_k_align)
    k_value_max = L0A_CAPACITY_BYTES // max(base_m, base_n) // 2
    if k_value_max >= base_k_align:
        base_k = min(k_value_align, floor_align(k_value_max, base_k_align))
        if base_k > BASEK_LIMIT:
            base_k = base_k // 2
        tail_win_logic = False
    else:
        base_m = ceil_align(min(m, BASIC_BLOCK_SIZE_256), L1_ALIGN_ELEMENTS)
        base_n = ceil_align(min(n, BASIC_BLOCK_SIZE_256), CUBE_BLOCK)
        base_k = ceil_align(min(k, BASIC_BLOCK_SIZE_512), MX_K_ALIGN)
    return base_m, base_n, base_k, tail_win_logic


def _optimize_edge_basic_block(
    m,
    n,
    k,
    base_m,
    base_n,
    *,
    aic_num,
):
    m_tiles = ceil_div(m, base_m)
    n_tiles = ceil_div(n, base_n)
    m_base_tail_split_count = 1
    n_base_tail_split_count = 1
    m_tail_main = 0
    n_tail_main = 0
    if m_tiles == 1 or n_tiles == 1:
        return 1, 1, 0, 0

    # FP4 packs 2 elements per byte: K bytes = K/2, so a 256B MTE2 cacheline
    # means K % 512 == 0.
    inner_axis_aligned = k % (MTE2_CACHELINE_BYTES * 2) == 0
    m_tail = m % base_m
    if m_tail and inner_axis_aligned:
        m_cnt = m_tiles
        base_tail_cnt_max = min((base_m - m_tail) // CUBE_BLOCK, m_cnt)
        window_size = min(WINDOW_LEN, m_cnt)
        main_window_num = m_cnt // window_size - 1
        tail_window_size = m_cnt - main_window_num * window_size
        perf_res = (main_window_num + 1) * base_m
        merge_window_num = 1
        for merge_len in range(tail_window_size - 1, base_tail_cnt_max, window_size):
            new_tail_main = ceil_align(
                ceil_div(merge_len * base_m + m_tail, merge_len + 1),
                CUBE_BLOCK,
            )
            cur_perf = (
                main_window_num + 1 - merge_window_num
            ) * base_m + merge_window_num * new_tail_main
            if cur_perf <= perf_res:
                perf_res = cur_perf
                m_tail_main = new_tail_main
                m_base_tail_split_count = merge_len + 1
            merge_window_num += 1

    n_tail = n % base_n
    balance_after_fixpipe = k < BASIC_BLOCK_SIZE_1024
    if (
        n_tail
        and (not balance_after_fixpipe)
        and (inner_axis_aligned or n >= LOAD_BALANCE_THRESHOLD)
    ):
        n_cnt = n_tiles
        m_cnt = m_tiles
        total_windows = ceil_div(n_cnt * m_cnt, aic_num)
        main_windows = ceil_div(
            (n_cnt - 1) * m_cnt + m_cnt % aic_num,
            aic_num,
        )
        if n_cnt * m_cnt <= aic_num or (
            m_cnt % aic_num == 0
            and (n_cnt % WINDOW_LEN == 0 or WINDOW_LEN % n_cnt == 0)
        ):
            main_windows = total_windows
        tail_windows = total_windows - main_windows
        perf_res = main_windows * base_n + tail_windows * n_tail
        base_tail_cnt_max = min((base_n - n_tail) // CUBE_BLOCK, n_cnt)
        for merge_len in range(1, base_tail_cnt_max):
            new_tail_main = ceil_align(
                ceil_div(merge_len * base_n + n_tail, merge_len + 1),
                CUBE_BLOCK,
            )
            if base_n < new_tail_main:
                continue
            new_tail_last = merge_len * (base_n - new_tail_main) + n_tail
            new_main_round = 0
            new_tail_round = 0
            if merge_len < n_cnt - 1:
                new_main_round = ceil_div(
                    (n_cnt - 1 - merge_len) * m_cnt + (merge_len + 1) * m_cnt % aic_num,
                    aic_num,
                )
            if merge_len:
                new_tail_round = min(
                    ceil_div(
                        merge_len * m_cnt + m_cnt % aic_num,
                        aic_num,
                    ),
                    total_windows - new_main_round,
                )
            cur_perf = (
                new_main_round * base_n
                + new_tail_round * new_tail_main
                + (total_windows - new_main_round - new_tail_round) * new_tail_last
            )
            if cur_perf < perf_res:
                perf_res = cur_perf
                n_tail_main = new_tail_main
                n_base_tail_split_count = merge_len + 1

    return (
        m_base_tail_split_count,
        n_base_tail_split_count,
        m_tail_main,
        n_tail_main,
    )


def _tail_split_valid(size, split):
    if split <= 1:
        return True
    full = ceil_div(size, split)
    last = size - (split - 1) * full
    if last <= 0:
        return False
    return True


def _calc_tail_basic_block_split(
    m,
    n,
    k,
    base_m,
    base_n,
    tail_win_logic=True,
    *,
    aic_num,
):
    if not tail_win_logic:
        return 1, 1
    m_tiles = ceil_div(m, base_m)
    n_tiles = ceil_div(n, base_n)
    tail_round = m_tiles * n_tiles % aic_num
    if tail_round == 0:
        return 1, 1

    m_tail = m - (m_tiles - 1) * base_m
    n_tail = n - (n_tiles - 1) * base_n
    split_first_m = m_tail >= n_tail
    if split_first_m:
        first_size, second_size = m_tail, n_tail
    else:
        first_size, second_size = n_tail, m_tail
    pre_split = 1
    sec_split = 1
    first_valid = 1
    second_valid = 1
    while (pre_split + 1) * sec_split * tail_round <= aic_num:
        pre_split += 1
        if _tail_split_valid(first_size, pre_split):
            first_valid = pre_split
        if pre_split * (sec_split + 1) * tail_round <= aic_num:
            sec_split += 1
            if _tail_split_valid(second_size, sec_split):
                second_valid = sec_split
    if split_first_m:
        return first_valid, second_valid
    return second_valid, first_valid


def get_dual_level_quant_matmul_tile_config(
    m,
    n,
    k,
    *,
    aic_num=None,
):
    """Return L1 tiles, L0 sub-blocks, active cores and tail-split counts.

    Return ``(tileM, tileN, baseM, baseN, baseK, usedCoreNum, mTailTile,
    nTailTile)``. tileM/tileN are the L1 basic blocks, baseM/baseN the L0
    sub-blocks, and baseK the L0 K step.
    """
    if aic_num is None:
        stream = torch.npu.current_stream()
        aic_num = int(get_platform_info(stream=stream).cube_core_num)
    tile_m, tile_n, _k_l1_size, tail_win_logic = _get_base_block(
        m,
        n,
        k,
        aic_num=aic_num,
    )
    base_m = min(L0_BASE_M, tile_m)
    base_n = min(L0_BASE_N, tile_n)
    base_k = L0_BASE_K
    m_tail_tile, n_tail_tile = _calc_tail_basic_block_split(
        m,
        n,
        k,
        tile_m,
        tile_n,
        tail_win_logic,
        aic_num=aic_num,
    )
    total_tiles = ceil_div(m, tile_m) * ceil_div(n, tile_n)
    tail_win_blocks = total_tiles % aic_num
    if total_tiles > aic_num or tail_win_blocks == 0:
        used_core_num = aic_num
    else:
        used_core_num = tail_win_blocks * m_tail_tile * n_tail_tile
    return (
        tile_m,
        tile_n,
        base_m,
        base_n,
        base_k,
        used_core_num,
        m_tail_tile,
        n_tail_tile,
    )


# ============================================================================
# 3. Kernel
# ============================================================================


class DualLevelQuantMatmulMxfp4Kernel:
    def __init__(
        self,
        m,
        n,
        k,
        tile_m,
        tile_n,
        base_m,
        base_n,
        base_k,
        used_core_num,
        output_dtype,
        has_bias,
        m_tail_tile=1,
        n_tail_tile=1,
        *,
        aic_num,
    ):
        self.m, self.n, self.k = m, n, k
        self.tile_m, self.tile_n = tile_m, tile_n
        self.base_m, self.base_n, self.base_k = base_m, base_n, base_k
        self.used_core_num = used_core_num
        self.output_dtype = output_dtype
        self.has_bias = has_bias
        self.m_tiles = ceil_div(m, tile_m)
        self.n_tiles = ceil_div(n, tile_n)
        self.m_l0_tiles = ceil_div(tile_m, base_m)
        self.n_l0_tiles = ceil_div(tile_n, base_n)
        self.reuse_l1 = self.m_l0_tiles > 1 or self.n_l0_tiles > 1
        self.level0_tiles = ceil_div(k, LEVEL0_GROUP_SIZE)
        self.scale_a0_full_load = self.level0_tiles <= LEVEL0_SCALE_CACHE_GROUPS
        self.scale_a0_stride = self.level0_tiles if self.scale_a0_full_load else 1
        self.step_k = LEVEL0_GROUP_SIZE // base_k
        self.scale_k_l1_factor = SCALE_K_L1_SIZE // LEVEL0_GROUP_SIZE
        self.scale_k_l1_tiles = ceil_div(self.level0_tiles, self.scale_k_l1_factor)
        # Scale capacity must cover the padded baseK used by L1->L0 loads.
        self.scale_k_l1_len = get_scale_k_len(max(base_k, min(k, SCALE_K_L1_SIZE)))
        self.scale_k_l1_groups = self.scale_k_l1_len // MX_SCALE_PAIR
        self.scale_k_l0_len = base_k // LEVEL1_GROUP_SIZE
        self.vector_m = ceil_div(base_m, 2)
        self.vector_tile_m = ceil_div(tile_m, 2)
        self.partial_stride = L0_BASE_N
        self.accumulator_stride = ceil_align(tile_n, L0_BASE_N)
        (
            self.m_base_tail_split_count,
            self.n_base_tail_split_count,
            m_tail_main,
            n_tail_main,
        ) = _optimize_edge_basic_block(
            m,
            n,
            k,
            tile_m,
            tile_n,
            aic_num=aic_num,
        )
        self.m_base_normal_count = self.m_tiles - self.m_base_tail_split_count
        self.n_base_normal_count = self.n_tiles - self.n_base_tail_split_count
        merged_m = m - self.m_base_normal_count * tile_m
        merged_n = n - self.n_base_normal_count * tile_n
        self.m_base_tail_main = (
            merged_m if self.m_base_tail_split_count == 1 else m_tail_main
        )
        self.n_base_tail_main = (
            merged_n if self.n_base_tail_split_count == 1 else n_tail_main
        )
        self.m_base_tail_last = (
            merged_m - (self.m_base_tail_split_count - 1) * self.m_base_tail_main
        )
        self.n_base_tail_last = (
            merged_n - (self.n_base_tail_split_count - 1) * self.n_base_tail_main
        )
        # Adaptive sliding window over M with snake walk over N.
        self.main_window = min(WINDOW_LEN, self.m_tiles)
        self.main_row = self.m_tiles // self.main_window - 1
        self.tail_window = self.m_tiles - self.main_window * self.main_row
        self.total_main_tiles = self.main_row * self.main_window * self.n_tiles
        # Tail-window resplit across idle cores.
        self.m_tail_tile = m_tail_tile
        self.n_tail_tile = n_tail_tile
        self.nz_repartition = self.n_base_tail_split_count > 1 or n_tail_tile > 1
        regions = (
            (0, self.tile_n, 0, self.n_base_normal_count),
            (
                self.n_base_normal_count * self.tile_n,
                self.n_base_tail_main,
                self.n_base_normal_count,
                self.n_tiles - 1,
            ),
            (
                self.n - self.n_base_tail_last,
                self.n_base_tail_last,
                self.n_tiles - 1,
                self.n_tiles,
            ),
        )
        self.nz_copy_regions = tuple(
            subregion
            for start, size, first, end in regions
            for subregion in (
                tuple(
                    (start + (idx - first) * size, size, idx, idx + 1)
                    for idx in range(first, end)
                )
                if start > 0 and start < self.n - start
                else ((start, size, first, end),)
            )
        )
        self.use_tail_split = m_tail_tile > 1 or n_tail_tile > 1
        total_tiles = self.m_tiles * self.n_tiles
        tail_tiles = total_tiles % used_core_num or used_core_num
        self.tail_win_block_cnt = min(total_tiles, tail_tiles)

    @jit
    def _init_accumulator(self, bias=None):
        """Initialize the FP32 output tile once before the level0 groups."""
        with vf(mode="simd"):
            mask = reg.create_mask(elem_bits=32)
            if const_expr(self.has_bias):
                for column in tuple(range(0, self.accumulator_stride, VECTOR_LENGTH)):
                    value = reg.vload(bias, column)
                    for row in range(self.vector_tile_m):
                        reg.vstore(
                            self.ub_accumulator,
                            row * self.accumulator_stride + column,
                            value,
                            mask,
                        )
            else:
                zero = reg.vdups(0.0, dtypes.float32)
                for offset in range(
                    0, self.vector_tile_m * self.accumulator_stride, VECTOR_LENGTH
                ):
                    reg.vstore(self.ub_accumulator, offset, zero, mask)
            reg.vmem_bar("vst_vld")

    @jit
    def _accumulate_level0_group(
        self,
        partial,
        scale_a0,
        scale_b0,
        accumulator,
        actual_m,
        actual_n,
        accumulator_row_stride,
        scale_a0_group,
        accumulator_offset,
        output=None,
    ):
        """Scale one level0 group's partial and accumulate."""
        with vf(mode="simd"):
            mask = reg.create_mask(elem_bits=32)
            scale_b_low = reg.vload(scale_b0)
            scale_b_high = scale_b_low
            if const_expr(self.base_n > VECTOR_LENGTH):
                scale_b_high = reg.vload(scale_b0, VECTOR_LENGTH)
            for row in range(actual_m):
                scale_a = reg.vload_broadcast(
                    scale_a0, row * self.scale_a0_stride + scale_a0_group
                )
                partial_row_offset = row * self.partial_stride
                accumulator_row_offset = (
                    accumulator_offset + row * accumulator_row_stride
                )
                partial_low = reg.vload(partial, partial_row_offset)
                partial_high = partial_low
                if const_expr(self.base_n > VECTOR_LENGTH):
                    partial_high = reg.vload(
                        partial, partial_row_offset + VECTOR_LENGTH
                    )
                value_low = reg.vload(accumulator, accumulator_row_offset)
                value_high = value_low
                if const_expr(self.base_n > VECTOR_LENGTH):
                    value_high = reg.vload(
                        accumulator, accumulator_row_offset + VECTOR_LENGTH
                    )
                scale_low = reg.vmul(scale_a, scale_b_low, mask=mask)
                scale_high = scale_low
                if const_expr(self.base_n > VECTOR_LENGTH):
                    scale_high = reg.vmul(scale_a, scale_b_high, mask=mask)
                value_low = reg.vmadd(partial_low, scale_low, value_low, mask=mask)
                if const_expr(self.base_n > VECTOR_LENGTH):
                    value_high = reg.vmadd(
                        partial_high, scale_high, value_high, mask=mask
                    )
                if const_expr(output is not None):
                    value_low = reg.vcast(value_low, self.output_dtype, mask=mask)
                    if const_expr(self.base_n > VECTOR_LENGTH):
                        value_high = reg.vcast(value_high, self.output_dtype, mask=mask)
                    reg.vstore_pack(
                        output,
                        partial_row_offset,
                        value_low,
                        mask,
                        pack_mode="b32_to_b16",
                    )
                    if const_expr(self.base_n > VECTOR_LENGTH):
                        reg.vstore_pack(
                            output,
                            partial_row_offset + VECTOR_LENGTH,
                            value_high,
                            mask,
                            pack_mode="b32_to_b16",
                        )
                else:
                    reg.vstore(accumulator, accumulator_row_offset, value_low, mask)
                    if const_expr(self.base_n > VECTOR_LENGTH):
                        reg.vstore(
                            accumulator,
                            accumulator_row_offset + VECTOR_LENGTH,
                            value_high,
                            mask,
                        )
            reg.vmem_bar("vst_vld")

    @jit
    def _copy_nz_b_tile(self, b_l1, b_gm, n_idx, n_start, current_n, level0_idx):
        """Copy the scheduled NZ window without changing the GM physical pitch."""
        for start, size, first, end in self.nz_copy_regions:
            if const_expr(first < end):
                if first <= n_idx < end:
                    if const_expr(start > 0):
                        region = tile_slice(b_gm, (start, self.k), (1, 0))
                    else:
                        region = b_gm
                    b_tile = tile_slice(
                        region,
                        (size, min(self.k, LEVEL0_GROUP_SIZE)),
                        (
                            n_idx - first,
                            0 if const_expr(self.level0_tiles == 1) else level0_idx,
                        ),
                    )
                    if const_expr(self.n_tail_tile > 1):
                        if current_n < size:
                            sub_n = ceil_align(
                                ceil_div(size, self.n_tail_tile), CUBE_BLOCK
                            )
                            sub_idx = (
                                n_start - start - (n_idx - first) * size
                            ) // sub_n
                            b_sub_tile = tile_slice(
                                b_tile,
                                (sub_n, min(self.k, LEVEL0_GROUP_SIZE)),
                                (sub_idx, 0),
                            )
                            mem_copy(b_l1, b_sub_tile, copy_options(l2_cache_ctl=0))
                        else:
                            mem_copy(b_l1, b_tile, copy_options(l2_cache_ctl=0))
                    else:
                        mem_copy(b_l1, b_tile, copy_options(l2_cache_ctl=0))

    @jit
    def _get_output_tile_bounds(self, m_idx, n_idx):
        """Rebalanced tails shift the GM origin as well as the valid M/N size."""
        m_size = self.tile_m
        if m_idx >= self.m_base_normal_count:
            m_size = (
                self.m_base_tail_last
                if m_idx == self.m_tiles - 1
                else self.m_base_tail_main
            )
        n_size = self.tile_n
        if n_idx >= self.n_base_normal_count:
            n_size = (
                self.n_base_tail_last
                if n_idx == self.n_tiles - 1
                else self.n_base_tail_main
            )
        m_start = m_idx * self.tile_m
        if m_idx > self.m_base_normal_count:
            m_start -= (m_idx - self.m_base_normal_count) * (
                self.tile_m - self.m_base_tail_main
            )
        n_start = n_idx * self.tile_n
        if n_idx > self.n_base_normal_count:
            n_start -= (n_idx - self.n_base_normal_count) * (
                self.tile_n - self.n_base_tail_main
            )
        if const_expr(self.m_base_tail_split_count == 1):
            m_start = m_idx * self.tile_m
            m_size = min(self.tile_m, self.m - m_start)
        if const_expr(self.n_base_tail_split_count == 1):
            n_start = n_idx * self.tile_n
            n_size = min(self.tile_n, self.n - n_start)
        return m_start, n_start, m_size, n_size

    @jit
    def _compute_level0_tile(
        self,
        out_tile,
        l1_tiles,
        scale_a0_gm,
        scale_b0_gm,
        scale_a0_read,
        scale_b0_read,
        tile_bounds,
        matrix_shape,
        level0_idx,
        k_l1_idx_in_scale,
        actual_k,
        l0_indices=(0, 0),
    ):
        """Compute one L0 tile, apply Level0 scales and store the final group."""
        a_l1_rd, b_l1_rd, scale_a1_l1_rd, scale_b1_l1_rd = l1_tiles
        m_start, m_end, n_start, n_end = tile_bounds
        m_l0_idx, n_l0_idx = l0_indices
        actual_m, actual_n = matrix_shape
        accumulator_offset = (
            m_l0_idx * self.vector_m * self.accumulator_stride + n_l0_idx * self.base_n
        )
        scale_l0_base = k_l1_idx_in_scale * self.step_k
        current_step_k = (
            1 if const_expr(self.step_k == 1) else ceil_div(actual_k, self.base_k)
        )
        l0c_slot = self.l0c.next_write(bare=True).view(make_layout(out_tile.shape))
        for k_l0_idx in range(current_step_k):
            a_l1_tile = tile_slice(
                a_l1_rd,
                (self.base_m, self.base_k),
                (m_l0_idx, k_l0_idx),
            )
            b_l1_tile = tile_slice(
                b_l1_rd,
                (self.base_n, self.base_k),
                (n_l0_idx, k_l0_idx),
            )
            scale_a1_l1_tile = tile_slice(
                scale_a1_l1_rd,
                (self.base_m, self.scale_k_l0_len),
                (m_l0_idx, scale_l0_base + k_l0_idx),
            )
            scale_b1_l1_tile = tile_slice(
                scale_b1_l1_rd,
                (self.scale_k_l0_len, self.base_n),
                (scale_l0_base + k_l0_idx, n_l0_idx),
            )
            valid_k = min(self.base_k, actual_k - k_l0_idx * self.base_k)
            valid_scale_k = ceil_div(valid_k, MX_K_ALIGN) * MX_SCALE_PAIR
            a_ready = self.l0a.next_write(bare=True).view(
                make_layout((out_tile.shape[0], valid_k))
            )
            b_ready = self.l0b.next_write(bare=True).view(
                make_layout((out_tile.shape[1], valid_k))
            )
            mem_copy(
                a_ready,
                a_l1_tile[: out_tile.shape[0], :valid_k],
                copy_options(
                    mx_scale=scale_a1_l1_tile[: out_tile.shape[0], :valid_scale_k]
                ),
            )
            mem_copy(
                b_ready,
                b_l1_tile[: out_tile.shape[1], :valid_k],
                copy_options(
                    mx_scale=scale_b1_l1_tile[:valid_scale_k, : out_tile.shape[1]]
                ),
            )
            is_last = k_l0_idx + 1 == current_step_k
            matmul(
                l0c_slot,
                a_ready,
                b_ready,
                init=(k_l0_idx == 0),
                unit_flag=3 if is_last else 2,
                mx_mode=True,
            )

        mem_copy(
            self.ub_partial.next_write(),
            l0c_slot.partition(dim=0),
            copy_options(unit_flag=3),
        )
        scale_a0_tile = scale_a0_read
        if const_expr(self.scale_a0_full_load):
            scale_a0_tile = tile_slice(
                scale_a0_read,
                (self.vector_m, self.scale_a0_stride),
                (m_l0_idx, 0),
            )
        else:
            scale_a0_tile = tile_slice(
                scale_a0_gm[m_start:m_end, level0_idx : level0_idx + 1],
                (self.base_m, 1),
                (0, 0),
            )
            mem_copy(
                self.ub_scale_a0.next_write(),
                scale_a0_tile.partition(dim=0, part_id=get_subblock_id()),
            )
            scale_a0_tile = self.ub_scale_a0.next_read()
        # The reuse path stages ScaleB0 once before its M/N sub-tile loops.
        if const_expr(not self.reuse_l1):
            scale_b0_tile = scale_b0_gm[level0_idx : level0_idx + 1, n_start:n_end]
            mem_copy(self.ub_scale_b0.next_write(), scale_b0_tile)
        partial = reinterpret(
            self.ub_partial.next_read(),
            layout=make_layout(matrix_shape, stride=(self.partial_stride, 1)),
        )
        if const_expr(not self.reuse_l1):
            scale_b0_read = self.ub_scale_b0.next_read()
        if level0_idx == self.level0_tiles - 1:
            self._accumulate_level0_group(
                partial,
                scale_a0_tile,
                scale_b0_read,
                self.ub_accumulator,
                actual_m,
                actual_n,
                self.accumulator_stride,
                level0_idx if const_expr(self.scale_a0_full_load) else 0,
                accumulator_offset,
                output=self.ub_output.next_write(),
            )
            output_view = reinterpret(
                self.ub_output.next_read(),
                layout=make_layout(matrix_shape, stride=(self.partial_stride, 1)),
            )
            mem_copy(
                out_tile.partition(dim=0, part_id=get_subblock_id()),
                output_view,
                copy_options(l2_cache_ctl=1),
            )
        else:
            self._accumulate_level0_group(
                partial,
                scale_a0_tile,
                scale_b0_read,
                self.ub_accumulator,
                actual_m,
                actual_n,
                self.accumulator_stride,
                level0_idx if const_expr(self.scale_a0_full_load) else 0,
                accumulator_offset,
            )

    @jit
    def _compute_output_tile(
        self,
        out_gm,
        a_gm,
        b_gm,
        scale_a0_gm,
        scale_b0_gm,
        scale_a1_gm,
        scale_b1_gm,
        bias_gm,
        m_idx,
        n_idx,
        m_start,
        n_start,
        current_m,
        current_n,
    ):
        """Stage shared L1 windows, keeping the simple and reuse loop layouts."""
        m_end = m_start + current_m
        n_end = n_start + current_n
        # Region dataflow requires defaults even for compile-time-only branches.
        out_tile = out_gm
        matrix_shape = out_gm.shape
        if const_expr(not self.reuse_l1):
            # Static bounds let cv_valid_extent prove the per-AIV row capacity.
            if const_expr(
                not self.use_tail_split
                and self.m_base_tail_split_count == self.n_base_tail_split_count == 1
            ):
                out_tile = tile_slice(
                    out_gm, (self.tile_m, self.tile_n), (m_idx, n_idx)
                )
            else:
                out_tile = tile_slice(
                    out_gm[m_start:m_end, n_start:n_end],
                    (self.tile_m, self.tile_n),
                    (0, 0),
                )
            actual_m = cv_valid_extent(out_tile, axis=0, part_id=get_subblock_id())
            matrix_shape = (actual_m, out_tile.shape[1])
        if const_expr(self.has_bias):
            mem_copy(self.ub_bias.next_write(), bias_gm[n_start:n_end,])
        self._init_accumulator(
            self.ub_bias.next_read() if const_expr(self.has_bias) else None
        )
        scale_a0_read = self.ub_accumulator
        if const_expr(self.scale_a0_full_load):
            if const_expr(self.reuse_l1):
                scale_a0_slot = self.ub_scale_a0.next_write()
                # Split each L0 M block between the two AIVs independently.
                for m_l0_idx in range(ceil_div(current_m, self.base_m)):
                    m_sub_start = m_start + m_l0_idx * self.base_m
                    m_sub_end = min(m_sub_start + self.base_m, m_end)
                    scale_a0_dst = tile_slice(
                        scale_a0_slot,
                        (self.vector_m, self.scale_a0_stride),
                        (m_l0_idx, 0),
                    )
                    scale_a0_tile = tile_slice(
                        scale_a0_gm[m_sub_start:m_sub_end, None],
                        (self.base_m, self.scale_a0_stride),
                        (0, 0),
                    )
                    mem_copy(
                        scale_a0_dst,
                        scale_a0_tile.partition(dim=0, part_id=get_subblock_id()),
                    )
            else:
                scale_a0_tile = tile_slice(
                    scale_a0_gm[m_start:m_end, None],
                    (self.base_m, self.scale_a0_stride),
                    (0, 0),
                )
                mem_copy(
                    self.ub_scale_a0.next_write(),
                    scale_a0_tile.partition(dim=0, part_id=get_subblock_id()),
                )
            scale_a0_read = self.ub_scale_a0.next_read()
        for scale_l1_idx in range(self.scale_k_l1_tiles):
            scale_a1_tile = tile_slice(
                scale_a1_gm[m_start:m_end, None, None],
                (self.tile_m, self.scale_k_l1_groups, MX_SCALE_PAIR),
                (0, scale_l1_idx, 0),
            )
            scale_b1_tile = tile_slice(
                scale_b1_gm[n_start:n_end, None, None],
                (self.tile_n, self.scale_k_l1_groups, MX_SCALE_PAIR),
                (0, scale_l1_idx, 0),
            )
            mem_copy(
                self.l1_scale_a.next_write(),
                scale_a1_tile.flatten(-2, -1),
                copy_options(l2_cache_ctl=0),
            )
            mem_copy(
                self.l1_scale_b.next_write(),
                scale_b1_tile.flatten(-2, -1).permute(1, 0),
                copy_options(l2_cache_ctl=0),
            )
            scale_a1_l1_rd = self.l1_scale_a.next_read()
            scale_b1_l1_rd = self.l1_scale_b.next_read()
            current_scale_factor = min(
                self.scale_k_l1_factor,
                self.level0_tiles - scale_l1_idx * self.scale_k_l1_factor,
            )
            for k_l1_idx_in_scale in range(current_scale_factor):
                level0_idx = scale_l1_idx * self.scale_k_l1_factor + k_l1_idx_in_scale
                k_offset = level0_idx * LEVEL0_GROUP_SIZE
                actual_k = min(LEVEL0_GROUP_SIZE, self.k - k_offset)
                a_tile = tile_slice(
                    a_gm[m_start:m_end, None],
                    (self.tile_m, LEVEL0_GROUP_SIZE),
                    (0, level0_idx),
                )
                b_tile = b_gm
                if const_expr(self.nz_repartition):
                    self._copy_nz_b_tile(
                        self.l1_b.next_write(),
                        b_gm,
                        n_idx,
                        n_start,
                        current_n,
                        level0_idx,
                    )
                else:
                    b_tile = tile_slice(
                        b_gm,
                        (min(self.n, self.tile_n), min(self.k, LEVEL0_GROUP_SIZE)),
                        (
                            n_idx,
                            0 if const_expr(self.level0_tiles == 1) else level0_idx,
                        ),
                    )
                mem_copy(self.l1_a.next_write(), a_tile, copy_options(l2_cache_ctl=0))
                if const_expr(not self.nz_repartition):
                    mem_copy(
                        self.l1_b.next_write(), b_tile, copy_options(l2_cache_ctl=0)
                    )
                # Select one read-slot pair per window, before all L0 sub-tiles.
                a_l1_rd = self.l1_a.next_read()
                b_l1_rd = self.l1_b.next_read()
                if const_expr(self.reuse_l1):
                    b_l1_rd = reinterpret(
                        b_l1_rd, layout=make_layout((self.tile_n, actual_k))
                    )
                    if const_expr(
                        self.tile_n % self.base_n
                        or self.n % self.base_n
                        or self.n_base_tail_split_count > 1
                        or self.n_tail_tile > 1
                    ):
                        b_l1_rd = b_l1_rd[0:current_n, None]
                    scale_b0_tile = scale_b0_gm[
                        level0_idx : level0_idx + 1, n_start:n_end
                    ]
                    mem_copy(self.ub_scale_b0.next_write(), scale_b0_tile)
                    scale_b0_read = self.ub_scale_b0.next_read()
                    for n_l0_idx in range(ceil_div(current_n, self.base_n)):
                        n_sub_start = n_start + n_l0_idx * self.base_n
                        n_sub_end = min(n_sub_start + self.base_n, n_end)
                        scale_b0_sub_tile = tile_slice(
                            scale_b0_read, (1, self.base_n), (0, n_l0_idx)
                        )
                        for m_l0_idx in range(ceil_div(current_m, self.base_m)):
                            m_sub_start = m_start + m_l0_idx * self.base_m
                            m_sub_end = min(m_sub_start + self.base_m, m_end)
                            out_sub_tile = tile_slice(
                                out_gm[m_sub_start:m_sub_end, n_sub_start:n_sub_end],
                                (self.base_m, self.base_n),
                                (0, 0),
                            )
                            actual_sub_m = cv_valid_extent(
                                out_sub_tile, axis=0, part_id=get_subblock_id()
                            )
                            self._compute_level0_tile(
                                out_sub_tile,
                                (a_l1_rd, b_l1_rd, scale_a1_l1_rd, scale_b1_l1_rd),
                                scale_a0_gm,
                                scale_b0_gm,
                                scale_a0_read,
                                scale_b0_sub_tile,
                                (m_sub_start, m_sub_end, n_sub_start, n_sub_end),
                                (actual_sub_m, out_sub_tile.shape[1]),
                                level0_idx,
                                k_l1_idx_in_scale,
                                actual_k,
                                (m_l0_idx, n_l0_idx),
                            )
                else:
                    self._compute_level0_tile(
                        out_tile,
                        (a_l1_rd, b_l1_rd, scale_a1_l1_rd, scale_b1_l1_rd),
                        scale_a0_gm,
                        scale_b0_gm,
                        scale_a0_read,
                        self.ub_accumulator,
                        (m_start, m_end, n_start, n_end),
                        matrix_shape,
                        level0_idx,
                        k_l1_idx_in_scale,
                        actual_k,
                    )

    @kernel
    def dual_level_quant_matmul_kernel(
        self,
        out_gm: Tensor,
        a_gm: Tensor,
        b_gm: Tensor,
        scale_a0_gm: Tensor,
        scale_b0_gm: Tensor,
        scale_a1_gm: Tensor,
        scale_b1_gm: Tensor,
        bias_gm: Tensor,
    ):
        tile_m, tile_n = self.tile_m, self.tile_n
        base_m, base_n, base_k = self.base_m, self.base_n, self.base_k
        # Keep the two data stages in opposite L1 halves, with scales between them.
        l1_storage = cannbotdsl.L1.view(L1_CAPACITY_BYTES)
        self.l1_a = make_ring_buffer(
            [
                reinterpret(
                    l1_storage,
                    dtype=dtypes.fp4x2_e2m1,
                    layout=make_layout((tile_m, LEVEL0_GROUP_SIZE)),
                    offset=offset,
                    data_format="nz",
                )
                for offset in L1_A_OFFSETS_BYTES
            ]
        )
        self.l1_b = make_ring_buffer(
            [
                reinterpret(
                    l1_storage,
                    dtype=dtypes.fp4x2_e2m1,
                    layout=make_layout((tile_n, LEVEL0_GROUP_SIZE)),
                    offset=offset,
                    data_format="nz",
                )
                for offset in L1_B_OFFSETS_BYTES
            ]
        )
        self.l1_scale_a = make_ring_buffer(
            [
                reinterpret(
                    l1_storage,
                    dtype=dtypes.float8_e8m0,
                    layout=make_layout((tile_m, self.scale_k_l1_len)),
                    offset=offset,
                    data_format="zn",
                )
                for offset in L1_SCALE_A_OFFSETS_BYTES
            ]
        )
        self.l1_scale_b = make_ring_buffer(
            [
                reinterpret(
                    l1_storage,
                    dtype=dtypes.float8_e8m0,
                    layout=make_layout((self.scale_k_l1_len, tile_n)),
                    offset=offset,
                    data_format="nz",
                )
                for offset in L1_SCALE_B_OFFSETS_BYTES
            ]
        )
        self.l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.fp4x2_e2m1,
            make_layout((base_m, base_k)),
            depth=2,
            data_format="nz",
        )
        self.l0b = RingBuffer(
            MemLoc.L0B,
            dtypes.fp4x2_e2m1,
            make_layout((base_n, base_k)),
            depth=2,
            data_format="nz",
        )
        # Rotate partial matrices over three slots so MMAD can overlap FixPipe.
        self.l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((base_m, base_n)),
            depth=3,
        )

        vector_shape = (self.vector_m, self.partial_stride)
        # The next FixPipe write can overlap the current Vector accumulation.
        self.ub_partial = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout(vector_shape),
            depth=2,
            data_format="nd",
        )
        self.ub_scale_a0 = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((self.vector_tile_m, self.scale_a0_stride)),
            depth=1,
        )
        self.ub_scale_b0 = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((1, ceil_align(tile_n, VECTOR_LENGTH))),
            depth=2,
        )
        if const_expr(self.has_bias):
            self.ub_bias = RingBuffer(
                MemLoc.UB,
                dtypes.float32,
                make_layout((self.accumulator_stride,)),
                depth=1,
            )
        self.ub_accumulator = Buffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((self.vector_tile_m, self.accumulator_stride)),
        )
        self.ub_output = RingBuffer(
            MemLoc.UB,
            self.output_dtype,
            make_layout(vector_shape),
            depth=2,
        )

        block_idx = get_block_idx()
        total_tiles = self.m_tiles * self.n_tiles
        end_block_idx = (total_tiles - 1) % self.used_core_num
        round_count = ceil_div(total_tiles, self.used_core_num)
        if const_expr(total_tiles % self.used_core_num != 0):
            if block_idx > end_block_idx:
                round_count -= 1
        new_end_block_idx = end_block_idx
        sub_tiles_per_tile = 1
        if const_expr(self.use_tail_split):
            sub_tiles_per_tile = self.m_tail_tile * self.n_tail_tile
            new_end_block_idx = end_block_idx + self.tail_win_block_cnt * (
                sub_tiles_per_tile - 1
            )
            if end_block_idx < block_idx <= new_end_block_idx:
                round_count += 1
        for round_idx in range(round_count):
            tail_split = False
            logical_idx = round_idx * self.used_core_num + block_idx
            sub_m_idx = 0
            sub_n_idx = 0
            if const_expr(self.use_tail_split):
                tail_split = (
                    round_idx == round_count - 1 and block_idx <= new_end_block_idx
                )
                if tail_split:
                    logical_idx = (
                        round_idx * self.used_core_num + block_idx // sub_tiles_per_tile
                    )
                    split_idx = block_idx % sub_tiles_per_tile
                    sub_m_idx = split_idx % self.m_tail_tile
                    sub_n_idx = split_idx // self.m_tail_tile
            m_idx = 0
            n_forward = 0
            row_idx = 0
            if const_expr(self.total_main_tiles == 0):
                m_idx = logical_idx % self.tail_window
                n_forward = (logical_idx // self.tail_window) % self.n_tiles
            elif logical_idx < self.total_main_tiles:
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
            m_start, n_start, current_m, current_n = self._get_output_tile_bounds(
                m_idx,
                n_idx,
            )
            if tail_split:
                sub_m = ceil_div(current_m, self.m_tail_tile)
                sub_n = ceil_align(ceil_div(current_n, self.n_tail_tile), CUBE_BLOCK)
                m_start += sub_m_idx * sub_m
                n_start += sub_n_idx * sub_n
                current_m = min(current_m - sub_m_idx * sub_m, sub_m)
                current_n = min(current_n - sub_n_idx * sub_n, sub_n)
            if (not const_expr(self.use_tail_split)) or (
                current_m > 0 and current_n > 0
            ):
                self._compute_output_tile(
                    out_gm,
                    a_gm,
                    b_gm,
                    scale_a0_gm,
                    scale_b0_gm,
                    scale_a1_gm,
                    scale_b1_gm,
                    bias_gm,
                    m_idx,
                    n_idx,
                    m_start,
                    n_start,
                    current_m,
                    current_n,
                )

    @host
    def run(
        self,
        out_gm: Tensor,
        a_gm: Tensor,
        b_gm: Tensor,
        scale_a0_gm: Tensor,
        scale_b0_gm: Tensor,
        scale_a1_gm: Tensor,
        scale_b1_gm: Tensor,
        bias_gm: Tensor,
    ):
        self.dual_level_quant_matmul_kernel[self.used_core_num](
            out_gm,
            a_gm,
            b_gm,
            scale_a0_gm,
            scale_b0_gm,
            scale_a1_gm,
            scale_b1_gm,
            bias_gm,
        )


# ============================================================================
# 4. Torch interface
# ============================================================================


_TORCH_OUTPUT_TO_DSL = {
    torch.float16: dtypes.float16,
    torch.bfloat16: dtypes.bfloat16,
}


def _build_aot_callable(
    m,
    n,
    k,
    has_bias,
    output_dtype,
    aic_num,
):
    (
        tile_m,
        tile_n,
        base_m,
        base_n,
        base_k,
        used_core_num,
        m_tail_tile,
        n_tail_tile,
    ) = get_dual_level_quant_matmul_tile_config(
        m,
        n,
        k,
        aic_num=aic_num,
    )
    kernel_obj = DualLevelQuantMatmulMxfp4Kernel(
        m,
        n,
        k,
        tile_m,
        tile_n,
        base_m,
        base_n,
        base_k,
        used_core_num,
        _TORCH_OUTPUT_TO_DSL[output_dtype],
        has_bias,
        m_tail_tile,
        n_tail_tile,
        aic_num=aic_num,
    )
    level0_groups = ceil_div(k, LEVEL0_GROUP_SIZE)
    level1_pairs = ceil_div(k, MX_K_ALIGN)
    specs = (
        TensorSpec(shape=(m, n), dtype=_TORCH_OUTPUT_TO_DSL[output_dtype]),
        TensorSpec(
            shape=(m, k),
            dtype=dtypes.fp4x2_e2m1,
        ),
        TensorSpec(
            shape=(n, k),
            dtype=dtypes.fp4x2_e2m1,
            storage_format="nz",
        ),
        TensorSpec(shape=(m, level0_groups), dtype=dtypes.float32),
        TensorSpec(shape=(level0_groups, n), dtype=dtypes.float32),
        TensorSpec(
            shape=(m, level1_pairs, MX_SCALE_PAIR),
            dtype=dtypes.float8_e8m0,
        ),
        TensorSpec(
            shape=(n, level1_pairs, MX_SCALE_PAIR),
            dtype=dtypes.float8_e8m0,
        ),
        TensorSpec(shape=(n if has_bias else 0,), dtype=dtypes.float32),
    )
    return cannbotdsl.compile(kernel_obj.run, *specs)


def _validate_dual_level_input(
    a,
    b,
    scale_a0,
    scale_b0,
    scale_a1,
    scale_b1,
    bias,
    output_dtype,
):
    if a.device.type != "npu":
        raise ValueError("a and b must be NPU tensors")
    if a.dtype != torch.float4_e2m1fn_x2:
        raise TypeError("a must use torch.float4_e2m1fn_x2")
    if torch_npu.get_npu_format(a) != torch_npu.Format.ND:
        raise ValueError("a must use ND storage format")
    if b.dtype != torch.int8:
        raise TypeError("b must use int8 FRACTAL_NZ storage containing packed FP4 data")
    if torch_npu.get_npu_format(b) != torch_npu.Format.FRACTAL_NZ:
        raise ValueError("b must use FRACTAL_NZ storage format")
    if output_dtype not in _TORCH_OUTPUT_TO_DSL:
        raise TypeError("output_dtype must be torch.float16 or torch.bfloat16")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("a and b must be rank-2 tensors")
    if scale_a0.dtype != torch.float32 or scale_b0.dtype != torch.float32:
        raise TypeError("Level0 scales must be float32")
    if scale_a1.dtype != torch.float8_e8m0fnu or scale_b1.dtype != torch.float8_e8m0fnu:
        raise TypeError("Level1 scales must be float8_e8m0fnu")
    tensors = (
        b,
        scale_a0,
        scale_b0,
        scale_a1,
        scale_b1,
    )
    if any(tensor.device != a.device for tensor in tensors):
        raise ValueError("all inputs must be on the same device")
    has_bias = bias is not None
    if has_bias and bias.dtype != torch.float32:
        raise TypeError("bias must be float32")
    if has_bias and bias.device != a.device:
        raise ValueError("bias and matrix inputs must be on the same device")
    m, packed_k = a.shape
    n, b_packed_k = b.shape
    if packed_k != b_packed_k:
        raise ValueError("a and b must have the same packed K length")
    k = packed_k * 2
    level0_groups = ceil_div(k, LEVEL0_GROUP_SIZE)
    level1_pairs = ceil_div(k, MX_K_ALIGN)
    for name, tensor, shape in (
        ("scale_a0", scale_a0, (m, level0_groups)),
        ("scale_b0", scale_b0, (level0_groups, n)),
        ("scale_a1", scale_a1, (m, level1_pairs, MX_SCALE_PAIR)),
        ("scale_b1", scale_b1, (n, level1_pairs, MX_SCALE_PAIR)),
    ):
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
    return m, n, k, has_bias


def _prepare_bias_arg(bias, n, device):
    """Normalize bias for the kernel call: validate the shape, or return a
    length-0 placeholder when no bias is used (ABI contract)."""
    if bias is None:
        return torch.empty(0, dtype=torch.float32, device=device)
    if tuple(bias.shape) != (n,):
        raise ValueError(f"bias must be float32 with shape ({n},)")
    return bias


def npu_dual_level_quant_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a0: torch.Tensor,
    scale_b0: torch.Tensor,
    scale_a1: torch.Tensor,
    scale_b1: torch.Tensor,
    *,
    bias: torch.Tensor = None,
    output_dtype=torch.float16,
):
    """Execute dual-level MXFP4 matmul with ND A and FRACTAL_NZ B.

    A is non-transposed and B is transposed. B uses int8 storage carrying
    packed FP4 bytes, copied to L1 without changing the NZ layout.
    """
    m, n, k, has_bias = _validate_dual_level_input(
        a,
        b,
        scale_a0,
        scale_b0,
        scale_a1,
        scale_b1,
        bias,
        output_dtype,
    )
    bias = _prepare_bias_arg(bias, n, a.device)
    output = torch.empty((m, n), dtype=output_dtype, device=a.device)
    stream = torch.npu.current_stream(a.device.index)
    aic_num = int(get_platform_info(stream=stream).cube_core_num)
    compiled = _build_aot_callable(
        m,
        n,
        k,
        has_bias,
        output_dtype,
        aic_num,
    )
    compiled(
        output,
        a.view(torch.int8),
        b,
        scale_a0,
        scale_b0,
        scale_a1,
        scale_b1,
        bias,
    )
    return output
