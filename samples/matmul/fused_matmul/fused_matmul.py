# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""C = act(A @ B.T + add) * mul, with optional per-column add/mul.

Host tiling selects the compute tile and buffer depths. The kernel uses a
Cube-only path for matmul/Bias/ReLU, or a Cube/Vector path for GELU/Mul.
"""

import os
import sys

import torch

from cannbotdsl import MemLoc, Tensor, dtypes, get_mem_size, make_layout
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang import control_flow
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg
from cannbotdsl.ops.arch import (
    get_block_idx,
    get_block_num,
    get_subblock_id,
    get_ctrl,
    set_ctrl,
)
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy, relu
from cannbotdsl.ops.sync import cube_sync_all
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import reinterpret, tile_slice

__all__ = ["fused_matmul"]

_MATMUL_SAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "matmul")
)
if _MATMUL_SAMPLE_DIR not in sys.path:
    sys.path.insert(0, _MATMUL_SAMPLE_DIR)


from matmul import MatmulTiling, ceil_align, ceil_div, floor_align  # noqa: E402


ACT_NONE = 0
ACT_RELU = 1
ACT_GELU = 2

# ============================================================================
# 1. Host-side Tiling
# ============================================================================


class FusedMatmulTiling(MatmulTiling):
    """Reuse matmul tiling, bound Vector UB usage, and reduce tail-core imbalance."""

    # Leave room for the per-column Add/Mul buffers beside the shared CV slots.
    UB_RESERVED_BYTES = 8 * 1024
    UB_TILE_ELEMENT_LIMIT = (get_mem_size("ub") - UB_RESERVED_BYTES) // 2
    CORE_LOAD_IMBALANCE_LIMIT = 1.2

    _TORCH_DTYPE_TO_DSL = {
        **MatmulTiling._TORCH_DTYPE_TO_DSL,
        torch.float32: (dtypes.float32, 4),
    }
    _DSL_DTYPE_TO_SIZE = {
        **MatmulTiling._DSL_DTYPE_TO_SIZE,
        dtypes.float32: (dtypes.float32, 4),
    }

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        a_dtype,
        b_dtype,
        c_dtype,
        *,
        need_vec: bool,
        has_bias: bool = False,
    ):
        self.need_vec = need_vec
        super().__init__(
            m,
            n,
            k,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=c_dtype,
            transpose_a=False,
            transpose_b=True,
            has_bias=has_bias,
        )

    def _compute(self):
        """Select the M/N tile before recomputing K tiling and buffer depths."""
        self._reset_base()
        base_mn_buffer_limit = self.L0C_SIZE
        if self.need_vec:
            # Search within the epilogue UB budget instead of clipping a larger tile afterwards.
            ub_tile_element_limit = self.UB_TILE_ELEMENT_LIMIT
            base_mn_buffer_limit = min(
                base_mn_buffer_limit, ub_tile_element_limit * self.DATA_SIZE_FP32
            )
        base_m_align_unit, base_n_align_unit, max_base_m, max_base_n = (
            self._cal_cube_bound_edge(
                self._get_hbm_bw(), self._get_l2_bw(), base_mn_buffer_limit
            )
        )
        # Keep a 128-byte K block in both double-buffered L0 operands.
        max_l0_outer_size = (
            self.L0A_SIZE
            // self.DB_SIZE
            // min(
                ceil_align(self.k, self.BASIC_BLOCK_16) * self.a_dtype_size,
                self.BASIC_BLOCK_K_128B,
            )
        )
        max_base_m = min(max_base_m, floor_align(max_l0_outer_size, base_m_align_unit))
        max_base_n = min(max_base_n, floor_align(max_l0_outer_size, base_n_align_unit))
        self.base_m = min(self.base_m, max_base_m)
        self.base_n = min(self.base_n, max_base_n)
        self.cube_bound_param = 1.0 / self.base_m + 1.0 / self.base_n
        self._search_optimal_base_mn(
            base_mn_buffer_limit,
            base_m_align_unit,
            base_n_align_unit,
            max_base_m,
            max_base_n,
        )
        self.base_m = min(ceil_align(self.m, self.BASIC_BLOCK_16), self.base_m)
        self.base_n = min(ceil_align(self.n, self.BASIC_BLOCK_16), self.base_n)
        if self.need_vec:
            self._limit_base_mn_for_vector()
        self._optimize_base_mn_for_load_balance()
        self._get_base_k()
        self._cal_l1_tiling()
        self._finalize()
        if self.need_vec:
            self.tile_vec_m = self.base_m // 2
            self.cv_buffer_num = min(
                2, ub_tile_element_limit // (self.base_m * self.base_n)
            )

    def _get_balance_rate_with_tail(self, base_m, base_n):
        """Score complete scheduling waves; this kernel does not split tail tiles."""
        total_tiles = ceil_div(self.m, base_m) * ceil_div(self.n, base_n)
        base_m = self.m if self.m <= self.BASIC_BLOCK_16 else base_m
        base_n = self.n if self.n <= self.BASIC_BLOCK_16 else base_n
        return (
            self.m
            * self.n
            / self.AIC_NUM
            / (ceil_div(total_tiles, self.AIC_NUM) * base_m * base_n)
        )

    def _limit_base_mn_for_vector(self):
        """Reserve at least one M fractal per AIV and keep its UB buffers in capacity."""
        self.base_m = max(self.base_m, 2 * self.BASIC_BLOCK_16)
        ub_tile_element_limit = self.UB_TILE_ELEMENT_LIMIT
        if self.base_m * self.base_n > ub_tile_element_limit:
            self.base_n = max(
                self.BASIC_BLOCK_16,
                floor_align(ub_tile_element_limit // self.base_m, self.BASIC_BLOCK_16),
            )
        if self.base_m * self.base_n > ub_tile_element_limit:
            self.base_m = max(
                2 * self.BASIC_BLOCK_16,
                floor_align(ub_tile_element_limit // self.base_n, self.BASIC_BLOCK_16),
            )

    def _get_core_load_imbalance(self, base_m: int, base_n: int) -> float:
        """Return maximum/mean output tiles per active core."""
        total_tiles = ceil_div(self.m, base_m) * ceil_div(self.n, base_n)
        if total_tiles <= self.AIC_NUM or total_tiles % self.AIC_NUM == 0:
            return 1.0
        return ceil_div(total_tiles, self.AIC_NUM) / (total_tiles / self.AIC_NUM)

    def _optimize_base_mn_for_load_balance(self):
        """Shrink N, then M, until core-load imbalance meets the configured limit."""
        min_base_m = 2 * self.BASIC_BLOCK_16 if self.need_vec else self.BASIC_BLOCK_16
        core_load_imbalance = self._get_core_load_imbalance(self.base_m, self.base_n)
        while core_load_imbalance > self.CORE_LOAD_IMBALANCE_LIMIT:
            if self.base_n > self.BASIC_BLOCK_16:
                self.base_n -= self.BASIC_BLOCK_16
            elif self.base_m > min_base_m:
                self.base_m -= self.BASIC_BLOCK_16
            else:
                break
            core_load_imbalance = self._get_core_load_imbalance(
                self.base_m, self.base_n
            )


# ============================================================================
# 2. Kernel
# ============================================================================

WINDOW_LEN = 4
FP32_VECTOR_LANES = 64
GELU_BETA = 0.044715
GELU_ALPHA = -1.5957691
MM_ROW_MAJOR_MASK = 1 << 51  # AscendC SetMMRowMajor uses CTRL[51].


@jit
def _tile_coords(
    tile_idx, n_tiles: int, main_window: int, main_row: int, tail_window: int
):
    """Use the same sliding-window order for Cube produce and Vector consume."""
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
    if row_idx % 2 != 0:
        n_idx = (n_tiles - 1) - n_idx
    return m_idx, n_idx


class FusedMatmulKernel:
    """Cube matmul with optional Bias/ReLU fusion or a split-M Vector epilogue."""

    def __init__(
        self,
        tiling: FusedMatmulTiling,
        *,
        has_add: bool,
        has_mul: bool,
        act_mode: int,
        out_dtype,
        add_dtype,
        mul_dtype,
    ):
        self.tiling = tiling
        self.has_add = has_add
        self.has_mul = has_mul
        self.act_mode = act_mode
        self.out_dtype = out_dtype
        self.add_dtype = add_dtype if has_add else out_dtype
        self.mul_dtype = mul_dtype if has_mul else out_dtype
        self.m_tiles = ceil_div(tiling.m, tiling.m_l1)
        self.n_tiles = ceil_div(tiling.n, tiling.n_l1)
        self.k_l1_tiles = ceil_div(tiling.k, tiling.k_l1)
        self.k_l0_per_l1 = ceil_div(tiling.k_l1, tiling.base_k)
        self.main_window = min(WINDOW_LEN, self.m_tiles)
        self.main_row = self.m_tiles // self.main_window - 1
        self.tail_window = self.m_tiles - self.main_row * self.main_window
        self.total_tiles = self.m_tiles * self.n_tiles
        # NOTALLOC_KEEP for operands without cross-tile reuse.
        self.l2_cache_ctl_a = 4 if tiling.l2_cache_disable in (1, 3) else 1
        self.l2_cache_ctl_b = 4 if tiling.l2_cache_disable in (2, 3) else 1

    @jit
    def _compute_output_tile(
        self,
        gm_a,
        gm_b,
        gm_c,
        gm_add,
        m_idx,
        n_idx,
        l1_a,
        l1_b,
        l0a,
        l0b,
        l0c,
        bias_l1,
        bias_bt,
        cv_ub,
    ):
        """Accumulate one output tile, then store it or hand it to both AIVs."""
        tiling = self.tiling
        actual_m = min(tiling.base_m, tiling.m - m_idx * tiling.base_m)
        actual_n = min(tiling.base_n, tiling.n - n_idx * tiling.base_n)
        gm_c_tile = gm_c[
            m_idx * tiling.base_m : m_idx * tiling.base_m + actual_m,
            n_idx * tiling.base_n : n_idx * tiling.base_n + actual_n,
        ]
        l0c_slot = l0c.next_write(bare=True).view(make_layout(gm_c_tile.shape))
        if const_expr(tiling.has_bias):
            bias_l1_view = reinterpret(
                bias_l1.next_write(), layout=make_layout((actual_n,))
            )
            mem_copy(bias_l1_view, tile_slice(gm_add, (tiling.base_n,), (n_idx,)))
            # Bias and MMAD must cover the same valid N columns.
            bias_full = bias_l1.next_read()
            bias_src = bias_bt.next_write(bare=True).view(make_layout((actual_n,)))
            mem_copy(bias_src, bias_full[: gm_c_tile.shape[1],])
        else:
            bias_src = l0c_slot  # Unused on the non-Bias path.
        for k_l1_idx in range(self.k_l1_tiles):
            gm_a_tile = tile_slice(
                gm_a, (tiling.base_m, tiling.k_l1), (m_idx, k_l1_idx)
            )
            gm_b_tile = tile_slice(
                gm_b, (tiling.base_n, tiling.k_l1), (n_idx, k_l1_idx)
            )
            l1_a_tensor = l1_a.next_write()[tuple(map(slice, gm_a_tile.shape))]
            l1_b_tensor = l1_b.next_write()[tuple(map(slice, gm_b_tile.shape))]
            mem_copy(
                l1_a_tensor, gm_a_tile, copy_options(l2_cache_ctl=self.l2_cache_ctl_a)
            )
            mem_copy(
                l1_b_tensor, gm_b_tile, copy_options(l2_cache_ctl=self.l2_cache_ctl_b)
            )
            for k_l0_idx in range(self.k_l0_per_l1):
                l1_a_slice = tile_slice(
                    l1_a_tensor, (tiling.base_m, tiling.base_k), (0, k_l0_idx)
                )
                l1_b_slice = tile_slice(
                    l1_b_tensor, (tiling.base_n, tiling.base_k), (0, k_l0_idx)
                )
                global_k_l0_idx = k_l1_idx * self.k_l0_per_l1 + k_l0_idx
                actual_k = (
                    tiling.base_k
                    if const_expr(tiling.k % tiling.k_l1 == 0)
                    else min(
                        tiling.base_k,
                        max(0, tiling.k - global_k_l0_idx * tiling.base_k),
                    )
                )
                l0a_tensor = l0a.next_write(bare=True).view(
                    make_layout((gm_c_tile.shape[0], actual_k))
                )
                l0b_tensor = l0b.next_write(bare=True).view(
                    make_layout((gm_c_tile.shape[1], actual_k))
                )
                mem_copy(l0a_tensor, l1_a_slice[: gm_c_tile.shape[0], :actual_k])
                mem_copy(l0b_tensor, l1_b_slice[: gm_c_tile.shape[1], :actual_k])
                # Padded zero-K MMADs are skipped; notify on the last valid block.
                is_final_k_block = global_k_l0_idx + 1 == ceil_div(
                    tiling.k, tiling.base_k
                )
                if const_expr(tiling.has_bias):
                    matmul(
                        l0c_slot,
                        l0a_tensor,
                        l0b_tensor,
                        init=(global_k_l0_idx == 0),
                        bias=bias_src,
                        unit_flag=3 if is_final_k_block else 2,
                    )
                else:
                    matmul(
                        l0c_slot,
                        l0a_tensor,
                        l0b_tensor,
                        init=(global_k_l0_idx == 0),
                        unit_flag=3 if is_final_k_block else 2,
                    )
        l0c_view = l0c_slot
        if const_expr(tiling.need_vec):
            mem_copy(
                cv_ub.next_write(), l0c_view.partition(dim=0), copy_options(unit_flag=3)
            )
        else:
            if const_expr(self.act_mode == ACT_RELU):
                mem_copy(gm_c_tile, relu(l0c_view), copy_options(unit_flag=3))
            else:
                mem_copy(gm_c_tile, l0c_view, copy_options(unit_flag=3))

    @jit
    def _compute_epilogue(self, gm_c, gm_add, gm_mul, cv_ub):
        """Consume Cube tiles in dispatch order and apply Add/activation/Mul in FP32."""
        tiling = self.tiling
        if const_expr(self.has_add):
            add_ub = RingBuffer(
                MemLoc.UB, self.add_dtype, make_layout((tiling.base_n,)), depth=2
            )
        if const_expr(self.has_mul):
            mul_ub = RingBuffer(
                MemLoc.UB, self.mul_dtype, make_layout((tiling.base_n,)), depth=2
            )
        subblock_id = get_subblock_id()
        block_idx, block_num = get_block_idx(), get_block_num()
        for tile_idx in range(block_idx, self.total_tiles, block_num):
            m_idx, n_idx = _tile_coords(
                tile_idx,
                self.n_tiles,
                self.main_window,
                self.main_row,
                self.tail_window,
            )
            m_start = m_idx * tiling.base_m
            n_start = n_idx * tiling.base_n
            actual_m = min(tiling.base_m, tiling.m - m_start)
            actual_n = min(tiling.base_n, tiling.n - n_start)
            first_subblock_rows = (actual_m + 1) // 2
            # Match FixPipe's ceil/floor split for odd M tails.
            subblock_rows = first_subblock_rows - subblock_id * (
                actual_m - (actual_m // 2) * 2
            )
            subblock_m_offset = subblock_id * first_subblock_rows
            if const_expr(self.has_add):
                add_ub_view = reinterpret(
                    add_ub.next_write(), layout=make_layout((actual_n,))
                )
                mem_copy(add_ub_view, tile_slice(gm_add, (tiling.base_n,), (n_idx,)))
            if const_expr(self.has_mul):
                mul_ub_view = reinterpret(
                    mul_ub.next_write(), layout=make_layout((actual_n,))
                )
                mem_copy(mul_ub_view, tile_slice(gm_mul, (tiling.base_n,), (n_idx,)))
            cv_slot = cv_ub.next_read()
            cv_view = reinterpret(
                cv_slot,
                layout=make_layout(
                    (subblock_rows, actual_n), stride=(tiling.base_n, 1)
                ),
            )
            # Forward traversal packs outputs into the already-consumed FP32 prefix.
            out_slot = reinterpret(
                cv_slot,
                dtype=self.out_dtype,
                layout=make_layout((tiling.tile_vec_m, tiling.base_n)),
                data_format="nd",
            )
            if const_expr(self.has_add):
                add_slot = add_ub.next_read()
            else:
                add_slot = out_slot
            if const_expr(self.has_mul):
                mul_slot = mul_ub.next_read()
            else:
                mul_slot = out_slot
            src_stride = cv_view.physical_stride[0]
            dst_stride = out_slot.physical_stride[0]
            with vf(mode="simd"):
                one = reg.vdups(1.0, dtypes.float32)
                for row in range(subblock_rows):
                    src_base = row * src_stride
                    dst_base = row * dst_stride
                    # Static columns expose independent vector groups to the compiler.
                    for col in control_flow.range(
                        0, tiling.base_n, FP32_VECTOR_LANES, unroll_full=True
                    ):
                        mask, _ = reg.update_mask(
                            actual_n - col if col < actual_n else 0, elem_bits=32
                        )
                        x = reg.vload(cv_view, src_base + col)
                        if const_expr(self.has_add):
                            bias = reg.vload_unpack(
                                add_slot, col, unpack_mode=reg.UnpackMode.B16_TO_B32
                            )
                            bias = reg.vcast(bias, dtypes.float32, mask=mask)
                            x = reg.vadd(x, bias, mask=mask)
                        if const_expr(self.act_mode == ACT_GELU):
                            x3 = reg.vmul(reg.vmul(x, x, mask=mask), x, mask=mask)
                            inner = reg.vadd(
                                x, reg.vmuls(x3, GELU_BETA, mask=mask), mask=mask
                            )
                            expo = reg.vexp(
                                reg.vmuls(inner, GELU_ALPHA, mask=mask), mask=mask
                            )
                            x = reg.vdiv(x, reg.vadd(expo, one, mask=mask), mask=mask)
                        if const_expr(self.act_mode == ACT_RELU):
                            x = reg.vrelu(x, mask=mask)
                        if const_expr(self.has_mul):
                            scale = reg.vload_unpack(
                                mul_slot, col, unpack_mode=reg.UnpackMode.B16_TO_B32
                            )
                            scale = reg.vcast(scale, dtypes.float32, mask=mask)
                            x = reg.vmul(x, scale, mask=mask)
                        if const_expr(self.out_dtype == dtypes.float32):
                            reg.vstore(out_slot, dst_base + col, x, mask)
                        else:
                            y = reg.vcast(
                                x,
                                self.out_dtype,
                                mask=mask,
                                rounding=reg.RoundingMode.RN,
                            )
                            reg.vstore_pack(
                                out_slot,
                                dst_base + col,
                                y,
                                mask,
                                pack_mode=reg.PackMode.B32_TO_B16,
                            )
                reg.vmem_bar("vst_vld")
            if subblock_rows > 0:
                output_m_start = m_start + subblock_m_offset
                gm_c_rows = gm_c[
                    output_m_start : output_m_start + subblock_rows,
                    n_start : n_start + actual_n,
                ]
                output_ub_view = reinterpret(
                    out_slot,
                    layout=make_layout(
                        (subblock_rows, actual_n), stride=(tiling.base_n, 1)
                    ),
                )
                mem_copy(gm_c_rows, output_ub_view)

    @kernel
    def fused_kernel(
        self, gm_a: Tensor, gm_b: Tensor, gm_c: Tensor, gm_add: Tensor, gm_mul: Tensor
    ):
        """Declare buffers, dispatch Cube tiles, and run the matching Vector epilogue."""
        tiling = self.tiling
        if const_expr(not tiling.need_vec):
            # Match AscendC SetMMRowMajor on Cube-only output paths.
            saved_ctrl = get_ctrl()
            set_ctrl(
                saved_ctrl
                + (1 - saved_ctrl // MM_ROW_MAJOR_MASK % 2) * MM_ROW_MAJOR_MASK
            )
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
            depth=tiling.DB_SIZE,
            data_format="nz",
        )
        l0b = RingBuffer(
            MemLoc.L0B,
            tiling.b_dtype,
            make_layout((tiling.base_n, tiling.base_k)),
            depth=tiling.DB_SIZE,
            data_format="nz",
        )
        l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((tiling.base_m, tiling.base_n)),
            depth=tiling.l0c_db,
        )
        if const_expr(tiling.has_bias):
            bias_l1 = RingBuffer(
                MemLoc.L1,
                self.add_dtype,
                make_layout((tiling.base_n,)),
                depth=2,
                data_format="nd",
            )
            bias_bt = RingBuffer(
                MemLoc.BIAS, dtypes.float32, make_layout((tiling.base_n,)), depth=2
            )
        else:
            bias_l1, bias_bt = l0c, l0c
        if const_expr(tiling.need_vec):
            cv_ub = RingBuffer(
                MemLoc.UB,
                dtypes.float32,
                make_layout((tiling.tile_vec_m, tiling.base_n)),
                depth=tiling.cv_buffer_num,
            )
        else:
            cv_ub = l0c
        block_idx, block_num = get_block_idx(), get_block_num()
        for tile_idx in range(block_idx, self.total_tiles, block_num):
            m_idx, n_idx = _tile_coords(
                tile_idx,
                self.n_tiles,
                self.main_window,
                self.main_row,
                self.tail_window,
            )
            self._compute_output_tile(
                gm_a,
                gm_b,
                gm_c,
                gm_add,
                m_idx,
                n_idx,
                l1_a,
                l1_b,
                l0a,
                l0b,
                l0c,
                bias_l1,
                bias_bt,
                cv_ub,
            )
        # Keep subblock-ID operations in an AIV @jit region, not the AIC clone.
        if const_expr(tiling.need_vec):
            if const_expr(self.has_add):
                gm_add_arg = gm_add
            else:
                gm_add_arg = gm_c
            if const_expr(self.has_mul):
                gm_mul_arg = gm_mul
            else:
                gm_mul_arg = gm_c
            self._compute_epilogue(gm_c, gm_add_arg, gm_mul_arg, cv_ub)
        if const_expr(not tiling.need_vec):
            cube_sync_all()
            set_ctrl(saved_ctrl)

    @host
    def run(
        self, gm_a: Tensor, gm_b: Tensor, gm_c: Tensor, gm_add: Tensor, gm_mul: Tensor
    ):
        """Launch the fused kernel on the cores selected by host tiling."""
        self.fused_kernel[self.tiling.used_core_num](gm_a, gm_b, gm_c, gm_add, gm_mul)


# ============================================================================
# 3. Torch Interface
# ============================================================================

_TORCH_TO_DSL_DTYPE = {
    torch.float16: dtypes.float16,
    torch.bfloat16: dtypes.bfloat16,
    torch.float32: dtypes.float32,
}
_ACT_CODES = {"none": ACT_NONE, "relu": ACT_RELU, "gelu": ACT_GELU}
_SHAPE_MIN = 16
_SHAPE_LIMIT = 16384


def _validate_fused_matmul_inputs(a, b, add, mul, activation, out_dtype):
    """Validate the public contract without changing input storage or layout."""
    if a.dtype != b.dtype:
        raise TypeError(f"a and b must have the same dtype, got {a.dtype} vs {b.dtype}")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(
            f"inputs must be 2-D, got a.dim()={a.dim()}, b.dim()={b.dim()}"
        )
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"inputs must be fp16/bf16, got {a.dtype}")
    if activation not in _ACT_CODES:
        raise ValueError(
            f"activation must be one of {sorted(_ACT_CODES)}, got {activation!r}"
        )
    m, k = a.shape
    n = b.shape[0]
    if b.shape[1] != k:
        raise ValueError(f"reduction mismatch: a is (M,{k}) but b is (N,{b.shape[1]})")
    for dim_value, name in ((m, "M"), (n, "N"), (k, "K")):
        if not (_SHAPE_MIN <= dim_value <= _SHAPE_LIMIT):
            raise ValueError(
                f"{name}={dim_value} outside supported range [{_SHAPE_MIN}, {_SHAPE_LIMIT}]"
            )
    for vec_in, name in ((add, "add"), (mul, "mul")):
        if vec_in is not None:
            if vec_in.dim() != 1 or vec_in.shape[0] != n:
                raise ValueError(
                    f"{name} must have shape ({n},), got {tuple(vec_in.shape)}"
                )
            if vec_in.dtype not in (torch.float16, torch.bfloat16):
                raise TypeError(f"{name} must be fp16/bf16, got {vec_in.dtype}")
    if out_dtype is None:
        out_dtype = a.dtype
    elif out_dtype not in (torch.float32, a.dtype):
        raise TypeError(
            f"out_dtype must be None, torch.float32 or the input dtype, got {out_dtype}"
        )
    for tensor, name in ((a, "a"), (b, "b"), (add, "add"), (mul, "mul")):
        if tensor is not None:
            if tensor.device != a.device:
                raise ValueError("a, b, add and mul must be on the same device")
            if name in ("a", "b") and not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
    if a.device.type != "npu":
        raise ValueError("a and b must be NPU tensors")
    return m, n, k, out_dtype


def fused_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    add: torch.Tensor = None,
    mul: torch.Tensor = None,
    activation: str = "none",
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    """Compute act(A[M,K] @ B[N,K].T + add[N]) * mul[N].

    Inputs and optional column vectors use FP16/BF16. Output defaults to the
    input dtype, or FP32 when requested. Activation is none, relu, or GELU.
    """
    m, n, k, out_dtype = _validate_fused_matmul_inputs(
        a, b, add, mul, activation, out_dtype
    )
    act_mode = _ACT_CODES[activation]
    has_bias = add is not None and mul is None and act_mode in (ACT_NONE, ACT_RELU)
    need_vec = act_mode == ACT_GELU or mul is not None
    tiling = FusedMatmulTiling(
        m,
        n,
        k,
        a_dtype=a.dtype,
        b_dtype=b.dtype,
        c_dtype=out_dtype,
        need_vec=need_vec,
        has_bias=has_bias,
    )
    c = torch.empty(m, n, dtype=out_dtype, device=a.device)
    op = FusedMatmulKernel(
        tiling,
        has_add=add is not None,
        has_mul=mul is not None,
        act_mode=act_mode,
        out_dtype=_TORCH_TO_DSL_DTYPE[out_dtype],
        add_dtype=_TORCH_TO_DSL_DTYPE[add.dtype] if add is not None else None,
        mul_dtype=_TORCH_TO_DSL_DTYPE[mul.dtype] if mul is not None else None,
    )
    op.run(a, b, c, add, mul)
    return c
