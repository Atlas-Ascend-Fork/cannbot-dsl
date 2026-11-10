# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MX-quantized GroupedMatmulFinalizeRouting (Ascend 950 / DAV_3510).

Interface and semantics aligned with the mainline A5 implementation of
aclnnGroupedMatmulFinalizeRoutingV3 (ops-transformer, arch35 MX path):

    y[rowIndex[i], :] += logit[i] * (dequant(x1)[i, :] @ dequant(x2)[g(i)])
    y[off:off+bsdp, :]  = sharedInputWeight * sharedInput[:, :]

MX ABI (matches the mainline 950 layout contract):
    x1             (m, k)               FP8 E4M3FN / E5M2, row-major
    x2             (e, k, n)            FP8 E4M3FN / E5M2, k-major
    pertokenScale  (m, ceil(k/64), 2)   E8M0 (per-row paired scales)
    scale          (e, ceil(k/64), n, 2) E8M0 (per-expert, per-column)
    bias           (e, n)               BF16 (optional)
    sharedInput    (bsdp, n)            BF16 (optional)
    logit          (m,)                 FP32
    rowIndex       (m,)                 INT64
    out            (batch, n)           FP32

Kernel structure (MIX AIC:AIV = 1:2, single forward handoff):
    AIC: GM->L1->L0A/L0B MX MMAD (mx_scale paired loads, bias via BT)
         -> FIXPIPE drain L0C->UB cross-core (partition(dim=0) split-M)
    AIV: consume UB tile -> per-row logit multiply (VF) -> per-row
         MTE3 atomic add into y[rowIndex[i]] (copy_options(atomic_add=True))

The y initialization (zero fill + shared-expert blend) is done by the
wrapper with torch ops on the same stream before the kernel launch, which
orders it before every atomic add (the mainline runs the same logic in an
AIV prologue before the main loop).
"""

__all__ = ["grouped_matmul_finalize_routing"]

import dataclasses
import functools
import types

import torch
import torch_npu  # noqa: F401

from cannbotdsl import (
    Buffer,
    MemLoc,
    Tensor,
    TensorSpec,
    compile as dsl_compile,
    dtypes,
    get_mem_size,
    get_platform_info,
    is_aic,
    make_layout,
    vf,
)
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx, get_subblock_id
from cannbotdsl.ops.matmul import matmul as dsl_matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops.reg import (
    create_mask,
    vload,
    vload_broadcast,
    vmul,
    vstore,
    vmem_bar,
)

from cannbotdsl.ops.memcpy import cube_fill_l1_zero
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import ceil_div, tile_slice

from cannbotdsl.tensor import from_tensor


# ============================================================================
# Constants
# ============================================================================

L1_CAPACITY_BYTES = get_mem_size("l1")
L0A_CAPACITY_BYTES = get_mem_size("l0a")
L0B_CAPACITY_BYTES = get_mem_size("l0b")
L0C_CAPACITY_BYTES = get_mem_size("l0c")

MX_GROUP_SIZE = 32  # One E8M0 scale is shared by every 32 K elements.
MX_K_ALIGN = 64  # Every 64 K elements map to one paired-scale group.
MX_SCALE_PAIR = 2

BASE_M_DEFAULT = 128
BASE_N_DEFAULT = 256
BASE_K_DEFAULT = 128
VREG_ELEM_NUM = 64  # f32 elements per vector register (256B)

GROUPLIST_TYPE_CUMSUM = 0
GROUPLIST_TYPE_COUNT = 1

_TORCH_X_DTYPE = (torch.float8_e4m3fn, torch.float8_e5m2)
_TORCH_X_TO_DSL = {
    torch.float8_e4m3fn: dtypes.float8_e4m3fn,
    torch.float8_e5m2: dtypes.float8_e5m2,
}


def _ceil_div(a, b):
    return (a + b - 1) // b


def _ceil_align(a, b):
    return _ceil_div(a, b) * b


def _scale_k_len(k):
    """Contiguous paired-scale length covering a K extent."""
    return _ceil_div(k, MX_K_ALIGN) * MX_SCALE_PAIR


# ============================================================================
# 1. Host-side tiling
# ============================================================================


class GmmFrTiling:
    """Host-side tiling.

    base_m/base_n: cube tile and the cross-core handoff slot shape
    (base_m/2 x base_n f32 per AIV, depth 2 -> 128 KB of the 248 KB UB).
    base_k: L0 K block (128 FP8 values = 128B MTE2 granularity).
    k_l1: K window staged per L1 pass with double-buffered A/B/scales.
    n_tiles: N tile count (shared by every group).
    """

    def __init__(self, m_raw, k, n, e, group_sizes, has_bias, aic_num):
        self.k = k
        self.n = n
        self.e = e
        self.m_raw = m_raw
        self.has_bias = has_bias

        self.base_m = BASE_M_DEFAULT
        self.base_n = min(BASE_N_DEFAULT, _ceil_align(n, 64)) if n else BASE_N_DEFAULT
        self.base_k = min(_ceil_align(k, 64), BASE_K_DEFAULT)
        # L0A/L0B hold two base_k blocks.
        l0a_cap = L0A_CAPACITY_BYTES // 2 // self.base_m
        l0b_cap = L0B_CAPACITY_BYTES // 2 // self.base_n
        self.base_k = min(self.base_k, (l0a_cap // 16) * 16, (l0b_cap // 16) * 16)
        if self.base_k < 16:
            raise ValueError("base_k underflow: tile too large for L0A/L0B")

        # k_l1: largest double-buffered window (data + scales + bias) in L1.
        self.k_l1 = self.base_k
        for step in (4, 3, 2, 1):
            k_l1 = step * self.base_k
            scale_len = k_l1 // MX_GROUP_SIZE
            used = (
                2 * self.base_m * k_l1
                + 2 * k_l1 * self.base_n
                + 2 * self.base_m * scale_len
                + 2 * scale_len * self.base_n
                + (self.base_n * 4 * 2 if has_bias else 0)
            )
            if used <= L1_CAPACITY_BYTES:
                self.k_l1 = k_l1
                break

        # full_load: keep one m-tile's A panel and paired scales resident in
        # L1 and stream only B across the m-tile's n-tiles. Requires the
        # scheduler to hand each block a contiguous tile run (so every
        # n-tile of an (g, mt) lands on the block that staged A) and a
        # 64-aligned K (the resident panel spans ceil(k/64) scale columns).
        self.full_load = False
        self.a_k_len = _ceil_align(k, MX_K_ALIGN)
        self.scale_k_len_full = _scale_k_len(self.a_k_len)
        if k % MX_K_ALIGN == 0:
            a_full_bytes = self.base_m * self.a_k_len
            sa_full_bytes = self.base_m * self.scale_k_len_full
            fixed_bytes = (
                a_full_bytes + sa_full_bytes + (self.base_n * 4 * 4 if has_bias else 0)
            )
            for step in (3, 2, 1):
                k_l1 = step * self.base_k
                scale_len = k_l1 // MX_GROUP_SIZE
                used = (
                    fixed_bytes + 2 * k_l1 * self.base_n + 2 * scale_len * self.base_n
                )
                if used <= L1_CAPACITY_BYTES:
                    self.full_load = True
                    self.k_l1 = k_l1
                    break

        self.n_tiles = _ceil_div(n, self.base_n)
        self.scale_group_total = _ceil_div(k, MX_K_ALIGN)
        self.scale_k_l1_len = self.k_l1 // MX_GROUP_SIZE

        total_tiles = sum(
            _ceil_div(m_g, self.base_m) * self.n_tiles for m_g in group_sizes
        )
        self.used_core_num = min(total_tiles, aic_num)


# ============================================================================
# 2. MIX kernel (AIC cube path + AIV routing epilogue)
# ============================================================================


@kernel
class GmmFrKernel:
    """One forward Cube->Vec handoff per output tile.

    Both core types walk the identical group/tile sequence derived from
    group_list, so the cross-core ring FIFOs stay aligned tile by tile:
    the AIC produces exactly once per tile and both AIVs consume exactly
    once per tile (empty halves still advance the cursor).
    """

    def __init__(self, tiling, a_dtype, b_dtype, group_list_type):
        self.t = tiling
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.is_cumsum = group_list_type == GROUPLIST_TYPE_CUMSUM

    def __call__(
        self,
        x1_gm: Tensor,
        x2_gm: Tensor,
        scale_a_gm: Tensor,
        scale_b_gm: Tensor,
        bias_gm: Tensor,
        logit_gm: Tensor,
        row_idx_gm: Tensor,
        group_list_gm: Tensor,
        y_gm: Tensor,
        params_gm: Tensor,
    ):
        t = self.t
        block_idx = get_block_idx()
        block_num = t.used_core_num
        n_tiles = t.n_tiles

        # On-chip channels (ST101: locals, handed to the @jit tile bodies).
        half_rows = t.base_m // 2
        if const_expr(t.full_load):
            # Resident A panel + paired scales for the current (g, mt);
            # written once per m-tile, read across all its n-tiles.
            l1_a = RingBuffer(
                MemLoc.L1, self.a_dtype, make_layout((t.base_m, t.a_k_len)), depth=1
            )
            l1_scale_a = RingBuffer(
                MemLoc.L1,
                dtypes.float8_e8m0,
                make_layout((t.scale_k_len_full, t.base_m)),
                depth=1,
            )
        else:
            l1_a = RingBuffer(
                MemLoc.L1, self.a_dtype, make_layout((t.base_m, t.k_l1)), depth=2
            )
            l1_scale_a = RingBuffer(
                MemLoc.L1,
                dtypes.float8_e8m0,
                make_layout((t.scale_k_l1_len, t.base_m)),
                depth=2,
            )
        l1_b = RingBuffer(
            MemLoc.L1, self.b_dtype, make_layout((t.k_l1, t.base_n)), depth=2
        )
        l1_scale_b = RingBuffer(
            MemLoc.L1,
            dtypes.float8_e8m0,
            make_layout((t.scale_k_l1_len, t.base_n)),
            depth=2,
        )
        l1_bias = None
        l1_bt = None
        if const_expr(t.has_bias):
            l1_bias = RingBuffer(
                MemLoc.L1,
                dtypes.float32,
                make_layout((t.base_n,)),
                depth=2,
                data_format="nd",
            )
            l1_bt = RingBuffer(
                MemLoc.BIAS, dtypes.float32, make_layout((t.base_n,)), depth=2
            )
        l0a = RingBuffer(
            MemLoc.L0A, self.a_dtype, make_layout((t.base_m, t.base_k)), depth=2
        )
        l0b = RingBuffer(
            MemLoc.L0B, self.b_dtype, make_layout((t.base_n, t.base_k)), depth=2
        )
        l0c = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((t.base_m, t.base_n)), depth=2
        )
        cv_ub = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((half_rows, t.base_n)),
            depth=2,
            data_format="nd",
        )
        logit_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((half_rows,)))
        out_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((half_rows, t.base_n)))

        cube_ctx = (
            x1_gm,
            x2_gm,
            scale_a_gm,
            scale_b_gm,
            bias_gm,
            l1_a,
            l1_b,
            l1_scale_a,
            l1_scale_b,
            l1_bias,
            l1_bt,
            l0a,
            l0b,
            l0c,
            cv_ub,
        )
        vec_ctx = (y_gm, logit_gm, row_idx_gm, cv_ub, logit_ub, out_ub)

        # Contiguous-run tile scheduler: each block takes one contiguous
        # span of the global tile stream (g-major, m-tile, n-tile), so all
        # n-tiles of an (g, mt) land on the block that staged its resident
        # A panel (full_load). Runs differ by at most one tile, keeping the
        # per-core load balanced. The host precomputes the global tile
        # count into params_gm, so each core reads it once instead of
        # walking group_list twice. Both core types run the identical
        # arithmetic, so the cross-core ring FIFOs stay aligned tile by tile.
        total_tiles = params_gm[0]
        base_run = total_tiles // block_num
        extra_tiles = total_tiles - base_run * block_num
        run_len = base_run + (1 if block_idx < extra_tiles else 0)
        run_start = block_idx * base_run + min(block_idx, extra_tiles)
        run_end = run_start + run_len

        stream_pos = 0
        pre_offset = 0
        for group_idx in range(t.e):
            group_list_val = group_list_gm[group_idx]
            m_g = group_list_val - pre_offset if self.is_cumsum else group_list_val
            group_start = pre_offset
            pre_offset = pre_offset + m_g

            if m_g > 0:
                group_tiles = ceil_div(m_g, t.base_m) * n_tiles
                lo = run_start - stream_pos
                hi = run_end - stream_pos
                if lo < 0:
                    lo = 0
                if hi > group_tiles:
                    hi = group_tiles
                for index in range(lo, hi):
                    mt = index // n_tiles
                    nt = index - mt * n_tiles
                    m_extent = min(t.base_m, m_g - mt * t.base_m)
                    n_extent = min(t.base_n, t.n - nt * t.base_n)
                    if is_aic():
                        # Stage the resident A panel at the first n-tile of
                        # an (g, mt) and at the run's very first tile (a run
                        # may start mid-m-tile staged by another block).
                        stage_a = 1 if nt == 0 else 0
                        if stream_pos + index == run_start:
                            stage_a = 1
                        # A single-m-tile group reads every B byte exactly
                        # once: stream it without allocating L2 lines.
                        single_mt = 1 if m_g <= t.base_m else 0
                        self._cube_tile(
                            cube_ctx,
                            group_idx,
                            group_start,
                            mt,
                            nt,
                            m_extent,
                            n_extent,
                            stage_a,
                            single_mt,
                        )
                    else:
                        self._vec_tile(
                            vec_ctx,
                            group_start,
                            mt,
                            nt,
                            m_extent,
                            n_extent,
                        )
                stream_pos = stream_pos + group_tiles

    @jit
    def _zero_b_k_tail(self, slot, valid_k, physical_k, outer, c0):
        # Pad the staged k-major B window to the 64-element scale grid with
        # zeros so the full-width MMAD's beyond-K contribution annihilates.
        padded_k = _ceil_align(valid_k, MX_K_ALIGN)
        missing_rows = padded_k - valid_k
        cube_fill_l1_zero(
            slot,
            offset_elems=valid_k * c0,
            repeat=_ceil_div(outer, c0),
            blk_num=missing_rows,
            dst_gap=physical_k - missing_rows,
        )

    @jit
    def _cube_tile(
        self,
        ctx,
        group_idx,
        group_start,
        mt,
        nt,
        m_extent,
        n_extent,
        stage_a,
        single_mt,
    ):
        t = self.t
        (
            x1_gm,
            x2_gm,
            scale_a_gm,
            scale_b_gm,
            bias_gm,
            l1_a,
            l1_b,
            l1_scale_a,
            l1_scale_b,
            l1_bias,
            l1_bt,
            l0a,
            l0b,
            l0c,
            cv_ub,
        ) = ctx

        m_start = group_start + mt * t.base_m
        n_start = nt * t.base_n
        m_end = m_start + m_extent
        n_end = n_start + n_extent
        # M view extent for the split-M dual-destination handoff: the
        # fractal frame (m1 = m_view/16) must halve evenly and the view must
        # stay inside the (base_m, base_n) L0C slot, so pad up to a multiple
        # of 32 and clamp to base_m. Padded rows carry garbage the AIV side
        # never reads.
        m_padded = m_extent + 31 - (m_extent + 31) % 32
        m_view = min(t.base_m, max(32, m_padded))

        if const_expr(t.full_load):
            # Stage the resident A panel + paired scales once per (g, mt);
            # every later n-tile of this m-tile reuses them from L1.
            if stage_a > 0:
                mem_copy(l1_a.next_write(), x1_gm[m_start:m_end, 0 : t.a_k_len])
                scale_a_full = scale_a_gm[
                    m_start:m_end, 0 : t.scale_group_total, 0:MX_SCALE_PAIR
                ]
                mem_copy(
                    l1_scale_a.next_write().permute(1, 0),
                    scale_a_full.flatten(-2, -1),
                )

        bias = None
        if const_expr(t.has_bias):
            bias_tile = bias_gm[group_idx * t.n + n_start : group_idx * t.n + n_end,]
            mem_copy(l1_bias.next_write(), bias_tile)
            mem_copy(l1_bt.next_write(), l1_bias.next_read())
            bias = l1_bt.next_read()

        l0c_view = from_tensor(
            l0c.next_write(),
            layout=make_layout((m_view, n_extent)),
            dtype=dtypes.float32,
            data_format="nz",
        )

        scale_k_l0_len = t.base_k // MX_GROUP_SIZE
        step_k = t.k_l1 // t.base_k
        k_l1_tiles = ceil_div(t.k, t.k_l1)
        for k_l1_idx in range(k_l1_tiles):
            k_start = k_l1_idx * t.k_l1
            k_end = (
                k_start + t.k_l1
                if const_expr(t.k % t.k_l1 == 0)
                else min(k_start + t.k_l1, t.k)
            )

            # GM -> L1 data staging (clipped to the real extents; the L1 tail
            # beyond a clipped extent is either zeroed (B, below) or sliced
            # away by the L0 views). full_load streams only B.
            if const_expr(not t.full_load):
                a_gm_tile = x1_gm[m_start:m_end, k_start:k_end]
                mem_copy(l1_a.next_write(), a_gm_tile)
            b_gm_tile = x2_gm[
                group_idx * t.k + k_start : group_idx * t.k + k_end,
                n_start:n_end,
            ]
            b_write = l1_b.next_write()
            mem_copy(
                b_write,
                b_gm_tile,
                options=copy_options(l2_cache_ctl=dtypes.int32(4 if single_mt else 0)),
            )
            if const_expr(t.k % MX_K_ALIGN != 0):
                if k_end == t.k:
                    self._zero_b_k_tail(
                        b_write,
                        k_end - k_start,
                        t.k_l1,
                        t.base_n,
                        self.b_dtype.c0_elems,
                    )

            # GM -> L1 paired-scale staging for this K window. The window's
            # group range is half-open on ceil(k_end/64): a partial final
            # 64-group still contributes its paired scale bytes.
            scale_group_end = _ceil_div(k_end, MX_K_ALIGN)
            if const_expr(not t.full_load):
                scale_a_gm_tile = scale_a_gm[
                    m_start:m_end,
                    k_start // MX_K_ALIGN : scale_group_end,
                    0:MX_SCALE_PAIR,
                ]
                mem_copy(
                    l1_scale_a.next_write().permute(1, 0),
                    scale_a_gm_tile.flatten(-2, -1),
                )
            scale_b_gm_tile = scale_b_gm[
                group_idx * t.scale_group_total + k_start // MX_K_ALIGN : group_idx
                * t.scale_group_total
                + scale_group_end,
                n_start:n_end,
                0:MX_SCALE_PAIR,
            ]
            mem_copy(
                l1_scale_b.next_write(),
                scale_b_gm_tile.permute((0, 2, 1)).flatten(0, 1),
            )

            a_l1_rd = l1_a.next_read()
            b_l1_rd = l1_b.next_read()
            scale_a_l1_rd = l1_scale_a.next_read().permute(1, 0)
            scale_b_l1_rd = l1_scale_b.next_read()

            current_k_l1 = (
                t.k_l1 if const_expr(t.k % t.k_l1 == 0) else min(t.k_l1, t.k - k_start)
            )
            current_step_k = (
                step_k
                if const_expr(t.k % t.k_l1 == 0)
                else ceil_div(current_k_l1, t.base_k)
            )
            for k_l0_idx in range(current_step_k):
                global_k_l0_idx = k_l1_idx * step_k + k_l0_idx
                valid_k = (
                    t.base_k
                    if const_expr(t.k % t.base_k == 0)
                    else min(t.base_k, t.k - global_k_l0_idx * t.base_k)
                )
                valid_scale_k = _ceil_div(valid_k, MX_K_ALIGN) * MX_SCALE_PAIR

                # L1 -> L0A with the paired ScaleA view (m-major operand).
                # full_load indexes the resident full-K panel globally.
                a_k_l0_idx = global_k_l0_idx if const_expr(t.full_load) else k_l0_idx
                a_l1_tile = tile_slice(a_l1_rd, (t.base_m, t.base_k), (0, a_k_l0_idx))
                scale_a_l1_tile = tile_slice(
                    scale_a_l1_rd, (t.base_m, scale_k_l0_len), (0, a_k_l0_idx)
                )
                a_source = a_l1_tile[:m_view, :valid_k]
                a_ready = from_tensor(
                    l0a.next_write(), layout=make_layout(a_source.shape)
                )
                mem_copy(
                    a_ready,
                    a_source,
                    options=copy_options(
                        mx_scale=scale_a_l1_tile[:m_view, :valid_scale_k]
                    ),
                )

                # L1 -> L0B with the paired ScaleB view (k-major staging is
                # transposed by the permuted source view).
                b_l1_tile = tile_slice(b_l1_rd, (t.base_k, t.base_n), (k_l0_idx, 0))
                scale_b_l1_tile = tile_slice(
                    scale_b_l1_rd, (scale_k_l0_len, t.base_n), (k_l0_idx, 0)
                )
                b_source = b_l1_tile.permute(1, 0)[:n_extent, :valid_k]
                b_ready = from_tensor(
                    l0b.next_write(), layout=make_layout(b_source.shape)
                )
                mem_copy(
                    b_ready,
                    b_source,
                    options=copy_options(
                        mx_scale=scale_b_l1_tile[:valid_scale_k, :n_extent]
                    ),
                )

                is_final_k = global_k_l0_idx + 1 == _ceil_div(t.k, t.base_k)
                dsl_matmul(
                    l0c_view,
                    a_ready,
                    b_ready,
                    init=(global_k_l0_idx == 0),
                    bias=bias,
                    unit_flag=3 if is_final_k else 2,
                    mx_mode=True,
                )

        # FIXPIPE drain L0C -> UB, split-M dual destination onto both AIVs.
        mem_copy(
            cv_ub.next_write(),
            l0c_view.partition(dim=0),
            options=copy_options(unit_flag=3),
        )

    @jit
    def _vec_tile(self, ctx, group_start, mt, nt, m_extent, n_extent):
        t = self.t
        y_gm, logit_gm, row_idx_gm, cv_ub, logit_ub, out_ub = ctx
        sub = get_subblock_id()
        m_padded = m_extent + 31 - (m_extent + 31) % 32
        m_view = min(t.base_m, max(32, m_padded))
        half = m_view // 2
        n_start = nt * t.base_n

        # Every sub-block advances the cursor exactly once per tile even
        # when its half holds no valid row (empty pieces would misalign
        # the ring's write/read FIFO).
        slot = cv_ub.next_read()

        valid = m_extent - sub * half
        if valid > half:
            valid = half

        slot_stride = slot.stride[0]
        out_stride = t.base_n
        rows_lo = group_start + mt * t.base_m + sub * half

        # Logit staging for the whole delivered half whenever those rows
        # exist inside x1; an end-overflowing half stages a clamped dummy
        # range (its rows are scaled but never written out).
        hi = rows_lo + half
        if hi > t.m_raw:
            hi = t.m_raw
        lo = rows_lo
        if lo >= hi:
            lo = hi - 1
        mem_copy(
            logit_ub[0 : hi - lo,],
            logit_gm[lo:hi,],
        )

        # Both halves always traverse the delivered slot rows in full: the
        # cross-core ring's consume handshake derives from actual Tensor
        # accesses, so short or skipped traversals can stall the producer.
        # Garbage rows scale into out_ub but are never atomically added.
        n_chunks = ceil_div(n_extent, VREG_ELEM_NUM)
        with vf():
            mask32 = create_mask(pattern="all", elem_bits=32)
            for i in range(half):
                lg = vload_broadcast(logit_ub, i)
                for c in range(n_chunks):
                    row = vload(slot, i * slot_stride + c * VREG_ELEM_NUM)
                    vstore(
                        out_ub,
                        i * out_stride + c * VREG_ELEM_NUM,
                        vmul(row, lg, mask=mask32),
                        mask=mask32,
                    )
            vmem_bar("vst_vld")

        for i in range(valid):
            dst_row = row_idx_gm[rows_lo + i]
            mem_copy(
                y_gm[dst_row, n_start : n_start + n_extent],
                out_ub[i, 0:n_extent],
                options=copy_options(atomic_add=True),
            )


# ============================================================================
# 3. Launcher + AOT compilation
# ============================================================================


class _GmmFrLauncher:
    def __init__(self, tiling, a_dtype, b_dtype, group_list_type):
        self.tiling = tiling
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.group_list_type = group_list_type

    @host
    def run(
        self,
        x1_gm: Tensor,
        x2_gm: Tensor,
        scale_a_gm: Tensor,
        scale_b_gm: Tensor,
        bias_gm: Tensor,
        logit_gm: Tensor,
        row_idx_gm: Tensor,
        group_list_gm: Tensor,
        y_gm: Tensor,
        params_gm: Tensor,
    ):
        GmmFrKernel(self.tiling, self.a_dtype, self.b_dtype, self.group_list_type)[
            self.tiling.used_core_num
        ](
            x1_gm,
            x2_gm,
            scale_a_gm,
            scale_b_gm,
            bias_gm,
            logit_gm,
            row_idx_gm,
            group_list_gm,
            y_gm,
            params_gm,
        )


@functools.lru_cache(maxsize=64)
def _compile_gmm_fr(
    m_raw,
    k,
    n,
    e,
    batch,
    a_dtype_name,
    b_dtype_name,
    has_bias,
    group_list_type,
    base_m,
    base_n,
    base_k,
    k_l1,
    full_load,
    used_core_num,
):
    a_dtype = getattr(dtypes, a_dtype_name)
    b_dtype = getattr(dtypes, b_dtype_name)
    scale_groups = _ceil_div(k, MX_K_ALIGN)

    # Rebuild the tiling from the cache key's primitive fields.
    tiling = types.SimpleNamespace(
        k=k,
        n=n,
        e=e,
        m_raw=m_raw,
        has_bias=has_bias,
        base_m=base_m,
        base_n=base_n,
        base_k=base_k,
        k_l1=k_l1,
        n_tiles=_ceil_div(n, base_n),
        scale_group_total=scale_groups,
        scale_k_l1_len=k_l1 // MX_GROUP_SIZE,
        full_load=full_load,
        a_k_len=_ceil_align(k, MX_K_ALIGN),
        scale_k_len_full=_scale_k_len(_ceil_align(k, MX_K_ALIGN)),
        used_core_num=used_core_num,
    )

    bias_size = (e * n,) if has_bias else (1,)
    specs = (
        TensorSpec((m_raw, k), a_dtype),
        TensorSpec((e * k, n), b_dtype),
        TensorSpec((m_raw, scale_groups, MX_SCALE_PAIR), dtypes.float8_e8m0),
        TensorSpec((e * scale_groups, n, MX_SCALE_PAIR), dtypes.float8_e8m0),
        TensorSpec(bias_size, dtypes.float32),
        TensorSpec((m_raw,), dtypes.float32),
        TensorSpec((m_raw,), dtypes.int64),
        TensorSpec((e,), dtypes.int64),
        TensorSpec((batch, n), dtypes.float32),
        TensorSpec((1,), dtypes.int64),
    )
    launcher = _GmmFrLauncher(tiling, a_dtype, b_dtype, group_list_type)
    return dsl_compile(launcher.run, *specs)


# ============================================================================
# 4. Torch interface
# ============================================================================


def _validate_inputs(
    x1,
    x2,
    scale,
    pertoken_scale,
    group_list,
    logit,
    row_index,
    bias,
    shared_input,
    shared_input_offset,
    group_list_type,
):
    def _check_tensor(t, name, rank, dtype=None):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not t.is_npu:
            raise ValueError(f"{name} must be on the NPU device")
        if t.dim() != rank:
            raise ValueError(f"{name} must be rank-{rank}, got {tuple(t.shape)}")
        if dtype is not None and t.dtype != dtype:
            raise TypeError(f"{name} must be {dtype}, got {t.dtype}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    _check_tensor(x1, "x1", 2)
    if x1.dtype not in _TORCH_X_DTYPE:
        raise TypeError("x1 must be float8_e4m3fn or float8_e5m2")
    _check_tensor(x2, "x2", 3)
    if x2.dtype not in _TORCH_X_DTYPE:
        raise TypeError("x2 must be float8_e4m3fn or float8_e5m2")
    m_raw, k = x1.shape
    e, k2, n = x2.shape
    if k2 != k:
        raise ValueError(f"x2 K ({k2}) must equal x1 K ({k})")
    if e < 1:
        raise ValueError("x2 expert count must be >= 1")
    if n < 0 or k < 1 or m_raw < 0:
        raise ValueError("invalid x1/x2 extents")
    for other in (x2, scale, pertoken_scale, group_list, logit, row_index):
        if other.device != x1.device:
            raise ValueError("all tensors must share one device")

    scale_groups = _ceil_div(k, MX_K_ALIGN)
    _check_tensor(scale, "scale", 4)
    if scale.dtype != torch.uint8 and scale.dtype != torch.float8_e8m0fnu:
        raise TypeError("scale must be uint8/float8_e8m0fnu storage")
    if tuple(scale.shape) != (e, scale_groups, n, 2):
        raise ValueError(
            f"scale must be (e, ceil(k/64), n, 2) = "
            f"{(e, scale_groups, n, 2)}, got {tuple(scale.shape)}"
        )
    _check_tensor(pertoken_scale, "pertoken_scale", 3)
    if (
        pertoken_scale.dtype != torch.uint8
        and pertoken_scale.dtype != torch.float8_e8m0fnu
    ):
        raise TypeError("pertoken_scale must be uint8/float8_e8m0fnu storage")
    if tuple(pertoken_scale.shape) != (m_raw, scale_groups, 2):
        raise ValueError(
            f"pertoken_scale must be (m, ceil(k/64), 2) = "
            f"{(m_raw, scale_groups, 2)}, got {tuple(pertoken_scale.shape)}"
        )
    _check_tensor(group_list, "group_list", 1, torch.int64)
    if group_list.numel() != e:
        raise ValueError(f"group_list must have length e={e}")
    _check_tensor(logit, "logit", 1, torch.float32)
    if logit.numel() != m_raw:
        raise ValueError(f"logit must have length m={m_raw}")
    _check_tensor(row_index, "row_index", 1, torch.int64)
    if row_index.numel() != m_raw:
        raise ValueError(f"row_index must have length m={m_raw}")

    if bias is not None:
        _check_tensor(bias, "bias", 2, torch.bfloat16)
        if tuple(bias.shape) != (e, n):
            raise ValueError(f"bias must be (e, n)={(e, n)}")
    if shared_input is not None:
        _check_tensor(shared_input, "shared_input", 2, torch.bfloat16)
        if shared_input.shape[1] != n:
            raise ValueError("shared_input N must match x2 N")
    if group_list_type not in (GROUPLIST_TYPE_CUMSUM, GROUPLIST_TYPE_COUNT):
        raise ValueError("group_list_type must be 0 (cumsum) or 1 (count)")


def _resolve_group_sizes(group_list, group_list_type, m_raw):
    """Validate group_list semantics and return the per-group sizes.

    Mainline A5 contract: cumsum is non-negative non-decreasing with the
    last value <= m; count entries are non-negative with sum <= m. Rows of
    x1 beyond the group total are ignored (golden-consistent truncation).
    """
    values = [int(v) for v in group_list.tolist()]
    if group_list_type == GROUPLIST_TYPE_CUMSUM:
        prev = 0
        for v in values:
            if v < prev:
                raise ValueError("group_list (cumsum) must be non-decreasing")
            prev = v
        if values and values[-1] > m_raw:
            raise ValueError("group_list (cumsum) last value must be <= m")
        sizes = [b - a for a, b in zip([0] + values[:-1], values)]
    else:
        if any(v < 0 for v in values):
            raise ValueError("group_list (count) must be non-negative")
        if sum(values) > m_raw:
            raise ValueError("group_list (count) total must be <= m")
        sizes = values
    return sizes


@dataclasses.dataclass
class _KernelCtx:
    """Resolved launch facts shared by the output init and the kernel run."""

    x1: torch.Tensor
    x2: torch.Tensor
    scale: torch.Tensor
    pertoken_scale: torch.Tensor
    group_list: torch.Tensor
    logit: torch.Tensor
    row_index: torch.Tensor
    bias: torch.Tensor | None
    shared_input: torch.Tensor | None
    out: torch.Tensor | None
    m_raw: int
    k: int
    n: int
    e: int
    batch: int
    group_sizes: list
    group_list_type: int
    shared_weight: float
    shared_offset: int
    validate: bool
    y: torch.Tensor = None


def _prepare_batch(out, batch_size, validate, row_index, m_raw, n, x1):
    """Resolve the output row count from out/batch_size/row_index."""
    if out is not None:
        if not out.is_npu:
            raise ValueError("out must be on the NPU device")
        if out.device != x1.device:
            raise ValueError("out must share x1's device")
        if out.dim() != 2 or out.shape[1] != n:
            raise ValueError(f"out must be (batch, n={n})")
        if out.dtype != torch.float32:
            raise TypeError("out must be float32")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        return out.shape[0]
    if batch_size is not None:
        batch = int(batch_size)
    elif m_raw > 0 and validate:
        batch = int(row_index.max().item()) + 1
    elif m_raw > 0:
        raise ValueError("batch_size (or out) is required when validate=False")
    else:
        batch = 1
    if batch < 0:
        raise ValueError("batch must be non-negative")
    return batch


def _resolve_shared_slot(shared_input, offset, batch):
    """Check the shared-expert blend window fits inside the output rows."""
    if shared_input is None:
        return
    if not 0 <= offset <= batch:
        raise ValueError("shared_input_offset out of range")
    if offset + shared_input.shape[0] > batch:
        raise ValueError("shared_input_offset + len exceeds batch")


def _check_row_index_range(row_index, batch, validate, m_raw):
    if validate and m_raw > 0:
        if int(row_index.min().item()) < 0 or int(row_index.max().item()) >= batch:
            raise ValueError("row_index values must be within [0, batch)")


def _init_output(ctx: _KernelCtx) -> torch.Tensor:
    """Zero-fill y and blend the shared-expert contribution (stream-ordered
    before the kernel's atomic adds, matching the mainline AIV prologue)."""
    with torch.npu.device(ctx.x1.device):
        if ctx.out is not None:
            y = ctx.out
            y.zero_()
        else:
            y = torch.zeros(
                (ctx.batch, ctx.n), dtype=torch.float32, device=ctx.x1.device
            )
        if ctx.shared_input is not None and ctx.n > 0:
            y[ctx.shared_offset : ctx.shared_offset + ctx.shared_input.shape[0]].add_(
                ctx.shared_input.to(torch.float32), alpha=float(ctx.shared_weight)
            )
    return y


def _degenerate_result(ctx: _KernelCtx):
    """Zero-work fast path: no routed rows or zero N leaves y as initialized."""
    if ctx.n > 0 and sum(ctx.group_sizes) > 0:
        return None
    return ctx.y


def _run_kernel(ctx: _KernelCtx) -> torch.Tensor:
    """Tiling, AOT compile-cache lookup and the MIX kernel launch."""
    with torch.npu.device(ctx.x1.device):
        has_bias = ctx.bias is not None
        x2_flat = ctx.x2.reshape(ctx.e * ctx.k, ctx.n)
        scale_b_flat = ctx.scale.reshape(ctx.e * ctx.scale.shape[1], ctx.n, 2)
        bias_flat = ctx.bias.reshape(-1).to(torch.float32) if has_bias else None

        stream = torch.npu.current_stream()
        aic_num = int(get_platform_info(stream=stream).cube_core_num)
        tiling = GmmFrTiling(
            ctx.m_raw, ctx.k, ctx.n, ctx.e, ctx.group_sizes, has_bias, aic_num
        )

        compiled = _compile_gmm_fr(
            ctx.m_raw,
            ctx.k,
            ctx.n,
            ctx.e,
            ctx.batch,
            _TORCH_X_TO_DSL[ctx.x1.dtype].name,
            _TORCH_X_TO_DSL[ctx.x2.dtype].name,
            has_bias,
            ctx.group_list_type,
            tiling.base_m,
            tiling.base_n,
            tiling.base_k,
            tiling.k_l1,
            tiling.full_load,
            tiling.used_core_num,
        )
        n_tiles = _ceil_div(ctx.n, tiling.base_n)
        total_tiles = sum(
            _ceil_div(m_g, tiling.base_m) * n_tiles for m_g in ctx.group_sizes
        )
        params = torch.tensor([total_tiles], dtype=torch.int64, device=ctx.x1.device)
        compiled(
            ctx.x1,
            x2_flat,
            ctx.pertoken_scale.view(torch.float8_e8m0fnu),
            scale_b_flat.view(torch.float8_e8m0fnu),
            bias_flat
            if has_bias
            else torch.zeros((1,), dtype=torch.float32, device=ctx.x1.device),
            ctx.logit,
            ctx.row_index,
            ctx.group_list,
            ctx.y,
            params,
        )
    return ctx.y


def grouped_matmul_finalize_routing(
    x1,
    x2,
    scale,
    pertoken_scale,
    group_list,
    logit,
    row_index,
    bias=None,
    shared_input=None,
    *,
    shared_input_weight=1.0,
    shared_input_offset=0,
    group_list_type=GROUPLIST_TYPE_COUNT,
    out=None,
    batch_size=None,
    validate=True,
):
    """MX GroupedMatmul fused with MoeFinalizeRouting (Ascend 950PR/DT),
    aligned with the mainline A5 ``aclnnGroupedMatmulFinalizeRoutingV3``:

    ``y[rowIndex[i], :] += logit[i] * (dequant(x1)[i, :] @ dequant(x2)[g(i)])``
    ``y[off:off+len, :] = sharedInputWeight * shared_input`` (rest zeroed)

    Args:
      x1: ``(m, k)`` fp8 e4m3fn/e5m2 activations, contiguous.
      x2: ``(e, k, n)`` per-expert fp8 weights, non-transposed, contiguous.
      scale: ``(e, ceil(k/64), n, 2)`` E8M0 paired weight scales
        (uint8 or float8_e8m0fnu storage).
      pertoken_scale: ``(m, ceil(k/64), 2)`` E8M0 paired activation scales.
      group_list: ``(e,)`` int64 group extents; count mode lists row counts,
        cumsum mode lists non-decreasing prefix sums (total <= m).
      logit: ``(m,)`` fp32 routing weights.
      row_index: ``(m,)`` int64 scatter-add destination rows within [0, batch).
      bias: optional ``(e, n)`` bf16, folded into the init MMAD (BT path).
      shared_input: optional ``(bsdp, n)`` bf16 shared-expert output blended
        at ``shared_input_offset`` scaled by ``shared_input_weight``.
      group_list_type: 0 (cumsum) or 1 (count, default).
      out: optional ``(batch, n)`` fp32 output rewritten in place.
      batch_size: output row count when ``out`` is not given.
      validate: set False to skip the row_index range check (two device
        syncs); ``batch_size`` or ``out`` is then required.

    Returns ``y``: ``(batch, n)`` fp32 with every routed contribution
    atomically accumulated (non-deterministic order, like the mainline).
    """
    _validate_inputs(
        x1,
        x2,
        scale,
        pertoken_scale,
        group_list,
        logit,
        row_index,
        bias,
        shared_input,
        shared_input_offset,
        group_list_type,
    )
    m_raw, k = x1.shape
    e, _, n = x2.shape

    group_sizes = _resolve_group_sizes(group_list, group_list_type, m_raw)
    batch = _prepare_batch(out, batch_size, validate, row_index, m_raw, n, x1)
    _resolve_shared_slot(shared_input, shared_input_offset, batch)
    _check_row_index_range(row_index, batch, validate, m_raw)

    ctx = _KernelCtx(
        x1=x1,
        x2=x2,
        scale=scale,
        pertoken_scale=pertoken_scale,
        group_list=group_list,
        logit=logit,
        row_index=row_index,
        bias=bias,
        shared_input=shared_input,
        out=out,
        m_raw=m_raw,
        k=k,
        n=n,
        e=e,
        batch=batch,
        group_sizes=group_sizes,
        group_list_type=group_list_type,
        shared_weight=shared_input_weight,
        shared_offset=shared_input_offset,
        validate=validate,
    )
    ctx.y = _init_output(ctx)
    degenerate = _degenerate_result(ctx)
    if degenerate is not None:
        return degenerate
    return _run_kernel(ctx)
