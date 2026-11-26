# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Native HIF8 (quant_mode=0) MLA FlashAttention operator.

Architecture (ported from the verified MXFP8 sample, dense first phase):

- AIC: ``S = Q @ K^T`` with HIF8 Q on L0A and HIF8 K on L0B, FP32 L0C
  laid out ``(M, S2)`` so the KV axis is the vector lane; the per-tensor
  ``softmax_scale * descale_q * descale_k`` factor folds into the FP32
  softmax pass as one compile-time constant.
- AIV: FP32 online softmax with one query row per Reduce: each row owns
  ``S2`` consecutive KV lanes (4 x 64), so its max/sum are one
  ``vreduce_*`` and the four FP8/HIF8 quarters pack straight into a
  ``(M, S2)`` probability tile (one DMA per block).
- AIC: ``O += P @ V`` with HIF8 P/V and FP32 L0C.
- AIV: ``acc = acc * corr + PV``, finalize ``O = acc * descale_v / l`` in
  BF16; the online sum ``l`` stays FP32 so the division cancels the
  p_scale factor exactly like the AscendC reference.

Tile 128x256x128, two AIVs of 64 Q rows per cube core, SKEW_LAG=2
pipeline, dense BNSD KV (no paged cache), causal or no mask.
"""

__all__ = ["flash_mla_fullquant_hif8", "flash_mla_fullquant_hif8_metadata"]
# HIF8 operator (quant_mode=0); the FP8 E4M3 sibling lives in
# flash_mla_fullquant_fp8.py.

import math
import os
import struct

import torch
from cannbotdsl import (
    MemLoc,
    RegLayout,
    RingBuffer,
    Tensor,
    const_expr,
    dtypes,
    from_tensor,
    get_block_idx,
    get_subblock_id,
    jit,
    kernel,
    make_layout,
    matmul,
    range_constexpr,
    vf,
)
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.host import host
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.reg import (
    update_mask,
    vadd,
    vadds,
    varange,
    vbitwise_and,
    vcast,
    vdiv,
    vexp_sub,
    vgather_reg,
    vles,
    vload,
    vload_deinterleave,
    vlog,
    vmadd,
    vmax,
    vmem_bar,
    vmul,
    vmuls,
    vreduce_max,
    vreduce_sum,
    vreinterpret,
    vreinterpret_lanes,
    vselect,
    vshl,
    vshr,
    vstore,
    vstore_first,
    vstore_strided,
)
from cannbotdsl.reg import (
    vbitwise_or as vor,
)
from cannbotdsl.reg import (
    vdups as vdup_scalar,
)
from cannbotdsl.reg import (
    vle as vcmp_le,
)
from cannbotdsl.reg import (
    vload_broadcast as vload_brc,
)
from cannbotdsl.tensor import (
    ceil_div,
    make_tiler,
    tile_slice,
)

from .flash_mla_fullquant_checker import validate_scales

M_BASE_SIZE = 64  # AscendC tile: single-L0C PV needs M=64
S2_BASE_SIZE = 256
D = 576  # MLA QK dim (512 nope + 64 rope)
D_V = 512  # MLA V dim

# QK chunking for D=576: 4x128 + 1x64 (64 is 16-aligned, NZ-fractal safe)
QK_CHUNKS = (128, 128, 128, 128, 64)
QK_OFFSETS = (0, 128, 256, 384, 512)
SKEW_LAG = 1


def _reinterpret_view(value, dtype, shape, *, offset=0):
    """0.8 replacement for the removed Tensor.reinterpret method.

    ``offset`` keeps the old element semantics; from_tensor expects bytes.
    L1/UB NZ views retain their physical pitch.
    """
    physical = None
    if value.memloc in (MemLoc.L1, MemLoc.UB) and value.data_format in ("nz", "zn"):
        physical = value.physical_layout
    return from_tensor(
        value,
        dtype=dtype,
        layout=make_layout(shape),
        offset=offset * (dtype.width // 8),
        physical_layout=physical,
    )


FP32_VECTOR_LANES = 64
NEG_MAX = -3.4028234663852886e38

MASK_NONE = 0
MASK_CAUSAL = 1
MASK_KV_TAIL = 2

PAGE_SIZE = 128  # Paged Attention page size (official MLA contract)

DEPTHS = {
    # V reuses the K slot, so a slot lives until bmm2 consumes it: keep
    # three KV slots so MTE2 can still prefetch one block ahead. Q is
    # loaded once per task, so one slot is enough (frees the L1 budget).
    "q_l1": 1,
    "kv_l1": 3,
    "p_l1": 2,
    "l0a": 2,
    "l0b": 2,
    "l0c_mm1": 2,
    "l0c_mm2": 1,
    "s_ub": 1,
    "pv_ub": 1,
    "p_ub": 1,
    "output_bf16_ub": 1,
}


def _query_row_tile(gm, layout, batch_idx, head_idx, query_token_start, rows, width):
    alignment = (16, 32) if isinstance(rows, int) and width == D else (1, 1)
    row_end = query_token_start + rows
    if layout != "TND":
        base = gm[batch_idx, head_idx, query_token_start:row_end, None]
        return tile_slice(base, make_tiler((rows, width), alignment=alignment), (0, 0))

    # Kernel-side TND storage is head-major (N, T, D) like the MXFP8 sample.
    base = gm[head_idx, query_token_start:row_end, None]
    return tile_slice(base, make_tiler((rows, width), alignment=alignment), (0, 0))


def _query_row_tile_cols(
    gm, layout, batch_idx, head_idx, query_token_start, rows, col0, width
):
    """Column-window GM tile for the multi-pass BF16 output writeback."""
    row_end = query_token_start + rows
    col_end = col0 + width
    if layout != "TND":
        base = gm[
            batch_idx,
            head_idx,
            query_token_start:row_end,
            col0:col_end,
        ]
    else:
        base = gm[
            head_idx,
            query_token_start:row_end,
            col0:col_end,
        ]
    return tile_slice(base, make_tiler((rows, width), alignment=(1, 1)), (0, 0))


def _lse_row_tile(gm, layout, batch_idx, head_idx, query_token_start, rows):
    """One flat FP32 row window for the LSE writeback."""
    row_end = query_token_start + rows
    if layout != "TND":
        base = gm[batch_idx, head_idx, query_token_start:row_end]
    else:
        base = gm[head_idx, query_token_start:row_end]
    return tile_slice(base, make_tiler((rows,), alignment=(1,)), (0,))


def _dense_kv_tile(gm, layout, batch_idx, kv_head, token_start, rows):
    """Dense KV rows for one 256-wide tile (BNSD/TND native layouts)."""
    row_end = token_start + rows
    if layout != "TND":
        base = gm[batch_idx, kv_head, token_start:row_end, None]
    else:
        base = gm[kv_head, token_start:row_end, None]
    return tile_slice(base, make_tiler((rows, D), alignment=(16, 32)), (0, 0))


def _kv_rows(gm, in_layout, batch_idx, kv_head, token_start, rows, kv_s_pad, kv_n):
    """Per-row contiguous GM views for one strided KV tile (BSND/TND input).

    The wrapper passes Q/K/V as rank-2 flats (tokens*heads, D); one strided
    tile is then a regular burst of D-byte rows. The static tuple-range loop
    unrolls at trace time; the row tuple feeds one multi-source mem_copy
    (single strided ND2NZ DMA, no permute copy).
    """
    out = [None] * rows
    if const_expr(in_layout == "BSND"):
        base = batch_idx * kv_s_pad * kv_n + kv_head
        for i in tuple(range(rows)):
            out[i] = gm[base + (token_start + i) * kv_n, 0:D]
    else:  # TND token-major: token_start already packs batches
        for i in tuple(range(rows)):
            out[i] = gm[(token_start + i) * kv_n + kv_head, 0:D]
    return tuple(out)


@jit
def _paged_kv_copy(
    dst, gm_pool, gm_bt, batch_idx, kv_block_idx, kv_len, s2_base, page=PAGE_SIZE
):
    """Load one S2 tile of paged KV, one stripe per page.

    ``gm_pool`` is the paged cache viewed as ``(pages, page, 1, D)`` and
    ``gm_bt[batch, logical_page]`` holds the physical page id, so the
    kernel resolves pages itself and no host-side dense gather is needed.
    A page lands in its own 128-row window of the tile slot; the window is
    expressed on ``dst`` because an NZ row window is only addressable from
    the destination side.

    Each page is staged as 16-row fractal stripes rather than one 128-row
    copy: measured on the FP8 sibling at 64K (3 interleaved rounds,
    medians) the striped form is 44.9us vs 48.3us main-kernel and much
    steadier, while costing only 1.0us at 32K.  Past ``cache_seqlens`` the
    last block clamps to the last real page; those rows are kv-tail
    masked by the caller, so only validity matters.
    """
    halves = s2_base // page
    base_col = kv_block_idx * halves
    last_col = ceil_div(kv_len, page) - 1
    for half in tuple(range(halves)):
        col = min(base_col + half, last_col)
        pg = dtypes.int64(gm_bt[batch_idx, col])
        page_view = gm_pool[pg, None, 0, None].view(page, D)
        for chunk in tuple(range(0, page, 16)):
            src = tile_slice(
                page_view[chunk:, 0:],
                make_tiler((16, D), alignment=(16, 32)),
                (0, 0),
            )
            base = tile_slice(dst, (16, D), (half * (page // 16) + chunk // 16, 0))
            mem_copy(base, src)


def _q_rows(gm, in_layout, batch_idx, head_idx, query_token_start, rows, q_s_pad, q_n):
    """Per-row contiguous GM views for one strided Q tile (BSND/TND input)."""
    out = [None] * rows
    if const_expr(in_layout == "BSND"):
        base = batch_idx * q_s_pad * q_n + head_idx
        for i in tuple(range(rows)):
            out[i] = gm[base + (query_token_start + i) * q_n, 0:D]
    else:  # TND token-major
        for i in tuple(range(rows)):
            out[i] = gm[(query_token_start + i) * q_n + head_idx, 0:D]
    return tuple(out)


class Matmul:
    """HIF8 Cube side: K-only streaming (V reuses K's slot), MM1 ``K@Q^T``, MM2 ``P^T@V``."""

    def __init__(
        self,
        m_base_size,
        s2_base_size,
        d,
        depths,
        kv_geom=(0, 0),
        storage_type=dtypes.hifloat8,
    ):
        self.m_base_size, self.s2_base_size, self.d = m_base_size, s2_base_size, d
        self.kv_s_pad, self.kv_n = kv_geom
        self.storage_type = storage_type
        self.q_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.int8,
            make_layout((m_base_size, d)),
            depth=depths["q_l1"],
        )
        self.kv_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.int8,
            make_layout((s2_base_size, d)),
            depth=depths["kv_l1"],
        )
        # MLA D=576: Q resident 36KB, KV double-buffered 2x147KB
        # MLA D=576: total L1 for Q = 128*576 = 73728, KV = 256*576 = 147456
        self.l0a = RingBuffer(
            MemLoc.L0A,
            storage_type,
            make_layout((s2_base_size, 128)),
            depth=depths["l0a"],
        )
        self.l0b = RingBuffer(
            MemLoc.L0B,
            storage_type,
            make_layout((s2_base_size, 128)),
            depth=depths["l0b"],
        )
        self.l0c_mm1 = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((m_base_size, s2_base_size)),
            depth=depths["l0c_mm1"],
        )
        # One (M, D_V) PV accumulator: the four N bands land in its column
        # windows, so one fixpipe moves the whole tile to UB per KV block.
        self.l0c_mm2 = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((m_base_size, D_V)),
            depth=1,
        )
        self.score_actual = (m_base_size, s2_base_size)
        self.output_actual = (m_base_size, 128)

    @jit
    def load_q(self, q_tile):
        q_slot = self.q_l1.next_write()
        mem_copy(q_slot, q_tile)
        return q_slot

    @jit
    def load_k(
        self,
        gm_key,
        layout,
        in_layout,
        batch_idx,
        kv_head,
        kv_block_idx,
        kv_token_base=0,
        gm_bt=None,
        kv_len=0,
    ):
        slot = self.kv_l1.next_write()
        if const_expr(in_layout == "PAGED"):
            # Page-table addressing in the kernel: one DMA per page.
            _paged_kv_copy(
                slot, gm_key, gm_bt, batch_idx, kv_block_idx, kv_len, self.s2_base_size
            )
        else:
            token_start = kv_token_base + kv_block_idx * self.s2_base_size
            if const_expr(in_layout == "BNSD"):
                src = _dense_kv_tile(
                    gm_key,
                    layout,
                    batch_idx,
                    kv_head,
                    token_start,
                    self.s2_base_size,
                )
            else:
                src = _kv_rows(
                    gm_key,
                    in_layout,
                    batch_idx,
                    kv_head,
                    token_start,
                    self.s2_base_size,
                    self.kv_s_pad,
                    self.kv_n,
                )
            mem_copy(slot, src)
        # The slot is handed to bmm1 and kept for bmm2 (V == K_nope);
        # only the consume side advances the read cursor.
        return slot

    @jit
    def compute_qk(self, score_ub, q_slot, kv_l1_slot):
        """Chunked QK for D=576: 5 MMADs accumulated into one (M, S2) L0C.

        Q rides L0A and K L0B so the product is laid out with the KV axis
        trailing: each AIV's half of the fixpipe is then a (m_per, S2)
        tile whose 64-element lanes are consecutive KV scores.  Each chunk
        produces a FRESH L0A/L0B slot (cursor cycles over the channel
        depth; the implicit matmul read releases the slot) and copies its
        128/64-wide column window from L1 with matching view shapes. Only
        the final chunk pairs with the fixpipe (unit_flag=3).
        """
        score_l0c = self.l0c_mm1.next_write()
        for chunk_idx in tuple(range(len(QK_CHUNKS))):
            chunk_w = QK_CHUNKS[chunk_idx]
            chunk_off = QK_OFFSETS[chunk_idx]
            query_l0a = _reinterpret_view(
                self.l0a.next_write(), self.storage_type, (self.m_base_size, chunk_w)
            )
            key_l0b = _reinterpret_view(
                self.l0b.next_write(), self.storage_type, (self.s2_base_size, chunk_w)
            )
            mem_copy(
                query_l0a,
                _reinterpret_view(
                    q_slot,
                    self.storage_type,
                    (self.m_base_size, chunk_w),
                    offset=chunk_off * self.m_base_size,
                ),
            )
            mem_copy(
                key_l0b,
                _reinterpret_view(
                    kv_l1_slot,
                    self.storage_type,
                    (self.s2_base_size, chunk_w),
                    offset=chunk_off * self.s2_base_size,
                ),
            )
            is_first = chunk_idx == 0
            is_last = chunk_idx == len(QK_CHUNKS) - 1
            if is_last:
                matmul(score_l0c, query_l0a, key_l0b, init=is_first, unit_flag=3)
            else:
                matmul(score_l0c, query_l0a, key_l0b, init=is_first)
        mem_copy(
            score_ub.next_write(),
            score_l0c.partition(dim=0),
            copy_options(unit_flag=3),
        )

    @jit
    def compute_pv(self, output_ub, p_source, v_source):
        """One (M, D_V) PV accumulation per KV block.

        P stays in L0A across the four N bands; each band writes its own
        column window of the single L0C accumulator, then one fixpipe
        moves the whole (M, D_V) tile to UB for the AIV accumulate.
        """
        probability_l0a = from_tensor(
            self.l0a.next_write(),
            self.storage_type,
            make_layout((self.m_base_size, self.s2_base_size)),
            data_format="nz",
        )
        mem_copy(probability_l0a, p_source)
        output_l0c = self.l0c_mm2.next_write()
        for v_band in tuple(range(D_V // 128)):
            value_l0b = from_tensor(
                self.l0b.next_write(),
                self.storage_type,
                make_layout((128, self.s2_base_size)),
                data_format="nz",
            )
            mem_copy(
                value_l0b,
                _reinterpret_view(
                    v_source,
                    self.storage_type,
                    (self.s2_base_size, 128),
                    offset=v_band * 128 * self.s2_base_size,
                ).permute(1, 0),
            )
            matmul(
                tile_slice(
                    output_l0c,
                    make_tiler((self.m_base_size, 128), alignment=(16, 16)),
                    (0, v_band),
                ),
                probability_l0a,
                value_l0b,
                init=True,
                unit_flag=3,
            )
        mem_copy(
            output_ub.next_write(),
            output_l0c.partition(dim=0),
            copy_options(unit_flag=3),
        )

    @jit
    def pv_consume_sources(self, p_l1):
        """Consume this tile's P and V from L1 exactly once."""
        return p_l1.next_read(), self.kv_l1.next_read()


class Vector:
    """HIF8 vector side: online softmax, HIF8 P pack, output accumulation."""

    def __init__(
        self,
        m_base_size_per_aiv,
        s2_base_size,
        d,
        depths,
        qk_scale,
        v_descale,
        p_scale,
        d_v=D_V,
        use_q_descale=False,
        k_scale_only=1.0,
        decode_only=False,
        storage_type=dtypes.hifloat8,
    ):
        self.m_base_size_per_aiv = m_base_size_per_aiv
        self.s2_base_size, self.d = s2_base_size, d
        self.d_v = d_v
        self.qk_scale = float(qk_scale)
        # MLA: if True, apply per-row Q descale from q_descale_ub
        self.use_q_descale = use_q_descale
        self.k_scale_only = float(k_scale_only)  # k_descale × softmax_scale
        self.decode_only = decode_only
        self.storage_type = storage_type
        self.v_descale = float(v_descale)
        self.log_p_scale = math.log(float(p_scale)) if p_scale else 0.0

        # Sentinel row_m of a fully-masked row that still passes through a
        # softmax block: f32(f32(NEG_MAX * qk_scale) - log(p_scale)), same
        # op sequence as the softmax Pass A (bit-exact). Rows skipped by
        # kv_block_count keep the init value NEG_MAX. finalize_output zeroes
        # any row whose max state is at or below this threshold, mirroring
        # AscendC's RowInvalidUpdateVF (masked softmax makes exp(0) = 1 per
        # key, which would otherwise emit mean(V) garbage instead of 0).
        def _f32(x):
            return struct.unpack("<f", struct.pack("<f", float(x)))[0]

        # Clamp the scaled sentinel into the FP32 range: qk_scale > 1 would
        # otherwise overflow struct.pack and abort the build.
        scaled_sentinel = NEG_MAX
        if math.isfinite(self.qk_scale) and self.qk_scale > 0.0:
            scaled_sentinel = max(NEG_MAX, NEG_MAX * self.qk_scale)
        _sentinel = _f32(_f32(scaled_sentinel) + _f32(-self.log_p_scale))
        self.invalid_row_threshold = max(_sentinel, NEG_MAX)
        # Softmax packs a whole query row's S2 probabilities (4 x 64 lanes
        # cast into the four byte quarters) into one contiguous row, so a
        # (m_per, S2) tile reaches L1 with a single DMA per block.
        # The staging tile carries the SHARED L1 slot's physical pitch
        # (m1 = M/16 blocks), so the UB->L1 copy is a uniform strided
        # move instead of two different n1 pitches.
        p_m1 = (2 * m_base_size_per_aiv) // 16
        self.p_ub = RingBuffer(
            MemLoc.UB,
            storage_type,
            make_layout((m_base_size_per_aiv, s2_base_size)),
            depth=depths["p_ub"],
            data_format="nz",
            physical_layout=make_layout(
                (s2_base_size // 32, p_m1, 16, 32),
                stride=(p_m1 * 512, 512, 32, 1),
            ),
        )
        # NZ pitch of the staging tile, read off the derived physical layout
        # (m0 = 16 rows, n0 = 32 bytes): a query row's 256 probabilities are
        # eight 32-byte blocks one n1 stride apart.
        p_strides = self.p_ub.physical_layout.stride
        self.p_nz_n1_stride = p_strides[0]
        self.p_nz_m1_stride = p_strides[1]
        self.p_nz_m0_stride = p_strides[2]
        self.p_nz_m0 = self.p_nz_m1_stride // self.p_nz_m0_stride
        self.p_nz_n0 = self.p_nz_m0_stride // p_strides[3]
        self.p_nz_block_stride = self.p_nz_n1_stride // self.p_nz_n0
        self.output_bf16_ub = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((m_base_size_per_aiv, 64)),
            depth=depths["output_bf16_ub"],
        )
        self.lse_stage = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((m_base_size_per_aiv,))
        )
        self.acc = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((m_base_size_per_aiv, D_V))
        )
        self.row_m = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((m_base_size_per_aiv, 1))
        )
        self.row_l = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((m_base_size_per_aiv, 1))
        )
        # MLA per-token-head Q descale (FP32, one per row)
        self.q_descale_ub = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((m_base_size_per_aiv,))
        )
        # per-row query token index (within batch) for causal limits
        self.q_token_ub = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((m_base_size_per_aiv,))
        )
        self.corr_buffers = [
            Buffer(MemLoc.UB, dtypes.float32, make_layout((m_base_size_per_aiv, 1)))
            for _ in range(SKEW_LAG + 1)
        ]

    @jit
    def initialize(self):
        with vf(mode="simd"):
            row_mask, _ = update_mask(self.m_base_size_per_aiv, elem_bits=32)
            vstore(
                self.row_m,
                0,
                vdup_scalar(NEG_MAX, dtypes.float32, mask=row_mask),
                row_mask,
            )
            vstore(
                self.row_l,
                0,
                vdup_scalar(0.0, dtypes.float32, mask=row_mask),
                row_mask,
            )
            mf, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
            zero = vdup_scalar(0.0, dtypes.float32, mask=mf)
            for row in range(self.m_base_size_per_aiv):
                for lane in tuple(range(0, self.d_v, FP32_VECTOR_LANES)):
                    vstore(self.acc, row * self.d_v + lane, zero, mf)

    @jit
    def softmax(
        self, score_ub, first_query_visible_limit, pipeline_slot, kv_block_mask_type
    ):
        """Per-row online softmax with the KV axis in the vector lanes.

        ``score_ub`` is one (m_per, S2) FP32 tile, so query row q owns S2
        consecutive KV scores (4 x 64 lanes) and its max/sum are one
        ``vreduce_*`` instead of a sweep over the KV rows:

        ``m_new = max(m_old, max(S*qk_scale) - log(p_scale))``;
        ``P = cast_hif8(exp(S*qk_scale - m_new))`` packed in KV order;
        ``l_new = l_old*corr + sum(exp)`` (FP32, pre-cast).
        """
        p_slot = self.p_ub.next_write()
        corr_buffer = self.corr_buffers[pipeline_slot]
        with vf(mode="simd"):
            fp32_mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
            scalar_mask, _ = update_mask(1, elem_bits=32)
            fp8_mask, _ = update_mask(4 * FP32_VECTOR_LANES, elem_bits=8)
            # Byte-gather index for the four cast quarters: byte 4i+k reads
            # source byte 16*(i%16) + i//16 + 4k (one byte per 32-bit lane).
            gather_lane = varange(0, dtypes.uint32)
            gather_base = vadd(
                vshl(
                    vbitwise_and(
                        gather_lane,
                        vdup_scalar(15, dtypes.uint32, mask=fp32_mask),
                        mask=fp32_mask,
                    ),
                    4,
                    mask=fp32_mask,
                ),
                vshr(gather_lane, 4, mask=fp32_mask),
                mask=fp32_mask,
            )
            nz_gather_index = vreinterpret_lanes(
                vadds(
                    vmuls(gather_base, 0x01010101, mask=fp32_mask),
                    0x0C080400,
                    mask=fp32_mask,
                ),
                dtypes.uint8,
            )
            # Masked-score sentinel is -inf (not NEG_MAX): a fully-masked
            # row then keeps score == -inf while row_m stays at its finite
            # NEG_MAX init, so exp(score - m) == 0 for every masked key and
            # l stays exactly 0. The finalize is_zero/invalid-row checks
            # then emit AscendC semantics (out = 0, LSE = +inf) instead of
            # the exp(0) = 1-per-key mean(V) garbage a finite sentinel
            # produces (score and max both equal NEG_MAX * descale).
            neg_one = vdup_scalar(-1.0, dtypes.float32, mask=fp32_mask)
            zero_f32 = vdup_scalar(0.0, dtypes.float32, mask=fp32_mask)
            negative_max = vdiv(neg_one, zero_f32, mask=fp32_mask)
            # Absolute KV index of every lane, one ramp per 64-lane chunk;
            # loop-invariant across query rows.
            kv_indices = []
            for chunk in tuple(range(self.s2_base_size // FP32_VECTOR_LANES)):
                kv_indices.append(
                    vcast(
                        varange(chunk * FP32_VECTOR_LANES, dtypes.int32),
                        dtypes.float32,
                        mask=fp32_mask,
                    )
                )

            # Pass A: raw loads, mask + write-back (the -inf sentinel is
            # scale-invariant), then the row max over the four chunks.
            for query_row_index in range(self.m_base_size_per_aiv):
                row_base = query_row_index * self.s2_base_size
                if const_expr(kv_block_mask_type != MASK_NONE):
                    if const_expr(kv_block_mask_type == MASK_CAUSAL):
                        # Head-packed rows: the per-lane query token index
                        # comes from GM; the scalar carries
                        # (causal_offset - block*s2).
                        visible_limit = vadds(
                            vload_brc(self.q_token_ub, query_row_index),
                            first_query_visible_limit,
                            mask=fp32_mask,
                        )
                    else:
                        visible_limit = vdup_scalar(
                            first_query_visible_limit,
                            dtypes.float32,
                            mask=fp32_mask,
                        )
                row_max = None
                for chunk in tuple(range(len(kv_indices))):
                    raw_score = vload(score_ub, row_base + chunk * FP32_VECTOR_LANES)
                    if const_expr(kv_block_mask_type != MASK_NONE):
                        raw_score = vselect(
                            raw_score,
                            negative_max,
                            cond_mask=vcmp_le(
                                kv_indices[chunk],
                                visible_limit,
                                mask=fp32_mask,
                            ),
                        )
                        vstore(
                            score_ub,
                            row_base + chunk * FP32_VECTOR_LANES,
                            raw_score,
                            fp32_mask,
                        )
                    if chunk == 0:
                        row_max = raw_score
                    else:
                        row_max = vmax(row_max, raw_score, mask=fp32_mask)
                block_max = vreduce_max(row_max, mask=fp32_mask)
                if const_expr(self.use_q_descale):
                    # MLA: block_max x per-row Q descale x
                    # (k_descale x softmax_scale)
                    block_max = vmul(
                        block_max,
                        vmul(
                            vload_brc(self.q_descale_ub, query_row_index),
                            vdup_scalar(
                                self.k_scale_only, dtypes.float32, mask=fp32_mask
                            ),
                            mask=fp32_mask,
                        ),
                        mask=scalar_mask,
                    )
                else:
                    block_max = vmuls(block_max, self.qk_scale, mask=scalar_mask)
                block_max = vadds(block_max, -self.log_p_scale, mask=scalar_mask)
                old_block_max = vload_brc(self.row_m, query_row_index)
                new_block_max = vmax(old_block_max, block_max, mask=scalar_mask)
                current_block_corr = vexp_sub(
                    old_block_max, new_block_max, mask=scalar_mask
                )
                vstore_first(self.row_m, query_row_index, new_block_max)
                vstore_first(corr_buffer, query_row_index, current_block_corr)

            # Pass A wrote the masked scores and the row state; Pass B
            # reloads both.
            vmem_bar("vst_vld")

            # Pass B: scale, exp, row sum, then pack the four FP8/HIF8
            # quarters of the row straight into KV order.
            for query_row_index in range(self.m_base_size_per_aiv):
                row_base = query_row_index * self.s2_base_size
                new_max_broadcast = vload_brc(self.row_m, query_row_index)
                if const_expr(self.use_q_descale):
                    row_factor = vmul(
                        vload_brc(self.q_descale_ub, query_row_index),
                        vdup_scalar(self.k_scale_only, dtypes.float32, mask=fp32_mask),
                        mask=fp32_mask,
                    )
                exp_scores = []
                for chunk in tuple(range(len(kv_indices))):
                    score = vload(score_ub, row_base + chunk * FP32_VECTOR_LANES)
                    if const_expr(self.use_q_descale):
                        score = vmul(score, row_factor, mask=fp32_mask)
                    elif const_expr(self.qk_scale != 1.0):
                        score = vmuls(score, self.qk_scale, mask=fp32_mask)
                    exp_scores.append(
                        vexp_sub(score, new_max_broadcast, mask=fp32_mask)
                    )
                partial_sum = exp_scores[0]
                for chunk in tuple(range(1, len(exp_scores))):
                    partial_sum = vadd(partial_sum, exp_scores[chunk], mask=fp32_mask)
                vstore_first(
                    self.row_l,
                    query_row_index,
                    vmadd(
                        vload_brc(self.row_l, query_row_index),
                        vload_brc(corr_buffer, query_row_index),
                        vreduce_sum(partial_sum, mask=fp32_mask),
                        mask=scalar_mask,
                    ),
                )
                p_row_offset = (
                    query_row_index // self.p_nz_m0
                ) * self.p_nz_m1_stride + (
                    query_row_index % self.p_nz_m0
                ) * self.p_nz_m0_stride
                # The four cast quarters land one byte per 32-bit lane, so
                # one register gather (byte 4i+k <- 16*(i%16) + i//16 + 4k)
                # restores KV order before the NZ block store; without it
                # only every fourth byte of a block would carry data.
                for quad in tuple(range(len(exp_scores) // 4)):
                    packed_quad = vor(
                        vor(
                            vreinterpret(
                                vcast(
                                    exp_scores[4 * quad],
                                    self.storage_type,
                                    mask=fp32_mask,
                                    saturate=True,
                                    rounding="rn",
                                    reg_layout=RegLayout.ZERO,
                                ),
                                dtypes.uint8,
                            ),
                            vreinterpret(
                                vcast(
                                    exp_scores[4 * quad + 1],
                                    self.storage_type,
                                    mask=fp32_mask,
                                    saturate=True,
                                    rounding="rn",
                                    reg_layout=RegLayout.ONE,
                                ),
                                dtypes.uint8,
                            ),
                            mask=fp8_mask,
                        ),
                        vor(
                            vreinterpret(
                                vcast(
                                    exp_scores[4 * quad + 2],
                                    self.storage_type,
                                    mask=fp32_mask,
                                    saturate=True,
                                    rounding="rn",
                                    reg_layout=RegLayout.TWO,
                                ),
                                dtypes.uint8,
                            ),
                            vreinterpret(
                                vcast(
                                    exp_scores[4 * quad + 3],
                                    self.storage_type,
                                    mask=fp32_mask,
                                    saturate=True,
                                    rounding="rn",
                                    reg_layout=RegLayout.THREE,
                                ),
                                dtypes.uint8,
                            ),
                            mask=fp8_mask,
                        ),
                        mask=fp8_mask,
                    )
                    vstore_strided(
                        p_slot,
                        p_row_offset + quad * 8 * self.p_nz_n1_stride,
                        vgather_reg(packed_quad, nz_gather_index),
                        fp8_mask,
                        block_stride=self.p_nz_block_stride,
                        repeat_stride=0,
                    )

    @jit
    def store_p(self, p_l1, vector_subblock_index):
        # P is already in KV order, so this AIV's m_per rows are one
        # contiguous (m_per, S2) window of the shared (M, S2) L1 tile.
        mem_copy(
            p_l1.next_write().partition(dim=0, part_id=vector_subblock_index),
            self.p_ub.next_read(),
        )

    @jit
    def load_q_descale(self, gm_q_descale, row_start, subblock_index):
        """Load this AIV's flat (token,head)-row descale window."""
        row0 = row_start + subblock_index * self.m_base_size_per_aiv
        row_end = row0 + self.m_base_size_per_aiv
        base = gm_q_descale[0, row0:row_end]
        mem_copy(
            self.q_descale_ub,
            tile_slice(
                base,
                make_tiler((self.m_base_size_per_aiv,), alignment=(16,)),
                (0,),
            ),
        )

    @jit
    def load_q_token(self, gm_q_token, row_start, subblock_index):
        """Load this AIV's per-row query token indices (causal limits)."""
        row0 = row_start + subblock_index * self.m_base_size_per_aiv
        row_end = row0 + self.m_base_size_per_aiv
        base = gm_q_token[0, row0:row_end]
        mem_copy(
            self.q_token_ub,
            tile_slice(
                base,
                make_tiler((self.m_base_size_per_aiv,), alignment=(16,)),
                (0,),
            ),
        )

    @jit
    def update_output_acc(self, pv_ub, pipeline_slot):
        corr_buffer = self.corr_buffers[pipeline_slot]
        with vf(mode="simd"):
            # Masks are loop-invariant (d is 64-aligned): build once.
            fp32_mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
            for query_row_index in range(self.m_base_size_per_aiv):
                corr_broadcast = vload_brc(corr_buffer, query_row_index)
                for d_offset in tuple(range(0, self.d_v, FP32_VECTOR_LANES)):
                    acc_offset = query_row_index * self.d_v + d_offset
                    pv_value = vload(pv_ub, query_row_index * self.d_v + d_offset)
                    if const_expr(self.v_descale != 1.0):
                        pv_value = vmuls(pv_value, self.v_descale, mask=fp32_mask)
                    vstore(
                        self.acc,
                        acc_offset,
                        vmadd(
                            vload(self.acc, acc_offset),
                            corr_broadcast,
                            pv_value,
                            mask=fp32_mask,
                        ),
                        fp32_mask,
                    )

    @jit
    def finalize_output(
        self,
        gm_output,
        gm_lse,
        total_rows,
        tile_row_start,
        valid_rows,
        vector_subblock_index,
    ):
        """``O = acc / l`` cast to BF16 into the flat (rows, D_V)
        output; LSE into the flat (1, rows).  The wrapper slices valid
        rows and restores layout_out.
        """
        row_base = tile_row_start + vector_subblock_index * self.m_base_size_per_aiv
        row_end = row_base + self.m_base_size_per_aiv
        for col_pass in tuple(range(self.d_v // 64)):
            output_slot = self.output_bf16_ub.next_write()
            with vf(mode="simd"):
                fp32_mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                bf16_mask, _ = update_mask(2 * FP32_VECTOR_LANES, elem_bits=16)
                for query_row_index in range(self.m_base_size_per_aiv):
                    normalization_denominator = vload_brc(self.row_l, query_row_index)
                    row_max = vload_brc(self.row_m, query_row_index)
                    threshold_vec = vdup_scalar(
                        self.invalid_row_threshold,
                        dtypes.float32,
                        mask=fp32_mask,
                    )
                    invalid_row = vcmp_le(row_max, threshold_vec, mask=fp32_mask)
                    one_vec = vdup_scalar(1.0, dtypes.float32, mask=fp32_mask)
                    zero_vec = vdup_scalar(0.0, dtypes.float32, mask=fp32_mask)
                    normalization_denominator = vselect(
                        one_vec,
                        normalization_denominator,
                        cond_mask=invalid_row,
                    )
                    acc_offset = query_row_index * self.d_v + col_pass * 64
                    even_acc, odd_acc = vload_deinterleave(
                        self.acc, acc_offset, width="b32"
                    )
                    even_acc = vselect(zero_vec, even_acc, cond_mask=invalid_row)
                    odd_acc = vselect(zero_vec, odd_acc, cond_mask=invalid_row)
                    even_acc = vdiv(
                        even_acc,
                        normalization_denominator,
                        mask=fp32_mask,
                    )
                    odd_acc = vdiv(
                        odd_acc,
                        normalization_denominator,
                        mask=fp32_mask,
                    )
                    even_bf16 = vcast(
                        even_acc,
                        dtypes.bfloat16,
                        mask=fp32_mask,
                        reg_layout=RegLayout.ZERO,
                    )
                    odd_bf16 = vcast(
                        odd_acc,
                        dtypes.bfloat16,
                        mask=fp32_mask,
                        reg_layout=RegLayout.ONE,
                    )
                    vstore(
                        output_slot,
                        query_row_index * 64,
                        vor(even_bf16, odd_bf16, mask=bf16_mask),
                        bf16_mask,
                    )
            col0 = col_pass * 64
            col_end = col0 + 64
            base = gm_output[
                row_base:row_end,
                col0:col_end,
            ]
            mem_copy(
                tile_slice(
                    base,
                    make_tiler((self.m_base_size_per_aiv, 64), alignment=(1, 1)),
                    (0, 0),
                ),
                output_slot,
            )
        if const_expr(gm_lse is not None):
            with vf(mode="simd"):
                m64, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                one_vec = vdup_scalar(1.0, dtypes.float32, mask=m64)
                zero_vec = vdup_scalar(0.0, dtypes.float32, mask=m64)
                inf_vec = vdiv(one_vec, zero_vec, mask=m64)
                row_m_vec = vload(self.row_m, 0)
                row_l_vec = vload(self.row_l, 0)
                is_zero = vles(row_l_vec, 0.0, mask=m64)
                safe_l = vselect(one_vec, row_l_vec, cond_mask=is_zero)
                lse_all = vadd(row_m_vec, vlog(safe_l, mask=m64), mask=m64)
                lse_all = vselect(inf_vec, lse_all, cond_mask=is_zero)
                vstore(self.lse_stage, 0, lse_all, m64)
            mem_copy(
                tile_slice(
                    gm_lse[0, row_base:row_end],
                    make_tiler((self.m_base_size_per_aiv,), alignment=(16,)),
                    (0,),
                ),
                self.lse_stage,
            )


@kernel
class FlashAttnHif8Kernel:
    """HIF8 dense GQA pipeline on the MXFP8 schedule."""

    def __init__(
        self,
        m_base_size=M_BASE_SIZE,
        s2_base_size=S2_BASE_SIZE,
        d=D,
        depths=None,
        causal=0,
        layout="BNSD",
        in_layout="BNSD",
        qk_scale=1.0,
        v_descale=1.0,
        p_scale=1.0,
        q_geom=(0, 0),
        kv_geom=(0, 0),
        d_v=D_V,
        use_q_descale=False,
        k_scale_only=1.0,
        decode_only=False,
        fd_mode=False,
        storage_type=dtypes.hifloat8,
    ):
        self.m_base_size, self.s2_base_size, self.d = m_base_size, s2_base_size, d
        self.decode_only = decode_only
        self.fd_mode = fd_mode
        self.storage_type = storage_type
        self.causal = int(causal)
        self.m_base_size_per_aiv = m_base_size // 2
        self.layout = layout
        self.q_s_pad, self.q_n = q_geom
        # in_layout: public GM layout of Q/K/V (BNSD/BSND/TND-token-major).
        # layout: kernel-internal output layout (BNSD or head-major TND).
        self.in_layout = in_layout
        self.tok_base_mul = 1 if layout == "TND" else 0
        self.causal = causal
        dep = dict(DEPTHS if depths is None else depths)
        self.vector_subblock_index = get_subblock_id()
        self.kernel_block_index = get_block_idx()
        self.vector = Vector(
            m_base_size // 2,
            s2_base_size,
            d,
            dep,
            qk_scale,
            v_descale,
            p_scale,
            d_v=d_v,
            use_q_descale=use_q_descale,
            k_scale_only=k_scale_only,
            decode_only=decode_only,
            storage_type=storage_type,
        )
        self.matmul = Matmul(
            m_base_size,
            s2_base_size,
            d,
            dep,
            kv_geom=kv_geom,
            storage_type=storage_type,
        )
        self.p_l1 = RingBuffer(
            MemLoc.L1,
            storage_type,
            make_layout((m_base_size, s2_base_size)),
            depth=dep["p_l1"],
        )
        self.s_ub = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((m_base_size // 2, s2_base_size)),
            depth=dep["s_ub"],
        )
        self.pv_ub = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((m_base_size // 2, D_V)),
            depth=2,
        )

    def __call__(
        self,
        gm_query: Tensor,
        gm_key: Tensor,
        gm_core_task_ranges: Tensor,
        gm_output: Tensor,
        batch: int,
        n_q_heads: int,
        head_group: int,
        gm_lse: Tensor = None,
        gm_q_descale: Tensor = None,
        gm_q_token: Tensor = None,
        gm_tile_meta: Tensor = None,
        total_rows: int = 1,
        gm_ws_m: Tensor = None,
        gm_ws_l: Tensor = None,
        gm_ws_acc: Tensor = None,
        n_slices: int = 1,
        gm_bt: Tensor = None,
    ):
        tile_begin = gm_core_task_ranges[0, self.kernel_block_index]
        tile_end = gm_core_task_ranges[1, self.kernel_block_index]
        pipeline_slot_count = SKEW_LAG + 1
        # Head-packed decode: one M-tile = M consecutive flat (token,
        # head) rows of the token-major Q stream; all rows share the
        # single MLA KV head, so a tile spans heads freely (T=1 packs
        # 64 heads per tile instead of one padded head per task).
        kv_head = 0

        for task_idx in range(tile_begin, tile_end):
            if const_expr(self.fd_mode):
                tile_idx = task_idx // n_slices
                slice_idx = task_idx - tile_idx * n_slices
            else:
                tile_idx = task_idx
                slice_idx = 0
            tile_row_start = gm_tile_meta[0, tile_idx]
            key_value_length = gm_tile_meta[1, tile_idx]
            kv_token_base = gm_tile_meta[2, tile_idx]
            causal_offset = gm_tile_meta[3, tile_idx]
            tile_batch = gm_tile_meta[4, tile_idx]
            rows = min(self.m_base_size, total_rows - tile_row_start)

            kv_block_capacity = ceil_div(key_value_length, self.s2_base_size)
            kv_block_count = kv_block_capacity
            mask_begin = kv_block_count - 1
            if const_expr(self.fd_mode):
                blocks_per_slice = ceil_div(kv_block_count, n_slices)
                block_begin = slice_idx * blocks_per_slice
                block_end = min(kv_block_count, block_begin + blocks_per_slice)
            else:
                block_begin = 0
                block_end = kv_block_count

            q_slot = self.vector_q_load(gm_query, tile_row_start)
            self.vector.initialize()
            if const_expr(gm_q_descale is not None):
                self.vector.load_q_descale(
                    gm_q_descale,
                    tile_row_start,
                    self.vector_subblock_index,
                )
            if const_expr(self.causal):
                self.vector.load_q_token(
                    gm_q_token,
                    tile_row_start,
                    self.vector_subblock_index,
                )

            for kv_block_idx in range(block_begin, block_end):
                k_slot = self.matmul.load_k(
                    gm_key,
                    self.layout,
                    self.in_layout,
                    tile_batch,
                    kv_head,
                    kv_block_idx,
                    kv_token_base,
                    gm_bt,
                    key_value_length,
                )
                self.matmul.compute_qk(self.s_ub, q_slot, k_slot)
                if const_expr(self.causal):
                    # Bottom-right causal: kv row visible iff
                    # kv_row <= token[lane] + (kv_len - q_len); the
                    # causal bound subsumes the page tail.
                    self._stage_softmax_p(
                        kv_block_idx,
                        causal_offset - kv_block_idx * self.s2_base_size,
                        key_value_length,
                        kv_block_idx % pipeline_slot_count,
                        MASK_CAUSAL,
                    )
                elif kv_block_idx >= mask_begin:
                    self._stage_softmax_p(
                        kv_block_idx,
                        0,
                        key_value_length,
                        kv_block_idx % pipeline_slot_count,
                        MASK_KV_TAIL,
                    )
                else:
                    self._stage_softmax_p(
                        kv_block_idx,
                        0,
                        key_value_length,
                        kv_block_idx % pipeline_slot_count,
                        MASK_NONE,
                    )

                if kv_block_idx >= block_begin + SKEW_LAG:
                    consume_block_idx = kv_block_idx - SKEW_LAG
                    self._consume(consume_block_idx % pipeline_slot_count)

            for drain in tuple(range(SKEW_LAG)):
                consume_block_idx = block_end + drain - SKEW_LAG
                if block_begin <= consume_block_idx < block_end:
                    self._consume(consume_block_idx % pipeline_slot_count)

            if const_expr(self.fd_mode):
                # FD partial epilogue: raw acc/m/l to GM workspace; the
                # merge kernel folds the slices (wrapper pads out/lse so
                # every row here is a safe write target).
                row_base = (
                    tile_row_start
                    + self.vector_subblock_index * self.m_base_size_per_aiv
                )
                row_end = row_base + self.m_base_size_per_aiv
                # One 2-D (m_per, D_V) tile per slice: a 1-leaf (D_V,)
                # tiler addresses only the trailing dim, so a per-row
                # coord >= 1 lands past the parent domain and copies
                # nothing (DSL 0.8 zero-extent window).
                mem_copy(
                    tile_slice(
                        gm_ws_acc[
                            slice_idx,
                            row_base:row_end,
                            0:D_V,
                        ],
                        make_tiler(
                            (self.m_base_size_per_aiv, D_V),
                            alignment=(16, 16),
                        ),
                        (0, 0),
                    ),
                    tile_slice(
                        self.vector.acc,
                        make_tiler(
                            (self.m_base_size_per_aiv, D_V),
                            alignment=(16, 16),
                        ),
                        (0, 0),
                    ),
                )
                mem_copy(
                    tile_slice(
                        gm_ws_m[slice_idx, row_base:row_end],
                        make_tiler((self.m_base_size_per_aiv,), alignment=(16,)),
                        (0,),
                    ),
                    self.vector.row_m.flatten(),
                )
                mem_copy(
                    tile_slice(
                        gm_ws_l[slice_idx, row_base:row_end],
                        make_tiler((self.m_base_size_per_aiv,), alignment=(16,)),
                        (0,),
                    ),
                    self.vector.row_l.flatten(),
                )
            else:
                self.vector.finalize_output(
                    gm_output,
                    gm_lse,
                    total_rows,
                    tile_row_start,
                    rows,
                    self.vector_subblock_index,
                )

    @jit
    def vector_q_load(self, gm_query, tile_row_start):
        # Head-packed tile: M consecutive flat (token, head) rows; the
        # wrapper zero-pads the stream to a multiple of M so this is a
        # single contiguous (M, D) GM window.
        row_end = tile_row_start + self.m_base_size
        base = gm_query[tile_row_start:row_end, None]
        q_src = tile_slice(
            base,
            make_tiler((self.m_base_size, self.d), alignment=(16, 32)),
            (0, 0),
        )
        return self.matmul.load_q(q_src)

    @jit
    def _stage_softmax_p(
        self,
        kv_block_idx,
        limit_scalar,
        s2_length,
        pipeline_slot,
        kv_block_mask_type,
    ):
        if const_expr(kv_block_mask_type == MASK_CAUSAL):
            visible_limit = limit_scalar
        elif const_expr(kv_block_mask_type == MASK_KV_TAIL):
            visible_limit = s2_length - kv_block_idx * self.s2_base_size - 1
        else:
            visible_limit = 0
        self.vector.softmax(
            self.s_ub.next_read(),
            visible_limit,
            pipeline_slot,
            kv_block_mask_type,
        )
        self.vector.store_p(self.p_l1, self.vector_subblock_index)

    @jit
    def _consume(self, pipeline_slot):
        # MLA: V == K_nope, so reuse the K slot bmm1 just streamed in;
        # bmm2 reads the first D_V columns of that slot, no separate DMA.
        p_source, v_source = self.matmul.pv_consume_sources(self.p_l1)
        self.matmul.compute_pv(self.pv_ub, p_source, v_source)
        self.vector.update_output_acc(self.pv_ub.next_read(), pipeline_slot)


@kernel
class FdMergeKernel:
    """Flash-decode merge: fold the per-slice partials (m_s, l_s, acc_s)
    into O = acc*/l* and LSE = m* + log l*.

    Tasks are (tile, half, col_chunk of 64, row_split); each core merges a
    row window of its column slice across all FD slices.  Col chunk 0 of a
    row window also emits the LSE.

    Same shape as the FP8 sibling: the fold is vector-op bound, so slices are
    folded in groups of ``k_group`` -- one acc read/modify/write per row per
    group instead of per slice -- and the task is split by rows to keep every
    core busy.  ws_acc stays slice-major here, so each slice's staging copy
    is one 2-D block out of a 2-D view of the slice.
    """

    def __init__(self, m_base_size, d_v, v_descale, n_slices, row_splits=1, k_group=4):
        self.m_base_size = m_base_size
        self.m_per_aiv = m_base_size // 2
        self.halves = m_base_size // self.m_per_aiv
        self.d_v = d_v
        self.v_descale = float(v_descale)
        self.n_slices = int(n_slices)
        self.col_chunks = d_v // 64
        self.row_splits = int(row_splits)
        self.k_group = int(k_group)
        self.m_task = self.m_per_aiv // self.row_splits
        # slice indices in fold groups (last group may be short); built here
        # because the kernel body may only iterate static tuples
        self.groups = tuple(
            tuple(range(base, min(base + self.k_group, self.n_slices)))
            for base in range(0, self.n_slices, self.k_group)
        )
        # GM->UB staging goes through channels: in a vector-only kernel a
        # raw Buffer copy gets no MTE2->V (or V->MTE2) ordering, so the
        # per-slice loads race the vector reads (same fix as FP8).
        self.m_chs = [
            RingBuffer(
                MemLoc.UB,
                dtypes.float32,
                make_layout((self.m_task,)),
                depth=2,
            )
            for _ in range(self.k_group)
        ]
        self.l_chs = [
            RingBuffer(
                MemLoc.UB,
                dtypes.float32,
                make_layout((self.m_task,)),
                depth=2,
            )
            for _ in range(self.k_group)
        ]
        # One staging channel holding a whole fold group as k_group row
        # windows (k*m_task .. (k+1)*m_task); depth 2 so the next group's
        # DMAs run while this group's rows are folded.
        self.acc_chs = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((self.k_group * self.m_task, 64)),
            depth=2,
        )
        self.den_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((self.m_task,)))
        self.corr_ubs = [
            Buffer(MemLoc.UB, dtypes.float32, make_layout((self.m_task, 1)))
            for _ in range(self.k_group)
        ]
        self.mstar_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((self.m_task,)))
        self.lstar_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((self.m_task,)))
        self.acc_star = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.m_task, 64))
        )
        self.out_ub = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((self.m_task, 64)),
            depth=1,
        )
        self.lse_ch = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((self.m_task,)),
            depth=1,
        )
        self.block_index = get_block_idx()

    def __call__(
        self,
        gm_ws_m: Tensor,
        gm_ws_l: Tensor,
        gm_ws_acc: Tensor,
        gm_output: Tensor,
        gm_lse: Tensor,
        gm_task_ranges: Tensor,
    ):
        task_begin = gm_task_ranges[0, self.block_index]
        task_end = gm_task_ranges[1, self.block_index]
        # m_per is the per-task row window (m_per_aiv / row_splits)
        m_per = self.m_task
        col_chunks = self.col_chunks
        halves = self.halves
        row_splits = self.row_splits
        n_slices = self.n_slices
        for task_idx in range(task_begin, task_end):
            # (tile, half, col_chunk, row_split) packed in the task id: a
            # vector-only launch gives one AIV per block, so get_subblock_id()
            # stays 0 and both m_base / 2 halves must be explicit tasks.
            group = col_chunks * halves * row_splits
            tile_idx = task_idx // group
            rem = task_idx - tile_idx * group
            col_chunk = rem % col_chunks
            half = (rem // col_chunks) % halves
            row_split = rem // (col_chunks * halves)
            row_base = (
                tile_idx * self.m_base_size + half * self.m_per_aiv + row_split * m_per
            )
            row_end = row_base + m_per
            # ---- pass 1: m* = max_s m_s ----
            with vf(mode="simd"):
                mper, _ = update_mask(m_per, elem_bits=32)
                vstore(
                    self.mstar_ub,
                    0,
                    vdup_scalar(NEG_MAX, dtypes.float32, mask=mper),
                    mper,
                )
                vstore(
                    self.lstar_ub,
                    0,
                    vdup_scalar(0.0, dtypes.float32, mask=mper),
                    mper,
                )
            for s in tuple(range(n_slices)):
                ms = self.m_chs[s % self.k_group].next_write()
                mem_copy(
                    ms,
                    tile_slice(
                        gm_ws_m[s, row_base:row_end],
                        make_tiler((m_per,), alignment=(16,)),
                        (0,),
                    ),
                )
                with vf(mode="simd"):
                    mper, _ = update_mask(m_per, elem_bits=32)
                    vstore(
                        self.mstar_ub,
                        0,
                        vmax(
                            vload(self.mstar_ub, 0),
                            vload(ms, 0),
                            mask=mper,
                        ),
                        mper,
                    )
                    vmem_bar("vst_vld")
            with vf(mode="simd"):
                m64, _ = update_mask(64, elem_bits=32)
                for r in tuple(range(m_per)):
                    vstore(
                        self.acc_star,
                        r * 64,
                        vdup_scalar(0.0, dtypes.float32, mask=m64),
                        m64,
                    )
            # ---- pass 2: l* = sum_s l_s*corr_s, acc* = sum_s acc_s*corr_s
            for group_idx in range_constexpr(len(self.groups)):
                group_slices = self.groups[group_idx]
                accs = self.acc_chs.next_write()
                for k in tuple(range(len(group_slices))):
                    s = group_slices[k]
                    acc_col0 = col_chunk * 64
                    acc_col1 = acc_col0 + 64
                    mem_copy(
                        # tile_slice index is in TILE units, not elements
                        tile_slice(accs, (m_per, 64), (k, 0)),
                        tile_slice(
                            gm_ws_acc[
                                s,
                                row_base:row_end,
                                acc_col0:acc_col1,
                            ],
                            make_tiler((m_per, 64), alignment=(16, 16)),
                            (0, 0),
                        ),
                    )
                    ms = self.m_chs[k].next_write()
                    mem_copy(
                        ms,
                        tile_slice(
                            gm_ws_m[s, row_base:row_end],
                            make_tiler((m_per,), alignment=(16,)),
                            (0,),
                        ),
                    )
                    ls = self.l_chs[k].next_write()
                    mem_copy(
                        ls,
                        tile_slice(
                            gm_ws_l[s, row_base:row_end],
                            make_tiler((m_per,), alignment=(16,)),
                            (0,),
                        ),
                    )
                    # corr/l* accumulate ONCE per slice (NOT per row)
                    with vf(mode="simd"):
                        mper, _ = update_mask(m_per, elem_bits=32)
                        corr = vexp_sub(
                            vload(ms, 0),
                            vload(self.mstar_ub, 0),
                            mask=mper,
                        )
                        vstore(self.corr_ubs[k], 0, corr, mper)
                        vmem_bar("vst_vld")
                        vstore(
                            self.lstar_ub,
                            0,
                            vadd(
                                vload(self.lstar_ub, 0),
                                vmul(corr, vload(ls, 0), mask=mper),
                                mask=mper,
                            ),
                            mper,
                        )
                # ---- 2-D fold: one acc read/write per row per group ----
                with vf(mode="simd"):
                    m64, _ = update_mask(64, elem_bits=32)
                    for r in tuple(range(m_per)):
                        folded = vload(self.acc_star, r * 64)
                        for k in tuple(range(len(group_slices))):
                            # vmadd is lhs*rhs + addend: weight the partial
                            # by corr, then add the running acc
                            folded = vmadd(
                                vload_brc(self.corr_ubs[k], r),
                                vload(accs, k * m_per * 64 + r * 64),
                                folded,
                                mask=m64,
                            )
                        vstore(self.acc_star, r * 64, folded, m64)
            # ---- finalize: O = acc*/l*, LSE = m* + log l* ----
            out_slot = self.out_ub.next_write()
            lse_slot = self.lse_ch.next_write()
            with vf(mode="simd"):
                mper, _ = update_mask(m_per, elem_bits=32)
                m64, _ = update_mask(64, elem_bits=32)
                bf16_mask, _ = update_mask(64, elem_bits=16)
                one = vdup_scalar(1.0, dtypes.float32, mask=mper)
                zero = vdup_scalar(0.0, dtypes.float32, mask=mper)
                inf_vec = vdiv(one, zero, mask=mper)
                l_star = vload(self.lstar_ub, 0)
                is_zero = vles(l_star, 0.0, mask=mper)
                safe_l = vselect(one, l_star, cond_mask=is_zero)
                vstore(self.den_ub, 0, safe_l, mper)
                # same hazard: vload_brc(den_ub, r) reads the store back
                vmem_bar("vst_vld")
                for r in tuple(range(m_per)):
                    denom = vload_brc(self.den_ub, r)
                    even_acc, odd_acc = vload_deinterleave(
                        self.acc_star, r * 64, width="b32"
                    )
                    even_acc = vdiv(even_acc, denom, mask=m64)
                    odd_acc = vdiv(odd_acc, denom, mask=m64)
                    # ws_acc already carries v_descale (baked in by the
                    # per-band PV accumulate); do not scale again here.
                    even_bf = vcast(
                        even_acc,
                        dtypes.bfloat16,
                        mask=m64,
                        reg_layout=RegLayout.ZERO,
                    )
                    odd_bf = vcast(
                        odd_acc,
                        dtypes.bfloat16,
                        mask=m64,
                        reg_layout=RegLayout.ONE,
                    )
                    vstore(
                        out_slot,
                        r * 64,
                        vor(even_bf, odd_bf, mask=bf16_mask),
                        bf16_mask,
                    )
                m_star = vload(self.mstar_ub, 0)
                lse_all = vadd(m_star, vlog(safe_l, mask=mper), mask=mper)
                lse_all = vselect(inf_vec, lse_all, cond_mask=is_zero)
                vstore(lse_slot, 0, lse_all, mper)
            col0 = col_chunk * 64
            col_hi = col0 + 64
            mem_copy(
                tile_slice(
                    gm_output[
                        row_base:row_end,
                        col0:col_hi,
                    ],
                    make_tiler((m_per, 64), alignment=(1, 1)),
                    (0, 0),
                ),
                out_slot,
            )
            if col_chunk == 0:
                mem_copy(
                    tile_slice(
                        gm_lse[0, row_base:row_end],
                        make_tiler((m_per,), alignment=(16,)),
                        (0,),
                    ),
                    lse_slot,
                )


# Flash-decode S2 split: kernel1 writes the per-slice partials (m/l/acc)
# and FdMergeKernel folds them on device.  Same five fixes as the FP8
# sibling (r-major/major layout, explicit half tasks, per-row 1-D acc
# copies, channel staging + vst_vld, vmadd operand order).


class FlashAttnHif8Launcher:
    def __init__(
        self,
        causal=False,
        layout="BNSD",
        in_layout="BNSD",
        qk_scale=1.0,
        v_descale=1.0,
        p_scale=1.0,
        q_geom=(0, 0),
        kv_geom=(0, 0),
        use_q_descale=False,
        k_scale_only=1.0,
        decode_only=False,
        fd_mode=False,
        m_base_size=M_BASE_SIZE,
        s2_base_size=S2_BASE_SIZE,
        storage_type=dtypes.hifloat8,
    ):
        self.m_base_size = int(m_base_size)
        self.s2_base_size = int(s2_base_size)
        self.causal = int(causal)
        self.decode_only = decode_only
        self.fd_mode = fd_mode
        self.storage_type = storage_type
        self.layout = layout
        self.in_layout = in_layout
        self.qk_scale = qk_scale
        self.v_descale = v_descale
        self.p_scale = p_scale
        self.q_geom = q_geom
        self.kv_geom = kv_geom
        self.use_q_descale = use_q_descale
        self.k_scale_only = k_scale_only

    @host
    def launch(
        self,
        gm_output,
        gm_query,
        gm_key,
        gm_core_task_ranges,
        batch: int,
        n_q_heads: int,
        head_group: int,
        block_dim: int,
        gm_lse: Tensor = None,
        gm_q_descale: Tensor = None,
        gm_q_token: Tensor = None,
        gm_tile_meta: Tensor = None,
        total_rows: int = 1,
        gm_ws_m: Tensor = None,
        gm_ws_l: Tensor = None,
        gm_ws_acc: Tensor = None,
        n_slices: int = 1,
        gm_bt: Tensor = None,
    ):
        op = FlashAttnHif8Kernel(
            depths=None,
            # Both tile widths must reach the kernel: the host plans tiles
            # with the same values, so a silent fallback to the class
            # defaults desynchronises the KV tile stride and the block
            # count (S2 mismatch shows up as corrupt KV from block 1 on).
            m_base_size=self.m_base_size,
            s2_base_size=self.s2_base_size,
            causal=self.causal,
            layout=self.layout,
            in_layout=self.in_layout,
            qk_scale=self.qk_scale,
            v_descale=self.v_descale,
            p_scale=self.p_scale,
            q_geom=self.q_geom,
            kv_geom=self.kv_geom,
            use_q_descale=self.use_q_descale,
            k_scale_only=self.k_scale_only,
            decode_only=self.decode_only,
            fd_mode=self.fd_mode,
            storage_type=self.storage_type,
        )
        op[block_dim](
            gm_query,
            gm_key,
            gm_core_task_ranges,
            gm_output,
            batch,
            n_q_heads,
            head_group,
            gm_lse,
            gm_q_descale,
            gm_q_token,
            gm_tile_meta,
            total_rows,
            gm_ws_m,
            gm_ws_l,
            gm_ws_acc,
            n_slices,
            gm_bt,
        )


def _launch_fd_merge(ws_m, ws_l, ws_acc, out, lse, n_slices, total_tiles, cores, vd):
    if not os.environ.get("FLA_FULLQUANT_FD_HOST_MERGE"):
        _launch_fd_merge_kernel(
            ws_m, ws_l, ws_acc, out, lse, n_slices, total_tiles, cores, vd
        )
        return
    # Host fallback fold of the per-slice partials (m*, l*, acc*):
    #   m* = max_s m_s;  corr_s = exp(m_s - m*);  l* = sum_s l_s*corr_s;
    #   acc* = sum_s acc_s*corr_s;  O = acc*/l*  (v_descale is already
    # baked into ws_acc by the per-band PV accumulate).
    m_star = ws_m.max(dim=0).values
    corr = torch.exp(ws_m - m_star.unsqueeze(0))
    l_star = (ws_l * corr).sum(dim=0)
    acc_star = (ws_acc * corr.unsqueeze(-1)).sum(dim=0)
    # all-masked rows: partial accs are exactly 0 and l_s == 0, so the
    # safe denominator yields out 0 and LSE +inf (AscendC semantics).
    safe_l = torch.where(l_star > 0, l_star, torch.ones_like(l_star))
    out.copy_(acc_star / safe_l.unsqueeze(-1))
    if lse is not None:
        rows = out.shape[0]
        inf = torch.full_like(l_star, float("inf"))
        lse[0, :rows].copy_(torch.where(l_star <= 0, inf, m_star + torch.log(safe_l)))


def _launch_fd_merge_kernel(
    ws_m, ws_l, ws_acc, out, lse, n_slices, total_tiles, cores, vd
):
    col_chunks = D_V // 64
    # (tile, half, col_chunk, row_split): one AIV per block, so halves are
    # separate tasks; the fold is vector-op bound, so a tile is also split
    # by rows -- target ~2 tasks per core instead of leaving half the
    # device idle.
    base_tasks = total_tiles * col_chunks * 2
    # A row split must keep m_task a multiple of 16 elements: the 1-D staging
    # DMA declares alignment (16,), so an 8-row leaf fails MLIR verification.
    max_row_splits = max(1, (M_BASE_SIZE // 2) // 16)
    row_splits = max(
        1, min(4, max_row_splits, (2 * cores + base_tasks - 1) // base_tasks)
    )
    row_splits = int(os.environ.get("FLA_FULLQUANT_FD_ROW_SPLITS", row_splits))
    row_splits = max(1, min(row_splits, max_row_splits))
    k_group = int(os.environ.get("FLA_FULLQUANT_FD_KGROUP", "4"))
    merge = FdMergeKernel(
        M_BASE_SIZE, D_V, vd, n_slices, row_splits=row_splits, k_group=k_group
    )
    merge_tasks = base_tasks * row_splits
    merge_ranges = [[0] * cores, [0] * cores]
    per = merge_tasks // cores
    extra = merge_tasks % cores
    start = 0
    for core in range(cores):
        count = per + (1 if core < extra else 0)
        merge_ranges[0][core] = start
        merge_ranges[1][core] = start + count
        start += count
    ranges_t = torch.tensor(merge_ranges, dtype=torch.int32).npu()

    merge_dim = max(1, min(cores, merge_tasks))

    @host
    def _merge_launch(a, b, c, out_t, lse_t, r):
        merge[merge_dim](a, b, c, out_t, lse_t, r)

    # LSE-off: write LSE partials into a throwaway tensor (never `out`,
    # the earlier dummy alias scribbled over the attention output).
    lse_arg = lse if lse is not None else torch.empty_like(ws_m[0:1])
    _merge_launch(ws_m, ws_l, ws_acc, out, lse_arg, ranges_t)


def _build_task_ranges(batch, q_lengths, n_q_heads, m_base_size, cores):
    boundaries = [0]
    for length in q_lengths:
        boundaries.append(boundaries[-1] + int(length))

    def _cd(a, b):
        return (int(a) + int(b) - 1) // int(b)

    total = sum(n_q_heads * _cd(length, m_base_size) for length in q_lengths)
    ranges = [[0] * cores, [0] * cores]
    per_core = total // cores
    extra = total % cores
    start = 0
    for core in range(cores):
        count = per_core + (1 if core < extra else 0)
        ranges[0][core] = start
        ranges[1][core] = start + count
        start += count
    return ranges, boundaries


_LAYOUT_Q_SET = ("BSND", "BNSD", "TND", "TND_NTD")
_LAYOUT_KV_SET = ("PA_BBND", "PA_BNBD", "PA_NZ")
_LAYOUT_OUT_SET = ("BSND", "BNSD", "TND", "NTD")
_Q_N_ALLOWED = {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128}


def _to_int8_view(t):
    """Equal-width int8 bit storage of an 8-bit float tensor."""
    if t.dtype == torch.int8:
        return t
    return t.view(torch.int8)


def _normalize_q(q, q_descale, layout_q, cu_seqlens_q):
    """Per-batch (token, head)-major row streams + flat descale.

    Returns (per_batch, q_n) where per_batch[b] = (rows_b (rows, 576)
    int8 view, qd_b (rows,) fp32, q_len_b).
    """
    if layout_q in ("BSND",):
        n_batch, q_seq, n_q_heads, d_model = q.shape
        qd_shape = tuple(q_descale.shape)
        if qd_shape != (n_batch, q_seq, n_q_heads):
            raise ValueError(
                f"q_descale must be {(n_batch, q_seq, n_q_heads)}, got {qd_shape}"
            )
        rows = _to_int8_view(q).reshape(n_batch, q_seq * n_q_heads, d_model)
        qd = q_descale.reshape(n_batch, q_seq * n_q_heads)
        return [(rows[b], qd[b], q_seq) for b in range(n_batch)], n_q_heads
    if layout_q == "BNSD":
        n_batch, n_q_heads, q_seq, d_model = q.shape
        qd_shape = tuple(q_descale.shape)
        if qd_shape != (n_batch, n_q_heads, q_seq):
            raise ValueError(
                f"q_descale must be {(n_batch, n_q_heads, q_seq)}, got {qd_shape}"
            )
        q_t = (
            _to_int8_view(q)
            .permute(0, 2, 1, 3)
            .reshape(n_batch, q_seq * n_q_heads, d_model)
        )
        qd = q_descale.permute(0, 2, 1).reshape(n_batch, q_seq * n_q_heads)
        return [(q_t[b], qd[b], q_seq) for b in range(n_batch)], n_q_heads
    if layout_q in ("TND", "TND_NTD"):
        q_tok, n_q_heads, d_model = q.shape
        qd_shape = tuple(q_descale.shape)
        if qd_shape != (q_tok, n_q_heads):
            raise ValueError(f"q_descale must be {(q_tok, n_q_heads)}, got {qd_shape}")
        rows_all = _to_int8_view(q).reshape(q_tok * n_q_heads, d_model)
        qd_all = q_descale.reshape(q_tok * n_q_heads)
        if cu_seqlens_q is None:
            return [(rows_all, qd_all, q_tok)], n_q_heads
        cu = cu_seqlens_q.detach().cpu().tolist()
        if len(cu) < 2 or cu[0] != 0 or cu[-1] != q_tok:
            raise ValueError("cu_seqlens_q must start at 0 and end at q_tok")
        out = []
        for b in range(len(cu) - 1):
            lo, hi = cu[b] * n_q_heads, cu[b + 1] * n_q_heads
            out.append((rows_all[lo:hi], qd_all[lo:hi], cu[b + 1] - cu[b]))
        return out, n_q_heads
    raise ValueError(f"unsupported layout_q={layout_q!r}")


def _check_paged_cache(k_cache, block_table, cache_seqlens, layout_kv):
    """Validate the paged cache and return (page-major view, kv_lens).

    PA_BBND ``(pages, 128, 1, 576)`` and PA_BNBD ``(pages, 1, 128, 576)``
    are both page-major once KV_N=1, so one ``(pages, page, 1, D)`` int8
    view serves the kernel's page-table addressing for either public
    layout.  The rows are never re-materialised on the host.
    """
    if layout_kv == "PA_NZ":
        raise NotImplementedError("PA_NZ cache layout is not supported yet")
    if k_cache.dim() != 4 or k_cache.shape[1] * k_cache.shape[2] != PAGE_SIZE:
        raise ValueError(
            f"paged KV cache must be (pages, {PAGE_SIZE}, 1, {D}) or "
            f"(pages, 1, {PAGE_SIZE}, {D}), got {tuple(k_cache.shape)}"
        )
    kv_lens = [int(x) for x in cache_seqlens.detach().cpu().tolist()]
    for b, kvl in enumerate(kv_lens):
        need = (kvl + PAGE_SIZE - 1) // PAGE_SIZE
        if need > block_table.shape[1]:
            raise ValueError(
                f"block_table has {block_table.shape[1]} pages, batch {b} needs {need}"
            )
    # aclnn-free: the 8-bit float cache is addressed by its int8 bit view.
    # Page-major (pages, page, 1, D) is what the kernel's page addressing
    # consumes; PA_BNBD only swaps the two singleton axes.
    pool = _to_int8_view(k_cache)
    if k_cache.shape[1] != PAGE_SIZE:
        pool = pool.transpose(1, 2)
    return pool, kv_lens


def _plan_tiles(per_batch, q_n, kv_lens):
    """Per-tile metadata (5, tiles): row_start / kv_len / kv_base /
    causal_offset / batch, plus per-batch padded row streams."""
    q_segs, qd_segs, tok_segs = [], [], []
    tile_rows, tile_kv, tile_base, tile_causal, tile_batch = [], [], [], [], []
    row_cursor = 0
    for b, (rows_b, qd_b, q_len_b) in enumerate(per_batch):
        rows = rows_b.shape[0]
        pad = (-rows) % M_BASE_SIZE
        if pad:
            q_segs.append(
                torch.cat(
                    [
                        rows_b,
                        torch.zeros(pad, D, dtype=torch.int8, device=rows_b.device),
                    ],
                    dim=0,
                )
            )
            qd_segs.append(
                torch.cat(
                    [qd_b, torch.ones(pad, dtype=torch.float32, device=qd_b.device)],
                    dim=0,
                )
            )
        else:
            q_segs.append(rows_b)
            qd_segs.append(qd_b)
        # per-row token index within the batch (padding -> 0); host-side
        # arange (rows is a python int) avoids two device kernels
        if pad:
            tok = torch.cat(
                [
                    torch.arange(rows, dtype=torch.float32) // q_n,
                    torch.zeros(pad, dtype=torch.float32),
                ]
            )
        else:
            tok = torch.arange(rows, dtype=torch.float32) // q_n
        tok_segs.append(tok)
        tiles = (rows + pad) // M_BASE_SIZE
        causal_off = kv_lens[b] - q_len_b
        for t in range(tiles):
            tile_rows.append(row_cursor + t * M_BASE_SIZE)
            tile_kv.append(kv_lens[b])
            tile_base.append(_plan_tiles.bases[b])
            tile_causal.append(causal_off)
            tile_batch.append(b)
        row_cursor += rows + pad
    meta = torch.tensor(
        [tile_rows, tile_kv, tile_base, tile_causal, tile_batch], dtype=torch.int32
    )
    q_flat = q_segs[0] if len(q_segs) == 1 else torch.cat(q_segs, dim=0)
    qd_flat = qd_segs[0] if len(qd_segs) == 1 else torch.cat(qd_segs, dim=0)
    tok_flat = (tok_segs[0] if len(tok_segs) == 1 else torch.cat(tok_segs, dim=0)).to(
        q_segs[0].device
    )
    return meta, q_flat, qd_flat, tok_flat, row_cursor


def flash_mla_fullquant_hif8_metadata(
    cache_seqlens,
    num_heads_q,
    num_heads_kv,
    quant_mode,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    head_dim_qk=576,
    head_dim_v=512,
    mask_mode=0,
    layout_q="BSND",
):
    """Plan the kernel task schedule; returns an int32 bundle.

    Layout: row 0 = header [n_batch, n_q_heads, tiles, quant_mode, mask_mode,
    layout_q_code, cores]; rows 1..4 = tile_meta (4, tiles); rows
    5..6 = core task ranges (2, cores).
    """
    if head_dim_qk != 576 or head_dim_v != 512:
        raise ValueError("MLA requires head_dim_qk=576, head_dim_v=512")
    if num_heads_kv != 1:
        raise ValueError("MLA requires num_heads_kv=1")
    if quant_mode not in (0, 1):
        raise ValueError(f"quant_mode must be 0 (HIF8) or 1 (FP8), got {quant_mode}")
    if mask_mode not in (0, 3):
        raise ValueError("mask_mode must be 0 (NO_MASK) or 3 (CAUSAL)")
    if layout_q not in _LAYOUT_Q_SET:
        raise ValueError(f"unsupported layout_q={layout_q!r}")
    n_batch = int(cache_seqlens.numel())
    if n_batch == 0:
        raise ValueError("cache_seqlens must be non-empty")
    kv_lens = [int(x) for x in cache_seqlens.detach().cpu().tolist()]
    max_kv = max(kv_lens) if kv_lens else 0
    if max_seqlen_kv > 0 and max(max_kv, 0) > max_seqlen_kv:
        raise ValueError("cache_seqlens exceeds max_seqlen_kv")
    # per-batch q length upper bound for planning
    if cu_seqlens_q is not None:
        cu = cu_seqlens_q.detach().cpu().tolist()
        if len(cu) - 1 != n_batch:
            raise ValueError("cu_seqlens_q length must be batch+1")
        q_lens = [cu[b + 1] - cu[b] for b in range(n_batch)]
    elif max_seqlen_q > 0:
        q_lens = [max_seqlen_q] * n_batch
    else:
        q_lens = [1] * n_batch  # decode default
    tiles = sum((ql * num_heads_q + M_BASE_SIZE - 1) // M_BASE_SIZE for ql in q_lens)
    from cannbotdsl import get_platform_info as _gpi

    try:
        cores = int(_gpi().cube_core_num) or 32
    except (OSError, RuntimeError, TypeError, ValueError):
        cores = 32
    # Flash-decode schedule: slice each tile's KV stream when M-tiles
    # under-fill the cores (mirrors the wrapper's n_slices decision).
    kv_blocks_max = max(
        1, max((kvlen + S2_BASE_SIZE - 1) // S2_BASE_SIZE for kvlen in kv_lens)
    )
    fd_tasks = tiles
    if tiles < cores:
        n_slices = max(1, min((cores + tiles - 1) // tiles, kv_blocks_max))
        fd_tasks = tiles * n_slices
    ranges = [[0] * cores, [0] * cores]
    per, extra = fd_tasks // cores, fd_tasks % cores
    start = 0
    for core in range(cores):
        count = per + (1 if core < extra else 0)
        ranges[0][core] = start
        ranges[1][core] = start + count
        start += count
    header = [
        n_batch,
        num_heads_q,
        fd_tasks,
        quant_mode,
        mask_mode,
        _LAYOUT_Q_SET.index(layout_q),
        cores,
    ]
    width = max(fd_tasks, cores, len(header))
    tile_rows = [[0] * width for _ in range(4)]
    rng = [[r[c] if c < cores else 0 for c in range(width)] for r in ranges]
    head_row = header + [0] * (width - len(header))
    return torch.tensor([head_row] + tile_rows + rng, dtype=torch.int32)


def flash_mla_fullquant_hif8(
    q,
    k_cache,
    q_descale,
    k_descale,
    block_table,
    cache_seqlens,
    quant_mode,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=512,
    softmax_scale=None,
    mask_mode=0,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BNBD",
    layout_out="BSND",
    return_softmax_lse=False,
):
    """HIF8 MLA decode (CANNBot-DSL native).

    The kernel computes
    QK/softmax/PV natively in 8-bit; V shares the K descale (K/V come
    from one cache); p_scale is an internal detail fixed at 1.
    """
    import torch_npu

    # ---------- validation ----------
    if quant_mode != 0:
        raise ValueError(f"the HIF8 operator requires quant_mode=0, got {quant_mode}")
    # HIF8: torch_npu.hifloat8 tensors cannot round-trip in this
    # build; the official API accepts hifloat8 dtype OR equal-width
    # int8 bit storage (DESIGN section 1).
    if q.dtype not in (getattr(torch_npu, "hifloat8", None), torch.int8):
        raise TypeError(f"q dtype {q.dtype} does not match quant_mode=0")
    if k_cache.dtype not in (getattr(torch_npu, "hifloat8", None), torch.int8):
        raise TypeError(f"k_cache dtype {k_cache.dtype} does not match quant_mode=0")
    _ss, kd, k_scale_only = validate_scales(q_descale, k_descale, softmax_scale)
    if block_table.dtype != torch.int32:
        raise TypeError("block_table must be int32")
    if cache_seqlens.dtype != torch.int32:
        raise TypeError("cache_seqlens must be int32")
    if layout_q not in _LAYOUT_Q_SET:
        raise ValueError(f"unsupported layout_q={layout_q!r}")
    if layout_kv not in _LAYOUT_KV_SET:
        raise ValueError(f"unsupported layout_kv={layout_kv!r}")
    if layout_out not in _LAYOUT_OUT_SET:
        raise ValueError(f"unsupported layout_out={layout_out!r}")
    if head_dim_v != 512:
        raise ValueError("head_dim_v must be 512")
    if q.shape[-1] != D:
        raise ValueError(f"head_dim_qk must be {D}, got {q.shape[-1]}")
    if k_cache.shape[-1] != D:
        raise ValueError(f"K cache last dim must be {D}")
    if mask_mode not in (0, 3):
        raise ValueError("mask_mode must be 0 or 3")
    if mask_mode == 3:
        if attn_mask is None:
            raise ValueError("mask_mode=3 requires attn_mask")
        if attn_mask.dtype != torch.int8 or tuple(attn_mask.shape) != (2048, 2048):
            raise ValueError("attn_mask must be int8 (2048, 2048)")
        # NOTE: the kernel applies the analytic bottom-right causal
        # pattern; a non-standard attn_mask contents is not honored.

    # ---------- normalize Q / gather KV ----------
    per_batch, q_n = _normalize_q(q, q_descale, layout_q, cu_seqlens_q)
    n_batch = len(per_batch)
    if int(cache_seqlens.numel()) != n_batch:
        raise ValueError("cache_seqlens batch does not match Q")
    if q_n not in _Q_N_ALLOWED:
        raise ValueError(f"unsupported num_heads_q={q_n}")
    k_flat, kv_lens = _check_paged_cache(k_cache, block_table, cache_seqlens, layout_kv)
    _plan_tiles.bases = [0] * n_batch
    in_layout_kv = "PAGED"
    gm_bt = block_table
    # One KV tile spans whole pages: the kernel writes each page into its
    # own 128-row half of the tile slot, so the block count (and the
    # per-block fixed cost) stays at the S2 tile granularity.
    s2_base = S2_BASE_SIZE
    if s2_base % PAGE_SIZE:
        raise ValueError(
            f"S2 tile {s2_base} must be a whole number of {PAGE_SIZE}-row pages"
        )
    tile_meta, q_flat, qd_flat, tok_flat, total_rows_padded = _plan_tiles(
        per_batch, q_n, kv_lens
    )
    total_tiles = tile_meta.shape[1]

    # ---------- metadata reuse ----------
    from cannbotdsl import get_platform_info as _gpi

    try:
        cores = int(_gpi().cube_core_num) or 32
    except (OSError, RuntimeError, TypeError, ValueError):
        cores = 32
    # Flash-decode split: when M-tiles under-fill the cores, slice each
    # tile's KV stream across n_slices tasks so all cores stream KV in
    # parallel; per-slice partials (m/l/acc) fold in the merge.
    kv_blocks_max = max(1, max((kvlen + s2_base - 1) // s2_base for kvlen in kv_lens))
    n_slices = 1
    if total_tiles < cores:
        n_slices = max(1, min((cores + total_tiles - 1) // total_tiles, kv_blocks_max))
    # debug knobs: cap the FD split / force the FD pipeline shape
    _cap = int(os.environ.get("FLA_FULLQUANT_FD_MAX_SLICES", "0") or 0)
    if _cap > 0:
        n_slices = max(1, min(n_slices, _cap))
    fd_tasks = total_tiles * n_slices
    core_task_ranges = None
    if metadata is not None:
        if metadata.dtype != torch.int32 or metadata.dim() != 2:
            raise TypeError("metadata must be a 2-D int32 tensor")
        h = metadata[0].tolist()
        if h[1] != q_n or h[3] != quant_mode or h[4] != mask_mode:
            raise ValueError("metadata does not match op parameters")
        if h[0] == n_batch and h[2] == fd_tasks and metadata.shape[0] >= 7:
            ranges_t = metadata[5:7].to(q.device)
            if tuple(ranges_t.shape) == (2, cores):
                core_task_ranges = ranges_t
    if core_task_ranges is None:
        ranges = [[0] * cores, [0] * cores]
        per, extra = fd_tasks // cores, fd_tasks % cores
        start = 0
        for core in range(cores):
            count = per + (1 if core < extra else 0)
            ranges[0][core] = start
            ranges[1][core] = start + count
            start += count
        core_task_ranges = torch.tensor(ranges, dtype=torch.int32, device=q.device)
    tile_meta = tile_meta.to(q.device)

    # ---------- scales ----------
    vd = kd  # V shares the K per-tensor descale (single cache)

    qd_gm = qd_flat.view(1, -1).to(q.device)
    tok_gm = tok_flat.view(1, -1).to(q.device)
    # every row is written by the kernel finalize / FD merge, so empty
    # is safe and skips the per-call zero-fill kernels
    out = torch.empty(total_rows_padded, D_V, dtype=torch.bfloat16, device=q.device)
    lse_flat = (
        torch.empty(1, total_rows_padded, dtype=torch.float32, device=q.device)
        if return_softmax_lse
        else None
    )

    storage_type = dtypes.hifloat8
    launcher = FlashAttnHif8Launcher(
        causal=(mask_mode == 3),
        layout="TND",
        in_layout=in_layout_kv,
        qk_scale=k_scale_only,
        v_descale=vd,
        p_scale=1.0,
        q_geom=(0, 1),
        kv_geom=(0, 1),
        use_q_descale=True,
        k_scale_only=k_scale_only,
        decode_only=False,
        fd_mode=(n_slices > 1) or bool(os.environ.get("FLA_FULLQUANT_FD_FORCE", "")),
        m_base_size=M_BASE_SIZE,
        s2_base_size=s2_base,
        storage_type=storage_type,
    )
    block_dim = max(1, min(cores, fd_tasks))
    ws_m = ws_l = ws_acc = None
    if n_slices > 1:
        ws_m = torch.empty(
            (n_slices, total_rows_padded), dtype=torch.float32, device=q.device
        )
        ws_l = torch.empty_like(ws_m)
        ws_acc = torch.empty(
            (n_slices, total_rows_padded, D_V), dtype=torch.float32, device=q.device
        )
    launcher.launch(
        out,
        q_flat,
        k_flat,
        core_task_ranges,
        n_batch,
        q_n,
        1,
        block_dim,
        lse_flat,
        qd_gm,
        tok_gm,
        tile_meta,
        total_rows_padded,
        gm_ws_m=ws_m,
        gm_ws_l=ws_l,
        gm_ws_acc=ws_acc,
        n_slices=n_slices,
        gm_bt=gm_bt,
    )
    if n_slices > 1:
        _launch_fd_merge(
            ws_m, ws_l, ws_acc, out, lse_flat, n_slices, total_tiles, cores, vd
        )

    # ---------- restore layout_out / LSE ----------
    out_flat = out
    lse_out = torch.empty(0, device=q.device)
    per_rows = [r.shape[0] for (r, _, _) in per_batch]
    per_q = [ql for (_, _, ql) in per_batch]
    outs, lses = [], []
    cursor = 0
    for b in range(n_batch):
        rows_b = per_rows[b]
        valid = rows_b
        if seqused_q is not None:
            su = int(seqused_q.reshape(-1)[b].item())
            valid = min(rows_b, su * q_n)
        valid_end = cursor + valid
        sl = out_flat[cursor:valid_end]
        if valid < rows_b:
            sl = torch.cat(
                [
                    sl,
                    torch.zeros(
                        rows_b - valid, D_V, dtype=torch.bfloat16, device=q.device
                    ),
                ],
                dim=0,
            )
        outs.append(sl.reshape(per_q[b], q_n, D_V))
        if return_softmax_lse:
            rows_end = cursor + rows_b
            lses.append(lse_flat[0, cursor:rows_end].reshape(per_q[b], q_n))
        cursor += rows_b + (-rows_b) % M_BASE_SIZE
    if layout_out in ("BSND", "BNSD"):
        if len(set(per_q)) > 1:
            raise ValueError("BSND/BNSD output requires uniform q_len")
        if layout_out == "BSND":
            attn = torch.stack(outs, dim=0) if n_batch > 1 else outs[0].unsqueeze(0)
            if return_softmax_lse:
                lse_out = torch.stack(lses, dim=0).permute(0, 2, 1).contiguous()
        else:
            attn = (
                (torch.stack(outs, dim=0) if n_batch > 1 else outs[0].unsqueeze(0))
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            if return_softmax_lse:
                lse_out = torch.stack(lses, dim=0).permute(0, 2, 1).contiguous()
    else:  # TND / NTD
        tnd = torch.cat(outs, dim=0)
        attn = tnd if layout_out == "TND" else tnd.permute(1, 0, 2).contiguous()
        if return_softmax_lse:
            lse_t = torch.cat(lses, dim=0)
            # Official contract: TND and NTD both return (Q_N, Q_T).
            lse_out = lse_t.t().contiguous()
    return attn, lse_out


# HIF8 operator (quant_mode=0); the FP8 E4M3 sibling lives in
# flash_mla_fullquant_fp8.py.
