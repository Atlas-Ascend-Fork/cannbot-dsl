# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""BF16 vector attention over window and compressed paged caches.

Two-pass tiling like `fused_attention_core.py`: K/V gathered per tile (BN=64)
while full-length score[K]/p[K] stay small.  Token ``l`` is in the window when
``l < win_len`` (= win_pages*page_size), else it indexes the compressed cache at
``l - win_len``.

Baked-in lessons:
  * invalid K/V rows are zeroed from a **GM zero tensor** via ``mem_copy``
    (bf16 vstore zeroing is unreliable);
  * ``-1`` lanes keep ``score = -inf`` (score is only written for valid keys);
  * the bf16 output uses ``vstore_pack(B32_TO_B16)``.
"""

from __future__ import annotations

from cannbotdsl import make_layout
from cannbotdsl import MemLoc, RegLayout, dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl import Tensor, tile_slice as tile_view

NEG_INF = -1.7e38
VECTOR_CHUNK_ELEMENTS = 64  # Logical feature elements processed per SIMD chunk.


class DualCacheVectorAttentionKernel:
    def __init__(
        self,
        m: int,
        d: int,
        k: int,
        page_size: int,
        win_pages: int,
        cmp_pages: int,
        bn: int = 64,
    ):
        if d % VECTOR_CHUNK_ELEMENTS or k % bn:
            raise ValueError("D a multiple of 64 and K a multiple of BN")
        self.m, self.d, self.k, self.page_size = m, d, k, page_size
        self.win_pages, self.cmp_pages = win_pages, cmp_pages
        self.win_len = win_pages * page_size
        self.bn = bn
        self.dc = d // VECTOR_CHUNK_ELEMENTS
        self.ntile = k // bn
        self.n_chunk = k // VECTOR_CHUNK_ELEMENTS

    @jit
    def _zero_bf16_kv_rows(self, buf, rows, zero_gm):
        for j in range(rows):
            mem_copy(tile_view(buf, (1, self.d), (j, 0)), zero_gm)

    @jit
    def _initialize_masked_scores(self, score_ub):
        with vf(mode="simd"):
            full = rr.full_mask()
            for c in range(self.n_chunk):
                rr.vstore(
                    score_ub,
                    c * VECTOR_CHUNK_ELEMENTS,
                    rr.vdups(NEG_INF, dtypes.float32, mask=full),
                    full,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def _upcast_query(self, qb_ub, q_ub):
        with vf(mode="simd"):
            full = rr.full_mask()
            for c in range(self.dc):
                rr.vstore(
                    q_ub,
                    c * VECTOR_CHUNK_ELEMENTS,
                    rr.vcast(
                        rr.vload_unpack(
                            qb_ub,
                            c * VECTOR_CHUNK_ELEMENTS,
                            unpack_mode=rr.UnpackMode.B16_TO_B32,
                        ),
                        dtypes.float32,
                        mask=full,
                    ),
                    full,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def _compute_scaled_query_key_dot(self, keyb_ub, j, q_ub, score_ub, base):
        with vf(mode="simd"):
            full = rr.full_mask()
            acc = rr.vdups(0.0, dtypes.float32, mask=full)
            for c in range(self.dc):
                kb = rr.vcast(
                    rr.vload_unpack(
                        keyb_ub,
                        j * self.d + c * VECTOR_CHUNK_ELEMENTS,
                        unpack_mode=rr.UnpackMode.B16_TO_B32,
                    ),
                    dtypes.float32,
                    mask=full,
                )
                acc = rr.vadd(
                    acc,
                    rr.vmul(rr.vload(q_ub, c * VECTOR_CHUNK_ELEMENTS), kb, mask=full),
                    mask=full,
                )
            s = rr.vreduce_sum(acc, mask=full)
            s = rr.vmuls(s, 1.0 / (self.d**0.5), mask=full)
            rr.vstore_first(score_ub, base + j, s)
            rr.vmem_bar("vst_vld")

    @jit
    def _compute_sink_softmax_statistics(self, score_ub, sink_ub, p_ub, denom_ub):
        with vf(mode="simd"):
            full = rr.full_mask()
            m = rr.vdups(NEG_INF, dtypes.float32, mask=full)
            for c in range(self.n_chunk):
                m = rr.vmax(
                    m,
                    rr.vreduce_max(
                        rr.vload(score_ub, c * VECTOR_CHUNK_ELEMENTS), mask=full
                    ),
                    mask=full,
                )
            sink_b = rr.vload_broadcast(sink_ub, 0)
            m = rr.vdup(rr.vmax(m, sink_b, mask=full), mask=full)
            s = rr.vdups(0.0, dtypes.float32, mask=full)
            for c in range(self.n_chunk):
                p = rr.vexp_sub(
                    rr.vload(score_ub, c * VECTOR_CHUNK_ELEMENTS), m, mask=full
                )
                rr.vstore(p_ub, c * VECTOR_CHUNK_ELEMENTS, p, full)
                s = rr.vadd(s, rr.vreduce_sum(p, mask=full), mask=full)
            rr.vstore_first(
                denom_ub, 0, rr.vadd(s, rr.vexp_sub(sink_b, m, mask=full), mask=full)
            )
            rr.vmem_bar("vst_vld")

    @jit
    def _accumulate_weighted_values(self, valb_ub, p_ub, base, res_acc_ub):
        with vf(mode="simd"):
            full = rr.full_mask()
            for c in range(self.dc):
                res = rr.vload(res_acc_ub, c * VECTOR_CHUNK_ELEMENTS)
                for j in range(self.bn):
                    pj = rr.vload_broadcast(p_ub, base + j)
                    vj = rr.vcast(
                        rr.vload_unpack(
                            valb_ub,
                            j * self.d + c * VECTOR_CHUNK_ELEMENTS,
                            unpack_mode=rr.UnpackMode.B16_TO_B32,
                        ),
                        dtypes.float32,
                        mask=full,
                    )
                    res = rr.vadd(res, rr.vmul(vj, pj, mask=full), mask=full)
                rr.vstore(res_acc_ub, c * VECTOR_CHUNK_ELEMENTS, res, full)
            rr.vmem_bar("vst_vld")

    @jit
    def _normalize_attention_output(self, res_acc_ub, denom_ub, outb_ub):
        with vf(mode="simd"):
            full = rr.full_mask()
            denom_b = rr.vload_broadcast(denom_ub, 0)
            for c in range(self.dc):
                out = rr.vdiv(
                    rr.vload(res_acc_ub, c * VECTOR_CHUNK_ELEMENTS), denom_b, mask=full
                )
                rr.vstore_pack(
                    outb_ub,
                    c * VECTOR_CHUNK_ELEMENTS,
                    rr.vcast(
                        out, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ZERO
                    ),
                    full,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            rr.vmem_bar("vst_vld")

    @kernel
    def dual_cache_vector_attention_kernel(
        self,
        out_gm: Tensor,
        q_gm: Tensor,
        wkc_gm: Tensor,
        wvc_gm: Tensor,
        wbt_gm: Tensor,
        ckc_gm: Tensor,
        cvc_gm: Tensor,
        cbt_gm: Tensor,
        idx_gm: Tensor,
        sink_gm: Tensor,
        zero_gm: Tensor,
    ):
        _m, d, k, ps, bn = self.m, self.d, self.k, self.page_size, self.bn
        qb_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, d)))
        keyb_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((bn, d)))
        valb_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((bn, d)))
        q_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, d)))
        score_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((k,)))
        p_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((k,)))
        res_acc_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, d)))
        outb_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, d)))
        denom_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 1)))
        sink_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 1)))

        # One block per query head/row: the 64 heads are independent (shared
        # KV), so distribute them over the AIV cores instead of serializing.
        row = get_block_idx()
        mem_copy(qb_ub, tile_view(q_gm, (1, d), (row, 0)))
        mem_copy(sink_ub, tile_view(sink_gm, (1, 1), (0, row)))
        self._upcast_query(qb_ub, q_ub)
        self._initialize_masked_scores(score_ub)
        with vf(mode="simd"):
            _full = rr.full_mask()
            for _c in range(self.dc):
                rr.vstore(
                    res_acc_ub,
                    _c * VECTOR_CHUNK_ELEMENTS,
                    rr.vdups(0.0, dtypes.float32, mask=_full),
                    _full,
                )
            rr.vmem_bar("vst_vld")
        # Zero K/V rows once.  Invalid lanes are already masked (score -inf ->
        # p = 0), so a per-tile re-zero is unnecessary: valid rows get
        # overwritten on gather and invalid rows stay 0 / finite stale data.
        self._zero_bf16_kv_rows(keyb_ub, bn, zero_gm)
        self._zero_bf16_kv_rows(valb_ub, bn, zero_gm)
        # Pass 1: gather K tiles and score valid keys.
        for t in range(self.ntile):
            for j in range(bn):
                logical = dtypes.int32(idx_gm[row, t * bn + j])
                if logical >= 0:
                    if logical < self.win_len:
                        page = dtypes.int32(wbt_gm[dtypes.int64(logical // ps)])
                        phys = dtypes.int64(page * ps + dtypes.int32(logical % ps))
                        mem_copy(
                            tile_view(keyb_ub, (1, d), (j, 0)),
                            tile_view(wkc_gm, (1, d), (phys, 0)),
                        )
                    else:
                        cidx = logical - self.win_len
                        page = dtypes.int32(cbt_gm[dtypes.int64(cidx // ps)])
                        phys = dtypes.int64(page * ps + dtypes.int32(cidx % ps))
                        mem_copy(
                            tile_view(keyb_ub, (1, d), (j, 0)),
                            tile_view(ckc_gm, (1, d), (phys, 0)),
                        )
                    self._compute_scaled_query_key_dot(
                        keyb_ub, j, q_ub, score_ub, t * bn
                    )
        self._compute_sink_softmax_statistics(score_ub, sink_ub, p_ub, denom_ub)
        # Pass 2: re-gather V tiles and accumulate.
        for t in range(self.ntile):
            for j in range(bn):
                logical = dtypes.int32(idx_gm[row, t * bn + j])
                if logical >= 0:
                    if logical < self.win_len:
                        page = dtypes.int32(wbt_gm[dtypes.int64(logical // ps)])
                        phys = dtypes.int64(page * ps + dtypes.int32(logical % ps))
                        mem_copy(
                            tile_view(valb_ub, (1, d), (j, 0)),
                            tile_view(wvc_gm, (1, d), (phys, 0)),
                        )
                    else:
                        cidx = logical - self.win_len
                        page = dtypes.int32(cbt_gm[dtypes.int64(cidx // ps)])
                        phys = dtypes.int64(page * ps + dtypes.int32(cidx % ps))
                        mem_copy(
                            tile_view(valb_ub, (1, d), (j, 0)),
                            tile_view(cvc_gm, (1, d), (phys, 0)),
                        )
            self._accumulate_weighted_values(valb_ub, p_ub, t * bn, res_acc_ub)
        self._normalize_attention_output(res_acc_ub, denom_ub, outb_ub)
        mem_copy(tile_view(out_gm, (1, d), (row, 0)), outb_ub)
