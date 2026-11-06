# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Token-local QK, online sink Softmax and PV pipeline with paired AIVs.

The Cube and Vector engines share bounded rings. Online maxima and sums
rescale prior PV accumulators when later KV tiles extend the Softmax domain.
"""

from __future__ import annotations

from cannbotdsl import make_layout, MemLoc, RegLayout, dtypes, tile_slice as tile_view
from cannbotdsl.buffer import Buffer
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import mem_copy

if __package__:
    from .attention_geometry import HEAD_DIM
    from .dual_cache_vector_attention import NEG_INF, VECTOR_CHUNK_ELEMENTS
else:
    from attention_geometry import HEAD_DIM
    from dual_cache_vector_attention import NEG_INF, VECTOR_CHUNK_ELEMENTS

D = HEAD_DIM


class _SmallKAttentionMatmul:
    """Token-local M32/M64, N128 QK/PV engine with L0 double buffering."""

    def __init__(self, tile_m=64, tile_n=128, tile_d=D):
        self.tile_m, self.tile_n, self.tile_d = tile_m, tile_n, tile_d
        # Stream 128-wide slices from resident L1.  QK and PV now have the
        # same B tile [128,128], so one 32-KiB-slot L0B channel can be shared;
        # depth=2 consumes the full 64-KiB L0B budget.
        self.l0_k = 128
        self.pv_n = 128
        self.q_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((tile_m, tile_d)), depth=1
        )
        # QK writes and reads a slot; delayed PV is its last consumer.
        # Three slots cover the two-tick QK -> PV lag plus current producer.
        self.kv_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((tile_n, tile_d)), depth=3
        )
        self.qk_l0a = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((tile_m, self.l0_k)), depth=2
        )
        self.l0b = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((tile_n, self.l0_k)), depth=2
        )
        self.qk_l0c = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((tile_m, tile_n)), depth=2
        )
        self.pv_l0a = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((tile_m, tile_n)), depth=2
        )
        self.pv_l0c = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((tile_m, self.pv_n)), depth=2
        )

    def load_query(self, q_gm):
        mem_copy(self.q_l1.next_write(), q_gm)
        return self.q_l1.next_read()

    def compute_query_key(self, k_gm, qk_ub, q_slot):
        k_slot = self.kv_l1.next_write()
        mem_copy(k_slot, k_gm)
        acc = self.qk_l0c.next_write()
        for ki in range(self.tile_d // self.l0_k):
            q_l1_slice = tile_view(q_slot, (self.tile_m, self.l0_k), (0, ki))
            k_l1_slice = tile_view(k_slot, (self.tile_n, self.l0_k), (0, ki))
            mem_copy(self.qk_l0a.next_write(), q_l1_slice)
            mem_copy(self.l0b.next_write(), k_l1_slice)
            matmul(acc, self.qk_l0a.next_read(), self.l0b.next_read(), init=(ki == 0))
        mem_copy(
            qk_ub,
            (self.qk_l0c.next_read())[: self.tile_m, : self.tile_n].partition(
                dim=0, alignment=16
            ),
        )

    def compute_probability_value(self, p_l1, pv_ub):
        v_slot = self.kv_l1.next_read()
        mem_copy(self.pv_l0a.next_write(), p_l1)
        p_slot = self.pv_l0a.next_read()
        for di in range(self.tile_d // self.pv_n):
            v_l1_slice = tile_view(v_slot, (self.tile_n, self.pv_n), (0, di))
            mem_copy(self.l0b.next_write(), (v_l1_slice).permute((1, 0)))
            acc = self.pv_l0c.next_write()
            matmul(acc, p_slot, self.l0b.next_read(), init=True)
            mem_copy(
                pv_ub.next_write(),
                (self.pv_l0c.next_read())[: self.tile_m, : self.pv_n].partition(
                    dim=0, alignment=16
                ),
            )


@jit
def _mask_index_prefix(idx_ub, width, valid_length):
    """只在 UB 将有效前缀之外置 -1；不改写调用方的索引。"""
    with vf(mode="simd"):
        full = rr.full_mask()
        invalid = rr.vdups(-1, dtypes.int32, mask=full)
        for col in range(0, width, 64):
            remaining = dtypes.int32(valid_length - col)
            if remaining < 0:
                remaining = dtypes.int32(0)
            prefix, _ = rr.update_mask(remaining, elem_bits=32)
            index = rr.vload(idx_ub, col)
            rr.vstore(idx_ub, col, rr.vselect(index, invalid, cond_mask=prefix), full)
        rr.vmem_bar("vst_vld")


class _SmallKAttentionVector:
    """Two-AIV half-M online softmax and online output accumulation."""

    def __init__(self, tile_vec_m=32, tile_n=128, tile_d=D):
        self.tile_vec_m = tile_vec_m
        self.tile_m = tile_vec_m * 2
        self.tile_n, self.tile_d = tile_n, tile_d
        self.subblock = get_subblock_id()
        self.sm_max = Buffer(MemLoc.UB, dtypes.float32, make_layout((tile_vec_m, 1)))
        self.sm_sum = Buffer(MemLoc.UB, dtypes.float32, make_layout((tile_vec_m, 1)))
        self.sm_exp = [
            Buffer(MemLoc.UB, dtypes.float32, make_layout((tile_vec_m, 1)))
            for _ in range(3)
        ]
        self.tmp_max = Buffer(MemLoc.UB, dtypes.float32, make_layout((tile_vec_m, 1)))
        self.tmp_sum = Buffer(MemLoc.UB, dtypes.float32, make_layout((tile_vec_m, 1)))
        self.res_o = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((tile_vec_m, tile_d))
        )
        self.p_ub = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((tile_vec_m, tile_n)),
            depth=2,
            data_format="nz",
            physical_layout=make_layout(
                ((tile_n + 15) // 16, (tile_vec_m + 15) // 16, 16, 16),
                stride=(((tile_vec_m + 15) // 16) * 256 + 16, 256, 16, 1),
            ),
        )
        self.o_ub = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((tile_vec_m, tile_d))
        )
        self.idx_ub = RingBuffer(
            MemLoc.UB, dtypes.int32, make_layout((1, tile_n)), depth=2
        )
        self.sink_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, tile_vec_m)))

    @jit
    def load_sink(self, attn_sink_gm, head0):
        if self.subblock == 0:
            mem_copy(self.sink_ub, attn_sink_gm[0:1, head0 : head0 + self.tile_vec_m])
        else:
            mem_copy(
                self.sink_ub,
                attn_sink_gm[0:1, head0 + self.tile_vec_m : head0 + self.tile_m],
            )

    @jit
    def _mask_invalid_kv_positions(self, qk_ub, idx_ub):
        with vf(mode="simd"):
            stride = qk_ub.physical_stride[0]
            for col in range(0, self.tile_n, 64):
                lanes, _ = rr.update_mask(64, elem_bits=32)
                idx = rr.vload(idx_ub, col)
                zero = rr.vdups(0, dtypes.int32, mask=lanes)
                invalid = rr.vlt(idx, zero, mask=lanes)
                neg = rr.vdups(NEG_INF, dtypes.float32, mask=lanes)
                for row in range(self.tile_vec_m):
                    off = row * stride + col
                    s = rr.vload(qk_ub, off)
                    rr.vstore(qk_ub, off, rr.vselect(neg, s, cond_mask=invalid), lanes)
            rr.vmem_bar("vst_vld")

    def _probability_nz_strides(self):
        # A step along N1 crosses all M1 blocks plus one padded 16-element
        # block: 33 for Mv=32 and 17 for the T16 Mv=16 specialization.
        m1 = self.tile_vec_m // 16
        return 16, 256, 16, m1 * 16 + 1

    def _pack_exponential_probability_row(
        self,
        qk_ub,
        p_ub,
        base,
        nz_off,
        max_buf,
        row,
        half_mask,
        b16_mask,
        full,
        block_stride,
    ):
        mx = rr.vload_broadcast(max_buf, row)
        ve, vo = rr.vload_deinterleave(qk_ub, base, width="b32")
        ve = rr.vexp_sub(ve, mx, mask=half_mask)
        vo = rr.vexp_sub(vo, mx, mask=half_mask)
        he = rr.vcast(ve, dtypes.bfloat16, mask=half_mask, reg_layout=RegLayout.ZERO)
        ho = rr.vcast(vo, dtypes.bfloat16, mask=half_mask, reg_layout=RegLayout.ONE)
        merged = rr.vbitwise_or(he, ho, mask=b16_mask)
        rr.vstore_strided(
            p_ub, nz_off, merged, b16_mask, block_stride=block_stride, repeat_stride=0
        )
        return ve, vo

    @jit
    def initialize_softmax_state(self, qk_ub, idx_ub, p_ub):
        self._mask_invalid_kv_positions(qk_ub, idx_ub)
        with vf(mode="simd"):
            lanes, _ = rr.update_mask(64, elem_bits=32)
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            stride = qk_ub.physical_stride[0]
            for row in range(self.tile_vec_m):
                base = row * stride
                val0 = rr.vload(qk_ub, base)
                val1 = rr.vload(qk_ub, base + 64)
                rr.vstore_first(
                    self.sm_max,
                    row,
                    rr.vreduce_max(rr.vmax(val0, val1, mask=lanes), mask=lanes),
                )
            rr.vmem_bar("vst_vld")
            old = rr.vload(self.sm_max, 0)
            attn_sink = rr.vload(self.sink_ub, 0)
            rr.vstore(self.sm_max, 0, rr.vmax(old, attn_sink, mask=rowmask), rowmask)
            rr.vmem_bar("vst_vld")
            m0, s_m1, s_m0, block_stride = self._probability_nz_strides()
            half, _ = rr.update_mask((self.tile_n + 1) // 2, elem_bits=32)
            b16, _ = rr.update_mask(self.tile_n, elem_bits=16)
            full, _ = rr.update_mask(64, elem_bits=32)
            for row in range(self.tile_vec_m):
                base = row * stride
                nz_off = (row // m0) * s_m1 + (row % m0) * s_m0
                ve, vo = self._pack_exponential_probability_row(
                    qk_ub,
                    p_ub,
                    base,
                    nz_off,
                    self.sm_max,
                    row,
                    half,
                    b16,
                    full,
                    block_stride,
                )
                rsum = rr.vreduce_sum(rr.vadd(ve, vo, mask=full), mask=half)
                rr.vstore_first(self.sm_sum, row, rsum)
            rr.vmem_bar("vst_vld")
            run_m = rr.vload(self.sm_max, 0)
            run_l = rr.vload(self.sm_sum, 0)
            sink_l = rr.vexp_sub(attn_sink, run_m, mask=rowmask)
            rr.vstore(self.sm_sum, 0, rr.vadd(run_l, sink_l, mask=rowmask), rowmask)
            rr.vmem_bar("vst_vld")

    @jit
    def update_softmax_state(self, qk_ub, idx_ub, p_ub, slot):
        self._mask_invalid_kv_positions(qk_ub, idx_ub)
        sm_exp = self.sm_exp[slot]
        with vf(mode="simd"):
            lanes, _ = rr.update_mask(64, elem_bits=32)
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            stride = qk_ub.physical_stride[0]
            for row in range(self.tile_vec_m):
                base = row * stride
                val0 = rr.vload(qk_ub, base)
                val1 = rr.vload(qk_ub, base + 64)
                rr.vstore_first(
                    self.tmp_max,
                    row,
                    rr.vreduce_max(rr.vmax(val0, val1, mask=lanes), mask=lanes),
                )
            rr.vmem_bar("vst_vld")
            old_m = rr.vload(self.sm_max, 0)
            new_m = rr.vmax(old_m, rr.vload(self.tmp_max, 0), mask=rowmask)
            rr.vstore(self.tmp_max, 0, new_m, rowmask)
            rr.vmem_bar("vst_vld")
            m0, s_m1, s_m0, block_stride = self._probability_nz_strides()
            half, _ = rr.update_mask((self.tile_n + 1) // 2, elem_bits=32)
            b16, _ = rr.update_mask(self.tile_n, elem_bits=16)
            full, _ = rr.update_mask(64, elem_bits=32)
            for row in range(self.tile_vec_m):
                base = row * stride
                nz_off = (row // m0) * s_m1 + (row % m0) * s_m0
                ve, vo = self._pack_exponential_probability_row(
                    qk_ub,
                    p_ub,
                    base,
                    nz_off,
                    self.tmp_max,
                    row,
                    half,
                    b16,
                    full,
                    block_stride,
                )
                rsum = rr.vreduce_sum(rr.vadd(ve, vo, mask=full), mask=half)
                rr.vstore_first(self.tmp_sum, row, rsum)
            rr.vmem_bar("vst_vld")
            new_m = rr.vload(self.tmp_max, 0)
            alpha = rr.vexp_sub(old_m, new_m, mask=rowmask)
            rr.vstore(sm_exp, 0, alpha, rowmask)
            rr.vstore(self.sm_max, 0, new_m, rowmask)
            old_l = rr.vload(self.sm_sum, 0)
            tile_l = rr.vload(self.tmp_sum, 0)
            rr.vstore(
                self.sm_sum, 0, rr.vmadd(old_l, alpha, tile_l, mask=rowmask), rowmask
            )
            rr.vmem_bar("vst_vld")

    def store_probability_tile(self, p_l1):
        mem_copy(
            (p_l1)[: self.tile_m, : self.tile_n].partition(
                dim=0, alignment=16, part_id=self.subblock
            ),
            self.p_ub.next_read(),
        )

    def initialize_output_chunk(self, pv_ub, d_idx):
        dst = tile_view(self.res_o, (self.tile_vec_m, 128), (0, d_idx))
        mem_copy(dst, pv_ub)

    @jit
    def accumulate_output_chunk(self, pv_ub, slot, d_idx):
        sm_exp = self.sm_exp[slot]
        with vf(mode="simd"):
            for row in range(self.tile_vec_m):
                alpha = rr.vload_broadcast(sm_exp, row)
                res_base = row * self.tile_d + d_idx * 128
                pv_base = row * 128
                for col in range(0, 128, VECTOR_CHUNK_ELEMENTS):
                    mask, _ = rr.update_mask(128 - col, elem_bits=32)
                    val = rr.vmadd(
                        rr.vload(self.res_o, res_base + col),
                        alpha,
                        rr.vload(pv_ub, pv_base + col),
                        mask=mask,
                    )
                    rr.vstore(self.res_o, res_base + col, val, mask)
            rr.vmem_bar("vst_vld")

    @jit
    def normalize_final_output_chunk(self, pv_ub, slot, d_idx):
        sm_exp = self.sm_exp[slot]
        with vf(mode="simd"):
            for row in range(self.tile_vec_m):
                alpha = rr.vload_broadcast(sm_exp, row)
                denom = rr.vload_broadcast(self.sm_sum, row)
                res_base = row * self.tile_d + d_idx * 128
                pv_base = row * 128
                for col in range(0, 128, VECTOR_CHUNK_ELEMENTS):
                    mask, _ = rr.update_mask(128 - col, elem_bits=32)
                    val = rr.vmadd(
                        rr.vload(self.res_o, res_base + col),
                        alpha,
                        rr.vload(pv_ub, pv_base + col),
                        mask=mask,
                    )
                    rr.vstore(
                        self.res_o, res_base + col, rr.vdiv(val, denom, mask=mask), mask
                    )
            rr.vmem_bar("vst_vld")

    @jit
    def store_attention_output(self, out_tile):
        with vf(mode="simd"):
            full = rr.full_mask()
            for row in range(self.tile_vec_m):
                base = row * self.tile_d
                for col in range(0, self.tile_d, VECTOR_CHUNK_ELEMENTS):
                    off = base + col
                    val = rr.vcast(
                        rr.vload(self.res_o, off),
                        dtypes.bfloat16,
                        mask=full,
                        reg_layout=RegLayout.ZERO,
                    )
                    rr.vstore_pack(
                        self.o_ub, off, val, full, pack_mode=rr.PackMode.B32_TO_B16
                    )
            rr.vmem_bar("vst_vld")
        if self.subblock == 0:
            mem_copy(out_tile[0 : self.tile_vec_m, 0 : self.tile_d], self.o_ub)
        else:
            mem_copy(
                out_tile[self.tile_vec_m : self.tile_m, 0 : self.tile_d], self.o_ub
            )


class _SmallKAttentionPipeline:
    """Per-token M32/M64 head-tile QK -> online softmax -> PV pipeline."""

    def __init__(self, k, tile_m=64):
        self.k = k
        self.tile_m, self.tile_vm = tile_m, tile_m // 2
        self.tile_n, self.tile_d = 128, D
        self.mm = _SmallKAttentionMatmul(self.tile_m, self.tile_n, self.tile_d)
        self.vec = _SmallKAttentionVector(self.tile_vm, self.tile_n, self.tile_d)
        self.qk_ub = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((self.tile_vm, self.tile_n)), depth=2
        )
        self.pv_ub = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((self.tile_vm, self.mm.pv_n)),
            depth=self.tile_d // self.mm.pv_n,
        )
        self.p_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((self.tile_m, self.tile_n)), depth=2
        )

    @jit
    def run_attention_head_tile(
        self,
        q_tile,
        kv_tile,
        win_idx_tile,
        cmp_idx_tile,
        win_len,
        win_valid,
        cmp_valid,
        use_lengths,
        attn_sink_gm,
        out_tile,
        head0,
    ):
        q_slot = self.mm.load_query(q_tile)
        self.vec.load_sink(attn_sink_gm, head0)
        tiles = self.k // self.tile_n
        for tick in range(tiles + 3):
            if tick < tiles:
                n0 = tick * self.tile_n
                self.mm.compute_query_key(
                    kv_tile[n0 : n0 + self.tile_n, 0 : self.tile_d],
                    self.qk_ub.next_write(),
                    q_slot,
                )
            if tick >= 1 and tick - 1 < tiles:
                nt = tick - 1
                n0 = nt * self.tile_n
                idx_write = self.vec.idx_ub.next_write()
                valid_length = dtypes.int32(win_valid - n0)
                if n0 < win_len:
                    mem_copy(idx_write, win_idx_tile[0:1, n0 : n0 + self.tile_n])
                else:
                    cmp_n0 = n0 - win_len
                    mem_copy(
                        idx_write, cmp_idx_tile[0:1, cmp_n0 : cmp_n0 + self.tile_n]
                    )
                    valid_length = dtypes.int32(cmp_valid - cmp_n0)
                idx_read = self.vec.idx_ub.next_read()
                if use_lengths:
                    _mask_index_prefix(idx_read, self.tile_n, valid_length)
                qk_read = self.qk_ub.next_read()
                p_write = self.vec.p_ub.next_write()
                if nt == 0:
                    self.vec.initialize_softmax_state(qk_read, idx_read, p_write)
                else:
                    self.vec.update_softmax_state(qk_read, idx_read, p_write, nt % 3)
                self.vec.store_probability_tile(self.p_l1.next_write())
            if tick >= 2 and tick - 2 < tiles:
                self.mm.compute_probability_value(self.p_l1.next_read(), self.pv_ub)
            if tick >= 3 and tick - 3 < tiles:
                nt = tick - 3
                for di in range(self.tile_d // self.mm.pv_n):
                    pv_read = self.pv_ub.next_read()
                    if nt == 0:
                        self.vec.initialize_output_chunk(pv_read, di)
                    elif nt + 1 == tiles:
                        self.vec.normalize_final_output_chunk(pv_read, nt % 3, di)
                    else:
                        self.vec.accumulate_output_chunk(pv_read, nt % 3, di)
        self.vec.store_attention_output(out_tile)
