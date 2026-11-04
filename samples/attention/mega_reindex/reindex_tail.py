# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Candidate/PA preparation, fused score and stable TopK phases."""

import cannbotdsl as cb
from cannbotdsl import dtypes as dt
from cannbotdsl import simt
from cannbotdsl.ops import reg as rr
from cannbotdsl.tensor import make_layout, make_pointer, make_tensor

from .reindex_runtime import (
    BF16_NEG_INF_BITS,
    CANDIDATE_BLOCK_SIZE,
    CANDIDATE_CAPACITY,
    INT32_BYTES,
    METADATA_UB_FRACTION,
    PA_KEY_BLOCK_BYTES,
    PA_RECORD_BYTES,
    PA_SCALE_BLOCK_BYTES,
    SCORE_TILE_TOKENS,
    SIMT_THREADS,
    VECTOR_SUBBLOCKS,
    Buffer,
    RingBuffer,
)


class CandidatePhase:
    def __init__(self, rows, batch, length, cube_cores):
        self.vector_blocks = cube_cores * VECTOR_SUBBLOCKS
        self.rows, self.batch, self.length = rows, batch, length
        self.seq = rows // batch
        self.padded = (
            (length + SCORE_TILE_TOKENS - 1)
            // SCORE_TILE_TOKENS
            * (SCORE_TILE_TOKENS // CANDIDATE_BLOCK_SIZE)
        )
        # Keep the original fast path when all metadata fits a small UB share.
        # Larger batches use fixed-size row-local loads; no batch limit is added.
        self.bulk_metadata = (
            2 * (batch + 1) + 2 * batch
        ) * INT32_BYTES <= cb.get_mem_size("ub") // METADATA_UB_FRACTION
        self.boundary_count = batch + 1 if self.bulk_metadata else 2
        self.length_count = batch if self.bulk_metadata else 1

    @cb.jit
    def phase(
        self,
        candidates: cb.Tensor,
        counts: cb.Tensor,
        cuq: cb.Tensor,
        cuk: cb.Tensor,
        usedq: cb.Tensor,
        usedk: cb.Tensor,
        mask: cb.Tensor,
        visible: cb.Tensor,
    ):
        candidate_ch = RingBuffer(
            cb.MemLoc.UB, (CANDIDATE_CAPACITY,), dt.int32, depth=1
        )
        count_ch = RingBuffer(cb.MemLoc.UB, (1,), dt.int32, depth=1)
        mask_ch = RingBuffer(cb.MemLoc.UB, (self.padded,), dt.uint8, depth=1)
        visible_ch = RingBuffer(cb.MemLoc.UB, (1,), dt.int32, depth=1)
        cuq_ub = Buffer(cb.MemLoc.UB, (self.boundary_count,), dt.int32)
        cuk_ub = Buffer(cb.MemLoc.UB, (self.boundary_count,), dt.int32)
        usedq_ub = Buffer(cb.MemLoc.UB, (self.length_count,), dt.int32)
        usedk_ub = Buffer(cb.MemLoc.UB, (self.length_count,), dt.int32)
        if cb.const_expr(self.bulk_metadata):
            cb.mem_copy(cuq_ub, cuq)
            cb.mem_copy(cuk_ub, cuk)
            cb.mem_copy(usedq_ub, usedq)
            cb.mem_copy(usedk_ub, usedk)
        for row in range(
            cb.get_block_idx() * VECTOR_SUBBLOCKS + cb.get_subblock_id(),
            self.rows,
            self.vector_blocks,
        ):
            if cb.const_expr(not self.bulk_metadata):
                batch_index = row // self.seq
                cb.mem_copy(cuq_ub, cuq.flatten()[batch_index : batch_index + 2,])
                cb.mem_copy(cuk_ub, cuk.flatten()[batch_index : batch_index + 2,])
                cb.mem_copy(usedq_ub, usedq.flatten()[batch_index : batch_index + 1,])
                cb.mem_copy(usedk_ub, usedk.flatten()[batch_index : batch_index + 1,])
            cb.mem_copy(
                candidate_ch.next_write(),
                cb.tile_slice(candidates.flatten(), (CANDIDATE_CAPACITY,), (row,)),
            )
            cb.mem_copy(
                count_ch.next_write(), cb.tile_slice(counts.flatten(), (1,), (row,))
            )
            row_candidates = candidate_ch.next_read()
            row_count = count_ch.next_read()
            row_mask = mask_ch.next_write()
            row_visible = visible_ch.next_write()
            with cb.vf(mode="simt", thread=SIMT_THREADS):
                tid = dt.int32(simt.thread_idx()[0])
                batch = dt.int32(row // self.seq)
                local = dt.int32(row % self.seq)
                if cb.const_expr(self.bulk_metadata):
                    metadata_batch = batch
                else:
                    metadata_batch = dt.int32(0)
                # tid//256 is zero for this 256-thread VF.
                for pos in range(tid, self.padded, SIMT_THREADS):
                    row_mask[pos] = dt.uint8(tid // SIMT_THREADS)
                if tid == 0:
                    limit = dt.int32(0)
                    if (
                        (cuq_ub[metadata_batch] == batch * self.seq)
                        and (cuq_ub[metadata_batch + 1] == (batch + 1) * self.seq)
                        and (cuk_ub[metadata_batch] == batch * self.length)
                        and (cuk_ub[metadata_batch + 1] == (batch + 1) * self.length)
                        and (usedq_ub[metadata_batch] >= 0)
                        and (usedq_ub[metadata_batch] <= self.seq)
                        and (usedk_ub[metadata_batch] >= 0)
                        and (usedk_ub[metadata_batch] <= self.length)
                        and (local < usedq_ub[metadata_batch])
                    ):
                        limit = (
                            usedk_ub[metadata_batch]
                            - usedq_ub[metadata_batch]
                            + local
                            + 1
                        )
                        if limit < 0:
                            limit = dt.int32(0)
                    row_visible[0] = limit
            # One byte per eight-token block. Separate VF orders clear before scatter.
            with cb.vf(mode="simt", thread=SIMT_THREADS):
                tid = dt.int32(simt.thread_idx()[0])
                count = row_count[0]
                if (count >= 0) and (count <= CANDIDATE_CAPACITY):
                    for ci in range(tid, count, SIMT_THREADS):
                        block = row_candidates[ci]
                        if (block >= 0) and (
                            block
                            < (self.length + CANDIDATE_BLOCK_SIZE - 1)
                            // CANDIDATE_BLOCK_SIZE
                        ):
                            row_mask[block] = dt.uint8(tid // SIMT_THREADS + 1)
            cb.mem_copy(
                cb.tile_slice(mask.flatten(), (self.padded,), (row,)),
                mask_ch.next_read(),
            )
            cb.mem_copy(cb.tile_slice(visible, (1,), (row,)), visible_ch.next_read())


class PaReadPhase:
    def __init__(self, batch, length, page_size, cube_cores):
        self.vector_blocks = cube_cores * VECTOR_SUBBLOCKS
        self.batch = batch
        self.page_size = page_size
        self.pages = length // page_size
        self.blocks = page_size // CANDIDATE_BLOCK_SIZE

    @cb.jit
    def phase(
        self,
        packed: cb.Tensor,
        table: cb.Tensor,
        usedk: cb.Tensor,
        outk: cb.Tensor,
        outs: cb.Tensor,
    ):
        page = Buffer(cb.MemLoc.UB, (self.blocks, PA_RECORD_BYTES), dt.uint8)
        keys = outk.view((self.batch * self.pages, self.blocks, PA_KEY_BLOCK_BYTES))
        scales = outs.view((self.batch * self.pages, self.blocks, PA_SCALE_BLOCK_BYTES))
        for task in range(
            cb.get_block_idx() * VECTOR_SUBBLOCKS + cb.get_subblock_id(),
            self.batch * self.pages,
            self.vector_blocks,
        ):
            batch = task // self.pages
            logical = task % self.pages
            if logical * self.page_size < usedk[batch]:
                physical = dt.int64(table[batch, logical])
                cb.mem_copy(
                    page,
                    packed[
                        physical : physical + 1, 0 : self.blocks, 0:PA_RECORD_BYTES
                    ].view((self.blocks, PA_RECORD_BYTES)),
                )
            else:
                with cb.vf(mode="simt", thread=SIMT_THREADS):
                    tid = dt.int32(simt.thread_idx()[0])
                    for pos in range(tid, self.blocks * PA_RECORD_BYTES, SIMT_THREADS):
                        page[pos // PA_RECORD_BYTES, pos % PA_RECORD_BYTES] = dt.uint8(
                            tid // SIMT_THREADS
                        )
            cb.mem_copy(
                keys[task : task + 1, 0 : self.blocks, 0:PA_KEY_BLOCK_BYTES].view(
                    (self.blocks, PA_KEY_BLOCK_BYTES)
                ),
                page[0 : self.blocks, 0:PA_KEY_BLOCK_BYTES],
            )
            cb.mem_copy(
                scales[task : task + 1, 0 : self.blocks, 0:PA_SCALE_BLOCK_BYTES].view(
                    (self.blocks, PA_SCALE_BLOCK_BYTES)
                ),
                page[0 : self.blocks, PA_KEY_BLOCK_BYTES:PA_RECORD_BYTES],
            )


class TailScorePhase:
    """Consume repository-native packed MXFP4 Q/K and E8M0 group scales."""

    def __init__(self, rows, batch, length, cube_cores):
        self.core_count = cube_cores
        self.n = SCORE_TILE_TOKENS
        self.rows = rows
        self.batch = batch
        self.seq = rows // batch
        self.length = length
        self.ntiles = (length + self.n - 1) // self.n

    @cb.jit
    def phase(
        self,
        q_address,
        qs: cb.Tensor,
        k_address,
        ks: cb.Tensor,
        w: cb.Tensor,
        mask: cb.Tensor,
        visible: cb.Tensor,
        out: cb.Tensor,
    ):
        q = make_tensor(
            make_pointer(dt.fp4x2_e2m1, dt.int64(q_address), cb.MemLoc.GM),
            make_layout((self.rows * 32, 128), stride=(128, 1)),
        )
        k = make_tensor(
            make_pointer(dt.fp4x2_e2m1, dt.int64(k_address), cb.MemLoc.GM),
            make_layout((self.batch * self.length, 128), stride=(128, 1)),
        )
        qs = qs.view(dtype=dt.float8_e8m0)
        ks = ks.view(dtype=dt.float8_e8m0)

        q_l1 = RingBuffer(
            cb.MemLoc.L1, (64, 128), dt.fp4x2_e2m1, depth=1, data_format="nz"
        )
        k_l1 = RingBuffer(
            cb.MemLoc.L1, (self.n, 128), dt.fp4x2_e2m1, depth=2, data_format="nz"
        )
        q_scale_l1 = Buffer(cb.MemLoc.L1, (64, 4), dt.float8_e8m0, data_format="zn")
        k_scale_l1 = RingBuffer(
            cb.MemLoc.L1, (4, self.n), dt.float8_e8m0, depth=2, data_format="nz"
        )
        l0a = RingBuffer(cb.MemLoc.L0A, (64, 128), dt.fp4x2_e2m1, depth=1)
        l0b = RingBuffer(
            cb.MemLoc.L0B, (self.n, 128), dt.fp4x2_e2m1, depth=2, data_format="nz"
        )
        l0c = RingBuffer(cb.MemLoc.L0C, (64, self.n), dt.float32, depth=2)
        qk_handoff = RingBuffer(cb.MemLoc.UB, (32, self.n), dt.bfloat16, depth=2)
        weights_ch = RingBuffer(cb.MemLoc.UB, (1, 32), dt.bfloat16, depth=1)
        mask_ch = RingBuffer(cb.MemLoc.UB, (1, self.n // 8), dt.uint8, depth=2)
        out_ch = RingBuffer(cb.MemLoc.UB, (1, self.n), dt.bfloat16, depth=2)
        visible_ch = RingBuffer(cb.MemLoc.UB, (8,), dt.int32, depth=1)
        neg_inf = Buffer(cb.MemLoc.UB, (1,), dt.bfloat16)

        block = cb.get_block_idx()
        sub = cb.get_subblock_id()
        # One 64-row MX tile combines two tokens.  The two Vector subcores each
        # consume one token's 32 heads, matching the original paired-token
        # schedule while halving K loads and Cube tasks.
        for task in range(block, (self.rows // 2) * 8, self.core_count):
            qg = task // 8
            part = task % 8
            row = qg * 2 + sub
            cb.mem_copy(q_l1.next_write(), cb.tile_slice(q, (64, 128), (qg, 0)))
            cb.mem_copy(
                q_scale_l1, cb.tile_slice(qs, (64, 2, 2), (qg, 0, 0)).flatten(-2, -1)
            )
            cb.mem_copy(
                l0a.next_write(),
                q_l1.next_read(),
                options=cb.copy_options(mx_scale=q_scale_l1),
            )
            a = l0a.next_read()
            cb.mem_copy(weights_ch.next_write(), cb.tile_slice(w, (1, 32), (row, 0)))
            weights = weights_ch.next_read()
            cb.mem_copy(visible_ch.next_write(), visible[row : row + 1,])
            limit = visible_ch.next_read()
            batch = row // self.seq
            for nt in range(part, self.ntiles, 8):
                begin = batch * self.length + nt * self.n
                end = batch * self.length + min((nt + 1) * self.n, self.length)
                cb.mem_copy(k_l1.next_write(), k[begin:end, 0:128])
                cb.mem_copy(
                    k_scale_l1.next_write(),
                    ks[begin:end, 0:2, 0:2].flatten(-2, -1).permute(1, 0),
                )
                cb.mem_copy(
                    l0b.next_write(),
                    k_l1.next_read(),
                    options=cb.copy_options(mx_scale=k_scale_l1.next_read()),
                )
                c = l0c.next_write()
                cb.matmul(c, a, l0b.next_read(), init=True, mx_mode=True)
                cv_write = qk_handoff.next_write()
                cb.mem_copy(
                    cv_write,
                    cb.tile_slice(c, (32, self.n), (0, 0)),
                    options=cb.copy_options(dst_subblock=0),
                )
                cb.mem_copy(
                    cv_write,
                    cb.tile_slice(c, (32, self.n), (1, 0)),
                    options=cb.copy_options(dst_subblock=1),
                )
                scores = qk_handoff.next_read()
                cb.mem_copy(
                    mask_ch.next_write(),
                    cb.tile_slice(mask, (1, self.n // 8), (row, nt)),
                )
                validmask = mask_ch.next_read()
                dest = out_ch.next_write()
                with cb.vf(mode="simd"):
                    m16 = rr.update_mask(128, elem_bits=16)[0]
                    zero = rr.vdups(0.0, dt.bfloat16, mask=m16)
                    rr.vstore(
                        neg_inf,
                        0,
                        rr.vreinterpret(
                            rr.vdups(BF16_NEG_INF_BITS, dt.uint16, mask=m16),
                            dt.bfloat16,
                        ),
                        rr.update_mask(1, elem_bits=16)[0],
                    )
                    for vecpart in range(self.n // 128):
                        acc = zero
                        for h in range(32):
                            z = rr.vload(scores, h * self.n + vecpart * 128)
                            wt = rr.vload_broadcast(weights, h)
                            z = rr.vmax(z, zero, mask=m16)
                            acc = rr.vmadd(z, wt, acc, mask=m16)
                        rr.vstore(dest, vecpart * 128, acc, m16)
                    rr.vmem_bar("vst_vld")
                with cb.vf(mode="simt", thread=self.n):
                    tid = dt.int32(simt.thread_idx()[0])
                    pos = dt.int32(nt * self.n) + tid
                    if tid < self.n:
                        if pos >= limit[0]:
                            dest[0, tid] = neg_inf[0]
                        if pos >= self.length:
                            dest[0, tid] = neg_inf[0]
                        if dt.int32(validmask[0, tid // 8]) == 0:
                            dest[0, tid] = neg_inf[0]
                cb.mem_copy(
                    cb.tile_slice(out, (1, self.n), (row, nt)), out_ch.next_read()
                )


BF16_SIGN_BIT = 1 << 15
UINT16_MAX = (1 << 16) - 1
BF16_MAGNITUDE_MASK = BF16_SIGN_BIT - 1


class TailTopkPhase:
    def __init__(self, rows, length, topk, cube_cores):
        self.vector_blocks = cube_cores * VECTOR_SUBBLOCKS
        self.rows = rows
        self.length = length
        self.topk = topk

    @cb.jit
    def phase(self, x: cb.Tensor, out: cb.Tensor):
        ch_x = RingBuffer(cb.MemLoc.UB, (self.length,), dt.bfloat16, depth=1)
        ties = Buffer(cb.MemLoc.UB, (self.length,), dt.uint16)
        countbuf = Buffer(cb.MemLoc.UB, (64,), dt.uint32)
        meta = Buffer(cb.MemLoc.UB, (128,), dt.uint16)
        ch_keys = Buffer(cb.MemLoc.UB, (self.length,), dt.uint16)
        ch_indices = Buffer(cb.MemLoc.UB, (self.topk,), dt.uint16)
        ch_output = RingBuffer(cb.MemLoc.UB, (self.topk,), dt.int32, depth=1)
        for row in range(
            cb.get_block_idx() * VECTOR_SUBBLOCKS + cb.get_subblock_id(),
            self.rows,
            self.vector_blocks,
        ):
            cb.mem_copy(ch_x.next_write(), cb.tile_slice(x, (self.length,), (row,)))
            xx = ch_x.next_read()
            keys = ch_keys
            indices = ch_indices
            outub = ch_output.next_write()
            with cb.vf(mode="simd"):
                m16 = rr.full_mask(elem_bits=16)
                m8 = rr.full_mask(elem_bits=8)
                zero = rr.vdups(0, dt.uint16, mask=m16)
                sign = rr.vdups(BF16_SIGN_BIT, dt.uint16, mask=m16)
                full = rr.vdups(UINT16_MAX, dt.uint16, mask=m16)
                mag = rr.vdups(BF16_MAGNITUDE_MASK, dt.uint16, mask=m16)
                high = zero
                threshold = zero
                for digit in cb.range_constexpr(2):
                    c0 = zero
                    c1 = zero
                    for part in range(self.length // 256):
                        if cb.const_expr(digit == 0):
                            ra = rr.vreinterpret_lanes(
                                rr.vload(xx, part * 256), dt.uint16
                            )
                            rb = rr.vreinterpret_lanes(
                                rr.vload(xx, part * 256 + 128), dt.uint16
                            )
                            a = rr.vselect(
                                rr.vbitwise_xor(ra, sign, mask=m16),
                                rr.vsub(full, ra, mask=m16),
                                cond_mask=rr.vlt(ra, sign, mask=m16),
                            )
                            b = rr.vselect(
                                rr.vbitwise_xor(rb, sign, mask=m16),
                                rr.vsub(full, rb, mask=m16),
                                cond_mask=rr.vlt(rb, sign, mask=m16),
                            )
                            a = rr.vselect(
                                sign,
                                a,
                                cond_mask=rr.veqs(
                                    rr.vbitwise_and(ra, mag, mask=m16), 0, mask=m16
                                ),
                            )
                            b = rr.vselect(
                                sign,
                                b,
                                cond_mask=rr.veqs(
                                    rr.vbitwise_and(rb, mag, mask=m16), 0, mask=m16
                                ),
                            )
                            rr.vstore(keys, part * 256, a, m16)
                            rr.vstore(keys, part * 256 + 128, b, m16)
                            lo, hi = rr.vdeinterleave(
                                rr.vreinterpret_lanes(a, dt.uint8),
                                rr.vreinterpret_lanes(b, dt.uint8),
                            )
                            vals = hi
                        else:
                            a = rr.vload(keys, part * 256)
                            b = rr.vload(keys, part * 256 + 128)
                            ah = rr.vshr(a, 8, mask=m16)
                            bh = rr.vshr(b, 8, mask=m16)
                            a = rr.vselect(
                                zero, a, cond_mask=rr.vlt(ah, high, mask=m16)
                            )
                            b = rr.vselect(
                                zero, b, cond_mask=rr.vlt(bh, high, mask=m16)
                            )
                            a = rr.vselect(
                                full, a, cond_mask=rr.vgt(ah, high, mask=m16)
                            )
                            b = rr.vselect(
                                full, b, cond_mask=rr.vgt(bh, high, mask=m16)
                            )
                            lo, hi = rr.vdeinterleave(
                                rr.vreinterpret_lanes(a, dt.uint8),
                                rr.vreinterpret_lanes(b, dt.uint8),
                            )
                            vals = lo
                        c0 = rr.vhistogram_accumulate(c0, vals, mask=m8, bin=0)
                        c1 = rr.vhistogram_accumulate(c1, vals, mask=m8, bin=1)
                    bins0 = rr.varange(0, dt.uint16)
                    bins1 = rr.varange(128, dt.uint16)
                    wanted = rr.vdups(self.length - self.topk + 1, dt.uint16, mask=m16)
                    i0 = rr.vselect(bins0, full, cond_mask=rr.vge(c0, wanted, mask=m16))
                    i1 = rr.vselect(bins1, full, cond_mask=rr.vge(c1, wanted, mask=m16))
                    selected = rr.vdup(
                        rr.vreduce_min(rr.vmin(i0, i1, mask=m16), mask=m16), mask=m16
                    )
                    if cb.const_expr(digit == 0):
                        high = selected
                        rr.vmem_bar("vst_vld")
                    else:
                        threshold = rr.vadd(
                            rr.vshl(high, 8, mask=m16), selected, mask=m16
                        )

                strict_count = zero
                ones = rr.vdups(1, dt.uint16, mask=m16)
                tie_ureg = rr.vstore_unalign_begin(ties)
                for part in range(self.length // 128):
                    k = rr.vload(keys, part * 128)
                    strict_count = rr.vadd(
                        strict_count,
                        rr.vselect(
                            ones, zero, cond_mask=rr.vgt(k, threshold, mask=m16)
                        ),
                        mask=m16,
                    )
                    pos = rr.varange(part * 128, dt.uint16)
                    squeezed = rr.vsqueeze_and_storeunalign_init(
                        pos, mask=rr.veq(k, threshold, mask=m16)
                    )
                    rr.vstore_unalign(ties, 0, squeezed, tie_ureg)
                rr.vstore_unalign_post(ties, 0, tie_ureg)
                m32 = rr.full_mask()
                packed = rr.vreinterpret_lanes(strict_count, dt.uint32)
                paired = rr.vadd(
                    rr.vbitwise_and(
                        packed, rr.vdups(UINT16_MAX, dt.uint32, mask=m32), mask=m32
                    ),
                    rr.vshr(packed, 16, mask=m32),
                    mask=m32,
                )
                rr.vstore(
                    countbuf, 0, rr.vreduce_sum(paired, mask=m32), rr.update_mask(1)[0]
                )
                rr.vstore(meta, 16, threshold, rr.update_mask(1, elem_bits=16)[0])
                rr.vmem_bar("vst_vld")
            with cb.vf(mode="simt", thread=32):
                if simt.thread_idx()[0] == 0:
                    count = dt.int32(countbuf[0])
                    meta[32] = dt.uint16(self.length - 1) - ties[self.topk - count - 1]
            with cb.vf(mode="simd"):
                m16 = rr.full_mask(elem_bits=16)
                full = rr.vdups(UINT16_MAX, dt.uint16, mask=m16)
                threshold = rr.vload_broadcast(meta, 16)
                revthreshold = rr.vload_broadcast(meta, 32)
                for part in range(self.topk // 128):
                    rr.vstore(indices, part * 128, full, m16)
                if cb.const_expr(self.topk % 128 != 0):
                    rr.vstore(
                        indices,
                        (self.topk // 128) * 128,
                        full,
                        rr.update_mask(self.topk % 128, elem_bits=16)[0],
                    )
                rr.vmem_bar("vst_vld")
                ureg = rr.vstore_unalign_begin(indices)
                for part in range(self.length // 128):
                    k = rr.vload(keys, part * 128)
                    pos = rr.varange(part * 128, dt.uint16)
                    rev = rr.vsub(
                        rr.vdups(self.length - 1, dt.uint16, mask=m16), pos, mask=m16
                    )
                    strict = rr.vgt(k, threshold, mask=m16)
                    eq = rr.veq(k, threshold, mask=m16)
                    tie = rr.mask_and(
                        eq, rr.vge(rev, revthreshold, mask=m16), exec_mask=m16
                    )
                    take = rr.mask_or(strict, tie, exec_mask=m16)
                    take = rr.mask_and(
                        take,
                        rr.vgt(k, rr.vdups(127, dt.uint16, mask=m16), mask=m16),
                        exec_mask=m16,
                    )
                    squeezed = rr.vsqueeze_and_storeunalign_init(pos, mask=take)
                    rr.vstore_unalign(indices, 0, squeezed, ureg)
                rr.vstore_unalign_post(indices, 0, ureg)
                rr.vmem_bar("vst_vld")
                m32 = rr.full_mask()
                for part in range(self.topk // 64):
                    v = rr.vload_unpack(
                        indices, part * 64, unpack_mode=rr.UnpackMode.B16_TO_B32
                    )
                    z = rr.vreinterpret_lanes(v, dt.int32)
                    z = rr.vselect(
                        rr.vdups(-1, dt.int32, mask=m32),
                        z,
                        cond_mask=rr.veqs(z, 65535, mask=m32),
                    )
                    rr.vstore(outub, part * 64, z, m32)
            cb.mem_copy(cb.tile_slice(out, (self.topk,), (row,)), ch_output.next_read())
