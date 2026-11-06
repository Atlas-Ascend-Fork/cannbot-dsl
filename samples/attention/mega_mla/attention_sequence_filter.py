# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Publish safe physical indices and write slots before attention consumers.

Physical indices and slots are private cross-core metadata. Optional common
Query metadata is a separate public snapshot. Each AIV owns disjoint elements;
the caller's barrier publishes writes before downstream consumers use them.
"""

from cannbotdsl import make_layout
from cannbotdsl import dtypes, MemLoc, tile_slice as tile_view, vec_sync_all
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang import vf
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang import const_expr
from cannbotdsl.ops.scalar import vec_store_bypass
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops import reg as rr


# A resource budget, not a limit on accepted page-table columns.
PAGE_TABLE_UB_COLUMNS = (32 * 1024) // 4


class AttentionSequenceFilter:
    def __init__(self, config, page_size, win_len, cmp_len, compressed_page_size=None):
        self.config = config
        self.page_size = page_size
        self.compressed_page_size = (
            page_size if compressed_page_size is None else compressed_page_size
        )
        self.win_len, self.cmp_len = win_len, cmp_len

    @jit
    def _query_details(self, tok, cu_q, used_q):
        cfg = self.config
        batch = dtypes.int64(0)
        start = dtypes.int64(0)
        end = dtypes.int64(0)
        if const_expr(cfg.has_cu_q):
            lo, hi = dtypes.int64(0), dtypes.int64(cfg.batches)
            while lo < hi:
                mid = (lo + hi) // 2
                if dtypes.int64(cu_q[mid + 1]) <= tok:
                    lo = mid + 1
                else:
                    hi = mid
            batch = min(lo, cfg.batches - 1)
            start = max(dtypes.int64(0), min(dtypes.int64(cu_q[batch]), cfg.tokens))
            end = max(start, min(dtypes.int64(cu_q[batch + 1]), cfg.tokens))
        else:
            batch = dtypes.int64(tok // (cfg.tokens // cfg.batches))
            start = batch * (cfg.tokens // cfg.batches)
            end = start + cfg.tokens // cfg.batches
        used = end - start
        if const_expr(cfg.has_used_q):
            used = max(dtypes.int64(0), min(dtypes.int64(used_q[batch]), used))
        active = dtypes.int32(0)
        if tok >= start and tok < start + used:
            active = dtypes.int32(1)
        return batch, start, end, used, active

    @jit
    def _query(self, tok, cu_q, used_q, query_meta=None):
        if const_expr(self.config.query_metadata_mode == "reuse"):
            original = dtypes.int64(query_meta[tok, 0])
            batch = max(dtypes.int64(0), min(original, self.config.batches - 1))
            active = dtypes.int32(query_meta[tok, 5])
            if original != batch:
                active = dtypes.int32(0)
        else:
            batch, start, end, used, active = self._query_details(tok, cu_q, used_q)
        return batch, active

    @jit
    def _mark_table_span(
        self, inputs, outputs, control, col_start, topk, last_index, page_shift
    ):
        with vf(mode="simd"):
            mask = rr.full_mask(elem_bits=32)
            index = rr.vload(inputs, 0)
            cols = rr.vadds(
                rr.varange(0, dtypes.int32), dtypes.int32(col_start), mask=mask
            )
            valid = rr.mask_and(
                rr.vges(index, 0, mask=mask),
                rr.vles(index, dtypes.int32(last_index), mask=mask),
                exec_mask=mask,
            )
            valid = rr.mask_and(
                valid, rr.vlts(cols, dtypes.int32(topk), mask=mask), exec_mask=mask
            )
            page_index = rr.vshr(index, page_shift, mask=mask)
            low = rr.vselect(
                page_index,
                rr.vdups(2147483647, dtypes.int32, mask=mask),
                cond_mask=valid,
            )
            high = rr.vselect(
                page_index, rr.vdups(-1, dtypes.int32, mask=mask), cond_mask=valid
            )
            rr.vstore(control, 0, rr.vreduce_min(low, mask=mask), mask)
            rr.vstore(control, 64, rr.vreduce_max(high, mask=mask), mask)
            rr.vstore(outputs, 0, rr.vdups(-1, dtypes.int32, mask=mask), mask)
            rr.vmem_bar("vst_vld")

    @jit
    def _mark_segment_presence(
        self, inputs, control, col_start, topk, last_index, begin, end, page_shift
    ):
        with vf(mode="simd"):
            mask = rr.full_mask(elem_bits=32)
            index = rr.vload(inputs, 0)
            cols = rr.vadds(
                rr.varange(0, dtypes.int32), dtypes.int32(col_start), mask=mask
            )
            valid = rr.mask_and(
                rr.vges(index, 0, mask=mask),
                rr.vles(index, dtypes.int32(last_index), mask=mask),
                exec_mask=mask,
            )
            valid = rr.mask_and(
                valid, rr.vlts(cols, dtypes.int32(topk), mask=mask), exec_mask=mask
            )
            page_index = rr.vshr(index, page_shift, mask=mask)
            selected = rr.mask_and(
                valid,
                rr.vges(page_index, dtypes.int32(begin), mask=mask),
                exec_mask=mask,
            )
            selected = rr.mask_and(
                selected,
                rr.vlts(page_index, dtypes.int32(end), mask=mask),
                exec_mask=mask,
            )
            present = rr.vselect(
                rr.vdups(1, dtypes.int32, mask=mask),
                rr.vdups(0, dtypes.int32, mask=mask),
                cond_mask=selected,
            )
            rr.vstore(control, 0, rr.vreduce_max(present, mask=mask), mask)
            rr.vmem_bar("vst_vld")

    @jit
    def _gather_table_segment(
        self,
        inputs,
        outputs,
        pages,
        col_start,
        topk,
        last_index,
        begin,
        end,
        cache_pages,
        page_shift,
    ):
        with vf(mode="simd"):
            mask = rr.full_mask(elem_bits=32)
            index = rr.vload(inputs, 0)
            cols = rr.vadds(
                rr.varange(0, dtypes.int32), dtypes.int32(col_start), mask=mask
            )
            valid = rr.mask_and(
                rr.vges(index, 0, mask=mask),
                rr.vles(index, dtypes.int32(last_index), mask=mask),
                exec_mask=mask,
            )
            valid = rr.mask_and(
                valid, rr.vlts(cols, dtypes.int32(topk), mask=mask), exec_mask=mask
            )
            page_index = rr.vshr(index, page_shift, mask=mask)
            selected = rr.mask_and(
                valid,
                rr.vges(page_index, dtypes.int32(begin), mask=mask),
                exec_mask=mask,
            )
            selected = rr.mask_and(
                selected,
                rr.vlts(page_index, dtypes.int32(end), mask=mask),
                exec_mask=mask,
            )
            offset = rr.vselect(
                rr.vadds(page_index, -dtypes.int32(begin), mask=mask),
                rr.vdups(0, dtypes.int32, mask=mask),
                cond_mask=selected,
            )
            page = rr.vgather(pages, rr.vreinterpret(offset, dtypes.uint32), mask=mask)
            valid = rr.mask_and(selected, rr.vges(page, 0, mask=mask), exec_mask=mask)
            valid = rr.mask_and(
                valid,
                rr.vlts(page, dtypes.int32(cache_pages), mask=mask),
                exec_mask=mask,
            )
            physical = rr.vadd(
                rr.vshl(page, page_shift, mask=mask),
                rr.vbitwise_and(
                    index,
                    rr.vdups((1 << page_shift) - 1, dtypes.int32, mask=mask),
                    mask=mask,
                ),
                mask=mask,
            )
            result = rr.vselect(
                physical, rr.vdups(-1, dtypes.int32, mask=mask), cond_mask=valid
            )
            rr.vstore(outputs, 0, result, selected)
            rr.vmem_bar("vst_vld")

    @jit
    def _route_chunked_tile(
        self,
        inputs,
        outputs,
        pages,
        table,
        batch,
        col_start,
        topk,
        last_index,
        cache_pages,
        page_shift,
    ):
        control = Buffer(MemLoc.UB, dtypes.int32, make_layout((2, 64)))
        # VF register/mask values stay within the corresponding jit step; only
        # Buffer contents and scalar segment bounds cross control-flow regions.
        self._mark_table_span(
            inputs, outputs, control, col_start, topk, last_index, page_shift
        )
        vec_sync_all()
        first = dtypes.int64(control[0, 0]) // pages.shape[1]
        stop = dtypes.int64(control[1, 0]) // pages.shape[1] + 1
        for segment in range(first, stop):
            begin = segment * pages.shape[1]
            end = begin + pages.shape[1]
            self._mark_segment_presence(
                inputs, control, col_start, topk, last_index, begin, end, page_shift
            )
            vec_sync_all()
            if dtypes.int32(control[0, 0]) != 0:
                mem_copy(pages, tile_view(table, (1, pages.shape[1]), (batch, segment)))
                vec_sync_all()
                self._gather_table_segment(
                    inputs,
                    outputs,
                    pages,
                    col_start,
                    topk,
                    last_index,
                    begin,
                    end,
                    cache_pages,
                    page_shift,
                )
        vec_sync_all()

    @jit
    def _route_vector(
        self,
        indices,
        dst,
        lengths,
        used_kv,
        table,
        cu_q,
        used_q,
        vec_id,
        vec_num,
        cache_pages,
        is_window,
        query_meta=None,
        tile_offset=0,
    ):
        cfg = self.config
        width = self.win_len if is_window else self.cmp_len
        page_size = self.page_size if is_window else self.compressed_page_size
        page_shift = 6 if page_size == 64 else 7
        has_used = cfg.has_used_win if is_window else cfg.has_used_cmp
        has_topk = cfg.has_win_topk if is_window else cfg.has_cmp_topk
        per_query = cfg.win_topk_per_query if is_window else cfg.cmp_topk_per_query
        capacity = table.shape[1] * page_size
        inputs = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, 64)))
        outputs = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, 64)))
        pages = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, 64)))
        for tile in range(
            (vec_id + vec_num - tile_offset) % vec_num,
            cfg.tokens * (width // 64),
            vec_num,
        ):
            tok, col_tile = tile // (width // 64), tile % (width // 64)
            col_start = col_tile * 64
            batch, active = self._query(tok, cu_q, used_q, query_meta)
            topk = dtypes.int64(width)
            if const_expr(has_topk):
                length_row = tok if per_query else batch
                topk = max(
                    dtypes.int64(0), min(dtypes.int64(lengths[length_row]), width)
                )
            limit = dtypes.int64(capacity)
            if const_expr(has_used):
                limit = max(
                    dtypes.int64(0), min(dtypes.int64(used_kv[batch]), capacity)
                )
            if active != 0 and col_start < topk and limit > 0:
                # The original index tile is consumed once by this stage.
                mem_copy(
                    inputs,
                    tile_view(indices, (1, 64), (tok, col_tile)),
                    options=copy_options(l2_cache_ctl=4),
                )
                # mem_copy clips the padded source tile to the real table row.
                mem_copy(pages, tile_view(table, (1, 64), (batch, 0)))
                vec_sync_all()
                with vf(mode="simd"):
                    mask = rr.full_mask(elem_bits=32)
                    index = rr.vload(inputs, 0)
                    cols = rr.vadds(
                        rr.varange(0, dtypes.int32), dtypes.int32(col_start), mask=mask
                    )
                    valid = rr.mask_and(
                        rr.vges(index, 0, mask=mask),
                        rr.vlts(index, dtypes.int32(limit), mask=mask),
                        exec_mask=mask,
                    )
                    valid = rr.mask_and(
                        valid,
                        rr.vlts(cols, dtypes.int32(topk), mask=mask),
                        exec_mask=mask,
                    )
                    zeros = rr.vdups(0, dtypes.int32, mask=mask)
                    page_index = rr.vselect(
                        rr.vshr(index, page_shift, mask=mask), zeros, cond_mask=valid
                    )
                    page = rr.vgather_reg(
                        rr.vload(pages, 0), rr.vreinterpret(page_index, dtypes.uint32)
                    )
                    valid = rr.mask_and(
                        valid, rr.vges(page, 0, mask=mask), exec_mask=mask
                    )
                    valid = rr.mask_and(
                        valid,
                        rr.vlts(page, dtypes.int32(cache_pages), mask=mask),
                        exec_mask=mask,
                    )
                    physical = rr.vadd(
                        rr.vshl(page, page_shift, mask=mask),
                        rr.vbitwise_and(
                            index,
                            rr.vdups((1 << page_shift) - 1, dtypes.int32, mask=mask),
                            mask=mask,
                        ),
                        mask=mask,
                    )
                    result = rr.vselect(
                        physical, rr.vdups(-1, dtypes.int32, mask=mask), cond_mask=valid
                    )
                    rr.vstore(outputs, 0, result, mask)
                    rr.vmem_bar("vst_vld")
            else:
                with vf(mode="simd"):
                    mask = rr.full_mask(elem_bits=32)
                    rr.vstore(outputs, 0, rr.vdups(-1, dtypes.int32, mask=mask), mask)
                    rr.vmem_bar("vst_vld")
            # Publish vector results before DMA, and drain DMA before UB reuse.
            vec_sync_all()
            mem_copy(tile_view(dst, (1, 64), (tok, col_tile)), outputs)
            vec_sync_all()

    @jit
    def _route_ub(
        self,
        indices,
        dst,
        lengths,
        used_kv,
        table,
        cu_q,
        used_q,
        vec_id,
        vec_num,
        cache_pages,
        is_window,
        query_meta=None,
    ):
        cfg = self.config
        width = self.win_len if is_window else self.cmp_len
        page_size = self.page_size if is_window else self.compressed_page_size
        page_shift = 6 if page_size == 64 else 7
        has_used = cfg.has_used_win if is_window else cfg.has_used_cmp
        has_topk = cfg.has_win_topk if is_window else cfg.has_cmp_topk
        per_query = cfg.win_topk_per_query if is_window else cfg.cmp_topk_per_query
        capacity = table.shape[1] * page_size
        cache_columns = min(((table.shape[1] + 63) // 64) * 64, PAGE_TABLE_UB_COLUMNS)
        use_register = const_expr(table.shape[1] <= 64)
        use_chunks = const_expr(table.shape[1] > PAGE_TABLE_UB_COLUMNS)
        inputs = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, 64)))
        outputs = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, 64)))
        pages = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, cache_columns)))
        cached_batch = dtypes.int64(-1)
        for tile in range(vec_id, cfg.tokens * (width // 64), vec_num):
            tok, col_tile = tile // (width // 64), tile % (width // 64)
            col_start = col_tile * 64
            batch, active = self._query(tok, cu_q, used_q, query_meta)
            topk = dtypes.int64(width)
            if const_expr(has_topk):
                length_row = tok if per_query else batch
                topk = max(
                    dtypes.int64(0), min(dtypes.int64(lengths[length_row]), width)
                )
            limit = dtypes.int64(capacity)
            if const_expr(has_used):
                limit = max(
                    dtypes.int64(0), min(dtypes.int64(used_kv[batch]), capacity)
                )
            # The table capacity may exceed the largest int32 index. Avoid
            # narrowing an exclusive 2**31 bound into a negative int32 value.
            last_index = dtypes.int32(min(limit - 1, 2147483647))
            if active != 0 and col_start < topk and limit > 0:
                mem_copy(
                    inputs,
                    tile_view(indices, (1, 64), (tok, col_tile)),
                    options=copy_options(l2_cache_ctl=4),
                )
                if const_expr(use_chunks):
                    vec_sync_all()
                    self._route_chunked_tile(
                        inputs,
                        outputs,
                        pages,
                        table,
                        batch,
                        col_start,
                        topk,
                        last_index,
                        cache_pages,
                        page_shift,
                    )
                else:
                    if const_expr(use_register):
                        mem_copy(
                            pages, tile_view(table, (1, cache_columns), (batch, 0))
                        )
                    else:
                        if batch != cached_batch:
                            mem_copy(
                                pages, tile_view(table, (1, cache_columns), (batch, 0))
                            )
                            cached_batch = batch
                    vec_sync_all()
                    with vf(mode="simd"):
                        mask = rr.full_mask(elem_bits=32)
                        index = rr.vload(inputs, 0)
                        cols = rr.vadds(
                            rr.varange(0, dtypes.int32),
                            dtypes.int32(col_start),
                            mask=mask,
                        )
                        valid = rr.mask_and(
                            rr.vges(index, 0, mask=mask),
                            rr.vles(index, last_index, mask=mask),
                            exec_mask=mask,
                        )
                        valid = rr.mask_and(
                            valid,
                            rr.vlts(cols, dtypes.int32(topk), mask=mask),
                            exec_mask=mask,
                        )
                        zeros = rr.vdups(0, dtypes.int32, mask=mask)
                        page_index = rr.vselect(
                            rr.vshr(index, page_shift, mask=mask),
                            zeros,
                            cond_mask=valid,
                        )
                        if const_expr(use_register):
                            page = rr.vgather_reg(
                                rr.vload(pages, 0),
                                rr.vreinterpret(page_index, dtypes.uint32),
                            )
                        else:
                            page = rr.vgather(
                                pages,
                                rr.vreinterpret(page_index, dtypes.uint32),
                                mask=mask,
                            )
                        valid = rr.mask_and(
                            valid, rr.vges(page, 0, mask=mask), exec_mask=mask
                        )
                        valid = rr.mask_and(
                            valid,
                            rr.vlts(page, dtypes.int32(cache_pages), mask=mask),
                            exec_mask=mask,
                        )
                        physical = rr.vadd(
                            rr.vshl(page, page_shift, mask=mask),
                            rr.vbitwise_and(
                                index,
                                rr.vdups(
                                    (1 << page_shift) - 1, dtypes.int32, mask=mask
                                ),
                                mask=mask,
                            ),
                            mask=mask,
                        )
                        result = rr.vselect(
                            physical,
                            rr.vdups(-1, dtypes.int32, mask=mask),
                            cond_mask=valid,
                        )
                        rr.vstore(outputs, 0, result, mask)
                        rr.vmem_bar("vst_vld")
            else:
                with vf(mode="simd"):
                    mask = rr.full_mask(elem_bits=32)
                    rr.vstore(outputs, 0, rr.vdups(-1, dtypes.int32, mask=mask), mask)
                    rr.vmem_bar("vst_vld")
            vec_sync_all()
            mem_copy(tile_view(dst, (1, 64), (tok, col_tile)), outputs)
            vec_sync_all()

    @jit
    def _route(
        self,
        indices,
        dst,
        lengths,
        used_kv,
        table,
        cu_q,
        used_q,
        vec_id,
        vec_num,
        cache_pages,
        is_window,
        query_meta=None,
        tile_offset=0,
    ):
        cfg = self.config
        width = self.win_len if is_window else self.cmp_len
        page_size = self.page_size if is_window else self.compressed_page_size
        # Register, full-UB or occupied-segment lookup according to table size.
        if const_expr(
            cfg.paged
            and page_size in (64, 128)
            and width % 64 == 0
            and table.shape[1] > 0
        ):
            if const_expr(table.shape[1] <= 64):
                self._route_vector(
                    indices,
                    dst,
                    lengths,
                    used_kv,
                    table,
                    cu_q,
                    used_q,
                    vec_id,
                    vec_num,
                    cache_pages,
                    is_window,
                    query_meta,
                    tile_offset,
                )
            else:
                self._route_ub(
                    indices,
                    dst,
                    lengths,
                    used_kv,
                    table,
                    cu_q,
                    used_q,
                    vec_id,
                    vec_num,
                    cache_pages,
                    is_window,
                    query_meta,
                )
        else:
            self._route_scalar(
                indices,
                dst,
                lengths,
                used_kv,
                table,
                cu_q,
                used_q,
                vec_id,
                vec_num,
                cache_pages,
                is_window,
                query_meta,
            )

    @jit
    def _route_scalar(
        self,
        indices,
        dst,
        lengths,
        used_kv,
        table,
        cu_q,
        used_q,
        vec_id,
        vec_num,
        cache_pages,
        is_window,
        query_meta=None,
    ):
        cfg = self.config
        width = self.win_len if is_window else self.cmp_len
        page_size = self.page_size if is_window else self.compressed_page_size
        has_used = cfg.has_used_win if is_window else cfg.has_used_cmp
        has_topk = cfg.has_win_topk if is_window else cfg.has_cmp_topk
        per_query = cfg.win_topk_per_query if is_window else cfg.cmp_topk_per_query
        capacity = table.shape[1] * page_size if cfg.paged else cache_pages * page_size
        for item in range(vec_id, cfg.tokens * width, vec_num):
            tok, col = item // width, item % width
            batch, active = self._query(tok, cu_q, used_q, query_meta)
            topk = dtypes.int64(width)
            if const_expr(has_topk):
                length_row = tok if per_query else batch
                topk = max(
                    dtypes.int64(0), min(dtypes.int64(lengths[length_row]), width)
                )
            limit = dtypes.int64(capacity)
            if const_expr(has_used):
                limit = max(
                    dtypes.int64(0), min(dtypes.int64(used_kv[batch]), capacity)
                )
            physical = dtypes.int32(-1)
            # Never dereference a page table for padding, inactive Q or bad index.
            if active != 0 and col < topk:
                index = dtypes.int64(indices[tok, col])
                if index >= 0 and index < limit:
                    if const_expr(cfg.paged):
                        page = dtypes.int64(table[batch, index // page_size])
                        if page >= 0 and page < cache_pages:
                            physical = dtypes.int32(
                                page * page_size + index % page_size
                            )
                    else:
                        physical = dtypes.int32(index)
            vec_store_bypass(dst.ptr((tok, col)), physical)

    @jit
    def _filter_sequences(
        self,
        win_idx,
        cmp_idx,
        win_lengths,
        cmp_lengths,
        win_table,
        cmp_table,
        cu_q,
        used_q,
        used_win,
        used_cmp,
        slot,
        win_out,
        cmp_out,
        slot_out,
        vec_id,
        vec_num,
        win_pages,
        cmp_pages,
        query_meta=None,
    ):
        if const_expr(self.config.query_metadata_mode == "export"):
            metadata_words = query_meta.view(dtypes.uint64)
        self._route(
            win_idx,
            win_out,
            win_lengths,
            used_win,
            win_table,
            cu_q,
            used_q,
            vec_id,
            vec_num,
            win_pages,
            True,
            query_meta,
        )
        cmp_tile_offset = 0
        if const_expr(
            self.config.paged
            and self.page_size in (64, 128)
            and self.compressed_page_size in (64, 128)
            and self.win_len % 64 == 0
            and self.cmp_len % 64 == 0
            and 0 < win_table.shape[1] <= 64
            and 0 < cmp_table.shape[1] <= 64
        ):
            cmp_tile_offset = self.config.tokens * (self.win_len // 64) % vec_num
        self._route(
            cmp_idx,
            cmp_out,
            cmp_lengths,
            used_cmp,
            cmp_table,
            cu_q,
            used_q,
            vec_id,
            vec_num,
            cmp_pages,
            False,
            query_meta,
            cmp_tile_offset,
        )
        for tok in range(vec_id, self.config.tokens, vec_num):
            if const_expr(self.config.query_metadata_mode == "export"):
                batch, start, end, used, active = self._query_details(tok, cu_q, used_q)
                # Pack pairs of nonnegative int32 fields into aligned 64-bit stores.
                vec_store_bypass(
                    metadata_words.ptr((tok, 0)),
                    dtypes.uint64(batch) + dtypes.uint64(start) * 4294967296,
                )
                vec_store_bypass(
                    metadata_words.ptr((tok, 1)),
                    dtypes.uint64(end - start) + dtypes.uint64(used) * 4294967296,
                )
                vec_store_bypass(
                    metadata_words.ptr((tok, 2)),
                    dtypes.uint64(tok - start) + dtypes.uint64(active) * 4294967296,
                )
                vec_store_bypass(metadata_words.ptr((tok, 3)), dtypes.uint64(0))
            else:
                batch, active = self._query(tok, cu_q, used_q, query_meta)
            write_slot = dtypes.int32(-1)
            if active != 0:
                original = dtypes.int64(slot[tok])
                if original >= self.page_size and original < win_pages * self.page_size:
                    write_slot = dtypes.int32(original)
            vec_store_bypass(slot_out.ptr((tok,)), write_slot)

    @jit
    def run(
        self,
        win_idx,
        cmp_idx,
        win_lengths,
        cmp_lengths,
        win_table,
        cmp_table,
        cu_q,
        used_q,
        used_win,
        used_cmp,
        slot,
        win_out,
        cmp_out,
        slot_out,
        vec_id,
        vec_num,
        win_pages,
        cmp_pages,
        query_meta=None,
    ):
        cfg = self.config
        if const_expr(
            cfg.query_metadata_mode == "reuse" and cfg.tokens == 8 and cfg.batches <= 8
        ):
            # Each launch refreshes this small snapshot; no pointer-based value cache.
            meta_local = Buffer(MemLoc.UB, dtypes.int32, make_layout((cfg.tokens, 8)))
            mem_copy(meta_local, query_meta)
            if const_expr(cfg.has_used_win):
                win_local = Buffer(MemLoc.UB, dtypes.int32, make_layout((cfg.batches,)))
                mem_copy(win_local, used_win)
            else:
                win_local = used_win
            if const_expr(cfg.has_used_cmp):
                cmp_local = Buffer(MemLoc.UB, dtypes.int32, make_layout((cfg.batches,)))
                mem_copy(cmp_local, used_cmp)
            else:
                cmp_local = used_cmp
            vec_sync_all()
            self._filter_sequences(
                win_idx,
                cmp_idx,
                win_lengths,
                cmp_lengths,
                win_table,
                cmp_table,
                cu_q,
                used_q,
                win_local,
                cmp_local,
                slot,
                win_out,
                cmp_out,
                slot_out,
                vec_id,
                vec_num,
                win_pages,
                cmp_pages,
                meta_local,
            )
        elif const_expr(
            cfg.query_metadata_mode != "reuse"
            and cfg.tokens <= 64
            and cfg.batches <= 64
        ):
            # Refresh small sequence vectors on every invocation. Scalar binary
            # search and per-tile length checks then read UB rather than GM.
            if const_expr(cfg.has_cu_q):
                cu_local = Buffer(MemLoc.UB, dtypes.int32, make_layout(cu_q.shape))
                mem_copy(cu_local, cu_q)
            else:
                cu_local = cu_q
            if const_expr(cfg.has_used_q):
                q_local = Buffer(MemLoc.UB, dtypes.int32, make_layout(used_q.shape))
                mem_copy(q_local, used_q)
            else:
                q_local = used_q
            if const_expr(cfg.has_used_win):
                win_local = Buffer(MemLoc.UB, dtypes.int32, make_layout(used_win.shape))
                mem_copy(win_local, used_win)
            else:
                win_local = used_win
            if const_expr(cfg.has_used_cmp):
                cmp_local = Buffer(MemLoc.UB, dtypes.int32, make_layout(used_cmp.shape))
                mem_copy(cmp_local, used_cmp)
            else:
                cmp_local = used_cmp
            if const_expr(cfg.has_win_topk):
                wl_local = Buffer(
                    MemLoc.UB, dtypes.int32, make_layout(win_lengths.shape)
                )
                mem_copy(wl_local, win_lengths)
            else:
                wl_local = win_lengths
            if const_expr(cfg.has_cmp_topk):
                cl_local = Buffer(
                    MemLoc.UB, dtypes.int32, make_layout(cmp_lengths.shape)
                )
                mem_copy(cl_local, cmp_lengths)
            else:
                cl_local = cmp_lengths
            slot_local = Buffer(MemLoc.UB, dtypes.int32, make_layout(slot.shape))
            mem_copy(slot_local, slot)
            vec_sync_all()
            self._filter_sequences(
                win_idx,
                cmp_idx,
                wl_local,
                cl_local,
                win_table,
                cmp_table,
                cu_local,
                q_local,
                win_local,
                cmp_local,
                slot_local,
                win_out,
                cmp_out,
                slot_out,
                vec_id,
                vec_num,
                win_pages,
                cmp_pages,
                query_meta,
            )
        else:
            self._filter_sequences(
                win_idx,
                cmp_idx,
                win_lengths,
                cmp_lengths,
                win_table,
                cmp_table,
                cu_q,
                used_q,
                used_win,
                used_cmp,
                slot,
                win_out,
                cmp_out,
                slot_out,
                vec_id,
                vec_num,
                win_pages,
                cmp_pages,
                query_meta,
            )
