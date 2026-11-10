# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Decode selected compressed records directly into the gathered-KV workspace.

Single-record fallback per helper: raw 384 B x depth2, decoded 1024 B x depth2,
and a read-only zero row of 1024 B: 3840 B before alignment.
Logical-index rows up to 4096 columns use a 16-KiB read-only UB cache.
Logical indices also use a 32-KiB page-table row cache: 52992 B maximum.
Physical-index caching adds (end_item-begin_item)*4 bytes of UB when enabled.
The T8/H8,H16 selected path groups eight records per transaction and copies
only each AIV's owned indices. At grid28/P1=24 steps, the two helpers use
24 or at most56 local int32 indices per AIV. Each helper has raw6144 B,
decoded16384 B and zero1024 B before index storage/alignment.
No full-cache BF16 workspace is read or written by this path.
"""

from cannbotdsl import from_tensor

from cannbotdsl import make_layout
from cannbotdsl import RingBuffer, MemLoc, dtypes, tile_slice as tile_view
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang import const_expr
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl import vec_sync_all

if __package__:
    from .cache_decode import Fp4CacheDecoder
else:
    from cache_decode import Fp4CacheDecoder
if __package__:
    from .paged_cache import PagedCacheAddressing
else:
    from paged_cache import PagedCacheAddressing


PAGE_TABLE_CACHE_COLUMNS = 8192
INDEX_CACHE_COLUMNS = 4096


class CompressedCacheGather:
    """AIV-owned raw-record load, BF16-scale decode and KV assembly."""

    def __init__(
        self,
        tokens,
        topk,
        page_size,
        win_len,
        length_stride=0,
        cache_batch_stride=0,
        begin_item=0,
        end_item=None,
        physical_index_cache=False,
        index_vec_num=0,
    ):
        self.index_vec_num = index_vec_num
        self.physical_index_cache = physical_index_cache and cache_batch_stride == 0
        self.begin_item = begin_item
        self.end_item = tokens * topk if end_item is None else end_item
        self.tokens = tokens
        self.topk = topk
        self.page_size = page_size
        self.win_len = win_len
        self.length_stride = length_stride
        self.cache_batch_stride = cache_batch_stride
        self.addressing = PagedCacheAddressing(page_size, cache_batch_stride)
        self.decoder = Fp4CacheDecoder(1, native_bf16_scales=True)

    @jit
    def _run_selected_batches(
        self, cache_gm, index_gm, zero_gm, kv_gm, vec_id, vec_num
    ):
        batch = 8
        raw = RingBuffer(MemLoc.UB, dtypes.uint8, make_layout((batch, 384)), depth=2)
        decoded = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((batch, 512)), depth=2
        )
        zero = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, 512)))
        # All copies have 32-byte inner spans. Complete rounds use a strided
        # GM rectangle; the final partial round is copied only by its owners.
        rounds = (self.end_item - self.begin_item) // (self.index_vec_num * batch)
        tail = (self.end_item - self.begin_item) % (self.index_vec_num * batch)
        indices = Buffer(
            MemLoc.UB,
            dtypes.int32,
            make_layout((rounds + (1 if tail else 0), max(8, batch))),
        )
        flat = index_gm.view((1, index_gm.shape[0] * self.topk))
        if const_expr(rounds > 0):
            matrix = flat[
                0:1,
                self.begin_item : self.begin_item + rounds * self.index_vec_num * batch,
            ].view((rounds, self.index_vec_num * batch))
            mem_copy(
                tile_view(indices, (rounds, batch), (0, 0)),
                tile_view(matrix, (rounds, batch), (0, vec_id)),
            )
        if const_expr(tail > 0):
            if vec_id * batch < tail:
                offset = (
                    self.begin_item
                    + rounds * self.index_vec_num * batch
                    + vec_id * batch
                )
                mem_copy(
                    tile_view(indices, (1, batch), (rounds, 0)),
                    flat[0:1, offset : offset + batch],
                )
        mem_copy(zero, zero_gm)
        vec_sync_all()
        for first in range(
            self.begin_item + vec_id * batch, self.end_item, vec_num * batch
        ):
            raw_write = raw.next_write()
            # Allocate a whole transaction, then issue all independent gathers.
            for row in range(batch):
                index = dtypes.int32(
                    indices[
                        (first - self.begin_item) // (self.index_vec_num * batch), row
                    ]
                )
                if index >= 0:
                    page, off = (
                        dtypes.int64(index // self.page_size),
                        dtypes.int64(index % self.page_size),
                    )
                    mem_copy(
                        tile_view(raw_write, (1, 320), (row, 0)),
                        tile_view(cache_gm[page, None, None], (1, 320), (off, 0)),
                    )
            raw_read = raw.next_read()
            decoded_write = decoded.next_write()
            for row in range(batch):
                index = dtypes.int32(
                    indices[
                        (first - self.begin_item) // (self.index_vec_num * batch), row
                    ]
                )
                if index >= 0:
                    self.decoder.decode(raw_read, decoded_write, row)
                else:
                    mem_copy(tile_view(decoded_write, (1, 512), (row, 0)), zero)
            tok, col = first // self.topk, first % self.topk
            kv_tok = tile_view(kv_gm, (self.win_len + self.topk, 512), (tok, 0))
            mem_copy(
                tile_view(kv_tok, (batch, 512), ((self.win_len + col) // batch, 0)),
                decoded.next_read(),
            )
        vec_sync_all()

    @jit
    def _run_records(
        self,
        cache_gm,
        index_gm,
        lengths_gm,
        zero_gm,
        kv_gm,
        vec_id,
        vec_num,
        block_table_gm,
    ):
        raw = RingBuffer(MemLoc.UB, dtypes.uint8, make_layout((1, 384)), depth=2)
        decoded = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, 512)), depth=2)
        zero = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, 512)))
        if const_expr(self.cache_batch_stride > 0 and self.topk <= INDEX_CACHE_COLUMNS):
            index_cache = Buffer(
                MemLoc.UB, dtypes.int32, make_layout((1, INDEX_CACHE_COLUMNS))
            )
        else:
            index_cache = from_tensor(zero, dtypes.int32, make_layout((1, 256)))
        if const_expr(self.cache_batch_stride > 0):
            table_cache = Buffer(
                MemLoc.UB, dtypes.int32, make_layout((1, PAGE_TABLE_CACHE_COLUMNS))
            )
        else:
            table_cache = from_tensor(zero, dtypes.int32, make_layout((1, 256)))
        if const_expr(self.physical_index_cache):
            # 每个阶段只缓存自己负责的索引区间；GM可覆盖多于T个query。
            physical_indices = Buffer(
                MemLoc.UB,
                dtypes.int32,
                make_layout((1, self.end_item - self.begin_item)),
            )
            flat_indices = index_gm.view((1, index_gm.shape[0] * self.topk))
            mem_copy(
                physical_indices, flat_indices[0:1, self.begin_item : self.end_item]
            )
        else:
            physical_indices = index_cache
        cached_batch = dtypes.int64(-1)
        cached_index_token = dtypes.int64(-1)
        mem_copy(zero, zero_gm)
        vec_sync_all()
        for item in range(vec_id + self.begin_item, self.end_item, vec_num):
            tok = item // self.topk
            col = item % self.topk
            index = dtypes.int32(0)
            if const_expr(self.physical_index_cache):
                index = dtypes.int32(physical_indices[0, item - self.begin_item])
            elif const_expr(
                self.cache_batch_stride > 0 and self.topk <= INDEX_CACHE_COLUMNS
            ):
                if tok != cached_index_token:
                    # 当前query的索引行一次DMA到UB，后续记录只进行本地标量读取。
                    mem_copy(
                        index_cache,
                        tile_view(index_gm, (1, INDEX_CACHE_COLUMNS), (tok, 0)),
                    )
                    vec_sync_all()
                    cached_index_token = tok
                index = dtypes.int32(index_cache[0, col])
            else:
                index = dtypes.int32(index_gm[tok, col])
            if self.length_stride:
                length = dtypes.int32(lengths_gm[tok // self.length_stride])
                if col >= length:
                    index = dtypes.int32(-1)
            kv_tok = tile_view(
                kv_gm.view(self.tokens * (self.win_len + self.topk), 512),
                (self.win_len + self.topk, 512),
                (tok, 0),
            )
            target = tile_view(kv_tok, (1, 512), (self.win_len + col, 0))
            if index >= 0:
                page = dtypes.int64(0)
                if const_expr(self.cache_batch_stride > 0):
                    if block_table_gm.shape[1] <= PAGE_TABLE_CACHE_COLUMNS:
                        batch = dtypes.int64(tok // self.cache_batch_stride)
                        if batch != cached_batch:
                            # 静态UB容量，GM尾部按真实页表列数裁剪；同一请求
                            # 的后续记录和query复用该行，每次launch重新加载。
                            mem_copy(
                                table_cache,
                                tile_view(
                                    block_table_gm,
                                    (1, PAGE_TABLE_CACHE_COLUMNS),
                                    (batch, 0),
                                ),
                            )
                            vec_sync_all()
                            cached_batch = batch
                        page = dtypes.int64(
                            table_cache[0, dtypes.int64(index // self.page_size)]
                        )
                    else:
                        page = self.addressing.page(index, block_table_gm, tok)
                else:
                    page = self.addressing.page(index, block_table_gm, tok)
                off = dtypes.int64(index % self.page_size)
                raw_write = raw.next_write()
                # GM 只读真实的320字节，UB补到384，保证BF16 unpack不越界。
                mem_copy(
                    tile_view(raw_write, (1, 320), (0, 0)),
                    tile_view(cache_gm[page, None, None], (1, 320), (off, 0)),
                )
                self.decoder.decode(raw.next_read(), decoded.next_write(), 0)
                mem_copy(target, decoded.next_read())
            else:
                # 零行在循环外已发布，只读复用；无效索引不访问压缩cache。
                mem_copy(target, zero)
        vec_sync_all()

    @jit
    def run(
        self,
        cache_gm,
        index_gm,
        lengths_gm,
        zero_gm,
        kv_gm,
        vec_id,
        vec_num,
        block_table_gm,
    ):
        if const_expr(
            self.physical_index_cache
            and self.index_vec_num in (56, 64)
            and self.topk == 512
            and not self.length_stride
            and self.begin_item % 8 == 0
            and self.end_item % 8 == 0
        ):
            self._run_selected_batches(
                cache_gm, index_gm, zero_gm, kv_gm, vec_id, vec_num
            )
        else:
            self._run_records(
                cache_gm,
                index_gm,
                lengths_gm,
                zero_gm,
                kv_gm,
                vec_id,
                vec_num,
                block_table_gm,
            )
