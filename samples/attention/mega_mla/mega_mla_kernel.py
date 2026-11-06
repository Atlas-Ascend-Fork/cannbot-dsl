# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Device projection/cache/gather/attention scheduling for mega_mla.

P0 initializes projection accumulators and the zero record; P1 distributes
28 logical split-K tasks per M tile over the launch grid. P2 normalizes Q/KV,
applies KV RoPE and writes the window cache; P3 produces full Q while spare
AIVs decode compressed history. Gather prepares Q/KV for attention.
Small-K attention uses the token-local online QK/Softmax/PV pipeline;
other shapes retain the flat QK, vector Softmax and PV paths.
"""

from __future__ import annotations
from cannbotdsl import (
    get_mem_size,
    make_layout,
    MemLoc,
    PIPE,
    RegLayout,
    Tensor,
    dtypes,
    global_sync_all,
    tile_slice as tile_view,
    vec_sync_all,
)
from cannbotdsl.ops.sync import (
    onchip_rewind,
    cube_sync_all,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    vec_sync_block_wait,
    vec_sync_notify,
    vec_sync_wait,
)
from cannbotdsl.buffer import Buffer
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang import const_expr, range_constexpr
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.scalar import vec_store_bypass
from cannbotdsl.ops.memcpy import copy_options, mem_copy

if __package__:
    from .cache_decode import Fp4CacheDecoder, Fp8CacheDecoder
    from .dual_cache_vector_attention import (
        DualCacheVectorAttentionKernel,
        NEG_INF,
        VECTOR_CHUNK_ELEMENTS,
    )
    from .attention_matmul import ProbabilityValueMatmulKernel, QueryKeyMatmulKernel
    from .rotary_position_embedding import CompactRotaryEmbeddingKernel
    from .attention_geometry import (
        HEAD_DIM,
        select_small_k_attention_head_tile,
        validate_attention_geometry,
        validate_query_heads,
    )
    from .mxfp8_matmul import _as_zn_view, L2_CACHE_DISABLE
    from .compressed_cache_gather import CompressedCacheGather
    from .paged_cache import PagedCacheAddressing
    from .qkv_projection import (
        BN1,
        BN2,
        BK1,
        BK2,
        HID,
        KVD,
        KL1_1,
        KL1_2,
        QLORA,
        SKL1_2,
        QkvProjectionKernel,
        _bind_projection_pipeline,
        _create_projection_pipeline,
    )
    from .attention_pipeline import _SmallKAttentionPipeline, _mask_index_prefix
else:
    from cache_decode import Fp4CacheDecoder, Fp8CacheDecoder
    from dual_cache_vector_attention import (
        DualCacheVectorAttentionKernel,
        NEG_INF,
        VECTOR_CHUNK_ELEMENTS,
    )
    from attention_matmul import ProbabilityValueMatmulKernel, QueryKeyMatmulKernel
    from rotary_position_embedding import CompactRotaryEmbeddingKernel
    from attention_geometry import (
        HEAD_DIM,
        select_small_k_attention_head_tile,
        validate_attention_geometry,
        validate_query_heads,
    )
    from mxfp8_matmul import _as_zn_view, L2_CACHE_DISABLE
    from compressed_cache_gather import CompressedCacheGather
    from paged_cache import PagedCacheAddressing
    from qkv_projection import (
        BN1,
        BN2,
        BK1,
        BK2,
        HID,
        KVD,
        KL1_1,
        KL1_2,
        QLORA,
        SKL1_2,
        QkvProjectionKernel,
        _bind_projection_pipeline,
        _create_projection_pipeline,
    )
    from attention_pipeline import _SmallKAttentionPipeline, _mask_index_prefix


D = HEAD_DIM
COMPRESSED_DECODE_UNROLL = 1
COMPRESSED_DECODE_DEPTH = 2
# Reserve alignment, synchronization scratch and compiler temporaries.
SOFTMAX_UB_RESERVE_BYTES = 16 * 1024


def _publish_qk_to_paired_aivs(flag_ids):
    """All QK AIC complete, then each AIC releases its paired 1:2 AIV."""
    cube_sync_all()
    vec_sync_all()
    cube_sync_block_arrive(PIPE.FIXPIPE, flag_ids[1], mode=0)
    cube_sync_block_wait(PIPE.S, flag_ids[1], mode=0)
    cube_sync_block_arrive(PIPE.MTE3, flag_ids[2], mode=2)
    vec_sync_block_wait(PIPE.S, flag_ids[2], mode=2)


class MegaMlaKernel:
    def __init__(
        self,
        k_attn: int,
        page_size: int,
        win_len: int,
        bn: int = 64,
        mode: str = "attn",
        rope_dim: int = 64,
        tokens: int = 1,
        gather_batch: int = 16,
        grid: int = 32,
        win_length_stride: int = 0,
        cmp_length_stride: int = 0,
        heads: int = 16,
        cmp_decode_strategy: str = "predecode",
        cache_batch_stride: int = 0,
        workspace_layout=None,
        compressed_page_size=None,
    ):
        validate_attention_geometry(
            k=k_attn, page_size=page_size, win_len=win_len, grid=grid, tokens=tokens
        )
        self.heads = validate_query_heads(heads)
        self.cv_tile_m = select_small_k_attention_head_tile(
            self.heads, tokens, grid, k_attn, win_len=win_len
        )
        self.small_token_vector_schedule = self.heads in (8, 16) or (
            self.heads == 64 and tokens < 16 and not self.cv_tile_m
        )
        self.tokens = tokens
        self.grid = grid
        self.win_len = win_len
        self.win_length_stride = win_length_stride
        self.cmp_length_stride = cmp_length_stride
        self.cache_batch_stride = cache_batch_stride
        self.workspace_layout = workspace_layout
        self.cache_addressing = PagedCacheAddressing(page_size, cache_batch_stride)
        self.compressed_page_size = (
            page_size if compressed_page_size is None else compressed_page_size
        )
        self.compressed_cache_addressing = PagedCacheAddressing(
            self.compressed_page_size, cache_batch_stride
        )
        if cmp_decode_strategy not in ("predecode", "indexed"):
            raise ValueError("kernel cmp_decode_strategy must be predecode or indexed")
        self.cmp_decode_indexed = cmp_decode_strategy == "indexed"
        # Partial-M16 GM attention uses both AIVs; T is the flattened B*S,
        # independent of how the caller partitions requests. H64 keeps its
        # existing CV schedule and the previously supported T8 vector path.
        self.dual_aiv = (
            self.small_token_vector_schedule and tokens < 16
        ) or tokens == 8
        softmax_workers = max(1, 2 * grid // tokens)
        softmax_row_tile = (self.heads + softmax_workers - 1) // softmax_workers
        self.softmax_batch_rows = (
            softmax_row_tile
            if tokens <= 2 * grid
            and (2 * grid) % tokens == 0
            and self.heads >= 4 * softmax_workers
            and k_attn * (6 * softmax_row_tile + 12) + self.heads * 4
            <= max(0, get_mem_size("ub") - SOFTMAX_UB_RESERVE_BYTES)
            else 0
        )
        overlap_cmp = (
            self.cmp_decode_indexed
            and mode == "attn"
            and tokens < 16
            and self.small_token_vector_schedule
            and k_attn == 640
            and grid in (28, 32)
            and page_size == 128
            and win_len == 128
            and cache_batch_stride == 0
            and win_length_stride == 0
            and cmp_length_stride == 0
        )
        cmp_p1_steps = 24 if overlap_cmp else 0
        cmp_p3_steps = 50 if overlap_cmp else 0
        cmp_index_cache = overlap_cmp
        # 按每个AIV的循环轮次划分任务，三个区间互斥且覆盖全部索引槽位。
        total_cmp_items = tokens * (k_attn - win_len)
        self.cmp_p1_items = min(cmp_p1_steps * grid * 2, total_cmp_items)
        self.cmp_p3_items = min(
            self.cmp_p1_items + cmp_p3_steps * grid * 2, total_cmp_items
        )
        self.compressed_cache_p1 = CompressedCacheGather(
            tokens,
            k_attn - win_len,
            self.compressed_page_size,
            win_len,
            cmp_length_stride,
            cache_batch_stride,
            end_item=self.cmp_p1_items,
            physical_index_cache=cmp_index_cache,
            index_vec_num=grid * 2,
        )
        self.compressed_cache_early = CompressedCacheGather(
            tokens,
            k_attn - win_len,
            self.compressed_page_size,
            win_len,
            cmp_length_stride,
            cache_batch_stride,
            begin_item=self.cmp_p1_items,
            end_item=self.cmp_p3_items,
            physical_index_cache=cmp_index_cache,
            index_vec_num=grid * 2,
        )
        self.compressed_cache_gather = CompressedCacheGather(
            tokens,
            k_attn - win_len,
            self.compressed_page_size,
            win_len,
            cmp_length_stride,
            cache_batch_stride,
            begin_item=self.cmp_p3_items,
            physical_index_cache=cmp_index_cache,
            index_vec_num=grid * 2,
        )
        # The predecode ping-pong preloads one batch per owner. Every owner
        # must have a compressed batch, including at very small T.
        max_owners = (2 * grid + tokens - 1) // tokens
        self.qkv_packed = (
            self.dual_aiv
            and k_attn % 128 == 0
            and win_len == 128
            and rope_dim == 64
            and (self.cmp_decode_indexed or max_owners <= (k_attn - win_len) // 16)
        )
        self.qkv_qpipe = self.qkv_packed
        self.win_decoder = Fp8CacheDecoder()
        self.cmp_decoder = Fp4CacheDecoder(
            COMPRESSED_DECODE_UNROLL, native_bf16_scales=True
        )
        self.cmp_decode_depth = COMPRESSED_DECODE_DEPTH
        self.proj = QkvProjectionKernel(tokens, self.heads)
        # With one M tile, each projection-weight element is read once.
        # SIMD copy controls use 4 for DISABLE (0 is NORMAL on dav-3510).
        if tokens <= self.proj.k2.base_m:
            self.proj.k2.l2_cache_ctl_b = L2_CACHE_DISABLE
            self.proj.k2.l2_cache_ctl_scale_b = L2_CACHE_DISABLE
        self.attn = DualCacheVectorAttentionKernel(
            self.heads,
            D,
            k_attn,
            page_size,
            win_len // page_size,
            (k_attn - win_len) // page_size,
            bn,
        )
        self.k = k_attn
        self.mode = mode
        self.rope = CompactRotaryEmbeddingKernel(1, D, rope_dim, forward=True)
        self.rope_chunk = CompactRotaryEmbeddingKernel(
            1, rope_dim, rope_dim, forward=True
        )
        self.gather_batch = gather_batch
        # QK partitions key rows and PV partitions output features in N64
        # tiles. PV's independent reduction tile requires K128 alignment.
        qk_bn = 64
        pv_bn = 64
        self.qk = QueryKeyMatmulKernel(self.heads, k_attn, D, base_n=qk_bn, base_k=128)
        self.pv = ProbabilityValueMatmulKernel(
            self.heads, D, k_attn, base_n=pv_bn, base_k=min(128, k_attn)
        )
        self.pv_tiles = self.pv.n // self.pv.base_n
        # Keep token-local Q/P reuse unless flattening reduces the longest
        # tile run in both matmuls. This avoids marginal QK-only switches.
        owners_per_token = max(1, grid // tokens)
        qk_tiles = self.qk.n // self.qk.base_n
        qk_owner_tiles = (qk_tiles + owners_per_token - 1) // owners_per_token
        pv_owner_tiles = (self.pv_tiles + owners_per_token - 1) // owners_per_token
        self.flat_attention_tiles = (
            self.heads in (8, 16)
            and tokens < 16
            and (tokens * qk_tiles + grid - 1) // grid < qk_owner_tiles
            and (tokens * self.pv_tiles + grid - 1) // grid < pv_owner_tiles
        )

    @jit
    def _clear_projection_buffers(self, qr_gm, kv_gm, zero_gm, block, nblk):
        """Share immutable zeros and one drain across independent P0 outputs."""
        z = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 128)))
        with vf(mode="simd"):
            mask = rr.full_mask()
            zero = rr.vdups(0.0, dtypes.float32, mask=mask)
            rr.vstore(z, 0, zero, mask)
            rr.vstore(z, 64, zero, mask)
            rr.vmem_bar("vst_vld")
        for it in range(block, self.proj.m * (QLORA // 128), nblk):
            mem_copy(
                tile_view(qr_gm, (1, 128), (it // (QLORA // 128), it % (QLORA // 128))),
                z,
            )
        for it in range(block, self.proj.m * (KVD // 128), nblk):
            mem_copy(
                tile_view(kv_gm, (1, 128), (it // (KVD // 128), it % (KVD // 128))), z
            )
        # The BF16 invalid record has the same 1024 bytes as this FP32 view.
        # The zero buffer is immutable until every MTE3 consumer has finished.
        for it in range(block, (D // 2) // 128, nblk):
            mem_copy(tile_view(zero_gm.view(dtypes.float32), (1, 128), (0, it)), z)
        vec_sync_all()

    @jit
    def _compute_split_k_projection_tile(
        self, k, out_gm, a_gm, b_gm, sa_gm, sb_gm, mi, ni, split
    ):
        """Two owners accumulate separate K2560 halves into one FP32 tile."""
        pm = self.proj.p1_m
        valid_m = pm
        if const_expr(self.proj.m % pm):
            valid_m = min(pm, self.proj.m - mi * pm)
        mem_copy(
            _as_zn_view(k.l1_scale_a.next_write())[:valid_m, : HID // 64],
            (tile_view(sa_gm, (pm, 40, 2), (mi, split, 0))).flatten(-2, -1),
            options=copy_options(l2_cache_ctl=1),
        )
        mem_copy(
            k.l1_scale_b.next_write(),
            (tile_view(sb_gm, (BN1, 40, 2), (ni, split, 0)))
            .flatten(-2, -1)
            .permute((1, 0)),
            options=copy_options(
                l2_cache_ctl=L2_CACHE_DISABLE if self.proj.p1_m_tiles == 1 else 1
            ),
        )
        scale_a_slot = _as_zn_view(k.l1_scale_a.next_read())
        scale_b_slot = k.l1_scale_b.next_read()
        l0c_slot = k.l0c.next_write()
        for ki in range_constexpr(5):
            gki = split * 5 + ki
            mem_copy(
                k.l1_a.next_write()[:valid_m, :KL1_1],
                tile_view(a_gm, (pm, KL1_1), (mi, gki)),
                options=copy_options(l2_cache_ctl=1),
            )
            mem_copy(
                k.l1_b.next_write(),
                tile_view(b_gm, (BN1, KL1_1), (ni, gki)),
                options=copy_options(
                    l2_cache_ctl=L2_CACHE_DISABLE if self.proj.p1_m_tiles == 1 else 1
                ),
            )
            a_slot = k.l1_a.next_read()
            b_slot = k.l1_b.next_read()
            for kk in range_constexpr(KL1_1 // BK1):
                si = ki * (KL1_1 // BK1) + kk
                mem_copy(
                    k.l0a.next_write(),
                    tile_view(a_slot, (pm, BK1), (0, kk)),
                    options=copy_options(
                        mx_scale=tile_view(scale_a_slot, (pm, BK1 // 32), (0, si))
                    ),
                )
                mem_copy(
                    k.l0b.next_write(),
                    tile_view(b_slot, (BN1, BK1), (0, kk)),
                    options=copy_options(
                        mx_scale=tile_view(scale_b_slot, (BK1 // 32, BN1), (si, 0))
                    ),
                )
                matmul(
                    l0c_slot,
                    k.l0a.next_read(),
                    k.l0b.next_read(),
                    init=(si == 0),
                    unit_flag=3 if si == 9 else 2,
                    mx_mode=True,
                )
        mem_copy(
            tile_view(out_gm, (pm, BN1), (mi, ni)),
            k.l0c.next_read()[:valid_m, :BN1],
            options=copy_options(l2_cache_ctl=1, atomic_add=True, unit_flag=3),
        )

    @jit
    def _run_qa_kv_projection(
        self,
        qr,
        kv,
        x8,
        query_down_proj_weight,
        kv_proj_weight,
        sx,
        query_down_proj_scale,
        kv_proj_scale,
        block,
    ):
        proj = self.proj
        # P1 FP8 L0B：2 × N128 × K256 × 1 B = 64 KiB，WqA/WKV 共用。
        # 双 buffer 由 RingBuffer 轮转，允许下一 K 块搬运与当前 MMAD 重叠。
        c1 = _create_projection_pipeline(
            proj.p1_m, BN1, BK1, KL1_1, HID // 2, l0b_depth=2
        )
        _bind_projection_pipeline(proj.k1, c1)
        _bind_projection_pipeline(proj.k3, c1)
        # Keep full-grid ownership unchanged. Smaller grids cycle over all
        # 28 logical split-K tasks before joining the common phase barrier.
        for mi in range(proj.p1_m_tiles):
            if const_expr(self.grid >= 2 * (proj.n1_blocks + proj.n3_blocks)):
                self._compute_qa_kv_projection_task(
                    qr,
                    kv,
                    x8,
                    query_down_proj_weight,
                    kv_proj_weight,
                    sx,
                    query_down_proj_scale,
                    kv_proj_scale,
                    mi,
                    block,
                )
            else:
                for task in range(
                    block, 2 * (proj.n1_blocks + proj.n3_blocks), self.grid
                ):
                    self._compute_qa_kv_projection_task(
                        qr,
                        kv,
                        x8,
                        query_down_proj_weight,
                        kv_proj_weight,
                        sx,
                        query_down_proj_scale,
                        kv_proj_scale,
                        mi,
                        task,
                    )

    @jit
    def _compute_qa_kv_projection_task(
        self,
        qr,
        kv,
        x8,
        query_down_proj_weight,
        kv_proj_weight,
        sx,
        query_down_proj_scale,
        kv_proj_scale,
        mi,
        task,
    ):
        """Map one of the 20 QA / 8 KV split-K tasks to its output tile."""
        proj = self.proj
        if task < 2 * proj.n1_blocks:
            self._compute_split_k_projection_tile(
                proj.k1,
                qr,
                x8,
                query_down_proj_weight,
                sx,
                query_down_proj_scale,
                mi,
                task // 2,
                task % 2,
            )
        elif task < 2 * (proj.n1_blocks + proj.n3_blocks):
            p = task - 2 * proj.n1_blocks
            self._compute_split_k_projection_tile(
                proj.k3,
                kv,
                x8,
                kv_proj_weight,
                sx,
                kv_proj_scale,
                mi,
                p // 2,
                p % 2,
            )

    @jit
    def _run_projection_stages(
        self,
        qr_gm,
        kv_gm,
        qfull_gm,
        hidden_states_gm,
        query_norm_weight_gm,
        kv_norm_weight_gm,
        query_down_proj_weight_gm,
        kv_proj_weight_gm,
        query_up_proj_weight_gm,
        x8_gm,
        sx_gm,
        qn8_gm,
        sqn_gm,
        query_down_proj_scale_gm,
        kv_proj_scale_gm,
        query_up_proj_scale_gm,
        kvn_gm,
        ckc_gm,
        cmp_dec_gm,
        extra_indices_in_kvcache_gm,
        cmp_lengths_gm,
        zero_gm,
        kg_gm,
        extra_block_table_gm,
        rope_cos_gm,
        rope_sin_gm,
        window_cache_slots_gm,
        wkc_gm,
        wvc_gm,
        wsc_gm,
        block,
        nblk,
        fid,
    ):
        proj = self.proj
        # P0 task split: AIV0 quantizes x; AIV1 clears atomic destinations.
        # These buffers are independent; the common B0 barrier publishes both.
        if get_subblock_id() == 0:
            proj._quantize_mxfp8_rows(
                hidden_states_gm, x8_gm, sx_gm, block, nblk, proj.m, HID
            )
        else:
            self._clear_projection_buffers(qr_gm, kv_gm, zero_gm, block, nblk)
        # Publish quantization and clear completion before P1 atomic stores.
        global_sync_all(flag_ids=fid)
        self._run_qa_kv_projection(
            qr_gm,
            kv_gm,
            x8_gm,
            query_down_proj_weight_gm,
            kv_proj_weight_gm,
            sx_gm,
            query_down_proj_scale_gm,
            kv_proj_scale_gm,
            block,
        )
        if self.mode != "proj":
            # All 2*nblk AIVs decode physical compressed rows while the
            # P1 AIC owners compute. This existing barrier also publishes
            # the decoded BF16 workspace before the later indexed gather.
            if const_expr(not self.cmp_decode_indexed):
                self._decode_full_compressed_cache(ckc_gm, cmp_dec_gm, block, nblk)
        if const_expr(
            self.mode != "proj" and self.cmp_decode_indexed and self.cmp_p1_items > 0
        ):
            self.compressed_cache_p1.run(
                ckc_gm,
                extra_indices_in_kvcache_gm,
                cmp_lengths_gm,
                zero_gm,
                kg_gm,
                block * 2 + get_subblock_id(),
                nblk * 2,
                extra_block_table_gm,
            )
        # All P1 atomic FIXPIPE stores complete before any norm load.
        global_sync_all(flag_ids=fid)
        # B2: each token's two AIV partners work independently after B1.
        # AIV0 prepares Q for WqB; AIV1 keeps KV norm/RoPE/quantization in UB
        # and writes the finished window record. B2 publishes both branches.
        if get_subblock_id() == 0:
            proj._rms_norm_quantize_query(
                qr_gm, query_norm_weight_gm, qn8_gm, sqn_gm, block, nblk, proj.m, QLORA
            )
        else:
            if const_expr(self.mode == "proj"):
                proj._rms_norm_key_value(
                    kv_gm, kv_norm_weight_gm, kvn_gm, block, nblk, proj.m, KVD
                )
            else:
                self._rms_norm_rope_quantize_window_cache(
                    kv_gm,
                    kv_norm_weight_gm,
                    rope_cos_gm,
                    rope_sin_gm,
                    window_cache_slots_gm,
                    wkc_gm,
                    block,
                    nblk,
                )
        vec_sync_all()
        global_sync_all(flag_ids=fid)
        onchip_rewind(reset_sync_id=True)
        c2 = _create_projection_pipeline(
            proj.k2.base_m, BN2, BK2, KL1_2, SKL1_2, l0b_depth=2
        )
        _bind_projection_pipeline(proj.k2, c2)
        for it in range(block, proj.m_tiles * proj.n2_blocks, nblk):
            proj.k2._compute_output_tile(
                qfull_gm,
                qn8_gm,
                query_up_proj_weight_gm,
                sqn_gm,
                query_up_proj_scale_gm,
                it // proj.n2_blocks,
                it % proj.n2_blocks,
                None,
            )
        if const_expr(
            self.mode != "proj"
            and self.cmp_decode_indexed
            and self.cmp_p3_items > self.cmp_p1_items
        ):
            # AIC执行WqB期间，AIV从独立的压缩cache读取并写最终kg压缩区。
            # helper排空MTE3后进入已有B3，QK仍在B4之后读取kg。
            self.compressed_cache_early.run(
                ckc_gm,
                extra_indices_in_kvcache_gm,
                cmp_lengths_gm,
                zero_gm,
                kg_gm,
                block * 2 + get_subblock_id(),
                nblk * 2,
                extra_block_table_gm,
            )

    @jit
    def _rms_norm_rope_quantize_window_cache(
        self,
        kv_gm,
        gamma_gm,
        rope_cos_gm,
        rope_sin_gm,
        window_cache_slots_gm,
        cache_gm,
        block,
        nblk,
    ):
        """AIV1 owns a token's KV norm -> RoPE -> 544-byte cache record in UB."""
        input_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, D)))
        gamma_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, D)))
        norm_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, D)))
        norm_bf16_round_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, D)))
        norm_fp32_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, D)))
        rope_input_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 64)))
        cos_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 64)))
        sin_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 64)))
        record_ub = Buffer(MemLoc.UB, dtypes.uint8, make_layout((1, 544)))
        scale_ub = record_ub.reinterpret(dtypes.int16, make_layout((1, 272)))
        ps = self.attn.page_size
        mem_copy(gamma_ub, tile_view(gamma_gm, (1, D), (0, 0)))
        for tok in range(block, self.tokens, nblk):
            mem_copy(
                input_ub,
                tile_view(kv_gm, (1, D), (tok, 0)),
                options=copy_options(l2_cache_ctl=L2_CACHE_DISABLE),
            )
            mem_copy(
                tile_view(cos_ub, (1, 32), (0, 0)),
                tile_view(rope_cos_gm, (1, 32), (tok, 0)),
            )
            mem_copy(
                tile_view(sin_ub, (1, 32), (0, 0)),
                tile_view(rope_sin_gm, (1, 32), (tok, 0)),
            )
            vec_sync_all()
            # One VF owns the complete UB chain. Each arithmetic fragment
            # keeps its VST->VLD barrier, including the explicit BF16 stages.
            with vf(mode="simd"):
                self.proj._rms_norm_key_value_row_vf(
                    input_ub, gamma_ub, norm_ub, norm_bf16_round_ub, norm_fp32_ub, D
                )
                full = rr.full_mask()
                tail = rr.vcast(
                    rr.vload_unpack(
                        norm_ub, D - 64, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full,
                )
                rr.vstore(rope_input_ub, 0, tail, full)
                rr.vmem_bar("vst_vld")
                self.rope_chunk._apply_rotary_embedding_vf(
                    rope_input_ub, cos_ub, sin_ub, norm_ub, D - 64
                )
                # Offsets are elements of the owning BF16/int16 allocation.
                # The record remains one contiguous 512 B + 32 B UB buffer.
                self._quantize_window_cache_row_vf(norm_ub, record_ub, scale_ub, D // 2)
            vec_sync_all()
            slot = dtypes.int64(window_cache_slots_gm[tok])
            if slot >= ps:
                page = dtypes.int64(slot // ps)
                offset = dtypes.int64(slot % ps)
                mem_copy(
                    tile_view(cache_gm, (1, 544), (page, offset)),
                    record_ub.reinterpret(dtypes.int8),
                )
            # Drain the record write before its UB is reused for another token.
            vec_sync_all()

    @jit
    def _decode_full_compressed_cache(self, data_gm, dst_gm, block, nblk):
        # Native FP4 conversion with stored BF16 scales, 2*nblk AIV owners. RingBuffer edges
        # replace per-row sync_all; the existing B1 barrier drains the ring.
        ps = self.compressed_page_size
        if const_expr(self.cmp_decode_depth > 0):
            fp4_ub = RingBuffer(
                MemLoc.UB,
                dtypes.uint8,
                make_layout((1, 384)),
                depth=self.cmp_decode_depth,
            )
            dst_ub = RingBuffer(
                MemLoc.UB,
                dtypes.bfloat16,
                make_layout((1, D)),
                depth=self.cmp_decode_depth,
            )
        else:
            fp4_ub = Buffer(MemLoc.UB, dtypes.uint8, make_layout((1, 384)))
            dst_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, D)))
        vec_id = block * 2 + get_subblock_id()
        vec_num = nblk * 2
        for flat in range(vec_id, data_gm.shape[0] * ps, vec_num):
            page = dtypes.int64(flat // ps)
            off = dtypes.int64(flat % ps)
            if const_expr(self.cmp_decode_depth > 0):
                fp4_write = fp4_ub.next_write()
            else:
                fp4_write = fp4_ub
            mem_copy(
                tile_view(fp4_write, (1, 320), (0, 0)),
                tile_view(data_gm[page, None, None], (1, 320), (off, 0)),
            )
            if self.cmp_decode_depth == 0:
                vec_sync_all()
            if const_expr(self.cmp_decode_depth > 0):
                fp4_read = fp4_ub.next_read()
                dst_write = dst_ub.next_write()
            else:
                fp4_read = fp4_ub
                dst_write = dst_ub
            self.cmp_decoder.decode(fp4_read, dst_write, 0)
            if self.cmp_decode_depth == 0:
                vec_sync_all()
            if const_expr(self.cmp_decode_depth > 0):
                dst_read = dst_ub.next_read()
            else:
                dst_read = dst_ub
            mem_copy(tile_view(dst_gm[page, None, None], (1, D), (off, 0)), dst_read)
            if self.cmp_decode_depth == 0:
                vec_sync_all()
        # The caller's existing global barrier drains the channel pipelines.

    @jit
    def _scale_bf16_query_row(self, x_ub, y_gm, r, n, mul_const):
        nc = n // VECTOR_CHUNK_ELEMENTS
        y_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, n)))
        with vf(mode="simd"):
            full = rr.full_mask()
            cmul = rr.vdups(mul_const, dtypes.float32, mask=full)
            for c in range(nc):
                off = c * VECTOR_CHUNK_ELEMENTS
                x = rr.vcast(
                    rr.vload_unpack(x_ub, off, unpack_mode=rr.UnpackMode.B16_TO_B32),
                    dtypes.float32,
                    mask=full,
                )
                x = rr.vmul(x, cmul, mask=full)
                rr.vstore_pack(
                    y_ub,
                    off,
                    rr.vcast(x, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ZERO),
                    full,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            rr.vmem_bar("vst_vld")
        mem_copy(tile_view(y_gm, (1, n), (r, 0)), y_ub)

    @jit
    def _quantize_window_cache_row_vf(self, x_ub, data_ub, scale_ub, scale_offset=0):
        """Full-row E4M3/BF16-scale arithmetic inside an active VF region."""
        full = rr.full_mask()
        low = rr.create_mask("vl32", elem_bits=32)
        high = rr.mask_not(low, exec_mask=full)
        zero = rr.vdups(0.0, dtypes.float32, mask=full)
        one = rr.vdups(1, dtypes.int32, mask=full)
        no = rr.vdups(0, dtypes.int32, mask=full)
        bias_twice = rr.vdups(254, dtypes.int32, mask=full)
        for c in range(8):
            x = rr.vcast(
                rr.vload_unpack(x_ub, c * 64, unpack_mode=rr.UnpackMode.B16_TO_B32),
                dtypes.float32,
                mask=full,
            )
            x0 = rr.vselect(zero, x, cond_mask=high)
            x1 = rr.vselect(zero, x, cond_mask=low)
            a0 = rr.vreduce_max(rr.vabs(x0, mask=full), mask=full)
            a1 = rr.vreduce_max(rr.vabs(x1, mask=full), mask=full)
            e0 = rr.vreinterpret(
                rr.vshr(rr.vreinterpret(a0, dtypes.uint32), 23, mask=full), dtypes.int32
            )
            e1 = rr.vreinterpret(
                rr.vshr(rr.vreinterpret(a1, dtypes.uint32), 23, mask=full), dtypes.int32
            )
            b0 = rr.vmaxs(rr.vadds(e0, -8, mask=full), 114, mask=full)
            b1 = rr.vmaxs(rr.vadds(e1, -8, mask=full), 114, mask=full)
            t0 = rr.vmuls(
                rr.vreinterpret(rr.vshl(b0, 23, mask=full), dtypes.float32),
                448.0,
                mask=full,
            )
            t1 = rr.vmuls(
                rr.vreinterpret(rr.vshl(b1, 23, mask=full), dtypes.float32),
                448.0,
                mask=full,
            )
            b0 = rr.vadd(
                b0, rr.vselect(one, no, cond_mask=rr.vgt(a0, t0, mask=full)), mask=full
            )
            b1 = rr.vadd(
                b1, rr.vselect(one, no, cond_mask=rr.vgt(a1, t1, mask=full)), mask=full
            )
            inv0 = rr.vdup(
                rr.vreinterpret(
                    rr.vshl(rr.vsub(bias_twice, b0, mask=full), 23, mask=full),
                    dtypes.float32,
                ),
                mask=full,
            )
            inv1 = rr.vdup(
                rr.vreinterpret(
                    rr.vshl(rr.vsub(bias_twice, b1, mask=full), 23, mask=full),
                    dtypes.float32,
                ),
                mask=full,
            )
            y = rr.vmul(x, rr.vselect(inv1, inv0, cond_mask=high), mask=full)
            y = rr.vmins(rr.vmaxs(y, -448.0, mask=full), 448.0, mask=full)
            rr.vstore_pack(
                data_ub,
                c * 64,
                rr.vcast(y, dtypes.float8_e4m3fn, mask=full, reg_layout=RegLayout.ZERO),
                full,
                pack_mode=rr.PackMode.B32_TO_B8,
            )
            rr.vstore_first(
                scale_ub,
                scale_offset + c * 2,
                rr.vreinterpret_lanes(rr.vshl(b0, 7, mask=full), dtypes.int16),
            )
            rr.vstore_first(
                scale_ub,
                scale_offset + c * 2 + 1,
                rr.vreinterpret_lanes(rr.vshl(b1, 7, mask=full), dtypes.int16),
            )
        rr.vmem_bar("vst_vld")

    @jit
    def _quantize_window_cache_row(self, x_ub, data_ub, scale_ub):
        """512 BF16 -> E4M3 plus 16 numeric BF16 power-of-two scales."""
        with vf(mode="simd"):
            self._quantize_window_cache_row_vf(x_ub, data_ub, scale_ub)

    @jit
    def _quantize_window_cache_chunk(self, x_ub, data_ub, scale_ub):
        """One 64-value cache chunk: 64 E4M3 bytes and two BF16 scales."""
        with vf(mode="simd"):
            full = rr.full_mask()
            low = rr.create_mask("vl32", elem_bits=32)
            high = rr.mask_not(low, exec_mask=full)
            zero = rr.vdups(0.0, dtypes.float32, mask=full)
            one = rr.vdups(1, dtypes.int32, mask=full)
            no = rr.vdups(0, dtypes.int32, mask=full)
            bias_twice = rr.vdups(254, dtypes.int32, mask=full)
            x = rr.vcast(
                rr.vload_unpack(x_ub, 0, unpack_mode=rr.UnpackMode.B16_TO_B32),
                dtypes.float32,
                mask=full,
            )
            x0 = rr.vselect(zero, x, cond_mask=high)
            x1 = rr.vselect(zero, x, cond_mask=low)
            a0 = rr.vreduce_max(rr.vabs(x0, mask=full), mask=full)
            a1 = rr.vreduce_max(rr.vabs(x1, mask=full), mask=full)
            e0 = rr.vreinterpret(
                rr.vshr(rr.vreinterpret(a0, dtypes.uint32), 23, mask=full), dtypes.int32
            )
            e1 = rr.vreinterpret(
                rr.vshr(rr.vreinterpret(a1, dtypes.uint32), 23, mask=full), dtypes.int32
            )
            b0 = rr.vmaxs(rr.vadds(e0, -8, mask=full), 114, mask=full)
            b1 = rr.vmaxs(rr.vadds(e1, -8, mask=full), 114, mask=full)
            t0 = rr.vmuls(
                rr.vreinterpret(rr.vshl(b0, 23, mask=full), dtypes.float32),
                448.0,
                mask=full,
            )
            t1 = rr.vmuls(
                rr.vreinterpret(rr.vshl(b1, 23, mask=full), dtypes.float32),
                448.0,
                mask=full,
            )
            b0 = rr.vadd(
                b0, rr.vselect(one, no, cond_mask=rr.vgt(a0, t0, mask=full)), mask=full
            )
            b1 = rr.vadd(
                b1, rr.vselect(one, no, cond_mask=rr.vgt(a1, t1, mask=full)), mask=full
            )
            inv0 = rr.vdup(
                rr.vreinterpret(
                    rr.vshl(rr.vsub(bias_twice, b0, mask=full), 23, mask=full),
                    dtypes.float32,
                ),
                mask=full,
            )
            inv1 = rr.vdup(
                rr.vreinterpret(
                    rr.vshl(rr.vsub(bias_twice, b1, mask=full), 23, mask=full),
                    dtypes.float32,
                ),
                mask=full,
            )
            y = rr.vmul(x, rr.vselect(inv1, inv0, cond_mask=high), mask=full)
            y = rr.vmins(rr.vmaxs(y, -448.0, mask=full), 448.0, mask=full)
            rr.vstore_pack(
                data_ub,
                0,
                rr.vcast(y, dtypes.float8_e4m3fn, mask=full, reg_layout=RegLayout.ZERO),
                full,
                pack_mode=rr.PackMode.B32_TO_B8,
            )
            # E8M0 byte b encodes 2**(b-127).  Its BF16 bits are b << 7.
            rr.vstore_first(
                scale_ub,
                0,
                rr.vreinterpret_lanes(rr.vshl(b0, 7, mask=full), dtypes.int16),
            )
            rr.vstore_first(
                scale_ub,
                1,
                rr.vreinterpret_lanes(rr.vshl(b1, 7, mask=full), dtypes.int16),
            )
            rr.vmem_bar("vst_vld")

    @jit
    def _dequantize_window_cache_row(self, data_ub, dst_ub, row, base):
        """E4M3 times stored BF16 scale -> BF16, one selected cache row."""
        scale_data = data_ub.reinterpret(
            dtypes.bfloat16, make_layout((data_ub.shape[0], data_ub.shape[1] // 2))
        )
        with vf(mode="simd"):
            full = rr.full_mask()
            zero_u = rr.vdups(0, dtypes.uint32, mask=full)
            one_u = rr.vdups(1, dtypes.uint32, mask=full)
            scale_values = rr.vcast(
                rr.vload_unpack(
                    scale_data,
                    row * (data_ub.shape[1] // 2) + (base + 512) // 2,
                    unpack_mode=rr.UnpackMode.B16_TO_B32,
                ),
                dtypes.float32,
                mask=full,
            )
            for c in range(8):
                bits = rr.vload_unpack(
                    data_ub,
                    row * data_ub.shape[1] + base + c * 64,
                    unpack_mode=rr.UnpackMode.B8_TO_B32,
                )
                sign = rr.vbitwise_and(rr.vshr(bits, 7, mask=full), one_u, mask=full)
                exponent = rr.vbitwise_and(
                    rr.vshr(bits, 3, mask=full),
                    rr.vdups(15, dtypes.uint32, mask=full),
                    mask=full,
                )
                mantissa = rr.vbitwise_and(
                    bits, rr.vdups(7, dtypes.uint32, mask=full), mask=full
                )
                normal = rr.vmul(
                    rr.vreinterpret(
                        rr.vshl(
                            rr.vadd(
                                exponent,
                                rr.vdups(120, dtypes.uint32, mask=full),
                                mask=full,
                            ),
                            23,
                            mask=full,
                        ),
                        dtypes.float32,
                    ),
                    rr.vadd(
                        rr.vdups(1.0, dtypes.float32, mask=full),
                        rr.vmuls(
                            rr.vcast(
                                rr.vreinterpret(mantissa, dtypes.int32),
                                dtypes.float32,
                                mask=full,
                            ),
                            0.125,
                            mask=full,
                        ),
                        mask=full,
                    ),
                    mask=full,
                )
                sub = rr.vmuls(
                    rr.vcast(
                        rr.vreinterpret(mantissa, dtypes.int32),
                        dtypes.float32,
                        mask=full,
                    ),
                    1.0 / 512.0,
                    mask=full,
                )
                val = rr.vselect(
                    sub, normal, cond_mask=rr.veq(exponent, zero_u, mask=full)
                )
                val = rr.vselect(
                    rr.vmuls(val, -1.0, mask=full),
                    val,
                    cond_mask=rr.veq(sign, one_u, mask=full),
                )
                s0 = rr.vreinterpret(
                    rr.vgather_reg(
                        rr.vreinterpret(scale_values, dtypes.uint32),
                        rr.vdups(c * 2, dtypes.uint32, mask=full),
                    ),
                    dtypes.float32,
                )
                s1 = rr.vreinterpret(
                    rr.vgather_reg(
                        rr.vreinterpret(scale_values, dtypes.uint32),
                        rr.vdups(c * 2 + 1, dtypes.uint32, mask=full),
                    ),
                    dtypes.float32,
                )
                half = rr.create_mask("vl32", elem_bits=32)
                high = rr.mask_not(half, exec_mask=full)
                value = rr.vmul(val, rr.vselect(s1, s0, cond_mask=high), mask=full)
                rr.vstore_pack(
                    dst_ub,
                    row * D + c * 64,
                    rr.vcast(
                        value, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ZERO
                    ),
                    full,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def _align_window_cache_record(self, raw_ub, aligned_ub, row, base):
        # A 544-byte record is 32-byte aligned: copy 512 data bytes and
        # 32 BF16-scale bytes into the padded UB row.
        with vf(mode="simd"):
            mask_all = rr.full_mask(elem_bits=8)
            mask_scale = rr.create_mask("vl32", elem_bits=8)
            src = row * raw_ub.shape[1] + base
            dst = row * aligned_ub.shape[1]
            rr.vload_unalign_init(raw_ub, src)
            r0 = rr.vload_unalign(raw_ub, src)
            rr.vstore(aligned_ub, dst, r0, mask_all)
            rr.vload_unalign_init(raw_ub, src + 256)
            r1 = rr.vload_unalign(raw_ub, src + 256)
            rr.vstore(aligned_ub, dst + 256, r1, mask_all)
            rr.vload_unalign_init(raw_ub, src + 512)
            rs = rr.vload_unalign(raw_ub, src + 512)
            rr.vstore(aligned_ub, dst + 512, rs, mask_scale)
            rr.vmem_bar("vst_vld")

    @jit
    def _update_window_cache(
        self,
        kvn_gm,
        rope_cos_gm,
        rope_sin_gm,
        window_cache_slots_gm,
        wkc_gm,
        wvc_gm,
        wsc_gm,
        block,
        nblk,
    ):
        """Distribute (token, 64-feature chunk) tasks across the actual AIV count."""
        d = self.attn.d
        rdim = self.rope.rope_dim
        ps = self.attn.page_size
        # Full-register UB allocation; only 32 coefficients cross GM -> UB.
        cos_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, rdim)))
        sin_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, rdim)))
        kvb_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, 64)))
        kv_f_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 64)))
        kv_rot_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, 64)))
        data_ub = Buffer(MemLoc.UB, dtypes.int8, make_layout((1, 64)))
        scale_ub = Buffer(MemLoc.UB, dtypes.int16, make_layout((1, 16)))
        scale_words = scale_ub.reinterpret(dtypes.uint32, make_layout((1, 8)))
        data_words = data_ub.reinterpret(dtypes.uint64, make_layout((1, 8)))
        vec_id = block * 2 + get_subblock_id()
        vec_num = nblk * 2
        for item in range(vec_id, self.tokens * (d // 64), vec_num):
            tok = item // (d // 64)
            chunk = item % (d // 64)
            mem_copy(
                kvb_ub,
                tile_view(kvn_gm, (1, 64), (tok, chunk)),
                options=copy_options(l2_cache_ctl=L2_CACHE_DISABLE),
            )
            if chunk == (d // 64) - 1:
                mem_copy(
                    tile_view(cos_ub, (1, 32), (0, 0)),
                    tile_view(rope_cos_gm, (1, 32), (tok, 0)),
                )
                mem_copy(
                    tile_view(sin_ub, (1, 32), (0, 0)),
                    tile_view(rope_sin_gm, (1, 32), (tok, 0)),
                )
            vec_sync_all()
            if chunk == (d // 64) - 1:
                self.rope_chunk._upcast_bf16_input(kvb_ub, kv_f_ub)
                self.rope_chunk._apply_rotary_embedding(
                    kv_f_ub, cos_ub, sin_ub, kv_rot_ub
                )
                vec_sync_all()
                self._quantize_window_cache_chunk(kv_rot_ub, data_ub, scale_ub)
            else:
                self._quantize_window_cache_chunk(kvb_ub, data_ub, scale_ub)
            vec_sync_all()
            slot = dtypes.int64(window_cache_slots_gm[tok])
            if slot >= ps:
                blk = dtypes.int64(slot // ps)
                off = dtypes.int64(slot % ps)
                if const_expr(wkc_gm.stride[0] % 8 == 0):
                    for word_id in range(8):
                        vec_store_bypass(
                            wvc_gm.ptr((blk, off * 68 + chunk * 8 + word_id)),
                            data_words[0, word_id],
                        )
                    # One aligned store writes the two numeric BF16 scales.
                    scale_off = off * ((d + 32) // 4) + d // 4 + chunk
                    vec_store_bypass(wsc_gm.ptr((blk, scale_off)), scale_words[0, 0])
                else:
                    # Byte DMA also supports a page pitch not divisible by 8;
                    # do not form an invalid uint64/uint32 GM dtype view.
                    payload = off * 544 + chunk * 64
                    scale_offset = off * 544 + 512 + chunk * 4
                    mem_copy(wkc_gm[blk : blk + 1, payload : payload + 64], data_ub)
                    mem_copy(
                        wkc_gm[blk : blk + 1, scale_offset : scale_offset + 4],
                        scale_ub.reinterpret(dtypes.int8, make_layout((1, 32)))[
                            0:1, 0:4
                        ],
                    )
            vec_sync_all()

    @jit
    def _apply_query_rope_scale_paired(
        self, src, scratch, cos_ub, sin_ub, dst, d, scale
    ):
        # D=512, RoPE=64 specialization. Preserve BF16 rounding after RoPE
        # and again after scaling, matching the baseline exactly.
        prefix = d - self.rope.rope_dim
        with vf(mode="simd"):
            full = rr.full_mask()
            for c in range(prefix // (2 * VECTOR_CHUNK_ELEMENTS)):
                x0 = rr.vload_unpack(
                    src,
                    2 * c * VECTOR_CHUNK_ELEMENTS,
                    unpack_mode=rr.UnpackMode.B16_TO_B32,
                )
                x1 = rr.vload_unpack(
                    src,
                    (2 * c + 1) * VECTOR_CHUNK_ELEMENTS,
                    unpack_mode=rr.UnpackMode.B16_TO_B32,
                )
                f0 = rr.vcast(x0, dtypes.float32, mask=full)
                f1 = rr.vcast(x1, dtypes.float32, mask=full)
                f0 = rr.vmuls(f0, scale, mask=full)
                f1 = rr.vmuls(f1, scale, mask=full)
                b0 = rr.vcast(f0, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ZERO)
                b1 = rr.vcast(f1, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ZERO)
                rr.vstore_pack(
                    dst,
                    2 * c * VECTOR_CHUNK_ELEMENTS,
                    b0,
                    full,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
                rr.vstore_pack(
                    dst,
                    (2 * c + 1) * VECTOR_CHUNK_ELEMENTS,
                    b1,
                    full,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            if (prefix // VECTOR_CHUNK_ELEMENTS) % 2:
                offset = prefix - VECTOR_CHUNK_ELEMENTS
                x = rr.vcast(
                    rr.vload_unpack(src, offset, unpack_mode=rr.UnpackMode.B16_TO_B32),
                    dtypes.float32,
                    mask=full,
                )
                b = rr.vcast(
                    rr.vmuls(x, scale, mask=full),
                    dtypes.bfloat16,
                    mask=full,
                    reg_layout=RegLayout.ZERO,
                )
                rr.vstore_pack(dst, offset, b, full, pack_mode=rr.PackMode.B32_TO_B16)
            for c in range(self.rope.rope_dim // VECTOR_CHUNK_ELEMENTS):
                x = rr.vcast(
                    rr.vload_unpack(
                        src,
                        prefix + c * VECTOR_CHUNK_ELEMENTS,
                        unpack_mode=rr.UnpackMode.B16_TO_B32,
                    ),
                    dtypes.float32,
                    mask=full,
                )
                rr.vstore(scratch, c * VECTOR_CHUNK_ELEMENTS, x, full)
            rr.vmem_bar("vst_vld")
        with vf(mode="simd"):
            full = rr.full_mask()
            mask64 = rr.create_mask("vl64", elem_bits=16)
            one = rr.vdups(1, dtypes.uint32, mask=full)
            rnd = rr.vdups(0x7FFF, dtypes.uint32, mask=full)
            hi = rr.vdups(0xFFFF0000, dtypes.uint32, mask=full)

            def round_to_bf16_bits(v):
                b = rr.vreinterpret(v, dtypes.uint32)
                t = rr.vadd(
                    rr.vadd(b, rnd, mask=full),
                    rr.vbitwise_and(rr.vshr(b, 16, mask=full), one, mask=full),
                    mask=full,
                )
                return rr.vbitwise_and(t, hi, mask=full)

            for c in range(self.rope.rope_dim // VECTOR_CHUNK_ELEMENTS):
                even, odd = rr.vload_deinterleave(
                    scratch, c * VECTOR_CHUNK_ELEMENTS, width="b32"
                )
                co = rr.vload(cos_ub, c * (VECTOR_CHUNK_ELEMENTS // 2))
                si = rr.vload(sin_ub, c * (VECTOR_CHUNK_ELEMENTS // 2))
                ec = rr.vmul(even, co, mask=full)
                oc = rr.vmul(odd, co, mask=full)
                es = rr.vmul(even, si, mask=full)
                os = rr.vmul(odd, si, mask=full)
                oe = rr.vadd(ec, rr.vmuls(os, -1.0, mask=full), mask=full)
                oo = rr.vsub(oc, rr.vmuls(es, -1.0, mask=full), mask=full)
                # Preserve the intermediate BF16 quantization before scale.
                oe = rr.vmuls(
                    rr.vreinterpret(round_to_bf16_bits(oe), dtypes.float32),
                    scale,
                    mask=full,
                )
                oo = rr.vmuls(
                    rr.vreinterpret(round_to_bf16_bits(oo), dtypes.float32),
                    scale,
                    mask=full,
                )
                packed = rr.vbitwise_or(
                    rr.vshr(round_to_bf16_bits(oe), 16, mask=full),
                    round_to_bf16_bits(oo),
                    mask=full,
                )
                rr.vstore(
                    dst,
                    prefix + c * VECTOR_CHUNK_ELEMENTS,
                    rr.vreinterpret_lanes(packed, dtypes.bfloat16),
                    mask64,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def _prepare_query_and_gather_kv(
        self,
        qfull_gm,
        rope_cos_gm,
        rope_sin_gm,
        wkc_gm,
        wvc_gm,
        block_table_gm,
        ckc_gm,
        wsc_gm,
        extra_block_table_gm,
        cmp_dec_gm,
        indices_in_kvcache_gm,
        extra_indices_in_kvcache_gm,
        topk_length_gm,
        extra_topk_length_gm,
        zero_gm,
        qb_gm,
        kg_gm,
        block,
        nblk,
    ):
        attn, rope = self.attn, self.rope
        d, k, ps, _bn = attn.d, attn.k, attn.page_size, attn.bn
        gather_k = attn.win_len if self.cmp_decode_indexed else k
        rdim = self.rope.rope_dim
        batch = 16 if self.qkv_packed else self.gather_batch
        cos_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, rdim)))
        sin_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, rdim)))
        if const_expr(self.qkv_qpipe):
            qb_ub = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, d)), depth=2)
        else:
            qb_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, d)))
        q_f_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, d)))
        if const_expr(self.qkv_qpipe):
            qb_rot_ub = RingBuffer(
                MemLoc.UB, dtypes.bfloat16, make_layout((1, d)), depth=2
            )
        else:
            qb_rot_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, d)))
        if const_expr(self.qkv_packed):
            kb_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((batch, d)))
        else:
            kb_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((batch, d)))
        if const_expr(self.qkv_packed):
            # GM records are 544 bytes; pad UB for the full BF16 unpack load.
            fp8_raw_ub = RingBuffer(
                MemLoc.UB, dtypes.int8, make_layout((1, 640)), depth=2
            )
            fp8_ub = Buffer(MemLoc.UB, dtypes.int8, make_layout((1, 640)))
            if const_expr(not self.cmp_decode_indexed):
                cmp_rows_ub = Buffer(
                    MemLoc.UB, dtypes.bfloat16, make_layout((batch, d))
                )
                cmp_rows_ub_1 = Buffer(
                    MemLoc.UB, dtypes.bfloat16, make_layout((batch, d))
                )
        else:
            fp8_raw_ub = Buffer(MemLoc.UB, dtypes.int8, make_layout((batch, 800)))
            fp8_ub = Buffer(MemLoc.UB, dtypes.int8, make_layout((batch, 640)))
            cmp_rows_ub = kb_ub  # Unused fallback alias; no additional UB allocation.
            cmp_rows_ub_1 = kb_ub
        win_idx_ub = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, attn.win_len)))
        if const_expr(not self.cmp_decode_indexed):
            cmp_idx_ub = Buffer(
                MemLoc.UB, dtypes.int32, make_layout((1, k - attn.win_len))
            )
        else:
            cmp_idx_ub = win_idx_ub  # Indexed helper owns compressed-index reads.
        inv_sqrt_d = 1.0 / (d**0.5)
        # Small T uses both physical AIVs with unique row ownership. Larger
        # T retains the existing token-local schedule and phase barrier.
        vec_id = block
        vec_num = nblk
        if self.dual_aiv:
            vec_id = block * 2 + get_subblock_id()
            vec_num = nblk * 2
        tok_begin = 0
        tok_end = self.tokens
        tok_stride = 1
        part = vec_id
        parts = vec_num
        if (
            self.dual_aiv
            and (2 * self.grid) % self.tokens != 0
            and self.tokens < vec_num
        ):
            # Balanced contiguous owner groups also cover T that does not
            # divide the AIV count. Boundaries are ceil(token*V/T): each AIV
            # visits one token, with no duplicated or missing Q/KV row.
            tok_begin = vec_id * self.tokens // vec_num
            tok_end = tok_begin + 1
            owner_begin = (tok_begin * vec_num + self.tokens - 1) // self.tokens
            owner_end = (tok_end * vec_num + self.tokens - 1) // self.tokens
            part = vec_id - owner_begin
            parts = owner_end - owner_begin
        elif self.tokens < vec_num and vec_num % self.tokens == 0:
            blocks_per_token = vec_num // self.tokens
            tok_begin = vec_id // blocks_per_token
            tok_end = tok_begin + 1
            part = vec_id % blocks_per_token
            parts = blocks_per_token
        elif self.tokens >= vec_num:
            tok_begin = vec_id
            tok_stride = vec_num
            part = 0
            parts = 1
        for tok in range(tok_begin, tok_end, tok_stride):
            mem_copy(
                tile_view(cos_ub, (1, 32), (0, 0)),
                tile_view(rope_cos_gm, (1, 32), (tok, 0)),
            )
            mem_copy(
                tile_view(sin_ub, (1, 32), (0, 0)),
                tile_view(rope_sin_gm, (1, 32), (tok, 0)),
            )
            qb_t = tile_view(
                qb_gm.view(self.tokens * self.heads, d), (self.heads, d), (tok, 0)
            )
            kg_t = tile_view(kg_gm.view(self.tokens * k, d), (k, d), (tok, 0))
            mem_copy(
                win_idx_ub,
                tile_view(indices_in_kvcache_gm, (1, attn.win_len), (tok, 0)),
            )
            if const_expr(not self.cmp_decode_indexed):
                mem_copy(
                    cmp_idx_ub,
                    tile_view(
                        extra_indices_in_kvcache_gm, (1, k - attn.win_len), (tok, 0)
                    ),
                )
            vec_sync_all()
            if self.win_length_stride:
                win_valid = dtypes.int32(topk_length_gm[tok // self.win_length_stride])
                _mask_index_prefix(win_idx_ub, attn.win_len, win_valid)
            if self.cmp_length_stride and not self.cmp_decode_indexed:
                cmp_valid = dtypes.int32(
                    extra_topk_length_gm[tok // self.cmp_length_stride]
                )
                _mask_index_prefix(cmp_idx_ub, k - attn.win_len, cmp_valid)
            if self.win_length_stride or self.cmp_length_stride:
                vec_sync_all()
            # (6) Q partial RoPE + 1/sqrt(d) -> bf16 q (no activation quant)
            for row in range(part, self.heads, parts):
                if const_expr(self.qkv_qpipe):
                    qb_write = qb_ub.next_write()
                else:
                    qb_write = qb_ub
                mem_copy(
                    qb_write,
                    tile_view(qfull_gm, (1, d), (tok, dtypes.int64(row))),
                    options=copy_options(l2_cache_ctl=L2_CACHE_DISABLE),
                )
                if not self.qkv_qpipe:
                    vec_sync_all()
                if const_expr(self.qkv_qpipe):
                    qb_read = qb_ub.next_read()
                    qb_rot_write = qb_rot_ub.next_write()
                else:
                    qb_read = qb_ub
                    qb_rot_write = qb_rot_ub
                if self.dual_aiv and d == 512 and rdim == 64:
                    self._apply_query_rope_scale_paired(
                        qb_read, q_f_ub, cos_ub, sin_ub, qb_rot_write, d, inv_sqrt_d
                    )
                    if not self.qkv_qpipe:
                        vec_sync_all()
                    if const_expr(self.qkv_qpipe):
                        qb_rot_read = qb_rot_ub.next_read()
                    else:
                        qb_rot_read = qb_rot_ub
                    mem_copy(tile_view(qb_t, (1, d), (row, 0)), qb_rot_read)
                else:
                    rope._upcast_bf16_input(qb_read, q_f_ub)
                    rope._apply_rotary_embedding(q_f_ub, cos_ub, sin_ub, qb_rot_write)
                    if const_expr(self.qkv_qpipe):
                        qb_rot_read = qb_rot_ub.next_read()
                    else:
                        qb_rot_read = qb_rot_ub
                    self._scale_bf16_query_row(qb_rot_read, qb_t, row, d, inv_sqrt_d)
                if not self.qkv_qpipe:
                    vec_sync_all()
            if self.qkv_qpipe:
                vec_sync_all()
            # 两路索引分别定位各自cache。提供页表时，按token所属batch
            # 将逻辑页映射到物理页；不提供时直接解释为物理行号。
            # Indexed strategy fills only window rows here; its compressed
            # rows are loaded and decoded by the separate AIV gather helper.
            if self.qkv_packed:
                # Window: MTE2 depth2 aligned record ring -> native decode.
                # One contiguous output batch per AIV at Kw=128.
                for j0 in range(part * batch, attn.win_len, parts * batch):
                    for b in range(batch):
                        index = dtypes.int32(win_idx_ub[0, j0 + b])
                        if index >= 0:
                            page = self.cache_addressing.page(
                                index, block_table_gm, tok
                            )
                            off = dtypes.int64(index % ps)
                            byte_off = off * 544
                            fp8_write = fp8_raw_ub.next_write()
                            mem_copy(
                                tile_view(fp8_write, (1, 544), (0, 0)),
                                wkc_gm[page : page + 1, byte_off : byte_off + 544],
                            )
                            fp8_read = fp8_raw_ub.next_read()
                            self.win_decoder.decode(
                                fp8_read, tile_view(kb_ub, (1, d), (b, 0)), 0, 0
                            )
                        else:
                            mem_copy(tile_view(kb_ub, (1, d), (b, 0)), zero_gm)
                    vec_sync_all()
                    mem_copy(tile_view(kg_t, (batch, d), (j0 // batch, 0)), kb_ub)
                    vec_sync_all()
                if const_expr(not self.cmp_decode_indexed):
                    # Ping-pong ordinary UB batches, including an odd final batch.
                    nbatches = (
                        k - attn.win_len - part * batch + parts * batch - 1
                    ) // (parts * batch)
                    for b in range(batch):
                        index = dtypes.int32(cmp_idx_ub[0, part * batch + b])
                        if index >= 0:
                            page = self.compressed_cache_addressing.page(
                                index, extra_block_table_gm, tok
                            )
                            off = dtypes.int64(index % self.compressed_page_size)
                            mem_copy(
                                tile_view(cmp_rows_ub, (1, d), (b, 0)),
                                tile_view(
                                    cmp_dec_gm[page, None, None], (1, d), (off, 0)
                                ),
                            )
                        else:
                            mem_copy(tile_view(cmp_rows_ub, (1, d), (b, 0)), zero_gm)
                    vec_sync_notify(PIPE.MTE2, PIPE.MTE3, 6)
                    for step in range(nbatches):
                        if step + 1 < nbatches:
                            jnext = part * batch + (step + 1) * parts * batch
                            if step % 2 == 0:
                                if step > 0:
                                    vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 7)
                                for b in range(batch):
                                    index = dtypes.int32(cmp_idx_ub[0, jnext + b])
                                    if index >= 0:
                                        page = self.compressed_cache_addressing.page(
                                            index, extra_block_table_gm, tok
                                        )
                                        off = dtypes.int64(
                                            index % self.compressed_page_size
                                        )
                                        mem_copy(
                                            tile_view(cmp_rows_ub_1, (1, d), (b, 0)),
                                            tile_view(
                                                cmp_dec_gm[page, None, None],
                                                (1, d),
                                                (off, 0),
                                            ),
                                        )
                                    else:
                                        mem_copy(
                                            tile_view(cmp_rows_ub_1, (1, d), (b, 0)),
                                            zero_gm,
                                        )
                                vec_sync_notify(PIPE.MTE2, PIPE.MTE3, 7)
                            else:
                                vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 6)
                                for b in range(batch):
                                    index = dtypes.int32(cmp_idx_ub[0, jnext + b])
                                    if index >= 0:
                                        page = self.compressed_cache_addressing.page(
                                            index, extra_block_table_gm, tok
                                        )
                                        off = dtypes.int64(
                                            index % self.compressed_page_size
                                        )
                                        mem_copy(
                                            tile_view(cmp_rows_ub, (1, d), (b, 0)),
                                            tile_view(
                                                cmp_dec_gm[page, None, None],
                                                (1, d),
                                                (off, 0),
                                            ),
                                        )
                                    else:
                                        mem_copy(
                                            tile_view(cmp_rows_ub, (1, d), (b, 0)),
                                            zero_gm,
                                        )
                                vec_sync_notify(PIPE.MTE2, PIPE.MTE3, 6)
                        j0 = part * batch + step * parts * batch
                        if step % 2 == 0:
                            vec_sync_wait(PIPE.MTE2, PIPE.MTE3, 6)
                            mem_copy(
                                tile_view(
                                    kg_t, (batch, d), ((j0 + attn.win_len) // batch, 0)
                                ),
                                cmp_rows_ub,
                            )
                            vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 6)
                        else:
                            vec_sync_wait(PIPE.MTE2, PIPE.MTE3, 7)
                            mem_copy(
                                tile_view(
                                    kg_t, (batch, d), ((j0 + attn.win_len) // batch, 0)
                                ),
                                cmp_rows_ub_1,
                            )
                            vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 7)
                    vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 6)
                    if nbatches > 1:
                        vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 7)
                    vec_sync_all()
            else:
                j_begin = part * batch if self.qkv_packed else part
                for j0 in range(j_begin, gather_k, parts * batch):
                    for b in range(batch):
                        j = j0 + b if self.qkv_packed else j0 + b * parts
                        if j < gather_k:
                            index = dtypes.int32(0)
                            if j < attn.win_len:
                                index = dtypes.int32(win_idx_ub[0, j])
                            else:
                                index = dtypes.int32(cmp_idx_ub[0, j - attn.win_len])
                            if index >= 0:
                                off = dtypes.int64(index % ps)
                                if j < attn.win_len:
                                    page = self.cache_addressing.page(
                                        index, block_table_gm, tok
                                    )
                                    byte_off = off * 544
                                    base = byte_off % 32
                                    aligned = byte_off - base
                                    mem_copy(
                                        tile_view(fp8_raw_ub, (1, 544), (b, 0)),
                                        wkc_gm[
                                            page : page + 1, aligned : aligned + 544
                                        ],
                                    )
                                else:
                                    page = self.compressed_cache_addressing.page(
                                        index, extra_block_table_gm, tok
                                    )
                                    off = dtypes.int64(
                                        index % self.compressed_page_size
                                    )
                                    mem_copy(
                                        tile_view(kb_ub, (1, d), (b, 0)),
                                        tile_view(
                                            cmp_dec_gm[page, None, None],
                                            (1, d),
                                            (off, 0),
                                        ),
                                    )
                            else:
                                mem_copy(tile_view(kb_ub, (1, d), (b, 0)), zero_gm)
                    vec_sync_all()
                    for b in range(batch):
                        j = j0 + b if self.qkv_packed else j0 + b * parts
                        if j < gather_k:
                            index = dtypes.int32(0)
                            if j < attn.win_len:
                                index = dtypes.int32(win_idx_ub[0, j])
                            else:
                                index = dtypes.int32(cmp_idx_ub[0, j - attn.win_len])
                            if index >= 0:
                                if j < attn.win_len:
                                    base = (dtypes.int64(index % ps) * 544) % 32
                                    self._align_window_cache_record(
                                        fp8_raw_ub, fp8_ub, b, base
                                    )
                                    if not self.qkv_packed:
                                        vec_sync_all()
                                    self._dequantize_window_cache_row(
                                        fp8_ub, kb_ub, b, 0
                                    )
                    vec_sync_all()
                    if self.qkv_packed:
                        mem_copy(tile_view(kg_t, (batch, d), (j0 // batch, 0)), kb_ub)
                    else:
                        for b in range(batch):
                            j = j0 + b * parts
                            if j < gather_k:
                                mem_copy(
                                    tile_view(kg_t, (1, d), (j, 0)),
                                    tile_view(kb_ub, (1, d), (b, 0)),
                                )
                    vec_sync_all()

    @jit
    def _mask_attention_scores(self, s_ub, win_idx_ub, cmp_idx_ub, score_ub, k):
        with vf(mode="simd"):
            full = rr.full_mask()
            zero = rr.vdups(0, dtypes.int32, mask=full)
            neg = rr.vdups(NEG_INF, dtypes.float32, mask=full)
            for c in range(k // VECTOR_CHUNK_ELEMENTS):
                if c * VECTOR_CHUNK_ELEMENTS < self.attn.win_len:
                    idx = rr.vload(win_idx_ub, c * VECTOR_CHUNK_ELEMENTS)
                    invalid = rr.vlt(idx, zero, mask=full)
                    s = rr.vload(s_ub, c * VECTOR_CHUNK_ELEMENTS)
                    rr.vstore(
                        score_ub,
                        c * VECTOR_CHUNK_ELEMENTS,
                        rr.vselect(neg, s, cond_mask=invalid),
                        full,
                    )
                else:
                    idx = rr.vload(
                        cmp_idx_ub, c * VECTOR_CHUNK_ELEMENTS - self.attn.win_len
                    )
                    invalid = rr.vlt(idx, zero, mask=full)
                    s = rr.vload(s_ub, c * VECTOR_CHUNK_ELEMENTS)
                    rr.vstore(
                        score_ub,
                        c * VECTOR_CHUNK_ELEMENTS,
                        rr.vselect(neg, s, cond_mask=invalid),
                        full,
                    )
            rr.vmem_bar("vst_vld")

    @jit
    def _compute_sink_softmax_statistics_online(
        self, score_ub, sink_ub, p_ub, denom_ub
    ):
        """Online (m,l) reduction over K chunks, including the attn_sink logit.

        This is the numerical recurrence only: QK still stages S in GM and
        PV still consumes a completed bf16 P in GM.  The later C/V pipeline
        can consume these same (m,l) states without changing the math.

        UB footprint: score/p = 2*K*4 B, attn_sink/denom = 8 B;
        at K=640 this is 5,128 B, far below the 256 KiB UB limit.
        """
        with vf(mode="simd"):
            full = rr.full_mask()
            m = rr.vload_broadcast(sink_ub, 0)
            sum_exp = rr.vdups(1.0, dtypes.float32, mask=full)
            for c in range(self.attn.k // VECTOR_CHUNK_ELEMENTS):
                s = rr.vload(score_ub, c * VECTOR_CHUNK_ELEMENTS)
                tile_max = rr.vreduce_max(s, mask=full)
                new_m = rr.vdup(rr.vmax(m, tile_max, mask=full), mask=full)
                alpha = rr.vexp_sub(m, new_m, mask=full)
                tile_sum = rr.vreduce_sum(rr.vexp_sub(s, new_m, mask=full), mask=full)
                sum_exp = rr.vadd(
                    rr.vmul(sum_exp, alpha, mask=full), tile_sum, mask=full
                )
                m = new_m
            for c in range(self.attn.k // VECTOR_CHUNK_ELEMENTS):
                p = rr.vexp_sub(
                    rr.vload(score_ub, c * VECTOR_CHUNK_ELEMENTS), m, mask=full
                )
                rr.vstore(p_ub, c * VECTOR_CHUNK_ELEMENTS, p, full)
            rr.vstore_first(denom_ub, 0, sum_exp)
            rr.vmem_bar("vst_vld")

    @jit
    def _run_attention_softmax(
        self,
        s_gm,
        indices_in_kvcache_gm,
        extra_indices_in_kvcache_gm,
        attn_sink_gm,
        p_gm,
        topk_length_gm,
        extra_topk_length_gm,
        block,
        nblk,
    ):
        if self.dual_aiv and self.softmax_batch_rows:
            self._run_softmax_batched_heads(
                s_gm,
                indices_in_kvcache_gm,
                extra_indices_in_kvcache_gm,
                attn_sink_gm,
                p_gm,
                topk_length_gm,
                extra_topk_length_gm,
                block,
                nblk,
            )
        elif self.dual_aiv:
            self._run_softmax_all_aivs(
                s_gm,
                indices_in_kvcache_gm,
                extra_indices_in_kvcache_gm,
                attn_sink_gm,
                p_gm,
                topk_length_gm,
                extra_topk_length_gm,
                block,
                nblk,
            )
        else:
            self._run_softmax_by_token(
                s_gm,
                indices_in_kvcache_gm,
                extra_indices_in_kvcache_gm,
                attn_sink_gm,
                p_gm,
                topk_length_gm,
                extra_topk_length_gm,
                block,
                nblk,
            )

    @jit
    def _compute_masked_sink_softmax_rows(
        self,
        scores_ub,
        window_indices_ub,
        compressed_indices_ub,
        attention_sink_ub,
        masked_scores_ub,
        exponentials_ub,
        probabilities_ub,
        key_count,
        head_start,
        row_count,
    ):
        """Keep mask, FP32 statistics and BF16 normalization in one VF."""
        with vf(mode="simd"):
            for row in range(row_count):
                lane_mask = rr.full_mask()
                zero_index = rr.vdups(0, dtypes.int32, mask=lane_mask)
                negative_logit = rr.vdups(NEG_INF, dtypes.float32, mask=lane_mask)
                row_max = rr.vdups(NEG_INF, dtypes.float32, mask=lane_mask)
                for chunk in range(key_count // VECTOR_CHUNK_ELEMENTS):
                    cache_index = rr.vdups(0, dtypes.int32, mask=lane_mask)
                    if chunk * VECTOR_CHUNK_ELEMENTS < self.attn.win_len:
                        cache_index = rr.vload(
                            window_indices_ub, chunk * VECTOR_CHUNK_ELEMENTS
                        )
                    else:
                        cache_index = rr.vload(
                            compressed_indices_ub,
                            chunk * VECTOR_CHUNK_ELEMENTS - self.attn.win_len,
                        )
                    masked_score = rr.vselect(
                        negative_logit,
                        rr.vload(
                            scores_ub, row * key_count + chunk * VECTOR_CHUNK_ELEMENTS
                        ),
                        cond_mask=rr.vlt(cache_index, zero_index, mask=lane_mask),
                    )
                    rr.vstore(
                        masked_scores_ub,
                        chunk * VECTOR_CHUNK_ELEMENTS,
                        masked_score,
                        lane_mask,
                    )
                    row_max = rr.vmax(
                        row_max,
                        rr.vreduce_max(masked_score, mask=lane_mask),
                        mask=lane_mask,
                    )
                rr.vmem_bar("vst_vld")
                sink = rr.vload_broadcast(attention_sink_ub, head_start + row)
                row_max = rr.vdup(
                    rr.vmax(row_max, sink, mask=lane_mask), mask=lane_mask
                )
                row_sum = rr.vdups(0.0, dtypes.float32, mask=lane_mask)
                for chunk in range(key_count // VECTOR_CHUNK_ELEMENTS):
                    value = rr.vexp_sub(
                        rr.vload(masked_scores_ub, chunk * VECTOR_CHUNK_ELEMENTS),
                        row_max,
                        mask=lane_mask,
                    )
                    rr.vstore(
                        exponentials_ub, chunk * VECTOR_CHUNK_ELEMENTS, value, lane_mask
                    )
                    row_sum = rr.vadd(
                        row_sum, rr.vreduce_sum(value, mask=lane_mask), mask=lane_mask
                    )
                denominator = rr.vdup(
                    rr.vadd(
                        row_sum,
                        rr.vexp_sub(sink, row_max, mask=lane_mask),
                        mask=lane_mask,
                    ),
                    mask=lane_mask,
                )
                rr.vmem_bar("vst_vld")
                for chunk in range(key_count // VECTOR_CHUNK_ELEMENTS):
                    probability = rr.vdiv(
                        rr.vload(exponentials_ub, chunk * VECTOR_CHUNK_ELEMENTS),
                        denominator,
                        mask=lane_mask,
                    )
                    rr.vstore_pack(
                        probabilities_ub,
                        row * key_count + chunk * VECTOR_CHUNK_ELEMENTS,
                        rr.vcast(
                            probability,
                            dtypes.bfloat16,
                            mask=lane_mask,
                            reg_layout=RegLayout.ZERO,
                        ),
                        lane_mask,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )
                rr.vmem_bar("vst_vld")

    @jit
    def _run_softmax_batched_heads(
        self,
        scores_gm,
        window_indices_gm,
        compressed_indices_gm,
        attention_sink_gm,
        probabilities_gm,
        window_topk_length_gm,
        compressed_topk_length_gm,
        block,
        nblk,
    ):
        """One contiguous Score/P transaction per token/head owner."""
        key_count = self.attn.k
        rows_per_owner = self.softmax_batch_rows
        owners_per_token = 2 * self.grid // self.tokens
        vec_id = block * 2 + get_subblock_id()
        token = vec_id // owners_per_token
        owner_index = vec_id % owners_per_token
        head_start = owner_index * rows_per_owner
        row_count = min(rows_per_owner, self.heads - head_start)
        scores_ub = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((rows_per_owner, key_count))
        )
        probabilities_ub = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((rows_per_owner, key_count))
        )
        window_indices_ub = Buffer(
            MemLoc.UB, dtypes.int32, make_layout((1, self.attn.win_len))
        )
        compressed_indices_ub = Buffer(
            MemLoc.UB, dtypes.int32, make_layout((1, key_count - self.attn.win_len))
        )
        attention_sink_ub = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((1, self.heads))
        )
        masked_scores_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((key_count,)))
        exponentials_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((key_count,)))
        if head_start < self.heads:
            token_scores = tile_view(
                scores_gm.view(self.tokens * self.heads, key_count),
                (self.heads, key_count),
                (token, 0),
            )
            token_probabilities = tile_view(
                probabilities_gm.view(self.tokens * self.heads, key_count),
                (self.heads, key_count),
                (token, 0),
            )
            mem_copy(
                scores_ub,
                tile_view(token_scores, (rows_per_owner, key_count), (owner_index, 0)),
                options=copy_options(l2_cache_ctl=L2_CACHE_DISABLE),
            )
            mem_copy(
                window_indices_ub,
                tile_view(window_indices_gm, (1, self.attn.win_len), (token, 0)),
            )
            mem_copy(
                compressed_indices_ub,
                tile_view(
                    compressed_indices_gm,
                    (1, key_count - self.attn.win_len),
                    (token, 0),
                ),
            )
            mem_copy(attention_sink_ub, attention_sink_gm)
            vec_sync_all()
            if self.win_length_stride:
                _mask_index_prefix(
                    window_indices_ub,
                    self.attn.win_len,
                    dtypes.int32(
                        window_topk_length_gm[token // self.win_length_stride]
                    ),
                )
            if self.cmp_length_stride:
                _mask_index_prefix(
                    compressed_indices_ub,
                    key_count - self.attn.win_len,
                    dtypes.int32(
                        compressed_topk_length_gm[token // self.cmp_length_stride]
                    ),
                )
            self._compute_masked_sink_softmax_rows(
                scores_ub,
                window_indices_ub,
                compressed_indices_ub,
                attention_sink_ub,
                masked_scores_ub,
                exponentials_ub,
                probabilities_ub,
                key_count,
                head_start,
                row_count,
            )
            vec_sync_all()
            # DMA bounds follow the UB source; clip the partial head batch explicitly.
            mem_copy(
                tile_view(
                    token_probabilities, (rows_per_owner, key_count), (owner_index, 0)
                ),
                probabilities_ub[:row_count, :key_count],
            )
            vec_sync_all()

    @jit
    def _run_softmax_all_aivs(
        self,
        s_gm,
        indices_in_kvcache_gm,
        extra_indices_in_kvcache_gm,
        attn_sink_gm,
        p_gm,
        topk_length_gm,
        extra_topk_length_gm,
        block,
        nblk,
    ):
        """Distribute T*heads softmax rows over both AIVs of every block."""
        attn = self.attn
        k = attn.k
        s_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, k)))
        win_idx_ub = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, attn.win_len)))
        cmp_idx_ub = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, k - attn.win_len)))
        sink_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 1)))
        score_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((k,)))
        p_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((k,)))
        pb_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, k)))
        vec_id = block * 2 + get_subblock_id()
        vec_num = nblk * 2
        for item in range(vec_id, self.tokens * self.heads, vec_num):
            tok = item // self.heads
            row = item % self.heads
            # tile_slice addresses trailing dimensions; flatten T/H before selecting a token.
            s_tok = tile_view(
                s_gm.view(self.tokens * self.heads, k), (self.heads, k), (tok, 0)
            )
            p_tok = tile_view(
                p_gm.view(self.tokens * self.heads, k), (self.heads, k), (tok, 0)
            )
            mem_copy(
                win_idx_ub,
                tile_view(indices_in_kvcache_gm, (1, attn.win_len), (tok, 0)),
            )
            mem_copy(
                cmp_idx_ub,
                tile_view(extra_indices_in_kvcache_gm, (1, k - attn.win_len), (tok, 0)),
            )
            mem_copy(
                s_ub,
                tile_view(s_tok, (1, k), (row, 0)),
                options=copy_options(l2_cache_ctl=L2_CACHE_DISABLE),
            )
            mem_copy(sink_ub, tile_view(attn_sink_gm, (1, 1), (0, row)))
            vec_sync_all()
            if self.win_length_stride:
                win_valid = dtypes.int32(topk_length_gm[tok // self.win_length_stride])
                _mask_index_prefix(win_idx_ub, attn.win_len, win_valid)
            if self.cmp_length_stride:
                cmp_valid = dtypes.int32(
                    extra_topk_length_gm[tok // self.cmp_length_stride]
                )
                _mask_index_prefix(cmp_idx_ub, k - attn.win_len, cmp_valid)
            self._compute_masked_sink_softmax_rows(
                s_ub, win_idx_ub, cmp_idx_ub, sink_ub, score_ub, p_ub, pb_ub, k, 0, 1
            )
            vec_sync_all()
            mem_copy(tile_view(p_tok, (1, k), (row, 0)), pb_ub)
            vec_sync_all()

    @jit
    def _run_softmax_by_token(
        self,
        s_gm,
        indices_in_kvcache_gm,
        extra_indices_in_kvcache_gm,
        attn_sink_gm,
        p_gm,
        topk_length_gm,
        extra_topk_length_gm,
        block,
        nblk,
    ):
        """Token-wise masked attn_sink softmax from Cube scores to BF16 P[H,K]."""
        attn = self.attn
        k = attn.k
        s_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, k)))
        # One token's indices are reused by several head rows. They have a
        # single UB lifetime, not a FIFO transaction per head consumption.
        win_idx_ub = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, attn.win_len)))
        cmp_idx_ub = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, k - attn.win_len)))
        sink_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 1)))
        score_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((k,)))
        p_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((k,)))
        denom_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 1)))
        pb_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, k)))
        for tok in range(self.tokens):
            # tile_slice addresses trailing dimensions; flatten T/H before selecting a token.
            s_tok = tile_view(
                s_gm.view(self.tokens * self.heads, k), (self.heads, k), (tok, 0)
            )
            p_tok = tile_view(
                p_gm.view(self.tokens * self.heads, k), (self.heads, k), (tok, 0)
            )
            # idx is shared by every head row of a token: load it once.
            mem_copy(
                win_idx_ub,
                tile_view(indices_in_kvcache_gm, (1, attn.win_len), (tok, 0)),
            )
            mem_copy(
                cmp_idx_ub,
                tile_view(extra_indices_in_kvcache_gm, (1, k - attn.win_len), (tok, 0)),
            )
            if self.win_length_stride or self.cmp_length_stride:
                vec_sync_all()
            if self.win_length_stride:
                win_valid = dtypes.int32(topk_length_gm[tok // self.win_length_stride])
                _mask_index_prefix(win_idx_ub, attn.win_len, win_valid)
            if self.cmp_length_stride:
                cmp_valid = dtypes.int32(
                    extra_topk_length_gm[tok // self.cmp_length_stride]
                )
                _mask_index_prefix(cmp_idx_ub, k - attn.win_len, cmp_valid)
            for row in range(block, self.heads, nblk):
                mem_copy(
                    s_ub,
                    tile_view(s_tok, (1, k), (row, 0)),
                    options=copy_options(l2_cache_ctl=L2_CACHE_DISABLE),
                )
                mem_copy(sink_ub, tile_view(attn_sink_gm, (1, 1), (0, row)))
                vec_sync_all()
                self._mask_attention_scores(s_ub, win_idx_ub, cmp_idx_ub, score_ub, k)
                # The recurrence is neutral at the target T=16 shape but
                # regresses T=8 by 3-5%; keep the old two-pass path elsewhere
                # until QK/softmax/PV are converted to a real C/V pipeline.
                if self.tokens == 16:
                    self._compute_sink_softmax_statistics_online(
                        score_ub, sink_ub, p_ub, denom_ub
                    )
                else:
                    attn._compute_sink_softmax_statistics(
                        score_ub, sink_ub, p_ub, denom_ub
                    )
                with vf(mode="simd"):
                    full = rr.full_mask()
                    denom_b = rr.vload_broadcast(denom_ub, 0)
                    for c in range(k // VECTOR_CHUNK_ELEMENTS):
                        p = rr.vdiv(
                            rr.vload(p_ub, c * VECTOR_CHUNK_ELEMENTS),
                            denom_b,
                            mask=full,
                        )
                        rr.vstore_pack(
                            pb_ub,
                            c * VECTOR_CHUNK_ELEMENTS,
                            rr.vcast(
                                p, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ZERO
                            ),
                            full,
                            pack_mode=rr.PackMode.B32_TO_B16,
                        )
                    rr.vmem_bar("vst_vld")
                vec_sync_all()
                mem_copy(tile_view(p_tok, (1, k), (row, 0)), pb_ub)
                vec_sync_all()

    @jit
    def _run_probability_value_matmul(self, out_gm, p_gm, v_gm, block, nblk):
        """BF16 Cube PV O[T,H,D] = P[T,H,K] @ V[T,K,D], all tokens.

        The ``(token, N-tile)`` work is flattened over the whole launch grid
        (``nblk``) instead of the ``n/base_n`` Cube blocks, so every block
        shares the PV work (probe ``probe_pv_owner.py``: nblk=8 -> 95us,
        nblk=32 -> 39us).
        """
        vec_sync_all()
        onchip_rewind(reset_sync_id=True)
        n_tiles_pv = self.pv.n // self.pv.base_n
        if self.tokens <= self.grid and not self.flat_attention_tiles:
            bpt_pv = max(1, self.grid // self.tokens)
            nt_per_pv = (n_tiles_pv + bpt_pv - 1) // bpt_pv
            self.pv.compute_owned_tiles(
                out_gm, p_gm, v_gm, block, self.tokens, bpt_pv, nt_per_pv
            )
        else:
            self.pv.compute_flat_tiles(out_gm, p_gm, v_gm, block, nblk, self.tokens)

    @jit
    def _run_small_k_attention_pipeline(
        self,
        qb_gm,
        kg_gm,
        indices_in_kvcache_gm,
        extra_indices_in_kvcache_gm,
        attn_sink_gm,
        out_gm,
        topk_length_gm,
        extra_topk_length_gm,
        block,
        nblk,
    ):
        """H64 aligned CV template; H8/H16 never enter this branch."""
        if self.tokens == 16 and self.grid == 32:
            pipe = _SmallKAttentionPipeline(self.k, tile_m=self.cv_tile_m)
            tok = block // 2
            hm = block % 2
            q_tok = tile_view(
                qb_gm.view(self.tokens * self.heads, D), (self.heads, D), (tok, 0)
            )
            kv_tok = tile_view(
                kg_gm.view(self.tokens * self.k, D), (self.k, D), (tok, 0)
            )
            win_idx_tok = tile_view(indices_in_kvcache_gm, (1, self.win_len), (tok, 0))
            cmp_idx_tok = tile_view(
                extra_indices_in_kvcache_gm, (1, self.k - self.win_len), (tok, 0)
            )
            out_tok = tile_view(
                out_gm.view(self.tokens * self.heads, D), (self.heads, D), (tok, 0)
            )
            h0 = hm * pipe.tile_m
            win_valid = dtypes.int32(self.win_len)
            cmp_valid = dtypes.int32(self.k - self.win_len)
            if self.win_length_stride:
                win_valid = dtypes.int32(topk_length_gm[tok // self.win_length_stride])
            if self.cmp_length_stride:
                cmp_valid = dtypes.int32(
                    extra_topk_length_gm[tok // self.cmp_length_stride]
                )
            pipe.run_attention_head_tile(
                q_tok[h0 : h0 + pipe.tile_m, 0:D],
                kv_tok,
                win_idx_tok,
                cmp_idx_tok,
                self.win_len,
                win_valid,
                cmp_valid,
                bool(self.win_length_stride or self.cmp_length_stride),
                attn_sink_gm,
                out_tok[h0 : h0 + pipe.tile_m, 0:D],
                h0,
            )
        else:
            pipe = _SmallKAttentionPipeline(self.k, tile_m=self.cv_tile_m)
            for tok in range(block, self.tokens, nblk):
                q_tok = tile_view(
                    qb_gm.view(self.tokens * self.heads, D), (self.heads, D), (tok, 0)
                )
                kv_tok = tile_view(
                    kg_gm.view(self.tokens * self.k, D), (self.k, D), (tok, 0)
                )
                win_idx_tok = tile_view(
                    indices_in_kvcache_gm, (1, self.win_len), (tok, 0)
                )
                cmp_idx_tok = tile_view(
                    extra_indices_in_kvcache_gm, (1, self.k - self.win_len), (tok, 0)
                )
                out_tok = tile_view(
                    out_gm.view(self.tokens * self.heads, D), (self.heads, D), (tok, 0)
                )
                win_valid = dtypes.int32(self.win_len)
                cmp_valid = dtypes.int32(self.k - self.win_len)
                if self.win_length_stride:
                    win_valid = dtypes.int32(
                        topk_length_gm[tok // self.win_length_stride]
                    )
                if self.cmp_length_stride:
                    cmp_valid = dtypes.int32(
                        extra_topk_length_gm[tok // self.cmp_length_stride]
                    )
                for hm in range(self.heads // pipe.tile_m):
                    h0 = hm * pipe.tile_m
                    pipe.run_attention_head_tile(
                        q_tok[h0 : h0 + pipe.tile_m, 0:D],
                        kv_tok,
                        win_idx_tok,
                        cmp_idx_tok,
                        self.win_len,
                        win_valid,
                        cmp_valid,
                        bool(self.win_length_stride or self.cmp_length_stride),
                        attn_sink_gm,
                        out_tok[h0 : h0 + pipe.tile_m, 0:D],
                        h0,
                    )

    @jit
    def _mega_mla_body(
        self,
        hidden_states_gm: Tensor,
        query_norm_weight_gm: Tensor,
        kv_norm_weight_gm: Tensor,
        query_down_proj_weight_gm: Tensor,
        kv_proj_weight_gm: Tensor,
        query_up_proj_weight_gm: Tensor,
        query_down_proj_scale_gm: Tensor,
        kv_proj_scale_gm: Tensor,
        query_up_proj_scale_gm: Tensor,
        k_cache_gm: Tensor,
        extra_k_cache_gm: Tensor,
        indices_in_kvcache_gm: Tensor,
        extra_indices_in_kvcache_gm: Tensor,
        attn_sink_gm: Tensor,
        rope_cos_gm: Tensor,
        rope_sin_gm: Tensor,
        window_cache_slots_gm: Tensor,
        block_table_gm: Tensor,
        extra_block_table_gm: Tensor,
        topk_length_gm: Tensor,
        extra_topk_length_gm: Tensor,
        out_gm: Tensor,
        workspace_gm: Tensor,
        workspace_bf16_gm: Tensor,
        workspace_fp8_gm: Tensor,
        workspace_e8m0_gm: Tensor,
        workspace_int32_gm: Tensor,
    ):
        # All private GM intermediates are disjoint views of one allocation.
        # Existing phase barriers publish them across their producer/consumer cores.
        w = self.workspace_layout.bind(
            workspace_gm,
            workspace_bf16_gm,
            workspace_fp8_gm,
            workspace_e8m0_gm,
            workspace_int32_gm,
        )
        qr_gm, kv_gm, qfull_gm = w["qr"], w["kv"], w["qfull"]
        x8_gm, sx_gm, qn8_gm, sqn_gm = w["x8"], w["sx"], w["qn8"], w["sqn"]
        kvn_gm, zero_gm = w["kvn"], w["zero"]
        qb_gm, kg_gm, s_gm, p_gm = w["qb"], w["kg"], w["s"], w["p"]
        v_gm = w["cmp_dec"]
        ps = self.attn.page_size
        if const_expr(k_cache_gm.stride[0] == ps * 544):
            win_flat = k_cache_gm.view(k_cache_gm.shape[0], ps * 544)
        else:
            # Only merge page-local contiguous axes; a packed reshape would
            # discard the caller's page pitch on the supported DSL package.
            win_flat = k_cache_gm.flatten(1, 3)
        wkc_gm = win_flat.view(dtypes.int8)
        if const_expr(win_flat.stride[0] % 8 == 0):
            wvc_gm = win_flat.view(dtypes.uint64)
            wsc_gm = win_flat.view(dtypes.uint32)
        else:
            wvc_gm, wsc_gm = wkc_gm, wkc_gm
        if const_expr(extra_k_cache_gm.stride[0] == self.compressed_page_size * 320):
            ckc_gm = extra_k_cache_gm.view(
                extra_k_cache_gm.shape[0], self.compressed_page_size, 320
            )
        else:
            ckc_gm = extra_k_cache_gm.flatten(1, 2)
        block, nblk = get_block_idx(), get_block_num()
        # One shared flag triple, reused by every barrier (manual id reuse).
        # This kernel owns one reusable global-barrier triple. Cross-core
        # attention channels are created only after the phase drain/rewind.
        fid = (0, 1, 2)
        self._run_projection_stages(
            qr_gm,
            kv_gm,
            qfull_gm,
            hidden_states_gm,
            query_norm_weight_gm,
            kv_norm_weight_gm,
            query_down_proj_weight_gm,
            kv_proj_weight_gm,
            query_up_proj_weight_gm,
            x8_gm,
            sx_gm,
            qn8_gm,
            sqn_gm,
            query_down_proj_scale_gm,
            kv_proj_scale_gm,
            query_up_proj_scale_gm,
            kvn_gm,
            ckc_gm,
            v_gm,
            extra_indices_in_kvcache_gm,
            extra_topk_length_gm,
            zero_gm,
            kg_gm,
            extra_block_table_gm,
            rope_cos_gm,
            rope_sin_gm,
            window_cache_slots_gm,
            wkc_gm,
            wvc_gm,
            wsc_gm,
            block,
            nblk,
            fid,
        )
        if self.mode != "proj":
            # QK consumers see completed WqB and quantized window scatter.
            global_sync_all(flag_ids=fid)
            # (6)+(9/12) Q RoPE + single K/V gather for all tokens
            self._prepare_query_and_gather_kv(
                qfull_gm,
                rope_cos_gm,
                rope_sin_gm,
                wkc_gm,
                wvc_gm,
                block_table_gm,
                ckc_gm,
                wsc_gm,
                extra_block_table_gm,
                v_gm,
                indices_in_kvcache_gm,
                extra_indices_in_kvcache_gm,
                topk_length_gm,
                extra_topk_length_gm,
                zero_gm,
                qb_gm,
                kg_gm,
                block,
                nblk,
            )
            if const_expr(
                self.cmp_decode_indexed
                and self.cmp_p3_items < self.tokens * (self.k - self.win_len)
            ):
                self.compressed_cache_gather.run(
                    ckc_gm,
                    extra_indices_in_kvcache_gm,
                    extra_topk_length_gm,
                    zero_gm,
                    kg_gm,
                    block * 2 + get_subblock_id(),
                    nblk * 2,
                    extra_block_table_gm,
                )
            global_sync_all(flag_ids=fid)
            if self.mode != "cache":
                # CV split-M is valid only for H64. H8/H16 keep logical M
                # and the existing GM bridge, with no padded head computation.
                if self.cv_tile_m:
                    vec_sync_all()
                    onchip_rewind(reset_sync_id=True)
                    self._run_small_k_attention_pipeline(
                        qb_gm,
                        kg_gm,
                        indices_in_kvcache_gm,
                        extra_indices_in_kvcache_gm,
                        attn_sink_gm,
                        out_gm,
                        topk_length_gm,
                        extra_topk_length_gm,
                        block,
                        nblk,
                    )
                else:
                    # QK Cube, then mask/sink softmax AIV, then PV Cube.
                    vec_sync_all()
                    onchip_rewind(reset_sync_id=True)
                    n_tiles_qk = self.qk.n // self.qk.base_n
                    if self.tokens <= self.grid and not self.flat_attention_tiles:
                        bpt_qk = max(1, self.grid // self.tokens)
                        nt_per_qk = (n_tiles_qk + bpt_qk - 1) // bpt_qk
                        self.qk.compute_owned_tiles(
                            s_gm, qb_gm, kg_gm, block, self.tokens, bpt_qk, nt_per_qk
                        )
                    else:
                        self.qk.compute_flat_tiles(
                            s_gm, qb_gm, kg_gm, block, nblk, self.tokens
                        )
                    if self.grid < self.tokens <= 2 * self.grid:
                        _publish_qk_to_paired_aivs(fid)
                    else:
                        global_sync_all(flag_ids=fid)
                    self._run_attention_softmax(
                        s_gm,
                        indices_in_kvcache_gm,
                        extra_indices_in_kvcache_gm,
                        attn_sink_gm,
                        p_gm,
                        topk_length_gm,
                        extra_topk_length_gm,
                        block,
                        nblk,
                    )
                    global_sync_all(flag_ids=fid)
                    self._run_probability_value_matmul(out_gm, p_gm, kg_gm, block, nblk)

    @kernel
    def mega_mla(
        self,
        hidden_states_gm: Tensor,
        query_norm_weight_gm: Tensor,
        kv_norm_weight_gm: Tensor,
        query_down_proj_weight_gm: Tensor,
        kv_proj_weight_gm: Tensor,
        query_up_proj_weight_gm: Tensor,
        query_down_proj_scale_gm: Tensor,
        kv_proj_scale_gm: Tensor,
        query_up_proj_scale_gm: Tensor,
        k_cache_gm: Tensor,
        extra_k_cache_gm: Tensor,
        indices_in_kvcache_gm: Tensor,
        extra_indices_in_kvcache_gm: Tensor,
        attn_sink_gm: Tensor,
        rope_cos_gm: Tensor,
        rope_sin_gm: Tensor,
        window_cache_slots_gm: Tensor,
        block_table_gm: Tensor,
        extra_block_table_gm: Tensor,
        topk_length_gm: Tensor,
        extra_topk_length_gm: Tensor,
        out_gm: Tensor,
        workspace_gm: Tensor,
        workspace_bf16_gm: Tensor,
        workspace_fp8_gm: Tensor,
        workspace_e8m0_gm: Tensor,
        workspace_int32_gm: Tensor,
    ):
        self._mega_mla_body(
            hidden_states_gm,
            query_norm_weight_gm,
            kv_norm_weight_gm,
            query_down_proj_weight_gm,
            kv_proj_weight_gm,
            query_up_proj_weight_gm,
            query_down_proj_scale_gm,
            kv_proj_scale_gm,
            query_up_proj_scale_gm,
            k_cache_gm,
            extra_k_cache_gm,
            indices_in_kvcache_gm,
            extra_indices_in_kvcache_gm,
            attn_sink_gm,
            rope_cos_gm,
            rope_sin_gm,
            window_cache_slots_gm,
            block_table_gm,
            extra_block_table_gm,
            topk_length_gm,
            extra_topk_length_gm,
            out_gm,
            workspace_gm,
            workspace_bf16_gm,
            workspace_fp8_gm,
            workspace_e8m0_gm,
            workspace_int32_gm,
        )

    @kernel
    def mega_mla_with_seqlens(
        self,
        hidden_states_gm: Tensor,
        query_norm_weight_gm: Tensor,
        kv_norm_weight_gm: Tensor,
        query_down_proj_weight_gm: Tensor,
        kv_proj_weight_gm: Tensor,
        query_up_proj_weight_gm: Tensor,
        query_down_proj_scale_gm: Tensor,
        kv_proj_scale_gm: Tensor,
        query_up_proj_scale_gm: Tensor,
        k_cache_gm: Tensor,
        extra_k_cache_gm: Tensor,
        indices_in_kvcache_gm: Tensor,
        extra_indices_in_kvcache_gm: Tensor,
        attn_sink_gm: Tensor,
        rope_cos_gm: Tensor,
        rope_sin_gm: Tensor,
        window_cache_slots_gm: Tensor,
        block_table_gm: Tensor,
        extra_block_table_gm: Tensor,
        topk_length_gm: Tensor,
        extra_topk_length_gm: Tensor,
        out_gm: Tensor,
        workspace_gm: Tensor,
        workspace_bf16_gm: Tensor,
        workspace_fp8_gm: Tensor,
        workspace_e8m0_gm: Tensor,
        workspace_int32_gm: Tensor,
        cu_seqlens_qo_gm: Tensor,
        seqused_q_gm: Tensor,
        cache_seqlens_gm: Tensor,
        extra_cache_seqlens_gm: Tensor,
    ):
        w = self.workspace_layout.bind(
            workspace_gm,
            workspace_bf16_gm,
            workspace_fp8_gm,
            workspace_e8m0_gm,
            workspace_int32_gm,
        )
        self.sequence_filter.run(
            indices_in_kvcache_gm,
            extra_indices_in_kvcache_gm,
            topk_length_gm,
            extra_topk_length_gm,
            block_table_gm,
            extra_block_table_gm,
            cu_seqlens_qo_gm,
            seqused_q_gm,
            cache_seqlens_gm,
            extra_cache_seqlens_gm,
            window_cache_slots_gm,
            w["seq_win_idx"],
            w["seq_cmp_idx"],
            w["seq_slot"],
            get_block_idx() * 2 + get_subblock_id(),
            get_block_num() * 2,
            k_cache_gm.shape[0],
            extra_k_cache_gm.shape[0],
        )
        # Drain local metadata writes. P0 only quantizes x and clears disjoint
        # projection buffers; its B0 barrier publishes metadata before the
        # first P1 indexed decode and the P2 cache writer consume it.
        vec_sync_all()
        # Sequence scratch is dead after the local drain; recycle its buffer IDs.
        onchip_rewind(reset_sync_id=True)
        if const_expr(not self.small_token_vector_schedule):
            global_sync_all(flag_ids=(0, 1, 2))
        self._mega_mla_body(
            hidden_states_gm,
            query_norm_weight_gm,
            kv_norm_weight_gm,
            query_down_proj_weight_gm,
            kv_proj_weight_gm,
            query_up_proj_weight_gm,
            query_down_proj_scale_gm,
            kv_proj_scale_gm,
            query_up_proj_scale_gm,
            k_cache_gm,
            extra_k_cache_gm,
            w["seq_win_idx"],
            w["seq_cmp_idx"],
            attn_sink_gm,
            rope_cos_gm,
            rope_sin_gm,
            w["seq_slot"],
            block_table_gm,
            extra_block_table_gm,
            topk_length_gm,
            extra_topk_length_gm,
            out_gm,
            workspace_gm,
            workspace_bf16_gm,
            workspace_fp8_gm,
            workspace_e8m0_gm,
            workspace_int32_gm,
        )

    @kernel
    def mega_mla_with_metadata(
        self,
        hidden_states_gm: Tensor,
        query_norm_weight_gm: Tensor,
        kv_norm_weight_gm: Tensor,
        query_down_proj_weight_gm: Tensor,
        kv_proj_weight_gm: Tensor,
        query_up_proj_weight_gm: Tensor,
        query_down_proj_scale_gm: Tensor,
        kv_proj_scale_gm: Tensor,
        query_up_proj_scale_gm: Tensor,
        k_cache_gm: Tensor,
        extra_k_cache_gm: Tensor,
        indices_in_kvcache_gm: Tensor,
        extra_indices_in_kvcache_gm: Tensor,
        attn_sink_gm: Tensor,
        rope_cos_gm: Tensor,
        rope_sin_gm: Tensor,
        window_cache_slots_gm: Tensor,
        block_table_gm: Tensor,
        extra_block_table_gm: Tensor,
        topk_length_gm: Tensor,
        extra_topk_length_gm: Tensor,
        out_gm: Tensor,
        workspace_gm: Tensor,
        workspace_bf16_gm: Tensor,
        workspace_fp8_gm: Tensor,
        workspace_e8m0_gm: Tensor,
        workspace_int32_gm: Tensor,
        cu_seqlens_qo_gm: Tensor,
        seqused_q_gm: Tensor,
        cache_seqlens_gm: Tensor,
        extra_cache_seqlens_gm: Tensor,
        query_metadata_gm: Tensor,
    ):
        w = self.workspace_layout.bind(
            workspace_gm,
            workspace_bf16_gm,
            workspace_fp8_gm,
            workspace_e8m0_gm,
            workspace_int32_gm,
        )
        self.sequence_filter.run(
            indices_in_kvcache_gm,
            extra_indices_in_kvcache_gm,
            topk_length_gm,
            extra_topk_length_gm,
            block_table_gm,
            extra_block_table_gm,
            cu_seqlens_qo_gm,
            seqused_q_gm,
            cache_seqlens_gm,
            extra_cache_seqlens_gm,
            window_cache_slots_gm,
            w["seq_win_idx"],
            w["seq_cmp_idx"],
            w["seq_slot"],
            get_block_idx() * 2 + get_subblock_id(),
            get_block_num() * 2,
            k_cache_gm.shape[0],
            extra_k_cache_gm.shape[0],
            query_metadata_gm,
        )
        vec_sync_all()
        # Sequence scratch is dead after the local drain; recycle its buffer IDs.
        onchip_rewind(reset_sync_id=True)
        if const_expr(not self.small_token_vector_schedule):
            global_sync_all(flag_ids=(0, 1, 2))
        self._mega_mla_body(
            hidden_states_gm,
            query_norm_weight_gm,
            kv_norm_weight_gm,
            query_down_proj_weight_gm,
            kv_proj_weight_gm,
            query_up_proj_weight_gm,
            query_down_proj_scale_gm,
            kv_proj_scale_gm,
            query_up_proj_scale_gm,
            k_cache_gm,
            extra_k_cache_gm,
            w["seq_win_idx"],
            w["seq_cmp_idx"],
            attn_sink_gm,
            rope_cos_gm,
            rope_sin_gm,
            w["seq_slot"],
            block_table_gm,
            extra_block_table_gm,
            topk_length_gm,
            extra_topk_length_gm,
            out_gm,
            workspace_gm,
            workspace_bf16_gm,
            workspace_fp8_gm,
            workspace_e8m0_gm,
            workspace_int32_gm,
        )
