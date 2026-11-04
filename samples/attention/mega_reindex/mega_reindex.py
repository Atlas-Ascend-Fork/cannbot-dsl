# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""DeepSeek V4.1 L24/L28/L32/L36 candidate-consumer indexer mega kernel."""

import functools
import math

import cannbotdsl as cb
import torch
from cannbotdsl import dtypes as dt
from cannbotdsl.ops.sync import global_sync_all, onchip_rewind

from .reindex_front import FrontPhases, FrontWorkspace
from .reindex_query import TailQueryPhase, get_query_tile_config
from .reindex_runtime import (
    CANDIDATE_BLOCK_SIZE,
    CANDIDATE_CAPACITY,
    DTYPES,
    HEAD_DIM,
    HEAD_SCALE_GROUPS,
    HIDDEN_SIZE,
    INPUT_SCALE_PAIRS,
    MAX_KV_LENGTH,
    MAX_PA_PAGE_SIZE,
    MX_SCALE_PAIR,
    NUM_HEADS,
    PA_PAGE_ALIGNMENT,
    PA_RECORD_BYTES,
    PACKED_HEAD_DIM,
    Q_LORA_RANK,
    QUERY_DIM,
    QUERY_SCALE_PAIRS,
    SCALE_DTYPE,
    SCALE_DTYPES,
    SCORE_TILE_TOKENS,
    TOPK_ALIGNMENT,
    ResourceCheckedProgram,
    check,
    check_nz,
    check_platform,
    signature,
)
from .reindex_tail import CandidatePhase, PaReadPhase, TailScorePhase, TailTopkPhase


class MegaReindexTailWorkspace:
    """Compact single-kernel workspace; use separately from the two-kernel tail."""

    def __init__(self, rows, kv_s, device, *, topk=512):
        self.rows, self.kv_s, self.topk = rows, kv_s, topk
        self.device = torch.device(device)
        self.padded = (
            (kv_s + SCORE_TILE_TOKENS - 1) // SCORE_TILE_TOKENS * SCORE_TILE_TOKENS
        )

        def alloc(shape, dtype):
            return torch.empty(shape, device=device, dtype=dtype)

        self.q = alloc((rows, NUM_HEADS, PACKED_HEAD_DIM), torch.uint8)
        self.q_scale = alloc(
            (rows, NUM_HEADS, HEAD_SCALE_GROUPS // MX_SCALE_PAIR, MX_SCALE_PAIR),
            torch.uint8,
        )
        self.score = alloc((rows, self.padded), torch.bfloat16)
        self.indices = alloc((rows, topk), torch.int32)
        self.bias = alloc((0,), torch.float32)
        self.block_mask = alloc(
            (rows, self.padded // CANDIDATE_BLOCK_SIZE), torch.uint8
        )
        self.visible_lengths = alloc((rows,), torch.int32)


class MegaReindex:
    def __init__(
        self, rows, batch, length, topk, norm_eps, softmax_scale, cube_cores=None
    ):
        self.cube_cores = check_platform() if cube_cores is None else cube_cores
        self.rows, self.batch, self.padded = (
            rows,
            batch,
            (length + SCORE_TILE_TOKENS - 1) // SCORE_TILE_TOKENS * SCORE_TILE_TOKENS,
        )
        front = FrontPhases(rows, norm_eps, softmax_scale, self.cube_cores)
        self.query_core_count = front.query.core_count
        qt = get_query_tile_config(self.cube_cores)
        self.t_query = TailQueryPhase(rows, QUERY_DIM, Q_LORA_RANK, qt, self.cube_cores)
        self.f_quant, self.f_query, self.f_weight, self.f_norm = (
            front.quant,
            front.query,
            front.weight,
            front.norm,
        )
        self.t_candidates = CandidatePhase(rows, batch, length, self.cube_cores)
        self.t_score = TailScorePhase(rows, batch, length, self.cube_cores)
        self.t_topk = TailTopkPhase(rows, self.padded, topk, self.cube_cores)

    @cb.kernel
    def mega_reindex_kernel(
        self,
        f_x: cb.Tensor,
        f_ww: cb.Tensor,
        f_w: cb.Tensor,
        f_xq: cb.Tensor,
        f_xs: cb.Tensor,
        f_a: cb.Tensor,
        f_sa: cb.Tensor,
        f_b: cb.Tensor,
        f_sb: cb.Tensor,
        f_qa: cb.Tensor,
        f_bias: cb.Tensor,
        f_g: cb.Tensor,
        f_qr: cb.Tensor,
        f_rq: cb.Tensor,
        f_rs: cb.Tensor,
        f_weights: cb.Tensor,
        t_q: cb.Tensor,
        t_a: cb.Tensor,
        t_b: cb.Tensor,
        t_sa: cb.Tensor,
        t_sb: cb.Tensor,
        t_bias: cb.Tensor,
        t_freq: cb.Tensor,
        t_qs: cb.Tensor,
        t_k: cb.Tensor,
        t_ks: cb.Tensor,
        t_w: cb.Tensor,
        t_mask: cb.Tensor,
        t_visible: cb.Tensor,
        t_score: cb.Tensor,
        t_indices: cb.Tensor,
        t_q_address,
        t_k_address,
        t_candidates: cb.Tensor,
        t_counts: cb.Tensor,
        t_cuq: cb.Tensor,
        t_cuk: cb.Tensor,
        t_usedq: cb.Tensor,
        t_usedk: cb.Tensor,
    ):
        self.f_quant.phase(f_x, f_ww, f_w, f_xq, f_xs)
        global_sync_all()
        onchip_rewind()
        if cb.get_block_idx() < self.query_core_count:
            self.f_query.phase(f_qa, f_a, f_b, f_sa, f_sb, f_bias)
        else:
            self.f_weight.phase(f_x, f_ww, f_w, f_xq, f_xs)
        # Independent Vector work overlaps front Cube projections.
        # All lanes join the following barrier before workspace reuse.
        self.t_candidates.phase(
            t_candidates, t_counts, t_cuq, t_cuk, t_usedq, t_usedk, t_mask, t_visible
        )
        global_sync_all()
        onchip_rewind()
        self.f_norm.phase(
            f_qa.view((self.rows * Q_LORA_RANK,)), f_g, f_qr, f_rq, f_rs, f_w, f_weights
        )
        global_sync_all()
        onchip_rewind()
        self.t_query.phase(t_q, t_a, t_b, t_sa, t_sb, t_bias, t_freq, t_qs)
        global_sync_all()
        onchip_rewind()
        self.t_score.phase(
            t_q_address,
            t_qs.view(
                (
                    self.rows * NUM_HEADS,
                    HEAD_SCALE_GROUPS // MX_SCALE_PAIR,
                    MX_SCALE_PAIR,
                )
            ),
            t_k_address,
            t_ks.view((self.batch * self.t_score.length, 2, 2)),
            t_w,
            t_mask,
            t_visible,
            t_score,
        )
        global_sync_all()
        onchip_rewind()
        self.t_topk.phase(t_score.view((self.rows * self.padded,)), t_indices)

    @cb.host
    def run(
        self,
        f_x,
        f_ww,
        f_w,
        f_xq,
        f_xs,
        f_a,
        f_sa,
        f_b,
        f_sb,
        f_qa,
        f_bias,
        f_g,
        f_qr,
        f_rq,
        f_rs,
        f_weights,
        t_q,
        t_a,
        t_b,
        t_sa,
        t_sb,
        t_bias,
        t_freq,
        t_qs,
        t_k,
        t_ks,
        t_w,
        t_mask,
        t_visible,
        t_score,
        t_indices,
        t_q_address,
        t_k_address,
        t_candidates,
        t_counts,
        t_cuq,
        t_cuk,
        t_usedq,
        t_usedk,
    ):
        self.mega_reindex_kernel[self.cube_cores](
            f_x,
            f_ww,
            f_w,
            f_xq,
            f_xs,
            f_a,
            f_sa,
            f_b,
            f_sb,
            f_qa,
            f_bias,
            f_g,
            f_qr,
            f_rq,
            f_rs,
            f_weights,
            t_q,
            t_a,
            t_b,
            t_sa,
            t_sb,
            t_bias,
            t_freq,
            t_qs,
            t_k,
            t_ks,
            t_w,
            t_mask,
            t_visible,
            t_score,
            t_indices,
            t_q_address,
            t_k_address,
            t_candidates,
            t_counts,
            t_cuq,
            t_cuk,
            t_usedq,
            t_usedk,
        )


class MegaReindexPA(MegaReindex):
    def __init__(
        self,
        rows,
        batch,
        length,
        topk,
        norm_eps,
        softmax_scale,
        page_size,
        cube_cores=None,
    ):
        super().__init__(rows, batch, length, topk, norm_eps, softmax_scale, cube_cores)
        self.pa_read = PaReadPhase(batch, length, page_size, self.cube_cores)

    @cb.kernel
    def mega_reindex_kernel(
        self,
        f_x: cb.Tensor,
        f_ww: cb.Tensor,
        f_w: cb.Tensor,
        f_xq: cb.Tensor,
        f_xs: cb.Tensor,
        f_a: cb.Tensor,
        f_sa: cb.Tensor,
        f_b: cb.Tensor,
        f_sb: cb.Tensor,
        f_qa: cb.Tensor,
        f_bias: cb.Tensor,
        f_g: cb.Tensor,
        f_qr: cb.Tensor,
        f_rq: cb.Tensor,
        f_rs: cb.Tensor,
        f_weights: cb.Tensor,
        t_q: cb.Tensor,
        t_a: cb.Tensor,
        t_b: cb.Tensor,
        t_sa: cb.Tensor,
        t_sb: cb.Tensor,
        t_bias: cb.Tensor,
        t_freq: cb.Tensor,
        t_qs: cb.Tensor,
        t_k: cb.Tensor,
        t_ks: cb.Tensor,
        t_w: cb.Tensor,
        t_mask: cb.Tensor,
        t_visible: cb.Tensor,
        t_score: cb.Tensor,
        t_indices: cb.Tensor,
        t_q_address,
        t_k_address,
        t_candidates: cb.Tensor,
        t_counts: cb.Tensor,
        t_cuq: cb.Tensor,
        t_cuk: cb.Tensor,
        t_usedq: cb.Tensor,
        t_usedk: cb.Tensor,
        pa_k: cb.Tensor,
        pa_table: cb.Tensor,
    ):
        self.f_quant.phase(f_x, f_ww, f_w, f_xq, f_xs)
        global_sync_all()
        onchip_rewind()
        if cb.get_block_idx() < self.query_core_count:
            self.f_query.phase(f_qa, f_a, f_b, f_sa, f_sb, f_bias)
        else:
            self.f_weight.phase(f_x, f_ww, f_w, f_xq, f_xs)
        # Independent Vector work overlaps front Cube projections.
        # All lanes join the following barrier before workspace reuse.
        self.t_candidates.phase(
            t_candidates, t_counts, t_cuq, t_cuk, t_usedq, t_usedk, t_mask, t_visible
        )
        self.pa_read.phase(pa_k, pa_table, t_usedk, t_k, t_ks)
        global_sync_all()
        onchip_rewind()
        self.f_norm.phase(
            f_qa.view((self.rows * Q_LORA_RANK,)), f_g, f_qr, f_rq, f_rs, f_w, f_weights
        )
        global_sync_all()
        onchip_rewind()
        self.t_query.phase(t_q, t_a, t_b, t_sa, t_sb, t_bias, t_freq, t_qs)
        global_sync_all()
        onchip_rewind()
        self.t_score.phase(
            t_q_address,
            t_qs.view(
                (
                    self.rows * NUM_HEADS,
                    HEAD_SCALE_GROUPS // MX_SCALE_PAIR,
                    MX_SCALE_PAIR,
                )
            ),
            t_k_address,
            t_ks.view((self.batch * self.t_score.length, 2, 2)),
            t_w,
            t_mask,
            t_visible,
            t_score,
        )
        global_sync_all()
        onchip_rewind()
        self.t_topk.phase(t_score.view((self.rows * self.padded,)), t_indices)

    @cb.host
    def run(
        self,
        f_x,
        f_ww,
        f_w,
        f_xq,
        f_xs,
        f_a,
        f_sa,
        f_b,
        f_sb,
        f_qa,
        f_bias,
        f_g,
        f_qr,
        f_rq,
        f_rs,
        f_weights,
        t_q,
        t_a,
        t_b,
        t_sa,
        t_sb,
        t_bias,
        t_freq,
        t_qs,
        t_k,
        t_ks,
        t_w,
        t_mask,
        t_visible,
        t_score,
        t_indices,
        t_q_address,
        t_k_address,
        t_candidates,
        t_counts,
        t_cuq,
        t_cuk,
        t_usedq,
        t_usedk,
        pa_k,
        pa_table,
    ):
        self.mega_reindex_kernel[self.cube_cores](
            f_x,
            f_ww,
            f_w,
            f_xq,
            f_xs,
            f_a,
            f_sa,
            f_b,
            f_sb,
            f_qa,
            f_bias,
            f_g,
            f_qr,
            f_rq,
            f_rs,
            f_weights,
            t_q,
            t_a,
            t_b,
            t_sa,
            t_sb,
            t_bias,
            t_freq,
            t_qs,
            t_k,
            t_ks,
            t_w,
            t_mask,
            t_visible,
            t_score,
            t_indices,
            t_q_address,
            t_k_address,
            t_candidates,
            t_counts,
            t_cuq,
            t_cuk,
            t_usedq,
            t_usedk,
            pa_k,
            pa_table,
        )


def prepare_front(
    x,
    wqa,
    descale_wqa,
    ww,
    norm_weight,
    *,
    softmax_scale=1 / 64,
    norm_eps=1e-20,
    workspace=None,
):
    """Validate front tensors and return (kernel_arguments, workspace).

    x[T,5120] and norm_weight[1280] are BF16. WQA[5120,1280] is FP8
    E4M3 or its byte view in FRACTAL_NZ [5120,1280]; descale_wqa[80,1280,2]
    is ND paired E8M0 bytes, ordered by K-group, output channel, scale pair.
    ww[32,5120] is unscaled BF16 FRACTAL_NZ. weights are
    BF16((FP32(x) @ FP32(ww).T) * softmax_scale).
    No CPU/numeric fallback. attn_norm is outside this operator.
    """
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError("x must be [T,5120], T>0")
    t, dev = x.shape[0], x.device
    check(x, "x", (t, HIDDEN_SIZE), torch.bfloat16, dev)
    check(
        wqa,
        "wqa",
        (HIDDEN_SIZE, Q_LORA_RANK),
        (torch.uint8, torch.int8, torch.float8_e4m3fn),
        dev,
    )
    check_nz(wqa, "wqa")
    check(
        descale_wqa,
        "descale_wqa",
        (INPUT_SCALE_PAIRS, Q_LORA_RANK, 2),
        SCALE_DTYPES,
        dev,
    )
    check(ww, "ww", (NUM_HEADS, HIDDEN_SIZE), torch.bfloat16, dev)
    check_nz(ww, "ww")
    check(norm_weight, "norm_weight", (Q_LORA_RANK,), torch.bfloat16, dev)
    if not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError("norm_eps must be finite and positive")
    if not math.isfinite(softmax_scale):
        raise ValueError("softmax_scale must be finite")
    ws = workspace if workspace is not None else FrontWorkspace(t, dev)
    if ws.rows != t or ws.device != dev:
        raise ValueError("workspace T/device mismatch")
    args = (
        x,
        ww,
        ws.w_raw,
        ws.xq.flatten(),
        ws.xs.flatten(),
        ws.xq.view(torch.int8),
        ws.xs.view(SCALE_DTYPE),
        wqa.view(torch.int8),
        descale_wqa.view(SCALE_DTYPE),
        ws.qa,
        ws.bias,
        norm_weight,
        ws.qr.flatten(),
        ws.rq.flatten(),
        ws.rs.flatten(),
        ws.w,
    )
    return args, ws


def prepare_tail(
    qr,
    descale_qr,
    wqb,
    descale_wqb,
    weights,
    rope,
    key,
    descale_key,
    candidate_block_indices,
    candidate_block_length,
    *,
    cu_seqlens_q,
    cu_seqlens_k=None,
    seqused_q,
    seqused_k,
    candidate_block_size=8,
    mask_mode=3,
    cmp_ratio=1,
    topk=512,
    workspace=None,
):
    """Validate tail tensors and return (kernel_arguments, workspace).

    T=qr.shape[0], B=seqused_q.numel(), kv_s=key.shape[0]//B, S=T//B.
    key[B*kv_s,1,64] is repository-native packed MXFP4 E2M1 storage;
    descale_key[B*kv_s,1,2,2] is its E8M0 scale (four groups per D=128).
    wqb[4096,1280] is preprocessed FRACTAL_NZ; descale_wqb[4096,20,2]
    remains ND E8M0 bytes, with scale groups along K=1280.
    rope[T,32,2] holds (cos,sin) for adjacent-pair RoPE on the last 64 dims.
    Candidates[T,1,2048] contain unique batch-local eight-token block IDs
    in each row's valid prefix. Counts[T,1] specify the prefix length.
    Equal-capacity Q/K batches only: cu_q[b]=b*S, cu_k[b]=b*kv_s.
    seqused_q/k may shorten each batch. Only mask_mode=3, cmp_ratio=1.
    Valid indices are ascending, ties prefer lower positions; missing slots
    and inactive queries return -1. No external dense mask or metadata.
    """
    if (candidate_block_size, mask_mode, cmp_ratio) != (CANDIDATE_BLOCK_SIZE, 3, 1):
        raise ValueError("supported: candidate_block_size=8, mask_mode=3, cmp_ratio=1")
    if qr.ndim != 2 or key.ndim != 3 or seqused_q.ndim != 1:
        raise ValueError("qr [T,1280], key [B*kv_s,1,64], seqused_q [B] required")
    t, b = qr.shape[0], seqused_q.numel()
    if b < 1 or key.shape[0] % b:
        raise ValueError("key.shape[0] must equal B*kv_s")
    length = key.shape[0] // b
    if b < 1 or t < 1 or t % b or (t // b) % 2:
        raise ValueError(
            "T must be B*S with positive even S for the paired-token score template"
        )
    if (
        not isinstance(topk, int)
        or topk < TOPK_ALIGNMENT
        or topk % TOPK_ALIGNMENT
        or not topk <= length <= MAX_KV_LENGTH
    ):
        raise ValueError("require 64-aligned topk, 64<=topk<=kv_s<=16384")
    dev = qr.device
    fp8 = (torch.uint8, torch.int8, torch.float8_e4m3fn)
    for x, name, shape, dtype in (
        (qr, "qr", (t, Q_LORA_RANK), fp8),
        (descale_qr, "descale_qr", (t, QUERY_SCALE_PAIRS, 2), SCALE_DTYPES),
        (wqb, "wqb", (QUERY_DIM, Q_LORA_RANK), fp8),
        (descale_wqb, "descale_wqb", (QUERY_DIM, QUERY_SCALE_PAIRS, 2), SCALE_DTYPES),
        (weights, "weights", (t, NUM_HEADS), torch.bfloat16),
        (rope, "rope", (t, HEAD_DIM // 4, 2), torch.float32),
        (key, "key", (b * length, 1, PACKED_HEAD_DIM), torch.uint8),
        (descale_key, "descale_key", (b * length, 1, 2, 2), SCALE_DTYPES),
        (
            candidate_block_indices,
            "candidate_block_indices",
            (t, 1, CANDIDATE_CAPACITY),
            torch.int32,
        ),
        (candidate_block_length, "candidate_block_length", (t, 1), torch.int32),
        (cu_seqlens_q, "cu_seqlens_q", (b + 1,), torch.int32),
        (cu_seqlens_k, "cu_seqlens_k", (b + 1,), torch.int32),
        (seqused_q, "seqused_q", (b,), torch.int32),
        (seqused_k, "seqused_k", (b,), torch.int32),
    ):
        check(x, name, shape, dtype, dev)
    ws = (
        workspace
        if workspace is not None
        else MegaReindexTailWorkspace(t, length, dev, topk=topk)
    )
    check_nz(wqb, "wqb")
    if (ws.rows, ws.kv_s, ws.topk, ws.device) != (t, length, topk, dev):
        raise ValueError("workspace T/kv_s/topk/device mismatch")
    if not hasattr(ws, "block_mask"):
        ws.block_mask = torch.empty(
            (t, ws.padded // CANDIDATE_BLOCK_SIZE), dtype=torch.uint8, device=dev
        )
    check(
        ws.block_mask,
        "workspace.block_mask",
        (t, ws.padded // CANDIDATE_BLOCK_SIZE),
        torch.uint8,
        dev,
    )
    args = (
        ws.q.view(t, NUM_HEADS * PACKED_HEAD_DIM),
        qr.view(torch.int8),
        wqb.view(torch.int8),
        descale_qr.view(SCALE_DTYPE),
        descale_wqb.view(SCALE_DTYPE),
        ws.bias,
        rope.flatten(),
        ws.q_scale.view(t, NUM_HEADS * HEAD_SCALE_GROUPS),
        key,
        descale_key.view(SCALE_DTYPE),
        weights,
        ws.block_mask,
        ws.visible_lengths,
        ws.score,
        ws.indices.flatten(),
        ws.q.data_ptr(),
        key.data_ptr(),
        candidate_block_indices,
        candidate_block_length,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
    )
    return args, ws


def prepare_pa(x, key, scale, table, usedq, usedk, cuk, ws, topk):
    if (
        key is None
        or key.ndim != 3
        or key.shape[2] != PA_RECORD_BYTES
        or key.shape[0] < 1
    ):
        raise ValueError("PA k must be UINT8 [P,page_size/8,544]")
    if scale is not None:
        raise ValueError("PA embeds scale in k; descale_k/descale_key must be None")
    page_size = key.shape[1] * CANDIDATE_BLOCK_SIZE
    if (
        page_size < PA_PAGE_ALIGNMENT
        or page_size > MAX_PA_PAGE_SIZE
        or page_size % PA_PAGE_ALIGNMENT
    ):
        raise ValueError("PA page_size must be a multiple of 16 in [16,1024]")
    if usedq.ndim != 1 or usedq.numel() < 1:
        raise ValueError("seqused_q must be [B], B>0")
    batch = usedq.numel()
    if (
        table is None
        or table.ndim != 2
        or table.shape[0] != batch
        or table.shape[1] < 1
    ):
        raise ValueError("PA requires block_table INT32 [B,max_pages]")
    check(table, "block_table", tuple(table.shape), torch.int32, x.device)
    check(usedk, "seqused_k", (batch,), torch.int32, x.device)
    if key.dtype != torch.uint8 or key.device != x.device:
        raise ValueError("PA k must be UINT8 on the same NPU")
    if (
        key.stride(2) != 1
        or key.stride(1) != PA_RECORD_BYTES
        or key.stride(0) < key.shape[1] * PA_RECORD_BYTES
    ):
        raise ValueError("PA k inner axes must be contiguous; pages cannot overlap")
    length = table.shape[1] * page_size
    if (
        not isinstance(topk, int)
        or topk < TOPK_ALIGNMENT
        or topk % TOPK_ALIGNMENT
        or not topk <= length <= MAX_KV_LENGTH
    ):
        raise ValueError("require 64-aligned topk, 64<=topk<=PA capacity<=16384")
    if cuk is not None:
        raise ValueError("PA does not accept cu_seqlens_k; use seqused_k")
    ws = (
        ws
        if ws is not None
        else MegaReindexTailWorkspace(x.shape[0], length, x.device, topk=topk)
    )
    if (ws.rows, ws.kv_s, ws.topk, ws.device) != (x.shape[0], length, topk, x.device):
        raise ValueError("PA workspace T/capacity/topk/device mismatch")
    sig = (batch, length)
    if getattr(ws, "_pa_signature", None) != sig:
        ws.pa_key = torch.empty(
            (batch * length, 1, PACKED_HEAD_DIM), dtype=torch.uint8, device=x.device
        )
        ws.pa_scale = torch.empty(
            (batch * length, 1, 2, 2), dtype=torch.uint8, device=x.device
        )
        ws.pa_cuk = torch.arange(batch + 1, dtype=torch.int32, device=x.device) * length
        ws._pa_signature = sig
    return ws.pa_key, ws.pa_scale, ws.pa_cuk, ws, (key, table), page_size


@functools.lru_cache(maxsize=32)
def compile_mega_reindex(
    rows,
    batch,
    length,
    topk,
    norm_eps,
    softmax_scale,
    device,
    specs,
    page_size,
    cube_cores,
):
    op = (
        MegaReindexPA(
            rows, batch, length, topk, norm_eps, softmax_scale, page_size, cube_cores
        )
        if page_size
        else MegaReindex(rows, batch, length, topk, norm_eps, softmax_scale, cube_cores)
    )

    def spec(i, s):
        if s[0] == "int64":
            return dt.int64
        _, shape, dtype, stride = s
        if i in (1, 7, 18):
            return cb.TensorSpec(shape, DTYPES[dtype], storage_format="nz")
        return cb.TensorSpec(shape, DTYPES[dtype], stride=stride)

    with torch.npu.device(device):
        return cb.compile(op.run, *(spec(i, s) for i, s in enumerate(specs)))


def prepare_mega_reindex(
    x,
    wqa,
    descale_wqa,
    ww,
    norm_weight,
    wqb,
    descale_wqb,
    rope,
    key=None,
    descale_key=None,
    *,
    front_workspace=None,
    tail_workspace=None,
    norm_eps=1e-20,
    softmax_scale=1 / 64,
    topk=512,
    candidate_block_indices,
    candidate_block_length,
    cu_seqlens_q,
    cu_seqlens_k=None,
    seqused_q,
    seqused_k,
    candidate_block_size=8,
    mask_mode=3,
    cmp_ratio=1,
    layout_k="TND",
    block_table=None,
    k=None,
    descale_k=None,
):
    if k is not None:
        if key is not None:
            raise ValueError("provide only one of key and k")
        key = k
    if descale_k is not None:
        if descale_key is not None:
            raise ValueError("provide only one of descale_key and descale_k")
        descale_key = descale_k
    pa_args = ()
    page_size = 0
    if layout_k == "PA_BBND":
        key, descale_key, cu_seqlens_k, tail_workspace, pa_args, page_size = prepare_pa(
            x,
            key,
            descale_key,
            block_table,
            seqused_q,
            seqused_k,
            cu_seqlens_k,
            tail_workspace,
            topk,
        )
    elif layout_k == "TND":
        if block_table is not None:
            raise ValueError("TND does not accept block_table")
        if key is None or descale_key is None or cu_seqlens_k is None:
            raise ValueError("TND requires key, descale_key and cu_seqlens_k")
        if descale_key.dtype == getattr(torch, "float8_e8m0fnu", None):
            descale_key = descale_key.view(torch.uint8)
    else:
        raise ValueError("layout_k must be TND or PA_BBND")
    fa, fw = prepare_front(
        x,
        wqa,
        descale_wqa,
        ww,
        norm_weight,
        workspace=front_workspace,
        norm_eps=norm_eps,
        softmax_scale=softmax_scale,
    )
    ta, tw = prepare_tail(
        fw.rq,
        fw.rs,
        wqb,
        descale_wqb,
        fw.w,
        rope,
        key,
        descale_key,
        candidate_block_indices,
        candidate_block_length,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        candidate_block_size=candidate_block_size,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        workspace=tail_workspace,
        topk=topk,
    )
    args = fa + ta + pa_args
    batch = seqused_q.numel()
    # Resource limits are dynamic and must not be hidden by the compile cache.
    with torch.npu.device(x.device):
        cube_cores = check_platform()
        compiler = functools.partial(
            compile_mega_reindex,
            x.shape[0],
            batch,
            key.shape[0] // batch,
            topk,
            norm_eps,
            softmax_scale,
            x.device.index,
            signature(args),
            page_size,
        )
        program = compiler(cube_cores)
    return ResourceCheckedProgram(program, x.device, cube_cores, compiler), args, fw, tw


def mega_reindex(
    x,
    wqa,
    descale_wqa,
    ww,
    norm_weight,
    wqb,
    descale_wqb,
    rope,
    key=None,
    descale_key=None,
    *,
    front_workspace=None,
    tail_workspace=None,
    norm_eps=1e-20,
    softmax_scale=1 / 64,
    topk=512,
    candidate_block_indices,
    candidate_block_length,
    cu_seqlens_q,
    cu_seqlens_k=None,
    seqused_q,
    seqused_k,
    candidate_block_size=8,
    mask_mode=3,
    cmp_ratio=1,
    layout_k="TND",
    block_table=None,
    k=None,
    descale_k=None,
):
    """Run L24/L28/L32/L36 candidate-consumer stages in one NPU kernel.

    Returns INT32 indices[T, topk]. Reuse both workspaces on one stream; the
    returned buffer is overwritten by the next invocation with that workspace.
    Exclude first compilation and workspace allocation from performance timing.
    """
    program, args, fw, tw = prepare_mega_reindex(
        x,
        wqa,
        descale_wqa,
        ww,
        norm_weight,
        wqb,
        descale_wqb,
        rope,
        key,
        descale_key,
        front_workspace=front_workspace,
        tail_workspace=tail_workspace,
        norm_eps=norm_eps,
        softmax_scale=softmax_scale,
        topk=topk,
        candidate_block_indices=candidate_block_indices,
        candidate_block_length=candidate_block_length,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        candidate_block_size=candidate_block_size,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_k=layout_k,
        block_table=block_table,
        k=k,
        descale_k=descale_k,
    )
    program(*args)
    return tw.indices
