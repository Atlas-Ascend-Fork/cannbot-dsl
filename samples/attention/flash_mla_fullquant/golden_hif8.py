# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent FP32 golden for the flash_mla_fullquant sample.

Implements real page reorder via block_table, both quant modes
(FP8 E4M3 / HIF8), causal mask, per-token-head q_descale, multi-batch
and LSE.  The P tile is quantized per S2-block to mirror the kernel's
online-softmax quantization points.
"""

import math

import torch

S2 = 256
PAGE = 128
D_QK = 576
D_V = 512


def _gather_rows(k_cache, block_table, kv_len, layout_kv):
    n_pages = (kv_len + PAGE - 1) // PAGE
    pages = block_table[:n_pages]
    if layout_kv == "PA_BBND" or layout_kv == "PA_BNBD":
        rows = k_cache[pages].reshape(-1, D_QK)[:kv_len]
    else:
        raise NotImplementedError(f"golden layout_kv={layout_kv}")
    return rows.float()


def _hif8_conv():
    import torch_npu

    return torch_npu._C._cd


def _to_hif8(values):
    """Quantize fp32 -> HIF8 bit codes (int8 storage) on NPU."""
    conv = _hif8_conv()
    shape = values.shape
    src = values.reshape(1, -1).npu()
    dst = torch.empty(src.numel(), dtype=torch.uint8, device="npu")
    conv.cast_to_fp8_noalloc(src, dst.view(1, -1), conv.DType.hifloat8)
    return dst.view(shape).view(torch.int8).cpu()


def _from_hif8(codes):
    """Dequantize HIF8 bit codes -> fp32 on NPU."""
    conv = _hif8_conv()
    shape = codes.shape
    src = codes.view(torch.uint8).reshape(1, -1).npu()
    return (
        conv.cast_from_fp8(
            src,
            conv.DType.hifloat8,
            conv.DType.float32,
        )
        .view(shape)
        .cpu()
    )


def _quantize_p(p_tile, quant_mode):
    """Model the kernel's P quantization (conv bit cast)."""
    return _from_hif8(_to_hif8(p_tile)).to(p_tile.device)


def golden_flash_mla_fullquant(
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
    mask_mode=0,
    softmax_scale=None,
    layout_q="BSND",
    layout_kv="PA_BNBD",
    layout_out="BSND",
    return_softmax_lse=False,
):
    """Reference: FP32 online softmax over 8-bit QK/PV per S2 blocks."""
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D_V)
    kd = float(k_descale.reshape(()))

    # HIF8 inputs arrive as int8 bit storage: decode to real values
    # (torch .float() on int8 gives the raw codes, unlike fp8 tensors
    # where .float() decodes correctly).
    if quant_mode == 0:
        if q.dtype == torch.int8:
            q = _from_hif8(q)
        if k_cache.dtype == torch.int8:
            k_cache = _from_hif8(k_cache)

    # ---- normalize Q to per-batch (token, head) rows ----
    if layout_q == "BSND":
        n_batch, q_seq, n_q_heads, _ = q.shape
        q_rows = q.reshape(n_batch, q_seq * n_q_heads, D_QK).float()
        qd_rows = q_descale.reshape(n_batch, q_seq * n_q_heads)
        q_lens = [q_seq] * n_batch
    elif layout_q == "BNSD":
        n_batch, n_q_heads, q_seq, _ = q.shape
        q_rows = q.permute(0, 2, 1, 3).reshape(n_batch, q_seq * n_q_heads, D_QK).float()
        qd_rows = q_descale.permute(0, 2, 1).reshape(n_batch, q_seq * n_q_heads)
        q_lens = [q_seq] * n_batch
    elif layout_q in ("TND", "TND_NTD"):
        q_tok, n_q_heads, _ = q.shape
        if cu_seqlens_q is not None:
            cu = cu_seqlens_q.tolist()
            n_batch = len(cu) - 1
            q_lens = [cu[b + 1] - cu[b] for b in range(n_batch)]
            q_rows, qd_rows = [], []
            for b in range(n_batch):
                lo, hi = cu[b] * n_q_heads, cu[b + 1] * n_q_heads
                q_rows.append(q.reshape(q_tok * n_q_heads, D_QK)[lo:hi].float())
                qd_rows.append(q_descale.reshape(q_tok * n_q_heads)[lo:hi])
        else:
            n_batch, q_lens = 1, [q_tok]
            q_rows = [q.reshape(q_tok * n_q_heads, D_QK).float()]
            qd_rows = [q_descale.reshape(q_tok * n_q_heads)]
    else:
        raise ValueError(layout_q)

    # FD split schedule mirroring the kernel wrapper: when M-tiles
    # under-fill the cores each tile's KV stream is sliced across
    # n_slices tasks (AscendC FD semantics); P is quantized against the
    # slice-local running max and partials merge with max rescaling.
    cores = 32
    m_tile = 64
    total_tiles = sum(
        (q_lens[b] * n_q_heads + m_tile - 1) // m_tile for b in range(n_batch)
    )
    kv_blocks_max = max(
        1, max((int(cache_seqlens[b]) + S2 - 1) // S2 for b in range(n_batch))
    )
    n_slices = 1
    if 0 < total_tiles < cores:
        n_slices = max(1, min((cores + total_tiles - 1) // total_tiles, kv_blocks_max))

    outs, lses = [], []
    for b in range(n_batch):
        rows_b = q_lens[b] * n_q_heads
        qf = q_rows[b]  # (rows_b, 576)
        qd = qd_rows[b]
        kv_len = int(cache_seqlens[b])
        kv = _gather_rows(k_cache, block_table[b], kv_len, layout_kv)
        v = kv[:, :D_V]

        raw = qf @ kv.t()  # (rows_b, kv_len)
        score = raw * qd[:, None] * kd * softmax_scale
        if mask_mode == 3:
            token = torch.arange(rows_b, dtype=torch.float32) // n_q_heads
            lim = token[:, None] + (kv_len - q_lens[b])
            pos = torch.arange(kv_len, dtype=torch.float32)[None]
            score = score.masked_fill(pos > lim, float("-inf"))

        blocks = (kv_len + S2 - 1) // S2
        blocks_per_slice = (blocks + n_slices - 1) // n_slices
        partials = []
        for seg in range(n_slices):
            b0 = seg * blocks_per_slice
            b1 = min(blocks, b0 + blocks_per_slice)
            m = torch.full((rows_b,), float("-inf"))
            lsum = torch.zeros(rows_b)
            acc = torch.zeros(rows_b, D_V)
            for blk in range(b0, b1):
                kv_lo = blk * S2
                kv_hi = kv_lo + S2
                s_b = score[:, kv_lo:kv_hi]
                v_b = v[kv_lo:kv_hi]
                m_new = torch.maximum(m, s_b.max(dim=-1).values)
                corr = torch.exp(m - m_new)
                corr = torch.nan_to_num(corr, nan=0.0)
                exps = torch.exp(s_b - m_new[:, None])
                exps = torch.nan_to_num(exps, nan=0.0)
                p_q = _quantize_p(exps, quant_mode)
                lsum = lsum * corr + exps.sum(dim=-1)
                acc = acc * corr[:, None] + p_q @ v_b
                m = m_new
            partials.append((m, lsum, acc))
        m = torch.stack([p[0] for p in partials])  # (S, rows)
        lsum = torch.stack([p[1] for p in partials])
        acc = torch.stack([p[2] for p in partials])  # (S, rows, D_V)
        m_star = m.max(dim=0).values
        corr = torch.where(
            torch.isneginf(m), torch.zeros_like(m), torch.exp(m - m_star)
        )
        lsum = (lsum * corr).sum(dim=0)
        acc = (acc * corr[:, :, None]).sum(dim=0)
        m = m_star
        out = acc / lsum[:, None] * kd
        lse = m + torch.log(lsum)
        # AscendC golden semantics for fully-masked rows (no visible key,
        # e.g. bottom-right causal with q_len > kv_len): output 0 and
        # LSE = +inf instead of the NaN/-inf a plain 0/0 softmax yields.
        all_masked = torch.isneginf(m)
        if all_masked.any():
            out = torch.where(all_masked[:, None], torch.zeros_like(out), out)
            lse = torch.where(all_masked, torch.full_like(lse, float("inf")), lse)
        if seqused_q is not None:
            su = int(seqused_q.reshape(-1)[b]) * n_q_heads
            if su < rows_b:
                out = torch.cat([out[:su], torch.zeros(rows_b - su, D_V)], dim=0)
        outs.append(out.reshape(q_lens[b], n_q_heads, D_V))
        lses.append(lse.reshape(q_lens[b], n_q_heads))

    # ---- restore layout_out ----
    if layout_out in ("BSND", "BNSD"):
        if len(set(q_lens)) > 1:
            raise ValueError("BSND/BNSD output requires uniform q_len")
        out_all = torch.stack(outs, dim=0) if n_batch > 1 else outs[0].unsqueeze(0)
        if layout_out == "BNSD":
            out_all = out_all.permute(0, 2, 1, 3).contiguous()
        lse = (
            torch.stack(lses, dim=0).permute(0, 2, 1).contiguous()
            if bool(return_softmax_lse)
            else None
        )
    else:
        out_all = torch.cat(outs, dim=0)
        if layout_out == "NTD":
            out_all = out_all.permute(1, 0, 2).contiguous()
        lse = None
        if bool(return_softmax_lse):
            lse_t = torch.cat(lses, dim=0)
            # Official contract: TND and NTD both return (Q_N, Q_T).
            lse = lse_t.t().contiguous()
    return out_all, lse


def make_case(
    tokens=1,
    cache_len=1024,
    heads=96,
    block=128,
    seed=7,
    batch=1,
    reorder=True,
    quant_mode=1,
    device="npu",
    q_len_list=None,
    kv_len_list=None,
):
    """Generate an official-API test case with real page reorder."""
    torch.manual_seed(seed)
    n_q_heads = heads
    if q_len_list is None:
        q_len_list = [tokens] * batch
    n_batch = len(q_len_list)
    if kv_len_list is None:
        kv_len_list = [cache_len] * n_batch
    if len(kv_len_list) != n_batch:
        raise ValueError("kv_len_list must hold one length per batch")

    {"BSND": (sum(q_len_list), n_q_heads, D_QK)}
    q_real = torch.randn(sum(q_len_list), n_q_heads, D_QK) * 0.3
    # per-token-head quantization
    q_abs = q_real.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    q_scale = 448.0 / q_abs
    q8 = (q_real * q_scale).clamp(-448, 448)
    # HIF8 Q: quantize the per-token-head-scaled values on NPU
    q8 = _to_hif8(q8).to(device)
    q_descale = (1.0 / q_scale).squeeze(-1)

    # shared KV cache: per-tensor quantization
    kv_lens = list(kv_len_list)
    total_kv = sum(kv_lens)
    k_real = torch.randn(total_kv, D_QK) * 0.3
    k_scale = torch.tensor([448.0 / k_real.abs().max()])
    k8 = (k_real * k_scale).clamp(-448, 448)
    k8 = _to_hif8(k8).to(device)
    k_descale = 1.0 / k_scale

    # pack into pages: per-batch logical pool, random physical order.
    # Every batch reserves the same padded page count so a rectangular
    # block_table addresses each batch from its own base.
    max_pages = max((v + block - 1) // block for v in kv_lens)
    n_pages = max_pages * n_batch
    perms = []
    cache = torch.zeros(n_pages, block, 1, D_QK, dtype=k8.dtype)
    cursor = 0
    for b in range(n_batch):
        perm = torch.randperm(max_pages) if reorder else torch.arange(max_pages)
        perms.append(perm)
        k_pad = torch.zeros(max_pages * block, D_QK, dtype=k8.dtype)
        kv_len_b = kv_lens[b]
        cursor_end = cursor + kv_len_b
        k_pad[:kv_len_b] = k8[cursor:cursor_end]
        for pg in range(max_pages):
            blk_lo = pg * block
            blk_hi = blk_lo + block
            cache[perm[pg] + b * max_pages, :, 0] = k_pad[blk_lo:blk_hi]
        cursor += kv_lens[b]
    block_table = torch.stack(
        [perms[b] + b * max_pages for b in range(n_batch)], dim=0
    ).to(torch.int32)
    seqlens = torch.tensor(kv_lens, dtype=torch.int32)

    dev = torch.device(device)
    return {
        "q": q8.to(dev),
        "k_cache": cache.to(dev),
        "q_descale": q_descale.to(dev),
        "k_descale": k_descale.to(dev),
        "block_table": block_table.to(dev),
        "cache_seqlens": seqlens.to(dev),
        "quant_mode": quant_mode,
        "cu_seqlens_q": (
            torch.tensor(
                [0] + list(torch.tensor(q_len_list).cumsum(0)), dtype=torch.int32
            ).to(dev)
            if len(set(q_len_list)) > 1 or n_batch > 1
            else None
        ),
        "ref": None,  # filled by callers via golden_flash_mla_fullquant
    }
