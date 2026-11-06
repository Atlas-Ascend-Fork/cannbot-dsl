# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FlashAttention FP8 全量化 的 TND 精度测试。"""

from __future__ import annotations

import logging
import math

import pytest
import torch

from _samples_path import load_sample

flash_attn_fp8_fullquant = load_sample(
    "attention/flash_attn_fp8_fullquant/flash_attn_fp8_fullquant.py"
).flash_attn_fp8_fullquant

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
NEG_MAX = -3.4028234663852886e38
SCALE_ROWS = 4


# 参考计算与数据构造
LOGGER = logging.getLogger(__name__)


def per_token_scale(t: torch.Tensor) -> torch.Tensor:
    """计算 BNSD 张量的逐 token 尺度：``scale=448/max(abs(x))``。"""
    row_max = t.abs().amax(dim=3, keepdim=True).clamp_min(1e-8)
    return (FP8_MAX / row_max).float()


def per_head_scale(t: torch.Tensor) -> torch.Tensor:
    """计算 BNSD 张量的逐 head 尺度：``scale=448/max(abs(x_head))``。"""
    head_max = t.abs().amax(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)
    return (FP8_MAX / head_max).float()


def quantize(t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """执行 ``cast_fp8(clamp(x*scale, -448, 448))``。"""
    return (t.float() * scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)


def broadcast_kv(n_q: int, n_kv: int, t: torch.Tensor) -> torch.Tensor:
    """将 KV head 扩展到 Q head 数。"""
    if n_q == n_kv:
        return t.contiguous()
    return t.repeat_interleave(n_q // n_kv, dim=1).contiguous()


def fa_golden(
    q_fp8,
    k_fp8,
    v_fp8,
    deq_q,
    deq_k,
    deq_v,
    *,
    causal=True,
    softmax_scale=None,
    quantize_p=True,
):
    """计算已量化 FP8 输入的全量参考结果，输入和输出均按 BNSD。

    ``S=(Q@K^T)*(deq_q/sqrt(D))*deq_k``；应用 sparse mode 3 因果 mask
    后计算 ``E=exp(S-rowmax(S))``。MM2 使用 ``P=cast_fp8(E)``，而分母
    使用 Cast 前的 ``sum(E)``，最终 ``O=(P@V)*deq_v/sum(E)``。
    """
    q = q_fp8.float()
    k = k_fp8.float()
    v = v_fp8.float()
    batch_size, num_query_heads, query_length, head_dim = q.shape
    num_kv_heads, kv_length = k.shape[1], k.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    k = broadcast_kv(num_query_heads, num_kv_heads, k)
    v = broadcast_kv(num_query_heads, num_kv_heads, v)
    dk = broadcast_kv(num_query_heads, num_kv_heads, deq_k.float())
    dv = broadcast_kv(
        num_query_heads, num_kv_heads, deq_v.float().expand(1, num_kv_heads, 1, 1)
    )
    dq = deq_q.float()

    s = torch.matmul(q, k.transpose(-1, -2))
    s = s * (dq * softmax_scale) * dk.transpose(-1, -2)

    if causal:
        qi = torch.arange(query_length).view(1, 1, -1, 1)
        ki = torch.arange(kv_length).view(1, 1, 1, -1)
        blocked = ki > (qi + (kv_length - query_length))
        s = s.masked_fill(blocked, NEG_MAX)

    m = s.amax(dim=-1, keepdim=True)
    p = torch.exp(s - m)
    denom = p.sum(dim=-1, keepdim=True)

    if quantize_p:
        p = p.clamp(-FP8_MAX, FP8_MAX).to(FP8).float()

    out = torch.matmul(p, v) * dv
    out = out / denom.clamp_min(1e-30)

    return torch.where(denom <= 0, torch.zeros_like(out), out)


def fa_golden_online(
    q_fp8,
    k_fp8,
    v_fp8,
    deq_q,
    deq_k,
    deq_v,
    *,
    s2_base_size=256,
    causal=True,
    softmax_scale=None,
    quantize_p=True,
    q_offset=0,
    s1_full=None,
    quant_scale_p=1.0,
):
    """按 KV 分块计算与设备流水一致的在线 Softmax 参考结果。

    对每个 KV 块 ``j``：

    ``m_new=max(m_old,rowmax(S_j)-ln(quant_scale_p))``；
    ``corr=exp(m_old-m_new)``；
    ``P_j=cast_fp8(exp(S_j-m_new))``；
    ``l=l*corr+rowsum(exp(S_j-m_new))``；
    ``acc=acc*corr+(P_j@V_j)*deq_v``；最后 ``O=acc/l``。

    ``q_offset`` 和 ``s1_full`` 用于只计算部分 Q 行，同时保持其在完整
    序列中的因果位置，避免长序列 golden 占用过多 CPU 时间和内存。
    """
    q = q_fp8.float()
    k = k_fp8.float()
    v = v_fp8.float()
    batch_size, num_query_heads, query_length, head_dim = q.shape
    num_kv_heads, kv_length = k.shape[1], k.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    k = broadcast_kv(num_query_heads, num_kv_heads, k)
    v = broadcast_kv(num_query_heads, num_kv_heads, v)
    dk = broadcast_kv(num_query_heads, num_kv_heads, deq_k.float())
    dv = broadcast_kv(
        num_query_heads, num_kv_heads, deq_v.float().expand(1, num_kv_heads, 1, 1)
    )
    dq = deq_q.float()

    m = torch.full(
        (batch_size, num_query_heads, query_length, 1), NEG_MAX, dtype=torch.float32
    )
    softmax_sum = torch.zeros(
        (batch_size, num_query_heads, query_length, 1), dtype=torch.float32
    )
    acc = torch.zeros(
        (batch_size, num_query_heads, query_length, head_dim), dtype=torch.float32
    )

    qi = torch.arange(q_offset, q_offset + query_length).view(1, 1, -1, 1)
    delta = kv_length - (query_length if s1_full is None else s1_full)
    for start in range(0, kv_length, s2_base_size):
        stop = min(start + s2_base_size, kv_length)
        kj = k[:, :, start:stop, :]
        vj = v[:, :, start:stop, :]
        dkj = dk[:, :, start:stop, :]

        s = torch.matmul(q, kj.transpose(-1, -2))
        s = s * (dq * softmax_scale) * dkj.transpose(-1, -2)

        if causal:
            ki = torch.arange(start, stop).view(1, 1, 1, -1)
            s = s.masked_fill(ki > (qi + delta), NEG_MAX)

        m_new = torch.maximum(
            m, s.amax(dim=-1, keepdim=True) - math.log(float(quant_scale_p))
        )
        corr = torch.exp(m - m_new)
        p = torch.exp(s - m_new)
        softmax_sum = softmax_sum * corr + p.sum(dim=-1, keepdim=True)
        if quantize_p:
            p = p.clamp(-FP8_MAX, FP8_MAX).to(FP8).float()
        acc = acc * corr + torch.matmul(p, vj) * dv
        m = m_new

    out = acc / softmax_sum.clamp_min(1e-30)
    return torch.where(softmax_sum <= 0, torch.zeros_like(out), out)


def make_block_table(seq_lens, block_size, num_blocks=None, seed=1234):
    """随机生成物理块映射表。"""
    per_batch = [math.ceil(s / block_size) for s in seq_lens]
    total = sum(per_batch)
    n = num_blocks or total
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g)[:total]
    table = torch.full((len(seq_lens), max(per_batch)), -1, dtype=torch.int64)
    i = 0
    for b, cnt in enumerate(per_batch):
        table[b, :cnt] = order[i : i + cnt]
        i += cnt
    return table


def pack_kv_cache(
    data_fp8, scale_f32, seq_lens, block_size, block_table, num_blocks=None
):
    """构造 ``[num_blocks,Nkv,block_size+4,D]`` 的 uint8 PA 缓存。

    前 ``block_size`` 行保存 FP8 数据；末 4 行的字节重新解释为
    ``block_size`` 个 FP32 K 反量化尺度。V 缓存不传 ``scale_f32``。
    """
    batch_size, num_kv_heads, _, head_dim = data_fp8.shape
    per_batch = [math.ceil(s / block_size) for s in seq_lens]
    nb = num_blocks or sum(per_batch)
    rows = block_size + SCALE_ROWS
    cache = torch.zeros(nb, num_kv_heads, rows, head_dim, dtype=torch.uint8)

    assert rows * head_dim % 4 == 0
    scale_off = block_size * head_dim // 4
    flat = cache.reshape(nb, num_kv_heads, rows * head_dim)
    view_f32 = flat.view(torch.float32)

    for b in range(batch_size):
        for blk in range(per_batch[b]):
            pid = int(block_table[b, blk])
            s0 = blk * block_size
            s1 = min(s0 + block_size, seq_lens[b])
            n = s1 - s0
            if n <= 0:
                continue
            cache[pid, :, :n, :] = data_fp8[b, :, s0:s1, :].view(torch.uint8)
            if scale_f32 is not None:
                view_f32[pid, :, scale_off : scale_off + n] = scale_f32[b, :, s0:s1, 0]

    return flat.reshape(nb, num_kv_heads, rows, head_dim).contiguous()


def cache_f32_view(cache_u8):
    """以 FP32 视图访问同一缓存中的尺度区域，不复制底层存储。"""
    nb, nkv, rows, d = cache_u8.shape
    return (
        cache_u8.reshape(nb, nkv, rows * d)
        .view(torch.float32)
        .reshape(nb, nkv, rows * d // 4 // d, d)
    )


def gen_case(
    batch_size,
    num_query_heads,
    num_kv_heads,
    query_length,
    kv_length,
    head_dim,
    *,
    block_size=128,
    seed=0,
    causal=True,
):
    """生成输入、尺度、缓存与参考结果。"""
    g = torch.Generator().manual_seed(seed)
    q16 = torch.randn(
        batch_size, num_query_heads, query_length, head_dim, generator=g
    ).half()
    k16 = torch.randn(batch_size, num_kv_heads, kv_length, head_dim, generator=g).half()
    v16 = torch.randn(batch_size, num_kv_heads, kv_length, head_dim, generator=g).half()

    sq, sk, sv = per_token_scale(q16), per_token_scale(k16), per_head_scale(v16)
    q8, k8, v8 = quantize(q16, sq), quantize(k16, sk), quantize(v16, sv)
    dq, dk, dv = 1.0 / sq, 1.0 / sk, 1.0 / sv

    out = fa_golden(q8, k8, v8, dq, dk, dv, causal=causal)

    table = make_block_table([kv_length] * batch_size, block_size)
    k_cache = pack_kv_cache(k8, dk, [kv_length] * batch_size, block_size, table)
    v_cache = pack_kv_cache(v8, None, [kv_length] * batch_size, block_size, table)

    return {
        "q_fp8": q8,
        "k_fp8": k8,
        "v_fp8": v8,
        "deq_q": dq,
        "deq_k": dk,
        "deq_v": dv,
        "block_table": table,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "golden": out,
    }


BLOCK_SIZE = 128
ULP_TOL = 6.0
FP16_RTOL = 0.005
FP16_ATOL = 0.000025
FAIL_RATIO_LIMIT = 0.005
PASS_PERCENT_LIMIT = (1.0 - FAIL_RATIO_LIMIT) * 100.0
MAX_RELATIVE_ERROR_LIMIT = 10.0
RELATIVE_FLOOR = (1.0 / (1 << 14)) / FP16_RTOL
RELATIVE_EPSILON = 1e-10

TND_CASES = (
    pytest.param([1], [1], [1], 1, 1, 128, 1, 0, "on", 1.0, id="on_tiny"),
    pytest.param(
        [127, 256, 513],
        [382, 256, 576],
        [127, 383, 896],
        2,
        2,
        128,
        4,
        2,
        "on",
        1.0,
        id="on_ragged_batch3",
    ),
    pytest.param(
        [127, 191, 128, 191, 512],
        [128, 192, 255, 382, 512],
        [127, 318, 446, 637, 1149],
        4,
        2,
        128,
        16,
        4,
        "on",
        0.5,
        id="on_gqa_p0p5",
    ),
    pytest.param(
        [1, 31, 63, 64, 65, 127, 128],
        [32, 542, 94, 95, 128, 255, 129],
        [1, 32, 95, 159, 224, 351, 479],
        8,
        1,
        128,
        1,
        6,
        "on",
        1.0,
        id="on_batch7_mixed_tail",
    ),
    pytest.param([1], [256], [1], 8, 4, 128, 4, 8, "on", 448.0, id="on_gqa_p448"),
    pytest.param(
        [1, 63, 64],
        [512, 831, 1087],
        [1, 64, 128],
        16,
        2,
        128,
        16,
        10,
        "on",
        1.0,
        id="on_large_gqa",
    ),
    pytest.param(
        [128, 129],
        [256, 385],
        [128, 257],
        4,
        1,
        128,
        8,
        12,
        "on",
        3.0,
        id="on_tile_boundary_p3",
    ),
    pytest.param(
        [64, 257, 511],
        [320, 640, 768],
        [64, 321, 832],
        8,
        2,
        128,
        12,
        14,
        "on",
        1.0,
        id="on_long_ragged",
    ),
    pytest.param(
        [1], [256], [1], 1, 1, 128, 1, 16, "off", 1.0, id="off_tiny_full_scan"
    ),
    pytest.param(
        [127, 256],
        [382, 512],
        [127, 383],
        2,
        2,
        128,
        4,
        18,
        "off",
        1.0,
        id="off_batch2",
    ),
    pytest.param(
        [129, 255, 384],
        [256, 511, 640],
        [129, 384, 768],
        8,
        1,
        128,
        8,
        20,
        "off",
        0.5,
        id="off_gqa_ragged",
    ),
    pytest.param(
        [64, 128],
        [128, 384],
        [64, 192],
        16,
        2,
        128,
        16,
        22,
        "off",
        torch.tensor([3.0], dtype=torch.float32),
        id="off_large_gqa_p_tensor",
    ),
)


# 精度判定
def _fp16_ulp(value):
    if value <= 0 or not math.isfinite(value):
        return 2.0**-24
    return max(2.0 ** (math.floor(math.log2(value)) - 10), 2.0**-24)


def _ascendc_fp16_metrics(got, expected):
    """计算 AscendC FIA 的 FP16 精度指标。

    元素使用 ``rtol=0.005``、``atol=2.5e-5`` 判定；整体要求通过率
    不低于 99.5%，且失败元素最大归一化相对误差小于 10。
    """
    real = got.detach().cpu().to(torch.float32).flatten()
    expect = expected.detach().cpu().to(torch.float32).flatten()
    if real.numel() != expect.numel():
        raise ValueError(
            f"output size mismatch: npu={real.numel()} golden={expect.numel()}"
        )
    if real.numel() == 0:
        return {
            "passed": True,
            "pass_percent": 100.0,
            "fail_count": 0,
            "total": 0,
            "max_relative_error": 0.0,
            "max_abs_error": 0.0,
        }

    close = torch.isclose(real, expect, rtol=FP16_RTOL, atol=FP16_ATOL, equal_nan=True)
    mismatch = ~close
    fail_count = int(mismatch.sum().item())
    total = real.numel()
    pass_percent = (total - fail_count) / total * 100.0

    diff_abs = torch.abs(real - expect)
    denominator = torch.maximum(torch.abs(real), torch.abs(expect))
    denominator = (
        torch.maximum(
            denominator,
            torch.tensor(RELATIVE_FLOOR, dtype=torch.float32),
        )
        + RELATIVE_EPSILON
    )
    relative_error = (diff_abs / denominator)[mismatch]
    relative_error = torch.where(
        torch.isnan(relative_error) | torch.isinf(relative_error),
        torch.tensor(float("inf"), dtype=torch.float32),
        relative_error,
    )
    max_relative_error = (
        float(relative_error.max().item()) if relative_error.numel() else 0.0
    )
    max_abs_error = float(diff_abs.max().item())
    passed = (
        pass_percent >= PASS_PERCENT_LIMIT
        and max_relative_error < MAX_RELATIVE_ERROR_LIMIT
    )
    return {
        "passed": passed,
        "pass_percent": pass_percent,
        "fail_count": fail_count,
        "total": total,
        "max_relative_error": max_relative_error,
        "max_abs_error": max_abs_error,
    }


def _assert_close(got, expected, label):
    got = got.cpu().float()
    expected = expected.float()
    metrics = _ascendc_fp16_metrics(got, expected)
    scale = max(expected.abs().max().item(), 1e-30)
    ulp = metrics["max_abs_error"] / _fp16_ulp(scale)
    LOGGER.info(
        f"{label}: pass={metrics['pass_percent']:.6f}% "
        f"fail={metrics['fail_count']}/{metrics['total']} "
        f"max_re={metrics['max_relative_error']:.6g} "
        f"max_abs={metrics['max_abs_error']:.6g} "
        f"fp16_ulp={ulp:.3f} (diagnostic only)"
    )
    assert metrics["passed"], (
        f"{label}: AscendC FP16 comparison failed: "
        f"pass={metrics['pass_percent']:.6f}% "
        f"(required >= {PASS_PERCENT_LIMIT:.1f}%), "
        f"max_re={metrics['max_relative_error']:.6g} "
        f"(required < {MAX_RELATIVE_ERROR_LIMIT:g})"
    )


def _run_tnd_case(
    q_lengths,
    kv_lengths,
    actual_seq,
    nq,
    nkv,
    d,
    block_dim,
    seed,
    mask_mode,
    quant_scale_p,
):
    """构造变长 TND 输入，并逐 batch 计算在线 Softmax golden。"""
    batch = len(q_lengths)
    s1 = max(q_lengths)
    s2 = max(kv_lengths)
    generator = torch.Generator().manual_seed(seed)

    q16_parts = [
        torch.randn(1, nq, q_len, d, generator=generator).half() for q_len in q_lengths
    ]
    k16 = torch.randn(batch, nkv, s2, d, generator=generator).half()
    v16 = torch.randn(batch, nkv, s2, d, generator=generator).half()

    q8_parts = []
    dq_parts = []
    for q16 in q16_parts:
        sq = per_token_scale(q16)
        q8_parts.append(quantize(q16, sq))
        dq_parts.append(1.0 / sq)

    sk = per_token_scale(k16)
    sv = per_head_scale(v16)
    k8 = quantize(k16, sk)
    v8 = quantize(v16, sv)
    dk = 1.0 / sk
    dv = 1.0 / sv

    table = make_block_table(kv_lengths, BLOCK_SIZE, seed=seed + 1234)
    k_cache = pack_kv_cache(k8, dk, kv_lengths, BLOCK_SIZE, table)
    v_cache = pack_kv_cache(v8, None, kv_lengths, BLOCK_SIZE, table)

    q_tnd = torch.cat(
        [q.permute(0, 2, 1, 3).reshape(-1, nq, d) for q in q8_parts], dim=0
    ).contiguous()
    dq_tnd = torch.cat(
        [scale.permute(1, 0, 2, 3).reshape(nq, -1, 1) for scale in dq_parts],
        dim=1,
    ).contiguous()

    expected_parts = []
    for batch_idx, q_len in enumerate(q_lengths):
        kv_len = kv_lengths[batch_idx]
        expected = fa_golden_online(
            q8_parts[batch_idx],
            k8[batch_idx : batch_idx + 1, :, :kv_len, :],
            v8[batch_idx : batch_idx + 1, :, :kv_len, :],
            dq_parts[batch_idx],
            dk[batch_idx : batch_idx + 1, :, :kv_len, :],
            dv,
            s2_base_size=256,
            causal=mask_mode != "off",
            s1_full=q_len,
            quant_scale_p=quant_scale_p,
        )
        expected_parts.append(expected.permute(0, 2, 1, 3).reshape(q_len, nq, d))
    expected_tnd = torch.cat(expected_parts, dim=0).contiguous()

    assert actual_seq == list(torch.tensor(q_lengths).cumsum(0).tolist())
    got = flash_attn_fp8_fullquant(
        q_tnd,
        k_cache,
        v_cache,
        table,
        dq_tnd,
        dv.reshape(nkv),
        s1,
        s2,
        nkv,
        block_dim=block_dim,
        layout="TND",
        actual_seq=actual_seq,
        actual_seq_kv=kv_lengths,
        mask_mode=mask_mode,
        **({} if float(quant_scale_p) == 1.0 else {"quant_scale_p": quant_scale_p}),
    )
    _assert_close(
        got,
        expected_tnd,
        f"TND {mask_mode} B{batch} N{nq}:{nkv} "
        f"S{s1}:{s2} Pscale={float(quant_scale_p):g}",
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "q_lengths,kv_lengths,actual_seq,nq,nkv,d,block_dim,seed,mask_mode,quant_scale_p",
    TND_CASES,
)
def test_flash_attn_fp8_fullquant_tnd(
    q_lengths,
    kv_lengths,
    actual_seq,
    nq,
    nkv,
    d,
    block_dim,
    seed,
    mask_mode,
    quant_scale_p,
):
    _run_tnd_case(
        q_lengths,
        kv_lengths,
        actual_seq,
        nq,
        nkv,
        d,
        block_dim,
        seed,
        mask_mode,
        quant_scale_p,
    )
