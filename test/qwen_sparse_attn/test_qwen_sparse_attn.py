# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""NPU precision tests for Qwen Sparse Attention (block_size=128)."""

import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.npu

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "samples" / "qwen_sparse_attn"
sys.path.insert(0, str(SAMPLE_DIR))
import qwen_sparse_attn as qsa  # noqa: E402
import qwen_sparse_attn_metadata as metadata128  # noqa: E402


def _causal_mask(device):
    return torch.triu(
        torch.ones((2048, 2048), dtype=torch.int8, device=device), diagonal=1
    )


SCENARIOS = (
    "mixed",
    "reordered",
    "duplicates",
    "shuffled_pages",
    "custom_scale",
    "q_longer_than_kv",
)


@pytest.mark.parametrize(
    "input_dtype,output_dtype",
    [
        (torch.float16, torch.float16),
        (torch.float8_e4m3fn, torch.float16),
        (torch.float8_e4m3fn, torch.bfloat16),
    ],
)
def test_single_key_dtype_paths(input_dtype, output_dtype):
    """A single visible key makes the exact result V for every query head."""
    pytest.importorskip("torch_npu")
    q = torch.zeros((1, 3, 128)).to(input_dtype).npu()
    k = torch.zeros((1, 128, 1, 128)).to(input_dtype).npu()
    v = torch.full((1, 128, 1, 128), 0.5).to(input_dtype).npu()
    indices = torch.zeros((1, 1, 1), dtype=torch.int32).npu()
    counts = torch.ones((1, 1), dtype=torch.int32).npu()
    cu = torch.tensor([0, 1], dtype=torch.int64).npu()
    lengths = torch.ones(1, dtype=torch.int32).npu()
    table = torch.zeros((1, 1), dtype=torch.int32).npu()
    planned = metadata128.qwen_sparse_attn_metadata(
        indices, counts, cu, lengths, q, table, seqused_q=lengths, pack_queries=10
    )
    mode = (
        qsa.QuantMode.FP8_FULL_QUANT
        if input_dtype == torch.float8_e4m3fn
        else qsa.QuantMode.NO_QUANT
    )
    actual, _ = qsa.qwen_sparse_attn(
        q,
        k,
        v,
        indices,
        counts,
        [1, 128],
        attn_mask=_causal_mask(q.device),
        cu_seqlens_q=cu,
        seqused_q=lengths,
        seqused_kv=lengths,
        block_table=table,
        metadata=planned,
        quant_mode=mode,
        attention_out_dtype=output_dtype,
    )
    torch.npu.synchronize()
    assert actual.dtype == output_dtype
    torch.testing.assert_close(
        actual.cpu(), torch.full((1, 3, 128), 0.5, dtype=output_dtype), rtol=0, atol=0
    )


def _cube_dot(lhs, rhs):
    acc = torch.zeros(lhs.shape[0], rhs.shape[1], dtype=torch.float32)
    lhs, rhs = lhs.double(), rhs.double()
    for begin in range(0, lhs.shape[1], 16):
        end = begin + 16
        acc = (acc.double() + lhs[:, begin:end] @ rhs[begin:end]).float()
    return acc


def _make_block128(scenario):
    generator = torch.Generator().manual_seed(7301)
    q = (torch.randn(5, 24, 128, generator=generator) * 0.3).bfloat16()
    k = torch.randn(8, 128, 2, 128, generator=generator).bfloat16()
    v = torch.randn(8, 128, 2, 128, generator=generator).bfloat16()
    cu = torch.tensor([0, 3, 5], dtype=torch.int64)
    used = torch.tensor([3, 1], dtype=torch.int32)
    kv = torch.tensor([258, 129], dtype=torch.int32)
    table = torch.tensor([[5, 2, 7, 1], [3, 0, 6, 4]], dtype=torch.int32)
    indices = torch.zeros(2, 5, 5, dtype=torch.int32)
    for head in range(2):
        values = [1, 0, 2, -1, 3] if head == 0 else [2, 1, 1, -1, 3]
        # every token of a head selects the same block ids; the values land in
        # slots 0..4, which is what counts[h][token] = 5 makes the kernel read
        indices[head] = torch.tensor(values, dtype=torch.int32)
    counts = torch.full((2, 5), 5, dtype=torch.int32)
    counts[0, 1] = 0
    scale = 0.0
    if scenario == "reordered":
        indices[..., :3] = indices[..., :3].flip(-1)
    elif scenario == "duplicates":
        indices[..., :3] = indices[..., :1]
    elif scenario == "shuffled_pages":
        table = table.flip(-1).contiguous()
    elif scenario == "custom_scale":
        scale = 0.125
    elif scenario == "q_longer_than_kv":
        kv[0] = 2
        indices[:, :3].zero_()
        counts[:, :3].fill_(1)
    return (q, k, v, indices, counts, cu, kv, table), used, scale


def _golden_block128(args, used, scale):
    q, k, v, indices, counts, cu, kv, table = args
    out = torch.zeros_like(q)
    scale = float(torch.tensor(scale or 128**-0.5).bfloat16().float())
    group = q.shape[1] // k.shape[2]
    for batch in range(cu.numel() - 1):
        lo, _hi = int(cu[batch]), int(cu[batch + 1])
        qlen = int(used[batch])
        if int(kv[batch]) == 0:
            continue
        for token in range(lo, lo + qlen):
            position = int(kv[batch]) - qlen + token - lo
            for head in range(k.shape[2]):
                group_start = head * group
                group_end = group_start + group
                query = q[token, group_start:group_end].float()
                maximum = torch.full((group,), -3e38)
                denominator = torch.zeros(group)
                numerator = torch.zeros(group, 128)
                for slot in range(int(counts[head, token])):
                    logical = int(indices[head, token, slot])
                    if logical < 0 or logical >= table.shape[1]:
                        continue
                    valid = min(128, max(0, position - logical * 128 + 1))
                    if not valid:
                        continue
                    page = int(table[batch, logical])
                    scores = _cube_dot(
                        query, k[page, :valid, head].float().T
                    ).bfloat16()
                    scores = (
                        (scores * torch.tensor(scale, dtype=torch.bfloat16))
                        .bfloat16()
                        .float()
                    )
                    new_max = torch.maximum(maximum, scores.max(-1).values)
                    alpha = (maximum - new_max).exp()
                    probability = (scores - new_max[:, None]).exp()
                    denominator = denominator * alpha + probability.sum(-1)
                    numerator = numerator * alpha[:, None] + _cube_dot(
                        probability.bfloat16(), v[page, :valid, head]
                    )
                    maximum = new_max
                nonzero = denominator != 0
                value = torch.zeros_like(numerator)
                value[nonzero] = numerator[nonzero] / denominator[nonzero, None]
                out[token, group_start:group_end] = value.bfloat16()
    return out


def _run_block128(scenario):
    args, used, scale = _make_block128(scenario)
    expected = _golden_block128(args, used, scale)
    device = [value.npu() for value in args]
    q, k, v, indices, counts, cu, kv, table = device
    used_device = used.npu()
    planned = metadata128.qwen_sparse_attn_metadata(
        indices,
        counts,
        cu,
        kv,
        q,
        table,
        seqused_q=used_device,
    )
    actual, _ = qsa.qwen_sparse_attn(
        q,
        k,
        v,
        indices,
        counts,
        [1, 128],
        attn_mask=_causal_mask(q.device),
        cu_seqlens_q=cu,
        seqused_q=used_device,
        seqused_kv=kv,
        block_table=table,
        metadata=planned,
        softmax_scale=scale,
    )
    return actual, expected.npu()


def _make_block128_group(group):
    """构造覆盖 packed-query、GQA 分块边界和尾块的单 KV-head 用例。"""
    generator = torch.Generator().manual_seed(9100 + group)
    token_num = 5
    q = (torch.randn(token_num, group, 128, generator=generator) * 0.3).bfloat16()
    k = torch.randn(4, 128, 1, 128, generator=generator).bfloat16()
    v = torch.randn(4, 128, 1, 128, generator=generator).bfloat16()
    indices = torch.tensor([[[0, 1, 2] for _ in range(token_num)]], dtype=torch.int32)
    counts = torch.full((1, token_num), 3, dtype=torch.int32)
    cu = torch.tensor([0, token_num], dtype=torch.int64)
    used = torch.tensor([token_num], dtype=torch.int32)
    # Query 位置为 255..259：前两页完整可见，第三页逐 token 增长。
    kv = torch.tensor([260], dtype=torch.int32)
    table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    return (q, k, v, indices, counts, cu, kv, table), used


def _run_block128_group(group):
    args, used = _make_block128_group(group)
    expected = _golden_block128(args, used, 0.0)
    q, k, v, indices, counts, cu, kv, table = [value.npu() for value in args]
    pack_queries = max(1, 32 // group)
    planned = metadata128.qwen_sparse_attn_metadata(
        indices,
        counts,
        cu,
        kv,
        q,
        table,
        seqused_q=used.npu(),
        pack_queries=pack_queries,
    )
    actual, _ = qsa.qwen_sparse_attn(
        q,
        k,
        v,
        indices,
        counts,
        [1, 128],
        attn_mask=_causal_mask(q.device),
        cu_seqlens_q=cu,
        seqused_q=used.npu(),
        seqused_kv=kv,
        block_table=table,
        metadata=planned,
    )
    return actual, expected.npu()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_block128_precision(scenario):
    pytest.importorskip("torch_npu")
    actual, expected = _run_block128(scenario)
    torch.npu.synchronize()
    error = (actual.float() - expected.float()).abs()
    over = error > 5e-3 * expected.float().abs()
    assert torch.isfinite(actual).all()
    assert int(over.sum()) * 200 <= actual.numel(), (
        f"{scenario} max_abs={error.max().item():.3e}, over={int(over.sum())}/{actual.numel()}"
    )


@pytest.mark.parametrize("group", [3, 7, 8, 10, 12, 16, 17, 31, 32, 33, 64, 65, 128])
def test_block128_group_precision(group):
    pytest.importorskip("torch_npu")
    actual, expected = _run_block128_group(group)
    torch.npu.synchronize()
    error = (actual.float() - expected.float()).abs()
    over = error > 5e-3 * expected.float().abs()
    assert torch.isfinite(actual).all()
    assert int(over.sum()) * 200 <= actual.numel(), (
        f"group={group} max_abs={error.max().item():.3e}, over={int(over.sum())}/{actual.numel()}"
    )
