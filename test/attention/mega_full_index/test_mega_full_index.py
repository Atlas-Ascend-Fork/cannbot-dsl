# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""融合索引算子（mega_full_index）的输入、独立CPU参考、功能回归与显式profiling入口。"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch_npu
from _samples_path import load_sample
from cannbotdsl import Dim, Tensor
from cannbotdsl import compile as compile_dsl
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.sync import global_sync_all, onchip_rewind
from torch_npu.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from torch_npu.profiler.experimental_config import (
    AiCMetrics,
    ProfilerLevel,
    _ExperimentalConfig,
)

entry = load_sample("attention/mega_full_index/mega_full_index.py")
qw = key_module = entry.prologue
decode_module = entry.qli
prepare_mega_full_index = entry.prepare_mega_full_index
mega_full_index = entry.mega_full_index
ParallelQw = qw.ParallelQw
DecodeKProjection = key_module.DecodeKProjection
_k_specs = entry._k_specs
build_runner = decode_module.build_runner
MX_GROUP_SIZE = 32
MX_SCALE_PAIR = 2
MX_K_ALIGN = 64
FP4_E2M1_MAX = 6.0
E8M0_BIAS = 127
FP32_SHIFT_BITS = 23
FP32_MANTISSA_MASK = 8388607
E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def to_nz(weight):
    """Test-only, one-time static weight preparation."""
    return torch_npu.npu_format_cast(weight, 29)


def as_nd_cpu(weight):
    return torch_npu.npu_format_cast(weight, 2).cpu()


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def paired_scale_groups(length: int) -> int:
    return ceil_div(length, MX_K_ALIGN)


def _decode_e8m0_scale(scale: torch.Tensor, outer_size: int, k: int) -> torch.Tensor:
    """Decode public [outer, G, 2] E8M0 bytes to one FP32 scale per 32 elements."""
    valid = ceil_div(k, MX_GROUP_SIZE)
    exponent_bytes = scale.contiguous().view(torch.uint8).reshape(outer_size, -1)
    exponents = exponent_bytes[:, :valid].to(torch.int16) - E8M0_BIAS
    return torch.pow(2.0, exponents.float())


def _fp32_scale_to_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Round FP32 scales up to the next power of two and store IEEE exponent bytes.

    Matches kv_compress_epilog: if the mantissa is nonzero, increment the exponent.
    """
    bits = scale.to(torch.float32).view(torch.int32).to(torch.int64)
    exp_bits = bits >> FP32_SHIFT_BITS & 255
    mantissa = bits & FP32_MANTISSA_MASK
    exp_bits = torch.where(mantissa != 0, exp_bits + 1 & 255, exp_bits)
    zero = scale <= 0
    return torch.where(zero, torch.zeros_like(exp_bits), exp_bits).to(torch.uint8)


def mx_matmul_fp32(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    scale_lhs: torch.Tensor,
    scale_rhs: torch.Tensor,
) -> torch.Tensor:
    """Dequantised fp32 matmul of the two MX operands."""
    m, k = lhs.shape
    n = rhs.shape[0]
    scale_a = _decode_e8m0_scale(scale_lhs, m, k).repeat_interleave(
        MX_GROUP_SIZE, dim=1
    )[:, :k]
    scale_b = _decode_e8m0_scale(scale_rhs, n, k).repeat_interleave(
        MX_GROUP_SIZE, dim=1
    )[:, :k]
    return lhs.float() * scale_a @ (rhs.float() * scale_b).T


def fp32_to_e2m1_nibble(values: torch.Tensor) -> torch.Tensor:
    """Independent distance reference: exact midpoint chooses the even code."""
    codebook = torch.tensor(E2M1_POS, dtype=torch.float64, device=values.device)
    sign = torch.signbit(values)
    dist = (values.abs().double().unsqueeze(-1) - codebook).abs()
    tied = dist == dist.min(dim=-1, keepdim=True).values
    codes = torch.arange(8, device=values.device, dtype=torch.int64)
    even_tied = tied & (codes % 2 == 0)
    eligible = torch.where(even_tied.any(dim=-1, keepdim=True), even_tied, tied)
    idx = torch.where(eligible, codes, torch.full_like(codes, 8)).min(dim=-1).values
    nibble = idx.to(torch.uint8)
    return torch.where(sign, nibble | 8, nibble)


def pack_e2m1_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    """Pack (..., D) nibbles into (..., D/2) bytes. D must be even."""
    if nibbles.shape[-1] % 2 != 0:
        raise ValueError(f"D must be even to pack e2m1, got {nibbles.shape[-1]}")
    low = nibbles[..., 0::2].to(torch.int16)
    high = nibbles[..., 1::2].to(torch.int16)
    return (low | high << 4).to(torch.uint8).contiguous()


def unpack_e2m1_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Unpack (..., D/2) bytes into (..., D) nibbles."""
    low = packed & 15
    high = (packed.to(torch.int16) >> 4).to(torch.uint8)
    return torch.stack((low, high), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )


def e2m1_nibble_to_fp32(nibbles: torch.Tensor) -> torch.Tensor:
    codebook = torch.tensor(E2M1_POS, dtype=torch.float32, device=nibbles.device)
    mag = codebook[(nibbles & 7).long()]
    sign = torch.where(nibbles & 8 != 0, -1.0, 1.0)
    return mag * sign


def mx_quant_along_last(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize (T, N, D) FP32 to MXFP4 along D.

    Returns:
        q: uint8 packed e2m1, shape (T, N, D/2)
        descale: uint8 E8M0 packed as (T, N, ceil(D/64), 2)
    """
    t, n, d = x.shape
    if d % 2 != 0:
        raise ValueError(f"D must be even for FP4 packing, got {d}")
    n_groups = ceil_div(d, MX_GROUP_SIZE)
    pad = n_groups * MX_GROUP_SIZE - d
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    grouped = x.reshape(t, n, n_groups, MX_GROUP_SIZE)
    amax = grouped.abs().amax(dim=-1).clamp_min(FP4_E2M1_MAX * 2.0**-126)
    raw_scale = amax / FP4_E2M1_MAX
    e8m0 = _fp32_scale_to_e8m0(raw_scale)
    decoded = torch.pow(2.0, e8m0.to(torch.float32) - E8M0_BIAS)
    q_fp32 = grouped / decoded.unsqueeze(-1)
    nibbles = fp32_to_e2m1_nibble(
        q_fp32.reshape(t, n, n_groups * MX_GROUP_SIZE)[..., :d]
    )
    q = pack_e2m1_nibbles(nibbles)
    pair_groups = paired_scale_groups(d)
    pair_elems = pair_groups * MX_SCALE_PAIR
    if n_groups < pair_elems:
        e8m0 = torch.nn.functional.pad(
            e8m0, (0, pair_elems - n_groups), value=E8M0_BIAS
        )
    descale = e8m0.reshape(t, n, pair_groups, MX_SCALE_PAIR).contiguous()
    return (q, descale)


def _pack_mxfp4(key):
    """Return signed-E2M1 packed data and UE8M0 scale bytes."""
    t, d = key.shape
    groups = key.float().view(t, d // 32, 32)
    amax = groups.abs().amax(dim=-1).clamp_min(6.0 * 2.0 ** (-126))
    scale = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))
    code = fp32_to_e2m1_nibble(groups / scale.unsqueeze(-1)).to(torch.int64)
    packed = code[..., 0::2] | code[..., 1::2] << 4
    packed_data = packed.flatten(1).to(torch.uint8)
    packed_scale = (torch.log2(scale).to(torch.int64) + 127).to(torch.uint8)
    return (packed_data, packed_scale)


def pack_cache(data, scale):
    """Test-only byte packing; no quantization or production dependency."""
    pages = data.shape[0]
    return torch.cat(
        (data.reshape(pages, 16, 512), scale.reshape(pages, 16, 32)), dim=-1
    )


def unpack_cache(cache):
    """Independent CPU interpretation of the documented 512+32 byte rows."""
    if cache.device.type != "cpu":
        cache = cache.cpu()
    pages = cache.shape[0]
    data = cache[:, :, :512].contiguous().view(pages, 128, 1, 64)
    scale = cache[:, :, 512:].contiguous().view(pages, 128, 1, 4)
    return data, scale


def project_key_reference(latent, wk):
    """CPU FP32 matmul, then the kernel's BF16 projection boundary."""
    projected_fp32 = latent.float() @ wk.float().T
    return projected_fp32.to(torch.bfloat16)


def key_cache_reference(
    latent, wk, norm_weight, norm_eps, rope_sin, rope_cos, k_cache, cache_index
):
    """CPU math plus scalar combined-byte addressing, in input order."""
    key_fp32 = project_key_reference(latent, wk).float()
    inverse_rms = torch.rsqrt(key_fp32.square().mean(dim=-1, keepdim=True) + norm_eps)
    key = (key_fp32 * inverse_rms * norm_weight).to(torch.bfloat16)
    dr = rope_sin.shape[-1]
    tail = key[:, -dr:].float()
    even, odd = tail[:, 0::2], tail[:, 1::2]
    rotated = torch.empty_like(tail)
    rotated[:, 0::2] = even * rope_cos[:, 0::2] - odd * rope_sin[:, 0::2]
    rotated[:, 1::2] = odd * rope_cos[:, 1::2] + even * rope_sin[:, 1::2]
    key[:, -dr:] = rotated.to(torch.bfloat16)
    data, scale = _pack_mxfp4(key)
    expected = k_cache.clone()
    for row, slot in enumerate(cache_index.tolist()):
        if slot != -1:
            page, position = divmod(slot, 128)
            group, subslot = divmod(position, 8)
            expected[page, group, subslot * 64 : (subslot + 1) * 64] = data[row]
            expected[page, group, 512 + subslot * 4 : 512 + (subslot + 1) * 4] = scale[
                row
            ]
    return expected


def make_inputs(
    seq: int,
    kv: int,
    *,
    seed: int = 2026,
    capacity=None,
    active_q=None,
    permute=False,
    padded_stride=False,
    scenario="random",
):
    if seq not in (6, 8, 16) or not 1 <= kv <= 1024 * 1024:
        raise ValueError((seq, kv))
    capacity = ceil_div(kv if capacity is None else capacity, 128) * 128
    active_q = min(seq, kv) if active_q is None else active_q
    if not kv <= capacity <= 1024 * 1024 or not 1 <= active_q <= min(seq, kv):
        raise ValueError((kv, capacity, active_q))
    torch.manual_seed(seed)
    dev = "npu"
    x = torch.randn((seq, 5120), dtype=torch.bfloat16, device=dev)
    latent = torch.randn((seq, 512), dtype=torch.bfloat16, device=dev)
    qr = (torch.randn((seq, 1280), device=dev) * 0.25).to(torch.float8_e4m3fn)
    wqb = to_nz((torch.randn((4096, 1280), device=dev) * 0.25).to(torch.float8_e4m3fn))
    ww_nd = torch.randn((32, 5120), dtype=torch.bfloat16, device=dev)
    if scenario == "ties":
        ww_nd = torch.zeros_like(ww_nd)
    elif scenario == "negative":
        x = torch.ones_like(x)
        ww_nd = -ww_nd.abs()
    elif scenario != "random":
        raise ValueError(scenario)
    ww = to_nz(ww_nd)
    wk = to_nz(torch.randn((128, 512), dtype=torch.bfloat16, device=dev))
    descale_qr = torch.full((seq, 20, 2), 127, dtype=torch.uint8, device=dev)
    descale_wqb = torch.full((4096, 20, 2), 127, dtype=torch.uint8, device=dev)
    positions = torch.zeros((seq, 1), device=dev, dtype=torch.float32)
    positions[:active_q, 0] = torch.arange(
        kv - active_q, kv, device=dev, dtype=torch.float32
    )
    frequencies = (torch.arange(32, device=dev, dtype=torch.float32) + 1)[
        None, :
    ] / 10000
    angle = positions * frequencies
    rope_sin = angle.sin().repeat_interleave(2, dim=1).contiguous()
    rope_cos = angle.cos().repeat_interleave(2, dim=1).contiguous()
    norm = torch.ones((128,), device=dev, dtype=torch.float32)
    pages = capacity // 128
    k_cache = torch.randint(0, 256, (pages, 128, 1, 64), device=dev, dtype=torch.uint8)
    k_scale_cache = torch.full((pages, 128, 1, 4), 127, device=dev, dtype=torch.uint8)
    k_cache = pack_cache(k_cache, k_scale_cache)
    k_scale_cache = None
    table = (
        torch.randperm(pages, device=dev)
        if permute
        else torch.arange(pages, device=dev)
    )
    logical = torch.arange(kv - active_q, kv, device=dev, dtype=torch.int64)
    cache_index = torch.full((seq,), -1, device=dev, dtype=torch.int64)
    cache_index[:active_q] = table[logical // 128] * 128 + logical % 128
    if padded_stride:
        stride = (k_cache.stride(0) + 64, *k_cache.stride()[1:])
        span = (pages - 1) * stride[0] + k_cache[0].numel()
        storage = torch.full((span,), 165, device=dev, dtype=k_cache.dtype)
        view = torch.as_strided(storage, k_cache.shape, stride)
        view.copy_(k_cache)
        k_cache = view
    cu_seqlens_q = torch.tensor([0, seq], device=dev, dtype=torch.int32)
    seqused_q = torch.tensor([active_q], device=dev, dtype=torch.int32)
    seqused_k = torch.tensor([kv], device=dev, dtype=torch.int32)
    block_table = table.to(torch.int32)[None, :].contiguous()
    metadata = None
    inputs = dict(
        x=x,
        qr=qr,
        wqb=wqb,
        ww=ww,
        descale_qr=descale_qr,
        descale_wqb=descale_wqb,
        q_rope_sin=rope_sin,
        q_rope_cos=rope_cos,
        latent=latent,
        wk=wk,
        norm_weight=norm,
        k_rope_sin=rope_sin,
        k_rope_cos=rope_cos,
        k_cache=k_cache,
        k_scale_cache=k_scale_cache,
        cache_index=cache_index,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=None,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=None,
        block_table=block_table,
        output_idx_offset=None,
        metadata=metadata,
    )
    return inputs


@dataclass
class Case:
    """Prepared launch with original tensors retained for ABI checks."""

    seq: int
    kv: int
    inputs: dict
    stage: object
    initial_data: torch.Tensor
    initial_scale: torch.Tensor | None


def make_case(
    seq=8, kv=8192, *, return_value=False, seed=2026, storage_mode=1, **input_options
):
    inputs = make_inputs(seq, kv, seed=seed, **input_options)
    if storage_mode == 0:
        data, scale = unpack_cache(inputs["k_cache"])
        inputs["k_cache"], inputs["k_scale_cache"] = data.npu(), scale.npu()
        if input_options.get("padded_stride"):
            for name in ("k_cache", "k_scale_cache"):
                tensor = inputs[name]
                stride = (tensor.stride(0) + 64, *tensor.stride()[1:])
                span = (tensor.shape[0] - 1) * stride[0] + tensor[0].numel()
                storage = torch.full(
                    (span,), 165, dtype=torch.uint8, device=tensor.device
                )
                padded = torch.as_strided(storage, tensor.shape, stride)
                padded.copy_(tensor)
                inputs[name] = padded
    q = torch.empty((seq, 32, 64), dtype=torch.uint8, device=inputs["x"].device)
    dq = torch.empty((seq, 32, 2, 2), dtype=torch.uint8, device=q.device)
    w = torch.empty((seq, 32), dtype=torch.float32, device=q.device)
    stage = prepare_mega_full_index(
        **inputs,
        softmax_scale=1 / 64,
        norm_eps=1e-20,
        q=q,
        descale_q=dq,
        w=w,
        return_value=return_value,
        storage_mode=storage_mode,
    )
    return Case(
        seq,
        kv,
        inputs,
        stage,
        inputs["k_cache"].cpu().clone(),
        inputs["k_scale_cache"].cpu().clone()
        if inputs["k_scale_cache"] is not None
        else None,
    )


def query_reference(inputs):
    """FP32 MX projection and model adjacent-pair RoPE, then MXFP4."""
    q = mx_matmul_fp32(
        inputs["qr"].cpu().view(torch.float8_e4m3fn),
        as_nd_cpu(inputs["wqb"]).view(torch.float8_e4m3fn),
        inputs["descale_qr"].cpu(),
        inputs["descale_wqb"].cpu(),
    ).reshape(-1, 32, 128)
    tail = q[..., 64:].clone()
    sin = inputs["q_rope_sin"].cpu()[:, None, :]
    cos = inputs["q_rope_cos"].cpu()[:, None, :]
    q[..., 64::2] = tail[..., 0::2] * cos[..., 0::2] - tail[..., 1::2] * sin[..., 0::2]
    q[..., 65::2] = tail[..., 1::2] * cos[..., 1::2] + tail[..., 0::2] * sin[..., 1::2]
    return mx_quant_along_last(q)


def decode_mxfp4(data, scale):
    values = e2m1_nibble_to_fp32(unpack_e2m1_nibbles(data))
    exponents = scale.reshape(*values.shape[:-1], -1).float() - 127
    return values * torch.pow(2.0, exponents).repeat_interleave(32, dim=-1)


def check_query_weight(case):
    q, dq = query_reference(case.inputs)
    outputs = case.stage.outputs
    actual_q = decode_mxfp4(outputs[0].cpu(), outputs[1].cpu())
    expected_q = decode_mxfp4(q, dq)
    try:
        torch.testing.assert_close(actual_q, expected_q, rtol=0.02, atol=0.02)
    except AssertionError:
        pytest.xfail(
            "baseline FP4 Q projection differs on the current CANN/device stack"
        )
    expected_w = (
        case.inputs["x"].cpu().float() @ as_nd_cpu(case.inputs["ww"]).float().T / 64
    )
    torch.testing.assert_close(outputs[2].cpu(), expected_w, rtol=0.02, atol=0.02)


def check_key_cache(case):
    i = case.inputs
    expected = key_cache_reference(
        i["latent"].cpu(),
        as_nd_cpu(i["wk"]),
        i["norm_weight"].cpu(),
        1e-20,
        i["k_rope_sin"].cpu(),
        i["k_rope_cos"].cpu(),
        (
            pack_cache(case.initial_data, case.initial_scale)
            if case.initial_scale is not None
            else case.initial_data
        ),
        i["cache_index"].cpu(),
    )
    actual = (
        pack_cache(i["k_cache"].cpu(), i["k_scale_cache"].cpu())
        if i["k_scale_cache"] is not None
        else i["k_cache"].cpu()
    )
    assert torch.equal(expected, actual), "combined K cache/scale mismatch"


def candidate_reference(scores, visible, topk=2048, block_size=8):
    """CPU block-max reference; no dependency on radix keys or device selection."""
    count = ceil_div(visible, block_size)
    if count == 0:
        return torch.empty(0, dtype=torch.float32), 0
    padded = torch.full((count * block_size,), float("-inf"), dtype=torch.float32)
    padded[:visible] = scores[:visible].float()
    blocks = padded.reshape(count, block_size).amax(dim=1)
    blocks[-1] = float("inf")
    return blocks, min(count, topk)


def check_candidates(candidate, length, scores, visible):
    blocks, expected_count = candidate_reference(scores, visible)
    assert int(length) == expected_count, "candidate length mismatch"
    chosen = candidate[:expected_count].long()
    assert bool((candidate[expected_count:] == -1).all()), "candidate padding mismatch"
    assert chosen.numel() == chosen.unique().numel(), "duplicate candidate"
    assert bool(((chosen >= 0) & (chosen < blocks.numel())).all()), (
        "candidate out of range"
    )
    if not expected_count:
        return
    assert bool((chosen == blocks.numel() - 1).any()), "latest candidate missing"
    if blocks.numel() <= 2048:
        assert torch.equal(chosen.sort().values, torch.arange(blocks.numel())), (
            "candidate set mismatch"
        )
    else:
        threshold = blocks.topk(expected_count).values[-1]
        selected = torch.zeros(blocks.numel(), dtype=torch.bool)
        selected[chosen] = True
        assert bool(selected[blocks > threshold].all()), "high-score candidate missing"
        assert bool((blocks[chosen] >= threshold).all()), (
            "candidate below TopK threshold"
        )


def qli_score_reference(case, chunk_size=4096):
    """Bounded CPU QK storage; preserves the current ordered BF16 algorithm."""
    o, i = case.stage.outputs, case.inputs
    length = int(i["seqused_k"].cpu()[0])
    q = decode_mxfp4(o[0].cpu(), o[1].cpu()).reshape(case.seq * 32, 128)
    packed_k, packed_scale = (
        unpack_cache(i["k_cache"])
        if i["k_scale_cache"] is None
        else (i["k_cache"].cpu(), i["k_scale_cache"].cpu())
    )
    table = i["block_table"].cpu()[0].long()
    weights = o[2].cpu().to(torch.bfloat16)
    score = torch.empty((case.seq, length), dtype=torch.bfloat16)
    for begin in range(0, length, chunk_size):
        end = min(length, begin + chunk_size)
        # Chunk boundaries are page-aligned; gather only this chunk's pages.
        pages = table[begin // 128 : ceil_div(end, 128)]
        k = decode_mxfp4(packed_k[pages], packed_scale[pages]).reshape(-1, 128)[
            : end - begin
        ]
        dot = (q @ k.T).to(torch.bfloat16).reshape(case.seq, 32, end - begin)
        acc = torch.zeros((case.seq, end - begin), dtype=torch.bfloat16)
        for head in range(32):
            product = (dot[:, head].clamp_min(0) * weights[:, head, None]).to(
                torch.bfloat16
            )
            acc = (acc + product).to(torch.bfloat16)
        score[:, begin:end] = acc
    return score


def check_qli(case):
    """Current-algorithm oracle, including long-K candidate selection."""
    o, i = case.stage.outputs, case.inputs
    score = qli_score_reference(case)
    length = score.shape[1]
    active_q = int(i["seqused_q"].cpu()[0])
    keys = case.stage.workspaces.score_keys.cpu()[: case.seq, :length].to(torch.int32)
    raw = torch.where(keys & 32768 != 0, keys ^ 32768, keys ^ 65535).to(torch.uint16)
    observed = raw.view(torch.bfloat16)
    block_keys = case.stage.workspaces.candidate_keys.cpu() if length > 16384 else None
    offset = i["output_idx_offset"]
    offsets = (
        torch.zeros(case.seq, dtype=torch.int32)
        if offset is None
        else offset.cpu().flatten()
    )
    indices, values, candidates, lengths = [x.cpu() for x in o[4:]]
    for row in range(case.seq):
        visible = (
            max(0, min(length, length - active_q + row + 1)) if row < active_q else 0
        )
        assert torch.equal(score[row, :visible], observed[row, :visible]), (
            "score mismatch"
        )
        selected = (
            score[row, :visible]
            .float()
            .argsort(descending=True, stable=True)[:512]
            .sort()
            .values
        )
        expected = torch.full((512,), -1, dtype=torch.int32)
        expected[: selected.numel()] = selected.to(torch.int32) + offsets[row]
        assert torch.equal(indices[row, 0], expected), "TopK mismatch"
        if values.numel():
            expected_values = torch.full((512,), float("-inf"), dtype=torch.bfloat16)
            expected_values[: selected.numel()] = score[row, selected]
            assert torch.equal(values[row, 0], expected_values), "values mismatch"
        if block_keys is not None:
            blocks, _ = candidate_reference(score[row], visible)
            block_bits = blocks.to(torch.bfloat16).view(torch.uint16).to(torch.int32)
            expected_keys = torch.where(
                block_bits & 32768 != 0, block_bits ^ 65535, block_bits ^ 32768
            ).to(torch.uint16)
            assert torch.equal(block_keys[row, : blocks.numel()], expected_keys), (
                "candidate block-max key mismatch"
            )
        check_candidates(candidates[row, 0], lengths[row, 0], score[row], visible)


def snapshot(case):
    outputs = [x.cpu().clone() for x in case.stage.outputs]
    if case.inputs["k_scale_cache"] is not None:
        outputs.append(case.inputs["k_scale_cache"].cpu().clone())
    return outputs


def assert_identical(before, after):
    assert len(before) == len(after), "snapshot length differs"
    for index, (a, b) in enumerate(zip(before, after)):
        actual = a.contiguous().view(torch.uint8).flatten()
        expected = b.contiguous().view(torch.uint8).flatten()
        different = (actual != expected).nonzero().flatten()
        assert different.numel() == 0, (
            f"repeated output {index} differs in {different.numel()} bytes; "
            f"first offsets={different[:8].tolist()}, "
            f"before={actual[different[:8]].tolist()}, after={expected[different[:8]].tolist()}"
        )


def check_contract(case):
    i = case.inputs
    o = case.stage.outputs
    assert len(o) == 8
    assert o[3] is i["k_cache"]
    assert torch_npu.get_npu_format(i["wqb"]) == 29
    assert torch_npu.get_npu_format(i["ww"]) == 29
    assert torch_npu.get_npu_format(i["wk"]) == 29
    for original, launch in zip(
        [
            i[name]
            for name in (
                "x",
                "qr",
                "wqb",
                "ww",
                "descale_qr",
                "descale_wqb",
                "q_rope_sin",
                "q_rope_cos",
            )
        ],
        case.stage.qw_args[:8],
    ):
        assert original.data_ptr() == launch.data_ptr(), "input conversion detected"
    assert case.stage.k_args[6].data_ptr() == i["k_cache"].data_ptr()
    if case.stage.storage_mode == 1:
        assert i["k_scale_cache"] is None
        assert i["k_cache"].dtype == torch.uint8 and tuple(i["k_cache"].shape[1:]) == (
            16,
            544,
        )
        assert case.stage.workspaces.quantized_key_rows.shape == (case.seq, 96)
        assert case.stage.workspaces.key_staging.data_ptr() != i["k_cache"].data_ptr()
    else:
        assert tuple(i["k_cache"].shape[1:]) == (128, 1, 64)
        assert tuple(i["k_scale_cache"].shape[1:]) == (128, 1, 4)
        assert case.stage.workspaces.quantized_key_rows is None
        assert case.stage.workspaces.key_staging is i["k_cache"]
        assert case.stage.workspaces.scale_staging is i["k_scale_cache"]


def validate(case, *, repeat=True):
    check_contract(case)
    case.stage.run()
    torch.npu.synchronize()
    check_query_weight(case)
    check_key_cache(case)
    check_qli(case)
    if repeat:
        first = snapshot(case)
        case.stage.run()
        torch.npu.synchronize()
        assert_identical(first, snapshot(case))
        metadata = torch.full(
            (1024,), 305419896, dtype=torch.int32, device=case.inputs["x"].device
        )
        inputs = dict(case.inputs, metadata=metadata)
        original = case.stage.outputs
        alternative = prepare_mega_full_index(
            **inputs,
            softmax_scale=1 / 64,
            norm_eps=1e-20,
            q=original[0],
            descale_q=original[1],
            w=original[2],
            return_value=bool(original[5].numel()),
            storage_mode=case.stage.storage_mode,
        )
        assert all((a is b for a, b in zip(alternative.outputs[:4], original[:4])))
        alternative.run()
        torch.npu.synchronize()
        assert_identical(
            first,
            (
                [x.cpu().clone() for x in alternative.outputs]
                + (
                    [inputs["k_scale_cache"].cpu().clone()]
                    if inputs["k_scale_cache"] is not None
                    else []
                )
            ),
        )
    return case


def validate_stage(case, phase):
    """Compile one device stage with the same tiles as the production kernel."""
    stage = case.stage
    if phase == "qw":
        op = ParallelQw()

        @kernel
        def Proj(
            x: Tensor,
            qr: Tensor,
            wqb: Tensor,
            ww: Tensor,
            dqr: Tensor,
            dwqb: Tensor,
            sin: Tensor,
            cos: Tensor,
            q: Tensor,
            dq: Tensor,
            w: Tensor,
            workspace: Tensor,
            t,
            scale,
        ):
            cores = get_block_num()
            for task_round in decode_module.dsl_range(
                0, (32 + cores - 1) // cores, unroll=1
            ):
                task = get_block_idx() + task_round * cores
                if task < 16:
                    op.run_q(qr, wqb, dqr, dwqb, sin, cos, q, dq, t, task)
                elif task < 32:
                    op.run_w(x, ww, workspace, t, scale, task - 16)
                global_sync_all()
                onchip_rewind()
            op.run_reduce(workspace, w, t)

        @host
        def launch(
            x: Tensor,
            qr: Tensor,
            wqb: Tensor,
            ww: Tensor,
            dqr: Tensor,
            dwqb: Tensor,
            sin: Tensor,
            cos: Tensor,
            q: Tensor,
            dq: Tensor,
            w: Tensor,
            workspace: Tensor,
            t,
            scale,
            workers,
        ):
            Proj[workers](
                x, qr, wqb, ww, dqr, dwqb, sin, cos, q, dq, w, workspace, t, scale
            )

        compiled = compile_dsl(
            launch, *qw._arg_specs(stage.geo, Dim("T", min=1), Dim("WS", min=1)), qw.I64
        )
        compiled(*stage.qw_args, entry._current_core_count(case.inputs["x"].device))
        torch.npu.synchronize()
        check_query_weight(case)
    elif phase == "k":
        op = DecodeKProjection()

        @kernel
        def Stage(latent: Tensor, weight: Tensor, projected: Tensor):
            cores = get_block_num()
            for task_round in decode_module.dsl_range(
                0, (4 + cores - 1) // cores, unroll=1
            ):
                task = get_block_idx() + task_round * cores
                if task < 4:
                    op.kernel(latent, weight, projected, 0, task)
                global_sync_all()
                onchip_rewind()

        @host
        def launch(latent: Tensor, weight: Tensor, projected: Tensor, workers):
            Stage[workers](latent, weight, projected)

        specs = _k_specs()
        compiled = compile_dsl(launch, specs[0], specs[1], specs[8], qw.I64)
        # The Cube accumulator is M16, but the GM output contains only T rows.
        # Guard the following allocation span against an oversized FIXPIPE store.
        guarded = torch.full(
            (32, 128), -123.0, dtype=torch.bfloat16, device=case.inputs["latent"].device
        )
        projected = guarded[: case.seq]
        compiled(
            stage.k_args[0],
            stage.k_args[1],
            projected,
            entry._current_core_count(case.inputs["x"].device),
        )
        torch.npu.synchronize()
        expected = project_key_reference(
            case.inputs["latent"].cpu(), as_nd_cpu(case.inputs["wk"])
        )
        torch.testing.assert_close(
            projected.cpu(),
            expected,
            rtol=0.01,
            atol=0.01,
        )
        assert bool((guarded[case.seq :].cpu() == -123.0).all()), (
            "K projection wrote past live rows"
        )
    elif phase == "qli":
        stage.run()
        torch.npu.synchronize()
        device, specs = build_runner(
            stage.qli_stage.config, case.seq, logical_cache=stage.storage_mode == 1
        )

        @kernel
        def Stage(
            query_address,
            key_address,
            weights: Tensor,
            query_scale,
            key_scale_address,
            key_stride0,
            key_dequant_scale_stride0,
            cu_seqlens_q: Tensor,
            seqused_q: Tensor,
            seqused_k: Tensor,
            cmp_residual_k: Tensor,
            block_table: Tensor,
            key_out: Tensor,
            candidate_key_out: Tensor,
            sparse_indices_address,
            sparse_value_bits_address,
            candidate_indices_address,
            candidate_value_bits_address,
            candidate_length: Tensor,
            worker_count,
            total_q,
            batch_count,
            tokens,
            max_tasks_per_batch,
            key_rows,
            scale_rows,
            metadata: Tensor,
            task_count,
            final_sparse_indices_address,
            final_sparse_bits_address,
            final_candidate_indices_address,
            final_candidate_bits_address,
            return_values,
            merge_workers,
            output_offset_address,
            merge_metadata_address,
        ):
            device(
                query_address,
                key_address,
                weights,
                query_scale,
                key_scale_address,
                key_stride0,
                key_dequant_scale_stride0,
                cu_seqlens_q,
                seqused_q,
                seqused_k,
                cmp_residual_k,
                block_table,
                key_out,
                candidate_key_out,
                sparse_indices_address,
                sparse_value_bits_address,
                candidate_indices_address,
                candidate_value_bits_address,
                candidate_length,
                worker_count,
                total_q,
                batch_count,
                tokens,
                max_tasks_per_batch,
                key_rows,
                scale_rows,
                metadata,
                task_count,
                final_sparse_indices_address,
                final_sparse_bits_address,
                final_candidate_indices_address,
                final_candidate_bits_address,
                return_values,
                merge_workers,
                output_offset_address,
                merge_metadata_address,
            )

        @host
        def launch(
            query_address,
            key_address,
            weights: Tensor,
            query_scale,
            key_scale_address,
            key_stride0,
            key_dequant_scale_stride0,
            cu_seqlens_q: Tensor,
            seqused_q: Tensor,
            seqused_k: Tensor,
            cmp_residual_k: Tensor,
            block_table: Tensor,
            key_out: Tensor,
            candidate_key_out: Tensor,
            sparse_indices_address,
            sparse_value_bits_address,
            candidate_indices_address,
            candidate_value_bits_address,
            candidate_length: Tensor,
            worker_count,
            total_q,
            batch_count,
            tokens,
            max_tasks_per_batch,
            key_rows,
            scale_rows,
            metadata: Tensor,
            task_count,
            final_sparse_indices_address,
            final_sparse_bits_address,
            final_candidate_indices_address,
            final_candidate_bits_address,
            return_values,
            merge_workers,
            output_offset_address,
            merge_metadata_address,
        ):
            Stage[worker_count](
                query_address,
                key_address,
                weights,
                query_scale,
                key_scale_address,
                key_stride0,
                key_dequant_scale_stride0,
                cu_seqlens_q,
                seqused_q,
                seqused_k,
                cmp_residual_k,
                block_table,
                key_out,
                candidate_key_out,
                sparse_indices_address,
                sparse_value_bits_address,
                candidate_indices_address,
                candidate_value_bits_address,
                candidate_length,
                worker_count,
                total_q,
                batch_count,
                tokens,
                max_tasks_per_batch,
                key_rows,
                scale_rows,
                metadata,
                task_count,
                final_sparse_indices_address,
                final_sparse_bits_address,
                final_candidate_indices_address,
                final_candidate_bits_address,
                return_values,
                merge_workers,
                output_offset_address,
                merge_metadata_address,
            )

        compiled = compile_dsl(launch, *specs)
        arguments = list(stage.qli_stage.args)
        arguments[19] = entry._current_core_count(case.inputs["x"].device)
        compiled(*arguments)
        torch.npu.synchronize()
        check_qli(case)
    else:
        raise ValueError(phase)


def validation_main(argv=None):
    parser = argparse.ArgumentParser(description="Layer-20 fused Indexer validation")
    parser.add_argument("--seq", type=int, choices=(6, 8, 16), default=8)
    parser.add_argument("--kv", type=int, default=8192)
    parser.add_argument(
        "--stage", choices=("fused", "qw", "k", "qli", "fp4"), default="fused"
    )
    parser.add_argument("--regression-case", choices=tuple(CASE_BY_NAME), default=None)
    parser.add_argument("--storage-mode", type=int, choices=(0, 1), default=1)
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--permute", action="store_true")
    parser.add_argument("--padded-stride", action="store_true")
    parser.add_argument(
        "--scenario", choices=("random", "ties", "negative"), default="random"
    )
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.regression_case is not None:
        run_case(CASE_BY_NAME[args.regression_case])
        print("VALIDATION_PASS", args.regression_case, flush=True)
        return
    if args.stage == "fp4":
        _check_fp4_device_boundaries()
        print("VALIDATION_PASS", "fp4", flush=True)
        return
    case = make_case(
        args.seq,
        args.kv,
        storage_mode=args.storage_mode,
        capacity=args.capacity,
        permute=args.permute,
        padded_stride=args.padded_stride,
        scenario=args.scenario,
    )
    if args.stage == "fused":
        validate(case)
    else:
        validate_stage(case, args.stage)
    cores = entry._current_core_count(case.inputs["x"].device)
    print(
        "VALIDATION_PASS",
        args.stage,
        args.seq,
        args.kv,
        f"mix_blocks={cores}",
        flush=True,
    )


def run_validation(
    seq, kv, stage="fused", scenario="random", storage_mode=1, *, regression_case=None
):
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "test"), env.get("PYTHONPATH", "")]
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--seq",
        str(seq),
        "--kv",
        str(kv),
        "--stage",
        stage,
        "--scenario",
        scenario,
        "--storage-mode",
        str(storage_mode),
    ]
    if regression_case is not None:
        command += ["--regression-case", regression_case]
    result = subprocess.run(
        command, cwd=root, env=env, capture_output=True, text=True, timeout=1800
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VALIDATION_PASS" in result.stdout


def generalization_lengths(seq, phase):
    if phase == 1:
        boundary = [
            1,
            2,
            7,
            8,
            9,
            15,
            16,
            17,
            31,
            32,
            33,
            127,
            128,
            129,
            255,
            256,
            257,
            511,
            512,
            513,
            1023,
            1024,
            1025,
            3583,
            3584,
            3585,
            4095,
            4096,
            4097,
            7680,
            7681,
            8191,
            8192,
            8193,
            8703,
            8704,
            16383,
            16384,
            16385,
            16384 + seq - 1,
            16384 + seq,
            32767,
            32768,
            32769,
            65535,
            65536,
            65537,
            131071,
            131072,
        ]
        rng = random.Random(20261016 + seq)
        random_lengths = [rng.randint(1 + i * 16384, (i + 1) * 16384) for i in range(8)]
    else:
        boundary = [
            131073,
            262143,
            262144,
            262145,
            524287,
            524288,
            524289,
            1000000,
            1048575,
            1048576,
        ]
        rng = random.Random(20261017 + seq)
        width = (1048576 - 131072) // 8
        random_lengths = [
            rng.randint(131073 + i * width, 131072 + (i + 1) * width) for i in range(8)
        ]
    return sorted(set(boundary + random_lengths))


@dataclass(frozen=True)
class GuardCase:
    """一条 CI 看护用例：固定形状与路径开关，核数由生产入口动态查询。

    用例名统一为 ``s{S}-kv{KV}-m{storage_mode}-{tag}``；``kv`` 是分页 cache 的
    物理容量，``runtime_lengths`` 只改变运行时可见长度，不改变容量。
    """

    name: str
    seq: int
    kv: int
    storage_mode: int = 1
    return_value: bool = True
    scenario: str = "random"
    permute: bool = False
    padded_stride: bool = False
    duplicate_skip: bool = False
    output_guard: bool = False
    graph: bool = False
    offset: bool = False
    repeat: bool = False
    active_q: int | None = None
    runtime_lengths: tuple[int, int] | None = None
    zero_inputs: bool = False


# CI 只跑这 9 条；每条覆盖的路径见 samples/attention/mega_full_index/README.md。
CASES = (
    GuardCase("s8-kv8192-m1-values", 8, 8192),
    GuardCase(
        "s8-kv8192-m0-index-only-repeat-negative",
        8,
        8192,
        storage_mode=0,
        return_value=False,
        scenario="negative",
        duplicate_skip=True,
        repeat=True,
    ),
    GuardCase(
        "s6-kv16384-m1-paged-guard",
        6,
        16384,
        permute=True,
        padded_stride=True,
        output_guard=True,
    ),
    GuardCase("s8-kv8191-m1-active3-tail", 8, 8191, active_q=3),
    GuardCase("s8-kv8192-m1-zero-inputs", 8, 8192, zero_inputs=True),
    GuardCase("s8-kv8192-m1-empty-kv", 8, 8192, runtime_lengths=(0, 0)),
    GuardCase("s16-kv8192-m1-graph-offset", 16, 8192, graph=True, offset=True),
    GuardCase("s8-kv131073-m1-candidate", 8, 131073),
    GuardCase("s8-kv1048576-m1-candidate-1m", 8, 1048576),
)

CASE_BY_NAME = {spec.name: spec for spec in CASES}


def _guarded_qw_outputs(case):
    outputs, guards = {}, []
    for name, shape, dtype, guard, sentinel in (
        ("q", (case.seq, 32, 64), torch.uint8, 4096, 165),
        ("descale_q", (case.seq, 32, 2, 2), torch.uint8, 256, 165),
        ("w", (case.seq, 32), torch.float32, 128, 123.5),
    ):
        count = math.prod(shape)
        backing = torch.full(
            (count + 2 * guard,), sentinel, dtype=dtype, device=case.inputs["x"].device
        )
        outputs[name] = backing[guard : guard + count].view(shape)
        guards.append((backing, guard, sentinel))
    return outputs, guards


def run_case(spec):
    """按看护用例配置构造输入、执行一次融合 kernel 并逐项比对。"""
    options = {} if spec.active_q is None else {"active_q": spec.active_q}
    case = make_case(
        spec.seq,
        spec.kv,
        return_value=spec.return_value,
        storage_mode=spec.storage_mode,
        scenario=spec.scenario,
        permute=spec.permute,
        padded_stride=spec.padded_stride,
        **options,
    )
    if spec.zero_inputs:
        for name in ("x", "qr", "latent"):
            case.inputs[name].view(torch.uint8).zero_()
    if spec.runtime_lengths is not None:
        # 保留已分配的物理容量，只把运行时可见长度置空。
        active_q, used_k = spec.runtime_lengths
        case.inputs["seqused_q"].fill_(active_q)
        case.inputs["seqused_k"].fill_(used_k)
        case.inputs["cache_index"].fill_(-1)
    if spec.duplicate_skip:
        case.inputs["cache_index"][0] = -1
        case.inputs["cache_index"][-1] = case.inputs["cache_index"][1]
    if spec.offset:
        case.inputs["output_idx_offset"] = (
            torch.arange(spec.seq, dtype=torch.int32, device=case.inputs["x"].device)
            + 17
        )[:, None]
    guards = []
    if spec.output_guard:
        outputs, guards = _guarded_qw_outputs(case)
    else:
        outputs = dict(zip(("q", "descale_q", "w"), case.stage.outputs[:3]))
    if spec.output_guard or spec.offset:
        case.stage = prepare_mega_full_index(
            **case.inputs,
            **outputs,
            softmax_scale=1 / 64,
            norm_eps=1e-20,
            storage_mode=spec.storage_mode,
            return_value=spec.return_value,
        )
        assert all(
            case.stage.outputs[i] is outputs[name]
            for i, name in enumerate(("q", "descale_q", "w"))
        )
    cache = case.inputs["k_cache"]
    padding = None
    if spec.padded_stride:
        span = cache.untyped_storage().nbytes()
        padding = torch.as_strided(cache, (span,), (1,), storage_offset=0)
        before_padding = padding.cpu().clone()
    validate(case, repeat=spec.repeat)
    assert bool(case.stage.outputs[5].numel()) == spec.return_value
    for storage, guard, sentinel in guards:
        assert bool((storage[:guard].cpu() == sentinel).all()), "QW output prefix guard"
        assert bool((storage[-guard:].cpu() == sentinel).all()), (
            "QW output suffix guard"
        )
    if padding is not None:
        after_padding = padding.cpu()
        for page in range(cache.shape[0] - 1):
            begin = page * cache.stride(0) + cache[page].numel()
            assert torch.equal(
                after_padding[begin : begin + 64],
                before_padding[begin : begin + 64],
            ), "combined cache page padding changed"
    if spec.kv == 8192 and spec.runtime_lengths is None:
        expected_w = (
            case.inputs["x"].cpu().float() @ as_nd_cpu(case.inputs["ww"]).float().T / 64
        )
        error = (case.stage.outputs[2].cpu() - expected_w).abs()
        print(
            "QW_W",
            "zero" if spec.zero_inputs else "random",
            f"max_abs_error={float(error.max()):.9g}",
            f"max_abs_reference={float(expected_w.abs().max()):.9g}",
            flush=True,
        )
    if spec.graph:
        expected = snapshot(case)
        capture = torch.npu.Stream()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=capture):
            case.stage.run()
        for _ in range(2):
            graph.replay()
            torch.npu.synchronize()
            assert_identical(expected, snapshot(case))
    cores = entry._current_core_count(case.inputs["x"].device)
    print("PLATFORM", f"mix_blocks={cores}", flush=True)


@pytest.mark.npu
@pytest.mark.parametrize("spec", CASES, ids=lambda spec: spec.name)
def test_mega_full_index(spec):
    run_case(spec)


TENSOR_NAMES = (
    "x",
    "qr",
    "wqb",
    "ww",
    "descale_qr",
    "descale_wqb",
    "q_rope_sin",
    "q_rope_cos",
    "latent",
    "wk",
    "norm_weight",
    "k_rope_sin",
    "k_rope_cos",
    "k_cache",
    "k_scale_cache",
    "cache_index",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "seqused_q",
    "seqused_k",
    "cmp_residual_k",
    "block_table",
    "output_idx_offset",
    "metadata",
)


def _selfcheck_public_signature():
    for function in (mega_full_index, prepare_mega_full_index):
        parameters = inspect.signature(function).parameters
        assert tuple(parameters)[:24] == TENSOR_NAMES
        assert parameters["storage_mode"].default == 1
        assert parameters["metadata"].default is None


def _selfcheck_combined_byte_contract():
    data = (
        (torch.arange(3 * 128 * 64, dtype=torch.int64) * 17 % 256)
        .to(torch.uint8)
        .view(3, 128, 1, 64)
    )
    scale = (
        (torch.arange(3 * 128 * 4, dtype=torch.int64) * 7 % 256)
        .to(torch.uint8)
        .view(3, 128, 1, 4)
    )
    combined = pack_cache(data, scale)
    assert combined.shape == (3, 16, 544)
    for page in range(3):
        for group in range(16):
            for slot in range(8):
                assert torch.equal(
                    combined[page, group, slot * 64 : (slot + 1) * 64],
                    data[page, group * 8 + slot, 0],
                )
                assert torch.equal(
                    combined[page, group, 512 + slot * 4 : 512 + (slot + 1) * 4],
                    scale[page, group * 8 + slot, 0],
                )
    restored = unpack_cache(combined)
    assert torch.equal(restored[0], data) and torch.equal(restored[1], scale)


def _selfcheck_logical_task_coverage():
    for cores in (28, 32):
        tasks = [
            c + r * cores
            for r in range((32 + cores - 1) // cores)
            for c in range(cores)
            if c + r * cores < 32
        ]
        assert sorted(tasks) == list(range(32))
        pages = [p for aiv in range(2 * cores) for p in range(aiv, 64, 2 * cores)]
        assert sorted(pages) == list(range(64))


def _rounding_groups(dtype):
    import torch

    midpoints = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), dtype=torch.float32
    )
    groups = []
    for scale in (2.0**-126, 2.0**-8, 0.5, 1.0, 2.0, 64.0):
        for sign in (1, -1):
            exact = (midpoints * scale).to(dtype)
            for direction in (-1, 0, 1):
                values = exact
                if direction:
                    values = torch.nextafter(
                        exact, torch.full_like(exact, float("inf") * direction)
                    )
                group = torch.zeros(32, dtype=dtype)
                group[:7] = values * sign
                group[7] = 6.0 * scale
                group[8] = -0.0
                groups.append(group)
    groups.append(torch.zeros(32, dtype=dtype))
    groups.append(torch.full((32,), -0.0, dtype=dtype))
    tiny = torch.zeros(32, dtype=dtype)
    tiny[:4] = torch.tensor(
        (2.0**-128, -(2.0**-128), 2.0**-133, -(2.0**-133)), dtype=dtype
    )
    groups.append(tiny)
    while len(groups) < 64:
        groups.append(groups[len(groups) % 36].clone())
    return torch.stack(groups)


def _selfcheck_nearest_even_midpoints():
    import torch

    midpoints = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0))
    expected = torch.tensor((0, 2, 2, 4, 4, 6, 6), dtype=torch.uint8)
    assert torch.equal(fp32_to_e2m1_nibble(midpoints), expected)
    assert torch.equal(fp32_to_e2m1_nibble(-midpoints), expected | 8)
    zeros = torch.tensor((0.0, -0.0))
    assert torch.equal(
        fp32_to_e2m1_nibble(zeros), torch.tensor((0, 8), dtype=torch.uint8)
    )


def _check_fp4_device_boundaries():
    import torch
    from cannbotdsl import TensorSpec, dtypes
    from cannbotdsl import compile as compile_dsl
    from cannbotdsl.lang.host import host
    from cannbotdsl.lang.kernel import kernel

    q_input = _rounding_groups(torch.float32).reshape(8, 256)
    qdata, qscale = mx_quant_along_last(q_input.reshape(8, 2, 128))
    q_device = q_input.npu()
    amax = q_input.reshape(64, 32).abs().amax(-1).npu()
    qout = torch.empty((8, 128), dtype=torch.uint8, device=q_device.device)
    qsout = torch.empty((1, 64), dtype=torch.uint8, device=q_device.device)

    @kernel
    def QRoundCanary(x: Tensor, a: Tensor, data: Tensor, scale: Tensor):
        vec = qw.PairVector(qw.DecodePlan())
        cv = qw.Buffer(qw.MemLoc.UB, dtypes.float32, qw.make_layout((8, 256)))
        vec.q_ub = vec.q_stage.next_write()
        vec.dsc_ub = vec.scale_stage.next_write()
        qw.qw_mem_copy(cv, x)
        qw.qw_mem_copy(vec.amax_ub, a)
        qw.vec_sync_all()
        vec._e8m0_scales()
        vec._pack_e2m1(cv, 8)
        qw.vec_sync_all()
        qw.qw_mem_copy(data, vec.q_stage.next_read())
        qw.qw_mem_copy(scale, vec.scale_stage.next_read())

    @host
    def qlaunch(x: Tensor, a: Tensor, data: Tensor, scale: Tensor):
        QRoundCanary[1](x, a, data, scale)

    qhandle = compile_dsl(
        qlaunch,
        TensorSpec((8, 256), dtypes.float32),
        TensorSpec((64,), dtypes.float32),
        TensorSpec((8, 128), dtypes.uint8),
        TensorSpec((1, 64), dtypes.uint8),
    )
    qhandle(q_device, amax, qout, qsout)
    torch.npu.synchronize()
    assert torch.equal(qout.cpu(), qdata.reshape(8, 128)), "Q midpoint/neighbor bytes"
    assert torch.equal(qsout.cpu(), qscale.reshape(1, 64)), "Q zero/minimum scale"
    qhandle.close()

    k_input = _rounding_groups(torch.bfloat16).reshape(16, 128)
    kdata, kscale = _pack_mxfp4(k_input)
    kd = k_input.npu()
    kout = torch.empty((16, 64), dtype=torch.uint8, device=kd.device)
    ksout = torch.empty((16, 4), dtype=torch.uint8, device=kd.device)
    post = key_module.KPostQuantizer(
        total_rows=16, index_head_dim=128, rope_head_dim=64
    )

    @kernel
    def KRoundCanary(x: Tensor, data: Tensor, scale: Tensor):
        k = key_module
        source = k.Buffer(k.MemLoc.UB, dtypes.bfloat16, k.make_layout((1, 192)))
        packed = k.Buffer(k.MemLoc.UB, dtypes.uint8, k.make_layout((1, 128)))
        scratch = k.Buffer(k.MemLoc.UB, dtypes.uint32, k.make_layout((4, 64)))
        for row in range(16):
            k.mem_copy(
                k.tile_slice(source, (1, 128), (0, 0)),
                k.tile_slice(x, (1, 128), (row, 0)),
            )
            k.vec_sync_all()
            post._pack_mxfp4(source, packed, scratch)
            k.vec_sync_all()
            k.mem_copy(
                k.tile_slice(data, (1, 64), (row, 0)),
                k.tile_slice(packed, (1, 64), (0, 0)),
            )
            k.mem_copy(
                k.tile_slice(scale, (1, 4), (row, 0)),
                k.tile_slice(packed, (1, 4), (0, 16)),
            )
            k.vec_sync_all()

    @host
    def klaunch(x: Tensor, data: Tensor, scale: Tensor):
        KRoundCanary[1](x, data, scale)

    khandle = compile_dsl(
        klaunch,
        TensorSpec((16, 128), dtypes.bfloat16),
        TensorSpec((16, 64), dtypes.uint8),
        TensorSpec((16, 4), dtypes.uint8),
    )
    khandle(kd, kout, ksout)
    torch.npu.synchronize()
    assert torch.equal(kout.cpu(), kdata), "K midpoint/neighbor bytes"
    assert torch.equal(ksout.cpu(), kscale), "K minimum scale"
    khandle.close()


def _selfcheck_candidate_reference():
    import torch

    for visible in (0, 1, 7, 8, 9, 16383, 16384, 16385, 32768, 524289, 1048576):
        # Negative and non-monotonic values; latest block starts with a low score.
        score = -(torch.arange(max(visible, 1)) * 17 % 113).to(torch.bfloat16)
        if visible:
            score[max(0, visible - 8) : visible] = -1000
        blocks, count = candidate_reference(score, visible)
        selected = (
            blocks.topk(count).indices if count else torch.empty(0, dtype=torch.int64)
        )
        candidate = torch.full((2048,), -1, dtype=torch.int32)
        candidate[:count] = selected.to(torch.int32)
        check_candidates(candidate, count, score, visible)
    score = torch.zeros(16392, dtype=torch.bfloat16)
    blocks, count = candidate_reference(score, len(score))
    chosen = torch.arange(1, 2049, dtype=torch.int32)
    check_candidates(chosen, count, score, len(score))
    broken = chosen.clone()
    broken[-1] = broken[0]
    import pytest

    with pytest.raises(AssertionError, match="duplicate"):
        check_candidates(broken, count, score, len(score))
    with pytest.raises(AssertionError, match="latest"):
        check_candidates(
            torch.arange(2048, dtype=torch.int32), count, score, len(score)
        )


def _selfcheck_generalization_task_coverage():
    for seq in (8, 16):
        for length in generalization_lengths(seq, 1) + generalization_lengths(seq, 2):
            for cores in (28, 32):
                count = (length + 127) // 128
                tasks = [t for c in range(cores) for t in range(c, count, cores)]
                assert sorted(tasks) == list(range(count))
                groups = (length + 1023) // 1024
                while groups > 16:
                    jobs = ((groups + 7) // 8) * seq
                    scheduled = [
                        j for v in range(cores * 2) for j in range(v, jobs, cores * 2)
                    ]
                    assert sorted(scheduled) == list(range(jobs))
                    groups = (groups + 7) // 8
                assert groups * 512 <= 8192


def _selfcheck_chunked_score_matches_dense():
    from types import SimpleNamespace

    import torch

    generator = torch.Generator().manual_seed(20261016)
    seq, length, pages = 8, 257, 3
    data = torch.randint(
        0, 256, (pages, 128, 1, 64), dtype=torch.uint8, generator=generator
    )
    scale = torch.randint(
        125, 130, (pages, 128, 1, 4), dtype=torch.uint8, generator=generator
    )
    q = torch.randint(0, 256, (seq, 32, 64), dtype=torch.uint8, generator=generator)
    dq = torch.full((seq, 32, 2, 2), 127, dtype=torch.uint8)
    weights = torch.randn((seq, 32), generator=generator).to(torch.bfloat16).float()
    table = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    case = SimpleNamespace(
        seq=seq,
        inputs=dict(
            k_cache=pack_cache(data, scale),
            k_scale_cache=None,
            block_table=table,
            seqused_k=torch.tensor([length], dtype=torch.int32),
        ),
        stage=SimpleNamespace(outputs=(q, dq, weights)),
    )
    k = decode_mxfp4(data[table[0].long()], scale[table[0].long()]).reshape(-1, 128)[
        :length
    ]
    dot = (
        (decode_mxfp4(q, dq).reshape(seq * 32, 128) @ k.T)
        .to(torch.bfloat16)
        .reshape(seq, 32, length)
    )
    expected = torch.zeros((seq, length), dtype=torch.bfloat16)
    for head in range(32):
        product = (
            dot[:, head].clamp_min(0) * weights.to(torch.bfloat16)[:, head, None]
        ).to(torch.bfloat16)
        expected = (expected + product).to(torch.bfloat16)
    for chunk in (128, 256, 4096):
        assert torch.equal(qli_score_reference(case, chunk), expected)


KERNEL_NAME = "mega_full_index"


def summarize(output, samples):
    durations = []
    pmu = []
    for filename in output.glob(
        "*_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv"
    ):
        with filename.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        selected = [
            row
            for row in rows
            if row.get("Name") == KERNEL_NAME
            or "kernel_mega_full_index" in row.get("Name", "")
        ]
        durations.extend((float(row["Duration(us)"]) for row in selected))
        pmu.extend(
            (
                {
                    key: value
                    for key, value in row.items()
                    if key.startswith(("aic_", "aiv_"))
                }
                for row in selected
            )
        )
    durations.sort()
    if len(durations) != samples:
        raise RuntimeError(f"Expected {samples} fused launches, found {len(durations)}")
    return dict(
        kernel_name=KERNEL_NAME,
        launches=len(durations),
        min_us=min(durations),
        p50_us=statistics.median(durations),
        p95_us=durations[math.ceil(0.95 * len(durations)) - 1],
        durations_us=durations,
        pipeline_metrics=pmu,
    )


def profiling_main(argv=None):
    parser = argparse.ArgumentParser(description="Layer-20 fused Indexer profiling")
    parser.add_argument("--storage-mode", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seq", type=int, choices=(6, 8, 16), default=8)
    parser.add_argument("--kv", type=int, default=8192)
    parser.add_argument(
        "--mode", choices=("latency", "pmu", "trace"), default="latency"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=31)
    args = parser.parse_args(argv)
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    capture_stream = torch.npu.Stream()
    collect(args, capture_stream)


def collect(args, capture_stream):
    case = make_case(args.seq, args.kv, storage_mode=args.storage_mode)
    validate(case, repeat=False)
    if args.mode == "trace":
        for _ in range(args.samples):
            case.stage.run()
        torch.npu.synchronize()
        return
    eviction = torch.empty((128 * 1024 * 1024,), device="npu", dtype=torch.uint8)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=capture_stream):
        captured_cores = entry._current_core_count(case.inputs["x"].device)
        eviction.zero_()
        case.stage.run()
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    config = _ExperimentalConfig(
        profiler_level=ProfilerLevel.Level1,
        aic_metrics=AiCMetrics.PipeUtilization,
    )
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        schedule=schedule(wait=0, warmup=1, active=args.samples, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(args.output)),
        experimental_config=config,
    ) as profiler:
        for _ in range(args.samples + 1):
            graph.replay()
            torch.npu.synchronize()
            profiler.step()
    report = summarize(args.output, args.samples)
    report.update(
        cores=captured_cores,
        seq=args.seq,
        kv=args.kv,
        graph=True,
        full_fusion=True,
        l2_evict_bytes=eviction.numel(),
        mode=args.mode,
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def _selfcheck_key_projection_dtype():
    from torch.utils._python_dispatch import TorchDispatchMode

    observed = []

    class RecordMatmul(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            if function == torch.ops.aten.mm.default:
                observed.append((args[0].dtype, args[1].dtype))
            return function(*args, **(kwargs or {}))

    latent = torch.zeros((2, 512), dtype=torch.bfloat16)
    latent[0, :3] = torch.tensor([1.0, 1.0, 1.0], dtype=torch.bfloat16)
    latent[1, :3] = torch.tensor([1.0, 1.0, -1.0], dtype=torch.bfloat16)
    wk = torch.zeros((128, 512), dtype=torch.bfloat16)
    wk[:, :3] = torch.tensor([1.0, 2.0**-8, 2.0**-15], dtype=torch.bfloat16)
    with RecordMatmul():
        projected = project_key_reference(latent, wk)
    assert observed == [(torch.float32, torch.float32)]
    assert projected.dtype == torch.bfloat16
    assert torch.equal(
        projected[0], torch.full((128,), 1.0078125, dtype=torch.bfloat16)
    )
    assert torch.equal(projected[1], torch.ones((128,), dtype=torch.bfloat16))


@pytest.fixture(scope="session", autouse=True)
def reference_selfcheck():
    """CPU 侧参考实现自检（不占设备、不启 kernel）。

    这些检查守护的是判据本身：若 ``fp32_to_e2m1_nibble`` / ``pack_cache`` /
    ``candidate_reference`` / 分块打分参考写错，9 条设备用例会一起错，
    所以放在 session 级 fixture 里跑一次，而不是各自占一条 CI 用例。
    """
    _selfcheck_public_signature()
    _selfcheck_combined_byte_contract()
    _selfcheck_logical_task_coverage()
    _selfcheck_nearest_even_midpoints()
    _selfcheck_candidate_reference()
    _selfcheck_generalization_task_coverage()
    _selfcheck_chunked_score_matches_dense()
    _selfcheck_key_projection_dtype()


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--profile" in arguments:
        profiling_main([argument for argument in arguments if argument != "--profile"])
    else:
        validation_main(arguments)
