# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compile and runtime smoke test for MegaReindex."""

import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import cannbotdsl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _samples_path import load_sample

_sample = load_sample("attention/mega_reindex/mega_reindex.py")
FrontWorkspace = _sample.FrontWorkspace
MegaReindexTailWorkspace = _sample.MegaReindexTailWorkspace
prepare_mega_reindex = _sample.prepare_mega_reindex
_front = load_sample("attention/mega_reindex/reindex_front.py")
_formats = load_sample("attention/mega_reindex/reindex_front.py")
prepare_weight_nz = _front.prepare_weight_nz
prepare_wqa_nz = _formats.prepare_wqa_nz
prepare_wqb_nz = _formats.prepare_wqb_nz

B, S, KV_S, TOPK, SEED = 1, 8, 8192, 512, 2026
T = B * S
torch.npu.set_device(0)
torch.manual_seed(SEED)

cpu = {
    "x": torch.randn(T, 5120).to(torch.bfloat16),
    "wa": (torch.randn(1280, 5120) * 0.02).to(torch.float8_e4m3fn),
    "sa": torch.full((1280, 80, 2), 127, dtype=torch.uint8),
    "wb": (torch.randn(4096, 1280) * 0.02).to(torch.float8_e4m3fn),
    "sb": torch.full((4096, 20, 2), 127, dtype=torch.uint8),
    "ww": (torch.randn(32, 5120) * 0.02).to(torch.bfloat16),
    "g": torch.ones(1280, dtype=torch.bfloat16),
    "rope": torch.stack((torch.ones(T, 32), torch.zeros(T, 32)), dim=-1),
}
packed_fixture = (
    torch.randint(0, 256, (KV_S, 1, 64), dtype=torch.uint8),
    torch.full((KV_S, 1, 2, 2), 127, dtype=torch.uint8),
)
p = {name: value.npu() for name, value in cpu.items()}
wqa, descale_wqa = prepare_wqa_nz(p["wa"], p["sa"])
wqb, descale_wqb = prepare_wqb_nz(p["wb"], p["sb"])
ww = prepare_weight_nz(p["ww"])
key, descale_key = packed_fixture
key, descale_key = key.npu(), descale_key.npu()

num_blocks = (KV_S + 7) // 8
candidate_block_indices = torch.full((T, 1, 2048), -1, dtype=torch.int32)
candidate_block_indices[:, 0, :num_blocks] = torch.arange(num_blocks, dtype=torch.int32)
candidate_block_indices = candidate_block_indices.npu()
candidate_block_length = torch.full((T, 1), num_blocks, dtype=torch.int32, device="npu")
cu_seqlens_q = torch.tensor([0, S], dtype=torch.int32, device="npu")
cu_seqlens_k = torch.tensor([0, KV_S], dtype=torch.int32, device="npu")
seqused_q = torch.tensor([S], dtype=torch.int32, device="npu")
seqused_k = torch.tensor([KV_S], dtype=torch.int32, device="npu")

front_workspace = FrontWorkspace(T, p["x"].device)
tail_workspace = MegaReindexTailWorkspace(T, KV_S, p["x"].device, topk=TOPK)

compile_start = time.perf_counter()
program, args, _, _ = prepare_mega_reindex(
    p["x"],
    wqa,
    descale_wqa,
    ww,
    p["g"],
    wqb,
    descale_wqb,
    p["rope"],
    key,
    descale_key,
    front_workspace=front_workspace,
    tail_workspace=tail_workspace,
    norm_eps=1e-20,
    softmax_scale=1 / 64,
    topk=TOPK,
    candidate_block_indices=candidate_block_indices,
    candidate_block_length=candidate_block_length,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    seqused_q=seqused_q,
    seqused_k=seqused_k,
    candidate_block_size=8,
    mask_mode=3,
    cmp_ratio=1,
)
compile_seconds = time.perf_counter() - compile_start

scrub = torch.ones(200 * 1024 * 1024 // 4, dtype=torch.float32, device="npu")
torch.sum(scrub)
program(*args)
torch.npu.synchronize()
indices = tail_workspace.indices.cpu()
torch.sum(scrub)
program(*args)
torch.npu.synchronize()
assert torch.equal(indices, tail_workspace.indices.cpu())
candidate_block_length.zero_()
torch.sum(scrub)
program(*args)
torch.npu.synchronize()
assert bool((tail_workspace.indices.cpu() == -1).all())
candidate_block_length.fill_(num_blocks)
torch.sum(scrub)
program(*args)
torch.npu.synchronize()
assert torch.equal(indices, tail_workspace.indices.cpu())
valid = indices.ge(0)
row_sorted = bool(torch.all((indices[:, 1:] >= indices[:, :-1]) | (indices[:, 1:] < 0)))
range_ok = bool(torch.all((indices >= -1) & (indices < KV_S)))
result = {
    "cannbotdsl_version": version("cannbot-dsl"),
    "module_version": cannbotdsl.__version__,
    "case": {"B": B, "S": S, "T": T, "kv_s": KV_S, "topk": TOPK},
    "compile_seconds": compile_seconds,
    "output_shape": list(indices.shape),
    "output_dtype": str(indices.dtype),
    "valid_counts": valid.sum(dim=1).tolist(),
    "range_ok": range_ok,
    "ascending_or_padded": row_sorted,
    "first_row_head": indices[0, :16].tolist(),
    "pass": (
        tuple(indices.shape) == (T, TOPK)
        and indices.dtype == torch.int32
        and range_ok
        and row_sorted
    ),
}
print(json.dumps(result, indent=2))
if not result["pass"]:
    raise SystemExit(1)


# Non-identity, pair-dependent RoPE with one-hot WQB gives an exact CPU oracle.
# Each head projects the same first 128 normalized query elements.
identity_wqb = torch.zeros((4096, 1280), dtype=torch.float32)
identity_wqb[torch.arange(4096), torch.arange(4096) % 128] = 1
probe_wqb, probe_scale = prepare_wqb_nz(
    identity_wqb.to(torch.float8_e4m3fn).npu(), p["sb"]
)
rotation = (torch.arange(T)[:, None] + torch.arange(32)[None, :]) % 2
probe_rope_cpu = torch.stack((1 - rotation, rotation), dim=-1).float()
probe_rope = probe_rope_cpu.npu()
probe_program, probe_args, probe_front, probe_tail = prepare_mega_reindex(
    p["x"],
    wqa,
    descale_wqa,
    ww,
    p["g"],
    probe_wqb,
    probe_scale,
    probe_rope,
    key,
    descale_key,
    front_workspace=front_workspace,
    tail_workspace=tail_workspace,
    candidate_block_indices=candidate_block_indices,
    candidate_block_length=candidate_block_length,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    seqused_q=seqused_q,
    seqused_k=seqused_k,
)
probe_program(*probe_args)
torch.npu.synchronize()
qr_scale = torch.pow(2.0, probe_front.rs.cpu().reshape(T, 40).float() - 127)
qr = probe_front.rq.cpu().view(torch.float8_e4m3fn).float()
qr = (qr * qr_scale.repeat_interleave(32, dim=1)).to(torch.bfloat16).float()
expected = qr[:, :128].clone()
even, odd = expected[:, 64::2].clone(), expected[:, 65::2].clone()
cosine, sine = probe_rope_cpu[:, :, 0], probe_rope_cpu[:, :, 1]
expected[:, 64::2] = even * cosine - odd * sine
expected[:, 65::2] = even * sine + odd * cosine
expected = expected.to(torch.bfloat16).float().reshape(T, 4, 32)
maxima = expected.abs().amax(dim=-1).clamp_min(6.0 * 2.0**-126)
exponents = torch.ceil(torch.log2(maxima / 6.0))
normalized = expected / torch.pow(2.0, exponents)[..., None]
absolute = normalized.abs()
codes = torch.zeros_like(absolute, dtype=torch.uint8)
for code, threshold, inclusive in (
    (1, 0.25, False),
    (2, 0.75, True),
    (3, 1.25, False),
    (4, 1.75, True),
    (5, 2.5, False),
    (6, 3.5, True),
    (7, 5.0, False),
):
    take = absolute >= threshold if inclusive else absolute > threshold
    codes = torch.where(take, code, codes)
codes += (normalized < 0).to(torch.uint8) * 8
codes = codes.reshape(T, 128)
packed_expected = codes[:, 0::2] + codes[:, 1::2] * 16
assert torch.equal(probe_tail.q.cpu(), packed_expected[:, None, :].expand(T, 32, 64))
assert torch.equal(
    probe_tail.q_scale.cpu().reshape(T, 32, 4),
    (exponents + 127).to(torch.uint8)[:, None, :].expand(T, 32, 4),
)
print("PASS non-identity RoPE: token/pair cos-sin order and shared head frequencies")


# Cross the metadata UB budget without changing the per-batch numeric fixture.
# One batch is the reference; duplicated batches must produce identical rows.
batch_results = []
metadata_stream = torch.npu.Stream()
for batch_count in (1, 256):
    batch_rows = batch_count * 2
    batch_x = p["x"][:2].repeat(batch_count, 1)
    batch_rope = p["rope"][:2].repeat(batch_count, 1, 1)
    batch_key = key[:512].repeat(batch_count, 1, 1)
    batch_scale = descale_key[:512].repeat(batch_count, 1, 1, 1)
    batch_candidates = torch.full((batch_rows, 1, 2048), -1, dtype=torch.int32)
    batch_candidates[:, 0, :64] = torch.arange(64, dtype=torch.int32)
    torch.npu.synchronize()
    with torch.npu.stream(metadata_stream):
        quota = 8 if batch_count == 256 else 32
        torch.npu.set_stream_limit(
            metadata_stream, cube_num=quota, vector_num=2 * quota
        )
        batch_program, batch_args, _, batch_tail = prepare_mega_reindex(
            batch_x,
            wqa,
            descale_wqa,
            ww,
            p["g"],
            wqb,
            descale_wqb,
            batch_rope,
            batch_key,
            batch_scale,
            topk=64,
            candidate_block_indices=batch_candidates.npu(),
            candidate_block_length=torch.full(
                (batch_rows, 1), 64, dtype=torch.int32, device="npu"
            ),
            cu_seqlens_q=torch.arange(batch_count + 1, dtype=torch.int32, device="npu")
            * 2,
            cu_seqlens_k=torch.arange(batch_count + 1, dtype=torch.int32, device="npu")
            * 512,
            seqused_q=torch.full((batch_count,), 2, dtype=torch.int32, device="npu"),
            seqused_k=torch.full((batch_count,), 512, dtype=torch.int32, device="npu"),
        )
        batch_program(*batch_args)
        torch.npu.synchronize()
    batch_results.append(
        (
            batch_tail.indices.cpu(),
            batch_tail.score.cpu(),
            batch_tail.visible_lengths.cpu(),
            batch_tail.block_mask.cpu(),
        )
    )
for reference, actual in zip(batch_results[0], batch_results[1]):
    repeats = (256,) + (1,) * (reference.ndim - 1)
    assert torch.equal(reference.repeat(*repeats), actual)
print(
    "PASS B=256 metadata fallback: same indices, score, visibility and mask as duplicated B=1"
)


# Native E8M0 parameters and the source patch's byte views encode the same scales.
assert descale_wqa.dtype == torch.float8_e8m0fnu
assert descale_wqb.dtype == torch.float8_e8m0fnu
scale_results = []
candidate_block_length.fill_(num_blocks)
for scale_dtype in (torch.uint8, torch.float8_e8m0fnu):
    scale_program, scale_args, scale_front, scale_tail = prepare_mega_reindex(
        p["x"],
        wqa,
        descale_wqa.view(scale_dtype),
        ww,
        p["g"],
        wqb,
        descale_wqb.view(scale_dtype),
        p["rope"],
        key,
        descale_key.view(scale_dtype),
        front_workspace=front_workspace,
        tail_workspace=tail_workspace,
        candidate_block_indices=candidate_block_indices,
        candidate_block_length=candidate_block_length,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
    )
    for index in (6, 8, 19, 20, 25):
        assert scale_args[index].dtype == torch.float8_e8m0fnu
    scale_program(*scale_args)
    torch.npu.synchronize()
    scale_results.append(
        {
            f"{stage}/{name}": tensor.cpu().contiguous().view(torch.uint8).clone()
            for stage, workspace in (("front", scale_front), ("tail", scale_tail))
            for name, tensor in vars(workspace).items()
            if isinstance(tensor, torch.Tensor) and tensor.numel()
        }
    )
assert scale_results[0].keys() == scale_results[1].keys()
for name in scale_results[0]:
    assert torch.equal(scale_results[0][name], scale_results[1][name]), name
print(
    "PASS native E8M0 and legacy UINT8 scale arguments: all workspace bytes identical"
)


# Reuse a prepared program after quota changes, including Graph capture.
program(*args)
torch.npu.synchronize()


def workspace_bytes():
    return {
        f"{stage}/{name}": tensor.cpu().contiguous().view(torch.uint8).clone()
        for stage, workspace in (("front", front_workspace), ("tail", tail_workspace))
        for name, tensor in vars(workspace).items()
        if isinstance(tensor, torch.Tensor) and tensor.numel()
    }


quota_reference = workspace_bytes()
quota_stream = torch.npu.Stream()
with torch.npu.stream(quota_stream):
    previous = torch.npu.get_stream_limit(quota_stream)
    try:
        for quota in (28, 8, 2, 32):
            torch.npu.set_stream_limit(
                quota_stream, cube_num=quota, vector_num=2 * quota
            )
            for workspace in (front_workspace, tail_workspace):
                for tensor in vars(workspace).values():
                    if isinstance(tensor, torch.Tensor) and tensor.numel():
                        tensor.fill_(37)
            program(*args)
            torch.npu.synchronize()
            for name, expected in quota_reference.items():
                assert torch.equal(workspace_bytes()[name], expected), (quota, name)
            assert program._core_count == quota
            if quota == 28:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=quota_stream):
                    program(*args)
                for workspace in (front_workspace, tail_workspace):
                    for tensor in vars(workspace).values():
                        if isinstance(tensor, torch.Tensor) and tensor.numel():
                            tensor.fill_(53)
                graph.replay()
                torch.npu.synchronize()
                for name, expected in quota_reference.items():
                    assert torch.equal(workspace_bytes()[name], expected), (
                        "graph",
                        quota,
                        name,
                    )
            print(f"PASS TND core quota {quota}: workspace bytes and Graph consistent")
    finally:
        torch.npu.set_stream_limit(
            quota_stream,
            cube_num=previous["cube_core_num"],
            vector_num=previous["vector_core_num"],
        )


# Exercise more M tiles than physical cores; no stale query/score data may survive.
large_reference = {
    name: getattr(batch_tail, name).cpu().contiguous().view(torch.uint8).clone()
    for name in ("q", "q_scale", "indices", "score", "visible_lengths", "block_mask")
}
with torch.npu.stream(quota_stream):
    previous = torch.npu.get_stream_limit(quota_stream)
    try:
        torch.npu.set_stream_limit(quota_stream, cube_num=8, vector_num=16)
        for name in large_reference:
            getattr(batch_tail, name).fill_(37)
        batch_program(*batch_args)
        torch.npu.synchronize()
        for name, expected in large_reference.items():
            actual = getattr(batch_tail, name).cpu().contiguous().view(torch.uint8)
            assert torch.equal(actual, expected), name
        print("PASS B=256 with 8 cores: all M tiles recomputed bitwise identically")
    finally:
        torch.npu.set_stream_limit(
            quota_stream,
            cube_num=previous["cube_core_num"],
            vector_num=previous["vector_core_num"],
        )
