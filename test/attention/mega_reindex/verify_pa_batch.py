# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "samples"))
source = Path(__file__).with_name("smoke.py")
s = source.read_text().split("compile_start = ")[0]
ns = {"__package__": "attention.mega_reindex", "__file__": str(source)}
exec(compile(s, str(source), "exec"), ns)
prepare = ns["prepare_mega_reindex"]
p = ns["p"]
x = p["x"][:6].repeat(2, 1)
rope = p["rope"][:6].repeat(2, 1, 1)
key = torch.cat((ns["key"].cpu(), ns["key"].cpu().flip(0))).npu()
scale = torch.cat((ns["descale_key"].cpu(), ns["descale_key"].cpu().flip(0))).npu()
candidates = ns["candidate_block_indices"][:6].repeat(2, 1, 1)
counts = ns["candidate_block_length"][:6].repeat(2, 1)
args = (
    x,
    ns["wqa"],
    ns["descale_wqa"],
    ns["ww"],
    p["g"],
    ns["wqb"],
    ns["descale_wqb"],
    rope,
)
common = dict(
    candidate_block_indices=candidates,
    candidate_block_length=counts,
    cu_seqlens_q=torch.tensor([0, 6, 12], dtype=torch.int32, device="npu"),
    seqused_q=torch.tensor([6, 4], dtype=torch.int32, device="npu"),
    seqused_k=torch.tensor([8192, 8001], dtype=torch.int32, device="npu"),
)
tnd, ta, _, tw = prepare(
    *args,
    k=key,
    descale_k=scale,
    cu_seqlens_k=torch.tensor([0, 8192, 16384], dtype=torch.int32, device="npu"),
    **common,
)
page = 128
pages = 128
perm = torch.randperm(pages)
packed = torch.cat(
    (key.cpu().reshape(pages, 16, 512), scale.cpu().reshape(pages, 16, 32)), dim=-1
)
physical = torch.empty_like(packed)
physical[perm] = packed
pa, paargs, _, pw = prepare(
    *args,
    k=physical.npu(),
    layout_k="PA_BBND",
    block_table=perm.to(torch.int32).reshape(2, 64).npu(),
    **common,
)
scrub = torch.ones(200 * 1024 * 1024 // 4, device="npu", dtype=torch.float32)
for prog, inputs in ((tnd, ta), (pa, paargs)):
    torch.sum(scrub)
    prog(*inputs)
    torch.npu.synchronize()
assert torch.equal(tw.score.cpu(), pw.score.cpu())
assert torch.equal(tw.indices.cpu(), pw.indices.cpu())
assert (pw.indices.cpu()[10:] == -1).all()
print(
    "PASS B=2 S=6 PA128 random distinct batch pages, shortened K and inactive Q",
    flush=True,
)


# The same logical PA pages must be independent of the available core quota.
reference = {
    name: getattr(pw, name).cpu().clone()
    for name in ("indices", "score", "pa_key", "pa_scale")
}
stream = torch.npu.Stream()
with torch.npu.stream(stream):
    previous = torch.npu.get_stream_limit(stream)
    try:
        for quota in (28, 8, 2, 32):
            torch.npu.set_stream_limit(stream, cube_num=quota, vector_num=2 * quota)
            for workspace in (tw, pw):
                for name, tensor in vars(workspace).items():
                    if (
                        name != "pa_cuk"
                        and isinstance(tensor, torch.Tensor)
                        and tensor.numel()
                    ):
                        tensor.fill_(37)
            for prog, inputs in ((tnd, ta), (pa, paargs)):
                prog(*inputs)
            torch.npu.synchronize()
            assert torch.equal(tw.score.cpu(), pw.score.cpu())
            assert torch.equal(tw.indices.cpu(), pw.indices.cpu())
            for name, expected in reference.items():
                assert torch.equal(getattr(pw, name).cpu(), expected), (quota, name)
            if quota == 28:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    pa(*paargs)
                for name, tensor in vars(pw).items():
                    if (
                        name != "pa_cuk"
                        and isinstance(tensor, torch.Tensor)
                        and tensor.numel()
                    ):
                        tensor.fill_(53)
                graph.replay()
                torch.npu.synchronize()
                for name, expected in reference.items():
                    assert torch.equal(getattr(pw, name).cpu(), expected), (
                        "graph",
                        quota,
                        name,
                    )
            print(
                f"PASS PA core quota {quota}: TND parity and Graph consistent",
                flush=True,
            )
    finally:
        torch.npu.set_stream_limit(
            stream,
            cube_num=previous["cube_core_num"],
            vector_num=previous["vector_core_num"],
        )
