# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import json
import sys
import time
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "samples"))
source = Path(__file__).with_name("smoke.py")
s = source.read_text().split("scrub = ")[0]
ns = {"__package__": "attention.mega_reindex", "__file__": str(source)}
exec(compile(s, str(source), "exec"), ns)
prepare = ns["prepare_mega_reindex"]
Workspace = ns["MegaReindexTailWorkspace"]
scrub = torch.ones(200 * 1024 * 1024 // 4, device="npu", dtype=torch.float32)


def run(prog, args):
    torch.sum(scrub)
    prog(*args)
    torch.npu.synchronize()


run(ns["program"], ns["args"])
gold = ns["tail_workspace"].indices.cpu()
score = ns["tail_workspace"].score.cpu()
results = []
for page_size in (128, 64, 16, 1024):
    pages = 8192 // page_size
    torch.manual_seed(900 + page_size)
    perm = torch.randperm(pages)
    # Pad physical page axis and reverse logical mapping.
    host = torch.zeros((pages, page_size // 8 + 1, 544), dtype=torch.uint8)
    data = torch.cat(
        (
            ns["key"].cpu().reshape(pages, page_size // 8, 512),
            ns["descale_key"].cpu().reshape(pages, page_size // 8, 32),
        ),
        dim=-1,
    )
    host[perm, : page_size // 8] = data
    packed = host.npu()[:, : page_size // 8, :]
    table = perm.to(torch.int32).view(1, pages).npu()
    common = dict(
        front_workspace=ns["front_workspace"],
        topk=512,
        candidate_block_indices=ns["candidate_block_indices"],
        candidate_block_length=ns["candidate_block_length"],
        cu_seqlens_q=ns["cu_seqlens_q"],
        seqused_q=ns["seqused_q"],
        seqused_k=ns["seqused_k"],
        layout_k="PA_BBND",
        block_table=table,
        k=packed,
        descale_k=None,
    )
    p = ns["p"]
    pos = (
        p["x"],
        ns["wqa"],
        ns["descale_wqa"],
        ns["ww"],
        p["g"],
        ns["wqb"],
        ns["descale_wqb"],
        p["rope"],
    )
    print("COMPILE_PA", page_size, flush=True)
    start = time.perf_counter()
    prog, args, fw, tw = prepare(*pos, **common)
    print("COMPILED", time.perf_counter() - start, flush=True)
    run(prog, args)
    check = {
        "key_exact": torch.equal(tw.pa_key.cpu(), ns["key"].cpu()),
        "scale_exact": torch.equal(tw.pa_scale.cpu(), ns["descale_key"].cpu()),
        "score_exact": torch.equal(tw.score.cpu(), score),
        "indices_exact": torch.equal(tw.indices.cpu(), gold),
    }
    print(page_size, check, flush=True)
    assert all(check.values())
    mega_reindex = ns["_sample"].mega_reindex
    torch.sum(scrub)
    public = mega_reindex(*pos, tail_workspace=tw, **common)
    torch.npu.synchronize()
    assert public.data_ptr() == tw.indices.data_ptr()
    assert torch.equal(public.cpu(), gold)
    for bad in (
        dict(descale_k=ns["descale_key"]),
        dict(block_table=None),
        dict(cu_seqlens_k=ns["cu_seqlens_k"]),
        dict(layout_k="invalid"),
    ):
        options = dict(common)
        options.update(bad)
        try:
            prepare(*pos, **options)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(("accepted invalid PA arguments", bad.keys()))

    for case in ("short", "empty", "inactive_query", "remap"):
        ns["seqused_k"].fill_(8192 - page_size - 3 if case == "short" else 8192)
        ns["candidate_block_length"].fill_(0 if case == "empty" else 1024)
        ns["seqused_q"].fill_(6 if case == "inactive_query" else 8)
        table.copy_(perm.to(torch.int32).view(1, pages).npu())
        if case == "short":
            table[0, -1] = -1
        if case == "remap":
            # Logical page zero changes in-place without recompilation.
            table[0, 0] = perm[1].item()
            ns["key"][:page_size].copy_(ns["key"][page_size : 2 * page_size].clone())
            ns["descale_key"][:page_size].copy_(
                ns["descale_key"][page_size : 2 * page_size].clone()
            )
        run(ns["program"], ns["args"])
        run(prog, args)
        assert torch.equal(tw.indices.cpu(), ns["tail_workspace"].indices.cpu()), case
        assert torch.equal(tw.score.cpu(), ns["tail_workspace"].score.cpu()), case
        check[case] = True
    # Restore both layouts and metadata, then verify workspace reuse.
    ns["key"].copy_(ns["packed_fixture"][0].npu())
    ns["descale_key"].copy_(ns["packed_fixture"][1].npu())
    ns["seqused_k"].fill_(8192)
    ns["seqused_q"].fill_(8)
    ns["candidate_block_length"].fill_(1024)
    table.copy_(perm.to(torch.int32).view(1, pages).npu())
    run(ns["program"], ns["args"])
    run(prog, args)
    assert torch.equal(tw.indices.cpu(), gold)
    results.append({"page_size": page_size, "accuracy": check})
    print("PASS", page_size, check, flush=True)

print(json.dumps(results, indent=2))
