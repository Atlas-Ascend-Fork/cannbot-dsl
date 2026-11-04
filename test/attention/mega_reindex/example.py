# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Call example for mega_reindex."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _samples_path import load_sample

_sample = load_sample("attention/mega_reindex/mega_reindex.py")
FrontWorkspace = _sample.FrontWorkspace
MegaReindexTailWorkspace = _sample.MegaReindexTailWorkspace
mega_reindex = _sample.mega_reindex


def run_example(
    tensors, *, topk=512, norm_eps=1e-20, softmax_scale=1 / 64, layout_k="TND"
):
    """Run prepared NPU tensors from a dict and return INT32 [T, topk] indices."""
    x = tensors["x"]
    batch = tensors["seqused_q"].numel()
    key = tensors.get("k", tensors.get("key"))
    kv_s = (
        tensors["block_table"].shape[1] * key.shape[1] * 8
        if layout_k == "PA_BBND"
        else key.shape[0] // batch
    )
    front_workspace = FrontWorkspace(x.shape[0], x.device)
    tail_workspace = MegaReindexTailWorkspace(x.shape[0], kv_s, x.device, topk=topk)
    return mega_reindex(
        x,
        tensors["wqa"],
        tensors["descale_wqa"],
        tensors["ww"],
        tensors["norm_weight"],
        tensors["wqb"],
        tensors["descale_wqb"],
        tensors["rope"],
        key,
        tensors.get("descale_k", tensors.get("descale_key")),
        front_workspace=front_workspace,
        tail_workspace=tail_workspace,
        norm_eps=norm_eps,
        softmax_scale=softmax_scale,
        topk=topk,
        candidate_block_indices=tensors["candidate_block_indices"],
        candidate_block_length=tensors["candidate_block_length"],
        cu_seqlens_q=tensors["cu_seqlens_q"],
        cu_seqlens_k=tensors.get("cu_seqlens_k"),
        seqused_q=tensors["seqused_q"],
        seqused_k=tensors["seqused_k"],
        layout_k=layout_k,
        block_table=tensors.get("block_table"),
    )
