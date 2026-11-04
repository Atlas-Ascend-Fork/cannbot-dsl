# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Capture a version-isolated fixture or compare migration tensors bit-exactly.

Run capture in separate processes with each DSL and sample on its own path.
The reference is the original implementation, not an independent CPU golden.
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def capture(source, output):
    source = source.resolve()
    test_root = next(
        (
            parent
            for parent in source.parents
            if (parent / "_samples_path.py").is_file()
        ),
        source.parent.parent,
    )
    sys.path.insert(0, str(test_root))
    namespace = {
        "__package__": "attention.mega_reindex",
        "__file__": str(source),
    }
    # Use exactly the sample's seeded inputs and preparation, without timing.
    setup = source.read_text().split("scrub = ")[0]
    exec(compile(setup, str(source), "exec"), namespace)
    tensors = {}
    for case in ("full", "empty", "restore"):
        namespace["candidate_block_length"].fill_(
            0 if case == "empty" else namespace["num_blocks"]
        )
        namespace["program"](*namespace["args"])
        torch.npu.synchronize()
        for prefix, key in (("front", "front_workspace"), ("tail", "tail_workspace")):
            for name, tensor in vars(namespace[key]).items():
                if isinstance(tensor, torch.Tensor) and tensor.numel():
                    tensors[f"{case}/{prefix}/{name}"] = tensor.cpu().clone()
    torch.save(tensors, output)
    print(json.dumps({"captured": len(tensors), "output": str(output)}))


def compare(reference, candidate):
    before = torch.load(reference, map_location="cpu", weights_only=True)
    after = torch.load(candidate, map_location="cpu", weights_only=True)
    assert before.keys() == after.keys(), "capture tensor sets differ"
    results = {}
    for name, expected in before.items():
        actual = after[name]
        results[name] = (
            actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )
    print(json.dumps({"comparisons": results, "pass": all(results.values())}, indent=2))
    assert all(results.values()), "migration differs from the reference"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("source", type=Path)
    cap.add_argument("output", type=Path)
    cmp = sub.add_parser("compare")
    cmp.add_argument("reference", type=Path)
    cmp.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.source, args.output)
    else:
        compare(args.reference, args.candidate)
