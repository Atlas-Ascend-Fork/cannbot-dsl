# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Cold-cache device profiler capture against an explicit reference fixture."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cannbotdsl
import torch
import torch_npu

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument(
    "--reference",
    type=Path,
    required=True,
    help="fixture from verify_migration.py capture on the reference implementation",
)
parser.add_argument("--samples", type=int, default=30)
args = parser.parse_args()
reference_path = args.reference.expanduser().resolve()
if not reference_path.is_file():
    parser.error(
        f"reference fixture not found: {reference_path}; "
        "generate it with verify_migration.py capture before profiling"
    )
reference = torch.load(reference_path, map_location="cpu", weights_only=True)
required = ("full/tail/indices", "full/tail/score")
if not isinstance(reference, dict) or any(
    name not in reference or not isinstance(reference[name], torch.Tensor)
    for name in required
):
    parser.error(f"reference fixture must contain tensor keys: {required}")
source = args.source.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
test_root = next(
    (parent for parent in source.parents if (parent / "_samples_path.py").is_file()),
    source.parent.parent,
)
sys.path.insert(0, str(test_root))
ns = {"__package__": "attention.mega_reindex", "__file__": str(source)}
exec(compile(source.read_text().split("scrub = ")[0], str(source), "exec"), ns)
program, inputs = ns["program"], ns["args"]
scrub = torch.ones(200_000_000, dtype=torch.float32, device="npu")


def clear_cache():
    torch.sum(scrub)
    torch.npu.synchronize()


for _ in range(5):
    clear_cache()
    program(*inputs)
    torch.npu.synchronize()
assert torch.equal(ns["tail_workspace"].indices.cpu(), reference["full/tail/indices"])
assert torch.equal(ns["tail_workspace"].score.cpu(), reference["full/tail/score"])
clear_cache()
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    program(*inputs)
torch.npu.synchronize()
for _ in range(3):
    clear_cache()
    graph.replay()
    torch.npu.synchronize()
metadata = {
    "dsl_version": cannbotdsl.__version__,
    "dsl_path": cannbotdsl.__file__,
    "sample_path": str(source),
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "case": {"B": 1, "S": 8, "KV": 8192, "topk": 512, "seed": 2026},
    "device": torch.npu.get_device_name(0),
    "platform": str(cannbotdsl.get_platform_info()),
    "samples": args.samples,
    "reference_path": str(reference_path),
    "clear_cache_elements": scrub.numel(),
    "clear_cache_bytes": scrub.numel() * scrub.element_size(),
    "cache_protocol": "FP32 ReduceSum then device synchronize before every invocation; eviction workload, not a cache-invalidate instruction",
    "accuracy": "score and indices bit-exact against the supplied reference fixture before and after profiling",
    "source_hashes": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(
            sys.modules[ns["prepare_mega_reindex"].__module__].__file__
        ).parent.glob("*.py")
    },
}
(output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("PROFILE", metadata["dsl_version"], args.samples, flush=True)
with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    schedule=torch_npu.profiler.schedule(
        wait=0, warmup=1, active=args.samples, repeat=1
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output)),
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
) as prof:
    for iteration in range(args.samples + 1):
        clear_cache()
        with torch.profiler.record_function(f"target_replay_{iteration:03d}"):
            graph.replay()
            torch.npu.synchronize()
        prof.step()
assert torch.equal(ns["tail_workspace"].indices.cpu(), reference["full/tail/indices"])
assert torch.equal(ns["tail_workspace"].score.cpu(), reference["full/tail/score"])
print("DONE", output, flush=True)
