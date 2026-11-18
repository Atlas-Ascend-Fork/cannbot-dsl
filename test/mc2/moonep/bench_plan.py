# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""moonep_plan 基线：双传输（stage.sh=UBMEM / stage_urma.sh=URMA）对拍。

口径同 bench_reduce_grad（§11 教训）：
- **kernel 时长**：profiler kernel_details.csv Duration 中位数（每拍恰
  1 次 plan 发射——单 launch 设计，口径成立）；
- 逐拍 span：对齐 barrier[i]→barrier[i+1]（含 host/队列间隙）；
- sustained：async 背靠背 + npu Event 总时（profiler 外——host 发射
  路径不被 profiler 污染）。

运行（A5_8P）：
    torchrun --nproc_per_node=4 test/mc2/moonep/bench_plan.py \
        --tokens 4096 --top-k 8 --epn 28 --iters 20
"""

import argparse
import csv
import glob
import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "samples", "mc2", "moonep"
    ),
)

import torch_npu  # noqa: E402,F401
from torch_npu.profiler import ProfilerActivity, ProfilerLevel, schedule  # noqa: E402

from moonep_context import create_moonep_context  # noqa: E402
from moonep_golden import RouteShape, make_router_inputs  # noqa: E402
from moonep_plan import moonep_plan  # noqa: E402

_PLAN_HINTS = ("plankernel",)


def _iter_kernels(out_dir):
    """逐 rank 只取最新 trace（陈旧 CSV 污染免疫——§13.1 教训）。"""
    latest = {}
    for path in glob.glob(
        os.path.join(out_dir, "**", "kernel_details.csv"), recursive=True
    ):
        key = os.path.dirname(os.path.dirname(os.path.dirname(path)))
        old = latest.get(key)
        if old is None or os.path.getmtime(path) > os.path.getmtime(old):
            latest[key] = path
    for path in sorted(latest.values()):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                yield row


def _summarize(out_dir):
    """(kernel 时长列表, 迭代 span 列表)。"""
    kern, tl = [], []
    for row in _iter_kernels(out_dir):
        name = row["Name"].lower()
        try:
            dur = float(row["Duration(us)"])
            start = float(row["Start Time(us)"])
        except ValueError:
            continue
        tl.append((start, start + dur))
        if any(h in name for h in _PLAN_HINTS):
            kern.append(dur)
    return kern, tl


def _fmt(values, unit):
    if not values:
        return "n/a"
    return (
        f"med {statistics.median(values):.1f}{unit} min {min(values):.1f} "
        f"max {max(values):.1f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--epn", type=int, default=28)
    parser.add_argument("--mode", default="balanced")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--sustained", action="store_true")
    parser.add_argument("--out", default="/tmp/plan_bench")
    args = parser.parse_args()

    dist.init_process_group(backend="hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.npu.set_device(rank)
    ctx = create_moonep_context()
    ctx.last_plan_shape = (args.tokens, args.top_k)

    shape = RouteShape(world, args.tokens, args.top_k, args.epn * world)
    topk_all, _ = make_router_inputs(shape, mode=args.mode, seed=args.seed)
    topk = topk_all[rank].to(torch.int32).npu()
    tpe = (
        torch.bincount(topk_all[rank].flatten(), minlength=shape.num_experts)
        .to(torch.int32)
        .npu()
    )

    for _ in range(args.warmup):
        moonep_plan(ctx, tpe, topk)
    torch.npu.synchronize()
    dist.barrier()

    out_dir = os.path.join(args.out, f"r{rank}")
    config = torch_npu.profiler._ExperimentalConfig(profiler_level=ProfilerLevel.Level0)
    with torch_npu.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        schedule=schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out_dir),
        experimental_config=config,
    ) as prof:
        for _ in range(args.iters):
            moonep_plan(ctx, tpe, topk)
            torch.npu.synchronize()
        prof.step()

    sus_us = None
    if args.sustained:
        dist.barrier()
        torch.npu.synchronize()
        ev0 = torch.npu.Event(enable_timing=True)
        ev1 = torch.npu.Event(enable_timing=True)
        ev0.record()
        for _ in range(args.iters):
            moonep_plan(ctx, tpe, topk)
        ev1.record()
        torch.npu.synchronize()
        sus_us = ev0.elapsed_time(ev1) * 1000.0 / args.iters

    kern, _tl = _summarize(out_dir)
    payload = {
        "rank": rank,
        "kernel_us": kern,
        "sustained_us": sus_us,
    }
    gathered = [None] * world
    dist.all_gather_object(gathered, payload)
    dist.barrier()

    if rank == 0:
        print(
            f"\n== moonep_plan 基线 ==\n"
            f"规格: R={world} S={args.tokens} K={args.top_k} epn={args.epn} "
            f"mode={args.mode} iters={args.iters}\n"
        )
        for item in gathered:
            k = item["kernel_us"]
            k_med = f"{statistics.median(k):.1f}us" if k else "n/a"
            k_gbs = (
                f"{args.tokens * args.top_k * 4 / (statistics.median(k) * 1e-6) / 1e9:.1f} MTok/s"
                if k
                else "n/a"
            )
            sus = f"{item['sustained_us']:.1f}us" if item.get("sustained_us") else "n/a"
            print(
                f"rank{item['rank']}: plan kernel med {k_med} ({k_gbs}) | sus {sus} | "
                f"kernel {_fmt(k, 'us')}"
            )
    ctx.close()


if __name__ == "__main__":
    main()
