# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""prefetch_weight 的设备性能基准。

使用显式窗口元数据和专家预取清单，通过 profiler 采集权重推送与同步内核耗时。

用法（R 卡）：
    torchrun --nproc_per_node=8 test/mc2/moonep/bench_prefetch_weight.py \
        --epn 28 --w-elems 114688 --pattern fanout --iters 20"""

import argparse
import csv
import glob
import os
import statistics
import sys

import torch
import torch.distributed as dist
import torch_npu

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "samples", "mc2", "moonep"
    ),
)

from moonep_context import create_moonep_context, sym_barrier  # noqa: E402
from prefetch_weight import prefetch_weight  # noqa: E402

# 精确匹配 DSL 生成的 kernel 名（mangled 名里含 "MoonepPrefetchPushKernel"、
# "MoonepBarrierKernel"），避免把 dist.barrier() 的 HCCL kernel 或 Fill 算进来。
_PUSH_HINTS = ("prefetchpush",)
_BARRIER_HINTS = ("moonepbarrierkernel",)


def _build_etc(pattern, rank_size, epn):
    """构造与 plan 输出同格式的 experts_to_copy [R, epn]，每行单一 home rank。

    fanout: 除 rank0 外的每个 dst 都从 home0 取 epn 个专家（最坏 fan-out）。
    ring  : dst d 从 home (d+1)%R 取，每个 rank 恰好服务一个 peer。
    """
    etc = torch.full((rank_size, epn), -1, dtype=torch.int32)
    for dst in range(rank_size):
        if pattern == "fanout":
            if dst == 0:
                continue
            etc[dst] = torch.arange(epn, dtype=torch.int32)
        elif pattern == "ring":
            home = (dst + 1) % rank_size
            etc[dst] = torch.arange(home * epn, (home + 1) * epn, dtype=torch.int32)
        else:
            raise ValueError(f"unknown pattern {pattern}")
    return etc


def _pushed_bytes(etc_cpu, rank, epn, w_elems, groups=3):
    """本 rank 作为 home 被请求的行数 → 推送字节数（bf16）。"""
    rows = sum(
        1 for row in etc_cpu.tolist() for e in row if e >= 0 and e // epn == rank
    )
    return rows * w_elems * 2 * groups


def _iter_kernels(out_dir):
    for path in glob.glob(
        os.path.join(out_dir, "**", "kernel_details.csv"), recursive=True
    ):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                yield row


def _summarize(out_dir):
    """返回 (push 时长列表, [(barrier 起始时间, 时长)], kernel 名集合)。"""
    push, barrier, names = [], [], set()
    for row in _iter_kernels(out_dir):
        name = row["Name"].lower()
        names.add(row["Name"])
        try:
            dur = float(row["Duration(us)"])
            start = float(row["Start Time(us)"])
        except ValueError:
            continue
        if any(hint in name for hint in _BARRIER_HINTS):
            barrier.append((start, dur))
        elif any(hint in name for hint in _PUSH_HINTS):
            push.append(dur)
    return push, barrier, names


def _fmt(values, unit):
    if not values:
        return "n/a"
    return (
        f"med {statistics.median(values):.1f}{unit} "
        f"min {min(values):.1f}{unit} "
        f"max {max(values):.1f}{unit}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epn", type=int, required=True, help="每 rank 本地专家数")
    parser.add_argument("--w-elems", type=int, required=True, help="每组权重视图行宽 W")
    parser.add_argument(
        "--pattern", default="fanout,ring", help="逗号分隔：fanout,ring"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", default="/tmp/moonep_prefetch_bench")
    args = parser.parse_args()

    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.npu.set_device(rank)

    ctx = create_moonep_context()
    ctx.last_plan_shape = (32, 8)  # 绕开 plan 的调用顺序守卫

    reals = [
        torch.full(
            (args.epn, args.w_elems), float(rank * 1000 + g), dtype=torch.bfloat16
        ).npu()
        for g in range(3)
    ]

    for pattern in args.pattern.split(","):
        etc_cpu = _build_etc(pattern, world, args.epn)
        etc = etc_cpu.npu()

        for _ in range(args.warmup):  # 首次调用含编译
            prefetch_weight(ctx, etc, reals, args.epn)
        torch.npu.synchronize()
        dist.barrier()

        # 每拍先全组对齐再测：对齐用样例自带的 sym_barrier（对称堆自旋，
        # 不走 HCCL）——本机把 dist.barrier() 放进 profiler 窗口会触发
        # HCCL AICPU 的 RunAicpuIndOpThreadInit 异常。
        # 每拍因此有 2 个 barrier kernel：对齐（调用前）+ 收尾（finish_op 内），
        # 按起始时间配对后分别统计。
        out_dir = os.path.join(args.out, pattern, f"r{rank}")
        config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level0
        )
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=0, active=1, repeat=1, skip_first=0
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out_dir),
            experimental_config=config,
        ) as prof:
            for _ in range(args.iters):
                sym_barrier(ctx)
                prefetch_weight(ctx, etc, reals, args.epn)
                torch.npu.synchronize()
            prof.step()

        push, barrier_rows, names = _summarize(out_dir)
        barrier_rows.sort()
        payload = {
            "rank": rank,
            "pushed_bytes": _pushed_bytes(etc_cpu, rank, args.epn, args.w_elems),
            "push": push,
            "align_barrier": [dur for _, dur in barrier_rows[0::2]],
            "barrier": [dur for _, dur in barrier_rows[1::2]],
            "names": sorted(names),
        }
        gathered = [None] * world
        dist.all_gather_object(gathered, payload)
        dist.barrier()

        if rank == 0:
            view_bytes = 2 * args.epn * args.w_elems * 2 * 3
            print(
                f"\n== prefetch_weight kernel 基线 ==\n"
                f"规格: R={world} epn={args.epn} W={args.w_elems} groups=3 "
                f"pattern={pattern} iters={args.iters}\n"
                f"权重视图: {view_bytes / 2**20:.2f} MiB/rank\n"
            )
            for item in gathered:
                push, barrier = item["push"], item["barrier"]
                gbs = "n/a"
                if push and item["pushed_bytes"]:
                    gbs = f"{item['pushed_bytes'] / (statistics.median(push) * 1e-6) / 1e9:.1f} GB/s"
                print(
                    f"rank{item['rank']}: pushed {item['pushed_bytes'] / 2**20:.3f} MiB | "
                    f"push {_fmt(push, 'us')} | "
                    f"barrier {_fmt(barrier, 'us')} | "
                    f"align {_fmt(item['align_barrier'], 'us')} | {gbs}"
                )
            print(f"rank0 kernels: {gathered[0]['names']}")


if __name__ == "__main__":
    main()
