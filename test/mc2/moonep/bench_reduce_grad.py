# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""reduce_grad 的 kernel 级性能基线（与 bench_prefetch_weight.py 同款口径）。

绕过 moonep_plan（手工补 context.last_plan_shape 并手工构造 experts_to_copy），
只用 profiler 采集 pull kernel 与 barrier kernel 的设备执行时间。

用法（R 卡）：
    torchrun --nproc_per_node=8 test/moonep/bench_reduce_grad.py \
        --epn 28 --w-elems 114688 --pattern fanout,ring --iters 20
"""

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
from reduce_grad import reduce_grad  # noqa: E402

# 精确匹配 DSL 生成的 kernel 名（mangled 名里含 "MoonepReducePullAccKernel"、
# "MoonepBarrierKernel"），避免把 dist.barrier() 的 HCCL kernel 或 Fill 算进来。
_PULL_HINTS = ("reducepull",)
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


def _pulled_bytes(etc_cpu, rank, epn, w_elems, groups=3):
    """本 rank 作为 owner 需远端拉取的副本行数 → 拉取字节数（bf16）。"""
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
    """返回 (pull 时长列表, [(barrier 起始时间, 时长)], kernel 名集合)。"""
    pull, barrier, names = [], [], set()
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
        elif any(hint in name for hint in _PULL_HINTS):
            pull.append(dur)
    return pull, barrier, names


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
    parser.add_argument("--w-elems", type=int, required=True, help="每组梯度表行宽 W")
    parser.add_argument(
        "--pattern", default="fanout,ring", help="逗号分隔：fanout,ring"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", default="/tmp/moonep_reduce_grad_bench")
    args = parser.parse_args()

    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.npu.set_device(rank)

    ctx = create_moonep_context()
    ctx.last_plan_shape = (32, 8)  # 绕开 plan 的调用顺序守卫

    # full-dW 契约：[2*epn, W] 张量（前 epn 行 home 种子、后 epn 行副本 dW
    # ——副本入槽由 reduce 内部完成；owner 远端读槽原子累加进 home 段）
    grads = []
    for g in range(3):
        t = torch.full(
            (2 * args.epn, args.w_elems), float(rank * 1000 + g), dtype=torch.bfloat16
        ).npu()
        grads.append(t)

    for pattern in args.pattern.split(","):
        etc_cpu = _build_etc(pattern, world, args.epn)
        etc = etc_cpu.npu()

        for _ in range(args.warmup):  # 首次调用含编译
            reduce_grad(ctx, etc, grads, args.epn)
        torch.npu.synchronize()
        dist.barrier()

        # 每拍先全组对齐再测（对齐用 sym_barrier——不走 HCCL）：
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
                # async_finish=True 保持历史测量结构（v2 前模块缺省 barrier=True
                # 即异步形态；正名后缺省翻转为阻塞——多一次内部 barrier 会
                # 改变 trace 的 barrier 行奇偶性分析）
                reduce_grad(ctx, etc, grads, args.epn, async_finish=True)
                torch.npu.synchronize()
            prof.step()

        pull, barrier_rows, names = _summarize(out_dir)
        barrier_rows.sort()
        payload = {
            "rank": rank,
            "pulled_bytes": _pulled_bytes(etc_cpu, rank, args.epn, args.w_elems),
            "pull": pull,
            "align_barrier": [dur for _, dur in barrier_rows[0::2]],
            "barrier": [dur for _, dur in barrier_rows[1::2]],
            "names": sorted(names),
        }
        gathered = [None] * world
        dist.all_gather_object(gathered, payload)
        dist.barrier()

        if rank == 0:
            print(
                f"\n== reduce_grad kernel 基线 ==\n"
                f"规格: R={world} epn={args.epn} W={args.w_elems} groups=3 "
                f"pattern={pattern} iters={args.iters}\n"
            )
            for item in gathered:
                pull, barrier = item["pull"], item["barrier"]
                gbs = "n/a"
                if pull and item["pulled_bytes"]:
                    gbs = f"{item['pulled_bytes'] / (statistics.median(pull) * 1e-6) / 1e9:.1f} GB/s"
                print(
                    f"rank{item['rank']}: pulled {item['pulled_bytes'] / 2**20:.3f} MiB | "
                    f"pull {_fmt(pull, 'us')} | "
                    f"barrier {_fmt(barrier, 'us')} | "
                    f"align {_fmt(item['align_barrier'], 'us')} | {gbs}"
                )
            print(f"rank0 kernels: {gathered[0]['names']}")


if __name__ == "__main__":
    main()
