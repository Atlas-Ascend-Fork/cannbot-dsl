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

# 精确匹配 DSL 生成的 kernel 名（mangled 名里含 "MoonepReducePullKernel"、
# "MoonepBarrierKernel"），避免把 dist.barrier() 的 HCCL kernel 或 Fill 算进来。
_PULL_HINTS = (
    "reducesw",
    "reducepull",
)  # v7 SWPUSH（ReduceSw）兼容 v4 pull（ReducePull）
_MERGE_HINTS = ("reducemerge",)  # 两段式第二段（迭代 span 终点成分）
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
    """逐 rank 只取**最新** trace 的 kernel_details.csv。

    out_dir 跨 run 累积（同 --out 复用）时全量递归会把旧 run 的 kernel
    时长掺入中位数——2026-11-15 实证：K=28 单发射真值 374.9µs 被旧
    K=8 的 112µs 稀释成 med 112.7（假带宽 513 vs 真 332.8）。按 rank
    目录分组取 mtime 最新一份即免疫。
    """
    latest = {}
    for path in glob.glob(
        os.path.join(out_dir, "**", "kernel_details.csv"), recursive=True
    ):
        # 分组键 = r{rank} 目录（path 上溯三级：kernel_details.csv ←
        # ASCEND_PROFILER_OUTPUT ← host 时间戳目录 ← r{rank}）
        key = os.path.dirname(os.path.dirname(os.path.dirname(path)))
        old = latest.get(key)
        if old is None or os.path.getmtime(path) > os.path.getmtime(old):
            latest[key] = path
    for path in sorted(latest.values()):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                yield row


def _summarize(out_dir):
    """返回 (pull 时长列表, [(barrier 起始时间, 时长)], kernel 名集合, 迭代 span 列表)。

    迭代 span：对齐 barrier[i] 起始 → barrier[i+1] 起始（末拍到本拍最后
    一个 pull/merge/barrier kernel 的结束）。**逐拍口径**——多批次两段式
    下 per-launch 中位数口径会把单次发射时长当整拍（×批数虚高，2026-11-15
    实证 313 GB/s vs 真值 20.7），带宽一律以 iter 口径为准。
    launch-med 口径仅在**每拍恰 1 次 pull 发射**时与 iter 等价（单
    kernel 设计：prefetch/K=epn 直累/UbMem）；多发射/拍一律看 iter。
    """
    pull, barrier, names = [], [], set()
    tl = []  # (start, end) 全部相关 kernel——迭代时间线
    for row in _iter_kernels(out_dir):
        name = row["Name"].lower()
        names.add(row["Name"])
        try:
            dur = float(row["Duration(us)"])
            start = float(row["Start Time(us)"])
        except ValueError:
            continue
        tl.append((start, start + dur))
        if any(hint in name for hint in _BARRIER_HINTS):
            barrier.append((start, dur))
        elif any(hint in name for hint in _PULL_HINTS):
            pull.append(dur)
    iter_spans = []
    bstarts = sorted(s for s, _ in barrier)
    if bstarts and tl:
        last_end = max(e for _, e in tl)
        for a, b in zip(bstarts, bstarts[1:]):
            iter_spans.append(b - a)
        iter_spans.append(last_end - bstarts[-1])
    return pull, barrier, names, iter_spans


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
    parser.add_argument(
        "--sustained",
        action="store_true",
        help="加测持续吞吐（async_finish 背靠背 + 事件总时——训练流水形态，host 与设备重叠）",
    )
    parser.add_argument("--out", default="/tmp/moonep_reduce_grad_bench")
    args = parser.parse_args()

    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.npu.set_device(rank)

    # workspace 按形状精算（分配即所需——context 缺省仅 plan 交换区下界）
    from moonep_buffer import _workspace_need

    ctx = create_moonep_context(
        workspace_bytes=_workspace_need(
            world, args.epn, args.w_elems, 1, torch.bfloat16, torch.bfloat16
        )
    )
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

        # 每拍先全组对齐再测（对齐用 sym_barrier——不走 HCCL）。
        # 0.8.2 起 finish_op 不再追加独立 barrier（到达等待内联在 pull
        # kernel 中）——每拍恰 1 个对齐 barrier kernel，直接全量统计。
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

        # —— 持续吞吐（训练流水形态：async 背靠背、host 与设备重叠、
        #     事件计总时/拍——与逐拍 sync 的串行口径互补；无 profiler）——
        sus_us = None
        if args.sustained:
            dist.barrier()
            torch.npu.synchronize()
            ev0 = torch.npu.Event(enable_timing=True)
            ev1 = torch.npu.Event(enable_timing=True)
            ev0.record()
            for _ in range(args.iters):
                # 无逐拍 leading barrier（跨 rank 序由 reduce 内部尾部
                # barrier + kernel 旗协议保——leading 会合点在流水形态
                # 纯属串行化开销）
                reduce_grad(ctx, etc, grads, args.epn, async_finish=True)
            ev1.record()
            torch.npu.synchronize()
            sus_us = ev0.elapsed_time(ev1) * 1000.0 / args.iters

        pull, barrier_rows, names, iter_spans = _summarize(out_dir)
        payload = {
            "rank": rank,
            "pulled_bytes": _pulled_bytes(etc_cpu, rank, args.epn, args.w_elems),
            "pull": pull,
            "iter": iter_spans,
            "align_barrier": [dur for _, dur in barrier_rows],
            "sustained_us": sus_us,
            "barrier": [],
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
                pull, barrier, it = item["pull"], item["barrier"], item["iter"]
                gbs, iter_gbs, iter_med = "n/a", "n/a", "n/a"
                if pull and item["pulled_bytes"]:
                    gbs = f"{item['pulled_bytes'] / (statistics.median(pull) * 1e-6) / 1e9:.1f} GB/s"
                if it and item["pulled_bytes"]:
                    iter_med = f"{statistics.median(it):.1f}us"
                    iter_gbs = f"{item['pulled_bytes'] / (statistics.median(it) * 1e-6) / 1e9:.1f} GB/s"
                if item.get("sustained_us") and item["pulled_bytes"]:
                    iter_med += f" | sus {item['sustained_us']:.1f}us"
                    iter_gbs += f" | sus {item['pulled_bytes'] / (item['sustained_us'] * 1e-6) / 1e9:.1f} GB/s"
                print(
                    f"rank{item['rank']}: pulled {item['pulled_bytes'] / 2**20:.3f} MiB | "
                    f"pull {_fmt(pull, 'us')} | "
                    f"iter med {iter_med} | **{iter_gbs}** | "
                    f"barrier {_fmt(barrier, 'us')} | "
                    f"align {_fmt(item['align_barrier'], 'us')} | launch-med {gbs}"
                )
            print(f"rank0 kernels: {gathered[0]['names']}")


if __name__ == "__main__":
    main()
