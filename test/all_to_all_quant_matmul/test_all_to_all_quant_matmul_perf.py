# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""性能基线驱动：精度门禁 + 重型算子对齐的 10 轮计时（取后 5 平均）。

复用 test_all_to_all_quant_matmul 的七钩子与 runner 编排，仅覆写两处：
  worker_prepare：追加重型算子操作数（8192³ fp16 matmul）
  execute：每轮 [重型 matmul → ev0 → 被测 kernel → ev1] 全部预入队，
           不逐轮 sync，末尾一次 synchronize——host 下发抖动被重型
           算子的执行窗口吸收，被测 kernel 的启动时刻由设备队列决定，
           避免快慢卡；NPU Event 取纯设备时长。返回最后一轮 y 交父
           进程 golden 比对，精度 FAIL 则该例性能数据作废。

用例选择：cases.csv 首列非空的行即性能用例（名字源自 aclnn 用例表）；
--cases 按名选择（如 case1,case_32m），缺省跑全部性能用例。

用法（每 case 独立进程，防编译缓存死锁；在仓库根目录执行）：
  python3 test/all_to_all_quant_matmul/test_all_to_all_quant_matmul_perf.py \
      --csv test/all_to_all_quant_matmul/cases.csv \
      --devices 0,1,2,3 --comm-mode urma --cases case1
"""

import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "samples" / "all_to_all_quant_matmul"))
sys.path.insert(0, str(_HERE))

import test_all_to_all_quant_matmul as TM  # noqa: E402 (依赖上方 sys.path 注入)


def log(tag, message):
    import logging

    logging.getLogger("perf").info(f"[{tag}] {message}")


HEAVY_N = 8192


def _named_perf_rows(csv_path):
    """cases.csv 中首列非空的数据行：用例名 -> 0-based 数据行下标。"""
    with open(csv_path, encoding="utf-8") as handle:
        rows = [
            row for row in csv.reader(handle) if row and any(c.strip() for c in row)
        ]
    names = {}
    for idx, row in enumerate(rows[1:]):
        name = row[0].strip()
        if name:
            names.setdefault(name, idx)
    return names


class PerfBaseline(TM.TestAllToAllQuantMatmul):
    """精度门禁 + 计时执行器。"""

    __test__ = False

    def worker_prepare(self, *args, **kwargs):
        res = super().worker_prepare(*args, **kwargs)
        import torch

        res["heavy_a"] = torch.randn(HEAVY_N, HEAVY_N, dtype=torch.float16).npu()
        res["heavy_b"] = torch.randn(HEAVY_N, HEAVY_N, dtype=torch.float16).npu()
        return res

    def execute(self, prepare_res, **params):
        import torch

        rounds = int(params.get("rounds", 10))
        kernel_main = prepare_res["main"]
        call_args = (
            prepare_res["data_npu"],
            prepare_res["barrier_npu"],
            prepare_res["a_npu"],
            prepare_res["sa_npu"],
            prepare_res["b_npu"],
            prepare_res["sb_npu"],
            prepare_res["y_npu"],
            prepare_res["bias_npu"],
        )
        events = [
            (
                torch.npu.Event(enable_timing=True),
                torch.npu.Event(enable_timing=True),
            )
            for _ in range(rounds)
        ]
        for r in range(rounds):
            torch.matmul(prepare_res["heavy_a"], prepare_res["heavy_b"])
            events[r][0].record()
            kernel_main(*call_args)
            events[r][1].record()
        torch.npu.synchronize()
        times = [e0.elapsed_time(e1) * 1000 for e0, e1 in events]
        log("perf", f"rounds_us = {[f'{t:.1f}' for t in times]}")
        log("perf", f"last5_avg_us = {sum(times[-5:]) / 5:.1f}")
        result = prepare_res["y_npu"].cpu().float().numpy()
        return {"y": result}


def main():
    test = PerfBaseline()
    test.set_kernel_name_pattern(TM.KERNEL_NAME_PATTERN)
    test.parser.add_argument(
        "--comm-mode",
        default="urma",
        choices=["urma", "ub"],
        help="data-plane comm mode (default: urma)",
    )
    test.parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="timed rounds per case; the last 5 are averaged (default: 10)",
    )
    TM.add_stage_arguments(test.parser)
    args = test.parser.parse_args()
    if args.rounds < 6:
        test.parser.error("--rounds must be at least 6 (5 averaged + warmup)")
    names = _named_perf_rows(args.csv)
    if not names:
        test.parser.error(f"no named perf rows found in {args.csv}")
    if args.cases:
        wanted = []
        for token in args.cases.split(","):
            token = token.strip()
            if not token:
                continue
            if token not in names:
                test.parser.error(
                    f"unknown perf case {token!r}; available: {','.join(names)}"
                )
            wanted.append(token)
    else:
        wanted = list(names)
    log("perf", f"perf cases: {','.join(wanted)}")

    args.cases = ",".join(str(names[name]) for name in wanted)
    extra_params = {"comm_mode": args.comm_mode, "rounds": args.rounds}
    ok = TM.run_with_stage(test, args, extra_params=extra_params)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
