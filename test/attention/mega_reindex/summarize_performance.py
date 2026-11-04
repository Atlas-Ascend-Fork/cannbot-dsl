# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate recorded device tasks and summarize A/B/B/A profiler captures."""

import argparse
import csv
import json
import statistics
from decimal import Decimal
from pathlib import Path

MAIN = "mega_reindex_kernel"
FILL = "aclnnInplaceFillScalar_FillAiCore_Fill"
SCRUB = "aclnnReduceSum_ReduceSumOpAiCore_ReduceSum"


def stats(values):
    return dict(
        n=len(values),
        p50_us=statistics.median(values),
        p95_us=statistics.quantiles(values, n=100, method="inclusive")[94],
        min_us=min(values),
        max_us=max(values),
    )


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
result = {
    "protocol": "A/B/B/A independent processes; 30 profiled replays per batch; 800 MB FP32 ReduceSum + synchronize before each replay",
    "batches": {},
}
combined = {"baseline": [], "candidate": []}
for batch in ("a1", "b1", "b2", "a2"):
    directory = args.root / batch
    files = list(directory.rglob("kernel_details.csv"))
    assert len(files) == 1, (batch, files)
    csv_path = files[0]
    trace_path = next(directory.rglob("trace_view.json"))
    trace = json.loads(trace_path.read_text())
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    device = {
        (x["name"], Decimal(str(x["ts"]))): x
        for x in events
        if x.get("ph") == "X"
        and x.get("name") in (MAIN, FILL, SCRUB)
        and "Task Type" in x.get("args", {})
    }
    rows = list(csv.DictReader(csv_path.open()))
    assert all(row["Name"] in (MAIN, FILL, SCRUB) for row in rows)
    rows.sort(key=lambda x: Decimal(x["Start Time(us)"].strip()))
    assert len(rows) == 120, (batch, len(rows))
    # Host profiler Step IDs may assign the next asynchronous scrub to the
    # previous step. The synchronized device stream has an unambiguous
    # ReduceSum -> Fill -> Fill -> main sequence for each replay.
    groups = {str(i + 1): rows[i * 4 : (i + 1) * 4] for i in range(30)}
    meta = json.loads((directory / "metadata.json").read_text())
    assert len(groups) == meta["samples"] == 30, (batch, len(groups))
    samples = []
    for step, rows in sorted(groups.items(), key=lambda x: int(x[0])):
        rows.sort(key=lambda x: Decimal(x["Start Time(us)"].strip()))
        assert [x["Name"] for x in rows] == [SCRUB, FILL, FILL, MAIN], (
            batch,
            step,
            rows,
        )
        tasks = []
        for row in rows:
            start = Decimal(row["Start Time(us)"].strip())
            event = device[(row["Name"], start)]
            duration = float(row["Duration(us)"])
            assert abs(float(event["dur"]) - duration) < 1e-6
            tasks.append(
                dict(
                    name=row["Name"],
                    start_us=str(start),
                    end_us=str(start + Decimal(row["Duration(us)"])),
                    duration_us=duration,
                    device=row["Device_id"],
                    stream=event["tid"],
                    task=event["args"],
                )
            )
        assert Decimal(tasks[0]["end_us"]) <= Decimal(tasks[1]["start_us"])
        samples.append(
            dict(
                iteration=int(step),
                main_us=tasks[3]["duration_us"],
                fill_sum_us=sum(t["duration_us"] for t in tasks[1:3]),
                total_kernel_us=sum(t["duration_us"] for t in tasks[1:]),
                tasks=tasks,
            )
        )
    variant = "baseline" if batch.startswith("a") else "candidate"
    assert meta.get("dsl_version")
    combined[variant].extend(samples)
    result["batches"][batch] = dict(
        metadata=meta,
        csv=str(csv_path),
        trace=str(trace_path),
        main=stats([s["main_us"] for s in samples]),
        total_kernel=stats([s["total_kernel_us"] for s in samples]),
        samples=samples,
    )
result["summary"] = {
    v: {
        metric: stats([s[key] for s in samples])
        for metric, key in [
            ("main", "main_us"),
            ("fill_sum", "fill_sum_us"),
            ("total_kernel", "total_kernel_us"),
        ]
    }
    for v, samples in combined.items()
}
result["change_percent"] = {
    metric: {
        p: 100
        * (
            result["summary"]["candidate"][metric][p]
            / result["summary"]["baseline"][metric][p]
            - 1
        )
        for p in ("p50_us", "p95_us")
    }
    for metric in ("main", "total_kernel")
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            "summary": result["summary"],
            "change_percent": result["change_percent"],
            "batches": {
                k: {"main": v["main"], "total_kernel": v["total_kernel"]}
                for k, v in result["batches"].items()
            },
        },
        indent=2,
    )
)
