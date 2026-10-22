# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Join msprof kernel timestamps with named device MSTX sample ranges.

Times are device task durations in microseconds, excluding Python execution and
compilation. Warmups are excluded. The case metric is the minimum sum of DSL
task durations in one measured repetition; per-kernel Task Duration minima and
PipeUtilization counters are written separately to perf_kernels.csv.
"""

import argparse
import bisect
import csv
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

FIELDS = (
    "op_name",
    "testcase_name",
    "sample_count",
    "min_kernel_time_us",
    "max_kernel_time_us",
    "kernel_count",
    "total_kernel_time_us",
    "mean_kernel_time_us",
    "status",
    "precision_status",
)
# DSL's generated AscendC entry points use kernel_ (including mangled symbols).
DSL_KERNEL = re.compile(r"(?:^|[^a-zA-Z])kernel_")


def number(value):
    result = Decimal(value.strip())
    if not result.is_finite() or result < 0:
        raise ValueError(f"invalid timing value: {value!r}")
    return result


def device_windows(profile_dir, intervals):
    """Read named ranges in the DEVICE clock, never host Start_time(us)."""
    expected = {}
    for index, sample in enumerate(intervals):
        for name in sample.get("device_markers", []):
            if name in expected:
                raise ValueError("duplicate performance sample marker: {}".format(name))
            expected[name] = index
    found = {}
    for path in sorted(profile_dir.rglob("msprof_tx*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                continue
            reader.fieldnames = [field.strip() for field in reader.fieldnames]
            for row in reader:
                name = (row.get("message") or "").strip()
                if name not in expected:
                    continue
                start_text = (row.get("Device Start_time(us)") or "").strip()
                end_text = (row.get("Device End_time(us)") or "").strip()
                if not start_text or not end_text:
                    continue
                start, end = number(start_text), number(end_text)
                if start == 0 or end <= start:
                    continue
                value = (row["Device_id"].strip(), start, end, expected[name])
                if name in found and found[name] != value:
                    raise ValueError(
                        "conflicting device ranges for marker: {}".format(name)
                    )
                found[name] = value
    windows = {}
    marker_counts = [0 for _ in intervals]
    for device, start, end, index in found.values():
        windows.setdefault(device, []).append((start, end, index))
        marker_counts[index] += 1
    for device, ranges in windows.items():
        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] < previous[1]:
                raise ValueError(
                    "overlapping device sample ranges on device {}".format(device)
                )
        windows[device] = ([item[0] for item in ranges], ranges)
    return windows, set(expected) - set(found), marker_counts, found


def summarize(profile_dir, cases_path, output_path):
    with cases_path.open(encoding="utf-8") as source:
        cases = [json.loads(line) for line in source if line.strip()]
    cases.sort(key=lambda case: case["start_ns"])
    intervals = []
    for case_index, case in enumerate(cases):
        for sample_index, sample in enumerate(
            case.get("samples", [dict(case, phase="measure")])
        ):
            intervals.append(
                dict(sample, case_index=case_index, sample_index=sample_index)
            )
    intervals.sort(key=lambda sample: sample["start_ns"])
    starts = [Decimal(sample["start_ns"]) / 1000 for sample in intervals]
    ends = [Decimal(sample["end_ns"]) / 1000 for sample in intervals]
    marked = any(case.get("association") == "device_mstx" for case in cases)
    if marked:
        windows, missing_markers, marker_counts, found_markers = device_windows(
            profile_dir, intervals
        )
    else:
        # Compatibility for archived reports created before device markers.
        windows, missing_markers, marker_counts, found_markers = (
            {},
            set(),
            [0 for _ in intervals],
            {},
        )
        for i, (start, end) in enumerate(zip(starts, ends)):
            if end < start or (i and start < ends[i - 1]):
                raise ValueError("case intervals overlap or clock moved backwards")
    times = [[] for _ in intervals]
    kernels = {}
    reports = sorted(profile_dir.rglob("op_summary*.csv"))
    if not reports:
        reports = sorted(profile_dir.rglob("task_time*.csv"))
    unmatched = 0
    unmatched_rows = []
    seen = set()
    for path in reports:
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                continue
            reader.fieldnames = [field.strip() for field in reader.fieldnames]
            fields = reader.fieldnames
            start_field = next(
                (f for f in ("Task Start Time(us)", "Start Time(us)") if f in fields),
                None,
            )
            duration_field = next(
                (f for f in ("Task Duration(us)", "Duration(us)") if f in fields), None
            )
            if "Op Name" not in fields or not start_field or not duration_field:
                raise ValueError(f"unsupported msprof timing columns in {path}")
            for row in reader:
                if not DSL_KERNEL.search(row["Op Name"] or ""):
                    continue
                start = number(row[start_field])
                duration = number(row[duration_field])
                # msprof may export the same task in multiple report locations.
                key = (
                    row.get("Device_id", row.get("Device ID", "")),
                    row.get("Stream ID", ""),
                    row.get("Task ID", ""),
                    row["Op Name"],
                    start,
                    duration,
                )
                if key in seen:
                    continue
                seen.add(key)
                index = -1
                if marked:
                    device_starts, ranges = windows.get(key[0], ([], []))
                    candidate = bisect.bisect_right(device_starts, start) - 1
                    if candidate >= 0 and start < ranges[candidate][1]:
                        index = ranges[candidate][2]
                else:
                    candidate = bisect.bisect_right(starts, start) - 1
                    if candidate >= 0 and start < ends[candidate]:
                        index = candidate
                if index < 0:
                    unmatched += 1
                    unmatched_rows.append(
                        {
                            "device_id": key[0],
                            "stream_id": key[1],
                            "task_id": key[2],
                            "kernel_name": row["Op Name"],
                            "task_start_us": str(start),
                            "task_duration_us": str(duration),
                        }
                    )
                    continue
                sample = intervals[index]
                times[index].append(duration)
                if sample["phase"] != "measure":
                    continue
                kernel_key = (sample["case_index"], key[0], row["Op Name"])
                entry = kernels.setdefault(
                    kernel_key, {"durations": [], "pipe": {}, "ratio": Decimal(-1)}
                )
                entry["durations"].append(duration)
                # As in the skill, retain the best-sampled row by mac ratio.
                pipe = {
                    name: value.strip()
                    for name, value in row.items()
                    if name and name.startswith(("aic_", "aiv_", "aicore_")) and value
                }
                ratio_text = pipe.get("aic_mac_ratio", pipe.get("aic_mac_ratio(%)", ""))
                try:
                    ratio = (
                        Decimal(ratio_text.rstrip("%")) if ratio_text else Decimal(-1)
                    )
                except InvalidOperation:
                    ratio = Decimal(-1)
                if ratio > entry["ratio"] or not entry["pipe"]:
                    entry.update(pipe=pipe, ratio=ratio)
    rows = []
    excluded_cases = set()
    for index, case in enumerate(cases):
        if case["status"] != "SKIPPED":
            continue
        if case.get("exclusion") not in (
            "aicpu_only",
            "precision_only",
            "graph_capture_only",
        ):
            continue
        if not any(sample["phase"] == "measure" for sample in case.get("samples", [])):
            excluded_cases.add(index)
    complete = (
        bool(cases)
        and (bool(reports) or len(excluded_cases) == len(cases))
        and unmatched == 0
        and not missing_markers
    )
    measurements = [[] for _ in cases]
    for sample, durations in zip(intervals, times):
        if sample["phase"] == "measure":
            measurements[sample["case_index"]].append((sample, durations))
    for case_index, case in enumerate(cases):
        measured = measurements[case_index]
        durations = [duration for _, values in measured for duration in values]
        totals = [sum(values, Decimal(0)) for _, values in measured if values]
        count = len(durations)
        status = case["status"] if count else "NO_DATA"
        valid = (
            count > 0
            and case["status"] == "PASSED"
            and len(measured) == case.get("expected_samples", 1)
            and all(
                values and sample["status"] == "PASSED" for sample, values in measured
            )
        )
        if case_index in excluded_cases and not count:
            status, valid = "SKIPPED", True
            print(
                "[PERF SKIP] {}/{}: {}".format(
                    case["op_name"],
                    case["testcase_name"],
                    case.get("skip_reason", "AICPU-only case"),
                )
            )
        if marked:
            valid = valid and all(
                sample.get("device_markers")
                and not missing_markers.intersection(sample["device_markers"])
                for sample in case.get("samples", [])
            )
        if count and status == "PASSED" and not valid:
            status = "INCOMPLETE"
        complete = complete and valid
        total = sum(durations, Decimal(0))
        rows.append(
            {
                "op_name": case["op_name"],
                "testcase_name": case["testcase_name"],
                "sample_count": len(totals),
                "min_kernel_time_us": f"{min(totals):.3f}" if totals else "",
                "max_kernel_time_us": f"{max(totals):.3f}" if totals else "",
                "kernel_count": count,
                "total_kernel_time_us": f"{total:.3f}" if count else "",
                "mean_kernel_time_us": f"{total / count:.3f}" if count else "",
                "status": status,
                "precision_status": case.get("precision_status", "NOT_RUN"),
            }
        )
        if not valid:
            print(
                "[ERROR] {}/{}: expected={}, recorded={}, matched={}".format(
                    case["op_name"],
                    case["testcase_name"],
                    case.get("expected_samples", 1),
                    len(measured),
                    len(totals),
                ),
                file=sys.stderr,
            )
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        "\n[PERF SUMMARY] DSL device time (us); warmups excluded; use Min(us) for comparison"
    )
    print(
        "Operator | Test case | Samples | Min(us) | Max(us) | Calls | Total(us) | "
        "Mean/task(us) | Status | Precision status"
    )
    for row in rows:
        print(" | ".join(str(row[field]) for field in FIELDS))
    print(f"[INFO] performance summary: {output_path}")
    pipe_fields = sorted(
        {field for entry in kernels.values() for field in entry["pipe"]}
    )
    kernel_fields = (
        "op_name",
        "testcase_name",
        "device_id",
        "kernel_name",
        "kernel_count",
        "min_task_time_us",
        "mean_task_time_us",
        "max_task_time_us",
        *pipe_fields,
    )
    kernel_path = output_path.with_name("perf_kernels.csv")
    with kernel_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=kernel_fields)
        writer.writeheader()
        print(
            "\n[KERNEL SUMMARY] Operator | Test case | Device | Kernel | Calls | Min(us) | Mean(us) | Max(us)"
        )
        for (case_index, device, name), entry in sorted(kernels.items()):
            values = entry["durations"]
            case = cases[case_index]
            row = dict(
                op_name=case["op_name"],
                testcase_name=case["testcase_name"],
                device_id=device,
                kernel_name=name,
                kernel_count=len(values),
                min_task_time_us=f"{min(values):.3f}",
                mean_task_time_us=f"{sum(values) / len(values):.3f}",
                max_task_time_us=f"{max(values):.3f}",
                **entry["pipe"],
            )
            writer.writerow(row)
            print(" | ".join(str(row[field]) for field in kernel_fields[:8]))
    print(f"[INFO] kernel summary: {kernel_path}")
    if marked:
        print(
            "[INFO] device MSTX association: markers={}/{}, matched DSL tasks={}, unmatched={}".format(
                len(found_markers),
                len(found_markers) + len(missing_markers),
                sum(len(values) for values in times),
                unmatched,
            )
        )
        range_fields = (
            "op_name",
            "testcase_name",
            "phase",
            "sample_index",
            "device_id",
            "marker_name",
            "device_start_us",
            "device_end_us",
        )
        with output_path.with_name("perf_ranges.csv").open(
            "w", encoding="utf-8", newline=""
        ) as output:
            writer = csv.DictWriter(output, fieldnames=range_fields)
            writer.writeheader()
            for name, (device, start, end, index) in sorted(found_markers.items()):
                sample = intervals[index]
                case = cases[sample["case_index"]]
                writer.writerow(
                    dict(
                        op_name=case["op_name"],
                        testcase_name=case["testcase_name"],
                        phase=sample["phase"],
                        sample_index=sample["sample_index"],
                        device_id=device,
                        marker_name=name,
                        device_start_us=str(start),
                        device_end_us=str(end),
                    )
                )
    sample_fields = (
        "op_name",
        "testcase_name",
        "phase",
        "sample_index",
        "status",
        "expected_markers",
        "exported_markers",
        "matched_kernels",
        "marker_names",
    )
    with output_path.with_name("perf_samples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=sample_fields)
        writer.writeheader()
        for index, (sample, durations) in enumerate(zip(intervals, times)):
            case = cases[sample["case_index"]]
            writer.writerow(
                dict(
                    op_name=case["op_name"],
                    testcase_name=case["testcase_name"],
                    phase=sample["phase"],
                    sample_index=sample["sample_index"],
                    status=sample["status"],
                    expected_markers=len(sample.get("device_markers", [])),
                    exported_markers=marker_counts[index],
                    matched_kernels=len(durations),
                    marker_names=";".join(sample.get("device_markers", [])),
                )
            )
    unmatched_fields = (
        "device_id",
        "stream_id",
        "task_id",
        "kernel_name",
        "task_start_us",
        "task_duration_us",
    )
    with output_path.with_name("perf_unmatched.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=unmatched_fields)
        writer.writeheader()
        writer.writerows(unmatched_rows)
    if missing_markers:
        print(
            "[ERROR] {} device MSTX sample markers were not exported".format(
                len(missing_markers)
            ),
            file=sys.stderr,
        )
    if unmatched:
        print(
            f"[ERROR] {unmatched} DSL tasks cannot be matched to a test case",
            file=sys.stderr,
        )
    if not complete:
        print(
            "[ERROR] performance summary incomplete or contains failed cases",
            file=sys.stderr,
        )
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        return summarize(args.profile_dir, args.cases, args.output)
    except (OSError, ValueError, KeyError, InvalidOperation) as error:
        print(f"[ERROR] cannot summarize performance: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
