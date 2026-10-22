# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Record exact pytest coverage; optionally observe original DSL launches.

The observer forwards calls unchanged. It neither rewrites tests nor compiles,
repeats or suppresses kernels. Performance selection uses passed original tests.
"""

from functools import wraps
import json
import logging
from pathlib import Path
import sys

import pytest

_CASE_RESULTS = pytest.StashKey[dict]()
_CASE_ERRORS = pytest.StashKey[list]()
_OBSERVED_ENTRIES = pytest.StashKey[set]()
_LOGGER = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption("--ci-case-results")
    parser.addoption("--ci-observe-dsl", action="store_true")
    parser.addoption("--ci-perf-selection")


def pytest_configure(config):
    if config.getoption("--ci-case-results"):
        config.stash[_CASE_RESULTS] = {}
        config.stash[_CASE_ERRORS] = []


def pytest_collection_modifyitems(config, items):
    selection = config.getoption("--ci-perf-selection")
    if selection:
        wanted = set(
            json.loads(Path(selection).read_text(encoding="utf-8"))["selected"]
        )
        found = {item.nodeid for item in items}
        if wanted - found:
            raise pytest.UsageError(
                "performance cases missing: " + str(sorted(wanted - found))
            )
        removed = [item for item in items if item.nodeid not in wanted]
        items[:] = [item for item in items if item.nodeid in wanted]
        config.hook.pytest_deselected(items=removed)
    if not config.getoption("--ci-case-results"):
        return
    for item in items:
        item.user_properties.append(("ci_nodeid", item.nodeid))


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    patches = []
    if item.config.getoption("--ci-observe-dsl") and item.get_closest_marker("npu"):
        from cannbotdsl.aicpu import toolchain
        from cannbotdsl.core.compiler import function, jit_executor

        kinds = set()
        item.stash[_OBSERVED_ENTRIES] = kinds

        def observe(original, kind):
            @wraps(original)
            def call(*args, **kwargs):
                observed = kind
                if kind == "aicore_mix":
                    import torch_npu

                    if torch_npu.npu.is_current_stream_capturing():
                        observed = "graph_capture"
                kinds.add(observed)
                return original(*args, **kwargs)

            return call

        for owner, name, kind in (
            (function.ProviderCallable, "__call__", "aicore_mix"),
            (jit_executor, "_execute_bound_call", "aicore_mix"),
            (toolchain.CompiledAicpuKernel, "launch", "aicpu"),
            (toolchain.CompiledAicpuKernel, "run_x86", "aicpu"),
        ):
            original = getattr(owner, name)
            patches.append((owner, name, original))
            setattr(owner, name, observe(original, kind))
    try:
        yield
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if not item.config.getoption("--ci-case-results"):
        return
    report = outcome.get_result()
    row = item.config.stash[_CASE_RESULTS].setdefault(
        item.nodeid,
        {"phases": {}, "entry_kinds": [], "npu": bool(item.get_closest_marker("npu"))},
    )
    row["phases"][report.when] = report.outcome.upper()
    row["entry_kinds"] = sorted(item.stash.get(_OBSERVED_ENTRIES, ()))
    if report.failed:
        row["failure_reason"] = str(report.longrepr)[-4000:]
    if report.skipped:
        row["skip_reason"] = str(report.longrepr)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    config = node.config
    if not config.getoption("--ci-case-results"):
        return
    data = getattr(node, "workeroutput", {}).get("ci_case_results")
    if error or data is None:
        config.stash[_CASE_ERRORS].append(
            "{}: test worker failed: {}".format(
                node.gateway.id, error or "missing case report"
            )
        )
        return
    for nodeid, row in data["cases"].items():
        if nodeid in config.stash[_CASE_RESULTS]:
            config.stash[_CASE_ERRORS].append("duplicate case report: " + nodeid)
        else:
            config.stash[_CASE_RESULTS][nodeid] = row
    config.stash[_CASE_ERRORS].extend(data["errors"])


def pytest_sessionfinish(session, exitstatus):
    path = session.config.getoption("--ci-case-results")
    if path:
        data = {
            "pytest_exitstatus": int(exitstatus),
            "cases": session.config.stash[_CASE_RESULTS],
            "errors": session.config.stash[_CASE_ERRORS],
        }
        if hasattr(session.config, "workerinput"):
            session.config.workeroutput["ci_case_results"] = data
            return
        if data["errors"]:
            session.exitstatus = data["pytest_exitstatus"] = 1
        result = Path(path)
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )


def read_results(path, expected=None):
    """Fail closed on incomplete execution, even if the process returned zero."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("errors"):
        raise RuntimeError("; ".join(data["errors"]))
    if expected is not None and set(data["cases"]) != {
        case["nodeid"] for case in expected
    }:
        raise RuntimeError("executed cases differ from planned selection: " + str(path))
    for row in data["cases"].values():
        phases = row["phases"]
        if "teardown" not in phases or not {"setup", "teardown"} <= phases.keys():
            raise RuntimeError("incomplete pytest lifecycle: " + str(path))
        if phases["setup"] == "PASSED" and "call" not in phases:
            raise RuntimeError("test call was not executed: " + str(path))
        row["status"] = (
            "FAILED"
            if "FAILED" in phases.values()
            else "SKIPPED"
            if "SKIPPED" in phases.values()
            else "PASSED"
        )
    return data


def select_performance(path, output):
    data = read_results(path)
    selected, excluded = [], []
    for nodeid, row in data["cases"].items():
        if not row["npu"]:
            continue
        if row["status"] == "PASSED" and "aicore_mix" in row["entry_kinds"]:
            selected.append(nodeid)
        else:
            reason = (
                row["status"]
                if row["status"] != "PASSED"
                else "no safe AICore/Mix entry (AICPU-only, capture-only, or no DSL entry)"
            )
            excluded.append({"nodeid": nodeid, "reason": reason})
            _LOGGER.info("[PERF SKIP] %s: %s", nodeid, reason)
    Path(output).write_text(
        json.dumps({"selected": sorted(selected), "excluded": excluded}, indent=2),
        encoding="utf-8",
    )
    _LOGGER.info("[PERF] eligible=%s, excluded=%s", len(selected), len(excluded))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser()
    parser.add_argument("--select-perf", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    select_performance(args.select_perf, args.output)
