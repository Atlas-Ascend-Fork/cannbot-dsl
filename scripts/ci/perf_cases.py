# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Profile passed AICore/Mix cases using their original fixtures and inputs.

Repeat the first safe compiled entry, then stop before precision assertions.
Native device MSTX ranges identify samples; host timestamps are diagnostic only.
"""

import dataclasses
from contextlib import contextmanager
import ctypes
import hashlib
import inspect
import json
import time
from pathlib import Path, PurePosixPath

import pytest

_PERF_STATE = pytest.StashKey[dict]()


@dataclasses.dataclass(frozen=True)
class _KernelCall:
    program: object
    args: tuple
    kwargs: dict
    devices: list


class _PerfComplete(BaseException):
    """End input preparation without executing subsequent precision checks."""


@contextmanager
def profiling_setup(item, label):
    """Only profiler preparation errors qualify; never wrap the kernel/checks."""
    try:
        yield
    except Exception as error:
        item.stash.setdefault(_PERF_STATE, {})["error"] = "{}: {}".format(label, error)
        raise


def pytest_addoption(parser):
    parser.addoption("--ci-perf-cases", help="JSONL file for performance samples")
    parser.addoption("--ci-perf-repeat", type=int, default=10)
    parser.addoption("--ci-perf-warmup", type=int, default=1)
    parser.addoption(
        "--ci-perf-entry-report",
        help="DSL entries observed during original NPU precision tests",
    )


def pytest_configure(config):
    path = config.getoption("--ci-perf-cases")
    if not path:
        return
    if config.getoption("--ci-perf-repeat") < 2:
        raise pytest.UsageError("--ci-perf-repeat must be >= 2")
    if config.getoption("--ci-perf-warmup") < 0:
        raise pytest.UsageError("--ci-perf-warmup must be >= 0")
    if getattr(config.option, "numprocesses", 0):
        raise pytest.UsageError("CI performance collection requires serial tests")
    Path(path).write_text("", encoding="utf-8")
    Path(path).with_suffix(".session.json").unlink(missing_ok=True)


def _synchronize():
    import torch_npu

    for device in range(torch_npu.npu.device_count()):
        torch_npu.npu.synchronize(device)


def _is_capturing():
    import torch_npu

    # Do not silently treat an unavailable/failed query as a safe sampling point.
    return torch_npu.npu.is_current_stream_capturing()


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    if not config.getoption("--ci-perf-cases"):
        return
    observed_path = config.getoption("--ci-perf-entry-report")
    if not observed_path:
        raise pytest.UsageError(
            "performance requires an original precision entry report"
        )
    observed = json.loads(Path(observed_path).read_text(encoding="utf-8"))["cases"]
    for item in items:
        row = observed.get(item.nodeid)
        if row is None or "aicore_mix" not in row.get("entry_kinds", []):
            raise pytest.UsageError("ineligible performance case: " + item.nodeid)


class _DeviceMarkers:
    """Native MSTX works with CLI msprof without a framework profiler session."""

    def __init__(self):
        self.library = ctypes.CDLL("libms_tools_ext.so")
        self.library.mstxRangeStartA.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        self.library.mstxRangeStartA.restype = ctypes.c_uint64
        self.library.mstxRangeEnd.argtypes = [ctypes.c_uint64]
        self.library.mstxRangeEnd.restype = None

    def start(self, label, devices):
        import torch_npu

        ranges = []
        try:
            for device in devices:
                with torch_npu.npu.device(device):
                    stream = torch_npu.npu.current_stream(device)
                    name = "{}:d{}".format(label, device)
                    identifier = self.library.mstxRangeStartA(
                        name.encode(), ctypes.c_void_p(stream.npu_stream)
                    )
                    if not identifier:
                        raise RuntimeError(
                            "MSTX device range failed; msprof requires --msproftx=on"
                        )
                    ranges.append({"name": name, "id": identifier})
        except BaseException:
            self.end(ranges)
            raise
        return ranges

    def end(self, ranges):
        for item in ranges:
            self.library.mstxRangeEnd(item["id"])


def _is_nz(value):
    if value.device.type not in ("npu", "privateuseone"):
        return False
    import torch_npu

    return torch_npu.get_npu_format(value) == torch_npu.Format.FRACTAL_NZ


def _snapshot_tensor(value):
    if _is_nz(value):
        import torch_npu

        # Generic clone/copy_ cannot handle NZ tensors on some devices.
        # Copy physical storage without changing the kernel's input layout.
        saved = torch_npu.empty_with_format(
            value.shape,
            dtype=value.dtype,
            device=value.device,
            acl_format=torch_npu.get_npu_format(value),
        )
        torch_npu.copy_memory_(saved, value.detach())
        return saved
    return value.detach().clone()


def _restore_tensor(value, saved):
    if _is_nz(value):
        import torch_npu

        torch_npu.copy_memory_(value, saved)
    else:
        value.copy_(saved)


def _snapshots(args, kwargs):
    """Preserve input/state buffers, including nested tensor lists and aliases."""
    import torch

    visited = set()
    snapshots = []

    def visit(value):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, torch.Tensor):
            snapshots.append((value, _snapshot_tensor(value)))
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                visit(getattr(value, field.name))

    visit(args)
    visit(kwargs)
    return snapshots


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if not pyfuncitem.config.getoption("--ci-perf-cases"):
        return None
    state = pyfuncitem.stash.setdefault(_PERF_STATE, {})
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        state["error"] = "profiling does not support asynchronous tests"
        pytest.fail("CI profiling requires a synchronous DSL input preparation test")
    from cannbotdsl.core.compiler.function import ProviderCallable

    original = ProviderCallable.__call__
    warmup = pyfuncitem.config.getoption("--ci-perf-warmup")
    repeat = pyfuncitem.config.getoption("--ci-perf-repeat")
    with profiling_setup(pyfuncitem, "profiling markers unavailable"):
        markers = _DeviceMarkers()
    case_key = hashlib.sha256(pyfuncitem.nodeid.encode()).hexdigest()[:20]
    sampled = False
    captured = False
    samples = []
    state["samples"] = samples
    excluded = state.get("exclusion")

    def invoke(kernel_call, phase, index):
        _synchronize()
        start_ns = time.time_ns()
        status = "FAILED"
        ranges = markers.start(
            "cann-ci-perf:{}:{}:{}".format(case_key, phase, index), kernel_call.devices
        )
        try:
            _synchronize()
            result = original(
                kernel_call.program, *kernel_call.args, **kernel_call.kwargs
            )
            _synchronize()
            status = "PASSED"
            return result
        finally:
            markers.end(ranges)
            _synchronize()
            samples.append(
                {
                    "phase": phase,
                    "start_ns": start_ns,
                    "end_ns": time.time_ns(),
                    "status": status,
                    "device_markers": [item["name"] for item in ranges],
                }
            )

    def profile(program, *args, **kwargs):
        import torch_npu

        nonlocal sampled, captured
        with profiling_setup(pyfuncitem, "profiling capture query unsupported"):
            capturing = _is_capturing()
        if capturing:
            captured = True
            # Capture must see the original launch, with no synchronization,
            # tensor snapshots/restores, repeat loop, or markers inserted here.
            return original(program, *args, **kwargs)
        if excluded:
            return original(program, *args, **kwargs)
        if sampled:
            return original(program, *args, **kwargs)
        sampled = True
        with profiling_setup(pyfuncitem, "profiling input snapshot unsupported"):
            snapshots = _snapshots(args, kwargs)
        devices = sorted(
            {
                value.device.index
                for value, _ in snapshots
                if value.device.type in ("npu", "privateuseone")
            }
        ) or [torch_npu.npu.current_device()]
        kernel_call = _KernelCall(program, args, kwargs, devices)
        for index in range(warmup + repeat):
            if index:
                import torch

                with profiling_setup(pyfuncitem, "profiling input restore unsupported"):
                    with torch.no_grad():
                        for value, saved in snapshots:
                            _restore_tensor(value, saved)
            phase = "warmup" if index < warmup else "measure"
            invoke(kernel_call, phase, index)
        raise _PerfComplete()

    ProviderCallable.__call__ = profile
    try:
        arguments = {
            name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames
        }
        try:
            pyfuncitem.obj(**arguments)
        except _PerfComplete:
            return True
        if not sampled and captured:
            state["exclusion"] = "graph_capture_only"
            state["skip_reason"] = (
                "DSL calls occur only during graph capture; no safe profiling launch"
            )
            return True
        state["error"] = (
            "No compiled DSL entry was reached; performance sample unavailable"
        )
        pytest.fail(state["error"])
    finally:
        ProviderCallable.__call__ = original


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    path = item.config.getoption("--ci-perf-cases")
    if not path:
        return
    state = item.stash.setdefault(_PERF_STATE, {})
    if not report.passed:
        state["outcome"] = report.outcome.upper()
    if report.when != "teardown":
        return
    samples = state.get("samples", [])
    source, testcase = item.nodeid.split("::", 1)
    operator = PurePosixPath(source).parent.relative_to("test").as_posix()
    outcome = state.get("outcome", "PASSED")
    excluded = state.get("exclusion")
    record = {
        "op_name": operator,
        "testcase_name": testcase,
        "nodeid": item.nodeid,
        "start_ns": samples[0]["start_ns"] if samples else time.time_ns(),
        "end_ns": samples[-1]["end_ns"] if samples else time.time_ns(),
        "status": "SKIPPED" if excluded and outcome == "PASSED" else outcome,
        "precision_status": "NOT_RUN",
        "expected_samples": 0
        if excluded
        else item.config.getoption("--ci-perf-repeat"),
        "exclusion": excluded,
        "skip_reason": state.get("skip_reason", ""),
        "samples": samples,
        "association": "device_mstx",
    }
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    path = session.config.getoption("--ci-perf-cases")
    if path:
        Path(path).with_suffix(".session.json").write_text(
            json.dumps({"pytest_exitstatus": int(session.exitstatus)}), encoding="utf-8"
        )
