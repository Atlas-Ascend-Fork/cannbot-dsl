# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared pytest configuration and markers for CANNBotDSL tests."""

import json
import os
import sys
from pathlib import Path

import pytest

_samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "samples")
_samples_dir = os.path.abspath(_samples_dir)
for _category in ("attention", "matmul", "vector", "mc2"):
    _category_dir = os.path.join(_samples_dir, _category)
    if _category_dir not in sys.path:
        sys.path.insert(0, _category_dir)


def _ci_exclusions():
    # run_tests.sh sets these for CI precision and performance respectively.
    # Manual diagnostics keep the original tests available without these flags.
    if not (
        os.environ.get("CI_CASE_RESULTS") or os.environ.get("CI_PERF_ENTRY_REPORT")
    ):
        return ()
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "catalog/operators.json").read_text(encoding="utf-8"))
    excluded = []
    for rule in catalog["ci"]["rules"]:
        reason = rule.get("skip_reason")
        if rule["enabled"] or not reason:
            continue
        for reference in rule["tests"]:
            if reference.startswith("@"):
                operator, field, *suffix = reference[1:].split("/")
                target = root / catalog["operators"][operator][field] / Path(*suffix)
            else:
                target = root / reference
            target = target.resolve()
            target.relative_to(root / "test")
            excluded.append((rule["name"], target, reason))
    return tuple(excluded)


_CI_EXCLUSIONS = _ci_exclusions()


class _TemporarilySkippedModule(pytest.Module):
    def collect(self):
        # Skip before import, including missing SDK communication modules.
        pytest.skip(self._ci_skip_reason, allow_module_level=True)


def pytest_pycollect_makemodule(module_path, parent):
    path = module_path.resolve()
    for _, target, reason in _CI_EXCLUSIONS:
        if path == target or target in path.parents:
            module = _TemporarilySkippedModule.from_parent(parent, path=module_path)
            module._ci_skip_reason = reason
            return module


def pytest_report_header(config):
    return [
        f"Temporary CI exclusion {name}: {reason}" for name, _, reason in _CI_EXCLUSIONS
    ]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "cannir_install: needs external CANNIR command-line tools, which are not distributed in the wheel.",
    )
    config.addinivalue_line(
        "markers",
        "ascendc_toolchain: exercises AscendC translation or compilation.",
    )
    config.addinivalue_line(
        "markers",
        "npu: needs Ascend NPU device / runtime.",
    )
    config.addinivalue_line(
        "markers",
        "slow: heavier/broader sweep (e.g. full shape-pool precision); deselect with -m 'not slow'.",
    )
    # PyTorch deprecates torch.jit.script_method; torch_npu (and similar) still hit torch.jit._script.
    # Not actionable in CANNBotDSL; remove when upstream stops emitting this.
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*torch\\.jit\\.script_method.*:DeprecationWarning:torch.jit._script",
    )


@pytest.fixture
def dump_ascendc(monkeypatch, tmp_path):
    monkeypatch.setenv("CANNBOTDSL_DUMP_ASCENDC", "1")


@pytest.fixture
def dump_mlir(monkeypatch, tmp_path):
    monkeypatch.setenv("CANNBOTDSL_DUMP_MLIR", "1")


@pytest.fixture
def auto_sync(monkeypatch, tmp_path):
    monkeypatch.setenv("CANNBOTDSL_AUTO_SYNC", "1")


@pytest.fixture
def auto_bufid_sync(monkeypatch, tmp_path):
    monkeypatch.setenv("CANNBOTDSL_AUTO_BUFID_SYNC", "1")


@pytest.fixture
def auto_intrablock_sync(monkeypatch, tmp_path):
    monkeypatch.setenv("CANNBOTDSL_AUTO_INTRABLOCKSYNC", "1")


@pytest.fixture
def pipe_stage_transform(monkeypatch):
    monkeypatch.setenv("CANNBOTDSL_PIPE_STAGE", "transform")


@pytest.fixture
def pipe_stage_translate(monkeypatch):
    monkeypatch.setenv("CANNBOTDSL_PIPE_STAGE", "translate")


@pytest.fixture
def pipe_stage_compile(monkeypatch):
    monkeypatch.setenv("CANNBOTDSL_PIPE_STAGE", "compile")
