# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Independent CPU precision and parameterized PA/TH contract tests.

Run this file with pytest after enabling the required NPU runtime.
Use -m functional / -m interception; Commit also retains -m companion.
Logs/provenance are collected by the external validation command, not test helpers.
"""

from __future__ import annotations
import fcntl
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture(scope="session")
def npu_session(request):
    # Register torch.npu when backend autoload is disabled.
    import torch_npu  # noqa: F401

    if getattr(request.config, "_kda_device0_lock", None) is None:
        path = Path("/var/lock/cannbot-ci/npu-test.lock.device-0")
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            torch.npu.set_device(0)
            torch.npu.init()
            assert torch.npu.is_available()
        except BaseException:
            handle.close()
            raise
        request.config._kda_device0_lock = handle
        request.addfinalizer(lambda: (torch.npu.synchronize(), handle.close()))


@pytest.fixture(autouse=True)
def require_real_npu(request):
    if request.node.get_closest_marker("npu"):
        request.getfixturevalue("npu_session")


NAMES = [
    "hidden_states",
    "qkv_projection_weight",
    "decay_projection_a_weight",
    "decay_projection_b_weight",
    "beta_projection_weight",
    "output_gate_projection_weight",
    "output_norm_weight",
    "output_projection_weight",
    "conv1d_weight",
    "conv_state",
    "checkpoint_state",
    "replay_u",
    "replay_k",
    "replay_decay",
    "state_indices",
    "conv_num_accepted_tokens",
    "A_log",
    "dt_bias",
]
WEIGHTS = (1, 2, 3, 4, 5, 7)


@torch.no_grad()
def _commit_reference_impl(
    checkpoint, u, k, decay, accepted, state_indices, *, fp64=True
):
    assert all(
        t.device.type == "cpu"
        for t in (checkpoint, u, k, decay, accepted, state_indices)
    ), "CPU reference must never execute on NPU"
    dtype = torch.float64 if fp64 else torch.float32
    result = checkpoint.clone()
    for batch, slot in enumerate(state_indices.tolist()):
        current = checkpoint[slot].to(dtype)
        for token in range(int(accepted[batch])):
            current = current * decay[slot, :, token].to(dtype).unsqueeze(-2) + u[
                slot, :, token
            ].to(dtype).unsqueeze(-1) * k[slot, :, token].to(dtype).unsqueeze(-2)
        result[slot] = current.to(torch.float32)
    return result


def _commit_reference(*args, **kwargs):
    with torch.device("cpu"):
        return _commit_reference_impl(*args, **kwargs)


# Both pytest and TTK execute only cases present in the canonical design CSV.
import os

_case_home = Path(__file__).resolve().parents[1] / "mega_recurrent_kda"
sys.path.insert(0, str(_case_home))
import kda_ttk_ops as adapter
from assets import inputs as designed_cases


@pytest.fixture(autouse=True)
def external_evidence(tmp_path_factory, monkeypatch):
    if not os.environ.get("MKDA_RUN_DIR"):
        monkeypatch.setenv(
            "MKDA_RUN_DIR", str(tmp_path_factory.getbasetemp() / "mega_kda_evidence")
        )


_selected = [c for c in designed_cases.select("pytest_smoke") if c["mode"] == "Commit"]
if os.environ.get("MEGA_KDA_SUITE"):
    wanted = {c["case_id"] for c in designed_cases.select(os.environ["MEGA_KDA_SUITE"])}
    _selected = [c for c in _selected if c["case_id"] in wanted]
_parameters = [
    pytest.param(
        c["case_id"],
        id=c["case_id"],
        marks=[
            pytest.mark.npu,
            pytest.mark.interception
            if c["expected"] == "REJECT_ATOMIC"
            else pytest.mark.functional,
            *([pytest.mark.companion] if "companion" in c["coverage"] else []),
        ],
    )
    for c in _selected
]


@pytest.mark.parametrize("case_id", _parameters)
def test_designed_case(case_id):
    torch.set_num_threads(4)
    adapter.run_pytest_case(
        case_id, reference_verify=None, reference_commit=_commit_reference
    )
