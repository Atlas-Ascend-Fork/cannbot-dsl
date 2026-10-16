# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent CPU-golden precision tests for ReplaySSM."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from _samples_path import load_sample


SAMPLE = "mega_recurrent_kda_replayssm/mega_recurrent_kda_replayssm.py"
COMMIT_SAMPLE = "commit_recurrent_kda_replayssm/commit_recurrent_kda_replayssm.py"
NZ_WEIGHT_ARGUMENTS = (1, 2, 3, 4, 5, 7)

_INPUT_NAMES = (
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
    "recurrent_state",
    "replay_u",
    "replay_k",
    "replay_decay",
    "conv_state_indices",
    "conv_num_accepted_tokens",
    "a_log",
    "dt_bias",
)


def _module():
    return load_sample(SAMPLE)


@lru_cache(maxsize=1)
def _cpu_golden_module():
    """Reuse CPU projection/Conv arithmetic without importing a second NPU op."""
    path = Path(__file__).parents[1] / "mega_recurrent_kda" / "test_mega_recurrent_kda.py"
    spec = importlib.util.spec_from_file_location("_mega_kda_cpu_golden", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_cpu_arguments(batch=16, seed=20260921):
    generator = torch.Generator().manual_seed(seed)
    sequence, hidden, heads, dim = 8, 7168, 6, 128
    projection = heads * dim

    def randn(shape, dtype=torch.bfloat16, scale=0.02):
        return (torch.randn(shape, generator=generator) * scale).to(dtype)

    conv_indices = torch.randperm(batch, generator=generator).int() + 1
    conv_indices[::7] = 0
    conv_accepted = torch.arange(batch, dtype=torch.int32) % (sequence + 1)
    return SimpleNamespace(
        hidden_states=randn((batch, sequence, hidden)),
        qkv_projection_weight=randn((3 * projection, hidden)),
        decay_projection_a_weight=randn((dim, hidden)),
        decay_projection_b_weight=randn((projection, dim)),
        beta_projection_weight=randn((heads, hidden)),
        output_gate_projection_weight=randn((projection, hidden)),
        output_norm_weight=randn((dim,), torch.float32, scale=0.2),
        output_projection_weight=randn((hidden, projection)),
        conv1d_weight=randn((4, 3 * projection)),
        conv_state=randn((batch + 1, sequence + 2, 3 * projection)),
        recurrent_state=randn((batch, heads, dim, dim), torch.float32, scale=0.002),
        replay_u=torch.empty(batch, sequence, heads, dim, dtype=torch.float16),
        replay_k=torch.empty(batch, sequence, heads, dim, dtype=torch.float16),
        replay_decay=torch.empty(batch, sequence, heads, dim, dtype=torch.float32),
        conv_state_indices=conv_indices,
        conv_num_accepted_tokens=conv_accepted,
        a_log=randn((heads,), torch.float32, scale=0.1),
        dt_bias=randn((heads, dim), torch.float32, scale=0.1),
    )


def _project_replay_inputs(arguments):
    """Prepare projections, convolution and activated recurrence inputs."""
    cpu = _cpu_golden_module()
    hidden = arguments.hidden_states
    qkv_w = arguments.qkv_projection_weight
    fa_w = arguments.decay_projection_a_weight
    fb_w = arguments.decay_projection_b_weight
    beta_w = arguments.beta_projection_weight
    gate_w = arguments.output_gate_projection_weight
    conv_w = arguments.conv1d_weight
    batch, sequence, _ = hidden.shape
    conv_state = arguments.conv_state.clone()
    raw_qkv = cpu.linear_reference(hidden, qkv_w, torch.float32)
    convolved = cpu.run_model_native_causal_conv1d_silu(
        raw_qkv, conv_w, conv_state, arguments.conv_state_indices, arguments.conv_num_accepted_tokens
    ).view(batch, sequence, 3, 6, 128)
    query, key, value = (convolved[:, :, index].float() for index in range(3))
    query *= torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
    key *= torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    raw_decay = cpu.linear_reference(cpu.linear_reference(hidden, fa_w, torch.bfloat16), fb_w, torch.bfloat16).view(
        batch, sequence, 6, 128
    )
    decay = torch.exp(
        -5.0
        * torch.sigmoid(
            torch.exp(arguments.a_log).view(1, 1, 6, 1) * (raw_decay.float() + arguments.dt_bias.view(1, 1, 6, 128))
        )
    )
    beta = torch.sigmoid(cpu.linear_reference(hidden, beta_w, torch.bfloat16).float())
    gate = cpu.linear_reference(hidden, gate_w, torch.bfloat16).view(batch, sequence, 6, 128)
    return SimpleNamespace(
        query=query,
        key=key,
        value=value,
        decay=decay,
        beta=beta,
        gate=gate,
        conv_state=conv_state,
    )


def _run_replay_cpu_golden(arguments, *, snapshots=False):
    """Evaluate FP32 recurrence and format records at the public dtype boundaries."""
    projected = _project_replay_inputs(arguments)
    batch, sequence, _ = arguments.hidden_states.shape
    query, key, value = projected.query, projected.key, projected.value
    decay, beta, gate = projected.decay, projected.beta, projected.gate
    current = arguments.recurrent_state.clone()
    replay_u = torch.empty(batch, sequence, 6, 128, dtype=torch.float16)
    recurrent = torch.empty(batch, sequence, 6, 128)
    states = []
    for token in range(sequence):
        current *= decay[:, token].unsqueeze(-2)
        delta = (value[:, token] - (current * key[:, token].unsqueeze(-2)).sum(-1)) * beta[:, token].unsqueeze(-1)
        replay_u[:, token] = delta.to(torch.float16)
        current += delta.unsqueeze(-1) * key[:, token].unsqueeze(-2)
        recurrent[:, token] = (current * query[:, token].unsqueeze(-2)).sum(-1) * 128**-0.5
        if snapshots:
            states.append(current.clone())
    normalized = (
        recurrent * torch.rsqrt(recurrent.square().mean(-1, keepdim=True) + 1e-5) * arguments.output_norm_weight.float()
    )
    gated = (normalized * torch.sigmoid(gate.float())).to(torch.bfloat16)
    output = _cpu_golden_module().linear_reference(gated.flatten(2), arguments.output_projection_weight, torch.bfloat16)
    return SimpleNamespace(
        output=output,
        conv_state=projected.conv_state,
        replay_u=replay_u,
        replay_k=key.to(torch.float16),
        replay_decay=decay,
        snapshots=states,
    )


def _reconstruct_commit_cpu(checkpoint, replay_u, replay_k, replay_decay, accepted):
    current = checkpoint.clone()
    for batch, count in enumerate(accepted):
        for token in range(int(count)):
            current[batch] *= replay_decay[batch, token].float().unsqueeze(-2)
            current[batch] += replay_u[batch, token].float().unsqueeze(-1) * replay_k[batch, token].float().unsqueeze(
                -2
            )
    return current


def _to_npu_arguments(arguments):
    torch_npu = pytest.importorskip("torch_npu")
    actual = [getattr(arguments, name).to("npu") for name in _INPUT_NAMES]
    for index in NZ_WEIGHT_ARGUMENTS:
        actual[index] = torch_npu.npu_format_cast(actual[index], torch_npu.Format.FRACTAL_NZ)
    return tuple(actual)


def _allocate_replay(batch_size, *, fill_nan=False):
    factory = torch.full if fill_nan else torch.empty
    extra = (float("nan"),) if fill_nan else ()
    replay_u = factory((batch_size, 8, 6, 128), *extra, dtype=torch.float16, device="npu")
    replay_k = factory((batch_size, 8, 6, 128), *extra, dtype=torch.float16, device="npu")
    replay_decay = factory((batch_size, 8, 6, 128), *extra, dtype=torch.float32, device="npu")
    return replay_u, replay_k, replay_decay


def _run_verify(module, arguments, cores, monkeypatch):
    replay_u, replay_k, replay_decay = _allocate_replay(len(arguments[0]), fill_nan=True)
    actual = (*arguments[:11], replay_u, replay_k, replay_decay, *arguments[14:])
    original_empty = torch.empty
    batch = len(arguments[0])

    def poisoned_output(*args, **kwargs):
        value = original_empty(*args, **kwargs)
        if (
            value.device.type == "npu"
            and value.dtype == torch.bfloat16
            and tuple(value.shape) in ((batch * 8, 7168), (batch, 8, 7168))
        ):
            value.fill_(float("nan"))
        return value

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", poisoned_output)
        if cores is not None:
            patch.setattr(module, "_device_block_num", lambda _ref: cores)
        output = module.mega_recurrent_kda_replayssm(*actual, 128**-0.5, -5.0, 1e-5)
    return output, replay_u, replay_k, replay_decay


_REPLAY_PRECISION_CASES = [
    pytest.param(16, "auto"),
    *[pytest.param(batch, 16) for batch in (16, 32, 512)],
    *[pytest.param(batch, "available", marks=pytest.mark.slow) for batch in (16, 512)],
]


@pytest.mark.npu
@pytest.mark.parametrize("batch,cores", _REPLAY_PRECISION_CASES)
def test_npu_precision_batch_generalization_and_core_control(batch, cores, monkeypatch):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    module = _module()
    stream = torch.npu.current_stream(torch.npu.current_device())
    available = int(module.cannbotdsl.get_platform_info(stream=stream).cube_core_num)
    grids = tuple(count for count in (8, 16, 24, 28, 32) if count <= available)
    if not grids:
        pytest.skip("Mega KDA requires at least 8 AIC cores in the current stream")
    controls = (None,) if cores == "auto" else grids if cores == "available" else (cores,)
    if cores == 16 and available < 16:
        pytest.skip("requires a stream quota of at least 16 AIC cores")
    cpu_arguments = _make_cpu_arguments(batch, seed=20260921 + batch)
    initial_conv = cpu_arguments.conv_state.clone()
    expected = _run_replay_cpu_golden(cpu_arguments)
    used_slots = set(cpu_arguments.conv_state_indices.tolist()) - {0}
    unchanged = [slot for slot in range(batch + 1) if slot not in used_slots]
    arguments = _to_npu_arguments(cpu_arguments)
    for control in controls:
        actual_arguments = list(arguments)
        actual_arguments[9] = arguments[9].clone()
        actual_arguments[10] = arguments[10].clone()
        output, replay_u, replay_k, replay_decay = _run_verify(module, actual_arguments, control, monkeypatch)
        torch.npu.synchronize()
        for name, actual in (
            ("output", output),
            ("replay_u", replay_u),
            ("replay_k", replay_k),
            ("replay_decay", replay_decay),
            ("conv_state", actual_arguments[9]),
        ):
            torch.testing.assert_close(
                actual.cpu().float(),
                getattr(expected, name).float(),
                atol=0.005,
                rtol=0.005,
                msg=lambda message, name=name: f"{name}: {message}",
            )
        torch.testing.assert_close(actual_arguments[9].cpu()[unchanged], initial_conv[unchanged], atol=0, rtol=0)
        torch.testing.assert_close(actual_arguments[10].cpu(), cpu_arguments.recurrent_state, atol=0, rtol=0)


@pytest.mark.npu
def test_npu_verify_records_commit_to_cpu_snapshots(monkeypatch):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    verify = _module()
    commit = load_sample(COMMIT_SAMPLE)
    cpu_arguments = _make_cpu_arguments(seed=20260922)
    expected = _run_replay_cpu_golden(cpu_arguments, snapshots=True)
    arguments = _to_npu_arguments(cpu_arguments)
    output, replay_u, replay_k, replay_decay = _run_verify(verify, arguments, None, monkeypatch)
    torch.npu.synchronize()
    accepted_cpu = torch.arange(16, dtype=torch.int32) % 9
    expected_state = _reconstruct_commit_cpu(
        cpu_arguments.recurrent_state, expected.replay_u, expected.replay_k, expected.replay_decay, accepted_cpu
    )
    # Quantized records reconstruct every accepted FP32 proposal snapshot.
    for batch, accepted in enumerate(accepted_cpu.tolist()):
        reference = cpu_arguments.recurrent_state[batch] if accepted == 0 else expected.snapshots[accepted - 1][batch]
        torch.testing.assert_close(expected_state[batch], reference, atol=0.005, rtol=0.005)
    state = arguments[10].clone()
    actual_cpu_state = _reconstruct_commit_cpu(
        cpu_arguments.recurrent_state, replay_u.cpu(), replay_k.cpu(), replay_decay.cpu(), accepted_cpu
    )
    commit.commit_recurrent_kda_replayssm(state, replay_u, replay_k, replay_decay, accepted_cpu.npu())
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu().float(), expected.output.float(), atol=0.005, rtol=0.005)
    torch.testing.assert_close(state.cpu(), expected_state, atol=0.005, rtol=0.005)
    torch.testing.assert_close(state.cpu(), actual_cpu_state, atol=0.005, rtol=0.005)
    torch.testing.assert_close(
        state.cpu()[accepted_cpu == 0], cpu_arguments.recurrent_state[accepted_cpu == 0], atol=0, rtol=0
    )
