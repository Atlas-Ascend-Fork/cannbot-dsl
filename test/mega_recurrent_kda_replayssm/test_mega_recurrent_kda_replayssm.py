# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent CPU golden arithmetic used only by ReplaySSM NPU whitebox tests."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from _samples_path import load_sample

SAMPLE = "mega_recurrent_kda_replayssm/mega_recurrent_kda_replayssm.py"
COMMIT_SAMPLE = "commit_recurrent_kda_replayssm/commit_recurrent_kda_replayssm.py"
NZ_WEIGHT_ARGUMENTS = (1, 2, 3, 4, 5, 7)
CONV_KERNEL_SIZE = 4


def _module():
    return load_sample(SAMPLE)


def _linear_reference(value: torch.Tensor, weight: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    result = torch.matmul(value.float(), weight.float().transpose(-1, -2))
    return result.to(output_dtype)


def _run_model_native_causal_conv1d_silu(raw_qkv: torch.Tensor, conv1d_weight: torch.Tensor, conv_state: torch.Tensor,
    conv_state_indices: torch.Tensor, conv_num_accepted_tokens: torch.Tensor) -> torch.Tensor:
    """Run pooled K=4 Conv with the raw-FP32/BF16-history boundary.

    ``conv_state_indices == 0`` denotes the immutable null slot. All proposal
    tokens are evaluated; the accepted count chooses the three history rows
    from the model-native speculative cache.
    """
    if raw_qkv.ndim != 3:
        raise ValueError("raw_qkv must be [B,S,3*P]")
    batch_size, seq_len, channels = map(int, raw_qkv.shape)
    if tuple(conv1d_weight.shape) != (CONV_KERNEL_SIZE, channels):
        raise ValueError("conv1d_weight must be [4,3*P]")
    if conv_state.ndim != 3 or int(conv_state.shape[2]) != channels:
        raise ValueError("conv_state must be [pool,state_length,3*P]")
    if int(conv_state.shape[1]) < seq_len + CONV_KERNEL_SIZE - 2:
        raise ValueError("conv_state state_length must be at least S+2")
    if tuple(conv_state_indices.shape) != (batch_size,):
        raise ValueError("conv_state_indices must be [B]")
    if tuple(conv_num_accepted_tokens.shape) != (batch_size,):
        raise ValueError("conv_num_accepted_tokens must be [B]")
    result = torch.zeros_like(raw_qkv, dtype=torch.bfloat16)
    original_state = conv_state.clone()
    filter_fp32 = conv1d_weight.float()
    for batch_index in range(batch_size):
        cache_index = int(conv_state_indices[batch_index])
        if cache_index == 0:
            continue
        accepted = int(conv_num_accepted_tokens[batch_index])
        offset = max(accepted - 1, 0)
        history_stop = offset + CONV_KERNEL_SIZE - 1
        history = original_state[cache_index, offset:history_stop].float().clone()
        if int(history.shape[0]) != CONV_KERNEL_SIZE - 1:
            raise ValueError("accepted Conv history exceeds state_length")
        for token in range(seq_len):
            current = raw_qkv[batch_index, token].float()
            window = torch.cat((history, current.unsqueeze(0)), dim=0)
            products = window * filter_fp32
            convolved = products[0] + products[1] + (products[2] + products[3])
            result[batch_index, token] = torch.nn.functional.silu(convolved).to(torch.bfloat16)
            history = torch.cat((history[1:], current.to(torch.bfloat16).float().unsqueeze(0)), dim=0)
        selected = original_state[cache_index, offset:history_stop]
        conv_state[cache_index, :2].copy_(selected[1:])
        proposal_stop = 2 + seq_len
        conv_state[cache_index, 2:proposal_stop].copy_(raw_qkv[batch_index])
    return result


def _make_cpu_arguments(batch=16, seed=20260921, heads=6):
    generator = torch.Generator().manual_seed(seed)
    sequence, hidden, dim = 8, 7168, 128
    projection = heads * dim

    def randn(shape, dtype=torch.bfloat16, scale=0.02):
        return (torch.randn(shape, generator=generator) * scale).to(dtype)

    conv_indices = torch.randperm(batch, generator=generator).int() + 1
    conv_indices[::7] = 0
    conv_accepted = torch.arange(batch, dtype=torch.int32) % (sequence + 1)
    return (
        randn((batch, sequence, hidden)),
        randn((3 * projection, hidden)),
        randn((dim, hidden)),
        randn((projection, dim)),
        randn((heads, hidden)),
        randn((projection, hidden)),
        randn((dim,), torch.float32, scale=0.2),
        randn((hidden, projection)),
        randn((4, 3 * projection)),
        randn((batch + 1, sequence + 2, 3 * projection)),
        randn((batch, heads, dim, dim), torch.float32, scale=0.002),
        torch.empty(batch, sequence, heads, dim, dtype=torch.float16),
        torch.empty(batch, sequence, heads, dim, dtype=torch.float16),
        torch.empty(batch, sequence, heads, dim, dtype=torch.float32),
        conv_indices,
        conv_accepted,
        randn((heads,), torch.float32, scale=0.1),
        randn((heads, dim), torch.float32, scale=0.1),)


def _run_replay_cpu_golden(arguments, *, snapshots=False, scale=128**-0.5, lower_bound=-5.0, rms_norm_eps=1e-5):
    """Evaluate FP32 recurrence and format records at the public dtype boundaries."""
    hidden, qkv_w, fa_w, fb_w, beta_w, gate_w, norm_w, output_w, conv_w = arguments[:9]
    batch, sequence, _ = hidden.shape
    heads = int(beta_w.shape[0])
    conv_state = arguments[9].clone()
    raw_qkv = _linear_reference(hidden, qkv_w, torch.float32)
    convolved = _run_model_native_causal_conv1d_silu(
        raw_qkv, conv_w, conv_state, arguments[14], arguments[15]
    ).view(batch, sequence, 3, heads, 128)
    query, key, value = (convolved[:, :, index].float() for index in range(3))
    query *= torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
    key *= torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    raw_decay = _linear_reference(_linear_reference(hidden, fa_w, torch.bfloat16), fb_w, torch.bfloat16).view(batch,
        sequence, heads, 128)
    decay = torch.exp(lower_bound * torch.sigmoid(
        torch.exp(arguments[16]).view(1, 1, heads, 1) * (raw_decay.float() + arguments[17].view(1, 1, heads, 128))))
    beta = torch.sigmoid(_linear_reference(hidden, beta_w, torch.bfloat16).float())
    gate = _linear_reference(hidden, gate_w, torch.bfloat16).view(batch, sequence, heads, 128)
    current = arguments[10].clone()
    replay_u = torch.empty(batch, sequence, heads, 128, dtype=torch.float16)
    recurrent = torch.empty(batch, sequence, heads, 128)
    states = []
    for token in range(sequence):
        current *= decay[:, token].unsqueeze(-2)
        delta = (value[:, token] - (current * key[:, token].unsqueeze(-2)).sum(-1)) * beta[:, token].unsqueeze(-1)
        replay_u[:, token] = delta.to(torch.float16)
        current += delta.unsqueeze(-1) * key[:, token].unsqueeze(-2)
        recurrent[:, token] = (current * query[:, token].unsqueeze(-2)).sum(-1) * scale
        if snapshots:
            states.append(current.clone())
    normalized = recurrent * torch.rsqrt(recurrent.square().mean(-1, keepdim=True) + rms_norm_eps) * norm_w.float()
    gated = (normalized * torch.sigmoid(gate.float())).to(torch.bfloat16)
    output = _linear_reference(gated.flatten(2), output_w, torch.bfloat16)
    return SimpleNamespace(output=output, conv_state=conv_state, replay_u=replay_u,
                           replay_k=key.to(torch.float16), replay_decay=decay, snapshots=states)


def _reconstruct_commit_cpu(checkpoint, replay_u, replay_k, replay_decay, accepted):
    current = checkpoint.clone()
    for batch, count in enumerate(accepted):
        for token in range(int(count)):
            current[batch] *= replay_decay[batch, token].float().unsqueeze(-2)
            current[batch] += replay_u[batch, token].float().unsqueeze(-1) * replay_k[batch,
                token].float().unsqueeze(-2)
    return current


def _to_npu_arguments(arguments):
    torch_npu = pytest.importorskip("torch_npu")
    actual = [value.to("npu") for value in arguments]
    for index in NZ_WEIGHT_ARGUMENTS:
        actual[index] = torch_npu.npu_format_cast(
            actual[index], torch_npu.Format.FRACTAL_NZ)
        torch.npu.synchronize()
        assert torch_npu.get_npu_format(actual[index]) == torch_npu.Format.FRACTAL_NZ
        assert actual[index]._base is None and actual[index].storage_offset() == 0
    return tuple(actual)


def _allocate_replay(batch_size, heads=6, *, fill_nan=False):
    factory = torch.full if fill_nan else torch.empty
    extra = (float("nan"),) if fill_nan else ()
    replay_u = factory(
        (batch_size, 8, heads, 128), *extra, dtype=torch.float16, device="npu")
    replay_k = factory(
        (batch_size, 8, heads, 128), *extra, dtype=torch.float16, device="npu")
    replay_decay = factory(
        (batch_size, 8, heads, 128), *extra, dtype=torch.float32, device="npu")
    return replay_u, replay_k, replay_decay



def _available_core_counts(module):
    hardware = torch.npu.get_device_properties(torch.npu.current_device())
    info = module.cannbotdsl.get_platform_info(stream=torch.npu.current_stream())
    cube = min(int(hardware.cube_core_num), int(info.cube_core_num))
    vector = min(int(hardware.vector_core_num), int(info.vector_core_num))
    return tuple(core for core in module.VALID_CORES if core <= cube and 2 * core <= vector)


@contextmanager
def _core_stream(module, cores):
    available = _available_core_counts(module)
    if not available:
        pytest.skip("requires at least 8 AIC and 16 AIV cores")
    requested = max(available) if cores in (None, "max") else cores
    if requested not in available:
        pytest.skip(f"quota {requested} exceeds available AIC/AIV resources {available}")
    stream = torch.npu.Stream()
    torch.npu.set_stream_limit(stream, cube_num=requested, vector_num=2 * requested)
    try:
        with torch.npu.stream(stream):
            yield requested
            torch.npu.synchronize()
    finally:
        torch.npu.synchronize()
        torch.npu.reset_stream_limit(stream)


def _run_verify(module, arguments, cores, monkeypatch):
    original_empty = torch.empty
    shape = (len(arguments[0]) * 8, 7168)

    def poisoned_output(*args, **kwargs):
        value = original_empty(*args, **kwargs)
        if value.device.type == "npu" and value.dtype == torch.bfloat16 and tuple(value.shape) == shape:
            value.fill_(float("nan"))
        return value

    with _core_stream(module, cores), monkeypatch.context() as patch:
        patch.setattr(torch, "empty", poisoned_output)
        replay = _allocate_replay(len(arguments[0]), int(arguments[4].shape[0]), fill_nan=True)
        actual = (*arguments[:11], *replay, *arguments[14:])
        output = module.mega_recurrent_kda_replayssm(*actual, 128**-0.5, -5.0, 1e-5)
        return output, *replay


@pytest.mark.npu
@pytest.mark.parametrize("tp", (4, 8, 16))
@pytest.mark.parametrize("model_batch", (1, 3, 8, 16, 32))
def test_npu_precision_batch_generalization_and_core_control(tp, model_batch, monkeypatch):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    module = _module()
    cpu = _make_cpu_arguments(tp * model_batch, seed=20260921 + model_batch, heads=96 // tp)
    expected = _run_replay_cpu_golden(cpu)
    arguments = _to_npu_arguments(cpu)
    controls = _available_core_counts(module) if model_batch == 3 else ("max",)
    if not controls:
        pytest.skip("requires at least 8 AIC and 16 AIV cores")
    used = set(cpu[14].tolist()) - {0}
    unchanged = [slot for slot in range(len(cpu[0]) + 1) if slot not in used]
    for core in controls:
        actual = list(arguments)
        actual[9] = arguments[9].clone()
        actual[10] = arguments[10].clone()
        output, replay_u, replay_k, replay_decay = _run_verify(module, actual, core, monkeypatch)
        for name, value in (("output", output), ("replay_u", replay_u), ("replay_k", replay_k),
            ("replay_decay", replay_decay), ("conv_state", actual[9])):
            torch.testing.assert_close(value.cpu().float(), getattr(expected, name).float(),
                atol=0.005, rtol=0.005, msg=lambda text, name=name: f"{name}: {text}")
        torch.testing.assert_close(actual[9].cpu()[unchanged], cpu[9][unchanged], atol=0, rtol=0)
        torch.testing.assert_close(actual[10].cpu(), cpu[10], atol=0, rtol=0)
    assert len(module._COMPILED_FNS) == 1


@pytest.mark.npu
@pytest.mark.parametrize("tp", (4, 8, 16))
def test_npu_verify_records_commit_to_cpu_snapshots(tp, monkeypatch):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    verify = _module()
    commit = load_sample(COMMIT_SAMPLE)
    batch_size = tp * 3
    cpu = _make_cpu_arguments(batch_size, seed=20260922, heads=96 // tp)
    expected = _run_replay_cpu_golden(cpu, snapshots=True)
    arguments = _to_npu_arguments(cpu)
    output, replay_u, replay_k, replay_decay = _run_verify(verify, arguments, "max", monkeypatch)
    accepted = torch.arange(batch_size, dtype=torch.int32) % 9
    expected_state = _reconstruct_commit_cpu(cpu[10], expected.replay_u, expected.replay_k,
        expected.replay_decay, accepted)
    for batch, count in enumerate(accepted.tolist()):
        snapshot = cpu[10][batch] if count == 0 else expected.snapshots[count - 1][batch]
        torch.testing.assert_close(expected_state[batch], snapshot, atol=0.005, rtol=0.005)
    state = arguments[10].clone()
    replay_state = _reconstruct_commit_cpu(cpu[10], replay_u.cpu(), replay_k.cpu(), replay_decay.cpu(), accepted)
    with _core_stream(verify, "max"):
        commit.commit_recurrent_kda_replayssm(state, replay_u, replay_k, replay_decay, accepted.npu())
    torch.testing.assert_close(output.cpu().float(), expected.output.float(), atol=0.005, rtol=0.005)
    torch.testing.assert_close(state.cpu(), expected_state, atol=0.005, rtol=0.005)
    torch.testing.assert_close(state.cpu(), replay_state, atol=0.005, rtol=0.005)
    torch.testing.assert_close(state.cpu()[accepted == 0], cpu[10][accepted == 0], atol=0, rtol=0)


@pytest.mark.npu
@pytest.mark.parametrize("tp", (4, 8, 16))
def test_npu_runtime_scalars_and_strong_checkpoint(tp):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    module = _module()
    cpu = list(_make_cpu_arguments(tp * 2, seed=20261020, heads=96 // tp))
    cpu[10].mul_(10.0)
    scalars = (0.125, 0.0, 0.002)
    expected = _run_replay_cpu_golden(cpu, snapshots=True, scale=scalars[0], lower_bound=scalars[1],
        rms_norm_eps=scalars[2])
    actual = _to_npu_arguments(cpu)

    def graph_call(*values):
        return module.mega_recurrent_kda_replayssm(*values[:16], A_log=values[16], dt_bias=values[17],
            scale=scalars[0], lower_bound=scalars[1], rms_norm_eps=scalars[2])

    compiled = torch.compile(graph_call, backend="eager", fullgraph=True)
    with _core_stream(module, "max"):
        output = compiled(*actual)
    for name, value in (("output", output), ("conv_state", actual[9]), ("replay_u", actual[11]),
        ("replay_k", actual[12]), ("replay_decay", actual[13])):
        torch.testing.assert_close(value.cpu().float(), getattr(expected, name).float(), atol=0.005, rtol=0.005)
    torch.testing.assert_close(actual[10].cpu(), cpu[10], atol=0, rtol=0)
    commit = load_sample(COMMIT_SAMPLE)
    accepted = torch.full((len(cpu[0]),), 8, dtype=torch.int32)
    expected_state = _reconstruct_commit_cpu(cpu[10], expected.replay_u, expected.replay_k,
        expected.replay_decay, accepted)
    state = actual[10].clone()
    with _core_stream(module, "max"):
        commit.commit_recurrent_kda_replayssm(state, actual[11], actual[12], actual[13], accepted.npu())
    torch.testing.assert_close(state.cpu(), expected_state, atol=0.005, rtol=0.005)
    torch.testing.assert_close(state.cpu(), expected.snapshots[-1], atol=0.005, rtol=0.005)
    assert len(module._COMPILED_FNS) == 1
