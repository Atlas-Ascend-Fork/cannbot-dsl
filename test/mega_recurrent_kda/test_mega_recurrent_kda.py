# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent precision test for the stream-controlled Mega KDA.

The independent CPU golden checks outputs, workspaces, and all state snapshots.
Run the parametrized NPU cases serially. Absolute and relative tolerances are 5e-3.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import pytest
import torch

from _samples_path import load_sample

SAMPLE = "mega_recurrent_kda/mega_recurrent_kda.py"
HEAD_DIM = 128
CONV_KERNEL_SIZE = 4
RMS_EPSILON = 1e-6
NZ_WEIGHT_ARGUMENTS = (1, 2, 3, 4, 5, 7)


@dataclasses.dataclass
class MegaRecurrentKDAWeights:
    """TP-local weights in the layouts consumed by the public operator."""

    qkv_projection_weight: torch.Tensor
    decay_projection_a_weight: torch.Tensor
    decay_projection_b_weight: torch.Tensor
    beta_projection_weight: torch.Tensor
    output_gate_projection_weight: torch.Tensor
    output_norm_weight: torch.Tensor
    output_projection_weight: torch.Tensor
    conv1d_weight: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor


@dataclasses.dataclass
class MegaRecurrentKDAGolden:
    """Outputs and intermediate values used by precision tests."""

    output: torch.Tensor
    raw_qkv: torch.Tensor
    convolved_qkv: torch.Tensor
    decay_a: torch.Tensor
    raw_decay: torch.Tensor
    raw_beta: torch.Tensor
    raw_output_gate: torch.Tensor
    recurrent_output: torch.Tensor
    gated_output: torch.Tensor
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor


class _RecurrentCase(NamedTuple):
    convolved_qkv: torch.Tensor
    raw_decay: torch.Tensor
    raw_beta: torch.Tensor
    recurrent_state: torch.Tensor
    ssm_state_indices: torch.Tensor
    num_accepted_tokens: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    scale: float
    lower_bound: float


class _GoldenCase(NamedTuple):
    hidden_states: torch.Tensor
    weights: MegaRecurrentKDAWeights
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor
    conv_state_indices: torch.Tensor
    ssm_state_indices: torch.Tensor
    conv_num_accepted_tokens: torch.Tensor
    num_accepted_tokens: torch.Tensor
    scale: float
    lower_bound: float
    rms_norm_eps: float


class _CpuArguments(NamedTuple):
    hidden_states: torch.Tensor
    qkv_projection_weight: torch.Tensor
    decay_projection_a_weight: torch.Tensor
    decay_projection_b_weight: torch.Tensor
    beta_projection_weight: torch.Tensor
    output_gate_projection_weight: torch.Tensor
    output_norm_weight: torch.Tensor
    output_projection_weight: torch.Tensor
    conv1d_weight: torch.Tensor
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor
    conv_state_indices: torch.Tensor
    ssm_state_indices: torch.Tensor
    conv_num_accepted_tokens: torch.Tensor
    num_accepted_tokens: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor


def _linear(value: torch.Tensor, weight: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    result = torch.matmul(value.float(), weight.float().transpose(-1, -2))
    return result.to(output_dtype)


def run_model_native_causal_conv1d_silu(raw_qkv: torch.Tensor, conv1d_weight: torch.Tensor, conv_state: torch.Tensor,
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


def _run_pooled_recurrent(case: _RecurrentCase) -> torch.Tensor:
    (convolved_qkv,
        raw_decay,
        raw_beta,
        recurrent_state,
        ssm_state_indices,
        num_accepted_tokens,
        a_log,
        dt_bias,
        scale,
        lower_bound,
    ) = case
    batch_size, seq_len, _, num_heads, head_dim = convolved_qkv.shape
    query = convolved_qkv[:, :, 0].float()
    key = convolved_qkv[:, :, 1].float()
    value = convolved_qkv[:, :, 2].float()
    query *= torch.rsqrt(query.square().sum(dim=-1, keepdim=True) + RMS_EPSILON)
    key *= torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + RMS_EPSILON)
    decay = float(lower_bound) * torch.sigmoid(
        torch.exp(a_log.float()).view(1, 1, num_heads, 1)
        * (raw_decay.float() + dt_bias.float().view(1, 1, num_heads, head_dim)))
    beta = torch.sigmoid(raw_beta.float())
    output = torch.empty(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32, device=convolved_qkv.device)
    state_indices = ssm_state_indices.reshape(batch_size, seq_len)
    for batch_index in range(batch_size):
        accepted = int(num_accepted_tokens[batch_index])
        initial_token = accepted - 1 if accepted > 0 else 0
        current = recurrent_state[int(state_indices[batch_index, initial_token])].float().clone()
        for token in range(seq_len):
            current *= torch.exp(decay[batch_index, token]).unsqueeze(-2)
            key_value = (current * key[batch_index, token].unsqueeze(-2)).sum(dim=-1)
            delta = (value[batch_index, token] - key_value) * beta[batch_index, token].unsqueeze(-1)
            current += delta.unsqueeze(-1) * key[batch_index, token].unsqueeze(-2)
            output[batch_index, token] = (current * query[batch_index, token].unsqueeze(-2)).sum(dim=-1) * float(scale)
            recurrent_state[int(state_indices[batch_index, token])].copy_(current.to(recurrent_state.dtype))
    return output


def run_mega_recurrent_kda_golden(case: _GoldenCase) -> MegaRecurrentKDAGolden:
    """Run the full fragment on CPU without calling production helpers."""
    (hidden_states,
        weights,
        conv_state,
        recurrent_state,
        conv_state_indices,
        ssm_state_indices,
        conv_num_accepted_tokens,
        num_accepted_tokens,
        scale,
        lower_bound,
        rms_norm_eps,
    ) = case
    assert hidden_states.device.type == "cpu", "golden must use independent CPU arithmetic"
    batch_size, seq_len, hidden_size = map(int, hidden_states.shape)
    num_heads = int(weights.beta_projection_weight.shape[0])
    projection_size = num_heads * HEAD_DIM
    raw_qkv = _linear(hidden_states, weights.qkv_projection_weight, torch.float32)
    convolved_flat = run_model_native_causal_conv1d_silu(raw_qkv, weights.conv1d_weight, conv_state,
        conv_state_indices, conv_num_accepted_tokens)
    convolved_qkv = convolved_flat.view(batch_size, seq_len, 3, num_heads, HEAD_DIM)
    decay_a = _linear(hidden_states, weights.decay_projection_a_weight, torch.bfloat16)
    raw_decay = _linear(decay_a, weights.decay_projection_b_weight, torch.bfloat16).view(
        batch_size, seq_len, num_heads, HEAD_DIM)
    raw_beta = _linear(hidden_states, weights.beta_projection_weight, torch.bfloat16).view(
        batch_size, seq_len, num_heads)
    raw_output_gate = _linear(hidden_states, weights.output_gate_projection_weight, torch.bfloat16).view(
        batch_size, seq_len, num_heads, HEAD_DIM)
    recurrent_output = _run_pooled_recurrent(
        _RecurrentCase(convolved_qkv, raw_decay, raw_beta, recurrent_state, ssm_state_indices, num_accepted_tokens,
            weights.a_log, weights.dt_bias, scale, lower_bound))
    inverse_rms = torch.rsqrt(recurrent_output.square().mean(dim=-1, keepdim=True) + float(rms_norm_eps))
    normalized = recurrent_output * inverse_rms * weights.output_norm_weight.float()
    gated_output = (normalized * torch.sigmoid(raw_output_gate.float())).to(torch.bfloat16)
    output = _linear(gated_output.view(batch_size, seq_len, projection_size), weights.output_projection_weight,
        torch.bfloat16).view(batch_size, seq_len, hidden_size)
    return MegaRecurrentKDAGolden(output=output, raw_qkv=raw_qkv, convolved_qkv=convolved_qkv, decay_a=decay_a,
        raw_decay=raw_decay, raw_beta=raw_beta, raw_output_gate=raw_output_gate, recurrent_output=recurrent_output,
        gated_output=gated_output, conv_state=conv_state, recurrent_state=recurrent_state)


def _make_cpu_arguments(batch_size, seq_len, seed, num_heads=6):
    generator = torch.Generator().manual_seed(seed)
    hidden_size, head_dim = (7168, 128)
    projection_size = num_heads * head_dim

    def randn(shape, dtype=torch.bfloat16, scale=0.02):
        return (torch.randn(shape, generator=generator) * scale).to(dtype)

    conv_indices = torch.randperm(batch_size, generator=generator).int() + 1
    conv_indices[::7] = 0
    state_indices = torch.randperm(batch_size * seq_len, generator=generator).int() + 1
    accepted = torch.arange(batch_size, dtype=torch.int32) % (seq_len + 1)
    return _CpuArguments(
        randn((batch_size, seq_len, hidden_size)),
        randn((3 * projection_size, hidden_size)),
        randn((head_dim, hidden_size)),
        randn((projection_size, head_dim)),
        randn((num_heads, hidden_size)),
        randn((projection_size, hidden_size)),
        randn((head_dim,), torch.float32, scale=0.2),
        randn((hidden_size, projection_size)),
        randn((4, 3 * projection_size)),
        randn((batch_size + 1, seq_len + 2, 3 * projection_size)),
        randn((batch_size * seq_len + 1, num_heads, head_dim, head_dim), torch.float32, scale=0.002),
        conv_indices,
        state_indices,
        accepted,
        (seq_len - accepted).contiguous(),
        randn((num_heads,), torch.float32, scale=0.1),
        randn((num_heads, head_dim), torch.float32, scale=0.1))


def _to_npu_arguments(arguments):
    torch_npu = pytest.importorskip("torch_npu")
    actual = [value.to("npu") for value in arguments]
    for index in NZ_WEIGHT_ARGUMENTS:
        actual[index] = torch_npu.npu_format_cast(actual[index], torch_npu.Format.FRACTAL_NZ)
        assert torch_npu.get_npu_format(actual[index]) == torch_npu.Format.FRACTAL_NZ
        assert getattr(actual[index], "_base") is None
        assert actual[index].storage_offset() == 0
    return tuple(actual)


def _assert_precision_results(module, actual, expected, output, initial):
    for value, truth in (
        (output, expected.output),
        (actual[9], expected.conv_state),
        (actual[10], expected.recurrent_state), ):
        torch.testing.assert_close(value.cpu().float(), truth.float(), atol=0.005, rtol=0.005)
    assert output.is_contiguous() and output.dtype == torch.bfloat16
    batch_size, seq_len = output.shape[:2]
    num_heads = actual[15].shape[0]
    launch = module._LAUNCHES[(num_heads, batch_size, actual[0].device)]
    assert len(launch.workspace_views) == 7
    assert launch.norm_exchange.shape == (128, 128)
    rows = batch_size * seq_len
    projection_size = num_heads * HEAD_DIM
    golden_views = (
        expected.gated_output.reshape(rows, projection_size),
        expected.convolved_qkv[:, :, 1].reshape(rows, projection_size),
        expected.convolved_qkv[:, :, 2].reshape(rows, projection_size),
        expected.raw_decay.reshape(rows, projection_size),
        expected.raw_output_gate.reshape(rows, projection_size), )
    for workspace, truth in zip(launch.workspace_views, golden_views):
        torch.testing.assert_close(workspace.cpu().float(), truth.float(), atol=0.005, rtol=0.005)
    torch.testing.assert_close(actual[9][0].cpu(), initial[0][0], atol=0, rtol=0)
    torch.testing.assert_close(actual[10][0].cpu(), initial[1][0], atol=0, rtol=0)


def _available_core_counts(module):
    device = torch.npu.current_device()
    hardware = torch.npu.get_device_properties(device)
    stream = torch.npu.current_stream(device)
    info = module.cannbotdsl.get_platform_info(stream=stream)
    cube = min(int(hardware.cube_core_num), int(info.cube_core_num))
    vector = min(int(hardware.vector_core_num), int(info.vector_core_num))
    return tuple(core for core in module.VALID_CORES if core <= cube and 2 * core <= vector)


def _check_precision(batch_size, seq_len, block_num, num_heads):
    pytest.importorskip("torch_npu")
    module = load_sample(SAMPLE)
    available = _available_core_counts(module)
    if not available:
        pytest.skip("Mega KDA requires at least 8 AIC and 16 AIV cores")
    quota = available[-1] if block_num is None else block_num
    if quota not in available:
        pytest.skip(f"AIC quota {quota} exceeds the device or current stream capacity")
    arguments = _make_cpu_arguments(batch_size, seq_len, 731 + batch_size + seq_len, num_heads)
    initial = (arguments[9].clone(), arguments[10].clone())
    weights = MegaRecurrentKDAWeights(*arguments[1:9], arguments[15], arguments[16])
    scale, lower = 128**-0.5, -5.0
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(original_threads, 8))
        expected = run_mega_recurrent_kda_golden(
            _GoldenCase(arguments[0], weights, initial[0].clone(), initial[1].clone(), *arguments[11:15], scale, lower,
                1e-5))
    finally:
        torch.set_num_threads(original_threads)
    actual = _to_npu_arguments(arguments)
    stream = torch.npu.Stream()
    torch.npu.synchronize()
    torch.npu.set_stream_limit(stream, cube_num=quota, vector_num=2 * quota)
    try:
        with torch.npu.stream(stream):
            for _ in range(2):
                actual[9].copy_(initial[0])
                actual[10].copy_(initial[1])
                y = module.mega_recurrent_kda(*actual, scale, lower, 1e-5)
                torch.npu.synchronize()
                _assert_precision_results(module, actual, expected, y, initial)
    finally:
        torch.npu.synchronize()
        torch.npu.reset_stream_limit(stream)


# Bound every device test by both physical cores and the current stream quota.
_PRECISION_CASES = [
    pytest.param(16, 8, 28, 6),
    pytest.param(16, 8, "max", 6),
    pytest.param(32, 8, "max", 6),
    pytest.param(16, 8, "auto", 6),
    pytest.param(32, 8, "auto", 6),
    *[pytest.param(batch, 8, 16, 6) for batch in (16, 32, 512)],
    *[pytest.param(batch, 8, "available", 6, marks=pytest.mark.slow) for batch in (16, 32, 48, 64, 256, 512)],
    pytest.param(4, 8, 8, 24, id="TP4-short"),
    pytest.param(12, 8, 24, 24, id="TP4-tail"),
    pytest.param(8, 8, 16, 12, id="TP8-short"),
    pytest.param(24, 8, "max", 12, id="TP8-tail"),
    pytest.param(32, 8, "max", 24, id="TP4-aligned-main"),
    pytest.param(64, 8, "max", 12, id="TP8-aligned-main"), ]


@pytest.mark.parametrize("seq_len", range(1, 8))
def test_rejects_sequence_lengths_outside_public_contract(seq_len):
    module = load_sample(SAMPLE)
    hidden = torch.empty((16, seq_len, 7168), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="S=8"):
        module.mega_recurrent_kda(hidden, *([None] * 16), 128**-0.5, -5.0, 1e-5)


@pytest.mark.npu
@pytest.mark.parametrize("batch_size,seq_len,cores,num_heads", _PRECISION_CASES)
def test_precision(batch_size, seq_len, cores, num_heads):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    module = load_sample(SAMPLE)
    grids = _available_core_counts(module)
    if not grids:
        pytest.skip("Mega KDA requires at least 8 AIC and 16 AIV cores")
    if cores == "available":
        controls = grids
    elif cores == "max":
        controls = (grids[-1],)
    else:
        controls = (None,) if cores == "auto" else (cores,)
    for block_num in controls:
        _check_precision(batch_size, seq_len, block_num, num_heads)


def test_public_signature():
    import inspect

    module = load_sample(SAMPLE)
    assert list(inspect.signature(module.mega_recurrent_kda).parameters) == [
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
        "conv_state_indices",
        "ssm_state_indices",
        "conv_num_accepted_tokens",
        "num_accepted_tokens",
        "A_log",
        "dt_bias",
        "scale",
        "lower_bound",
        "rms_norm_eps", ]
    schema = str(module.mega_recurrent_kda_op.default._schema)
    assert "core_num" not in schema
    assert "Tensor A_log" in schema
    assert "Tensor(a!) conv_state" in schema
    assert "Tensor(b!) recurrent_state" in schema
