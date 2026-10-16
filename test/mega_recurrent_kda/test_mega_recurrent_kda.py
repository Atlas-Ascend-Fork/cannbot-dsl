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


def linear_reference(
    value: torch.Tensor, weight: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor:
    result = torch.matmul(value.float(), weight.float().transpose(-1, -2))
    return result.to(output_dtype)


def run_model_native_causal_conv1d_silu(
    raw_qkv: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv_state: torch.Tensor,
    conv_state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
) -> torch.Tensor:
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
            result[batch_index, token] = torch.nn.functional.silu(convolved).to(
                torch.bfloat16
            )
            history = torch.cat(
                (history[1:], current.to(torch.bfloat16).float().unsqueeze(0)), dim=0
            )
        selected = original_state[cache_index, offset:history_stop]
        conv_state[cache_index, :2].copy_(selected[1:])
        proposal_stop = 2 + seq_len
        conv_state[cache_index, 2:proposal_stop].copy_(raw_qkv[batch_index])
    return result


def _run_pooled_recurrent(case: _RecurrentCase) -> torch.Tensor:
    (
        convolved_qkv,
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
        * (raw_decay.float() + dt_bias.float().view(1, 1, num_heads, head_dim))
    )
    beta = torch.sigmoid(raw_beta.float())
    output = torch.empty(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        dtype=torch.float32,
        device=convolved_qkv.device,
    )
    state_indices = ssm_state_indices.reshape(batch_size, seq_len)
    for batch_index in range(batch_size):
        accepted = int(num_accepted_tokens[batch_index])
        initial_token = accepted - 1 if accepted > 0 else 0
        current = (
            recurrent_state[int(state_indices[batch_index, initial_token])]
            .float()
            .clone()
        )
        for token in range(seq_len):
            current *= torch.exp(decay[batch_index, token]).unsqueeze(-2)
            key_value = (current * key[batch_index, token].unsqueeze(-2)).sum(dim=-1)
            delta = (value[batch_index, token] - key_value) * beta[
                batch_index, token
            ].unsqueeze(-1)
            current += delta.unsqueeze(-1) * key[batch_index, token].unsqueeze(-2)
            output[batch_index, token] = (
                current * query[batch_index, token].unsqueeze(-2)
            ).sum(dim=-1) * float(scale)
            recurrent_state[int(state_indices[batch_index, token])].copy_(
                current.to(recurrent_state.dtype)
            )
    return output


def run_mega_recurrent_kda_golden(case: _GoldenCase) -> MegaRecurrentKDAGolden:
    """Run the full fragment on CPU without calling production helpers."""
    (
        hidden_states,
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
    assert hidden_states.device.type == "cpu", (
        "golden must use independent CPU arithmetic"
    )
    batch_size, seq_len, hidden_size = map(int, hidden_states.shape)
    num_heads = int(weights.beta_projection_weight.shape[0])
    projection_size = num_heads * HEAD_DIM
    raw_qkv = linear_reference(hidden_states, weights.qkv_projection_weight, torch.float32)
    convolved_flat = run_model_native_causal_conv1d_silu(
        raw_qkv,
        weights.conv1d_weight,
        conv_state,
        conv_state_indices,
        conv_num_accepted_tokens,
    )
    convolved_qkv = convolved_flat.view(batch_size, seq_len, 3, num_heads, HEAD_DIM)
    decay_a = linear_reference(hidden_states, weights.decay_projection_a_weight, torch.bfloat16)
    raw_decay = linear_reference(
        decay_a, weights.decay_projection_b_weight, torch.bfloat16
    ).view(batch_size, seq_len, num_heads, HEAD_DIM)
    raw_beta = linear_reference(
        hidden_states, weights.beta_projection_weight, torch.bfloat16
    ).view(batch_size, seq_len, num_heads)
    raw_output_gate = linear_reference(
        hidden_states, weights.output_gate_projection_weight, torch.bfloat16
    ).view(batch_size, seq_len, num_heads, HEAD_DIM)
    recurrent_output = _run_pooled_recurrent(
        _RecurrentCase(
            convolved_qkv,
            raw_decay,
            raw_beta,
            recurrent_state,
            ssm_state_indices,
            num_accepted_tokens,
            weights.a_log,
            weights.dt_bias,
            scale,
            lower_bound,
        )
    )
    inverse_rms = torch.rsqrt(
        recurrent_output.square().mean(dim=-1, keepdim=True) + float(rms_norm_eps)
    )
    normalized = recurrent_output * inverse_rms * weights.output_norm_weight.float()
    gated_output = (normalized * torch.sigmoid(raw_output_gate.float())).to(
        torch.bfloat16
    )
    output = linear_reference(
        gated_output.view(batch_size, seq_len, projection_size),
        weights.output_projection_weight,
        torch.bfloat16,
    ).view(batch_size, seq_len, hidden_size)
    return MegaRecurrentKDAGolden(
        output=output,
        raw_qkv=raw_qkv,
        convolved_qkv=convolved_qkv,
        decay_a=decay_a,
        raw_decay=raw_decay,
        raw_beta=raw_beta,
        raw_output_gate=raw_output_gate,
        recurrent_output=recurrent_output,
        gated_output=gated_output,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
    )


def _make_cpu_arguments(batch_size, seq_len, seed):
    generator = torch.Generator().manual_seed(seed)
    hidden_size, num_heads, head_dim = (7168, 6, 128)
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
        randn(
            (batch_size * seq_len + 1, num_heads, head_dim, head_dim),
            torch.float32,
            scale=0.002,
        ),
        conv_indices,
        state_indices,
        accepted,
        (seq_len - accepted).contiguous(),
        randn((num_heads,), torch.float32, scale=0.1),
        randn((num_heads, head_dim), torch.float32, scale=0.1),
    )


def _to_npu_arguments(arguments):
    torch_npu = pytest.importorskip("torch_npu")
    actual = [value.to("npu") for value in arguments]
    for index in NZ_WEIGHT_ARGUMENTS:
        actual[index] = torch_npu.npu_format_cast(
            actual[index], torch_npu.Format.FRACTAL_NZ
        )
        assert torch_npu.get_npu_format(actual[index]) == torch_npu.Format.FRACTAL_NZ
        assert getattr(actual[index], "_base") is None
        assert actual[index].storage_offset() == 0
    return tuple(actual)


def _check_precision(
    batch_size, seq_len, block_num, monkeypatch, *, strong_state=False
):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    module = load_sample(SAMPLE)
    arguments = _make_cpu_arguments(batch_size, seq_len, 731 + batch_size + seq_len)
    if strong_state:
        arguments[10].mul_(10.0)
    actual_arguments = _to_npu_arguments(arguments)
    initial_conv_state = arguments[9].clone()
    initial_null_snapshot = arguments[10][0].clone()
    weights = MegaRecurrentKDAWeights(
        qkv_projection_weight=arguments[1],
        decay_projection_a_weight=arguments[2],
        decay_projection_b_weight=arguments[3],
        beta_projection_weight=arguments[4],
        output_gate_projection_weight=arguments[5],
        output_norm_weight=arguments[6],
        output_projection_weight=arguments[7],
        conv1d_weight=arguments[8],
        a_log=arguments[15],
        dt_bias=arguments[16],
    )
    scale, lower_bound = (128 ** (-0.5), 0.0 if strong_state else -5.0)
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(original_threads, 8))
        expected = run_mega_recurrent_kda_golden(
            _GoldenCase(
                arguments[0],
                weights,
                arguments[9],
                arguments[10],
                *arguments[11:15],
                scale,
                lower_bound,
                1e-5,
            )
        )
    finally:
        torch.set_num_threads(original_threads)
    if strong_state:
        assert expected.output.abs().max().item() > 0.05
        written = expected.recurrent_state[arguments[12].long()]
        assert written.abs().flatten(1).amax(1).min().item() > 0.01
    original_empty = torch.empty

    def poisoned_output(*args, **kwargs):
        value = original_empty(*args, **kwargs)
        if (
            value.device.type == "npu"
            and value.dtype == torch.bfloat16
            and (
                tuple(value.shape)
                in ((batch_size * seq_len, 7168), (batch_size, seq_len, 7168))
            )
        ):
            value.fill_(float("nan"))
        return value

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", poisoned_output)
        if block_num is not None:
            patch.setattr(module, "_device_block_num", lambda _ref: block_num)
        actual = module.mega_recurrent_kda(*actual_arguments, scale, lower_bound, 1e-5)
    torch.npu.synchronize()
    assert actual.dtype == torch.bfloat16
    assert tuple(actual.shape) == (batch_size, seq_len, 7168)
    assert actual.is_contiguous()
    launch = getattr(module, "_COMPILED_LAUNCH")
    workspace_views = launch.workspace_views
    assert len(workspace_views) == 7
    rows, projection_size = (batch_size * seq_len, 6 * HEAD_DIM)
    expected_workspaces = (
        ("gated_output", expected.gated_output.reshape(rows, projection_size)),
        (
            "convolved_key",
            expected.convolved_qkv[:, :, 1].reshape(rows, projection_size),
        ),
        (
            "convolved_value",
            expected.convolved_qkv[:, :, 2].reshape(rows, projection_size),
        ),
        ("raw_decay", expected.raw_decay.reshape(rows, projection_size)),
        ("raw_output_gate", expected.raw_output_gate.reshape(rows, projection_size)),
    )
    for workspace, (name, reference) in zip(workspace_views[:5], expected_workspaces):
        torch.testing.assert_close(
            workspace.cpu().float(),
            reference.float(),
            atol=0.005,
            rtol=0.005,
            msg=lambda message, name=name: f"{name}: {message}",
        )
    assert tuple(workspace_views[5].shape) == (rows, 16)
    torch.testing.assert_close(
        workspace_views[5][:, :6].cpu().float(),
        expected.raw_beta.reshape(rows, 6).float(),
        atol=0.005,
        rtol=0.005,
        msg=lambda message: f"raw_beta: {message}",
    )
    beta_padding = workspace_views[5][:, 6:].cpu()
    torch.testing.assert_close(
        beta_padding,
        torch.zeros_like(beta_padding),
        atol=0,
        rtol=0,
        msg=lambda message: f"beta N16 padding must be zero: {message}",
    )
    torch.testing.assert_close(
        actual.cpu().float(),
        expected.output.float(),
        atol=0.005,
        rtol=0.005,
        msg=lambda message: f"output: {message}",
    )
    actual_conv_state = actual_arguments[9].cpu()
    torch.testing.assert_close(
        actual_conv_state.float(),
        expected.conv_state.float(),
        atol=0.005,
        rtol=0.005,
        msg=lambda message: f"Conv state pool: {message}",
    )
    used_conv_slots = set(arguments[11].tolist()) - {0}
    unused_conv_slots = [
        slot for slot in range(batch_size + 1) if slot not in used_conv_slots
    ]
    torch.testing.assert_close(
        actual_conv_state[unused_conv_slots],
        initial_conv_state[unused_conv_slots],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        actual_arguments[10][0].cpu(), initial_null_snapshot, atol=0, rtol=0
    )
    for batch_index in range(batch_size):
        for token in range(seq_len):
            slot = int(arguments[12][batch_index * seq_len + token])
            torch.testing.assert_close(
                actual_arguments[10][slot].cpu(),
                expected.recurrent_state[slot],
                atol=0.005,
                rtol=0.005,
                msg=lambda message,
                b=batch_index,
                t=token: f"snapshot b={b}, token={t}: {message}",
            )


# Retain the existing explicit 28/32-core cases and cover automatic quotas.
_PRECISION_CASES = [
    pytest.param(16, 8, 28),
    pytest.param(16, 8, 32),
    pytest.param(32, 8, 32),
    pytest.param(16, 8, "auto"),
    pytest.param(32, 8, "auto"),
    *[pytest.param(batch, 8, 16) for batch in (16, 32, 512)],
    *[
        pytest.param(batch, 8, "available", marks=pytest.mark.slow)
        for batch in (16, 32, 48, 64, 256, 512)
    ],
]


@pytest.mark.npu
@pytest.mark.parametrize("batch_size,seq_len,cores", _PRECISION_CASES)
def test_precision(batch_size, seq_len, cores, monkeypatch):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    module = load_sample(SAMPLE)
    stream = torch.npu.current_stream(torch.npu.current_device())
    available = int(module.cannbotdsl.get_platform_info(stream=stream).cube_core_num)
    grids = tuple(count for count in (8, 16, 24, 28, 32) if count <= available)
    if not grids:
        pytest.skip("Mega KDA requires at least 8 AIC cores in the current stream")
    controls = (None,) if cores == "auto" else grids if cores == "available" else (cores,)
    if isinstance(cores, int) and available < cores:
        pytest.skip(f"requires a stream quota of at least {cores} AIC cores")
    for block_num in controls:
        _check_precision(batch_size, seq_len, block_num, monkeypatch)
