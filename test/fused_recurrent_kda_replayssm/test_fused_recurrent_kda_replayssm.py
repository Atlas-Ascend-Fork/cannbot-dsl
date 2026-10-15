# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent FP32 oracle for standalone ReplaySSM verify and commit."""

from dataclasses import dataclass

import pytest
import torch
from cannbotdsl import get_platform_info

from _samples_path import load_sample

SAMPLE = "fused_recurrent_kda_replayssm/fused_recurrent_kda_replayssm.py"
COMMIT_SAMPLE = "commit_recurrent_kda_replayssm/commit_recurrent_kda_replayssm.py"
DIM = 128
SENTINEL = -7.0


@dataclass(frozen=True)
class _PrecisionInputs:
    """Dense CPU inputs shared by the oracle and layout packing."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    gate: torch.Tensor
    beta: torch.Tensor
    checkpoint: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    lengths: tuple[int, ...]


def _make_inputs(heads, lengths):
    batch, capacity = len(lengths), max(lengths) + 2
    generator = torch.Generator().manual_seed(20261014 + heads + sum(lengths))

    def random(shape, dtype=torch.bfloat16, scale=0.2):
        return (torch.randn(shape, generator=generator) * scale).to(dtype)

    shape = (batch, capacity, heads, DIM)
    return _PrecisionInputs(
        *(random(shape) for _ in range(4)),
        random((batch, capacity, heads, 1)),
        random((batch, heads, DIM, DIM), torch.float32, 0.02),
        random((heads,), torch.float32),
        random((heads, DIM), torch.float32),
        tuple(lengths),
    )


def _reference(inputs):
    query = inputs.query
    key = inputs.key
    value = inputs.value
    gate = inputs.gate
    beta = inputs.beta
    checkpoint = inputs.checkpoint
    a_log = inputs.a_log
    dt_bias = inputs.dt_bias
    lengths = inputs.lengths
    query = query.float()
    key = key.float()
    query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    decay = torch.exp(-5.0 * torch.sigmoid(a_log.exp()[None, None, :, None] * (gate.float() + dt_bias)))
    output = torch.zeros_like(value)
    records_u = torch.full_like(value, SENTINEL, dtype=torch.float16)
    records_k = torch.full_like(value, SENTINEL, dtype=torch.float16)
    records_decay = torch.full_like(value, SENTINEL, dtype=torch.float32)
    snapshots = []
    for batch, length in enumerate(lengths):
        state = checkpoint[batch].clone()
        states = [state.clone()]
        for token in range(length):
            state *= decay[batch, token].unsqueeze(-2)
            delta = (value[batch, token].float() - (state * key[batch, token].unsqueeze(-2)).sum(-1)) * beta[
                batch, token
            ].float().sigmoid()
            state += delta.unsqueeze(-1) * key[batch, token].unsqueeze(-2)
            output[batch, token] = ((state * query[batch, token].unsqueeze(-2)).sum(-1) * DIM**-0.5).bfloat16()
            records_u[batch, token] = delta.half()
            records_k[batch, token] = key[batch, token].half()
            records_decay[batch, token] = decay[batch, token]
            states.append(state.clone())
        snapshots.append(states)
    return output, records_u, records_k, records_decay, snapshots


def _pack_layout(inputs, layout):
    query, key, value = inputs.query, inputs.key, inputs.value
    gate, beta, lengths = inputs.gate, inputs.beta, inputs.lengths
    # Dense layouts contain the common sequence; records keep a sentinel suffix.
    if layout != "TND":
        query, key, value, gate, beta = (tensor[:, : lengths[0]] for tensor in (query, key, value, gate, beta))
    mixed = torch.cat((query, key, value), dim=2)
    if layout == "BSND":
        return mixed.flatten(2).contiguous(), gate.contiguous(), beta.contiguous(), None
    if layout == "BNSD":
        return (
            mixed.transpose(1, 2).contiguous(),
            gate.transpose(1, 2).contiguous(),
            beta.transpose(1, 2).contiguous(),
            None,
        )
    packed = [torch.cat([x[b, :length] for b, length in enumerate(lengths)], dim=0) for x in (mixed, gate, beta)]
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    return packed[0].flatten(1).contiguous(), packed[1].contiguous(), packed[2].contiguous(), cu


def _check_output(output, expected, layout, lengths):
    if layout == "TND":
        expected = torch.cat([expected[batch, :length] for batch, length in enumerate(lengths)], dim=0)
    else:
        expected = expected[:, : lengths[0]]
        if layout == "BNSD":
            expected = expected.transpose(1, 2)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output.cpu().float(), expected.float(), atol=0.005, rtol=0.005)


def _check_records(records, references, lengths):
    for actual, reference in zip(records, references):
        torch.testing.assert_close(actual.cpu().float(), reference.float(), atol=0.005, rtol=0.005)
        for batch, length in enumerate(lengths):
            torch.testing.assert_close(actual.cpu()[batch, length:], reference[batch, length:], atol=0, rtol=0)


def _check_commit(commit, state, records, snapshots, control):
    # Commit consumes exactly the FP16 U/K and FP32 decay records from verify.
    accepted, cores = control
    checkpoint = state.cpu()
    committed = state.clone()
    commit.commit_recurrent_kda_replayssm(
        committed, *records, torch.tensor(accepted, dtype=torch.int32, device="npu"), block_num=cores
    )
    torch.npu.synchronize()
    expected_state = torch.stack([snapshots[batch][count] for batch, count in enumerate(accepted)])
    torch.testing.assert_close(committed.cpu(), expected_state, atol=0.005, rtol=0.005)
    for batch, count in enumerate(accepted):
        if count == 0:
            torch.testing.assert_close(committed.cpu()[batch], checkpoint[batch], atol=0, rtol=0)


def _precision_case(layout, heads, lengths, accepted, cores):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    module = load_sample(SAMPLE)
    commit = load_sample(COMMIT_SAMPLE)
    inputs = _make_inputs(heads, lengths)
    mixed, gate, beta, cu = _pack_layout(inputs, layout)
    mixed = mixed.npu()
    if cores is not None:
        stream = torch.npu.current_stream(mixed.device)
        available_cores = get_platform_info(stream=stream).cube_core_num
        if cores > available_cores:
            pytest.skip(f"requires {cores} AIC cores; current stream exposes {available_cores}")
    expected = _reference(inputs)
    records = [
        torch.full(inputs.query.shape, SENTINEL, dtype=dtype, device="npu")
        for dtype in (torch.float16, torch.float16, torch.float32)
    ]
    state = inputs.checkpoint.npu()
    output = module.fused_recurrent_kda_replayssm(
        mixed,
        state,
        beta.npu(),
        gate.npu(),
        None,
        A_log=inputs.a_log.npu(),
        dt_bias=inputs.dt_bias.npu(),
        layout_qkv=layout,
        replay_u=records[0],
        replay_k=records[1],
        replay_decay=records[2],
        cu_seqlens=None if cu is None else cu.npu(),
        block_num=cores,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(state.cpu(), inputs.checkpoint, atol=0, rtol=0)
    _check_output(output, expected[0], layout, lengths)
    _check_records(records, expected[1:4], lengths)
    _check_commit(commit, state, records, expected[4], (accepted, cores))
    torch.testing.assert_close(state.cpu(), inputs.checkpoint, atol=0, rtol=0)


@pytest.mark.npu
@pytest.mark.parametrize(
    "layout,heads,lengths,accepted,cores",
    [
        pytest.param("BSND", 1, (8,), (8,), 24, id="bsnd-small-row-24-cores"),
        pytest.param("BNSD", 17, (9, 9), (0, 9), 8, id="bnsd-full-row-multiwave"),
        pytest.param("TND", 8, (5, 1, 3), (5, 0, 2), None, id="tnd-medium-row-variable-length"),
    ],
)
def test_precision_layouts_and_commit(layout, heads, lengths, accepted, cores):
    _precision_case(layout, heads, lengths, accepted, cores)


@pytest.mark.npu
def test_minimal_full_operator():
    _precision_case("BSND", 1, (1,), (1,), None)
