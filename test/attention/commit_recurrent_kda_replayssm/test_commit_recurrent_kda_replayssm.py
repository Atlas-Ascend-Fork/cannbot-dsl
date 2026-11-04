# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent contract, graph, precision and scheduling tests for Mega commit."""

from __future__ import annotations


import pytest
import torch

from _samples_path import load_sample


SAMPLE = "attention/commit_recurrent_kda_replayssm/commit_recurrent_kda_replayssm.py"


def _module():
    return load_sample(SAMPLE)


def _inputs(
    batch_size: int, *, heads: int = 6, sequence_length: int = 8, accepted: int = 0
):
    state = torch.zeros(batch_size, heads, 128, 128, dtype=torch.float32)
    replay_u = torch.zeros(batch_size, sequence_length, heads, 128, dtype=torch.float16)
    replay_k = torch.zeros_like(replay_u)
    replay_decay = torch.ones(
        batch_size, sequence_length, heads, 128, dtype=torch.float32
    )
    accepted_tokens = torch.full((batch_size,), accepted, dtype=torch.int32)
    return state, replay_u, replay_k, replay_decay, accepted_tokens


def _reference_commit(state, replay_u, replay_k, replay_decay, accepted):
    expected = state.cpu().clone()
    for batch_index in range(state.shape[0]):
        for token in range(int(accepted[batch_index].cpu())):
            expected[batch_index] = expected[batch_index] * replay_decay[
                batch_index, token
            ].cpu().unsqueeze(-2) + replay_u[
                batch_index, token
            ].cpu().float().unsqueeze(-1) * replay_k[
                batch_index, token
            ].cpu().float().unsqueeze(-2)
    return expected


def _run_commit_and_assert(module, tensors, expected, block_num):
    state, replay_u, replay_k, replay_decay, accepted = tensors
    actual = state.npu()
    module.commit_recurrent_kda_replayssm(
        actual,
        replay_u.npu(),
        replay_k.npu(),
        replay_decay.npu(),
        accepted.npu(),
        block_num=block_num,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.npu
@pytest.mark.parametrize(
    "batch_size,heads,sequence_length",
    ((16, 6, 8),),
)
@pytest.mark.parametrize("accepted_count", (1,))
def test_npu_commit_precision_batch_acceptance_and_auto_control(
    batch_size, heads, sequence_length, accepted_count
):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")

    module = _module()
    generator = torch.Generator().manual_seed(20260921 + batch_size + accepted_count)
    state = torch.randn(batch_size, heads, 128, 128, generator=generator) * 0.002
    replay_u = (
        torch.randn(batch_size, sequence_length, heads, 128, generator=generator) * 0.02
    ).half()
    replay_k = (
        torch.randn(batch_size, sequence_length, heads, 128, generator=generator) * 0.02
    ).half()
    replay_decay = torch.sigmoid(
        torch.randn(batch_size, sequence_length, heads, 128, generator=generator)
    )
    accepted = torch.full((batch_size,), accepted_count, dtype=torch.int32)
    expected = _reference_commit(state, replay_u, replay_k, replay_decay, accepted)

    actual = state.npu()
    module.commit_recurrent_kda_replayssm(
        actual,
        replay_u.npu(),
        replay_k.npu(),
        replay_decay.npu(),
        accepted.npu(),
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.npu
@pytest.mark.parametrize("block_num", (32,))
def test_npu_commit_all_explicit_core_controls(block_num):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    if int(torch.npu.get_device_properties().cube_core_num) < block_num:
        pytest.skip(f"requires {block_num} AIC cores")

    module = _module()
    state, replay_u, replay_k, replay_decay, accepted = _inputs(16, accepted=1)
    replay_u.fill_(0.25)
    replay_k.fill_(0.125)
    expected = _reference_commit(state, replay_u, replay_k, replay_decay, accepted)
    _run_commit_and_assert(
        module,
        (state, replay_u, replay_k, replay_decay, accepted),
        expected,
        block_num,
    )


@pytest.mark.npu
@pytest.mark.parametrize("block_num", (8,))
def test_npu_commit_uses_each_batch_accepted_prefix(block_num):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")

    module = _module()
    if int(torch.npu.get_device_properties().cube_core_num) < block_num:
        pytest.skip(f"requires {block_num} AIC cores")

    generator = torch.Generator().manual_seed(20260922)
    state = torch.randn(16, 6, 128, 128, generator=generator) * 0.002
    replay_u = (torch.randn(16, 8, 6, 128, generator=generator) * 0.02).half()
    replay_k = (torch.randn(16, 8, 6, 128, generator=generator) * 0.02).half()
    replay_decay = torch.sigmoid(torch.randn(16, 8, 6, 128, generator=generator))
    accepted = torch.tensor(
        [0, 8, 1, 0, 4, 2, 0, 7, 3, 0, 5, 0, 6, 0, 8, 0],
        dtype=torch.int32,
    )
    expected = _reference_commit(state, replay_u, replay_k, replay_decay, accepted)
    _run_commit_and_assert(
        module,
        (state, replay_u, replay_k, replay_decay, accepted),
        expected,
        block_num,
    )
