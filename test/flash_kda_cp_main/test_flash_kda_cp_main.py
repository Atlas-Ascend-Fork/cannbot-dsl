# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Metadata ABI and numerical results of the CP main operator."""

import numpy as np
import pytest
import torch
from _samples_path import load_sample

sample = load_sample("flash_kda_cp_main/flash_kda_cp_main.py")
metadata_sample = load_sample("flash_kda_cp_main/flash_kda_cp_main_metadata.py")


def test_metadata_capacity_and_decoder_cover_shared_cp_abi():
    assert metadata_sample.round_record_words(1) == 142
    assert metadata_sample.chunk_capacity_upper_bound(1, 64) == 1
    assert metadata_sample.metadata_capacity_upper_bound(1, 64, is_packed=False) > metadata_sample.HEADER_WORDS
    with pytest.raises(ValueError, match="positive"):
        metadata_sample.round_record_words(0)


def test_minimal_full_metadata_schedule():
    capacity = metadata_sample.metadata_capacity_upper_bound(1, 64, is_packed=False)
    metadata = np.zeros(capacity, dtype=np.uint32)
    status = metadata_sample.flash_kda_cp_main_metadata_kernel.run_host(
        cu_seqlens=0,
        metadata=metadata,
        metadata_word_capacity=capacity,
        batch=1,
        physical_batches=1,
        storage_length=64,
        value_heads=1,
        has_cu=0,
        is_packed=0,
        core_num=32,
        workspace_slots=metadata_sample.WORKSPACE_SLOTS,
        max_chunk_group=512,
    )
    assert status == metadata_sample.STATUS_OK
    decoded = metadata_sample.decode_metadata(metadata.tolist())
    assert decoded["stage12_round_num"] == 1
    assert decoded["batch"] == 1


@pytest.mark.parametrize(
    "seq_len,layout,n_v",
    [
        pytest.param(17, "BNSD", 3, marks=pytest.mark.whitebox),
        pytest.param(49, "BSND", 5, marks=pytest.mark.whitebox),
        pytest.param(577, "TND", 96, marks=pytest.mark.whitebox),
        pytest.param(64, "BNSD", 1, id="minimal_full"),
    ],
)
@pytest.mark.npu
def test_main_matches_chunk_recurrence(seq_len, layout, n_v):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU runtime is unavailable")
    generator = torch.Generator().manual_seed(19)
    chunks, width = (seq_len + 63) // 64, 128

    def random_bf16(shape):
        return (torch.randn(shape, generator=generator) * 0.02).bfloat16().npu()

    shape = (seq_len, n_v, width) if layout == "TND" else (1, n_v, seq_len, width)
    qg, kg, u, w = (random_bf16(shape) for _ in range(4))
    if layout == "TND":
        qg_bnsd, kg_bnsd, u_bnsd, w_bnsd = (tensor.transpose(0, 1).unsqueeze(0) for tensor in (qg, kg, u, w))
    else:
        qg_bnsd, kg_bnsd, u_bnsd, w_bnsd = qg, kg, u, w
    mqk_cpu = torch.randn((1, n_v, chunks, 64, 64), generator=generator) * 0.02
    mqk = torch.tril(mqk_cpu).bfloat16().npu()
    decay = torch.full((1, n_v, chunks, width), 0.99, dtype=torch.float32).npu()
    initial = (torch.randn((1, n_v, width, width), generator=generator) * 0.02).npu()
    transition = (
        torch.eye(width, dtype=torch.float32)
        .reshape(1, 1, width, width)
        .expand(1, n_v, width, width)
        .contiguous()
        .npu()
    )
    state_ext = torch.zeros_like(initial)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32).npu() if layout == "TND" else None
    metadata = metadata_sample.flash_kda_cp_main_metadata(qg, initial, layout, cu_seqlens)
    output, final_state = sample.flash_kda_cp_main(
        transition, state_ext, decay, qg, kg, u, w, mqk, initial, layout, metadata, cu_seqlens
    )
    torch.npu.synchronize()

    state = initial.cpu()[0].float()
    expected_chunks = []
    q_cpu, k_cpu = qg_bnsd.cpu()[0].float(), kg_bnsd.cpu()[0].float()
    u_cpu, w_cpu = u_bnsd.cpu()[0].float(), w_bnsd.cpu()[0].float()
    mqk_cpu_npu = mqk.cpu()[0].float()
    decay_cpu = decay.cpu()[0].float()
    for chunk_idx in range(chunks):
        start, end = chunk_idx * 64, min((chunk_idx + 1) * 64, seq_len)
        query, key = q_cpu[:, start:end], k_cpu[:, start:end]
        value, weight = u_cpu[:, start:end], w_cpu[:, start:end]
        # stage12w reloads the recurrent state through BF16 L1 at every
        # chunk boundary and stores the U correction back to BF16.
        state_for_mm = state.bfloat16().float()
        delta = (value + torch.matmul(weight, state_for_mm.transpose(-1, -2))).bfloat16().float()
        mqk_chunk = mqk_cpu_npu[:, chunk_idx, : end - start, : end - start]
        expected_chunks.append(
            (torch.matmul(query, state_for_mm.transpose(-1, -2)) + torch.matmul(mqk_chunk, delta)).bfloat16()
        )
        state_next = state_for_mm * decay_cpu[:, chunk_idx].unsqueeze(-2)
        state_next = state_next + torch.matmul(delta.transpose(-1, -2), key)
        state = state_next if chunk_idx + 1 == chunks else state_next.bfloat16().float()
    expected_output = torch.cat(expected_chunks, dim=1)
    if layout == "BNSD":
        actual_output = output.cpu()[0]
    elif layout == "BSND":
        actual_output = output.cpu()[0].transpose(0, 1)
    else:
        actual_output = output.cpu().transpose(0, 1)
    torch.testing.assert_close(actual_output, expected_output, atol=3e-3, rtol=3e-2)
    torch.testing.assert_close(final_state.cpu()[0], state, atol=3e-3, rtol=3e-2)
