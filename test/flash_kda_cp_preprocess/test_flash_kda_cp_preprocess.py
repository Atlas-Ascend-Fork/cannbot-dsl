# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CP preprocess metadata ABI and numerical outputs."""

from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from _samples_path import load_sample

sample = load_sample("flash_kda_cp_preprocess/flash_kda_cp_preprocess.py")
metadata_sample = load_sample("flash_kda_cp_preprocess/flash_kda_cp_preprocess_metadata.py")


def test_metadata_capacity_and_decoder_cover_cp_summary_schedule():
    assert metadata_sample.round_record_words(1) == 142
    assert metadata_sample.chunk_capacity_upper_bound(1, 64) == 1
    assert metadata_sample.metadata_capacity_upper_bound(1, 64, is_packed=False) > metadata_sample.HEADER_WORDS
    with pytest.raises(ValueError, match="positive"):
        metadata_sample.round_record_words(0)


def test_minimal_full_metadata_schedule():
    capacity = metadata_sample.metadata_capacity_upper_bound(1, 64, is_packed=False)
    metadata = np.zeros(capacity, dtype=np.uint32)
    status = metadata_sample.flash_kda_cp_preprocess_metadata_kernel.run_host(
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


def _require_npu():
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU runtime is unavailable")


def _normalize(tensor):
    return tensor * torch.rsqrt(tensor.float().square().sum(dim=-1, keepdim=True) + 1e-6)


@dataclass(frozen=True)
class ReferenceInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    seq_len: int


def _reference(inputs):
    """Independent FP32 chunk recurrence for the eight public outputs."""
    q, k, v, g, beta = inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta
    a_log, dt_bias, seq_len = inputs.a_log, inputs.dt_bias, inputs.seq_len
    batch, heads, _, dim = v.shape
    key_heads = k.shape[1]
    k = k.repeat_interleave(heads // key_heads, dim=1)
    q = q.repeat_interleave(heads // key_heads, dim=1)
    gate = -0.1 * torch.sigmoid(a_log.view(1, heads, 1, 1).exp() * (g.float() + dt_bias.view(1, heads, 1, dim)))
    beta = torch.sigmoid(beta.float())
    normalized_k = _normalize(k).to(k.dtype).float()
    normalized_q = _normalize(q).to(q.dtype).float()
    padded_len = ((seq_len + 63) // 64) * 64
    pad_rows = padded_len - seq_len
    key_chunks = functional.pad(normalized_k, (0, 0, 0, pad_rows)).reshape(batch, heads, -1, 64, dim)
    value_chunks = functional.pad(v.float(), (0, 0, 0, pad_rows)).reshape(batch, heads, -1, 64, dim)
    gate_chunks = functional.pad(gate, (0, 0, 0, pad_rows)).reshape(batch, heads, -1, 64, dim)
    beta_chunks = functional.pad(beta, (0, pad_rows)).reshape(batch, heads, -1, 64)
    query_chunks = functional.pad(normalized_q, (0, 0, 0, pad_rows)).reshape(batch, heads, -1, 64, dim)
    causal = torch.tril(torch.ones(64, 64, dtype=torch.bool), diagonal=-1)
    mqk_causal = torch.tril(torch.ones(64, 64, dtype=torch.bool))
    identity = torch.eye(64)
    transition = torch.eye(dim).expand(batch, heads, dim, dim).clone()
    state_ext = torch.zeros(batch, heads, dim, dim)
    qg_chunks, kg_chunks, u_chunks, w_chunks, mqk_chunks, decay_chunks = [], [], [], [], [], []
    for chunk_idx in range(key_chunks.shape[2]):
        key = key_chunks[:, :, chunk_idx]
        value = value_chunks[:, :, chunk_idx]
        query = query_chunks[:, :, chunk_idx]
        cumulative = gate_chunks[:, :, chunk_idx].cumsum(dim=-2)
        beta_chunk = beta_chunks[:, :, chunk_idx]
        key_pair_decay = (cumulative.unsqueeze(-2) - cumulative.unsqueeze(-3)).exp()
        kk = torch.einsum("bhid,bhijd,bhjd->bhij", key, key_pair_decay, key)
        strict = (kk * beta_chunk.unsqueeze(-1)).masked_fill(~causal, 0)
        inverse = torch.linalg.inv(identity + strict)
        inverse_beta = inverse * beta_chunk.unsqueeze(-2)
        corrected_key = inverse_beta @ (cumulative.exp() * key)
        corrected_value = inverse_beta @ value
        last_decay = cumulative[:, :, -1]
        restored_key = (last_decay.unsqueeze(-2) - cumulative).exp() * key
        chunk_transition = torch.diag_embed(last_decay.exp()) - restored_key.transpose(-1, -2) @ corrected_key
        chunk_state = restored_key.transpose(-1, -2) @ corrected_value
        state_ext = chunk_transition @ state_ext + chunk_state
        transition = chunk_transition @ transition
        qg_chunks.append((query * cumulative.exp() * (dim**-0.5)).to(torch.bfloat16))
        kg_chunks.append(restored_key.to(torch.bfloat16))
        u_chunks.append(corrected_value.to(torch.bfloat16))
        w_chunks.append((-corrected_key).to(torch.bfloat16))
        decay_chunks.append(last_decay.exp())
        mqk = torch.einsum("bhid,bhijd,bhjd->bhij", query * (dim**-0.5), key_pair_decay, key)
        mqk_chunks.append(mqk.masked_fill(~mqk_causal, 0).to(torch.bfloat16))

    def flatten(chunks):
        return torch.stack(chunks, dim=2).reshape(batch, heads, padded_len, dim)[:, :, :seq_len]

    return (
        transition.transpose(-1, -2),
        state_ext.transpose(-1, -2),
        torch.stack(decay_chunks, dim=2),
        flatten(qg_chunks),
        flatten(kg_chunks),
        flatten(u_chunks),
        flatten(w_chunks),
        torch.stack(mqk_chunks, dim=2),
    )


def _reference_inputs(seq_len, n_v, n_kv):
    generator = torch.Generator().manual_seed(53)
    q = torch.randn(1, n_kv, seq_len, 128, generator=generator).bfloat16()
    k = torch.randn(1, n_kv, seq_len, 128, generator=generator).bfloat16()
    v = torch.randn(1, n_v, seq_len, 128, generator=generator).bfloat16()
    g = torch.randn(v.shape, generator=generator).bfloat16()
    beta = torch.randn(1, n_v, seq_len, generator=generator).bfloat16()
    a_log = torch.linspace(-1.0, 0.5, n_v, dtype=torch.float32)
    dt_bias = torch.randn(n_v, 128, generator=generator)
    return ReferenceInputs(q, k, v, g, beta, a_log, dt_bias, seq_len)


def _run_preprocess(inputs, layout):
    q, k, v, g, beta = inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta
    if layout == "TND":

        def place(tensor):
            return tensor[0].transpose(0, 1).contiguous().npu()

        q_npu, k_npu, v_npu, g_npu = (place(t) for t in (q, k, v, g))
        beta_npu = beta[0].transpose(0, 1).contiguous().npu()
    else:

        def place(tensor):
            return (tensor if layout == "BNSD" else tensor.transpose(1, 2).contiguous()).npu()

        q_npu, k_npu, v_npu, g_npu, beta_npu = (place(t) for t in (q, k, v, g, beta))
    cu_seqlens = torch.tensor([0, inputs.seq_len], dtype=torch.int32).npu() if layout == "TND" else None
    metadata = metadata_sample.flash_kda_cp_preprocess_metadata(q_npu, v_npu, layout, cu_seqlens)
    return sample.flash_kda_cp_preprocess(
        q_npu,
        k_npu,
        v_npu,
        g_npu,
        beta_npu,
        128**-0.5,
        inputs.a_log.npu(),
        inputs.dt_bias.npu(),
        -0.1,
        layout,
        metadata,
        cu_seqlens,
    )


def _assert_output_close(observed, reference, name):
    assert observed.shape == reference.shape, name
    error = (observed.cpu().float() - reference.float()).abs()
    tolerance = 0.005 + 0.005 * reference.float().abs()
    normalized_error = (error / tolerance).flatten().quantile(0.999).item()
    assert normalized_error <= 1.0, (name, normalized_error, error.flatten().quantile(0.999).item())


@pytest.mark.npu
@pytest.mark.parametrize(
    "layout,seq_len,n_v,n_kv",
    [
        pytest.param("BNSD", 17, 3, 1, marks=pytest.mark.whitebox),
        pytest.param("BSND", 49, 5, 1, marks=pytest.mark.whitebox),
        pytest.param("TND", 577, 96, 3, marks=pytest.mark.whitebox),
        pytest.param("BNSD", 64, 1, 1, id="minimal_full"),
    ],
)
def test_preprocess_outputs_match_torch_reference(layout, seq_len, n_v, n_kv):
    _require_npu()
    inputs = _reference_inputs(seq_len, n_v, n_kv)
    expected = _reference(inputs)
    actual = _run_preprocess(inputs, layout)
    for name, observed, reference in zip(
        ("transition", "state_ext", "decay", "qg", "kg", "u", "w", "mqk"), actual, expected
    ):
        if layout == "TND" and name in {"qg", "kg", "u", "w"}:
            observed = observed.transpose(0, 1).unsqueeze(0)
        _assert_output_close(observed, reference, name)


def _run_packed_preprocess(cases):
    def packed(field):
        tensors = (getattr(case, field)[0].transpose(0, 1) for case in cases)
        return torch.cat(tuple(tensors), dim=0).contiguous().npu()

    q, k, v, g, beta = (packed(field) for field in ("q", "k", "v", "g", "beta"))
    boundaries = [0]
    for case in cases:
        boundaries.append(boundaries[-1] + case.seq_len)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32).npu()
    metadata = metadata_sample.flash_kda_cp_preprocess_metadata(q, v, "TND", cu_seqlens)
    return sample.flash_kda_cp_preprocess(
        q, k, v, g, beta, 128**-0.5, cases[0].a_log.npu(), cases[0].dt_bias.npu(), -0.1, "TND", metadata, cu_seqlens
    )


def _packed_sequence_outputs(actual, sequence, start, length):
    chunks = (length + 63) // 64
    sequence_slice = slice(sequence, sequence + 1)
    token_slice = slice(start, start + length)
    return (
        actual[0][sequence_slice],
        actual[1][sequence_slice],
        actual[2][sequence_slice, :, :chunks],
        *(tensor[token_slice].transpose(0, 1).unsqueeze(0) for tensor in actual[3:7]),
        actual[7][sequence_slice, :, :chunks],
    )


@pytest.mark.npu
@pytest.mark.whitebox
def test_packed_tnd_multiple_sequences_match_independent_reference():
    _require_npu()
    first = _reference_inputs(17, 3, 1)
    second = replace(_reference_inputs(65, 3, 1), dt_bias=first.dt_bias)
    actual = _run_packed_preprocess((first, second))
    assert actual[2].shape[2] == 3
    start = 0
    for sequence, case in enumerate((first, second)):
        observed_outputs = _packed_sequence_outputs(actual, sequence, start, case.seq_len)
        for name, observed, expected in zip(
            ("transition", "state_ext", "decay", "qg", "kg", "u", "w", "mqk"), observed_outputs, _reference(case)
        ):
            _assert_output_close(observed, expected, name)
        start += case.seq_len
