# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import ctypes
from pathlib import Path
import sys

import pytest
import torch

_source = (
    Path(__file__).resolve().parents[3]
    / "samples/attention/mega_compressor/mega_compressor.py"
)
_spec = importlib.util.spec_from_file_location("mega_compressor_sample", _source)
op = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = op
_spec.loader.exec_module(op)


def _inputs(batch, seq):
    g = torch.Generator().manual_seed(42)
    x = (torch.randn(batch * seq, 5120, generator=g) * 0.5).bfloat16()
    wk = (torch.randn(512, 5120, generator=g) * 0.02).bfloat16()
    wg = (torch.randn(512, 5120, generator=g) * 0.02).bfloat16()
    state = torch.randn(batch + 1, 8, 1024, generator=g)
    gamma = (torch.rand(512, generator=g) + 0.5).bfloat16()
    starts = torch.full((batch,), 0 if seq == 8192 else 8192, dtype=torch.int32)
    used = torch.full((batch,), seq, dtype=torch.int32)
    cu = torch.arange(batch + 1, dtype=torch.int32) * seq
    table = torch.arange(batch, 0, -1, dtype=torch.int32)
    return [x, wk, wg, state, gamma, starts, used, cu, table]


def _reference(tensors, ratio):
    x, wk, wg, state, gamma, starts, used, cu, table = tensors
    projected = torch.cat((x.float() @ wk.float().T, x.float() @ wg.float().T), -1)
    cache = state.clone()
    written = torch.zeros(state.shape[:2], dtype=torch.bool)
    outputs = []
    for b, (start, length, block) in enumerate(
        zip(starts.tolist(), used.tolist(), table.tolist())
    ):
        rows = projected[int(cu[b]) : int(cu[b]) + length]
        groups = (start + length) // ratio - start // ratio
        window = rows
        if start % ratio:
            window = torch.cat((state[block, (start - 1) % 8].view(1, 1024), window))
        window = window[: groups * ratio].reshape(groups, ratio, 1024)
        pooled = (
            (window[..., 512:].softmax(1) * window[..., :512])
            .sum(1)
            .to(x.dtype)
            .float()
        )
        outputs.append(
            (
                pooled
                * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + 1e-6)
                * gamma.float()
            ).to(x.dtype)
        )
        for token in range(max(0, length - 8), length):
            slot = (start + token) % 8
            cache[block, slot] = rows[token]
            written[block, slot] = True
    return outputs, cache, written


def _check(output, state, cpu, ratio):
    expected, cache, written = _reference(cpu, ratio)
    actual = output.cpu()
    offset = 0
    for b, ref in enumerate(expected):
        n = ref.shape[0]
        got = actual[b, :n] if actual.ndim == 3 else actual[offset : offset + n]
        assert torch.isfinite(got).all()
        if n:
            assert (
                torch.isclose(got.float(), ref.float(), atol=1e-4, rtol=0.0078125)
                .float()
                .mean()
                >= 0.995
            )
        offset += n
    if actual.ndim == 2:
        assert torch.count_nonzero(actual[offset:]) == 0
    got = state.cpu()
    torch.testing.assert_close(got, cache, atol=2e-5, rtol=2e-4)
    assert torch.equal(
        got[~written].view(torch.int32), cpu[3][~written].view(torch.int32)
    )


def _replay(call, state, initial, output):
    expected_output = output.cpu().contiguous().view(torch.int16)
    expected_state = state.cpu().view(torch.int32)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = call()
    for _ in range(5):
        state.copy_(initial)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(
            captured.cpu().contiguous().view(torch.int16), expected_output
        )
        assert torch.equal(state.cpu().view(torch.int32), expected_state)
    return graph, captured


@pytest.mark.npu
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("seq", [1, 6, 8, 8192])
@pytest.mark.parametrize("ratio", [1, 2])
def test_mega_compressor_bsh(batch, seq, ratio):
    pytest.importorskip("torch_npu")
    cpu = _inputs(batch, seq)
    x, wk, wg, state, gamma, starts, _, _, table = [t.npu() for t in cpu]
    x = x.view(batch, seq, 5120)
    attrs = dict(
        cmp_ratio=ratio, state_block_table=table, start_pos=starts, norm_eps=1e-6
    )
    output = op.mega_compressor(x, wk, wg, state, norm_weight=gamma, **attrs)
    _check(output, state, cpu, ratio)
    state.copy_(cpu[3])
    with op.prepare_mega_compressor(x, wk, wg, state, gamma, **attrs) as prepared:
        prepared.output.zero_()  # S1/R2/even start has no valid output.
        output = prepared()
        _check(output, state, cpu, ratio)
        _replay(prepared, state, cpu[3], output)


@pytest.mark.npu
@pytest.mark.parametrize("ratio", [1, 2])
def test_mega_compressor_int32_start_boundary(ratio):
    pytest.importorskip("torch_npu")
    batch, seq = 1, 6
    cpu = _inputs(batch, seq)
    cpu[5] = torch.full((batch,), 2**31 - 1, dtype=torch.int32)
    x, wk, wg, state, gamma, starts, used, cu, table = [t.npu() for t in cpu]
    x = x.view(batch, seq, 5120)
    attrs = dict(
        cmp_ratio=ratio, state_block_table=table, start_pos=starts, norm_eps=1e-6
    )
    output = op.mega_compressor(x, wk, wg, state, norm_weight=gamma, **attrs)
    _check(output, state, cpu, ratio)
    state.copy_(cpu[3])
    plan = op.DecodePlan()
    packed = op.pack_weights(wk, wg, plan)
    workspace = torch.empty(plan.workspace_shape(batch * seq), device=x.device)
    th_output = x.new_zeros((min(batch * seq, batch * seq // ratio + batch), 512))
    host = op.THDecodeHost(batch, batch * seq, ratio, 8, 1e-6, plan)
    args = (
        x,
        packed,
        gamma.view(1, 512),
        state.view(-1, 1024),
        starts,
        table,
        workspace,
        th_output,
        cu,
        used,
    )
    program = op.compile_args(host.run, args)
    try:
        th_output.zero_()
        program(*args)
        _check(th_output, state, cpu, ratio)
    finally:
        program.close()


@pytest.mark.npu
@pytest.mark.parametrize(
    "batch,seq,ratio", [(b, s, r) for b in (1, 4) for s in (1, 6, 8) for r in (1, 2)]
)
def test_mega_compressor_th(batch, seq, ratio):
    pytest.importorskip("torch_npu")
    cpu = _inputs(batch, seq)
    tensors = [t.npu() for t in cpu]
    x, wk, wg, state, gamma, starts, used, cu, table = tensors
    plan = op.DecodePlan()
    packed = op.pack_weights(wk, wg, plan)
    workspace = torch.empty(plan.workspace_shape(batch * seq), device=x.device)
    output = x.new_zeros((min(batch * seq, batch * seq // ratio + batch), 512))
    host = op.THDecodeHost(batch, batch * seq, ratio, 8, 1e-6, plan)
    args = (
        x,
        packed,
        gamma.view(1, 512),
        state.view(-1, 1024),
        starts,
        table,
        workspace,
        output,
        cu,
        used,
    )
    program = op.compile_args(host.run, args)

    def call():
        output.zero_()
        program(*args)
        return output

    try:
        _check(call(), state, cpu, ratio)
        graph, captured = _replay(call, state, cpu[3], output)
        # Reuse the same capture with live odd starts, ragged rows and mapping.
        cpu[5].add_(1)
        cpu[8] = cpu[8].flip(0).contiguous()
        if batch == 4 and seq == 8:
            cpu[7] = torch.tensor([0, 2, 15, 15, 32], dtype=torch.int32)
            cpu[6] = torch.tensor([2, 13, 0, 5], dtype=torch.int32)
        for i in (3, 5, 6, 7, 8):
            tensors[i].copy_(cpu[i])
        graph.replay()
        _check(captured, state, cpu, ratio)
        used.zero_()
        before = state.clone()
        graph.replay()
        torch.npu.synchronize()
        assert torch.count_nonzero(captured.cpu()) == 0
        assert torch.equal(
            state.cpu().view(torch.int32), before.cpu().view(torch.int32)
        )
    finally:
        program.close()


@pytest.mark.npu
@pytest.mark.parametrize(
    "seq,ratio,dtype,odd",
    [
        (1, 2, torch.bfloat16, True),
        (6, 2, torch.bfloat16, True),
        (8, 1, torch.float16, False),
        (6, 2, torch.float16, False),
        (8192, 2, torch.bfloat16, True),
    ],
)
def test_mega_compressor_general_paths(seq, ratio, dtype, odd):
    pytest.importorskip("torch_npu")
    cpu = _inputs(1, seq)
    for index in (0, 1, 2, 4):
        cpu[index] = cpu[index].to(dtype)
    if odd:
        cpu[5].add_(1)
    x, wk, wg, state, gamma, starts, used, _, table = [value.npu() for value in cpu]
    output = op.mega_compressor(
        x.view(1, seq, 5120),
        wk,
        wg,
        state,
        cmp_ratio=ratio,
        norm_weight=gamma,
        norm_eps=1e-6,
        state_block_table=table,
        start_pos=starts,
        seqused=used,
        backend="legacy",
    )
    _check(output, state, cpu, ratio)


@pytest.mark.npu
@pytest.mark.parametrize("backend", ["auto", "legacy"])
def test_mega_compressor_cpu_metadata_and_defaults(backend):
    pytest.importorskip("torch_npu")
    cpu = _inputs(1, 6)
    cpu[5].zero_()
    x, wk, wg, state, gamma = [value.npu() for value in cpu[:5]]
    output = op.mega_compressor(
        x.view(1, 6, 5120),
        wk,
        wg,
        state,
        cmp_ratio=2,
        norm_weight=gamma,
        norm_eps=1e-6,
        state_block_table=cpu[8],
        backend=backend,
    )
    _check(output, state, cpu, 2)


@pytest.mark.npu
def test_packed_weights_match_cpu_layout():
    pytest.importorskip("torch_npu")
    cpu = _inputs(1, 6)
    expected = op.pack_weights(cpu[1], cpu[2])
    actual = op.pack_weights(cpu[1].npu(), cpu[2].npu()).cpu()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.npu
@pytest.mark.parametrize("batch,is_th", [(1, False), (4, False), (1, True)])
def test_mega_compressor_strided_input(batch, is_th):
    pytest.importorskip("torch_npu")
    cpu = _inputs(batch, 6)
    x, wk, wg, state, gamma, starts, used, cu, table = [value.npu() for value in cpu]
    if is_th:
        x = x.T.contiguous().T
    else:
        x = x.view(batch, 6, 5120).transpose(1, 2).contiguous().transpose(1, 2)
    assert not x.is_contiguous()
    output = op.mega_compressor(
        x,
        wk,
        wg,
        state,
        cmp_ratio=2,
        norm_weight=gamma,
        norm_eps=1e-6,
        state_block_table=table,
        start_pos=starts,
        seqused=used,
        cu_seqlens=cu if is_th else None,
        backend="legacy",
    )
    _check(output[:3] if is_th else output, state, cpu, 2)


@pytest.mark.npu
@pytest.mark.parametrize("backend", ["auto", "legacy"])
def test_mega_compressor_stream_core_limit(backend):
    pytest.importorskip("torch_npu")
    cpu = _inputs(4, 8)
    x, wk, wg, state, gamma, starts, used, _, table = [value.npu() for value in cpu]
    runtime = ctypes.CDLL("libascendcl.so")
    set_limit = runtime.aclrtSetStreamResLimit
    set_limit.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
    set_limit.restype = ctypes.c_int
    reset_limit = runtime.aclrtResetStreamResLimit
    reset_limit.argtypes = [ctypes.c_void_p]
    reset_limit.restype = ctypes.c_int
    stream = torch.npu.Stream()
    handle = ctypes.c_void_p(stream.npu_stream)
    torch.npu.synchronize()
    try:
        assert set_limit(handle, 0, 28) == 0
        assert set_limit(handle, 1, 56) == 0
        with torch.npu.stream(stream):
            assert op._effective_cube_cores(x.device) == 28
            output = op.mega_compressor(
                x.view(4, 8, 5120),
                wk,
                wg,
                state,
                cmp_ratio=2,
                norm_weight=gamma,
                norm_eps=1e-6,
                state_block_table=table,
                start_pos=starts,
                seqused=used,
                backend=backend,
            )
            stream.synchronize()
        _check(output, state, cpu, 2)
    finally:
        stream.synchronize()
        assert reset_limit(handle) == 0
