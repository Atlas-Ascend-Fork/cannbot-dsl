# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TP1 native precision, graph replay and tiling tests."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys
import logging
import inspect
import argparse
import json
from importlib.metadata import version

import pytest
import torch

pytest.importorskip("torch_npu")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "samples/attn_epilogue"))
import attn_epilogue as tp1
import attn_epilogue_common as common

torch_npu = sys.modules["torch_npu"]

NG, F, O_LORA, DIM = 8, 4096, 1024, 5120


def wrapper_arguments():
    t = 72
    m, k = 80, 8192
    rows = 8 * m
    specs = {
        "o": ((8 * t, 4096), torch.bfloat16),
        "rope_cos": ((t, 64), torch.float32),
        "rope_sin": ((t, 64), torch.float32),
        "woaq": ((8192, 4096), torch.float8_e4m3fn),
        "dwa": ((8192, 64, 2), torch.uint8),
        "wobq": ((5120, 8192), torch.float8_e4m3fn),
        "dwb": ((5120, 128, 2), torch.uint8),
        "aq": ((rows, 4096), torch.uint8),
        "asc": ((rows, 128), torch.uint8),
        "y": ((m, k), torch.bfloat16),
        "yq": ((m, k), torch.uint8),
        "ysc": ((m, k // 32), torch.uint8),
        "out": ((m, 5120), torch.bfloat16),
    }
    args = {
        name: torch.empty(shape, dtype=dtype, device="meta")
        for name, (shape, dtype) in specs.items()
    }
    return args


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "weight_shape",
        "scale_dtype",
        "stride",
        "device",
        "scratch",
        "output",
        "input_rank",
    ],
)
def test_wrapper_contract(monkeypatch, fault):
    wrapper = tp1._attn_epilogue_out
    compiled = Mock()
    monkeypatch.setattr(tp1, "core_budget", lambda _: (0, 28, 56))
    monkeypatch.setattr(tp1, "_compiled", compiled)
    args = wrapper_arguments()
    if fault == "weight_shape":
        args["woaq"] = args["woaq"][:, :-1]
    elif fault == "scale_dtype":
        args["dwb"] = args["dwb"].to(torch.int8)
    elif fault == "stride":
        args["woaq"] = torch.empty(
            (4096, 8192), dtype=torch.float8_e4m3fn, device="meta"
        ).t()
    elif fault == "device":
        args["rope_cos"] = torch.empty((72, 64), dtype=torch.float32)
    elif fault == "scratch":
        args["ysc"] = args["ysc"][:-1]
    elif fault == "output":
        args["out"] = args["out"][:, :-1]
    elif fault == "input_rank":
        args["o"] = args["o"].reshape(-1)
    if fault == "none":
        assert wrapper(**args) is args["out"]
        compiled.assert_called_once()
        compiled.return_value.assert_called_once()
    else:
        with pytest.raises((ValueError, TypeError)):
            wrapper(**args)
        compiled.assert_not_called()


def input_fixture(t, scale=1.0):
    generator = torch.Generator().manual_seed(2026)
    return (torch.randn(t, NG, F, generator=generator) * (0.3 * scale)).to(
        torch.bfloat16
    )


def weight_fixture(seed=2026):
    """Unquantized projection weights for the native reference."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    woa = (torch.randn(8, 1024, 4096, generator=gen) * 0.3).to(torch.bfloat16)
    wob = (torch.randn(5120, 8192, generator=gen) * 0.2).to(torch.bfloat16)
    return woa, wob


def native_projection(a, wa, wb):
    """Native MX projection with intermediate BF16 rounding; CPU inputs."""

    m, groups, k = a.shape
    n = wa.shape[1]
    e8 = torch.float8_e8m0fnu
    aq, asc = torch_npu.npu_dynamic_mx_quant(a.reshape(-1, k).npu(), dst_type=292)
    wq, ws = torch_npu.npu_dynamic_mx_quant(wa.reshape(-1, k).npu(), dst_type=292)
    y = torch_npu.npu_transpose_quant_batchmatmul(
        aq.reshape(m, groups, k).npu(),
        wq.reshape(groups, n, k).npu(),
        15,
        x1_scale=asc.reshape(m, groups, k // 64, 2).npu().view(e8),
        x2_scale=ws.reshape(groups, n, k // 64, 2).npu().view(e8),
        group_sizes=[1, 1, 32],
        perm_x1=[1, 0, 2],
        perm_x2=[0, 2, 1],
        perm_y=[1, 0, 2],
    ).reshape(m, groups * n)
    yq, ys = torch_npu.npu_dynamic_mx_quant(y, dst_type=292)
    wq, ws = torch_npu.npu_dynamic_mx_quant(wb.npu(), dst_type=292)
    return torch_npu.npu_quant_matmul(
        yq,
        wq.npu().t(),
        ws.npu().view(e8).transpose(0, 1),
        pertoken_scale=ys.view(e8),
        output_dtype=torch.bfloat16,
        group_sizes=[1, 1, 32],
    ).cpu()


def rope_cos_sin(num_token, rope_dim=64, base=10000.0):
    """Pair-repeated (T, Dr) cos/sin as the operator contract expects."""
    j = torch.arange(0, rope_dim, 2, dtype=torch.float32)
    theta = 1.0 / (base ** (j / rope_dim))
    freqs = torch.outer(
        torch.arange(num_token, dtype=torch.float32), theta
    )  # (T, Dr/2)
    cos = torch.empty(num_token, rope_dim, dtype=torch.float32)
    sin = torch.empty(num_token, rope_dim, dtype=torch.float32)
    cos[:, 0::2] = freqs.cos()
    cos[:, 1::2] = freqs.cos()
    sin[:, 0::2] = freqs.sin()
    sin[:, 1::2] = freqs.sin()
    return freqs, cos.contiguous(), sin.contiguous()


def inverse_rope_golden(
    o: torch.Tensor, freqs: torch.Tensor, nope_dim: int
) -> torch.Tensor:
    """fp32 inverse RoPE on the last ``Dr`` dims. o (T, N, D) bf16 -> fp32."""
    x = o.float()
    a = x[..., nope_dim + 0 :: 2]
    b = x[..., nope_dim + 1 :: 2]
    c = freqs.cos()[:, None, :]
    s = freqs.sin()[:, None, :]
    out = x.clone()
    out[..., nope_dim + 0 :: 2] = a * c + b * s
    out[..., nope_dim + 1 :: 2] = b * c - a * s
    return out


def rotated_input(x, freqs):
    t = x.shape[0]
    return (
        inverse_rope_golden(x.reshape(t, 64, 512), freqs, 448)
        .reshape(t, NG, F)
        .to(torch.bfloat16)
    )


def quantized_weights(wa, wb):
    aq, asc = torch_npu.npu_dynamic_mx_quant(wa.reshape(-1, F).npu(), dst_type=292)
    bq, bsc = torch_npu.npu_dynamic_mx_quant(wb.npu(), dst_type=292)
    return aq, asc.view(torch.uint8), bq, bsc.view(torch.uint8)


@pytest.fixture(scope="module")
def weights():
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(0)
    wa, wb = weight_fixture()
    return wa, wb, quantized_weights(wa, wb)


@pytest.mark.parametrize(
    "cube,vector,expected", [(28, 56, 28), (32, 64, 32), (8, 16, 8), (32, 48, 24)]
)
def test_current_stream_budget(cube, vector, expected):
    ref = SimpleNamespace(device=SimpleNamespace(type="npu", index=3))
    stream = object()
    with (
        patch.object(torch.npu, "current_stream", return_value=stream) as current,
        patch.object(
            tp1.cannbotdsl,
            "get_platform_info",
            return_value=SimpleNamespace(cube_core_num=cube, vector_core_num=vector),
        ) as query,
    ):
        assert tp1.core_budget(ref) == (3, expected, vector)
        current.assert_called_once_with(3)
        query.assert_called_once_with(stream=stream)


def test_changed_quota_is_not_cached():
    ref = SimpleNamespace(device=SimpleNamespace(type="npu", index=0))
    with (
        patch.object(torch.npu, "current_stream"),
        patch.object(
            tp1.cannbotdsl,
            "get_platform_info",
            side_effect=[
                SimpleNamespace(cube_core_num=28, vector_core_num=56),
                SimpleNamespace(cube_core_num=8, vector_core_num=16),
            ],
        ),
    ):
        assert tp1.core_budget(ref)[1:] == (28, 56)
        assert tp1.core_budget(ref)[1:] == (8, 16)


@pytest.mark.parametrize("cube,vector", [(0, 56), (28, 0), (-1, 56)])
def test_invalid_quota(cube, vector):
    ref = SimpleNamespace(device=SimpleNamespace(type="npu", index=0))
    with (
        patch.object(torch.npu, "current_stream"),
        patch.object(
            tp1.cannbotdsl,
            "get_platform_info",
            return_value=SimpleNamespace(cube_core_num=cube, vector_core_num=vector),
        ),
    ):
        with pytest.raises(RuntimeError):
            tp1.core_budget(ref)


def test_tiling_capacity_and_padding_independence():
    for cores in (8, 28, 32):
        for t in range(1, 257):
            m, n1, n2 = tp1.tiling_for_cores(cores, t)
            n = max(n1, n2)
            assert all(value > 0 and value % 16 == 0 for value in (m, n1, n2))
            assert m * common.BASE_K * 2 <= 65536
            assert n * common.BASE_K * 2 <= 65536
            step, depth1, depth2 = tp1._matmul_resources(m, n1, n2)
            assert m * n1 * 4 * depth1 <= 262144
            assert m * n2 * 4 * depth2 <= 262144
            assert (m + n) * (common.BASE_K * step * 2 + 256) <= 524288
            padded = (t + 15) // 16 * 16
            assert (m, n1, n2) == tp1.tiling_for_cores(cores, t, padded + 16)


@pytest.mark.parametrize(
    "cores,t,mpad", [(0, 72, 80), (32, 0, 16), (32, 72, 64), (32, 72, 81)]
)
def test_invalid_tiling(cores, t, mpad):
    with pytest.raises(ValueError):
        tp1.tiling_for_cores(cores, t, mpad)


def check_intermediate_quantization(rotated, quantized, activation, intermediate):
    t = rotated.shape[0]
    aq, asc = activation
    yq, ysc = intermediate
    m = yq.shape[0]
    reference_q, reference_s = torch_npu.npu_dynamic_mx_quant(
        rotated.permute(1, 0, 2).reshape(NG * t, F).npu(), dst_type=292
    )
    assert torch.equal(
        aq.view(NG, m, F)[:, :t].cpu(),
        reference_q.view(torch.uint8).view(NG, t, F).cpu(),
    )
    assert torch.equal(
        asc.view(NG, m, F // 32)[:, :t].cpu(),
        reference_s.view(torch.uint8).view(NG, t, F // 32).cpu(),
    )

    # Check the C->V handoff's data and scale layout before MM2 can mask a
    # quantization error. Cover split-M tails, local N blocks and padding.
    waq, was, _, _ = quantized
    native_aq, native_as = torch_npu.npu_dynamic_mx_quant(rotated.npu(), dst_type=292)
    native_y = torch_npu.npu_transpose_quant_batchmatmul(
        native_aq,
        waq.view(NG, O_LORA, F),
        15,
        x1_scale=native_as.view(torch.float8_e8m0fnu),
        x2_scale=was.view(torch.float8_e8m0fnu).view(NG, O_LORA, F // 64, 2),
        group_sizes=[1, 1, 32],
        perm_x1=[1, 0, 2],
        perm_x2=[0, 2, 1],
        perm_y=[1, 0, 2],
    ).view(t, NG * O_LORA)
    native_yq, native_ys = torch_npu.npu_dynamic_mx_quant(native_y, dst_type=292)
    assert torch.equal(yq[:t].cpu(), native_yq.view(torch.uint8).cpu())
    assert torch.equal(ysc[:t].cpu(), native_ys.view(torch.uint8).view(t, -1).cpu())


@pytest.mark.npu
@pytest.mark.parametrize(
    "t,padding,scale",
    [
        (1, 0, 1.0),
        (33, 0, 1.0),
        (72, 16, 1.0),
        (129, 0, 1.0),
        (193, 0, 1.0),
        (255, 0, 1.0),
        (257, 16, 1.0),
        (1000, 0, 1.0),
        (72, 0, 0.0),
        (256, 0, 1e-37),
    ],
)
def test_tp1_native_and_replay(weights, t, padding, scale):
    wa, wb, quantized = weights
    m = (t + 15) // 16 * 16 + padding
    x = input_fixture(t, scale=scale)
    freqs, cos, sin = rope_cos_sin(t)
    rotated = rotated_input(x, freqs)
    expected = [native_projection(rotated, wa, wb), native_projection(-rotated, wa, wb)]
    o, cos, sin = x.reshape(t * NG, F).npu(), cos.npu(), sin.npu()
    device = o.device
    aq = torch.zeros((NG * m, F), dtype=torch.uint8, device=device)
    asc = torch.zeros((NG * m, F // 32), dtype=torch.uint8, device=device)
    y = torch.zeros((m, NG * O_LORA), dtype=torch.bfloat16, device=device)
    yq = torch.zeros_like(y, dtype=torch.uint8)
    ysc = torch.zeros((m, NG * O_LORA // 32), dtype=torch.uint8, device=device)
    output = torch.zeros((m, DIM), dtype=torch.bfloat16, device=device)
    args = (o, cos, sin, *quantized, aq, asc, y, yq, ysc, output)
    tp1._attn_epilogue_out(*args)
    torch.npu.synchronize()
    torch.testing.assert_close(output[:t].cpu(), expected[0], rtol=0, atol=0)

    check_intermediate_quantization(rotated, quantized, (aq, asc), (yq, ysc))

    # Snapshot before the next launch overwrites shared output.
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        snapshots = []
        for _ in range(2):
            o.neg_()
            tp1._attn_epilogue_out(*args)
            snapshots.append(output[:t].clone())
    for _ in range(2):
        graph.replay()
        torch.npu.synchronize()
        for i, snapshot in enumerate(snapshots):
            torch.testing.assert_close(
                snapshot.cpu(), expected[(i + 1) % 2], rtol=0, atol=0
            )


@pytest.mark.npu
def test_tp1_independent_stage_tile_storage(weights, monkeypatch):
    """Check unequal-N storage reuse with the actual launch quota."""
    t = 72
    select = tp1.tiling_for_cores
    assert select(32, t)[1] != select(32, t)[2]
    monkeypatch.setattr(
        tp1, "tiling_for_cores", lambda cores, t=72, m_pad=None: select(32, t, m_pad)
    )
    test_tp1_native_and_replay(weights, t, 0, 1.0)


def test_public_signature():
    assert list(inspect.signature(tp1.attn_epilogue).parameters) == [
        "o",
        "woa",
        "wob",
        "descale_woa",
        "descale_wob",
        "rope_sin",
        "rope_cos",
    ]


@pytest.mark.npu
@pytest.mark.parametrize("t", [1, 72, 256, 512, 1024])
def test_public_tp1(weights, t):
    wa, wb, quantized = weights
    waq, sa, wbq, sb = quantized
    x = input_fixture(t)
    freqs, cos, sin = rope_cos_sin(t)
    expected = [native_projection(rotated_input(v, freqs), wa, wb) for v in (x, -x)]
    o = x.reshape(t, 64, 512).npu()
    args = (
        o,
        waq.reshape(8, 1024, 4096),
        wbq,
        sa.view(torch.float8_e8m0fnu).reshape(8, 1024, 64, 2),
        sb.view(torch.float8_e8m0fnu),
        sin.npu(),
        cos.npu(),
    )
    out = tp1.attn_epilogue(*args)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), expected[0], rtol=0, atol=0)
    cache_ids = {
        key: tuple(v.data_ptr() for v in values)
        for key, values in common._SCRATCH.items()
    }
    o.neg_()
    other = tp1.attn_epilogue(*args)
    assert out.data_ptr() != other.data_ptr()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), expected[0], rtol=0, atol=0)
    torch.testing.assert_close(other.cpu(), expected[1], rtol=0, atol=0)
    assert cache_ids == {
        key: tuple(v.data_ptr() for v in values)
        for key, values in common._SCRATCH.items()
    }

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        o.neg_()
        first = tp1.attn_epilogue(*args)
        o.neg_()
        second = tp1.attn_epilogue(*args)
    assert first.data_ptr() != second.data_ptr()
    for _ in range(2):
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(first.cpu(), expected[0], rtol=0, atol=0)
        torch.testing.assert_close(second.cpu(), expected[1], rtol=0, atol=0)


@pytest.mark.npu
def test_public_validation(weights):
    _, _, (wa, sa, wb, sb) = weights
    o = torch.zeros((72, 64, 512), device=wa.device, dtype=torch.bfloat16)
    table = torch.zeros((72, 64), device=wa.device)
    args = [
        o,
        wa.view(8, 1024, 4096),
        wb,
        sa.view(torch.float8_e8m0fnu).view(8, 1024, 64, 2),
        sb.view(torch.float8_e8m0fnu),
        table,
        table,
    ]
    for index, value in [(0, o.reshape(576, 4096)), (1, wa), (3, sa)]:
        bad = args.copy()
        bad[index] = value
        with pytest.raises((TypeError, ValueError)):
            tp1.attn_epilogue(*bad)
    bad = args.copy()
    bad[2] = torch_npu.npu_format_cast(wb, 29)
    with pytest.raises(ValueError, match="ND storage"):
        tp1.attn_epilogue(*bad)


def benchmark_inputs(t, mode):
    torch.npu.set_device(0)
    wa, wb = weight_fixture()
    waq, sa, wbq, sb = quantized_weights(wa, wb)
    x = input_fixture(t)
    freqs, cos, sin = rope_cos_sin(t)
    expected = native_projection(rotated_input(x, freqs), wa, wb)
    o = x.reshape(t, 64, 512).npu()
    cos, sin = cos.npu(), sin.npu()
    m = (t + 15) // 16 * 16
    specs = [
        ((8 * m, 4096), torch.uint8),
        ((8 * m, 128), torch.uint8),
        ((m, 8192), torch.bfloat16),
        ((m, 8192), torch.uint8),
        ((m, 256), torch.uint8),
        ((m, 5120), torch.bfloat16),
    ]
    buffers = tuple(
        torch.empty(shape, dtype=dtype, device="npu") for shape, dtype in specs
    )
    public_args = (
        o,
        waq.view(8, 1024, 4096),
        wbq,
        sa.view(torch.float8_e8m0fnu).view(8, 1024, 64, 2),
        sb.view(torch.float8_e8m0fnu),
        sin,
        cos,
    )

    def call():
        if mode == "public":
            return tp1.attn_epilogue(*public_args)
        return tp1._attn_epilogue_out(
            o.view(t * 8, 4096),
            cos,
            sin,
            waq,
            sa,
            wbq,
            sb,
            *buffers,
        )[:t]

    return call, expected, o


def check_benchmark_replays(call, expected):
    for _ in range(10):
        out = call()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    eviction = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float32, device="npu")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = []
        for _ in range(16):
            sink = eviction.sum()
            outputs.append(call())
    for _ in range(6):
        graph.replay()
        torch.npu.synchronize()
    assert sink.item() == eviction.numel()
    for out in outputs:
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def benchmark():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("public", "out"), default="public")
    parser.add_argument("--tokens", type=int, default=72)
    args = parser.parse_args()
    t = args.tokens
    if t < 1:
        parser.error("--tokens must be positive")
    call, expected, o = benchmark_inputs(t, args.mode)
    check_benchmark_replays(call, expected)
    device, cube, vector = tp1.core_budget(o)
    logging.info(
        json.dumps(
            dict(
                mode=args.mode,
                tokens=t,
                device=device,
                aic=cube,
                aiv=vector,
                tiling=tp1.tiling_for_cores(cube, t),
                calls=96,
                cannbotdsl=version("cannbotdsl"),
                exact=True,
            )
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    benchmark()
