# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Precision, tiling-budget and input-contract tests for fused_matmul."""

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cannbotdsl import get_mem_size

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "samples/matmul/fused_matmul")
)
from fused_matmul import FusedMatmulTiling, MatmulTiling, fused_matmul  # noqa: E402


_FP16, _BF16, _FP32 = torch.float16, torch.bfloat16, torch.float32
_SEED = 42
_MAX_ERROR_RATIO = 1e-2
# Tanh-approximation GELU coefficients, evaluated independently in CPU FP64.
_GELU_ALPHA, _GELU_BETA = -1.5957691, 0.044715


@dataclass(frozen=True)
class CaseSpec:
    m: int
    n: int
    k: int
    activation: str = "none"
    has_add: bool = False
    has_mul: bool = False
    in_dtype: torch.dtype = _FP16
    bias_dtype: torch.dtype | None = None
    out_dtype: torch.dtype | None = None


# activation, Add, Mul, FP16-input output, BF16-input output; None keeps input dtype.
_FEATURE_CONFIGS = [
    ("none", False, False, None, _FP32),
    ("none", True, False, _FP32, None),
    ("none", False, True, _FP32, None),
    ("none", True, True, None, _FP32),
    ("relu", False, False, _FP32, None),
    ("relu", True, False, None, _FP32),
    ("relu", False, True, None, _FP32),
    ("relu", True, True, _FP32, None),
    ("gelu", False, False, None, _FP32),
    ("gelu", True, False, _FP32, None),
    ("gelu", False, True, _FP32, None),
    ("gelu", True, True, None, _FP32),
]

_PRECISION_CASES = [
    pytest.param(
        CaseSpec(
            1024,
            1024,
            1024,
            activation,
            has_add,
            has_mul,
            in_dtype=dtype,
            out_dtype=output,
        ),
        id=f"{activation}-add{int(has_add)}-mul{int(has_mul)}-{dtype}-{output or dtype}",
    )
    for activation, has_add, has_mul, fp16_output, bf16_output in _FEATURE_CONFIGS
    for dtype, output in ((_FP16, fp16_output), (_BF16, bf16_output))
] + [
    pytest.param(CaseSpec(16, 32, 1024), id="cube-small"),
    pytest.param(
        CaseSpec(16, 32, 1024, "gelu", True, True, out_dtype=_FP32),
        id="vector-small-fp32",
    ),
    pytest.param(CaseSpec(17, 33, 65, "gelu", True, True), id="vector-mnk-tail"),
    pytest.param(CaseSpec(64, 128, 17, "gelu", True, True), id="vector-deep-k-tail"),
    pytest.param(
        CaseSpec(64, 17, 64, "gelu", True, True), id="vector-single-n-fractal"
    ),
    pytest.param(CaseSpec(1088, 1040, 1088), id="cube-n-k-tail"),
    pytest.param(CaseSpec(1088, 1056, 1088), id="cube-32-aligned-n-tail"),
    pytest.param(CaseSpec(1000, 1088, 1024), id="cube-m-n-tail"),
    pytest.param(
        CaseSpec(1088, 1088, 1088, "gelu", True, True), id="vector-base-block-tails"
    ),
    pytest.param(CaseSpec(2048, 1100, 512, "gelu"), id="vector-n-tail-unitflag"),
    pytest.param(
        CaseSpec(1088, 1088, 1088, "relu", True, False, out_dtype=_FP32),
        id="bias-multiwave-mnk-tail",
    ),
    pytest.param(CaseSpec(31, 33, 65, "none", True, False), id="bias-fp16-mnk-tail"),
    pytest.param(
        CaseSpec(130, 129, 257, "relu", True, False, in_dtype=_BF16),
        id="bias-bf16-mnk-tail",
    ),
    pytest.param(
        CaseSpec(17, 33, 65, "none", True, False, out_dtype=_FP32),
        id="bias-fp32-small-tail",
    ),
    pytest.param(
        CaseSpec(33, 17, 1024, "relu", True, False, out_dtype=_FP32),
        id="bias-relu-fp32-n-tail",
    ),
    pytest.param(
        CaseSpec(47, 65, 97, "none", True, False, bias_dtype=_BF16),
        id="bias-fp16-bf16-add",
    ),
    pytest.param(
        CaseSpec(
            1024,
            1024,
            256,
            "none",
            True,
            False,
            in_dtype=_BF16,
            bias_dtype=_FP16,
            out_dtype=_FP32,
        ),
        id="bias-bf16-fp16-add-fp32",
    ),
    pytest.param(
        CaseSpec(1024, 1024, 256, "gelu", True, True, in_dtype=_BF16, bias_dtype=_FP16),
        id="vector-bf16-fp16-vectors",
    ),
    pytest.param(
        CaseSpec(1024, 1024, 256, "relu", True, True, bias_dtype=_BF16),
        id="vector-fp16-bf16-vectors",
    ),
    pytest.param(CaseSpec(16, 4096, 1024, "gelu", True, True), id="skinny-m"),
    pytest.param(CaseSpec(4096, 16, 1024, "gelu", True, True), id="skinny-n"),
    pytest.param(CaseSpec(4096, 4096, 4096), id="large-cube", marks=pytest.mark.slow),
    pytest.param(
        CaseSpec(4096, 4096, 4096, "gelu", True, True),
        id="large-vector",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        CaseSpec(2047, 6145, 895, "gelu", True, True, out_dtype=_FP32),
        id="large-mnk-tail-fp32",
    ),
    pytest.param(
        CaseSpec(6145, 4097, 833, "gelu", True, True, in_dtype=_BF16),
        id="large-mnk-tail-bf16",
    ),
    pytest.param(CaseSpec(8192, 1793, 1280), id="large-cube-n-tail"),
    pytest.param(
        CaseSpec(16384, 769, 2304, "none", True, False),
        id="m-upper-bound",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        CaseSpec(16, 16384, 16384, "gelu", True, True),
        id="nk-upper-bound",
        marks=pytest.mark.slow,
    ),
]


def _golden(a, b, add, mul, activation, output_dtype):
    result = a.double() @ b.double().T
    if add is not None:
        result = result + add.double()
    if activation == "relu":
        result = result.clamp_min(0.0)
    elif activation == "gelu":
        result = result / (
            1.0 + torch.exp(_GELU_ALPHA * (result + _GELU_BETA * result.pow(3)))
        )
    if mul is not None:
        result = result * mul.double()
    return result.to(output_dtype)


def _assert_precision(actual, expected):
    assert actual.shape == expected.shape, "output shape mismatch"
    assert actual.dtype == expected.dtype, "output dtype mismatch"
    actual, expected = actual.detach().cpu().double(), expected.double()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), (
        "precision check requires finite output and golden"
    )
    error = (actual - expected).abs().max().item()
    threshold = _MAX_ERROR_RATIO * max(expected.abs().max().item(), 1.0)
    assert error < threshold, f"max|err|={error:.6e} >= threshold={threshold:.6e}"


@pytest.mark.npu
@pytest.mark.parametrize("spec", _PRECISION_CASES)
def test_fused_matmul_precision(spec):
    pytest.importorskip("torch_npu")
    generator = torch.Generator(device="cpu").manual_seed(_SEED)
    a = torch.randn((spec.m, spec.k), generator=generator, dtype=spec.in_dtype)
    b = torch.randn((spec.n, spec.k), generator=generator, dtype=spec.in_dtype)
    vector_dtype = spec.bias_dtype or spec.in_dtype
    add = (
        torch.randn(spec.n, generator=generator, dtype=vector_dtype)
        if spec.has_add
        else None
    )
    mul = (
        torch.randn(spec.n, generator=generator, dtype=vector_dtype)
        if spec.has_mul
        else None
    )
    expected = _golden(a, b, add, mul, spec.activation, spec.out_dtype or spec.in_dtype)
    actual = fused_matmul(
        a.npu(),
        b.npu(),
        add=add.npu() if add is not None else None,
        mul=mul.npu() if mul is not None else None,
        activation=spec.activation,
        out_dtype=spec.out_dtype,
    )
    torch.npu.synchronize()
    _assert_precision(actual, expected)
    if spec.activation == "relu" and not spec.has_mul:
        assert (actual.cpu() >= 0).all(), "ReLU emitted negative values"


@pytest.mark.parametrize("core_num", [28, 32])
@pytest.mark.parametrize(
    "shape,need_vec,out_fp32,has_bias",
    [
        pytest.param((16, 32, 1024), True, True, False, id="small-vector"),
        pytest.param((16, 16, 4096), False, True, True, id="bias-l1-budget"),
        pytest.param((1024, 1024, 1024), True, True, False, id="vector-fp32"),
        pytest.param((1024, 1088, 1024), False, False, False, id="cube-tail-balance"),
        pytest.param((129, 131, 97), True, False, False, id="mnk-tail"),
        pytest.param((4096, 4096, 4096), True, False, False, id="single-l0c-buffer"),
        pytest.param((16384, 16, 16384), True, False, False, id="skinny-n"),
        pytest.param((16, 16384, 16384), True, False, False, id="skinny-m"),
    ],
)
def test_tiling_budgets(shape, need_vec, out_fp32, has_bias, core_num, monkeypatch):
    stream = object()
    platform_query = Mock(return_value=SimpleNamespace(cube_core_num=core_num))
    monkeypatch.setattr(torch.npu, "current_stream", lambda: stream)
    monkeypatch.setattr(
        sys.modules[MatmulTiling.__module__], "get_platform_info", platform_query
    )
    tiling = FusedMatmulTiling(
        *shape,
        _FP16,
        _FP16,
        _FP32 if out_fp32 else _FP16,
        need_vec=need_vec,
        has_bias=has_bias,
    )
    platform_query.assert_called_once_with(stream=stream)
    input_bytes, accumulator_bytes, double_buffer = 2, 4, 2
    bias_n = (tiling.base_n + 15) // 16 * 16
    l1_bytes = (
        (tiling.base_m + tiling.base_n)
        * tiling.k_l1
        * input_bytes
        * tiling.l1_buffer_num
    )
    if has_bias:
        l1_bytes += bias_n * input_bytes * double_buffer
        assert bias_n * accumulator_bytes * double_buffer <= get_mem_size("bt")
    for memory, required in (
        ("l1", l1_bytes),
        ("l0a", tiling.base_m * tiling.base_k * input_bytes * double_buffer),
        ("l0b", tiling.base_n * tiling.base_k * input_bytes * double_buffer),
        ("l0c", tiling.base_m * tiling.base_n * accumulator_bytes * tiling.l0c_db),
    ):
        assert required <= get_mem_size(memory), f"{memory} requires {required} bytes"
    assert tiling.base_m % 16 == tiling.base_n % 16 == 0
    assert 1 <= tiling.used_core_num <= core_num
    if need_vec:
        assert tiling.base_m >= 32
        assert tiling.tile_vec_m == tiling.base_m // 2
        ub_bytes = (
            tiling.tile_vec_m * tiling.base_n * accumulator_bytes * tiling.cv_buffer_num
            + tiling.base_n * input_bytes * 4
        )
        assert ub_bytes <= get_mem_size("ub") - tiling.UB_RESERVED_BYTES
    m_tiles = (shape[0] + tiling.base_m - 1) // tiling.base_m
    n_tiles = (shape[1] + tiling.base_n - 1) // tiling.base_n
    total_tiles = m_tiles * n_tiles
    if total_tiles > core_num and total_tiles % core_num:
        imbalance = ((total_tiles + core_num - 1) // core_num) / (
            total_tiles / core_num
        )
        assert imbalance <= tiling.CORE_LOAD_IMBALANCE_LIMIT + 1e-9


@pytest.mark.parametrize(
    "actual_value,golden_value",
    [
        pytest.param(float("nan"), 0.0, id="output-nan"),
        pytest.param(float("inf"), 0.0, id="output-inf"),
        pytest.param(0.0, float("nan"), id="golden-nan"),
        pytest.param(0.0, float("inf"), id="golden-inf"),
    ],
)
def test_precision_rejects_nonfinite(actual_value, golden_value):
    with pytest.raises(AssertionError, match="finite output and golden"):
        _assert_precision(
            torch.full((16, 16), actual_value, dtype=_FP16),
            torch.full((16, 16), golden_value, dtype=_FP16),
        )


@pytest.mark.parametrize(
    "updates,error,message",
    [
        pytest.param(
            {"b": torch.empty(64, 64, dtype=_BF16)},
            TypeError,
            "same dtype",
            id="input-dtypes",
        ),
        pytest.param(
            {"a": torch.empty(1, 64, 64, dtype=_FP16)}, ValueError, "2-D", id="a-rank"
        ),
        pytest.param(
            {"b": torch.empty(1, 64, 64, dtype=_FP16)}, ValueError, "2-D", id="b-rank"
        ),
        pytest.param(
            {
                "a": torch.empty(64, 64, dtype=_FP32),
                "b": torch.empty(64, 64, dtype=_FP32),
            },
            TypeError,
            "fp16/bf16",
            id="fp32-inputs",
        ),
        pytest.param({"activation": "silu"}, ValueError, "activation", id="activation"),
        pytest.param(
            {
                "a": torch.empty(64, 64, dtype=_BF16),
                "b": torch.empty(64, 64, dtype=_BF16),
                "out_dtype": _FP16,
            },
            TypeError,
            "out_dtype",
            id="bf16-fp16-output",
        ),
        pytest.param(
            {"out_dtype": _BF16}, TypeError, "out_dtype", id="fp16-bf16-output"
        ),
        pytest.param(
            {"out_dtype": torch.int32}, TypeError, "out_dtype", id="int-output"
        ),
        pytest.param(
            {"a": torch.empty(15, 64, dtype=_FP16)},
            ValueError,
            "outside supported range",
            id="m-below-min",
        ),
        pytest.param(
            {
                "a": torch.empty(64, 16385, dtype=_FP16, device="meta"),
                "b": torch.empty(64, 16385, dtype=_FP16, device="meta"),
            },
            ValueError,
            "outside supported range",
            id="k-above-max",
        ),
        pytest.param(
            {"add": torch.empty(63, dtype=_FP16)},
            ValueError,
            "add must have shape",
            id="add-shape",
        ),
        pytest.param(
            {"add": torch.empty(64, dtype=_FP32)},
            TypeError,
            "fp16/bf16",
            id="add-dtype",
        ),
        pytest.param(
            {"mul": torch.empty(1, 64, dtype=_FP16)},
            ValueError,
            "mul must have shape",
            id="mul-shape",
        ),
        pytest.param(
            {"mul": torch.empty(64, dtype=_FP32)},
            TypeError,
            "fp16/bf16",
            id="mul-dtype",
        ),
        pytest.param(
            {"b": torch.empty(64, 32, dtype=_FP16)},
            ValueError,
            "reduction mismatch",
            id="k-mismatch",
        ),
        pytest.param(
            {"a": torch.empty(64, 64, dtype=_FP16).T},
            ValueError,
            "a must be contiguous",
            id="a-stride",
        ),
        pytest.param(
            {"b": torch.empty(64, 64, dtype=_FP16).T},
            ValueError,
            "b must be contiguous",
            id="b-stride",
        ),
        pytest.param(
            {"b": torch.empty(64, 64, dtype=_FP16, device="meta")},
            ValueError,
            "same device",
            id="b-device",
        ),
        pytest.param(
            {"add": torch.empty(64, dtype=_FP16, device="meta")},
            ValueError,
            "same device",
            id="add-device",
        ),
        pytest.param(
            {"mul": torch.empty(64, dtype=_FP16, device="meta")},
            ValueError,
            "same device",
            id="mul-device",
        ),
        pytest.param({}, ValueError, "NPU tensors", id="cpu-inputs"),
    ],
)
def test_reject_invalid_inputs(updates, error, message):
    inputs = {
        "a": torch.empty(64, 64, dtype=_FP16),
        "b": torch.empty(64, 64, dtype=_FP16),
    }
    inputs.update(updates)
    with pytest.raises(error, match=message):
        fused_matmul(**inputs)
