# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Precision tests for the unified MXFP8/MXFP4 QBMM kernel.

Formula: Y[M,N] = dequant(X1)[M,K] @ dequant(X2)[K,N]

The golden paths decode the public rank-3 E8M0 paired-scale tensors on CPU.
MXFP4 additionally quantizes and decodes packed E2M1 data independently of
the NPU kernel.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "samples",
        "matmul",
        "quant_matmul",
    ),
)

from quant_batch_matmul_mx import (  # noqa: E402
    MX_GROUP_SIZE,
    MX_SCALE_PAIR,
    ceil_div,
    get_scale_k_len,
    npu_quant_matmul,
)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


_E4 = torch.float8_e4m3fn
_E5 = torch.float8_e5m2
_FP4X2 = torch.float4_e2m1fn_x2


def _mx_case(
    m,
    n,
    k,
    *,
    case_id,
    a_dtype=_E4,
    b_dtype=None,
    output_dtype=torch.float32,
    transpose_a=False,
    transpose_b=True,
    has_bias=False,
):
    return pytest.param(
        m,
        n,
        k,
        a_dtype,
        a_dtype if b_dtype is None else b_dtype,
        output_dtype,
        transpose_a,
        transpose_b,
        has_bias,
        id=case_id,
    )


_MXFP8_CASES = [
    _mx_case(256, 256, 256, case_id="aligned-ft"),
    _mx_case(130, 130, 70, case_id="mnk-tail-ft"),
    _mx_case(1536, 31, 384, case_id="n-tail-ft"),
    _mx_case(15, 256, 192, case_id="m-tail-ff", transpose_b=False),
    _mx_case(31, 64, 320, case_id="narrow-m-tail-ff", transpose_b=False),
    _mx_case(31, 256, 1024, case_id="mtail-m31-tt", transpose_a=True),
    _mx_case(
        256, 256, 256, case_id="e4-e5-bf16", b_dtype=_E5, output_dtype=torch.bfloat16
    ),
    _mx_case(
        256,
        256,
        256,
        case_id="e5-e4-fp16",
        a_dtype=_E5,
        b_dtype=_E4,
        output_dtype=torch.float16,
    ),
    _mx_case(128, 128, 96, case_id="k-96-group-tail-ff", transpose_b=False),
    _mx_case(1024, 1024, 1056, case_id="kl1-remainer-ff", transpose_b=False),
    _mx_case(1024, 1024, 2048, case_id="scalekl1-full-cover-ff", transpose_b=False),
    _mx_case(960, 1088, 1024, case_id="kl1-mn-tail-ff", transpose_b=False),
    _mx_case(512, 512, 320, case_id="k-320-stepk2-ff", transpose_b=False),
    _mx_case(256, 256, 4096, case_id="scalekl1-cache-ff", transpose_b=False),
    _mx_case(256, 15, 1024, case_id="tt-n-tail-kl1", transpose_a=True),
    _mx_case(
        15, 256, 1024, case_id="tf-m-tail-kl1", transpose_a=True, transpose_b=False
    ),
    _mx_case(128, 130, 128, case_id="bias-n-tail-ft", has_bias=True),
    _mx_case(64, 16384, 256, case_id="bias-a-full-load-ft", has_bias=True),
    _mx_case(
        72, 25600, 6144, case_id="streamed-panels-bf16-ft", output_dtype=torch.bfloat16
    ),
    _mx_case(
        4097,
        5118,
        8064,
        case_id="panels-mnk-tail-fp16-ft",
        b_dtype=_E5,
        output_dtype=torch.float16,
    ),
]

_MXFP4_CASES = [
    _mx_case(4096, 4096, 4098, a_dtype=_FP4X2, case_id="k-tail-4098"),
    _mx_case(4001, 4002, 2048, a_dtype=_FP4X2, case_id="bias-mn-tail", has_bias=True),
    _mx_case(
        4093, 3997, 3110, a_dtype=_FP4X2, case_id="odd-fp16", output_dtype=torch.float16
    ),
    _mx_case(17, 4096, 2048, a_dtype=_FP4X2, case_id="small-m-bias", has_bias=True),
    _mx_case(4096, 4096, 2, a_dtype=_FP4X2, case_id="min-k"),
    _mx_case(64, 16382, 250, a_dtype=_FP4X2, case_id="fl-tails-bias", has_bias=True),
    _mx_case(
        4000,
        4096,
        2048,
        a_dtype=_FP4X2,
        case_id="tf-bias-m-tail",
        transpose_a=True,
        transpose_b=False,
        has_bias=True,
    ),
    _mx_case(
        4000,
        4096,
        2048,
        a_dtype=_FP4X2,
        case_id="tt-bf16-m-tail",
        output_dtype=torch.bfloat16,
        transpose_a=True,
    ),
    _mx_case(
        4096,
        3072,
        96,
        a_dtype=_FP4X2,
        case_id="tf-k-single-tail-96",
        transpose_a=True,
        transpose_b=False,
    ),
    _mx_case(
        4096,
        3072,
        300,
        a_dtype=_FP4X2,
        case_id="ff-k-mixed-tail-300",
        transpose_b=False,
    ),
    _mx_case(
        4096,
        3072,
        136,
        a_dtype=_FP4X2,
        case_id="tf-k-mixed-tail-136",
        transpose_a=True,
        transpose_b=False,
    ),
    _mx_case(
        4096, 3072, 440, a_dtype=_FP4X2, case_id="tt-k-mixed-tail-440", transpose_a=True
    ),
    _mx_case(
        4096,
        32,
        2048,
        a_dtype=_FP4X2,
        case_id="ff-n32-bias",
        transpose_b=False,
        has_bias=True,
    ),
    _mx_case(4096, 34, 2048, a_dtype=_FP4X2, case_id="ft-n34-bias", has_bias=True),
    _mx_case(
        4096,
        63,
        2048,
        a_dtype=_FP4X2,
        case_id="tt-n63-bias",
        transpose_a=True,
        has_bias=True,
    ),
    _mx_case(
        510,
        4002,
        3104,
        a_dtype=_FP4X2,
        case_id="ff-mnk-tail-510-4002-3104",
        transpose_b=False,
    ),
    _mx_case(510, 4002, 3104, a_dtype=_FP4X2, case_id="ft-mnk-tail-510-4002-3104"),
    _mx_case(
        3998,
        4094,
        2051,
        a_dtype=_FP4X2,
        case_id="tf-m-tail-k2051",
        transpose_a=True,
        transpose_b=False,
    ),
    _mx_case(
        3998,
        4093,
        2048,
        a_dtype=_FP4X2,
        case_id="tt-mn-tail-3998-4093",
        transpose_a=True,
    ),
]


def _make_paired_scale(
    outer_size: int,
    scale_k_len: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Build public E8M0 storage with shape [outer,G,2]."""
    exponents = torch.randint(
        -2,
        3,
        (outer_size, scale_k_len),
        dtype=torch.int16,
        generator=generator,
    )
    return (
        (exponents + 127)
        .to(torch.uint8)
        .view(torch.int8)
        .reshape(outer_size, scale_k_len // MX_SCALE_PAIR, MX_SCALE_PAIR)
        .contiguous()
    )


@dataclasses.dataclass
class MxInputs:
    a: torch.Tensor
    b: torch.Tensor
    scale_a: torch.Tensor
    scale_b: torch.Tensor
    transpose_a: bool
    transpose_b: bool
    bias: torch.Tensor | None = None


def _make_mxfp8_inputs(
    m: int,
    k: int,
    n: int,
    *,
    transpose_a: bool = False,
    transpose_b: bool = True,
    a_dtype: torch.dtype = _E4,
    b_dtype: torch.dtype | None = None,
    seed: int = 42,
    has_bias: bool = False,
) -> MxInputs:
    """Build deterministic X1/X2 and public rank-3 paired-scale tensors."""
    b_dtype = a_dtype if b_dtype is None else b_dtype
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a_logical = (torch.randn((m, k), generator=generator) * 0.5).to(a_dtype)
    b_logical = (torch.randn((n, k), generator=generator) * 0.5).to(b_dtype)
    scale_k_len = get_scale_k_len(k)
    scale_a_and = _make_paired_scale(m, scale_k_len, generator=generator)
    scale_b_bdn = _make_paired_scale(n, scale_k_len, generator=generator)
    bias = (
        (torch.randn((n,), generator=generator) * 0.5).to(torch.float32)
        if has_bias
        else None
    )

    return MxInputs(
        a=a_logical.T.contiguous() if transpose_a else a_logical,
        b=b_logical if transpose_b else b_logical.T.contiguous(),
        scale_a=(
            scale_a_and.permute(1, 0, 2).contiguous() if transpose_a else scale_a_and
        ),
        scale_b=(
            scale_b_bdn if transpose_b else scale_b_bdn.permute(1, 0, 2).contiguous()
        ),
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        bias=bias,
    )


_E2M1_MAGNITUDES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
_E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _quantize_to_mxfp4(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [rows,K] FP32 data to MXFP4 (E2M1 nibbles + E8M0 scale bytes,
    one scale per 32-element group along K)."""
    rows, k = tensor.shape
    group_count = ceil_div(k, MX_GROUP_SIZE)
    padded = torch.zeros(rows, group_count * MX_GROUP_SIZE)
    padded[:, :k] = tensor

    scale_bytes = torch.zeros(rows, group_count, dtype=torch.uint8)
    nibbles = torch.zeros(rows, group_count * MX_GROUP_SIZE, dtype=torch.uint8)
    midpoints = (_E2M1_MAGNITUDES[1:] + _E2M1_MAGNITUDES[:-1]) / 2
    for group in range(group_count):
        begin = group * MX_GROUP_SIZE
        segment = padded[:, begin : begin + MX_GROUP_SIZE]
        amax = segment.abs().amax(dim=1, keepdim=True)
        exponent = (
            torch.where(
                amax > 0,
                torch.ceil(torch.log2(amax / 6.0)),
                torch.zeros_like(amax),
            )
            .clamp(-127, 127)
            .to(torch.int32)
        )
        scale = torch.pow(2.0, exponent.float())
        code = torch.bucketize((segment / scale).abs().contiguous(), midpoints)
        sign = (segment < 0).to(torch.uint8)
        nibbles[:, begin : begin + MX_GROUP_SIZE] = code.to(torch.uint8) + sign * 8
        scale_bytes[:, group] = (exponent + 127).to(torch.uint8).squeeze(1)
    return nibbles[:, :k].contiguous(), scale_bytes


def _pack_mxfp4_cols(nibbles: torch.Tensor) -> torch.Tensor:
    """Pack the innermost axis, two E2M1 values per byte."""
    return (
        nibbles[:, 0::2].to(torch.uint8) | (nibbles[:, 1::2].to(torch.uint8) << 4)
    ).contiguous()


def _pack_mxfp4_rows(nibbles: torch.Tensor) -> torch.Tensor:
    """Transpose, then pack the original outer axis two values per byte."""
    transposed = nibbles.T.contiguous()
    return (
        transposed[:, 0::2].to(torch.uint8) | (transposed[:, 1::2].to(torch.uint8) << 4)
    ).contiguous()


def _pair_e8m0_scales(scale_bytes: torch.Tensor) -> torch.Tensor:
    """Convert [outer,ceil(K/32)] scales to [outer,ceil(K/64),2]."""
    rows, scale_count = scale_bytes.shape
    pair_count = ceil_div(scale_count, MX_SCALE_PAIR)
    paired = torch.zeros(rows, pair_count * MX_SCALE_PAIR, dtype=torch.uint8)
    paired[:, :scale_count] = scale_bytes
    return paired.reshape(rows, pair_count, MX_SCALE_PAIR).contiguous()


def _make_mxfp4_inputs(
    m: int,
    k: int,
    n: int,
    *,
    transpose_a: bool = False,
    transpose_b: bool = True,
    seed: int = 42,
    has_bias: bool = False,
) -> MxInputs:
    """Build packed E2M1 data and public paired E8M0 scales."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a_float = (torch.randn((m, k), generator=generator) * 0.5).float()
    b_float = (torch.randn((n, k), generator=generator) * 0.5).float()
    a_nibbles, a_scale_bytes = _quantize_to_mxfp4(a_float)
    b_nibbles, b_scale_bytes = _quantize_to_mxfp4(b_float)

    scale_a = _pair_e8m0_scales(a_scale_bytes)
    scale_b = _pair_e8m0_scales(b_scale_bytes)
    bias = (torch.randn((n,), generator=generator) * 0.5).float() if has_bias else None
    return MxInputs(
        a=_pack_mxfp4_rows(a_nibbles) if transpose_a else _pack_mxfp4_cols(a_nibbles),
        b=_pack_mxfp4_cols(b_nibbles) if transpose_b else _pack_mxfp4_rows(b_nibbles),
        scale_a=scale_a.permute(1, 0, 2).contiguous() if transpose_a else scale_a,
        scale_b=scale_b if transpose_b else scale_b.permute(1, 0, 2).contiguous(),
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        bias=bias,
    )


def _decode_e2m1_packed(packed: torch.Tensor, logical_inner: int) -> torch.Tensor:
    """Decode packed E2M1 bytes along the tensor's innermost axis."""
    outer_shape = packed.shape[:-1]
    outer_count = math.prod(outer_shape)
    raw = packed.contiguous().view(torch.uint8).reshape(-1)
    decoded = torch.stack(
        [_E2M1_LUT[(raw & 0x0F).long()], _E2M1_LUT[(raw >> 4).long()]],
        dim=1,
    ).reshape(-1)
    return decoded[: outer_count * logical_inner].reshape(*outer_shape, logical_inner)


def _decode_e8m0_scale(scale: torch.Tensor, outer_size: int, k: int) -> torch.Tensor:
    """Decode [outer,G,2] E8M0 bytes to one FP32 value per 32 K elements."""
    valid_scale_k_len = ceil_div(k, MX_GROUP_SIZE)
    exponent_bytes = scale.contiguous().view(torch.uint8).reshape(outer_size, -1)
    exponents = exponent_bytes[:, :valid_scale_k_len].to(torch.int16) - 127
    return torch.pow(2.0, exponents.float())


def _mxfp8_golden(inputs: MxInputs) -> torch.Tensor:
    """CPU golden that decodes paired Scale and performs FP32 matmul."""
    a = inputs.a.T.contiguous() if inputs.transpose_a else inputs.a
    b = inputs.b if inputs.transpose_b else inputs.b.T.contiguous()
    m, k = a.shape
    n = b.shape[0]

    scale_a = (
        inputs.scale_a.permute(1, 0, 2).contiguous()
        if inputs.transpose_a
        else inputs.scale_a
    )
    scale_b = (
        inputs.scale_b
        if inputs.transpose_b
        else inputs.scale_b.permute(1, 0, 2).contiguous()
    )
    scale_a = _decode_e8m0_scale(scale_a, m, k)
    scale_b = _decode_e8m0_scale(scale_b, n, k)
    scale_a = scale_a.repeat_interleave(MX_GROUP_SIZE, dim=1)[:, :k]
    scale_b = scale_b.repeat_interleave(MX_GROUP_SIZE, dim=1)[:, :k]
    result = (a.float() * scale_a) @ (b.float() * scale_b).T
    if inputs.bias is not None:
        result = result + inputs.bias.to(torch.float32)
    return result


def _mxfp4_golden(inputs: MxInputs, m: int, k: int, n: int) -> torch.Tensor:
    """Decode packed E2M1 data and paired E8M0 scales on CPU."""
    if inputs.transpose_a:
        a = _decode_e2m1_packed(inputs.a, m).T
        scale_a_pairs = inputs.scale_a.permute(1, 0, 2).contiguous()
    else:
        a = _decode_e2m1_packed(inputs.a, k)
        scale_a_pairs = inputs.scale_a
    if inputs.transpose_b:
        b = _decode_e2m1_packed(inputs.b, k)
        scale_b_pairs = inputs.scale_b
    else:
        b = _decode_e2m1_packed(inputs.b, n).T
        scale_b_pairs = inputs.scale_b.permute(1, 0, 2).contiguous()

    scale_a = _decode_e8m0_scale(scale_a_pairs, m, k)
    scale_b = _decode_e8m0_scale(scale_b_pairs, n, k)
    scale_a = scale_a.repeat_interleave(MX_GROUP_SIZE, dim=1)[:, :k]
    scale_b = scale_b.repeat_interleave(MX_GROUP_SIZE, dim=1)[:, :k]
    result = (a * scale_a) @ (b * scale_b).T
    if inputs.bias is not None:
        result = result + inputs.bias.float()
    return result


def _assert_isclose(
    actual: torch.Tensor,
    golden: torch.Tensor,
    *,
    label: str,
    atol: float,
) -> None:
    """Require at least 99.9% of elements to satisfy the precision bound."""
    rtol, ptol = 1e-3, 1e-3
    np_actual = actual.float().cpu().numpy()
    np_golden = golden.float().cpu().numpy()
    match = np.isclose(np_actual, np_golden, rtol=rtol, atol=atol, equal_nan=True)
    mismatch_ratio = 1.0 - float(match.sum() / match.size)
    if mismatch_ratio <= ptol:
        return
    max_error = float((actual.float() - golden.float()).abs().max())
    raise AssertionError(
        f"QBMM {label} mismatch: max|err|={max_error:.4e}, "
        f"mismatch={mismatch_ratio:.4%}, "
        f"rtol={rtol}, atol={atol}, ptol={ptol}"
    )


def _require_mx_npu() -> None:
    pytest.importorskip("torch_npu")
    if not os.environ.get("ASCEND_HOME_PATH"):
        pytest.skip("ASCEND_HOME_PATH not set")
    if torch.npu.device_count() == 0:
        pytest.skip("no NPU device is available")
    if "Ascend950" not in torch.npu.get_device_name(0):
        pytest.skip("MX MMAD requires Ascend950")


def _run_npu_case(
    inputs: MxInputs,
    golden: torch.Tensor,
    output_dtype: torch.dtype,
    *,
    is_mxfp4: bool,
) -> None:
    """Run the unified MX entry and compare it with the supplied CPU Golden."""
    a = inputs.a.view(torch.float4_e2m1fn_x2) if is_mxfp4 else inputs.a
    b = inputs.b.view(torch.float4_e2m1fn_x2) if is_mxfp4 else inputs.b
    scale_a = inputs.scale_a.npu().view(torch.float8_e8m0fnu)
    scale_b = inputs.scale_b.npu().view(torch.float8_e8m0fnu)
    bias = inputs.bias.npu() if inputs.bias is not None else None
    actual = npu_quant_matmul(
        a.npu(),
        b.npu(),
        scale_a,
        scale_b,
        bias=bias,
        output_dtype=output_dtype,
    )
    torch.npu.synchronize()
    _assert_isclose(
        actual.cpu(),
        golden.to(output_dtype),
        label="MXFP4" if is_mxfp4 else "MXFP8",
        atol=1e-8 if is_mxfp4 else 1e-3,
    )


# ---------------------------------------------------------------------------
# NPU accuracy tests
# ---------------------------------------------------------------------------


@pytest.mark.npu
@pytest.mark.parametrize(
    "m,n,k,a_dtype,b_dtype,output_dtype,transpose_a,transpose_b,has_bias",
    _MXFP8_CASES + _MXFP4_CASES,
)
def test_quant_batch_matmul_mx(
    m: int,
    n: int,
    k: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    output_dtype: torch.dtype,
    transpose_a: bool,
    transpose_b: bool,
    has_bias: bool,
) -> None:
    """Verify MX QBMM precision across supported shapes and dtypes."""
    _require_mx_npu()
    is_mxfp4 = a_dtype == _FP4X2
    if is_mxfp4:
        inputs = _make_mxfp4_inputs(
            m,
            k,
            n,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            has_bias=has_bias,
        )
        golden = _mxfp4_golden(inputs, m, k, n)
    else:
        inputs = _make_mxfp8_inputs(
            m,
            k,
            n,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            has_bias=has_bias,
        )
        golden = _mxfp8_golden(inputs)
    _run_npu_case(inputs, golden, output_dtype, is_mxfp4=is_mxfp4)
