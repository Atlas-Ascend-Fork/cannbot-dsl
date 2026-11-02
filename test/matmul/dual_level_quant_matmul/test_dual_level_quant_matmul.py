# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Precision tests for dual-level MXFP4 matrix multiplication with NZ weights."""

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
        "dual_level_quant_matmul",
    ),
)

from dual_level_quant_matmul import npu_dual_level_quant_matmul  # noqa: E402
import dual_level_quant_matmul  # noqa: E402


LEVEL0_GROUP_SIZE = 512
LEVEL1_GROUP_SIZE = 32
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


def _require_npu():
    pytest.importorskip("torch_npu")
    if not os.environ.get("ASCEND_HOME_PATH"):
        pytest.skip("ASCEND_HOME_PATH not set")
    if torch.npu.device_count() == 0:
        pytest.skip("no NPU device is available")
    if "Ascend950" not in torch.npu.get_device_name(0):
        pytest.skip("MXFP4 MMAD requires Ascend950")


def _unpack_fp4(packed):
    packed = packed.contiguous().view(torch.uint8).cpu()
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    return _E2M1_LUT[codes.long()]


def _decode_e8m0(scale, rows, k):
    count = (k + LEVEL1_GROUP_SIZE - 1) // LEVEL1_GROUP_SIZE
    raw = scale.contiguous().view(torch.uint8).cpu().reshape(rows, -1)[:, :count]
    return torch.pow(2.0, raw.to(torch.int16).sub(127).float())


def _golden(a_data, b_data, scale_a0, scale_b0, scale_a1, scale_b1, k):
    a = _unpack_fp4(a_data)[:, :k]
    b = _unpack_fp4(b_data)[:, :k]
    m, n = a.shape[0], b.shape[0]
    level1_a = _decode_e8m0(scale_a1, m, k).repeat_interleave(
        LEVEL1_GROUP_SIZE,
        dim=1,
    )[:, :k]
    level1_b = _decode_e8m0(scale_b1, n, k).repeat_interleave(
        LEVEL1_GROUP_SIZE,
        dim=1,
    )[:, :k]
    level0_a = (
        scale_a0.float()
        .cpu()
        .repeat_interleave(
            LEVEL0_GROUP_SIZE,
            dim=1,
        )[:, :k]
    )
    level0_b = (
        scale_b0.float()
        .cpu()
        .repeat_interleave(
            LEVEL0_GROUP_SIZE,
            dim=1,
        )[:, :k]
    )
    return (a * level1_a * level0_a) @ (b * level1_b * level0_b).T


def _assert_close(actual, expected):
    actual = actual.float().cpu().numpy()
    expected = expected.float().cpu().numpy()
    close = np.isclose(actual, expected, rtol=1e-3, atol=1e-3, equal_nan=True)
    mismatch_ratio = 1.0 - float(close.sum() / close.size)
    assert mismatch_ratio <= 1e-3, (
        f"mismatch={mismatch_ratio:.4%}, "
        f"max_error={float(np.max(np.abs(actual - expected))):.6g}"
    )


def test_reject_cpu_inputs_before_compile(monkeypatch):
    def unexpected_compile(*args):
        pytest.fail("CPU inputs must be rejected before compiling the kernel")

    monkeypatch.setattr(
        dual_level_quant_matmul, "_build_aot_callable", unexpected_compile
    )
    a = torch.empty((16, 32), dtype=torch.int8).view(torch.float4_e2m1fn_x2)
    b = torch.empty((32, 32), dtype=torch.int8).view(torch.float4_e2m1fn_x2)
    with pytest.raises(ValueError, match="NPU tensors"):
        npu_dual_level_quant_matmul(
            a,
            b,
            torch.empty((16, 1)),
            torch.empty((1, 32)),
            torch.empty((16, 1, 2), dtype=torch.float8_e8m0fnu),
            torch.empty((32, 1, 2), dtype=torch.float8_e8m0fnu),
        )


@pytest.mark.npu
def test_reject_cpu_bias_before_compile(monkeypatch):
    _require_npu()
    import torch_npu

    def unexpected_compile(*args):
        pytest.fail("A cross-device Bias must be rejected before compiling the kernel")

    monkeypatch.setattr(
        dual_level_quant_matmul, "_build_aot_callable", unexpected_compile
    )
    a = torch.empty((16, 32), dtype=torch.int8, device="npu").view(
        torch.float4_e2m1fn_x2
    )
    b = torch_npu.npu_format_cast(
        torch.empty((32, 32), dtype=torch.int8, device="npu"), 29
    )
    with pytest.raises(ValueError, match="same device"):
        npu_dual_level_quant_matmul(
            a,
            b,
            torch.empty((16, 1), device="npu"),
            torch.empty((1, 32), device="npu"),
            torch.empty((16, 1, 2), dtype=torch.uint8, device="npu").view(
                torch.float8_e8m0fnu
            ),
            torch.empty((32, 1, 2), dtype=torch.uint8, device="npu").view(
                torch.float8_e8m0fnu
            ),
            bias=torch.empty((32,), dtype=torch.float32),
        )


@pytest.mark.npu
@pytest.mark.parametrize("b_dtype", [torch.int8, torch.float4_e2m1fn_x2])
def test_reject_nd_weights_before_compile(monkeypatch, b_dtype):
    _require_npu()

    def unexpected_compile(*args):
        pytest.fail("ND weights must be rejected before compiling the kernel")

    monkeypatch.setattr(
        dual_level_quant_matmul, "_build_aot_callable", unexpected_compile
    )
    a = torch.empty((16, 32), dtype=torch.int8, device="npu").view(
        torch.float4_e2m1fn_x2
    )
    b = torch.empty((32, 32), dtype=torch.int8, device="npu").view(b_dtype)
    with pytest.raises((TypeError, ValueError), match="FRACTAL_NZ"):
        npu_dual_level_quant_matmul(
            a,
            b,
            torch.empty((16, 1), device="npu"),
            torch.empty((1, 32), device="npu"),
            torch.empty((16, 1, 2), dtype=torch.uint8, device="npu").view(
                torch.float8_e8m0fnu
            ),
            torch.empty((32, 1, 2), dtype=torch.uint8, device="npu").view(
                torch.float8_e8m0fnu
            ),
        )


# NZ weights: distinct copy, scale-tail, and N-repartition paths.
@pytest.mark.npu
@pytest.mark.parametrize(
    "output_dtype,has_bias",
    [
        pytest.param(torch.float16, False, id="fp16"),
        pytest.param(torch.bfloat16, True, id="bf16-bias"),
    ],
)
@pytest.mark.parametrize(
    "m,n,k",
    [
        pytest.param(96, 80, 1024, id="nz-n-k-tiles"),
        pytest.param(96, 1, 512, id="nz-single-n-tail"),
        pytest.param(96, 16, 96, id="nz-k-paired-scale-tail"),
        pytest.param(130, 126, 770, id="nz-mnk-tail"),
        pytest.param(2048, 3072, 2560, id="nz-large-aligned"),
        pytest.param(2051, 2048, 3072, id="nz-tail-round-n-split"),
        pytest.param(3072, 2051, 2048, id="nz-edge-n-rebalance"),
        pytest.param(2048, 3072, 2050, id="nz-large-k-tail"),
    ],
)
def test_nz_weight_direct_copy_precision(m, n, k, output_dtype, has_bias):
    _require_npu()
    import torch_npu

    torch.manual_seed(42)
    a = torch.randn(m, k, dtype=torch.float16, device="npu")
    b = torch.randn(n, k, dtype=torch.float16, device="npu")
    a_data, scale_a0, scale_a1 = torch_npu.npu_dynamic_dual_level_mx_quant(a)
    b_data, scale_b0, scale_b1 = torch_npu.npu_dynamic_dual_level_mx_quant(b)
    b_nz = torch_npu.npu_format_cast(b_data.view(torch.int8), 29)
    bias = torch.randn(n, dtype=torch.float32, device="npu") if has_bias else None
    actual = npu_dual_level_quant_matmul(
        a_data.view(torch.float4_e2m1fn_x2),
        b_nz,
        scale_a0,
        scale_b0.T.contiguous(),
        scale_a1.view(torch.float8_e8m0fnu),
        scale_b1.view(torch.float8_e8m0fnu),
        bias=bias,
        output_dtype=output_dtype,
    )
    expected = _golden(
        a_data,
        b_data,
        scale_a0,
        scale_b0,
        scale_a1,
        scale_b1,
        k,
    )
    if bias is not None:
        expected = expected + bias.cpu()
    expected = expected.to(output_dtype)
    torch.npu.synchronize()
    _assert_close(actual, expected)
