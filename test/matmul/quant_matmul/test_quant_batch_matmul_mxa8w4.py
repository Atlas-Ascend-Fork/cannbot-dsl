# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Precision tests for the MXA8W4 mix-quantized matmul kernel.

Formula:
  C[M,N] = (A[M,K] * sa) @ (B[N,K] * sb)^T + bias

  A:        fp8 e4m3 with per-32-group E8M0 scales (MXFP8)
  B:        packed fp4 e2m1 with per-32-group E8M0 scales (MXFP4)
  C:        fp16

Golden path mirrors the kernel: bit-exact fp4/fp8 decode, fp32 dequant
and accumulation, fp16 output.

A/B/C are never padded on GM (engines pad tails in flight); only the
E8M0 scale and bias descriptors are aligned to the tile grid. Both AIVs
convert B in parallel (row halves of the CrossCore L1 ring slot). K
domain: multiple of 32, otherwise unbounded — aligned and mixed-tail K
alike run the window ring; K tails need no padding.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys

import numpy as np
import pytest
import torch
from ml_dtypes import float4_e2m1fn, float8_e4m3fn

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

from quant_batch_matmul_mxa8w4 import matmul_mix_quant


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

_TEST_SHAPES = [
    # --- basics: aligned K, single/multi tile ---
    pytest.param(128, 256, 128, id="M128-K256-N128"),
    pytest.param(256, 256, 256, id="M256-K256-N256-multiTile"),
    # --- small / degenerate ---
    pytest.param(1, 32, 1, id="M1-K32-N1-minimal"),
    pytest.param(96, 96, 48, id="M96-K96-N48-smallK"),
    pytest.param(2, 1024, 3, id="M2-K1024-N3-4winTiny"),
    pytest.param(17, 1056, 33, id="M17-K1056-N33-tinyMultiWin"),
    # --- K/M/N tails ---
    pytest.param(200, 192, 72, id="M200-K192-N72-tailKMN"),
    pytest.param(64, 128, 136, id="M64-K128-N136-tailN"),
    pytest.param(256, 736, 96, id="M256-K736-N96-mixedTail3"),
    pytest.param(255, 544, 513, id="M255-K544-N513-primeTail"),
    pytest.param(127, 512, 129, id="M127-K512-N129-oddPair"),
    # --- window rings: sub_blocks_per_window x sub_block_k matrix ---
    pytest.param(128, 1024, 128, id="M128-K1024-N128-4winRing"),
    pytest.param(256, 512, 256, id="M256-K512-N256-kw256s2"),
    pytest.param(256, 384, 256, id="M256-K384-N256-kw256step3"),
    # K2560: k % sub_block_k == 0 but the last L1 window is partial — the
    # static-trip guard regression
    pytest.param(256, 2560, 256, id="M256-K2560-N256-kw256s10"),
    pytest.param(512, 512, 512, id="M512-K512-N512-kw128s4"),
    pytest.param(2048, 4608, 2048, id="M2048-K4608-N2048-kw128s36"),
    # --- tails on wide rings (partial last L1 window) ---
    pytest.param(256, 320, 256, id="M256-K320-N256-s2tail"),
    pytest.param(2048, 2176, 2048, id="M2048-K2176-N2048-b1tail"),
    # --- tall / wide / big grids ---
    pytest.param(1024, 128, 32, id="M1024-K128-N32-tinyN"),
    pytest.param(128, 2048, 4096, id="M128-K2048-N4096-wideN"),
    pytest.param(512, 512, 576, id="M512-K512-N576-9grp"),
    pytest.param(1024, 1024, 1024, id="M1024-K1024-N1024-full32"),
    pytest.param(2048, 7168, 2048, id="M2048-K7168-N2048-kw128"),
    pytest.param(2048, 7104, 2048, id="M2048-K7104-N2048-kw128tail"),
]


# ---------------------------------------------------------------------------
# Golden kit
# ---------------------------------------------------------------------------


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclasses.dataclass
class Mxa8w4Inputs:
    """Deterministic MXA8W4 golden inputs (e4m3 A, packed fp4 B, E8M0 scales, optional bias)."""

    a: np.ndarray
    codes: np.ndarray
    packed: np.ndarray
    sa: np.ndarray
    sb: np.ndarray
    bias: np.ndarray | None


def make_inputs(m: int, k: int, n: int, *, bias: bool, seed: int = 42) -> Mxa8w4Inputs:
    rng = np.random.default_rng(seed)
    # activations: e4m3-exact values
    a_vals = rng.choice(
        np.array([0.5, 1.0, -1.0, 2.0, -0.5, 1.5, -2.0], dtype=np.float32),
        size=(m, k),
    )
    a = a_vals.astype(float8_e4m3fn)
    # weights: random fp4 codes (all 16 codes exercised)
    codes = rng.integers(0, 16, size=(n, k), dtype=np.uint8)
    packed = (codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)
    # scales: E8M0 exponents in [-3, 3]
    scale_cols = _align(k, 64) // 32
    sa = (rng.integers(-3, 4, size=(m, scale_cols)) + 127).astype(np.uint8)
    sb = (rng.integers(-3, 4, size=(n, scale_cols)) + 127).astype(np.uint8)
    bias_np = (
        rng.choice(np.array([0.5, -1.0, 2.0, -0.25], dtype=np.float32), size=(n,))
        if bias
        else None
    )
    return Mxa8w4Inputs(a=a, codes=codes, packed=packed, sa=sa, sb=sb, bias=bias_np)


def golden(inputs: Mxa8w4Inputs) -> np.ndarray:
    a = inputs.a
    codes = inputs.codes
    _, k = a.shape
    a_deq = a.astype(np.float32)
    sa_exp = inputs.sa.astype(np.int32) - 127
    sa_deq = np.repeat(np.power(2.0, sa_exp.astype(np.float32)), 32, axis=1)[:, :k]
    b_deq = codes.view(float4_e2m1fn).astype(np.float32)
    sb_exp = inputs.sb.astype(np.int32) - 127
    sb_deq = np.repeat(np.power(2.0, sb_exp.astype(np.float32)), 32, axis=1)[:, :k]
    c = (a_deq * sa_deq) @ (b_deq * sb_deq).T
    if inputs.bias is not None:
        c = c + inputs.bias[None, :]
    return c.astype(np.float16)


def assert_isclose(actual: np.ndarray, golden_np: np.ndarray) -> None:
    actual = actual.astype(np.float32)
    gold = golden_np.astype(np.float32)
    # NaN guard FIRST: rel-error comparisons are NaN-blind (NaN > tol
    # is False), so a NaN-producing kernel can pass the tolerance check
    err = np.abs(actual - gold)
    nan_mask = np.isnan(actual) | np.isinf(actual)
    if nan_mask.any():
        idx = np.argwhere(nan_mask)[:5]
        details = "\n".join(
            f"  [{i},{j}] actual={actual[i, j]} golden={gold[i, j]:.4f}" for i, j in idx
        )
        raise AssertionError(
            f"matmul_mix_quant produced {nan_mask.sum()} NaN/Inf elements "
            f"({nan_mask.mean() * 100:.3f}%)\n{details}"
        )
    rel = err / np.maximum(np.abs(gold), 1.0)
    bad = rel > 2e-2
    if bad.mean() > 1e-3:
        idx = np.argwhere(bad)[:5]
        details = "\n".join(
            f"  [{i},{j}] actual={actual[i, j]:.4f} golden={gold[i, j]:.4f}"
            for i, j in idx
        )
        raise AssertionError(
            f"matmul_mix_quant mismatch: bad_ratio={bad.mean():.6f}, "
            f"max_abs_err={err.max():.4e}\n{details}"
        )


# ---------------------------------------------------------------------------
# NPU accuracy tests
# ---------------------------------------------------------------------------


@pytest.mark.npu
@pytest.mark.parametrize("m,k,n", _TEST_SHAPES)
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
def test_quant_batch_matmul_mxa8w4(m: int, k: int, n: int, bias: bool) -> None:
    pytest.importorskip("torch_npu")
    if not os.environ.get("ASCEND_HOME_PATH"):
        pytest.skip("ASCEND_HOME_PATH not set")
    if torch.npu.device_count() == 0:
        pytest.skip("no NPU device is available")
    if "Ascend950" not in torch.npu.get_device_name(0):
        pytest.skip("MXA8W4 MX matmul requires Ascend950")

    inputs = make_inputs(m, k, n, bias=bias)
    gold = golden(inputs)

    gm_a = torch.from_numpy(inputs.a.view(np.uint8)).npu().view(torch.float8_e4m3fn)
    gm_b = torch.from_numpy(np.ascontiguousarray(inputs.packed)).npu()
    gm_sa = torch.from_numpy(inputs.sa.copy()).npu()
    gm_sb = torch.from_numpy(inputs.sb.copy()).npu()
    gm_bias = (
        torch.from_numpy(inputs.bias).npu()
        if inputs.bias is not None
        else torch.zeros(n, dtype=torch.float32, device="npu")
    )

    c = matmul_mix_quant(gm_a, gm_b, gm_sa, gm_sb, gm_bias)
    torch.npu.synchronize()

    actual = c.cpu().numpy()
    assert_isclose(actual, gold)
    max_err = float(np.abs(actual.astype(np.float32) - gold.astype(np.float32)).max())
    logging.info(
        "matmul_mix_quant (M=%s,K=%s,N=%s,bias=%s) max|err|=%.4e finished!",
        m,
        k,
        n,
        bias,
        max_err,
    )


@pytest.mark.npu
def test_input_validation() -> None:
    pytest.importorskip("torch_npu")
    if not os.environ.get("ASCEND_HOME_PATH"):
        pytest.skip("ASCEND_HOME_PATH not set")
    if torch.npu.device_count() == 0:
        pytest.skip("no NPU device is available")
    if "Ascend950" not in torch.npu.get_device_name(0):
        pytest.skip("MXA8W4 MX matmul requires Ascend950")

    m, k, n = 64, 128, 64
    inputs = make_inputs(m, k, n, bias=False)
    gm_a = torch.from_numpy(inputs.a.view(np.uint8)).npu().view(torch.float8_e4m3fn)
    gm_b = torch.from_numpy(np.ascontiguousarray(inputs.packed)).npu()
    gm_sa = torch.from_numpy(inputs.sa.copy()).npu()
    gm_sb = torch.from_numpy(inputs.sb.copy()).npu()
    gm_bias = torch.zeros(n, dtype=torch.float32, device="npu")

    # K not a multiple of 32 (MX group size)
    bad_b = torch.zeros(n, 33, dtype=torch.uint8, device="npu")
    with pytest.raises(ValueError):
        matmul_mix_quant(gm_a, bad_b, gm_sa, gm_sb, gm_bias)

    # K is unbounded (windowed on-chip state; tails host-padded) —
    # mixed-tail and aligned large K alike run the window ring; no
    # NotImplementedError caps remain

    # wrong scale width
    bad_sa = torch.zeros(m, 3, dtype=torch.uint8, device="npu")
    with pytest.raises(ValueError):
        matmul_mix_quant(gm_a, gm_b, bad_sa, gm_sb, gm_bias)

    # wrong bias length
    bad_bias = torch.zeros(n + 1, dtype=torch.float32, device="npu")
    with pytest.raises(ValueError):
        matmul_mix_quant(gm_a, gm_b, gm_sa, gm_sb, bad_bias)
