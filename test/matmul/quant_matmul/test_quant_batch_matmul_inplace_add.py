# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Layered tests for the Ascend950 MX ``quant_batch_matmul_inplace_add`` kernel.

Formula: Y_out[M,N] = Y_in[M,N] + dequant(X1)[M,K] @ dequant(X2)[K,N]

The CPU golden is implemented independently of the kernel: it decodes the
public rank-3 E8M0 paired-scale tensors (the (G, outer, 2) storage needs
``permute(1, 0, 2)`` before byte decoding), performs an FP32 per-32-group
matmul, and adds the initial ``y0``.

Layers:
  L0 codegen (ascendc_toolchain) - AOT compile via the module builder, .so asserted
  L1 functional (npu)            - 4 MX dtype combos x 3 shapes x 3 y0 modes
  L2 boundary (npu)              - m/n/k tail shapes and K group tails
  invalid inputs (npu)           - clean rejection of contract violations
  tail-split regression (npu)    - shapes that crashed before the tiling fix
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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

from quant_batch_matmul_inplace_add import (  # noqa: E402
    quant_batch_matmul_inplace_add,
)

_E4 = torch.float8_e4m3fn
_E5 = torch.float8_e5m2

# Public MX ABI constants, duplicated here so the golden stays
# independent of the kernel module internals.
MX_GROUP_SIZE = 32
MX_K_ALIGN = 64
MX_SCALE_PAIR = 2

_RTOL = _ATOL = 1e-3
_PTOL = 1e-3  # allowed element mismatch ratio (0.1%)


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


# ---------------------------------------------------------------------------
# Input construction and CPU golden (independent of the kernel internals)
# ---------------------------------------------------------------------------


def _make_paired_scale(outer_size, scale_k_len, *, generator):
    """Build the public E8M0 storage with shape [outer, G, 2]."""
    exponents = torch.randint(
        -2, 3, (outer_size, scale_k_len), dtype=torch.int16, generator=generator
    )
    return (
        (exponents + 127)
        .to(torch.uint8)
        .view(torch.int8)
        .reshape(outer_size, scale_k_len // MX_SCALE_PAIR, MX_SCALE_PAIR)
        .contiguous()
    )


def _make_inputs(m, k, n, *, a_dtype=_E4, b_dtype=_E4, seed=42):
    """Build deterministic X1 (k, m) / X2 (k, n) and (G, outer, 2) scales.

    The frozen transA=T / transB=F contract stores X1 as the physical
    (k, m) transpose of the logical (m, k) operand; X2 keeps (k, n).
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a_logical = (torch.randn((m, k), generator=generator) * 0.5).to(a_dtype)
    b_logical = (torch.randn((n, k), generator=generator) * 0.5).to(b_dtype)
    scale_k_len = _ceil_div(k, MX_K_ALIGN) * MX_SCALE_PAIR
    scale_a = _make_paired_scale(m, scale_k_len, generator=generator)
    scale_b = _make_paired_scale(n, scale_k_len, generator=generator)
    return (
        a_logical.T.contiguous(),
        b_logical.T.contiguous(),
        scale_a.permute(1, 0, 2).contiguous(),
        scale_b.permute(1, 0, 2).contiguous(),
    )


def _decode_e8m0_scale(scale_pairs, outer_size, k):
    """Decode (G, outer, 2) E8M0 bytes to one FP32 value per 32 K elements.

    The permute(1, 0, 2) first is essential: the pair lane is K-major, so a
    plain reshape of the public layout mixes group bytes across outers.
    """
    valid_scale_k_len = _ceil_div(k, MX_GROUP_SIZE)
    per_row = (
        scale_pairs.permute(1, 0, 2)
        .contiguous()
        .view(torch.uint8)
        .reshape(outer_size, -1)
    )
    exponents = per_row[:, :valid_scale_k_len].to(torch.int16) - 127
    return torch.pow(2.0, exponents.float())


def _golden_inplace_add(x1, x2, x1_scale, x2_scale, y0):
    """CPU golden: E8M0 decode -> FP32 per-32-group matmul -> + y0."""
    k, m = x1.shape
    n = x2.shape[1]
    a = x1.T.contiguous().float()
    b = x2.T.contiguous().float()
    scale_a = _decode_e8m0_scale(x1_scale, m, k)
    scale_b = _decode_e8m0_scale(x2_scale, n, k)
    scale_a = scale_a.repeat_interleave(MX_GROUP_SIZE, dim=1)[:, :k]
    scale_b = scale_b.repeat_interleave(MX_GROUP_SIZE, dim=1)[:, :k]
    return (a * scale_a) @ (b * scale_b).T + y0


def _make_y0(mode, m, n):
    if mode == "zeros":
        return torch.zeros((m, n))
    if mode == "ones":
        return torch.ones((m, n))
    generator = torch.Generator().manual_seed(1234)
    return torch.randn((m, n), generator=generator)


def _assert_isclose(actual, golden, *, label):
    """Require at least 99.9% of elements inside atol=rtol=1e-3."""
    np_actual = actual.float().cpu().numpy()
    np_golden = golden.float().cpu().numpy()
    match = np.isclose(np_actual, np_golden, rtol=_RTOL, atol=_ATOL, equal_nan=True)
    mismatch_ratio = 1.0 - float(match.sum() / match.size)
    if mismatch_ratio <= _PTOL:
        return
    max_error = float((actual.float() - golden.float()).abs().max())
    raise AssertionError(
        f"QBMM-IA {label} mismatch: max|err|={max_error:.4e}, "
        f"mismatch={mismatch_ratio:.4%}, "
        f"rtol={_RTOL}, atol={_ATOL}, ptol={_PTOL}"
    )


def _require_mx_npu():
    pytest.importorskip("torch_npu")
    if not os.environ.get("ASCEND_HOME_PATH"):
        pytest.skip("ASCEND_HOME_PATH not set")
    if torch.npu.device_count() == 0:
        pytest.skip("no NPU device is available")
    if "Ascend950" not in torch.npu.get_device_name(0):
        pytest.skip("MX MMAD requires Ascend950")


def _run_npu_case(m, k, n, *, a_dtype=_E4, b_dtype=_E4, y0_mode="randn", label=""):
    """Run the public interface in place and compare with the CPU golden."""
    x1, x2, x1_scale, x2_scale = _make_inputs(m, k, n, a_dtype=a_dtype, b_dtype=b_dtype)
    y0 = _make_y0(y0_mode, m, n)
    golden = _golden_inplace_add(x1, x2, x1_scale, x2_scale, y0)
    y = y0.clone().npu()
    out = quant_batch_matmul_inplace_add(
        x1.npu(),
        x2.npu(),
        x1_scale.npu().view(torch.float8_e8m0fnu),
        x2_scale.npu().view(torch.float8_e8m0fnu),
        y,
    )
    torch.npu.synchronize()
    assert out is y, "in-place contract: wrapper must return the same y tensor"
    _assert_isclose(out.cpu(), golden, label=label or f"{m}x{k}x{n}")


# ---------------------------------------------------------------------------
# L0: codegen tests (AOT compile chain produces a .so)
# ---------------------------------------------------------------------------

_L0_SHAPES = [
    (8, 16, 8, "tail-min"),
    (128, 512, 128, "aligned"),
]
_L0_DTYPES = [
    (_E4, _E4, "e4e4"),
    (_E4, _E5, "e4e5"),
    (_E5, _E4, "e5e4"),
    (_E5, _E5, "e5e5"),
]


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize("a_dtype,b_dtype,dtype_id", _L0_DTYPES)
@pytest.mark.parametrize("m,k,n,shape_id", _L0_SHAPES)
def test_codegen_produces_so(m, k, n, shape_id, a_dtype, b_dtype, dtype_id):
    """The frozen T-F template compiles through the full AOT chain."""
    # Host tiling needs cube_core_num from the NPU runtime, so this L0
    # still requires an initialized Ascend device.
    _require_mx_npu()
    from cannbotdsl import dtypes

    from quant_batch_matmul_inplace_add import _build_inplace_add_aot_callable

    torch_to_dsl = {
        _E4: dtypes.float8_e4m3fn,
        _E5: dtypes.float8_e5m2,
    }
    # Same builder/cache the public wrapper uses, so L1 reuses this compile.
    fn = _build_inplace_add_aot_callable(
        m, n, k, torch_to_dsl[a_dtype], torch_to_dsl[b_dtype]
    )
    assert fn.so_path, "AOT compile must produce a .so"
    assert Path(fn.so_path).is_file()
    assert not fn.closed


# ---------------------------------------------------------------------------
# L1: functional precision tests
# ---------------------------------------------------------------------------

_L1_SHAPES = [
    (8, 16, 8, "min"),
    (128, 512, 128, "small"),
    (1024, 512, 1024, "mid"),
]
_Y0_MODES = ["zeros", "ones", "randn"]


def _l1_cases():
    cases = []
    for a_dtype, b_dtype, dtype_id in _L0_DTYPES:
        for m, k, n, shape_id in _L1_SHAPES:
            for y0 in _Y0_MODES:
                cases.append(
                    pytest.param(
                        m,
                        k,
                        n,
                        a_dtype,
                        b_dtype,
                        y0,
                        id=f"func-{dtype_id}-{shape_id}-{m}x{k}x{n}-y0{y0}",
                    )
                )
    return cases


@pytest.mark.npu
@pytest.mark.parametrize(
    "m,k,n,a_dtype,b_dtype,y0_mode",
    _l1_cases(),
)
def test_l1_functional(m, k, n, a_dtype, b_dtype, y0_mode):
    """Verify precision across dtype combos, representative shapes and y0."""
    _require_mx_npu()
    _run_npu_case(
        m,
        k,
        n,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        y0_mode=y0_mode,
        label=f"l1-{m}x{k}x{n}",
    )


# ---------------------------------------------------------------------------
# L2: boundary tests (m/n/k tails and K group tails)
# ---------------------------------------------------------------------------

_L2_CASES = [
    # m/n/k tails, K group tail, and the 3-buffer L1 tiling regime.
    pytest.param(3998, 4093, 512, _E4, _E4, id="bnd-mnk-tail-3998x4093x512"),
    pytest.param(2051, 2048, 1024, _E4, _E4, id="bnd-m-tail-multiround-2051x2048x1024"),
    pytest.param(256, 2051, 256, _E4, _E4, id="bnd-k-tail-2051"),
    # k < 64 half-filled MX groups and the 64-crossing window.
    pytest.param(64, 1, 64, _E4, _E4, id="bnd-d1-k1"),
    pytest.param(64, 33, 64, _E4, _E4, id="bnd-d1-k33-halfgroup"),
    pytest.param(64, 63, 64, _E4, _E4, id="bnd-d1-k63-halfgroup"),
    pytest.param(64, 65, 64, _E4, _E4, id="bnd-d1-k65-secondgroup"),
    pytest.param(64, 70, 64, _E4, _E4, id="bnd-d1-k70-grouptail"),
    # Large-K set.
    pytest.param(128, 4096, 2048, _E4, _E4, id="bnd-largek-128x4096x2048"),
    # k > 16384 with a single output block (single-core K loop).
    pytest.param(128, 32768, 128, _E4, _E4, id="bnd-d4d5-singleblock-k32768"),
    # extreme m / n of 1 (no tail re-split on the tiny axis).
    pytest.param(1, 64, 64, _E4, _E4, id="bnd-d8-m1"),
    pytest.param(64, 64, 1, _E4, _E4, id="bnd-d8-n1"),
    # Single-core minimum (compile cache shared with L1).
    pytest.param(8, 16, 8, _E4, _E4, id="bnd-singlecore-min-8x16x8"),
    # dtype x half-group-tail cross coverage.
    pytest.param(64, 70, 64, _E5, _E5, id="bnd-d1-k70-e5e5-cross"),
]


@pytest.mark.npu
@pytest.mark.parametrize("m,k,n,a_dtype,b_dtype", _L2_CASES)
def test_l2_boundary(m, k, n, a_dtype, b_dtype):
    """Verify tail blocks, half-filled MX groups, large K and tiny M/N."""
    _require_mx_npu()
    _run_npu_case(
        m,
        k,
        n,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        y0_mode="randn",
        label=f"l2-{m}x{k}x{n}",
    )


# ---------------------------------------------------------------------------
# Invalid inputs: clean rejection without device-side work
# ---------------------------------------------------------------------------


def _valid_npu_tensors(m, k, n):
    x1, x2, s1, s2 = _make_inputs(m, k, n)
    return (
        x1.npu(),
        x2.npu(),
        s1.npu().view(torch.float8_e8m0fnu),
        s2.npu().view(torch.float8_e8m0fnu),
        torch.zeros((m, n)).npu(),
    )


@pytest.mark.npu
def test_reject_x1_wrong_dtype():
    _require_mx_npu()
    x1, x2, s1, s2, y = _valid_npu_tensors(8, 64, 8)
    with pytest.raises(TypeError):
        quant_batch_matmul_inplace_add(x1.cpu().float().npu(), x2, s1, s2, y)


@pytest.mark.npu
def test_reject_x2_wrong_dtype():
    _require_mx_npu()
    x1, x2, s1, s2, y = _valid_npu_tensors(8, 64, 8)
    with pytest.raises(TypeError):
        quant_batch_matmul_inplace_add(x1, x2.cpu().half().npu(), s1, s2, y)


@pytest.mark.npu
def test_reject_scale_wrong_dtype():
    _require_mx_npu()
    x1, x2, s1, s2, y = _valid_npu_tensors(8, 64, 8)
    with pytest.raises(TypeError):
        quant_batch_matmul_inplace_add(x1, x2, s1.view(torch.uint8), s2, y)


@pytest.mark.npu
def test_reject_scale_wrong_rank():
    _require_mx_npu()
    x1, x2, s1, s2, y = _valid_npu_tensors(8, 64, 8)
    with pytest.raises(ValueError):
        quant_batch_matmul_inplace_add(x1, x2, s1.reshape(-1), s2, y)


@pytest.mark.npu
def test_reject_scale_wrong_pair_lane():
    _require_mx_npu()
    m, k, n = 8, 64, 8
    x1, x2, s1, s2, y = _valid_npu_tensors(m, k, n)
    bad = torch.zeros((1, m, 3), dtype=torch.uint8).npu()
    with pytest.raises(ValueError):
        quant_batch_matmul_inplace_add(x1, x2, s1, bad.view(torch.float8_e8m0fnu), y)


@pytest.mark.npu
def test_reject_scale_wrong_group_count():
    _require_mx_npu()
    m, k, n = 8, 128, 8
    x1, x2, s1, s2, y = _valid_npu_tensors(m, k, n)
    # k=128 needs ceil(128/64)=2 groups; supply 1.
    bad = torch.zeros((1, m, 2), dtype=torch.uint8).npu()
    with pytest.raises(ValueError):
        quant_batch_matmul_inplace_add(x1, x2, bad.view(torch.float8_e8m0fnu), s2, y)


@pytest.mark.npu
def test_reject_y_wrong_dtype():
    _require_mx_npu()
    x1, x2, s1, s2, _ = _valid_npu_tensors(8, 64, 8)
    with pytest.raises(TypeError):
        quant_batch_matmul_inplace_add(
            x1, x2, s1, s2, torch.zeros((8, 8), dtype=torch.float16).npu()
        )


@pytest.mark.npu
def test_reject_y_wrong_shape():
    _require_mx_npu()
    m, k, n = 8, 64, 16
    x1, x2, s1, s2, _ = _valid_npu_tensors(m, k, n)
    with pytest.raises(ValueError):
        quant_batch_matmul_inplace_add(x1, x2, s1, s2, torch.zeros((n, m)).npu())


@pytest.mark.npu
def test_reject_non_tf_layout_x1():
    """x1 stored as logical (m, k) instead of the frozen (k, m) physical."""
    _require_mx_npu()
    m, k, n = 8, 64, 16
    x1, x2, s1, s2, y = _valid_npu_tensors(m, k, n)
    wrong_x1 = x1.reshape(m, k).npu()  # (m, k) view with m != k
    with pytest.raises(ValueError):
        quant_batch_matmul_inplace_add(wrong_x1, x2, s1, s2, y)


@pytest.mark.npu
def test_reject_k_mismatch():
    _require_mx_npu()
    m, k, n = 8, 64, 16
    x1, x2, s1, s2, y = _valid_npu_tensors(m, k, n)
    other_x2, _, _, _ = _make_inputs(m, k + 64, n)
    with pytest.raises(ValueError):
        quant_batch_matmul_inplace_add(x1, other_x2.npu(), s1, s2, y)


@pytest.mark.npu
def test_reject_rank3_x1():
    _require_mx_npu()
    m, k, n = 8, 64, 16
    x1, x2, s1, s2, y = _valid_npu_tensors(m, k, n)
    with pytest.raises(ValueError):
        quant_batch_matmul_inplace_add(x1.reshape(k, m, 1), x2, s1, s2, y)


@pytest.mark.npu
def test_reject_device_mismatch():
    _require_mx_npu()
    m, k, n = 8, 64, 16
    x1, x2, s1, s2, y = _valid_npu_tensors(m, k, n)
    with pytest.raises(ValueError):
        quant_batch_matmul_inplace_add(x1, x2.cpu(), s1, s2, y)


# ---------------------------------------------------------------------------
# Mainline interface parity (npu): kw-only group_sizes / dtype overrides
# ---------------------------------------------------------------------------
# The mainline torch.ops.npu.npu_add_quant_matmul_ surface adds kw-only
# parameters; the DSL wrapper accepts the same categories with identical
# accept/reject behavior (empirically pinned against the aclnn on
# Ascend950DT, torch_npu 2.7.1.post11):
#   * group_sizes=None (default) and [1, 1, 32] are the MX canonical value;
#     the aclnn rejects non-MX group modes on this path (error 161001).
#   * *_dtype overrides are bitcast hints: uint8/int8 transport views are
#     reinterpreted to their FP8/E8M0 payload dtypes before validation.
#   * x1_scale=None is rejected (aclnn: "Expected a proper Tensor but got
#     null").


@pytest.mark.npu
@pytest.mark.parametrize("group_sizes", [None, [1, 1, 32], (1, 1, 32)])
def test_group_sizes_mx_canonical_accepted(group_sizes):
    _require_mx_npu()
    m, k, n = 128, 256, 128
    x1, x2, s1, s2 = _make_inputs(m, k, n)
    y0 = _make_y0("randn", m, n)
    golden = _golden_inplace_add(x1, x2, s1, s2, y0)
    y = y0.clone().npu()
    out = quant_batch_matmul_inplace_add(
        x1.npu(),
        x2.npu(),
        s1.npu().view(torch.float8_e8m0fnu),
        s2.npu().view(torch.float8_e8m0fnu),
        y,
        group_sizes=group_sizes,
    )
    torch.npu.synchronize()
    _assert_isclose(out.cpu(), golden, label=f"group-sizes-{group_sizes}")


@pytest.mark.npu
@pytest.mark.parametrize(
    "bad",
    [
        pytest.param([1, 1, 128], id="group-k-128"),
        pytest.param([1, 1, 64], id="group-k-64"),
        pytest.param([2, 1, 32], id="group-m-2"),
        pytest.param([1, 1], id="two-elems"),
        pytest.param(32, id="not-iterable"),
    ],
)
def test_reject_non_mx_group_sizes(bad):
    _require_mx_npu()
    x1, x2, s1, s2, y = _valid_npu_tensors(8, 64, 8)
    with pytest.raises((ValueError, TypeError)):
        quant_batch_matmul_inplace_add(x1, x2, s1, s2, y, group_sizes=bad)


@pytest.mark.npu
def test_dtype_overrides_bitcast_views():
    """uint8/int8 transport views + *_dtype hints decode identically."""
    _require_mx_npu()
    m, k, n = 128, 256, 128
    x1, x2, s1, s2 = _make_inputs(m, k, n)
    y0 = _make_y0("randn", m, n)
    golden = _golden_inplace_add(x1, x2, s1, s2, y0)
    y = y0.clone().npu()
    out = quant_batch_matmul_inplace_add(
        x1.npu().view(torch.uint8),
        x2.npu().view(torch.uint8),
        s1.npu().view(torch.uint8),
        s2.npu().view(torch.uint8),
        y,
        x1_dtype=torch.float8_e4m3fn,
        x2_dtype=torch.float8_e4m3fn,
        x1_scale_dtype=torch.float8_e8m0fnu,
        x2_scale_dtype=torch.float8_e8m0fnu,
    )
    torch.npu.synchronize()
    _assert_isclose(out.cpu(), golden, label="dtype-overrides")


@pytest.mark.npu
def test_reject_dtype_override_int():
    """torch dtypes are not int-convertible; int overrides are rejected."""
    _require_mx_npu()
    x1, x2, s1, s2, y = _valid_npu_tensors(8, 64, 8)
    with pytest.raises(TypeError):
        quant_batch_matmul_inplace_add(x1, x2, s1, s2, y, x1_dtype=12345)


@pytest.mark.npu
def test_reject_dtype_override_size_mismatch():
    _require_mx_npu()
    x1, x2, s1, s2, y = _valid_npu_tensors(8, 64, 8)
    with pytest.raises(TypeError):
        quant_batch_matmul_inplace_add(x1, x2, s1, s2, y, x1_dtype=torch.bfloat16)


@pytest.mark.npu
@pytest.mark.parametrize("which", ["x1_scale", "x2_scale"])
def test_reject_null_scale_mainline_parity(which):
    """x1_scale/x2_scale=None is rejected (mainline aclnn parity)."""
    _require_mx_npu()
    x1, x2, s1, s2, y = _valid_npu_tensors(8, 64, 8)
    with pytest.raises(TypeError):
        quant_batch_matmul_inplace_add(
            x1,
            x2,
            None if which == "x1_scale" else s1,
            None if which == "x2_scale" else s2,
            y,
        )


# ---------------------------------------------------------------------------
# Regression: negative tail-split sub-block extents (was: aicore error 82)
# ---------------------------------------------------------------------------
# Before the host-tiling fix, _calc_tail_basic_block_split capped the split
# count by the full block extent (min(axis, base)), not the tail remainder.
# When the tail remainder r and the split count s satisfied
# (s-1)*ceil(r/s) > r (e.g. r in {1, 2} with a 4-way split, r=1 with
# 3-way), get_block computed min(current - idx*ceil(r/s), ceil(r/s)) < 0:
# a negative-length GM slice reached MTE and the kernel faulted on NPU
# (aicore error 82, MTE address out of bounds); the process did not
# recover. The fix caps the splits by the tail remainder, so these shapes
# now produce only >= 1-row/col sub-blocks and pass precision.


@pytest.mark.npu
@pytest.mark.parametrize(
    "m,k,n",
    [
        pytest.param(1, 64, 129, id="defect-neg-subblock-n-tail1"),
        pytest.param(1, 64, 130, id="defect-neg-subblock-n-tail2"),
        pytest.param(257, 64, 64, id="defect-neg-subblock-m-tail1"),
        pytest.param(129, 64, 129, id="defect-neg-subblock-mn-tail1"),
    ],
)
def test_known_defect_negative_tail_split(m, k, n):
    _require_mx_npu()
    _run_npu_case(m, k, n, y0_mode="randn", label=f"defect-{m}x{k}x{n}")
