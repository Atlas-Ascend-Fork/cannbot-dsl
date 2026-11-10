# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Precision tests for the MX-quantized GroupedMatmulFinalizeRouting kernel.

Formula (per expert g with row window [start_g, end_g) from group_list):
  gmm_out[i, :] = (x1[i, :] * pertokenScale[i]) @ (x2[g] * scale[g])  (+ bias[g])
  y[rowIndex[i], :] += logit[i] * gmm_out[i, :]
  y[off:off+len, :]  = sharedInputWeight * sharedInput[:, :]

The CPU golden decodes the public E8M0 paired-scale tensors independently of
the NPU kernel (2^(byte-127); pair (g, j) covers K elements [64g+32j, 64g+32j+32)).
"""

from __future__ import annotations

import os
import sys

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
        "grouped_matmul_finalize_routing",
    ),
)

from grouped_matmul_finalize_routing import grouped_matmul_finalize_routing  # noqa: E402


_E4 = torch.float8_e4m3fn
_E5 = torch.float8_e5m2
_E8M0_LUT = 2.0 ** (torch.arange(256).to(torch.uint8).int() - 127)
_RTOL = 2e-2
_ATOL = 2e-2


# ---------------------------------------------------------------------------
# CPU golden
# ---------------------------------------------------------------------------


def _e8m0_bytes(tensor):
    return tensor.view(torch.uint8).long()


def golden(
    x1,
    x2,
    scale,
    pertoken_scale,
    group_sizes,
    logit,
    row_index,
    batch,
    bias=None,
    shared=None,
    shared_weight=1.0,
    shared_offset=0,
):
    m, k = x1.shape
    e, _, n = x2.shape
    kk = torch.arange(k)
    x1d = (
        x1.float()
        * _E8M0_LUT[_e8m0_bytes(pertoken_scale)[:, kk // 64, (kk % 64) // 32]]
    )
    y = torch.zeros(batch, n)
    prev = 0
    for g in range(e):
        end = prev + group_sizes[g]
        sel = _E8M0_LUT[_e8m0_bytes(scale[g])[kk // 64, :, (kk % 64) // 32]]
        out = x1d[prev:end] @ (x2[g].float() * sel)
        if bias is not None:
            out = out + bias[g].float()
        for i in range(group_sizes[g]):
            y[int(row_index[prev + i])] += float(logit[prev + i]) * out[i]
        prev = end
    if shared is not None:
        y[shared_offset : shared_offset + shared.shape[0]] += (
            shared_weight * shared.float()
        )
    return y


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------


def _paired_scale(shape, gen, low=120, high=130):
    return torch.randint(low, high, shape, generator=gen, dtype=torch.int16).to(
        torch.uint8
    )


def make_inputs(
    m,
    k,
    n,
    e,
    batch,
    group_sizes=None,
    *,
    seed=42,
    x_dtype=_E4,
    w_dtype=None,
    has_bias=False,
    shared_len=0,
    scale_low=120,
    scale_high=130,
    row_index=None,
):
    w_dtype = x_dtype if w_dtype is None else w_dtype
    gen = torch.Generator().manual_seed(seed)
    g = (k + 63) // 64
    if group_sizes is None:
        base, extra = divmod(m, e)
        group_sizes = [base + (1 if gi < extra else 0) for gi in range(e)]
    total_m = sum(group_sizes)
    x1 = (torch.randn((m, k), generator=gen) * 0.5).to(x_dtype)
    x2 = (torch.randn((e, k, n), generator=gen) * 0.5).to(w_dtype)
    pertoken_scale = _paired_scale((m, g, 2), gen, scale_low, scale_high)
    scale = _paired_scale((e, g, n, 2), gen, scale_low, scale_high)
    logit = torch.randn((m,), generator=gen)
    if row_index is None:
        row_index = torch.arange(total_m, dtype=torch.int64) % batch
        if total_m < m:
            rest = torch.arange(m - total_m, dtype=torch.int64) % batch
            row_index = torch.cat([row_index, rest])
        row_index = row_index.clamp(0, batch - 1)
    bias = (
        (torch.randn((e, n), generator=gen) * 0.5).to(torch.bfloat16)
        if has_bias
        else None
    )
    shared = (
        (torch.randn((shared_len, n), generator=gen) * 0.5).to(torch.bfloat16)
        if shared_len
        else None
    )
    return {
        "x1": x1,
        "x2": x2,
        "scale": scale,
        "pertoken_scale": pertoken_scale,
        "group_sizes": group_sizes,
        "logit": logit,
        "row_index": row_index,
        "bias": bias,
        "shared": shared,
    }


def run_kernel(inputs, batch, group_list_type, shared_weight=1.5, shared_offset=0):
    group_sizes = inputs["group_sizes"]
    if group_list_type == 0:
        group_list = torch.tensor(
            [sum(group_sizes[: i + 1]) for i in range(len(group_sizes))],
            dtype=torch.int64,
        )
    else:
        group_list = torch.tensor(group_sizes, dtype=torch.int64)
    kwargs = {}
    if inputs["bias"] is not None:
        kwargs["bias"] = inputs["bias"].npu()
    if inputs["shared"] is not None:
        kwargs["shared_input"] = inputs["shared"].npu()
        kwargs["shared_input_weight"] = shared_weight
        kwargs["shared_input_offset"] = shared_offset
    return grouped_matmul_finalize_routing(
        inputs["x1"].npu(),
        inputs["x2"].npu(),
        inputs["scale"].npu(),
        inputs["pertoken_scale"].npu(),
        group_list.npu(),
        inputs["logit"].npu(),
        inputs["row_index"].npu(),
        group_list_type=group_list_type,
        batch_size=batch,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# L1 functional cases: dtype x bias x group_list_type x shared
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m,k,n,e,batch,x_dtype,w_dtype,has_bias,glt,shared_len",
    [
        (128, 256, 256, 1, 128, _E4, _E4, False, 1, 0),
        (256, 256, 256, 4, 64, _E4, _E4, False, 1, 0),
        (256, 256, 256, 4, 64, _E4, _E4, True, 1, 0),
        (256, 256, 256, 4, 64, _E4, _E4, True, 0, 16),
        (256, 256, 256, 4, 64, _E5, _E5, True, 0, 16),
        (256, 256, 256, 4, 64, _E4, _E5, False, 1, 8),
        (128, 512, 512, 2, 64, _E4, _E4, True, 1, 0),
        (1024, 512, 512, 8, 128, _E4, _E4, False, 1, 0),
    ],
    ids=[
        "single-expert",
        "count",
        "count-bias",
        "cumsum-bias-shared",
        "e5-e5",
        "e4-e5-mixed",
        "wide-k",
        "multi-block",
    ],
)
def test_functional(m, k, n, e, batch, x_dtype, w_dtype, has_bias, glt, shared_len):
    inputs = make_inputs(
        m,
        k,
        n,
        e,
        batch,
        x_dtype=x_dtype,
        w_dtype=w_dtype,
        has_bias=has_bias,
        shared_len=shared_len,
    )
    shared_offset = 8 if shared_len else 0
    got = run_kernel(inputs, batch, glt, 1.5, shared_offset).cpu()
    want = golden(
        inputs["x1"],
        inputs["x2"],
        inputs["scale"],
        inputs["pertoken_scale"],
        inputs["group_sizes"],
        inputs["logit"],
        inputs["row_index"],
        batch,
        bias=inputs["bias"],
        shared=inputs["shared"],
        shared_weight=1.5,
        shared_offset=shared_offset,
    )
    assert torch.allclose(got, want, rtol=_RTOL, atol=_ATOL), (
        f"maxdiff={(got - want).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# L2 boundary cases: tails, empty groups, extreme scales
# ---------------------------------------------------------------------------


def _run_and_compare(inputs, batch, glt=1, shared_weight=1.5, shared_offset=8):
    got = run_kernel(inputs, batch, glt, shared_weight, shared_offset).cpu()
    want = golden(
        inputs["x1"],
        inputs["x2"],
        inputs["scale"],
        inputs["pertoken_scale"],
        inputs["group_sizes"],
        inputs["logit"],
        inputs["row_index"],
        batch,
        bias=inputs["bias"],
        shared=inputs["shared"],
        shared_weight=shared_weight,
        shared_offset=shared_offset,
    )
    assert torch.allclose(got, want, rtol=_RTOL, atol=_ATOL), (
        f"maxdiff={(got - want).abs().max().item()}"
    )


@pytest.mark.parametrize(
    "m,k,n,e,batch",
    [
        (300, 200, 300, 3, 100),  # m/n/k tails together
        (257, 192, 320, 2, 32),  # odd m tail
        (130, 96, 130, 5, 26),  # every axis ragged
        (64, 64, 64, 2, 32),  # tiny single-tile-per-group
        (2048, 512, 512, 8, 256),  # many tiles/windows/blocks
        (32768, 2048, 7168, 32, 1024),  # prefill-scale
    ],
    ids=["triple-tail", "odd-m", "all-ragged", "tiny", "many-tiles", "prefill"],
)
def test_tail_shapes(m, k, n, e, batch):
    inputs = make_inputs(m, k, n, e, batch, has_bias=True, seed=7)
    _run_and_compare(inputs, batch)


def test_group_sum_less_than_m():
    # Rows of x1 beyond the group total are ignored (mainline semantics).
    m, k, n, e, batch = 256, 256, 256, 2, 64
    inputs = make_inputs(m, k, n, e, batch, group_sizes=[64, 64], seed=3)
    _run_and_compare(inputs, batch)


def test_empty_groups():
    m, k, n, e, batch = 256, 256, 256, 3, 64
    inputs = make_inputs(m, k, n, e, batch, group_sizes=[64, 0, 64], seed=4)
    _run_and_compare(inputs, batch)


def test_zero_groups_shared_only():
    m, k, n, e, batch = 128, 256, 256, 2, 64
    inputs = make_inputs(m, k, n, e, batch, group_sizes=[0, 0], shared_len=16)
    got = run_kernel(inputs, batch, 1, shared_weight=1.0, shared_offset=0).cpu()
    want = torch.zeros(batch, n)
    want[0:16] = inputs["shared"].float()
    assert torch.allclose(got, want, rtol=1e-3, atol=1e-3)


def test_extreme_scales():
    # E8M0 = 0x00 -> 2^-127 annihilates the elements it covers; high bytes
    # grow the products. Byte ranges are chosen so no fp32 intermediate
    # overflows (exponent sums stay well below 127).
    m, k, n, e, batch = 128, 256, 256, 2, 64
    gen = torch.Generator().manual_seed(9)
    g = (k + 63) // 64
    inputs = make_inputs(m, k, n, e, batch, seed=9)
    ps = torch.randint(120, 136, (m, g, 2), generator=gen, dtype=torch.int16).to(
        torch.uint8
    )
    # Sprinkle zero scales: the elements they cover must vanish exactly.
    ps[::8, 0, 0] = 0
    ps[::16, 1, 1] = 0
    inputs["pertoken_scale"] = ps
    inputs["scale"] = torch.randint(
        120, 132, (e, g, n, 2), generator=gen, dtype=torch.int16
    ).to(torch.uint8)
    _run_and_compare(inputs, batch, glt=1, shared_weight=1.0, shared_offset=0)


def test_row_index_permuted():
    # Repeated and shuffled destinations exercise the atomic scatter-add.
    m, k, n, e, batch = 256, 256, 256, 4, 64
    gen = torch.Generator().manual_seed(11)
    row_index = torch.randint(0, batch, (m,), generator=gen, dtype=torch.int64)
    inputs = make_inputs(m, k, n, e, batch, row_index=row_index, has_bias=True, seed=11)
    _run_and_compare(inputs, batch)


def test_out_parameter_accumulates_shared():
    m, k, n, e, batch = 256, 256, 256, 2, 64
    inputs = make_inputs(m, k, n, e, batch, group_sizes=[64, 64], shared_len=16)
    out = torch.zeros(batch, n, dtype=torch.float32).npu()
    kwargs = dict(
        shared_input=inputs["shared"].npu(),
        shared_input_weight=1.5,
        shared_input_offset=8,
        group_list_type=1,
        out=out,
    )
    y = grouped_matmul_finalize_routing(
        inputs["x1"].npu(),
        inputs["x2"].npu(),
        inputs["scale"].npu(),
        inputs["pertoken_scale"].npu(),
        torch.tensor([64, 64], dtype=torch.int64).npu(),
        inputs["logit"].npu(),
        inputs["row_index"].npu(),
        **kwargs,
    )
    assert y is out
    want = golden(
        inputs["x1"],
        inputs["x2"],
        inputs["scale"],
        inputs["pertoken_scale"],
        inputs["group_sizes"],
        inputs["logit"],
        inputs["row_index"],
        batch,
        shared=inputs["shared"],
        shared_weight=1.5,
        shared_offset=8,
    )
    assert torch.allclose(y.cpu(), want, rtol=_RTOL, atol=_ATOL)


def test_validate_false_matches():
    # validate=False skips the row_index range syncs; the fast path stays
    # golden-accurate. Two invocations are not compared bit-exactly: the
    # scatter-add accumulates atomically in nondeterministic order (the
    # mainline op has the same property).
    m, k, n, e, batch = 256, 256, 256, 4, 64
    inputs = make_inputs(m, k, n, e, batch, seed=13)
    group_list = torch.tensor(inputs["group_sizes"], dtype=torch.int64)
    got_fast = grouped_matmul_finalize_routing(
        inputs["x1"].npu(),
        inputs["x2"].npu(),
        inputs["scale"].npu(),
        inputs["pertoken_scale"].npu(),
        group_list.npu(),
        inputs["logit"].npu(),
        inputs["row_index"].npu(),
        group_list_type=1,
        batch_size=batch,
        validate=False,
    ).cpu()
    assert tuple(got_fast.shape) == (batch, n)
    want = golden(
        inputs["x1"],
        inputs["x2"],
        inputs["scale"],
        inputs["pertoken_scale"],
        inputs["group_sizes"],
        inputs["logit"],
        inputs["row_index"],
        batch,
        bias=inputs["bias"],
        shared=inputs["shared"],
    )
    assert torch.allclose(got_fast, want, rtol=_RTOL, atol=_ATOL), (
        f"maxdiff={(got_fast - want).abs().max().item()}"
    )


def test_validate_false_requires_batch_size():
    m, k, n, e, batch = 128, 256, 256, 2, 64
    inputs = make_inputs(m, k, n, e, batch, seed=14)
    with pytest.raises(ValueError):
        grouped_matmul_finalize_routing(
            inputs["x1"].npu(),
            inputs["x2"].npu(),
            inputs["scale"].npu(),
            inputs["pertoken_scale"].npu(),
            torch.tensor([64, 64], dtype=torch.int64).npu(),
            inputs["logit"].npu(),
            inputs["row_index"].npu(),
            group_list_type=1,
            validate=False,
        )


# ---------------------------------------------------------------------------
# Validation contract (mainline-aligned error cases)
# ---------------------------------------------------------------------------


def _base_valid(m=128, k=256, n=256, e=2, batch=64):
    return make_inputs(m, k, n, e, batch, seed=5)


def test_reject_bad_group_list_type():
    inputs = _base_valid()
    with pytest.raises(ValueError):
        grouped_matmul_finalize_routing(
            inputs["x1"].npu(),
            inputs["x2"].npu(),
            inputs["scale"].npu(),
            inputs["pertoken_scale"].npu(),
            torch.tensor([64, 64], dtype=torch.int64).npu(),
            inputs["logit"].npu(),
            inputs["row_index"].npu(),
            group_list_type=2,
            batch_size=64,
        )


def test_reject_cumsum_not_monotonic():
    inputs = _base_valid()
    with pytest.raises(ValueError):
        grouped_matmul_finalize_routing(
            inputs["x1"].npu(),
            inputs["x2"].npu(),
            inputs["scale"].npu(),
            inputs["pertoken_scale"].npu(),
            torch.tensor([64, 32, 128], dtype=torch.int64)[:2].npu(),
            inputs["logit"].npu(),
            inputs["row_index"].npu(),
            group_list_type=0,
            batch_size=64,
        )


def test_reject_count_total_exceeds_m():
    inputs = _base_valid()
    with pytest.raises(ValueError):
        grouped_matmul_finalize_routing(
            inputs["x1"].npu(),
            inputs["x2"].npu(),
            inputs["scale"].npu(),
            inputs["pertoken_scale"].npu(),
            torch.tensor([200, 200], dtype=torch.int64).npu(),
            inputs["logit"].npu(),
            inputs["row_index"].npu(),
            group_list_type=1,
            batch_size=64,
        )


def test_reject_bad_scale_shape():
    inputs = _base_valid()
    with pytest.raises(ValueError):
        grouped_matmul_finalize_routing(
            inputs["x1"].npu(),
            inputs["x2"].npu(),
            inputs["scale"][:, :, :128, :].npu(),
            inputs["pertoken_scale"].npu(),
            torch.tensor([64, 64], dtype=torch.int64).npu(),
            inputs["logit"].npu(),
            inputs["row_index"].npu(),
            group_list_type=1,
            batch_size=64,
        )


def test_reject_row_index_out_of_range():
    inputs = _base_valid()
    bad = inputs["row_index"].clone()
    bad[0] = 999
    with pytest.raises(ValueError):
        grouped_matmul_finalize_routing(
            inputs["x1"].npu(),
            inputs["x2"].npu(),
            inputs["scale"].npu(),
            inputs["pertoken_scale"].npu(),
            torch.tensor([64, 64], dtype=torch.int64).npu(),
            inputs["logit"].npu(),
            bad.npu(),
            group_list_type=1,
            batch_size=64,
        )


def test_reject_out_wrong_device():
    # A CPU or wrong-device out would be zeroed and then returned as the
    # NPU op's result when the routing work is degenerate.
    m, k, n, e, batch = 128, 256, 256, 2, 64
    inputs = make_inputs(m, k, n, e, batch, group_sizes=[0, 0], seed=15)
    cpu_out = torch.zeros(batch, n, dtype=torch.float32)
    with pytest.raises(ValueError):
        grouped_matmul_finalize_routing(
            inputs["x1"].npu(),
            inputs["x2"].npu(),
            inputs["scale"].npu(),
            inputs["pertoken_scale"].npu(),
            torch.tensor([0, 0], dtype=torch.int64).npu(),
            inputs["logit"].npu(),
            inputs["row_index"].npu(),
            group_list_type=1,
            out=cpu_out,
        )


def test_reject_shared_offset_overflow():
    inputs = _base_valid()
    with pytest.raises(ValueError):
        grouped_matmul_finalize_routing(
            inputs["x1"].npu(),
            inputs["x2"].npu(),
            inputs["scale"].npu(),
            inputs["pertoken_scale"].npu(),
            torch.tensor([64, 64], dtype=torch.int64).npu(),
            inputs["logit"].npu(),
            inputs["row_index"].npu(),
            shared_input=(torch.randn(16, 256) * 0.5).to(torch.bfloat16).npu(),
            shared_input_offset=60,  # 60 + 16 > batch=64
            group_list_type=1,
            batch_size=64,
        )
