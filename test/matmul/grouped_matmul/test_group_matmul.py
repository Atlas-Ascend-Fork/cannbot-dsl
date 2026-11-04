# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Precision tests for the group_matmul kernel (non-quantized).

Formula (per group g; rows_g = the g-th M-slice, k_g = the g-th K-slice):
  SPLIT_M (gt=0):   y[rows_g, n] = x[rows_g, k] @ weight_g[k, n]    (g row groups on M)
  NO_SPLIT (gt=-1): y_g[i, n]    = x_g[i, k]    @ weight_g[k, n]  (one independent pair per group)
  SPLIT_K (gt=2):   y_g[i, n]    = x[i, k_g]    @ weight_g[k_g, n] (no cross-group reduction;
                    y_g is the g-th slice of a 3-D [G, M, N] output (S5), or its own
                    [M, N_g] tensor with per-group N (S6))

Scenarios covered (see the operator docstring for the full matrix):
  S1 multi-x/multi-w/multi-y, S2 single/3-D-weight/single,
  S3 single/multi-w/single, S4 multi-x/multi-w/single (group_list optional),
  S5 K-split single/single/3-D-y, S6 K-split single/multi/multi.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "samples",
        "matmul",
        "grouped_matmul",
    ),
)

from group_matmul import group_matmul


NO_SPLIT = -1
SPLIT_M = 0
SPLIT_K = 2

_RTOL = 1e-3
_ATOL = 1e-3
_PTOL = 0.001


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def _case(
    scenario,
    m_list,
    k,
    n,
    *,
    tw=True,
    dtype=torch.float16,
    group_list_type=0,
    gl_none=False,
    case_id=None,
):
    group_type = {
        "S1": NO_SPLIT,
        "S2": SPLIT_M,
        "S3": SPLIT_M,
        "S4": SPLIT_M,
        "S5": SPLIT_K,
        "S6": SPLIT_K,
    }[scenario]
    split_item = 0 if scenario in ("S1", "S6") else (3 if scenario == "S5" else 2)
    if case_id is None:
        case_id = f"{scenario}_M{m_list}_K{k}_N{n}" + ("_ntw" if not tw else "")
        if gl_none:
            case_id += "_glNone"
        if group_list_type:
            case_id += "_count"
        if dtype == torch.bfloat16:
            case_id += "_bf16"
    return pytest.param(
        scenario,
        m_list,
        k,
        n,
        tw,
        dtype,
        group_type,
        split_item,
        group_list_type,
        gl_none,
        id=case_id,
    )


_FP16 = torch.float16
_BF16 = torch.bfloat16

_TEST_CASES = [
    # S1: multi-x multi-w multi-y (per-group m/k/n may differ).
    _case("S1", [128, 128], [1024, 1024], [512, 512], dtype=_FP16),
    _case("S1", [100, 17, 233], [33, 64, 512], [300, 128, 64], tw=False, dtype=_BF16),
    # S2: single-x, single 3-D weight.
    _case("S2", [128, 128], 1024, 512, dtype=_FP16),
    _case("S2", [128, 100], 1024, 512, tw=False, dtype=_BF16),
    _case("S2", [64, 0, 128, 64], 256, 128, dtype=_FP16),
    _case("S2", [100, 200, 150], 512, 33, dtype=_BF16),
    _case("S2", [300, 500], 2048, 1000, dtype=_BF16),
    # S3: single-x, multi 2-D weight.
    _case("S3", [128, 128], 1024, 512, dtype=_FP16),
    _case("S3", [100, 200, 150], 512, 256, tw=False, group_list_type=1, dtype=_BF16),
    # S4: multi-x, multi 2-D weight (group_list optional).
    _case("S4", [128, 128], 1024, 512, tw=False, dtype=_FP16),
    _case("S4", [128, 100], 1024, 512, gl_none=True, dtype=_BF16),
    _case("S4", [64, 0, 128, 64], 256, 128, dtype=_FP16),
    _case("S4", [33, 67, 17, 53, 47], 256, 128, dtype=_BF16),
    # S5: K-split, transposed x, single weight, 3-D y.
    _case("S5", [128], [192, 320, 64], 384, dtype=_FP16),
    _case("S5", [128], [100, 200, 150], 256, dtype=_BF16),
    _case("S5", [128], [192, 320, 64], 384, group_list_type=1, dtype=_FP16),
    _case("S5", [64], [33, 67, 17, 53, 47], 128, dtype=_BF16),
    # S6: K-split, transposed x, multi weight, multi y.
    _case("S6", [128], [256, 512], 256, dtype=_FP16),
    _case("S6", [128], [256, 512], 256, gl_none=True, dtype=_BF16),
    _case("S6", [64], [33, 67, 17], 128, dtype=_FP16),
]


# ---------------------------------------------------------------------------
# Input construction + CPU golden
# ---------------------------------------------------------------------------


def torch_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().view(bfloat16)
    return tensor.numpy()


def make_inputs(scenario, m_list, k, n, tw, dtype, group_list_type, gl_none, seed=42):
    """Build deterministic tensorlist inputs (CPU) for one case.

    Returns (x_list, w_list, group_list); the tensors follow the same view
    contracts as npu_grouped_matmul (SPLIT_K x is a transposed view,
    transposed weights are transpose(-1,-2) views).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    group_cnt = (
        len(k)
        if isinstance(k, list) and scenario in ("S5", "S6")
        else (len(m_list) if isinstance(k, list) else len(m_list))
    )

    def group_list_of(sizes):
        if group_list_type == 0:
            acc, gl = 0, []
            for s in sizes:
                acc += s
                gl.append(acc)
        else:
            gl = list(sizes)
        return torch.tensor(gl, dtype=torch.int64)

    if scenario == "S1":
        ks, ns = k, n
        x = [
            torch.randn(m, kk, generator=gen, dtype=dtype) for m, kk in zip(m_list, ks)
        ]
        w = [
            torch.randn(nn, kk, generator=gen, dtype=dtype).transpose(-1, -2)
            for kk, nn in zip(ks, ns)
        ]
        return x, w, None
    if scenario in ("S2", "S3"):
        k_val, n_val = k, n
        m_val = sum(m_list)
        group_cnt = len(m_list)
        x = [torch.randn(m_val, k_val, generator=gen, dtype=dtype)]
        if scenario == "S2":
            base = torch.randn(group_cnt, n_val, k_val, generator=gen, dtype=dtype)
            w = [base.transpose(-1, -2)]
        else:
            w = [
                torch.randn(n_val, k_val, generator=gen, dtype=dtype).transpose(-1, -2)
                for _ in range(group_cnt)
            ]
        return x, w, group_list_of(m_list)
    if scenario == "S4":
        k_val, n_val = k, n
        group_cnt = len(m_list)
        x = [torch.randn(m, k_val, generator=gen, dtype=dtype) for m in m_list]
        if tw:
            w = [
                torch.randn(n_val, k_val, generator=gen, dtype=dtype).transpose(-1, -2)
                for _ in range(group_cnt)
            ]
        else:
            w = [
                torch.randn(k_val, n_val, generator=gen, dtype=dtype)
                for _ in range(group_cnt)
            ]
        return x, w, (None if gl_none else group_list_of(m_list))
    # S5 / S6: x is a [M, k_total] transposed view; weight carries the K split.
    m_val = m_list[0]
    k_list = k
    k_total = sum(k_list)
    n_val = n
    x_base = torch.randn(k_total, m_val, generator=gen, dtype=dtype)
    x = [x_base.transpose(-1, -2)]
    if scenario == "S5":
        w = [torch.randn(k_total, n_val, generator=gen, dtype=dtype)]
    else:
        w = [torch.randn(kk, n_val, generator=gen, dtype=dtype) for kk in k_list]
    return x, w, (None if gl_none else group_list_of(k_list))


def group_matmul_golden(scenario, x_list, w_list, group_list, group_list_type):
    """CPU golden: per-group fp32 matmul, cast back to the input dtype."""
    dtype = x_list[0].dtype
    outs = []
    if scenario == "S1":
        for xg, wg in zip(x_list, w_list):
            outs.append((xg.float() @ wg.float()).to(dtype))
        return outs
    if scenario in ("S2", "S3"):
        x = x_list[0].float()
        if scenario == "S2":
            # Single 3-D weight (one [K, N] slice per group).
            w3d = w_list[0].float()
            sizes = _sizes_from_group_list(group_list, group_list_type, x.shape[0])
            start = 0
            for g, size in enumerate(sizes):
                outs.append((x[start : start + size] @ w3d[g]).to(dtype))
                start += size
        else:
            sizes = _sizes_from_group_list(group_list, group_list_type, x.shape[0])
            start = 0
            for wg, size in zip(w_list, sizes):
                outs.append((x[start : start + size] @ wg.float()).to(dtype))
                start += size
        return [torch.cat(outs, dim=0)]
    if scenario == "S4":
        for xg, wg in zip(x_list, w_list):
            outs.append((xg.float() @ wg.float()).to(dtype))
        return [torch.cat(outs, dim=0)]
    # S5 / S6: K split — slice x columns per group.  S6 carries one
    # weight per group; S5 has a single [K_total, N] weight whose rows
    # are split by group_list.
    x = x_list[0].float()
    if scenario == "S5":
        w = w_list[0].float()
        sizes = _sizes_from_group_list(group_list, group_list_type, x.shape[1])
        start = 0
        for size in sizes:
            outs.append(
                (x[:, start : start + size] @ w[start : start + size, :]).to(dtype)
            )
            start += size
        return [torch.stack(outs)]
    start = 0
    for wg in w_list:
        kk = wg.shape[0]
        outs.append((x[:, start : start + kk] @ wg.float()).to(dtype))
        start += kk
    return outs


def _sizes_from_group_list(group_list, group_list_type, total):
    vals = group_list.tolist()
    if group_list_type == 0:
        sizes, prev = [], 0
        for v in vals:
            sizes.append(v - prev)
            prev = v
        return sizes
    return vals


def assert_isclose(actual: np.ndarray, golden: np.ndarray, case_id: str) -> None:
    """Precision-tolerant comparison: allows a small fraction (ptol) of mismatches."""
    actual = actual.reshape(-1).astype(np.float32, copy=False)
    golden = golden.reshape(-1).astype(np.float32, copy=False)
    if actual.size != golden.size:
        raise AssertionError(
            f"[{case_id}] size mismatch: actual={actual.size}, golden={golden.size}"
        )
    if actual.size == 0:
        return
    diff = np.abs(actual - golden)
    rel = diff / (np.abs(golden) + 1e-30)
    fail = (rel > _RTOL) & (diff > _ATOL)
    if fail.sum() / actual.size > _PTOL:
        first = int(np.where(fail)[0][0])
        raise AssertionError(
            f"[{case_id}] precision FAIL: {fail.sum()}/{actual.size} elements "
            f"exceed rtol={_RTOL}, atol={_ATOL} (ptol={_PTOL}). "
            f"max|err|={diff.max():.6e}, first_idx={first}, "
            f"actual={actual[first]:.6e}, golden={golden[first]:.6e}"
        )
    logging.info(
        "[%s] PASS: max|err|=%.4e, elements=%d", case_id, diff.max(), actual.size
    )


# ---------------------------------------------------------------------------
# NPU precision tests
# ---------------------------------------------------------------------------


def _copy_input_to_npu(tensor: torch.Tensor) -> torch.Tensor:
    """Copy the contiguous base, then restore a last-two-axis transpose."""
    if tensor.is_contiguous():
        return tensor.npu()
    return tensor.transpose(-1, -2).contiguous().npu().transpose(-1, -2)


@pytest.mark.npu
@pytest.mark.parametrize(
    "scenario,m_list,k,n,tw,dtype,group_type,split_item,group_list_type,gl_none",
    _TEST_CASES,
)
def test_group_matmul_precision(
    scenario, m_list, k, n, tw, dtype, group_type, split_item, group_list_type, gl_none
) -> None:
    pytest.importorskip("torch_npu")

    x_list, w_list, group_list = make_inputs(
        scenario, m_list, k, n, tw, dtype, group_list_type, gl_none
    )
    golden = group_matmul_golden(scenario, x_list, w_list, group_list, group_list_type)

    x_npu = [_copy_input_to_npu(t) for t in x_list]
    w_npu = [_copy_input_to_npu(t) for t in w_list]
    gl_npu = group_list.npu() if group_list is not None else None

    y_npu = group_matmul(
        x_npu,
        w_npu,
        group_list=gl_npu,
        group_list_type=group_list_type,
        group_type=group_type,
        split_item=split_item,
    )
    torch.npu.synchronize()

    for g, (y_o, y_ref) in enumerate(zip(y_npu, golden)):
        result = torch_tensor_to_numpy(y_o.cpu())
        ref = torch_tensor_to_numpy(
            y_ref.cpu() if y_ref.device.type == "npu" else y_ref
        )
        assert result.shape == ref.shape, (
            f"[{scenario}][g{g}] shape mismatch: {result.shape} vs {ref.shape}"
        )
        assert_isclose(result, ref, f"{scenario}_g{g}")
