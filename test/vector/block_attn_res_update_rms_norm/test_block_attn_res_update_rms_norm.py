# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""BF16 NPU accuracy test for block_attn_res_update_rms_norm.

Ten cases: seven D=7168 network shapes (T=1,2,4,8,32,254,512), one seed each,
plus three D=7168 cancellation/finite-FTZ regressions. The T=1 case also checks
T=0/T=3 and rejects unsupported D/delta dtypes in eager and Meta entries.
Reference: embedded CPU Torch/NumPy golden, matching ops-transformer spec.py,
followed by FP32 RMSNorm of the rounded BF16 h. Validate partial and y.
Precision: isclose rtol=0.001, ptol=0.001, atol=0; nonfinite values fail.
"""

import json
import os

import numpy as np
import pytest
import torch
from _samples_path import load_sample

_SAMPLE = load_sample(
    "vector/block_attn_res_update_rms_norm/block_attn_res_update_rms_norm.py"
)
block_attn_res_update_rms_norm = _SAMPLE.block_attn_res_update_rms_norm
clear_caches = _SAMPLE.clear_caches
pytestmark = pytest.mark.npu

SCORE_EPS = 1e-6
NORM_EPS = 1e-6
_FP32_TINY = np.float32(np.finfo(np.float32).tiny)


def _compare(actual, golden):
    a = actual.detach().cpu().float().numpy()
    g = golden.detach().cpu().float().numpy()
    if a.shape != g.shape:
        return {"pass": False, "reason": "shape mismatch"}
    finite = bool(np.isfinite(a).all() and np.isfinite(g).all())
    close = np.isclose(a, g, rtol=0.001, atol=0.0, equal_nan=False)
    bad = int(np.count_nonzero(~close))
    # Same precision calculation and threshold as TTK isclose; no TTK dependency.
    precision = (a.size - bad) / a.size if a.size else 1.0
    passed = (1.0 - precision) <= 0.001
    diff = np.abs(a.astype(np.float64) - g.astype(np.float64))
    relative = np.divide(diff, np.abs(g), out=np.full_like(diff, np.inf), where=g != 0)
    relative[diff == 0] = 0
    max_relative = float(relative.max()) if relative.size else 0.0
    return {
        "pass": bool(passed and finite),
        "bad_count": bad,
        "elements": int(a.size),
        "bad_fraction": bad / a.size if a.size else 0.0,
        "all_elements_pass": bad == 0 and finite,
        "max_abs_error": float(diff.max()) if diff.size else 0.0,
        "max_relative_error": max_relative if np.isfinite(max_relative) else "inf",
        "finite": finite,
    }


def _make_inputs(t, d, seed):
    torch.manual_seed(seed)
    # Input intervals follow the e2e cases in the ops-transformer ST CSV.
    p = torch.empty(t, d).uniform_(-1.0, 1.0)
    delta = torch.empty(t, d).uniform_(-0.25, 0.25).bfloat16()
    q = torch.empty(d).uniform_(-0.25, 0.25)
    num = torch.empty(t, d).uniform_(-1.0, 1.0)
    mx = torch.empty(t).uniform_(-0.5, 0.5)
    ell = torch.empty(t).uniform_(1.0, 2.0)
    gamma = (torch.randn(d) * 0.5 + 0.5).bfloat16()
    if d > 1:
        gamma[0] = 0
    return [p, delta, q, num, mx, ell], gamma


# Reference arithmetic below is copied from ops-transformer:
# attention/block_attn_res_update/tests/assets/spec.py
# Calculation order and BF16 rounding are preserved.


def _round_to_bfloat16_float32(value):
    """Round FP32 values to BF16 (RNE) in an FP32 NumPy container."""
    value = np.ascontiguousarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    nan_mask = np.isnan(value)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded_bits = (bits + rounding_bias) & np.uint32(0xFFFF0000)
    # Keep NaNs as NaNs even when all payload bits are in the truncated half.
    rounded_bits = np.where(
        nan_mask,
        (bits & np.uint32(0xFFFF0000)) | np.uint32(0x00010000),
        rounded_bits,
    ).astype(np.uint32, copy=False)
    return rounded_bits.view(np.float32)


def _ftz_float32(value):
    """Flush FP32 subnormal values to signed zero."""
    value = np.asarray(value, dtype=np.float32)
    signed_zero = np.copysign(np.float32(0.0), value)
    return np.where(np.abs(value) < _FP32_TINY, signed_zero, value).astype(
        np.float32, copy=False
    )


def _div_ftz_float32(lhs, rhs):
    """Model Ascend 950's default FP32 Div behavior with ``--cce-ftz=true``."""
    lhs = _ftz_float32(lhs)
    rhs = _ftz_float32(rhs)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        quotient = np.divide(lhs, rhs)
    return _ftz_float32(quotient)


def _sqrt_ftz_float32(value):
    """Model Ascend 950's default FP32 Sqrt behavior with ``--cce-ftz=true``."""
    value = _ftz_float32(value)
    with np.errstate(invalid="ignore", under="ignore"):
        result = np.sqrt(value)
    return _ftz_float32(result)


def _exp_sub_ftz_float32(lhs, rhs):
    """Model the default-FTZ ``ExpSub(lhs, rhs)`` instruction boundary."""
    lhs = _ftz_float32(lhs)
    rhs = _ftz_float32(rhs)
    difference = _ftz_float32(lhs - rhs)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.exp(difference).astype(np.float32, copy=False)
    return _ftz_float32(result)


def _fma_float32(dst, src0, src1):
    """Model ``Reg::MulDstAdd<float>`` with one FP32 rounding."""
    dst = np.asarray(dst, dtype=np.float32)
    src0 = np.asarray(src0, dtype=np.float32)
    src1 = np.asarray(src1, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = dst.astype(np.float64) * src0.astype(np.float64) + src1.astype(
            np.float64
        )
    return result.astype(np.float32, copy=False)


def _torch_ftz_float32(value):
    """Torch counterpart of :func:`_ftz_float32`."""
    tiny = torch.tensor(
        torch.finfo(torch.float32).tiny, dtype=value.dtype, device=value.device
    )
    signed_zero = torch.copysign(torch.zeros_like(value), value)
    return torch.where(torch.abs(value) < tiny, signed_zero, value)


def _torch_div_ftz_float32(lhs, rhs):
    """Torch counterpart of :func:`_div_ftz_float32`."""
    quotient = torch.div(_torch_ftz_float32(lhs), _torch_ftz_float32(rhs))
    return _torch_ftz_float32(quotient)


def _torch_sqrt_ftz_float32(value):
    """Torch counterpart of :func:`_sqrt_ftz_float32`."""
    return _torch_ftz_float32(torch.sqrt(_torch_ftz_float32(value)))


def _torch_exp_sub_ftz_float32(lhs, rhs):
    """Torch counterpart of :func:`_exp_sub_ftz_float32`."""
    difference = _torch_ftz_float32(_torch_ftz_float32(lhs) - _torch_ftz_float32(rhs))
    return _torch_ftz_float32(torch.exp(difference))


def _torch_fma_float32(dst, src0, src1):
    """Torch counterpart of :func:`_fma_float32`."""
    result = dst.to(dtype=torch.float64) * src0.to(dtype=torch.float64) + src1.to(
        dtype=torch.float64
    )
    return result.to(dtype=torch.float32)


def _torch_golden(
    partial_block,
    delta,
    pseudo_query,
    numerator,
    logit_max,
    exp_sum,
    eps,
):
    """Return ``(updated_partial_block, h)`` as CPU torch tensors."""
    partial = partial_block.to(dtype=torch.float32)
    delta_fp32 = delta.to(dtype=torch.float32)
    pseudo_query_fp32 = pseudo_query.to(dtype=torch.float32)
    numerator_fp32 = numerator.to(dtype=torch.float32)
    logit_max_fp32 = logit_max.to(dtype=torch.float32)
    exp_sum_fp32 = exp_sum.to(dtype=torch.float32)

    # Keep CPU inputs unchanged for the device call and the other reference.
    partial_out = partial + delta_fp32
    if partial_out.numel() == 0:
        return partial_out, torch.empty_like(delta)

    square_sum = torch.sum(partial_out * partial_out, dim=-1)
    dot_sum = torch.sum(partial_out * pseudo_query_fp32, dim=-1)
    inv_d = torch.tensor(
        float(np.float32(1.0 / partial_out.shape[-1])),
        dtype=torch.float32,
        device=partial_out.device,
    )
    rms = _torch_sqrt_ftz_float32(square_sum * inv_d + float(eps))
    score = _torch_div_ftz_float32(dot_sum, rms)

    current_max = torch.maximum(logit_max_fp32, score)
    alpha = _torch_exp_sub_ftz_float32(logit_max_fp32, current_max)
    beta = _torch_exp_sub_ftz_float32(score, current_max)
    denominator = _torch_fma_float32(exp_sum_fp32, alpha, beta)
    inv_denominator = _torch_div_ftz_float32(torch.ones_like(denominator), denominator)
    alpha = _torch_ftz_float32(alpha * inv_denominator)
    beta = _torch_ftz_float32(beta * inv_denominator)
    partial_scaled = partial_out * beta[:, None]
    h_fp32 = _torch_fma_float32(numerator_fp32, alpha[:, None], partial_scaled)
    return partial_out, h_fp32.to(dtype=torch.bfloat16)


def _numpy_golden(
    partial_block,
    delta,
    pseudo_query,
    numerator,
    logit_max,
    exp_sum,
    eps=1e-6,
):
    partial = np.asarray(partial_block, dtype=np.float32)
    delta_fp32 = np.asarray(delta, dtype=np.float32)
    pseudo_query = np.asarray(pseudo_query, dtype=np.float32)
    numerator = np.asarray(numerator, dtype=np.float32)
    logit_max = np.asarray(logit_max, dtype=np.float32)
    exp_sum = np.asarray(exp_sum, dtype=np.float32)

    # Keep CPU inputs unchanged for the device call and the other reference.
    partial_out = np.add(partial, delta_fp32).astype(np.float32, copy=False)

    square_sum = np.sum(partial_out * partial_out, axis=-1, dtype=np.float32)
    dot_sum = np.sum(partial_out * pseudo_query, axis=-1, dtype=np.float32)
    inv_d = np.float32(1.0 / partial_out.shape[-1])
    rms = _sqrt_ftz_float32(square_sum * inv_d + np.float32(eps))
    score = _div_ftz_float32(dot_sum, rms)

    current_max = np.maximum(logit_max, score)
    alpha = _exp_sub_ftz_float32(logit_max, current_max)
    beta = _exp_sub_ftz_float32(score, current_max)
    denominator = _fma_float32(exp_sum, alpha, beta)
    inv_denominator = _div_ftz_float32(np.ones_like(denominator), denominator)
    alpha = _ftz_float32(alpha * inv_denominator)
    beta = _ftz_float32(beta * inv_denominator)

    partial_scaled = partial_out * beta[:, None]
    h_fp32 = _fma_float32(numerator, alpha[:, None], partial_scaled)
    h = _round_to_bfloat16_float32(h_fp32)
    return [partial_out, h]


def _assert_results(inputs, gain, output, cpu_inputs, gamma, goldens, t, d):
    reports = {}
    for name, (expected_p, expected_h) in goldens.items():
        h32 = expected_h.float()
        rstd = torch.rsqrt((h32 * h32).mean(-1, keepdim=True) + NORM_EPS)
        expected_y = (h32 * rstd * gamma.float()).bfloat16()
        reports[name] = {
            "partial": _compare(inputs[0], expected_p),
            "output": _compare(output, expected_y),
        }
    readonly_ok = all(
        torch.equal(x.cpu(), c) for x, c in zip(inputs[1:], cpu_inputs[1:])
    )
    readonly_ok = readonly_ok and torch.equal(gain.cpu(), gamma)
    output_contract = (
        output.dtype == torch.bfloat16
        and tuple(output.shape) == (t, d)
        and output.device == inputs[0].device
    )
    partial_actual = inputs[0].cpu()
    partial_contract = partial_actual.dtype == torch.float32 and tuple(
        partial_actual.shape
    ) == (t, d)
    assert readonly_ok, "Read-only inputs were modified"
    assert output_contract, "Output shape, dtype or device mismatch"
    assert partial_contract, "Partial shape or dtype mismatch"
    for name, result in reports.items():
        for tensor_name, metric in result.items():
            assert metric["pass"], f"{name}/{tensor_name}: {json.dumps(metric)}"


_CASES = [pytest.param(t, 7168, id=f"T{t}_D7168") for t in (1, 254, 512)]
_SEED = 20260907


def _run_case(cpu_inputs, gamma):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    device = int(os.environ.get("DEVICE_ID", "0"))
    t, d = cpu_inputs[0].shape
    with torch.npu.device(device), torch.no_grad():
        torch_p, torch_h = _torch_golden(*cpu_inputs, SCORE_EPS)
        numpy_p, numpy_h = _numpy_golden(
            *(x.float().numpy() for x in cpu_inputs), eps=SCORE_EPS
        )
        goldens = {
            "torch_spec": (torch_p, torch_h),
            "numpy_spec": (
                torch.from_numpy(numpy_p),
                torch.from_numpy(numpy_h).bfloat16(),
            ),
        }
        inputs = [x.to(f"npu:{device}") for x in cpu_inputs]
        gain = gamma.to(f"npu:{device}")
        output = block_attn_res_update_rms_norm(*inputs, gain, SCORE_EPS, NORM_EPS)
        torch.npu.synchronize()
        _assert_results(inputs, gain, output, cpu_inputs, gamma, goldens, t, d)


@pytest.mark.parametrize("t,d", _CASES)
def test_block_attn_res_update_rms_norm(t, d, monkeypatch):
    cpu_inputs, gamma = _make_inputs(t, d, _SEED)
    if t == 1:
        # Reject unsupported delta dtypes before device execution.
        for dtype in (torch.float16, torch.float32):
            invalid_inputs = [cpu_inputs[0], cpu_inputs[1].to(dtype), *cpu_inputs[2:]]
            with pytest.raises(ValueError, match="delta 必须为 bfloat16"):
                block_attn_res_update_rms_norm(*invalid_inputs, gamma)
            with pytest.raises(ValueError, match="delta 必须为 bfloat16"):
                _call_meta(invalid_inputs, gamma)
        for invalid_d in (0, 64, 7167, 7169, 8192):
            invalid_inputs, invalid_gamma = _make_inputs(1, invalid_d, _SEED)
            with pytest.raises(ValueError, match="D 必须固定为 7168"):
                block_attn_res_update_rms_norm(*invalid_inputs, invalid_gamma)
            with pytest.raises(RuntimeError, match="D 必须固定为 7168"):
                _call_meta(invalid_inputs, invalid_gamma)
        for extra_t in (0, 3):
            extra_inputs, extra_gamma = _make_inputs(extra_t, d, _SEED)
            meta_output = _call_meta(extra_inputs, extra_gamma)
            assert meta_output.shape == (extra_t, d)
            assert meta_output.dtype == torch.bfloat16
            assert meta_output.device.type == "meta"
            if extra_t == 0:
                with monkeypatch.context() as patch:
                    patch.setattr(
                        _SAMPLE,
                        "_get_compiled_kernel",
                        lambda *args, **kwargs: pytest.fail(
                            "T=0 must not compile or launch a kernel"
                        ),
                    )
                    _run_case(extra_inputs, extra_gamma)
            else:
                _run_case(extra_inputs, extra_gamma)
    _run_case(cpu_inputs, gamma)


def _call_meta(cpu_inputs, gamma):
    return _SAMPLE.block_attn_res_update_rms_norm_op(
        *(x.to("meta") for x in cpu_inputs), gamma.to("meta"), SCORE_EPS, NORM_EPS
    )


def teardown_module():
    clear_caches()
