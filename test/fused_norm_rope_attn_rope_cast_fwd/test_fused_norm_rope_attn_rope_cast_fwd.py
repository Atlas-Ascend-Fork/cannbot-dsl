# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""One small NPU precision case for the fused prefill operator.

S=4, T=32, H=64, K=32, D=512; Q Norm and sink are enabled.
Q/KV use randn * 0.1. The CPU golden preserves BF16 PV operands and
64-token online state. Original strict comparison thresholds are unchanged.
"""

import ctypes
import ctypes.util
from dataclasses import dataclass
from functools import lru_cache
import inspect

import pytest
import torch

from _samples_path import load_sample


@lru_cache(maxsize=1)
def _get_fmaf():
    # Python 3.12 has no math.fma. C99 fmaf supplies one FP32 rounding,
    # including cancellation cases; float64 multiply/add can double-round.
    libname = ctypes.util.find_library("m")
    if libname is None:
        raise RuntimeError("CPU golden requires a C99 libm with fmaf")
    fn = ctypes.CDLL(libname).fmaf
    fn.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    fn.restype = ctypes.c_float
    return fn


def _fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    if any(x.device.type != "cpu" or x.dtype != torch.float32 for x in (a, b, c)):
        raise ValueError("reference FMA requires FP32 CPU tensors")
    a, b, c = torch.broadcast_tensors(a, b, c)
    fmaf = _get_fmaf()
    values = [
        fmaf(x, y, z)
        for x, y, z in zip(
            a.flatten().tolist(), b.flatten().tolist(), c.flatten().tolist()
        )
    ]
    return torch.tensor(values, dtype=torch.float32).reshape(a.shape)


def _rope_inplace(q_or_o, conjugate, cos_sin_table, token_positions, rope_dim=64):
    cos = cos_sin_table[token_positions.long(), : rope_dim // 2].unsqueeze(1)
    sin = cos_sin_table[token_positions.long(), rope_dim // 2 :].unsqueeze(1)
    if conjugate:
        sin = -sin
    rope_part = q_or_o[..., -rope_dim:]
    x0, x1 = rope_part[..., 0::2], rope_part[..., 1::2]
    # The standalone product rounds to FP32 before entering the fused op.
    # Read both results before writing x0.
    y0 = _fma(x0, cos, -x1 * sin)
    y1 = _fma(x0, sin, x1 * cos)
    rope_part[..., 0::2] = y0
    rope_part[..., 1::2] = y1


@torch.no_grad()
def quantize_prefill_output(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 logical Y -> E4M3FN and packed, column-major UE8M0 (CPU)."""
    if y.device.type != "cpu" or y.dtype != torch.float32:
        raise TypeError("quantization reference requires FP32 CPU Y")
    if (
        y.ndim != 3
        or y.shape[0] < 1
        or y.shape[1] not in (64, 128)
        or y.shape[2] != 512
    ):
        raise ValueError("Y must have shape [S>=1,64|128,512]")
    if not torch.isfinite(y).all():
        raise ValueError("nonfinite Y is outside the golden quantization domain")
    s, n, _ = y.shape
    g, aligned_s = n // 8, (s + 3) // 4 * 4
    blocks = y.reshape(s, n, 16, 32)
    amax = blocks.abs().amax(-1).clamp_min(1e-4)
    raw = (amax / 448.0).contiguous()
    bits = raw.view(torch.int32)
    codes = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scales = (codes << 23).view(torch.float32)
    fp8 = (blocks / scales.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    # Move bytes, avoiding FP8 arithmetic / unsupported indexing operations.
    output_bytes = (
        fp8.view(torch.uint8)
        .reshape(s, g, 8, 16, 32)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(s, g, 4096)
    )
    output = output_bytes.view(torch.float8_e4m3fn)
    code_bytes = (
        codes.reshape(s, g, 8, 16).permute(0, 1, 3, 2).reshape(s, g, 32, 4).long()
    )
    words = sum(code_bytes[..., b] << (8 * b) for b in range(4)).to(torch.int32)
    backing = torch.zeros((g, 32, aligned_s), dtype=torch.int32)
    sf = backing.permute(2, 0, 1)[:s]
    sf.copy_(words)
    return output, sf


def dequantize_prefill_output(
    out_fp8: torch.Tensor, out_sf: torch.Tensor
) -> torch.Tensor:
    """Decode actual/golden output to logical FP32 [S,n,512] on CPU."""
    if out_fp8.dtype != torch.float8_e4m3fn or out_sf.dtype != torch.int32:
        raise TypeError("expected E4M3FN output and packed int32 SF")
    if out_fp8.ndim != 3 or out_fp8.shape[2] != 4096:
        raise ValueError("out_fp8 must have shape [S,g,4096]")
    s, g, _ = out_fp8.shape
    if tuple(out_sf.shape) != (s, g, 32):
        raise ValueError("out_sf must have shape [S,g,32]")
    words = out_sf.detach().cpu().long()
    codes = torch.stack([(words >> (8 * b)) & 255 for b in range(4)], -1)
    codes = codes.reshape(s, g, 16, 8).permute(0, 1, 3, 2).contiguous().to(torch.int32)
    scales = (codes << 23).view(torch.float32).reshape(s, g * 8, 16, 1)
    values = (
        out_fp8.detach()
        .cpu()
        .float()
        .reshape(s, g, 16, 8, 32)
        .permute(0, 1, 3, 2, 4)
        .reshape(s, g * 8, 16, 32)
    )
    return (values * scales).reshape(s, g * 8, 512)


@torch.no_grad()
def reference_prefill(
    q,
    kv,
    indices,
    scale,
    positions,
    cache,
    norm=False,
    eps=1e-6,
    sink=None,
    lengths=None,
    chunk=64,
):
    """Independent CPU model of BF16-PV operands and device online state."""
    if chunk != 64:
        raise ValueError("the device-arithmetic reference requires 64-token tiles")
    q = q.detach().cpu()
    kv = kv.detach().cpu().float()
    indices = indices.detach().cpu()
    positions = positions.detach().cpu()
    cache = cache.detach().cpu()
    sink = None if sink is None else sink.detach().cpu()
    lengths = None if lengths is None else lengths.detach().cpu()
    s, h, _ = q.shape
    k = indices.shape[-1]
    t = kv.shape[0]
    logical = q.reshape(s, 32, h, 16).permute(0, 2, 1, 3).reshape(s, h, 512).float()
    pre_scale = norm or scale <= 0
    softmax_scale = 1.0 if pre_scale else float(scale)
    alpha = None
    if pre_scale:
        alpha = torch.full((s, h), float(scale), dtype=torch.float32)
        if norm:
            alpha *= torch.rsqrt(logical.square().sum(-1) / 512 + eps)
    _rope_inplace(logical, False, cache, positions)
    rotated = logical.bfloat16().float()
    out = torch.zeros((s, h, 512), dtype=torch.float32)
    maximum = torch.full((s, h), -float("inf"))
    lse = torch.full((s, h), float("inf"))
    # Each 64-head Cube job has two independent 32-head Softmax bases.
    # Either AIV requesting rescale flushes the shared output segment.
    for row in range(s):
        count = k if lengths is None else int(lengths[row])
        row_max = torch.full((h,), -float("inf"))
        base = row_max.clone()
        row_sum = torch.zeros(h)
        history = torch.zeros((h, 512))
        segment = torch.zeros_like(history)
        for begin in range(0, max(count, 1), chunk):
            end = min(begin + chunk, count)
            ids = torch.full((chunk,), t, dtype=torch.long)
            ids[: end - begin] = indices[row, 0, begin:end].long()
            ids.masked_fill_((ids < 0) | (ids >= t), t)
            ids = ids.reshape(-1, 2).sort(-1).values.reshape(-1)
            valid = ids < t
            values = torch.zeros((chunk, 512), dtype=torch.float32)
            values[valid] = kv[ids[valid], 0]
            raw = rotated[row] @ values.T
            if pre_scale:
                raw *= alpha[row, :, None]
            raw.masked_fill_(~valid[None, :], -float("inf"))
            row_max = torch.maximum(row_max, raw.amax(-1))
            if begin == 0:
                new_base = row_max.clone()
                factors = torch.ones_like(base)
            else:
                changed = (row_max - base).reshape(h // 64, 2, 32).amax(-1)
                changed = changed > 6.0 / softmax_scale
                update = changed[..., None].expand(-1, -1, 32).reshape(h)
                new_base = torch.where(update, row_max, base)
                factors = torch.where(
                    new_base == base,
                    1.0,
                    torch.exp(base * softmax_scale - new_base * softmax_scale),
                )
                flush = changed.any(-1).repeat_interleave(64)[:, None]
                history = torch.where(
                    flush, (history + segment) * factors[:, None], history
                )
                segment = torch.where(flush, 0.0, segment)
            probability = torch.exp(
                raw * softmax_scale - new_base[:, None] * softmax_scale
            )
            probability.masked_fill_(~valid[None, :], 0.0)
            row_sum = _fma(row_sum, factors, probability.sum(-1))
            # Convert back to FP32 only to accumulate CPU GEMMs with exact
            # BF16 operand values, rather than returning a BF16 matmul result.
            pv_weights = probability.bfloat16().float()
            segment += pv_weights @ values
            base = new_base
        valid = row_sum > 0
        maximum[row] = row_max * softmax_scale
        lse[row] = torch.where(
            valid, base * softmax_scale + row_sum.log(), float("inf")
        )
        if sink is not None:
            denominator = row_sum + torch.exp(sink - base * softmax_scale)
        else:
            denominator = row_sum
        out[row] = torch.where(
            valid[:, None], (history + segment) / denominator[:, None], 0.0
        )
    _rope_inplace(out, True, cache, positions)
    fp8, sf = quantize_prefill_output(out)
    return (fp8, sf, maximum, lse), out


def comparison(actual, expected, abs_tol, rel_tol, cos_diff_tol):
    a = actual.detach().cpu().float().flatten()
    b = expected.detach().cpu().float().flatten()
    special = (~torch.isfinite(a)) | (~torch.isfinite(b))
    special_bad = sum(
        int((ma != mb).sum())
        for ma, mb in [
            (a == float("inf"), b == float("inf")),
            (a == -float("inf"), b == -float("inf")),
            (torch.isnan(a), torch.isnan(b)),
        ]
    )
    a = a[~special]
    b = b[~special]
    diff = (a - b).abs()
    # Exact project rule: relative error uses abs(reference)+1e-6. An element
    # passes either strict absolute or strict relative comparison.
    relative = diff / (b.abs() + 1e-6)
    bad = int(((diff >= abs_tol) & (relative >= rel_tol)).sum())
    a = a.double()
    b = b.double()
    dot = (a * b).sum()
    den = (a * a).sum() + (b * b).sum()
    cos_diff = float(1 - 2 * dot / den) if (b * b).sum() >= 1e-12 else 0.0
    return dict(
        passed=(bad + special_bad == 0 and abs(cos_diff) <= cos_diff_tol),
        bad=bad,
        special_bad=special_bad,
        max_abs=float(diff.max()) if len(diff) else 0.0,
        cos_diff=cos_diff,
    )


@dataclass(frozen=True)
class PrefillCase:
    name: str
    s: int = 4
    t: int = 32
    h: int = 64
    k: int = 32
    norm: bool = False
    sink: bool = False
    length_none: bool = False
    strided: bool = False
    scale: float = 512**-0.5
    seed: int = 0


CASES = (PrefillCase("h64_norm_sink_small", norm=True, sink=True, seed=101),)


def _padded_row_copy(tensor, padding, offset=17):
    """Keep a contiguous row in storage with a gap and a nonzero origin."""
    row_stride = tensor[0].numel() + padding
    storage = torch.zeros(offset + tensor.shape[0] * row_stride, dtype=tensor.dtype)
    result = storage.as_strided(
        tensor.shape, (row_stride, *tensor.stride()[1:]), offset
    )
    result.copy_(tensor)
    return result


def make_cpu_inputs(case):
    generator = torch.Generator().manual_seed(case.seed)
    logical = (torch.randn(case.s, case.h, 512, generator=generator) * 0.1).bfloat16()
    q = (
        logical.reshape(case.s, case.h, 32, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(case.s, case.h, 512)
    )
    kv = (torch.randn(case.t, 1, 512, generator=generator) * 0.1).bfloat16()
    indices = torch.randint(
        0, case.t, (case.s, 1, case.k), generator=generator, dtype=torch.int32
    )
    lengths = torch.randint(
        0, case.k, (case.s,), generator=generator, dtype=torch.int32
    )
    positions = torch.randint(0, 37, (case.s,), generator=generator, dtype=torch.int32)
    theta = torch.randn(37, 32, generator=generator)
    cache = torch.cat((theta.cos(), theta.sin()), dim=1)
    sink = torch.randn(case.h, generator=generator) if case.sink else None

    boundaries = (
        0,
        0,
        1,
        63,
        64,
        65,
        127,
        128,
        639,
        640,
        641,
        1279,
        1280,
        1281,
        2560,
        2561,
        case.k,
    )
    for row, length in enumerate(boundaries[: case.s]):
        lengths[row] = min(length, case.k)
    if case.s > 2:
        indices[2, 0] = 0  # Duplicate keys, including one-element rows.
    if case.s > 3:
        indices[3, 0, ::2] = -1  # Single valid element in a gather pair.
    if case.s > 4:
        indices[4, 0] = case.t  # Empty attention with a positive prefix length.
    if case.s > 5:
        indices[5, 0, ::3] = case.t + 1
    if case.s > 6:
        indices[6, 0, 1::2] = -2
    if sink is not None:
        sink[0], sink[1] = float("inf"), -float("inf")
    if case.strided:
        q = _padded_row_copy(q, 64)
        kv = _padded_row_copy(kv, 16)
        indices = _padded_row_copy(indices, 16)
    return (
        q,
        kv,
        indices,
        float(case.scale),
        512,
        sink,
        None if case.length_none else lengths,
        case.norm,
        1e-6,
        positions,
        False,
        64,
        cache,
        case.h // 8,
        32,
        True,
        True,
        True,
    )


def copy_inputs_to_device(values, device):
    """Preserve row strides and storage offsets when creating device inputs."""
    result = []
    for value in values:
        if not isinstance(value, torch.Tensor):
            result.append(value)
        elif value.is_contiguous() and value.storage_offset() == 0:
            result.append(value.to(device))
        else:
            capacity = (
                value.storage_offset()
                + 1
                + sum(
                    (size - 1) * stride
                    for size, stride in zip(value.shape, value.stride())
                )
            )
            storage = torch.zeros(capacity, dtype=value.dtype, device=device)
            view = storage.as_strided(
                value.shape, value.stride(), value.storage_offset()
            )
            view.copy_(value)
            result.append(view)
    return tuple(result)


SAMPLE = "fused_norm_rope_attn_rope_cast_fwd/fused_norm_rope_attn_rope_cast_fwd.py"
OUT_TOLERANCE = (1.1e-3, 1.01 / 8, 1e-3)
STATS_TOLERANCE = (1e-5, 4.01 / 65536, 1e-7)
PARAMETERS = (
    "q",
    "kv",
    "indices",
    "sm_scale",
    "d_v",
    "attn_sink",
    "topk_length",
    "enable_q_norm",
    "rms_norm_eps",
    "token_positions",
    "is_rope_neox_style",
    "rope_dim",
    "cos_sin_cache",
    "n_wv_group",
    "num_per_channels",
    "use_tma_aligned_col_major_sf",
    "round_sf",
    "use_packed_ue8m0",
)


@pytest.fixture(scope="module")
def sample():
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU runtime is unavailable")
    torch.npu.set_device(0)
    module = load_sample(SAMPLE)
    signature = inspect.signature(module.fused_norm_rope_attn_rope_cast_fwd)
    assert tuple(signature.parameters) == PARAMETERS
    assert all(
        signature.parameters[name].default is inspect.Parameter.empty
        for name in PARAMETERS
    )
    assert all(
        signature.parameters[name].kind is inspect.Parameter.POSITIONAL_ONLY
        for name in PARAMETERS
    )
    placeholders = (None,) * len(PARAMETERS)
    signature.bind(*placeholders)
    with pytest.raises(TypeError):
        signature.bind(*placeholders[:-1])
    with pytest.raises(TypeError):
        signature.bind(*placeholders, None)
    with pytest.raises(TypeError):
        signature.bind(**dict.fromkeys(PARAMETERS))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 8))
    try:
        yield module
    finally:
        torch.set_num_threads(previous_threads)


def _cpu_outputs(outputs):
    return tuple(
        value.view(torch.uint8).cpu() if index == 0 else value.cpu()
        for index, value in enumerate(outputs)
    )


def _assert_exact_outputs(actual, expected):
    actual_cpu, expected_cpu = _cpu_outputs(actual), _cpu_outputs(expected)
    assert all(torch.equal(a, b) for a, b in zip(actual_cpu, expected_cpu))


def _assert_output_contract(values, outputs):
    s, h, _ = values[0].shape
    g, aligned_s = h // 8, (s + 3) // 4 * 4
    expected = (
        ((s, g, 4096), torch.float8_e4m3fn, (g * 4096, 4096, 1)),
        ((s, g, 32), torch.int32, (1, 32 * aligned_s, aligned_s)),
        ((s, h), torch.float32, (h, 1)),
        ((s, h), torch.float32, (h, 1)),
    )
    assert isinstance(outputs, (tuple, list)) and len(outputs) == 4
    for output, (shape, dtype, stride) in zip(outputs, expected):
        assert (
            output.shape == shape
            and output.dtype == dtype
            and output.stride() == stride
        )
        assert output.device == values[0].device
    sf = outputs[1]
    backing = sf.as_strided((g, 32, aligned_s), (32 * aligned_s, aligned_s, 1))
    assert int(torch.count_nonzero(backing[:, :, s:].cpu())) == 0


def _assert_empty_rows(cpu_values, outputs):
    indices, kv, lengths = cpu_values[2], cpu_values[1], cpu_values[6]
    empty = []
    for row in range(indices.shape[0]):
        length = indices.shape[-1] if lengths is None else int(lengths[row])
        prefix = indices[row, 0, :length]
        if not ((prefix >= 0) & (prefix < kv.shape[0])).any():
            empty.append(row)
    if empty:
        assert bool((outputs[0].cpu().float()[empty] == 0).all())
        assert bool((outputs[1].cpu()[empty] == 0x69696969).all())
        assert bool(torch.isneginf(outputs[2].cpu()[empty]).all())
        assert bool(torch.isposinf(outputs[3].cpu()[empty]).all())


def _assert_precision(cpu_values, outputs):
    expected, y = reference_prefill(
        cpu_values[0],
        cpu_values[1],
        cpu_values[2],
        cpu_values[3],
        cpu_values[9],
        cpu_values[12],
        cpu_values[7],
        cpu_values[8],
        cpu_values[5],
        cpu_values[6],
    )
    checks = dict(
        out=comparison(
            dequantize_prefill_output(outputs[0], outputs[1]), y, *OUT_TOLERANCE
        ),
        max=comparison(outputs[2], expected[2], *STATS_TOLERANCE),
        lse=comparison(outputs[3], expected[3], *STATS_TOLERANCE),
    )
    assert all(result["passed"] for result in checks.values()), checks


@pytest.mark.npu
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_prefill_matches_cpu_and_preserves_inputs(sample, case):
    cpu_values = make_cpu_inputs(case)
    values = copy_inputs_to_device(cpu_values, "npu")
    outputs = sample.fused_norm_rope_attn_rope_cast_fwd(*values)
    torch.npu.synchronize()
    _assert_output_contract(values, outputs)
    _assert_precision(cpu_values, outputs)
    _assert_empty_rows(cpu_values, outputs)
    repeated = sample.fused_norm_rope_attn_rope_cast_fwd(*values)
    torch.npu.synchronize()
    _assert_exact_outputs(repeated, outputs)
    for actual, original in zip(values, cpu_values):
        if isinstance(actual, torch.Tensor):
            assert torch.equal(actual.cpu(), original)
            assert actual.stride() == original.stride()
            assert actual.storage_offset() == original.storage_offset()
