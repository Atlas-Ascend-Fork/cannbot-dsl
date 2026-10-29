# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Four mode cases with self-contained CPU references and v8 comparators.

The two-way BF16 reference and independent three-way FP32 reference retain
their separate entry functions. Shared helpers are AST-identical to both
frozen sources. No reference calls the device operator. Graph audits are
optional execution modes of the same four cases, not extra collected tests.
"""

from __future__ import annotations

import ast
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch_npu

TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT))
from _samples_path import load_sample  # noqa: E402

# CPU reference helpers shared without changing their numerical definitions.

_KV_LORA = 512
_GROUP_SIZE = 128
_BLOCK_SIZE = 128
_FP8_MAX = 448.0
_HIF8_MAX = 32768.0
_GROUP_AMAX_FLOOR = 1.0e-4


class CacheMode(Enum):
    """Public symbolic values; integer encodings are deliberately not exposed."""

    PA_BSND = "PA_BSND"
    PA_NZ = "PA_NZ"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor(name, value, dtype=None, shape=None):
    _require(isinstance(value, torch.Tensor), f"{name} must be a Tensor")
    _require(value.device.type == "cpu", f"{name} must be on CPU for the golden")
    _require(value.is_contiguous(), f"{name} must be contiguous")
    if dtype is not None:
        _require(
            value.dtype == dtype, f"{name}.dtype must be {dtype}, got {value.dtype}"
        )
    if shape is not None:
        _require(
            tuple(value.shape) == tuple(shape),
            f"{name}.shape must be {shape}, got {tuple(value.shape)}",
        )


def _finite_scalar(name, value, positive=False):
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be a real scalar",
    )
    _require(math.isfinite(value), f"{name} must be finite")
    scalar = torch.tensor(value, dtype=torch.float32).item()
    _require(math.isfinite(scalar), f"{name} must be finite in FP32")
    if positive:
        _require(scalar > 0.0, f"{name} must be positive in FP32")
    return scalar


def _scale(value, scalar):
    return value * scalar


def pack_weight_nz(weight):
    """Create a CPU test panel; does not create a genuine NPU NZ tensor."""
    _tensor("weight", weight, torch.bfloat16)
    _require(weight.ndim == 2, "weight must be a logical 2D matrix")
    out_features, in_features = weight.shape
    _require(
        out_features % 16 == 0 and in_features % 16 == 0,
        "NZ fixture dimensions must be multiples of 16",
    )
    return (
        weight.reshape(out_features // 16, 16, in_features // 16, 16)
        .permute(0, 2, 3, 1)
        .contiguous()
    )


def _weight_logical(weight):
    if weight.ndim == 2:
        return weight
    _require(
        weight.ndim == 4 and tuple(weight.shape[-2:]) == (16, 16),
        "CPU NZ fixture must have shape (out/16,in/16,16,16)",
    )
    return (
        weight.permute(0, 3, 1, 2)
        .contiguous()
        .reshape(weight.shape[0] * 16, weight.shape[1] * 16)
    )


def _rms_norm(value, gamma, epsilon):
    """FP32 mathematical RMSNorm, independent of hardware reduction order."""
    value = value.float()
    _require(value.shape[-1] > 0, "RMSNorm width must be positive")
    mean_square = value.square().mean(dim=-1, keepdim=True)
    return value / torch.sqrt(mean_square + epsilon) * gamma.float()


def _rope(value, sin, cos, mode):
    if mode == 0 or value.shape[-1] == 0:
        return value
    sin, cos = sin.float(), cos.float()
    if value.ndim == 3:
        sin, cos = sin[:, None, :], cos[:, None, :]
    half = value.shape[-1] // 2
    if mode == 1:
        even, odd = value[..., 0::2], value[..., 1::2]
        # Interleave-half rotates even/odd pairs and emits two contiguous halves.
        lower = even * cos[..., :half] - odd * sin[..., :half]
        upper = odd * cos[..., half:] + even * sin[..., half:]
        return torch.cat((lower, upper), dim=-1)
    first, second = value[..., :half], value[..., half:]
    rotated = torch.cat((-second, first), dim=-1)
    return value * cos + rotated * sin


def _fp8(value):
    # E4M3FN: nearest-even, finite saturation and NaN -> +0.
    value = value.float().nan_to_num(nan=0.0, posinf=_FP8_MAX, neginf=-_FP8_MAX)
    return value.clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)


def _decode_hif8_values(code):
    """HiF8 sign/dot/exponent/mantissa mapping from the published format."""
    raw = code.to(torch.int64)
    magnitude = raw & 127
    result = torch.zeros_like(raw, dtype=torch.float32)
    for lo, hi, mant_bits, exponent_bits, base in (
        (16, 32, 3, 1, 1),
        (32, 64, 3, 2, 2),
        (64, 96, 2, 3, 4),
        (96, 128, 1, 4, 8),
    ):
        exponent_field = (magnitude >> mant_bits) & ((1 << exponent_bits) - 1)
        exponent_abs = base + (exponent_field & (base - 1))
        exponent = torch.where(
            (exponent_field & base) != 0, -exponent_abs, exponent_abs
        )
        mantissa = (magnitude & ((1 << mant_bits) - 1)).float()
        normal = torch.ldexp(1.0 + mantissa / (1 << mant_bits), exponent.int())
        result = torch.where((magnitude >= lo) & (magnitude < hi), normal, result)
    result = torch.where(
        (magnitude >= 8) & (magnitude < 16), 1.0 + (magnitude & 7).float() / 8.0, result
    )
    denormal = torch.ldexp(torch.ones_like(result), magnitude.int() - 23)
    result = torch.where((magnitude > 0) & (magnitude < 8), denormal, result)
    result = torch.where((raw & 128) != 0, -result, result)
    result = torch.where(
        magnitude == 111,
        torch.copysign(torch.full_like(result, float("inf")), result),
        result,
    )
    return torch.where(raw == 128, torch.full_like(result, float("nan")), result)


def decode_hif8(tensor):
    """Decode CPU uint8 HiF8 bit patterns into FP32 values (no descale applied)."""
    _tensor("HiF8 bytes", tensor, torch.uint8)
    return _decode_hif8_values(tensor)


def _hif8(value):
    # HiF8: nearest, halfway away from zero, finite saturation and NaN -> zero.
    value = value.float()
    codes = torch.arange(128, dtype=torch.int64)
    codes = codes[codes != 111]  # positive infinity is not a saturation target
    levels = _decode_hif8_values(codes)
    order = levels.argsort()
    levels, codes = levels[order], codes[order]
    absolute = value.abs().nan_to_num(nan=0.0, posinf=_HIF8_MAX).clamp(max=_HIF8_MAX)
    upper = torch.searchsorted(levels, absolute.contiguous()).clamp(max=len(levels) - 1)
    lower = (upper - 1).clamp(min=0)
    choose_upper = (absolute - levels[lower]) >= (levels[upper] - absolute)
    selected = torch.where(choose_upper, codes[upper], codes[lower])
    signed = selected | (torch.signbit(value).to(torch.int64) << 7)
    signed = torch.where(selected == 0, 0, signed)  # only one zero encoding
    return signed.to(torch.uint8).contiguous()


def _dynamic_descale(maximum, limit):
    return maximum.float() / limit


def _dynamic_q(value, mode):
    # Statistics use FP32 mathematical Query, before public output conversion.
    value = value.float()
    if mode in (0, 1):
        return value.to(torch.bfloat16), torch.empty(0, dtype=torch.float32)
    if mode == 2:
        maximum = value.abs().amax(dim=-1, keepdim=True)
        descale = _dynamic_descale(maximum, _FP8_MAX)
    else:
        maximum = value.abs().amax().reshape(1)
        descale = _dynamic_descale(maximum, _HIF8_MAX)
    # Protect genuinely zero inputs only, not a nonzero amax's scale underflow.
    # The safe denominator is internal; returned descale remains exactly zero.
    denominator = torch.where(maximum == 0, torch.ones_like(descale), descale)
    normalized = value / denominator
    normalized = torch.where(maximum == 0, torch.zeros_like(normalized), normalized)
    return (_fp8(normalized) if mode == 2 else _hif8(normalized)), descale


def _cache_payload(ckv, kr, mode, quant_scale_kv):
    if mode == 0:
        return torch.cat((ckv, kr), dim=-1).to(torch.bfloat16)
    if mode == 1:
        groups = ckv.reshape(ckv.shape[0], _KV_LORA // _GROUP_SIZE, _GROUP_SIZE)
        descales = _dynamic_descale(
            groups.abs().amax(dim=-1).clamp_min(_GROUP_AMAX_FLOOR), _FP8_MAX
        )
        quantized = _fp8(groups / descales[..., None]).reshape(ckv.shape[0], _KV_LORA)
        kr_bytes = kr.to(torch.bfloat16).contiguous().view(torch.uint8)
        scale_bytes = descales.contiguous().view(torch.uint8)
        return torch.cat((quantized.view(torch.uint8), kr_bytes, scale_bytes), dim=-1)
    scaled = torch.cat((ckv, kr), dim=-1) * quant_scale_kv.item()
    return _fp8(scaled) if mode == 2 else _hif8(scaled)


def _write_cache(cache, index, payload, mode):
    # Commit only after the entire output/payload has been computed.
    bytes_per_row = cache.shape[-1] * cache.element_size()
    raw_cache = cache.view(torch.uint8).reshape(-1)
    raw_rows = payload.contiguous().view(torch.uint8).reshape(len(index), bytes_per_row)
    if mode == CacheMode.PA_BSND:
        rows = raw_cache.reshape(-1, bytes_per_row)
        for token, slot in enumerate(index.tolist()):
            if slot < 0:
                continue
            rows[slot].copy_(raw_rows[token])
    else:
        _require(bytes_per_row % 32 == 0, "PA_NZ payload must be 32-byte aligned")
        physical = raw_cache.reshape(
            cache.shape[0], bytes_per_row // 32, _BLOCK_SIZE, 32
        )
        for token, slot in enumerate(index.tolist()):
            if slot < 0:
                continue
            block, row = divmod(slot, _BLOCK_SIZE)
            physical[block, :, row, :].copy_(raw_rows[token].reshape(-1, 32))


def cache_to_logical(cache, cache_mode):
    """Decode the PA address layout, preserving mixed payload as raw bytes."""
    _tensor("cache", cache)
    _require(isinstance(cache_mode, CacheMode), "cache_mode must be CacheMode")
    _require(
        cache.ndim == 4 and cache.shape[1:3] == (_BLOCK_SIZE, 1),
        "cache shape must be (blocks,128,1,width)",
    )
    if cache_mode == CacheMode.PA_BSND:
        return cache
    row_bytes = cache.shape[-1] * cache.element_size()
    _require(row_bytes % 32 == 0, "PA_NZ row must be 32-byte aligned")
    raw = (
        cache.view(torch.uint8)
        .reshape(cache.shape[0], row_bytes // 32, _BLOCK_SIZE, 32)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return raw.reshape(-1).view(cache.dtype).reshape(cache.shape)


@torch.no_grad()
def golden_mla_prolog(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    quant_scale_kv=None,
    rope_sin=None,
    rope_cos=None,
    *,
    norm_eps=1e-5,
    qkv_quant_mode=0,
    cache_mode=CacheMode.PA_BSND,
    qr_scale=1.0,
    kvr_scale=1.0,
    rope_mode=1,
    return_qr=False,
):
    """Return exactly ``(q, descale_q, qr)``; update ``kv_cache`` in place.

    All compute inputs are CPU tensors. For NPU tests keep independent CPU
    fixtures, and give the device operator separate NPU copies of those data.
    CPU layout/dtype and numerical limitations are documented in this module.
    """
    _require(isinstance(cache_mode, CacheMode), "cache_mode must be a CacheMode enum")
    _require(
        type(qkv_quant_mode) is int and qkv_quant_mode in (0, 1, 2, 3),
        "qkv_quant_mode must be 0/1/2/3",
    )
    _require(
        type(rope_mode) is int and rope_mode in (0, 1, 2), "rope_mode must be 0/1/2"
    )
    _require(type(return_qr) is bool, "return_qr must be bool")
    _require(
        not (qkv_quant_mode == 1 and cache_mode == CacheMode.PA_NZ),
        "mode 1 supports PA_BSND only",
    )
    norm_eps = _finite_scalar("norm_eps", norm_eps, positive=True)
    qr_scale = _finite_scalar("qr_scale", qr_scale)
    kvr_scale = _finite_scalar("kvr_scale", kvr_scale)
    for name, tensor in (
        ("x", x),
        ("wqa", wqa),
        ("wqb", wqb),
        ("wkva", wkva),
        ("wkb", wkb),
    ):
        _tensor(name, tensor, torch.bfloat16)
    _require(x.ndim == 2, "x must have shape (T,dim)")
    _require(
        wqa.ndim == wqb.ndim == wkva.ndim and wqa.ndim in (2, 4),
        "wqa/wqb/wkva must all be ND matrices or all explicit CPU NZ panels",
    )
    qa, qb, kva = map(_weight_logical, (wqa, wqb, wkva))
    t, dim = x.shape
    q_lora = qa.shape[0]
    _require(
        1024 <= dim <= 8192 and dim % 256 == 0, "dim must be 1024..8192, 256 aligned"
    )
    _require(q_lora in (640, 1024, 1536, 2048), "unsupported q_lora")
    _require(wkb.ndim == 3, "wkb must have shape (N,Dn,512)")
    n, dn, kv_lora = wkb.shape
    dr = kva.shape[0] - _KV_LORA
    _require(1 <= n <= 128 and kv_lora == _KV_LORA, "N must be 1..128 and kv_lora=512")
    _require((dn, dr) in ((128, 64), (192, 64), (256, 0)), "unsupported (Dn,Dr)")
    _tensor("wqa logical", qa, shape=(q_lora, dim))
    _tensor("wqb logical", qb, shape=(n * (dn + dr), q_lora))
    _tensor("wkva logical", kva, shape=(_KV_LORA + dr, dim))
    _tensor("norm_weight_qa", norm_weight_qa, torch.float32, (q_lora,))
    _tensor("norm_weight_kva", norm_weight_kva, torch.float32, (_KV_LORA,))
    _tensor("cache_index", cache_index, torch.int64, (t,))
    _tensor("kv_cache", kv_cache)
    _require(
        kv_cache.ndim == 4 and kv_cache.shape[0] > 0,
        "kv_cache must have shape (positive_blocks,128,1,width)",
    )
    width = _KV_LORA + 2 * dr + 16 if qkv_quant_mode == 1 else _KV_LORA + dr
    _require(
        tuple(kv_cache.shape[1:]) == (_BLOCK_SIZE, 1, width),
        f"kv_cache trailing shape must be (128,1,{width})",
    )
    cache_dtype = (
        torch.bfloat16
        if qkv_quant_mode == 0
        else torch.uint8
        if qkv_quant_mode == 3
        else torch.float8_e4m3fn
    )
    _require(
        kv_cache.dtype == cache_dtype,
        f"CPU cache dtype for mode {qkv_quant_mode} must be {cache_dtype}",
    )
    _require(dr != 0 or rope_mode == 0, "Dr=0 forces rope_mode=0")
    if dr and rope_mode:
        _tensor("rope_sin", rope_sin, torch.bfloat16, (t, dr))
        _tensor("rope_cos", rope_cos, torch.bfloat16, (t, dr))
    if qkv_quant_mode in (2, 3):
        _tensor("quant_scale_kv", quant_scale_kv, torch.float32, (1,))
    else:
        _require(quant_scale_kv is None, "mode 0/1 does not take quant_scale_kv")
    if t == 0:
        q_dtype = cache_dtype if qkv_quant_mode != 1 else torch.bfloat16
        return (
            torch.empty((0, n, _KV_LORA + dr), dtype=q_dtype),
            torch.empty(0, dtype=torch.float32),
            torch.empty((0, q_lora) if return_qr else (0,), dtype=torch.bfloat16),
        )
    write_index = cache_index[cache_index >= 0]
    _require(
        torch.unique(write_index).numel() == write_index.numel(),
        "repeated nonnegative cache_index is not allowed",
    )
    _require(
        bool((write_index < kv_cache.shape[0] * _BLOCK_SIZE).all()),
        "nonnegative cache_index is outside the cache capacity",
    )
    if quant_scale_kv is not None:
        _require(
            bool(torch.isfinite(quant_scale_kv).all()) and quant_scale_kv.item() > 0,
            "quant_scale_kv must be finite and positive",
        )
    for name, tensor in (
        ("x", x),
        ("wqa", qa),
        ("wqb", qb),
        ("wkva", kva),
        ("wkb", wkb),
        ("norm_weight_qa", norm_weight_qa),
        ("norm_weight_kva", norm_weight_kva),
    ):
        _require(bool(torch.isfinite(tensor).all()), f"{name} contains NaN/Inf")
    if dr and rope_mode:
        _require(
            bool(torch.isfinite(rope_sin).all())
            and bool(torch.isfinite(rope_cos).all()),
            "active rope_sin/rope_cos contains NaN/Inf",
        )
    _require(
        sys.byteorder == "little", "packed Cache golden requires little-endian CPU"
    )
    cq_raw = (x.float() @ qa.float().T).to(torch.bfloat16).float()
    cq = _scale(_rms_norm(cq_raw, norm_weight_qa, norm_eps), qr_scale).to(
        torch.bfloat16
    )
    q_projected = (
        (cq.float() @ qb.float().T).to(torch.bfloat16).float().reshape(t, n, dn + dr)
    )
    q_nope = torch.einsum("tnd,ndk->tnk", q_projected[..., :dn], wkb.float())
    q_rope = _rope(q_projected[..., dn:], rope_sin, rope_cos, rope_mode)
    q, descale_q = _dynamic_q(torch.cat((q_nope, q_rope), dim=-1), qkv_quant_mode)
    kva_raw = (x.float() @ kva.float().T).to(torch.bfloat16).float()
    ckv = _scale(_rms_norm(kva_raw[:, :_KV_LORA], norm_weight_kva, norm_eps), kvr_scale)
    kr = _rope(kva_raw[:, _KV_LORA:], rope_sin, rope_cos, rope_mode)
    payload = _cache_payload(ckv, kr, qkv_quant_mode, quant_scale_kv)
    qr = cq.to(torch.bfloat16) if return_qr else torch.empty(0, dtype=torch.bfloat16)
    _write_cache(kv_cache, cache_index, payload, cache_mode)
    return q, descale_q, qr


# Independent FP32 entry: never substitute the BF16 reference for G.


@torch.no_grad()
def math_mla_prolog(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    quant_scale_kv=None,
    rope_sin=None,
    rope_cos=None,
    *,
    norm_eps=1e-5,
    qkv_quant_mode=0,
    cache_mode=CacheMode.PA_BSND,
    qr_scale=1.0,
    kvr_scale=1.0,
    rope_mode=1,
    return_qr=False,
):
    """Return exactly ``(q, descale_q, qr)``; update ``kv_cache`` in place.

    All compute inputs are CPU tensors. For NPU tests keep independent CPU
    fixtures, and give the device operator separate NPU copies of those data.
    CPU layout/dtype and numerical limitations are documented in this module.
    """
    _require(isinstance(cache_mode, CacheMode), "cache_mode must be a CacheMode enum")
    _require(
        type(qkv_quant_mode) is int and qkv_quant_mode in (0, 1, 2, 3),
        "qkv_quant_mode must be 0/1/2/3",
    )
    _require(
        type(rope_mode) is int and rope_mode in (0, 1, 2), "rope_mode must be 0/1/2"
    )
    _require(type(return_qr) is bool, "return_qr must be bool")
    _require(
        not (qkv_quant_mode == 1 and cache_mode == CacheMode.PA_NZ),
        "mode 1 supports PA_BSND only",
    )
    norm_eps = _finite_scalar("norm_eps", norm_eps, positive=True)
    qr_scale = _finite_scalar("qr_scale", qr_scale)
    kvr_scale = _finite_scalar("kvr_scale", kvr_scale)
    for name, tensor in (
        ("x", x),
        ("wqa", wqa),
        ("wqb", wqb),
        ("wkva", wkva),
        ("wkb", wkb),
    ):
        _tensor(name, tensor, torch.bfloat16)
    _require(x.ndim == 2, "x must have shape (T,dim)")
    _require(
        wqa.ndim == wqb.ndim == wkva.ndim and wqa.ndim in (2, 4),
        "wqa/wqb/wkva must all be ND matrices or all explicit CPU NZ panels",
    )
    qa, qb, kva = map(_weight_logical, (wqa, wqb, wkva))
    t, dim = x.shape
    q_lora = qa.shape[0]
    _require(
        1024 <= dim <= 8192 and dim % 256 == 0, "dim must be 1024..8192, 256 aligned"
    )
    _require(q_lora in (640, 1024, 1536, 2048), "unsupported q_lora")
    _require(wkb.ndim == 3, "wkb must have shape (N,Dn,512)")
    n, dn, kv_lora = wkb.shape
    dr = kva.shape[0] - _KV_LORA
    _require(1 <= n <= 128 and kv_lora == _KV_LORA, "N must be 1..128 and kv_lora=512")
    _require((dn, dr) in ((128, 64), (192, 64), (256, 0)), "unsupported (Dn,Dr)")
    _tensor("wqa logical", qa, shape=(q_lora, dim))
    _tensor("wqb logical", qb, shape=(n * (dn + dr), q_lora))
    _tensor("wkva logical", kva, shape=(_KV_LORA + dr, dim))
    _tensor("norm_weight_qa", norm_weight_qa, torch.float32, (q_lora,))
    _tensor("norm_weight_kva", norm_weight_kva, torch.float32, (_KV_LORA,))
    _tensor("cache_index", cache_index, torch.int64, (t,))
    _tensor("kv_cache", kv_cache)
    _require(
        kv_cache.ndim == 4 and kv_cache.shape[0] > 0,
        "kv_cache must have shape (positive_blocks,128,1,width)",
    )
    width = _KV_LORA + 2 * dr + 16 if qkv_quant_mode == 1 else _KV_LORA + dr
    _require(
        tuple(kv_cache.shape[1:]) == (_BLOCK_SIZE, 1, width),
        f"kv_cache trailing shape must be (128,1,{width})",
    )
    cache_dtype = (
        torch.bfloat16
        if qkv_quant_mode == 0
        else torch.uint8
        if qkv_quant_mode == 3
        else torch.float8_e4m3fn
    )
    _require(
        kv_cache.dtype == cache_dtype,
        f"CPU cache dtype for mode {qkv_quant_mode} must be {cache_dtype}",
    )
    _require(dr != 0 or rope_mode == 0, "Dr=0 forces rope_mode=0")
    if dr and rope_mode:
        _tensor("rope_sin", rope_sin, torch.bfloat16, (t, dr))
        _tensor("rope_cos", rope_cos, torch.bfloat16, (t, dr))
    if qkv_quant_mode in (2, 3):
        _tensor("quant_scale_kv", quant_scale_kv, torch.float32, (1,))
    else:
        _require(quant_scale_kv is None, "mode 0/1 does not take quant_scale_kv")
    if t == 0:
        q_dtype = cache_dtype if qkv_quant_mode != 1 else torch.bfloat16
        return (
            torch.empty((0, n, _KV_LORA + dr), dtype=q_dtype),
            torch.empty(0, dtype=torch.float32),
            torch.empty((0, q_lora) if return_qr else (0,), dtype=torch.bfloat16),
        )
    write_index = cache_index[cache_index >= 0]
    _require(
        torch.unique(write_index).numel() == write_index.numel(),
        "repeated nonnegative cache_index is not allowed",
    )
    _require(
        bool((write_index < kv_cache.shape[0] * _BLOCK_SIZE).all()),
        "nonnegative cache_index is outside the cache capacity",
    )
    if quant_scale_kv is not None:
        _require(
            bool(torch.isfinite(quant_scale_kv).all()) and quant_scale_kv.item() > 0,
            "quant_scale_kv must be finite and positive",
        )
    for name, tensor in (
        ("x", x),
        ("wqa", qa),
        ("wqb", qb),
        ("wkva", kva),
        ("wkb", wkb),
        ("norm_weight_qa", norm_weight_qa),
        ("norm_weight_kva", norm_weight_kva),
    ):
        _require(bool(torch.isfinite(tensor).all()), f"{name} contains NaN/Inf")
    if dr and rope_mode:
        _require(
            bool(torch.isfinite(rope_sin).all())
            and bool(torch.isfinite(rope_cos).all()),
            "active rope_sin/rope_cos contains NaN/Inf",
        )
    _require(
        sys.byteorder == "little", "packed Cache golden requires little-endian CPU"
    )
    cq_raw = x.float() @ qa.float().T
    cq = _scale(_rms_norm(cq_raw, norm_weight_qa, norm_eps), qr_scale)
    q_projected = (cq @ qb.float().T).reshape(t, n, dn + dr)
    q_nope = torch.einsum("tnd,ndk->tnk", q_projected[..., :dn], wkb.float())
    q_rope = _rope(q_projected[..., dn:], rope_sin, rope_cos, rope_mode)
    q, descale_q = _dynamic_q(torch.cat((q_nope, q_rope), dim=-1), qkv_quant_mode)
    kva_raw = x.float() @ kva.float().T
    ckv = _scale(_rms_norm(kva_raw[:, :_KV_LORA], norm_weight_kva, norm_eps), kvr_scale)
    kr = _rope(kva_raw[:, _KV_LORA:], rope_sin, rope_cos, rope_mode)
    payload = _cache_payload(ckv, kr, qkv_quant_mode, quant_scale_kv)
    qr = cq.to(torch.bfloat16) if return_qr else torch.empty(0, dtype=torch.bfloat16)
    _write_cache(kv_cache, cache_index, payload, cache_mode)
    return q, descale_q, qr


# Lightweight namespaces keep the comparison adapters explicit and unchanged.
_REFERENCE_HELPERS = dict(
    CacheMode=CacheMode,
    _rms_norm=_rms_norm,
    _rope=_rope,
    _dynamic_q=_dynamic_q,
    _cache_payload=_cache_payload,
    decode_hif8=decode_hif8,
    cache_to_logical=cache_to_logical,
)
golden = SimpleNamespace(mla_prolog=golden_mla_prolog, **_REFERENCE_HELPERS)
math_reference = SimpleNamespace(mla_prolog=math_mla_prolog, **_REFERENCE_HELPERS)


def ordinary_cpu(golden, inputs, attrs):
    """Independent CPU Cast chain. No device values or hardware reduction simulation."""
    mode = int(attrs["qkv_quant_mode"])
    x = inputs["x"].float()
    tokens = len(x)
    heads, dn, _ = inputs["wkb"].shape
    dr = inputs["wkva"].shape[0] - 512
    raw = (x @ inputs["wqa"].float().T).bfloat16().float()
    cq = (
        golden._rms_norm(raw, inputs["norm_weight_qa"], attrs["norm_eps"])
        * attrs["qr_scale"]
    ).bfloat16()
    projected = (
        (cq.float() @ inputs["wqb"].float().T)
        .bfloat16()
        .reshape(tokens, heads, dn + dr)
    )
    nope = torch.einsum(
        "tnd,ndk->tnk", projected[..., :dn].float(), inputs["wkb"].float()
    )
    rope = golden._rope(
        projected[..., dn:].float(),
        inputs["rope_sin"],
        inputs["rope_cos"],
        attrs["rope_mode"],
    )
    if mode < 2:
        nope, rope = nope.bfloat16().float(), rope.bfloat16().float()
    q, descale = golden._dynamic_q(torch.cat((nope, rope), -1), mode)
    qr = cq if attrs["return_qr"] else torch.empty(0, dtype=torch.bfloat16)
    kva = (x @ inputs["wkva"].float().T).bfloat16().float()
    ckv = (
        golden._rms_norm(kva[:, :512], inputs["norm_weight_kva"], attrs["norm_eps"])
        * attrs["kvr_scale"]
    )
    kr = golden._rope(
        kva[:, 512:], inputs["rope_sin"], inputs["rope_cos"], attrs["rope_mode"]
    )
    return (q, descale, qr), golden._cache_payload(
        ckv, kr, mode, inputs["quant_scale_kv"]
    )


# Three-way statistics and limits: frozen v8 comparison policy.


DEFAULT_CONFIG = dict(
    small_value=2.0**-10,
    small_value_atol=2.0**-16,
    max_re_rtol=10.0,
    avg_re_rtol=2.0,
    rmse_rtol=2.0,
)


def error_metrics(value, golden, config):
    if value.shape != golden.shape:
        raise ValueError(f"shape mismatch: {value.shape} != {golden.shape}")
    v, g = (
        value.detach().double().flatten().clone(),
        golden.detach().double().flatten().clone(),
    )
    special = {}
    for name, predicate in (
        ("nan", torch.isnan),
        ("inf", torch.isposinf),
        ("ninf", torch.isneginf),
    ):
        special["err_total_" + name] = int((predicate(v) ^ predicate(g)).sum())
    # Same replacement as the supplied statistics, but on independent clones.
    v[~torch.isfinite(v)] = 1
    g[~torch.isfinite(g)] = 1
    big = g.abs() >= config["small_value"]
    error = (v - g).abs()
    rel = error[big] / g[big].abs()
    big_error = torch.where(big, error, 0.0)
    return dict(
        elements=g.numel(),
        big_elements=int(big.sum()),
        small_elements=int((~big).sum()),
        max_re=float(rel.max()) if rel.numel() else 0.0,
        avg_re=float(rel.mean()) if rel.numel() else 0.0,
        small_bad=int((error[~big] > config["small_value_atol"]).sum()),
        rmse=float(big_error.square().mean().sqrt()) if g.numel() else 0.0,
        full_rmse_diagnostic=float(error.square().mean().sqrt()) if g.numel() else 0.0,
        max_abs_diagnostic=float(error.max()) if g.numel() else 0.0,
        **special,
    )


def compare_three_way(npu, cpu, cpu_gold, output_dtype, config=None):
    """Return raw CPU errors separately from the dtype-floor-adjusted limits."""
    cfg = dict(DEFAULT_CONFIG if config is None else config)
    if set(cfg) != set(DEFAULT_CONFIG) or not all(
        isinstance(v, (int, float)) and 0 < v < float("inf") for v in cfg.values()
    ):
        raise ValueError("all five comparison parameters must be finite and positive")
    if output_dtype not in ("bf16", "fp16", "fp32", "hf32", "float8_e4m3fn", "e8m0"):
        raise ValueError("explicit canonical comparison dtype required")
    if npu.shape != cpu.shape or npu.shape != cpu_gold.shape:
        raise ValueError("all three shapes must match")
    d, b = error_metrics(npu, cpu_gold, cfg), error_metrics(cpu, cpu_gold, cfg)
    floor = {"bf16": 2.0**-8, "fp16": 2.0**-11}.get(output_dtype, 0.0)
    effective = dict(b)
    for k in ("max_re", "avg_re", "rmse"):
        effective[k] = max(effective[k], floor)
    limits = dict(
        max_re=effective["max_re"] * cfg["max_re_rtol"],
        avg_re=effective["avg_re"] * cfg["avg_re_rtol"],
        small_bad=effective["small_bad"] * cfg["avg_re_rtol"],
        rmse=effective["rmse"] * cfg["rmse_rtol"],
    )
    errors = [k for k, limit in limits.items() if d[k] > limit]
    warnings = [k for k in limits if d[k] > effective[k] and k not in errors]
    for k in ("err_total_nan", "err_total_inf", "err_total_ninf"):
        if d[k] > b[k]:
            errors.append(k)
        elif d[k] > 0:
            warnings.append(k)
    return dict(
        status="Failed" if errors else "warning" if warnings else "Pass",
        npu=d,
        cpu=b,
        cpu_effective=effective,
        limits=limits,
        errors=errors,
        warnings=warnings,
        output_dtype=output_dtype,
        benchmark_floor=floor,
        config=cfg,
        empty=d["elements"] == 0,
    )


# Two-way dtype-specific comparisons and three-way acceptance.


POLICY = "20261021_mla_prolog_v8_threshold_alignment"


ADAPTATION = "mla_prolog_two_way_then_three_way_warning_accepted"


BM_CMP_STD = {
    name: dict(
        max_re_rtol=10.0,
        avg_re_rtol=2.0,
        rmse_rtol=2.0,
        small_value=0.0009765625,
        small_value_atol=1.52587890625e-05,
    )
    for name in ("fp32", "fp16", "bf16")
}


def _pair(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("comparison requires identical shapes")
    if actual.device.type != "cpu" or expected.device.type != "cpu":
        raise ValueError("comparison requires independent CPU snapshots")


def _record(method, elements, failed, allowed, passed, **details):
    return dict(
        method=method,
        elements=elements,
        failed_elements=failed,
        allowed_failures=allowed,
        passed=bool(passed),
        status="EMPTY" if elements == 0 else "PASS" if passed else "FAIL",
        **details,
    )


def mixed(actual, expected, *, scale=False):
    """BF16 or FP32 scale isclose, segment quota and relative-error hard limit."""
    _pair(actual, expected)
    a, b = actual.double().flatten(), expected.double().flatten()
    m = a.numel()
    atol, rtol = (0.000025, 0.005) if scale else (0.0001, 1 / 128)
    finite = torch.isfinite(a) & torch.isfinite(b)
    same_special = (
        (torch.isnan(a) & torch.isnan(b))
        | (torch.isposinf(a) & torch.isposinf(b))
        | (torch.isneginf(a) & torch.isneginf(b))
    )
    # v8 uses FP64 statistics and torch.isclose, without changing Golden.
    close = torch.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
    bad = ~close
    finite_bad = finite & bad
    denominator = torch.maximum(a.abs(), b.abs()).clamp_min((1 / 16384) / 0.005) + 2e-9
    relative = (a - b).abs() / denominator
    hard = float(relative[finite_bad].max()) if finite_bad.any() else 0.0
    af, bf = a[finite], b[finite]
    delta = (af - bf).abs()
    norm_a, norm_b = torch.linalg.vector_norm(af), torch.linalg.vector_norm(bf)
    nonzero = bf != 0
    failed, allowed = int(bad.sum()), math.floor(m * 0.005)
    return _record(
        1,
        m,
        failed,
        allowed,
        failed <= allowed and hard < 10,
        dtype_rule="FP32 scale" if scale else "BF16",
        atol=atol,
        rtol=rtol,
        pct_thd=0.005,
        max_bad_relative=hard,
        hard_limit=10,
        special_mismatches=int((~finite & ~same_special).sum()),
        max_abs=float(delta.max()) if delta.numel() else None,
        max_rel_nonzero=float((delta[nonzero] / bf[nonzero].abs()).max())
        if nonzero.any()
        else None,
        relative_l2=float(torch.linalg.vector_norm(af - bf) / norm_b)
        if norm_b > 0
        else None,
        cosine=float(torch.dot(af, bf) / (norm_a * norm_b))
        if norm_a > 0 and norm_b > 0
        else None,
    )


def e4m3fn(actual, expected):
    """method10, only explicitly typed E4M3FN payloads (never arbitrary uint8)."""
    _pair(actual, expected)
    if actual.dtype != torch.float8_e4m3fn or expected.dtype != torch.float8_e4m3fn:
        raise TypeError("method10 requires E4M3FN typed payloads")
    a = actual.contiguous().view(torch.uint8).flatten().to(torch.int16)
    b = expected.contiguous().view(torch.uint8).flatten().to(torch.int16)
    a = torch.where((a == 127) | (a == 255), 0, a)
    b = torch.where((b == 127) | (b == 255), 0, b)
    distance = (a - b).abs()
    distance = torch.minimum(distance, 256 - distance)
    signed_zero = ((a == 0) & (b == 128)) | ((a == 128) & (b == 0))
    failed = int((distance > 1).sum())
    hard = int(((distance > 2) & ~signed_zero).sum())
    allowed = a.numel() // 200
    return _record(
        10,
        a.numel(),
        failed,
        allowed,
        failed <= allowed and hard == 0,
        pct_thd=0.005,
        hard_failures=hard,
        signed_zero_pairs=int(signed_zero.sum()),
        max_code_distance=int(distance.max()) if distance.numel() else 0,
    )


def hif8(actual_decoded, reference_fp32, *, diff_thd=0.005):
    """method6 against pre-Cast FP32 in quantization domain, with SAT."""
    if not 0 <= diff_thd <= 1:
        raise ValueError("diff_thd must be finite and in [0, 1]")
    _pair(actual_decoded, reference_fp32)
    if reference_fp32.dtype != torch.float32:
        raise TypeError("HiF8 reference must be pre-Cast FP32, not quantized bytes")
    h = actual_decoded.double().flatten()
    x = (
        reference_fp32.double()
        .flatten()
        .nan_to_num(nan=0.0, posinf=32768.0, neginf=-32768.0)
        .clamp(-32768, 32768)
    )
    ex = torch.log2(x.abs() + 2.0**-1000).clamp_min(-22)
    exponent = ex.floor()
    mantissa = torch.zeros_like(exponent)
    for limit, bits in ((15, 1), (7, 2), (3, 3)):
        mantissa = torch.where(exponent.abs() <= limit, bits, mantissa)
    sign_h = torch.where(h >= 0, 1, -1)
    sign_x = torch.where(x >= 0, 1, -1)
    ulp = torch.where(
        mantissa > 0,
        (h - x) * torch.pow(2.0, -exponent + mantissa),
        torch.log2(h.abs() + 2.0**-1000) * sign_h - ex * sign_x,
    )
    ulp = torch.where(x.abs() < 2.0**-23, torch.where(h == 0, 0.0, 1.5), ulp)
    invalid = ~torch.isfinite(h) | ~torch.isfinite(ulp)
    failures = invalid | (ulp.abs() > 1)
    m = h.numel()
    allowed = m // 1000 + 1 if m > 100 else int(m * diff_thd)
    valid = ulp[~invalid].abs()
    return _record(
        6,
        m,
        int(failures.sum()),
        allowed,
        int(failures.sum()) <= allowed,
        invalid_ulp=int(invalid.sum()),
        max_abs_ulp=float(valid.max()) if valid.numel() else None,
        saturation="SAT",
        short_segment_pct_thd=diff_thd,
        effective_fail_fraction=0.001 + 1 / m if m > 100 else diff_thd,
    )


def golden_with_reference(golden, inputs, attrs):
    """Observe frozen Golden call arguments; never reconstruct pre-Cast from bytes.

    sys.setprofile is thread-local and restored in finally. Only CPU Golden is
    observed, outside any device execution. No function is replaced and no
    intermediate value is mutated. Empty input legitimately has no captures.
    """
    captured = {}
    targets = {
        golden._dynamic_q.__code__: ("q_fp", ("value",)),
        golden._cache_payload.__code__: ("cache_fp", ("ckv", "kr")),
    }

    def observe(frame, event, arg):
        if event == "call" and frame.f_code in targets:
            name, names = targets[frame.f_code]
            if name in captured:
                raise RuntimeError("ambiguous repeated Golden reference capture")
            captured[name] = tuple(
                frame.f_locals[key].detach().clone() for key in names
            )

    previous = sys.getprofile()
    if previous is not None:
        raise RuntimeError("Golden capture requires no active Python profiler")
    try:
        sys.setprofile(observe)
        output = golden.mla_prolog(**inputs, **attrs)
    finally:
        sys.setprofile(previous)
    if inputs["x"].shape[0] and set(captured) != {"q_fp", "cache_fp"}:
        raise RuntimeError("missing pre-quantization Golden reference")
    return output, captured


def compare_numeric_outputs(
    golden, actual, expected, cache_actual, cache_expected, inputs, attrs, captured
):
    """All numeric segments, independent quotas; structural checks stay in runner."""
    mode = int(attrs["qkv_quant_mode"])
    layout = golden.CacheMode[attrs["cache_mode"].name]
    dr = inputs["wkva"].shape[0] - 512
    results = {}
    if mode == 3 and inputs["x"].shape[0]:
        q_fp = captured["q_fp"][0]
        denominator = torch.where(expected[1] == 0, 1.0, expected[1])
        q_reference = torch.where(expected[1] == 0, 0.0, q_fp / denominator)
    else:
        q_reference = torch.empty_like(expected[0], dtype=torch.float32)
    for label, section in (("q_nope", slice(0, 512)), ("q_rope", slice(512, None))):
        a, b = actual[0][..., section], expected[0][..., section]
        results[label] = (
            mixed(a, b)
            if mode < 2
            else e4m3fn(a, b)
            if mode == 2
            else hif8(golden.decode_hif8(a.contiguous()), q_reference[..., section])
        )
    results["descale_q"] = mixed(actual[1], expected[1], scale=True)
    results["qr"] = mixed(actual[2], expected[2])
    valid = inputs["cache_index"] >= 0
    hits = inputs["cache_index"][valid]
    width = cache_actual.shape[-1]
    a = golden.cache_to_logical(cache_actual, layout).reshape(-1, width)[hits]
    b = golden.cache_to_logical(cache_expected, layout).reshape(-1, width)[hits]
    if mode == 1:
        a, b = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        results["kv"] = e4m3fn(
            a[:, :512].contiguous().view(torch.float8_e4m3fn),
            b[:, :512].contiguous().view(torch.float8_e4m3fn),
        )
        results["kr"] = mixed(
            a[:, 512 : 512 + 2 * dr].contiguous().view(torch.bfloat16),
            b[:, 512 : 512 + 2 * dr].contiguous().view(torch.bfloat16),
        )
        sa = a[:, 512 + 2 * dr :].contiguous().view(torch.float32)
        sb = b[:, 512 + 2 * dr :].contiguous().view(torch.float32)
        # v8 treats the four group scales as one independent FP32 scale segment.
        results["cache_descale"] = mixed(sa, sb, scale=True)
    else:
        if mode == 3 and inputs["x"].shape[0]:
            cache_ref = (
                torch.cat(captured["cache_fp"], dim=-1)[valid]
                * inputs["quant_scale_kv"].item()
            )
        else:
            cache_ref = torch.empty_like(b, dtype=torch.float32)
        for label, section in (("kv", slice(0, 512)), ("kr", slice(512, None))):
            av, bv = a[..., section], b[..., section]
            results[label] = (
                mixed(av, bv)
                if mode == 0
                else e4m3fn(av, bv)
                if mode == 2
                else hif8(golden.decode_hif8(av.contiguous()), cache_ref[..., section])
            )
    return results


def precision_status(value):
    """Missing/malformed comparisons must not silently become a successful run."""
    records = []

    def visit(item):
        if isinstance(item, dict):
            if "method" in item and "passed" in item:
                records.append(item)
            else:
                for child in item.values():
                    visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    if not records:
        return "UNVERIFIED"
    if any(not record["passed"] for record in records):
        return "FAIL"
    return "PASS" if any(record["elements"] for record in records) else "EMPTY"


def three_way(
    device,
    ordinary_cpu,
    cpu_gold,
    dtype,
    *,
    golden_inf_switch=False,
    inf_clip_switch=False,
    nan_toZero_switch=False,
):
    """v8 numeric rules; preserve this operator's independent CPU reference contract."""
    _pair(device, cpu_gold)
    _pair(ordinary_cpu, cpu_gold)
    if golden_inf_switch or inf_clip_switch or nan_toZero_switch:
        raise ValueError("v8 performs no optional Inf clipping or NaN-to-zero")
    dtype_name = {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
        torch.float64: "fp32",
        torch.float8_e4m3fn: "float8_e4m3fn",
    }.get(dtype)
    if dtype_name is None:
        raise ValueError(
            "unsupported three-way logical dtype; byte carrier is not a dtype policy"
        )
    result = compare_three_way(device, ordinary_cpu, cpu_gold, dtype_name)
    return dict(
        method=12,
        diagnostic_only=False,
        **result,
        configuration=result["config"],
        entry="v8/compare_three_way",
        input_dtypes=dict(
            device=str(device.dtype),
            cpu=str(ordinary_cpu.dtype),
            gold=str(cpu_gold.dtype),
        ),
    )


def acceptance_status(direct, fallback, *, structural_pass=True):
    """User-authorized whole-output fallback, never pointwise selection."""
    if not structural_pass:
        return "FAIL"
    if direct in ("PASS", "EMPTY"):
        return "PASS"
    return "PASS" if direct == "FAIL" and fallback in ("Pass", "warning") else "FAIL"


policy = SimpleNamespace(
    _pair=_pair,
    _record=_record,
    mixed=mixed,
    e4m3fn=e4m3fn,
    hif8=hif8,
    golden_with_reference=golden_with_reference,
    compare_outputs=compare_numeric_outputs,
    precision_status=precision_status,
    three_way=three_way,
    acceptance_status=acceptance_status,
)


def load_math_reference(golden):
    """Return the independent frozen FP32 entry, never the BF16 entry."""
    verify_reference_integrity()
    return math_reference


def fallback_segments(golden, inputs, attrs, actual, after, expected, captured):
    """D/C/G use the same inputs; G is never the lower-precision two-way reference."""
    # Retain the call signature, but deliberately rebuild G independently.
    # A cloned Cache prevents the high-precision reference from changing the
    # two-way Cache result or any caller-owned tensor.
    math_reference = load_math_reference(golden)
    math_inputs = dict(inputs, kv_cache=inputs["kv_cache"].clone())
    math_attrs = dict(
        attrs, cache_mode=math_reference.CacheMode[attrs["cache_mode"].name]
    )
    expected, captured = policy.golden_with_reference(
        math_reference, math_inputs, math_attrs
    )
    mode = int(attrs["qkv_quant_mode"])
    layout = golden.CacheMode[attrs["cache_mode"].name]
    ordinary, payload = ordinary_cpu(golden, inputs, attrs)
    decode = golden.decode_hif8 if mode == 3 else lambda t: t.float()
    d, c = decode(actual[0].contiguous()), decode(ordinary[0].contiguous())
    g = expected[0].float()
    if mode >= 2:
        denominator = torch.where(expected[1] == 0, 1.0, expected[1])
        g = torch.where(expected[1] == 0, 0.0, captured["q_fp"][0] / denominator)
        if mode == 3:
            g = g.nan_to_num(nan=0.0, posinf=32768.0, neginf=-32768.0).clamp(
                -32768, 32768
            )
    triples = {}
    qdtype = torch.float32 if mode == 3 else actual[0].dtype
    for name, section in (("q_nope", slice(0, 512)), ("q_rope", slice(512, None))):
        triples[name] = (d[..., section], c[..., section], g[..., section], qdtype)
    for name, i in (("descale_q", 1), ("qr", 2)):
        triples[name] = (
            actual[i].float(),
            ordinary[i].float(),
            expected[i].float(),
            actual[i].dtype,
        )
    valid = inputs["cache_index"] >= 0
    hits = inputs["cache_index"][valid]
    width = after.shape[-1]
    da = golden.cache_to_logical(after, layout).reshape(-1, width)[hits]
    ge = golden.cache_to_logical(math_inputs["kv_cache"], layout).reshape(-1, width)[
        hits
    ]
    ca = payload[valid]
    dr = inputs["wkva"].shape[0] - 512
    if mode == 1:
        da, ca, ge = (v.contiguous().view(torch.uint8) for v in (da, ca, ge))
        for name, start, end, dtype in (
            ("kv", 0, 512, torch.float8_e4m3fn),
            ("kr", 512, 512 + 2 * dr, torch.bfloat16),
        ):
            triples[name] = (
                *[
                    v[:, start:end].contiguous().view(dtype).float()
                    for v in (da, ca, ge)
                ],
                dtype,
            )
        triples["cache_descale"] = (
            *[
                v[:, 512 + 2 * dr :].contiguous().view(torch.float32)
                for v in (da, ca, ge)
            ],
            torch.float32,
        )
    else:
        dv, cv, gv = (decode(v.contiguous()) for v in (da, ca, ge))
        dtype = torch.float32 if mode == 3 else da.dtype
        for name, section in (("kv", slice(0, 512)), ("kr", slice(512, None))):
            triples[name] = (
                dv[..., section],
                cv[..., section],
                gv[..., section],
                dtype,
            )
    required = {"q_nope", "q_rope", "qr", "descale_q", "kv", "kr"}
    if mode == 1:
        required.add("cache_descale")
    if set(triples) != required:
        raise ValueError("missing or extra three-way segment")
    records = {name: policy.three_way(*values) for name, values in triples.items()}
    states = {r["status"] for r in records.values()}
    status = (
        "Failed" if "Failed" in states else "warning" if "warning" in states else "Pass"
    )
    return (
        dict(
            status=status,
            segments=records,
            reference="D=device; C=independent single-BF16 CPU chain; G=frozen Golden; quantized Q uses pre-Cast normalized G",
            hif8_binding=(
                "decoded numeric domain, FP32 comparison with zero floor; SAT reference retained"
                if mode == 3
                else None
            ),
        ),
        triples,
    )


# Freeze numerical definitions, not the formatting or file organization.
_FROZEN_NUMERIC_AST = "da9d231dda8f0247fc36ec97bfa89c92e2764738dc7453fb9d3ab0f8da2153da"
_FROZEN_NUMERIC_NAMES = (
    "_KV_LORA",
    "_GROUP_SIZE",
    "_BLOCK_SIZE",
    "_FP8_MAX",
    "_HIF8_MAX",
    "_GROUP_AMAX_FLOOR",
    "ordinary_cpu",
    "CacheMode",
    "_require",
    "_tensor",
    "_finite_scalar",
    "_scale",
    "pack_weight_nz",
    "_weight_logical",
    "_rms_norm",
    "_rope",
    "_fp8",
    "_decode_hif8_values",
    "decode_hif8",
    "_hif8",
    "_dynamic_descale",
    "_dynamic_q",
    "_cache_payload",
    "_write_cache",
    "cache_to_logical",
    "golden_mla_prolog",
    "math_mla_prolog",
    "DEFAULT_CONFIG",
    "error_metrics",
    "compare_three_way",
    "POLICY",
    "ADAPTATION",
    "BM_CMP_STD",
    "_pair",
    "_record",
    "mixed",
    "e4m3fn",
    "hif8",
    "golden_with_reference",
    "compare_numeric_outputs",
    "precision_status",
    "three_way",
    "acceptance_status",
)


def verify_reference_integrity():
    tree = ast.parse(Path(__file__).read_text())
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            definitions[node.targets[0].id] = node
    # Python 3.12 adds an empty type_params field to non-generic definitions.
    # Normalize that metadata only; preserve all numerical AST fields.
    payload = "\n".join(
        ast.dump(definitions[name]).replace(", type_params=[]", "")
        for name in _FROZEN_NUMERIC_NAMES
    )
    assert hashlib.sha256(payload.encode()).hexdigest() == _FROZEN_NUMERIC_AST, (
        "Frozen reference or precision policy changed"
    )


# Device fixtures, graph audit and the four public-mode cases.
op = load_sample("mla_prolog/mla_prolog.py")
pytestmark = pytest.mark.npu
CASES = (
    (0, "ND", "PA_BSND", 128, 64, 1, True),
    (1, "NZ", "PA_BSND", 192, 64, 2, False),
    (2, "NZ", "PA_NZ", 128, 64, 1, False),
    (3, "ND", "PA_NZ", 256, 0, 0, True),
)


def make_inputs(case, device):
    mode, weight_format, cache_mode, dn, dr, rope_mode, return_qr = case
    rng = torch.Generator().manual_seed(20261027 + mode)
    tokens, dim, q_lora, heads = 3, 1024, 640, 2

    def random_bf16(shape):
        return (torch.randn(shape, generator=rng) * 0.1).bfloat16()

    cache_dtype = (
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        torch.uint8,
    )[mode]
    width = 512 + 2 * dr + 16 if mode == 1 else 512 + dr
    cpu = dict(
        x=random_bf16((tokens, dim)),
        wqa=random_bf16((q_lora, dim)),
        wqb=random_bf16((heads * (dn + dr), q_lora)),
        wkva=random_bf16((512 + dr, dim)),
        wkb=random_bf16((heads, dn, 512)),
        norm_weight_qa=torch.linspace(0.75, 1.25, q_lora),
        norm_weight_kva=torch.linspace(0.75, 1.25, 512),
        kv_cache=torch.full((2, 128, 1, width), 0.25).to(cache_dtype),
        cache_index=torch.tensor([128, 0, 127], dtype=torch.int64),
        quant_scale_kv=torch.tensor([1.25]) if mode >= 2 else None,
        rope_sin=random_bf16((tokens, dr)) if dr else None,
        rope_cos=random_bf16((tokens, dr)) if dr else None,
    )
    cpu["x"] *= torch.tensor([0.5, 1.0, 2.0], dtype=torch.bfloat16)[:, None]
    actual = {
        name: value.to(device) if value is not None else None
        for name, value in cpu.items()
    }
    if weight_format == "NZ":
        for name in ("wqa", "wqb", "wkva"):
            actual[name] = torch_npu.npu_format_cast(actual[name], 29)
            assert torch_npu.get_npu_format(actual[name]) == 29
    attrs = dict(
        norm_eps=1e-5,
        qkv_quant_mode=mode,
        cache_mode=golden.CacheMode[cache_mode],
        qr_scale=1.75,
        kvr_scale=0.625,
        rope_mode=rope_mode,
        return_qr=return_qr,
    )
    public_attrs = dict(
        attrs,
        qkv_quant_mode=op.QkvQuantMode(mode),
        cache_mode=op.CacheMode[cache_mode],
        rope_mode=op.RopeMode(rope_mode),
    )
    return cpu, actual, attrs, public_attrs


def cache_identity(cache):
    return (
        cache.data_ptr(),
        cache.untyped_storage()._cdata,
        cache.dtype,
        tuple(cache.shape),
    )


def read_trace(path):
    payload = json.loads(path.read_text())
    events = payload if isinstance(payload, list) else payload["traceEvents"]
    assert events, "Empty trace cannot prove absence of copies"
    return events


def audit_trace(trace, execution):
    events = read_trace(trace)
    tasks = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("args", {}).get("Task Type")
    ]
    models = {
        event["args"].get("Model Id")
        for event in tasks
        if event["args"]["Task Type"] == "MODEL_EXECUTE"
    }
    assert models, "No real graph replay observed"
    children = [event for event in tasks if event["args"].get("Model Id") in models]
    kernels = [
        event
        for event in children
        if event["args"]["Task Type"]
        in ("AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV")
        or event["args"]["Task Type"].startswith("KERNEL_")
    ]
    expected = 3 if execution == "npugraph-ex" else 1
    assert len(kernels) == expected, (
        f"Expected {expected} graph kernels, got {len(kernels)}"
    )
    assert all("MlaPrologKernel" in event["name"] for event in kernels), kernels
    forbidden_device = [
        event
        for event in tasks
        if any(
            token in (event["name"] + " " + str(event["args"])).lower()
            for token in (
                "tensormove",
                "foreachcopy",
                "memcpy",
                "copyback",
                "copy_back",
            )
        )
    ]
    windows = [
        (float(event["ts"]), float(event["ts"]) + float(event["dur"]))
        for event in events
        if event.get("name") == "MLAP_REPLAY" and event.get("ph") == "X"
    ]
    assert windows, "Replay scope is missing"
    forbidden_host = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("name") in ("aten::clone", "aten::copy_", "aten::_to_copy")
        and any(start <= float(event["ts"]) < end for start, end in windows)
    ]
    assert not forbidden_device and not forbidden_host, (
        forbidden_device,
        forbidden_host,
    )
    return dict(
        status="PASS",
        graph_models=sorted(models),
        graph_kernels=len(kernels),
        forbidden_device_moves=0,
        replay_host_copies=0,
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
    )


def run_entry(actual, attrs, execution, trace_dir):
    # Preflight and compilation are outside capture; replay must not read values to CPU.
    result = op.mla_prolog(**actual, **attrs)
    torch.npu.synchronize()
    if execution == "eager":
        assert trace_dir is None, "Trace audit requires aclgraph or npugraph-ex"
        return result, None, None
    if execution == "aclgraph":
        graph = torch.npu.NPUGraph()

        def capture_and_replay():
            with torch.npu.graph(graph):
                output = op.mla_prolog(**actual, **attrs)
            with torch.profiler.record_function("MLAP_REPLAY"):
                graph.replay()
            return output

        def replay():
            graph.replay()
            return result
    elif execution == "npugraph-ex":
        torch.compiler.reset()
        compiled = torch.compile(
            op.mla_prolog,
            backend="npugraph_ex",
            fullgraph=True,
            dynamic=False,
            options={"force_eager": False, "clone_input": False},
        )

        def replay():
            return compiled(**actual, **attrs)

        def capture_and_replay():
            with torch.profiler.record_function("MLAP_CAPTURE"):
                output = replay()
            with torch.profiler.record_function("MLAP_REPLAY"):
                output = replay()
                output = replay()
            return output
    else:
        raise ValueError("MLA_PROLOG_EXECUTION must be eager, aclgraph or npugraph-ex")
    if trace_dir is None:
        result = capture_and_replay()
        torch.npu.synchronize()
        return result, replay, None
    trace_dir.mkdir(parents=True, exist_ok=False)
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            data_simplification=False,
        ),
    ) as profiler:
        result = capture_and_replay()
        torch.npu.synchronize()
        profiler.step()
    trace = trace_dir / "trace.json"
    profiler.export_chrome_trace(str(trace))
    return result, replay, audit_trace(trace, execution)


def compare_outputs(cpu, actual, attrs, outputs, before):
    expected, captured = policy.golden_with_reference(golden, cpu, attrs)
    host = tuple(value.cpu() for value in outputs)
    after = actual["kv_cache"].cpu()
    for value, reference in zip(host, expected):
        assert value.shape == reference.shape and value.dtype == reference.dtype
    direct = policy.compare_outputs(
        golden,
        host,
        expected,
        after,
        cpu["kv_cache"],
        cpu,
        attrs,
        captured,
    )
    direct_status = policy.precision_status(direct)
    report = dict(two_way=direct, category="TWO_WAY_PASS")
    if direct_status == "FAIL":
        fallback, _ = fallback_segments(
            golden, cpu, attrs, host, after, expected, captured
        )
        report.update(three_way=fallback, category="THREE_WAY_PASS")
        assert fallback["status"] in ("Pass", "warning"), report
    else:
        assert direct_status == "PASS", report
    layout = attrs["cache_mode"]
    old_bytes = (
        golden.cache_to_logical(before, layout)
        .contiguous()
        .view(torch.uint8)
        .reshape(256, -1)
    )
    new_bytes = (
        golden.cache_to_logical(after, layout)
        .contiguous()
        .view(torch.uint8)
        .reshape(256, -1)
    )
    untouched = torch.ones(256, dtype=torch.bool)
    untouched[cpu["cache_index"]] = False
    assert torch.equal(old_bytes[untouched], new_bytes[untouched]), (
        "Untouched Cache bytes changed"
    )
    return report


def check_changed_input_replay(cpu, actual, attrs, replay, original, before):
    original_q = original[0].cpu().contiguous().view(torch.uint8).clone()
    actual["x"].copy_((-cpu["x"]).to(actual["x"].device))
    actual["cache_index"].copy_(torch.tensor([255, 129, 1], device=actual["x"].device))
    actual["kv_cache"].copy_(before.to(actual["x"].device))
    updated = replay()
    torch.npu.synchronize()
    observed = [value.cpu().contiguous().view(torch.uint8).clone() for value in updated]
    cache = actual["kv_cache"].cpu().view(torch.uint8).clone()
    actual["kv_cache"].copy_(before.to(actual["x"].device))
    expected = op.mla_prolog(**actual, **attrs)
    torch.npu.synchronize()
    assert not torch.equal(original_q, observed[0]), "Replay reused stale Q"
    assert all(
        torch.equal(value, ref.cpu().contiguous().view(torch.uint8))
        for value, ref in zip(observed, expected)
    )
    assert torch.equal(cache, actual["kv_cache"].cpu().view(torch.uint8)), (
        "Replay reused stale Cache indices"
    )


@pytest.mark.parametrize("name", ("norm_eps", "qr_scale", "kvr_scale"))
@pytest.mark.parametrize("value", (True, False, "1.0", None, 1j))
@pytest.mark.parametrize("compiling", (False, True))
def test_public_scalar_types(name, value, compiling, monkeypatch):
    def forbidden_dispatch(*args, **kwargs):
        pytest.fail("Invalid Python scalar reached the Torch dispatcher")

    namespace = torch.ops.cannbot_dsl_mla_prolog
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    monkeypatch.setattr(namespace.mla_prolog, "default", forbidden_dispatch)
    monkeypatch.setattr(namespace._mla_prolog_graph, "default", forbidden_dispatch)
    with pytest.raises(TypeError, match=name):
        op.mla_prolog(*([torch.empty(0)] * 9), **{name: value})


@pytest.mark.parametrize(
    "epsilon", (0.0, -1.0, float("nan"), float("inf"), 1e40, 1e-50)
)
def test_public_invalid_epsilon(epsilon):
    library = torch.library.Library("cannbot_dsl_mla_prolog", "IMPL", "CPU")
    library.impl("mla_prolog", op.mla_prolog_npu)
    with pytest.raises(ValueError, match="norm_eps"):
        op.mla_prolog(*([torch.empty(0)] * 9), norm_eps=epsilon)


def test_public_scalar_defaults(monkeypatch):
    observed = []

    def capture_dispatch(*args, **kwargs):
        observed.append(kwargs)
        return ()

    monkeypatch.setattr(
        torch.ops.cannbot_dsl_mla_prolog.mla_prolog, "default", capture_dispatch
    )
    op.mla_prolog(*([torch.empty(0)] * 9))
    assert observed[-1]["norm_eps"] == 1e-5
    op.mla_prolog(*([torch.empty(0)] * 9), norm_eps=1, qr_scale=0, kvr_scale=-1)
    assert observed[-1]["qr_scale"] == 0
    assert observed[-1]["kvr_scale"] == -1


@pytest.mark.parametrize("mode", range(4))
@pytest.mark.parametrize("return_qr", (False, True))
@pytest.mark.parametrize("tokens", (0, 1))
def test_output_metadata(mode, return_qr, tokens):
    meta = SimpleNamespace(
        tokens=tokens,
        heads=2,
        kvr=576,
        q_lora=640,
        qkv_quant_mode=mode,
        return_qr=return_qr,
    )
    expected_scale = (
        (tokens, 2, 1) if mode == 2 else (1,) if mode == 3 and tokens else (0,)
    )
    expected_shapes = (
        (tokens, 2, 576),
        expected_scale,
        (tokens, 640) if return_qr else (0,),
    )
    expected_dtypes = (
        (torch.bfloat16, torch.bfloat16, torch.float8_e4m3fn, torch.uint8)[mode],
        torch.float32,
        torch.bfloat16,
    )
    fake_inputs = dict(
        x=torch.empty((tokens, 1024), device="meta"),
        wqa=torch.empty((640, 1024), device="meta"),
        wqb=torch.empty((384, 640), device="meta"),
        wkva=torch.empty((576, 1024), device="meta"),
        wkb=torch.empty((2, 128, 512), device="meta"),
        norm_weight_qa=torch.empty(640, device="meta"),
        norm_weight_kva=torch.empty(512, device="meta"),
        kv_cache=torch.empty(0, device="meta"),
        cache_index=torch.empty(tokens, device="meta"),
    )
    for outputs in (
        op._empty_outputs(meta, "cpu"),
        op.mla_prolog_fake(**fake_inputs, qkv_quant_mode=mode, return_qr=return_qr),
    ):
        assert tuple(tuple(value.shape) for value in outputs) == expected_shapes
        assert tuple(value.dtype for value in outputs) == expected_dtypes


@pytest.mark.parametrize("mode", range(4))
@pytest.mark.parametrize("return_qr", (False, True))
def test_empty_public_entry(mode, return_qr, monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    device = int(os.environ.get("MLA_PROLOG_DEVICE", "0"))
    torch.npu.set_device(device)
    case = (mode, "ND", "PA_BSND", 128, 64, 1, return_qr)
    _, actual, _, attrs = make_inputs(case, torch.device(f"npu:{device}"))
    for name in ("x", "cache_index", "rope_sin", "rope_cos"):
        actual[name] = actual[name][:0]
    before = actual["kv_cache"].cpu().view(torch.uint8).clone()
    identity = cache_identity(actual["kv_cache"])

    def forbidden_launch(*args, **kwargs):
        pytest.fail("Empty input reached DSL workspace allocation or Kernel launch")

    monkeypatch.setattr(op, "_launch", forbidden_launch)
    q, descale, qr = op.mla_prolog(**actual, **attrs)
    assert tuple(q.shape) == (0, 2, 576)
    assert (
        q.dtype
        == (torch.bfloat16, torch.bfloat16, torch.float8_e4m3fn, torch.uint8)[mode]
    )
    assert tuple(descale.shape) == ((0, 2, 1) if mode == 2 else (0,))
    assert descale.dtype == torch.float32
    assert tuple(qr.shape) == ((0, 640) if return_qr else (0,))
    assert qr.dtype == torch.bfloat16
    assert cache_identity(actual["kv_cache"]) == identity
    assert torch.equal(actual["kv_cache"].cpu().view(torch.uint8), before)
    with pytest.raises(ValueError, match="norm_eps"):
        op.mla_prolog(**actual, **dict(attrs, norm_eps=0.0))
    with pytest.raises(ValueError, match="cache_index"):
        op.mla_prolog(
            **dict(actual, cache_index=actual["cache_index"].reshape(0, 1)), **attrs
        )


@pytest.mark.parametrize("case", CASES, ids=("mode0", "mode1", "mode2", "mode3"))
def test_mla_prolog(case):
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    verify_reference_integrity()
    torch.set_num_threads(4)
    device = int(os.environ.get("MLA_PROLOG_DEVICE", "0"))
    torch.npu.set_device(device)
    cpu, actual, attrs, public_attrs = make_inputs(case, torch.device(f"npu:{device}"))
    before = cpu["kv_cache"].clone()
    identity = cache_identity(actual["kv_cache"])
    execution = os.environ.get("MLA_PROLOG_EXECUTION", "eager")
    output_root = os.environ.get("MLA_PROLOG_TRACE_DIR")
    trace_dir = (
        Path(output_root) / f"mode{case[0]}-{execution}" if output_root else None
    )
    outputs, replay, audit = run_entry(actual, public_attrs, execution, trace_dir)
    assert cache_identity(actual["kv_cache"]) == identity
    report = compare_outputs(cpu, actual, attrs, outputs, before)
    if replay is not None:
        check_changed_input_replay(cpu, actual, public_attrs, replay, outputs, before)
        assert cache_identity(actual["kv_cache"]) == identity
    report.update(
        mode=case[0],
        execution=execution,
        graph_audit=audit,
        source_sha256=hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest(),
    )
    if trace_dir is not None:
        (trace_dir / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )
    print(json.dumps(report, ensure_ascii=False), flush=True)
