# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Four public-interface cases with CPU Golden and complete precision gates.

Golden describes mathematical quantization, not device scheduling. HiF8 uses
raw uint8 carriers. Cache is a physical storage shadow; only untouched rows
are compared byte-for-byte. Three-way statistics retain the supplied source's
special-value switches, dtype arithmetic and warning decisions.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import logging
import math
from pathlib import Path
import struct
import sys
from unittest.mock import patch

import pytest
import torch

LOGGER = logging.getLogger(__name__)


MXFP8_GROUP_SIZE = 32


CACHE_GROUP_SIZE = 128


E4M3FN_QMAX = 448.0


HIF8_QMAX = 32768.0


CACHE_TILE_AMAX_FLOOR = 1.0e-4


KV_LORA = 512


VALID_Q_LORA = {640, 1024, 1536, 2048}


VALID_DN_DR = {(128, 64), (192, 64), (256, 0)}


VALID_MODE_PAIRS = {(0, 1), (0, 2), (1, 1), (2, 1), (3, 2)}


def _native_dtype(name: str):
    return getattr(torch, name, None)


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).lower()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_cpu(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is not None:
        _require(
            tensor.device.type == "cpu", f"{name} must be a CPU tensor for the golden"
        )


def _require_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    _require(
        tuple(tensor.shape) == tuple(shape),
        f"{name}.shape must be {shape}, got {tuple(tensor.shape)}",
    )


def _is_e4m3fn(tensor: torch.Tensor) -> bool:
    native = _native_dtype("float8_e4m3fn")
    return (
        native is not None and tensor.dtype == native
    ) or tensor.dtype == torch.uint8


def _is_e8m0(tensor: torch.Tensor) -> bool:
    native = _native_dtype("float8_e8m0fnu")
    return (
        native is not None and tensor.dtype == native
    ) or tensor.dtype == torch.uint8


def _is_hif8_carrier(tensor: torch.Tensor) -> bool:
    name = _dtype_name(tensor)
    return (
        tensor.dtype in (torch.uint8, torch.int8)
        or "hifloat8" in name
        or "hif8" in name
    )


def _raw_u8(tensor: torch.Tensor) -> torch.Tensor:
    _require(tensor.is_contiguous(), "byte-carrier tensor must be contiguous")
    if tensor.dtype == torch.uint8:
        return tensor
    return tensor.view(torch.uint8)


def _bf16_sat_input(value: torch.Tensor) -> torch.Tensor:
    """Local BF16 SAT semantics: NaN -> +0; overflow -> signed finite limit."""
    limit = torch.finfo(torch.bfloat16).max
    return torch.nan_to_num(value.float(), nan=0.0, posinf=limit, neginf=-limit).clamp(
        -limit, limit
    )


def _bf16_round_away(value: torch.Tensor) -> torch.Tensor:
    """FP32 -> BF16 SAT/ROUND: nearest, ties away, preserving signed zero."""
    value = _bf16_sat_input(value).contiguous()
    bits = value.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rounded = ((bits + 0x8000) >> 16) & 0xFFFF
    bf16_bits = rounded.to(torch.int16)
    return bf16_bits.view(torch.bfloat16).to(torch.float32)


def _e4_positive_table(device: torch.device) -> torch.Tensor:
    code = torch.arange(127, dtype=torch.int64, device=device)
    exp = code >> 3
    mant = code & 7
    subnormal = mant.to(torch.float32) * (2.0**-9)
    normal = torch.ldexp(1.0 + mant.to(torch.float32) / 8.0, exp.to(torch.int32) - 7)
    return torch.where(exp == 0, subnormal, normal)


def _e4m3fn_decode(tensor: torch.Tensor) -> torch.Tensor:
    native = _native_dtype("float8_e4m3fn")
    if native is not None and tensor.dtype == native:
        return tensor.to(torch.float32)
    raw = _raw_u8(tensor).to(torch.int64)
    mag = raw & 0x7F
    exp = mag >> 3
    mant = mag & 7
    subnormal = mant.to(torch.float32) * (2.0**-9)
    normal = torch.ldexp(1.0 + mant.to(torch.float32) / 8.0, exp.to(torch.int32) - 7)
    value = torch.where(exp == 0, subnormal, normal)
    value = torch.where((raw & 0x80) != 0, -value, value)
    return torch.where(mag == 0x7F, torch.full_like(value, float("nan")), value)


def _e4m3fn_encode_raw(value: torch.Tensor) -> torch.Tensor:
    """FP32 -> E4M3FN with saturate and round-to-nearest-even."""
    value = value.to(torch.float32)
    sign = torch.signbit(value)
    absolute = value.abs()
    table = _e4_positive_table(value.device)
    finite = torch.nan_to_num(absolute, nan=0.0, posinf=E4M3FN_QMAX, neginf=E4M3FN_QMAX)
    finite = finite.clamp(max=E4M3FN_QMAX)

    high = torch.bucketize(finite, table).clamp(max=table.numel() - 1)
    low = (high - 1).clamp(min=0)
    low_value = table[low]
    high_value = table[high]
    low_distance = finite - low_value
    high_distance = high_value - finite
    tie = low_distance == high_distance
    choose_high = (high_distance < low_distance) | (tie & ((high & 1) == 0))
    magnitude = torch.where(choose_high, high, low).to(torch.uint8)
    raw = magnitude | (sign.to(torch.uint8) << 7)
    # Quantization SAT maps NaN to zero; decoding an input NaN is distinct.
    raw = torch.where(torch.isnan(value), torch.zeros_like(raw), raw)
    return raw.contiguous()


def _e4m3fn_from_raw(raw: torch.Tensor) -> torch.Tensor:
    native = _native_dtype("float8_e4m3fn")
    raw = raw.to(torch.uint8).contiguous()
    return raw.view(native) if native is not None else raw


def _e4m3fn_quantize(value: torch.Tensor) -> torch.Tensor:
    return _e4m3fn_from_raw(_e4m3fn_encode_raw(value))


def _e8m0_decode(tensor: torch.Tensor) -> torch.Tensor:
    native = _native_dtype("float8_e8m0fnu")
    if native is not None and tensor.dtype == native:
        return tensor.to(torch.float32)
    raw = _raw_u8(tensor).to(torch.int32)
    value = torch.ldexp(torch.ones_like(raw, dtype=torch.float32), raw - 127)
    return torch.where(raw == 255, torch.full_like(value, float("nan")), value)


def _e8m0_from_raw(raw: torch.Tensor) -> torch.Tensor:
    native = _native_dtype("float8_e8m0fnu")
    raw = raw.to(torch.uint8).contiguous()
    return raw.view(native) if native is not None else raw


def _hif8_decode_raw(raw_tensor: torch.Tensor) -> torch.Tensor:
    raw = _raw_u8(raw_tensor).to(torch.int64)
    magnitude = raw & 0x7F
    sign = torch.where((raw & 0x80) != 0, -1.0, 1.0)
    value = torch.zeros_like(magnitude, dtype=torch.float32)

    subnormal = magnitude < 8
    subnormal_value = torch.where(
        magnitude == 0,
        torch.zeros_like(value),
        torch.ldexp(torch.ones_like(value), magnitude.to(torch.int32) - 23),
    )
    value = torch.where(subnormal, subnormal_value, value)

    d0001 = (magnitude >= 8) & (magnitude < 16)
    value = torch.where(d0001, 1.0 + (magnitude & 7).to(torch.float32) / 8.0, value)

    d001 = (magnitude >= 16) & (magnitude < 32)
    exp_bits = (magnitude >> 3) & 1
    exponent = torch.where(
        exp_bits == 0, torch.ones_like(exp_bits), -torch.ones_like(exp_bits)
    )
    normal = torch.ldexp(
        1.0 + (magnitude & 7).to(torch.float32) / 8.0, exponent.to(torch.int32)
    )
    value = torch.where(d001, normal, value)

    d01 = (magnitude >= 32) & (magnitude < 64)
    exp_bits = (magnitude >> 3) & 3
    exp_abs = 2 + (exp_bits & 1)
    exponent = torch.where((exp_bits >> 1) == 0, exp_abs, -exp_abs)
    normal = torch.ldexp(
        1.0 + (magnitude & 7).to(torch.float32) / 8.0, exponent.to(torch.int32)
    )
    value = torch.where(d01, normal, value)

    d10 = (magnitude >= 64) & (magnitude < 96)
    exp_bits = (magnitude >> 2) & 7
    exp_abs = 4 + (exp_bits & 3)
    exponent = torch.where((exp_bits >> 2) == 0, exp_abs, -exp_abs)
    normal = torch.ldexp(
        1.0 + (magnitude & 3).to(torch.float32) / 4.0, exponent.to(torch.int32)
    )
    value = torch.where(d10, normal, value)

    d11 = magnitude >= 96
    exp_bits = (magnitude >> 1) & 15
    exp_abs = 8 + (exp_bits & 7)
    exponent = torch.where((exp_bits >> 3) == 0, exp_abs, -exp_abs)
    normal = torch.ldexp(
        1.0 + (magnitude & 1).to(torch.float32) / 2.0, exponent.to(torch.int32)
    )
    value = torch.where(d11, normal, value)

    value = value * sign
    value = torch.where(magnitude == 111, sign * float("inf"), value)
    value = torch.where(raw == 128, torch.full_like(value, float("nan")), value)
    return value


def _hif8_positive_table(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    codes = torch.arange(128, dtype=torch.uint8, device=device)
    finite_mask = codes != 111
    finite_codes = codes[finite_mask]
    values = _hif8_decode_raw(finite_codes)
    order = torch.argsort(values, stable=True)
    return values[order], finite_codes[order]


def _hif8_encode_raw(value: torch.Tensor) -> torch.Tensor:
    """FP32 -> HiF8: saturation and nearest, ties away from zero."""
    value = value.to(torch.float32)
    sign = torch.signbit(value)
    absolute = value.abs()
    table, codes = _hif8_positive_table(value.device)
    # SAT must be applied before nearest-value selection. 40960 -> Inf is a
    # non-saturating conversion rule and must not be used by this operator.
    finite = torch.nan_to_num(absolute, nan=0.0, posinf=HIF8_QMAX).clamp(max=HIF8_QMAX)

    high = torch.bucketize(finite, table).clamp(max=table.numel() - 1)
    low = (high - 1).clamp(min=0)
    low_distance = finite - table[low]
    high_distance = table[high] - finite
    choose_high = high_distance <= low_distance
    magnitude = torch.where(choose_high, codes[high], codes[low]).to(torch.uint8)
    raw = magnitude | (sign.to(torch.uint8) << 7)
    raw = torch.where(torch.isnan(value), torch.zeros_like(raw), raw)
    # HiF8 reserves 0x80 for NaN: finite values rounded to magnitude zero
    # must use 0x00 even when their source sign is negative (device SAT+RNA).
    raw = torch.where(
        (magnitude == 0) & torch.isfinite(value), torch.zeros_like(raw), raw
    )
    return raw.contiguous()


def _hif8_quantize(value: torch.Tensor) -> torch.Tensor:
    return _hif8_encode_raw(value)


def _empty_quantized(w_qr_quant_mode: int, *, scale: bool) -> torch.Tensor:
    if w_qr_quant_mode == 1:
        if scale:
            return _e8m0_from_raw(torch.empty((0,), dtype=torch.uint8))
        return _e4m3fn_from_raw(torch.empty((0,), dtype=torch.uint8))
    return torch.empty((0,), dtype=torch.float32 if scale else torch.uint8)


def _unpack_nz_weight(weight: torch.Tensor) -> torch.Tensor:
    # Excel physical shape: (O/32, K/16, 16, 32) == (O1, K1, K0, O0).
    o1, k1, k0, o0 = weight.shape
    _require(k0 == 16 and o0 == 32, "NZ weight tail must be (16,32)")
    return weight.permute(0, 3, 1, 2).contiguous().reshape(o1 * o0, k1 * k0)


def _unpack_mxfp8_weight_scale(scale: torch.Tensor, weight_format: str) -> torch.Tensor:
    if weight_format == "ND":
        # (O,K/64,2) -> (O,K/32)
        return _e8m0_decode(scale).reshape(scale.shape[0], -1)
    # (O/16,K/64,16,2) == (O1,K1,O0,Kscale0) -> (O,K/32)
    o1, k1, o0, scale0 = scale.shape
    _require(o0 == 16 and scale0 == 2, "NZ MXFP8 weight descale tail must be (16,2)")
    raw = _e8m0_decode(scale)
    return raw.permute(0, 2, 1, 3).contiguous().reshape(o1 * o0, k1 * scale0)


def _dequant_mxfp8_activation(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, width = data.shape
    _require(
        width % MXFP8_GROUP_SIZE == 0, "MXFP8 activation K must be divisible by 32"
    )
    decoded = _e4m3fn_decode(data).reshape(
        rows, width // MXFP8_GROUP_SIZE, MXFP8_GROUP_SIZE
    )
    return (decoded * _e8m0_decode(scale).reshape(rows, -1, 1)).reshape(rows, width)


def _dequant_mxfp8_weight(
    data: torch.Tensor, scale: torch.Tensor, weight_format: str
) -> torch.Tensor:
    logical = data if weight_format == "ND" else _unpack_nz_weight(data)
    rows, width = logical.shape
    decoded = _e4m3fn_decode(logical).reshape(
        rows, width // MXFP8_GROUP_SIZE, MXFP8_GROUP_SIZE
    )
    logical_scale = _unpack_mxfp8_weight_scale(scale, weight_format)
    return (decoded * logical_scale.unsqueeze(-1)).reshape(rows, width)


def _mxfp8_projection_result(value: torch.Tensor) -> torch.Tensor:
    """Apply the user-approved MX FP32 output subnormal rule (2026-10-20).

    Preserve FP32 accumulation/output and all non-subnormal values; do not
    add a BF16 conversion or change the process-wide floating-point mode.
    """
    _require(value.dtype == torch.float32, "MX projection result must be FP32")
    magnitude = value.abs()
    subnormal = (magnitude > 0) & (magnitude < torch.finfo(torch.float32).tiny)
    return torch.where(subnormal, torch.zeros_like(value), value)


def _mxfp8_matmul(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_descale: torch.Tensor,
    weight_descale: torch.Tensor,
    weight_format: str,
) -> torch.Tensor:
    """MX mathematical reference with FP32 projection outputs.

    Finite encoded operands can represent values outside FP32 although their
    products fit FP32. If expansion introduces Inf, use FP64 range for this
    CPU reference operation, then convert the projection to FP32. Ordinary
    representable operands retain the existing FP32 Matmul. Both paths apply
    the agreed FP32 output subnormal rule. This is not a model of device
    accumulation order or a device recomputation branch.
    """
    left = _dequant_mxfp8_activation(activation, activation_descale)
    right = _dequant_mxfp8_weight(weight, weight_descale, weight_format)
    if not (torch.isinf(left).any() or torch.isinf(right).any()):
        return _mxfp8_projection_result(left @ right.T)
    logical = weight if weight_format == "ND" else _unpack_nz_weight(weight)
    left = (
        _e4m3fn_decode(activation).double().reshape(activation.shape[0], -1, 32)
        * _e8m0_decode(activation_descale).double().reshape(activation.shape[0], -1, 1)
    ).reshape(activation.shape)
    right = (
        _e4m3fn_decode(logical).double().reshape(logical.shape[0], -1, 32)
        * _unpack_mxfp8_weight_scale(weight_descale, weight_format)
        .double()
        .unsqueeze(-1)
    ).reshape(logical.shape)
    return _mxfp8_projection_result((left @ right.T).float())


def _hif8_matmul_dequant(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_descale: torch.Tensor,
    weight_descale: torch.Tensor,
    weight_format: str,
) -> torch.Tensor:
    """FP32 coded-value Matmul -> weight descale -> activation descale.

    Multiplications are deliberately separate: moving scales into the two
    operands or merging them changes FP32 rounding before the next BF16 cast.
    This interface uses one global activation descale, not one per token.
    """
    logical = weight if weight_format == "ND" else _unpack_nz_weight(weight)
    coded_result = _hif8_decode_raw(activation) @ _hif8_decode_raw(logical).T
    channel_dequant = coded_result * weight_descale.to(torch.float32).reshape(1, -1)
    return channel_dequant * activation_descale.to(torch.float32).reshape(1, 1)


def _mxfp8_dynamic_quant(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BF16 -> E4M3FN + E8M0 per 32 elements, using numerical exponents.

    BF16 pre-rounding and the BF16 normalized-value rounding are preserved
    numerical conversion points, not a simulation of vector instructions.
    """
    _require(
        value.shape[-1] % MXFP8_GROUP_SIZE == 0,
        "MXFP8 dynamic-quant width must divide by 32",
    )
    value_bf16 = _bf16_round_away(value)
    groups = value_bf16.reshape(*value_bf16.shape[:-1], -1, MXFP8_GROUP_SIZE)
    amax = groups.abs().amax(dim=-1)
    has_special = ~torch.isfinite(amax)
    zero_exponent = amax < torch.finfo(torch.bfloat16).tiny
    safe_amax = torch.where(has_special | zero_exponent, torch.ones_like(amax), amax)
    # frexp exponent e gives floor(log2(amax))=e-1. E8M0 byte is floor+119.
    _, exponent = torch.frexp(safe_amax)
    scale_raw = (exponent + 118).clamp(min=0, max=254).to(torch.uint8)
    scale_raw = torch.where(has_special, torch.full_like(scale_raw, 255), scale_raw)
    scale_raw = torch.where(zero_exponent, torch.zeros_like(scale_raw), scale_raw)
    inverse = torch.ldexp(torch.ones_like(amax), 127 - scale_raw.to(torch.int32))
    inverse = torch.where(zero_exponent, torch.zeros_like(inverse), inverse)
    inverse = torch.where(has_special, torch.full_like(inverse, float("nan")), inverse)
    normalized = (groups * inverse.unsqueeze(-1)).to(torch.bfloat16).float()
    quantized = _e4m3fn_quantize(normalized.reshape_as(value_bf16))
    scale = _e8m0_decode(_e8m0_from_raw(scale_raw))
    reconstructed = _e4m3fn_decode(quantized).reshape_as(groups) * scale.unsqueeze(-1)
    return quantized, _e8m0_from_raw(scale_raw), reconstructed.reshape_as(value_bf16)


def _normalize_dynamic(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Guard only zero scales; leave NaN/Inf arithmetic to Torch."""
    zero_scale = scale == 0
    denominator = torch.where(zero_scale, torch.ones_like(scale), scale)
    return torch.where(zero_scale, torch.zeros_like(value), value / denominator)


def _hif8_dynamic_quant(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    amax = value.abs().amax().to(torch.float32)
    scale = (amax / HIF8_QMAX).reshape(1)
    normalized = _normalize_dynamic(value, scale)
    quantized = _hif8_quantize(normalized)
    reconstructed = _hif8_decode_raw(quantized) * scale
    return quantized, scale, reconstructed


def _e4_per_row_quant(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = value.abs().amax(dim=-1, keepdim=True)
    scale = amax / E4M3FN_QMAX
    normalized = _normalize_dynamic(value, scale)
    return _e4m3fn_quantize(normalized), scale.to(torch.float32)


def _e4_per_cache_group_quant(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _require(
        value.shape[-1] % CACHE_GROUP_SIZE == 0,
        "KV width must be divisible by cache group size 128",
    )
    groups = value.reshape(value.shape[0], -1, CACHE_GROUP_SIZE)
    amax = groups.abs().amax(dim=-1)
    scale = amax.clamp_min(CACHE_TILE_AMAX_FLOOR) / E4M3FN_QMAX
    normalized = groups / scale.unsqueeze(-1)
    quantized = _e4m3fn_quantize(normalized.reshape_as(value))
    return quantized, scale.to(torch.float32)


def _rms_norm(value: torch.Tensor, gamma: torch.Tensor, epsilon: float) -> torch.Tensor:
    value = value.float()
    # Mathematical Torch reference, not a hardware lane/reduction-tree model.
    mean_square = value.square().mean(dim=-1, keepdim=True)
    return value / torch.sqrt(mean_square + epsilon) * gamma.float()


def _apply_norm_scale(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply the public scale once after RMSNorm, in FP32."""
    scale_fp32 = torch.tensor(scale, dtype=torch.float32)
    _require(bool(torch.isfinite(scale_fp32)), "norm scale must be finite in FP32")
    return value * scale_fp32


def _apply_rope(
    value: torch.Tensor,
    rope_sin: torch.Tensor | None,
    rope_cos: torch.Tensor | None,
    rope_mode: int,
    cache_sat: bool = False,
) -> torch.Tensor:
    if value.shape[-1] == 0 or rope_mode == 0:
        if cache_sat:
            value = _bf16_sat_input(value)
        return value.to(torch.bfloat16).to(torch.float32)
    sin = rope_sin.to(torch.float32)
    cos = rope_cos.to(torch.float32)
    while sin.ndim < value.ndim:
        sin = sin.unsqueeze(-2)
        cos = cos.unsqueeze(-2)
    if rope_mode == 1:
        # Interleave-half: pair even/odd values and concatenate two half regions.
        even = value[..., 0::2]
        odd = value[..., 1::2]
        half = value.shape[-1] // 2
        lower = cos[..., :half] * even - sin[..., :half] * odd
        upper = sin[..., half:] * even + cos[..., half:] * odd
        result = torch.cat((lower, upper), dim=-1)
    else:
        # Standard half mode requested by the interface: rotate(x)=[-x2,x1].
        first, second = value.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        result = value * cos + rotated * sin
    if cache_sat:
        result = _bf16_sat_input(result)
    return result.to(torch.bfloat16).to(torch.float32)


def _validate_cache_storage(kv_cache: torch.Tensor, cache_mode: int) -> int:
    _require_cpu("kv_cache", kv_cache)
    _require(cache_mode in (0, 1), "cache_mode must be 0(PA_BSND) or 1(PA_NZ)")
    _require(
        kv_cache.ndim == 4 and kv_cache.shape[2] == 1,
        "cache storage shadow must have public shape (block_number,block_size,1,D)",
    )
    _require(
        kv_cache.is_contiguous(), "physical cache storage shadow must be contiguous"
    )
    _require(kv_cache.element_size() in (1, 2), "cache must use 1B or 2B elements")
    c0 = 32 // kv_cache.element_size()
    if cache_mode == 1:
        _require(kv_cache.shape[-1] % c0 == 0, "PA_NZ width must be C0-aligned")
    return c0


def logical_cache_to_storage(
    logical_cache: torch.Tensor, cache_mode: int
) -> torch.Tensor:
    """CPU test helper: allocate a physical storage shadow from logical BSND.

    Keep the public shape/dtype; only permute raw bytes. This CPU helper is NOT
    a device format cast or a new argument to the public operator.
    """
    c0 = _validate_cache_storage(logical_cache, cache_mode)
    if cache_mode == 0:
        return logical_cache.clone()
    blocks, block_size, _, width = logical_cache.shape
    raw = _raw_u8(logical_cache).reshape(
        blocks, block_size, width // c0, c0, logical_cache.element_size()
    )
    packed = raw.permute(0, 2, 1, 3, 4).contiguous().reshape(-1)
    return packed.view(logical_cache.dtype).reshape_as(logical_cache)


def cache_to_logical(kv_cache: torch.Tensor, cache_mode: int) -> torch.Tensor:
    """CPU test helper: allocate a logical_cache; never use it as raw PA_NZ."""
    c0 = _validate_cache_storage(kv_cache, cache_mode)
    if cache_mode == 0:
        return kv_cache.clone()
    blocks, block_size, _, width = kv_cache.shape
    raw = _raw_u8(kv_cache).reshape(
        blocks, width // c0, block_size, c0, kv_cache.element_size()
    )
    logical = raw.permute(0, 2, 1, 3, 4).contiguous().reshape(-1)
    return logical.view(kv_cache.dtype).reshape_as(kv_cache)


def cache_to_values(
    kv_cache: torch.Tensor,
    cache_mode: int,
    qkv_quant_mode: int,
    quant_scale_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    """CPU comparison helper: FP32 logical/dequantized combined KV/KR cache.

    This is NOT a public operator argument or output. It unpacks mode-1 mixed
    payload sections and applies their per-group descales; modes 2/3 divide by
    a positive input quantization multiplier. At zero, return decoded payload
    values for comparison: the original values cannot be recovered.
    Shape is (blocks,block_size,1,512+Dr).
    Compare nonnegative cache_index locations; negative indices skip Cache writes.
    Untouched slots may be uninitialized.
    """
    _require(qkv_quant_mode in (0, 1, 2, 3), "qkv_quant_mode must be 0..3")
    _require(
        not (qkv_quant_mode == 1 and cache_mode == 1),
        "qkv_quant_mode=1 does not support PA_NZ",
    )
    logical = cache_to_logical(kv_cache, cache_mode)
    width = logical.shape[-1]
    dr = (width - KV_LORA - 16) // 2 if qkv_quant_mode == 1 else width - KV_LORA
    _require(dr in (0, 64), "cache width must match Dr=0 or Dr=64")
    if qkv_quant_mode == 0:
        _require(logical.dtype == torch.bfloat16, "mode-0 cache must be BF16")
        return logical.float()
    if qkv_quant_mode == 1:
        _require(width == KV_LORA + 2 * dr + 16, "mode-1 cache width mismatch")
        _require(logical.element_size() == 1, "mode-1 cache needs a byte carrier")
        raw = _raw_u8(logical)
        kv = _e4m3fn_decode(raw[..., :KV_LORA].contiguous())
        kr = (
            raw[..., KV_LORA : KV_LORA + 2 * dr]
            .contiguous()
            .view(torch.bfloat16)
            .float()
        )
        descale = raw[..., KV_LORA + 2 * dr :].contiguous().view(torch.float32)
        groups = kv.reshape(*kv.shape[:-1], 4, CACHE_GROUP_SIZE)
        kv = (groups * descale.unsqueeze(-1)).reshape_as(kv)
        return torch.cat((kv, kr), dim=-1)
    _require(quant_scale_kv is not None, "mode-2/3 cache requires quant_scale_kv")
    _require_cpu("quant_scale_kv", quant_scale_kv)
    _require_shape("quant_scale_kv", quant_scale_kv, (1,))
    _require(quant_scale_kv.dtype == torch.float32, "quant_scale_kv must be FP32")
    _require(
        bool(torch.isfinite(quant_scale_kv).all() and (quant_scale_kv >= 0).all()),
        "quant_scale_kv must be finite and nonnegative",
    )
    decoded = (
        _e4m3fn_decode(logical) if qkv_quant_mode == 2 else _hif8_decode_raw(logical)
    )
    return decoded if quant_scale_kv.item() == 0 else decoded / quant_scale_kv


def _store_cache_rows(
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    rows: torch.Tensor,
    cache_mode: int = 0,
) -> None:
    c0 = _validate_cache_storage(kv_cache, cache_mode)
    block_number, block_size = kv_cache.shape[:2]
    width = kv_cache.shape[-1]
    _require(rows.shape == (cache_index.numel(), width), "cache row shape mismatch")
    _require(rows.dtype == kv_cache.dtype, "cache row dtype must match storage dtype")
    # Raw-byte copies support native float8 and preserve signed zeros / NaN bits.
    row_bytes = _raw_u8(rows.contiguous()).reshape(cache_index.numel(), -1)
    cache_bytes = _raw_u8(kv_cache)
    if cache_mode == 1:
        physical = cache_bytes.reshape(block_number, width // c0, block_size, 32)
    else:
        physical = cache_bytes.reshape(block_number, block_size, -1)
    index = cache_index.to(torch.int64)
    for token in range(index.numel()):
        position = int(index[token].item())
        # Negative indices suppress only the Cache write, not Q/QR computation.
        # Check before address arithmetic, including for INT64_MIN.
        if position < 0:
            continue
        block = position // block_size
        offset = position % block_size
        _require(
            0 <= block < block_number,
            f"cache_index[{token}]={position} is out of range",
        )
        if cache_mode == 1:
            # Element offset: block*block_size*D + (d//C0)*block_size*C0
            #                 + offset*C0 + d%C0.
            physical[block, :, offset, :].copy_(row_bytes[token].reshape(-1, 32))
        else:
            physical[block, offset].copy_(row_bytes[token])


def _store_cache_bytes(
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    payload: torch.Tensor,
    cache_mode: int = 0,
) -> None:
    _require(
        kv_cache.element_size() == 1,
        "opaque cache carrier must use one byte per element",
    )
    _store_cache_rows(
        _raw_u8(kv_cache), cache_index, payload.to(torch.uint8), cache_mode
    )


def quant_mla_prolog_golden(
    x: torch.Tensor,
    wqa: torch.Tensor,
    wqb: torch.Tensor,
    wkva: torch.Tensor,
    wkb: torch.Tensor,
    descale_x: torch.Tensor,
    descale_wqa: torch.Tensor,
    descale_wqb: torch.Tensor,
    descale_wkva: torch.Tensor,
    norm_weight_qa: torch.Tensor,
    norm_weight_kva: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    quant_scale_kv: torch.Tensor | None = None,
    rope_sin: torch.Tensor | None = None,
    rope_cos: torch.Tensor | None = None,
    *,
    norm_eps: float = 1.0e-5,
    w_qr_quant_mode: int,
    qkv_quant_mode: int,
    cache_mode: int,
    qr_scale: float = 1.0,
    kvr_scale: float = 1.0,
    rope_mode: int = 1,
    return_qr: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(q, descale_q, qr, descale_qr)`` and mutate ``kv_cache``.

    The declared argument order is exactly the Excel order: all INPUT/INOUT
    tensors followed by ATTR values. Inactive RoPE inputs may be None or a valid pair;
    ATTR values are keyword-only so mode selection cannot be shifted by mistake.
    ``kv_cache`` contains physical bytes for the selected cache_mode; plain
    ``kv_cache[block,row,0]`` is NOT a logical row when cache_mode=1.
    Negative cache_index skips the entire corresponding Cache payload only.
    Filter indices >= 0 before comparing Cache values; compare all Q/QR rows.
    """
    tensors = {
        "x": x,
        "wqa": wqa,
        "wqb": wqb,
        "wkva": wkva,
        "wkb": wkb,
        "descale_x": descale_x,
        "descale_wqa": descale_wqa,
        "descale_wqb": descale_wqb,
        "descale_wkva": descale_wkva,
        "norm_weight_qa": norm_weight_qa,
        "norm_weight_kva": norm_weight_kva,
        "kv_cache": kv_cache,
        "cache_index": cache_index,
        "quant_scale_kv": quant_scale_kv,
        "rope_sin": rope_sin,
        "rope_cos": rope_cos,
    }
    for name, tensor in tensors.items():
        _require_cpu(name, tensor)

    _require(
        (qkv_quant_mode, w_qr_quant_mode) in VALID_MODE_PAIRS,
        f"unsupported (qkv_quant_mode,w_qr_quant_mode)=({qkv_quant_mode},{w_qr_quant_mode})",
    )
    _require(cache_mode in (0, 1), "cache_mode must be 0(PA_BSND) or 1(PA_NZ)")
    _require(
        not (qkv_quant_mode == 1 and cache_mode == 1),
        "qkv_quant_mode=1 does not support PA_NZ",
    )
    _require(rope_mode in (0, 1, 2), "rope_mode must be 0, 1, or 2")
    if type(norm_eps) not in (float, int):
        raise TypeError("norm_eps must be a Python float or int, not bool or Tensor")
    try:
        norm_eps = struct.unpack("f", struct.pack("f", norm_eps))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(
            "norm_eps is not representable as positive finite FP32"
        ) from error
    _require(
        math.isfinite(norm_eps) and norm_eps > 0,
        "norm_eps must be finite and strictly positive",
    )
    for name, value in (("qr_scale", qr_scale), ("kvr_scale", kvr_scale)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a Python float or int, not bool or Tensor")
        try:
            fp32_value = struct.unpack("f", struct.pack("f", float(value)))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError(f"{name} must be finite in FP32") from error
        _require(
            math.isfinite(fp32_value),
            f"{name} must be finite in FP32",
        )
    _require(x.ndim == 2, "x must be (T,dim)")

    t, dim = x.shape
    q_lora = norm_weight_qa.numel()
    kv_lora = norm_weight_kva.numel()
    _require(
        1024 <= dim <= 8192 and dim % 256 == 0,
        "dim must be [1024,8192] and 256-aligned",
    )
    _require(q_lora in VALID_Q_LORA, f"q_lora must be one of {sorted(VALID_Q_LORA)}")
    _require(kv_lora == KV_LORA, "kv_lora must be 512")
    _require(wkb.ndim == 3, "wkb must be (N,Dn,kv_lora)")
    n, dn, wkb_kv = wkb.shape
    _require(1 <= n <= 128, "N must be in [1,128]")
    _require(wkb_kv == kv_lora, "wkb.shape[2] must equal kv_lora")

    ranks = (wqa.ndim, wqb.ndim, wkva.ndim)
    _require(
        ranks in ((2, 2, 2), (4, 4, 4)),
        "wqa/wqb/wkva must be uniformly ND or uniformly NZ",
    )
    weight_format = "ND" if ranks[0] == 2 else "NZ"
    logical_wqa_shape = (
        tuple(wqa.shape)
        if weight_format == "ND"
        else (wqa.shape[0] * 32, wqa.shape[1] * 16)
    )
    logical_wqb_shape = (
        tuple(wqb.shape)
        if weight_format == "ND"
        else (wqb.shape[0] * 32, wqb.shape[1] * 16)
    )
    logical_wkva_shape = (
        tuple(wkva.shape)
        if weight_format == "ND"
        else (wkva.shape[0] * 32, wkva.shape[1] * 16)
    )
    _require(logical_wqa_shape == (q_lora, dim), "wqa logical shape mismatch")
    _require(
        logical_wqb_shape[1] == q_lora and logical_wqb_shape[0] % n == 0,
        "wqb logical shape mismatch",
    )
    dr = logical_wkva_shape[0] - kv_lora
    _require(logical_wkva_shape == (kv_lora + dr, dim), "wkva logical shape mismatch")
    _require((dn, dr) in VALID_DN_DR, f"(Dn,Dr)=({dn},{dr}) is unsupported")
    _require(
        logical_wqb_shape == (n * (dn + dr), q_lora), "wqb must be (N*(Dn+Dr),q_lora)"
    )
    if dr == 0:
        _require(rope_mode == 0, "Dr=0 requires rope_mode=0")

    _require_shape("norm_weight_qa", norm_weight_qa, (q_lora,))
    _require_shape("norm_weight_kva", norm_weight_kva, (kv_lora,))
    _require(
        norm_weight_qa.dtype == torch.float32
        and norm_weight_kva.dtype == torch.float32,
        "norm weights must be FP32",
    )
    _require(wkb.dtype == torch.bfloat16, "wkb must be BF16")
    _require(cache_index.dtype == torch.int64, "cache_index must be INT64")
    _require_shape("cache_index", cache_index, (t,))
    _require(
        kv_cache.ndim == 4 and kv_cache.shape[2] == 1,
        "kv_cache must be (block_number,block_size,1,D)",
    )
    _require(
        kv_cache.shape[0] > 0 and kv_cache.shape[1] > 0,
        "cache block_number/block_size must be positive",
    )
    _validate_cache_storage(kv_cache, cache_mode)

    if (dr and rope_mode) or rope_sin is not None or rope_cos is not None:
        _require(
            rope_sin is not None and rope_cos is not None,
            "rope_sin/rope_cos must be supplied together for active RoPE or omitted together",
        )
        _require_shape("rope_sin", rope_sin, (t, dr))
        _require_shape("rope_cos", rope_cos, (t, dr))
        _require(
            rope_sin.dtype == torch.bfloat16 and rope_cos.dtype == torch.bfloat16,
            "rope_sin/rope_cos must be BF16",
        )

    if qkv_quant_mode in (2, 3):
        _require(
            quant_scale_kv is not None,
            "quant_scale_kv is required for qkv_quant_mode=2/3",
        )
        _require_shape("quant_scale_kv", quant_scale_kv, (1,))
        _require(quant_scale_kv.dtype == torch.float32, "quant_scale_kv must be FP32")
        if t != 0:
            kv_scale = float(quant_scale_kv.item())
            _require(
                math.isfinite(kv_scale) and kv_scale >= 0,
                "quant_scale_kv must be finite and nonnegative",
            )

    dtile = kv_lora + 2 * dr + 4 * math.ceil(kv_lora / CACHE_GROUP_SIZE)
    expected_cache_tail = dtile if qkv_quant_mode == 1 else kv_lora + dr
    _require(
        kv_cache.shape[-1] == expected_cache_tail,
        f"kv_cache last dim must be {expected_cache_tail} for qkv_quant_mode={qkv_quant_mode}",
    )
    if qkv_quant_mode == 0:
        _require(kv_cache.dtype == torch.bfloat16, "mode 0 kv_cache must be BF16")
    elif qkv_quant_mode in (1, 2):
        _require(
            _is_e4m3fn(kv_cache),
            "mode 1/2 kv_cache must be E4M3FN or uint8 raw carrier",
        )
    else:
        _require(
            _is_hif8_carrier(kv_cache), "mode 3 kv_cache must be HiF8 byte carrier"
        )

    if w_qr_quant_mode == 1:
        _require(
            all(_is_e4m3fn(tensor) for tensor in (x, wqa, wqb, wkva)),
            "mode 1 x/wqa/wqb/wkva must be E4M3FN or uint8 raw carriers",
        )
        _require(
            all(
                _is_e8m0(tensor)
                for tensor in (descale_x, descale_wqa, descale_wqb, descale_wkva)
            ),
            "mode 1 descales must be E8M0 or uint8 raw carriers",
        )
        _require_shape("descale_x", descale_x, (t, dim // 32))
        if weight_format == "ND":
            _require_shape("descale_wqa", descale_wqa, (q_lora, dim // 64, 2))
            _require_shape("descale_wqb", descale_wqb, (n * (dn + dr), q_lora // 64, 2))
            _require_shape("descale_wkva", descale_wkva, (kv_lora + dr, dim // 64, 2))
        else:
            _require_shape("descale_wqa", descale_wqa, (q_lora // 16, dim // 64, 16, 2))
            _require_shape(
                "descale_wqb", descale_wqb, (n * (dn + dr) // 16, q_lora // 64, 16, 2)
            )
            _require_shape(
                "descale_wkva", descale_wkva, ((kv_lora + dr) // 16, dim // 64, 16, 2)
            )
    else:
        _require(
            all(_is_hif8_carrier(tensor) for tensor in (x, wqa, wqb, wkva)),
            "mode 2 x/wqa/wqb/wkva must be HiF8 byte carriers",
        )
        _require(
            all(
                tensor.dtype == torch.float32
                for tensor in (descale_x, descale_wqa, descale_wqb, descale_wkva)
            ),
            "HiF8 descales must be FP32 ND",
        )
        _require_shape("descale_x", descale_x, (1,))
        _require_shape("descale_wqa", descale_wqa, (1, q_lora))
        _require_shape("descale_wqb", descale_wqb, (1, n * (dn + dr)))
        _require_shape("descale_wkva", descale_wkva, (1, kv_lora + dr))

    effective_cache_index = cache_index[cache_index >= 0]
    if effective_cache_index.numel():
        _require(
            torch.unique(effective_cache_index).numel()
            == effective_cache_index.numel(),
            "nonnegative cache_index values must be unique within one call",
        )
        _require(
            int(effective_cache_index.max()) < kv_cache.shape[0] * kv_cache.shape[1],
            "cache_index is out of cache range",
        )

    if t == 0:
        if qkv_quant_mode in (0, 1):
            q = torch.empty((0, n, KV_LORA + dr), dtype=torch.bfloat16)
        elif qkv_quant_mode == 2:
            q = _e4m3fn_from_raw(torch.empty((0, n, KV_LORA + dr), dtype=torch.uint8))
        else:
            q = torch.empty((0, n, KV_LORA + dr), dtype=torch.uint8)
        descale_q = torch.empty(
            (0, n, 1) if qkv_quant_mode == 2 else (0,), dtype=torch.float32
        )
        qr = _empty_quantized(w_qr_quant_mode, scale=False)
        descale_qr = _empty_quantized(w_qr_quant_mode, scale=True)
        if return_qr:
            qr = qr.reshape(0, q_lora)
            descale_qr = (
                descale_qr.reshape(0, q_lora // 32)
                if w_qr_quant_mode == 1
                else torch.empty((0,), dtype=torch.float32)
            )
        return q, descale_q, qr, descale_qr

    if w_qr_quant_mode == 1:
        qa = _mxfp8_matmul(x, wqa, descale_x, descale_wqa, weight_format)
        kva_proj = _mxfp8_matmul(x, wkva, descale_x, descale_wkva, weight_format)
    else:
        qa = _hif8_matmul_dequant(x, wqa, descale_x, descale_wqa, weight_format)
        kva_proj = _hif8_matmul_dequant(x, wkva, descale_x, descale_wkva, weight_format)

    qa_scaled = _apply_norm_scale(_rms_norm(qa, norm_weight_qa, norm_eps), qr_scale)
    if w_qr_quant_mode == 1:
        qr_all, descale_qr_all, _ = _mxfp8_dynamic_quant(qa_scaled)
        q_proj = _mxfp8_matmul(qr_all, wqb, descale_qr_all, descale_wqb, weight_format)
    else:
        qr_all, descale_qr_all, _ = _hif8_dynamic_quant(qa_scaled)
        q_proj = _hif8_matmul_dequant(
            qr_all, wqb, descale_qr_all, descale_wqb, weight_format
        )

    q_proj = q_proj.reshape(t, n, dn + dr)
    q_nope = q_proj[..., :dn].to(torch.bfloat16).to(torch.float32)
    q_nope = torch.einsum("tnd,ndk->tnk", q_nope, wkb.to(torch.float32))
    q_nope = q_nope.to(torch.bfloat16).to(torch.float32)
    q_rope = _apply_rope(q_proj[..., dn:], rope_sin, rope_cos, rope_mode)
    q_fp = torch.cat((q_nope, q_rope), dim=-1)

    kv_raw = kva_proj[..., :kv_lora]
    kr_raw = kva_proj[..., kv_lora:]
    kv_fp = _apply_norm_scale(_rms_norm(kv_raw, norm_weight_kva, norm_eps), kvr_scale)
    kr_fp = _apply_rope(kr_raw, rope_sin, rope_cos, rope_mode, cache_sat=True)

    if qkv_quant_mode in (0, 1):
        q = q_fp.to(torch.bfloat16)
        descale_q = torch.empty((0,), dtype=torch.float32)
    elif qkv_quant_mode == 2:
        q, descale_q = _e4_per_row_quant(q_fp)
    else:
        q, descale_q, _ = _hif8_dynamic_quant(q_fp)

    if qkv_quant_mode == 0:
        cache_rows = _bf16_sat_input(torch.cat((kv_fp, kr_fp), dim=-1)).to(
            torch.bfloat16
        )
        _store_cache_rows(kv_cache, cache_index, cache_rows, cache_mode)
    elif qkv_quant_mode == 1:
        kv_quant, kv_descale = _e4_per_cache_group_quant(kv_fp)
        kv_bytes = _raw_u8(kv_quant).reshape(t, kv_lora)
        kr_bytes = (
            kr_fp.to(torch.bfloat16).contiguous().view(torch.uint8).reshape(t, 2 * dr)
        )
        scale_bytes = kv_descale.contiguous().view(torch.uint8).reshape(t, -1)
        payload = torch.cat((kv_bytes, kr_bytes, scale_bytes), dim=-1)
        _require(payload.shape == (t, dtile), "internal mode-1 payload shape mismatch")
        _store_cache_bytes(kv_cache, cache_index, payload, cache_mode)
    elif qkv_quant_mode == 2:
        scale = quant_scale_kv.to(torch.float32)
        cache_rows = _e4m3fn_quantize(torch.cat((kv_fp, kr_fp), dim=-1) * scale)
        if kv_cache.dtype == torch.uint8:
            _store_cache_bytes(kv_cache, cache_index, _raw_u8(cache_rows), cache_mode)
        else:
            _store_cache_rows(kv_cache, cache_index, cache_rows, cache_mode)
    else:
        scale = quant_scale_kv.to(torch.float32)
        cache_rows = _hif8_quantize(torch.cat((kv_fp, kr_fp), dim=-1) * scale)
        _store_cache_bytes(kv_cache, cache_index, cache_rows, cache_mode)

    if return_qr:
        qr, descale_qr = qr_all, descale_qr_all
    else:
        qr = _empty_quantized(w_qr_quant_mode, scale=False)
        descale_qr = _empty_quantized(w_qr_quant_mode, scale=True)
    return q, descale_q, qr, descale_qr


# Comparator adapted from ops-transformer v9.1.1:
# https://github.com/hicann/ops-transformer/blob/v9.1.1/attention/flash_attn/tests/pytests/utils/compare.py
# Source-file license: CANN Open Software License Agreement Version 2.0.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Upstream license: https://github.com/hicann/ops-transformer/blob/v9.1.1/LICENSE
# The source-file notice is retained in this file header; repository-wide
# third-party license detections do not describe this comparator file.
# Local adaptations enable all three special-value switches,
# suppress logging, and expose metrics through the test adapter.
class ConfigFmk:
    golden_inf_switch = True
    inf_clip_switch = True
    nan_toZero_switch = True


def print_log(*args):
    """Logging only; the adapter serializes the returned metrics."""


class Result:
    def __init__(
        self,
        result_name,
        total_big_num,
        total_big_ratio,
        diff_big_max,
        diff_big_avg,
        diff_big_sum,
        err_w1_num,
        err_w1_ratio,
        err_k1_num,
        err_k1_ratio,
        err_k5_num,
        err_k5_ratio,
        err_h1_num,
        err_h1_ratio,
        total_small_num,
        total_small_ratio,
        err_small_num,
        err_small_ratio,
        diff_rmse,
        rst_eb,
        diff_eb,
        diff_big_ratio_max,
        diff_big_ratio_avg,
        diff_big_ratio_rmse,
        bm_cmp_std,
        num_total_nan=0,
        err_total_nan=0,
        num_total_inf=0,
        err_total_inf=0,
        num_total_ninf=0,
        err_total_ninf=0,
    ):
        self.result_name = result_name
        self.total_big_num = total_big_num
        self.total_big_ratio = total_big_ratio
        self.diff_big_max = diff_big_max
        self.diff_big_avg = diff_big_avg
        self.diff_big_sum = diff_big_sum
        self.err_w1_num = err_w1_num
        self.err_w1_ratio = err_w1_ratio
        self.err_k1_num = err_k1_num
        self.err_k1_ratio = err_k1_ratio
        self.err_k5_num = err_k5_num
        self.err_k5_ratio = err_k5_ratio
        self.err_h1_num = err_h1_num
        self.err_h1_ratio = err_h1_ratio
        self.total_small_num = total_small_num
        self.total_small_ratio = total_small_ratio
        self.err_small_num = err_small_num
        self.err_small_ratio = err_small_ratio
        self.diff_rmse = diff_rmse
        self.rst_eb = rst_eb
        self.diff_eb = diff_eb
        self.bm_cmp_std = bm_cmp_std
        self.num_total_nan = num_total_nan
        self.err_total_nan = err_total_nan
        self.num_total_inf = num_total_inf
        self.err_total_inf = err_total_inf
        self.num_total_ninf = num_total_ninf
        self.err_total_ninf = err_total_ninf
        self.diff_big_ratio_max = diff_big_ratio_max
        self.diff_big_ratio_avg = diff_big_ratio_avg
        self.diff_big_ratio_rmse = diff_big_ratio_rmse

    def print_result(self):
        print_log(f"正在打印结果：{self.result_name}")
        print_log(f" 大值总数：{self.total_big_num}")
        print_log(f" 大值占比：{self.total_big_ratio:.2%}")
        print_log(f" 大值最大绝对误差：{self.diff_big_max:.8f}")
        print_log(f" 大值平均绝对误差：{self.diff_big_avg:.8f}")
        print_log(f" 大值绝对误差总和：{self.diff_big_sum:.2f}")
        print_log(f" 大值最大相对误差：{self.diff_big_ratio_max:.8f}")
        print_log(f" 大值平均相对误差：{self.diff_big_ratio_avg:.8f}")
        print_log(f" 大值相对误差均方根（RMSE）：{self.diff_big_ratio_rmse:.8f}")
        print_log(f" 大值万分之1误差数：{self.err_w1_num}，占比{self.err_w1_ratio:.2%}")
        print_log(f" 大值千分之1误差数：{self.err_k1_num}，占比{self.err_k1_ratio:.2%}")
        print_log(f" 大值千分之5误差数：{self.err_k5_num}，占比{self.err_k5_ratio:.2%}")
        print_log(f" 大值百分之1误差数：{self.err_h1_num}，占比{self.err_h1_ratio:.2%}")
        print_log(f" 小值总数：{self.total_small_num}")
        print_log(f" 小值占比：{self.total_small_ratio:.2%}")
        print_log(f" 小值错误数：{self.err_small_num}，占比{self.err_small_ratio:.2%}")
        print_log(f" 误差均方根（RMSE）：{self.diff_rmse:.8f}")
        print_log(f" 均衡性偏差计数：{self.rst_eb}")
        print_log(f" 均衡性diff总和：{self.diff_eb:.8f}")
        if (
            self.num_total_nan + self.num_total_inf + self.num_total_ninf != 0
            or self.err_total_nan + self.err_total_inf + self.err_total_ninf != 0
            or True
        ):
            print_log(f" golden nan总数：{self.num_total_nan}")
            print_log(f" nan误差数：{self.err_total_nan}")
            print_log(f" golden inf总数：{self.num_total_inf}")
            print_log(f" inf误差数：{self.err_total_inf}")
            print_log(f" golden -inf总数：{self.num_total_ninf}")
            print_log(f" -inf误差数：{self.err_total_ninf}")

    def check_result_debug(self, benchmark, new=False, output_dtype="fp16"):
        if new:
            lo_bound = 0.0
            if output_dtype == "fp16":
                lo_bound = 2 ** (-11)
            if output_dtype == "bf16":
                lo_bound = 2 ** (-8)
            benchmark.diff_big_ratio_max = benchmark.diff_big_ratio_max.maximum(
                torch.tensor(lo_bound)
            )
            benchmark.diff_big_ratio_avg = benchmark.diff_big_ratio_avg.maximum(
                torch.tensor(lo_bound)
            )
            benchmark.diff_rmse = benchmark.diff_rmse.maximum(torch.tensor(lo_bound))
        reason_str = ""
        if (
            self.diff_big_ratio_max
            > benchmark.diff_big_ratio_max * self.bm_cmp_std["max_re_rtol"]
        ):
            reason_str += " diff_big_ratio_max error/"
        elif self.diff_big_ratio_max > benchmark.diff_big_ratio_max:
            reason_str += " diff_big_ratio_max warning/"
        if (
            self.diff_big_ratio_avg
            > benchmark.diff_big_ratio_avg * self.bm_cmp_std["avg_re_rtol"]
        ):
            reason_str += " diff_big_ratio_avg error/"
        elif self.diff_big_ratio_avg > benchmark.diff_big_ratio_avg:
            reason_str += " diff_big_ratio_avg warning/"
        if not new:
            if (
                self.diff_big_sum
                > benchmark.diff_big_sum * self.bm_cmp_std["avg_re_rtol"]
            ):
                reason_str += " diff_big_sum error/"
            elif self.diff_big_sum > benchmark.diff_big_sum:
                reason_str += " diff_big_sum warning/"
            if (
                self.diff_big_ratio_rmse
                > benchmark.diff_big_ratio_rmse * self.bm_cmp_std["rmse_rtol"]
            ):
                reason_str += " diff_big_ratio_rmse error/"
            elif self.diff_big_ratio_rmse > benchmark.diff_big_ratio_rmse:
                reason_str += " diff_big_ratio_rmse warning/"
        if (
            self.err_small_num
            > benchmark.err_small_num * self.bm_cmp_std["avg_re_rtol"]
        ):
            reason_str += " err_small_num error/"
        elif self.err_small_num > benchmark.err_small_num:
            reason_str += " err_small_num warning/"
        if self.diff_rmse > benchmark.diff_rmse * self.bm_cmp_std["rmse_rtol"]:
            reason_str += " diff_rmse error/"
        elif self.diff_rmse > benchmark.diff_rmse:
            reason_str += " diff_rmse warning/"
        if self.err_total_nan > benchmark.err_total_nan:
            reason_str += " err_total_nan error/"
        elif self.err_total_nan > 0:
            reason_str += " err_total_nan warning/"
        if (
            self.err_total_inf > benchmark.err_total_inf
            or self.err_total_ninf > benchmark.err_total_ninf
        ):
            reason_str += " err_total_inf error/"
        elif self.err_total_inf > 0 or self.err_total_ninf > 0:
            reason_str += " err_total_inf warning"
        return reason_str

    def check_result(self, benchmark, new=False, output_dtype="fp16"):
        print(f"comparing result: {self.result_name} VS {benchmark.result_name}")
        reason_str = self.check_result_debug(benchmark, new, output_dtype)
        if "error" in reason_str:
            print(self.result_name + " compare result: error")
            return ("Failed", reason_str)
        elif "warning" in reason_str:
            print(self.result_name + " compare result: warning")
            return ("warning", reason_str)
        else:
            print(self.result_name + " compare result: ok")
        return ("Pass", "")


def checkResultNew(params, value, golden, name, output_dtype):
    cfg_fk = ConfigFmk()
    debug_switch = False
    if output_dtype not in ["bf16", "fp16", "fp32", "hf32"]:
        output_dtype = "fp16"
    red = params["red_range"][output_dtype]
    red_list_str = red.split("/")
    red_list = [float(i) for i in red_list_str]
    bm_cmp_std = params["bm_cmp_std"][output_dtype]
    print(f"info：开始计算 {name} 精度。")
    if value.shape == golden.shape:
        if cfg_fk.golden_inf_switch:
            golden[golden > torch.finfo(value.dtype).max] = torch.inf
            golden[golden < torch.finfo(value.dtype).min] = -torch.inf
        if cfg_fk.inf_clip_switch:
            golden[torch.isinf(golden)] = torch.finfo(value.dtype).max
            value[torch.isinf(value)] = torch.finfo(value.dtype).max
        if cfg_fk.nan_toZero_switch:
            golden[torch.isnan(golden)] = 0
            value[torch.isnan(value)] = 0
        mask_golden_is_nan = torch.isnan(golden)
        mask_value_is_nan = torch.isnan(value)
        num_total_nan = torch.sum(mask_golden_is_nan)
        err_total_nan = torch.sum(mask_golden_is_nan.logical_xor(mask_value_is_nan))
        mask_golden_is_inf = torch.isinf(golden) & (golden > 0)
        mask_value_is_inf = torch.isinf(value) & (value > 0)
        num_total_inf = torch.sum(mask_golden_is_inf)
        err_total_inf = torch.sum(mask_golden_is_inf.logical_xor(mask_value_is_inf))
        mask_golden_is_ninf = torch.isinf(golden) & (golden < 0)
        mask_value_is_ninf = torch.isinf(value) & (value < 0)
        num_total_ninf = torch.sum(mask_golden_is_ninf)
        err_total_ninf = torch.sum(mask_golden_is_ninf.logical_xor(mask_value_is_ninf))
        if debug_switch:
            print(f" inf/nan总数：{num_total_nan + num_total_inf + num_total_ninf}")
            print(f" inf/nan误差数：{err_total_nan + err_total_inf + err_total_ninf}")
        golden[torch.isinf(golden)] = 1
        value[torch.isinf(value)] = 1
        golden[torch.isnan(golden)] = 1
        value[torch.isnan(value)] = 1
        total_big_num = torch.sum(golden.abs() >= bm_cmp_std["small_value"])
        total_big_ratio = total_big_num / golden.numel()
        value_big = value.clone()
        value_big[golden.abs() < bm_cmp_std["small_value"]] = 1
        golden_big = golden.clone()
        golden_big[golden.abs() < bm_cmp_std["small_value"]] = 1
        diff_big = torch.abs(value_big.sub(golden_big))
        diff_big_max = diff_big.max()
        diff_big_sum = diff_big.sum()
        diff_big_avg = diff_big_sum / total_big_num
        diff_big_rmse = torch.sqrt(torch.mean(torch.square(diff_big)))
        diff_big_ratio = diff_big / golden_big.abs()
        diff_big_ratio_max = diff_big_ratio.max()
        diff_big_ratio_avg = diff_big_ratio.sum() / total_big_num
        diff_big_ratio_rmse = torch.sqrt(torch.mean(torch.square(diff_big_ratio)))
        err_w1_num = torch.sum(diff_big_ratio > red_list[0])
        err_w1_ratio = err_w1_num / total_big_num
        err_k1_num = torch.sum(diff_big_ratio > red_list[1])
        err_k1_ratio = err_k1_num / total_big_num
        err_k5_num = torch.sum(diff_big_ratio > red_list[2])
        err_k5_ratio = err_k5_num / total_big_num
        err_h1_num = torch.sum(diff_big_ratio > red_list[3])
        err_h1_ratio = err_h1_num / total_big_num
        if debug_switch:
            print_log(f" 大值总数：{total_big_num}")
            print_log(f" 大值占比：{total_big_ratio:.2%}")
            print_log(f" 大值绝对误差均方根（RMSE）：{diff_big_rmse:.8f}")
            print_log(f" 大值平均绝对误差：{diff_big_avg:.8f}")
            print_log(f" 大值绝对误差总和：{diff_big_sum:.2f}")
            print_log(f" 大值最大相对误差：{diff_big_ratio_max:.8f}")
            print_log(f" 大值平均相对误差：{diff_big_ratio_avg:.8f}")
            print_log(f" 大值相对误差均方根（RMSE）：{diff_big_ratio_rmse:.8f}")
            print_log(f" 大值{red_list[0]}误差数：{err_w1_num}，占比{err_w1_ratio:.2%}")
            print_log(f" 大值{red_list[1]}误差数：{err_k1_num}，占比{err_k1_ratio:.2%}")
            print_log(f" 大值{red_list[2]}误差数：{err_k5_num}，占比{err_k5_ratio:.2%}")
            print_log(f" 大值{red_list[3]}误差数：{err_h1_num}，占比{err_h1_ratio:.2%}")
        total_small_num = torch.sum(golden.abs() < bm_cmp_std["small_value"])
        total_small_ratio = total_small_num / golden.numel()
        value_small = value.clone()
        value_small[golden.abs() > bm_cmp_std["small_value"]] = 1
        golden_small = golden.clone()
        golden_small[golden.abs() > bm_cmp_std["small_value"]] = 1
        diff_small = torch.abs(value_small.sub(golden_small))
        err_small_num = torch.sum(diff_small > bm_cmp_std["small_value_atol"])
        err_small_ratio = err_small_num / total_small_num
        if debug_switch:
            print_log(f" 小值总数：{total_small_num}")
            print_log(f" 小值占比：{total_small_ratio:.2%}")
            print_log(f" 小值错误数：{err_small_num}，占比{err_small_ratio:.2%}")
        diff = torch.abs(value.sub(golden))
        diff_rmse = torch.sqrt(torch.mean(torch.square(diff)))
        diff_rmse = diff_big_rmse
        if debug_switch:
            print_log(f" 绝对误差均方根（RMSE）：{diff_rmse:.8f}")
        eb_bigger = torch.sum(value > golden)
        eb_smaller = torch.sum(value < golden)
        rst_eb = torch.abs(eb_bigger.sub(eb_smaller))
        diff_eb = torch.sum(value.sub(golden))
        if debug_switch:
            print_log(f" 均衡性偏差计数：{rst_eb}")
            print_log(f" 均衡性diff总和：{diff_eb:.8f}")
        return Result(
            name,
            total_big_num,
            total_big_ratio,
            diff_big_max,
            diff_big_avg,
            diff_big_sum,
            err_w1_num,
            err_w1_ratio,
            err_k1_num,
            err_k1_ratio,
            err_k5_num,
            err_k5_ratio,
            err_h1_num,
            err_h1_ratio,
            total_small_num,
            total_small_ratio,
            err_small_num,
            err_small_ratio,
            diff_rmse,
            rst_eb,
            diff_eb,
            diff_big_ratio_max,
            diff_big_ratio_avg,
            diff_big_ratio_rmse,
            bm_cmp_std,
            num_total_nan,
            err_total_nan,
            num_total_inf,
            err_total_inf,
            num_total_ninf,
            err_total_ninf,
        )
    else:
        print_log(f"error: {name}计算结果错误，shape与标杆不匹配，用例执行失败！！！")
        print_log(f"debug: value {value.shape}")
        print_log(f"debug: golden {golden.shape}")
    return


POLICY_ID = "cpu_three_way_user_source_all_switches_true_20261026"


DEFAULT_CONFIG = dict(
    small_value=2.0**-10,
    small_value_atol=2.0**-16,
    max_re_rtol=10.0,
    avg_re_rtol=2.0,
    rmse_rtol=2.0,
)


ACCEPTANCE_POLICY_ID = "20261030_q_cache_whole_or_split_two_or_three"


_FIELDS = dict(
    max_re="diff_big_ratio_max",
    avg_re="diff_big_ratio_avg",
    small_bad="err_small_num",
    rmse="diff_rmse",
)


def _params(config):
    # red_range only produces descriptive counts; Result(new=True) never gates
    # on them. These four labels are the ones printed by the provided Result.
    dtypes = ("fp16", "bf16", "fp32", "hf32")
    return dict(
        red_range={d: "0.0001/0.001/0.005/0.01" for d in dtypes},
        bm_cmp_std={d: dict(config) for d in dtypes},
    )


def _metrics(result):
    values = {k: getattr(result, v).item() for k, v in _FIELDS.items()}
    values.update(
        big_elements=int(result.total_big_num),
        small_elements=int(result.total_small_num),
    )
    values["elements"] = values["big_elements"] + values["small_elements"]
    for k in ("err_total_nan", "err_total_inf", "err_total_ninf"):
        values[k] = int(getattr(result, k))
    # Retain undefined source statistics, rather than silently changing NaN to 0.
    values["undefined_metrics"] = [
        k
        for k, v in values.items()
        if isinstance(v, float) and not (-float("inf") < v < float("inf"))
    ]
    return values


def error_metrics(value, golden, config):
    if value.shape != golden.shape:
        raise ValueError(f"shape mismatch: {value.shape} != {golden.shape}")
    v, g = value.detach().flatten().clone(), golden.detach().flatten().clone()
    with contextlib.redirect_stdout(io.StringIO()):
        result = checkResultNew(_params(config), v, g, "metrics", "fp32")
    metrics = _metrics(result)
    metrics.update(
        full_rmse_diagnostic=float((v - g).abs().square().mean().sqrt()),
        max_abs_diagnostic=float((v - g).abs().max()),
    )
    return metrics


def compare_three_way(npu, cpu, cpu_gold, output_dtype, config=None):
    """Source-compatible decisions for this adapter's supplied comparison segment.

    MX E4M3 values decode to FP32 per get_pt_dtype. BF16 tensors retain BF16.
    The supplied cpu_gold keeps its dtype; do not force FP64 arithmetic.
    """
    cfg = dict(DEFAULT_CONFIG if config is None else config)
    if set(cfg) != set(DEFAULT_CONFIG) or not all(
        isinstance(v, (int, float)) and 0 < v < float("inf") for v in cfg.values()
    ):
        raise ValueError("all five comparison parameters must be finite and positive")
    if output_dtype not in ("bf16", "fp16", "fp32", "hf32", "float8_e4m3fn", "e8m0"):
        raise ValueError("explicit canonical comparison dtype required")
    if npu.shape != cpu.shape or npu.shape != cpu_gold.shape:
        raise ValueError("all three shapes must match")
    switches = {
        k: getattr(ConfigFmk, k)
        for k in ("golden_inf_switch", "inf_clip_switch", "nan_toZero_switch")
    }
    if npu.numel() == 0:
        return dict(
            status="Pass",
            npu=dict(elements=0),
            cpu=dict(elements=0),
            cpu_effective={},
            limits={},
            errors=[],
            warnings=[],
            output_dtype=output_dtype,
            benchmark_floor=0.0,
            config=cfg,
            switches=switches,
            empty=True,
            reason="",
            numerically_compared=False,
        )
    v, bvalue = npu.detach().flatten().clone(), cpu.detach().flatten().clone()
    if output_dtype == "float8_e4m3fn":
        v, bvalue = v.float(), bvalue.float()
    elif output_dtype == "bf16":
        v, bvalue = v.float().bfloat16(), bvalue.float().bfloat16()
    g = cpu_gold.detach().flatten().clone()
    with contextlib.redirect_stdout(io.StringIO()):
        dr = checkResultNew(_params(cfg), v, g, "NPU", output_dtype)
        # Reuse the modified Golden, as torch.from_numpy(cpu_golden) does in
        # the source wrapper. No CPU/NPU input storage is changed externally.
        br = checkResultNew(_params(cfg), bvalue, g, "CPU", output_dtype)
        d, b = _metrics(dr), _metrics(br)
        status, reason = dr.check_result(br, True, output_dtype)
    floor = {"bf16": 2.0**-8, "fp16": 2.0**-11}.get(output_dtype, 0.0)
    effective = _metrics(br)
    multipliers = dict(
        max_re="max_re_rtol",
        avg_re="avg_re_rtol",
        small_bad="avg_re_rtol",
        rmse="rmse_rtol",
    )
    limits = {
        k: (getattr(br, _FIELDS[k]) * cfg[m]).item() for k, m in multipliers.items()
    }
    names = {v: k for k, v in _FIELDS.items()}
    errors, warnings = [], []
    for entry in reason.split("/"):
        if entry.strip():
            metric, severity = entry.split()
            (errors if severity == "error" else warnings).append(
                names.get(metric, metric)
            )
    return dict(
        status=status,
        npu=d,
        cpu=b,
        cpu_effective=effective,
        limits=limits,
        errors=errors,
        warnings=warnings,
        output_dtype=output_dtype,
        benchmark_floor=floor,
        config=cfg,
        switches=switches,
        reason=reason,
        input_dtypes=dict(npu=str(v.dtype), cpu=str(bvalue.dtype), golden=str(g.dtype)),
        empty=False,
        numerically_compared=True,
    )


def evaluate_segments(parts, config=None):
    records = []
    for p in parts:
        r = compare_three_way(p["npu"], p["cpu"], p["cpu_gold"], p["dtype"], config)
        records.append(
            dict(name=p["name"], reference=p["reference"], gating=p["gating"], **r)
        )
    return dict(
        policy_id=POLICY_ID,
        eligible_for_gate=False,
        segments=records,
        failed=[r["name"] for r in records if r["gating"] and r["status"] == "Failed"],
        warnings=[
            r["name"] for r in records if r["gating"] and r["status"] == "warning"
        ],
    )


def output_acceptance(name, whole_two, whole_three, branches, *, homogeneous=True):
    """Accept a whole output first, then require every branch to have a route."""

    def choose(two, three):
        if any(record.get("passed") is True for record in two):
            return {"passed": True, "route": "two_way", "warning": False}
        accepted = [
            record
            for record in three
            if record.get("status") in ("Pass", "warning")
            and record.get("acceptance_definitive", True)
        ]
        if accepted:
            warning = all(record["status"] == "warning" for record in accepted)
            return {"passed": True, "route": "three_way", "warning": warning}
        return {"passed": False, "route": "none", "warning": False}

    whole = choose(whole_two, whole_three) if homogeneous else None
    branch_results = []
    if whole and whole["passed"]:
        passed, warning, route = True, whole["warning"], "whole_" + whole["route"]
    else:
        branch_results = [
            dict(name=branch["name"], **choose(branch["two_way"], branch["three_way"]))
            for branch in branches
        ]
        if not branch_results:
            raise ValueError("output acceptance requires explicit branch evidence")
        passed = all(record["passed"] for record in branch_results)
        warning = passed and any(record["warning"] for record in branch_results)
        route = "split" if homogeneous else "separate_dtypes"
    return dict(
        name=name + " acceptance",
        method=ACCEPTANCE_POLICY_ID,
        gating=True,
        passed=passed,
        status=("ACCEPTED_WITH_WARNING" if warning else "ACCEPTED")
        if passed
        else "FAILED",
        route=route,
        whole=whole,
        branches=branch_results,
    )


def apply_output_acceptance(metrics, three_way=None):
    records = [dict(record) for record in metrics]
    two = {record["name"]: record for record in records}
    segments = [] if three_way is None else three_way["segments"]
    three = {record["name"]: record for record in segments}
    if len(two) != len(records) or len(three) != len(segments):
        raise ValueError("comparison component names must be unique")
    for name in ("Q", "Cache"):
        mixed_cache = name == "Cache" and "Cache KV" in two
        pairs = (
            (("Cache KV", "Cache NoPE"), ("Cache KR", "Cache RoPE"))
            if mixed_cache
            else (
                (name + " NoPE/KV", name + " NoPE"),
                (name + " RoPE/KR", name + " RoPE"),
            )
        )
        branches = []
        for two_name, three_name in pairs:
            if two_name not in two:
                raise ValueError("missing branch comparison: " + two_name)
            branches.append(
                dict(
                    name=two_name,
                    two_way=[two[two_name]],
                    three_way=[three[three_name]] if three_name in three else [],
                )
            )
            two[two_name]["gating"] = False
        if not mixed_cache:
            if name not in two:
                raise ValueError("missing whole-output comparison: " + name)
            two[name]["gating"] = False
        records.append(
            output_acceptance(
                name,
                [two[name]] if name in two else [],
                [three[name]] if name in three else [],
                branches,
                homogeneous=not mixed_cache,
            )
        )
    return records


def golden_with_observations(function, args, kwargs):
    """Read reference locals at return, without changing code, globals or outputs.

    HiF8 ULP needs the pre-cast reference in the quantizer's numeric domain.
    The frozen Golden already computes it; do not reconstruct it from payload.
    """
    observed = {}
    previous = sys.getprofile()

    def capture(frame, event, arg):
        if event == "return" and frame.f_code is function.__code__:
            for name in ("qa_scaled", "q_fp", "kv_fp", "kr_fp"):
                value = frame.f_locals.get(name)
                if isinstance(value, torch.Tensor):
                    observed[name] = value.detach().clone()
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(capture)
    try:
        outputs = function(*args, **kwargs)
    finally:
        sys.setprofile(previous)
    return outputs, observed


def _record(actual, expected, matched, name, method, allowed_failures, errors=None):
    actual, expected = actual.double(), expected.double()
    assert actual.shape == expected.shape, f"{name}: shape mismatch"
    total = actual.numel()
    failed = int((~matched).sum())
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    difference = (actual - expected).abs()
    finite_diff = difference[finite]
    norm_ref = expected[finite].norm().item()
    norm_actual = actual[finite].norm().item()
    nonzero = finite & (expected != 0)
    classification_bad = (
        (torch.isnan(actual) != torch.isnan(expected))
        | (torch.isposinf(actual) != torch.isposinf(expected))
        | (torch.isneginf(actual) != torch.isneginf(expected))
    )
    # A whole segment supplies the denominator, including nonfinite elements.
    # Mixed-tolerance hard relative limit follows the supplied method 14/17.
    relative = difference / (
        torch.maximum(
            torch.maximum(actual.abs(), expected.abs()),
            torch.tensor((1 / 16384) / 0.005, dtype=torch.float64),
        )
        + 2e-9
    )
    bad_finite = (~matched) & finite
    max_failed_relative = relative[bad_finite].max().item() if bad_finite.any() else 0.0
    hard_pass = (
        method in ("hif8_method6_sat", "fp8_method10") or max_failed_relative < 10.0
    )
    result = {
        "name": name,
        "method": method,
        "elements": total,
        "failed_elements": failed,
        "allowed_failures": allowed_failures,
        "matched_ratio": (total - failed) / total if total else 1.0,
        "finite_elements": int(finite.sum()),
        "classification_mismatches": int(classification_bad.sum()),
        "max_abs": finite_diff.max().item() if finite_diff.numel() else 0.0,
        "max_rel": (difference[nonzero] / expected[nonzero].abs()).max().item()
        if nonzero.any()
        else None,
        "relative_l2": finite_diff.norm().item() / norm_ref if norm_ref else None,
        "cosine": torch.nn.functional.cosine_similarity(
            actual[finite].reshape(1, -1), expected[finite].reshape(1, -1)
        ).item()
        if norm_ref and norm_actual
        else None,
        "max_failed_relative_error": max_failed_relative,
        "hard_cap_pass": hard_pass,
        "passed": failed <= allowed_failures and hard_pass,
        "empty_segment": total == 0,
    }
    bad_indices = (~matched).flatten().nonzero().flatten()[:8].tolist()
    result["first_failed_elements"] = [
        {
            "flat_index": i,
            "actual": actual.flatten()[i].item(),
            "expected": expected.flatten()[i].item(),
        }
        for i in bad_indices
    ]
    if errors is not None:
        finite_error = errors[torch.isfinite(errors)]
        result["max_abs_ulp"] = (
            finite_error.abs().max().item() if finite_error.numel() else 0.0
        )
        result["invalid_ulp_elements"] = int(torch.isnan(errors).sum())
    return result


def mixed(actual, expected, name, *, scale=False):
    actual, expected = actual.double(), expected.double()
    assert actual.shape == expected.shape, f"{name}: shape mismatch"
    atol, rtol = (0.000025, 0.005) if scale else (0.0001, 0.0078125)
    matched = torch.isclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)
    result = _record(
        actual,
        expected,
        matched,
        name,
        "fp32_scale" if scale else "bf16",
        actual.numel() // 200,
    )
    result.update(atol=atol, rtol=rtol, pct_thd=0.005)
    return result


def e8m0_int8(actual, expected, name="descale_qr E8M0"):
    """View E8M0 bytes as signed int8, then widen before method1 subtraction."""
    allowed = {torch.uint8, torch.int8, getattr(torch, "float8_e8m0fnu", None)}
    if actual.dtype not in allowed or expected.dtype not in allowed:
        raise ValueError(
            "E8M0 comparison requires native E8M0 or explicit byte carriers"
        )
    a = actual.contiguous().view(torch.int8).double()
    b = expected.contiguous().view(torch.int8).double()
    if a.shape != b.shape:
        raise ValueError("E8M0 byte shapes must match")
    matched = torch.isclose(a, b, atol=1.0, rtol=0.0, equal_nan=True)
    result = _record(a, b, matched, name, "e8m0_int8_method1", 0)
    result.update(
        gating=True,
        precision_method=1,
        atol=1.0,
        rtol=0.0,
        pct_thd=0.0,
        diff_thd=0.005,
        max_diff_thd=10.0,
        representation="signed_int8_raw_bytes",
        different_codes=int((a != b).sum()),
    )
    return result


def encoded(actual, expected, name):
    """User-specified FP8 method 10, also for known FP8 raw-byte carriers.

    Match the user's method10 decision, with the confirmed pct_thd=0.005.
    Preserve its percentage arithmetic and err_inx_2.any() behavior, including
    skipping the hard check when the only distance >2 occurs at index zero.
    Canonicalize FP8 NaNs to +0 on adapter copies; keep caller tensors intact.
    Signed zero pairs still count toward the >1 failure fraction.
    Empty slots remain un-compared, as required by the public-output contract.
    This is neither a floating relative-error nor a bit-equality test.
    """
    e5 = getattr(torch, "float8_e5m2", None)
    e4 = getattr(torch, "float8_e4m3fn", None)
    allowed = {torch.uint8, torch.int8, e4, e5}
    if actual.dtype not in allowed or expected.dtype not in allowed:
        raise ValueError("method10 requires FP8 or explicitly FP8 byte carriers")
    if actual.dtype in (e4, e5) and expected.dtype in (e4, e5):
        if actual.dtype != expected.dtype:
            raise ValueError("method10 FP8 formats must match")
    is_e5 = actual.dtype == e5 or expected.dtype == e5
    a = actual.contiguous().view(torch.uint8).to(torch.int16)
    b = expected.contiguous().view(torch.uint8).to(torch.int16)
    assert a.shape == b.shape, f"{name}: payload shape mismatch"
    # Raw-byte callers in compare_outputs are E4M3FN Q/QR/cache segments,
    # never arbitrary uint8 or HiF8/E8M0. E5M2 supports native typed calls.
    nan_a = (a & 0x7F) > 0x7C if is_e5 else (a & 0x7F) == 0x7F
    nan_b = (b & 0x7F) > 0x7C if is_e5 else (b & 0x7F) == 0x7F
    a, b = torch.where(nan_a, 0, a), torch.where(nan_b, 0, b)
    linear_distance = (a - b).abs()
    distance = torch.minimum(linear_distance, 256 - linear_distance)
    signed_zero_pair = ((a == 0) & (b == 128)) | ((a == 128) & (b == 0))
    hard_bad = (distance > 2) & ~signed_zero_pair
    result = _record(a, b, distance <= 1, name, "fp8_method10", a.numel() // 200)
    # Intentionally reproduce the supplied source, NOT a corrected size check.
    # Test all >2 indices before applying the signed-zero exception.
    over_two_indices = (distance > 2).flatten().nonzero().flatten()
    hard_check_entered = bool(over_two_indices.any())
    indices = hard_bad.flatten().nonzero().flatten()
    fulfill_percent = (
        (float(a.numel() - result["failed_elements"]) / float(a.numel()) * 100.0)
        if a.numel()
        else None
    )
    result.update(
        pct_thd=0.005,
        code_distance_threshold=1,
        hard_code_distance_threshold=2,
        hard_failed_elements=int(indices.numel()),
        hard_cap_pass=not hard_check_entered or indices.numel() == 0,
        source_hard_check_entered=hard_check_entered,
        source_index_zero_hard_check_skipped=(
            over_two_indices.numel() == 1 and not hard_check_entered
        ),
        fulfill_percent=fulfill_percent,
        required_fulfill_percent=(1 - 0.005) * 100.0,
        decision_policy="user_method10_source_compat_pct005",
        signed_zero_pairs=int(signed_zero_pair.sum()),
        nan_canonicalized_actual=int(nan_a.sum()),
        nan_canonicalized_reference=int(nan_b.sum()),
        max_circular_code_distance=int(distance.max()) if distance.numel() else 0,
        first_hard_failed_elements=[
            dict(
                flat_index=i,
                actual_code=int(a.flatten()[i]),
                reference_code=int(b.flatten()[i]),
                circular_distance=int(distance.flatten()[i]),
            )
            for i in indices[:8].tolist()
        ],
    )
    result["passed"] = (
        a.numel() == 0 or fulfill_percent >= (1 - 0.005) * 100.0
    ) and result["hard_cap_pass"]
    return result


def normalize_dynamic(value, scale):
    """Same zero-scale rule as the frozen mathematical reference."""
    return torch.where(
        scale == 0,
        torch.zeros_like(value),
        value / torch.where(scale == 0, torch.ones_like(scale), scale),
    )


def hif8(actual_decoded, reference_scaled, name, *, diff_thd=0.005):
    """HiF8 method 6: source size-dependent quota, with approved SAT retained."""
    if not 0 <= diff_thd <= 1:
        raise ValueError("method6 diff_thd must be a finite fraction in [0, 1]")
    # SAT is a reference Cast semantic, not projection/norm clamping.
    # Never turn finite >32768 into Inf; NaN->0 and Inf->signed finite max.
    x = (
        torch.nan_to_num(
            reference_scaled.float(), nan=0.0, posinf=32768.0, neginf=-32768.0
        )
        .clamp(-32768.0, 32768.0)
        .double()
    )
    h = actual_decoded.double()
    assert h.shape == x.shape, f"{name}: HiF8 shape mismatch"
    ex = torch.log2(x.abs() + 2.0**-1000).clamp(min=-22)
    exponent = ex.floor()
    width = torch.zeros_like(x)
    width = torch.where(exponent.abs() <= 15, 1.0, width)
    width = torch.where(exponent.abs() <= 7, 2.0, width)
    width = torch.where(exponent.abs() <= 3, 3.0, width)
    ulp = (h - x) * torch.pow(2.0, -exponent + width)
    signed_ex = ex * torch.where(x >= 0, 1.0, -1.0)
    signed_eh = torch.log2(h.abs() + 2.0**-1000) * torch.where(h >= 0, 1.0, -1.0)
    ulp = torch.where(width == 0, signed_eh - signed_ex, ulp)
    ulp = torch.where(
        (x == 0) | (x.abs() < 2.0**-23), torch.where(h == 0, 0.0, 1.5), ulp
    )
    # Invalid device values must not disappear in finite-only statistics.
    ulp = torch.where(~torch.isfinite(h), torch.full_like(ulp, float("nan")), ulp)
    matched = torch.isfinite(ulp) & (ulp.abs() <= 1.0)
    m = x.numel()
    # Source: m<=100 retains caller diff_thd; m>100 replaces it with
    # (m * flag + 1) / m, where the HiF8 branch sets flag=.001.
    # Exact count form of >= percentage gate avoids floating boundary drift.
    allowed = m // 1000 + 1 if m > 100 else int(m * diff_thd)
    result = _record(h, x, matched, name, "hif8_method6_sat", allowed, ulp)
    result.update(
        precision_method=6,
        ulp_threshold=1.0,
        flag=0.001,
        short_segment_pct_thd=diff_thd,
        effective_fail_fraction=(0.001 + 1 / m) if m > 100 else diff_thd,
        extra_allowed_element=m > 100,
        reference_domain="pre_cast_scaled_SAT",
    )
    return result


def _validate_q_dtype(actual, expected, quant_mode):
    """Check Q's semantic format before using one comparison domain.

    Unlike mode-1 Cache, Q has no mixed-dtype payload in the public contract.
    A uint8 carrier is homogeneous only for the explicitly declared FP8 format.
    """
    formats = {
        0: {torch.bfloat16},
        1: {torch.bfloat16},
        2: {torch.float8_e4m3fn, torch.uint8},
        3: {torch.uint8},
    }
    if quant_mode not in formats or any(
        value.dtype not in formats[quant_mode] for value in (actual, expected)
    ):
        raise ValueError("Q dtype does not match the homogeneous public output format")


def compare_outputs(
    actual, expected, actual_cache, reference_cache, args, attrs, observed
):
    """Compare every public component; never stop at the first numeric failure."""
    results = []
    q, sq, qr, sqr = actual
    eq, esq, eqr, esqr = expected
    qm, wm = attrs["qkv_quant_mode"], attrs["w_qr_quant_mode"]
    _validate_q_dtype(q, eq, qm)
    dr = q.shape[-1] - 512

    def add_whole(a, b, operation, prefix):
        results.append(operation(a, b, prefix))
        for suffix, sl in (
            ("NoPE/KV", slice(None, 512)),
            ("RoPE/KR", slice(512, None)),
        ):
            record = operation(a[..., sl], b[..., sl], prefix + " " + suffix)
            record.update(
                gating=False,
                diagnostic_reason="branch fallback after both whole-output routes fail",
            )
            results.append(record)

    if qm in (0, 1):
        add_whole(q, eq, mixed, "Q")
    elif qm == 2:
        add_whole(q, eq, encoded, "Q")
    else:
        scaled = normalize_dynamic(observed["q_fp"], esq)
        add_whole(_hif8_decode_raw(q), scaled, hif8, "Q")
    if qm in (2, 3):
        results.append(mixed(sq, esq, "descale_q", scale=True))
    if attrs["return_qr"]:
        if wm == 1:
            results.append(encoded(qr, eqr, "QR"))
        else:
            results.append(
                hif8(
                    _hif8_decode_raw(qr),
                    normalize_dynamic(observed["qa_scaled"], esqr),
                    "QR",
                )
            )
            results.append(mixed(sqr, esqr, "descale_qr", scale=True))

    if wm == 1:
        # Empty when return_qr=False; structural checks independently enforce that slot.
        results.append(e8m0_int8(sqr, esqr))

    capacity = actual_cache.shape[0] * actual_cache.shape[1]
    hits = args[12][args[12] >= 0]
    a = cache_to_logical(actual_cache, attrs["cache_mode"])
    b = cache_to_logical(reference_cache, attrs["cache_mode"])
    # FP8 CPU advanced indexing is not universally supported; index raw bytes.
    a = a.contiguous().view(torch.uint8).reshape(capacity, -1)[hits]
    b = b.contiguous().view(torch.uint8).reshape(capacity, -1)[hits]
    if qm == 0:
        add_whole(
            a.contiguous().view(torch.bfloat16),
            b.contiguous().view(torch.bfloat16),
            mixed,
            "Cache",
        )
    elif qm == 1:
        results.append(encoded(a[..., :512], b[..., :512], "Cache KV"))
        results.append(
            mixed(
                a[..., 512 : 512 + 2 * dr].contiguous().view(torch.bfloat16),
                b[..., 512 : 512 + 2 * dr].contiguous().view(torch.bfloat16),
                "Cache KR",
            )
        )
        results.append(
            mixed(
                a[..., 512 + 2 * dr :].contiguous().view(torch.float32),
                b[..., 512 + 2 * dr :].contiguous().view(torch.float32),
                "Cache descale",
                scale=True,
            )
        )
    elif qm == 2:
        add_whole(a, b, encoded, "Cache")
    else:
        # Row order follows input tokens, not sorted physical slots.
        values = torch.cat((observed["kv_fp"], observed["kr_fp"]), dim=-1)
        scaled = (values * args[13].float())[args[12] >= 0]
        add_whole(_hif8_decode_raw(a), scaled, hif8, "Cache")
    return results


def compare_for_acceptance(
    actual, expected, actual_cache, reference_cache, args, attrs, observed
):
    metrics = compare_outputs(
        actual, expected, actual_cache, reference_cache, args, attrs, observed
    )
    decisions = apply_output_acceptance(metrics)
    q_failed = any(m["name"] == "Q acceptance" and not m["passed"] for m in decisions)
    if q_failed:
        three_way = supplied_q_three_way(actual, args, attrs)
        if three_way is not None:
            decisions = apply_output_acceptance(metrics, three_way)
            decisions.append(
                dict(
                    name="Supplied high Golden diagnostics",
                    gating=False,
                    three_way=three_way,
                )
            )
    return decisions


def pack_weight(weight):
    outputs, reduction = weight.shape
    return (
        weight.reshape(outputs // 32, 32, reduction // 16, 16)
        .permute(0, 2, 3, 1)
        .contiguous()
    )


def pack_scale(scale):
    outputs, groups, pair = scale.shape
    return (
        scale.reshape(outputs // 16, 16, groups, pair).permute(0, 2, 1, 3).contiguous()
    )


def make_inputs(
    qm,
    wm,
    nz,
    cache_mode,
    dn,
    dr,
    rope_mode,
    *,
    tokens=3,
    ql=1024,
    dim=1024,
    heads=2,
    zero=False,
):
    torch.manual_seed(42)
    columns = (ql, heads * (dn + dr), 512 + dr)
    reductions = (dim, ql, dim)

    def encode(value):
        return _e4m3fn_quantize(value) if wm == 1 else _hif8_quantize(value)

    x = encode(torch.zeros(tokens, dim) if zero else torch.randn(tokens, dim) * 0.3)
    weights = [encode(torch.randn(o, k) * 0.1) for o, k in zip(columns, reductions)]
    if wm == 1:
        sx = _e8m0_from_raw(
            torch.randint(124, 128, (tokens, dim // 32), dtype=torch.uint8)
        )
        sw = [
            _e8m0_from_raw(torch.randint(125, 129, (o, k // 64, 2), dtype=torch.uint8))
            for o, k in zip(columns, reductions)
        ]
    else:
        sx = torch.tensor([0.5])
        sw = [torch.linspace(0.5, 1.5, o).reshape(1, o) for o in columns]
    dtype = (
        torch.bfloat16
        if qm == 0
        else torch.float8_e4m3fn
        if qm in (1, 2)
        else torch.uint8
    )
    width = 512 + 2 * dr + 16 if qm == 1 else 512 + dr
    cache = torch.full((2, 128, 1, width), 0x35, dtype=torch.uint8)
    if dtype == torch.bfloat16:
        cache = torch.full((2, 128, 1, width), 123.0, dtype=dtype)
    else:
        cache = cache.view(dtype)
    index = torch.arange(tokens, dtype=torch.int64)
    if tokens >= 3:
        index[0] = -1
        index[-1] = 255
    kwargs = dict(
        norm_eps=1e-5,
        w_qr_quant_mode=wm,
        qkv_quant_mode=qm,
        cache_mode=cache_mode,
        qr_scale=1.25,
        kvr_scale=0.75,
        rope_mode=rope_mode,
        return_qr=True,
    )
    sine = (torch.randn(tokens, dr) * 0.2).bfloat16() if dr and rope_mode else None
    cosine = (
        (torch.randn(tokens, dr) * 0.2 + 0.8).bfloat16() if dr and rope_mode else None
    )
    fixed = [
        torch.randn(heads, dn, 512).bfloat16() * 0.02,
        sx,
        *sw,
        torch.linspace(0.8, 1.2, ql),
        torch.linspace(0.7, 1.3, 512),
        cache,
        index,
        torch.tensor([2.0]) if qm in (2, 3) else None,
        sine,
        cosine,
    ]
    golden_weights = [pack_weight(w) for w in weights] if nz else weights
    if nz and wm == 1:
        fixed[2:5] = [pack_scale(s) for s in sw]
    cpu_args = [x, *golden_weights, *fixed]
    return cpu_args, weights, kwargs


def check_return_qr_contract(operator):
    inputs = [None] * 13
    attrs = dict(w_qr_quant_mode=1, qkv_quant_mode=0, cache_mode=0)
    with patch.object(
        torch.ops.cannbotdsl_quant_mla_prolog.quant_mla_prolog, "default"
    ) as dispatch:
        for value in (0, 1, -1, 0.0, 1.0, None, "true", [], torch.tensor(True)):
            with pytest.raises(TypeError, match="^return_qr must be bool$"):
                operator(*inputs, **attrs, return_qr=value)
        dispatch.assert_not_called()
        for value in (False, True):
            assert operator(*inputs, **attrs, return_qr=value) is dispatch.return_value
            assert dispatch.call_args.kwargs["return_qr"] is value
        operator(*inputs, **attrs)
        assert dispatch.call_args.kwargs["return_qr"] is False


@pytest.fixture(scope="module")
def npu_operator():
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    path = (
        Path(__file__).resolve().parents[2]
        / "samples/quant_mla_prolog/quant_mla_prolog.py"
    )
    spec = importlib.util.spec_from_file_location("quant_mla_prolog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check_return_qr_contract(module.quant_mla_prolog)
    return module, torch_npu


CASES = (
    pytest.param(
        0, 1, True, 0, 128, 64, 1, 3, 1024, 4, True, id="mxfp8_mode0_t3_rope_tail"
    ),
    pytest.param(
        1, 1, True, 0, 192, 64, 1, 16, 1536, 4, True, id="mxfp8_mode1_t16_head_handoff"
    ),
    pytest.param(
        2, 1, True, 1, 192, 64, 1, 33, 2048, 104, True, id="mxfp8_mode2_t33_batch_tail"
    ),
    pytest.param(3, 2, False, 1, 128, 64, 2, 3, 1024, 2, False, id="hif8_mode3_rope2"),
)


@pytest.mark.npu
@pytest.mark.parametrize(
    "qm,wm,nz,cache_mode,dn,dr,rope_mode,tokens,ql,heads,expect_handoff", CASES
)
def test_quant_mla_prolog(
    qm,
    wm,
    nz,
    cache_mode,
    dn,
    dr,
    rope_mode,
    tokens,
    ql,
    heads,
    expect_handoff,
    npu_operator,
):
    module, torch_npu = npu_operator
    operator = module.quant_mla_prolog
    cpu_args, weights, attrs = make_inputs(
        qm, wm, nz, cache_mode, dn, dr, rope_mode, tokens=tokens, ql=ql, heads=heads
    )
    device_weights = [
        torch_npu.npu_format_cast(weight.npu(), 29) if nz else weight.npu()
        for weight in weights
    ]
    device_args = [
        cpu_args[0].npu(),
        *device_weights,
        *(value.npu() if value is not None else None for value in cpu_args[4:]),
    ]
    cache = device_args[11]
    pointer = cache.data_ptr()
    storage_pointer = cache.untyped_storage().data_ptr()
    cache_format = torch_npu.get_npu_format(cache)
    initial = cpu_args[11].clone()
    expected, observed = golden_with_observations(
        quant_mla_prolog_golden, cpu_args, attrs
    )
    factory = operator.__globals__["_program"]
    selected_configs = []

    def checked_program(*config):
        selected_configs.append(config)
        assert (config[-1] != (0, 0)) is expect_handoff
        if qm == 2:
            assert config[12] >= 32 and tokens % 8 != 0
        return factory(*config)

    module.quant_mla_prolog_precheck(*device_args, **attrs)
    with patch.dict(operator.__globals__, {"_program": checked_program}):
        actual = operator(*device_args, **attrs)
    assert len(selected_configs) == 1
    torch.npu.synchronize()

    assert len(actual) == 4
    for got, wanted in zip(actual, expected):
        assert got.shape == wanted.shape
        assert got.dtype == wanted.dtype
        assert torch_npu.get_npu_format(got) == 2
    assert cache.data_ptr() == pointer
    assert cache.untyped_storage().data_ptr() == storage_pointer
    assert torch_npu.get_npu_format(cache) == cache_format
    actual = tuple(value.cpu() for value in actual)
    actual_cache = cache.cpu()
    metrics = compare_for_acceptance(
        actual, expected, actual_cache, cpu_args[11], cpu_args, attrs, observed
    )
    capacity = cache.shape[0] * cache.shape[1]
    hits = cpu_args[12][cpu_args[12] >= 0]
    untouched = torch.ones(capacity, dtype=torch.bool)
    untouched[hits] = False
    initial_rows = (
        cache_to_logical(initial, cache_mode).view(torch.uint8).reshape(capacity, -1)
    )
    actual_rows = (
        cache_to_logical(actual_cache, cache_mode)
        .view(torch.uint8)
        .reshape(capacity, -1)
    )
    assert torch.equal(initial_rows[untouched], actual_rows[untouched])
    LOGGER.info("precision metrics: %s", metrics)
    failed = [
        metric["name"]
        for metric in metrics
        if metric.get("gating", True) and metric.get("passed") is False
    ]
    assert not failed, f"precision segments failed: {', '.join(failed)}"


# Supplied Golden source SHA256: 9d9fda1bbf34dcdc32afeb3915f986e08d6b2e1f5af529963eb8d282482c23fb
# Mathematical functions retained; standalone diagnostic prints removed.
def _supplied_golden_functions():
    import copy
    import re
    import numpy as np
    from functools import lru_cache

    COLOR_YELLOW = YELLOW_RESET = COLOR_GREEN = GREEN_RESET = ""  # noqa: F841 -- unchanged supplied Golden, guarded MXFP8 path
    INT8_DTYPE_MAX = 127.0
    FP8_E4M3_DTYPE_MAX = 448.0
    HIF8_DTYPE_MAX = 32768.0
    WEIGHT_QUANT_MODE_FULL_INT8 = 2
    WEIGHT_QUANT_MODE_MXFP8_FULL = 3
    WEIGHT_QUANT_MODE_FULL_FP8_E4M3 = 4
    WEIGHT_QUANT_MODE_FULL_HIF8 = 5

    def _get_legacy_device_type():
        return "cpu"

    def _get_hif8_fraction_bits_number(exponent):
        if exponent < -22:
            return (-1, 3, 0)
        if -22 <= exponent < -15:
            return (0, 3, 0)
        if exponent == 0:
            return (1, 0, 3)
        if abs(exponent) == 1:
            return (2, 1, 3)
        if 2 <= abs(exponent) <= 3:
            return (4, 2, 3)
        if 4 <= abs(exponent) <= 7:
            return (8, 3, 2)
        if 8 <= abs(exponent) <= 15:
            return (12, 4, 1)
        if exponent > 15:
            return (12, 4, -1)
        raise ValueError(f"unsupported exponent for hif8: {exponent}")

    def _fp32_ta_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
        if exponent == -23:
            return (True, 0)
        hif8_value_tmp = fraction32_int >> 23 - (hif8_bits_num + 1)
        if hif8_value_tmp == pow(2, hif8_bits_num + 1) - 1:
            return (True, 0)
        if hif8_value_tmp == 0:
            return (False, 0)
        if hif8_value_tmp % 2 == 1:
            hif8_value_tmp += 1
            return (False, hif8_value_tmp >> 1)
        return (False, hif8_value_tmp >> 1)

    def _fp32_ssr_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
        t14_mask = 16383
        if exponent == -23:
            f14_values = (fraction32_int >> 10) + 8192
            t14_values = fraction32_int & t14_mask
            hif8_value = 0
        else:
            hif8_value = fraction32_int >> 23 - hif8_bits_num
            f14_t14 = fraction32_int - (hif8_value << 23 - hif8_bits_num)
            f14_values = f14_t14 >> 23 - hif8_bits_num - 14
            t14_values = f14_t14 & t14_mask
        if f14_values >= t14_values:
            if hif8_value == pow(2, hif8_bits_num) - 1:
                return (True, 0)
            return (False, hif8_value + 1)
        return (False, hif8_value)

    def _fp16_ta_round_to_hif8(fraction16_int, hif8_bits_num, exponent):
        if exponent == -23:
            return (True, 0)
        hif8_value_tmp = fraction16_int >> 10 - (hif8_bits_num + 1)
        if hif8_value_tmp == pow(2, hif8_bits_num + 1) - 1:
            return (True, 0)
        if hif8_value_tmp == 0:
            return (False, 0)
        if hif8_value_tmp % 2 == 1:
            hif8_value_tmp += 1
            return (False, hif8_value_tmp >> 1)
        return (False, hif8_value_tmp >> 1)

    def _fp16_ssr_round_to_hif8(fraction16_int, hif8_bits_num, exponent):
        t2_mask = 1
        t2_values = (fraction16_int & t2_mask) * 2 + 1
        if exponent == -23:
            f2_values = 2 + (fraction16_int >> 9)
            hif8_value = 0
        else:
            hif8_value = fraction16_int >> 10 - hif8_bits_num
            f2_t2 = fraction16_int - (hif8_value << 10 - hif8_bits_num)
            f2_values = f2_t2 >> 10 - hif8_bits_num - 2
        if f2_values >= t2_values:
            if hif8_value == pow(2, hif8_bits_num):
                return (True, 0)
            return (False, hif8_value + 1)
        return (False, hif8_value)

    def cvt_hifuint8_to_float(x, over_mode=True):
        x = int(x)
        if x == 0:
            return float(0)
        if x == 128:
            return np.nan if over_mode else float(0)
        if x == 239:
            return -np.inf if over_mode else -32768.0
        if x == 111:
            return np.inf if over_mode else 32768.0
        sign = -1.0 if x >= 128 else 1.0
        dot_4_value = (x & 120) >> 3
        if dot_4_value >= 12:
            exponent_int = (x & 30) >> 1
            exponent_value = -exponent_int if exponent_int >= 8 else exponent_int + 8
            m_value = 1.0 + (x & 1) * 0.5
        elif dot_4_value >= 8:
            exponent_int = (x & 28) >> 2
            exponent_value = -exponent_int if exponent_int >= 4 else exponent_int + 4
            m_value = 1.0 + (x & 3) * 0.25
        elif dot_4_value >= 4:
            exponent_int = (x & 24) >> 3
            exponent_value = -exponent_int if exponent_int >= 2 else exponent_int + 2
            m_value = 1.0 + (x & 7) * 0.125
        elif dot_4_value >= 2:
            exponent_value = -1 if (x & 8) >> 3 >= 1 else 1
            m_value = 1.0 + (x & 7) * 0.125
        elif dot_4_value == 1:
            exponent_value = 0
            m_value = 1.0 + (x & 7) * 0.125
        elif dot_4_value == 0:
            exponent_value = (x & 7) - 23
            m_value = 1.0
        else:
            return float(0)
        return sign * pow(2.0, exponent_value) * m_value

    @lru_cache(maxsize=2)
    def _decode_table(over_mode):
        return np.array(
            [cvt_hifuint8_to_float(i, over_mode=over_mode) for i in range(256)],
            dtype=np.float32,
        )

    @lru_cache(maxsize=1)
    def _positive_round_values_and_codes():
        decode_table = _decode_table(True)
        positive_codes = []
        positive_values = []
        for code in range(128):
            value = decode_table[code]
            if np.isfinite(value) and value >= 0:
                positive_codes.append(code)
                positive_values.append(float(value))
        positive_codes.append(int(HIF8_POS_INF_CODE))  # noqa: F821 -- unchanged supplied Golden, guarded MXFP8 path
        positive_values.append(float(_HIF8_POS_INF_ROUND_VALUE))  # noqa: F821 -- unchanged supplied Golden, guarded MXFP8 path
        order = np.argsort(np.asarray(positive_values, dtype=np.float64))
        values = np.asarray(positive_values, dtype=np.float32)[order]
        codes = np.asarray(positive_codes, dtype=np.uint8)[order]
        return (values, codes)

    def decode_hif8_uint8_tensor_to_float32(in_tensor, over_mode=True):
        if torch.is_tensor(in_tensor):
            array = in_tensor.detach().cpu().numpy().astype(np.uint8, copy=False)
        else:
            array = np.asarray(in_tensor, dtype=np.uint8)
        out = (
            _decode_table(over_mode)[array.reshape(-1)]
            .reshape(array.shape)
            .astype(np.float32, copy=False)
        )
        return out

    def _encode_float_array_to_hif8_uint8(array, round_mode="hybrid", over_mode=True):
        if round_mode != "hybrid":
            raise ValueError(f"unsupported hif8 round_mode: {round_mode}")
        values = np.asarray(array, dtype=np.float32).reshape(-1)
        out = np.zeros(values.shape, dtype=np.uint8)
        nan_mask = np.isnan(values)
        pos_inf_mask = np.isposinf(values)
        neg_inf_mask = np.isneginf(values)
        finite_mask = np.isfinite(values)
        out[nan_mask] = HIF8_NAN_CODE  # noqa: F821 -- unchanged supplied Golden, guarded MXFP8 path
        out[pos_inf_mask] = HIF8_POS_INF_CODE  # noqa: F821 -- unchanged supplied Golden, guarded MXFP8 path
        out[neg_inf_mask] = HIF8_NEG_INF_CODE  # noqa: F821 -- unchanged supplied Golden, guarded MXFP8 path
        if not np.any(finite_mask):
            return out.reshape(array.shape)
        finite_values = values[finite_mask]
        sign_mask = np.signbit(finite_values)
        abs_values = np.abs(finite_values).astype(np.float32, copy=False)
        (positive_round_values, positive_codes) = _positive_round_values_and_codes()
        idx = np.searchsorted(positive_round_values, abs_values, side="left")
        idx = np.clip(idx, 0, positive_round_values.shape[0] - 1)
        left_idx = np.clip(idx - 1, 0, positive_round_values.shape[0] - 1)
        right_idx = idx
        left_values = positive_round_values[left_idx]
        right_values = positive_round_values[right_idx]
        choose_right = abs_values - left_values > right_values - abs_values
        chosen_idx = np.where(choose_right, right_idx, left_idx)
        chosen_codes = positive_codes[chosen_idx].astype(np.uint8, copy=False)
        negative_codes = np.where(chosen_codes == 0, 0, chosen_codes | 128).astype(
            np.uint8, copy=False
        )
        out[finite_mask] = np.where(sign_mask, negative_codes, chosen_codes)
        if not over_mode:
            out = np.where(out == HIF8_POS_INF_CODE, np.uint8(110), out)  # noqa: F821 -- unchanged supplied Golden, guarded MXFP8 path
            out = np.where(out == HIF8_NEG_INF_CODE, np.uint8(238), out)  # noqa: F821 -- unchanged supplied Golden, guarded MXFP8 path
        return out.reshape(array.shape)

    def encode_float_tensor_to_hif8_uint8(
        in_tensor, round_mode="hybrid", over_mode=True
    ):
        if torch.is_tensor(in_tensor):
            array = in_tensor.detach().cpu().to(torch.float32).numpy()
        else:
            array = np.asarray(in_tensor, dtype=np.float32)
        return _encode_float_array_to_hif8_uint8(
            array, round_mode=round_mode, over_mode=over_mode
        )

    def quantize_tensor_to_hif8_native_float32(
        in_tensor, round_mode="hybrid", over_mode=True
    ):
        encoded = encode_float_tensor_to_hif8_uint8(
            in_tensor, round_mode=round_mode, over_mode=over_mode
        )
        decoded = decode_hif8_uint8_tensor_to_float32(encoded, over_mode=over_mode)
        if torch.is_tensor(in_tensor):
            return torch.from_numpy(decoded).to(torch.float32)
        return decoded

    def ensure_hif8_native_float32_tensor(
        in_tensor, round_mode="hybrid", over_mode=True
    ):
        if in_tensor is None:
            return None
        return quantize_tensor_to_hif8_native_float32(
            in_tensor, round_mode=round_mode, over_mode=over_mode
        )

    def _require_legacy_tools():
        if tools is None:  # noqa: F821 -- unchanged supplied Golden, guarded MXFP8 path
            raise RuntimeError("libs.tools is required for legacy input rewrite paths")
        return tools  # noqa: F821 -- unchanged supplied Golden, guarded MXFP8 path

    def str_to_bool_list(s: str):
        bool_list = []
        for item in s:
            if item == 0:
                bool_list.append(False)
            else:
                bool_list.append(True)
        return bool_list

    def trans_input_dtype(input_dtype):
        if input_dtype == "fp16":
            return "float16"
        elif input_dtype == "int8":
            return "int8"
        elif input_dtype == "uint8":
            return "uint8"
        elif input_dtype == "bf16":
            return "bfloat16"
        elif input_dtype == "bool":
            return "bool"
        elif input_dtype == "int32":
            return "int32"
        elif input_dtype == "fp32":
            return "float32"
        else:
            return input_dtype

    def _get_hif8_dtype():
        if torch_npu is None:  # noqa: F821 -- unchanged supplied Golden, guarded MXFP8 path
            return None
        return getattr(torch_npu, "hifloat8", None)  # noqa: F821 -- unchanged supplied Golden, guarded MXFP8 path

    def _get_full_quant_mode_config(weight_quant_mode):
        if weight_quant_mode == WEIGHT_QUANT_MODE_FULL_INT8:
            return {
                "input_dtype": torch.int8,
                "dtype_max": INT8_DTYPE_MAX,
                "query_quant_dtype": torch.int8,
            }
        if weight_quant_mode == WEIGHT_QUANT_MODE_FULL_FP8_E4M3:
            if not hasattr(torch, "float8_e4m3fn"):
                return None
            return {
                "input_dtype": torch.float8_e4m3fn,
                "dtype_max": FP8_E4M3_DTYPE_MAX,
                "query_quant_dtype": torch.float8_e4m3fn,
            }
        if weight_quant_mode == WEIGHT_QUANT_MODE_FULL_HIF8:
            return {
                "input_dtype": torch.float32,
                "dtype_max": HIF8_DTYPE_MAX,
                "query_quant_dtype": torch.float32,
            }
        return None

    def _get_quant_dtype_max(quant_dtype):
        if quant_dtype == torch.int8:
            return INT8_DTYPE_MAX
        if hasattr(torch, "float8_e4m3fn") and quant_dtype == torch.float8_e4m3fn:
            return FP8_E4M3_DTYPE_MAX
        hif8_dtype = _get_hif8_dtype()
        if hif8_dtype is not None and quant_dtype == hif8_dtype:
            return HIF8_DTYPE_MAX
        raise ValueError(f"unsupported quant dtype: {quant_dtype}")

    def _is_hif8_surrogate_quant(quant_dtype, dtype_max):
        return quant_dtype == torch.float32 and float(dtype_max) == float(
            HIF8_DTYPE_MAX
        )

    def _quantize_hif8_surrogate(values):
        return quantize_tensor_to_hif8_native_float32(
            values, round_mode="hybrid", over_mode=True
        )

    def _dynamic_quant_clip_range(dtype_max, quant_dtype):
        if quant_dtype == torch.int8:
            return (-128.0, 127.0)
        return (-float(dtype_max), float(dtype_max))

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.concatenate((-x2, x1), dim=-1)

    def scatter_pa_nz(cache, inputs, index, data_size=16):
        bn = cache.shape[0]  # noqa: F841 -- unchanged supplied Golden, guarded MXFP8 path
        bs = cache.shape[1]
        n = cache.shape[2]
        h = cache.shape[3]
        bxs1 = inputs.shape[0]
        if len(cache.shape) != 4 or len(inputs.shape) != 2 or len(index.shape) != 1:
            pass
            return cache
        elif inputs.shape[1] != h:
            pass
            return cache
        elif index.shape[0] != bxs1:
            pass
            return cache
        data_num = math.ceil(h / data_size)
        for input_index in range(bxs1):
            index_num = index[input_index]
            block_num_index = int(index_num / bs)
            for data_index in range(data_num):
                data_index_in_block = data_index * bs + index_num % bs
                block_size_index = int(data_index_in_block / data_num)
                h_start_index = int(data_index_in_block % data_num) * data_size
                h_end_index = h_start_index + data_size
                input_start_index = data_index * data_size
                input_end_index = input_start_index + data_size
                for n_index in range(n):
                    cache[
                        block_num_index,
                        block_size_index,
                        n_index,
                        h_start_index:h_end_index,
                    ] = inputs[input_index, input_start_index:input_end_index]
        return cache

    def scatter_pa_blk_bsnd(
        cache: np.ndarray, input_: np.ndarray, index: np.ndarray, seq_len, B: int
    ):
        """
        将 input_ 写入 PA_BLK_BSND 格式的 cache（CPU端参考实现，按页写入）。
        - cache: (num_pa_blocks, blk_size, N, H)
        - input_: (total_tokens, H)  # 按 batch 拼接后的 token 序列
        - index: (sum_pages,)  # 扁平一维，指向 cache 第0维（块 id）
                 第 i 个元素给出第 i 个“页”的 pa_blk_id（无需 // blk_size）
        - seq_len: BSND: int；TND: 1D 数组（长度表或前缀和）
        - B: batch size（TND 中会被 seq_len 推导覆盖）
        """
        assert cache.ndim == 4, "cache 必须是 (num_pa_blocks, blk_size, N, H)"
        (num_pa_blocks, blk_size, N, H) = cache.shape
        if isinstance(seq_len, (int, np.integer)):
            S1 = int(seq_len)
            assert input_.shape[0] == B * S1, (
                f"BSND: input_.shape[0] 必须等于 B*S1={B * S1}"
            )
            pages_per_b = math.ceil(S1 / blk_size)
            assert index.ndim == 1 and index.shape[0] == B * pages_per_b, (
                f"BSND: index 长度应为 B*ceil(S1/blk)={B * pages_per_b}，而现在是 {index.shape[0]}"
            )
            seq_list = np.full((B,), S1, dtype=np.int64)
            start_list = np.arange(B, dtype=np.int64) * S1
        else:
            seq_arr = np.asarray(seq_len).reshape(-1)
            B = int(seq_arr.shape[0])
            if B >= 2 and seq_arr[0] >= 0:
                seq_list = np.empty(B, dtype=np.int64)
                seq_list[0] = int(seq_arr[0])
                if B > 1:
                    seq_list[1:] = np.diff(seq_arr).astype(np.int64)
                start_list = np.empty(B, dtype=np.int64)
                start_list[0] = 0
                if B > 1:
                    start_list[1:] = seq_arr[:-1].astype(np.int64)
            else:
                seq_list = seq_arr.astype(np.int64)
                start_list = np.zeros(B, dtype=np.int64)
                if B > 1:
                    start_list[1:] = np.cumsum(seq_list[:-1])
            expected_len = int(np.sum((seq_list + blk_size - 1) // blk_size))
            assert index.ndim == 1 and index.shape[0] == expected_len, (
                f"TND: index 长度应为 {expected_len}，而现在是 {index.shape[0]}"
            )
        flat_page_ptr = 0
        for b in range(B):
            len_b = int(seq_list[b])
            base_in = int(start_list[b])
            if len_b <= 0:
                continue
            pages_b = (len_b + blk_size - 1) // blk_size
            for p in range(pages_b):
                pa_blk_id = int(index[flat_page_ptr + p])
                assert 0 <= pa_blk_id < num_pa_blocks, (
                    f"pa_blk_id={pa_blk_id} 越界[0,{num_pa_blocks}), (b={b}, p={p})"
                )
                start = base_in + p * blk_size
                end = min(start + blk_size, base_in + len_b)
                sz = end - start
                if sz <= 0:
                    continue
                cache[pa_blk_id, :sz, :, :H] = input_[start:end, :H][:, None, :]
            flat_page_ptr += pages_b
        assert flat_page_ptr == index.shape[0], (
            f"index 被消费 {flat_page_ptr} 项，但长度为 {index.shape[0]}（不一致）"
        )
        return cache

    def scatter_pa_blk_nz(
        cache: np.ndarray,
        input_: np.ndarray,
        index: np.ndarray,
        seq_len,
        B,
        data_size: int = 16,
    ):
        """
        支持 BSND/TND 的 PA_BLK_NZ scatter 参考实现（CPU 端验证用）。

        Args:
            cache: (num_pa_blocks, blk_size, N, H_pad)
            input_:
                - BSND: (B*S1, H)
                - TND:  (sum(seq_b), H)，seq_b 可不同
            index: [ceil(S1[i]/blk_size)]，元素为 cachePaOffset (单位：行)
            seq_len:
                - int:   S1，所有 batch 的序列长度一致（BSND）
                - array: shape [B] 或 [B,1]
                         * 若严格单调非减 → 当作“前缀和”
                         * 否则 → 当作“每个 batch 的长度列表”
            data_size: 列打包粒度（常见 16/32/64）

        Returns:
            cache: 原位写入并返回
        """
        assert cache.ndim == 4, "cache 必须是 (num_pa_blocks, blk_size, N, H)"
        (num_pa_blocks, blk_size, N, H) = cache.shape
        if isinstance(seq_len, (int, np.integer)):
            S1 = int(seq_len)
            assert input_.shape[0] == B * S1, (
                f"BSND: input_.shape[0] 必须等于 B*S1={B * S1}, 但现在是{input_.shape[0]}"
            )
            pages_per_b = math.ceil(S1 / blk_size)  # noqa: F841 -- unchanged supplied Golden, guarded MXFP8 path
            seq_list = np.full((B,), S1, dtype=np.int64)
            start_list = np.arange(B, dtype=np.int64) * S1
            expected_pages = math.ceil(S1 / blk_size)
            assert index.shape[1] == expected_pages, (
                f"index second dim({index.shape[1]}) must be ceil(S1/blk_size)({expected_pages})"
            )
        else:
            seq_arr = np.asarray(seq_len).reshape(-1)
            B = int(seq_arr.shape[0])
            if B >= 2 and seq_arr[0] >= 0:
                seq_list = np.empty(B, dtype=np.int64)
                seq_list[0] = int(seq_arr[0])
                if B > 1:
                    seq_list[1:] = np.diff(seq_arr).astype(np.int64)
                start_list = np.empty(B, dtype=np.int64)
                start_list[0] = 0
                if B > 1:
                    start_list[1:] = seq_arr[:-1].astype(np.int64)
            else:
                seq_list = seq_arr.astype(np.int64)
                start_list = np.zeros(B, dtype=np.int64)
                if B > 1:
                    start_list[1:] = np.cumsum(seq_list[:-1])
            expected_len = int(np.sum((seq_list + blk_size - 1) // blk_size))
            assert index.ndim == 1 and index.shape[0] == expected_len, (
                f"TND: index 长度应为 {expected_len}，而现在是 {index.shape[0]}"
            )
        data_num = math.ceil(H / data_size)
        total_rows = num_pa_blocks * blk_size
        page_ptr = 0
        for b in range(B):
            S_b = int(seq_list[b])
            if S_b <= 0:
                continue
            pages_b = (S_b + blk_size - 1) // blk_size
            for p in range(pages_b):
                cachePaOffset_rows = int(index[page_ptr]) * blk_size
                page_ptr += 1
                assert 0 <= cachePaOffset_rows < total_rows, (
                    f"index 中的 cachePaOffset({cachePaOffset_rows}) 越界，总行数={total_rows}"
                )
                token_begin_in_b = p * blk_size
                tokens_in_page = min(blk_size, S_b - token_begin_in_b)
                if tokens_in_page <= 0:
                    continue
                for r in range(tokens_in_page):
                    in_row = int(start_list[b] + token_begin_in_b + r)
                    global_row = cachePaOffset_rows + r
                    pa_block_id = global_row // blk_size
                    row_in_block = global_row % blk_size
                    assert 0 <= pa_block_id < num_pa_blocks, (
                        f"pa_block_id={pa_block_id} 越界 (num_pa_blocks={num_pa_blocks})"
                    )
                    for data_index in range(data_num):
                        in_c0 = data_index * data_size
                        in_c1 = min(in_c0 + data_size, H)
                        data_index_in_block = data_index * blk_size + row_in_block
                        block_size_index = data_index_in_block // data_num
                        h_start_index = data_index_in_block % data_num * data_size
                        h_end_index = min(h_start_index + data_size, H)
                        width = min(in_c1 - in_c0, h_end_index - h_start_index)
                        if width <= 0:
                            continue
                        cache[
                            pa_block_id,
                            block_size_index,
                            :,
                            h_start_index : h_start_index + width,
                        ] = input_[in_row, in_c0 : in_c0 + width][None, :]
        return cache

    def s8_saturation(inputdata):
        inputdata = torch.where(inputdata > 127, 127, inputdata)
        inputdata = torch.where(inputdata < -128, -128, inputdata)
        return inputdata.to(torch.int8)

    def s9_saturation(inputdata):
        inputdata = torch.where(inputdata > 255, 255, inputdata)
        inputdata = torch.where(inputdata < -256, -256, inputdata)
        return inputdata

    def quant(x, qscale):
        scaled_values = (x * qscale).round().to(torch.float32)
        s9_res = s9_saturation(scaled_values)
        s8_res_cal = s8_saturation(s9_res)
        return s8_res_cal

    def numpy_float8_e4m3fn():
        try:
            from ml_dtypes import float8_e4m3fn

            return float8_e4m3fn
        except ModuleNotFoundError:
            raise RuntimeError(
                "ml_dtypes is needed to support float8_e4m3fn dtype!!! Please install with `pip3 install ml-dtypes`"
            )

    def quant_ckv_per_tensor(input, quant_scale_ckv):
        scaled_value = input * quant_scale_ckv
        scaled_value = np.round(scaled_value, 8)
        scaled_value = torch.from_numpy(scaled_value)
        scaled_value = torch.nan_to_num(scaled_value, nan=0.0)
        return torch.clamp(
            scaled_value, min=-FP8_E4M3_DTYPE_MAX, max=FP8_E4M3_DTYPE_MAX
        ).to(torch.float8_e4m3fn)

    def quant_with_scale(mla_param, x, qscale, quant_dtype=torch.int8, dtype_max=None):
        if quant_dtype == torch.int8 and (
            dtype_max is None or float(dtype_max) == float(INT8_DTYPE_MAX)
        ):
            return quant(x, qscale)
        if dtype_max is None:
            dtype_max = _get_quant_dtype_max(quant_dtype)
        scaled_values = (x * qscale).to(torch.float32)
        scaled_values = torch.nan_to_num(scaled_values, nan=0.0)
        (clip_min, clip_max) = _dynamic_quant_clip_range(dtype_max, quant_dtype)
        if _is_hif8_surrogate_quant(quant_dtype, dtype_max):
            scaled_values = torch.from_numpy(
                trans_np_float_tensor_to_hifuint8(scaled_values.numpy())  # noqa: F821 -- supplied unsupported HiF8 branch
            )  # noqa: F821 -- unchanged supplied Golden, guarded MXFP8 path
            scaled_values = torch.from_numpy(
                decode_hif8_uint8_tensor_to_float32(
                    scaled_values.numpy(), over_mode=False
                )
            )
            return scaled_values
        if mla_param["action_type"] == "bm_output_gold":
            return torch.clamp(scaled_values, min=clip_min, max=clip_max)
        return torch.clamp(scaled_values, min=clip_min, max=clip_max).to(quant_dtype)

    def _tensor_to_uint8_rows(tensor):
        tensor = tensor.contiguous()
        return tensor.view(torch.uint8)

    def _merge_quant_cache_repo_payload(
        mla_param, base_tensor, quant_dtype, rotary_bf16=None, deq_scale=None
    ):
        if quant_dtype == torch.float32:
            parts = [base_tensor.to(torch.float32)]
            if rotary_bf16 is not None and rotary_bf16.numel() != 0:
                if mla_param["action_type"] == "bm_output_gold":
                    parts.append(
                        _tensor_to_uint8_rows(rotary_bf16)
                        .to(torch.bfloat16)
                        .to(torch.float32)
                    )
                else:
                    parts.append(_tensor_to_uint8_rows(rotary_bf16).to(torch.float32))
            if deq_scale is not None and deq_scale.numel() != 0:
                if mla_param["action_type"] == "bm_output_gold":
                    parts.append(
                        _tensor_to_uint8_rows(deq_scale)
                        .to(torch.float32)
                        .to(torch.float64)
                    )
                else:
                    parts.append(_tensor_to_uint8_rows(deq_scale).to(torch.float32))
            if len(parts) == 1:
                return parts[0]
            return torch.cat(parts, axis=-1)
        parts = [_tensor_to_uint8_rows(base_tensor)]
        if rotary_bf16 is not None and rotary_bf16.numel() != 0:
            parts.append(_tensor_to_uint8_rows(rotary_bf16))
        if deq_scale is not None and deq_scale.numel() != 0:
            parts.append(_tensor_to_uint8_rows(deq_scale))
        if len(parts) == 1:
            return base_tensor
        merged_bytes = torch.cat(parts, axis=-1)
        item_size = torch.empty((), dtype=quant_dtype).element_size()
        if merged_bytes.shape[-1] % item_size != 0:
            raise ValueError(
                f"cannot reinterpret merged byte payload of shape {tuple(merged_bytes.shape)} to {quant_dtype}"
            )
        return (
            merged_bytes.contiguous()
            .view(quant_dtype)
            .reshape(merged_bytes.shape[0], -1)
        )

    def dynamic_quant(
        mla_param,
        inputs,
        smooth_scale,
        dtype_max=INT8_DTYPE_MAX,
        quant_dtype=torch.int32,
    ):
        T = inputs.size(0)
        H = inputs.size(1)
        y = torch.zeros(T, H).to(torch.float32)
        scale = torch.zeros(T, 1).to(torch.float32)
        inputs = inputs.reshape(T, H).to(torch.float32)
        if smooth_scale != None and len(smooth_scale) != 0:  # noqa: E711 -- unchanged supplied Golden, guarded MXFP8 path
            if len(smooth_scale.shape) != 2 or smooth_scale.shape[1] != H:
                pass
                return (None, None)
            smooth_scale = smooth_scale.to(torch.float32)
            for bs_index in range(T):
                abs_bs_tensor = torch.abs(inputs[bs_index, :] * smooth_scale[0, :])
                scale_bs = abs_bs_tensor.max() / float(dtype_max)
                scale[bs_index] = scale_bs
                y[bs_index:] = inputs[bs_index:] * smooth_scale[0, :] / scale_bs
        else:
            for bs_index in range(T):
                abs_bs_tensor = torch.abs(inputs[bs_index, :])
                scale_bs = abs_bs_tensor.max() / float(dtype_max)
                scale[bs_index] = scale_bs
                y[bs_index:] = inputs[bs_index:] / scale_bs
        y = torch.nan_to_num(y, nan=0.0)
        if quant_dtype == torch.int32:
            y = torch.round(y.to(torch.int32))
            scale = torch.round(scale)
        elif _is_hif8_surrogate_quant(quant_dtype, dtype_max):
            y = torch.from_numpy(trans_np_float_tensor_to_hifuint8(y.numpy()))  # noqa: F821 -- unchanged supplied Golden, guarded MXFP8 path
            y = torch.from_numpy(
                decode_hif8_uint8_tensor_to_float32(y.numpy(), over_mode=False)
            )
        else:
            (clip_min, clip_max) = _dynamic_quant_clip_range(dtype_max, quant_dtype)
            if mla_param["action_type"] != "bm_output_gold":
                y = torch.clamp(y, min=clip_min, max=clip_max).to(quant_dtype)
            else:
                y = torch.clamp(y, min=clip_min, max=clip_max)
        return (y, scale)

    def dynamic_quant_int8(inputs, smooth_scale):
        T = inputs.size(0)
        H = inputs.size(1)
        y = torch.zeros(T, H).to(torch.float32)
        scale = torch.zeros(T, 1).to(torch.float32)
        inputs = inputs.reshape(T, H).to(torch.float32)
        if smooth_scale != None:  # noqa: E711 -- unchanged supplied Golden, guarded MXFP8 path
            if len(smooth_scale.shape) != 2 or smooth_scale.shape[1] != H:
                pass
                return (None, None)
            smooth_scale = smooth_scale.to(torch.float32)
            for bs_index in range(T):
                abs_bs_tensor = torch.abs(inputs[bs_index, :] * smooth_scale[0, :])
                scale_bs = abs_bs_tensor.max() / 127
                scale[bs_index] = scale_bs
                y[bs_index:] = torch.round(
                    inputs[bs_index:] * smooth_scale[0, :] / scale_bs
                ).to(torch.int8)
        else:
            for bs_index in range(T):
                abs_bs_tensor = torch.abs(inputs[bs_index, :])
                scale_bs = abs_bs_tensor.max() / 127
                scale[bs_index] = scale_bs
                y[bs_index:] = torch.round(inputs[bs_index:] / scale_bs).to(torch.int8)
        y = torch.nan_to_num(y, nan=0.0)
        return (y, scale)

    def get_dtype_range(dt):
        if "bfloat16" in str(dt):
            return (-float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127"))
        if "uint4" in str(dt):
            return (0, 15)
        if "int4" in str(dt):
            return (-8, 7)
        if "bool" in str(dt):
            return (0, 1)
        if "float4_e2m1" in str(dt):
            return (-float.fromhex("0x1.8p2"), float.fromhex("0x1.8p2"))
        if "float4_e1m2" in str(dt):
            return (-float.fromhex("0x1.Cp0"), float.fromhex("0x1.Cp0"))
        if "float8_e8m0" in str(dt):
            return (float.fromhex("0x1.p-127"), float.fromhex("0x1.p127"))
        if "float8_e5m2" in str(dt):
            return (-float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15"))
        if "float8_e4m3fn" in str(dt):
            return (-float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8"))
        if "hifloat8" in str(dt):
            return (-float.fromhex("0x1.p15"), float.fromhex("0x1.p15"))
        if "complex32" in str(dt):
            dt = "float16"
        numpy_dtype = np.dtype(dt)
        if numpy_dtype.kind in "iu":
            numpy_info = np.iinfo(numpy_dtype)
        else:
            numpy_info = np.finfo(numpy_dtype)
        return (numpy_info.min, numpy_info.max)

    def _mx_reshape_to_blocks(fp_array: np.ndarray, axis: int, block_size: int):
        fp_array = np.expand_dims(fp_array, axis=axis + 1)
        orig_shape = fp_array.shape
        pad = [[0, 0] for _ in range(len(orig_shape))]
        pad_size = orig_shape[axis] % block_size
        pad[axis][1] = block_size - pad_size
        if pad_size > 0:
            fp_array = np.pad(fp_array, pad, "constant")
        padded_shape = fp_array.shape
        reshape = list(padded_shape)
        reshape[axis + 1] = block_size
        reshape[axis] = reshape[axis] // block_size
        fp_array = fp_array.reshape(reshape)
        return (fp_array, orig_shape, padded_shape)

    def _mx_calculate_share_exp(
        fp_array: np.ndarray, scale_axis: int, mx_ele_dtype: str
    ):
        FP32_EXPONENT_BIAS = 127
        FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
        max_norm = get_dtype_range(mx_ele_dtype)[1]
        ele_emax = int(np.log2(max_norm))
        fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdims=True)
        res = (
            np.floor(
                np.log2(
                    fp_abs_max.astype(np.float32) + FP32_MIN_NORMAL * (fp_abs_max == 0)
                )
            )
            - ele_emax
        )
        res = res + FP32_EXPONENT_BIAS
        res[fp_abs_max == 0] = 0
        return res

    def _mx_round_mantissa(fp_array: np.ndarray, round_mode: str):
        """
        For example:
        fp_array  = [-4.5, -3.5, -2.5, -2.0, -1.7, -1.5, -1.4, -0.5, -0.2, -0.0, 0.0, 0.2, 0.5, 1.4, 1.5, 1.7, 2.0, 2.5, 3.4, 4.5]
        - rint    = [-4.,  -4.,  -2.,  -2.,  -2.,  -2.,  -1.,  -0.,  -0.,  -0.,  0.,  0.,  0.,  1.,  2.,  2.,  2.,  2.,  3.,  4.]
        - nearest = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -1.,  -1.,  -0.,  -0.,  0.,  0.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  5.]
        - floor   = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -2.,  -1.,  -1.,  -0.,  0,   0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
        - ceil    = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  -0.,  0.,  1.,  1.,  2.,  2.,  2.,  2.,  3.,  4.,  5.]
        - trunc   = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  -0.,  0.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
        """
        if round_mode in ("rint", "even"):
            fp_array = np.rint(fp_array)
        elif round_mode in ("round", "nearest"):
            sign = np.signbit(fp_array)
            rounded_abs = np.floor(
                np.abs(fp_array) + np.array([0.5], dtype=fp_array.dtype)
            )
            fp_array = np.where(sign, -rounded_abs, rounded_abs)
        elif round_mode == "floor":
            fp_array = np.floor(fp_array)
        elif round_mode == "ceil":
            fp_array = np.ceil(fp_array)
        elif round_mode == "trunc":
            fp_array = np.trunc(fp_array)
        else:
            raise Exception(f"Unrecognized round method {round_mode}")
        return fp_array

    def _mx_quantize_to_element_format(
        fp_array: np.ndarray, share_exp: np.ndarray, mx_ele_dtype: str, round_mode: str
    ):
        mx_dtype = str(mx_ele_dtype)
        match = re.search("e(\\d+)m(\\d+)", mx_dtype)
        if match:
            exp_bits = int(match.group(1))
            mantissa_bits = int(match.group(2))
        else:
            raise ValueError(f"mx element dtype [{mx_ele_dtype}] is not recognized.")
        ret = fp_array / 2 ** (share_exp - 127)
        private_exp = np.floor(
            np.log2(np.abs(ret.astype(np.float32)) + (ret == 0))
        ).astype(fp_array.dtype, copy=False)
        min_exp = 0 if "float4_e1m2" in mx_dtype else -(2 ** (exp_bits - 1)) + 2
        private_exp = private_exp.clip(min=min_exp)
        ret = ret / 2**private_exp * 2**mantissa_bits
        ret = _mx_round_mantissa(ret, round_mode)
        ret = ret / 2**mantissa_bits * 2**private_exp
        max_norm = get_dtype_range(mx_dtype)[1]
        np.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
        return ret

    def pad_to_even(tensor: np.ndarray, axis: int) -> np.ndarray:
        """
        在指定的 axis 上将 tensor 对齐到偶数长度（即按 2 对齐），不足补 0。

        参数:
            tensor (np.ndarray): 输入数组
            axis (int): 要对齐的轴（从 0 开始）

        返回:
            np.ndarray: 对齐后的数组
        """
        if not isinstance(tensor, np.ndarray):
            raise ValueError("Input must be a numpy ndarray.")
        if axis < 0 or axis >= tensor.ndim:
            raise ValueError(
                f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions."
            )
        shape = tensor.shape
        length = shape[axis]
        if length % 2 == 0:
            return tensor
        pad_width = [(0, 0)] * tensor.ndim
        pad_width[axis] = (0, 1)
        padded_tensor = np.pad(
            tensor, pad_width, mode="constant", constant_values=2 ** (-127)
        )
        return padded_tensor

    def _mx_undo_reshape_to_blocks(
        fp_array: np.ndarray, axis: int, orig_shape: tuple, padded_shape: tuple
    ):
        fp_array = fp_array.reshape(padded_shape)
        if tuple(padded_shape) != tuple(orig_shape):
            slices = [slice(0, x) for x in orig_shape]
            fp_array = fp_array[tuple(slices)]
        fp_array = np.squeeze(fp_array, axis=axis + 1)
        return fp_array

    def interleave(tensor: np.ndarray, axis: int, n_group: int = 2) -> np.ndarray:
        if not isinstance(tensor, np.ndarray):
            raise ValueError("Input must be a numpy ndarray.")
        if axis < 0 or axis >= tensor.ndim:
            raise ValueError(
                f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions."
            )
        length = tensor.shape[axis]
        if length % n_group != 0:
            raise ValueError(
                f"Axis length ({length}) must be divisible by n_group ({n_group})"
            )
        group_length = length // n_group
        shape = list(tensor.shape)
        new_shape = shape[:axis] + [group_length, 2] + shape[axis + 1 :]
        reshaped = tensor.reshape(new_shape)
        transpose_order = (
            list(range(0, axis + 1))
            + list(range(axis + 2, len(new_shape)))
            + [axis + 1]
        )
        transposed = reshaped.transpose(transpose_order)
        return transposed

    def _require_ml_dtypes():
        try:
            import ml_dtypes
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ml_dtypes is required for MXFP8 CPU reference paths"
            ) from exc
        return ml_dtypes

    def _get_ml_dtype(name):
        ml_dtypes = _require_ml_dtypes()
        if not hasattr(ml_dtypes, name):
            raise RuntimeError(f"ml_dtypes does not provide {name}")
        return getattr(ml_dtypes, name)

    def dynamic_mx_quant_cq(
        fp_array: np.ndarray,
        mx_ele_dtype: str = "float4_e2m1",
        axis: int = -1,
        block_size: int = 32,
        round_mode: str = "rint",
    ) -> tuple:
        if not isinstance(fp_array, np.ndarray):
            raise RuntimeError(
                f"Input tensor to be quantized should be numpy array. But got {type(fp_array)}"
            )
        if fp_array.dtype.name not in ("bfloat16", "float16", "float32"):
            raise RuntimeError(
                f"Dtype of input tensor to be quantized is not supported: {fp_array.dtype.name}"
            )
        if mx_ele_dtype not in (
            "float4_e2m1",
            "float4_e1m2",
            "float8_e4m3fn",
            "float8_e5m2",
        ):
            raise NotImplementedError(f"Not support {mx_ele_dtype} yet!")
        axis = len(fp_array.shape) + axis if axis < 0 else axis
        (fp_array, orig_shape, padded_shape) = _mx_reshape_to_blocks(
            fp_array, axis, block_size
        )
        share_exp = _mx_calculate_share_exp(
            fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype
        )
        scale_emax = 2 ** (8 - 1) - 1
        share_exp[share_exp - 127 > scale_emax] = 127
        share_exp[share_exp - 127 < -scale_emax] = -scale_emax
        ele_array = _mx_quantize_to_element_format(
            fp_array, share_exp, mx_ele_dtype, round_mode
        )
        ele_array = _mx_undo_reshape_to_blocks(
            ele_array, axis, orig_shape, padded_shape
        )
        share_exp = np.squeeze(share_exp, axis=axis + 1)
        ele_dtype_np = _get_ml_dtype(mx_ele_dtype)
        scale_array = share_exp
        if ele_array.dtype.name == "bfloat16":
            ele_array = ele_array.astype("float32", copy=False)
        ele_array = np.nan_to_num(ele_array, nan=0.0, copy=False)
        ele_array = ele_array.astype(ele_dtype_np, copy=False)
        scale_array_pad = pad_to_even(scale_array, axis=axis)
        result_shape = copy.deepcopy(list(scale_array_pad.shape))
        result_shape.append(2)
        result_shape[axis] = scale_array_pad.shape[axis] // 2
        if axis != len(fp_array.shape) - 1:
            scale_array_pad = interleave(scale_array_pad, axis=axis)
        scale_array_pad = scale_array_pad.reshape(result_shape)
        scale_array = scale_array_pad.astype("uint8", copy=False)
        return (scale_array, ele_array)

    def dynamic_mx_quant_qn(x, mla_param):
        scale_max = np.float32(448.0)
        x = x.astype("float32")
        input_mul = x
        input_abs = np.abs(input_mul)
        input_max = np.max(input_abs, axis=-1, keepdims=True)
        scale = input_max * (np.float32(1.0) / scale_max)
        input_scaled = input_mul / scale
        input_scaled = np.nan_to_num(input_scaled, nan=0.0)
        if mla_param["action_type"] != "bm_output_gold":
            output_data = torch.clamp(
                torch.from_numpy(input_scaled),
                min=-FP8_E4M3_DTYPE_MAX,
                max=FP8_E4M3_DTYPE_MAX,
            ).to(torch.float8_e4m3fn)
        else:
            output_data = torch.from_numpy(input_scaled)
        return (scale, output_data)

    def dynamic_quant_ckv_with_amax(
        inputs: torch.Tensor,
        amax: torch.Tensor,
        smooth_scale: torch.Tensor = None,
        quant_dtype: torch.dtype = torch.int8,
        dtype_max=None,
    ):
        """
        三维动态量化 FP32 -> INT8（每行 scale 由外部提供 amax）
        inputs: [T, N, H] 张量
        amax: [T, N, 1] 每行 H 的最大绝对值
        smooth_scale: None 或 [1, N, H] 张量
        返回:
            y: [T, N, H] 量化结果
            scale: [T, N] 每行 scale
        """
        (T, N, H) = inputs.shape
        inputs = inputs.to(torch.float32)
        amax = amax.to(torch.float32).clamp(min=1e-08)
        if smooth_scale is not None:
            if smooth_scale.ndim != 3 or smooth_scale.shape[1:] != (N, H):
                pass
                return (None, None)
            smooth_scale = smooth_scale.to(torch.float32)
            scaled_inputs = inputs * smooth_scale
        else:
            scaled_inputs = inputs
        if dtype_max is None:
            dtype_max = _get_quant_dtype_max(quant_dtype)
        scale = amax / float(dtype_max)
        if quant_dtype == torch.int8:
            y = torch.round(scaled_inputs / scale)
        else:
            y = scaled_inputs / scale
        y = torch.nan_to_num(y, nan=0.0)
        if _is_hif8_surrogate_quant(quant_dtype, dtype_max):
            y = torch.from_numpy(trans_np_float_tensor_to_hifuint8(y.numpy()))  # noqa: F821 -- unchanged supplied Golden, guarded MXFP8 path
            y = torch.from_numpy(
                decode_hif8_uint8_tensor_to_float32(y.numpy(), over_mode=False)
            )
        else:
            (clip_min, clip_max) = _dynamic_quant_clip_range(dtype_max, quant_dtype)
            y = torch.clamp(y, min=clip_min, max=clip_max).to(quant_dtype)
        return (y, scale)

    def dynamic_quant_ckv_with_amax_fp8e4m3(inputs: torch.Tensor, amax: torch.Tensor):
        """
        三维动态量化 FP32 -> INT8（每行 scale 由外部提供 amax）
        inputs: [T, N] 张量
        amax: [T, 1] 最大绝对值
        返回:
            y: [T, N] 量化结果
            scale: [T, 1] 每行 scale
        """
        (T, N) = inputs.shape
        scale = amax / 448.0
        y = torch.clamp(inputs / scale, min=-448, max=448)
        y = torch.nan_to_num(y, nan=0.0)
        return (y, scale)

    def dynamic_quant_without_smooth_scale(
        inputs, out_deqq_shape_shape, dtype_max=INT8_DTYPE_MAX, quant_dtype=torch.int8
    ):
        T = inputs.size(0)
        N = inputs.size(1)
        H = inputs.size(2)
        quant_loops = inputs.size(0)
        eles_with_one_scale = inputs.size(1) * inputs.size(2)
        if len(out_deqq_shape_shape) == 3:
            quant_loops = inputs.size(0) * inputs.size(1)
            eles_with_one_scale = inputs.size(2)
        y = torch.zeros(quant_loops, eles_with_one_scale).to(torch.float32)
        scale = torch.zeros(quant_loops).to(torch.float32)
        inputs = inputs.reshape(quant_loops, eles_with_one_scale).to(torch.float32)
        (max_values, _) = torch.max(torch.abs(inputs), dim=-1, keepdim=True)
        scale = max_values / float(dtype_max)
        y = inputs / scale
        y = torch.nan_to_num(y, nan=0.0)
        if _is_hif8_surrogate_quant(quant_dtype, dtype_max):
            y = torch.from_numpy(trans_np_float_tensor_to_hifuint8(y.numpy()))  # noqa: F821 -- unchanged supplied Golden, guarded MXFP8 path
            y = torch.from_numpy(
                decode_hif8_uint8_tensor_to_float32(y.numpy(), over_mode=False)
            )
        else:
            (clip_min, clip_max) = _dynamic_quant_clip_range(dtype_max, quant_dtype)
            y = torch.clamp(y, min=clip_min, max=clip_max).to(quant_dtype)
        if len(out_deqq_shape_shape) == 2:
            pass
            return (y.reshape(T, N, H), scale.reshape(quant_loops, 1).to(torch.float64))
        else:
            pass
            return (y.reshape(T, N, H), scale.reshape(T, N, 1).to(torch.float64))

    def dynamic_quant_without_smooth_scale_int8(inputs, out_deqq_shape_shape):
        T = inputs.size(0)
        N = inputs.size(1)
        H = inputs.size(2)
        quant_loops = inputs.size(0)
        eles_with_one_scale = inputs.size(1) * inputs.size(2)
        if len(out_deqq_shape_shape) == 3:
            quant_loops = inputs.size(0) * inputs.size(1)
            eles_with_one_scale = inputs.size(2)
        y = torch.zeros(quant_loops, eles_with_one_scale).to(torch.float32)
        scale = torch.zeros(quant_loops).to(torch.float32)
        inputs = inputs.reshape(quant_loops, eles_with_one_scale).to(torch.float32)
        (max_values, _) = torch.max(torch.abs(inputs), dim=-1, keepdim=True)
        scale = max_values / 127
        y = torch.round(inputs / scale)
        y = s8_saturation(y)
        y = torch.nan_to_num(y, nan=0.0)
        if len(out_deqq_shape_shape) == 2:
            pass
            return (y.reshape(T, N, H), scale.reshape(quant_loops, 1).to(torch.float64))
        else:
            pass
            return (y.reshape(T, N, H), scale.reshape(T, N, 1).to(torch.float64))

    def dequant(inputs, deq_scale_q_nope, quant_scale_ckv):
        org_shape = inputs.size()
        quant_loops = inputs.size(0)
        eles_with_one_scale = inputs.size(1) * inputs.size(2)
        if len(deq_scale_q_nope.size()) == 3:
            quant_loops = inputs.size(0) * inputs.size(1)
            eles_with_one_scale = inputs.size(2)
        inputs = inputs.reshape(quant_loops, eles_with_one_scale)
        deq_scale_q_nope = deq_scale_q_nope.reshape(quant_loops, 1)
        for quant_idx in range(quant_loops):
            inputs[quant_idx, :] = (
                inputs[quant_idx, :] * quant_scale_ckv / deq_scale_q_nope[quant_idx, 0]
            )
        return inputs.reshape(org_shape)

    def trans_torch_fp8_e8m0_to_bf16(array):
        new_array = torch.zeros_like(array).to(torch.bfloat16)
        for i in range(array.size(0)):
            for j in range(array.size(1)):
                new_value = ieee_754_conversion(0, int(array[i][j]), 0)
                new_array[i][j] = new_value
        return new_array

    def ieee_754_conversion(sign, exponent_raw, mantissa, exp_len=8, mant_len=7):
        """Convert binary data into the floating point value"""
        sign_mult = -1 if sign == 1 else 1
        exponent = exponent_raw - (2 ** (exp_len - 1) - 1)
        mant_mult = 1
        for b in range(mant_len - 1, -1, -1):
            if mantissa & 2**b:
                mant_mult += 1 / 2 ** (mant_len - b)
        return sign_mult * 2**exponent * mant_mult

    def convert_dict_values_to_torch(input_dict):
        output_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, np.ndarray):
                if value.dtype == "float8_e4m3fn":
                    pass
                    output_dict[key] = torch.tensor(value.astype(np.float32)).to(
                        torch.float8_e4m3fn
                    )
                elif value.dtype == "float8_e8m0":
                    pass
                    output_dict[key] = torch.tensor(value.astype(np.int8))
                else:
                    pass
                    output_dict[key] = torch.from_numpy(value)
            else:
                output_dict[key] = value
        return output_dict

    def get_current_seq_len(token_idx: int, actual_seq_length: np.ndarray) -> int:
        """
        给定 T 维下的 token 索引 (token_idx)，返回它所属 batch 的 sequence length。

        Args:
            token_idx (int): 当前 token 在 TND layout 中的 index (0-based)
            actual_seq_length (np.ndarray): shape [B,1]，每个 batch 的序列长度前缀和

        Returns:
            int: 当前 token 所属 batch 的序列长度
        """
        prefix_sum = actual_seq_length.flatten().tolist()
        batch_id = bisect.bisect_right(prefix_sum, token_idx)  # noqa: F821 -- unchanged supplied Golden, guarded MXFP8 path
        if batch_id == 0:
            seq_len = prefix_sum[0]
        else:
            seq_len = prefix_sum[batch_id] - prefix_sum[batch_id - 1]
        return seq_len

    def get_param(torch_tensor_list, params):
        res = {}
        res["normal_flag"] = True
        res["action_type"] = params["action_type"]
        res["device"] = _get_legacy_device_type()
        res["flaglist"] = str_to_bool_list(params["flaglist"])
        res["B"] = params["b"]
        res["N1"] = params["n1"]
        res["N2"] = params["n2"]
        res["D"] = params["d"]
        res["Dr"] = params["dr"]
        res["S1"] = params["s1"]
        res["S2"] = params["s2"]
        res["He"] = params["he"]
        res["Hcq"] = params["hcq"]
        res["Hckv"] = params["hckv"]
        res["T"] = params["t"]
        res["block_size"] = params["block_size"]
        res["cache_mode"] = params["cache_mode"]
        res["qnorm_flag"] = params["qnorm_flag"]
        res["actual_seq_len"] = None
        if res["flaglist"][24]:
            res["actual_seq_len"] = params["actual_seq_value"]
        res["x_shape"] = params["shape_input"][0]
        res["x_dtype"] = trans_input_dtype(params["dtype_input"][0])
        res["x_tensor"] = torch_tensor_list[0]
        if res["x_dtype"] == "hifloat8":
            res["x_tensor"] = decode_hif8_uint8_tensor_to_float32(
                res["x_tensor"], over_mode=False
            )
        res["w_dq_shape"] = params["shape_input"][1]
        res["w_dq_dtype"] = trans_input_dtype(params["dtype_input"][1])
        res["w_dq_tensor"] = torch_tensor_list[1]
        if res["w_dq_dtype"] == "hifloat8":
            res["w_dq_tensor"] = decode_hif8_uint8_tensor_to_float32(
                res["w_dq_tensor"], over_mode=False
            )
        res["w_uq_qr_shape"] = params["shape_input"][2]
        res["w_uq_qr_dtype"] = trans_input_dtype(params["dtype_input"][2])
        res["w_uq_qr_tensor"] = torch_tensor_list[2]
        if res["w_uq_qr_dtype"] == "hifloat8":
            res["w_uq_qr_tensor"] = decode_hif8_uint8_tensor_to_float32(
                res["w_uq_qr_tensor"], over_mode=False
            )
        res["w_uk_shape"] = params["shape_input"][3]
        res["w_uk_dtype"] = trans_input_dtype(params["dtype_input"][3])
        res["w_uk_tensor"] = torch_tensor_list[3]
        res["w_dkv_kr_shape"] = params["shape_input"][4]
        res["w_dkv_kr_dtype"] = trans_input_dtype(params["dtype_input"][4])
        res["w_dkv_kr_tensor"] = torch_tensor_list[4]
        if res["w_dkv_kr_dtype"] == "hifloat8":
            res["w_dkv_kr_tensor"] = decode_hif8_uint8_tensor_to_float32(
                res["w_dkv_kr_tensor"], over_mode=False
            )
        res["gamma_cq_shape"] = params["shape_input"][5]
        res["gamma_cq_dtype"] = trans_input_dtype(params["dtype_input"][5])
        res["gamma_cq_tensor"] = torch_tensor_list[5]
        res["gamma_ckv_shape"] = params["shape_input"][6]
        res["gamma_ckv_dtype"] = trans_input_dtype(params["dtype_input"][6])
        res["gamma_ckv_tensor"] = torch_tensor_list[6]
        res["sin_shape"] = params["shape_input"][7]
        res["sin_dtype"] = trans_input_dtype(params["dtype_input"][7])
        res["sin_tensor"] = torch_tensor_list[7]
        res["cos_shape"] = params["shape_input"][8]
        res["cos_dtype"] = trans_input_dtype(params["dtype_input"][8])
        res["cos_tensor"] = torch_tensor_list[8]
        res["cache_index_shape"] = params["shape_input"][9]
        res["cache_index_dtype"] = trans_input_dtype(params["dtype_input"][9])
        res["cache_index_tensor"] = torch_tensor_list[9]
        res["kv_cache_shape"] = params["shape_input"][10]
        res["kv_cache_dtype"] = trans_input_dtype(params["dtype_input"][10])
        res["kv_cache_tensor"] = torch_tensor_list[10]
        if res["kv_cache_dtype"] == "hifloat8":
            res["kv_cache_tensor"] = decode_hif8_uint8_tensor_to_float32(
                res["kv_cache_tensor"], over_mode=False
            )
        res["kr_cache_shape"] = params["shape_input"][11]
        res["kr_cache_dtype"] = trans_input_dtype(params["dtype_input"][11])
        res["kr_cache_tensor"] = torch_tensor_list[11]
        res["epsilon_cq"] = params["epsilon_cq"]
        res["epsilon_ckv"] = params["epsilon_ckv"]
        res["block_size"] = params["block_size"]
        res["weight_quant_mode"] = params["weight_quant_mode"]
        res["kv_quant_mode"] = params["kv_quant_mode"]
        res["query_quant_mode"] = params["query_quant_mode"]
        res["ckvkr_repo_mode"] = params["ckvkr_repo_mode"]
        res["quant_scale_repo_mode"] = params["quant_scale_repo_mode"]
        res["tile_size"] = params["tile_size"]
        res["qc_qr_scale"] = params["qc_qr_scale"]
        res["kc_scale"] = params["kc_scale"]
        res["out_q_shape"] = params["shape_output"][0]
        res["out_q_dtype"] = trans_input_dtype(params["dtype_output"][0])
        res["out_qrope_shape"] = params["shape_output"][1]
        res["out_qrope_dtype"] = trans_input_dtype(params["dtype_output"][1])
        res["out_kvcache_shape"] = params["shape_output"][2]
        res["out_kvcache_dtype"] = trans_input_dtype(params["dtype_output"][2])
        res["out_krcache_shape"] = params["shape_output"][3]
        res["out_krcache_dtype"] = trans_input_dtype(params["dtype_output"][3])
        output_opt_idx = 4
        res["enable_quant_output"] = False
        if res["flaglist"][21]:
            res["enable_quant_output"] = True
            res["out_deqq_shape"] = params["shape_output"][output_opt_idx]
            res["out_deqq_dtype"] = trans_input_dtype(
                params["dtype_output"][output_opt_idx]
            )
            output_opt_idx += 1
        if res["flaglist"][22]:
            res["out_qnorm_shape"] = params["shape_output"][output_opt_idx]
            res["out_qnorm_dtype"] = trans_input_dtype(
                params["dtype_output"][output_opt_idx]
            )
            output_opt_idx += 1
        if res["flaglist"][23]:
            res["out_deq_qnorm_shape"] = params["shape_output"][output_opt_idx]
            res["out_deq_qnorm_dtype"] = trans_input_dtype(
                params["dtype_output"][output_opt_idx]
            )
            output_opt_idx += 1
        res["deq_scale_x_shape"] = params["shape_input"][12]
        res["deq_scale_x_dtype"] = trans_input_dtype(params["dtype_input"][12])
        res["deq_scale_x_tensor"] = torch_tensor_list[12]
        res["deq_scale_w_dq_shape"] = params["shape_input"][13]
        res["deq_scale_w_dq_dtype"] = trans_input_dtype(params["dtype_input"][13])
        res["deq_scale_w_dq_tensor"] = torch_tensor_list[13]
        res["deq_scale_w_uqqr_shape"] = params["shape_input"][14]
        res["deq_scale_w_uqqr_dtype"] = trans_input_dtype(params["dtype_input"][14])
        res["deq_scale_w_uqqr_tensor"] = torch_tensor_list[14]
        res["deq_scale_w_dkvkr_shape"] = params["shape_input"][15]
        res["deq_scale_w_dkvkr_dtype"] = trans_input_dtype(params["dtype_input"][15])
        res["deq_scale_w_dkvkr_tensor"] = torch_tensor_list[15]
        res["quant_scale_ckv_shape"] = params["shape_input"][16]
        res["quant_scale_ckv_dtype"] = trans_input_dtype(params["dtype_input"][16])
        res["quant_scale_ckv_tensor"] = torch_tensor_list[16]
        res["quant_scale_ckr_shape"] = params["shape_input"][17]
        res["quant_scale_ckr_dtype"] = trans_input_dtype(params["dtype_input"][17])
        res["quant_scale_ckr_tensor"] = torch_tensor_list[17]
        res["smo_scale_cq_tensor"] = None
        if res["flaglist"][20]:
            res["smo_scale_cq_shape"] = params["shape_input"][18]
            res["smo_scale_cq_dtype"] = trans_input_dtype(params["dtype_input"][18])
            res["smo_scale_cq_tensor"] = torch_tensor_list[18]
        res["k_nope_clip_alpha_shape"] = params["shape_input"][20]
        res["k_nope_clip_alpha_dtype"] = trans_input_dtype(params["dtype_input"][20])
        res["k_nope_clip_alpha_tensor"] = torch_tensor_list[20]
        res["pa_flag"] = False
        position = res["cache_mode"].find("PA")
        if position != -1:
            res["pa_flag"] = True
        res["t_flag"] = False
        if res["x_tensor"].ndim == 2:
            res["t_flag"] = True
        return res

    def _as_bool(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(x)
        s = str(x).strip().lower()
        return s in ("1", "true", "yes", "y", "on")

    def cal_mlaprolog(mla_param):
        B = mla_param["B"]
        S1 = mla_param["S1"]
        S2 = mla_param["S2"]
        D = mla_param["D"]
        Dr = mla_param["Dr"]
        N1 = mla_param["N1"]
        N2 = mla_param["N2"]
        He = mla_param["He"]
        Hckv = mla_param["Hckv"]
        Hcq = mla_param["Hcq"]
        T = mla_param["T"]
        BlockSize = mla_param["block_size"]
        index_table = mla_param["cache_index_tensor"]
        pass
        x_dtype = mla_param["x_dtype"]
        uqqr_dtype = mla_param["w_uq_qr_dtype"]
        uk_dtype = mla_param["w_uk_dtype"]
        kv_dtype = mla_param["kv_cache_dtype"]
        kr_dtype = mla_param["kr_cache_dtype"]
        token_x = mla_param["x_tensor"]
        cos = mla_param["cos_tensor"]
        sin = mla_param["sin_tensor"]
        qnorm_flag = mla_param["qnorm_flag"]
        deq_scale_q_nope = None
        deq_scale_qcqr = None
        out_qnorm = None
        out_deq_qnorm = None
        full_quant_mode_config = _get_full_quant_mode_config(
            mla_param["weight_quant_mode"]
        )
        mode2_family_full_quant = mla_param["weight_quant_mode"] in (
            WEIGHT_QUANT_MODE_FULL_INT8,
            WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
            WEIGHT_QUANT_MODE_FULL_HIF8,
        )
        enable_quant_output = (
            True
            if mla_param["query_quant_mode"] == 1
            and mla_param["weight_quant_mode"]
            in (
                WEIGHT_QUANT_MODE_FULL_INT8,
                WEIGHT_QUANT_MODE_MXFP8_FULL,
                WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
                WEIGHT_QUANT_MODE_FULL_HIF8,
            )
            else False
        )
        quant_scale_ckv = mla_param["quant_scale_ckv_tensor"]
        actual_seq_lengths = mla_param["actual_seq_len"]
        if enable_quant_output:
            out_deqq_shape_shape = mla_param["out_deqq_shape"]
        if not mla_param["t_flag"]:
            pass
            T = B * S1
            token_x = token_x.reshape(T, He)
            cos = cos.reshape(T, Dr)
            sin = sin.reshape(T, Dr)
            seq_len = S1
        else:
            seq_len = actual_seq_lengths
        pass
        pass
        pass
        w_dq = mla_param["w_dq_tensor"]
        matmul1_dtype = torch.float32
        pass
        token_x_new = token_x
        if (
            mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_FULL_INT8
            and x_dtype == "int8"
        ):
            token_x_new = token_x_new.to(torch.int32)
            token_x = token_x.to(torch.int32)
            w_dq = w_dq.to(torch.int32)
            matmul1_dtype = torch.int32
            if mla_param["device"] == "gpu":
                token_x = token_x.to("cpu")
                w_dq = w_dq.to("cpu")
        elif (
            mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_MXFP8_FULL
            and x_dtype == "float8_e4m3fn"
        ):
            token_x_new = token_x_new.to(torch.bfloat16)
            token_x = token_x.to(torch.bfloat16)
            deq_scale_x = mla_param["deq_scale_x_tensor"]
            pass
            if not mla_param["t_flag"]:
                deq_scale_x = deq_scale_x.reshape(B * S1, He // 32)
            else:
                deq_scale_x = deq_scale_x.reshape(T, He // 32)
            deq_scale_x = trans_torch_fp8_e8m0_to_bf16(deq_scale_x)
            xs0 = deq_scale_x.shape[0]
            xs1 = deq_scale_x.shape[1]
            grp_size = 32
            for xs0_idx in range(xs0):
                for xs1_idx in range(xs1):
                    token_cur = token_x_new[
                        xs0_idx : xs0_idx + 1,
                        xs1_idx * grp_size : (xs1_idx + 1) * grp_size,
                    ]
                    scale_x = deq_scale_x[xs0_idx : xs0_idx + 1, xs1_idx : xs1_idx + 1]
                    scale_x = torch.full((1, grp_size), scale_x.item())
                    token_cur = token_cur * scale_x
                    token_x_new[
                        xs0_idx : xs0_idx + 1,
                        xs1_idx * grp_size : (xs1_idx + 1) * grp_size,
                    ] = token_cur
            w_dq = mla_param["w_dq_tensor"]
            w_dq = w_dq.to(torch.bfloat16)
            deq_scale_w_dq = mla_param["deq_scale_w_dq_tensor"]
            deq_scale_w_dq = trans_torch_fp8_e8m0_to_bf16(deq_scale_w_dq)
            dqs0 = deq_scale_w_dq.shape[0]
            dqs1 = deq_scale_w_dq.shape[1]
            for dqs0_idx in range(dqs0):
                for dqs1_idx in range(dqs1):
                    scale_w_dq = deq_scale_w_dq[
                        dqs0_idx : dqs0_idx + 1, dqs1_idx : dqs1_idx + 1
                    ]
                    w_dq[
                        dqs1_idx * grp_size : (dqs1_idx + 1) * grp_size,
                        dqs0_idx : dqs0_idx + 1,
                    ] *= scale_w_dq
            if mla_param["device"] == "gpu":
                token_x_new = token_x_new.to("cpu")
                token_x = token_x.to("cpu")
                w_dq = w_dq.to("cpu")
        x_shape = "(T,He)" if mla_param["t_flag"] else "(B*S1,He)"  # noqa: F841 -- unchanged supplied Golden, guarded MXFP8 path
        pass
        token_x_new = token_x_new.to(torch.float32)
        w_dq = w_dq.to(torch.float32)
        matmul1_res = torch.matmul(token_x_new, w_dq).to(matmul1_dtype)
        pass
        if mode2_family_full_quant:
            if mla_param["device"] == "gpu":
                matmul1_res = matmul1_res.to("cuda")
            deq_scale_x = mla_param["deq_scale_x_tensor"]
            deq_scale_w_dq = mla_param["deq_scale_w_dq_tensor"]
            matmul1_res = matmul1_res.to(torch.float32)
            for t_index in range(T):
                matmul1_res[t_index, :] = (
                    matmul1_res[t_index, :] * deq_scale_x[t_index, 0]
                )
            for h_index in range(Hcq):
                matmul1_res[:, h_index] = (
                    matmul1_res[:, h_index] * deq_scale_w_dq[0, h_index]
                )
            pass
        elif x_dtype == "bfloat16":
            pass
            matmul1_res = matmul1_res.to(torch.bfloat16).to(torch.float32)
        matmul1_res_shape = "(T,Hcq)" if mla_param["t_flag"] else "(B*S1,Hcq)"  # noqa: F841 -- unchanged supplied Golden, guarded MXFP8 path
        pass
        pass
        ep1 = float(mla_param["epsilon_cq"])
        gamma1 = mla_param["gamma_cq_tensor"]
        pass
        norm1_res = matmul1_res / torch.sqrt(
            torch.mean(matmul1_res**2, dim=-1, keepdim=True) + ep1
        )
        pass
        norm1_res *= gamma1
        qc_qr_scale = float(mla_param["qc_qr_scale"])
        norm1_res *= qc_qr_scale
        pass
        pass
        w_uq_qr = mla_param["w_uq_qr_tensor"]
        matmul2_dtype = torch.float32
        if (
            mla_param["weight_quant_mode"] == 1
            or mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_FULL_INT8
        ):
            w_uq_qr = w_uq_qr.to(torch.int32)
            matmul2_dtype = torch.int32
            smo_scale_cq = mla_param["smo_scale_cq_tensor"]
            (norm1_res, deq_scale_qcqr) = dynamic_quant_int8(norm1_res, smo_scale_cq)
            if qnorm_flag:
                if mla_param["action_type"] == "bm_output_gold":
                    out_qnorm = norm1_res.to(torch.float32)
                    out_deq_qnorm = deq_scale_qcqr.to(torch.float64)
                else:
                    out_qnorm = norm1_res
                    out_deq_qnorm = deq_scale_qcqr
            if mla_param["device"] == "gpu":
                w_uq_qr = w_uq_qr.to("cpu")
                deq_scale_qcqr = deq_scale_qcqr.to("cuda")
            pass
        elif mla_param["weight_quant_mode"] in (
            WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
            WEIGHT_QUANT_MODE_FULL_HIF8,
        ):
            w_uq_qr = w_uq_qr.to(full_quant_mode_config["input_dtype"])
            smo_scale_cq = mla_param["smo_scale_cq_tensor"]
            (norm1_res, deq_scale_qcqr) = dynamic_quant(
                mla_param,
                norm1_res,
                smo_scale_cq,
                dtype_max=full_quant_mode_config["dtype_max"],
                quant_dtype=full_quant_mode_config["input_dtype"],
            )
            if qnorm_flag:
                if mla_param["action_type"] == "bm_output_gold":
                    out_qnorm = (
                        norm1_res
                        if mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_FULL_HIF8
                        else norm1_res.to(torch.float32)
                    )
                    out_deq_qnorm = deq_scale_qcqr.to(torch.float64)
                else:
                    out_qnorm = norm1_res
                    out_deq_qnorm = deq_scale_qcqr
            pass
            pass
        elif mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_MXFP8_FULL:
            w_uq_qr = w_uq_qr.to(torch.bfloat16)
            deq_scale_uqqr = mla_param["deq_scale_w_uqqr_tensor"]
            deq_scale_uqqr = trans_torch_fp8_e8m0_to_bf16(deq_scale_uqqr)
            uqqrs0 = deq_scale_uqqr.shape[0]
            uqqrs1 = deq_scale_uqqr.shape[1]
            grp_size = 32
            for uqqrs0_idx in range(uqqrs0):
                for uqqrs1_idx in range(uqqrs1):
                    scale_uqqr = deq_scale_uqqr[
                        uqqrs0_idx : uqqrs0_idx + 1, uqqrs1_idx : uqqrs1_idx + 1
                    ]
                    w_uq_qr[
                        uqqrs1_idx * grp_size : (uqqrs1_idx + 1) * grp_size,
                        uqqrs0_idx : uqqrs0_idx + 1,
                    ] *= scale_uqqr
            norm1_res = norm1_res.to(torch.bfloat16)
            (deq_scale_qcqr_np, norm1_res_np) = dynamic_mx_quant_cq(
                norm1_res.float().numpy(), "float8_e4m3fn"
            )
            norm1_res = torch.clamp(
                torch.tensor(norm1_res_np.astype(np.float32)),
                min=-FP8_E4M3_DTYPE_MAX,
                max=FP8_E4M3_DTYPE_MAX,
            ).to(torch.float8_e4m3fn)
            deq_scale_qcqr = torch.from_numpy(deq_scale_qcqr_np)
            deq_scale_qcqr = deq_scale_qcqr.reshape(
                deq_scale_qcqr.shape[0],
                deq_scale_qcqr.shape[1] * deq_scale_qcqr.shape[2],
            )
            if qnorm_flag:
                if mla_param["action_type"] == "bm_output_gold":
                    out_qnorm = norm1_res.to(torch.float32)
                    out_deq_qnorm = deq_scale_qcqr.to(torch.uint8)
                else:
                    out_qnorm = norm1_res
                    out_deq_qnorm = deq_scale_qcqr
                    pass
            norm1_res = norm1_res.to(torch.bfloat16)
            deq_scale_qcqr = trans_torch_fp8_e8m0_to_bf16(deq_scale_qcqr)
            qcqrs0 = deq_scale_qcqr.shape[0]
            qcqrs1 = deq_scale_qcqr.shape[1]
            grp_size = 32
            for qcqrs0_idx in range(qcqrs0):
                for qcqrs1_idx in range(qcqrs1):
                    normal_cur = norm1_res[
                        qcqrs0_idx : qcqrs0_idx + 1,
                        qcqrs1_idx * grp_size : (qcqrs1_idx + 1) * grp_size,
                    ]
                    scale_qcqr = deq_scale_qcqr[
                        qcqrs0_idx : qcqrs0_idx + 1, qcqrs1_idx : qcqrs1_idx + 1
                    ]
                    scale_qcqr = torch.full((1, grp_size), scale_qcqr.item())
                    normal_cur = normal_cur * scale_qcqr
                    norm1_res[
                        qcqrs0_idx : qcqrs0_idx + 1,
                        qcqrs1_idx * grp_size : (qcqrs1_idx + 1) * grp_size,
                    ] = normal_cur
        elif mla_param["weight_quant_mode"] == 0:
            pass
            norm1_res = norm1_res.to(torch.bfloat16).to(torch.float32)
            if qnorm_flag:
                out_qnorm = norm1_res
                out_qnorm = torch.nan_to_num(out_qnorm, nan=0.0)
        pass
        if mla_param["weight_quant_mode"] in (
            0,
            WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
            WEIGHT_QUANT_MODE_MXFP8_FULL,
        ):
            norm1_res = norm1_res.to(torch.float32)
            w_uq_qr = w_uq_qr.to(torch.float32)
        elif mla_param["weight_quant_mode"] in (1, WEIGHT_QUANT_MODE_FULL_INT8):
            norm1_res = norm1_res.to(torch.int32)
            w_uq_qr = w_uq_qr.to(torch.int32)
        matmul2_res = torch.matmul(norm1_res, w_uq_qr).to(matmul2_dtype)
        pass
        matmul2_res = matmul2_res.reshape(T, N1, D + Dr)
        pass
        matmul2_res = matmul2_res.reshape(T, N1 * (D + Dr))
        if mla_param["weight_quant_mode"] in (
            1,
            WEIGHT_QUANT_MODE_FULL_INT8,
            WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
            WEIGHT_QUANT_MODE_FULL_HIF8,
        ):
            if mla_param["device"] == "gpu":
                matmul2_res = matmul2_res.to("cuda")
            deq_scale_uqqr = mla_param["deq_scale_w_uqqr_tensor"]
            matmul2_res = matmul2_res.to(torch.float32)
            for t_index in range(T):
                matmul2_res[t_index, :] = (
                    matmul2_res[t_index, :] * deq_scale_qcqr[t_index]
                )
            pass
            for nddr_index in range(matmul2_res.shape[1]):
                matmul2_res[:, nddr_index] = (
                    matmul2_res[:, nddr_index] * deq_scale_uqqr[0, nddr_index]
                )
            pass
        elif uqqr_dtype == "bfloat16":
            pass
            matmul2_res = matmul2_res.to(torch.bfloat16).to(torch.float32)
        matmul2_res = matmul2_res.reshape(T, N1, D + Dr)
        pass
        splitd1_res1 = matmul2_res[:, :, :D]
        splitd1_res2 = matmul2_res[:, :, D:]
        pass
        splitd1_res1_shape = "(T,N1,D)" if mla_param["t_flag"] else "(B*S1,N1,D)"  # noqa: F841 -- unchanged supplied Golden, guarded MXFP8 path
        splitd1_res2_shape = "(T,N1,Dr)" if mla_param["t_flag"] else "(B*S1,N1,Dr)"
        pass
        w_uk = mla_param["w_uk_tensor"]
        splitd1_res1 = splitd1_res1.transpose(0, 1)
        out1 = torch.zeros((N1, T, Hckv))
        matmul3_dtype = torch.float32
        if mla_param["weight_quant_mode"] in (
            1,
            WEIGHT_QUANT_MODE_FULL_INT8,
            WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
            WEIGHT_QUANT_MODE_FULL_HIF8,
        ):
            matmul3_dtype = torch.bfloat16
            w_uk = w_uk.to(torch.bfloat16)
            splitd1_res1 = splitd1_res1.to(torch.bfloat16)
            splitd1_res1 = torch.nan_to_num(splitd1_res1, nan=0.0)
        else:
            pass
            w_uk = w_uk.to(torch.float32)
            if uk_dtype == "bfloat16":
                splitd1_res1 = splitd1_res1.to(torch.bfloat16).to(torch.float32)
        pass
        for n1_index in range(N1):
            out1[n1_index, :, :] = torch.matmul(
                splitd1_res1[n1_index, :, :].to(torch.float32),
                w_uk[n1_index, :, :].to(torch.float32),
            ).to(matmul3_dtype)
        pass
        out1 = out1.transpose(0, 1)
        if enable_quant_output:
            if mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_MXFP8_FULL:
                out1 = out1.to(torch.bfloat16).to(torch.float32)
                (deq_scale_q_nope_np, out1_np) = dynamic_mx_quant_qn(
                    out1.numpy(), mla_param
                )
                if mla_param["action_type"] == "bm_output_gold":
                    out1 = torch.tensor(out1_np.to(torch.float32))
                    deq_scale_q_nope = torch.from_numpy(deq_scale_q_nope_np).to(
                        torch.float64
                    )
                else:
                    out1 = torch.tensor(out1_np.to(torch.float32)).to(
                        torch.float8_e4m3fn
                    )
                    deq_scale_q_nope = torch.from_numpy(deq_scale_q_nope_np)
            elif mla_param["weight_quant_mode"] in (
                WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
                WEIGHT_QUANT_MODE_FULL_HIF8,
            ):
                out1 = out1.to(torch.bfloat16).to(torch.float32)
                (out1, deq_scale_q_nope) = dynamic_quant_without_smooth_scale(
                    out1,
                    out_deqq_shape_shape,
                    dtype_max=full_quant_mode_config["dtype_max"],
                    quant_dtype=full_quant_mode_config["query_quant_dtype"],
                )
                if mla_param["action_type"] == "bm_output_gold":
                    out1 = (
                        out1
                        if mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_FULL_HIF8
                        else out1.to(torch.float32)
                    )
                    deq_scale_q_nope = deq_scale_q_nope.to(torch.float64)
            else:
                out1 = out1.to(torch.bfloat16).to(torch.float32)
                (out1, deq_scale_q_nope) = dynamic_quant_without_smooth_scale_int8(
                    out1, out_deqq_shape_shape
                )
                if mla_param["action_type"] == "bm_output_gold":
                    out1 = out1.to(torch.float32)
                    deq_scale_q_nope = deq_scale_q_nope.to(torch.float64)
                else:
                    out1 = out1.to(torch.int8)
                    deq_scale_q_nope = deq_scale_q_nope
        out1 = out1 if mla_param["t_flag"] else out1.reshape(B, S1, N1, Hckv)
        out1_shape = "(T,N1,Hckv)" if mla_param["t_flag"] else "(B,S1,N,Hckv)"  # noqa: F841 -- unchanged supplied Golden, guarded MXFP8 path
        pass
        pass
        splitd1_res2_shape = "(T,N1,Dr)" if mla_param["t_flag"] else "(B*S1,N1,Dr)"  # noqa: F841 -- unchanged supplied Golden, guarded MXFP8 path
        cos_shape = "(T,Dr)" if mla_param["t_flag"] else "(B*S1,Dr)"  # noqa: F841 -- unchanged supplied Golden, guarded MXFP8 path
        pass
        expanded_cos = cos.unsqueeze(1).repeat(1, N1, 1)
        expanded_sin = sin.unsqueeze(1).repeat(1, N1, 1)
        q = (
            splitd1_res2.reshape(T, N1, int(Dr / 2), 2)
            .transpose(3, 2)
            .reshape(T, N1, Dr)
        )
        out2 = q * expanded_cos + rotate_half(q) * expanded_sin
        if enable_quant_output:
            out2 = out2.to(torch.bfloat16).to(torch.float32)
            out2 = dequant(out2, deq_scale_q_nope, quant_scale_ckv)
        out2 = out2 if mla_param["t_flag"] else out2.reshape(B, S1, N1, Dr)
        if out2.dtype == torch.float8_e4m3fn:
            out2 = torch.from_numpy(np.nan_to_num(out2.numpy(), nan=0.0))
        else:
            out2 = torch.nan_to_num(out2, nan=0.0)
        out2_shape = "(T,N1,Dr)" if mla_param["t_flag"] else "(B,S1,N1,Dr)"  # noqa: F841 -- unchanged supplied Golden, guarded MXFP8 path
        pass
        pass
        w_kv_kr = mla_param["w_dkv_kr_tensor"]
        pass
        matmul4_dtype = torch.float32
        if mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_FULL_INT8:
            w_kv_kr = w_kv_kr.to(torch.int32)
            matmul4_dtype = torch.int32
            if mla_param["device"] == "gpu":
                w_kv_kr = w_kv_kr.to("cpu")
        elif mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_MXFP8_FULL:
            w_kv_kr = w_kv_kr.to(torch.bfloat16)
            deq_scale_dkvkr = mla_param["deq_scale_w_dkvkr_tensor"]
            deq_scale_dkvkr = trans_torch_fp8_e8m0_to_bf16(deq_scale_dkvkr)
            dkvkrs0 = deq_scale_dkvkr.shape[0]
            dkvkrs1 = deq_scale_dkvkr.shape[1]
            for dkvkrs0_idx in range(dkvkrs0):
                for dkvkrs1_idx in range(dkvkrs1):
                    scale_dkvkr = deq_scale_dkvkr[
                        dkvkrs0_idx : dkvkrs0_idx + 1, dkvkrs1_idx : dkvkrs1_idx + 1
                    ]
                    w_kv_kr[
                        dkvkrs1_idx * grp_size : (dkvkrs1_idx + 1) * grp_size,
                        dkvkrs0_idx : dkvkrs0_idx + 1,
                    ] *= scale_dkvkr
        pass
        pass
        pass
        matmul4_res = torch.matmul(
            token_x_new.to(torch.float32), w_kv_kr.to(torch.float32)
        ).to(matmul4_dtype)
        pass
        if mode2_family_full_quant:
            if mla_param["device"] == "gpu":
                matmul4_res = matmul4_res.to("cuda")
            deq_scale_x = mla_param["deq_scale_x_tensor"]
            deq_scale_dkvkr = mla_param["deq_scale_w_dkvkr_tensor"]
            matmul4_res = matmul4_res.to(torch.float32)
            for t_index in range(T):
                matmul4_res[t_index, :] = (
                    matmul4_res[t_index, :] * deq_scale_x[t_index, 0]
                )
            for h_index in range(Hckv + Dr):
                matmul4_res[:, h_index] = (
                    matmul4_res[:, h_index] * deq_scale_dkvkr[0, h_index]
                )
            pass
        elif x_dtype == "bfloat16":
            pass
            matmul4_res = matmul4_res.to(torch.bfloat16).to(torch.float32)
        matmul4_res_shape = "(T,Hckv+Dr)" if mla_param["t_flag"] else "(B*S1,Hckv+Dr)"  # noqa: F841 -- unchanged supplied Golden, guarded MXFP8 path
        pass
        splitd2_res1 = matmul4_res[:, :Hckv]
        splitd2_res2 = matmul4_res[:, Hckv:]
        pass
        splitd2_res1_shape = "(T,Hckv)" if mla_param["t_flag"] else "(B*S1,Hckv)"  # noqa: F841 -- unchanged supplied Golden, guarded MXFP8 path
        splitd2_res2_shape = "(T,Dr)" if mla_param["t_flag"] else "(B*S1,Dr)"  # noqa: F841 -- unchanged supplied Golden, guarded MXFP8 path
        pass
        pass
        k = splitd2_res2.reshape(T, 1, int(Dr / 2), 2).transpose(3, 2).reshape(T, Dr)
        rotary2_res = (k * cos + rotate_half(k) * sin).to(torch.bfloat16)
        pass
        pass
        pass
        quant_scale_ckr = mla_param["quant_scale_ckr_tensor"]
        if mla_param["weight_quant_mode"] == 1 and mla_param["kv_quant_mode"] == 2:
            rotary2_res = quant(rotary2_res.to(torch.float32), quant_scale_ckr)
            rotary2_res = rotary2_res.to(torch.bfloat16)
            pass
        rotary2_res = torch.nan_to_num(
            rotary2_res,
            nan=0.0,
            posinf=torch.finfo(rotary2_res.dtype).max,
            neginf=torch.finfo(rotary2_res.dtype).min,
        )
        pass
        ep2 = float(mla_param["epsilon_ckv"])
        gamma2 = mla_param["gamma_ckv_tensor"]
        pass
        norm2_res = splitd2_res1 / torch.sqrt(
            torch.mean(splitd2_res1**2, dim=-1, keepdim=True) + ep2
        )
        norm2_res *= gamma2
        kc_scale = float(mla_param["kc_scale"])
        norm2_res *= kc_scale
        pass
        Dtile = Hckv
        if mla_param["kv_quant_mode"] == 1 or mla_param["kv_quant_mode"] == 2:
            if mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_MXFP8_FULL:
                quant_scale_ckv = mla_param["quant_scale_ckv_tensor"]
                norm2_res = quant_ckv_per_tensor(
                    norm2_res.numpy(), quant_scale_ckv.numpy()
                )
                if mla_param["action_type"] == "bm_output_gold":
                    norm2_res = torch.tensor(norm2_res.to(torch.float32))
                else:
                    norm2_res = torch.tensor(norm2_res.to(torch.float32)).to(
                        torch.float8_e4m3fn
                    )
                pass
            elif mla_param["weight_quant_mode"] in (
                WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
                WEIGHT_QUANT_MODE_FULL_HIF8,
            ):
                pass
                quant_scale_ckv = mla_param["quant_scale_ckv_tensor"]
                norm2_res = quant_with_scale(
                    mla_param,
                    norm2_res,
                    quant_scale_ckv,
                    quant_dtype=full_quant_mode_config["input_dtype"],
                    dtype_max=full_quant_mode_config["dtype_max"],
                )
                pass
                pass
                pass
                if mla_param["action_type"] == "bm_output_gold":
                    norm2_res = torch.tensor(norm2_res.to(torch.float32))
            else:
                pass
                pass
                quant_scale_ckv = mla_param["quant_scale_ckv_tensor"]
                norm2_res = quant(norm2_res, quant_scale_ckv)
                if mla_param["action_type"] == "bm_output_gold":
                    norm2_res = torch.tensor(norm2_res.to(torch.float32))
                else:
                    norm2_res = torch.tensor(norm2_res.to(torch.float32)).to(torch.int8)
                norm2_res = torch.nan_to_num(norm2_res, nan=0.0)
                pass
                pass
        elif mla_param["kv_quant_mode"] == 3:
            if mla_param["weight_quant_mode"] in (
                WEIGHT_QUANT_MODE_MXFP8_FULL,
                WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
            ):
                norm2_res = norm2_res.reshape(
                    T * Hckv // mla_param["tile_size"], mla_param["tile_size"]
                )
                eps = 0.0001
                amax = torch.max(torch.abs(norm2_res), dim=-1, keepdim=True)[0]
                amax = torch.clamp(amax, min=eps)
                (norm2_res, deq_scale_ckv) = dynamic_quant_ckv_with_amax_fp8e4m3(
                    norm2_res, amax
                )
                deq_scale_ckv = deq_scale_ckv.reshape(T, -1)
                norm2_res = norm2_res.reshape(T, Hckv).to(torch.float8_e4m3fn)
                rotary2_res_bf16 = rotary2_res.to(torch.bfloat16)
                norm2_res = _merge_quant_cache_repo_payload(
                    mla_param,
                    norm2_res,
                    torch.float8_e4m3fn,
                    rotary_bf16=rotary2_res_bf16
                    if mla_param["ckvkr_repo_mode"] == 1
                    else None,
                    deq_scale=deq_scale_ckv
                    if mla_param["quant_scale_repo_mode"] == 1
                    else None,
                )
                if mla_param["ckvkr_repo_mode"] == 1:
                    Dtile = Dtile + Dr * 2
                if mla_param["quant_scale_repo_mode"] == 1:
                    Dtile = Dtile + Hckv // mla_param["tile_size"] * 4
                pass
                pass
            elif mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_FULL_HIF8:
                norm2_res = norm2_res.reshape(
                    T, Hckv // mla_param["tile_size"], mla_param["tile_size"]
                )
                eps = 1e-08
                amax = torch.max(torch.abs(norm2_res), dim=-1, keepdim=True)[0]
                amax = torch.clamp(amax, min=eps)
                clip_res = torch.clamp(norm2_res, min=-amax, max=amax)
                (norm2_res, deq_scale_ckv) = dynamic_quant_ckv_with_amax(
                    clip_res,
                    amax,
                    quant_dtype=full_quant_mode_config["input_dtype"],
                    dtype_max=full_quant_mode_config["dtype_max"],
                )
                deq_scale_ckv = deq_scale_ckv.reshape(T, -1)
                norm2_res = norm2_res.reshape(T, Hckv).to(
                    full_quant_mode_config["input_dtype"]
                )
                rotary2_res_bf16 = rotary2_res.to(torch.bfloat16)
                norm2_res = _merge_quant_cache_repo_payload(
                    mla_param,
                    norm2_res,
                    full_quant_mode_config["input_dtype"],
                    rotary_bf16=rotary2_res_bf16
                    if mla_param["ckvkr_repo_mode"] == 1
                    else None,
                    deq_scale=deq_scale_ckv
                    if mla_param["quant_scale_repo_mode"] == 1
                    else None,
                )
                if mla_param["ckvkr_repo_mode"] == 1:
                    Dtile = Dtile + Dr * 2
                if mla_param["quant_scale_repo_mode"] == 1:
                    Dtile = Dtile + Hckv // mla_param["tile_size"] * 4
                if mla_param["action_type"] == "bm_output_gold":
                    norm2_res = norm2_res.to(torch.float32)
                    deq_scale_ckv = deq_scale_ckv.to(torch.float64)
                pass
            else:
                norm2_res = norm2_res.reshape(
                    T, Hckv // mla_param["tile_size"], mla_param["tile_size"]
                )
                eps = 1e-08
                amax = torch.max(torch.abs(norm2_res), dim=-1, keepdim=True)[0]
                amax = (
                    torch.clamp(amax, min=eps) * mla_param["k_nope_clip_alpha_tensor"]
                )
                clip_res = torch.clamp(norm2_res, min=-amax, max=amax)
                (norm2_res, deq_scale_ckv) = dynamic_quant_ckv_with_amax(clip_res, amax)
                deq_scale_ckv = deq_scale_ckv.reshape(T, -1)
                norm2_res = norm2_res.reshape(T, Hckv).to(torch.int8)
                rotary2_res_bf16 = rotary2_res.to(torch.bfloat16)
                pass
                if mla_param["ckvkr_repo_mode"] == 1:
                    norm2_res = torch.cat(
                        (norm2_res, rotary2_res_bf16.view(torch.int8)), axis=-1
                    )
                    Dtile = Dtile + Dr * 2
                    rotary2_res_1 = rotary2_res_bf16.view(torch.bfloat16)  # noqa: F841 -- unchanged supplied Golden, guarded MXFP8 path
                    pass
                if mla_param["quant_scale_repo_mode"] == 1:
                    norm2_res = torch.cat(
                        (norm2_res, deq_scale_ckv.view(torch.int8)), axis=-1
                    )
                    Dtile = Dtile + Hckv // mla_param["tile_size"] * 4
                pass
        else:
            norm2_res = torch.nan_to_num(norm2_res, nan=0.0)
        kv_cache = copy.deepcopy(mla_param["kv_cache_tensor"])
        if kv_cache.numel() == 0:
            out3 = kv_cache
            out4 = copy.deepcopy(mla_param["kr_cache_tensor"])
            return (out1, out2, out3, out4, deq_scale_q_nope, out_qnorm, out_deq_qnorm)
        out3_shape = kv_cache.shape
        out3_info = "(B,B,N2,Hckv)"
        if not mla_param["pa_flag"]:
            kv_cache = kv_cache.transpose(2, 1)
            out3_info = "(B,N2,S2,Hckv)"  # noqa: F841 -- unchanged supplied Golden, guarded MXFP8 path
        if kv_dtype in ["int8", "float8_e4m3fn", "hifloat8"]:
            scatter_size = 32
        elif kv_dtype == "bf16":
            scatter_size = 16
            kv_cache = kv_cache.to(torch.bfloat16)
        else:
            scatter_size = 16
        pass
        if mla_param["cache_mode"] == "PA_BLK_NZ":
            kv_cache = scatter_pa_blk_nz(
                kv_cache, norm2_res, index_table, seq_len, B, scatter_size
            )
        elif mla_param["cache_mode"] == "PA_BLK_BSND":
            kv_cache = scatter_pa_blk_bsnd(kv_cache, norm2_res, index_table, seq_len, B)
        elif mla_param["cache_mode"] == "PA_NZ":
            kv_cache = scatter_pa_nz(
                kv_cache, norm2_res, index_table.reshape(T), scatter_size
            )
        elif mla_param["cache_mode"] == "PA_BSND":
            B_num = out3_shape[0]
            kv_cache = kv_cache.reshape(B_num * BlockSize, N2, Dtile)
            for i in range(T):
                for j in range(N2):
                    pass
                    pass
                    pass
                    pass
                    kv_cache[index_table.reshape(T)[i], j, :] = norm2_res[i, :]
        else:
            kv_cache = kv_cache.reshape(B * S2, N2, Dtile)
            for i in range(T):
                for j in range(N2):
                    kv_cache[i, j, :] = norm2_res[i, :]
        out3 = kv_cache.reshape(out3_shape)
        pass
        pass
        pass
        pass
        k = splitd2_res2.reshape(T, 1, int(Dr / 2), 2).transpose(3, 2).reshape(T, Dr)
        rotary2_res = (k * cos + rotate_half(k) * sin).to(torch.bfloat16)
        pass
        quant_scale_ckr = mla_param["quant_scale_ckr_tensor"]
        if mla_param["weight_quant_mode"] == 1 and mla_param["kv_quant_mode"] == 2:
            rotary2_res = quant(rotary2_res.to(torch.float32), quant_scale_ckr)
            rotary2_res = rotary2_res.to(torch.bfloat16)
            pass
        rotary2_res = torch.nan_to_num(
            rotary2_res,
            nan=0.0,
            posinf=torch.finfo(rotary2_res.dtype).max,
            neginf=torch.finfo(rotary2_res.dtype).min,
        )
        if mla_param["ckvkr_repo_mode"] == 1:
            out4 = copy.deepcopy(mla_param["kr_cache_tensor"])
        else:
            kr_cache = copy.deepcopy(mla_param["kr_cache_tensor"])
            out4_shape = kr_cache.shape
            out4_info = "(B,B,N2,Dr)"
            if not mla_param["pa_flag"]:
                kr_cache = kr_cache.transpose(2, 1)
                out4_info = "(B,N2,S2,Dr)"  # noqa: F841 -- unchanged supplied Golden, guarded MXFP8 path
            if kr_dtype in ["int8", "float8_e4m3fn", "hifloat8"]:
                scatter_size = 32
            elif kr_dtype == "bfloat16":
                scatter_size = 16
                kr_cache = kr_cache.to(torch.bfloat16)
            else:
                scatter_size = 16
            pass
            if mla_param["cache_mode"] == "PA_BLK_NZ":
                kr_cache = scatter_pa_blk_nz(
                    kr_cache, rotary2_res, index_table, seq_len, B, scatter_size
                )
            elif mla_param["cache_mode"] == "PA_BLK_BSND":
                kr_cache = scatter_pa_blk_bsnd(
                    kr_cache, rotary2_res, index_table, seq_len, B
                )
            elif mla_param["cache_mode"] == "PA_NZ":
                kr_cache = scatter_pa_nz(
                    kr_cache, rotary2_res, index_table.reshape(T), scatter_size
                )
            elif mla_param["cache_mode"] == "PA_BSND":
                B_num = out4_shape[0]
                kr_cache = kr_cache.reshape(B_num * BlockSize, N2, Dr)
                for i in range(T):
                    for j in range(N2):
                        kr_cache[index_table.reshape(T)[i], j, :] = rotary2_res[i, :]
            else:
                kr_cache = kr_cache.reshape(B * S2, N2, Dr)
                for i in range(T):
                    for j in range(N2):
                        kr_cache[i, j, :] = rotary2_res[i, :]
            out4 = kr_cache.reshape(out4_shape)
            pass
            pass
        pass
        pass
        pass
        return (out1, out2, out3, out4, deq_scale_q_nope, out_qnorm, out_deq_qnorm)

    return dict(
        get_param=get_param,
        cal_mlaprolog=cal_mlaprolog,
        dynamic_mx_quant_qn=dynamic_mx_quant_qn,
        dequant=dequant,
    )


_SUPPLIED_GOLDEN_TEMPLATES = {
    "high": {
        "action_type": "bm_output_gold",
        "actual_seq_value": [],
        "atol": 2.5e-05,
        "attr_1": "cache_mode",
        "attr_10": "hcq",
        "attr_11": "hckv",
        "attr_12": "t",
        "attr_13": "block_size",
        "attr_14": "weight_quant_mode",
        "attr_15": "kv_quant_mode",
        "attr_16": "query_quant_mode",
        "attr_17": "ckvkr_repo_mode",
        "attr_18": "quant_scale_repo_mode",
        "attr_19": "tile_size",
        "attr_2": "b",
        "attr_20": "qc_qr_scale",
        "attr_21": "kc_scale",
        "attr_22": "epsilon_cq",
        "attr_23": "epsilon_ckv",
        "attr_24": "flaglist",
        "attr_25": "mla_flag",
        "attr_26": "qnorm_flag",
        "attr_27": "actual_seq_value",
        "attr_3": "n1",
        "attr_4": "n2",
        "attr_5": "d",
        "attr_6": "dr",
        "attr_7": "s1",
        "attr_8": "s2",
        "attr_9": "he",
        "b": 128,
        "block_size": 128,
        "bm_cmp_std": {
            "bf16": {
                "avg_re_rtol": 2.0,
                "max_re_rtol": 10.0,
                "rmse_rtol": 2.0,
                "small_value": 0.0009765625,
                "small_value_atol": 1.52587890625e-05,
            },
            "fp16": {
                "avg_re_rtol": 2.0,
                "max_re_rtol": 10.0,
                "rmse_rtol": 2.0,
                "small_value": 0.0009765625,
                "small_value_atol": 1.52587890625e-05,
            },
            "fp32": {
                "avg_re_rtol": 2.0,
                "max_re_rtol": 10.0,
                "rmse_rtol": 2.0,
                "small_value": 0.0009765625,
                "small_value_atol": 1.52587890625e-05,
            },
        },
        "cache_mode": "PA_BSND",
        "case_name": "aclnnMlaPrologV3WeightNz_decode_mxfp8_quant_054",
        "ckvkr_repo_mode": 0,
        "d": 128,
        "device_type": "npu",
        "diff_thd": 0.005,
        "distribution_input": "uniform",
        "dr": 64,
        "dtype_input": [
            "fp32",
            "fp32",
            "fp32",
            "fp32",
            "fp32",
            "fp32",
            "fp32",
            "fp32",
            "fp32",
            "int64",
            "fp32",
            "fp32",
            "fp8_e8m0",
            "fp8_e8m0",
            "fp8_e8m0",
            "fp8_e8m0",
            "fp64",
            "fp32",
            "fp32",
            "fp32",
            "fp32",
        ],
        "dtype_output": ["fp32", "fp32", "fp32", "fp32", "fp64"],
        "epsilon_ckv": 1e-05,
        "epsilon_cq": 1e-05,
        "flaglist": [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
        ],
        "format_input": [
            "ND",
            "FRACTAL_NZ",
            "FRACTAL_NZ",
            "ND",
            "FRACTAL_NZ",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
        ],
        "framework_type": "aclnn",
        "hckv": 512,
        "hcq": 1536,
        "he": 7168,
        "kc_scale": 1,
        "kv_quant_mode": 1,
        "max_diff_thd": 10.0,
        "mla_flag": "no",
        "n1": 16,
        "n2": 1,
        "op_name": "aclnnMlaPrologV3WeightNz",
        "pct_thd": 0.005,
        "precision_method": 0,
        "precision_method_list": [12, 1, 10, 1, 1],
        "precision_param_list": [
            {
                "atol": 0.0001,
                "diff_thd": 0.005,
                "max_diff_thd": 10.0,
                "pct_thd": 0.005,
                "rtol": 0.0078125,
            },
            {
                "atol": 0.0001,
                "diff_thd": 0.005,
                "max_diff_thd": 10.0,
                "pct_thd": 0.005,
                "rtol": 0.0078125,
            },
            {
                "atol": 0.0001,
                "diff_thd": 0.005,
                "max_diff_thd": 10.0,
                "pct_thd": 0.005,
                "rtol": 0.0078125,
            },
            {
                "atol": 0.0001,
                "diff_thd": 0.005,
                "max_diff_thd": 10.0,
                "pct_thd": 0.005,
                "rtol": 0.0078125,
            },
            {
                "atol": 2.5e-05,
                "diff_thd": 0.005,
                "max_diff_thd": 10.0,
                "pct_thd": 0.005,
                "rtol": 0.005,
            },
        ],
        "qc_qr_scale": 1,
        "qnorm_flag": False,
        "quant_scale_repo_mode": 0,
        "query_quant_mode": 1,
        "range_input": [
            [-100, 100],
            [-100, 100],
            [-100, 100],
            [-100, 100],
            [-100, 100],
            [-1000, 1000],
            [-1000, 1000],
            [-1, 1],
            [-1, 1],
            [-10.0, 10.0],
            [-100, 100],
            [-100, 100],
            [120, 128],
            [120, 128],
            [120, 128],
            [120, 128],
            [0, 1.27],
            ["null", "null"],
            ["null", "null"],
            [1, 1],
            ["null", "null"],
        ],
        "red_range": {
            "bf16": "0.001/0.002/0.005/0.01",
            "fp16": "0.001/0.002/0.005/0.01",
            "fp32": "0.000001/0.00001/0.0001/0.0005",
            "hf32": "0.001/0.002/0.005/0.01",
        },
        "required_actual_seq_value": 1,
        "required_b": 1,
        "required_block_size": 1,
        "required_cache_mode": 1,
        "required_ckvkr_repo_mode": 1,
        "required_d": 1,
        "required_dr": 1,
        "required_epsilon_ckv": 1,
        "required_epsilon_cq": 1,
        "required_flaglist": 1,
        "required_hckv": 1,
        "required_hcq": 1,
        "required_he": 1,
        "required_kc_scale": 1,
        "required_kv_quant_mode": 1,
        "required_mla_flag": 1,
        "required_n1": 1,
        "required_n2": 1,
        "required_qc_qr_scale": 1,
        "required_qnorm_flag": 1,
        "required_quant_scale_repo_mode": 1,
        "required_query_quant_mode": 1,
        "required_s1": 1,
        "required_s2": 1,
        "required_t": 1,
        "required_tile_size": 1,
        "required_weight_quant_mode": 1,
        "rtol": 0.005,
        "s1": 1,
        "s2": 1,
        "shape_input": [
            [128, 7168],
            [7168, 1536],
            [1536, 3072],
            [16, 128, 512],
            [7168, 576],
            [1536],
            [512],
            [128, 64],
            [128, 64],
            [128],
            [1, 128, 1, 512],
            [1, 128, 1, 64],
            [128, 224],
            [1536, 224],
            [3072, 48],
            [576, 224],
            [1],
            [1],
            [1],
            [1],
            [1],
        ],
        "shape_output": [
            [128, 16, 512],
            [128, 16, 64],
            [1, 128, 1, 512],
            [1, 128, 1, 64],
            [128, 16, 1],
        ],
        "t": 128,
        "test_type": "training",
        "tile_size": 128,
        "type_input": [
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
        ],
        "weight_quant_mode": 3,
    },
    "ordinary": {
        "actual_seq_value": [],
        "atol": 2.5e-05,
        "attr_1": "cache_mode",
        "attr_10": "hcq",
        "attr_11": "hckv",
        "attr_12": "t",
        "attr_13": "block_size",
        "attr_14": "weight_quant_mode",
        "attr_15": "kv_quant_mode",
        "attr_16": "query_quant_mode",
        "attr_17": "ckvkr_repo_mode",
        "attr_18": "quant_scale_repo_mode",
        "attr_19": "tile_size",
        "attr_2": "b",
        "attr_20": "qc_qr_scale",
        "attr_21": "kc_scale",
        "attr_22": "epsilon_cq",
        "attr_23": "epsilon_ckv",
        "attr_24": "flaglist",
        "attr_25": "mla_flag",
        "attr_26": "qnorm_flag",
        "attr_27": "actual_seq_value",
        "attr_3": "n1",
        "attr_4": "n2",
        "attr_5": "d",
        "attr_6": "dr",
        "attr_7": "s1",
        "attr_8": "s2",
        "attr_9": "he",
        "b": 128,
        "block_size": 128,
        "bm_cmp_std": {
            "bf16": {
                "avg_re_rtol": 2.0,
                "max_re_rtol": 10.0,
                "rmse_rtol": 2.0,
                "small_value": 0.0009765625,
                "small_value_atol": 1.52587890625e-05,
            },
            "fp16": {
                "avg_re_rtol": 2.0,
                "max_re_rtol": 10.0,
                "rmse_rtol": 2.0,
                "small_value": 0.0009765625,
                "small_value_atol": 1.52587890625e-05,
            },
            "fp32": {
                "avg_re_rtol": 2.0,
                "max_re_rtol": 10.0,
                "rmse_rtol": 2.0,
                "small_value": 0.0009765625,
                "small_value_atol": 1.52587890625e-05,
            },
        },
        "cache_mode": "PA_BSND",
        "case_name": "aclnnMlaPrologV3WeightNz_decode_mxfp8_quant_050",
        "ckvkr_repo_mode": 0,
        "d": 128,
        "diff_thd": 0.005,
        "distribution_input": "uniform",
        "dr": 64,
        "dtype_input": [
            "float8_e4m3fn",
            "float8_e4m3fn",
            "float8_e4m3fn",
            "bf16",
            "float8_e4m3fn",
            "bf16",
            "bf16",
            "bf16",
            "bf16",
            "int64",
            "float8_e4m3fn",
            "bf16",
            "float8_e8m0",
            "float8_e8m0",
            "float8_e8m0",
            "float8_e8m0",
            "fp32",
            "fp16",
            "fp16",
            "fp16",
            "fp16",
        ],
        "dtype_output": ["float8_e4m3fn", "bf16", "float8_e4m3fn", "bf16", "fp32"],
        "epsilon_ckv": 1e-05,
        "epsilon_cq": 1e-05,
        "flaglist": [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
        ],
        "format_input": [
            "ND",
            "FRACTAL_NZ",
            "FRACTAL_NZ",
            "ND",
            "FRACTAL_NZ",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
            "ND",
        ],
        "hckv": 512,
        "hcq": 1536,
        "he": 7168,
        "kc_scale": 1,
        "kv_quant_mode": 1,
        "max_diff_thd": 10.0,
        "mla_flag": "no",
        "n1": 1,
        "n2": 1,
        "pct_thd": 0.005,
        "precision_method_list": [10, 1, 10, 1, 1],
        "precision_param_list": [
            {
                "atol": 0.0001,
                "diff_thd": 0.005,
                "max_diff_thd": 10.0,
                "pct_thd": 0.005,
                "rtol": 0.0078125,
            },
            {
                "atol": 0.0001,
                "diff_thd": 0.005,
                "max_diff_thd": 10.0,
                "pct_thd": 0.005,
                "rtol": 0.0078125,
            },
            {
                "atol": 0.0001,
                "diff_thd": 0.005,
                "max_diff_thd": 10.0,
                "pct_thd": 0.005,
                "rtol": 0.0078125,
            },
            {
                "atol": 0.0001,
                "diff_thd": 0.005,
                "max_diff_thd": 10.0,
                "pct_thd": 0.005,
                "rtol": 0.0078125,
            },
            {
                "atol": 2.5e-05,
                "diff_thd": 0.005,
                "max_diff_thd": 10.0,
                "pct_thd": 0.005,
                "rtol": 0.005,
            },
        ],
        "qc_qr_scale": 1,
        "qnorm_flag": False,
        "quant_scale_repo_mode": 0,
        "query_quant_mode": 1,
        "range_input": [
            [-100, 100],
            [-100, 100],
            [-100, 100],
            [-100, 100],
            [-100, 100],
            [-1000, 1000],
            [-1000, 1000],
            [-1, 1],
            [-1, 1],
            [0, 1],
            [-100, 100],
            [-100, 100],
            [120, 128],
            [120, 128],
            [120, 128],
            [120, 128],
            [0, 1.27],
            ["null", "null"],
            ["null", "null"],
            [1, 1],
            ["null", "null"],
        ],
        "red_range": {
            "bf16": "0.001/0.002/0.005/0.01",
            "fp16": "0.001/0.002/0.005/0.01",
            "fp32": "0.000001/0.00001/0.0001/0.0005",
            "hf32": "0.001/0.002/0.005/0.01",
        },
        "required_actual_seq_value": 1,
        "required_b": 1,
        "required_block_size": 1,
        "required_cache_mode": 1,
        "required_ckvkr_repo_mode": 1,
        "required_d": 1,
        "required_dr": 1,
        "required_epsilon_ckv": 1,
        "required_epsilon_cq": 1,
        "required_flaglist": 1,
        "required_hckv": 1,
        "required_hcq": 1,
        "required_he": 1,
        "required_kc_scale": 1,
        "required_kv_quant_mode": 1,
        "required_mla_flag": 1,
        "required_n1": 1,
        "required_n2": 1,
        "required_qc_qr_scale": 1,
        "required_qnorm_flag": 1,
        "required_quant_scale_repo_mode": 1,
        "required_query_quant_mode": 1,
        "required_s1": 1,
        "required_s2": 1,
        "required_t": 1,
        "required_tile_size": 1,
        "required_weight_quant_mode": 1,
        "rtol": 0.005,
        "s1": 1,
        "s2": 1,
        "shape_input": [
            [128, 7168],
            [7168, 1536],
            [1536, 192],
            [1, 128, 512],
            [7168, 576],
            [1536],
            [512],
            [128, 64],
            [128, 64],
            [128],
            [1, 128, 1, 512],
            [1, 128, 1, 64],
            [128, 224],
            [1536, 224],
            [192, 48],
            [576, 224],
            [1],
            [1],
            [1],
            [1],
            [1],
        ],
        "shape_output": [
            [128, 1, 512],
            [128, 1, 64],
            [1, 128, 1, 512],
            [1, 128, 1, 64],
            [128, 1, 1],
        ],
        "t": 128,
        "tile_size": 128,
        "type_input": [
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
            "tensor",
        ],
        "weight_quant_mode": 3,
    },
}


def supplied_q_references(args, attrs):
    """Run the supplied ordinary/high models, then quantize homogeneous Q jointly.

    Preserve the supplied high branch, including its QA descale dtype branch.
    This is source-aligned acceptance, not a claim that casting ordinary Golden
    intermediates to FP64 creates an independent high-precision reference.
    """
    import copy

    if not (
        attrs["w_qr_quant_mode"] == 1
        and attrs["qkv_quant_mode"] == 2
        and attrs["rope_mode"] == 1
        and args[14].shape[-1] == 64
    ):
        return None
    namespace = _supplied_golden_functions()
    t, dim = args[0].shape
    n, dn, _ = args[4].shape
    ql = args[9].numel()
    weights = []
    scales = []
    for i in (1, 2, 3):
        w = args[i]
        nz = w.ndim == 4
        w = _unpack_nz_weight(w.view(torch.uint8)).view(w.dtype) if nz else w
        weights.append(w.view(torch.uint8).t().contiguous().view(w.dtype))
        scale = args[i + 5]
        raw = scale.view(torch.uint8)
        if raw.ndim == 4:
            raw = raw.permute(0, 2, 1, 3).contiguous()
        scales.append(raw.reshape(w.shape[0], -1))
    cache = cache_to_logical(args[11], attrs["cache_mode"])
    tensors = [
        args[0],
        *weights[:2],
        args[4],
        weights[2],
        args[9],
        args[10],
        args[14],
        args[15],
        args[12],
        cache[..., :512].contiguous(),
        cache[..., 512:].float().bfloat16(),
        args[5].view(torch.uint8),
        *scales,
        args[13],
        torch.ones(1),
        torch.ones(1),
        None,
        torch.ones(1),
    ]
    results = []
    for high in (False, True):
        p = copy.deepcopy(_SUPPLIED_GOLDEN_TEMPLATES["high" if high else "ordinary"])
        p.update(
            b=1,
            s1=t,
            s2=cache.shape[0] * cache.shape[1],
            n1=n,
            n2=1,
            d=dn,
            dr=64,
            he=dim,
            hcq=ql,
            hckv=512,
            t=t,
            block_size=cache.shape[1],
            epsilon_cq=attrs["norm_eps"],
            epsilon_ckv=attrs["norm_eps"],
            qc_qr_scale=attrs["qr_scale"],
            kc_scale=attrs["kvr_scale"],
            kv_quant_mode=1,
            query_quant_mode=1,
            action_type="bm_output_gold" if high else "bm_output",
        )
        values = [x.clone() if x is not None else None for x in tensors]
        if high:
            types = {"fp32": torch.float32, "fp64": torch.float64, "int64": torch.int64}
            for i, dtype in enumerate(p["dtype_input"]):
                if values[i] is not None and dtype in types:
                    values[i] = values[i].to(types[dtype])
        p["shape_input"] = [list(x.shape) if x is not None else [1] for x in values]
        p["shape_output"] = [
            [t, n, 512],
            [t, n, 64],
            list(tensors[10].shape),
            list(tensors[11].shape),
            [t, n, 1],
        ]
        params = namespace["get_param"](values, p)
        observed = {}

        def capture(frame, event, arg):
            if event != "call":
                return
            if frame.f_code is namespace["dynamic_mx_quant_qn"].__code__:
                observed["nope"] = torch.from_numpy(frame.f_locals["x"].copy())
                observed["params"] = dict(frame.f_locals["mla_param"])
            elif frame.f_code is namespace["dequant"].__code__:
                observed["rope"] = frame.f_locals["inputs"].detach().clone()

        previous = sys.getprofile()
        try:
            sys.setprofile(capture)
            namespace["cal_mlaprolog"](params)
        finally:
            sys.setprofile(previous)
        joint = torch.cat((observed["nope"], observed["rope"]), dim=-1).float()
        _, q = namespace["dynamic_mx_quant_qn"](joint.numpy(), observed["params"])
        if tuple(q.shape) != (t, n, 576):
            raise ValueError("supplied Q reference shape mismatch")
        results.append(q)
    return results


def supplied_q_three_way(actual, args, attrs):
    references = supplied_q_references(args, attrs)
    if references is None:
        return None
    ordinary, high = references
    parts = []
    for name, sl in [
        ("Q", slice(None)),
        ("Q NoPE", slice(None, 512)),
        ("Q RoPE", slice(512, None)),
    ]:
        parts.append(
            dict(
                name=name,
                npu=actual[0][..., sl],
                cpu=ordinary[..., sl],
                cpu_gold=high[..., sl].float(),
                dtype="float8_e4m3fn",
                reference="Supplied bm_output / bm_output_gold; joint homogeneous Q quantization",
                gating=name == "Q",
            )
        )
    report = evaluate_segments(parts)
    report["reference_source"] = "supplied independent bm_output_gold branch"
    report["limitation"] = (
        "The supplied high dtype branch skips QA descale; source-aligned acceptance."
    )
    return report
