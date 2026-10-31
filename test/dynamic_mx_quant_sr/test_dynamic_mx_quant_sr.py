# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Contract, precision, and performance entry points for the mixed SR sample.

Run pytest for regressions, or execute this file with --shape and --output-dir
for CPU golden validation and optional msprof measurements. Timed tasks use
preallocated buffers and exclude allocation, compilation, and layout copies.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16
import cannbotdsl
import torch_npu  # noqa: F401  # Register torch.npu even when backend autoload is disabled.
from cannbotdsl import make_layout, dtypes
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang import host, kernel, vf
from cannbotdsl.ops import reg
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.tensor import MemLoc, tile_slice

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _samples_path import load_sample


# Independent CPU reference.
# CPU byte reference for the DynamicMxQuantV3 stochastic DSL sample.
#
# Quantization semantics originate from ops-nn commit
# 22c0528222199af3a276237c1291080329cd86a8, quant/dynamic_mx_quant/tests/assets/golden.py.
# RNG uses16384 fixed virtual streams and the CANN32-bit-word offset convention.
# Logical group g=i//16 maps to stream g%16384 and draw g//16384; the low64
# counter is offset//4+draw. FP32 selects the eight-field map for supported
# tail/transpose layouts with N>=2097152; otherwise four fields. BF16 tail and eligible width16 layouts use
# the adjacent-byte map. Selection is independent of UB, core count and eager/graph execution.
# Generic fields use H/rotate8(W).low16/L/rotate8(W).high16. Optimized
# tails use dtype-specific eight-field mappings; full fields are correlated.
# This reference deliberately
# implements FP32/BF16 -> E4M3FN/E5M2, any axis, blocksize=32,
# round_mode="stochastic", scale_alg=0/1 and scale_alg=1 max_low_bound. It has no DSL or
# device dependencies. Both outputs contain raw uint8 encoding bytes.

# Independent reference constants: do not import device implementation values.
# IEEE storage widths and FP8/E8M0 encodings.
_GOLDEN_BITS_PER_BYTE = 8
_GOLDEN_UINT16_BITS = 16
_GOLDEN_UINT32_BITS = 32
_GOLDEN_UINT64_BITS = 64
_GOLDEN_UINT32_MASK = (1 << _GOLDEN_UINT32_BITS) - 1
_GOLDEN_UINT16_MASK = (1 << _GOLDEN_UINT16_BITS) - 1
_GOLDEN_UINT8_MASK = 0xFF
_GOLDEN_FP32_FRACTION_BITS = 23
_GOLDEN_FP32_FRACTION_MASK = 0x7FFFFF
_GOLDEN_FP32_IMPLICIT_BIT = 0x800000
_GOLDEN_FP32_HALF_SIGNIFICAND_BIT = 0x400000
_GOLDEN_FP32_SIGNIFICAND_BITS = _GOLDEN_FP32_FRACTION_BITS + 1
_GOLDEN_FP32_EXPONENT_BIAS = 127
_GOLDEN_FP32_SUBNORMAL_EXPONENT_BIAS = _GOLDEN_FP32_EXPONENT_BIAS - 1
_GOLDEN_FP32_EXPONENT_MASK = 0xFF
_GOLDEN_FP32_INF_BITS = 0x7F800000
_GOLDEN_UINT32_HIGH_BYTE_SHIFT = _GOLDEN_UINT32_BITS - _GOLDEN_BITS_PER_BYTE
_GOLDEN_FP8_SIGN_MASK = 0x80
_GOLDEN_FP8_NAN_CODE = 0x7F
_GOLDEN_E4M3_FRACTION_BITS = 3
_GOLDEN_E5M2_FRACTION_BITS = 2
_GOLDEN_E4M3_EXPONENT_BIAS = 7
_GOLDEN_E5M2_EXPONENT_BIAS = 15
_GOLDEN_E4M3_MAX_FINITE_CODE = 0x7E
_GOLDEN_E5M2_MAX_FINITE_CODE = 0x7B
_GOLDEN_E4M3_SCALE_EXPONENT_SHIFT = 8
_GOLDEN_E5M2_SCALE_EXPONENT_SHIFT = 15
_GOLDEN_E4M3_INV_MAX_FP32_BITS = 0x3B124925
_GOLDEN_E5M2_INV_MAX_FP32_BITS = 0x37924925
_GOLDEN_BF16_FRACTION_BITS = 7
_GOLDEN_BF16_NAN_BITS = 0x7F81
_GOLDEN_BF16_MIN_RECIPROCAL_BITS = 0x0040
_GOLDEN_E8M0_INVALID_CODE = 0xFF
_GOLDEN_E8M0_RECIPROCAL_EXPONENT_SUM = 254
_GOLDEN_MX_BLOCK_ELEMENTS = 32
_GOLDEN_MX_BLOCK_ROUND_UP_ELEMENTS = _GOLDEN_MX_BLOCK_ELEMENTS - 1
_GOLDEN_SCALES_PER_PAIR = 2
_GOLDEN_SCALE_PAIR_ELEMENTS = _GOLDEN_MX_BLOCK_ELEMENTS * _GOLDEN_SCALES_PER_PAIR
_GOLDEN_SCALE_PAIR_ROUND_UP_ELEMENTS = _GOLDEN_SCALE_PAIR_ELEMENTS - 1

# Random123 Philox4x32-10 and the independently specified NPU field ABI.
_GOLDEN_PHILOX_ROUNDS = 10
_GOLDEN_PHILOX_M0 = 0xD2511F53
_GOLDEN_PHILOX_M1 = 0xCD9E8D57
_GOLDEN_PHILOX_W0 = 0x9E3779B9
_GOLDEN_PHILOX_W1 = 0xBB67AE85
_GOLDEN_PHILOX_WORDS_PER_BLOCK = 4
_GOLDEN_PHILOX_WORD_ROUND_UP = _GOLDEN_PHILOX_WORDS_PER_BLOCK - 1
_GOLDEN_RNG_VIRTUAL_STREAMS = 16384
_GOLDEN_RNG_GROUP_ELEMENTS = 16
_GOLDEN_RNG_FIELDS_PER_WORD = 4
_GOLDEN_RNG_FIELD_ROUND_UP = _GOLDEN_RNG_FIELDS_PER_WORD - 1
_GOLDEN_RNG_EIGHT_GROUP_ELEMENTS = 32
_GOLDEN_RNG_EIGHT_GROUP_ROUND_UP = _GOLDEN_RNG_EIGHT_GROUP_ELEMENTS - 1
_GOLDEN_RNG_EIGHT_FIELDS_PER_WORD = 8
_GOLDEN_FIELDS_PER_ADJACENT_BYTE_PAIR = 2
_GOLDEN_ADJACENT_WINDOW_BASE_BITS = _GOLDEN_UINT16_BITS - _GOLDEN_BF16_FRACTION_BITS

# Layout admission and mapping boundaries, in input elements. These remain
# independent of UB capacity/AIV quota, matching the reference RNG contract.
_GOLDEN_PACKED_K_ALIGNMENT_ELEMENTS = 256
_GOLDEN_FP32_EIGHT_MIN_ELEMENTS = 2 * 1024 * 1024
_GOLDEN_STRIDE2_SUFFIX_WIDTH = 2
_GOLDEN_STRIDE2_MIN_QUANT_LENGTH = 512
_GOLDEN_STRIDE2_ALIGNMENT = 64
_GOLDEN_WIDE8_SUFFIX_WIDTH = 8
_GOLDEN_WIDE16_SUFFIX_WIDTH = 16
_GOLDEN_WIDE32_SUFFIX_WIDTH = 32
_GOLDEN_WIDE64_SUFFIX_WIDTH = 64
_GOLDEN_WIDE128_SUFFIX_WIDTH = 128
_GOLDEN_WIDE_ALIGNMENT = 64
_GOLDEN_BF16_WIDE8_MAX_ELEMENTS = 128 * 1024
_GOLDEN_WIDE16_MAX_ELEMENTS = 2 * 1024 * 1024
# Bound the vectorized integer oracle's temporary arrays, in input elements.
_GOLDEN_REFERENCE_CHUNK_LOG2 = 18

_GOLDEN_FP8_SR_PARAMS = {
    "float8_e4m3fn": (
        _GOLDEN_E4M3_EXPONENT_BIAS,
        _GOLDEN_E4M3_FRACTION_BITS,
        _GOLDEN_E4M3_MAX_FINITE_CODE,
    ),
    "float8_e5m2": (
        _GOLDEN_E5M2_EXPONENT_BIAS,
        _GOLDEN_E5M2_FRACTION_BITS,
        _GOLDEN_E5M2_MAX_FINITE_CODE,
    ),
}


def _golden_philox4x32_10(c0, c1, c2, c3, k0=0, k1=0):
    """Philox4x32-10 words, with uint32 wrap after every key update."""
    c0, c1, c2, c3, k0, k1 = (
        np.asarray(value, dtype=np.uint32) for value in (c0, c1, c2, c3, k0, k1)
    )
    for _ in range(_GOLDEN_PHILOX_ROUNDS):
        p0 = c0.astype(np.uint64) * np.uint64(_GOLDEN_PHILOX_M0)
        p1 = c2.astype(np.uint64) * np.uint64(_GOLDEN_PHILOX_M1)
        c0, c1, c2, c3 = (
            (p1 >> np.uint64(_GOLDEN_UINT32_BITS)).astype(np.uint32) ^ c1 ^ k0,
            p1.astype(np.uint32),
            (p0 >> np.uint64(_GOLDEN_UINT32_BITS)).astype(np.uint32) ^ c3 ^ k1,
            p0.astype(np.uint32),
        )
        k0 = (k0.astype(np.uint64) + np.uint64(_GOLDEN_PHILOX_W0)).astype(np.uint32)
        k1 = (k1.astype(np.uint64) + np.uint64(_GOLDEN_PHILOX_W1)).astype(np.uint32)
    return (c0, c1, c2, c3)


def _golden_philox_blocks(groups, *, quant_length, seed=0, block_offset=0):
    """Independent NPU linear counter map, word-order (3,2,0,1), returning (..., 4) uint32 words."""
    uint64_max = 2**_GOLDEN_UINT64_BITS - 1
    for name, value in (
        ("seed", seed),
        ("block_offset", block_offset),
        ("quant_length", quant_length),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be a Python int")
        if not 0 <= value <= uint64_max:
            raise ValueError(f"{name} must be uint64")
    if quant_length == 0 or quant_length % _GOLDEN_RNG_GROUP_ELEMENTS:
        raise ValueError("quant_length must be a positive multiple of 16")
    if (
        block_offset % _GOLDEN_PHILOX_WORDS_PER_BLOCK
        or block_offset > uint64_max - _GOLDEN_PHILOX_WORDS_PER_BLOCK
    ):
        raise ValueError(
            "block_offset must be four-aligned with a nonwrapping reservation"
        )
    if isinstance(groups, np.ndarray) and groups.dtype.kind in "iu":
        if groups.dtype.kind == "i" and np.any(groups < 0):
            raise ValueError("groups must be uint64")
    else:
        groups = np.asarray(groups, dtype=object)
        for group in groups.flat:
            if isinstance(group, (bool, np.bool_)) or not isinstance(
                group, (int, np.integer)
            ):
                raise TypeError(
                    "groups must contain integers, not bool or converted scalars"
                )
            if not 0 <= int(group) <= uint64_max:
                raise ValueError("groups must be uint64")
    groups = np.asarray(groups, dtype=np.uint64)
    subsequence = groups % np.uint64(_GOLDEN_RNG_VIRTUAL_STREAMS)
    draws = groups // np.uint64(_GOLDEN_RNG_VIRTUAL_STREAMS)
    counter = np.uint64(block_offset // _GOLDEN_PHILOX_WORDS_PER_BLOCK) + draws
    words = _golden_philox4x32_10(
        (subsequence >> np.uint64(_GOLDEN_UINT32_BITS)).astype(np.uint32),
        subsequence.astype(np.uint32),
        counter.astype(np.uint32),
        (counter >> np.uint64(_GOLDEN_UINT32_BITS)).astype(np.uint32),
        seed & _GOLDEN_UINT32_MASK,
        seed >> _GOLDEN_UINT32_BITS,
    )
    return np.stack(words, axis=-1)


def _golden_philox_word_stream(num_words, *, quant_length, seed=0, block_offset=0):
    """Four consecutive words per mapped logical group, with runtime state."""
    groups = np.arange(
        (num_words + _GOLDEN_PHILOX_WORD_ROUND_UP) // _GOLDEN_PHILOX_WORDS_PER_BLOCK,
        dtype=np.uint64,
    )
    return _golden_philox_blocks(
        groups, quant_length=quant_length, seed=seed, block_offset=block_offset
    ).reshape(-1)[:num_words]


def _golden_random_bits(numel, *, quant_length, seed=0, block_offset=0):
    """ABI5 fields H, rotate8(W).low16, L, rotate8(W).high16.

    Each field is uniform16 for a uniform Philox32 word. The four top8
    fields use disjoint bytes; the full16 fields are related, not independent.
    Counter mapping and word consumption are unchanged from ABI4.
    """
    words = _golden_philox_word_stream(
        (numel + _GOLDEN_RNG_FIELD_ROUND_UP) // _GOLDEN_RNG_FIELDS_PER_WORD,
        quant_length=quant_length,
        seed=seed,
        block_offset=block_offset,
    )
    rotate = words << np.uint32(_GOLDEN_BITS_PER_BYTE) | words >> np.uint32(
        _GOLDEN_UINT32_HIGH_BYTE_SHIFT
    )
    return np.stack(
        (
            words >> np.uint32(_GOLDEN_UINT16_BITS),
            rotate & np.uint32(_GOLDEN_UINT16_MASK),
            words & np.uint32(_GOLDEN_UINT16_MASK),
            rotate >> np.uint32(_GOLDEN_UINT16_BITS),
        ),
        axis=-1,
    ).reshape(-1)[:numel]


def _golden_uses_fp32_eight_mapping(shape, axis):
    if int(np.prod(shape)) < _GOLDEN_FP32_EIGHT_MIN_ELEMENTS:
        return False
    axis %= len(shape)
    k = shape[axis]
    if k % _GOLDEN_PACKED_K_ALIGNMENT_ELEMENTS:
        return False
    if axis == len(shape) - 1:
        return True
    width = int(np.prod(shape[axis + 1 :]))
    n = int(np.prod(shape))
    if (
        width == _GOLDEN_STRIDE2_SUFFIX_WIDTH
        and k >= _GOLDEN_STRIDE2_MIN_QUANT_LENGTH
        and (k % _GOLDEN_STRIDE2_ALIGNMENT == 0)
    ):
        return False
    if width == _GOLDEN_WIDE8_SUFFIX_WIDTH and k >= _GOLDEN_PACKED_K_ALIGNMENT_ELEMENTS:
        return False
    if (
        width == _GOLDEN_WIDE16_SUFFIX_WIDTH
        and k >= _GOLDEN_PACKED_K_ALIGNMENT_ELEMENTS
        and (n <= _GOLDEN_WIDE16_MAX_ELEMENTS)
    ):
        return False
    if (
        width
        in (
            _GOLDEN_WIDE32_SUFFIX_WIDTH,
            _GOLDEN_WIDE64_SUFFIX_WIDTH,
            _GOLDEN_WIDE128_SUFFIX_WIDTH,
        )
        and k >= _GOLDEN_WIDE_ALIGNMENT
        and (k % _GOLDEN_WIDE_ALIGNMENT == 0)
    ):
        return False
    return True


def _golden_random_bits_fp32_eight(
    numel, *, quant_length, seed=0, block_offset=0, dst_dtype=None
):
    words = _golden_philox_blocks(
        np.arange(
            (numel + _GOLDEN_RNG_EIGHT_GROUP_ROUND_UP)
            // _GOLDEN_RNG_EIGHT_GROUP_ELEMENTS,
            dtype=np.uint64,
        ),
        quant_length=quant_length,
        seed=seed,
        block_offset=block_offset,
    ).reshape(-1)
    ids = np.arange(numel, dtype=np.uint64)
    pairs = np.asarray(
        [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 0), (0, 3)],
        dtype=np.uint32,
    )
    word = words[(ids // _GOLDEN_RNG_EIGHT_FIELDS_PER_WORD).astype(np.int64)]
    pair = pairs[(ids % _GOLDEN_RNG_EIGHT_FIELDS_PER_WORD).astype(np.int64)]
    return word >> pair[:, 0] * np.uint32(_GOLDEN_BITS_PER_BYTE) & np.uint32(
        _GOLDEN_UINT8_MASK
    ) | (
        word >> pair[:, 1] * np.uint32(_GOLDEN_BITS_PER_BYTE)
        & np.uint32(_GOLDEN_UINT8_MASK)
    ) << np.uint32(_GOLDEN_BITS_PER_BYTE)


def _golden_uses_nibble_mapping(shape, axis, dtype_name, dst_dtype):
    if dtype_name == "float32":
        return "fp32_eight" if _golden_uses_fp32_eight_mapping(shape, axis) else False
    if dtype_name != "bfloat16" or dst_dtype not in ("float8_e4m3fn", "float8_e5m2"):
        return False
    axis %= len(shape)
    k = shape[axis]
    if k % _GOLDEN_PACKED_K_ALIGNMENT_ELEMENTS:
        return False
    if axis == len(shape) - 1:
        return True
    width = int(np.prod(shape[axis + 1 :]))
    n = int(np.prod(shape))
    if (
        width == _GOLDEN_STRIDE2_SUFFIX_WIDTH
        and k >= _GOLDEN_STRIDE2_MIN_QUANT_LENGTH
        and (k % _GOLDEN_STRIDE2_ALIGNMENT == 0)
    ):
        return False
    if (
        width == _GOLDEN_WIDE8_SUFFIX_WIDTH
        and k >= _GOLDEN_PACKED_K_ALIGNMENT_ELEMENTS
        and (k % _GOLDEN_PACKED_K_ALIGNMENT_ELEMENTS == 0)
        and (n <= _GOLDEN_BF16_WIDE8_MAX_ELEMENTS)
    ):
        return False
    # BF16 width16 uses adjacent-byte fields for direct and transpose layouts.
    if (
        width
        in (
            _GOLDEN_WIDE32_SUFFIX_WIDTH,
            _GOLDEN_WIDE64_SUFFIX_WIDTH,
            _GOLDEN_WIDE128_SUFFIX_WIDTH,
        )
        and k >= _GOLDEN_WIDE_ALIGNMENT
        and (k % _GOLDEN_WIDE_ALIGNMENT == 0)
    ):
        return False
    return True


def _golden_random_bits_nibble(
    numel, *, quant_length, seed=0, block_offset=0, dst_dtype="float8_e4m3fn"
):
    words = _golden_philox_blocks(
        np.arange(
            (numel + _GOLDEN_RNG_EIGHT_GROUP_ROUND_UP)
            // _GOLDEN_RNG_EIGHT_GROUP_ELEMENTS,
            dtype=np.uint64,
        ),
        quant_length=quant_length,
        seed=seed,
        block_offset=block_offset,
    ).reshape(-1)
    ids = np.arange(numel, dtype=np.uint64)
    f = (ids % _GOLDEN_RNG_EIGHT_FIELDS_PER_WORD).astype(np.uint32)
    mantissa = np.uint32(_GOLDEN_FP8_SR_PARAMS[dst_dtype][1])
    normal_start = f // np.uint32(_GOLDEN_FIELDS_PER_ADJACENT_BYTE_PAIR) * np.uint32(
        _GOLDEN_BITS_PER_BYTE
    ) + (np.uint32(1) + mantissa) * (
        f % np.uint32(_GOLDEN_FIELDS_PER_ADJACENT_BYTE_PAIR)
    )
    shift = (
        normal_start
        + np.uint32(_GOLDEN_UINT32_BITS)
        - np.uint32(_GOLDEN_ADJACENT_WINDOW_BASE_BITS)
        - mantissa
    ) % np.uint32(_GOLDEN_UINT32_BITS)
    w = words[(ids // _GOLDEN_RNG_EIGHT_FIELDS_PER_WORD).astype(np.int64)]
    return (
        w >> shift
        | w << (np.uint32(_GOLDEN_UINT32_BITS) - shift) % np.uint32(_GOLDEN_UINT32_BITS)
    ) & np.uint32(_GOLDEN_UINT16_MASK)


def _golden_sr_scalar(bits, r16, dst_dtype):
    """Integer scalar Algorithm 19; no host FP8 cast or floating rounding."""
    bias, mantissa_bits, max_code = _GOLDEN_FP8_SR_PARAMS[dst_dtype]
    sign = bits >> _GOLDEN_UINT32_HIGH_BYTE_SHIFT & _GOLDEN_FP8_SIGN_MASK
    exponent = bits >> _GOLDEN_FP32_FRACTION_BITS & _GOLDEN_FP32_EXPONENT_MASK
    fraction = bits & _GOLDEN_FP32_FRACTION_MASK
    if exponent == _GOLDEN_FP32_EXPONENT_MASK and fraction:
        return _GOLDEN_FP8_NAN_CODE
    source_exp = (
        exponent - _GOLDEN_FP32_EXPONENT_BIAS
        if exponent
        else -_GOLDEN_FP32_SUBNORMAL_EXPONENT_BIAS
    )
    significand = fraction | _GOLDEN_FP32_IMPLICIT_BIT if exponent else fraction
    emin = 1 - bias
    normal = source_exp >= emin
    if not normal:
        significand >>= min(emin - source_exp, _GOLDEN_FP32_SIGNIFICAND_BITS)
    cut = _GOLDEN_FP32_FRACTION_BITS - mantissa_bits
    base = significand >> cut
    discarded = significand >> cut - _GOLDEN_UINT16_BITS & _GOLDEN_UINT16_MASK
    carry = discarded + (r16 & _GOLDEN_UINT16_MASK) >> _GOLDEN_UINT16_BITS
    if normal:
        base = source_exp + bias << mantissa_bits | base & (1 << mantissa_bits) - 1
    return sign | min(base + carry, max_code)


def _golden_sr_cast_fp32_to_fp8(values, r16, dst_dtype):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    random = np.broadcast_to(np.asarray(r16, dtype=np.uint32), bits.shape)
    result = np.fromiter(
        (
            _golden_sr_scalar(int(value), int(rand), dst_dtype)
            for value, rand in zip(bits.flat, random.flat)
        ),
        dtype=np.uint8,
        count=bits.size,
    )
    return result.reshape(bits.shape)


def _golden_sr_cast_fp32_to_fp8_vectorized(values, r16, dst_dtype):
    """Chunked integer Algorithm 19, checked against the scalar oracle.

    The independent scalar routine above remains available to tests. This
    follows its exponent/significand/discarded-bit algorithm, not the device's
    Q16 implementation. Temporaries are bounded to 262144 input elements.
    """
    bias, mantissa_bits, max_code = _GOLDEN_FP8_SR_PARAMS[dst_dtype]
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    random = np.broadcast_to(np.asarray(r16, dtype=np.uint32), bits.shape)
    result = np.empty(bits.size, dtype=np.uint8)
    flat_bits, flat_random = (bits.reshape(-1), random.reshape(-1))
    for start in range(0, bits.size, 1 << _GOLDEN_REFERENCE_CHUNK_LOG2):
        stop = min(start + (1 << _GOLDEN_REFERENCE_CHUNK_LOG2), bits.size)
        word, field = (flat_bits[start:stop], flat_random[start:stop])
        exponent = (
            word >> np.uint32(_GOLDEN_FP32_FRACTION_BITS)
            & np.uint32(_GOLDEN_FP32_EXPONENT_MASK)
        ).astype(np.int32)
        fraction = word & np.uint32(_GOLDEN_FP32_FRACTION_MASK)
        source_exp = np.where(
            exponent != 0,
            exponent - _GOLDEN_FP32_EXPONENT_BIAS,
            -_GOLDEN_FP32_SUBNORMAL_EXPONENT_BIAS,
        )
        significand = fraction | np.where(
            exponent != 0, np.uint32(_GOLDEN_FP32_IMPLICIT_BIT), np.uint32(0)
        )
        emin = 1 - bias
        normal = source_exp >= emin
        significand >>= np.clip(
            emin - source_exp, 0, _GOLDEN_FP32_SIGNIFICAND_BITS
        ).astype(np.uint32)
        cut = _GOLDEN_FP32_FRACTION_BITS - mantissa_bits
        base = significand >> np.uint32(cut)
        discarded = significand >> np.uint32(cut - _GOLDEN_UINT16_BITS) & np.uint32(
            _GOLDEN_UINT16_MASK
        )
        carry = discarded + (field & np.uint32(_GOLDEN_UINT16_MASK)) >> np.uint32(
            _GOLDEN_UINT16_BITS
        )
        normal_base = (source_exp + bias << mantissa_bits).astype(np.uint32)
        normal_base |= base & np.uint32((1 << mantissa_bits) - 1)
        base = np.where(normal, normal_base, base)
        packed = np.minimum(base + carry, np.uint32(max_code))
        packed |= word >> np.uint32(_GOLDEN_UINT32_HIGH_BYTE_SHIFT) & np.uint32(
            _GOLDEN_FP8_SIGN_MASK
        )
        result[start:stop] = np.where(
            (exponent == _GOLDEN_FP32_EXPONENT_MASK) & (fraction != 0),
            np.uint32(_GOLDEN_FP8_NAN_CODE),
            packed,
        ).astype(np.uint8)
    return result.reshape(bits.shape)


def _golden_mx_scale_alg0_kernel(abs_max_f32, dst_dtype):
    """ASC ComputeScaleOcp, including the reciprocal-zero underflow rule."""
    shift = {
        "float8_e4m3fn": _GOLDEN_E4M3_SCALE_EXPONENT_SHIFT,
        "float8_e5m2": _GOLDEN_E5M2_SCALE_EXPONENT_SHIFT,
    }[dst_dtype]
    exponent = (
        np.ascontiguousarray(abs_max_f32, dtype=np.float32).view(np.uint32)
        >> _GOLDEN_FP32_FRACTION_BITS
    )
    shared = np.maximum(exponent, shift) - shift
    reciprocal = (
        _GOLDEN_E8M0_RECIPROCAL_EXPONENT_SUM - shared << _GOLDEN_BF16_FRACTION_BITS
    ).astype(np.uint16)
    reciprocal = np.where(shared == 0, 0, reciprocal)
    reciprocal = np.where(
        shared == _GOLDEN_E8M0_RECIPROCAL_EXPONENT_SUM,
        _GOLDEN_BF16_MIN_RECIPROCAL_BITS,
        reciprocal,
    )
    reciprocal = np.where(
        exponent == _GOLDEN_FP32_EXPONENT_MASK, _GOLDEN_BF16_NAN_BITS, reciprocal
    ).astype(np.uint16)
    code = np.where(
        exponent == _GOLDEN_FP32_EXPONENT_MASK, _GOLDEN_E8M0_INVALID_CODE, shared
    ).astype(np.uint8)
    return (code, reciprocal)


def _golden_mx_scale_alg1_kernel(abs_max_f32, dst_dtype):
    """ComputeScaleCeilAlg: return E8M0 bytes and BF16 reciprocal bits."""
    inv_bits = {
        "float8_e4m3fn": _GOLDEN_E4M3_INV_MAX_FP32_BITS,
        "float8_e5m2": _GOLDEN_E5M2_INV_MAX_FP32_BITS,
    }[dst_dtype]
    inv_max = np.uint32(inv_bits).view(np.float32)
    maxima = np.ascontiguousarray(abs_max_f32, dtype=np.float32)
    bits = maxima.view(np.uint32)
    finite, zero = (bits < np.uint32(_GOLDEN_FP32_INF_BITS), bits == 0)
    with np.errstate(invalid="ignore", under="ignore"):
        product = (maxima * inv_max).astype(np.float32).view(np.uint32)
    exponent, fraction = (
        product >> _GOLDEN_FP32_FRACTION_BITS,
        product & np.uint32(_GOLDEN_FP32_FRACTION_MASK),
    )
    ceil_up = (exponent > 0) & (exponent < _GOLDEN_E8M0_RECIPROCAL_EXPONENT_SUM) & (
        fraction > 0
    ) | (exponent == 0) & (fraction > np.uint32(_GOLDEN_FP32_HALF_SIGNIFICAND_BIT))
    scale = np.where(ceil_up, exponent + 1, exponent)
    scale = np.where(finite, scale, np.uint32(_GOLDEN_E8M0_INVALID_CODE))
    scale = np.where(zero, np.uint32(0), scale)
    reciprocal = np.uint32(_GOLDEN_E8M0_RECIPROCAL_EXPONENT_SUM) - scale << np.uint32(
        _GOLDEN_BF16_FRACTION_BITS
    )
    reciprocal = np.where(finite, reciprocal, np.uint32(_GOLDEN_BF16_NAN_BITS))
    reciprocal = np.where(zero, np.uint32(0), reciprocal)
    return (
        scale.astype(np.uint8),
        (reciprocal & np.uint32(_GOLDEN_UINT16_MASK)).astype(np.uint16),
    )


def dynamic_mx_quant_golden(
    x,
    dst_dtype="float8_e4m3fn",
    scale_alg=1,
    max_low_bound=0.0,
    axis=-1,
    *,
    seed=0,
    block_offset=0,
    _use_nibble=None,
):
    """Return raw FP8 and interleaved E8M0 bytes for the selected axis."""
    if not isinstance(x, np.ndarray) or x.dtype.name not in ("float32", "bfloat16"):
        raise TypeError("x must be a NumPy FP32 or ml_dtypes BF16 array")
    if dst_dtype not in _GOLDEN_FP8_SR_PARAMS:
        raise ValueError("only float8_e4m3fn and float8_e5m2 are supported")
    if x.ndim == 0 or x.size == 0:
        raise ValueError("x must be nonempty")
    if type(axis) is not int or not -x.ndim <= axis < x.ndim:
        raise ValueError("axis is out of range")
    axis %= x.ndim
    if _use_nibble is None:
        _use_nibble = _golden_uses_nibble_mapping(
            x.shape, axis, x.dtype.name, dst_dtype
        )
    if axis != x.ndim - 1:
        moved = np.ascontiguousarray(np.moveaxis(x, axis, -1))
        y, scale = dynamic_mx_quant_golden(
            moved,
            dst_dtype,
            scale_alg,
            max_low_bound,
            axis=-1,
            seed=seed,
            block_offset=block_offset,
            _use_nibble=_use_nibble,
        )
        return (
            np.ascontiguousarray(np.moveaxis(y, -1, axis)),
            np.ascontiguousarray(np.moveaxis(scale, -2, axis)),
        )
    if x.shape[-1] % _GOLDEN_RNG_GROUP_ELEMENTS:
        raise ValueError("K must be a positive multiple of 16")
    k = x.shape[-1]
    groups = (k + _GOLDEN_MX_BLOCK_ROUND_UP_ELEMENTS) // _GOLDEN_MX_BLOCK_ELEMENTS
    rows = x.reshape(-1, k)
    blocks = np.pad(
        rows, ((0, 0), (0, groups * _GOLDEN_MX_BLOCK_ELEMENTS - k))
    ).reshape(-1, groups, _GOLDEN_MX_BLOCK_ELEMENTS)
    maxima = np.max(np.abs(blocks.astype(np.float32)), axis=-1)
    if scale_alg not in (0, 1):
        raise ValueError("scale_alg must be 0 or 1")
    if (
        not isinstance(max_low_bound, (int, float))
        or not np.isfinite(max_low_bound)
        or max_low_bound < 0
    ):
        raise ValueError("max_low_bound must be a finite non-negative number")
    if scale_alg != 1 and max_low_bound != 0:
        raise ValueError("max_low_bound must be zero when scale_alg != 1")
    if max_low_bound != 0:
        maxima = np.where(
            maxima == 0, maxima, np.maximum(maxima, np.float32(max_low_bound))
        )
    scale_fn = (
        _golden_mx_scale_alg0_kernel if scale_alg == 0 else _golden_mx_scale_alg1_kernel
    )
    scale, reciprocal_bits = scale_fn(maxima, dst_dtype)
    reciprocal = np.repeat(
        reciprocal_bits.view(bfloat16), _GOLDEN_MX_BLOCK_ELEMENTS, axis=-1
    )[:, :k]
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        normalized = rows.astype(np.float32) * reciprocal.astype(np.float32)
        if x.dtype.name == "bfloat16":
            normalized = normalized.astype(bfloat16).astype(np.float32)
    random_fn = (
        (
            _golden_random_bits_fp32_eight
            if _use_nibble == "fp32_eight"
            else _golden_random_bits_nibble
        )
        if _use_nibble
        else _golden_random_bits
    )
    fields = random_fn(
        x.size,
        quant_length=k,
        seed=seed,
        block_offset=block_offset,
        **{"dst_dtype": dst_dtype} if _use_nibble else {},
    ).reshape(x.shape)
    codes = _golden_sr_cast_fp32_to_fp8_vectorized(
        normalized.reshape(x.shape), fields, dst_dtype
    )
    scale = np.pad(
        scale, ((0, 0), (0, groups % _GOLDEN_SCALES_PER_PAIR)), constant_values=0
    )
    return (
        codes,
        scale.reshape(
            *x.shape[:-1],
            (k + _GOLDEN_SCALE_PAIR_ROUND_UP_ELEMENTS) // _GOLDEN_SCALE_PAIR_ELEMENTS,
            _GOLDEN_SCALES_PER_PAIR,
        ),
    )


# Benchmark isolation and clock units.
BENCHMARK_CHECKPOINT_SEED = 9817
BENCHMARK_CHECKPOINT_OFFSET_WORDS = 36
NANOSECONDS_PER_MICROSECOND = 1000
MODULE_IDENTITY_HEX_DIGITS = 20
RNG_MAX_PADDED_GROUPS = 63
RNG_STREAM_ROUND_UP_GROUPS = _GOLDEN_RNG_VIRTUAL_STREAMS - 1
mixed = load_sample("dynamic_mx_quant_sr/dynamic_mx_quant_sr.py")


# Test inputs, byte assertions and element probes.
E4M3_DST_TYPE = 24
E5M2_DST_TYPE = 23
# Element-rounding probe geometry, in bytes and uint32 lanes.
UINT32_BYTES = _GOLDEN_UINT32_BITS // _GOLDEN_BITS_PER_BYTE
SIMD_REGISTER_BYTES = 256
SIMD_U32_LANES = SIMD_REGISTER_BYTES // UINT32_BYTES
Q16_PROBE_BLOCKS = 32
# Deterministic input data and guard sentinels.
PRECISION_INPUT_SEED = 20260921
AXIS_INPUT_SEED = 20260924
RANDOM_INPUT_AMPLITUDE = 7
INPUT_GUARD_ELEMENTS = 16
INPUT_GUARD_VALUE = 8192.0
FP32_SIGN_MASK = 1 << (_GOLDEN_UINT32_BITS - 1)
UINT16_VALUE_COUNT = 1 << _GOLDEN_UINT16_BITS
FLOAT_EXPONENT_COUNT = 1 << _GOLDEN_BITS_PER_BYTE
REFERENCE_RANDOM_SAMPLE_COUNT = 65536
DTYPE_PAIRS = (
    (torch.float32, torch.float8_e4m3fn),
    (torch.bfloat16, torch.float8_e4m3fn),
    (torch.float32, torch.float8_e5m2),
    (torch.bfloat16, torch.float8_e5m2),
)


def _input_numpy(cpu):
    values = cpu.float().numpy()
    return values.astype(bfloat16) if cpu.dtype == torch.bfloat16 else values


def _expected_scale_shape(shape, axis=-1):
    axis %= len(shape)
    result = list(shape)
    result[axis] = (
        result[axis] + _GOLDEN_SCALE_PAIR_ROUND_UP_ELEMENTS
    ) // _GOLDEN_SCALE_PAIR_ELEMENTS
    return (*result, _GOLDEN_SCALES_PER_PAIR)


def _framework_state_words(state, device):
    """Inspect a reserved carrier after execution, without a helper kernel."""
    captured, secondary, seed, offset, intragraph = state
    assert not secondary
    if captured:
        assert seed.device == offset.device == device
        seed, offset = int(seed.item()), int(offset.item()) + intragraph
    return [
        int(seed) & ((1 << _GOLDEN_UINT64_BITS) - 1),
        int(offset) & ((1 << _GOLDEN_UINT64_BITS) - 1),
    ]


def _public_replay(input, *, rng_seed=0, rng_block_offset=0, **attributes):
    """Exercise the real public API at a known framework state, eager tests only.

    Restore the caller's generator after enqueueing; eager main-kernel scalar
    arguments carry the reserved values. Default-resampling/graph tests do not use
    this helper, so precision replay cannot hide a broken default lifecycle.
    """
    with torch.npu.device(input.device):
        saved = torch.npu.get_rng_state(input.device)
        generator = torch.npu.default_generators[input.device.index]
        try:
            generator.manual_seed(rng_seed)
            generator.set_offset(rng_block_offset)
            return mixed.dynamic_mx_quant_sr(input=input, **attributes)
        finally:
            torch.npu.set_rng_state(saved, input.device)


def _assert_result(
    cpu,
    result,
    axis,
    src,
    dst_code,
    algorithm,
    bound,
    *,
    rng_seed=0,
    rng_block_offset=0,
):
    y, scale = result
    dst = torch.float8_e5m2 if dst_code == E5M2_DST_TYPE else torch.float8_e4m3fn
    expected = dynamic_mx_quant_golden(
        _input_numpy(cpu),
        str(dst).removeprefix("torch."),
        axis=axis,
        scale_alg=algorithm,
        max_low_bound=bound,
        seed=rng_seed,
        block_offset=rng_block_offset,
    )
    assert cpu.dtype == src and y.shape == cpu.shape and y.dtype == dst
    assert y.device.type in ("npu", "privateuseone") and scale.device == y.device
    assert scale.shape == _expected_scale_shape(tuple(cpu.shape), axis)
    assert scale.dtype == torch.float8_e8m0fnu
    actual = tuple(value.view(torch.uint8).cpu().numpy() for value in result)
    for value, reference in zip(actual, expected):
        np.testing.assert_array_equal(value, reference)
    assert all(value.is_contiguous() for value in result)
    return actual


def _axis_input(shape, axis, src):
    generator = torch.Generator().manual_seed(AXIS_INPUT_SEED)
    cpu = (torch.randn(shape, generator=generator) * 7).to(src)
    logical = cpu.movedim(axis, -1).contiguous().reshape(-1, shape[axis])
    logical[0, :32] = 0
    logical[0, 1:32:2] = -0.0
    if logical.numel() >= 40:
        # Invalid values get their own group. Finite underflow is tested separately.
        logical.reshape(-1)[32:40] = torch.tensor(
            [float("inf"), -float("inf"), float("nan"), 0.5, -0.5, 1.125, -1.125, 0.0]
        ).to(src)
    return logical.reshape(cpu.movedim(axis, -1).shape).movedim(-1, axis).contiguous()


def _random_input(shape, src, seed=PRECISION_INPUT_SEED):
    return (
        torch.randn(shape, generator=torch.Generator().manual_seed(seed))
        * RANDOM_INPUT_AMPLITUDE
    ).to(src)


@kernel
class Q16Probe:
    def __init__(self, mantissa, bias, max_code):
        self.mantissa, self.bias, self.max_code = (mantissa, bias, max_code)
        self.x = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, SIMD_U32_LANES)), depth=1
        )
        self.random = RingBuffer(
            MemLoc.UB, dtypes.uint32, make_layout((1, SIMD_U32_LANES)), depth=1
        )
        self.y = RingBuffer(
            MemLoc.UB, dtypes.uint8, make_layout((1, SIMD_U32_LANES)), depth=1
        )

    def __call__(self, x, random, out):
        for tile in range(get_block_idx(), x.shape[0], get_block_num()):
            mem_copy(self.x.next_write(), tile_slice(x, (1, SIMD_U32_LANES), (tile, 0)))
            mem_copy(
                self.random.next_write(),
                tile_slice(random, (1, SIMD_U32_LANES), (tile, 0)),
            )
            x_tile = self.x.next_read()
            random_tile = self.random.next_read()
            y_tile = self.y.next_write()
            with vf(mode="simd"):
                mask = reg.full_mask()
                codes = mixed._sr_fp32_to_fp8_q16(
                    reg.vload(x_tile, 0),
                    reg.vload(random_tile, 0),
                    self.mantissa,
                    self.bias,
                    self.max_code,
                    mask,
                )
                reg.vstore_pack(
                    y_tile, 0, codes, mask, pack_mode=reg.PackMode.B32_TO_B8
                )
                reg.vmem_bar("vst_vld")
            mixed._copy_to_gm(
                tile_slice(out, (1, SIMD_U32_LANES), (tile, 0)), self.y.next_read()
            )


class Q16Launch:
    def __init__(self, dst):
        self.bias, self.mantissa, self.max_code = _GOLDEN_FP8_SR_PARAMS[dst]

    @host
    def run(self, x, random, out):
        op = Q16Probe(self.mantissa, self.bias, self.max_code)
        op[Q16_PROBE_BLOCKS](x, random, out)


def _q16_inputs(family):
    random_edges = np.array(
        [0, 1, 32767, 32768, 65534, 65535, 2882371584, 4294967295], dtype=np.uint32
    )
    if family == "bf16_all_bits":
        bits = np.arange(UINT16_VALUE_COUNT, dtype=np.uint32) << np.uint32(
            _GOLDEN_UINT16_BITS
        )
    elif family == "exponent_edges":
        fractions = np.array(
            [
                0,
                1,
                7,
                15,
                16,
                31,
                32,
                131071,
                262143,
                4194303,
                4194304,
                8388606,
                8388607,
            ],
            dtype=np.uint32,
        )
        positive = (
            (
                np.arange(FLOAT_EXPONENT_COUNT, dtype=np.uint32)
                << np.uint32(_GOLDEN_FP32_FRACTION_BITS)
            )[:, None]
            | fractions
        ).reshape(-1)
        bits = np.concatenate((positive, positive | np.uint32(FP32_SIGN_MASK)))
    else:
        rng = np.random.default_rng(PRECISION_INPUT_SEED)
        bits = rng.integers(
            0,
            2**_GOLDEN_UINT32_BITS,
            size=REFERENCE_RANDOM_SAMPLE_COUNT,
            dtype=np.uint32,
        )
        fields = rng.integers(
            0,
            2**_GOLDEN_UINT32_BITS,
            size=REFERENCE_RANDOM_SAMPLE_COUNT,
            dtype=np.uint32,
        )
        return (
            bits.view(np.float32).reshape(-1, SIMD_U32_LANES),
            fields.reshape(-1, SIMD_U32_LANES),
        )
    return (
        np.repeat(bits, random_edges.size).view(np.float32).reshape(-1, SIMD_U32_LANES),
        np.tile(random_edges, bits.size).reshape(-1, SIMD_U32_LANES),
    )


def _finite_scale_boundary_input(src, dst_code, algorithm):
    # E4M3 uses exponent shift 8, E5M2 uses 15. Build exact representable
    # neighbors of the largest amax whose E8M0 code is zero. BF16 needs
    # adjacent BF16 values, rather than FP32 neighbors that collapse on cast.
    shift = (
        _GOLDEN_E5M2_SCALE_EXPONENT_SHIFT
        if dst_code == E5M2_DST_TYPE
        else _GOLDEN_E4M3_SCALE_EXPONENT_SHIFT
    )
    fraction_bits = (
        _GOLDEN_BF16_FRACTION_BITS
        if src == torch.bfloat16
        else _GOLDEN_FP32_FRACTION_BITS
    )
    if algorithm == 0:
        last_zero = ((shift + 1) << fraction_bits) - 1
    elif src == torch.bfloat16:
        last_zero = (shift << _GOLDEN_BF16_FRACTION_BITS) | 0x60
    else:
        last_zero = (shift << _GOLDEN_FP32_FRACTION_BITS) | 0x600001
    bits = [
        0,
        1,
        last_zero - 1,
        last_zero,
        last_zero + 1,
        last_zero + 2,
        (shift + 2) << fraction_bits,
        _GOLDEN_FP32_EXPONENT_BIAS << fraction_bits,
    ]
    raw = torch.tensor(bits, dtype=torch.int32)
    if src == torch.bfloat16:
        raw = raw << _GOLDEN_UINT16_BITS
    magnitudes = raw.view(torch.float32).to(src)
    cpu = magnitudes.repeat_interleave(_GOLDEN_MX_BLOCK_ELEMENTS).reshape(
        8, _GOLDEN_MX_BLOCK_ELEMENTS
    )
    cpu[:, 1::2].neg_()
    return cpu


# RNG isolation, fixtures and functional cases.


def _restore_rng_state(saved, device):
    """Restore the generator even if pending device work fails."""
    try:
        torch.npu.synchronize(device)
    finally:
        torch.npu.set_rng_state(saved, device)


@contextmanager
def _preserve_rng_device(device):
    """Restore entry RNG even when warmup, capture, or synchronization fails."""
    with torch.npu.device(device):
        saved = torch.npu.get_rng_state(device).clone()
        try:
            yield device
        finally:
            _restore_rng_state(saved, device)


@pytest.fixture
def rng_device():
    device = torch.device("npu", torch.npu.current_device())
    with _preserve_rng_device(device):
        yield device


@pytest.fixture
def benchmark_rng_checkpoint(rng_device):
    generator = torch.npu.default_generators[rng_device.index]
    generator.manual_seed(BENCHMARK_CHECKPOINT_SEED)
    generator.set_offset(BENCHMARK_CHECKPOINT_OFFSET_WORDS)
    return torch.npu.get_rng_state(rng_device).clone()


def _release_graph(graph, device):
    try:
        torch.npu.synchronize(device)
    finally:
        graph.reset()


@pytest.fixture
def sdk_identity_probe(monkeypatch):
    mixed.clear_caches()
    state = {"native": "sdk-a", "compiler": "compiler-a"}
    runtime_calls, compiler_calls = [], []

    def runtime():
        runtime_calls.append(state["native"])
        return state["native"]

    def compiler():
        compiler_calls.append(state["compiler"])
        return state["compiler"]

    monkeypatch.setattr(mixed.cache_key, "_runtime_identity", runtime)
    monkeypatch.setattr(mixed.cache_key, "_toolchain_version", compiler)
    try:
        yield state, runtime_calls, compiler_calls
    finally:
        mixed.clear_caches()


# Public precision cases use one configuration per pytest item. Layout rows
# cover both source types; format and algorithm combinations rotate by layout.
# Larger tail cases cover packed RNG, partial tiles and compact BF16 workloads.
PRECISION_CASES = [
    pytest.param((3, 512), -1, torch.float32, 24, 0, 0.0, id="tail-fp32-e4-alg0"),
    pytest.param((16,), 0, torch.bfloat16, 23, 1, 32.0, id="minimum-bf16-e5-bound"),
    pytest.param((2, 512, 2), 1, torch.float32, 23, 1, 0.0, id="stride2-fused-fp32"),
    pytest.param((2, 512, 2), -2, torch.bfloat16, 24, 0, 0.0, id="stride2-fused-bf16"),
    pytest.param((2, 576, 2), 1, torch.float32, 24, 1, 0.0, id="stride2-fp32"),
    pytest.param((2, 576, 2), -2, torch.bfloat16, 23, 0, 0.0, id="stride2-bf16"),
    pytest.param((256, 8), 0, torch.float32, 23, 0, 0.0, id="width8-fp32"),
    pytest.param((256, 8), -2, torch.bfloat16, 24, 1, 0.0, id="width8-bf16"),
    pytest.param((256, 16), 0, torch.float32, 24, 0, 0.0, id="width16-fp32"),
    pytest.param((256, 16), -2, torch.bfloat16, 23, 1, 0.0, id="width16-bf16"),
    pytest.param((64, 32), 0, torch.float32, 23, 1, 0.0, id="width32-fp32"),
    pytest.param((64, 32), -2, torch.bfloat16, 24, 0, 0.0, id="width32-bf16"),
    pytest.param((64, 64), 0, torch.float32, 24, 1, 0.0, id="width64-fp32"),
    pytest.param((64, 64), -2, torch.bfloat16, 23, 0, 0.0, id="width64-bf16"),
    pytest.param((64, 128), 0, torch.float32, 23, 0, 0.0, id="width128-fp32"),
    pytest.param((64, 128), -2, torch.bfloat16, 24, 1, 0.0, id="width128-bf16"),
    pytest.param((2, 48, 3), 1, torch.float32, 24, 0, 0.0, id="generic-axis-fp32"),
    pytest.param((2, 48, 3), -2, torch.bfloat16, 23, 1, 0.0, id="generic-axis-bf16"),
    pytest.param((8192, 256), -1, torch.float32, 24, 1, 0.0, id="large-fp32-packed"),
    pytest.param(
        (2, 32, 136, 256), -1, torch.float32, 23, 0, 0.0, id="rank4-multiple-tiles"
    ),
    pytest.param((32768, 256), -1, torch.bfloat16, 24, 0, 0.0, id="large-bf16-panels"),
    pytest.param((512, 4096), -1, torch.bfloat16, 24, 1, 0.0, id="bf16-compact"),
]

# Named replay and guard values are shared by functional and memory checks.
TEST_RNG_SEED = (1 << _GOLDEN_UINT64_BITS) - 1
TEST_WORD_CARRY_OFFSET = (1 << _GOLDEN_UINT32_BITS) - _GOLDEN_PHILOX_WORDS_PER_BLOCK
TEST_COUNTER_CARRY_OFFSET = _GOLDEN_PHILOX_WORDS_PER_BLOCK * _GOLDEN_UINT32_MASK
TEST_INPUT_OFFSET = 7
OUTPUT_GUARD_BYTES = 64
OUTPUT_GUARD_CAPACITY_BYTES = 65536
OUTPUT_GUARD_VALUE = 0xA5
OUTPUT_GUARD_BLOCKS = 4
SPECIAL_VALUE_BOUND = 32.0


def _public_check(cpu, axis, dst, algorithm, device, bound=0.0):
    """Check one public call, both byte outputs, input storage and RNG isolation."""
    backing_cpu = torch.full(
        (TEST_INPUT_OFFSET + cpu.numel() + INPUT_GUARD_ELEMENTS,),
        INPUT_GUARD_VALUE,
        dtype=cpu.dtype,
    )
    backing_cpu[TEST_INPUT_OFFSET : TEST_INPUT_OFFSET + cpu.numel()].copy_(
        cpu.reshape(-1)
    )
    backing = backing_cpu.to(device)
    x = backing[TEST_INPUT_OFFSET : TEST_INPUT_OFFSET + cpu.numel()].reshape(cpu.shape)
    saved = torch.npu.get_rng_state(device).clone()
    result = _public_replay(
        x,
        axis=axis,
        dst_type=dst,
        scale_alg=algorithm,
        max_low_bound=bound,
        rng_seed=TEST_RNG_SEED,
        rng_block_offset=TEST_WORD_CARRY_OFFSET,
    )
    _assert_result(
        cpu,
        result,
        axis,
        cpu.dtype,
        dst,
        algorithm,
        bound,
        rng_seed=TEST_RNG_SEED,
        rng_block_offset=TEST_WORD_CARRY_OFFSET,
    )
    assert torch.equal(torch.npu.get_rng_state(device), saved)
    assert all(value.data_ptr() != x.data_ptr() for value in result)
    assert torch.equal(backing.cpu().view(torch.uint8), backing_cpu.view(torch.uint8))
    return result


@pytest.mark.npu
@pytest.mark.parametrize("shape,axis,src,dst,algorithm,bound", PRECISION_CASES)
def test_public_precision(shape, axis, src, dst, algorithm, bound, rng_device):
    cpu = _axis_input(shape, axis, src)
    if cpu.numel() <= _GOLDEN_MX_BLOCK_ELEMENTS:
        # Minimum BF16 rows must also contain finite nonzero values.
        cpu = _random_input(shape, src)
        cpu.reshape(-1)[:2] = torch.tensor([0.0, -0.0], dtype=src)
    _public_check(cpu, axis, dst, algorithm, rng_device, bound)


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS)
@pytest.mark.parametrize("algorithm", [0, 1])
def test_scale_underflow_and_signed_zero(src, dst, algorithm, rng_device):
    code = E5M2_DST_TYPE if dst == torch.float8_e5m2 else E4M3_DST_TYPE
    cpu = _finite_scale_boundary_input(src, code, algorithm)
    # Keep both packed and short-row geometry in the parameter table's cases.
    if algorithm == 0:
        cpu = cpu.reshape(1, -1)
    else:
        cpu = torch.cat((cpu, cpu[:, :_GOLDEN_RNG_GROUP_ELEMENTS]), dim=1)
    _public_check(cpu, -1, code, algorithm, rng_device)


@pytest.mark.npu
@pytest.mark.parametrize(
    "family,dst",
    [
        pytest.param("bf16_all_bits", "float8_e4m3fn", id="bf16-all-bits-e4m3"),
        pytest.param("exponent_edges", "float8_e5m2", id="fp32-exponent-edges-e5m2"),
    ],
)
def test_element_rounding_boundaries(family, dst, rng_device):
    values, fields = _q16_inputs(family)
    expected = _golden_sr_cast_fp32_to_fp8(values, fields, dst)
    x = torch.from_numpy(values).to(rng_device)
    random = torch.from_numpy(fields).to(rng_device)
    output = torch.empty(values.shape, dtype=torch.uint8, device=rng_device)
    Q16Launch(dst).run(x, random, output)
    np.testing.assert_array_equal(output.cpu().numpy(), expected)


@pytest.mark.npu
@pytest.mark.parametrize(
    "shape,axis,src,dst,algorithm",
    [
        pytest.param((2, 80), 1, torch.bfloat16, 24, 0, id="short-row"),
        pytest.param((2, 512, 2), 1, torch.float32, 24, 0, id="stride2"),
        pytest.param((8193, 256), 1, torch.float32, 24, 1, id="packed-tail"),
        pytest.param((512, 256), 1, torch.bfloat16, 24, 0, id="full-tile"),
    ],
)
def test_output_memory_guards(shape, axis, src, dst, algorithm, rng_device):
    cpu = _axis_input(shape, axis, src)
    x = cpu.to(rng_device)
    target = torch.float8_e5m2 if dst == E5M2_DST_TYPE else torch.float8_e4m3fn
    prepared = mixed._prepare_launch(
        x, target, axis, algorithm, 0, block_dim=OUTPUT_GUARD_BLOCKS
    )
    storage = [
        torch.full(
            (OUTPUT_GUARD_BYTES + value.numel() + OUTPUT_GUARD_CAPACITY_BYTES,),
            OUTPUT_GUARD_VALUE,
            dtype=torch.uint8,
            device=rng_device,
        )
        for value in (prepared.encoded, prepared.scales)
    ]
    encoded, scales = [
        backing[OUTPUT_GUARD_BYTES : OUTPUT_GUARD_BYTES + value.numel()].view(
            value.shape
        )
        for backing, value in zip(storage, (prepared.encoded, prepared.scales))
    ]
    arguments = list(prepared.arguments)
    stride2 = prepared.plan.layout in (
        mixed.Layout.NON_TAIL,
        mixed.Layout.NON_TAIL_FUSED,
    )
    arguments[2 if stride2 else 1] = encoded.view(prepared.plan.kernel_shape)
    arguments[3 if stride2 else 2] = scales.view(prepared.plan.kernel_scale_shape)
    prepared = replace(
        prepared, arguments=tuple(arguments), encoded=encoded, scales=scales
    )
    saved = torch.npu.get_rng_state(rng_device).clone()
    state = mixed._make_rng_state(
        TEST_RNG_SEED, TEST_WORD_CARRY_OFFSET, device=rng_device
    )
    prepared.launch(rng_state=state)
    torch.npu.synchronize(rng_device)
    for backing, value in zip(storage, (encoded, scales)):
        assert torch.all(backing[:OUTPUT_GUARD_BYTES] == OUTPUT_GUARD_VALUE).item()
        assert torch.all(
            backing[OUTPUT_GUARD_BYTES + value.numel() :] == OUTPUT_GUARD_VALUE
        ).item()
    _assert_result(
        cpu,
        prepared.outputs(),
        axis,
        src,
        dst,
        algorithm,
        0,
        rng_seed=TEST_RNG_SEED,
        rng_block_offset=TEST_WORD_CARRY_OFFSET,
    )
    assert torch.equal(torch.npu.get_rng_state(rng_device), saved)
    assert torch.equal(x.cpu().view(torch.uint8), cpu.view(torch.uint8))


@pytest.mark.npu
@pytest.mark.parametrize(
    "shape,axis,src,dst,algorithm,nondefault_stream",
    [
        pytest.param((2, 1024), -1, torch.bfloat16, 23, 0, False, id="tail"),
        pytest.param((2, 48, 3), 1, torch.float32, 24, 1, True, id="non-tail-stream"),
    ],
)
def test_graph_replay_and_state_restore(
    shape, axis, src, dst, algorithm, nondefault_stream, rng_device
):
    cpu = _random_input(shape, src)
    x = cpu.to(rng_device)
    attributes = dict(axis=axis, dst_type=dst, scale_alg=algorithm)
    mixed.dynamic_mx_quant_sr(
        x, **attributes
    )  # Compile and initialize outside capture.
    torch.npu.synchronize(rng_device)
    generator = torch.npu.default_generators[rng_device.index]
    generator.manual_seed(TEST_RNG_SEED)
    increment = _rng_budget(x.numel())
    stream = (
        torch.npu.Stream(device=rng_device)
        if nondefault_stream
        else torch.npu.current_stream(rng_device)
    )
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph, stream=stream if nondefault_stream else None):
            result = mixed.dynamic_mx_quant_sr(x, **attributes)
        generator.set_offset(TEST_COUNTER_CARRY_OFFSET)
        checkpoint = torch.npu.get_rng_state(rng_device).clone()
        with torch.npu.stream(stream):
            graph.replay()
        torch.npu.current_stream(rng_device).wait_stream(stream)
        _assert_result(
            cpu,
            result,
            axis,
            src,
            dst,
            algorithm,
            0,
            rng_seed=TEST_RNG_SEED,
            rng_block_offset=TEST_COUNTER_CARRY_OFFSET,
        )
        with torch.npu.stream(stream):
            graph.replay()
        torch.npu.current_stream(rng_device).wait_stream(stream)
        _assert_result(
            cpu,
            result,
            axis,
            src,
            dst,
            algorithm,
            0,
            rng_seed=TEST_RNG_SEED,
            rng_block_offset=TEST_COUNTER_CARRY_OFFSET + increment,
        )
        assert generator.get_offset() == TEST_COUNTER_CARRY_OFFSET + 2 * increment
        torch.npu.set_rng_state(checkpoint, rng_device)
        with torch.npu.stream(stream):
            graph.replay()
            eager = mixed.dynamic_mx_quant_sr(x, **attributes)
        torch.npu.current_stream(rng_device).wait_stream(stream)
        _assert_result(
            cpu,
            result,
            axis,
            src,
            dst,
            algorithm,
            0,
            rng_seed=TEST_RNG_SEED,
            rng_block_offset=TEST_COUNTER_CARRY_OFFSET,
        )
        _assert_result(
            cpu,
            eager,
            axis,
            src,
            dst,
            algorithm,
            0,
            rng_seed=TEST_RNG_SEED,
            rng_block_offset=TEST_COUNTER_CARRY_OFFSET + increment,
        )
        assert generator.get_offset() == TEST_COUNTER_CARRY_OFFSET + 2 * increment
    finally:
        _release_graph(graph, rng_device)


@pytest.mark.parametrize(
    "attributes,error,match",
    [
        pytest.param({"axis": True}, TypeError, "axis", id="axis-type"),
        pytest.param(
            {"scale_alg": 2}, NotImplementedError, "scale_alg", id="scale-algorithm"
        ),
        pytest.param(
            {"max_low_bound": float("nan")},
            ValueError,
            "max_low_bound",
            id="nonfinite-bound",
        ),
    ],
)
def test_rejects_invalid_attributes_before_device_work(
    attributes, error, match, monkeypatch
):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid attributes reached device setup")

    monkeypatch.setattr(torch.npu, "device", unexpected)
    with pytest.raises(error, match=match):
        mixed.dynamic_mx_quant_sr(torch.ones((2, 64)), **attributes)


def test_sdk_identity_reuses_loaded_runtime_and_clear_refreshes(sdk_identity_probe):
    state, runtime_calls, compiler_calls = sdk_identity_probe
    first = mixed._build_identity()
    state.update(native="sdk-b", compiler="compiler-b")
    second = mixed._build_identity()
    assert first[1:3] == ("sdk-a", "compiler-a")
    # SDK file edits have an explicit refresh boundary; toolchain identity is live.
    assert second[1:3] == ("sdk-a", "compiler-b")
    assert runtime_calls == ["sdk-a"]
    assert compiler_calls == ["compiler-a", "compiler-b"]
    mixed.clear_caches()
    assert mixed._build_identity()[1:3] == ("sdk-b", "compiler-b")
    assert runtime_calls == ["sdk-a", "sdk-b"]


@pytest.mark.npu
def test_rng_eager_overflow_does_not_advance(rng_device):
    generator = torch.npu.default_generators[rng_device.index]
    generator.set_offset(2**64 - 4)
    with pytest.raises(ValueError, match="overflow"):
        mixed._reserve_rng_carrier(rng_device, 4)
    assert generator.get_offset() == 2**64 - 4


@pytest.mark.npu
def test_rng_reservation_concurrent(rng_device):
    mixed._reserve_rng_carrier(rng_device, 4)
    torch.npu.synchronize(rng_device)
    generator = torch.npu.default_generators[rng_device.index]
    generator.manual_seed(11)
    with ThreadPoolExecutor(4) as executor:
        futures = [
            executor.submit(mixed._reserve_rng_carrier, rng_device, 8)
            for _ in range(16)
        ]
        values = [
            _framework_state_words(future.result(30), rng_device) for future in futures
        ]
    assert all(seed == 11 for seed, _ in values)
    assert sorted(offset for _, offset in values) == list(range(0, 128, 8))
    assert generator.get_offset() == 128


@pytest.mark.npu
def test_mixed_non_tail_fast_path_reuses_one_binary_for_dynamic_k(rng_device):
    api = mixed.dynamic_mx_quant_sr
    before = mixed.cache_info()
    for columns in (512, 1024):
        x = torch.ones((2, columns, 2), dtype=torch.float32).to(rng_device)
        y, scale = api(x, axis=-2, scale_alg=0)
        assert y.is_contiguous() and scale.is_contiguous()
        np.testing.assert_array_equal(
            y.view(torch.uint8).cpu().numpy(), np.full(x.shape, 0x78, dtype=np.uint8)
        )
        np.testing.assert_array_equal(
            scale.view(torch.uint8).cpu().numpy(),
            np.full((2, columns // 64, 2, 2), 119, dtype=np.uint8),
        )
    after = mixed.cache_info()
    assert after.misses - before.misses <= 1
    assert after.hits - before.hits >= 1


def test_golden_philox_known_zero_vector():
    # Published Philox4x32-10 zero counter / zero key vector, independent of DSL.
    expected = np.array(
        [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8], dtype=np.uint32
    )
    np.testing.assert_array_equal(_golden_philox4x32_10(0, 0, 0, 0), expected)
    np.testing.assert_array_equal(
        _golden_philox_word_stream(4, quant_length=16), expected
    )
    np.testing.assert_array_equal(
        _golden_random_bits(8, quant_length=16),
        np.array(
            [0x6627, 0xD566, 0xE8D5, 0x27E8, 0xE169, 0x8DE1, 0xC58D, 0x69C5],
            dtype=np.uint32,
        ),
    )


@pytest.mark.npu
def test_rng_public_resamples_and_restore_replays_sequence(rng_device):
    src, dst = torch.float32, torch.float8_e4m3fn
    # Missing default reservations or captured host constants break the
    # independently computed second result and the restored full sequence.
    cpu = torch.randn((2, 1024), generator=torch.Generator().manual_seed(821)).to(src)
    x = cpu.to(rng_device)
    code = 23 if dst == torch.float8_e5m2 else 24
    seed = 2**63 + 71
    torch.npu.manual_seed(seed)
    saved = torch.npu.get_rng_state(rng_device).clone()
    sequences = []
    for replay in range(2):
        torch.npu.set_rng_state(saved, rng_device)
        sequence = []
        for offset in (0, 4, 8):
            result = mixed.dynamic_mx_quant_sr(x, dst_type=code)
            actual = _assert_result(
                cpu, result, -1, src, code, 1, 0, rng_seed=seed, rng_block_offset=offset
            )
            sequence.append(actual)
        assert torch.npu.default_generators[rng_device.index].get_offset() == 12
        sequences.append(sequence)
    assert not np.array_equal(sequences[0][0][0], sequences[0][1][0])
    for first, restored in zip(*sequences):
        for original_bytes, restored_bytes in zip(first, restored):
            np.testing.assert_array_equal(original_bytes, restored_bytes)


@pytest.mark.npu
def test_benchmark_real_sr_and_shared_launch_plan(
    tmp_path, rng_device, benchmark_rng_checkpoint
):
    src, dst, algorithm = "bf16", "e5m2", 1
    bench = sys.modules[__name__]
    bench.main(
        [
            "--device",
            str(rng_device.index),
            "--shape",
            "3x80",
            "--src",
            src,
            "--dst",
            dst,
            "--scale-alg",
            str(algorithm),
            "--block-dim",
            "7",
            "--rng-seed",
            str(2**64 - 1),
            "--rng-block-offset",
            str(2**63),
            "--warmup",
            "1",
            "--repeat",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert torch.npu.current_device() == rng_device.index
    assert torch.equal(torch.npu.get_rng_state(rng_device), benchmark_rng_checkpoint)
    records = list(tmp_path.glob("*/metadata.json"))
    assert len(records) == 1
    metadata = json.loads(records[0].read_text())
    assert metadata["api_entry"] == "dynamic_mx_quant_sr"
    assert metadata["runner"] == "direct_aot_shared_execution_plan"
    assert metadata["validation"]["passed"] and metadata["final_validation"]["passed"]
    assert metadata["workload_completed"] and metadata["actual_blocks"] == 6
    assert metadata["compiled_so_sha256"]
    assert metadata["measurement_scope"] == "preallocated_kernel_only"
    assert metadata["settings"]["rng_seed"] == 2**64 - 1
    assert metadata["settings"]["rng_block_offset"] == 2**63
    assert (
        metadata["settings"]["rng_mapping"]
        == "npu_linear16384_c1_bf16_width16_unified_fp32_bytepair_sr_v15"
    )
    assert (
        metadata["settings"]["offset_unit"] == metadata["offset_unit"] == "uint32_word"
    )
    assert metadata["rng_abi_version"] == 60 and metadata["rng_increment"] == 4
    assert metadata["rng_state_mode"] == "explicit_device_state_replay"


# Performance CLI: byte validation, export and msprof measurements.
_SR_SAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "samples/dynamic_mx_quant_sr/dynamic_mx_quant_sr.py"
)

BENCHMARK_INPUT_SEED = 20260920  # CPU input generation, not a public SR seed attribute.


RNG_ABI_VERSION = 60


RNG_MAPPING = "npu_linear16384_c1_bf16_width16_unified_fp32_bytepair_sr_v15"


RNG_INCREMENT = 4  # Minimum; per-case budgets include logical padding.


def _rng_budget(numel):
    return _GOLDEN_PHILOX_WORDS_PER_BLOCK * (
        (
            numel // _GOLDEN_RNG_GROUP_ELEMENTS
            + RNG_MAX_PADDED_GROUPS
            + RNG_STREAM_ROUND_UP_GROUPS
        )
        // _GOLDEN_RNG_VIRTUAL_STREAMS
    )


def _shape(value):
    try:
        result = tuple(int(part) for part in re.split("[xX,]", value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use dimensions such as 8x4096 or 2,3,80"
        ) from error
    if not result or any(dimension <= 0 for dimension in result):
        raise argparse.ArgumentTypeError("all dimensions must be positive")
    return result


def _load(path, name):
    """Load a single-file implementation under a path-specific module name."""
    path = Path(path).resolve(strict=True)
    qualified_name = (
        "_sr_benchmark_"
        + name
        + "_"
        + hashlib.sha256(str(path).encode()).hexdigest()[:MODULE_IDENTITY_HEX_DIGITS]
    )
    if qualified_name in sys.modules:
        return sys.modules[qualified_name]
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SR benchmark module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(qualified_name, None)
        raise
    return module


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _case_name(shape, args):
    axis = args.axis % len(shape)
    bound = float(args.max_low_bound).hex().replace("+", "p").replace("-", "m")
    return f"{args.src}_{args.dst}_alg{args.scale_alg}_axis{axis}_bound{bound}_{'x'.join(map(str, shape))}"


def _check_bytes(actual, expected):
    import numpy as np

    # Never broadcast or coerce an invalid carrier into matching byte values.
    # Both outputs have already been viewed as uint8 by the caller.
    if (
        actual.dtype != np.uint8
        or expected.dtype != np.uint8
        or actual.shape != expected.shape
        or actual.size != expected.size
    ):
        return {
            "passed": False,
            "mismatched_bytes": None,
            "bytes": int(expected.nbytes),
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "actual_dtype": str(actual.dtype),
            "expected_dtype": str(expected.dtype),
        }
    count = int(np.count_nonzero(actual != expected))
    return {
        "passed": count == 0,
        "mismatched_bytes": count,
        "bytes": int(expected.size),
    }


def _device_kernel_symbols(binary, identity):
    """Device entry strings in the real ELF, not the exported Host run_*."""
    symbols = sorted(
        set(
            value.decode("ascii")
            for value in re.findall(
                rb"_Z[A-Za-z0-9_]*kernel__[A-Za-z0-9_]*KernelArgTensor[A-Za-z0-9_]+",
                Path(binary).read_bytes(),
            )
            if identity.encode("ascii") in value
        )
    )
    if not symbols:
        raise RuntimeError(f"no device kernel symbols for {identity} in {binary}")
    return symbols


def _validate_task_identity(rows, metadata, symbols, blocks):
    visible = metadata.get("visible_devices")
    logical = metadata["device"]
    physical = str(logical) if not visible else visible.split(",")[logical].strip()
    if not symbols:
        raise RuntimeError("device kernel symbol identity is required")
    for row in rows:
        if row.get("Device_id") != physical:
            raise RuntimeError("wrong profiler device identity")
        if row.get("Block Num") != str(blocks):
            raise RuntimeError("wrong profiler blocks identity")
        name = row.get("Op Name", "").strip('"')
        kind = row.get("OP Type", row.get("Op Type", "")).strip('"')
        if name not in symbols or kind not in symbols:
            raise RuntimeError("wrong compiled device kernel symbol identity")


def _run_case(shape, args):
    import numpy as np
    import torch

    if sys.byteorder != "little":
        raise RuntimeError(
            "raw C++ replay bundles currently require a little-endian host"
        )
    case_dir = args.output_dir / _case_name(shape, args)
    case_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = case_dir / "metadata.json"
    metadata = {
        "shape": list(shape),
        "src": args.src,
        "dst": args.dst,
        "input_seed": BENCHMARK_INPUT_SEED,
        "input_distribution": "CPU torch.randn * 7, then cast to source dtype",
        "module_path": str(args.module_path),
        "source_sha256": hashlib.sha256(args.module_path.read_bytes()).hexdigest(),
        "byteorder": "little",
        "wheel_version": cannbotdsl.__version__,
        "settings": {
            "axis": args.axis,
            "block_size": _GOLDEN_MX_BLOCK_ELEMENTS,
            "scale_alg": args.scale_alg,
            "max_low_bound": args.max_low_bound,
            "round_mode": "stochastic",
            "rng_seed": args.rng_seed,
            "rng_block_offset": args.rng_block_offset,
            "rng_mapping": RNG_MAPPING,
            "offset_unit": "uint32_word",
            "block_dim": args.block_dim,
            "warmup": args.warmup,
            "repeat": args.repeat,
        },
        "device": args.device,
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "requested_block_dim": args.block_dim,
        "block_dim_source": "stream_vector_core_num"
        if args.block_dim is None
        else "explicit",
        "validation": {"passed": False, "status": "not_run"},
        "final_validation": {"passed": False, "status": "not_run"},
        "measurement_scope": "preallocated_kernel_only",
        "rng_state_mode": "explicit_device_state_replay",
        "rng_abi_version": RNG_ABI_VERSION,
        "rng_increment": _rng_budget(math.prod(shape)),
        "offset_unit": "uint32_word",
        "default_rng": {
            "seed": args.rng_seed,
            "framework_offset": 0,
            "offset_interpretation": "uint32_word",
            "reservation_count": 1
            + (args.warmup + args.repeat if args.rng_overheads else 0),
        },
        "excluded_operations": [
            "compilation",
            "allocation",
            "validation",
            "input/output layout conversions",
            "generator reservation",
        ],
        "profiling": {"metric": "msprof Task Duration(us)", "status": "not_collected"},
    }
    _write_json(metadata_path, metadata)
    try:
        module = _load(args.module_path, "_dynamic_mx_quant_benchmark_operator")
        metadata["sources_sha256"] = {
            str(path.relative_to(args.module_path.parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(args.module_path.parent.rglob("*"))
            if path.suffix in (".py", ".cpp")
        }
        metadata["test_source_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        input_dtype = torch.float32 if args.src == "fp32" else torch.bfloat16
        output_dtype = (
            torch.float8_e4m3fn if args.dst == "e4m3fn" else torch.float8_e5m2
        )
        x_cpu = (
            torch.randn(
                shape, generator=torch.Generator().manual_seed(BENCHMARK_INPUT_SEED)
            )
            * RANDOM_INPUT_AMPLITUDE
        ).to(input_dtype)
        input_bytes = x_cpu.view(torch.uint8).numpy().tobytes()
        (case_dir / "x.bin").write_bytes(input_bytes)
        metadata["input_sha256"] = hashlib.sha256(input_bytes).hexdigest()
        metadata["input_bytes"] = len(input_bytes)
        numpy_input = x_cpu.float().numpy()
        if args.src == "bf16":
            numpy_input = numpy_input.astype(bfloat16)
        expected = dynamic_mx_quant_golden(
            numpy_input,
            str(output_dtype).removeprefix("torch."),
            axis=args.axis,
            scale_alg=args.scale_alg,
            max_low_bound=args.max_low_bound,
            seed=args.rng_seed,
            block_offset=args.rng_block_offset,
        )
        public_expected = dynamic_mx_quant_golden(
            numpy_input,
            str(output_dtype).removeprefix("torch."),
            axis=args.axis,
            scale_alg=args.scale_alg,
            max_low_bound=args.max_low_bound,
            seed=args.rng_seed,
            block_offset=0,
        )
        with _preserve_rng_device(torch.device("npu", args.device)):
            x = x_cpu.to(f"npu:{args.device}")
            metadata["api_entry"] = "dynamic_mx_quant_sr"
            metadata["validation_launch_policy"] = "automatic"
            increment = module._rng_increment(x.numel())
            reservations = metadata["default_rng"]["reservation_count"]
            if increment != _rng_budget(x.numel()):
                raise RuntimeError(
                    "benchmark requires RNG ABI60 linear mapping reservation"
                )
            module._validate_rng_args(args.rng_seed, args.rng_block_offset)
            module._validate_rng_args(args.rng_seed, (reservations - 1) * increment)
            state = module._make_rng_state(
                args.rng_seed, args.rng_block_offset, device=x.device
            )
            generator = torch.npu.default_generators[x.device.index]
            generator.manual_seed(args.rng_seed)
            generator.set_offset(0)
            torch.npu.synchronize()
            first_call_start = time.perf_counter_ns()
            y, scales = module.dynamic_mx_quant_sr(
                input=x,
                axis=args.axis,
                dst_type=E5M2_DST_TYPE if args.dst == "e5m2" else E4M3_DST_TYPE,
                scale_alg=args.scale_alg,
                max_low_bound=args.max_low_bound,
            )
            torch.npu.synchronize()
            metadata["first_public_call_us"] = (
                time.perf_counter_ns() - first_call_start
            ) / NANOSECONDS_PER_MICROSECOND
            metadata["first_call_scope"] = (
                "case-first public call including cache lookup/build and synchronize; "
                "process and disk caches may be warm"
            )
            actual = (
                y.view(torch.uint8).cpu().numpy(),
                scales.view(torch.uint8).cpu().numpy(),
            )
            validation = {
                name: _check_bytes(value, reference)
                for name, value, reference in zip(
                    ("y", "scale"), actual, public_expected
                )
            }
            validation["passed"] = all(value["passed"] for value in validation.values())
            validation["status"] = "passed" if validation["passed"] else "mismatch"
            metadata["validation"] = validation
            metadata["scale_shape"] = list(scales.shape)
            _write_json(metadata_path, metadata)
            if not validation["passed"]:
                raise RuntimeError(
                    f"public API output differs from golden: {validation}"
                )

            axis, dst, bound_bits = module._validate_attributes(
                x,
                args.axis,
                "stochastic",
                E5M2_DST_TYPE if args.dst == "e5m2" else E4M3_DST_TYPE,
                _GOLDEN_MX_BLOCK_ELEMENTS,
                args.scale_alg,
                0.0,
                args.max_low_bound,
            )
            prepared = module._prepare_launch(
                x, dst, axis, args.scale_alg, bound_bits, block_dim=args.block_dim
            )
            program = prepared.program
            if not program.so_path:
                raise RuntimeError(
                    "benchmark requires a real binary, not a translation-only stub"
                )
            metadata["runner"] = "direct_aot_shared_execution_plan"
            metadata["layout"] = prepared.plan.layout.value
            metadata["kernel_input_shape"] = list(prepared.arguments[0].shape)
            metadata["kernel_scale_shape"] = list(prepared.plan.kernel_scale_shape)
            metadata["settings"]["block_dim"] = args.block_dim
            metadata["executed_so_path"] = program.so_path
            metadata["compiled_so_path"] = str(
                program.save(str(case_dir / "kernel.so"))
            )
            metadata["compiled_symbol"] = program.symbol_name
            metadata["compiled_so_sha256"] = hashlib.sha256(
                Path(metadata["compiled_so_path"]).read_bytes()
            ).hexdigest()
            metadata["device_kernel_symbols"] = _device_kernel_symbols(
                metadata["compiled_so_path"], "DynamicMxQuantSrKernel"
            )
            public_prepared = module._prepare_launch(
                x, dst, axis, args.scale_alg, bound_bits
            )
            public_scalar = getattr(public_prepared, "scalar_program", None)
            public_program = public_scalar or public_prepared.framework_programs[0]
            public_path = str(public_program.save(str(case_dir / "public_kernel.so")))
            metadata["public_device_kernel_symbols"] = _device_kernel_symbols(
                public_path, "DynamicMxQuantSrKernel"
            )
            metadata["public_compiled_so_sha256"] = hashlib.sha256(
                Path(public_path).read_bytes()
            ).hexdigest()
            metadata["public_actual_blocks"] = public_prepared.plan.blocks
            del public_prepared
            metadata["kernel_task_identity"] = "DynamicMxQuantSrKernel"
            metadata["actual_blocks"] = prepared.plan.blocks
            metadata["expected_kernel_tasks"] = 1 + args.warmup + args.repeat
            metadata["excluded_kernel_tasks"] = 1 + args.warmup
            metadata["measured_kernel_tasks"] = args.repeat
            for _ in range(args.warmup):
                prepared.launch(rng_state=state)
            torch.npu.synchronize()
            if args.warmup:
                try:
                    for value, reference in zip(prepared.outputs(), expected):
                        np.testing.assert_array_equal(
                            value.view(torch.uint8).cpu().numpy(), reference
                        )
                except AssertionError:
                    metadata["validation"].update(
                        passed=False, status="warmup_mismatch"
                    )
                    raise
            # Input generation, golden, first compilation, and byte checks are done.
            for _ in range(args.repeat):
                prepared.launch(rng_state=state)
            torch.npu.synchronize()
            final = tuple(
                value.view(torch.uint8).cpu().numpy() for value in prepared.outputs()
            )
            final_validation = {
                name: _check_bytes(value, reference)
                for name, value, reference in zip(("y", "scale"), final, expected)
            }
            final_validation["passed"] = all(
                item["passed"] for item in final_validation.values()
            )
            metadata["final_validation"] = final_validation
            if not final_validation["passed"]:
                raise RuntimeError(
                    f"measured kernel output differs from golden: {final_validation}"
                )
            for name, value in zip(("y", "scale"), final):
                payload = value.tobytes()
                (case_dir / f"{name}.bin").write_bytes(payload)
                metadata[f"{name}_sha256"] = hashlib.sha256(payload).hexdigest()
                metadata[f"{name}_bytes"] = len(payload)
            if args.rng_overheads:
                metadata["rng_overheads"] = _measure_rng_overheads(
                    module, x, numpy_input, output_dtype, args, increment
                )
                metadata["expected_kernel_tasks"] += args.warmup + args.repeat
            metadata["workload_completed"] = True
            _write_json(metadata_path, metadata)
            print(
                json.dumps(
                    {
                        "case": str(case_dir),
                        "validation": True,
                        "repetitions": args.repeat,
                    }
                ),
                flush=True,
            )
    except Exception as error:
        metadata["error"] = f"{type(error).__name__}: {error}"
        metadata["workload_completed"] = False
        _write_json(metadata_path, metadata)
        raise


def _latencies(values):
    return {
        "count": len(values),
        "min_us": min(values),
        "median_us": statistics.median(values),
        "mean_us": statistics.mean(values),
        "max_us": max(values),
        "samples_us": values,
    }


def _measure_rng_overheads(module, x, numpy_input, output_dtype, args, increment):
    """Measure full eager calls separately; profiler supplies device-only times."""
    import numpy as np
    import torch

    enqueue, synchronized = [], []
    for iteration in range(args.warmup + args.repeat):
        torch.npu.synchronize(x.device)
        start = time.perf_counter_ns()
        result = module.dynamic_mx_quant_sr(
            x,
            axis=args.axis,
            dst_type=E5M2_DST_TYPE if args.dst == "e5m2" else E4M3_DST_TYPE,
            scale_alg=args.scale_alg,
            max_low_bound=args.max_low_bound,
        )
        queued = time.perf_counter_ns()
        torch.npu.synchronize(x.device)
        finished = time.perf_counter_ns()
        if iteration >= args.warmup:
            enqueue.append((queued - start) / NANOSECONDS_PER_MICROSECOND)
            synchronized.append((finished - start) / NANOSECONDS_PER_MICROSECOND)
    # One initial public validation preceded the fixed-state quant benchmark.
    final_offset = increment * (args.warmup + args.repeat)
    expected = dynamic_mx_quant_golden(
        numpy_input,
        str(output_dtype).removeprefix("torch."),
        axis=args.axis,
        scale_alg=args.scale_alg,
        max_low_bound=args.max_low_bound,
        seed=args.rng_seed,
        block_offset=final_offset,
    )
    for actual, reference in zip(result, expected):
        np.testing.assert_array_equal(actual.view(torch.uint8).cpu().numpy(), reference)
    final_framework_offset = (1 + args.warmup + args.repeat) * increment
    assert (
        torch.npu.default_generators[x.device.index].get_offset()
        == final_framework_offset
    )
    return {
        "public_host_enqueue": _latencies(enqueue),
        "public_synchronized": _latencies(synchronized),
        "host_scope": (
            "default eager public API including validation/allocation/framework RNG reservation/layout; "
            "synchronized includes final device wait"
        ),
        "public_validation": True,
        "rng_seed": args.rng_seed,
        "initial_framework_offset": 0,
        "final_public_block_offset": final_offset,
        "final_framework_offset": final_framework_offset,
        "reservation_count": 1 + args.warmup + args.repeat,
        "public_kernel_begin": 1 + 2 * args.warmup + args.repeat,
        "measured_tasks": args.repeat,
        "device_profiling": {"status": "not_collected"},
    }


def _collect_task_rows(profile_dir, expected_count, identity):
    rows = []
    for path in sorted(profile_dir.rglob("op_summary*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                task_identity = (
                    row.get("Op Name", "")
                    + " "
                    + row.get("OP Type", row.get("Op Type", ""))
                )
                if identity not in task_identity:
                    continue
                duration_key = next(
                    (
                        key
                        for key in row
                        if key.startswith("Task Duration") and "us" in key
                    ),
                    None,
                )
                start_key = next(
                    (key for key in row if key.startswith("Task Start Time")), None
                )
                if duration_key is None or start_key is None:
                    raise RuntimeError(
                        f"Task Duration(us) or Task Start Time missing in {path}"
                    )
                duration = float(row[duration_key])
                if not math.isfinite(duration) or duration <= 0:
                    raise RuntimeError(
                        f"invalid device task duration in {path}: {row[duration_key]!r}"
                    )
                start = float(row[start_key])
                if not math.isfinite(start) or start < 0:
                    raise RuntimeError(
                        f"invalid device task start time in {path}: {row[start_key]!r}"
                    )
                rows.append(
                    {
                        **row,
                        "source_csv": str(path),
                        "duration_us": duration,
                        "start_time": start,
                    }
                )
    rows.sort(key=lambda row: row["start_time"])
    if len(rows) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} {identity} device tasks, found {len(rows)} in {profile_dir}"
        )
    return rows


def _summarize(case_dir, profile_dir):
    metadata_path = case_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    settings = metadata.get("settings", {})
    if (
        metadata.get("rng_abi_version") != RNG_ABI_VERSION
        or metadata.get("rng_increment") != _rng_budget(math.prod(metadata["shape"]))
        or metadata.get("offset_unit") != "uint32_word"
        or settings.get("offset_unit") != "uint32_word"
        or settings.get("rng_mapping") != RNG_MAPPING
    ):
        raise RuntimeError(
            "RNG ABI60 linear mapping, shape budget and 32-bit-word metadata are required"
        )
    if (
        not metadata["validation"]["passed"]
        or not metadata.get("workload_completed")
        or not metadata.get("final_validation", {}).get("passed")
    ):
        raise RuntimeError(
            "refusing to report timings for an incomplete or incorrect workload"
        )
    rows = _collect_task_rows(
        profile_dir, metadata["expected_kernel_tasks"], metadata["kernel_task_identity"]
    )
    if "public_device_kernel_symbols" in metadata:
        public_symbols = metadata["public_device_kernel_symbols"]
        public_blocks = metadata["public_actual_blocks"]
        _validate_task_identity(rows[:1], metadata, public_symbols, public_blocks)
        fixed_end = 1 + settings["warmup"] + settings["repeat"]
        _validate_task_identity(
            rows[1:fixed_end],
            metadata,
            metadata.get("device_kernel_symbols"),
            metadata["actual_blocks"],
        )
        if rows[fixed_end:]:
            _validate_task_identity(
                rows[fixed_end:], metadata, public_symbols, public_blocks
            )
    else:
        _validate_task_identity(
            rows,
            metadata,
            metadata.get("device_kernel_symbols"),
            metadata["actual_blocks"],
        )
    begin = metadata["excluded_kernel_tasks"]
    selected = rows[
        begin : begin + metadata.get("measured_kernel_tasks", len(rows) - begin)
    ]
    if (
        type(begin) is not int
        or begin < 0
        or not selected
        or len(selected) != metadata.get("measured_kernel_tasks")
    ):
        raise RuntimeError("incomplete quant measurement slice")
    durations = [row["duration_us"] for row in selected]
    with (case_dir / "task_rows.csv").open("w", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in selected for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    metadata["profiling"] = {
        "status": "collected",
        "metric": "msprof Task Duration(us)",
        "count": len(durations),
        "scope": metadata["measurement_scope"],
        "min_us": min(durations),
        "median_us": statistics.median(durations),
        "mean_us": statistics.mean(durations),
        "max_us": max(durations),
        "task_rows": str(case_dir / "task_rows.csv"),
        "profile_dir": str(profile_dir),
        "groups": sorted(
            {
                (
                    row.get("Op Name", ""),
                    row.get("OP Type", row.get("Op Type", "")),
                    row.get("Input Shapes", ""),
                )
                for row in selected
            }
        ),
        "excluded_initial_tasks": metadata["excluded_kernel_tasks"],
    }
    if "rng_overheads" in metadata:
        overhead = metadata["rng_overheads"]
        public_begin = overhead["public_kernel_begin"]
        public_selected = rows[public_begin : public_begin + overhead["measured_tasks"]]
        if len(public_selected) != overhead["measured_tasks"]:
            raise RuntimeError("incomplete default-public measurement slice")
        overhead["device_profiling"] = {
            "status": "collected",
            "metric": "msprof Task Duration(us)",
            "default_public_quant_kernel_only": _latencies(
                [row["duration_us"] for row in public_selected]
            ),
            "note": "Kernel-only timing excludes Host overhead; see public_synchronized",
        }
    _write_json(metadata_path, metadata)
    print(json.dumps({"case": str(case_dir), **metadata["profiling"]}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--shape", "--shapes", dest="shapes", nargs="+", action="extend", type=_shape
    )
    parser.add_argument("--src", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--dst", choices=("e4m3fn", "e5m2"), default="e4m3fn")
    parser.add_argument("--scale-alg", type=int, choices=(0, 1), default=1)
    parser.add_argument("--axis", type=int, default=-1)
    parser.add_argument("--max-low-bound", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument(
        "--rng-block-offset",
        type=int,
        default=0,
        help="replay offset in 32-bit words; must be four-aligned",
    )
    parser.add_argument(
        "--rng-overheads",
        action="store_true",
        help="separately measure default public calls with framework RNG reservation",
    )
    parser.add_argument(
        "--block-dim",
        type=int,
        default=None,
        help="explicit AIV block budget as a positive uint32; default: query selected device/stream",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--module-path", type=Path, default=_SR_SAMPLE_PATH)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="run a separate external msprof process for each shape",
    )
    parser.add_argument(
        "--summarize",
        type=Path,
        help="parse an existing profile directory for one completed case; no NPU execution",
    )
    args = parser.parse_args(argv)
    args.shapes = args.shapes or [(8, 4096)]
    if (
        not 0 <= args.rng_seed < 2**64
        or not 0 <= args.rng_block_offset <= (2**64 - 1) - RNG_INCREMENT
        or args.rng_block_offset % 4
    ):
        parser.error(
            "rng-seed must be uint64; RNG word offset must be four-aligned and allow a nonwrapping block"
        )
    if (
        args.warmup < 0
        or args.repeat < 1
        or (
            args.block_dim is not None
            and not 1 <= args.block_dim <= _GOLDEN_UINT32_MASK
        )
        or args.device < 0
    ):
        parser.error(
            "warmup must be >=0, repeat >=1, block-dim must be a positive uint32, and device >=0"
        )
    for shape in args.shapes:
        if not -len(shape) <= args.axis < len(shape):
            parser.error("axis is out of range for input dimensions")
        length = shape[args.axis]
        if length % _GOLDEN_RNG_GROUP_ELEMENTS or (
            args.src == "fp32" and length < _GOLDEN_MX_BLOCK_ELEMENTS
        ):
            parser.error("K must be a multiple of 16, and at least 32 for fp32")
    if (
        not math.isfinite(args.max_low_bound)
        or args.max_low_bound < 0
        or (args.scale_alg != 1 and args.max_low_bound != 0)
    ):
        parser.error(
            "max-low-bound must be finite/nonnegative and zero for scale-alg=0"
        )
    if len(set(args.shapes)) != len(args.shapes):
        parser.error("duplicate shapes would overwrite a case bundle")
    if args.summarize and (args.profile or len(args.shapes) != 1):
        parser.error(
            "--summarize accepts one shape and cannot be combined with --profile"
        )
    args.output_dir = args.output_dir.resolve()
    args.module_path = args.module_path.resolve(strict=True)
    if args.summarize:
        _summarize(
            args.output_dir / _case_name(args.shapes[0], args), args.summarize.resolve()
        )
        return
    if not args.profile:
        for shape in args.shapes:
            _run_case(shape, args)
        return
    msprof = shutil.which("msprof")
    if msprof is None:
        parser.error("msprof is unavailable; source the CANN environment first")
    for shape in args.shapes:
        name = _case_name(shape, args)
        profile_dir = args.output_dir / "profiles" / name
        profile_dir.mkdir(parents=True, exist_ok=False)
        command = [
            msprof,
            "--aic-metrics=PipeUtilization",
            "--task-time=on",
            "--ai-core=on",
            f"--output={profile_dir}",
            sys.executable,
            str(Path(__file__).resolve()),
            "--shape",
            "x".join(map(str, shape)),
            "--src",
            args.src,
            "--dst",
            args.dst,
            "--scale-alg",
            str(args.scale_alg),
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--axis",
            str(args.axis),
            "--max-low-bound",
            str(args.max_low_bound),
            "--rng-seed",
            str(args.rng_seed),
            "--rng-block-offset",
            str(args.rng_block_offset),
            "--device",
            str(args.device),
            "--output-dir",
            str(args.output_dir),
            "--module-path",
            str(args.module_path),
        ]
        if args.block_dim is not None:
            command.extend(["--block-dim", str(args.block_dim)])
        if args.rng_overheads:
            command.append("--rng-overheads")
        _write_json(profile_dir / "command.json", {"argv": command})
        with (args.output_dir / f"{name}.msprof.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        _summarize(args.output_dir / name, profile_dir)


if __name__ == "__main__":
    main()
