# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Input contracts, runtime adapters and inlined MX quantization primitives."""

import inspect

import cannbotdsl as cb
import torch
import torch_npu
from cannbotdsl import dtypes as dt
from cannbotdsl import simt
from cannbotdsl.buffer import Buffer as _Buffer
from cannbotdsl.ring_buffer import RingBuffer as _RingBuffer
from cannbotdsl.tensor import make_layout

# Fixed model dimensions.
HIDDEN_SIZE = 5120
Q_LORA_RANK = 1280
NUM_HEADS = 32
HEAD_DIM = 128
QUERY_DIM = NUM_HEADS * HEAD_DIM
PACKED_HEAD_DIM = HEAD_DIM // 2
# MX scale layout: two group-32 E8M0 bytes per pair.
MX_GROUP_SIZE = 32
MX_SCALE_PAIR = 2
MX_PAIR_ELEMENTS = MX_GROUP_SIZE * MX_SCALE_PAIR
INPUT_SCALE_PAIRS = HIDDEN_SIZE // MX_PAIR_ELEMENTS
QUERY_SCALE_PAIRS = Q_LORA_RANK // MX_PAIR_ELEMENTS
HEAD_SCALE_GROUPS = HEAD_DIM // MX_GROUP_SIZE
# Scheduling and supported-size limits.
PROJECTION_CORE_WEIGHTS = (27, 5)
VECTOR_SUBBLOCKS = 2
INPUT_QUANT_TILE = 1024
SIMT_THREADS = 256
SCORE_TILE_TOKENS = 256
TOPK_ALIGNMENT = 64
MAX_KV_LENGTH = 16384
# Candidate blocks and PA cache storage.
CANDIDATE_BLOCK_SIZE = 8
CANDIDATE_CAPACITY = 2048
PA_KEY_BLOCK_BYTES = CANDIDATE_BLOCK_SIZE * PACKED_HEAD_DIM
PA_SCALE_BLOCK_BYTES = CANDIDATE_BLOCK_SIZE * HEAD_SCALE_GROUPS
PA_RECORD_BYTES = PA_KEY_BLOCK_BYTES + PA_SCALE_BLOCK_BYTES
PA_PAGE_ALIGNMENT = 16
MAX_PA_PAGE_SIZE = 1024
FRACTAL_NZ = 29
INT32_BYTES = torch.iinfo(torch.int32).bits // 8
METADATA_UB_FRACTION = 64
BF16_NEG_INF_BITS = 0xFF80
SCALE_DTYPE = torch.float8_e8m0fnu
SCALE_DTYPES = (SCALE_DTYPE, torch.uint8)
DTYPES = {
    SCALE_DTYPE: dt.float8_e8m0,
    torch.float8_e4m3fn: dt.float8_e4m3fn,
    torch.bfloat16: dt.bfloat16,
    torch.float32: dt.float32,
    torch.int32: dt.int32,
    torch.int8: dt.int8,
    torch.uint8: dt.uint8,
}


def check(tensor, name, shape, dtype, device):
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name}: expected {tuple(shape)}, got {tuple(tensor.shape)}")
    if tensor.dtype not in (dtype if isinstance(dtype, tuple) else (dtype,)):
        raise TypeError(f"{name}: expected {dtype}, got {tensor.dtype}")
    if tensor.device != device or device.type != "npu":
        raise ValueError(f"{name}: all tensors must be on the same NPU")
    if not tensor.is_contiguous():
        raise ValueError(f"{name}: contiguous storage required")


def check_platform():
    """Return the Cube quota supported by both engines on the current stream."""
    platform = cb.get_platform_info(
        force_refresh=True, stream=torch.npu.current_stream()
    )
    cores = min(platform.cube_core_num, platform.vector_core_num // VECTOR_SUBBLOCKS)
    if cores < 2:
        raise RuntimeError("MegaReindex needs at least 2 Cube / 4 Vector cores")
    return cores


def projection_core_counts(cores):
    """Scale the validated projection split without constraining launch size."""
    query_weight, weight_weight = PROJECTION_CORE_WEIGHTS
    total = query_weight + weight_weight
    weight_cores = max(1, min(cores - 1, (cores * weight_weight + total // 2) // total))
    return cores - weight_cores, weight_cores


def check_nz(tensor, name):
    if torch_npu.get_npu_format(tensor) != FRACTAL_NZ:
        raise ValueError(f"{name}: preprocessed FRACTAL_NZ (format 29) required")


def signature(args):
    return tuple(
        ("tensor", tuple(x.shape), x.dtype, tuple(x.stride()))
        if isinstance(x, torch.Tensor)
        else ("int64",)
        for x in args
    )


_LAYOUT_API = "layout" in inspect.signature(_RingBuffer).parameters


class _TransposedRing:
    """Expose a logical ZN channel backed by a transposed NZ allocation."""

    def __init__(self, channel):
        self.channel = channel

    def next_write(self):
        return self.channel.next_write().permute(1, 0)

    def next_read(self):
        return self.channel.next_read().permute(1, 0)


def RingBuffer(mem_loc, shape, dtype, *, depth, data_format=None):
    if not _LAYOUT_API:
        return _RingBuffer(mem_loc, shape, dtype, depth=depth, data_format=data_format)
    if data_format == "zn":
        channel = _RingBuffer(
            mem_loc,
            dtype,
            make_layout(tuple(reversed(shape))),
            depth=depth,
            data_format="nz",
        )
        return _TransposedRing(channel)
    return _RingBuffer(
        mem_loc, dtype, make_layout(shape), depth=depth, data_format=data_format
    )


def Buffer(mem_loc, shape, dtype, *, data_format=None):
    if not _LAYOUT_API:
        return _Buffer(mem_loc, shape, dtype, data_format=data_format)
    if data_format == "zn":
        return _Buffer(
            mem_loc, dtype, make_layout(tuple(reversed(shape))), data_format="nz"
        ).permute(1, 0)
    return _Buffer(mem_loc, dtype, make_layout(shape), data_format=data_format)


class ResourceCheckedProgram:
    """Select a cached schedule for the input device's current stream quota."""

    def __init__(self, program, device, core_count, compiler):
        self._program = program
        self._device = device
        self._core_count = core_count
        self._compiler = compiler

    def __call__(self, *args, **kwargs):
        with torch.npu.device(self._device):
            cores = check_platform()
            if cores != self._core_count:
                self._program = self._compiler(cores)
                self._core_count = cores
            return self._program(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._program, name)


# IEEE FP32 fields and FP8 E4M3 encoding constants.
FLOAT32_MANTISSA_BITS = 23
FLOAT32_MANTISSA_UNIT = 1 << FLOAT32_MANTISSA_BITS
FLOAT32_EXPONENT_VALUES = 1 << 8
FLOAT32_SIGN_BIT = 1 << 31
FP8_MANTISSA_BITS = 3
FP8_SIGN_BIT = 1 << 7
FP8_MAX = 448.0
FP8_AMAX_FLOOR = 1e-4
FP8_MIN_NORMAL = 2.0**-6
FP8_SUBNORMAL_SCALE = 2.0**9
# Drop 20 mantissa bits; halfway-minus-one plus retained LSB implements RNE.
FP8_ROUND_UNIT = 1 << (FLOAT32_MANTISSA_BITS - FP8_MANTISSA_BITS)
FP8_ROUND_BIAS = FP8_ROUND_UNIT // 2 - 1
FP8_EXPONENT_BIAS_OFFSET = (127 - 7) * (1 << FP8_MANTISSA_BITS)


@cb.jit
def ceil_mx_scale(scaled):
    """Return E8M0 exponent and FP32 scale, rounding positive scaled upward."""
    bits = dt.int32(simt.bitcast(scaled, dtype=dt.int32))
    exponent = (bits // FLOAT32_MANTISSA_UNIT) % FLOAT32_EXPONENT_VALUES
    if (bits % FLOAT32_MANTISSA_UNIT) != 0:
        exponent = exponent + 1
    scale = simt.bitcast(exponent * FLOAT32_MANTISSA_UNIT, dtype=dt.float32)
    return exponent, scale


@cb.jit
def encode_fp8_e4m3(z):
    """Encode FP8 E4M3 with the original saturation and ties-to-even rules."""
    z = simt.fmin(simt.fmax(z, dt.float32(-FP8_MAX)), dt.float32(FP8_MAX))
    az = simt.abs(z)
    zb = dt.int32(simt.bitcast(az, dtype=dt.int32))
    code = dt.int32(0)
    if az >= FP8_MIN_NORMAL:
        rounded = (zb + FP8_ROUND_BIAS + ((zb // FP8_ROUND_UNIT) % 2)) // FP8_ROUND_UNIT
        code = dt.int32(rounded) - FP8_EXPONENT_BIAS_OFFSET
    else:
        subnormal = az * dt.float32(FP8_SUBNORMAL_SCALE)
        code = dt.int32(subnormal)
        fraction = subnormal - dt.float32(code)
        if (fraction > 0.5) or ((fraction == 0.5) and (code % 2 == 1)):
            code = code + 1
    if dt.uint32(simt.bitcast(z, dtype=dt.uint32)) >= dt.uint32(FLOAT32_SIGN_BIT):
        code = code + FP8_SIGN_BIT
    return dt.uint8(code)


@cb.jit
def encode_fp4_e2m1(z):
    """Encode E2M1 RNE; alternating > and >= preserve midpoint tie behavior."""
    magnitude = simt.abs(z)
    code = dt.uint8(0)
    if magnitude > 0.25:
        code = dt.uint8(1)
    if magnitude >= 0.75:
        code = dt.uint8(2)
    if magnitude > 1.25:
        code = dt.uint8(3)
    if magnitude >= 1.75:
        code = dt.uint8(4)
    if magnitude > 2.5:
        code = dt.uint8(5)
    if magnitude >= 3.5:
        code = dt.uint8(6)
    if magnitude > 5.0:
        code = dt.uint8(7)
    if z < 0.0:
        code = code + dt.uint8(8)
    return code


@cb.jit
def quantize_fp8_group(value):
    """Return FP8 bits and E8M0 exponent for one participating group-32 lane.

    Call from a SIMT region with all 32 lanes of each MX group active.
    Loading, indexing, leader-only scale stores and DMA stay in the caller.
    """
    amax = simt.reduce_max(simt.abs(value))
    amax = simt.fmax(amax, dt.float32(FP8_AMAX_FLOOR))
    scaled = amax * dt.float32(1.0 / FP8_MAX)
    exponent, scale = ceil_mx_scale(scaled)
    z = value / scale
    return encode_fp8_e4m3(z), exponent
