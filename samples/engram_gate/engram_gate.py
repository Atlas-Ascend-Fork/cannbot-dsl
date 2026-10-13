# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CANNBot-DSL implementation of the Engram residual gate."""

import math

import torch

from cannbotdsl import (
    MemLoc,
    Tensor,
    TensorSpec,
    compile as dsl_compile,
    const_expr,
    dtypes,
)
from cannbotdsl.aot import export
from cannbotdsl.buffer import Buffer
from cannbotdsl.tensor import tile_slice
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.info import get_mem_size, get_platform_info
from cannbotdsl.ops.memcpy import mem_copy

__all__ = ["EngramGate", "EngramGateKernel", "engram_gate"]


VL = 64  # fp32 lanes in one 256-byte raw vector register
BYTES_PER_ELEMENT_RESIDENT = 30  # 11 BF16 row buffers + out pair + FP32 weight/x caches
BYTES_PER_ELEMENT_CHUNKED = 12  # 4 BF16 chunk buffers + one FP32 weight chunk buffer
CHUNKED_UB_RESERVE = 8192  # FP32 accumulators, gate buffer and alignment slack
PAIRS_PER_GROUP = 3  # row pairs unrolled per pipeline group


def _vector_core_count() -> int:
    info = get_platform_info()
    count = info.vector_core_num
    if not isinstance(count, int) or count <= 0:
        raise RuntimeError(
            "Cannot determine the NPU vector core count "
            f"(platform info available={info.available}, "
            f"vector_core_num={count!r}); "
            "pass max_blocks explicitly to EngramGate"
        )
    return count


@kernel
class EngramGateKernel:
    """One vector task per ``(token, hc)`` row."""

    def __init__(self, total_tokens: int, hc_mult: int, dim: int):
        self.total_tokens = int(total_tokens)
        self.hc_mult = int(hc_mult)
        self.dim = int(dim)
        self.num_full = self.dim // VL  # number of full 64-lane segments per row
        self.tail = self.dim - self.num_full * VL  # lanes in the trailing partial segment
        self.w = self.dim + (VL - self.tail) % VL  # padded row width in elements, a multiple of VL

        ub_capacity = get_mem_size("ub")
        if self.w * BYTES_PER_ELEMENT_RESIDENT <= ub_capacity:
            self.chunked = False
            self.x_a = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.x_b = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.x_c = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.key_a = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.key_b = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.key_c = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.value_a = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.value_b = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.value_c = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.weight_ub = Buffer(MemLoc.UB, (self.w,), dtypes.float32)
            self.out_a = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.out_b = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.x_fp32_ub = Buffer(MemLoc.UB, (self.w,), dtypes.float32)
            self.x_ub = self.x_a
            self.key_ub = self.key_a
            self.value_ub = self.value_a
            self.out_ub = self.out_a
        else:
            budget = ub_capacity - CHUNKED_UB_RESERVE
            max_chunk = budget // BYTES_PER_ELEMENT_CHUNKED // VL * VL
            if max_chunk < VL:
                raise RuntimeError(
                    f"EngramGateKernel needs at least one {VL}-element chunk "
                    f"({BYTES_PER_ELEMENT_CHUNKED} B/elem) but runtime reports {ub_capacity} B UB"
                )
            self.chunked = True
            self.chunk = min(self.w, int(max_chunk))
            self.num_chunks = (self.w + self.chunk - 1) // self.chunk
            self.last_len = self.w - (self.num_chunks - 1) * self.chunk
            self.last_gm_len = self.dim - (self.num_chunks - 1) * self.chunk
            self.last_full = self.last_gm_len // VL
            self.x_ch = Buffer(MemLoc.UB, (self.chunk,), dtypes.bfloat16)
            self.key_ch = Buffer(MemLoc.UB, (self.chunk,), dtypes.bfloat16)
            self.value_ch = Buffer(MemLoc.UB, (self.chunk,), dtypes.bfloat16)
            self.weight_ch = Buffer(MemLoc.UB, (self.chunk,), dtypes.float32)
            self.out_ch = Buffer(MemLoc.UB, (self.chunk,), dtypes.bfloat16)
            self.acc_x2 = Buffer(MemLoc.UB, (VL,), dtypes.float32)
            self.acc_k2 = Buffer(MemLoc.UB, (VL,), dtypes.float32)
            self.acc_dot = Buffer(MemLoc.UB, (VL,), dtypes.float32)
            self.gate_buf = Buffer(MemLoc.UB, (VL,), dtypes.float32)

    @jit
    def _compute(self, epsilon: dtypes.float32, clamp_value: dtypes.float32):
        dim = self.dim
        with vf(mode="simd"):
            full_mask = rr.full_mask()
            lane0 = rr.update_mask(1, elem_bits=32)[0]
            x_square_acc = rr.vdups(0.0, dtypes.float32, mask=full_mask)
            key_square_acc = rr.vdups(0.0, dtypes.float32, mask=full_mask)
            dot_acc = rr.vdups(0.0, dtypes.float32, mask=full_mask)

            for segment in range(self.num_full):
                segment_offset = segment * VL
                x = rr.vcast(
                    rr.vload_unpack(
                        self.x_ub, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full_mask,
                )
                rr.vstore(self.x_fp32_ub, segment_offset, x, full_mask)
                key = rr.vcast(
                    rr.vload_unpack(
                        self.key_ub, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full_mask,
                )
                weight = rr.vload(self.weight_ub, segment_offset)
                x_square_acc = rr.vadd(
                    x_square_acc,
                    rr.vmul(x, x, mask=full_mask),
                    mask=full_mask,
                )
                key_square_acc = rr.vadd(
                    key_square_acc,
                    rr.vmul(key, key, mask=full_mask),
                    mask=full_mask,
                )
                dot_acc = rr.vadd(
                    dot_acc,
                    rr.vmul(rr.vmul(x, weight, mask=full_mask), key, mask=full_mask),
                    mask=full_mask,
                )

            x_square_sum = rr.vreduce_sum(x_square_acc, mask=full_mask)
            key_square_sum = rr.vreduce_sum(key_square_acc, mask=full_mask)
            dot_sum = rr.vreduce_sum(dot_acc, mask=full_mask)

            if const_expr(self.tail != 0):
                tail_mask = rr.update_mask(self.tail, elem_bits=32)[0]
                tail_offset = self.num_full * VL
                x_t = rr.vcast(
                    rr.vload_unpack(
                        self.x_ub, tail_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=tail_mask,
                )
                key_t = rr.vcast(
                    rr.vload_unpack(
                        self.key_ub, tail_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=tail_mask,
                )
                weight_t = rr.vload(self.weight_ub, tail_offset)
                x_square_sum = rr.vadd(
                    x_square_sum,
                    rr.vreduce_sum(rr.vmul(x_t, x_t, mask=tail_mask), mask=tail_mask),
                    mask=lane0,
                )
                key_square_sum = rr.vadd(
                    key_square_sum,
                    rr.vreduce_sum(rr.vmul(key_t, key_t, mask=tail_mask), mask=tail_mask),
                    mask=lane0,
                )
                dot_sum = rr.vadd(
                    dot_sum,
                    rr.vreduce_sum(
                        rr.vmul(
                            rr.vmul(x_t, weight_t, mask=tail_mask), key_t, mask=tail_mask
                        ),
                        mask=tail_mask,
                    ),
                    mask=lane0,
                )
                rr.vstore(self.x_fp32_ub, tail_offset, x_t, tail_mask)

            one = rr.vdups(1.0, dtypes.float32, mask=lane0)
            x_rms = rr.vsqrt(
                rr.vadds(
                    rr.vmuls(x_square_sum, 1.0 / dim, mask=lane0),
                    epsilon,
                    mask=lane0,
                ),
                mask=lane0,
            )
            key_rms = rr.vsqrt(
                rr.vadds(
                    rr.vmuls(key_square_sum, 1.0 / dim, mask=lane0),
                    epsilon,
                    mask=lane0,
                ),
                mask=lane0,
            )
            normalized_dot = rr.vdiv(
                rr.vmuls(dot_sum, 1.0 / math.sqrt(dim), mask=lane0),
                rr.vmul(x_rms, key_rms, mask=lane0),
                mask=lane0,
            )

            magnitude = rr.vsqrt(
                rr.vmaxs(rr.vabs(normalized_dot, mask=lane0), clamp_value, mask=lane0),
                mask=lane0,
            )
            negative = rr.vlts(normalized_dot, 0.0, mask=lane0)
            signed_root = rr.vselect(
                rr.vmuls(magnitude, -1.0, mask=lane0),
                magnitude,
                cond_mask=negative,
            )
            gate = rr.vdiv(
                one,
                rr.vadd(
                    one,
                    rr.vexp(rr.vmuls(signed_root, -1.0, mask=lane0), mask=lane0),
                    mask=lane0,
                ),
                mask=lane0,
            )
            gate = rr.vdup(gate, mask=full_mask)

            for segment in range(self.num_full):
                segment_offset = segment * VL
                x = rr.vload(self.x_fp32_ub, segment_offset)
                value = rr.vcast(
                    rr.vload_unpack(
                        self.value_ub, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full_mask,
                )
                out = rr.vadd(x, rr.vmul(gate, value, mask=full_mask), mask=full_mask)
                rr.vstore_pack(
                    self.out_ub,
                    segment_offset,
                    rr.vcast(out, dtypes.bfloat16, mask=full_mask),
                    full_mask,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            if const_expr(self.tail != 0):
                tail_mask = rr.update_mask(self.tail, elem_bits=32)[0]
                tail_offset = self.num_full * VL
                x_t = rr.vload(self.x_fp32_ub, tail_offset)
                value_t = rr.vcast(
                    rr.vload_unpack(
                        self.value_ub, tail_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=tail_mask,
                )
                out_t = rr.vadd(x_t, rr.vmul(gate, value_t, mask=tail_mask), mask=tail_mask)
                rr.vstore_pack(
                    self.out_ub,
                    tail_offset,
                    rr.vcast(out_t, dtypes.bfloat16, mask=tail_mask),
                    tail_mask,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def _acc_init(self):
        with vf(mode="simd"):
            full_mask = rr.full_mask()
            zero_reg = rr.vdups(0.0, dtypes.float32, mask=full_mask)
            rr.vstore(self.acc_x2, 0, zero_reg, full_mask)
            rr.vstore(self.acc_k2, 0, zero_reg, full_mask)
            rr.vstore(self.acc_dot, 0, zero_reg, full_mask)

    @jit
    def _acc_full_chunk(self):
        with vf(mode="simd"):
            full_mask = rr.full_mask()
            lane0 = rr.update_mask(1, elem_bits=32)[0]
            x_acc = rr.vdups(0.0, dtypes.float32, mask=full_mask)
            k_acc = rr.vdups(0.0, dtypes.float32, mask=full_mask)
            d_acc = rr.vdups(0.0, dtypes.float32, mask=full_mask)
            for segment in range(self.chunk // VL):
                segment_offset = segment * VL
                x = rr.vcast(
                    rr.vload_unpack(
                        self.x_ch, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full_mask,
                )
                key = rr.vcast(
                    rr.vload_unpack(
                        self.key_ch, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full_mask,
                )
                weight = rr.vload(self.weight_ch, segment_offset)
                x_acc = rr.vadd(x_acc, rr.vmul(x, x, mask=full_mask), mask=full_mask)
                k_acc = rr.vadd(k_acc, rr.vmul(key, key, mask=full_mask), mask=full_mask)
                d_acc = rr.vadd(
                    d_acc,
                    rr.vmul(rr.vmul(x, weight, mask=full_mask), key, mask=full_mask),
                    mask=full_mask,
                )
            rr.vstore(
                self.acc_x2, 0,
                rr.vadd(rr.vload(self.acc_x2, 0), rr.vreduce_sum(x_acc, mask=full_mask), mask=lane0),
                full_mask,
            )
            rr.vstore(
                self.acc_k2, 0,
                rr.vadd(rr.vload(self.acc_k2, 0), rr.vreduce_sum(k_acc, mask=full_mask), mask=lane0),
                full_mask,
            )
            rr.vstore(
                self.acc_dot, 0,
                rr.vadd(rr.vload(self.acc_dot, 0), rr.vreduce_sum(d_acc, mask=full_mask), mask=lane0),
                full_mask,
            )

    @jit
    def _acc_last_chunk(self):
        with vf(mode="simd"):
            full_mask = rr.full_mask()
            lane0 = rr.update_mask(1, elem_bits=32)[0]
            x_acc = rr.vdups(0.0, dtypes.float32, mask=full_mask)
            k_acc = rr.vdups(0.0, dtypes.float32, mask=full_mask)
            d_acc = rr.vdups(0.0, dtypes.float32, mask=full_mask)
            for segment in range(self.last_full):
                segment_offset = segment * VL
                x = rr.vcast(
                    rr.vload_unpack(
                        self.x_ch, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full_mask,
                )
                key = rr.vcast(
                    rr.vload_unpack(
                        self.key_ch, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full_mask,
                )
                weight = rr.vload(self.weight_ch, segment_offset)
                x_acc = rr.vadd(x_acc, rr.vmul(x, x, mask=full_mask), mask=full_mask)
                k_acc = rr.vadd(k_acc, rr.vmul(key, key, mask=full_mask), mask=full_mask)
                d_acc = rr.vadd(
                    d_acc,
                    rr.vmul(rr.vmul(x, weight, mask=full_mask), key, mask=full_mask),
                    mask=full_mask,
                )
            if const_expr(self.tail != 0):
                tail_mask = rr.update_mask(self.tail, elem_bits=32)[0]
                x_t = rr.vcast(
                    rr.vload_unpack(
                        self.x_ch, self.last_full * VL,
                        unpack_mode=rr.UnpackMode.B16_TO_B32,
                    ),
                    dtypes.float32,
                    mask=tail_mask,
                )
                key_t = rr.vcast(
                    rr.vload_unpack(
                        self.key_ch, self.last_full * VL,
                        unpack_mode=rr.UnpackMode.B16_TO_B32,
                    ),
                    dtypes.float32,
                    mask=tail_mask,
                )
                weight_t = rr.vload(self.weight_ch, self.last_full * VL)
                x_sum_t = rr.vreduce_sum(rr.vmul(x_t, x_t, mask=tail_mask), mask=tail_mask)
                k_sum_t = rr.vreduce_sum(rr.vmul(key_t, key_t, mask=tail_mask), mask=tail_mask)
                d_sum_t = rr.vreduce_sum(
                    rr.vmul(rr.vmul(x_t, weight_t, mask=tail_mask), key_t, mask=tail_mask),
                    mask=tail_mask,
                )
                rr.vstore(
                    self.acc_x2, 0,
                    rr.vadd(
                        rr.vadd(rr.vload(self.acc_x2, 0), rr.vreduce_sum(x_acc, mask=full_mask), mask=lane0),
                        x_sum_t, mask=lane0,
                    ),
                    full_mask,
                )
                rr.vstore(
                    self.acc_k2, 0,
                    rr.vadd(
                        rr.vadd(rr.vload(self.acc_k2, 0), rr.vreduce_sum(k_acc, mask=full_mask), mask=lane0),
                        k_sum_t, mask=lane0,
                    ),
                    full_mask,
                )
                rr.vstore(
                    self.acc_dot, 0,
                    rr.vadd(
                        rr.vadd(rr.vload(self.acc_dot, 0), rr.vreduce_sum(d_acc, mask=full_mask), mask=lane0),
                        d_sum_t, mask=lane0,
                    ),
                    full_mask,
                )
            else:
                rr.vstore(
                    self.acc_x2, 0,
                    rr.vadd(rr.vload(self.acc_x2, 0), rr.vreduce_sum(x_acc, mask=full_mask), mask=lane0),
                    full_mask,
                )
                rr.vstore(
                    self.acc_k2, 0,
                    rr.vadd(rr.vload(self.acc_k2, 0), rr.vreduce_sum(k_acc, mask=full_mask), mask=lane0),
                    full_mask,
                )
                rr.vstore(
                    self.acc_dot, 0,
                    rr.vadd(rr.vload(self.acc_dot, 0), rr.vreduce_sum(d_acc, mask=full_mask), mask=lane0),
                    full_mask,
                )

    @jit
    def _finalize_gate(self, epsilon: dtypes.float32, clamp_value: dtypes.float32):
        dim = self.dim
        with vf(mode="simd"):
            full_mask = rr.full_mask()
            lane0 = rr.update_mask(1, elem_bits=32)[0]
            x_square_sum = rr.vload(self.acc_x2, 0)
            key_square_sum = rr.vload(self.acc_k2, 0)
            dot_sum = rr.vload(self.acc_dot, 0)
            one = rr.vdups(1.0, dtypes.float32, mask=lane0)
            x_rms = rr.vsqrt(
                rr.vadds(rr.vmuls(x_square_sum, 1.0 / dim, mask=lane0), epsilon, mask=lane0),
                mask=lane0,
            )
            key_rms = rr.vsqrt(
                rr.vadds(rr.vmuls(key_square_sum, 1.0 / dim, mask=lane0), epsilon, mask=lane0),
                mask=lane0,
            )
            normalized_dot = rr.vdiv(
                rr.vmuls(dot_sum, 1.0 / math.sqrt(dim), mask=lane0),
                rr.vmul(x_rms, key_rms, mask=lane0),
                mask=lane0,
            )
            magnitude = rr.vsqrt(
                rr.vmaxs(rr.vabs(normalized_dot, mask=lane0), clamp_value, mask=lane0),
                mask=lane0,
            )
            negative = rr.vlts(normalized_dot, 0.0, mask=lane0)
            signed_root = rr.vselect(
                rr.vmuls(magnitude, -1.0, mask=lane0), magnitude, cond_mask=negative
            )
            gate = rr.vdiv(
                one,
                rr.vadd(
                    one,
                    rr.vexp(rr.vmuls(signed_root, -1.0, mask=lane0), mask=lane0),
                    mask=lane0,
                ),
                mask=lane0,
            )
            rr.vstore(self.gate_buf, 0, rr.vdup(gate, mask=full_mask), full_mask)

    @jit
    def _emit_full_chunk(self):
        with vf(mode="simd"):
            full_mask = rr.full_mask()
            gate = rr.vload(self.gate_buf, 0)
            for segment in range(self.chunk // VL):
                segment_offset = segment * VL
                x = rr.vcast(
                    rr.vload_unpack(
                        self.x_ch, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full_mask,
                )
                value = rr.vcast(
                    rr.vload_unpack(
                        self.value_ch, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full_mask,
                )
                out = rr.vadd(x, rr.vmul(gate, value, mask=full_mask), mask=full_mask)
                rr.vstore_pack(
                    self.out_ch,
                    segment_offset,
                    rr.vcast(out, dtypes.bfloat16, mask=full_mask),
                    full_mask,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def _emit_last_chunk(self):
        with vf(mode="simd"):
            full_mask = rr.full_mask()
            gate = rr.vload(self.gate_buf, 0)
            for segment in range(self.last_full):
                segment_offset = segment * VL
                x = rr.vcast(
                    rr.vload_unpack(
                        self.x_ch, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full_mask,
                )
                value = rr.vcast(
                    rr.vload_unpack(
                        self.value_ch, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full_mask,
                )
                out = rr.vadd(x, rr.vmul(gate, value, mask=full_mask), mask=full_mask)
                rr.vstore_pack(
                    self.out_ch,
                    segment_offset,
                    rr.vcast(out, dtypes.bfloat16, mask=full_mask),
                    full_mask,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            if const_expr(self.tail != 0):
                tail_mask = rr.update_mask(self.tail, elem_bits=32)[0]
                x_t = rr.vcast(
                    rr.vload_unpack(
                        self.x_ch, self.last_full * VL,
                        unpack_mode=rr.UnpackMode.B16_TO_B32,
                    ),
                    dtypes.float32,
                    mask=tail_mask,
                )
                value_t = rr.vcast(
                    rr.vload_unpack(
                        self.value_ch, self.last_full * VL,
                        unpack_mode=rr.UnpackMode.B16_TO_B32,
                    ),
                    dtypes.float32,
                    mask=tail_mask,
                )
                out_t = rr.vadd(x_t, rr.vmul(gate, value_t, mask=tail_mask), mask=tail_mask)
                rr.vstore_pack(
                    self.out_ch,
                    self.last_full * VL,
                    rr.vcast(out_t, dtypes.bfloat16, mask=tail_mask),
                    tail_mask,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def _init_rotation_buffers(self):
        with vf(mode="simd"):
            full_mask = rr.full_mask()
            zero_reg = rr.vcast(
                rr.vdups(0.0, dtypes.float32, mask=full_mask),
                dtypes.bfloat16,
                mask=full_mask,
            )
            rr.vstore_pack(self.x_b, 0, zero_reg, full_mask, pack_mode=rr.PackMode.B32_TO_B16)
            rr.vstore_pack(self.x_c, 0, zero_reg, full_mask, pack_mode=rr.PackMode.B32_TO_B16)
            rr.vstore_pack(self.key_b, 0, zero_reg, full_mask, pack_mode=rr.PackMode.B32_TO_B16)
            rr.vstore_pack(self.key_c, 0, zero_reg, full_mask, pack_mode=rr.PackMode.B32_TO_B16)
            rr.vstore_pack(self.value_b, 0, zero_reg, full_mask, pack_mode=rr.PackMode.B32_TO_B16)
            rr.vstore_pack(self.value_c, 0, zero_reg, full_mask, pack_mode=rr.PackMode.B32_TO_B16)

    @jit
    def _chunked_row(self, x_row, key_row, w_row, v_row, o_row,
                     epsilon: dtypes.float32, clamp_value: dtypes.float32):
        """Accumulate and emit one (token, hc) row through the chunk buffers."""
        self._acc_init()
        for chunk_idx in range(self.num_chunks):
            mem_copy(self.x_ch, tile_slice(x_row, (1, self.chunk), (0, chunk_idx)))
            mem_copy(self.key_ch, tile_slice(key_row, (1, self.chunk), (0, chunk_idx)))
            mem_copy(self.weight_ch, tile_slice(w_row, (self.chunk,), (chunk_idx,)))
            if chunk_idx + 1 < self.num_chunks:
                self._acc_full_chunk()
            else:
                self._acc_last_chunk()
        self._finalize_gate(epsilon, clamp_value)
        for chunk_idx in range(self.num_chunks):
            if self.num_chunks > 1:
                mem_copy(self.x_ch, tile_slice(x_row, (1, self.chunk), (0, chunk_idx)))
            mem_copy(self.value_ch, tile_slice(v_row, (self.chunk,), (chunk_idx,)))
            if chunk_idx + 1 < self.num_chunks:
                self._emit_full_chunk()
            else:
                self._emit_last_chunk()
            mem_copy(tile_slice(o_row, (1, self.chunk), (0, chunk_idx)), self.out_ch)

    @jit
    def _load_row(self, slot, gm_x, gm_key, gm_value, gm_mask, token, hc):
        """Prefetch one row into a rotation slot, trimmed by the mask."""
        if gm_mask is None:
            mem_copy(slot[2], gm_value[token, None])
            mem_copy(slot[0], gm_x[token, hc, None])
            mem_copy(slot[1], gm_key[token, hc, None])
        else:
            masked = gm_mask[token]
            if masked:
                mem_copy(slot[0], gm_x[token, hc, None])
            else:
                mem_copy(slot[2], gm_value[token, None])
                mem_copy(slot[0], gm_x[token, hc, None])
                mem_copy(slot[1], gm_key[token, hc, None])

    @jit
    def _emit_row(self, gm_out, gm_mask, gm_weight, token, hc,
                  epsilon: dtypes.float32, clamp_value: dtypes.float32,
                  weight_resident):
        """Compute (or pass through) the bound row and store it."""
        if gm_mask is None:
            if not weight_resident:
                mem_copy(self.weight_ub, gm_weight[hc, None])
            self._compute(epsilon, clamp_value)
            mem_copy(gm_out[token, hc, None], self.out_ub)
        else:
            masked = gm_mask[token]
            if masked:
                mem_copy(self.out_ub, self.x_ub)
                mem_copy(gm_out[token, hc, None], self.out_ub)
            else:
                if not weight_resident:
                    mem_copy(self.weight_ub, gm_weight[hc, None])
                self._compute(epsilon, clamp_value)
                mem_copy(gm_out[token, hc, None], self.out_ub)

    @jit
    def _prologue(self, gm_x, gm_key, gm_value, gm_weight, gm_mask,
                  first_token, first_hc, weight_resident):
        """Preload the first row into slot a before the pair pipeline."""
        if gm_mask is None:
            mem_copy(self.x_a, gm_x[first_token, first_hc, None])
            mem_copy(self.key_a, gm_key[first_token, first_hc, None])
            mem_copy(self.weight_ub, gm_weight[first_hc, None])
            mem_copy(self.value_a, gm_value[first_token, None])
        else:
            first_masked = gm_mask[first_token]
            if first_masked:
                mem_copy(self.x_a, gm_x[first_token, first_hc, None])
                if weight_resident:
                    mem_copy(self.weight_ub, gm_weight[first_hc, None])
            else:
                mem_copy(self.x_a, gm_x[first_token, first_hc, None])
                mem_copy(self.key_a, gm_key[first_token, first_hc, None])
                mem_copy(self.weight_ub, gm_weight[first_hc, None])
                mem_copy(self.value_a, gm_value[first_token, None])

    @jit
    def _pair_step(self, gm_x, gm_key, gm_value, gm_weight, gm_out, gm_mask,
                   row0, rows, block_num, pair_stride,
                   epsilon: dtypes.float32, clamp_value: dtypes.float32,
                   weight_resident, s0, s1, s2,
                   load_s0, prefetch_row2, guard_row2):
        """Run one (row0, row0 + block_num) pair of the rotation pipeline.

        s0/s1/s2 are (x, key, value) buffer triples for the current row, the
        pair's second row, and the next pair's first row.  load_s0 pulls the
        current row into s0 here (remainder tail only); prefetch_row2 and
        guard_row2 control the next-pair prefetch, which is skipped or
        bounds-guarded at the pipeline tail.
        """
        row1 = row0 + block_num
        row2 = row0 + pair_stride
        token0 = row0 // self.hc_mult
        token1 = row1 // self.hc_mult
        token2 = row2 // self.hc_mult
        hc0 = row0 - token0 * self.hc_mult
        hc1 = row1 - token1 * self.hc_mult
        hc2 = row2 - token2 * self.hc_mult
        if load_s0:
            self._load_row(s0, gm_x, gm_key, gm_value, gm_mask, token0, hc0)
        if row1 < rows:
            self._load_row(s1, gm_x, gm_key, gm_value, gm_mask, token1, hc1)
        self.x_ub = s0[0]
        self.key_ub = s0[1]
        self.value_ub = s0[2]
        self.out_ub = self.out_a
        self._emit_row(gm_out, gm_mask, gm_weight, token0, hc0,
                       epsilon, clamp_value, weight_resident)
        if prefetch_row2:
            if guard_row2:
                if row2 < rows:
                    self._load_row(s2, gm_x, gm_key, gm_value, gm_mask, token2, hc2)
            else:
                self._load_row(s2, gm_x, gm_key, gm_value, gm_mask, token2, hc2)
        if row1 < rows:
            self.x_ub = s1[0]
            self.key_ub = s1[1]
            self.value_ub = s1[2]
            self.out_ub = self.out_b
            self._emit_row(gm_out, gm_mask, gm_weight, token1, hc1,
                           epsilon, clamp_value, weight_resident)

    @jit
    def _resident_pipeline(self, gm_x, gm_key, gm_value, gm_weight, gm_out,
                           gm_mask, block_idx, block_num, rows,
                           epsilon: dtypes.float32, clamp_value: dtypes.float32,
                           weight_resident):
        """Row-pair rotation pipeline over this block's rows.

        The three slots rotate (a, b, c) -> (c, a, b) -> (b, c, a) per pair,
        so the pair loop prefetches two rows ahead of the current compute.
        """
        slot_a = (self.x_a, self.key_a, self.value_a)
        slot_b = (self.x_b, self.key_b, self.value_b)
        slot_c = (self.x_c, self.key_c, self.value_c)
        first_row = block_idx
        pair_stride = 2 * block_num
        pairs = (rows - first_row + pair_stride - 1) // pair_stride
        first_token = first_row // self.hc_mult
        first_hc = first_row - first_token * self.hc_mult
        self._prologue(gm_x, gm_key, gm_value, gm_weight, gm_mask,
                       first_token, first_hc, weight_resident)
        n_groups = pairs // PAIRS_PER_GROUP
        for group_idx in range(n_groups):
            base = first_row + group_idx * PAIRS_PER_GROUP * pair_stride
            self._pair_step(
                gm_x, gm_key, gm_value, gm_weight, gm_out, gm_mask,
                base, rows, block_num, pair_stride,
                epsilon, clamp_value, weight_resident,
                slot_a, slot_b, slot_c, False, True, False,
            )
            self._pair_step(
                gm_x, gm_key, gm_value, gm_weight, gm_out, gm_mask,
                base + pair_stride, rows, block_num, pair_stride,
                epsilon, clamp_value, weight_resident,
                slot_c, slot_a, slot_b, False, True, False,
            )
            self._pair_step(
                gm_x, gm_key, gm_value, gm_weight, gm_out, gm_mask,
                base + (PAIRS_PER_GROUP - 1) * pair_stride, rows, block_num, pair_stride,
                epsilon, clamp_value, weight_resident,
                slot_b, slot_c, slot_a, False, True, True,
            )
        rem = pairs - PAIRS_PER_GROUP * n_groups
        if rem >= 1:
            base = first_row + n_groups * PAIRS_PER_GROUP * pair_stride
            self._pair_step(
                gm_x, gm_key, gm_value, gm_weight, gm_out, gm_mask,
                base, rows, block_num, pair_stride,
                epsilon, clamp_value, weight_resident,
                slot_a, slot_b, slot_c, False, False, False,
            )
            if rem >= 2:
                self._pair_step(
                    gm_x, gm_key, gm_value, gm_weight, gm_out, gm_mask,
                    base + pair_stride, rows, block_num, pair_stride,
                    epsilon, clamp_value, weight_resident,
                    slot_c, slot_a, slot_b, True, False, False,
                )

    def __call__(
        self,
        gm_x: Tensor,
        gm_key: Tensor,
        gm_value: Tensor,
        gm_weight: Tensor,
        gm_out: Tensor,
        epsilon: dtypes.float32,
        clamp_value: dtypes.float32,
        gm_mask: Tensor | None = None,
    ):
        block_idx = get_block_idx()
        block_num = get_block_num()
        rows = self.total_tokens * self.hc_mult
        if const_expr(self.chunked):
            for token_idx in range(block_idx, self.total_tokens, block_num):
                for hc_idx in dsl_range(
                    dtypes.int64(0), dtypes.int64(self.hc_mult), dtypes.int64(1)
                ):
                    if gm_mask is None:
                        x_row = gm_x[token_idx, hc_idx, None]
                        key_row = gm_key[token_idx, hc_idx, None]
                        w_row = gm_weight[hc_idx, None]
                        v_row = gm_value[token_idx, None]
                        o_row = gm_out[token_idx, hc_idx, None]
                        self._chunked_row(
                            x_row, key_row, w_row, v_row, o_row, epsilon, clamp_value
                        )
                    else:
                        masked = gm_mask[token_idx]
                        if masked:
                            x_row = gm_x[token_idx, hc_idx, None]
                            o_row = gm_out[token_idx, hc_idx, None]
                            for chunk_idx in range(self.num_chunks):
                                mem_copy(
                                    self.x_ch,
                                    tile_slice(x_row, (1, self.chunk), (0, chunk_idx)),
                                )
                                mem_copy(
                                    tile_slice(o_row, (1, self.chunk), (0, chunk_idx)),
                                    self.x_ch,
                                )
                        else:
                            x_row = gm_x[token_idx, hc_idx, None]
                            key_row = gm_key[token_idx, hc_idx, None]
                            w_row = gm_weight[hc_idx, None]
                            v_row = gm_value[token_idx, None]
                            o_row = gm_out[token_idx, hc_idx, None]
                            self._chunked_row(
                                x_row, key_row, w_row, v_row, o_row, epsilon, clamp_value
                            )
            return
        self._init_rotation_buffers()
        if block_num % self.hc_mult == 0:
            self._resident_pipeline(
                gm_x, gm_key, gm_value, gm_weight, gm_out, gm_mask,
                block_idx, block_num, rows, epsilon, clamp_value, True,
            )
        else:
            self._resident_pipeline(
                gm_x, gm_key, gm_value, gm_weight, gm_out, gm_mask,
                block_idx, block_num, rows, epsilon, clamp_value, False,
            )


class EngramGate:
    """Host-side CANNBot-DSL launcher for one static Engram shape."""

    def __init__(
        self,
        eps: float,
        clamp_value: float,
        dim: int = 5120,
        hc_mult: int = 4,
        max_blocks: int = 0,
    ):
        self.dim = int(dim)
        self.hc_mult = int(hc_mult)
        self.eps = float(eps)
        self.clamp_value = float(clamp_value)
        if max_blocks <= 0:
            raise ValueError(
                "max_blocks is required: pass the device vector core count "
                "(e.g. from engram_gate, which queries it automatically)"
            )
        self.max_blocks = int(max_blocks)

    @host
    def run(
        self,
        gm_x,
        gm_key,
        gm_value,
        gm_weight,
        gm_out,
        mask: Tensor | None = None,
    ):
        total_tokens = gm_x.shape[0]
        if const_expr(
            len(gm_x.shape) != 3
            or gm_x.shape[1] != self.hc_mult
            or gm_x.shape[2] != self.dim
        ):
            raise ValueError(
                f"x must have shape [T, {self.hc_mult}, {self.dim}] to match "
                "this EngramGate instance"
            )
        if const_expr(
            tuple(gm_key.shape) != tuple(gm_x.shape)
            or tuple(gm_out.shape) != tuple(gm_x.shape)
            or tuple(gm_value.shape) != (total_tokens, self.dim)
            or tuple(gm_weight.shape) != (self.hc_mult, self.dim)
        ):
            raise ValueError(
                "key and out must match x; value must have shape "
                f"[{total_tokens}, {self.dim}]; weight must have shape "
                f"[{self.hc_mult}, {self.dim}]"
            )
        if const_expr(mask is not None and mask.shape[0] != total_tokens):
            raise ValueError(f"mask must have shape [{total_tokens}] to match x")
        rows = total_tokens * self.hc_mult
        block_num = min(self.max_blocks, rows)
        op = EngramGateKernel(total_tokens, self.hc_mult, self.dim)
        if mask is not None:
            op[block_num](
                gm_x,
                gm_key,
                gm_value,
                gm_weight,
                gm_out,
                self.eps,
                self.clamp_value,
                mask,
            )
        else:
            op[block_num](
                gm_x,
                gm_key,
                gm_value,
                gm_weight,
                gm_out,
                self.eps,
                self.clamp_value,
            )

    def __call__(self, x, key, value, weight, image_mask=None):
        return engram_gate(
            x,
            key,
            value,
            weight,
            image_mask=image_mask,
            eps=self.eps,
            clamp_value=self.clamp_value,
        )



_RUN_CALLABLE_CACHE = {}


def _validate_engram_gate_inputs(x, key, value, weight, image_mask):
    """Validate the public entry contract in one pass.

    Returns ``(lead, hc_mult, dim, total_tokens)`` with the leading dims that
    flatten into tokens.
    """
    if x.ndim < 2:
        raise ValueError("x must have shape [..., hc_mult, dim] with ndim >= 2")
    if x.ndim == 2:
        lead = x.shape[:-1]
        hc_mult = 1
    else:
        lead = x.shape[:-2]
        hc_mult = int(x.shape[-2])
    dim = int(x.shape[-1])
    total_tokens = 1
    for size in lead:
        total_tokens *= int(size)
    if total_tokens <= 0 or hc_mult <= 0 or dim <= 0:
        raise ValueError("x dimensions must be positive")
    if tuple(key.shape) != tuple(x.shape):
        raise ValueError("key must have the same shape as x")
    if tuple(value.shape) != tuple(lead) + (dim,):
        raise ValueError(f"value must have shape {tuple(lead) + (dim,)}")
    if tuple(weight.shape) != (hc_mult, dim):
        raise ValueError(f"weight must have shape {(hc_mult, dim)}")
    if (x.dtype, key.dtype, value.dtype, weight.dtype) != (
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
    ):
        raise TypeError("x, key and value must be bfloat16 and weight must be float32")
    tensors = (x, key, value, weight)
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("x, key, value and weight must be contiguous")
    if any(tensor.device != x.device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on the same device")
    if image_mask is not None:
        if tuple(image_mask.shape) != tuple(lead) or image_mask.dtype != torch.bool:
            raise ValueError(f"image_mask must be bool with shape {tuple(lead)}")
        if image_mask.device != x.device:
            raise ValueError("image_mask must be on the same device as x")
        if not image_mask.is_contiguous():
            raise ValueError("image_mask must be contiguous")
    return lead, hc_mult, dim, total_tokens


def _get_run_callable(eps, clamp_value, dim, hc_mult, max_blocks, total_tokens, has_mask):
    """Return a cached compiled launcher; each specialization compiles once.

    eps/clamp_value are baked into the instance at compile time, so they are
    part of the cache key.  Repeated calls reuse the compiled artifact and
    launch directly instead of re-tracing the @host entry.
    """
    key = (eps, clamp_value, dim, hc_mult, max_blocks, total_tokens, has_mask)
    fn = _RUN_CALLABLE_CACHE.get(key)
    if fn is None:
        specs = (
            TensorSpec((total_tokens, hc_mult, dim), dtypes.bfloat16),
            TensorSpec((total_tokens, hc_mult, dim), dtypes.bfloat16),
            TensorSpec((total_tokens, dim), dtypes.bfloat16),
            TensorSpec((hc_mult, dim), dtypes.float32),
            TensorSpec((total_tokens, hc_mult, dim), dtypes.bfloat16),
        )
        if has_mask:
            specs = specs + (TensorSpec((total_tokens,), dtypes.bool_),)
        fn = dsl_compile(
            EngramGate(eps, clamp_value, dim, hc_mult, max_blocks).run, *specs
        )
        _RUN_CALLABLE_CACHE[key] = fn
    return fn


def engram_gate(
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    image_mask: torch.Tensor | None = None,
    *,
    eps: float = 1.0e-6,
    clamp_value: float = 1.0e-6,
) -> torch.Tensor:
    """Apply the Engram gate and return a BF16 residual tensor.

    ``x``/``key`` accept any ndim >= 2: the last dim is ``dim``, the
    second-to-last is ``hc_mult`` (a 2-D input means ``hc_mult == 1``), and
    all leading dims flatten into tokens. ``value`` and ``image_mask`` follow
    the leading dims; the output keeps ``x``'s shape.
    """
    lead, hc_mult, dim, total_tokens = _validate_engram_gate_inputs(
        x, key, value, weight, image_mask
    )

    out = torch.empty_like(x)
    if x.ndim == 3:
        x3, key3, value2, out3, mask1 = x, key, value, out, image_mask
    elif x.ndim == 2:
        x3 = x.view(total_tokens, hc_mult, dim)
        key3 = key.view(total_tokens, hc_mult, dim)
        value2 = value
        out3 = out.view(total_tokens, hc_mult, dim)
        mask1 = image_mask
    else:
        x3 = x.flatten(0, -3)
        key3 = key.flatten(0, -3)
        value2 = value.flatten(0, -2)
        out3 = out.flatten(0, -3)
        mask1 = image_mask.flatten() if image_mask is not None else None
    max_blocks = _vector_core_count()
    fn = _get_run_callable(
        eps, clamp_value, dim, hc_mult, max_blocks, total_tokens, mask1 is not None
    )
    if mask1 is not None:
        fn(x3, key3, value2, weight, out3, mask1)
    else:
        fn(x3, key3, value2, weight, out3)
    return out


@export("engram_gate")
def export_engram_gate():
    """Compile the default model profiles without NPU execution.

    ``total_tokens`` enters the compiled program (tensor shapes and the
    launch grid), and ``eps``/``clamp_value``/``max_blocks`` are captured by
    the compiled host launcher, so the catalogue enumerates the default
    token count for both supported vector-core quotas and both mask ABI
    variants; other values fall back to JIT.
    ``CANNBOTDSL_DS41_PROFILES`` may replace the list with an ``engram_gate``
    array of ``{"max_blocks": int, "tokens": [int, ...], "has_mask":
    [bool, ...]}`` entries; unlisted fields keep the defaults below.
    """
    import json
    import os

    dim, hc_mult = 5120, 4
    eps, clamp_value = 1.0e-6, 1.0e-6
    default_tokens = (72,)
    default_masks = (False, True)
    entries = [
        {"max_blocks": 56, "tokens": default_tokens, "has_mask": default_masks},
        {"max_blocks": 64, "tokens": default_tokens, "has_mask": default_masks},
    ]
    profile_path = os.environ.get("CANNBOTDSL_DS41_PROFILES")
    if profile_path:
        with open(profile_path, encoding="utf-8") as stream:
            loaded = json.load(stream).get("engram_gate")
        if loaded is not None:
            entries = loaded
    profiles = []
    for entry in entries:
        quota = int(entry.get("max_blocks", 56))
        for tokens in entry.get("tokens", default_tokens):
            for has_mask in entry.get("has_mask", default_masks):
                profiles.append((quota, int(tokens), bool(has_mask)))
    for handle in _RUN_CALLABLE_CACHE.values():
        handle.close()
    _RUN_CALLABLE_CACHE.clear()
    try:
        for max_blocks, tokens, has_mask in dict.fromkeys(profiles):
            _get_run_callable(
                eps, clamp_value, dim, hc_mult, max_blocks, tokens, has_mask
            )
    finally:
        for handle in _RUN_CALLABLE_CACHE.values():
            handle.close()
        _RUN_CALLABLE_CACHE.clear()
