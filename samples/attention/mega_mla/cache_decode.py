# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Window FP8 and compressed FP4 cache decoding with stored BF16 scales.

FP8 uses E4M3FN values; FP4 uses packed E2M1 values. Both gather FP32 scale
bit patterns through uint32 registers and return decoded BF16 records.
"""

from cannbotdsl import from_tensor

from cannbotdsl import make_layout, dtypes, RegLayout
from cannbotdsl.lang import jit, vf, const_expr
from cannbotdsl.ops import reg as rr


class Fp8CacheDecoder:
    @jit
    def decode(self, data, dst, row, base):
        scale_data = from_tensor(
            data, dtypes.bfloat16, make_layout((data.shape[0], data.shape[1] // 2))
        )
        with vf(mode="simd"):
            m32 = rr.full_mask(elem_bits=32)
            m8 = rr.full_mask(elem_bits=8)
            groups = rr.vshr(rr.varange(0, dtypes.uint32), 5, mask=m32)
            sb = rr.vcast(
                rr.vload_unpack(
                    scale_data,
                    row * (data.shape[1] // 2) + (base + 512) // 2,
                    unpack_mode=rr.UnpackMode.B16_TO_B32,
                ),
                dtypes.float32,
                mask=m32,
            )
            for c in range(0, 8, 2):
                b0 = rr.vload_unpack(
                    data,
                    row * data.shape[1] + base + c * 64,
                    unpack_mode=rr.UnpackMode.B8_TO_B32,
                )
                b1 = rr.vload_unpack(
                    data,
                    row * data.shape[1] + base + (c + 1) * 64,
                    unpack_mode=rr.UnpackMode.B8_TO_B32,
                )
                s0 = rr.vreinterpret(
                    rr.vgather_reg(
                        rr.vreinterpret(sb, dtypes.uint32),
                        rr.vadds(groups, c * 2, mask=m32),
                    ),
                    dtypes.float32,
                )
                s1 = rr.vreinterpret(
                    rr.vgather_reg(
                        rr.vreinterpret(sb, dtypes.uint32),
                        rr.vadds(groups, (c + 1) * 2, mask=m32),
                    ),
                    dtypes.float32,
                )
                f0 = rr.vcast(
                    rr.vreinterpret_lanes(b0, dtypes.float8_e4m3fn),
                    dtypes.float32,
                    mask=m8,
                    reg_layout=RegLayout.ZERO,
                )
                f1 = rr.vcast(
                    rr.vreinterpret_lanes(b1, dtypes.float8_e4m3fn),
                    dtypes.float32,
                    mask=m8,
                    reg_layout=RegLayout.ZERO,
                )
                f0 = rr.vmul(f0, s0, mask=m32)
                f1 = rr.vmul(f1, s1, mask=m32)
                o0 = rr.vcast(f0, dtypes.bfloat16, mask=m32, reg_layout=RegLayout.ZERO)
                o1 = rr.vcast(f1, dtypes.bfloat16, mask=m32, reg_layout=RegLayout.ZERO)
                rr.vstore_pack(
                    dst, row * 512 + c * 64, o0, m32, pack_mode=rr.PackMode.B32_TO_B16
                )
                rr.vstore_pack(
                    dst,
                    row * 512 + (c + 1) * 64,
                    o1,
                    m32,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            rr.vmem_bar("vst_vld")


class Fp4CacheDecoder:
    def __init__(self, unroll=2, *, native_bf16_scales=False):
        if unroll not in (1, 2, 4):
            raise ValueError("unroll must be 1, 2 or 4")
        self.unroll = unroll
        self.native_bf16_scales = native_bf16_scales

    @jit
    def decode(self, data_ub, dst_ub, row):
        if self.unroll == 1:
            self.decode_unroll_1(data_ub, dst_ub, row)
        elif self.unroll == 2:
            self.decode_unroll_2(data_ub, dst_ub, row)
        else:
            self.decode_unroll_4(data_ub, dst_ub, row)

    @jit
    def _prepare_scales(self, data_ub, row):
        """Prepare shared scale registers inside the caller SIMD scope."""
        scale_data = from_tensor(
            data_ub,
            dtypes.bfloat16,
            make_layout((data_ub.shape[0], data_ub.shape[1] // 2)),
        )
        m32 = rr.full_mask(elem_bits=32)
        if const_expr(self.native_bf16_scales):
            sb = rr.vcast(
                rr.vload_unpack(
                    scale_data,
                    row * (data_ub.shape[1] // 2) + 128,
                    unpack_mode=rr.UnpackMode.B16_TO_B32,
                ),
                dtypes.float32,
                mask=m32,
            )
        else:
            sb = rr.vload_unpack(
                data_ub,
                row * data_ub.shape[1] + 256,
                unpack_mode=rr.UnpackMode.B8_TO_B32,
            )
            scale_idx = rr.vshl(
                rr.vbitwise_and(
                    rr.varange(0, dtypes.uint32),
                    rr.vdups(31, dtypes.uint32, mask=m32),
                    mask=m32,
                ),
                1,
                mask=m32,
            )
            low = rr.vgather_reg(sb, scale_idx)
            high = rr.vgather_reg(sb, rr.vadds(scale_idx, 1, mask=m32))
            sb = rr.vreinterpret(
                rr.vbitwise_or(
                    rr.vshl(low, 16, mask=m32), rr.vshl(high, 24, mask=m32), mask=m32
                ),
                dtypes.float32,
            )
        groups = rr.vshr(rr.varange(0, dtypes.uint32), 3, mask=m32)
        return sb, groups

    @jit
    def decode_unroll_1(self, data_ub, dst_ub, row):
        with vf(mode="simd"):
            m32 = rr.full_mask(elem_bits=32)
            m8 = rr.full_mask(elem_bits=8)
            m16 = rr.full_mask(elem_bits=16)
            sb, groups = self._prepare_scales(data_ub, row)
            for c in range(0, 4, 1):
                b0 = rr.vload_unpack(
                    data_ub,
                    row * data_ub.shape[1] + (c + 0) * 64,
                    unpack_mode=rr.UnpackMode.B8_TO_B32,
                )
                s0 = rr.vreinterpret(
                    rr.vgather_reg(
                        rr.vreinterpret(sb, dtypes.uint32),
                        rr.vadds(groups, (c + 0) * 8, mask=m32),
                    ),
                    dtypes.float32,
                )
                h0 = rr.vcast(
                    rr.vreinterpret_lanes(b0, dtypes.fp4x2_e2m1),
                    dtypes.bfloat16,
                    mask=m8,
                    reg_layout=RegLayout.ZERO,
                )
                e0 = rr.vcast(h0, dtypes.float32, mask=m16, reg_layout=RegLayout.ZERO)
                o0 = rr.vcast(h0, dtypes.float32, mask=m16, reg_layout=RegLayout.ONE)
                e0 = rr.vmul(e0, s0, mask=m32)
                o0 = rr.vmul(o0, s0, mask=m32)
                a0, z0 = rr.vinterleave(e0, o0)
                a0 = rr.vcast(a0, dtypes.bfloat16, mask=m32, reg_layout=RegLayout.ZERO)
                z0 = rr.vcast(z0, dtypes.bfloat16, mask=m32, reg_layout=RegLayout.ZERO)
                rr.vstore_pack(
                    dst_ub,
                    row * 512 + (c + 0) * 128,
                    a0,
                    m32,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
                rr.vstore_pack(
                    dst_ub,
                    row * 512 + (c + 0) * 128 + 64,
                    z0,
                    m32,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def decode_unroll_2(self, data_ub, dst_ub, row):
        with vf(mode="simd"):
            m32 = rr.full_mask(elem_bits=32)
            m8 = rr.full_mask(elem_bits=8)
            m16 = rr.full_mask(elem_bits=16)
            sb, groups = self._prepare_scales(data_ub, row)
            for c in range(0, 4, 2):
                b0 = rr.vload_unpack(
                    data_ub,
                    row * data_ub.shape[1] + (c + 0) * 64,
                    unpack_mode=rr.UnpackMode.B8_TO_B32,
                )
                b1 = rr.vload_unpack(
                    data_ub,
                    row * data_ub.shape[1] + (c + 1) * 64,
                    unpack_mode=rr.UnpackMode.B8_TO_B32,
                )
                s0 = rr.vreinterpret(
                    rr.vgather_reg(
                        rr.vreinterpret(sb, dtypes.uint32),
                        rr.vadds(groups, (c + 0) * 8, mask=m32),
                    ),
                    dtypes.float32,
                )
                s1 = rr.vreinterpret(
                    rr.vgather_reg(
                        rr.vreinterpret(sb, dtypes.uint32),
                        rr.vadds(groups, (c + 1) * 8, mask=m32),
                    ),
                    dtypes.float32,
                )
                h0 = rr.vcast(
                    rr.vreinterpret_lanes(b0, dtypes.fp4x2_e2m1),
                    dtypes.bfloat16,
                    mask=m8,
                    reg_layout=RegLayout.ZERO,
                )
                h1 = rr.vcast(
                    rr.vreinterpret_lanes(b1, dtypes.fp4x2_e2m1),
                    dtypes.bfloat16,
                    mask=m8,
                    reg_layout=RegLayout.ZERO,
                )
                e0 = rr.vcast(h0, dtypes.float32, mask=m16, reg_layout=RegLayout.ZERO)
                e1 = rr.vcast(h1, dtypes.float32, mask=m16, reg_layout=RegLayout.ZERO)
                o0 = rr.vcast(h0, dtypes.float32, mask=m16, reg_layout=RegLayout.ONE)
                o1 = rr.vcast(h1, dtypes.float32, mask=m16, reg_layout=RegLayout.ONE)
                e0 = rr.vmul(e0, s0, mask=m32)
                e1 = rr.vmul(e1, s1, mask=m32)
                o0 = rr.vmul(o0, s0, mask=m32)
                o1 = rr.vmul(o1, s1, mask=m32)
                a0, z0 = rr.vinterleave(e0, o0)
                a1, z1 = rr.vinterleave(e1, o1)
                a0 = rr.vcast(a0, dtypes.bfloat16, mask=m32, reg_layout=RegLayout.ZERO)
                a1 = rr.vcast(a1, dtypes.bfloat16, mask=m32, reg_layout=RegLayout.ZERO)
                z0 = rr.vcast(z0, dtypes.bfloat16, mask=m32, reg_layout=RegLayout.ZERO)
                z1 = rr.vcast(z1, dtypes.bfloat16, mask=m32, reg_layout=RegLayout.ZERO)
                rr.vstore_pack(
                    dst_ub,
                    row * 512 + (c + 0) * 128,
                    a0,
                    m32,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
                rr.vstore_pack(
                    dst_ub,
                    row * 512 + (c + 0) * 128 + 64,
                    z0,
                    m32,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
                rr.vstore_pack(
                    dst_ub,
                    row * 512 + (c + 1) * 128,
                    a1,
                    m32,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
                rr.vstore_pack(
                    dst_ub,
                    row * 512 + (c + 1) * 128 + 64,
                    z1,
                    m32,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def decode_unroll_4(self, data_ub, dst_ub, row):
        with vf(mode="simd"):
            m32 = rr.full_mask(elem_bits=32)
            m8 = rr.full_mask(elem_bits=8)
            m16 = rr.full_mask(elem_bits=16)
            sb, groups = self._prepare_scales(data_ub, row)
            for c in range(0, 4, 4):
                b0 = rr.vload_unpack(
                    data_ub,
                    row * data_ub.shape[1] + (c + 0) * 64,
                    unpack_mode=rr.UnpackMode.B8_TO_B32,
                )
                b1 = rr.vload_unpack(
                    data_ub,
                    row * data_ub.shape[1] + (c + 1) * 64,
                    unpack_mode=rr.UnpackMode.B8_TO_B32,
                )
                b2 = rr.vload_unpack(
                    data_ub,
                    row * data_ub.shape[1] + (c + 2) * 64,
                    unpack_mode=rr.UnpackMode.B8_TO_B32,
                )
                b3 = rr.vload_unpack(
                    data_ub,
                    row * data_ub.shape[1] + (c + 3) * 64,
                    unpack_mode=rr.UnpackMode.B8_TO_B32,
                )
                s0 = rr.vreinterpret(
                    rr.vgather_reg(
                        rr.vreinterpret(sb, dtypes.uint32),
                        rr.vadds(groups, (c + 0) * 8, mask=m32),
                    ),
                    dtypes.float32,
                )
                s1 = rr.vreinterpret(
                    rr.vgather_reg(
                        rr.vreinterpret(sb, dtypes.uint32),
                        rr.vadds(groups, (c + 1) * 8, mask=m32),
                    ),
                    dtypes.float32,
                )
                s2 = rr.vreinterpret(
                    rr.vgather_reg(
                        rr.vreinterpret(sb, dtypes.uint32),
                        rr.vadds(groups, (c + 2) * 8, mask=m32),
                    ),
                    dtypes.float32,
                )
                s3 = rr.vreinterpret(
                    rr.vgather_reg(
                        rr.vreinterpret(sb, dtypes.uint32),
                        rr.vadds(groups, (c + 3) * 8, mask=m32),
                    ),
                    dtypes.float32,
                )
                h0 = rr.vcast(
                    rr.vreinterpret_lanes(b0, dtypes.fp4x2_e2m1),
                    dtypes.bfloat16,
                    mask=m8,
                    reg_layout=RegLayout.ZERO,
                )
                h1 = rr.vcast(
                    rr.vreinterpret_lanes(b1, dtypes.fp4x2_e2m1),
                    dtypes.bfloat16,
                    mask=m8,
                    reg_layout=RegLayout.ZERO,
                )
                h2 = rr.vcast(
                    rr.vreinterpret_lanes(b2, dtypes.fp4x2_e2m1),
                    dtypes.bfloat16,
                    mask=m8,
                    reg_layout=RegLayout.ZERO,
                )
                h3 = rr.vcast(
                    rr.vreinterpret_lanes(b3, dtypes.fp4x2_e2m1),
                    dtypes.bfloat16,
                    mask=m8,
                    reg_layout=RegLayout.ZERO,
                )
                e0 = rr.vcast(h0, dtypes.float32, mask=m16, reg_layout=RegLayout.ZERO)
                e1 = rr.vcast(h1, dtypes.float32, mask=m16, reg_layout=RegLayout.ZERO)
                e2 = rr.vcast(h2, dtypes.float32, mask=m16, reg_layout=RegLayout.ZERO)
                e3 = rr.vcast(h3, dtypes.float32, mask=m16, reg_layout=RegLayout.ZERO)
                o0 = rr.vcast(h0, dtypes.float32, mask=m16, reg_layout=RegLayout.ONE)
                o1 = rr.vcast(h1, dtypes.float32, mask=m16, reg_layout=RegLayout.ONE)
                o2 = rr.vcast(h2, dtypes.float32, mask=m16, reg_layout=RegLayout.ONE)
                o3 = rr.vcast(h3, dtypes.float32, mask=m16, reg_layout=RegLayout.ONE)
                e0 = rr.vmul(e0, s0, mask=m32)
                e1 = rr.vmul(e1, s1, mask=m32)
                e2 = rr.vmul(e2, s2, mask=m32)
                e3 = rr.vmul(e3, s3, mask=m32)
                o0 = rr.vmul(o0, s0, mask=m32)
                o1 = rr.vmul(o1, s1, mask=m32)
                o2 = rr.vmul(o2, s2, mask=m32)
                o3 = rr.vmul(o3, s3, mask=m32)
                a0, z0 = rr.vinterleave(e0, o0)
                a1, z1 = rr.vinterleave(e1, o1)
                a2, z2 = rr.vinterleave(e2, o2)
                a3, z3 = rr.vinterleave(e3, o3)
                a0 = rr.vcast(a0, dtypes.bfloat16, mask=m32, reg_layout=RegLayout.ZERO)
                a1 = rr.vcast(a1, dtypes.bfloat16, mask=m32, reg_layout=RegLayout.ZERO)
                a2 = rr.vcast(a2, dtypes.bfloat16, mask=m32, reg_layout=RegLayout.ZERO)
                a3 = rr.vcast(a3, dtypes.bfloat16, mask=m32, reg_layout=RegLayout.ZERO)
                z0 = rr.vcast(z0, dtypes.bfloat16, mask=m32, reg_layout=RegLayout.ZERO)
                z1 = rr.vcast(z1, dtypes.bfloat16, mask=m32, reg_layout=RegLayout.ZERO)
                z2 = rr.vcast(z2, dtypes.bfloat16, mask=m32, reg_layout=RegLayout.ZERO)
                z3 = rr.vcast(z3, dtypes.bfloat16, mask=m32, reg_layout=RegLayout.ZERO)
                rr.vstore_pack(
                    dst_ub,
                    row * 512 + (c + 0) * 128,
                    a0,
                    m32,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
                rr.vstore_pack(
                    dst_ub,
                    row * 512 + (c + 0) * 128 + 64,
                    z0,
                    m32,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
                rr.vstore_pack(
                    dst_ub,
                    row * 512 + (c + 1) * 128,
                    a1,
                    m32,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
                rr.vstore_pack(
                    dst_ub,
                    row * 512 + (c + 1) * 128 + 64,
                    z1,
                    m32,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
                rr.vstore_pack(
                    dst_ub,
                    row * 512 + (c + 2) * 128,
                    a2,
                    m32,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
                rr.vstore_pack(
                    dst_ub,
                    row * 512 + (c + 2) * 128 + 64,
                    z2,
                    m32,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
                rr.vstore_pack(
                    dst_ub,
                    row * 512 + (c + 3) * 128,
                    a3,
                    m32,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
                rr.vstore_pack(
                    dst_ub,
                    row * 512 + (c + 3) * 128 + 64,
                    z3,
                    m32,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            rr.vmem_bar("vst_vld")
