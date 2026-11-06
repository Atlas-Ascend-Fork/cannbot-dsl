# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Forward or inverse partial interleaved rotary position embedding.

RotaryEmbeddingKernel accepts duplicated interleaved coefficients.
CompactRotaryEmbeddingKernel accepts 32 coefficients for the final 64 features
and is the mega-kernel path. Both retain FP32 arithmetic and BF16 rounding.
"""

from __future__ import annotations

from cannbotdsl import (
    make_layout,
    MemLoc,
    RegLayout,
    dtypes,
    Tensor,
    tile_slice as tile_view,
    vec_sync_all,
)
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ring_buffer import RingBuffer


VECTOR_CHUNK_ELEMENTS = 64  # Logical feature elements processed per SIMD chunk.


class RotaryEmbeddingKernel:
    def __init__(self, m: int, d: int, rope_dim: int, forward: bool = False):
        if (
            d % VECTOR_CHUNK_ELEMENTS
            or rope_dim > d
            or rope_dim % VECTOR_CHUNK_ELEMENTS
        ):
            raise ValueError("D and rope_dim must be multiples of 64")
        self.m, self.d, self.rope_dim = m, d, rope_dim
        self.forward = forward

    @jit
    def _upcast_bf16_input(self, xb_ub, x_f_ub):
        with vf(mode="simd"):
            full = rr.full_mask()
            for r in range(self.m):
                for c in range(self.d // VECTOR_CHUNK_ELEMENTS):
                    rr.vstore(
                        x_f_ub,
                        r * self.d + c * VECTOR_CHUNK_ELEMENTS,
                        rr.vcast(
                            rr.vload_unpack(
                                xb_ub,
                                r * self.d + c * VECTOR_CHUNK_ELEMENTS,
                                unpack_mode=rr.UnpackMode.B16_TO_B32,
                            ),
                            dtypes.float32,
                            mask=full,
                        ),
                        full,
                    )
            rr.vmem_bar("vst_vld")

    @jit
    def _apply_rotary_embedding(self, x_f_ub, cos_ub, sin_ub, outb_ub):
        with vf(mode="simd"):
            full = rr.full_mask()
            mask64_16 = rr.create_mask("vl64", elem_bits=16)
            one = rr.vdups(1, dtypes.uint32, mask=full)
            rnd = rr.vdups(0x7FFF, dtypes.uint32, mask=full)
            hi = rr.vdups(0xFFFF0000, dtypes.uint32, mask=full)
            head_chunks = (self.d - self.rope_dim) // VECTOR_CHUNK_ELEMENTS

            def round_to_bf16_bits(b):
                t = rr.vadd(
                    rr.vadd(b, rnd, mask=full),
                    rr.vbitwise_and(rr.vshr(b, 16, mask=full), one, mask=full),
                    mask=full,
                )
                return rr.vbitwise_and(t, hi, mask=full)

            for r in range(self.m):
                for c in range(self.d // VECTOR_CHUNK_ELEMENTS):
                    off = r * self.d + c * VECTOR_CHUNK_ELEMENTS
                    if c < head_chunks:
                        # non-RoPE prefix: narrow back to bf16 unchanged
                        rr.vstore_pack(
                            outb_ub,
                            off,
                            rr.vcast(
                                rr.vload(x_f_ub, off),
                                dtypes.bfloat16,
                                mask=full,
                                reg_layout=RegLayout.ZERO,
                            ),
                            full,
                            pack_mode=rr.PackMode.B32_TO_B16,
                        )
                    else:
                        even, odd = rr.vload_deinterleave(x_f_ub, off, width="b32")
                        # cos/sin table row is [T, rope_dim] in the network's
                        # interleaved layout (each frequency duplicated); the
                        # even lanes are the rope_dim/2 frequencies.
                        tbase = (
                            r * self.rope_dim
                            + (c - head_chunks) * VECTOR_CHUNK_ELEMENTS
                        )
                        cos, _ = rr.vload_deinterleave(cos_ub, tbase, width="b32")
                        sin, _ = rr.vload_deinterleave(sin_ub, tbase, width="b32")
                        sgn = -1.0 if self.forward else 1.0
                        oe = rr.vadd(
                            rr.vmul(even, cos, mask=full),
                            rr.vmuls(rr.vmul(odd, sin, mask=full), sgn, mask=full),
                            mask=full,
                        )
                        oo = rr.vsub(
                            rr.vmul(odd, cos, mask=full),
                            rr.vmuls(rr.vmul(even, sin, mask=full), sgn, mask=full),
                            mask=full,
                        )
                        packed = rr.vbitwise_or(
                            rr.vshr(
                                round_to_bf16_bits(rr.vreinterpret(oe, dtypes.uint32)),
                                16,
                                mask=full,
                            ),
                            rr.vbitwise_and(
                                round_to_bf16_bits(rr.vreinterpret(oo, dtypes.uint32)),
                                hi,
                                mask=full,
                            ),
                            mask=full,
                        )
                        rr.vstore(
                            outb_ub,
                            off,
                            rr.vreinterpret_lanes(packed, dtypes.bfloat16),
                            mask64_16,
                        )
            rr.vmem_bar("vst_vld")

    @kernel
    def rotary_embedding_kernel(
        self, out_gm: Tensor, x_gm: Tensor, cos_gm: Tensor, sin_gm: Tensor
    ):
        m, d = self.m, self.d
        xb_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((m, d)))
        x_f_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((m, d)))
        outb_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((m, d)))
        cos_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, m * self.rope_dim)))
        sin_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, m * self.rope_dim)))
        mem_copy(xb_ub, x_gm)
        mem_copy(cos_ub, tile_view(cos_gm, (1, m * self.rope_dim), (0, 0)))
        mem_copy(sin_ub, tile_view(sin_gm, (1, m * self.rope_dim), (0, 0)))
        self._upcast_bf16_input(xb_ub, x_f_ub)
        self._apply_rotary_embedding(x_f_ub, cos_ub, sin_ub, outb_ub)
        mem_copy(out_gm, outb_ub)


class RotaryEmbeddingOp:
    def __init__(self, m: int, d: int, rope_dim: int, forward: bool = False):
        self.kernel = RotaryEmbeddingKernel(m, d, rope_dim, forward=forward)

    @host
    def run(self, out_gm, x_gm, cos_gm, sin_gm):
        self.kernel.rotary_embedding_kernel[1](out_gm, x_gm, cos_gm, sin_gm)


class CompactRotaryEmbeddingKernel(RotaryEmbeddingKernel):
    def __init__(self, m, d, rope_dim, forward=False):
        if m != 1 or rope_dim != 64:
            raise ValueError("compact RoPE requires one row and rope_dim=64")
        super().__init__(m, d, rope_dim, forward=forward)

    @jit
    def _apply_rotary_embedding_vf(
        self, x_f_ub, cos_ub, sin_ub, outb_ub, output_offset=0
    ):
        """Rotate into an owning UB allocation; offset counts BF16 elements."""
        full = rr.full_mask()
        mask64_16 = rr.create_mask("vl64", elem_bits=16)
        one = rr.vdups(1, dtypes.uint32, mask=full)
        rnd = rr.vdups(0x7FFF, dtypes.uint32, mask=full)
        hi = rr.vdups(0xFFFF0000, dtypes.uint32, mask=full)
        head_chunks = (self.d - self.rope_dim) // VECTOR_CHUNK_ELEMENTS

        def round_to_bf16_bits(b):
            t = rr.vadd(
                rr.vadd(b, rnd, mask=full),
                rr.vbitwise_and(rr.vshr(b, 16, mask=full), one, mask=full),
                mask=full,
            )
            return rr.vbitwise_and(t, hi, mask=full)

        for c in range(self.d // VECTOR_CHUNK_ELEMENTS):
            off = c * VECTOR_CHUNK_ELEMENTS
            if c < head_chunks:
                rr.vstore_pack(
                    outb_ub,
                    output_offset + off,
                    rr.vcast(
                        rr.vload(x_f_ub, off),
                        dtypes.bfloat16,
                        mask=full,
                        reg_layout=RegLayout.ZERO,
                    ),
                    full,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            else:
                even, odd = rr.vload_deinterleave(x_f_ub, off, width="b32")
                cos, sin = rr.vload(cos_ub, 0), rr.vload(sin_ub, 0)
                sgn = -1.0 if self.forward else 1.0
                oe = rr.vadd(
                    rr.vmul(even, cos, mask=full),
                    rr.vmuls(rr.vmul(odd, sin, mask=full), sgn, mask=full),
                    mask=full,
                )
                oo = rr.vsub(
                    rr.vmul(odd, cos, mask=full),
                    rr.vmuls(rr.vmul(even, sin, mask=full), sgn, mask=full),
                    mask=full,
                )
                packed = rr.vbitwise_or(
                    rr.vshr(
                        round_to_bf16_bits(rr.vreinterpret(oe, dtypes.uint32)),
                        16,
                        mask=full,
                    ),
                    rr.vbitwise_and(
                        round_to_bf16_bits(rr.vreinterpret(oo, dtypes.uint32)),
                        hi,
                        mask=full,
                    ),
                    mask=full,
                )
                rr.vstore(
                    outb_ub,
                    output_offset + off,
                    rr.vreinterpret_lanes(packed, dtypes.bfloat16),
                    mask64_16,
                )
        rr.vmem_bar("vst_vld")

    @jit
    def _apply_rotary_embedding(self, x_f_ub, cos_ub, sin_ub, outb_ub):
        with vf(mode="simd"):
            self._apply_rotary_embedding_vf(x_f_ub, cos_ub, sin_ub, outb_ub)

    @kernel
    def compact_rotary_embedding_kernel(
        self, out_gm: Tensor, x_gm: Tensor, cos_gm: Tensor, sin_gm: Tensor
    ):
        xb_ub = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, self.d)), depth=1
        )
        x_f_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, self.d)))
        outb_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, self.d)))
        cos_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 64)))
        sin_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 64)))
        mem_copy(xb_ub.next_write(), x_gm)
        mem_copy(tile_view(cos_ub, (1, 32), (0, 0)), cos_gm)
        mem_copy(tile_view(sin_ub, (1, 32), (0, 0)), sin_gm)
        vec_sync_all()
        self._upcast_bf16_input(xb_ub, x_f_ub)
        self._apply_rotary_embedding(x_f_ub, cos_ub, sin_ub, outb_ub)
        vec_sync_all()
        mem_copy(out_gm, outb_ub)
