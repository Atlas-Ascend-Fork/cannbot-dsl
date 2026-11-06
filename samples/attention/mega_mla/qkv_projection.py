# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Variable-head Q/KV projection with split-K=2 FP32 atomic accumulation.

Standalone projection phases (mega reuses the same RMS math inside its UB
KV norm/RoPE/cache-quantization path; launch grid is selected by the caller):
  P0 AIV : device MX-quant of x                     -> x8, xs
  barrier
  P1 AIC : wq_a on blocks 0..19, wkv on blocks 20..27; M<=64,K=2560/core
  barrier
  P2 AIV : q_norm + MX-quant on qr -> qn8, qns;  kv_norm on kv -> kvn
  barrier / rewind
  P3 AIC : wq_b with baseN=256 and block-stride N scheduling -> qfull

GM fallback (D1) for every Cube->Vector and Vector->Cube handoff, with
``global_sync_all`` (AIC FIX / AIV MTE2 / MTE3 funnelled through the AIC) and
``onchip_rewind`` to reuse on-chip addresses across execution-disjoint phases.

Host ABI: callers prepare all three E4M3 FRACTAL_NZ weight roots once with
torch_npu.npu_format_cast(weight, torch_npu.Format.FRACTAL_NZ) before run();
scales are [M-or-N, K/64, 2]. A/B L1 use depth=4 (not full-K A loading),
scales depth=2, L0A/L0C depth=2. P1 L0B depth=2 (2*32 KiB); standalone P3
stays depth=1 (the enclosing mega-kernel already selects P3 depth=2).
QR/KV are distinct FP32 GM accumulators, cleared on Vector every launch.
Norm rounds the completed FP32 accumulator to BF16 in UB, then widens it to
FP32 for both reduction and normalization. KV norm output remains BF16.
"""

from __future__ import annotations


from cannbotdsl import make_layout
from cannbotdsl import MemLoc, RegLayout, dtypes, global_sync_all, vec_sync_all
from cannbotdsl.ops.sync import onchip_rewind
from cannbotdsl.buffer import Buffer
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl import Tensor, tile_slice as tile_view

# Scalar dtype classes were replaced by canonical dtype objects in 0.5.x.
Float8E4M3FN = dtypes.float8_e4m3fn
Float8E8M0 = dtypes.float8_e8m0


if __package__:
    from .mxfp8_matmul import _as_zn_view
else:
    from mxfp8_matmul import _as_zn_view
if __package__:
    from .mxfp8_matmul import (  # noqa: E402
        L2_CACHE_DISABLE,
        Mxfp8MatmulKernel,
        get_scale_k_len,
    )
else:
    from mxfp8_matmul import (  # noqa: E402
        L2_CACHE_DISABLE,
        Mxfp8MatmulKernel,
        get_scale_k_len,
    )

if __package__:
    from .attention_geometry import HEAD_DIM, validate_query_heads
else:
    from attention_geometry import HEAD_DIM, validate_query_heads

HID = 5120
QLORA = 1280
KVD = 512
CUBES = 32
BN1, BK1, KL1_1, SKL1_1 = 128, 256, 512, 5120
# Legacy diagnostic fixtures import N2; variable-head kernels never use it.
N2 = 16 * HEAD_DIM
BN2, BK2, KL1_2, SKL1_2 = 256, 128, 256, 1280
GROUP = 32
VECTOR_CHUNK_ELEMENTS = 64  # Logical feature elements processed per SIMD chunk.
E4M3_ELE_EMAX = 8
E8M0_BIAS = 127
E4M3_MAX = 448.0


def _make_qa_kv_projection_matmul(n, k, m=16):
    bm = min(64, ((m + 15) // 16) * 16)
    return Mxfp8MatmulKernel(
        m,
        n,
        k,
        bm,
        BN1,
        BK1,
        KL1_1,
        HID // 2,
        l1_buffers=4,
        l0c_buffers=2,
        full_load=False,
        transpose_a=False,
        transpose_b=True,
        scale_a_k_outer=False,
        scale_b_k_outer=False,
        b_gm_nz=True,
        used_core_num=n // BN1,
    )


def _make_query_expansion_matmul(m=16, heads=16):
    n = validate_query_heads(heads) * HEAD_DIM
    bm = min(16, m)
    return Mxfp8MatmulKernel(
        m,
        n,
        QLORA,
        bm,
        BN2,
        BK2,
        KL1_2,
        SKL1_2,
        l1_buffers=4,
        l0c_buffers=2,
        full_load=False,
        transpose_a=False,
        transpose_b=True,
        scale_a_k_outer=False,
        scale_b_k_outer=False,
        b_gm_nz=True,
        used_core_num=CUBES,
    )


def _bind_projection_pipeline(k, c):
    for name in ("l1_a", "l1_b", "l1_scale_a", "l1_scale_b", "l0a", "l0b", "l0c"):
        setattr(k, name, c[name])


def _create_projection_pipeline(m, n, bk, kl1, skl1, l0b_depth=1):
    g = get_scale_k_len(skl1)
    return {
        "l1_a": RingBuffer(
            MemLoc.L1, Float8E4M3FN, make_layout((m, kl1)), depth=4, data_format="nz"
        ),
        "l1_b": RingBuffer(
            MemLoc.L1, Float8E4M3FN, make_layout((n, kl1)), depth=4, data_format="nz"
        ),
        "l1_scale_a": RingBuffer(
            MemLoc.L1, Float8E8M0, make_layout((g, m)), depth=2, data_format="nz"
        ),
        "l1_scale_b": RingBuffer(
            MemLoc.L1, Float8E8M0, make_layout((g, n)), depth=2, data_format="nz"
        ),
        "l0a": RingBuffer(MemLoc.L0A, Float8E4M3FN, make_layout((m, bk)), depth=2),
        "l0b": RingBuffer(
            MemLoc.L0B, Float8E4M3FN, make_layout((n, bk)), depth=l0b_depth
        ),
        "l0c": RingBuffer(MemLoc.L0C, dtypes.float32, make_layout((m, n)), depth=2),
    }


class QkvProjectionKernel:
    def __init__(self, m=16, heads=16):
        self.heads = validate_query_heads(heads)
        self.q_width = self.heads * HEAD_DIM
        self.m = m
        self.m_tiles = (m + 15) // 16
        # T64 uses M64. Avoid over-wide static MX tails for short-token cases.
        self.p1_m = min(64, m)
        self.p1_m_tiles = (m + self.p1_m - 1) // self.p1_m
        self.k1 = _make_qa_kv_projection_matmul(QLORA, HID, m)
        self.k3 = _make_qa_kv_projection_matmul(KVD, HID, m)
        self.k2 = _make_query_expansion_matmul(m, self.heads)
        self.n1_blocks = QLORA // BN1
        self.n3_blocks = KVD // BN1
        self.n2_blocks = self.q_width // BN2

    # ---- P0: per-row MX quant of x (rows distributed over blocks) ----
    @jit
    def _quantize_mxfp8_rows(self, x_gm, y_gm, s_gm, block, nblk, rows, n):
        x_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, n)))
        y_ub = Buffer(MemLoc.UB, dtypes.uint8, make_layout((1, n)))
        s_ub = Buffer(
            MemLoc.UB, dtypes.uint8, make_layout((1, n // VECTOR_CHUNK_ELEMENTS, 2))
        )
        nc = n // VECTOR_CHUNK_ELEMENTS
        for r in range(block, rows, nblk):
            mem_copy(
                x_ub,
                tile_view(x_gm, (1, n), (r, 0)),
                options=copy_options(l2_cache_ctl=L2_CACHE_DISABLE),
            )
            vec_sync_all()
            with vf(mode="simd"):
                full = rr.full_mask()
                lane32 = rr.create_mask("vl32", elem_bits=32)
                high = rr.mask_not(lane32, exec_mask=full)
                zero = rr.vdups(0.0, dtypes.float32, mask=full)
                c254 = rr.vdups(2 * E8M0_BIAS, dtypes.int32, mask=full)
                for c in range(nc):
                    off = c * VECTOR_CHUNK_ELEMENTS
                    x = rr.vcast(
                        rr.vload_unpack(
                            x_ub, off, unpack_mode=rr.UnpackMode.B16_TO_B32
                        ),
                        dtypes.float32,
                        mask=full,
                    )
                    x_lo = rr.vselect(zero, x, cond_mask=high)
                    x_hi = rr.vselect(zero, x, cond_mask=lane32)
                    amax_lo = rr.vreduce_max(rr.vabs(x_lo, mask=full), mask=full)
                    amax_hi = rr.vreduce_max(rr.vabs(x_hi, mask=full), mask=full)
                    b_lo = rr.vmaxs(
                        rr.vadds(
                            rr.vreinterpret(
                                rr.vshr(
                                    rr.vreinterpret(amax_lo, dtypes.uint32),
                                    23,
                                    mask=full,
                                ),
                                dtypes.int32,
                            ),
                            -E4M3_ELE_EMAX,
                            mask=full,
                        ),
                        0,
                        mask=full,
                    )
                    b_hi = rr.vmaxs(
                        rr.vadds(
                            rr.vreinterpret(
                                rr.vshr(
                                    rr.vreinterpret(amax_hi, dtypes.uint32),
                                    23,
                                    mask=full,
                                ),
                                dtypes.int32,
                            ),
                            -E4M3_ELE_EMAX,
                            mask=full,
                        ),
                        0,
                        mask=full,
                    )
                    inv_lo = rr.vdup(
                        rr.vreinterpret(
                            rr.vshl(rr.vsub(c254, b_lo, mask=full), 23, mask=full),
                            dtypes.float32,
                        ),
                        mask=full,
                    )
                    inv_hi = rr.vdup(
                        rr.vreinterpret(
                            rr.vshl(rr.vsub(c254, b_hi, mask=full), 23, mask=full),
                            dtypes.float32,
                        ),
                        mask=full,
                    )
                    inv = rr.vselect(inv_hi, inv_lo, cond_mask=high)
                    y = rr.vmins(
                        rr.vmaxs(rr.vmul(x, inv, mask=full), -E4M3_MAX, mask=full),
                        E4M3_MAX,
                        mask=full,
                    )
                    y = rr.vcast(
                        y, dtypes.float8_e4m3fn, mask=full, reg_layout=RegLayout.ZERO
                    )
                    rr.vstore_pack(y_ub, off, y, full, pack_mode=rr.PackMode.B32_TO_B8)
                    rr.vstore_first(
                        s_ub, 2 * c, rr.vreinterpret_lanes(b_lo, dtypes.uint8)
                    )
                    rr.vstore_first(
                        s_ub, 2 * c + 1, rr.vreinterpret_lanes(b_hi, dtypes.uint8)
                    )
                rr.vmem_bar("vst_vld")
            vec_sync_all()
            mem_copy(tile_view(y_gm, (1, n), (r, 0)), y_ub)
            mem_copy(tile_view(s_gm, (1, nc, 2), (r, 0, 0)), s_ub)
            vec_sync_all()

    # ---- P2a: RMSNorm + MX quant in one pass ----
    @jit
    def _rms_norm_quantize_query(self, h_gm, g_gm, y_gm, s_gm, block, nblk, rows, n):
        h_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, n)))
        h_bf16_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, n)))
        g_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, n)))
        xf_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, n)))
        y_ub = Buffer(MemLoc.UB, dtypes.uint8, make_layout((1, n)))
        s_ub = Buffer(
            MemLoc.UB, dtypes.uint8, make_layout((1, n // VECTOR_CHUNK_ELEMENTS, 2))
        )
        nc = n // VECTOR_CHUNK_ELEMENTS
        mem_copy(g_ub, tile_view(g_gm, (1, n), (0, 0)))
        for r in range(block, rows, nblk):
            mem_copy(
                h_ub,
                tile_view(h_gm, (1, n), (r, 0)),
                options=copy_options(l2_cache_ctl=L2_CACHE_DISABLE),
            )
            vec_sync_all()
            with vf(mode="simd"):
                full = rr.full_mask()
                lane32 = rr.create_mask("vl32", elem_bits=32)
                high = rr.mask_not(lane32, exec_mask=full)
                # r=FP32(BF16_RN(h))；平方和与归一化分子均使用 r。
                # pack/unpack 保持 64 个值连续，禁止把窄化寄存器当 FP32 重读。
                for c in range(nc):
                    rounded = rr.vcast(
                        rr.vload(h_ub, c * VECTOR_CHUNK_ELEMENTS),
                        dtypes.bfloat16,
                        mask=full,
                        rounding="rn",
                        reg_layout=RegLayout.ZERO,
                    )
                    rr.vstore_pack(
                        h_bf16_ub,
                        c * VECTOR_CHUNK_ELEMENTS,
                        rounded,
                        full,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )
                rr.vmem_bar("vst_vld")
                acc = rr.vdups(0.0, dtypes.float32, mask=full)
                for c in range(nc):
                    xf = rr.vcast(
                        rr.vload_unpack(
                            h_bf16_ub,
                            c * VECTOR_CHUNK_ELEMENTS,
                            unpack_mode=rr.UnpackMode.B16_TO_B32,
                        ),
                        dtypes.float32,
                        mask=full,
                    )
                    rr.vstore(xf_ub, c * VECTOR_CHUNK_ELEMENTS, xf, full)
                    acc = rr.vadd(acc, rr.vmul(xf, xf, mask=full), mask=full)
                rr.vmem_bar("vst_vld")
                mean = rr.vmuls(rr.vreduce_sum(acc, mask=full), 1.0 / n, mask=full)
                rstd = rr.vdup(
                    rr.vdiv(
                        rr.vdups(1.0, dtypes.float32, mask=full),
                        rr.vsqrt(rr.vadds(mean, 1.0e-20, mask=full), mask=full),
                        mask=full,
                    ),
                    mask=full,
                )
                zero = rr.vdups(0.0, dtypes.float32, mask=full)
                c254 = rr.vdups(2 * E8M0_BIAS, dtypes.int32, mask=full)
                for c in range(nc):
                    off = c * VECTOR_CHUNK_ELEMENTS
                    y = rr.vmul(
                        rr.vmul(rr.vload(xf_ub, off), rstd, mask=full),
                        rr.vload(g_ub, off),
                        mask=full,
                    )
                    y_lo = rr.vselect(zero, y, cond_mask=high)
                    y_hi = rr.vselect(zero, y, cond_mask=lane32)
                    amax_lo = rr.vreduce_max(rr.vabs(y_lo, mask=full), mask=full)
                    amax_hi = rr.vreduce_max(rr.vabs(y_hi, mask=full), mask=full)
                    b_lo = rr.vmaxs(
                        rr.vadds(
                            rr.vreinterpret(
                                rr.vshr(
                                    rr.vreinterpret(amax_lo, dtypes.uint32),
                                    23,
                                    mask=full,
                                ),
                                dtypes.int32,
                            ),
                            -E4M3_ELE_EMAX,
                            mask=full,
                        ),
                        0,
                        mask=full,
                    )
                    b_hi = rr.vmaxs(
                        rr.vadds(
                            rr.vreinterpret(
                                rr.vshr(
                                    rr.vreinterpret(amax_hi, dtypes.uint32),
                                    23,
                                    mask=full,
                                ),
                                dtypes.int32,
                            ),
                            -E4M3_ELE_EMAX,
                            mask=full,
                        ),
                        0,
                        mask=full,
                    )
                    inv_lo = rr.vdup(
                        rr.vreinterpret(
                            rr.vshl(rr.vsub(c254, b_lo, mask=full), 23, mask=full),
                            dtypes.float32,
                        ),
                        mask=full,
                    )
                    inv_hi = rr.vdup(
                        rr.vreinterpret(
                            rr.vshl(rr.vsub(c254, b_hi, mask=full), 23, mask=full),
                            dtypes.float32,
                        ),
                        mask=full,
                    )
                    inv = rr.vselect(inv_hi, inv_lo, cond_mask=high)
                    yq = rr.vmins(
                        rr.vmaxs(rr.vmul(y, inv, mask=full), -E4M3_MAX, mask=full),
                        E4M3_MAX,
                        mask=full,
                    )
                    yq = rr.vcast(
                        yq, dtypes.float8_e4m3fn, mask=full, reg_layout=RegLayout.ZERO
                    )
                    rr.vstore_pack(y_ub, off, yq, full, pack_mode=rr.PackMode.B32_TO_B8)
                    rr.vstore_first(
                        s_ub, 2 * c, rr.vreinterpret_lanes(b_lo, dtypes.uint8)
                    )
                    rr.vstore_first(
                        s_ub, 2 * c + 1, rr.vreinterpret_lanes(b_hi, dtypes.uint8)
                    )
                rr.vmem_bar("vst_vld")
            vec_sync_all()
            mem_copy(tile_view(y_gm, (1, n), (r, 0)), y_ub)
            mem_copy(tile_view(s_gm, (1, nc, 2), (r, 0, 0)), s_ub)
            vec_sync_all()

    # ---- P2b: plain RMSNorm (BF16 output in GM or caller-owned UB) ----
    @jit
    def _rms_norm_key_value_row_vf(self, h_ub, g_ub, o_ub, h_bf16_ub, xf_ub, n):
        """RMS arithmetic fragment; caller owns scratch and an active VF region."""
        nc = n // VECTOR_CHUNK_ELEMENTS
        full = rr.full_mask()
        # r=FP32(BF16_RN(h))；先舍入完整 atomic 累加结果，再计算 RMS。
        for c in range(nc):
            rounded = rr.vcast(
                rr.vload(h_ub, c * VECTOR_CHUNK_ELEMENTS),
                dtypes.bfloat16,
                mask=full,
                rounding="rn",
                reg_layout=RegLayout.ZERO,
            )
            rr.vstore_pack(
                h_bf16_ub,
                c * VECTOR_CHUNK_ELEMENTS,
                rounded,
                full,
                pack_mode=rr.PackMode.B32_TO_B16,
            )
        rr.vmem_bar("vst_vld")
        acc = rr.vdups(0.0, dtypes.float32, mask=full)
        for c in range(nc):
            xf = rr.vcast(
                rr.vload_unpack(
                    h_bf16_ub,
                    c * VECTOR_CHUNK_ELEMENTS,
                    unpack_mode=rr.UnpackMode.B16_TO_B32,
                ),
                dtypes.float32,
                mask=full,
            )
            rr.vstore(xf_ub, c * VECTOR_CHUNK_ELEMENTS, xf, full)
            acc = rr.vadd(acc, rr.vmul(xf, xf, mask=full), mask=full)
        rr.vmem_bar("vst_vld")
        mean = rr.vmuls(rr.vreduce_sum(acc, mask=full), 1.0 / n, mask=full)
        rstd = rr.vdup(
            rr.vdiv(
                rr.vdups(1.0, dtypes.float32, mask=full),
                rr.vsqrt(rr.vadds(mean, 1.0e-20, mask=full), mask=full),
                mask=full,
            ),
            mask=full,
        )
        for c in range(nc):
            y = rr.vmul(
                rr.vmul(rr.vload(xf_ub, c * VECTOR_CHUNK_ELEMENTS), rstd, mask=full),
                rr.vload(g_ub, c * VECTOR_CHUNK_ELEMENTS),
                mask=full,
            )
            rr.vstore_pack(
                o_ub,
                c * VECTOR_CHUNK_ELEMENTS,
                rr.vcast(y, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ZERO),
                full,
                pack_mode=rr.PackMode.B32_TO_B16,
            )
        rr.vmem_bar("vst_vld")

    @jit
    def _rms_norm_key_value_row_ub(self, h_ub, g_ub, o_ub, n):
        """One row's original RMS math, with explicit BF16 rounding, entirely in UB."""
        h_bf16_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, n)))
        xf_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, n)))
        _nc = n // VECTOR_CHUNK_ELEMENTS
        with vf(mode="simd"):
            self._rms_norm_key_value_row_vf(h_ub, g_ub, o_ub, h_bf16_ub, xf_ub, n)

    @jit
    def _rms_norm_key_value(self, h_gm, g_gm, out_tensor, block, nblk, rows, n):
        h_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, n)))
        g_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, n)))
        o_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, n)))
        mem_copy(g_ub, tile_view(g_gm, (1, n), (0, 0)))
        for r in range(block, rows, nblk):
            mem_copy(
                h_ub,
                tile_view(h_gm, (1, n), (r, 0)),
                options=copy_options(l2_cache_ctl=L2_CACHE_DISABLE),
            )
            vec_sync_all()
            self._rms_norm_key_value_row_ub(h_ub, g_ub, o_ub, n)
            vec_sync_all()
            mem_copy(tile_view(out_tensor, (1, n), (r, 0)), o_ub)
            vec_sync_all()

    @jit
    def _clear_fp32_rows(self, out_gm, block, nblk, rows, n):
        # Each AIV writes disjoint 128-wide tiles, including every tail row.
        z = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 128)))
        for it in range(block * 2 + get_subblock_id(), rows * (n // 128), nblk * 2):
            with vf(mode="simd"):
                mask = rr.full_mask()
                zero = rr.vdups(0.0, dtypes.float32, mask=mask)
                rr.vstore(z, 0, zero, mask)
                rr.vstore(z, 64, zero, mask)
                rr.vmem_bar("vst_vld")
            mem_copy(
                tile_view(out_gm, (1, 128), (it // (n // 128), it % (n // 128))), z
            )
        vec_sync_all()

    @jit
    def _compute_split_k_projection_tile(
        self, k, out_gm, a_gm, b_gm, sa_gm, sb_gm, mi, ni, split
    ):
        valid_m = self.p1_m
        if const_expr(self.m % self.p1_m):
            valid_m = min(self.p1_m, self.m - mi * self.p1_m)
        # K=5120 -> two disjoint 2560 slices, each 5x512 / 10x256.
        # Keep root GM descriptors: offset tile coordinates, never reshape NZ.
        mem_copy(
            _as_zn_view(k.l1_scale_a.next_write())[:valid_m, : HID // 64],
            (tile_view(sa_gm, (self.p1_m, 40, 2), (mi, split, 0))).flatten(-2, -1),
            options=copy_options(l2_cache_ctl=1),
        )
        mem_copy(
            k.l1_scale_b.next_write(),
            (tile_view(sb_gm, (128, 40, 2), (ni, split, 0)))
            .flatten(-2, -1)
            .permute((1, 0)),
            options=copy_options(l2_cache_ctl=1),
        )
        scale_a_slot = _as_zn_view(k.l1_scale_a.next_read())
        scale_b_slot = k.l1_scale_b.next_read()
        l0c_slot = k.l0c.next_write()
        for ki in range(5):
            gki = split * 5 + ki
            mem_copy(
                k.l1_a.next_write()[:valid_m, :KL1_1],
                tile_view(a_gm, (self.p1_m, 512), (mi, gki)),
                options=copy_options(l2_cache_ctl=1),
            )
            mem_copy(
                k.l1_b.next_write(),
                tile_view(b_gm, (128, 512), (ni, gki)),
                options=copy_options(l2_cache_ctl=1),
            )
            a_slot = k.l1_a.next_read()
            b_slot = k.l1_b.next_read()
            for kk in range(2):
                si = ki * 2 + kk
                mem_copy(
                    k.l0a.next_write(),
                    tile_view(a_slot, (self.p1_m, 256), (0, kk)),
                    options=copy_options(
                        mx_scale=tile_view(scale_a_slot, (self.p1_m, 8), (0, si))
                    ),
                )
                mem_copy(
                    k.l0b.next_write(),
                    tile_view(b_slot, (128, 256), (0, kk)),
                    options=copy_options(
                        mx_scale=tile_view(scale_b_slot, (8, 128), (si, 0))
                    ),
                )
                matmul(
                    l0c_slot,
                    k.l0a.next_read(),
                    k.l0b.next_read(),
                    init=(si == 0),
                    unit_flag=3 if si == 9 else 2,
                    mx_mode=True,
                )
        mem_copy(
            tile_view(out_gm, (self.p1_m, 128), (mi, ni)),
            k.l0c.next_read()[:valid_m, :BN1],
            options=copy_options(l2_cache_ctl=1, atomic_add=True, unit_flag=3),
        )

    @jit
    def _run_qa_kv_projection(self, qr, kv, x8, wa, wkv, sx, swa, swkv, block):
        # 仅 P1 的 FP8 L0B 双缓冲；N128×K256 每份 32 KiB。
        c1 = _create_projection_pipeline(
            self.p1_m, BN1, BK1, KL1_1, HID // 2, l0b_depth=2
        )
        _bind_projection_pipeline(self.k1, c1)
        _bind_projection_pipeline(self.k3, c1)
        # P1 alone uses 28 AIC; blocks 28..31 still join all phase barriers.
        # T64 has one M64 tile, not four M16 tiles.
        for mi in range(self.p1_m_tiles):
            if block < 20:
                self._compute_split_k_projection_tile(
                    self.k1, qr, x8, wa, sx, swa, mi, block // 2, block % 2
                )
            elif block < 28:
                self._compute_split_k_projection_tile(
                    self.k3, kv, x8, wkv, sx, swkv, mi, (block - 20) // 2, block % 2
                )

    @kernel
    def qkv_projection_kernel(
        self,
        qr_gm: Tensor,
        kv_gm: Tensor,
        qfull_gm: Tensor,
        x_gm: Tensor,
        gfq_gm: Tensor,
        gfk_gm: Tensor,
        wa_gm: Tensor,
        wkv_gm: Tensor,
        qn_gm: Tensor,
        wb_gm: Tensor,
        x8_gm: Tensor,
        sx_gm: Tensor,
        qn8_gm: Tensor,
        sqn_gm: Tensor,
        swa_gm: Tensor,
        swkv_gm: Tensor,
        swb_gm: Tensor,
        kvn_gm: Tensor,
    ):
        block, nblk = get_block_idx(), get_block_num()
        mt = self.m_tiles
        # Vector zero is included in the P0 completion barrier.
        self._clear_fp32_rows(qr_gm, block, nblk, self.m, QLORA)
        self._clear_fp32_rows(kv_gm, block, nblk, self.m, KVD)
        # P0
        self._quantize_mxfp8_rows(x_gm, x8_gm, sx_gm, block, nblk, self.m, HID)
        global_sync_all()
        self._run_qa_kv_projection(
            qr_gm, kv_gm, x8_gm, wa_gm, wkv_gm, sx_gm, swa_gm, swkv_gm, block
        )
        global_sync_all()
        # P2a: q_norm + MX quant (needed by wq_b)
        self._rms_norm_quantize_query(
            qr_gm, gfq_gm, qn8_gm, sqn_gm, block, nblk, self.m, QLORA
        )
        global_sync_all()
        # P3: wq_b (AIC) overlapped with kv_norm (AIV)
        vec_sync_all()
        onchip_rewind(reset_sync_id=True)
        self._rms_norm_key_value(kv_gm, gfk_gm, kvn_gm, block, nblk, self.m, KVD)
        c2 = _create_projection_pipeline(16, BN2, BK2, KL1_2, SKL1_2)
        _bind_projection_pipeline(self.k2, c2)
        for it in range(block, mt * self.n2_blocks, CUBES):
            self.k2._compute_output_tile(
                qfull_gm,
                qn8_gm,
                wb_gm,
                sqn_gm,
                swb_gm,
                it // self.n2_blocks,
                it % self.n2_blocks,
                None,
            )


class QkvProjectionOp:
    def __init__(self, m=16):
        self.kernel = QkvProjectionKernel(m)

    @host
    def run(
        self,
        qr,
        kv,
        qfull,
        x,
        gfq,
        gfk,
        wa,
        wkv,
        qn,
        wb,
        x8,
        sx,
        qn8,
        sqn,
        swa,
        swkv,
        swb,
        kvn,
    ):
        self.kernel.qkv_projection_kernel[CUBES](
            qr,
            kv,
            qfull,
            x,
            gfq,
            gfk,
            wa,
            wkv,
            qn,
            wb,
            x8,
            sx,
            qn8,
            sqn,
            swa,
            swkv,
            swb,
            kvn,
        )
