# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Native HIF8 FlashAttention on the flash_attn_fp8_fullquant pipeline.

Architecture (ported from the verified FP8 full-quant sample, dense first
phase):

- AIC: ``S = K @ Q^T`` with HIF8 K on L0A and HIF8 Q on L0B, FP32 L0C;
  the per-tensor ``softmax_scale * descale_q * descale_k`` factor folds
  into the FP32 softmax pass as one compile-time constant.
- AIV: FP32 online softmax (max state carries ``-log(p_scale)``), the
  probability tile is cast to HIF8 with the four-slot register pack and
  byte-gathered into the NZ layout (same primitive shape as the FP8 P).
- AIC: ``O += P^T @ V`` with HIF8 P/V and FP32 L0C.
- AIV: ``acc = acc * corr + PV * descale_v``, finalize ``O = acc / l`` in
  BF16; the online sum ``l`` stays FP32 so the division cancels the
  p_scale factor exactly like the AscendC reference.

Tile 128x256x128, two AIVs of 64 Q rows per cube core, SKEW_LAG=2
pipeline, dense BNSD KV (no paged cache), causal or no mask.
"""

__all__ = ["npu_flash_atten_hif8_fullquant"]

import math
import struct
import threading
import weakref

import cannbotdsl
import torch
from cannbotdsl import (
    MemLoc,
    RegLayout,
    RingBuffer,
    Tensor,
    const_expr,
    dtypes,
    from_tensor,
    get_block_idx,
    get_subblock_id,
    jit,
    kernel,
    matmul,
    vf,
)
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.host import host
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.reg import (
    update_mask,
    vadd,
    vadds,
    varange,
    vcast,
    vdiv,
    vexp_sub,
    vgather_reg,
    vles,
    vload,
    vload_deinterleave,
    vlog,
    vmadd,
    vmax,
    vmem_bar,
    vmul,
    vmuls,
    vreinterpret,
    vselect,
    vstore,
    vstore_first,
    vstore_strided,
)
from cannbotdsl.reg import (
    vbitwise_or as vor,
)
from cannbotdsl.reg import (
    vdups as vdup_scalar,
)
from cannbotdsl.reg import (
    vle as vcmp_le,
)
from cannbotdsl.reg import (
    vload_broadcast as vload_brc,
)
from cannbotdsl.tensor import (
    ceil_div,
    make_layout,
    make_tiler,
    tile_slice,
)

try:
    from .hif8_check import (
        validate_lengths,
        validate_scales,
        validate_static,
    )
except ImportError:
    # Loaded as a standalone module (by file path, no package context):
    # import the sibling checker directly so the sample stays self-contained.
    import importlib.util
    from pathlib import Path

    _checker_spec = importlib.util.spec_from_file_location(
        "hif8_check",
        Path(__file__).with_name("hif8_check.py"),
    )
    _checker = importlib.util.module_from_spec(_checker_spec)
    _checker_spec.loader.exec_module(_checker)
    validate_lengths = _checker.validate_lengths
    validate_scales = _checker.validate_scales
    validate_static = _checker.validate_static

M_BASE_SIZE = 128
S2_BASE_SIZE = 256
D = 128
# MM2 trails the softmax by this many KV blocks. Three blocks beat two by
# ~1% at both 32K and 64K (paired runs): the extra P slot buys more slack
# between the AIV softmax and the AIC MM2, paid for by one K/V L1 slot.
# Four is worse again -- dropping kv_l1 to two slots exposes the K/V DMA.
SKEW_LAG = 3
PASS_A_UNROLL = 4
P_NZ_FRAGMENT_WIDTH = 32
P_NZ_FRAGMENTS_PER_L1_TILE_ROW = 2
NZ_N1_PADDING = 32
FP32_VECTOR_LANES = 64
NEG_MAX = -3.4028234663852886e38
# Cap on Flash Decode KV splits per (batch, head) row block. The merge
# kernel keeps one weight row per split in UB, so this bounds its footprint.
FD_MAX_SPLITS = 32


def _f32(value):
    """Round a Python float through FP32, for bit-exact host constants."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


MASK_NONE = 0
MASK_CAUSAL = 1
MASK_KV_TAIL = 2

# TND KV repack threshold: at or above this many KV heads the token-major
# row stride (nkv * 128 B) wastes over 75% of every DMA burst, so a one-pass
# contiguous repack pays for itself (see the wrapper).
KV_DENSE_COPY_MIN_HEADS = 4
KV_DENSE_COPY_MIN_TOKENS = 1024

# Q repack threshold: the token-major row stride is nq * 128 B, so the useful
# fraction of a burst is 1/nq. From eight heads upwards one linear repack is
# repaid by the dense tile fetches of a long sequence.
Q_DENSE_COPY_MIN_HEADS = 8
Q_DENSE_COPY_MIN_TOKENS = 4096

# L1 budget: q_l1 2x16K + kv_l1 3x32K + p_l1 4x32K = 256 KB.
DEPTHS = {
    "q_l1": 2,
    "kv_l1": 3,
    "p_l1": 4,
    "l0a": 2,
    "l0b": 2,
    "l0c_mm1": 1,
    "l0c_mm2": 2,
    "s_ub": 2,
    "pv_ub": 1,
    "p_nz": 2,
    "output_bf16_ub": 1,
}


def nz_block_stride(s2_base_size=S2_BASE_SIZE):
    return (s2_base_size // 4) | 1


def nz_gather_index(m_base_size_per_aiv):
    return [i + (j << 2) for i in range(4) for j in range(m_base_size_per_aiv)]


def _query_row_tile(gm, layout, batch_idx, head_idx, query_token_start, rows, width):
    alignment = (16, 32) if isinstance(rows, int) and width == D else (1, 1)
    token_end = query_token_start + rows
    if layout != "TND":
        base = gm[batch_idx, head_idx, query_token_start:token_end, None]
        return tile_slice(base, make_tiler((rows, width), alignment=alignment), (0, 0))

    # Kernel-side TND storage is head-major (N, T, D) like the FP8 sample.
    base = gm[head_idx, query_token_start:token_end, None]
    return tile_slice(base, make_tiler((rows, width), alignment=alignment), (0, 0))


def _lse_row_tile(gm, layout, batch_idx, head_idx, query_token_start, rows):
    """One flat FP32 row window for the LSE writeback."""
    token_end = query_token_start + rows
    if layout != "TND":
        base = gm[batch_idx, head_idx, query_token_start:token_end]
    else:
        base = gm[head_idx, query_token_start:token_end]
    return tile_slice(base, make_tiler((rows,), alignment=(16,)), (0,))


def _dense_kv_tile(gm, layout, batch_idx, kv_head, token_start, rows):
    """Dense KV rows for one 256-wide tile (BNSD/TND native layouts)."""
    token_end = token_start + rows
    if layout != "TND":
        base = gm[batch_idx, kv_head, token_start:token_end, None]
    else:
        base = gm[kv_head, token_start:token_end, None]
    return tile_slice(base, make_tiler((rows, D), alignment=(16, 32)), (0, 0))


def _kv_rows(gm, in_layout, batch_idx, kv_head, token_start, rows, kv_s_pad, kv_n):
    """Per-row contiguous GM views for one strided KV tile (BSND/TND input).

    The wrapper passes Q/K/V as rank-2 flats (tokens*heads, D); one strided
    tile is then a regular burst of D-byte rows. The static tuple-range loop
    unrolls at trace time; the row tuple feeds one multi-source mem_copy
    (single strided ND2NZ DMA, no permute copy).
    """
    out = [None] * rows
    if const_expr(in_layout == "BSND"):
        base = batch_idx * kv_s_pad * kv_n + kv_head
        for i in tuple(range(rows)):
            out[i] = gm[base + (token_start + i) * kv_n, 0:D]
    else:  # TND token-major: token_start already packs batches
        for i in tuple(range(rows)):
            out[i] = gm[(token_start + i) * kv_n + kv_head, 0:D]
    return tuple(out)


def _q_rows(gm, in_layout, batch_idx, head_idx, query_token_start, rows, q_s_pad, q_n):
    """Per-row contiguous GM views for one strided Q tile (BSND/TND input)."""
    out = [None] * rows
    if const_expr(in_layout == "BSND"):
        base = batch_idx * q_s_pad * q_n + head_idx
        for i in tuple(range(rows)):
            out[i] = gm[base + (query_token_start + i) * q_n, 0:D]
    else:  # TND token-major
        for i in tuple(range(rows)):
            out[i] = gm[(query_token_start + i) * q_n + head_idx, 0:D]
    return tuple(out)


class Matmul:
    """HIF8 Cube side: dense K/V loads, MM1 ``K@Q^T``, MM2 ``P^T@V``."""

    def __init__(self, m_base_size, s2_base_size, d, depths, kv_geom=(0, 0)):
        self.m_base_size, self.s2_base_size, self.d = m_base_size, s2_base_size, d
        # Strided-input geometry stays compile-time: the multi-source row DMA
        # must prove equal source intervals, which needs constant strides.
        self.kv_s_pad, self.kv_n = kv_geom
        self.q_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.int8,
            make_layout((m_base_size, d)),
            depth=depths["q_l1"],
        )
        self.kv_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.int8,
            make_layout((s2_base_size, d)),
            depth=depths["kv_l1"],
        )
        self.l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.hifloat8,
            make_layout((s2_base_size, d)),
            depth=depths["l0a"],
        )
        self.l0b = RingBuffer(
            MemLoc.L0B,
            dtypes.hifloat8,
            make_layout((s2_base_size, d)),
            depth=depths["l0b"],
        )
        self.l0c_mm1 = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((s2_base_size, m_base_size)),
            depth=depths["l0c_mm1"],
        )
        self.l0c_mm2 = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((m_base_size, d)),
            depth=depths["l0c_mm2"],
        )

    @jit
    def load_q(self, q_tile):
        q_slot = self.q_l1.next_write()
        mem_copy(q_slot, q_tile)
        return q_slot

    @jit
    def load_k(
        self,
        gm_key,
        layout,
        in_layout,
        batch_idx,
        kv_head,
        kv_block_idx,
        kv_token_base=0,
    ):
        slot = self.kv_l1.next_write()
        token_start = kv_token_base + kv_block_idx * self.s2_base_size
        if const_expr(in_layout != "BSND"):
            src = _dense_kv_tile(
                gm_key,
                layout,
                batch_idx,
                kv_head,
                token_start,
                self.s2_base_size,
            )
        else:
            src = _kv_rows(
                gm_key,
                in_layout,
                batch_idx,
                kv_head,
                token_start,
                self.s2_base_size,
                self.kv_s_pad,
                self.kv_n,
            )
        mem_copy(slot, src)

    @jit
    def compute_qk(self, score_ub, q_slot):
        key_l0a = self.l0a.next_write()
        query_l0b = from_tensor(
            self.l0b.next_write(),
            dtypes.hifloat8,
            make_layout((self.m_base_size, self.d)),
            data_format="nz",
        )
        score_l0c = self.l0c_mm1.next_write()
        mem_copy(
            key_l0a,
            from_tensor(
                self.kv_l1.next_read(),
                dtypes.hifloat8,
                make_layout((self.s2_base_size, self.d)),
                data_format="nz",
            ),
        )
        mem_copy(
            query_l0b,
            from_tensor(
                q_slot,
                dtypes.hifloat8,
                make_layout((self.m_base_size, self.d)),
                data_format="nz",
            ),
        )
        # UnitFlag is intentionally off: partition(dim=1) is a dual-target
        # fixpipe (each AIV reads half of the 256x128 L0C tile), while the
        # UnitFlag contract requires the moved volume to match the MMAD
        # volume; a partial move leaves the fractal flags inconsistent and
        # the reader can observe an unwritten UB half.
        matmul(score_l0c, key_l0a, query_l0b, init=True, unit_flag=3)
        mem_copy(
            score_ub.next_write(),
            score_l0c.partition(dim=1),
            options=copy_options(unit_flag=3),
        )

    @jit
    def load_v(
        self,
        gm_value,
        layout,
        in_layout,
        batch_idx,
        kv_head,
        kv_block_idx,
        kv_token_base=0,
    ):
        slot = self.kv_l1.next_write()
        token_start = kv_token_base + kv_block_idx * self.s2_base_size
        if const_expr(in_layout != "BSND"):
            src = _dense_kv_tile(
                gm_value,
                layout,
                batch_idx,
                kv_head,
                token_start,
                self.s2_base_size,
            )
        else:
            src = _kv_rows(
                gm_value,
                in_layout,
                batch_idx,
                kv_head,
                token_start,
                self.s2_base_size,
                self.kv_s_pad,
                self.kv_n,
            )
        mem_copy(slot, src)

    @jit
    def compute_pv(self, p_l1, output_ub):
        probability_l0a = from_tensor(
            self.l0a.next_write(),
            dtypes.hifloat8,
            make_layout((self.m_base_size, self.s2_base_size)),
            data_format="nz",
        )
        value_l0b = from_tensor(
            self.l0b.next_write(),
            dtypes.hifloat8,
            make_layout((self.d, self.s2_base_size)),
            data_format="nz",
        )
        output_l0c = self.l0c_mm2.next_write()
        mem_copy(probability_l0a, p_l1.next_read().permute(1, 0))
        mem_copy(
            value_l0b,
            from_tensor(
                self.kv_l1.next_read(),
                dtypes.hifloat8,
                make_layout((self.s2_base_size, self.d)),
                data_format="nz",
            ).permute(1, 0),
        )
        matmul(output_l0c, probability_l0a, value_l0b, init=True, unit_flag=3)
        mem_copy(
            output_ub.next_write(),
            output_l0c.partition(dim=0),
            options=copy_options(unit_flag=3),
        )


class Vector:
    """HIF8 vector side: online softmax, HIF8 P pack, output accumulation."""

    def __init__(
        self, m_base_size_per_aiv, s2_base_size, d, depths, qk_scale, v_descale, p_scale
    ):
        self.m_base_size_per_aiv = m_base_size_per_aiv
        self.s2_base_size, self.d = s2_base_size, d
        # Scales are trace-time constants folded into the softmax: a caller
        # that holds its descales/p_scale fixed gets the leanest ABI, and the
        # compiled-kernel cache below bounds the variants. Changing a value
        # compiles a new artifact.
        self.qk_scale = float(qk_scale)
        self.v_descale = float(v_descale)
        self.log_p_scale = math.log(float(p_scale)) if p_scale else 0.0
        # Sentinel row_m of a fully-masked row that still passes through a
        # softmax block: f32(f32(NEG_MAX * qk_scale) - log(p_scale)), the same
        # op sequence as the softmax Pass A (bit-exact). Rows skipped by
        # kv_block_count keep the init value NEG_MAX. finalize_output zeroes
        # any row whose max state is at or below this threshold, mirroring
        # AscendC's RowInvalidUpdateVF (masked softmax makes exp(0) = 1 per
        # key, which would otherwise emit mean(V) garbage instead of 0).
        # For qk_scale > 1 the FP32 product overflows: hardware Muls turns
        # the masked max into -inf and the max() against the NEG_MAX init
        # state keeps NEG_MAX, so clamp the trace-time constant the same way.
        _scaled_neg = max(NEG_MAX * self.qk_scale, NEG_MAX)
        _sentinel = _f32(_f32(_scaled_neg) + _f32(-self.log_p_scale))
        self.invalid_row_threshold = max(_sentinel, NEG_MAX)
        self.p_nz = RingBuffer(
            MemLoc.UB,
            dtypes.hifloat8,
            make_layout((m_base_size_per_aiv, s2_base_size)),
            depth=depths["p_nz"],
            data_format="nz",
            physical_layout=make_layout(
                (s2_base_size // 32, m_base_size_per_aiv // 16, 16, 32),
                stride=(m_base_size_per_aiv * 32 + NZ_N1_PADDING, 16 * 32, 32, 1),
            ),
        )
        self.nz_gather_indices = nz_gather_index(m_base_size_per_aiv)
        self.nz_gather_indices_ub = Buffer(
            MemLoc.UB,
            dtypes.uint8,
            make_layout((1, len(self.nz_gather_indices))),
        )
        self.nz_gather_indices_channel = RingBuffer(
            MemLoc.UB,
            dtypes.uint8,
            make_layout((1, len(self.nz_gather_indices))),
            depth=1,
        ).next_write()
        self.output_bf16_ub = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((m_base_size_per_aiv, d)),
            depth=depths["output_bf16_ub"],
        )
        self.lse_stage = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((m_base_size_per_aiv,))
        )
        self.acc = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((m_base_size_per_aiv, d))
        )
        self.row_m = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((m_base_size_per_aiv, 1))
        )
        self.row_l = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((m_base_size_per_aiv, 1))
        )
        self.corr_buffers = [
            Buffer(MemLoc.UB, dtypes.float32, make_layout((m_base_size_per_aiv, 1)))
            for _ in range(SKEW_LAG + 1)
        ]
        self.p_nz_fragment_width = P_NZ_FRAGMENT_WIDTH
        self.nz_group_rows = s2_base_size // 4
        self.nz_block_stride_value = nz_block_stride(s2_base_size)

    @jit
    def initialize(self):
        with vf(mode="simd"):
            row_mask, _ = update_mask(self.m_base_size_per_aiv, elem_bits=32)
            vstore(
                self.row_m,
                0,
                vdup_scalar(NEG_MAX, dtypes.float32, mask=row_mask),
                row_mask,
            )
            vstore(
                self.row_l,
                0,
                vdup_scalar(0.0, dtypes.float32, mask=row_mask),
                row_mask,
            )
            mf, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
            zero = vdup_scalar(0.0, dtypes.float32, mask=mf)
            for row in range(self.m_base_size_per_aiv):
                vstore(self.acc, row * self.d, zero, mf)
                vstore(self.acc, row * self.d + FP32_VECTOR_LANES, zero, mf)

    @jit
    def softmax(
        self, score_ub, first_query_visible_limit, pipeline_slot, kv_block_mask_type
    ):
        """Per-tensor-constant version of the FP8 full-quant softmax pass.

        ``m_new = max(m_old, max(S*qk_scale) - log(p_scale))``;
        ``P = cast_hif8(exp(S*qk_scale - m_new))``;
        ``l_new = l_old*corr + sum(exp)`` (FP32, pre-cast).
        """
        p_slot = self.p_nz.next_write()
        corr_buffer = self.corr_buffers[pipeline_slot]
        with vf(mode="simd"):
            fp32_mask, _ = update_mask(self.m_base_size_per_aiv, elem_bits=32)
            fp8_mask, _ = update_mask(
                PASS_A_UNROLL * self.m_base_size_per_aiv, elem_bits=8
            )
            nz_gather_indices = vload(self.nz_gather_indices_ub, 0)
            negative_max = vdup_scalar(NEG_MAX, dtypes.float32, mask=fp32_mask)

            visible_limit_vectors = []
            if const_expr(kv_block_mask_type != MASK_NONE):
                if const_expr(kv_block_mask_type == MASK_CAUSAL):
                    visible_limit_vector = vadd(
                        vdup_scalar(
                            first_query_visible_limit,
                            dtypes.float32,
                            mask=fp32_mask,
                        ),
                        varange(0, dtypes.float32),
                        mask=fp32_mask,
                    )
                else:
                    visible_limit_vector = vdup_scalar(
                        first_query_visible_limit,
                        dtypes.float32,
                        mask=fp32_mask,
                    )
                visible_limit_vectors.append(visible_limit_vector)
                visible_limit_vectors.append(
                    vadds(visible_limit_vector, -1.0, mask=fp32_mask)
                )
                visible_limit_vectors.append(
                    vadds(visible_limit_vector, -2.0, mask=fp32_mask)
                )
                visible_limit_vectors.append(
                    vadds(visible_limit_vector, -3.0, mask=fp32_mask)
                )

            kv_group_start = vdup_scalar(0.0, dtypes.float32, mask=fp32_mask)

            # AscendC semantics (vf_mul_sel...dn.h): the no-mask Pass A is a
            # pure load+max on RAW scores; the scale folds into one tile-max
            # multiply. Masked blocks keep the select on raw values (the
            # NEG_MAX sentinel is scale-invariant) with the write-back that
            # Pass B then reads pre-masked.
            block_max = negative_max
            for kv_row_start in range(0, self.s2_base_size, PASS_A_UNROLL):
                raw_score_0 = vload(
                    score_ub, (kv_row_start + 0) * self.m_base_size_per_aiv
                )
                raw_score_1 = vload(
                    score_ub, (kv_row_start + 1) * self.m_base_size_per_aiv
                )
                raw_score_2 = vload(
                    score_ub, (kv_row_start + 2) * self.m_base_size_per_aiv
                )
                raw_score_3 = vload(
                    score_ub, (kv_row_start + 3) * self.m_base_size_per_aiv
                )
                if const_expr(kv_block_mask_type != MASK_NONE):
                    raw_score_0 = vselect(
                        raw_score_0,
                        negative_max,
                        cond_mask=vcmp_le(
                            kv_group_start, visible_limit_vectors[0], mask=fp32_mask
                        ),
                    )
                    raw_score_1 = vselect(
                        raw_score_1,
                        negative_max,
                        cond_mask=vcmp_le(
                            kv_group_start, visible_limit_vectors[1], mask=fp32_mask
                        ),
                    )
                    raw_score_2 = vselect(
                        raw_score_2,
                        negative_max,
                        cond_mask=vcmp_le(
                            kv_group_start, visible_limit_vectors[2], mask=fp32_mask
                        ),
                    )
                    raw_score_3 = vselect(
                        raw_score_3,
                        negative_max,
                        cond_mask=vcmp_le(
                            kv_group_start, visible_limit_vectors[3], mask=fp32_mask
                        ),
                    )
                    kv_group_start = vadds(
                        kv_group_start, float(PASS_A_UNROLL), mask=fp32_mask
                    )
                    vstore(
                        score_ub,
                        (kv_row_start + 0) * self.m_base_size_per_aiv,
                        raw_score_0,
                        fp32_mask,
                    )
                    vstore(
                        score_ub,
                        (kv_row_start + 1) * self.m_base_size_per_aiv,
                        raw_score_1,
                        fp32_mask,
                    )
                    vstore(
                        score_ub,
                        (kv_row_start + 2) * self.m_base_size_per_aiv,
                        raw_score_2,
                        fp32_mask,
                    )
                    vstore(
                        score_ub,
                        (kv_row_start + 3) * self.m_base_size_per_aiv,
                        raw_score_3,
                        fp32_mask,
                    )
                pair_max_01 = vmax(raw_score_0, raw_score_1, mask=fp32_mask)
                pair_max_23 = vmax(raw_score_2, raw_score_3, mask=fp32_mask)
                four_row_max = vmax(pair_max_01, pair_max_23, mask=fp32_mask)
                block_max = vmax(block_max, four_row_max, mask=fp32_mask)

            block_max = vmuls(block_max, self.qk_scale, mask=fp32_mask)
            adjusted_block_max = vadds(block_max, -self.log_p_scale, mask=fp32_mask)
            old_block_max = vload(self.row_m, 0)
            new_block_max = vmax(old_block_max, adjusted_block_max, mask=fp32_mask)
            current_block_corr = vexp_sub(old_block_max, new_block_max, mask=fp32_mask)
            vstore(self.row_m, 0, new_block_max, fp32_mask)
            vstore(corr_buffer, 0, current_block_corr, fp32_mask)

            vmem_bar("vst_vld")

            zero = vdup_scalar(0.0, dtypes.float32, mask=fp32_mask)
            partial_sum_0, partial_sum_1 = zero, zero
            partial_sum_2, partial_sum_3 = zero, zero
            partial_sum_4, partial_sum_5 = zero, zero
            partial_sum_6, partial_sum_7 = zero, zero

            for nz_group_row in range(0, self.nz_group_rows, 2):
                score_row_0 = vload(
                    score_ub,
                    (nz_group_row + 0 * self.nz_group_rows) * self.m_base_size_per_aiv,
                )
                score_row_1 = vload(
                    score_ub,
                    (nz_group_row + 1 * self.nz_group_rows) * self.m_base_size_per_aiv,
                )
                score_row_2 = vload(
                    score_ub,
                    (nz_group_row + 2 * self.nz_group_rows) * self.m_base_size_per_aiv,
                )
                score_row_3 = vload(
                    score_ub,
                    (nz_group_row + 3 * self.nz_group_rows) * self.m_base_size_per_aiv,
                )
                score_row_4 = vload(
                    score_ub,
                    (nz_group_row + 0 * self.nz_group_rows + 1)
                    * self.m_base_size_per_aiv,
                )
                score_row_5 = vload(
                    score_ub,
                    (nz_group_row + 1 * self.nz_group_rows + 1)
                    * self.m_base_size_per_aiv,
                )
                score_row_6 = vload(
                    score_ub,
                    (nz_group_row + 2 * self.nz_group_rows + 1)
                    * self.m_base_size_per_aiv,
                )
                score_row_7 = vload(
                    score_ub,
                    (nz_group_row + 3 * self.nz_group_rows + 1)
                    * self.m_base_size_per_aiv,
                )
                if const_expr(self.qk_scale != 1.0):
                    score_row_0 = vmuls(score_row_0, self.qk_scale, mask=fp32_mask)
                    score_row_1 = vmuls(score_row_1, self.qk_scale, mask=fp32_mask)
                    score_row_2 = vmuls(score_row_2, self.qk_scale, mask=fp32_mask)
                    score_row_3 = vmuls(score_row_3, self.qk_scale, mask=fp32_mask)
                    score_row_4 = vmuls(score_row_4, self.qk_scale, mask=fp32_mask)
                    score_row_5 = vmuls(score_row_5, self.qk_scale, mask=fp32_mask)
                    score_row_6 = vmuls(score_row_6, self.qk_scale, mask=fp32_mask)
                    score_row_7 = vmuls(score_row_7, self.qk_scale, mask=fp32_mask)
                exp_score_0 = vexp_sub(score_row_0, new_block_max, mask=fp32_mask)
                exp_score_1 = vexp_sub(score_row_1, new_block_max, mask=fp32_mask)
                exp_score_2 = vexp_sub(score_row_2, new_block_max, mask=fp32_mask)
                exp_score_3 = vexp_sub(score_row_3, new_block_max, mask=fp32_mask)
                exp_score_4 = vexp_sub(score_row_4, new_block_max, mask=fp32_mask)
                exp_score_5 = vexp_sub(score_row_5, new_block_max, mask=fp32_mask)
                exp_score_6 = vexp_sub(score_row_6, new_block_max, mask=fp32_mask)
                exp_score_7 = vexp_sub(score_row_7, new_block_max, mask=fp32_mask)
                partial_sum_0 = vadd(partial_sum_0, exp_score_0, mask=fp32_mask)
                partial_sum_1 = vadd(partial_sum_1, exp_score_1, mask=fp32_mask)
                partial_sum_2 = vadd(partial_sum_2, exp_score_2, mask=fp32_mask)
                partial_sum_3 = vadd(partial_sum_3, exp_score_3, mask=fp32_mask)
                partial_sum_4 = vadd(partial_sum_4, exp_score_4, mask=fp32_mask)
                partial_sum_5 = vadd(partial_sum_5, exp_score_5, mask=fp32_mask)
                partial_sum_6 = vadd(partial_sum_6, exp_score_6, mask=fp32_mask)
                partial_sum_7 = vadd(partial_sum_7, exp_score_7, mask=fp32_mask)

                hif8_exp_0 = vcast(
                    exp_score_0,
                    dtypes.hifloat8,
                    mask=fp32_mask,
                    saturate=True,
                    rounding="rn",
                    reg_layout=RegLayout.ZERO,
                )
                hif8_exp_1 = vcast(
                    exp_score_1,
                    dtypes.hifloat8,
                    mask=fp32_mask,
                    saturate=True,
                    rounding="rn",
                    reg_layout=RegLayout.ONE,
                )
                hif8_exp_2 = vcast(
                    exp_score_2,
                    dtypes.hifloat8,
                    mask=fp32_mask,
                    saturate=True,
                    rounding="rn",
                    reg_layout=RegLayout.TWO,
                )
                hif8_exp_3 = vcast(
                    exp_score_3,
                    dtypes.hifloat8,
                    mask=fp32_mask,
                    saturate=True,
                    rounding="rn",
                    reg_layout=RegLayout.THREE,
                )
                packed_group_0 = vor(
                    vor(
                        vreinterpret(hif8_exp_0, dtypes.uint8),
                        vreinterpret(hif8_exp_1, dtypes.uint8),
                        mask=fp8_mask,
                    ),
                    vor(
                        vreinterpret(hif8_exp_2, dtypes.uint8),
                        vreinterpret(hif8_exp_3, dtypes.uint8),
                        mask=fp8_mask,
                    ),
                    mask=fp8_mask,
                )
                vstore_strided(
                    p_slot,
                    (nz_group_row + 0) * self.p_nz_fragment_width,
                    vgather_reg(packed_group_0, nz_gather_indices),
                    fp8_mask,
                    block_stride=self.nz_block_stride_value,
                    repeat_stride=0,
                )

                hif8_exp_4 = vcast(
                    exp_score_4,
                    dtypes.hifloat8,
                    mask=fp32_mask,
                    saturate=True,
                    rounding="rn",
                    reg_layout=RegLayout.ZERO,
                )
                hif8_exp_5 = vcast(
                    exp_score_5,
                    dtypes.hifloat8,
                    mask=fp32_mask,
                    saturate=True,
                    rounding="rn",
                    reg_layout=RegLayout.ONE,
                )
                hif8_exp_6 = vcast(
                    exp_score_6,
                    dtypes.hifloat8,
                    mask=fp32_mask,
                    saturate=True,
                    rounding="rn",
                    reg_layout=RegLayout.TWO,
                )
                hif8_exp_7 = vcast(
                    exp_score_7,
                    dtypes.hifloat8,
                    mask=fp32_mask,
                    saturate=True,
                    rounding="rn",
                    reg_layout=RegLayout.THREE,
                )
                packed_group_1 = vor(
                    vor(
                        vreinterpret(hif8_exp_4, dtypes.uint8),
                        vreinterpret(hif8_exp_5, dtypes.uint8),
                        mask=fp8_mask,
                    ),
                    vor(
                        vreinterpret(hif8_exp_6, dtypes.uint8),
                        vreinterpret(hif8_exp_7, dtypes.uint8),
                        mask=fp8_mask,
                    ),
                    mask=fp8_mask,
                )
                vstore_strided(
                    p_slot,
                    (nz_group_row + 1) * self.p_nz_fragment_width,
                    vgather_reg(packed_group_1, nz_gather_indices),
                    fp8_mask,
                    block_stride=self.nz_block_stride_value,
                    repeat_stride=0,
                )

            current_block_sum = vadd(
                vadd(
                    vadd(partial_sum_0, partial_sum_1, mask=fp32_mask),
                    vadd(partial_sum_2, partial_sum_3, mask=fp32_mask),
                    mask=fp32_mask,
                ),
                vadd(
                    vadd(partial_sum_4, partial_sum_5, mask=fp32_mask),
                    vadd(partial_sum_6, partial_sum_7, mask=fp32_mask),
                    mask=fp32_mask,
                ),
                mask=fp32_mask,
            )
            vstore(
                self.row_l,
                0,
                vadd(
                    vmul(vload(self.row_l, 0), current_block_corr, mask=fp32_mask),
                    current_block_sum,
                    mask=fp32_mask,
                ),
                fp32_mask,
            )

    @jit
    def store_p(self, p_l1, vector_subblock_index):
        p_source = self.p_nz.next_read()
        p_destination = p_l1.next_write()
        fragment_shape = (self.m_base_size_per_aiv, self.p_nz_fragment_width)
        fragment_count = self.s2_base_size // self.p_nz_fragment_width
        for fragment_index in tuple(range(fragment_count)):
            mem_copy(
                tile_slice(
                    p_destination,
                    fragment_shape,
                    (
                        fragment_index // P_NZ_FRAGMENTS_PER_L1_TILE_ROW,
                        vector_subblock_index * P_NZ_FRAGMENTS_PER_L1_TILE_ROW
                        + fragment_index % P_NZ_FRAGMENTS_PER_L1_TILE_ROW,
                    ),
                ),
                tile_slice(p_source, fragment_shape, (0, fragment_index)),
            )

    @jit
    def update_output_acc(self, pv_ub, pipeline_slot):
        corr_buffer = self.corr_buffers[pipeline_slot]
        with vf(mode="simd"):
            # Masks are loop-invariant (d is 64-aligned): build once.
            fp32_mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
            for query_row_index in range(self.m_base_size_per_aiv):
                corr_broadcast = vload_brc(corr_buffer, query_row_index)
                for d_offset in (0, FP32_VECTOR_LANES):
                    output_offset = query_row_index * self.d + d_offset
                    pv_value = vload(pv_ub, output_offset)
                    if const_expr(self.v_descale != 1.0):
                        pv_value = vmuls(pv_value, self.v_descale, mask=fp32_mask)
                    vstore(
                        self.acc,
                        output_offset,
                        vmadd(
                            vload(self.acc, output_offset),
                            corr_broadcast,
                            pv_value,
                            mask=fp32_mask,
                        ),
                        fp32_mask,
                    )

    @jit
    def finalize_output(
        self,
        gm_output,
        gm_lse,
        layout,
        vector_subblock_index,
        batch_index,
        query_head_index,
        query_token_start,
        valid_query_rows,
    ):
        """``O = acc / l`` cast to BF16 and written back to GM.

        When ``gm_lse`` is provided, also emits the true FP32
        ``LSE = (m + log p_scale) + log(l / p_scale) = m + log l`` per row:
        the p_scale factors carried by row_m and row_l cancel exactly.
        Fully masked rows keep l = 0 and export +inf.
        """
        output_slot = self.output_bf16_ub.next_write()
        with vf(mode="simd"):
            fp32_mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
            bf16_mask, _ = update_mask(2 * FP32_VECTOR_LANES, elem_bits=16)
            for query_row_index in range(self.m_base_size_per_aiv):
                normalization_denominator = vload_brc(self.row_l, query_row_index)
                row_offset = query_row_index * self.d
                even_acc, odd_acc = vload_deinterleave(
                    self.acc, row_offset, width="b32"
                )
                # Fully-masked rows (causal q > kv): zero the row so it
                # matches AscendC semantics instead of mean(V) garbage.
                row_max = vload_brc(self.row_m, query_row_index)
                threshold_vec = vdup_scalar(
                    self.invalid_row_threshold, dtypes.float32, mask=fp32_mask
                )
                invalid_row = vcmp_le(row_max, threshold_vec, mask=fp32_mask)
                one_vec = vdup_scalar(1.0, dtypes.float32, mask=fp32_mask)
                zero_vec = vdup_scalar(0.0, dtypes.float32, mask=fp32_mask)
                normalization_denominator = vselect(
                    one_vec, normalization_denominator, cond_mask=invalid_row
                )
                even_acc = vselect(zero_vec, even_acc, cond_mask=invalid_row)
                odd_acc = vselect(zero_vec, odd_acc, cond_mask=invalid_row)
                even_acc = vdiv(even_acc, normalization_denominator, mask=fp32_mask)
                odd_acc = vdiv(odd_acc, normalization_denominator, mask=fp32_mask)
                even_bf16 = vcast(
                    even_acc,
                    dtypes.bfloat16,
                    mask=fp32_mask,
                    reg_layout=RegLayout.ZERO,
                )
                odd_bf16 = vcast(
                    odd_acc,
                    dtypes.bfloat16,
                    mask=fp32_mask,
                    reg_layout=RegLayout.ONE,
                )
                vstore(
                    output_slot,
                    row_offset,
                    vor(even_bf16, odd_bf16, mask=bf16_mask),
                    bf16_mask,
                )
        # A zero-row writeback is not a no-op on this DMA path: the copy
        # degenerates and lands 64 rows at the head base of the output,
        # clobbering a sibling task. Tail Q tiles (rows <= subblock*64) must
        # skip both stores entirely. The source is truncated to the same
        # valid row count as the destination: without it a ragged TND tail
        # tile writes its full 64-row UB slot and overwrites the next
        # sequence's rows (a race against that sequence's own writeback).
        if valid_query_rows > 0:
            mem_copy(
                _query_row_tile(
                    gm_output,
                    layout,
                    batch_index,
                    query_head_index,
                    query_token_start
                    + vector_subblock_index * self.m_base_size_per_aiv,
                    valid_query_rows,
                    self.d,
                ),
                output_slot[:valid_query_rows, : self.d],
            )
        if const_expr(gm_lse is not None):
            with vf(mode="simd"):
                one_lane, _ = update_mask(1, elem_bits=32)
                one_vec = vdup_scalar(1.0, dtypes.float32, mask=one_lane)
                inf_vec = vdiv(
                    one_vec,
                    vdup_scalar(0.0, dtypes.float32, mask=one_lane),
                    mask=one_lane,
                )
                # Per-row scalar path: broadcast loads sidestep the
                # 64-lane buffer read that aliases the fresh LSE channel.
                for query_row_index in range(self.m_base_size_per_aiv):
                    row_m_vec = vload_brc(self.row_m, query_row_index)
                    row_l_vec = vload_brc(self.row_l, query_row_index)
                    is_zero = vles(row_l_vec, 0.0, mask=one_lane)
                    safe_l = vselect(one_vec, row_l_vec, cond_mask=is_zero)
                    lse_value = vadd(
                        row_m_vec, vlog(safe_l, mask=one_lane), mask=one_lane
                    )
                    lse_value = vselect(inf_vec, lse_value, cond_mask=is_zero)
                    vstore_first(self.lse_stage, query_row_index, lse_value)
            if valid_query_rows > 0:
                mem_copy(
                    _lse_row_tile(
                        gm_lse,
                        layout,
                        batch_index,
                        query_head_index,
                        query_token_start
                        + vector_subblock_index * self.m_base_size_per_aiv,
                        valid_query_rows,
                    ),
                    self.lse_stage[:valid_query_rows],
                )

    @jit
    def finalize_partial(
        self,
        gm_partial_output,
        gm_partial_lse,
        layout,
        vector_subblock_index,
        batch_index,
        query_head_index,
        partial_row_start,
        valid_query_rows,
    ):
        """FP32 partial ``O_i = acc / l`` and ``LSE_i = m + log l`` for one split.

        Flash Decode keeps every KV split's output normalized by its own
        ``l`` and lets the merge kernel re-weight the splits by their LSE, so
        this path stores straight from ``acc`` (already FP32) instead of
        casting to BF16.
        """
        with vf(mode="simd"):
            fp32_mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
            zero_vec = vdup_scalar(0.0, dtypes.float32, mask=fp32_mask)
            one_vec = vdup_scalar(1.0, dtypes.float32, mask=fp32_mask)
            threshold_vec = vdup_scalar(
                self.invalid_row_threshold, dtypes.float32, mask=fp32_mask
            )
            for query_row_index in range(self.m_base_size_per_aiv):
                row_max = vload_brc(self.row_m, query_row_index)
                invalid_row = vcmp_le(row_max, threshold_vec, mask=fp32_mask)
                denominator = vload_brc(self.row_l, query_row_index)
                denominator = vselect(one_vec, denominator, cond_mask=invalid_row)
                row_offset = query_row_index * self.d
                for d_offset in (0, FP32_VECTOR_LANES):
                    value = vload(self.acc, row_offset + d_offset)
                    value = vselect(zero_vec, value, cond_mask=invalid_row)
                    vstore(
                        self.acc,
                        row_offset + d_offset,
                        vdiv(value, denominator, mask=fp32_mask),
                        fp32_mask,
                    )
        if valid_query_rows > 0:
            mem_copy(
                _query_row_tile(
                    gm_partial_output,
                    layout,
                    batch_index,
                    query_head_index,
                    partial_row_start
                    + vector_subblock_index * self.m_base_size_per_aiv,
                    valid_query_rows,
                    self.d,
                ),
                self.acc[:valid_query_rows, : self.d],
            )
        if const_expr(gm_partial_lse is not None):
            with vf(mode="simd"):
                one_lane, _ = update_mask(1, elem_bits=32)
                one_vec = vdup_scalar(1.0, dtypes.float32, mask=one_lane)
                inf_vec = vdiv(
                    one_vec,
                    vdup_scalar(0.0, dtypes.float32, mask=one_lane),
                    mask=one_lane,
                )
                for query_row_index in range(self.m_base_size_per_aiv):
                    row_m_vec = vload_brc(self.row_m, query_row_index)
                    row_l_vec = vload_brc(self.row_l, query_row_index)
                    is_zero = vles(row_l_vec, 0.0, mask=one_lane)
                    safe_l = vselect(one_vec, row_l_vec, cond_mask=is_zero)
                    lse_value = vadd(
                        row_m_vec, vlog(safe_l, mask=one_lane), mask=one_lane
                    )
                    lse_value = vselect(inf_vec, lse_value, cond_mask=is_zero)
                    vstore_first(self.lse_stage, query_row_index, lse_value)
            if valid_query_rows > 0:
                mem_copy(
                    _lse_row_tile(
                        gm_partial_lse,
                        layout,
                        batch_index,
                        query_head_index,
                        partial_row_start
                        + vector_subblock_index * self.m_base_size_per_aiv,
                        valid_query_rows,
                    ),
                    self.lse_stage[:valid_query_rows],
                )


@kernel
class FlashAttnHif8Kernel:
    """HIF8 dense GQA pipeline on the FP8 full-quant schedule."""

    def __init__(
        self,
        m_base_size=M_BASE_SIZE,
        s2_base_size=S2_BASE_SIZE,
        d=D,
        depths=None,
        causal=0,
        layout="BNSD",
        in_layout="BNSD",
        q_geom=(0, 0),
        kv_geom=(0, 0),
        qk_scale=1.0,
        v_descale=1.0,
        p_scale=1.0,
        fd=0,
    ):
        self.m_base_size, self.s2_base_size, self.d = m_base_size, s2_base_size, d
        self.m_base_size_per_aiv = m_base_size // 2
        self.layout = layout
        self.fd = fd
        self.q_s_pad, self.q_n = q_geom
        # in_layout: public GM layout of Q/K/V (BNSD/BSND/TND-token-major).
        # layout: kernel-internal output layout (BNSD or head-major TND).
        self.in_layout = in_layout
        self.tok_base_mul = 1 if layout == "TND" else 0
        self.causal = causal
        dep = dict(DEPTHS if depths is None else depths)
        self.vector_subblock_index = get_subblock_id()
        self.kernel_block_index = get_block_idx()
        self.vector = Vector(
            m_base_size // 2,
            s2_base_size,
            d,
            dep,
            qk_scale,
            v_descale,
            p_scale,
        )
        self.matmul = Matmul(m_base_size, s2_base_size, d, dep, kv_geom=kv_geom)
        self.p_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.hifloat8,
            make_layout((s2_base_size, m_base_size)),
            depth=dep["p_l1"],
        )
        self.s_ub = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((s2_base_size, m_base_size // 2)),
            depth=dep["s_ub"],
        )
        self.pv_ub = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((m_base_size // 2, d)),
            depth=dep["pv_ub"],
        )

    def __call__(
        self,
        gm_query: Tensor,
        gm_key: Tensor,
        gm_value: Tensor,
        gm_core_task_ranges: Tensor,
        gm_query_sequence_boundaries: Tensor,
        gm_key_value_lengths: Tensor,
        gm_nz_gather_indices: Tensor,
        gm_output: Tensor,
        batch: int,
        n_q_heads: int,
        head_group: int,
        gm_lse: Tensor = None,
        gm_fd_splits: Tensor = None,
        gm_partial_output: Tensor = None,
        gm_partial_lse: Tensor = None,
    ):
        mem_copy(
            self.vector.nz_gather_indices_channel,
            gm_nz_gather_indices,
        )
        with vf(mode="simd"):
            idx_mask, _ = update_mask(len(self.vector.nz_gather_indices), elem_bits=8)
            vstore(
                self.vector.nz_gather_indices_ub,
                0,
                vload(self.vector.nz_gather_indices_channel, 0),
                idx_mask,
            )

        tile_begin = gm_core_task_ranges[0, self.kernel_block_index]
        tile_end = gm_core_task_ranges[1, self.kernel_block_index]
        pipeline_slot_count = SKEW_LAG + 1
        for tile_idx in range(tile_begin, tile_end):
            accumulated_task_count = 0
            batch_idx = 0
            batch_task_start = 0
            previous_q_boundary = gm_query_sequence_boundaries[0, 0]
            for candidate_batch in range(1, batch):
                current_q_boundary = gm_query_sequence_boundaries[0, candidate_batch]
                if const_expr(self.fd):
                    # Flash Decode: one Q tile per (batch, head), split by KV.
                    accumulated_task_count = (
                        accumulated_task_count
                        + n_q_heads * (gm_fd_splits[0, candidate_batch])
                    )
                else:
                    accumulated_task_count = accumulated_task_count + n_q_heads * (
                        ceil_div(
                            current_q_boundary - previous_q_boundary, self.m_base_size
                        )
                    )
                belongs_to_later_batch = tile_idx >= accumulated_task_count
                batch_idx = candidate_batch if belongs_to_later_batch else batch_idx
                batch_task_start = (
                    accumulated_task_count
                    if belongs_to_later_batch
                    else batch_task_start
                )
                previous_q_boundary = current_q_boundary

            task_index_in_batch = tile_idx - batch_task_start
            batch_query_token_start = gm_query_sequence_boundaries[0, batch_idx]
            query_length = (
                gm_query_sequence_boundaries[0, batch_idx + 1] - batch_query_token_start
            )
            split_idx = 0
            split_count = 1
            if const_expr(self.fd):
                split_count = gm_fd_splits[0, batch_idx]
                head_idx = task_index_in_batch // split_count
                split_idx = task_index_in_batch - head_idx * split_count
                query_tiles_per_head = 1
                q_tile_linear = 0
            else:
                query_tiles_per_head = ceil_div(query_length, self.m_base_size)
                head_idx = task_index_in_batch // query_tiles_per_head
                q_tile_linear = task_index_in_batch - head_idx * query_tiles_per_head
            # Causal work per Q tile is triangular: tile 0 reads one KV block,
            # the last tile reads them all. A core's contiguous slice of
            # (head, tile) ids would then be all-light or all-heavy, ~2x load
            # skew. Walk each head's tiles from both ends instead
            # (0, T-1, 1, T-2, ...): a pure arithmetic bijection that mixes
            # light and heavy tiles inside every slice.
            q_tile_half = q_tile_linear // 2
            q_tile_idx = q_tile_linear
            if const_expr(self.causal):
                q_tile_idx = (
                    query_tiles_per_head - 1 - q_tile_half
                    if q_tile_linear - q_tile_half * 2 != 0
                    else q_tile_half
                )

            query_token_start = (
                batch_query_token_start * self.tok_base_mul
                + q_tile_idx * self.m_base_size
            )

            key_value_length = gm_key_value_lengths[0, batch_idx]
            causal_offset = key_value_length - query_length

            # TND packs all batches into one token stream: offset the KV
            # reads of this task by the preceding batches' KV lengths.
            kv_token_base = 0
            if const_expr(self.layout == "TND"):
                for prior_batch in range(batch_idx):
                    kv_token_base = kv_token_base + gm_key_value_lengths[0, prior_batch]

            rows = min(self.m_base_size, query_length - q_tile_idx * self.m_base_size)
            kv_head = head_idx // head_group

            kv_block_capacity = ceil_div(key_value_length, self.s2_base_size)
            if const_expr(self.causal == 0):
                kv_block_count = kv_block_capacity
            else:
                kv_block_count = min(
                    kv_block_capacity,
                    ceil_div(
                        (q_tile_idx + 1) * self.m_base_size + causal_offset,
                        self.s2_base_size,
                    ),
                )
            kv_block_lo = 0
            kv_block_hi = kv_block_count
            if const_expr(self.fd):
                # Balanced partition of this head's KV blocks across splits.
                kv_block_lo = split_idx * kv_block_count // split_count
                kv_block_hi = (split_idx + 1) * kv_block_count // split_count

            q_slot = self.vector_q_load(
                gm_query, batch_idx, head_idx, query_token_start
            )
            self.vector.initialize()
            if const_expr(self.causal == 0):
                mask_begin = kv_block_count - 1
            else:
                # A block needs the causal select as soon as its last key is
                # beyond the first row's visible limit
                # (q_tile * M + causal_offset). Taking kv_block_count - 1
                # instead leaks the keys between that limit and the block end
                # whenever kv_len > q_len, which is the normal ragged-TND
                # decode shape.
                mask_begin = max(
                    0,
                    min(
                        kv_block_count - 1,
                        (q_tile_idx * self.m_base_size + causal_offset + 1)
                        // self.s2_base_size,
                    ),
                )

            for kv_block_idx in range(kv_block_lo, kv_block_hi):
                self.matmul.load_k(
                    gm_key,
                    self.layout,
                    self.in_layout,
                    batch_idx,
                    kv_head,
                    kv_block_idx,
                    kv_token_base,
                )
                self.matmul.compute_qk(self.s_ub, q_slot)
                if kv_block_idx >= mask_begin:
                    if const_expr(self.causal == 0):
                        self._stage_softmax_p(
                            kv_block_idx,
                            q_tile_idx,
                            causal_offset,
                            key_value_length,
                            kv_block_idx % pipeline_slot_count,
                            MASK_KV_TAIL,
                        )
                    else:
                        self._stage_softmax_p(
                            kv_block_idx,
                            q_tile_idx,
                            causal_offset,
                            key_value_length,
                            kv_block_idx % pipeline_slot_count,
                            MASK_CAUSAL,
                        )
                else:
                    self._stage_softmax_p(
                        kv_block_idx,
                        q_tile_idx,
                        causal_offset,
                        key_value_length,
                        kv_block_idx % pipeline_slot_count,
                        MASK_NONE,
                    )

                if kv_block_idx - kv_block_lo >= SKEW_LAG:
                    consume_block_idx = kv_block_idx - SKEW_LAG
                    self._consume(
                        gm_value,
                        batch_idx,
                        consume_block_idx,
                        kv_head,
                        consume_block_idx % pipeline_slot_count,
                        kv_token_base,
                    )

            for drain in tuple(range(SKEW_LAG)):
                consume_block_idx = kv_block_hi + drain - SKEW_LAG
                if consume_block_idx >= kv_block_lo:
                    self._consume(
                        gm_value,
                        batch_idx,
                        consume_block_idx,
                        kv_head,
                        consume_block_idx % pipeline_slot_count,
                        kv_token_base,
                    )

            valid_query_rows = max(
                0,
                min(
                    rows - self.vector_subblock_index * self.m_base_size_per_aiv,
                    self.m_base_size_per_aiv,
                ),
            )
            if const_expr(self.fd):
                partial_row_start = split_idx * self.m_base_size
                if const_expr(self.layout == "TND"):
                    # Partials live in one (N, splits, batch*M) token stream.
                    partial_row_start = (
                        split_idx * batch * self.m_base_size
                        + batch_idx * self.m_base_size
                    )
                self.vector.finalize_partial(
                    gm_partial_output,
                    gm_partial_lse,
                    self.layout,
                    self.vector_subblock_index,
                    batch_idx,
                    head_idx,
                    partial_row_start,
                    valid_query_rows,
                )
            else:
                self.vector.finalize_output(
                    gm_output,
                    gm_lse,
                    self.layout,
                    self.vector_subblock_index,
                    batch_idx,
                    head_idx,
                    query_token_start,
                    valid_query_rows,
                )

    @jit
    def vector_q_load(self, gm_query, batch_idx, head_idx, query_token_start):
        # Full 128-row Q tile per cube task; both AIVs share it on L0B.
        # Tail tiles clamp the GM window to the physical rows; the padded
        # lanes never reach the output (finalize writes valid rows only).
        if const_expr(self.in_layout != "BSND"):
            q_src = _query_row_tile(
                gm_query,
                self.layout,
                batch_idx,
                head_idx,
                query_token_start,
                self.m_base_size,
                self.d,
            )
        else:
            q_src = _q_rows(
                gm_query,
                self.in_layout,
                batch_idx,
                head_idx,
                query_token_start,
                self.m_base_size,
                self.q_s_pad,
                self.q_n,
            )
        return self.matmul.load_q(q_src)

    @jit
    def _stage_softmax_p(
        self,
        kv_block_idx,
        q_tile_idx,
        causal_offset,
        s2_length,
        pipeline_slot,
        kv_block_mask_type,
    ):
        if const_expr(kv_block_mask_type == MASK_KV_TAIL):
            visible_limit = s2_length - kv_block_idx * self.s2_base_size - 1
        else:
            visible_limit = (
                q_tile_idx * self.m_base_size
                + causal_offset
                - kv_block_idx * self.s2_base_size
                + self.vector_subblock_index * self.m_base_size_per_aiv
            )
        self.vector.softmax(
            self.s_ub.next_read(),
            visible_limit,
            pipeline_slot,
            kv_block_mask_type,
        )
        self.vector.store_p(self.p_l1, self.vector_subblock_index)

    @jit
    def _consume(
        self, gm_value, batch_idx, kv_block_idx, kv_head, pipeline_slot, kv_token_base
    ):
        self.matmul.load_v(
            gm_value,
            self.layout,
            self.in_layout,
            batch_idx,
            kv_head,
            kv_block_idx,
            kv_token_base,
        )
        self.matmul.compute_pv(self.p_l1, self.pv_ub)
        self.vector.update_output_acc(self.pv_ub.next_read(), pipeline_slot)


@kernel
class FlashAttnHif8CombineKernel:
    """Merge Flash-Decode partial O / LSE back into one row block.

    Partials are ``O_i`` normalized by each split's own ``l`` and
    ``LSE_i = m_i + log l_i``. The merged row is
    ``O = sum_i exp(LSE_i - LSE) * O_i`` with
    ``LSE = logsumexp_i(LSE_i)``, i.e. the usual weighted merge.
    """

    def __init__(self, m_base_size=M_BASE_SIZE, d=D, layout="BNSD", core_count=32):
        self.m_base_size, self.d, self.layout = m_base_size, d, layout
        self.core_count = core_count
        self.core_index = get_block_idx()
        self.lse_split = Buffer(MemLoc.UB, dtypes.float32, make_layout((m_base_size,)))
        self.weights = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((FD_MAX_SPLITS * m_base_size,))
        )
        self.row_max = Buffer(MemLoc.UB, dtypes.float32, make_layout((m_base_size,)))
        self.row_denom = Buffer(MemLoc.UB, dtypes.float32, make_layout((m_base_size,)))
        self.row_inv = Buffer(MemLoc.UB, dtypes.float32, make_layout((m_base_size,)))
        self.acc = Buffer(MemLoc.UB, dtypes.float32, make_layout((m_base_size, d)))
        self.partial = Buffer(MemLoc.UB, dtypes.float32, make_layout((m_base_size, d)))
        self.out_bf16 = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((m_base_size, d))
        )

    @jit
    def _partial_lse_tile(self, gm_partial_lse, batch_index, head_index, token_start):
        return _lse_row_tile(
            gm_partial_lse,
            self.layout,
            batch_index,
            head_index,
            token_start,
            self.m_base_size,
        )

    @jit
    def _partial_o_tile(self, gm_partial_output, batch_index, head_index, token_start):
        return _query_row_tile(
            gm_partial_output,
            self.layout,
            batch_index,
            head_index,
            token_start,
            self.m_base_size,
            self.d,
        )

    def __call__(
        self,
        gm_partial_output: Tensor,
        gm_partial_lse: Tensor,
        gm_splits: Tensor,
        gm_query_sequence_boundaries: Tensor,
        gm_output: Tensor,
        batch: int,
        n_q_heads: int,
        gm_lse: Tensor = None,
    ):
        pair_count = batch * n_q_heads
        pair_begin = pair_count * self.core_index // self.core_count
        pair_end = pair_count * (self.core_index + 1) // self.core_count
        for pair in range(pair_begin, pair_end):
            batch_index = pair // n_q_heads
            head_index = pair - batch_index * n_q_heads
            batch_query_start = gm_query_sequence_boundaries[0, batch_index]
            query_length = (
                gm_query_sequence_boundaries[0, batch_index + 1] - batch_query_start
            )
            rows = min(self.m_base_size, query_length)
            splits = gm_splits[0, batch_index]
            # Padded partial workspace: every (batch, split) owns one
            # m_base_size row block; dense keeps (B, N, splits*M), TND packs
            # (N, splits, batch*M) into one token stream.
            partial_stride = self.m_base_size
            partial_batch_offset = 0
            output_token_start = 0
            if const_expr(self.layout == "TND"):
                partial_stride = batch * self.m_base_size
                partial_batch_offset = batch_index * self.m_base_size
                output_token_start = batch_query_start

            # Pass A: per-row max over splits.
            with vf(mode="simd"):
                fp32_mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                for offset in range(0, self.m_base_size, FP32_VECTOR_LANES):
                    vstore(
                        self.row_max,
                        offset,
                        vdup_scalar(NEG_MAX, dtypes.float32, mask=fp32_mask),
                        fp32_mask,
                    )
            for split_index in range(splits):
                token_start = split_index * partial_stride + partial_batch_offset
                mem_copy(
                    self.lse_split,
                    self._partial_lse_tile(
                        gm_partial_lse, batch_index, head_index, token_start
                    ),
                )
                with vf(mode="simd"):
                    fp32_mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                    for offset in range(0, self.m_base_size, FP32_VECTOR_LANES):
                        vstore(
                            self.row_max,
                            offset,
                            vmax(
                                vload(self.row_max, offset),
                                vload(self.lse_split, offset),
                                mask=fp32_mask,
                            ),
                            fp32_mask,
                        )
            # Pass B: weights and their row sums.
            with vf(mode="simd"):
                fp32_mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                for offset in range(0, self.m_base_size, FP32_VECTOR_LANES):
                    vstore(
                        self.row_denom,
                        offset,
                        vdup_scalar(0.0, dtypes.float32, mask=fp32_mask),
                        fp32_mask,
                    )
            for split_index in range(splits):
                token_start = split_index * partial_stride + partial_batch_offset
                mem_copy(
                    self.lse_split,
                    self._partial_lse_tile(
                        gm_partial_lse, batch_index, head_index, token_start
                    ),
                )
                with vf(mode="simd"):
                    fp32_mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                    for offset in range(0, self.m_base_size, FP32_VECTOR_LANES):
                        weight = vexp_sub(
                            vload(self.lse_split, offset),
                            vload(self.row_max, offset),
                            mask=fp32_mask,
                        )
                        vstore(
                            self.weights,
                            split_index * self.m_base_size + offset,
                            weight,
                            fp32_mask,
                        )
                        vstore(
                            self.row_denom,
                            offset,
                            vadd(vload(self.row_denom, offset), weight, mask=fp32_mask),
                            fp32_mask,
                        )
            # Pass C: weighted accumulation of the split outputs.
            with vf(mode="simd"):
                fp32_mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                for row in range(self.m_base_size):
                    for d_offset in (0, FP32_VECTOR_LANES):
                        vstore(
                            self.acc,
                            row * self.d + d_offset,
                            vdup_scalar(0.0, dtypes.float32, mask=fp32_mask),
                            fp32_mask,
                        )
            for split_index in range(splits):
                token_start = split_index * partial_stride + partial_batch_offset
                mem_copy(
                    self.partial,
                    self._partial_o_tile(
                        gm_partial_output, batch_index, head_index, token_start
                    ),
                )
                with vf(mode="simd"):
                    fp32_mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                    for row in range(self.m_base_size):
                        weight = vload_brc(
                            self.weights, split_index * self.m_base_size + row
                        )
                        for d_offset in (0, FP32_VECTOR_LANES):
                            vstore(
                                self.acc,
                                row * self.d + d_offset,
                                vmadd(
                                    vload(self.partial, row * self.d + d_offset),
                                    weight,
                                    vload(self.acc, row * self.d + d_offset),
                                    mask=fp32_mask,
                                ),
                                fp32_mask,
                            )
            # Pass D: normalize, cast and export.
            with vf(mode="simd"):
                fp32_mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                bf16_mask, _ = update_mask(2 * FP32_VECTOR_LANES, elem_bits=16)
                for row in range(self.m_base_size):
                    denominator = vload_brc(self.row_denom, row)
                    even_acc, odd_acc = vload_deinterleave(
                        self.acc, row * self.d, width="b32"
                    )
                    even_acc = vdiv(even_acc, denominator, mask=fp32_mask)
                    odd_acc = vdiv(odd_acc, denominator, mask=fp32_mask)
                    even_bf16 = vcast(
                        even_acc,
                        dtypes.bfloat16,
                        mask=fp32_mask,
                        reg_layout=RegLayout.ZERO,
                    )
                    odd_bf16 = vcast(
                        odd_acc,
                        dtypes.bfloat16,
                        mask=fp32_mask,
                        reg_layout=RegLayout.ONE,
                    )
                    vstore(
                        self.out_bf16,
                        row * self.d,
                        vor(even_bf16, odd_bf16, mask=bf16_mask),
                        bf16_mask,
                    )
                for offset in range(0, self.m_base_size, FP32_VECTOR_LANES):
                    vstore(
                        self.row_inv,
                        offset,
                        vadd(
                            vload(self.row_max, offset),
                            vlog(
                                vload(self.row_denom, offset),
                                mask=fp32_mask,
                            ),
                            mask=fp32_mask,
                        ),
                        fp32_mask,
                    )
            if rows > 0:
                mem_copy(
                    _query_row_tile(
                        gm_output,
                        self.layout,
                        batch_index,
                        head_index,
                        output_token_start,
                        rows,
                        self.d,
                    ),
                    self.out_bf16[:rows, : self.d],
                )
                if const_expr(gm_lse is not None):
                    mem_copy(
                        _lse_row_tile(
                            gm_lse,
                            self.layout,
                            batch_index,
                            head_index,
                            output_token_start,
                            rows,
                        ),
                        self.row_inv[:rows],
                    )


class FlashAttnHif8Launcher:
    def __init__(
        self,
        causal=False,
        layout="BNSD",
        in_layout="BNSD",
        q_geom=(0, 0),
        kv_geom=(0, 0),
        qk_scale=1.0,
        v_descale=1.0,
        p_scale=1.0,
        fd=0,
    ):
        self.causal = int(causal)
        self.layout = layout
        self.in_layout = in_layout
        self.q_geom = q_geom
        self.kv_geom = kv_geom
        self.qk_scale = qk_scale
        self.v_descale = v_descale
        self.p_scale = p_scale
        self.fd = int(fd)

    @host
    def launch(
        self,
        gm_output,
        gm_query,
        gm_key,
        gm_value,
        gm_core_task_ranges,
        gm_query_sequence_boundaries,
        gm_key_value_lengths,
        gm_nz_gather_indices,
        batch: int,
        n_q_heads: int,
        head_group: int,
        block_dim: int,
        gm_lse: Tensor = None,
        gm_fd_splits: Tensor = None,
        gm_partial_output: Tensor = None,
        gm_partial_lse: Tensor = None,
    ):
        op = FlashAttnHif8Kernel(
            depths=None,
            causal=self.causal,
            layout=self.layout,
            in_layout=self.in_layout,
            q_geom=self.q_geom,
            kv_geom=self.kv_geom,
            qk_scale=self.qk_scale,
            v_descale=self.v_descale,
            p_scale=self.p_scale,
            fd=self.fd,
        )
        call_args = [
            gm_query,
            gm_key,
            gm_value,
            gm_core_task_ranges,
            gm_query_sequence_boundaries,
            gm_key_value_lengths,
            gm_nz_gather_indices,
            gm_output,
            batch,
            n_q_heads,
            head_group,
        ]
        if self.fd:
            call_args += [
                gm_lse,
                gm_fd_splits,
                gm_partial_output,
                gm_partial_lse,
            ]
        elif gm_lse is not None:
            call_args.append(gm_lse)
        op[block_dim](*call_args)


def _build_task_ranges(q_lengths, n_q_heads, m_base_size, cores):
    boundaries = [0]
    for length in q_lengths:
        boundaries.append(boundaries[-1] + int(length))

    def _cd(a, b):
        return (int(a) + int(b) - 1) // int(b)

    total = sum(n_q_heads * _cd(length, m_base_size) for length in q_lengths)
    ranges = [[0] * cores, [0] * cores]
    per_core = total // cores
    extra = total % cores
    start = 0
    for core in range(cores):
        count = per_core + (1 if core < extra else 0)
        ranges[0][core] = start
        ranges[1][core] = start + count
        start += count
    return ranges, boundaries


class FlashAttnHif8CombineLauncher:
    """Launch the Flash-Decode merge kernel."""

    def __init__(self, layout="BNSD", with_lse=True, core_count=32):
        self.layout = layout
        self.with_lse = with_lse
        self.core_count = core_count

    @host
    def launch(
        self,
        gm_partial_output,
        gm_partial_lse,
        gm_fd_splits,
        gm_query_sequence_boundaries,
        gm_output,
        batch: int,
        n_q_heads: int,
        gm_lse: Tensor = None,
    ):
        op = FlashAttnHif8CombineKernel(layout=self.layout, core_count=self.core_count)
        call_args = [
            gm_partial_output,
            gm_partial_lse,
            gm_fd_splits,
            gm_query_sequence_boundaries,
            gm_output,
            batch,
            n_q_heads,
        ]
        if self.with_lse:
            call_args.append(gm_lse)
        op[self.core_count](*call_args)


def _dynamic_combine_specs(in_layout, with_lse):
    """AOT contract for the merge kernel (same dynamic-dim style as the main)."""
    batch = cannbotdsl.Dim("batch", min=1)
    query_heads = cannbotdsl.Dim("query_heads", min=1)
    output_tokens = cannbotdsl.Dim("output_tokens", min=1)
    partial_tokens = cannbotdsl.Dim("partial_tokens", min=1)
    boundary_width = cannbotdsl.Dim("query_boundary_width", min=2)
    partial_batch_stride = cannbotdsl.Dim("partial_batch_stride", min=D)
    partial_head_stride = cannbotdsl.Dim("partial_head_stride", min=D)
    partial_lse_batch_stride = cannbotdsl.Dim("partial_lse_batch_stride", min=1)
    partial_lse_head_stride = cannbotdsl.Dim("partial_lse_head_stride", min=1)
    out_batch_stride = cannbotdsl.Dim("out_batch_stride", min=D)
    out_head_stride = cannbotdsl.Dim("out_head_stride", min=D)
    lse_batch_stride = cannbotdsl.Dim("lse_batch_stride", min=1)
    lse_head_stride = cannbotdsl.Dim("lse_head_stride", min=1)
    tensor_spec = cannbotdsl.TensorSpec
    if in_layout == "TND":
        partial_output_spec = tensor_spec(
            (query_heads, partial_tokens, D),
            dtypes.float32,
            stride=(partial_head_stride, D, 1),
        )
        partial_lse_spec = tensor_spec(
            (query_heads, partial_tokens),
            dtypes.float32,
            stride=(partial_lse_head_stride, 1),
        )
        output_spec = tensor_spec(
            (query_heads, output_tokens, D),
            dtypes.bfloat16,
            stride=(out_head_stride, D, 1),
        )
        lse_spec = tensor_spec(
            (query_heads, output_tokens),
            dtypes.float32,
            stride=(lse_head_stride, 1),
        )
    else:
        partial_output_spec = tensor_spec(
            (batch, query_heads, partial_tokens, D),
            dtypes.float32,
            stride=(partial_batch_stride, partial_head_stride, D, 1),
        )
        partial_lse_spec = tensor_spec(
            (batch, query_heads, partial_tokens),
            dtypes.float32,
            stride=(partial_lse_batch_stride, partial_lse_head_stride, 1),
        )
        output_spec = tensor_spec(
            (batch, query_heads, output_tokens, D),
            dtypes.bfloat16,
            stride=(out_batch_stride, out_head_stride, D, 1),
        )
        lse_spec = tensor_spec(
            (batch, query_heads, output_tokens),
            dtypes.float32,
            stride=(lse_batch_stride, lse_head_stride, 1),
        )
    specs = [
        partial_output_spec,
        partial_lse_spec,
        tensor_spec((1, batch), dtypes.int32, stride=(batch, 1)),
        tensor_spec((1, boundary_width), dtypes.int32, stride=(boundary_width, 1)),
        output_spec,
    ]
    specs += [
        dtypes.int64,  # batch
        dtypes.int64,  # n_q_heads
    ]
    if with_lse:
        specs.append(lse_spec)
    return tuple(specs)


_COMPILED_COMBINE = {}


def _get_compiled_combine(layout, with_lse, cores):
    key = (layout, with_lse, cores)
    with _COMPILED_KERNELS_LOCK:
        compiled = _COMPILED_COMBINE.get(key)
        if compiled is not None:
            return compiled
        launcher = FlashAttnHif8CombineLauncher(
            layout=layout, with_lse=with_lse, core_count=cores
        )
        compiled = cannbotdsl.compile(
            launcher.launch, *_dynamic_combine_specs(layout, with_lse)
        )
        _COMPILED_COMBINE[key] = compiled
        return compiled


def _split_task_ranges(total, cores):
    """Even split of ``total`` unit tasks over ``cores`` kernel blocks."""
    ranges = [[0] * cores, [0] * cores]
    per_core = total // cores
    extra = total % cores
    start = 0
    for core in range(cores):
        count = per_core + (1 if core < extra else 0)
        ranges[0][core] = start
        ranges[1][core] = start + count
        start += count
    return ranges


def _plan_fd_splits(batch, q_lengths, n_q_heads, kv_lengths, cores):
    """KV split count for Flash Decode, or 0 to keep the plain path.

    Only the decode regime (one 128-row Q tile per sequence) splits: the
    merge kernel writes one padded row block per (batch, split). FD pays off
    when the base grid (batch * heads) leaves cores idle and every sequence
    owns at least two KV tiles.
    """
    if max(q_lengths) > M_BASE_SIZE:
        return 0
    base_tasks = batch * n_q_heads
    if base_tasks >= cores:
        return 0
    min_blocks = min(
        max(1, (int(length) + S2_BASE_SIZE - 1) // S2_BASE_SIZE)
        for length in kv_lengths
    )
    splits = min(cores // base_tasks, min_blocks, FD_MAX_SPLITS)
    return splits if splits >= 2 else 0


_COMPILED_KERNELS = {}
_COMPILED_KERNELS_LOCK = threading.Lock()
# Scales fold into the artifact, so a caller that sweeps them would otherwise
# pin one compiled kernel per value; this cap bounds the cache instead.
_COMPILED_KERNELS_LIMIT = 64

# Read-only launch tensors (task ranges / boundaries / KV lengths / gather
# indices) depend only on the shape, so a steady-state caller should not pay
# the blocking H2D copy again per call. Keyed by device, capped to keep a
# pathological shape sweep from growing without bound.
_AUX_TENSORS = {}
_AUX_TENSORS_LOCK = threading.Lock()
_AUX_TENSORS_LIMIT = 512


def _aux_tensor(key, factory):
    with _AUX_TENSORS_LOCK:
        cached = _AUX_TENSORS.get(key)
        if cached is None:
            if len(_AUX_TENSORS) >= _AUX_TENSORS_LIMIT:
                _AUX_TENSORS.clear()
            cached = factory()
            _AUX_TENSORS[key] = cached
        return cached


# Host-side reads of device scalars / seqused lists (descale, p_scale, per-batch
# lengths) are blocking D2H copies; at 0.1-0.2 ms each they dominate the launch
# path of a small attention call. Memoise them per tensor identity and
# invalidate on in-place mutation via the version counter: steady-state callers
# that reuse the same tensors pay nothing, and a caller that rewrites seqused
# every step still sees the fresh values.
_HOST_VALUE_CACHE = {}
_HOST_VALUE_CACHE_LOCK = threading.Lock()
_HOST_VALUE_CACHE_LIMIT = 512


def _host_value(t, reader):
    key = id(t)
    version = t._version
    with _HOST_VALUE_CACHE_LOCK:
        entry = _HOST_VALUE_CACHE.get(key)
        if entry is not None:
            ref, cached_version, value = entry
            if ref() is t and cached_version == version:
                return value
        value = reader(t)
        if len(_HOST_VALUE_CACHE) >= _HOST_VALUE_CACHE_LIMIT:
            _HOST_VALUE_CACHE.clear()
        _HOST_VALUE_CACHE[key] = (weakref.ref(t), version, value)
        return value


def _host_scalar(t):
    return float(t.detach().cpu().reshape(()))


def _host_ints(t):
    return tuple(int(v) for v in t.detach().cpu().reshape(-1).tolist())


_PLATFORM_CORES = None


def _platform_cube_cores():
    """Probe the AIC count once; the platform cannot change inside a process."""
    global _PLATFORM_CORES
    if _PLATFORM_CORES is None:
        from cannbotdsl import get_platform_info

        try:
            _PLATFORM_CORES = int(get_platform_info().cube_core_num) or 32
        except (OSError, RuntimeError, TypeError, ValueError):
            _PLATFORM_CORES = 32
    return _PLATFORM_CORES


def _dynamic_tensor_specs(in_layout, fd=False, with_lse=True):
    """One dynamic AOT call contract reused across shapes per input layout."""
    batch = cannbotdsl.Dim("batch", min=1)
    query_heads = cannbotdsl.Dim("query_heads", min=1)
    key_value_heads = cannbotdsl.Dim("key_value_heads", min=1)
    cube_core_count = cannbotdsl.Dim("cube_core_count", min=1, max=64)
    boundary_width = cannbotdsl.Dim("query_boundary_width", min=2)
    token_rows = cannbotdsl.Dim("token_rows", min=1)
    key_value_rows = cannbotdsl.Dim("key_value_rows", min=1)
    out_batch_stride = cannbotdsl.Dim("out_batch_stride", min=D)
    out_head_stride = cannbotdsl.Dim("out_head_stride", min=D)
    lse_batch_stride = cannbotdsl.Dim("lse_batch_stride", min=1)
    lse_head_stride = cannbotdsl.Dim("lse_head_stride", min=1)
    partial_tokens = cannbotdsl.Dim("partial_tokens", min=1)
    partial_batch_stride = cannbotdsl.Dim("partial_batch_stride", min=D)
    partial_head_stride = cannbotdsl.Dim("partial_head_stride", min=D)
    partial_lse_batch_stride = cannbotdsl.Dim("partial_lse_batch_stride", min=1)
    partial_lse_head_stride = cannbotdsl.Dim("partial_lse_head_stride", min=1)
    tensor_spec = cannbotdsl.TensorSpec

    if in_layout == "TND":
        output_tokens = cannbotdsl.Dim("output_tokens", min=1)
        query_storage_tokens = cannbotdsl.Dim("query_storage_tokens", min=1)
        key_value_storage_tokens = cannbotdsl.Dim("kv_storage_tokens", min=1)
        q_head_stride = cannbotdsl.Dim("q_packed_head_stride", min=D)
        q_head_plane = cannbotdsl.Dim("q_head_plane_stride", min=D)
        kv_head_stride = cannbotdsl.Dim("kv_packed_head_stride", min=D)
        kv_head_plane = cannbotdsl.Dim("kv_head_plane_stride", min=D)
        output_spec = tensor_spec(
            (query_heads, output_tokens, D),
            dtypes.bfloat16,
            stride=(out_head_stride, D, 1),
        )
        query_spec = tensor_spec(
            (query_heads, query_storage_tokens, D),
            dtypes.int8,
            # Head-plane stride is dynamic: a token-major view is D, the
            # dense Q repack used for many Q heads is rows * D.
            stride=(q_head_plane, q_head_stride, 1),
        )
        key_spec = tensor_spec(
            (key_value_heads, key_value_storage_tokens, D),
            dtypes.int8,
            # Head-plane stride is dynamic: a token-major view is D, while the
            # dense K/V repack used for many KV heads is rows * D.
            stride=(kv_head_plane, kv_head_stride, 1),
        )
        value_spec = key_spec
        lse_spec = tensor_spec(
            (query_heads, output_tokens),
            dtypes.float32,
            stride=(lse_head_stride, 1),
        )
    else:
        query_tokens = cannbotdsl.Dim("query_tokens", min=1)
        output_tokens = cannbotdsl.Dim("output_tokens", min=1)
        key_value_tokens = cannbotdsl.Dim("key_value_tokens", min=1)
        batch_query_stride = cannbotdsl.Dim("batch_query_stride", min=D)
        head_query_stride = cannbotdsl.Dim("head_query_stride", min=D)
        batch_kv_stride = cannbotdsl.Dim("batch_kv_stride", min=D)
        head_kv_stride = cannbotdsl.Dim("head_kv_stride", min=D)
        if in_layout == "BSND":
            query_spec = tensor_spec(
                (token_rows, D),
                dtypes.int8,
                stride=(D, 1),
            )
            key_spec = tensor_spec(
                (key_value_rows, D),
                dtypes.int8,
                stride=(D, 1),
            )
            value_spec = key_spec
        else:
            query_spec = tensor_spec(
                (batch, query_heads, query_tokens, D),
                dtypes.int8,
                stride=(batch_query_stride, head_query_stride, D, 1),
            )
            key_spec = tensor_spec(
                (batch, key_value_heads, key_value_tokens, D),
                dtypes.int8,
                stride=(batch_kv_stride, head_kv_stride, D, 1),
            )
            value_spec = key_spec
        # Q is padded to the tile multiple while the output keeps the live
        # query length, so the two token extents are independent symbols.
        output_spec = tensor_spec(
            (batch, query_heads, output_tokens, D),
            dtypes.bfloat16,
            stride=(out_batch_stride, out_head_stride, D, 1),
        )
        lse_spec = tensor_spec(
            (batch, query_heads, output_tokens),
            dtypes.float32,
            stride=(lse_batch_stride, lse_head_stride, 1),
        )

    core_task_ranges_spec = tensor_spec(
        (2, cube_core_count), dtypes.int32, stride=(cube_core_count, 1)
    )
    boundaries_spec = tensor_spec(
        (1, boundary_width), dtypes.int32, stride=(boundary_width, 1)
    )
    kv_lengths_spec = tensor_spec((1, batch), dtypes.int32, stride=(batch, 1))
    gather_indices_spec = tensor_spec(
        (1, len(nz_gather_index(M_BASE_SIZE // 2))), dtypes.uint8
    )
    fd_splits_spec = tensor_spec((1, batch), dtypes.int32, stride=(batch, 1))
    if in_layout == "TND":
        partial_output_spec = tensor_spec(
            (query_heads, partial_tokens, D),
            dtypes.float32,
            stride=(partial_head_stride, D, 1),
        )
        partial_lse_spec = tensor_spec(
            (query_heads, partial_tokens),
            dtypes.float32,
            stride=(partial_lse_head_stride, 1),
        )
    else:
        partial_output_spec = tensor_spec(
            (batch, query_heads, partial_tokens, D),
            dtypes.float32,
            stride=(partial_batch_stride, partial_head_stride, D, 1),
        )
        partial_lse_spec = tensor_spec(
            (batch, query_heads, partial_tokens),
            dtypes.float32,
            stride=(partial_lse_batch_stride, partial_lse_head_stride, 1),
        )
    specs = [
        output_spec,
        query_spec,
        key_spec,
        value_spec,
        core_task_ranges_spec,
        boundaries_spec,
        kv_lengths_spec,
        gather_indices_spec,
        dtypes.int64,  # batch
        dtypes.int64,  # n_q_heads
        dtypes.int64,  # head_group
        dtypes.int64,  # block_dim
    ]
    if with_lse or fd:
        specs.append(lse_spec)
    if fd:
        # Flash Decode always computes the split LSE: the merge needs it.
        specs += [fd_splits_spec, partial_output_spec, partial_lse_spec]
    return tuple(specs)


def _get_compiled_kernel(
    causal,
    layout,
    in_layout,
    q_geom,
    kv_geom,
    qk_scale,
    v_descale,
    p_scale,
    with_lse,
    fd=False,
):
    # Scales are compile-time constants; the cap below keeps a caller that
    # sweeps scale values from pinning every artifact.
    key = (
        causal,
        layout,
        in_layout,
        qk_scale,
        v_descale,
        p_scale,
        q_geom,
        kv_geom,
        with_lse,
        fd,
    )
    with _COMPILED_KERNELS_LOCK:
        compiled = _COMPILED_KERNELS.get(key)
        if compiled is not None:
            return compiled
        if len(_COMPILED_KERNELS) >= _COMPILED_KERNELS_LIMIT:
            _COMPILED_KERNELS.clear()
        launcher = FlashAttnHif8Launcher(
            causal=causal,
            layout=layout,
            in_layout=in_layout,
            q_geom=q_geom,
            kv_geom=kv_geom,
            qk_scale=qk_scale,
            v_descale=v_descale,
            p_scale=p_scale,
            fd=fd,
        )
        specs = _dynamic_tensor_specs(in_layout, fd=fd, with_lse=with_lse)
        compiled = cannbotdsl.compile(launcher.launch, *specs)
        _COMPILED_KERNELS[key] = compiled
        return compiled


def npu_flash_atten_hif8_fullquant(
    query,
    key,
    value,
    descale_q,
    descale_k,
    descale_v,
    *,
    p_scale=None,
    softmax_scale=None,
    mask_mode=0,
    layout="BNSD",
    seqused_kv=None,
    seqused_q=None,
    return_softmax_lse=False,
):
    """Native HIF8 FlashAttention on the FP8 full-quant pipeline.

    Q/K/V are int8 storage of HIF8 codes in matching BNSD/BSND/TND layouts;
    ``layout`` selects the shared layout (normalized to BNSD strides here).
    """
    info = validate_static(query, key, value, mask_mode, layout, seqused_kv, seqused_q)

    def _scalar(t):
        if t is None:
            return 1.0
        return _host_value(t, _host_scalar)

    descale_q_value = _scalar(descale_q)
    descale_k_value = _scalar(descale_k)
    descale_v_value = _scalar(descale_v)
    p_scale_value = _scalar(p_scale) if p_scale is not None else 1.0
    validate_scales(
        descale_q_value,
        descale_k_value,
        descale_v_value,
        p_scale_value,
        softmax_scale,
    )
    qk_scale = (
        (softmax_scale if softmax_scale is not None else 1.0)
        * descale_q_value
        * descale_k_value
    )
    v_descale = descale_v_value
    # The masked-row sentinel and the p_scale log are trace-time constants
    # inside Vector, so a caller that holds its scales fixed pays no per-call
    # scalar traffic (see the cache cap in _get_compiled_kernel).

    public_layout = layout
    in_layout = layout  # GM Q/K/V stay in the public layout (native reads)
    batch = info["batch"]
    n_q_heads = info["n_q_heads"]
    kv_heads = info["kv_heads"]
    q_extent, kv_extent = info["q_extent"], info["kv_extent"]
    q_len = q_extent
    if layout == "BSND":
        layout = "BNSD"  # kernel-internal output layout
        q_view, k_view, v_view = query, key, value
    elif layout == "BNSD":
        q_view, k_view, v_view = query, key, value
    else:  # TND: (T, N, D) with per-batch boundaries via seqused_kv lengths
        kv_lens = list(_host_value(seqused_kv, _host_ints))
        if seqused_q is not None:
            q_lens = list(_host_value(seqused_q, _host_ints))
        elif query.shape[0] == sum(kv_lens):
            q_lens = kv_lens
        else:
            raise ValueError(
                "TND requires seqused_q (or seqused_kv covering the packed "
                "query tokens) to derive per-batch query lengths"
            )
        # Public TND is token-major (T, N, D); Q/K/V are read natively via
        # multi-source strided DMA. The kernel output stays head-major and
        # is permuted back once on return.
        q_view, k_view, v_view = query, key, value

    q_lengths = [q_len] * batch if layout != "TND" else q_lens
    _kv_tok_dim = {"BNSD": 2, "BSND": 1, "TND": 0}[public_layout]
    kv_lengths = (
        kv_lens
        if layout == "TND"
        else (
            list(_host_value(seqused_kv, _host_ints))
            if seqused_kv is not None
            else [key.shape[_kv_tok_dim]] * batch
        )
    )
    validate_lengths(q_lengths, kv_lengths, q_extent, kv_extent, layout == "TND")
    head_group = n_q_heads // kv_heads

    # The platform's real AIC count: hardcoding 36 on a 32-AIC part
    # over-subscribes the grid and serializes a second wave of blocks.
    cores = _platform_cube_cores()
    # Decode regime only: one Q tile per sequence, split KV when the base
    # grid (batch * heads) cannot fill the cores.
    fd_splits = _plan_fd_splits(batch, q_lengths, n_q_heads, kv_lengths, cores)

    # Pad Q/K/V to static tile multiples: tail reads beyond the logical
    # length load padding that the KV-tail mask / row-limited writeback
    # neutralize, but statically short extents fail layout validation.
    # Token dim per public layout: BNSD=2, BSND=1, TND=0.
    _tok_dim = _kv_tok_dim

    def _pad_tokens(t, multiple, extra=0):
        k = _tok_dim
        pad = (-t.shape[k]) % multiple + extra
        if pad == 0:
            return t
        spec = [0, 0] * (t.dim() - k - 1) + [0, pad] + [0, 0] * k
        return torch.nn.functional.pad(t, tuple(spec))

    def _pad_rows(t, rows):
        if rows == 0:
            return t
        k = _tok_dim
        spec = [0, 0] * (t.dim() - k - 1) + [0, rows] + [0, 0] * k
        return torch.nn.functional.pad(t, tuple(spec))

    if public_layout == "TND":
        # TND packs every sequence into one stream. A sequence's last tile
        # starts at a ragged offset and still fetches a full tile, but only
        # the LAST sequence can run past the storage end (an earlier
        # sequence's overrun lands on the next sequence's rows, which the
        # tail mask already neutralises). The required extent is therefore
        # (-last_length) % tile, not "one extra tile": an aligned tail then
        # costs no copy at all, which matters at 64K where Q is 167 MB.
        q_view = _pad_rows(q_view, (-q_lens[-1]) % M_BASE_SIZE)
        kv_over = (-kv_lens[-1]) % S2_BASE_SIZE
        k_view = _pad_rows(k_view, kv_over)
        v_view = _pad_rows(v_view, kv_over)
    else:
        # Dense layouts address each batch from its own base, so the
        # per-batch pad below covers the tail tile.
        q_view = _pad_tokens(q_view, M_BASE_SIZE)
        k_view = _pad_tokens(k_view, S2_BASE_SIZE)
        v_view = _pad_tokens(v_view, S2_BASE_SIZE)

    # Strided-input layouts feed the kernel rank-2 flats (tokens*heads, D):
    # one multi-source ND2NZ DMA per tile replaces the wrapper permutes.
    # Padded per-batch token extent (NOT the pad increment): the flat
    # row base for batch b is b * s_pad * heads + head. Only BSND uses
    # it (the flat row DMA needs constant strides); the stride comes
    # from the padded STORAGE extent, never from seqused_kv[0], so a
    # batch with a shorter effective KV length still addresses its own
    # storage slice.
    if public_layout == "BSND":
        q_s_pad = q_view.shape[1]
        kv_s_pad = k_view.shape[1]
    else:
        q_s_pad = 0
        kv_s_pad = 0
    if public_layout == "BSND":
        q_view = q_view.view(-1, D)
        k_view = k_view.view(-1, D)
        v_view = v_view.view(-1, D)
    elif public_layout == "TND":
        # Head-major strided views (N, T, D): one ND2NZ DMA per tile instead
        # of per-row multi-source gathers.
        q_view = q_view.permute(1, 0, 2)
        k_view = k_view.permute(1, 0, 2)
        v_view = v_view.permute(1, 0, 2)
        if (
            n_q_heads >= Q_DENSE_COPY_MIN_HEADS
            and max(q_lens) >= Q_DENSE_COPY_MIN_TOKENS
        ):
            # Same burst-density argument as K/V below, but stronger: with 20
            # Q heads a token-major row stride of 2560 B leaves 5% of every
            # fetch useful, and Q is read once per tile. Long sequences pay
            # for the single linear repack many times over.
            q_view = q_view.contiguous()
        if (
            kv_heads >= KV_DENSE_COPY_MIN_HEADS
            and mask_mode != 3
            and max(kv_lengths) >= KV_DENSE_COPY_MIN_TOKENS
        ):
            # A token-major stride of nkv * D leaves only D / (nkv * D) of
            # every DMA burst useful: at nkv=8 a 256-row KV tile is 12.5%
            # dense and the repeated K/V fetch (once per Q head) exposes it.
            # Repacking K/V head-major costs one linear pass over the tensor
            # and makes every later tile read fully contiguous; measured to
            # win from four KV heads upwards, neutral below that.
            k_view = k_view.contiguous()
            v_view = v_view.contiguous()

    # Use the platform's real AIC count: hardcoding 36 on a 32-AIC part
    # over-subscribes the grid and serializes a second wave of blocks.
    if fd_splits:
        ranges = _split_task_ranges(batch * n_q_heads * fd_splits, cores)
        _, boundaries = _build_task_ranges(q_lengths, n_q_heads, M_BASE_SIZE, cores)
    else:
        ranges, boundaries = _build_task_ranges(
            q_lengths, n_q_heads, M_BASE_SIZE, cores
        )
    device_key = str(query.device)
    # Build on the tensor's device, not the process-wide current one: with
    # more than one visible NPU a `.npu()` default can land on the wrong card.
    core_task_ranges = _aux_tensor(
        (device_key, "ranges", tuple(ranges[0]), tuple(ranges[1])),
        lambda: torch.tensor(ranges, dtype=torch.int32, device=query.device),
    )
    q_boundaries = _aux_tensor(
        (device_key, "boundaries", tuple(boundaries)),
        lambda: torch.tensor([boundaries], dtype=torch.int32, device=query.device),
    )
    kv_lengths_t = _aux_tensor(
        (device_key, "kv_lengths", tuple(kv_lengths)),
        lambda: torch.tensor([kv_lengths], dtype=torch.int32, device=query.device),
    )
    nz_gather = _aux_tensor(
        (device_key, "nz_gather"),
        lambda: torch.tensor(
            [nz_gather_index(M_BASE_SIZE // 2)],
            dtype=torch.uint8,
            device=query.device,
        ),
    )

    # Output stays in the kernel-internal layout (BNSD / head-major TND) so
    # the finalize store stays one contiguous DMA; BSND/TND permute back
    # once on return (1 copy instead of 4).
    kernel_layout = "BNSD" if public_layout in ("BNSD", "BSND") else "TND"
    if layout == "TND":
        out = torch.empty(
            (n_q_heads, boundaries[-1], D), dtype=torch.bfloat16, device=query.device
        )
    else:
        out = torch.empty(
            (batch, n_q_heads, q_len, D), dtype=torch.bfloat16, device=query.device
        )

    lse = None
    if return_softmax_lse or fd_splits:
        # AscendC contract: TND exports (N, T), every dense layout exports
        # (B, N, S). That is also the kernel-native head-major layout, so the
        # buffer is passed and returned as-is (no permute view, whose stride
        # the ABI guard rejects).
        if public_layout == "TND":
            lse_shape = (n_q_heads, boundaries[-1])
        else:
            lse_shape = (batch, n_q_heads, q_len)
        lse = torch.full(
            lse_shape, float("nan"), dtype=torch.float32, device=query.device
        )

    fd_splits_t = partial_o = partial_lse = None
    if fd_splits:
        fd_splits_t = _aux_tensor(
            (device_key, "fd_splits", fd_splits, batch),
            lambda: torch.tensor(
                [[fd_splits] * batch], dtype=torch.int32, device=query.device
            ),
        )
        # One padded row block per (batch, split); TND packs the batch axis
        # into the token stream so the partial indexing stays 2-D.
        if kernel_layout == "TND":
            partial_tokens = fd_splits * batch * M_BASE_SIZE
            partial_o = torch.zeros(
                (n_q_heads, partial_tokens, D),
                dtype=torch.float32,
                device=query.device,
            )
            partial_lse = torch.full(
                (n_q_heads, partial_tokens),
                -1e30,
                dtype=torch.float32,
                device=query.device,
            )
        else:
            partial_tokens = fd_splits * M_BASE_SIZE
            partial_o = torch.zeros(
                (batch, n_q_heads, partial_tokens, D),
                dtype=torch.float32,
                device=query.device,
            )
            partial_lse = torch.full(
                (batch, n_q_heads, partial_tokens),
                -1e30,
                dtype=torch.float32,
                device=query.device,
            )

    kernel = _get_compiled_kernel(
        causal=(mask_mode == 3),
        layout=kernel_layout,
        in_layout=in_layout,
        qk_scale=qk_scale,
        v_descale=v_descale,
        p_scale=p_scale_value,
        q_geom=(q_s_pad, n_q_heads),
        kv_geom=(kv_s_pad, kv_heads),
        with_lse=return_softmax_lse or bool(fd_splits),
        fd=bool(fd_splits),
    )
    call_args = (
        out,
        q_view,
        k_view,
        v_view,
        core_task_ranges,
        q_boundaries,
        kv_lengths_t,
        nz_gather,
        batch,
        n_q_heads,
        head_group,
        cores,
    )
    if fd_splits:
        kernel(*(call_args + (lse, fd_splits_t, partial_o, partial_lse)))
        combine = _get_compiled_combine(kernel_layout, return_softmax_lse, cores)
        combine_args = (
            partial_o,
            partial_lse,
            fd_splits_t,
            q_boundaries,
            out,
            batch,
            n_q_heads,
        )
        combine(*(combine_args + ((lse,) if return_softmax_lse else ())))
    else:
        kernel(*(call_args + ((lse,) if return_softmax_lse else ())))
    result_lse = lse if return_softmax_lse else None
    if public_layout == "BSND":
        return out.permute(0, 2, 1, 3).contiguous(), result_lse
    if public_layout == "TND":
        return out.permute(1, 0, 2).contiguous(), result_lse
    return out, result_lse
