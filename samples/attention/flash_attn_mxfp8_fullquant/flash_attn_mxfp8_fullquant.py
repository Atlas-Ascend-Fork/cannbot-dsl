# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""GQA MXFP8 Flash Attention with DN prefill and ND decode kernels.

Q, K, and V are FP8 E4M3 values with E8M0 scales per 32-element block.
Each query-head group shares one KV head. The products below broadcast
each block scale over its FP8 values; M masks invalid or causal keys.

    QK:      S = softmax_scale * ((Q * Q_Scale) @ (K * K_Scale).T) + M
    Softmax: P = FP8_E4M3(p_scale * exp(S - m_tile))
             sum = p_scale * reduce_sum(exp(S - m), axis=KV)
    PV:      O += (P * P_Scale) @ (V * V_Scale)
    O/sum:   output = BF16(O / (sum + epsilon))

m is the running row maximum; m_tile is the maximum used to form P.
P_Scale is an E8M0 correction that aligns P with the current max frame;
it is 1 when no correction is needed. Earlier O and sum are rescaled
when m grows. The FP32 p_scale multiplies both O and sum (before P
rounds to FP8). For nonempty rows, optional FP32 LSE is
m + log(sum / p_scale). The public entry supports contiguous or paged KV.
"""

__all__ = ["npu_flash_atten_mxfp8_fullquant"]

from collections import OrderedDict
from contextlib import nullcontext
import math as _math
import os
from threading import RLock
from types import SimpleNamespace

import torch

from cannbotdsl import (
    Constexpr,
    MemLoc,
    Tensor,
    TensorSpec,
    compile as compile_program,
    ceil_div,
    const_expr,
    dtypes,
    get_block_idx,
    get_platform_info,
    get_subblock_id,
    idx2crd,
    jit,
    kernel,
    range_constexpr,
    copy_options,
    matmul,
    mem_copy as _mem_copy,
    reg as rr,
    vf,
)
from cannbotdsl import Array, from_tensor, make_layout
from cannbotdsl.buffer import Buffer as _Buffer
from cannbotdsl.lang.host import host
from cannbotdsl.ring_buffer import RingBuffer as _RingBuffer
from cannbotdsl.ops.sync import (
    PIPE,
    cube_sync_pipe,
    global_sync_all,
    onchip_rewind,
)
from cannbotdsl.ops.memcpy import cube_fill_l1, cube_fill_l1_zero
from cannbotdsl.tensor import tile_slice
from .mxfp8_metadata import (
    AttentionMetadataState as AttentionMetadataState,
    FA_START_BN,
    FA_START_M,
    FA_START_S2,
    FA_END_BN,
    FA_END_M,
    FA_END_S2,
    FA_FIRST_SLOT,
    FD_BN,
    FD_M,
    FD_FIRST_SLOT,
    FD_PARTS,
    FD_ROW_BEGIN,
    FD_ROWS,
    METADATA_STRIDE,
    SECTION_COUNT,
    HAS_FD,
    prepare_metadata,
)
from .mxfp8_check import (
    DECODE_QUERY_ROWS,
    HEAD_DIM_64,
    HEAD_DIM_72,
    HEAD_DIM_128,
    HEAD_DIM_256,
    KVLayout,
    QLayout,
    MASK_MODE_NO_MASK,
    MASK_MODE_RIGHT_DOWN_CAUSAL,
    check_pa_tile,
    check_public_inputs,
    _check_output_token_count,
    check_mask_mode,
)

# =============================================================================
# Public API and module constants
# =============================================================================

Int64 = dtypes.int64

LN2 = _math.log(2.0)
INV_LN2 = 1.0 / _math.log(2.0)

# Raw vector widths and element bit sizes.
FP32_VECTOR_LANES = 64
FP32_ELEM_BITS = 32
INT8_ELEM_BITS = 8
FP8_REPACK_VECTOR_BYTES = 256

# MXFP8 and fractal-layout granularities.
FP8_NZ_K_BLOCK = 32
P_TILE_N_ALIGNMENT = 64
P_SCALE_ZN_M0_ROWS = 16
MXFP8_SCALE_GROUP_ELEMENTS = 64
MXFP8_SCALE_LANES_PER_GROUP = 2
VSCALE_D_BLOCK = 16
E8M0_SCALE_ONE_BYTE = 127

# Pipeline and split-M partitioning.
SOFTMAX_DELAY_SLOTS = 3
# One FP32 statistic per 32-byte DMA row; remaining lanes are padding.
FD_STAT_ROW_ELEMENTS = 8
AIV_SUBBLOCK_COUNT = 2
SPLIT_M_ALIGNMENT = 16
SPLIT_M_TOTAL_ALIGNMENT = 32

# Softmax numeric sentinels.
SOFTMAX_MASKED_SCORE_FLOOR = -3.4028232635611926e38
SOFTMAX_DENOMINATOR_EPSILON = 1e-30

_COMPILED_ATTENTION_PROGRAMS = OrderedDict()
_ATTENTION_COMPILE_LOCK = RLock()
_ATTENTION_PROGRAM_CACHE_LIMIT = 32
_ATTENTION_TENSOR_DTYPES = {
    torch.bfloat16: dtypes.bfloat16,
    torch.float32: dtypes.float32,
    torch.int8: dtypes.int8,
    torch.float8_e4m3fn: dtypes.float8_e4m3fn,
    torch.float8_e8m0fnu: dtypes.float8_e8m0,
    torch.int32: dtypes.int32,
    torch.int64: dtypes.int64,
}

# =============================================================================
# Host-side shape, padding, and PA_NZ conversion helpers
# =============================================================================


# ---- Shape and MXFP8 geometry ------------------------------------------------
def _ceil_div_int(value, divisor):
    return (int(value) + divisor - 1) // divisor


def _div_blocks32(size):
    return _ceil_div_int(size, FP8_NZ_K_BLOCK)


def _mx_groups64(size):
    return _ceil_div_int(size, MXFP8_SCALE_GROUP_ELEMENTS)


def _aligned_tile_head_dim(head_dim):
    """Use the internal 128-D tile plan for public padded-72-D inputs.

    D=72 input padding is supplied by the DSL compatibility kernel; the
    final attention output retains its logical 72 columns.
    """
    if const_expr(head_dim == HEAD_DIM_72):
        return HEAD_DIM_128
    return head_dim


def select_tile_config(head_dim):
    """Return (tile_cube_m, tile_vec_m, tile_n, tile_d).

    tile_cube_m is the Q rows in one Cube tile; tile_vec_m is the Q rows
    handled by each of the two AIV subblocks. tile_n is the KV tokens in
    one QK/softmax subtile (up to two subtiles form a macro). tile_d is the
    internal head dimension, with public D=72 padded to 128.
    """
    tile_head_dim = _aligned_tile_head_dim(head_dim)
    if const_expr(tile_head_dim == HEAD_DIM_256):
        # D=256 uses two 128-column subtiles per 256-token macro.
        return 128, 64, 128, 256
    if const_expr(tile_head_dim == HEAD_DIM_128):
        # D=128 and padded D=72 use 64 rows per AIV and two 256-column subtiles.
        return 128, 64, 256, 128
    if const_expr(tile_head_dim == HEAD_DIM_64):
        return 128, 64, 256, 64


# =============================================================================
# Shared DSL helpers
# =============================================================================


@jit
def _clamp_valid_lanes(value, width):
    """Clamp a dynamic lane count into [0, width]."""
    non_negative = 0 if value < 0 else value
    return width if non_negative > width else non_negative


@jit
def dynamic_min(a: dtypes.int64, b: dtypes.int64) -> dtypes.int64:
    return a if a < b else b


@jit
def dynamic_max(a: dtypes.int64, b: dtypes.int64) -> dtypes.int64:
    return a if a > b else b


# ---- Cube: QK/PV matmul -----------------------------------------------------
class _DelaySlot:
    def __init__(self, values, clock, depth):
        self.values = values
        self.clock = clock
        self.depth = depth

    def tap(self, lag):
        if not isinstance(lag, int) or not 0 < lag < self.depth:
            raise ValueError("delay must be a positive integer smaller than depth")
        return self.values[(self.clock[0] - lag) % self.depth]


class DelayLineGroup:
    """Keep named scalar streams at a shared pipeline position."""

    def __init__(self, depth, *fields):
        if depth < 2 or not fields:
            raise ValueError("a delay line needs fields and at least two slots")
        self.depth = depth
        self.fields = fields
        self.clock = Array(dtypes.int64, 1, init=[0])
        self.values = {field: Array(dtypes.int64, depth) for field in fields}

    def push(self, **values):
        if set(values) != set(self.fields):
            raise ValueError("delay line fields do not match")
        slot = self.clock[0] % self.depth
        for name in self.fields:
            self.values[name][slot] = dtypes.int64(values[name])

    def tap(self, lag):
        slot = (self.clock[0] - lag) % self.depth
        return SimpleNamespace(
            **{name: self.values[name][slot] for name in self.fields}
        )

    def advance(self):
        self.clock[0] = self.clock[0] + 1

    def __getattr__(self, name):
        values = object.__getattribute__(self, "values")
        if name not in values:
            raise AttributeError(name)
        return _DelaySlot(values[name], self.clock, self.depth)


def mem_copy(dst, src, **kwargs):
    _mem_copy(dst, src, **kwargs)
    return dst


def _make_buffer(mem_loc, shape, dtype, *, data_format=None, n1_pad=0):
    if data_format == "zn":
        slot = _Buffer(
            mem_loc, dtype, make_layout(tuple(reversed(shape))), data_format="nz"
        )
        return from_tensor(slot, dtype, make_layout(shape), data_format="zn")
    return _Buffer(
        mem_loc,
        dtype,
        make_layout(shape),
        physical_layout=_padded_layout(shape, dtype, n1_pad),
        data_format=data_format,
    )


def _padded_layout(shape, dtype, n1_pad):
    if n1_pad:
        m1 = (shape[0] + 15) // 16
        k0 = 256 // dtype.width
        n1 = (shape[1] + k0 - 1) // k0
        return make_layout(
            (n1, m1, 16, k0), stride=(m1 * 16 * k0 + n1_pad, 16 * k0, k0, 1)
        )


def _make_ring_buffer(
    mem_loc,
    shape=None,
    dtype=None,
    *,
    depth,
    data_format=None,
    n1_pad=0,
    size_bytes=None,
):
    if size_bytes is not None:
        shape, dtype, data_format = (size_bytes,), dtypes.uint8, "nd"
    if data_format == "zn":
        ring = _RingBuffer(
            mem_loc,
            dtype,
            make_layout(tuple(reversed(shape))),
            depth=depth,
            data_format="nz",
        )
        return _SharedL1View(ring, _View(shape, dtype, "zn"))
    return _RingBuffer(
        mem_loc,
        dtype,
        make_layout(shape),
        depth=depth,
        physical_layout=_padded_layout(shape, dtype, n1_pad),
        data_format=data_format,
    )


def reinterpret(
    value, dtype=None, shape=None, *, stride=None, offset=0, data_format=None
):
    return from_tensor(
        value,
        dtype,
        make_layout(shape, stride=stride) if shape is not None else None,
        offset=offset,
        data_format=data_format,
    )


def _compact_l0(tensor, shape, bound=None):
    return from_tensor(tensor, layout=make_layout(shape), data_format="nz")


class _View:
    def __init__(self, shape, dtype, fmt):
        self.shape, self.dtype, self.fmt = shape, dtype, fmt

    def bind(self, tensor):
        return from_tensor(
            tensor, self.dtype, make_layout(self.shape), data_format=self.fmt
        )

    def storage_bytes(self, *, mem_loc):
        from cannbotdsl.tensor import _physical_layout_spec

        shape, stride = _physical_layout_spec(
            self.dtype, self.shape, self.fmt, 0, mem_loc=mem_loc
        )
        return (
            (1 + sum((n - 1) * s for n, s in zip(shape, stride)))
            * self.dtype.width
            // 8
        )


class _SharedL1View:
    """Bind one role-specific layout to a shared four-slot L1 pool."""

    def __init__(self, ring, spec):
        self.ring = ring
        self.spec = spec

    @property
    def shape(self):
        return self.spec.shape

    def next_write(self):
        return self.spec.bind(self.ring.next_write())

    def next_read(self):
        return self.spec.bind(self.ring.next_read())


class _CubeMatmulBase:
    """Stage MXFP8 operands and compute P @ V for both attention paths.

    Subclasses own QK layout and L0 channel allocation.
    """

    def __init__(self, tile_n, tile_d):
        """Configure shared tile dimensions and copy engines."""
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.regular_p = False
        self.share_l0_ring_buffers = False
        self.pv_l0a_spec = None
        self.pv_l0b_spec = None
        self.transposed_qk = False
        self.raw_pa_v = False

    # GM-to-L1 staging

    def load_k(self, k_l1, k_tile_gm: Tensor):
        """Stage a K tile from GM to L1."""
        slot = k_l1.next_write()
        if const_expr(len(k_tile_gm.shape) == 3):
            # PA_NZ [D/32, N, 32] is exactly the K L1 NZ fractal stream.
            raw = reinterpret(
                slot,
                dtype=dtypes.int8,
                shape=(self.tile_d // FP8_NZ_K_BLOCK, self.tile_n, FP8_NZ_K_BLOCK),
                data_format="nd",
            )
            mem_copy(raw, k_tile_gm.view(dtypes.int8))
        else:
            if const_expr(k_tile_gm.shape[-1] == HEAD_DIM_72):
                cube_fill_l1_zero(slot, blk_num=self.tile_n * self.tile_d // 32)
                # Finish initialization before the overlapping partial GM copy.
                cube_sync_pipe(PIPE.MTE2)
                mem_copy(slot[None, :HEAD_DIM_72], k_tile_gm)
            else:
                mem_copy(slot, k_tile_gm)

    def load_v(self, slot, v_tile_gm: Tensor):
        """Load V tile from GM into the configured NZ or ZN L1 layout."""
        if const_expr(len(v_tile_gm.shape) == 3 and self.raw_pa_v):
            # Keep the page's NZ byte order; transpose during L1-to-L0B load.
            raw = reinterpret(
                slot,
                dtype=dtypes.int8,
                shape=(self.tile_d // FP8_NZ_K_BLOCK, self.tile_n, FP8_NZ_K_BLOCK),
                data_format="nd",
            )
            mem_copy(raw, v_tile_gm.view(dtypes.int8))
        elif const_expr(len(v_tile_gm.shape) == 3):
            # V's GM NZ stream must become the PV ZN stream. Each D32 slice
            # is an affine ND matrix and uses the existing DN-to-NZ DMA.
            for d_block in range_constexpr(self.tile_d // FP8_NZ_K_BLOCK):
                source = tile_slice(
                    v_tile_gm[d_block, None, None],
                    (self.tile_n, FP8_NZ_K_BLOCK),
                    (0, 0),
                )
                destination = slot[
                    None, d_block * FP8_NZ_K_BLOCK: (d_block + 1) * FP8_NZ_K_BLOCK
                ]  # fmt: skip
                mem_copy(destination, source)
        else:
            if const_expr(v_tile_gm.shape[-1] == HEAD_DIM_72):
                cube_fill_l1_zero(slot, blk_num=self.tile_n * self.tile_d // 32)
                cube_sync_pipe(PIPE.MTE2)
                mem_copy(slot[None, :HEAD_DIM_72], v_tile_gm)
            else:
                mem_copy(slot, v_tile_gm)

    def load_k_scale(self, k_scale_l1, k_scale_tile_gm: Tensor):
        """Stage K E8M0 scales from GM to L1."""
        slot = k_scale_l1.next_write()
        if const_expr(len(k_scale_tile_gm.shape) == 4):
            # Both ND and DN K-scale slots use [N/16, D/64, 16, 2].
            raw = reinterpret(
                slot,
                dtype=dtypes.int8,
                shape=(
                    self.tile_n // P_SCALE_ZN_M0_ROWS,
                    self.tile_d // MXFP8_SCALE_GROUP_ELEMENTS,
                    P_SCALE_ZN_M0_ROWS,
                    MXFP8_SCALE_LANES_PER_GROUP,
                ),
                data_format="nd",
            )
            mem_copy(raw, k_scale_tile_gm.view(dtypes.int8))
        else:
            scale = k_scale_tile_gm.flatten(-2, -1)
            if not self.transposed_qk:
                scale = scale.permute(1, 0)
            mem_copy(slot, scale)

    def load_v_scale(self, v_scale_l1, v_scale_tile_gm: Tensor):
        """Stage V E8M0 scales from GM to L1."""
        slot = v_scale_l1.next_write()
        if const_expr(len(v_scale_tile_gm.shape) == 4):
            # PA_NZ V scales already match [D/16, N/64, 16, 2].
            raw = reinterpret(
                slot,
                dtype=dtypes.int8,
                shape=(
                    self.tile_d // VSCALE_D_BLOCK,
                    self.tile_n // MXFP8_SCALE_GROUP_ELEMENTS,
                    VSCALE_D_BLOCK,
                    MXFP8_SCALE_LANES_PER_GROUP,
                ),
                data_format="nd",
            )
            mem_copy(raw, v_scale_tile_gm.view(dtypes.int8))
        else:
            if const_expr(v_scale_tile_gm.shape[-2] == HEAD_DIM_72):
                # E8M0 unit scales for padded V columns, matching the old GM pad.
                raw = reinterpret(
                    slot,
                    dtype=dtypes.uint32,
                    shape=(self.tile_d * 2 * (self.tile_n // 64) // 4,),
                    data_format="nd",
                )
                cube_fill_l1(
                    raw, 0x7F7F7F7F, blk_num=self.tile_d * 2 * (self.tile_n // 64) // 32
                )
                cube_sync_pipe(PIPE.MTE2)
                mem_copy(
                    slot[None, :HEAD_DIM_72],
                    v_scale_tile_gm.permute((0, 2, 1)).flatten(0, 1),
                )
            else:
                mem_copy(slot, v_scale_tile_gm.permute((0, 2, 1)).flatten(0, 1))

    # PV matmul

    def compute_pv(
        self,
        p_l1,
        v_l1,
        p_scale_l1,
        v_scale_l1,
        init=True,
        actual_k=None,
        accumulator=None,
    ):
        if actual_k is None:
            actual_k = self.tile_n
        # Preserve view provenance on the original NZ path. Only the regular
        # D256 P representation needs static-K dispatch for transposed MX loads.
        if self.regular_p:
            self._compute_pv_regular_dispatch(
                p_l1, v_l1, p_scale_l1, v_scale_l1, init, actual_k, accumulator
            )
        else:
            self._compute_pv_impl(
                p_l1, v_l1, p_scale_l1, v_scale_l1, init, actual_k, accumulator
            )

    @jit
    def _compute_pv_regular_dispatch(
        self, p_l1, v_l1, p_scale_l1, v_scale_l1, init, actual_k, accumulator
    ):
        if const_expr(self.tile_n == 256):
            if actual_k == 256:
                self._compute_pv_impl(
                    p_l1, v_l1, p_scale_l1, v_scale_l1, init, 256, accumulator
                )
            elif actual_k == 192:
                self._compute_pv_impl(
                    p_l1, v_l1, p_scale_l1, v_scale_l1, init, 192, accumulator
                )
            elif actual_k == 128:
                self._compute_pv_impl(
                    p_l1, v_l1, p_scale_l1, v_scale_l1, init, 128, accumulator
                )
            else:
                self._compute_pv_impl(
                    p_l1, v_l1, p_scale_l1, v_scale_l1, init, 64, accumulator
                )
        elif const_expr(self.tile_n == 128):
            if actual_k == 128:
                self._compute_pv_impl(
                    p_l1, v_l1, p_scale_l1, v_scale_l1, init, 128, accumulator
                )
            else:
                self._compute_pv_impl(
                    p_l1, v_l1, p_scale_l1, v_scale_l1, init, 64, accumulator
                )
        else:
            self._compute_pv_impl(
                p_l1, v_l1, p_scale_l1, v_scale_l1, init, 64, accumulator
            )

    def _compute_pv_impl(
        self,
        p_l1,
        v_l1,
        p_scale_l1,
        v_scale_l1,
        init=True,
        actual_k=None,
        accumulator=None,
    ):
        """Compute P @ V using explicit operand views and MX destination metadata."""
        if self.share_l0_ring_buffers:
            l0a_pv, l0b_pv, l0c_pv = self.l0a, self.l0b, self.l0c
        else:
            l0a_pv, l0b_pv, l0c_pv = self.l0a_pv, self.l0b_pv, self.l0c_pv
        if actual_k is None:
            actual_k = self.tile_n
        if const_expr(self.regular_p):
            p_view = p_l1[:actual_k, : self.tile_m].permute(1, 0)
        else:
            p_view = p_l1[: p_l1.shape[0], :actual_k]
        groups = MXFP8_SCALE_LANES_PER_GROUP * ceil_div(
            actual_k, MXFP8_SCALE_GROUP_ELEMENTS
        )
        p_scale_view = p_scale_l1[: p_scale_l1.shape[0], :groups]
        v_view = v_l1[:actual_k, : self.tile_d].permute(1, 0)
        v_scale_view = v_scale_l1[:groups, : self.tile_d]
        p_dst = self._l0_slot(l0a_pv, True, self.pv_l0a_spec)
        v_dst = self._l0_slot(l0b_pv, True, self.pv_l0b_spec)
        p_read = mem_copy(
            _compact_l0(p_dst, (p_view.shape[0], actual_k)),
            p_view,
            options=copy_options(mx_scale=p_scale_view),
        )
        v_read = mem_copy(
            _compact_l0(
                v_dst, (self.tile_d, actual_k), bound=(self.tile_d, self.tile_n)
            ),
            v_view,
            options=copy_options(mx_scale=v_scale_view),
        )
        if accumulator is None:
            accumulator = l0c_pv.next_write()
        matmul(accumulator, p_read, v_read, init=init, mx_mode=True)

    def _l0_slot(self, ring, write, spec):
        slot = ring.next_write() if write else ring.next_read()
        return spec.bind(slot)


def _merge_softmax_sum(
    old_sum, old_correction, first_sum, first_correction, second_sum, mask
):
    """Combine per-Q sums already expressed in their respective max frames."""
    return rr.vadd(
        rr.vmul(old_sum, old_correction, mask=mask),
        rr.vadd(rr.vmul(first_sum, first_correction, mask=mask), second_sum, mask=mask),
        mask=mask,
    )


def _e8m0_exponent_pairs(correction, mask, use_dn=True):
    """Pack per-Q power-of-two corrections; caller owns row selection/layout."""
    even = rr.vcast(
        correction,
        dtypes.bfloat16,
        mask=mask,
        rounding=rr.RoundingMode.RN,
        saturate=True,
        reg_layout=rr.RegLayout.ZERO,
    )
    odd = rr.vcast(
        correction,
        dtypes.bfloat16,
        mask=mask,
        rounding=rr.RoundingMode.RN,
        saturate=True,
        reg_layout=rr.RegLayout.ONE,
    )
    pairs_mask, _ = rr.update_mask(32, elem_bits=16)
    if not use_dn:
        even_scale = rr.vcast(
            even, dtypes.float8_e8m0, mask=pairs_mask, reg_layout=rr.RegLayout.ZERO
        )
        odd_scale = rr.vcast(
            odd, dtypes.float8_e8m0, mask=pairs_mask, reg_layout=rr.RegLayout.ZERO
        )
        bytes_mask, _ = rr.update_mask(256, elem_bits=8)
        compact = rr.vbitwise_or(
            rr.vreinterpret_lanes(even_scale, dtypes.uint8),
            rr.vreinterpret_lanes(odd_scale, dtypes.uint8),
            mask=bytes_mask,
        )
        unit = rr.vdups(E8M0_SCALE_ONE_BYTE, dtypes.uint8, mask=bytes_mask)
        pairs, _ = rr.vdeinterleave(compact, unit)
        return pairs
    words = rr.vbitwise_or(
        rr.vreinterpret_lanes(even, dtypes.uint16),
        rr.vreinterpret_lanes(odd, dtypes.uint16),
        mask=pairs_mask,
    )
    scales = rr.vcast(
        rr.vreinterpret_lanes(words, dtypes.bfloat16),
        dtypes.float8_e8m0,
        mask=pairs_mask,
        reg_layout=rr.RegLayout.ZERO,
    )
    return rr.vpack(rr.vreinterpret_lanes(scales, dtypes.uint16), dtypes.uint8)


# ---- Vector: online softmax and p quantization ------------------------------
class _VectorSoftmaxBase:
    """Shared online-softmax and MXFP8 P-scale flow.

    This stage scales and masks QK, updates running row max/sum, emits FP8 P,
    encodes E8M0 Scale-A, and publishes both to L1 for PV.  Prefill/decode only
    specialize the physical QK/P layout access and P-tile publishing.
    """

    def __init__(
        self,
        tile_vec_m,
        tile_n,
        mask_mode,
        subblock_idx,
        materialize_nd_p=True,
        regular_p=False,
        p_double_buffer=False,
    ):
        """Allocate shared UB buffers for online softmax and P-scale output."""
        self.tile_vec_m = tile_vec_m
        self.tile_m = tile_vec_m * 2
        self.tile_n = tile_n
        self.mask_mode = mask_mode
        self.subblock_idx = subblock_idx

        if materialize_nd_p:
            self.short_indices = _make_buffer(MemLoc.UB, (256,), dtypes.uint8)
        self.gather_indices = _make_buffer(MemLoc.UB, (256,), dtypes.uint8)
        self.p_double_buffer = p_double_buffer

        # NZ p storage uses FP8_NZ_K_BLOCK-element blocks on the K axis.
        if regular_p:
            # Eight payload blocks plus one padded block cover DMA view bounds.
            self.p_block_bytes = tile_n * 8
            self.p_padded_block_bytes = self.p_block_bytes + FP8_NZ_K_BLOCK
            self.p_data_bytes = 8 * self.p_padded_block_bytes
            self.p_capacity_bytes = 9 * self.p_padded_block_bytes
            if p_double_buffer:
                self.p_ub = _make_ring_buffer(
                    MemLoc.UB, (self.p_capacity_bytes,), dtypes.float8_e4m3fn, depth=2
                )
            else:
                self.p_ub = _make_buffer(
                    MemLoc.UB, (self.p_capacity_bytes,), dtypes.float8_e4m3fn
                )
        elif p_double_buffer:
            # Vector writes and MTE3 reads the same selected P slot; lowering orders reuse.
            self.p_ub = _make_ring_buffer(
                MemLoc.UB,
                shape=(tile_vec_m, tile_n),
                dtype=dtypes.float8_e4m3fn,
                depth=2,
                data_format="nz",
                n1_pad=FP8_NZ_K_BLOCK,
            )
        else:
            self.p_ub = _make_buffer(
                MemLoc.UB,
                shape=(tile_vec_m, tile_n),
                dtype=dtypes.float8_e4m3fn,
                data_format="nz",
                n1_pad=FP8_NZ_K_BLOCK,
            )

        # P is PV's A operand; its scale width follows tile_n / FP8_NZ_K_BLOCK.
        # Each AIV publishes only its own rows through the partitioned copy.
        self.p_tile_n_blocks32 = _div_blocks32(tile_n)

        # Per-AIV softmax state for three delayed macro slots.
        self.softmax_max_ub = [
            _make_buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
            for _ in range(SOFTMAX_DELAY_SLOTS)
        ]
        self.softmax_sum_ub = [
            _make_buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
            for _ in range(SOFTMAX_DELAY_SLOTS)
        ]
        # Retain each prior-max correction until its delayed PV tile is consumed.
        self.prev_running_max_correction_ub = [
            _make_buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
            for _ in range(SOFTMAX_DELAY_SLOTS)
        ]
        self.current_subtile_exp_sum_ub = _make_buffer(
            MemLoc.UB, (tile_vec_m, 1), dtypes.float32
        )

        # Scratch for merging the two subtiles in one running-max frame.
        self.max_first_subtile_ub = _make_buffer(
            MemLoc.UB, (tile_vec_m, 1), dtypes.float32
        )
        self.max_second_subtile_ub = _make_buffer(
            MemLoc.UB, (tile_vec_m, 1), dtypes.float32
        )
        self.updated_running_max_ub = _make_buffer(
            MemLoc.UB, (tile_vec_m, 1), dtypes.float32
        )
        self.first_subtile_max_correction_ub = _make_buffer(
            MemLoc.UB, (tile_vec_m, 1), dtypes.float32
        )
        self.first_subtile_exp_sum_ub = _make_buffer(
            MemLoc.UB, (tile_vec_m, 1), dtypes.float32
        )
        # ZN E8M0 Scale-A repeats each row correction across paired lanes
        # and local K groups; the partitioned copy writes this AIV's rows.
        self.p_scale_compact_ub = _make_buffer(
            MemLoc.UB,
            shape=(self.tile_vec_m, self.p_tile_n_blocks32),
            dtype=dtypes.float8_e8m0,
            data_format="zn",
        )

    def _align_maximum_to_log2_scale(self, vreg_row_max_values, mask):
        """Round row maxima upward to an ln2 multiple for exact E8M0 scales."""
        exponent = rr.vmuls(vreg_row_max_values, INV_LN2, mask=mask)
        return rr.vmuls(rr.vceil(exponent, mask=mask), LN2, mask=mask)

    @jit
    def _store_compact_scale_pairs(self, pair, row_block, mask):
        """Store common ZN Scale-A pairs; source loading stays ND/DN-specific."""
        rows_per_block = P_SCALE_ZN_M0_ROWS
        base = row_block * self.p_tile_n_blocks32 * rows_per_block
        for k1 in range(self.p_tile_n_blocks32 // MXFP8_SCALE_LANES_PER_GROUP):
            rr.vstore(
                self.p_scale_compact_ub,
                base + k1 * rows_per_block * MXFP8_SCALE_LANES_PER_GROUP,
                pair,
                mask,
            )

    def encode_unit_p_scale(self):
        """Encode the neutral E8M0 scale used by uncorrected P tiles."""

        @jit
        def _encode_unit_p_scale_bytes():
            total_bytes = self.tile_vec_m * self.p_tile_n_blocks32
            with vf(mode="simd"):
                full, _ = rr.update_mask(
                    FP8_REPACK_VECTOR_BYTES, elem_bits=INT8_ELEM_BITS
                )
                vreg_exponent_one = rr.vdups(
                    E8M0_SCALE_ONE_BYTE, dtypes.int8, mask=full
                )
                remaining = rr.mask_counter(total_bytes)
                for offset in range(0, total_bytes, FP8_REPACK_VECTOR_BYTES):
                    mask, remaining = rr.update_mask(
                        remaining, elem_bits=INT8_ELEM_BITS
                    )
                    rr.vstore(self.p_scale_compact_ub, offset, vreg_exponent_one, mask)
                rr.vmem_bar("vst_vld")

        _encode_unit_p_scale_bytes()

    def store_compact_p_scale(self, p_scale_l1, actual_m=None):
        """Copy this AIV's compact ZN Scale-A rows into the shared L1 slot."""

        @jit
        def _copy_compact_scale_to_l1():
            mem_copy(
                p_scale_l1.next_write().partition(
                    dim=0, part_id=self.subblock_idx, alignment=SPLIT_M_ALIGNMENT
                ),
                self.p_scale_compact_ub,
            )

        _copy_compact_scale_to_l1()


# ---- Optional softmax-LSE output --------------------------------------------
class SoftmaxLseOutput:
    """Materialize final FP32 LSE from the online-softmax row state.

    Prefill and Decode store maxima shifted by -ln(p_scale) and their exp
    sums. Public LSE is shifted_max + ln(sum), including FlashDecode merge. Rows with no visible KV element keep the
    established Flash Attention convention of +inf.
    """

    def __init__(self, tile_vec_m):
        self.tile_vec_m = tile_vec_m
        self.values = _make_buffer(
            MemLoc.UB,
            shape=(tile_vec_m, 1),
            dtype=dtypes.float32,
        )

    @jit
    def compute(self, maximum, denominator, actual_vec_m):
        with vf(mode="simd"):
            mask, _ = rr.update_mask(actual_vec_m, elem_bits=FP32_ELEM_BITS)
            vreg_row_max = rr.vload(maximum, 0)
            vreg_row_exp_sum = rr.vload(denominator, 0)
            vreg_invalid = rr.veqs(vreg_row_exp_sum, 0.0, mask=mask)
            vreg_one = rr.vdups(1.0, dtypes.float32, mask=mask)
            vreg_safe_exp_sum = rr.vselect(
                vreg_one, vreg_row_exp_sum, cond_mask=vreg_invalid
            )
            vreg_lse = rr.vadd(
                vreg_row_max, rr.vlog(vreg_safe_exp_sum, mask=mask), mask=mask
            )
            vreg_zero = rr.vdups(0.0, dtypes.float32, mask=mask)
            vreg_infinity = rr.vdiv(vreg_one, vreg_zero, mask=mask)
            vreg_lse = rr.vselect(vreg_infinity, vreg_lse, cond_mask=vreg_invalid)
            rr.vstore(self.values, 0, vreg_lse, mask)
            rr.vmem_bar("vst_vld")

    @jit
    def store(
        self,
        maximum,
        denominator,
        p_scale,
        softmax_lse_flat: Tensor,
        row_base,
        actual_vec_m,
    ):
        self.compute(maximum, denominator, actual_vec_m)
        if actual_vec_m > 0:
            mem_copy(
                tile_slice(
                    softmax_lse_flat[
                        row_base: row_base + actual_vec_m,
                        None,
                    ],
                    (self.tile_vec_m, 1),
                    (0, 0),
                ),
                reinterpret(
                    self.values,
                    shape=(actual_vec_m, 1),
                    stride=(1, 1),
                ),
            )  # fmt: skip


# ---- Vector: output accumulation and normalization --------------------------
class _VectorOutputBase:
    """Hold the FP32 PV numerator and pack normalized BF16 rows."""

    def __init__(self, tile_vec_m, tile_d, subblock_idx):
        """Allocate FP32 accumulation and BF16 store buffers for one AIV."""
        self.tile_d = tile_d
        self.subblock_idx = subblock_idx

        # Final BF16 output aliases dead FP32 numerator storage. Forward
        # row/column conversion writes behind all not-yet-read FP32 elements.
        self.res_o_ub = _make_buffer(MemLoc.UB, (tile_vec_m, tile_d), dtypes.float32)
        self.cast_ub_bf16 = reinterpret(
            self.res_o_ub, shape=(tile_vec_m, tile_d), dtype=dtypes.bfloat16
        )
        # Select the final macro's alpha once before the three sum-slot cases.
        # This avoids alpha-slot x sum-slot duplication of normalization VFs.
        self.final_correction_ub = _make_buffer(
            MemLoc.UB, (tile_vec_m, 1), dtypes.float32
        )

    @jit
    def init_invalid_row_check(self, causal, use_dn, scale, p_scale):
        self.check_invalid_rows = causal
        if const_expr(causal):
            self.invalid_max = _make_buffer(MemLoc.UB, (8,), dtypes.float32)
            with vf(mode="simd"):
                mask, _ = rr.update_mask(8, elem_bits=32)
                floor = rr.vdups(SOFTMAX_MASKED_SCORE_FLOOR, dtypes.float32, mask=mask)
                if const_expr(use_dn):
                    floor = rr.vmuls(floor, scale, mask=mask)
                    floor = rr.vmuls(
                        rr.vceil(rr.vmuls(floor, INV_LN2, mask=mask), mask=mask),
                        LN2,
                        mask=mask,
                    )
                    floor = rr.vsub(
                        floor,
                        rr.vlog(
                            rr.vdups(p_scale, dtypes.float32, mask=mask), mask=mask
                        ),
                        mask=mask,
                    )
                rr.vstore(self.invalid_max, 0, floor, mask)
                rr.vmem_bar("vst_vld")

    @jit
    def init_output(self, pv_ub):
        """Copy contiguous FP32 PV rows in full vector-width chunks."""
        elements = dtypes.int32(pv_ub.shape[0]) * self.tile_d
        with vf(mode="simd"):
            full, _ = rr.update_mask(FP32_VECTOR_LANES, elem_bits=32)
            for offset in range(0, elements, FP32_VECTOR_LANES):
                value = rr.vload(pv_ub, offset)
                rr.vstore(self.res_o_ub, offset, value, full)
            rr.vmem_bar("vst_vld")

    @jit
    def normalize_single_pv(self, pv, denominator, maximum, rows, check_invalid):
        """Divide the sole PV tile and write its BF16 representation directly."""
        with vf(mode="simd"):
            full, _ = rr.update_mask(FP32_VECTOR_LANES, elem_bits=32)
            if const_expr(check_invalid):
                zero = rr.vdups(0.0, dtypes.float32, mask=full)
                floor = rr.vload_broadcast(self.invalid_max, 0)
            for row in range(rows):
                den = rr.vload_broadcast(denominator, row)
                if const_expr(check_invalid):
                    invalid = rr.veq(rr.vload_broadcast(maximum, row), floor, mask=full)
                    rr.vstore_first(
                        denominator, row, rr.vselect(zero, den, cond_mask=invalid)
                    )
                for col in range(0, self.tile_d, FP32_VECTOR_LANES):
                    pos = row * self.tile_d + col
                    value = rr.vdiv(rr.vload(pv, pos), den, mask=full)
                    if const_expr(check_invalid):
                        value = rr.vselect(zero, value, cond_mask=invalid)
                    packed = rr.vcast(value, dtypes.bfloat16, mask=full)
                    rr.vstore_pack(
                        self.cast_ub_bf16, pos, packed, full, pack_mode="b32_to_b16"
                    )
            rr.vmem_bar("vst_vld")

    @jit
    def update_output(self, pv_ub, prev_running_max_correction_ub):
        """Apply O_new = O_old * exp(old_max - new_max) + PV_new."""
        rows = pv_ub.shape[0]
        with vf(mode="simd"):
            for i in range(rows):
                base = i * self.tile_d
                vreg_previous_max_correction_broadcast = rr.vload_broadcast(
                    prev_running_max_correction_ub, i
                )
                for j in range(0, self.tile_d, FP32_VECTOR_LANES):
                    mask, _ = rr.update_mask(
                        min(FP32_VECTOR_LANES, self.tile_d - j),
                        elem_bits=FP32_ELEM_BITS,
                    )
                    vreg_res = rr.vload(self.res_o_ub, base + j)
                    vreg_pv = rr.vload(pv_ub, base + j)
                    vreg_res = rr.vmul(
                        vreg_res, vreg_previous_max_correction_broadcast, mask=mask
                    )
                    vreg_res = rr.vadd(vreg_res, vreg_pv, mask=mask)
                    rr.vstore(self.res_o_ub, base + j, vreg_res, mask)

    @jit
    def _normalize_rows(
        self,
        softmax_sum_ub,
        actual_vec_m,
        final_pv=None,
        correction=None,
        maximum_ub=None,
        check_block=False,
    ):
        """Optionally merge final PV, divide live rows and pack BF16."""
        with vf(mode="simd"):
            for i in range(actual_vec_m):
                base = i * self.tile_d
                if const_expr(final_pv is not None):
                    alpha = rr.vload_broadcast(correction, i)
                vreg_exp_sum_broadcast = rr.vload_broadcast(softmax_sum_ub, i)
                full, _ = rr.update_mask(FP32_VECTOR_LANES, elem_bits=FP32_ELEM_BITS)
                for j in range(0, self.tile_d, FP32_VECTOR_LANES):
                    mask, _ = rr.update_mask(
                        min(FP32_VECTOR_LANES, self.tile_d - j),
                        elem_bits=FP32_ELEM_BITS,
                    )
                    vreg_res = rr.vload(self.res_o_ub, base + j)
                    if const_expr(final_pv is not None):
                        current = rr.vload(final_pv, base + j)
                        vreg_res = rr.vadd(
                            rr.vmul(vreg_res, alpha, mask=mask), current, mask=mask
                        )
                    vreg_res = rr.vdiv(vreg_res, vreg_exp_sum_broadcast, mask=mask)
                    rr.vstore(self.res_o_ub, base + j, vreg_res, mask)
            rr.vmem_bar("vst_vld")

        if const_expr(self.check_invalid_rows):
            if check_block:
                with vf(mode="simd"):
                    full, _ = rr.update_mask(64, elem_bits=32)
                    zero_f = rr.vdups(0.0, dtypes.float32, mask=full)
                    floor = rr.vload_broadcast(self.invalid_max, 0)
                    for i in range(actual_vec_m):
                        invalid = rr.veq(
                            rr.vload_broadcast(maximum_ub, i), floor, mask=full
                        )
                        sm = rr.vload_broadcast(softmax_sum_ub, i)
                        rr.vstore_first(
                            softmax_sum_ub, i, rr.vselect(zero_f, sm, cond_mask=invalid)
                        )
                        for j in range(0, self.tile_d, FP32_VECTOR_LANES):
                            pos = i * self.tile_d + j
                            lanes, _ = rr.update_mask(
                                min(64, self.tile_d - j), elem_bits=32
                            )
                            value = rr.vload(self.res_o_ub, pos)
                            rr.vstore(
                                self.res_o_ub,
                                pos,
                                rr.vselect(zero_f, value, cond_mask=invalid),
                                lanes,
                            )
                    rr.vmem_bar("vst_vld")
        with vf(mode="simd"):
            for i in range(actual_vec_m):
                for j in range(0, self.tile_d, FP32_VECTOR_LANES):
                    pos = i * self.tile_d + j
                    lanes, _ = rr.update_mask(min(64, self.tile_d - j), elem_bits=32)
                    value = rr.vcast(
                        rr.vload(self.res_o_ub, pos), dtypes.bfloat16, mask=lanes
                    )
                    rr.vstore_pack(
                        self.cast_ub_bf16, pos, value, lanes, pack_mode="b32_to_b16"
                    )
            rr.vmem_bar("vst_vld")


# ---- Sparse tile-range policy -----------------------------------------------
class BlockInfo:
    """Resolve Q/KV work ranges for full or right-down-causal attention."""

    def __init__(self, tile_m, tile_n, mask_mode):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.mask_mode = mask_mode

    def get_kv_tile_range(self, seqlen_q, seqlen_k, m_idx):
        n_block_max = ceil_div(seqlen_k, self.tile_n)
        if const_expr(self.mask_mode == MASK_MODE_NO_MASK):
            return 0, n_block_max
        if const_expr(self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL):
            m_idx_max = dynamic_min((m_idx + 1) * self.tile_m, seqlen_q)
            # mask_mode=MASK_MODE_RIGHT_DOWN_CAUSAL follows AscendC RIGHT_DOWN_CAUSAL semantics:
            # k <= q + (S_kv - S_q).
            head_idx_right = seqlen_k - seqlen_q + m_idx_max
            head_idx_right = dynamic_max(head_idx_right, Int64(0))
            n_block_max = dynamic_min(
                n_block_max, ceil_div(head_idx_right, self.tile_n)
            )
            return 0, n_block_max

    @jit
    def has_q_tile_work(self, seqlen_q, seqlen_k, m_idx):
        """Skip Q tiles with no query row or no legal causal KV element."""
        m_tile_start = m_idx * self.tile_m
        if const_expr(self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL):
            m_tile_end = dynamic_min((m_idx + 1) * self.tile_m, seqlen_q)
            causal_right = seqlen_k - seqlen_q + m_tile_end
            return causal_right > Int64(0) if m_tile_start < seqlen_q else False
        # Empty KV must not reset state while an older PV is still pending.
        return m_tile_start < seqlen_q and seqlen_k > Int64(0)

    @jit
    def get_macro_kv_blocks(self, n_idx_start, n_idx_end, macro_idx, seqlen_k):
        """Return both KV subtiles, their live lengths, and tail presence."""
        n_block_a = n_idx_start + macro_idx * 2
        n_block_b = n_block_a + 1
        valid_n_a = dynamic_min(Int64(self.tile_n), seqlen_k - n_block_a * self.tile_n)
        valid_n_b = dynamic_min(Int64(self.tile_n), seqlen_k - n_block_b * self.tile_n)
        has_second = Int64(1) if n_block_b < n_idx_end else Int64(0)
        return n_block_a, n_block_b, valid_n_a, valid_n_b, has_second

    def get_visible_kv_lengths(
        self,
        m_idx,
        local_q_offset,
        seqlen_q,
        seqlen_k,
        n_block_a,
        n_block_b,
        valid_n_a,
        valid_n_b,
    ):
        """Return the KV lengths visible to this AIV for both subtiles."""
        if const_expr(self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL):
            local_q_start = m_idx * self.tile_m + local_q_offset
            dynamic_valid_n_a = (
                local_q_start + seqlen_k - seqlen_q - n_block_a * self.tile_n + Int64(1)
            )
            dynamic_valid_n_b = (
                local_q_start + seqlen_k - seqlen_q - n_block_b * self.tile_n + Int64(1)
            )
            return dynamic_valid_n_a, dynamic_valid_n_b
        return valid_n_a, valid_n_b


# ---- Shared delayed PV consumption ------------------------------------------
class _KVTileAccess:
    """Share GM KV views and staging; QK/PV layouts remain path-specific."""

    @jit
    def _pa_nz_key_views(self, key, scales, page, head, in_page_tile):
        data = tile_slice(
            key[page, head, None, None, None],
            (self.tile_d // FP8_NZ_K_BLOCK, self.page_tile_n, FP8_NZ_K_BLOCK),
            (0, in_page_tile, 0),
        )
        scale = tile_slice(
            scales[page, head, None, None, None, None],
            (
                self.page_tile_n // P_SCALE_ZN_M0_ROWS,
                self.qk_scale_groups,
                P_SCALE_ZN_M0_ROWS,
                MXFP8_SCALE_LANES_PER_GROUP,
            ),
            (in_page_tile, 0, 0, 0),
        )
        return data, scale

    @jit
    def _pa_nz_value_views(self, value, scales, page, head, in_page_tile):
        data = tile_slice(
            value[page, head, None, None, None],
            (self.tile_d // FP8_NZ_K_BLOCK, self.page_tile_n, FP8_NZ_K_BLOCK),
            (0, in_page_tile, 0),
        )
        scale = tile_slice(
            scales[page, head, None, None, None, None],
            (
                self.tile_d // VSCALE_D_BLOCK,
                self.page_tile_n // MXFP8_SCALE_GROUP_ELEMENTS,
                VSCALE_D_BLOCK,
                MXFP8_SCALE_LANES_PER_GROUP,
            ),
            (0, in_page_tile, 0, 0),
        )
        return data, scale

    @jit
    def _paged_key_views(
        self, key, k_scale_gm, physical_page, kv_head_idx, in_page_tile
    ):
        if const_expr(self.pa_nz):
            return self._pa_nz_key_views(
                key, k_scale_gm, physical_page, kv_head_idx, in_page_tile
            )
        if const_expr(self.pa_bbnd):
            key_page = key[physical_page, None, kv_head_idx, None]
            scale_page = k_scale_gm[physical_page, None, kv_head_idx, None, None]
        else:
            key_page = key[physical_page, kv_head_idx, None, None]
            scale_page = k_scale_gm[physical_page, kv_head_idx, None, None, None]
        key_tile = tile_slice(
            key_page, (self.page_tile_n, self.tile_d), (in_page_tile, 0)
        )
        scale_tile = tile_slice(
            scale_page,
            (self.page_tile_n, self.qk_scale_groups, 2),
            (in_page_tile, 0, 0),
        )
        return key_tile, scale_tile

    @jit
    def _key_and_scale_tile(
        self,
        key,
        k_scale_gm,
        block_table,
        batch_idx,
        kv_head_idx,
        kv_token_start,
        kv_length,
        n_block,
    ):
        if const_expr(self.paged):
            page_idx = n_block // self.tiles_per_page
            in_page_tile = n_block % self.tiles_per_page
            physical_page = block_table[batch_idx, page_idx]
            return self._paged_key_views(
                key, k_scale_gm, physical_page, kv_head_idx, in_page_tile
            )

        key_ntd = key.permute((1, 0, 2))
        key_head = key_ntd[
            kv_head_idx,
            kv_token_start: kv_token_start + kv_length,
            None,
        ]  # fmt: skip
        key_tile = tile_slice(key_head, (self.tile_n, key.shape[-1]), (n_block, 0))
        scale_ntg2 = k_scale_gm.permute((1, 0, 2, 3))
        scale_head = scale_ntg2[
            kv_head_idx,
            kv_token_start: kv_token_start + kv_length,
            None,
            None,
        ]  # fmt: skip
        scale_tile = tile_slice(
            scale_head, (self.tile_n, self.qk_scale_groups, 2), (n_block, 0, 0)
        )
        return key_tile, scale_tile

    @jit
    def _load_key_and_scale_tile(
        self,
        key,
        k_scale_gm,
        block_table,
        batch_idx,
        kv_head_idx,
        kv_token_start,
        kv_length,
        n_block,
    ):
        if const_expr(self.paged):
            self._stage_paged_tile(
                key,
                k_scale_gm,
                block_table,
                batch_idx,
                kv_head_idx,
                kv_length,
                n_block,
                self.k_l1.next_write(),
                self.k_scale_l1.next_write(),
                False,
            )
            return
        key_tile, scale_tile = self._key_and_scale_tile(
            key,
            k_scale_gm,
            block_table,
            batch_idx,
            kv_head_idx,
            kv_token_start,
            kv_length,
            n_block,
        )
        self.matmul.load_k(self.k_l1, key_tile)
        self.matmul.load_k_scale(self.k_scale_l1, scale_tile)

    @jit
    def _paged_value_views(
        self, value, v_scale_gm, physical_page, kv_head_idx, in_page_tile
    ):
        if const_expr(self.pa_nz):
            return self._pa_nz_value_views(
                value, v_scale_gm, physical_page, kv_head_idx, in_page_tile
            )
        if const_expr(self.pa_bbnd):
            value_page = value[physical_page, None, kv_head_idx, None]
            scale_page = v_scale_gm[physical_page, None, kv_head_idx, None, None]
        else:
            value_page = value[physical_page, kv_head_idx, None, None]
            scale_page = v_scale_gm[physical_page, kv_head_idx, None, None, None]
        value_tile = tile_slice(
            value_page, (self.page_tile_n, value.shape[-1]), (in_page_tile, 0)
        )
        scale_tile = tile_slice(
            scale_page,
            (self.page_tile_n // MXFP8_SCALE_GROUP_ELEMENTS, self.tile_d, 2),
            (in_page_tile, 0, 0),
        )
        return value_tile, scale_tile

    @jit
    def _value_and_scale_tile(
        self,
        value,
        v_scale_gm,
        block_table,
        batch_idx,
        kv_head_idx,
        kv_token_start,
        v_scale_start,
        kv_length,
        n_block,
    ):
        if const_expr(self.paged):
            page_idx = n_block // self.tiles_per_page
            in_page_tile = n_block % self.tiles_per_page
            physical_page = block_table[batch_idx, page_idx]
            return self._paged_value_views(
                value, v_scale_gm, physical_page, kv_head_idx, in_page_tile
            )

        value_ntd = value.permute((1, 0, 2))
        value_head = value_ntd[
            kv_head_idx,
            kv_token_start: kv_token_start + kv_length,
            None,
        ]  # fmt: skip
        value_tile = tile_slice(
            value_head, (self.tile_n, value.shape[-1]), (n_block, 0)
        )
        scale_group_count = ceil_div(kv_length, MXFP8_SCALE_GROUP_ELEMENTS)
        scale_ntd2 = v_scale_gm.permute((1, 0, 2, 3))
        scale_head = scale_ntd2[
            kv_head_idx, v_scale_start: v_scale_start + scale_group_count, None, None
        ]  # fmt: skip
        scale_tile = tile_slice(
            scale_head, (self.v_scale_groups, v_scale_gm.shape[-2], 2), (n_block, 0, 0)
        )
        return value_tile, scale_tile

    @jit
    def _load_value_and_scale_tile(
        self,
        value,
        v_scale_gm,
        block_table,
        batch_idx,
        kv_head_idx,
        kv_token_start,
        v_scale_start,
        kv_length,
        n_block,
    ):
        if const_expr(self.paged):
            self._stage_paged_tile(
                value,
                v_scale_gm,
                block_table,
                batch_idx,
                kv_head_idx,
                kv_length,
                n_block,
                self.v_l1.next_write(),
                self.v_scale_l1.next_write(),
                True,
            )
            return
        value_tile, scale_tile = self._value_and_scale_tile(
            value,
            v_scale_gm,
            block_table,
            batch_idx,
            kv_head_idx,
            kv_token_start,
            v_scale_start,
            kv_length,
            n_block,
        )
        slot = self.v_l1.next_write()
        if const_expr(value_tile.shape[-1] != HEAD_DIM_72):
            # PV reads up to the next group of 64, including unwritten tail rows.
            valid_k = dynamic_min(Int64(self.tile_n), kv_length - n_block * self.tile_n)
            if valid_k % MXFP8_SCALE_GROUP_ELEMENTS != 0:
                if const_expr(self.v_l1.spec.fmt == "zn"):
                    # ZN packs each 64-token group into contiguous bytes across D.
                    group_bytes = MXFP8_SCALE_GROUP_ELEMENTS * self.tile_d
                    raw = reinterpret(
                        slot,
                        dtype=dtypes.uint8,
                        shape=(self.tile_n * self.tile_d,),
                        data_format="nd",
                    )
                    offset = (valid_k // MXFP8_SCALE_GROUP_ELEMENTS) * group_bytes
                    cube_fill_l1_zero(
                        raw[offset: offset + group_bytes], blk_num=group_bytes // 32
                    )  # fmt: skip
                else:
                    cube_fill_l1_zero(slot, blk_num=self.tile_n * self.tile_d // 32)
                cube_sync_pipe(PIPE.MTE2)
        self.matmul.load_v(slot, value_tile)
        self.matmul.load_v_scale(self.v_scale_l1, scale_tile)

    @jit
    def _compute_qk_tile(
        self,
        qk_ub,
        q_l1_tile,
        q_scale_l1_tile,
        key,
        k_scale_gm,
        block_table,
        batch_idx,
        kv_head_idx,
        kv_token_start,
        kv_length,
        n_block,
        actual_m,
    ):
        qk_slot = qk_ub.next_write()
        self._load_key_and_scale_tile(
            key,
            k_scale_gm,
            block_table,
            batch_idx,
            kv_head_idx,
            kv_token_start,
            kv_length,
            n_block,
        )
        # N 是本 subtile 的有效 KV 行数；不改变 L1 槽与 UB 行距。
        actual_n = dynamic_min(Int64(self.tile_n), kv_length - n_block * self.tile_n)
        self.matmul.compute_qk(
            q_l1_tile, self.k_l1, q_scale_l1_tile, self.k_scale_l1, actual_m, actual_n
        )
        self.matmul.store_qk_result(qk_slot, actual_m, actual_n)

    @jit
    def _stage_paged_tile(
        self,
        tensor,
        scales,
        block_table,
        batch,
        head,
        kv_length,
        n_block,
        slot,
        scale_slot,
        is_value,
    ):
        valid = dynamic_min(Int64(self.tile_n), kv_length - n_block * self.tile_n)
        if const_expr(is_value):
            # PV consumes K in groups of 64. Initialize padding before a partial V copy.
            if valid % MXFP8_SCALE_GROUP_ELEMENTS != 0:
                cube_fill_l1_zero(slot, blk_num=self.tile_n * self.tile_d // 32)
                cube_sync_pipe(PIPE.MTE2)
        first_page = n_block * self.pages_per_tile // self.tiles_per_page
        in_page = n_block % self.tiles_per_page
        for part in range_constexpr(self.pages_per_tile):
            offset = part * self.page_tile_n
            if offset < valid:
                page = block_table[batch, first_page + part]
                count = dynamic_min(Int64(self.page_tile_n), valid - offset)
                if const_expr(is_value):
                    data, scale = self._paged_value_views(
                        tensor, scales, page, head, in_page
                    )
                else:
                    data, scale = self._paged_key_views(
                        tensor, scales, page, head, in_page
                    )
                self._copy_page_tail(
                    slot, scale_slot, data, scale, offset, count, is_value
                )

    @jit
    def _copy_page_tail(self, slot, scale_slot, data, scale, offset, count, is_value):
        if const_expr(self.pa_nz):
            # Iterate D planes so each raw transfer is contiguous at fixed source/L1 pitch.
            raw = reinterpret(
                slot,
                dtype=dtypes.int8,
                shape=(self.tile_d // FP8_NZ_K_BLOCK, self.tile_n, FP8_NZ_K_BLOCK),
                data_format="nd",
            )
            for plane in range_constexpr(self.tile_d // FP8_NZ_K_BLOCK):
                mem_copy(
                    raw[plane, offset: offset + count, None],
                    data[plane, :count, None].view(dtypes.int8),
                )  # fmt: skip
            if const_expr(is_value):
                groups = ceil_div(count, MXFP8_SCALE_GROUP_ELEMENTS)
                group_start = offset // MXFP8_SCALE_GROUP_ELEMENTS
                raw_scale = reinterpret(
                    scale_slot,
                    dtype=dtypes.int8,
                    shape=(
                        self.tile_d // VSCALE_D_BLOCK,
                        self.v_scale_groups,
                        VSCALE_D_BLOCK,
                        2,
                    ),
                    data_format="nd",
                )
                for plane in range_constexpr(self.tile_d // VSCALE_D_BLOCK):
                    mem_copy(
                        raw_scale[
                            plane, group_start: group_start + groups, None, None
                        ],
                        scale[plane, :groups, None, None].view(dtypes.int8),
                    )  # fmt: skip
            else:
                blocks = ceil_div(count, P_SCALE_ZN_M0_ROWS)
                start = offset // P_SCALE_ZN_M0_ROWS
                raw_scale = reinterpret(
                    scale_slot,
                    dtype=dtypes.int8,
                    shape=(
                        self.tile_n // P_SCALE_ZN_M0_ROWS,
                        self.qk_scale_groups,
                        P_SCALE_ZN_M0_ROWS,
                        2,
                    ),
                    data_format="nd",
                )
                mem_copy(
                    raw_scale[start: start + blocks, None, None, None],
                    scale[:blocks, None, None, None].view(dtypes.int8),
                )  # fmt: skip
        else:
            if const_expr(is_value):
                # PV output columns are independent; columns beyond logical D
                # are never stored. K retains full zero-padded reduction D.
                # Existing partial-N clearing still protects PV's K tail.
                mem_copy(
                    slot[offset: offset + count, : self.copy_v_d],
                    data[:count, : self.copy_v_d],
                )  # fmt: skip
            else:
                mem_copy(slot[offset: offset + count, None], data[:count, None])  # fmt: skip
            if const_expr(is_value):
                lanes = scale.permute((0, 2, 1)).flatten(0, 1)
                start = 2 * (offset // MXFP8_SCALE_GROUP_ELEMENTS)
                width = 2 * ceil_div(count, MXFP8_SCALE_GROUP_ELEMENTS)
                mem_copy(scale_slot[start: start + width, None], lanes[:width, None])  # fmt: skip
            else:
                lanes = scale.flatten(-2, -1)
                if const_expr(self.matmul.transposed_qk):
                    mem_copy(
                        scale_slot[offset: offset + count, None], lanes[:count, None]
                    )  # fmt: skip
                else:
                    transposed = lanes.permute(1, 0)
                    mem_copy(
                        scale_slot[None, offset: offset + count],
                        transposed[None, :count],
                    )  # fmt: skip


class _DelayedPVConsumer(_KVTileAccess):
    @jit
    def _save_fd_partial(
        self,
        slot,
        row_offset,
        rows,
        maximum,
        denominator,
        final_pv=None,
        correction=None,
    ):
        """Publish local normalized FP32 output and its unrounded max/sum."""
        with vf(mode="simd"):
            for row in range(rows):
                mask, _ = rr.update_mask(FP32_VECTOR_LANES, elem_bits=32)
                den = rr.vload_broadcast(denominator, row)
                den = rr.vadds(den, SOFTMAX_DENOMINATOR_EPSILON, mask=mask)
                if const_expr(correction is not None):
                    alpha = rr.vload_broadcast(correction, row)
                for col in range(0, self.tile_d, FP32_VECTOR_LANES):
                    offset = row * self.tile_d + col
                    if const_expr(final_pv is not None and correction is None):
                        value = rr.vload(final_pv, offset)
                    else:
                        value = rr.vload(self.vector_output.res_o_ub, offset)
                    if const_expr(correction is not None):
                        current = rr.vload(final_pv, offset)
                        value = rr.vadd(
                            rr.vmul(value, alpha, mask=mask), current, mask=mask
                        )
                    value = rr.vdiv(value, den, mask=mask)
                    rr.vstore(self.vector_output.res_o_ub, offset, value, mask)
                stat_mask, _ = rr.update_mask(FD_STAT_ROW_ELEMENTS, elem_bits=32)
                mx = rr.vload_broadcast(maximum, row)
                sm = rr.vload_broadcast(denominator, row)
                rr.vstore(self.fd_max_pack, row * FD_STAT_ROW_ELEMENTS, mx, stat_mask)
                rr.vstore(self.fd_sum_pack, row * FD_STAT_ROW_ELEMENTS, sm, stat_mask)
            rr.vmem_bar("vst_vld")
        if rows > 0:
            dst = tile_slice(
                self.fd_output_gm.view(
                    (
                        self.fd_output_gm.shape[0] // self.tile_cube_m,
                        self.tile_cube_m,
                        self.tile_d,
                    )
                )[slot, row_offset: row_offset + rows, None],
                (self.tile_vec_m, self.tile_d),
                (0, 0),
            )  # fmt: skip
            mem_copy(dst, self.vector_output.res_o_ub[:rows, None])
            dst_sum = tile_slice(
                self.fd_sum_gm.view(
                    (
                        self.fd_sum_gm.shape[0] // self.tile_cube_m,
                        self.tile_cube_m,
                        FD_STAT_ROW_ELEMENTS,
                    )
                )[slot, row_offset: row_offset + rows, None],
                (self.tile_vec_m, FD_STAT_ROW_ELEMENTS),
                (0, 0),
            )  # fmt: skip
            dst_max = tile_slice(
                self.fd_max_gm.view(
                    (
                        self.fd_max_gm.shape[0] // self.tile_cube_m,
                        self.tile_cube_m,
                        FD_STAT_ROW_ELEMENTS,
                    )
                )[slot, row_offset: row_offset + rows, None],
                (self.tile_vec_m, FD_STAT_ROW_ELEMENTS),
                (0, 0),
            )  # fmt: skip
            mem_copy(dst_sum, self.fd_sum_pack[:rows, None])
            mem_copy(dst_max, self.fd_max_pack[:rows, None])

    @jit
    def _init_fd_accumulator(self, rows):
        with vf(mode="simd"):
            mask, _ = rr.update_mask(rows, elem_bits=32)
            initial_max = rr.vdups(
                SOFTMAX_MASKED_SCORE_FLOOR, dtypes.float32, mask=mask
            )
            zero = rr.vdups(0.0, dtypes.float32, mask=mask)
            rr.vstore(self.fd_running_max, 0, initial_max, mask)
            rr.vstore(self.fd_running_sum, 0, zero, mask)
            for row in range(rows):
                full, _ = rr.update_mask(FP32_VECTOR_LANES, elem_bits=32)
                z = rr.vdups(0.0, dtypes.float32, mask=full)
                for col in range(0, self.tile_d, FP32_VECTOR_LANES):
                    rr.vstore(
                        self.vector_output.res_o_ub, row * self.tile_d + col, z, full
                    )
            rr.vmem_bar("vst_vld")

    @jit
    def _update_fd_accumulator(self, rows):
        part_input = self.fd_part_input.next_read()
        sum_input = self.fd_sum_input.next_read()
        max_input = self.fd_max_input.next_read()
        with vf(mode="simd"):
            for row in range(rows):
                mask, _ = rr.update_mask(FP32_VECTOR_LANES, elem_bits=32)
                old_m = rr.vload_broadcast(self.fd_running_max, row)
                old_s = rr.vload_broadcast(self.fd_running_sum, row)
                part_m = rr.vload_broadcast(max_input, row * FD_STAT_ROW_ELEMENTS)
                part_s = rr.vload_broadcast(sum_input, row * FD_STAT_ROW_ELEMENTS)
                new_m = rr.vmax(old_m, part_m, mask=mask)
                alpha = rr.vexp(rr.vsub(old_m, new_m, mask=mask), mask=mask)
                beta = rr.vmul(
                    part_s,
                    rr.vexp(rr.vsub(part_m, new_m, mask=mask), mask=mask),
                    mask=mask,
                )
                new_s = rr.vadd(rr.vmul(old_s, alpha, mask=mask), beta, mask=mask)
                rr.vstore_first(self.fd_running_max, row, new_m)
                rr.vstore_first(self.fd_running_sum, row, new_s)
                for col in range(0, self.tile_d, FP32_VECTOR_LANES):
                    offset = row * self.tile_d + col
                    old_o = rr.vload(self.vector_output.res_o_ub, offset)
                    part_o = rr.vload(part_input, offset)
                    new_o = rr.vadd(
                        rr.vmul(old_o, alpha, mask=mask),
                        rr.vmul(part_o, beta, mask=mask),
                        mask=mask,
                    )
                    rr.vstore(self.vector_output.res_o_ub, offset, new_o, mask)
            rr.vmem_bar("vst_vld")

    @jit
    def _load_fd_part(self, slot, row_offset, rows):
        part = tile_slice(
            self.fd_output_gm.view(
                (
                    self.fd_output_gm.shape[0] // self.tile_cube_m,
                    self.tile_cube_m,
                    self.tile_d,
                )
            )[slot, row_offset: row_offset + rows, None],
            (self.tile_vec_m, self.tile_d),
            (0, 0),
        )  # fmt: skip
        sums = tile_slice(
            self.fd_sum_gm.view(
                (
                    self.fd_sum_gm.shape[0] // self.tile_cube_m,
                    self.tile_cube_m,
                    FD_STAT_ROW_ELEMENTS,
                )
            )[slot, row_offset: row_offset + rows, None],
            (self.tile_vec_m, FD_STAT_ROW_ELEMENTS),
            (0, 0),
        )  # fmt: skip
        maxima = tile_slice(
            self.fd_max_gm.view(
                (
                    self.fd_max_gm.shape[0] // self.tile_cube_m,
                    self.tile_cube_m,
                    FD_STAT_ROW_ELEMENTS,
                )
            )[slot, row_offset: row_offset + rows, None],
            (self.tile_vec_m, FD_STAT_ROW_ELEMENTS),
            (0, 0),
        )  # fmt: skip
        mem_copy(self.fd_part_input.next_write()[:rows, None], part)
        mem_copy(self.fd_sum_input.next_write()[:rows, None], sums)
        mem_copy(self.fd_max_input.next_write()[:rows, None], maxima)

    @jit
    def _store_default_range(self, attn_out, softmax_lse_gm, head, begin, count):
        """Write only rows omitted by the attention schedule; no live row overlaps."""
        if count > 0:
            with vf(mode="simd"):
                full, _ = rr.update_mask(FP32_VECTOR_LANES, elem_bits=FP32_ELEM_BITS)
                zero = rr.vdups(0.0, dtypes.float32, mask=full)
                bf16_zero = rr.vcast(zero, dtypes.bfloat16, mask=full)
                for row in range(self.tile_vec_m):
                    for col in range(0, self.tile_d, FP32_VECTOR_LANES):
                        rr.vstore_pack(
                            self.vector_output.cast_ub_bf16,
                            row * self.tile_d + col,
                            bf16_zero,
                            full,
                            pack_mode="b32_to_b16",
                        )
                if const_expr(self.return_softmax_lse):
                    one = rr.vdups(1.0, dtypes.float32, mask=full)
                    infinity = rr.vdiv(one, zero, mask=full)
                    rr.vstore(self.softmax_lse_output.values, 0, infinity, full)
                rr.vmem_bar("vst_vld")
            output_ntd = attn_out.permute((1, 0, 2))
            for offset in range(0, count, self.tile_vec_m):
                rows = dynamic_min(Int64(self.tile_vec_m), count - offset)
                target = tile_slice(
                    output_ntd[head, begin + offset: begin + offset + rows, None],
                    (self.tile_vec_m, self.tile_d),
                    (0, 0),
                )  # fmt: skip
                mem_copy(
                    target, self.vector_output.cast_ub_bf16[:rows, : attn_out.shape[2]]
                )
                if const_expr(self.return_softmax_lse):
                    target_lse = tile_slice(
                        softmax_lse_gm.view(
                            (softmax_lse_gm.shape[0] * softmax_lse_gm.shape[1], 1)
                        )[
                            head * attn_out.shape[0] + begin + offset: head
                            * attn_out.shape[0]
                            + begin
                            + offset
                            + rows,
                            None,
                        ],
                        (self.tile_vec_m, 1),
                        (0, 0),
                    )  # fmt: skip
                    mem_copy(target_lse, self.softmax_lse_output.values[:rows, None])

    @jit
    def _store_skipped_outputs(
        self,
        attn_out,
        softmax_lse_gm,
        seqused_q,
        seqused_kv,
        group,
        decode=False,
        force_empty=False,
    ):
        if const_expr(
            decode
            and self.mask_mode == MASK_MODE_NO_MASK
            and not self.return_softmax_lse
        ):
            # Reuse output scratch before attention starts; all stores target
            # omitted token rows, disjoint from live attention/FD writers.
            with vf(mode="simd"):
                full, _ = rr.update_mask(FP32_VECTOR_LANES, elem_bits=FP32_ELEM_BITS)
                zero = rr.vdups(0.0, dtypes.float32, mask=full)
                packed = rr.vcast(zero, dtypes.bfloat16, mask=full)
                for row in range(self.tile_vec_m):
                    for col in range(0, self.tile_d, FP32_VECTOR_LANES):
                        rr.vstore_pack(
                            self.vector_output.cast_ub_bf16,
                            row * self.tile_d + col,
                            packed,
                            full,
                            pack_mode="b32_to_b16",
                        )
                rr.vmem_bar("vst_vld")
            heads = attn_out.shape[1]
            capacity = self.tile_cube_m * heads
            tiles_per_batch = ceil_div(capacity, self.tile_vec_m)
            flat = attn_out.view((attn_out.shape[0] * heads, attn_out.shape[2]))
            worker = self.block_idx * AIV_SUBBLOCK_COUNT + self.subblock_idx
            workers = self.block_num * AIV_SUBBLOCK_COUNT
            for task in range(worker, seqused_q.shape[0] * tiles_per_batch, workers):
                batch = task // tiles_per_batch
                tile = task % tiles_per_batch
                live = seqused_q[batch] * heads
                if seqused_kv[batch] == 0:
                    live = Int64(0)
                if const_expr(force_empty):
                    live = Int64(0)
                begin = dynamic_max(tile * self.tile_vec_m, live)
                end = dynamic_min((tile + 1) * self.tile_vec_m, capacity)
                if end > begin:
                    target = tile_slice(
                        flat[batch * capacity + begin: batch * capacity + end, None],
                        (self.tile_vec_m, self.tile_d),
                        (0, 0),
                    )  # fmt: skip
                    mem_copy(
                        target,
                        self.vector_output.cast_ub_bf16[
                            : end - begin, : attn_out.shape[2]
                        ],
                    )
        else:
            # These head/batch owners are independent of FlashDecode KV splitting.
            # Only omitted Q tiles and padded rows are written, so no barrier with
            # the normal output writers is needed. Reuse pre-pipeline UB storage.
            heads = attn_out.shape[1]
            worker = self.block_idx * AIV_SUBBLOCK_COUNT + self.subblock_idx
            workers = self.block_num * AIV_SUBBLOCK_COUNT
            offset_batch = Int64(0)
            q_prefix = Int64(0)
            for task in range(worker, seqused_q.shape[0] * heads, workers):
                batch, head = task // heads, task % heads
                q_length, kv_length = seqused_q[batch], seqused_kv[batch]
                if const_expr(not decode):
                    while offset_batch < batch:
                        q_prefix += seqused_q[offset_batch]
                        offset_batch += 1
                    begin = q_prefix
                else:
                    begin, _, _ = self._batch_tnd_offsets(batch, seqused_q, seqused_kv)
                capacity = q_length
                if const_expr(decode):
                    capacity = Int64(self.tile_cube_m)
                omitted = Int64(0)
                if const_expr(force_empty):
                    omitted = capacity
                else:
                    if kv_length == 0:
                        omitted = q_length
                    else:
                        if const_expr(self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL):
                            masked_tokens = dynamic_max(Int64(0), q_length - kv_length)
                            masked_rows = masked_tokens * group
                            skipped_rows = (
                                masked_rows // self.tile_cube_m * self.tile_cube_m
                            )
                            omitted = dynamic_max(
                                Int64(0),
                                (skipped_rows + group - 1 - head % group) // group,
                            )
                    omitted = dynamic_min(q_length, omitted)
                self._store_default_range(
                    attn_out, softmax_lse_gm, head, begin, omitted
                )
                if const_expr(decode and not force_empty):
                    self._store_default_range(
                        attn_out,
                        softmax_lse_gm,
                        head,
                        begin + q_length,
                        capacity - q_length,
                    )

    @jit
    def _consume_delayed_pv(
        self,
        attn_out,
        softmax_lse_gm,
        value,
        v_scale_gm,
        block_table,
        seqused_q,
        seqused_kv,
        p_scale,
        head_group_num,
        delay_batch_idx,
        delay_head_idx,
        delay_m_idx,
        delay_n_block_a,
        delay_n_block_b,
        delay_has_second,
        delay_is_first_macro,
        delay_is_last_macro,
        delay_tile_triple,
        delay_m_axis_triple,
        delay_fd_slot=-1,
        delay_q_token_start=None,
        delay_kv_token_start=None,
        delay_v_scale_start=None,
    ):
        # Complete PV and output update for one previously produced macro.
        delay_kv_head_idx = delay_head_idx // head_group_num
        delay_seqlen_k = seqused_kv[delay_batch_idx]
        if const_expr(delay_q_token_start is None):
            delay_q_token_start, delay_kv_token_start, delay_v_scale_start = (
                self._batch_tnd_offsets(delay_batch_idx, seqused_q, seqused_kv)
            )

        accumulator = self.matmul.begin_pv()

        # PV for n_block_a + n_block_b with L0C accumulation
        self._compute_pv_tile(
            self.p_l1_a,
            self.p_scale_l1_a,
            value,
            v_scale_gm,
            block_table,
            delay_batch_idx,
            delay_kv_head_idx,
            delay_kv_token_start,
            delay_v_scale_start,
            delay_seqlen_k,
            delay_n_block_a,
            True,
            accumulator,
        )

        if delay_has_second:
            self._compute_pv_tile(
                self.p_l1_b,
                self.p_scale_l1_b,
                value,
                v_scale_gm,
                block_table,
                delay_batch_idx,
                delay_kv_head_idx,
                delay_kv_token_start,
                delay_v_scale_start,
                delay_seqlen_k,
                delay_n_block_b,
                False,
                accumulator,
            )

        self.matmul.store_pv_result(self.pv_ub)

        # Select the produced PV slot once; output helpers receive a Tensor.
        pv_tile = self.pv_ub.next_read()
        if delay_is_last_macro:
            delay_seqlen_q = seqused_q[delay_batch_idx]
            if delay_is_first_macro:
                self._finalize_outputs(
                    attn_out,
                    softmax_lse_gm,
                    delay_q_token_start,
                    delay_seqlen_q,
                    delay_head_idx,
                    delay_m_idx,
                    delay_m_axis_triple,
                    p_scale,
                    final_pv=pv_tile,
                    kv_length=delay_seqlen_k,
                    fd_slot=delay_fd_slot,
                )
            else:
                if delay_tile_triple == 0:
                    mem_copy(
                        self.vector_output.final_correction_ub,
                        self.vector_softmax.prev_running_max_correction_ub[0],
                    )
                elif delay_tile_triple == 1:
                    mem_copy(
                        self.vector_output.final_correction_ub,
                        self.vector_softmax.prev_running_max_correction_ub[1],
                    )
                else:
                    mem_copy(
                        self.vector_output.final_correction_ub,
                        self.vector_softmax.prev_running_max_correction_ub[2],
                    )
                self._finalize_outputs(
                    attn_out,
                    softmax_lse_gm,
                    delay_q_token_start,
                    delay_seqlen_q,
                    delay_head_idx,
                    delay_m_idx,
                    delay_m_axis_triple,
                    p_scale,
                    final_pv=pv_tile,
                    correction=self.vector_output.final_correction_ub,
                    kv_length=delay_seqlen_k,
                    fd_slot=delay_fd_slot,
                )
        else:
            if delay_is_first_macro:
                self.vector_output.init_output(pv_tile)
            else:
                if delay_tile_triple == 0:
                    self.vector_output.update_output(
                        pv_tile, self.vector_softmax.prev_running_max_correction_ub[0]
                    )
                elif delay_tile_triple == 1:
                    self.vector_output.update_output(
                        pv_tile, self.vector_softmax.prev_running_max_correction_ub[1]
                    )
                else:
                    self.vector_output.update_output(
                        pv_tile, self.vector_softmax.prev_running_max_correction_ub[2]
                    )


# =============================================================================
# Prefill implementation
# =============================================================================


class PrefillCubeMatmul(_CubeMatmulBase):
    """DN-path Cube matmul for K @ Q^T and P @ V.

    QK and PV share two rotating physical slots with role-specific views.
    A PV macro holds one accumulator until both subtiles have completed.
    """

    def __init__(self, tile_m, tile_n, tile_d, regular_p=False):
        """Configure L0 channels and copy engines for the tile geometry."""
        super().__init__(tile_n, tile_d)
        self.regular_p = regular_p

        # Shared rotating slots bound all QK/PV roles to one storage lifetime.
        # D<=256: L0A<=64KiB, L0B<=64KiB, L0C<=256KiB including both slots.
        self.tile_m = tile_m
        self.share_l0_ring_buffers = True
        a_rows = max(tile_n, tile_m * tile_n // tile_d)
        self.l0a = _make_ring_buffer(
            MemLoc.L0A, shape=(a_rows, tile_d), dtype=dtypes.float8_e4m3fn, depth=2
        )
        self.l0b = _make_ring_buffer(
            MemLoc.L0B,
            shape=(max(tile_m, tile_n), tile_d),
            dtype=dtypes.float8_e4m3fn,
            depth=2,
        )
        self.l0c = _make_ring_buffer(
            MemLoc.L0C,
            shape=(max(tile_n, tile_d), tile_m),
            dtype=dtypes.float32,
            depth=2,
        )
        # Bind each role from storage so native NZ pitches are rebuilt.
        self.qk_l0a_spec = _View((tile_n, tile_d), dtypes.float8_e4m3fn, "nz")
        self.qk_l0b_spec = _View((tile_m, tile_d), dtypes.float8_e4m3fn, "nz")
        self.qk_l0c_spec = _View((tile_n, tile_m), dtypes.float32, "nz")
        self.pv_l0a_spec = _View((tile_m, tile_n), dtypes.float8_e4m3fn, "nz")
        self.pv_l0b_spec = _View((tile_n, tile_d), dtypes.float8_e4m3fn, "nz")
        self.pv_l0c_spec = _View((tile_m, tile_d), dtypes.float32, "nz")
        self.l0a_qk = self.l0a
        self.l0b_qk = self.l0b
        self.l0c_qk = self.l0c

        # Copy engines
        self.transposed_qk = True

    def view_p_for_pv(self, p_slot):
        # Rebind before any dynamic page offset: a formatted storage window
        # requires statically known backing allocations and byte origins.
        if self.regular_p:
            return reinterpret(
                p_slot, shape=(p_slot.shape[1], p_slot.shape[0]), data_format="nz"
            )
        return p_slot

    def begin_pv(self):
        return _compact_l0(
            self.pv_l0c_spec.bind(self.l0c.next_write()), (self.tile_m, self.tile_d)
        )

    def store_pv_result(self, pv_ub):
        result = _compact_l0(
            self.pv_l0c_spec.bind(self.l0c.next_read()), (self.tile_m, self.tile_d)
        ).partition(dim=0)
        destination = pv_ub.next_write()
        mem_copy(destination[: result.shape[0], : result.shape[1]], result)

    # QK matmul

    def compute_qk(self, q_l1, k_l1, q_scale_l1, k_scale_l1, actual_m, actual_n=None):
        """Compute K @ Q^T with compact L0 views for the live tile."""
        if actual_n is None:
            actual_n = self.tile_n
        k_view = k_l1.next_read()[:actual_n, : self.tile_d]
        k_scale = k_scale_l1.next_read()[:actual_n, : k_scale_l1.shape[1]]
        a_slot = self.qk_l0a_spec.bind(self.l0a_qk.next_write())
        a = mem_copy(
            _compact_l0(a_slot, (actual_n, self.tile_d)),
            k_view,
            options=copy_options(mx_scale=k_scale),
        )
        q_view = q_l1[:actual_m, : self.tile_d]
        q_scale = q_scale_l1[: q_scale_l1.shape[0], :actual_m]
        b_slot = self.qk_l0b_spec.bind(self.l0b_qk.next_write())
        b = mem_copy(
            _compact_l0(b_slot, (actual_m, self.tile_d)),
            q_view,
            options=copy_options(mx_scale=q_scale),
        )
        result = _compact_l0(
            self.qk_l0c_spec.bind(self.l0c_qk.next_write()), (actual_n, actual_m)
        )
        matmul(result, a, b, init=True, mx_mode=True)

    def store_qk_result(self, qk_ub, actual_m, actual_n=None):
        if actual_n is None:
            actual_n = self.tile_n
        result = _compact_l0(
            self.qk_l0c_spec.bind(self.l0c_qk.next_read()), (actual_n, actual_m)
        ).partition(dim=1)
        mem_copy(qk_ub[: result.shape[0], : result.shape[1]], result)


class PrefillVectorSoftmax(_VectorSoftmaxBase):
    """DN softmax: contiguous vector lanes own Q rows while loops traverse K.

    Exp values feed the FP32 sum and four-K FP8 packing in registers. Packed
    words are written directly into P's padded NZ destination.
    """

    def __init__(
        self, tile_vec_m, tile_n, mask_mode, subblock_idx, p_double_buffer=False
    ):
        """Alias four packed FP8 bytes as one word without allocating UB.

        DN pipeline UB including LSE and final-alpha scratch:
        D64 195616 B, D128/D72 236576 B, D256 243488 B; limit 262144 B.
        P双槽预算：D64 214336 B，D128/D72 255296 B，D256 252992 B。
        """
        self.regular_p = tile_vec_m == 64 and tile_n in (128, 256)
        super().__init__(
            tile_vec_m,
            tile_n,
            mask_mode,
            subblock_idx,
            materialize_nd_p=False,
            regular_p=self.regular_p,
            p_double_buffer=p_double_buffer,
        )
        self.p_word_elements = FP32_ELEM_BITS // INT8_ELEM_BITS
        self.p_words_per_block = FP8_NZ_K_BLOCK // self.p_word_elements
        if self.regular_p and not self.p_double_buffer:
            self.p_bytes = reinterpret(
                self.p_ub, dtype=dtypes.uint8, shape=(self.p_data_bytes,)
            )
            self.p_copy0 = reinterpret(
                self.p_ub,
                shape=(4, tile_n * 8),
                stride=(tile_n * 16 + 64, 1),
                offset=0,
                data_format="nd",
            )
            self.p_copy1 = reinterpret(
                self.p_ub,
                shape=(4, tile_n * 8),
                stride=(tile_n * 16 + 64, 1),
                offset=tile_n * 8 + 32,
                data_format="nd",
            )
        elif not self.regular_p:
            self.p_nz_block_words = (
                self.p_ub.physical_layout.stride[0] // self.p_word_elements
            )
            if not self.p_double_buffer:
                self.p_words_ub = reinterpret(
                    self.p_ub,
                    dtype=dtypes.uint32,
                    shape=(self.p_nz_block_words * (tile_n // FP8_NZ_K_BLOCK),),
                    data_format="nd",
                )

    @jit
    def init_gather_indices(self):
        with vf(mode="simd"):
            mask, _ = rr.update_mask(64, elem_bits=32)
            bytes_mask, _ = rr.update_mask(256, elem_bits=8)
            lanes = rr.varange(0, dtypes.uint32)
            low = rr.vbitwise_and(
                lanes, rr.vdups(15, dtypes.uint32, mask=mask), mask=mask
            )
            base = rr.vadd(
                rr.vshl(low, 4, mask=mask), rr.vshr(lanes, 4, mask=mask), mask=mask
            )
            packed_idx = rr.vadds(
                rr.vmuls(base, 16843009, mask=mask), 201851904, mask=mask
            )
            idx = rr.vreinterpret_lanes(packed_idx, dtypes.uint8)
            rr.vstore(self.gather_indices, 0, idx, bytes_mask)
            rr.vmem_bar("vst_vld")

    @jit
    def _next_p_quant_output_write(self):
        # Select outside VF; vector stores and the later DMA retain the same slot owner.
        if const_expr(self.p_double_buffer):
            slot = self.p_ub.next_write()
            if const_expr(self.regular_p):
                return reinterpret(slot, dtype=dtypes.uint8, shape=(self.p_data_bytes,))
            else:
                return reinterpret(
                    slot,
                    dtype=dtypes.uint32,
                    shape=(self.p_nz_block_words * (self.tile_n // FP8_NZ_K_BLOCK),),
                    data_format="nd",
                )
        else:
            if const_expr(self.regular_p):
                return self.p_bytes
            else:
                return self.p_words_ub

    @jit
    def _dn_tile_max(self, qk_ub, scale, valid_n, tile_valid_n, full_tile=False):
        mask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
        floor = rr.vdups(SOFTMAX_MASKED_SCORE_FLOOR, dtypes.float32, mask=mask)
        for col in range(tile_valid_n, self.tile_n):
            rr.vstore(qk_ub, col * self.tile_vec_m, floor, mask)
        rr.vmem_bar("vst_vld")
        visible = rr.vdups(valid_n, dtypes.int32, mask=mask)
        if const_expr(self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL):
            visible = rr.vadds(rr.varange(0, dtypes.int32), valid_n, mask=mask)
        m0 = floor
        m1 = floor
        m2 = floor
        m3 = floor
        for group in range(self.tile_n // 4):
            c0 = group * 4 + 0
            x0 = rr.vload(qk_ub, c0 * self.tile_vec_m)
            if const_expr(
                not full_tile and self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL
            ):
                x0 = rr.vselect(x0, floor, cond_mask=rr.vgts(visible, c0, mask=mask))
                rr.vstore(qk_ub, c0 * self.tile_vec_m, x0, mask)
            m0 = rr.vmax(m0, x0, mask=mask)
            c1 = group * 4 + 1
            x1 = rr.vload(qk_ub, c1 * self.tile_vec_m)
            if const_expr(
                not full_tile and self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL
            ):
                x1 = rr.vselect(x1, floor, cond_mask=rr.vgts(visible, c1, mask=mask))
                rr.vstore(qk_ub, c1 * self.tile_vec_m, x1, mask)
            m1 = rr.vmax(m1, x1, mask=mask)
            c2 = group * 4 + 2
            x2 = rr.vload(qk_ub, c2 * self.tile_vec_m)
            if const_expr(
                not full_tile and self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL
            ):
                x2 = rr.vselect(x2, floor, cond_mask=rr.vgts(visible, c2, mask=mask))
                rr.vstore(qk_ub, c2 * self.tile_vec_m, x2, mask)
            m2 = rr.vmax(m2, x2, mask=mask)
            c3 = group * 4 + 3
            x3 = rr.vload(qk_ub, c3 * self.tile_vec_m)
            if const_expr(
                not full_tile and self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL
            ):
                x3 = rr.vselect(x3, floor, cond_mask=rr.vgts(visible, c3, mask=mask))
                rr.vstore(qk_ub, c3 * self.tile_vec_m, x3, mask)
            m3 = rr.vmax(m3, x3, mask=mask)
        raw = rr.vmax(rr.vmax(m0, m2, mask=mask), rr.vmax(m1, m3, mask=mask), mask=mask)
        return self._align_maximum_to_log2_scale(rr.vmuls(raw, scale, mask=mask), mask)

    @jit
    def _exp_quant_values(self, qk_ub, maximum, mask, p_output, scale):
        bytes_mask, _ = rr.update_mask(256, elem_bits=8)
        idx = rr.vload(self.gather_indices, 0)
        t0 = rr.vdups(0.0, dtypes.float32, mask=mask)
        t1 = rr.vdups(0.0, dtypes.float32, mask=mask)
        t2 = rr.vdups(0.0, dtypes.float32, mask=mask)
        t3 = rr.vdups(0.0, dtypes.float32, mask=mask)
        t4 = rr.vdups(0.0, dtypes.float32, mask=mask)
        t5 = rr.vdups(0.0, dtypes.float32, mask=mask)
        t6 = rr.vdups(0.0, dtypes.float32, mask=mask)
        t7 = rr.vdups(0.0, dtypes.float32, mask=mask)
        for i in range(self.tile_n // 8):
            c0 = 2 * i + 0 + 0 * (self.tile_n // 4)
            x0 = rr.vmuls(rr.vload(qk_ub, c0 * self.tile_vec_m), scale, mask=mask)
            e0 = rr.vexp_sub(x0, maximum, mask=mask)
            t0 = rr.vadd(t0, e0, mask=mask)
            p0 = rr.vcast(
                e0,
                dtypes.float8_e4m3fn,
                mask=mask,
                reg_layout=rr.RegLayout.ZERO,
                saturate=True,
            )
            w0 = rr.vreinterpret_lanes(p0, dtypes.uint32)
            c1 = 2 * i + 0 + 1 * (self.tile_n // 4)
            x1 = rr.vmuls(rr.vload(qk_ub, c1 * self.tile_vec_m), scale, mask=mask)
            e1 = rr.vexp_sub(x1, maximum, mask=mask)
            t1 = rr.vadd(t1, e1, mask=mask)
            p1 = rr.vcast(
                e1,
                dtypes.float8_e4m3fn,
                mask=mask,
                reg_layout=rr.RegLayout.ONE,
                saturate=True,
            )
            w1 = rr.vreinterpret_lanes(p1, dtypes.uint32)
            c2 = 2 * i + 0 + 2 * (self.tile_n // 4)
            x2 = rr.vmuls(rr.vload(qk_ub, c2 * self.tile_vec_m), scale, mask=mask)
            e2 = rr.vexp_sub(x2, maximum, mask=mask)
            t2 = rr.vadd(t2, e2, mask=mask)
            p2 = rr.vcast(
                e2,
                dtypes.float8_e4m3fn,
                mask=mask,
                reg_layout=rr.RegLayout.TWO,
                saturate=True,
            )
            w2 = rr.vreinterpret_lanes(p2, dtypes.uint32)
            c3 = 2 * i + 0 + 3 * (self.tile_n // 4)
            x3 = rr.vmuls(rr.vload(qk_ub, c3 * self.tile_vec_m), scale, mask=mask)
            e3 = rr.vexp_sub(x3, maximum, mask=mask)
            t3 = rr.vadd(t3, e3, mask=mask)
            p3 = rr.vcast(
                e3,
                dtypes.float8_e4m3fn,
                mask=mask,
                reg_layout=rr.RegLayout.THREE,
                saturate=True,
            )
            w3 = rr.vreinterpret_lanes(p3, dtypes.uint32)
            packed = rr.vbitwise_or(
                rr.vbitwise_or(w0, w1, mask=mask),
                rr.vbitwise_or(w2, w3, mask=mask),
                mask=mask,
            )
            ordered = rr.vgather_reg(rr.vreinterpret_lanes(packed, dtypes.uint8), idx)
            rr.vstore_strided(
                p_output,
                (2 * i + 0) * 32,
                ordered,
                bytes_mask,
                block_stride=self.tile_n // 4 + 1,
                repeat_stride=0,
            )
            c4 = 2 * i + 1 + 0 * (self.tile_n // 4)
            x4 = rr.vmuls(rr.vload(qk_ub, c4 * self.tile_vec_m), scale, mask=mask)
            e4 = rr.vexp_sub(x4, maximum, mask=mask)
            t4 = rr.vadd(t4, e4, mask=mask)
            p4 = rr.vcast(
                e4,
                dtypes.float8_e4m3fn,
                mask=mask,
                reg_layout=rr.RegLayout.ZERO,
                saturate=True,
            )
            w4 = rr.vreinterpret_lanes(p4, dtypes.uint32)
            c5 = 2 * i + 1 + 1 * (self.tile_n // 4)
            x5 = rr.vmuls(rr.vload(qk_ub, c5 * self.tile_vec_m), scale, mask=mask)
            e5 = rr.vexp_sub(x5, maximum, mask=mask)
            t5 = rr.vadd(t5, e5, mask=mask)
            p5 = rr.vcast(
                e5,
                dtypes.float8_e4m3fn,
                mask=mask,
                reg_layout=rr.RegLayout.ONE,
                saturate=True,
            )
            w5 = rr.vreinterpret_lanes(p5, dtypes.uint32)
            c6 = 2 * i + 1 + 2 * (self.tile_n // 4)
            x6 = rr.vmuls(rr.vload(qk_ub, c6 * self.tile_vec_m), scale, mask=mask)
            e6 = rr.vexp_sub(x6, maximum, mask=mask)
            t6 = rr.vadd(t6, e6, mask=mask)
            p6 = rr.vcast(
                e6,
                dtypes.float8_e4m3fn,
                mask=mask,
                reg_layout=rr.RegLayout.TWO,
                saturate=True,
            )
            w6 = rr.vreinterpret_lanes(p6, dtypes.uint32)
            c7 = 2 * i + 1 + 3 * (self.tile_n // 4)
            x7 = rr.vmuls(rr.vload(qk_ub, c7 * self.tile_vec_m), scale, mask=mask)
            e7 = rr.vexp_sub(x7, maximum, mask=mask)
            t7 = rr.vadd(t7, e7, mask=mask)
            p7 = rr.vcast(
                e7,
                dtypes.float8_e4m3fn,
                mask=mask,
                reg_layout=rr.RegLayout.THREE,
                saturate=True,
            )
            w7 = rr.vreinterpret_lanes(p7, dtypes.uint32)
            packed = rr.vbitwise_or(
                rr.vbitwise_or(w4, w5, mask=mask),
                rr.vbitwise_or(w6, w7, mask=mask),
                mask=mask,
            )
            ordered = rr.vgather_reg(rr.vreinterpret_lanes(packed, dtypes.uint8), idx)
            rr.vstore_strided(
                p_output,
                (2 * i + 1) * 32,
                ordered,
                bytes_mask,
                block_stride=self.tile_n // 4 + 1,
                repeat_stride=0,
            )
        even = rr.vadd(
            rr.vadd(t0, t1, mask=mask), rr.vadd(t2, t3, mask=mask), mask=mask
        )
        odd = rr.vadd(rr.vadd(t4, t5, mask=mask), rr.vadd(t6, t7, mask=mask), mask=mask)
        return rr.vadd(even, odd, mask=mask)

    def _encode_correction_values(self, correction):
        """Encode a live row correction without a temporary FP32 UB roundtrip."""
        rows_per_block = P_SCALE_ZN_M0_ROWS
        mask_f32, _ = rr.update_mask(rows_per_block, elem_bits=FP32_ELEM_BITS)
        mask_e8m0, _ = rr.update_mask(
            rows_per_block * MXFP8_SCALE_LANES_PER_GROUP, elem_bits=INT8_ELEM_BITS
        )
        for row_block in range(self.tile_vec_m // rows_per_block):
            indices = rr.varange(row_block * rows_per_block, dtypes.uint32)
            values = rr.vgather_reg(correction, indices)
            pair = _e8m0_exponent_pairs(values, mask_f32)
            self._store_compact_scale_pairs(pair, row_block, mask_e8m0)

    def encode_p_scale(self, use_unit_scale=False):
        if use_unit_scale:
            _VectorSoftmaxBase.encode_unit_p_scale(self)
        # The non-unit correction was encoded in compute_second_subtile_softmax.

    @jit
    def compute_first_subtile_softmax(
        self, running_max_ub, qk_ub, scale, p_scale, valid_n, tile_valid_n, update=True
    ):
        self.compute_first_subtile_softmax_dispatch(
            running_max_ub,
            qk_ub,
            scale,
            p_scale,
            valid_n,
            tile_valid_n,
            update=update,
        )

    @jit
    def compute_first_subtile_softmax_dispatch(
        self,
        running_max_ub,
        qk_ub,
        scale,
        p_scale,
        valid_n,
        tile_valid_n,
        update=True,
    ):
        # valid_n is the first Q lane's visible KV count; later causal lanes see at least as much.
        # Skip masking only when both visibility and the physical KV tail cover the full tile.
        if valid_n >= tile_valid_n:
            self.compute_first_subtile_softmax_impl(
                running_max_ub,
                qk_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                True,
                update=update,
            )
        else:
            self.compute_first_subtile_softmax_impl(
                running_max_ub,
                qk_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                False,
                update=update,
            )

    @jit
    def compute_first_subtile_softmax_impl(
        self,
        running_max_ub,
        qk_ub,
        scale,
        p_scale,
        valid_n,
        tile_valid_n,
        full_tile=False,
        update=True,
    ):
        qk_ub = qk_ub.next_read()
        p_output = self._next_p_quant_output_write()
        with vf(mode="simd"):
            mask, _ = rr.update_mask(self.tile_vec_m, elem_bits=FP32_ELEM_BITS)
            maximum = self._dn_tile_max(qk_ub, scale, valid_n, tile_valid_n, full_tile)
            # Preserve the existing two alignment passes while changing dataflow.
            log_p_scale = rr.vlog(
                rr.vdups(p_scale, dtypes.float32, mask=mask), mask=mask
            )
            maximum = rr.vsub(maximum, log_p_scale, mask=mask)
            if const_expr(update):
                maximum = rr.vmax(maximum, rr.vload(running_max_ub, 0), mask=mask)
            rr.vstore(self.max_first_subtile_ub, 0, maximum, mask)
            rr.vmem_bar("vst_vld")
            total = self._exp_quant_values(qk_ub, maximum, mask, p_output, scale)
            rr.vstore(self.first_subtile_exp_sum_ub, 0, total, mask)
            rr.vmem_bar("vst_vld")

    @jit
    def compute_second_subtile_softmax(
        self,
        qk_ub,
        running_max_ub,
        running_exp_sum_ub,
        prev_running_max_correction_ub,
        scale,
        p_scale,
        valid_n,
        tile_valid_n,
        update=True,
    ):
        self.compute_second_subtile_softmax_dispatch(
            qk_ub,
            running_max_ub,
            running_exp_sum_ub,
            prev_running_max_correction_ub,
            scale,
            p_scale,
            valid_n,
            tile_valid_n,
            update=update,
        )

    @jit
    def compute_second_subtile_softmax_dispatch(
        self,
        qk_ub,
        running_max_ub,
        running_exp_sum_ub,
        prev_running_max_correction_ub,
        scale,
        p_scale,
        valid_n,
        tile_valid_n,
        update=True,
    ):
        if valid_n >= tile_valid_n:
            self.compute_second_subtile_softmax_impl(
                qk_ub,
                running_max_ub,
                running_exp_sum_ub,
                prev_running_max_correction_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                True,
                update=update,
            )
        else:
            self.compute_second_subtile_softmax_impl(
                qk_ub,
                running_max_ub,
                running_exp_sum_ub,
                prev_running_max_correction_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                False,
                update=update,
            )

    @jit
    def compute_second_subtile_softmax_impl(
        self,
        qk_ub,
        running_max_ub,
        running_exp_sum_ub,
        prev_running_max_correction_ub,
        scale,
        p_scale,
        valid_n,
        tile_valid_n,
        full_tile=False,
        update=True,
    ):
        qk_ub = qk_ub.next_read()
        p_output = self._next_p_quant_output_write()
        with vf(mode="simd"):
            mask, _ = rr.update_mask(self.tile_vec_m, elem_bits=FP32_ELEM_BITS)
            second_max = self._dn_tile_max(
                qk_ub, scale, valid_n, tile_valid_n, full_tile
            )
            log_p_scale = rr.vlog(
                rr.vdups(p_scale, dtypes.float32, mask=mask), mask=mask
            )
            second_max = rr.vsub(second_max, log_p_scale, mask=mask)
            first_max = rr.vload(self.max_first_subtile_ub, 0)
            new_max = rr.vmax(first_max, second_max, mask=mask)
            correction = rr.vexp_sub(first_max, new_max, mask=mask)
            # These two stores cross iterations/stages and remain necessary.
            if const_expr(update):
                old_max = rr.vload(running_max_ub, 0)
                old_correction = rr.vexp_sub(old_max, new_max, mask=mask)
                rr.vstore(prev_running_max_correction_ub, 0, old_correction, mask)
            rr.vstore(running_max_ub, 0, new_max, mask)
            rr.vmem_bar("vst_vld")
            second_sum = self._exp_quant_values(qk_ub, new_max, mask, p_output, scale)
            first_sum = rr.vload(self.first_subtile_exp_sum_ub, 0)
            if const_expr(update):
                old_sum = rr.vload(running_exp_sum_ub, 0)
                new_sum = _merge_softmax_sum(
                    old_sum, old_correction, first_sum, correction, second_sum, mask
                )
            else:
                new_sum = rr.vadd(
                    rr.vmul(first_sum, correction, mask=mask), second_sum, mask=mask
                )
            rr.vstore(running_exp_sum_ub, 0, new_sum, mask)
            self._encode_correction_values(correction)
            rr.vmem_bar("vst_vld")

    @jit
    def compute_single_subtile_softmax(
        self,
        qk_ub,
        running_max_ub,
        running_exp_sum_ub,
        prev_running_max_correction_ub,
        scale,
        p_scale,
        valid_n,
        tile_valid_n,
        update=True,
    ):
        self.compute_single_subtile_softmax_dispatch(
            qk_ub,
            running_max_ub,
            running_exp_sum_ub,
            prev_running_max_correction_ub,
            scale,
            p_scale,
            valid_n,
            tile_valid_n,
            update=update,
        )

    @jit
    def compute_single_subtile_softmax_dispatch(
        self,
        qk_ub,
        running_max_ub,
        running_exp_sum_ub,
        prev_running_max_correction_ub,
        scale,
        p_scale,
        valid_n,
        tile_valid_n,
        update=True,
    ):
        if valid_n >= tile_valid_n:
            self.compute_single_subtile_softmax_impl(
                qk_ub,
                running_max_ub,
                running_exp_sum_ub,
                prev_running_max_correction_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                True,
                update=update,
            )
        else:
            self.compute_single_subtile_softmax_impl(
                qk_ub,
                running_max_ub,
                running_exp_sum_ub,
                prev_running_max_correction_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                False,
                update=update,
            )

    @jit
    def compute_single_subtile_softmax_impl(
        self,
        qk_ub,
        running_max_ub,
        running_exp_sum_ub,
        prev_running_max_correction_ub,
        scale,
        p_scale,
        valid_n,
        tile_valid_n,
        full_tile=False,
        update=True,
    ):
        qk_ub = qk_ub.next_read()
        p_output = self._next_p_quant_output_write()
        with vf(mode="simd"):
            mask, _ = rr.update_mask(self.tile_vec_m, elem_bits=FP32_ELEM_BITS)
            current_max = self._dn_tile_max(
                qk_ub, scale, valid_n, tile_valid_n, full_tile
            )
            log_p_scale = rr.vlog(
                rr.vdups(p_scale, dtypes.float32, mask=mask), mask=mask
            )
            current_max = rr.vsub(current_max, log_p_scale, mask=mask)
            if const_expr(update):
                old_max = rr.vload(running_max_ub, 0)
                new_max = rr.vmax(old_max, current_max, mask=mask)
                correction = rr.vexp_sub(old_max, new_max, mask=mask)
                rr.vstore(prev_running_max_correction_ub, 0, correction, mask)
            else:
                new_max = current_max
            rr.vstore(running_max_ub, 0, new_max, mask)
            rr.vmem_bar("vst_vld")
            current_sum = self._exp_quant_values(qk_ub, new_max, mask, p_output, scale)
            if const_expr(update):
                old_sum = rr.vload(running_exp_sum_ub, 0)
                new_sum = rr.vadd(
                    rr.vmul(old_sum, correction, mask=mask), current_sum, mask=mask
                )
            else:
                new_sum = current_sum
            rr.vstore(running_exp_sum_ub, 0, new_sum, mask)
            rr.vmem_bar("vst_vld")

    def store_quantized_p(self, p_l1, copy_n):
        if const_expr(self.regular_p and self.tile_n == 256):
            slot = reinterpret(p_l1.next_write(), shape=(4, 8192), data_format="nd")
            dst0 = slot[self.subblock_idx * 2, None].flatten().unflatten(0, (4, 2048))
            dst1 = (
                slot[self.subblock_idx * 2 + 1, None].flatten().unflatten(0, (4, 2048))
            )
            source = self.p_ub.next_read() if self.p_double_buffer else self.p_ub
            p_copy0 = reinterpret(
                source, shape=(4, 2048), stride=(4160, 1), offset=0, data_format="nd"
            )
            p_copy1 = reinterpret(
                source, shape=(4, 2048), stride=(4160, 1), offset=2080, data_format="nd"
            )
            mem_copy(dst0, p_copy0)
            mem_copy(dst1, p_copy1)
        elif const_expr(self.regular_p):
            slot = reinterpret(p_l1.next_write(), shape=(4, 4096), data_format="nd")
            # ZN L1: [Q/32, K, Q%32]. Each AIV owns two 32-row blocks.
            dst0 = slot[self.subblock_idx * 2, None].flatten().unflatten(0, (4, 1024))
            dst1 = (
                slot[self.subblock_idx * 2 + 1, None].flatten().unflatten(0, (4, 1024))
            )
            if self.p_double_buffer:
                source = self.p_ub.next_read()
                p_copy0 = reinterpret(
                    source,
                    shape=(4, self.tile_n * 8),
                    stride=(self.tile_n * 16 + 64, 1),
                    offset=0,
                    data_format="nd",
                )
                p_copy1 = reinterpret(
                    source,
                    shape=(4, self.tile_n * 8),
                    stride=(self.tile_n * 16 + 64, 1),
                    offset=self.tile_n * 8 + 32,
                    data_format="nd",
                )
            else:
                p_copy0, p_copy1 = self.p_copy0, self.p_copy1
            mem_copy(dst0, p_copy0)
            mem_copy(dst1, p_copy1)
        else:
            aligned_n = ceil_div(copy_n, P_TILE_N_ALIGNMENT) * P_TILE_N_ALIGNMENT
            target = p_l1.next_write()[: self.tile_m, :aligned_n].partition(
                dim=0, part_id=self.subblock_idx
            )
            p_ub = self.p_ub.next_read() if self.p_double_buffer else self.p_ub
            mem_copy(target, p_ub[: target.shape[0], : target.shape[1]])


class PrefillVectorOutput(_VectorOutputBase):
    """Store prefill output with aligned split-M tail ownership."""

    @jit
    def update_output(self, pv_ub, prev_running_max_correction_ub):
        """Unroll fixed-width Prefill chunks without changing multiply/add rounding."""
        # D256每行的四个向量块使用同一满掩码，保持逐元素先乘后加。
        if const_expr(self.tile_d == 256):
            rows = pv_ub.shape[0]
            with vf(mode="simd"):
                mask, _ = rr.update_mask(FP32_VECTOR_LANES, elem_bits=FP32_ELEM_BITS)
                for row in range(rows):
                    base = row * self.tile_d
                    alpha = rr.vload_broadcast(prev_running_max_correction_ub, row)
                    for block in range_constexpr(4):
                        offset = base + block * FP32_VECTOR_LANES
                        old = rr.vload(self.res_o_ub, offset)
                        current = rr.vload(pv_ub, offset)
                        updated = rr.vadd(
                            rr.vmul(old, alpha, mask=mask), current, mask=mask
                        )
                        rr.vstore(self.res_o_ub, offset, updated, mask)
        else:
            _VectorOutputBase.update_output(self, pv_ub, prev_running_max_correction_ub)

    @jit
    def finalize_output(
        self,
        o_tile_gm: Tensor,
        softmax_sum_ub,
        final_pv=None,
        correction=None,
        maximum_ub=None,
        check_block=False,
    ):
        """Merge optional final PV, normalize and store BF16 output."""
        actual_m = o_tile_gm.shape[0]
        aligned_m = (
            ceil_div(actual_m, SPLIT_M_TOTAL_ALIGNMENT) * SPLIT_M_TOTAL_ALIGNMENT
        )
        first_part_m = (
            actual_m
            if actual_m <= SPLIT_M_ALIGNMENT
            else aligned_m // AIV_SUBBLOCK_COUNT
        )
        actual_vec_m = (
            first_part_m if self.subblock_idx == 0 else actual_m - first_part_m
        )
        if const_expr(final_pv is not None and correction is None):
            if const_expr(self.check_invalid_rows):
                if check_block:
                    self.normalize_single_pv(
                        final_pv, softmax_sum_ub, maximum_ub, actual_vec_m, True
                    )
                else:
                    self.normalize_single_pv(
                        final_pv, softmax_sum_ub, maximum_ub, actual_vec_m, False
                    )
            else:
                self.normalize_single_pv(
                    final_pv, softmax_sum_ub, maximum_ub, actual_vec_m, False
                )
        else:
            self._normalize_rows(
                softmax_sum_ub,
                actual_vec_m,
                final_pv,
                correction,
                maximum_ub,
                check_block,
            )

        target = o_tile_gm[:actual_m, : o_tile_gm.shape[1]].partition(
            dim=0, alignment=SPLIT_M_ALIGNMENT, part_id=self.subblock_idx
        )
        mem_copy(target, self.cast_ub_bf16[: target.shape[0], : target.shape[1]])


# ---- Prefill kernel orchestration -------------------------------------------
@kernel
class FlashAttentionGQAMxfp8PrefillKernel(_DelayedPVConsumer):
    """DN-path Flash Attention kernel with delayed PV consumption.

    QK/softmax production runs two macro tiles ahead of PV consumption;
    the delay line carries metadata across the three pipeline slots.
    """

    def __init__(
        self,
        tile_cube_m,
        tile_vec_m,
        tile_n,
        tile_d,
        mask_mode,
        kv_layout,
        block_num,
        q_layout=QLayout.TND,
        paged=False,
        block_size=128,
        return_softmax_lse=False,
        enable_flash_decode=False,
    ):
        self.enable_flash_decode = enable_flash_decode
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.mask_mode = mask_mode
        self.q_layout = q_layout
        # Current kernel indexing is TND-only: Q is [T, N_q, D], and all
        # token/head offsets and scale indexing rely on that contract.
        self.kv_layout = kv_layout
        self.paged = self.kv_layout != KVLayout.TND
        self.pa_bbnd = self.kv_layout == KVLayout.PA_BBND
        self.pa_nz = self.kv_layout == KVLayout.PA_NZ
        self.block_size = block_size
        self.return_softmax_lse = return_softmax_lse
        self.page_tile_n = min(tile_n, block_size)
        self.tiles_per_page = max(1, block_size // tile_n)
        self.pages_per_tile = (tile_n + block_size - 1) // block_size
        check_pa_tile(block_size, tile_n, self.paged)
        self.preload_num = SOFTMAX_DELAY_SLOTS

        # Separate QK slots feed the two softmax subtiles; PV has one slot.
        self.qk_ub_a = _make_ring_buffer(
            MemLoc.UB, shape=(tile_n, tile_vec_m), dtype=dtypes.float32, depth=1
        )
        self.qk_ub_b = _make_ring_buffer(
            MemLoc.UB, shape=(tile_n, tile_vec_m), dtype=dtypes.float32, depth=1
        )
        self.pv_ub = _make_ring_buffer(
            MemLoc.UB, shape=(tile_vec_m, tile_d), dtype=dtypes.float32, depth=1
        )

        # P tiles are triple-buffered across the delayed PV pipeline.
        self.p_l1_a = _make_ring_buffer(
            MemLoc.L1,
            shape=(tile_cube_m, tile_n),
            dtype=dtypes.float8_e4m3fn,
            depth=SOFTMAX_DELAY_SLOTS,
            data_format="zn" if tile_vec_m == 64 and tile_n in (128, 256) else "nz",
        )
        self.p_l1_b = _make_ring_buffer(
            MemLoc.L1,
            shape=(tile_cube_m, tile_n),
            dtype=dtypes.float8_e4m3fn,
            depth=SOFTMAX_DELAY_SLOTS,
            data_format="zn" if tile_vec_m == 64 and tile_n in (128, 256) else "nz",
        )

        # Q has two slots; K and V share one four-slot pool in execution order.
        self.q_l1 = _make_ring_buffer(
            MemLoc.L1, shape=(tile_cube_m, tile_d), dtype=dtypes.float8_e4m3fn, depth=2
        )
        self.kv_l1 = _make_ring_buffer(
            MemLoc.L1, (self.tile_n, tile_d), dtypes.float8_e4m3fn, depth=4
        )
        self.k_l1 = _SharedL1View(
            self.kv_l1, _View((self.tile_n, tile_d), dtypes.float8_e4m3fn, "nz")
        )
        self.v_l1 = _SharedL1View(
            self.kv_l1,
            _View(
                (self.tile_n, tile_d),
                dtypes.float8_e4m3fn,
                "nz",
            ),
        )

        # E8M0 scales use MXFP8_SCALE_LANES_PER_GROUP lanes per MXFP8_SCALE_GROUP_ELEMENTS-element reduction group.
        self.qk_scale_groups = _mx_groups64(tile_d)  # MX groups for Q/K reduce (D axis)
        self.v_scale_groups = _mx_groups64(self.tile_n)  # groups for one Cube K chunk
        self.p_scale_groups = _mx_groups64(
            tile_n
        )  # MX groups for P reduce (tile_n axis)
        self.q_scale_l1 = _make_ring_buffer(
            MemLoc.L1,
            shape=(2 * self.qk_scale_groups, tile_cube_m),
            dtype=dtypes.float8_e8m0,
            depth=2,
            data_format="nz",
        )
        k_scale_spec = _View(
            (self.tile_n, 2 * self.qk_scale_groups), dtypes.float8_e8m0, "zn"
        )
        v_scale_spec = _View(
            (2 * self.v_scale_groups, tile_d), dtypes.float8_e8m0, "nz"
        )
        scale_bytes = max(
            k_scale_spec.storage_bytes(mem_loc=MemLoc.L1),
            v_scale_spec.storage_bytes(mem_loc=MemLoc.L1),
        )
        self.kv_scale_l1 = _make_ring_buffer(MemLoc.L1, size_bytes=scale_bytes, depth=4)
        self.k_scale_l1 = _SharedL1View(self.kv_scale_l1, k_scale_spec)
        self.v_scale_l1 = _SharedL1View(self.kv_scale_l1, v_scale_spec)
        # P and correction scales retain matching slots until delayed PV reads them.
        self.p_scale_l1_a = _make_ring_buffer(
            MemLoc.L1,
            shape=(tile_cube_m, 2 * self.p_scale_groups),
            dtype=dtypes.float8_e8m0,
            depth=SOFTMAX_DELAY_SLOTS,
            data_format="zn",
        )
        self.p_scale_l1_b = _make_ring_buffer(
            MemLoc.L1,
            shape=(tile_cube_m, 2 * self.p_scale_groups),
            dtype=dtypes.float8_e8m0,
            depth=SOFTMAX_DELAY_SLOTS,
            data_format="zn",
        )

        self.block_idx = get_block_idx()
        self.subblock_idx = get_subblock_id()
        self.block_num = Int64(block_num)

        # P representation follows the DN softmax tile rather than the physical KV page.
        self.matmul = PrefillCubeMatmul(
            self.tile_cube_m,
            self.tile_n,
            self.tile_d,
            regular_p=self.tile_vec_m == 64 and self.tile_n in (128, 256),
        )
        self.matmul.raw_pa_v = self.pa_nz
        self.vector_softmax = PrefillVectorSoftmax(
            self.tile_vec_m,
            self.tile_n,
            self.mask_mode,
            self.subblock_idx,
            p_double_buffer=True,
        )
        self.vector_output = PrefillVectorOutput(
            self.tile_vec_m, self.tile_d, self.subblock_idx
        )
        if return_softmax_lse:
            self.softmax_lse_output = SoftmaxLseOutput(self.tile_vec_m)
        self.block_info = BlockInfo(self.tile_cube_m, self.tile_n, self.mask_mode)
        if enable_flash_decode:
            # FD producer adds 2*64*8*4=4096 bytes, no L1/L0 allocations.
            # Merge reuses dead FA arenas after all-core producer completion.
            self.fd_sum_pack = _make_buffer(
                MemLoc.UB, (tile_vec_m, FD_STAT_ROW_ELEMENTS), dtypes.float32
            )
            self.fd_max_pack = _make_buffer(
                MemLoc.UB, (tile_vec_m, FD_STAT_ROW_ELEMENTS), dtypes.float32
            )

    # ---- Tensor views and address helpers -------------------------------------

    @jit
    def _batch_tnd_offsets(self, batch_idx, seqused_q, seqused_kv):
        """Return compact TND offsets for Q/KV tokens and V-scale groups."""
        q_token_start = Int64(0)
        kv_token_start = Int64(0)
        v_scale_start = Int64(0)
        for previous_batch in range(0, batch_idx):
            q_token_start += seqused_q[previous_batch]
            previous_kv_length = seqused_kv[previous_batch]
            kv_token_start += previous_kv_length
            v_scale_start += ceil_div(previous_kv_length, MXFP8_SCALE_GROUP_ELEMENTS)
        return q_token_start, kv_token_start, v_scale_start

    @jit
    def _query_tnd_tile(self, query, q_token_start, q_length, head_idx, m_idx):
        query_ntd = query.permute((1, 0, 2))
        query_head = query_ntd[head_idx, q_token_start: q_token_start + q_length, None]  # fmt: skip
        return tile_slice(query_head, (self.tile_cube_m, query.shape[-1]), (m_idx, 0))

    @jit
    def _query_scale_tnd_tile(
        self, q_scale_gm, q_token_start, q_length, head_idx, m_idx
    ):
        scale_ntg2 = q_scale_gm.permute((1, 0, 2, 3))
        scale_head = scale_ntg2[
            head_idx, q_token_start: q_token_start + q_length, None, None
        ]  # fmt: skip
        return tile_slice(
            scale_head, (self.tile_cube_m, self.qk_scale_groups, 2), (m_idx, 0, 0)
        )

    # ---- GM-to-L1 staging -----------------------------------------------------

    # ---- QK/PV tile compute ---------------------------------------------------

    @jit
    def _compute_pv_tile(
        self,
        p_l1,
        p_scale_l1,
        value,
        v_scale_gm,
        block_table,
        batch_idx,
        kv_head_idx,
        kv_token_start,
        v_scale_start,
        kv_length,
        n_block,
        init,
        accumulator=None,
    ):
        self._load_value_and_scale_tile(
            value,
            v_scale_gm,
            block_table,
            batch_idx,
            kv_head_idx,
            kv_token_start,
            v_scale_start,
            kv_length,
            n_block,
        )
        # PV 的 MX 归约维按 64 对齐；页内 scale 原点保持不变。
        valid_k = dynamic_min(
            Int64(self.tile_n),
            kv_length - n_block * self.tile_n,
        )
        aligned_k = (
            ceil_div(valid_k, MXFP8_SCALE_GROUP_ELEMENTS) * MXFP8_SCALE_GROUP_ELEMENTS
        )
        p_slot = self.matmul.view_p_for_pv(p_l1.next_read())
        p_scale_slot = p_scale_l1.next_read()
        self.matmul.compute_pv(
            p_slot,
            self.v_l1.next_read(),
            p_scale_slot,
            self.v_scale_l1.next_read(),
            init=init,
            actual_k=aligned_k,
            accumulator=accumulator,
        )

    # ---- Output views and finalization ---------------------------------------

    @jit
    def _output_tnd_tile(self, attn_out, q_token_start, q_length, head_idx, m_idx):
        output_ntd = attn_out.permute((1, 0, 2))
        output_head = output_ntd[
            head_idx, q_token_start: q_token_start + q_length, None
        ]  # fmt: skip
        return tile_slice(output_head, (self.tile_cube_m, self.tile_d), (m_idx, 0))

    @jit
    def _finalize_whole_outputs(
        self,
        attn_out,
        softmax_lse_gm,
        q_token_start,
        q_length,
        head_idx,
        m_idx,
        state,
        p_scale,
        final_pv=None,
        correction=None,
        kv_length=0,
    ):
        o_tile_gm = self._output_tnd_tile(
            attn_out, q_token_start, q_length, head_idx, m_idx
        )
        if const_expr(self.return_softmax_lse):
            actual_m = o_tile_gm.shape[0]
            aligned_m = (
                ceil_div(actual_m, SPLIT_M_TOTAL_ALIGNMENT) * SPLIT_M_TOTAL_ALIGNMENT
            )
            first_part_m = (
                actual_m
                if actual_m <= SPLIT_M_ALIGNMENT
                else aligned_m // AIV_SUBBLOCK_COUNT
            )
            local_row_offset = 0 if self.subblock_idx == 0 else first_part_m
            actual_vec_m = (
                first_part_m if self.subblock_idx == 0 else actual_m - first_part_m
            )
            softmax_lse_flat = softmax_lse_gm.view(
                (
                    softmax_lse_gm.shape[0] * softmax_lse_gm.shape[1],
                    1,
                )
            )
            lse_row_base = (
                head_idx * softmax_lse_gm.shape[1]
                + q_token_start
                + m_idx * self.tile_cube_m
                + local_row_offset
            )
        if state == 0:
            self.vector_output.finalize_output(
                o_tile_gm,
                self.vector_softmax.softmax_sum_ub[0],
                final_pv,
                correction,
                self.vector_softmax.softmax_max_ub[0],
                m_idx * self.tile_cube_m < q_length - kv_length,
            )
            if const_expr(self.return_softmax_lse):
                self.softmax_lse_output.store(
                    self.vector_softmax.softmax_max_ub[0],
                    self.vector_softmax.softmax_sum_ub[0],
                    p_scale,
                    softmax_lse_flat,
                    lse_row_base,
                    actual_vec_m,
                )
        elif state == 1:
            self.vector_output.finalize_output(
                o_tile_gm,
                self.vector_softmax.softmax_sum_ub[1],
                final_pv,
                correction,
                self.vector_softmax.softmax_max_ub[1],
                m_idx * self.tile_cube_m < q_length - kv_length,
            )
            if const_expr(self.return_softmax_lse):
                self.softmax_lse_output.store(
                    self.vector_softmax.softmax_max_ub[1],
                    self.vector_softmax.softmax_sum_ub[1],
                    p_scale,
                    softmax_lse_flat,
                    lse_row_base,
                    actual_vec_m,
                )
        else:
            self.vector_output.finalize_output(
                o_tile_gm,
                self.vector_softmax.softmax_sum_ub[2],
                final_pv,
                correction,
                self.vector_softmax.softmax_max_ub[2],
                m_idx * self.tile_cube_m < q_length - kv_length,
            )
            if const_expr(self.return_softmax_lse):
                self.softmax_lse_output.store(
                    self.vector_softmax.softmax_max_ub[2],
                    self.vector_softmax.softmax_sum_ub[2],
                    p_scale,
                    softmax_lse_flat,
                    lse_row_base,
                    actual_vec_m,
                )

    @jit
    def _finalize_outputs(
        self,
        attn_out,
        softmax_lse_gm,
        q_token_start,
        q_length,
        head_idx,
        m_idx,
        state,
        p_scale,
        final_pv=None,
        correction=None,
        fd_slot=-1,
        kv_length=0,
    ):
        if const_expr(self.enable_flash_decode):
            if fd_slot >= 0:
                actual_m = dynamic_min(
                    Int64(self.tile_cube_m), q_length - m_idx * self.tile_cube_m
                )
                first = (
                    actual_m
                    if actual_m <= SPLIT_M_ALIGNMENT
                    else ceil_div(actual_m, SPLIT_M_TOTAL_ALIGNMENT) * SPLIT_M_ALIGNMENT
                )
                offset = self.subblock_idx * first
                rows = dynamic_max(Int64(0), dynamic_min(first, actual_m - offset))
                if state == 0:
                    self._save_fd_partial(
                        fd_slot,
                        offset,
                        rows,
                        self.vector_softmax.softmax_max_ub[0],
                        self.vector_softmax.softmax_sum_ub[0],
                        final_pv,
                        correction,
                    )
                elif state == 1:
                    self._save_fd_partial(
                        fd_slot,
                        offset,
                        rows,
                        self.vector_softmax.softmax_max_ub[1],
                        self.vector_softmax.softmax_sum_ub[1],
                        final_pv,
                        correction,
                    )
                else:
                    self._save_fd_partial(
                        fd_slot,
                        offset,
                        rows,
                        self.vector_softmax.softmax_max_ub[2],
                        self.vector_softmax.softmax_sum_ub[2],
                        final_pv,
                        correction,
                    )
            else:
                self._finalize_whole_outputs(
                    attn_out,
                    softmax_lse_gm,
                    q_token_start,
                    q_length,
                    head_idx,
                    m_idx,
                    state,
                    p_scale,
                    final_pv,
                    correction,
                    kv_length,
                )
        else:
            self._finalize_whole_outputs(
                attn_out,
                softmax_lse_gm,
                q_token_start,
                q_length,
                head_idx,
                m_idx,
                state,
                p_scale,
                final_pv,
                correction,
                kv_length,
            )

    @jit
    def _setup_fd_merge(self):
        # D256 maximum: output 65792 + double input 131072 + stats 8192
        # + running stats 512 + optional LSE 256 = 205824 bytes < 256KiB.
        self.vector_output = PrefillVectorOutput(
            self.tile_vec_m, self.tile_d, self.subblock_idx
        )
        self.fd_part_input = _make_ring_buffer(
            MemLoc.UB, (self.tile_vec_m, self.tile_d), dtypes.float32, depth=2
        )
        self.fd_sum_input = _make_ring_buffer(
            MemLoc.UB, (self.tile_vec_m, FD_STAT_ROW_ELEMENTS), dtypes.float32, depth=2
        )
        self.fd_max_input = _make_ring_buffer(
            MemLoc.UB, (self.tile_vec_m, FD_STAT_ROW_ELEMENTS), dtypes.float32, depth=2
        )
        self.fd_running_max = _make_buffer(
            MemLoc.UB, (self.tile_vec_m,), dtypes.float32
        )
        self.fd_running_sum = _make_buffer(
            MemLoc.UB, (self.tile_vec_m,), dtypes.float32
        )
        if const_expr(self.return_softmax_lse):
            self.softmax_lse_output = SoftmaxLseOutput(self.tile_vec_m)

    @jit
    def _merge_fd_task(
        self,
        attn_out,
        softmax_lse_gm,
        seqused_q,
        seqused_kv,
        heads,
        group,
        metadata,
        record,
        slot_base,
        p_scale,
    ):
        row_count = metadata[record + FD_ROWS]
        if row_count > 0:
            bn = metadata[record + FD_BN]
            batch, head = bn // heads, bn % heads
            m = metadata[record + FD_M]
            begin = metadata[record + FD_ROW_BEGIN]
            first = slot_base + metadata[record + FD_FIRST_SLOT]
            parts = metadata[record + FD_PARTS]
            token_start, _, _ = self._batch_tnd_offsets(batch, seqused_q, seqused_kv)
            for chunk in range(ceil_div(row_count, self.tile_vec_m)):
                offset = chunk * self.tile_vec_m
                rows = dynamic_min(Int64(self.tile_vec_m), row_count - offset)
                row_start = m * self.tile_cube_m + begin + offset
                self._init_fd_accumulator(rows)
                self._load_fd_part(first, begin + offset, rows)
                for part in range(parts):
                    if part + 1 < parts:
                        self._load_fd_part(first + part + 1, begin + offset, rows)
                    self._update_fd_accumulator(rows)
                self.vector_output._normalize_rows(
                    self.fd_running_sum,
                    rows,
                    maximum_ub=self.fd_running_max,
                    check_block=row_start
                    < (seqused_q[batch] - seqused_kv[batch]) * group,
                )
                target = tile_slice(
                    attn_out.permute((1, 0, 2))[
                        head,
                        token_start + row_start: token_start + row_start + rows,
                        None,
                    ],
                    (self.tile_vec_m, self.tile_d),
                    (0, 0),
                )  # fmt: skip
                mem_copy(
                    target, self.vector_output.cast_ub_bf16[:rows, : attn_out.shape[2]]
                )
                if const_expr(self.return_softmax_lse):
                    flat = softmax_lse_gm.view(
                        (softmax_lse_gm.shape[0] * softmax_lse_gm.shape[1], 1)
                    )
                    self.softmax_lse_output.store(
                        self.fd_running_max,
                        self.fd_running_sum,
                        p_scale,
                        flat,
                        head * softmax_lse_gm.shape[1] + token_start + row_start,
                        rows,
                    )

    # ---- Kernel entry point and pipeline schedule -----------------------------

    def __call__(
        self,
        attn_out: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_scale_gm: Tensor,
        k_scale_gm: Tensor,
        v_scale_gm: Tensor,
        p_scale_gm: Tensor,
        scale: dtypes.float32,
        block_table: Tensor,
        seqused_kv: Tensor,
        seqused_q: Tensor,
        max_seqlen_q: dtypes.int64,
        softmax_lse_gm: Tensor | None = None,
        fd_output_gm: Tensor | None = None,
        fd_sum_gm: Tensor | None = None,
        fd_max_gm: Tensor | None = None,
        fd_metadata_gm: Tensor | None = None,
    ):
        self.copy_v_d = attn_out.shape[2]
        self.fd_output_gm = fd_output_gm
        self.fd_sum_gm = fd_sum_gm
        self.fd_max_gm = fd_max_gm
        self.vector_softmax.init_gather_indices()
        macro_global_idx = Int64(0)

        # The p quantization scale is a GM scalar.
        p_scale = p_scale_gm[0]
        self.vector_output.init_invalid_row_check(
            self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL, True, scale, p_scale
        )
        head_num_q = Int64(query.shape[1])
        if const_expr(self.pa_bbnd):
            head_num_kv = Int64(key.shape[2])
        else:
            head_num_kv = Int64(key.shape[1])
        head_group_num = head_num_q // head_num_kv
        seqlen_q_max = Int64(max_seqlen_q)
        self._store_skipped_outputs(
            attn_out, softmax_lse_gm, seqused_q, seqused_kv, Int64(1), force_empty=False
        )

        delayed_metadata = DelayLineGroup(
            self.preload_num,
            "batch_idx",
            "head_idx",
            "m_idx",
            "n_block_a",
            "n_block_b",
            "has_second",
            "is_first_macro",
            "is_last_macro",
            "tile_triple",
            "m_axis_triple",
            "fd_slot",
            "q_token_start",
            "kv_token_start",
            "v_scale_start",
        )

        sections = fd_metadata_gm[SECTION_COUNT]
        fd_active = fd_metadata_gm[HAS_FD]
        m_tiles_per_head = ceil_div(seqlen_q_max * 1, self.tile_cube_m)
        active_m_tile_idx = Int64(0)
        offset_batch = Int64(0)
        q_token_start = Int64(0)
        kv_token_start = Int64(0)
        v_scale_start = Int64(0)

        for section in range(sections):
            record = (1 + section * self.block_num + self.block_idx) * METADATA_STRIDE
            start_tile_idx = (
                fd_metadata_gm[record + FA_START_BN] * m_tiles_per_head
                + fd_metadata_gm[record + FA_START_M]
            )
            end_boundary_tile = (
                fd_metadata_gm[record + FA_END_BN] * m_tiles_per_head
                + fd_metadata_gm[record + FA_END_M]
            )
            first_macro = fd_metadata_gm[record + FA_START_S2]
            last_macro = fd_metadata_gm[record + FA_END_S2]
            first_slot = (
                section * 2 * self.block_num + fd_metadata_gm[record + FA_FIRST_SLOT]
            )
            end_tile_idx = end_boundary_tile
            if last_macro > 0:
                end_tile_idx += 1
            bn_stride = dynamic_max(Int64(1), m_tiles_per_head)
            for bn in range(
                start_tile_idx // bn_stride, ceil_div(end_tile_idx, bn_stride)
            ):
                batch_idx, head_idx = bn // head_num_q, bn % head_num_q
                seqlen_q_actual = seqused_q[batch_idx]
                base_tile = bn * m_tiles_per_head
                first_m = dynamic_max(Int64(0), start_tile_idx - base_tile)
                stop_m = dynamic_min(
                    ceil_div(seqlen_q_actual, self.tile_cube_m),
                    end_tile_idx - base_tile,
                )
                for m_idx in range(first_m, stop_m):
                    tile_idx = base_tile + m_idx
                    seqlen_k = seqused_kv[batch_idx]
                    # Contiguous tile ranges visit batches monotonically on each core.
                    while offset_batch < batch_idx:
                        q_token_start += seqused_q[offset_batch]
                        if const_expr(not self.paged):
                            kv_length_before = seqused_kv[offset_batch]
                            kv_token_start += kv_length_before
                            v_scale_start += ceil_div(
                                kv_length_before, MXFP8_SCALE_GROUP_ELEMENTS
                            )
                        offset_batch += 1
                    n_idx_start, n_idx_end = self.block_info.get_kv_tile_range(
                        seqlen_q_actual, seqlen_k, m_idx
                    )
                    macro_end = ceil_div(n_idx_end - n_idx_start, 2)
                    fd_slot = Int64(-1)
                    if const_expr(self.enable_flash_decode):
                        if fd_active:
                            macro_start = Int64(0)
                            macro_stop = macro_end
                            if tile_idx == start_tile_idx:
                                macro_start = first_macro
                            if tile_idx == end_boundary_tile:
                                macro_stop = last_macro
                            if macro_start > 0 or macro_stop < macro_end:
                                fd_slot = first_slot
                                if first_macro > 0 and tile_idx != start_tile_idx:
                                    fd_slot += 1
                            n_idx_start = macro_start * 2
                            n_idx_end = dynamic_min(n_idx_end, macro_stop * 2)
                            macro_end = macro_stop - macro_start

                    m_axis_triple = active_m_tile_idx % self.preload_num
                    do_tile = self.block_info.has_q_tile_work(
                        seqlen_q_actual, seqlen_k, m_idx
                    )
                    if do_tile:
                        # Load Q once per m-tile
                        q_tile_gm = self._query_tnd_tile(
                            query, q_token_start, seqlen_q_actual, head_idx, m_idx
                        )
                        actual_m = q_tile_gm.shape[0]
                        aligned_m = (
                            ceil_div(actual_m, SPLIT_M_TOTAL_ALIGNMENT)
                            * SPLIT_M_TOTAL_ALIGNMENT
                        )
                        first_part_m = (
                            actual_m
                            if actual_m <= SPLIT_M_ALIGNMENT
                            else aligned_m // AIV_SUBBLOCK_COUNT
                        )
                        local_q_offset = 0 if self.subblock_idx == 0 else first_part_m
                        q_scale_tile_gm = self._query_scale_tnd_tile(
                            q_scale_gm, q_token_start, seqlen_q_actual, head_idx, m_idx
                        )

                        # Capture one L1 slot for all KV subtiles of this Q tile.
                        q_l1_tile = self.q_l1.next_write()
                        q_scale_l1_tile = self.q_scale_l1.next_write()
                        if const_expr(query.shape[-1] == HEAD_DIM_72):
                            cube_fill_l1_zero(
                                q_l1_tile, blk_num=self.tile_cube_m * self.tile_d // 32
                            )
                            cube_sync_pipe(PIPE.MTE2)
                            mem_copy(q_l1_tile[None, :HEAD_DIM_72], q_tile_gm)
                        else:
                            mem_copy(q_l1_tile, q_tile_gm)
                        mem_copy(
                            q_scale_l1_tile,
                            q_scale_tile_gm.flatten(-2, -1).permute(1, 0),
                        )

                        for macro_idx in range(0, macro_end):
                            n_block_a, n_block_b, valid_n_a, valid_n_b, has_second = (
                                self.block_info.get_macro_kv_blocks(
                                    n_idx_start, n_idx_end, macro_idx, seqlen_k
                                )
                            )
                            dynamic_valid_n_a, dynamic_valid_n_b = (
                                self.block_info.get_visible_kv_lengths(
                                    m_idx,
                                    local_q_offset,
                                    seqlen_q_actual,
                                    seqlen_k,
                                    n_block_a,
                                    n_block_b,
                                    valid_n_a,
                                    valid_n_b,
                                )
                            )
                            if const_expr(self.paged):
                                p_copy_n_a = Int64(self.tile_n)
                                p_copy_n_b = Int64(self.tile_n)
                            else:
                                p_copy_n_a = valid_n_a
                                p_copy_n_b = valid_n_b
                            is_first_macro = Int64(1) if macro_idx == 0 else Int64(0)
                            is_last_macro = (
                                Int64(1) if macro_idx == macro_end - 1 else Int64(0)
                            )
                            tile_triple = macro_global_idx % SOFTMAX_DELAY_SLOTS

                            # Retain this macro's metadata for its later PV stage.
                            delayed_metadata.push(
                                batch_idx=batch_idx,
                                head_idx=head_idx,
                                m_idx=m_idx,
                                n_block_a=n_block_a,
                                n_block_b=n_block_b,
                                has_second=has_second,
                                is_first_macro=is_first_macro,
                                is_last_macro=is_last_macro,
                                tile_triple=tile_triple,
                                m_axis_triple=m_axis_triple,
                                fd_slot=fd_slot,
                                q_token_start=q_token_start,
                                kv_token_start=kv_token_start,
                                v_scale_start=v_scale_start,
                            )

                            kv_head_idx = head_idx // head_group_num

                            # Produce QK subtiles before vector consumption.
                            self._compute_qk_tile(
                                self.qk_ub_a,
                                q_l1_tile,
                                q_scale_l1_tile,
                                key,
                                k_scale_gm,
                                block_table,
                                batch_idx,
                                kv_head_idx,
                                kv_token_start,
                                seqlen_k,
                                n_block_a,
                                actual_m,
                            )

                            if has_second:
                                self._compute_qk_tile(
                                    self.qk_ub_b,
                                    q_l1_tile,
                                    q_scale_l1_tile,
                                    key,
                                    k_scale_gm,
                                    block_table,
                                    batch_idx,
                                    kv_head_idx,
                                    kv_token_start,
                                    seqlen_k,
                                    n_block_b,
                                    actual_m,
                                )

                                # P_a uses its local maximum and gets a correction scale;
                                # P_b is emitted in the merged frame with 1.0 scale.
                                if is_first_macro:
                                    self.vector_softmax.compute_first_subtile_softmax(
                                        self.vector_softmax.softmax_max_ub[
                                            m_axis_triple
                                        ],
                                        self.qk_ub_a,
                                        scale,
                                        p_scale,
                                        dynamic_valid_n_a,
                                        valid_n_a,
                                        update=False,
                                    )
                                else:
                                    self.vector_softmax.compute_first_subtile_softmax(
                                        self.vector_softmax.softmax_max_ub[
                                            m_axis_triple
                                        ],
                                        self.qk_ub_a,
                                        scale,
                                        p_scale,
                                        dynamic_valid_n_a,
                                        valid_n_a,
                                        update=True,
                                    )
                                self.vector_softmax.store_quantized_p(
                                    self.p_l1_a, p_copy_n_a
                                )
                                if is_first_macro:
                                    self.vector_softmax.compute_second_subtile_softmax(
                                        self.qk_ub_b,
                                        self.vector_softmax.softmax_max_ub[
                                            m_axis_triple
                                        ],
                                        self.vector_softmax.softmax_sum_ub[
                                            m_axis_triple
                                        ],
                                        self.vector_softmax.prev_running_max_correction_ub[
                                            tile_triple
                                        ],
                                        scale,
                                        p_scale,
                                        dynamic_valid_n_b,
                                        valid_n_b,
                                        update=False,
                                    )
                                else:
                                    self.vector_softmax.compute_second_subtile_softmax(
                                        self.qk_ub_b,
                                        self.vector_softmax.softmax_max_ub[
                                            m_axis_triple
                                        ],
                                        self.vector_softmax.softmax_sum_ub[
                                            m_axis_triple
                                        ],
                                        self.vector_softmax.prev_running_max_correction_ub[
                                            tile_triple
                                        ],
                                        scale,
                                        p_scale,
                                        dynamic_valid_n_b,
                                        valid_n_b,
                                        update=True,
                                    )
                                self.vector_softmax.encode_p_scale()
                                self.vector_softmax.store_compact_p_scale(
                                    self.p_scale_l1_a, actual_m
                                )
                                self.vector_softmax.store_quantized_p(
                                    self.p_l1_b, p_copy_n_b
                                )
                                self.vector_softmax.encode_p_scale(use_unit_scale=True)
                                self.vector_softmax.store_compact_p_scale(
                                    self.p_scale_l1_b, actual_m
                                )
                            else:
                                if is_first_macro:
                                    self.vector_softmax.compute_single_subtile_softmax(
                                        self.qk_ub_a,
                                        self.vector_softmax.softmax_max_ub[
                                            m_axis_triple
                                        ],
                                        self.vector_softmax.softmax_sum_ub[
                                            m_axis_triple
                                        ],
                                        self.vector_softmax.prev_running_max_correction_ub[
                                            tile_triple
                                        ],
                                        scale,
                                        p_scale,
                                        dynamic_valid_n_a,
                                        valid_n_a,
                                        update=False,
                                    )
                                else:
                                    self.vector_softmax.compute_single_subtile_softmax(
                                        self.qk_ub_a,
                                        self.vector_softmax.softmax_max_ub[
                                            m_axis_triple
                                        ],
                                        self.vector_softmax.softmax_sum_ub[
                                            m_axis_triple
                                        ],
                                        self.vector_softmax.prev_running_max_correction_ub[
                                            tile_triple
                                        ],
                                        scale,
                                        p_scale,
                                        dynamic_valid_n_a,
                                        valid_n_a,
                                        update=True,
                                    )
                                self.vector_softmax.store_quantized_p(
                                    self.p_l1_a, p_copy_n_a
                                )
                                self.vector_softmax.encode_p_scale(use_unit_scale=True)
                                self.vector_softmax.store_compact_p_scale(
                                    self.p_scale_l1_a, actual_m
                                )

                            # Three P/scale slots allow current QK production first.
                            # Consume the two-round-old macro before advancing metadata.
                            if macro_global_idx >= self.preload_num - 1:
                                delayed = delayed_metadata.tap(self.preload_num - 1)
                                self._consume_delayed_pv(
                                    attn_out,
                                    softmax_lse_gm,
                                    value,
                                    v_scale_gm,
                                    block_table,
                                    seqused_q,
                                    seqused_kv,
                                    p_scale,
                                    head_group_num,
                                    delayed.batch_idx,
                                    delayed.head_idx,
                                    delayed.m_idx,
                                    delayed.n_block_a,
                                    delayed.n_block_b,
                                    delayed.has_second,
                                    delayed.is_first_macro,
                                    delayed.is_last_macro,
                                    delayed.tile_triple,
                                    delayed.m_axis_triple,
                                    delayed.fd_slot,
                                    delayed.q_token_start,
                                    delayed.kv_token_start,
                                    delayed.v_scale_start,
                                )

                            delayed_metadata.advance()
                            macro_global_idx += 1

                        active_m_tile_idx += 1

        # Drain up to preload_num - 1 stages with explicit branches so the
        # compiler cannot hoist RingBuffer waits out of a dynamic loop.
        # Drain the oldest outstanding macro.
        if macro_global_idx >= self.preload_num - 1:
            delayed = delayed_metadata.tap(self.preload_num - 1)
            self._consume_delayed_pv(
                attn_out,
                softmax_lse_gm,
                value,
                v_scale_gm,
                block_table,
                seqused_q,
                seqused_kv,
                p_scale,
                head_group_num,
                delayed.batch_idx,
                delayed.head_idx,
                delayed.m_idx,
                delayed.n_block_a,
                delayed.n_block_b,
                delayed.has_second,
                delayed.is_first_macro,
                delayed.is_last_macro,
                delayed.tile_triple,
                delayed.m_axis_triple,
                delayed.fd_slot,
                delayed.q_token_start,
                delayed.kv_token_start,
                delayed.v_scale_start,
            )

        # Drain the final macro (also the only one in a single-macro run).
        if macro_global_idx > 0:
            delayed = delayed_metadata.tap(1)
            self._consume_delayed_pv(
                attn_out,
                softmax_lse_gm,
                value,
                v_scale_gm,
                block_table,
                seqused_q,
                seqused_kv,
                p_scale,
                head_group_num,
                delayed.batch_idx,
                delayed.head_idx,
                delayed.m_idx,
                delayed.n_block_a,
                delayed.n_block_b,
                delayed.has_second,
                delayed.is_first_macro,
                delayed.is_last_macro,
                delayed.tile_triple,
                delayed.m_axis_triple,
                delayed.fd_slot,
                delayed.q_token_start,
                delayed.kv_token_start,
                delayed.v_scale_start,
            )

        if const_expr(self.enable_flash_decode):
            if fd_active:
                # All launched blocks participate, including idle AIV halves.
                # Final delayed PV and all producer DMAs precede this barrier.
                global_sync_all()
                onchip_rewind(reset_sync_id=True)
                self._setup_fd_merge()
                self.vector_output.init_invalid_row_check(
                    self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL,
                    True,
                    scale,
                    p_scale,
                )
                for section in range(sections):
                    vec = self.block_idx * AIV_SUBBLOCK_COUNT + self.subblock_idx
                    record = (
                        1
                        + sections * self.block_num
                        + section * self.block_num * AIV_SUBBLOCK_COUNT
                        + vec
                    ) * METADATA_STRIDE
                    self._merge_fd_task(
                        attn_out,
                        softmax_lse_gm,
                        seqused_q,
                        seqused_kv,
                        head_num_q,
                        Int64(1),
                        fd_metadata_gm,
                        record,
                        section * 2 * self.block_num,
                        p_scale,
                    )
                # Finish the merge on every core before leaving the mixed kernel.
                global_sync_all()


# ---- Prefill JIT launcher ---------------------------------------------------
class FlashAttentionGQAMxfp8PrefillLauncher:
    """JIT launcher for the DN prefill kernel."""

    def __init__(
        self,
        mask_mode,
        kv_layout,
        q_layout=QLayout.TND,
        paged=False,
        block_size=128,
        return_softmax_lse=False,
        enable_flash_decode=False,
        *,
        block_dim,
    ):
        self.enable_flash_decode = enable_flash_decode
        check_mask_mode(mask_mode)
        self.mask_mode = mask_mode
        self.q_layout = q_layout

        self.kv_layout = kv_layout
        self.paged = self.kv_layout != KVLayout.TND
        self.block_size = block_size
        self.return_softmax_lse = return_softmax_lse
        self.block_dim = block_dim

    @host
    def launch(
        self,
        attn_out: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_scale_gm: Tensor,
        k_scale_gm: Tensor,
        v_scale_gm: Tensor,
        p_scale_gm: Tensor,
        scale: dtypes.float32,
        block_table: Tensor,
        seqused_kv: Tensor,
        seqused_q: Tensor,
        max_seqlen_q: int,
        softmax_lse_gm: Tensor | None = None,
        fd_output_gm: Tensor | None = None,
        fd_sum_gm: Tensor | None = None,
        fd_max_gm: Tensor | None = None,
        fd_metadata_gm: Tensor | None = None,
    ):
        head_dim = query.shape[2]
        tile_cube_m, tile_vec_m, tile_n, tile_d = select_tile_config(head_dim)
        op = FlashAttentionGQAMxfp8PrefillKernel(
            block_num=self.block_dim,
            enable_flash_decode=self.enable_flash_decode,
            tile_cube_m=tile_cube_m,
            tile_vec_m=tile_vec_m,
            tile_n=tile_n,
            tile_d=tile_d,
            mask_mode=self.mask_mode,
            q_layout=self.q_layout,
            paged=self.paged,
            block_size=self.block_size,
            kv_layout=self.kv_layout,
            return_softmax_lse=self.return_softmax_lse,
        )
        op[self.block_dim](
            attn_out,
            query,
            key,
            value,
            q_scale_gm,
            k_scale_gm,
            v_scale_gm,
            p_scale_gm,
            scale,
            block_table,
            seqused_kv,
            seqused_q,
            max_seqlen_q,
            softmax_lse_gm,
            fd_output_gm,
            fd_sum_gm,
            fd_max_gm,
            fd_metadata_gm,
        )


# =============================================================================
# Decode implementation
# =============================================================================


class DecodeCubeMatmul(_CubeMatmulBase):
    """ND-path Cube matmul for Q @ K^T and P @ V.

    QK and PV reuse the same two physical L0 slots, binding operation-specific
    NZ extents for square and rectangular tiles alike.
    """

    def __init__(self, tile_m, tile_n, tile_d):
        """Configure L0 channels and copy engines for the tile geometry."""
        super().__init__(tile_n, tile_d)

        # Two shared physical slots; QK and PV bind their own NZ extents.
        # L0A=2*M*max(D,N), L0B=2*N*D, L0C=8*M*max(D,N) bytes.
        self.rectangular = False
        self.share_l0_ring_buffers = True
        self.l0a = _make_ring_buffer(
            MemLoc.L0A, (tile_m, max(tile_d, tile_n)), dtypes.float8_e4m3fn, depth=2
        )
        self.l0b = _make_ring_buffer(
            MemLoc.L0B, (tile_n, tile_d), dtypes.float8_e4m3fn, depth=2
        )
        self.l0c = _make_ring_buffer(
            MemLoc.L0C, (tile_m, max(tile_n, tile_d)), dtypes.float32, depth=2
        )
        self.qk_l0c_spec = _View((tile_m, tile_n), dtypes.float32, "nz")
        self.pv_l0c_spec = _View((tile_m, tile_d), dtypes.float32, "nz")
        self.qk_l0a_spec = _View((tile_m, tile_d), dtypes.float8_e4m3fn, "nz")
        self.qk_l0b_spec = _View((tile_n, tile_d), dtypes.float8_e4m3fn, "nz")
        self.pv_l0a_spec = _View((tile_m, tile_n), dtypes.float8_e4m3fn, "nz")
        self.pv_l0b_spec = _View((tile_d, tile_n), dtypes.float8_e4m3fn, "nz")

        # Copy engines
        self.transposed_qk = False

    # QK matmul

    def compute_qk(self, q_l1, k_l1, q_scale_l1, k_scale_l1, actual_m, actual_n=None):
        """Compute Q @ K^T; source Q and scale remain persistent L1 views."""
        if self.rectangular:
            l0a, l0b, l0c = self.l0a_qk, self.l0b_qk, self.l0c_qk
        else:
            l0a, l0b, l0c = self.l0a, self.l0b, self.l0c
        if actual_n is None:
            actual_n = self.tile_n
        a = mem_copy(
            _compact_l0(
                self._l0_slot(l0a, True, self.qk_l0a_spec), (actual_m, self.tile_d)
            ),
            q_l1[:actual_m, : self.tile_d],
            options=copy_options(mx_scale=q_scale_l1[:actual_m, : q_scale_l1.shape[1]]),
        )
        b = mem_copy(
            _compact_l0(
                self._l0_slot(l0b, True, self.qk_l0b_spec), (actual_n, self.tile_d)
            ),
            k_l1.next_read()[:actual_n, : self.tile_d],
            options=copy_options(
                mx_scale=k_scale_l1.next_read()[: k_scale_l1.shape[0], :actual_n]
            ),
        )
        result = _compact_l0(
            self._l0_slot(l0c, True, self.qk_l0c_spec), (actual_m, actual_n)
        )
        matmul(result, a, b, init=True, mx_mode=True)

    def store_qk_result(self, qk_ub, actual_m, actual_n=None):
        ring = self.l0c_qk if self.rectangular else self.l0c
        if actual_n is None:
            actual_n = self.tile_n
        result = _compact_l0(
            self._l0_slot(ring, False, self.qk_l0c_spec), (actual_m, actual_n)
        ).partition(dim=0, alignment=SPLIT_M_ALIGNMENT)
        mem_copy(qk_ub[: result.shape[0], : result.shape[1]], result)

    def compute_pv(
        self,
        p_l1,
        v_l1,
        p_scale_l1,
        v_scale_l1,
        init=True,
        actual_k=None,
        accumulator=None,
    ):
        super()._compute_pv_impl(
            p_l1, v_l1, p_scale_l1, v_scale_l1, init, actual_k, accumulator
        )

    def begin_pv(self, actual_m):
        ring = self.l0c if self.share_l0_ring_buffers else self.l0c_pv
        return _compact_l0(
            self._l0_slot(ring, True, self.pv_l0c_spec), (actual_m, self.tile_d)
        )

    def store_pv_result(self, pv_ub, actual_m):
        ring = self.l0c if self.share_l0_ring_buffers else self.l0c_pv
        result = _compact_l0(
            self._l0_slot(ring, False, self.pv_l0c_spec), (actual_m, self.tile_d)
        ).partition(dim=0, alignment=SPLIT_M_ALIGNMENT)
        destination = pv_ub.next_write()
        mem_copy(destination[: result.shape[0], : result.shape[1]], result)


class DecodeVectorSoftmax(_VectorSoftmaxBase):
    """ND softmax fuses contiguous K-row exp, FP32 sum and FP8 quantization.

    Each Q row uses horizontal reductions. Its exp register feeds both the
    FP32 sum and packed FP8 ND storage. A separate wide copy emits P in NZ;
    this avoids the extra register packs needed for narrow direct NZ stores.
    """

    def __init__(self, tile_vec_m, tile_n, mask_mode, subblock_idx):
        super().__init__(
            tile_vec_m, tile_n, mask_mode, subblock_idx, p_double_buffer=True
        )

    @jit
    def init_gather_indices(self):
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            bytes128, _ = rr.update_mask(128, elem_bits=8)
            lanes = rr.varange(0, dtypes.uint32)
            idx128 = rr.vadds(
                rr.vmuls(lanes, 134744072, mask=full), 100925952, mask=full
            )
            idx64 = rr.vadds(
                rr.vmuls(lanes, 269488144, mask=full), 201851904, mask=full
            )
            first16, _ = rr.update_mask(16, elem_bits=32)
            padding = rr.vdups(16843009, dtypes.uint32, mask=full)
            idx64 = rr.vselect(idx64, padding, cond_mask=first16)
            rr.vstore(
                self.gather_indices,
                0,
                rr.vreinterpret_lanes(idx128, dtypes.uint8),
                bytes128,
            )
            rr.vstore(
                self.short_indices,
                0,
                rr.vreinterpret_lanes(idx64, dtypes.uint8),
                bytes128,
            )
            rr.vmem_bar("vst_vld")

    @jit
    def compute_first_subtile_softmax(
        self,
        running_max_ub,
        qk_ub,
        scale,
        p_scale,
        valid_n,
        tile_valid_n,
        row_start,
        group_num,
        actual_m,
        update=True,
    ):
        if const_expr(self.mask_mode == MASK_MODE_NO_MASK):
            if valid_n == self.tile_n:
                self.compute_first_subtile_softmax_full(
                    running_max_ub,
                    qk_ub,
                    scale,
                    p_scale,
                    tile_valid_n,
                    actual_m,
                    update=update,
                )
            else:
                self.compute_first_subtile_softmax_general(
                    running_max_ub,
                    qk_ub,
                    scale,
                    p_scale,
                    valid_n,
                    tile_valid_n,
                    row_start,
                    group_num,
                    actual_m,
                    update=update,
                )
        elif const_expr(self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL):
            if valid_n >= self.tile_n and tile_valid_n == self.tile_n:
                self.compute_first_subtile_softmax_full(
                    running_max_ub,
                    qk_ub,
                    scale,
                    p_scale,
                    tile_valid_n,
                    actual_m,
                    update=update,
                )
            else:
                self.compute_first_subtile_softmax_general(
                    running_max_ub,
                    qk_ub,
                    scale,
                    p_scale,
                    valid_n,
                    tile_valid_n,
                    row_start,
                    group_num,
                    actual_m,
                    update=update,
                )
        else:
            self.compute_first_subtile_softmax_general(
                running_max_ub,
                qk_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                row_start,
                group_num,
                actual_m,
                update=update,
            )

    @jit
    def compute_first_subtile_softmax_full(
        self,
        running_max_ub,
        qk_ub,
        scale,
        p_scale,
        tile_valid_n,
        actual_m,
        update=True,
    ):
        """Emit the first sub-tile in its local-max frame and cache its exponent sum."""
        vec_rows = ceil_div(actual_m, SPLIT_M_TOTAL_ALIGNMENT) * SPLIT_M_ALIGNMENT
        qk_ub = qk_ub.next_read()
        dst = self.p_ub.next_write()
        with vf(mode="simd"):
            for i in range(vec_rows):
                self._scale_full_scores_and_compute_row_max(
                    qk_ub, scale, self.max_first_subtile_ub, i, self.tile_n
                )
            rr.vmem_bar("vst_vld")
            rowmask, _ = rr.update_mask(vec_rows, elem_bits=FP32_ELEM_BITS)
            vreg_max_first_subtile = rr.vload(self.max_first_subtile_ub, 0)
            vreg_max_first_subtile_aligned = vreg_max_first_subtile
            log_p_scale = rr.vlog(
                rr.vdups(p_scale, dtypes.float32, mask=rowmask), mask=rowmask
            )
            vreg_max_first_subtile_aligned = rr.vmaxs(
                rr.vsub(vreg_max_first_subtile_aligned, log_p_scale, mask=rowmask),
                SOFTMAX_MASKED_SCORE_FLOOR,
                mask=rowmask,
            )
            if const_expr(update):
                vreg_max_first_subtile_aligned = rr.vmax(
                    vreg_max_first_subtile_aligned,
                    rr.vload(running_max_ub, 0),
                    mask=rowmask,
                )
            rr.vstore(
                self.max_first_subtile_ub, 0, vreg_max_first_subtile_aligned, rowmask
            )
            rr.vmem_bar("vst_vld")
            for i in range(vec_rows):
                self._compute_row_exp_and_sum(
                    qk_ub,
                    self.max_first_subtile_ub,
                    self.first_subtile_exp_sum_ub,
                    i,
                    dst,
                    self.tile_n,
                    tile_valid_n,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def compute_first_subtile_softmax_general(
        self,
        running_max_ub,
        qk_ub,
        scale,
        p_scale,
        valid_n,
        tile_valid_n,
        row_start,
        group_num,
        actual_m,
        update=True,
    ):
        if tile_valid_n == 128:
            self.compute_first_subtile_softmax_general_width(
                running_max_ub,
                qk_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                row_start,
                group_num,
                actual_m,
                128,
                True,
                update=update,
            )
        elif tile_valid_n <= 64:
            self.compute_first_subtile_softmax_general_width(
                running_max_ub,
                qk_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                row_start,
                group_num,
                actual_m,
                64,
                False,
                update=update,
            )
        elif tile_valid_n < 128:
            self.compute_first_subtile_softmax_general_width(
                running_max_ub,
                qk_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                row_start,
                group_num,
                actual_m,
                128,
                False,
                update=update,
            )
        else:
            self.compute_first_subtile_softmax_general_width(
                running_max_ub,
                qk_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                row_start,
                group_num,
                actual_m,
                self.tile_n,
                False,
                update=update,
            )

    @jit
    def compute_first_subtile_softmax_general_width(
        self,
        running_max_ub,
        qk_ub,
        scale,
        p_scale,
        valid_n,
        tile_valid_n,
        row_start,
        group_num,
        actual_m,
        width,
        exact_width,
        update=True,
    ):
        """Emit the first sub-tile in its local-max frame and cache its exponent sum."""
        vec_rows = ceil_div(actual_m, SPLIT_M_TOTAL_ALIGNMENT) * SPLIT_M_ALIGNMENT
        qk_ub = qk_ub.next_read()
        dst = self.p_ub.next_write()
        with vf(mode="simd"):
            for i in range(vec_rows):
                if const_expr(self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL):
                    row_valid_n = dynamic_min(
                        # Consecutive heads of the same Q token share a boundary.
                        valid_n
                        + dtypes.int32(row_start + i) // dtypes.int32(group_num),
                        tile_valid_n,
                    )
                else:
                    row_valid_n = valid_n
                self._scale_valid_scores_and_compute_row_max(
                    qk_ub,
                    scale,
                    self.max_first_subtile_ub,
                    i,
                    row_valid_n,
                    width,
                    exact_width,
                    tile_valid_n,
                )
            rr.vmem_bar("vst_vld")
            rowmask, _ = rr.update_mask(vec_rows, elem_bits=FP32_ELEM_BITS)
            vreg_max_first_subtile = rr.vload(self.max_first_subtile_ub, 0)
            vreg_max_first_subtile_aligned = vreg_max_first_subtile
            log_p_scale = rr.vlog(
                rr.vdups(p_scale, dtypes.float32, mask=rowmask), mask=rowmask
            )
            vreg_max_first_subtile_aligned = rr.vmaxs(
                rr.vsub(vreg_max_first_subtile_aligned, log_p_scale, mask=rowmask),
                SOFTMAX_MASKED_SCORE_FLOOR,
                mask=rowmask,
            )
            if const_expr(update):
                vreg_max_first_subtile_aligned = rr.vmax(
                    vreg_max_first_subtile_aligned,
                    rr.vload(running_max_ub, 0),
                    mask=rowmask,
                )
            rr.vstore(
                self.max_first_subtile_ub, 0, vreg_max_first_subtile_aligned, rowmask
            )
            rr.vmem_bar("vst_vld")
            for i in range(vec_rows):
                self._compute_row_exp_and_sum(
                    qk_ub,
                    self.max_first_subtile_ub,
                    self.first_subtile_exp_sum_ub,
                    i,
                    dst,
                    width,
                    tile_valid_n,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def compute_second_subtile_softmax(
        self,
        qk_ub,
        running_max_ub,
        running_exp_sum_ub,
        prev_running_max_correction_ub,
        scale,
        p_scale,
        valid_n,
        tile_valid_n,
        row_start,
        group_num,
        actual_m,
        update=True,
    ):
        if const_expr(self.mask_mode == MASK_MODE_NO_MASK):
            if valid_n == self.tile_n:
                self.compute_second_subtile_softmax_full(
                    qk_ub,
                    running_max_ub,
                    running_exp_sum_ub,
                    prev_running_max_correction_ub,
                    scale,
                    p_scale,
                    tile_valid_n,
                    actual_m,
                    update=update,
                )
            else:
                self.compute_second_subtile_softmax_general(
                    qk_ub,
                    running_max_ub,
                    running_exp_sum_ub,
                    prev_running_max_correction_ub,
                    scale,
                    p_scale,
                    valid_n,
                    tile_valid_n,
                    row_start,
                    group_num,
                    actual_m,
                    update=update,
                )
        elif const_expr(self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL):
            if valid_n >= self.tile_n and tile_valid_n == self.tile_n:
                self.compute_second_subtile_softmax_full(
                    qk_ub,
                    running_max_ub,
                    running_exp_sum_ub,
                    prev_running_max_correction_ub,
                    scale,
                    p_scale,
                    tile_valid_n,
                    actual_m,
                    update=update,
                )
            else:
                self.compute_second_subtile_softmax_general(
                    qk_ub,
                    running_max_ub,
                    running_exp_sum_ub,
                    prev_running_max_correction_ub,
                    scale,
                    p_scale,
                    valid_n,
                    tile_valid_n,
                    row_start,
                    group_num,
                    actual_m,
                    update=update,
                )
        else:
            self.compute_second_subtile_softmax_general(
                qk_ub,
                running_max_ub,
                running_exp_sum_ub,
                prev_running_max_correction_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                row_start,
                group_num,
                actual_m,
                update=update,
            )

    @jit
    def compute_second_subtile_softmax_full(
        self,
        qk_ub,
        running_max_ub,
        running_exp_sum_ub,
        prev_running_max_correction_ub,
        scale,
        p_scale,
        tile_valid_n,
        actual_m,
        update=True,
    ):
        """Merge the second sub-tile, update running max/sum, and emit its p."""
        vec_rows = ceil_div(actual_m, SPLIT_M_TOTAL_ALIGNMENT) * SPLIT_M_ALIGNMENT
        qk_ub = qk_ub.next_read()
        dst = self.p_ub.next_write()
        with vf(mode="simd"):
            for i in range(vec_rows):
                self._scale_full_scores_and_compute_row_max(
                    qk_ub,
                    scale,
                    self.max_second_subtile_ub,
                    i,
                    self.tile_n,
                )
            rr.vmem_bar("vst_vld")
            rowmask, _ = rr.update_mask(vec_rows, elem_bits=FP32_ELEM_BITS)
            vreg_max_second_subtile = rr.vload(self.max_second_subtile_ub, 0)
            vreg_max_second_subtile_aligned = vreg_max_second_subtile
            log_p_scale = rr.vlog(
                rr.vdups(p_scale, dtypes.float32, mask=rowmask), mask=rowmask
            )
            vreg_max_second_subtile_aligned = rr.vmaxs(
                rr.vsub(vreg_max_second_subtile_aligned, log_p_scale, mask=rowmask),
                SOFTMAX_MASKED_SCORE_FLOOR,
                mask=rowmask,
            )
            rr.vstore(
                self.max_second_subtile_ub, 0, vreg_max_second_subtile_aligned, rowmask
            )
            vreg_max_first_subtile = rr.vload(self.max_first_subtile_ub, 0)
            vreg_updated_running_max = rr.vmax(
                vreg_max_first_subtile, vreg_max_second_subtile_aligned, mask=rowmask
            )
            rr.vstore(self.updated_running_max_ub, 0, vreg_updated_running_max, rowmask)
            vreg_first_subtile_max_correction = rr.vexp_sub(
                vreg_max_first_subtile, vreg_updated_running_max, mask=rowmask
            )
            rr.vstore(
                self.first_subtile_max_correction_ub,
                0,
                vreg_first_subtile_max_correction,
                rowmask,
            )
            if const_expr(update):
                vreg_previous_running_max = rr.vload(running_max_ub, 0)
                vreg_previous_max_correction = rr.vexp_sub(
                    vreg_previous_running_max, vreg_updated_running_max, mask=rowmask
                )
                rr.vstore(
                    prev_running_max_correction_ub,
                    0,
                    vreg_previous_max_correction,
                    rowmask,
                )
            rr.vstore(running_max_ub, 0, vreg_updated_running_max, rowmask)
            rr.vmem_bar("vst_vld")
        with vf(mode="simd"):
            for i in range(vec_rows):
                # Separate exp/sum and FP8 conversion to limit live registers.
                self._compute_row_exp_and_sum(
                    qk_ub,
                    self.updated_running_max_ub,
                    self.current_subtile_exp_sum_ub,
                    i,
                    dst,
                    self.tile_n,
                    tile_valid_n,
                )
            rr.vmem_bar("vst_vld")
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(vec_rows, elem_bits=FP32_ELEM_BITS)
            vreg_first_subtile_exp_sum = rr.vload(self.first_subtile_exp_sum_ub, 0)
            vreg_second_subtile_exp_sum = rr.vload(self.current_subtile_exp_sum_ub, 0)
            vreg_first_subtile_max_correction = rr.vload(
                self.first_subtile_max_correction_ub, 0
            )
            if const_expr(update):
                vreg_previous_running_exp_sum = rr.vload(running_exp_sum_ub, 0)
                vreg_previous_max_correction = rr.vload(
                    prev_running_max_correction_ub, 0
                )
                vreg_updated_running_exp_sum = _merge_softmax_sum(
                    vreg_previous_running_exp_sum,
                    vreg_previous_max_correction,
                    vreg_first_subtile_exp_sum,
                    vreg_first_subtile_max_correction,
                    vreg_second_subtile_exp_sum,
                    rowmask,
                )
            else:
                vreg_updated_running_exp_sum = rr.vadd(
                    rr.vmul(
                        vreg_first_subtile_exp_sum,
                        vreg_first_subtile_max_correction,
                        mask=rowmask,
                    ),
                    vreg_second_subtile_exp_sum,
                    mask=rowmask,
                )
            rr.vstore(running_exp_sum_ub, 0, vreg_updated_running_exp_sum, rowmask)
            rr.vmem_bar("vst_vld")

    @jit
    def compute_second_subtile_softmax_general(
        self,
        qk_ub,
        running_max_ub,
        running_exp_sum_ub,
        prev_running_max_correction_ub,
        scale,
        p_scale,
        valid_n,
        tile_valid_n,
        row_start,
        group_num,
        actual_m,
        update=True,
    ):
        if tile_valid_n == 128:
            self.compute_second_subtile_softmax_general_width(
                qk_ub,
                running_max_ub,
                running_exp_sum_ub,
                prev_running_max_correction_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                row_start,
                group_num,
                actual_m,
                128,
                True,
                update=update,
            )
        elif tile_valid_n <= 64:
            self.compute_second_subtile_softmax_general_width(
                qk_ub,
                running_max_ub,
                running_exp_sum_ub,
                prev_running_max_correction_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                row_start,
                group_num,
                actual_m,
                64,
                False,
                update=update,
            )
        elif tile_valid_n < 128:
            self.compute_second_subtile_softmax_general_width(
                qk_ub,
                running_max_ub,
                running_exp_sum_ub,
                prev_running_max_correction_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                row_start,
                group_num,
                actual_m,
                128,
                False,
                update=update,
            )
        else:
            self.compute_second_subtile_softmax_general_width(
                qk_ub,
                running_max_ub,
                running_exp_sum_ub,
                prev_running_max_correction_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                row_start,
                group_num,
                actual_m,
                self.tile_n,
                False,
                update=update,
            )

    @jit
    def compute_second_subtile_softmax_general_width(
        self,
        qk_ub,
        running_max_ub,
        running_exp_sum_ub,
        prev_running_max_correction_ub,
        scale,
        p_scale,
        valid_n,
        tile_valid_n,
        row_start,
        group_num,
        actual_m,
        width,
        exact_width,
        update=True,
    ):
        """Merge the second sub-tile, update running max/sum, and emit its p."""
        vec_rows = ceil_div(actual_m, SPLIT_M_TOTAL_ALIGNMENT) * SPLIT_M_ALIGNMENT
        qk_ub = qk_ub.next_read()
        dst = self.p_ub.next_write()
        with vf(mode="simd"):
            for i in range(vec_rows):
                if const_expr(self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL):
                    row_valid_n = dynamic_min(
                        # Consecutive heads of the same Q token share a boundary.
                        valid_n
                        + dtypes.int32(row_start + i) // dtypes.int32(group_num),
                        tile_valid_n,
                    )
                else:
                    row_valid_n = valid_n
                self._scale_valid_scores_and_compute_row_max(
                    qk_ub,
                    scale,
                    self.max_second_subtile_ub,
                    i,
                    row_valid_n,
                    width,
                    exact_width,
                    tile_valid_n,
                )
            rr.vmem_bar("vst_vld")
            rowmask, _ = rr.update_mask(vec_rows, elem_bits=FP32_ELEM_BITS)
            vreg_max_second_subtile = rr.vload(self.max_second_subtile_ub, 0)
            vreg_max_second_subtile_aligned = vreg_max_second_subtile
            log_p_scale = rr.vlog(
                rr.vdups(p_scale, dtypes.float32, mask=rowmask), mask=rowmask
            )
            vreg_max_second_subtile_aligned = rr.vmaxs(
                rr.vsub(vreg_max_second_subtile_aligned, log_p_scale, mask=rowmask),
                SOFTMAX_MASKED_SCORE_FLOOR,
                mask=rowmask,
            )
            rr.vstore(
                self.max_second_subtile_ub, 0, vreg_max_second_subtile_aligned, rowmask
            )
            vreg_max_first_subtile = rr.vload(self.max_first_subtile_ub, 0)
            vreg_updated_running_max = rr.vmax(
                vreg_max_first_subtile, vreg_max_second_subtile_aligned, mask=rowmask
            )
            rr.vstore(self.updated_running_max_ub, 0, vreg_updated_running_max, rowmask)
            vreg_first_subtile_max_correction = rr.vexp_sub(
                vreg_max_first_subtile, vreg_updated_running_max, mask=rowmask
            )
            rr.vstore(
                self.first_subtile_max_correction_ub,
                0,
                vreg_first_subtile_max_correction,
                rowmask,
            )
            if const_expr(update):
                vreg_previous_running_max = rr.vload(running_max_ub, 0)
                vreg_previous_max_correction = rr.vexp_sub(
                    vreg_previous_running_max, vreg_updated_running_max, mask=rowmask
                )
                rr.vstore(
                    prev_running_max_correction_ub,
                    0,
                    vreg_previous_max_correction,
                    rowmask,
                )
            rr.vstore(running_max_ub, 0, vreg_updated_running_max, rowmask)
            rr.vmem_bar("vst_vld")
        with vf(mode="simd"):
            for i in range(vec_rows):
                # Separate exp/sum and FP8 conversion to limit live registers.
                self._compute_row_exp_and_sum(
                    qk_ub,
                    self.updated_running_max_ub,
                    self.current_subtile_exp_sum_ub,
                    i,
                    dst,
                    width,
                    tile_valid_n,
                )
            rr.vmem_bar("vst_vld")
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(vec_rows, elem_bits=FP32_ELEM_BITS)
            vreg_first_subtile_exp_sum = rr.vload(self.first_subtile_exp_sum_ub, 0)
            vreg_second_subtile_exp_sum = rr.vload(self.current_subtile_exp_sum_ub, 0)
            vreg_first_subtile_max_correction = rr.vload(
                self.first_subtile_max_correction_ub, 0
            )
            if const_expr(update):
                vreg_previous_running_exp_sum = rr.vload(running_exp_sum_ub, 0)
                vreg_previous_max_correction = rr.vload(
                    prev_running_max_correction_ub, 0
                )
                vreg_updated_running_exp_sum = _merge_softmax_sum(
                    vreg_previous_running_exp_sum,
                    vreg_previous_max_correction,
                    vreg_first_subtile_exp_sum,
                    vreg_first_subtile_max_correction,
                    vreg_second_subtile_exp_sum,
                    rowmask,
                )
            else:
                vreg_updated_running_exp_sum = rr.vadd(
                    rr.vmul(
                        vreg_first_subtile_exp_sum,
                        vreg_first_subtile_max_correction,
                        mask=rowmask,
                    ),
                    vreg_second_subtile_exp_sum,
                    mask=rowmask,
                )
            rr.vstore(running_exp_sum_ub, 0, vreg_updated_running_exp_sum, rowmask)
            rr.vmem_bar("vst_vld")

    @jit
    def compute_single_subtile_softmax(
        self,
        qk_ub,
        running_max_ub,
        running_exp_sum_ub,
        prev_running_max_correction_ub,
        scale,
        p_scale,
        valid_n,
        tile_valid_n,
        row_start,
        group_num,
        actual_m,
        update=True,
    ):
        if const_expr(self.mask_mode == MASK_MODE_NO_MASK):
            if valid_n == self.tile_n:
                self.compute_single_subtile_softmax_full(
                    qk_ub,
                    running_max_ub,
                    running_exp_sum_ub,
                    prev_running_max_correction_ub,
                    scale,
                    p_scale,
                    tile_valid_n,
                    actual_m,
                    update=update,
                )
            else:
                self.compute_single_subtile_softmax_general(
                    qk_ub,
                    running_max_ub,
                    running_exp_sum_ub,
                    prev_running_max_correction_ub,
                    scale,
                    p_scale,
                    valid_n,
                    tile_valid_n,
                    row_start,
                    group_num,
                    actual_m,
                    update=update,
                )
        elif const_expr(self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL):
            if valid_n >= self.tile_n and tile_valid_n == self.tile_n:
                self.compute_single_subtile_softmax_full(
                    qk_ub,
                    running_max_ub,
                    running_exp_sum_ub,
                    prev_running_max_correction_ub,
                    scale,
                    p_scale,
                    tile_valid_n,
                    actual_m,
                    update=update,
                )
            else:
                self.compute_single_subtile_softmax_general(
                    qk_ub,
                    running_max_ub,
                    running_exp_sum_ub,
                    prev_running_max_correction_ub,
                    scale,
                    p_scale,
                    valid_n,
                    tile_valid_n,
                    row_start,
                    group_num,
                    actual_m,
                    update=update,
                )
        else:
            self.compute_single_subtile_softmax_general(
                qk_ub,
                running_max_ub,
                running_exp_sum_ub,
                prev_running_max_correction_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                row_start,
                group_num,
                actual_m,
                update=update,
            )

    @jit
    def compute_single_subtile_softmax_full(
        self,
        qk_ub,
        running_max_ub,
        running_exp_sum_ub,
        prev_running_max_correction_ub,
        scale,
        p_scale,
        tile_valid_n,
        actual_m,
        update=True,
    ):
        """Process a one-subtile macro directly in the running common frame."""
        vec_rows = ceil_div(actual_m, SPLIT_M_TOTAL_ALIGNMENT) * SPLIT_M_ALIGNMENT
        qk_ub = qk_ub.next_read()
        dst = self.p_ub.next_write()
        with vf(mode="simd"):
            for i in range(vec_rows):
                self._scale_full_scores_and_compute_row_max(
                    qk_ub, scale, self.max_first_subtile_ub, i, self.tile_n
                )
            rr.vmem_bar("vst_vld")
            rowmask, _ = rr.update_mask(vec_rows, elem_bits=FP32_ELEM_BITS)
            vreg_max_current_subtile = rr.vload(self.max_first_subtile_ub, 0)
            log_p_scale = rr.vlog(
                rr.vdups(p_scale, dtypes.float32, mask=rowmask), mask=rowmask
            )
            vreg_max_current_subtile = rr.vmaxs(
                rr.vsub(vreg_max_current_subtile, log_p_scale, mask=rowmask),
                SOFTMAX_MASKED_SCORE_FLOOR,
                mask=rowmask,
            )
            if const_expr(update):
                vreg_previous_running_max = rr.vload(running_max_ub, 0)
                vreg_updated_running_max = rr.vmax(
                    vreg_previous_running_max, vreg_max_current_subtile, mask=rowmask
                )
                vreg_previous_max_correction = rr.vexp_sub(
                    vreg_previous_running_max, vreg_updated_running_max, mask=rowmask
                )
                rr.vstore(
                    prev_running_max_correction_ub,
                    0,
                    vreg_previous_max_correction,
                    rowmask,
                )
            else:
                vreg_updated_running_max = vreg_max_current_subtile
            rr.vstore(running_max_ub, 0, vreg_updated_running_max, rowmask)
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            for i in range(vec_rows):
                self._compute_row_exp_and_sum(
                    qk_ub,
                    running_max_ub,
                    self.current_subtile_exp_sum_ub,
                    i,
                    dst,
                    self.tile_n,
                    tile_valid_n,
                )
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(vec_rows, elem_bits=FP32_ELEM_BITS)
            if const_expr(update):
                vreg_previous_running_exp_sum = rr.vload(running_exp_sum_ub, 0)
                vreg_previous_max_correction = rr.vload(
                    prev_running_max_correction_ub, 0
                )
                vreg_updated_running_exp_sum = rr.vadd(
                    rr.vmul(
                        vreg_previous_running_exp_sum,
                        vreg_previous_max_correction,
                        mask=rowmask,
                    ),
                    rr.vload(self.current_subtile_exp_sum_ub, 0),
                    mask=rowmask,
                )
            else:
                vreg_updated_running_exp_sum = rr.vload(
                    self.current_subtile_exp_sum_ub, 0
                )
            rr.vstore(running_exp_sum_ub, 0, vreg_updated_running_exp_sum, rowmask)
            rr.vmem_bar("vst_vld")

    @jit
    def compute_single_subtile_softmax_general(
        self,
        qk_ub,
        running_max_ub,
        running_exp_sum_ub,
        prev_running_max_correction_ub,
        scale,
        p_scale,
        valid_n,
        tile_valid_n,
        row_start,
        group_num,
        actual_m,
        update=True,
    ):
        if tile_valid_n == 128:
            self.compute_single_subtile_softmax_general_width(
                qk_ub,
                running_max_ub,
                running_exp_sum_ub,
                prev_running_max_correction_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                row_start,
                group_num,
                actual_m,
                128,
                True,
                update=update,
            )
        elif tile_valid_n <= 64:
            self.compute_single_subtile_softmax_general_width(
                qk_ub,
                running_max_ub,
                running_exp_sum_ub,
                prev_running_max_correction_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                row_start,
                group_num,
                actual_m,
                64,
                False,
                update=update,
            )
        elif tile_valid_n < 128:
            self.compute_single_subtile_softmax_general_width(
                qk_ub,
                running_max_ub,
                running_exp_sum_ub,
                prev_running_max_correction_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                row_start,
                group_num,
                actual_m,
                128,
                False,
                update=update,
            )
        else:
            self.compute_single_subtile_softmax_general_width(
                qk_ub,
                running_max_ub,
                running_exp_sum_ub,
                prev_running_max_correction_ub,
                scale,
                p_scale,
                valid_n,
                tile_valid_n,
                row_start,
                group_num,
                actual_m,
                self.tile_n,
                False,
                update=update,
            )

    @jit
    def compute_single_subtile_softmax_general_width(
        self,
        qk_ub,
        running_max_ub,
        running_exp_sum_ub,
        prev_running_max_correction_ub,
        scale,
        p_scale,
        valid_n,
        tile_valid_n,
        row_start,
        group_num,
        actual_m,
        width,
        exact_width,
        update=True,
    ):
        """Process a one-subtile macro directly in the running common frame."""
        vec_rows = ceil_div(actual_m, SPLIT_M_TOTAL_ALIGNMENT) * SPLIT_M_ALIGNMENT
        qk_ub = qk_ub.next_read()
        dst = self.p_ub.next_write()
        with vf(mode="simd"):
            for i in range(vec_rows):
                if const_expr(self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL):
                    row_valid_n = dynamic_min(
                        # Consecutive heads of the same Q token share a boundary.
                        valid_n
                        + dtypes.int32(row_start + i) // dtypes.int32(group_num),
                        tile_valid_n,
                    )
                else:
                    row_valid_n = valid_n
                self._scale_valid_scores_and_compute_row_max(
                    qk_ub,
                    scale,
                    self.max_first_subtile_ub,
                    i,
                    row_valid_n,
                    width,
                    exact_width,
                    tile_valid_n,
                )
            rr.vmem_bar("vst_vld")
            rowmask, _ = rr.update_mask(vec_rows, elem_bits=FP32_ELEM_BITS)
            vreg_max_current_subtile = rr.vload(self.max_first_subtile_ub, 0)
            log_p_scale = rr.vlog(
                rr.vdups(p_scale, dtypes.float32, mask=rowmask), mask=rowmask
            )
            vreg_max_current_subtile = rr.vmaxs(
                rr.vsub(vreg_max_current_subtile, log_p_scale, mask=rowmask),
                SOFTMAX_MASKED_SCORE_FLOOR,
                mask=rowmask,
            )
            if const_expr(update):
                vreg_previous_running_max = rr.vload(running_max_ub, 0)
                vreg_updated_running_max = rr.vmax(
                    vreg_previous_running_max, vreg_max_current_subtile, mask=rowmask
                )
                vreg_previous_max_correction = rr.vexp_sub(
                    vreg_previous_running_max, vreg_updated_running_max, mask=rowmask
                )
                rr.vstore(
                    prev_running_max_correction_ub,
                    0,
                    vreg_previous_max_correction,
                    rowmask,
                )
            else:
                vreg_updated_running_max = vreg_max_current_subtile
            rr.vstore(running_max_ub, 0, vreg_updated_running_max, rowmask)
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            for i in range(vec_rows):
                self._compute_row_exp_and_sum(
                    qk_ub,
                    running_max_ub,
                    self.current_subtile_exp_sum_ub,
                    i,
                    dst,
                    width,
                    tile_valid_n,
                )
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(vec_rows, elem_bits=FP32_ELEM_BITS)
            if const_expr(update):
                vreg_previous_running_exp_sum = rr.vload(running_exp_sum_ub, 0)
                vreg_previous_max_correction = rr.vload(
                    prev_running_max_correction_ub, 0
                )
                vreg_updated_running_exp_sum = rr.vadd(
                    rr.vmul(
                        vreg_previous_running_exp_sum,
                        vreg_previous_max_correction,
                        mask=rowmask,
                    ),
                    rr.vload(self.current_subtile_exp_sum_ub, 0),
                    mask=rowmask,
                )
            else:
                vreg_updated_running_exp_sum = rr.vload(
                    self.current_subtile_exp_sum_ub, 0
                )
            rr.vstore(running_exp_sum_ub, 0, vreg_updated_running_exp_sum, rowmask)
            rr.vmem_bar("vst_vld")

    def _scale_full_scores_and_compute_row_max(
        self, qk_ub, scale, row_max_ub, row, width
    ):
        self._scale_valid_scores_and_compute_row_max(
            qk_ub, scale, row_max_ub, row, width, width, True, width, full_tile=True
        )

    def _scale_score_chunk(
        self, qk_ub, scale, base, col, valid_n, full, floor, needs_mask
    ):
        values = rr.vload(qk_ub, base + col)
        if const_expr(needs_mask):
            active, _ = rr.update_mask(
                _clamp_valid_lanes(valid_n - col, 64), elem_bits=32
            )
            values = rr.vmuls(values, scale, mask=active)
            values = rr.vselect(values, floor, cond_mask=active)
        else:
            values = rr.vmuls(values, scale, mask=full)
        rr.vstore(qk_ub, base + col, values, full)
        return values

    def _scale_valid_scores_and_compute_row_max(
        self,
        qk_ub,
        scale,
        row_max_ub,
        row,
        valid_n,
        width,
        exact_width,
        physical_n,
        full_tile=False,
    ):
        full, _ = rr.update_mask(64, elem_bits=32)
        floor = rr.vdups(SOFTMAX_MASKED_SCORE_FLOOR, dtypes.float32, mask=full)
        base = row * self.tile_n
        causal = self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL and not full_tile
        needs_mask = causal or not exact_width
        if const_expr(causal):
            live_n = valid_n
        else:
            live_n = physical_n
        left = self._scale_score_chunk(
            qk_ub, scale, base, 0, live_n, full, floor, needs_mask
        )
        if const_expr(width > 64):
            right = self._scale_score_chunk(
                qk_ub, scale, base, 64, live_n, full, floor, needs_mask
            )
            left = rr.vmax(left, right, mask=full)
        if const_expr(width > 128):
            x2 = self._scale_score_chunk(
                qk_ub, scale, base, 128, live_n, full, floor, needs_mask
            )
        if const_expr(width > 192):
            x3 = self._scale_score_chunk(
                qk_ub, scale, base, 192, live_n, full, floor, needs_mask
            )
            left = rr.vmax(left, rr.vmax(x2, x3, mask=full), mask=full)
        if const_expr(width == 64):
            # Reduce over the physical tail, including masked positions set to floor.
            active, _ = rr.update_mask(physical_n, elem_bits=32)
            maximum = rr.vreduce_max(left, mask=active)
        else:
            maximum = rr.vreduce_max(left, mask=full)
        rr.vstore_first(
            row_max_ub, row, self._align_maximum_to_log2_scale(maximum, full)
        )

    def _compute_row_exp_and_sum(
        self, qk_ub, row_max_ub, row_exp_sum_ub, row, dst, width, valid_n
    ):
        full, _ = rr.update_mask(64, elem_bits=32)
        maximum = rr.vload_broadcast(row_max_ub, row)
        stride = dst.physical_layout.stride[0] // 32
        if const_expr(width == 64):
            active, _ = rr.update_mask(valid_n, elem_bits=32)
            values = rr.vexp_sub(
                rr.vload(qk_ub, row * self.tile_n), maximum, mask=active
            )
            total = rr.vreduce_sum(values, mask=active)
            p = rr.vcast(
                values,
                dtypes.float8_e4m3fn,
                mask=full,
                reg_layout=rr.RegLayout.ZERO,
                saturate=True,
            )
            indices = rr.vload(self.short_indices, 0)
            ordered = rr.vgather_reg(rr.vreinterpret_lanes(p, dtypes.uint8), indices)
            outmask, _ = rr.update_mask(128, elem_bits=8)
            rr.vstore_strided(
                dst, row * 32, ordered, outmask, block_stride=stride, repeat_stride=0
            )
        else:
            idx = rr.vload(self.gather_indices, 0)
            outmask, _ = rr.update_mask(128, elem_bits=8)
            even, odd = rr.vload_deinterleave(qk_ub, row * self.tile_n + 0 * 128)
            even = rr.vexp_sub(even, maximum, mask=full)
            odd = rr.vexp_sub(odd, maximum, mask=full)
            sum0 = rr.vadd(even, odd, mask=full)
            p0 = rr.vcast(
                even,
                dtypes.float8_e4m3fn,
                mask=full,
                reg_layout=rr.RegLayout.ZERO,
                saturate=True,
            )
            p1 = rr.vcast(
                odd,
                dtypes.float8_e4m3fn,
                mask=full,
                reg_layout=rr.RegLayout.TWO,
                saturate=True,
            )
            packed = rr.vbitwise_or(
                rr.vreinterpret_lanes(p0, dtypes.uint32),
                rr.vreinterpret_lanes(p1, dtypes.uint32),
                mask=full,
            )
            ordered = rr.vgather_reg(rr.vreinterpret_lanes(packed, dtypes.uint8), idx)
            rr.vstore_strided(
                dst,
                row * 32 + 0 * 4 * dst.physical_layout.stride[0],
                ordered,
                outmask,
                block_stride=stride,
                repeat_stride=0,
            )
            partial = sum0
            if const_expr(width == 256):
                even, odd = rr.vload_deinterleave(qk_ub, row * self.tile_n + 1 * 128)
                even = rr.vexp_sub(even, maximum, mask=full)
                odd = rr.vexp_sub(odd, maximum, mask=full)
                sum1 = rr.vadd(even, odd, mask=full)
                p0 = rr.vcast(
                    even,
                    dtypes.float8_e4m3fn,
                    mask=full,
                    reg_layout=rr.RegLayout.ZERO,
                    saturate=True,
                )
                p1 = rr.vcast(
                    odd,
                    dtypes.float8_e4m3fn,
                    mask=full,
                    reg_layout=rr.RegLayout.TWO,
                    saturate=True,
                )
                packed = rr.vbitwise_or(
                    rr.vreinterpret_lanes(p0, dtypes.uint32),
                    rr.vreinterpret_lanes(p1, dtypes.uint32),
                    mask=full,
                )
                ordered = rr.vgather_reg(
                    rr.vreinterpret_lanes(packed, dtypes.uint8), idx
                )
                rr.vstore_strided(
                    dst,
                    row * 32 + 1 * 4 * dst.physical_layout.stride[0],
                    ordered,
                    outmask,
                    block_stride=stride,
                    repeat_stride=0,
                )
                partial = rr.vadd(sum0, sum1, mask=full)
            total = rr.vreduce_sum(partial, mask=full)
        written = max(width, 128)
        if const_expr(written < self.tile_n):
            tail, _ = rr.update_mask(self.tile_n - written, elem_bits=8)
            zero = rr.vdups(0, dtypes.uint8, mask=tail)
            rr.vstore_strided(
                dst,
                row * 32 + (written // 32) * dst.physical_layout.stride[0],
                zero,
                tail,
                block_stride=stride,
                repeat_stride=0,
            )
        rr.vstore_first(row_exp_sum_ub, row, total)

    def store_quantized_p(self, p_l1, actual_m):
        aligned_m = (
            ceil_div(actual_m, SPLIT_M_TOTAL_ALIGNMENT) * SPLIT_M_TOTAL_ALIGNMENT
        )
        vec_rows = aligned_m // AIV_SUBBLOCK_COUNT
        slot = p_l1.next_write()
        target = slot[:aligned_m, : self.tile_n]
        source = self.p_ub.next_read()[:vec_rows, : self.tile_n]
        mem_copy(
            target.partition(
                dim=0, part_id=self.subblock_idx, alignment=SPLIT_M_ALIGNMENT
            ),
            source,
        )

    def encode_p_scale(self, use_unit_scale=False, actual_m=None):
        """Encode P's compact E8M0 scale bytes for this AIV tile.

        ``use_unit_scale`` emits the 1.0 scale used when P already matches
        the current max frame; otherwise this emits first-subtile correction
        scales so PV can consume that P tile in the updated max frame.
        """
        vec_rows = ceil_div(actual_m, SPLIT_M_TOTAL_ALIGNMENT) * SPLIT_M_ALIGNMENT
        if use_unit_scale:

            @jit
            def _encode_unit_p_scale_bytes():
                total_bytes = vec_rows * self.p_tile_n_blocks32
                with vf(mode="simd"):
                    full, _ = rr.update_mask(
                        FP8_REPACK_VECTOR_BYTES, elem_bits=INT8_ELEM_BITS
                    )
                    vreg_exponent_one = rr.vdups(
                        E8M0_SCALE_ONE_BYTE, dtypes.int8, mask=full
                    )
                    remaining = rr.mask_counter(total_bytes)
                    for offset in range(0, total_bytes, FP8_REPACK_VECTOR_BYTES):
                        mask, remaining = rr.update_mask(
                            remaining, elem_bits=INT8_ELEM_BITS
                        )
                        rr.vstore(
                            self.p_scale_compact_ub, offset, vreg_exponent_one, mask
                        )
                    rr.vmem_bar("vst_vld")

            _encode_unit_p_scale_bytes()
            return

        @jit
        def _encode_scale_correction_bytes():
            with vf(mode="simd"):
                rows_per_block = P_SCALE_ZN_M0_ROWS
                m1_per_subblock = vec_rows // rows_per_block
                mask_f32, _ = rr.update_mask(rows_per_block, elem_bits=FP32_ELEM_BITS)
                mask_e8m0, _ = rr.update_mask(
                    rows_per_block * MXFP8_SCALE_LANES_PER_GROUP,
                    elem_bits=INT8_ELEM_BITS,
                )
                for row_block in range(m1_per_subblock):
                    vreg_first_subtile_max_correction = rr.vload(
                        self.first_subtile_max_correction_ub, row_block * rows_per_block
                    )
                    vreg_exponent_pair = _e8m0_exponent_pairs(
                        vreg_first_subtile_max_correction, mask_f32, False
                    )
                    self._store_compact_scale_pairs(
                        vreg_exponent_pair, row_block, mask_e8m0
                    )
                rr.vmem_bar("vst_vld")

        _encode_scale_correction_bytes()

    def store_compact_p_scale(self, p_scale_l1, actual_m):
        aligned_m = (
            ceil_div(actual_m, SPLIT_M_TOTAL_ALIGNMENT) * SPLIT_M_TOTAL_ALIGNMENT
        )
        vec_rows = aligned_m // AIV_SUBBLOCK_COUNT
        slot = p_scale_l1.next_write()
        target = slot[:aligned_m, : slot.shape[1]]
        source = self.p_scale_compact_ub[:vec_rows, : self.p_tile_n_blocks32]
        mem_copy(
            target.partition(
                dim=0, part_id=self.subblock_idx, alignment=SPLIT_M_ALIGNMENT
            ),
            source,
        )


class DecodeVectorOutput(_VectorOutputBase):
    """Accumulate and normalize AIV-local rows; the kernel scatters merged rows."""


# ---- Decode kernel orchestration --------------------------------------------
@kernel
class FlashAttentionGQAMxfp8DecodeKernel(_DelayedPVConsumer):
    """ND-path Flash Attention kernel with delayed PV consumption.

    QK/softmax production runs two macro tiles ahead of PV consumption;
    the delay line carries metadata across the three pipeline slots.
    """

    def __init__(
        self,
        tile_cube_m,
        tile_vec_m,
        tile_n,
        tile_d,
        mask_mode,
        kv_layout,
        block_num,
        q_layout=QLayout.TND,
        paged=False,
        block_size=128,
        return_softmax_lse=False,
        kv_heads=1,
        enable_flash_decode=False,
        short_lse=False,
    ):
        self.short_lse = short_lse
        self.enable_flash_decode = enable_flash_decode
        self.kv_heads = kv_heads
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.mask_mode = mask_mode
        self.q_layout = q_layout
        # Current kernel indexing is TND-only: Q is [T, N_q, D], and all
        # token/head offsets and scale indexing rely on that contract.
        self.kv_layout = kv_layout
        self.paged = self.kv_layout != KVLayout.TND
        self.pa_bbnd = self.kv_layout == KVLayout.PA_BBND
        self.pa_nz = self.kv_layout == KVLayout.PA_NZ
        self.block_size = block_size
        self.return_softmax_lse = return_softmax_lse
        self.page_tile_n = min(tile_n, block_size)
        self.tiles_per_page = max(1, block_size // tile_n)
        self.pages_per_tile = (tile_n + block_size - 1) // block_size
        check_pa_tile(block_size, tile_n, self.paged)
        self.preload_num = SOFTMAX_DELAY_SLOTS

        # Separate QK slots feed the two softmax subtiles; PV has one slot.
        self.qk_ub_a = _make_ring_buffer(
            MemLoc.UB, shape=(tile_vec_m, tile_n), dtype=dtypes.float32, depth=1
        )
        self.qk_ub_b = _make_ring_buffer(
            MemLoc.UB, shape=(tile_vec_m, tile_n), dtype=dtypes.float32, depth=1
        )
        self.pv_ub = _make_ring_buffer(
            MemLoc.UB, shape=(tile_vec_m, tile_d), dtype=dtypes.float32, depth=1
        )

        # P tiles are triple-buffered across the delayed PV pipeline.
        self.p_l1_a = _make_ring_buffer(
            MemLoc.L1,
            shape=(tile_cube_m, tile_n),
            dtype=dtypes.float8_e4m3fn,
            depth=SOFTMAX_DELAY_SLOTS,
        )
        self.p_l1_b = _make_ring_buffer(
            MemLoc.L1,
            shape=(tile_cube_m, tile_n),
            dtype=dtypes.float8_e4m3fn,
            depth=SOFTMAX_DELAY_SLOTS,
        )

        # Q has two slots; K and V share one four-slot pool in execution order.
        self.q_l1 = _make_ring_buffer(
            MemLoc.L1,
            shape=(tile_cube_m + SPLIT_M_ALIGNMENT, tile_d),
            dtype=dtypes.float8_e4m3fn,
            depth=2,
            data_format="nz",
        )
        self.kv_l1 = _make_ring_buffer(
            MemLoc.L1, (self.tile_n, tile_d), dtypes.float8_e4m3fn, depth=4
        )
        self.k_l1 = _SharedL1View(
            self.kv_l1, _View((self.tile_n, tile_d), dtypes.float8_e4m3fn, "nz")
        )
        v_spec = _View(
            (self.tile_n, tile_d), dtypes.float8_e4m3fn, "nz" if self.pa_nz else "zn"
        )
        self.v_l1 = _SharedL1View(self.kv_l1, v_spec)

        # E8M0 scales use MXFP8_SCALE_LANES_PER_GROUP lanes per MXFP8_SCALE_GROUP_ELEMENTS-element reduction group.
        self.qk_scale_groups = _mx_groups64(tile_d)  # MX groups for Q/K reduce (D axis)
        self.v_scale_groups = _mx_groups64(self.tile_n)  # groups for one Cube K chunk
        self.p_scale_groups = _mx_groups64(
            tile_n
        )  # MX groups for P reduce (tile_n axis)
        self.q_scale_l1 = _make_ring_buffer(
            MemLoc.L1,
            shape=(tile_cube_m, 2 * self.qk_scale_groups),
            dtype=dtypes.float8_e8m0,
            depth=2,
            data_format="zn",
        )
        k_scale_spec = _View(
            (2 * self.qk_scale_groups, self.tile_n), dtypes.float8_e8m0, "nz"
        )
        v_scale_spec = _View(
            (2 * self.v_scale_groups, tile_d), dtypes.float8_e8m0, "nz"
        )
        scale_bytes = max(
            k_scale_spec.storage_bytes(mem_loc=MemLoc.L1),
            v_scale_spec.storage_bytes(mem_loc=MemLoc.L1),
        )
        self.kv_scale_l1 = _make_ring_buffer(MemLoc.L1, size_bytes=scale_bytes, depth=4)
        self.k_scale_l1 = _SharedL1View(self.kv_scale_l1, k_scale_spec)
        self.v_scale_l1 = _SharedL1View(self.kv_scale_l1, v_scale_spec)
        # P correction scales cross from AIV to AIC.
        self.p_scale_l1_a = _make_ring_buffer(
            MemLoc.L1,
            shape=(tile_cube_m, 2 * self.p_scale_groups),
            dtype=dtypes.float8_e8m0,
            depth=SOFTMAX_DELAY_SLOTS,
            data_format="zn",
        )
        self.p_scale_l1_b = _make_ring_buffer(
            MemLoc.L1,
            shape=(tile_cube_m, 2 * self.p_scale_groups),
            dtype=dtypes.float8_e8m0,
            depth=SOFTMAX_DELAY_SLOTS,
            data_format="zn",
        )

        self.block_idx = get_block_idx()
        self.subblock_idx = get_subblock_id()
        self.block_num = Int64(block_num)

        self.matmul = DecodeCubeMatmul(self.tile_cube_m, self.tile_n, self.tile_d)
        self.matmul.raw_pa_v = self.pa_nz
        self.vector_softmax = DecodeVectorSoftmax(
            self.tile_vec_m, self.tile_n, self.mask_mode, self.subblock_idx
        )
        self.vector_output = DecodeVectorOutput(
            self.tile_vec_m, self.tile_d, self.subblock_idx
        )
        if return_softmax_lse:
            self.softmax_lse_output = SoftmaxLseOutput(self.tile_vec_m)
            # MTE3 reads each short FP32 source row from a 32-byte origin.
            # 64 padded rows use 2048B; this replaces the old 32B scalar slot.
            self.lse_dma_row_elems = 8
            self.lse_packed_ub = _make_buffer(
                MemLoc.UB, (tile_vec_m, self.lse_dma_row_elems), dtypes.float32
            )
        self.block_info = BlockInfo(self.tile_cube_m, self.tile_n, self.mask_mode)
        if enable_flash_decode:
            # Producer UB adds 4096B: two 64-row, 32B-aligned statistic arrays.
            # Audit peak UB from compiled offsets; shape and LSE affect allocation.
            self.fd_sum_pack = _make_buffer(
                MemLoc.UB, (tile_vec_m, FD_STAT_ROW_ELEMENTS), dtypes.float32
            )
            self.fd_max_pack = _make_buffer(
                MemLoc.UB, (tile_vec_m, FD_STAT_ROW_ELEMENTS), dtypes.float32
            )
        self.default_worker_count = block_num * AIV_SUBBLOCK_COUNT

    @jit
    def _output_token_start(self, batch, seqused_q):
        begin = Int64(0)
        for previous in range(batch):
            begin += seqused_q[previous]
        return begin

    @jit
    def _store_compact_defaults(
        self, attn_out, softmax_lse_gm, seqused_q, seqused_kv, group
    ):
        # With one head, batch ownership is already the complete work mapping.
        # Use the direct omitted-row check without an extra head-writing loop.
        if const_expr(attn_out.shape[1] == 1):
            heads = attn_out.shape[1]
            worker = self.block_idx * AIV_SUBBLOCK_COUNT + self.subblock_idx
            workers = self.block_num * AIV_SUBBLOCK_COUNT
            for task in range(worker, seqused_q.shape[0] * heads, workers):
                batch, head = task // heads, task % heads
                q_length, kv_length = seqused_q[batch], seqused_kv[batch]
                omitted = Int64(0)
                if kv_length == 0:
                    omitted = q_length
                else:
                    if const_expr(self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL):
                        masked = dynamic_max(Int64(0), q_length - kv_length)
                        skipped = masked * group // self.tile_cube_m * self.tile_cube_m
                        omitted = dynamic_min(
                            q_length,
                            dynamic_max(
                                Int64(0), (skipped + group - 1 - head % group) // group
                            ),
                        )
                if omitted > 0:
                    begin = self._output_token_start(batch, seqused_q)
                    self._store_default_range(
                        attn_out, softmax_lse_gm, head, begin, omitted
                    )
        else:
            heads = attn_out.shape[1]
            worker = self.block_idx * AIV_SUBBLOCK_COUNT + self.subblock_idx
            # Batch ownership is known from the launch shape. Avoid a modulo
            # when every worker already owns an independent batch.
            if const_expr(seqused_q.shape[0] >= self.default_worker_count):
                batch_worker = worker
                batch_workers = self.default_worker_count
            else:
                batch_worker = worker % seqused_q.shape[0]
                batch_workers = seqused_q.shape[0]
            # Surplus head workers must not repeat length checks for empty work.
            if const_expr(seqused_q.shape[0] * heads < self.default_worker_count):
                if worker >= seqused_q.shape[0] * heads:
                    batch_worker = Int64(seqused_q.shape[0])
            for batch in range(batch_worker, seqused_q.shape[0], batch_workers):
                q_length, kv_length = seqused_q[batch], seqused_kv[batch]
                skipped = Int64(0)
                if const_expr(self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL):
                    masked = dynamic_max(Int64(0), q_length - kv_length)
                    skipped = masked * group // self.tile_cube_m * self.tile_cube_m
                if kv_length == 0 or skipped > 0:
                    # Calculate head ownership only when a write is required.
                    head_worker = Int64(0)
                    head_workers = Int64(1)
                    if const_expr(seqused_q.shape[0] < self.default_worker_count):
                        head_worker = worker // seqused_q.shape[0]
                        active_workers = self.default_worker_count
                        if const_expr(
                            seqused_q.shape[0] * heads < self.default_worker_count
                        ):
                            active_workers = seqused_q.shape[0] * heads
                        head_workers = ceil_div(
                            active_workers - batch_worker, batch_workers
                        )
                    begin = self._output_token_start(batch, seqused_q)
                    for head in range(head_worker, heads, head_workers):
                        omitted = q_length
                        if kv_length != 0:
                            omitted = dynamic_min(
                                q_length,
                                dynamic_max(
                                    Int64(0),
                                    (skipped + group - 1 - head % group) // group,
                                ),
                            )
                        if omitted > 0:
                            self._store_default_range(
                                attn_out, softmax_lse_gm, head, begin, omitted
                            )

    @jit
    def _clear_padding_chunk(
        self, attn_out, seqused_q, seqused_kv, scratch, cursor, end
    ):
        # Single 16 KiB (D128) DMA, using only accumulator rows that the
        # small-M guard proves are dead throughout attention. No new UB.
        heads = attn_out.shape[1]
        capacity = self.tile_cube_m * heads
        batch = cursor // capacity
        live = seqused_q[batch] * heads
        if seqused_kv[batch] == 0:
            live = Int64(0)
        cursor = dynamic_max(cursor, batch * capacity + live)
        stop = dynamic_min(end, (batch + 1) * capacity)
        rows = dynamic_min(Int64(self.tile_vec_m), dynamic_max(Int64(0), stop - cursor))
        if rows > 0:
            flat = attn_out.view((attn_out.shape[0] * heads, attn_out.shape[2]))
            target = tile_slice(
                flat[cursor: cursor + rows, None],
                (self.tile_vec_m, self.tile_d),
                (0, 0),
            )  # fmt: skip
            mem_copy(target, scratch[:rows, : attn_out.shape[2]])
        cursor += rows
        return cursor

    # ---- Tensor views and address helpers -------------------------------------

    @jit
    def _consume_delayed_pv(
        self,
        attn_out,
        softmax_lse_gm,
        value,
        v_scale_gm,
        block_table,
        seqused_q,
        seqused_kv,
        p_scale,
        head_group_num,
        delay_batch_idx,
        delay_head_idx,
        delay_m_idx,
        delay_n_block_a,
        delay_n_block_b,
        delay_has_second,
        delay_is_first_macro,
        delay_is_last_macro,
        delay_tile_triple,
        delay_m_axis_triple,
        delay_actual_m,
        delay_fd_slot,
        delay_output_start,
        delay_kv_token_start,
        delay_v_scale_start,
    ):
        # Complete PV and output update for one previously produced macro.
        delay_kv_head_idx = delay_head_idx // head_group_num
        delay_seqlen_k = seqused_kv[delay_batch_idx]
        # 生产QK时已算好的地址随DelayLine传递，PV不再重复扫描batch前缀。

        delay_vec_rows = (
            ceil_div(delay_actual_m, SPLIT_M_TOTAL_ALIGNMENT) * SPLIT_M_ALIGNMENT
        )
        accumulator = self.matmul.begin_pv(delay_actual_m)

        # PV for n_block_a + n_block_b with L0C accumulation
        self._compute_pv_tile(
            self.p_l1_a,
            self.p_scale_l1_a,
            value,
            v_scale_gm,
            block_table,
            delay_batch_idx,
            delay_kv_head_idx,
            delay_kv_token_start,
            delay_v_scale_start,
            delay_seqlen_k,
            delay_n_block_a,
            True,
            accumulator,
            delay_actual_m,
        )

        if delay_has_second:
            self._compute_pv_tile(
                self.p_l1_b,
                self.p_scale_l1_b,
                value,
                v_scale_gm,
                block_table,
                delay_batch_idx,
                delay_kv_head_idx,
                delay_kv_token_start,
                delay_v_scale_start,
                delay_seqlen_k,
                delay_n_block_b,
                False,
                accumulator,
                delay_actual_m,
            )

        self.matmul.store_pv_result(self.pv_ub, delay_actual_m)

        # Select the produced PV slot once; output helpers receive a Tensor.
        pv_slot = self.pv_ub.next_read()
        pv_tile = pv_slot[:delay_vec_rows, : self.tile_d]
        if delay_is_last_macro:
            delay_seqlen_q = seqused_q[delay_batch_idx]
            delay_q_token_start = delay_output_start
            if delay_is_first_macro:
                self._finalize_outputs(
                    attn_out,
                    softmax_lse_gm,
                    delay_q_token_start,
                    delay_seqlen_q,
                    delay_head_idx,
                    delay_m_idx,
                    delay_m_axis_triple,
                    p_scale,
                    final_pv=pv_tile,
                    kv_length=delay_seqlen_k,
                    fd_slot=delay_fd_slot,
                )
            else:
                if delay_tile_triple == 0:
                    mem_copy(
                        self.vector_output.final_correction_ub,
                        self.vector_softmax.prev_running_max_correction_ub[0],
                    )
                elif delay_tile_triple == 1:
                    mem_copy(
                        self.vector_output.final_correction_ub,
                        self.vector_softmax.prev_running_max_correction_ub[1],
                    )
                else:
                    mem_copy(
                        self.vector_output.final_correction_ub,
                        self.vector_softmax.prev_running_max_correction_ub[2],
                    )
                self._finalize_outputs(
                    attn_out,
                    softmax_lse_gm,
                    delay_q_token_start,
                    delay_seqlen_q,
                    delay_head_idx,
                    delay_m_idx,
                    delay_m_axis_triple,
                    p_scale,
                    final_pv=pv_tile,
                    correction=self.vector_output.final_correction_ub,
                    kv_length=delay_seqlen_k,
                    fd_slot=delay_fd_slot,
                )
        else:
            if delay_is_first_macro:
                self.vector_output.init_output(pv_tile)
            else:
                if delay_tile_triple == 0:
                    self.vector_output.update_output(
                        pv_tile, self.vector_softmax.prev_running_max_correction_ub[0]
                    )
                elif delay_tile_triple == 1:
                    self.vector_output.update_output(
                        pv_tile, self.vector_softmax.prev_running_max_correction_ub[1]
                    )
                else:
                    self.vector_output.update_output(
                        pv_tile, self.vector_softmax.prev_running_max_correction_ub[2]
                    )

    @jit
    def _batch_tnd_offsets(self, batch_idx, seqused_q, seqused_kv):
        """Return compact TND offsets for Q/KV tokens and V-scale groups."""
        q_token_start = batch_idx * self.tile_cube_m
        kv_token_start = Int64(0)
        v_scale_start = Int64(0)
        if const_expr(not self.paged):
            for previous_batch in range(0, batch_idx):
                previous_kv_length = seqused_kv[previous_batch]
                kv_token_start += previous_kv_length
                v_scale_start += ceil_div(
                    previous_kv_length, MXFP8_SCALE_GROUP_ELEMENTS
                )
        return q_token_start, kv_token_start, v_scale_start

    @jit
    def _load_merged_query(
        self, q_slot, query, q_token_start, q_length, head_idx, group_num, m_idx
    ):
        row_start = m_idx * self.tile_cube_m
        rows = dynamic_min(Int64(self.tile_cube_m), q_length * group_num - row_start)
        if const_expr(self.kv_heads == 1):
            flat = query.view((query.shape[0] * query.shape[1], query.shape[2]))
            begin = q_token_start * group_num + row_start
            source = tile_slice(
                flat[begin: begin + rows, None],
                (self.tile_cube_m, self.tile_d),
                (0, 0),
            )  # fmt: skip
            mem_copy(q_slot, source)
            return
        if const_expr(query.shape[1] == self.kv_heads):
            by_head = query.permute((1, 0, 2))
            begin = q_token_start + row_start
            source = tile_slice(
                by_head[head_idx, begin: begin + rows, None],
                (self.tile_cube_m, self.tile_d),
                (0, 0),
            )  # fmt: skip
            mem_copy(q_slot, source)
            return
        group_count = query.shape[1] // self.kv_heads
        # Split a row interval into an optional partial token on either side
        # and one strided GM view covering every complete token in between.
        prefix = Int64(0)
        if const_expr(self.tile_cube_m % group_count != 0):
            prefix = dynamic_min(
                rows, (group_count - row_start % group_count) % group_count
            )
        full_tokens = (rows - prefix) // group_count
        if full_tokens >= 2:
            if const_expr(self.tile_cube_m % group_count != 0):
                if prefix > 0:
                    self._load_merged_query_segments(
                        q_slot,
                        query,
                        q_token_start,
                        head_idx,
                        group_num,
                        row_start,
                        prefix,
                    )
            token_begin = q_token_start + (row_start + prefix) // group_count
            end = prefix + full_tokens * group_count
            source = tile_slice(
                query[
                    token_begin: token_begin + full_tokens,
                    head_idx: head_idx + group_count,
                    None,
                ],
                (self.tile_cube_m, group_count, self.tile_d),
                (0, 0, 0),
            )  # fmt: skip
            if const_expr(self.tile_cube_m % group_count == 0):
                mem_copy(q_slot, source.flatten(0, 1))
            else:
                mem_copy(q_slot[prefix:end, None], source.flatten(0, 1))
                if end < rows:
                    self._load_merged_query_segments(
                        q_slot[end:rows, None],
                        query,
                        q_token_start,
                        head_idx,
                        group_num,
                        row_start + end,
                        rows - end,
                    )
        else:
            self._load_merged_query_segments(
                q_slot, query, q_token_start, head_idx, group_num, row_start, rows
            )

    @jit
    def _load_merged_query_segments(
        self, q_slot, query, q_token_start, head_idx, group_num, row_start, rows
    ):
        copied = Int64(0)
        for segment in range(ceil_div(row_start % group_num + rows, group_num)):
            row = row_start + copied
            token, group = row // group_num, row % group_num
            take = dynamic_min(group_num - group, rows - copied)
            source = tile_slice(
                query[
                    q_token_start + token,
                    head_idx + group: head_idx + group + take,
                    None,
                ],
                (self.tile_cube_m, self.tile_d),
                (0, 0),
            )  # fmt: skip
            mem_copy(q_slot[copied: copied + take, None], source)  # fmt: skip
            copied += take

    @jit
    def _query_scale_merged_tile(
        self, q_scale_gm, q_token_start, q_length, head_idx, group_num, m_idx
    ):
        scale = q_scale_gm.view(
            (
                q_scale_gm.shape[0],
                q_scale_gm.shape[1] * q_scale_gm.shape[2],
                self.qk_scale_groups,
                2,
            )
        )
        row_start = q_token_start * group_num + m_idx * self.tile_cube_m
        rows = dynamic_min(
            Int64(self.tile_cube_m), q_length * group_num - m_idx * self.tile_cube_m
        )
        return tile_slice(
            scale[head_idx // group_num, row_start: row_start + rows, None, None],
            (self.tile_cube_m, self.qk_scale_groups, 2),
            (0, 0, 0),
        )  # fmt: skip

    # ---- GM-to-L1 staging -----------------------------------------------------

    # ---- QK/PV tile compute ---------------------------------------------------

    @jit
    def _compute_pv_tile(
        self,
        p_l1,
        p_scale_l1,
        value,
        v_scale_gm,
        block_table,
        batch_idx,
        kv_head_idx,
        kv_token_start,
        v_scale_start,
        kv_length,
        n_block,
        init,
        accumulator=None,
        actual_m=None,
    ):
        self._load_value_and_scale_tile(
            value,
            v_scale_gm,
            block_table,
            batch_idx,
            kv_head_idx,
            kv_token_start,
            v_scale_start,
            kv_length,
            n_block,
        )
        # PV 的 MX 归约维按 64 对齐；页内 scale 原点保持不变。
        valid_k = dynamic_min(
            Int64(self.tile_n),
            kv_length - n_block * self.tile_n,
        )
        aligned_k = (
            ceil_div(valid_k, MXFP8_SCALE_GROUP_ELEMENTS) * MXFP8_SCALE_GROUP_ELEMENTS
        )
        self.matmul.compute_pv(
            p_l1.next_read()[:actual_m, : self.tile_n],
            self.v_l1.next_read(),
            p_scale_l1.next_read()[:actual_m, : 2 * self.p_scale_groups],
            self.v_scale_l1.next_read(),
            init=init,
            actual_k=aligned_k,
            accumulator=accumulator,
        )

    # ---- Output views and finalization ---------------------------------------

    @jit
    def _store_merged_output(
        self, attn_out, q_token_start, head_idx, group_num, row_start, valid_rows
    ):
        if const_expr(self.kv_heads == 1):
            flat = attn_out.view(
                (attn_out.shape[0] * attn_out.shape[1], attn_out.shape[2])
            )
            begin = q_token_start * group_num + row_start
            dst = tile_slice(
                flat[begin: begin + valid_rows, None],
                (self.tile_vec_m, self.tile_d),
                (0, 0),
            )  # fmt: skip
            mem_copy(
                dst, self.vector_output.cast_ub_bf16[:valid_rows, : attn_out.shape[2]]
            )
            return
        if const_expr(attn_out.shape[1] == self.kv_heads):
            by_head = attn_out.permute((1, 0, 2))
            begin = q_token_start + row_start
            dst = tile_slice(
                by_head[head_idx, begin: begin + valid_rows, None],
                (self.tile_vec_m, self.tile_d),
                (0, 0),
            )  # fmt: skip
            mem_copy(
                dst, self.vector_output.cast_ub_bf16[:valid_rows, : attn_out.shape[2]]
            )
            return
        group_count = attn_out.shape[1] // self.kv_heads
        prefix = dynamic_min(
            valid_rows, (group_count - row_start % group_count) % group_count
        )
        tokens = (valid_rows - prefix) // group_count
        if tokens >= 2:
            if prefix > 0:
                self._store_merged_output_segments(
                    attn_out, q_token_start, head_idx, group_num, row_start, prefix
                )
            token_begin = q_token_start + (row_start + prefix) // group_count
            end = prefix + tokens * group_count
            if const_expr(attn_out.shape[2] == self.tile_d):
                width = group_count * self.tile_d
                flat = attn_out.view(
                    (attn_out.shape[0], attn_out.shape[1] * self.tile_d)
                )
                column = head_idx * self.tile_d
                dst = tile_slice(
                    flat[token_begin: token_begin + tokens, column: column + width],
                    (self.tile_vec_m, width),
                    (0, 0),
                )  # fmt: skip
                src = reinterpret(
                    self.vector_output.cast_ub_bf16,
                    shape=(tokens, width),
                    stride=(width, 1),
                    offset=prefix * self.tile_d * 2,
                )
                mem_copy(dst, src)
            else:
                self._store_padded_token_block(
                    attn_out, token_begin, head_idx, group_count, prefix, tokens
                )
            if end < valid_rows:
                self._store_merged_output_segments(
                    attn_out,
                    q_token_start,
                    head_idx,
                    group_num,
                    row_start + end,
                    valid_rows - end,
                    source_row=end,
                )
        else:
            self._store_merged_output_segments(
                attn_out, q_token_start, head_idx, group_num, row_start, valid_rows
            )

    @jit
    def _store_padded_token_block(
        self, attn_out, token_begin, head_idx, group_count, source_row, tokens
    ):
        # D=72 occupies 128 BF16 elements per normalized UB row. Pack two
        # heads at a time so every vector-store origin is 32-byte aligned.
        d = attn_out.shape[2]
        width = group_count * d
        pitch = ((width + 15) // 16) * 16
        packed = reinterpret(
            self.vector_output.res_o_ub,
            dtype=dtypes.bfloat16,
            shape=(self.tile_vec_m * self.tile_d,),
            stride=(1,),
            offset=self.tile_vec_m * self.tile_d * 2,
        )
        with vf(mode="simd"):
            full, _ = rr.update_mask(128, elem_bits=16)
            rest, _ = rr.update_mask(2 * d - 128, elem_bits=16)
            one, _ = rr.update_mask(d, elem_bits=16)
            lane = rr.varange(0, dtypes.uint16)
            second = rr.vges(lane, dtypes.uint16(d), mask=full)
            shifted = rr.vadds(lane, dtypes.uint16(self.tile_d - d), mask=full)
            first_indices = rr.vselect(shifted, lane, cond_mask=second)
            last_indices = rr.vadds(
                lane, dtypes.uint16(128 + self.tile_d - d), mask=rest
            )
            for t in range(tokens):
                src = (source_row + t * group_count) * self.tile_d
                dst = t * pitch
                for pair in range(group_count // 2):
                    base = dtypes.uint16(src + 2 * pair * self.tile_d)
                    left = rr.vgather(
                        self.vector_output.cast_ub_bf16,
                        rr.vadds(first_indices, base, mask=full),
                        mask=full,
                    )
                    right = rr.vgather(
                        self.vector_output.cast_ub_bf16,
                        rr.vadds(last_indices, base, mask=rest),
                        mask=rest,
                    )
                    rr.vstore(packed, dst + 2 * pair * d, left, full)
                    rr.vstore(packed, dst + 2 * pair * d + 128, right, rest)
                if const_expr(group_count % 2 != 0):
                    last = rr.vload(
                        self.vector_output.cast_ub_bf16,
                        src + (group_count - 1) * self.tile_d,
                    )
                    rr.vstore(packed, dst + (group_count - 1) * d, last, one)
            rr.vmem_bar("vst_vld")
        flat = attn_out.view((attn_out.shape[0], attn_out.shape[1] * d))
        column = head_idx * d
        target = tile_slice(
            flat[token_begin: token_begin + tokens, column: column + width],
            (self.tile_vec_m, width),
            (0, 0),
        )  # fmt: skip
        source = reinterpret(packed, shape=(tokens, width), stride=(pitch, 1))
        mem_copy(target, source)

    @jit
    def _store_merged_output_segments(
        self,
        attn_out,
        q_token_start,
        head_idx,
        group_num,
        row_start,
        valid_rows,
        source_row=0,
    ):
        copied = Int64(0)
        for segment in range(ceil_div(row_start % group_num + valid_rows, group_num)):
            row = row_start + copied
            token, group = row // group_num, row % group_num
            take = dynamic_min(group_num - group, valid_rows - copied)
            dst = tile_slice(
                attn_out[
                    q_token_start + token,
                    head_idx + group: head_idx + group + take,
                    None,
                ],
                (self.tile_vec_m, self.tile_d),
                (0, 0),
            )  # fmt: skip
            src = self.vector_output.cast_ub_bf16[
                source_row + copied: source_row + copied + take, : attn_out.shape[2]
            ]  # fmt: skip
            mem_copy(dst, src)
            copied += take

    @jit
    def _finalize_merged_rows(
        self,
        attn_out,
        softmax_lse_gm,
        q_token_start,
        q_length,
        head_idx,
        m_idx,
        p_scale,
        maximum,
        denominator,
        final_pv,
        correction,
        fd_slot=-1,
        kv_length=0,
    ):
        if const_expr(self.enable_flash_decode):
            if fd_slot >= 0:
                group = attn_out.shape[1] // self.kv_heads
                logical_m = dynamic_min(
                    Int64(self.tile_cube_m), q_length * group - m_idx * self.tile_cube_m
                )
                half = ceil_div(logical_m, SPLIT_M_TOTAL_ALIGNMENT) * SPLIT_M_ALIGNMENT
                offset = self.subblock_idx * half
                rows = dynamic_max(Int64(0), dynamic_min(half, logical_m - offset))
                self._save_fd_partial(
                    fd_slot, offset, rows, maximum, denominator, final_pv, correction
                )
            else:
                self._finalize_whole_merged_rows(
                    attn_out,
                    softmax_lse_gm,
                    q_token_start,
                    q_length,
                    head_idx,
                    m_idx,
                    p_scale,
                    maximum,
                    denominator,
                    final_pv,
                    correction,
                    kv_length,
                )
        else:
            self._finalize_whole_merged_rows(
                attn_out,
                softmax_lse_gm,
                q_token_start,
                q_length,
                head_idx,
                m_idx,
                p_scale,
                maximum,
                denominator,
                final_pv,
                correction,
                kv_length,
            )

    @jit
    def _finalize_whole_merged_rows(
        self,
        attn_out,
        softmax_lse_gm,
        q_token_start,
        q_length,
        head_idx,
        m_idx,
        p_scale,
        maximum,
        denominator,
        final_pv,
        correction,
        kv_length,
    ):
        group_num = attn_out.shape[1] // self.kv_heads
        logical_m = dynamic_min(
            Int64(self.tile_cube_m), q_length * group_num - m_idx * self.tile_cube_m
        )
        vec_rows = ceil_div(logical_m, SPLIT_M_TOTAL_ALIGNMENT) * SPLIT_M_ALIGNMENT
        local_offset = self.subblock_idx * vec_rows
        valid_rows = dynamic_max(
            Int64(0), dynamic_min(vec_rows, logical_m - local_offset)
        )
        row_start = m_idx * self.tile_cube_m + local_offset
        if const_expr(final_pv is not None and correction is None):
            if const_expr(self.vector_output.check_invalid_rows):
                if row_start < (q_length - kv_length) * group_num:
                    self.vector_output.normalize_single_pv(
                        final_pv, denominator, maximum, valid_rows, True
                    )
                else:
                    self.vector_output.normalize_single_pv(
                        final_pv, denominator, maximum, valid_rows, False
                    )
            else:
                self.vector_output.normalize_single_pv(
                    final_pv, denominator, maximum, valid_rows, False
                )
        else:
            self.vector_output._normalize_rows(
                denominator,
                valid_rows,
                final_pv,
                correction,
                maximum,
                row_start < (q_length - kv_length) * group_num,
            )
        if valid_rows > 0:
            self._store_merged_output(
                attn_out, q_token_start, head_idx, group_num, row_start, valid_rows
            )
        if const_expr(self.return_softmax_lse):
            # LSE has [head, token] layout, unlike the TND attention output.
            with vf(mode="simd"):
                mask, _ = rr.update_mask(valid_rows, elem_bits=FP32_ELEM_BITS)
                mx = rr.vload(maximum, 0)
                sm = rr.vload(denominator, 0)
                invalid = rr.veqs(sm, 0.0, mask=mask)
                one = rr.vdups(1.0, dtypes.float32, mask=mask)
                safe_sum = rr.vselect(one, sm, cond_mask=invalid)
                lse = rr.vadd(mx, rr.vlog(safe_sum, mask=mask), mask=mask)
                zero = rr.vdups(0.0, dtypes.float32, mask=mask)
                infinity = rr.vdiv(one, zero, mask=mask)
                lse = rr.vselect(infinity, lse, cond_mask=invalid)
                rr.vstore(self.softmax_lse_output.values, 0, lse, mask)
                rr.vmem_bar("vst_vld")
            if valid_rows > 0:
                self._store_merged_lse(
                    softmax_lse_gm,
                    q_token_start,
                    head_idx,
                    group_num,
                    row_start,
                    valid_rows,
                )

    @jit
    def _store_merged_lse(
        self, softmax_lse_gm, q_token_start, head_idx, group_num, row_start, valid_rows
    ):
        group_count = softmax_lse_gm.shape[0] // self.kv_heads
        if const_expr(group_count == 1):
            # A single Q head already has contiguous token values in UB.
            token_begin = q_token_start + row_start
            target = tile_slice(
                softmax_lse_gm[head_idx, token_begin: token_begin + valid_rows],
                (1, self.tile_vec_m),
                (0, 0),
            )  # fmt: skip
            source = reinterpret(
                self.softmax_lse_output.values,
                shape=(1, self.tile_vec_m),
                stride=(self.tile_vec_m, 1),
            )
            mem_copy(target, source[None, :valid_rows])
            return

        token_segments = ceil_div(row_start % group_num + valid_rows, group_num)
        heads = dynamic_min(Int64(group_count), valid_rows)
        # The existing short path already needs at most three DMA submissions.
        # Longer ranges can be packed as a head-major rectangle. Split at the
        # two partial-token boundaries to obtain at most three GM rectangles.
        if const_expr(
            not self.short_lse and group_count > 3 and group_count <= self.tile_vec_m
        ):
            if token_segments > 3 and heads > 3:
                phase = row_start % group_num
                stop = (phase + valid_rows) % group_num
                pitch = ceil_div(token_segments, 8) * 8
                with vf(mode="simd"):
                    mask, _ = rr.update_mask(valid_rows, elem_bits=32)
                    positions = rr.vadds(
                        rr.varange(0, dtypes.uint32), dtypes.uint32(phase), mask=mask
                    )
                    # positions < 128, group_count <= 64. The upward-rounded
                    # Q16 reciprocal gives exact unsigned division on this interval.
                    recip = (65536 + group_count - 1) // group_count
                    products = rr.vmuls(positions, dtypes.uint32(recip), mask=mask)
                    tokens = rr.vshr(products, 16, mask=mask)
                    h = rr.vsub(
                        positions,
                        rr.vmuls(tokens, dtypes.uint32(group_count), mask=mask),
                        mask=mask,
                    )
                    later = rr.vlts(h, dtypes.uint32(phase), mask=mask)
                    relative = rr.vselect(
                        rr.vadds(tokens, dtypes.uint32(4294967295), mask=mask),
                        tokens,
                        cond_mask=later,
                    )
                    indices = rr.vadd(
                        rr.vmuls(h, dtypes.uint32(pitch), mask=mask),
                        relative,
                        mask=mask,
                    )
                    values = rr.vload(self.softmax_lse_output.values, 0)
                    rr.vscatter(self.lse_packed_ub, values, indices, mask=mask)
                    rr.vmem_bar("vst_vld")
                low = dynamic_min(phase, stop)
                high = dynamic_max(phase, stop)
                token_base = q_token_start + row_start // group_num
                self._store_lse_head_span(
                    softmax_lse_gm,
                    head_idx,
                    group_num,
                    phase,
                    valid_rows,
                    token_base,
                    pitch,
                    Int64(0),
                    low,
                )
                self._store_lse_head_span(
                    softmax_lse_gm,
                    head_idx,
                    group_num,
                    phase,
                    valid_rows,
                    token_base,
                    pitch,
                    low,
                    high,
                )
                self._store_lse_head_span(
                    softmax_lse_gm,
                    head_idx,
                    group_num,
                    phase,
                    valid_rows,
                    token_base,
                    pitch,
                    high,
                    group_num,
                )
            else:
                self._store_short_lse(
                    softmax_lse_gm,
                    q_token_start,
                    head_idx,
                    group_num,
                    row_start,
                    valid_rows,
                    token_segments,
                    heads,
                )
        else:
            self._store_short_lse(
                softmax_lse_gm,
                q_token_start,
                head_idx,
                group_num,
                row_start,
                valid_rows,
                token_segments,
                heads,
            )

    @jit
    def _store_short_lse(
        self,
        softmax_lse_gm,
        q_token_start,
        head_idx,
        group_num,
        row_start,
        valid_rows,
        token_segments,
        heads,
    ):
        elem_bytes = FP32_ELEM_BITS // INT8_ELEM_BITS
        if token_segments > heads:
            for first in range(heads):
                tokens = ceil_div(valid_rows - first, group_num)
                row = row_start + first
                token_begin = q_token_start + row // group_num
                # Compact each head's tokens so MTE3 issues one long burst.
                with vf(mode="simd"):
                    mask, _ = rr.update_mask(tokens, elem_bits=FP32_ELEM_BITS)
                    indices = rr.vmuls(
                        rr.varange(0, dtypes.uint32),
                        dtypes.uint32(group_num),
                        mask=mask,
                    )
                    indices = rr.vadds(indices, dtypes.uint32(first), mask=mask)
                    values = rr.vgather(
                        self.softmax_lse_output.values, indices, mask=mask
                    )
                    rr.vstore(self.lse_packed_ub, 0, values, mask)
                    rr.vmem_bar("vst_vld")
                target = tile_slice(
                    softmax_lse_gm[
                        head_idx + row % group_num, token_begin: token_begin + tokens
                    ],
                    (1, self.tile_vec_m),
                    (0, 0),
                )  # fmt: skip
                dense = reinterpret(
                    self.lse_packed_ub,
                    shape=(1, self.tile_vec_m),
                    stride=(self.tile_vec_m, 1),
                )
                mem_copy(target, dense[None, :tokens])
        else:
            # Pack once. Every possible DMA source origin is aligned, including
            # the first row of a partial token/group at either AIV boundary.
            with vf(mode="simd"):
                mask, _ = rr.update_mask(valid_rows, elem_bits=FP32_ELEM_BITS)
                values = rr.vload(self.softmax_lse_output.values, 0)
                indices = rr.vmuls(
                    rr.varange(0, dtypes.uint32), self.lse_dma_row_elems, mask=mask
                )
                rr.vscatter(self.lse_packed_ub, values, indices, mask=mask)
                rr.vmem_bar("vst_vld")

            copied = Int64(0)
            for segment in range(token_segments):
                row = row_start + copied
                group = row % group_num
                token = q_token_start + row // group_num
                take = dynamic_min(group_num - group, valid_rows - copied)
                target = tile_slice(
                    softmax_lse_gm[head_idx + group: head_idx + group + take, token],
                    (self.tile_vec_m, 1),
                    (0, 0),
                )  # fmt: skip
                source = reinterpret(
                    self.lse_packed_ub,
                    shape=(take, 1),
                    stride=(self.lse_dma_row_elems, 1),
                    offset=copied * self.lse_dma_row_elems * elem_bytes,
                )
                mem_copy(target, source)
                copied += take

    @jit
    def _store_lse_head_span(
        self, out, head_idx, group, phase, rows, token_base, pitch, first, end
    ):
        if first < end:
            skip = Int64(0)
            if first < phase:
                skip = Int64(1)
            count = (phase + rows - 1 - first) // group + 1 - skip
            dst = tile_slice(
                out[
                    head_idx + first: head_idx + end,
                    token_base + skip: token_base + skip + count,
                ],
                (self.tile_vec_m, self.tile_vec_m),
                (0, 0),
            )  # fmt: skip
            src = reinterpret(
                self.lse_packed_ub,
                shape=(end - first, count),
                stride=(pitch, 1),
                offset=first * pitch * 4,
            )
            mem_copy(dst, src)

    @jit
    def _finalize_outputs(
        self,
        attn_out,
        softmax_lse_gm,
        q_token_start,
        q_length,
        head_idx,
        m_idx,
        state,
        p_scale,
        final_pv=None,
        correction=None,
        fd_slot=-1,
        kv_length=0,
    ):
        if state == 0:
            self._finalize_merged_rows(
                attn_out,
                softmax_lse_gm,
                q_token_start,
                q_length,
                head_idx,
                m_idx,
                p_scale,
                self.vector_softmax.softmax_max_ub[0],
                self.vector_softmax.softmax_sum_ub[0],
                final_pv,
                correction,
                fd_slot,
                kv_length,
            )
        elif state == 1:
            self._finalize_merged_rows(
                attn_out,
                softmax_lse_gm,
                q_token_start,
                q_length,
                head_idx,
                m_idx,
                p_scale,
                self.vector_softmax.softmax_max_ub[1],
                self.vector_softmax.softmax_sum_ub[1],
                final_pv,
                correction,
                fd_slot,
                kv_length,
            )
        else:
            self._finalize_merged_rows(
                attn_out,
                softmax_lse_gm,
                q_token_start,
                q_length,
                head_idx,
                m_idx,
                p_scale,
                self.vector_softmax.softmax_max_ub[2],
                self.vector_softmax.softmax_sum_ub[2],
                final_pv,
                correction,
                fd_slot,
                kv_length,
            )

    @jit
    def _setup_fd_merge(self):
        """Reuse dead FA storage; double FD inputs fit within 256 KiB at D256."""
        self.vector_output = DecodeVectorOutput(
            self.tile_vec_m, self.tile_d, self.subblock_idx
        )
        self.fd_part_input = _make_ring_buffer(
            MemLoc.UB, (self.tile_vec_m, self.tile_d), dtypes.float32, depth=2
        )
        self.fd_sum_input = _make_ring_buffer(
            MemLoc.UB, (self.tile_vec_m, FD_STAT_ROW_ELEMENTS), dtypes.float32, depth=2
        )
        self.fd_max_input = _make_ring_buffer(
            MemLoc.UB, (self.tile_vec_m, FD_STAT_ROW_ELEMENTS), dtypes.float32, depth=2
        )
        self.fd_running_max = _make_buffer(
            MemLoc.UB, (self.tile_vec_m,), dtypes.float32
        )
        self.fd_running_sum = _make_buffer(
            MemLoc.UB, (self.tile_vec_m,), dtypes.float32
        )
        if const_expr(self.return_softmax_lse):
            self.softmax_lse_output = SoftmaxLseOutput(self.tile_vec_m)
            self.lse_packed_ub = _make_buffer(
                MemLoc.UB, (self.tile_vec_m, self.lse_dma_row_elems), dtypes.float32
            )

    @jit
    def _merge_fd_task(
        self,
        attn_out,
        softmax_lse_gm,
        seqused_q,
        seqused_kv,
        heads,
        group,
        metadata,
        record,
        slot_base,
        p_scale,
    ):
        row_count = metadata[record + FD_ROWS]
        if row_count > 0:
            bn = metadata[record + FD_BN]
            batch, head = bn // heads, bn % heads
            m = metadata[record + FD_M]
            begin = metadata[record + FD_ROW_BEGIN]
            first = slot_base + metadata[record + FD_FIRST_SLOT]
            parts = metadata[record + FD_PARTS]
            token_start = self._output_token_start(batch, seqused_q)
            for chunk in range(ceil_div(row_count, self.tile_vec_m)):
                offset = chunk * self.tile_vec_m
                rows = dynamic_min(Int64(self.tile_vec_m), row_count - offset)
                row_start = m * self.tile_cube_m + begin + offset
                self._init_fd_accumulator(rows)
                self._load_fd_part(first, begin + offset, rows)
                for part in range(parts):
                    if part + 1 < parts:
                        self._load_fd_part(first + part + 1, begin + offset, rows)
                    self._update_fd_accumulator(rows)
                self.vector_output._normalize_rows(
                    self.fd_running_sum,
                    rows,
                    maximum_ub=self.fd_running_max,
                    check_block=row_start
                    < (seqused_q[batch] - seqused_kv[batch]) * group,
                )
                self._store_merged_output(
                    attn_out, token_start, head * group, group, row_start, rows
                )
                if const_expr(self.return_softmax_lse):
                    self.softmax_lse_output.compute(
                        self.fd_running_max, self.fd_running_sum, rows
                    )
                    self._store_merged_lse(
                        softmax_lse_gm,
                        token_start,
                        head * group,
                        group,
                        row_start,
                        rows,
                    )

    # ---- Kernel entry point and pipeline schedule -----------------------------

    def __call__(
        self,
        attn_out: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_scale_gm: Tensor,
        k_scale_gm: Tensor,
        v_scale_gm: Tensor,
        p_scale_gm: Tensor,
        scale: dtypes.float32,
        block_table: Tensor,
        seqused_kv: Tensor,
        seqused_q: Tensor,
        max_seqlen_q: dtypes.int64,
        softmax_lse_gm: Tensor | None = None,
        fd_output_gm: Tensor | None = None,
        fd_sum_gm: Tensor | None = None,
        fd_max_gm: Tensor | None = None,
        fd_metadata_gm: Tensor | None = None,
    ):
        self.copy_v_d = attn_out.shape[2]
        self.fd_output_gm = fd_output_gm
        self.fd_sum_gm = fd_sum_gm
        self.fd_max_gm = fd_max_gm
        self.vector_softmax.init_gather_indices()
        # The p quantization scale is a GM scalar.
        p_scale = p_scale_gm[0]
        self.vector_output.init_invalid_row_check(
            self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL, False, scale, p_scale
        )

        macro_global_idx = Int64(0)
        head_num_q = Int64(query.shape[1])
        if const_expr(self.pa_bbnd):
            head_num_kv = Int64(key.shape[2])
        else:
            head_num_kv = Int64(key.shape[1])
        head_group_num = head_num_q // head_num_kv
        seqlen_q_max = Int64(max_seqlen_q)
        clear_scratch = reinterpret(
            self.vector_output.res_o_ub,
            shape=(self.tile_vec_m, self.tile_d),
            dtype=dtypes.bfloat16,
            offset=(self.tile_vec_m // 2) * self.tile_d * 4,
        )
        overlap_clear = Int64(0)
        clear_cursor, clear_end = Int64(0), Int64(0)
        # Compact output has no capacity padding. Initialize only valid rows
        # omitted by empty KV or wholly masked causal tiles.
        self._store_compact_defaults(
            attn_out, softmax_lse_gm, seqused_q, seqused_kv, head_group_num
        )

        delayed_metadata = DelayLineGroup(
            self.preload_num,
            "batch_idx",
            "head_idx",
            "m_idx",
            "n_block_a",
            "n_block_b",
            "has_second",
            "is_first_macro",
            "is_last_macro",
            "tile_triple",
            "m_axis_triple",
            "actual_m",
            "fd_slot",
            "output_start",
            "kv_token_start",
            "v_scale_start",
        )

        sections = fd_metadata_gm[SECTION_COUNT]
        fd_active = fd_metadata_gm[HAS_FD]
        m_tiles_per_head = ceil_div(seqlen_q_max * head_group_num, self.tile_cube_m)
        active_m_tile_idx = Int64(0)
        # 连续task区间内batch单调递增。每核最多扫描一次前缀，后续递增维护。
        offset_batch = Int64(0)
        output_start = Int64(0)
        kv_token_start = Int64(0)
        v_scale_start = Int64(0)

        for section in range(sections):
            record = (1 + section * self.block_num + self.block_idx) * METADATA_STRIDE
            start_tile_idx = (
                fd_metadata_gm[record + FA_START_BN] * m_tiles_per_head
                + fd_metadata_gm[record + FA_START_M]
            )
            end_boundary_tile = (
                fd_metadata_gm[record + FA_END_BN] * m_tiles_per_head
                + fd_metadata_gm[record + FA_END_M]
            )
            first_macro = fd_metadata_gm[record + FA_START_S2]
            last_macro = fd_metadata_gm[record + FA_END_S2]
            first_slot = (
                section * 2 * self.block_num + fd_metadata_gm[record + FA_FIRST_SLOT]
            )
            end_tile_idx = end_boundary_tile
            if last_macro > 0:
                end_tile_idx += 1
            bn_stride = dynamic_max(Int64(1), m_tiles_per_head)
            for bn in range(
                start_tile_idx // bn_stride, ceil_div(end_tile_idx, bn_stride)
            ):
                batch_idx, kv_head_idx = bn // head_num_kv, bn % head_num_kv
                head_idx = kv_head_idx * head_group_num
                seqlen_q_actual = seqused_q[batch_idx]
                base_tile = bn * m_tiles_per_head
                first_m = dynamic_max(Int64(0), start_tile_idx - base_tile)
                stop_m = dynamic_min(
                    ceil_div(seqlen_q_actual * head_group_num, self.tile_cube_m),
                    end_tile_idx - base_tile,
                )
                for m_idx in range(first_m, stop_m):
                    tile_idx = base_tile + m_idx
                    seqlen_k = seqused_kv[batch_idx]
                    while offset_batch < batch_idx:
                        output_start += seqused_q[offset_batch]
                        if const_expr(not self.paged):
                            kv_length_before = seqused_kv[offset_batch]
                            kv_token_start += kv_length_before
                            v_scale_start += ceil_div(
                                kv_length_before, MXFP8_SCALE_GROUP_ELEMENTS
                            )
                        offset_batch += 1
                    q_token_start = batch_idx * self.tile_cube_m
                    merged_end = dynamic_min(
                        (m_idx + 1) * self.tile_cube_m, seqlen_q_actual * head_group_num
                    )
                    token_end = ceil_div(merged_end, head_group_num)
                    visible_kv = seqlen_k
                    if const_expr(self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL):
                        visible_kv = dynamic_max(
                            Int64(0),
                            dynamic_min(
                                seqlen_k, seqlen_k - seqlen_q_actual + token_end
                            ),
                        )
                    n_idx_start = Int64(0)
                    n_idx_end = ceil_div(visible_kv, self.tile_n)

                    macro_end = ceil_div(n_idx_end - n_idx_start, 2)
                    fd_slot = Int64(-1)
                    if const_expr(self.enable_flash_decode):
                        if fd_active:
                            macro_start = Int64(0)
                            macro_stop = macro_end
                            if tile_idx == start_tile_idx:
                                macro_start = first_macro
                            if tile_idx == end_boundary_tile:
                                macro_stop = last_macro
                            if macro_start > 0 or macro_stop < macro_end:
                                fd_slot = first_slot
                                if first_macro > 0 and tile_idx != start_tile_idx:
                                    fd_slot += 1
                            n_idx_start = macro_start * 2
                            n_idx_end = dynamic_min(n_idx_end, macro_stop * 2)
                            macro_end = macro_stop - macro_start

                    m_axis_triple = active_m_tile_idx % self.preload_num

                    do_tile = (
                        m_idx * self.tile_cube_m < seqlen_q_actual * head_group_num
                        and visible_kv > 0
                    )
                    if do_tile:
                        q_l1_tile = self.q_l1.next_write()
                        self._load_merged_query(
                            q_l1_tile,
                            query,
                            q_token_start,
                            seqlen_q_actual,
                            head_idx,
                            head_group_num,
                            m_idx,
                        )
                        q_scale_tile_gm = self._query_scale_merged_tile(
                            q_scale_gm,
                            q_token_start,
                            seqlen_q_actual,
                            head_idx,
                            head_group_num,
                            m_idx,
                        )
                        q_scale_l1_tile = self.q_scale_l1.next_write()
                        mem_copy(q_scale_l1_tile, q_scale_tile_gm.flatten(-2, -1))
                        actual_m = dynamic_min(
                            Int64(self.tile_cube_m),
                            seqlen_q_actual * head_group_num - m_idx * self.tile_cube_m,
                        )
                        vec_rows = (
                            ceil_div(actual_m, SPLIT_M_TOTAL_ALIGNMENT)
                            * SPLIT_M_ALIGNMENT
                        )
                        local_q_offset = self.subblock_idx * vec_rows
                        merged_row_start = m_idx * self.tile_cube_m + local_q_offset

                        for macro_idx in range(0, macro_end):
                            if overlap_clear and clear_cursor < clear_end:
                                clear_cursor = self._clear_padding_chunk(
                                    attn_out,
                                    seqused_q,
                                    seqused_kv,
                                    clear_scratch,
                                    clear_cursor,
                                    clear_end,
                                )

                            n_block_a, n_block_b, valid_n_a, valid_n_b, has_second = (
                                self.block_info.get_macro_kv_blocks(
                                    n_idx_start, n_idx_end, macro_idx, seqlen_k
                                )
                            )
                            dynamic_valid_n_a, dynamic_valid_n_b = valid_n_a, valid_n_b
                            if const_expr(
                                self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL
                            ):
                                dynamic_valid_n_a = (
                                    seqlen_k
                                    - seqlen_q_actual
                                    + 1
                                    - n_block_a * self.tile_n
                                )
                                dynamic_valid_n_b = (
                                    seqlen_k
                                    - seqlen_q_actual
                                    + 1
                                    - n_block_b * self.tile_n
                                )
                            is_first_macro = Int64(1) if macro_idx == 0 else Int64(0)
                            is_last_macro = (
                                Int64(1) if macro_idx == macro_end - 1 else Int64(0)
                            )
                            tile_triple = macro_global_idx % SOFTMAX_DELAY_SLOTS

                            # Consume delayed P before producing new QK. This returns
                            # P/scale L1 credits and avoids a QK-UB/P-L1 wait cycle.
                            if macro_global_idx >= self.preload_num - 1:
                                delayed = delayed_metadata.tap(self.preload_num - 1)
                                self._consume_delayed_pv(
                                    attn_out,
                                    softmax_lse_gm,
                                    value,
                                    v_scale_gm,
                                    block_table,
                                    seqused_q,
                                    seqused_kv,
                                    p_scale,
                                    head_group_num,
                                    delayed.batch_idx,
                                    delayed.head_idx,
                                    delayed.m_idx,
                                    delayed.n_block_a,
                                    delayed.n_block_b,
                                    delayed.has_second,
                                    delayed.is_first_macro,
                                    delayed.is_last_macro,
                                    delayed.tile_triple,
                                    delayed.m_axis_triple,
                                    delayed.actual_m,
                                    delayed.fd_slot,
                                    delayed.output_start,
                                    delayed.kv_token_start,
                                    delayed.v_scale_start,
                                )

                            # Retain this macro's metadata for its later PV stage.
                            delayed_metadata.push(
                                batch_idx=batch_idx,
                                head_idx=head_idx,
                                m_idx=m_idx,
                                n_block_a=n_block_a,
                                n_block_b=n_block_b,
                                has_second=has_second,
                                is_first_macro=is_first_macro,
                                is_last_macro=is_last_macro,
                                tile_triple=tile_triple,
                                m_axis_triple=m_axis_triple,
                                actual_m=actual_m,
                                fd_slot=fd_slot,
                                output_start=output_start,
                                kv_token_start=kv_token_start,
                                v_scale_start=v_scale_start,
                            )

                            kv_head_idx = head_idx // head_group_num

                            # Produce QK subtiles before vector consumption.
                            self._compute_qk_tile(
                                self.qk_ub_a,
                                q_l1_tile,
                                q_scale_l1_tile,
                                key,
                                k_scale_gm,
                                block_table,
                                batch_idx,
                                kv_head_idx,
                                kv_token_start,
                                seqlen_k,
                                n_block_a,
                                actual_m,
                            )

                            if has_second:
                                self._compute_qk_tile(
                                    self.qk_ub_b,
                                    q_l1_tile,
                                    q_scale_l1_tile,
                                    key,
                                    k_scale_gm,
                                    block_table,
                                    batch_idx,
                                    kv_head_idx,
                                    kv_token_start,
                                    seqlen_k,
                                    n_block_b,
                                    actual_m,
                                )

                                # P_a uses its local maximum and gets a correction scale;
                                # P_b is emitted in the merged frame with 1.0 scale.
                                if is_first_macro:
                                    self.vector_softmax.compute_first_subtile_softmax(
                                        self.vector_softmax.softmax_max_ub[
                                            m_axis_triple
                                        ],
                                        self.qk_ub_a,
                                        scale,
                                        p_scale,
                                        dynamic_valid_n_a,
                                        valid_n_a,
                                        merged_row_start,
                                        head_group_num,
                                        actual_m,
                                        update=False,
                                    )
                                else:
                                    self.vector_softmax.compute_first_subtile_softmax(
                                        self.vector_softmax.softmax_max_ub[
                                            m_axis_triple
                                        ],
                                        self.qk_ub_a,
                                        scale,
                                        p_scale,
                                        dynamic_valid_n_a,
                                        valid_n_a,
                                        merged_row_start,
                                        head_group_num,
                                        actual_m,
                                        update=True,
                                    )
                                self.vector_softmax.store_quantized_p(
                                    self.p_l1_a, actual_m
                                )
                                if is_first_macro:
                                    self.vector_softmax.compute_second_subtile_softmax(
                                        self.qk_ub_b,
                                        self.vector_softmax.softmax_max_ub[
                                            m_axis_triple
                                        ],
                                        self.vector_softmax.softmax_sum_ub[
                                            m_axis_triple
                                        ],
                                        self.vector_softmax.prev_running_max_correction_ub[
                                            tile_triple
                                        ],
                                        scale,
                                        p_scale,
                                        dynamic_valid_n_b,
                                        valid_n_b,
                                        merged_row_start,
                                        head_group_num,
                                        actual_m,
                                        update=False,
                                    )
                                else:
                                    self.vector_softmax.compute_second_subtile_softmax(
                                        self.qk_ub_b,
                                        self.vector_softmax.softmax_max_ub[
                                            m_axis_triple
                                        ],
                                        self.vector_softmax.softmax_sum_ub[
                                            m_axis_triple
                                        ],
                                        self.vector_softmax.prev_running_max_correction_ub[
                                            tile_triple
                                        ],
                                        scale,
                                        p_scale,
                                        dynamic_valid_n_b,
                                        valid_n_b,
                                        merged_row_start,
                                        head_group_num,
                                        actual_m,
                                        update=True,
                                    )
                                self.vector_softmax.encode_p_scale(actual_m=actual_m)
                                self.vector_softmax.store_compact_p_scale(
                                    self.p_scale_l1_a, actual_m
                                )
                                self.vector_softmax.store_quantized_p(
                                    self.p_l1_b, actual_m
                                )
                                self.vector_softmax.encode_p_scale(
                                    use_unit_scale=True, actual_m=actual_m
                                )
                                self.vector_softmax.store_compact_p_scale(
                                    self.p_scale_l1_b, actual_m
                                )
                            else:
                                if is_first_macro:
                                    self.vector_softmax.compute_single_subtile_softmax(
                                        self.qk_ub_a,
                                        self.vector_softmax.softmax_max_ub[
                                            m_axis_triple
                                        ],
                                        self.vector_softmax.softmax_sum_ub[
                                            m_axis_triple
                                        ],
                                        self.vector_softmax.prev_running_max_correction_ub[
                                            tile_triple
                                        ],
                                        scale,
                                        p_scale,
                                        dynamic_valid_n_a,
                                        valid_n_a,
                                        merged_row_start,
                                        head_group_num,
                                        actual_m,
                                        update=False,
                                    )
                                else:
                                    self.vector_softmax.compute_single_subtile_softmax(
                                        self.qk_ub_a,
                                        self.vector_softmax.softmax_max_ub[
                                            m_axis_triple
                                        ],
                                        self.vector_softmax.softmax_sum_ub[
                                            m_axis_triple
                                        ],
                                        self.vector_softmax.prev_running_max_correction_ub[
                                            tile_triple
                                        ],
                                        scale,
                                        p_scale,
                                        dynamic_valid_n_a,
                                        valid_n_a,
                                        merged_row_start,
                                        head_group_num,
                                        actual_m,
                                        update=True,
                                    )
                                self.vector_softmax.store_quantized_p(
                                    self.p_l1_a, actual_m
                                )
                                self.vector_softmax.encode_p_scale(
                                    use_unit_scale=True, actual_m=actual_m
                                )
                                self.vector_softmax.store_compact_p_scale(
                                    self.p_scale_l1_a, actual_m
                                )

                            delayed_metadata.advance()
                            macro_global_idx += 1

                        active_m_tile_idx += 1

        # Drain up to preload_num - 1 stages with explicit branches so the
        # compiler cannot hoist RingBuffer waits out of a dynamic loop.
        # Drain the oldest outstanding macro.
        if macro_global_idx >= self.preload_num - 1:
            delayed = delayed_metadata.tap(self.preload_num - 1)
            self._consume_delayed_pv(
                attn_out,
                softmax_lse_gm,
                value,
                v_scale_gm,
                block_table,
                seqused_q,
                seqused_kv,
                p_scale,
                head_group_num,
                delayed.batch_idx,
                delayed.head_idx,
                delayed.m_idx,
                delayed.n_block_a,
                delayed.n_block_b,
                delayed.has_second,
                delayed.is_first_macro,
                delayed.is_last_macro,
                delayed.tile_triple,
                delayed.m_axis_triple,
                delayed.actual_m,
                delayed.fd_slot,
                delayed.output_start,
                delayed.kv_token_start,
                delayed.v_scale_start,
            )

        # Drain the final macro (also the only one in a single-macro run).
        if macro_global_idx > 0:
            delayed = delayed_metadata.tap(1)
            self._consume_delayed_pv(
                attn_out,
                softmax_lse_gm,
                value,
                v_scale_gm,
                block_table,
                seqused_q,
                seqused_kv,
                p_scale,
                head_group_num,
                delayed.batch_idx,
                delayed.head_idx,
                delayed.m_idx,
                delayed.n_block_a,
                delayed.n_block_b,
                delayed.has_second,
                delayed.is_first_macro,
                delayed.is_last_macro,
                delayed.tile_triple,
                delayed.m_axis_triple,
                delayed.actual_m,
                delayed.fd_slot,
                delayed.output_start,
                delayed.kv_token_start,
                delayed.v_scale_start,
            )

        # Finish all padding before the FD barrier/allocator rewind. On
        # lightly loaded cores this tail can overlap other cores' attention.
        if overlap_clear:
            while clear_cursor < clear_end:
                clear_cursor = self._clear_padding_chunk(
                    attn_out,
                    seqused_q,
                    seqused_kv,
                    clear_scratch,
                    clear_cursor,
                    clear_end,
                )

        if const_expr(self.enable_flash_decode):
            if fd_active:
                # All launched blocks participate, including idle AIV halves.
                # Final delayed PV and all producer DMAs precede this barrier.
                global_sync_all()
                onchip_rewind(reset_sync_id=True)
                self._setup_fd_merge()
                self.vector_output.init_invalid_row_check(
                    self.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL,
                    False,
                    scale,
                    p_scale,
                )
                for section in range(sections):
                    vec = self.block_idx * AIV_SUBBLOCK_COUNT + self.subblock_idx
                    record = (
                        1
                        + sections * self.block_num
                        + section * self.block_num * AIV_SUBBLOCK_COUNT
                        + vec
                    ) * METADATA_STRIDE
                    self._merge_fd_task(
                        attn_out,
                        softmax_lse_gm,
                        seqused_q,
                        seqused_kv,
                        head_num_kv,
                        head_group_num,
                        fd_metadata_gm,
                        record,
                        section * 2 * self.block_num,
                        p_scale,
                    )
                # Finish the merge on every core before leaving the mixed kernel.
                global_sync_all()


# ---- Decode JIT launcher -----------------------------------------------------
class FlashAttentionGQAMxfp8DecodeLauncher:
    """JIT launcher for the ND decode kernel."""

    def __init__(
        self,
        kv_layout,
        mask_mode=MASK_MODE_NO_MASK,
        q_layout=QLayout.TND,
        paged=False,
        block_size=128,
        return_softmax_lse=False,
        enable_flash_decode=False,
        short_lse=False,
        *,
        block_dim,
    ):
        self.enable_flash_decode = enable_flash_decode
        self.short_lse = short_lse
        check_mask_mode(mask_mode)
        self.mask_mode = mask_mode
        self.q_layout = q_layout

        self.kv_layout = kv_layout
        self.paged = self.kv_layout != KVLayout.TND
        self.block_size = block_size
        self.return_softmax_lse = return_softmax_lse
        self.block_dim = block_dim

    @host
    def launch(
        self,
        attn_out: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_scale_gm: Tensor,
        k_scale_gm: Tensor,
        v_scale_gm: Tensor,
        p_scale_gm: Tensor,
        scale: dtypes.float32,
        block_table: Tensor,
        seqused_kv: Tensor,
        seqused_q: Tensor,
        max_seqlen_q: int,
        softmax_lse_gm: Tensor | None = None,
        fd_output_gm: Tensor | None = None,
        fd_sum_gm: Tensor | None = None,
        fd_max_gm: Tensor | None = None,
        fd_metadata_gm: Tensor | None = None,
    ):
        head_dim = query.shape[2]
        tile_cube_m, tile_vec_m, tile_n, tile_d = select_tile_config(head_dim)
        op = FlashAttentionGQAMxfp8DecodeKernel(
            short_lse=self.short_lse,
            block_num=self.block_dim,
            enable_flash_decode=self.enable_flash_decode,
            kv_heads=key.shape[2]
            if self.kv_layout == KVLayout.PA_BBND
            else key.shape[1],
            tile_cube_m=tile_cube_m,
            tile_vec_m=tile_vec_m,
            tile_n=tile_n,
            tile_d=tile_d,
            mask_mode=self.mask_mode,
            q_layout=self.q_layout,
            paged=self.paged,
            block_size=self.block_size,
            kv_layout=self.kv_layout,
            return_softmax_lse=self.return_softmax_lse,
        )
        op[self.block_dim](
            attn_out,
            query,
            key,
            value,
            q_scale_gm,
            k_scale_gm,
            v_scale_gm,
            p_scale_gm,
            scale,
            block_table,
            seqused_kv,
            seqused_q,
            max_seqlen_q,
            softmax_lse_gm,
            fd_output_gm,
            fd_sum_gm,
            fd_max_gm,
            fd_metadata_gm,
        )


# =============================================================================
# Public API
# =============================================================================


@kernel
class D72OperandsPadKernel:
    """Pad FP8 bytes and scale bytes in one launch; 16 KiB UB per AIV.

    This is a compatibility path for GM->L1 copies which do not support
    explicit padding with ND2NZ. No dequantization or requantization occurs.
    """

    def __init__(self, block_num, copy_v):
        self.copy_v = copy_v
        self.block_num = Int64(block_num)
        self.rows = 64
        self.scratch = _make_buffer(
            MemLoc.UB, (self.rows, HEAD_DIM_128 * 2), dtypes.int8, data_format="nd"
        )

    @jit
    def _pad_tensor(self, source, destination, fill):
        source = source.view(dtypes.int8)
        destination = destination.view(dtypes.int8)
        worker = get_block_idx()
        workers = self.block_num
        width = destination.shape[-1]
        scratch_width = (width + FP8_NZ_K_BLOCK - 1) // FP8_NZ_K_BLOCK * FP8_NZ_K_BLOCK
        scratch = reinterpret(
            self.scratch,
            dtype=dtypes.int8,
            shape=(self.rows, scratch_width),
            stride=(scratch_width, 1),
            data_format="nd",
        )
        if const_expr(len(source.shape) == 3):
            src = source.permute((1, 0, 2))
            dst = destination.permute((1, 0, 2))
            tiles = ceil_div(source.shape[0], self.rows)
            for task in range(worker, source.shape[1] * tiles, workers):
                head, tile = task // tiles, task % tiles
                begin = tile * self.rows
                end = dynamic_min(Int64(source.shape[0]), begin + self.rows)
                matrix = tile_slice(
                    src[head, begin:end, None], (self.rows, source.shape[-1]), (0, 0)
                )
                mem_copy(
                    scratch,
                    matrix.pad((0, scratch_width - source.shape[-1]), value=fill),
                    options=copy_options(engine="nddma"),
                )
                target = tile_slice(
                    dst[head, begin:end, None], (self.rows, width), (0, 0)
                )
                mem_copy(target, scratch[: end - begin, :width])
        else:
            tiles = ceil_div(source.shape[2], self.rows)
            for task in range(
                worker, source.shape[0] * source.shape[1] * tiles, workers
            ):
                page, head, tile = idx2crd(
                    task, [source.shape[0], source.shape[1], tiles]
                )
                begin = tile * self.rows
                end = dynamic_min(Int64(source.shape[2]), begin + self.rows)
                matrix = tile_slice(
                    source[page, head, begin:end, None],
                    (self.rows, source.shape[-1]),
                    (0, 0),
                )
                mem_copy(
                    scratch,
                    matrix.pad((0, scratch_width - source.shape[-1]), value=fill),
                    options=copy_options(engine="nddma"),
                )
                target = tile_slice(
                    destination[page, head, begin:end, None], (self.rows, width), (0, 0)
                )
                mem_copy(target, scratch[: end - begin, :width])

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        vs: Tensor,
        qp: Tensor,
        kp: Tensor,
        vp: Tensor,
        vsp: Tensor,
    ):
        self._pad_tensor(q, qp, 0)
        self._pad_tensor(k, kp, 0)
        if const_expr(self.copy_v):
            self._pad_tensor(v, vp, 0)
        self._pad_tensor(
            vs.flatten(len(vs.shape) - 2, len(vs.shape) - 1),
            vsp.flatten(len(vsp.shape) - 2, len(vsp.shape) - 1),
            E8M0_SCALE_ONE_BYTE,
        )


@host
def _pad_d72_operands(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    vs: Tensor,
    qp: Tensor,
    kp: Tensor,
    vp: Tensor,
    vsp: Tensor,
    block_dim: Constexpr[int],
    copy_v: Constexpr[bool],
):
    op = D72OperandsPadKernel(block_dim * AIV_SUBBLOCK_COUNT, copy_v)
    op[block_dim * AIV_SUBBLOCK_COUNT](q, k, v, vs, qp, kp, vp, vsp)


@kernel
class EmptyAttentionOutputKernel:
    """Fill nonempty padded outputs when the attention launch bound is zero.

    Each AIV uses 4096 B BF16 UB plus 256 B FP32 UB; no Cube storage.
    """

    def __init__(self, head_dim, with_lse, block_num):
        self.block_num = Int64(block_num)
        self.head_dim = head_dim
        self.with_lse = with_lse
        self.rows = 8
        self.output_ub = _make_buffer(MemLoc.UB, (self.rows, head_dim), dtypes.bfloat16)
        self.lse_ub = _make_buffer(MemLoc.UB, (FP32_VECTOR_LANES, 1), dtypes.float32)

    def __call__(self, output: Tensor, lse: Tensor | None = None):
        worker = get_block_idx()
        workers = self.block_num
        with vf(mode="simd"):
            full, _ = rr.update_mask(FP32_VECTOR_LANES, elem_bits=FP32_ELEM_BITS)
            zero = rr.vdups(0.0, dtypes.float32, mask=full)
            packed = rr.vcast(zero, dtypes.bfloat16, mask=full)
            for offset in range(0, self.rows * self.head_dim, FP32_VECTOR_LANES):
                mask, _ = rr.update_mask(
                    min(FP32_VECTOR_LANES, self.rows * self.head_dim - offset),
                    elem_bits=FP32_ELEM_BITS,
                )
                rr.vstore_pack(
                    self.output_ub, offset, packed, mask, pack_mode="b32_to_b16"
                )
            if const_expr(self.with_lse):
                one = rr.vdups(1.0, dtypes.float32, mask=full)
                infinity = rr.vdiv(one, zero, mask=full)
                rr.vstore(self.lse_ub, 0, infinity, full)
            rr.vmem_bar("vst_vld")
        flat = output.view((output.shape[0] * output.shape[1], self.head_dim))
        for tile in range(worker, ceil_div(flat.shape[0], self.rows), workers):
            target = tile_slice(flat, (self.rows, self.head_dim), (tile, 0))
            mem_copy(target, self.output_ub[: target.shape[0], None])
        if const_expr(self.with_lse):
            flat_lse = lse.view((lse.shape[0] * lse.shape[1], 1))
            for tile in range(
                worker, ceil_div(flat_lse.shape[0], FP32_VECTOR_LANES), workers
            ):
                target = tile_slice(flat_lse, (FP32_VECTOR_LANES, 1), (tile, 0))
                mem_copy(target, self.lse_ub[: target.shape[0], None])


@host
def _fill_empty_attention_output(output: Tensor, lse: Tensor | None = None):
    op = EmptyAttentionOutputKernel(output.shape[2], lse is not None, block_num=1)
    op[1](output, lse)


def _launch_precompiled_attention(launcher, *args):
    contracts = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            contracts.append(
                (tuple(arg.shape), tuple(arg.stride()), arg.dtype, arg.device)
            )
        elif arg is None:
            contracts.append(None)
        elif isinstance(arg, float):
            contracts.append(float)
        elif isinstance(arg, int):
            contracts.append(int)
        else:
            return launcher.launch(*args)
    key = (
        os.getpid(),
        type(launcher),
        tuple(sorted(vars(launcher).items())),
        tuple(contracts),
    )
    with _ATTENTION_COMPILE_LOCK:
        program = _COMPILED_ATTENTION_PROGRAMS.get(key)
        if program is None:
            specs = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    specs.append(
                        TensorSpec(
                            tuple(arg.shape),
                            _ATTENTION_TENSOR_DTYPES[arg.dtype],
                            stride=tuple(arg.stride()),
                        )
                    )
                elif arg is None:
                    specs.append(None)
                elif isinstance(arg, float):
                    specs.append(dtypes.float32)
                else:
                    specs.append(dtypes.int64)
            program = compile_program(launcher.launch, *specs)
            _COMPILED_ATTENTION_PROGRAMS[key] = program
            if len(_COMPILED_ATTENTION_PROGRAMS) > _ATTENTION_PROGRAM_CACHE_LIMIT:
                _COMPILED_ATTENTION_PROGRAMS.popitem(last=False)
        _COMPILED_ATTENTION_PROGRAMS.move_to_end(key)
    # Only the program is retained; bind this call's tensors and runtime scalars.
    program(*(arg for arg in args if arg is not None))


def npu_flash_atten_mxfp8_fullquant(
    query,
    key,
    value,
    q_scale_gm,
    k_scale_gm,
    v_scale_gm,
    p_scale_gm,
    *,
    softmax_scale,
    attn_mask,
    block_table,
    seqused_kv,
    seqused_q,
    mask_mode=MASK_MODE_NO_MASK,
    q_layout=QLayout.TND,
    paged=False,
    block_size=128,
    kv_layout,
    return_softmax_lse=False,
    max_seqlen_q=-1,
    output_token_count=None,
    metadata_state=None,
    lengths_epoch=None,
):
    """Run GQA MXFP8 attention and return BF16 output and optional FP32 LSE.

    q_layout is explicit and currently supports only TND. In this layout query
    is [T, N_q, D]; token offsets, head grouping, and query-scale indexing all
    depend on that shape. Other layouts must be added together with their
    shape checks and indexing path before entering the kernel.
    Q/K/V scales are native torch.float8_e8m0fnu tensors; p_scale_gm
    remains FP32. Rank-4 Q scales select Prefill; rank-5 select Decode.
    FD never changes scene. Causal visibility is computed in the kernel.
    Output is compact TND [sum(seqused_q), N_q, D], LSE [N_q, sum(seqused_q)].
    Decode input Q remains padded to 128 tokens per batch. Optional host integer
    output_token_count MUST equal sum(seqused_q); caller owns this invariant.
    Omitting it derives the count with a device-to-host synchronization.
    ``max_seqlen_q=-1`` leaves the maximum unspecified. Scheduling uses Q
    storage capacity (128 rows per batch for Decode) as a conservative bound;
    seqused_q supplies actual lengths, independently of mask capacity;
    zero skips attention computation and initializes padded empty outputs.
    Metadata is generated on AICPU. Pass AttentionMetadataState to reuse it;
    advance lengths_epoch after external writes to length vectors.
    """
    check_public_inputs(
        query,
        key,
        value,
        q_scale_gm,
        k_scale_gm,
        v_scale_gm,
        p_scale_gm,
        softmax_scale,
        attn_mask,
        block_table,
        seqused_kv,
        seqused_q,
        mask_mode,
        paged,
        block_size,
        kv_layout,
        q_layout,
        return_softmax_lse,
        max_seqlen_q,
    )
    device_context = (
        torch.npu.device(query.device) if query.device.type == "npu" else nullcontext()
    )
    with device_context:
        # Metadata and flattened tile indices need a nonnegative capacity bound.
        # Actual per-batch lengths are read from seqused_q by AICPU and the kernel.
        effective_max_seqlen_q = max_seqlen_q
        if max_seqlen_q == -1:
            effective_max_seqlen_q = (
                DECODE_QUERY_ROWS if q_scale_gm.ndim == 5 else query.shape[0]
            )
        head_dim = query.shape[2]

        # Decode inputs retain padded Q storage; returned TND output is compact.
        # A caller-known count avoids a device-to-host synchronization. Otherwise
        # derive it explicitly; never hide that cost in the kernel timing claim.
        if q_scale_gm.ndim == 5:
            if output_token_count is None:
                output_token_count = int(seqused_q.sum().item())
            _check_output_token_count(output_token_count, query.shape[0])
            output_tokens = output_token_count
        else:
            output_tokens = query.shape[0]
        output = torch.empty(
            (output_tokens, query.shape[1], head_dim),
            dtype=torch.bfloat16,
            device=query.device,
        )
        softmax_lse = (
            torch.empty(
                (query.shape[1], output_tokens),
                dtype=torch.float32,
                device=query.device,
            )
            if return_softmax_lse
            else None
        )
        if output_tokens == 0:
            return output, softmax_lse
        if effective_max_seqlen_q == 0:
            if query.device.type == "cpu":
                # CPU-only empty-bound contract checks do not invoke device code.
                output.zero_()
                if softmax_lse is not None:
                    softmax_lse.fill_(float("inf"))
            else:
                _fill_empty_attention_output(output, softmax_lse)
            return output, softmax_lse

        stream = torch.npu.current_stream(query.device)
        # The DSL caches hardware counts process-wide. Refresh on multi-device hosts.
        platform = get_platform_info(
            force_refresh=torch.npu.device_count() > 1, stream=stream
        )
        cores = int(platform.cube_core_num)
        vectors = int(platform.vector_core_num)
        if cores <= 0 or vectors != AIV_SUBBLOCK_COUNT * cores:
            raise ValueError(
                "Arch35 mixed attention needs two available AIVs per Cube core"
            )

        is_prefill = q_scale_gm.ndim == 4

        if kv_layout == KVLayout.PA_BBND:
            # Expose the head axis before page tokens without copying PA_BBND data.
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)
            k_scale_gm = k_scale_gm.permute(0, 2, 1, 3, 4)
            v_scale_gm = v_scale_gm.permute(0, 2, 1, 3, 4)
            kv_layout = KVLayout.PA_BNBD
        # Normalize paged strides before padding so tiles span tokens, not heads.
        if head_dim == HEAD_DIM_72 and not (
            q_scale_gm.ndim == 4 and kv_layout == KVLayout.TND
        ):
            # Empty allocation only; all bytes are initialized by the DSL pad kernel.
            # Reuse row-contiguous paged V for Decode. Prefill also requires
            # contiguous token rows; any required copy retains logical D.
            copy_v = (
                kv_layout == KVLayout.TND
                or value.stride(-1) != 1
                or (q_scale_gm.ndim == 4 and value.stride(-2) != value.shape[-1])
            )
            padded = tuple(
                tensor
                if index == 2 and not copy_v
                else torch.empty(
                    (
                        *tensor.shape[:-1],
                        tensor.shape[-1]
                        if index == 2 and kv_layout != KVLayout.TND
                        else HEAD_DIM_128,
                    ),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                for index, tensor in enumerate((query, key, value))
            )
            padded_v_scale = torch.empty(
                (*v_scale_gm.shape[:-2], HEAD_DIM_128, 2),
                dtype=v_scale_gm.dtype,
                device=v_scale_gm.device,
            )
            _pad_d72_operands(
                query,
                key,
                value,
                v_scale_gm,
                *padded,
                padded_v_scale,
                cores,
                copy_v,
            )
            query, key, value = padded
            v_scale_gm = padded_v_scale

        # Capacity bounds FD workspace needs without synchronizing length tensors.
        kv_capacity = (
            block_table.shape[1] * block_size
            if kv_layout != KVLayout.TND
            else key.shape[0]
        )
        launcher_type = (
            FlashAttentionGQAMxfp8PrefillLauncher
            if is_prefill
            else FlashAttentionGQAMxfp8DecodeLauncher
        )
        tile_m, _, tile_n, _ = select_tile_config(head_dim)
        macro_n = 2 * tile_n
        heads = query.shape[1] if is_prefill else q_scale_gm.shape[0]
        fd_metadata_gm, slots = prepare_metadata(
            seqused_q,
            seqused_kv,
            heads=heads,
            group=query.shape[1] // heads,
            max_q=effective_max_seqlen_q,
            tile_m=tile_m,
            macro_n=macro_n,
            sparse=mask_mode,
            cores=cores,
            vectors=vectors,
            head_dim=head_dim,
            kv_capacity=kv_capacity,
            state=metadata_state,
            lengths_epoch=lengths_epoch,
        )
        decode_workspace = (None, None, None)
        if slots:
            decode_workspace = (
                torch.empty(
                    (slots * tile_m, _aligned_tile_head_dim(head_dim)),
                    dtype=torch.float32,
                    device=query.device,
                ),
                torch.empty(
                    (slots * tile_m, FD_STAT_ROW_ELEMENTS),
                    dtype=torch.float32,
                    device=query.device,
                ),
                torch.empty(
                    (slots * tile_m, FD_STAT_ROW_ELEMENTS),
                    dtype=torch.float32,
                    device=query.device,
                ),
            )
        launcher = launcher_type(
            **(
                {}
                if is_prefill
                else {"short_lse": return_softmax_lse and effective_max_seqlen_q <= 3}
            ),
            block_dim=cores,
            enable_flash_decode=bool(slots),
            mask_mode=mask_mode,
            q_layout=q_layout,
            paged=kv_layout != KVLayout.TND,
            block_size=block_size,
            return_softmax_lse=return_softmax_lse,
            kv_layout=kv_layout,
        )
        _launch_precompiled_attention(
            launcher,
            output,
            query,
            key,
            value,
            q_scale_gm,
            k_scale_gm,
            v_scale_gm,
            p_scale_gm,
            softmax_scale,
            block_table,
            seqused_kv,
            seqused_q,
            effective_max_seqlen_q,
            softmax_lse,
            *decode_workspace,
            fd_metadata_gm,
        )
        return output, softmax_lse
