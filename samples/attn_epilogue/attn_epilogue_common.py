# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared MXFP8 matmul and quantization for attention epilogues."""

from collections import OrderedDict
from dataclasses import dataclass
import inspect
import threading

import cannbotdsl as dsl
import torch
import torch_npu
from cannbotdsl import make_layout
from cannbotdsl import Buffer, RegLayout, dtypes
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import (
    get_block_idx,
    get_block_num,
    get_subblock_dim,
    get_subblock_id,
)
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops import reg
from cannbotdsl.ops.reg.cast import RoundingMode
from cannbotdsl.tensor import MemLoc, tile_slice

from cannbotdsl.ops.sync import (
    PIPE,
    vec_sync_block_arrive,
    vec_sync_block_wait,
    cube_sync_block_wait,
)
from cannbotdsl.ops.matmul import matmul


NG = 8
F = 4096
O_LORA = 1024
DIM = 5120
BASE_N = 192


BASE_K = 128
STEPK = 4


# Keep the sample usable with both layout-based DSL builds and the 0.8.0
# CI wheel, whose constructors still expose the shape-based call convention.
_RING_BUFFER_USES_LAYOUT = "layout" in inspect.signature(RingBuffer).parameters
_BUFFER_POSITIONAL_PARAMETERS = tuple(
    name
    for name, parameter in inspect.signature(Buffer).parameters.items()
    if parameter.kind
    in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
)
_BUFFER_USES_LAYOUT = _BUFFER_POSITIONAL_PARAMETERS[1:3] == ("dtype", "layout")
_REINTERPRET_USES_LAYOUT = "layout" in inspect.signature(dsl.reinterpret).parameters
_SCALE_CHANNEL_USES_LAYOUT = _RING_BUFFER_USES_LAYOUT


def _make_ring_buffer(location, dtype, shape, *, depth, data_format=None):
    if _RING_BUFFER_USES_LAYOUT:
        return RingBuffer(location, dtype, make_layout(shape), depth=depth)
    return RingBuffer(location, shape, dtype, depth=depth, data_format=data_format)


def _make_buffer(location, dtype, shape):
    if _BUFFER_USES_LAYOUT:
        return Buffer(location, dtype, make_layout(shape))
    return Buffer(location, shape, dtype)


HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
NOPE_CHUNKS = NOPE_DIM // 64
QUANT_GROUP = 32
MX_DIVISOR = 64
FP8 = dtypes.float8_e4m3fn
E8M0 = dtypes.float8_e8m0


@jit
def matrix_view(tensor, row_begin, row_end, col_begin, col_end):
    """View a matrix tile using exclusive row and column bounds."""
    return tensor[row_begin:row_end, col_begin:col_end]


@jit
def scale_view(tensor, row_begin, row_end, group_begin, group_end):
    """View packed MX scales as rows of interleaved scale-byte pairs."""
    if const_expr(len(tensor.shape) == 3):
        return tensor[row_begin:row_end, group_begin:group_end, 0:2].reshape(
            row_end - row_begin, (group_end - group_begin) * 2
        )
    return tensor[row_begin:row_end, group_begin * 2 : group_end * 2]


@dataclass(frozen=True)
class ProjectionWeights:
    """The two projection weights and their matching MX scales."""

    first: object
    first_scale: object
    second: object
    second_scale: object


@dataclass(frozen=True)
class ProjectionShape:
    """Logical sequence, group and projection dimensions."""

    t: int
    ng: int
    f: int
    o_lora: int
    dim: int


@dataclass(frozen=True)
class ProjectionTiles:
    """Shared M tile and the N tile of each projection."""

    rows: int
    first_n: int
    second_n: int


@dataclass(frozen=True)
class RopeInputs:
    """BF16 activations and the sine/cosine rows applied to their head tails."""

    source: object
    cos: object
    sin: object


@dataclass(frozen=True)
class QuantOutput:
    """Quantized bytes and their per-group scales."""

    data: object
    scales: object


@dataclass(frozen=True)
class TileExtent:
    """Logical output origin and the valid tile extent."""

    row: object
    col: object
    rows: object
    cols: object


@dataclass(frozen=True)
class RopeBatch:
    """Rows and K slab processed by one vector lane."""

    row: object
    group: object
    chunk: object
    count: object


@dataclass(frozen=True)
class QuantBuffers:
    """Input, quantized output, scales and temporary vector storage."""

    source: object
    output: object
    scales: object
    reciprocal: object
    even: object
    odd: object


@jit
def _quant_row(buffers, num_col):
    ch = num_col // 256
    with vf(mode="simd"):
        m16 = reg.create_mask("all", elem_bits=16)
        fm32 = reg.create_mask("all", elem_bits=32)
        vl16b = reg.create_mask("vl16", elem_bits=8)
        vl16u = reg.create_mask("vl16", elem_bits=16)
        _quant_scales(buffers, ch, m16, vl16b, vl16u)
        reg.vmem_bar(mode="vst_vld")
        _quant_values(buffers, ch, m16, fm32)
        reg.vmem_bar(mode="vst_vld")
        _quant_interleave(buffers, ch)


@jit
def _quant_scales(buffers, ch, m16, vl16b, vl16u):
    em = reg.vdups(0x7F80, dtypes.uint16, mask=m16)
    c8 = reg.vdups(8, dtypes.uint16, mask=m16)
    c262 = reg.vdups(262, dtypes.uint16, mask=m16)
    idx = reg.varange(0, dtypes.uint8)
    one = reg.vdups(1, dtypes.uint8, mask=vl16b)
    even_mask = reg.veqs(reg.vbitwise_and(idx, one, mask=vl16b), 0, mask=vl16b)
    # Scale bytes and reciprocals share the same group exponent reduction.
    ureg_s = reg.vstore_unalign_begin(buffers.scales)
    for chunk in dsl_range(0, ch, 1):
        x0, x1 = reg.vload_deinterleave(buffers.source, chunk * 256)
        a0 = reg.vbitwise_and(reg.vreinterpret(x0, dtypes.uint16), em, mask=m16)
        a1 = reg.vbitwise_and(reg.vreinterpret(x1, dtypes.uint16), em, mask=m16)
        m = reg.vmax(a0, a1, mask=m16)
        gm = reg.vreduce_max_datablock(m, mask=m16)
        e = reg.vmaxs(reg.vshr(gm, 7, mask=m16), 8, mask=m16)
        sb = reg.vsub(e, c8, mask=m16)
        sb8 = reg.vcast(sb, dtypes.uint8, mask=m16, reg_layout=RegLayout.ZERO)
        sqz = reg.vsqueeze_and_storeunalign_init(sb8, mask=even_mask)
        reg.vsqueeze_and_storeunalign(buffers.scales, 0, sqz, ureg_s)
        rev = reg.vshl(reg.vsub(c262, e, mask=m16), 7, mask=m16)
        rev = reg.vselect(
            reg.vdups(0, dtypes.uint16, mask=m16),
            rev,
            cond_mask=reg.veqs(e, 8, mask=m16),
        )
        dup, _ = reg.vinterleave(rev, rev)
        reg.vstore(buffers.reciprocal, chunk * 16, dup, vl16u)
    reg.vsqueeze_and_storeunalign_finalize(buffers.scales, 0, ureg_s)


@jit
def _quant_values(buffers, ch, m16, fm32):
    for chunk in dsl_range(0, ch, 1):
        for half in range(2):
            off = chunk * 256 + half * 128
            xv = reg.vload(buffers.source, off)
            rs = reg.vload_broadcast(
                buffers.reciprocal, chunk * 16 + half * 8, mode="datablock"
            )
            rbf = reg.vreinterpret(rs, dtypes.bfloat16)
            y = reg.vmul(xv, rbf, mask=m16)
            yz = reg.vcast(y, dtypes.float32, mask=m16, reg_layout=RegLayout.ZERO)
            yo = reg.vcast(y, dtypes.float32, mask=m16, reg_layout=RegLayout.ONE)
            qz = reg.vcast(
                yz,
                dtypes.float8_e4m3fn,
                mask=fm32,
                reg_layout=RegLayout.ZERO,
                saturate=True,
            )
            qo = reg.vcast(
                yo,
                dtypes.float8_e4m3fn,
                mask=fm32,
                reg_layout=RegLayout.ZERO,
                saturate=True,
            )
            reg.vstore_pack(
                buffers.even,
                chunk * 128 + half * 64,
                qz,
                fm32,
                pack_mode=reg.PackMode.B32_TO_B8,
            )
            reg.vstore_pack(
                buffers.odd,
                chunk * 128 + half * 64,
                qo,
                fm32,
                pack_mode=reg.PackMode.B32_TO_B8,
            )


@jit
def _quant_interleave(buffers, ch):
    for chunk in dsl_range(0, ch, 1):
        for half in range(2):
            off = chunk * 256 + half * 128
            qz8 = reg.vload(buffers.even, chunk * 128 + half * 64)
            qo8 = reg.vload(buffers.odd, chunk * 128 + half * 64)
            reg.vstore_interleave(buffers.output, off, qz8, qo8, width="b8")


def _operand_channel(
    storage,
    location,
    shape,
    dtype,
    depth,
    *,
    offset=0,
    data_format=None,
    element_bytes=1,
):
    """Build an owned channel or identically laid out views of shared storage."""
    if storage is None:
        return _make_ring_buffer(
            location, dtype, shape, depth=depth, data_format=data_format
        )
    buffers = []
    stride = shape[0] * shape[1] * element_bytes
    for slot in range(depth):
        buffers.append(
            dsl.reinterpret(
                storage,
                dtype=dtype,
                **(
                    {"layout": make_layout(shape)}
                    if _REINTERPRET_USES_LAYOUT
                    else {"shape": shape}
                ),
                offset=offset + slot * stride,
                data_format=data_format or "nz",
            )
        )
    return dsl.make_ring_buffer(buffers)


@dataclass(frozen=True)
class MatmulShape:
    """Grouped matrix dimensions including the padded input row extent."""

    groups: object
    n: object
    k: object
    m: object


@dataclass(frozen=True)
class MxOperands:
    """Projection operands and their packed MX scales."""

    a: object
    b: object
    a_scale: object
    b_scale: object


@dataclass(frozen=True)
class TileIndex:
    """Output tile coordinates and the state of its weight prefetch."""

    m: object
    n: object
    k: object
    prefetched: object = 0


@dataclass(frozen=True)
class MxAccumulator:
    """Accumulator, B scales and live output extents for one output tile."""

    output: object
    b_scale: object
    m: object
    n: object


@dataclass(frozen=True)
class KTileIndex:
    """K tile positions in the projection and in the current scale slab."""

    global_tile: object
    local_tile: object
    k: object
    prefetched: object


@dataclass(frozen=True)
class L1Operands:
    """Live L1 data and A scales consumed by one Cube stage."""

    a: object
    b: object
    a_scale: object


class MatmulMx:
    """One (base_m, base_n) output tile of an MXFP8 GEMM, K streamed in base_k."""

    def __init__(
        self,
        k,
        base_m,
        base_n=BASE_N,
        base_k=BASE_K,
        step_k=1,
        reuse_n=0,
        l0c_depth=2,
        prefetch_tiles=1,
        prefetch_scales=False,
        stream_k=0,
    ):
        self.k = int(k)
        self.stream_k = int(stream_k)
        self.scale_k = self.stream_k or self.k
        self.base_m = int(base_m)
        self.base_n = int(base_n)
        self.base_k = int(base_k)
        self.step_k = int(step_k)  # base_k blocks staged per L1 slot (native stepKb)
        self.k_l1 = self.base_k * self.step_k
        self.prefetch_scales = bool(prefetch_scales)
        self.prefetch_tiles = min(int(prefetch_tiles), 2, self.k // self.k_l1)
        self.sk_l0 = self.base_k // 32  # packed scale length per base_k
        self.sk_full = (self.k // 64) * 2  # packed scale length for the full K
        # Reuse Cube storage only after the previous Cube phase has drained.
        reuse_n = int(reuse_n)
        if reuse_n and reuse_n < self.base_n:
            raise ValueError("reuse_n must cover the physical N tile")
        self._init_l1_channels(reuse_n)
        self._init_l0_channels(reuse_n, l0c_depth)

    @jit
    def prefetch(self, b, n0, sb=None):
        for kk in range(self.prefetch_tiles):
            mem_copy(
                self.l1_b.next_write(),
                matrix_view(
                    b,
                    n0 * self.base_n,
                    min((n0 + 1) * self.base_n, b.shape[0]),
                    kk * self.k_l1,
                    (kk + 1) * self.k_l1,
                ),
            )
        if const_expr(self.prefetch_scales):
            mem_copy(
                self.l1_sb.next_write(),
                matrix_view(
                    sb,
                    n0 * self.base_n,
                    min((n0 + 1) * self.base_n, b.shape[0]),
                    0,
                    sb.shape[1],
                ).permute(1, 0),
            )

    @jit
    def compute(self, operands, index):
        m_end = min((index.m + 1) * self.base_m, operands.a.shape[0])
        n_end = min((index.n + 1) * self.base_n, operands.b.shape[0])
        actual_m = m_end - index.m * self.base_m
        actual_n = n_end - index.n * self.base_n
        b_scales = self._load_scales(operands, index)
        accumulator = self.l0c.next_write()
        state = MxAccumulator(accumulator, b_scales, actual_m, actual_n)
        scale_k = self.stream_k or index.k
        for chunk in range(index.k // scale_k):
            self._compute_chunk(operands, index, state, chunk)
        return accumulator

    @jit
    def tile(self, operands, out, index):
        b = operands.b
        m0, n0 = index.m, index.n
        accumulator = self.compute(operands, index)
        mem_copy(
            matrix_view(
                out,
                m0 * self.base_m,
                min((m0 + 1) * self.base_m, out.shape[0]),
                n0 * self.base_n,
                min((n0 + 1) * self.base_n, b.shape[0]),
            ),
            accumulator,
            options=copy_options(unit_flag=3),
        )

    @jit
    def _load_scales(self, operands, index):
        g = index.k // MX_DIVISOR
        m_begin = index.m * self.base_m
        n_begin = index.n * self.base_n
        m_end = min(m_begin + self.base_m, operands.a_scale.shape[0])
        n_end = min(n_begin + self.base_n, operands.b.shape[0])
        if const_expr(not self.stream_k):
            if const_expr(_SCALE_CHANNEL_USES_LAYOUT):
                sa_dst = self.l1_sa.next_write().permute(1, 0)
            else:
                sa_dst = self.l1_sa.next_write()
            mem_copy(
                sa_dst,
                matrix_view(
                    operands.a_scale, m_begin, m_end, 0, operands.a_scale.shape[1]
                ),
            )
        if not self.prefetch_scales or index.prefetched == 0:
            mem_copy(
                self.l1_sb.next_write(),
                matrix_view(
                    operands.b_scale, n_begin, n_end, 0, operands.b_scale.shape[1]
                ).permute(1, 0),
            )
        return self.l1_sb.next_read()[0 : 2 * g, 0 : self.base_n]

    @jit
    def _compute_chunk(self, operands, index, state, chunk):
        scale_k = self.stream_k or index.k
        if const_expr(self.stream_k):
            if index.prefetched != 0:
                cube_sync_block_wait(PIPE.MTE2, 3 + chunk)
        if const_expr(self.stream_k):
            m_begin = index.m * self.base_m
            m_end = min(m_begin + self.base_m, operands.a_scale.shape[0])
            g_begin = chunk * (scale_k // 64)
            g_end = (chunk + 1) * (scale_k // 64)
            if const_expr(_SCALE_CHANNEL_USES_LAYOUT):
                sa_dst = self.l1_sa.next_write().permute(1, 0)
            else:
                sa_dst = self.l1_sa.next_write()
            mem_copy(
                sa_dst,
                matrix_view(operands.a_scale, m_begin, m_end, g_begin * 2, g_end * 2),
            )
        if const_expr(_SCALE_CHANNEL_USES_LAYOUT):
            a_scales = self.l1_sa.next_read().permute(1, 0)[
                0 : self.base_m, 0 : scale_k // 32
            ]
        else:
            a_scales = self.l1_sa.next_read()[0 : self.base_m, 0 : scale_k // 32]
        for local_kk in range(scale_k // self.k_l1):
            kk = chunk * (scale_k // self.k_l1) + local_kk
            position = KTileIndex(kk, local_kk, index.k, index.prefetched)
            self._compute_l1(operands, index, state, a_scales, position)

    @jit
    def _compute_l1(self, operands, index, state, a_scales, position):
        m_begin = index.m * self.base_m
        n_begin = index.n * self.base_n
        m_end = min(m_begin + self.base_m, operands.a.shape[0])
        n_end = min(n_begin + self.base_n, operands.b.shape[0])
        k_begin = position.global_tile * self.k_l1
        k_end = k_begin + self.k_l1
        mem_copy(
            self.l1_a.next_write(),
            matrix_view(operands.a, m_begin, m_end, k_begin, k_end),
        )
        if position.global_tile >= self.prefetch_tiles or position.prefetched == 0:
            mem_copy(
                self.l1_b.next_write(),
                matrix_view(operands.b, n_begin, n_end, k_begin, k_end),
            )
        a_tile = self.l1_a.next_read()[0 : self.base_m, 0 : self.k_l1]
        b_tile = self.l1_b.next_read()[0 : self.base_n, 0 : self.k_l1]
        tiles = L1Operands(a_tile, b_tile, a_scales)
        self._compute_l0(tiles, state, position)

    @jit
    def _compute_l0(self, tiles, state, position):
        kk, local_kk = position.global_tile, position.local_tile
        steps = position.k // self.k_l1
        for j in range(self.step_k):
            kstep = kk * self.step_k + j
            a_operand = self.l0a.next_write()
            b_operand = self.l0b.next_write()
            a_slice = tile_slice(tiles.a, (self.base_m, self.base_k), (0, j))
            b_slice = tile_slice(tiles.b, (self.base_n, self.base_k), (0, j))
            a_scale = tile_slice(
                tiles.a_scale,
                (self.base_m, self.sk_l0),
                (0, local_kk * self.step_k + j),
            )
            b_scale = tile_slice(state.b_scale, (self.sk_l0, self.base_n), (kstep, 0))
            mem_copy(a_operand, a_slice, options=copy_options(mx_scale=a_scale))
            mem_copy(b_operand, b_slice, options=copy_options(mx_scale=b_scale))
            matmul(
                state.output,
                a_operand,
                b_operand,
                init=(kk == 0 and j == 0),
                unit_flag=3 if (kk == steps - 1 and j == self.step_k - 1) else 2,
                mx_mode=True,
            )

    def _init_l1_channels(self, reuse_n):
        storage = dsl.L1.view(524288) if reuse_n else None
        a_shape = (self.base_m, self.k_l1)
        b_shape = (self.base_n, self.k_l1)
        # The layout API uses (scale, M); shape-based builds require (M, scale)
        # with the legacy zn physical format for MX ScaleA operands.
        if _SCALE_CHANNEL_USES_LAYOUT:
            sa_shape = (self.scale_k // 32, self.base_m)
            sa_format = "nz"
        else:
            sa_shape = (self.base_m, self.scale_k // 32)
            sa_format = "zn"
        sb_shape = (self.sk_full, self.base_n)
        b_addr = self.base_m * self.k_l1 * 2
        sa_addr = (self.base_m + reuse_n) * self.k_l1 * 2
        sb_addr = sa_addr + self.base_m * self.sk_full
        self.l1_a = _operand_channel(
            storage, MemLoc.L1, a_shape, FP8, 2, data_format="nz"
        )
        self.l1_b = _operand_channel(
            storage, MemLoc.L1, b_shape, FP8, 2, offset=b_addr, data_format="nz"
        )
        self.l1_sa = _operand_channel(
            storage, MemLoc.L1, sa_shape, E8M0, 1, offset=sa_addr, data_format=sa_format
        )
        self.l1_sb = _operand_channel(
            storage, MemLoc.L1, sb_shape, E8M0, 1, offset=sb_addr, data_format="nz"
        )

    def _init_l0_channels(self, reuse_n, depth):
        a_storage = dsl.L0A.view(65536) if reuse_n else None
        b_storage = dsl.L0B.view(65536) if reuse_n else None
        c_storage = dsl.L0C.view(262144) if reuse_n else None
        a_shape = (self.base_m, self.base_k)
        b_shape = (self.base_n, self.base_k)
        c_shape = (self.base_m, self.base_n)
        self.l0a = _operand_channel(a_storage, MemLoc.L0A, a_shape, FP8, 2)
        self.l0b = _operand_channel(b_storage, MemLoc.L0B, b_shape, FP8, 2)
        self.l0c = _operand_channel(
            c_storage, MemLoc.L0C, c_shape, dtypes.float32, depth, element_bytes=4
        )


class QuantTile:
    """Quantize a full Cube tile using bounded Vec temporary storage."""

    def __init__(self, base_m, base_n, reserved_ub=65536):
        self.rows = int(base_m) // 2
        self.cols = int(base_n)
        # The caller reserves RoPE/QuantA storage, including input buffers.
        # The full FP32 tile stays live while bounded scratch is reused.
        choices = []
        for rows in range(1, self.rows + 1):
            count = rows * self.cols
            if count % 256:
                continue
            chunks = (self.rows + rows - 1) // rows
            padded = chunks * rows
            # Scratch per element: BF16 (2), packed FP8 (2), two byte
            # temporaries (1), reciprocal (1/8), and MX scale (1/32).
            # Padded scales need 32 bytes per row; leave alignment headroom.
            ub_bytes = (
                reserved_ub
                + 4 * padded * self.cols
                + count * 165 // 32
                + rows * 32
                + 512
            )
            if ub_bytes <= 262144:
                choices.append((chunks, padded - self.rows, -rows))
        if not choices:
            raise ValueError("no quant chunk fits UB")
        self.chunks, _, negative_rows = min(choices)
        self.chunk_rows = -negative_rows
        self.padded_rows = self.chunks * self.chunk_rows
        self.count = self.chunk_rows * self.cols
        count = self.count
        self.x = _make_ring_buffer(
            MemLoc.UB, dtypes.float32, (self.padded_rows, self.cols), depth=1
        )
        self.bf = _make_buffer(MemLoc.UB, dtypes.bfloat16, (1, count))
        self.q = _make_buffer(MemLoc.UB, dtypes.uint8, (1, 2 * count))
        self.s = _make_buffer(MemLoc.UB, dtypes.uint8, (1, count // 32 + 256))
        self.spad = _make_buffer(MemLoc.UB, dtypes.uint8, (self.chunk_rows, 32))
        self.recip = _make_buffer(MemLoc.UB, dtypes.uint16, (count // 16,))
        self.qz = _make_buffer(MemLoc.UB, dtypes.uint8, (count // 2,))
        self.qo = _make_buffer(MemLoc.UB, dtypes.uint8, (count // 2,))

    @jit
    def run(self, accumulator, output, tile):
        actual_m = tile.rows
        if const_expr(self.padded_rows == self.rows):
            mem_copy(
                self.x.next_write(),
                accumulator.partition(dim=0),
                options=copy_options(unit_flag=3),
            )
        else:
            mem_copy(
                matrix_view(self.x.next_write(), 0, self.rows, 0, self.cols),
                accumulator,
                options=copy_options(unit_flag=3),
            )
        src = self.x.next_read()
        half_m = actual_m // 2
        if const_expr(self.chunks == 1):
            self._chunk(src, output, tile, 0, half_m)
        else:
            for rb in dsl_range(0, half_m, self.chunk_rows):
                valid = min(self.chunk_rows, half_m - rb)
                self._chunk(src, output, tile, rb, valid)

    @jit
    def _chunk(self, src, output, tile, rb, valid):
        gm_q, gm_s = output.data, output.scales
        row, col = tile.row, tile.col
        half_m, actual_n = tile.rows // 2, tile.cols
        with vf(mode="simd"):
            mask = reg.full_mask()
            for offset in dsl_range(0, self.count, 64):
                value = reg.vload(src, rb * self.cols + offset)
                bf = reg.vcast(
                    value, dtypes.bfloat16, mask=mask, rounding=RoundingMode.RN
                )
                reg.vstore_pack(
                    self.bf, offset, bf, mask, pack_mode=reg.PackMode.B32_TO_B16
                )
        _quant_row(
            QuantBuffers(self.bf, self.q, self.s, self.recip, self.qz, self.qo),
            self.count,
        )
        with vf(mode="simd"):
            smask, _ = reg.update_mask(self.cols // 32, elem_bits=8)
            for row_index in dsl_range(0, self.chunk_rows, 1):
                offset = row_index * (self.cols // 32)
                reg.vload_unalign_init(self.s, offset)
                sv = reg.vload_unalign(self.s, offset)
                reg.vstore(self.spad, row_index * 32, sv, smask)
        begin = row + get_subblock_id() * half_m + rb
        q = self.q[0:1, 0 : self.count].reshape(self.chunk_rows, self.cols)
        mem_copy(
            matrix_view(gm_q, begin, begin + valid, col, col + actual_n),
            q[0:valid, 0:actual_n],
        )
        mem_copy(
            matrix_view(gm_s, begin, begin + valid, col // 32, (col + actual_n) // 32),
            matrix_view(self.spad, 0, valid, 0, actual_n // 32),
        )


class RopeQuant:
    """inverse RoPE + OCP MX quant for T tokens -> aq (g*T+t, F) / asc."""

    def __init__(self, ng=NG, f=F, stream_k=0, batch_rows=1):
        self.ng = int(ng)
        self.rows = int(batch_rows)
        self.full_f = int(f)
        self.stream_k = int(stream_k)
        self.f = self.stream_k or self.full_f
        self.num_heads = self.f // HEAD_DIM
        self.ngroups = self.f // QUANT_GROUP
        self.count = self.rows * self.f
        self.input_depth = 2 if self.stream_k else 1
        self.extra_ub = (self.input_depth - 1) * (
            2 * self.count + 8 * self.rows * ROPE_DIM
        )
        self.in_ch = _make_ring_buffer(
            MemLoc.UB, dtypes.bfloat16, (self.rows, self.f), depth=self.input_depth
        )
        self.out_ch = _make_ring_buffer(
            MemLoc.UB, dtypes.bfloat16, (self.rows, self.f), depth=1
        )
        # q_ch is 2x: reg.vstore_interleave(b8) writes ~2x the logical bytes.
        self.q_ch = _make_ring_buffer(
            MemLoc.UB, dtypes.uint8, (1, 2 * self.count), depth=1
        )
        self.s_ch = _make_ring_buffer(
            MemLoc.UB, dtypes.uint8, (self.rows, self.ngroups), depth=1
        )
        self.cos_ch = _make_ring_buffer(
            MemLoc.UB, dtypes.float32, (self.rows, ROPE_DIM), depth=self.input_depth
        )
        self.sin_ch = _make_ring_buffer(
            MemLoc.UB, dtypes.float32, (self.rows, ROPE_DIM), depth=self.input_depth
        )
        self.rope_buf = _make_buffer(
            MemLoc.UB, dtypes.float32, (self.rows * self.num_heads, ROPE_DIM)
        )
        nchunk = self.count // 256
        self.recip_f = _make_buffer(MemLoc.UB, dtypes.uint16, (nchunk * 16,))
        self.scr_qz = _make_buffer(MemLoc.UB, dtypes.uint8, (nchunk * 128,))
        self.scr_qo = _make_buffer(MemLoc.UB, dtypes.uint8, (nchunk * 128,))

    @jit
    def run(self, inputs, output, num_token, m_pad):
        if const_expr(self.stream_k):
            self._run_stream(inputs, output, num_token, m_pad)
        else:
            self._run_full(inputs, output, num_token, m_pad)

    @jit
    def _run_full(self, inputs, output, num_token, m_pad):
        gm_o, gm_cos, gm_sin = inputs.source, inputs.cos, inputs.sin
        gm_aq, gm_asc = output.data, output.scales
        ng = self.ng
        f = self.f
        ngroups = self.ngroups
        aiv = get_block_idx() * get_subblock_dim() + get_subblock_id()
        naiv = get_block_num() * get_subblock_dim()
        total = num_token * ng
        tpb = (total + naiv - 1) // naiv
        for it in dsl_range(0, tpb, 1):
            u = aiv + it * naiv
            if u < total:
                t = u // ng
                g = u % ng
                mem_copy(
                    self.in_ch.next_write(), tile_slice(gm_o, (1, f), (t * ng + g, 0))
                )
                mem_copy(
                    self.cos_ch.next_write(), tile_slice(gm_cos, (1, ROPE_DIM), (t, 0))
                )
                mem_copy(
                    self.sin_ch.next_write(), tile_slice(gm_sin, (1, ROPE_DIM), (t, 0))
                )
                self.apply_inverse_rope()
                self.quant_row_fast()
                mem_copy(
                    tile_slice(gm_aq, (1, f), (g * m_pad + t, 0)),
                    tile_slice(self.q_ch.next_read(), (1, f), (0, 0)),
                )
                mem_copy(
                    tile_slice(gm_asc, (1, ngroups), (g * m_pad + t, 0)),
                    tile_slice(self.s_ch.next_read(), (1, ngroups), (0, 0)),
                )

    @jit
    def _run_stream(self, inputs, output, num_token, m_pad):
        aiv = get_block_idx() * get_subblock_dim() + get_subblock_id()
        naiv = get_block_num() * get_subblock_dim()
        lanes = naiv // self.ng
        group, lane = aiv % self.ng, aiv // self.ng
        source = inputs.source.view(num_token, self.ng * self.full_f)
        views = RopeInputs(source, inputs.cos, inputs.sin)
        for chunk in range(self.full_f // self.f):
            for t in dsl_range(lane * self.rows, num_token, lanes * self.rows):
                actual = min(self.rows, num_token - t)
                batch = RopeBatch(t, group, chunk, actual)
                self._stream_batch(views, output, batch, m_pad)
            # Publish immutable slab data only after all vector stores complete.
            vec_sync_block_arrive(PIPE.MTE3, 11 + chunk, mode=0)
            vec_sync_block_wait(PIPE.MTE3, 11 + chunk, mode=0)
            vec_sync_block_arrive(PIPE.MTE3, 3 + chunk)

    @jit
    def _stream_batch(self, inputs, output, batch, m_pad):
        t, actual, f = batch.row, batch.count, self.f
        row_end = t + actual
        col_begin = batch.group * self.full_f + batch.chunk * f
        col_end = col_begin + f
        mem_copy(
            self.in_ch.next_write()[0:actual, 0:f],
            matrix_view(inputs.source, t, row_end, col_begin, col_end),
        )
        mem_copy(
            self.cos_ch.next_write()[0:actual, 0:ROPE_DIM],
            matrix_view(inputs.cos, t, row_end, 0, ROPE_DIM),
        )
        mem_copy(
            self.sin_ch.next_write()[0:actual, 0:ROPE_DIM],
            matrix_view(inputs.sin, t, row_end, 0, ROPE_DIM),
        )
        self.apply_inverse_rope()
        self.quant_row_fast()
        self._store_stream(output, batch, m_pad)

    @jit
    def _store_stream(self, output, batch, m_pad):
        actual, f, ngroups = batch.count, self.f, self.ngroups
        begin = batch.group * m_pad + batch.row
        end = begin + actual
        q_begin, q_end = batch.chunk * f, (batch.chunk + 1) * f
        s_begin = batch.chunk * ngroups
        s_end = (batch.chunk + 1) * ngroups
        q = self.q_ch.next_read()[0:1, 0 : self.count].reshape(self.rows, f)
        mem_copy(matrix_view(output.data, begin, end, q_begin, q_end), q[0:actual, 0:f])
        mem_copy(
            matrix_view(output.scales, begin, end, s_begin, s_end),
            self.s_ch.next_read()[0:actual, 0:ngroups],
        )

    @jit
    def quant_row_fast(self):
        _quant_row(
            QuantBuffers(
                self.out_ch.next_read(),
                self.q_ch.next_write(),
                self.s_ch.next_write(),
                self.recip_f,
                self.scr_qz,
                self.scr_qo,
            ),
            self.count,
        )

    @jit
    def apply_inverse_rope(self):
        num_heads = self.num_heads
        src = self.in_ch.next_read()
        cos = self.cos_ch.next_read()
        sin = self.sin_ch.next_read()
        dst = self.out_ch.next_write()
        with vf(mode="simd"):
            full = reg.full_mask()
            if const_expr(self.stream_k):
                # Copy BF16 directly, then overwrite the rotated head tails.
                # The following VST->VLD fence also completes these stores.
                copy_mask = reg.create_mask("all", elem_bits=16)
                for offset in dsl_range(0, self.count, 128):
                    reg.vstore(dst, offset, reg.vload(src, offset), copy_mask)
            for h in dsl_range(0, self.rows * num_heads, 1):
                xu = reg.vload_unpack(
                    src, h * HEAD_DIM + NOPE_DIM, unpack_mode=reg.UnpackMode.B16_TO_B32
                )
                reg.vstore(
                    self.rope_buf,
                    h * ROPE_DIM,
                    reg.vcast(xu, dtypes.float32, mask=full),
                    full,
                )
            reg.vmem_bar(mode="vst_vld")
            cos_lanes, _ = reg.vload_deinterleave(cos, 0)
            sin_lanes, _ = reg.vload_deinterleave(sin, 0)
            for h in dsl_range(0, self.rows * num_heads, 1):
                if const_expr(self.stream_k):
                    cos_lanes, _ = reg.vload_deinterleave(
                        cos, (h // num_heads) * ROPE_DIM
                    )
                    sin_lanes, _ = reg.vload_deinterleave(
                        sin, (h // num_heads) * ROPE_DIM
                    )
                base = h * HEAD_DIM
                if const_expr(not self.stream_k):
                    self._copy_nope(src, dst, base, full)
                self._rotate_tail(dst, h, cos_lanes, sin_lanes, full)

    @jit
    def _copy_nope(self, src, dst, base, full):
        for c in dsl_range(0, NOPE_CHUNKS, 1):
            reg.vstore_pack(
                dst,
                base + c * 64,
                reg.vload_unpack(
                    src,
                    base + c * 64,
                    unpack_mode=reg.UnpackMode.B16_TO_B32,
                ),
                full,
                pack_mode=reg.PackMode.B32_TO_B16,
            )

    @jit
    def _rotate_tail(self, dst, h, cos_lanes, sin_lanes, full):
        even, odd = reg.vload_deinterleave(self.rope_buf, h * ROPE_DIM)
        new_even = reg.vadd(
            reg.vmul(even, cos_lanes, mask=full),
            reg.vmul(odd, sin_lanes, mask=full),
            mask=full,
        )
        new_odd = reg.vsub(
            reg.vmul(odd, cos_lanes, mask=full),
            reg.vmul(even, sin_lanes, mask=full),
            mask=full,
        )
        merged, _ = reg.vinterleave(new_even, new_odd)
        reg.vstore_pack(
            dst,
            h * HEAD_DIM + NOPE_DIM,
            reg.vcast(merged, dtypes.bfloat16, mask=full, rounding=RoundingMode.RN),
            full,
            pack_mode=reg.PackMode.B32_TO_B16,
        )


def validate_tensors(device, specs):
    for name, tensor, shape, dtype in specs:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def validate_inputs(t, inputs, weights):
    o, rope_cos, rope_sin = inputs.source, inputs.cos, inputs.sin
    woaq, dwa = weights.first, weights.first_scale
    wobq, dwb = weights.second, weights.second_scale
    validate_tensors(
        o.device,
        (
            ("o", o, (t * NG, F), torch.bfloat16),
            ("rope_cos", rope_cos, (t, ROPE_DIM), torch.float32),
            ("rope_sin", rope_sin, (t, ROPE_DIM), torch.float32),
            ("woaq", woaq, (NG * O_LORA, F), torch.float8_e4m3fn),
            ("dwa", dwa, (NG * O_LORA, F // MX_DIVISOR, 2), torch.uint8),
            ("wobq", wobq, (DIM, NG * O_LORA), torch.float8_e4m3fn),
            ("dwb", dwb, (DIM, NG * O_LORA // MX_DIVISOR, 2), torch.uint8),
        ),
    )


_SCRATCH = OrderedDict()
SCRATCH_LOCK = threading.RLock()
_CACHE_LIMIT = 8


def get_scratch(device, t, capture):
    stream = torch.npu.current_stream(device)
    key = (device, stream.npu_stream, t)
    # Capture gets graph-pool storage, independent of eager calls and other graphs.
    with SCRATCH_LOCK:
        if not capture and key in _SCRATCH:
            _SCRATCH.move_to_end(key)
            return _SCRATCH[key]
        m = (t + 15) // 16 * 16
        k = NG * O_LORA
        rows = NG * m
        specs = [
            ((rows, 4096), torch.uint8),
            ((rows, 128), torch.uint8),
            ((m, k), torch.bfloat16),
            ((m, k), torch.uint8),
            ((m, k // 32), torch.uint8),
        ]
        buffers = tuple(
            torch.empty(shape, dtype=dtype, device=device) for shape, dtype in specs
        )
        if not capture:
            _SCRATCH[key] = buffers
            if len(_SCRATCH) > _CACHE_LIMIT:
                _SCRATCH.popitem(last=False)
        return buffers


def validate_network_inputs(inputs, weights):
    o, rope_cos, rope_sin = inputs.source, inputs.cos, inputs.sin
    woa, descale_woa = weights.first, weights.first_scale
    wob, descale_wob = weights.second, weights.second_scale
    if (
        not isinstance(o, torch.Tensor)
        or o.ndim != 3
        or tuple(o.shape[1:]) != (64, 512)
    ):
        raise ValueError("o must have shape (T, 64, 512)")
    t = int(o.shape[0])
    device = o.device
    if device.type not in ("npu", "privateuseone"):
        raise ValueError("attn_epilogue requires NPU inputs")
    validate_tensors(
        device,
        (
            ("o", o, (t, 64, 512), torch.bfloat16),
            ("woa", woa, (8, 1024, 4096), torch.float8_e4m3fn),
            ("wob", wob, (5120, 8192), torch.float8_e4m3fn),
            ("descale_woa", descale_woa, (8, 1024, 64, 2), torch.float8_e8m0fnu),
            ("descale_wob", descale_wob, (5120, 128, 2), torch.float8_e8m0fnu),
            ("rope_sin", rope_sin, (t, 64), torch.float32),
            ("rope_cos", rope_cos, (t, 64), torch.float32),
        ),
    )
    # Storage-format queries need real NPU storage; while dynamo traces, inputs are
    # FakeTensors and npu.get_npu_format has no fake kernel, so skip the check here.
    if not torch.compiler.is_compiling():
        for name, tensor in (
            ("o", o),
            ("woa", woa),
            ("descale_woa", descale_woa),
            ("descale_wob", descale_wob),
            ("rope_sin", rope_sin),
            ("rope_cos", rope_cos),
        ):
            if torch_npu.get_npu_format(tensor) not in (0, 2):
                raise ValueError(f"{name} must use ND storage")
        if torch_npu.get_npu_format(wob) not in (0, 2):
            raise ValueError(
                "wob must use ND storage; direct FRACTAL_NZ support is pending"
            )
    return t
