# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Deterministic mega_compressor: projection, pooling, RMSNorm and ring cache.

Single-file BSH/TH implementation for H5120/D512, B1/B4, ratio 1/2,
S1/S6/S8 decode and S8192 prefill. Static preparation stays outside execution.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Optional
import weakref

import torch
import cannbotdsl as cb
from cannbotdsl.aot import export
from cannbotdsl import Buffer, MemLoc, RingBuffer, Tensor, dtypes, get_platform_info
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import cube as cube_control
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import (
    get_block_idx,
    get_block_num,
    get_subblock_dim,
    get_subblock_id,
)
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops.sync import (
    PIPE,
    global_sync_all,
    cube_sync_all,
    vec_sync_all,
    vec_sync_block_wait,
    vec_sync_block_arrive,
)
from cannbotdsl.tensor import ceil_div, make_layout, tile_slice, reinterpret


# ---- General kernels ----

_M_TILE = 32
_K_TILE = 128
_D_TILE = 128
_VEC_D_TILE = 64
FP32_VECTOR_LANES = 64
_MAX_RATIO = 128
_MIN_HIDDEN = 1024
_MAX_HIDDEN = 10240
_HIDDEN_ALIGN = 512
_MIN_BLOCK_SIZE = 1
_MAX_BLOCK_SIZE = 1024
_DECODE_MAX_ROWS = 256
_SPLITK_KG = 4
_SPLITK_DL = 64
_SPLITK_NL = 64
_STREAM_M_TILE = 128
_STREAM_D_TILE = 128
_STREAM_L1_K_TILE = 256
_STREAM_L0_K_TILE = 128
_MIX_VECTOR_PARTS = 2
_STREAM_VECTOR_ROWS = _STREAM_M_TILE // _MIX_VECTOR_PARTS
_CUBE_ROW_ALIGNMENT = 16
_COPY_ELEMENTS = 256


def _effective_cube_cores(device=None):
    device_index = None if device is None else device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    stream = torch.npu.current_stream(device_index)
    info = get_platform_info(stream=stream)
    effective_cores = min(
        int(info.cube_core_num), int(info.vector_core_num) // _MIX_VECTOR_PARTS
    )
    if effective_cores <= 0:
        raise RuntimeError(
            "unable to query the effective Cube core count for the current stream"
        )
    return effective_cores


class _MetadataHost:
    def __init__(self, values, device):
        self.values = tuple(values)
        self.block_count = min(_effective_cube_cores(device), 1)

    @kernel
    def compute(self, output: Tensor):
        for index in range_constexpr(len(self.values)):
            output[index] = self.values[index]

    @host
    def run(self, output: Tensor):
        self.compute[self.block_count](output)


def _metadata_values(tensor, default):
    return list(default) if tensor is None else tensor.detach().cpu().tolist()


def _device_metadata(tensor, values, device):
    if tensor is not None and tensor.device == device and tensor.is_contiguous():
        return tensor
    output = torch.empty((len(values),), dtype=torch.int32, device=device)
    if values:
        _MetadataHost(values, device).run(output)
    return output


class _ContiguousHost:
    def __init__(self, shape, device):
        self.shape = tuple(shape)
        self.tile_shape = (1,) * (len(shape) - 1) + (_COPY_ELEMENTS,)
        self.rows = math.prod(shape[:-1])
        if shape[-1] % _COPY_ELEMENTS:
            raise ValueError("input width must be a multiple of the copy tile")
        self.chunks = shape[-1] // _COPY_ELEMENTS
        self.block_count = min(_effective_cube_cores(device), self.rows * self.chunks)

    @kernel
    def compute(self, source: Tensor, output: Tensor):
        buffer = Buffer(MemLoc.UB, source.dtype, make_layout(self.tile_shape))
        for task in range(get_block_idx(), self.rows * self.chunks, get_block_num()):
            row, chunk = task // self.chunks, task % self.chunks
            if const_expr(len(self.shape) == 3):
                window = tile_slice(
                    source,
                    self.tile_shape,
                    (row // self.shape[1], row % self.shape[1], chunk),
                )
            else:
                window = tile_slice(source, self.tile_shape, (row, chunk))
            mem_copy(buffer, window, options=copy_options(engine="nddma"))
            mem_copy(
                tile_slice(output, (_COPY_ELEMENTS,), (task,)),
                buffer.view(_COPY_ELEMENTS),
            )

    @host
    def run(self, source: Tensor, output: Tensor):
        self.compute[self.block_count](source, output)


def _device_contiguous(tensor):
    if tensor.is_contiguous() or tensor.numel() == 0:
        return tensor
    output = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    _ContiguousHost(tensor.shape, tensor.device).run(tensor, output.view(-1))
    return output


class _RMSNorm:
    def __init__(self, batch, dim, ratio, is_th, output_stride, norm_eps):
        self.batch, self.dim, self.ratio = batch, dim, ratio
        self.is_th, self.output_stride = is_th, output_stride
        self.norm_eps = norm_eps

    @jit
    def run(self, out, gamma, starts, useds):
        """Consume rounded pooling rows after the existing publication barrier."""
        value_ub = Buffer(MemLoc.UB, out.dtype, make_layout((1, self.dim)))
        gamma_ub = Buffer(MemLoc.UB, gamma.dtype, make_layout((1, self.dim)))
        normalized = Buffer(MemLoc.UB, out.dtype, make_layout((1, self.dim)))
        stats = Buffer(MemLoc.UB, dtypes.float32, make_layout((64,)))
        mem_copy(gamma_ub, gamma)
        vector = get_block_idx() * 2 + get_subblock_id()
        vectors = get_block_num() * 2
        prefix = 0
        for owner in range(self.batch):
            count = (starts[owner] + useds[owner]) // self.ratio - starts[
                owner
            ] // self.ratio
            for row in range(vector, count, vectors):
                index = prefix + row
                if const_expr(not self.is_th):
                    index = owner * self.output_stride + row
                mem_copy(value_ub, tile_slice(out, (1, self.dim), (index, 0)))
                with vf(mode="simd"):
                    mask = rr.full_mask()
                    total = rr.vdups(0.0, dtypes.float32, mask=mask)
                    for col in range(self.dim // 64):
                        value = rr.vcast(
                            rr.vload_unpack(
                                value_ub, col * 64, unpack_mode=rr.UnpackMode.B16_TO_B32
                            ),
                            dtypes.float32,
                            mask=mask,
                        )
                        total = rr.vadd(
                            total, rr.vmul(value, value, mask=mask), mask=mask
                        )
                    rr.vstore_first(stats, 0, rr.vreduce_sum(total, mask=mask))
                with vf(mode="simd"):
                    mask = rr.full_mask()
                    variance = rr.vadds(
                        rr.vmuls(
                            rr.vload_broadcast(stats, 0), 1.0 / self.dim, mask=mask
                        ),
                        self.norm_eps,
                        mask=mask,
                    )
                    scale = rr.vdiv(
                        rr.vdups(1.0, dtypes.float32, mask=mask),
                        rr.vsqrt(variance, mask=mask),
                        mask=mask,
                    )
                    for col in range(self.dim // 64):
                        value = rr.vcast(
                            rr.vload_unpack(
                                value_ub, col * 64, unpack_mode=rr.UnpackMode.B16_TO_B32
                            ),
                            dtypes.float32,
                            mask=mask,
                        )
                        if const_expr(gamma.dtype == dtypes.float32):
                            weight = rr.vload(gamma_ub, col * 64)
                        else:
                            weight = rr.vcast(
                                rr.vload_unpack(
                                    gamma_ub,
                                    col * 64,
                                    unpack_mode=rr.UnpackMode.B16_TO_B32,
                                ),
                                dtypes.float32,
                                mask=mask,
                            )
                        result = rr.vcast(
                            rr.vmul(
                                rr.vmul(value, scale, mask=mask), weight, mask=mask
                            ),
                            out.dtype,
                            mask=mask,
                        )
                        rr.vstore_pack(
                            normalized,
                            col * 64,
                            result,
                            mask,
                            pack_mode=rr.PackMode.B32_TO_B16,
                        )
                mem_copy(tile_slice(out, (1, self.dim), (index, 0)), normalized)
            prefix += count


class _FusedStream:
    def __init__(
        self,
        batch,
        dim,
        hidden,
        ratio,
        is_th,
        seqlen,
        output_stride,
        slots,
        total,
        sync_base,
        block_dim,
        fused_norm=False,
        norm_eps=1e-20,
    ):
        self.fused_norm = fused_norm
        self.batch = batch
        self.dim = dim
        self.hidden = hidden
        self.ratio = int(ratio)
        self.is_th = is_th
        self.seqlen = seqlen
        self.output_stride = output_stride
        self.slots = slots
        self.total = total
        self.tail_start = max(0, total - _STREAM_M_TILE)
        self.sync_base = sync_base
        self.block_dim = int(block_dim)
        self.norm = _RMSNorm(
            self.batch, self.dim, self.ratio, self.is_th, self.output_stride, norm_eps
        )

    @jit
    def _pairs(self, out, kv, score, count, dtype16):
        with vf(mode="simd"):
            mask = rr.update_mask(64, elem_bits=32)[0]
            for row in range(count):
                for col in range(0, _STREAM_D_TILE, FP32_VECTOR_LANES):
                    a = rr.vload(score, (row * 2) * _STREAM_D_TILE + col)
                    b = rr.vload(score, (row * 2 + 1) * _STREAM_D_TILE + col)
                    m = rr.vmax(a, b, mask=mask)
                    ea = rr.vexp(rr.vsub(a, m, mask=mask), mask=mask)
                    eb = rr.vexp(rr.vsub(b, m, mask=mask), mask=mask)
                    den = rr.vadd(ea, eb, mask=mask)
                    pa = rr.vdiv(ea, den, mask=mask)
                    pb = rr.vdiv(eb, den, mask=mask)
                    va = rr.vload(kv, (row * 2) * _STREAM_D_TILE + col)
                    vb = rr.vload(kv, (row * 2 + 1) * _STREAM_D_TILE + col)
                    val = rr.vadd(
                        rr.vmul(pa, va, mask=mask),
                        rr.vmul(pb, vb, mask=mask),
                        mask=mask,
                    )
                    packed = rr.vcast(val, dtype16, mask=mask)
                    rr.vstore_pack(
                        out,
                        row * _STREAM_D_TILE + col,
                        packed,
                        mask,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )

    @jit
    def _single(self, out, kv, count, dtype16):
        with vf(mode="simd"):
            mask = rr.update_mask(64, elem_bits=32)[0]
            for row in range(count):
                for col in range(0, _STREAM_D_TILE, FP32_VECTOR_LANES):
                    val = rr.vload(kv, row * _STREAM_D_TILE + col)
                    packed = rr.vcast(val, dtype16, mask=mask)
                    rr.vstore_pack(
                        out,
                        row * _STREAM_D_TILE + col,
                        packed,
                        mask,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )

    @jit
    def _prefix(self, out, kv, score, old_kv, old_score, dtype16):
        with vf(mode="simd"):
            mask = rr.update_mask(64, elem_bits=32)[0]
            for col in range(0, _STREAM_D_TILE, FP32_VECTOR_LANES):
                a = rr.vload(old_score, col)
                b = rr.vload(score, col)
                m = rr.vmax(a, b, mask=mask)
                ea = rr.vexp(rr.vsub(a, m, mask=mask), mask=mask)
                eb = rr.vexp(rr.vsub(b, m, mask=mask), mask=mask)
                den = rr.vadd(ea, eb, mask=mask)
                pa = rr.vdiv(ea, den, mask=mask)
                pb = rr.vdiv(eb, den, mask=mask)
                va = rr.vload(old_kv, col)
                vb = rr.vload(kv, col)
                val = rr.vadd(
                    rr.vmul(pa, va, mask=mask), rr.vmul(pb, vb, mask=mask), mask=mask
                )
                packed = rr.vcast(val, dtype16, mask=mask)
                rr.vstore_pack(out, col, packed, mask, pack_mode=rr.PackMode.B32_TO_B16)

    @kernel
    def compute(
        self,
        out: Tensor,
        x: Tensor,
        tail: Tensor,
        wkv: Tensor,
        wgate: Tensor,
        state: Tensor,
        table: Tensor,
        cu: Tensor,
        useds: Tensor,
        starts: Tensor,
        cache_tmp: Tensor,
        norm_weight: Tensor = None,
    ):
        dtype16 = dtypes.bfloat16 if x.dtype == dtypes.bfloat16 else dtypes.float16
        x_l1 = RingBuffer(
            MemLoc.L1,
            dtype16,
            make_layout((_STREAM_M_TILE, _STREAM_L1_K_TILE)),
            depth=2,
        )
        w_l1 = RingBuffer(
            MemLoc.L1,
            dtype16,
            make_layout((2 * _STREAM_D_TILE, _STREAM_L1_K_TILE)),
            depth=2,
        )
        a_l0 = RingBuffer(
            MemLoc.L0A,
            dtype16,
            make_layout((_STREAM_M_TILE, _STREAM_L0_K_TILE)),
            depth=2,
        )
        b_l0 = RingBuffer(
            MemLoc.L0B,
            dtype16,
            make_layout((_STREAM_D_TILE, _STREAM_L0_K_TILE)),
            depth=2,
        )
        acc_kv = Buffer(
            MemLoc.L0C, dtypes.float32, make_layout((_STREAM_M_TILE, _STREAM_D_TILE))
        )
        acc_sc = Buffer(
            MemLoc.L0C, dtypes.float32, make_layout((_STREAM_M_TILE, _STREAM_D_TILE))
        )
        kv_cv = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((_STREAM_VECTOR_ROWS, _STREAM_D_TILE)),
            depth=2,
        )
        score_cv = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((_STREAM_VECTOR_ROWS, _STREAM_D_TILE)),
            depth=2,
        )
        out_ch = RingBuffer(
            MemLoc.UB,
            dtype16,
            make_layout((_STREAM_VECTOR_ROWS, _STREAM_D_TILE)),
            depth=2,
        )
        old_kv = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, _STREAM_D_TILE)))
        old_score = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, _STREAM_D_TILE)))
        cache_buf = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 2 * self.dim)))
        core = get_block_idx()
        cores = get_block_num()
        sub = get_subblock_id()
        tail_row = Buffer(MemLoc.UB, dtype16, make_layout((1, self.hidden)))
        for row in range(
            core * get_subblock_dim() + sub,
            2 * _STREAM_M_TILE,
            cores * get_subblock_dim(),
        ):
            if row < min(_STREAM_M_TILE, self.total):
                mem_copy(
                    tail_row,
                    tile_slice(x, (1, self.hidden), (self.tail_start + row, 0)),
                )
            else:
                with vf(mode="simd"):
                    mask = rr.full_mask()
                    zero = rr.vcast(
                        rr.vdups(0.0, dtypes.float32, mask=mask), dtype16, mask=mask
                    )
                    for column in range(self.hidden // FP32_VECTOR_LANES):
                        rr.vstore_pack(
                            tail_row,
                            column * FP32_VECTOR_LANES,
                            zero,
                            mask,
                            pack_mode=rr.PackMode.B32_TO_B16,
                        )
            mem_copy(tile_slice(tail, (1, self.hidden), (row, 0)), tail_row)
        global_sync_all(
            flag_ids=(self.sync_base, self.sync_base + 1, self.sync_base + 2)
        )
        d_tiles = self.dim // _STREAM_D_TILE
        total_tiles = 0
        for b in range(self.batch):
            used = useds[b]
            prefix = min(used, starts[b] % self.ratio)
            total_tiles += (
                (used - prefix + _STREAM_M_TILE - 1) // _STREAM_M_TILE + prefix
            ) * d_tiles

        for task in range(core, total_tiles, cores):
            owner = 0
            local_task = 0
            preceding = 0
            output_base = 0
            preceding_groups = 0
            for b in range(self.batch):
                used = useds[b]
                prefix = min(used, starts[b] % self.ratio)
                count = (
                    (used - prefix + _STREAM_M_TILE - 1) // _STREAM_M_TILE + prefix
                ) * d_tiles
                if task >= preceding and task < preceding + count:
                    owner = b
                    local_task = task - preceding
                    output_base = preceding_groups
                preceding += count
                preceding_groups += (starts[b] + used) // self.ratio - starts[
                    b
                ] // self.ratio
            used = useds[owner]
            begin = starts[owner]
            prefix = min(used, begin % self.ratio)
            m_idx = local_task // d_tiles
            d_idx = local_task % d_tiles
            token_start = (m_idx - prefix) * _STREAM_M_TILE + prefix
            prefix_task = self.ratio == 2 and prefix == 1 and m_idx == 0
            if prefix_task:
                token_start = 0
            if const_expr(self.is_th):
                seq_base = cu[owner]
            else:
                seq_base = owner * self.seqlen
            row_start = seq_base + token_start

            for slab in range(self.hidden // _STREAM_L1_K_TILE):
                x_slot = x_l1.next_write()
                if const_expr(self.total < _STREAM_M_TILE):
                    mem_copy(
                        x_slot,
                        tile_slice(
                            tail[row_start : row_start + _STREAM_M_TILE, None],
                            (_STREAM_M_TILE, _STREAM_L1_K_TILE),
                            (0, slab),
                        ),
                    )
                elif row_start + _STREAM_M_TILE <= self.total:
                    mem_copy(
                        x_slot,
                        tile_slice(
                            x[row_start : row_start + _STREAM_M_TILE, None],
                            (_STREAM_M_TILE, _STREAM_L1_K_TILE),
                            (0, slab),
                        ),
                    )
                else:
                    offset = row_start - self.tail_start
                    mem_copy(
                        x_slot,
                        tile_slice(
                            tail[offset : offset + _STREAM_M_TILE, None],
                            (_STREAM_M_TILE, _STREAM_L1_K_TILE),
                            (0, slab),
                        ),
                    )
                w_slot = w_l1.next_write()
                mem_copy(
                    tile_slice(w_slot, (_STREAM_D_TILE, _STREAM_L1_K_TILE), (0, 0)),
                    tile_slice(wkv, (_STREAM_D_TILE, _STREAM_L1_K_TILE), (d_idx, slab)),
                )
                mem_copy(
                    tile_slice(w_slot, (_STREAM_D_TILE, _STREAM_L1_K_TILE), (1, 0)),
                    tile_slice(
                        wgate, (_STREAM_D_TILE, _STREAM_L1_K_TILE), (d_idx, slab)
                    ),
                )
                x_read = x_l1.next_read()
                w_read = w_l1.next_read()
                for inner in range(_STREAM_L1_K_TILE // _STREAM_L0_K_TILE):
                    a_slot = a_l0.next_write()
                    mem_copy(
                        a_slot,
                        tile_slice(
                            x_read, (_STREAM_M_TILE, _STREAM_L0_K_TILE), (0, inner)
                        ),
                    )
                    bk = b_l0.next_write()
                    mem_copy(
                        bk,
                        tile_slice(
                            w_read, (_STREAM_D_TILE, _STREAM_L0_K_TILE), (0, inner)
                        ),
                    )
                    matmul(acc_kv, a_slot, bk, init=(slab == 0 and inner == 0))
                    bg = b_l0.next_write()
                    mem_copy(
                        bg,
                        tile_slice(
                            w_read, (_STREAM_D_TILE, _STREAM_L0_K_TILE), (1, inner)
                        ),
                    )
                    matmul(acc_sc, a_slot, bg, init=(slab == 0 and inner == 0))
            mem_copy(kv_cv.next_write(), acc_kv.partition(dim=0))
            mem_copy(score_cv.next_write(), acc_sc.partition(dim=0))

            kv = kv_cv.next_read()
            score = score_cv.next_read()
            valid = max(
                0,
                min(
                    _STREAM_VECTOR_ROWS, used - token_start - sub * _STREAM_VECTOR_ROWS
                ),
            )
            if prefix_task:
                valid = max(0, 1 - sub * _STREAM_VECTOR_ROWS)
                if sub == 0:
                    old = state[table[owner], (begin - 1) % self.slots, None].view(
                        1, 2 * self.dim
                    )
                    mem_copy(old_kv, tile_slice(old, (1, _STREAM_D_TILE), (0, d_idx)))
                    mem_copy(
                        old_score,
                        tile_slice(old, (1, _STREAM_D_TILE), (0, d_idx + d_tiles)),
                    )
                    dst = out_ch.next_write()
                    self._prefix(dst, kv, score, old_kv, old_score, dtype16)
                    ready = out_ch.next_read()
                    index = output_base
                    if const_expr(not self.is_th):
                        index = owner * self.output_stride
                    mem_copy(
                        tile_slice(
                            out[index : index + 1, None],
                            (1, _STREAM_D_TILE),
                            (0, d_idx),
                        ),
                        tile_slice(ready, (1, _STREAM_D_TILE), (0, 0)),
                    )
            else:
                if self.ratio == 1:
                    if valid > 0:
                        dst = out_ch.next_write()
                        self._single(dst, kv, valid, dtype16)
                        ready = out_ch.next_read()
                        index = output_base + token_start + sub * _STREAM_VECTOR_ROWS
                        if const_expr(not self.is_th):
                            index = (
                                owner * self.output_stride
                                + token_start
                                + sub * _STREAM_VECTOR_ROWS
                            )
                        if const_expr(self.total >= _STREAM_M_TILE):
                            if valid == _STREAM_VECTOR_ROWS:
                                out_tile = tile_slice(
                                    out[index : index + _STREAM_VECTOR_ROWS, None],
                                    (_STREAM_VECTOR_ROWS, _STREAM_D_TILE),
                                    (0, d_idx),
                                )
                                mem_copy(out_tile, ready)
                        if valid < _STREAM_VECTOR_ROWS or self.total < _STREAM_M_TILE:
                            for r_out in range(valid):
                                row_out = index + r_out
                                out_row = tile_slice(
                                    out[row_out : row_out + 1, None],
                                    (1, _STREAM_D_TILE),
                                    (0, d_idx),
                                )
                                mem_copy(
                                    out_row,
                                    tile_slice(ready, (1, _STREAM_D_TILE), (r_out, 0)),
                                )
                else:
                    pairs = valid // 2
                    if pairs > 0:
                        dst = out_ch.next_write()
                        self._pairs(dst, kv, score, pairs, dtype16)
                        ready = out_ch.next_read()
                        index = (
                            output_base
                            + prefix
                            + (token_start - prefix) // 2
                            + sub * (_STREAM_VECTOR_ROWS // 2)
                        )
                        if const_expr(not self.is_th):
                            index = (
                                owner * self.output_stride
                                + prefix
                                + (token_start - prefix) // 2
                                + sub * (_STREAM_VECTOR_ROWS // 2)
                            )
                        if const_expr(self.total >= _STREAM_M_TILE):
                            if pairs == _STREAM_VECTOR_ROWS // 2:
                                out_tile = tile_slice(
                                    out[index : index + _STREAM_VECTOR_ROWS // 2, None],
                                    (_STREAM_VECTOR_ROWS // 2, _STREAM_D_TILE),
                                    (0, d_idx),
                                )
                                mem_copy(
                                    out_tile,
                                    tile_slice(
                                        ready,
                                        (_STREAM_VECTOR_ROWS // 2, _STREAM_D_TILE),
                                        (0, 0),
                                    ),
                                )
                        if (
                            pairs < _STREAM_VECTOR_ROWS // 2
                            or self.total < _STREAM_M_TILE
                        ):
                            # Dynamic slices of a selected ring-buffer slot currently
                            # lose the selected slot during layout lowering. Keep
                            # the common full tile bulk DMA and use static row
                            # views for the rare partial tile.
                            for r_out in range(pairs):
                                row_out = index + r_out
                                out_row = tile_slice(
                                    out[row_out : row_out + 1, None],
                                    (1, _STREAM_D_TILE),
                                    (0, d_idx),
                                )
                                mem_copy(
                                    out_row,
                                    tile_slice(ready, (1, _STREAM_D_TILE), (r_out, 0)),
                                )
            # Stage only live final ring rows. Historical reads finish before
            # the global barrier, so ring wrap cannot race those reads.
            first = max(0, used - self.slots - token_start - sub * _STREAM_VECTOR_ROWS)
            for r in range(first, valid):
                token = token_start + sub * _STREAM_VECTOR_ROWS + r
                slot = (begin + token) % self.slots
                cache_row = cache_tmp[owner, slot, None].view(1, 2 * self.dim)
                mem_copy(
                    tile_slice(cache_row, (1, _STREAM_D_TILE), (0, d_idx)),
                    tile_slice(kv, (1, _STREAM_D_TILE), (r, 0)),
                )
                mem_copy(
                    tile_slice(cache_row, (1, _STREAM_D_TILE), (0, d_idx + d_tiles)),
                    tile_slice(score, (1, _STREAM_D_TILE), (r, 0)),
                )

        global_sync_all(
            flag_ids=(self.sync_base, self.sync_base + 1, self.sync_base + 2)
        )
        if const_expr(self.fused_norm):
            self.norm.run(out, norm_weight, starts, useds)

        for item in range(core * 2 + sub, self.batch * self.slots, cores * 2):
            b = item // self.slots
            i = item % self.slots
            used = useds[b]
            if i < min(used, self.slots):
                token = max(0, used - self.slots) + i
                slot = (starts[b] + token) % self.slots
                mem_copy(cache_buf, cache_tmp[b, slot, None].view(1, 2 * self.dim))
                mem_copy(state[table[b], slot, None].view(1, 2 * self.dim), cache_buf)

    @host
    def run(
        self,
        out: Tensor,
        x: Tensor,
        tail: Tensor,
        wkv: Tensor,
        wgate: Tensor,
        state: Tensor,
        table: Tensor,
        cu: Tensor,
        useds: Tensor,
        starts: Tensor,
        cache_tmp: Tensor,
    ) -> None:
        self.compute[self.block_dim](
            out, x, tail, wkv, wgate, state, table, cu, useds, starts, cache_tmp
        )

    @host
    def run_fused(
        self,
        out: Tensor,
        x: Tensor,
        tail: Tensor,
        wkv: Tensor,
        wgate: Tensor,
        state: Tensor,
        table: Tensor,
        cu: Tensor,
        useds: Tensor,
        starts: Tensor,
        cache_tmp: Tensor,
        norm_weight: Tensor,
    ) -> None:
        self.compute[self.block_dim](
            out,
            x,
            tail,
            wkv,
            wgate,
            state,
            table,
            cu,
            useds,
            starts,
            cache_tmp,
            norm_weight,
        )


class _SplitKDecode:
    """Split-K decode projection (structure from cannbot-dsl MR221).

    Each task owns one K shard and one D tile and loads its whole K shard into
    L1 in a single panel; KV and gate share one combined L0C accumulator.  The
    partials go to a GM workspace, then the AIVs reduce the shards at whole-row
    granularity before pooling and updating the ring.  The reference's fused
    RMSNorm is not used here; the public ring/B/S/ratio contract is kept.
    """

    def __init__(
        self,
        batch,
        dim,
        hidden,
        ratio,
        is_th,
        seqlen,
        output_stride,
        slots,
        total,
        sync_base,
        block_dim,
        kl1,
        l0_slots,
        kl0,
        fused_norm=False,
        norm_eps=1e-20,
    ):
        self.fused_norm = fused_norm
        self.batch = batch
        self.dim = dim
        self.hidden = hidden
        self.ratio = int(ratio)
        self.is_th = is_th
        self.seqlen = seqlen
        self.output_stride = output_stride
        self.slots = slots
        self.total = total
        self.rows = int(total)
        self.kg = _SPLITK_KG
        self.kl1 = int(kl1)
        self.l0_slots = int(l0_slots)
        self.kl0 = int(kl0)
        self.sync_base = sync_base
        self.block_dim = int(block_dim)
        self.norm = _RMSNorm(
            self.batch, self.dim, self.ratio, self.is_th, self.output_stride, norm_eps
        )

    @kernel
    def compute(
        self,
        out: Tensor,
        x: Tensor,
        wkv: Tensor,
        wgate: Tensor,
        state: Tensor,
        table: Tensor,
        cu: Tensor,
        useds: Tensor,
        starts: Tensor,
        workspace: Tensor,
        norm_weight: Tensor = None,
    ):
        dtype16 = dtypes.bfloat16 if x.dtype == dtypes.bfloat16 else dtypes.float16
        rows = self.rows
        DL = _SPLITK_DL
        NL = _SPLITK_NL
        KL0 = self.kl0
        KG = self.kg
        DT = self.dim // DL
        kl1 = self.kl1
        panels = (self.hidden // KG) // kl1
        x_l1 = RingBuffer(MemLoc.L1, dtype16, make_layout((rows, kl1)), depth=1)
        w_l1 = RingBuffer(MemLoc.L1, dtype16, make_layout((2 * DL, kl1)), depth=1)
        a_l0 = RingBuffer(
            MemLoc.L0A, dtype16, make_layout((rows, KL0)), depth=self.l0_slots
        )
        b_l0 = RingBuffer(MemLoc.L0B, dtype16, make_layout((NL, KL0)), depth=2)
        acc = Buffer(MemLoc.L0C, dtypes.float32, make_layout((rows, 2 * DL)))
        core = get_block_idx()
        for task in range(core, KG * DT, get_block_num()):
            kg_i = task % KG
            dg = task // KG
            for panel in range(panels):
                kpanel = kg_i * panels + panel
                xs = x_l1.next_write()
                mem_copy(xs, tile_slice(x, (rows, kl1), (0, kpanel)))
                w1 = w_l1.next_write()
                mem_copy(
                    tile_slice(w1, (DL, kl1), (0, 0)),
                    tile_slice(wkv, (DL, kl1), (dg, kpanel)),
                )
                mem_copy(
                    tile_slice(w1, (DL, kl1), (1, 0)),
                    tile_slice(wgate, (DL, kl1), (dg, kpanel)),
                )
                xr = x_l1.next_read()
                wr = w_l1.next_read()
                for ki in range(kl1 // KL0):
                    a = a_l0.next_write()
                    mem_copy(a, tile_slice(xr, (rows, KL0), (0, ki)))
                    a0 = a_l0.next_read()
                    for ni in range(2 * DL // NL):
                        b = b_l0.next_write()
                        mem_copy(b, tile_slice(wr, (NL, KL0), (ni, ki)))
                        b0 = b_l0.next_read()
                        matmul(
                            tile_slice(acc, (rows, NL), (0, ni)),
                            a0,
                            b0,
                            init=(panel == 0 and ki == 0),
                        )
            mem_copy(
                tile_slice(workspace, (rows, DL), (kg_i, dg)),
                tile_slice(acc, (rows, DL), (0, 0)),
            )
            mem_copy(
                tile_slice(workspace, (rows, DL), (kg_i, dg + DT)),
                tile_slice(acc, (rows, DL), (0, 1)),
            )

        global_sync_all(
            flag_ids=(self.sync_base, self.sync_base + 1, self.sync_base + 2)
        )

        units = self.dim // 64
        red = Buffer(MemLoc.UB, dtypes.float32, make_layout((self.ratio, 2 * self.dim)))
        tmp = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 2 * self.dim)))
        ex_d = Buffer(MemLoc.UB, dtypes.float32, make_layout((self.ratio, 64)))
        out_row = Buffer(MemLoc.UB, dtype16, make_layout((1, self.dim)))
        vector_idx = get_block_idx() * 2 + get_subblock_id()
        vector_num = get_block_num() * 2
        total_groups = 0
        for b in range(self.batch):
            total_groups += (starts[b] + useds[b]) // self.ratio - starts[
                b
            ] // self.ratio
        for gt in range(vector_idx, total_groups, vector_num):
            owner_batch = 0
            owner_group = 0
            prefix_groups = 0
            for b in range(self.batch):
                b_groups = (starts[b] + useds[b]) // self.ratio - starts[
                    b
                ] // self.ratio
                if gt >= prefix_groups and gt < prefix_groups + b_groups:
                    owner_batch = b
                    owner_group = gt - prefix_groups
                prefix_groups += b_groups
            if const_expr(self.is_th):
                seq_base = cu[owner_batch]
            else:
                seq_base = owner_batch * self.seqlen
            abs_start = starts[owner_batch]
            used = useds[owner_batch]
            group_begin = (abs_start // self.ratio + owner_group) * self.ratio
            for r in range(self.ratio):
                absolute = group_begin + r
                if absolute < abs_start:
                    row = state[table[owner_batch], absolute % self.slots, None].view(
                        1, 2 * self.dim
                    )
                    mem_copy(tile_slice(red, (1, 2 * self.dim), (r, 0)), row)
                elif absolute < abs_start + used:
                    gr = seq_base + (absolute - abs_start)
                    mem_copy(
                        tile_slice(red, (1, 2 * self.dim), (r, 0)),
                        tile_slice(workspace, (1, 2 * self.dim), (gr, 0)),
                    )
                    for s in range(1, KG):
                        mem_copy(
                            tmp,
                            tile_slice(
                                workspace, (1, 2 * self.dim), (s * rows + gr, 0)
                            ),
                        )
                        with vf(mode="simd"):
                            mask = rr.update_mask(64, elem_bits=32)[0]
                            for col in range(0, 2 * self.dim, 64):
                                rr.vstore(
                                    red,
                                    r * 2 * self.dim + col,
                                    rr.vadd(
                                        rr.vload(red, r * 2 * self.dim + col),
                                        rr.vload(tmp, col),
                                        mask=mask,
                                    ),
                                    mask,
                                )
                else:
                    with vf(mode="simd"):
                        mask = rr.update_mask(64, elem_bits=32)[0]
                        for col in range(0, 2 * self.dim, 64):
                            rr.vstore(
                                red,
                                r * 2 * self.dim + col,
                                rr.vdups(-3.4e38, dtypes.float32, mask=mask),
                                mask,
                            )
                        for col in range(0, self.dim, 64):
                            rr.vstore(
                                red,
                                r * 2 * self.dim + col,
                                rr.vdups(0.0, dtypes.float32, mask=mask),
                                mask,
                            )
            for unit in range(units):
                with vf(mode="simd"):
                    mask = rr.update_mask(64, elem_bits=32)[0]
                    maximum = rr.vdups(-3.4e38, dtypes.float32, mask=mask)
                    for r in range(self.ratio):
                        maximum = rr.vmax(
                            maximum,
                            rr.vload(red, r * 2 * self.dim + self.dim + unit * 64),
                            mask=mask,
                        )
                    denominator = rr.vdups(0.0, dtypes.float32, mask=mask)
                    for r in range(self.ratio):
                        e = rr.vexp(
                            rr.vsub(
                                rr.vload(red, r * 2 * self.dim + self.dim + unit * 64),
                                maximum,
                                mask=mask,
                            ),
                            mask=mask,
                        )
                        denominator = rr.vadd(denominator, e, mask=mask)
                        rr.vstore(ex_d, r * 64, e, mask)
                    rr.vmem_bar()
                    weighted = rr.vdups(0.0, dtypes.float32, mask=mask)
                    for r in range(self.ratio):
                        p = rr.vdiv(rr.vload(ex_d, r * 64), denominator, mask=mask)
                        weighted = rr.vadd(
                            weighted,
                            rr.vmul(
                                p,
                                rr.vload(red, r * 2 * self.dim + unit * 64),
                                mask=mask,
                            ),
                            mask=mask,
                        )
                    packed = rr.vcast(weighted, dtype16, mask=mask)
                    rr.vstore_pack(
                        out_row,
                        unit * 64,
                        packed,
                        mask,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )
            if const_expr(self.is_th):
                out_index = gt
            else:
                out_index = owner_batch * self.output_stride + owner_group
            mem_copy(
                tile_slice(out[out_index : out_index + 1, None], (1, self.dim), (0, 0)),
                out_row,
            )

        global_sync_all(
            flag_ids=(self.sync_base, self.sync_base + 1, self.sync_base + 2)
        )

        if const_expr(self.fused_norm):
            self.norm.run(out, norm_weight, starts, useds)

        cache_buf = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 2 * self.dim)))
        for item in range(vector_idx, self.batch * self.slots, vector_num):
            b = item // self.slots
            i = item % self.slots
            used = useds[b]
            if i < min(used, self.slots):
                token = max(0, used - self.slots) + i
                if const_expr(self.is_th):
                    seq_base = cu[b]
                else:
                    seq_base = b * self.seqlen
                gr = seq_base + token
                mem_copy(cache_buf, tile_slice(workspace, (1, 2 * self.dim), (gr, 0)))
                for s in range(1, KG):
                    mem_copy(
                        tmp,
                        tile_slice(workspace, (1, 2 * self.dim), (s * rows + gr, 0)),
                    )
                    with vf(mode="simd"):
                        mask = rr.update_mask(64, elem_bits=32)[0]
                        for col in range(0, 2 * self.dim, 64):
                            rr.vstore(
                                cache_buf,
                                col,
                                rr.vadd(
                                    rr.vload(cache_buf, col),
                                    rr.vload(tmp, col),
                                    mask=mask,
                                ),
                                mask,
                            )
                mem_copy(
                    state[table[b], (starts[b] + token) % self.slots, None].view(
                        1, 2 * self.dim
                    ),
                    cache_buf,
                )

    @host
    def run(
        self,
        out: Tensor,
        x: Tensor,
        wkv: Tensor,
        wgate: Tensor,
        state: Tensor,
        table: Tensor,
        cu: Tensor,
        useds: Tensor,
        starts: Tensor,
        workspace: Tensor,
    ) -> None:
        self.compute[self.block_dim](
            out, x, wkv, wgate, state, table, cu, useds, starts, workspace
        )

    @host
    def run_fused(
        self,
        out: Tensor,
        x: Tensor,
        wkv: Tensor,
        wgate: Tensor,
        state: Tensor,
        table: Tensor,
        cu: Tensor,
        useds: Tensor,
        starts: Tensor,
        workspace: Tensor,
        norm_weight: Tensor,
    ) -> None:
        self.compute[self.block_dim](
            out, x, wkv, wgate, state, table, cu, useds, starts, workspace, norm_weight
        )


class MegaCompressorKernel:
    """Mixed AIC/AIV kernel with Cube projection and Vector compression."""

    def __init__(
        self,
        head_dim: int,
        hidden_size: int,
        cmp_ratio: int,
        batch_size: int,
        x_is_th: bool,
        block_size: int,
        output_stride: int,
        sync_base: int = 0,
        sequence_length: int = 0,
        short_pipeline: bool = False,
        block_dim: int = 1,
        fused_norm=False,
        norm_eps=1e-20,
    ):
        self.fused_norm = fused_norm
        self.batch, self.dim = batch_size, head_dim
        self.is_th, self.ratio = x_is_th, cmp_ratio
        self.head_dim = int(head_dim)
        self.hidden_size = int(hidden_size)
        # Bound long FP32 accumulation chains and expose Cube parallelism for
        # decode. Source-supported H is 512-aligned, so four equal K ranges
        # are also aligned to the 128-element L0 tile.
        self.k_splits = 4 if hidden_size >= 2048 and hidden_size % 512 == 0 else 1
        self.k_chunk = int(hidden_size) // self.k_splits
        self.short_pipeline = bool(short_pipeline)
        self.vector_d_parts = (head_dim // _VEC_D_TILE) if short_pipeline else 1
        self.cmp_ratio = int(cmp_ratio)
        self.batch_size = int(batch_size)
        self.x_is_th = bool(x_is_th)
        self.block_size = int(block_size)
        self.output_stride = int(output_stride)
        self.sync_base = int(sync_base)
        self.sequence_length = (
            int(sequence_length)
            if sequence_length
            else int(output_stride * cmp_ratio)
            if not x_is_th
            else 0
        )
        self.cache_parallel = bool(
            short_pipeline
            and (
                self.sequence_length > batch_size
                if x_is_th
                else self.sequence_length > 1
            )
        )
        self.block_dim = int(block_dim)
        self.norm = _RMSNorm(
            self.batch, self.dim, self.ratio, self.is_th, self.output_stride, norm_eps
        )

    def _load_projection_tile(
        self,
        x_l1,
        weight_l1,
        l0a,
        l0b,
        x,
        weight,
        row_start,
        d_start,
        k_start,
        row_count,
        d_count,
        dtype16,
    ):
        """Load one public ND tile into the Cube-local ring buffers."""
        if const_expr(self.x_is_th):
            x_tile = tile_slice(
                x, (row_count, _K_TILE), (row_start // _M_TILE, k_start // _K_TILE)
            )
        else:
            x_flat = x.view(x.shape[0] * x.shape[1], x.shape[2])
            x_tile = tile_slice(
                x_flat, (row_count, _K_TILE), (row_start // _M_TILE, k_start // _K_TILE)
            )
        w_tile = tile_slice(
            weight, (d_count, _K_TILE), (d_start // _D_TILE, k_start // _K_TILE)
        )
        mem_copy(x_l1.next_write(), x_tile)
        mem_copy(weight_l1.next_write(), w_tile)
        a_slot = l0a.next_write()
        b_slot = l0b.next_write()
        mem_copy(a_slot, x_l1.next_read())
        mem_copy(b_slot, weight_l1.next_read())
        return a_slot, b_slot

    def _prefetched_projection(
        self,
        x_full,
        weight_full,
        l0a,
        l0b,
        l0c,
        x_nd,
        weight,
        dst,
        m_idx,
        d_idx,
        k_split,
    ):
        acc = l0c.next_write()
        # Two K128 L0 steps consume each K256 L1 slab. L1 depth2 permits
        # MTE2 to fetch the next slab while MTE1/MMAD use the current one.
        for slab in range(self.k_chunk // 256):
            k_slab = k_split * (self.k_chunk // 256) + slab
            x_slot = x_full.next_write()
            mem_copy(x_slot, tile_slice(x_nd, (_M_TILE, 256), (m_idx, k_slab)))
            w_slot = weight_full.next_write()
            mem_copy(
                w_slot,
                tile_slice(weight, (_D_TILE, 256), (d_idx, k_slab)),
            )
            x_read = x_full.next_read()
            w_read = weight_full.next_read()
            for inner in range(2):
                a_slot = l0a.next_write()
                b_slot = l0b.next_write()
                mem_copy(a_slot, tile_slice(x_read, (_M_TILE, _K_TILE), (0, inner)))
                mem_copy(b_slot, tile_slice(w_read, (_D_TILE, _K_TILE), (0, inner)))
                matmul(acc, a_slot, b_slot, init=(slab == 0 and inner == 0))
        mem_copy(dst, acc)

    @kernel
    def compute(
        self,
        cmp_kv_out: Tensor,
        x: Tensor,
        wkv: Tensor,
        wgate: Tensor,
        state_cache: Tensor,
        state_block_table: Tensor,
        cu_seqlens: Tensor,
        seqused: Tensor,
        start_pos: Tensor,
        workspace: Tensor,
        norm_weight: Tensor = None,
    ):
        dtype16 = dtypes.bfloat16 if x.dtype == dtypes.bfloat16 else dtypes.float16
        # Cube-owned local pipeline. Projection workspace is FP32 and has two
        # adjacent planes: [all T,D] KV followed by [all T,D] score.
        if const_expr(self.short_pipeline):
            # Larger GM transfers with double-buffered L1 and K128 L0 steps.
            # L1: 2*(32+128)*256*2 = 160 KiB.
            x_full = RingBuffer(
                MemLoc.L1, dtype16, make_layout((_M_TILE, 256)), depth=2
            )
            weight_full = RingBuffer(
                MemLoc.L1, dtype16, make_layout((_D_TILE, 256)), depth=2
            )
        else:
            x_l1 = RingBuffer(
                MemLoc.L1, dtype16, make_layout((_M_TILE, _K_TILE)), depth=2
            )
            weight_l1 = RingBuffer(
                MemLoc.L1, dtype16, make_layout((_D_TILE, _K_TILE)), depth=2
            )
        l0a = RingBuffer(MemLoc.L0A, dtype16, make_layout((_M_TILE, _K_TILE)), depth=2)
        l0b = RingBuffer(MemLoc.L0B, dtype16, make_layout((_D_TILE, _K_TILE)), depth=2)
        l0c = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((_M_TILE, _D_TILE)), depth=2
        )

        block_idx = get_block_idx()
        block_num = get_block_num()
        total_tokens = (
            x.shape[0] if const_expr(self.x_is_th) else x.shape[0] * x.shape[1]
        )
        d_tiles = ceil_div(self.head_dim, _D_TILE)
        m_tiles = ceil_div(total_tokens, _M_TILE)
        projection_tiles = m_tiles * d_tiles * self.k_splits

        # Every mixed launch executes Cube operations on AIC and Vector
        # operations on AIV. This branch is intentionally scalar-free: the
        # same logical tile id is used for both projection planes.
        for tile_id in range(block_idx, projection_tiles, block_num):
            k_split = tile_id % self.k_splits
            md_tile = tile_id // self.k_splits
            m_idx = md_tile // d_tiles
            d_idx = md_tile - m_idx * d_tiles
            row_start = m_idx * _M_TILE
            d_start = d_idx * _D_TILE
            # Static tile leaves are required by the current DSL compiler.
            # Public inputs are padded to these boundaries by the host wrapper.
            row_count = _M_TILE
            d_count = _D_TILE
            if row_count > 0 and d_count > 0:
                if const_expr(self.short_pipeline):
                    if const_expr(self.x_is_th):
                        x_nd = x
                    else:
                        x_nd = x.view(x.shape[0] * x.shape[1], x.shape[2])
                    kv_full_dst = tile_slice(
                        workspace,
                        (_M_TILE, _D_TILE),
                        (m_idx + k_split * m_tiles, d_idx),
                    )
                    self._prefetched_projection(
                        x_full,
                        weight_full,
                        l0a,
                        l0b,
                        l0c,
                        x_nd,
                        wkv,
                        kv_full_dst,
                        m_idx,
                        d_idx,
                        k_split,
                    )
                    score_full_dst = tile_slice(
                        workspace,
                        (_M_TILE, _D_TILE),
                        (m_idx + k_split * m_tiles, d_idx + d_tiles),
                    )
                    self._prefetched_projection(
                        x_full,
                        weight_full,
                        l0a,
                        l0b,
                        l0c,
                        x_nd,
                        wgate,
                        score_full_dst,
                        m_idx,
                        d_idx,
                        k_split,
                    )
                else:
                    acc_kv = l0c.next_write()
                    for k_offset in range(0, self.k_chunk, _K_TILE):
                        k_start = k_split * self.k_chunk + k_offset
                        a_slot, b_slot = self._load_projection_tile(
                            x_l1,
                            weight_l1,
                            l0a,
                            l0b,
                            x,
                            wkv,
                            row_start,
                            d_start,
                            k_start,
                            row_count,
                            d_count,
                            dtype16,
                        )
                        matmul(acc_kv, a_slot, b_slot, init=(k_offset == 0))
                    kv_tile = tile_slice(
                        workspace,
                        (_M_TILE, _D_TILE),
                        (m_idx + k_split * m_tiles, d_idx),
                    )
                    mem_copy(kv_tile, acc_kv)

                    acc_score = l0c.next_write()
                    for k_offset in range(0, self.k_chunk, _K_TILE):
                        k_start = k_split * self.k_chunk + k_offset
                        a_slot, b_slot = self._load_projection_tile(
                            x_l1,
                            weight_l1,
                            l0a,
                            l0b,
                            x,
                            wgate,
                            row_start,
                            d_start,
                            k_start,
                            row_count,
                            d_count,
                            dtype16,
                        )
                        matmul(acc_score, a_slot, b_slot, init=(k_offset == 0))
                    score_tile = tile_slice(
                        workspace,
                        (_M_TILE, _D_TILE),
                        (m_idx + k_split * m_tiles, d_idx + d_tiles),
                    )
                    mem_copy(score_tile, acc_score)

        # All Cube cores publish before any Vector core reads the workspace.
        global_sync_all(
            flag_ids=(self.sync_base, self.sync_base + 1, self.sync_base + 2)
        )

        # Vector-owned group tiles. FP32 buffers are sized for the
        # maximum supported ratio and one 64-wide feature window. The actual
        # ratio and D tail are handled by masks and scalar loop bounds.
        kv_ub = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((_MAX_RATIO, _VEC_D_TILE))
        )
        score_ub = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((_MAX_RATIO, _VEC_D_TILE))
        )
        exp_ub = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((_MAX_RATIO, _VEC_D_TILE))
        )
        out_ub = Buffer(MemLoc.UB, dtype16, make_layout((1, _VEC_D_TILE)))
        cache_ub = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, self.head_dim * 2)), depth=1
        )
        partial_kv = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, _VEC_D_TILE)))
        partial_score = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, _VEC_D_TILE)))
        partial_cache = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((1, self.head_dim * 2))
        )
        gather_kv = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.k_splits, _VEC_D_TILE))
        )
        gather_score = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.k_splits, _VEC_D_TILE))
        )
        plane_rows = workspace.shape[0] // self.k_splits
        vector_idx = get_block_idx() * 2 + get_subblock_id()
        vector_num = get_block_num() * 2

        # Flattened group numbering is deterministic for both BSH and TH.
        total_groups = 0
        for b in range(self.batch_size):
            total_groups += (start_pos[b] + seqused[b]) // self.cmp_ratio - start_pos[
                b
            ] // self.cmp_ratio
        for group_task in range(
            vector_idx, total_groups * self.vector_d_parts, vector_num
        ):
            flat_group = group_task // self.vector_d_parts
            owner_batch = 0
            owner_group = 0
            prefix_groups = 0
            for b in range(self.batch_size):
                b_groups = (start_pos[b] + seqused[b]) // self.cmp_ratio - start_pos[
                    b
                ] // self.cmp_ratio
                if (
                    flat_group >= prefix_groups
                    and flat_group < prefix_groups + b_groups
                ):
                    owner_batch = b
                    owner_group = flat_group - prefix_groups
                prefix_groups += b_groups

            if const_expr(self.x_is_th):
                seq_base = cu_seqlens[owner_batch]
                seq_len = cu_seqlens[owner_batch + 1] - seq_base
            else:
                seq_base = owner_batch * self.sequence_length
                seq_len = self.sequence_length
            abs_start = start_pos[owner_batch]
            if const_expr(seqused is not None):
                used = seqused[owner_batch]
            else:
                used = seq_len
            group_begin = (abs_start // self.cmp_ratio + owner_group) * self.cmp_ratio
            d_tiles_vec = ceil_div(self.head_dim, _VEC_D_TILE)
            d_first = 0
            d_last = d_tiles_vec
            if const_expr(self.vector_d_parts > 1):
                d_first = group_task % self.vector_d_parts
                d_last = d_first + 1
            for d_tile in range(d_first, d_last):
                d_start = d_tile * _VEC_D_TILE
                d_count = _VEC_D_TILE
                if d_count > 0:
                    score_slot = score_ub
                    kv_slot = kv_ub
                    for r in range(self.cmp_ratio):
                        absolute = group_begin + r
                        if absolute < abs_start:
                            block = state_block_table[owner_batch]
                            slot = absolute % self.block_size
                            state_row = state_cache[block, slot, None].view(
                                1, 2 * self.head_dim
                            )
                            mem_copy(
                                tile_slice(kv_slot, (1, d_count), (r, 0)),
                                tile_slice(
                                    state_row, (1, d_count), (0, d_start // _VEC_D_TILE)
                                ),
                            )
                            mem_copy(
                                tile_slice(score_slot, (1, d_count), (r, 0)),
                                tile_slice(
                                    state_row,
                                    (1, d_count),
                                    (0, (self.head_dim + d_start) // _VEC_D_TILE),
                                ),
                            )
                        elif absolute < abs_start + used:
                            row = seq_base + absolute - abs_start
                            if const_expr(self.short_pipeline):
                                for part in range(self.k_splits):
                                    part_row = tile_slice(
                                        workspace,
                                        (1, 2 * self.head_dim),
                                        (row + part * plane_rows, 0),
                                    )
                                    mem_copy(
                                        tile_slice(
                                            gather_kv, (1, _VEC_D_TILE), (part, 0)
                                        ),
                                        tile_slice(
                                            part_row, (1, _VEC_D_TILE), (0, d_tile)
                                        ),
                                    )
                                    mem_copy(
                                        tile_slice(
                                            gather_score, (1, _VEC_D_TILE), (part, 0)
                                        ),
                                        tile_slice(
                                            part_row,
                                            (1, _VEC_D_TILE),
                                            (0, self.head_dim // _VEC_D_TILE + d_tile),
                                        ),
                                    )
                                with vf(mode="simd"):
                                    mask = rr.update_mask(_VEC_D_TILE, elem_bits=32)[0]
                                    value = rr.vload(gather_kv, 0)
                                    score_value = rr.vload(gather_score, 0)
                                    for part in range(1, self.k_splits):
                                        value = rr.vadd(
                                            value,
                                            rr.vload(gather_kv, part * _VEC_D_TILE),
                                            mask=mask,
                                        )
                                        score_value = rr.vadd(
                                            score_value,
                                            rr.vload(gather_score, part * _VEC_D_TILE),
                                            mask=mask,
                                        )
                                    rr.vstore(kv_slot, r * _VEC_D_TILE, value, mask)
                                    rr.vstore(
                                        score_slot, r * _VEC_D_TILE, score_value, mask
                                    )
                            else:
                                proj_row = tile_slice(
                                    workspace, (1, 2 * self.head_dim), (row, 0)
                                )
                                mem_copy(
                                    tile_slice(kv_slot, (1, d_count), (r, 0)),
                                    tile_slice(
                                        proj_row,
                                        (1, d_count),
                                        (0, d_start // _VEC_D_TILE),
                                    ),
                                )
                                mem_copy(
                                    tile_slice(score_slot, (1, d_count), (r, 0)),
                                    tile_slice(
                                        proj_row,
                                        (1, d_count),
                                        (0, (self.head_dim + d_start) // _VEC_D_TILE),
                                    ),
                                )
                                if const_expr(self.k_splits > 1):
                                    for part in range(1, self.k_splits):
                                        part_row = tile_slice(
                                            workspace,
                                            (1, 2 * self.head_dim),
                                            (row + part * plane_rows, 0),
                                        )
                                        mem_copy(
                                            partial_kv,
                                            tile_slice(
                                                part_row,
                                                (1, d_count),
                                                (0, d_start // _VEC_D_TILE),
                                            ),
                                        )
                                        mem_copy(
                                            partial_score,
                                            tile_slice(
                                                part_row,
                                                (1, d_count),
                                                (
                                                    0,
                                                    (self.head_dim + d_start)
                                                    // _VEC_D_TILE,
                                                ),
                                            ),
                                        )
                                        with vf(mode="simd"):
                                            mask = rr.update_mask(
                                                _VEC_D_TILE, elem_bits=32
                                            )[0]
                                            a = rr.vload(kv_slot, r * _VEC_D_TILE)
                                            b = rr.vload(partial_kv, 0)
                                            rr.vstore(
                                                kv_slot,
                                                r * _VEC_D_TILE,
                                                rr.vadd(a, b, mask=mask),
                                                mask,
                                            )
                                            a_score = rr.vload(
                                                score_slot, r * _VEC_D_TILE
                                            )
                                            b_score = rr.vload(partial_score, 0)
                                            rr.vstore(
                                                score_slot,
                                                r * _VEC_D_TILE,
                                                rr.vadd(a_score, b_score, mask=mask),
                                                mask,
                                            )
                        else:
                            # Incomplete final groups are excluded by total_groups;
                            # this padded path only keeps the vector transaction
                            # count balanced for a dynamic tail.
                            with vf(mode="simd"):
                                mask = rr.update_mask(d_count, elem_bits=32)[0]
                                rr.vstore(
                                    score_slot,
                                    r * _VEC_D_TILE,
                                    rr.vdups(-3.4e38, dtypes.float32, mask=mask),
                                    mask,
                                )
                                rr.vstore(
                                    kv_slot,
                                    r * _VEC_D_TILE,
                                    rr.vdups(0.0, dtypes.float32, mask=mask),
                                    mask,
                                )
                    kv_read = kv_ub
                    score_read = score_ub
                    exp_slot = exp_ub
                    out_slot = out_ub
                    with vf(mode="simd"):
                        mask = rr.update_mask(d_count, elem_bits=32)[0]
                        maximum = rr.vdups(-3.4e38, dtypes.float32, mask=mask)
                        for r in range(self.cmp_ratio):
                            value = rr.vload(score_read, r * _VEC_D_TILE)
                            maximum = rr.vmax(maximum, value, mask=mask)
                        denominator = rr.vdups(0.0, dtypes.float32, mask=mask)
                        for r in range(self.cmp_ratio):
                            value = rr.vload(score_read, r * _VEC_D_TILE)
                            e = rr.vexp(rr.vsub(value, maximum, mask=mask), mask=mask)
                            denominator = rr.vadd(denominator, e, mask=mask)
                            rr.vstore(exp_slot, r * _VEC_D_TILE, e, mask)
                        rr.vmem_bar()
                        weighted = rr.vdups(0.0, dtypes.float32, mask=mask)
                        for r in range(self.cmp_ratio):
                            e = rr.vload(exp_slot, r * _VEC_D_TILE)
                            p = rr.vdiv(e, denominator, mask=mask)
                            value = rr.vload(kv_read, r * _VEC_D_TILE)
                            weighted = rr.vadd(
                                weighted, rr.vmul(p, value, mask=mask), mask=mask
                            )
                        if const_expr(dtype16 == dtypes.float32):
                            rr.vstore(out_slot, 0, weighted, mask)
                        else:
                            packed = rr.vcast(weighted, dtype16, mask=mask)
                            rr.vstore_pack(
                                out_slot,
                                0,
                                packed,
                                mask,
                                pack_mode=rr.PackMode.B32_TO_B16,
                            )
                    if const_expr(self.x_is_th):
                        out_index = flat_group
                    else:
                        out_index = owner_batch * self.output_stride + owner_group
                    mem_copy(
                        tile_slice(
                            cmp_kv_out[out_index : out_index + 1, None],
                            (1, d_count),
                            (0, d_tile),
                        ),
                        out_slot,
                    )

        global_sync_all(
            flag_ids=(self.sync_base, self.sync_base + 1, self.sync_base + 2)
        )

        if const_expr(self.fused_norm):
            self.norm.run(cmp_kv_out, norm_weight, start_pos, seqused)

        # Short prompts distribute cache rows across AIVs, avoiding a single
        # AIV serially reducing all six tokens in each batch. The generic
        # path retains its original per-batch schedule.
        cache_tasks = self.batch_size
        if const_expr(self.cache_parallel):
            cache_tasks = self.sequence_length
            if const_expr(not self.x_is_th):
                cache_tasks = self.batch_size * self.sequence_length
        for cache_task in range(vector_idx, cache_tasks, vector_num):
            b = cache_task
            local_token = 0
            if const_expr(self.cache_parallel):
                if const_expr(self.x_is_th):
                    b = 0
                    for candidate in range(self.batch_size):
                        if (
                            cache_task >= cu_seqlens[candidate]
                            and cache_task < cu_seqlens[candidate + 1]
                        ):
                            b = candidate
                    local_token = cache_task - cu_seqlens[b]
                else:
                    b = cache_task // self.sequence_length
                    local_token = cache_task % self.sequence_length
            if const_expr(self.x_is_th):
                seq_base = cu_seqlens[b]
                seq_len = cu_seqlens[b + 1] - seq_base
            else:
                seq_base = b * self.sequence_length
                seq_len = self.sequence_length
            begin = start_pos[b]
            if const_expr(seqused is not None):
                used = seqused[b]
            else:
                used = seq_len
            first = max(0, used - self.block_size)
            last = used
            if const_expr(self.cache_parallel):
                first = local_token
                last = min(used, local_token + 1)
                if local_token < max(0, used - self.block_size):
                    last = local_token
            block = state_block_table[b]
            for token in range(first, last):
                row = seq_base + token
                slot = (begin + token) % self.block_size
                proj = tile_slice(workspace, (1, 2 * self.head_dim), (row, 0))
                dst = state_cache[block, slot, None].view(1, 2 * self.head_dim)
                cache_slot = cache_ub.next_write()
                mem_copy(cache_slot, proj)
                if const_expr(self.k_splits > 1):
                    for part in range(1, self.k_splits):
                        part_row = tile_slice(
                            workspace,
                            (1, 2 * self.head_dim),
                            (row + part * plane_rows, 0),
                        )
                        mem_copy(partial_cache, part_row)
                        with vf(mode="simd"):
                            mask = rr.update_mask(_VEC_D_TILE, elem_bits=32)[0]
                            for col in range(0, 2 * self.head_dim, _VEC_D_TILE):
                                value = rr.vload(cache_slot, col)
                                partial = rr.vload(partial_cache, col)
                                rr.vstore(
                                    cache_slot,
                                    col,
                                    rr.vadd(value, partial, mask=mask),
                                    mask,
                                )
                cache_read = cache_ub.next_read()
                mem_copy(dst, cache_read)

    @host
    def run(
        self,
        cmp_kv_out: Tensor,
        x: Tensor,
        wkv: Tensor,
        wgate: Tensor,
        state_cache: Tensor,
        state_block_table: Tensor,
        cu_seqlens: Tensor,
        seqused: Tensor,
        start_pos: Tensor,
        workspace: Tensor,
    ) -> None:
        self.compute[self.block_dim](
            cmp_kv_out,
            x,
            wkv,
            wgate,
            state_cache,
            state_block_table,
            cu_seqlens,
            seqused,
            start_pos,
            workspace,
        )

    @host
    def run_fused(
        self,
        cmp_kv_out: Tensor,
        x: Tensor,
        wkv: Tensor,
        wgate: Tensor,
        state_cache: Tensor,
        state_block_table: Tensor,
        cu_seqlens: Tensor,
        seqused: Tensor,
        start_pos: Tensor,
        workspace: Tensor,
        norm_weight: Tensor,
    ) -> None:
        self.compute[self.block_dim](
            cmp_kv_out,
            x,
            wkv,
            wgate,
            state_cache,
            state_block_table,
            cu_seqlens,
            seqused,
            start_pos,
            workspace,
            norm_weight,
        )


# ---- Decode: projection, deterministic reduction, pooling and RMSNorm ----


@dataclass(frozen=True)
class DecodePlan:
    kg: int = 4
    dl: int = 64
    k: int = 160
    n: int = 64

    def workspace_shape(self, total):
        """One disjoint FP32 projection per K shard; no atomic accumulation."""
        return (total, self.kg * 1024)


def _validate_decode_plan(plan):
    if any(
        type(value) is not int or value <= 0
        for value in (plan.kg, plan.dl, plan.k, plan.n)
    ):
        raise ValueError("decode tile dimensions must be positive integers")
    if (
        512 % plan.dl
        or 5120 % (plan.kg * plan.k)
        or plan.dl % plan.n
        or plan.k % _CUBE_ROW_ALIGNMENT
    ):
        raise ValueError(
            "decode tiles must exactly partition D512 and K5120 with K16 alignment"
        )


class _PackWeightsLaunch:
    def __init__(self, plan, block_count):
        self.plan = plan
        self.panels = 5120 // plan.kg // plan.k
        self.chunks = 2 * plan.dl // plan.n
        self.tasks = plan.kg * (512 // plan.dl) * self.panels * self.chunks
        self.block_count = min(block_count, self.tasks)

    @jit
    def _copy_panel(self, buffer, weight, row_tile, k_tile):
        p = self.plan
        source = tile_slice(
            weight.view(512, 5120 // _CUBE_ROW_ALIGNMENT, _CUBE_ROW_ALIGNMENT),
            (p.n, p.k // _CUBE_ROW_ALIGNMENT, _CUBE_ROW_ALIGNMENT),
            (row_tile, k_tile, 0),
        )
        mem_copy(buffer, source.permute(1, 0, 2), options=copy_options(engine="nddma"))

    @kernel
    def compute(self, wkv: Tensor, wgate: Tensor, packed: Tensor):
        p = self.plan
        buffer = Buffer(
            MemLoc.UB,
            wkv.dtype,
            make_layout((p.k // _CUBE_ROW_ALIGNMENT, p.n, _CUBE_ROW_ALIGNMENT)),
        )
        for task in range(get_block_idx(), self.tasks, get_block_num()):
            projection_task = task // (self.panels * self.chunks)
            panel = task // self.chunks % self.panels
            chunk = task % self.chunks
            row_tile = projection_task // p.kg * (p.dl // p.n) + chunk % (
                self.chunks // 2
            )
            k_tile = projection_task % p.kg * self.panels + panel
            if chunk < self.chunks // 2:
                self._copy_panel(buffer, wkv, row_tile, k_tile)
            else:
                self._copy_panel(buffer, wgate, row_tile, k_tile)
            mem_copy(
                tile_slice(packed, (1, p.n * p.k), (task, 0)), buffer.view(1, p.n * p.k)
            )

    @host
    def run(self, wkv: Tensor, wgate: Tensor, packed: Tensor):
        self.compute[self.block_count](wkv, wgate, packed)


class _PackWeightsHost(_PackWeightsLaunch):
    def __init__(self, plan, device):
        super().__init__(plan, _effective_cube_cores(device))


def pack_weights(wkv, wgate, plan=DecodePlan()):
    _validate_decode_plan(plan)
    if (
        wkv.shape != (512, 5120)
        or wgate.shape != wkv.shape
        or wkv.dtype not in (torch.bfloat16, torch.float16)
        or wgate.dtype != wkv.dtype
        or wgate.device != wkv.device
    ):
        raise ValueError(
            "packing requires matching BF16/FP16 [512,5120] weights on one device"
        )
    if wkv.device.type == "cpu":
        return _pack_cpu_weights(wkv, wgate, plan)
    host_entry = _PackWeightsHost(plan, wkv.device)
    packed = torch.empty(
        (host_entry.tasks, plan.n * plan.k), dtype=wkv.dtype, device=wkv.device
    )
    host_entry.run(wkv, wgate, packed)
    return packed


def _pack_cpu_weights(wkv, wgate, plan):
    panels = 5120 // plan.kg // plan.k
    parts = []
    for feature_group in range(512 // plan.dl):
        for shard in range(plan.kg):
            for panel in range(panels):
                first_column = (shard * panels + panel) * plan.k
                for weight in (wkv, wgate):
                    for chunk in range(plan.dl // plan.n):
                        first_row = feature_group * plan.dl + chunk * plan.n
                        tile = weight[
                            first_row : first_row + plan.n,
                            first_column : first_column + plan.k,
                        ]
                        parts.append(
                            tile.reshape(
                                plan.n,
                                plan.k // _CUBE_ROW_ALIGNMENT,
                                _CUBE_ROW_ALIGNMENT,
                            )
                            .permute(1, 0, 2)
                            .reshape(-1)
                        )
    return torch.stack(parts)


class _DecodeKernel:
    """Shared BSH/TH decode; layout branches are resolved at compile time."""

    def __init__(self, batch, seq_len, ratio, slots, epsilon, plan, is_th=False):
        self.batch, self.seq = batch, seq_len
        self.ratio, self.slots = ratio, slots
        self.epsilon, self.p = epsilon, plan
        self.is_th = is_th
        self.total = seq_len if is_th else batch * seq_len
        self.m = self.total if is_th or self.total <= 64 else math.gcd(self.total, 64)
        self.output_stride = (seq_len + ratio - 1) // ratio
        self.x_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((self.m, plan.k)), depth=2
        )
        self.w_chunks = [
            RingBuffer(
                MemLoc.L1, dtypes.bfloat16, make_layout((plan.n, plan.k)), depth=2
            )
            for _ in range(2 * plan.dl // plan.n)
        ]
        self.a = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((self.m, plan.k)), depth=2
        )
        self.b = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((plan.n, plan.k)), depth=2
        )
        self.acc = Buffer(
            MemLoc.L0C, dtypes.float32, make_layout((self.m, 2 * plan.dl))
        )
        # Bound the reduction staging buffer to 96 KiB, including packed TH.
        self.partial_rows = min(self.seq, max(1, 24 // plan.kg))
        if seq_len in ((8, 32) if is_th else (8,)):
            # S8 uses four-row transfers, including packed B4/S8 (T=32).
            self.partial_rows = min(self.partial_rows, 4)
        self.partials = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.partial_rows, plan.kg * 1024))
        )
        self.current = Buffer(MemLoc.UB, dtypes.float32, make_layout((seq_len, 1024)))
        self.history = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 1024)))
        self.pool = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((self.output_stride, 512))
        )
        # Packed T=32 can belong to one request. Reuse pool to fit 237 KiB UB.
        self.output = (
            self.pool
            if is_th and seq_len == 32
            else Buffer(
                MemLoc.UB, dtypes.bfloat16, make_layout((self.output_stride, 512))
            )
        )
        self.gamma = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, 512)))
        self.statistics = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.output_stride * 64,))
        )

    # One writer per partial; explicit vadd order also stabilizes FP32 cache.
    @jit
    def _reduce_rows(self, base, count):
        with vf(mode="simd"):
            mask = rr.full_mask()
            for row in range(count):
                for col in range(16):
                    offset = row * self.p.kg * 1024 + col * 64
                    value = rr.vload(self.partials, offset)
                    for shard in range_constexpr(1, self.p.kg):
                        value = rr.vadd(
                            value,
                            rr.vload(self.partials, offset + shard * 1024),
                            mask=mask,
                        )
                    rr.vstore(self.current, (base + row) * 1024 + col * 64, value, mask)

    @jit
    def _pool_window(
        self, first: Tensor, first_row, second: Tensor, second_row, window
    ):
        with vf(mode="simd"):
            mask = rr.full_mask()
            cast_mask = rr.create_mask(pattern="all", elem_bits=16)
            total = rr.vdups(0.0, dtypes.float32, mask=mask)
            for col in range(8):
                a_offset, b_offset = (
                    first_row * 1024 + col * 64,
                    second_row * 1024 + col * 64,
                )
                s0 = rr.vload(first, a_offset + 512)
                s1 = rr.vload(second, b_offset + 512)
                maximum = rr.vmax(s0, s1, mask=mask)
                w0 = rr.vexp(rr.vsub(s0, maximum, mask=mask), mask=mask)
                w1 = rr.vexp(rr.vsub(s1, maximum, mask=mask), mask=mask)
                zero = rr.vdups(0.0, dtypes.float32, mask=mask)
                denominator = rr.vadd(rr.vadd(zero, w0, mask=mask), w1, mask=mask)
                a = rr.vmul(
                    rr.vdiv(w0, denominator, mask=mask),
                    rr.vload(first, a_offset),
                    mask=mask,
                )
                b = rr.vmul(
                    rr.vdiv(w1, denominator, mask=mask),
                    rr.vload(second, b_offset),
                    mask=mask,
                )
                result = rr.vcast(
                    rr.vadd(rr.vadd(zero, a, mask=mask), b, mask=mask),
                    dtypes.bfloat16,
                    mask=mask,
                )
                rr.vstore_pack(
                    self.pool,
                    window * 512 + col * 64,
                    result,
                    mask,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
                rounded = rr.vcast(
                    result, dtypes.float32, mask=cast_mask, reg_layout=rr.RegLayout.ZERO
                )
                total = rr.vadd(total, rr.vmul(rounded, rounded, mask=mask), mask=mask)
            rr.vstore_first(
                self.statistics, window * 64, rr.vreduce_sum(total, mask=mask)
            )

    @jit
    def _normalize(self, count):
        with vf(mode="simd"):
            mask = rr.full_mask()
            for window in range(count):
                variance = rr.vadds(
                    rr.vmuls(
                        rr.vload_broadcast(self.statistics, window * 64),
                        1.0 / 512,
                        mask=mask,
                    ),
                    self.epsilon,
                    mask=mask,
                )
                scale = rr.vdiv(
                    rr.vdups(1.0, dtypes.float32, mask=mask),
                    rr.vsqrt(variance, mask=mask),
                    mask=mask,
                )
                for col in range(8):
                    offset = window * 512 + col * 64
                    value = rr.vcast(
                        rr.vload_unpack(
                            self.pool, offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                        ),
                        dtypes.float32,
                        mask=mask,
                    )
                    gamma = rr.vcast(
                        rr.vload_unpack(
                            self.gamma, col * 64, unpack_mode=rr.UnpackMode.B16_TO_B32
                        ),
                        dtypes.float32,
                        mask=mask,
                    )
                    result = rr.vcast(
                        rr.vmul(rr.vmul(value, scale, mask=mask), gamma, mask=mask),
                        dtypes.bfloat16,
                        mask=mask,
                    )
                    rr.vstore_pack(
                        self.output,
                        offset,
                        result,
                        mask,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )

    @jit
    def _pool_single(self, row):
        with vf(mode="simd"):
            mask = rr.full_mask()
            cast_mask = rr.create_mask(pattern="all", elem_bits=16)
            total = rr.vdups(0.0, dtypes.float32, mask=mask)
            for col in range(8):
                value = rr.vcast(
                    rr.vload(self.current, row * 1024 + col * 64),
                    dtypes.bfloat16,
                    mask=mask,
                )
                rr.vstore_pack(
                    self.pool,
                    row * 512 + col * 64,
                    value,
                    mask,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
                fp32 = rr.vcast(
                    value, dtypes.float32, mask=cast_mask, reg_layout=rr.RegLayout.ZERO
                )
                total = rr.vadd(total, rr.vmul(fp32, fp32, mask=mask), mask=mask)
            rr.vstore_first(self.statistics, row * 64, rr.vreduce_sum(total, mask=mask))

    def __call__(
        self,
        x: Tensor,
        wkv: Tensor,
        gamma: Tensor,
        state: Tensor,
        starts: Tensor,
        table: Tensor,
        workspace: Tensor,
        output: Tensor,
        cu: Tensor = None,
        useds: Tensor = None,
    ):
        p = self.p
        core = get_block_idx()
        sub = get_subblock_id()
        cube_control.set_mmad_direction("n")
        for task in range(core, p.kg * (512 // p.dl), get_block_num()):
            kg, dg = task % p.kg, task // p.kg
            for mt in range(self.total // self.m):
                panels = 5120 // p.kg // p.k
                chunks = 2 * p.dl // p.n
                for panel in range_constexpr(panels):
                    mem_copy(
                        self.x_l1.next_write(),
                        tile_slice(x, (self.m, p.k), (mt, kg * panels + panel)),
                    )
                    a1 = self.x_l1.next_read()
                    mem_copy(self.a.next_write(), a1)
                    a0 = self.a.next_read()
                    for ni in range_constexpr(chunks):
                        weight_slot = self.w_chunks[ni].next_write()
                        mem_copy(
                            reinterpret(
                                weight_slot,
                                layout=make_layout((1, p.n * p.k)),
                                data_format="nd",
                            ),
                            tile_slice(
                                wkv,
                                (1, p.n * p.k),
                                ((task * panels + panel) * chunks + ni, 0),
                            ),
                        )
                        weight_l1 = self.w_chunks[ni].next_read()
                        mem_copy(self.b.next_write(), weight_l1)
                        weight_l0 = self.b.next_read()
                        matmul(
                            tile_slice(self.acc, (self.m, p.n), (0, ni)),
                            a0,
                            weight_l0,
                            init=(panel == 0),
                            unit_flag=(3 if panel == panels - 1 else 2),
                        )
                for ni in range_constexpr(2 * p.dl // p.n):
                    column = (
                        dg * p.dl // p.n + ni
                        if ni < p.dl // p.n
                        else (512 + dg * p.dl) // p.n + ni - p.dl // p.n
                    )
                    mem_copy(
                        tile_slice(
                            workspace, (self.m, p.n), (mt, kg * 1024 // p.n + column)
                        ),
                        tile_slice(self.acc, (self.m, p.n), (0, ni)),
                        options=copy_options(unit_flag=3),
                    )

        global_sync_all()
        for batch in range(
            core * get_subblock_dim() + sub,
            self.batch,
            get_block_num() * get_subblock_dim(),
        ):
            start, block = starts[batch], table[batch]
            # Position metadata is INT32; widen before summing so that a legal
            # upper-bound start cannot wrap the derived output counts.
            start64 = dtypes.int64(start)
            # Preserve scalar scheduling separately for each layout.
            if const_expr(self.is_th):
                used = useds[batch]
                count = (start64 + dtypes.int64(used)) // self.ratio - (
                    start64 // self.ratio
                )
                output_base = dtypes.int64(0)
                for previous in range(batch):
                    previous_start = dtypes.int64(starts[previous])
                    previous_end = previous_start + dtypes.int64(useds[previous])
                    output_base += (
                        previous_end // self.ratio - previous_start // self.ratio
                    )
            else:
                used = self.seq
                output_base = batch * self.output_stride
                count = (start64 + dtypes.int64(used)) // self.ratio - (
                    start64 // self.ratio
                )
            if const_expr(self.is_th):
                for group in range(used // self.partial_rows):
                    row = group * self.partial_rows
                    base = cu[batch] + row
                    mem_copy(
                        self.partials, workspace[base : base + self.partial_rows, None]
                    )
                    self._reduce_rows(row, self.partial_rows)
                for row in range(used // self.partial_rows * self.partial_rows, used):
                    mem_copy(
                        tile_slice(self.partials, (1, p.kg * 1024), (0, 0)),
                        tile_slice(workspace, (1, p.kg * 1024), (cu[batch] + row, 0)),
                    )
                    self._reduce_rows(row, 1)
            else:
                for row in range(
                    0, self.seq - self.seq % self.partial_rows, self.partial_rows
                ):
                    base = batch * self.seq + row
                    mem_copy(
                        self.partials, workspace[base : base + self.partial_rows, None]
                    )
                    self._reduce_rows(row, self.partial_rows)
                if const_expr(self.seq % self.partial_rows != 0):
                    tail = self.seq % self.partial_rows
                    row = self.seq - tail
                    mem_copy(
                        tile_slice(self.partials, (tail, p.kg * 1024), (0, 0)),
                        workspace[
                            batch * self.seq + row : (batch + 1) * self.seq, None
                        ],
                    )
                    self._reduce_rows(row, tail)
            mem_copy(self.gamma, gamma)
            if const_expr(self.ratio == 2):
                if start % 2 == 1:
                    mem_copy(
                        self.history,
                        tile_slice(
                            state,
                            (1, 1024),
                            (block * self.slots + (start - 1) % self.slots, 0),
                        ),
                    )
                for window in range(count):
                    first = window * 2 - start % 2
                    if first < 0:
                        self._pool_window(self.history, 0, self.current, 0, window)
                    else:
                        self._pool_window(
                            self.current, first, self.current, first + 1, window
                        )
            else:
                for window in range(used):
                    self._pool_single(window)
            self._normalize(count)
            for window in range(count):
                mem_copy(
                    tile_slice(output, (1, 512), (output_base + window, 0)),
                    tile_slice(self.output, (1, 512), (window, 0)),
                )
            for row in range(max(0, used - self.slots), used):
                mem_copy(
                    tile_slice(
                        state,
                        (1, 1024),
                        (block * self.slots + (start64 + row) % self.slots, 0),
                    ),
                    tile_slice(self.current, (1, 1024), (row, 0)),
                )
        cube_sync_all()
        vec_sync_all()


# Preserve the launch names while sharing all decode implementation.
@kernel
class DecodeKernel(_DecodeKernel):
    pass


@kernel
class THDecodeKernel(_DecodeKernel):
    pass


class _DecodeLaunch:
    def __init__(self, batch, seq_len, ratio, slots, epsilon, plan, block_count):
        _validate_decode_plan(plan)
        self.batch, self.seq = batch, seq_len
        self.ratio, self.slots = ratio, slots
        self.epsilon, self.plan = epsilon, plan
        self.block_count = min(block_count, plan.kg * (512 // plan.dl))

    @host
    def run(
        self,
        x: Tensor,
        packed: Tensor,
        gamma: Tensor,
        state: Tensor,
        starts: Tensor,
        table: Tensor,
        workspace: Tensor,
        output: Tensor,
    ):
        op = DecodeKernel(
            self.batch, self.seq, self.ratio, self.slots, self.epsilon, self.plan
        )
        op[self.block_count](x, packed, gamma, state, starts, table, workspace, output)


class DecodeHost(_DecodeLaunch):
    def __init__(
        self, batch, seq_len, ratio, slots, epsilon, plan=DecodePlan(), *, device=None
    ):
        super().__init__(
            batch, seq_len, ratio, slots, epsilon, plan, _effective_cube_cores(device)
        )


class _THDecodeLaunch:
    def __init__(self, batch, total, ratio, slots, epsilon, plan, block_count):
        _validate_decode_plan(plan)
        self.batch, self.total = batch, total
        self.ratio, self.slots = ratio, slots
        self.epsilon, self.plan = epsilon, plan
        self.block_count = min(block_count, plan.kg * (512 // plan.dl))

    @host
    def run(
        self,
        x: Tensor,
        packed: Tensor,
        gamma: Tensor,
        state: Tensor,
        starts: Tensor,
        table: Tensor,
        workspace: Tensor,
        output: Tensor,
        cu: Tensor,
        useds: Tensor,
    ):
        op = THDecodeKernel(
            self.batch,
            self.total,
            self.ratio,
            self.slots,
            self.epsilon,
            self.plan,
            is_th=True,
        )
        op[self.block_count](
            x, packed, gamma, state, starts, table, workspace, output, cu, useds
        )


class THDecodeHost(_THDecodeLaunch):
    def __init__(
        self, batch, total, ratio, slots, epsilon, plan=DecodePlan(), *, device=None
    ):
        super().__init__(
            batch, total, ratio, slots, epsilon, plan, _effective_cube_cores(device)
        )


# ---- 8K prefill ----


@dataclass(frozen=True)
class PrefillPlan:
    m: int = 256
    d: int = 128
    kl1: int = 128
    ml0: int = 128
    nl0: int = 64
    kl0: int = 128
    mg: int = 8
    l1_slots: int = 2
    l0_slots: int = 2


@kernel
class PrefillKernel:
    def __init__(
        self,
        s,
        h,
        d,
        p,
        seq_len=8192,
        ratio=2,
        slots=8,
        epsilon=1e-20,
    ):
        self.s, self.h, self.d, self.p = s, h, d, p
        self.seq_len, self.ratio = seq_len, ratio
        self.slots, self.epsilon = slots, epsilon
        self.dn = d // p.d
        self.nm = p.m // p.ml0
        self.nn = 2 * p.d // p.nl0
        self.windows = p.ml0 // (2 * ratio)
        self.rounds = (s // p.m + p.mg - 1) // p.mg
        self.stat_pitch = FP32_VECTOR_LANES
        self.x_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((p.m, p.kl1)), depth=p.l1_slots
        )
        self.w_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((2 * p.d, p.kl1)), depth=p.l1_slots
        )
        self.a_l0 = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((p.ml0, p.kl0)), depth=p.l0_slots
        )
        self.b_l0 = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((p.nl0, p.kl0)), depth=4
        )
        self.grouped_acc = self.nm * self.nn > 24
        if self.grouped_acc:
            self.acc = tuple(
                Buffer(MemLoc.L0C, dtypes.float32, make_layout((p.ml0, 2 * p.d)))
                for _ in range(self.nm)
            )
        else:
            self.acc = tuple(
                Buffer(MemLoc.L0C, dtypes.float32, make_layout((p.ml0, p.nl0)))
                for _ in range(self.nm * self.nn)
            )
        self.kv = tuple(
            Buffer(MemLoc.UB, dtypes.float32, make_layout((p.ml0 // 2, p.d)))
            for _ in range(self.nm)
        )
        self.score = tuple(
            Buffer(MemLoc.UB, dtypes.float32, make_layout((p.ml0 // 2, p.d)))
            for _ in range(self.nm)
        )
        self.pool = tuple(
            Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((self.windows, p.d)))
            for _ in range(self.nm)
        )
        self.norm_out = tuple(
            Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((self.windows, p.d)))
            for _ in range(self.nm)
        )
        self.stats = tuple(
            Buffer(MemLoc.UB, dtypes.float32, make_layout((1, self.windows)))
            for _ in range(self.nm)
        )
        self.all_stats = tuple(
            Buffer(MemLoc.UB, dtypes.float32, make_layout((self.dn, self.stat_pitch)))
            for _ in range(self.nm)
        )
        self.gamma = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, p.d)))

    @jit
    def _load_panel(self, x, wkv, wgate, tile, dg, panel):
        p = self.p
        mem_copy(
            self.x_l1.next_write(),
            tile_slice(x, (p.m, p.kl1), (tile, panel)),
        )
        w_l1 = self.w_l1.next_write()
        mem_copy(
            tile_slice(w_l1, (p.d, p.kl1), (0, 0)),
            tile_slice(wkv, (p.d, p.kl1), (dg, panel)),
        )
        mem_copy(
            tile_slice(w_l1, (p.d, p.kl1), (1, 0)),
            tile_slice(wgate, (p.d, p.kl1), (dg, panel)),
        )

    @jit
    def _write_direct(self, mi, ni):
        p = self.p
        if const_expr(self.grouped_acc):
            acc = reinterpret(
                self.acc[mi],
                layout=make_layout((p.ml0, p.nl0)),
                offset=ni * p.ml0 * p.nl0 * 4,
                data_format="nz",
            )
        else:
            acc = self.acc[mi * self.nn + ni]
        if const_expr(ni * p.nl0 < p.d):
            mem_copy(
                tile_slice(self.kv[mi], (p.ml0 // 2, p.nl0), (0, ni)),
                acc.partition(dim=0),
            )
        else:
            mem_copy(
                tile_slice(self.score[mi], (p.ml0 // 2, p.nl0), (0, ni - p.d // p.nl0)),
                acc.partition(dim=0),
            )

    @jit
    def _pool_pair(self, mi):
        with vf(mode="simd"):
            mask = rr.full_mask()
            for window in range(self.windows):
                for col in range(self.p.d // 64):
                    offset0 = window * 2 * self.p.d + col * 64
                    offset1 = offset0 + self.p.d
                    s0 = rr.vload(self.score[mi], offset0)
                    s1 = rr.vload(self.score[mi], offset1)
                    maximum = rr.vmax(s0, s1, mask=mask)
                    w0 = rr.vexp(rr.vsub(s0, maximum, mask=mask), mask=mask)
                    w1 = rr.vexp(rr.vsub(s1, maximum, mask=mask), mask=mask)
                    zero = rr.vdups(0.0, dtypes.float32, mask=mask)
                    denom = rr.vadd(rr.vadd(zero, w0, mask=mask), w1, mask=mask)
                    a = rr.vmul(
                        rr.vdiv(w0, denom, mask=mask),
                        rr.vload(self.kv[mi], offset0),
                        mask=mask,
                    )
                    b = rr.vmul(
                        rr.vdiv(w1, denom, mask=mask),
                        rr.vload(self.kv[mi], offset1),
                        mask=mask,
                    )
                    result = rr.vcast(
                        rr.vadd(rr.vadd(zero, a, mask=mask), b, mask=mask),
                        dtypes.bfloat16,
                        mask=mask,
                    )
                    rr.vstore_pack(
                        self.pool[mi],
                        window * self.p.d + col * 64,
                        result,
                        mask,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )

    @jit
    def _pool(self, mi):
        if const_expr(self.ratio == 2):
            self._pool_pair(mi)
        else:
            with vf(mode="simd"):
                mask = rr.full_mask()
                for row in range(self.windows):
                    for col in range(self.p.d // 64):
                        offset = row * self.p.d + col * 64
                        value = rr.vcast(
                            rr.vload(self.kv[mi], offset), dtypes.bfloat16, mask=mask
                        )
                        rr.vstore_pack(
                            self.pool[mi],
                            offset,
                            value,
                            mask,
                            pack_mode=rr.PackMode.B32_TO_B16,
                        )

    @jit
    def _statistics(self, mi):
        with vf(mode="simd"):
            mask = rr.full_mask()
            for window in range(self.windows):
                total = rr.vdups(0.0, dtypes.float32, mask=mask)
                for col in range(self.p.d // 64):
                    raw = rr.vload_unpack(
                        self.pool[mi],
                        window * self.p.d + col * 64,
                        unpack_mode=rr.UnpackMode.B16_TO_B32,
                    )
                    value = rr.vcast(raw, dtypes.float32, mask=mask)
                    total = rr.vadd(total, rr.vmul(value, value, mask=mask), mask=mask)
                rr.vstore_first(
                    self.stats[mi], window, rr.vreduce_sum(total, mask=mask)
                )

    @jit
    def _normalize(self, mi):
        with vf(mode="simd"):
            mask = rr.full_mask()
            live_mask, _ = rr.update_mask(self.windows, elem_bits=32)
            total = rr.vdups(0.0, dtypes.float32, mask=live_mask)
            for group in range(self.dn):
                total = rr.vadd(
                    total,
                    rr.vload(self.all_stats[mi], group * self.stat_pitch),
                    mask=live_mask,
                )
            variance = rr.vadds(
                rr.vmuls(total, 1.0 / self.d, mask=live_mask),
                self.epsilon,
                mask=live_mask,
            )
            reciprocal = rr.vdiv(
                rr.vdups(1.0, dtypes.float32, mask=live_mask),
                rr.vsqrt(variance, mask=live_mask),
                mask=live_mask,
            )
            for col in range(self.p.d // FP32_VECTOR_LANES):
                gamma = rr.vcast(
                    rr.vload_unpack(
                        self.gamma,
                        col * FP32_VECTOR_LANES,
                        unpack_mode=rr.UnpackMode.B16_TO_B32,
                    ),
                    dtypes.float32,
                    mask=mask,
                )
                for window in range(self.windows):
                    offset = window * self.p.d + col * FP32_VECTOR_LANES
                    index = rr.vdups(window, dtypes.uint32, mask=mask)
                    window_reciprocal = rr.vgather_reg(reciprocal, index)
                    value = rr.vcast(
                        rr.vload_unpack(
                            self.pool[mi], offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                        ),
                        dtypes.float32,
                        mask=mask,
                    )
                    result = rr.vcast(
                        rr.vmul(
                            rr.vmul(value, window_reciprocal, mask=mask),
                            gamma,
                            mask=mask,
                        ),
                        dtypes.bfloat16,
                        mask=mask,
                    )
                    rr.vstore_pack(
                        self.norm_out[mi],
                        offset,
                        result,
                        mask,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )

    @jit
    def _project_panel(self, x, wkv, wgate, tile, dg, panel, first, last):
        p = self.p
        if const_expr(last):
            if tile + p.mg < self.s // p.m:
                self._load_panel(x, wkv, wgate, tile + p.mg, dg, 0)
        else:
            self._load_panel(x, wkv, wgate, tile, dg, panel + 1)
        x_l1 = self.x_l1.next_read()
        w_l1 = self.w_l1.next_read()
        for mi in range_constexpr(self.nm):
            for ki in range_constexpr(p.kl1 // p.kl0):
                mem_copy(
                    self.a_l0.next_write(), tile_slice(x_l1, (p.ml0, p.kl0), (mi, ki))
                )
                a = self.a_l0.next_read()
                for ni in range_constexpr(self.nn):
                    mem_copy(
                        self.b_l0.next_write(),
                        tile_slice(w_l1, (p.nl0, p.kl0), (ni, ki)),
                    )
                    b = self.b_l0.next_read()
                    if const_expr(self.grouped_acc):
                        acc = reinterpret(
                            self.acc[mi],
                            layout=make_layout((p.ml0, p.nl0)),
                            offset=ni * p.ml0 * p.nl0 * 4,
                            data_format="nz",
                        )
                    else:
                        acc = self.acc[mi * self.nn + ni]
                    matmul(acc, a, b, init=(first and ki == 0))
            if const_expr(last):
                for ni in range_constexpr(self.nn):
                    self._write_direct(mi, ni)

    def __call__(
        self,
        x: Tensor,
        wkv: Tensor,
        wgate: Tensor,
        gamma: Tensor,
        state: Tensor,
        output: Tensor,
        stats_gm: Tensor,
        table: Tensor,
        starts: Tensor,
    ):
        p = self.p
        core = get_block_idx()
        sub = get_subblock_id()
        dg = core % self.dn
        mg = core // self.dn
        mem_copy(self.gamma, tile_slice(gamma, (1, p.d), (0, dg)))
        for epoch in range(self.rounds):
            tile = epoch * p.mg + mg
            if tile < self.s // p.m:
                if epoch == 0:
                    self._load_panel(x, wkv, wgate, tile, dg, 0)
                if const_expr(self.h // p.kl1 == 1):
                    self._project_panel(x, wkv, wgate, tile, dg, 0, True, True)
                else:
                    self._project_panel(x, wkv, wgate, tile, dg, 0, True, False)
                    for panel in range(1, self.h // p.kl1 - 1):
                        self._project_panel(
                            x, wkv, wgate, tile, dg, panel, False, False
                        )
                    self._project_panel(
                        x,
                        wkv,
                        wgate,
                        tile,
                        dg,
                        self.h // p.kl1 - 1,
                        False,
                        True,
                    )
            for mi in range_constexpr(self.nm):
                if tile < self.s // p.m:
                    self._pool(mi)
                    self._statistics(mi)
                    window_tile = tile * 2 * self.nm + mi * 2 + sub
                    mem_copy(
                        tile_slice(stats_gm, (1, self.windows), (dg, window_tile)),
                        self.stats[mi],
                    )
                    vector_base = tile * p.m + mi * p.ml0 + sub * (p.ml0 // 2)
                    owner = vector_base // self.seq_len
                    token_base = vector_base % self.seq_len
                    if token_base + p.ml0 // 2 > self.seq_len - self.slots:
                        for row in range(
                            max(0, self.seq_len - self.slots - token_base), p.ml0 // 2
                        ):
                            slot = (starts[owner] + token_base + row) % self.slots
                            target = table[owner] * self.slots + slot
                            mem_copy(
                                tile_slice(state, (1, p.d), (target, dg)),
                                tile_slice(self.kv[mi], (1, p.d), (row, 0)),
                            )
                            mem_copy(
                                tile_slice(state, (1, p.d), (target, dg + self.dn)),
                                tile_slice(self.score[mi], (1, p.d), (row, 0)),
                            )
                vec_sync_all()
                vec_sync_block_arrive(PIPE.MTE3, 5, mode=0)
                vec_sync_block_wait(PIPE.S, 5, mode=0)
                if tile < self.s // p.m:
                    window_tile = tile * 2 * self.nm + mi * 2 + sub
                    for partition in range_constexpr(self.dn):
                        mem_copy(
                            tile_slice(
                                self.all_stats[mi], (1, self.windows), (partition, 0)
                            ),
                            tile_slice(
                                stats_gm, (1, self.windows), (partition, window_tile)
                            ),
                        )
                    self._normalize(mi)
                    mem_copy(
                        tile_slice(output, (self.windows, p.d), (window_tile, dg)),
                        self.norm_out[mi],
                    )
            if epoch + 1 == self.rounds:
                cube_sync_all()
                vec_sync_all()


class _PrefillLaunch:
    def __init__(self, batch, seq_len, ratio, slots, epsilon, plan, block_count):
        self.batch, self.seq_len = batch, seq_len
        self.ratio, self.slots = ratio, slots
        self.epsilon, self.plan = epsilon, plan
        required_blocks = plan.mg * (512 // plan.d)
        self.block_count = min(block_count, required_blocks)
        if self.block_count < required_blocks:
            raise ValueError(f"prefill requires {required_blocks} available Mix blocks")

    @host
    def run(
        self,
        x: Tensor,
        wkv: Tensor,
        wgate: Tensor,
        gamma: Tensor,
        state: Tensor,
        output: Tensor,
        stats: Tensor,
        table: Tensor,
        starts: Tensor,
    ):
        op = PrefillKernel(
            self.batch * self.seq_len,
            5120,
            512,
            self.plan,
            seq_len=self.seq_len,
            ratio=self.ratio,
            slots=self.slots,
            epsilon=self.epsilon,
        )
        op[self.block_count](x, wkv, wgate, gamma, state, output, stats, table, starts)


class PrefillHost(_PrefillLaunch):
    def __init__(
        self, batch, seq_len, ratio, slots, epsilon, plan=PrefillPlan(), *, device=None
    ):
        super().__init__(
            batch, seq_len, ratio, slots, epsilon, plan, _effective_cube_cores(device)
        )


# ---- Host launch and Torch interface ----

_PACKED = OrderedDict()


def cached_weights(wkv, wgate, plan):
    # Version checks invalidate a pack after an in-place optimizer/copy update.
    try:
        versions = (wkv._version, wgate._version)
    except RuntimeError:
        # Inference tensors have no version counter; never reuse a stale pack.
        return pack_weights(wkv, wgate, plan)
    key = (id(wkv), id(wgate), *versions, plan)
    entry = _PACKED.get(key)
    if entry is not None and entry[0]() is wkv and entry[1]() is wgate:
        _PACKED.move_to_end(key)
        return entry[2]
    packed = pack_weights(wkv, wgate, plan)
    _PACKED[key] = (weakref.ref(wkv), weakref.ref(wgate), packed)
    while len(_PACKED) > 8:
        _PACKED.popitem(last=False)
    return packed


@dataclass
class PreparedMegaCompressor:
    program: object
    args: tuple
    output: torch.Tensor
    backend: str

    def __call__(self):
        self.program(*self.args)
        return self.output

    def close(self):
        torch.npu.synchronize()
        self.program.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def prepare_mega_compressor(
    x,
    wkv,
    wgate,
    state_cache,
    norm_weight,
    *,
    cmp_ratio=2,
    state_block_table,
    start_pos=None,
    norm_eps=1e-20,
    packed_weights=None,
    decode_plan=DecodePlan(),
):
    """Compile/bind once. Mutable X/state/starts are read on every invocation.

    Buffer geometry, block mapping and weights remain fixed. Re-prepare after a
    weight update. Prefill requires full, ratio-aligned sequences; decode starts
    may vary. Static weight packing is outside repeated execution.
    """
    if x.ndim != 3 or x.shape[-1] != 5120 or x.shape[1] not in (1, 6, 8, 8192):
        raise ValueError("prepared mega_compressor requires BSH, H5120, S1/S6/S8/S8192")
    batch, seq, _ = x.shape
    if batch not in (1, 4, 8, 16, 32) or cmp_ratio not in (1, 2):
        raise ValueError("prepared mega_compressor requires B1/4/8/16/32, R1/2")
    if x.device.type not in ("npu", "privateuseone"):
        raise ValueError("prepared mega_compressor requires NPU tensors")
    for name, tensor, shape in [
        ("x", x, x.shape),
        ("wkv", wkv, (512, 5120)),
        ("wgate", wgate, (512, 5120)),
        ("norm_weight", norm_weight, (512,)),
    ]:
        if (
            tensor.shape != shape
            or tensor.dtype != torch.bfloat16
            or tensor.device != x.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} requires contiguous BF16 {tuple(shape)} on the input device"
            )
    if (
        state_cache.ndim != 3
        or state_cache.shape[-1] != 1024
        or not 1 <= state_cache.shape[1] <= 1024
        or state_cache.dtype != torch.float32
        or state_cache.device != x.device
        or not state_cache.is_contiguous()
    ):
        raise ValueError(
            "state_cache requires contiguous FP32 [blocks,1..1024,1024] on input device"
        )
    if not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError("norm_eps must be finite and positive")
    slots = state_cache.shape[1]
    for name, tensor in (
        ("state_block_table", state_block_table),
        ("start_pos", start_pos),
    ):
        if tensor is None and name == "start_pos":
            continue
        if tensor.shape != (batch,) or tensor.dtype != torch.int32:
            raise ValueError(f"{name} requires INT32 [B]")
    blocks = _metadata_values(state_block_table, ())
    positions = _metadata_values(start_pos, [0] * batch)
    if (
        len(set(blocks)) != batch
        or min(blocks) < 0
        or max(blocks) >= state_cache.shape[0]
    ):
        raise ValueError("state_block_table requires distinct allocated blocks")
    if min(positions) < 0:
        raise ValueError("start_pos must be nonnegative")
    if seq == 8192 and any(p % cmp_ratio for p in positions):
        raise ValueError("prepared prefill requires ratio-aligned start_pos")
    table = _device_metadata(state_block_table, blocks, x.device)
    starts = _device_metadata(start_pos, positions, x.device)
    output = torch.empty(
        (batch, (seq + cmp_ratio - 1) // cmp_ratio, 512), dtype=x.dtype, device=x.device
    )
    xf, sf, gamma, out = (
        x.view(-1, 5120),
        state_cache.view(-1, 1024),
        norm_weight.view(1, 512),
        output.view(-1, 512),
    )
    if seq == 8192:
        entry = PrefillHost(
            batch, seq, cmp_ratio, slots, float(norm_eps), device=x.device
        )
        stats = torch.empty(
            (4, batch * seq // cmp_ratio), device=x.device, dtype=torch.float32
        )
        args = (xf, wkv, wgate, gamma, sf, out, stats, table, starts)
        backend = "pr221_prefill"
    else:
        entry = DecodeHost(
            batch, seq, cmp_ratio, slots, float(norm_eps), decode_plan, device=x.device
        )
        packed = (
            cached_weights(wkv, wgate, decode_plan)
            if packed_weights is None
            else packed_weights
        )
        expected = (
            decode_plan.kg
            * (512 // decode_plan.dl)
            * (5120 // decode_plan.kg // decode_plan.k)
            * (2 * decode_plan.dl // decode_plan.n),
            decode_plan.n * decode_plan.k,
        )
        if (
            packed.shape != expected
            or packed.dtype != x.dtype
            or packed.device != x.device
            or not packed.is_contiguous()
        ):
            raise ValueError(
                "packed_weights does not match DecodePlan and input dtype/device"
            )
        workspace = torch.empty(
            decode_plan.workspace_shape(batch * seq),
            device=x.device,
            dtype=torch.float32,
        )
        args = (xf, packed, gamma, sf, starts, table, workspace, out)
        backend = "pr221_decode"
    program = compile_args(entry.run, args)
    return PreparedMegaCompressor(program, args, output, backend)


def compile_args(fn, args):
    types = {
        torch.bfloat16: cb.dtypes.bfloat16,
        torch.float16: cb.dtypes.float16,
        torch.float32: cb.dtypes.float32,
        torch.int32: cb.dtypes.int32,
    }
    return cb.compile(
        fn,
        *(
            cb.TensorSpec(tuple(t.shape), types[t.dtype], stride=tuple(t.stride()))
            for t in args
        ),
    )


@dataclass(frozen=True)
class _CallMetadata:
    batch: int
    sequence_length: int
    is_th: bool
    lengths: list
    used: list
    starts: list
    blocks: list
    cumulative: Optional[list]


def _validate_inputs(
    x,
    wkv,
    wgate,
    state_cache,
    cmp_ratio,
    state_block_table,
    cu_seqlens,
    seqused,
    start_pos,
    norm_weight,
    norm_eps,
    backend,
):
    if x.device.type not in ("npu", "privateuseone"):
        raise ValueError("mega_compressor requires an NPU tensor")
    if x.ndim not in (2, 3) or wkv.ndim != 2 or wgate.shape != wkv.shape:
        raise ValueError("expected x [B,S,H]/[T,H] and matching [D,H] weights")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("x must be float16 or bfloat16")
    if wkv.device != x.device or wgate.device != x.device:
        raise ValueError("x, wkv and wgate must share a device")
    if wkv.dtype != x.dtype or wgate.dtype != x.dtype:
        raise ValueError("wkv and wgate must use x dtype")
    hidden_size = int(x.shape[-1])
    if (
        hidden_size != int(wkv.shape[1])
        or hidden_size < _MIN_HIDDEN
        or hidden_size > _MAX_HIDDEN
        or hidden_size % _HIDDEN_ALIGN
    ):
        raise ValueError(
            "hidden size must match weights and be 1024..10240, aligned to 512"
        )
    if int(wkv.shape[0]) != 512:
        raise ValueError("mega_compressor requires head_dim=512")
    if int(cmp_ratio) not in (1, 2):
        raise ValueError("mega_compressor requires cmp_ratio=1 or 2")
    if hidden_size != 5120:
        raise ValueError("mega_compressor requires hidden_size=5120")
    if backend not in ("auto", "legacy", "pr221"):
        raise ValueError("backend must be auto, legacy or pr221")
    if norm_weight is not None:
        if (
            norm_weight.shape != (512,)
            or norm_weight.device != x.device
            or norm_weight.dtype not in (x.dtype, torch.float32)
            or not norm_weight.is_contiguous()
        ):
            raise ValueError(
                "norm_weight requires contiguous [512], input dtype/FP32, on x device"
            )
        if not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")
    x_is_th = x.ndim == 2
    logical_seqlen = int(x.shape[0]) if x_is_th else int(x.shape[1])
    if x_is_th and cu_seqlens is None:
        raise ValueError("TH input requires cu_seqlens")
    if not x_is_th and cu_seqlens is not None:
        raise ValueError("BSH input does not accept cu_seqlens")
    if state_block_table is None:
        raise ValueError("AscendC mega_compressor requires state_block_table")
    batch = int(cu_seqlens.numel() - 1) if x_is_th else int(x.shape[0])
    if batch not in (1, 4, 8, 16, 32):
        raise ValueError("mega_compressor requires batch 1/4/8/16/32")
    if not x_is_th and int(x.shape[1]) not in (1, 6, 8, 8192):
        raise ValueError("mega_compressor requires S=1, 6, 8, or 8192")
    if state_cache.ndim != 3 or state_cache.dtype != torch.float32:
        raise ValueError("state_cache must be FP32 [blocks, block_size, 2D]")
    if state_cache.device != x.device or int(state_cache.shape[2]) != 2 * int(
        wkv.shape[0]
    ):
        raise ValueError("state_cache must be on x device with width 2D")
    if (
        int(state_cache.shape[0]) < batch
        or int(state_cache.shape[1]) < _MIN_BLOCK_SIZE
        or int(state_cache.shape[1]) > _MAX_BLOCK_SIZE
    ):
        raise ValueError("state_cache must have B blocks and 1..1024 ring slots")
    if (
        state_block_table.ndim != 1
        or int(state_block_table.numel()) != batch
        or state_block_table.dtype != torch.int32
    ):
        raise ValueError("state_block_table must be int32 with B entries")
    if x_is_th:
        if (
            cu_seqlens.ndim != 1
            or int(cu_seqlens.numel()) != batch + 1
            or cu_seqlens.dtype != torch.int32
        ):
            raise ValueError("cu_seqlens requires INT32 [B+1]")
    for name, tensor in (("seqused", seqused), ("start_pos", start_pos)):
        if tensor is not None and (
            tensor.ndim != 1 or tensor.numel() != batch or tensor.dtype != torch.int32
        ):
            raise ValueError(f"{name} requires INT32 [B]")
    cu_host = _metadata_values(cu_seqlens, ()) if x_is_th else None
    if x_is_th:
        if (
            cu_host[0] != 0
            or cu_host[-1] != x.shape[0]
            or any(b < a for a, b in zip(cu_host, cu_host[1:]))
        ):
            raise ValueError(
                "cu_seqlens must be a nondecreasing prefix sum ending at T"
            )
        lengths = [b - a for a, b in zip(cu_host, cu_host[1:])]
    else:
        lengths = [logical_seqlen] * batch
    used_host = _metadata_values(seqused, lengths)
    start_host = _metadata_values(start_pos, [0] * batch)
    table_host = _metadata_values(state_block_table, ())
    if any(u < 0 or u > length for u, length in zip(used_host, lengths)):
        raise ValueError("seqused must satisfy 0 <= seqused[b] <= sequence length")
    if any(position < 0 for position in start_host):
        raise ValueError("start_pos must be nonnegative")
    if any(block < 0 or block >= state_cache.shape[0] for block in table_host):
        raise ValueError("state_block_table contains an invalid cache block")
    return _CallMetadata(
        batch,
        logical_seqlen,
        x_is_th,
        lengths,
        used_host,
        start_host,
        table_host,
        cu_host,
    )


def _specialized_input(
    x, wkv, wgate, state_cache, norm_weight, cmp_ratio, backend, metadata
):
    return (
        norm_weight is not None
        and backend != "legacy"
        and not metadata.is_th
        and x.dtype == torch.bfloat16
        and norm_weight.dtype == torch.bfloat16
        and (metadata.batch in (1, 4) or backend == "pr221")
        and metadata.used == metadata.lengths
        and len(set(metadata.blocks)) == metadata.batch
        and all(tensor.is_contiguous() for tensor in (x, wkv, wgate, state_cache))
        and (
            metadata.sequence_length != 8192
            or (
                all(position % cmp_ratio == 0 for position in metadata.starts)
                and _effective_cube_cores(x.device)
                >= PrefillPlan().mg * (512 // PrefillPlan().d)
            )
        )
    )


def _splitk_tiling(total_rows, hidden):
    padded_rows = max(
        _CUBE_ROW_ALIGNMENT,
        (total_rows + _CUBE_ROW_ALIGNMENT - 1)
        // _CUBE_ROW_ALIGNMENT
        * _CUBE_ROW_ALIGNMENT,
    )
    k_shard = hidden // _SPLITK_KG
    candidates = [
        width
        for width in (1280, 640, 256, 128)
        if width <= k_shard and k_shard % width == 0
    ]
    l1_k = next(
        (
            width
            for width in candidates
            if (padded_rows + 2 * _SPLITK_DL) * width * 2 <= 512 * 1024
        ),
        candidates[-1],
    )
    l0_k = 128 if padded_rows * 128 * 2 * 2 <= 65536 else 64
    slots = 2 if padded_rows * l0_k * 2 * 2 <= 65536 else 1
    return l1_k, l0_k, slots


def _run_general(
    x,
    wkv,
    wgate,
    state_cache,
    ratio,
    table,
    starts,
    used,
    cu,
    norm_weight,
    norm_eps,
    metadata,
):
    total_rows = x.numel() // x.shape[-1]
    output_stride = (metadata.sequence_length + ratio - 1) // ratio
    if metadata.is_th:
        output_stride = min(total_rows, total_rows // ratio + metadata.batch)
        output_shape = (output_stride, 512)
    else:
        output_shape = (metadata.batch, output_stride, 512)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    if total_rows == 0:
        return output
    rows = _device_contiguous(x).view(total_rows, 5120)
    flat_output = output.view(-1, 512)
    core_count = _effective_cube_cores(x.device)
    slots = state_cache.shape[1]
    common = (
        metadata.batch,
        512,
        5120,
        ratio,
        metadata.is_th,
        metadata.sequence_length,
        output_stride,
        slots,
        total_rows,
    )
    if total_rows <= _DECODE_MAX_ROWS and total_rows != metadata.batch:
        l1_k, l0_k, l0_slots = _splitk_tiling(total_rows, 5120)
        entry = _SplitKDecode(
            *common,
            0,
            min(core_count, _SPLITK_KG * (512 // _SPLITK_DL)),
            l1_k,
            l0_slots,
            l0_k,
            norm_weight is not None,
            float(norm_eps),
        )
        workspace = torch.empty(
            (_SPLITK_KG * total_rows, 1024), dtype=torch.float32, device=x.device
        )
        args = (
            flat_output,
            rows,
            wkv,
            wgate,
            state_cache,
            table,
            cu,
            used,
            starts,
            workspace,
        )
    else:
        entry = _FusedStream(
            *common, 4, core_count, norm_weight is not None, float(norm_eps)
        )
        tail = torch.empty((2 * _STREAM_M_TILE, 5120), dtype=x.dtype, device=x.device)
        cache_workspace = torch.empty(
            (metadata.batch, slots, 1024), dtype=torch.float32, device=x.device
        )
        args = (
            flat_output,
            rows,
            tail,
            wkv,
            wgate,
            state_cache,
            table,
            cu,
            used,
            starts,
            cache_workspace,
        )
    if norm_weight is None:
        entry.run(*args)
    else:
        entry.run_fused(*args, norm_weight.view(1, -1))
    return output


def mega_compressor(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    cmp_ratio: int = 2,
    *,
    state_block_table: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-20,
    backend: str = "auto",
) -> torch.Tensor:
    """Run mega_compressor on NPU and return compressed KV.

    Optional metadata is normalized to small device tensors so one kernel ABI
    remains stable; ``cu_seqlens`` is required for TH and ``state_block_table``
    is required by the ring-cache template. Passing ``norm_weight`` fuses
    full-head RMSNorm after the pooling dtype conversion. Without it the
    original unnormalized output contract is preserved.
    """
    metadata = _validate_inputs(
        x,
        wkv,
        wgate,
        state_cache,
        cmp_ratio,
        state_block_table,
        cu_seqlens,
        seqused,
        start_pos,
        norm_weight,
        norm_eps,
        backend,
    )
    table = _device_metadata(state_block_table, metadata.blocks, x.device)
    starts = _device_metadata(start_pos, metadata.starts, x.device)
    if _specialized_input(
        x, wkv, wgate, state_cache, norm_weight, cmp_ratio, backend, metadata
    ):
        with prepare_mega_compressor(
            x,
            wkv,
            wgate,
            state_cache,
            norm_weight,
            cmp_ratio=cmp_ratio,
            state_block_table=table,
            start_pos=starts,
            norm_eps=norm_eps,
        ) as prepared:
            return prepared()
    if norm_weight is not None and backend == "pr221":
        raise ValueError(
            "shape/metadata is outside the specialization; use backend='legacy'"
        )
    used = _device_metadata(seqused, metadata.used, x.device)
    cu = (
        _device_metadata(cu_seqlens, metadata.cumulative, x.device)
        if metadata.is_th
        else torch.empty((1,), dtype=torch.int32, device=x.device)
    )
    return _run_general(
        x,
        wkv,
        wgate,
        state_cache,
        int(cmp_ratio),
        table,
        starts,
        used,
        cu,
        norm_weight,
        norm_eps,
        metadata,
    )


# ---- Native publication: compile the same launches without a device query ----


@dataclass(frozen=True)
class _NativeProfile:
    batch: int
    seq_len: int
    layout: str = "BSH"
    ratio: int = 2
    slots: int = 8
    cache_blocks: Optional[int] = None
    norm_eps: float = 1e-6
    cube_cores: int = 32

    def __post_init__(self):
        dimensions = (self.batch, self.seq_len, self.ratio, self.slots, self.cube_cores)
        if any(type(value) is not int for value in dimensions):
            raise ValueError("native profile dimensions must be integers")
        if self.layout not in ("BSH", "TH") or self.batch not in (1, 4, 8, 16, 32):
            raise ValueError("native profiles require BSH/TH and B1/4/8/16/32")
        lengths = (1, 6, 8, 8192) if self.layout == "BSH" else (1, 6, 8)
        if self.seq_len not in lengths or self.ratio not in (1, 2):
            raise ValueError("native profile sequence length or ratio is unsupported")
        if not 1 <= self.slots <= 1024 or not 1 <= self.cube_cores <= 32:
            raise ValueError(
                "native profiles require 1..1024 slots and 1..32 Cube cores"
            )
        if self.cache_blocks is not None and (
            type(self.cache_blocks) is not int or self.cache_blocks < self.batch
        ):
            raise ValueError("native cache_blocks must be an integer at least batch")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("native norm_eps must be finite and positive")
        required = PrefillPlan().mg * (512 // PrefillPlan().d)
        if self.seq_len == 8192 and self.cube_cores < required:
            raise ValueError(f"native prefill requires {required} Mix blocks")


def _native_profiles():
    import json
    import os

    profiles = [
        _NativeProfile(batch, seq, layout=layout, cube_cores=cores)
        for batch in (1, 4)
        for seq in (1, 6, 8)
        for layout in ("BSH", "TH")
        for cores in (28, 32)
    ] + [_NativeProfile(batch, 8192) for batch in (1, 4)]
    path = os.environ.get("CANNBOTDSL_DS41_PROFILES")
    if path:
        with open(path, encoding="utf-8") as stream:
            entries = json.load(stream).get("mega_compressor")
        if entries is not None:
            if not isinstance(entries, list) or not entries:
                raise ValueError(
                    "mega_compressor native profiles must be a nonempty list"
                )
            profiles = [_NativeProfile(**entry) for entry in entries]
    return list(dict.fromkeys(profiles))


def _native_launch(profile):
    spec = cb.TensorSpec
    bf16, fp32, i32 = dtypes.bfloat16, dtypes.float32, dtypes.int32
    batch, seq, ratio = profile.batch, profile.seq_len, profile.ratio
    total = batch * seq
    cache_rows = (profile.cache_blocks or batch) * profile.slots
    x = spec((total, 5120), bf16)
    gamma = spec((1, 512), bf16)
    state = spec((cache_rows, 1024), fp32)
    metadata = spec((batch,), i32)
    geometry = (batch, seq, ratio, profile.slots, float(profile.norm_eps))
    if seq == 8192:
        entry = _PrefillLaunch(*geometry, PrefillPlan(), profile.cube_cores)
        weight = spec((512, 5120), bf16)
        output = spec((total // ratio, 512), bf16)
        stats = spec((4, total // ratio), fp32)
        return entry.run, (
            x,
            weight,
            weight,
            gamma,
            state,
            output,
            stats,
            metadata,
            metadata,
        )
    plan = DecodePlan()
    packed = _PackWeightsLaunch(plan, profile.cube_cores)
    weight = spec((packed.tasks, plan.n * plan.k), bf16)
    workspace = spec(plan.workspace_shape(total), fp32)
    if profile.layout == "BSH":
        entry = _DecodeLaunch(*geometry, plan, profile.cube_cores)
        output = spec((batch * ((seq + ratio - 1) // ratio), 512), bf16)
        return entry.run, (
            x,
            weight,
            gamma,
            state,
            metadata,
            metadata,
            workspace,
            output,
        )
    entry = _THDecodeLaunch(
        batch,
        total,
        ratio,
        profile.slots,
        float(profile.norm_eps),
        plan,
        profile.cube_cores,
    )
    output = spec((min(total, total // ratio + batch), 512), bf16)
    return entry.run, (
        x,
        weight,
        gamma,
        state,
        metadata,
        metadata,
        workspace,
        output,
        spec((batch + 1,), i32),
        metadata,
    )


@export("mega_compressor")
def export_mega_compressor():
    """Publish BF16 BSH/TH decode, prefill and their weight-packing programs.

    CANNBOTDSL_DS41_PROFILES may replace the default catalogue with a
    ``mega_compressor`` array of _NativeProfile fields. Tensor shapes, cache
    capacity, epsilon and stream quota must match a published profile.
    Runtime Hosts still query the actual stream; only publication supplies
    the compile-time profile capacity directly to the shared launch classes.
    """
    profiles = _native_profiles()
    plan = DecodePlan()
    for cores in sorted({p.cube_cores for p in profiles if p.seq_len != 8192}):
        entry = _PackWeightsLaunch(plan, cores)
        weight = cb.TensorSpec((512, 5120), dtypes.bfloat16)
        packed = cb.TensorSpec((entry.tasks, plan.n * plan.k), dtypes.bfloat16)
        program = cb.compile(entry.run, weight, weight, packed)
        program.close()
    for profile in profiles:
        function, specs = _native_launch(profile)
        program = cb.compile(function, *specs)
        program.close()


__all__ = ["mega_compressor"]
