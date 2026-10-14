# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FlashKDA CP main public operator."""

import dataclasses
import os
import threading
from collections import OrderedDict
from functools import lru_cache
from typing import Optional

import cannbotdsl
import torch
from cannbotdsl import PIPE, ChannelKind, MemLoc, Tensor, dtypes, get_platform_info
from cannbotdsl.buffer import Buffer
from cannbotdsl.channel import Channel
from cannbotdsl.lang import vf
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops import reg as reg_ops
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    update_mask,
    vadd,
    vcast,
    vdups,
    vload,
    vload_unpack,
    vmem_bar,
    vmul,
    vstore,
    vstore_pack,
)
from cannbotdsl.ops.sync import (
    channel_rewind,
    cube_sync_all,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
)
from cannbotdsl.tensor import idx2crd, reinterpret, tile_slice

try:
    from cannbotdsl import aot
except ImportError:  # pragma: no cover - compatibility with pre-0.7 hosts
    aot = None

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

compile_cache = getattr(aot, "compile_cache", lru_cache(maxsize=1))


@jit
def cast_tile(dst, src):
    """Convert a 2D f32/f16/bf16 tile while preserving physical row strides."""
    rows, cols = dst.shape
    src_stride = src.physical_stride[0]
    dst_stride = dst.physical_stride[0]
    with vf(mode="simd"):
        for row in range(rows):
            if const_expr(isinstance(cols, int) and cols == 1):
                value = reg_ops.vload_broadcast(src, row * src_stride)
                converted = reg_ops.vcast(value, dst.dtype, mask=reg_ops.full_mask())
                reg_ops.vstore_first(dst, row * dst_stride, converted)
            else:
                for col in range(0, cols, 64):
                    mask, _ = reg_ops.update_mask(cols - col, elem_bits=32)
                    if const_expr(src.dtype == dtypes.float32):
                        value = reg_ops.vload(src, row * src_stride + col)
                    else:
                        value = reg_ops.vload_unpack(
                            src, row * src_stride + col, unpack_mode=reg_ops.UnpackMode.B16_TO_B32
                        )
                    converted = reg_ops.vcast(value, dst.dtype, mask=mask)
                    if const_expr(dst.dtype == dtypes.float32):
                        reg_ops.vstore(dst, row * dst_stride + col, converted, mask)
                    else:
                        reg_ops.vstore_pack(
                            dst, row * dst_stride + col, converted, mask, pack_mode=reg_ops.PackMode.B32_TO_B16
                        )


@jit
def nd2nz_tile(dst, src, elem_bits: int = 16):
    """Move a row-major tile into the NZ layout with raw vector stores."""
    rows, cols = src.shape
    src_stride = src.physical_stride[0]
    dst_stride = dst.physical_stride[0]
    n0 = 256 // elem_bits
    block_stride = dst_stride * elem_bits // 256
    with vf(mode="simd"):
        for row in range(rows):
            remaining = reg_ops.mask_counter(cols)
            for col in range(0, cols, 2048 // elem_bits):
                mask, remaining = reg_ops.update_mask(remaining, elem_bits=elem_bits)
                values = reg_ops.vload(src, row * src_stride + col)
                offset = row * n0 + (col // n0) * dst_stride
                reg_ops.vstore_strided(dst, offset, values, mask, block_stride=block_stride, repeat_stride=0)


CHUNK_SIZE = 64
SUPPORTED_HEAD_DIM = 128
LOW_DTYPE = torch.bfloat16
HIGH_DTYPE = torch.float32
D_HALF = SUPPORTED_HEAD_DIM // 2

# Ascend 9.x raw cache-control encodings used by the shared CP copy schedule.
_L2_CACHE_LAST_USE = 1
_L2_CACHE_DISABLE = 4


def _chunk_count(seq_len):
    return (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE


def _chunk_capacity(seq_len, batch):
    """Uniform intermediate capacity shared by all CP layouts."""
    return _chunk_count(seq_len) + batch - 1


@dataclasses.dataclass
class _ChunkPosition:
    storage_batch: object
    logical_batch: object
    head: object
    dv_index: object
    subblock: object


def _o_head_chunk_tile(gm_output, position, chunk_idx, dv_base, valid_rows=CHUNK_SIZE):
    """Return one logical BNSD output chunk while preserving physical strides."""
    full_chunk = tile_slice(
        gm_output[position.storage_batch, position.head, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk_idx, 0)
    )
    return _valid_rows_view(tile_slice(full_chunk, (CHUNK_SIZE, dv_base), (0, position.dv_index)), valid_rows, dv_base)


def _valid_rows_view(full_tile, valid_rows, cols):
    return full_tile[:valid_rows, :cols]


def _sequence_head_chunk_tile(gm_tensor, batch_idx, head_idx, chunk_idx):
    # The parent sequence view carries its exact runtime row extent.  Keep the
    # tile shape intact so MTE2 applies the shared CP tail handling.
    return tile_slice(gm_tensor[batch_idx, head_idx, None, None], (CHUNK_SIZE, gm_tensor.shape[3]), (chunk_idx, 0))


def _chunk_matrix_tile(gm_tensor, batch_idx, head_idx, chunk_idx):
    return tile_slice(gm_tensor[batch_idx, head_idx, chunk_idx, None, None], (CHUNK_SIZE, CHUNK_SIZE), (0, 0))


@dataclasses.dataclass
class _CPExecutionEntry:
    launch_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


class _CPExecutionCache:
    """Keep one launch lock per NPU stream and launch grid."""

    def __init__(self, max_entries: int = 16):
        self._max_entries = max_entries
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _stream_identity(ref: torch.Tensor):
        device = ref.device
        if device.type not in {"npu", "privateuseone"}:
            return device.type, device.index, None
        device_index = ref.get_device() if hasattr(ref, "get_device") else device.index
        stream = torch.npu.current_stream(device_index)
        stream_id = getattr(stream, "npu_stream", stream)
        return device.type, device_index, int(stream_id)

    def get_or_create(self, *, ref: torch.Tensor, block_num: int) -> _CPExecutionEntry:
        key = (*self._stream_identity(ref), block_num)
        with self._lock:
            cached = self._entries.pop(key, None)
            if cached is None:
                cached = _CPExecutionEntry()
            self._entries[key] = cached
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            return cached

    def clear(self):
        with self._lock:
            self._entries.clear()


_CP_EXECUTION_CACHE = _CPExecutionCache()


def clear_execution_cache():
    _CP_EXECUTION_CACHE.clear()


VL = CHUNK_SIZE  # A 256-byte register holds 64 f32 lanes.


class Matmul:
    def __init__(self, *, dv_base: int):
        self.dv_base = dv_base
        self.l0a_chunk = Channel(MemLoc.L0A, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=2)
        self.l0a_kr_t = Channel(
            MemLoc.L0A, shape=(CHUNK_SIZE, dv_base), dtype=dtypes.bfloat16, depth=1, data_format="zn"
        ).produce()
        self.l0a_mqk = Channel(MemLoc.L0A, shape=(CHUNK_SIZE, CHUNK_SIZE), dtype=dtypes.bfloat16, depth=1).produce()
        self.l0b_u = Channel(MemLoc.L0B, shape=(CHUNK_SIZE, dv_base), dtype=dtypes.bfloat16, depth=1).produce()
        self.l0b_state = Channel(
            MemLoc.L0B, shape=(dv_base, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=1
        ).produce()
        self.l0b_kr = Channel(
            MemLoc.L0B, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=1
        ).produce()
        self.l0c_u = Channel(MemLoc.L0C, shape=(CHUNK_SIZE, dv_base), dtype=dtypes.float32, depth=1).produce()
        self.l0c_o = Channel(MemLoc.L0C, shape=(CHUNK_SIZE, dv_base), dtype=dtypes.float32, depth=1).produce()
        self.l0c_state = Channel(
            MemLoc.L0C, shape=(dv_base, SUPPORTED_HEAD_DIM), dtype=dtypes.float32, depth=1
        ).produce()
        self.gm2l1 = make_copy_engine(format_transform="nd2nz")
        self.fixpipe_state = make_copy_engine(split_axis=1)
        self.fixpipe_u = make_copy_engine(split_axis=0)

    def load_q_decayed(self, q_decayed_l1, gm_tile, valid_rows):
        self._load_padded_chunk(q_decayed_l1, gm_tile, valid_rows)

    def load_state(self, state_l1):
        mem_copy(self.l0b_state, state_l1.consume())

    def compute_w_state(self, w_l1):
        mem_copy(self.l0a_chunk.produce(), w_l1.consume())
        matmul(self.l0c_u, self.l0a_chunk.consume(), self.l0b_state, init=True)

    def compute_q_state(self, q_decayed_l1):
        mem_copy(self.l0a_chunk.produce(), q_decayed_l1.consume())
        matmul(self.l0c_o, self.l0a_chunk.consume(), self.l0b_state, init=True)

    def load_k_restored_operand(self, k_restored_l1):
        mem_copy(self.l0b_kr, k_restored_l1.consume(), transpose=True)

    def load_mqk_operand(self, mqk_l1):
        mem_copy(self.l0a_mqk, mqk_l1.consume())

    def update_state(self, u_slot):
        u_src = reinterpret(u_slot, shape=(CHUNK_SIZE, self.dv_base), offset=0)
        mem_copy(self.l0a_kr_t, u_src, transpose=True)
        matmul(self.l0c_state, self.l0a_kr_t, self.l0b_kr, init=True)

    def accumulate_mqk_u(self, u_slot):
        u_tile = reinterpret(u_slot, shape=(CHUNK_SIZE, self.dv_base), offset=0)
        mem_copy(self.l0b_u, u_tile, transpose=True)
        matmul(self.l0c_o, self.l0a_mqk, self.l0b_u, init=False)

    def store_u_delta(self, ub):
        mem_copy(ub, tile_slice(self.l0c_u, (CHUNK_SIZE, self.dv_base), (0, 0)), engine=self.fixpipe_u)

    def store_output(self, gm_tile):
        """FIXPIPE 按目的 GM Access.actual_rows 写入逻辑尾块。"""
        mem_copy(
            gm_tile,
            self.l0c_o,
            l2_cache_ctl=_L2_CACHE_DISABLE,
        )

    def store_state_delta(self, ub):
        mem_copy(ub, self.l0c_state, engine=self.fixpipe_state)

    def load_k_cumdecay(self, k_cumdecay_l1, gm_tile, valid_rows):
        self._load_padded_chunk(k_cumdecay_l1, gm_tile, valid_rows)

    def load_k_restored(self, k_restored_l1, gm_tile, valid_rows):
        self._load_padded_chunk(k_restored_l1, gm_tile, valid_rows, l2_cache_ctl=_L2_CACHE_LAST_USE)

    def load_mqk(self, mqk_l1, gm_tile, valid_rows):
        self._load_padded_chunk(mqk_l1, gm_tile, valid_rows, l2_cache_ctl=_L2_CACHE_LAST_USE)

    @jit
    def _load_padded_chunk(self, channel, gm_tile, valid_rows, *, l2_cache_ctl=0):
        slot = channel.produce()
        if valid_rows < CHUNK_SIZE:
            for col_block in range(gm_tile.shape[1] // 16):
                gm_col = tile_slice(gm_tile, (CHUNK_SIZE, 16), (0, col_block))
                l1_col = tile_slice(slot, (CHUNK_SIZE, 16), (0, col_block))
                mem_copy(l1_col, _valid_rows_view(gm_col, valid_rows, 16), l2_cache_ctl=l2_cache_ctl, engine=self.gm2l1)
        else:
            mem_copy(slot, gm_tile, l2_cache_ctl=l2_cache_ctl, engine=self.gm2l1)


class Vector:
    def __init__(self, *, u_row_block: int, dv_base: int):
        self.dv_base = dv_base
        self.u_row_block = u_row_block
        self.half_dk = D_HALF
        self.ub_state_fp32 = Buffer(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.float32)
        self.tmp_u_bf16 = Buffer(MemLoc.UB, (u_row_block, dv_base), dtypes.bfloat16)
        self.state_nd = Channel(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.bfloat16, depth=1).produce()
        self.state_nz = Channel(
            MemLoc.UB, (self.dv_base, self.half_dk), dtypes.bfloat16, depth=1, data_format="nz", n1_pad=16
        ).produce()
        self.u_nz = Channel(
            MemLoc.UB, (u_row_block, dv_base), dtypes.bfloat16, depth=1, data_format="nz", n1_pad=16
        ).produce()
        self.state_in = Channel(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.float32, depth=1).produce()
        self.state_out = Channel(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.float32, depth=1).produce()
        self.u_in = Channel(MemLoc.UB, (u_row_block, dv_base), dtypes.bfloat16, depth=2)

    def load_gamma_row(self, gamma_ub, gamma_row, subblock_idx):
        """Load one row of ``decay[B,N,C,D]`` as the column consumed by VF."""
        row = tile_slice(gamma_row, (1, self.half_dk), (0, subblock_idx))
        slot = gamma_ub.produce()
        mem_copy(slot, row)

    def load_initial_state_and_write_l1(self, current_state, state_l1, subblock_idx):
        in_slot = self.state_in
        mem_copy(in_slot, current_state)
        in_read = self.state_in
        ub_state = self.ub_state_fp32
        mem_copy(ub_state, in_read)
        self._produce_state_l1(state_l1, subblock_idx)

    def write_state_l1_snapshot(self, state_l1, subblock_idx):
        """Publish the accumulated state for the next Cube step."""
        self._produce_state_l1(state_l1, subblock_idx)

    @jit
    def scale_and_add_delta(self, tmp_delta, gamma_ub):
        """Fuse state decay and delta accumulation into one full-state VF pass."""
        gamma_read = gamma_ub.consume()
        ub_gamma = gamma_read
        ub_state = self.ub_state_fp32
        delta = tmp_delta
        with vf(mode="simd"):
            mask, _ = update_mask(VL, elem_bits=32)
            gamma_vec = vload(ub_gamma, 0)
            for base in dsl_range(0, self.dv_base * self.half_dk, VL, unroll=4):
                state_v = vload(ub_state, base)
                delta_v = vload(delta, base)
                state_scaled = vmul(state_v, gamma_vec, mask=mask)
                state_next = vadd(state_scaled, delta_v, mask=mask)
                vstore(ub_state, base, state_next, mask)

    @jit
    def issue_u_rows(self, gm_u_tile, subblock_idx, valid_rows):
        row_start = subblock_idx * self.u_row_block
        if row_start < valid_rows:
            rows = valid_rows - row_start
            if rows > self.u_row_block:
                rows = self.u_row_block
            gm_u_rows = tile_slice(gm_u_tile, (self.u_row_block, self.dv_base), (subblock_idx, 0))
            mem_copy(
                _valid_rows_view(self.u_in.produce(), rows, self.dv_base),
                _valid_rows_view(gm_u_rows, rows, self.dv_base),
            )

    @jit
    def finish_u_delta_to_l1(self, u_l1, tmp_u_delta, subblock_idx, valid_rows):
        u_l1_rows = tile_slice(u_l1.produce(), (self.u_row_block, self.dv_base), (subblock_idx, 0))

        delta = tmp_u_delta
        row_start = subblock_idx * self.u_row_block
        rows = valid_rows - row_start
        if rows < 0:
            rows = 0
        if rows > self.u_row_block:
            rows = self.u_row_block
        if rows == self.u_row_block:
            u_in_read = self.u_in.consume()
            with vf(mode="simd"):
                mask, _ = update_mask(VL, elem_bits=32)
                for base in range(0, self.u_row_block * self.dv_base, 2 * VL):
                    u_raw_pre = vload_unpack(u_in_read, base, unpack_mode=UnpackMode.B16_TO_B32)
                    u_raw_post = vload_unpack(u_in_read, base + VL, unpack_mode=UnpackMode.B16_TO_B32)
                    u_fp32_pre = vcast(u_raw_pre, dtypes.float32, mask=mask)
                    u_fp32_post = vcast(u_raw_post, dtypes.float32, mask=mask)
                    delta_pre = vload(delta, base)
                    delta_post = vload(delta, base + VL)
                    u_next_pre = vadd(u_fp32_pre, delta_pre, mask=mask)
                    u_next_post = vadd(u_fp32_post, delta_post, mask=mask)
                    u_bf16_pre = vcast(u_next_pre, dtypes.bfloat16, mask=mask)
                    u_bf16_post = vcast(u_next_post, dtypes.bfloat16, mask=mask)
                    vstore_pack(self.tmp_u_bf16, base, u_bf16_pre, mask, pack_mode=PackMode.B32_TO_B16)
                    vstore_pack(self.tmp_u_bf16, base + VL, u_bf16_post, mask, pack_mode=PackMode.B32_TO_B16)
                vmem_bar("vst_vld")
        else:
            with vf(mode="simd"):
                bf16_mask, _ = update_mask(self.dv_base, elem_bits=16)
                zero = vdups(0.0, dtypes.bfloat16, mask=bf16_mask)
                for row in range(0, self.u_row_block):
                    vstore(self.tmp_u_bf16, row * self.dv_base, zero, bf16_mask)
                vmem_bar("vst_vld")
        if rows > 0 and rows < self.u_row_block:
            u_tail_read = self.u_in.consume()
            with vf(mode="simd"):
                for row in dsl_range(0, rows, 1, unroll=2):
                    for col in range(0, self.dv_base, VL):
                        mask, _ = update_mask(self.dv_base - col, elem_bits=32)
                        base = row * self.dv_base + col
                        u_raw = vload_unpack(u_tail_read, base, unpack_mode=UnpackMode.B16_TO_B32)
                        u_fp32 = vcast(u_raw, dtypes.float32, mask=mask)
                        delta_value = vload(delta, base)
                        u_next = vadd(u_fp32, delta_value, mask=mask)
                        u_bf16 = vcast(u_next, dtypes.bfloat16, mask=mask)
                        vstore_pack(self.tmp_u_bf16, base, u_bf16, mask, pack_mode=PackMode.B32_TO_B16)
                vmem_bar("vst_vld")

        u_nz_slot = self.u_nz
        nd2nz_tile(u_nz_slot, self.tmp_u_bf16)
        u_nz_read = self.u_nz
        mem_copy(u_l1_rows, u_nz_read)

    def store_state_to_gm(self, gm_state_rows):
        """末态 ub_state_fp32 → GM（FP32）。累加在 V pipe、写 GM 在 MTE3，跨 pipe，经 Channel 交接。"""
        out_slot = self.state_out
        mem_copy(out_slot, self.ub_state_fp32)
        out_read = self.state_out
        mem_copy(gm_state_rows, out_read)

    def _produce_state_l1(self, state_l1, subblock_idx):
        ub_state = self.ub_state_fp32
        nd_slot = self.state_nd
        cast_tile(nd_slot, ub_state)
        nd_read = self.state_nd
        nz_slot = self.state_nz
        nd2nz_tile(nz_slot, nd_read)
        nz_read = self.state_nz
        slot = state_l1.produce()
        state_l1_rows = tile_slice(slot, (self.dv_base, self.half_dk), (0, subblock_idx))
        mem_copy(state_l1_rows, nz_read)


@dataclasses.dataclass
class _SequenceTensors:
    kg: object
    qg: object
    w: object
    u: object
    output: object


@dataclasses.dataclass
class _StateTensors:
    decay: object
    state_in: object
    state_out: object
    mqk: object


@dataclasses.dataclass
class _MainTensors:
    sequence: _SequenceTensors
    state: _StateTensors
    metadata: object


@dataclasses.dataclass
class _SequenceWork:
    chunks_total: object
    seq_len: object
    group_start: object
    group_count: object
    logical_batch: object


@dataclasses.dataclass
class _WorkPartition:
    dv_base: object
    head_count: object
    logical_core_num: object
    item_start: object
    item_end: object


class _Resources:
    """Own the channels for one allocation region and preserve their creation order."""

    def __init__(self, dv_base):
        self.q_l1 = Channel(MemLoc.L1, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=2)
        self.w_l1 = Channel(MemLoc.L1, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=2)
        self.k_restored_l1 = Channel(
            MemLoc.L1, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=2, data_format="nz"
        )
        self.mqk_l1 = Channel(MemLoc.L1, shape=(CHUNK_SIZE, CHUNK_SIZE), dtype=dtypes.bfloat16, depth=2)
        self.state_l1 = Channel(
            MemLoc.L1,
            shape=(dv_base, SUPPORTED_HEAD_DIM),
            dtype=dtypes.bfloat16,
            depth=2,
            data_format="nz",
            kind=ChannelKind.CrossCore,
        )
        self.u_l1 = Channel(
            MemLoc.L1,
            shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM),
            dtype=dtypes.bfloat16,
            depth=2,
            kind=ChannelKind.CrossCore,
        )
        self.matmul = Matmul(dv_base=dv_base)
        self.gamma_ub = Channel(MemLoc.UB, shape=(1, D_HALF), dtype=dtypes.float32, depth=2)
        self.tmp_delta = Channel(
            MemLoc.UB, shape=(dv_base, D_HALF), dtype=dtypes.float32, depth=1, kind=ChannelKind.CrossCore
        ).produce()
        self.tmp_u_delta = Channel(
            MemLoc.UB, shape=(CHUNK_SIZE // 2, dv_base), dtype=dtypes.float32, depth=1, kind=ChannelKind.CrossCore
        ).produce()
        self.vector = Vector(u_row_block=CHUNK_SIZE // 2, dv_base=dv_base)


@jit
def _initialize_recurrence(tensors, work, resources, position):
    write_state_ref = tensors.state.state_out[position.logical_batch, position.head, None, None]
    first_valid_rows = CHUNK_SIZE
    if work.group_start + work.group_count == work.chunks_total:
        first_valid_rows = min(CHUNK_SIZE, work.seq_len - work.group_start * CHUNK_SIZE)

    first_w_chunk = _sequence_head_chunk_tile(
        tensors.sequence.w, position.storage_batch, position.head, work.group_start
    )
    first_q_chunk = _sequence_head_chunk_tile(
        tensors.sequence.qg, position.storage_batch, position.head, work.group_start
    )
    first_k_restored_chunk = _sequence_head_chunk_tile(
        tensors.sequence.kg, position.storage_batch, position.head, work.group_start
    )
    first_mqk_tile = _chunk_matrix_tile(tensors.state.mqk, position.logical_batch, position.head, work.group_start)
    first_decay_full = tensors.state.decay[position.logical_batch, position.head, None, None]
    first_decay_row = tile_slice(first_decay_full, (1, SUPPORTED_HEAD_DIM), (work.group_start, 0))
    first_u_tile_full = _sequence_head_chunk_tile(
        tensors.sequence.u, position.storage_batch, position.head, work.group_start
    )
    first_u_tile = tile_slice(first_u_tile_full, (CHUNK_SIZE, resources.matmul.dv_base), (0, position.dv_index))
    resources.matmul.load_k_cumdecay(resources.w_l1, first_w_chunk, first_valid_rows)
    resources.matmul.load_q_decayed(resources.q_l1, first_q_chunk, first_valid_rows)
    resources.matmul.load_k_restored(resources.k_restored_l1, first_k_restored_chunk, first_valid_rows)
    resources.matmul.load_mqk(resources.mqk_l1, first_mqk_tile, first_valid_rows)
    if work.group_start == 0:
        in_state_ref = tensors.state.state_in[position.logical_batch, position.head, None, None]
        first_state_rows = tile_slice(
            in_state_ref, (resources.matmul.dv_base, D_HALF), (position.dv_index, position.subblock)
        )
        resources.vector.load_initial_state_and_write_l1(first_state_rows, resources.state_l1, position.subblock)
    else:
        first_state_rows = tile_slice(
            write_state_ref, (resources.matmul.dv_base, D_HALF), (position.dv_index, position.subblock)
        )
        resources.vector.load_initial_state_and_write_l1(first_state_rows, resources.state_l1, position.subblock)
    resources.vector.load_gamma_row(resources.gamma_ub, first_decay_row, position.subblock)
    resources.vector.issue_u_rows(first_u_tile, position.subblock, first_valid_rows)


@jit
def _run_chunks(tensors, work, resources, position):
    write_state_ref = tensors.state.state_out[position.logical_batch, position.head, None, None]
    for chunk_id in range(0, work.group_count):
        k = work.group_start + chunk_id
        valid_rows = CHUNK_SIZE
        if k + 1 == work.chunks_total:
            valid_rows = work.seq_len - k * CHUNK_SIZE
        next_valid_rows = CHUNK_SIZE
        if k + 2 == work.chunks_total:
            next_valid_rows = work.seq_len - (k + 1) * CHUNK_SIZE
        o_chunk = _o_head_chunk_tile(tensors.sequence.output, position, k, resources.matmul.dv_base, valid_rows)
        resources.matmul.load_state(resources.state_l1)
        resources.matmul.compute_w_state(resources.w_l1)
        resources.matmul.compute_q_state(resources.q_l1)
        resources.matmul.load_k_restored_operand(resources.k_restored_l1)
        resources.matmul.load_mqk_operand(resources.mqk_l1)
        if chunk_id + 1 < work.group_count:
            next_q_chunk = _sequence_head_chunk_tile(tensors.sequence.qg, position.storage_batch, position.head, k + 1)
            next_w_chunk = _sequence_head_chunk_tile(tensors.sequence.w, position.storage_batch, position.head, k + 1)
            next_gamma_full = tensors.state.decay[position.logical_batch, position.head, None, None]
            next_decay_row = tile_slice(next_gamma_full, (1, SUPPORTED_HEAD_DIM), (k + 1, 0))
            resources.matmul.load_q_decayed(resources.q_l1, next_q_chunk, next_valid_rows)
            resources.matmul.load_k_cumdecay(resources.w_l1, next_w_chunk, next_valid_rows)
            resources.vector.load_gamma_row(resources.gamma_ub, next_decay_row, position.subblock)
            next_u_tile_full = _sequence_head_chunk_tile(
                tensors.sequence.u, position.storage_batch, position.head, k + 1
            )
            next_u_tile = tile_slice(next_u_tile_full, (CHUNK_SIZE, resources.matmul.dv_base), (0, position.dv_index))
            resources.vector.issue_u_rows(next_u_tile, position.subblock, next_valid_rows)
        resources.matmul.store_u_delta(resources.tmp_u_delta)

        resources.vector.finish_u_delta_to_l1(resources.u_l1, resources.tmp_u_delta, position.subblock, valid_rows)

        u_slot = resources.u_l1.consume()
        resources.matmul.update_state(u_slot)
        resources.matmul.store_state_delta(resources.tmp_delta)

        resources.matmul.accumulate_mqk_u(u_slot)
        if chunk_id + 1 < work.group_count:
            next_k_restored_chunk = _sequence_head_chunk_tile(
                tensors.sequence.kg, position.storage_batch, position.head, k + 1
            )
            next_mqk_tile = _chunk_matrix_tile(tensors.state.mqk, position.logical_batch, position.head, k + 1)
            resources.matmul.load_k_restored(resources.k_restored_l1, next_k_restored_chunk, next_valid_rows)
            resources.matmul.load_mqk(resources.mqk_l1, next_mqk_tile, next_valid_rows)
        resources.matmul.store_output(o_chunk)

        resources.vector.scale_and_add_delta(resources.tmp_delta, resources.gamma_ub)
        if chunk_id + 1 < work.group_count:
            resources.vector.write_state_l1_snapshot(resources.state_l1, position.subblock)
        else:
            state_rows = tile_slice(
                write_state_ref, (resources.matmul.dv_base, D_HALF), (position.dv_index, position.subblock)
            )
            resources.vector.store_state_to_gm(state_rows)


@jit
def _run_flash_kda_cp_body(tensors, work, partition):
    block_idx = get_block_idx()
    subblock_idx = get_subblock_id()
    num_dv_base = SUPPORTED_HEAD_DIM // partition.dv_base
    resources = _Resources(partition.dv_base)
    if block_idx < partition.logical_core_num:
        for item in range(partition.item_start, partition.item_end):
            head_id, dv_idx = idx2crd(item, [partition.head_count, num_dv_base])
            position = _ChunkPosition(0, work.logical_batch, head_id, dv_idx, subblock_idx)
            _initialize_recurrence(tensors, work, resources, position)
            _run_chunks(tensors, work, resources, position)


# mode=0 below is a whole-grid barrier.  The launch must fit in one resident
# cube-core wave, otherwise resident blocks wait for blocks that cannot be
# scheduled.  Keep 32 as the algorithmic cap and select the actual launch size
# from the target device on the host.
ALGORITHM_MAX_BLOCK_NUM = 32

ROUND_BOUNDARY_FLAG_ID = 1
_ROUND_BOUNDARY_AIC_FLAG_OFFSET = 4
_ROUND_BOUNDARY_AIC_AIV_FLAG_OFFSET = 8


def _rewind_buffers():
    channel_rewind(reset_sync_id=False)


def _round_boundary_reset(flag_id: int):
    cube_sync_all()
    vec_sync_all()

    aiv_ready_flag_id = flag_id
    aic_ready_flag_id = flag_id + _ROUND_BOUNDARY_AIC_FLAG_OFFSET
    aic_aiv_ready_flag_id = flag_id + _ROUND_BOUNDARY_AIC_AIV_FLAG_OFFSET

    vec_sync_block_arrive(PIPE.MTE3, aiv_ready_flag_id, mode=2)
    cube_sync_block_wait(PIPE.S, aiv_ready_flag_id, mode=2)

    cube_sync_block_arrive(PIPE.FIXPIPE, aic_ready_flag_id, mode=0)
    cube_sync_block_wait(PIPE.S, aic_ready_flag_id, mode=0)

    cube_sync_block_arrive(PIPE.MTE3, aic_aiv_ready_flag_id, mode=2)
    vec_sync_block_wait(PIPE.S, aic_aiv_ready_flag_id, mode=2)
    _rewind_buffers()


def _rewind_split_branch():
    """End the allocation region before selecting another dv split variant."""
    channel_rewind(reset_sync_id=True)


@dataclasses.dataclass
class _RoundSchedule:
    head_count: object
    active_batch_num: object
    batch_idx_base: object
    storage_batch_idx_base: object
    token_start_base: object
    valid_seq_len_base: object
    chunk_start_base: object
    chunk_num_base: object
    task_prefix_base: object
    core_start: object
    core_end: object
    logical_core_num: object


@jit
def _run_flash_kda_cp_round(tensors, schedule, dv_base):
    for active_batch_idx in range(0, schedule.active_batch_num):
        prefix_start = tensors.metadata[schedule.task_prefix_base + active_batch_idx]
        prefix_end = tensors.metadata[schedule.task_prefix_base + active_batch_idx + 1]
        intersection_start = schedule.core_start
        if intersection_start < prefix_start:
            intersection_start = prefix_start
        intersection_end = schedule.core_end
        if intersection_end > prefix_end:
            intersection_end = prefix_end
        item_start = 0
        item_end = 0
        if intersection_start < intersection_end:
            item_start = intersection_start - prefix_start
            item_end = intersection_end - prefix_start

        logical_batch = tensors.metadata[schedule.batch_idx_base + active_batch_idx]
        storage_batch = tensors.metadata[schedule.storage_batch_idx_base + active_batch_idx]
        token_start = tensors.metadata[schedule.token_start_base + active_batch_idx]
        seq_len = dtypes.int64(tensors.metadata[schedule.valid_seq_len_base + active_batch_idx])
        group_start = tensors.metadata[schedule.chunk_start_base + active_batch_idx]
        group_count = tensors.metadata[schedule.chunk_num_base + active_batch_idx]
        chunks_total = (seq_len + (CHUNK_SIZE - 1)) // CHUNK_SIZE
        token_stop = token_start + seq_len

        # Rebase each logical sequence to a bounded view.  This is valid for
        # both TND and host-padded BNSD/BSND storage and makes the tile
        # coordinate in _sequence_head_chunk_tile a chunk index.
        kg = tensors.sequence.kg[storage_batch, None, token_start:token_stop, None]
        qg = tensors.sequence.qg[storage_batch, None, token_start:token_stop, None]
        w = tensors.sequence.w[storage_batch, None, token_start:token_stop, None]
        u = tensors.sequence.u[storage_batch, None, token_start:token_stop, None]
        output = tensors.sequence.output[storage_batch, None, token_start:token_stop, None]
        sequence = _SequenceTensors(kg, qg, w, u, output)
        bounded_tensors = _MainTensors(sequence, tensors.state, tensors.metadata)
        work = _SequenceWork(chunks_total, seq_len, group_start, group_count, logical_batch)
        partition = _WorkPartition(dv_base, schedule.head_count, schedule.logical_core_num, item_start, item_end)
        _run_flash_kda_cp_body(bounded_tensors, work, partition)


@jit
def _run_flash_kda_cp_dynamic(tensors, schedule, dv_splits_num):
    if dv_splits_num == 1:
        _rewind_split_branch()
        _run_flash_kda_cp_round(tensors, schedule, 128)
    elif dv_splits_num == 2:
        _rewind_split_branch()
        _run_flash_kda_cp_round(tensors, schedule, 64)
    elif dv_splits_num == 4:
        _rewind_split_branch()
        _run_flash_kda_cp_round(tensors, schedule, 32)
    else:
        _rewind_split_branch()
        _run_flash_kda_cp_round(tensors, schedule, 16)


@kernel(
    profile=cannbotdsl.ProfileSpec(
        name="flash_kda_cp_main",
        op_type="FlashKDACPMain",
        inputs=("gm_kg", "gm_decay", "gm_qg", "gm_w", "gm_u", "gm_mqk", "gm_state_in", "gm_metadata"),
        outputs=("gm_output", "gm_state_out"),
        inouts=(),
    )
)
class FlashKdaCpKernel:
    """Run the recurrence from CP-prepared GM tensors."""

    def __init__(self, head_dim: int = SUPPORTED_HEAD_DIM):
        self.head_dim = head_dim

    def __call__(
        self,
        gm_kg: Tensor,
        gm_decay: Tensor,
        gm_qg: Tensor,
        gm_w: Tensor,
        gm_u: Tensor,
        gm_mqk: Tensor,
        gm_output: Tensor,
        gm_state_in: Tensor,
        gm_state_out: Tensor,
        gm_metadata: Tensor,
        bn_total: int,
        seq_len: int,
    ):
        if const_expr(self.head_dim != SUPPORTED_HEAD_DIM):
            raise ValueError("FlashKDA CP requires head_dim=128")
        sequence = _SequenceTensors(gm_kg, gm_qg, gm_w, gm_u, gm_output)
        state = _StateTensors(gm_decay, gm_state_in, gm_state_out, gm_mqk)
        tensors = _MainTensors(sequence, state, gm_metadata)
        block_idx = get_block_idx()
        block_num = get_block_num()
        metadata_status = gm_metadata[HEADER_STATUS]
        metadata_core_num = gm_metadata[HEADER_CORE_NUM]
        metadata_batch = dtypes.int64(gm_metadata[HEADER_BATCH])
        metadata_value_heads = dtypes.int64(gm_metadata[HEADER_VALUE_HEADS])
        if (
            metadata_status == STATUS_OK
            and metadata_core_num == block_num
            and metadata_batch * metadata_value_heads == bn_total
        ):
            round_num = gm_metadata[HEADER_ROUND_COUNT]
            for round_idx in range(0, round_num):
                record = gm_metadata[HEADER_WORDS + round_idx]
                active_batch_num = gm_metadata[record + 1]
                batch_idx_base = record + 2
                storage_batch_idx_base = batch_idx_base + active_batch_num
                token_start_base = storage_batch_idx_base + active_batch_num
                valid_seq_len_base = token_start_base + active_batch_num
                group_num_index = valid_seq_len_base + active_batch_num
                chunk_start_base = group_num_index + 1
                chunk_num_base = chunk_start_base + active_batch_num
                # Skip preprocess scheduling fields reserved by the shared CP metadata ABI.
                reserved_task_prefix_base = chunk_num_base + active_batch_num
                reserved_core_ranges_base = reserved_task_prefix_base + active_batch_num + 1
                dv_splits_base = reserved_core_ranges_base + MAX_AIC_CORES * 2
                task_prefix_base = dv_splits_base + active_batch_num
                core_ranges_base = task_prefix_base + active_batch_num + 1
                core_range_word = core_ranges_base + block_idx * 2
                core_start = gm_metadata[core_range_word]
                core_end = gm_metadata[core_range_word + 1]
                dv_splits_num = gm_metadata[dv_splits_base]
                schedule = _RoundSchedule(
                    head_count=metadata_value_heads,
                    active_batch_num=active_batch_num,
                    batch_idx_base=batch_idx_base,
                    storage_batch_idx_base=storage_batch_idx_base,
                    token_start_base=token_start_base,
                    valid_seq_len_base=valid_seq_len_base,
                    chunk_start_base=chunk_start_base,
                    chunk_num_base=chunk_num_base,
                    task_prefix_base=task_prefix_base,
                    core_start=core_start,
                    core_end=core_end,
                    logical_core_num=block_num,
                )
                _run_flash_kda_cp_dynamic(tensors, schedule, dv_splits_num)
                if round_idx + 1 < round_num:
                    _round_boundary_reset(ROUND_BOUNDARY_FLAG_ID)


# Retain the historical exported kernel name for existing callers.
flash_kda_cp_kernel = FlashKdaCpKernel


class FlashKDACP:
    """Host/JIT wrapper for the standalone CP Main kernel."""

    def __init__(self, head_dim: int = SUPPORTED_HEAD_DIM):
        self.head_dim = head_dim

    @host
    def run(
        self,
        kg: Tensor,
        decay: Tensor,
        qg: Tensor,
        w: Tensor,
        u: Tensor,
        mqk: Tensor,
        output: Tensor,
        state_in: Tensor,
        state_out: Tensor,
        metadata: Tensor,
        batch_head_count: int,
        seq_len: int,
        core_num: int,
    ):
        op = FlashKdaCpKernel(self.head_dim)
        op[core_num](kg, decay, qg, w, u, mqk, output, state_in, state_out, metadata, batch_head_count, seq_len)


def _device_block_num(ref: Optional[torch.Tensor] = None) -> int:
    """Return the grid-barrier-safe AICore count for the selected NPU."""
    if ref is not None and ref.device.type not in {"npu", "privateuseone"}:
        return ALGORITHM_MAX_BLOCK_NUM

    npu = getattr(torch, "npu", None)
    if npu is None:
        return ALGORITHM_MAX_BLOCK_NUM

    device_index = ref.device.index if ref is not None else None
    if device_index is None:
        try:
            device_index = npu.current_device()
        except RuntimeError:
            # Preserve CPU-only/offline tracing.  A targeted offline compile
            # can still pass block_num explicitly.
            return ALGORITHM_MAX_BLOCK_NUM

    stream = npu.current_stream(device_index)
    platform_info = get_platform_info(stream=stream)
    try:
        aicore_num = int(platform_info.cube_core_num)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"NPU {device_index} does not expose a valid AICore count") from exc
    if aicore_num <= 0:
        raise RuntimeError(f"NPU {device_index} reports invalid AICore count={aicore_num}")
    return min(aicore_num, ALGORITHM_MAX_BLOCK_NUM)


def _cp_sequence_tensor_spec(dtype, shape, prefix):
    """Dynamic CP input/output ABI with runtime physical strides."""
    strides = (
        cannbotdsl.Dim(f"{prefix}_stride_0"),
        cannbotdsl.Dim(f"{prefix}_stride_1"),
        cannbotdsl.Dim(f"{prefix}_stride_2"),
        1,
    )
    return cannbotdsl.TensorSpec(shape, dtype, stride=strides)


@compile_cache
def _compile_cp_main_kernel():
    physical_batch = cannbotdsl.Dim("P")
    batch = cannbotdsl.Dim("B")
    seq_len = cannbotdsl.Dim("S")
    output_len = cannbotdsl.Dim("OL")
    value_heads = cannbotdsl.Dim("NV")
    metadata_capacity = cannbotdsl.Dim("META")
    # Match preprocess and the TND upper-bound allocation.  Per-sequence
    # valid chunk counts remain in metadata, so these extra slots are never
    # traversed by the kernel.
    chunks = _chunk_capacity(seq_len, batch)
    fake = cannbotdsl.TensorSpec
    kg_spec = _cp_sequence_tensor_spec(
        dtypes.bfloat16, (physical_batch, value_heads, seq_len, SUPPORTED_HEAD_DIM), "KG"
    )
    qg_spec = _cp_sequence_tensor_spec(
        dtypes.bfloat16, (physical_batch, value_heads, seq_len, SUPPORTED_HEAD_DIM), "QG"
    )
    w_spec = _cp_sequence_tensor_spec(dtypes.bfloat16, (physical_batch, value_heads, seq_len, SUPPORTED_HEAD_DIM), "W")
    u_spec = _cp_sequence_tensor_spec(dtypes.bfloat16, (physical_batch, value_heads, seq_len, SUPPORTED_HEAD_DIM), "U")
    decay_spec = fake((batch, value_heads, chunks, SUPPORTED_HEAD_DIM), dtypes.float32)
    mqk_spec = fake((batch, value_heads, chunks, CHUNK_SIZE, CHUNK_SIZE), dtypes.bfloat16)
    output_spec = _cp_sequence_tensor_spec(
        dtypes.bfloat16, (physical_batch, value_heads, output_len, SUPPORTED_HEAD_DIM), "O"
    )
    state_spec = fake((batch, value_heads, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), dtypes.float32)
    metadata_spec = fake((metadata_capacity,), dtypes.int32)
    return cannbotdsl.compile(
        FlashKDACP().run,
        kg_spec,
        decay_spec,
        qg_spec,
        w_spec,
        u_spec,
        mqk_spec,
        output_spec,
        state_spec,
        state_spec,
        metadata_spec,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
    )


@dataclasses.dataclass
class _CPShape:
    batch: int
    seq_len: int
    heads: int
    width: int

    @property
    def padded_len(self):
        return _chunk_count(self.seq_len) * CHUNK_SIZE


def _validate_sequence_layout(sequences, layout, cu_seqlens):
    if layout not in ("BNSD", "BSND", "TND"):
        raise ValueError("layout_qkv must be BNSD, BSND, or TND")
    qg = sequences["qg"]
    rank = 3 if layout == "TND" else 4
    if qg.dim() != rank:
        raise ValueError(f"qg must be rank-{rank} for {layout}")
    if any(tensor.shape != qg.shape for tensor in sequences.values()):
        raise ValueError("qg, kg, u, and w must have the same shape")
    if layout != "TND" and cu_seqlens is not None:
        raise ValueError("cu_seqlens is only supported for TND")


def _packed_sequence_views(sequences, cu_seqlens):
    qg = sequences["qg"]
    if cu_seqlens is None or cu_seqlens.dtype != torch.int32 or cu_seqlens.dim() != 1:
        raise ValueError("cu_seqlens must be a rank-1 int32 tensor for TND")
    if not cu_seqlens.is_contiguous() or cu_seqlens.device != qg.device:
        raise ValueError("cu_seqlens must be contiguous and on the same device as qg")
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least two boundaries")
    seq_len, heads, width = qg.shape
    if int(cu_seqlens[0].item()) != 0 or int(cu_seqlens[-1].item()) != seq_len:
        raise ValueError("cu_seqlens must start at zero and end at the token count")
    shape = _CPShape(cu_seqlens.numel() - 1, seq_len, heads, width)
    views = tuple(
        torch.as_strided(tensor, (1, heads, seq_len, width), (seq_len * heads * width, width, heads * width, 1))
        for tensor in sequences.values()
    )
    return shape, views


def _dense_sequence_views(sequences, initial_state, layout):
    if initial_state.dim() != 4:
        raise ValueError("local_initial_state must have rank-4 shape [batch, heads, 128, 128]")
    qg = sequences["qg"]
    state_heads = initial_state.shape[1]
    # Prefer canonical BNSD intermediates; accept the historical BSND test layout.
    if layout == "BSND" and qg.shape[1] != state_heads and qg.shape[2] == state_heads:
        batch, seq_len, heads, width = qg.shape
        views = tuple(tensor.transpose(1, 2) for tensor in sequences.values())
    else:
        if qg.shape[1] != state_heads:
            raise ValueError("qg head count must match local_initial_state")
        batch, heads, seq_len, width = qg.shape
        views = tuple(sequences.values())
    return _CPShape(batch, seq_len, heads, width), views


def _validate_tensor_contract(inputs, shape):
    if shape.width != SUPPORTED_HEAD_DIM or shape.seq_len <= 0:
        raise ValueError("head dimension must be 128 and sequence length must be positive")
    chunks = _chunk_capacity(shape.seq_len, shape.batch)
    state_shape = (shape.batch, shape.heads, shape.width, shape.width)
    expected_shapes = {
        "decay": (shape.batch, shape.heads, chunks, shape.width),
        "mqk": (shape.batch, shape.heads, chunks, CHUNK_SIZE, CHUNK_SIZE),
        "local_initial_state": state_shape,
        "transition": state_shape,
        "state_ext": state_shape,
    }
    for name, expected in expected_shapes.items():
        if tuple(inputs[name].shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(inputs[name].shape)}")
    for name, tensor in inputs.items():
        expected_dtype = LOW_DTYPE if name in {"qg", "kg", "u", "w", "mqk"} else HIGH_DTYPE
        if tensor.dtype != expected_dtype:
            raise ValueError(f"{name} must have dtype {expected_dtype}, got {tensor.dtype}")


def _validate_devices_and_metadata(inputs, metadata):
    qg = inputs["qg"]
    if any(tensor.device != qg.device for tensor in inputs.values()):
        raise ValueError("all input tensors must be on the same device as qg")
    if qg.device.type not in {"npu", "privateuseone"}:
        raise ValueError("FlashKDA CP requires an NPU tensor")
    if metadata.dtype != torch.int32:
        raise ValueError("metadata must be int32")
    if metadata.dim() != 1:
        raise ValueError("metadata must be rank-1")
    if not metadata.is_contiguous():
        raise ValueError("metadata must be contiguous")
    if metadata.device != qg.device:
        raise ValueError("metadata must be on the same device as qg")


def _pad_sequence_views(views, shape):
    if shape.padded_len == shape.seq_len:
        return views
    padded_views = []
    for tensor in views:
        padded = torch.zeros(
            (shape.batch, shape.heads, shape.padded_len, shape.width), dtype=tensor.dtype, device=tensor.device
        )
        padded[:, :, : shape.seq_len, :].copy_(tensor)
        padded_views.append(padded)
    return tuple(padded_views)


def _allocate_output(shape, layout, device):
    if layout == "TND":
        storage = torch.empty((shape.padded_len, shape.heads, shape.width), dtype=LOW_DTYPE, device=device)
        physical = storage[: shape.seq_len]
        output = torch.as_strided(
            storage,
            (1, shape.heads, shape.padded_len, shape.width),
            (shape.padded_len * shape.heads * shape.width, shape.width, shape.heads * shape.width, 1),
        )
    else:
        output = torch.empty((shape.batch, shape.heads, shape.padded_len, shape.width), dtype=LOW_DTYPE, device=device)
        physical = output[:, :, : shape.seq_len]
        if layout == "BSND":
            physical = physical.transpose(1, 2)
    return physical, output


def flash_kda_cp_main(
    transition: torch.Tensor,
    state_ext: torch.Tensor,
    decay: torch.Tensor,
    qg: torch.Tensor,
    kg: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    mqk: torch.Tensor,
    local_initial_state: torch.Tensor,
    layout_qkv: str,
    metadata: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the FlashKDA CP recurrence from preprocessed tensors and a rank-local initial state.

    Dimensions:
        B: Number of logical sequences.
        Nv: Number of value heads.
        S: Tokens per sequence for dense BNSD/BSND layouts.
        T: Total packed tokens for TND layout.
        D: Query/key and value head dimension, fixed at 128.
        C: Chunk capacity, ceil(S / 64) + B - 1 for dense layouts or ceil(T / 64) + B - 1 for TND.
        M: Metadata word capacity returned by the metadata factory.

    Args:
        transition (torch.Tensor): Rank-local affine transition from preprocess, float32 ``[B, Nv, D, D]``.
        state_ext (torch.Tensor): Rank-local affine additive state from preprocess, float32 ``[B, Nv, D, D]``.
        decay (torch.Tensor): Per-chunk terminal channel decay, float32 ``[B, Nv, C, D]``.
        qg (torch.Tensor): Decayed query including QK scale, bfloat16 ``[B, Nv, S, D]`` for both BNSD and
            BSND, or ``[T, Nv, D]`` for TND. Dense preprocess intermediates always use BNSD storage.
        kg (torch.Tensor): State-update key, bfloat16 ``[B, Nv, S, D]`` for both BNSD and BSND, or
            ``[T, Nv, D]`` for TND.
        u (torch.Tensor): Corrected preprocess value, bfloat16 ``[B, Nv, S, D]`` for both BNSD and BSND,
            or ``[T, Nv, D]`` for TND.
        w (torch.Tensor): Signed key correction, bfloat16 ``[B, Nv, S, D]`` for both BNSD and BSND, or
            ``[T, Nv, D]`` for TND.
        mqk (torch.Tensor): Intra-chunk QK matrix with diagonal-inclusive lower-triangular mask,
            bfloat16 ``[B, Nv, C, 64, 64]``.
        local_initial_state (torch.Tensor): State produced by the external CP prefix scan,
            float32 ``[B, Nv, D, D]``.
        layout_qkv (str): ``"BNSD"``, ``"BSND"`` or ``"TND"`` output layout. BSND changes only the output
            layout; its dense intermediates have the same BNSD shape as BNSD intermediates.
        metadata (torch.Tensor): Schedule from ``flash_kda_cp_main_metadata()``, int32 ``[M]``.
        cu_seqlens (torch.Tensor, optional): TND sequence boundaries, int32 ``[B + 1]``, required for TND
            and None otherwise. Values strictly increase from 0 to T, giving positive sequence lengths.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (out, final_state).
        out (torch.Tensor): Rank-local output, bfloat16 ``[B, Nv, S, D]`` for BNSD, ``[B, S, Nv, D]`` for
            BSND, or ``[T, Nv, D]`` for TND.
        final_state (torch.Tensor): Final rank-local recurrent state, float32 ``[B, Nv, D, D]``.

    Notes:
        All input tensors must be contiguous and on the same NPU. B, Nv and S (dense) or T (TND)
        must be positive. Each logical sequence consumes ceil(length / 64) valid chunks; extra
        capacity slots in decay and mqk are not consumed.
        Generate metadata with the same tensor attributes, layout, sequence boundaries and current
        stream core count used for this call.
        The external CP prefix scan combines transition and state_ext with the global initial state
        to produce local_initial_state. This interface validates transition and state_ext but does
        not use their values in the recurrence.
    """
    sequences = {"qg": qg, "kg": kg, "u": u, "w": w}
    _validate_sequence_layout(sequences, layout_qkv, cu_seqlens)
    if layout_qkv == "TND":
        shape, views = _packed_sequence_views(sequences, cu_seqlens)
    else:
        shape, views = _dense_sequence_views(sequences, local_initial_state, layout_qkv)
    inputs = {
        **sequences,
        "transition": transition,
        "state_ext": state_ext,
        "decay": decay,
        "mqk": mqk,
        "local_initial_state": local_initial_state,
    }
    _validate_tensor_contract(inputs, shape)
    _validate_devices_and_metadata(inputs, metadata)
    if layout_qkv != "TND":
        views = _pad_sequence_views(views, shape)
    qg_view, kg_view, u_view, w_view = views
    block_num = _device_block_num(qg)
    execution = _CP_EXECUTION_CACHE.get_or_create(ref=qg, block_num=block_num)
    output_physical, output = _allocate_output(shape, layout_qkv, qg.device)
    final_state = torch.empty_like(local_initial_state)
    main_fn = _compile_cp_main_kernel()
    with execution.launch_lock:
        main_fn(
            kg_view,
            decay,
            qg_view,
            w_view,
            u_view,
            mqk,
            output,
            local_initial_state,
            final_state,
            metadata,
            shape.batch * shape.heads,
            shape.seq_len,
            block_num,
        )
    return output_physical, final_state


MAX_AIC_CORES = ALGORITHM_MAX_BLOCK_NUM
ABI_VERSION = 2
HEADER_STATUS = 2
HEADER_CORE_NUM = 3
HEADER_ROUND_COUNT = 4
HEADER_BATCH = 5
HEADER_VALUE_HEADS = 6
HEADER_WORDS = 9
STATUS_OK = 0

if aot is not None and hasattr(aot, "export"):

    @aot.export("flash_kda_cp_main")
    def export_flash_kda_cp_main():
        """Export the dynamic CP main AICore binary."""
        _compile_cp_main_kernel()


__all__ = ["flash_kda_cp_main", "FlashKdaCpKernel", "flash_kda_cp_kernel", "clear_execution_cache"]
