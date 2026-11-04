# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# fmt: off

"""FlashKDA CP preprocess public operator."""

import dataclasses
import os
import threading
from collections import OrderedDict
from functools import lru_cache
from typing import NamedTuple, Optional

import cannbotdsl
import torch
from cannbotdsl import (
    PIPE,
    Array,
    MemLoc,
    Tensor,
    dtypes,
    get_platform_info,
    make_layout,
)
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang import vf
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops import reg as reg_ops
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    full_mask,
    mask_xor,
    update_mask,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdup,
    vdups,
    vexp,
    vload,
    vload_broadcast,
    vload_unpack,
    vmem_bar,
    vmul,
    vmuls,
    vreduce_sum,
    vsqrt,
    vstore,
    vstore_pack,
    vsub,
)
from cannbotdsl.ops.reg import vselect as vselect_raw
from cannbotdsl.ops.sync import (
    cube_fill_l1_zero,
    cube_sync_all,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    cube_sync_pipe,
    onchip_rewind,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
    vec_sync_notify,
    vec_sync_wait,
)
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import idx2crd, reinterpret, tile_slice, tile_slice_multi

try:
    from cannbotdsl import aot
except ImportError:  # pragma: no cover - compatibility with pre-0.7 hosts
    aot = None

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

compile_cache = getattr(aot, "compile_cache", lru_cache(maxsize=1))


@jit
def _cast_vector_row(dst, src, row):
    cols = dst.shape[1]
    src_stride = src.physical_stride[0]
    dst_stride = dst.physical_stride[0]
    for col in range(0, cols, 64):
        mask, _ = reg_ops.update_mask(cols - col, elem_bits=32)
        if const_expr(src.dtype == dtypes.float32):
            value = reg_ops.vload(src, row * src_stride + col)
        else:
            value = reg_ops.vload_unpack(src, row * src_stride + col, unpack_mode=reg_ops.UnpackMode.B16_TO_B32)
        converted = reg_ops.vcast(value, dst.dtype, mask=mask)
        if const_expr(dst.dtype == dtypes.float32):
            reg_ops.vstore(dst, row * dst_stride + col, converted, mask)
        else:
            reg_ops.vstore_pack(dst, row * dst_stride + col, converted, mask, pack_mode=reg_ops.PackMode.B32_TO_B16)


@jit
def cast_tile(dst, src):
    """Convert a 2D tile while preserving physical row strides."""
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
                _cast_vector_row(dst, src, row)


@jit
def prefix_sum_rows(dst, src):
    """Inclusive f32 prefix along rows, with the original sequential add order."""
    rows, cols = dst.shape
    src_stride = src.physical_stride[0]
    dst_stride = dst.physical_stride[0]
    with vf(mode="simd"):
        for col in range(0, cols, 64):
            mask, _ = reg_ops.update_mask(cols - col, elem_bits=32)
            prefix = reg_ops.vload(src, col)
            reg_ops.vstore(dst, col, prefix, mask)
            for row in range(1, rows):
                row_value = reg_ops.vload(src, row * src_stride + col)
                prefix = reg_ops.vadd(prefix, row_value, mask=mask)
                reg_ops.vstore(dst, row * dst_stride + col, prefix, mask)


@jit
def nd2nz_tile(dst, src, elem_bits: int = 16):
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
HALF_CHUNK_SIZE = CHUNK_SIZE // 2
D_HALF = SUPPORTED_HEAD_DIM // 2
LOW_DTYPE = torch.bfloat16
HIGH_DTYPE = torch.float32
SUMMARY_MTE2_TO_V_EVENT_ID = 0
SUMMARY_RANK_PHI_READY_FLAG_ID = 10


@dataclasses.dataclass(frozen=True)
class _StageOneInputs:
    k: object = None
    v: object = None
    q: object = None
    beta: object = None
    g: object = None


@dataclasses.dataclass(frozen=True)
class _StageOneOutputs:
    qg: object = None
    mqk: object = None
    kg: object = None
    decay: object = None
    u: object = None
    w: object = None


@dataclasses.dataclass(frozen=True)
class _StageOneResources:
    exchange: object = None
    identity: object = None
    head_dim: object = None
    scale: object = None
    gate: object = None


@dataclasses.dataclass(frozen=True)
class _GateParameters:
    alpha: object = None
    bias: object = None
    lower_bound: object = None


@dataclasses.dataclass(frozen=True)
class _StageOneSchedule:
    head_base: object = None
    start: object = None
    count: object = None
    tail_rows: object = None
    tail_group: object = None
    tasks: object = None


@dataclasses.dataclass(frozen=True)
class _TaskRange:
    start: object = None
    end: object = None


@dataclasses.dataclass(frozen=True)
class _StageOneRuntime:
    matmul: object = None
    vector: object = None
    identity: object = None
    subblock: object = None
    head_dim: object = None
    block: object = None


@dataclasses.dataclass(frozen=True)
class _ChunkLocation:
    batch: object = None
    head: object = None
    chunk: object = None
    parity: object = None
    valid_rows: object = None
    qk_head: object = None


@dataclasses.dataclass(frozen=True)
class _StageOneTask:
    schedule: object
    index: object
    task_indices: object
    valid_row_counts: object
    write_slot: object


@dataclasses.dataclass(frozen=True)
class _SummaryInputs:
    kg: object = None
    decay: object = None
    u: object = None
    w: object = None


@dataclasses.dataclass(frozen=True)
class _SummaryOutputs:
    transition: object = None
    state: object = None


@dataclasses.dataclass(frozen=True)
class _SummarySchedule:
    head_base: object = None
    head_count: object = None
    group_start: object = None
    group_count: object = None
    item_start: object = None
    item_end: object = None


@dataclasses.dataclass(frozen=True)
class _SummarySequence:
    storage_batch: object = None
    logical_batch: object = None
    length: object = None


@dataclasses.dataclass(frozen=True)
class _SequenceShape:
    batch: object = None
    length: object = None
    heads: object = None
    width: object = None
    key_heads: object = None
    chunks: object = None


@dataclasses.dataclass(frozen=True)
class _PhiDestination:
    phi: object = None
    rank: object = None
    transition: object = None


@dataclasses.dataclass(frozen=True)
class _PhiPublication:
    half: object = None
    write_transition: object = None
    publish_phi: object = None
    initialize_rank: object = None
    state: object = None


@dataclasses.dataclass(frozen=True)
class _SummaryWorkspace:
    matmul: object = None
    vector: object = None
    w: object = None
    kg: object = None
    phi_ub: object = None
    phi: object = None
    initial_phi: object = None
    u: object = None
    recurrent_phi: object = None
    psi: object = None
    loaded_phi: object = None
    loaded_psi: object = None
    psi_ub: object = None


@dataclasses.dataclass(frozen=True)
class _SummaryChunk:
    subblock: object = None
    state: object = None
    group_start: object = None
    index: object = None
    group_end: object = None
    first: object = None
    last: object = None


@dataclasses.dataclass(frozen=True)
class _RoundSchedule:
    heads: object = None
    batch_count: object = None
    batch_base: object = None
    storage_base: object = None
    token_base: object = None
    length_base: object = None
    chunk_start_base: object = None
    chunk_count_base: object = None
    task_prefix_base: object = None
    core_start: object = None
    core_end: object = None
    splits_base: object = None


def _chunk_count(seq_len):
    return (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE


def _chunk_capacity(seq_len, batch):
    """Uniform intermediate capacity shared by all CP layouts."""
    return _chunk_count(seq_len) + batch - 1


def _tail_rows(seq_len):
    return seq_len - (_chunk_count(seq_len) - 1) * CHUNK_SIZE


def _state_head_tile(gm_state, seq_idx):
    """Select one logical matrix from rank-4 batch/head storage."""
    nv = gm_state.shape[1]
    batch_idx = seq_idx // nv
    head_idx = seq_idx - batch_idx * nv
    return tile_slice(gm_state[batch_idx, head_idx, None, None], (gm_state.shape[2], gm_state.shape[3]), (0, 0))


def _sequence_head_chunk_tile(gm_tensor, batch_idx, head_idx, chunk_idx):
    return tile_slice(gm_tensor[batch_idx, head_idx, None, None], (CHUNK_SIZE, gm_tensor.shape[3]), (chunk_idx, 0))


def _chunk_matrix_tile(gm_tensor, batch_idx, head_idx, chunk_idx):
    return tile_slice(gm_tensor[batch_idx, head_idx, chunk_idx, None, None], (CHUNK_SIZE, CHUNK_SIZE), (0, 0))


class _Stage1Constants(NamedTuple):
    identity16: torch.Tensor


_STAGE1_CONSTANTS_CACHE: dict[tuple[str, int | None, torch.dtype], _Stage1Constants] = {}


def _empty_like_device(
    shape: tuple[int, ...],
    *,
    ref: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=ref.device)


def allocate_qk_exchange_workspace(
    *,
    active_block_num: int,
    ref: torch.Tensor,
) -> torch.Tensor:
    """Allocate one pair of fp32 Q/K half-exchange tiles per physical block."""
    if not (active_block_num > 0):
        raise ValueError("active_block_num must be positive")
    return _empty_like_device((active_block_num, 2, CHUNK_SIZE, D_HALF), ref=ref, dtype=HIGH_DTYPE)


def _resolved_device_key(ref: torch.Tensor) -> tuple[str, int | None]:
    device = ref.device
    device_type = getattr(device, "type", "cpu")
    device_index = getattr(device, "index", None)
    if device_type in {"npu", "privateuseone"} and device_index is None:
        npu = getattr(torch, "npu", None)
        current_device = getattr(npu, "current_device", None) if npu is not None else None
        if current_device is not None:
            device_index = int(current_device())
    return device_type, device_index


def _stage1_constants_for(ref: torch.Tensor, *, identity_dtype: torch.dtype = torch.float16) -> _Stage1Constants:
    device_type, device_index = _resolved_device_key(ref)
    key = (device_type, device_index, identity_dtype)
    cached = _STAGE1_CONSTANTS_CACHE.get(key)
    if cached is not None:
        return cached

    constants = _Stage1Constants(
        identity16=torch.eye(NEUMANN_BLOCK_SIZE, dtype=identity_dtype, device=ref.device).contiguous()
    )
    _STAGE1_CONSTANTS_CACHE[key] = constants
    return constants


def _stream_identity(ref: torch.Tensor) -> tuple[str, int | None, int | None]:
    """Identify the stream that owns the cached exchange workspace."""
    device = ref.device
    if device.type not in {"npu", "privateuseone"}:
        return device.type, device.index, None
    device_index = ref.get_device() if hasattr(ref, "get_device") else device.index
    stream = torch.npu.current_stream(device_index)
    stream_id = getattr(stream, "npu_stream", stream)
    return device.type, device_index, int(stream_id)


@dataclasses.dataclass
class _CPExecutionEntry:
    qk_exchange: torch.Tensor
    launch_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


class _CPExecutionCache:
    """Keep stream-private CP buffers until explicit cleanup or LRU eviction."""

    def __init__(self, max_entries: int = 16):
        self._max_entries = max_entries
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    def get_or_create(self, *, ref: torch.Tensor, block_num: int) -> _CPExecutionEntry:
        key = (*_stream_identity(ref), block_num)
        with self._lock:
            cached = self._entries.pop(key, None)
            if cached is None:
                cached = _CPExecutionEntry(
                    qk_exchange=allocate_qk_exchange_workspace(
                        active_block_num=block_num,
                        ref=ref,
                    )
                )
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


VL = 64  # 一个 256B 向量寄存器的 f32/b32 lane 数（raw VF 步长）；此处恰等于 CHUNK_SIZE，一行 = 一寄存器

NEUMANN_BLOCK_SIZE = 16
NEUMANN_DIAG_BLOCK_NUM = 4
NEUMANN_FRACTAL_ELEMS = 256  # 一个 16x16 块的元素数。


class StageOneMatmul:
    """StageOne Cube 侧矩阵乘计算。"""

    def __init__(self, *, head_dim: int):
        self.head_dim = head_dim
        # Neumann packed64 求逆工作区。
        self.tile_power_l1 = RingBuffer(MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1).next_write()
        self.packed_inv64_l1 = RingBuffer(MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1).next_write()
        self.tile_inv_l1 = RingBuffer(MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1).next_write()
        self.scratch_l1 = RingBuffer(MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1).next_write()
        self.neumann_power_handoff_l1 = RingBuffer(MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1).next_write()
        self.packed_identity_l1 = RingBuffer(MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1).next_write()
        self.zero_l1 = Buffer(MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)))
        self.resident_inv_l1 = RingBuffer(MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1).next_write()
        # L0 — double-buffer + Neumann16
        self.l0a_db = RingBuffer(MemLoc.L0A, dtypes.bfloat16, make_layout((CHUNK_SIZE, head_dim)), depth=2)
        self.l0b_db = RingBuffer(MemLoc.L0B, dtypes.bfloat16, make_layout((CHUNK_SIZE, head_dim)), depth=2)
        self.l0c_db = RingBuffer(MemLoc.L0C, dtypes.float32, make_layout((CHUNK_SIZE, head_dim)), depth=2)
        self.per_d_left_k_l0a = RingBuffer(MemLoc.L0A, dtypes.bfloat16, make_layout((NEUMANN_BLOCK_SIZE, head_dim)), depth=1).next_write()
        self.per_d_left_q_l0a = RingBuffer(MemLoc.L0A, dtypes.bfloat16, make_layout((NEUMANN_BLOCK_SIZE, head_dim)), depth=1).next_write()
        self.per_d_l0b_db = RingBuffer(MemLoc.L0B, dtypes.bfloat16, make_layout((NEUMANN_BLOCK_SIZE, head_dim)), depth=2)
        self.raw_l0a_db = RingBuffer(MemLoc.L0A, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=2)
        self.raw_l0b_db = RingBuffer(MemLoc.L0B, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=2)
        self.raw_l0c_main = RingBuffer(MemLoc.L0C, dtypes.float32, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1).next_write()
        self.raw_l0c_aux = RingBuffer(MemLoc.L0C, dtypes.float32, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1).next_write()
        # 引擎

    def packed64_block(self, l1_matrix, row_block: int, col_block: int):
        # l1_matrix 是 64x64 NZ compact 矩阵。逻辑 16x16 子块按 packed64 block offset 定位。
        return l1_matrix, row_block, col_block

    @staticmethod
    def l0_matrix_block_offset_elems(row_block: int, col_block: int):
        return (row_block * NEUMANN_DIAG_BLOCK_NUM + col_block) * NEUMANN_FRACTAL_ELEMS

    @staticmethod
    def c64_diag_nz_offset_elems(diag_block: int):
        return diag_block * (NEUMANN_DIAG_BLOCK_NUM + 1) * NEUMANN_FRACTAL_ELEMS

    @staticmethod
    def zero_half_l1(l1):
        cube_fill_l1_zero(l1, repeat=256, blk_num=1, dst_gap=0)

    @staticmethod
    def _packed64_block_offset_elems(row_block: int, col_block: int):
        return col_block * CHUNK_SIZE * NEUMANN_BLOCK_SIZE + row_block * NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE

    def load_k_decayed(self, k_decayed_l1):
        mem_copy(self.l0a_db.next_write(), k_decayed_l1)

    def load_q_decayed(self, q_decayed_l1):
        mem_copy(self.l0a_db.next_write(), q_decayed_l1)

    def load_k_inv(self, k_inv_l1):
        mem_copy(self.l0b_db.next_write(), k_inv_l1)

    def mm_kkt(self):
        """KKT = K_decayed @ K_inv^T；K_inv 的 L0B slot 留给 Mqk 复用。"""
        right = self.l0b_db.next_read()
        matmul(self.l0c_db.next_write(), self.l0a_db.next_read(), right, init=True)
        return right

    def mm_mqk(self, k_inv_l0b):
        """Mqk = Q_decayed @ K_inv^T，消费 mm_kkt 保留的 K_inv slot。"""
        matmul(self.l0c_db.next_write(), self.l0a_db.next_read(), k_inv_l0b, init=True)

    def load_per_d_left_k(self, l1_channel):
        mem_copy(self.per_d_left_k_l0a, l1_channel.next_read())

    def load_per_d_left_q(self, l1_channel):
        mem_copy(self.per_d_left_q_l0a, l1_channel.next_read())

    def load_per_d_right_k(self, l1_channel):
        mem_copy(self.per_d_l0b_db.next_write(), l1_channel.next_read())
        return self.per_d_l0b_db.next_read()

    def mm_per_d_kkt(self, row_block: int, col_block: int, right_k):
        kkt_tile = tile_slice(self.raw_l0c_main, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), (row_block, col_block))
        matmul(reinterpret(kkt_tile, layout=make_layout((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)), data_format="nz"), self.per_d_left_k_l0a, right_k, init=True)

    def mm_per_d_mqk(self, row_block: int, col_block: int, right_k):
        mqk_tile = tile_slice(self.raw_l0c_aux, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), (row_block, col_block))
        matmul(reinterpret(mqk_tile, layout=make_layout((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)), data_format="nz"), self.per_d_left_q_l0a, right_k, init=True)

    def finish_per_d_matrices(self, ub_kkt, ub_mqk):
        mem_copy(ub_kkt, self.raw_l0c_main.partition(dim=0))
        mem_copy(ub_mqk, self.raw_l0c_aux.partition(dim=0))

    def store_l0c_to_gm(self, gm_tile):
        mem_copy(gm_tile, self.l0c_db.next_read()[:gm_tile.shape[0], :gm_tile.shape[1]])

    def store_l0c_to_ub(self, ub_buffer, *, shape=(CHUNK_SIZE, CHUNK_SIZE), ub_shape=None):
        ub_shape = ub_shape or shape
        mem_copy(ub_buffer.flatten()[0:ub_shape[0] * ub_shape[1],].reshape(ub_shape), self.l0c_db[0:shape[0], 0:shape[1]].partition(dim=0))

    def c64_odd_even_pair_dst_gap_fractals(self):
        return (self.c64_diag_nz_offset_elems(2) // NEUMANN_FRACTAL_ELEMS) - 1

    def zero_packed64_l0_operands(self, l0a_slot, l0b_slot):
        mem_copy(l0a_slot, self.zero_l1)
        mem_copy(l0b_slot, self.zero_l1)

    def load_packed64_diag_only_operands_to_l0(
        self, left_l1, right_l1, l0a_slot, l0b_slot, *, clear_l0: bool = True,):
        if clear_l0:
            self.zero_packed64_l0_operands(l0a_slot, l0b_slot)
        mem_copy(
            tile_slice_multi(
                l0a_slot, (16, 16),
                repeats=NEUMANN_DIAG_BLOCK_NUM, step=(16, 16),
            ),
            tile_slice_multi(
                left_l1, (16, 16),
                repeats=NEUMANN_DIAG_BLOCK_NUM, step=(16, 16),
            ),
        )
        mem_copy(
            tile_slice_multi(
                l0b_slot, (16, 16),
                repeats=NEUMANN_DIAG_BLOCK_NUM, step=(16, 16),
            ),
            tile_slice_multi(
                right_l1.permute(1, 0), (16, 16),
                repeats=NEUMANN_DIAG_BLOCK_NUM, step=(16, 16),
            ),
        )

    def issue_packed64_diag_only_operands_to_l0_db(
        self,
        left_l1,
        right_l1,
        *,
        clear_l0: bool = True,
    ):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        self.load_packed64_diag_only_operands_to_l0(left_l1, right_l1, l0a_write, l0b_write, clear_l0=clear_l0)

    def begin_mmad_raw_l0ab_db_packed64(
        self,
        c_tile,
        *,
        init_c: bool,
        unit_flag: int,
    ):
        """发起 full64 raw MMAD，显式指定 init/accumulate。"""
        l0a_read = self.raw_l0a_db.next_read()
        l0b_read = self.raw_l0b_db.next_read()
        matmul(c_tile, l0a_read, l0b_read, init=init_c, unit_flag=unit_flag)

    def raw_mmad_packed64_diag_only(
        self, left_l1, right_l1, c_tile, *, init_c: bool, unit_flag: int, clear_l0: bool = True
    ):
        self.issue_packed64_diag_only_operands_to_l0_db(left_l1, right_l1, clear_l0=clear_l0)
        self.begin_mmad_raw_l0ab_db_packed64(c_tile, init_c=init_c, unit_flag=unit_flag)

    def raw_mmad_packed64_diag_only_db_pair(
        self,
        left_l10,
        right_l10,
        left_l11,
        right_l11,
        c_tile,
        *,
        init_c0: bool,
        unit_flag0: int,
        init_c1: bool,
        unit_flag1: int,
        clear_l0: bool = True,
    ):
        self.issue_packed64_diag_only_operands_to_l0_db(left_l10, right_l10, clear_l0=clear_l0)
        self.issue_packed64_diag_only_operands_to_l0_db(left_l11, right_l11, clear_l0=clear_l0)
        self.begin_mmad_raw_l0ab_db_packed64(c_tile, init_c=init_c0, unit_flag=unit_flag0)
        self.begin_mmad_raw_l0ab_db_packed64(c_tile, init_c=init_c1, unit_flag=unit_flag1)

    def fixpipe_packed64_full_to_half_l1(
        self,
        dst_channel,
        c_tile,
    ):
        mem_copy(dst_channel, c_tile, options=copy_options(unit_flag=3))

    def fixpipe_packed64_full_to_bf16_l1(
        self,
        dst_channel,
        c_tile,
    ):
        mem_copy(dst_channel, c_tile, options=copy_options(unit_flag=3))

    def half_l1_power_packed64_diag_only_on_c(
        self,
        left_l1,
        right_l1,
        dst_channel,
        *,
        clear_l0: bool,
    ):
        c_power = self.raw_l0c_aux
        self.raw_mmad_packed64_diag_only(left_l1, right_l1, c_power, init_c=True, unit_flag=3, clear_l0=clear_l0)
        c_power_read = self.raw_l0c_aux
        self.fixpipe_packed64_full_to_half_l1(dst_channel, c_power_read)

    def prepare_packed64_identity_l1(self, gm_identity16):
        """构造 packed 64x64 identity。"""
        identity_write = self.packed_identity_l1
        self.zero_half_l1(identity_write)
        cube_sync_pipe(PIPE.MTE2)
        mem_copy(
            tile_slice_multi(
                identity_write, (16, 16),
                repeats=NEUMANN_DIAG_BLOCK_NUM, step=(16, 16),
            ),
            tile_slice_multi(
                gm_identity16, (16, 16),
                repeats=NEUMANN_DIAG_BLOCK_NUM, step=(0, 0),
            ),
        )
        return self.packed_identity_l1

    def issue_negative_t_identity_operands(self, negative_t, identity):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        mem_copy(l0a_write, negative_t)
        mem_copy(l0b_write, identity)

    def issue_identity_identity_operands(self, identity):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        mem_copy(l0a_write, identity)
        mem_copy(l0b_write, identity)

    def prepare_inv_init_from_negative_t_cube(self, negative_t, identity):
        """Cube 侧在 L0C 构造 scratch_l1 = (-T) @ I + I @ I。"""
        self.issue_negative_t_identity_operands(negative_t, identity)
        self.issue_identity_identity_operands(identity)
        c_inv_init = self.raw_l0c_main
        self.begin_mmad_raw_l0ab_db_packed64(c_inv_init, init_c=True, unit_flag=2)
        self.begin_mmad_raw_l0ab_db_packed64(c_inv_init, init_c=False, unit_flag=3)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.scratch_l1, c_inv_read)
        return self.scratch_l1

    def load_packed64_diagonal_inputs_from_l1_handoff(
        self,
        power_handoff_l1,
        identity,
    ):
        """消费 AIV `-T` handoff，并在 Cube 侧构造 `I + (-T)`。"""
        self.zero_half_l1(self.zero_l1)
        negative_t = power_handoff_l1
        scratch = self.prepare_inv_init_from_negative_t_cube(negative_t, identity)
        return negative_t, identity, scratch

    def neumann_diag_power2_update(self, negative_t, identity, scratch):
        self.half_l1_power_packed64_diag_only_on_c(negative_t, negative_t, self.tile_inv_l1, clear_l0=True)
        power2 = self.tile_inv_l1

        c_inv = self.raw_l0c_main
        self.raw_mmad_packed64_diag_only_db_pair(
            scratch,
            identity,
            scratch,
            power2,
            c_inv,
            init_c0=True,
            unit_flag0=2,
            init_c1=False,
            unit_flag1=3,
            clear_l0=False,
        )
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return power2, self.packed_inv64_l1

    def neumann_diag_power4_update(self, power2, packed_inv):
        self.half_l1_power_packed64_diag_only_on_c(power2, power2, self.tile_power_l1, clear_l0=False)
        power4 = self.tile_power_l1

        c_inv = self.raw_l0c_main
        self.raw_mmad_packed64_diag_only(packed_inv, power4, c_inv, init_c=False, unit_flag=3, clear_l0=False)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return power4, self.packed_inv64_l1

    def neumann_diag_power8_update(self, power4, packed_inv):
        self.half_l1_power_packed64_diag_only_on_c(power4, power4, self.tile_inv_l1, clear_l0=False)
        power8 = self.tile_inv_l1

        c_inv = self.raw_l0c_main
        self.raw_mmad_packed64_diag_only(packed_inv, power8, c_inv, init_c=False, unit_flag=3, clear_l0=False)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return self.packed_inv64_l1

    def issue_odd_lower_even_inv_full64(self, scratch, packed_inv):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        self._load_packed16_pair_to_full64_l0a_with_gap(
            scratch,
            l0a_write,
            src0_row_block=1,
            src0_col_block=0,
            src1_row_block=3,
            src1_col_block=2,
            dst_offset_elems=NEUMANN_DIAG_BLOCK_NUM * NEUMANN_FRACTAL_ELEMS,
            dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals(),
        )
        self._load_packed16_pair_to_full64_l0b_trans_with_gap(
            packed_inv,
            l0b_write,
            src0_row_block=0,
            src0_col_block=0,
            src1_row_block=2,
            src1_col_block=2,
            dst_offset_elems=self.c64_diag_nz_offset_elems(0),
            dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals(),
        )

    def issue_odd_inv_odd_tmp_full64(
        self,
        packed_inv,
        tmp64,
    ):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        self._load_l1_block_to_l0a_block(self.packed64_block(packed_inv, 1, 1), l0a_write, 1, 1)
        self._load_l1_block_to_l0a_block(self.packed64_block(packed_inv, 3, 3), l0a_write, 3, 3)
        self._load_packed16_pair_to_full64_l0b_trans_with_gap(
            tmp64,
            l0b_write,
            src0_row_block=1,
            src0_col_block=0,
            src1_row_block=3,
            src1_col_block=2,
            dst_offset_elems=NEUMANN_DIAG_BLOCK_NUM * NEUMANN_FRACTAL_ELEMS,
            dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals(),
        )

    def compose_odd_even_lower16_to32_accum_full64(
        self,
        scratch,
        packed_inv,
    ):
        """组合两个 32x32 对角子块内部的 lower 逆矩。"""
        self.issue_odd_lower_even_inv_full64(scratch, packed_inv)
        c_tmp = self.raw_l0c_aux
        self.begin_mmad_raw_l0ab_db_packed64(c_tmp, init_c=True, unit_flag=3)
        c_tmp_read = self.raw_l0c_aux
        self.fixpipe_packed64_full_to_half_l1(self.tile_inv_l1, c_tmp_read)
        tmp64 = self.tile_inv_l1

        self.issue_odd_inv_odd_tmp_full64(packed_inv, tmp64)
        c_inv = self.raw_l0c_main
        self.begin_mmad_raw_l0ab_db_packed64(c_inv, init_c=False, unit_flag=3)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return self.packed_inv64_l1

    def issue_odd32_lower_even32_inv_full64(self, scratch, packed_inv):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        mem_copy(l0a_write, scratch)
        self._load_packed32_block_to_full64_l0b_raw(
            packed_inv,
            l0b_write,
            src_row=0,
            src_col=0,
            dst_row=0,
            dst_col=0,
            src_packed_size=CHUNK_SIZE,
            transpose=True,
        )

    def prepare_odd32_inv_odd32_tmp_full64(self, packed_inv):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        mem_copy(l0a_write, packed_inv)
        return l0a_write, l0b_write

    def complete_odd32_inv_odd32_tmp_full64(
        self,
        tmp64,
        l0a_write,
        l0b_write,
    ):
        self._load_packed32_block_to_full64_l0b_raw(
            tmp64,
            l0b_write,
            src_row=HALF_CHUNK_SIZE,
            src_col=0,
            dst_row=HALF_CHUNK_SIZE,
            dst_col=0,
            src_packed_size=CHUNK_SIZE,
            transpose=True,
        )

    def compose_odd_even_lower32_to64_accum_full64(
        self,
        scratch,
        packed_inv,
    ):
        """组合 64x64 左下 32x32 lower 逆矩并生成 BF16 resident inverse。"""
        self.issue_odd32_lower_even32_inv_full64(scratch, packed_inv)
        second_l0a, second_l0b = self.prepare_odd32_inv_odd32_tmp_full64(packed_inv)
        c_tmp = self.raw_l0c_aux
        self.begin_mmad_raw_l0ab_db_packed64(c_tmp, init_c=True, unit_flag=3)
        c_tmp_read = self.raw_l0c_aux
        self.fixpipe_packed64_full_to_half_l1(self.tile_inv_l1, c_tmp_read)
        tmp64 = self.tile_inv_l1

        self.complete_odd32_inv_odd32_tmp_full64(tmp64, second_l0a, second_l0b)
        c_inv = self.raw_l0c_main
        self.begin_mmad_raw_l0ab_db_packed64(c_inv, init_c=False, unit_flag=3)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_bf16_l1(self.resident_inv_l1, c_inv_read)

    def compute_u_pre_and_w_from_resident_inv_beta(
        self,
        u_chunk,
        w_chunk,
        v_beta_l1,
        k_decayed_beta_l1,
    ):
        # β·K 先产出，故先算 W；W 的 MMAD/store 与后续 β·V handoff 重叠。
        l0a_resident = self._load_resident_inv_to_l0a()

        l0b_beta = self._load_beta_to_l0b(k_decayed_beta_l1)
        self._mm_resident_inv_beta(l0a_resident, l0b_beta)
        self.store_l0c_to_gm(w_chunk)

        l0b_beta = self._load_beta_to_l0b(v_beta_l1)
        self._mm_resident_inv_beta(l0a_resident, l0b_beta)
        self.store_l0c_to_gm(u_chunk)

    def _raw_l0a_block(self, l0a_slot, row_block: int, col_block: int):
        return reinterpret(l0a_slot, layout=make_layout((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)), offset=self.l0_matrix_block_offset_elems(row_block, col_block) * 2, physical_layout=make_layout(((NEUMANN_BLOCK_SIZE + 15) // 16, (NEUMANN_BLOCK_SIZE + 15) // 16, 16, 16), stride=l0a_slot.physical_layout.stride))

    def _raw_l0b_block(self, l0b_slot, row_block: int, col_block: int):
        return reinterpret(l0b_slot, layout=make_layout((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)), offset=self.l0_matrix_block_offset_elems(row_block, col_block) * 2, physical_layout=make_layout(((NEUMANN_BLOCK_SIZE + 15) // 16, (NEUMANN_BLOCK_SIZE + 15) // 16, 16, 16), stride=l0b_slot.physical_layout.stride))

    def _load_l1_block_to_l0a_block(
        self, l1_block, l0a_slot, row_block: int, col_block: int,):
        """把一个 packed L1 16x16 fractal 装入 full64 L0A 的指定块位置。"""
        l1_parent, _, _ = l1_block
        l0_tile = self._raw_l0a_block(l0a_slot, row_block, col_block)
        l1_tile = tile_slice(l1_parent, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), (row_block, col_block))
        mem_copy(l0_tile, l1_tile)

    def _load_l1_block_to_l0b_block(
        self, l1_block, l0b_slot, row_block: int, col_block: int,):
        """把一个 packed L1 16x16 fractal 转置装入 full64 L0B 的指定块位置。"""
        l1_parent, _, _ = l1_block
        l0_tile = self._raw_l0b_block(l0b_slot, row_block, col_block)
        l1_tile = tile_slice(l1_parent, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), (row_block, col_block))
        mem_copy(l0_tile, (l1_tile).permute(1, 0))

    def _load_packed32_block_to_full64_l0b_raw(
        self, l1_matrix, l0b_slot,
        *,
        src_row: int, src_col: int, dst_row: int, dst_col: int,
        src_packed_size: int = CHUNK_SIZE, transpose: bool = True):
        for row_offset in (0, NEUMANN_BLOCK_SIZE):
            l1_offset = (src_col // NEUMANN_BLOCK_SIZE) * src_packed_size * NEUMANN_BLOCK_SIZE
            l1_offset += (src_row + row_offset) * NEUMANN_BLOCK_SIZE + (src_col % NEUMANN_BLOCK_SIZE)
            l0_offset = self.l0_matrix_block_offset_elems((dst_row + row_offset) // NEUMANN_BLOCK_SIZE, dst_col // NEUMANN_BLOCK_SIZE)
            # A transpose view has shape [32, 16]; the L0 destination must match.
            l0_shape = (HALF_CHUNK_SIZE, NEUMANN_BLOCK_SIZE) if transpose else (NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE)
            l0_tile = reinterpret(l0b_slot, layout=make_layout(l0_shape), offset=l0_offset * 2, physical_layout=make_layout(((l0_shape[1] + 15) // 16, (l0_shape[0] + 15) // 16, 16, 16), stride=l0b_slot.physical_layout.stride))
            l1_tile = reinterpret(l1_matrix, layout=make_layout((NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE)), offset=l1_offset * 2, physical_layout=make_layout(((HALF_CHUNK_SIZE + 15) // 16, (NEUMANN_BLOCK_SIZE + 15) // 16, 16, 16), stride=l1_matrix.physical_layout.stride))
            mem_copy(l0_tile, (l1_tile).permute(1, 0) if transpose else l1_tile)

    def _load_packed32_block_to_full64_l0a_raw(
        self, l1_matrix, l0a_slot,
        *,
        src_row: int, src_col: int, dst_row: int, dst_col: int,
        src_packed_size: int = CHUNK_SIZE, transpose: bool = False):
        """把 packed32 L1 block 装入 full64 L0A。"""
        for row_offset in (0, NEUMANN_BLOCK_SIZE):
            l1_offset = (src_col // NEUMANN_BLOCK_SIZE) * src_packed_size * NEUMANN_BLOCK_SIZE
            l1_offset += (src_row + row_offset) * NEUMANN_BLOCK_SIZE + (src_col % NEUMANN_BLOCK_SIZE)
            l0_offset = self.l0_matrix_block_offset_elems((dst_row + row_offset) // NEUMANN_BLOCK_SIZE, dst_col // NEUMANN_BLOCK_SIZE)
            l0_offset -= 6 * NEUMANN_FRACTAL_ELEMS
            # A transpose view has shape [32, 16]; the L0 destination must match.
            l0_shape = (HALF_CHUNK_SIZE, NEUMANN_BLOCK_SIZE) if transpose else (NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE)
            l0_tile = reinterpret(l0a_slot, layout=make_layout(l0_shape), offset=l0_offset * 2, physical_layout=make_layout(((l0_shape[1] + 15) // 16, (l0_shape[0] + 15) // 16, 16, 16), stride=l0a_slot.physical_layout.stride))
            l1_tile = reinterpret(l1_matrix, layout=make_layout((NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE)), offset=l1_offset * 2, physical_layout=make_layout(((HALF_CHUNK_SIZE + 15) // 16, (NEUMANN_BLOCK_SIZE + 15) // 16, 16, 16), stride=l1_matrix.physical_layout.stride))
            mem_copy(l0_tile, (l1_tile).permute(1, 0) if transpose else l1_tile)

    def _load_packed16_pair_to_full64_l0a_with_gap(
        self, l1_matrix, l0a_slot,
        *,
        src0_row_block: int, src0_col_block: int, src1_row_block: int, src1_col_block: int,
        dst_offset_elems: int, dst_gap_fractals: int,):
        """把一对 packed16 L1 block 装入 full64 L0A，目标块之间保留 gap。"""
        dst0_block = dst_offset_elems // NEUMANN_FRACTAL_ELEMS
        dst1_block = dst0_block + dst_gap_fractals + 1
        for src_row_block, src_col_block, dst_block in (
            (src0_row_block, src0_col_block, dst0_block),
            (src1_row_block, src1_col_block, dst1_block),):
            # 3510 L0A 的 16x16 单块地址仍需补偿一个 full64 M span。
            dst_offset = (dst_block - (NEUMANN_DIAG_BLOCK_NUM - 1)) * NEUMANN_FRACTAL_ELEMS
            l0_tile = reinterpret(l0a_slot, layout=make_layout((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)), offset=dst_offset * 2, physical_layout=make_layout(((NEUMANN_BLOCK_SIZE + 15) // 16, (NEUMANN_BLOCK_SIZE + 15) // 16, 16, 16), stride=l0a_slot.physical_layout.stride))
            l1_tile = tile_slice(l1_matrix, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), (src_row_block, src_col_block))
            mem_copy(l0_tile, l1_tile)

    def _load_packed16_pair_to_full64_l0b_trans_with_gap(
        self,
        l1_matrix,
        l0b_slot,
        *,
        src0_row_block: int,
        src0_col_block: int,
        src1_row_block: int,
        src1_col_block: int,
        dst_offset_elems: int,
        dst_gap_fractals: int,
    ):
        """把一对 packed16 L1 block 转置装入 full64 L0B，目标块之间保留 gap。"""
        dst0_block = dst_offset_elems // NEUMANN_FRACTAL_ELEMS
        dst1_block = dst0_block + dst_gap_fractals + 1
        self._load_l1_block_to_l0b_block(
            self.packed64_block(l1_matrix, src0_row_block, src0_col_block),
            l0b_slot,
            dst0_block // NEUMANN_DIAG_BLOCK_NUM,
            dst0_block % NEUMANN_DIAG_BLOCK_NUM,
        )
        self._load_l1_block_to_l0b_block(
            self.packed64_block(l1_matrix, src1_row_block, src1_col_block),
            l0b_slot,
            dst1_block // NEUMANN_DIAG_BLOCK_NUM,
            dst1_block % NEUMANN_DIAG_BLOCK_NUM,
        )

    def _load_resident_inv_to_l0a(self):
        resident_inv = self.resident_inv_l1
        l0a_write = self.l0a_db.next_write()
        mem_copy(l0a_write[0:CHUNK_SIZE, 0:CHUNK_SIZE], resident_inv)
        return self.l0a_db.next_read()

    def _load_beta_to_l0b(self, beta_channel):
        beta_read = beta_channel
        l0b_slot = self.l0b_db.next_write(bare=True)
        l0b_beta = l0b_slot.view(make_layout((beta_read.shape[1], beta_read.shape[0])), dtype=beta_read.dtype, data_format="nz")
        mem_copy(l0b_beta, beta_read.permute(1, 0))
        self.l0b_db.next_read()
        return l0b_beta

    def _mm_resident_inv_beta(self, l0a_resident, l0b_beta):
        l0c_write = self.l0c_db.next_write()
        matmul(l0c_write, l0a_resident[0:CHUNK_SIZE, 0:CHUNK_SIZE], l0b_beta, init=True)


class StageOneVector:
    """Vector 侧标量、mask、搬运和状态更新封装。"""

    def __init__(self, *, head_dim: int, scale_value, subblock_idx):
        self.head_dim = head_dim
        self.scale_value = scale_value
        self.subblock_idx = subblock_idx
        self.ub_q = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)), depth=1).next_write()
        self.ub_k = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)), depth=1).next_write()
        self.ub_qk_exchange = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, D_HALF)))
        self.ub_g_raw = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, D_HALF)), depth=1).next_write()
        self.ub_gate_activated = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, D_HALF)), depth=1).next_write()
        self.ub_dt_bias = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((D_HALF,)), depth=1).next_write()
        self.ub_alpha = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((1,)), depth=1).next_write()
        self.ub_beta_raw = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, 1)), depth=1).next_write()
        self.ub_beta = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, 1)))
        self.ub_q_bf16 = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, D_HALF)), depth=1, data_format='nd').next_write()
        self.ub_k_bf16 = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, D_HALF)), depth=1, data_format='nd').next_write()
        self.ub_v_raw = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, D_HALF)), depth=1).next_write()
        self.ub_gamma = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, D_HALF)))
        self.ub_gamma_c = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((1, D_HALF)), depth=2)
        self.ub_cs_snapshot = [Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, D_HALF))) for _ in range(2)]
        self.ub_mqk = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((HALF_CHUNK_SIZE, CHUNK_SIZE)), depth=2)
        self.ub_bf16_nz = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, D_HALF)), depth=2, data_format='nz', physical_layout=make_layout(((D_HALF + 15) // 16, (CHUNK_SIZE + 15) // 16, 16, 16), stride=((CHUNK_SIZE + 15) // 16 * 256 + 16, 256, 16, 1)))
        self.ub_cast_bf16 = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, D_HALF)), depth=1, data_format='nd').next_write()
        self.ub_kkt_fp16 = Buffer(MemLoc.UB, dtypes.float16, make_layout((HALF_CHUNK_SIZE, CHUNK_SIZE)))
        self.ub_mqk_bf16 = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((HALF_CHUNK_SIZE, CHUNK_SIZE)), depth=1, data_format='nd').next_write()
        self.ub_g_cumsum = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, D_HALF)), depth=1).next_write()
        self.ub_raw_k = [Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, D_HALF))) for _ in range(2)]
        self.ub_kkt = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((HALF_CHUNK_SIZE, CHUNK_SIZE)), depth=1).next_write()
        self.ub_kkt_nz = RingBuffer(MemLoc.UB, dtypes.float16, make_layout((HALF_CHUNK_SIZE, CHUNK_SIZE)), depth=1, data_format='nz', physical_layout=make_layout(((CHUNK_SIZE + 15) // 16, (HALF_CHUNK_SIZE + 15) // 16, 16, 16), stride=((HALF_CHUNK_SIZE + 15) // 16 * 256 + 16, 256, 16, 1))).next_write()
        self.ub_per_d_nd = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((NEUMANN_BLOCK_SIZE, D_HALF)), depth=2)
        self.ub_per_d_nz = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((NEUMANN_BLOCK_SIZE, D_HALF)), depth=2, data_format='nz', physical_layout=make_layout(((D_HALF + 15) // 16, (NEUMANN_BLOCK_SIZE + 15) // 16, 16, 16), stride=((NEUMANN_BLOCK_SIZE + 15) // 16 * 256 + 16, 256, 16, 1)))
        # One FIFO carries target-scoped left K/Q and column-scoped right K.
        # Its three original L1 slots let AIV prepare left Q while AIC consumes
        # the first right K; Cube keeps left K/Q resident in distinct L0A roots.
        self.l1_per_d_operand = RingBuffer(MemLoc.L1, dtypes.bfloat16, make_layout((NEUMANN_BLOCK_SIZE, self.head_dim)), depth=3)
        self.l1_v_beta = RingBuffer(MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, self.head_dim)), depth=1).next_write()
        self.l1_K_decayed_beta_for_W = RingBuffer(MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, self.head_dim)), depth=1).next_write()

    @staticmethod
    @jit
    def publish_normalized_qk_half(q_to_k, outgoing_half):
        """Publish this AIV's normalized Q/K half before unrelated V work."""
        vec_sync_notify(PIPE.V, PIPE.MTE3, QK_V_TO_MTE3_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.MTE3, QK_V_TO_MTE3_EVENT_ID)
        mem_copy(q_to_k.flatten(), outgoing_half.flatten())
        vec_sync_all()
        vec_sync_block_arrive(PIPE.MTE3, QK_PUBLISH_FLAG_ID, mode=1)

    @staticmethod
    @jit
    def consume_normalized_qk_half(k_to_q, incoming_half):
        """Consume the peer half after independent gate activation has run."""
        vec_sync_block_wait(PIPE.MTE2, QK_PUBLISH_FLAG_ID, mode=1)
        mem_copy(incoming_half.flatten(), k_to_q.flatten())
        vec_sync_all()
        vec_sync_block_arrive(PIPE.MTE2, QK_CONSUME_FLAG_ID, mode=1)
        vec_sync_block_wait(PIPE.MTE3, QK_CONSUME_FLAG_ID, mode=1)
        vec_sync_notify(PIPE.MTE2, PIPE.V, QK_MTE2_TO_V_EVENT_ID)
        vec_sync_wait(PIPE.MTE2, PIPE.V, QK_MTE2_TO_V_EVENT_ID)

    @staticmethod
    @jit
    def normalize_full_d_bf16(tile):
        """Normalize each bf16 [128] row in place using fp32 arithmetic."""
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            one = vdups(1.0, dtypes.float32, mask=lane_mask)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * SUPPORTED_HEAD_DIM
                first_packed = vload_unpack(tile, offset, unpack_mode=UnpackMode.B16_TO_B32)
                second_packed = vload_unpack(tile, offset + D_HALF, unpack_mode=UnpackMode.B16_TO_B32)
                first = vcast(first_packed, dtypes.float32, mask=lane_mask)
                second = vcast(second_packed, dtypes.float32, mask=lane_mask)
                first_square = vmul(first, first, mask=lane_mask)
                second_square = vmul(second, second, mask=lane_mask)
                square_sum = vadd(first_square, second_square, mask=lane_mask)
                sum_squares = vreduce_sum(square_sum, mask=lane_mask)
                sum_vector = vdup(sum_squares, mask=lane_mask)
                stabilized_sum = vadds(sum_vector, 1.0e-6, mask=lane_mask)
                norm = vsqrt(stabilized_sum, mask=lane_mask)
                inverse_norm = vdiv(one, norm, mask=lane_mask)
                normalized_first = vmul(first, inverse_norm, mask=lane_mask)
                normalized_second = vmul(second, inverse_norm, mask=lane_mask)
                first_bf16 = vcast(normalized_first, dtypes.bfloat16, mask=lane_mask)
                second_bf16 = vcast(normalized_second, dtypes.bfloat16, mask=lane_mask)
                vstore_pack(tile, offset, first_bf16, lane_mask, pack_mode=PackMode.B32_TO_B16)
                vstore_pack(tile, offset + D_HALF, second_bf16, lane_mask, pack_mode=PackMode.B32_TO_B16)
            vmem_bar("vst_vld")

    @staticmethod
    def save_gamma_c_row(g_slot, gamma_c_write):
        src_pre = (CHUNK_SIZE - 1) * D_HALF  # cumsum 最后一行元素起点
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            gamma_c = vload(g_slot, src_pre)
            vstore(gamma_c_write, 0, gamma_c, row_mask)

    @staticmethod
    def _copy_bf16_gm_half_tile_to_ub(gm_tile, ub_channel, *, cols: int = SUPPORTED_HEAD_DIM):
        ub_write = ub_channel
        mem_copy(ub_write, gm_tile)

    @staticmethod
    @jit
    def _fill_bf16_invalid_rows(ub_tile, *, valid_rows, cols: int, value: float):
        with vf(mode="simd"):
            lane_mask, _ = update_mask(cols, elem_bits=16)
            fill_bf16 = vdups(value, dtypes.bfloat16, mask=lane_mask)
            for row in dsl_range(valid_rows, CHUNK_SIZE, 1, unroll=4):
                vstore(ub_tile, row * cols, fill_bf16, lane_mask)
            vmem_bar("vst_vld")

    @staticmethod
    def _cast_bf16_ub_tile_to_f32(
        ub_cast_bf16, ub_f32_tile, *, rows: int = HALF_CHUNK_SIZE, cols: int = SUPPORTED_HEAD_DIM
    ):
        bf16_valid = reinterpret(ub_cast_bf16, offset=0, layout=make_layout((rows, cols)))
        f32_valid = reinterpret(ub_f32_tile, offset=0, layout=make_layout((rows, cols)))
        cast_tile(f32_valid, bf16_valid)

    def scratch_half_tile(self, gm_scratch, block_idx, cols: int):
        return tile_slice(gm_scratch, (HALF_CHUNK_SIZE, cols), (block_idx * 2 + self.subblock_idx, 0))

    def scratch_dhalf_tile(self, gm_scratch, block_idx):
        return tile_slice(gm_scratch, (CHUNK_SIZE, D_HALF), (block_idx, self.subblock_idx))

    @jit
    def prepare_normalized_qk_local_half(self, inputs, gm_qk_exchange, location, block_idx):
        """Normalize full Q/K rows and leave the peer half ready for publication."""
        q_to_k = gm_qk_exchange[block_idx, 0, None, None]
        k_to_q = gm_qk_exchange[block_idx, 1, None, None]
        q_chunk = tile_slice(
            inputs.q[location.batch, location.qk_head, None, None],
            (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
            (location.chunk, 0),
        )
        k_chunk = tile_slice(
            inputs.k[location.batch, location.qk_head, None, None],
            (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
            (location.chunk, 0),
        )

        if self.subblock_idx == 0:
            if location.valid_rows < CHUNK_SIZE:
                q_slot = self._load_tail_bf16_tensor(
                    q_chunk, self.ub_q, cols=SUPPORTED_HEAD_DIM, ready_event=MTE2_TO_V_EVENT_ID
                )
                self._normalize_and_publish_q(q_slot, q_to_k)
            else:
                mem_copy(self.ub_q, q_chunk)
                vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
                self._normalize_and_publish_q(self.ub_q, q_to_k)
        else:
            if location.valid_rows < CHUNK_SIZE:
                k_slot = self._load_tail_bf16_tensor(
                    k_chunk, self.ub_k, cols=SUPPORTED_HEAD_DIM, ready_event=MTE2_TO_V_SUBBLOCK1_EVENT_ID
                )
                self._normalize_and_publish_k(k_slot, k_to_q)
            else:
                mem_copy(self.ub_k, k_chunk)
                vec_sync_all()
                vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_SUBBLOCK1_EVENT_ID)
                self._normalize_and_publish_k(self.ub_k, k_to_q)

        return q_to_k, k_to_q

    @jit
    def finish_normalized_qk_exchange(self, q_to_k, k_to_q):
        """Receive the peer half and return local normalized K/Q halves."""
        if self.subblock_idx == 0:
            self.consume_normalized_qk_half(k_to_q, self.ub_qk_exchange)
            cast_tile(self.ub_k_bf16, self.ub_qk_exchange)
        else:
            self.consume_normalized_qk_half(q_to_k, self.ub_qk_exchange)
            cast_tile(self.ub_q_bf16, self.ub_qk_exchange)
        return self.ub_k_bf16, self.ub_q_bf16

    @jit
    def activate_raw_gate(self, gm_g, gate, location):
        """Apply the source gate formula to one D half, then take its cumsum."""
        g_chunk = tile_slice(
            gm_g[location.batch, location.head, None, None], (CHUNK_SIZE, D_HALF), (location.chunk, self.subblock_idx)
        )
        bias_half = tile_slice(gate.bias[location.head, None], (D_HALF,), (self.subblock_idx,))
        alpha = tile_slice(gate.alpha, (1,), (location.head,))
        mem_copy(self.ub_g_raw, g_chunk)
        mem_copy(self.ub_dt_bias, bias_half)
        mem_copy(self.ub_alpha, alpha)
        vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        rows_to_activate = CHUNK_SIZE
        if location.valid_rows < CHUNK_SIZE:
            with vf(mode="simd"):
                lane_mask, _ = update_mask(D_HALF, elem_bits=32)
                zero = vdups(0.0, dtypes.float32, mask=lane_mask)
                for row in dsl_range(location.valid_rows, CHUNK_SIZE, 1, unroll=4):
                    vstore(self.ub_gate_activated, row * D_HALF, zero, lane_mask)
                vmem_bar("vst_vld")
            rows_to_activate = location.valid_rows
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            bound = vdups(gate.lower_bound, dtypes.float32, mask=lane_mask)
            one = vdups(1.0, dtypes.float32, mask=lane_mask)
            alpha = vload_broadcast(self.ub_alpha, 0)
            alpha_exp = vexp(alpha, mask=lane_mask)
            neg_exp_a = vmuls(alpha_exp, -1.0, mask=lane_mask)
            for row in dsl_range(0, rows_to_activate, 1, unroll=4):
                offset = row * D_HALF
                raw_g_packed = vload_unpack(self.ub_g_raw, offset, unpack_mode=UnpackMode.B16_TO_B32)
                raw_g = vcast(raw_g_packed, dtypes.float32, mask=lane_mask)
                dt_bias = vload(self.ub_dt_bias, 0)
                gate_input = vadd(raw_g, dt_bias, mask=lane_mask)
                gate_logit = vmul(neg_exp_a, gate_input, mask=lane_mask)
                gate_exp = vexp(gate_logit, mask=lane_mask)
                denominator = vadd(one, gate_exp, mask=lane_mask)
                gate = vdiv(bound, denominator, mask=lane_mask)
                vstore(self.ub_gate_activated, offset, gate, lane_mask)
            vmem_bar("vst_vld")
        vec_sync_all()
        prefix_sum_rows(self.ub_g_cumsum, self.ub_gate_activated)
        return self.ub_g_cumsum

    @jit
    def activate_raw_beta(self, gm_beta, location):
        """Apply sigmoid to one raw BF16 beta-logit chunk."""
        beta_chunk = tile_slice(
            gm_beta[location.batch, location.head, None, None], (CHUNK_SIZE, 1), (location.chunk, 0)
        )
        mem_copy(self.ub_beta_raw, beta_chunk)
        vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        cast_tile(self.ub_beta, self.ub_beta_raw)
        with vf(mode="simd"):
            lane_mask, _ = update_mask(VL, elem_bits=32)
            one = vdups(1.0, dtypes.float32, mask=lane_mask)
            raw_beta = vload(self.ub_beta, 0)
            neg_beta = vmuls(raw_beta, -1.0, mask=lane_mask)
            exp_neg_beta = vexp(neg_beta, mask=lane_mask)
            sigmoid_denominator = vadd(one, exp_neg_beta, mask=lane_mask)
            beta = vdiv(one, sigmoid_denominator, mask=lane_mask)
            if location.valid_rows < CHUNK_SIZE:
                zero = vdups(0.0, dtypes.float32, mask=lane_mask)
                valid_mask, _ = update_mask(location.valid_rows, elem_bits=32)
                invalid_mask = mask_xor(lane_mask, valid_mask, exec_mask=lane_mask)
                vstore(self.ub_beta, 0, beta, valid_mask)
                vstore(self.ub_beta, 0, zero, invalid_mask)
            else:
                vstore(self.ub_beta, 0, beta, lane_mask)
        return self.ub_beta

    @jit
    def apply_beta_and_finish_t_for_neumann(self, beta_slot, power_handoff_l1, subblock_base: int, buffer_parity: int):
        row_base = subblock_base * HALF_CHUNK_SIZE  # 本半块全局行起点(0 或 32)
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            zero_reg = vdups(0.0, dtypes.float32, mask=row_mask)
            for row in dsl_range(0, HALF_CHUNK_SIZE, 1, unroll=4):
                row_off = row * CHUNK_SIZE
                t_true = vload(self.ub_kkt, row_off)
                beta_brc = vload_broadcast(beta_slot, row_base + row)  # β[row] 广播到整行(逐行标量)
                t_beta = vmul(t_true, beta_brc, mask=row_mask)  # T[row,:] *= β[row]
                # 严格下三角：全局行 i 保留前 i 列(对角也清)；update_mask(i) 的前 i 个 lane = 保留谓词。
                tril_mask, _ = update_mask(row_base + row, elem_bits=32)
                t_tril = vselect_raw(t_beta, zero_reg, cond_mask=tril_mask)  # lane<i 取 t_beta，其余取 0
                neg_t = vmuls(t_tril, -1.0, mask=row_mask)  # -T
                neg_t_fp16 = vcast(neg_t, dtypes.float16, mask=row_mask)  # f32 -> fp16
                vstore_pack(self.ub_kkt_fp16, row_off, neg_t_fp16, row_mask, pack_mode="b32_to_b16")
        # 手写 ND→NZ VF；同 V pipe 程序序保持 RaW。
        nd2nz_tile(self.ub_kkt_nz, self.ub_kkt_fp16)
        self._store_neumann_tmp_to_handoff_l1(power_handoff_l1, subblock_base)

    @jit
    def prepare_beta_weighted_c3_operands(self, gm_V, beta_slot, location):
        """Construct the two C3 operands from one full S by D-half vector tile."""
        raw_k_snapshot = self.ub_raw_k[location.parity]
        v_half = tile_slice(
            gm_V[location.batch, location.head, None, None], (CHUNK_SIZE, D_HALF), (location.chunk, self.subblock_idx)
        )
        if location.valid_rows < CHUNK_SIZE:
            v_slot = self._load_tail_bf16_tensor(
                v_half, self.ub_v_raw, cols=D_HALF, ready_event=MTE2_TO_V_V_STAGING_EVENT_ID
            )
            self._finish_beta_weighted_c3_operands(v_slot, beta_slot, raw_k_snapshot)
        else:
            self._copy_bf16_gm_half_tile_to_ub(v_half, self.ub_v_raw, cols=D_HALF)
            vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_V_STAGING_EVENT_ID)
            self._finish_beta_weighted_c3_operands(self.ub_v_raw, beta_slot, raw_k_snapshot)

    @jit
    def prepare_normalized_chunk_inputs(
        self, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf, buffer_parity: int
    ):
        raw_k_snapshot = self.ub_raw_k[buffer_parity]
        gate_cumsum_snapshot = self.ub_cs_snapshot[buffer_parity]
        log_gamma_c_write = self.ub_gamma_c.next_write()
        self.save_gamma_c_row(gate_cumsum_dhalf, log_gamma_c_write)
        mem_copy(self.ub_gamma, gate_cumsum_dhalf)
        mem_copy(gate_cumsum_snapshot, gate_cumsum_dhalf)

        mem_copy(raw_k_snapshot, normalized_k_dhalf)

        cast_write = self.ub_cast_bf16
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                gamma = vload(self.ub_gamma, offset)
                exp_value = vexp(gamma, mask=row_mask)
                q_packed = vload_unpack(normalized_q_dhalf, offset, unpack_mode=UnpackMode.B16_TO_B32)
                q_value = vcast(q_packed, dtypes.float32, mask=row_mask)
                gated_q = vmul(q_value, exp_value, mask=row_mask)
                q_decayed = vmuls(gated_q, self.scale_value, mask=row_mask)
                q_decayed_bf16 = vcast(q_decayed, dtypes.bfloat16, mask=row_mask)
                vstore_pack(cast_write, offset, q_decayed_bf16, row_mask, pack_mode="b32_to_b16")
        return self.ub_cast_bf16, gate_cumsum_dhalf, normalized_k_dhalf, normalized_q_dhalf

    def finish_mqk(self, mqk_chunk, subblock_base: int):
        Mqk_part = tile_slice(mqk_chunk, (HALF_CHUNK_SIZE, CHUNK_SIZE), (subblock_base, 0))
        self._store_masked_mqk_to_bf16_gm(Mqk_part)
        return self.ub_gamma_c.next_read()

    def store_q_decayed_scratch(self, qg_chunk, q_decayed_slot):
        # Q_decayed 的 bf16 已在 prepare_chunk 的 Q_decay vf 里 cast 好、留在 ub_cast_bf16(buf9)；
        # 本方法紧跟 prepare_chunk、中间无人碰 buf9 → 直接把它搬到 GM，省一次 f32→bf16 cast。
        Q_decayed_slot_part = tile_slice(qg_chunk, (CHUNK_SIZE, D_HALF), (0, self.subblock_idx))
        mem_copy(Q_decayed_slot_part, q_decayed_slot[:Q_decayed_slot_part.shape[0], :D_HALF])

    @jit
    def store_restored_state_inputs(
        self,
        kg_chunk,
        decay_chunk,
        gamma_c_slot,
        buffer_parity: int,
    ):
        raw_k_snapshot = self.ub_raw_k[buffer_parity]
        gate_cumsum_snapshot = self.ub_cs_snapshot[buffer_parity]
        K_restored_dhalf = tile_slice(kg_chunk, (CHUNK_SIZE, D_HALF), (0, self.subblock_idx))
        cast_write = self.ub_cast_bf16
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            cs_last = vload(gamma_c_slot, 0)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                k_packed = vload_unpack(raw_k_snapshot, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k_value = vcast(k_packed, dtypes.float32, mask=row_mask)
                gate_cumsum = vload(gate_cumsum_snapshot, offset)
                remaining_gate = vsub(cs_last, gate_cumsum, mask=row_mask)
                remaining_decay = vexp(remaining_gate, mask=row_mask)
                restored = vmul(k_value, remaining_decay, mask=row_mask)
                restored_bf16 = vcast(restored, dtypes.bfloat16, mask=row_mask)
                vstore_pack(cast_write, offset, restored_bf16, row_mask, pack_mode="b32_to_b16")
        cast_read = self.ub_cast_bf16
        mem_copy(K_restored_dhalf, cast_read[:K_restored_dhalf.shape[0], :D_HALF])

        # gamma_C can underflow to zero, which is the correct state-decay value; it is never multiplied by K_inv.
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            gamma_c_log = vload(gamma_c_slot, 0)
            gamma_c = vexp(gamma_c_log, mask=row_mask)
            vstore(gamma_c_slot, 0, gamma_c, row_mask)
        gamma_C_gm = tile_slice(decay_chunk, (1, D_HALF), (0, self.subblock_idx))
        mem_copy(gamma_C_gm, gamma_c_slot)

    @jit
    def build_per_d_left_k(self, normalized_k_dhalf, gate_cumsum_dhalf, row_block: int, left_k_l1_operand):
        k_read = tile_slice(normalized_k_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), (row_block, 0))
        ref_offset = row_block * NEUMANN_BLOCK_SIZE * D_HALF

        nd_write = self.ub_per_d_nd.next_write()
        with vf(mode="simd"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            ref = vload(gate_cumsum_dhalf, ref_offset)
            for row in dsl_range(0, NEUMANN_BLOCK_SIZE, 1, unroll=2):
                offset = row * D_HALF
                g_offset = ((row_block * NEUMANN_BLOCK_SIZE) + row) * D_HALF
                k_packed = vload_unpack(k_read, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k_value = vcast(k_packed, dtypes.float32, mask=mask)
                gate_cumsum = vload(gate_cumsum_dhalf, g_offset)
                gate_delta = vsub(gate_cumsum, ref, mask=mask)
                gate_factor = vexp(gate_delta, mask=mask)
                weighted_k = vmul(k_value, gate_factor, mask=mask)
                weighted_k_bf16 = vcast(weighted_k, dtypes.bfloat16, mask=mask)
                vstore_pack(nd_write, offset, weighted_k_bf16, mask, pack_mode="b32_to_b16")
        nd_read = self.ub_per_d_nd.next_read()
        self._write_per_d_full_d_handoff(left_k_l1_operand, nd_read)

    @jit
    def build_per_d_left_q(self, normalized_q_dhalf, gate_cumsum_dhalf, row_block: int, left_q_l1_operand):
        q_read = tile_slice(normalized_q_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), (row_block, 0))
        ref_offset = row_block * NEUMANN_BLOCK_SIZE * D_HALF
        nd_write = self.ub_per_d_nd.next_write()
        with vf(mode="simd"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            ref = vload(gate_cumsum_dhalf, ref_offset)
            for row in dsl_range(0, NEUMANN_BLOCK_SIZE, 1, unroll=2):
                offset = row * D_HALF
                g_offset = ((row_block * NEUMANN_BLOCK_SIZE) + row) * D_HALF
                q_packed = vload_unpack(q_read, offset, unpack_mode=UnpackMode.B16_TO_B32)
                q_value = vcast(q_packed, dtypes.float32, mask=mask)
                gate_cumsum = vload(gate_cumsum_dhalf, g_offset)
                gate_delta = vsub(gate_cumsum, ref, mask=mask)
                gate_factor = vexp(gate_delta, mask=mask)
                gated_q = vmul(q_value, gate_factor, mask=mask)
                scaled_q = vmuls(gated_q, self.scale_value, mask=mask)
                scaled_q_bf16 = vcast(scaled_q, dtypes.bfloat16, mask=mask)
                vstore_pack(nd_write, offset, scaled_q_bf16, mask, pack_mode="b32_to_b16")
        nd_read = self.ub_per_d_nd.next_read()
        self._write_per_d_full_d_handoff(left_q_l1_operand, nd_read)

    @jit
    def build_per_d_right(
        self, normalized_k_dhalf, gate_cumsum_dhalf, row_block: int, col_block: int, right_k_l1_operand
    ):
        k_read = tile_slice(normalized_k_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), (col_block, 0))
        ref_offset = row_block * NEUMANN_BLOCK_SIZE * D_HALF

        nd_write = self.ub_per_d_nd.next_write()
        with vf(mode="simd"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            ref = vload(gate_cumsum_dhalf, ref_offset)
            for row in dsl_range(0, NEUMANN_BLOCK_SIZE, 1, unroll=2):
                offset = row * D_HALF
                g_offset = ((col_block * NEUMANN_BLOCK_SIZE) + row) * D_HALF
                k_packed = vload_unpack(k_read, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k_value = vcast(k_packed, dtypes.float32, mask=mask)
                gate_cumsum = vload(gate_cumsum_dhalf, g_offset)
                gate_delta = vsub(ref, gate_cumsum, mask=mask)
                gate_factor = vexp(gate_delta, mask=mask)
                restored_k = vmul(k_value, gate_factor, mask=mask)
                restored_k_bf16 = vcast(restored_k, dtypes.bfloat16, mask=mask)
                vstore_pack(nd_write, offset, restored_k_bf16, mask, pack_mode="b32_to_b16")
        nd_read = self.ub_per_d_nd.next_read()
        self._write_per_d_full_d_handoff(right_k_l1_operand, nd_read)

    @jit
    def _normalize_and_publish_q(self, q_slot, q_to_k):
        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        self.normalize_full_d_bf16(q_slot)
        mem_copy(self.ub_q_bf16, reinterpret(q_slot, offset=0, layout=make_layout((CHUNK_SIZE, D_HALF), stride=(SUPPORTED_HEAD_DIM, 1))))
        cast_tile(self.ub_qk_exchange, reinterpret(q_slot, offset=D_HALF * 2, layout=make_layout((CHUNK_SIZE, D_HALF), stride=(SUPPORTED_HEAD_DIM, 1))))
        self.publish_normalized_qk_half(q_to_k, self.ub_qk_exchange)

    @jit
    def _normalize_and_publish_k(self, k_slot, k_to_q):
        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_SUBBLOCK1_EVENT_ID)
        self.normalize_full_d_bf16(k_slot)
        mem_copy(self.ub_k_bf16, reinterpret(k_slot, offset=D_HALF * 2, layout=make_layout((CHUNK_SIZE, D_HALF), stride=(SUPPORTED_HEAD_DIM, 1))))
        cast_tile(self.ub_qk_exchange, k_slot)
        self.publish_normalized_qk_half(k_to_q, self.ub_qk_exchange)

    @jit
    def _load_tail_bf16_tensor(self, source, channel, *, cols: int, ready_event: int):
        producer = channel
        self._fill_bf16_invalid_rows(producer, valid_rows=source.shape[0], cols=cols, value=0.0)
        vec_sync_notify(PIPE.V, PIPE.MTE2, TAIL_V_TO_MTE2_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.MTE2, TAIL_V_TO_MTE2_EVENT_ID)
        mem_copy(producer[:source.shape[0], :source.shape[1]], source)
        vec_sync_notify(PIPE.MTE2, PIPE.V, ready_event)
        return producer

    @jit
    def _finish_beta_weighted_c3_operands(self, v_slot, beta_slot, raw_k_snapshot):
        cast_write = self.ub_cast_bf16
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                beta = vload_broadcast(beta_slot, row)
                neg_beta = vmuls(beta, -1.0, mask=row_mask)
                cs_value = vload(self.ub_gamma, offset)
                k_packed = vload_unpack(raw_k_snapshot, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k_value = vcast(k_packed, dtypes.float32, mask=row_mask)
                gamma_exp = vexp(cs_value, mask=row_mask)
                k_decayed = vmul(k_value, gamma_exp, mask=row_mask)
                k_decayed_beta = vmul(k_decayed, neg_beta, mask=row_mask)
                k_bf16 = vcast(k_decayed_beta, dtypes.bfloat16, mask=row_mask)
                vstore_pack(cast_write, offset, k_bf16, row_mask, pack_mode="b32_to_b16")
        cast_read = self.ub_cast_bf16
        nd2nz_tile(self.ub_bf16_nz.next_write(), cast_read)
        self._store_dhalf_nz_ub_to_full_l1(self.l1_K_decayed_beta_for_W, self.ub_bf16_nz.next_read())

        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_V_STAGING_EVENT_ID)
        cast_write = self.ub_cast_bf16
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                beta = vload_broadcast(beta_slot, row)
                v_packed = vload_unpack(v_slot, offset, unpack_mode=UnpackMode.B16_TO_B32)
                v_value = vcast(v_packed, dtypes.float32, mask=row_mask)
                weighted_v = vmul(v_value, beta, mask=row_mask)
                weighted_v_bf16 = vcast(weighted_v, dtypes.bfloat16, mask=row_mask)
                vstore_pack(cast_write, offset, weighted_v_bf16, row_mask, pack_mode="b32_to_b16")
        cast_read = self.ub_cast_bf16
        nd2nz_tile(self.ub_bf16_nz.next_write(), cast_read)
        self._store_dhalf_nz_ub_to_full_l1(self.l1_v_beta, self.ub_bf16_nz.next_read())

    @jit
    def _store_masked_mqk_to_bf16_gm(self, gm_tile):
        # Mqk 保留含对角下三角：update_mask(全局行 i+1) 产生寄存器谓词，vselect 清零其余列。
        mqk_slot = self.ub_mqk.next_read()
        bf16_write = self.ub_mqk_bf16
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            zero_reg = vdups(0.0, dtypes.float32, mask=row_mask)
            for row in dsl_range(0, HALF_CHUNK_SIZE, 1, unroll=4):
                off = row * CHUNK_SIZE
                keep, _ = update_mask(self.subblock_idx * HALF_CHUNK_SIZE + row + 1, elem_bits=32)
                mqk = vload(mqk_slot, off)
                mqk_tril = vselect_raw(mqk, zero_reg, cond_mask=keep)  # 保留 lane<=i（含对角）
                mqk_bf16 = vcast(mqk_tril, dtypes.bfloat16, mask=row_mask)
                vstore_pack(bf16_write, off, mqk_bf16, row_mask, pack_mode="b32_to_b16")
        mem_copy(gm_tile, self.ub_mqk_bf16)

    @jit
    def _write_per_d_full_d_handoff(self, full_channel, nd_slot):
        nd2nz_tile(self.ub_per_d_nz.next_write(), nd_slot)
        mem_copy(full_channel.next_write().partition(dim=1, part_id=self.subblock_idx), self.ub_per_d_nz.next_read())

    def _store_dhalf_nz_ub_to_full_l1(self, l1_dst, ub_nz_src):
        mem_copy(l1_dst.partition(dim=1, part_id=self.subblock_idx), ub_nz_src)

    def _store_neumann_tmp_to_handoff_l1(self, l1_dst, subblock_base: int):
        mem_copy(l1_dst.partition(dim=0, part_id=dtypes.int64(subblock_base)), self.ub_kkt_nz)


@jit
def _prepare_stage1_delayed(runtime, gm_V, location):
    runtime.vector.apply_beta_and_finish_t_for_neumann(
        runtime.vector.ub_beta, runtime.matmul.neumann_power_handoff_l1, runtime.subblock, location.parity
    )
    runtime.vector.prepare_beta_weighted_c3_operands(
        gm_V,
        runtime.vector.ub_beta,
        location,
    )
    negative_lower_matrix, identity_matrix, neumann_scratch = (
        runtime.matmul.load_packed64_diagonal_inputs_from_l1_handoff(
            runtime.matmul.neumann_power_handoff_l1, runtime.identity
        )
    )
    second_power, packed_inverse = runtime.matmul.neumann_diag_power2_update(
        negative_lower_matrix, identity_matrix, neumann_scratch
    )
    fourth_power, packed_inverse = runtime.matmul.neumann_diag_power4_update(second_power, packed_inverse)
    packed_inverse = runtime.matmul.neumann_diag_power8_update(fourth_power, packed_inverse)
    packed_inverse = runtime.matmul.compose_odd_even_lower16_to32_accum_full64(neumann_scratch, packed_inverse)
    runtime.matmul.compose_odd_even_lower32_to64_accum_full64(neumann_scratch, packed_inverse)


@jit
def _compute_stage1_products_first_half(matmul_ops, vector, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf):
    vector.build_per_d_left_k(normalized_k_dhalf, gate_cumsum_dhalf, 0, vector.l1_per_d_operand)
    matmul_ops.load_per_d_left_k(vector.l1_per_d_operand)
    vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 0, 0, vector.l1_per_d_operand)
    right_k = matmul_ops.load_per_d_right_k(vector.l1_per_d_operand)
    matmul_ops.mm_per_d_kkt(0, 0, right_k)
    vector.build_per_d_left_q(normalized_q_dhalf, gate_cumsum_dhalf, 0, vector.l1_per_d_operand)
    matmul_ops.load_per_d_left_q(vector.l1_per_d_operand)
    matmul_ops.mm_per_d_mqk(0, 0, right_k)

    vector.build_per_d_left_k(normalized_k_dhalf, gate_cumsum_dhalf, 1, vector.l1_per_d_operand)
    matmul_ops.load_per_d_left_k(vector.l1_per_d_operand)
    vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 1, 0, vector.l1_per_d_operand)
    right_k = matmul_ops.load_per_d_right_k(vector.l1_per_d_operand)
    matmul_ops.mm_per_d_kkt(1, 0, right_k)
    vector.build_per_d_left_q(normalized_q_dhalf, gate_cumsum_dhalf, 1, vector.l1_per_d_operand)
    matmul_ops.load_per_d_left_q(vector.l1_per_d_operand)
    matmul_ops.mm_per_d_mqk(1, 0, right_k)
    vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 1, 1, vector.l1_per_d_operand)
    right_k = matmul_ops.load_per_d_right_k(vector.l1_per_d_operand)
    matmul_ops.mm_per_d_kkt(1, 1, right_k)
    matmul_ops.mm_per_d_mqk(1, 1, right_k)


@jit
def _compute_stage1_products_second_half(matmul_ops, vector, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf):
    vector.build_per_d_left_k(normalized_k_dhalf, gate_cumsum_dhalf, 2, vector.l1_per_d_operand)
    matmul_ops.load_per_d_left_k(vector.l1_per_d_operand)
    vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 2, 0, vector.l1_per_d_operand)
    right_k = matmul_ops.load_per_d_right_k(vector.l1_per_d_operand)
    matmul_ops.mm_per_d_kkt(2, 0, right_k)
    vector.build_per_d_left_q(normalized_q_dhalf, gate_cumsum_dhalf, 2, vector.l1_per_d_operand)
    matmul_ops.load_per_d_left_q(vector.l1_per_d_operand)
    matmul_ops.mm_per_d_mqk(2, 0, right_k)
    vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 2, 1, vector.l1_per_d_operand)
    right_k = matmul_ops.load_per_d_right_k(vector.l1_per_d_operand)
    matmul_ops.mm_per_d_kkt(2, 1, right_k)
    matmul_ops.mm_per_d_mqk(2, 1, right_k)
    vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 2, 2, vector.l1_per_d_operand)
    right_k = matmul_ops.load_per_d_right_k(vector.l1_per_d_operand)
    matmul_ops.mm_per_d_kkt(2, 2, right_k)
    matmul_ops.mm_per_d_mqk(2, 2, right_k)

    vector.build_per_d_left_k(normalized_k_dhalf, gate_cumsum_dhalf, 3, vector.l1_per_d_operand)
    matmul_ops.load_per_d_left_k(vector.l1_per_d_operand)
    vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 3, 0, vector.l1_per_d_operand)
    right_k = matmul_ops.load_per_d_right_k(vector.l1_per_d_operand)
    matmul_ops.mm_per_d_kkt(3, 0, right_k)
    vector.build_per_d_left_q(normalized_q_dhalf, gate_cumsum_dhalf, 3, vector.l1_per_d_operand)
    matmul_ops.load_per_d_left_q(vector.l1_per_d_operand)
    matmul_ops.mm_per_d_mqk(3, 0, right_k)
    vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 3, 1, vector.l1_per_d_operand)
    right_k = matmul_ops.load_per_d_right_k(vector.l1_per_d_operand)
    matmul_ops.mm_per_d_kkt(3, 1, right_k)
    matmul_ops.mm_per_d_mqk(3, 1, right_k)
    vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 3, 2, vector.l1_per_d_operand)
    right_k = matmul_ops.load_per_d_right_k(vector.l1_per_d_operand)
    matmul_ops.mm_per_d_kkt(3, 2, right_k)
    matmul_ops.mm_per_d_mqk(3, 2, right_k)
    vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 3, 3, vector.l1_per_d_operand)
    right_k = matmul_ops.load_per_d_right_k(vector.l1_per_d_operand)
    matmul_ops.mm_per_d_kkt(3, 3, right_k)
    matmul_ops.mm_per_d_mqk(3, 3, right_k)


@jit
def _compute_stage1_products(matmul_ops, vector, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf):
    _compute_stage1_products_first_half(matmul_ops, vector, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf)
    _compute_stage1_products_second_half(matmul_ops, vector, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf)


@jit
def _publish_stage1_delayed(runtime, outputs, location):
    mqk_chunk = _chunk_matrix_tile(outputs.mqk, location.batch, location.head, location.chunk)
    kg_chunk = _sequence_head_chunk_tile(outputs.kg, location.batch, location.head, location.chunk)
    decay_chunk = tile_slice(
        outputs.decay[location.batch, location.head, None, None], (1, runtime.head_dim), (location.chunk, 0)
    )
    u_chunk = _sequence_head_chunk_tile(outputs.u, location.batch, location.head, location.chunk)
    w_chunk = _sequence_head_chunk_tile(outputs.w, location.batch, location.head, location.chunk)
    log_gamma_c_dhalf = runtime.vector.finish_mqk(mqk_chunk, runtime.subblock)
    runtime.vector.store_restored_state_inputs(kg_chunk, decay_chunk, log_gamma_c_dhalf, location.parity)
    runtime.matmul.compute_u_pre_and_w_from_resident_inv_beta(
        u_chunk, w_chunk, runtime.vector.l1_v_beta, runtime.vector.l1_K_decayed_beta_for_W
    )


@jit
def _allocate_summary_workspace(head_dim):
    kg_l1 = RingBuffer(MemLoc.L1, dtypes.float32, make_layout((CHUNK_SIZE, head_dim)), depth=1).next_write()
    u_l1 = RingBuffer(MemLoc.L1, dtypes.float32, make_layout((CHUNK_SIZE, head_dim)), depth=1).next_write()
    w_l1 = RingBuffer(MemLoc.L1, dtypes.float32, make_layout((CHUNK_SIZE, head_dim)), depth=1).next_write()
    phi_l1 = RingBuffer(MemLoc.L1, dtypes.float32, make_layout((head_dim, head_dim)), depth=1).next_write()
    initial_rank_phi_l1 = Buffer(MemLoc.L1, dtypes.float32, make_layout((head_dim, head_dim)))
    recurrent_rank_phi_l1 = RingBuffer(MemLoc.L1, dtypes.float32, make_layout((head_dim, head_dim)), depth=1).next_write()
    rank_psi_l1 = RingBuffer(MemLoc.L1, dtypes.float32, make_layout((head_dim, head_dim)), depth=1).next_write()
    loaded_rank_phi_l1 = RingBuffer(MemLoc.L1, dtypes.float32, make_layout((head_dim, head_dim)), depth=1).next_write()
    loaded_rank_psi_l1 = RingBuffer(MemLoc.L1, dtypes.float32, make_layout((head_dim, head_dim)), depth=1).next_write()
    stage_two_matmul = StageTwoMatmul(head_dim=head_dim)
    phi_ub = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((head_dim // 2, head_dim)), depth=1).next_write()
    psi_ub = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((head_dim // 2, head_dim)), depth=1).next_write()
    stage_two_vector = StageTwoVector(head_dim=head_dim)

    return _SummaryWorkspace(
        kg=kg_l1,
        u=u_l1,
        w=w_l1,
        phi=phi_l1,
        initial_phi=initial_rank_phi_l1,
        recurrent_phi=recurrent_rank_phi_l1,
        psi=rank_psi_l1,
        loaded_phi=loaded_rank_phi_l1,
        loaded_psi=loaded_rank_psi_l1,
        matmul=stage_two_matmul,
        phi_ub=phi_ub,
        psi_ub=psi_ub,
        vector=stage_two_vector,
    )


@jit
def _prepare_summary_phi(workspace, decay_tile, gm_transition, chunk):
    workspace.matmul.compute_chunk_product(workspace.w, workspace.kg, workspace.matmul.l0c_phi)
    workspace.matmul.store_summary_to_ub(workspace.phi_ub, workspace.matmul.l0c_phi)
    is_group_first = chunk.index == chunk.group_start
    is_group_last = chunk.index + 1 == chunk.group_end
    if chunk.group_start == 0 and is_group_first and is_group_last:
        workspace.vector.add_decay_and_publish_phi(
            workspace.phi_ub,
            decay_tile,
            _PhiDestination(phi=workspace.phi, rank=workspace.initial_phi, transition=gm_transition),
            _PhiPublication(
                half=chunk.subblock, write_transition=True, publish_phi=False, initialize_rank=False, state=chunk.state
            ),
        )
    elif chunk.group_start == 0 and is_group_first:
        workspace.vector.add_decay_and_publish_phi(
            workspace.phi_ub,
            decay_tile,
            _PhiDestination(phi=workspace.phi, rank=workspace.initial_phi, transition=gm_transition),
            _PhiPublication(
                half=chunk.subblock, write_transition=False, publish_phi=False, initialize_rank=True, state=chunk.state
            ),
        )
        vec_sync_block_arrive(PIPE.MTE3, SUMMARY_RANK_PHI_READY_FLAG_ID, mode=2)
    else:
        workspace.vector.add_decay_and_publish_phi(
            workspace.phi_ub,
            decay_tile,
            _PhiDestination(phi=workspace.phi, rank=workspace.initial_phi, transition=gm_transition),
            _PhiPublication(
                half=chunk.subblock, write_transition=False, publish_phi=True, initialize_rank=False, state=chunk.state
            ),
        )

    return is_group_first, is_group_last


@jit
def _compose_summary_chunk(workspace, chunk):
    workspace.matmul.compute_chunk_product(workspace.u, workspace.kg, workspace.matmul.l0c_psi)
    if chunk.group_start > 0 and chunk.first:
        workspace.matmul.compute_rank_product(workspace.loaded_phi, workspace.phi, workspace.matmul.l0c_phi, init=True)
        workspace.matmul.compute_rank_product(workspace.loaded_psi, workspace.phi, workspace.matmul.l0c_psi, init=False)
    elif chunk.group_start > 0 or not chunk.first:
        if chunk.group_start == 0 and chunk.index == chunk.group_start + 1:
            cube_sync_block_wait(PIPE.S, SUMMARY_RANK_PHI_READY_FLAG_ID, mode=2)
            workspace.matmul.compute_rank_product(
                workspace.initial_phi, workspace.phi, workspace.matmul.l0c_phi, init=True
            )
        else:
            workspace.matmul.compute_rank_product(
                workspace.recurrent_phi, workspace.phi, workspace.matmul.l0c_phi, init=True
            )
        workspace.matmul.compute_rank_product(workspace.psi, workspace.phi, workspace.matmul.l0c_psi, init=False)


@jit
def _store_summary_chunk(workspace, gm_state_ext, chunk, transition_tile):
    if chunk.group_start > 0 or not chunk.first:
        if not chunk.last:
            workspace.matmul.store_summary_to_l1(workspace.recurrent_phi, workspace.matmul.l0c_phi)
        else:
            mem_copy(transition_tile, workspace.matmul.l0c_phi)
    elif chunk.last:
        pass

    if chunk.group_start == 0 and chunk.first:
        if not chunk.last:
            workspace.matmul.store_summary_to_l1(workspace.psi, workspace.matmul.l0c_psi)
        else:
            workspace.matmul.store_summary_to_ub(workspace.psi_ub, workspace.matmul.l0c_psi)
            workspace.vector.write_crosscore_matrix_to_gm(
                workspace.psi_ub, gm_state_ext, subblock_idx=chunk.subblock, state_index=chunk.state
            )
    else:
        if not chunk.last:
            workspace.matmul.store_summary_to_l1(workspace.psi, workspace.matmul.l0c_psi)
        else:
            workspace.matmul.store_summary_to_ub(workspace.psi_ub, workspace.matmul.l0c_psi)
            workspace.vector.write_crosscore_matrix_to_gm(
                workspace.psi_ub, gm_state_ext, subblock_idx=chunk.subblock, state_index=chunk.state
            )


@jit
def _run_stage1_round(inputs, outputs, resources, gm_metadata, schedule):
    for active_batch_idx in range(0, schedule.batch_count):
        prefix_start = gm_metadata[schedule.task_prefix_base + active_batch_idx]
        prefix_end = gm_metadata[schedule.task_prefix_base + active_batch_idx + 1]
        intersection_start = schedule.core_start
        if intersection_start < prefix_start:
            intersection_start = prefix_start
        intersection_end = schedule.core_end
        if intersection_end > prefix_end:
            intersection_end = prefix_end
        task_start = 0
        task_end = 0
        if intersection_start < intersection_end:
            task_start = intersection_start - prefix_start
            task_end = intersection_end - prefix_start

        logical_batch = gm_metadata[schedule.batch_base + active_batch_idx]
        storage_batch = gm_metadata[schedule.storage_base + active_batch_idx]
        token_start = gm_metadata[schedule.token_base + active_batch_idx]
        valid_seq_len = dtypes.int64(gm_metadata[schedule.length_base + active_batch_idx])
        group_start = gm_metadata[schedule.chunk_start_base + active_batch_idx]
        group_count = gm_metadata[schedule.chunk_count_base + active_batch_idx]
        chunks_total = (valid_seq_len + (CHUNK_SIZE - 1)) // CHUNK_SIZE
        tail_rows = valid_seq_len - (chunks_total - 1) * CHUNK_SIZE
        token_stop = token_start + valid_seq_len

        k = inputs.k[storage_batch, None, token_start:token_stop, None]
        v = inputs.v[storage_batch, None, token_start:token_stop, None]
        q = inputs.q[storage_batch, None, token_start:token_stop, None]
        beta = inputs.beta[storage_batch, None, token_start:token_stop, None]
        gate = inputs.g[storage_batch, None, token_start:token_stop, None]
        qg = outputs.qg[storage_batch, None, token_start:token_stop, None]
        kg = outputs.kg[storage_batch, None, token_start:token_stop, None]
        u = outputs.u[storage_batch, None, token_start:token_stop, None]
        w = outputs.w[storage_batch, None, token_start:token_stop, None]
        gamma = outputs.decay[logical_batch, None, None, None]
        mqk = outputs.mqk[logical_batch, None, None, None, None]
        _run_stage1_body(
            _StageOneInputs(k=k, v=v, q=q, beta=beta, g=gate),
            _StageOneOutputs(qg=qg, mqk=mqk, kg=kg, decay=gamma, u=u, w=w),
            _StageOneResources(
                exchange=resources.exchange,
                identity=resources.identity,
                head_dim=resources.head_dim,
                scale=resources.scale,
                gate=resources.gate,
            ),
            _StageOneSchedule(
                head_base=0,
                start=group_start,
                count=group_count,
                tail_rows=tail_rows,
                tail_group=group_start + group_count == chunks_total,
                tasks=_TaskRange(start=task_start, end=task_end),
            ),
        )


@jit
def _run_summary_round(head_dim, inputs, outputs, gm_metadata, schedule):
    for active_batch_idx in range(0, schedule.batch_count):
        prefix_start = gm_metadata[schedule.task_prefix_base + active_batch_idx]
        prefix_end = gm_metadata[schedule.task_prefix_base + active_batch_idx + 1]
        intersection_start = schedule.core_start
        if intersection_start < prefix_start:
            intersection_start = prefix_start
        intersection_end = schedule.core_end
        if intersection_end > prefix_end:
            intersection_end = prefix_end
        task_start = 0
        task_end = 0
        if intersection_start < intersection_end:
            task_start = intersection_start - prefix_start
            task_end = intersection_end - prefix_start
        dv_splits_num = gm_metadata[schedule.splits_base + active_batch_idx]
        logical_batch = gm_metadata[schedule.batch_base + active_batch_idx]
        storage_batch = gm_metadata[schedule.storage_base + active_batch_idx]
        token_start = gm_metadata[schedule.token_base + active_batch_idx]
        valid_seq_len = dtypes.int64(gm_metadata[schedule.length_base + active_batch_idx])
        group_start = gm_metadata[schedule.chunk_start_base + active_batch_idx]
        group_count = gm_metadata[schedule.chunk_count_base + active_batch_idx]
        token_stop = token_start + valid_seq_len
        summary_start = (task_start + dv_splits_num - 1) // dv_splits_num
        summary_end = (task_end + dv_splits_num - 1) // dv_splits_num
        if summary_start < summary_end:
            _run_cp_prepare_summary_body(
                head_dim,
                _SummaryInputs(
                    kg=inputs.kg[storage_batch, None, token_start:token_stop, None],
                    decay=inputs.decay,
                    u=inputs.u[storage_batch, None, token_start:token_stop, None],
                    w=inputs.w[storage_batch, None, token_start:token_stop, None],
                ),
                outputs,
                _SummarySchedule(
                    head_base=0,
                    head_count=schedule.heads,
                    group_start=group_start,
                    group_count=group_count,
                    item_start=summary_start,
                    item_end=summary_end,
                ),
                _SummarySequence(storage_batch=0, logical_batch=logical_batch, length=valid_seq_len),
            )


@jit
def _decode_stage1_round(gm_metadata, round_idx, block_idx):
    record = gm_metadata[HEADER_WORDS + round_idx]
    active_batch_num = gm_metadata[record + 1]
    batch_idx_base = record + 2
    storage_batch_idx_base = batch_idx_base + active_batch_num
    token_start_base = storage_batch_idx_base + active_batch_num
    valid_seq_len_base = token_start_base + active_batch_num
    group_num_index = valid_seq_len_base + active_batch_num
    chunk_start_base = group_num_index + 1
    chunk_num_base = chunk_start_base + active_batch_num
    stage1_task_prefix_base = chunk_num_base + active_batch_num
    stage1_core_ranges_base = stage1_task_prefix_base + active_batch_num + 1
    stage1_core_range_word = stage1_core_ranges_base + block_idx * 2
    stage1_core_start = gm_metadata[stage1_core_range_word]
    stage1_core_end = gm_metadata[stage1_core_range_word + 1]

    return (
        active_batch_num,
        batch_idx_base,
        storage_batch_idx_base,
        token_start_base,
        valid_seq_len_base,
        chunk_start_base,
        chunk_num_base,
        stage1_task_prefix_base,
        stage1_core_ranges_base,
        stage1_core_start,
        stage1_core_end,
    )


@jit
def _decode_summary_round(gm_metadata, stage1_core_ranges_base, active_batch_num, block_idx):
    dv_splits_base = stage1_core_ranges_base + (MAX_AIC_CORES * 2)
    stage2_task_prefix_base = dv_splits_base + active_batch_num
    stage2_core_ranges_base = stage2_task_prefix_base + active_batch_num + 1
    stage2_core_range_word = stage2_core_ranges_base + block_idx * 2
    stage2_core_start = gm_metadata[stage2_core_range_word]
    stage2_core_end = gm_metadata[stage2_core_range_word + 1]
    # Reuse the shared Stage2 partition for the CP affine summary.
    # Stage2 assigns ``head * dv_splits + split`` tasks; split zero
    # is the unique owner of each complete head summary.
    return dv_splits_base, stage2_task_prefix_base, stage2_core_start, stage2_core_end


@jit
def _stage1_task_coordinates(task_index, inputs, schedule):
    local_head_index = task_index // schedule.count
    group_chunk_index = task_index - local_head_index * schedule.count
    chunk_index = schedule.start + group_chunk_index
    global_head_index = schedule.head_base + local_head_index
    batch_index, value_head_index = idx2crd(global_head_index, [inputs.k.shape[0], inputs.v.shape[1]])
    buffer_parity = task_index % 2
    return batch_index, value_head_index, chunk_index, buffer_parity


@jit
def _prepare_stage1_previous(runtime, inputs, schedule, task_indices, valid_row_counts, read_slot):
    delayed_task_index = task_indices[read_slot]
    batch, head, chunk, parity = _stage1_task_coordinates(delayed_task_index, inputs, schedule)
    valid_rows = valid_row_counts[read_slot]
    _prepare_stage1_delayed(
        runtime, inputs.v, _ChunkLocation(batch=batch, head=head, chunk=chunk, parity=parity, valid_rows=valid_rows)
    )


@jit
def _publish_stage1_previous(runtime, inputs, outputs, schedule, task_indices, read_slot):
    delayed_task_index = task_indices[read_slot]
    batch, head, chunk, parity = _stage1_task_coordinates(delayed_task_index, inputs, schedule)
    _publish_stage1_delayed(runtime, outputs, _ChunkLocation(batch=batch, head=head, chunk=chunk, parity=parity))


@jit
def _run_stage1_current(runtime, inputs, outputs, resources, location):
    q_to_k_exchange, k_to_q_exchange = runtime.vector.prepare_normalized_qk_local_half(
        inputs,
        resources.exchange,
        location,
        runtime.block,
    )
    gate_cumsum_dhalf = runtime.vector.activate_raw_gate(
        inputs.g,
        resources.gate,
        location,
    )
    normalized_k_dhalf, normalized_q_dhalf = runtime.vector.finish_normalized_qk_exchange(
        q_to_k_exchange, k_to_q_exchange
    )
    decayed_q_dhalf, gate_cumsum_dhalf, normalized_k_dhalf, normalized_q_dhalf = (
        runtime.vector.prepare_normalized_chunk_inputs(
            normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf, location.parity
        )
    )
    qg_chunk = _sequence_head_chunk_tile(outputs.qg, location.batch, location.head, location.chunk)
    runtime.vector.store_q_decayed_scratch(qg_chunk, decayed_q_dhalf)

    _compute_stage1_products(runtime.matmul, runtime.vector, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf)

    runtime.vector.activate_raw_beta(
        inputs.beta,
        location,
    )
    runtime.matmul.finish_per_d_matrices(runtime.vector.ub_kkt, runtime.vector.ub_mqk.next_write())


@jit
def _issue_stage1_task(runtime, inputs, outputs, resources, task):
    batch, head, chunk, parity = _stage1_task_coordinates(task.index, inputs, task.schedule)
    heads_per_key_head = inputs.v.shape[1] // inputs.k.shape[1]
    qk_head = head // heads_per_key_head
    group_chunk = chunk - task.schedule.start
    valid_rows = CHUNK_SIZE
    if task.schedule.tail_group and group_chunk + 1 == task.schedule.count:
        valid_rows = task.schedule.tail_rows
    task.task_indices[task.write_slot] = task.index
    task.valid_row_counts[task.write_slot] = valid_rows
    _run_stage1_current(
        runtime,
        inputs,
        outputs,
        resources,
        _ChunkLocation(batch=batch, head=head, chunk=chunk, parity=parity, valid_rows=valid_rows, qk_head=qk_head),
    )


@jit
def _run_stage1_tasks(runtime, inputs, outputs, resources, schedule):
    task_indices = Array(dtypes.int64, 2)
    valid_row_counts = Array(dtypes.int64, 2)
    pipeline_tick = 0
    issued_task_count = 0
    for task_index in range(schedule.tasks.start, schedule.tasks.end):
        if pipeline_tick >= 1 and pipeline_tick - 1 < issued_task_count:
            _prepare_stage1_previous(runtime, inputs, schedule, task_indices, valid_row_counts, (pipeline_tick - 1) % 2)
        _issue_stage1_task(runtime, inputs, outputs, resources, _StageOneTask(schedule, task_index, task_indices, valid_row_counts, pipeline_tick % 2))
        issued_task_count = issued_task_count + 1
        if pipeline_tick >= 1 and pipeline_tick - 1 < issued_task_count:
            _publish_stage1_previous(runtime, inputs, outputs, schedule, task_indices, (pipeline_tick - 1) % 2)
        pipeline_tick = pipeline_tick + 1
    if pipeline_tick >= 1 and pipeline_tick - 1 < issued_task_count:
        _prepare_stage1_previous(runtime, inputs, schedule, task_indices, valid_row_counts, (pipeline_tick - 1) % 2)
        _publish_stage1_previous(runtime, inputs, outputs, schedule, task_indices, (pipeline_tick - 1) % 2)


@jit
def _run_stage1_body(inputs, outputs, resources, schedule):
    """Solve each chunk's triangular system with the delayed Neumann pipeline."""
    subblock_idx = get_subblock_id()
    block_idx = get_block_idx()
    if schedule.tasks.start < schedule.tasks.end:
        matmul_ops = StageOneMatmul(head_dim=resources.head_dim)
        vector = StageOneVector(head_dim=resources.head_dim, scale_value=resources.scale, subblock_idx=subblock_idx)
        # Publish the immutable identity once before every delayed consumer.
        identity = matmul_ops.prepare_packed64_identity_l1(resources.identity)
        runtime = _StageOneRuntime(
            matmul=matmul_ops,
            vector=vector,
            identity=identity,
            subblock=subblock_idx,
            head_dim=resources.head_dim,
            block=block_idx,
        )
        _run_stage1_tasks(runtime, inputs, outputs, resources, schedule)


class StageTwoMatmul:
    """FP32 chunk products and rank-local affine recurrence on Cube."""

    def __init__(self, *, head_dim: int):
        self.head_dim = head_dim
        # Chunk products and recurrent products share FP32 storage and dtype.
        self.l0a = RingBuffer(MemLoc.L0A, dtypes.float32, make_layout((head_dim, head_dim)), depth=1).next_write()
        self.l0b = RingBuffer(MemLoc.L0B, dtypes.float32, make_layout((head_dim, head_dim)), depth=1).next_write()
        self.l0a_chunk = reinterpret(self.l0a, layout=make_layout((CHUNK_SIZE, head_dim)), data_format="zn")
        self.l0b_chunk = reinterpret(self.l0b, layout=make_layout((CHUNK_SIZE, head_dim)), data_format="zn")
        self.l0c_phi = RingBuffer(MemLoc.L0C, dtypes.float32, make_layout((head_dim, head_dim)), depth=1).next_write()
        self.l0c_psi = RingBuffer(MemLoc.L0C, dtypes.float32, make_layout((head_dim, head_dim)), depth=1).next_write()

    @staticmethod
    def store_summary_to_l1(l1_dst, l0c_src):
        mem_copy(l1_dst, l0c_src)

    def compute_chunk_product(self, lhs_l1, kg_l1, l0c_dst):
        """Compute ``lhs.T @ kg`` from FP32 [64, D] operands."""
        mem_copy(self.l0a_chunk, lhs_l1)
        mem_copy(self.l0b_chunk, kg_l1)
        matmul(l0c_dst, self.l0a_chunk.permute(1, 0), self.l0b_chunk, init=True)

    def compute_rank_product(self, rank_l1, phi_l1, l0c_dst, *, init: bool):
        """Compute ``rank @ phi`` with FP32 operands and accumulation."""
        mem_copy(self.l0a, rank_l1)
        mem_copy(self.l0b, phi_l1.permute(1, 0))
        matmul(l0c_dst, self.l0a, self.l0b, init=init)

    def store_summary_to_ub(self, output_ub, l0c_src):
        mem_copy(output_ub, l0c_src.partition(dim=0))

    def load_summary(self, l1_dst, gm_src):
        mem_copy(l1_dst, gm_src)


class StageTwoVector:
    """Vector side of the CP summary diagonal update and L1 publication."""

    def __init__(self, *, head_dim: int):
        self.head_dim = head_dim
        self.half_dk = head_dim // 2
        self.gamma_ub = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((1, self.half_dk)), depth=1).next_write()
        self.phi_out_ub = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((self.half_dk, head_dim)), depth=1).next_write()
        self.phi_nz_ub = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((self.half_dk, head_dim)), depth=1, data_format='nz').next_write()
        self.operand_bf16_ub = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((HALF_CHUNK_SIZE, head_dim)), depth=1).next_write()
        self.operand_fp32_ub = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((HALF_CHUNK_SIZE, head_dim)), depth=1).next_write()
        self.operand_nz_ub = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((HALF_CHUNK_SIZE, head_dim)), depth=1, data_format='nz').next_write()

    @jit
    def load_chunk_operand(self, gm_tile, operand_l1, *, subblock_idx: int, valid_rows):
        """Promote each fixed 32-row half to FP32 and zero its invalid rows."""
        operand_bf16 = self.operand_bf16_ub
        with vf(mode="simd"):
            row_mask = full_mask(elem_bits=16)
            zero = vdups(0.0, dtypes.bfloat16, mask=row_mask)
            for row in range(HALF_CHUNK_SIZE):
                vstore(operand_bf16, row * self.head_dim, zero, row_mask)
            vmem_bar("vst_vld")
        if valid_rows > subblock_idx * HALF_CHUNK_SIZE:
            operand_half = tile_slice(gm_tile, (HALF_CHUNK_SIZE, self.head_dim), (subblock_idx, 0))
            mem_copy(operand_bf16, operand_half)
        cast_tile(self.operand_fp32_ub, operand_bf16)
        nd2nz_tile(self.operand_nz_ub, self.operand_fp32_ub, 32)
        mem_copy(operand_l1.partition(dim=0, part_id=subblock_idx), self.operand_nz_ub)

    @jit
    def write_crosscore_matrix_to_gm(self, source_ub, gm_matrix, *, subblock_idx: int, state_index: int):
        if subblock_idx == 0:
            self._write_static_half_to_gm(source_ub, gm_matrix, half_index=0, state_index=state_index)
        else:
            self._write_static_half_to_gm(source_ub, gm_matrix, half_index=1, state_index=state_index)

    @jit
    def add_decay_and_publish_phi(self, phi_ub, decay_tile, destination, publication):
        if publication.half == 0:
            self._add_decay_static_half(phi_ub, decay_tile, half_index=0)
            self._publish_static_half(
                phi_ub,
                destination,
                _PhiPublication(
                    half=0,
                    write_transition=publication.write_transition,
                    publish_phi=publication.publish_phi,
                    initialize_rank=publication.initialize_rank,
                    state=publication.state,
                ),
            )
        else:
            self._add_decay_static_half(phi_ub, decay_tile, half_index=1)
            self._publish_static_half(
                phi_ub,
                destination,
                _PhiPublication(
                    half=1,
                    write_transition=publication.write_transition,
                    publish_phi=publication.publish_phi,
                    initialize_rank=publication.initialize_rank,
                    state=publication.state,
                ),
            )

    @jit
    def _add_decay_static_half(self, phi_ub, gamma_full, *, half_index: int):
        """Add one fixed 64-row diagonal block with aligned Phi loads."""
        gamma_half = tile_slice(gamma_full, (1, self.half_dk), (0, half_index))
        gamma_slot = self.gamma_ub
        mem_copy(gamma_slot, gamma_half)
        vec_sync_notify(PIPE.MTE2, PIPE.V, SUMMARY_MTE2_TO_V_EVENT_ID)
        vec_sync_wait(PIPE.MTE2, PIPE.V, SUMMARY_MTE2_TO_V_EVENT_ID)
        with vf(mode="simd"):
            full_rows = full_mask()
            for local_row in range(self.half_dk):
                row_offset = local_row * self.head_dim + (half_index * self.half_dk)
                original = vload(phi_ub, row_offset)
                decay_value = vload_broadcast(gamma_slot, local_row)
                incremented = vadd(original, decay_value, mask=full_rows)
                through_diagonal, _ = update_mask(local_row + 1, elem_bits=32)
                before_diagonal, _ = update_mask(local_row, elem_bits=32)
                changed_prefix = vselect_raw(incremented, original, cond_mask=through_diagonal)
                diagonal_only = vselect_raw(original, changed_prefix, cond_mask=before_diagonal)
                vstore(phi_ub, row_offset, diagonal_only, full_rows)
            vmem_bar("vst_vld")

    @jit
    def _copy_crosscore_to_output_ub(self, phi_ub):
        """Materialize a CrossCore slab before the single-chunk GM write."""
        phi_out_slot = self.phi_out_ub
        with vf(mode="simd"):
            lane_mask, _ = update_mask(32, elem_bits=32)
            for row in dsl_range(0, self.half_dk, 1, unroll=4):
                row_offset = row * self.head_dim
                for col in dsl_range(0, self.head_dim, 32, unroll=4):
                    offset = row_offset + col
                    value = vload(phi_ub, offset)
                    vstore(phi_out_slot, offset, value, lane_mask)
            vmem_bar("vst_vld")
        return phi_out_slot

    @jit
    def _write_static_half_to_gm(self, source_ub, gm_matrix, *, half_index: int, state_index: int):
        local_phi = self._copy_crosscore_to_output_ub(source_ub)
        matrix_full = _state_head_tile(gm_matrix, state_index)
        matrix_half = tile_slice(matrix_full, (self.half_dk, self.head_dim), (half_index, 0))
        mem_copy(matrix_half, local_phi)

    @jit
    def _publish_static_half(self, phi_ub, destination, publication):
        """Publish one compile-time-sized 128x64 half of Phi."""
        if const_expr(publication.write_transition):
            phi_out_slot = self._copy_crosscore_to_output_ub(phi_ub)
            transition_full = _state_head_tile(destination.transition, publication.state)
            transition_half = tile_slice(transition_full, (self.half_dk, self.head_dim), (publication.half, 0))
            mem_copy(transition_half, phi_out_slot)
        else:
            phi_nz_slot = self.phi_nz_ub
            nd2nz_tile(phi_nz_slot, phi_ub, 32)
            if const_expr(publication.publish_phi):
                mem_copy(destination.phi.partition(dim=0, part_id=publication.half), phi_nz_slot)
            if const_expr(publication.initialize_rank):
                mem_copy(destination.rank.partition(dim=0, part_id=publication.half), phi_nz_slot)


@jit
def _run_cp_prepare_summary_body(head_dim, inputs, outputs, schedule, sequence):
    if schedule.item_start < schedule.item_end:
        _run_cp_prepare_summary_active_body(
            head_dim,
            inputs,
            outputs,
            schedule,
            sequence,
        )


@jit
def _run_cp_prepare_summary_active_body(head_dim, inputs, outputs, schedule, sequence):
    subblock_idx = get_subblock_id()

    workspace = _allocate_summary_workspace(head_dim)

    group_end = schedule.group_start + schedule.group_count
    for local_head_idx in range(schedule.item_start, schedule.item_end):
        head_idx = schedule.head_base + local_head_idx
        state_index = sequence.logical_batch * schedule.head_count + head_idx
        transition_tile = tile_slice(
            outputs.transition[sequence.logical_batch, head_idx, None, None], (head_dim, head_dim), (0, 0)
        )
        state_ext_tile = tile_slice(
            outputs.state[sequence.logical_batch, head_idx, None, None], (head_dim, head_dim), (0, 0)
        )
        if schedule.group_start > 0:
            workspace.matmul.load_summary(workspace.loaded_phi, transition_tile)
            workspace.matmul.load_summary(workspace.loaded_psi, state_ext_tile)

        for chunk_id in range(schedule.group_start, group_end):
            kg_tile = _sequence_head_chunk_tile(inputs.kg, sequence.storage_batch, head_idx, chunk_id)
            u_tile = _sequence_head_chunk_tile(inputs.u, sequence.storage_batch, head_idx, chunk_id)
            w_tile = _sequence_head_chunk_tile(inputs.w, sequence.storage_batch, head_idx, chunk_id)
            decay_tile = tile_slice(
                inputs.decay[sequence.logical_batch, head_idx, None, None], (1, head_dim), (chunk_id, 0)
            )
            valid_rows = min(CHUNK_SIZE, sequence.length - chunk_id * CHUNK_SIZE)
            workspace.vector.load_chunk_operand(kg_tile, workspace.kg, subblock_idx=subblock_idx, valid_rows=valid_rows)
            workspace.vector.load_chunk_operand(u_tile, workspace.u, subblock_idx=subblock_idx, valid_rows=valid_rows)
            workspace.vector.load_chunk_operand(w_tile, workspace.w, subblock_idx=subblock_idx, valid_rows=valid_rows)

            is_group_first, is_group_last = _prepare_summary_phi(
                workspace,
                decay_tile,
                outputs.transition,
                _SummaryChunk(
                    subblock=subblock_idx,
                    state=state_index,
                    group_start=schedule.group_start,
                    index=chunk_id,
                    group_end=group_end,
                ),
            )

            _compose_summary_chunk(
                workspace,
                _SummaryChunk(group_start=schedule.group_start, index=chunk_id, first=is_group_first),
            )

            _store_summary_chunk(
                workspace,
                outputs.state,
                _SummaryChunk(
                    subblock=subblock_idx,
                    state=state_index,
                    group_start=schedule.group_start,
                    first=is_group_first,
                    last=is_group_last,
                ),
                transition_tile,
            )


# mode=0 below is a whole-grid barrier.  The launch must fit in one resident
# cube-core wave, otherwise resident blocks wait for blocks that cannot be
# scheduled.  Keep 32 as the algorithmic cap and select the actual launch size
# from the target device on the host.
ALGORITHM_MAX_BLOCK_NUM = 32

STAGE1_BOUNDARY_FLAG_ID = 0
STAGE2_BOUNDARY_FLAG_ID = 1
_STAGE_BOUNDARY_AIC_FLAG_OFFSET = 4
_STAGE_BOUNDARY_AIC_AIV_FLAG_OFFSET = 8
QK_PUBLISH_FLAG_ID = 10
QK_CONSUME_FLAG_ID = 11
MTE2_TO_V_EVENT_ID = 0
TAIL_V_TO_MTE2_EVENT_ID = 0
MTE2_TO_V_V_STAGING_EVENT_ID = 1
MTE2_TO_V_SUBBLOCK1_EVENT_ID = 4
QK_V_TO_MTE3_EVENT_ID = 2
QK_MTE2_TO_V_EVENT_ID = 3


def _rewind_buffers():
    onchip_rewind(reset_sync_id=False)


def _stage_boundary_reset(flag_id: int):
    cube_sync_all()
    vec_sync_all()

    aiv_ready_flag_id = flag_id
    aic_ready_flag_id = flag_id + _STAGE_BOUNDARY_AIC_FLAG_OFFSET
    aic_aiv_ready_flag_id = flag_id + _STAGE_BOUNDARY_AIC_AIV_FLAG_OFFSET

    vec_sync_block_arrive(PIPE.MTE3, aiv_ready_flag_id, mode=2)
    cube_sync_block_wait(PIPE.S, aiv_ready_flag_id, mode=2)

    cube_sync_block_arrive(PIPE.FIXPIPE, aic_ready_flag_id, mode=0)
    cube_sync_block_wait(PIPE.S, aic_ready_flag_id, mode=0)

    cube_sync_block_arrive(PIPE.MTE3, aic_aiv_ready_flag_id, mode=2)
    vec_sync_block_wait(PIPE.S, aic_aiv_ready_flag_id, mode=2)
    _rewind_buffers()


@kernel(
    profile=cannbotdsl.ProfileSpec(
        name="flash_kda_cp_preprocess",
        op_type="FlashKDACPPreprocess",
        inputs=("gm_K", "gm_V", "gm_Q", "gm_beta_brc", "gm_g", "gm_A_log", "gm_dt_bias", "gm_metadata"),
        outputs=(
            "gm_transition",
            "gm_state_ext",
            "gm_gamma",
            "gm_q_decayed",
            "gm_k_restored",
            "gm_U",
            "gm_w",
            "gm_Mqk",
        ),
        inouts=(),
    )
)
class FlashKdaCpPreprocessKernel:
    """Run CP preprocess StageOne and StageTwo in one kernel launch."""

    def __init__(self, head_dim: int = SUPPORTED_HEAD_DIM):
        self.head_dim = head_dim

    def __call__(
        self,
        gm_K: Tensor,
        gm_V: Tensor,
        gm_Q: Tensor,
        gm_beta_brc: Tensor,
        gm_g: Tensor,
        gm_qk_exchange: Tensor,
        gm_A_log: Tensor,
        gm_dt_bias: Tensor,
        gm_k_restored: Tensor,
        gm_gamma: Tensor,
        gm_q_decayed: Tensor,
        gm_w: Tensor,
        gm_Mqk: Tensor,
        gm_U: Tensor,
        gm_transition: Tensor,
        gm_state_ext: Tensor,
        gm_identity16: Tensor,
        gm_metadata: Tensor,
        bn_total: int,
        seq_len: int,
        scale_value: float,
        lower_bound: float,
    ):
        block_idx = get_block_idx()
        block_num = get_block_num()
        metadata_status = gm_metadata[HEADER_STATUS]
        metadata_core_num = gm_metadata[HEADER_CORE_NUM]
        metadata_batch = dtypes.int64(gm_metadata[HEADER_BATCH])
        metadata_value_heads = dtypes.int64(gm_metadata[HEADER_VALUE_HEADS])
        if (
            gm_metadata[HEADER_MAGIC] == MAGIC
            and gm_metadata[HEADER_ABI_VERSION] == ABI_VERSION
            and metadata_status == STATUS_OK
            and metadata_core_num == block_num
            and metadata_value_heads == gm_V.shape[1]
            and metadata_batch * metadata_value_heads == bn_total
        ):
            round_num = gm_metadata[HEADER_STAGE12_ROUND_NUM]
            for round_idx in range(0, round_num):
                (
                    active_batch_num,
                    batch_idx_base,
                    storage_batch_idx_base,
                    token_start_base,
                    valid_seq_len_base,
                    chunk_start_base,
                    chunk_num_base,
                    stage1_task_prefix_base,
                    stage1_core_ranges_base,
                    stage1_core_start,
                    stage1_core_end,
                ) = _decode_stage1_round(gm_metadata, round_idx, block_idx)

                _run_stage1_round(
                    _StageOneInputs(k=gm_K, v=gm_V, q=gm_Q, beta=gm_beta_brc, g=gm_g),
                    _StageOneOutputs(qg=gm_q_decayed, kg=gm_k_restored, u=gm_U, w=gm_w, decay=gm_gamma, mqk=gm_Mqk),
                    _StageOneResources(
                        exchange=gm_qk_exchange,
                        identity=gm_identity16,
                        head_dim=self.head_dim,
                        scale=scale_value,
                        gate=_GateParameters(alpha=gm_A_log, bias=gm_dt_bias, lower_bound=lower_bound),
                    ),
                    gm_metadata,
                    _RoundSchedule(
                        heads=metadata_value_heads,
                        batch_count=active_batch_num,
                        batch_base=batch_idx_base,
                        storage_base=storage_batch_idx_base,
                        token_base=token_start_base,
                        length_base=valid_seq_len_base,
                        chunk_start_base=chunk_start_base,
                        chunk_count_base=chunk_num_base,
                        task_prefix_base=stage1_task_prefix_base,
                        core_start=stage1_core_start,
                        core_end=stage1_core_end,
                    ),
                )
                _stage_boundary_reset(STAGE1_BOUNDARY_FLAG_ID)

                dv_splits_base, stage2_task_prefix_base, stage2_core_start, stage2_core_end = _decode_summary_round(
                    gm_metadata, stage1_core_ranges_base, active_batch_num, block_idx
                )

                _run_summary_round(
                    self.head_dim,
                    _SummaryInputs(kg=gm_k_restored, u=gm_U, w=gm_w, decay=gm_gamma),
                    _SummaryOutputs(transition=gm_transition, state=gm_state_ext),
                    gm_metadata,
                    _RoundSchedule(
                        heads=metadata_value_heads,
                        batch_count=active_batch_num,
                        batch_base=batch_idx_base,
                        storage_base=storage_batch_idx_base,
                        token_base=token_start_base,
                        length_base=valid_seq_len_base,
                        chunk_start_base=chunk_start_base,
                        chunk_count_base=chunk_num_base,
                        splits_base=dv_splits_base,
                        task_prefix_base=stage2_task_prefix_base,
                        core_start=stage2_core_start,
                        core_end=stage2_core_end,
                    ),
                )
                has_next_group = round_idx + 1 < round_num
                if has_next_group:
                    _stage_boundary_reset(STAGE2_BOUNDARY_FLAG_ID)


flash_kda_cp_preprocess_kernel = FlashKdaCpPreprocessKernel


class FlashKDACPPreprocess:
    """Host/JIT wrapper for the single-launch CP preprocess kernel."""

    def __init__(self, head_dim: int = SUPPORTED_HEAD_DIM):
        self.head_dim = head_dim

    @host
    def run(
        self,
        K: Tensor,
        V: Tensor,
        Q: Tensor,
        beta: Tensor,
        g: Tensor,
        qk_exchange: Tensor,
        A_log: Tensor,
        dt_bias: Tensor,
        kg: Tensor,
        decay: Tensor,
        qg: Tensor,
        w: Tensor,
        mqk: Tensor,
        u: Tensor,
        transition: Tensor,
        state_ext: Tensor,
        identity16: Tensor,
        metadata: Tensor,
        bn_total: int,
        seq_len: int,
        scale_value: float,
        lower_bound: float,
        core_num: int,
    ):
        op = FlashKdaCpPreprocessKernel(self.head_dim)
        op[core_num](
            K,
            V,
            Q,
            beta,
            g,
            qk_exchange,
            A_log,
            dt_bias,
            kg,
            decay,
            qg,
            w,
            mqk,
            u,
            transition,
            state_ext,
            identity16,
            metadata,
            bn_total,
            seq_len,
            scale_value,
            lower_bound,
        )


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


def _cp_sequence_tensor_spec(dtype, shape, layout_qkv, prefix):
    """Dynamic CP input/output ABI with runtime physical strides."""
    if layout_qkv == "BNSD":
        shape = (shape.batch, shape.heads, shape.length, shape.width)
    else:
        shape = (shape.batch, shape.length, shape.heads, shape.width)
    return cannbotdsl.TensorSpec(
        shape,
        dtype,
        stride=(
            cannbotdsl.Dim(f"{prefix}_stride_0"),
            cannbotdsl.Dim(f"{prefix}_stride_1"),
            cannbotdsl.Dim(f"{prefix}_stride_2"),
            1,
        ),
    )


@compile_cache
def _compile_cp_preprocess_kernel():
    physical_batch = cannbotdsl.Dim("P")
    batch = cannbotdsl.Dim("B")
    seq_len = cannbotdsl.Dim("S")
    qk_heads = cannbotdsl.Dim("NQK")
    value_heads = cannbotdsl.Dim("NV")
    metadata_capacity = cannbotdsl.Dim("META")
    core_num = cannbotdsl.Dim("CORE_NUM", min=1, max=ALGORITHM_MAX_BLOCK_NUM)
    # Keep one ABI for BNSD, BSND and TND.  The extra ``B - 1`` slots
    # cover one tail chunk per additional logical sequence; metadata carries
    # the exact valid chunk count for each sequence.
    chunks = _chunk_capacity(seq_len, batch)
    fake = cannbotdsl.TensorSpec
    return cannbotdsl.compile(
        FlashKDACPPreprocess().run,
        _cp_sequence_tensor_spec(
            dtypes.bfloat16,
            _SequenceShape(batch=physical_batch, length=seq_len, heads=qk_heads, width=SUPPORTED_HEAD_DIM),
            "BNSD",
            "K",
        ),
        _cp_sequence_tensor_spec(
            dtypes.bfloat16,
            _SequenceShape(batch=physical_batch, length=seq_len, heads=value_heads, width=SUPPORTED_HEAD_DIM),
            "BNSD",
            "V",
        ),
        _cp_sequence_tensor_spec(
            dtypes.bfloat16,
            _SequenceShape(batch=physical_batch, length=seq_len, heads=qk_heads, width=SUPPORTED_HEAD_DIM),
            "BNSD",
            "Q",
        ),
        _cp_sequence_tensor_spec(
            dtypes.bfloat16,
            _SequenceShape(batch=physical_batch, length=seq_len, heads=value_heads, width=1),
            "BNSD",
            "beta",
        ),
        _cp_sequence_tensor_spec(
            dtypes.bfloat16,
            _SequenceShape(batch=physical_batch, length=seq_len, heads=value_heads, width=SUPPORTED_HEAD_DIM),
            "BNSD",
            "g",
        ),
        fake((core_num, 2, CHUNK_SIZE, D_HALF), dtypes.float32),
        fake((value_heads,), dtypes.float32),
        fake((value_heads, SUPPORTED_HEAD_DIM), dtypes.float32),
        _cp_sequence_tensor_spec(
            dtypes.bfloat16,
            _SequenceShape(batch=physical_batch, length=seq_len, heads=value_heads, width=SUPPORTED_HEAD_DIM),
            "BNSD",
            "KG",
        ),
        fake((batch, value_heads, chunks, SUPPORTED_HEAD_DIM), dtypes.float32),
        _cp_sequence_tensor_spec(
            dtypes.bfloat16,
            _SequenceShape(batch=physical_batch, length=seq_len, heads=value_heads, width=SUPPORTED_HEAD_DIM),
            "BNSD",
            "QG",
        ),
        _cp_sequence_tensor_spec(
            dtypes.bfloat16,
            _SequenceShape(batch=physical_batch, length=seq_len, heads=value_heads, width=SUPPORTED_HEAD_DIM),
            "BNSD",
            "W",
        ),
        fake((batch, value_heads, chunks, CHUNK_SIZE, CHUNK_SIZE), dtypes.bfloat16),
        _cp_sequence_tensor_spec(
            dtypes.bfloat16,
            _SequenceShape(batch=physical_batch, length=seq_len, heads=value_heads, width=SUPPORTED_HEAD_DIM),
            "BNSD",
            "U",
        ),
        fake((batch, value_heads, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), dtypes.float32),
        fake((batch, value_heads, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), dtypes.float32),
        fake((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), dtypes.float16),
        fake((metadata_capacity,), dtypes.int32),
        dtypes.int64,
        dtypes.int64,
        dtypes.float32,
        dtypes.float32,
        dtypes.int64,
    )


def _validate_preprocess_layout(inputs, layout_qkv, cu_seqlens):
    if layout_qkv not in ("BNSD", "BSND", "TND"):
        raise ValueError("layout_qkv must be BNSD, BSND, or TND")
    if layout_qkv == "TND":
        if cu_seqlens is None:
            raise ValueError("cu_seqlens is required for TND")
        if not (cu_seqlens.dtype == torch.int32 and cu_seqlens.dim() == 1):
            raise ValueError("cu_seqlens must be rank-1 int32")
        if not (cu_seqlens.is_contiguous() and cu_seqlens.device == inputs.q.device):
            raise ValueError("cu_seqlens must be contiguous and on the same device")
        if not (cu_seqlens.numel() >= 2):
            raise ValueError("cu_seqlens must have shape [B+1] with positive batch")
        if not (int(cu_seqlens[0].item()) == 0 and int(cu_seqlens[-1].item()) == inputs.q.shape[0]):
            raise ValueError("cu_seqlens endpoints must span the packed tokens")
        batch = cu_seqlens.numel() - 1
    elif cu_seqlens is not None:
        raise ValueError("cu_seqlens is only valid for TND")
    if layout_qkv == "TND":
        if not (inputs.q.dim() == 3 and inputs.k.dim() == 3 and inputs.v.dim() == 3 and inputs.g.dim() == 3):
            raise ValueError("TND q, k, v, and g must be rank-3")
        seq_len, n_kv, dim = inputs.q.shape[0], inputs.q.shape[1], inputs.q.shape[2]
        if not (inputs.k.shape == inputs.q.shape and inputs.v.shape[0] == seq_len and inputs.g.shape == inputs.v.shape):
            raise ValueError("TND k must match q and v/g must match the packed token length")
        n_v = inputs.v.shape[1]
        if not (inputs.beta.dim() == 2 and inputs.beta.shape == (seq_len, n_v)):
            raise ValueError("TND beta shape must be [T,Nv]")
    else:
        if not (inputs.q.dim() == 4 and inputs.k.dim() == 4 and inputs.v.dim() == 4 and inputs.g.dim() == 4):
            raise ValueError("BNSD/BSND q, k, v, and g must be rank-4")
        batch, dim = inputs.q.shape[0], inputs.q.shape[3]
        seq_len = inputs.q.shape[1] if layout_qkv == "BSND" else inputs.q.shape[2]
        n_v = inputs.v.shape[2] if layout_qkv == "BSND" else inputs.v.shape[1]
        n_kv = inputs.q.shape[2] if layout_qkv == "BSND" else inputs.q.shape[1]

    return batch, seq_len, n_kv, n_v, dim


def _validate_preprocess_parameters(inputs, gate, metadata, shape):
    if not (shape.width == SUPPORTED_HEAD_DIM and shape.length > 0):
        raise ValueError("head dimension requires D=128 and sequence length must be positive")
    if not (shape.key_heads > 0 and shape.heads > 0 and shape.heads % shape.key_heads == 0):
        raise ValueError("GQA requires Nv % Nk == 0")
    if not (inputs.q.dtype == inputs.k.dtype == inputs.v.dtype == inputs.g.dtype == LOW_DTYPE):
        raise ValueError("q, k, v, and g must be bf16")
    if not (inputs.beta.dtype == LOW_DTYPE and inputs.beta.dim() in (2, 3)):
        raise ValueError("beta must be bf16 with rank 2 or 3")
    if not (gate.alpha.dtype == HIGH_DTYPE and gate.bias.dtype == HIGH_DTYPE):
        raise ValueError("A_log and dt_bias must be fp32")
    if not (tuple(gate.alpha.shape) == (shape.heads,) and tuple(gate.bias.shape) == (shape.heads, shape.width)):
        raise ValueError("A_log/dt_bias shape must match Nv and D=128")
    if not (-5.0 <= float(gate.lower_bound) <= 0.0):
        raise ValueError("lower_bound must be in [-5, 0]")
    tensors = (inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta, gate.alpha, gate.bias)
    if not (all(t.device == inputs.q.device for t in tensors)):
        raise ValueError("all input tensors must be on the same device")
    if not (gate.alpha.is_contiguous() and gate.bias.is_contiguous()):
        raise ValueError("A_log and dt_bias must be contiguous")
    if inputs.q.device.type not in {"npu", "privateuseone"}:
        raise ValueError("FlashKDA CP requires an NPU tensor")
    if not (metadata.dtype == torch.int32):
        raise ValueError("metadata must be int32")
    if not (metadata.dim() == 1):
        raise ValueError("metadata must be rank-1")
    if not (metadata.is_contiguous()):
        raise ValueError("metadata must be contiguous")
    if not (metadata.device == inputs.q.device):
        raise ValueError("metadata must be on the same device")


def _preprocess_sequence_views(inputs, layout_qkv, shape):
    stage1_k, stage1_v, stage1_q, stage1_beta, stage1_g = inputs.k, inputs.v, inputs.q, inputs.beta, inputs.g
    if layout_qkv == "TND":
        # The kernel consumes logical BNSD views. Packed TND uses one physical
        # storage batch; metadata maps each logical sequence to its token
        # interval without copying [T,N,D].
        stage1_k = torch.as_strided(
            inputs.k,
            (1, shape.key_heads, shape.length, shape.width),
            (shape.length * shape.key_heads * shape.width, shape.width, shape.key_heads * shape.width, 1),
        )
        stage1_v = torch.as_strided(
            inputs.v,
            (1, shape.heads, shape.length, shape.width),
            (shape.length * shape.heads * shape.width, shape.width, shape.heads * shape.width, 1),
        )
        stage1_q = torch.as_strided(
            inputs.q,
            (1, shape.key_heads, shape.length, shape.width),
            (shape.length * shape.key_heads * shape.width, shape.width, shape.key_heads * shape.width, 1),
        )
        stage1_g = torch.as_strided(
            inputs.g,
            (1, shape.heads, shape.length, shape.width),
            (shape.length * shape.heads * shape.width, shape.width, shape.heads * shape.width, 1),
        )
        stage1_beta = torch.as_strided(
            inputs.beta, (1, shape.heads, shape.length, 1), (shape.length * shape.heads, 1, shape.heads, 1)
        )
    elif layout_qkv == "BSND":
        # Keep physical BSND storage but present BNSD logical metadata before
        # tracing.  Rewriting a dynamic layout inside JIT leaves AscVec with
        # dynamic logical dimensions, which it cannot lower.
        stage1_k = inputs.k.transpose(1, 2)
        stage1_v = inputs.v.transpose(1, 2)
        stage1_q = inputs.q.transpose(1, 2)
        stage1_beta = inputs.beta.transpose(1, 2)
        stage1_g = inputs.g.transpose(1, 2)
    return stage1_k, stage1_v, stage1_q, stage1_beta, stage1_g


def _allocate_preprocess_outputs(q, shape, layout_qkv):
    transition = torch.empty((shape.batch, shape.heads, shape.width, shape.width), dtype=HIGH_DTYPE, device=q.device)
    state_ext = torch.empty_like(transition)
    decay = torch.empty((shape.batch, shape.heads, shape.chunks, shape.width), dtype=HIGH_DTYPE, device=q.device)
    if layout_qkv == "TND":
        qg_physical = torch.empty((shape.length, shape.heads, shape.width), dtype=LOW_DTYPE, device=q.device)
        kg_physical = torch.empty_like(qg_physical)
        u_physical = torch.empty_like(qg_physical)
        w_physical = torch.empty_like(qg_physical)
        packed_stride = (shape.length * shape.heads * shape.width, shape.width, shape.heads * shape.width, 1)
        qg = torch.as_strided(qg_physical, (1, shape.heads, shape.length, shape.width), packed_stride)
        kg = torch.as_strided(kg_physical, (1, shape.heads, shape.length, shape.width), packed_stride)
        u = torch.as_strided(u_physical, (1, shape.heads, shape.length, shape.width), packed_stride)
        w = torch.as_strided(w_physical, (1, shape.heads, shape.length, shape.width), packed_stride)
    else:
        qg = torch.empty((shape.batch, shape.heads, shape.length, shape.width), dtype=LOW_DTYPE, device=q.device)
        kg = torch.empty_like(qg)
        u = torch.empty_like(qg)
        w = torch.empty_like(qg)
    mqk = torch.empty(
        (shape.batch, shape.heads, shape.chunks, CHUNK_SIZE, CHUNK_SIZE), dtype=LOW_DTYPE, device=q.device
    )
    physical = (qg_physical, kg_physical, u_physical, w_physical) if layout_qkv == "TND" else (qg, kg, u, w)
    return transition, state_ext, decay, qg, kg, u, w, mqk, physical


def flash_kda_cp_preprocess(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    layout_qkv: str,
    metadata: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Prepare chunk tensors and the rank-local affine state summary for CP prefill.

    Dimensions:
        B: Number of logical sequences.
        Nqk: Number of query/key heads.
        Nv: Number of value heads.
        S: Tokens per sequence for dense BNSD/BSND layouts.
        T: Total packed tokens for TND layout.
        D: Query/key and value head dimension, fixed at 128.
        C: Chunk capacity, ceil(S / 64) + B - 1 for dense layouts or ceil(T / 64) + B - 1 for TND.
        M: Metadata word capacity returned by flash_kda_cp_preprocess_metadata.

    Args:
        q (torch.Tensor): Raw Query, bfloat16. BNSD: ``[B, Nqk, S, D]``;
            BSND: ``[B, S, Nqk, D]``; TND: ``[T, Nqk, D]``.
        k (torch.Tensor): Raw Key, bfloat16. BNSD: ``[B, Nqk, S, D]``;
            BSND: ``[B, S, Nqk, D]``; TND: ``[T, Nqk, D]``.
        v (torch.Tensor): Value, bfloat16. BNSD: ``[B, Nv, S, D]``;
            BSND: ``[B, S, Nv, D]``; TND: ``[T, Nv, D]``.
        g (torch.Tensor): Raw per-channel gate input, bfloat16. BNSD: ``[B, Nv, S, D]``;
            BSND: ``[B, S, Nv, D]``; TND: ``[T, Nv, D]``.
        beta (torch.Tensor): Raw per-token write logits, bfloat16. BNSD: ``[B, Nv, S]``;
            BSND: ``[B, S, Nv]``; TND: ``[T, Nv]``.
        scale (float): Query-key scaling factor, conventionally D ** -0.5.
        A_log (torch.Tensor): Per-Value-head log gate parameter, float32, ``[Nv]``.
        dt_bias (torch.Tensor): Per-Value-head gate bias, float32, ``[Nv, D]``.
        lower_bound (float): Activated gate lower bound, in ``[-5, 0]``.
        layout_qkv (str): Input layout: "BNSD", "BSND", or "TND".
        metadata (torch.Tensor): Matching schedule from
            flash_kda_cp_preprocess_metadata, int32, ``[M]``.
        cu_seqlens (torch.Tensor, optional): Packed sequence boundaries, int32,
            ``[B + 1]``. Required for TND and None for BNSD/BSND. Starts at 0,
            ends at T, and increases strictly between nonempty sequences.

    Returns:
        transition (torch.Tensor): Rank-local affine state multiplier, float32, ``[B, Nv, D, D]``.
        state_ext (torch.Tensor): Rank-local additive state from zero initial state, float32, ``[B, Nv, D, D]``.
        decay (torch.Tensor): Per-chunk terminal gate decay, float32, ``[B, Nv, C, D]``.
        qg (torch.Tensor): Decayed and scaled Query, bfloat16. BNSD/BSND inputs: ``[B, Nv, S, D]``;
            TND input: ``[T, Nv, D]``.
        kg (torch.Tensor): Key restored relative to the chunk end, bfloat16.
            BNSD/BSND inputs: ``[B, Nv, S, D]``; TND input: ``[T, Nv, D]``.
        u (torch.Tensor): Corrected Value, bfloat16. BNSD/BSND inputs: ``[B, Nv, S, D]``;
            TND input: ``[T, Nv, D]``.
        w (torch.Tensor): Signed Key correction, bfloat16. BNSD/BSND inputs: ``[B, Nv, S, D]``;
            TND input: ``[T, Nv, D]``. Contains the negative sign used by CP main.
        mqk (torch.Tensor): Causal chunk QK matrix including the diagonal, bfloat16, ``[B, Nv, C, 64, 64]``.

    Notes:
        B >= 1, 1 <= Nqk <= Nv, Nv % Nqk == 0, S > 0 for BNSD/BSND, T > 0 for TND, and D == 128.
        All tensors must share one NPU. q/k/v/g require unit innermost stride;
        A_log, dt_bias, metadata, and cu_seqlens must be contiguous.
        Generate metadata with matching input shapes/layout and the current
        stream's effective core count. The metadata producer requires contiguous q/v.
        qg/kg/u/w use BNSD for both dense input layouts; TND outputs stay packed.
        Each sequence consumes ceil(its_length / 64) chunks. Extra capacity slots
        in decay/mqk are unused. An external CP prefix scan combines transition
        and state_ext with the global initial state before calling CP main.
    """
    batch, seq_len, n_kv, n_v, dim = _validate_preprocess_layout(
        _StageOneInputs(q=q, k=k, v=v, g=g, beta=beta), layout_qkv, cu_seqlens
    )
    _validate_preprocess_parameters(
        _StageOneInputs(q=q, k=k, v=v, g=g, beta=beta),
        _GateParameters(alpha=A_log, bias=dt_bias, lower_bound=lower_bound),
        metadata,
        _SequenceShape(width=dim, length=seq_len, key_heads=n_kv, heads=n_v),
    )

    beta = beta.unsqueeze(-1)
    if not (all(t.stride(-1) == 1 for t in (q, k, v, g, beta))):
        raise ValueError("q, k, v, g, and beta must have unit innermost stride")
    stage1_k, stage1_v, stage1_q, stage1_beta, stage1_g = _preprocess_sequence_views(
        _StageOneInputs(k=k, v=v, q=q, beta=beta, g=g),
        layout_qkv,
        _SequenceShape(key_heads=n_kv, heads=n_v, length=seq_len, width=dim),
    )
    bn_total = batch * n_v
    chunks_total = _chunk_capacity(seq_len, batch)
    block_num = _device_block_num(q)
    constants = _stage1_constants_for(q)
    execution = _CP_EXECUTION_CACHE.get_or_create(ref=q, block_num=block_num)

    transition, state_ext, decay, qg, kg, u, w, mqk, physical = _allocate_preprocess_outputs(
        q, _SequenceShape(batch=batch, heads=n_v, width=dim, length=seq_len, chunks=chunks_total), layout_qkv
    )
    fn = _compile_cp_preprocess_kernel()
    with execution.launch_lock:
        fn(
            stage1_k,
            stage1_v,
            stage1_q,
            stage1_beta,
            stage1_g,
            execution.qk_exchange,
            A_log,
            dt_bias,
            kg,
            decay,
            qg,
            w,
            mqk,
            u,
            transition,
            state_ext,
            constants.identity16,
            metadata,
            bn_total,
            seq_len,
            float(scale),
            float(lower_bound),
            block_num,
        )

    if layout_qkv == "TND":
        return transition, state_ext, decay, *physical, mqk
    return transition, state_ext, decay, qg, kg, u, w, mqk


MAGIC = 0x4B444132
ABI_VERSION = 2
HEADER_MAGIC = 0
HEADER_ABI_VERSION = 1
HEADER_STATUS = 2
HEADER_CORE_NUM = 3
HEADER_STAGE12_ROUND_NUM = 4
HEADER_BATCH = 5
HEADER_VALUE_HEADS = 6
HEADER_HAS_CU = 7
HEADER_WORKSPACE_SLOTS = 8
HEADER_WORDS = 9
MAX_AIC_CORES = ALGORITHM_MAX_BLOCK_NUM
WORKSPACE_SLOTS = 861

STATUS_OK = 0

if aot is not None and hasattr(aot, "export"):

    @aot.export("flash_kda_cp_preprocess")
    def export_flash_kda_cp_preprocess():
        """Export the dynamic CP preprocess AICore binary."""
        _compile_cp_preprocess_kernel()


__all__ = ["FlashKdaCpPreprocessKernel", "flash_kda_cp_preprocess", "flash_kda_cp_preprocess_kernel"]
