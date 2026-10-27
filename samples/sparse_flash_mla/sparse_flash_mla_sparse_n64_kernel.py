# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Metadata-scheduled sparse flash MLA kernels for N1 <= 64."""

import contextlib
import fcntl
import functools
import hashlib
import importlib.metadata
import json
import math
import numbers
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from dataclasses import dataclass
import torch
import cannbotdsl
from cannbotdsl import (
    Dim,
    MemLoc,
    Tensor,
    TensorSpec,
    TensorViewSpec,
    ViewLayout,
    copy_options,
    datastruct,
    dtypes,
    host,
    is_aic,
    load,
)
from cannbotdsl.ops.cube import set_mmad_direction
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.types.delay_line import DelayLineGroup
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.ops.arch import get_subblock_id, get_block_idx
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops import matmul
from cannbotdsl.tensor import reinterpret, tile_slice
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.sync import (
    PIPE,
    onchip_rewind,
    cube_sync_block_arrive,
    cube_sync_intra_wait,
    vec_sync_block_wait,
    vec_sync_intra_arrive,
)
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr

_BITS_PER_BYTE = 8
_F32_BITS = 32
_F16_BITS = 16
_I32_BITS = 32
_VECTOR_REGISTER_BITS = 2048
_F16_BYTES = _F16_BITS // _BITS_PER_BYTE
_F32_BYTES = _F32_BITS // _BITS_PER_BYTE
_I32_BYTES = _I32_BITS // _BITS_PER_BYTE
_HEAD_DIM = 512
_BASE_TILE_SIZE = 128
_QUERY_HEAD_TILE_SIZE = 64
_GATHER_TILE_ROWS = 16
# A 2048-bit vector holds 64 float32 values.
_FP32_VECTOR_LENGTH = _VECTOR_REGISTER_BITS // _F32_BITS
# Fixed 16-bit specialization and Cube/NZ tile dimensions.
_SMALL_QUERY_HEAD_TILE_SIZE = _QUERY_HEAD_TILE_SIZE // 2
_VECTOR_SUBBLOCKS = 2
_NZ_ROW = 16
_NZ_FACE = _NZ_ROW * _NZ_ROW
_NZ_PADDED_ROWS = _GATHER_TILE_ROWS + 1
_NZ_FRAGMENT_ELEMS = _NZ_PADDED_ROWS * _HEAD_DIM
_NZ_FRAGMENT_BYTES = _NZ_FRAGMENT_ELEMS * _F16_BYTES
_NZ_QUARTER_STRIDE = _NZ_PADDED_ROWS * _BASE_TILE_SIZE
# One 32 KiB zero tile covers all heads owned by one Vector subblock.
_ZERO_ATTENTION_HEADS = 32
_UB_VIEW_ALIGNMENT_BYTES = 32
# A DAV-3510 vector register is 2048 bits wide.
_FP16_VECTOR_LENGTH = _VECTOR_REGISTER_BITS // _F16_BITS
_KV_ROWS_PER_VECTOR = _BASE_TILE_SIZE // _VECTOR_SUBBLOCKS
_GATHER_CHUNKS = 2
_GATHER_PAIRS = _GATHER_TILE_ROWS // 2
_GATHER_ELEMS = _GATHER_TILE_ROWS * _HEAD_DIM
_GATHER_BYTES = _GATHER_ELEMS * _F16_BYTES
_RESULT_HALF_COLS = _HEAD_DIM // 2
# UB output reuses FP32 partial-output storage as 16-bit data, preserving row pitch.
_OUT_UB_ROW_ELEMS = _HEAD_DIM * _F32_BYTES // _F16_BYTES
_Q_WIDE_COLS = 3 * _BASE_TILE_SIZE
_Q_SLOT_WIDTHS = (_Q_WIDE_COLS, _Q_WIDE_COLS, _BASE_TILE_SIZE)
_Q_L1_RESERVED_BYTES = 56 * 1024
_P_L1_RESERVED_BYTES = 8 * 1024
_KV_SLOT_ELEMS = _KV_ROWS_PER_VECTOR * _HEAD_DIM
_KV_RING_DEPTH = 3
_L0A_RING_DEPTH = 4
_L0B_SLOTS = 4
_D_F32_TILES = _HEAD_DIM // _FP32_VECTOR_LENGTH
_WIDE_F32_TILES = _Q_WIDE_COLS // _FP32_VECTOR_LENGTH
_F32_NZ_TILES = _FP32_VECTOR_LENGTH // _NZ_ROW
_TILE_NZ_TILES = _BASE_TILE_SIZE // _NZ_ROW
# Physical memory arenas for this kernel; their used ranges are checked below.
_UB_ARENA_BYTES = 256 * 1024
_L0C_ARENA_BYTES = 256 * 1024
_INT32_LIMIT = 1 << 31
# Index load at tick 0; gather at 1; softmax at 3; output update at 4.
# Five delay-line slots retain all in-flight tiles, including the drain tick.
_VEC0_PIPELINE_DELAY = 1
_VEC1_PIPELINE_DELAY = 3
_VEC2_PIPELINE_DELAY = 4
_PIPELINE_SLOTS = _VEC2_PIPELINE_DELAY + 1
# Each slot stores pipeline metadata in the order declared by the delay-line fields.
_PIPELINE_FIELDS_PER_SLOT = 14
_PIPELINE_METADATA_SIZE = _PIPELINE_SLOTS * _PIPELINE_FIELDS_PER_SLOT
_F32_NEG_INF_BITS = 0xFF800000
_F32_POS_INF_BITS = 0x7F800000
_F32_MIN = -3.4028234663852886e38
# Host tensor ranks and public mask-mode codes.
_TND_RANK = 3
_BSND_RANK = 4
_BATCH_MODE_SINGLE = 0
_BATCH_MODE_BSND = 1
_BATCH_MODE_TND = 2
_KV_MODE_LINEAR = 0
_KV_MODE_PA = 1
_MASK_CAUSAL = 3
_MASK_SLIDING_WINDOW = 4
_METADATA_STRIDE = 9
_METADATA_LAST_BATCH = 4
_METADATA_LAST_QUERY = 5
# Storage alignment, cache sizing, and device instruction options.
_UB_CAPACITY_ALIGNMENT_BYTES = 8
_COMPILED_VARIANT_CACHE_SIZE = 8
_BUILD_HASH_PREFIX_LENGTH = 16
_KV_COPY_L2_CACHE_CONTROL = 4
_MMAD_FINAL_UNIT_FLAG = 3
# PA physical-row prepass owns UB before the attention pipeline is allocated.
# The three fixed regions are reused by ORI and CMP, then released by rewind.
_PA_BLOCK_TABLE_CAPACITY = 14336
_PA_INDEX_CAPACITY = 24576
_PA_PHYSICAL_ROW_CAPACITY = 24576
_PA_PREPASS_WORDS = (
    _PA_BLOCK_TABLE_CAPACITY + _PA_INDEX_CAPACITY + _PA_PHYSICAL_ROW_CAPACITY
)
_PA_BATCH_ROWS = 6
_PA_QUERY_GROUP_SIZE = 3
_PA_BUFFER_COUNT = 2
_PA_BUFFER_CAPACITY = _PA_INDEX_CAPACITY // _PA_BUFFER_COUNT
_PA_SYNC_ID = 9
_PA_SECOND_VECTOR_SYNC_OFFSET = 16


_COMPILER_FLAGS = (
    "-Ofast",
    "-mllvm",
    "-enable-hiipu-vf-loop-unroll",
    "-mllvm",
    "-cce-aicore-dcpreload-args=false",
    "--cce-res-usage",
)
_COMPILER_WRAPPER = (
    "import sys, subprocess\nreal, flags = _compiler_config\nargs = sys"
    ".argv[1:]\nif '-c' in args and '-x' in args and 'asc' in args:\n  "
    "  args = list(flags) + [x for x in args if x != '-O3']\nsys.exit("
    "subprocess.run([real] + args).returncode)\n"
)

_COMPILATION_LOCK = threading.RLock()

_STORAGE_DTYPE_LABELS = {
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
}
_DSL_DTYPE_BY_LABEL = {
    "bf16": dtypes.bfloat16,
    "fp16": dtypes.float16,
}
_BATCH_MODE_BY_LABEL = {
    "single": _BATCH_MODE_SINGLE,
    "bsnd": _BATCH_MODE_BSND,
    "tnd": _BATCH_MODE_TND,
}
_KV_MODE_BY_LABEL = {
    "linear": _KV_MODE_LINEAR,
    "pa": _KV_MODE_PA,
}


def _storage_dtype_label(dtype):
    try:
        return _STORAGE_DTYPE_LABELS[dtype]
    except KeyError as error:
        raise ValueError("FP16/BF16 query and KV dtypes are required") from error


@jit
def scalar_select(condition, yes, no):
    return dtypes.int64(yes) if condition else dtypes.int64(no)


def view_nz(tensor, rows, cols):
    return reinterpret(tensor, shape=(rows, cols), data_format="nz")


def get_nz_column_tile(tensor, rows, column_tile):
    matrix = view_nz(tensor, rows, math.prod(tensor.spec.shape_bound) // rows)
    return tile_slice(matrix, (rows, _FP32_VECTOR_LENGTH), (0, column_tile))


def get_bmm1_operand_half(tensor, rows, row_group_idx):
    # QK joins two NZ row groups in one operand buffer.
    return reinterpret(
        tensor,
        shape=(rows, _FP32_VECTOR_LENGTH),
        offset=row_group_idx * rows * _NZ_ROW * _F16_BYTES,
    )


def get_bmm2_operand_half(tensor, kv_row_group_idx):
    return reinterpret(
        tensor,
        shape=(_FP32_VECTOR_LENGTH, _KV_ROWS_PER_VECTOR),
        offset=kv_row_group_idx
        * _FP32_VECTOR_LENGTH
        * _KV_ROWS_PER_VECTOR
        * _F16_BYTES,
        data_format="nz",
    )


def view_padded_nz_columns(tensor, columns, rows, row_capacity):
    return tensor.view(columns // _NZ_ROW, row_capacity * _NZ_ROW)[
        None, 0: rows * _NZ_ROW
    ]  # fmt: skip


def _restore_compilation_environment(old):
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@contextlib.contextmanager
def _compilation_environment(output):
    """Use the public compiler override; do not patch DSL internals."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    real_compiler = (
        os.environ.get("CANNBOTDSL_BISHENG_PATH")
        or os.environ.get("CANNBOTDSL_BISHENG")
        or shutil.which("bisheng")
    )
    if not real_compiler:
        raise RuntimeError("Source the CANN environment before compiling")
    real_compiler = str(Path(real_compiler).resolve())
    wrapper = output / "bisheng-wrapper"
    wrapper_code = _COMPILER_WRAPPER
    wrapper_code = wrapper_code.replace(
        "_compiler_config", repr((real_compiler, _COMPILER_FLAGS)), 1
    )
    wrapper.write_text("#!" + sys.executable + "\n" + wrapper_code)
    wrapper.chmod(0o755)
    settings = {
        "CANNBOTDSL_BISHENG_PATH": str(wrapper),
        "CANNBOTDSL_NPU_ARCH": "dav-3510",
        "CANNBOTDSL_AUTO_BUFFER_SYNC": "1",
        "CANNBOTDSL_BARALL": "0",
        "CANNBOTDSL_DEBUG_ASC_SYNC": "",
        "CANNBOTDSL_CACHE_MEM_DISABLE": "1",
        "CANNBOTDSL_CACHE_DISK_DISABLE": "1",
    }
    with _COMPILATION_LOCK:
        old = {key: os.environ.get(key) for key in settings}
        os.environ.update(settings)
        try:
            yield
        finally:
            _restore_compilation_environment(old)


def allocate_buffer(
    spaces, loc, element_count, dtype, address, shape=None, cross=False
):
    del spaces, address
    tensor_shape = shape or (element_count,)
    if cross:
        return RingBuffer(loc, tensor_shape, dtype, depth=1, data_format="nd")
    return Buffer(loc, tensor_shape, dtype, data_format="nd")


def build_ub_buffer_plan(query_head_tile_size, has_sinks, dtype):
    if not 1 <= query_head_tile_size <= _QUERY_HEAD_TILE_SIZE:
        raise ValueError("N1 must be in [1, 64]")
    subblock_query_head_count = (query_head_tile_size + 1) // _VECTOR_SUBBLOCKS
    fields = [
        ("kv_in_ub_bufs", _GATHER_CHUNKS * _GATHER_ELEMS, dtype),
        ("kv_out_ub_bufs", _GATHER_CHUNKS * _NZ_FRAGMENT_ELEMS, dtype),
        ("attention_acc_ub", subblock_query_head_count * _HEAD_DIM, dtypes.float32),
        ("bmm2_res_ub", subblock_query_head_count * _HEAD_DIM, dtypes.float32),
        (
            "bmm1_res_ub_ring",
            subblock_query_head_count * _BASE_TILE_SIZE,
            dtypes.float32,
        ),
        ("vec1_res_ub", (subblock_query_head_count + 1) * _BASE_TILE_SIZE, dtype),
        ("sinks_ub", subblock_query_head_count, dtypes.float32),
        ("softmax_max_ub", 2 * _FP32_VECTOR_LENGTH, dtypes.float32),
        ("softmax_sum_ub", 2 * _FP32_VECTOR_LENGTH, dtypes.float32),
        ("softmax_rescale_ub", 2 * 2 * _FP32_VECTOR_LENGTH, dtypes.float32),
        ("sparse_indices_ub_bufs", _BASE_TILE_SIZE, dtypes.int32),
        ("zero_attention_out_ub", subblock_query_head_count * _HEAD_DIM, dtype),
        # LSE writes at most one FP32 value for each head owned by this subblock.
        ("zero_softmax_lse_ub", subblock_query_head_count, dtypes.float32),
    ]
    offset = 0
    plan = {}
    for name, element_count, dtype in fields:
        if name in ("zero_attention_out_ub", "zero_softmax_lse_ub"):
            continue
        # Align each field to its element size for typed UB views.
        alignment = dtype.width // _BITS_PER_BYTE
        offset = (offset + alignment - 1) // alignment * alignment
        plan[name] = (offset, element_count, dtype)
        offset += element_count * alignment
    # The MTE3 source views below are copied as whole 32-byte blocks.
    for name in ("zero_attention_out_ub", "zero_softmax_lse_ub"):
        offset = (
            (offset + _UB_VIEW_ALIGNMENT_BYTES - 1)
            // _UB_VIEW_ALIGNMENT_BYTES
            * _UB_VIEW_ALIGNMENT_BYTES
        )
        address, element_count, field_dtype = next(
            field for field in fields if field[0] == name
        )
        plan[name] = (offset, element_count, field_dtype)
        offset += element_count * (field_dtype.width // _BITS_PER_BYTE)
    # The arena capacity check rounds total UB usage up to an 8-byte boundary.
    if (
        offset + (_UB_CAPACITY_ALIGNMENT_BYTES - 1)
    ) // _UB_CAPACITY_ALIGNMENT_BYTES * _UB_CAPACITY_ALIGNMENT_BYTES > _UB_ARENA_BYTES:
        raise ValueError("UB plan exceeds the reserved arena")
    return plan


class HighPerfSmlaStorage:
    # Shared physical buffer layout.

    def __init__(self, query_head_tile_size, has_sinks, dtype):
        # PA prepass storage was reclaimed at the phase boundary.  Keep all
        # main-pipeline storage allocator-owned so onchip_rewind remains legal.
        spaces = {}
        self.query_head_tile_size, self.subblock_query_head_count = (
            query_head_tile_size,
            (query_head_tile_size + 1) // _VECTOR_SUBBLOCKS,
        )
        self.lower_head_count = self.subblock_query_head_count
        self.upper_head_count = query_head_tile_size - self.lower_head_count
        self.aligned_head_count = (
            (query_head_tile_size + _NZ_ROW - 1) // _NZ_ROW * _NZ_ROW
        )
        self.dtype = dtype
        self.ub_buffer_plan = build_ub_buffer_plan(
            query_head_tile_size, has_sinks, dtype
        )
        for name, (address, element_count, field_dtype) in self.ub_buffer_plan.items():
            if element_count and name not in (
                "attention_acc_ub",
                "bmm2_res_ub",
                "kv_in_ub_bufs",
                "kv_out_ub_bufs",
                "sparse_indices_ub_bufs",
            ):
                setattr(
                    self,
                    name,
                    allocate_buffer(
                        spaces,
                        MemLoc.UB,
                        element_count,
                        field_dtype,
                        address,
                        (self.subblock_query_head_count, _BASE_TILE_SIZE)
                        if name == "bmm1_res_ub_ring"
                        else None,
                        cross=name == "bmm1_res_ub_ring",
                    ),
                )
        # Match MQSMLA: each 256-column result half owns an independent,
        # row-contiguous protocol slot.  The first/second row groups are the
        # accumulator/partial result for the same D half.
        self.bmm2_res_ub_rings = [
            RingBuffer(
                MemLoc.UB,
                (self.subblock_query_head_count * 2, _RESULT_HALF_COLS),
                dtypes.float32,
                depth=1,
                data_format="nd",
            )
            for _ in range(2)
        ]
        self.kv_in_ub_bufs = [
            allocate_buffer(
                spaces, MemLoc.UB, _GATHER_ELEMS, dtype, half * _GATHER_BYTES
            )
            for half in range(2)
        ]
        self.kv_out_ub_bufs = [
            allocate_buffer(
                spaces,
                MemLoc.UB,
                _NZ_FRAGMENT_ELEMS,
                dtype,
                self.ub_buffer_plan["kv_out_ub_bufs"][0]
                + fragment * _NZ_FRAGMENT_BYTES,
            )
            for fragment in range(2)
        ]
        self.ub_buffer_plan["sparse_indices_ub_bufs"][0]
        self.sparse_indices_ub_ring = RingBuffer(
            MemLoc.UB,
            (_KV_ROWS_PER_VECTOR,),
            dtypes.int32,
            depth=2,
            data_format="nd",
        )
        # Q 的 N1 行连续存放；只在 NZ 物理分型上按 16 行对齐。
        self.query_heads_l1_ring = RingBuffer(
            MemLoc.L1,
            (self.query_head_tile_size, _Q_WIDE_COLS),
            dtype,
            data_format="nz",
            depth=2,
        )
        self.query_heads_l1_tail = Buffer(
            MemLoc.L1,
            (self.query_head_tile_size, _BASE_TILE_SIZE),
            dtype,
            data_format="nz",
        )
        self.kv_l1_rings = [
            RingBuffer(
                MemLoc.L1,
                (_KV_SLOT_ELEMS,),
                dtype,
                depth=_KV_RING_DEPTH,
                data_format="nd",
            )
        ]
        self.bmm2_a_l1_ring = RingBuffer(
            MemLoc.L1,
            (self.aligned_head_count * _BASE_TILE_SIZE,),
            dtype,
            depth=1,
            data_format="nd",
        )
        self.kv_l1_rings.append(
            RingBuffer(
                MemLoc.L1,
                (_KV_SLOT_ELEMS,),
                dtype,
                depth=_KV_RING_DEPTH,
                data_format="nd",
            )
        )
        self.l0a_ring = RingBuffer(
            MemLoc.L0A,
            depth=_L0A_RING_DEPTH,
            view=TensorViewSpec(
                (query_head_tile_size, _BASE_TILE_SIZE),
                dtype,
                layout=ViewLayout(
                    "nz",
                    physical_shape=(
                        _TILE_NZ_TILES,
                        self.aligned_head_count // _NZ_ROW,
                        _NZ_ROW,
                        _NZ_ROW,
                    ),
                    physical_stride=(
                        self.aligned_head_count * _NZ_ROW,
                        _NZ_FACE,
                        _NZ_ROW,
                        1,
                    ),
                ),
            ),
        )
        self.l0b_ring = RingBuffer(
            MemLoc.L0B,
            depth=_L0B_SLOTS,
            view=TensorViewSpec(
                (_BASE_TILE_SIZE, _FP32_VECTOR_LENGTH),
                self.dtype,
                layout=ViewLayout(
                    "nz",
                    physical_shape=(_F32_NZ_TILES, _TILE_NZ_TILES, _NZ_ROW, _NZ_ROW),
                    physical_stride=(_BASE_TILE_SIZE * _NZ_ROW, _NZ_FACE, _NZ_ROW, 1),
                ),
            ),
        )
        self.bmm2_res_l0c_ring = RingBuffer(
            MemLoc.L0C,
            (self.query_head_tile_size, _FP32_VECTOR_LENGTH),
            dtypes.float32,
            depth=_D_F32_TILES,
            data_format="nz",
        )
        self.bmm1_res_l0c = Buffer(
            MemLoc.L0C,
            (self.query_head_tile_size, _BASE_TILE_SIZE),
            dtypes.float32,
            data_format="nz",
        )
        self.cross_core_rings = {
            name: getattr(self, name) for name in ("bmm1_res_ub_ring", "bmm2_a_l1_ring")
        }

    def get_next_bmm1_res_ub(self, is_write):
        channel = self.cross_core_rings["bmm1_res_ub_ring"]
        return channel.next_write() if is_write else channel.next_read()

    def get_next_kv_l1_buffers(self, is_write):
        if is_write:
            return (self.kv_l1_rings[0].next_write(), self.kv_l1_rings[1].next_write())
        return (self.kv_l1_rings[0].next_read(), self.kv_l1_rings[1].next_read())

    def get_next_bmm2_res_ub_buffers(self, is_write):
        is_first_tile = (
            self.bmm2_res_ub_rings[0].next_write()
            if is_write
            else self.bmm2_res_ub_rings[0].next_read()
        )
        second = (
            self.bmm2_res_ub_rings[1].next_write()
            if is_write
            else self.bmm2_res_ub_rings[1].next_read()
        )
        return (
            tile_slice(
                is_first_tile,
                (self.subblock_query_head_count, _RESULT_HALF_COLS),
                (0, 0),
            ),
            tile_slice(
                is_first_tile,
                (self.subblock_query_head_count, _RESULT_HALF_COLS),
                (1, 0),
            ),
            tile_slice(
                second, (self.subblock_query_head_count, _RESULT_HALF_COLS), (0, 0)
            ),
            tile_slice(
                second, (self.subblock_query_head_count, _RESULT_HALF_COLS), (1, 0)
            ),
        )


class HighPerfSmlaVec0Indices:
    def __init__(self, buffer_storage, batch_mode):
        self.buffer_storage = buffer_storage
        self.batch_mode = batch_mode

    @jit
    def init_vec0_buffers(self):
        for slot in range_constexpr(2):
            # VF gather initialization: zero each local 16-bit gather slot.
            with vf(mode="simd"):
                mask = rr.full_mask(elem_bits=_F16_BITS)
                zero_reg = rr.vdups(0.0, self.buffer_storage.dtype, mask=mask)
                for i in range(_GATHER_ELEMS // _FP16_VECTOR_LENGTH):
                    rr.vstore(
                        self.buffer_storage.kv_in_ub_bufs[slot],
                        i * _FP16_VECTOR_LENGTH,
                        zero_reg,
                        mask,
                    )

    @jit
    def copy_in_sparse_indices(
        self,
        indices,
        query_row_idx,
        kv_tile_idx,
        valid_kv_length,
        sparse_index_stride,
        prior_query_tile_count,
        global_tile,
        target,
    ):
        kv_index_start = (
            kv_tile_idx * _BASE_TILE_SIZE + get_subblock_id() * _KV_ROWS_PER_VECTOR
        )
        remaining_kv_indices = valid_kv_length - kv_index_start
        valid_index_count = scalar_select(
            remaining_kv_indices > 0,
            scalar_select(
                remaining_kv_indices < _KV_ROWS_PER_VECTOR,
                remaining_kv_indices,
                _KV_ROWS_PER_VECTOR,
            ),
            0,
        )
        if valid_index_count > 0:
            index_row = indices[query_row_idx * sparse_index_stride:,].view(
                sparse_index_stride
            )  # fmt: skip
            mem_copy(
                target[0:valid_index_count,],
                index_row[kv_index_start: kv_index_start + valid_index_count,],
            )  # fmt: skip

    @jit
    def copy_in_sparse_kv(
        self,
        kv_gm,
        kv_base_row,
        global_tile,
        kv_half_idx,
        kv_len,
        chunk,
        valid_kv_length,
        stream_kv_tile_idx,
        kv_indices,
    ):
        valid_segment_kv_rows = scalar_select(
            valid_kv_length < kv_len, valid_kv_length, kv_len
        )
        remaining_kv_indices = (
            valid_segment_kv_rows
            - stream_kv_tile_idx * _BASE_TILE_SIZE
            - get_subblock_id() * _KV_ROWS_PER_VECTOR
        )
        valid_subblock_kv_rows = scalar_select(
            remaining_kv_indices > 0,
            scalar_select(
                remaining_kv_indices < _KV_ROWS_PER_VECTOR,
                remaining_kv_indices,
                _KV_ROWS_PER_VECTOR,
            ),
            0,
        )
        kv_row_start = (
            kv_half_idx * (_KV_ROWS_PER_VECTOR // 2) + chunk * _GATHER_TILE_ROWS
        )
        if valid_subblock_kv_rows >= kv_row_start + _GATHER_TILE_ROWS:
            self.copy_in_kv_rows(
                kv_gm,
                kv_base_row,
                kv_indices,
                chunk,
                kv_row_start,
                valid_subblock_kv_rows,
                True,
            )
        else:
            self.copy_in_kv_rows(
                kv_gm,
                kv_base_row,
                kv_indices,
                chunk,
                kv_row_start,
                valid_subblock_kv_rows,
                False,
            )

    @jit
    def copy_in_pa_sparse_kv(
        self,
        kv_gm,
        kv_physical_rows_gm,
        query_row_idx,
        kv_half_idx,
        chunk,
        valid_kv_length,
        stream_kv_tile_idx,
        sparse_index_stride,
    ):
        subblock_offset = (
            stream_kv_tile_idx * _BASE_TILE_SIZE
            + get_subblock_id() * _KV_ROWS_PER_VECTOR
        )
        remaining_kv_indices = valid_kv_length - subblock_offset
        valid_subblock_kv_rows = scalar_select(
            remaining_kv_indices > 0,
            scalar_select(
                remaining_kv_indices < _KV_ROWS_PER_VECTOR,
                remaining_kv_indices,
                _KV_ROWS_PER_VECTOR,
            ),
            0,
        )
        kv_row_start = (
            kv_half_idx * (_KV_ROWS_PER_VECTOR // 2) + chunk * _GATHER_TILE_ROWS
        )
        index_row = kv_physical_rows_gm[query_row_idx * sparse_index_stride:,].view(
            sparse_index_stride
        )  # fmt: skip
        kv_indices = index_row[subblock_offset:,]
        if valid_subblock_kv_rows >= kv_row_start + _GATHER_TILE_ROWS:
            self.copy_in_kv_rows(
                kv_gm,
                dtypes.int64(0),
                kv_indices,
                chunk,
                kv_row_start,
                valid_subblock_kv_rows,
                True,
            )
        else:
            self.copy_in_kv_rows(
                kv_gm,
                dtypes.int64(0),
                kv_indices,
                chunk,
                kv_row_start,
                valid_subblock_kv_rows,
                False,
            )

    @jit
    def copy_in_kv_rows(
        self,
        kv_gm,
        kv_base_row,
        kv_indices,
        kv_buffer_idx,
        kv_row_start,
        valid_subblock_kv_rows,
        is_full_gather_chunk,
    ):
        for i in range_constexpr(_GATHER_PAIRS):
            # Each gather pair copies two adjacent destination KV rows.
            kv_row = kv_row_start + i * 2
            if const_expr(is_full_gather_chunk):
                first_index = dtypes.int64(dtypes.uint32(kv_indices[kv_row]))
                second_index = dtypes.int64(dtypes.uint32(kv_indices[kv_row + 1]))
                if const_expr(self.batch_mode == _BATCH_MODE_SINGLE):
                    first_kv_row = first_index
                    second_kv_row = second_index
                else:
                    first_kv_row = kv_base_row + first_index
                    second_kv_row = kv_base_row + second_index
                lower_kv_row = scalar_select(
                    first_kv_row < second_kv_row, first_kv_row, second_kv_row
                )
                higher_kv_row = scalar_select(
                    first_kv_row < second_kv_row, second_kv_row, first_kv_row
                )
                src = kv_gm.view(math.prod(kv_gm.shape) // _HEAD_DIM, _HEAD_DIM)
                dst = self.buffer_storage.kv_in_ub_bufs[kv_buffer_idx].view(
                    _GATHER_TILE_ROWS, _HEAD_DIM
                )
                mem_copy(
                    tile_slice(dst, (2, _HEAD_DIM), (i, 0)),
                    (
                        tile_slice(src, (1, _HEAD_DIM), (lower_kv_row, 0)).view(
                            1, _HEAD_DIM
                        ),
                        tile_slice(src, (1, _HEAD_DIM), (higher_kv_row, 0)).view(
                            1, _HEAD_DIM
                        ),
                    ),
                    options=copy_options(l2_cache_ctl=1),
                )
            else:
                first_kv_index = dtypes.int32(-1)
                second_kv_index = dtypes.int32(-1)
                if kv_row < valid_subblock_kv_rows:
                    first_kv_index = kv_indices[kv_row]
                if kv_row + 1 < valid_subblock_kv_rows:
                    second_kv_index = kv_indices[kv_row + 1]
                if first_kv_index >= 0 and second_kv_index >= 0:
                    first_index = dtypes.int64(dtypes.uint32(first_kv_index))
                    second_index = dtypes.int64(dtypes.uint32(second_kv_index))
                    if const_expr(self.batch_mode == _BATCH_MODE_SINGLE):
                        first_kv_row = first_index
                        second_kv_row = second_index
                    else:
                        first_kv_row = kv_base_row + first_index
                        second_kv_row = kv_base_row + second_index
                    lower_kv_row = scalar_select(
                        first_kv_row < second_kv_row, first_kv_row, second_kv_row
                    )
                    higher_kv_row = scalar_select(
                        first_kv_row < second_kv_row, second_kv_row, first_kv_row
                    )
                    src = kv_gm.view(math.prod(kv_gm.shape) // _HEAD_DIM, _HEAD_DIM)
                    dst = self.buffer_storage.kv_in_ub_bufs[kv_buffer_idx].view(
                        _GATHER_TILE_ROWS, _HEAD_DIM
                    )
                    mem_copy(
                        tile_slice(dst, (2, _HEAD_DIM), (i, 0)),
                        (
                            tile_slice(src, (1, _HEAD_DIM), (lower_kv_row, 0)).view(
                                1, _HEAD_DIM
                            ),
                            tile_slice(src, (1, _HEAD_DIM), (higher_kv_row, 0)).view(
                                1, _HEAD_DIM
                            ),
                        ),
                        options=copy_options(l2_cache_ctl=1),
                    )
                elif first_kv_index >= 0 or second_kv_index >= 0:
                    valid_kv_index = scalar_select(
                        first_kv_index >= 0, first_kv_index, second_kv_index
                    )
                    if not const_expr(self.batch_mode == _BATCH_MODE_SINGLE):
                        valid_kv_index = kv_base_row + valid_kv_index
                    mem_copy(
                        tile_slice(
                            self.buffer_storage.kv_in_ub_bufs[kv_buffer_idx].view(
                                _GATHER_TILE_ROWS, _HEAD_DIM
                            ),
                            (1, _HEAD_DIM),
                            (2 * i, 0),
                        ),
                        tile_slice(
                            kv_gm.view(math.prod(kv_gm.shape) // _HEAD_DIM, _HEAD_DIM),
                            (1, _HEAD_DIM),
                            (valid_kv_index, 0),
                        ),
                        options=copy_options(l2_cache_ctl=1),
                    )

    @jit
    def copy_in_pa_dense_kv(
        self,
        kv_gm,
        block_table_gm,
        batch_idx,
        kv_half_idx,
        chunk,
        valid_kv_length,
        stream_kv_tile_idx,
        ori_kv_start_row,
        page_size,
        page_stride_rows,
        block_table_width,
    ):
        subblock_start = (
            stream_kv_tile_idx * _BASE_TILE_SIZE
            + get_subblock_id() * _KV_ROWS_PER_VECTOR
            + kv_half_idx * (_KV_ROWS_PER_VECTOR // 2)
            + chunk * _GATHER_TILE_ROWS
        )
        remaining = valid_kv_length - subblock_start
        valid_rows = scalar_select(
            remaining > 0,
            scalar_select(remaining < _GATHER_TILE_ROWS, remaining, _GATHER_TILE_ROWS),
            0,
        )
        copied = dtypes.int64(0)
        src = kv_gm.view(math.prod(kv_gm.shape) // _HEAD_DIM, _HEAD_DIM)
        dst = self.buffer_storage.kv_in_ub_bufs[chunk].view(
            _GATHER_TILE_ROWS, _HEAD_DIM
        )
        # Dense CSA remains a continuous GM copy within each physical page.
        # A 16-row chunk is split only when it crosses a logical page boundary.
        while copied < valid_rows:
            logical = ori_kv_start_row + subblock_start + copied
            logical_page_idx = logical // page_size
            in_page_row = logical % page_size
            physical_block_idx = dtypes.int64(
                block_table_gm[batch_idx * block_table_width + logical_page_idx]
            )
            page_rows = page_size - in_page_row
            left = valid_rows - copied
            copy_rows = scalar_select(page_rows < left, page_rows, left)
            physical_row = physical_block_idx * page_stride_rows + in_page_row
            mem_copy(
                dst[copied: copied + copy_rows, 0:_HEAD_DIM],
                src[physical_row: physical_row + copy_rows, 0:_HEAD_DIM],
                options=copy_options(l2_cache_ctl=1),
            )  # fmt: skip
            copied = copied + copy_rows

    @jit
    def copy_in_dense_kv(
        self,
        kv_gm,
        kv_base_row,
        kv_half_idx,
        kv_len,
        chunk,
        valid_kv_length,
        stream_kv_tile_idx,
        ori_kv_start_row,
    ):
        valid_segment_rows = scalar_select(
            valid_kv_length < kv_len, valid_kv_length, kv_len
        )
        subblock_start = (
            stream_kv_tile_idx * _BASE_TILE_SIZE
            + get_subblock_id() * _KV_ROWS_PER_VECTOR
            + kv_half_idx * (_KV_ROWS_PER_VECTOR // 2)
            + chunk * _GATHER_TILE_ROWS
        )
        remaining = valid_segment_rows - subblock_start
        valid_rows = scalar_select(
            remaining > 0,
            scalar_select(
                remaining < _GATHER_TILE_ROWS,
                remaining,
                _GATHER_TILE_ROWS,
            ),
            0,
        )
        if valid_rows > 0:
            source_start = kv_base_row + ori_kv_start_row + subblock_start
            source = kv_gm.view(math.prod(kv_gm.shape) // _HEAD_DIM, _HEAD_DIM)[
                source_start: source_start + valid_rows,
                0:_HEAD_DIM,
            ]  # fmt: skip
            mem_copy(
                self.buffer_storage.kv_in_ub_bufs[chunk].view(
                    _GATHER_TILE_ROWS, _HEAD_DIM
                )[
                    0:valid_rows,
                    0:_HEAD_DIM,
                ],
                source,
                options=copy_options(l2_cache_ctl=1),
            )

    @jit
    def convert_kv_nd_to_nz(self, chunk):
        source = self.buffer_storage.kv_in_ub_bufs[chunk]
        # VF layout conversion: restage gathered KV rows into the padded NZ fragment.
        with vf(mode="simd"):
            mask = rr.full_mask(elem_bits=_F16_BITS)
            for kv_row in range(_GATHER_TILE_ROWS):
                for quarter in range_constexpr(_HEAD_DIM // _FP16_VECTOR_LENGTH):
                    value = rr.vload(
                        source, kv_row * _HEAD_DIM + quarter * _FP16_VECTOR_LENGTH
                    )
                    rr.vstore_strided(
                        self.buffer_storage.kv_out_ub_bufs[chunk],
                        quarter * _NZ_QUARTER_STRIDE + kv_row * _NZ_ROW,
                        value,
                        mask,
                        block_stride=_NZ_PADDED_ROWS,
                        repeat_stride=0,
                    )


class HighPerfSmlaVector:
    def __init__(self, buffer_storage, return_softmax_lse):
        self.buffer_storage = buffer_storage
        self.return_softmax_lse = return_softmax_lse
        self.subblock_query_head_count = buffer_storage.subblock_query_head_count

    @jit
    def init_no_sink_buffer(self):
        # No sink GM access: initialize every logical head, including padding,
        # in the AIV-owned UB buffer before the common softmax path consumes it.
        with vf(mode="simd"):
            mask, _ = rr.update_mask(
                self.subblock_query_head_count, elem_bits=_F32_BITS
            )
            negative_inf = rr.vreinterpret(
                rr.vdups(_F32_NEG_INF_BITS, dtypes.uint32, mask=mask),
                dtypes.float32,
            )
            rr.vstore(self.buffer_storage.sinks_ub, 0, negative_inf, mask)

    @jit
    def init_empty_output_buffers(self):
        active_heads = self.subblock_query_head_count
        if const_expr(
            self.buffer_storage.lower_head_count != self.buffer_storage.upper_head_count
        ):
            active_heads = scalar_select(
                get_subblock_id() == 0,
                self.buffer_storage.lower_head_count,
                self.buffer_storage.upper_head_count,
            )
        with vf(mode="simd"):
            mask_fp16 = rr.full_mask(elem_bits=_F16_BITS)
            zero_reg = rr.vdups(0.0, self.buffer_storage.dtype, mask=mask_fp16)
            for i in cannbotdsl.range(
                dtypes.int32(0),
                dtypes.int32(active_heads * (_HEAD_DIM // _FP16_VECTOR_LENGTH)),
                dtypes.int32(1),
                unroll=1,
            ):
                rr.vstore(
                    self.buffer_storage.zero_attention_out_ub,
                    i * _FP16_VECTOR_LENGTH,
                    zero_reg,
                    mask_fp16,
                )
            if const_expr(self.return_softmax_lse):
                lse_mask, _ = rr.update_mask(active_heads, elem_bits=_F32_BITS)
                empty_lse = rr.vdups(0.0, dtypes.float32, mask=lse_mask)
                rr.vstore(
                    self.buffer_storage.zero_softmax_lse_ub, 0, empty_lse, lse_mask
                )
        # These ordinary UB buffers are tracked by automatic buffer synchronization.

    @jit
    def copy_out_empty_query(
        self,
        attention_out_gm,
        softmax_lse_gm,
        const_info,
        query_row_idx,
        head_start,
        valid_head_count,
    ):
        if valid_head_count > 0:
            # The zero buffer covers ceil(N1/2) heads, including each valid prefix.
            # Copy the valid prefix directly, avoiding a redundant clamp/guard.
            output_start = (
                dtypes.int64(query_row_idx) * const_info.num_heads_q
                + dtypes.int64(head_start)
            ) * _HEAD_DIM
            output_stop = output_start + valid_head_count * _HEAD_DIM
            mem_copy(
                attention_out_gm[output_start:output_stop,],
                self.buffer_storage.zero_attention_out_ub[
                    0: valid_head_count * _HEAD_DIM,
                ],
                options=copy_options(l2_cache_ctl=_KV_COPY_L2_CACHE_CONTROL),
            )  # fmt: skip
            if const_expr(self.return_softmax_lse):
                lse_start = dtypes.int64(
                    query_row_idx
                ) * const_info.num_heads_q + dtypes.int64(head_start)
                lse_stop = lse_start + dtypes.int64(valid_head_count)
                mem_copy(
                    softmax_lse_gm[lse_start:lse_stop,],
                    self.buffer_storage.zero_softmax_lse_ub[0:valid_head_count,],
                )

    @jit
    def compute_vec1_softmax(
        self,
        active_query_idx,
        kv_tile_idx,
        scale,
        is_first_tile,
        is_full_kv_tile,
        bmm1_res_ub,
        vec1_res_ub,
        valid_tile_kv_rows,
    ):
        # VF softmax: reduce QK scores, update online state and pack 16-bit P in NZ order.
        active_heads = self.subblock_query_head_count
        if const_expr(
            self.buffer_storage.lower_head_count != self.buffer_storage.upper_head_count
        ):
            active_heads = scalar_select(
                get_subblock_id() == 0,
                self.buffer_storage.lower_head_count,
                self.buffer_storage.upper_head_count,
            )
        with vf(mode="simd"):
            mask = rr.full_mask(elem_bits=_F32_BITS)
            mask_fp16 = rr.full_mask(elem_bits=_F16_BITS)
            # Keep two FP32 copies per head for the interleaved online softmax state.
            two_lanes, _ = rr.update_mask(2, elem_bits=_F32_BITS)
            softmax_state_offset = active_query_idx % 2 * _FP32_VECTOR_LENGTH
            if const_expr(is_first_tile):
                sink_reg = rr.vload(self.buffer_storage.sinks_ub)
                neg_inf = rr.veqs(
                    rr.vreinterpret(sink_reg, dtypes.uint32),
                    _F32_NEG_INF_BITS,
                    mask=mask,
                )
                # Preserve +inf so online softmax matches the upstream golden.

                scaled_sink_reg = rr.vdups(
                    _F32_MIN,
                    dtypes.float32,
                    mask=neg_inf,
                    mode="merging",
                    merge=sink_reg,
                )
                # Only -inf needs a finite sentinel and zero initial mass.
                # For +inf, exp(old_max - new_max) propagates NaN as upstream.

                replicated, unused = rr.vinterleave(scaled_sink_reg, scaled_sink_reg)
                rr.vstore(
                    self.buffer_storage.softmax_max_ub,
                    softmax_state_offset,
                    replicated,
                    mask,
                )
                # Keep downstream source positions stable for the DSL source capturer.
                # Scaled-domain state removes the former divide-by-scale sequence.
                # The extra source lines intentionally preserve decorated method locations.
                # This avoids stale positional lookup in the AOT frontend.
                # Do not collapse these lines without updating the source capture cache.
                initial_sum = rr.vdups(1.0, dtypes.float32)
                initial_sum = rr.vdups(
                    0.0, dtypes.float32, mask=neg_inf, mode="merging", merge=initial_sum
                )
                replicated, unused = rr.vinterleave(initial_sum, initial_sum)
                rr.vstore(
                    self.buffer_storage.softmax_sum_ub,
                    softmax_state_offset,
                    replicated,
                    mask,
                )
                # Order these state stores before the following state loads.
                rr.vmem_bar("vst_vld")
            old_max = rr.vload(self.buffer_storage.softmax_max_ub, softmax_state_offset)
            rr.vmem_bar("vld_vst")
            old_sum = rr.vload(self.buffer_storage.softmax_sum_ub, softmax_state_offset)

            neg_inf_reg = rr.vreinterpret(
                rr.vdups(_F32_NEG_INF_BITS, dtypes.uint32), dtypes.float32
            )
            if const_expr(not is_full_kv_tile):
                even_index_reg = rr.vmuls(rr.varange(0, dtypes.int32), 2, mask=mask)
                odd_index_reg = rr.vadds(even_index_reg, 1, mask=mask)
                invalid_even = rr.vges(
                    even_index_reg, dtypes.int32(valid_tile_kv_rows), mask=mask
                )
                invalid_odd = rr.vges(
                    odd_index_reg, dtypes.int32(valid_tile_kv_rows), mask=mask
                )
            max_cursor = rr.vstore_unalign_begin(self.buffer_storage.softmax_max_ub)
            # Explicit int32 loop values select the VF scalar ABI.
            for query_head_row in cannbotdsl.range(
                dtypes.int32(0),
                dtypes.int32(active_heads),
                dtypes.int32(1),
                unroll=1,
            ):
                score_even_reg, score_odd_reg = rr.vload_deinterleave(
                    bmm1_res_ub, query_head_row * _BASE_TILE_SIZE
                )
                score_even_reg = rr.vmuls(score_even_reg, scale, mask=mask)
                score_odd_reg = rr.vmuls(score_odd_reg, scale, mask=mask)
                if const_expr(not is_full_kv_tile):
                    score_even_reg = rr.vselect(
                        neg_inf_reg, score_even_reg, cond_mask=invalid_even
                    )
                    score_odd_reg = rr.vselect(
                        neg_inf_reg, score_odd_reg, cond_mask=invalid_odd
                    )
                row_max_reg = rr.vmax(
                    rr.vreduce_max(score_even_reg, mask=mask),
                    rr.vreduce_max(score_odd_reg, mask=mask),
                    mask=mask,
                )
                row_max_reg = rr.vdup(row_max_reg, mask=mask)
                prior_max_reg = rr.vload_broadcast(
                    self.buffer_storage.softmax_max_ub,
                    softmax_state_offset + 2 * query_head_row,
                )
                row_max_reg = rr.vmax(row_max_reg, prior_max_reg, mask=mask)
                squeezed = rr.vsqueeze_and_storeunalign_init(
                    row_max_reg, mask=two_lanes
                )
                rr.vsqueeze_and_storeunalign(
                    self.buffer_storage.softmax_max_ub,
                    softmax_state_offset,
                    squeezed,
                    max_cursor,
                )
            rr.vsqueeze_and_storeunalign_finalize(
                self.buffer_storage.softmax_max_ub, softmax_state_offset, max_cursor
            )
            rr.vmem_bar("vst_vld")
            sum_cursor = rr.vstore_unalign_begin(self.buffer_storage.softmax_sum_ub)
            for query_head_row in cannbotdsl.range(
                dtypes.int32(0),
                dtypes.int32(active_heads),
                dtypes.int32(1),
                unroll=1,
            ):
                score_even_reg, score_odd_reg = rr.vload_deinterleave(
                    bmm1_res_ub, query_head_row * _BASE_TILE_SIZE
                )
                score_even_reg = rr.vmuls(score_even_reg, scale, mask=mask)
                score_odd_reg = rr.vmuls(score_odd_reg, scale, mask=mask)
                if const_expr(not is_full_kv_tile):
                    score_even_reg = rr.vselect(
                        neg_inf_reg, score_even_reg, cond_mask=invalid_even
                    )
                    score_odd_reg = rr.vselect(
                        neg_inf_reg, score_odd_reg, cond_mask=invalid_odd
                    )
                row_max_reg = rr.vload_broadcast(
                    self.buffer_storage.softmax_max_ub,
                    softmax_state_offset + 2 * query_head_row,
                )
                exp_even_reg = rr.vexp_sub(score_even_reg, row_max_reg, mask=mask)
                exp_odd_reg = rr.vexp_sub(score_odd_reg, row_max_reg, mask=mask)
                row_sum_reg = rr.vadd(
                    rr.vreduce_sum(exp_even_reg, mask=mask),
                    rr.vreduce_sum(exp_odd_reg, mask=mask),
                    mask=mask,
                )
                row_sum_reg = rr.vdup(row_sum_reg, mask=mask)
                squeezed = rr.vsqueeze_and_storeunalign_init(
                    row_sum_reg, mask=two_lanes
                )
                rr.vsqueeze_and_storeunalign(
                    self.buffer_storage.softmax_sum_ub,
                    softmax_state_offset,
                    squeezed,
                    sum_cursor,
                )
                p_even_reg = rr.vcast(
                    exp_even_reg,
                    self.buffer_storage.dtype,
                    mask=mask,
                    rounding="rn",
                    reg_layout=rr.RegLayout.ZERO,
                )
                p_odd_reg = rr.vcast(
                    exp_odd_reg,
                    self.buffer_storage.dtype,
                    mask=mask,
                    rounding="rn",
                    reg_layout=rr.RegLayout.ONE,
                )
                packed_p_reg = rr.vadd(p_even_reg, p_odd_reg, mask=mask_fp16)
                rr.vstore_strided(
                    vec1_res_ub,
                    query_head_row * _NZ_ROW,
                    packed_p_reg,
                    mask_fp16,
                    block_stride=self.subblock_query_head_count + 1,
                    repeat_stride=0,
                )
            rr.vsqueeze_and_storeunalign_finalize(
                self.buffer_storage.softmax_sum_ub, softmax_state_offset, sum_cursor
            )
            rr.vmem_bar("vst_vld")
            row_max_reg = rr.vload(
                self.buffer_storage.softmax_max_ub, softmax_state_offset
            )
            rescale_factor_reg = rr.vexp_sub(old_max, row_max_reg, mask=mask)
            rr.vstore(
                self.buffer_storage.softmax_rescale_ub,
                active_query_idx % 2 * _BASE_TILE_SIZE
                + kv_tile_idx % 2 * _FP32_VECTOR_LENGTH,
                rescale_factor_reg,
                mask,
            )
            row_sum_reg = rr.vmadd(
                old_sum,
                rescale_factor_reg,
                rr.vload(self.buffer_storage.softmax_sum_ub, softmax_state_offset),
                mask=mask,
            )
            rr.vstore(
                self.buffer_storage.softmax_sum_ub,
                softmax_state_offset,
                row_sum_reg,
                mask,
            )

    @jit
    def finalize_vec1_softmax_state(self, active_query_idx, scale):
        # VF softmax finalization: compact per-head state and optionally produce LSE.
        with vf(mode="simd"):
            mask = rr.full_mask(elem_bits=_F32_BITS)
            softmax_state_offset = active_query_idx % 2 * _FP32_VECTOR_LENGTH
            softmax_max_reg, unused = rr.vdeinterleave(
                rr.vload(self.buffer_storage.softmax_max_ub, softmax_state_offset),
                rr.vload(self.buffer_storage.softmax_max_ub, softmax_state_offset),
            )
            softmax_sum_reg, unused = rr.vdeinterleave(
                rr.vload(self.buffer_storage.softmax_sum_ub, softmax_state_offset),
                rr.vload(self.buffer_storage.softmax_sum_ub, softmax_state_offset),
            )
            empty_row_mask = rr.veqs(softmax_max_reg, _F32_MIN, mask=mask)
            pos_inf_reg = rr.vreinterpret(
                rr.vdups(_F32_POS_INF_BITS, dtypes.uint32), dtypes.float32
            )
            if const_expr(self.return_softmax_lse):
                softmax_lse_reg = rr.vadd(
                    rr.vlog(softmax_sum_reg, mask=mask), softmax_max_reg, mask=mask
                )
                softmax_lse_reg = rr.vselect(
                    pos_inf_reg, softmax_lse_reg, cond_mask=empty_row_mask
                )
                # Preserve the online result, including NaN for a +inf sink.
                # Empty-KV queries use the separate zero-output fast path.

                rr.vstore(
                    self.buffer_storage.softmax_max_ub,
                    softmax_state_offset,
                    softmax_lse_reg,
                    mask,
                )
            softmax_sum_reg = rr.vselect(
                pos_inf_reg, softmax_sum_reg, cond_mask=empty_row_mask
            )
            rr.vstore(
                self.buffer_storage.softmax_sum_ub,
                softmax_state_offset,
                softmax_sum_reg,
                mask,
            )

    @jit
    def compute_vec2_attention_out(
        self,
        active_query_idx,
        kv_tile_idx,
        is_first_tile,
        is_last_tile,
        attention_acc_ub,
        bmm2_res_ub,
        attention_out_ub,
    ):
        # VF output update: accumulate PV tiles, then normalize and pack the final result.
        active_heads = self.subblock_query_head_count
        if const_expr(
            self.buffer_storage.lower_head_count != self.buffer_storage.upper_head_count
        ):
            active_heads = scalar_select(
                get_subblock_id() == 0,
                self.buffer_storage.lower_head_count,
                self.buffer_storage.upper_head_count,
            )
        with vf(mode="simd"):
            mask = rr.full_mask(elem_bits=_F32_BITS)
            for query_head_row in cannbotdsl.range(
                dtypes.int32(0), dtypes.int32(active_heads), dtypes.int32(1), unroll=1
            ):
                rescale_factor_reg = rr.vdups(1.0, dtypes.float32)
                inv_softmax_sum_reg = rr.vdups(1.0, dtypes.float32)
                if const_expr(not is_first_tile):
                    rescale_factor_reg = rr.vload_broadcast(
                        self.buffer_storage.softmax_rescale_ub,
                        active_query_idx % 2 * _BASE_TILE_SIZE
                        + kv_tile_idx % 2 * _FP32_VECTOR_LENGTH
                        + query_head_row * 2,
                    )
                if const_expr(is_last_tile):
                    inv_softmax_sum_reg = rr.vdiv(
                        rr.vdups(1.0, dtypes.float32),
                        rr.vload_broadcast(
                            self.buffer_storage.softmax_sum_ub,
                            active_query_idx % 2 * _FP32_VECTOR_LENGTH + query_head_row,
                        ),
                        mask=mask,
                    )
                for d_vector_tile in range_constexpr(
                    _RESULT_HALF_COLS // _FP32_VECTOR_LENGTH
                ):
                    output_element_offset = (
                        query_head_row * _RESULT_HALF_COLS
                        + d_vector_tile * _FP32_VECTOR_LENGTH
                    )
                    if const_expr(is_first_tile):
                        output_reg = rr.vload(attention_acc_ub, output_element_offset)
                    else:
                        output_reg = rr.vload(bmm2_res_ub, output_element_offset)
                        rescaled_previous_output_reg = rr.vmul(
                            rr.vload(attention_acc_ub, output_element_offset),
                            rescale_factor_reg,
                            mask=mask,
                        )
                        output_reg = rr.vadd(
                            rescaled_previous_output_reg, output_reg, mask=mask
                        )
                    if const_expr(is_last_tile):
                        output_reg = rr.vmul(output_reg, inv_softmax_sum_reg, mask=mask)
                        packed_output_reg = rr.vcast(
                            output_reg,
                            self.buffer_storage.dtype,
                            mask=mask,
                            rounding="rn",
                        )
                        rr.vstore_pack(
                            attention_out_ub,
                            query_head_row
                            * (_RESULT_HALF_COLS * _F32_BYTES // _F16_BYTES)
                            + d_vector_tile * _FP32_VECTOR_LENGTH,
                            packed_output_reg,
                            mask,
                            pack_mode="b32_to_b16",
                        )
                    else:
                        rr.vstore(
                            attention_acc_ub,
                            output_element_offset,
                            output_reg,
                            mask,
                        )


class HighPerfSmlaMatmul:
    def __init__(self, buffer_storage):
        (
            self.buffer_storage,
            self.query_head_tile_size,
            self.subblock_query_head_count,
        ) = (
            buffer_storage,
            buffer_storage.query_head_tile_size,
            buffer_storage.subblock_query_head_count,
        )

    @jit
    def copy_q_gm_to_l1(
        self, query_gm, query_row_idx, active_query_idx, num_heads_q, load_d_tail
    ):
        query_heads = query_gm[query_row_idx * num_heads_q * _HEAD_DIM:,].view(
            num_heads_q, _HEAD_DIM
        )  # fmt: skip
        d_column_start = 0
        d_column_count = _Q_WIDE_COLS
        if const_expr(load_d_tail):
            query_heads_l1 = self.buffer_storage.query_heads_l1_tail
            d_column_start = _Q_WIDE_COLS
            d_column_count = _BASE_TILE_SIZE
        else:
            query_heads_l1 = self.buffer_storage.query_heads_l1_ring.next_write()
        mem_copy(
            query_heads_l1,
            query_heads[
                0: self.query_head_tile_size,
                d_column_start: d_column_start + d_column_count,
            ],
            options=copy_options(l2_cache_ctl=_KV_COPY_L2_CACHE_CONTROL),
        )  # fmt: skip

        return query_heads_l1

    @jit
    def iterate_bmm1(
        self, active_query_idx, kv_lower_rows_l1, kv_upper_rows_l1, query_heads_l1_wide
    ):
        set_mmad_direction("m")
        # Eight D-axis slices are accumulated into one QK score tile.
        for d_tile in range_constexpr(_D_F32_TILES):
            l0_a = self.buffer_storage.l0a_ring.next_write()
            if const_expr(d_tile < _WIDE_F32_TILES):
                query_heads_l1 = query_heads_l1_wide
                q_column_tile = d_tile
            else:
                query_heads_l1 = self.buffer_storage.query_heads_l1_tail
                q_column_tile = d_tile - _WIDE_F32_TILES
            mem_copy(
                view_nz(l0_a, self.query_head_tile_size, _FP32_VECTOR_LENGTH),
                get_nz_column_tile(
                    query_heads_l1, self.query_head_tile_size, q_column_tile
                ),
            )
            l0_b = self.buffer_storage.l0b_ring.next_write()
            mem_copy(
                get_bmm1_operand_half(l0_b, _KV_ROWS_PER_VECTOR, 0),
                get_nz_column_tile(kv_lower_rows_l1, _KV_ROWS_PER_VECTOR, d_tile),
            )
            mem_copy(
                get_bmm1_operand_half(l0_b, _KV_ROWS_PER_VECTOR, 1),
                get_nz_column_tile(kv_upper_rows_l1, _KV_ROWS_PER_VECTOR, d_tile),
            )
            matmul(
                self.buffer_storage.bmm1_res_l0c,
                view_nz(l0_a, self.query_head_tile_size, _FP32_VECTOR_LENGTH),
                view_nz(l0_b, _BASE_TILE_SIZE, _FP32_VECTOR_LENGTH),
                init=d_tile == 0,
                unit_flag=_MMAD_FINAL_UNIT_FLAG if d_tile == _D_F32_TILES - 1 else 2,
            )

    @jit
    def copy_bmm1_res_l0c_to_ub(self, bmm1_res_ub):
        bmm1_res_l0c_view = self.buffer_storage.bmm1_res_l0c
        mem_copy(
            bmm1_res_ub,
            bmm1_res_l0c_view.partition(dim=0),
            options=copy_options(unit_flag=_MMAD_FINAL_UNIT_FLAG),
        )

    @jit
    def iterate_bmm2(
        self,
        kv_tile_idx,
        kv_lower_rows_l1,
        kv_upper_rows_l1,
        bmm2_a_l1,
        attention_acc_lower_d_ub,
        bmm2_res_lower_d_ub,
        attention_acc_upper_d_ub,
        bmm2_res_upper_d_ub,
    ):
        set_mmad_direction("n")
        l0_a = self.buffer_storage.l0a_ring.next_write()
        for d_tile in range_constexpr(_D_F32_TILES):
            l0_b = self.buffer_storage.l0b_ring.next_write()
            mem_copy(
                get_bmm2_operand_half(l0_b, 0),
                get_nz_column_tile(
                    kv_lower_rows_l1, _KV_ROWS_PER_VECTOR, d_tile
                ).transpose(0, 1),
            )
            mem_copy(
                get_bmm2_operand_half(l0_b, 1),
                get_nz_column_tile(
                    kv_upper_rows_l1, _KV_ROWS_PER_VECTOR, d_tile
                ).transpose(0, 1),
            )
            if const_expr(d_tile == 0):
                mem_copy(
                    view_nz(l0_a, self.query_head_tile_size, _BASE_TILE_SIZE),
                    view_nz(bmm2_a_l1, self.query_head_tile_size, _BASE_TILE_SIZE),
                )
            pv_partial_l0c = self.buffer_storage.bmm2_res_l0c_ring.next_write()
            matmul(
                pv_partial_l0c,
                view_nz(l0_a, self.query_head_tile_size, _BASE_TILE_SIZE),
                view_nz(l0_b, _FP32_VECTOR_LENGTH, _BASE_TILE_SIZE),
                init=True,
                unit_flag=3,
            )
            if const_expr(d_tile >= _D_F32_TILES // 2):
                acc_target, pv_target = (
                    attention_acc_upper_d_ub,
                    bmm2_res_upper_d_ub,
                )
            else:
                acc_target, pv_target = (attention_acc_lower_d_ub, bmm2_res_lower_d_ub)
            if kv_tile_idx == 0:
                target = tile_slice(
                    acc_target,
                    (self.subblock_query_head_count, _FP32_VECTOR_LENGTH),
                    (0, d_tile % _L0B_SLOTS),
                )
                mem_copy(
                    target,
                    pv_partial_l0c.partition(dim=0),
                    options=copy_options(unit_flag=3),
                )
            else:
                target = tile_slice(
                    pv_target,
                    (self.subblock_query_head_count, _FP32_VECTOR_LENGTH),
                    (0, d_tile % _L0B_SLOTS),
                )
                mem_copy(
                    target,
                    pv_partial_l0c.partition(dim=0),
                    options=copy_options(unit_flag=3),
                )


@datastruct
class HighPerfSmlaConstInfo:
    # Runtime scalar parameters passed to the kernel.
    query_row_count: dtypes.int32
    ori_kv_capacity: dtypes.int32
    cmp_kv_capacity: dtypes.int32
    cmp_sparse_index_width: dtypes.int32
    has_cmp_topk_length: dtypes.int32
    num_heads_q: dtypes.int32
    ori_sparse_index_width: dtypes.int32
    ori_win_left: dtypes.int32
    ori_win_right: dtypes.int32
    has_ori_topk_length: dtypes.int32
    has_seqused_ori_kv: dtypes.int32
    has_seqused_cmp_kv: dtypes.int32
    has_cmp: dtypes.int32
    batch_mode: dtypes.int32
    batch_count: dtypes.int32
    query_rows_per_batch: dtypes.int32
    cube_core_count: dtypes.int32
    has_seqused_q: dtypes.int32
    ori_mask_mode: dtypes.int32
    cmp_mask_mode: dtypes.int32
    cmp_ratio: dtypes.int32
    is_ori_dense: dtypes.int32
    kv_mode: dtypes.int32
    ori_page_size: dtypes.int32
    cmp_page_size: dtypes.int32
    ori_page_stride_rows: dtypes.int32
    cmp_page_stride_rows: dtypes.int32
    ori_block_table_width: dtypes.int32
    cmp_block_table_width: dtypes.int32
    ori_page_size_reciprocal: dtypes.uint32
    cmp_page_size_reciprocal: dtypes.uint32
    ori_page_size_log2: dtypes.int32
    cmp_page_size_log2: dtypes.int32
    ori_page_size_is_power_of_two: dtypes.int32
    cmp_page_size_is_power_of_two: dtypes.int32
    softmax_scale: dtypes.float32


@jit
def calc_ori_sparse_capacity(const_info):
    ori_capacity = dtypes.int64(const_info.ori_sparse_index_width)
    if const_info.ori_win_left >= 0:
        ori_window_length = const_info.ori_win_left + 1 + const_info.ori_win_right
        ori_capacity = scalar_select(
            ori_capacity < ori_window_length, ori_capacity, ori_window_length
        )
    return ori_capacity


@jit
def get_metadata_task_count(metadata_gm, seqused_q_gm, cu_seqlens_q, const_info):
    base = get_block_idx() * _METADATA_STRIDE
    count = dtypes.int64(0)
    if dtypes.int32(metadata_gm[base]) != 0:
        first_batch = dtypes.int64(metadata_gm[base + 1])
        first_query = dtypes.int64(metadata_gm[base + 2])
        last_batch = dtypes.int64(metadata_gm[base + _METADATA_LAST_BATCH])
        last_query = dtypes.int64(metadata_gm[base + _METADATA_LAST_QUERY])
        # The exclusive end can be the (B, 0) sentinel.
        batch_stop = scalar_select(
            last_batch < const_info.batch_count, last_batch + 1, const_info.batch_count
        )
        for batch_idx in range(first_batch, batch_stop):
            q_length = dtypes.int64(const_info.query_rows_per_batch)
            if const_info.batch_mode == _BATCH_MODE_TND:
                q_length = dtypes.int64(
                    cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx]
                )
            begin = scalar_select(batch_idx == first_batch, first_query, 0)
            end = scalar_select(batch_idx == last_batch, last_query, q_length)
            count = count + scalar_select(end > begin, end - begin, 0)
    return count


@jit
def get_metadata_task_context(
    metadata_gm, seqused_q_gm, cu_seqlens_q, const_info, metadata_task_idx
):
    """Resolve one metadata task without rescanning its entire batch range.

    Uniform packed Q normally hits the checked arithmetic estimate. Ragged Q,
    including repeated prefixes for empty batches, falls back to binary search.
    The estimate is never accepted without checking its actual prefix interval.
    """
    base = get_block_idx() * _METADATA_STRIDE
    query_row_idx = dtypes.int64(0)
    batch_result = dtypes.int64(0)
    query_pos_result = dtypes.int64(0)
    valid_q_result = dtypes.int64(0)
    found = dtypes.int64(0)
    if dtypes.int32(metadata_gm[base]) != 0:
        first_batch = dtypes.int64(metadata_gm[base + 1])
        first_query = dtypes.int64(metadata_gm[base + 2])
        last_batch = dtypes.int64(metadata_gm[base + _METADATA_LAST_BATCH])
        last_query = dtypes.int64(metadata_gm[base + _METADATA_LAST_QUERY])
        if const_info.batch_mode == _BATCH_MODE_TND:
            range_start = dtypes.int64(cu_seqlens_q[first_batch]) + first_query
            range_end = dtypes.int64(cu_seqlens_q[const_info.batch_count])
            if last_batch < const_info.batch_count:
                last_base = dtypes.int64(cu_seqlens_q[last_batch])
                last_length = dtypes.int64(cu_seqlens_q[last_batch + 1]) - last_base
                range_end = last_base + scalar_select(
                    last_query < last_length, last_query, last_length
                )
            candidate_row = range_start + metadata_task_idx
            if metadata_task_idx >= 0:
                if candidate_row < range_end:
                    final_batch = scalar_select(
                        last_batch < const_info.batch_count,
                        last_batch,
                        dtypes.int64(const_info.batch_count) - 1,
                    )
                    selected_batch = first_batch
                    selected_base = dtypes.int64(cu_seqlens_q[first_batch])
                    selected_end = dtypes.int64(cu_seqlens_q[first_batch + 1])
                    if first_batch < final_batch:
                        prefix_start = dtypes.int64(cu_seqlens_q[0])
                        prefix_end = dtypes.int64(cu_seqlens_q[const_info.batch_count])
                        average_span = (
                            prefix_end - prefix_start
                        ) // const_info.batch_count
                        average_span = scalar_select(average_span > 0, average_span, 1)
                        guess = (candidate_row - prefix_start) // average_span
                        guess = scalar_select(guess < first_batch, first_batch, guess)
                        guess = scalar_select(guess > final_batch, final_batch, guess)
                        selected_batch = guess
                        selected_base = dtypes.int64(cu_seqlens_q[guess])
                        selected_end = dtypes.int64(cu_seqlens_q[guess + 1])
                        if (
                            candidate_row < selected_base
                            or candidate_row >= selected_end
                        ):
                            low = first_batch
                            high = final_batch
                            while low < high:
                                middle = (low + high) // 2
                                middle_end = dtypes.int64(cu_seqlens_q[middle + 1])
                                if middle_end <= candidate_row:
                                    low = middle + 1
                                else:
                                    high = middle
                            selected_batch = low
                            selected_base = dtypes.int64(cu_seqlens_q[low])
                            selected_end = dtypes.int64(cu_seqlens_q[low + 1])
                    query_row_idx = candidate_row
                    batch_result = selected_batch
                    query_pos_result = candidate_row - selected_base
                    valid_q_result = selected_end - selected_base
                    if const_info.has_seqused_q != 0:
                        valid_q_result = dtypes.int64(seqused_q_gm[selected_batch])
                    found = dtypes.int64(1)
        else:
            rows_per_batch = dtypes.int64(const_info.query_rows_per_batch)
            range_start = first_batch * rows_per_batch + first_query
            range_end = last_batch * rows_per_batch + scalar_select(
                last_query < rows_per_batch, last_query, rows_per_batch
            )
            candidate_row = range_start + metadata_task_idx
            if metadata_task_idx >= 0:
                if candidate_row < range_end:
                    selected_batch = candidate_row // rows_per_batch
                    query_row_idx = candidate_row
                    batch_result = selected_batch
                    query_pos_result = candidate_row - selected_batch * rows_per_batch
                    valid_q_result = rows_per_batch
                    if const_info.has_seqused_q != 0:
                        valid_q_result = dtypes.int64(seqused_q_gm[selected_batch])
                    found = dtypes.int64(1)
    return query_row_idx, batch_result, query_pos_result, valid_q_result, found


@jit
def get_batch_kv_context(cu_seqlens_ori_kv, cu_seqlens_cmp_kv, const_info, batch_idx):
    ori_kv_base_row = batch_idx * const_info.ori_kv_capacity
    cmp_kv_base_row = batch_idx * const_info.cmp_kv_capacity
    ori_kv_length = dtypes.int64(const_info.ori_kv_capacity)
    cmp_kv_length = dtypes.int64(const_info.cmp_kv_capacity)
    if const_info.kv_mode == _KV_MODE_PA:
        # PA batches share physical storage.  The page-table row selects the
        # logical batch, so no linear KV prefix is applied.
        ori_kv_base_row = dtypes.int64(0)
        cmp_kv_base_row = dtypes.int64(0)
    elif const_info.batch_mode == _BATCH_MODE_TND:
        ori_kv_base_row = dtypes.int64(cu_seqlens_ori_kv[batch_idx])
        ori_kv_length = dtypes.int64(cu_seqlens_ori_kv[batch_idx + 1]) - ori_kv_base_row
        if const_info.has_cmp != 0:
            cmp_kv_base_row = dtypes.int64(cu_seqlens_cmp_kv[batch_idx])
            cmp_kv_length = (
                dtypes.int64(cu_seqlens_cmp_kv[batch_idx + 1]) - cmp_kv_base_row
            )
    if const_info.has_cmp == 0:
        cmp_kv_base_row = dtypes.int64(0)
        cmp_kv_length = dtypes.int64(0)
    return ori_kv_base_row, cmp_kv_base_row, ori_kv_length, cmp_kv_length


@jit
def calc_sparse_query_kv_lengths(
    ori_topk_length_gm,
    cmp_topk_length_gm,
    seqused_ori_kv_gm,
    seqused_cmp_kv_gm,
    cmp_residual_kv_gm,
    cu_seqlens_ori_kv,
    cu_seqlens_cmp_kv,
    const_info,
    query_row_idx,
    batch_idx,
    query_seq_idx,
    valid_query_count,
):
    ori_kv_base_row, cmp_kv_base_row, ori_kv_length, cmp_kv_length = (
        get_batch_kv_context(
            cu_seqlens_ori_kv, cu_seqlens_cmp_kv, const_info, batch_idx
        )
    )
    ori_effective = ori_kv_length
    if const_info.has_seqused_ori_kv != 0:
        requested_ori = dtypes.int64(seqused_ori_kv_gm[batch_idx])
        ori_effective = scalar_select(
            requested_ori > 0,
            scalar_select(requested_ori < ori_kv_length, requested_ori, ori_kv_length),
            0,
        )
    cmp_effective = cmp_kv_length
    if const_info.has_cmp != 0 and const_info.has_seqused_cmp_kv != 0:
        requested_cmp_kv = dtypes.int64(seqused_cmp_kv_gm[batch_idx])
        cmp_effective = scalar_select(
            requested_cmp_kv > 0,
            scalar_select(
                requested_cmp_kv < cmp_kv_length, requested_cmp_kv, cmp_kv_length
            ),
            0,
        )

    ori_kv_start_row = dtypes.int64(0)
    ori_end = ori_effective
    if const_info.ori_mask_mode == _MASK_CAUSAL:
        ori_end = ori_effective - valid_query_count + query_seq_idx + 1
        ori_end = scalar_select(
            ori_end > 0,
            scalar_select(ori_end < ori_effective, ori_end, ori_effective),
            0,
        )
    if const_info.ori_mask_mode == _MASK_SLIDING_WINDOW:
        if const_info.ori_win_left >= 0:
            ori_kv_start_row = (
                ori_effective
                - valid_query_count
                + query_seq_idx
                - const_info.ori_win_left
            )
            ori_kv_start_row = scalar_select(
                ori_kv_start_row > 0,
                scalar_select(
                    ori_kv_start_row < ori_effective, ori_kv_start_row, ori_effective
                ),
                0,
            )
        if const_info.ori_win_right >= 0:
            ori_end = (
                ori_effective
                - valid_query_count
                + query_seq_idx
                + const_info.ori_win_right
                + 1
            )
            ori_end = scalar_select(
                ori_end > 0,
                scalar_select(ori_end < ori_effective, ori_end, ori_effective),
                0,
            )
    ori_valid_kv_length = scalar_select(
        ori_end > ori_kv_start_row, ori_end - ori_kv_start_row, 0
    )
    if const_info.is_ori_dense == 0:
        ori_valid_kv_length = scalar_select(
            ori_valid_kv_length < const_info.ori_sparse_index_width,
            ori_valid_kv_length,
            const_info.ori_sparse_index_width,
        )
        if const_info.has_ori_topk_length != 0:
            requested = dtypes.int64(ori_topk_length_gm[query_row_idx])
            ori_valid_kv_length = scalar_select(
                requested > 0,
                scalar_select(
                    ori_valid_kv_length < requested, ori_valid_kv_length, requested
                ),
                0,
            )

    cmp_valid_kv_length = dtypes.int64(0)
    if const_info.has_cmp != 0:
        cmp_limit = cmp_effective
        if const_info.cmp_mask_mode == _MASK_CAUSAL:
            restored = cmp_effective * const_info.cmp_ratio
            if const_info.cmp_ratio != 1:
                restored = restored + dtypes.int64(cmp_residual_kv_gm[batch_idx])
            cmp_limit = (
                restored - valid_query_count + query_seq_idx + 1
            ) // const_info.cmp_ratio
            cmp_limit = scalar_select(
                cmp_limit > 0,
                scalar_select(cmp_limit < cmp_effective, cmp_limit, cmp_effective),
                0,
            )
        cmp_valid_kv_length = scalar_select(
            cmp_limit < const_info.cmp_sparse_index_width,
            cmp_limit,
            const_info.cmp_sparse_index_width,
        )
        if const_info.has_cmp_topk_length != 0:
            requested_cmp = dtypes.int64(cmp_topk_length_gm[query_row_idx])
            cmp_valid_kv_length = scalar_select(
                requested_cmp > 0,
                scalar_select(
                    cmp_valid_kv_length < requested_cmp,
                    cmp_valid_kv_length,
                    requested_cmp,
                ),
                0,
            )
    ori_tiles = (ori_valid_kv_length + _BASE_TILE_SIZE - 1) // _BASE_TILE_SIZE
    return (
        ori_kv_base_row,
        cmp_kv_base_row,
        ori_kv_length,
        cmp_kv_length,
        ori_kv_start_row,
        ori_valid_kv_length,
        cmp_valid_kv_length,
        ori_tiles,
        ori_tiles + (cmp_valid_kv_length + _BASE_TILE_SIZE - 1) // _BASE_TILE_SIZE,
    )


@jit
def get_next_nonempty_metadata_task(
    ori_topk_length_gm,
    cmp_topk_length_gm,
    seqused_ori_kv_gm,
    seqused_cmp_kv_gm,
    cmp_residual_kv_gm,
    metadata_gm,
    seqused_q_gm,
    cu_seqlens_q,
    cu_seqlens_ori_kv,
    cu_seqlens_cmp_kv,
    const_info,
    metadata_task_idx,
    metadata_task_count,
):
    query_row_idx = dtypes.int64(0)
    batch_idx = dtypes.int64(0)
    ori_kv_base_row = dtypes.int64(0)
    cmp_kv_base_row = dtypes.int64(0)
    ori_kv_length = dtypes.int64(0)
    cmp_kv_length = dtypes.int64(0)
    ori_kv_start_row = dtypes.int64(0)
    ori_valid_kv_length = dtypes.int64(0)
    cmp_valid_kv_length = dtypes.int64(0)
    ori_tile_count = dtypes.int64(0)
    kv_tile_count = dtypes.int64(0)
    found = dtypes.int64(0)
    scan = scalar_select(metadata_task_idx < metadata_task_count, 1, 0)
    while scan != 0:
        query_row_idx, batch_idx, query_seq_idx, valid_query_count, context_found = (
            get_metadata_task_context(
                metadata_gm, seqused_q_gm, cu_seqlens_q, const_info, metadata_task_idx
            )
        )
        if context_found != 0 and query_seq_idx < valid_query_count:
            (
                ori_kv_base_row,
                cmp_kv_base_row,
                ori_kv_length,
                cmp_kv_length,
                ori_kv_start_row,
                ori_valid_kv_length,
                cmp_valid_kv_length,
                ori_tile_count,
                kv_tile_count,
            ) = calc_sparse_query_kv_lengths(
                ori_topk_length_gm,
                cmp_topk_length_gm,
                seqused_ori_kv_gm,
                seqused_cmp_kv_gm,
                cmp_residual_kv_gm,
                cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv,
                const_info,
                query_row_idx,
                batch_idx,
                query_seq_idx,
                valid_query_count,
            )
        found = scalar_select(
            ori_valid_kv_length > 0, 1, scalar_select(cmp_valid_kv_length > 0, 1, 0)
        )
        if found == 0:
            metadata_task_idx = metadata_task_idx + 1
        scan = scalar_select(
            metadata_task_idx < metadata_task_count, scalar_select(found == 0, 1, 0), 0
        )
    return (
        metadata_task_idx,
        query_row_idx,
        batch_idx,
        ori_kv_base_row,
        cmp_kv_base_row,
        ori_kv_length,
        cmp_kv_length,
        ori_kv_start_row,
        ori_valid_kv_length,
        cmp_valid_kv_length,
        ori_tile_count,
        kv_tile_count,
        found,
    )


@kernel
class SparseN64Kernel:
    # Coordinates Cube and Vector stages.

    def __init__(
        self,
        query_head_tile_size,
        has_sink,
        return_softmax_lse,
        dtype,
        batch_mode,
        kv_mode,
        is_ori_dense,
    ):
        self.query_head_tile_size = query_head_tile_size
        self.dtype = dtype
        self.return_softmax_lse = return_softmax_lse
        self.has_sink = has_sink
        self.batch_mode = batch_mode
        self.kv_mode = kv_mode
        self.is_ori_dense = is_ori_dense

    def __call__(
        self,
        query_gm: Tensor,
        ori_kv_gm: Tensor,
        cmp_kv_gm: Tensor,
        ori_sparse_indices_gm: Tensor,
        cmp_sparse_indices_gm: Tensor,
        ori_block_table_gm: Tensor,
        cmp_block_table_gm: Tensor,
        ori_kv_physical_rows_gm: Tensor,
        cmp_kv_physical_rows_gm: Tensor,
        ori_topk_length_gm: Tensor,
        cmp_topk_length_gm: Tensor,
        seqused_ori_kv_gm: Tensor,
        seqused_cmp_kv_gm: Tensor,
        cmp_residual_kv_gm: Tensor,
        metadata_gm: Tensor,
        seqused_q_gm: Tensor,
        cu_seqlens_q: Tensor,
        cu_seqlens_ori_kv: Tensor,
        cu_seqlens_cmp_kv: Tensor,
        sinks_gm: Tensor,
        attention_out_gm: Tensor,
        softmax_lse_gm: Tensor,
        const_info: HighPerfSmlaConstInfo,
    ):
        if const_expr(self.kv_mode == _KV_MODE_PA):
            self.init_pa_address_buffers()
            if is_aic():
                cube_sync_intra_wait(PIPE.MTE1, _PA_SYNC_ID)
                cube_sync_intra_wait(
                    PIPE.MTE1, _PA_SYNC_ID + _PA_SECOND_VECTOR_SYNC_OFFSET
                )
                cube_sync_block_arrive(PIPE.MTE1, _PA_SYNC_ID, mode=2)
            else:
                self.calc_kv_physical_rows(
                    ori_sparse_indices_gm,
                    cmp_sparse_indices_gm,
                    ori_kv_physical_rows_gm,
                    cmp_kv_physical_rows_gm,
                    ori_block_table_gm,
                    cmp_block_table_gm,
                    ori_topk_length_gm,
                    cmp_topk_length_gm,
                    seqused_ori_kv_gm,
                    seqused_cmp_kv_gm,
                    cmp_residual_kv_gm,
                    metadata_gm,
                    seqused_q_gm,
                    cu_seqlens_q,
                    cu_seqlens_ori_kv,
                    cu_seqlens_cmp_kv,
                    const_info,
                )
            # Both AIV stores and the AIC relay are complete here. Reclaim the
            # allocator-owned PA arena before declaring main-pipeline storage.
            onchip_rewind(reset_sync_id=False)
        self.init_main_pipeline_buffers()
        if is_aic():
            self.process_cube(
                query_gm,
                ori_topk_length_gm,
                cmp_topk_length_gm,
                seqused_ori_kv_gm,
                seqused_cmp_kv_gm,
                cmp_residual_kv_gm,
                metadata_gm,
                seqused_q_gm,
                cu_seqlens_q,
                cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv,
                const_info,
            )
        else:
            self.process_vector(
                ori_kv_gm,
                cmp_kv_gm,
                ori_sparse_indices_gm,
                cmp_sparse_indices_gm,
                ori_block_table_gm,
                cmp_block_table_gm,
                ori_kv_physical_rows_gm,
                cmp_kv_physical_rows_gm,
                ori_topk_length_gm,
                cmp_topk_length_gm,
                seqused_ori_kv_gm,
                seqused_cmp_kv_gm,
                cmp_residual_kv_gm,
                metadata_gm,
                seqused_q_gm,
                cu_seqlens_q,
                cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv,
                sinks_gm,
                attention_out_gm,
                softmax_lse_gm,
                const_info,
            )

    @jit
    def init_main_pipeline_buffers(self):
        # PA prepass UB is dead and rewound before these long-lived buffers exist.
        self.buffer_storage = HighPerfSmlaStorage(
            self.query_head_tile_size, self.has_sink, self.dtype
        )
        self.cube_block = HighPerfSmlaMatmul(self.buffer_storage)
        self.vec0_block = HighPerfSmlaVec0Indices(self.buffer_storage, self.batch_mode)
        self.vector_block = HighPerfSmlaVector(
            self.buffer_storage, self.return_softmax_lse
        )
        self.run_info_delay = DelayLineGroup(
            _PIPELINE_SLOTS,
            "active_query_idx",
            "query_row_idx",
            "kv_tile_idx",
            "valid_kv_length",
            "kv_tile_count",
            "prior_query_tile_count",
            "stream_kv_tile_idx",
            "batch_idx",
            "ori_kv_base_row",
            "cmp_kv_base_row",
            "ori_kv_length",
            "cmp_kv_length",
            "ori_kv_start_row",
            "is_cmp",
        )

    @jit
    def init_pa_address_buffers(self):
        # Match MQSMLA's pipelined prepass: page-table, index slots and address
        # slots have distinct protocol owners.  The allocator still packs their
        # combined 248 KiB footprint, and rewind reclaims it before the main UB.
        self.pa_block_table_ub = Buffer(
            MemLoc.UB, (1, _PA_BLOCK_TABLE_CAPACITY), dtypes.int32
        )
        self.pa_sparse_indices_ub_ring = RingBuffer(
            MemLoc.UB,
            (1, _PA_BUFFER_CAPACITY),
            dtypes.int32,
            depth=_PA_BUFFER_COUNT,
        )
        self.pa_physical_rows_ub_ring = RingBuffer(
            MemLoc.UB,
            (1, _PA_BUFFER_CAPACITY),
            dtypes.int32,
            depth=_PA_BUFFER_COUNT,
        )

    @jit
    def calc_kv_physical_rows_chunk(
        self,
        indices_ub,
        addresses_ub,
        chunk,
        padded_width,
        valid_length,
        page_size,
        page_stride_rows,
        page_size_reciprocal,
        page_size_log2,
        page_size_is_power_of_two,
        row_offset=0,
    ):
        lanes = scalar_select(
            padded_width - chunk * _FP32_VECTOR_LENGTH < _FP32_VECTOR_LENGTH,
            padded_width - chunk * _FP32_VECTOR_LENGTH,
            _FP32_VECTOR_LENGTH,
        )
        mask = rr.update_mask(lanes, elem_bits=32)[0]
        lane = rr.varange(0, dtypes.uint32)
        column = rr.vadds(lane, chunk * _FP32_VECTOR_LENGTH, mask=mask)
        column = rr.vmins(column, dtypes.uint32(valid_length - 1), mask=mask)
        buffer_column = rr.vadds(column, row_offset, mask=mask)
        logical_kv_row_reg = rr.vgather(indices_ub, buffer_column, mask=mask)
        logical_kv_row_reg = rr.vmaxs(logical_kv_row_reg, 0, mask=mask)
        logical_kv_row_u32_reg = rr.vreinterpret(logical_kv_row_reg, dtypes.uint32)
        logical_page_idx_reg = logical_kv_row_u32_reg
        in_page_row_reg = logical_kv_row_u32_reg
        if page_size == 1:
            in_page_row_reg = rr.vdups(0, dtypes.uint32, mask=mask)
        else:
            _low, logical_page_idx_reg = rr.vmull(
                logical_kv_row_u32_reg,
                rr.vdups(page_size_reciprocal, dtypes.uint32, mask=mask),
                dtypes.uint32,
                mask=mask,
            )
            product = rr.vmuls(logical_page_idx_reg, page_size, mask=mask)
            logical_page_idx_reg = rr.vselect(
                rr.vsub(
                    logical_page_idx_reg,
                    rr.vdups(1, dtypes.uint32, mask=mask),
                    mask=mask,
                ),
                logical_page_idx_reg,
                cond_mask=rr.vgt(product, logical_kv_row_u32_reg, mask=mask),
            )
            in_page_row_reg = rr.vsub(
                logical_kv_row_u32_reg,
                rr.vmuls(logical_page_idx_reg, page_size, mask=mask),
                mask=mask,
            )
        physical_block_idx_reg = rr.vreinterpret(
            rr.vgather(self.pa_block_table_ub, logical_page_idx_reg, mask=mask),
            dtypes.uint32,
        )
        physical_kv_row_reg = rr.vadd(
            rr.vmuls(physical_block_idx_reg, page_stride_rows, mask=mask),
            in_page_row_reg,
            mask=mask,
        )
        rr.vstore(
            addresses_ub,
            row_offset + chunk * _FP32_VECTOR_LENGTH,
            rr.vreinterpret(physical_kv_row_reg, dtypes.int32),
            mask,
        )

    @jit
    def calc_kv_physical_rows_for_query(
        self,
        sparse_indices_gm,
        kv_physical_rows_gm,
        block_table_gm,
        query_row_idx,
        batch_idx,
        valid_length,
        sparse_index_stride,
        block_table_width,
        page_size,
        page_stride_rows,
        page_size_reciprocal,
        page_size_log2,
        page_size_is_power_of_two,
        staged_batch,
    ):
        padded_width = (sparse_index_stride + _NZ_ROW - 1) // _NZ_ROW * _NZ_ROW
        indices_ub = self.pa_sparse_indices_ub_ring.next_write()
        addresses_ub = self.pa_physical_rows_ub_ring.next_write()
        if batch_idx != staged_batch:
            mem_copy(
                reinterpret(self.pa_block_table_ub, shape=(1, block_table_width)),
                block_table_gm[
                    batch_idx * block_table_width: (batch_idx + 1) * block_table_width,
                ].view(1, block_table_width),
            )  # fmt: skip
        mem_copy(
            reinterpret(indices_ub, shape=(1, sparse_index_stride)),
            sparse_indices_gm[
                query_row_idx * sparse_index_stride: (query_row_idx + 1)
                * sparse_index_stride,
            ].view(1, sparse_index_stride),
        )  # fmt: skip
        with vf(mode="simd"):
            for chunk in range(
                (padded_width + _FP32_VECTOR_LENGTH - 1) // _FP32_VECTOR_LENGTH
            ):
                self.calc_kv_physical_rows_chunk(
                    indices_ub,
                    addresses_ub,
                    chunk,
                    padded_width,
                    valid_length,
                    page_size,
                    page_stride_rows,
                    page_size_reciprocal,
                    page_size_log2,
                    page_size_is_power_of_two,
                )
        mem_copy(
            kv_physical_rows_gm[
                query_row_idx * padded_width: (query_row_idx + 1) * padded_width,
            ].view(1, padded_width),
            reinterpret(addresses_ub, shape=(1, padded_width)),
        )  # fmt: skip
        # Preserve fallback slot 0; selection alone has no storage access.
        self.pa_sparse_indices_ub_ring.next_write(bare=True)
        self.pa_physical_rows_ub_ring.next_write(bare=True)
        return batch_idx

    @jit
    def calc_kv_physical_rows_for_query_group(
        self,
        indices_ub,
        addresses_ub,
        sparse_indices_gm,
        kv_physical_rows_gm,
        block_table_gm,
        ori_topk_length_gm,
        cmp_topk_length_gm,
        seqused_ori_kv_gm,
        seqused_cmp_kv_gm,
        cmp_residual_kv_gm,
        metadata_gm,
        seqused_q_gm,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        const_info,
        task_start,
        rows,
        sparse_index_stride,
        block_table_width,
        page_size,
        page_stride_rows,
        page_size_reciprocal,
        page_size_log2,
        page_size_is_power_of_two,
        staged_batch,
        is_cmp: cannbotdsl.Constexpr[bool],
    ):
        """Batch consecutive index rows through one MTE2/VF/MTE3 prepass group."""
        padded_width = (sparse_index_stride + _NZ_ROW - 1) // _NZ_ROW * _NZ_ROW
        first_query_row, _batch, _query_pos, _valid_q, _found = (
            get_metadata_task_context(
                metadata_gm, seqused_q_gm, cu_seqlens_q, const_info, task_start
            )
        )
        # Keep grouped DMA for all-valid Query groups. Mixed groups must not
        # include padding rows in an index DMA, even as unused neighbours.
        all_queries_valid = dtypes.int64(1)
        for row in range_constexpr(rows):
            _qr, _bi, _qp, _vq, _ok = get_metadata_task_context(
                metadata_gm,
                seqused_q_gm,
                cu_seqlens_q,
                const_info,
                task_start + row,
            )
            if _ok == 0 or _qp >= _vq:
                all_queries_valid = dtypes.int64(0)
        indices_loaded = dtypes.int64(0)
        for row in range_constexpr(rows):
            query_row_idx, batch_idx, query_seq_idx, valid_query_count, _found = (
                get_metadata_task_context(
                    metadata_gm,
                    seqused_q_gm,
                    cu_seqlens_q,
                    const_info,
                    task_start + row,
                )
            )
            if _found != 0 and query_seq_idx < valid_query_count:
                (
                    _ori_base,
                    _cmp_base,
                    _ori_kv_len,
                    _cmp_kv_len,
                    _ori_begin,
                    ori_valid_kv_length,
                    cmp_valid_kv_length,
                    _ori_tile_count,
                    _kv_tile_count,
                ) = calc_sparse_query_kv_lengths(
                    ori_topk_length_gm,
                    cmp_topk_length_gm,
                    seqused_ori_kv_gm,
                    seqused_cmp_kv_gm,
                    cmp_residual_kv_gm,
                    cu_seqlens_ori_kv,
                    cu_seqlens_cmp_kv,
                    const_info,
                    query_row_idx,
                    batch_idx,
                    query_seq_idx,
                    valid_query_count,
                )
                valid_length = (
                    cmp_valid_kv_length if const_expr(is_cmp) else ori_valid_kv_length
                )
                if valid_length > 0:
                    if all_queries_valid != 0:
                        if indices_loaded == 0:
                            mem_copy(
                                reinterpret(
                                    indices_ub, shape=(1, rows * sparse_index_stride)
                                ),
                                sparse_indices_gm[
                                    first_query_row * sparse_index_stride: (
                                        first_query_row + rows
                                    )
                                    * sparse_index_stride,
                                ].view(1, rows * sparse_index_stride),
                            )  # fmt: skip
                            indices_loaded = dtypes.int64(1)
                    else:
                        mem_copy(
                            reinterpret(indices_ub, shape=(rows * padded_width,))[
                                row * padded_width: row * padded_width
                                + sparse_index_stride,
                            ].view(1, sparse_index_stride),
                            sparse_indices_gm[
                                query_row_idx * sparse_index_stride: (
                                    query_row_idx + 1
                                )
                                * sparse_index_stride,
                            ].view(1, sparse_index_stride),
                        )  # fmt: skip
                    if batch_idx != staged_batch:
                        mem_copy(
                            reinterpret(
                                self.pa_block_table_ub, shape=(1, block_table_width)
                            ),
                            block_table_gm[
                                batch_idx * block_table_width: (batch_idx + 1)
                                * block_table_width,
                            ].view(1, block_table_width),
                        )  # fmt: skip
                    with vf(mode="simd"):
                        for chunk in range(
                            (padded_width + _FP32_VECTOR_LENGTH - 1)
                            // _FP32_VECTOR_LENGTH
                        ):
                            self.calc_kv_physical_rows_chunk(
                                indices_ub,
                                addresses_ub,
                                chunk,
                                padded_width,
                                valid_length,
                                page_size,
                                page_stride_rows,
                                page_size_reciprocal,
                                page_size_log2,
                                page_size_is_power_of_two,
                                row * padded_width,
                            )
                    staged_batch = batch_idx
                    if all_queries_valid == 0:
                        mem_copy(
                            kv_physical_rows_gm[
                                query_row_idx * padded_width: (query_row_idx + 1)
                                * padded_width,
                            ].view(1, padded_width),
                            reinterpret(addresses_ub, shape=(rows * padded_width,))[
                                row * padded_width: (row + 1) * padded_width,
                            ].view(1, padded_width),
                        )  # fmt: skip
        if all_queries_valid != 0 and indices_loaded != 0:
            mem_copy(
                kv_physical_rows_gm[
                    first_query_row * padded_width: (first_query_row + rows)
                    * padded_width,
                ].view(1, rows * padded_width),
                reinterpret(addresses_ub, shape=(1, rows * padded_width)),
            )  # fmt: skip
        return staged_batch

    @jit
    def calc_kv_physical_rows_for_query_batch(
        self,
        sparse_indices_gm,
        kv_physical_rows_gm,
        block_table_gm,
        ori_topk_length_gm,
        cmp_topk_length_gm,
        seqused_ori_kv_gm,
        seqused_cmp_kv_gm,
        cmp_residual_kv_gm,
        metadata_gm,
        seqused_q_gm,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        const_info,
        task_start,
        rows,
        sparse_index_stride,
        block_table_width,
        page_size,
        page_stride_rows,
        page_size_reciprocal,
        page_size_log2,
        page_size_is_power_of_two,
        staged_batch,
        is_cmp: cannbotdsl.Constexpr[bool],
    ):
        indices_ub = self.pa_sparse_indices_ub_ring.next_write()
        addresses_ub = self.pa_physical_rows_ub_ring.next_write()
        staged_batch = self.calc_kv_physical_rows_for_query_group(
            indices_ub,
            addresses_ub,
            sparse_indices_gm,
            kv_physical_rows_gm,
            block_table_gm,
            ori_topk_length_gm,
            cmp_topk_length_gm,
            seqused_ori_kv_gm,
            seqused_cmp_kv_gm,
            cmp_residual_kv_gm,
            metadata_gm,
            seqused_q_gm,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
            const_info,
            task_start,
            _PA_QUERY_GROUP_SIZE,
            sparse_index_stride,
            block_table_width,
            page_size,
            page_stride_rows,
            page_size_reciprocal,
            page_size_log2,
            page_size_is_power_of_two,
            staged_batch,
            is_cmp,
        )
        indices_ub = self.pa_sparse_indices_ub_ring.next_write()
        addresses_ub = self.pa_physical_rows_ub_ring.next_write()
        return self.calc_kv_physical_rows_for_query_group(
            indices_ub,
            addresses_ub,
            sparse_indices_gm,
            kv_physical_rows_gm,
            block_table_gm,
            ori_topk_length_gm,
            cmp_topk_length_gm,
            seqused_ori_kv_gm,
            seqused_cmp_kv_gm,
            cmp_residual_kv_gm,
            metadata_gm,
            seqused_q_gm,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
            const_info,
            task_start + _PA_QUERY_GROUP_SIZE,
            _PA_QUERY_GROUP_SIZE,
            sparse_index_stride,
            block_table_width,
            page_size,
            page_stride_rows,
            page_size_reciprocal,
            page_size_log2,
            page_size_is_power_of_two,
            staged_batch,
            is_cmp,
        )

    @jit
    def calc_kv_physical_rows(
        self,
        ori_sparse_indices_gm,
        cmp_sparse_indices_gm,
        ori_kv_physical_rows_gm,
        cmp_kv_physical_rows_gm,
        ori_block_table_gm,
        cmp_block_table_gm,
        ori_topk_length_gm,
        cmp_topk_length_gm,
        seqused_ori_kv_gm,
        seqused_cmp_kv_gm,
        cmp_residual_kv_gm,
        metadata_gm,
        seqused_q_gm,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        const_info,
    ):
        metadata_task_count = get_metadata_task_count(
            metadata_gm, seqused_q_gm, cu_seqlens_q, const_info
        )
        per = metadata_task_count // _VECTOR_SUBBLOCKS
        tail = metadata_task_count % _VECTOR_SUBBLOCKS
        worker = dtypes.int64(get_subblock_id())
        start = per * worker + scalar_select(worker < tail, worker, tail)
        count = per + scalar_select(worker < tail, 1, 0)
        end = start + count
        # ORI and CMP share the same UB staging buffers.  Process complete
        # streams instead of alternating them per query so a CMP table never
        # replaces the cached ORI table before the next ORI row.
        if const_expr(not self.is_ori_dense):
            ori_staged_batch = dtypes.int64(-1)
            ori_padded_width = (
                (const_info.ori_sparse_index_width + _NZ_ROW - 1) // _NZ_ROW * _NZ_ROW
            )
            ori_can_batch = scalar_select(
                ori_padded_width == const_info.ori_sparse_index_width, 1, 0
            ) * scalar_select(
                ori_padded_width * _PA_QUERY_GROUP_SIZE <= _PA_BUFFER_CAPACITY, 1, 0
            )
            ori_batch_rows = scalar_select(ori_can_batch != 0, _PA_BATCH_ROWS, 1)
            ori_bulk_end = start
            if ori_batch_rows == _PA_BATCH_ROWS:
                ori_bulk_end = start + count // _PA_BATCH_ROWS * _PA_BATCH_ROWS
                for metadata_task_idx in range(start, ori_bulk_end, _PA_BATCH_ROWS):
                    ori_staged_batch = self.calc_kv_physical_rows_for_query_batch(
                        ori_sparse_indices_gm,
                        ori_kv_physical_rows_gm,
                        ori_block_table_gm,
                        ori_topk_length_gm,
                        cmp_topk_length_gm,
                        seqused_ori_kv_gm,
                        seqused_cmp_kv_gm,
                        cmp_residual_kv_gm,
                        metadata_gm,
                        seqused_q_gm,
                        cu_seqlens_q,
                        cu_seqlens_ori_kv,
                        cu_seqlens_cmp_kv,
                        const_info,
                        metadata_task_idx,
                        _PA_BATCH_ROWS,
                        const_info.ori_sparse_index_width,
                        const_info.ori_block_table_width,
                        const_info.ori_page_size,
                        const_info.ori_page_stride_rows,
                        const_info.ori_page_size_reciprocal,
                        const_info.ori_page_size_log2,
                        const_info.ori_page_size_is_power_of_two,
                        ori_staged_batch,
                        False,
                    )
            for metadata_task_idx in range(ori_bulk_end, end):
                query_row_idx, batch_idx, query_seq_idx, valid_query_count, _found = (
                    get_metadata_task_context(
                        metadata_gm,
                        seqused_q_gm,
                        cu_seqlens_q,
                        const_info,
                        metadata_task_idx,
                    )
                )
                if _found != 0 and query_seq_idx < valid_query_count:
                    (
                        _ori_base,
                        _cmp_base,
                        _ori_kv_len,
                        _cmp_kv_len,
                        _ori_begin,
                        ori_valid_kv_length,
                        _cmp_length,
                        _ori_tile_count,
                        _kv_tile_count,
                    ) = calc_sparse_query_kv_lengths(
                        ori_topk_length_gm,
                        cmp_topk_length_gm,
                        seqused_ori_kv_gm,
                        seqused_cmp_kv_gm,
                        cmp_residual_kv_gm,
                        cu_seqlens_ori_kv,
                        cu_seqlens_cmp_kv,
                        const_info,
                        query_row_idx,
                        batch_idx,
                        query_seq_idx,
                        valid_query_count,
                    )
                    if ori_valid_kv_length > 0:
                        ori_staged_batch = self.calc_kv_physical_rows_for_query(
                            ori_sparse_indices_gm,
                            ori_kv_physical_rows_gm,
                            ori_block_table_gm,
                            query_row_idx,
                            batch_idx,
                            ori_valid_kv_length,
                            const_info.ori_sparse_index_width,
                            const_info.ori_block_table_width,
                            const_info.ori_page_size,
                            const_info.ori_page_stride_rows,
                            const_info.ori_page_size_reciprocal,
                            const_info.ori_page_size_log2,
                            const_info.ori_page_size_is_power_of_two,
                            ori_staged_batch,
                        )
        if const_info.has_cmp != 0:
            cmp_staged_batch = dtypes.int64(-1)
            cmp_padded_width = (
                (const_info.cmp_sparse_index_width + _NZ_ROW - 1) // _NZ_ROW * _NZ_ROW
            )
            cmp_can_batch = scalar_select(
                cmp_padded_width == const_info.cmp_sparse_index_width, 1, 0
            ) * scalar_select(
                cmp_padded_width * _PA_QUERY_GROUP_SIZE <= _PA_BUFFER_CAPACITY, 1, 0
            )
            cmp_batch_rows = scalar_select(cmp_can_batch != 0, _PA_BATCH_ROWS, 1)
            cmp_bulk_end = start
            if cmp_batch_rows == _PA_BATCH_ROWS:
                cmp_bulk_end = start + count // _PA_BATCH_ROWS * _PA_BATCH_ROWS
                for metadata_task_idx in range(start, cmp_bulk_end, _PA_BATCH_ROWS):
                    cmp_staged_batch = self.calc_kv_physical_rows_for_query_batch(
                        cmp_sparse_indices_gm,
                        cmp_kv_physical_rows_gm,
                        cmp_block_table_gm,
                        ori_topk_length_gm,
                        cmp_topk_length_gm,
                        seqused_ori_kv_gm,
                        seqused_cmp_kv_gm,
                        cmp_residual_kv_gm,
                        metadata_gm,
                        seqused_q_gm,
                        cu_seqlens_q,
                        cu_seqlens_ori_kv,
                        cu_seqlens_cmp_kv,
                        const_info,
                        metadata_task_idx,
                        _PA_BATCH_ROWS,
                        const_info.cmp_sparse_index_width,
                        const_info.cmp_block_table_width,
                        const_info.cmp_page_size,
                        const_info.cmp_page_stride_rows,
                        const_info.cmp_page_size_reciprocal,
                        const_info.cmp_page_size_log2,
                        const_info.cmp_page_size_is_power_of_two,
                        cmp_staged_batch,
                        True,
                    )
            for metadata_task_idx in range(cmp_bulk_end, end):
                query_row_idx, batch_idx, query_seq_idx, valid_query_count, _found = (
                    get_metadata_task_context(
                        metadata_gm,
                        seqused_q_gm,
                        cu_seqlens_q,
                        const_info,
                        metadata_task_idx,
                    )
                )
                if _found != 0 and query_seq_idx < valid_query_count:
                    (
                        _ori_base,
                        _cmp_base,
                        _ori_kv_len,
                        _cmp_kv_len,
                        _ori_begin,
                        _ori_length,
                        cmp_valid_kv_length,
                        _ori_tile_count,
                        _kv_tile_count,
                    ) = calc_sparse_query_kv_lengths(
                        ori_topk_length_gm,
                        cmp_topk_length_gm,
                        seqused_ori_kv_gm,
                        seqused_cmp_kv_gm,
                        cmp_residual_kv_gm,
                        cu_seqlens_ori_kv,
                        cu_seqlens_cmp_kv,
                        const_info,
                        query_row_idx,
                        batch_idx,
                        query_seq_idx,
                        valid_query_count,
                    )
                    if cmp_valid_kv_length > 0:
                        cmp_staged_batch = self.calc_kv_physical_rows_for_query(
                            cmp_sparse_indices_gm,
                            cmp_kv_physical_rows_gm,
                            cmp_block_table_gm,
                            query_row_idx,
                            batch_idx,
                            cmp_valid_kv_length,
                            const_info.cmp_sparse_index_width,
                            const_info.cmp_block_table_width,
                            const_info.cmp_page_size,
                            const_info.cmp_page_stride_rows,
                            const_info.cmp_page_size_reciprocal,
                            const_info.cmp_page_size_log2,
                            const_info.cmp_page_size_is_power_of_two,
                            cmp_staged_batch,
                        )
        vec_sync_intra_arrive(PIPE.MTE3, _PA_SYNC_ID)
        vec_sync_block_wait(PIPE.S, _PA_SYNC_ID, mode=2)

    @jit
    def process_vec1(
        self,
        softmax_lse_gm,
        const_info,
        active_query_idx,
        query_row_idx,
        kv_tile_idx,
        stream_kv_tile_idx,
        kv_len,
        kv_tile_count,
        valid_head_count,
        bmm1_res_ub,
        vec1_res_ub,
        bmm2_a_l1,
        valid_kv_length,
    ):
        valid_segment_kv_rows = scalar_select(
            valid_kv_length < kv_len, valid_kv_length, kv_len
        )
        remaining_kv_rows = valid_segment_kv_rows - stream_kv_tile_idx * _BASE_TILE_SIZE
        valid_tile_kv_rows = scalar_select(
            remaining_kv_rows > 0,
            scalar_select(
                remaining_kv_rows < _BASE_TILE_SIZE, remaining_kv_rows, _BASE_TILE_SIZE
            ),
            0,
        )
        if valid_tile_kv_rows == _BASE_TILE_SIZE:
            if kv_tile_idx == 0:
                self.vector_block.compute_vec1_softmax(
                    active_query_idx,
                    kv_tile_idx,
                    const_info.softmax_scale,
                    True,
                    True,
                    bmm1_res_ub,
                    vec1_res_ub,
                    valid_tile_kv_rows,
                )
            else:
                self.vector_block.compute_vec1_softmax(
                    active_query_idx,
                    kv_tile_idx,
                    const_info.softmax_scale,
                    False,
                    True,
                    bmm1_res_ub,
                    vec1_res_ub,
                    valid_tile_kv_rows,
                )
        elif kv_tile_idx == 0:
            self.vector_block.compute_vec1_softmax(
                active_query_idx,
                kv_tile_idx,
                const_info.softmax_scale,
                True,
                False,
                bmm1_res_ub,
                vec1_res_ub,
                valid_tile_kv_rows,
            )
        else:
            self.vector_block.compute_vec1_softmax(
                active_query_idx,
                kv_tile_idx,
                const_info.softmax_scale,
                False,
                False,
                bmm1_res_ub,
                vec1_res_ub,
                valid_tile_kv_rows,
            )
        # P 的物理 NZ 列距按 ceil16(N1)；两个 AIV 仅写各自真实行数。
        if get_subblock_id() == 0:
            source = view_padded_nz_columns(
                vec1_res_ub,
                _BASE_TILE_SIZE,
                self.buffer_storage.lower_head_count,
                self.buffer_storage.subblock_query_head_count + 1,
            )
            target = bmm2_a_l1.view(
                _TILE_NZ_TILES, self.buffer_storage.aligned_head_count * _NZ_ROW
            )[None, 0: self.buffer_storage.lower_head_count * _NZ_ROW]  # fmt: skip
            mem_copy(target, source)
        elif const_expr(self.buffer_storage.upper_head_count > 0):
            source = view_padded_nz_columns(
                vec1_res_ub,
                _BASE_TILE_SIZE,
                self.buffer_storage.upper_head_count,
                self.buffer_storage.subblock_query_head_count + 1,
            )
            target = bmm2_a_l1.view(
                _TILE_NZ_TILES, self.buffer_storage.aligned_head_count * _NZ_ROW
            )[
                None,
                self.buffer_storage.lower_head_count
                * _NZ_ROW: self.buffer_storage.query_head_tile_size * _NZ_ROW,
            ]  # fmt: skip
            mem_copy(target, source)
        if kv_tile_idx + 1 == kv_tile_count:
            self.vector_block.finalize_vec1_softmax_state(
                active_query_idx, const_info.softmax_scale
            )
            if const_expr(self.return_softmax_lse):
                mem_copy(
                    softmax_lse_gm[
                        query_row_idx * const_info.num_heads_q
                        + get_subblock_id()
                        * self.buffer_storage.subblock_query_head_count: (
                            query_row_idx * const_info.num_heads_q
                            + get_subblock_id()
                            * self.buffer_storage.subblock_query_head_count
                        )
                        + valid_head_count,
                    ],
                    self.buffer_storage.softmax_max_ub[
                        active_query_idx % 2 * _FP32_VECTOR_LENGTH: (
                            active_query_idx % 2 * _FP32_VECTOR_LENGTH
                        )
                        + valid_head_count,
                    ],
                )  # fmt: skip

    @jit
    def process_vec2(
        self,
        attention_out_gm,
        const_info,
        active_query_idx,
        query_row_idx,
        kv_tile_idx,
        d_half_idx,
        kv_tile_count,
        valid_head_count,
        attention_acc_ub,
        bmm2_res_ub,
        attention_out_ub,
    ):
        is_last_tile = kv_tile_idx + 1 == kv_tile_count
        if kv_tile_idx == 0:
            # VF first-tile store: read and write the first FP32 lane of the PV accumulator.
            with vf(mode="simd"):
                first_element, _ = rr.update_mask(1, elem_bits=_F32_BITS)
                value = rr.vload(attention_acc_ub, 0)
                rr.vstore(attention_acc_ub, 0, value, first_element)
        if kv_tile_idx == 0:
            if is_last_tile:
                self.vector_block.compute_vec2_attention_out(
                    active_query_idx,
                    kv_tile_idx,
                    True,
                    True,
                    attention_acc_ub,
                    bmm2_res_ub,
                    attention_out_ub,
                )
        elif is_last_tile:
            self.vector_block.compute_vec2_attention_out(
                active_query_idx,
                kv_tile_idx,
                False,
                True,
                attention_acc_ub,
                bmm2_res_ub,
                attention_out_ub,
            )
        else:
            self.vector_block.compute_vec2_attention_out(
                active_query_idx,
                kv_tile_idx,
                False,
                False,
                attention_acc_ub,
                bmm2_res_ub,
                attention_out_ub,
            )
        if is_last_tile:
            if valid_head_count > 0:
                mem_copy(
                    attention_out_gm[
                        dtypes.int64(query_row_idx)
                        * const_info.num_heads_q
                        * _HEAD_DIM:,
                    ].view(const_info.num_heads_q, _HEAD_DIM)[
                        get_subblock_id()
                        * self.buffer_storage.subblock_query_head_count: get_subblock_id()
                        * self.buffer_storage.subblock_query_head_count
                        + valid_head_count,
                        d_half_idx * _RESULT_HALF_COLS: (d_half_idx + 1)
                        * _RESULT_HALF_COLS,
                    ],
                    attention_out_ub[0:valid_head_count, 0:_RESULT_HALF_COLS],
                    options=copy_options(l2_cache_ctl=_KV_COPY_L2_CACHE_CONTROL),
                )  # fmt: skip

    @jit
    def process_vector_step(
        self,
        ori_kv_gm,
        cmp_kv_gm,
        ori_sparse_indices_gm,
        cmp_sparse_indices_gm,
        ori_block_table_gm,
        cmp_block_table_gm,
        ori_kv_physical_rows_gm,
        cmp_kv_physical_rows_gm,
        attention_out_gm,
        softmax_lse_gm,
        const_info,
        pipeline_tick,
        active_query_idx,
        query_row_idx,
        kv_tile_idx,
        ori_valid_kv_length,
        cmp_valid_kv_length,
        ori_tile_count,
        kv_tile_count,
        prior_query_tile_count,
        batch_idx,
        ori_kv_base_row,
        cmp_kv_base_row,
        ori_kv_length,
        cmp_kv_length,
        ori_kv_start_row,
        valid_head_count,
        kv_lower_rows_l1_write,
        kv_upper_rows_l1_write,
        indices_read,
    ):
        is_cmp = kv_tile_idx >= ori_tile_count
        stream_kv_tile_idx = scalar_select(
            is_cmp, kv_tile_idx - ori_tile_count, kv_tile_idx
        )
        valid_len = scalar_select(is_cmp, cmp_valid_kv_length, ori_valid_kv_length)
        self.run_info_delay.push(
            active_query_idx=dtypes.int64(active_query_idx),
            query_row_idx=dtypes.int64(query_row_idx),
            kv_tile_idx=dtypes.int64(kv_tile_idx),
            valid_kv_length=dtypes.int64(valid_len),
            kv_tile_count=dtypes.int64(kv_tile_count),
            prior_query_tile_count=dtypes.int64(prior_query_tile_count),
            stream_kv_tile_idx=dtypes.int64(stream_kv_tile_idx),
            batch_idx=dtypes.int64(batch_idx),
            ori_kv_base_row=dtypes.int64(ori_kv_base_row),
            cmp_kv_base_row=dtypes.int64(cmp_kv_base_row),
            ori_kv_length=dtypes.int64(ori_kv_length),
            cmp_kv_length=dtypes.int64(cmp_kv_length),
            ori_kv_start_row=dtypes.int64(ori_kv_start_row),
            is_cmp=dtypes.int64(is_cmp),
        )
        bmm1_res_ub_read = self.buffer_storage.get_next_bmm1_res_ub(False)
        vec1_res_ub = self.buffer_storage.vec1_res_ub
        bmm2_a_l1_write = self.buffer_storage.cross_core_rings[
            "bmm2_a_l1_ring"
        ].next_write()
        (
            attention_acc_lower_d_ub,
            bmm2_res_lower_d_ub,
            attention_acc_upper_d_ub,
            bmm2_res_upper_d_ub,
        ) = self.buffer_storage.get_next_bmm2_res_ub_buffers(
            False,
        )
        attention_out_lower_d_ub = reinterpret(
            bmm2_res_lower_d_ub,
            self.buffer_storage.dtype,
            (self.buffer_storage.subblock_query_head_count, _RESULT_HALF_COLS),
            stride=(_RESULT_HALF_COLS * _F32_BYTES // _F16_BYTES, 1),
        )
        attention_out_upper_d_ub = reinterpret(
            bmm2_res_upper_d_ub,
            self.buffer_storage.dtype,
            (self.buffer_storage.subblock_query_head_count, _RESULT_HALF_COLS),
            stride=(_RESULT_HALF_COLS * _F32_BYTES // _F16_BYTES, 1),
        )
        if kv_tile_idx >= 0:
            if const_expr(self.kv_mode == _KV_MODE_LINEAR):
                indices_write = self.buffer_storage.sparse_indices_ub_ring.next_write()
                if is_cmp:
                    self.vec0_block.copy_in_sparse_indices(
                        cmp_sparse_indices_gm,
                        query_row_idx,
                        stream_kv_tile_idx,
                        valid_len,
                        const_info.cmp_sparse_index_width,
                        prior_query_tile_count,
                        kv_tile_idx,
                        indices_write,
                    )
                elif const_expr(not self.is_ori_dense):
                    self.vec0_block.copy_in_sparse_indices(
                        ori_sparse_indices_gm,
                        query_row_idx,
                        stream_kv_tile_idx,
                        valid_len,
                        const_info.ori_sparse_index_width,
                        prior_query_tile_count,
                        kv_tile_idx,
                        indices_write,
                    )
        vec0_run_info = self.run_info_delay.tap(_VEC0_PIPELINE_DELAY)
        vec1_run_info = self.run_info_delay.tap(_VEC1_PIPELINE_DELAY)
        vec2_run_info = self.run_info_delay.tap(_VEC2_PIPELINE_DELAY)
        softmax_kv_len = scalar_select(
            vec1_run_info.is_cmp,
            vec1_run_info.cmp_kv_length,
            vec1_run_info.ori_kv_length,
        )
        if pipeline_tick >= _VEC0_PIPELINE_DELAY and vec0_run_info.kv_tile_idx >= 0:
            if vec0_run_info.is_cmp:
                self.process_vec0(
                    cmp_kv_gm,
                    cmp_sparse_indices_gm,
                    cmp_block_table_gm,
                    cmp_kv_physical_rows_gm,
                    vec0_run_info.query_row_idx,
                    vec0_run_info.batch_idx,
                    vec0_run_info.cmp_kv_base_row,
                    vec0_run_info.prior_query_tile_count + vec0_run_info.kv_tile_idx,
                    0,
                    vec0_run_info.cmp_kv_length,
                    vec0_run_info.valid_kv_length,
                    vec0_run_info.stream_kv_tile_idx,
                    kv_lower_rows_l1_write,
                    kv_upper_rows_l1_write,
                    dtypes.int64(0),
                    dtypes.int64(0),
                    const_info.cmp_sparse_index_width,
                    const_info.cmp_page_size,
                    const_info.cmp_page_stride_rows,
                    const_info.cmp_block_table_width,
                    indices_read,
                )
            else:
                self.process_vec0(
                    ori_kv_gm,
                    ori_sparse_indices_gm,
                    ori_block_table_gm,
                    ori_kv_physical_rows_gm,
                    vec0_run_info.query_row_idx,
                    vec0_run_info.batch_idx,
                    vec0_run_info.ori_kv_base_row,
                    vec0_run_info.prior_query_tile_count + vec0_run_info.kv_tile_idx,
                    0,
                    vec0_run_info.ori_kv_length,
                    vec0_run_info.valid_kv_length,
                    vec0_run_info.stream_kv_tile_idx,
                    kv_lower_rows_l1_write,
                    kv_upper_rows_l1_write,
                    vec0_run_info.ori_kv_start_row,
                    const_info.is_ori_dense,
                    const_info.ori_sparse_index_width,
                    const_info.ori_page_size,
                    const_info.ori_page_stride_rows,
                    const_info.ori_block_table_width,
                    indices_read,
                )
        if pipeline_tick >= _VEC1_PIPELINE_DELAY and vec1_run_info.kv_tile_idx >= 0:
            self.process_vec1(
                softmax_lse_gm,
                const_info,
                vec1_run_info.active_query_idx,
                vec1_run_info.query_row_idx,
                vec1_run_info.kv_tile_idx,
                vec1_run_info.stream_kv_tile_idx,
                softmax_kv_len,
                vec1_run_info.kv_tile_count,
                valid_head_count,
                bmm1_res_ub_read,
                vec1_res_ub,
                bmm2_a_l1_write,
                vec1_run_info.valid_kv_length,
            )
        if pipeline_tick >= _VEC0_PIPELINE_DELAY and vec0_run_info.kv_tile_idx >= 0:
            if vec0_run_info.is_cmp:
                self.process_vec0(
                    cmp_kv_gm,
                    cmp_sparse_indices_gm,
                    cmp_block_table_gm,
                    cmp_kv_physical_rows_gm,
                    vec0_run_info.query_row_idx,
                    vec0_run_info.batch_idx,
                    vec0_run_info.cmp_kv_base_row,
                    vec0_run_info.prior_query_tile_count + vec0_run_info.kv_tile_idx,
                    1,
                    vec0_run_info.cmp_kv_length,
                    vec0_run_info.valid_kv_length,
                    vec0_run_info.stream_kv_tile_idx,
                    kv_lower_rows_l1_write,
                    kv_upper_rows_l1_write,
                    dtypes.int64(0),
                    dtypes.int64(0),
                    const_info.cmp_sparse_index_width,
                    const_info.cmp_page_size,
                    const_info.cmp_page_stride_rows,
                    const_info.cmp_block_table_width,
                    indices_read,
                )
            else:
                self.process_vec0(
                    ori_kv_gm,
                    ori_sparse_indices_gm,
                    ori_block_table_gm,
                    ori_kv_physical_rows_gm,
                    vec0_run_info.query_row_idx,
                    vec0_run_info.batch_idx,
                    vec0_run_info.ori_kv_base_row,
                    vec0_run_info.prior_query_tile_count + vec0_run_info.kv_tile_idx,
                    1,
                    vec0_run_info.ori_kv_length,
                    vec0_run_info.valid_kv_length,
                    vec0_run_info.stream_kv_tile_idx,
                    kv_lower_rows_l1_write,
                    kv_upper_rows_l1_write,
                    vec0_run_info.ori_kv_start_row,
                    const_info.is_ori_dense,
                    const_info.ori_sparse_index_width,
                    const_info.ori_page_size,
                    const_info.ori_page_stride_rows,
                    const_info.ori_block_table_width,
                    indices_read,
                )
        if pipeline_tick >= _VEC2_PIPELINE_DELAY and vec2_run_info.kv_tile_idx >= 0:
            self.process_vec2(
                attention_out_gm,
                const_info,
                vec2_run_info.active_query_idx,
                vec2_run_info.query_row_idx,
                vec2_run_info.kv_tile_idx,
                0,
                vec2_run_info.kv_tile_count,
                valid_head_count,
                attention_acc_lower_d_ub,
                bmm2_res_lower_d_ub,
                attention_out_lower_d_ub,
            )
            self.process_vec2(
                attention_out_gm,
                const_info,
                vec2_run_info.active_query_idx,
                vec2_run_info.query_row_idx,
                vec2_run_info.kv_tile_idx,
                1,
                vec2_run_info.kv_tile_count,
                valid_head_count,
                attention_acc_upper_d_ub,
                bmm2_res_upper_d_ub,
                attention_out_upper_d_ub,
            )
        if pipeline_tick >= _VEC0_PIPELINE_DELAY and vec0_run_info.kv_tile_idx >= 0:
            kv_lower_rows_l1_write = self.buffer_storage.kv_l1_rings[0].next_write()
            kv_upper_rows_l1_write = self.buffer_storage.kv_l1_rings[1].next_write()
            if const_expr(self.kv_mode == _KV_MODE_LINEAR):
                indices_read = self.buffer_storage.sparse_indices_ub_ring.next_read()
        self.run_info_delay.advance()
        return (kv_lower_rows_l1_write, kv_upper_rows_l1_write, indices_read)

    @jit
    def process_vec0(
        self,
        kv_gm,
        sparse_indices_gm,
        block_table_gm,
        kv_physical_rows_gm,
        query_row_idx,
        batch_idx,
        kv_base_row,
        global_tile,
        kv_half_idx,
        kv_len,
        valid_kv_length,
        stream_kv_tile_idx,
        kv_lower_rows_l1_write,
        kv_upper_rows_l1_write,
        ori_kv_start_row,
        is_dense,
        sparse_index_stride,
        page_size,
        page_stride_rows,
        block_table_width,
        indices_read,
    ):
        for chunk in range_constexpr(_GATHER_CHUNKS):
            if const_expr(self.kv_mode == _KV_MODE_PA):
                if const_expr(self.is_ori_dense):
                    if is_dense != 0:
                        self.vec0_block.copy_in_pa_dense_kv(
                            kv_gm,
                            block_table_gm,
                            batch_idx,
                            kv_half_idx,
                            chunk,
                            valid_kv_length,
                            stream_kv_tile_idx,
                            ori_kv_start_row,
                            page_size,
                            page_stride_rows,
                            block_table_width,
                        )
                    else:
                        self.vec0_block.copy_in_pa_sparse_kv(
                            kv_gm,
                            kv_physical_rows_gm,
                            query_row_idx,
                            kv_half_idx,
                            chunk,
                            valid_kv_length,
                            stream_kv_tile_idx,
                            (sparse_index_stride + _NZ_ROW - 1) // _NZ_ROW * _NZ_ROW,
                        )
                else:
                    self.vec0_block.copy_in_pa_sparse_kv(
                        kv_gm,
                        kv_physical_rows_gm,
                        query_row_idx,
                        kv_half_idx,
                        chunk,
                        valid_kv_length,
                        stream_kv_tile_idx,
                        (sparse_index_stride + _NZ_ROW - 1) // _NZ_ROW * _NZ_ROW,
                    )
            elif const_expr(self.is_ori_dense):
                if is_dense != 0:
                    self.vec0_block.copy_in_dense_kv(
                        kv_gm,
                        kv_base_row,
                        kv_half_idx,
                        kv_len,
                        chunk,
                        valid_kv_length,
                        stream_kv_tile_idx,
                        ori_kv_start_row,
                    )
                else:
                    self.vec0_block.copy_in_sparse_kv(
                        kv_gm,
                        kv_base_row,
                        global_tile,
                        kv_half_idx,
                        kv_len,
                        chunk,
                        valid_kv_length,
                        stream_kv_tile_idx,
                        indices_read,
                    )
            else:
                self.vec0_block.copy_in_sparse_kv(
                    kv_gm,
                    kv_base_row,
                    global_tile,
                    kv_half_idx,
                    kv_len,
                    chunk,
                    valid_kv_length,
                    stream_kv_tile_idx,
                    indices_read,
                )
        for chunk in range_constexpr(_GATHER_CHUNKS):
            self.vec0_block.convert_kv_nd_to_nz(chunk)
            source = view_padded_nz_columns(
                self.buffer_storage.kv_out_ub_bufs[chunk],
                _HEAD_DIM,
                _GATHER_TILE_ROWS,
                _NZ_PADDED_ROWS,
            )
            row_tile = kv_half_idx * _GATHER_CHUNKS + chunk
            if get_subblock_id() == 0:
                target = tile_slice(
                    kv_lower_rows_l1_write.view(
                        _HEAD_DIM // _NZ_ROW, _KV_ROWS_PER_VECTOR * _NZ_ROW
                    ),
                    (_HEAD_DIM // _NZ_ROW, _GATHER_TILE_ROWS * _NZ_ROW),
                    (0, row_tile),
                )
                mem_copy(target, source)
            else:
                target = tile_slice(
                    kv_upper_rows_l1_write.view(
                        _HEAD_DIM // _NZ_ROW, _KV_ROWS_PER_VECTOR * _NZ_ROW
                    ),
                    (_HEAD_DIM // _NZ_ROW, _GATHER_TILE_ROWS * _NZ_ROW),
                    (0, row_tile),
                )
                mem_copy(target, source)

    @jit
    def process_vector(
        self,
        ori_kv_gm,
        cmp_kv_gm,
        ori_sparse_indices_gm,
        cmp_sparse_indices_gm,
        ori_block_table_gm,
        cmp_block_table_gm,
        ori_kv_physical_rows_gm,
        cmp_kv_physical_rows_gm,
        ori_topk_length_gm,
        cmp_topk_length_gm,
        seqused_ori_kv_gm,
        seqused_cmp_kv_gm,
        cmp_residual_kv_gm,
        metadata_gm,
        seqused_q_gm,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        sinks_gm,
        attention_out_gm,
        softmax_lse_gm,
        const_info,
    ):
        calc_ori_sparse_capacity(const_info)
        head_start = get_subblock_id() * self.buffer_storage.subblock_query_head_count
        valid_head_count = scalar_select(
            const_info.num_heads_q > head_start,
            scalar_select(
                const_info.num_heads_q - head_start
                < self.buffer_storage.subblock_query_head_count,
                const_info.num_heads_q - head_start,
                self.buffer_storage.subblock_query_head_count,
            ),
            0,
        )
        if const_expr(self.has_sink):
            if valid_head_count > 0:
                mem_copy(
                    self.buffer_storage.sinks_ub[0:valid_head_count,],
                    sinks_gm[head_start: head_start + valid_head_count,],
                )  # fmt: skip
        else:
            self.vector_block.init_no_sink_buffer()
        self.vector_block.init_empty_output_buffers()
        # Zero only the per-subblock gather storage; no full-core barrier is needed.
        self.vec0_block.init_vec0_buffers()
        # Preserve the original zero-filled history without changing any stage.
        # A full turn leaves the write slot unchanged and avoids negative taps.
        for _ in range_constexpr(_PIPELINE_SLOTS):
            self.run_info_delay.push(
                active_query_idx=dtypes.int64(0),
                query_row_idx=dtypes.int64(0),
                kv_tile_idx=dtypes.int64(0),
                valid_kv_length=dtypes.int64(0),
                kv_tile_count=dtypes.int64(0),
                prior_query_tile_count=dtypes.int64(0),
                stream_kv_tile_idx=dtypes.int64(0),
                batch_idx=dtypes.int64(0),
                ori_kv_base_row=dtypes.int64(0),
                cmp_kv_base_row=dtypes.int64(0),
                ori_kv_length=dtypes.int64(0),
                cmp_kv_length=dtypes.int64(0),
                ori_kv_start_row=dtypes.int64(0),
                is_cmp=dtypes.int64(0),
            )
            self.run_info_delay.advance()
        active_query_idx = dtypes.int64(0)
        metadata_task_count = get_metadata_task_count(
            metadata_gm, seqused_q_gm, cu_seqlens_q, const_info
        )
        metadata_task_idx = dtypes.int64(0)
        query_row_idx = dtypes.int64(0)
        prior_query_tile_count = dtypes.int64(0)
        pipeline_tick = dtypes.int64(0)
        active_query_count = dtypes.int64(0)
        indices_read = self.buffer_storage.sparse_indices_ub_ring.next_read()
        kv_lower_rows_l1_write = self.buffer_storage.kv_l1_rings[0].next_write()
        kv_upper_rows_l1_write = self.buffer_storage.kv_l1_rings[1].next_write()
        # One final iteration drains pending Vector stages after this metadata range.
        for _ in range(metadata_task_count + 1):
            is_draining = metadata_task_idx >= metadata_task_count
            ori_valid_kv_length = dtypes.int64(0)
            cmp_valid_kv_length = dtypes.int64(0)
            ori_tile_count = dtypes.int64(0)
            kv_tile_count = dtypes.int64(0)
            batch_idx = dtypes.int64(0)
            ori_kv_base_row = dtypes.int64(0)
            cmp_kv_base_row = dtypes.int64(0)
            ori_kv_length = dtypes.int64(0)
            cmp_kv_length = dtypes.int64(0)
            ori_kv_start_row = dtypes.int64(0)
            # Empty rows do not advance the delay line.  If this core processed
            # any nonempty row, a fixed five ticks are therefore required to flush
            # Gather, Softmax, and Update after all assigned rows have been visited.
            steps = scalar_select(
                active_query_count > 0, dtypes.int64(_PIPELINE_SLOTS), dtypes.int64(0)
            )
            is_empty_query = dtypes.int64(0)
            if not is_draining:
                (
                    query_row_idx,
                    batch_idx,
                    query_seq_idx,
                    valid_query_count,
                    _context_found,
                ) = get_metadata_task_context(
                    metadata_gm,
                    seqused_q_gm,
                    cu_seqlens_q,
                    const_info,
                    metadata_task_idx,
                )
                if _context_found != 0 and query_seq_idx < valid_query_count:
                    (
                        ori_kv_base_row,
                        cmp_kv_base_row,
                        ori_kv_length,
                        cmp_kv_length,
                        ori_kv_start_row,
                        ori_valid_kv_length,
                        cmp_valid_kv_length,
                        ori_tile_count,
                        kv_tile_count,
                    ) = calc_sparse_query_kv_lengths(
                        ori_topk_length_gm,
                        cmp_topk_length_gm,
                        seqused_ori_kv_gm,
                        seqused_cmp_kv_gm,
                        cmp_residual_kv_gm,
                        cu_seqlens_ori_kv,
                        cu_seqlens_cmp_kv,
                        const_info,
                        query_row_idx,
                        batch_idx,
                        query_seq_idx,
                        valid_query_count,
                    )
                is_empty_query = scalar_select(
                    ori_valid_kv_length == 0,
                    scalar_select(
                        cmp_valid_kv_length == 0, dtypes.int64(1), dtypes.int64(0)
                    ),
                    dtypes.int64(0),
                )
                if ori_valid_kv_length == 0 and cmp_valid_kv_length == 0:
                    self.vector_block.copy_out_empty_query(
                        attention_out_gm,
                        softmax_lse_gm,
                        const_info,
                        query_row_idx,
                        head_start,
                        valid_head_count,
                    )
                steps = kv_tile_count
            for step in range(steps):
                kv_tile_idx = scalar_select(is_draining, -1, step)
                kv_lower_rows_l1_write, kv_upper_rows_l1_write, indices_read = (
                    self.process_vector_step(
                        ori_kv_gm,
                        cmp_kv_gm,
                        ori_sparse_indices_gm,
                        cmp_sparse_indices_gm,
                        ori_block_table_gm,
                        cmp_block_table_gm,
                        ori_kv_physical_rows_gm,
                        cmp_kv_physical_rows_gm,
                        attention_out_gm,
                        softmax_lse_gm,
                        const_info,
                        pipeline_tick,
                        active_query_idx,
                        query_row_idx,
                        kv_tile_idx,
                        ori_valid_kv_length,
                        cmp_valid_kv_length,
                        ori_tile_count,
                        kv_tile_count,
                        prior_query_tile_count,
                        batch_idx,
                        ori_kv_base_row,
                        cmp_kv_base_row,
                        ori_kv_length,
                        cmp_kv_length,
                        ori_kv_start_row,
                        valid_head_count,
                        kv_lower_rows_l1_write,
                        kv_upper_rows_l1_write,
                        indices_read,
                    )
                )
                pipeline_tick = pipeline_tick + 1
            if not is_draining:
                prior_query_tile_count = prior_query_tile_count + kv_tile_count
                active_query_id = scalar_select(
                    is_empty_query == 0, dtypes.int64(1), dtypes.int64(0)
                )
                active_query_idx = active_query_idx + active_query_id
                active_query_count = active_query_count + active_query_id
            metadata_task_idx = metadata_task_idx + 1

    @jit
    def process_bmm1(
        self, active_query_idx, kv_lower_rows_l1, kv_upper_rows_l1, query_heads_l1_wide
    ):
        self.cube_block.iterate_bmm1(
            active_query_idx, kv_lower_rows_l1, kv_upper_rows_l1, query_heads_l1_wide
        )
        self.cube_block.copy_bmm1_res_l0c_to_ub(
            self.buffer_storage.get_next_bmm1_res_ub(True)
        )

    @jit
    def process_bmm2(self, kv_tile_idx, kv_lower_rows_l1, kv_upper_rows_l1):
        bmm2_a_l1 = self.buffer_storage.cross_core_rings["bmm2_a_l1_ring"].next_read()
        (
            attention_acc_lower_d_ub,
            bmm2_res_lower_d_ub,
            attention_acc_upper_d_ub,
            bmm2_res_upper_d_ub,
        ) = self.buffer_storage.get_next_bmm2_res_ub_buffers(True)
        self.cube_block.iterate_bmm2(
            kv_tile_idx,
            kv_lower_rows_l1,
            kv_upper_rows_l1,
            bmm2_a_l1,
            attention_acc_lower_d_ub,
            bmm2_res_lower_d_ub,
            attention_acc_upper_d_ub,
            bmm2_res_upper_d_ub,
        )

    @jit
    def process_cube(
        self,
        query_gm,
        ori_topk_length_gm,
        cmp_topk_length_gm,
        seqused_ori_kv_gm,
        seqused_cmp_kv_gm,
        cmp_residual_kv_gm,
        metadata_gm,
        seqused_q_gm,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        const_info,
    ):
        metadata_task_count = get_metadata_task_count(
            metadata_gm, seqused_q_gm, cu_seqlens_q, const_info
        )
        metadata_task_idx = dtypes.int64(0)
        active_query_idx = dtypes.int64(0)
        (
            metadata_task_idx,
            query_row_idx,
            _batch,
            _ori_base,
            _cmp_base,
            _ori_kv_len,
            _cmp_kv_len,
            _ori_begin,
            ori_valid_kv_length,
            cmp_valid_kv_length,
            ori_tile_count,
            kv_tile_count,
            found,
        ) = get_next_nonempty_metadata_task(
            ori_topk_length_gm,
            cmp_topk_length_gm,
            seqused_ori_kv_gm,
            seqused_cmp_kv_gm,
            cmp_residual_kv_gm,
            metadata_gm,
            seqused_q_gm,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
            const_info,
            metadata_task_idx,
            metadata_task_count,
        )
        if found != 0:
            current_q = self.cube_block.copy_q_gm_to_l1(
                query_gm, query_row_idx, active_query_idx, const_info.num_heads_q, False
            )
            self.cube_block.copy_q_gm_to_l1(
                query_gm, query_row_idx, active_query_idx, const_info.num_heads_q, True
            )
            # The two L1 buffers hold the lower and upper KV-row halves of this tile.
            kv_lower_rows_l1, kv_upper_rows_l1 = (
                self.buffer_storage.get_next_kv_l1_buffers(False)
            )
            self.process_bmm1(
                active_query_idx, kv_lower_rows_l1, kv_upper_rows_l1, current_q
            )
            has_current_query = dtypes.int64(1)
            while has_current_query != 0:
                next_q = current_q
                next_task_idx = metadata_task_idx + 1
                next_kv_tile_count = dtypes.int64(0)
                (
                    next_task_idx,
                    next_query_row_idx,
                    _next_batch,
                    _next_ori_base,
                    _next_cmp_base,
                    _next_ori_kv_len,
                    _next_cmp_kv_len,
                    _next_ori_begin,
                    _next_ori_length,
                    _next_cmp_length,
                    _next_ori_tile_count,
                    next_kv_tile_count,
                    next_found,
                ) = get_next_nonempty_metadata_task(
                    ori_topk_length_gm,
                    cmp_topk_length_gm,
                    seqused_ori_kv_gm,
                    seqused_cmp_kv_gm,
                    cmp_residual_kv_gm,
                    metadata_gm,
                    seqused_q_gm,
                    cu_seqlens_q,
                    cu_seqlens_ori_kv,
                    cu_seqlens_cmp_kv,
                    const_info,
                    next_task_idx,
                    metadata_task_count,
                )
                if next_found != 0:
                    # Prefetch the next query's first three D tiles while this query runs.
                    next_q = self.cube_block.copy_q_gm_to_l1(
                        query_gm,
                        next_query_row_idx,
                        active_query_idx + 1,
                        const_info.num_heads_q,
                        False,
                    )
                for kv_tile_idx in range(kv_tile_count - 1):
                    previous_kv_lower_rows_l1, previous_kv_upper_rows_l1 = (
                        kv_lower_rows_l1,
                        kv_upper_rows_l1,
                    )
                    kv_lower_rows_l1, kv_upper_rows_l1 = (
                        self.buffer_storage.get_next_kv_l1_buffers(False)
                    )
                    self.process_bmm1(
                        active_query_idx, kv_lower_rows_l1, kv_upper_rows_l1, current_q
                    )
                    self.process_bmm2(
                        kv_tile_idx,
                        previous_kv_lower_rows_l1,
                        previous_kv_upper_rows_l1,
                    )
                previous_kv_lower_rows_l1, previous_kv_upper_rows_l1 = (
                    kv_lower_rows_l1,
                    kv_upper_rows_l1,
                )
                if next_found != 0:
                    self.cube_block.copy_q_gm_to_l1(
                        query_gm,
                        next_query_row_idx,
                        active_query_idx + 1,
                        const_info.num_heads_q,
                        True,
                    )
                    kv_lower_rows_l1, kv_upper_rows_l1 = (
                        self.buffer_storage.get_next_kv_l1_buffers(False)
                    )
                    self.process_bmm1(
                        active_query_idx + 1, kv_lower_rows_l1, kv_upper_rows_l1, next_q
                    )
                self.process_bmm2(
                    kv_tile_count - 1,
                    previous_kv_lower_rows_l1,
                    previous_kv_upper_rows_l1,
                )
                kv_tile_count = next_kv_tile_count
                query_row_idx = next_query_row_idx
                metadata_task_idx = next_task_idx
                current_q = next_q
                active_query_idx = active_query_idx + 1
                has_current_query = next_found


class HighPerfSparseFlashMlaLauncher:
    def __init__(
        self,
        query_head_tile_size,
        has_sink,
        return_softmax_lse,
        dtype,
        batch_mode,
        kv_mode,
        is_ori_dense,
    ):
        self.query_head_tile_size, self.has_sink = (query_head_tile_size, has_sink)
        self.return_softmax_lse = return_softmax_lse
        self.dtype = dtype
        self.batch_mode = batch_mode
        self.kv_mode = kv_mode
        self.is_ori_dense = is_ori_dense

    @host
    def run(
        self,
        query_gm: Tensor,
        ori_kv_gm: Tensor,
        cmp_kv_gm: Tensor,
        ori_sparse_indices_gm: Tensor,
        cmp_sparse_indices_gm: Tensor,
        ori_block_table_gm: Tensor,
        cmp_block_table_gm: Tensor,
        ori_kv_physical_rows_gm: Tensor,
        cmp_kv_physical_rows_gm: Tensor,
        ori_topk_length_gm: Tensor,
        cmp_topk_length_gm: Tensor,
        seqused_ori_kv_gm: Tensor,
        seqused_cmp_kv_gm: Tensor,
        cmp_residual_kv_gm: Tensor,
        metadata_gm: Tensor,
        seqused_q_gm: Tensor,
        cu_seqlens_q: Tensor,
        cu_seqlens_ori_kv: Tensor,
        cu_seqlens_cmp_kv: Tensor,
        sinks_gm: Tensor,
        attention_out_gm: Tensor,
        softmax_lse_gm: Tensor,
        const_info: HighPerfSmlaConstInfo,
        blocks: dtypes.int32,
    ):
        SparseN64Kernel(
            self.query_head_tile_size,
            self.has_sink,
            self.return_softmax_lse,
            self.dtype,
            self.batch_mode,
            self.kv_mode,
            self.is_ori_dense,
        )[blocks](
            query_gm,
            ori_kv_gm,
            cmp_kv_gm,
            ori_sparse_indices_gm,
            cmp_sparse_indices_gm,
            ori_block_table_gm,
            cmp_block_table_gm,
            ori_kv_physical_rows_gm,
            cmp_kv_physical_rows_gm,
            ori_topk_length_gm,
            cmp_topk_length_gm,
            seqused_ori_kv_gm,
            seqused_cmp_kv_gm,
            cmp_residual_kv_gm,
            metadata_gm,
            seqused_q_gm,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
            sinks_gm,
            attention_out_gm,
            softmax_lse_gm,
            const_info,
        )


def _build_compile_identity(
    query_head_tile_size,
    has_sink,
    return_softmax_lse,
    dtype_label,
    batch_mode,
    kv_mode,
    is_ori_dense,
):
    compiler = (
        os.environ.get("CANNBOTDSL_BISHENG_PATH")
        or os.environ.get("CANNBOTDSL_BISHENG")
        or shutil.which("bisheng")
    )
    if not compiler:
        raise RuntimeError("Source the CANN environment before compiling or loading")
    compiler = Path(compiler).resolve()
    compiler_stat = compiler.stat()
    distributions = {}
    for name in ("cannbot-dsl", "opkit"):
        distribution = importlib.metadata.distribution(name)
        record = distribution.read_text("RECORD") or ""
        distributions[name] = {
            "version": distribution.version,
            "record_sha256": hashlib.sha256(record.encode()).hexdigest(),
        }
    return {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "query_tile_rows": query_head_tile_size,
        "has_sink": bool(has_sink),
        "return_softmax_lse": bool(return_softmax_lse),
        "dtype": dtype_label,
        "batch_mode": batch_mode,
        "kv_mode": kv_mode,
        "ori_dense": bool(is_ori_dense),
        "compiler_flags": list(_COMPILER_FLAGS),
        "architecture": "dav-3510",
        "compiler": str(compiler),
        "compiler_size": compiler_stat.st_size,
        "compiler_mtime_ns": compiler_stat.st_mtime_ns,
        "cann_home": os.environ.get("ASCEND_HOME_PATH"),
        "distributions": distributions,
    }


def _sparse_n64_tensor_specs(storage_dtype):
    tensor_types = (
        ("q_numel", storage_dtype),
        ("ori_kv_numel", storage_dtype),
        ("cmp_numel", storage_dtype),
        ("ori_index_numel", dtypes.int32),
        ("cmp_index_numel", dtypes.int32),
        ("ori_block_table_numel", dtypes.int32),
        ("cmp_block_table_numel", dtypes.int32),
        ("ori_physical_numel", dtypes.int32),
        ("cmp_physical_numel", dtypes.int32),
        ("ori_length_numel", dtypes.int32),
        ("cmp_length_numel", dtypes.int32),
        ("ori_seqused_numel", dtypes.int32),
        ("cmp_seqused_numel", dtypes.int32),
        ("residual_numel", dtypes.int32),
        ("metadata_numel", dtypes.int32),
        ("q_length_numel", dtypes.int32),
        ("cu_q_numel", dtypes.int32),
        ("cu_ori_kv_numel", dtypes.int32),
        ("cu_cmp_kv_numel", dtypes.int32),
        ("sink_numel", dtypes.float32),
        ("out_numel", storage_dtype),
        ("lse_numel", dtypes.float32),
    )
    specs = [
        TensorSpec((Dim(name, min=0 if name == "lse_numel" else 1),), dtype)
        for name, dtype in tensor_types
    ]
    return specs


@functools.lru_cache(maxsize=_COMPILED_VARIANT_CACHE_SIZE)
def _get_compiled_sparse_n64(
    query_head_tile_size,
    has_sink,
    return_softmax_lse=False,
    dtype="bf16",
    batch_mode="single",
    kv_mode="linear",
    is_ori_dense=False,
):
    try:
        batch_mode_value = _BATCH_MODE_BY_LABEL[batch_mode]
    except KeyError as error:
        raise ValueError(f"Unsupported batch mode: {batch_mode}") from error
    try:
        storage_dtype = _DSL_DTYPE_BY_LABEL[dtype]
    except KeyError as error:
        raise ValueError(f"Unsupported storage dtype: {dtype}") from error
    try:
        kv_mode_value = _KV_MODE_BY_LABEL[kv_mode]
    except KeyError as error:
        raise ValueError(f"Unsupported KV mode: {kv_mode}") from error
    specs = _sparse_n64_tensor_specs(storage_dtype)

    def compile_kernel():
        return cannbotdsl.compile(
            HighPerfSparseFlashMlaLauncher(
                query_head_tile_size,
                has_sink,
                return_softmax_lse,
                storage_dtype,
                batch_mode_value,
                kv_mode_value,
                is_ori_dense,
            ).run,
            *specs,
            HighPerfSmlaConstInfo,
            dtypes.int32,
        )

    return _load_or_compile_kernel(
        query_head_tile_size,
        has_sink,
        return_softmax_lse,
        dtype,
        batch_mode,
        kv_mode,
        is_ori_dense,
        compile_kernel,
    )


def _load_cached_n64_artifact(artifact, manifest, identity):
    if not artifact.is_file() or not manifest.is_file():
        return None
    try:
        if json.loads(manifest.read_text()) == identity:
            return load(str(artifact))
    except (OSError, ValueError):
        pass  # Rebuild invalid entries while the caller still owns the lock.
    return None


def _publish_n64_manifest(build_dir, manifest, identity):
    with tempfile.NamedTemporaryFile(
        mode="w", dir=build_dir, suffix=".json", delete=False
    ) as output:
        json.dump(identity, output, indent=2)
        temporary_manifest = Path(output.name)
    temporary_manifest.replace(manifest)


def _load_or_compile_kernel(
    query_head_tile_size,
    has_sink,
    return_softmax_lse,
    dtype,
    batch_mode,
    kv_mode,
    is_ori_dense,
    compile_kernel,
):
    identity = _build_compile_identity(
        query_head_tile_size,
        has_sink,
        return_softmax_lse,
        dtype,
        batch_mode,
        kv_mode,
        is_ori_dense,
    )
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:_BUILD_HASH_PREFIX_LENGTH]
    build_root = Path(__file__).parent
    build_dir = (
        build_root
        / (
            f"build_h{query_head_tile_size}_sink{int(has_sink)}_lse{int(return_softmax_lse)}"
            f"_dtype{dtype}_batch{batch_mode}"
            f"_kv{kv_mode}_csa{int(is_ori_dense)}"
        )
        / fingerprint
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    artifact = build_dir / "kernel.so"
    manifest = build_dir / "build_identity.json"
    # Serialize publication across processes and keep the .so/sidecar pair intact.
    with _COMPILATION_LOCK, (build_dir / "compile.lock").open("a") as compile_lock:
        fcntl.flock(compile_lock, fcntl.LOCK_EX)
        result = _load_cached_n64_artifact(artifact, manifest, identity)
        if result is not None:
            return result
        with _compilation_environment(build_dir):
            result = compile_kernel()
            result.save(str(artifact))
        _publish_n64_manifest(build_dir, manifest, identity)
        return result


def _require(condition, message):
    if not condition:
        raise ValueError(f"SMLA high-performance: {message}")


def _validate_tensor(tensor, dtype, device, name):
    _require(isinstance(tensor, torch.Tensor), f"{name} must be a Tensor")
    _require(
        tensor.dtype == dtype and tensor.device == device,
        f"{name} has an invalid dtype or device",
    )
    _require(tensor.is_contiguous(), f"{name} must be contiguous")


def _validate_scale(sm_scale):
    _require(
        isinstance(sm_scale, numbers.Real)
        and not isinstance(sm_scale, bool)
        and math.isfinite(sm_scale),
        "Scale must be a finite real number",
    )


def _validate_seqused_kv(tensor, batch_count, query, name):
    _validate_tensor(tensor, torch.int32, query.device, name)
    _require(
        tuple(tensor.shape) == (batch_count,),
        f"{name} must be INT32 [{batch_count}]",
    )


def _pa_division_params(page_size):
    if page_size <= 0:
        return 0, 0, 0
    is_power_of_two = page_size & (page_size - 1) == 0
    return (
        0 if page_size == 1 else ((1 << 32) + page_size - 1) // page_size,
        page_size.bit_length() - 1 if is_power_of_two else 0,
        int(is_power_of_two),
    )


@dataclass(frozen=True)
class SparseN64Call:
    query_gm: torch.Tensor
    ori_kv: torch.Tensor
    cmp_kv: object = None
    ori_sparse_indices: object = None
    cmp_sparse_indices: object = None
    ori_block_table: object = None
    cmp_block_table: object = None
    cu_seqlens_q: object = None
    cu_seqlens_ori_kv: object = None
    cu_seqlens_cmp_kv: object = None
    seqused_q: object = None
    seqused_ori_kv: object = None
    seqused_cmp_kv: object = None
    cmp_residual_kv: object = None
    ori_topk_length: object = None
    cmp_topk_length: object = None
    sinks: object = None
    metadata: object = None
    softmax_scale: float = 1.0
    cmp_ratio: int = 1
    ori_mask_mode: int = 0
    cmp_mask_mode: int = 0
    ori_win_left: int = -1
    ori_win_right: int = -1
    layout_q: str = "BSND"
    layout_kv: str = "BSND"
    topk_value_mode: int = 1
    return_softmax_lse: bool = False

    @property
    def has_cmp(self):
        return self.cmp_kv is not None

    @property
    def is_ori_dense(self):
        return self.ori_sparse_indices is None

    @property
    def pa_mode(self):
        return self.layout_kv == "PA_BBND"


@dataclass(frozen=True)
class SparseN64Layout:
    rank: int
    batch_count: int
    query_rows_per_batch: int
    batch_mode: str


@dataclass(frozen=True)
class SparseN64Cache:
    tensor: object
    rows: int
    page_size: int = 0
    stride_rows: int = 0
    table_width: int = 0


@dataclass(frozen=True)
class SparseN64Storage:
    ori: SparseN64Cache
    cmp: SparseN64Cache


@dataclass(frozen=True)
class SparseN64Prepared:
    query: torch.Tensor
    ori_indices: torch.Tensor
    cmp_indices: object
    ori_topk_length: object
    cmp_topk_length: object
    lse_shape: tuple
    cube_core_count: int


@dataclass(frozen=True)
class SparseN64Lengths:
    ori_topk_length_gm: torch.Tensor
    cmp_topk_length_gm: torch.Tensor
    seqused_ori_kv_gm: torch.Tensor
    seqused_cmp_kv_gm: torch.Tensor


@dataclass(frozen=True)
class SparseN64Scratch:
    ori_table: torch.Tensor
    cmp_table: torch.Tensor
    ori_kv_physical_rows_gm: torch.Tensor
    cmp_kv_physical_rows_gm: torch.Tensor
    cmp_residual_kv_gm: torch.Tensor
    metadata_tensor: torch.Tensor
    seqused_q_gm: torch.Tensor
    cu_q: torch.Tensor
    cu_ori_kv: torch.Tensor
    cu_cmp_kv: torch.Tensor
    sinks_gm: torch.Tensor


def _validate_n64_query(call):
    _validate_scale(call.softmax_scale)
    _require(
        call.layout_q in ("BSND", "TND")
        and (call.pa_mode or call.layout_kv == call.layout_q),
        "Query must be BSND/TND and KV must match or use PA_BBND",
    )
    rank = _BSND_RANK if call.layout_q == "BSND" else _TND_RANK
    _require(
        call.query_gm.ndim == rank and call.query_gm.shape[-1] == _HEAD_DIM,
        "Query rank or D is invalid",
    )
    _require(
        1 <= call.query_gm.shape[-2] <= _QUERY_HEAD_TILE_SIZE,
        "SparseN64Kernel requires 1..64 query heads",
    )
    _validate_tensor(call.query_gm, call.query_gm.dtype, call.query_gm.device, "Query")
    _require(
        call.query_gm.dtype in (torch.float16, torch.bfloat16), "FP16/BF16 required"
    )
    _require(
        call.metadata is not None
        and call.metadata.dtype == torch.int32
        and call.metadata.numel() == 1024
        and call.metadata.is_contiguous(),
        "INT32[1024] metadata is required",
    )
    return rank


def _validate_n64_modes(call):
    if call.sinks is not None:
        _validate_tensor(call.sinks, torch.float32, call.query_gm.device, "Sinks")
        _require(
            tuple(call.sinks.shape) == (call.query_gm.shape[-2],),
            "Sinks must be FP32[N1] when provided",
        )
    _require(
        call.ori_mask_mode in (0, _MASK_CAUSAL, _MASK_SLIDING_WINDOW)
        and call.cmp_mask_mode in (0, _MASK_CAUSAL),
        "Unsupported mask mode",
    )
    _require(call.ori_win_left >= -1 and call.ori_win_right >= -1, "Invalid ORI window")
    _require(call.topk_value_mode == 1, "Topk value mode 1 is required")
    _require(1 <= call.cmp_ratio <= _BASE_TILE_SIZE, "Invalid CMP ratio")
    _require(
        not call.is_ori_dense
        or (call.cmp_kv is not None and call.cmp_sparse_indices is not None),
        "Dense ORI is supported only for CSA",
    )
    _require(
        call.is_ori_dense or call.cmp_kv is None or call.cmp_sparse_indices is not None,
        "ORI sparse with dense CMP is unsupported",
    )


def _validate_n64_sparse_selection(call, rank):
    if not call.is_ori_dense:
        _validate_sparse_indices(
            call.query_gm,
            call.ori_sparse_indices,
            (
                (call.query_gm.numel() // (call.query_gm.shape[-2] * _HEAD_DIM),),
                (call.query_gm.numel() // (call.query_gm.shape[-2] * _HEAD_DIM), 1),
            )
            if rank == _TND_RANK
            else (
                (call.query_gm.shape[0], call.query_gm.shape[1]),
                (call.query_gm.shape[0], call.query_gm.shape[1], 1),
            ),
            "ORI sparse indices",
        )
        _require(
            call.ori_mask_mode != 0 or call.ori_topk_length is not None,
            "ORI mask0 requires ori_topk_length",
        )
    if call.cmp_kv is not None:
        _require(call.cmp_sparse_indices is not None, "CMP sparse indices are required")
        _require(
            call.cmp_mask_mode != 0 or call.cmp_topk_length is not None,
            "CMP mask0 requires cmp_topk_length",
        )
        _require(
            call.cmp_mask_mode != _MASK_CAUSAL
            or call.cmp_ratio == 1
            or call.cmp_residual_kv is not None,
            "Compressed causal residual is required",
        )
    else:
        _require(
            call.cmp_ratio == 1 and call.cmp_mask_mode == 0,
            "ORI-only requires CMP ratio1/mask0",
        )


def _validate_n64_layout(call, rank):
    if rank == _BSND_RANK:
        batch_count = call.query_gm.shape[0]
        query_rows_per_batch = call.query_gm.shape[1]
        batch_mode = "single" if batch_count == 1 else "bsnd"
    else:
        _require(call.cu_seqlens_q is not None, "TND requires Q prefixes")
        batch_count = call.cu_seqlens_q.numel() - 1
        query_rows_per_batch = call.query_gm.shape[0]
        batch_mode = "single" if batch_count == 1 else "tnd"
        if not call.pa_mode:
            _require(
                call.cu_seqlens_ori_kv is not None,
                "Linear TND requires ORI KV prefixes",
            )
        if call.cmp_kv is not None and not call.pa_mode:
            _require(call.cu_seqlens_cmp_kv is not None, "TND CMP prefix is required")
    return SparseN64Layout(rank, batch_count, query_rows_per_batch, batch_mode)


def _validate_n64_lengths(call, layout):
    for tensor, name in (
        (call.seqused_q, "seqused_q"),
        (call.seqused_ori_kv, "seqused_ori_kv"),
        (call.seqused_cmp_kv, "seqused_cmp_kv"),
        (call.cmp_residual_kv, "cmp_residual_kv"),
    ):
        if tensor is not None:
            _validate_tensor(tensor, torch.int32, call.query_gm.device, name)
            _require(
                tuple(tensor.shape) == (layout.batch_count,), f"{name} must be [B]"
            )
    if call.cmp_kv is not None:
        _require(
            (call.seqused_ori_kv is None) == (call.seqused_cmp_kv is None),
            "ORI/CMP KV seqused must be provided together or both omitted",
        )


def _validate_n64_pa_widths(call):
    if not call.is_ori_dense:
        _require(
            (call.ori_sparse_indices.shape[-1] + _NZ_ROW - 1) // _NZ_ROW * _NZ_ROW
            <= _PA_INDEX_CAPACITY,
            "ORI sparse width exceeds PA prepass capacity",
        )
    if call.cmp_kv is not None:
        _require(
            (call.cmp_sparse_indices.shape[-1] + _NZ_ROW - 1) // _NZ_ROW * _NZ_ROW
            <= _PA_INDEX_CAPACITY,
            "CMP sparse width exceeds PA prepass capacity",
        )


def _prepare_n64_pa_cache(query, kv, table, seqused, batch_count, name):
    tensor, page_size, stride, width = _prepare_pa_kv(
        query, kv, table, batch_count, seqused, name
    )
    return SparseN64Cache(tensor, page_size * width, page_size, stride, width)


def _prepare_n64_linear_cache(query, kv, layout, name):
    _validate_sparse_kv(query, kv, layout.rank, layout.batch_count, name)
    rows = kv.shape[1] if layout.rank == _BSND_RANK else kv.shape[0]
    return SparseN64Cache(kv.reshape(-1, 1, _HEAD_DIM), rows)


def _prepare_n64_storage(call, layout):
    if call.pa_mode:
        _validate_n64_pa_widths(call)
        ori = _prepare_n64_pa_cache(
            call.query_gm,
            call.ori_kv,
            call.ori_block_table,
            call.seqused_ori_kv,
            layout.batch_count,
            "ORI KV",
        )
        if call.has_cmp:
            cmp = _prepare_n64_pa_cache(
                call.query_gm,
                call.cmp_kv,
                call.cmp_block_table,
                call.seqused_cmp_kv,
                layout.batch_count,
                "CMP KV",
            )
        else:
            _require(call.cmp_block_table is None, "CMP block table requires CMP KV")
            cmp = SparseN64Cache(None, 0)
    else:
        _require(
            call.ori_block_table is None and call.cmp_block_table is None,
            "Block tables require PA KV layout",
        )
        ori = _prepare_n64_linear_cache(call.query_gm, call.ori_kv, layout, "ORI KV")
        cmp = (
            _prepare_n64_linear_cache(call.query_gm, call.cmp_kv, layout, "CMP KV")
            if call.has_cmp
            else SparseN64Cache(None, 0)
        )
    return SparseN64Storage(ori, cmp)


def _reshape_optional_length(tensor):
    return None if tensor is None else tensor.reshape(-1)


def _prepare_n64_call_descriptors(call, layout):
    query_row_count = call.query_gm.numel() // (call.query_gm.shape[-2] * _HEAD_DIM)
    ori_k = 1 if call.is_ori_dense else call.ori_sparse_indices.shape[-1]
    ori_sparse_indices_gm = (
        torch.empty(
            (query_row_count, 1, 1), dtype=torch.int32, device=call.query_gm.device
        )
        if call.is_ori_dense
        else call.ori_sparse_indices.reshape(query_row_count, 1, ori_k)
    )
    cmp_sparse_indices_gm = (
        None
        if call.cmp_sparse_indices is None
        else call.cmp_sparse_indices.reshape(
            query_row_count, 1, call.cmp_sparse_indices.shape[-1]
        )
    )
    lse_shape = (
        (1, query_row_count, call.query_gm.shape[-2])
        if call.layout_q == "TND"
        else (
            layout.batch_count,
            1,
            layout.query_rows_per_batch,
            call.query_gm.shape[-2],
        )
    )
    cube_core_count = torch.npu.get_device_properties(
        call.query_gm.device
    ).cube_core_num
    return SparseN64Prepared(
        query=call.query_gm.reshape(
            query_row_count, call.query_gm.shape[-2], _HEAD_DIM
        ),
        ori_indices=ori_sparse_indices_gm,
        cmp_indices=cmp_sparse_indices_gm,
        ori_topk_length=_reshape_optional_length(call.ori_topk_length),
        cmp_topk_length=_reshape_optional_length(call.cmp_topk_length),
        lse_shape=lse_shape if call.return_softmax_lse else (0,),
        cube_core_count=cube_core_count,
    )


def _prepare_n64_length_descriptors(call, prepared):
    ori_topk_length_gm = (
        prepared.ori_topk_length
        if prepared.ori_topk_length is not None
        else torch.empty(1, dtype=torch.int32, device=prepared.query.device)
    )
    cmp_topk_length_gm = (
        prepared.cmp_topk_length
        if prepared.cmp_topk_length is not None
        else torch.empty(1, dtype=torch.int32, device=prepared.query.device)
    )
    seqused_ori_kv_gm = (
        call.seqused_ori_kv
        if call.seqused_ori_kv is not None
        else torch.empty(1, dtype=torch.int32, device=prepared.query.device)
    )
    seqused_cmp_kv_gm = (
        call.seqused_cmp_kv
        if call.seqused_cmp_kv is not None
        else torch.empty(1, dtype=torch.int32, device=prepared.query.device)
    )
    return SparseN64Lengths(
        ori_topk_length_gm, cmp_topk_length_gm, seqused_ori_kv_gm, seqused_cmp_kv_gm
    )


def _prepare_n64_metadata_descriptor(call, layout, prepared):
    metadata_tensor = (
        call.metadata
        if call.metadata is not None
        else torch.zeros(1024, dtype=torch.int32, device=prepared.query.device)
    )
    if call.metadata is None:
        metadata_tensor[0] = 1
        metadata_tensor[4] = layout.batch_count - 1
        metadata_tensor[5] = (
            prepared.query.shape[0]
            if layout.batch_count == 1
            else layout.query_rows_per_batch
        )
    return metadata_tensor


def _prepare_n64_physical_rows(call, prepared, empty_int32):
    ori_addr_stride = (prepared.ori_indices.shape[2] + _NZ_ROW - 1) // _NZ_ROW * _NZ_ROW
    cmp_addr_stride = (
        (prepared.cmp_indices.shape[2] + _NZ_ROW - 1) // _NZ_ROW * _NZ_ROW
        if call.has_cmp
        else 1
    )
    ori_kv_physical_rows_gm = (
        torch.empty(
            prepared.query.shape[0] * ori_addr_stride,
            dtype=torch.int32,
            device=prepared.query.device,
        )
        if ("pa" if call.pa_mode else "linear") == "pa" and not call.is_ori_dense
        else empty_int32
    )
    cmp_kv_physical_rows_gm = (
        torch.empty(
            prepared.query.shape[0] * cmp_addr_stride,
            dtype=torch.int32,
            device=prepared.query.device,
        )
        if ("pa" if call.pa_mode else "linear") == "pa" and call.has_cmp
        else empty_int32
    )
    return ori_kv_physical_rows_gm, cmp_kv_physical_rows_gm


def _prepare_n64_scratch(call, layout, prepared):
    empty_int32 = torch.empty(1, dtype=torch.int32, device=prepared.query.device)
    ori_table = (
        call.ori_block_table if call.ori_block_table is not None else empty_int32
    )
    cmp_table = (
        (call.cmp_block_table if call.has_cmp else call.ori_block_table)
        if (call.cmp_block_table if call.has_cmp else call.ori_block_table) is not None
        else empty_int32
    )
    ori_kv_physical_rows_gm, cmp_kv_physical_rows_gm = _prepare_n64_physical_rows(
        call, prepared, empty_int32
    )
    cmp_residual_kv_gm = (
        call.cmp_residual_kv if call.cmp_residual_kv is not None else empty_int32
    )
    metadata_tensor = _prepare_n64_metadata_descriptor(call, layout, prepared)
    seqused_q_gm = call.seqused_q if call.seqused_q is not None else empty_int32
    cu_q = call.cu_seqlens_q if call.cu_seqlens_q is not None else empty_int32
    cu_ori_kv = (
        call.cu_seqlens_ori_kv if call.cu_seqlens_ori_kv is not None else empty_int32
    )
    cu_cmp_kv = (
        call.cu_seqlens_cmp_kv if call.cu_seqlens_cmp_kv is not None else empty_int32
    )
    sinks_gm = (
        call.sinks
        if call.sinks is not None
        else torch.empty(1, dtype=torch.float32, device=prepared.query.device)
    )
    return SparseN64Scratch(
        ori_table=ori_table,
        cmp_table=cmp_table,
        ori_kv_physical_rows_gm=ori_kv_physical_rows_gm,
        cmp_kv_physical_rows_gm=cmp_kv_physical_rows_gm,
        cmp_residual_kv_gm=cmp_residual_kv_gm,
        metadata_tensor=metadata_tensor,
        seqused_q_gm=seqused_q_gm,
        cu_q=cu_q,
        cu_ori_kv=cu_ori_kv,
        cu_cmp_kv=cu_cmp_kv,
        sinks_gm=sinks_gm,
    )


def _build_n64_runtime_constants(call, layout, storage, prepared):
    ori_reciprocal, ori_shift, ori_pow2 = _pa_division_params(storage.ori.page_size)
    cmp_reciprocal, cmp_shift, cmp_pow2 = _pa_division_params(storage.cmp.page_size)
    const_info = HighPerfSmlaConstInfo(
        query_row_count=prepared.query.shape[0],
        ori_kv_capacity=storage.ori.rows,
        cmp_kv_capacity=storage.cmp.rows if call.has_cmp else 0,
        cmp_sparse_index_width=prepared.cmp_indices.shape[2] if call.has_cmp else 0,
        has_cmp_topk_length=int(prepared.cmp_topk_length is not None),
        num_heads_q=prepared.query.shape[1],
        ori_sparse_index_width=prepared.ori_indices.shape[2],
        ori_win_left=call.ori_win_left,
        ori_win_right=call.ori_win_right,
        has_ori_topk_length=int(prepared.ori_topk_length is not None),
        has_seqused_ori_kv=int(call.seqused_ori_kv is not None),
        has_seqused_cmp_kv=int(call.seqused_cmp_kv is not None),
        has_cmp=int(call.has_cmp),
        batch_mode=_BATCH_MODE_BY_LABEL[layout.batch_mode],
        batch_count=layout.batch_count,
        query_rows_per_batch=(
            prepared.query.shape[0]
            if layout.batch_count == 1
            else layout.query_rows_per_batch
        ),
        cube_core_count=prepared.cube_core_count,
        has_seqused_q=int(call.seqused_q is not None),
        ori_mask_mode=call.ori_mask_mode,
        cmp_mask_mode=call.cmp_mask_mode,
        cmp_ratio=call.cmp_ratio,
        is_ori_dense=int(call.is_ori_dense),
        kv_mode=_KV_MODE_BY_LABEL[("pa" if call.pa_mode else "linear")],
        ori_page_size=storage.ori.page_size,
        cmp_page_size=storage.cmp.page_size if call.has_cmp else 0,
        ori_page_stride_rows=storage.ori.stride_rows,
        cmp_page_stride_rows=storage.cmp.stride_rows if call.has_cmp else 0,
        ori_block_table_width=storage.ori.table_width,
        cmp_block_table_width=storage.cmp.table_width if call.has_cmp else 0,
        ori_page_size_reciprocal=ori_reciprocal,
        cmp_page_size_reciprocal=cmp_reciprocal if call.has_cmp else 0,
        ori_page_size_log2=ori_shift,
        cmp_page_size_log2=cmp_shift if call.has_cmp else 0,
        ori_page_size_is_power_of_two=ori_pow2,
        cmp_page_size_is_power_of_two=cmp_pow2 if call.has_cmp else 0,
        softmax_scale=call.softmax_scale,
    )
    return const_info


def _pack_n64_kernel_arguments(
    call, prepared, storage, lengths, scratch, outputs, const_info
):
    cmp_kv_gm = storage.cmp.tensor if call.has_cmp else storage.ori.tensor
    cmp_sparse_indices_gm = (
        prepared.cmp_indices if call.has_cmp else prepared.ori_indices
    )
    arguments = (
        prepared.query.view(-1),
        storage.ori.tensor.reshape(-1),
        cmp_kv_gm.reshape(-1),
        prepared.ori_indices.view(-1),
        cmp_sparse_indices_gm.view(-1),
        scratch.ori_table.view(-1),
        scratch.cmp_table.view(-1),
        scratch.ori_kv_physical_rows_gm,
        scratch.cmp_kv_physical_rows_gm,
        lengths.ori_topk_length_gm,
        lengths.cmp_topk_length_gm,
        lengths.seqused_ori_kv_gm,
        lengths.seqused_cmp_kv_gm,
        scratch.cmp_residual_kv_gm,
        scratch.metadata_tensor,
        scratch.seqused_q_gm,
        scratch.cu_q.reshape(-1),
        scratch.cu_ori_kv.reshape(-1),
        scratch.cu_cmp_kv.reshape(-1),
        scratch.sinks_gm,
        *(tensor.reshape(-1) for tensor in outputs),
        const_info,
        prepared.cube_core_count
        if call.metadata is not None
        else min(prepared.query.shape[0], prepared.cube_core_count),
    )
    return arguments


def _prepare_kernel_arguments(call, layout, storage, prepared):
    """Pack validated host descriptors without changing the device argument ABI."""
    dtype_label = _storage_dtype_label(prepared.query.dtype)
    outputs = (
        torch.empty_like(prepared.query, memory_format=torch.contiguous_format),
        torch.empty(
            prepared.lse_shape, dtype=torch.float32, device=prepared.query.device
        ),
    )
    if prepared.query.shape[0] == 0:
        return (), outputs, None
    lengths = _prepare_n64_length_descriptors(call, prepared)
    scratch = _prepare_n64_scratch(call, layout, prepared)
    const_info = _build_n64_runtime_constants(call, layout, storage, prepared)
    arguments = _pack_n64_kernel_arguments(
        call, prepared, storage, lengths, scratch, outputs, const_info
    )
    specialization = (
        prepared.query.shape[1],
        call.sinks is not None,
        call.return_softmax_lse,
        dtype_label,
        layout.batch_mode,
        "pa" if call.pa_mode else "linear",
        bool(call.is_ori_dense),
    )
    return arguments, outputs, specialization


def _validate_sparse_kv(query, kv, rank, batch_count, name):
    _validate_tensor(kv, query.dtype, query.device, name)
    _require(
        kv.ndim == rank and tuple(kv.shape[-2:]) == (1, _HEAD_DIM) and kv.numel() > 0,
        f"{name} has an invalid shape",
    )
    if rank == _BSND_RANK:
        _require(kv.shape[0] == batch_count, f"{name} batch size must match query")


def _validate_sparse_indices(query, indices, prefixes, name):
    _validate_tensor(indices, torch.int32, query.device, name)
    _require(
        indices.ndim >= 2
        and tuple(indices.shape[:-1]) in prefixes
        and indices.shape[-1] > 0,
        f"{name} has an invalid shape",
    )


def _prepare_pa_kv(query, kv, block_table, batch_count, seqused, name):
    _require(isinstance(kv, torch.Tensor), f"{name} must be a Tensor")
    _require(
        kv.dtype == query.dtype and kv.device == query.device,
        f"{name} has an invalid dtype or device",
    )
    _require(
        kv.ndim == _BSND_RANK
        and kv.shape[0] > 0
        and kv.shape[1] > 0
        and tuple(kv.shape[-2:]) == (1, _HEAD_DIM),
        f"{name} PA shape must be [blocks, page_size, 1, {_HEAD_DIM}]",
    )
    _require(
        kv.stride(-1) == 1
        and kv.stride(-2) == _HEAD_DIM
        and kv.stride(1) == _HEAD_DIM
        and kv.stride(0) % _HEAD_DIM == 0,
        f"{name} PA inner rows must be contiguous",
    )
    page_size = kv.shape[1]
    _require(page_size <= 1024, f"{name} PA page size must be <= 1024")
    page_stride_rows = kv.stride(0) // _HEAD_DIM
    _require(
        page_stride_rows >= page_size,
        f"{name} PA page stride is smaller than page size",
    )
    _validate_tensor(block_table, torch.int32, query.device, f"{name} block table")
    _require(
        block_table.ndim == 2
        and block_table.shape[0] == batch_count
        and block_table.shape[1] > 0,
        f"{name} block table must be INT32 [B, max_pages]",
    )
    _require(
        block_table.shape[1] * page_size < _INT32_LIMIT,
        f"{name} PA logical capacity exceeds the int32 ABI",
    )
    _require(
        block_table.shape[1] <= _PA_BLOCK_TABLE_CAPACITY,
        f"{name} block table exceeds PA prepass capacity",
    )
    _require(seqused is not None, f"{name} PA requires seqused KV")
    _validate_seqused_kv(seqused, batch_count, query, f"{name} seqused KV")
    storage_rows = (kv.shape[0] - 1) * page_stride_rows + page_size
    # This row view includes physical padding between pages.  It is contiguous
    # over the underlying allocation and can be flattened without repacking.
    buffer_storage = kv.as_strided(
        (storage_rows, 1, _HEAD_DIM),
        (_HEAD_DIM, _HEAD_DIM, 1),
    )
    return buffer_storage, page_size, page_stride_rows, block_table.shape[1]


def sparse_n64_kernel(
    query_gm,
    *,
    ori_kv,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False,
):
    """Metadata-scheduled N1<=64 sparse SMLA, including PA KV."""
    call = SparseN64Call(
        query_gm=query_gm,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        topk_value_mode=topk_value_mode,
        return_softmax_lse=return_softmax_lse,
    )
    rank = _validate_n64_query(call)
    _validate_n64_modes(call)
    _validate_n64_sparse_selection(call, rank)
    layout = _validate_n64_layout(call, rank)
    _validate_n64_lengths(call, layout)
    storage = _prepare_n64_storage(call, layout)
    prepared = _prepare_n64_call_descriptors(call, layout)
    arguments, outputs, specialization = _prepare_kernel_arguments(
        call, layout, storage, prepared
    )
    if specialization is not None:
        _get_compiled_sparse_n64(*specialization)(*arguments)
    attention_out_gm, softmax_lse_gm = outputs
    return attention_out_gm.view(query_gm.shape), softmax_lse_gm
