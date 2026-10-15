# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""High-performance sparse flash MLA kernels and standalone-aligned admission."""

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
from types import SimpleNamespace
import torch
import cannbotdsl
from cannbotdsl import (
    CArray,
    Dim,
    L0C,
    MemLoc,
    Tensor,
    TensorSpec,
    TensorViewSpec,
    UB,
    ViewLayout,
    copy_options,
    datastruct,
    dtypes,
    host,
    is_aic,
    load,
    make_buffer,
    make_layout,
    make_ring_buffer,
)
from cannbotdsl.ops.cube import set_mmad_direction
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.ops.arch import get_subblock_id, get_block_idx
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops import matmul
from cannbotdsl.tensor import reinterpret, tile_slice
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr

_BITS_PER_BYTE = 8
_F32_BITS = 32
_F16_BITS = 16
_I32_BITS = 32
_VECTOR_REGISTER_BITS = 2048
_BF16_BYTES = _F16_BITS // _BITS_PER_BYTE
_F32_BYTES = _F32_BITS // _BITS_PER_BYTE
_I32_BYTES = _I32_BITS // _BITS_PER_BYTE
_D = 512
_TILE = 128
_Q_TILE_ROWS = 64
_GATHER_TILE_ROWS = 16
# A 2048-bit vector holds 64 float32 values.
_F32_VL = _VECTOR_REGISTER_BITS // _F32_BITS
# Fixed BF16 specialization and Cube/NZ tile dimensions.
_SMALL_Q_TILE_ROWS = _Q_TILE_ROWS // 2
_VECTOR_SUBBLOCKS = 2
_NZ_ROW = 16
_NZ_FACE = _NZ_ROW * _NZ_ROW
_NZ_PADDED_ROWS = _GATHER_TILE_ROWS + 1
_NZ_FRAGMENT_ELEMS = _NZ_PADDED_ROWS * _D
_NZ_FRAGMENT_BYTES = _NZ_FRAGMENT_ELEMS * _BF16_BYTES
_NZ_QUARTER_STRIDE = _NZ_PADDED_ROWS * _TILE
# A DAV-3510 vector register is 2048 bits wide.
_BF16_VL = _VECTOR_REGISTER_BITS // _F16_BITS
_KV_ROWS_PER_VECTOR = _TILE // _VECTOR_SUBBLOCKS
_GATHER_CHUNKS = 2
_GATHER_PAIRS = _GATHER_TILE_ROWS // 2
_GATHER_ELEMS = _GATHER_TILE_ROWS * _D
_GATHER_BYTES = _GATHER_ELEMS * _BF16_BYTES
_RESULT_HALF_COLS = _D // 2
# UB output reuses FP32 partial-output storage as BF16, preserving row pitch.
_OUT_UB_ROW_ELEMS = _D * _F32_BYTES // _BF16_BYTES
_Q_WIDE_COLS = 3 * _TILE
_Q_SLOT_WIDTHS = (_Q_WIDE_COLS, _Q_WIDE_COLS, _TILE)
_Q_L1_RESERVED_BYTES = 56 * 1024
_P_L1_RESERVED_BYTES = 8 * 1024
_KV_SLOT_ELEMS = _KV_ROWS_PER_VECTOR * _D
_KV_RING_DEPTH = 3
_L0A_RING_DEPTH = 4
_L0B_SLOTS = 4
_D_F32_TILES = _D // _F32_VL
_WIDE_F32_TILES = _Q_WIDE_COLS // _F32_VL
_F32_NZ_TILES = _F32_VL // _NZ_ROW
_TILE_NZ_TILES = _TILE // _NZ_ROW
# Physical memory arenas for this kernel; their used ranges are checked below.
_UB_ARENA_BYTES = 256 * 1024
_L0C_ARENA_BYTES = 256 * 1024
_INT32_LIMIT = 1 << 31
# Index load at tick 0; gather at 1; softmax at 3; output update at 4.
# Five delay-line slots retain all in-flight tiles, including the drain tick.
_STAGE_GATHER = 1
_STAGE_SOFTMAX = 3
_STAGE_UPDATE = 4
_PIPELINE_SLOTS = _STAGE_UPDATE + 1
# Each slot stores pipeline metadata in the order declared by the delay-line fields.
_PIPELINE_FIELDS_PER_SLOT = 9
_PIPELINE_METADATA_SIZE = _PIPELINE_SLOTS * _PIPELINE_FIELDS_PER_SLOT
_F32_NEG_INF_BITS = 0xFF800000
_F32_POS_INF_BITS = 0x7F800000
_F32_MIN = -3.4028234663852886e38

# Host tensor ranks and public mask-mode codes.
_TND_RANK = 3
_BSND_RANK = 4
_MASK_CAUSAL = 3
_MASK_SLIDING_WINDOW = 4
# Storage alignment, cache sizing, and device instruction options.
_UB_CAPACITY_ALIGNMENT_BYTES = 8
_COMPILED_VARIANT_CACHE_SIZE = 8
_BUILD_HASH_PREFIX_LENGTH = 16
_KV_COPY_L2_CACHE_CONTROL = 4
_MMAD_FINAL_UNIT_FLAG = 3


_COMPILER_FLAGS = (
    "-Ofast",
    "-mllvm",
    "-enable-hiipu-vf-loop-unroll",
    "-mllvm",
    "-cce-aicore-dcpreload-args=false",
    "--cce-res-usage",
)
_COMPILER_WRAPPER = """import sys, subprocess
real, flags = CONFIG
args = sys.argv[1:]
if '-c' in args and '-x' in args and 'asc' in args:
    args = list(flags) + [x for x in args if x != '-O3']
sys.exit(subprocess.run([real] + args).returncode)
"""

_COMPILATION_LOCK = threading.RLock()


@jit
def scalar_select(condition, yes, no):
    return dtypes.int64(yes) if condition else dtypes.int64(no)


def nz_view(tensor, rows, cols):
    return reinterpret(tensor, shape=(rows, cols), data_format="nz")


def nz_column_tile(tensor, rows, column_tile):
    matrix = nz_view(tensor, rows, math.prod(tensor.shape) // rows)
    return tile_slice(matrix, (rows, _F32_VL), (0, column_tile))


def qk_operand_half(tensor, rows, half):
    # QK joins two NZ row groups in one operand buffer.
    return reinterpret(
        tensor, shape=(rows, _F32_VL), offset=half * rows * _NZ_ROW * _BF16_BYTES
    )


def pv_operand_half(tensor, half):
    return reinterpret(
        tensor,
        shape=(_F32_VL, _KV_ROWS_PER_VECTOR),
        offset=half * _F32_VL * _KV_ROWS_PER_VECTOR * _BF16_BYTES,
        data_format="nz",
    )


def padded_nz_columns(tensor, columns, rows, row_capacity):
    return tensor.view(columns // _NZ_ROW, row_capacity * _NZ_ROW)[
        None, 0 : rows * _NZ_ROW
    ]


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
        "CONFIG", repr((real_compiler, _COMPILER_FLAGS)), 1
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
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def allocate_buffer(
    spaces, loc, element_count, dtype, address, shape=None, cross=False
):
    view = (
        spaces[loc][
            address : address + element_count * (dtype.width // _BITS_PER_BYTE),
        ]
        .view(dtype=dtype)
        .view(shape or (element_count,))
    )
    if cross:
        return make_ring_buffer([view])
    return make_buffer(view)


def make_ub_plan(query_tile_rows, sink):
    if query_tile_rows not in (_SMALL_Q_TILE_ROWS, _Q_TILE_ROWS):
        raise ValueError("Unsupported query tile capacity")
    subblock_query_rows = query_tile_rows // _VECTOR_SUBBLOCKS
    fields = [
        ("gather", _GATHER_CHUNKS * _GATHER_ELEMS, dtypes.bfloat16),
        ("nz", _GATHER_CHUNKS * _NZ_FRAGMENT_ELEMS, dtypes.bfloat16),
        ("output_accumulator", subblock_query_rows * _D, dtypes.float32),
        ("output_partial", subblock_query_rows * _D, dtypes.float32),
        ("score", subblock_query_rows * _TILE, dtypes.float32),
        ("p_out", (subblock_query_rows + 1) * _TILE, dtypes.bfloat16),
        ("sink", subblock_query_rows if sink else 0, dtypes.float32),
        ("maximum", 2 * _F32_VL, dtypes.float32),
        ("sums", 2 * _F32_VL, dtypes.float32),
        ("alpha", 2 * 2 * _F32_VL, dtypes.float32),
        ("raw_indices", _TILE, dtypes.int32),
    ]
    offset = 0
    plan = {}
    for name, element_count, dtype in fields:
        # Align each field to its element size for typed UB views.
        alignment = dtype.width // _BITS_PER_BYTE
        offset = (offset + alignment - 1) // alignment * alignment
        plan[name] = (offset, element_count, dtype)
        offset += element_count * alignment
    # The arena capacity check rounds total UB usage up to an 8-byte boundary.
    if (
        offset + (_UB_CAPACITY_ALIGNMENT_BYTES - 1)
    ) // _UB_CAPACITY_ALIGNMENT_BYTES * _UB_CAPACITY_ALIGNMENT_BYTES > _UB_ARENA_BYTES:
        raise ValueError("UB plan exceeds the reserved arena")
    return plan


class HighPerfSmlaStorage:
    # Shared physical buffer layout.

    def __init__(self, query_tile_rows, sink):
        spaces = {
            MemLoc.UB: UB.view(_UB_ARENA_BYTES),
            MemLoc.L0C: L0C.view(_L0C_ARENA_BYTES),
        }
        self.query_tile_rows, self.subblock_query_rows = (
            query_tile_rows,
            query_tile_rows // _VECTOR_SUBBLOCKS,
        )
        self.plan = make_ub_plan(query_tile_rows, sink)
        for name, (address, element_count, dtype) in self.plan.items():
            if element_count and name not in (
                "output_accumulator",
                "output_partial",
                "gather",
                "nz",
                "raw_indices",
            ):
                setattr(
                    self,
                    name,
                    allocate_buffer(
                        spaces,
                        MemLoc.UB,
                        element_count,
                        dtype,
                        address,
                        (self.subblock_query_rows, _TILE) if name == "score" else None,
                        cross=name == "score",
                    ),
                )
        result_address, result_count, result_dtype = self.plan["output_accumulator"]
        result_full = (
            spaces[MemLoc.UB][
                result_address : result_address + result_count * _F32_BYTES * 2,
            ]
            .view(dtype=result_dtype)
            .view(self.subblock_query_rows * 2, _D)
        )
        self.result_channels = [
            make_ring_buffer(
                [
                    tile_slice(
                        result_full,
                        (self.subblock_query_rows * 2, _RESULT_HALF_COLS),
                        (0, half),
                    )
                ]
            )
            for half in range(2)
        ]
        self.gather = [
            allocate_buffer(
                spaces, MemLoc.UB, _GATHER_ELEMS, dtypes.bfloat16, half * _GATHER_BYTES
            )
            for half in range(2)
        ]
        self.nz_fragments = [
            allocate_buffer(
                spaces,
                MemLoc.UB,
                _NZ_FRAGMENT_ELEMS,
                dtypes.bfloat16,
                self.plan["nz"][0] + fragment * _NZ_FRAGMENT_BYTES,
            )
            for fragment in range(2)
        ]
        raw_address = self.plan["raw_indices"][0]
        self.index_slots = [
            allocate_buffer(
                spaces,
                MemLoc.UB,
                _KV_ROWS_PER_VECTOR,
                dtypes.int32,
                raw_address + slot * _KV_ROWS_PER_VECTOR * _I32_BYTES,
            )
            for slot in range(2)
        ]
        self.q_lower_heads_l1 = [
            Buffer(
                MemLoc.L1,
                (self.subblock_query_rows, width),
                dtypes.bfloat16,
                data_format="nz",
            )
            for width in _Q_SLOT_WIDTHS
        ]
        if self.subblock_query_rows < _SMALL_Q_TILE_ROWS:
            self.q_lower_heads_l1_gap = Buffer(
                MemLoc.L1,
                (
                    _Q_L1_RESERVED_BYTES
                    - self.subblock_query_rows * sum(_Q_SLOT_WIDTHS) * _BF16_BYTES,
                ),
                dtypes.uint8,
                data_format="nd",
            )
        self.kv_channels = [
            RingBuffer(
                MemLoc.L1,
                (_KV_SLOT_ELEMS,),
                dtypes.bfloat16,
                depth=_KV_RING_DEPTH,
                data_format="nd",
            )
        ]
        if self.query_tile_rows < _Q_TILE_ROWS:
            self.p_gap0 = Buffer(
                MemLoc.L1,
                (_P_L1_RESERVED_BYTES - self.query_tile_rows * _TILE,),
                dtypes.uint8,
                data_format="nd",
            )
        self.p_l1 = RingBuffer(
            MemLoc.L1,
            (query_tile_rows * _TILE,),
            dtypes.bfloat16,
            depth=1,
            data_format="nd",
        )
        if self.query_tile_rows < _Q_TILE_ROWS:
            self.p_gap1 = Buffer(
                MemLoc.L1,
                (_P_L1_RESERVED_BYTES - self.query_tile_rows * _TILE,),
                dtypes.uint8,
                data_format="nd",
            )
        self.kv_channels.append(
            RingBuffer(
                MemLoc.L1,
                (_KV_SLOT_ELEMS,),
                dtypes.bfloat16,
                depth=_KV_RING_DEPTH,
                data_format="nd",
            )
        )
        self.q_upper_heads_l1 = [
            Buffer(
                MemLoc.L1,
                (self.subblock_query_rows, width),
                dtypes.bfloat16,
                data_format="nz",
            )
            for width in _Q_SLOT_WIDTHS
        ]
        self.l0_a_ring = RingBuffer(
            MemLoc.L0A,
            depth=_L0A_RING_DEPTH,
            view=TensorViewSpec(
                (query_tile_rows, _TILE),
                dtypes.bfloat16,
                layout=ViewLayout(
                    "nz",
                    physical_shape=(
                        _TILE_NZ_TILES,
                        query_tile_rows // _NZ_ROW,
                        _NZ_ROW,
                        _NZ_ROW,
                    ),
                    physical_stride=(query_tile_rows * _NZ_ROW, _NZ_FACE, _NZ_ROW, 1),
                ),
            ),
        )
        self.l0_b_slots = [
            Buffer(
                MemLoc.L0B,
                dtype=dtypes.bfloat16,
                capacity=(_TILE, _F32_VL),
                data_format="nz",
                layout=make_layout((_TILE, _F32_VL)),
                physical_layout=make_layout(
                    (_F32_NZ_TILES, _TILE_NZ_TILES, _NZ_ROW, _NZ_ROW),
                    stride=(_TILE * _NZ_ROW, _NZ_FACE, _NZ_ROW, 1),
                ),
            )
            for slot in range(_L0B_SLOTS)
        ]
        self.pv_l0c_buffers = [
            allocate_buffer(
                spaces,
                MemLoc.L0C,
                query_tile_rows * _F32_VL,
                dtypes.float32,
                col * query_tile_rows * _F32_VL * _F32_BYTES,
            )
            for col in range(_D_F32_TILES)
        ]
        self.qk_l0c_buffer = allocate_buffer(
            spaces,
            MemLoc.L0C,
            query_tile_rows * _TILE,
            dtypes.float32,
            query_tile_rows * _D * _F32_BYTES,
        )
        self.cross_channels = {name: getattr(self, name) for name in ("score", "p_l1")}

    def next_score(self, write):
        channel = self.cross_channels["score"]
        return channel.head_next() if write else channel.tail_next()

    def next_kv(self, write):
        if write:
            return (self.kv_channels[0].head_next(), self.kv_channels[1].head_next())
        return (self.kv_channels[0].tail_next(), self.kv_channels[1].tail_next())

    def next_result(self, write):
        first = (
            self.result_channels[0].head_next()
            if write
            else self.result_channels[0].tail_next()
        )
        second = (
            self.result_channels[1].head_next()
            if write
            else self.result_channels[1].tail_next()
        )
        return (
            tile_slice(first, (self.subblock_query_rows, _RESULT_HALF_COLS), (0, 0)),
            tile_slice(first, (self.subblock_query_rows, _RESULT_HALF_COLS), (1, 0)),
            tile_slice(second, (self.subblock_query_rows, _RESULT_HALF_COLS), (0, 0)),
            tile_slice(second, (self.subblock_query_rows, _RESULT_HALF_COLS), (1, 0)),
        )


class HighPerfSmlaVec0Indices:
    def __init__(self, storage):
        self.storage = storage

    @jit
    def initialize(self):
        for slot in range_constexpr(2):
            # VF gather initialization: zero each local BF16 gather slot.
            with vf(mode="simd"):
                mask = rr.full_mask(elem_bits=_F16_BITS)
                zero_reg = rr.vdups(0.0, dtypes.bfloat16, mask=mask)
                for i in range(_GATHER_ELEMS // _BF16_VL):
                    rr.vstore(self.storage.gather[slot], i * _BF16_VL, zero_reg, mask)

    @jit
    def load_idx_tile(
        self,
        indices,
        query_row_idx,
        tile,
        length,
        index_stride,
        prior_query_tile_count,
        global_tile,
    ):
        kv_index_start = tile * _TILE + get_subblock_id() * _KV_ROWS_PER_VECTOR
        remaining_kv_indices = length - kv_index_start
        valid_index_count = scalar_select(
            remaining_kv_indices > 0,
            scalar_select(
                remaining_kv_indices < _KV_ROWS_PER_VECTOR,
                remaining_kv_indices,
                _KV_ROWS_PER_VECTOR,
            ),
            0,
        )
        target = self.storage.index_slots[0]
        if (prior_query_tile_count + global_tile) % 2 != 0:
            target = self.storage.index_slots[1]
        if valid_index_count > 0:
            index_row = indices[query_row_idx * index_stride :,].view(index_stride)
            mem_copy(
                target[0:valid_index_count,],
                index_row[kv_index_start : kv_index_start + valid_index_count,],
            )

    @jit
    def gather_batch(self, kv, global_tile, half, kv_len, chunk, length, local_tile):
        kv_indices = self.storage.index_slots[0]
        if global_tile % 2 != 0:
            kv_indices = self.storage.index_slots[1]
        valid_segment_kv_rows = scalar_select(length < kv_len, length, kv_len)
        remaining_kv_indices = (
            valid_segment_kv_rows
            - local_tile * _TILE
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
        kv_row_start = half * (_KV_ROWS_PER_VECTOR // 2) + chunk * _GATHER_TILE_ROWS
        if valid_subblock_kv_rows >= kv_row_start + _GATHER_TILE_ROWS:
            self.gather_rows(
                kv, kv_indices, chunk, kv_row_start, valid_subblock_kv_rows, True
            )
        else:
            self.gather_rows(
                kv, kv_indices, chunk, kv_row_start, valid_subblock_kv_rows, False
            )

    @jit
    def gather_rows(
        self,
        kv,
        kv_indices,
        half,
        kv_row_start,
        valid_subblock_kv_rows,
        is_full_gather_chunk,
    ):
        for i in range_constexpr(_GATHER_PAIRS):
            # Each gather pair copies two adjacent destination KV rows.
            kv_row = kv_row_start + i * 2
            if const_expr(is_full_gather_chunk):
                first_kv_row = dtypes.int64(dtypes.uint32(kv_indices[kv_row]))
                second_kv_row = dtypes.int64(dtypes.uint32(kv_indices[kv_row + 1]))
                lower_kv_row = scalar_select(
                    first_kv_row < second_kv_row, first_kv_row, second_kv_row
                )
                higher_kv_row = scalar_select(
                    first_kv_row < second_kv_row, second_kv_row, first_kv_row
                )
                src = kv.view(math.prod(kv.shape) // _D, _D)
                dst = self.storage.gather[half].view(_GATHER_TILE_ROWS, _D)
                mem_copy(
                    tile_slice(dst, (2, _D), (i, 0)),
                    (
                        tile_slice(src, (1, _D), (lower_kv_row, 0)).view(1, _D),
                        tile_slice(src, (1, _D), (higher_kv_row, 0)).view(1, _D),
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
                    first_kv_row = dtypes.int64(dtypes.uint32(first_kv_index))
                    second_kv_row = dtypes.int64(dtypes.uint32(second_kv_index))
                    lower_kv_row = scalar_select(
                        first_kv_row < second_kv_row, first_kv_row, second_kv_row
                    )
                    higher_kv_row = scalar_select(
                        first_kv_row < second_kv_row, second_kv_row, first_kv_row
                    )
                    src = kv.view(math.prod(kv.shape) // _D, _D)
                    dst = self.storage.gather[half].view(_GATHER_TILE_ROWS, _D)
                    mem_copy(
                        tile_slice(dst, (2, _D), (i, 0)),
                        (
                            tile_slice(src, (1, _D), (lower_kv_row, 0)).view(1, _D),
                            tile_slice(src, (1, _D), (higher_kv_row, 0)).view(1, _D),
                        ),
                        options=copy_options(l2_cache_ctl=1),
                    )
                elif first_kv_index >= 0 or second_kv_index >= 0:
                    valid_kv_index = scalar_select(
                        first_kv_index >= 0, first_kv_index, second_kv_index
                    )
                    mem_copy(
                        tile_slice(
                            self.storage.gather[half].view(_GATHER_TILE_ROWS, _D),
                            (1, _D),
                            (2 * i, 0),
                        ),
                        tile_slice(
                            kv.view(math.prod(kv.shape) // _D, _D),
                            (1, _D),
                            (valid_kv_index, 0),
                        ),
                        options=copy_options(l2_cache_ctl=1),
                    )

    @jit
    def restage_nz_fragment(self, chunk):
        source = self.storage.gather[chunk]
        # VF layout conversion: restage gathered KV rows into the padded NZ fragment.
        with vf(mode="simd"):
            mask = rr.full_mask(elem_bits=_F16_BITS)
            for kv_row in range(_GATHER_TILE_ROWS):
                for quarter in range_constexpr(_D // _BF16_VL):
                    value = rr.vload(source, kv_row * _D + quarter * _BF16_VL)
                    rr.vstore_strided(
                        self.storage.nz_fragments[chunk],
                        quarter * _NZ_QUARTER_STRIDE + kv_row * _NZ_ROW,
                        value,
                        mask,
                        block_stride=_NZ_PADDED_ROWS,
                        repeat_stride=0,
                    )


class HighPerfSmlaVector:
    def __init__(self, storage, has_sink, return_softmax_lse):
        self.storage, self.has_sink = (storage, has_sink)
        self.return_softmax_lse = return_softmax_lse
        self.subblock_query_rows = storage.subblock_query_rows

    @jit
    def softmax_rest_full(
        self, job, tile, scale, first, all_valid, buf_score, buf_p, valid_tile_kv_rows
    ):
        # VF softmax: reduce QK scores, update online state and pack BF16 P in NZ order.
        with vf(mode="simd"):
            mask = rr.full_mask(elem_bits=_F32_BITS)
            mask16 = rr.full_mask(elem_bits=_F16_BITS)
            # Keep two FP32 copies per head for the interleaved online softmax state.
            two_lanes, _ = rr.update_mask(2, elem_bits=_F32_BITS)
            softmax_state_offset = job % 2 * _F32_VL
            if const_expr(first and self.has_sink):
                sink = rr.vload(self.storage.sink)
                neg_inf = rr.veqs(
                    rr.vreinterpret(sink, dtypes.uint32), _F32_NEG_INF_BITS, mask=mask
                )
                pos_inf = rr.veqs(
                    rr.vreinterpret(sink, dtypes.uint32), _F32_POS_INF_BITS, mask=mask
                )
                unscaled_sink_reg = rr.vmuls(sink, 1.0 / scale, mask=mask)
                unscaled_sink_reg = rr.vdups(
                    _F32_MIN,
                    dtypes.float32,
                    mask=neg_inf,
                    mode="merging",
                    merge=unscaled_sink_reg,
                )
                unscaled_sink_reg = rr.vdups(
                    -_F32_MIN,
                    dtypes.float32,
                    mask=pos_inf,
                    mode="merging",
                    merge=unscaled_sink_reg,
                )
                replicated, unused = rr.vinterleave(
                    unscaled_sink_reg, unscaled_sink_reg
                )
                rr.vstore(self.storage.maximum, softmax_state_offset, replicated, mask)
                initial_sum = rr.vdups(1.0, dtypes.float32)
                initial_sum = rr.vdups(
                    0.0, dtypes.float32, mask=neg_inf, mode="merging", merge=initial_sum
                )
                replicated, unused = rr.vinterleave(initial_sum, initial_sum)
                rr.vstore(self.storage.sums, softmax_state_offset, replicated, mask)
                # Order these state stores before the following state loads.
                rr.vmem_bar("vst_vld")
            if const_expr(not first or self.has_sink):
                old_max = rr.vload(self.storage.maximum, softmax_state_offset)
                rr.vmem_bar("vld_vst")
                old_sum = rr.vload(self.storage.sums, softmax_state_offset)
                old_max = rr.vmuls(old_max, scale, mask=mask)

            neg_inf_reg = rr.vreinterpret(
                rr.vdups(_F32_NEG_INF_BITS, dtypes.uint32), dtypes.float32
            )
            if const_expr(not all_valid):
                even_index_reg = rr.vmuls(rr.varange(0, dtypes.int32), 2, mask=mask)
                odd_index_reg = rr.vadds(even_index_reg, 1, mask=mask)
                invalid_even = rr.vges(
                    even_index_reg, dtypes.int32(valid_tile_kv_rows), mask=mask
                )
                invalid_odd = rr.vges(
                    odd_index_reg, dtypes.int32(valid_tile_kv_rows), mask=mask
                )
            max_cursor = rr.vstore_unalign_begin(self.storage.maximum)
            # Explicit int32 loop values select the VF scalar ABI.
            for query_head_row in cannbotdsl.range(
                dtypes.int32(0),
                dtypes.int32(self.subblock_query_rows),
                dtypes.int32(1),
                unroll=1,
            ):
                score_even_reg, score_odd_reg = rr.vload_deinterleave(
                    buf_score, query_head_row * _TILE
                )
                if const_expr(not all_valid):
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
                if const_expr(first and not self.has_sink):
                    prior_max_reg = rr.vdups(_F32_MIN, dtypes.float32)
                else:
                    prior_max_reg = rr.vload_broadcast(
                        self.storage.maximum, softmax_state_offset + 2 * query_head_row
                    )
                row_max_reg = rr.vmax(row_max_reg, prior_max_reg, mask=mask)
                squeezed = rr.vsqueeze_and_storeunalign_init(
                    row_max_reg, mask=two_lanes
                )
                rr.vsqueeze_and_storeunalign(
                    self.storage.maximum, softmax_state_offset, squeezed, max_cursor
                )
            rr.vsqueeze_and_storeunalign_finalize(
                self.storage.maximum, softmax_state_offset, max_cursor
            )
            rr.vmem_bar("vst_vld")
            if const_expr(not first or self.has_sink):
                row_max_reg = rr.vmuls(
                    rr.vload(self.storage.maximum, softmax_state_offset),
                    scale,
                    mask=mask,
                )
                rescale_factor_reg = rr.vexp_sub(old_max, row_max_reg, mask=mask)
                rr.vstore(
                    self.storage.alpha,
                    job % 2 * _TILE + tile % 2 * _F32_VL,
                    rescale_factor_reg,
                    mask,
                )
            sum_cursor = rr.vstore_unalign_begin(self.storage.sums)
            for query_head_row in cannbotdsl.range(
                dtypes.int32(0),
                dtypes.int32(self.subblock_query_rows),
                dtypes.int32(1),
                unroll=1,
            ):
                score_even_reg, score_odd_reg = rr.vload_deinterleave(
                    buf_score, query_head_row * _TILE
                )
                if const_expr(not all_valid):
                    score_even_reg = rr.vselect(
                        neg_inf_reg, score_even_reg, cond_mask=invalid_even
                    )
                    score_odd_reg = rr.vselect(
                        neg_inf_reg, score_odd_reg, cond_mask=invalid_odd
                    )
                row_max_reg = rr.vmuls(
                    rr.vload_broadcast(
                        self.storage.maximum, softmax_state_offset + 2 * query_head_row
                    ),
                    scale,
                    mask=mask,
                )
                exp_even_reg = rr.vexp_sub(
                    rr.vmuls(score_even_reg, scale, mask=mask), row_max_reg, mask=mask
                )
                exp_odd_reg = rr.vexp_sub(
                    rr.vmuls(score_odd_reg, scale, mask=mask), row_max_reg, mask=mask
                )
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
                    self.storage.sums, softmax_state_offset, squeezed, sum_cursor
                )
                p_even_reg = rr.vcast(
                    exp_even_reg,
                    dtypes.bfloat16,
                    mask=mask,
                    rounding="rn",
                    reg_layout=rr.RegLayout.ZERO,
                )
                p_odd_reg = rr.vcast(
                    exp_odd_reg,
                    dtypes.bfloat16,
                    mask=mask,
                    rounding="rn",
                    reg_layout=rr.RegLayout.ONE,
                )
                packed_p_reg = rr.vadd(p_even_reg, p_odd_reg, mask=mask16)
                rr.vstore_strided(
                    buf_p,
                    query_head_row * _NZ_ROW,
                    packed_p_reg,
                    mask16,
                    block_stride=self.subblock_query_rows + 1,
                    repeat_stride=0,
                )
            rr.vsqueeze_and_storeunalign_finalize(
                self.storage.sums, softmax_state_offset, sum_cursor
            )
            if const_expr(not first or self.has_sink):
                rr.vmem_bar("vst_vld")
                row_sum_reg = rr.vmadd(
                    old_sum,
                    rescale_factor_reg,
                    rr.vload(self.storage.sums, softmax_state_offset),
                    mask=mask,
                )
                rr.vstore(self.storage.sums, softmax_state_offset, row_sum_reg, mask)

    @jit
    def finalize_softmax_state(self, job, scale):
        # VF softmax finalization: compact per-head state and optionally produce LSE.
        with vf(mode="simd"):
            mask = rr.full_mask(elem_bits=_F32_BITS)
            softmax_state_offset = job % 2 * _F32_VL
            softmax_max_reg, unused = rr.vdeinterleave(
                rr.vload(self.storage.maximum, softmax_state_offset),
                rr.vload(self.storage.maximum, softmax_state_offset),
            )
            softmax_sum_reg, unused = rr.vdeinterleave(
                rr.vload(self.storage.sums, softmax_state_offset),
                rr.vload(self.storage.sums, softmax_state_offset),
            )
            empty_row_mask = rr.veqs(softmax_max_reg, _F32_MIN, mask=mask)
            pos_inf_reg = rr.vreinterpret(
                rr.vdups(_F32_POS_INF_BITS, dtypes.uint32), dtypes.float32
            )
            if const_expr(self.return_softmax_lse):
                scaled_max_reg = rr.vmuls(softmax_max_reg, scale, mask=mask)
                lse = rr.vadd(
                    rr.vlog(softmax_sum_reg, mask=mask), scaled_max_reg, mask=mask
                )
                lse = rr.vselect(pos_inf_reg, lse, cond_mask=empty_row_mask)
                rr.vstore(self.storage.maximum, softmax_state_offset, lse, mask)
            softmax_sum_reg = rr.vselect(
                pos_inf_reg, softmax_sum_reg, cond_mask=empty_row_mask
            )
            rr.vstore(self.storage.sums, softmax_state_offset, softmax_sum_reg, mask)

    @jit
    def update_o(
        self,
        job,
        tile,
        first,
        last_loop,
        buf_output_accumulator,
        buf_output_partial,
        buf_out,
    ):
        # VF output update: accumulate PV tiles, then normalize and pack the final result.
        with vf(mode="simd"):
            mask = rr.full_mask(elem_bits=_F32_BITS)
            for query_head_row in range(self.subblock_query_rows):
                rescale_factor_reg = rr.vdups(1.0, dtypes.float32)
                inv_softmax_sum_reg = rr.vdups(1.0, dtypes.float32)
                if const_expr(not first):
                    rescale_factor_reg = rr.vload_broadcast(
                        self.storage.alpha,
                        job % 2 * _TILE + tile % 2 * _F32_VL + query_head_row * 2,
                    )
                if const_expr(last_loop):
                    inv_softmax_sum_reg = rr.vdiv(
                        rr.vdups(1.0, dtypes.float32),
                        rr.vload_broadcast(
                            self.storage.sums, job % 2 * _F32_VL + query_head_row
                        ),
                        mask=mask,
                    )
                for d_vector_tile in range_constexpr(_RESULT_HALF_COLS // _F32_VL):
                    output_element_offset = (
                        query_head_row * _D + d_vector_tile * _F32_VL
                    )
                    if const_expr(first):
                        output_reg = rr.vload(
                            buf_output_accumulator, output_element_offset
                        )
                    else:
                        output_reg = rr.vload(buf_output_partial, output_element_offset)
                        rescaled_previous_output_reg = rr.vmul(
                            rr.vload(buf_output_accumulator, output_element_offset),
                            rescale_factor_reg,
                            mask=mask,
                        )
                        output_reg = rr.vadd(
                            rescaled_previous_output_reg, output_reg, mask=mask
                        )
                    if const_expr(last_loop):
                        output_reg = rr.vmul(output_reg, inv_softmax_sum_reg, mask=mask)
                        packed_output_reg = rr.vcast(
                            output_reg, dtypes.bfloat16, mask=mask, rounding="rn"
                        )
                        rr.vstore_pack(
                            buf_out,
                            query_head_row * _OUT_UB_ROW_ELEMS
                            + d_vector_tile * _F32_VL,
                            packed_output_reg,
                            mask,
                            pack_mode="b32_to_b16",
                        )
                    else:
                        rr.vstore(
                            buf_output_accumulator,
                            output_element_offset,
                            output_reg,
                            mask,
                        )


class HighPerfSmlaMatmul:
    def __init__(self, storage):
        self.storage, self.query_tile_rows, self.subblock_query_rows = (
            storage,
            storage.query_tile_rows,
            storage.subblock_query_rows,
        )

    @jit
    def load_q_wide(self, q, query_row_idx, job, heads, load_d_tail):
        query_heads = q[query_row_idx * heads * _D :,].view(heads, _D)
        q_lo, q_hi = (
            self.storage.q_lower_heads_l1[0],
            self.storage.q_upper_heads_l1[0],
        )
        d_column_start = 0
        d_column_count = _Q_WIDE_COLS
        if const_expr(load_d_tail):
            q_lo, q_hi = (
                self.storage.q_lower_heads_l1[2],
                self.storage.q_upper_heads_l1[2],
            )
            d_column_start = _Q_WIDE_COLS
            d_column_count = _TILE
        elif job % 2 != 0:
            q_lo, q_hi = (
                self.storage.q_lower_heads_l1[1],
                self.storage.q_upper_heads_l1[1],
            )
        lower_query_heads = scalar_select(
            heads < self.subblock_query_rows, heads, self.subblock_query_rows
        )
        upper_query_heads = scalar_select(
            heads > self.subblock_query_rows, heads - self.subblock_query_rows, 0
        )
        if lower_query_heads > 0:
            mem_copy(
                q_lo,
                query_heads[
                    0:lower_query_heads,
                    d_column_start : d_column_start + d_column_count,
                ],
                options=copy_options(l2_cache_ctl=_KV_COPY_L2_CACHE_CONTROL),
            )
        if upper_query_heads > 0:
            mem_copy(
                q_hi,
                query_heads[
                    self.subblock_query_rows : self.subblock_query_rows
                    + upper_query_heads,
                    d_column_start : d_column_start + d_column_count,
                ],
                options=copy_options(l2_cache_ctl=_KV_COPY_L2_CACHE_CONTROL),
            )

    @jit
    def bmm1_fanout_q(self, job, kv_lower_rows_l1, kv_upper_rows_l1):
        set_mmad_direction("m")
        q_lo, q_hi = (
            self.storage.q_lower_heads_l1[0],
            self.storage.q_upper_heads_l1[0],
        )
        # Eight D-axis slices are accumulated into one QK score tile.
        for d_tile in range_constexpr(_D_F32_TILES):
            l0_a = self.storage.l0_a_ring.head_next()
            if const_expr(d_tile < _WIDE_F32_TILES):
                if job % 2 == 0:
                    q_lo, q_hi = (
                        self.storage.q_lower_heads_l1[0],
                        self.storage.q_upper_heads_l1[0],
                    )
                else:
                    q_lo, q_hi = (
                        self.storage.q_lower_heads_l1[1],
                        self.storage.q_upper_heads_l1[1],
                    )
                q_column_tile = d_tile
            else:
                q_lo, q_hi = (
                    self.storage.q_lower_heads_l1[2],
                    self.storage.q_upper_heads_l1[2],
                )
                q_column_tile = d_tile - _WIDE_F32_TILES
            mem_copy(
                qk_operand_half(l0_a, self.subblock_query_rows, 0),
                nz_column_tile(q_lo, self.subblock_query_rows, q_column_tile),
            )
            mem_copy(
                qk_operand_half(l0_a, self.subblock_query_rows, 1),
                nz_column_tile(q_hi, self.subblock_query_rows, q_column_tile),
            )
            l0_b = self.storage.l0_b_slots[d_tile % _L0B_SLOTS]
            mem_copy(
                qk_operand_half(l0_b, _KV_ROWS_PER_VECTOR, 0),
                nz_column_tile(kv_lower_rows_l1, _KV_ROWS_PER_VECTOR, d_tile),
            )
            mem_copy(
                qk_operand_half(l0_b, _KV_ROWS_PER_VECTOR, 1),
                nz_column_tile(kv_upper_rows_l1, _KV_ROWS_PER_VECTOR, d_tile),
            )
            matmul(
                nz_view(self.storage.qk_l0c_buffer, self.query_tile_rows, _TILE),
                nz_view(l0_a, self.query_tile_rows, _F32_VL),
                nz_view(l0_b, _TILE, _F32_VL),
                init=d_tile == 0,
                unit_flag=_MMAD_FINAL_UNIT_FLAG if d_tile == _D_F32_TILES - 1 else 2,
            )

    @jit
    def store_s(self, buf_score):
        score = nz_view(self.storage.qk_l0c_buffer, self.query_tile_rows, _TILE)
        mem_copy(
            buf_score,
            score.partition(dim=0),
            options=copy_options(unit_flag=_MMAD_FINAL_UNIT_FLAG),
        )

    @jit
    def compute_pv_fanout(
        self,
        tile,
        kv_lower_rows_l1,
        kv_upper_rows_l1,
        p_l1,
        buf_output_accumulator,
        buf_output_partial,
        buf_output_accumulator_hi,
        buf_output_partial_hi,
    ):
        set_mmad_direction("n")
        l0_a = self.storage.l0_a_ring.head_next()
        for d_tile in range_constexpr(_D_F32_TILES):
            l0_b = self.storage.l0_b_slots[d_tile % _L0B_SLOTS]
            mem_copy(
                pv_operand_half(l0_b, 0),
                nz_column_tile(kv_lower_rows_l1, _KV_ROWS_PER_VECTOR, d_tile).transpose(
                    0, 1
                ),
            )
            mem_copy(
                pv_operand_half(l0_b, 1),
                nz_column_tile(kv_upper_rows_l1, _KV_ROWS_PER_VECTOR, d_tile).transpose(
                    0, 1
                ),
            )
            if const_expr(d_tile == 0):
                mem_copy(
                    nz_view(l0_a, self.query_tile_rows, _TILE),
                    nz_view(p_l1, self.query_tile_rows, _TILE),
                )
            pv_partial_l0c = nz_view(
                self.storage.pv_l0c_buffers[d_tile], self.query_tile_rows, _F32_VL
            )
            matmul(
                pv_partial_l0c,
                nz_view(l0_a, self.query_tile_rows, _TILE),
                nz_view(l0_b, _F32_VL, _TILE),
                init=True,
                unit_flag=0,
            )
            if const_expr(d_tile >= _D_F32_TILES // 2):
                acc_target, pv_target = (
                    buf_output_accumulator_hi,
                    buf_output_partial_hi,
                )
            else:
                acc_target, pv_target = (buf_output_accumulator, buf_output_partial)
            if tile == 0:
                target = tile_slice(
                    acc_target,
                    (self.subblock_query_rows, _F32_VL),
                    (0, d_tile % _L0B_SLOTS),
                )
                mem_copy(
                    target,
                    pv_partial_l0c.partition(dim=0),
                    options=copy_options(unit_flag=0),
                )
            else:
                target = tile_slice(
                    pv_target,
                    (self.subblock_query_rows, _F32_VL),
                    (0, d_tile % _L0B_SLOTS),
                )
                mem_copy(
                    target,
                    pv_partial_l0c.partition(dim=0),
                    options=copy_options(unit_flag=0),
                )


@datastruct
class HighPerfSmlaConstInfo:
    # Runtime scalar parameters passed to the kernel.
    sq: dtypes.int32
    ori_skv: dtypes.int32
    cmp_skv: dtypes.int32
    cmp_topk: dtypes.int32
    cmp_has_length: dtypes.int32
    cmp_i_stride: dtypes.int32
    heads: dtypes.int32
    ori_topk: dtypes.int32
    ori_win_left: dtypes.int32
    ori_win_right: dtypes.int32
    ori_has_length: dtypes.int32
    ori_i_stride: dtypes.int32
    num_sm: dtypes.int32
    scale: dtypes.float32


@jit
def ori_capacity_limit(constinfo):
    ori_capacity = dtypes.int64(constinfo.ori_topk)
    if constinfo.ori_win_left >= 0:
        ori_window_length = constinfo.ori_win_left + 1 + constinfo.ori_win_right
        ori_capacity = scalar_select(
            ori_capacity < ori_window_length, ori_capacity, ori_window_length
        )
    return ori_capacity


@jit
def query_length(ori_lengths, cmp_lengths, constinfo, query_row_idx, ori_capacity):
    ori_requested = ori_capacity
    if constinfo.ori_has_length != 0:
        ori_requested = ori_lengths[query_row_idx]
    ori_length = scalar_select(
        ori_requested > 0,
        scalar_select(ori_requested < ori_capacity, ori_requested, ori_capacity),
        0,
    )
    requested_cmp = constinfo.cmp_topk
    if constinfo.cmp_has_length != 0:
        requested_cmp = cmp_lengths[query_row_idx]
    cmp_length = scalar_select(
        requested_cmp > 0,
        scalar_select(
            requested_cmp < constinfo.cmp_topk, requested_cmp, constinfo.cmp_topk
        ),
        0,
    )
    ori_tile_count = (ori_length + _TILE - 1) // _TILE
    kv_tile_count = ori_tile_count + (cmp_length + _TILE - 1) // _TILE
    ori_tile_count = scalar_select(kv_tile_count > 0, ori_tile_count, 1)
    kv_tile_count = scalar_select(kv_tile_count > 0, kv_tile_count, 1)
    return (ori_length, cmp_length, ori_tile_count, kv_tile_count)


@datastruct
class PipelineRingState:
    clock: CArray[dtypes.int64, 1]
    metadata: CArray[dtypes.int64, _PIPELINE_METADATA_SIZE]


class SlotMajorDelayLineGroup:
    def __init__(self):
        self.state = PipelineRingState(
            clock=[0], metadata=[0] * _PIPELINE_METADATA_SIZE
        )
        self.fields = (
            "job",
            "query_row_idx",
            "tile",
            "length",
            "kv_tile_count",
            "prior_query_tile_count",
            "local_tile",
            "kv_len",
            "is_cmp",
        )

    def push(self, **kwargs):
        metadata_slot_offset = (
            self.state.clock[0] % _PIPELINE_SLOTS * _PIPELINE_FIELDS_PER_SLOT
        )
        for i, field in enumerate(self.fields):
            self.state.metadata[metadata_slot_offset + i] = dtypes.int64(kwargs[field])

    def tap(self, lag):
        metadata_slot_offset = (
            (self.state.clock[0] + _PIPELINE_SLOTS - lag)
            % _PIPELINE_SLOTS
            * _PIPELINE_FIELDS_PER_SLOT
        )
        return SimpleNamespace(
            **{
                field: self.state.metadata[metadata_slot_offset + i]
                for i, field in enumerate(self.fields)
            }
        )

    def advance(self):
        self.state.clock[0] = (self.state.clock[0] + 1) % _PIPELINE_SLOTS


@kernel
class HighPerfSparseFlashMlaKernel:
    # Coordinates Cube and Vector stages.

    def __init__(self, query_tile_rows, has_sink, return_softmax_lse):
        self.storage = HighPerfSmlaStorage(query_tile_rows, has_sink)
        self.matmul = HighPerfSmlaMatmul(self.storage)
        self.vec0 = HighPerfSmlaVec0Indices(self.storage)
        self.vector = HighPerfSmlaVector(self.storage, has_sink, return_softmax_lse)
        self.return_softmax_lse = return_softmax_lse
        self.has_sink = has_sink
        self.delay = SlotMajorDelayLineGroup()

    @jit
    def stage_softmax(
        self,
        lse,
        constinfo,
        job,
        query_row_idx,
        tile,
        local_tile,
        kv_len,
        kv_tile_count,
        heads,
        buf_score,
        buf_p,
        p_l1,
        length,
    ):
        valid_segment_kv_rows = scalar_select(length < kv_len, length, kv_len)
        remaining_kv_rows = valid_segment_kv_rows - local_tile * _TILE
        valid_tile_kv_rows = scalar_select(
            remaining_kv_rows > 0,
            scalar_select(remaining_kv_rows < _TILE, remaining_kv_rows, _TILE),
            0,
        )
        if valid_tile_kv_rows == _TILE:
            if tile == 0:
                self.vector.softmax_rest_full(
                    job,
                    tile,
                    constinfo.scale,
                    True,
                    True,
                    buf_score,
                    buf_p,
                    valid_tile_kv_rows,
                )
            else:
                self.vector.softmax_rest_full(
                    job,
                    tile,
                    constinfo.scale,
                    False,
                    True,
                    buf_score,
                    buf_p,
                    valid_tile_kv_rows,
                )
        elif tile == 0:
            self.vector.softmax_rest_full(
                job,
                tile,
                constinfo.scale,
                True,
                False,
                buf_score,
                buf_p,
                valid_tile_kv_rows,
            )
        else:
            self.vector.softmax_rest_full(
                job,
                tile,
                constinfo.scale,
                False,
                False,
                buf_score,
                buf_p,
                valid_tile_kv_rows,
            )
        source = padded_nz_columns(
            buf_p,
            _TILE,
            self.storage.subblock_query_rows,
            self.storage.subblock_query_rows + 1,
        )
        target = tile_slice(
            p_l1.view(_TILE_NZ_TILES, self.storage.query_tile_rows * _NZ_ROW),
            (_TILE_NZ_TILES, self.storage.subblock_query_rows * _NZ_ROW),
            (0, get_subblock_id()),
        )
        mem_copy(target, source)
        if tile + 1 == kv_tile_count:
            self.vector.finalize_softmax_state(job, constinfo.scale)
            if const_expr(self.return_softmax_lse):
                mem_copy(
                    lse[
                        query_row_idx * constinfo.heads
                        + get_subblock_id() * self.storage.subblock_query_rows : (
                            query_row_idx * constinfo.heads
                            + get_subblock_id() * self.storage.subblock_query_rows
                        )
                        + heads,
                    ],
                    self.storage.maximum[
                        job % 2 * _F32_VL : (job % 2 * _F32_VL) + heads,
                    ],
                )

    @jit
    def stage_update(
        self,
        out,
        constinfo,
        job,
        query_row_idx,
        tile,
        half,
        kv_tile_count,
        heads,
        buf_output_accumulator,
        buf_output_partial,
        buf_out,
    ):
        last_loop = tile + 1 == kv_tile_count
        if tile == 0:
            # VF first-tile store: read and write the first FP32 lane of the PV accumulator.
            with vf(mode="simd"):
                first_element, _ = rr.update_mask(1, elem_bits=_F32_BITS)
                value = rr.vload(buf_output_accumulator, 0)
                rr.vstore(buf_output_accumulator, 0, value, first_element)
        if tile == 0:
            if last_loop:
                self.vector.update_o(
                    job,
                    tile,
                    True,
                    True,
                    buf_output_accumulator,
                    buf_output_partial,
                    buf_out,
                )
        elif last_loop:
            self.vector.update_o(
                job,
                tile,
                False,
                True,
                buf_output_accumulator,
                buf_output_partial,
                buf_out,
            )
        else:
            self.vector.update_o(
                job,
                tile,
                False,
                False,
                buf_output_accumulator,
                buf_output_partial,
                buf_out,
            )
        if last_loop:
            if heads > 0:
                mem_copy(
                    out[dtypes.int64(query_row_idx) * constinfo.heads * _D :,].view(
                        constinfo.heads, _D
                    )[
                        get_subblock_id()
                        * self.storage.subblock_query_rows : get_subblock_id()
                        * self.storage.subblock_query_rows
                        + heads,
                        half * _RESULT_HALF_COLS : (half + 1) * _RESULT_HALF_COLS,
                    ],
                    buf_out.view(self.storage.subblock_query_rows, _OUT_UB_ROW_ELEMS)[
                        0:heads, 0:_RESULT_HALF_COLS
                    ],
                    options=copy_options(l2_cache_ctl=_KV_COPY_L2_CACHE_CONTROL),
                )

    @jit
    def run_vector_pipeline_step(
        self,
        ori_kv,
        cmp_kv,
        ori_indices,
        cmp_indices,
        out,
        lse,
        constinfo,
        tick,
        job,
        query_row_idx,
        tile,
        ori_length,
        cmp_length,
        ori_tile_count,
        kv_tile_count,
        prior_query_tile_count,
        heads,
        kv_lower_rows_l1_write,
        kv_upper_rows_l1_write,
    ):
        is_cmp = tile >= ori_tile_count
        local_tile = scalar_select(is_cmp, tile - ori_tile_count, tile)
        valid_len = scalar_select(is_cmp, cmp_length, ori_length)
        kv_len = scalar_select(is_cmp, constinfo.cmp_skv, constinfo.ori_skv)
        self.delay.push(
            job=job,
            query_row_idx=query_row_idx,
            tile=tile,
            length=valid_len,
            kv_tile_count=kv_tile_count,
            prior_query_tile_count=prior_query_tile_count,
            local_tile=local_tile,
            kv_len=kv_len,
            is_cmp=is_cmp,
        )
        score_read = self.storage.next_score(False)
        score_p = self.storage.p_out
        p_l1_write = self.storage.cross_channels["p_l1"].head_next()
        (
            output_accumulator_lo,
            output_partial_lo,
            output_accumulator_hi,
            output_partial_hi,
        ) = self.storage.next_result(
            False,
        )
        output_bf16_lo = reinterpret(
            output_partial_lo,
            dtypes.bfloat16,
            (self.storage.subblock_query_rows * _OUT_UB_ROW_ELEMS,),
        )
        output_bf16_hi = reinterpret(
            output_partial_hi,
            dtypes.bfloat16,
            (self.storage.subblock_query_rows * _OUT_UB_ROW_ELEMS,),
        )
        if tile >= 0:
            if is_cmp:
                self.vec0.load_idx_tile(
                    cmp_indices,
                    query_row_idx,
                    local_tile,
                    valid_len,
                    constinfo.cmp_i_stride,
                    prior_query_tile_count,
                    tile,
                )
            else:
                self.vec0.load_idx_tile(
                    ori_indices,
                    query_row_idx,
                    local_tile,
                    valid_len,
                    constinfo.ori_i_stride,
                    prior_query_tile_count,
                    tile,
                )
        gather = self.delay.tap(_STAGE_GATHER)
        softmax = self.delay.tap(_STAGE_SOFTMAX)
        update = self.delay.tap(_STAGE_UPDATE)
        if tick >= _STAGE_GATHER and gather.tile >= 0:
            if gather.is_cmp:
                self.gather_convert_copy(
                    cmp_kv,
                    gather.prior_query_tile_count + gather.tile,
                    0,
                    gather.kv_len,
                    gather.length,
                    gather.local_tile,
                    kv_lower_rows_l1_write,
                    kv_upper_rows_l1_write,
                )
            else:
                self.gather_convert_copy(
                    ori_kv,
                    gather.prior_query_tile_count + gather.tile,
                    0,
                    gather.kv_len,
                    gather.length,
                    gather.local_tile,
                    kv_lower_rows_l1_write,
                    kv_upper_rows_l1_write,
                )
        if tick >= _STAGE_SOFTMAX and softmax.tile >= 0:
            self.stage_softmax(
                lse,
                constinfo,
                softmax.job,
                softmax.query_row_idx,
                softmax.tile,
                softmax.local_tile,
                softmax.kv_len,
                softmax.kv_tile_count,
                heads,
                score_read,
                score_p,
                p_l1_write,
                softmax.length,
            )
        if tick >= _STAGE_GATHER and gather.tile >= 0:
            if gather.is_cmp:
                self.gather_convert_copy(
                    cmp_kv,
                    gather.prior_query_tile_count + gather.tile,
                    1,
                    gather.kv_len,
                    gather.length,
                    gather.local_tile,
                    kv_lower_rows_l1_write,
                    kv_upper_rows_l1_write,
                )
            else:
                self.gather_convert_copy(
                    ori_kv,
                    gather.prior_query_tile_count + gather.tile,
                    1,
                    gather.kv_len,
                    gather.length,
                    gather.local_tile,
                    kv_lower_rows_l1_write,
                    kv_upper_rows_l1_write,
                )
        if tick >= _STAGE_UPDATE and update.tile >= 0:
            self.stage_update(
                out,
                constinfo,
                update.job,
                update.query_row_idx,
                update.tile,
                0,
                update.kv_tile_count,
                heads,
                output_accumulator_lo,
                output_partial_lo,
                output_bf16_lo,
            )
            self.stage_update(
                out,
                constinfo,
                update.job,
                update.query_row_idx,
                update.tile,
                1,
                update.kv_tile_count,
                heads,
                output_accumulator_hi,
                output_partial_hi,
                output_bf16_hi,
            )
        if tick >= _STAGE_GATHER and gather.tile >= 0:
            kv_lower_rows_l1_write = self.storage.kv_channels[0].head_next()
            kv_upper_rows_l1_write = self.storage.kv_channels[1].head_next()
        self.delay.advance()
        return (kv_lower_rows_l1_write, kv_upper_rows_l1_write)

    @jit
    def gather_convert_copy(
        self,
        kv,
        global_tile,
        half,
        kv_len,
        length,
        local_tile,
        kv_lower_rows_l1_write,
        kv_upper_rows_l1_write,
    ):
        for chunk in range_constexpr(_GATHER_CHUNKS):
            self.vec0.gather_batch(
                kv, global_tile, half, kv_len, chunk, length, local_tile
            )
        for chunk in range_constexpr(_GATHER_CHUNKS):
            self.vec0.restage_nz_fragment(chunk)
            source = padded_nz_columns(
                self.storage.nz_fragments[chunk], _D, _GATHER_TILE_ROWS, _NZ_PADDED_ROWS
            )
            row_tile = half * _GATHER_CHUNKS + chunk
            if get_subblock_id() == 0:
                target = tile_slice(
                    kv_lower_rows_l1_write.view(
                        _D // _NZ_ROW, _KV_ROWS_PER_VECTOR * _NZ_ROW
                    ),
                    (_D // _NZ_ROW, _GATHER_TILE_ROWS * _NZ_ROW),
                    (0, row_tile),
                )
                mem_copy(target, source)
            else:
                target = tile_slice(
                    kv_upper_rows_l1_write.view(
                        _D // _NZ_ROW, _KV_ROWS_PER_VECTOR * _NZ_ROW
                    ),
                    (_D // _NZ_ROW, _GATHER_TILE_ROWS * _NZ_ROW),
                    (0, row_tile),
                )
                mem_copy(target, source)

    @jit
    def pipeline_vector(
        self,
        ori_kv,
        cmp_kv,
        ori_indices,
        cmp_indices,
        ori_lengths,
        cmp_lengths,
        sink,
        out,
        lse,
        constinfo,
    ):
        ori_capacity = ori_capacity_limit(constinfo)
        head_start = get_subblock_id() * self.storage.subblock_query_rows
        heads = scalar_select(
            constinfo.heads > head_start,
            scalar_select(
                constinfo.heads - head_start < self.storage.subblock_query_rows,
                constinfo.heads - head_start,
                self.storage.subblock_query_rows,
            ),
            0,
        )
        if const_expr(self.has_sink):
            if heads > 0:
                mem_copy(
                    self.storage.sink[0:heads,], sink[head_start : head_start + heads,]
                )
        # Zero only the per-subblock gather storage; no full-core barrier is needed.
        self.vec0.initialize()
        job = dtypes.int64(0)
        # Core c handles query rows c, c + num_sm, c + 2 * num_sm, ...
        query_row_idx = dtypes.int64(get_block_idx())
        prior_query_tile_count = dtypes.int64(0)
        tick = dtypes.int64(0)
        kv_lower_rows_l1_write = self.storage.kv_channels[0].head_next()
        kv_upper_rows_l1_write = self.storage.kv_channels[1].head_next()
        # Subtract the current query row to count the work remaining on this core.
        # The extra iteration drains pending Vector stages after the last query.
        for _ in range(
            (constinfo.sq + constinfo.num_sm - 1 - query_row_idx) // constinfo.num_sm
            + 1
        ):
            draining = query_row_idx >= constinfo.sq
            ori_length = dtypes.int64(0)
            cmp_length = dtypes.int64(0)
            ori_tile_count = dtypes.int64(0)
            kv_tile_count = dtypes.int64(0)
            # Drain the delay line after the final query has entered the pipeline.
            steps = dtypes.int64(_PIPELINE_SLOTS)
            if not draining:
                ori_length, cmp_length, ori_tile_count, kv_tile_count = query_length(
                    ori_lengths, cmp_lengths, constinfo, query_row_idx, ori_capacity
                )
                steps = kv_tile_count
            for step in range(steps):
                tile = scalar_select(draining, -1, step)
                kv_lower_rows_l1_write, kv_upper_rows_l1_write = (
                    self.run_vector_pipeline_step(
                        ori_kv,
                        cmp_kv,
                        ori_indices,
                        cmp_indices,
                        out,
                        lse,
                        constinfo,
                        tick,
                        job,
                        query_row_idx,
                        tile,
                        ori_length,
                        cmp_length,
                        ori_tile_count,
                        kv_tile_count,
                        prior_query_tile_count,
                        heads,
                        kv_lower_rows_l1_write,
                        kv_upper_rows_l1_write,
                    )
                )
                tick = tick + 1
            prior_query_tile_count = prior_query_tile_count + kv_tile_count
            query_row_idx = query_row_idx + constinfo.num_sm
            job = job + 1

    @jit
    def matmul_qk(self, job, kv_lower_rows_l1, kv_upper_rows_l1):
        self.matmul.bmm1_fanout_q(job, kv_lower_rows_l1, kv_upper_rows_l1)
        self.matmul.store_s(self.storage.next_score(True))

    @jit
    def cube_pv(self, tile, kv_lower_rows_l1, kv_upper_rows_l1):
        p_l1 = self.storage.cross_channels["p_l1"].tail_next()
        output_accumulator, output_partial, output_accumulator_hi, output_partial_hi = (
            self.storage.next_result(True)
        )
        self.matmul.compute_pv_fanout(
            tile,
            kv_lower_rows_l1,
            kv_upper_rows_l1,
            p_l1,
            output_accumulator,
            output_partial,
            output_accumulator_hi,
            output_partial_hi,
        )

    @jit
    def pipeline_cube(self, q, ori_lengths, cmp_lengths, constinfo):
        ori_capacity = ori_capacity_limit(constinfo)
        # Core c handles query rows c, c + num_sm, c + 2 * num_sm, ...
        query_row_idx = dtypes.int64(get_block_idx())
        job = dtypes.int64(0)
        ori_length, cmp_length, ori_tile_count, kv_tile_count = query_length(
            ori_lengths, cmp_lengths, constinfo, query_row_idx, ori_capacity
        )
        self.matmul.load_q_wide(q, query_row_idx, job, constinfo.heads, False)
        self.matmul.load_q_wide(q, query_row_idx, job, constinfo.heads, True)
        # The two L1 buffers hold the lower and upper KV-row halves of this tile.
        kv_lower_rows_l1, kv_upper_rows_l1 = self.storage.next_kv(False)
        self.matmul_qk(job, kv_lower_rows_l1, kv_upper_rows_l1)
        # Subtract the current query row to count the work remaining on this core.
        for _ in range(
            (constinfo.sq + constinfo.num_sm - 1 - query_row_idx) // constinfo.num_sm
        ):
            next_query_row_idx = query_row_idx + constinfo.num_sm
            next_kv_tile_count = dtypes.int64(0)
            if next_query_row_idx < constinfo.sq:
                (
                    next_ori_length,
                    next_cmp_length,
                    next_ori_tile_count,
                    next_kv_tile_count,
                ) = query_length(
                    ori_lengths,
                    cmp_lengths,
                    constinfo,
                    next_query_row_idx,
                    ori_capacity,
                )
                # Prefetch the next query's first three D tiles while this query runs.
                self.matmul.load_q_wide(
                    q, next_query_row_idx, job + 1, constinfo.heads, False
                )
            for tile in range(kv_tile_count - 1):
                previous_kv_lower_rows_l1, previous_kv_upper_rows_l1 = (
                    kv_lower_rows_l1,
                    kv_upper_rows_l1,
                )
                kv_lower_rows_l1, kv_upper_rows_l1 = self.storage.next_kv(False)
                self.matmul_qk(job, kv_lower_rows_l1, kv_upper_rows_l1)
                self.cube_pv(tile, previous_kv_lower_rows_l1, previous_kv_upper_rows_l1)
            previous_kv_lower_rows_l1, previous_kv_upper_rows_l1 = (
                kv_lower_rows_l1,
                kv_upper_rows_l1,
            )
            if next_query_row_idx < constinfo.sq:
                self.matmul.load_q_wide(
                    q, next_query_row_idx, job + 1, constinfo.heads, True
                )
                kv_lower_rows_l1, kv_upper_rows_l1 = self.storage.next_kv(False)
                self.matmul_qk(job + 1, kv_lower_rows_l1, kv_upper_rows_l1)
            self.cube_pv(
                kv_tile_count - 1, previous_kv_lower_rows_l1, previous_kv_upper_rows_l1
            )
            kv_tile_count = next_kv_tile_count
            query_row_idx = next_query_row_idx
            job = job + 1

    def __call__(
        self,
        q: Tensor,
        ori_kv: Tensor,
        cmp_kv: Tensor,
        ori_indices: Tensor,
        cmp_indices: Tensor,
        ori_lengths: Tensor,
        cmp_lengths: Tensor,
        sink: Tensor,
        out: Tensor,
        lse: Tensor,
        constinfo: HighPerfSmlaConstInfo,
    ):
        if is_aic():
            self.pipeline_cube(q, ori_lengths, cmp_lengths, constinfo)
        else:
            self.pipeline_vector(
                ori_kv,
                cmp_kv,
                ori_indices,
                cmp_indices,
                ori_lengths,
                cmp_lengths,
                sink,
                out,
                lse,
                constinfo,
            )


class HighPerfSparseFlashMlaLauncher:
    def __init__(self, query_tile_rows, has_sink, return_softmax_lse):
        self.query_tile_rows, self.has_sink = (query_tile_rows, has_sink)
        self.return_softmax_lse = return_softmax_lse

    @host
    def run(
        self,
        q: Tensor,
        ori_kv: Tensor,
        cmp_kv: Tensor,
        ori_indices: Tensor,
        cmp_indices: Tensor,
        ori_lengths: Tensor,
        cmp_lengths: Tensor,
        sink: Tensor,
        out: Tensor,
        lse: Tensor,
        constinfo: HighPerfSmlaConstInfo,
        blocks: dtypes.int32,
    ):
        HighPerfSparseFlashMlaKernel(
            self.query_tile_rows, self.has_sink, self.return_softmax_lse
        )[blocks](
            q,
            ori_kv,
            cmp_kv,
            ori_indices,
            cmp_indices,
            ori_lengths,
            cmp_lengths,
            sink,
            out,
            lse,
            constinfo,
        )


def _build_identity(query_tile_rows, has_sink, return_softmax_lse):
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
        "query_tile_rows": query_tile_rows,
        "has_sink": bool(has_sink),
        "return_softmax_lse": bool(return_softmax_lse),
        "compiler_flags": list(_COMPILER_FLAGS),
        "architecture": "dav-3510",
        "compiler": str(compiler),
        "compiler_size": compiler_stat.st_size,
        "compiler_mtime_ns": compiler_stat.st_mtime_ns,
        "cann_home": os.environ.get("ASCEND_HOME_PATH"),
        "distributions": distributions,
    }


@functools.lru_cache(maxsize=_COMPILED_VARIANT_CACHE_SIZE)
def high_perf_compiled_hp(query_tile_rows, has_sink, return_softmax_lse=False):
    tensor_types = (
        ("q_storage", dtypes.bfloat16),
        ("ori_kv_storage", dtypes.bfloat16),
        ("cmp_storage", dtypes.bfloat16),
        ("ori_index_storage", dtypes.int32),
        ("cmp_index_storage", dtypes.int32),
        ("ori_length_storage", dtypes.int32),
        ("cmp_length_storage", dtypes.int32),
        ("sink_storage", dtypes.float32),
        ("out_storage", dtypes.bfloat16),
        ("lse_storage", dtypes.float32),
    )
    specs = [
        TensorSpec((Dim(name, min=0 if name == "lse_storage" else 1),), dtype)
        for name, dtype in tensor_types
    ]

    def compile_kernel():
        return cannbotdsl.compile(
            HighPerfSparseFlashMlaLauncher(
                query_tile_rows, has_sink, return_softmax_lse
            ).run,
            *specs,
            HighPerfSmlaConstInfo,
            dtypes.int32,
        )

    return _load_or_compile(
        query_tile_rows, has_sink, return_softmax_lse, compile_kernel
    )


def _load_or_compile(query_tile_rows, has_sink, return_softmax_lse, compile_kernel):
    identity = _build_identity(query_tile_rows, has_sink, return_softmax_lse)
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:_BUILD_HASH_PREFIX_LENGTH]
    build_root = Path(__file__).parent
    build_dir = (
        build_root
        / f"build_h{query_tile_rows}_sink{int(has_sink)}_lse{int(return_softmax_lse)}"
        / fingerprint
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    artifact = build_dir / "kernel.so"
    manifest = build_dir / "build_identity.json"
    # Serialize publication across processes and keep the .so/sidecar pair intact.
    with _COMPILATION_LOCK, (build_dir / "compile.lock").open("a") as compile_lock:
        fcntl.flock(compile_lock, fcntl.LOCK_EX)
        if artifact.is_file() and manifest.is_file():
            try:
                if json.loads(manifest.read_text()) == identity:
                    result = load(str(artifact))
                    return result
            except (OSError, ValueError):
                pass  # An invalid cache entry is rebuilt under the same lock.
        with _compilation_environment(build_dir):
            result = compile_kernel()
            result.save(str(artifact))
        with tempfile.NamedTemporaryFile(
            mode="w", dir=build_dir, suffix=".json", delete=False
        ) as output:
            json.dump(identity, output, indent=2)
            temporary_manifest = Path(output.name)
        temporary_manifest.replace(manifest)
        return result


def _require(condition, message):
    if not condition:
        raise ValueError(f"SMLA high-performance: {message}")


def _validate_tensor(tensor, dtype, device, name, *, contiguous=True):
    _require(isinstance(tensor, torch.Tensor), f"{name} must be a Tensor")
    _require(
        tensor.dtype == dtype and tensor.device == device,
        f"{name} has an invalid dtype or device",
    )
    if contiguous:
        _require(tensor.is_contiguous(), f"{name} must be contiguous")


def _validate_int32_abi(tensors):
    for tensor in tensors:
        _require(
            all(
                0 <= value < _INT32_LIMIT for value in (*tensor.shape, *tensor.stride())
            ),
            "Shape or stride exceeds the int32 ABI",
        )


def _validate_scale(sm_scale):
    _require(
        isinstance(sm_scale, numbers.Real)
        and not isinstance(sm_scale, bool)
        and math.isfinite(sm_scale)
        and sm_scale > 0,
        "Scale must be finite and positive",
    )


def _validate_query(query, d_v):
    _require(isinstance(query, torch.Tensor), "Query must be a Tensor")
    _require(query.ndim == _TND_RANK, "Query must have rank 3")
    _require(query.device.type == "npu", "Query must use an NPU")
    _require(query.dtype == torch.bfloat16, "Query must use BF16")
    _require(
        1 <= query.shape[1] <= _Q_TILE_ROWS and query.shape[2] == d_v == _D,
        f"Query requires 1..{_Q_TILE_ROWS} heads and D={_D}",
    )
    _require(query.is_contiguous(), "Query must be contiguous")
    _validate_int32_abi((query,))


def _validate_optional_vector(tensor, length, dtype, device, name):
    if tensor is None:
        return
    _validate_tensor(tensor, dtype, device, name)
    _require(tuple(tensor.shape) == (length,), f"{name} has an invalid shape")


def _validate_cache(query, kv, indices, topk_length, name):
    _validate_tensor(kv, torch.bfloat16, query.device, f"{name} KV")
    _validate_tensor(indices, torch.int32, query.device, f"{name} indices")
    _require(
        kv.ndim == _TND_RANK and tuple(kv.shape[1:]) == (1, _D),
        f"{name} KV requires rank 3 and one KV head",
    )
    _require(
        indices.ndim == _TND_RANK and tuple(indices.shape[:2]) == (query.shape[0], 1),
        f"{name} indices require rank 3 and matching query count",
    )
    _require(indices.stride(-1) == 1, f"{name} indices K axis must be contiguous")
    _validate_optional_vector(
        topk_length, query.shape[0], torch.int32, query.device, f"{name} topk length"
    )
    _validate_int32_abi((kv, indices))


def _prepare_high_perf_inputs(
    q,
    ori_kv,
    cmp_kv=None,
    ori_indices=None,
    cmp_indices=None,
    sm_scale=1.0,
    d_v=_D,
    attn_sink=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    *,
    num_sm,
    ori_win_left=-1,
    ori_win_right=-1,
    return_softmax_lse=False,
):
    _require(isinstance(return_softmax_lse, bool), "Return LSE must be a bool")
    _validate_query(q, d_v)
    _validate_scale(sm_scale)
    _require(ori_indices is not None, "ORI indices are required")
    has_cmp = cmp_kv is not None
    if not has_cmp:
        _require(
            cmp_indices is None and cmp_topk_length is None,
            "CMP indices and lengths require cmp_kv",
        )
        cmp_kv, cmp_indices = ori_kv, ori_indices
    else:
        _require(cmp_indices is not None, "cmp_kv requires explicit CMP indices")
    _validate_cache(q, ori_kv, ori_indices, ori_topk_length, "ORI")
    _validate_cache(q, cmp_kv, cmp_indices, cmp_topk_length, "CMP")
    query_rows, query_heads, _ = q.shape
    _validate_optional_vector(
        attn_sink, query_heads, torch.float32, q.device, "Attention sink"
    )
    outputs = (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty(
            (query_rows, query_heads) if return_softmax_lse else (0,),
            dtype=torch.float32,
            device=q.device,
        ),
    )
    if query_rows == 0:
        return (), outputs, None
    # Unused descriptors are guarded by Runtime flags in the kernel.
    # They must never alias the unrelated metadata descriptor.
    ori_lengths = (
        ori_topk_length
        if ori_topk_length is not None
        else torch.empty(1, dtype=torch.int32, device=q.device)
    )
    cmp_lengths = (
        cmp_topk_length
        if cmp_topk_length is not None
        else torch.empty(1, dtype=torch.int32, device=q.device)
    )
    sink = (
        attn_sink
        if attn_sink is not None
        else torch.empty(1, dtype=torch.float32, device=q.device)
    )
    constinfo = HighPerfSmlaConstInfo(
        sq=query_rows,
        ori_skv=ori_kv.shape[0],
        cmp_skv=cmp_kv.shape[0] if has_cmp else 0,
        cmp_topk=cmp_indices.shape[2] if has_cmp else 0,
        cmp_has_length=int(cmp_topk_length is not None),
        cmp_i_stride=cmp_indices.stride(0),
        heads=query_heads,
        ori_topk=ori_indices.shape[2],
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        ori_has_length=int(ori_topk_length is not None),
        ori_i_stride=ori_indices.stride(0),
        num_sm=num_sm,
        scale=sm_scale,
    )
    arguments = (
        q.view(-1),
        ori_kv.reshape(-1),
        cmp_kv.reshape(-1),
        ori_indices.view(-1),
        cmp_indices.view(-1),
        ori_lengths,
        cmp_lengths,
        sink,
        *(tensor.reshape(-1) for tensor in outputs),
        constinfo,
        min(query_rows, num_sm),
    )
    capacity = _SMALL_Q_TILE_ROWS if query_heads <= _SMALL_Q_TILE_ROWS else _Q_TILE_ROWS
    return arguments, outputs, (capacity, attn_sink is not None, return_softmax_lse)


def _validate_sparse_layout(query, layout_q, layout_kv):
    _require(isinstance(query, torch.Tensor), "Query must be a Tensor")
    _require(
        layout_q in ("TND", "BSND") and layout_kv == layout_q,
        "Matching TND or BSND layouts are required",
    )
    rank = _TND_RANK if layout_q == "TND" else _BSND_RANK
    _require(
        query.ndim == rank and query.device.type == "npu",
        "Invalid query rank or device",
    )
    _require(
        query.dtype == torch.bfloat16 and query.is_contiguous(),
        "Contiguous BF16 query is required",
    )
    _require(
        query.shape[-1] == _D and 1 <= query.shape[-2] <= _Q_TILE_ROWS,
        f"Query requires D={_D} and 1..{_Q_TILE_ROWS} heads",
    )
    if rank == _BSND_RANK:
        _require(query.shape[0] == 1, "BSND requires batch size 1")
    query_rows = query.numel() // (query.shape[-2] * _D)
    _require(query_rows > 0, "Query must be nonempty")
    prefixes = (
        ((query_rows,), (query_rows, 1))
        if rank == _TND_RANK
        else ((1, query_rows), (1, query_rows, 1))
    )
    return rank, query_rows, prefixes


def _validate_sparse_kv(query, kv, rank, name):
    _validate_tensor(kv, torch.bfloat16, query.device, name)
    _require(
        kv.ndim == rank and tuple(kv.shape[-2:]) == (1, _D) and kv.numel() > 0,
        f"{name} has an invalid shape",
    )
    if rank == _BSND_RANK:
        _require(kv.shape[0] == 1, f"{name} requires batch size 1")


def _validate_sparse_indices(query, indices, prefixes, name):
    _validate_tensor(indices, torch.int32, query.device, name)
    _require(
        indices.ndim >= 2
        and tuple(indices.shape[:-1]) in prefixes
        and indices.shape[-1] > 0,
        f"{name} has an invalid shape",
    )


def _validate_sparse_modes(
    query,
    ori_kv,
    cmp_kv,
    ori_indices,
    cmp_indices,
    ori_topk_length,
    cmp_topk_length,
    prefixes,
    cmp_ratio,
    ori_mask_mode,
    cmp_mask_mode,
    ori_win_left,
    ori_win_right,
    topk_value_mode,
    return_softmax_lse,
    cmp_residual_kv,
):
    scalar_modes = (
        (cmp_ratio, "CMP ratio"),
        (ori_mask_mode, "ORI mask mode"),
        (cmp_mask_mode, "CMP mask mode"),
        (ori_win_left, "ORI left window"),
        (ori_win_right, "ORI right window"),
        (topk_value_mode, "Topk value mode"),
    )
    for value, name in scalar_modes:
        _require(
            isinstance(value, numbers.Integral) and not isinstance(value, bool),
            f"{name} must be an integer",
        )
    _require(
        1 <= cmp_ratio <= _TILE and topk_value_mode == 1,
        "Unsupported CMP ratio or topk mode",
    )
    _require(
        isinstance(return_softmax_lse, bool),
        "Return LSE must be a bool",
    )
    has_cmp = cmp_kv is not None
    if not has_cmp:
        _require(
            cmp_ratio == 1 and cmp_mask_mode == 0 and cmp_residual_kv is None,
            "ORI-only requires CMP ratio 1, mask 0 and no residual",
        )
    elif cmp_residual_kv is not None:
        _validate_tensor(cmp_residual_kv, torch.int32, query.device, "CMP residual KV")
        _require(
            tuple(cmp_residual_kv.shape) == (1,), "CMP residual KV has an invalid shape"
        )
    legacy = (ori_mask_mode, cmp_mask_mode) == (
        _MASK_SLIDING_WINDOW,
        _MASK_CAUSAL if has_cmp else 0,
    )
    if legacy:
        ori_capacity = ori_indices.shape[-1]
        cmp_capacity = cmp_indices.shape[-1] if has_cmp else 0
        ori_rows = ori_kv.numel() // _D
        cmp_rows = cmp_kv.numel() // _D if has_cmp else 0
        query_rows = query.numel() // (query.shape[-2] * _D)
        _require(
            (ori_win_left, ori_win_right) == (_TILE - 1, 0),
            "Legacy mode requires window 127/0",
        )
        _require(
            ori_topk_length is None and cmp_topk_length is None,
            "Legacy topk lengths must be None",
        )
        _require(
            ori_capacity >= _TILE
            and ori_capacity % _TILE == 0
            and cmp_capacity % _TILE == 0,
            "Invalid legacy sparse capacities",
        )
        _require(
            ori_rows - query_rows >= _TILE
            and (
                not has_cmp
                or (
                    (ori_rows - query_rows) // cmp_ratio >= cmp_capacity
                    and min(
                        cmp_rows,
                        max(0, (cmp_rows * cmp_ratio - query_rows + 1) // cmp_ratio),
                    )
                    >= cmp_capacity
                )
            ),
            "Legacy mode requires full causal coverage",
        )
    else:
        _require(
            (ori_mask_mode, cmp_mask_mode) == (0, 0)
            and (ori_win_left, ori_win_right) == (-1, -1),
            "Mask 0/0 or legacy mask 4/(3 with CMP, 0 without CMP) is required",
        )
        for tensor, name in (
            (ori_topk_length, "ORI topk length"),
            (cmp_topk_length, "CMP topk length"),
        ):
            if name == "CMP topk length" and not has_cmp:
                continue
            _validate_tensor(tensor, torch.int32, query.device, name)
            _require(tuple(tensor.shape) in prefixes, f"{name} has an invalid shape")
    return legacy


def _validate_high_perf_call(
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
    query = query_gm
    rank, query_rows, prefixes = _validate_sparse_layout(query, layout_q, layout_kv)
    _validate_scale(softmax_scale)
    _validate_sparse_kv(query, ori_kv, rank, "ORI KV")
    _validate_sparse_indices(query, ori_sparse_indices, prefixes, "ORI sparse indices")
    if cmp_kv is not None:
        _validate_sparse_kv(query, cmp_kv, rank, "CMP KV")
        _validate_sparse_indices(
            query, cmp_sparse_indices, prefixes, "CMP sparse indices"
        )
    else:
        _require(
            all(
                value is None
                for value in (
                    cmp_sparse_indices,
                    cmp_topk_length,
                    cmp_residual_kv,
                    cmp_block_table,
                    cu_seqlens_cmp_kv,
                    seqused_cmp_kv,
                )
            ),
            "CMP descriptors require cmp_kv",
        )
    for descriptor in (
        ori_block_table,
        cmp_block_table,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
    ):
        _require(
            descriptor is None,
            "Page tables and seqused are outside the full-storage specialization",
        )
    if rank == _BSND_RANK:
        sequence_offsets = (cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv)
        _require(
            all(descriptor is None for descriptor in sequence_offsets),
            "Sequence offsets must be None for BSND",
        )
    _validate_tensor(sinks, torch.float32, query.device, "Sinks")
    _require(tuple(sinks.shape) == (query.shape[-2],), "Sinks have an invalid shape")
    legacy = _validate_sparse_modes(
        query,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_topk_length,
        cmp_topk_length,
        prefixes,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        topk_value_mode,
        return_softmax_lse,
        cmp_residual_kv,
    )
    _validate_int32_abi(
        tuple(
            tensor
            for tensor in (
                query,
                ori_kv,
                cmp_kv,
                ori_sparse_indices,
                cmp_sparse_indices,
            )
            if tensor is not None
        )
    )
    num_sm = torch.npu.get_device_properties(query.device).cube_core_num
    return {
        "route": "asc_high_perf",
        "reasons": [],
        "query_rows": query_rows,
        "legacy": legacy,
        "cores": num_sm,
        "use_topk": not legacy,
    }


def high_perf_admission(query_gm, **kwargs):
    """Validate descriptors for the shared dispatcher's route selection."""
    try:
        return _validate_high_perf_call(query_gm, **kwargs)
    except (ValueError, TypeError) as error:
        return {"route": "reject", "reasons": [str(error)]}


def _run_admitted_high_perf(query_gm, args, admission):
    """Adapt the shared SMLA API after a successful admission check."""
    query = query_gm
    query_rows = admission["query_rows"]
    num_sm = admission["cores"]
    cmp_kv = args["cmp_kv"]
    cmp_sparse_indices = args["cmp_sparse_indices"]
    cmp_topk_length = args["cmp_topk_length"]
    ori_kv = args["ori_kv"]
    ori_sparse_indices = args["ori_sparse_indices"]
    ori_topk_length = args["ori_topk_length"]
    return_softmax_lse = args["return_softmax_lse"]
    sinks = args["sinks"]
    softmax_scale = args["softmax_scale"]
    query_3d = query.view(-1, query.shape[-2], _D)
    ori_indices_3d = ori_sparse_indices.view(query_rows, 1, -1)
    cmp_indices_3d = (
        None
        if cmp_sparse_indices is None
        else cmp_sparse_indices.view(query_rows, 1, -1)
    )
    ori_lengths = None if ori_topk_length is None else ori_topk_length.view(-1)
    cmp_lengths = None if cmp_topk_length is None else cmp_topk_length.view(-1)
    arguments, outputs, specialization = _prepare_high_perf_inputs(
        query_3d,
        ori_kv.view(-1, 1, _D),
        None if cmp_kv is None else cmp_kv.view(-1, 1, _D),
        ori_indices_3d,
        cmp_indices_3d,
        softmax_scale,
        attn_sink=sinks,
        ori_topk_length=ori_lengths,
        cmp_topk_length=cmp_lengths,
        num_sm=num_sm,
        ori_win_left=args["ori_win_left"],
        ori_win_right=args["ori_win_right"],
        return_softmax_lse=return_softmax_lse,
    )
    if specialization is not None:
        high_perf_compiled_hp(*specialization)(*arguments)
    out, lse = outputs
    return out.view(query.shape), lse.view(
        query.shape[:-1]
    ) if return_softmax_lse else lse


def high_perf_sparse_flash_mla(
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
    _admission=None,
):
    args = locals().copy()
    args.pop("query_gm")
    # Admission is owned by the shared dispatcher (or by standalone test code).
    admission = args.pop("_admission")
    return _run_admitted_high_perf(query_gm, args, admission)
