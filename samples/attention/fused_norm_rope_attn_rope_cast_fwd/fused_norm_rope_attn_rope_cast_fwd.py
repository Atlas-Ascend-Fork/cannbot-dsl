# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Third-party source notices are retained with the repository reference tests.

"""Fused sparse attention prefill for Ascend NPUs.

Optional Q Norm, Q RoPE and NZ sparse attention use a six-stage cross-query
pipeline with fixed on-chip buffers. S=T=8192,H=64,K in {128,640},positive
scale uses a private FP32 attention workspace and a second pure-AIV kernel
for inverse O RoPE, MXFP8 quantization and packed scale output. Other shapes
use the complete single AIC/AIV MIX kernel.

Each call owns its workspace and outputs on the current NPU stream.
Only compiled programs are cached, using tensor metadata and core quotas.
All host/device implementation dependencies are contained in this file.
"""

import ctypes
import cannbotdsl as dsl
import math
import os
import torch
import torch_npu  # noqa: F401 - registers the torch.npu backend
from cannbotdsl import (
    CArray,
    Constexpr,
    L0A,
    L0B,
    L0C,
    L1,
    MemLoc,
    RegLayout,
    Tensor,
    UB,
    const_expr,
    datastruct,
    dtypes,
    get_block_idx,
    get_subblock_id,
    host,
    is_aic,
    is_aiv,
    jit,
    kernel,
    make_buffer,
    make_ring_buffer,
    copy_options,
    matmul,
    mem_copy,
    range_constexpr,
    reg as rr,
    vf,
)
from cannbotdsl.ops.cube import set_mmad_direction
from cannbotdsl.ops.sync import (
    PIPE,
    cube_get_buf,
    cube_release_buf,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    cube_sync_intra_arrive,
    cube_sync_intra_wait,
    cube_sync_notify,
    cube_sync_wait,
    vec_get_buf,
    vec_release_buf,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
    vec_sync_intra_arrive,
    vec_sync_intra_wait,
    vec_sync_notify,
    vec_sync_wait,
)
from cannbotdsl.tensor import (
    make_layout,
    make_pointer,
    make_tensor,
    reinterpret,
    tile_slice,
    tile_slice_multi,
)
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from types import SimpleNamespace
from typing import Optional, Tuple


# layout: production definitions.

UB_CAPACITY = int(dsl.get_mem_size("ub"))
if UB_CAPACITY <= 0:
    raise RuntimeError("the compilation target does not report a usable UB capacity")

# Fixed index workspace geometry shared by allocation and pipeline scheduling.
KV_TILE_SIZE = 64
INDEX_GROUP_SIZE = 640
INDEX_PAIRS_PER_GROUP = INDEX_GROUP_SIZE // 2
INDEX_TILES_PER_GROUP = INDEX_GROUP_SIZE // KV_TILE_SIZE
INDEX_PAIRS_PER_TILE = KV_TILE_SIZE // 2


@dataclass(frozen=True)
class Region:
    name: str
    offset: int
    size: int

    @property
    def end(self) -> int:
        return self.offset + self.size


def ub_regions(have_sink: bool = False) -> tuple[Region, ...]:
    """Calculate the C ABI offsets, including zero-length array alignment."""
    u16, f32 = ctypes.c_uint16, ctypes.c_float
    i32, i64 = ctypes.c_int32, ctypes.c_int64
    fields = (
        ("gathered_kv", u16, 3 * 32 * 512),
        ("fp8_scales_bf16", u16, 0),
        ("gathered_kv_in_nz", u16, 33 * 512),
        ("output_accum", f32, 32 * 512),
        ("cur_output_frag", f32, 2 * 32 * 128),
        ("p", f32, 32 * 64),
        ("attn_sink", f32, 32 if have_sink else 0),
        ("row_max", f32, 2 * 64),
        ("row_max_for_o", f32, 2 * 64),
        ("row_sum", f32, 2 * 64),
        ("row_denorm", f32, 2 * 64),
        ("row_scales", f32, 2 * 2 * 64),
        ("indices", i32, INDEX_GROUP_SIZE),
        ("global_kv_ptr_offset", i64, INDEX_PAIRS_PER_GROUP),
        ("src_stride_in_gather2", i64, INDEX_PAIRS_PER_GROUP),
        ("index_validity", ctypes.c_uint8, 4 * INDEX_GROUP_SIZE),
        ("gathered_kv_fp4", ctypes.c_uint8, 0),
        ("fp4_scales_bf16", u16, 0),
        ("row_max_max_delta", f32, 4),
    )

    class MemoryPlan(ctypes.Structure):
        _fields_ = [(name, dtype * count) for name, dtype, count in fields]

    regions = tuple(
        Region(name, getattr(MemoryPlan, name).offset, ctypes.sizeof(dtype) * count)
        for name, dtype, count in fields
        if count
    )
    expected = 252048 if have_sink else 251920
    assert ctypes.sizeof(MemoryPlan) == expected
    assert regions[-1].end == expected
    if expected > UB_CAPACITY:
        raise RuntimeError(
            f"UB layout requires {expected} bytes, target provides {UB_CAPACITY} bytes"
        )
    return regions


def ub_offsets(have_sink: bool = False) -> dict[str, int]:
    return {region.name: region.offset for region in ub_regions(have_sink)}


@jit
def store_index_validity(validity, group, start, smaller, larger, kv_count, full):
    one = rr.vdups(1, dtypes.uint32)
    zero = rr.vdups(0, dtypes.uint32)
    small_valid = rr.vselect(one, zero, cond_mask=rr.vlts(smaller, kv_count, mask=full))
    large_valid = rr.vselect(one, zero, cond_mask=rr.vlts(larger, kv_count, mask=full))
    first, second = rr.vinterleave(small_valid, large_valid)
    offset = (dtypes.int64(group) % 4) * INDEX_GROUP_SIZE + start
    rr.vstore_pack(validity, offset, first, full, pack_mode="b32_to_b8")
    rr.vstore_pack(validity, offset + 64, second, full, pack_mode="b32_to_b8")


@jit
def load_index_validity(validity, group, tile):
    offset = (dtypes.int64(group) % 4) * INDEX_GROUP_SIZE + dtypes.int64(
        tile
    ) * KV_TILE_SIZE
    return rr.vload_unpack(validity, offset, unpack_mode="b8_to_b32")


# pipeline_tuning: production definitions.

GATHER_UNSIGNED_ROWS = 8388608
NZ_SCORE_TWO_COPIES = 67108864
NZ_SCORE_DUAL_COPY = 134217728
NZ_SCORE_PATHS = NZ_SCORE_TWO_COPIES | NZ_SCORE_DUAL_COPY
NZ_MASKED_EXP = 268435456
NZ_SUM_FIRST_VALUE = 536870912
NZ_SHARED_FULL_MASK = 1073741824
GATHER_FLAT_INTERVAL = 2147483648
GATHER_BYTE_INTERVAL = 4294967296
SSBUF_EARLY_NOTIFY = 8589934592
NATIVE_INDEX_MASKING = 17179869184
NATIVE_INDEX_FETCH = 34359738368
NZ_MAXIMUM_CURSORS = 68719476736
INDEX_MASKED_SUB = 137438953472
NATIVE_INDEX_PREPROCESS = 274877906944
INDEX_NARROW_LENGTH = 549755813888
KV_LOAD_CURSORS = 1099511627776
INDEX_INTERLEAVE_STORE = 2199023255552
GATHER_U32_ADDRESS = 4398046511104
KV_ROWS_UNROLL2 = 8796093022208
KV_ROWS_UNROLL4 = 17592186044416
KV_ROWS_UNROLL8 = 35184372088832
INDEX_LOAD_DEINTERLEAVE = 70368744177664
CUBE_SIMPLE_CURSOR = 140737488355328
SM_MASK_STRIDED = 1125899906842624
SM_LOCAL_VIEWS = 2251799813685248
KV_LOCAL_CURSORS = 4503599627370496
UNSIGNED_METADATA = 9007199254740992
UNSIGNED_SCHEDULE = 18014398509481984
SM_EARLY_SUM = 72057594037927936
CUBE_UNSIGNED_SCHEDULE = 144115188075855872
METADATA_SKIP_UNUSED_LENGTH = 288230376151711744
NATIVE_KV_PAIR_ROLLED = 1048576
ND_PIPELINE_OPTIMIZATIONS = 2 | 8 | 16 | 2048 | 524288 | 256 | GATHER_UNSIGNED_ROWS
BASELINE_PIPELINE_OPTIMIZATIONS = ND_PIPELINE_OPTIMIZATIONS | NZ_SCORE_DUAL_COPY
PREVIOUS_DEFAULT_PIPELINE_OPTIMIZATIONS = (
    BASELINE_PIPELINE_OPTIMIZATIONS
    | NZ_MASKED_EXP
    | NZ_SUM_FIRST_VALUE
    | NZ_SHARED_FULL_MASK
    | GATHER_BYTE_INTERVAL
)
DEFAULT_PIPELINE_OPTIMIZATIONS = (
    PREVIOUS_DEFAULT_PIPELINE_OPTIMIZATIONS | NATIVE_KV_PAIR_ROLLED
)
K640_DEFAULT_PIPELINE_OPTIMIZATIONS = (
    DEFAULT_PIPELINE_OPTIMIZATIONS | INDEX_MASKED_SUB | INDEX_INTERLEAVE_STORE
)
K640_SINK_DEFAULT_PIPELINE_OPTIMIZATIONS = (
    K640_DEFAULT_PIPELINE_OPTIMIZATIONS | CUBE_SIMPLE_CURSOR
)
K640_NOSINK_DEFAULT_PIPELINE_OPTIMIZATIONS = (
    K640_DEFAULT_PIPELINE_OPTIMIZATIONS | UNSIGNED_METADATA | SM_EARLY_SUM
)
K128_NOSINK_DEFAULT_PIPELINE_OPTIMIZATIONS = (
    DEFAULT_PIPELINE_OPTIMIZATIONS
    | INDEX_LOAD_DEINTERLEAVE
    | CUBE_SIMPLE_CURSOR
    | UNSIGNED_METADATA
)


def default_pipeline_optimizations(k, have_sink=False):
    if k == 640:
        return (
            K640_SINK_DEFAULT_PIPELINE_OPTIMIZATIONS
            if have_sink
            else K640_NOSINK_DEFAULT_PIPELINE_OPTIMIZATIONS
        )
    return (
        DEFAULT_PIPELINE_OPTIMIZATIONS
        if have_sink
        else K128_NOSINK_DEFAULT_PIPELINE_OPTIMIZATIONS
    )


NATIVE_SUM_MADD = 4194304
UINT32_PIPELINE_STATE = 16384
NATIVE_KV_PACK_ORDER = 32768
SSBUF_UINT64 = 131072
FUSED_HISTORY_STATE = 524288
BOUNDED_PIPELINE_STATE = 16777216
GATHER_PAIR_ROLLED = 33554432


# attention: production definitions.


def ub_view(view):
    """Bind an alias of the fixed UB root to its buffer resource."""
    return make_buffer(view)


def fixed_view(root, shape, dtype, address=0, data_format="nz"):
    return make_buffer(
        reinterpret(
            root,
            layout=make_layout(shape),
            dtype=dtype,
            offset=address,
            data_format=data_format,
        )
    )


class Vector:
    """AIV 固定 UB、配对索引、NZ softmax 和输出历史累加。"""

    def __init__(self, have_sink, nz_score=False, manual_gather=False):
        self.nz_score = nz_score
        self.part = get_subblock_id()
        offsets = ub_offsets(have_sink)
        root = UB.view(UB_CAPACITY)
        if manual_gather:
            self.gather_buffers = tuple(
                ub_view(
                    reinterpret(
                        root,
                        dtype=dtypes.bfloat16,
                        layout=make_layout((32, 512)),
                        offset=32768 * slot,
                    )
                )
                for slot in range(3)
            )
        else:
            self.gather = make_ring_buffer(
                tuple(
                    reinterpret(
                        root,
                        dtype=dtypes.bfloat16,
                        layout=make_layout((32, 512)),
                        offset=32768 * slot,
                    )
                    for slot in range(3)
                )
            )
        scratch = ub_view(
            reinterpret(
                root,
                dtype=dtypes.bfloat16,
                layout=make_layout((33, 512)),
                offset=98304,
            )
        )
        self.nz = reinterpret(
            scratch,
            layout=make_layout((32, 512)),
            data_format="nz",
            physical_layout=make_layout((32, 2, 16, 16), stride=(528, 256, 16, 1)),
        )
        self.accum = ub_view(
            reinterpret(
                root,
                dtype=dtypes.float32,
                layout=make_layout((32, 512)),
                offset=132096,
            )
        )
        self.final = reinterpret(
            self.accum,
            dtype=dtypes.bfloat16,
            layout=make_layout((32, 512)),
        )
        # Four logical publications share the original two physical slots.
        # Each view is read once per flush, so automatic read sharing cannot
        # retain a slot across the explicit acknowledgement and its overwrite.
        self.fragments = tuple(
            ub_view(
                reinterpret(
                    root,
                    dtype=dtypes.float32,
                    layout=make_layout((32, 128)),
                    offset=197632 + 16384 * (slot % 2),
                )
            )
            for slot in range(4)
        )
        self.score = ub_view(
            reinterpret(
                root,
                dtype=dtypes.float32,
                layout=make_layout((32, 64)),
                offset=230400,
            )
        )
        self.prob = reinterpret(
            self.score,
            dtype=dtypes.bfloat16,
            layout=make_layout((32, 64)),
        )
        if nz_score:
            self.prob = reinterpret(
                self.score,
                dtype=dtypes.bfloat16,
                layout=make_layout((4096,)),
            )
            # The raw payload has N0=16; canonical UB fp32 NZ has N0=8.
            # The parent preserves the native 512-float N1 DMA pitch.
            self.score_nz = tile_slice(
                reinterpret(self.score, layout=make_layout((64, 64)), data_format="nz"),
                (32, 64),
                (0, 0),
            )
            # Pack BF16 in place with a 512-element gap per N1.
            self.prob_nz = reinterpret(
                self.score,
                dtype=dtypes.bfloat16,
                layout=make_layout((32, 64)),
                data_format="nz",
                physical_layout=make_layout((4, 2, 16, 16), stride=(1024, 256, 16, 1)),
            )
        else:
            prob_storage = ub_view(
                reinterpret(
                    root,
                    dtype=dtypes.bfloat16,
                    layout=make_layout((32, 64)),
                    offset=234496,
                )
            )
            self.prob_nz = reinterpret(
                prob_storage,
                layout=make_layout((32, 64)),
                data_format="nz",
            )
        self.maximum = ub_view(
            reinterpret(
                root,
                dtype=dtypes.float32,
                layout=make_layout((128,)),
                offset=offsets["row_max"],
            )
        )
        self.base = ub_view(
            reinterpret(
                root,
                dtype=dtypes.float32,
                layout=make_layout((128,)),
                offset=offsets["row_max_for_o"],
            )
        )
        self.sum = ub_view(
            reinterpret(
                root,
                dtype=dtypes.float32,
                layout=make_layout((128,)),
                offset=offsets["row_sum"],
            )
        )
        self.denorm = ub_view(
            reinterpret(
                root,
                dtype=dtypes.float32,
                layout=make_layout((128,)),
                offset=offsets["row_denorm"],
            )
        )
        self.scales = ub_view(
            reinterpret(
                root,
                dtype=dtypes.float32,
                layout=make_layout((256,)),
                offset=offsets["row_scales"],
            )
        )
        self.indices = ub_view(
            reinterpret(
                root,
                dtype=dtypes.int32,
                layout=make_layout((INDEX_GROUP_SIZE,)),
                offset=offsets["indices"],
            )
        )
        self.validity = ub_view(
            reinterpret(
                root,
                dtype=dtypes.uint8,
                layout=make_layout((4 * INDEX_GROUP_SIZE,)),
                offset=offsets["index_validity"],
            )
        )
        self.delta = ub_view(
            reinterpret(
                root,
                dtype=dtypes.float32,
                layout=make_layout((4,)),
                offset=offsets["row_max_max_delta"],
            )
        )
        if have_sink:
            self.sink = ub_view(
                reinterpret(
                    root,
                    dtype=dtypes.float32,
                    layout=make_layout((32,)),
                    offset=offsets["attn_sink"],
                )
            )

    @jit
    def initialize(self, gm_sink, have_sink: Constexpr[bool]):
        # Gather preserves skipped pairs in the three rotating slots.
        # Define the slots once, instead of clearing 32 KiB for every block.
        for slot in range_constexpr(3):
            stage = self.gather.next_write()
            with vf():
                full = rr.full_mask(16)
                zero = rr.vdups(0.0, dtypes.bfloat16)
                for start in range(0, 16384, 128):
                    rr.vstore(stage, start, zero, full)
        with vf():
            full = rr.full_mask(32)
            zero = rr.vdups(0.0, dtypes.float32)
            for start in range(0, 16384, 64):
                rr.vstore(self.accum, start, zero, full)
        if const_expr(have_sink):
            mem_copy(self.sink, tile_slice(gm_sink, (32,), (self.part,)))

    @jit
    def prepare(self, gm_indices, gm_sink, query, have_sink: Constexpr[bool]):
        with vf():
            full = rr.full_mask(32)
            negative = rr.vdups(-3.4028234663852886e38, dtypes.float32)
            zero = rr.vdups(0.0, dtypes.float32)
            rr.vstore(self.maximum, 0, negative, full)
            rr.vstore(self.base, 0, negative, full)
            rr.vstore(self.sum, 0, zero, full)
            rr.vstore_first(self.denorm, 0, zero)
        mem_copy(
            tile_slice(self.indices, (gm_indices.shape[1],), (0,)),
            tile_slice(gm_indices, (1, gm_indices.shape[1]), (query, 0)).view(
                (gm_indices.shape[1],)
            ),
        )

    @jit
    def gather_tile(self, kv, tile, valid_len):
        stage = self.gather.next_write()
        with vf():
            full = rr.full_mask(32)
            indices = rr.vload(self.indices, dtypes.int64(tile) * 64)
            ordinal = rr.varange(dtypes.int64(tile) * 64, dtypes.int32)
            prefix = rr.vlts(ordinal, valid_len, mask=full)
            nonnegative = rr.vges(indices, 0, mask=full)
            bounded = rr.vlts(indices, kv.shape[0], mask=full)
            valid = rr.mask_and(
                rr.mask_and(prefix, nonnegative, exec_mask=full),
                bounded,
                exec_mask=full,
            )
            invalid = rr.vdups(kv.shape[0], dtypes.int32)
            indices = rr.vselect(indices, invalid, cond_mask=valid)
            even, odd = rr.vdeinterleave(indices, indices)
            smaller = rr.vmin(even, odd, mask=full)
            larger = rr.vmax(even, odd, mask=full)
            ordered, unused = rr.vinterleave(smaller, larger)
            rr.vstore(self.indices, dtypes.int64(tile) * 64, ordered, full)
        for pair in range(16):
            first = dtypes.int64(
                self.indices[tile * KV_TILE_SIZE + (self.part * 16 + pair) * 2]
            )
            second = dtypes.int64(
                self.indices[tile * KV_TILE_SIZE + (self.part * 16 + pair) * 2 + 1]
            )
            if first < kv.shape[0]:
                if second >= kv.shape[0]:
                    second = first
                mem_copy(
                    tile_slice(stage, (2, 512), (pair, 0)),
                    (
                        tile_slice(kv, (1, 512), (first, 0)),
                        tile_slice(kv, (1, 512), (second, 0)),
                    ),
                    copy_options(l2_cache_ctl=1),
                )
        return stage

    @jit
    def push_kv(self, stage, lo, hi):
        with vf():
            mask = rr.full_mask(16)
            for row in range(32):
                for chunk in range(4):
                    values = rr.vload(stage, row * 512 + chunk * 128)
                    rr.vstore_strided(
                        self.nz,
                        row * 16 + chunk * 128 * 33,
                        values,
                        mask,
                        block_stride=33,
                        repeat_stride=0,
                    )
        if self.part == 0:
            mem_copy(lo, self.nz)
        else:
            mem_copy(hi, self.nz)
        vec_sync_block_arrive(PIPE.MTE3, 10)

    @jit
    def softmax(self, tile, scale):
        if const_expr(self.nz_score):
            self.softmax_nz(tile, scale)
        else:
            self.softmax_nd(tile, scale)

    @jit
    def softmax_nd(self, tile, scale):
        vec_sync_block_wait(PIPE.V, 11)
        with vf():
            full = rr.full_mask(32)
            neg = rr.vdups(-3.4028234663852886e38, dtypes.float32)
            indices = rr.vload(self.indices, dtypes.int64(tile) * 64)
            valid = rr.vlts(indices, self.kv_count, mask=full)
            for row in range(32):
                score = rr.vload(self.score, row * 64)
                masked = rr.vselect(score, neg, cond_mask=valid)
                current = rr.vreduce_max(masked, mask=full)
                previous = rr.vload_broadcast(self.maximum, row * 2)
                current = rr.vmax(current, previous, mask=full)
                rr.vstore_first(self.maximum, row * 2, current)
                rr.vstore_first(self.maximum, row * 2 + 1, current)
            rr.vmem_bar("vst_vld")
            maximum = rr.vload(self.maximum)
            first = rr.veqs(rr.vdups(dtypes.int64(tile), dtypes.int32), 0, mask=full)
            old_base = rr.vload(self.base)
            difference = rr.vsub(maximum, old_base, mask=full)
            flag = rr.vdup(rr.vreduce_max(difference, mask=full), mask=full)
            flag = rr.vselect(rr.vdups(0.0, dtypes.float32), flag, cond_mask=first)
            update = rr.mask_or(
                first, rr.vgts(flag, 6.0 / scale, mask=full), exec_mask=full
            )
            new_base = rr.vselect(maximum, old_base, cond_mask=update)
            rr.vstore(self.base, 0, new_base, full)
            old_scaled = rr.vmuls(old_base, scale, mask=full)
            new_scaled = rr.vmuls(new_base, scale, mask=full)
            factors = rr.vexp_sub(old_scaled, new_scaled, mask=full)
            factors = rr.vselect(
                rr.vdups(1.0, dtypes.float32), factors, cond_mask=first
            )
            rr.vstore(self.scales, 0, factors, full)
            rr.vstore_first(self.delta, 0, flag)
            rr.vmem_bar("vst_vld")
            zero = rr.vdups(0.0, dtypes.float32)
            for row in range(32):
                # 概率按 ND 原地打包，后续 MTE3 转 NZ；写地址不覆盖尚未读的 score 行。
                score = rr.vmuls(rr.vload(self.score, row * 64), scale, mask=full)
                base = rr.vmuls(
                    rr.vload_broadcast(self.base, row * 2), scale, mask=full
                )
                probability = rr.vexp_sub(score, base, mask=full)
                probability = rr.vselect(probability, zero, cond_mask=valid)
                total = rr.vreduce_sum(probability, mask=full)
                previous_sum = rr.vload_broadcast(self.sum, row * 2)
                factor = rr.vload_broadcast(self.scales, row * 2)
                total = rr.vadd(
                    rr.vmul(previous_sum, factor, mask=full), total, mask=full
                )
                rr.vstore_first(self.sum, row * 2, total)
                rr.vstore_first(self.sum, row * 2 + 1, total)
                packed = rr.vcast(probability, dtypes.bfloat16, mask=full)
                rr.vstore_pack(
                    self.prob, row * 64, packed, full, pack_mode="b32_to_b16"
                )
            rr.vmem_bar("vst_vld")
            bfmask = rr.create_mask("vl64", 16)
            for row in range(32):
                probability = rr.vload(self.prob, row * 64)
                rr.vstore_strided(
                    self.prob_nz,
                    row * 16,
                    probability,
                    bfmask,
                    block_stride=32,
                    repeat_stride=0,
                )

    @jit
    def softmax_nz(self, tile, scale):
        vec_sync_block_wait(PIPE.V, 11)
        with vf():
            full = rr.full_mask(32)
            eight = rr.create_mask("vl8", 32)
            bfmask = rr.full_mask(16)
            neg = rr.vdups(-3.4028234663852886e38, dtypes.float32)
            zero = rr.vdups(0.0, dtypes.float32)
            indices = rr.vload(self.indices, dtypes.int64(tile) * 64)
            lanes = rr.varange(0, dtypes.uint32)
            columns = rr.vbitwise_and(lanes, rr.vdups(15, dtypes.uint32), mask=full)
            mask0 = rr.vlts(rr.vgather_reg(indices, columns), self.kv_count, mask=full)
            mask1 = rr.vlts(
                rr.vgather_reg(indices, rr.vadds(columns, 16, mask=full)),
                self.kv_count,
                mask=full,
            )
            mask2 = rr.vlts(
                rr.vgather_reg(indices, rr.vadds(columns, 32, mask=full)),
                self.kv_count,
                mask=full,
            )
            mask3 = rr.vlts(
                rr.vgather_reg(indices, rr.vadds(columns, 48, mask=full)),
                self.kv_count,
                mask=full,
            )
            for group in range(8):
                current = rr.vreinterpret(
                    rr.vload_broadcast(self.maximum, group * 8, mode="elem2datablock"),
                    dtypes.float32,
                )
                p0 = rr.vselect(rr.vload(self.score, group * 64), neg, cond_mask=mask0)
                p1 = rr.vselect(
                    rr.vload(self.score, group * 64 + 512), neg, cond_mask=mask1
                )
                p2 = rr.vselect(
                    rr.vload(self.score, group * 64 + 1024), neg, cond_mask=mask2
                )
                p3 = rr.vselect(
                    rr.vload(self.score, group * 64 + 1536), neg, cond_mask=mask3
                )
                current = rr.vmax(current, p0, mask=full)
                current = rr.vmax(current, p1, mask=full)
                current = rr.vmax(current, p2, mask=full)
                current = rr.vmax(current, p3, mask=full)
                current = rr.vreduce_max_datablock(current, mask=full)
                lo, hi = rr.vdeinterleave(current, current)
                current = rr.vmax(lo, hi, mask=full)
                repeated, unused = rr.vinterleave(current, current)
                rr.vstore(self.maximum, group * 8, repeated, eight)
            rr.vmem_bar("vst_vld")
            maximum = rr.vload(self.maximum)
            first = rr.veqs(rr.vdups(dtypes.int64(tile), dtypes.int32), 0, mask=full)
            old_base = rr.vload(self.base)
            difference = rr.vsub(maximum, old_base, mask=full)
            flag = rr.vdup(rr.vreduce_max(difference, mask=full), mask=full)
            flag = rr.vselect(zero, flag, cond_mask=first)
            update = rr.mask_or(
                first, rr.vgts(flag, 6.0 / scale, mask=full), exec_mask=full
            )
            new_base = rr.vselect(maximum, old_base, cond_mask=update)
            rr.vstore(self.base, 0, new_base, full)
            factors = rr.vexp_sub(
                rr.vmuls(old_base, scale, mask=full),
                rr.vmuls(new_base, scale, mask=full),
                mask=full,
            )
            factors = rr.vselect(
                rr.vdups(1.0, dtypes.float32), factors, cond_mask=first
            )
            rr.vstore(self.scales, 0, factors, full)
            rr.vstore_first(self.delta, 0, flag)
            old_sum = rr.vload(self.sum)
            rr.vmem_bar("vst_vld")
            for start_row in range(0, 32, 8):
                base0 = rr.vreinterpret(
                    rr.vload_broadcast(self.base, start_row * 2, mode="elem2datablock"),
                    dtypes.float32,
                )
                base1 = rr.vreinterpret(
                    rr.vload_broadcast(
                        self.base, start_row * 2 + 8, mode="elem2datablock"
                    ),
                    dtypes.float32,
                )
                base0 = rr.vmuls(base0, scale, mask=full)
                base1 = rr.vmuls(base1, scale, mask=full)
                sum0, sum1 = zero, zero
                for pair in range_constexpr(2):
                    offset = start_row * 16 + pair * 1024
                    p00 = rr.vmuls(rr.vload(self.score, offset), scale, mask=full)
                    p10 = rr.vmuls(rr.vload(self.score, offset + 64), scale, mask=full)
                    p01 = rr.vmuls(rr.vload(self.score, offset + 512), scale, mask=full)
                    p11 = rr.vmuls(rr.vload(self.score, offset + 576), scale, mask=full)
                    if const_expr(pair == 0):
                        valid0, valid1 = mask0, mask1
                    else:
                        valid0, valid1 = mask2, mask3
                    p00 = rr.vselect(
                        rr.vexp_sub(p00, base0, mask=full), zero, cond_mask=valid0
                    )
                    p10 = rr.vselect(
                        rr.vexp_sub(p10, base1, mask=full), zero, cond_mask=valid0
                    )
                    p01 = rr.vselect(
                        rr.vexp_sub(p01, base0, mask=full), zero, cond_mask=valid1
                    )
                    p11 = rr.vselect(
                        rr.vexp_sub(p11, base1, mask=full), zero, cond_mask=valid1
                    )
                    sum0 = rr.vadd(sum0, p00, mask=full)
                    sum0 = rr.vadd(sum0, p01, mask=full)
                    sum1 = rr.vadd(sum1, p10, mask=full)
                    sum1 = rr.vadd(sum1, p11, mask=full)
                    s00 = rr.vcast(
                        p00, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ZERO
                    )
                    s10 = rr.vcast(
                        p10, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ZERO
                    )
                    s01 = rr.vcast(
                        p01, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ONE
                    )
                    s11 = rr.vcast(
                        p11, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ONE
                    )
                    packed0 = rr.vadd(s00, s01, mask=bfmask)
                    packed1 = rr.vadd(s10, s11, mask=bfmask)
                    out0, out1 = rr.vdeinterleave(packed0, packed1)
                    rr.vstore(self.prob, start_row * 16 + pair * 2048, out0, bfmask)
                    rr.vstore(
                        self.prob, start_row * 16 + pair * 2048 + 1024, out1, bfmask
                    )
                sum0 = rr.vpair_reduce_sum(
                    rr.vreduce_sum_datablock(sum0, mask=full), mask=full
                )
                sum1 = rr.vpair_reduce_sum(
                    rr.vreduce_sum_datablock(sum1, mask=full), mask=full
                )
                repeat0, unused = rr.vinterleave(sum0, sum0)
                repeat1, unused = rr.vinterleave(sum1, sum1)
                rr.vstore(self.sum, start_row * 2, repeat0, eight)
                rr.vstore(self.sum, start_row * 2 + 8, repeat1, eight)
            rr.vmem_bar("vst_vld")
            total = rr.vadd(
                rr.vmul(old_sum, factors, mask=full), rr.vload(self.sum), mask=full
            )
            rr.vstore(self.sum, 0, total, full)

    @jit
    def drain_fragment(self, fragment, column, add_history, scale_history):
        with vf():
            mask = rr.full_mask(32)
            for row in range(32):
                factor = rr.vload_broadcast(self.scales, row * 2)
                for part in range(2):
                    value = rr.vload(fragment, row * 128 + part * 64)
                    offset = row * 512 + column * 128 + part * 64
                    if add_history != dtypes.uint32(0):
                        value = rr.vadd(value, rr.vload(self.accum, offset), mask=mask)
                    if scale_history != dtypes.uint32(0):
                        value = rr.vmul(value, factor, mask=mask)
                    rr.vstore(self.accum, offset, value, mask)

    @jit
    def drain(self, scale_history):
        for column in range_constexpr(4):
            vec_sync_block_wait(PIPE.V, 14)
            self.drain_fragment(
                self.fragments[column],
                column,
                dtypes.int32(self.denorm[0]),
                scale_history,
            )
            vec_sync_block_arrive(PIPE.V, 15)

    @jit
    def finish(self, out, maximum, lse, query, scale, have_sink: Constexpr[bool]):
        with vf():
            full = rr.full_mask(32)
            headmask = rr.create_mask("vl32", 32)
            raw_maximum, unused = rr.vdeinterleave(
                rr.vload(self.maximum), rr.vload(self.maximum)
            )
            base, unused = rr.vdeinterleave(rr.vload(self.base), rr.vload(self.base))
            total, unused = rr.vdeinterleave(rr.vload(self.sum), rr.vload(self.sum))
            empty = rr.veqs(raw_maximum, -3.4028234663852886e38, mask=full)
            scaled_max = rr.vmuls(raw_maximum, scale, mask=full)
            scaled_base = rr.vmuls(base, scale, mask=full)
            negative_inf = rr.vreinterpret(
                rr.vdups(0xFF800000, dtypes.uint32), dtypes.float32
            )
            positive_inf = rr.vreinterpret(
                rr.vdups(0x7F800000, dtypes.uint32), dtypes.float32
            )
            logits = rr.vselect(negative_inf, scaled_max, cond_mask=empty)
            logsum = rr.vadd(rr.vlog(total, mask=full), scaled_base, mask=full)
            logsum = rr.vselect(positive_inf, logsum, cond_mask=empty)
            rr.vstore(self.maximum, 0, logits, headmask)
            rr.vstore(self.sum, 0, logsum, headmask)
            denominator = total
            if const_expr(have_sink):
                sink = rr.vexp_sub(rr.vload(self.sink), scaled_base, mask=full)
                denominator = rr.vadd(total, sink, mask=full)
            inverse = rr.vdiv(rr.vdups(1.0, dtypes.float32), denominator, mask=full)
            inverse = rr.vselect(
                rr.vdups(0.0, dtypes.float32), inverse, cond_mask=empty
            )
            rr.vstore(self.denorm, 0, inverse, headmask)
            rr.vmem_bar("vst_vld")
            for row in range(32):
                factor = rr.vload_broadcast(self.denorm, row)
                for column in range(8):
                    value = rr.vmul(
                        rr.vload(self.accum, row * 512 + column * 64), factor, mask=full
                    )
                    packed = rr.vcast(value, dtypes.bfloat16, mask=full)
                    rr.vstore_pack(
                        self.final,
                        row * 512 + column * 64,
                        packed,
                        full,
                        pack_mode="b32_to_b16",
                    )
        mem_copy(tile_slice(out, (32, 512), (query * 2 + self.part, 0)), self.final)
        mem_copy(
            tile_slice(maximum, (32,), (query * 2 + self.part,)),
            tile_slice(self.maximum, (32,), (0,)),
        )
        mem_copy(
            tile_slice(lse, (32,), (query * 2 + self.part,)),
            tile_slice(self.sum, (32,), (0,)),
        )


class Cube:
    """AIC 的原 L1 半区、四 L0 槽位和 QK/PV FP32 累加。"""

    def __init__(self):
        l1 = L1.view(524288)
        l0a = L0A.view(65536)
        l0b = L0B.view(65536)
        l0c = L0C.view(262144)
        self.q_lo = tuple(
            fixed_view(l1, (32, 512), dtypes.bfloat16, i * 32768) for i in range(2)
        )
        self.q_hi = tuple(
            fixed_view(l1, (32, 512), dtypes.bfloat16, 458752 + i * 32768)
            for i in range(2)
        )
        # One storage root makes the two independently written half views
        # participate in the same automatic two-AIV completion handshake.
        self.k_pool = fixed_view(l1, (180224,), dtypes.bfloat16, 65536, "nd")
        self.k_lo = tuple(
            reinterpret(
                self.k_pool,
                layout=make_layout((32, 512)),
                data_format="nz",
                offset=i * 32768,
            )
            for i in range(4)
        )
        self.k_hi = tuple(
            reinterpret(
                self.k_pool,
                layout=make_layout((32, 512)),
                data_format="nz",
                offset=229376 + i * 32768,
            )
            for i in range(4)
        )
        self.s = fixed_view(l1, (64, 64), dtypes.bfloat16, 258048)
        self.a = tuple(
            fixed_view(l0a, (64, 128), dtypes.bfloat16, i * 16384) for i in range(4)
        )
        self.b = tuple(
            fixed_view(l0b, (64, 128), dtypes.bfloat16, i * 16384) for i in range(4)
        )
        self.score = fixed_view(l0c, (64, 64), dtypes.float32, 131072)
        self.out = fixed_view(l0c, (64, 512), dtypes.float32)
        self.out_fragments = tuple(
            reinterpret(
                tile_slice(self.out, (64, 128), (0, column)),
                layout=make_layout((64, 128)),
                data_format="nz",
            )
            for column in range(4)
        )

    @jit
    def kv_slots(self, tile):
        return self.k_lo[0], self.k_hi[0]

    @jit
    def load_query(self, q, query):
        lo, hi = self.q_lo[0], self.q_hi[0]
        mem_copy(lo, tile_slice(q, (32, 512), (query * 2, 0)))
        mem_copy(hi, tile_slice(q, (32, 512), (query * 2 + 1, 0)))
        return lo, hi

    @jit
    def qk(self, qlo, qhi, klo, khi, destination, nz_score: Constexpr[bool]):
        set_mmad_direction("m")
        cube_sync_block_wait(PIPE.MTE1, 10)
        for column in range_constexpr(4):
            a, b = self.a[column], self.b[column]
            self.load_qk_halves(a, qlo, qhi, column)
            self.load_qk_halves(b, klo, khi, column)
            matmul(
                self.score, a, b, init=column == 0, unit_flag=3 if column == 3 else 2
            )
        mem_copy(destination, self.score.partition(dim=0), copy_options(unit_flag=3))
        cube_sync_block_arrive(PIPE.FIXPIPE, 11)

    @jit
    def load_qk_halves(self, destination, lo, hi, column: Constexpr[int]):
        # Walk the 16x16 fractals directly. The flat views have exact spans;
        # their source/destination steps use strides 2/4.
        for half in range_constexpr(2):
            if const_expr(half == 0):
                source = lo
            else:
                source = hi
            for row in range_constexpr(2):
                dst = reinterpret(
                    destination,
                    layout=make_layout((16, 464)),
                    offset=half * 1024 + row * 512,
                    data_format="nz",
                )
                src = reinterpret(
                    source,
                    layout=make_layout((16, 240)),
                    offset=column * 8192 + row * 512,
                    data_format="nz",
                )
                mem_copy(
                    tile_slice_multi(dst, (16, 16), repeats=8, step=(0, 64)),
                    tile_slice_multi(src, (16, 16), repeats=8, step=(0, 32)),
                )

    @jit
    def pv(self, klo, khi, clear, first_block, last_block):
        set_mmad_direction("n")
        a = reinterpret(self.a[0], layout=make_layout((64, 64)), data_format="nz")
        mem_copy(a, self.s)
        unit_flag = dtypes.int32(0)
        if first_block != 0:
            unit_flag = dtypes.int32(2)
        if last_block != 0:
            unit_flag = dtypes.int32(3)
        for column in range_constexpr(4):
            b_slot = self.b[column]
            b = reinterpret(b_slot, layout=make_layout((128, 64)), data_format="nz")
            # Transposed K=64, N=128: each K=32 half is a contiguous 8 KiB.
            mem_copy(
                reinterpret(b, layout=make_layout((128, 32)), data_format="nz"),
                tile_slice(klo, (32, 128), (0, column)).permute(1, 0),
            )
            mem_copy(
                reinterpret(
                    b,
                    layout=make_layout((128, 32)),
                    offset=8192,
                    data_format="nz",
                ),
                tile_slice(khi, (32, 128), (0, column)).permute(1, 0),
            )
            matmul(
                tile_slice(self.out, (64, 128), (0, column)),
                a,
                b,
                init=clear,
                unit_flag=unit_flag,
            )

    @jit
    def flush(self, fragments, last_block):
        for column in range_constexpr(4):
            mem_copy(
                fragments[column],
                self.out_fragments[column].partition(dim=0),
                copy_options(
                    unit_flag=dtypes.int32(3) if last_block else dtypes.int32(0)
                ),
            )
            cube_sync_block_arrive(PIPE.FIXPIPE, 14)
            cube_sync_block_wait(PIPE.FIXPIPE, 15)


# native_pipeline_state: production definitions.


@datastruct(c_name="SparsePrefillCubeCursor")
class CubeCursor:
    slot: dtypes.uint32


@datastruct(c_name="SparsePrefillUInt32Metadata")
class UInt32Metadata:
    head: dtypes.uint32
    active: CArray[dtypes.uint32, 6]
    job: CArray[dtypes.uint32, 6]
    query: CArray[dtypes.uint32, 6]
    tile: CArray[dtypes.uint32, 6]
    blocks: CArray[dtypes.uint32, 6]
    length: CArray[dtypes.uint32, 6]
    global_block: CArray[dtypes.uint32, 6]


class UInt32DelayLine:
    """Six stages with a bounded cursor, preserving the existing tap order."""

    def __init__(self):
        self.state = UInt32Metadata(
            head=0,
            active=[0] * 6,
            job=[0] * 6,
            query=[0] * 6,
            tile=[0] * 6,
            blocks=[0] * 6,
            length=[0] * 6,
            global_block=[0] * 6,
        )

    def push(self, *, active, job, query, tile, blocks, length, global_block):
        slot = self.state.head
        self.state.active[slot] = dtypes.uint32(active)
        self.state.job[slot] = dtypes.uint32(job)
        self.state.query[slot] = dtypes.uint32(query)
        self.state.tile[slot] = dtypes.uint32(tile)
        self.state.blocks[slot] = dtypes.uint32(blocks)
        self.state.length[slot] = dtypes.uint32(length)
        self.state.global_block[slot] = dtypes.uint32(global_block)

    def tap(self, lag):
        slot = (self.state.head + dtypes.uint32(6 - lag)) % dtypes.uint32(6)
        return SimpleNamespace(
            active=dtypes.int32(self.state.active[slot]),
            job=dtypes.int32(self.state.job[slot]),
            query=dtypes.int32(self.state.query[slot]),
            tile=dtypes.int32(self.state.tile[slot]),
            blocks=dtypes.int32(self.state.blocks[slot]),
            length=dtypes.int32(self.state.length[slot]),
            global_block=dtypes.int32(self.state.global_block[slot]),
        )

    @jit
    def advance(self):
        next_head = self.state.head + dtypes.uint32(1)
        if next_head == 6:
            next_head = dtypes.uint32(0)
        self.state.head = next_head


@datastruct(c_name="SparsePrefillBoundedInt64Metadata")
class BoundedInt64Metadata:
    head: dtypes.int64
    active: CArray[dtypes.int64, 6]
    job: CArray[dtypes.int64, 6]
    query: CArray[dtypes.int64, 6]
    tile: CArray[dtypes.int64, 6]
    blocks: CArray[dtypes.int64, 6]
    length: CArray[dtypes.int64, 6]
    global_block: CArray[dtypes.int64, 6]


class BoundedInt64DelayLine:
    """Keep the existing i64 payload; replace unbounded clock modulo by wrap."""

    def __init__(self):
        self.state = BoundedInt64Metadata(
            head=0,
            active=[0] * 6,
            job=[0] * 6,
            query=[0] * 6,
            tile=[0] * 6,
            blocks=[0] * 6,
            length=[0] * 6,
            global_block=[0] * 6,
        )

    def push(self, *, active, job, query, tile, blocks, length, global_block):
        slot = self.state.head
        self.state.active[slot] = dtypes.int64(active)
        self.state.job[slot] = dtypes.int64(job)
        self.state.query[slot] = dtypes.int64(query)
        self.state.tile[slot] = dtypes.int64(tile)
        self.state.blocks[slot] = dtypes.int64(blocks)
        self.state.length[slot] = dtypes.int64(length)
        self.state.global_block[slot] = dtypes.int64(global_block)

    def tap(self, lag):
        slot = self.state.head + (6 - lag)
        slot = slot - dtypes.int64(slot >= 6) * 6
        return SimpleNamespace(
            active=self.state.active[slot],
            job=self.state.job[slot],
            query=self.state.query[slot],
            tile=self.state.tile[slot],
            blocks=self.state.blocks[slot],
            length=self.state.length[slot],
            global_block=self.state.global_block[slot],
        )

    @jit
    def advance(self):
        next_head = self.state.head + 1
        if next_head == 6:
            next_head = dtypes.int64(0)
        self.state.head = next_head


class UInt32UnsignedDelayLine(UInt32DelayLine):
    """Keep non-negative scheduling fields unsigned, widening query for GM.

    Scheduling fields stay UINT32 through delay-line indexing.
    The ring wraps explicitly at its six-slot capacity.
    Payload order and the six stages are identical to UInt32DelayLine.
    The query field is widened to int64 for the DSL's GM offset expressions.
    """

    @jit
    def tap(self, lag):
        slot = self.state.head + dtypes.uint32(6 - lag)
        if slot >= 6:
            slot = slot - dtypes.uint32(6)
        return SimpleNamespace(
            active=self.state.active[slot],
            job=self.state.job[slot],
            query=dtypes.int64(self.state.query[slot]),
            tile=self.state.tile[slot],
            blocks=self.state.blocks[slot],
            length=self.state.length[slot],
            global_block=self.state.global_block[slot],
        )


class UInt32LiveDelayLine(UInt32UnsignedDelayLine):
    """Same allocation and taps; length is consumed only by the issuing tick.

    activate uses the current length for fetch/preprocess. None of its
    delayed consumers reads tap.length, so keep that field initialized to
    zero and omit its per-tick write. Every other payload is unchanged.
    """

    def push(self, *, active, job, query, tile, blocks, length, global_block):
        slot = self.state.head
        self.state.active[slot] = dtypes.uint32(active)
        self.state.job[slot] = dtypes.uint32(job)
        self.state.query[slot] = dtypes.uint32(query)
        self.state.tile[slot] = dtypes.uint32(tile)
        self.state.blocks[slot] = dtypes.uint32(blocks)
        self.state.global_block[slot] = dtypes.uint32(global_block)


# nd_score_permutation: production definitions.


@jit
def nd_score_to_native_nz(score: Tensor):
    # Retain the full 8 KiB tile before any scatter overwrites an ND row.
    with vf():
        full = rr.full_mask(32)
        lanes = rr.varange(0, dtypes.uint32)
        columns = rr.vbitwise_and(lanes, rr.vdups(15, dtypes.uint32), mask=full)
        n1 = rr.vshl(rr.vshr(lanes, 4, mask=full), 9, mask=full)
        offsets = rr.vadd(n1, columns, mask=full)
        row0 = rr.vload(score, 0)
        row1 = rr.vload(score, 64)
        row2 = rr.vload(score, 128)
        row3 = rr.vload(score, 192)
        row4 = rr.vload(score, 256)
        row5 = rr.vload(score, 320)
        row6 = rr.vload(score, 384)
        row7 = rr.vload(score, 448)
        row8 = rr.vload(score, 512)
        row9 = rr.vload(score, 576)
        row10 = rr.vload(score, 640)
        row11 = rr.vload(score, 704)
        row12 = rr.vload(score, 768)
        row13 = rr.vload(score, 832)
        row14 = rr.vload(score, 896)
        row15 = rr.vload(score, 960)
        row16 = rr.vload(score, 1024)
        row17 = rr.vload(score, 1088)
        row18 = rr.vload(score, 1152)
        row19 = rr.vload(score, 1216)
        row20 = rr.vload(score, 1280)
        row21 = rr.vload(score, 1344)
        row22 = rr.vload(score, 1408)
        row23 = rr.vload(score, 1472)
        row24 = rr.vload(score, 1536)
        row25 = rr.vload(score, 1600)
        row26 = rr.vload(score, 1664)
        row27 = rr.vload(score, 1728)
        row28 = rr.vload(score, 1792)
        row29 = rr.vload(score, 1856)
        row30 = rr.vload(score, 1920)
        row31 = rr.vload(score, 1984)
        rr.vmem_bar("vld_vst")
        rr.vscatter(score, row0, rr.vadds(offsets, 0, mask=full), mask=full)
        rr.vscatter(score, row1, rr.vadds(offsets, 16, mask=full), mask=full)
        rr.vscatter(score, row2, rr.vadds(offsets, 32, mask=full), mask=full)
        rr.vscatter(score, row3, rr.vadds(offsets, 48, mask=full), mask=full)
        rr.vscatter(score, row4, rr.vadds(offsets, 64, mask=full), mask=full)
        rr.vscatter(score, row5, rr.vadds(offsets, 80, mask=full), mask=full)
        rr.vscatter(score, row6, rr.vadds(offsets, 96, mask=full), mask=full)
        rr.vscatter(score, row7, rr.vadds(offsets, 112, mask=full), mask=full)
        rr.vscatter(score, row8, rr.vadds(offsets, 128, mask=full), mask=full)
        rr.vscatter(score, row9, rr.vadds(offsets, 144, mask=full), mask=full)
        rr.vscatter(score, row10, rr.vadds(offsets, 160, mask=full), mask=full)
        rr.vscatter(score, row11, rr.vadds(offsets, 176, mask=full), mask=full)
        rr.vscatter(score, row12, rr.vadds(offsets, 192, mask=full), mask=full)
        rr.vscatter(score, row13, rr.vadds(offsets, 208, mask=full), mask=full)
        rr.vscatter(score, row14, rr.vadds(offsets, 224, mask=full), mask=full)
        rr.vscatter(score, row15, rr.vadds(offsets, 240, mask=full), mask=full)
        rr.vscatter(score, row16, rr.vadds(offsets, 256, mask=full), mask=full)
        rr.vscatter(score, row17, rr.vadds(offsets, 272, mask=full), mask=full)
        rr.vscatter(score, row18, rr.vadds(offsets, 288, mask=full), mask=full)
        rr.vscatter(score, row19, rr.vadds(offsets, 304, mask=full), mask=full)
        rr.vscatter(score, row20, rr.vadds(offsets, 320, mask=full), mask=full)
        rr.vscatter(score, row21, rr.vadds(offsets, 336, mask=full), mask=full)
        rr.vscatter(score, row22, rr.vadds(offsets, 352, mask=full), mask=full)
        rr.vscatter(score, row23, rr.vadds(offsets, 368, mask=full), mask=full)
        rr.vscatter(score, row24, rr.vadds(offsets, 384, mask=full), mask=full)
        rr.vscatter(score, row25, rr.vadds(offsets, 400, mask=full), mask=full)
        rr.vscatter(score, row26, rr.vadds(offsets, 416, mask=full), mask=full)
        rr.vscatter(score, row27, rr.vadds(offsets, 432, mask=full), mask=full)
        rr.vscatter(score, row28, rr.vadds(offsets, 448, mask=full), mask=full)
        rr.vscatter(score, row29, rr.vadds(offsets, 464, mask=full), mask=full)
        rr.vscatter(score, row30, rr.vadds(offsets, 480, mask=full), mask=full)
        rr.vscatter(score, row31, rr.vadds(offsets, 496, mask=full), mask=full)
        rr.vmem_bar("vst_vld")


# nd_grouped_reduction: production definitions.


@jit
def update_maximum_grouped(score, indices, maximum, kv_count):
    full = rr.full_mask(32)
    sixteen = rr.create_mask("vl16", 32)
    neg = rr.vdups(-3.4028234663852886e38, dtypes.float32)
    lanes = rr.varange(0, dtypes.uint32)
    columns = rr.vbitwise_and(lanes, rr.vdups(7, dtypes.uint32), mask=full)
    for head in range(0, 32, 8):
        current = neg
        for column in range_constexpr(0, 64, 8):
            values = rr.vload_strided(score, head * 64 + column, full, block_stride=8)
            valid = rr.vlts(
                rr.vgather_reg(indices, rr.vadds(columns, column, mask=full)),
                kv_count,
                mask=full,
            )
            current = rr.vselect(
                rr.vmax(current, values, mask=full), current, cond_mask=valid
            )
        current = rr.vreduce_max_datablock(current, mask=full)
        repeated, unused = rr.vinterleave(current, current)
        previous = rr.vload_strided(maximum, head * 2, sixteen, block_stride=1)
        rr.vstore(maximum, head * 2, rr.vmax(repeated, previous, mask=sixteen), sixteen)


# nd_register_reduction: production definitions.


@jit
def maximum_in_registers(score, indices, maximum, kv_count):
    full = rr.full_mask(32)
    neg = rr.vdups(-3.4028234663852886e38, dtypes.float32)
    valid = rr.vlts(indices, kv_count, mask=full)
    heads = rr.vshr(rr.varange(0, dtypes.uint32), 1, mask=full)
    maxima = rr.vload(maximum)
    for row in range(32):
        values = rr.vselect(rr.vload(score, row * 64), neg, cond_mask=valid)
        current = rr.vdup(rr.vreduce_max(values, mask=full), mask=full)
        maxima = rr.vselect(
            rr.vmax(maxima, current, mask=full),
            maxima,
            cond_mask=rr.veqs(heads, row, mask=full),
        )
    rr.vstore(maximum, 0, maxima, full)


@jit
def probability_in_registers(
    score, prob, indices, base, total_sum, scales, kv_count, scale
):
    full = rr.full_mask(32)
    zero = rr.vdups(0.0, dtypes.float32)
    valid = rr.vlts(indices, kv_count, mask=full)
    heads = rr.vshr(rr.varange(0, dtypes.uint32), 1, mask=full)
    previous = rr.vload(total_sum)
    factors = rr.vload(scales)
    sums = zero
    for row in range(32):
        values = rr.vexp_sub(
            rr.vmuls(rr.vload(score, row * 64), scale, mask=full),
            rr.vmuls(rr.vload_broadcast(base, row * 2), scale, mask=full),
            mask=full,
        )
        values = rr.vselect(values, zero, cond_mask=valid)
        current = rr.vdup(rr.vreduce_sum(values, mask=full), mask=full)
        sums = rr.vselect(current, sums, cond_mask=rr.veqs(heads, row, mask=full))
        packed = rr.vcast(values, dtypes.bfloat16, mask=full)
        rr.vstore_pack(prob, row * 64, packed, full, pack_mode="b32_to_b16")
    rr.vstore(
        total_sum,
        0,
        rr.vadd(rr.vmul(previous, factors, mask=full), sums, mask=full),
        full,
    )


# nd_grouped_probability: production definitions.


@jit
def probability_grouped(
    score,
    prob,
    indices,
    base,
    total_sum,
    scales,
    kv_count,
    scale,
    first: Constexpr[bool] = False,
):
    full = rr.full_mask(32)
    eight = rr.create_mask("vl8", 32)
    bf64 = rr.create_mask("vl64", 16)
    zero = rr.vdups(0.0, dtypes.float32)
    lanes = rr.varange(0, dtypes.uint32)
    columns = rr.vbitwise_and(lanes, rr.vdups(15, dtypes.uint32), mask=full)
    blocks = rr.vadd(
        rr.vmuls(rr.vshr(lanes, 1, mask=full), 8, mask=full),
        rr.vbitwise_and(lanes, rr.vdups(1, dtypes.uint32), mask=full),
        mask=full,
    )
    offsets = rr.vmuls(blocks, 32, mask=full)
    if const_expr(not first):
        old_total = rr.vload(total_sum)
        factors = rr.vload(scales)
    valid0 = rr.vlts(rr.vgather_reg(indices, columns), kv_count, mask=full)
    valid1 = rr.vlts(
        rr.vgather_reg(indices, rr.vadds(columns, 16, mask=full)), kv_count, mask=full
    )
    valid2 = rr.vlts(
        rr.vgather_reg(indices, rr.vadds(columns, 32, mask=full)), kv_count, mask=full
    )
    valid3 = rr.vlts(
        rr.vgather_reg(indices, rr.vadds(columns, 48, mask=full)), kv_count, mask=full
    )
    for head in range(0, 32, 4):
        base_scaled = rr.vmuls(
            rr.vreinterpret(
                rr.vload_broadcast(base, head * 2, mode="elem2datablock"),
                dtypes.float32,
            ),
            scale,
            mask=full,
        )
        # Retain all 64 columns of these four heads before a BF16 store can
        # overwrite a lower-address FP32 score row in the shared 8 KiB tile.
        address = rr.vadds(offsets, head * 256, mask=full)
        p0 = rr.vgather_datablock(score, address, mask=full)
        p1 = rr.vgather_datablock(score, rr.vadds(address, 64, mask=full), mask=full)
        p2 = rr.vgather_datablock(score, rr.vadds(address, 128, mask=full), mask=full)
        p3 = rr.vgather_datablock(score, rr.vadds(address, 192, mask=full), mask=full)
        p0 = rr.vselect(
            rr.vexp_sub(rr.vmuls(p0, scale, mask=full), base_scaled, mask=full),
            zero,
            cond_mask=valid0,
        )
        p1 = rr.vselect(
            rr.vexp_sub(rr.vmuls(p1, scale, mask=full), base_scaled, mask=full),
            zero,
            cond_mask=valid1,
        )
        p2 = rr.vselect(
            rr.vexp_sub(rr.vmuls(p2, scale, mask=full), base_scaled, mask=full),
            zero,
            cond_mask=valid2,
        )
        p3 = rr.vselect(
            rr.vexp_sub(rr.vmuls(p3, scale, mask=full), base_scaled, mask=full),
            zero,
            cond_mask=valid3,
        )
        summed = rr.vadd(
            rr.vadd(rr.vadd(p0, p1, mask=full), p2, mask=full), p3, mask=full
        )
        summed = rr.vpair_reduce_sum(
            rr.vreduce_sum_datablock(summed, mask=full), mask=full
        )
        repeated, unused = rr.vinterleave(summed, summed)
        rr.vstore(total_sum, head * 2, repeated, eight)
        rr.vmem_bar("vld_vst")
        narrow0 = rr.vcast(p0, dtypes.bfloat16, mask=full)
        s0, unused = rr.vdeinterleave(narrow0, narrow0)
        narrow1 = rr.vcast(p1, dtypes.bfloat16, mask=full)
        s1, unused = rr.vdeinterleave(narrow1, narrow1)
        narrow2 = rr.vcast(p2, dtypes.bfloat16, mask=full)
        s2, unused = rr.vdeinterleave(narrow2, narrow2)
        narrow3 = rr.vcast(p3, dtypes.bfloat16, mask=full)
        s3, unused = rr.vdeinterleave(narrow3, narrow3)
        rr.vstore_strided(prob, head * 64, s0, bf64, block_stride=4)
        rr.vstore_strided(prob, head * 64 + 16, s1, bf64, block_stride=4)
        rr.vstore_strided(prob, head * 64 + 32, s2, bf64, block_stride=4)
        rr.vstore_strided(prob, head * 64 + 48, s3, bf64, block_stride=4)
    rr.vmem_bar("vst_vld")
    if const_expr(not first):
        total = rr.vadd(
            rr.vmul(old_total, factors, mask=full), rr.vload(total_sum), mask=full
        )
        rr.vstore(total_sum, 0, total, full)


@jit
def probability_nd_to_nz_grouped(prob, prob_nz):
    full = rr.full_mask(16)
    for head in range(0, 32, 8):
        for column in range_constexpr(0, 64, 16):
            values = rr.vload_strided(prob, head * 64 + column, full, block_stride=4)
            rr.vstore(prob_nz, (column // 16) * 512 + head * 16, values, full)


# nd_fused_probability: production definitions.


@jit
def probability_to_nz(
    score, prob_nz, valid, base, total_sum, scales, scale, first: Constexpr[bool]
):
    full = rr.full_mask(32)
    bf64 = rr.create_mask("vl64", 16)
    zero = rr.vdups(0.0, dtypes.float32)
    # Capture every FP32 score that aliases the 4 KiB NZ destination.
    upper16 = rr.vload(score, 1024)
    upper17 = rr.vload(score, 1088)
    upper18 = rr.vload(score, 1152)
    upper19 = rr.vload(score, 1216)
    upper20 = rr.vload(score, 1280)
    upper21 = rr.vload(score, 1344)
    upper22 = rr.vload(score, 1408)
    upper23 = rr.vload(score, 1472)
    upper24 = rr.vload(score, 1536)
    upper25 = rr.vload(score, 1600)
    upper26 = rr.vload(score, 1664)
    upper27 = rr.vload(score, 1728)
    upper28 = rr.vload(score, 1792)
    upper29 = rr.vload(score, 1856)
    upper30 = rr.vload(score, 1920)
    upper31 = rr.vload(score, 1984)
    rr.vmem_bar("vld_vst")
    p0 = rr.vexp_sub(
        rr.vmuls(rr.vload(score, 0), scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 0), scale, mask=full),
        mask=full,
    )
    p0 = rr.vselect(p0, zero, cond_mask=valid)
    sum0 = rr.vreduce_sum(p0, mask=full)
    if const_expr(not first):
        sum0 = rr.vadd(
            sum0,
            rr.vmul(
                rr.vload_broadcast(total_sum, 0),
                rr.vload_broadcast(scales, 0),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 0, sum0)
    rr.vstore_first(total_sum, 1, sum0)
    narrow0 = rr.vcast(p0, dtypes.bfloat16, mask=full)
    packed0, unused0 = rr.vdeinterleave(narrow0, narrow0)
    rr.vstore_strided(prob_nz, 0, packed0, bf64, block_stride=32)
    p1 = rr.vexp_sub(
        rr.vmuls(rr.vload(score, 64), scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 2), scale, mask=full),
        mask=full,
    )
    p1 = rr.vselect(p1, zero, cond_mask=valid)
    sum1 = rr.vreduce_sum(p1, mask=full)
    if const_expr(not first):
        sum1 = rr.vadd(
            sum1,
            rr.vmul(
                rr.vload_broadcast(total_sum, 2),
                rr.vload_broadcast(scales, 2),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 2, sum1)
    rr.vstore_first(total_sum, 3, sum1)
    narrow1 = rr.vcast(p1, dtypes.bfloat16, mask=full)
    packed1, unused1 = rr.vdeinterleave(narrow1, narrow1)
    rr.vstore_strided(prob_nz, 16, packed1, bf64, block_stride=32)
    p2 = rr.vexp_sub(
        rr.vmuls(rr.vload(score, 128), scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 4), scale, mask=full),
        mask=full,
    )
    p2 = rr.vselect(p2, zero, cond_mask=valid)
    sum2 = rr.vreduce_sum(p2, mask=full)
    if const_expr(not first):
        sum2 = rr.vadd(
            sum2,
            rr.vmul(
                rr.vload_broadcast(total_sum, 4),
                rr.vload_broadcast(scales, 4),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 4, sum2)
    rr.vstore_first(total_sum, 5, sum2)
    narrow2 = rr.vcast(p2, dtypes.bfloat16, mask=full)
    packed2, unused2 = rr.vdeinterleave(narrow2, narrow2)
    rr.vstore_strided(prob_nz, 32, packed2, bf64, block_stride=32)
    p3 = rr.vexp_sub(
        rr.vmuls(rr.vload(score, 192), scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 6), scale, mask=full),
        mask=full,
    )
    p3 = rr.vselect(p3, zero, cond_mask=valid)
    sum3 = rr.vreduce_sum(p3, mask=full)
    if const_expr(not first):
        sum3 = rr.vadd(
            sum3,
            rr.vmul(
                rr.vload_broadcast(total_sum, 6),
                rr.vload_broadcast(scales, 6),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 6, sum3)
    rr.vstore_first(total_sum, 7, sum3)
    narrow3 = rr.vcast(p3, dtypes.bfloat16, mask=full)
    packed3, unused3 = rr.vdeinterleave(narrow3, narrow3)
    rr.vstore_strided(prob_nz, 48, packed3, bf64, block_stride=32)
    p4 = rr.vexp_sub(
        rr.vmuls(rr.vload(score, 256), scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 8), scale, mask=full),
        mask=full,
    )
    p4 = rr.vselect(p4, zero, cond_mask=valid)
    sum4 = rr.vreduce_sum(p4, mask=full)
    if const_expr(not first):
        sum4 = rr.vadd(
            sum4,
            rr.vmul(
                rr.vload_broadcast(total_sum, 8),
                rr.vload_broadcast(scales, 8),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 8, sum4)
    rr.vstore_first(total_sum, 9, sum4)
    narrow4 = rr.vcast(p4, dtypes.bfloat16, mask=full)
    packed4, unused4 = rr.vdeinterleave(narrow4, narrow4)
    rr.vstore_strided(prob_nz, 64, packed4, bf64, block_stride=32)
    p5 = rr.vexp_sub(
        rr.vmuls(rr.vload(score, 320), scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 10), scale, mask=full),
        mask=full,
    )
    p5 = rr.vselect(p5, zero, cond_mask=valid)
    sum5 = rr.vreduce_sum(p5, mask=full)
    if const_expr(not first):
        sum5 = rr.vadd(
            sum5,
            rr.vmul(
                rr.vload_broadcast(total_sum, 10),
                rr.vload_broadcast(scales, 10),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 10, sum5)
    rr.vstore_first(total_sum, 11, sum5)
    narrow5 = rr.vcast(p5, dtypes.bfloat16, mask=full)
    packed5, unused5 = rr.vdeinterleave(narrow5, narrow5)
    rr.vstore_strided(prob_nz, 80, packed5, bf64, block_stride=32)
    p6 = rr.vexp_sub(
        rr.vmuls(rr.vload(score, 384), scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 12), scale, mask=full),
        mask=full,
    )
    p6 = rr.vselect(p6, zero, cond_mask=valid)
    sum6 = rr.vreduce_sum(p6, mask=full)
    if const_expr(not first):
        sum6 = rr.vadd(
            sum6,
            rr.vmul(
                rr.vload_broadcast(total_sum, 12),
                rr.vload_broadcast(scales, 12),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 12, sum6)
    rr.vstore_first(total_sum, 13, sum6)
    narrow6 = rr.vcast(p6, dtypes.bfloat16, mask=full)
    packed6, unused6 = rr.vdeinterleave(narrow6, narrow6)
    rr.vstore_strided(prob_nz, 96, packed6, bf64, block_stride=32)
    p7 = rr.vexp_sub(
        rr.vmuls(rr.vload(score, 448), scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 14), scale, mask=full),
        mask=full,
    )
    p7 = rr.vselect(p7, zero, cond_mask=valid)
    sum7 = rr.vreduce_sum(p7, mask=full)
    if const_expr(not first):
        sum7 = rr.vadd(
            sum7,
            rr.vmul(
                rr.vload_broadcast(total_sum, 14),
                rr.vload_broadcast(scales, 14),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 14, sum7)
    rr.vstore_first(total_sum, 15, sum7)
    narrow7 = rr.vcast(p7, dtypes.bfloat16, mask=full)
    packed7, unused7 = rr.vdeinterleave(narrow7, narrow7)
    rr.vstore_strided(prob_nz, 112, packed7, bf64, block_stride=32)
    p8 = rr.vexp_sub(
        rr.vmuls(rr.vload(score, 512), scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 16), scale, mask=full),
        mask=full,
    )
    p8 = rr.vselect(p8, zero, cond_mask=valid)
    sum8 = rr.vreduce_sum(p8, mask=full)
    if const_expr(not first):
        sum8 = rr.vadd(
            sum8,
            rr.vmul(
                rr.vload_broadcast(total_sum, 16),
                rr.vload_broadcast(scales, 16),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 16, sum8)
    rr.vstore_first(total_sum, 17, sum8)
    narrow8 = rr.vcast(p8, dtypes.bfloat16, mask=full)
    packed8, unused8 = rr.vdeinterleave(narrow8, narrow8)
    rr.vstore_strided(prob_nz, 128, packed8, bf64, block_stride=32)
    p9 = rr.vexp_sub(
        rr.vmuls(rr.vload(score, 576), scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 18), scale, mask=full),
        mask=full,
    )
    p9 = rr.vselect(p9, zero, cond_mask=valid)
    sum9 = rr.vreduce_sum(p9, mask=full)
    if const_expr(not first):
        sum9 = rr.vadd(
            sum9,
            rr.vmul(
                rr.vload_broadcast(total_sum, 18),
                rr.vload_broadcast(scales, 18),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 18, sum9)
    rr.vstore_first(total_sum, 19, sum9)
    narrow9 = rr.vcast(p9, dtypes.bfloat16, mask=full)
    packed9, unused9 = rr.vdeinterleave(narrow9, narrow9)
    rr.vstore_strided(prob_nz, 144, packed9, bf64, block_stride=32)
    p10 = rr.vexp_sub(
        rr.vmuls(rr.vload(score, 640), scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 20), scale, mask=full),
        mask=full,
    )
    p10 = rr.vselect(p10, zero, cond_mask=valid)
    sum10 = rr.vreduce_sum(p10, mask=full)
    if const_expr(not first):
        sum10 = rr.vadd(
            sum10,
            rr.vmul(
                rr.vload_broadcast(total_sum, 20),
                rr.vload_broadcast(scales, 20),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 20, sum10)
    rr.vstore_first(total_sum, 21, sum10)
    narrow10 = rr.vcast(p10, dtypes.bfloat16, mask=full)
    packed10, unused10 = rr.vdeinterleave(narrow10, narrow10)
    rr.vstore_strided(prob_nz, 160, packed10, bf64, block_stride=32)
    p11 = rr.vexp_sub(
        rr.vmuls(rr.vload(score, 704), scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 22), scale, mask=full),
        mask=full,
    )
    p11 = rr.vselect(p11, zero, cond_mask=valid)
    sum11 = rr.vreduce_sum(p11, mask=full)
    if const_expr(not first):
        sum11 = rr.vadd(
            sum11,
            rr.vmul(
                rr.vload_broadcast(total_sum, 22),
                rr.vload_broadcast(scales, 22),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 22, sum11)
    rr.vstore_first(total_sum, 23, sum11)
    narrow11 = rr.vcast(p11, dtypes.bfloat16, mask=full)
    packed11, unused11 = rr.vdeinterleave(narrow11, narrow11)
    rr.vstore_strided(prob_nz, 176, packed11, bf64, block_stride=32)
    p12 = rr.vexp_sub(
        rr.vmuls(rr.vload(score, 768), scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 24), scale, mask=full),
        mask=full,
    )
    p12 = rr.vselect(p12, zero, cond_mask=valid)
    sum12 = rr.vreduce_sum(p12, mask=full)
    if const_expr(not first):
        sum12 = rr.vadd(
            sum12,
            rr.vmul(
                rr.vload_broadcast(total_sum, 24),
                rr.vload_broadcast(scales, 24),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 24, sum12)
    rr.vstore_first(total_sum, 25, sum12)
    narrow12 = rr.vcast(p12, dtypes.bfloat16, mask=full)
    packed12, unused12 = rr.vdeinterleave(narrow12, narrow12)
    rr.vstore_strided(prob_nz, 192, packed12, bf64, block_stride=32)
    p13 = rr.vexp_sub(
        rr.vmuls(rr.vload(score, 832), scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 26), scale, mask=full),
        mask=full,
    )
    p13 = rr.vselect(p13, zero, cond_mask=valid)
    sum13 = rr.vreduce_sum(p13, mask=full)
    if const_expr(not first):
        sum13 = rr.vadd(
            sum13,
            rr.vmul(
                rr.vload_broadcast(total_sum, 26),
                rr.vload_broadcast(scales, 26),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 26, sum13)
    rr.vstore_first(total_sum, 27, sum13)
    narrow13 = rr.vcast(p13, dtypes.bfloat16, mask=full)
    packed13, unused13 = rr.vdeinterleave(narrow13, narrow13)
    rr.vstore_strided(prob_nz, 208, packed13, bf64, block_stride=32)
    p14 = rr.vexp_sub(
        rr.vmuls(rr.vload(score, 896), scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 28), scale, mask=full),
        mask=full,
    )
    p14 = rr.vselect(p14, zero, cond_mask=valid)
    sum14 = rr.vreduce_sum(p14, mask=full)
    if const_expr(not first):
        sum14 = rr.vadd(
            sum14,
            rr.vmul(
                rr.vload_broadcast(total_sum, 28),
                rr.vload_broadcast(scales, 28),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 28, sum14)
    rr.vstore_first(total_sum, 29, sum14)
    narrow14 = rr.vcast(p14, dtypes.bfloat16, mask=full)
    packed14, unused14 = rr.vdeinterleave(narrow14, narrow14)
    rr.vstore_strided(prob_nz, 224, packed14, bf64, block_stride=32)
    p15 = rr.vexp_sub(
        rr.vmuls(rr.vload(score, 960), scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 30), scale, mask=full),
        mask=full,
    )
    p15 = rr.vselect(p15, zero, cond_mask=valid)
    sum15 = rr.vreduce_sum(p15, mask=full)
    if const_expr(not first):
        sum15 = rr.vadd(
            sum15,
            rr.vmul(
                rr.vload_broadcast(total_sum, 30),
                rr.vload_broadcast(scales, 30),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 30, sum15)
    rr.vstore_first(total_sum, 31, sum15)
    narrow15 = rr.vcast(p15, dtypes.bfloat16, mask=full)
    packed15, unused15 = rr.vdeinterleave(narrow15, narrow15)
    rr.vstore_strided(prob_nz, 240, packed15, bf64, block_stride=32)
    p16 = rr.vexp_sub(
        rr.vmuls(upper16, scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 32), scale, mask=full),
        mask=full,
    )
    p16 = rr.vselect(p16, zero, cond_mask=valid)
    sum16 = rr.vreduce_sum(p16, mask=full)
    if const_expr(not first):
        sum16 = rr.vadd(
            sum16,
            rr.vmul(
                rr.vload_broadcast(total_sum, 32),
                rr.vload_broadcast(scales, 32),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 32, sum16)
    rr.vstore_first(total_sum, 33, sum16)
    narrow16 = rr.vcast(p16, dtypes.bfloat16, mask=full)
    packed16, unused16 = rr.vdeinterleave(narrow16, narrow16)
    rr.vstore_strided(prob_nz, 256, packed16, bf64, block_stride=32)
    p17 = rr.vexp_sub(
        rr.vmuls(upper17, scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 34), scale, mask=full),
        mask=full,
    )
    p17 = rr.vselect(p17, zero, cond_mask=valid)
    sum17 = rr.vreduce_sum(p17, mask=full)
    if const_expr(not first):
        sum17 = rr.vadd(
            sum17,
            rr.vmul(
                rr.vload_broadcast(total_sum, 34),
                rr.vload_broadcast(scales, 34),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 34, sum17)
    rr.vstore_first(total_sum, 35, sum17)
    narrow17 = rr.vcast(p17, dtypes.bfloat16, mask=full)
    packed17, unused17 = rr.vdeinterleave(narrow17, narrow17)
    rr.vstore_strided(prob_nz, 272, packed17, bf64, block_stride=32)
    p18 = rr.vexp_sub(
        rr.vmuls(upper18, scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 36), scale, mask=full),
        mask=full,
    )
    p18 = rr.vselect(p18, zero, cond_mask=valid)
    sum18 = rr.vreduce_sum(p18, mask=full)
    if const_expr(not first):
        sum18 = rr.vadd(
            sum18,
            rr.vmul(
                rr.vload_broadcast(total_sum, 36),
                rr.vload_broadcast(scales, 36),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 36, sum18)
    rr.vstore_first(total_sum, 37, sum18)
    narrow18 = rr.vcast(p18, dtypes.bfloat16, mask=full)
    packed18, unused18 = rr.vdeinterleave(narrow18, narrow18)
    rr.vstore_strided(prob_nz, 288, packed18, bf64, block_stride=32)
    p19 = rr.vexp_sub(
        rr.vmuls(upper19, scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 38), scale, mask=full),
        mask=full,
    )
    p19 = rr.vselect(p19, zero, cond_mask=valid)
    sum19 = rr.vreduce_sum(p19, mask=full)
    if const_expr(not first):
        sum19 = rr.vadd(
            sum19,
            rr.vmul(
                rr.vload_broadcast(total_sum, 38),
                rr.vload_broadcast(scales, 38),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 38, sum19)
    rr.vstore_first(total_sum, 39, sum19)
    narrow19 = rr.vcast(p19, dtypes.bfloat16, mask=full)
    packed19, unused19 = rr.vdeinterleave(narrow19, narrow19)
    rr.vstore_strided(prob_nz, 304, packed19, bf64, block_stride=32)
    p20 = rr.vexp_sub(
        rr.vmuls(upper20, scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 40), scale, mask=full),
        mask=full,
    )
    p20 = rr.vselect(p20, zero, cond_mask=valid)
    sum20 = rr.vreduce_sum(p20, mask=full)
    if const_expr(not first):
        sum20 = rr.vadd(
            sum20,
            rr.vmul(
                rr.vload_broadcast(total_sum, 40),
                rr.vload_broadcast(scales, 40),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 40, sum20)
    rr.vstore_first(total_sum, 41, sum20)
    narrow20 = rr.vcast(p20, dtypes.bfloat16, mask=full)
    packed20, unused20 = rr.vdeinterleave(narrow20, narrow20)
    rr.vstore_strided(prob_nz, 320, packed20, bf64, block_stride=32)
    p21 = rr.vexp_sub(
        rr.vmuls(upper21, scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 42), scale, mask=full),
        mask=full,
    )
    p21 = rr.vselect(p21, zero, cond_mask=valid)
    sum21 = rr.vreduce_sum(p21, mask=full)
    if const_expr(not first):
        sum21 = rr.vadd(
            sum21,
            rr.vmul(
                rr.vload_broadcast(total_sum, 42),
                rr.vload_broadcast(scales, 42),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 42, sum21)
    rr.vstore_first(total_sum, 43, sum21)
    narrow21 = rr.vcast(p21, dtypes.bfloat16, mask=full)
    packed21, unused21 = rr.vdeinterleave(narrow21, narrow21)
    rr.vstore_strided(prob_nz, 336, packed21, bf64, block_stride=32)
    p22 = rr.vexp_sub(
        rr.vmuls(upper22, scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 44), scale, mask=full),
        mask=full,
    )
    p22 = rr.vselect(p22, zero, cond_mask=valid)
    sum22 = rr.vreduce_sum(p22, mask=full)
    if const_expr(not first):
        sum22 = rr.vadd(
            sum22,
            rr.vmul(
                rr.vload_broadcast(total_sum, 44),
                rr.vload_broadcast(scales, 44),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 44, sum22)
    rr.vstore_first(total_sum, 45, sum22)
    narrow22 = rr.vcast(p22, dtypes.bfloat16, mask=full)
    packed22, unused22 = rr.vdeinterleave(narrow22, narrow22)
    rr.vstore_strided(prob_nz, 352, packed22, bf64, block_stride=32)
    p23 = rr.vexp_sub(
        rr.vmuls(upper23, scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 46), scale, mask=full),
        mask=full,
    )
    p23 = rr.vselect(p23, zero, cond_mask=valid)
    sum23 = rr.vreduce_sum(p23, mask=full)
    if const_expr(not first):
        sum23 = rr.vadd(
            sum23,
            rr.vmul(
                rr.vload_broadcast(total_sum, 46),
                rr.vload_broadcast(scales, 46),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 46, sum23)
    rr.vstore_first(total_sum, 47, sum23)
    narrow23 = rr.vcast(p23, dtypes.bfloat16, mask=full)
    packed23, unused23 = rr.vdeinterleave(narrow23, narrow23)
    rr.vstore_strided(prob_nz, 368, packed23, bf64, block_stride=32)
    p24 = rr.vexp_sub(
        rr.vmuls(upper24, scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 48), scale, mask=full),
        mask=full,
    )
    p24 = rr.vselect(p24, zero, cond_mask=valid)
    sum24 = rr.vreduce_sum(p24, mask=full)
    if const_expr(not first):
        sum24 = rr.vadd(
            sum24,
            rr.vmul(
                rr.vload_broadcast(total_sum, 48),
                rr.vload_broadcast(scales, 48),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 48, sum24)
    rr.vstore_first(total_sum, 49, sum24)
    narrow24 = rr.vcast(p24, dtypes.bfloat16, mask=full)
    packed24, unused24 = rr.vdeinterleave(narrow24, narrow24)
    rr.vstore_strided(prob_nz, 384, packed24, bf64, block_stride=32)
    p25 = rr.vexp_sub(
        rr.vmuls(upper25, scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 50), scale, mask=full),
        mask=full,
    )
    p25 = rr.vselect(p25, zero, cond_mask=valid)
    sum25 = rr.vreduce_sum(p25, mask=full)
    if const_expr(not first):
        sum25 = rr.vadd(
            sum25,
            rr.vmul(
                rr.vload_broadcast(total_sum, 50),
                rr.vload_broadcast(scales, 50),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 50, sum25)
    rr.vstore_first(total_sum, 51, sum25)
    narrow25 = rr.vcast(p25, dtypes.bfloat16, mask=full)
    packed25, unused25 = rr.vdeinterleave(narrow25, narrow25)
    rr.vstore_strided(prob_nz, 400, packed25, bf64, block_stride=32)
    p26 = rr.vexp_sub(
        rr.vmuls(upper26, scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 52), scale, mask=full),
        mask=full,
    )
    p26 = rr.vselect(p26, zero, cond_mask=valid)
    sum26 = rr.vreduce_sum(p26, mask=full)
    if const_expr(not first):
        sum26 = rr.vadd(
            sum26,
            rr.vmul(
                rr.vload_broadcast(total_sum, 52),
                rr.vload_broadcast(scales, 52),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 52, sum26)
    rr.vstore_first(total_sum, 53, sum26)
    narrow26 = rr.vcast(p26, dtypes.bfloat16, mask=full)
    packed26, unused26 = rr.vdeinterleave(narrow26, narrow26)
    rr.vstore_strided(prob_nz, 416, packed26, bf64, block_stride=32)
    p27 = rr.vexp_sub(
        rr.vmuls(upper27, scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 54), scale, mask=full),
        mask=full,
    )
    p27 = rr.vselect(p27, zero, cond_mask=valid)
    sum27 = rr.vreduce_sum(p27, mask=full)
    if const_expr(not first):
        sum27 = rr.vadd(
            sum27,
            rr.vmul(
                rr.vload_broadcast(total_sum, 54),
                rr.vload_broadcast(scales, 54),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 54, sum27)
    rr.vstore_first(total_sum, 55, sum27)
    narrow27 = rr.vcast(p27, dtypes.bfloat16, mask=full)
    packed27, unused27 = rr.vdeinterleave(narrow27, narrow27)
    rr.vstore_strided(prob_nz, 432, packed27, bf64, block_stride=32)
    p28 = rr.vexp_sub(
        rr.vmuls(upper28, scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 56), scale, mask=full),
        mask=full,
    )
    p28 = rr.vselect(p28, zero, cond_mask=valid)
    sum28 = rr.vreduce_sum(p28, mask=full)
    if const_expr(not first):
        sum28 = rr.vadd(
            sum28,
            rr.vmul(
                rr.vload_broadcast(total_sum, 56),
                rr.vload_broadcast(scales, 56),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 56, sum28)
    rr.vstore_first(total_sum, 57, sum28)
    narrow28 = rr.vcast(p28, dtypes.bfloat16, mask=full)
    packed28, unused28 = rr.vdeinterleave(narrow28, narrow28)
    rr.vstore_strided(prob_nz, 448, packed28, bf64, block_stride=32)
    p29 = rr.vexp_sub(
        rr.vmuls(upper29, scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 58), scale, mask=full),
        mask=full,
    )
    p29 = rr.vselect(p29, zero, cond_mask=valid)
    sum29 = rr.vreduce_sum(p29, mask=full)
    if const_expr(not first):
        sum29 = rr.vadd(
            sum29,
            rr.vmul(
                rr.vload_broadcast(total_sum, 58),
                rr.vload_broadcast(scales, 58),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 58, sum29)
    rr.vstore_first(total_sum, 59, sum29)
    narrow29 = rr.vcast(p29, dtypes.bfloat16, mask=full)
    packed29, unused29 = rr.vdeinterleave(narrow29, narrow29)
    rr.vstore_strided(prob_nz, 464, packed29, bf64, block_stride=32)
    p30 = rr.vexp_sub(
        rr.vmuls(upper30, scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 60), scale, mask=full),
        mask=full,
    )
    p30 = rr.vselect(p30, zero, cond_mask=valid)
    sum30 = rr.vreduce_sum(p30, mask=full)
    if const_expr(not first):
        sum30 = rr.vadd(
            sum30,
            rr.vmul(
                rr.vload_broadcast(total_sum, 60),
                rr.vload_broadcast(scales, 60),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 60, sum30)
    rr.vstore_first(total_sum, 61, sum30)
    narrow30 = rr.vcast(p30, dtypes.bfloat16, mask=full)
    packed30, unused30 = rr.vdeinterleave(narrow30, narrow30)
    rr.vstore_strided(prob_nz, 480, packed30, bf64, block_stride=32)
    p31 = rr.vexp_sub(
        rr.vmuls(upper31, scale, mask=full),
        rr.vmuls(rr.vload_broadcast(base, 62), scale, mask=full),
        mask=full,
    )
    p31 = rr.vselect(p31, zero, cond_mask=valid)
    sum31 = rr.vreduce_sum(p31, mask=full)
    if const_expr(not first):
        sum31 = rr.vadd(
            sum31,
            rr.vmul(
                rr.vload_broadcast(total_sum, 62),
                rr.vload_broadcast(scales, 62),
                mask=full,
            ),
            mask=full,
        )
    rr.vstore_first(total_sum, 62, sum31)
    rr.vstore_first(total_sum, 63, sum31)
    narrow31 = rr.vcast(p31, dtypes.bfloat16, mask=full)
    packed31, unused31 = rr.vdeinterleave(narrow31, narrow31)
    rr.vstore_strided(prob_nz, 496, packed31, bf64, block_stride=32)


# nd_first_tile_softmax: production definitions.


@jit
def first_tile_softmax(
    score,
    prob,
    prob_nz,
    validity,
    maximum,
    base,
    total_sum,
    scales,
    mask_offset,
    kv_count,
    scale,
    grouped_probability: Constexpr[bool],
    fused_probability: Constexpr[bool],
    history_words,
    history_index,
    reset_history: Constexpr[bool],
):
    with vf():
        if const_expr(reset_history):
            rr.vstore_first(history_words, history_index, rr.vdups(0, dtypes.uint32))
        full = rr.full_mask(32)
        neg = rr.vdups(-3.4028234663852886e38, dtypes.float32)
        zero = rr.vdups(0.0, dtypes.float32)
        flags = rr.vload_unpack(validity, mask_offset, unpack_mode="b8_to_b32")
        valid = rr.vgts(flags, 0, mask=full)
        indices = rr.vselect(
            rr.vdups(0, dtypes.uint32),
            rr.vdups(kv_count, dtypes.uint32),
            cond_mask=valid,
        )
        for row in range(32):
            values = rr.vselect(rr.vload(score, row * 64), neg, cond_mask=valid)
            current = rr.vreduce_max(values, mask=full)
            rr.vstore_first(maximum, row * 2, current)
            rr.vstore_first(maximum, row * 2 + 1, current)
        rr.vmem_bar("vst_vld")
        rr.vstore(base, 0, rr.vload(maximum), full)
        rr.vmem_bar("vst_vld")
        # No old maxima/sums, rescale threshold, scale factors or flag writes
        # are needed in the first-tile VF. Their tile-zero slots are unused.
        if const_expr(fused_probability):
            probability_to_nz(
                score, prob_nz, valid, base, total_sum, scales, scale, True
            )
        elif const_expr(grouped_probability):
            probability_grouped(
                score, prob, indices, base, total_sum, scales, kv_count, scale, True
            )
        else:
            for row in range(32):
                probability = rr.vexp_sub(
                    rr.vmuls(rr.vload(score, row * 64), scale, mask=full),
                    rr.vmuls(rr.vload_broadcast(base, row * 2), scale, mask=full),
                    mask=full,
                )
                probability = rr.vselect(probability, zero, cond_mask=valid)
                current_sum = rr.vreduce_sum(probability, mask=full)
                rr.vstore_first(total_sum, row * 2, current_sum)
                rr.vstore_first(total_sum, row * 2 + 1, current_sum)
                packed = rr.vcast(probability, dtypes.bfloat16, mask=full)
                rr.vstore_pack(prob, row * 64, packed, full, pack_mode="b32_to_b16")
        rr.vmem_bar("vst_vld")
        if const_expr(not fused_probability):
            bfmask = rr.create_mask("vl64", 16)
            for row in range(32):
                rr.vstore_strided(
                    prob_nz, row * 16, rr.vload(prob, row * 64), bfmask, block_stride=32
                )


# nd_four_row_state: production definitions.


@jit
def repeat_four_rows(a, b, c, d, after2, after4, after6, full):
    a = rr.vdup(a, mask=full)
    b = rr.vdup(b, mask=full)
    c = rr.vdup(c, mask=full)
    d = rr.vdup(d, mask=full)
    packed = rr.vselect(b, a, cond_mask=after2)
    packed = rr.vselect(c, packed, cond_mask=after4)
    return rr.vselect(d, packed, cond_mask=after6)


@jit
def maximum_four_rows(score, valid, maximum):
    full = rr.full_mask(32)
    eight = rr.create_mask("vl8", 32)
    neg = rr.vdups(-3.4028234663852886e38, dtypes.float32)
    lanes = rr.varange(0, dtypes.uint32)
    after2 = rr.vgts(lanes, 1, mask=full)
    after4 = rr.vgts(lanes, 3, mask=full)
    after6 = rr.vgts(lanes, 5, mask=full)
    for head in range(0, 32, 4):
        p0 = rr.vselect(rr.vload(score, head * 64), neg, cond_mask=valid)
        p1 = rr.vselect(rr.vload(score, head * 64 + 64), neg, cond_mask=valid)
        p2 = rr.vselect(rr.vload(score, head * 64 + 128), neg, cond_mask=valid)
        p3 = rr.vselect(rr.vload(score, head * 64 + 192), neg, cond_mask=valid)
        a = rr.vreduce_max(p0, mask=full)
        b = rr.vreduce_max(p1, mask=full)
        c = rr.vreduce_max(p2, mask=full)
        d = rr.vreduce_max(p3, mask=full)
        packed = repeat_four_rows(a, b, c, d, after2, after4, after6, full)
        packed = rr.vmax(packed, rr.vload(maximum, head * 2), mask=eight)
        rr.vstore(maximum, head * 2, packed, eight)


@jit
def probability_four_rows(score, prob, valid, base, total_sum, scales, scale):
    full = rr.full_mask(32)
    eight = rr.create_mask("vl8", 32)
    zero = rr.vdups(0.0, dtypes.float32)
    lanes = rr.varange(0, dtypes.uint32)
    after2 = rr.vgts(lanes, 1, mask=full)
    after4 = rr.vgts(lanes, 3, mask=full)
    after6 = rr.vgts(lanes, 5, mask=full)
    for head in range(0, 32, 4):
        p0 = rr.vexp_sub(
            rr.vmuls(rr.vload(score, head * 64), scale, mask=full),
            rr.vmuls(rr.vload_broadcast(base, head * 2), scale, mask=full),
            mask=full,
        )
        p1 = rr.vexp_sub(
            rr.vmuls(rr.vload(score, head * 64 + 64), scale, mask=full),
            rr.vmuls(rr.vload_broadcast(base, head * 2 + 2), scale, mask=full),
            mask=full,
        )
        p2 = rr.vexp_sub(
            rr.vmuls(rr.vload(score, head * 64 + 128), scale, mask=full),
            rr.vmuls(rr.vload_broadcast(base, head * 2 + 4), scale, mask=full),
            mask=full,
        )
        p3 = rr.vexp_sub(
            rr.vmuls(rr.vload(score, head * 64 + 192), scale, mask=full),
            rr.vmuls(rr.vload_broadcast(base, head * 2 + 6), scale, mask=full),
            mask=full,
        )
        p0 = rr.vselect(p0, zero, cond_mask=valid)
        p1 = rr.vselect(p1, zero, cond_mask=valid)
        p2 = rr.vselect(p2, zero, cond_mask=valid)
        p3 = rr.vselect(p3, zero, cond_mask=valid)
        a = rr.vreduce_sum(p0, mask=full)
        b = rr.vreduce_sum(p1, mask=full)
        c = rr.vreduce_sum(p2, mask=full)
        d = rr.vreduce_sum(p3, mask=full)
        packed = repeat_four_rows(a, b, c, d, after2, after4, after6, full)
        packed = rr.vadd(
            packed,
            rr.vmul(
                rr.vload(total_sum, head * 2), rr.vload(scales, head * 2), mask=eight
            ),
            mask=eight,
        )
        rr.vstore(total_sum, head * 2, packed, eight)
        rr.vstore_pack(
            prob,
            head * 64,
            rr.vcast(p0, dtypes.bfloat16, mask=full),
            full,
            pack_mode="b32_to_b16",
        )
        rr.vstore_pack(
            prob,
            head * 64 + 64,
            rr.vcast(p1, dtypes.bfloat16, mask=full),
            full,
            pack_mode="b32_to_b16",
        )
        rr.vstore_pack(
            prob,
            head * 64 + 128,
            rr.vcast(p2, dtypes.bfloat16, mask=full),
            full,
            pack_mode="b32_to_b16",
        )
        rr.vstore_pack(
            prob,
            head * 64 + 192,
            rr.vcast(p3, dtypes.bfloat16, mask=full),
            full,
            pack_mode="b32_to_b16",
        )


# nd_output_packing: production definitions.


@jit
def final_output_native(
    fragment,
    accum,
    final,
    denorm,
    column: Constexpr[int],
    have_history: Constexpr[bool],
    use_cursors: Constexpr[bool],
):
    if const_expr(use_cursors):
        # Separate, fixed-base views avoid common-subexpression elimination
        # of identical post-update loads. They alias the original two slots;
        # each cursor advances by two rows, with no new UB allocation.
        f0 = make_tensor(
            make_pointer(dtypes.float32, 197632 + (column % 2) * 16384, MemLoc.UB),
            make_layout((16, 64), stride=(256, 1)),
        )
        f1 = make_tensor(
            make_pointer(
                dtypes.float32, 197632 + (column % 2) * 16384 + 256, MemLoc.UB
            ),
            make_layout((16, 64), stride=(256, 1)),
        )
        f2 = make_tensor(
            make_pointer(
                dtypes.float32, 197632 + (column % 2) * 16384 + 512, MemLoc.UB
            ),
            make_layout((16, 64), stride=(256, 1)),
        )
        f3 = make_tensor(
            make_pointer(
                dtypes.float32, 197632 + (column % 2) * 16384 + 768, MemLoc.UB
            ),
            make_layout((16, 64), stride=(256, 1)),
        )
        final_cursor = make_tensor(
            make_pointer(dtypes.bfloat16, 132096 + column * 512, MemLoc.UB),
            make_layout((32, 128), stride=(1024, 1)),
        )
        if const_expr(have_history):
            h0 = make_tensor(
                make_pointer(dtypes.float32, 132096 + column * 512, MemLoc.UB),
                make_layout((16, 64), stride=(1024, 1)),
            )
            h1 = make_tensor(
                make_pointer(dtypes.float32, 132096 + column * 512 + 256, MemLoc.UB),
                make_layout((16, 64), stride=(1024, 1)),
            )
            h2 = make_tensor(
                make_pointer(dtypes.float32, 132096 + column * 512 + 2048, MemLoc.UB),
                make_layout((16, 64), stride=(1024, 1)),
            )
            h3 = make_tensor(
                make_pointer(dtypes.float32, 132096 + column * 512 + 2304, MemLoc.UB),
                make_layout((16, 64), stride=(1024, 1)),
            )
    with vf():
        full = rr.full_mask(32)
        bfmask = rr.full_mask(16)
        for row in range(0, 32, 2):
            factor0 = rr.vload_broadcast(denorm, row)
            factor1 = rr.vload_broadcast(denorm, row + 1)
            if const_expr(use_cursors):
                p00 = rr.vload(f0, 256, post_update=True)
                p01 = rr.vload(f1, 256, post_update=True)
                p10 = rr.vload(f2, 256, post_update=True)
                p11 = rr.vload(f3, 256, post_update=True)
            else:
                p00 = rr.vload(fragment, row * 128)
                p01 = rr.vload(fragment, row * 128 + 64)
                p10 = rr.vload(fragment, (row + 1) * 128)
                p11 = rr.vload(fragment, (row + 1) * 128 + 64)
            if const_expr(have_history):
                # Only a view offset changes: these loads address the
                # [32,512] history region, never a separately allocated tile.
                if const_expr(use_cursors):
                    h00 = rr.vload(h0, 1024, post_update=True)
                    h01 = rr.vload(h1, 1024, post_update=True)
                    h10 = rr.vload(h2, 1024, post_update=True)
                    h11 = rr.vload(h3, 1024, post_update=True)
                else:
                    h00 = rr.vload(accum, row * 512)
                    h01 = rr.vload(accum, row * 512 + 64)
                    h10 = rr.vload(accum, (row + 1) * 512)
                    h11 = rr.vload(accum, (row + 1) * 512 + 64)
                p00 = rr.vadd(p00, h00, mask=full)
                p01 = rr.vadd(p01, h01, mask=full)
                p10 = rr.vadd(p10, h10, mask=full)
                p11 = rr.vadd(p11, h11, mask=full)
            s00 = rr.vcast(
                rr.vmul(p00, factor0, mask=full),
                dtypes.bfloat16,
                mask=full,
                reg_layout=RegLayout.ZERO,
            )
            s01 = rr.vcast(
                rr.vmul(p01, factor0, mask=full),
                dtypes.bfloat16,
                mask=full,
                reg_layout=RegLayout.ZERO,
            )
            s10 = rr.vcast(
                rr.vmul(p10, factor1, mask=full),
                dtypes.bfloat16,
                mask=full,
                reg_layout=RegLayout.ONE,
            )
            s11 = rr.vcast(
                rr.vmul(p11, factor1, mask=full),
                dtypes.bfloat16,
                mask=full,
                reg_layout=RegLayout.ONE,
            )
            packed0 = rr.vadd(s00, s10, mask=bfmask)
            packed1 = rr.vadd(s01, s11, mask=bfmask)
            output0, output1 = rr.vdeinterleave(packed0, packed1)
            if const_expr(use_cursors):
                rr.vstore(final_cursor, 1024, output0, bfmask, post_update=True)
                rr.vstore(final_cursor, 1024, output1, bfmask, post_update=True)
            else:
                rr.vstore(final, row * 1024, output0, bfmask)
                rr.vstore(final, (row + 1) * 1024, output1, bfmask)


# manual_pipelined_attention: production definitions.

KV_FLAG = 6
S_FLAG = 1
SCORE_FLAG = 0
RESCALE_FLAG = 2
FRAGMENT_FLAG = 4


@jit
def cube_pair_arrive(pipe: Constexpr[PIPE], flag):
    cube_sync_intra_arrive(pipe, flag)
    cube_sync_intra_arrive(pipe, flag + 16)


@jit
def cube_pair_wait(pipe: Constexpr[PIPE], flag):
    cube_sync_intra_wait(pipe, flag)
    cube_sync_intra_wait(pipe, flag + 16)


def identity_view(view):
    return view


def pointer_space(space, capacity):
    return make_tensor(
        make_pointer(dtypes.uint8, 0, space),
        make_layout((capacity,)),
        data_format="nd",
    )


def unbound_view(root, shape, dtype, address=0, data_format="nz"):
    return reinterpret(
        root,
        layout=make_layout(shape),
        dtype=dtype,
        offset=address,
        data_format=data_format,
    )


class DirectVector(Vector):
    def __init__(
        self, have_sink, nz_score=False, manual_gather=True, raw_nz_score=False
    ):
        self.nz_score = nz_score
        self.part = get_subblock_id()
        offsets = ub_offsets(have_sink)
        root = pointer_space(MemLoc.UB, UB_CAPACITY)
        if manual_gather:
            self.gather_buffers = tuple(
                identity_view(
                    reinterpret(
                        root,
                        dtype=dtypes.bfloat16,
                        layout=make_layout((32, 512)),
                        offset=32768 * slot,
                    )
                )
                for slot in range(3)
            )
        else:
            self.gather = make_ring_buffer(
                tuple(
                    reinterpret(
                        root,
                        dtype=dtypes.bfloat16,
                        layout=make_layout((32, 512)),
                        offset=32768 * slot,
                    )
                    for slot in range(3)
                )
            )
        scratch = identity_view(
            reinterpret(
                root,
                dtype=dtypes.bfloat16,
                layout=make_layout((33, 512)),
                offset=98304,
            )
        )
        self.nz = reinterpret(
            scratch,
            layout=make_layout((32, 512)),
            data_format="nz",
            physical_layout=make_layout((32, 2, 16, 16), stride=(528, 256, 16, 1)),
        )
        self.accum = identity_view(
            reinterpret(
                root,
                dtype=dtypes.float32,
                layout=make_layout((32, 512)),
                offset=132096,
            )
        )
        self.final = reinterpret(
            self.accum,
            dtype=dtypes.bfloat16,
            layout=make_layout((32, 512)),
        )
        # Four logical publications share the original two physical slots.
        # Each view is read once per flush, so automatic read sharing cannot
        # retain a slot across the explicit acknowledgement and its overwrite.
        self.fragments = tuple(
            identity_view(
                reinterpret(
                    root,
                    dtype=dtypes.float32,
                    layout=make_layout((32, 128)),
                    offset=197632 + 16384 * (slot % 2),
                )
            )
            for slot in range(4)
        )
        self.score = identity_view(
            reinterpret(
                root,
                dtype=dtypes.float32,
                layout=make_layout((32, 64)),
                offset=230400,
            )
        )
        self.prob = reinterpret(
            self.score,
            dtype=dtypes.bfloat16,
            layout=make_layout((32, 64)),
        )
        if nz_score:
            self.prob = reinterpret(
                self.score,
                dtype=dtypes.bfloat16,
                layout=make_layout((4096,)),
            )
            # The raw payload has N0=16; canonical UB fp32 NZ has N0=8.
            # The parent preserves the native 512-float N1 DMA pitch.
            if raw_nz_score:
                self.score_nz = make_tensor(
                    make_pointer(dtypes.float32, 230400, MemLoc.UB),
                    make_layout((32, 64)),
                    physical_layout=make_layout(
                        (4, 2, 16, 16), stride=(512, 256, 16, 1)
                    ),
                    data_format="nz",
                )
            else:
                self.score_nz = tile_slice(
                    reinterpret(
                        self.score,
                        layout=make_layout((64, 64)),
                        data_format="nz",
                    ),
                    (32, 64),
                    (0, 0),
                )
            # Pack BF16 in place with a 512-element gap per N1.
            self.prob_nz = reinterpret(
                self.score,
                dtype=dtypes.bfloat16,
                layout=make_layout((32, 64)),
                data_format="nz",
                physical_layout=make_layout((4, 2, 16, 16), stride=(1024, 256, 16, 1)),
            )
        else:
            prob_storage = identity_view(
                reinterpret(
                    root,
                    dtype=dtypes.bfloat16,
                    layout=make_layout((32, 64)),
                    offset=234496,
                )
            )
            self.prob_nz = reinterpret(
                prob_storage,
                layout=make_layout((32, 64)),
                data_format="nz",
            )
        self.maximum = identity_view(
            reinterpret(
                root,
                dtype=dtypes.float32,
                layout=make_layout((128,)),
                offset=offsets["row_max"],
            )
        )
        self.base = identity_view(
            reinterpret(
                root,
                dtype=dtypes.float32,
                layout=make_layout((128,)),
                offset=offsets["row_max_for_o"],
            )
        )
        self.sum = identity_view(
            reinterpret(
                root,
                dtype=dtypes.float32,
                layout=make_layout((128,)),
                offset=offsets["row_sum"],
            )
        )
        self.denorm = identity_view(
            reinterpret(
                root,
                dtype=dtypes.float32,
                layout=make_layout((128,)),
                offset=offsets["row_denorm"],
            )
        )
        self.scales = identity_view(
            reinterpret(
                root,
                dtype=dtypes.float32,
                layout=make_layout((256,)),
                offset=offsets["row_scales"],
            )
        )
        self.indices = identity_view(
            reinterpret(
                root,
                dtype=dtypes.int32,
                layout=make_layout((INDEX_GROUP_SIZE,)),
                offset=offsets["indices"],
            )
        )
        self.validity = identity_view(
            reinterpret(
                root,
                dtype=dtypes.uint8,
                layout=make_layout((4 * INDEX_GROUP_SIZE,)),
                offset=offsets["index_validity"],
            )
        )
        self.addresses = identity_view(
            reinterpret(
                root,
                dtype=dtypes.uint64,
                layout=make_layout((INDEX_PAIRS_PER_GROUP,)),
                offset=offsets["global_kv_ptr_offset"],
            )
        )
        self.strides = identity_view(
            reinterpret(
                root,
                dtype=dtypes.uint64,
                layout=make_layout((INDEX_PAIRS_PER_GROUP,)),
                offset=offsets["src_stride_in_gather2"],
            )
        )
        self.delta = identity_view(
            reinterpret(
                root,
                dtype=dtypes.float32,
                layout=make_layout((4,)),
                offset=offsets["row_max_max_delta"],
            )
        )
        if have_sink:
            self.sink = identity_view(
                reinterpret(
                    root,
                    dtype=dtypes.float32,
                    layout=make_layout((32,)),
                    offset=offsets["attn_sink"],
                )
            )


class DirectCube(Cube):
    def __init__(self):
        l1 = pointer_space(MemLoc.L1, 524288)
        l0a = pointer_space(MemLoc.L0A, 65536)
        l0b = pointer_space(MemLoc.L0B, 65536)
        l0c = pointer_space(MemLoc.L0C, 262144)
        self.q_lo = tuple(
            unbound_view(l1, (32, 512), dtypes.bfloat16, i * 32768) for i in range(2)
        )
        self.q_hi = tuple(
            unbound_view(l1, (32, 512), dtypes.bfloat16, 458752 + i * 32768)
            for i in range(2)
        )
        # One storage root makes the two independently written half views
        # participate in the same automatic two-AIV completion handshake.
        self.k_pool = unbound_view(l1, (180224,), dtypes.bfloat16, 65536, "nd")
        self.k_lo = tuple(
            reinterpret(
                self.k_pool,
                layout=make_layout((32, 512)),
                data_format="nz",
                offset=i * 32768,
            )
            for i in range(4)
        )
        self.k_hi = tuple(
            reinterpret(
                self.k_pool,
                layout=make_layout((32, 512)),
                data_format="nz",
                offset=229376 + i * 32768,
            )
            for i in range(4)
        )
        self.s = unbound_view(l1, (64, 64), dtypes.bfloat16, 258048)
        self.a = tuple(
            unbound_view(l0a, (64, 128), dtypes.bfloat16, i * 16384) for i in range(4)
        )
        self.b = tuple(
            unbound_view(l0b, (64, 128), dtypes.bfloat16, i * 16384) for i in range(4)
        )
        self.score = unbound_view(l0c, (64, 64), dtypes.float32, 131072)
        self.out = unbound_view(l0c, (64, 512), dtypes.float32)
        self.out_fragments = tuple(
            reinterpret(
                tile_slice(self.out, (64, 128), (0, column)),
                layout=make_layout((64, 128)),
                data_format="nz",
            )
            for column in range(4)
        )


class PipelineVector(DirectVector):
    def __init__(self, have_sink, nz_score=False, optimizations=0):
        DirectVector.__init__(
            self,
            have_sink,
            nz_score,
            manual_gather=True,
            raw_nz_score=bool(optimizations & NZ_SCORE_PATHS),
        )
        self.optimizations = optimizations
        self.kv_row_unroll = (
            8
            if optimizations & KV_ROWS_UNROLL8
            else 4
            if optimizations & KV_ROWS_UNROLL4
            else 2
        )
        if optimizations & NZ_MAXIMUM_CURSORS:
            # Four independent cursors walk the original NZ N1 columns.
            # Views alias score at 230400; no allocation or layout change.
            self.maximum_cols = tuple(
                make_tensor(
                    make_pointer(dtypes.float32, 230400 + column * 2048, MemLoc.UB),
                    make_layout((8, 64)),
                )
                for column in range(4)
            )

    @jit
    def gathered(self, block):
        stage = self.gather_buffers[0]
        if block % 3 == 1:
            stage = self.gather_buffers[1]
        if block % 3 == 2:
            stage = self.gather_buffers[2]
        return stage

    @jit
    def initialize(self, sink, have_sink: Constexpr[bool]):
        for slot in range_constexpr(3):
            with vf():
                full = rr.full_mask(16)
                zero = rr.vdups(0.0, dtypes.bfloat16)
                for start in range(0, 16384, 128):
                    rr.vstore(self.gather_buffers[slot], start, zero, full)
        if const_expr(not (self.optimizations & 2048)):
            with vf():
                full = rr.full_mask(32)
                zero = rr.vdups(0.0, dtypes.float32)
                for start in range(0, 16384, 64):
                    rr.vstore(self.accum, start, zero, full)
        if const_expr(have_sink):
            mem_copy(self.sink, tile_slice(sink, (32,), (self.part,)))

    @jit
    def initialize_states(self):
        # The verifier cannot correlate separate lag predicates. Define
        # state and payload storage once, outside the conditional stages.
        with vf():
            full = rr.full_mask(32)
            zero = rr.vdups(0.0, dtypes.float32)
            invalid = rr.vdups(self.kv_count, dtypes.int32)
            for start in range(0, 128, 64):
                rr.vstore(self.maximum, start, zero, full)
                rr.vstore(self.base, start, zero, full)
                rr.vstore(self.sum, start, zero, full)
                rr.vstore(self.denorm, start, zero, full)
            for start in range(0, 256, 64):
                rr.vstore(self.scales, start, zero, full)
            rr.vstore_first(self.delta, 0, zero)
            for start in range(0, INDEX_GROUP_SIZE, 64):
                rr.vstore(self.indices, start, invalid, full)
            flag_zero = rr.vdups(0, dtypes.uint32)
            for start in range(0, 4 * INDEX_GROUP_SIZE, 64):
                rr.vstore_pack(
                    self.validity, start, flag_zero, full, pack_mode="b32_to_b8"
                )
            for start in range(0, 2048, 64):
                rr.vstore(self.score, start, zero, full)
            for column in range_constexpr(4):
                for start in range(0, 4096, 64):
                    rr.vstore(self.fragments[column], start, zero, full)

    @jit
    def fetch(self, indices, query, length):
        vec_get_buf(PIPE.MTE2, 0)
        if const_expr(self.optimizations & NATIVE_INDEX_FETCH):
            # Dynamic intervals limit the index DMA to the effective prefix.
            # Keep one full index group in the raw workspace for both K values.
            if length > 0:
                raw = reinterpret(
                    self.indices,
                    dtype=dtypes.int32,
                    layout=make_layout((1, INDEX_GROUP_SIZE)),
                    data_format="nd",
                )
                mem_copy(raw[0:1, 0:length], indices[query : query + 1, 0:length])
        else:
            mem_copy(
                tile_slice(self.indices, (indices.shape[1],), (0,)),
                tile_slice(indices, (1, indices.shape[1]), (query, 0)).view(
                    (indices.shape[1],)
                ),
            )
        vec_release_buf(PIPE.MTE2, 0)

    @jit
    def preprocess(self, length, job, k: Constexpr[int]):
        # Four index-mask groups protect older softmax consumers from the
        # next query's raw-index DMA. Sort each pair into ascending order.
        vec_get_buf(PIPE.V, 0)
        vec_get_buf(PIPE.V, 1)
        length32 = dtypes.int32(length)
        with vf():
            full = rr.full_mask(32)
            invalid = rr.vdups(self.kv_count, dtypes.uint32)
            stride = rr.vdups(1024, dtypes.uint32)
            address_words = reinterpret(
                self.addresses,
                dtype=dtypes.uint32,
                layout=make_layout((2 * INDEX_PAIRS_PER_GROUP,)),
            )
            stride_words = reinterpret(
                self.strides,
                dtype=dtypes.uint32,
                layout=make_layout((2 * INDEX_PAIRS_PER_GROUP,)),
            )
            # The frontend checks region assignments before pruning constexpr
            # alternatives. The ordinary-load branch overwrites both values.
            raw0, raw1 = invalid, invalid
            if const_expr(self.optimizations & NATIVE_INDEX_PREPROCESS):
                # Hoist the pair positions; all five 128-index groups
                # still execute, including empty rows and K=128.
                positions_even, positions_odd = rr.vdeinterleave(
                    rr.varange(0, dtypes.int32), rr.varange(64, dtypes.int32)
                )
            if const_expr(self.optimizations & INDEX_LOAD_DEINTERLEAVE):
                positions_even = rr.vshl(rr.varange(0, dtypes.int32), 1, mask=full)
                positions_odd = rr.vadds(positions_even, 1, mask=full)
            for start_index in range(0, INDEX_GROUP_SIZE, 128):
                start = start_index
                if const_expr(self.optimizations & UNSIGNED_SCHEDULE):
                    start = dtypes.uint32(start_index)
                if const_expr(self.optimizations & INDEX_LOAD_DEINTERLEAVE):
                    even, odd = rr.vload_deinterleave(self.indices, start)
                    even = rr.vreinterpret(even, dtypes.uint32)
                    odd = rr.vreinterpret(odd, dtypes.uint32)
                    remaining = length32 - dtypes.int32(start)
                    prefix_even = rr.vlts(positions_even, remaining, mask=full)
                    prefix_odd = rr.vlts(positions_odd, remaining, mask=full)
                    even = rr.vselect(even, invalid, cond_mask=prefix_even)
                    odd = rr.vselect(odd, invalid, cond_mask=prefix_odd)
                else:
                    raw0 = rr.vreinterpret(rr.vload(self.indices, start), dtypes.uint32)
                    raw1 = rr.vreinterpret(
                        rr.vload(self.indices, start + 64), dtypes.uint32
                    )
                    even, odd = rr.vdeinterleave(raw0, raw1)
                if const_expr(self.optimizations & INDEX_LOAD_DEINTERLEAVE):
                    pass
                elif const_expr(self.optimizations & NATIVE_INDEX_PREPROCESS):
                    remaining = length - start
                    if const_expr(self.optimizations & INDEX_NARROW_LENGTH):
                        remaining = length32 - dtypes.int32(start)
                    if remaining < 128:
                        invalid_even = rr.vges(positions_even, remaining, mask=full)
                        invalid_odd = rr.vges(positions_odd, remaining, mask=full)
                        even = rr.vdups(
                            self.kv_count,
                            dtypes.uint32,
                            mask=invalid_even,
                            mode="merging",
                            merge=even,
                        )
                        odd = rr.vdups(
                            self.kv_count,
                            dtypes.uint32,
                            mask=invalid_odd,
                            mode="merging",
                            merge=odd,
                        )
                elif const_expr(self.optimizations & NATIVE_INDEX_MASKING):
                    if length - start < 128:
                        invalid0 = rr.vges(
                            rr.varange(start, dtypes.int32), length, mask=full
                        )
                        invalid1 = rr.vges(
                            rr.varange(start + 64, dtypes.int32), length, mask=full
                        )
                        raw0 = rr.vdups(
                            self.kv_count,
                            dtypes.uint32,
                            mask=invalid0,
                            mode="merging",
                            merge=raw0,
                        )
                        raw1 = rr.vdups(
                            self.kv_count,
                            dtypes.uint32,
                            mask=invalid1,
                            mode="merging",
                            merge=raw1,
                        )
                else:
                    prefix0 = rr.vlts(
                        rr.varange(start, dtypes.int32), length, mask=full
                    )
                    prefix1 = rr.vlts(
                        rr.varange(start + 64, dtypes.int32), length, mask=full
                    )
                    raw0 = rr.vselect(raw0, invalid, cond_mask=prefix0)
                    raw1 = rr.vselect(raw1, invalid, cond_mask=prefix1)
                if const_expr(
                    not (
                        self.optimizations
                        & (NATIVE_INDEX_PREPROCESS | INDEX_LOAD_DEINTERLEAVE)
                    )
                ):
                    even, odd = rr.vdeinterleave(raw0, raw1)
                smaller = rr.vmin(even, odd, mask=full)
                larger = rr.vmax(even, odd, mask=full)
                smaller = rr.vmin(smaller, invalid, mask=full)
                store_index_validity(
                    self.validity, job, start, smaller, larger, self.kv_count, full
                )
                valid_larger = rr.vlts(larger, self.kv_count, mask=full)
                if const_expr(self.optimizations & INDEX_MASKED_SUB):
                    # Masked subtraction zeroes inactive lanes: invalid second
                    # indices gather the first row twice with zero stride.
                    delta = rr.vsub(larger, smaller, mask=valid_larger)
                else:
                    safe_larger = rr.vselect(larger, smaller, cond_mask=valid_larger)
                    delta = rr.vsub(safe_larger, smaller, mask=full)
                addr_lo, addr_hi = rr.vmull(smaller, stride, dtypes.uint32, mask=full)
                gap_lo, gap_hi = rr.vmull(delta, stride, dtypes.uint32, mask=full)
                if const_expr(self.optimizations & INDEX_INTERLEAVE_STORE):
                    # Store the same 64-bit address/stride table directly
                    # from its 32-bit words, with no temporary interleave.
                    rr.vstore_interleave(address_words, start, addr_lo, addr_hi)
                    rr.vstore_interleave(stride_words, start, gap_lo, gap_hi)
                else:
                    addr0, addr1 = rr.vinterleave(addr_lo, addr_hi)
                    gap0, gap1 = rr.vinterleave(gap_lo, gap_hi)
                    rr.vstore(address_words, start, addr0, full)
                    rr.vstore(address_words, start + 64, addr1, full)
                    rr.vstore(stride_words, start, gap0, full)
                    rr.vstore(stride_words, start + 64, gap1, full)
        vec_release_buf(PIPE.V, 1)
        vec_release_buf(PIPE.V, 0)

    @jit
    def gather_block(self, kv, job, tile, global_block):
        stage = self.gathered(global_block)
        if const_expr(self.optimizations & GATHER_BYTE_INTERVAL):
            stage = reinterpret(
                stage,
                dtype=dtypes.uint8,
                layout=make_layout((32, 1024)),
            )
        vec_get_buf(PIPE.MTE2, 8 + global_block % 3)
        vec_get_buf(PIPE.S, 1)
        if const_expr(self.optimizations & GATHER_PAIR_ROLLED):
            for group in range(8):
                for lane in range_constexpr(2):
                    self.gather_pair(stage, kv, tile, group * 2 + lane)
        else:
            for pair in range_constexpr(16):
                self.gather_pair(stage, kv, tile, pair)
        vec_release_buf(PIPE.S, 1)
        vec_release_buf(PIPE.MTE2, 8 + global_block % 3)

    @jit
    def gather_pair(self, stage, kv, tile, pair):
        part = self.part
        if const_expr(self.optimizations & UNSIGNED_SCHEDULE):
            part = dtypes.uint32(self.part)
        offset = tile * INDEX_PAIRS_PER_TILE + part * 16 + pair
        first = self.addresses[offset]
        gap = self.strides[offset]
        if const_expr(
            (self.optimizations & GATHER_U32_ADDRESS)
            and self.kv_count * 1024 < 4294967296
        ):
            first = dtypes.uint32(first)
            gap = dtypes.uint32(gap)
        if first != self.kv_count * 1024:
            if const_expr(self.optimizations & GATHER_BYTE_INTERVAL):
                # Byte views change DMA addressing only; KV and its fixed UB
                # payload remain BF16. Both intervals retain one GM base.
                flat = kv.view(dtypes.uint8).view(self.kv_count * 1024)
                begin0 = dtypes.int64(first)
                begin1 = dtypes.int64(first + gap)
                src0 = flat[begin0 : begin0 + 1024,].view((1, 1024))
                src1 = flat[begin1 : begin1 + 1024,].view((1, 1024))
                mem_copy(
                    tile_slice(stage, (2, 1024), (pair, 0)),
                    (src0, src1),
                    copy_options(l2_cache_ctl=1),
                )
            elif const_expr(self.optimizations & GATHER_FLAT_INTERVAL):
                flat = kv.view(self.kv_count * 512)
                if const_expr(self.kv_count * 1024 < 4294967296):
                    begin0 = dtypes.int64(dtypes.uint32(first) // 2)
                    begin1 = dtypes.int64(dtypes.uint32(first + gap) // 2)
                else:
                    begin0 = dtypes.int64(first // 2)
                    begin1 = dtypes.int64((first + gap) // 2)
                # Match the one-leaf shape tree explicitly. These intervals
                # keep a shared GM base and lower to one two-burst DMA.
                src0 = flat[begin0 : begin0 + 512,].view((1, 512))
                src1 = flat[begin1 : begin1 + 512,].view((1, 512))
            elif const_expr(
                (self.optimizations & GATHER_UNSIGNED_ROWS)
                and self.kv_count * 1024 < 4294967296
            ):
                # Addresses remain byte offsets in the original UB table.
                # Larger address spaces use the original 64-bit path.
                row0 = dtypes.int64(dtypes.uint32(first)) // 1024
                row1 = dtypes.int64(dtypes.uint32(first + gap)) // 1024
                src0 = tile_slice(kv, (1, 512), (row0, 0)).view((1, 512))
                src1 = tile_slice(kv, (1, 512), (row1, 0)).view((1, 512))
            else:
                row0 = dtypes.int64(first // 1024)
                row1 = dtypes.int64((first + gap) // 1024)
                src0 = tile_slice(kv, (1, 512), (row0, 0)).view((1, 512))
                src1 = tile_slice(kv, (1, 512), (row1, 0)).view((1, 512))
            if const_expr(not bool(self.optimizations & GATHER_BYTE_INTERVAL)):
                mem_copy(
                    tile_slice(stage, (2, 512), (pair, 0)),
                    (src0, src1),
                    copy_options(l2_cache_ctl=1),
                )

    @jit
    def push_block(self, lo, hi, global_block):
        stage = self.gathered(global_block)
        vec_get_buf(PIPE.V, 8 + global_block % 3)
        vec_get_buf(PIPE.V, 11)
        sources = tuple(
            reinterpret(
                stage,
                dtype=dtypes.bfloat16,
                layout=make_layout((16000,)),
                offset=column * 256,
                data_format="nd",
            )
            for column in range(4)
        )
        with vf():
            mask = rr.full_mask(16)
            if const_expr(self.optimizations & KV_LOCAL_CURSORS):
                local_sources = tuple(
                    reinterpret(
                        stage,
                        dtype=dtypes.bfloat16,
                        layout=make_layout((16000,)),
                        offset=column * 256,
                        data_format="nd",
                    )
                    for column in range(4)
                )
                for pair in range_constexpr(2):
                    for row in range(32):
                        first = rr.vload(local_sources[pair * 2], 512, post_update=True)
                        second = rr.vload(
                            local_sources[pair * 2 + 1], 512, post_update=True
                        )
                        rr.vstore_strided(
                            self.nz,
                            row * 16 + pair * 256 * 33,
                            first,
                            mask,
                            block_stride=33,
                            repeat_stride=0,
                        )
                        rr.vstore_strided(
                            self.nz,
                            row * 16 + (pair * 2 + 1) * 128 * 33,
                            second,
                            mask,
                            block_stride=33,
                            repeat_stride=0,
                        )
            elif const_expr(
                self.optimizations
                & (KV_ROWS_UNROLL2 | KV_ROWS_UNROLL4 | KV_ROWS_UNROLL8)
            ):
                for chunk in range(0, 4, 2):
                    for group in range(32 // self.kv_row_unroll):
                        for lane in range_constexpr(self.kv_row_unroll):
                            row = group * self.kv_row_unroll + lane
                            first = rr.vload(stage, row * 512 + chunk * 128)
                            second = rr.vload(stage, row * 512 + (chunk + 1) * 128)
                            rr.vstore_strided(
                                self.nz,
                                row * 16 + chunk * 128 * 33,
                                first,
                                mask,
                                block_stride=33,
                                repeat_stride=0,
                            )
                            rr.vstore_strided(
                                self.nz,
                                row * 16 + (chunk + 1) * 128 * 33,
                                second,
                                mask,
                                block_stride=33,
                                repeat_stride=0,
                            )
            elif const_expr(self.optimizations & KV_LOAD_CURSORS):
                for pair in range_constexpr(2):
                    for row in range(32):
                        first = rr.vload(sources[pair * 2], 512, post_update=True)
                        second = rr.vload(sources[pair * 2 + 1], 512, post_update=True)
                        rr.vstore_strided(
                            self.nz,
                            row * 16 + pair * 256 * 33,
                            first,
                            mask,
                            block_stride=33,
                            repeat_stride=0,
                        )
                        rr.vstore_strided(
                            self.nz,
                            row * 16 + (pair * 2 + 1) * 128 * 33,
                            second,
                            mask,
                            block_stride=33,
                            repeat_stride=0,
                        )
            elif const_expr(self.optimizations & NATIVE_KV_PACK_ORDER):
                for chunk in range(0, 4, 2):
                    for row in range_constexpr(32):
                        first = rr.vload(stage, row * 512 + chunk * 128)
                        second = rr.vload(stage, row * 512 + (chunk + 1) * 128)
                        rr.vstore_strided(
                            self.nz,
                            row * 16 + chunk * 128 * 33,
                            first,
                            mask,
                            block_stride=33,
                            repeat_stride=0,
                        )
                        rr.vstore_strided(
                            self.nz,
                            row * 16 + (chunk + 1) * 128 * 33,
                            second,
                            mask,
                            block_stride=33,
                            repeat_stride=0,
                        )
            elif const_expr(self.optimizations & NATIVE_KV_PAIR_ROLLED):
                for chunk in range(0, 4, 2):
                    for row in range(32):
                        first = rr.vload(stage, row * 512 + chunk * 128)
                        second = rr.vload(stage, row * 512 + (chunk + 1) * 128)
                        rr.vstore_strided(
                            self.nz,
                            row * 16 + chunk * 128 * 33,
                            first,
                            mask,
                            block_stride=33,
                            repeat_stride=0,
                        )
                        rr.vstore_strided(
                            self.nz,
                            row * 16 + (chunk + 1) * 128 * 33,
                            second,
                            mask,
                            block_stride=33,
                            repeat_stride=0,
                        )
            else:
                for row in range(32):
                    for chunk in range_constexpr(4):
                        value = rr.vload(stage, row * 512 + chunk * 128)
                        rr.vstore_strided(
                            self.nz,
                            row * 16 + chunk * 128 * 33,
                            value,
                            mask,
                            block_stride=33,
                            repeat_stride=0,
                        )
        vec_release_buf(PIPE.V, 11)
        vec_release_buf(PIPE.V, 8 + global_block % 3)
        vec_get_buf(PIPE.MTE3, 11)
        vec_sync_intra_wait(PIPE.MTE3, KV_FLAG + global_block % 4)
        if self.part == 0:
            mem_copy(lo, self.nz)
        else:
            mem_copy(hi, self.nz)
        vec_sync_intra_arrive(PIPE.MTE3, KV_FLAG + global_block % 4)
        vec_release_buf(PIPE.MTE3, 11)

    @jit
    def softmax_block(self, job, tile, scale):
        if const_expr(self.nz_score):
            if const_expr(
                bool(self.optimizations & NZ_SCORE_PATHS)
                and bool(self.optimizations & 256)
            ):
                if tile == 0:
                    self.softmax_repacked(job, tile, scale, 1)
                else:
                    self.softmax_repacked(job, tile, scale, 0)
            else:
                self.softmax_repacked(job, tile, scale)
        else:
            if const_expr(self.optimizations & 256):
                if tile == 0:
                    self.softmax_first_block(job, scale)
                else:
                    self.softmax_nd_block(job, tile, scale)
            else:
                self.softmax_nd_block(job, tile, scale)

    @jit
    def softmax_first_block(self, job, scale):
        maximum = reinterpret(
            self.maximum,
            layout=make_layout((64,)),
            offset=(job % 2) * 256,
        )
        base = reinterpret(self.base, layout=make_layout((64,)), offset=(job % 2) * 256)
        total_sum = reinterpret(
            self.sum,
            layout=make_layout((64,)),
            offset=(job % 2) * 256,
        )
        scales = reinterpret(
            self.scales,
            layout=make_layout((64,)),
            offset=(job % 2) * 512,
        )
        vec_sync_intra_wait(PIPE.V, SCORE_FLAG)
        vec_get_buf(PIPE.V, 2 + job % 2)
        first_tile_softmax(
            self.score,
            self.prob,
            self.prob_nz,
            self.validity,
            maximum,
            base,
            total_sum,
            scales,
            (job % 4) * INDEX_GROUP_SIZE,
            self.kv_count,
            scale,
            bool(self.optimizations & 32),
            bool(self.optimizations & 512),
            reinterpret(self.denorm, dtype=dtypes.uint32, layout=make_layout((128,))),
            (dtypes.int64(job) % 2) * 64 + 32,
            bool(self.optimizations & FUSED_HISTORY_STATE),
        )
        vec_release_buf(PIPE.V, 2 + job % 2)
        vec_sync_notify(PIPE.V, PIPE.MTE3, 0)

    @jit
    def softmax_repacked(self, job, tile, scale, first_tile: Constexpr[int] = -1):
        maximum_buf = reinterpret(
            self.maximum,
            layout=make_layout((64,)),
            offset=(job % 2) * 256,
        )
        base_buf = reinterpret(
            self.base,
            layout=make_layout((64,)),
            offset=(job % 2) * 256,
        )
        sum_buf = reinterpret(
            self.sum,
            layout=make_layout((64,)),
            offset=(job % 2) * 256,
        )
        scales_buf = reinterpret(
            self.scales,
            layout=make_layout((64,)),
            offset=(job % 2) * 512 + (tile % 2) * 256,
        )
        history_words = reinterpret(
            self.denorm,
            dtype=dtypes.uint32,
            layout=make_layout((128,)),
        )
        vec_sync_intra_wait(PIPE.V, SCORE_FLAG)
        vec_get_buf(PIPE.V, 2 + job % 2)
        if const_expr(not bool(self.optimizations & NZ_SCORE_PATHS)):
            nd_score_to_native_nz(self.score)
        with vf():
            if const_expr(self.optimizations & SM_LOCAL_VIEWS):
                job32 = dtypes.uint32(dtypes.int64(job))
                tile32 = dtypes.uint32(dtypes.int64(tile))
                maximum_buf = reinterpret(
                    self.maximum, layout=make_layout((64,)), offset=(job32 % 2) * 256
                )
                base_buf = reinterpret(
                    self.base,
                    layout=make_layout((64,)),
                    offset=(job32 % 2) * 256,
                )
                sum_buf = reinterpret(
                    self.sum,
                    layout=make_layout((64,)),
                    offset=(job32 % 2) * 256,
                )
                scales_buf = reinterpret(
                    self.scales,
                    layout=make_layout((64,)),
                    offset=(job32 % 2) * 512 + (tile32 % 2) * 256,
                )
                history_words = reinterpret(
                    self.denorm, dtype=dtypes.uint32, layout=make_layout((128,))
                )
            if const_expr(self.optimizations & NZ_SHARED_FULL_MASK):
                # Share one all-active b16 predicate between FP32 and BF16 operations.
                bfmask = rr.full_mask(16)
                # The physical predicate is also all-active for FP32.  Mask
                # describes the consumer granularity without another PSET.
                full = rr.Mask(bfmask, 32)
            else:
                full = rr.full_mask(32)
                bfmask = rr.full_mask(16)
            eight = rr.create_mask("vl8", 32)
            neg = rr.vdups(-3.4028234663852886e38, dtypes.float32)
            zero = rr.vdups(0.0, dtypes.float32)
            if const_expr(first_tile < 0):
                if dtypes.int64(tile) == 0:
                    rr.vstore(maximum_buf, 0, neg, full)
                    rr.vstore(base_buf, 0, neg, full)
                    rr.vstore(sum_buf, 0, zero, full)
                    if const_expr(self.optimizations & FUSED_HISTORY_STATE):
                        rr.vstore_first(
                            history_words,
                            (dtypes.int64(job) % 2) * 64 + 32,
                            rr.vdups(0, dtypes.uint32),
                        )
                    rr.vmem_bar("vst_vld")
            elif const_expr(first_tile == 1):
                if const_expr(self.optimizations & FUSED_HISTORY_STATE):
                    rr.vstore_first(
                        history_words,
                        (dtypes.int64(job) % 2) * 64 + 32,
                        rr.vdups(0, dtypes.uint32),
                    )
            validity = load_index_validity(self.validity, job, tile)
            lanes = rr.varange(0, dtypes.uint32)
            columns = rr.vbitwise_and(lanes, rr.vdups(15, dtypes.uint32), mask=full)
            mask0 = rr.vgts(rr.vgather_reg(validity, columns), 0, mask=full)
            mask1 = rr.vgts(
                rr.vgather_reg(validity, rr.vadds(columns, 16, mask=full)), 0, mask=full
            )
            mask2 = rr.vgts(
                rr.vgather_reg(validity, rr.vadds(columns, 32, mask=full)), 0, mask=full
            )
            mask3 = rr.vgts(
                rr.vgather_reg(validity, rr.vadds(columns, 48, mask=full)), 0, mask=full
            )
            old_sum = zero
            if const_expr(bool(self.optimizations & SM_EARLY_SUM) and first_tile != 1):
                old_sum = rr.vload(sum_buf)
                rr.vmem_bar("vld_vst")
            for group in range(8):
                if const_expr(first_tile == 1):
                    current = neg
                else:
                    current = rr.vreinterpret(
                        rr.vload_broadcast(
                            maximum_buf, group * 8, mode="elem2datablock"
                        ),
                        dtypes.float32,
                    )
                if const_expr(self.optimizations & NZ_MAXIMUM_CURSORS):
                    p0 = rr.vselect(
                        rr.vload(self.maximum_cols[0], 64, post_update=True),
                        neg,
                        cond_mask=mask0,
                    )
                    p1 = rr.vselect(
                        rr.vload(self.maximum_cols[1], 64, post_update=True),
                        neg,
                        cond_mask=mask1,
                    )
                    p2 = rr.vselect(
                        rr.vload(self.maximum_cols[2], 64, post_update=True),
                        neg,
                        cond_mask=mask2,
                    )
                    p3 = rr.vselect(
                        rr.vload(self.maximum_cols[3], 64, post_update=True),
                        neg,
                        cond_mask=mask3,
                    )
                else:
                    p0 = rr.vselect(
                        rr.vload(self.score, group * 64), neg, cond_mask=mask0
                    )
                    p1 = rr.vselect(
                        rr.vload(self.score, group * 64 + 512), neg, cond_mask=mask1
                    )
                    p2 = rr.vselect(
                        rr.vload(self.score, group * 64 + 1024), neg, cond_mask=mask2
                    )
                    p3 = rr.vselect(
                        rr.vload(self.score, group * 64 + 1536), neg, cond_mask=mask3
                    )
                current = rr.vmax(current, p0, mask=full)
                current = rr.vmax(current, p1, mask=full)
                current = rr.vmax(current, p2, mask=full)
                current = rr.vmax(current, p3, mask=full)
                current = rr.vreduce_max_datablock(current, mask=full)
                lo, hi = rr.vdeinterleave(current, current)
                current = rr.vmax(lo, hi, mask=full)
                repeated, unused = rr.vinterleave(current, current)
                rr.vstore(maximum_buf, group * 8, repeated, eight)
            rr.vmem_bar("vst_vld")
            maximum = rr.vload(maximum_buf)
            if const_expr(first_tile == 1):
                # The first tile initializes state without history or rescaling.
                rr.vstore(base_buf, 0, maximum, full)
            else:
                old_base = rr.vload(base_buf)
                difference = rr.vsub(maximum, old_base, mask=full)
                flag = rr.vdup(rr.vreduce_max(difference, mask=full), mask=full)
                update = rr.vgts(flag, 6.0 / scale, mask=full)
                if const_expr(first_tile < 0):
                    first = rr.veqs(
                        rr.vdups(dtypes.int64(tile), dtypes.int32), 0, mask=full
                    )
                    flag = rr.vselect(zero, flag, cond_mask=first)
                    update = rr.mask_or(first, update, exec_mask=full)
                new_base = rr.vselect(maximum, old_base, cond_mask=update)
                rr.vstore(base_buf, 0, new_base, full)
                factors = rr.vexp_sub(
                    rr.vmuls(old_base, scale, mask=full),
                    rr.vmuls(new_base, scale, mask=full),
                    mask=full,
                )
                if const_expr(first_tile < 0):
                    factors = rr.vselect(
                        rr.vdups(1.0, dtypes.float32), factors, cond_mask=first
                    )
                rr.vstore(scales_buf, 0, factors, full)
                rr.vstore_first(
                    self.delta,
                    (dtypes.int64(job) % 2) * 2 + dtypes.int64(tile) % 2,
                    flag,
                )
                if const_expr(not bool(self.optimizations & SM_EARLY_SUM)):
                    old_sum = rr.vload(sum_buf)
            rr.vmem_bar("vst_vld")
            for start_row in range(0, 32, 8):
                base0 = rr.vreinterpret(
                    rr.vload_broadcast(base_buf, start_row * 2, mode="elem2datablock"),
                    dtypes.float32,
                )
                base1 = rr.vreinterpret(
                    rr.vload_broadcast(
                        base_buf, start_row * 2 + 8, mode="elem2datablock"
                    ),
                    dtypes.float32,
                )
                base0 = rr.vmuls(base0, scale, mask=full)
                base1 = rr.vmuls(base1, scale, mask=full)
                sum0, sum1 = zero, zero
                for pair in range_constexpr(2):
                    offset = start_row * 16 + pair * 1024
                    p00 = rr.vmuls(rr.vload(self.score, offset), scale, mask=full)
                    p10 = rr.vmuls(rr.vload(self.score, offset + 64), scale, mask=full)
                    p01 = rr.vmuls(rr.vload(self.score, offset + 512), scale, mask=full)
                    p11 = rr.vmuls(rr.vload(self.score, offset + 576), scale, mask=full)
                    if const_expr(pair == 0):
                        valid0, valid1 = mask0, mask1
                    else:
                        valid0, valid1 = mask2, mask3
                    if const_expr(self.optimizations & NZ_MASKED_EXP):
                        p00 = rr.vexp_sub(p00, base0, mask=valid0)
                        p10 = rr.vexp_sub(p10, base1, mask=valid0)
                        p01 = rr.vexp_sub(p01, base0, mask=valid1)
                        p11 = rr.vexp_sub(p11, base1, mask=valid1)
                    else:
                        p00 = rr.vselect(
                            rr.vexp_sub(p00, base0, mask=full), zero, cond_mask=valid0
                        )
                        p10 = rr.vselect(
                            rr.vexp_sub(p10, base1, mask=full), zero, cond_mask=valid0
                        )
                        p01 = rr.vselect(
                            rr.vexp_sub(p01, base0, mask=full), zero, cond_mask=valid1
                        )
                        p11 = rr.vselect(
                            rr.vexp_sub(p11, base1, mask=full), zero, cond_mask=valid1
                        )
                    if const_expr(
                        pair == 0 and bool(self.optimizations & NZ_SUM_FIRST_VALUE)
                    ):
                        sum0 = p00
                    else:
                        sum0 = rr.vadd(sum0, p00, mask=full)
                    sum0 = rr.vadd(sum0, p01, mask=full)
                    if const_expr(
                        pair == 0 and bool(self.optimizations & NZ_SUM_FIRST_VALUE)
                    ):
                        sum1 = p10
                    else:
                        sum1 = rr.vadd(sum1, p10, mask=full)
                    sum1 = rr.vadd(sum1, p11, mask=full)
                    s00 = rr.vcast(
                        p00, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ZERO
                    )
                    s10 = rr.vcast(
                        p10, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ZERO
                    )
                    s01 = rr.vcast(
                        p01, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ONE
                    )
                    s11 = rr.vcast(
                        p11, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ONE
                    )
                    packed0 = rr.vadd(s00, s01, mask=bfmask)
                    packed1 = rr.vadd(s10, s11, mask=bfmask)
                    out0, out1 = rr.vdeinterleave(packed0, packed1)
                    rr.vstore(self.prob, start_row * 16 + pair * 2048, out0, bfmask)
                    rr.vstore(
                        self.prob, start_row * 16 + pair * 2048 + 1024, out1, bfmask
                    )
                sum0 = rr.vpair_reduce_sum(
                    rr.vreduce_sum_datablock(sum0, mask=full), mask=full
                )
                sum1 = rr.vpair_reduce_sum(
                    rr.vreduce_sum_datablock(sum1, mask=full), mask=full
                )
                repeat0, unused = rr.vinterleave(sum0, sum0)
                repeat1, unused = rr.vinterleave(sum1, sum1)
                rr.vstore(sum_buf, start_row * 2, repeat0, eight)
                rr.vstore(sum_buf, start_row * 2 + 8, repeat1, eight)
            if const_expr(first_tile != 1):
                rr.vmem_bar("vst_vld")
                if const_expr(
                    self.optimizations & (NZ_SCORE_DUAL_COPY | NATIVE_SUM_MADD)
                ):
                    total = rr.vmadd(old_sum, factors, rr.vload(sum_buf), mask=full)
                else:
                    total = rr.vadd(
                        rr.vmul(old_sum, factors, mask=full),
                        rr.vload(sum_buf),
                        mask=full,
                    )
                rr.vstore(sum_buf, 0, total, full)

        vec_release_buf(PIPE.V, 2 + job % 2)
        vec_sync_notify(PIPE.V, PIPE.MTE3, 0)
        if tile != 0:
            vec_sync_notify(PIPE.V, PIPE.S, 0)

    @jit
    def softmax_nd_block(self, job, tile, scale):
        maximum = reinterpret(
            self.maximum,
            layout=make_layout((64,)),
            offset=(job % 2) * 256,
        )
        base = reinterpret(self.base, layout=make_layout((64,)), offset=(job % 2) * 256)
        total_sum = reinterpret(
            self.sum,
            layout=make_layout((64,)),
            offset=(job % 2) * 256,
        )
        scales = reinterpret(
            self.scales,
            layout=make_layout((64,)),
            offset=(job % 2) * 512 + (tile % 2) * 256,
        )
        history_words = reinterpret(
            self.denorm,
            dtype=dtypes.uint32,
            layout=make_layout((128,)),
        )
        vec_sync_intra_wait(PIPE.V, SCORE_FLAG)
        vec_get_buf(PIPE.V, 2 + job % 2)
        with vf():
            full = rr.full_mask(32)
            neg = rr.vdups(-3.4028234663852886e38, dtypes.float32)
            zero = rr.vdups(0.0, dtypes.float32)
            flags = load_index_validity(self.validity, job, tile)
            valid = rr.vgts(flags, 0, mask=full)
            indices = rr.vselect(
                rr.vdups(0, dtypes.uint32),
                rr.vdups(self.kv_count, dtypes.uint32),
                cond_mask=valid,
            )
            if dtypes.int64(tile) == 0:
                rr.vstore(maximum, 0, neg, full)
                rr.vstore(base, 0, neg, full)
                rr.vstore(total_sum, 0, zero, full)
                if const_expr(self.optimizations & FUSED_HISTORY_STATE):
                    rr.vstore_first(
                        history_words,
                        (dtypes.int64(job) % 2) * 64 + 32,
                        rr.vdups(0, dtypes.uint32),
                    )
                rr.vmem_bar("vst_vld")
            if const_expr(self.optimizations & 1024):
                maximum_four_rows(self.score, valid, maximum)
            elif const_expr(self.optimizations & 128):
                maximum_in_registers(self.score, indices, maximum, self.kv_count)
            elif const_expr(self.optimizations & 1):
                update_maximum_grouped(self.score, indices, maximum, self.kv_count)
            else:
                for row in range(32):
                    score = rr.vselect(
                        rr.vload(self.score, row * 64), neg, cond_mask=valid
                    )
                    cur = rr.vreduce_max(score, mask=full)
                    cur = rr.vmax(cur, rr.vload_broadcast(maximum, row * 2), mask=full)
                    rr.vstore_first(maximum, row * 2, cur)
                    rr.vstore_first(maximum, row * 2 + 1, cur)
            rr.vmem_bar("vst_vld")
            cur = rr.vload(maximum)
            previous = rr.vload(base)
            difference = rr.vsub(cur, previous, mask=full)
            flag = rr.vdup(rr.vreduce_max(difference, mask=full), mask=full)
            first = rr.veqs(rr.vdups(dtypes.int64(tile), dtypes.int32), 0, mask=full)
            flag = rr.vselect(zero, flag, cond_mask=first)
            update = rr.mask_or(
                first, rr.vgts(flag, 6.0 / scale, mask=full), exec_mask=full
            )
            next_base = rr.vselect(cur, previous, cond_mask=update)
            rr.vstore(base, 0, next_base, full)
            factors = rr.vexp_sub(
                rr.vmuls(previous, scale, mask=full),
                rr.vmuls(next_base, scale, mask=full),
                mask=full,
            )
            factors = rr.vselect(
                rr.vdups(1.0, dtypes.float32), factors, cond_mask=first
            )
            rr.vstore(scales, 0, factors, full)
            rr.vstore_first(
                self.delta, (dtypes.int64(job) % 2) * 2 + dtypes.int64(tile) % 2, flag
            )
            rr.vmem_bar("vst_vld")
            if const_expr(self.optimizations & 1024):
                probability_four_rows(
                    self.score, self.prob, valid, base, total_sum, scales, scale
                )
            elif const_expr(self.optimizations & 512):
                probability_to_nz(
                    self.score,
                    self.prob_nz,
                    valid,
                    base,
                    total_sum,
                    scales,
                    scale,
                    False,
                )
            elif const_expr(self.optimizations & 128):
                probability_in_registers(
                    self.score,
                    self.prob,
                    indices,
                    base,
                    total_sum,
                    scales,
                    self.kv_count,
                    scale,
                )
            elif const_expr(self.optimizations & 32):
                probability_grouped(
                    self.score,
                    self.prob,
                    indices,
                    base,
                    total_sum,
                    scales,
                    self.kv_count,
                    scale,
                )
            else:
                for row in range(32):
                    probability = rr.vexp_sub(
                        rr.vmuls(rr.vload(self.score, row * 64), scale, mask=full),
                        rr.vmuls(rr.vload_broadcast(base, row * 2), scale, mask=full),
                        mask=full,
                    )
                    probability = rr.vselect(probability, zero, cond_mask=valid)
                    current_sum = rr.vreduce_sum(probability, mask=full)
                    if const_expr(self.optimizations & NATIVE_SUM_MADD):
                        current_sum = rr.vmadd(
                            rr.vload_broadcast(total_sum, row * 2),
                            rr.vload_broadcast(scales, row * 2),
                            current_sum,
                            mask=full,
                        )
                    else:
                        current_sum = rr.vadd(
                            current_sum,
                            rr.vmul(
                                rr.vload_broadcast(total_sum, row * 2),
                                rr.vload_broadcast(scales, row * 2),
                                mask=full,
                            ),
                            mask=full,
                        )
                    rr.vstore_first(total_sum, row * 2, current_sum)
                    rr.vstore_first(total_sum, row * 2 + 1, current_sum)
                    packed = rr.vcast(probability, dtypes.bfloat16, mask=full)
                    rr.vstore_pack(
                        self.prob, row * 64, packed, full, pack_mode="b32_to_b16"
                    )
            rr.vmem_bar("vst_vld")
            if const_expr(self.optimizations & 512):
                pass
            elif const_expr(self.optimizations & 64):
                probability_nd_to_nz_grouped(self.prob, self.prob_nz)
            else:
                bfmask = rr.create_mask("vl64", 16)
                for row in range(32):
                    rr.vstore_strided(
                        self.prob_nz,
                        row * 16,
                        rr.vload(self.prob, row * 64),
                        bfmask,
                        block_stride=32,
                        repeat_stride=0,
                    )
        vec_release_buf(PIPE.V, 2 + job % 2)
        vec_sync_notify(PIPE.V, PIPE.MTE3, 0)
        if tile != 0:
            vec_sync_notify(PIPE.V, PIPE.S, 0)

    @jit
    def push_probability(self, destination):
        vec_sync_wait(PIPE.V, PIPE.MTE3, 0)
        vec_sync_intra_wait(PIPE.MTE3, S_FLAG)
        mem_copy(destination.partition(dim=0, part_id=self.part), self.prob_nz)
        vec_sync_intra_arrive(PIPE.MTE3, S_FLAG)
        vec_sync_intra_arrive(PIPE.MTE3, SCORE_FLAG)

    @jit
    def publish_rescale(self, flags, job, tile, scale):
        if tile != 0:
            vec_sync_wait(PIPE.V, PIPE.S, 0)
            flags.vec[job % 8, tile % 8, self.part] = dtypes.uint32(
                self.delta[(job % 2) * 2 + tile % 2] > 6.0 / scale
            )
            vec_sync_intra_arrive(PIPE.S, RESCALE_FLAG)

    @jit
    def finish_state(
        self, maximum_out, lse_out, job, query, scale, have_sink: Constexpr[bool]
    ):
        maximum = reinterpret(
            self.maximum,
            layout=make_layout((64,)),
            offset=(job % 2) * 256,
        )
        base = reinterpret(self.base, layout=make_layout((64,)), offset=(job % 2) * 256)
        sums = reinterpret(self.sum, layout=make_layout((64,)), offset=(job % 2) * 256)
        denorm = reinterpret(
            self.denorm,
            layout=make_layout((64,)),
            offset=(job % 2) * 256,
        )
        vec_get_buf(PIPE.V, 2 + job % 2)
        with vf():
            full = rr.full_mask(32)
            heads = rr.create_mask("vl32", 32)
            m, unused = rr.vdeinterleave(rr.vload(maximum), rr.vload(maximum))
            b, unused = rr.vdeinterleave(rr.vload(base), rr.vload(base))
            s, unused = rr.vdeinterleave(rr.vload(sums), rr.vload(sums))
            empty = rr.veqs(m, -3.4028234663852886e38, mask=full)
            b = rr.vmuls(b, scale, mask=full)
            ni = rr.vreinterpret(rr.vdups(0xFF800000, dtypes.uint32), dtypes.float32)
            pi = rr.vreinterpret(rr.vdups(0x7F800000, dtypes.uint32), dtypes.float32)
            rr.vstore(
                maximum,
                0,
                rr.vselect(ni, rr.vmuls(m, scale, mask=full), cond_mask=empty),
                heads,
            )
            rr.vstore(
                sums,
                0,
                rr.vselect(
                    pi, rr.vadd(rr.vlog(s, mask=full), b, mask=full), cond_mask=empty
                ),
                heads,
            )
            if const_expr(have_sink):
                s = rr.vadd(
                    s, rr.vexp_sub(rr.vload(self.sink), b, mask=full), mask=full
                )
            inverse = rr.vdiv(rr.vdups(1.0, dtypes.float32), s, mask=full)
            rr.vstore(
                denorm,
                0,
                rr.vselect(rr.vdups(0.0, dtypes.float32), inverse, cond_mask=empty),
                heads,
            )
        vec_release_buf(PIPE.V, 2 + job % 2)
        vec_get_buf(PIPE.MTE3, 2 + job % 2)
        mem_copy(
            tile_slice(maximum_out, (32,), (query * 2 + self.part,)),
            tile_slice(maximum, (32,), (0,)),
        )
        mem_copy(
            tile_slice(lse_out, (32,), (query * 2 + self.part,)),
            tile_slice(sums, (32,), (0,)),
        )
        vec_release_buf(PIPE.MTE3, 2 + job % 2)

    @jit
    def output_block(self, out, flags, job, query, tile, blocks, history):
        # Word 32 in each denorm bank is outside its 32 final inverse values.
        history_words = reinterpret(
            self.denorm,
            dtype=dtypes.uint32,
            layout=make_layout((128,)),
        )
        if const_expr(not bool(self.optimizations & FUSED_HISTORY_STATE)):
            vec_get_buf(PIPE.S, 2 + job % 2)
            history = history_words[(dtypes.int64(job) % 2) * 64 + 32]
            vec_release_buf(PIPE.S, 2 + job % 2)
            if tile == 0:
                history = dtypes.uint32(0)
        rescale = dtypes.uint32(0)
        if tile != 0:
            vec_sync_intra_wait(PIPE.S, RESCALE_FLAG)
            if const_expr(self.optimizations & SSBUF_UINT64):
                packed_flags = reinterpret(
                    flags,
                    dtype=dtypes.uint64,
                    layout=make_layout((64,)),
                )
                rescale = dtypes.uint32(packed_flags.vec[(job % 8) * 8 + tile % 8] != 0)
            else:
                rescale = (
                    flags.vec[job % 8, tile % 8, 0] + flags.vec[job % 8, tile % 8, 1]
                )
        if const_expr(self.optimizations & FUSED_HISTORY_STATE):
            history = dtypes.uint32(0)
            if tile != 0 and (rescale != 0 or tile + 1 == blocks):
                vec_get_buf(PIPE.S, 2 + job % 2)
                history = history_words[(dtypes.int64(job) % 2) * 64 + 32]
                vec_release_buf(PIPE.S, 2 + job % 2)
        scales = reinterpret(
            self.scales,
            layout=make_layout((64,)),
            offset=(job % 2) * 512 + (tile % 2) * 256,
        )
        if rescale != 0:
            for column in range_constexpr(4):
                vec_get_buf(PIPE.V, 4 + column)
                vec_sync_intra_wait(PIPE.V, FRAGMENT_FLAG + column % 2)
                fragment = self.fragments[column]
                if const_expr(
                    column == 3 and bool(self.optimizations & FUSED_HISTORY_STATE)
                ):
                    vec_get_buf(PIPE.V, 2 + job % 2)
                with vf():
                    full = rr.full_mask(32)
                    for row in range(32):
                        factor = rr.vload_broadcast(scales, row * 2)
                        for part in range_constexpr(2):
                            offset = row * 512 + column * 128 + part * 64
                            value = rr.vload(fragment, row * 128 + part * 64)
                            if history != dtypes.uint32(0):
                                value = rr.vadd(
                                    value, rr.vload(self.accum, offset), mask=full
                                )
                            rr.vstore(
                                self.accum,
                                offset,
                                rr.vmul(value, factor, mask=full),
                                full,
                            )
                    if const_expr(
                        column == 3 and bool(self.optimizations & FUSED_HISTORY_STATE)
                    ):
                        rr.vstore_first(
                            history_words,
                            (dtypes.int64(job) % 2) * 64 + 32,
                            rr.vdups(history + dtypes.uint32(1), dtypes.uint32),
                        )
                if const_expr(
                    column == 3 and bool(self.optimizations & FUSED_HISTORY_STATE)
                ):
                    vec_release_buf(PIPE.V, 2 + job % 2)
                vec_sync_intra_arrive(PIPE.V, FRAGMENT_FLAG + column % 2)
                vec_release_buf(PIPE.V, 4 + column)
            history = history + dtypes.uint32(1)
        if tile + 1 == blocks:
            denorm = reinterpret(
                self.denorm,
                layout=make_layout((64,)),
                offset=(job % 2) * 256,
            )
            for column in range_constexpr(4):
                vec_get_buf(PIPE.V, 4 + column)
                vec_sync_intra_wait(PIPE.V, FRAGMENT_FLAG + column % 2)
                fragment = self.fragments[column]
                final = reinterpret(
                    self.accum,
                    dtype=dtypes.bfloat16,
                    layout=make_layout((32, 128), stride=(1024, 1)),
                    offset=column * 512,
                )
                if const_expr(self.optimizations & 8):
                    history_view = reinterpret(
                        self.accum,
                        layout=make_layout((32, 128), stride=(512, 1)),
                        offset=column * 512,
                    )
                    if history != dtypes.uint32(0):
                        final_output_native(
                            fragment,
                            history_view,
                            final,
                            denorm,
                            column,
                            True,
                            bool(self.optimizations & 16),
                        )
                    else:
                        final_output_native(
                            fragment,
                            history_view,
                            final,
                            denorm,
                            column,
                            False,
                            bool(self.optimizations & 16),
                        )
                else:
                    self.output_original(fragment, final, denorm, column, history)
                vec_sync_intra_arrive(PIPE.V, FRAGMENT_FLAG + column % 2)
                vec_release_buf(PIPE.V, 4 + column)
                vec_get_buf(PIPE.MTE3, 4 + column)
                mem_copy(
                    tile_slice(out, (32, 128), (query * 2 + self.part, column)), final
                )
                vec_release_buf(PIPE.MTE3, 4 + column)
        if const_expr(not bool(self.optimizations & FUSED_HISTORY_STATE)):
            vec_get_buf(PIPE.V, 2 + job % 2)
            with vf():
                rr.vstore_first(
                    history_words,
                    (dtypes.int64(job) % 2) * 64 + 32,
                    rr.vdups(history, dtypes.uint32),
                )
            vec_release_buf(PIPE.V, 2 + job % 2)

    @jit
    def output_original(self, fragment, final, denorm, column: Constexpr[int], history):
        with vf():
            full = rr.full_mask(32)
            if const_expr(self.optimizations & 4):
                bfmask = rr.full_mask(16)
                for row in range(0, 32, 2):
                    factor0 = rr.vload_broadcast(denorm, row)
                    factor1 = rr.vload_broadcast(denorm, row + 1)
                    p00 = rr.vload(fragment, row * 128)
                    p01 = rr.vload(fragment, row * 128 + 64)
                    p10 = rr.vload(fragment, (row + 1) * 128)
                    p11 = rr.vload(fragment, (row + 1) * 128 + 64)
                    if history != dtypes.uint32(0):
                        p00 = rr.vadd(
                            p00,
                            rr.vload(self.accum, row * 512 + column * 128),
                            mask=full,
                        )
                        p01 = rr.vadd(
                            p01,
                            rr.vload(self.accum, row * 512 + column * 128 + 64),
                            mask=full,
                        )
                        p10 = rr.vadd(
                            p10,
                            rr.vload(self.accum, (row + 1) * 512 + column * 128),
                            mask=full,
                        )
                        p11 = rr.vadd(
                            p11,
                            rr.vload(self.accum, (row + 1) * 512 + column * 128 + 64),
                            mask=full,
                        )
                    s00 = rr.vcast(
                        rr.vmul(p00, factor0, mask=full),
                        dtypes.bfloat16,
                        mask=full,
                        reg_layout=RegLayout.ZERO,
                    )
                    s01 = rr.vcast(
                        rr.vmul(p01, factor0, mask=full),
                        dtypes.bfloat16,
                        mask=full,
                        reg_layout=RegLayout.ZERO,
                    )
                    s10 = rr.vcast(
                        rr.vmul(p10, factor1, mask=full),
                        dtypes.bfloat16,
                        mask=full,
                        reg_layout=RegLayout.ONE,
                    )
                    s11 = rr.vcast(
                        rr.vmul(p11, factor1, mask=full),
                        dtypes.bfloat16,
                        mask=full,
                        reg_layout=RegLayout.ONE,
                    )
                    packed0 = rr.vadd(s00, s10, mask=bfmask)
                    packed1 = rr.vadd(s01, s11, mask=bfmask)
                    output0, output1 = rr.vdeinterleave(packed0, packed1)
                    rr.vstore(final, row * 1024, output0, bfmask)
                    rr.vstore(final, (row + 1) * 1024, output1, bfmask)
            else:
                for row in range(32):
                    factor = rr.vload_broadcast(denorm, row)
                    for part in range_constexpr(2):
                        value = rr.vload(fragment, row * 128 + part * 64)
                        if history != dtypes.uint32(0):
                            value = rr.vadd(
                                value,
                                rr.vload(
                                    self.accum, row * 512 + column * 128 + part * 64
                                ),
                                mask=full,
                            )
                        value = rr.vmul(value, factor, mask=full)
                        packed = rr.vcast(value, dtypes.bfloat16, mask=full)
                        rr.vstore_pack(
                            final,
                            row * 1024 + part * 64,
                            packed,
                            full,
                            pack_mode="b32_to_b16",
                        )


class PipelineCube(DirectCube):
    def __init__(self, optimizations=0):
        DirectCube.__init__(self)
        self.optimizations = optimizations
        self.native_cursor = CubeCursor(slot=0)

    @jit
    def q_notify(self, src: Constexpr[PIPE], dst: Constexpr[PIPE], job):
        if job % 2 == 0:
            cube_sync_notify(src, dst, 0)
        else:
            cube_sync_notify(src, dst, 1)

    @jit
    def q_wait(self, src: Constexpr[PIPE], dst: Constexpr[PIPE], job):
        if job % 2 == 0:
            cube_sync_wait(src, dst, 0)
        else:
            cube_sync_wait(src, dst, 1)

    @jit
    def a_slot(self, slot):
        a = self.a[0]
        if slot == 1:
            a = self.a[1]
        if slot == 2:
            a = self.a[2]
        if slot == 3:
            a = self.a[3]
        return a

    @jit
    def slots(self, global_block):
        lo, hi = self.k_lo[0], self.k_hi[0]
        if global_block % 4 == 1:
            lo, hi = self.k_lo[1], self.k_hi[1]
        if global_block % 4 == 2:
            lo, hi = self.k_lo[2], self.k_hi[2]
        if global_block % 4 == 3:
            lo, hi = self.k_lo[3], self.k_hi[3]
        return lo, hi

    @jit
    def prefetch_query(self, q, query, job):
        qlo = self.q_lo[0]
        qhi = self.q_hi[0]
        if job % 2 != 0:
            qlo = self.q_lo[1]
            qhi = self.q_hi[1]
        self.q_wait(PIPE.MTE1, PIPE.MTE2, job)
        mem_copy(qlo, tile_slice(q, (32, 512), (query * 2, 0)))
        mem_copy(qhi, tile_slice(q, (32, 512), (query * 2 + 1, 0)))
        self.q_notify(PIPE.MTE2, PIPE.MTE1, job)

    @jit
    def qk_block(self, q, query, job, tile, blocks, global_block, destination):
        qlo = self.q_lo[0]
        qhi = self.q_hi[0]
        if job % 2 != 0:
            qlo = self.q_lo[1]
            qhi = self.q_hi[1]
        klo, khi = self.slots(global_block)
        if const_expr(not (self.optimizations & 2)):
            cube_pair_wait(PIPE.MTE1, KV_FLAG + global_block % 4)
        set_mmad_direction("m")
        cursor = self.native_cursor.slot
        for column in range_constexpr(4):
            a_index = (cursor + column) % 4
            a, b = self.a_slot(a_index), self.b[column]
            if const_expr(column == 0):
                if tile == 0:
                    self.q_wait(PIPE.MTE2, PIPE.MTE1, job)
            cube_get_buf(PIPE.MTE1, a_index)
            if const_expr(not (self.optimizations & 2)):
                cube_get_buf(PIPE.MTE1, 4 + column)
            self.load_qk_halves(a, qlo, qhi, column)
            if const_expr(self.optimizations & 2):
                cube_release_buf(PIPE.MTE1, a_index)
                if const_expr(column == 3):
                    if tile + 1 == blocks:
                        self.q_notify(PIPE.MTE1, PIPE.MTE2, job)
                if const_expr(column == 0):
                    cube_pair_wait(PIPE.MTE1, KV_FLAG + global_block % 4)
                cube_get_buf(PIPE.MTE1, 4 + column)
            self.load_qk_halves(b, klo, khi, column)
            if const_expr(not (self.optimizations & 2)):
                cube_release_buf(PIPE.MTE1, a_index)
            cube_release_buf(PIPE.MTE1, 4 + column)
            cube_get_buf(PIPE.M, a_index)
            cube_get_buf(PIPE.M, 4 + column)
            matmul(
                self.score, a, b, init=column == 0, unit_flag=3 if column == 3 else 2
            )
            cube_release_buf(PIPE.M, a_index)
            cube_release_buf(PIPE.M, 4 + column)
        if const_expr(not (self.optimizations & 2)):
            if tile + 1 == blocks:
                self.q_notify(PIPE.MTE1, PIPE.MTE2, job)
        cube_pair_wait(PIPE.FIXPIPE, SCORE_FLAG)
        if const_expr(self.optimizations & NZ_SCORE_DUAL_COPY):
            mem_copy(
                destination,
                self.score.partition(dim=0),
                copy_options(unit_flag=3, channel_split=False),
            )
        elif const_expr(self.optimizations & NZ_SCORE_TWO_COPIES):
            for subblock in range_constexpr(2):
                mem_copy(
                    destination,
                    tile_slice(self.score, (32, 64), (subblock, 0)),
                    copy_options(
                        dst_subblock=subblock, unit_flag=3, channel_split=False
                    ),
                )
        else:
            mem_copy(
                destination, self.score.partition(dim=0), copy_options(unit_flag=3)
            )
        cube_pair_arrive(PIPE.FIXPIPE, SCORE_FLAG)

    @jit
    def flush_block(self, fragments, last):
        if not last:
            cube_sync_notify(PIPE.M, PIPE.FIXPIPE, 0)
            cube_sync_wait(PIPE.M, PIPE.FIXPIPE, 0)
        for column in range_constexpr(4):
            cube_pair_wait(PIPE.FIXPIPE, FRAGMENT_FLAG + column % 2)
            mem_copy(
                fragments[column],
                self.out_fragments[column].partition(dim=0),
                copy_options(unit_flag=dtypes.int32(3) if last else dtypes.int32(0)),
            )
            cube_pair_arrive(PIPE.FIXPIPE, FRAGMENT_FLAG + column % 2)
        if not last:
            cube_sync_notify(PIPE.FIXPIPE, PIPE.M, 0)
            cube_sync_wait(PIPE.FIXPIPE, PIPE.M, 0)

    @jit
    def pv_block(self, vector, flags, job, tile, blocks, global_block):
        klo, khi = self.slots(global_block)
        rescale = dtypes.uint32(0)
        set_mmad_direction("n")
        a_index = self.native_cursor.slot
        a = reinterpret(
            self.a_slot(a_index),
            layout=make_layout((64, 64)),
            data_format="nz",
        )
        self.native_cursor.slot = dtypes.uint32((a_index + 1) % 4)
        unit = dtypes.int32(0)
        if tile == 0:
            unit = dtypes.int32(2)
        if tile + 1 == blocks:
            unit = dtypes.int32(3)
        for column in range_constexpr(4):
            b = reinterpret(
                self.b[column],
                layout=make_layout((128, 64)),
                data_format="nz",
            )
            cube_get_buf(PIPE.MTE1, 4 + column)
            mem_copy(
                reinterpret(b, layout=make_layout((128, 32)), data_format="nz"),
                tile_slice(klo, (32, 128), (0, column)).permute(1, 0),
            )
            mem_copy(
                reinterpret(
                    b,
                    layout=make_layout((128, 32)),
                    offset=8192,
                    data_format="nz",
                ),
                tile_slice(khi, (32, 128), (0, column)).permute(1, 0),
            )
            cube_release_buf(PIPE.MTE1, 4 + column)
            if const_expr(column == 0):
                cube_pair_wait(PIPE.MTE1, S_FLAG)
                cube_get_buf(PIPE.MTE1, a_index)
                mem_copy(a, self.s)
                cube_release_buf(PIPE.MTE1, a_index)
                cube_pair_arrive(PIPE.MTE1, S_FLAG)
                cube_get_buf(PIPE.M, a_index)
                if tile != 0:
                    cube_pair_wait(PIPE.S, RESCALE_FLAG)
                    if const_expr(self.optimizations & SSBUF_EARLY_NOTIFY):
                        cube_pair_arrive(PIPE.S, RESCALE_FLAG)
                    if const_expr(self.optimizations & SSBUF_UINT64):
                        packed_flags = reinterpret(
                            flags, dtype=dtypes.uint64, layout=make_layout((64,))
                        )
                        rescale = dtypes.uint32(
                            packed_flags.cube[(job % 8) * 8 + tile % 8] != 0
                        )
                    else:
                        rescale = (
                            flags.cube[job % 8, tile % 8, 0]
                            + flags.cube[job % 8, tile % 8, 1]
                        )
                    if const_expr(not bool(self.optimizations & SSBUF_EARLY_NOTIFY)):
                        cube_pair_arrive(PIPE.S, RESCALE_FLAG)
                    if rescale != 0:
                        self.flush_block(vector.fragments, False)
            if const_expr(column == 3):
                cube_pair_arrive(PIPE.MTE1, KV_FLAG + global_block % 4)
            cube_get_buf(PIPE.M, 4 + column)
            matmul(
                tile_slice(self.out, (64, 128), (0, column)),
                a,
                b,
                init=tile == 0 or rescale != 0,
                unit_flag=unit,
            )
            cube_release_buf(PIPE.M, 4 + column)
        cube_release_buf(PIPE.M, a_index)
        if tile + 1 == blocks:
            self.flush_block(vector.fragments, True)


def schedule_scalar_dtype(optimizations):
    return dtypes.uint32 if optimizations & UNSIGNED_SCHEDULE else dtypes.int64


def cube_schedule_scalar_dtype(optimizations):
    return (
        dtypes.uint32
        if optimizations & (UNSIGNED_SCHEDULE | CUBE_UNSIGNED_SCHEDULE)
        else dtypes.int64
    )


# fused_attention_pipeline: production definitions.


@jit
def final_output_fp32(fragment, final, denorm, have_history: Constexpr[bool]):
    with vf():
        full = rr.full_mask(32)
        for row in range(0, 32, 2):
            factor0 = rr.vload_broadcast(denorm, row)
            factor1 = rr.vload_broadcast(denorm, row + 1)
            p00 = rr.vload(fragment, row * 128)
            p01 = rr.vload(fragment, row * 128 + 64)
            p10 = rr.vload(fragment, (row + 1) * 128)
            p11 = rr.vload(fragment, (row + 1) * 128 + 64)
            if const_expr(have_history):
                p00 = rr.vadd(p00, rr.vload(final, row * 512), mask=full)
                p01 = rr.vadd(p01, rr.vload(final, row * 512 + 64), mask=full)
                p10 = rr.vadd(p10, rr.vload(final, (row + 1) * 512), mask=full)
                p11 = rr.vadd(p11, rr.vload(final, (row + 1) * 512 + 64), mask=full)
            rr.vstore(final, row * 512, rr.vmul(p00, factor0, mask=full), full)
            rr.vstore(final, row * 512 + 64, rr.vmul(p01, factor0, mask=full), full)
            rr.vstore(final, (row + 1) * 512, rr.vmul(p10, factor1, mask=full), full)
            rr.vstore(
                final, (row + 1) * 512 + 64, rr.vmul(p11, factor1, mask=full), full
            )


class FusedVector(PipelineVector):
    def __init__(self, have_sink, optimizations, score_mode, score_scale, multi_group):
        PipelineVector.__init__(self, have_sink, True, optimizations)
        self.score_mode = score_mode  # 0=positive raw; 1=per-head norm; 2=scalar
        self.score_scale = score_scale
        self.multi_group = multi_group
        self.fuse_score_norm = False

    @jit
    def fetch_group(self, indices, query, length, tile):
        vec_get_buf(PIPE.MTE2, 0)
        start = dtypes.int64(tile // INDEX_TILES_PER_GROUP) * INDEX_GROUP_SIZE
        count = min(max(dtypes.int64(length) - start, 0), INDEX_GROUP_SIZE)
        if count > 0:
            raw = reinterpret(
                self.indices,
                dtype=dtypes.int32,
                layout=make_layout((1, INDEX_GROUP_SIZE)),
                data_format="nd",
            )
            mem_copy(
                raw[0:1, 0:count], indices[query : query + 1, start : start + count]
            )
        vec_release_buf(PIPE.MTE2, 0)

    @jit
    def prescale(self, alpha, query):
        for group in range(8):
            a0 = self.score_scale
            a1 = self.score_scale
            a2 = self.score_scale
            a3 = self.score_scale
            if const_expr(self.score_mode == 1):
                offset = (
                    dtypes.int64(query) * 64 + dtypes.int64(self.part) * 32 + group * 4
                )
                a0 = alpha[offset]
                a1 = alpha[offset + 1]
                a2 = alpha[offset + 2]
                a3 = alpha[offset + 3]
            with vf():
                full = rr.full_mask(32)
                heads = rr.vshr(rr.varange(0, dtypes.uint32), 4, mask=full)
                factor = rr.vdups(a3, dtypes.float32)
                factor = rr.vselect(
                    rr.vdups(a2, dtypes.float32),
                    factor,
                    cond_mask=rr.veqs(heads, 2, mask=full),
                )
                factor = rr.vselect(
                    rr.vdups(a1, dtypes.float32),
                    factor,
                    cond_mask=rr.veqs(heads, 1, mask=full),
                )
                factor = rr.vselect(
                    rr.vdups(a0, dtypes.float32),
                    factor,
                    cond_mask=rr.veqs(heads, 0, mask=full),
                )
                for column in range_constexpr(4):
                    offset = group * 64 + column * 512
                    value = rr.vload(self.score, offset)
                    rr.vstore(
                        self.score, offset, rr.vmul(value, factor, mask=full), full
                    )
                rr.vmem_bar("vst_vld")

    @jit
    def softmax_block_fused(self, alpha, query, job, tile, index_group, scale):
        if tile == 0:
            self.softmax_fused(alpha, query, job, tile, index_group, scale, 1)
        else:
            self.softmax_fused(alpha, query, job, tile, index_group, scale, 0)

    @jit
    def softmax_fused(
        self, alpha, query, job, tile, index_group, scale, first_tile: Constexpr[int]
    ):
        mask_tile = tile
        if const_expr(self.multi_group):
            mask_tile = tile % INDEX_TILES_PER_GROUP
        maximum_buf = reinterpret(
            self.maximum,
            layout=make_layout((64,)),
            offset=(job % 2) * 256,
        )
        base_buf = reinterpret(
            self.base,
            layout=make_layout((64,)),
            offset=(job % 2) * 256,
        )
        sum_buf = reinterpret(
            self.sum,
            layout=make_layout((64,)),
            offset=(job % 2) * 256,
        )
        scales_buf = reinterpret(
            self.scales,
            layout=make_layout((64,)),
            offset=(job % 2) * 512 + (tile % 2) * 256,
        )
        history_words = reinterpret(
            self.denorm,
            dtype=dtypes.uint32,
            layout=make_layout((128,)),
        )
        vec_sync_intra_wait(PIPE.V, SCORE_FLAG)
        vec_get_buf(PIPE.V, 2 + job % 2)
        if const_expr(not bool(self.optimizations & NZ_SCORE_PATHS)):
            nd_score_to_native_nz(self.score)
        if const_expr(self.score_mode != 0 and not self.fuse_score_norm):
            self.prescale(alpha, query)
        with vf():
            if const_expr(self.optimizations & SM_LOCAL_VIEWS):
                job32 = dtypes.uint32(dtypes.int64(job))
                tile32 = dtypes.uint32(dtypes.int64(tile))
                maximum_buf = reinterpret(
                    self.maximum, layout=make_layout((64,)), offset=(job32 % 2) * 256
                )
                base_buf = reinterpret(
                    self.base,
                    layout=make_layout((64,)),
                    offset=(job32 % 2) * 256,
                )
                sum_buf = reinterpret(
                    self.sum,
                    layout=make_layout((64,)),
                    offset=(job32 % 2) * 256,
                )
                scales_buf = reinterpret(
                    self.scales,
                    layout=make_layout((64,)),
                    offset=(job32 % 2) * 512 + (tile32 % 2) * 256,
                )
                history_words = reinterpret(
                    self.denorm, dtype=dtypes.uint32, layout=make_layout((128,))
                )
            if const_expr(self.optimizations & NZ_SHARED_FULL_MASK):
                # Share one all-active b16 predicate between FP32 and BF16 operations.
                bfmask = rr.full_mask(16)
                # The physical predicate is also all-active for FP32.  Mask
                # describes the consumer granularity without another PSET.
                full = rr.Mask(bfmask, 32)
            else:
                full = rr.full_mask(32)
                bfmask = rr.full_mask(16)
            eight = rr.create_mask("vl8", 32)
            neg = rr.vdups(-3.4028234663852886e38, dtypes.float32)
            zero = rr.vdups(0.0, dtypes.float32)
            if const_expr(first_tile < 0):
                if dtypes.int64(tile) == 0:
                    rr.vstore(maximum_buf, 0, neg, full)
                    rr.vstore(base_buf, 0, neg, full)
                    rr.vstore(sum_buf, 0, zero, full)
                    if const_expr(self.optimizations & FUSED_HISTORY_STATE):
                        rr.vstore_first(
                            history_words,
                            (dtypes.int64(job) % 2) * 64 + 32,
                            rr.vdups(0, dtypes.uint32),
                        )
                    rr.vmem_bar("vst_vld")
            elif const_expr(first_tile == 1):
                if const_expr(self.optimizations & FUSED_HISTORY_STATE):
                    rr.vstore_first(
                        history_words,
                        (dtypes.int64(job) % 2) * 64 + 32,
                        rr.vdups(0, dtypes.uint32),
                    )
            validity = load_index_validity(self.validity, index_group, mask_tile)
            lanes = rr.varange(0, dtypes.uint32)
            columns = rr.vbitwise_and(lanes, rr.vdups(15, dtypes.uint32), mask=full)
            mask0 = rr.vgts(rr.vgather_reg(validity, columns), 0, mask=full)
            mask1 = rr.vgts(
                rr.vgather_reg(validity, rr.vadds(columns, 16, mask=full)), 0, mask=full
            )
            mask2 = rr.vgts(
                rr.vgather_reg(validity, rr.vadds(columns, 32, mask=full)), 0, mask=full
            )
            mask3 = rr.vgts(
                rr.vgather_reg(validity, rr.vadds(columns, 48, mask=full)), 0, mask=full
            )
            old_sum = zero
            if const_expr(bool(self.optimizations & SM_EARLY_SUM) and first_tile != 1):
                old_sum = rr.vload(sum_buf)
                rr.vmem_bar("vld_vst")
            if const_expr(self.fuse_score_norm):
                norm = reinterpret(
                    self.denorm,
                    layout=make_layout((64,)),
                    offset=(dtypes.int64(job) % 2) * 256,
                )
                norm_coefficients = rr.vload(norm)
                norm_heads = rr.vshr(rr.varange(0, dtypes.uint32), 4, mask=full)
            for group in range(8):
                if const_expr(first_tile == 1):
                    current = neg
                else:
                    current = rr.vreinterpret(
                        rr.vload_broadcast(
                            maximum_buf, group * 8, mode="elem2datablock"
                        ),
                        dtypes.float32,
                    )
                if const_expr(self.fuse_score_norm):
                    factor = rr.vgather_reg(
                        norm_coefficients, rr.vadds(norm_heads, group * 4, mask=full)
                    )
                    p0 = rr.vmul(rr.vload(self.score, group * 64), factor, mask=full)
                    p1 = rr.vmul(
                        rr.vload(self.score, group * 64 + 512), factor, mask=full
                    )
                    p2 = rr.vmul(
                        rr.vload(self.score, group * 64 + 1024), factor, mask=full
                    )
                    p3 = rr.vmul(
                        rr.vload(self.score, group * 64 + 1536), factor, mask=full
                    )
                    rr.vstore(self.score, group * 64, p0, full)
                    rr.vstore(self.score, group * 64 + 512, p1, full)
                    rr.vstore(self.score, group * 64 + 1024, p2, full)
                    rr.vstore(self.score, group * 64 + 1536, p3, full)
                    p0 = rr.vselect(p0, neg, cond_mask=mask0)
                    p1 = rr.vselect(p1, neg, cond_mask=mask1)
                    p2 = rr.vselect(p2, neg, cond_mask=mask2)
                    p3 = rr.vselect(p3, neg, cond_mask=mask3)
                elif const_expr(self.optimizations & NZ_MAXIMUM_CURSORS):
                    p0 = rr.vselect(
                        rr.vload(self.maximum_cols[0], 64, post_update=True),
                        neg,
                        cond_mask=mask0,
                    )
                    p1 = rr.vselect(
                        rr.vload(self.maximum_cols[1], 64, post_update=True),
                        neg,
                        cond_mask=mask1,
                    )
                    p2 = rr.vselect(
                        rr.vload(self.maximum_cols[2], 64, post_update=True),
                        neg,
                        cond_mask=mask2,
                    )
                    p3 = rr.vselect(
                        rr.vload(self.maximum_cols[3], 64, post_update=True),
                        neg,
                        cond_mask=mask3,
                    )
                else:
                    p0 = rr.vselect(
                        rr.vload(self.score, group * 64), neg, cond_mask=mask0
                    )
                    p1 = rr.vselect(
                        rr.vload(self.score, group * 64 + 512), neg, cond_mask=mask1
                    )
                    p2 = rr.vselect(
                        rr.vload(self.score, group * 64 + 1024), neg, cond_mask=mask2
                    )
                    p3 = rr.vselect(
                        rr.vload(self.score, group * 64 + 1536), neg, cond_mask=mask3
                    )
                current = rr.vmax(current, p0, mask=full)
                current = rr.vmax(current, p1, mask=full)
                current = rr.vmax(current, p2, mask=full)
                current = rr.vmax(current, p3, mask=full)
                current = rr.vreduce_max_datablock(current, mask=full)
                lo, hi = rr.vdeinterleave(current, current)
                current = rr.vmax(lo, hi, mask=full)
                repeated, unused = rr.vinterleave(current, current)
                rr.vstore(maximum_buf, group * 8, repeated, eight)
            rr.vmem_bar("vst_vld")
            maximum = rr.vload(maximum_buf)
            if const_expr(first_tile == 1):
                # The first tile initializes state without history or rescaling.
                rr.vstore(base_buf, 0, maximum, full)
            else:
                old_base = rr.vload(base_buf)
                difference = rr.vsub(maximum, old_base, mask=full)
                flag = rr.vdup(rr.vreduce_max(difference, mask=full), mask=full)
                update = rr.vgts(flag, 6.0 / scale, mask=full)
                if const_expr(first_tile < 0):
                    first = rr.veqs(
                        rr.vdups(dtypes.int64(tile), dtypes.int32), 0, mask=full
                    )
                    flag = rr.vselect(zero, flag, cond_mask=first)
                    update = rr.mask_or(first, update, exec_mask=full)
                new_base = rr.vselect(maximum, old_base, cond_mask=update)
                rr.vstore(base_buf, 0, new_base, full)
                if const_expr(self.fuse_score_norm):
                    # Norm scores already include per-head scale. The
                    # Softmax scale is compile-time 1.0 on this path.
                    factors = rr.vexp_sub(old_base, new_base, mask=full)
                else:
                    factors = rr.vexp_sub(
                        rr.vmuls(old_base, scale, mask=full),
                        rr.vmuls(new_base, scale, mask=full),
                        mask=full,
                    )
                if const_expr(first_tile < 0):
                    factors = rr.vselect(
                        rr.vdups(1.0, dtypes.float32), factors, cond_mask=first
                    )
                rr.vstore(scales_buf, 0, factors, full)
                rr.vstore_first(
                    self.delta,
                    (dtypes.int64(job) % 2) * 2 + dtypes.int64(tile) % 2,
                    flag,
                )
                if const_expr(not bool(self.optimizations & SM_EARLY_SUM)):
                    old_sum = rr.vload(sum_buf)
            rr.vmem_bar("vst_vld")
            for start_row in range(0, 32, 8):
                base0 = rr.vreinterpret(
                    rr.vload_broadcast(base_buf, start_row * 2, mode="elem2datablock"),
                    dtypes.float32,
                )
                base1 = rr.vreinterpret(
                    rr.vload_broadcast(
                        base_buf, start_row * 2 + 8, mode="elem2datablock"
                    ),
                    dtypes.float32,
                )
                if const_expr(not self.fuse_score_norm):
                    base0 = rr.vmuls(base0, scale, mask=full)
                    base1 = rr.vmuls(base1, scale, mask=full)
                sum0, sum1 = zero, zero
                for pair in range_constexpr(2):
                    offset = start_row * 16 + pair * 1024
                    if const_expr(self.fuse_score_norm):
                        p00 = rr.vload(self.score, offset)
                        p10 = rr.vload(self.score, offset + 64)
                        p01 = rr.vload(self.score, offset + 512)
                        p11 = rr.vload(self.score, offset + 576)
                    else:
                        p00 = rr.vmuls(rr.vload(self.score, offset), scale, mask=full)
                        p10 = rr.vmuls(
                            rr.vload(self.score, offset + 64), scale, mask=full
                        )
                        p01 = rr.vmuls(
                            rr.vload(self.score, offset + 512), scale, mask=full
                        )
                        p11 = rr.vmuls(
                            rr.vload(self.score, offset + 576), scale, mask=full
                        )
                    if const_expr(pair == 0):
                        valid0, valid1 = mask0, mask1
                    else:
                        valid0, valid1 = mask2, mask3
                    if const_expr(self.optimizations & NZ_MASKED_EXP):
                        p00 = rr.vexp_sub(p00, base0, mask=valid0)
                        p10 = rr.vexp_sub(p10, base1, mask=valid0)
                        p01 = rr.vexp_sub(p01, base0, mask=valid1)
                        p11 = rr.vexp_sub(p11, base1, mask=valid1)
                    else:
                        p00 = rr.vselect(
                            rr.vexp_sub(p00, base0, mask=full), zero, cond_mask=valid0
                        )
                        p10 = rr.vselect(
                            rr.vexp_sub(p10, base1, mask=full), zero, cond_mask=valid0
                        )
                        p01 = rr.vselect(
                            rr.vexp_sub(p01, base0, mask=full), zero, cond_mask=valid1
                        )
                        p11 = rr.vselect(
                            rr.vexp_sub(p11, base1, mask=full), zero, cond_mask=valid1
                        )
                    if const_expr(
                        pair == 0 and bool(self.optimizations & NZ_SUM_FIRST_VALUE)
                    ):
                        sum0 = p00
                    else:
                        sum0 = rr.vadd(sum0, p00, mask=full)
                    sum0 = rr.vadd(sum0, p01, mask=full)
                    if const_expr(
                        pair == 0 and bool(self.optimizations & NZ_SUM_FIRST_VALUE)
                    ):
                        sum1 = p10
                    else:
                        sum1 = rr.vadd(sum1, p10, mask=full)
                    sum1 = rr.vadd(sum1, p11, mask=full)
                    s00 = rr.vcast(
                        p00, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ZERO
                    )
                    s10 = rr.vcast(
                        p10, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ZERO
                    )
                    s01 = rr.vcast(
                        p01, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ONE
                    )
                    s11 = rr.vcast(
                        p11, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ONE
                    )
                    packed0 = rr.vadd(s00, s01, mask=bfmask)
                    packed1 = rr.vadd(s10, s11, mask=bfmask)
                    out0, out1 = rr.vdeinterleave(packed0, packed1)
                    rr.vstore(self.prob, start_row * 16 + pair * 2048, out0, bfmask)
                    rr.vstore(
                        self.prob, start_row * 16 + pair * 2048 + 1024, out1, bfmask
                    )
                sum0 = rr.vpair_reduce_sum(
                    rr.vreduce_sum_datablock(sum0, mask=full), mask=full
                )
                sum1 = rr.vpair_reduce_sum(
                    rr.vreduce_sum_datablock(sum1, mask=full), mask=full
                )
                repeat0, unused = rr.vinterleave(sum0, sum0)
                repeat1, unused = rr.vinterleave(sum1, sum1)
                rr.vstore(sum_buf, start_row * 2, repeat0, eight)
                rr.vstore(sum_buf, start_row * 2 + 8, repeat1, eight)
            if const_expr(first_tile != 1):
                rr.vmem_bar("vst_vld")
                if const_expr(
                    self.optimizations & (NZ_SCORE_DUAL_COPY | NATIVE_SUM_MADD)
                ):
                    total = rr.vmadd(old_sum, factors, rr.vload(sum_buf), mask=full)
                else:
                    total = rr.vadd(
                        rr.vmul(old_sum, factors, mask=full),
                        rr.vload(sum_buf),
                        mask=full,
                    )
                rr.vstore(sum_buf, 0, total, full)

        vec_release_buf(PIPE.V, 2 + job % 2)
        vec_sync_notify(PIPE.V, PIPE.MTE3, 0)
        if tile != 0:
            vec_sync_notify(PIPE.V, PIPE.S, 0)

    @jit
    def output_block(self, out, flags, job, query, tile, blocks, history):
        # Word 32 in each denorm bank is outside its 32 final inverse values.
        history_words = reinterpret(
            self.denorm,
            dtype=dtypes.uint32,
            layout=make_layout((128,)),
        )
        if const_expr(not bool(self.optimizations & FUSED_HISTORY_STATE)):
            vec_get_buf(PIPE.S, 2 + job % 2)
            history = history_words[(dtypes.int64(job) % 2) * 64 + 32]
            vec_release_buf(PIPE.S, 2 + job % 2)
            if tile == 0:
                history = dtypes.uint32(0)
        rescale = dtypes.uint32(0)
        if tile != 0:
            vec_sync_intra_wait(PIPE.S, RESCALE_FLAG)
            if const_expr(self.optimizations & SSBUF_UINT64):
                packed_flags = reinterpret(
                    flags,
                    dtype=dtypes.uint64,
                    layout=make_layout((64,)),
                )
                rescale = dtypes.uint32(packed_flags.vec[(job % 8) * 8 + tile % 8] != 0)
            else:
                rescale = (
                    flags.vec[job % 8, tile % 8, 0] + flags.vec[job % 8, tile % 8, 1]
                )
        if const_expr(self.optimizations & FUSED_HISTORY_STATE):
            history = dtypes.uint32(0)
            if tile != 0 and (rescale != 0 or tile + 1 == blocks):
                vec_get_buf(PIPE.S, 2 + job % 2)
                history = history_words[(dtypes.int64(job) % 2) * 64 + 32]
                vec_release_buf(PIPE.S, 2 + job % 2)
        scales = reinterpret(
            self.scales,
            layout=make_layout((64,)),
            offset=(job % 2) * 512 + (tile % 2) * 256,
        )
        if rescale != 0:
            for column in range_constexpr(4):
                vec_get_buf(PIPE.V, 4 + column)
                vec_sync_intra_wait(PIPE.V, FRAGMENT_FLAG + column % 2)
                fragment = self.fragments[column]
                if const_expr(
                    column == 3 and bool(self.optimizations & FUSED_HISTORY_STATE)
                ):
                    vec_get_buf(PIPE.V, 2 + job % 2)
                with vf():
                    full = rr.full_mask(32)
                    for row in range(32):
                        factor = rr.vload_broadcast(scales, row * 2)
                        for part in range_constexpr(2):
                            offset = row * 512 + column * 128 + part * 64
                            value = rr.vload(fragment, row * 128 + part * 64)
                            if history != dtypes.uint32(0):
                                value = rr.vadd(
                                    value, rr.vload(self.accum, offset), mask=full
                                )
                            rr.vstore(
                                self.accum,
                                offset,
                                rr.vmul(value, factor, mask=full),
                                full,
                            )
                    if const_expr(
                        column == 3 and bool(self.optimizations & FUSED_HISTORY_STATE)
                    ):
                        rr.vstore_first(
                            history_words,
                            (dtypes.int64(job) % 2) * 64 + 32,
                            rr.vdups(history + dtypes.uint32(1), dtypes.uint32),
                        )
                if const_expr(
                    column == 3 and bool(self.optimizations & FUSED_HISTORY_STATE)
                ):
                    vec_release_buf(PIPE.V, 2 + job % 2)
                vec_sync_intra_arrive(PIPE.V, FRAGMENT_FLAG + column % 2)
                vec_release_buf(PIPE.V, 4 + column)
            history = history + dtypes.uint32(1)
        if tile + 1 == blocks:
            denorm = reinterpret(
                self.denorm,
                layout=make_layout((64,)),
                offset=(job % 2) * 256,
            )
            for column in range_constexpr(4):
                vec_get_buf(PIPE.V, 4 + column)
                vec_sync_intra_wait(PIPE.V, FRAGMENT_FLAG + column % 2)
                fragment = self.fragments[column]
                final = reinterpret(
                    self.accum,
                    dtype=dtypes.float32,
                    layout=make_layout((32, 128), stride=(512, 1)),
                    offset=column * 512,
                )
                if history != dtypes.uint32(0):
                    final_output_fp32(fragment, final, denorm, True)
                else:
                    final_output_fp32(fragment, final, denorm, False)
                vec_sync_intra_arrive(PIPE.V, FRAGMENT_FLAG + column % 2)
                vec_release_buf(PIPE.V, 4 + column)
                vec_get_buf(PIPE.MTE3, 4 + column)
                mem_copy(
                    tile_slice(out, (32, 128), (query * 2 + self.part, column)), final
                )
                vec_release_buf(PIPE.MTE3, 4 + column)
        if const_expr(not bool(self.optimizations & FUSED_HISTORY_STATE)):
            vec_get_buf(PIPE.V, 2 + job % 2)
            with vf():
                rr.vstore_first(
                    history_words,
                    (dtypes.int64(job) % 2) * 64 + 32,
                    rr.vdups(history, dtypes.uint32),
                )
            vec_release_buf(PIPE.V, 2 + job % 2)


@datastruct(c_name="FusedPrefillGroupMetadata")
class GroupMetadata:
    head: dtypes.uint32
    active: CArray[dtypes.uint32, 6]
    job: CArray[dtypes.uint32, 6]
    query: CArray[dtypes.uint32, 6]
    tile: CArray[dtypes.uint32, 6]
    blocks: CArray[dtypes.uint32, 6]
    length: CArray[dtypes.uint32, 6]
    global_block: CArray[dtypes.uint32, 6]
    index_group: CArray[dtypes.uint32, 6]


class GroupDelayLine:
    def __init__(self):
        self.state = GroupMetadata(
            head=0,
            active=[0] * 6,
            job=[0] * 6,
            query=[0] * 6,
            tile=[0] * 6,
            blocks=[0] * 6,
            length=[0] * 6,
            global_block=[0] * 6,
            index_group=[0] * 6,
        )

    def push(
        self, *, active, job, query, tile, blocks, length, global_block, index_group
    ):
        slot = self.state.head
        self.state.active[slot] = dtypes.uint32(active)
        self.state.job[slot] = dtypes.uint32(job)
        self.state.query[slot] = dtypes.uint32(query)
        self.state.tile[slot] = dtypes.uint32(tile)
        self.state.blocks[slot] = dtypes.uint32(blocks)
        self.state.length[slot] = dtypes.uint32(length)
        self.state.global_block[slot] = dtypes.uint32(global_block)
        self.state.index_group[slot] = dtypes.uint32(index_group)

    @jit
    def tap(self, lag):
        slot = self.state.head + dtypes.uint32(6 - lag)
        if slot >= 6:
            slot = slot - dtypes.uint32(6)
        return SimpleNamespace(
            active=self.state.active[slot],
            job=self.state.job[slot],
            query=dtypes.int64(self.state.query[slot]),
            tile=self.state.tile[slot],
            blocks=self.state.blocks[slot],
            length=self.state.length[slot],
            global_block=self.state.global_block[slot],
            index_group=self.state.index_group[slot],
        )

    @jit
    def advance(self):
        next_head = self.state.head + dtypes.uint32(1)
        if next_head == 6:
            next_head = dtypes.uint32(0)
        self.state.head = next_head


# single_kernel_ops: production definitions.


@jit
def pack_sf_words(codes, words):
    with vf():
        full = rr.full_mask(32)
        low = rr.create_mask("vl32", 32)
        ids = rr.varange(0, dtypes.uint32)
        group = rr.vshr(ids, 3, mask=full)
        local_word = rr.vbitwise_and(ids, rr.vdups(7, dtypes.uint32), mask=full)
        block = rr.vshr(local_word, 1, mask=full)
        half = rr.vbitwise_and(local_word, rr.vdups(1, dtypes.uint32), mask=full)
        base = rr.vadd(
            rr.vshl(group, 5, mask=full), rr.vshl(half, 4, mask=full), mask=full
        )
        base = rr.vadd(base, block, mask=full)
        word = rr.vreinterpret(rr.vgather(codes, base, mask=low), dtypes.uint32)
        for byte in range_constexpr(1, 4):
            value = rr.vreinterpret(
                rr.vgather(codes, rr.vadds(base, byte * 4, mask=full), mask=low),
                dtypes.uint32,
            )
            word = rr.vbitwise_or(word, rr.vshl(value, byte * 8, mask=full), mask=full)
        # One aligned UB source per exact 4-byte GM burst.
        rr.vscatter(
            words,
            rr.vreinterpret(word, dtypes.int32),
            rr.vshl(ids, 3, mask=full),
            mask=low,
        )
        rr.vstore(words, 256, rr.vdups(0, dtypes.int32), full)


@jit
def dma_sf_store(words, sf, query, head_tile, part, column, s: Constexpr[int]):
    aligned_s = (s + 3) // 4 * 4
    target = sf.view((-1, aligned_s))
    for group in range_constexpr(4):
        g = head_tile * 8 + part * 4 + group
        mem_copy(
            tile_slice(target, (8, 1), (g * 4 + column, query)),
            reinterpret(
                words,
                layout=make_layout((8, 1), stride=(8, 1)),
                offset=group * 256,
                data_format="nd",
            ),
        )
        if const_expr(aligned_s > s):
            if query + 1 == s:
                mem_copy(
                    target[g * 32 + column * 8 : g * 32 + column * 8 + 8, s:aligned_s],
                    reinterpret(
                        words,
                        layout=make_layout((8, aligned_s - s), stride=(8, 1)),
                        offset=1024,
                        data_format="nd",
                    ),
                )


@jit
def rope_prepared_register(
    value, cosine, sine, negative_sine, even_ids, odd_ids, even_mask, full
):
    even = rr.vgather_reg(value, even_ids)
    odd = rr.vgather_reg(value, odd_ids)
    p0 = rr.vmul(odd, negative_sine, mask=full)
    p1 = rr.vmul(odd, cosine, mask=full)
    y0 = rr.vmadd(even, cosine, p0, mask=full)
    y1 = rr.vmadd(even, sine, p1, mask=full)
    return rr.vselect(y0, y1, cond_mask=even_mask)


# block_pair_quant: production definitions.


@jit
def quantize_block_pair(value, full, low, upper_ids):
    absolute = rr.vabs(value, mask=full)
    lower = rr.vdup(rr.vreduce_max(absolute, mask=low), mask=full)
    upper = rr.vdup(
        rr.vreduce_max(rr.vgather_reg(absolute, upper_ids), mask=low), mask=full
    )
    maximum = rr.vselect(lower, upper, cond_mask=low)
    raw = rr.vmuls(maximum, 1.0 / 448, mask=full)
    bits = rr.vreinterpret(raw, dtypes.uint32)
    # Ceil exponent of a nonnegative IEEE FP32 value. Carry into the
    # exponent occurs exactly for a nonzero mantissa.
    code = rr.vshr(rr.vadds(bits, 0x7FFFFF, mask=full), 23, mask=full)
    code = rr.vmaxs(code, 105, mask=full)
    inverse = rr.vreinterpret(
        rr.vshl(rr.vsub(rr.vdups(254, dtypes.uint32), code, mask=full), 23, mask=full),
        dtypes.float32,
    )
    # The upward-rounded power-of-two scale guarantees abs(scaled)<=448
    # for finite input. Conversion also keeps its saturation setting.
    scaled = rr.vmul(value, inverse, mask=full)
    packed = rr.vcast(
        scaled,
        dtypes.float8_e4m3fn,
        mask=full,
        rounding=rr.RoundingMode.RN,
        reg_layout=rr.RegLayout.ZERO,
        saturate=True,
    )
    # FP8 ZERO uses a 32-bit carrier lane; gather carriers before casting
    # the view back to FP8. The two output blocks have different UB offsets.
    upper_packed = rr.vreinterpret_lanes(
        rr.vgather_reg(rr.vreinterpret_lanes(packed, dtypes.uint32), upper_ids),
        dtypes.float8_e4m3fn,
    )
    return packed, upper_packed, code, rr.vgather_reg(code, upper_ids)


# fused_index_preprocess: production definitions.


@jit
def preprocess_index_prefix(self, length, job, k: Constexpr[int]):
    # Four index-mask groups protect older softmax consumers from the
    # next query's raw-index DMA. Sort each pair into ascending order.
    vec_get_buf(PIPE.V, 0)
    vec_get_buf(PIPE.V, 1)
    length32 = dtypes.int32(length)
    with vf():
        full = rr.full_mask(32)
        invalid = rr.vdups(self.kv_count, dtypes.uint32)
        stride = rr.vdups(1024, dtypes.uint32)
        address_words = reinterpret(
            self.addresses,
            dtype=dtypes.uint32,
            layout=make_layout((2 * INDEX_PAIRS_PER_GROUP,)),
        )
        stride_words = reinterpret(
            self.strides,
            dtype=dtypes.uint32,
            layout=make_layout((2 * INDEX_PAIRS_PER_GROUP,)),
        )
        # The frontend checks region assignments before pruning constexpr
        # alternatives. The ordinary-load branch overwrites both values.
        raw0, raw1 = invalid, invalid
        if const_expr(self.optimizations & NATIVE_INDEX_PREPROCESS):
            # Hoist the pair positions for the capacity prefix;
            # keep one 128-index group even when the row is empty.
            positions_even, positions_odd = rr.vdeinterleave(
                rr.varange(0, dtypes.int32), rr.varange(64, dtypes.int32)
            )
        if const_expr(self.optimizations & INDEX_LOAD_DEINTERLEAVE):
            positions_even = rr.vshl(rr.varange(0, dtypes.int32), 1, mask=full)
            positions_odd = rr.vadds(positions_even, 1, mask=full)
        for start_index in range(
            0, max(128, min(INDEX_GROUP_SIZE, ((k + 127) // 128) * 128)), 128
        ):
            start = start_index
            if const_expr(self.optimizations & UNSIGNED_SCHEDULE):
                start = dtypes.uint32(start_index)
            if const_expr(self.optimizations & INDEX_LOAD_DEINTERLEAVE):
                even, odd = rr.vload_deinterleave(self.indices, start)
                even = rr.vreinterpret(even, dtypes.uint32)
                odd = rr.vreinterpret(odd, dtypes.uint32)
                remaining = length32 - dtypes.int32(start)
                prefix_even = rr.vlts(positions_even, remaining, mask=full)
                prefix_odd = rr.vlts(positions_odd, remaining, mask=full)
                even = rr.vselect(even, invalid, cond_mask=prefix_even)
                odd = rr.vselect(odd, invalid, cond_mask=prefix_odd)
            else:
                raw0 = rr.vreinterpret(rr.vload(self.indices, start), dtypes.uint32)
                raw1 = rr.vreinterpret(
                    rr.vload(self.indices, start + 64), dtypes.uint32
                )
                even, odd = rr.vdeinterleave(raw0, raw1)
            if const_expr(self.optimizations & INDEX_LOAD_DEINTERLEAVE):
                pass
            elif const_expr(self.optimizations & NATIVE_INDEX_PREPROCESS):
                remaining = length - start
                if const_expr(self.optimizations & INDEX_NARROW_LENGTH):
                    remaining = length32 - dtypes.int32(start)
                if remaining < 128:
                    invalid_even = rr.vges(positions_even, remaining, mask=full)
                    invalid_odd = rr.vges(positions_odd, remaining, mask=full)
                    even = rr.vdups(
                        self.kv_count,
                        dtypes.uint32,
                        mask=invalid_even,
                        mode="merging",
                        merge=even,
                    )
                    odd = rr.vdups(
                        self.kv_count,
                        dtypes.uint32,
                        mask=invalid_odd,
                        mode="merging",
                        merge=odd,
                    )
            elif const_expr(self.optimizations & NATIVE_INDEX_MASKING):
                if length - start < 128:
                    invalid0 = rr.vges(
                        rr.varange(start, dtypes.int32), length, mask=full
                    )
                    invalid1 = rr.vges(
                        rr.varange(start + 64, dtypes.int32), length, mask=full
                    )
                    raw0 = rr.vdups(
                        self.kv_count,
                        dtypes.uint32,
                        mask=invalid0,
                        mode="merging",
                        merge=raw0,
                    )
                    raw1 = rr.vdups(
                        self.kv_count,
                        dtypes.uint32,
                        mask=invalid1,
                        mode="merging",
                        merge=raw1,
                    )
            else:
                prefix0 = rr.vlts(rr.varange(start, dtypes.int32), length, mask=full)
                prefix1 = rr.vlts(
                    rr.varange(start + 64, dtypes.int32), length, mask=full
                )
                raw0 = rr.vselect(raw0, invalid, cond_mask=prefix0)
                raw1 = rr.vselect(raw1, invalid, cond_mask=prefix1)
            if const_expr(
                not (
                    self.optimizations
                    & (NATIVE_INDEX_PREPROCESS | INDEX_LOAD_DEINTERLEAVE)
                )
            ):
                even, odd = rr.vdeinterleave(raw0, raw1)
            smaller = rr.vmin(even, odd, mask=full)
            larger = rr.vmax(even, odd, mask=full)
            smaller = rr.vmin(smaller, invalid, mask=full)
            store_index_validity(
                self.validity, job, start, smaller, larger, self.kv_count, full
            )
            valid_larger = rr.vlts(larger, self.kv_count, mask=full)
            if const_expr(self.optimizations & INDEX_MASKED_SUB):
                # Masked subtraction zeroes inactive lanes: invalid second
                # indices gather the first row twice with zero stride.
                delta = rr.vsub(larger, smaller, mask=valid_larger)
            else:
                safe_larger = rr.vselect(larger, smaller, cond_mask=valid_larger)
                delta = rr.vsub(safe_larger, smaller, mask=full)
            addr_lo, addr_hi = rr.vmull(smaller, stride, dtypes.uint32, mask=full)
            gap_lo, gap_hi = rr.vmull(delta, stride, dtypes.uint32, mask=full)
            if const_expr(self.optimizations & INDEX_INTERLEAVE_STORE):
                # Store the same 64-bit address/stride table directly
                # from its 32-bit words, with no temporary interleave.
                rr.vstore_interleave(address_words, start, addr_lo, addr_hi)
                rr.vstore_interleave(stride_words, start, gap_lo, gap_hi)
            else:
                addr0, addr1 = rr.vinterleave(addr_lo, addr_hi)
                gap0, gap1 = rr.vinterleave(gap_lo, gap_hi)
                rr.vstore(address_words, start, addr0, full)
                rr.vstore(address_words, start + 64, addr1, full)
                rr.vstore(stride_words, start, gap0, full)
                rr.vstore(stride_words, start + 64, gap1, full)
    vec_release_buf(PIPE.V, 1)
    vec_release_buf(PIPE.V, 0)


# single_kernel_prefill_pipeline: production definitions.

Q_FLAG = 10  # 10/11, distinct from score/S/rescale/fragment/KV intra-block IDs


class SingleCube(PipelineCube):
    def __init__(self, optimizations, h):
        PipelineCube.__init__(self, optimizations)
        self.head_tiles = h // 64
        self.h = h

    @jit
    def q_notify(self, src: Constexpr[PIPE], dst: Constexpr[PIPE], job):
        PipelineCube.q_notify(self, src, dst, job)
        if const_expr(src == PIPE.MTE1):
            cube_pair_arrive(PIPE.MTE1, Q_FLAG + job % 2)

    @jit
    def prefetch_query(self, q, query, job):
        actual = query // self.head_tiles
        head_tile = query % self.head_tiles
        self.q_wait(PIPE.MTE1, PIPE.MTE2, job)
        source = tile_slice(q, (1, self.h * 512), (actual, 0)).view((32, self.h * 16))
        # Copy inside each branch: merging formatted aliases into a Tensor
        # SSA value makes this DSL materialize an anonymous L1 allocation.
        if job % 2 == 0:
            mem_copy(
                tile_slice(
                    reinterpret(
                        self.q_lo[0],
                        layout=make_layout((32, 512)),
                        data_format="nd",
                    ),
                    (28, 512),
                    (0, 0),
                ),
                tile_slice(source, (28, 512), (0, head_tile * 2)),
            )
            mem_copy(
                tile_slice(
                    reinterpret(
                        self.q_hi[0],
                        layout=make_layout((32, 512)),
                        data_format="nd",
                    ),
                    (28, 512),
                    (0, 0),
                ),
                tile_slice(source, (28, 512), (0, head_tile * 2 + 1)),
            )
        else:
            mem_copy(
                tile_slice(
                    reinterpret(
                        self.q_lo[1],
                        layout=make_layout((32, 512)),
                        data_format="nd",
                    ),
                    (28, 512),
                    (0, 0),
                ),
                tile_slice(source, (28, 512), (0, head_tile * 2)),
            )
            mem_copy(
                tile_slice(
                    reinterpret(
                        self.q_hi[1],
                        layout=make_layout((32, 512)),
                        data_format="nd",
                    ),
                    (28, 512),
                    (0, 0),
                ),
                tile_slice(source, (28, 512), (0, head_tile * 2 + 1)),
            )
        self.q_notify(PIPE.MTE2, PIPE.MTE1, job)

    @jit
    def qk_block(self, q, query, job, tile, blocks, global_block, destination):
        if tile == 0:
            cube_pair_wait(PIPE.MTE1, Q_FLAG + job % 2)
        PipelineCube.qk_block(
            self, q, query, job, tile, blocks, global_block, destination
        )


class SingleVector(FusedVector):
    def __init__(
        self, have_sink, optimizations, score_mode, scale, multi_group, h, cores
    ):
        FusedVector.__init__(
            self, have_sink, optimizations, score_mode, scale, multi_group
        )
        self.h = h
        self.head_tiles = h // 64
        self.cores = cores
        self.fuse_score_norm = score_mode == 1
        self.early_output = False

    @jit
    def prepare_query(self, q, positions, cache, cube, query, job):
        actual = query // self.head_tiles
        head_tile = query % self.head_tiles
        # Full four-head loads stay inside the 4 KiB physical Q tail.
        scratch = reinterpret(
            self.nz,
            layout=make_layout((2048,)),
            dtype=dtypes.bfloat16,
            data_format="nd",
        )
        coeff = reinterpret(
            self.nz,
            layout=make_layout((64,)),
            dtype=dtypes.float32,
            offset=4096,
            data_format="nd",
        )
        vec_sync_intra_wait(PIPE.MTE2, Q_FLAG + job % 2)
        vec_get_buf(PIPE.MTE2, 11)
        source = tile_slice(q, (1, self.h * 512), (actual, 0)).view((32, self.h * 16))
        mem_copy(
            reinterpret(scratch, layout=make_layout((4, 512)), data_format="nd"),
            tile_slice(source, (4, 512), (7, head_tile * 2 + self.part)),
        )
        position = positions[actual]
        mem_copy(coeff, tile_slice(cache, (1, 64), (position, 0)).view((64,)))
        vec_release_buf(PIPE.MTE2, 11)
        vec_get_buf(PIPE.V, 11)
        with vf():
            full = rr.full_mask(32)
            table = rr.vload(coeff)
            ids = rr.varange(0, dtypes.uint32)
            even_ids = rr.vshl(rr.vshr(ids, 1, mask=full), 1, mask=full)
            odd_ids = rr.vadds(even_ids, 1, mask=full)
            even_mask = rr.veqs(
                rr.vbitwise_and(ids, rr.vdups(1, dtypes.uint32), mask=full),
                0,
                mask=full,
            )
            for chunk in range_constexpr(4):
                pair = rr.vadds(
                    rr.vshr(
                        rr.vbitwise_and(ids, rr.vdups(15, dtypes.uint32), mask=full),
                        1,
                        mask=full,
                    ),
                    chunk * 8,
                    mask=full,
                )
                cosine = rr.vgather_reg(table, pair)
                sine = rr.vgather_reg(table, rr.vadds(pair, 32, mask=full))
                negative_sine = rr.vmuls(sine, -1.0, mask=full)
                # Select the coefficients once for each four-head dimension
                # chunk. Each lane retains its original rounded product and
                # fused-multiply-add order before BF16 conversion.
                even_coefficient = rr.vselect(cosine, sine, cond_mask=even_mask)
                odd_coefficient = rr.vselect(negative_sine, cosine, cond_mask=even_mask)
                for group in range(8):
                    offset = chunk * 512 + group * 64
                    value = rr.vcast(
                        rr.vload_unpack(
                            scratch, offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                        ),
                        dtypes.float32,
                        mask=full,
                    )
                    even = rr.vgather_reg(value, even_ids)
                    odd = rr.vgather_reg(value, odd_ids)
                    product = rr.vmul(odd, odd_coefficient, mask=full)
                    rotated = rr.vmadd(even, even_coefficient, product, mask=full)
                    packed = rr.vcast(
                        rotated, dtypes.bfloat16, mask=full, rounding=rr.RoundingMode.RN
                    )
                    rr.vstore_pack(
                        scratch, offset, packed, full, pack_mode="b32_to_b16"
                    )
        vec_release_buf(PIPE.V, 11)
        vec_get_buf(PIPE.MTE3, 11)
        if self.part == 0:
            if job % 2 == 0:
                mem_copy(
                    reinterpret(
                        cube.q_lo[0],
                        layout=make_layout((2048,)),
                        offset=28672,
                        data_format="nd",
                    ),
                    tile_slice(scratch, (2048,), (0,)),
                )
            else:
                mem_copy(
                    reinterpret(
                        cube.q_lo[1],
                        layout=make_layout((2048,)),
                        offset=28672,
                        data_format="nd",
                    ),
                    tile_slice(scratch, (2048,), (0,)),
                )
        else:
            if job % 2 == 0:
                mem_copy(
                    reinterpret(
                        cube.q_hi[0],
                        layout=make_layout((2048,)),
                        offset=28672,
                        data_format="nd",
                    ),
                    tile_slice(scratch, (2048,), (0,)),
                )
            else:
                mem_copy(
                    reinterpret(
                        cube.q_hi[1],
                        layout=make_layout((2048,)),
                        offset=28672,
                        data_format="nd",
                    ),
                    tile_slice(scratch, (2048,), (0,)),
                )
        vec_sync_intra_arrive(PIPE.MTE3, Q_FLAG + job % 2)
        vec_release_buf(PIPE.MTE3, 11)

    @jit
    def preprocess(self, length, job, k: Constexpr[int]):
        preprocess_index_prefix(self, length, job, k)

    @jit
    def fetch(self, indices, query, length):
        if const_expr(indices.shape[1] == 0):
            vec_get_buf(PIPE.MTE2, 0)
            vec_release_buf(PIPE.MTE2, 0)
        else:
            PipelineVector.fetch(self, indices, query // self.head_tiles, length)

    @jit
    def fetch_group(self, indices, query, length, tile):
        FusedVector.fetch_group(self, indices, query // self.head_tiles, length, tile)

    @jit
    def prefetch_norm_query(self, q, query, global_block):
        # Early tap3(g): tap2(g+1) consumed this Gather bank. Late tap3(g)
        # releases it after tap5 output, before next tick's Gather(g+4).
        actual = query // self.head_tiles
        head_tile = query % self.head_tiles
        scratch = reinterpret(
            self.gathered(global_block + 1),
            layout=make_layout((32, 512)),
            data_format="nd",
        )
        source = tile_slice(q, (1, self.h * 512), (actual, 0)).view((32, self.h * 16))
        vec_get_buf(PIPE.MTE2, 8 + (global_block + 1) % 3)
        mem_copy(scratch, tile_slice(source, (32, 512), (0, head_tile * 2 + self.part)))
        vec_release_buf(PIPE.MTE2, 8 + (global_block + 1) % 3)

    @jit
    def norm_coefficients(self, q, query, job, global_block, eps):
        # Raw Q arrives at early tap3. Produce coefficients at late tap3,
        # after tap5 consumes the prior owner of denorm's two-job bank.
        scratch = reinterpret(
            self.gathered(global_block + 1),
            layout=make_layout((32, 512)),
            data_format="nd",
        )
        vec_get_buf(PIPE.V, 8 + (global_block + 1) % 3)
        vec_get_buf(PIPE.V, 2 + job % 2)
        with vf():
            full = rr.full_mask(32)
            low16 = rr.create_mask("vl16", 32)
            ids = rr.varange(0, dtypes.uint32)
            lane = rr.vbitwise_and(ids, rr.vdups(15, dtypes.uint32), mask=full)
            thirty_two = rr.create_mask("vl32", 32)
            for group in range(8):
                total = rr.vdups(0.0, dtypes.float32)
                for dim in range(32):
                    value = rr.vcast(
                        rr.vload_unpack(
                            scratch,
                            dim * 512 + group * 64,
                            unpack_mode=rr.UnpackMode.B16_TO_B32,
                        ),
                        dtypes.float32,
                        mask=full,
                    )
                    total = rr.vadd(total, rr.vmul(value, value, mask=full), mask=full)
                for head in range_constexpr(4):
                    value = rr.vgather_reg(total, rr.vadds(lane, head * 16, mask=full))
                    head_sum = rr.vreduce_sum(value, mask=low16)
                    rr.vstore_first(
                        self.denorm,
                        (dtypes.int64(job) % 2) * 64 + group * 4 + head,
                        head_sum,
                    )
            # All 32 FP32 reductions are stored without conversion. The
            # barrier orders those scalar stores before the full-vector load.
            # Only the low 32 lanes are consumed by the final coefficient store;
            # denorm word 32 (history) and the remaining padding are untouched.
            rr.vmem_bar("vst_vld")
            sums = rr.vload(self.denorm, (dtypes.int64(job) % 2) * 64)
            mean = rr.vmuls(sums, 1.0 / 512, mask=full)
            inverse = rr.vdiv(
                rr.vdups(1.0, dtypes.float32),
                rr.vsqrt(rr.vadds(mean, eps, mask=full), mask=full),
                mask=full,
            )
            rr.vstore(
                self.denorm,
                (dtypes.int64(job) % 2) * 64,
                rr.vmuls(inverse, self.score_scale, mask=full),
                thirty_two,
            )
        vec_release_buf(PIPE.V, 2 + job % 2)
        vec_release_buf(PIPE.V, 8 + (global_block + 1) % 3)

    @jit
    def prescale(self, alpha, query):
        if const_expr(self.score_mode != 1):
            FusedVector.prescale(self, alpha, query)
        else:
            bank = (query // self.cores) % 2
            for group in range(8):
                with vf():
                    full = rr.full_mask(32)
                    heads = rr.vshr(rr.varange(0, dtypes.uint32), 4, mask=full)
                    norm = reinterpret(
                        self.denorm,
                        layout=make_layout((64,)),
                        offset=dtypes.int64(bank) * 256,
                    )
                    factors = rr.vload(norm)
                    factors = rr.vgather_reg(
                        factors, rr.vadds(heads, group * 4, mask=full)
                    )
                    for column in range_constexpr(4):
                        offset = group * 64 + column * 512
                        rr.vstore(
                            self.score,
                            offset,
                            rr.vmul(rr.vload(self.score, offset), factors, mask=full),
                            full,
                        )
                    rr.vmem_bar("vst_vld")

    @jit
    def load_sink(self, sink, query):
        head_tile = query % self.head_tiles
        vec_get_buf(PIPE.MTE2, 12)
        start = head_tile * 64 + self.part * 32
        mem_copy(self.sink, sink[start : start + 32,])
        vec_release_buf(PIPE.MTE2, 12)
        vec_get_buf(PIPE.V, 12)
        vec_release_buf(PIPE.V, 12)

    @jit
    def quantized_output(self, out, sf, positions, cache, query, column):
        fragment = self.fragments[column]
        packed = reinterpret(
            fragment,
            dtype=dtypes.float8_e4m3fn,
            layout=make_layout((4096,)),
        )
        codes = reinterpret(
            fragment,
            dtype=dtypes.int32,
            layout=make_layout((128,)),
            offset=4096,
        )
        words = reinterpret(
            fragment,
            dtype=dtypes.int32,
            layout=make_layout((320,)),
            offset=4864,
        )
        coeff = reinterpret(
            fragment,
            dtype=dtypes.float32,
            layout=make_layout((64,)),
            offset=4608,
        )
        actual = query // self.head_tiles
        head_tile = query % self.head_tiles
        if const_expr(column == 3):
            position = positions[actual]
            vec_release_buf(PIPE.V, 4 + column)
            vec_get_buf(PIPE.MTE2, 4 + column)
            mem_copy(coeff, tile_slice(cache, (1, 64), (position, 0)).view((64,)))
            vec_release_buf(PIPE.MTE2, 4 + column)
            vec_get_buf(PIPE.V, 4 + column)
        with vf():
            full = rr.full_mask(32)
            low = rr.create_mask("vl32", 32)
            ids = rr.varange(0, dtypes.uint32)
            upper_ids = rr.vadds(
                rr.vbitwise_and(ids, rr.vdups(31, dtypes.uint32), mask=full),
                32,
                mask=full,
            )
            if const_expr(column == 3):
                table = rr.vload(coeff)
                pair = rr.vshr(ids, 1, mask=full)
                even_ids = rr.vshl(pair, 1, mask=full)
                odd_ids = rr.vadds(even_ids, 1, mask=full)
                even_mask = rr.veqs(
                    rr.vbitwise_and(ids, rr.vdups(1, dtypes.uint32), mask=full),
                    0,
                    mask=full,
                )
                cosine = rr.vgather_reg(table, pair)
                sine = rr.vmuls(
                    rr.vgather_reg(table, rr.vadds(pair, 32, mask=full)),
                    -1.0,
                    mask=full,
                )
                negative_sine = rr.vmuls(sine, -1.0, mask=full)
                for row in range(32):
                    value = rope_prepared_register(
                        rr.vload(self.accum, row * 512 + 448),
                        cosine,
                        sine,
                        negative_sine,
                        even_ids,
                        odd_ids,
                        even_mask,
                        full,
                    )
                    rr.vstore(self.accum, row * 512 + 448, value, full)
            rr.vmem_bar("vst_vld")
            for row in range(32):
                for half in range_constexpr(2):
                    value = rr.vload(self.accum, row * 512 + column * 128 + half * 64)
                    first, second, code0, code1 = quantize_block_pair(
                        value, full, low, upper_ids
                    )
                    base = (row // 8) * 1024 + half * 512 + (row % 8) * 32
                    rr.vstore_pack(packed, base, first, low, pack_mode="b32_to_b8")
                    rr.vstore_pack(
                        packed, base + 256, second, low, pack_mode="b32_to_b8"
                    )
                    rr.vstore_first(
                        codes, row * 4 + half * 2, rr.vreinterpret(code0, dtypes.int32)
                    )
                    rr.vstore_first(
                        codes,
                        row * 4 + half * 2 + 1,
                        rr.vreinterpret(code1, dtypes.int32),
                    )
        pack_sf_words(codes, words)
        vec_release_buf(PIPE.V, 4 + column)
        vec_get_buf(PIPE.MTE3, 4 + column)
        dma_sf_store(
            words,
            sf,
            actual,
            head_tile,
            self.part,
            column,
            out.shape[0] // (self.h // 8),
        )
        group = actual * (self.h // 8) + head_tile * 8 + self.part * 4
        mem_copy(
            tile_slice(out, (4, 1024), (group // 4, column)),
            reinterpret(packed, layout=make_layout((4, 1024)), data_format="nd"),
        )
        vec_release_buf(PIPE.MTE3, 4 + column)
        # FixPipe may overwrite this physical fragment only after its FP8
        # DMA has read it. The V acquisition supplies MTE3 -> V completion.
        vec_get_buf(PIPE.V, 4 + column)
        vec_sync_intra_arrive(PIPE.V, FRAGMENT_FLAG + column % 2)
        vec_release_buf(PIPE.V, 4 + column)

    @jit
    def quantized_output_norm_scratch(self, out, sf, positions, cache, query, column):
        self.fragments[column]
        packed = reinterpret(
            self.nz,
            dtype=dtypes.float8_e4m3fn,
            layout=make_layout((4096,)),
            data_format="nd",
        )
        codes = reinterpret(
            self.nz,
            dtype=dtypes.int32,
            layout=make_layout((128,)),
            offset=4096,
            data_format="nd",
        )
        words = reinterpret(
            self.nz,
            dtype=dtypes.int32,
            layout=make_layout((320,)),
            offset=4864,
            data_format="nd",
        )
        coeff = reinterpret(
            self.nz,
            dtype=dtypes.float32,
            layout=make_layout((64,)),
            offset=4608,
            data_format="nd",
        )
        actual = query // self.head_tiles
        head_tile = query % self.head_tiles
        if const_expr(column == 3):
            position = positions[actual]
            vec_get_buf(PIPE.MTE2, 11)
            mem_copy(coeff, tile_slice(cache, (1, 64), (position, 0)).view((64,)))
            vec_release_buf(PIPE.MTE2, 11)
        vec_get_buf(PIPE.V, 11)
        with vf():
            full = rr.full_mask(32)
            low = rr.create_mask("vl32", 32)
            ids = rr.varange(0, dtypes.uint32)
            upper_ids = rr.vadds(
                rr.vbitwise_and(ids, rr.vdups(31, dtypes.uint32), mask=full),
                32,
                mask=full,
            )
            if const_expr(column == 3):
                table = rr.vload(coeff)
                pair = rr.vshr(ids, 1, mask=full)
                even_ids = rr.vshl(pair, 1, mask=full)
                odd_ids = rr.vadds(even_ids, 1, mask=full)
                even_mask = rr.veqs(
                    rr.vbitwise_and(ids, rr.vdups(1, dtypes.uint32), mask=full),
                    0,
                    mask=full,
                )
                cosine = rr.vgather_reg(table, pair)
                sine = rr.vmuls(
                    rr.vgather_reg(table, rr.vadds(pair, 32, mask=full)),
                    -1.0,
                    mask=full,
                )
                negative_sine = rr.vmuls(sine, -1.0, mask=full)
                for row in range(32):
                    value = rope_prepared_register(
                        rr.vload(self.accum, row * 512 + 448),
                        cosine,
                        sine,
                        negative_sine,
                        even_ids,
                        odd_ids,
                        even_mask,
                        full,
                    )
                    rr.vstore(self.accum, row * 512 + 448, value, full)
            rr.vmem_bar("vst_vld")
            for row in range(32):
                for half in range_constexpr(2):
                    value = rr.vload(self.accum, row * 512 + column * 128 + half * 64)
                    first, second, code0, code1 = quantize_block_pair(
                        value, full, low, upper_ids
                    )
                    base = (row // 8) * 1024 + half * 512 + (row % 8) * 32
                    rr.vstore_pack(packed, base, first, low, pack_mode="b32_to_b8")
                    rr.vstore_pack(
                        packed, base + 256, second, low, pack_mode="b32_to_b8"
                    )
                    rr.vstore_first(
                        codes, row * 4 + half * 2, rr.vreinterpret(code0, dtypes.int32)
                    )
                    rr.vstore_first(
                        codes,
                        row * 4 + half * 2 + 1,
                        rr.vreinterpret(code1, dtypes.int32),
                    )
        pack_sf_words(codes, words)
        # Final-output and quantization VFs both precede this V ACK.
        # DMA now uses separate scratch11, so fragment ownership can end here.
        vec_sync_intra_arrive(PIPE.V, FRAGMENT_FLAG + column % 2)
        vec_release_buf(PIPE.V, 4 + column)
        vec_release_buf(PIPE.V, 11)
        vec_get_buf(PIPE.MTE3, 11)
        dma_sf_store(
            words,
            sf,
            actual,
            head_tile,
            self.part,
            column,
            out.shape[0] // (self.h // 8),
        )
        group = actual * (self.h // 8) + head_tile * 8 + self.part * 4
        mem_copy(
            tile_slice(out, (4, 1024), (group // 4, column)),
            reinterpret(packed, layout=make_layout((4, 1024)), data_format="nd"),
        )
        vec_release_buf(PIPE.MTE3, 11)
        # Preserve output-scratch protection through DMA completion.
        vec_get_buf(PIPE.V, 11)
        vec_release_buf(PIPE.V, 11)

    @jit
    def output_block(
        self, out, sf, positions, cache, flags, job, query, tile, blocks, history
    ):
        # Static dtype dispatch preserves the original FP8 body below.
        if const_expr(out.dtype == dtypes.float32):
            FusedVector.output_block(
                self, out, flags, job, query, tile, blocks, history
            )
            return
        # Word 32 in each denorm bank is outside its 32 final inverse values.
        history_words = reinterpret(
            self.denorm,
            dtype=dtypes.uint32,
            layout=make_layout((128,)),
        )
        if const_expr(not bool(self.optimizations & FUSED_HISTORY_STATE)):
            vec_get_buf(PIPE.S, 2 + job % 2)
            history = history_words[(dtypes.int64(job) % 2) * 64 + 32]
            vec_release_buf(PIPE.S, 2 + job % 2)
            if tile == 0:
                history = dtypes.uint32(0)
        rescale = dtypes.uint32(0)
        if tile != 0:
            vec_sync_intra_wait(PIPE.S, RESCALE_FLAG)
            if const_expr(self.optimizations & SSBUF_UINT64):
                packed_flags = reinterpret(
                    flags,
                    dtype=dtypes.uint64,
                    layout=make_layout((64,)),
                )
                rescale = dtypes.uint32(packed_flags.vec[(job % 8) * 8 + tile % 8] != 0)
            else:
                rescale = (
                    flags.vec[job % 8, tile % 8, 0] + flags.vec[job % 8, tile % 8, 1]
                )
        if const_expr(self.optimizations & FUSED_HISTORY_STATE):
            history = dtypes.uint32(0)
            if tile != 0 and (rescale != 0 or tile + 1 == blocks):
                vec_get_buf(PIPE.S, 2 + job % 2)
                history = history_words[(dtypes.int64(job) % 2) * 64 + 32]
                vec_release_buf(PIPE.S, 2 + job % 2)
        scales = reinterpret(
            self.scales,
            layout=make_layout((64,)),
            offset=(job % 2) * 512 + (tile % 2) * 256,
        )
        if rescale != 0:
            for column in range_constexpr(4):
                vec_get_buf(PIPE.V, 4 + column)
                vec_sync_intra_wait(PIPE.V, FRAGMENT_FLAG + column % 2)
                fragment = self.fragments[column]
                if const_expr(
                    column == 3 and bool(self.optimizations & FUSED_HISTORY_STATE)
                ):
                    vec_get_buf(PIPE.V, 2 + job % 2)
                with vf():
                    full = rr.full_mask(32)
                    for row in range(32):
                        factor = rr.vload_broadcast(scales, row * 2)
                        for part in range_constexpr(2):
                            offset = row * 512 + column * 128 + part * 64
                            value = rr.vload(fragment, row * 128 + part * 64)
                            if history != dtypes.uint32(0):
                                value = rr.vadd(
                                    value, rr.vload(self.accum, offset), mask=full
                                )
                            rr.vstore(
                                self.accum,
                                offset,
                                rr.vmul(value, factor, mask=full),
                                full,
                            )
                    if const_expr(
                        column == 3 and bool(self.optimizations & FUSED_HISTORY_STATE)
                    ):
                        rr.vstore_first(
                            history_words,
                            (dtypes.int64(job) % 2) * 64 + 32,
                            rr.vdups(history + dtypes.uint32(1), dtypes.uint32),
                        )
                if const_expr(
                    column == 3 and bool(self.optimizations & FUSED_HISTORY_STATE)
                ):
                    vec_release_buf(PIPE.V, 2 + job % 2)
                vec_sync_intra_arrive(PIPE.V, FRAGMENT_FLAG + column % 2)
                vec_release_buf(PIPE.V, 4 + column)
            history = history + dtypes.uint32(1)
        if tile + 1 == blocks:
            denorm = reinterpret(
                self.denorm,
                layout=make_layout((64,)),
                offset=(job % 2) * 256,
            )
            for column in range_constexpr(4):
                vec_get_buf(PIPE.V, 4 + column)
                vec_sync_intra_wait(PIPE.V, FRAGMENT_FLAG + column % 2)
                fragment = self.fragments[column]
                final = reinterpret(
                    self.accum,
                    dtype=dtypes.float32,
                    layout=make_layout((32, 128), stride=(512, 1)),
                    offset=column * 512,
                )
                if const_expr(self.early_output):
                    if history != dtypes.uint32(0):
                        final_output_fp32(fragment, final, denorm, True)
                    else:
                        final_output_fp32(fragment, final, denorm, False)
                    self.quantized_output_norm_scratch(
                        out, sf, positions, cache, query, column
                    )
                else:
                    if history != dtypes.uint32(0):
                        final_output_fp32(fragment, final, denorm, True)
                    else:
                        final_output_fp32(fragment, final, denorm, False)
                    self.quantized_output(out, sf, positions, cache, query, column)
        if const_expr(not bool(self.optimizations & FUSED_HISTORY_STATE)):
            vec_get_buf(PIPE.V, 2 + job % 2)
            with vf():
                rr.vstore_first(
                    history_words,
                    (dtypes.int64(job) % 2) * 64 + 32,
                    rr.vdups(history, dtypes.uint32),
                )
            vec_release_buf(PIPE.V, 2 + job % 2)


@jit
def activate_single(
    vector,
    cube,
    line,
    flags,
    q,
    kv,
    indices,
    positions,
    cache,
    sink,
    out,
    sf,
    maximum,
    lse,
    eps,
    job,
    query,
    tile,
    blocks,
    length,
    global_block,
    index_group,
    active,
    history,
    scale,
    have_sink: Constexpr[bool],
):
    if active != 0:
        if const_expr(indices.shape[1] > INDEX_GROUP_SIZE):
            if tile % INDEX_TILES_PER_GROUP == 0:
                vector.fetch_group(indices, query, length, tile)
        else:
            if tile == 0:
                vector.fetch(indices, query, length)
    p2 = line.tap(2)
    if p2.active != 0:
        if p2.tile == 0:
            vector.prepare_query(q, positions, cache, cube, p2.query, p2.job)
        lo, hi = cube.slots(p2.global_block)
        vector.push_block(lo, hi, p2.global_block)
    if const_expr(vector.score_mode == 1):
        p3 = line.tap(3)
        if p3.active != 0 and p3.tile == 0:
            vector.prefetch_norm_query(q, p3.query, p3.global_block)
    p4 = line.tap(4)
    if p4.active != 0:
        group = p4.job
        if const_expr(indices.shape[1] > INDEX_GROUP_SIZE):
            group = p4.index_group
        vector.softmax_block_fused(cache, p4.query, p4.job, p4.tile, group, scale)
        vector.push_probability(cube.s)
        if p4.tile + 1 == p4.blocks:
            if const_expr(have_sink and vector.h > 64):
                vector.load_sink(sink, p4.query)
            vector.finish_state(maximum, lse, p4.job, p4.query, scale, have_sink)
    if active != 0:
        if const_expr(indices.shape[1] > INDEX_GROUP_SIZE):
            if tile % INDEX_TILES_PER_GROUP == 0:
                vector.preprocess(
                    min(
                        max(
                            dtypes.int64(length)
                            - dtypes.int64(tile // INDEX_TILES_PER_GROUP)
                            * INDEX_GROUP_SIZE,
                            0,
                        ),
                        INDEX_GROUP_SIZE,
                    ),
                    index_group,
                    INDEX_GROUP_SIZE,
                )
        else:
            if tile == 0:
                vector.preprocess(length, job, indices.shape[1])
    p5 = line.tap(5)
    if p5.active != 0:
        vector.output_block(
            out,
            sf,
            positions,
            cache,
            flags,
            p5.job,
            p5.query,
            p5.tile,
            p5.blocks,
            history,
        )
    if active != 0:
        local_tile = tile
        if const_expr(indices.shape[1] > INDEX_GROUP_SIZE):
            local_tile = tile % INDEX_TILES_PER_GROUP
        if local_tile == 0:
            if p4.active != 0:
                vector.publish_rescale(flags, p4.job, p4.tile, scale)
            vector.gather_block(kv, index_group, local_tile, global_block)
        else:
            vector.gather_block(kv, index_group, local_tile, global_block)
            if p4.active != 0:
                vector.publish_rescale(flags, p4.job, p4.tile, scale)
    else:
        if p4.active != 0:
            vector.publish_rescale(flags, p4.job, p4.tile, scale)

    # Norm only: p3's Q DMA was issued before p4. After p5 has consumed
    # any job n-2 final inverse, its two-job denorm bank is free for job n.
    # Publish current rescale/Gather work first so Cube can advance while
    # this RMS VF runs. The p3(g) Q scratch uses Gather(g+1), whereas this
    # tick issues Gather(g+3); it is released before next tick's Gather(g+4).
    if const_expr(vector.score_mode == 1):
        if p3.active != 0 and p3.tile == 0:
            vector.norm_coefficients(q, p3.query, p3.job, p3.global_block, eps)


@jit
def query_length(
    lengths, query, k: Constexpr[int], h: Constexpr[int], have_lengths: Constexpr[bool]
):
    if const_expr(have_lengths):
        return lengths[query // (h // 64)]
    else:
        return dtypes.int32(k)


@kernel
def fused_norm_rope_attn_rope_cast_fwd_kernel(
    q: Tensor,
    kv: Tensor,
    indices: Tensor,
    lengths: Tensor,
    positions: Tensor,
    cache: Tensor,
    sink: Tensor,
    out: Tensor,
    sf: Tensor,
    maximum: Tensor,
    lse: Tensor,
    scale: float,
    eps: float,
    cores: Constexpr[int],
    h: Constexpr[int],
    have_sink: Constexpr[bool],
    have_lengths: Constexpr[bool],
    score_mode: Constexpr[int],
    optimizations: Constexpr[int],
):
    scalar_type = schedule_scalar_dtype(optimizations)
    vector = SingleVector(
        have_sink,
        optimizations,
        score_mode,
        scale,
        indices.shape[1] > INDEX_GROUP_SIZE,
        h,
        cores,
    )
    # Measured output choice uses public compilation metadata only.
    if const_expr(
        score_mode == 1 and h == 64 and indices.shape[1] == 640 and not have_sink
    ):
        vector.early_output = True
    softmax_scale = scale
    if const_expr(score_mode != 0):
        softmax_scale = 1.0
    vector.kv_count = kv.shape[0]
    cube = SingleCube(optimizations, h)
    if const_expr(optimizations & NZ_SCORE_PATHS):
        score_destination = vector.score_nz
    else:
        score_destination = vector.score
    flags = unbound_view(
        pointer_space(MemLoc.SSBUF, 3072), (8, 8, 2), dtypes.uint32, 0, "nd"
    )
    vector.initialize(sink, False)
    jobs = q.shape[0] * (h // 64)
    vec_sync_all()
    vec_sync_intra_arrive(PIPE.MTE3, SCORE_FLAG)
    for slot in range_constexpr(2):
        vec_sync_intra_arrive(PIPE.MTE3, FRAGMENT_FLAG + slot)
        cube_sync_notify(PIPE.MTE1, PIPE.MTE2, slot)
        cube_pair_arrive(PIPE.MTE1, Q_FLAG + slot)
    cube_pair_arrive(PIPE.MTE1, S_FLAG)
    for slot in range_constexpr(4):
        cube_pair_arrive(PIPE.MTE1, KV_FLAG + slot)
    if is_aic():
        cube_scalar_type = cube_schedule_scalar_dtype(optimizations)
        first_query = get_block_idx()
        first_blocks = max(
            (
                cube_scalar_type(
                    query_length(
                        lengths, first_query, indices.shape[1], h, have_lengths
                    )
                )
                + (KV_TILE_SIZE - 1)
            )
            // KV_TILE_SIZE,
            1,
        )
        cube.prefetch_query(q, first_query, 0)
        cube.qk_block(q, first_query, 0, 0, first_blocks, 0, score_destination)
        job = cube_scalar_type(0)
        base = cube_scalar_type(0)
        for query_index in range(first_query, jobs, cores):
            query = cube_scalar_type(query_index)
            blocks = max(
                (
                    cube_scalar_type(
                        query_length(lengths, query, indices.shape[1], h, have_lengths)
                    )
                    + (KV_TILE_SIZE - 1)
                )
                // KV_TILE_SIZE,
                1,
            )
            next_query = query + cores
            if next_query < jobs:
                cube.prefetch_query(q, next_query, job + 1)
            for tile_index in range(blocks - 1):
                tile = cube_scalar_type(tile_index)
                cube.qk_block(
                    q, query, job, tile + 1, blocks, base + tile + 1, score_destination
                )
                cube.pv_block(vector, flags, job, tile, blocks, base + tile)
            if next_query < jobs:
                next_blocks = max(
                    (
                        cube_scalar_type(
                            query_length(
                                lengths, next_query, indices.shape[1], h, have_lengths
                            )
                        )
                        + (KV_TILE_SIZE - 1)
                    )
                    // KV_TILE_SIZE,
                    1,
                )
                cube.qk_block(
                    q,
                    next_query,
                    job + 1,
                    0,
                    next_blocks,
                    base + blocks,
                    score_destination,
                )
            cube.pv_block(vector, flags, job, blocks - 1, blocks, base + blocks - 1)
            base = base + blocks
            job = job + 1
    if is_aiv():
        # H64 owns the same 32 sink values for every Query on this AIV.
        # H128 alternates head tiles and keeps its per-Query loading.
        if const_expr(have_sink and h == 64):
            vector.load_sink(sink, 0)
        if const_expr(indices.shape[1] > INDEX_GROUP_SIZE):
            line = GroupDelayLine()
        elif const_expr(optimizations & METADATA_SKIP_UNUSED_LENGTH):
            line = UInt32LiveDelayLine()
        elif const_expr(optimizations & (UNSIGNED_METADATA | UNSIGNED_SCHEDULE)):
            line = UInt32UnsignedDelayLine()
        elif const_expr(optimizations & BOUNDED_PIPELINE_STATE):
            line = BoundedInt64DelayLine()
        elif const_expr(optimizations & UINT32_PIPELINE_STATE):
            line = UInt32DelayLine()
        else:
            line = BoundedInt64DelayLine()
        for unused in range_constexpr(6):
            if const_expr(indices.shape[1] > INDEX_GROUP_SIZE):
                line.push(
                    active=0,
                    job=0,
                    query=0,
                    tile=0,
                    blocks=0,
                    length=0,
                    global_block=0,
                    index_group=0,
                )
            else:
                line.push(
                    active=0, job=0, query=0, tile=0, blocks=0, length=0, global_block=0
                )
            line.advance()
        query = get_block_idx()
        group_base = scalar_type(0)
        job = scalar_type(0)
        global_block = scalar_type(0)
        length = scalar_type(
            query_length(lengths, query, indices.shape[1], h, have_lengths)
        )
        blocks = max((length + (KV_TILE_SIZE - 1)) // KV_TILE_SIZE, 1)
        draining = scalar_type(0)
        history = scalar_type(0)
        while draining < 2:
            count = blocks
            active = scalar_type(1)
            if draining == 1:
                count = scalar_type(6)
                active = scalar_type(0)
            for tile_index in range(count):
                tile = scalar_type(tile_index)
                index_group = job
                if const_expr(indices.shape[1] > INDEX_GROUP_SIZE):
                    index_group = group_base + tile // INDEX_TILES_PER_GROUP
                    line.push(
                        active=active,
                        job=job,
                        query=query,
                        tile=tile,
                        blocks=blocks,
                        length=length,
                        global_block=global_block,
                        index_group=index_group,
                    )
                else:
                    line.push(
                        active=active,
                        job=job,
                        query=query,
                        tile=tile,
                        blocks=blocks,
                        length=length,
                        global_block=global_block,
                    )
                activate_single(
                    vector,
                    cube,
                    line,
                    flags,
                    q,
                    kv,
                    indices,
                    positions,
                    cache,
                    sink,
                    out,
                    sf,
                    maximum,
                    lse,
                    eps,
                    job,
                    query,
                    tile,
                    blocks,
                    length,
                    global_block,
                    index_group,
                    active,
                    history,
                    softmax_scale,
                    have_sink,
                )
                line.advance()
                global_block = global_block + active
            if draining == 1:
                draining = scalar_type(2)
            else:
                if const_expr(indices.shape[1] > INDEX_GROUP_SIZE):
                    group_base = group_base + max(
                        (length + (INDEX_GROUP_SIZE - 1)) // INDEX_GROUP_SIZE, 1
                    )
                query = query + cores
                job = job + 1
                if query < jobs:
                    length = scalar_type(
                        query_length(lengths, query, indices.shape[1], h, have_lengths)
                    )
                    blocks = max((length + (KV_TILE_SIZE - 1)) // KV_TILE_SIZE, 1)
                else:
                    draining = scalar_type(1)


@host
def single_launch(
    q: Tensor,
    kv: Tensor,
    indices: Tensor,
    lengths: Tensor,
    positions: Tensor,
    cache: Tensor,
    sink: Tensor,
    out: Tensor,
    sf: Tensor,
    maximum: Tensor,
    lse: Tensor,
    scale: float,
    eps: float,
    cores: Constexpr[int],
    h: Constexpr[int],
    have_sink: Constexpr[bool],
    have_lengths: Constexpr[bool],
    score_mode: Constexpr[int],
    optimizations: Constexpr[int],
):
    fused_norm_rope_attn_rope_cast_fwd_kernel[cores](
        q,
        kv,
        indices,
        lengths,
        positions,
        cache,
        sink,
        out,
        sf,
        maximum,
        lse,
        scale,
        eps,
        cores,
        h,
        have_sink,
        have_lengths,
        score_mode,
        optimizations,
    )


_COMPILE_ENV_LOCK = Lock()


def _compile_with_manual_sync(*args):
    """Keep the temporary compiler policy scoped across all three entry points."""
    with _COMPILE_ENV_LOCK:
        previous = os.environ.get("CANNBOTDSL_AUTO_BUFFER_SYNC")
        os.environ["CANNBOTDSL_AUTO_BUFFER_SYNC"] = "0"
        try:
            return dsl.compile(*args)
        finally:
            if previous is None:
                os.environ.pop("CANNBOTDSL_AUTO_BUFFER_SYNC", None)
            else:
                os.environ["CANNBOTDSL_AUTO_BUFFER_SYNC"] = previous


@lru_cache(maxsize=128)
def compiled_single(
    s,
    t,
    k,
    h,
    cores,
    have_sink,
    have_lengths,
    score_mode,
    q_stride,
    kv_stride,
    index_stride,
    cache_rows,
):
    optimizations = default_pipeline_optimizations(640 if k >= 640 else 128, have_sink)
    if kv_stride != 512:
        optimizations &= ~(GATHER_BYTE_INTERVAL | GATHER_FLAT_INTERVAL)
    specs = (
        dsl.TensorSpec((s, h * 512), dtypes.bfloat16, stride=(q_stride, 1)),
        dsl.TensorSpec((t, 512), dtypes.bfloat16, stride=(kv_stride, 1)),
        dsl.TensorSpec((s, k), dtypes.int32, stride=(index_stride, 1)),
        dsl.TensorSpec((s,), dtypes.int32),
        dsl.TensorSpec((s,), dtypes.int32),
        dsl.TensorSpec((cache_rows, 64), dtypes.float32),
        dsl.TensorSpec((h if have_sink else 64,), dtypes.float32),
        dsl.TensorSpec((s * (h // 8), 4096), dtypes.float8_e4m3fn),
        dsl.TensorSpec(((h // 8) * 32 * ((s + 3) // 4 * 4),), dtypes.int32),
        dsl.TensorSpec((s * h,), dtypes.float32),
        dsl.TensorSpec((s * h,), dtypes.float32),
    )
    return _compile_with_manual_sync(
        single_launch,
        *specs,
        float,
        float,
        cores,
        h,
        have_sink,
        have_lengths,
        score_mode,
        optimizations,
    )


# Workspace output pipeline: inverse RoPE, quantization and packing.
# All tail-local names are prefixed to preserve the baseline 4-tuple quantizer.

WORKSPACE_UB_VALUES = 0
WORKSPACE_UB_PACKED = 65536
WORKSPACE_UB_CODES = 81920
WORKSPACE_UB_SF_WORDS = 83968
WORKSPACE_UB_COEFF = 84992
WORKSPACE_UB_USED_BYTES = 86016
WORKSPACE_UB_BANK_BYTES = 86016
WORKSPACE_UB_TOTAL_BYTES = 172032
if WORKSPACE_UB_TOTAL_BYTES > UB_CAPACITY:
    raise RuntimeError(
        f"output UB layout requires {WORKSPACE_UB_TOTAL_BYTES} bytes, "
        f"target provides {UB_CAPACITY} bytes"
    )
WORKSPACE_INPUT_MUTEX_BASE = 0
WORKSPACE_OUTPUT_MUTEX_BASE = 2


def workspace_pointer_space(space, capacity):
    return make_tensor(
        make_pointer(dtypes.uint8, 0, space),
        make_layout((capacity,)),
        data_format="nd",
    )


@jit
def workspace_rope_prepared_register(
    value, cosine, sine, negative_sine, even_ids, odd_ids, even_mask, full
):
    even = rr.vgather_reg(value, even_ids)
    odd = rr.vgather_reg(value, odd_ids)
    p0 = rr.vmul(odd, negative_sine, mask=full)
    p1 = rr.vmul(odd, cosine, mask=full)
    y0 = rr.vmadd(even, cosine, p0, mask=full)
    y1 = rr.vmadd(even, sine, p1, mask=full)
    return rr.vselect(y0, y1, cond_mask=even_mask)


@jit
def workspace_quantize_block_pair(value, full, low, high, upper_ids):
    absolute = rr.vabs(value, mask=full)
    lower = rr.vdup(rr.vreduce_max(absolute, mask=low), mask=low)
    upper = rr.vreduce_max(absolute, mask=high)
    maximum = rr.vdup(upper, mask=high, mode="merging", merge=lower)
    raw = rr.vmuls(maximum, 1.0 / 448, mask=full)
    bits = rr.vreinterpret(raw, dtypes.uint32)
    code = rr.vshr(rr.vadds(bits, 8388607, mask=full), 23, mask=full)
    code = rr.vmaxs(code, 105, mask=full)
    inverse = rr.vreinterpret(
        rr.vshl(rr.vsub(rr.vdups(254, dtypes.uint32), code, mask=full), 23, mask=full),
        dtypes.float32,
    )
    scaled = rr.vmul(value, inverse, mask=full)
    packed = rr.vcast(
        scaled,
        dtypes.float8_e4m3fn,
        mask=full,
        rounding=rr.RoundingMode.RN,
        reg_layout=rr.RegLayout.ZERO,
        saturate=True,
    )
    return (packed, code)


@jit
def workspace_inverse_rope_tile(values, coeff):
    # Preparation has the same signs, separate products, and FMA operands as
    # quantized_output(column=3). Only coefficient selection now varies by q.
    with vf():
        full = rr.full_mask(32)
        ids = rr.varange(0, dtypes.uint32)
        pair = rr.vshr(ids, 1, mask=full)
        even_ids = rr.vshl(pair, 1, mask=full)
        odd_ids = rr.vadds(even_ids, 1, mask=full)
        even_mask = rr.veqs(
            rr.vbitwise_and(ids, rr.vdups(1, dtypes.uint32), mask=full), 0, mask=full
        )
        for query in range(4):
            table = rr.vload(coeff, query * 64)
            cosine = rr.vgather_reg(table, pair)
            sine = rr.vmuls(
                rr.vgather_reg(table, rr.vadds(pair, 32, mask=full)), -1.0, mask=full
            )
            negative_sine = rr.vmuls(sine, -1.0, mask=full)
            for head in range(8):
                row = query * 8 + head
                value = workspace_rope_prepared_register(
                    rr.vload(values, row * 512 + 448),
                    cosine,
                    sine,
                    negative_sine,
                    even_ids,
                    odd_ids,
                    even_mask,
                    full,
                )
                rr.vstore(values, row * 512 + 448, value, full)
        rr.vmem_bar("vst_vld")


@jit
def workspace_quantize_tile(values, packed, codes):
    # Keep the existing four column VFs and the two MX32 halves per load.
    # A row now indexes q*8+head, rather than four groups of a single query.
    for column in range_constexpr(4):
        with vf():
            full = rr.full_mask(32)
            low = rr.create_mask("vl32", 32)
            high = rr.mask_not(low, exec_mask=full)
            ids = rr.varange(0, dtypes.uint32)
            upper_ids = rr.vadds(
                rr.vbitwise_and(ids, rr.vdups(31, dtypes.uint32), mask=full),
                32,
                mask=full,
            )
            code_offsets = rr.vshr(ids, 5, mask=full)
            code_lanes = rr.veqs(
                rr.vbitwise_and(ids, rr.vdups(31, dtypes.uint32), mask=full),
                0,
                mask=full,
            )
            for row in range(32):
                for half in range_constexpr(2):
                    value = rr.vload(values, row * 512 + column * 128 + half * 64)
                    first, code0 = workspace_quantize_block_pair(
                        value, full, low, high, upper_ids
                    )
                    base = (
                        (row // 8) * 4096 + column * 1024 + half * 512 + (row % 8) * 32
                    )
                    rr.vstore_pack(packed, base, first, low, pack_mode="b32_to_b8")
                    rr.vstore_pack(
                        packed, base + 224, first, high, pack_mode="b32_to_b8"
                    )
                    rr.vscatter(
                        codes,
                        rr.vreinterpret(code0, dtypes.int32),
                        rr.vadds(
                            code_offsets, row * 16 + column * 4 + half * 2, mask=full
                        ),
                        mask=code_lanes,
                    )


@jit
def workspace_pack_sf_tile(codes, words):
    # codes[q*128+head*16+dim]. Each word combines four consecutive heads.
    # words[(dim*2+head//4)*8+q] has source stride 32 bytes for the SF DMA.
    with vf():
        # All four quantization VFs' scatter stores precede these gathers.
        rr.vmem_bar("vst_vld")
        full = rr.full_mask(32)
        low = rr.create_mask("vl32", 32)
        ids = rr.varange(0, dtypes.uint32)
        dim = rr.vshr(ids, 1, mask=full)
        half = rr.vbitwise_and(ids, rr.vdups(1, dtypes.uint32), mask=full)
        base = rr.vadd(rr.vshl(half, 6, mask=full), dim, mask=full)
        for query in range(4):
            query_base = rr.vadds(base, query * 128, mask=full)
            word = rr.vreinterpret(
                rr.vgather(codes, query_base, mask=low), dtypes.uint32
            )
            for byte in range_constexpr(1, 4):
                value = rr.vreinterpret(
                    rr.vgather(
                        codes, rr.vadds(query_base, byte * 16, mask=full), mask=low
                    ),
                    dtypes.uint32,
                )
                word = rr.vbitwise_or(
                    word, rr.vshl(value, byte * 8, mask=full), mask=full
                )
            target = rr.vadds(rr.vshl(ids, 3, mask=full), query, mask=full)
            rr.vscatter(words, rr.vreinterpret(word, dtypes.int32), target, mask=low)


@jit
def workspace_load_epilogue_bank(
    workspace, positions, cache, root, tile, h: Constexpr[int], bank: Constexpr[int]
):
    # bank is a compile-time literal at every call. There are no Tensor phis,
    # dynamic UB allocations, or shared IDs between either bank or input/output.
    values = reinterpret(
        root,
        dtype=dtypes.float32,
        layout=make_layout((4, 4096)),
        offset=bank * WORKSPACE_UB_BANK_BYTES + WORKSPACE_UB_VALUES,
    )
    coeff = reinterpret(
        root,
        dtype=dtypes.float32,
        layout=make_layout((4, 64)),
        offset=bank * WORKSPACE_UB_BANK_BYTES + WORKSPACE_UB_COEFF,
    )
    batch = tile // (h // 8)
    group = tile % (h // 8)
    # Input reuse waits only the previous V readers, not the disjoint output
    # MTE3 readers. IDs 0/1 protect values and coeff; IDs 2/3 protect outputs.
    vec_get_buf(PIPE.MTE2, WORKSPACE_INPUT_MUTEX_BASE + bank)
    mem_copy(values, tile_slice(workspace, (4, 4096), (batch, group)))
    for query in range_constexpr(4):
        position = positions[batch * 4 + query]
        mem_copy(
            tile_slice(coeff, (1, 64), (query, 0)),
            tile_slice(cache, (1, 64), (position, 0)),
        )
    vec_release_buf(PIPE.MTE2, WORKSPACE_INPUT_MUTEX_BASE + bank)


@jit
def workspace_compute_epilogue_bank(root, bank: Constexpr[int]):
    values = reinterpret(
        root,
        dtype=dtypes.float32,
        layout=make_layout((4, 4096)),
        offset=bank * WORKSPACE_UB_BANK_BYTES + WORKSPACE_UB_VALUES,
    )
    packed = reinterpret(
        root,
        dtype=dtypes.float8_e4m3fn,
        layout=make_layout((4, 4096)),
        offset=bank * WORKSPACE_UB_BANK_BYTES + WORKSPACE_UB_PACKED,
    )
    codes = reinterpret(
        root,
        dtype=dtypes.int32,
        layout=make_layout((512,)),
        offset=bank * WORKSPACE_UB_BANK_BYTES + WORKSPACE_UB_CODES,
    )
    words = reinterpret(
        root,
        dtype=dtypes.int32,
        layout=make_layout((32, 8)),
        offset=bank * WORKSPACE_UB_BANK_BYTES + WORKSPACE_UB_SF_WORDS,
    )
    coeff = reinterpret(
        root,
        dtype=dtypes.float32,
        layout=make_layout((4, 64)),
        offset=bank * WORKSPACE_UB_BANK_BYTES + WORKSPACE_UB_COEFF,
    )
    vec_get_buf(PIPE.V, WORKSPACE_INPUT_MUTEX_BASE + bank)
    vec_get_buf(PIPE.V, WORKSPACE_OUTPUT_MUTEX_BASE + bank)
    workspace_inverse_rope_tile(values, coeff)
    workspace_quantize_tile(values, packed, codes)
    # No subsequent instruction reads values/coeff. Release their ownership
    # before SF packing so MTE2 can refill them while V uses only codes/words.
    vec_release_buf(PIPE.V, WORKSPACE_INPUT_MUTEX_BASE + bank)
    workspace_pack_sf_tile(codes, words)
    vec_release_buf(PIPE.V, WORKSPACE_OUTPUT_MUTEX_BASE + bank)


@jit
def workspace_store_epilogue_bank(
    root, out, sf, tile, h: Constexpr[int], bank: Constexpr[int]
):
    packed = reinterpret(
        root,
        dtype=dtypes.float8_e4m3fn,
        layout=make_layout((4, 4096)),
        offset=bank * WORKSPACE_UB_BANK_BYTES + WORKSPACE_UB_PACKED,
    )
    words = reinterpret(
        root,
        dtype=dtypes.int32,
        layout=make_layout((32, 8)),
        offset=bank * WORKSPACE_UB_BANK_BYTES + WORKSPACE_UB_SF_WORDS,
    )
    sf_source = reinterpret(
        words,
        layout=make_layout((32, 4), stride=(8, 1)),
        data_format="nd",
    )
    sf_target = sf.view((h // 8 * 32, out.shape[0]))
    batch = tile // (h // 8)
    group = tile % (h // 8)
    vec_get_buf(PIPE.MTE3, WORKSPACE_OUTPUT_MUTEX_BASE + bank)
    mem_copy(tile_slice(out, (4, 4096), (batch, group)), packed)
    mem_copy(tile_slice(sf_target, (32, 4), (group, batch)), sf_source)
    vec_release_buf(PIPE.MTE3, WORKSPACE_OUTPUT_MUTEX_BASE + bank)


@kernel
def workspace_output_epilogue_upper_mask_kernel(
    workspace: Tensor,
    positions: Tensor,
    cache: Tensor,
    out: Tensor,
    sf: Tensor,
    h: Constexpr[int],
    vector_num: Constexpr[int],
):
    root = workspace_pointer_space(MemLoc.UB, UB_CAPACITY)
    tile = get_block_idx()
    tiles = workspace.shape[0] // 4 * (h // 8)
    if tile < tiles:
        # Prologue: load the first tile into bank 0. Every block walks its own
        # grid-stride subsequence. Even/odd local iterations use banks 0/1.
        workspace_load_epilogue_bank(workspace, positions, cache, root, tile, h, 0)
        while tile < tiles:
            next_tile = tile + vector_num
            if next_tile < tiles:
                workspace_load_epilogue_bank(
                    workspace, positions, cache, root, next_tile, h, 1
                )
            workspace_compute_epilogue_bank(root, 0)
            workspace_store_epilogue_bank(root, out, sf, tile, h, 0)
            tile = next_tile
            if tile < tiles:
                next_tile = tile + vector_num
                if next_tile < tiles:
                    workspace_load_epilogue_bank(
                        workspace, positions, cache, root, next_tile, h, 0
                    )
                workspace_compute_epilogue_bank(root, 1)
                workspace_store_epilogue_bank(root, out, sf, tile, h, 1)
                tile = next_tile
        # Each used output mutex ends after its final MTE3 release. Explicit
        # V ACKs drain both output DMA consumers before kernel exit. Inputs
        # end at V release, no prefetch is left without its matching compute.
        vec_get_buf(PIPE.V, WORKSPACE_OUTPUT_MUTEX_BASE + 0)
        vec_release_buf(PIPE.V, WORKSPACE_OUTPUT_MUTEX_BASE + 0)
        if get_block_idx() + vector_num < tiles:
            vec_get_buf(PIPE.V, WORKSPACE_OUTPUT_MUTEX_BASE + 1)
            vec_release_buf(PIPE.V, WORKSPACE_OUTPUT_MUTEX_BASE + 1)
    vec_sync_all()


@host
def workspace_epilogue_launch_upper_mask(
    workspace: Tensor,
    positions: Tensor,
    cache: Tensor,
    out: Tensor,
    sf: Tensor,
    h: Constexpr[int],
    vector_num: Constexpr[int],
):
    workspace_output_epilogue_upper_mask_kernel[vector_num](
        workspace, positions, cache, out, sf, h, vector_num
    )


def workspace_epilogue_specs(s, h=64, cache_rows=37):
    """Return the fixed five-tensor ABI; this function does not query a device."""
    if not isinstance(s, int) or s <= 0 or s % 4:
        raise ValueError("prototype requires positive S divisible by four")
    if h != 64:
        raise ValueError("prototype requires H=64")
    if not isinstance(cache_rows, int) or cache_rows <= 0:
        raise ValueError("cache_rows must be positive; every position must be in range")
    return (
        dsl.TensorSpec((s, h * 512), dtypes.float32),
        dsl.TensorSpec((s,), dtypes.int32),
        dsl.TensorSpec((cache_rows, 64), dtypes.float32),
        dsl.TensorSpec((s, (h // 8) * 4096), dtypes.float8_e4m3fn),
        dsl.TensorSpec(((h // 8) * 32 * s,), dtypes.int32),
    )


@lru_cache(maxsize=32)
def _workspace_compiled_epilogue_upper_mask(s, h, vector_num, cache_rows):
    specs = workspace_epilogue_specs(s, h, cache_rows)
    return _compile_with_manual_sync(
        workspace_epilogue_launch_upper_mask, *specs, h, vector_num
    )


def compiled_workspace_epilogue_upper_mask(s, h=64, vector_num=None, cache_rows=37):
    """Compile the split-input/output double-buffered pure-vector stage.

    A caller that already obtained the stream quota may pass vector_num=56
    explicitly. All launches of this returned program use that compiled grid.
    Changing stream quotas requires obtaining a program with the new count.
    """
    if vector_num is None:
        vector_num = int(dsl.get_platform_info(force_refresh=True).vector_core_num)
    if not isinstance(vector_num, int) or vector_num <= 0:
        raise ValueError("vector_num must be a positive runtime AIV core count")
    return _workspace_compiled_epilogue_upper_mask(s, h, vector_num, cache_rows)


@lru_cache(maxsize=128)
def compiled_workspace_core(
    s,
    t,
    k,
    h,
    cores,
    have_sink,
    have_lengths,
    score_mode,
    q_stride,
    kv_stride,
    index_stride,
    cache_rows,
):
    optimizations = default_pipeline_optimizations(640 if k >= 640 else 128, have_sink)
    if kv_stride != 512:
        optimizations &= ~(GATHER_BYTE_INTERVAL | GATHER_FLAT_INTERVAL)
    specs = (
        dsl.TensorSpec((s, h * 512), dtypes.bfloat16, stride=(q_stride, 1)),
        dsl.TensorSpec((t, 512), dtypes.bfloat16, stride=(kv_stride, 1)),
        dsl.TensorSpec((s, k), dtypes.int32, stride=(index_stride, 1)),
        dsl.TensorSpec((s,), dtypes.int32),
        dsl.TensorSpec((s,), dtypes.int32),
        dsl.TensorSpec((cache_rows, 64), dtypes.float32),
        dsl.TensorSpec((h if have_sink else 64,), dtypes.float32),
        dsl.TensorSpec((s * h, 512), dtypes.float32),
        dsl.TensorSpec((1,), dtypes.int32),
        dsl.TensorSpec((s * h,), dtypes.float32),
        dsl.TensorSpec((s * h,), dtypes.float32),
    )
    return _compile_with_manual_sync(
        single_launch,
        *specs,
        float,
        float,
        cores,
        h,
        have_sink,
        have_lengths,
        score_mode,
        optimizations,
    )


# Workspace integration: programs are cached only by metadata.
# Scratch tensors and their contents are never cached or shared by calls.
_WORKSPACE_TAIL_IMPLEMENTATION = "upper_mask"
_WORKSPACE_TAIL_COMPILERS = {"upper_mask": compiled_workspace_epilogue_upper_mask}
_workspace_core_programs = {}
_workspace_tail_programs = {}


def compiled_workspace_epilogue(
    s, h=64, vector_num=None, cache_rows=37, implementation=None
):
    # This registry is the single insertion point for a subsequently selected
    # device-validated tail. The selected implementation is upper_mask.
    choice = (
        _WORKSPACE_TAIL_IMPLEMENTATION if implementation is None else implementation
    )
    compiler = _WORKSPACE_TAIL_COMPILERS.get(choice)
    if compiler is None:
        raise ValueError(f"workspace tail implementation is unavailable: {choice}")
    return compiler(s, h, vector_num, cache_rows)


def _workspace_network_metadata(s, t, k, h, sm_scale):
    return s == 8192 and t == 8192 and h == 64 and k in (128, 640) and sm_scale > 0


def _workspace_network_forward(
    q,
    kv,
    indices,
    sm_scale,
    attn_sink,
    topk_length,
    enable_q_norm,
    rms_norm_eps,
    token_positions,
    cos_sin_cache,
    fp8,
    backing,
    sf,
    maximum,
    lse,
):
    # The public wrapper already validated the contract and selected q.device.
    # All allocation and both launches use its current stream. Each invocation
    # owns a new workspace tensor; allocator reuse remains stream ordered.
    s, h, _ = q.shape
    g = h // 8
    info = dsl.get_platform_info(force_refresh=True)
    cores = min(s * (h // 64), int(info.cube_core_num))
    vector_num = int(info.vector_core_num)
    if cores <= 0 or vector_num <= 0:
        raise RuntimeError("workspace path requires available AIC and AIV cores")
    score_mode = 1 if enable_q_norm else 0
    core_key = (
        s,
        kv.shape[0],
        indices.shape[-1],
        h,
        cores,
        attn_sink is not None,
        topk_length is not None,
        score_mode,
        q.stride(0),
        kv.stride(0),
        indices.stride(0),
        cos_sin_cache.shape[0],
    )
    core_program = _workspace_core_programs.get(core_key)
    if core_program is None:
        core_program = compiled_workspace_core(*core_key)
        _workspace_core_programs[core_key] = core_program
    choice = _WORKSPACE_TAIL_IMPLEMENTATION
    tail_key = (s, h, vector_num, cos_sin_cache.shape[0], choice)
    tail_program = _workspace_tail_programs.get(tail_key)
    if tail_program is None:
        tail_program = compiled_workspace_epilogue(
            s, h, vector_num, cos_sin_cache.shape[0], choice
        )
        _workspace_tail_programs[tail_key] = tail_program
    # S8192,H64: exactly 8192*64*512*4 = 1,073,741,824 bytes.
    workspace = torch.empty((s * h, 512), dtype=torch.float32, device=q.device)
    sf_dummy = torch.empty((1,), dtype=torch.int32, device=q.device)
    q_raw = q.as_strided((s, h * 512), (q.stride(0), 1))
    kv_raw = kv.as_strided((kv.shape[0], 512), (kv.stride(0), 1))
    idx_raw = indices.as_strided((s, indices.shape[-1]), (indices.stride(0), 1))
    lengths = token_positions if topk_length is None else topk_length
    sink = cos_sin_cache.view(-1)[:64] if attn_sink is None else attn_sink
    core_program(
        q_raw,
        kv_raw,
        idx_raw,
        lengths,
        token_positions,
        cos_sin_cache,
        sink,
        workspace,
        sf_dummy,
        maximum.view(-1),
        lse.view(-1),
        sm_scale,
        rms_norm_eps,
    )
    tail_program(
        workspace.view(s, h * 512),
        token_positions,
        cos_sin_cache,
        fp8.view(s, g * 4096),
        backing.view(-1),
    )
    return fp8, sf, maximum, lse


# prefill_contract: production definitions.


def _validate(
    enable_q_norm,
    rms_norm_eps,
    token_positions,
    is_rope_neox_style,
    rope_dim,
    cos_sin_cache,
    n_wv_group,
    num_per_channels,
    use_tma_aligned_col_major_sf,
    round_sf,
    use_packed_ue8m0,
    q,
    kv,
    indices,
    sm_scale,
    d_v,
    attn_sink,
    topk_length,
):
    if not isinstance(q, torch.Tensor) or q.ndim != 3:
        raise ValueError("q must be [S,H,512]")
    s, h, d = q.shape
    if q.device.type != "npu" or q.dtype != torch.bfloat16:
        raise ValueError("q must be BF16 on NPU")
    if h not in (64, 128) or d != 512 or d_v != 512 or n_wv_group != h // 8:
        raise ValueError("requires H=64/128, D_QK=D_V=512 and 8 heads/WV group")
    if q.stride(2) != 1 or q.stride(1) != 512:
        raise ValueError("q head/dimension strides must be (512,1)")
    if not isinstance(kv, torch.Tensor) or kv.ndim != 3 or kv.shape[1:] != (1, 512):
        raise ValueError("kv must be [T,1,512]")
    if kv.dtype != torch.bfloat16 or kv.device != q.device or kv.stride(2) != 1:
        raise ValueError("kv must be BF16 on q.device with contiguous last dimension")
    if (
        not isinstance(indices, torch.Tensor)
        or indices.ndim != 3
        or indices.shape[:2] != (s, 1)
    ):
        raise ValueError("indices must be [S,1,K]")
    if (
        indices.dtype != torch.int32
        or indices.device != q.device
        or indices.stride(2) != 1
    ):
        raise ValueError(
            "indices must be INT32 on q.device with contiguous last dimension"
        )
    for name, x, shape, dtype in (
        ("token_positions", token_positions, (s,), torch.int32),
        ("attn_sink", attn_sink, (h,), torch.float32),
        ("topk_length", topk_length, (s,), torch.int32),
    ):
        if x is None and name != "token_positions":
            continue
        if (
            not isinstance(x, torch.Tensor)
            or x.shape != shape
            or x.dtype != dtype
            or x.device != q.device
            or not x.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous {dtype} {shape} on q.device")
    if (
        not isinstance(cos_sin_cache, torch.Tensor)
        or cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != 64
        or cos_sin_cache.dtype != torch.float32
        or cos_sin_cache.device != q.device
        or not cos_sin_cache.is_contiguous()
    ):
        raise ValueError("cos_sin_cache must be contiguous [P,64] FP32 on q.device")
    if is_rope_neox_style or rope_dim != 64 or num_per_channels != 32:
        raise ValueError("requires interleave RoPE64 and quantization granularity32")
    if not all((use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)):
        raise ValueError("all three SF format switches must be True")
    if (
        not math.isfinite(sm_scale)
        or not math.isfinite(rms_norm_eps)
        or rms_norm_eps < 0
    ):
        raise ValueError(
            "sm_scale must be finite; rms_norm_eps must be finite and nonnegative"
        )


# single_kernel_prefill: production definitions.

_target_programs = {}


def fused_norm_rope_attn_rope_cast_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
    enable_q_norm: bool,
    rms_norm_eps: float,
    token_positions: torch.Tensor,
    is_rope_neox_style: bool,
    rope_dim: int,
    cos_sin_cache: torch.Tensor,
    n_wv_group: int,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool,
    round_sf: bool,
    use_packed_ue8m0: bool,
    /,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate(
        enable_q_norm,
        rms_norm_eps,
        token_positions,
        is_rope_neox_style,
        rope_dim,
        cos_sin_cache,
        n_wv_group,
        num_per_channels,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        q,
        kv,
        indices,
        sm_scale,
        d_v,
        attn_sink,
        topk_length,
    )
    s, h, _ = q.shape
    g = h // 8
    a = (s + 3) // 4 * 4
    with torch.npu.device(q.device):
        fp8 = torch.empty((s, g, 4096), dtype=torch.float8_e4m3fn, device=q.device)
        backing = torch.empty((g, 32, a), dtype=torch.int32, device=q.device)
        sf = backing.as_strided((s, g, 32), (1, 32 * a, a))
        maximum = torch.empty((s, h), dtype=torch.float32, device=q.device)
        lse = torch.empty((s, h), dtype=torch.float32, device=q.device)
        if s == 0:
            return fp8, sf, maximum, lse
        if _workspace_network_metadata(s, kv.shape[0], indices.shape[-1], h, sm_scale):
            return _workspace_network_forward(
                q,
                kv,
                indices,
                sm_scale,
                attn_sink,
                topk_length,
                enable_q_norm,
                rms_norm_eps,
                token_positions,
                cos_sin_cache,
                fp8,
                backing,
                sf,
                maximum,
                lse,
            )
        cores = min(
            s * (h // 64), int(dsl.get_platform_info(force_refresh=True).cube_core_num)
        )
        if cores <= 0:
            raise RuntimeError("no available AIC cores")
        score_mode = 1 if enable_q_norm else (0 if sm_scale > 0 else 2)
        key = (
            s,
            kv.shape[0],
            indices.shape[-1],
            h,
            cores,
            attn_sink is not None,
            topk_length is not None,
            score_mode,
            q.stride(0),
            kv.stride(0),
            indices.stride(0),
            cos_sin_cache.shape[0],
        )
        program = _target_programs.get(key)
        if program is None:
            program = compiled_single(*key)
        q_raw = q.as_strided((s, h * 512), (q.stride(0), 1))
        kv_raw = kv.as_strided((kv.shape[0], 512), (kv.stride(0), 1))
        idx_raw = indices.as_strided((s, indices.shape[-1]), (indices.stride(0), 1))
        lengths = token_positions if topk_length is None else topk_length
        sink = cos_sin_cache.view(-1)[:64] if attn_sink is None else attn_sink
        program(
            q_raw,
            kv_raw,
            idx_raw,
            lengths,
            token_positions,
            cos_sin_cache,
            sink,
            fp8.view(s * g, 4096),
            backing.view(-1),
            maximum.view(-1),
            lse.view(-1),
            sm_scale,
            rms_norm_eps,
        )
        return fp8, sf, maximum, lse


__all__ = ["fused_norm_rope_attn_rope_cast_fwd"]
