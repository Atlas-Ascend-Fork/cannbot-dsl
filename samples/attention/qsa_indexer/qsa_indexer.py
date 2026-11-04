# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""QSA Indexer：计算压缩 K 分数、执行 UINT16 TopK，并展开输出原始 token 索引。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from threading import RLock

import cannbotdsl as dsl
from cannbotdsl import make_layout
from cannbotdsl.lang.host import host
import torch
from cannbotdsl import Constexpr, Dim, TensorSpec, const_expr, dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.sync import vec_sync_all
from cannbotdsl.reg import (
    update_mask,
    vadd,
    vadds,
    varange,
    vcast,
    vgather,
    vhistogram_accumulate,
    vload,
    vmem_bar,
    vmax,
    vmaxs,
    vmins,
    vpack,
    vreinterpret,
    vreinterpret_lanes,
    vselect,
    vshl,
    vshr,
    vstore,
    vstore_first,
    vsub,
)
import cannbotdsl.reg as _raw_reg
from cannbotdsl.tensor import MemLoc, Tensor, tile_slice

if __package__:
    from .qsa_indexer_checker import (
        get_effective_core_counts as get_effective_core_counts,
        validate_and_resolve,
    )
else:
    from qsa_indexer_checker import (
        get_effective_core_counts as get_effective_core_counts,
        validate_and_resolve,
    )

HEAD_METADATA_STRIDE = 16
CORE_METADATA_STRIDE = 16
AIC_METADATA_CORE_CAPACITY = 36
SECTION_METADATA_STRIDE = AIC_METADATA_CORE_CAPACITY * CORE_METADATA_STRIDE
METADATA_ALIGNMENT_ELEMS = 4096
QUERY_TILE = 32
PAGE_SIZE = 256

HEAD_SECTION_COUNT_INDEX = 0
CORE_BN_BEGIN_INDEX = 0
CORE_M_BEGIN_INDEX = 1
CORE_BN_END_INDEX = 3
CORE_M_END_INDEX = 4


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def _align_up(n: int, alignment: int) -> int:
    return _ceil_div(n, alignment) * alignment


def metadata_capacity(batch_size: int) -> int:
    raw = HEAD_METADATA_STRIDE + max(1, batch_size) * SECTION_METADATA_STRIDE
    return _ceil_div(raw, METADATA_ALIGNMENT_ELEMS) * METADATA_ALIGNMENT_ELEMS


vcmp_eq = _raw_reg.veq
vcmp_eq_scalar = _raw_reg.veqs
vcmp_ge = _raw_reg.vge
vcmp_ge_scalar = _raw_reg.vges
vcmp_gt = _raw_reg.vgt
vcmp_lt = _raw_reg.vlt
vcmp_lt_scalar = _raw_reg.vlts
vcmp_ne_scalar = _raw_reg.vnes
vdup_lane0 = _raw_reg.vdup
vdup_scalar = _raw_reg.vdups
vdeintlv = _raw_reg.vdeinterleave
vload_brc = _raw_reg.vload_broadcast
vand = _raw_reg.vbitwise_and
vxor = _raw_reg.vbitwise_xor


def vload_unpack(tensor, offset=0, *, mode):
    return _raw_reg.vload_unpack(tensor, offset, unpack_mode=mode)


def vload_brc_b8(tensor, offset=0):
    return _raw_reg.vload_broadcast(tensor, offset, width="b8")


def vload_deinterleave_b8(tensor, offset=0):
    return _raw_reg.vload_deinterleave(tensor, offset, width="b8")


_store_unalign_regs = {}


def vstore_unalign_begin(tensor, no_clear_ar=False):
    reg = _raw_reg.vstore_unalign_begin(tensor, no_clear_ar=no_clear_ar)
    _store_unalign_regs[id(tensor)] = reg
    return reg


def vsqueeze_store_reg(value, *, mask):
    return _raw_reg.vsqueeze_and_storeunalign_init(value, mask=mask)


def vstore_unalign(tensor, offset, value):
    _raw_reg.vsqueeze_and_storeunalign(
        tensor, offset, value, _store_unalign_regs[id(tensor)]
    )


def vstore_unalign_post(tensor, offset):
    reg = _store_unalign_regs.pop(id(tensor))
    _raw_reg.vsqueeze_and_storeunalign_finalize(tensor, offset, reg)


M_TILE = 128
K_COMPRESSED_UNITS_PER_BLOCK = 256
N_CHUNK = 128
HEADS = 4
KV_HEADS = 1
LAUNCH_CACHE_CAPACITY = 32
DIM = 128
M_VECTOR = 16
TOKEN_COMPRESSION_RATIO = 4
TOKEN_COMPRESSION_SHIFT = 2
MAX_TOPK = 512
K_COMPRESSED_UNITS_PER_TOPK_CHUNK = 4096
K_BLOCKS_PER_TOPK_CHUNK = (
    K_COMPRESSED_UNITS_PER_TOPK_CHUNK // K_COMPRESSED_UNITS_PER_BLOCK
)
STREAM_INPUT = MAX_TOPK + K_COMPRESSED_UNITS_PER_TOPK_CHUNK
STREAM_GUARD_ELEMENTS = 16
STREAM_POS_STRIDE = STREAM_INPUT + STREAM_GUARD_ELEMENTS
RADIX_ROWS = 2
HISTOGRAM_BINS = 256
VL_U16 = 128
VL_U32 = 64
VEC_PACK_COLUMNS = 2 * VL_U32
# 同一行的每个扫描分片有 256 个 UINT16 分数，分放在两个 128-lane 寄存器中。
RADIX_SCAN_CHUNK = 2 * VL_U16
OUTPUT_PACK_CHUNK = TOKEN_COMPRESSION_RATIO * VL_U32
PARAM_ROW_ALIGNMENT_BYTES = 32
WS_OUT_KEYS = 0
OUTPUT_TRAILER_LANES = 8
MAX_EXPANDED_TOPK = MAX_TOPK * TOKEN_COMPRESSION_RATIO
MAX_CAUSAL_TAIL = TOKEN_COMPRESSION_RATIO - 1
OUTPUT_STRIDE = MAX_EXPANDED_TOPK + OUTPUT_TRAILER_LANES
COUNT_COLUMN = MAX_EXPANDED_TOPK + MAX_CAUSAL_TAIL
OUTPUT_LENGTH_COLUMN = COUNT_COLUMN
OUTPUT_COLUMN_COUNT = COUNT_COLUMN + 1

VECTOR_SUBBLOCKS_PER_CUBE = M_TILE // (M_VECTOR * HEADS)
SCORE_ROWS_PER_VECTOR = M_VECTOR * HEADS

U16_BYTES = 2
U32_BYTES = 4
PARAM_ROW_STRIDE = PARAM_ROW_ALIGNMENT_BYTES // U32_BYTES
FP32_BYTES = 4
TOPK_INDEX_SLOT_COUNT = 2
OUTPUT_CHANNEL_DEPTH = 2
SCORE_INPUT_CHANNEL_DEPTH = 2
SCORE_CHANNEL_DEPTH = 2
KEY_L1_CHANNEL_DEPTH = 2
CUBE_CHANNEL_DEPTH = 2
K_L0_CHANNEL_DEPTH = 2
UB_ADDRESS_ALIGNMENT_BYTES = 4096
UB_CAPACITY_BYTES = 248 * 1024

UB_MERGED_CANDIDATE_SCORES_ADDR = 0
UB_MERGED_CANDIDATE_SCORES_BYTES = RADIX_ROWS * STREAM_INPUT * U16_BYTES
UB_HISTORY_INDEX_SLOTS_ADDR = (
    UB_MERGED_CANDIDATE_SCORES_ADDR + UB_MERGED_CANDIDATE_SCORES_BYTES
)
UB_HISTORY_INDEX_SLOTS_BYTES = TOPK_INDEX_SLOT_COUNT * RADIX_ROWS * MAX_TOPK * U32_BYTES
UB_FILTER_MAP_ADDR = UB_HISTORY_INDEX_SLOTS_ADDR + UB_HISTORY_INDEX_SLOTS_BYTES
UB_FILTER_MAP_BYTES = RADIX_ROWS * STREAM_POS_STRIDE * U16_BYTES
UB_RADIX_POSITIONS_ADDR = UB_FILTER_MAP_ADDR + UB_FILTER_MAP_BYTES
UB_RADIX_POSITIONS_BYTES = RADIX_ROWS * STREAM_POS_STRIDE * U16_BYTES
UB_SCORE_WORKSPACE_ADDR = UB_RADIX_POSITIONS_ADDR + UB_RADIX_POSITIONS_BYTES
UB_SCORE_WORKSPACE_BYTES = RADIX_ROWS * (MAX_TOPK // 2) * FP32_BYTES
UB_HISTOGRAM_ADDR = UB_SCORE_WORKSPACE_ADDR + UB_SCORE_WORKSPACE_BYTES
UB_HISTOGRAM_BYTES = RADIX_ROWS * HISTOGRAM_BINS * U16_BYTES
UB_HISTOGRAM_ROW_BYTES = HISTOGRAM_BINS * U16_BYTES
UB_IDX_HIGH_ADDR = UB_HISTOGRAM_ADDR + UB_HISTOGRAM_BYTES
UB_IDX_HIGH_BYTES = RADIX_ROWS * HISTOGRAM_BINS * U16_BYTES
UB_IDX_LOW_ADDR = UB_IDX_HIGH_ADDR + UB_IDX_HIGH_BYTES
UB_IDX_LOW_BYTES = RADIX_ROWS * HISTOGRAM_BINS * U16_BYTES
UB_NEXT_K_ADDR = UB_IDX_LOW_ADDR + UB_IDX_LOW_BYTES
UB_NEXT_K_BYTES = RADIX_ROWS * VL_U16 * U16_BYTES
# 每行标量占一个对齐单元，兼容 Vector 标量写出及压缩状态写回。
UB_PARAMETER_BUFFER_BYTES = RADIX_ROWS * PARAM_ROW_ALIGNMENT_BYTES
UB_VISIBLE_COMPRESSED_K_ADDR = UB_NEXT_K_ADDR + UB_NEXT_K_BYTES
UB_KEEP_COUNT_ADDR = UB_VISIBLE_COMPRESSED_K_ADDR + UB_PARAMETER_BUFFER_BYTES
UB_VISIBLE_UNCOMPRESSED_K_ADDR = UB_KEEP_COUNT_ADDR + UB_PARAMETER_BUFFER_BYTES
UB_FILTERED_INDICES_BYTES_ADDR = (
    UB_VISIBLE_UNCOMPRESSED_K_ADDR + UB_PARAMETER_BUFFER_BYTES
)
UB_OUTPUT_ADDR = UB_FILTERED_INDICES_BYTES_ADDR + UB_PARAMETER_BUFFER_BYTES
UB_OUTPUT_BYTES = OUTPUT_CHANNEL_DEPTH * OUTPUT_STRIDE * U32_BYTES
UB_SCORE_INPUT_ADDR = UB_OUTPUT_ADDR + UB_OUTPUT_BYTES
UB_SCORE_INPUT_BYTES = (
    SCORE_INPUT_CHANNEL_DEPTH
    * RADIX_ROWS
    * K_COMPRESSED_UNITS_PER_TOPK_CHUNK
    * U16_BYTES
)
TOPK_UB_END = UB_SCORE_INPUT_ADDR + UB_SCORE_INPUT_BYTES
CROSS_CORE_SCORE_ADDR = _align_up(TOPK_UB_END, UB_ADDRESS_ALIGNMENT_BYTES)
CROSS_CORE_SCORE_BYTES = (
    SCORE_CHANNEL_DEPTH
    * SCORE_ROWS_PER_VECTOR
    * K_COMPRESSED_UNITS_PER_BLOCK
    * FP32_BYTES
)


@dataclass(frozen=True)
class Specialization:
    """保持 DSL Kernel 实例的编译期网格特化。"""

    grid: int


class Matmul:
    """管理 Query/Key 搬运、Matmul 点积和跨核分数输出。"""

    def __init__(self):
        # 每个物理页包含一个完整 N256 评分分组。
        self.q_l1 = Buffer(MemLoc.L1, dtypes.bfloat16, make_layout((M_TILE, DIM)))
        self.k_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((K_COMPRESSED_UNITS_PER_BLOCK, DIM)),
            depth=KEY_L1_CHANNEL_DEPTH,
        )
        self.l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((M_TILE, DIM)),
            depth=CUBE_CHANNEL_DEPTH,
        )
        self.l0b = RingBuffer(
            MemLoc.L0B,
            dtypes.bfloat16,
            make_layout((N_CHUNK, DIM)),
            depth=K_L0_CHANNEL_DEPTH,
        )
        self.l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((M_TILE, K_COMPRESSED_UNITS_PER_BLOCK)),
            depth=CUBE_CHANNEL_DEPTH,
        )

    @jit
    def load_q_resident(self, q, q_block_start):
        base = q[q_block_start * HEADS :, None]
        mem_copy(self.q_l1, tile_slice(base, (M_TILE, DIM), (0, 0)))

    @jit
    def load_key(self, k, page_table, bn, key_base, valid_blocks):
        # 一个物理页恰好对应一个 N256 评分分组，可直接写入 K-L1 Channel。
        logical_page = key_base // PAGE_SIZE
        physical_page = 0
        if logical_page * PAGE_SIZE < valid_blocks:
            physical_page = dtypes.int64(page_table[bn, logical_page])
        source = tile_slice(k[physical_page, None, None], (PAGE_SIZE, DIM), (0, 0))
        mem_copy(self.k_l1.next_write(), source)

    @jit
    def compute_qk(self):
        k_ready = self.k_l1.next_read()
        score_write = self.l0c.next_write()
        for n_sub in tuple(range(K_COMPRESSED_UNITS_PER_BLOCK // N_CHUNK)):
            q_l0 = self.l0a.next_write()
            k_l0 = self.l0b.next_write()
            mem_copy(q_l0, self.q_l1)
            mem_copy(k_l0, tile_slice(k_ready, (N_CHUNK, DIM), (n_sub, 0)))
            matmul(
                tile_slice(score_write, (M_TILE, N_CHUNK), (0, n_sub)),
                q_l0,
                k_l0,
                init=True,
            )

    @jit
    def store_score(self, score_channel):
        mem_copy(score_channel.next_write(), self.l0c.next_read().partition(dim=0))


class Vector:
    """执行逐 Head ReLU、分数合并、UINT16 TopK 和索引写出。"""

    def __init__(self):
        ub_storage = dsl.UB.view(262144)
        self.subblock = get_subblock_id()
        # TopK 使用 UB 区间 [UB_MERGED_CANDIDATE_SCORES_ADDR, TOPK_UB_END)。
        # 跨核分数 Channel 在该区间之后对齐分配，避免与 TopK 工作区重叠。
        # score_output 与 merged_candidate_scores 在 Vector 的不同计算阶段复用空间；
        # stream_indices 是 topk_indices_slots 的视图，stream_positions 是 filter_map
        # 的逐行视图，各视图与对应源 Buffer 共享存储。
        # 历史与新增候选合并后的分数，以 UINT16 排序键编码存储。
        self.merged_candidate_scores = dsl.make_buffer(
            ub_storage[
                UB_MERGED_CANDIDATE_SCORES_ADDR : UB_MERGED_CANDIDATE_SCORES_ADDR
                + RADIX_ROWS * STREAM_INPUT * 2,
            ]
            .view(dtype=dtypes.uint16)
            .view((RADIX_ROWS, STREAM_INPUT)),
        )
        self.topk_indices_slots = dsl.make_buffer(
            ub_storage[
                UB_HISTORY_INDEX_SLOTS_ADDR : UB_HISTORY_INDEX_SLOTS_ADDR
                + TOPK_INDEX_SLOT_COUNT * RADIX_ROWS * MAX_TOPK * 4,
            ]
            .view(dtype=dtypes.uint32)
            .view((TOPK_INDEX_SLOT_COUNT * RADIX_ROWS, MAX_TOPK)),
        )
        self.stream_indices = tile_slice(
            self.topk_indices_slots, (RADIX_ROWS, MAX_TOPK), (0, 0)
        )
        self.filter_map = dsl.make_buffer(
            ub_storage[
                UB_FILTER_MAP_ADDR : UB_FILTER_MAP_ADDR
                + RADIX_ROWS * STREAM_POS_STRIDE * 2,
            ]
            .view(dtype=dtypes.uint16)
            .view((RADIX_ROWS, STREAM_POS_STRIDE)),
        )
        self.stream_positions = tuple(
            tile_slice(self.filter_map, (1, STREAM_POS_STRIDE), (i, 0)).view(
                STREAM_POS_STRIDE
            )
            for i in range(RADIX_ROWS)
        )
        self.radix_positions = tuple(
            dsl.make_buffer(
                ub_storage[
                    UB_RADIX_POSITIONS_ADDR
                    + i * STREAM_POS_STRIDE * U16_BYTES : UB_RADIX_POSITIONS_ADDR
                    + i * STREAM_POS_STRIDE * U16_BYTES
                    + STREAM_POS_STRIDE * 2,
                ]
                .view(dtype=dtypes.uint16)
                .view((STREAM_POS_STRIDE,)),
            )
            for i in range(RADIX_ROWS)
        )
        self.score_workspace = dsl.make_buffer(
            ub_storage[
                UB_SCORE_WORKSPACE_ADDR : UB_SCORE_WORKSPACE_ADDR
                + RADIX_ROWS * (MAX_TOPK // 2) * 4,
            ]
            .view(dtype=dtypes.float32)
            .view((RADIX_ROWS, MAX_TOPK // 2)),
        )
        self.hist = dsl.make_buffer(
            ub_storage[
                UB_HISTOGRAM_ADDR : UB_HISTOGRAM_ADDR + RADIX_ROWS * HISTOGRAM_BINS * 2,
            ]
            .view(dtype=dtypes.uint16)
            .view((RADIX_ROWS, HISTOGRAM_BINS)),
        )
        self.idx_high = tuple(
            dsl.make_buffer(
                ub_storage[
                    UB_IDX_HIGH_ADDR + i * UB_HISTOGRAM_ROW_BYTES : UB_IDX_HIGH_ADDR
                    + i * UB_HISTOGRAM_ROW_BYTES
                    + HISTOGRAM_BINS * 2,
                ]
                .view(dtype=dtypes.uint16)
                .view((HISTOGRAM_BINS,)),
            )
            for i in range(RADIX_ROWS)
        )
        self.idx_low = tuple(
            dsl.make_buffer(
                ub_storage[
                    UB_IDX_LOW_ADDR + i * UB_HISTOGRAM_ROW_BYTES : UB_IDX_LOW_ADDR
                    + i * UB_HISTOGRAM_ROW_BYTES
                    + HISTOGRAM_BINS * 2,
                ]
                .view(dtype=dtypes.uint16)
                .view((HISTOGRAM_BINS,)),
            )
            for i in range(RADIX_ROWS)
        )
        # fmt: off
        self.next_k = dsl.make_buffer(
            ub_storage[
                UB_NEXT_K_ADDR : UB_NEXT_K_ADDR + RADIX_ROWS * VL_U16 * 2,
            ]
            .view(dtype=dtypes.uint16)
            .view((RADIX_ROWS, VL_U16)),
        )
        # fmt: on
        self.visible_compressed_k_ub = dsl.make_buffer(
            ub_storage[
                UB_VISIBLE_COMPRESSED_K_ADDR : UB_VISIBLE_COMPRESSED_K_ADDR
                + RADIX_ROWS * PARAM_ROW_STRIDE * 4,
            ]
            .view(dtype=dtypes.int32)
            .view((RADIX_ROWS, PARAM_ROW_STRIDE)),
        )
        self.keep_count_ub = dsl.make_buffer(
            ub_storage[
                UB_KEEP_COUNT_ADDR : UB_KEEP_COUNT_ADDR
                + RADIX_ROWS * PARAM_ROW_STRIDE * 4,
            ]
            .view(dtype=dtypes.int32)
            .view((RADIX_ROWS, PARAM_ROW_STRIDE)),
        )
        self.visible_uncompressed_k_ub = dsl.make_buffer(
            ub_storage[
                UB_VISIBLE_UNCOMPRESSED_K_ADDR : UB_VISIBLE_UNCOMPRESSED_K_ADDR
                + RADIX_ROWS * PARAM_ROW_STRIDE * 4,
            ]
            .view(dtype=dtypes.int32)
            .view((RADIX_ROWS, PARAM_ROW_STRIDE)),
        )
        self.filtered_indices_bytes_ub = dsl.make_buffer(
            ub_storage[
                UB_FILTERED_INDICES_BYTES_ADDR : UB_FILTERED_INDICES_BYTES_ADDR
                + RADIX_ROWS * PARAM_ROW_STRIDE * 4,
            ]
            .view(dtype=dtypes.int32)
            .view((RADIX_ROWS, PARAM_ROW_STRIDE)),
        )
        self.output = dsl.make_ring_buffer(
            [
                ub_storage[
                    UB_OUTPUT_ADDR + slot * (1 * OUTPUT_STRIDE * 4) : UB_OUTPUT_ADDR
                    + slot * (1 * OUTPUT_STRIDE * 4)
                    + 1 * OUTPUT_STRIDE * 4,
                ]
                .view(dtype=dtypes.int32)
                .view((1, OUTPUT_STRIDE))
                for slot in range(OUTPUT_CHANNEL_DEPTH)
            ]
        )
        self.score_input = dsl.make_ring_buffer(
            [
                ub_storage[
                    UB_SCORE_INPUT_ADDR
                    + slot
                    * (
                        RADIX_ROWS * K_COMPRESSED_UNITS_PER_TOPK_CHUNK * 2
                    ) : UB_SCORE_INPUT_ADDR
                    + slot * (RADIX_ROWS * K_COMPRESSED_UNITS_PER_TOPK_CHUNK * 2)
                    + RADIX_ROWS * K_COMPRESSED_UNITS_PER_TOPK_CHUNK * 2,
                ]
                .view(dtype=dtypes.uint16)
                .view((RADIX_ROWS, K_COMPRESSED_UNITS_PER_TOPK_CHUNK))
                for slot in range(SCORE_INPUT_CHANNEL_DEPTH)
            ]
        )
        self.score_output = dsl.make_ring_buffer(
            [
                ub_storage[
                    UB_MERGED_CANDIDATE_SCORES_ADDR
                    + slot
                    * (
                        M_VECTOR * K_COMPRESSED_UNITS_PER_BLOCK * 2
                    ) : UB_MERGED_CANDIDATE_SCORES_ADDR
                    + slot * (M_VECTOR * K_COMPRESSED_UNITS_PER_BLOCK * 2)
                    + M_VECTOR * K_COMPRESSED_UNITS_PER_BLOCK * 2,
                ]
                .view(dtype=dtypes.uint16)
                .view((M_VECTOR, K_COMPRESSED_UNITS_PER_BLOCK))
                for slot in range(SCORE_CHANNEL_DEPTH)
            ]
        )
        self.empty_indices = self.stream_indices
        # 单个 AIV 的 TopK UB 峰值由 TOPK_UB_END 给出。

    @jit
    def initialize(self):
        # 空 K 路径仍会加载索引再屏蔽，先初始化双槽，保证所有读取都有定义。
        with vf(mode="simd"):
            mask32, _ = update_mask(VL_U32, elem_bits=32)
            reg_zero = vdup_scalar(0, dtypes.uint32, mask=mask32)
            for chunk in range(TOPK_INDEX_SLOT_COUNT * RADIX_ROWS * MAX_TOPK // VL_U32):
                vstore(self.topk_indices_slots, chunk * VL_U32, reg_zero, mask32)

    @jit
    def topk_parameters(self, query_positions, q_block_start, q_block_count, row_base):
        query = self.subblock * M_VECTOR + row_base
        # p0/p1 对应本组两行 Query；尾块不足两行时，无效行保留 -1。
        # 仅对有效行读取位置，避免越界；(-1 + 1) // 4 = 0，使无效行的可见压缩块数为 0。
        p0 = dtypes.int64(-1)
        p1 = dtypes.int64(-1)
        if query < q_block_count:
            p0 = dtypes.int64(query_positions[q_block_start + query])
        if query + 1 < q_block_count:
            p1 = dtypes.int64(query_positions[q_block_start + query + 1])
        values = (p0, p1)
        # 参数初始化 VF：将本组两行的可见 K 数量和初始保留数写入参数 UB，供选择与输出使用。
        with vf(mode="simd"):
            mask, _ = update_mask(1, elem_bits=32)
            for row_in_group in tuple(range(RADIX_ROWS)):
                position = vdup_scalar(values[row_in_group], dtypes.int32, mask=mask)
                visible_uncompressed = vadds(position, 1, mask=mask)
                visible = vshr(visible_uncompressed, TOKEN_COMPRESSION_SHIFT, mask=mask)
                # 保存本行可见的完整压缩 K 单元数：update_keep() 用它限制 TopK 数量，
                # _pack_one() 用它定位完整压缩块之后的原始 token 尾部。
                vstore_first(
                    self.visible_compressed_k_ub,
                    row_in_group * PARAM_ROW_STRIDE,
                    visible,
                )
                # 将本行保留候选数初始化为 0；后续 update_keep() 按可见压缩 K 数量更新，
                # 供 radix 选择、历史阈值过滤和最终索引展开使用。
                vstore_first(
                    self.keep_count_ub,
                    row_in_group * PARAM_ROW_STRIDE,
                    vdup_scalar(0, dtypes.int32, mask=mask),
                )
                # 保存原始 K 的可见数量（原位置加一）；无效 Query 行保存 0。
                # _pack_one() 据此计算因果尾部 token 数及最终输出长度。
                vstore_first(
                    self.visible_uncompressed_k_ub,
                    row_in_group * PARAM_ROW_STRIDE,
                    visible_uncompressed,
                )

    @jit
    def update_keep(self):
        # 候选数量 VF：读取可见压缩 K 数，写回每行目标保留数 min(可见数量, MAX_TOPK)。
        with vf(mode="simd"):
            mask, _ = update_mask(1, elem_bits=32)
            for row_in_group in tuple(range(RADIX_ROWS)):
                visible = vload_brc(
                    self.visible_compressed_k_ub, row_in_group * PARAM_ROW_STRIDE
                )
                keep = vmins(visible, MAX_TOPK, mask=mask)
                vstore_first(self.keep_count_ub, row_in_group * PARAM_ROW_STRIDE, keep)

    @jit
    def relu_reduce_stream(self, head_scores):
        score_slot = self.score_output.next_write()
        # 评分 VF：逐 head 将负分数置零，再累加四个 head；转换为 UINT16 排序分数写入输出槽。
        with vf(mode="simd"):
            fp32_mask, _ = update_mask(VL_U32, elem_bits=32)
            uint16_mask, _ = update_mask(VL_U16, elem_bits=16)
            reg_zero_fp32 = vdup_scalar(0.0, dtypes.float32, mask=fp32_mask)
            sort_key_sign_bit = vdup_scalar(0x8000, dtypes.uint16, mask=uint16_mask)
            for query_row in range(M_VECTOR):
                self._reduce_heads_and_pack(
                    head_scores,
                    query_row,
                    0,
                    fp32_mask,
                    uint16_mask,
                    reg_zero_fp32,
                    sort_key_sign_bit,
                    score_slot,
                )
                self._reduce_heads_and_pack(
                    head_scores,
                    query_row,
                    VEC_PACK_COLUMNS,
                    fp32_mask,
                    uint16_mask,
                    reg_zero_fp32,
                    sort_key_sign_bit,
                    score_slot,
                )

    @jit
    def save_scores(self, gm_score_workspace, core_base, group):
        dst = gm_score_workspace[core_base:, group * K_COMPRESSED_UNITS_PER_BLOCK :]
        mem_copy(
            tile_slice(dst, (M_VECTOR, K_COMPRESSED_UNITS_PER_BLOCK), (0, 0)),
            self.score_output.next_read(),
        )

    @jit
    def zero_scores(self):
        score_slot = self.score_output.next_write()
        # 补零 VF：将当前评分输出槽填为无效分数 0，补齐最后一个 TopK chunk 的缺失基本块。
        with vf(mode="simd"):
            mask, _ = update_mask(VL_U16, elem_bits=16)
            reg_zero = vdup_scalar(0, dtypes.uint16, mask=mask)
            for chunk in tuple(
                range(M_VECTOR * K_COMPRESSED_UNITS_PER_BLOCK // VL_U16)
            ):
                vstore(score_slot, chunk * VL_U16, reg_zero, mask)

    @jit
    def load_scores(self, gm_score_workspace, core_base, row_base, topk_chunk_index):
        src = gm_score_workspace[
            core_base + row_base :,
            topk_chunk_index * K_COMPRESSED_UNITS_PER_TOPK_CHUNK :,
        ]
        mem_copy(
            self.score_input.next_write(),
            tile_slice(src, (RADIX_ROWS, K_COMPRESSED_UNITS_PER_TOPK_CHUNK), (0, 0)),
        )

    @jit
    def merge_scores(self, input_scores, key_base):
        chunk_start = dtypes.int64(key_base)
        # 首个 topk_chunk_index 每个 256 分数片只加载一次，同时保留分数 并累计高字节直方图。
        # 首轮合并 VF：将可见输入分数写到预留历史区之后，不可见位置置零，并生成高字节直方图。
        with vf(mode="simd"):
            mask8, _ = update_mask(256, elem_bits=8)
            mask16, _ = update_mask(VL_U16, elem_bits=16)
            reg_zero = vdup_scalar(0, dtypes.uint16, mask=mask16)
            for row_in_group in tuple(range(RADIX_ROWS)):
                # 复用参数区的总可见长度，在 INT32 中截断后再转 UINT16。
                visible32 = vload_brc(
                    self.visible_compressed_k_ub, row_in_group * PARAM_ROW_STRIDE
                )
                mask32, _ = update_mask(VL_U32, elem_bits=32)
                visible32 = vsub(
                    visible32,
                    vdup_scalar(chunk_start, dtypes.int32, mask=mask32),
                    mask=mask32,
                )
                visible32 = vmins(
                    vmaxs(visible32, 0, mask=mask32),
                    K_COMPRESSED_UNITS_PER_TOPK_CHUNK,
                    mask=mask32,
                )
                visible_limit = vdup_lane0(
                    vreinterpret_lanes(visible32, dtypes.uint16), mask=mask16
                )
                count0 = vdup_scalar(0, dtypes.uint16, mask=mask16)
                count1 = vdup_scalar(0, dtypes.uint16, mask=mask16)
                # 每片用两次 UINT16 加载读取 256 个键，先在寄存器中屏蔽无效位置，
                # 再提取 256 个高字节，与直方图的 256-lane 输入匹配并累加计数。
                # 当前 4096 个键分 16 片扫描，合成同一份直方图，并非执行 16 次 TopK。
                # 每片的 256 个输入元素与高字节的 256 种桶取值是不同概念。
                for part in range(K_BLOCKS_PER_TOPK_CHUNK):
                    # 当前行、当前分片在输入分数 Buffer 中的起始元素偏移，用于加载分数。
                    input_offset = (
                        row_in_group * K_COMPRESSED_UNITS_PER_TOPK_CHUNK
                        + part * K_COMPRESSED_UNITS_PER_BLOCK
                    )
                    scores0 = vload(input_scores, input_offset)
                    scores1 = vload(input_scores, input_offset + VL_U16)
                    positions0 = varange(
                        part * K_COMPRESSED_UNITS_PER_BLOCK, dtypes.uint16
                    )
                    positions1 = varange(
                        part * K_COMPRESSED_UNITS_PER_BLOCK + VL_U16, dtypes.uint16
                    )
                    scores0 = vselect(
                        scores0,
                        reg_zero,
                        cond_mask=vcmp_lt(positions0, visible_limit, mask=mask16),
                    )
                    scores1 = vselect(
                        scores1,
                        reg_zero,
                        cond_mask=vcmp_lt(positions1, visible_limit, mask=mask16),
                    )
                    _, scores_high8 = vdeintlv(
                        vreinterpret_lanes(scores0, dtypes.uint8),
                        vreinterpret_lanes(scores1, dtypes.uint8),
                    )
                    vstore(
                        self.merged_candidate_scores,
                        row_in_group * STREAM_INPUT
                        + MAX_TOPK
                        + part * K_COMPRESSED_UNITS_PER_BLOCK,
                        scores0,
                        mask16,
                    )
                    vstore(
                        self.merged_candidate_scores,
                        row_in_group * STREAM_INPUT
                        + MAX_TOPK
                        + part * K_COMPRESSED_UNITS_PER_BLOCK
                        + VL_U16,
                        scores1,
                        mask16,
                    )
                    count0 = vhistogram_accumulate(
                        count0, scores_high8, mask=mask8, bin=0
                    )
                    count1 = vhistogram_accumulate(
                        count1, scores_high8, mask=mask8, bin=1
                    )
                vstore(self.hist, row_in_group * HISTOGRAM_BINS, count0, mask16)
                vstore(
                    self.hist, row_in_group * HISTOGRAM_BINS + VL_U16, count1, mask16
                )

    @jit
    def filter_positions(self, input_scores, key_base):
        chunk_start = dtypes.int64(key_base)
        # 仅保存新增候选的原始位置。每行独立完成一次 AR 压缩流，并将输出
        # 字节数写入当前行的 filtered_indices_bytes_ub。
        # 过滤 VF：只保留可见且大于历史阈值的新增候选，写出 chunk 内位置及位置列表的字节数。
        with vf(mode="simd"):
            mask16, _ = update_mask(VL_U16, elem_bits=16)
            for row_in_group in tuple(range(RADIX_ROWS)):
                # 复用参数区的总可见长度，在 INT32 中截断后再转 UINT16。
                visible32 = vload_brc(
                    self.visible_compressed_k_ub, row_in_group * PARAM_ROW_STRIDE
                )
                mask32, _ = update_mask(VL_U32, elem_bits=32)
                visible32 = vsub(
                    visible32,
                    vdup_scalar(chunk_start, dtypes.int32, mask=mask32),
                    mask=mask32,
                )
                visible32 = vmins(
                    vmaxs(visible32, 0, mask=mask32),
                    K_COMPRESSED_UNITS_PER_TOPK_CHUNK,
                    mask=mask32,
                )
                visible_limit = vdup_lane0(
                    vreinterpret_lanes(visible32, dtypes.uint16), mask=mask16
                )
                history_threshold_high8 = vload_brc(self.idx_high[row_in_group], 0)
                history_threshold_low8 = vload_brc(self.idx_low[row_in_group], 0)
                threshold = vadd(
                    vshl(history_threshold_high8, 8, mask=mask16),
                    history_threshold_low8,
                    mask=mask16,
                )
                cur_row_stream_positions = self.stream_positions[row_in_group]
                vstore_unalign_begin(cur_row_stream_positions)
                for chunk in range(K_COMPRESSED_UNITS_PER_TOPK_CHUNK // VL_U16):
                    scores = vload(
                        input_scores,
                        row_in_group * K_COMPRESSED_UNITS_PER_TOPK_CHUNK
                        + chunk * VL_U16,
                    )
                    key_positions_in_chunk = varange(chunk * VL_U16, dtypes.uint16)
                    # 当前 chunk 中，本轮加载的128个位置的可见性 mask
                    visible = vcmp_lt(
                        key_positions_in_chunk, visible_limit, mask=mask16
                    )
                    selected = vcmp_gt(scores, threshold, mask=visible)
                    vstore_unalign(
                        cur_row_stream_positions,
                        0,
                        vsqueeze_store_reg(key_positions_in_chunk, mask=selected),
                    )
                vstore_unalign_post(cur_row_stream_positions, 0)
                _raw_reg.vstorealign_squeeze_status(
                    self.filtered_indices_bytes_ub, row_in_group * PARAM_ROW_STRIDE
                )

    @jit
    def filtered_length(self):
        # 等待 Vector 写完过滤结果的字节数，确保 Scalar 能正确读取。
        vec_sync_all()
        # 字节数除以 2，得到两行各自筛出的新增候选数量。
        c0 = dtypes.int64(self.filtered_indices_bytes_ub[0, 0]) // U16_BYTES
        c1 = dtypes.int64(self.filtered_indices_bytes_ub[1, 0]) // U16_BYTES
        # 取两行最大数量，并在返回时向上对齐到 256，作为统一的新候选扫描长度。
        count = max(c0, c1)
        # 加上前面的 512 个历史候选槽，返回总扫描长度。
        return (
            MAX_TOPK
            + ((count + RADIX_SCAN_CHUNK - 1) // RADIX_SCAN_CHUNK) * RADIX_SCAN_CHUNK
        )

    @jit
    def gather_filtered(self, input_scores, scan_len):
        # 新增候选需要扫描的分片数，不包含前面的历史候选区。
        chunks = (scan_len - MAX_TOPK) // RADIX_SCAN_CHUNK
        # 收集 VF：按过滤位置读取分数，追加到历史候选之后；对齐空位补零，并累加高字节直方图。
        with vf(mode="simd"):
            mask16, _ = update_mask(VL_U16, elem_bits=16)
            mask8, _ = update_mask(256, elem_bits=8)
            reg_zero = vdup_scalar(0, dtypes.uint16, mask=mask16)
            for row_in_group in tuple(range(RADIX_ROWS)):
                # 读取当前行筛选出的索引字节数。
                bytes32 = vload_brc(
                    self.filtered_indices_bytes_ub, row_in_group * PARAM_ROW_STRIDE
                )
                # 转换为实际候选数量，用于屏蔽对齐产生的填充位置。
                count32 = vshr(bytes32, 1, mask=update_mask(VL_U32, elem_bits=32)[0])
                count16 = vdup_lane0(
                    vreinterpret_lanes(count32, dtypes.uint16), mask=mask16
                )
                # 加载历史候选的高字节直方图，后续累加新增候选的计数。
                count0 = vload(self.hist, row_in_group * HISTOGRAM_BINS)
                count1 = vload(self.hist, row_in_group * HISTOGRAM_BINS + VL_U16)
                # 每轮处理同一行的 256 个候选；一个寄存器容纳 128 个 UINT16 分数，
                # 因此 scores0/scores1 分别收集前/后 128 个候选，不代表两行或双缓冲。
                # 将超出实际候选数量的位置置零，再提取两组共256个分数的高字节统计直方图。
                for chunk in range(chunks):
                    # 前 128 个候选在过滤结果中的序号，用于判断是否有效。
                    lane0 = varange(chunk * RADIX_SCAN_CHUNK, dtypes.uint16)
                    valid0 = vcmp_lt(lane0, count16, mask=mask16)
                    # 读取候选在原始输入 chunk 中的位置。
                    position0 = vload(
                        self.stream_positions[row_in_group], chunk * RADIX_SCAN_CHUNK
                    )
                    # 填充位置使用安全下标，避免后续 gather 越界。
                    position0 = vselect(position0, reg_zero, cond_mask=valid0)
                    # 候选在输入分数 Buffer 中的元素偏移。
                    address0 = vadds(
                        position0,
                        row_in_group * K_COMPRESSED_UNITS_PER_TOPK_CHUNK,
                        mask=mask16,
                    )
                    # 按过滤位置收集分数，无效候选的分数置零。
                    scores0 = vgather(input_scores, address0, mask=mask16)
                    scores0 = vselect(scores0, reg_zero, cond_mask=valid0)
                    # 新增分数写到历史候选之后，前 512 个历史候选槽保持不变。
                    vstore(
                        self.merged_candidate_scores,
                        row_in_group * STREAM_INPUT
                        + MAX_TOPK
                        + chunk * RADIX_SCAN_CHUNK,
                        scores0,
                        mask16,
                    )
                    # 按相同方式收集后 128 个候选分数。
                    lane1 = varange(chunk * RADIX_SCAN_CHUNK + VL_U16, dtypes.uint16)
                    valid1 = vcmp_lt(lane1, count16, mask=mask16)
                    position1 = vload(
                        self.stream_positions[row_in_group],
                        chunk * RADIX_SCAN_CHUNK + VL_U16,
                    )
                    position1 = vselect(position1, reg_zero, cond_mask=valid1)
                    address1 = vadds(
                        position1,
                        row_in_group * K_COMPRESSED_UNITS_PER_TOPK_CHUNK,
                        mask=mask16,
                    )
                    scores1 = vgather(input_scores, address1, mask=mask16)
                    scores1 = vselect(scores1, reg_zero, cond_mask=valid1)
                    # 新增分数写到历史候选之后，前 512 个历史候选槽保持不变。
                    vstore(
                        self.merged_candidate_scores,
                        row_in_group * STREAM_INPUT
                        + MAX_TOPK
                        + chunk * RADIX_SCAN_CHUNK
                        + VL_U16,
                        scores1,
                        mask16,
                    )
                    # 提取这 256 个分数的高字节，供本轮阈值桶统计使用。
                    _, scores_high8 = vdeintlv(
                        vreinterpret_lanes(scores0, dtypes.uint8),
                        vreinterpret_lanes(scores1, dtypes.uint8),
                    )
                    # 将新增分数（含填充零）的计数累加到历史直方图。
                    count0 = vhistogram_accumulate(
                        count0, scores_high8, mask=mask8, bin=0
                    )
                    count1 = vhistogram_accumulate(
                        count1, scores_high8, mask=mask8, bin=1
                    )
                # 保存合并后的直方图，供后续 radix TopK 定位阈值。
                vstore(self.hist, row_in_group * HISTOGRAM_BINS, count0, mask16)
                vstore(
                    self.hist, row_in_group * HISTOGRAM_BINS + VL_U16, count1, mask16
                )

    @jit
    def stream_topk_batch(
        self,
        row_base,
        key_base,
        score_workspace,
        radix_positions,
        position_map,
        history_indices,
        output_indices,
        input_offset: Constexpr[int],
        valid_len,
        retain_scores: Constexpr[bool] = True,
    ):
        """按行组对填充后的候选集合执行 radix TopK。"""
        hist_end = (input_offset + valid_len) // RADIX_SCAN_CHUNK
        position_end = (input_offset + valid_len) // VL_U16

        # VF 1：用 keep_count 将第 K 大转换为升序秩，定位首个累计计数达到目标的高字节桶。
        # 桶编号写入 idx_high；减去前一桶的累计计数，得到桶内目标秩并写入 next_k。
        with vf(mode="simd"):
            mask16, _ = update_mask(VL_U16, elem_bits=16)
            mask32, _ = update_mask(VL_U32, elem_bits=32)
            reg_zero_u16 = vdup_scalar(0, dtypes.uint16, mask=mask16)
            one = vdup_scalar(1, dtypes.uint16, mask=mask16)
            topk0 = vload_brc(self.keep_count_ub, (row_base) * PARAM_ROW_STRIDE)
            topk1 = vload_brc(self.keep_count_ub, (row_base + 1) * PARAM_ROW_STRIDE)
            topk_regs = (topk0, topk1)
            for row_in_group in tuple(range(RADIX_ROWS)):
                topk32 = topk_regs[row_in_group]
                # 填充行或空行的K为0；基数排序仍需[1, valid_len]内的有效秩，输出掩码保留原始K值。
                rank_topk32 = vmaxs(topk32, 1, mask=mask32)
                bottom32 = vsub(
                    vdup_scalar(valid_len + 1, dtypes.int32, mask=mask32),
                    rank_topk32,
                    mask=mask32,
                )
                bottom16 = vdup_lane0(
                    vreinterpret_lanes(bottom32, dtypes.uint16), mask=mask16
                )
                hist_base = row_in_group * HISTOGRAM_BINS
                idx_high = self.idx_high[row_in_group]
                vstore_unalign_begin(idx_high)
                # 一行直方图有 256 个 UINT16 计数，分两次加载到 128-lane 寄存器。
                for chunk in range(HISTOGRAM_BINS // VL_U16):
                    indices = varange(chunk * VL_U16, dtypes.uint16)
                    counts = vload(self.hist, hist_base + chunk * VL_U16)
                    reached = vcmp_ge(counts, bottom16, mask=mask16)
                    vstore_unalign(
                        idx_high,
                        0,
                        vsqueeze_store_reg(indices, mask=reached),
                    )
                vstore_unalign_post(idx_high, 0)
            vmem_bar("vst_vld")

            # 将全局秩转换为选中高字节桶内的秩，更新next_k，供后续定位低字节阈值。
            for row_in_group in tuple(range(RADIX_ROWS)):
                topk32 = topk_regs[row_in_group]
                rank_topk32 = vmaxs(topk32, 1, mask=mask32)
                bottom32 = vsub(
                    vdup_scalar(valid_len + 1, dtypes.int32, mask=mask32),
                    rank_topk32,
                    mask=mask32,
                )
                bottom16 = vdup_lane0(
                    vreinterpret_lanes(bottom32, dtypes.uint16), mask=mask16
                )
                hist_base = row_in_group * HISTOGRAM_BINS
                threshold_high8 = vload_brc(self.idx_high[row_in_group], 0)
                # 已经找到高字节桶后，把“在所有候选中的排名”换算成“在这个桶里的排名”
                # 下方先将读取位置设为 0，避免访问 hist[-1]，再将读回的累计数量置零。
                threshold_high8_is_zero = vcmp_eq_scalar(
                    threshold_high8, 0, mask=mask16
                )
                previous_bucket_idx = vselect(
                    reg_zero_u16,
                    vsub(threshold_high8, one, mask=mask16),
                    cond_mask=threshold_high8_is_zero,
                )
                previous_bucket_offset = vadd(
                    previous_bucket_idx,
                    vdup_scalar(hist_base, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                previous_count = vgather(self.hist, previous_bucket_offset, mask=mask16)
                previous_count = vselect(
                    reg_zero_u16, previous_count, cond_mask=threshold_high8_is_zero
                )
                vstore(
                    self.next_k,
                    row_in_group * VL_U16,
                    vsub(bottom16, previous_count, mask=mask16),
                    mask16,
                )

        # VF 2：扫描本轮全部候选，只统计高字节等于 idx_high 的候选。
        # 将这些候选的低字节累计直方图写入 hist，供下一个 VF 定位低字节阈值。
        with vf(mode="simd"):
            mask8, _ = update_mask(256, elem_bits=8)
            mask16, _ = update_mask(VL_U16, elem_bits=16)
            for row_in_group in tuple(range(RADIX_ROWS)):
                row_idx = row_base + row_in_group
                threshold_high8_u16 = vload_brc(self.idx_high[row_in_group], 0)
                threshold_high8_u8 = vdup_lane0(
                    vreinterpret_lanes(threshold_high8_u16, dtypes.uint8), mask=mask8
                )
                count0 = vdup_scalar(0, dtypes.uint16, mask=mask16)
                count1 = vdup_scalar(0, dtypes.uint16, mask=mask16)
                for chunk in range(
                    input_offset // RADIX_SCAN_CHUNK,
                    hist_end,
                ):
                    scores_low8, scores_high8 = vload_deinterleave_b8(
                        self.merged_candidate_scores,
                        row_idx * STREAM_INPUT + chunk * RADIX_SCAN_CHUNK,
                    )
                    matching = vcmp_eq(scores_high8, threshold_high8_u8, mask=mask8)
                    count0 = vhistogram_accumulate(
                        count0, scores_low8, mask=matching, bin=0
                    )
                    count1 = vhistogram_accumulate(
                        count1, scores_low8, mask=matching, bin=1
                    )
                hist_base = row_in_group * HISTOGRAM_BINS
                vstore(self.hist, hist_base, count0, mask16)
                vstore(self.hist, hist_base + VL_U16, count1, mask16)

        # VF 3：读取低字节累计直方图和桶内目标秩 next_k，
        # 将满足累计计数条件的桶号依次写入 idx_low；第一个桶号即阈值低字节。
        with vf(mode="simd"):
            mask16, _ = update_mask(VL_U16, elem_bits=16)
            for row_in_group in tuple(range(RADIX_ROWS)):
                hist_base = row_in_group * HISTOGRAM_BINS
                next_k = vload_brc(self.next_k, row_in_group * VL_U16)
                idx_low = self.idx_low[row_in_group]
                vstore_unalign_begin(idx_low)
                # 一行直方图有 256 个 UINT16 计数，分两次加载到 128-lane 寄存器。
                for chunk in range(HISTOGRAM_BINS // VL_U16):
                    indices = varange(chunk * VL_U16, dtypes.uint16)
                    counts = vload(self.hist, hist_base + chunk * VL_U16)
                    reached = vcmp_ge(counts, next_k, mask=mask16)
                    vstore_unalign(
                        idx_low,
                        0,
                        vsqueeze_store_reg(indices, mask=reached),
                    )
                vstore_unalign_post(idx_low, 0)

        # VF 4：拼接高、低字节得到完整阈值；先压缩写出大于阈值的候选位置，
        # 再追加等于阈值的位置，写入 radix_positions；后续只取 keep_count 个位置。
        # 这里保存的是合并候选区中的局部位置，尚未还原为全局压缩 K 索引。
        with vf(mode="simd"):
            mask16, _ = update_mask(VL_U16, elem_bits=16)
            for row_in_group in tuple(range(RADIX_ROWS)):
                row_idx = row_base + row_in_group
                positions_out = radix_positions[row_in_group]
                threshold_high8 = vload_brc(self.idx_high[row_in_group], 0)
                threshold_low8 = vload_brc(self.idx_low[row_in_group], 0)
                target = vadd(
                    vshl(threshold_high8, 8, mask=mask16), threshold_low8, mask=mask16
                )
                vstore_unalign_begin(positions_out)
                # 先压缩写入严格大于阈值的候选位置。
                for chunk in range(
                    input_offset // VL_U16,
                    position_end,
                ):
                    scores = vload(
                        self.merged_candidate_scores,
                        row_idx * STREAM_INPUT + chunk * VL_U16,
                    )
                    candidate_positions = varange(chunk * VL_U16, dtypes.uint16)
                    selected = vcmp_gt(scores, target, mask=mask16)
                    vstore_unalign(
                        positions_out,
                        0,
                        vsqueeze_store_reg(candidate_positions, mask=selected),
                    )
                # 再追加等于阈值的候选位置，供截取前 keep 个时补足数量。
                for chunk in range(
                    input_offset // VL_U16,
                    position_end,
                ):
                    scores = vload(
                        self.merged_candidate_scores,
                        row_idx * STREAM_INPUT + chunk * VL_U16,
                    )
                    candidate_positions = varange(chunk * VL_U16, dtypes.uint16)
                    selected = vcmp_eq(scores, target, mask=mask16)
                    vstore_unalign(
                        positions_out,
                        0,
                        vsqueeze_store_reg(candidate_positions, mask=selected),
                    )
                vstore_unalign_post(positions_out, 0)

        if const_expr(retain_scores):
            # VF 5（仅 retain_scores=True）：按 radix_positions 收集入选候选分数，
            # 将 UINT16 分数按位打包到 score_workspace，超出 keep_count 的槽置零。
            # 先暂存分数，避免直接回写历史区时覆盖尚未读取的候选；最后一轮跳过。
            with vf(mode="simd"):
                mask16, _ = update_mask(VL_U16, elem_bits=16)
                mask32, _ = update_mask(VL_U32, elem_bits=32)
                reg_zero_u16 = vdup_scalar(0, dtypes.uint16, mask=mask16)
                topk0 = vload_brc(self.keep_count_ub, (row_base) * PARAM_ROW_STRIDE)
                topk1 = vload_brc(self.keep_count_ub, (row_base + 1) * PARAM_ROW_STRIDE)
                topk_regs = (topk0, topk1)
                for row_in_group in tuple(range(RADIX_ROWS)):
                    row_idx = row_base + row_in_group
                    topk16 = vdup_lane0(
                        vreinterpret_lanes(topk_regs[row_in_group], dtypes.uint16),
                        mask=mask16,
                    )
                    score_row_offset = vdup_scalar(
                        row_idx * STREAM_INPUT, dtypes.uint16, mask=mask16
                    )
                    for chunk in range(MAX_TOPK // VL_U16):
                        lane = varange(chunk * VL_U16, dtypes.uint16)
                        active = vcmp_lt(lane, topk16, mask=mask16)
                        position = vload(radix_positions[row_in_group], chunk * VL_U16)
                        position = vselect(position, reg_zero_u16, cond_mask=active)
                        absolute = vadd(position, score_row_offset, mask=mask16)
                        scores = vgather(
                            self.merged_candidate_scores, absolute, mask=mask16
                        )
                        vstore(
                            score_workspace,
                            (WS_OUT_KEYS + row_in_group * MAX_TOPK + chunk * VL_U16)
                            // 2,
                            vreinterpret_lanes(
                                vselect(scores, reg_zero_u16, cond_mask=active),
                                dtypes.float32,
                            ),
                            mask32,
                        )

        # VF 6：将入选的局部候选位置还原为全局压缩 K 索引，写入 output_indices。
        # 历史候选读取 history_indices；新增候选加 key_base，经过过滤的候选先查 position_map。
        # 索引运算使用 UINT32，仅 keep_count 范围内的输出有效。
        with vf(mode="simd"):
            # 一次处理 64 个 UINT32 索引；全局压缩 K 索引可能超过 UINT16 范围。
            mask32, _ = update_mask(VL_U32, elem_bits=32)
            # 读取两行实际保留数量；后续 active 只允许这些输出槽生效。
            topk0 = vload_brc(self.keep_count_ub, (row_base) * PARAM_ROW_STRIDE)
            topk1 = vload_brc(self.keep_count_ub, (row_base + 1) * PARAM_ROW_STRIDE)
            topk_regs = (topk0, topk1)
            reg_zero_u32 = vdup_scalar(0, dtypes.uint32, mask=mask32)
            # 历史候选区固定预留 MAX_TOPK=512 个槽，不等于当前实际保留数。
            history_capacity32 = vdup_scalar(MAX_TOPK, dtypes.uint32, mask=mask32)
            # 当前 TopK chunk 在整个压缩 K 序列中的起始索引。
            chunk_start32 = vdup_scalar(key_base, dtypes.uint32, mask=mask32)
            for row_in_group in tuple(range(RADIX_ROWS)):
                row_idx = row_base + row_in_group
                topk32 = vreinterpret(topk_regs[row_in_group], dtypes.uint32)
                # history_indices 每行有 512 个 UINT32；这里是当前行的元素偏移，不是字节偏移。
                # 例如 row_idx=1、历史槽号为7，则读取偏移512+7，等价于 history_indices[1][7]。
                index_offset = vdup_scalar(
                    row_idx * MAX_TOPK, dtypes.uint32, mask=mask32
                )
                # 最多输出 512 个索引，每轮处理 64 个，共 8 轮。
                for chunk in range(MAX_TOPK // VL_U32):
                    lane = varange(chunk * VL_U32, dtypes.uint32)
                    active = vcmp_lt(lane, topk32, mask=mask32)
                    # radix_positions 保存合并候选区内的位置；读取时从 UINT16 扩展为 UINT32。
                    # 位置0～511属于历史区，位置512起属于新增候选区。
                    position = vreinterpret(
                        vload_unpack(
                            radix_positions[row_in_group],
                            chunk * VL_U32,
                            mode="b16_to_b32",
                        ),
                        dtypes.uint32,
                    )
                    # 无效输出槽先使用安全位置0，最终写出时也会清零。
                    position = vselect(position, reg_zero_u32, cond_mask=active)
                    current_index = vsub(
                        vadd(position, chunk_start32, mask=mask32),
                        history_capacity32,
                        mask=mask32,
                    )
                    # 当前调用中首轮传入 input_offset=512，跳过历史区，只选择新增候选。
                    # 因此这里可直接使用上述索引；历史区内容不参与首轮 TopK。
                    if const_expr(input_offset == MAX_TOPK):
                        index = current_index
                    else:
                        # 后续轮从候选区位置区分历史和新增：小于512且 active 的才是历史候选。
                        historical = vcmp_lt(position, history_capacity32, mask=active)
                        # 非历史候选不使用历史查表结果，将查表槽号设为0以保证读取安全。
                        previous = vselect(position, reg_zero_u32, cond_mask=historical)
                        absolute = vadd(previous, index_offset, mask=mask32)
                        # 查回上一轮保存的全局压缩 K 索引，而不是直接输出历史槽号。
                        previous_index = vgather(history_indices, absolute, mask=mask32)
                        # 过滤后新增候选已压紧；其序号不再等于原 chunk 内位置，需要查询位置表。
                        # 减512得到过滤后的紧凑序号；历史或无效 lane 使用安全查表序号0。
                        compact = vsub(position, history_capacity32, mask=mask32)
                        compact = vselect(reg_zero_u32, compact, cond_mask=historical)
                        compact = vselect(compact, reg_zero_u32, cond_mask=active)
                        # 将64个查表序号压成UINT16；位置表内的索引仍在当前4096 chunk范围内。
                        compact16 = vpack(compact, dtypes.uint16, part="lower")
                        mask_idx16, _ = update_mask(VL_U32, elem_bits=16)
                        map_offset = vadds(
                            compact16, row_in_group * STREAM_POS_STRIDE, mask=mask_idx16
                        )
                        # 查回过滤前的 chunk 内位置。例如紧凑序号3可能映射到原位置300。
                        original16 = vgather(position_map, map_offset, mask=mask_idx16)
                        # 查表得到的是 chunk 内位置（0～4095），UINT16 足够；但加上
                        # chunk 起点后全局压缩 K 索引可能超过65535，需先扩展为 UINT32。
                        original32 = _raw_reg.vunpack(
                            original16, dtypes.uint32, part="lower"
                        )
                        # 加 chunk 起点得到全局压缩 K 索引，例如4096+300=4396。
                        current_index = vadd(original32, chunk_start32, mask=mask32)
                        # 历史候选采用上一轮索引；新增候选采用本轮还原后的索引。
                        index = vselect(
                            previous_index, current_index, cond_mask=historical
                        )
                    # 写入本轮输出索引槽，超出 keep_count 的输出位置置零。
                    vstore(
                        output_indices,
                        row_in_group * MAX_TOPK + chunk * VL_U32,
                        vselect(index, reg_zero_u32, cond_mask=active),
                        mask32,
                    )

        if const_expr(retain_scores):
            # VF 7（仅 retain_scores=True）：将暂存分数写回合并候选区的前 MAX_TOPK 个历史槽，
            # 同时重建“本轮保留候选”的高字节累计直方图，作为下一轮的历史计数。
            # 不保留被淘汰候选的计数；未使用历史槽以零填充。
            with vf(mode="simd"):
                mask8, _ = update_mask(256, elem_bits=8)
                mask16, _ = update_mask(VL_U16, elem_bits=16)
                for row_in_group in tuple(range(RADIX_ROWS)):
                    row_idx = row_base + row_in_group
                    count0 = vdup_scalar(0, dtypes.uint16, mask=mask16)
                    count1 = vdup_scalar(0, dtypes.uint16, mask=mask16)
                    # 每轮两次加载，各读取 128 个 UINT16 分数，合计 256 个。
                    # 因此 512 个历史槽循环 2 次（共 4 次加载），每轮统计 256 个高字节。
                    for chunk in range(MAX_TOPK // RADIX_SCAN_CHUNK):
                        # 每个 FP32 存储单元承载两个 UINT16 编码分数；偏移按存储单元换算。
                        packed_scores_per_element = FP32_BYTES // U16_BYTES
                        # WS_OUT_KEYS：本轮入选分数在 score_workspace 中的起始偏移，
                        # 以 UINT16 分数元素为单位，当前为 0；不是索引区偏移或字节偏移。
                        # 加上行、分片偏移后，除以每个 FP32 单元容纳的分数个数，得到加载偏移。
                        workspace_offset = (
                            WS_OUT_KEYS
                            + row_in_group * MAX_TOPK
                            + chunk * RADIX_SCAN_CHUNK
                        ) // packed_scores_per_element
                        # 直接读取前、后 128 个分数；按位解释，不做浮点数值转换。
                        scores0 = vreinterpret_lanes(
                            vload(score_workspace, workspace_offset), dtypes.uint16
                        )
                        scores1 = vreinterpret_lanes(
                            vload(
                                score_workspace,
                                workspace_offset + VL_U16 // packed_scores_per_element,
                            ),
                            dtypes.uint16,
                        )
                        vstore(
                            self.merged_candidate_scores,
                            row_idx * STREAM_INPUT + chunk * RADIX_SCAN_CHUNK,
                            scores0,
                            mask16,
                        )
                        vstore(
                            self.merged_candidate_scores,
                            row_idx * STREAM_INPUT + chunk * RADIX_SCAN_CHUNK + VL_U16,
                            scores1,
                            mask16,
                        )
                        # 写回后从两组分数提取 256 个高字节，重建本轮保留候选的直方图。
                        _, scores_high8 = vdeintlv(
                            vreinterpret_lanes(scores0, dtypes.uint8),
                            vreinterpret_lanes(scores1, dtypes.uint8),
                        )
                        count0 = vhistogram_accumulate(
                            count0, scores_high8, mask=mask8, bin=0
                        )
                        count1 = vhistogram_accumulate(
                            count1, scores_high8, mask=mask8, bin=1
                        )
                    # 固定扫描 512 个历史槽，未使用槽以零填充。
                    vstore(self.hist, row_in_group * HISTOGRAM_BINS, count0, mask16)
                    vstore(
                        self.hist,
                        row_in_group * HISTOGRAM_BINS + VL_U16,
                        count1,
                        mask16,
                    )
                # 下一轮合并新增候选时复用这些历史分数及其高字节累计直方图。

    @jit
    def store_task(self, gm_out, q_block_start, q_block_count, row_base, index_source):
        for row_in_group in range(RADIX_ROWS):
            query = self.subblock * M_VECTOR + row_base + row_in_group
            if query < q_block_count:
                self._pack_one(row_in_group, index_source)
                dest = gm_out[q_block_start + query :, None]
                mem_copy(
                    tile_slice(dest, (1, OUTPUT_STRIDE), (0, 0)),
                    self.output.next_read(),
                )

    @jit
    def select_chunk(
        self,
        topk_chunk_index,
        k_topk_chunk_count,
        key_base,
        scan_len,
        history_indices,
        output_indices,
        scratch,
        radix_positions,
    ):
        # 四个分支的高字节直方图均已由 merge/gather 准备，直接复用。
        if topk_chunk_index + 1 == k_topk_chunk_count:
            # 最后一轮：输出最终索引，不再保存供下一轮使用的候选分数。
            if topk_chunk_index == 0:
                # 只有一个 chunk：没有历史候选，从 MAX_TOPK 偏移处扫描整片新增分数。
                # 新增候选未经过位置压缩，索引直接按 chunk 内位置还原。
                self.stream_topk_batch(
                    0,
                    key_base,
                    scratch,
                    radix_positions,
                    self.filter_map,
                    history_indices,
                    output_indices,
                    MAX_TOPK,
                    K_COMPRESSED_UNITS_PER_TOPK_CHUNK,
                    False,
                )
            else:
                # 最后一轮且非首轮：从偏移 0 扫描历史候选及过滤后的新增候选。
                # 扫描长度为 scan_len；新增候选通过过滤位置表还原原始 K 索引。
                self.stream_topk_batch(
                    0,
                    key_base,
                    scratch,
                    radix_positions,
                    self.filter_map,
                    history_indices,
                    output_indices,
                    0,
                    scan_len,
                    False,
                )
        else:
            # 后面还有 chunk：除索引外，还需保留本轮入选分数供下一轮合并。
            if topk_chunk_index == 0:
                # 首轮：跳过预留的历史区，扫描整片新增分数，建立首批历史 TopK。
                self.stream_topk_batch(
                    0,
                    key_base,
                    scratch,
                    radix_positions,
                    self.filter_map,
                    history_indices,
                    output_indices,
                    MAX_TOPK,
                    K_COMPRESSED_UNITS_PER_TOPK_CHUNK,
                    True,
                )
            else:
                # 中间轮：合并历史与过滤后的新增候选，按 scan_len 选择新的 TopK。
                # 根据过滤位置表还原新增索引，并保留入选分数继续下一轮。
                self.stream_topk_batch(
                    0,
                    key_base,
                    scratch,
                    radix_positions,
                    self.filter_map,
                    history_indices,
                    output_indices,
                    0,
                    scan_len,
                    True,
                )

    @jit
    def select_and_store(
        self,
        gm_score_workspace,
        core_base,
        query_positions,
        q_block_start,
        q_block_count,
        k_topk_chunk_count,
        gm_out,
    ):
        # 每个 AIV 按 RADIX_ROWS 行一组处理 Query；当前配置为每组 2 行。
        # 一组遍历完全部 K chunk 后写出最终 TopK，再处理下一组 Query。
        for row_base in range(0, M_VECTOR, RADIX_ROWS):
            # 初始化各行的可见压缩 K 数、保留候选数和可见原始 K 数。
            self.topk_parameters(
                query_positions, q_block_start, q_block_count, row_base
            )
            # 启动输入流水：先将首个 chunk 的分数从 GM 搬入 score_input。
            if k_topk_chunk_count > 0:
                self.load_scores(gm_score_workspace, core_base, row_base, 0)
            for topk_chunk_index in range(k_topk_chunk_count):
                # 索引双槽交替读写：读取上一轮保留的索引，另一槽接收本轮结果。
                history_indices = tile_slice(
                    self.topk_indices_slots,
                    (RADIX_ROWS, MAX_TOPK),
                    (topk_chunk_index % TOPK_INDEX_SLOT_COUNT, 0),
                )
                output_indices = tile_slice(
                    self.topk_indices_slots,
                    (RADIX_ROWS, MAX_TOPK),
                    ((topk_chunk_index + 1) % TOPK_INDEX_SLOT_COUNT, 0),
                )
                input_scores = self.score_input.next_read()
                # 输入双缓冲：提交下一 chunk 的搬运，再消费当前 chunk，
                # 使 MTE2 搬运有机会与当前组的 Vector 计算重叠。
                if topk_chunk_index + 1 < k_topk_chunk_count:
                    self.load_scores(
                        gm_score_workspace, core_base, row_base, topk_chunk_index + 1
                    )
                # VF 从参数区读取总可见长度，减去 chunk 起点后屏蔽不可见位置。
                key_base = topk_chunk_index * K_COMPRESSED_UNITS_PER_TOPK_CHUNK
                scan_len = dtypes.int64(STREAM_INPUT)
                if topk_chunk_index == 0:
                    # 首轮没有历史候选：一次读取就完成搬运、mask 和直方图统计。
                    self.merge_scores(input_scores, key_base)
                else:
                    # 后续轮先按历史阈值筛选可见候选，记录其 chunk 内位置。
                    self.filter_positions(input_scores, key_base)
                    # 等待过滤计数写回，由 Scalar 计算包含历史区的对齐扫描长度。
                    scan_len = self.filtered_length()
                    # 从同一输入槽收集入选分数，放到历史候选之后，并累加直方图。
                    self.gather_filtered(input_scores, scan_len)
                # 本轮保留数受 MAX_TOPK 和每行可见压缩 K 数量限制。
                self.update_keep()
                # 对历史候选与新增候选做 radix TopK，更新保留分数及输出索引槽。
                self.select_chunk(
                    topk_chunk_index,
                    k_topk_chunk_count,
                    key_base,
                    scan_len,
                    history_indices,
                    output_indices,
                    self.score_workspace,
                    self.radix_positions,
                )
            # 最后一轮写入的槽由 chunk 总数的奇偶决定；展开压缩块索引，
            # 拼接可见尾部 token，并写出本组 Query 的索引与有效长度。
            if k_topk_chunk_count > 0:
                self.store_task(
                    gm_out,
                    q_block_start,
                    q_block_count,
                    row_base,
                    tile_slice(
                        self.topk_indices_slots,
                        (RADIX_ROWS, MAX_TOPK),
                        (k_topk_chunk_count % TOPK_INDEX_SLOT_COUNT, 0),
                    ),
                )
            else:
                # 没有压缩 K chunk 时走空候选输出路径，仍处理尾部及有效长度。
                self.store_task(
                    gm_out, q_block_start, q_block_count, row_base, self.stream_indices
                )

    @jit
    def _reduce_heads_and_pack(
        self,
        head_scores,
        query_row,
        column_offset: Constexpr[int],
        fp32_mask,
        uint16_mask,
        reg_zero_fp32,
        sort_key_sign_bit,
        score_slot,
    ):
        # 在调用方的同一个 VF 内处理 128 列；low_64/high_64 表示前/后 64 列。
        # 保持两片交错加载、逐 head ReLU 和顺序累加，最后打包写出排序键。
        head0_low_64 = vload(
            head_scores,
            (query_row * HEADS) * K_COMPRESSED_UNITS_PER_BLOCK + column_offset,
        )
        head0_high_64 = vload(
            head_scores,
            (query_row * HEADS) * K_COMPRESSED_UNITS_PER_BLOCK + column_offset + VL_U32,
        )
        head1_low_64 = vload(
            head_scores,
            (query_row * HEADS + 1) * K_COMPRESSED_UNITS_PER_BLOCK + column_offset,
        )
        head1_high_64 = vload(
            head_scores,
            (query_row * HEADS + 1) * K_COMPRESSED_UNITS_PER_BLOCK
            + column_offset
            + VL_U32,
        )
        head2_low_64 = vload(
            head_scores,
            (query_row * HEADS + 2) * K_COMPRESSED_UNITS_PER_BLOCK + column_offset,
        )
        head2_high_64 = vload(
            head_scores,
            (query_row * HEADS + 2) * K_COMPRESSED_UNITS_PER_BLOCK
            + column_offset
            + VL_U32,
        )
        head3_low_64 = vload(
            head_scores,
            (query_row * HEADS + 3) * K_COMPRESSED_UNITS_PER_BLOCK + column_offset,
        )
        head3_high_64 = vload(
            head_scores,
            (query_row * HEADS + 3) * K_COMPRESSED_UNITS_PER_BLOCK
            + column_offset
            + VL_U32,
        )
        head0_low_64 = vmax(head0_low_64, reg_zero_fp32, mask=fp32_mask)
        head0_high_64 = vmax(head0_high_64, reg_zero_fp32, mask=fp32_mask)
        head1_low_64 = vmax(head1_low_64, reg_zero_fp32, mask=fp32_mask)
        head1_high_64 = vmax(head1_high_64, reg_zero_fp32, mask=fp32_mask)
        head2_low_64 = vmax(head2_low_64, reg_zero_fp32, mask=fp32_mask)
        head2_high_64 = vmax(head2_high_64, reg_zero_fp32, mask=fp32_mask)
        head3_low_64 = vmax(head3_low_64, reg_zero_fp32, mask=fp32_mask)
        head3_high_64 = vmax(head3_high_64, reg_zero_fp32, mask=fp32_mask)
        head_sum_low_64 = vadd(head0_low_64, head1_low_64, mask=fp32_mask)
        head_sum_high_64 = vadd(head0_high_64, head1_high_64, mask=fp32_mask)
        head_sum_low_64 = vadd(head_sum_low_64, head2_low_64, mask=fp32_mask)
        head_sum_high_64 = vadd(head_sum_high_64, head2_high_64, mask=fp32_mask)
        head_sum_low_64 = vadd(head_sum_low_64, head3_low_64, mask=fp32_mask)
        head_sum_high_64 = vadd(head_sum_high_64, head3_high_64, mask=fp32_mask)
        bf16_low_64 = vcast(head_sum_low_64, dtypes.bfloat16, mask=fp32_mask)
        bf16_high_64 = vcast(head_sum_high_64, dtypes.bfloat16, mask=fp32_mask)
        # 将两组各 64 个 BF16 分数的有效位紧凑合并为 128 个 UINT16 分数。vdeintlv交织去除fp32->bf16的空隙。
        packed_score_bits, _ = vdeintlv(
            vreinterpret_lanes(bf16_low_64, dtypes.uint16),
            vreinterpret_lanes(bf16_high_64, dtypes.uint16),
        )
        # 翻转最高位，保持非负分数的大小顺序，并使有效零分数与无效填充值 0 区分。
        sort_scores = vxor(packed_score_bits, sort_key_sign_bit, mask=uint16_mask)
        vstore(
            score_slot,
            query_row * K_COMPRESSED_UNITS_PER_BLOCK + column_offset,
            sort_scores,
            uint16_mask,
        )

    @jit
    def _pack_one(self, row_in_group, index_source):
        output_slot = self.output.next_write()
        # 将组内第 row_in_group 行的压缩 K 逻辑索引打包到 output UB；GM 写出由调用方完成。
        # 输出 VF：将入选压缩块展开为原始 token 索引，追加可见尾部，填充无效列并写入有效数量。
        with vf(mode="simd"):
            mask, _ = update_mask(VL_U32, elem_bits=32)
            # 广播本行参数：完整可见压缩块数、实际入选块数、可见原始 token 数。
            visible = vload_brc(
                self.visible_compressed_k_ub, row_in_group * PARAM_ROW_STRIDE
            )
            keep = vload_brc(self.keep_count_ub, row_in_group * PARAM_ROW_STRIDE)
            visible_uncompressed = vload_brc(
                self.visible_uncompressed_k_ub, row_in_group * PARAM_ROW_STRIDE
            )
            # 入选块展开后占 keep*4 列；完整可见块覆盖 visible*4 个原始 token。
            # 二者不同：尾部的原始位置由 visible 决定，输出位置由 keep 决定。
            selected_tokens = vshl(keep, TOKEN_COMPRESSION_SHIFT, mask=mask)
            complete_tokens = vshl(visible, TOKEN_COMPRESSION_SHIFT, mask=mask)
            # 未完成压缩块贡献 0～3 个尾部 token，最终有效数量为 keep*4 + tail。
            tail = vsub(visible_uncompressed, complete_tokens, mask=mask)
            count = vadd(selected_tokens, tail, mask=mask)
            vdup_scalar(0, dtypes.int32, mask=mask)
            invalid = vdup_scalar(-1, dtypes.int32, mask=mask)
            lane0 = varange(0, dtypes.int32)
            # 压缩比例为 4，lane0 & 3 生成重复的 [0, 1, 2, 3]，
            offsets = vand(
                lane0, vdup_scalar(MAX_CAUSAL_TAIL, dtypes.int32, mask=mask), mask=mask
            )
            # 静态展开 8 组：每组加载 64 个块索引，展开并写出 256 个 token 位置。
            for group in tuple(range(MAX_TOPK // VL_U32)):
                # 每个压缩块索引只加载一次，再复制为四个原始 token 索引。
                block = vreinterpret(
                    vload(index_source, row_in_group * MAX_TOPK + group * VL_U32),
                    dtypes.int32,
                )
                base = vshl(block, TOKEN_COMPRESSION_SHIFT, mask=mask)
                # 两级交错等价于每个块起点复制四次，如 [8,20,...] -> [8,8,8,8,20,20,20,20,...]。
                pair0, pair1 = _raw_reg.vinterleave(base, base)
                quad0, quad1 = _raw_reg.vinterleave(pair0, pair0)
                quad2, quad3 = _raw_reg.vinterleave(pair1, pair1)
                quads = (quad0, quad1, quad2, quad3)
                for part in tuple(range(TOKEN_COMPRESSION_RATIO)):
                    # lane 是整行输出列号；小于 selected_tokens 的列使用入选块展开结果。
                    lane = varange(
                        group * OUTPUT_PACK_CHUNK + part * VL_U32, dtypes.int32
                    )
                    selected = vcmp_lt(lane, selected_tokens, mask=mask)
                    token = vadd(quads[part], offsets, mask=mask)
                    # 尾部紧接入选块：输出第 selected_tokens 列对应原始 token complete_tokens。
                    tail_token = vadd(
                        complete_tokens,
                        vsub(lane, selected_tokens, mask=mask),
                        mask=mask,
                    )
                    token = vselect(token, tail_token, cond_mask=selected)
                    # 只保留前 count 个索引，其余填 -1；不足 512 块时，尾部也在此写出。
                    valid = vcmp_lt(lane, count, mask=mask)
                    vstore(
                        output_slot,
                        group * OUTPUT_PACK_CHUNK + part * VL_U32,
                        vselect(token, invalid, cond_mask=valid),
                        mask,
                    )
            # 以下处理 output UB 的最后 8 列（2048～2055），不是读取输入的最后一块。
            tail_mask, _ = update_mask(OUTPUT_STRIDE - MAX_EXPANDED_TOPK, elem_bits=32)
            lane = varange(MAX_EXPANDED_TOPK, dtypes.int32)
            token = vadd(
                complete_tokens, vsub(lane, selected_tokens, mask=mask), mask=mask
            )
            valid = vcmp_lt(lane, count, mask=tail_mask)
            # 仅实际存在的尾部位置保留 token 索引，其余位置先填 -1。
            result = vselect(token, invalid, cond_mask=valid)
            # 将第 2051 列单独标记，写出时改存有效数量；2052～2055 保持 -1 作为对齐填充。
            is_count = vcmp_eq_scalar(lane, COUNT_COLUMN, mask=tail_mask)
            vstore(
                output_slot,
                MAX_EXPANDED_TOPK,
                vselect(count, result, cond_mask=is_count),
                tail_mask,
            )


@kernel
class QSAIndexerKernel:
    """按照 Metadata 分配的 Q32 任务执行评分、TopK 和索引展开。"""

    def __init__(self, spec):
        self.spec = spec

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        page_table: Tensor,
        actual_seq: Tensor,
        query_positions: Tensor,
        metadata: Tensor,
        gm_out: Tensor,
        gm_score_workspace: Tensor,
    ):
        ub_storage = dsl.UB.view(262144)
        score_channel = dsl.make_ring_buffer(
            [
                ub_storage[
                    CROSS_CORE_SCORE_ADDR
                    + slot
                    * (
                        SCORE_ROWS_PER_VECTOR * K_COMPRESSED_UNITS_PER_BLOCK * 4
                    ) : CROSS_CORE_SCORE_ADDR
                    + slot * (SCORE_ROWS_PER_VECTOR * K_COMPRESSED_UNITS_PER_BLOCK * 4)
                    + SCORE_ROWS_PER_VECTOR * K_COMPRESSED_UNITS_PER_BLOCK * 4,
                ]
                .view(dtype=dtypes.float32)
                .view((SCORE_ROWS_PER_VECTOR, K_COMPRESSED_UNITS_PER_BLOCK))
                for slot in range(SCORE_CHANNEL_DEPTH)
            ]
        )
        cube_ops = Matmul()
        vector = Vector()
        vector.initialize()
        section_count = dtypes.int64(metadata[HEAD_SECTION_COUNT_INDEX])
        for section_index in range(section_count):
            base = (
                HEAD_METADATA_STRIDE
                + section_index * SECTION_METADATA_STRIDE
                + get_block_idx() * CORE_METADATA_STRIDE
            )
            start_bn = dtypes.int64(metadata[base + CORE_BN_BEGIN_INDEX])
            m_begin_metadata = dtypes.int64(metadata[base + CORE_M_BEGIN_INDEX])
            end_bn = dtypes.int64(metadata[base + CORE_BN_END_INDEX])
            m_end_metadata = dtypes.int64(metadata[base + CORE_M_END_INDEX])
            # 当前 KV head 数为 1，组合调度编号 bn 等于 batch 索引。
            for bn in range(start_bn, end_bn + 1):
                batch_q_begin = dtypes.int64(actual_seq[bn])
                batch_q_end = dtypes.int64(actual_seq[bn + 1])
                first_m = dtypes.int64(0)
                if bn == start_bn:
                    first_m = m_begin_metadata
                last_m = (batch_q_end - batch_q_begin + QUERY_TILE - 1) // QUERY_TILE
                if bn == end_bn:
                    last_m = m_end_metadata
                for metadata_m in range(first_m, last_m):
                    self._process_metadata_m(
                        q,
                        k,
                        page_table,
                        query_positions,
                        gm_out,
                        gm_score_workspace,
                        cube_ops,
                        vector,
                        score_channel,
                        bn,
                        batch_q_begin,
                        batch_q_end,
                        metadata_m,
                    )

    @jit
    def _compute_score_group(
        self,
        cube_ops,
        vector,
        score_channel,
        gm_score_workspace,
        core_base,
        k_block_idx,
    ):
        cube_ops.compute_qk()
        cube_ops.store_score(score_channel)
        vector.relu_reduce_stream(score_channel.next_read())
        vector.save_scores(gm_score_workspace, core_base, k_block_idx)

    @jit
    def _process_metadata_m(
        self,
        q,
        k,
        page_table,
        query_positions,
        gm_out,
        gm_score_workspace,
        cube_ops,
        vector,
        score_channel,
        bn,
        batch_q_begin,
        batch_q_end,
        metadata_m,
    ):
        # 当前 Q 块在 Q 张量中的起始行号。
        q_block_start = batch_q_begin + metadata_m * QUERY_TILE
        # 当前 Q 块的有效 Query 行数。
        q_block_count = min(QUERY_TILE, batch_q_end - q_block_start)
        # 当前 Q 块需要扫描的压缩 K 单元数量。
        k_compressed_unit_count = dtypes.int64(0)
        # 当前 Q 块内的 Query 行号，用于统计可见 K 范围。
        for row in range(q_block_count):
            k_compressed_unit_count = max(
                k_compressed_unit_count,
                (dtypes.int64(query_positions[q_block_start + row]) + 1)
                // TOKEN_COMPRESSION_RATIO,
            )
        # 当前 AIV 在 GM 分数工作区中的起始行偏移。
        core_base = (
            get_block_idx() * VECTOR_SUBBLOCKS_PER_CUBE + get_subblock_id()
        ) * M_VECTOR
        # 评分阶段需要处理的 K 基本块数量。
        k_block_count = (
            k_compressed_unit_count + K_COMPRESSED_UNITS_PER_BLOCK - 1
        ) // K_COMPRESSED_UNITS_PER_BLOCK
        # TopK 阶段需要处理的 chunk 数量。
        k_topk_chunk_count = (
            k_compressed_unit_count + K_COMPRESSED_UNITS_PER_TOPK_CHUNK - 1
        ) // K_COMPRESSED_UNITS_PER_TOPK_CHUNK
        if k_block_count > 0:
            cube_ops.load_q_resident(q, q_block_start)
            # 当前参与评分的 K 基本块编号。
            for k_block_idx in range(k_block_count):
                cube_ops.load_key(
                    k,
                    page_table,
                    bn,
                    k_block_idx * K_COMPRESSED_UNITS_PER_BLOCK,
                    k_compressed_unit_count,
                )
                self._compute_score_group(
                    cube_ops,
                    vector,
                    score_channel,
                    gm_score_workspace,
                    core_base,
                    k_block_idx,
                )
            # 最后一个 TopK chunk 中需要补零的 K 基本块编号。
            for k_block_idx in range(
                k_block_count, k_topk_chunk_count * K_BLOCKS_PER_TOPK_CHUNK
            ):
                vector.zero_scores()
                vector.save_scores(gm_score_workspace, core_base, k_block_idx)
            # 评分输出与 TopK 候选复用 UB；先完成 MTE3 写出，再切换到
            # MTE2 读取工作区及 Vector 覆写候选，避免短 K 路径读到旧分数。
            vec_sync_all()
            vector.select_and_store(
                gm_score_workspace,
                core_base,
                query_positions,
                q_block_start,
                q_block_count,
                k_topk_chunk_count,
                gm_out,
            )
        else:
            # 当前 AIV 内待输出 Query 组的起始行号。
            for row_base in range(0, M_VECTOR, RADIX_ROWS):
                vector.topk_parameters(
                    query_positions, q_block_start, q_block_count, row_base
                )
                vector.store_task(
                    gm_out, q_block_start, q_block_count, row_base, vector.empty_indices
                )


class QSAIndexerLauncher:
    """保存编译配置并发射融合 QSA Indexer Kernel。"""

    def __init__(self, total_query, score_groups, block_dim):
        self.total_query = total_query
        self.score_groups = score_groups
        self.block_dim = block_dim
        self.spec = Specialization(block_dim)

    @property
    def output_shape(self):
        return self.total_query, OUTPUT_STRIDE

    @property
    def workspace_shape(self):
        return (
            self.block_dim * VECTOR_SUBBLOCKS_PER_CUBE * M_VECTOR,
            self.score_groups * K_COMPRESSED_UNITS_PER_BLOCK,
        )

    @host
    def launch(
        self,
        q: Tensor,
        k: Tensor,
        page_table: Tensor,
        actual_seq: Tensor,
        query_positions: Tensor,
        metadata: Tensor,
        gm_out: Tensor,
        gm_score_workspace: Tensor,
    ):
        QSAIndexerKernel(self.spec)[self.block_dim](
            q,
            k,
            page_table,
            actual_seq,
            query_positions,
            metadata,
            gm_out,
            gm_score_workspace,
        )


def _make_launcher(tiling):
    """根据已校验的 Tiling 信息构造 Launcher。"""
    return QSAIndexerLauncher(
        total_query=tiling.total_query,
        score_groups=tiling.score_groups,
        block_dim=tiling.block_dim,
    )


_LAUNCH_CACHE_LOCK = RLock()


def _dynamic_launch_specs(block_dim):
    """构造覆盖不同 Batch、Q 长度和压缩 K 容量的主 Kernel 规格。"""
    guarded_query_rows = Dim("GUARDED_QUERY_ROWS", min=QUERY_TILE)
    compressed_page_count = Dim("COMPRESSED_PAGE_COUNT", min=1)
    page_table_batch = Dim("PAGE_TABLE_BATCH", min=1)
    page_table_width = Dim("PAGE_TABLE_WIDTH", min=1)
    sequence_offset_count = Dim("SEQUENCE_OFFSET_COUNT", min=3)
    query_count = Dim("QUERY_COUNT", min=1)
    metadata_capacity_dim = Dim(
        "METADATA_CAPACITY",
        min=METADATA_ALIGNMENT_ELEMS,
        multiple_of=METADATA_ALIGNMENT_ELEMS,
    )
    score_workspace_width = Dim(
        "SCORE_WORKSPACE_WIDTH",
        min=K_COMPRESSED_UNITS_PER_TOPK_CHUNK,
        multiple_of=K_COMPRESSED_UNITS_PER_TOPK_CHUNK,
    )
    return (
        TensorSpec((guarded_query_rows, DIM), dtypes.bfloat16),
        TensorSpec((compressed_page_count, PAGE_SIZE, DIM), dtypes.bfloat16),
        TensorSpec((page_table_batch, page_table_width), dtypes.int32),
        TensorSpec((sequence_offset_count,), dtypes.int32),
        TensorSpec((query_count,), dtypes.int32),
        TensorSpec((metadata_capacity_dim,), dtypes.int32),
        TensorSpec((query_count, OUTPUT_STRIDE), dtypes.int32),
        TensorSpec(
            (
                block_dim * VECTOR_SUBBLOCKS_PER_CUBE * M_VECTOR,
                score_workspace_width,
            ),
            dtypes.uint16,
        ),
    )


@lru_cache(maxsize=LAUNCH_CACHE_CAPACITY)
def _compile_launch(block_dim, device, process_id):
    """按设备、进程和 Kernel 网格缓存动态形状编译句柄。"""
    # 设备与进程参与缓存键，避免跨设备或跨进程复用编译句柄。
    del device, process_id
    # total_query 和 score_groups 只决定 Host 输出/工作区容量，不参与 Kernel 特化。
    launcher = QSAIndexerLauncher(total_query=1, score_groups=1, block_dim=block_dim)
    return dsl.compile(launcher.launch, *_dynamic_launch_specs(block_dim))


def _launch_cached(launcher, args):
    """复用编译对象，每次绑定当前 Tensor 地址及当前 NPU stream。"""
    with torch.npu.device(args[0].device):
        # 锁仅保护编译与缓存查找，不覆盖执行阶段。
        with _LAUNCH_CACHE_LOCK:
            program = _compile_launch(launcher.block_dim, args[0].device, os.getpid())
        program(*args)


def _prepare_inputs(
    q,
    compressed_k,
    block_table,
    actual_seq,
    query_positions,
    *,
    metadata,
    block_dim,
):
    """准备主 Kernel 使用的张量、输出和工作区。"""
    tiling = validate_and_resolve(
        q,
        compressed_k,
        block_table,
        actual_seq,
        query_positions,
        metadata,
        block_dim,
        q_heads=HEADS,
        kv_heads=KV_HEADS,
        head_dim=DIM,
        page_size=PAGE_SIZE,
        metadata_capacity=metadata_capacity,
        aic_capacity=AIC_METADATA_CORE_CAPACITY,
        compressed_units_per_topk_chunk=K_COMPRESSED_UNITS_PER_TOPK_CHUNK,
        blocks_per_topk_chunk=K_BLOCKS_PER_TOPK_CHUNK,
    )

    # Query、压缩 Key 和 Metadata 自动连续化；辅助输入搬到 Query 所在设备后连续化。
    q = q.contiguous()
    compressed_k = compressed_k.contiguous()
    device_table = block_table.to(q.device).contiguous()
    device_positions = query_positions.to(q.device).contiguous()
    device_starts = actual_seq.to(q.device).contiguous()
    metadata = metadata.contiguous()

    total = tiling.total_query
    launcher = _make_launcher(tiling)
    output = torch.empty(launcher.output_shape, dtype=torch.int32, device=q.device)
    if total == 0:
        return launcher, None, output[:, :OUTPUT_COLUMN_COUNT]

    guarded = torch.cat(
        (
            q,
            torch.zeros((QUERY_TILE - 1, HEADS, DIM), dtype=q.dtype, device=q.device),
        ),
        dim=0,
    )
    cache = compressed_k
    if cache.shape[0] == 0:
        cache = torch.zeros(
            (1, PAGE_SIZE, KV_HEADS, DIM), dtype=q.dtype, device=q.device
        )
    if block_table.shape[1] == 0:
        device_table = torch.zeros(
            (block_table.shape[0], 1), dtype=torch.int32, device=q.device
        )
    # Metadata 的排他结束坐标允许 bn==B，因此追加一个零任务哨兵请求。
    sentinel = torch.full((1,), total, dtype=torch.int32, device=q.device)
    device_starts = torch.cat((device_starts, sentinel))
    gm_score_workspace = torch.empty(
        launcher.workspace_shape, dtype=torch.uint16, device=q.device
    )
    call_args = (
        guarded.view(-1, DIM),
        cache.view(-1, PAGE_SIZE, DIM),
        device_table,
        device_starts,
        device_positions,
        metadata,
        output,
        gm_score_workspace,
    )
    return launcher, call_args, output[:, :OUTPUT_COLUMN_COUNT]


def qsa_indexer(
    q,
    compressed_k,
    block_table,
    actual_seq,
    query_positions,
    *,
    metadata=None,
    block_dim=None,
):
    """使用外部 AICPU Metadata 发射算子，不主动同步设备。"""
    launcher, call_args, output = _prepare_inputs(
        q,
        compressed_k,
        block_table,
        actual_seq,
        query_positions,
        metadata=metadata,
        block_dim=block_dim,
    )
    if call_args is not None:
        _launch_cached(launcher, call_args)
    return output
