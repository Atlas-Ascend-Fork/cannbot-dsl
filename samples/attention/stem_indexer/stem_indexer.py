# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stem Indexer：Host 准备、AOT 入口与 p1/p2 融合 Cube+Vector Kernel。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from threading import RLock

from cannbotdsl import make_layout
from cannbotdsl.lang.host import host
import torch
import torch.nn.functional as F
import cannbotdsl as dsl
from cannbotdsl.reg import vpack, vmin, vscatter, vdiv, vshr

import cannbotdsl.reg as _raw_reg

from cannbotdsl import (
    Constexpr,
    Dim,
    MemLoc,
    Tensor,
    TensorSpec,
    const_expr,
    dtypes,
)
from cannbotdsl.ops.arch import get_block_idx, get_subblock_id
from cannbotdsl.buffer import Buffer
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.tensor import tile_slice
from cannbotdsl.reg import (
    update_mask,
    vadd,
    vadds,
    varange,
    vcast,
    vgather,
    vhistogram_accumulate,
    vload,
    vload_deinterleave,
    vmem_bar,
    vmax,
    vmaxs,
    vmins,
    vmadd,
    vmuls,
    vnot,
    vreinterpret,
    vreinterpret_lanes,
    vselect,
    vshl,
    vstore,
    vstore_first,
    vsub,
)

if __package__:
    from .stem_indexer_checker import (
        get_effective_core_counts as get_effective_core_counts,
        validate_and_resolve,
    )
else:
    from stem_indexer_checker import (
        get_effective_core_counts as get_effective_core_counts,
        validate_and_resolve,
    )

MAX_TOPK = 256
TOPK_BUDGET_DIRECT_PROMPT_LIMIT = 56
TOPK_BUDGET_LONG_PROMPT_THRESHOLD = 160
TOPK_BUDGET_MEDIUM_PROMPT_RATIO = 0.2
TOPK_BUDGET_LONG_PROMPT_RATIO = 0.1
TOPK_BUDGET_BASE_OFFSET = 30.0

# Vector寄存器操作。
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


# 输入属性与 Host specialization
@dataclass(frozen=True)
class FixedAttributes:
    causal: bool = True
    stem_block_size: int = 128
    stem_stride: int = 16
    alpha: float = 1.0
    initial_blocks: int = 4
    window_size: int = 4
    topk_score_precision: int = 2


M_TILE = 64

N_TILE = 256

ROW_PARAM_STRIDE = 16  # 64字节，满足标量广播加载的安全要求。

LENS_STORAGE_STRIDE = 8  # 每行32字节，满足GM MTE写入要求。
P1_RADIX_STATE_STRIDE = 8  # p1阈值字节和剩余排名的UB行间距，与GM输出布局独立。


def ceil_div(value: int, divisor: int) -> int:
    return value // divisor + (value % divisor != 0)


def align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


# 设备侧融合 Kernel

M_VECTOR = 32

N_CHUNK = 64

N_OUTER = 256

K_L1 = 1024

K_L0 = 256
K_L0_CHUNKS_PER_L1 = K_L1 // K_L0

RADIX_ROWS = 4

# 单次radix处理一个uint8字节，对应0～255共256个累计桶。
HISTOGRAM_BINS = 256

VL_U16 = 128

VL_U32 = 64

STREAM_INPUT = 512

STREAM_POS_STRIDE = STREAM_INPUT + 16

OUTPUT_CAPACITY = 320

WS_HIST = 0

WS_IDX_HIGH = WS_HIST + RADIX_ROWS * HISTOGRAM_BINS

WS_IDX_LOW = WS_IDX_HIGH + RADIX_ROWS * HISTOGRAM_BINS

WS_NEXT_K = WS_IDX_LOW + RADIX_ROWS * HISTOGRAM_BINS

WS_POSITIONS = WS_NEXT_K + RADIX_ROWS * VL_U16

WS_OUT_KEYS = WS_POSITIONS + RADIX_ROWS * STREAM_POS_STRIDE

WS_OUT_INDICES = WS_OUT_KEYS + RADIX_ROWS * MAX_TOPK

WS_USED_U16 = WS_OUT_INDICES + RADIX_ROWS * MAX_TOPK

PARAM_VISIBLE_BYTE_BASE = 4

HEAD_METADATA_STRIDE = 16

FA_METADATA_STRIDE = 16

AIC_METADATA_CORE_CAPACITY = 36

SECTION_METADATA_STRIDE = AIC_METADATA_CORE_CAPACITY * FA_METADATA_STRIDE


class Matmul:
    """管理Cube侧Channel，完成Q/K加载、矩阵乘和分数搬运。"""

    def __init__(self):
        # 遍历整个S2期间，将完整Q[64, 2048]分块驻留在L1；两个单槽Channel分别保存K维上的1024元素分块。
        # Q前1024维的L1驻留分片，供整个S2循环的MTE1读取。
        self.q_l1_0 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((M_TILE, K_L1)), depth=1
        ).next_write()
        # Q后1024维的L1驻留分片，与前半分片分别管理生命周期。
        self.q_l1_1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((M_TILE, K_L1)), depth=1
        ).next_write()
        # 两个64×1024 K-L1槽由MTE2交替使用，仅在槽复用时等待，使后续K块加载与另一槽的MTE1/MMAD计算重叠。
        # K前1024维的L1分片，承接MTE2加载并交给MTE1。
        self.k_l1_0 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((N_CHUNK, K_L1)), depth=1
        ).next_write()
        # K后1024维的L1分片，供后续特征维分块计算。
        self.k_l1_1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((N_CHUNK, K_L1)), depth=1
        ).next_write()
        # Q的L0A偶数分片槽，将MTE1搬入的64×256数据交给MMAD。
        self.l0a_0 = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((M_TILE, K_L0)), depth=1
        ).next_write()
        # Q的L0A奇数分片槽，与偶数槽交替使用。
        self.l0a_1 = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((M_TILE, K_L0)), depth=1
        ).next_write()
        # K的L0B偶数分片槽，将MTE1搬入的64×256数据交给MMAD。
        self.l0b_0 = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((N_CHUNK, K_L0)), depth=1
        ).next_write()
        # K的L0B奇数分片槽，与对应L0A槽配对参与MMAD。
        self.l0b_1 = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((N_CHUNK, K_L0)), depth=1
        ).next_write()
        # 保存64×64的FP32累加结果，双槽管理MMAD到Fixpipe的交接与复用。
        self.l0c = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((M_TILE, N_CHUNK)), depth=2
        )

    @staticmethod
    def _q_tile(gm_q, batch, q_head, q_block0, k_chunk):
        base = gm_q[batch, q_head, k_chunk, q_block0:, None]
        return tile_slice(base, (M_TILE, K_L1), (0, 0))

    @staticmethod
    def _k_tile(gm_k, batch, kv_head, key_block0, k_chunk):
        base = gm_k[batch, kv_head, k_chunk, key_block0:, None]
        return tile_slice(base, (N_CHUNK, K_L1), (0, 0))

    def load_q_resident(self, gm_q, batch, q_head, q_block_start):
        # Q-L1只由MTE1读取；确保读取先于下一次MTE2覆盖完成，不排空无关的MMAD/Fixpipe工作。
        mem_copy(
            self.q_l1_0,
            self._q_tile(gm_q, batch, q_head, q_block_start, 0),
            options=copy_options(l2_cache_ctl=1),
        )
        mem_copy(
            self.q_l1_1,
            self._q_tile(gm_q, batch, q_head, q_block_start, 1),
            options=copy_options(l2_cache_ctl=1),
        )

    @jit
    def compute_qk(self, gm_k, batch, kv_head, chunk_base):
        # 所有K分块共用同一个累加器，保留中间累加结果。
        accumulator = self.l0c.next_write()
        for k_l1_chunk in tuple(range(2)):
            q_l1 = self.q_l1_0 if k_l1_chunk == 0 else self.q_l1_1
            key_op_index = k_l1_chunk
            if const_expr(key_op_index % 2 == 0):
                k_l1 = self.k_l1_0
            else:
                k_l1 = self.k_l1_1
            mem_copy(
                k_l1,
                self._k_tile(gm_k, batch, kv_head, chunk_base, k_l1_chunk),
                options=copy_options(l2_cache_ctl=1),
            )
            for k_l0_chunk in tuple(range(K_L0_CHUNKS_PER_L1)):
                op_index = k_l1_chunk * K_L0_CHUNKS_PER_L1 + k_l0_chunk
                if const_expr(op_index % 2 == 0):
                    l0a = self.l0a_0
                    l0b = self.l0b_0
                else:
                    l0a = self.l0a_1
                    l0b = self.l0b_1
                mem_copy(l0a, tile_slice(q_l1, (M_TILE, K_L0), (0, k_l0_chunk)))
                mem_copy(l0b, tile_slice(k_l1, (N_CHUNK, K_L0), (0, k_l0_chunk)))
                global_k = k_l1_chunk * K_L0_CHUNKS_PER_L1 + k_l0_chunk
                matmul(
                    accumulator,
                    l0a,
                    l0b,
                    init=(global_k == 0),
                )

    def store_score(self, score_channel):
        mem_copy(score_channel.next_write(), self.l0c.next_read().partition(dim=0))


class Vector:
    """管理Vector侧UB，执行行参数生成、Scale/Bias、TopK和输出。"""

    def __init__(self, precision, attrs, subblock):
        self.precision = precision
        # 固定片上输出容量，使不同KV物理长度能够共用同一份动态编译产物。
        self.output_capacity = OUTPUT_CAPACITY
        self.causal = attrs.causal
        self.alpha_is_one = attrs.alpha == 1.0
        self.stem_block_size = attrs.stem_block_size
        self.initial_blocks = attrs.initial_blocks
        self.window_size = attrs.window_size
        self.score_scale = 1.0 / ((attrs.stem_block_size // attrs.stem_stride) ** 2)
        self.subblock = subblock
        # Vector工作区：p1存当前N256的u32排序键，p2复用为TopK临时区；不直接接收Fixpipe。
        self.score_ub = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((M_VECTOR, N_OUTER))
        )
        # MTE2加载当前N256的偏置，Vector各Q行共享读取。
        self.bias_ub = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((N_OUTER,)), depth=1
        ).next_write()
        # Vector批量生成32行可见范围、预算等参数，供后续VF及S2轮次复用。
        self.params_ub = Buffer(
            MemLoc.UB, dtypes.int32, make_layout((M_VECTOR, ROW_PARAM_STRIDE))
        )

        # p1基数排序状态：保存阈值字节和剩余排名，按广播加载需要保留行间距。
        self.p1_radix_state = Buffer(
            MemLoc.UB, dtypes.int32, make_layout((M_VECTOR, P1_RADIX_STATE_STRIDE))
        )
        # Vector打包最终索引后交给MTE3；非直出行最多为4+256+4项。
        self.out_rows = RingBuffer(
            MemLoc.UB,
            dtypes.int32,
            make_layout((M_VECTOR, self.output_capacity)),
            depth=1,
        ).next_write()
        # 长度暂存仅在Vector侧使用，最终长度通过独立Channel从V流水交给MTE3。
        self.output_lens = RingBuffer(
            MemLoc.UB, dtypes.int32, make_layout((M_VECTOR, 1)), depth=1
        ).next_write()
        if self.precision == 2:
            # p2每行保存历史256项与当前256项的u16排序键，供流式合并。
            self.merge_values = Buffer(
                MemLoc.UB, dtypes.uint16, make_layout((M_VECTOR, STREAM_INPUT))
            )
            # p2历史保留项的全局u32索引，跨N256轮次保留。
            self.global_indices = Buffer(
                MemLoc.UB, dtypes.uint32, make_layout((M_VECTOR, MAX_TOPK))
            )
            # vstore_unalign要求目标数据类型一致，因此这些压缩和基数排序区域独立于MM1分数槽。
            # p2四行radix的256桶计数，高低字节阶段复用。
            self.hist = Buffer(
                MemLoc.UB, dtypes.uint16, make_layout((RADIX_ROWS, HISTOGRAM_BINS))
            )
            # 每行使用独立的物理压缩缓冲区，使同一VF生成四个不同的C++ UnalignRegForStore对象。
            # 暂存每行高字节阈值桶的压缩查找结果。
            self.idx_high = tuple(
                Buffer(MemLoc.UB, dtypes.uint16, make_layout((HISTOGRAM_BINS,)))
                for _ in range(RADIX_ROWS)
            )
            # 暂存每行低字节阈值桶的压缩查找结果。
            self.idx_low = tuple(
                Buffer(MemLoc.UB, dtypes.uint16, make_layout((HISTOGRAM_BINS,)))
                for _ in range(RADIX_ROWS)
            )
            # 高字节筛选后剩余的目标排名，交给低字节阶段读取。
            self.next_k = Buffer(
                MemLoc.UB, dtypes.uint16, make_layout((RADIX_ROWS, VL_U16))
            )
            # 四行各自的候选局部位置，供GT/EQ压缩和gather使用。
            self.tmp_indices = tuple(
                Buffer(MemLoc.UB, dtypes.uint16, make_layout((STREAM_POS_STRIDE,)))
                for _ in range(RADIX_ROWS)
            )
            # p2本轮选中的四行全局索引暂存，随后更新历史状态。
            self.stream_out_indices = Buffer(
                MemLoc.UB, dtypes.uint32, make_layout((RADIX_ROWS, MAX_TOPK))
            )
        else:
            # precision=1与precision=2采用相同的历史256项加当前256项的流式生命周期。当前u32排序键覆盖已消费的FP32 MM1分数槽，仅历史键和索引跨N256组保留。
            # p1每行历史保留的256个u32排序键，跨N256轮次复用。
            self.p1_merge_values = Buffer(
                MemLoc.UB, dtypes.uint32, make_layout((M_VECTOR, MAX_TOPK))
            )
            # p1历史保留键对应的全局u32索引。
            self.p1_global_indices = Buffer(
                MemLoc.UB, dtypes.uint32, make_layout((M_VECTOR, MAX_TOPK))
            )
            # p1四行radix桶计数，四轮字节筛选复用。
            self.p1_hist = Buffer(
                MemLoc.UB, dtypes.uint16, make_layout((RADIX_ROWS, HISTOGRAM_BINS))
            )
            # p1每行当前字节阈值桶的压缩查找结果。
            self.p1_idx_bucket = tuple(
                Buffer(MemLoc.UB, dtypes.uint16, make_layout((HISTOGRAM_BINS,)))
                for _ in range(RADIX_ROWS)
            )
            # p1四行候选的局部位置，供最终选择和gather使用。
            self.p1_tmp_indices = tuple(
                Buffer(MemLoc.UB, dtypes.int32, make_layout((STREAM_POS_STRIDE,)))
                for _ in range(RADIX_ROWS)
            )
            # p1本轮选中的四行排序键暂存，防止更新时覆盖输入。
            self.p1_out_keys = Buffer(
                MemLoc.UB, dtypes.uint32, make_layout((RADIX_ROWS, MAX_TOPK))
            )
            # p1本轮选中键对应的全局索引暂存。
            self.p1_out_indices = Buffer(
                MemLoc.UB, dtypes.uint32, make_layout((RADIX_ROWS, MAX_TOPK))
            )

    @staticmethod
    def _bias_tile(gm_bias, batch, kv_head, key_block0):
        base = gm_bias[batch, kv_head, key_block0:]
        return tile_slice(base, (N_OUTER,), (0,))

    @staticmethod
    def _lens_prefix_tile(gm_lens, batch, q_head, q_block0, rows=M_VECTOR):
        base = gm_lens[batch, q_head, q_block0:, None]
        return tile_slice(base, (rows, 1), (0, 0))

    @jit
    def build_task_params(
        self,
        gm_q_seq_lens,
        gm_kv_seq_lens,
        gm_num_prompt_tokens,
        batch,
        q_block_start,
        q_block_count,
        q_blocks,
        kv_blocks,
        shared_budget,
        decode,
        prompt_blocks,
        alpha,
    ):
        valid_rows = dtypes.int32(
            max(0, min(q_block_count - self.subblock * M_VECTOR, M_VECTOR))
        )
        q_block0 = q_block_start + self.subblock * M_VECTOR
        visible_base = kv_blocks
        visible_step = dtypes.int64(0)
        if const_expr(self.causal):
            if decode == 0:
                visible_base = kv_blocks - q_blocks + q_block0 + 1
                visible_step = dtypes.int64(1)
        base_i32 = dtypes.int32(visible_base)
        step_i32 = dtypes.int32(visible_step)
        kv_i32 = dtypes.int32(kv_blocks)
        budget_i32 = dtypes.int32(shared_budget)
        start_i32 = dtypes.int32(0)
        position_i32 = dtypes.int32(0)
        decay_i32 = dtypes.int32(0)
        denominator_f = dtypes.float32(1.0)
        start_f = dtypes.float32(0.0)
        delta_f = dtypes.float32(0.0)
        if const_expr(not self.alpha_is_one):
            start = prompt_blocks
            if prompt_blocks >= TOPK_BUDGET_DIRECT_PROMPT_LIMIT:
                if prompt_blocks < TOPK_BUDGET_LONG_PROMPT_THRESHOLD:
                    start = dtypes.int64(
                        dtypes.float32(prompt_blocks) * TOPK_BUDGET_MEDIUM_PROMPT_RATIO
                        + TOPK_BUDGET_BASE_OFFSET
                    )
                else:
                    start = dtypes.int64(
                        dtypes.float32(prompt_blocks) * TOPK_BUDGET_LONG_PROMPT_RATIO
                        + TOPK_BUDGET_BASE_OFFSET
                    )
            start_i32 = dtypes.int32(start)
            position_i32 = dtypes.int32(q_block0 + kv_blocks - q_blocks)
            decay_i32 = dtypes.int32(prompt_blocks - start)
            # 使用安全分母，同时覆盖未启用衰减的分支。
            denominator_f = dtypes.float32(max(prompt_blocks - start - 1, 1))
            start_f = dtypes.float32(start)
            end_f = start_f * alpha
            delta_f = end_f - start_f
        # 32个lane各处理一行，计算可见范围、TopK选择数、输出长度和直出标记，
        # 按[32, ROW_PARAM_STRIDE]逐行布局写入params_ub。
        with vf(mode="simd"):
            mask32, _ = update_mask(M_VECTOR, elem_bits=32)
            row = varange(0, dtypes.int32)
            base = vdup_scalar(base_i32, dtypes.int32, mask=mask32)
            kv = vdup_scalar(kv_i32, dtypes.int32, mask=mask32)
            budget = vdup_scalar(budget_i32, dtypes.int32, mask=mask32)
            if const_expr(not self.alpha_is_one):
                start_reg = vdup_scalar(start_i32, dtypes.int32, mask=mask32)
                position = vadds(row, position_i32, mask=mask32)
                relative = vsub(position, start_reg, mask=mask32)
                relative_f = vcast(relative, dtypes.float32, mask=mask32)
                denominator = vdup_scalar(denominator_f, dtypes.float32, mask=mask32)
                ratio = vdiv(relative_f, denominator, mask=mask32)
                # 与标量 FP32 插值保持一次舍入，避免整数边界处两次舍入使 budget 多 1。
                interpolated = vmadd(
                    ratio,
                    vdup_scalar(delta_f, dtypes.float32, mask=mask32),
                    vdup_scalar(start_f, dtypes.float32, mask=mask32),
                    mask=mask32,
                )
                decayed = vcast(interpolated, dtypes.int32, mask=mask32, rounding="rz")
                decayed = vmin(vmaxs(decayed, 1, mask=mask32), start_reg, mask=mask32)
                budget = vselect(
                    decayed,
                    start_reg,
                    cond_mask=vcmp_ge(position, start_reg, mask=mask32),
                )
                decay_reg = vdup_scalar(decay_i32, dtypes.int32, mask=mask32)
                one_reg = vdup_scalar(1, dtypes.int32, mask=mask32)
                budget = vselect(
                    budget,
                    start_reg,
                    cond_mask=vcmp_gt(decay_reg, one_reg, mask=mask32),
                )
                budget = vmins(budget, MAX_TOPK, mask=mask32)
            rows = vdup_scalar(valid_rows, dtypes.int32, mask=mask32)
            zero = vdup_scalar(0, dtypes.int32, mask=mask32)
            one = vdup_scalar(1, dtypes.int32, mask=mask32)
            visible = vmin(
                vmaxs(
                    vadd(base, vmuls(row, step_i32, mask=mask32), mask=mask32),
                    0,
                    mask=mask32,
                ),
                kv,
                mask=mask32,
            )
            valid = vcmp_lt(row, rows, mask=mask32)
            visible = vselect(visible, zero, cond_mask=valid)
            sink = vmins(visible, self.initial_blocks, mask=mask32)
            window = vmax(
                vadds(visible, -self.window_size, mask=mask32), sink, mask=mask32
            )
            candidates = vsub(window, sink, mask=mask32)
            selected = vmin(budget, candidates, mask=mask32)
            output_len = vadd(
                vsub(visible, candidates, mask=mask32), selected, mask=mask32
            )
            direct = vselect(
                one, zero, cond_mask=vcmp_eq(selected, candidates, mask=mask32)
            )
            offsets = vmuls(varange(0, dtypes.uint32), ROW_PARAM_STRIDE, mask=mask32)
            vscatter(self.params_ub, visible, offsets, mask=mask32)
            vscatter(
                self.params_ub,
                selected,
                vadds(offsets, 1, mask=mask32),
                mask=mask32,
            )
            vscatter(
                self.params_ub,
                output_len,
                vadds(offsets, 2, mask=mask32),
                mask=mask32,
            )
            vscatter(
                self.params_ub, direct, vadds(offsets, 3, mask=mask32), mask=mask32
            )
            if const_expr(self.precision == 1):
                # p1重建uint32可见范围时分别广播每个字节，保持现有参数布局。
                byte_mask = vdup_scalar(255, dtypes.int32, mask=mask32)
                for byte_index in tuple(range(4)):
                    visible_byte = vand(
                        vshr(visible, byte_index * 8, mask=mask32),
                        byte_mask,
                        mask=mask32,
                    )
                    vscatter(
                        self.params_ub,
                        visible_byte,
                        vadds(
                            offsets, PARAM_VISIBLE_BYTE_BASE + byte_index, mask=mask32
                        ),
                        mask=mask32,
                    )

    @jit
    def p2_scale_bias_stream(
        self,
        key_base,
        score_ready,
        bias_ready,
        compute_blocks,
        kv_col,
    ):
        """完全在寄存器内将N64分数转换为可排序u16键，再写入存储。"""
        # 全局边界使用32位计算，历史全局索引保持uint32。
        with vf(mode="simd"):
            mask32, _ = update_mask(N_CHUNK, elem_bits=32)
            mask16, _ = update_mask(N_CHUNK, elem_bits=16)
            zero16 = vdup_scalar(0, dtypes.uint16, mask=mask16)
            sign_flip = vdup_scalar(0x8000, dtypes.uint16, mask=mask16)
            # 同一VF内的32行共享一个N64偏置向量。
            bias = vload(bias_ready, kv_col)
            # 全局边界使用有符号32位计算，仅将[0,64]内的局部边界收窄为uint16以匹配分数lane。
            key_idx = varange(0, dtypes.uint16)
            compute_local = max(0, min(N_CHUNK, compute_blocks - key_base - kv_col))
            compute16 = vdup_scalar(compute_local, dtypes.uint16, mask=mask16)
            for row in range(M_VECTOR):
                dst = row * STREAM_INPUT + MAX_TOPK + kv_col
                score = vload(score_ready, row * N_CHUNK)
                score = vmuls(score, self.score_scale, mask=mask32)
                score = vadd(score, bias, mask=mask32)
                half = vcast(score, dtypes.bfloat16, mask=mask32)
                # BF16转换将有效数据放入各b32槽的低半部，再在寄存器内打包为64个连续u16值。
                bits = vpack(
                    vreinterpret_lanes(half, dtypes.uint32),
                    dtypes.uint16,
                    part="lower",
                )
                canonical_nan = vcmp_eq_scalar(bits, 0x7FC0, mask=mask16)
                # 按符号位变换，使浮点数值顺序与uint16无符号排序顺序一致。
                sign_bits = vand(bits, sign_flip, mask=mask16)
                negative = vcmp_ne_scalar(sign_bits, 0, mask=mask16)
                positive_key = vxor(bits, sign_flip, mask=mask16)
                negative_key = vnot(bits, mask=mask16)
                key = vselect(negative_key, positive_key, cond_mask=negative)
                key = vselect(zero16, key, cond_mask=canonical_nan)

                # 将全局边界转换为当前N64内的局部边界，再生成有效性掩码。
                visible32 = vload_brc(self.params_ub, row * ROW_PARAM_STRIDE)
                visible_local = vmins(
                    vmaxs(
                        vadds(visible32, -(key_base + kv_col), mask=mask32),
                        0,
                        mask=mask32,
                    ),
                    N_CHUNK,
                    mask=mask32,
                )
                visible16 = vdup_lane0(
                    vreinterpret_lanes(visible_local, dtypes.uint16), mask=mask16
                )
                sink_end32 = vmins(visible32, self.initial_blocks, mask=mask32)
                sink_local = vmins(
                    vmaxs(
                        vadds(sink_end32, -(key_base + kv_col), mask=mask32),
                        0,
                        mask=mask32,
                    ),
                    N_CHUNK,
                    mask=mask32,
                )
                sink_end16 = vdup_lane0(
                    vreinterpret_lanes(sink_local, dtypes.uint16), mask=mask16
                )
                window_start32 = vmaxs(
                    vadds(visible32, -self.window_size, mask=mask32),
                    0,
                    mask=mask32,
                )
                window_start32 = vmax(window_start32, sink_end32, mask=mask32)
                window_local = vmins(
                    vmaxs(
                        vadds(window_start32, -(key_base + kv_col), mask=mask32),
                        0,
                        mask=mask32,
                    ),
                    N_CHUNK,
                    mask=mask32,
                )
                window_start16 = vdup_lane0(
                    vreinterpret_lanes(window_local, dtypes.uint16),
                    mask=mask16,
                )
                valid = vcmp_lt(key_idx, visible16, mask=mask16)
                valid = vcmp_lt(key_idx, compute16, mask=valid)
                valid = vcmp_ge(key_idx, sink_end16, mask=valid)
                valid = vcmp_lt(key_idx, window_start16, mask=valid)
                key = vselect(key, zero16, cond_mask=valid)
                vstore(self.merge_values, dst, key, mask16)

    @jit
    def stream_topk_batch(
        self,
        row_base,
        key_base,
        score_workspace,
        input_offset: Constexpr[int],
        valid_len,
        retain_keys: Constexpr[bool] = True,
    ):
        """按直方图、基数排序和收集阶段拆分四行TopK。"""
        """VF1：统计四行输入value高8位的累计直方图。"""
        with vf(mode="simd"):
            mask8, _ = update_mask(256, elem_bits=8)
            mask16, _ = update_mask(VL_U16, elem_bits=16)

            # 统计四行的高字节直方图。
            for slot in range(RADIX_ROWS):
                row = row_base + slot
                # 一个u16寄存器只有128个lane，count0/count1分别保存桶0～127和128～255的累计计数。
                count0 = vdup_scalar(0, dtypes.uint16, mask=mask16)
                count1 = vdup_scalar(0, dtypes.uint16, mask=mask16)
                # 首轮只扫描当前N256；后续轮依次扫描历史TopK和当前N256，并累加到同一组计数器。
                for chunk in range(
                    input_offset // 256,
                    (input_offset + valid_len) // 256,
                ):
                    # 每次读取256个u16 value并拆出高8位，低8位留到确定目标高字节后再统计。
                    _, high8 = vload_deinterleave_b8(
                        self.merge_values, row * STREAM_INPUT + chunk * 256
                    )
                    # bin=0/1选择完整256桶的前/后128桶；返回值继续作为下一chunk的累计输入。
                    count0 = vhistogram_accumulate(count0, high8, mask=mask8, bin=0)
                    count1 = vhistogram_accumulate(count1, high8, mask=mask8, bin=1)
                hist_base = slot * HISTOGRAM_BINS
                vstore(self.hist, hist_base, count0, mask16)
                vstore(self.hist, hist_base + VL_U16, count1, mask16)

        # 定位累计计数达到bottom-K的首个高字节桶。
        """VF2：定位第K大边界的高字节桶，并计算该桶内的排名next_k。"""
        with vf(mode="simd"):
            mask16, _ = update_mask(VL_U16, elem_bits=16)
            mask32, _ = update_mask(VL_U32, elem_bits=32)
            zero = vdup_scalar(0, dtypes.uint16, mask=mask16)
            one = vdup_scalar(1, dtypes.uint16, mask=mask16)
            topk0 = vload_brc(self.params_ub, (row_base + 0) * ROW_PARAM_STRIDE + 1)
            topk1 = vload_brc(self.params_ub, (row_base + 1) * ROW_PARAM_STRIDE + 1)
            topk2 = vload_brc(self.params_ub, (row_base + 2) * ROW_PARAM_STRIDE + 1)
            topk3 = vload_brc(self.params_ub, (row_base + 3) * ROW_PARAM_STRIDE + 1)
            topk_regs = (topk0, topk1, topk2, topk3)
            for slot in tuple(range(RADIX_ROWS)):
                topk32 = topk_regs[slot]
                # K=0会得到不存在的第valid_len+1小元素，因此radix内部临时按K=1走合法排名；最终输出仍由原始K=0屏蔽。
                rank_topk32 = vmaxs(topk32, 1, mask=mask32)
                # 将“第K大”转换为从小到大的第(valid_len+1-K)个元素。
                bottom32 = vsub(
                    vdup_scalar(valid_len + 1, dtypes.int32, mask=mask32),
                    rank_topk32,
                    mask=mask32,
                )
                bottom16 = vdup_lane0(
                    vreinterpret_lanes(bottom32, dtypes.uint16), mask=mask16
                )
                hist_base = slot * HISTOGRAM_BINS
                idx_high = self.idx_high[slot]
                # 打开变长连续写流，后续两次squeeze结果按实际长度首尾追加到idx_high。
                vstore_unalign_begin(idx_high)
                for chunk in tuple(range(2)):
                    # u16向量有128个lane：两轮分别生成桶号0～127和128～255。
                    indices = varange(chunk * VL_U16, dtypes.uint16)
                    counts = vload(self.hist, hist_base + chunk * VL_U16)
                    # 累计计数首次达到bottom的桶即目标value的高8位；压缩结果第0项就是该桶号。
                    reached = vcmp_ge(counts, bottom16, mask=mask16)
                    vstore_unalign(
                        idx_high,
                        0,
                        vsqueeze_store_reg(indices, mask=reached),
                    )
                vstore_unalign_post(idx_high, 0)
            vmem_bar("vst_vld")

            # 将全局秩转换为目标高字节桶内的秩next_k，供下一VF定位低字节桶。
            for slot in tuple(range(RADIX_ROWS)):
                topk32 = topk_regs[slot]
                rank_topk32 = vmaxs(topk32, 1, mask=mask32)
                bottom32 = vsub(
                    vdup_scalar(valid_len + 1, dtypes.int32, mask=mask32),
                    rank_topk32,
                    mask=mask32,
                )
                bottom16 = vdup_lane0(
                    vreinterpret_lanes(bottom32, dtypes.uint16), mask=mask16
                )
                hist_base = slot * HISTOGRAM_BINS
                # 标量等价：high8 = idx_high[slot][0]。
                high8 = vload_brc(self.idx_high[slot], 0)
                # 标量等价：high8_zero = (high8 == 0)，用于保护high8 - 1不发生u16下溢。
                high8_zero = vcmp_eq_scalar(high8, 0, mask=mask16)
                # 标量等价：previous_bucket = 0 if high8_zero else high8 - 1。
                previous = vselect(
                    zero,
                    vsub(high8, one, mask=mask16),
                    cond_mask=high8_zero,
                )
                # 标量等价：previous_abs = hist_base + previous_bucket，转换为hist一维Buffer的绝对下标。
                previous = vadd(
                    previous,
                    vdup_scalar(hist_base, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                # 标量等价：previous_count = hist[previous_abs]。
                previous_count = vgather(self.hist, previous, mask=mask16)
                # 标量等价：if high8_zero: previous_count = 0；桶0之前没有更小的高字节桶。
                previous_count = vselect(zero, previous_count, cond_mask=high8_zero)
                # 扣掉所有更小高字节的数量，得到目标元素在当前高字节桶内部的排名。
                vstore(
                    self.next_k,
                    slot * VL_U16,
                    vsub(bottom16, previous_count, mask=mask16),
                    mask16,
                )
        """VF3：筛选目标高字节桶，统计其中低8位的累计直方图。"""
        with vf(mode="simd"):
            mask8, _ = update_mask(256, elem_bits=8)
            mask16, _ = update_mask(VL_U16, elem_bits=16)
            for slot in tuple(range(RADIX_ROWS)):
                row = row_base + slot
                target16 = vload_brc(self.idx_high[slot], 0)
                target8 = vdup_lane0(
                    vreinterpret_lanes(target16, dtypes.uint8), mask=mask8
                )
                count0 = vdup_scalar(0, dtypes.uint16, mask=mask16)
                count1 = vdup_scalar(0, dtypes.uint16, mask=mask16)
                for chunk in range(
                    input_offset // 256,
                    (input_offset + valid_len) // 256,
                ):
                    low8, high8 = vload_deinterleave_b8(
                        self.merge_values, row * STREAM_INPUT + chunk * 256
                    )
                    matching = vcmp_eq(high8, target8, mask=mask8)
                    count0 = vhistogram_accumulate(count0, low8, mask=matching, bin=0)
                    count1 = vhistogram_accumulate(count1, low8, mask=matching, bin=1)
                hist_base = slot * HISTOGRAM_BINS
                vstore(self.hist, hist_base, count0, mask16)
                vstore(self.hist, hist_base + VL_U16, count1, mask16)

            # 定位低字节桶，生成完整的u16第K个排序键。
        """VF4：根据next_k定位边界value的低字节桶。"""
        with vf(mode="simd"):
            mask16, _ = update_mask(VL_U16, elem_bits=16)
            for slot in tuple(range(RADIX_ROWS)):
                hist_base = slot * HISTOGRAM_BINS
                next_k = vload_brc(self.next_k, slot * VL_U16)
                idx_low = self.idx_low[slot]
                # 与高字节阶段相同，连续压缩首个累计计数达到next_k的低字节桶。
                vstore_unalign_begin(idx_low)
                for chunk in tuple(range(2)):
                    indices = varange(chunk * VL_U16, dtypes.uint16)
                    counts = vload(self.hist, hist_base + chunk * VL_U16)
                    reached = vcmp_ge(counts, next_k, mask=mask16)
                    vstore_unalign(
                        idx_low,
                        0,
                        vsqueeze_store_reg(indices, mask=reached),
                    )
                vstore_unalign_post(idx_low, 0)
            # 对四行执行稳定压缩，先处理GT，再处理EQ。
        """VF5：按完整边界value压缩GT和EQ候选的局部索引。"""
        with vf(mode="simd"):
            mask16, _ = update_mask(VL_U16, elem_bits=16)
            for slot in tuple(range(RADIX_ROWS)):
                row = row_base + slot
                positions_out = self.tmp_indices[slot]
                # 直接使用第K个排序键。
                high8 = vload_brc(self.idx_high[slot], 0)
                low8 = vload_brc(self.idx_low[slot], 0)
                target = vadd(vshl(high8, 8, mask=mask16), low8, mask=mask16)
                vstore_unalign_begin(positions_out)
                for chunk in range(
                    input_offset // VL_U16,
                    (input_offset + valid_len) // VL_U16,
                ):
                    values = vload(
                        self.merge_values,
                        row * STREAM_INPUT + chunk * VL_U16,
                    )
                    positions = varange(chunk * VL_U16, dtypes.uint16)
                    selected = vcmp_gt(values, target, mask=mask16)
                    vstore_unalign(
                        positions_out,
                        0,
                        vsqueeze_store_reg(positions, mask=selected),
                    )
                for chunk in range(
                    input_offset // VL_U16,
                    (input_offset + valid_len) // VL_U16,
                ):
                    values = vload(
                        self.merge_values,
                        row * STREAM_INPUT + chunk * VL_U16,
                    )
                    positions = varange(chunk * VL_U16, dtypes.uint16)
                    selected = vcmp_eq(values, target, mask=mask16)
                    vstore_unalign(
                        positions_out,
                        0,
                        vsqueeze_store_reg(positions, mask=selected),
                    )
                vstore_unalign_post(positions_out, 0)

            # 通过局部u16位置收集分数键，全局索引运算单独使用b32完成。
        """VF6：根据tmp_indices收集本轮TopK values，供后续N256复用。"""
        with vf(mode="simd"):
            mask16, _ = update_mask(VL_U16, elem_bits=16)
            mask32, _ = update_mask(VL_U32, elem_bits=32)
            zero = vdup_scalar(0, dtypes.uint16, mask=mask16)
            topk0 = vload_brc(self.params_ub, (row_base + 0) * ROW_PARAM_STRIDE + 1)
            topk1 = vload_brc(self.params_ub, (row_base + 1) * ROW_PARAM_STRIDE + 1)
            topk2 = vload_brc(self.params_ub, (row_base + 2) * ROW_PARAM_STRIDE + 1)
            topk3 = vload_brc(self.params_ub, (row_base + 3) * ROW_PARAM_STRIDE + 1)
            topk_regs = (topk0, topk1, topk2, topk3)
            if const_expr(retain_keys):
                for slot in tuple(range(RADIX_ROWS)):
                    row = row_base + slot
                    topk16 = vdup_lane0(
                        vreinterpret_lanes(topk_regs[slot], dtypes.uint16), mask=mask16
                    )
                    key_row_offset = vdup_scalar(
                        row * STREAM_INPUT, dtypes.uint16, mask=mask16
                    )
                    for chunk in range(MAX_TOPK // VL_U16):
                        lane = varange(chunk * VL_U16, dtypes.uint16)
                        active = vcmp_lt(lane, topk16, mask=mask16)
                        position = vload(self.tmp_indices[slot], chunk * VL_U16)
                        position = vselect(position, zero, cond_mask=active)
                        absolute = vadd(position, key_row_offset, mask=mask16)
                        key = vgather(self.merge_values, absolute, mask=mask16)
                        vstore(
                            score_workspace,
                            (WS_OUT_KEYS + slot * MAX_TOPK + chunk * VL_U16) // 2,
                            vreinterpret_lanes(
                                vselect(key, zero, cond_mask=active), dtypes.float32
                            ),
                            mask32,
                        )

        """VF7：将局部位置映射为历史或当前N256的S2全局索引。"""
        with vf(mode="simd"):
            mask32, _ = update_mask(VL_U32, elem_bits=32)
            topk0 = vload_brc(self.params_ub, (row_base + 0) * ROW_PARAM_STRIDE + 1)
            topk1 = vload_brc(self.params_ub, (row_base + 1) * ROW_PARAM_STRIDE + 1)
            topk2 = vload_brc(self.params_ub, (row_base + 2) * ROW_PARAM_STRIDE + 1)
            topk3 = vload_brc(self.params_ub, (row_base + 3) * ROW_PARAM_STRIDE + 1)
            topk_regs = (topk0, topk1, topk2, topk3)
            zero32 = vdup_scalar(0, dtypes.uint32, mask=mask32)
            merge_split32 = vdup_scalar(MAX_TOPK, dtypes.uint32, mask=mask32)
            key_base32 = vdup_scalar(key_base, dtypes.uint32, mask=mask32)
            for slot in tuple(range(RADIX_ROWS)):
                row = row_base + slot
                topk32 = vreinterpret(topk_regs[slot], dtypes.uint32)
                history_row_base = vdup_scalar(
                    row * MAX_TOPK, dtypes.uint32, mask=mask32
                )
                for chunk in range(MAX_TOPK // VL_U32):
                    output_rank = varange(chunk * VL_U32, dtypes.uint32)
                    active_output = vcmp_lt(output_rank, topk32, mask=mask32)
                    merge_position = vreinterpret(
                        vload_unpack(
                            self.tmp_indices[slot],
                            chunk * VL_U32,
                            mode="b16_to_b32",
                        ),
                        dtypes.uint32,
                    )
                    merge_position = vselect(
                        merge_position, zero32, cond_mask=active_output
                    )
                    # merge_values布局：[0, MAX_TOPK)为历史TopK，
                    # [MAX_TOPK, 2 * MAX_TOPK)为当前N256输入。
                    # 当前区索引：将合并区位置减去历史区容量，再加本轮K块起点。
                    current_global_index = vsub(
                        vadd(merge_position, key_base32, mask=mask32),
                        merge_split32,
                        mask=mask32,
                    )
                    if const_expr(input_offset == MAX_TOPK):
                        # 第一轮没有历史TopK，选中结果全部来自当前区。
                        global_index = current_global_index
                    else:
                        from_history = vcmp_lt(
                            merge_position, merge_split32, mask=active_output
                        )
                        history_position = vselect(
                            merge_position, zero32, cond_mask=from_history
                        )
                        # 将[row, history_position]展平为global_indices UB中的一维元素地址。
                        history_address = vadd(
                            history_position, history_row_base, mask=mask32
                        )
                        # 各lane按离散地址读取历史S2全局索引；保持lane对应关系，不做压缩。
                        history_global_index = vgather(
                            self.global_indices, history_address, mask=mask32
                        )
                        # 索引选择：历史区位置读取已保留的全局索引，
                        # 当前区位置使用本轮K块起点换算出的全局索引。
                        global_index = vselect(
                            history_global_index,
                            current_global_index,
                            cond_mask=from_history,
                        )
                    vstore(
                        self.stream_out_indices,
                        slot * MAX_TOPK + chunk * VL_U32,
                        vselect(global_index, zero32, cond_mask=active_output),
                        mask32,
                    )
        """VF8：提交本轮TopK values和global indices到流式历史区。"""
        with vf(mode="simd"):
            mask16, _ = update_mask(VL_U16, elem_bits=16)
            mask32, _ = update_mask(VL_U32, elem_bits=32)
            for slot in range(RADIX_ROWS):
                row = row_base + slot
                if const_expr(retain_keys):
                    for chunk in range(MAX_TOPK // VL_U16):
                        vstore(
                            self.merge_values,
                            row * STREAM_INPUT + chunk * VL_U16,
                            vreinterpret_lanes(
                                vload(
                                    score_workspace,
                                    (WS_OUT_KEYS + slot * MAX_TOPK + chunk * VL_U16)
                                    // 2,
                                ),
                                dtypes.uint16,
                            ),
                            mask16,
                        )
                for chunk in range(MAX_TOPK // VL_U32):
                    vstore(
                        self.global_indices,
                        row * MAX_TOPK + chunk * VL_U32,
                        vload(
                            self.stream_out_indices, slot * MAX_TOPK + chunk * VL_U32
                        ),
                        mask32,
                    )

    @jit
    def p1_scale_bias_stream(self, key_base, score_ready, bias_ready, kv_col):
        """消费N64分数Channel，将u32键写入TopK工作区。"""
        with vf(mode="simd"):
            mask32, _ = update_mask(N_CHUNK, elem_bits=32)
            for row in range(M_VECTOR):
                key = self._p1_sortable_key(
                    score_ready,
                    bias_ready,
                    row,
                    key_base + kv_col,
                    kv_col,
                    mask32,
                )
                vstore(
                    self.score_ub,
                    row * N_OUTER + kv_col,
                    vreinterpret(key, dtypes.float32),
                    mask32,
                )

    @jit
    def p1_stream_topk_batch(
        self,
        row_base,
        key_base,
        score_workspace,
        input_offset: Constexpr[int],
        valid_len: Constexpr[int],
    ):
        for pass_index in tuple(range(4)):
            self._p1_stream_radix_pass(
                row_base,
                score_workspace,
                input_offset,
                valid_len,
                pass_index,
            )
        self._p1_stream_select_gather(
            row_base,
            key_base,
            score_workspace,
            input_offset,
            valid_len,
        )

    @jit
    def keys_ready(self):
        with vf(mode="simd"):
            vmem_bar("vst_vld")

    @jit
    def zero_keys_chunk(self, column):
        with vf(mode="simd"):
            if const_expr(self.precision == 2):
                mask16, _ = update_mask(N_CHUNK, elem_bits=16)
                zero = vdup_scalar(0, dtypes.uint16, mask=mask16)
                for row in range(M_VECTOR):
                    # 当前N64整体超出compute_blocks时，清零其64个value，避免TopK读到UB历史数据。
                    vstore(
                        self.merge_values,
                        row * STREAM_INPUT + MAX_TOPK + column,
                        zero,
                        mask16,
                    )
            else:
                mask32, _ = update_mask(N_CHUNK, elem_bits=32)
                zero = vdup_scalar(0.0, dtypes.float32, mask=mask32)
                for row in range(M_VECTOR):
                    vstore(self.score_ub, row * N_OUTER + column, zero, mask32)

    @jit
    def store_task(
        self, gm_indices, gm_lens, batch, q_head, q_block_start, q_block_count
    ):
        q_block0 = q_block_start + self.subblock * M_VECTOR
        valid_rows = max(0, min(q_block_count - self.subblock * M_VECTOR, M_VECTOR))
        if valid_rows == M_VECTOR:
            with vf(mode="simd"):
                for row in range(M_VECTOR):
                    self._pack_one(row, row)
            mem_copy(
                self._indices_prefix_tile(gm_indices, batch, q_head, q_block0),
                self.out_rows,
            )
            mem_copy(
                self._lens_prefix_tile(gm_lens, batch, q_head, q_block0),
                self.output_lens,
            )
        else:
            # 每个尾块迭代生成并消费一次完整输出事务，不写入跨头片段中的无效行。
            for row in range(valid_rows):
                with vf(mode="simd"):
                    self._pack_one(row, 0)
                mem_copy(
                    self._indices_prefix_tile(
                        gm_indices, batch, q_head, q_block0 + row, 1
                    ),
                    tile_slice(self.out_rows, (1, self.output_capacity), (0, 0)),
                )
                mem_copy(
                    self._lens_prefix_tile(gm_lens, batch, q_head, q_block0 + row, 1),
                    tile_slice(self.output_lens, (1, 1), (0, 0)),
                )

    def load_bias(self, gm_bias, batch, kv_head, key_base):
        mem_copy(self.bias_ub, self._bias_tile(gm_bias, batch, kv_head, key_base))

    def _indices_prefix_tile(self, gm_indices, batch, q_head, q_block0, rows=M_VECTOR):
        base = gm_indices[batch, q_head, q_block0:, None]
        return tile_slice(base, (rows, self.output_capacity), (0, 0))

    @jit
    def _pack_one(self, row, output_slot):
        """将一行索引按直出或[Sink, TopK, Window]格式打包到输出Channel。"""
        mask32, _ = update_mask(VL_U32, elem_bits=32)
        minus_one = vdup_scalar(-1, dtypes.int32, mask=mask32)
        output_base = output_slot * self.output_capacity
        # 先将当前输出槽全部填为-1；后续只顺序覆盖output_len个有效位置。
        for chunk in range(self.output_capacity // VL_U32):
            vstore(
                self.out_rows,
                output_base + chunk * VL_U32,
                minus_one,
                mask32,
            )
        # 每行参数依次为：可见K块数、普通TopK数量、最终输出长度、是否全部直出。
        visible = vload_brc(self.params_ub, row * ROW_PARAM_STRIDE)
        topk = vload_brc(self.params_ub, row * ROW_PARAM_STRIDE + 1)
        output_len = vload_brc(self.params_ub, row * ROW_PARAM_STRIDE + 2)
        direct = vload_brc(self.params_ub, row * ROW_PARAM_STRIDE + 3)
        # direct_row和sparse_row互斥，分别控制下面两条打包路径。
        direct_row = vcmp_ne_scalar(direct, 0, mask=mask32)
        sparse_row = vcmp_eq_scalar(direct, 0, mask=mask32)
        # 稀疏路径固定保留开头Sink和末尾Window；Window起点不早于Sink末端。
        sink_end = vmins(visible, self.initial_blocks, mask=mask32)
        window_start = vmaxs(
            vadds(visible, -self.window_size, mask=mask32), 0, mask=mask32
        )
        window_start = vmax(window_start, sink_end, mask=mask32)
        window_count = vsub(visible, window_start, mask=mask32)
        # unalign store维护一个顺序写游标；每次squeeze后的有效项会紧接前一段追加。
        vstore_unalign_begin(self.out_rows)

        # 直出路径：无需TopK，输出全部可见索引[0, visible)。
        for chunk in range(self.output_capacity // VL_U32):
            positions = varange(chunk * VL_U32, dtypes.int32)
            valid_direct = vcmp_lt(positions, visible, mask=direct_row)
            squeezed = vsqueeze_store_reg(positions, mask=valid_direct)
            vstore_unalign(self.out_rows, output_base, squeezed)

        # 稀疏路径第1段：追加Sink索引[0, sink_end)。
        sink_values = varange(0, dtypes.int32)
        sink_mask = vcmp_lt(sink_values, sink_end, mask=sparse_row)
        squeezed_sink = vsqueeze_store_reg(sink_values, mask=sink_mask)
        vstore_unalign(self.out_rows, output_base, squeezed_sink)

        # 稀疏路径第2段：追加流式TopK保存的前topk个S2全局索引。
        for chunk in range(MAX_TOPK // VL_U32):
            topk_positions = varange(chunk * VL_U32, dtypes.int32)
            # precision为编译期Kernel配置。使用静态分派，避免选定向量跨出前端动态if区域。
            if const_expr(self.precision == 2):
                selected = vreinterpret(
                    vload(self.global_indices, row * MAX_TOPK + chunk * VL_U32),
                    dtypes.int32,
                )
            else:
                selected = vreinterpret(
                    vload(
                        self.p1_global_indices,
                        row * MAX_TOPK + chunk * VL_U32,
                    ),
                    dtypes.int32,
                )
            # 流式TopK的紧凑历史保存在槽0，将这些项紧接在sink前缀之后输出。
            in_topk = vcmp_lt(topk_positions, topk, mask=sparse_row)
            squeezed = vsqueeze_store_reg(selected, mask=in_topk)
            vstore_unalign(self.out_rows, output_base, squeezed)

        # 稀疏路径第3段：追加Window索引[window_start, visible)。
        window_lane = varange(0, dtypes.int32)
        in_window = vcmp_lt(window_lane, window_count, mask=sparse_row)
        window_values = vadd(window_lane, window_start, mask=mask32)
        squeezed_window = vsqueeze_store_reg(window_values, mask=in_window)
        vstore_unalign(self.out_rows, output_base, squeezed_window)
        # 完成本行顺序写，并记录有效前缀长度；其余槽位保持-1。
        # 等价：direct ? range(visible) : sink + selected_topk + window。
        vstore_unalign_post(self.out_rows, output_base)
        vstore_first(self.output_lens, output_slot, output_len)

    @jit
    def _p1_accumulate_radix_histogram(
        self,
        source,
        source_offset,
        row,
        count0,
        count1,
        mask8,
        pass_index: Constexpr[int],
    ):
        """在调用方VF内复用字节筛选和直方图累计，不新增VF边界。"""
        word_hi0, word_lo0 = vload_deinterleave(source, source_offset, width="b16")
        word_hi1, word_lo1 = vload_deinterleave(
            source, source_offset + VL_U16, width="b16"
        )
        byte2, byte3 = vdeintlv(
            vreinterpret_lanes(word_lo0, dtypes.uint8),
            vreinterpret_lanes(word_lo1, dtypes.uint8),
        )
        byte0, byte1 = vdeintlv(
            vreinterpret_lanes(word_hi0, dtypes.uint8),
            vreinterpret_lanes(word_hi1, dtypes.uint8),
        )
        prefix8 = mask8
        byte = byte3
        if const_expr(pass_index >= 1):
            target0 = vreinterpret_lanes(
                vload_brc_b8(self.p1_radix_state, row * P1_RADIX_STATE_STRIDE + 1),
                dtypes.uint8,
            )
            prefix8 = vcmp_eq(byte3, target0, mask=mask8)
            byte = byte2
        if const_expr(pass_index >= 2):
            target1 = vreinterpret_lanes(
                vload_brc_b8(self.p1_radix_state, row * P1_RADIX_STATE_STRIDE + 2),
                dtypes.uint8,
            )
            prefix8 = vcmp_eq(byte2, target1, mask=prefix8)
            byte = byte1
        if const_expr(pass_index >= 3):
            target2 = vreinterpret_lanes(
                vload_brc_b8(self.p1_radix_state, row * P1_RADIX_STATE_STRIDE + 3),
                dtypes.uint8,
            )
            prefix8 = vcmp_eq(byte1, target2, mask=prefix8)
            byte = byte0
        count0 = vhistogram_accumulate(count0, byte, mask=prefix8, bin=0)
        count1 = vhistogram_accumulate(count1, byte, mask=prefix8, bin=1)
        return count0, count1

    @jit
    def _p1_stream_radix_pass(
        self,
        row_base,
        score_workspace,
        input_offset: Constexpr[int],
        valid_len: Constexpr[int],
        pass_index: Constexpr[int],
    ):
        """对历史256项与当前256项执行一次高位优先的基数排序。"""
        with vf(mode="simd"):
            mask8, _ = update_mask(256, elem_bits=8)
            mask16, _ = update_mask(VL_U16, elem_bits=16)
            mask32, _ = update_mask(VL_U32, elem_bits=32)
            zero16 = vdup_scalar(0, dtypes.uint16, mask=mask16)
            one16 = vdup_scalar(1, dtypes.uint16, mask=mask16)
            byte_mask32 = vdup_scalar(0xFF, dtypes.uint32, mask=mask32)
            u16_mask32 = vdup_scalar(0xFFFF, dtypes.uint32, mask=mask32)

            for slot in range(RADIX_ROWS):
                row = row_base + slot
                count0 = vdup_scalar(0, dtypes.uint16, mask=mask16)
                count1 = vdup_scalar(0, dtypes.uint16, mask=mask16)

                # 后续组先扫描历史256项；首组仅扫描当前项，保持原来的累计顺序。
                if const_expr(input_offset == 0):
                    count0, count1 = self._p1_accumulate_radix_histogram(
                        self.p1_merge_values,
                        row * MAX_TOPK,
                        row,
                        count0,
                        count1,
                        mask8,
                        pass_index,
                    )
                count0, count1 = self._p1_accumulate_radix_histogram(
                    score_workspace,
                    row * N_OUTER,
                    row,
                    count0,
                    count1,
                    mask8,
                    pass_index,
                )
                hist_base = slot * HISTOGRAM_BINS
                vstore(self.p1_hist, hist_base, count0, mask16)
                vstore(self.p1_hist, hist_base + VL_U16, count1, mask16)
            vmem_bar("vst_vld")

            for slot in tuple(range(RADIX_ROWS)):
                row = row_base + slot
                if const_expr(pass_index == 0):
                    topk32 = vload_brc(self.params_ub, row * ROW_PARAM_STRIDE + 1)
                    # K为0时使用安全的内部秩，不改变params_ub及筛选、收集、输出的元素数量。
                    rank_topk32 = vmaxs(topk32, 1, mask=mask32)
                    bottom32 = vsub(
                        vdup_scalar(valid_len + 1, dtypes.int32, mask=mask32),
                        rank_topk32,
                        mask=mask32,
                    )
                    bottom16 = vdup_lane0(
                        vreinterpret_lanes(bottom32, dtypes.uint16),
                        mask=mask16,
                    )
                else:
                    next32 = vload_brc(
                        self.p1_radix_state, row * P1_RADIX_STATE_STRIDE + 5
                    )
                    bottom16 = vdup_lane0(
                        vreinterpret_lanes(next32, dtypes.uint16),
                        mask=mask16,
                    )
                bucket = self.p1_idx_bucket[slot]
                hist_base = slot * HISTOGRAM_BINS
                vstore_unalign_begin(bucket)
                for chunk in tuple(range(2)):
                    indices = varange(chunk * VL_U16, dtypes.uint16)
                    counts = vload(self.p1_hist, hist_base + chunk * VL_U16)
                    reached = vcmp_ge(counts, bottom16, mask=mask16)
                    vstore_unalign(
                        bucket,
                        0,
                        vsqueeze_store_reg(indices, mask=reached),
                    )
                vstore_unalign_post(bucket, 0)
            vmem_bar("vst_vld")

            for slot in tuple(range(RADIX_ROWS)):
                row = row_base + slot
                if const_expr(pass_index == 0):
                    topk32 = vload_brc(self.params_ub, row * ROW_PARAM_STRIDE + 1)
                    rank_topk32 = vmaxs(topk32, 1, mask=mask32)
                    bottom32 = vsub(
                        vdup_scalar(valid_len + 1, dtypes.int32, mask=mask32),
                        rank_topk32,
                        mask=mask32,
                    )
                    bottom16 = vdup_lane0(
                        vreinterpret_lanes(bottom32, dtypes.uint16),
                        mask=mask16,
                    )
                else:
                    next32 = vload_brc(
                        self.p1_radix_state, row * P1_RADIX_STATE_STRIDE + 5
                    )
                    bottom16 = vdup_lane0(
                        vreinterpret_lanes(next32, dtypes.uint16),
                        mask=mask16,
                    )
                bucket16 = vload_brc(self.p1_idx_bucket[slot], 0)
                bucket32 = vand(
                    vreinterpret_lanes(bucket16, dtypes.uint32),
                    byte_mask32,
                    mask=mask32,
                )
                vstore_first(
                    self.p1_radix_state,
                    row * P1_RADIX_STATE_STRIDE + 1 + pass_index,
                    vreinterpret(bucket32, dtypes.int32),
                )

                bucket_zero = vcmp_eq_scalar(bucket16, 0, mask=mask16)
                previous = vselect(
                    zero16,
                    vsub(bucket16, one16, mask=mask16),
                    cond_mask=bucket_zero,
                )
                previous = vadd(
                    previous,
                    vdup_scalar(
                        slot * HISTOGRAM_BINS,
                        dtypes.uint16,
                        mask=mask16,
                    ),
                    mask=mask16,
                )
                previous_count = vgather(self.p1_hist, previous, mask=mask16)
                previous_count = vselect(zero16, previous_count, cond_mask=bucket_zero)
                next16 = vsub(bottom16, previous_count, mask=mask16)
                next32 = vand(
                    vreinterpret_lanes(next16, dtypes.uint32),
                    u16_mask32,
                    mask=mask32,
                )
                vstore_first(
                    self.p1_radix_state,
                    row * P1_RADIX_STATE_STRIDE + 5,
                    vreinterpret(next32, dtypes.int32),
                )

    @jit
    def _p1_stream_select_gather(
        self,
        row_base,
        key_base,
        score_workspace,
        input_offset: Constexpr[int],
        valid_len: Constexpr[int],
    ):
        """压缩四行并更新其保留的u32键和索引状态。"""
        with vf(mode="simd"):
            mask32, _ = update_mask(VL_U32, elem_bits=32)
            zero_i32 = vdup_scalar(0, dtypes.int32, mask=mask32)
            zero_u32 = vdup_scalar(0, dtypes.uint32, mask=mask32)
            byte_mask = vdup_scalar(0xFF, dtypes.uint32, mask=mask32)

            for slot in tuple(range(RADIX_ROWS)):
                row = row_base + slot
                b0 = vand(
                    vload_brc_b8(self.p1_radix_state, row * P1_RADIX_STATE_STRIDE + 1),
                    byte_mask,
                    mask=mask32,
                )
                b1 = vand(
                    vload_brc_b8(self.p1_radix_state, row * P1_RADIX_STATE_STRIDE + 2),
                    byte_mask,
                    mask=mask32,
                )
                b2 = vand(
                    vload_brc_b8(self.p1_radix_state, row * P1_RADIX_STATE_STRIDE + 3),
                    byte_mask,
                    mask=mask32,
                )
                b3 = vand(
                    vload_brc_b8(self.p1_radix_state, row * P1_RADIX_STATE_STRIDE + 4),
                    byte_mask,
                    mask=mask32,
                )
                kth = vadd(
                    vshl(b0, 24, mask=mask32),
                    vshl(b1, 16, mask=mask32),
                    mask=mask32,
                )
                kth = vadd(kth, vshl(b2, 8, mask=mask32), mask=mask32)
                kth = vadd(kth, b3, mask=mask32)

                positions_out = self.p1_tmp_indices[slot]
                vstore_unalign_begin(positions_out)
                if const_expr(input_offset == 0):
                    for chunk in tuple(range(MAX_TOPK // VL_U32)):
                        values = vload(
                            self.p1_merge_values,
                            row * MAX_TOPK + chunk * VL_U32,
                        )
                        positions = varange(chunk * VL_U32, dtypes.int32)
                        selected = vcmp_gt(values, kth, mask=mask32)
                        vstore_unalign(
                            positions_out,
                            0,
                            vsqueeze_store_reg(positions, mask=selected),
                        )
                for chunk in tuple(
                    range(
                        max(input_offset, MAX_TOPK) // VL_U32,
                        (input_offset + valid_len) // VL_U32,
                    )
                ):
                    values = vreinterpret(
                        vload(
                            score_workspace,
                            row * N_OUTER + (chunk - MAX_TOPK // VL_U32) * VL_U32,
                        ),
                        dtypes.uint32,
                    )
                    positions = varange(chunk * VL_U32, dtypes.int32)
                    selected = vcmp_gt(values, kth, mask=mask32)
                    vstore_unalign(
                        positions_out,
                        0,
                        vsqueeze_store_reg(positions, mask=selected),
                    )
                if const_expr(input_offset == 0):
                    for chunk in tuple(range(MAX_TOPK // VL_U32)):
                        values = vload(
                            self.p1_merge_values,
                            row * MAX_TOPK + chunk * VL_U32,
                        )
                        positions = varange(chunk * VL_U32, dtypes.int32)
                        selected = vcmp_eq(values, kth, mask=mask32)
                        vstore_unalign(
                            positions_out,
                            0,
                            vsqueeze_store_reg(positions, mask=selected),
                        )
                for chunk in tuple(
                    range(
                        max(input_offset, MAX_TOPK) // VL_U32,
                        (input_offset + valid_len) // VL_U32,
                    )
                ):
                    values = vreinterpret(
                        vload(
                            score_workspace,
                            row * N_OUTER + (chunk - MAX_TOPK // VL_U32) * VL_U32,
                        ),
                        dtypes.uint32,
                    )
                    positions = varange(chunk * VL_U32, dtypes.int32)
                    selected = vcmp_eq(values, kth, mask=mask32)
                    vstore_unalign(
                        positions_out,
                        0,
                        vsqueeze_store_reg(positions, mask=selected),
                    )
                vstore_unalign_post(positions_out, 0)
            vmem_bar("vst_vld")

            for slot in tuple(range(RADIX_ROWS)):
                row = row_base + slot
                topk = vload_brc(self.params_ub, row * ROW_PARAM_STRIDE + 1)
                retained_row = vdup_scalar(row * MAX_TOPK, dtypes.int32, mask=mask32)
                current_row = vdup_scalar(row * N_OUTER, dtypes.int32, mask=mask32)
                for chunk in tuple(range(MAX_TOPK // VL_U32)):
                    lane = varange(chunk * VL_U32, dtypes.int32)
                    active = vcmp_lt(lane, topk, mask=mask32)
                    position = vload(self.p1_tmp_indices[slot], chunk * VL_U32)
                    position = vselect(position, zero_i32, cond_mask=active)
                    from_retained = vcmp_lt_scalar(position, MAX_TOPK, mask=active)
                    current = vcmp_ge_scalar(position, MAX_TOPK, mask=active)

                    # 首组没有历史保留结果，无需读取历史数据。
                    if const_expr(input_offset == MAX_TOPK):
                        previous_key = zero_u32
                        previous_index = zero_u32
                    else:
                        previous_pos = vselect(
                            position, zero_i32, cond_mask=from_retained
                        )
                        previous_abs = vadd(previous_pos, retained_row, mask=mask32)
                        previous_abs_u32 = vreinterpret(previous_abs, dtypes.uint32)
                        previous_key = vgather(
                            self.p1_merge_values, previous_abs_u32, mask=mask32
                        )
                        previous_index = vgather(
                            self.p1_global_indices, previous_abs_u32, mask=mask32
                        )

                    current_pos = vadds(position, -MAX_TOPK, mask=mask32)
                    current_pos = vselect(current_pos, zero_i32, cond_mask=current)
                    current_abs = vadd(current_pos, current_row, mask=mask32)
                    current_abs_u32 = vreinterpret(current_abs, dtypes.uint32)
                    current_key = vreinterpret(
                        vgather(score_workspace, current_abs_u32, mask=mask32),
                        dtypes.uint32,
                    )
                    current_index = vreinterpret(
                        vadds(current_pos, key_base, mask=mask32),
                        dtypes.uint32,
                    )

                    key = vselect(
                        previous_key,
                        current_key,
                        cond_mask=from_retained,
                    )
                    index = vselect(
                        previous_index,
                        current_index,
                        cond_mask=from_retained,
                    )
                    vstore(
                        self.p1_out_keys,
                        slot * MAX_TOPK + chunk * VL_U32,
                        vselect(key, zero_u32, cond_mask=active),
                        mask32,
                    )
                    vstore(
                        self.p1_out_indices,
                        slot * MAX_TOPK + chunk * VL_U32,
                        vselect(index, zero_u32, cond_mask=active),
                        mask32,
                    )
            vmem_bar("vst_vld")

            for slot in range(RADIX_ROWS):
                row = row_base + slot
                for chunk in tuple(range(MAX_TOPK // VL_U32)):
                    vstore(
                        self.p1_merge_values,
                        row * MAX_TOPK + chunk * VL_U32,
                        vload(
                            self.p1_out_keys,
                            slot * MAX_TOPK + chunk * VL_U32,
                        ),
                        mask32,
                    )
                    vstore(
                        self.p1_global_indices,
                        row * MAX_TOPK + chunk * VL_U32,
                        vload(
                            self.p1_out_indices,
                            slot * MAX_TOPK + chunk * VL_U32,
                        ),
                        mask32,
                    )

    @jit
    def _p1_param_u32(self, row, byte_base, mask32):
        byte_mask = vdup_scalar(0xFF, dtypes.uint32, mask=mask32)
        value = vand(
            vload_brc_b8(self.params_ub, row * ROW_PARAM_STRIDE + byte_base),
            byte_mask,
            mask=mask32,
        )
        for byte_index in tuple(range(1, 4)):
            byte = vand(
                vload_brc_b8(
                    self.params_ub,
                    row * ROW_PARAM_STRIDE + byte_base + byte_index,
                ),
                byte_mask,
                mask=mask32,
            )
            value = vadd(
                value,
                vshl(byte, byte_index * 8, mask=mask32),
                mask=mask32,
            )
        return vreinterpret(value, dtypes.int32)

    @jit
    def _p1_sortable_key(self, score_ready, bias_ready, row, key_base, kv_col, mask32):
        score = vmuls(vload(score_ready, row * N_CHUNK), self.score_scale, mask=mask32)
        score = vadd(score, vload(bias_ready, kv_col), mask=mask32)
        bits = vreinterpret(score, dtypes.uint32)
        sign_mask = vdup_scalar(0x80000000, dtypes.uint32, mask=mask32)
        sign = vand(bits, sign_mask, mask=mask32)
        negative = vcmp_ne_scalar(sign, 0, mask=mask32)
        positive_key = vxor(bits, sign_mask, mask=mask32)
        negative_key = vnot(bits, mask=mask32)
        key = vselect(negative_key, positive_key, cond_mask=negative)
        zero = vdup_scalar(0, dtypes.uint32, mask=mask32)
        canonical_nan = vcmp_eq_scalar(bits, 0x7FC00000, mask=mask32)
        key = vselect(zero, key, cond_mask=canonical_nan)

        key_idx = varange(key_base, dtypes.int32)
        visible = self._p1_param_u32(row, PARAM_VISIBLE_BYTE_BASE, mask32)
        sink_end = vmins(visible, self.initial_blocks, mask=mask32)
        window_start = vmaxs(
            vadds(visible, -self.window_size, mask=mask32), 0, mask=mask32
        )
        window_start = vmax(window_start, sink_end, mask=mask32)
        valid = vcmp_lt(key_idx, visible, mask=mask32)
        valid = vcmp_ge(key_idx, sink_end, mask=valid)
        valid = vcmp_lt(key_idx, window_start, mask=valid)
        return vselect(key, zero, cond_mask=valid)


@kernel
class StemIndexerKernel:
    """遍历metadata并调度同一融合Kernel中的Cube和Vector计算。"""

    def __init__(self, attrs, precision):
        self.precision = precision
        self.causal = attrs.causal
        self.alpha_is_one = attrs.alpha == 1.0
        self.stem_block_size = attrs.stem_block_size
        self.initial_blocks = attrs.initial_blocks
        self.window_size = attrs.window_size
        self.block_idx = get_block_idx()
        self.subblock = get_subblock_id()
        self.matmul = Matmul()
        # 接收Fixpipe分发的32×64原始分数，三槽跨核交给Vector的Scale/Bias消费。
        self.score_channel = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((M_VECTOR, N_CHUNK)), depth=3
        )
        self.vector = Vector(precision, attrs, self.subblock)

    def __call__(
        self,
        gm_q: Tensor,
        gm_k: Tensor,
        gm_bias: Tensor,
        gm_metadata: Tensor,
        gm_q_seq_lens: Tensor,
        gm_kv_seq_lens: Tensor,
        gm_num_prompt_tokens: Tensor,
        gm_indices: Tensor,
        gm_lens: Tensor,
        kv_heads: int,
        group_size: int,
        alpha: float,
    ):
        # 读取当前section范围，按metadata遍历本核任务。
        section_count = dtypes.int64(gm_metadata[0])
        for section_index in range(0, section_count):
            metadata_base = (
                HEAD_METADATA_STRIDE
                + section_index * SECTION_METADATA_STRIDE
                + self.block_idx * FA_METADATA_STRIDE
            )
            start_bn = dtypes.int64(gm_metadata[metadata_base])
            # metadata中的m是G×S1_blocks合轴后按M64切分的任务序号。
            start_m = dtypes.int64(gm_metadata[metadata_base + 1])
            end_bn = dtypes.int64(gm_metadata[metadata_base + 3])
            end_m = dtypes.int64(gm_metadata[metadata_base + 4])

            # end_bn可能是排他结束位置B×Nkv，先裁剪以避免读取不存在的batch。
            for bn in range(
                start_bn, min(end_bn + 1, gm_q_seq_lens.shape[0] * kv_heads)
            ):
                batch = bn // kv_heads
                kv_head = bn - batch * kv_heads
                q_tokens = dtypes.int64(gm_q_seq_lens[batch])
                kv_tokens = dtypes.int64(gm_kv_seq_lens[batch])
                q_blocks = (q_tokens + self.stem_block_size - 1) // self.stem_block_size
                kv_blocks = (
                    kv_tokens + self.stem_block_size - 1
                ) // self.stem_block_size
                m_begin = 0
                if bn == start_bn:
                    m_begin = start_m
                m_end = (group_size * q_blocks + M_TILE - 1) // M_TILE
                if bn == end_bn:
                    m_end = end_m
                for metadata_m in range(m_begin, m_end):
                    self._process_metadata_m(
                        gm_bias,
                        gm_q,
                        gm_k,
                        gm_q_seq_lens,
                        gm_kv_seq_lens,
                        gm_num_prompt_tokens,
                        gm_indices,
                        gm_lens,
                        batch,
                        kv_head,
                        group_size,
                        alpha,
                        metadata_m,
                        q_blocks,
                        kv_blocks,
                    )

    @jit
    def _dynamic_topk_budget(self, q_block, q_blocks, kv_blocks, prompt_blocks, alpha):
        # 标量FP32插值先应用MAX_TOPK上限，再按候选数量裁剪。
        start = prompt_blocks
        if prompt_blocks >= TOPK_BUDGET_DIRECT_PROMPT_LIMIT:
            if prompt_blocks < TOPK_BUDGET_LONG_PROMPT_THRESHOLD:
                start = dtypes.int64(
                    dtypes.float32(prompt_blocks) * TOPK_BUDGET_MEDIUM_PROMPT_RATIO
                    + TOPK_BUDGET_BASE_OFFSET
                )
            else:
                start = dtypes.int64(
                    dtypes.float32(prompt_blocks) * TOPK_BUDGET_LONG_PROMPT_RATIO
                    + TOPK_BUDGET_BASE_OFFSET
                )
        budget = start
        position = q_block + kv_blocks - q_blocks
        decay_length = prompt_blocks - start
        if const_expr(not self.alpha_is_one):
            if position >= start:
                if decay_length > 1:
                    start_f = dtypes.float32(start)
                    end_f = start_f * alpha
                    ratio = dtypes.float32(position - start) / dtypes.float32(
                        decay_length - 1
                    )
                    budget = dtypes.int64(start_f + ratio * (end_f - start_f))
                    budget = min(max(budget, 1), start)
        return min(budget, MAX_TOPK)

    @jit
    def _visible_blocks(self, q_block, q_blocks, kv_blocks, decode):
        visible = kv_blocks
        if const_expr(self.causal):
            if decode == 0:
                visible = max(min(kv_blocks - q_blocks + q_block + 1, kv_blocks), 0)
        return visible

    @jit
    def _compute_score_group(
        self,
        gm_bias,
        gm_q,
        gm_k,
        batch,
        q_head,
        kv_head,
        q_block_start,
        key_base,
        compute_blocks,
    ):
        self.vector.load_bias(gm_bias, batch, kv_head, key_base)
        for n_sub in range(N_OUTER // N_CHUNK):
            chunk_base = key_base + n_sub * N_CHUNK
            if chunk_base < compute_blocks:
                self.matmul.compute_qk(gm_k, batch, kv_head, chunk_base)
                self.matmul.store_score(self.score_channel)
                score = self.score_channel.next_read()
                if const_expr(self.precision == 2):
                    self.vector.p2_scale_bias_stream(
                        key_base,
                        score,
                        self.vector.bias_ub,
                        compute_blocks,
                        n_sub * N_CHUNK,
                    )
                else:
                    self.vector.p1_scale_bias_stream(
                        key_base,
                        score,
                        self.vector.bias_ub,
                        n_sub * N_CHUNK,
                    )
            else:
                self.vector.zero_keys_chunk(n_sub * N_CHUNK)

    @jit
    def _process_head_segment(
        self,
        gm_bias,
        gm_q,
        gm_k,
        gm_q_seq_lens,
        gm_kv_seq_lens,
        gm_num_prompt_tokens,
        gm_indices,
        gm_lens,
        batch,
        q_head,
        kv_head,
        q_block_start,
        q_block_count,
        q_blocks,
        kv_blocks,
        alpha,
    ):
        # 在一个Q头片段内，直出/稀疏状态及裁剪后的K范围单调变化，因此末行可确定整段的最大计算范围。
        # 先用末行确定整段任务的计算范围；各行自己的预算和可见范围在下方单独生成。
        last_q_block = q_block_start + q_block_count - 1
        q_tokens = dtypes.int64(gm_q_seq_lens[batch])
        kv_tokens = dtypes.int64(gm_kv_seq_lens[batch])
        prompt_tokens = dtypes.int64(gm_num_prompt_tokens[batch])
        prompt_blocks = (
            prompt_tokens + self.stem_block_size - 1
        ) // self.stem_block_size
        # 单token且prompt覆盖当前KV长度时走decode路径，全部有效K块可见。
        decode = 0
        if q_tokens == 1:
            if prompt_tokens >= kv_tokens:
                decode = 1
        # 末行的可见K块数和TopK预算。
        last_visible = self._visible_blocks(last_q_block, q_blocks, kv_blocks, decode)
        last_budget = self._dynamic_topk_budget(
            last_q_block, q_blocks, kv_blocks, prompt_blocks, alpha
        )
        # sink和末尾window固定保留；窗口起点不早于sink终点，避免重叠计数。
        last_sink_end = min(self.initial_blocks, last_visible)
        last_window_start = max(last_visible - self.window_size, last_sink_end)
        last_forced = last_sink_end + last_visible - last_window_start
        # 扣除固定保留区域，得到中间候选数量。
        last_candidates = max(last_visible - last_forced, 0)
        # 计算范围覆盖到末行窗口起点，按N64向上对齐且不超过有效KV块数。
        # 对齐范围中的非候选位置由后续逐行掩码过滤。
        window_start = max(last_visible - self.window_size, self.initial_blocks)
        compute_blocks = min(
            kv_blocks,
            ((window_start + N_CHUNK - 1) // N_CHUNK) * N_CHUNK,
        )

        # 每个Vector侧批量生成至多32行的可见范围、预算及输出参数，跨S2轮次复用。
        self.vector.build_task_params(
            gm_q_seq_lens,
            gm_kv_seq_lens,
            gm_num_prompt_tokens,
            batch,
            q_block_start,
            q_block_count,
            q_blocks,
            kv_blocks,
            last_budget,
            decode,
            prompt_blocks,
            alpha,
        )
        # 候选数量超过预算时才计算分数并执行TopK，否则直接打包输出。
        if last_candidates > last_budget:
            # Q完整特征分片加载到L1，在当前head区段的所有N256轮次中驻留复用。
            self.matmul.load_q_resident(gm_q, batch, q_head, q_block_start)
            # 沿K块方向每轮处理256个位置，尾轮有效范围由compute_blocks约束。
            key_group_count = (compute_blocks + N_OUTER - 1) // N_OUTER
            # 每个KV组对应一个生产者及包围它的消费者作用域，Channel槽的获取、提交和释放均由SDK管理。
            for key_group in range(key_group_count):
                key_base = key_group * N_OUTER
                # 当前N256内部拆成四个N64：MMAD、Fixpipe，再由Vector直接做Scale/Bias。
                self._compute_score_group(
                    gm_bias,
                    gm_q,
                    gm_k,
                    batch,
                    q_head,
                    kv_head,
                    q_block_start,
                    key_base,
                    compute_blocks,
                )
                if const_expr(self.precision == 2):
                    # N64分数Channel由Scale/Bias直接消费；TopK读取前同步所有已生成的排序键列。
                    self.vector.keys_ready()
                    # p2每四行一组做流式TopK，遍历当前M32半块的八组行。
                    for row_base in range(0, M_VECTOR, RADIX_ROWS):
                        if key_group == 0:
                            # 首轮无历史状态，只读取当前256项（从MAX_TOPK偏移开始）。
                            # 最后一轮传False，省去后续轮次不再需要的保留键写回。
                            if key_group + 1 == key_group_count:
                                self.vector.stream_topk_batch(
                                    row_base,
                                    key_base,
                                    self.vector.score_ub,
                                    MAX_TOPK,
                                    N_OUTER,
                                    False,
                                )
                            else:
                                self.vector.stream_topk_batch(
                                    row_base,
                                    key_base,
                                    self.vector.score_ub,
                                    MAX_TOPK,
                                    N_OUTER,
                                    True,
                                )
                        else:
                            # 后续轮次从偏移0读取历史256项与当前256项，共512项合并筛选。
                            if key_group + 1 == key_group_count:
                                self.vector.stream_topk_batch(
                                    row_base,
                                    key_base,
                                    self.vector.score_ub,
                                    0,
                                    STREAM_INPUT,
                                    False,
                                )
                            else:
                                self.vector.stream_topk_batch(
                                    row_base,
                                    key_base,
                                    self.vector.score_ub,
                                    0,
                                    STREAM_INPUT,
                                    True,
                                )
                else:
                    self.vector.keys_ready()
                    # p1同样按四行流式合并，但使用u32排序键和四轮字节radix。
                    for row_base in range(0, M_VECTOR, RADIX_ROWS):
                        if key_group == 0:
                            self.vector.p1_stream_topk_batch(
                                row_base,
                                key_base,
                                self.vector.score_ub,
                                MAX_TOPK,
                                N_OUTER,
                            )
                        else:
                            self.vector.p1_stream_topk_batch(
                                row_base,
                                key_base,
                                self.vector.score_ub,
                                0,
                                STREAM_INPUT,
                            )
        # 两条路径统一输出：拼接固定保留项和TopK结果（或直出索引），写回索引及长度。
        self.vector.store_task(
            gm_indices,
            gm_lens,
            batch,
            q_head,
            q_block_start,
            q_block_count,
        )

    @jit
    def _process_metadata_m(
        self,
        gm_bias,
        gm_q,
        gm_k,
        gm_q_seq_lens,
        gm_kv_seq_lens,
        gm_num_prompt_tokens,
        gm_indices,
        gm_lens,
        batch,
        kv_head,
        group_size,
        alpha,
        metadata_m,
        q_blocks,
        kv_blocks,
    ):
        flat_start = metadata_m * M_TILE
        flat_end = min(flat_start + M_TILE, group_size * q_blocks)
        # 变长S1下，metadata的M64任务可能跨Q头边界。仅在Kernel内拆分该任务，任务归属仍由紧凑metadata范围决定。
        for local_q_head in range(group_size):
            # 当前Q head在展平的G×q_blocks轴上的起点。
            head_start = local_q_head * q_blocks
            # 当前head的排他终点，范围为[head_start, head_end)。
            head_end = head_start + q_blocks
            # 当前M64任务与该head求交集：取两者较晚的起点。
            segment_start = max(flat_start, head_start)
            # 取两者较早的终点；这些坐标仍是展平轴上的坐标。
            segment_end = min(flat_end, head_end)
            # 交集长度即本任务需处理的Q block数；无交集时置零并跳过。
            q_block_count = max(0, segment_end - segment_start)
            if q_block_count > 0:
                self._process_head_segment(
                    gm_bias,
                    gm_q,
                    gm_k,
                    gm_q_seq_lens,
                    gm_kv_seq_lens,
                    gm_num_prompt_tokens,
                    gm_indices,
                    gm_lens,
                    batch,
                    kv_head * group_size + local_q_head,
                    kv_head,
                    segment_start - head_start,
                    q_block_count,
                    q_blocks,
                    kv_blocks,
                    alpha,
                )


class StemIndexerLauncher:
    """保存编译配置并发射融合Kernel。"""

    def __init__(self, topk_score_precision, attrs):
        self.topk_score_precision = topk_score_precision
        self.attrs = attrs

    @host
    def launch(
        self,
        q: Tensor,
        k: Tensor,
        bias: Tensor,
        metadata: Tensor,
        q_seq_lens: Tensor,
        kv_seq_lens: Tensor,
        num_prompt_tokens: Tensor,
        indices: Tensor,
        lens: Tensor,
        kv_heads: int,
        group_size: int,
        alpha: float,
        block_dim: int,
    ):
        StemIndexerKernel(
            self.attrs,
            self.topk_score_precision,
        )[block_dim](
            q,
            k,
            bias,
            metadata,
            q_seq_lens,
            kv_seq_lens,
            num_prompt_tokens,
            indices,
            lens,
            kv_heads,
            group_size,
            alpha,
        )


def _make_launcher(tiling):
    """根据已校验的 Tiling 信息构造 Launcher。"""
    return StemIndexerLauncher(
        tiling.attrs.topk_score_precision,
        tiling.attrs,
    )


_LAUNCH_CACHE_LOCK = RLock()


def _dynamic_stride(name):
    """构造只参与Tensor物理布局契约的动态stride。"""
    return Dim(name, min=1)


@lru_cache(maxsize=8)
def _compile_launch(
    topk_score_precision,
    stem_block_size,
    stem_stride,
    initial_blocks,
    window_size,
    causal,
    alpha_is_one,
    device,
    process_id,
):
    """按静态算法配置缓存动态shape编译句柄，不保留执行Tensor。"""
    # 设备与进程参与缓存键，避免跨设备或跨进程复用编译句柄。
    del device, process_id
    attrs = FixedAttributes(
        stem_block_size=stem_block_size,
        stem_stride=stem_stride,
        initial_blocks=initial_blocks,
        window_size=window_size,
        topk_score_precision=topk_score_precision,
        causal=causal,
        alpha=1.0 if alpha_is_one else 0.5,
    )
    launcher = StemIndexerLauncher(topk_score_precision, attrs)

    batch = Dim("B", min=1, max=65536)
    q_heads = Dim("NQ", min=32, max=64, multiple_of=32)
    kv_heads = Dim("NKV", min=2, max=8, multiple_of=2)
    q_blocks = Dim("QB", min=M_TILE, multiple_of=M_TILE)
    kv_blocks = Dim("KB", min=N_TILE, multiple_of=N_TILE)
    output_blocks = Dim("OB", min=OUTPUT_CAPACITY, multiple_of=64)
    metadata_capacity = Dim("META", min=1)

    q_spec = TensorSpec(
        (batch, q_heads, 2, q_blocks, K_L1),
        dtypes.bfloat16,
        stride=(
            _dynamic_stride("Q_S0"),
            _dynamic_stride("Q_S1"),
            K_L1,
            2 * K_L1,
            1,
        ),
    )
    k_spec = TensorSpec(
        (batch, kv_heads, 2, kv_blocks, K_L1),
        dtypes.bfloat16,
        stride=(
            _dynamic_stride("K_S0"),
            _dynamic_stride("K_S1"),
            K_L1,
            2 * K_L1,
            1,
        ),
    )
    bias_spec = TensorSpec(
        (batch, kv_heads, kv_blocks),
        dtypes.float32,
        stride=(_dynamic_stride("BIAS_S0"), kv_blocks, 1),
    )
    indices_spec = TensorSpec(
        (batch, q_heads, q_blocks, output_blocks),
        dtypes.int32,
        stride=(
            _dynamic_stride("OUT_S0"),
            _dynamic_stride("OUT_S1"),
            output_blocks,
            1,
        ),
    )
    lens_spec = TensorSpec(
        (batch, q_heads, q_blocks, LENS_STORAGE_STRIDE),
        dtypes.int32,
        stride=(
            _dynamic_stride("LENS_S0"),
            _dynamic_stride("LENS_S1"),
            LENS_STORAGE_STRIDE,
            1,
        ),
    )
    contiguous_i32_batch = TensorSpec((batch,), dtypes.int32)
    return dsl.compile(
        launcher.launch,
        q_spec,
        k_spec,
        bias_spec,
        TensorSpec((metadata_capacity,), dtypes.int32),
        contiguous_i32_batch,
        contiguous_i32_batch,
        contiguous_i32_batch,
        indices_spec,
        lens_spec,
        dtypes.int64,
        dtypes.int64,
        dtypes.float32,
        dtypes.int64,
    )


def _launch_cached(launcher, args):
    """复用动态编译句柄，并为本次调用绑定Tensor地址和当前stream。"""
    attrs = launcher.attrs
    cache_key = (
        launcher.topk_score_precision,
        attrs.stem_block_size,
        attrs.stem_stride,
        attrs.initial_blocks,
        attrs.window_size,
        bool(attrs.causal),
        attrs.alpha == 1.0,
        args[0].device,
        os.getpid(),
    )
    with torch.npu.device(args[0].device):
        with _LAUNCH_CACHE_LOCK:
            program = _compile_launch(*cache_key)
        program(*args)


def _prepare_inputs(
    qflat: torch.Tensor,
    kflat: torch.Tensor,
    vbias: torch.Tensor,
    q_seq_lens: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    num_prompt_tokens: torch.Tensor,
    *,
    attrs: FixedAttributes,
    block_dim: int | None,
    metadata: torch.Tensor | None = None,
):
    """准备主算子所需的张量、输出和调度参数。"""
    tiling = validate_and_resolve(
        qflat,
        kflat,
        vbias,
        q_seq_lens,
        kv_seq_lens,
        num_prompt_tokens,
        metadata,
        block_dim,
        attrs=attrs,
        attributes_type=FixedAttributes,
        m_tile=M_TILE,
        n_tile=N_TILE,
    )

    target = tiling.target_device
    qflat = qflat.to(target).contiguous()
    kflat = kflat.to(target).contiguous()
    vbias = vbias.to(target).contiguous()
    launcher = _make_launcher(tiling)

    q_pad = tiling.q_blocks_padded - tiling.q_blocks_max
    kv_pad = tiling.kv_blocks_padded - tiling.kv_blocks_max
    q_padded = F.pad(qflat, (0, 0, 0, q_pad)) if q_pad else qflat
    k_padded = F.pad(kflat, (0, 0, 0, kv_pad)) if kv_pad else kflat
    bias_padded = F.pad(vbias, (0, kv_pad)) if kv_pad else vbias
    output_blocks_padded = max(OUTPUT_CAPACITY, tiling.kv_blocks_padded)
    index_shape = (
        tiling.batch_size,
        tiling.q_heads,
        tiling.q_blocks_padded,
        output_blocks_padded,
    )
    len_shape = (
        tiling.batch_size,
        tiling.q_heads,
        tiling.q_blocks_padded,
        LENS_STORAGE_STRIDE,
    )
    indices_padded = torch.empty(index_shape, dtype=torch.int32, device=target)
    lens_padded = torch.empty(len_shape, dtype=torch.int32, device=target)
    # 无效Q行以及未写到的索引尾部仍遵循-1/0输出约定。
    indices_padded.fill_(-1)
    lens_padded.zero_()
    # 将特征维拆成[2,1024]，用于两个 L1 特征分片的 ND2NZ 搬运。
    q_kernel = q_padded.view(*q_padded.shape[:-1], 2, K_L1).permute(0, 1, 3, 2, 4)
    k_kernel = k_padded.view(*k_padded.shape[:-1], 2, K_L1).permute(0, 1, 3, 2, 4)
    call_args = (
        q_kernel,
        k_kernel,
        bias_padded,
        metadata,
        q_seq_lens.to(target).contiguous(),
        kv_seq_lens.to(target).contiguous(),
        num_prompt_tokens.to(target).contiguous(),
        indices_padded,
        lens_padded,
        tiling.kv_heads,
        tiling.q_heads // tiling.kv_heads,
        float(attrs.alpha),
        tiling.block_dim,
    )
    indices = indices_padded[:, :, : tiling.q_blocks_max, : tiling.kv_blocks_max]
    lens = lens_padded[:, :, : tiling.q_blocks_max, 0]
    return launcher, call_args, indices, lens


def stem_indexer(
    q,
    k,
    bias,
    q_seq_lens,
    kv_seq_lens,
    num_prompt_tokens,
    *,
    attrs=FixedAttributes(),
    metadata=None,
    block_dim=None,
):
    """使用外部 AICPU Metadata 执行算子，返回索引和长度，不主动同步设备。"""
    launcher, args, indices, lens = _prepare_inputs(
        q,
        k,
        bias,
        q_seq_lens,
        kv_seq_lens,
        num_prompt_tokens,
        attrs=attrs,
        block_dim=block_dim,
        metadata=metadata,
    )
    _launch_cached(launcher, args)
    return indices, lens
