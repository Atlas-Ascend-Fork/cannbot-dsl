# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""QuantBlockSparseAttn：mode1 多宏块在线 Cube/Vector 融合实现。

当前执行范围：非等长 TND/NTD Query，1<=B<=65536，1<=N2<=8，1<=N1/N2<=16，D=128，
PA_BNBD（连续或按页 K/V/scale 交错存储），mask_mode=0/3，可选 FP32 LSE。
支持 Q/KV 尾块、空列表及奇数块。
每个 Q 块的有效稀疏列表内逻辑索引不得重复；不同逻辑块允许共享物理页。
每两个稀疏块组成 256 列宏块，保持原在线最大值及 FP8 P 量化边界。

Q 直搬 L1 -> K@Q.T -> DN softmax/FP8 P(NZ UB/L1) -> P.T@V -> FP32 累加 -> BF16 GM。
Matmul 管理 Q/K/V L1 和 L0；Vector 持有 UB；kernel 持有 QK/P/PV 三个跨核 Channel。
每个 AIV 处理 64 行。QK/P/PV 交接深度为 2/3/1，无 GM 中间结果或手工地址别名。
"""

import math
from threading import Lock
from typing import NamedTuple
from types import SimpleNamespace

import cannbotdsl
from cannbotdsl.ring_buffer import RingBuffer
import torch
from cannbotdsl import (
    Buffer,
    CArray,
    MemLoc,
    RegLayout,
    Tensor,
    TensorSpec,
    const_expr,
    datastruct,
    dtypes,
    get_block_idx,
    get_platform_info,
    get_subblock_id,
    jit,
    kernel,
    copy_options,
    matmul,
    mem_copy,
    permute,
    tile_slice,
    vf,
)
from cannbotdsl import reg as rr
from cannbotdsl import make_layout, reinterpret
from cannbotdsl.lang.host import host

TILE_M = 128
VECTOR_M = 64
MACRO_N = 256
HEAD_DIM = 128
SPARSE_BLOCK = 128
HALF_SPARSE_BLOCK = SPARSE_BLOCK // 2
# P is FP8, so its element offsets are byte offsets.  The padded NZ P-UB
# layout stores two 64-DataBlock halves 65 DataBlocks apart; four bursts from
# each half are packed contiguously into the corresponding P-L1 AIV region.
DMA_BLOCK_BYTES = 32
P_DMA_BURST_COUNT = MACRO_N // VECTOR_M
P_DMA_BURST_BLOCKS = MACRO_N // 4
P_DMA_BURST_BYTES = P_DMA_BURST_BLOCKS * DMA_BLOCK_BYTES
P_UB_HALF_OFFSET_BYTES = (P_DMA_BURST_BLOCKS + 1) * DMA_BLOCK_BYTES
P_UB_BURST_STEP_BYTES = 2 * P_UB_HALF_OFFSET_BYTES
P_UB_SLOT_BYTES = P_DMA_BURST_COUNT * P_UB_BURST_STEP_BYTES
P_L1_HALF_BYTES = P_DMA_BURST_COUNT * P_DMA_BURST_BYTES
P_L1_SUBBLOCK_BYTES = 2 * P_L1_HALF_BYTES
# metadata ABI 的槽位上限；它们不是目标设备的实际核数。
METADATA_HEADER = 8
METADATA_CORE_FIELDS = 8
METADATA_AIC_SLOTS = 36
METADATA_AIV_SLOTS = 72
METADATA_SECTION_SIZE = METADATA_CORE_FIELDS * METADATA_AIC_SLOTS
METADATA_FD_SIZE = METADATA_CORE_FIELDS * METADATA_AIV_SLOTS

SOFTMAX_PIPELINE_LAG = 1
PV_PIPELINE_LAG = 2
UPDATE_PIPELINE_LAG = 3
PIPELINE_DEPTH = UPDATE_PIPELINE_LAG + 1
PIPELINE_DRAIN_STEPS = UPDATE_PIPELINE_LAG


@datastruct(c_name="QbsaPipelineState")
class PipelineState:
    clock: CArray[dtypes.int64, 1]
    values: CArray[dtypes.int64, 60]


class PipelineMetadataRing:
    """Keep four iterations of scalar metadata for the staged attention pipeline."""

    def __init__(self, depth, *fields):
        self.depth = depth
        self.fields = fields
        self.state = PipelineState(clock=[0], values=[0] * (depth * len(fields)))

    def push(self, **values):
        offset = self.state.clock[0] * len(self.fields)
        for i, field in enumerate(self.fields):
            self.state.values[offset + i] = dtypes.int64(values[field])

    def tap(self, lag):
        offset = (
            (self.state.clock[0] + self.depth - lag) % self.depth * len(self.fields)
        )
        return SimpleNamespace(
            **{
                field: self.state.values[offset + i]
                for i, field in enumerate(self.fields)
            }
        )

    def advance(self):
        self.state.clock[0] = (self.state.clock[0] + 1) % self.depth


@datastruct(c_name="QbsaVectorScaleState")
class VectorScaleState:
    """Three in-flight V descale values for the vector pipeline."""

    values: CArray[dtypes.float32, 3]


@datastruct(c_name="QbsaSparsePair")
class SparsePair:
    logical0: dtypes.int64
    logical1: dtypes.int64
    page0: dtypes.int64
    page1: dtypes.int64
    valid0: dtypes.int64
    valid1: dtypes.int64


@datastruct(c_name="QbsaSparseWork")
class SparseWork:
    batch: dtypes.int64
    q_head: dtypes.int64
    q_block: dtypes.int64
    kv_used: dtypes.int64
    sparse_kv_block_count: dtypes.int64
    kv_block_pair_loop_idx: dtypes.int64


# ---------------------------------------------------------------------------
# Cube：满宏块共享 L0A 双槽；真实 K 尾块使用独立操作数槽。
# ---------------------------------------------------------------------------


class Matmul:
    def __init__(self, shared_l0a):
        self.shared_l0a = shared_l0a
        fp8 = dtypes.float8_e4m3fn
        # 子块只有 128 行，但目标 NZ 的 C0 间距必须按父宏块 256 行计算。
        self.q_l1 = RingBuffer(
            MemLoc.L1, fp8, make_layout((TILE_M, HEAD_DIM)), depth=1
        ).next_write()
        self.k_l1 = RingBuffer(
            MemLoc.L1, fp8, make_layout((MACRO_N, HEAD_DIM)), depth=2
        )
        # 满宏块使用跨宏块持久双槽；尾宏块在 compute_pv 中按真实 K
        # 使用独立单槽，避免无效的补零与 Cube 计算。
        self.v_l1 = RingBuffer(
            MemLoc.L1, fp8, make_layout((MACRO_N, HEAD_DIM)), depth=2, data_format="nz"
        )
        if shared_l0a:
            self.k_l0a = RingBuffer(
                MemLoc.L0A, fp8, make_layout((MACRO_N, HEAD_DIM)), depth=2
            )
            self.p_l0a = self.k_l0a
        else:
            self.k_l0a = RingBuffer(
                MemLoc.L0A, fp8, make_layout((MACRO_N, HEAD_DIM)), depth=1
            )
            self.p_l0a = RingBuffer(
                MemLoc.L0A,
                fp8,
                make_layout((MACRO_N, TILE_M)),
                depth=1,
                data_format="nz",
            )
        self.q_l0b = RingBuffer(
            MemLoc.L0B, fp8, make_layout((TILE_M, HEAD_DIM)), depth=1
        ).next_write()
        self.v_l0b = RingBuffer(
            MemLoc.L0B, fp8, make_layout((HEAD_DIM, MACRO_N)), depth=1
        ).next_write()
        self.qk_l0c = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((MACRO_N, TILE_M)), depth=2
        )
        # QK and PV share the same ring; each compute phase selects one slot.

    def load_query(self, query):
        # 搬运真实 Q 行数，但物理 NZ 的 C0 步长固定为父缓冲的 128 行。
        mem_copy(self.q_l1, query)

    def compute_qk(self, key0, key1, qk_ub):
        # 同一对稀疏槽位已按逻辑块编号排序，物理页可以不连续。
        k_input = self.k_l0a.next_write()
        accumulator = self.qk_l0c.next_write()
        k_slot = self.k_l1.next_write()
        mem_copy(tile_slice(k_slot, (128, 128), (0, 0)), key0)
        mem_copy(tile_slice(k_slot, (128, 128), (1, 0)), key1)
        mem_copy(k_input, k_slot)
        mem_copy(self.q_l0b, self.q_l1)
        matmul(accumulator, k_input, self.q_l0b, init=True)
        mem_copy(qk_ub.next_write(), accumulator.partition(dim=1))

    @jit
    def _compute_full_pv(self, p_l1, value0, value1, p_input, accumulator):
        v_slot = reinterpret(self.v_l1.next_write(), data_format="zn")
        mem_copy(tile_slice(v_slot, (SPARSE_BLOCK, HEAD_DIM), (0, 0)), value0)
        mem_copy(tile_slice(v_slot, (SPARSE_BLOCK, HEAD_DIM), (1, 0)), value1)
        mem_copy(p_input, p_l1.permute(1, 0))
        mem_copy(self.v_l0b, v_slot.permute(1, 0))
        matmul(accumulator, p_input, self.v_l0b, init=True)

    @jit
    def compute_pv(self, p_l1, value0, value1, pair: SparsePair, pv_ub):
        p_input = reinterpret(
            self.p_l0a.next_write(),
            layout=make_layout((TILE_M, MACRO_N)),
            data_format="nz",
        )
        accumulator = reinterpret(
            self.qk_l0c.next_write(), layout=make_layout((TILE_M, HEAD_DIM))
        )
        if const_expr(self.shared_l0a):
            # 该编译特化仅适用于每个宏块均由两个完整页组成的调用。
            self._compute_full_pv(p_l1, value0, value1, p_input, accumulator)
        elif pair.valid0 + pair.valid1 == MACRO_N:
            self._compute_full_pv(p_l1, value0, value1, p_input, accumulator)
        else:
            # 每个Q最多一个尾宏块，位于流水排空端；保留动态真实K单槽，
            # 避免整槽清零的并发风险和固定K=256的额外Cube计算。
            tail_v_l1 = RingBuffer(
                MemLoc.L1,
                dtypes.float8_e4m3fn,
                make_layout((MACRO_N, HEAD_DIM)),
                depth=1,
                data_format="nz",
            ).next_write()
            tail_slot = reinterpret(tail_v_l1, data_format="zn")
            mem_copy(
                tile_slice(tail_slot, (SPARSE_BLOCK, HEAD_DIM), (0, 0)),
                value0[: pair.valid0, None],
            )
            if pair.valid1 > 0:
                mem_copy(
                    tile_slice(tail_slot, (SPARSE_BLOCK, HEAD_DIM), (1, 0)),
                    value1[: pair.valid1, None],
                )
            valid_k = pair.valid0 + pair.valid1
            mem_copy(p_input, p_l1.permute(1, 0))
            mem_copy(
                self.v_l0b[None, :valid_k],
                tail_slot[:valid_k, None].permute(1, 0),
            )
            matmul(
                accumulator,
                p_input[None, :valid_k],
                self.v_l0b[None, :valid_k],
                init=True,
            )
        mem_copy(pv_ub, accumulator.partition(dim=0))


# ---------------------------------------------------------------------------
# Vector：FP32 descale/softmax、FP8 NZ 写入、FP32 归一化与 BF16 写回。
# ---------------------------------------------------------------------------


class Vector:
    def __init__(self, subblock_idx, mask_mode, return_softmax_lse):
        self.subblock_idx = subblock_idx
        self.mask_mode = mask_mode
        if mask_mode == 3:
            self.mask0 = RingBuffer(
                MemLoc.UB, dtypes.uint8, make_layout((SPARSE_BLOCK, VECTOR_M)), depth=1
            ).next_write()
            self.mask1 = RingBuffer(
                MemLoc.UB, dtypes.uint8, make_layout((SPARSE_BLOCK, VECTOR_M)), depth=1
            ).next_write()
        # DMA 单次生产、softmax 单次消费；双槽重叠下一宏块加载与当前 VF 读取。
        self.q_scale = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((TILE_M,)), depth=2
        )
        self.k_scale0 = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((SPARSE_BLOCK,)), depth=1
        ).next_write()
        self.k_scale1 = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((SPARSE_BLOCK,)), depth=1
        ).next_write()
        # Vec1 到 Vec2 相距两拍，三组 Q 状态覆盖同时在途的独立任务。
        self.v_scale_state = VectorScaleState(values=[0.0, 0.0, 0.0])
        self.row_max = Buffer(MemLoc.UB, dtypes.float32, make_layout((3 * VECTOR_M, 1)))
        self.row_sum = Buffer(MemLoc.UB, dtypes.float32, make_layout((3 * VECTOR_M, 1)))
        # Softmax在第1拍生产，输出更新在第3拍消费；三槽覆盖两拍依赖距离。
        self.rescale = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((VECTOR_M, 1)), depth=3
        )
        # 单一根 Channel 管理 32 KiB：累加阶段存 FP32 原始位，结束后原地压成 BF16。
        # 存储类型不代表累加精度；跨位宽寄存器视图保持所有数据位，不发生数值转换。
        self.output_storage = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((VECTOR_M * 2, HEAD_DIM)), depth=1
        ).next_write()
        self.p_ub = RingBuffer(
            MemLoc.UB,
            dtypes.float8_e4m3fn,
            make_layout((VECTOR_M, MACRO_N)),
            depth=2,
            data_format="nz",
            physical_layout=make_layout(
                (MACRO_N // 32, VECTOR_M // 16, 16, 32),
                stride=(VECTOR_M * 32 + 32, 512, 32, 1),
            ),
        )
        self.out_ub = tile_slice(self.output_storage, (VECTOR_M, HEAD_DIM), (0, 0))
        if return_softmax_lse:
            self.lse_ub = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((VECTOR_M,)), depth=2
            )

    @jit
    def initialize_empty(self, q_valid_rows: int, state_slot: int):
        # 空任务显式写出行状态和零输出；非空任务由首块直接初始化。
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            storage_mask, _ = rr.update_mask(128, elem_bits=16)
            zero = rr.vdups(0.0, dtypes.float32, mask=full)
            minimum = rr.vdups(-3.4028234663852886e38, dtypes.float32, mask=full)
            state_offset = state_slot * VECTOR_M
            rr.vstore(self.row_max, state_offset, minimum, full)
            rr.vstore(self.row_sum, state_offset, zero, full)
            zero_bits = rr.vreinterpret_lanes(zero, dtypes.bfloat16)
            for row in range(q_valid_rows):
                for col in tuple(range(0, HEAD_DIM, 64)):
                    rr.vstore(
                        self.output_storage,
                        (row * HEAD_DIM + col) * 2,
                        zero_bits,
                        storage_mask,
                    )

    def remember_v_scale(self, v_scale, state_slot: int):
        self.v_scale_state.values[state_slot] = v_scale[0]

    @jit
    def load_qk_scales(self, q_scale, k_scale0, k_scale1):
        mem_copy(
            self.q_scale.next_write(), q_scale, options=copy_options(engine="nddma")
        )
        # DMA 是唯一生产者；未写尾行在 softmax 寄存器内隔离。
        mem_copy(self.k_scale0, k_scale0)
        mem_copy(self.k_scale1, k_scale1)

    @jit
    def _load_mask_block(self, destination, atten_mask, delta):
        # DN 模板与 score 同向：KV 行、Q 列，每个 AIV 读取连续 64 个 Q。
        q_offset = (MACRO_N if delta > MACRO_N else delta) if delta >= 0 else 0
        kv_offset = (TILE_M if -delta > TILE_M else -delta) if delta < 0 else 0
        q_begin = q_offset + self.subblock_idx * VECTOR_M
        mask_tile = atten_mask[
            kv_offset : kv_offset + SPARSE_BLOCK,
            q_begin : q_begin + VECTOR_M,
        ]
        mem_copy(destination, mask_tile, options=copy_options(engine="nddma"))

    @jit
    def load_mask(self, atten_mask, delta0, delta1, valid0, valid1):
        # 与原算子一致：只有因果边界穿过当前子块时才搬入真实 mask。
        # 完全可见的子块不产生 GM→UB 搬运，softmax 也不会读取对应 Channel。
        if delta0 < valid0 - 1:
            self._load_mask_block(self.mask0, atten_mask, delta0)
        if delta1 < valid1 - 1:
            self._load_mask_block(self.mask1, atten_mask, delta1)

    @jit
    def softmax(
        self,
        qk_ub,
        softmax_scale: float,
        p_scale_value: float,
        valid0: int,
        valid1: int,
        full_kv: bool,
        mask0_enabled: bool,
        mask1_enabled: bool,
        q_valid_rows: int,
        kv_block_pair_loop_idx: int,
        state_slot: int,
    ):
        if kv_block_pair_loop_idx == 0:
            self._softmax(
                qk_ub,
                softmax_scale,
                p_scale_value,
                valid0,
                valid1,
                full_kv,
                mask0_enabled,
                mask1_enabled,
                q_valid_rows,
                state_slot,
                False,
            )
        else:
            self._softmax(
                qk_ub,
                softmax_scale,
                p_scale_value,
                valid0,
                valid1,
                full_kv,
                mask0_enabled,
                mask1_enabled,
                q_valid_rows,
                state_slot,
                True,
            )

    @jit
    def _softmax(
        self,
        qk_ub,
        softmax_scale: float,
        p_scale_value: float,
        valid0: int,
        valid1: int,
        full_kv: bool,
        mask0_enabled: bool,
        mask1_enabled: bool,
        q_valid_rows: int,
        state_slot: int,
        is_update: bool,
    ):
        # DN 的 lane 是 Q 行；KV 轴采用四条 max、八条 sum 累加链。
        q_scale_slot = self.q_scale.next_read()
        p_ub_slot = self.p_ub.next_write()
        if const_expr(is_update):
            rescale_slot = self.rescale.next_write()
        stride_n1, _, stride_m0, _ = p_ub_slot.physical_stride
        mask_channel0 = None
        mask_channel1 = None
        if const_expr(self.mask_mode == 3):
            mask_channel0 = self.mask0
            mask_channel1 = self.mask1
        page_configs = (
            (0, self.k_scale0, valid0, mask_channel0, mask0_enabled),
            (1, self.k_scale1, valid1, mask_channel1, mask1_enabled),
        )
        # Resolve uniform tail counts before opening the register VF region.
        # The second conditional consumes the first conditional's scalar result.
        q_offset = dtypes.int32(self.subblock_idx * VECTOR_M)
        q_count = (
            q_valid_rows - q_offset if q_valid_rows > q_offset else dtypes.int32(0)
        )
        q_count = (
            dtypes.int32(VECTOR_M) if q_count > dtypes.int32(VECTOR_M) else q_count
        )
        with vf(mode="simd"):
            full, _ = rr.update_mask(VECTOR_M, elem_bits=32)
            bytes_full, _ = rr.update_mask(256, elem_bits=8)
            minimum = rr.vdups(-3.4028234663852886e38, dtypes.float32, mask=full)
            zero = rr.vdups(0.0, dtypes.float32, mask=full)
            # 尾行只在寄存器中置零，避免原位回写推进 q_scale 的生产游标。
            q_values = rr.vload(q_scale_slot, self.subblock_idx * VECTOR_M)
            q_active, _ = rr.update_mask(q_count, elem_bits=32)
            q_values = rr.vselect(q_values, zero, cond_mask=q_active)
            q_scale = rr.vmuls(q_values, softmax_scale, mask=full)
            log_p = rr.vlog(
                rr.vdups(p_scale_value, dtypes.float32, mask=full),
                mask=full,
            )
            maximum0 = minimum
            maximum1 = minimum
            maximum2 = minimum
            maximum3 = minimum
            for page, k_scale, valid, mask_channel, page_mask_enabled in page_configs:
                # Four rows feed four independent max chains per iteration,
                # increasing vector instruction-level parallelism.
                for row in range(0, SPARSE_BLOCK, 4):
                    if const_expr(full_kv):
                        s0 = self._descale_dn_full_row(
                            qk_ub, k_scale, row, page, q_scale, full
                        )
                        s1 = self._descale_dn_full_row(
                            qk_ub, k_scale, row + 1, page, q_scale, full
                        )
                        s2 = self._descale_dn_full_row(
                            qk_ub, k_scale, row + 2, page, q_scale, full
                        )
                        s3 = self._descale_dn_full_row(
                            qk_ub, k_scale, row + 3, page, q_scale, full
                        )
                    else:
                        s0 = self._descale_dn_row(
                            qk_ub,
                            k_scale,
                            row,
                            page,
                            valid,
                            q_scale,
                            minimum,
                            full,
                        )
                        s1 = self._descale_dn_row(
                            qk_ub,
                            k_scale,
                            row + 1,
                            page,
                            valid,
                            q_scale,
                            minimum,
                            full,
                        )
                        s2 = self._descale_dn_row(
                            qk_ub,
                            k_scale,
                            row + 2,
                            page,
                            valid,
                            q_scale,
                            minimum,
                            full,
                        )
                        s3 = self._descale_dn_row(
                            qk_ub,
                            k_scale,
                            row + 3,
                            page,
                            valid,
                            q_scale,
                            minimum,
                            full,
                        )
                    if const_expr(page_mask_enabled):
                        s0 = self._apply_mask(
                            s0, mask_channel, row * VECTOR_M, minimum, full
                        )
                        s1 = self._apply_mask(
                            s1, mask_channel, (row + 1) * VECTOR_M, minimum, full
                        )
                        s2 = self._apply_mask(
                            s2, mask_channel, (row + 2) * VECTOR_M, minimum, full
                        )
                        s3 = self._apply_mask(
                            s3, mask_channel, (row + 3) * VECTOR_M, minimum, full
                        )
                    rr.vstore(qk_ub, (page * SPARSE_BLOCK + row) * VECTOR_M, s0, full)
                    rr.vstore(
                        qk_ub, (page * SPARSE_BLOCK + row + 1) * VECTOR_M, s1, full
                    )
                    rr.vstore(
                        qk_ub, (page * SPARSE_BLOCK + row + 2) * VECTOR_M, s2, full
                    )
                    rr.vstore(
                        qk_ub, (page * SPARSE_BLOCK + row + 3) * VECTOR_M, s3, full
                    )
                    maximum0 = rr.vmax(maximum0, s0, mask=full)
                    maximum1 = rr.vmax(maximum1, s1, mask=full)
                    maximum2 = rr.vmax(maximum2, s2, mask=full)
                    maximum3 = rr.vmax(maximum3, s3, mask=full)
            block_max = rr.vmax(
                rr.vmax(maximum0, maximum2, mask=full),
                rr.vmax(maximum1, maximum3, mask=full),
                mask=full,
            )
            state_offset = state_slot * VECTOR_M
            if const_expr(is_update):
                previous_max = rr.vload(self.row_max, state_offset)
            else:
                previous_max = minimum
            updated_max = rr.vsub(block_max, log_p, mask=full)
            updated_max = rr.vselect(
                minimum,
                updated_max,
                cond_mask=rr.veq(block_max, minimum, mask=full),
            )
            updated_max = rr.vmax(updated_max, previous_max, mask=full)
            alpha = rr.vexp_sub(previous_max, updated_max, mask=full)
            rr.vstore(self.row_max, state_offset, updated_max, full)
            # 首块输出直接采用 PV，不生产没有消费者的 rescale 槽。
            if const_expr(is_update):
                rr.vstore(rescale_slot, 0, alpha, full)
            rr.vmem_bar("vst_vld")
            sum0 = zero
            sum1 = zero
            sum2 = zero
            sum3 = zero
            sum4 = zero
            sum5 = zero
            sum6 = zero
            sum7 = zero
            # 每个 uint32 同时构造四个 Gather 字节索引，不增加 UB 或 Host 输入。
            lane = rr.varange(0, dtypes.uint32)
            base = rr.vshl(
                rr.vbitwise_and(
                    lane, rr.vdups(15, dtypes.uint32, mask=full), mask=full
                ),
                4,
                mask=full,
            )
            quarter = rr.vshr(lane, 4, mask=full)
            base = rr.vadd(base, quarter, mask=full)
            indexes = rr.vreinterpret_lanes(
                rr.vadds(rr.vmuls(base, 0x01010101, mask=full), 0x0C080400, mask=full),
                dtypes.uint8,
            )
            # One iteration handles two rows from each half of both sparse blocks.
            # The eight independent sum chains hide vector dependency latency.
            for row in range(0, HALF_SPARSE_BLOCK, 2):
                if const_expr(full_kv):
                    e0 = self._exp_dn_full_row(qk_ub, row, 0, updated_max, full)
                    e1 = self._exp_dn_full_row(
                        qk_ub,
                        row + HALF_SPARSE_BLOCK,
                        0,
                        updated_max,
                        full,
                    )
                    e2 = self._exp_dn_full_row(qk_ub, row, 1, updated_max, full)
                    e3 = self._exp_dn_full_row(
                        qk_ub,
                        row + HALF_SPARSE_BLOCK,
                        1,
                        updated_max,
                        full,
                    )
                    e4 = self._exp_dn_full_row(qk_ub, row + 1, 0, updated_max, full)
                    e5 = self._exp_dn_full_row(
                        qk_ub,
                        row + HALF_SPARSE_BLOCK + 1,
                        0,
                        updated_max,
                        full,
                    )
                    e6 = self._exp_dn_full_row(qk_ub, row + 1, 1, updated_max, full)
                    e7 = self._exp_dn_full_row(
                        qk_ub,
                        row + HALF_SPARSE_BLOCK + 1,
                        1,
                        updated_max,
                        full,
                    )
                else:
                    e0 = self._exp_dn_row(
                        qk_ub, row, 0, valid0, updated_max, zero, full
                    )
                    e1 = self._exp_dn_row(
                        qk_ub,
                        row + HALF_SPARSE_BLOCK,
                        0,
                        valid0,
                        updated_max,
                        zero,
                        full,
                    )
                    e2 = self._exp_dn_row(
                        qk_ub, row, 1, valid1, updated_max, zero, full
                    )
                    e3 = self._exp_dn_row(
                        qk_ub,
                        row + HALF_SPARSE_BLOCK,
                        1,
                        valid1,
                        updated_max,
                        zero,
                        full,
                    )
                    e4 = self._exp_dn_row(
                        qk_ub, row + 1, 0, valid0, updated_max, zero, full
                    )
                    e5 = self._exp_dn_row(
                        qk_ub,
                        row + HALF_SPARSE_BLOCK + 1,
                        0,
                        valid0,
                        updated_max,
                        zero,
                        full,
                    )
                    e6 = self._exp_dn_row(
                        qk_ub, row + 1, 1, valid1, updated_max, zero, full
                    )
                    e7 = self._exp_dn_row(
                        qk_ub,
                        row + HALF_SPARSE_BLOCK + 1,
                        1,
                        valid1,
                        updated_max,
                        zero,
                        full,
                    )
                # 第一遍 descale 已将被 mask 的 QK 写为 minimum 并回写 qk_ub。
                # 与原算子一致，第二遍直接 exp，不重复读取和应用 mask。
                sum0 = rr.vadd(e0, sum0, mask=full)
                sum1 = rr.vadd(e1, sum1, mask=full)
                sum2 = rr.vadd(e2, sum2, mask=full)
                sum3 = rr.vadd(e3, sum3, mask=full)
                sum4 = rr.vadd(e4, sum4, mask=full)
                sum5 = rr.vadd(e5, sum5, mask=full)
                sum6 = rr.vadd(e6, sum6, mask=full)
                sum7 = rr.vadd(e7, sum7, mask=full)
                packed0, packed1 = self._pack_dn_register_pair(
                    e0,
                    e1,
                    e2,
                    e3,
                    e4,
                    e5,
                    e6,
                    e7,
                    indexes,
                    full,
                    bytes_full,
                )
                rr.vstore_strided(
                    p_ub_slot,
                    row * stride_m0,
                    packed0,
                    bytes_full,
                    block_stride=stride_n1 // 32,
                )
                rr.vstore_strided(
                    p_ub_slot,
                    (row + 1) * stride_m0,
                    packed1,
                    bytes_full,
                    block_stride=stride_n1 // 32,
                )
            current_row_sum = rr.vadd(
                rr.vadd(
                    rr.vadd(sum2, sum0, mask=full),
                    rr.vadd(sum3, sum1, mask=full),
                    mask=full,
                ),
                rr.vadd(
                    rr.vadd(sum6, sum4, mask=full),
                    rr.vadd(sum7, sum5, mask=full),
                    mask=full,
                ),
                mask=full,
            )
            if const_expr(is_update):
                previous_row_sum = rr.vload(self.row_sum, state_offset)
                rescaled_previous_row_sum = rr.vmul(alpha, previous_row_sum, mask=full)
                rescaled_previous_row_sum = rr.vselect(
                    zero,
                    rescaled_previous_row_sum,
                    cond_mask=rr.veq(previous_row_sum, zero, mask=full),
                )
                rr.vstore(
                    self.row_sum,
                    state_offset,
                    rr.vadd(rescaled_previous_row_sum, current_row_sum, mask=full),
                    full,
                )
            else:
                rr.vstore(self.row_sum, state_offset, current_row_sum, full)

    @jit
    def _descale_dn_row(self, qk_ub, k_scale, row, page, valid, q_scale, minimum, full):
        offset = (page * SPARSE_BLOCK + row) * VECTOR_M
        score = rr.vmul(rr.vload(qk_ub, offset), q_scale, mask=full)
        score = rr.vmul(score, rr.vload_broadcast(k_scale, row), mask=full)
        active, _ = rr.update_mask(VECTOR_M if row < valid else 0, elem_bits=32)
        return rr.vselect(score, minimum, cond_mask=active)

    @jit
    def _exp_dn_row(self, qk_ub, row, page, valid, updated_max, zero, full):
        prob = rr.vexp_sub(
            rr.vload(qk_ub, (page * SPARSE_BLOCK + row) * VECTOR_M),
            updated_max,
            mask=full,
        )
        active, _ = rr.update_mask(VECTOR_M if row < valid else 0, elem_bits=32)
        return rr.vselect(prob, zero, cond_mask=active)

    @jit
    def store_p(self, p_l1):
        # 与原算子的两次 stride DMA 一致：每次搬四个 2048 B burst，
        # 相邻源 burst 跨 4160 B，目的端连续写入一个 8192 B 半区。
        p_ub_slot = self.p_ub.next_read()
        slot = p_l1.next_write()
        src_pairs = reinterpret(
            p_ub_slot,
            data_format="nd",
            layout=make_layout(
                (1, 2, P_DMA_BURST_COUNT, P_DMA_BURST_BYTES),
                stride=(
                    P_UB_SLOT_BYTES,
                    P_UB_HALF_OFFSET_BYTES,
                    P_UB_BURST_STEP_BYTES,
                    1,
                ),
            ),
        )
        src_first = tile_slice(
            src_pairs,
            (1, 1, P_DMA_BURST_COUNT, P_DMA_BURST_BYTES),
            (0, 0, 0, 0),
        )
        src_second = tile_slice(
            src_pairs,
            (1, 1, P_DMA_BURST_COUNT, P_DMA_BURST_BYTES),
            (0, 1, 0, 0),
        )
        dst_pairs = reinterpret(
            slot,
            data_format="nd",
            layout=make_layout(
                (TILE_M // VECTOR_M, 2, P_DMA_BURST_COUNT, P_DMA_BURST_BYTES),
                stride=(P_L1_SUBBLOCK_BYTES, P_L1_HALF_BYTES, P_DMA_BURST_BYTES, 1),
            ),
        )
        dst_first = tile_slice(
            dst_pairs,
            (1, 1, P_DMA_BURST_COUNT, P_DMA_BURST_BYTES),
            (self.subblock_idx, 0, 0, 0),
        )
        dst_second = tile_slice(
            dst_pairs,
            (1, 1, P_DMA_BURST_COUNT, P_DMA_BURST_BYTES),
            (self.subblock_idx, 1, 0, 0),
        )
        mem_copy(dst_first, src_first)
        mem_copy(dst_second, src_second)

    @jit
    def update_output(
        self,
        pv_ub,
        kv_block_pair_loop_idx: int,
        kv_block_pair_loop_count: int,
        q_valid_rows: int,
        state_slot: int,
    ):
        # 原 Vec2：首块保留 raw PV；第二块为两个项分别乘 V scale 后相加。
        # 后续累加器已经反量化，只对新 PV 乘 scale；不能改为每块先归一化。
        if kv_block_pair_loop_idx == 0:
            if kv_block_pair_loop_count == 1:
                self._update_first_output(pv_ub, q_valid_rows, state_slot, True)
            else:
                self._update_first_output(pv_ub, q_valid_rows, state_slot, False)
        elif kv_block_pair_loop_idx == 1:
            self._update_output(pv_ub, q_valid_rows, state_slot, True)
        else:
            self._update_output(pv_ub, q_valid_rows, state_slot, False)

    @jit
    def _update_first_output(
        self, pv_ub, q_valid_rows: int, state_slot: int, single: bool
    ):
        v_scale_value = self.v_scale_state.values[state_slot]
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            storage_mask, _ = rr.update_mask(128, elem_bits=16)
            v_scale = rr.vdups(v_scale_value, dtypes.float32, mask=full)
            for row in range(q_valid_rows):
                for col in tuple(range(0, HEAD_DIM, 64)):
                    offset = row * HEAD_DIM + col
                    value = rr.vload(pv_ub, offset)
                    if const_expr(single):
                        value = rr.vmul(value, v_scale, mask=full)
                    rr.vstore(
                        self.output_storage,
                        offset * 2,
                        rr.vreinterpret_lanes(value, dtypes.bfloat16),
                        storage_mask,
                    )

    @jit
    def _update_output(self, pv_ub, q_valid_rows: int, state_slot: int, second: bool):
        rescale_slot = self.rescale.next_read()
        v_scale_value = self.v_scale_state.values[state_slot]
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            storage_mask, _ = rr.update_mask(128, elem_bits=16)
            v_scale = rr.vdups(v_scale_value, dtypes.float32, mask=full)
            for row in range(q_valid_rows):
                alpha = rr.vload_broadcast(rescale_slot, row)
                for col in tuple(range(0, HEAD_DIM, 64)):
                    offset = row * HEAD_DIM + col
                    raw_pv = rr.vload(pv_ub, offset)
                    current_output = rr.vmul(raw_pv, v_scale, mask=full)
                    previous_output = rr.vreinterpret_lanes(
                        rr.vload(self.output_storage, offset * 2),
                        dtypes.float32,
                    )
                    previous_output = rr.vmul(previous_output, alpha, mask=full)
                    if const_expr(second):
                        previous_output = rr.vmul(previous_output, v_scale, mask=full)
                    updated_output = rr.vadd(previous_output, current_output, mask=full)
                    rr.vstore(
                        self.output_storage,
                        offset * 2,
                        rr.vreinterpret_lanes(updated_output, dtypes.bfloat16),
                        storage_mask,
                    )

    @jit
    def finalize(self, q_valid_rows: int, state_slot: int):
        # 分母是未量化的 E 行和；只有正行和参与输出，零/NaN 行和均输出零。
        # 按地址递增读 FP32、写 BF16，压缩写入不会覆盖后续尚未读取的 FP32 数据。
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            zero = rr.vdups(0.0, dtypes.float32, mask=full)
            for row in range(q_valid_rows):
                row_sum = rr.vload_broadcast(
                    self.row_sum,
                    state_slot * VECTOR_M + row,
                )
                maximum = rr.vload_broadcast(
                    self.row_max,
                    state_slot * VECTOR_M + row,
                )
                has_positive_row_sum = rr.vgts(row_sum, 0.0, mask=full)
                invalid = rr.veqs(
                    maximum,
                    -3.4028234663852886e38,
                    mask=full,
                )
                for col in tuple(range(0, HEAD_DIM, 64)):
                    value = rr.vreinterpret_lanes(
                        rr.vload(self.output_storage, (row * HEAD_DIM + col) * 2),
                        dtypes.float32,
                    )
                    value = rr.vdiv(value, row_sum, mask=full)
                    value = rr.vselect(value, zero, cond_mask=has_positive_row_sum)
                    value = rr.vselect(zero, value, cond_mask=invalid)
                    result = rr.vcast(
                        value, dtypes.bfloat16, mask=full, reg_layout=RegLayout.ZERO
                    )
                    rr.vstore_pack(
                        self.out_ub,
                        row * HEAD_DIM + col,
                        result,
                        full,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )

    @jit
    def store_out(self, out, valid_rows: int):
        # 调用方保证当前 AIV 至少有一行；tile_slice 截断真实 GM 行边界。
        mem_copy(
            tile_slice(out, (VECTOR_M, HEAD_DIM), (self.subblock_idx, 0)),
            self.out_ub[:valid_rows, None],
        )

    @jit
    def finalize_lse(self, state_slot: int):
        # row_max 已包含 -log(p_scale)；LSE 使用未量化的行和，不重复补偿 scale。
        # 连续 64 个 lane 对应 64 行；空任务和全遮蔽行保留 -FLT_MAX。
        lse_slot = self.lse_ub.next_write()
        with vf(mode="simd"):
            full, _ = rr.update_mask(VECTOR_M, elem_bits=32)
            state_offset = state_slot * VECTOR_M
            row_sum = rr.vload(self.row_sum, state_offset)
            row_max = rr.vload(self.row_max, state_offset)
            empty_value = rr.vdups(-3.4028234663852886e38, dtypes.float32, mask=full)
            lse = rr.vadd(rr.vlog(row_sum, mask=full), row_max, mask=full)
            lse = rr.vselect(
                empty_value,
                lse,
                cond_mask=rr.veqs(row_max, -3.4028234663852886e38, mask=full),
            )
            lse = rr.vselect(
                lse, empty_value, cond_mask=rr.vgts(row_sum, 0.0, mask=full)
            )
            rr.vstore(lse_slot, 0, lse, full)

    @jit
    def store_lse(self, softmax_lse, valid_rows: int):
        # DMA derives its extent from the source, so clip both UB and GM views.
        source = self.lse_ub.next_read()
        mem_copy(
            tile_slice(softmax_lse, (VECTOR_M,), (self.subblock_idx,)),
            source[:valid_rows,],
        )

    @staticmethod
    def _apply_mask(values, mask_channel, offset, fill, lane_mask):
        visible = rr.vload_unpack(
            mask_channel, offset, unpack_mode=rr.UnpackMode.B8_TO_B32
        )
        visible = rr.veqs(visible, 1, mask=lane_mask)
        return rr.vselect(values, fill, cond_mask=visible)

    @staticmethod
    def _descale_dn_full_row(qk_ub, k_scale, row, page, q_scale, full):
        offset = (page * SPARSE_BLOCK + row) * VECTOR_M
        score = rr.vmul(rr.vload(qk_ub, offset), q_scale, mask=full)
        return rr.vmul(score, rr.vload_broadcast(k_scale, row), mask=full)

    @staticmethod
    def _exp_dn_full_row(qk_ub, row, page, updated_max, full):
        return rr.vexp_sub(
            rr.vload(qk_ub, (page * SPARSE_BLOCK + row) * VECTOR_M),
            updated_max,
            mask=full,
        )

    @staticmethod
    def _pack_dn_register_pair(
        e0,
        e1,
        e2,
        e3,
        e4,
        e5,
        e6,
        e7,
        indexes,
        full,
        bytes_full,
    ):
        # 与原算子一致，交错发射两条独立的 FP8 打包链，掩盖 Cast/Or/Gather RAW 延迟。
        p0 = rr.vcast(e0, dtypes.float8_e4m3fn, mask=full, reg_layout=RegLayout.ZERO)
        q0 = rr.vcast(e4, dtypes.float8_e4m3fn, mask=full, reg_layout=RegLayout.ZERO)
        p1 = rr.vcast(e1, dtypes.float8_e4m3fn, mask=full, reg_layout=RegLayout.ONE)
        q1 = rr.vcast(e5, dtypes.float8_e4m3fn, mask=full, reg_layout=RegLayout.ONE)
        p2 = rr.vcast(e2, dtypes.float8_e4m3fn, mask=full, reg_layout=RegLayout.TWO)
        q2 = rr.vcast(e6, dtypes.float8_e4m3fn, mask=full, reg_layout=RegLayout.TWO)
        p3 = rr.vcast(e3, dtypes.float8_e4m3fn, mask=full, reg_layout=RegLayout.THREE)
        q3 = rr.vcast(e7, dtypes.float8_e4m3fn, mask=full, reg_layout=RegLayout.THREE)
        bits0 = rr.vbitwise_or(
            rr.vreinterpret(p0, dtypes.uint8),
            rr.vreinterpret(p1, dtypes.uint8),
            mask=bytes_full,
        )
        bits1 = rr.vbitwise_or(
            rr.vreinterpret(q0, dtypes.uint8),
            rr.vreinterpret(q1, dtypes.uint8),
            mask=bytes_full,
        )
        bits0 = rr.vbitwise_or(
            bits0, rr.vreinterpret(p2, dtypes.uint8), mask=bytes_full
        )
        bits1 = rr.vbitwise_or(
            bits1, rr.vreinterpret(q2, dtypes.uint8), mask=bytes_full
        )
        bits0 = rr.vbitwise_or(
            bits0, rr.vreinterpret(p3, dtypes.uint8), mask=bytes_full
        )
        bits1 = rr.vbitwise_or(
            bits1, rr.vreinterpret(q3, dtypes.uint8), mask=bytes_full
        )
        packed0 = rr.vreinterpret(rr.vgather_reg(bits0, indexes), dtypes.float8_e4m3fn)
        packed1 = rr.vreinterpret(rr.vgather_reg(bits1, indexes), dtypes.float8_e4m3fn)
        return packed0, packed1


# ---------------------------------------------------------------------------
# Kernel / Launcher：全核任务跨 Q/section 连续四拍，最后有效 section 后排空。
# ---------------------------------------------------------------------------


@kernel
class QuantBlockSparseAttnKernel:
    def __init__(
        self, n1, q_capacity, group_size, mask_mode, return_softmax_lse, shared_l0a
    ):
        self.n1 = n1
        self.q_capacity = q_capacity
        self.group_size = group_size
        self.mask_mode = mask_mode
        self.return_softmax_lse = return_softmax_lse
        self.block_idx = get_block_idx()
        self.subblock_idx = get_subblock_id()
        self.qk_ub = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((MACRO_N, VECTOR_M)), depth=2
        )
        self.p_l1 = RingBuffer(
            MemLoc.L1, dtypes.float8_e4m3fn, make_layout((MACRO_N, TILE_M)), depth=3
        )
        self.pv_ub = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((VECTOR_M, HEAD_DIM)), depth=1
        ).next_write()
        self.matmul = Matmul(shared_l0a)
        self.vector = Vector(self.subblock_idx, mask_mode, return_softmax_lse)

    def __call__(
        self,
        out: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_descale: Tensor,
        k_descale: Tensor,
        v_descale: Tensor,
        p_scale: Tensor | None,
        sparse_indices: Tensor,
        sparse_seq_len: Tensor,
        metadata: Tensor,
        block_table: Tensor,
        seqused_kv: Tensor,
        cu_seqlens_q: Tensor,
        softmax_scale: float,
        atten_mask: Tensor | None,
        softmax_lse: Tensor | None,
    ):
        # p_scale 是运行时 Tensor 输入；只在设备侧读取，不回传 Host。
        p_scale_value = 1.0
        if p_scale is not None:
            p_scale_value = p_scale[0]
        # 同一 AIC 及其两个 AIV 按相同顺序读取各 section，Channel 跨任务复用。
        pipeline = PipelineMetadataRing(
            PIPELINE_DEPTH,
            "active",
            "batch",
            "q_head",
            "q_block",
            "kv_head",
            "kv_used",
            "q_rows",
            "q_begin",
            "q_end",
            "sparse_kv_block_count",
            "kv_block_pair_loop_idx",
            "kv_block_pair_loop_count",
            "q_valid_rows",
            "state_slot",
            "is_last",
        )
        tick = 0
        issued = 0
        task_sequence = 0
        started = 0
        for section_idx in range(
            dtypes.int64(metadata[0]),
        ):
            tick, issued, task_sequence, started = self._process_section(
                out,
                query,
                key,
                value,
                q_descale,
                k_descale,
                v_descale,
                p_scale_value,
                sparse_indices,
                sparse_seq_len,
                metadata,
                block_table,
                seqused_kv,
                cu_seqlens_q,
                softmax_scale,
                atten_mask,
                softmax_lse,
                section_idx,
                pipeline,
                tick,
                issued,
                task_sequence,
                started,
            )
        if started != 0:
            # 静态展开保证 qk_ub 的自动等待留在有效 Vec1 分支内。
            for _ in tuple(range(PIPELINE_DRAIN_STEPS)):
                self._run_delayed_stages(
                    pipeline,
                    tick,
                    issued,
                    out,
                    value,
                    q_descale,
                    k_descale,
                    sparse_indices,
                    block_table,
                    softmax_lse,
                    softmax_scale,
                    p_scale_value,
                    atten_mask,
                )
                pipeline.advance()
                tick = tick + 1

    @jit
    def _resolve_sparse_kv_block_pair(
        self,
        sparse_indices,
        block_table,
        work: SparseWork,
    ):
        """Resolve the sparse KV block pair processed by one pipeline iteration.

        Return its sorted logical block indices, physical page indices, and
        valid token counts. The final iteration may contain only one block.
        """
        slot0 = work.kv_block_pair_loop_idx * 2
        has_second = slot0 + 1 < work.sparse_kv_block_count
        logical0 = dtypes.int64(
            sparse_indices[work.batch, work.q_head, work.q_block, slot0]
        )
        # 奇数末块复用有效页，只用valid1=0描述第二页为空。
        logical1 = logical0
        if has_second:
            logical1 = dtypes.int64(
                sparse_indices[work.batch, work.q_head, work.q_block, slot0 + 1],
            )
        first = logical0 if logical0 < logical1 else logical1
        second = logical1 if logical0 < logical1 else logical0
        logical0, logical1 = first, second
        valid0 = work.kv_used - logical0 * SPARSE_BLOCK
        valid0 = SPARSE_BLOCK if valid0 > SPARSE_BLOCK else valid0
        valid1 = work.kv_used - logical1 * SPARSE_BLOCK
        valid1 = SPARSE_BLOCK if valid1 > SPARSE_BLOCK else valid1
        valid1 = valid1 if has_second else 0
        page0 = dtypes.int64(block_table[work.batch, logical0])
        page1 = dtypes.int64(block_table[work.batch, logical1])
        return SparsePair(
            logical0=logical0,
            logical1=logical1,
            page0=page0,
            page1=page1,
            valid0=valid0,
            valid1=valid1,
        )

    @jit
    def _stage_qk(
        self, key, sparse_indices, block_table, kv_head: int, work: SparseWork
    ):
        pair = self._resolve_sparse_kv_block_pair(
            sparse_indices,
            block_table,
            work,
        )
        key0 = tile_slice(
            key[pair.page0, kv_head, None, None],
            (SPARSE_BLOCK, HEAD_DIM),
            (0, 0),
        )
        key1 = tile_slice(
            key[pair.page1, kv_head, None, None],
            (SPARSE_BLOCK, HEAD_DIM),
            (0, 0),
        )
        self.matmul.compute_qk(key0, key1, self.qk_ub)

    @jit
    def _stage_softmax(
        self,
        q_scale,
        k_descale,
        sparse_indices,
        block_table,
        kv_head: int,
        q_rows: int,
        work: SparseWork,
        state_slot: int,
        softmax_scale: float,
        p_scale_value: float,
        atten_mask,
    ):
        pair = self._resolve_sparse_kv_block_pair(
            sparse_indices,
            block_table,
            work,
        )
        self._prepare_softmax(
            q_scale,
            kv_head,
            k_descale,
            pair.page0,
            pair.page1,
            softmax_scale,
            p_scale_value,
            pair.valid0,
            pair.valid1,
            atten_mask,
            work.q_block * TILE_M
            + work.kv_used
            - q_rows
            - pair.logical0 * SPARSE_BLOCK,
            work.q_block * TILE_M
            + work.kv_used
            - q_rows
            - pair.logical1 * SPARSE_BLOCK,
            work.kv_block_pair_loop_idx,
            state_slot,
        )

    @jit
    def _stage_pv(
        self, value, sparse_indices, block_table, kv_head: int, work: SparseWork
    ):
        pair = self._resolve_sparse_kv_block_pair(
            sparse_indices,
            block_table,
            work,
        )
        value0 = tile_slice(
            value[pair.page0, kv_head, None, None],
            (SPARSE_BLOCK, HEAD_DIM),
            (0, 0),
        )
        value1 = tile_slice(
            value[pair.page1, kv_head, None, None],
            (SPARSE_BLOCK, HEAD_DIM),
            (0, 0),
        )
        self.matmul.compute_pv(
            self.p_l1.next_read(),
            value0,
            value1,
            pair,
            self.pv_ub,
        )

    @jit
    def _stage_update(
        self,
        kv_block_pair_loop_idx: int,
        kv_block_pair_loop_count: int,
        q_valid_rows: int,
        state_slot: int,
    ):
        self.vector.update_output(
            self.pv_ub,
            kv_block_pair_loop_idx,
            kv_block_pair_loop_count,
            q_valid_rows,
            state_slot,
        )

    @jit
    def _dispatch_softmax(
        self,
        softmax_scale: float,
        p_scale_value: float,
        valid0: int,
        valid1: int,
        delta0: int,
        delta1: int,
        full_kv: bool,
        q_valid_rows: int,
        kv_block_pair_loop_idx: int,
        state_slot: int,
    ):
        qk_slot = self.qk_ub.next_read()
        if const_expr(self.mask_mode == 3):
            if delta0 < valid0 - 1:
                if delta1 < valid1 - 1:
                    self.vector.softmax(
                        qk_slot,
                        softmax_scale,
                        p_scale_value,
                        valid0,
                        valid1,
                        full_kv,
                        True,
                        True,
                        q_valid_rows,
                        kv_block_pair_loop_idx,
                        state_slot,
                    )
                else:
                    self.vector.softmax(
                        qk_slot,
                        softmax_scale,
                        p_scale_value,
                        valid0,
                        valid1,
                        full_kv,
                        True,
                        False,
                        q_valid_rows,
                        kv_block_pair_loop_idx,
                        state_slot,
                    )
            elif delta1 < valid1 - 1:
                self.vector.softmax(
                    qk_slot,
                    softmax_scale,
                    p_scale_value,
                    valid0,
                    valid1,
                    full_kv,
                    False,
                    True,
                    q_valid_rows,
                    kv_block_pair_loop_idx,
                    state_slot,
                )
            else:
                self.vector.softmax(
                    qk_slot,
                    softmax_scale,
                    p_scale_value,
                    valid0,
                    valid1,
                    full_kv,
                    False,
                    False,
                    q_valid_rows,
                    kv_block_pair_loop_idx,
                    state_slot,
                )
        else:
            self.vector.softmax(
                qk_slot,
                softmax_scale,
                p_scale_value,
                valid0,
                valid1,
                full_kv,
                False,
                False,
                q_valid_rows,
                kv_block_pair_loop_idx,
                state_slot,
            )

    @jit
    def _prepare_softmax(
        self,
        q_scale,
        kv_head: int,
        k_descale,
        page0: int,
        page1: int,
        softmax_scale: float,
        p_scale_value: float,
        valid0: int,
        valid1: int,
        atten_mask,
        delta0: int,
        delta1: int,
        kv_block_pair_loop_idx: int,
        state_slot: int,
    ):
        # 先按真实页 stride 选定连续 [128,1] 子块，再合并该子块的单例轴。
        k_scale0 = k_descale[page0, kv_head, None, None].view((SPARSE_BLOCK,))
        k_scale1 = k_descale[page1, kv_head, None, None].view((SPARSE_BLOCK,))
        if const_expr(self.mask_mode == 3):
            self.vector.load_mask(atten_mask, delta0, delta1, valid0, valid1)
        # 与原算子 ProcessVec1Dn 一致，优先发起 mask 的 GM→UB 搬运，
        # 让它与随后 scale 搬运及等待 MM1 结果的阶段重叠。
        self.vector.load_qk_scales(q_scale, k_scale0, k_scale1)
        if valid0 == SPARSE_BLOCK and valid1 == SPARSE_BLOCK:
            self._dispatch_softmax(
                softmax_scale,
                p_scale_value,
                valid0,
                valid1,
                delta0,
                delta1,
                True,
                dtypes.int32(q_scale.shape[0]),
                kv_block_pair_loop_idx,
                state_slot,
            )
        else:
            self._dispatch_softmax(
                softmax_scale,
                p_scale_value,
                valid0,
                valid1,
                delta0,
                delta1,
                False,
                dtypes.int32(q_scale.shape[0]),
                kv_block_pair_loop_idx,
                state_slot,
            )
        self.vector.store_p(self.p_l1)

    @jit
    def _finalize(
        self,
        out,
        softmax_lse,
        q_head: int,
        q_begin: int,
        q_end: int,
        q_block: int,
        q_valid_rows: int,
        state_slot: int,
    ):
        if self.subblock_idx * VECTOR_M < out.shape[0]:
            self.vector.finalize(q_valid_rows, state_slot)
            self.vector.store_out(out, q_valid_rows)
            if const_expr(self.return_softmax_lse):
                lse_tile = tile_slice(
                    softmax_lse[q_head, q_begin:q_end], (TILE_M,), (q_block,)
                )
                self.vector.finalize_lse(state_slot)
                self.vector.store_lse(lse_tile, q_valid_rows)

    @jit
    def _run_delayed_stages(
        self,
        pipeline,
        tick: int,
        issued: int,
        out,
        value,
        q_descale,
        k_descale,
        sparse_indices,
        block_table,
        softmax_lse,
        softmax_scale: float,
        p_scale_value: float,
        atten_mask,
    ):
        if tick >= SOFTMAX_PIPELINE_LAG and tick - SOFTMAX_PIPELINE_LAG < issued:
            delayed = pipeline.tap(SOFTMAX_PIPELINE_LAG)
            if delayed.active:
                q_scale = tile_slice(
                    q_descale[delayed.q_head, delayed.q_begin : delayed.q_end],
                    (TILE_M,),
                    (delayed.q_block,),
                )
                work = SparseWork(
                    batch=delayed.batch,
                    q_head=delayed.q_head,
                    q_block=delayed.q_block,
                    kv_used=delayed.kv_used,
                    sparse_kv_block_count=delayed.sparse_kv_block_count,
                    kv_block_pair_loop_idx=delayed.kv_block_pair_loop_idx,
                )
                self._stage_softmax(
                    q_scale,
                    k_descale,
                    sparse_indices,
                    block_table,
                    delayed.kv_head,
                    delayed.q_rows,
                    work,
                    delayed.state_slot,
                    softmax_scale,
                    p_scale_value,
                    atten_mask,
                )
        # PV UB 只有一槽：先消费 lag3 的旧结果，再生产 lag2 的新结果。
        if tick >= UPDATE_PIPELINE_LAG and tick - UPDATE_PIPELINE_LAG < issued:
            delayed = pipeline.tap(UPDATE_PIPELINE_LAG)
            out_tile = tile_slice(
                out[delayed.q_head, delayed.q_begin : delayed.q_end, None],
                (TILE_M, HEAD_DIM),
                (delayed.q_block, 0),
            )
            if delayed.active:
                self._stage_update(
                    delayed.kv_block_pair_loop_idx,
                    delayed.kv_block_pair_loop_count,
                    delayed.q_valid_rows,
                    delayed.state_slot,
                )
            else:
                self.vector.initialize_empty(
                    delayed.q_valid_rows,
                    delayed.state_slot,
                )
            if delayed.is_last:
                self._finalize(
                    out_tile,
                    softmax_lse,
                    delayed.q_head,
                    delayed.q_begin,
                    delayed.q_end,
                    delayed.q_block,
                    delayed.q_valid_rows,
                    delayed.state_slot,
                )
        if tick >= PV_PIPELINE_LAG and tick - PV_PIPELINE_LAG < issued:
            delayed = pipeline.tap(PV_PIPELINE_LAG)
            if delayed.active:
                work = SparseWork(
                    batch=delayed.batch,
                    q_head=delayed.q_head,
                    q_block=delayed.q_block,
                    kv_used=delayed.kv_used,
                    sparse_kv_block_count=delayed.sparse_kv_block_count,
                    kv_block_pair_loop_idx=delayed.kv_block_pair_loop_idx,
                )
                self._stage_pv(
                    value,
                    sparse_indices,
                    block_table,
                    delayed.kv_head,
                    work,
                )

    @jit
    def _process_section(
        self,
        out: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_descale: Tensor,
        k_descale: Tensor,
        v_descale: Tensor,
        p_scale_value: float,
        sparse_indices: Tensor,
        sparse_seq_len: Tensor,
        metadata: Tensor,
        block_table: Tensor,
        seqused_kv: Tensor,
        cu_seqlens_q: Tensor,
        softmax_scale: float,
        atten_mask: Tensor | None,
        softmax_lse: Tensor | None,
        section_idx: int,
        pipeline,
        tick: int,
        issued: int,
        task_sequence: int,
        started: int,
    ):
        # 矩形 BN/Q 坐标只用于遍历；读取稀疏数据前必须过滤每 batch 的填充 Q 槽位。
        record = (
            METADATA_HEADER
            + section_idx * METADATA_SECTION_SIZE
            + self.block_idx * METADATA_CORE_FIELDS
        )
        if metadata[record] != 0:
            if started == 0:
                # 全部 section 共享一条连续任务流，只建立一次真实生产者。
                self.vector.initialize_empty(VECTOR_M, 0)
            started = 1
            task_start = dtypes.int64(
                metadata[record + 1]
            ) * self.q_capacity + dtypes.int64(metadata[record + 2])
            task_end = dtypes.int64(
                metadata[record + 4]
            ) * self.q_capacity + dtypes.int64(metadata[record + 5])
            for task in range(task_start, task_end):
                bn = task // self.q_capacity
                batch = bn // self.n1
                q_head = bn % self.n1
                q_block = task % self.q_capacity
                kv_head = q_head // self.group_size
                q_begin = dtypes.int64(cu_seqlens_q[batch])
                q_end = dtypes.int64(cu_seqlens_q[batch + 1])
                q_rows = q_end - q_begin
                if q_block * TILE_M < q_rows:
                    sparse_kv_block_count = dtypes.int64(
                        sparse_seq_len[batch, q_head, q_block],
                    )
                    kv_used = dtypes.int64(seqused_kv[batch])
                    kv_block_pair_loop_count = (sparse_kv_block_count + 1) // 2
                    query_tile = tile_slice(
                        query[q_head, q_begin:q_end, None],
                        (TILE_M, HEAD_DIM),
                        (q_block, 0),
                    )
                    q_tile_rows = dtypes.int32(query_tile.shape[0])
                    q_valid_rows = q_tile_rows - dtypes.int32(
                        self.subblock_idx * VECTOR_M,
                    )
                    q_valid_rows = q_valid_rows if q_valid_rows > 0 else dtypes.int32(0)
                    q_valid_rows = (
                        dtypes.int32(VECTOR_M)
                        if q_valid_rows > VECTOR_M
                        else q_valid_rows
                    )
                    state_slot = task_sequence % 3
                    pipeline_issue_count = (
                        kv_block_pair_loop_count if kv_block_pair_loop_count > 0 else 1
                    )
                    if kv_block_pair_loop_count > 0:
                        self.matmul.load_query(query_tile)
                    for kv_block_pair_loop_idx in range(pipeline_issue_count):
                        active = kv_block_pair_loop_idx < kv_block_pair_loop_count
                        if active:
                            work = SparseWork(
                                batch=batch,
                                q_head=q_head,
                                q_block=q_block,
                                kv_used=kv_used,
                                sparse_kv_block_count=sparse_kv_block_count,
                                kv_block_pair_loop_idx=dtypes.int64(
                                    kv_block_pair_loop_idx
                                ),
                            )
                            self._stage_qk(
                                key,
                                sparse_indices,
                                block_table,
                                kv_head,
                                work,
                            )
                        pipeline.push(
                            active=(dtypes.int64(1) if active else dtypes.int64(0)),
                            batch=batch,
                            q_head=q_head,
                            q_block=q_block,
                            kv_head=kv_head,
                            kv_used=kv_used,
                            q_rows=q_rows,
                            q_begin=q_begin,
                            q_end=q_end,
                            sparse_kv_block_count=sparse_kv_block_count,
                            kv_block_pair_loop_idx=dtypes.int64(
                                kv_block_pair_loop_idx,
                            ),
                            kv_block_pair_loop_count=kv_block_pair_loop_count,
                            q_valid_rows=dtypes.int64(q_valid_rows),
                            state_slot=state_slot,
                            is_last=(
                                dtypes.int64(1)
                                if kv_block_pair_loop_idx + 1 == pipeline_issue_count
                                else dtypes.int64(0)
                            ),
                        )
                        self._run_delayed_stages(
                            pipeline,
                            tick,
                            issued,
                            out,
                            value,
                            q_descale,
                            k_descale,
                            sparse_indices,
                            block_table,
                            softmax_lse,
                            softmax_scale,
                            p_scale_value,
                            atten_mask,
                        )
                        if active and kv_block_pair_loop_idx == 0:
                            # 必须在 lag3 读取旧 Q 的同槽 descale 后才覆盖。
                            v_scale = tile_slice(v_descale, (1,), (kv_head,))
                            self.vector.remember_v_scale(v_scale, state_slot)
                        pipeline.advance()
                        tick = tick + 1
                        issued = issued + 1
                    task_sequence = task_sequence + 1
        return tick, issued, task_sequence, started


class QuantBlockSparseAttnLauncher:
    def __init__(
        self,
        mask_mode=0,
        block_dim=1,
        return_softmax_lse=False,
        *,
        layout_q="TND",
        shared_l0a=False,
    ):
        self.mask_mode = mask_mode
        self.block_dim = block_dim
        self.return_softmax_lse = return_softmax_lse
        self.layout_q = layout_q
        self.shared_l0a = shared_l0a

    @host
    def launch(
        self,
        out: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_descale: Tensor,
        k_descale: Tensor,
        v_descale: Tensor,
        p_scale: Tensor | None,
        sparse_indices: Tensor,
        sparse_seq_len: Tensor,
        metadata: Tensor,
        block_table: Tensor,
        seqused_kv: Tensor,
        cu_seqlens_q: Tensor,
        softmax_scale: float,
        atten_mask: Tensor | None = None,
        softmax_lse: Tensor | None = None,
    ):
        # Q/scale 统一按 head 取 GM 视图；只调整视图轴，不重排或重新量化。
        # TND 的 token stride 为 N1*128 / N1，NTD 为 128 / 1。
        if const_expr(self.layout_q == "TND"):
            query = permute(query, (1, 0, 2))
            q_descale = permute(q_descale, (1, 0))
        # 输出始终保持公开 TND 布局，不随 Query layout 改变。
        out = permute(out, (1, 0, 2))
        # k_descale 保留公开四维 stride，仅在设备选页后将连续子块转成一维视图。
        op = QuantBlockSparseAttnKernel(
            query.shape[0],
            sparse_seq_len.shape[2],
            query.shape[0] // key.shape[1],
            self.mask_mode,
            self.return_softmax_lse,
            self.shared_l0a,
        )
        op[self.block_dim](
            out,
            query,
            key,
            value,
            q_descale,
            k_descale,
            v_descale,
            p_scale,
            sparse_indices,
            sparse_seq_len,
            metadata,
            block_table,
            seqused_kv,
            cu_seqlens_q,
            softmax_scale,
            atten_mask,
            softmax_lse,
        )


# ---------------------------------------------------------------------------
# Torch 接口：验证当前支持范围、分配输出并启动 DSL，不执行参考数学计算。
# ---------------------------------------------------------------------------


class KernelSpec(NamedTuple):
    mask_mode: int
    block_dim: int
    return_lse: bool
    layout_q: str
    shared_l0a: bool
    total_q: int
    n1: int
    n2: int
    pages: int
    batch: int
    q_capacity: int
    sparse_capacity: int
    table_capacity: int
    interleaved_kv: bool
    has_p_scale: bool


_COMPILED_KERNELS = {}
_COMPILED_KERNEL_LOCK = Lock()


def get_kernel(spec: KernelSpec):
    """Cache a compiled QBSA provider by its complete static tensor contract."""
    with _COMPILED_KERNEL_LOCK:
        compiled = _COMPILED_KERNELS.get(spec)
        if compiled is not None:
            return compiled
        compiled = _compile_kernel(spec)
        _COMPILED_KERNELS[spec] = compiled
        return compiled


def _compile_kernel(spec: KernelSpec):
    (
        mask_mode,
        block_dim,
        return_lse,
        layout_q,
        shared_l0a,
        total_q,
        n1,
        n2,
        pages,
        batch,
        q_capacity,
        sparse_capacity,
        table_capacity,
        interleaved_kv,
        has_p_scale,
    ) = spec
    fp8 = dtypes.float8_e4m3fn
    fp32 = dtypes.float32
    int32 = dtypes.int32
    query_shape = (
        (total_q, n1, HEAD_DIM) if layout_q == "TND" else (n1, total_q, HEAD_DIM)
    )
    q_scale_shape = query_shape[:2]
    kv_shape = (pages, n2, SPARSE_BLOCK, HEAD_DIM)
    k_scale_shape = (pages, n2, SPARSE_BLOCK, 1)
    if interleaved_kv:
        kv_page_bytes = n2 * SPARSE_BLOCK * HEAD_DIM
        page_bytes = 2 * kv_page_bytes + n2 * SPARSE_BLOCK * 4
        kv_stride = (page_bytes, SPARSE_BLOCK * HEAD_DIM, HEAD_DIM, 1)
        k_scale_stride = (page_bytes // 4, SPARSE_BLOCK, 1, 1)
        key_spec = TensorSpec(kv_shape, fp8, stride=kv_stride)
        value_spec = TensorSpec(kv_shape, fp8, stride=kv_stride)
        k_scale_spec = TensorSpec(k_scale_shape, fp32, stride=k_scale_stride)
    else:
        key_spec = TensorSpec(kv_shape, fp8)
        value_spec = TensorSpec(kv_shape, fp8)
        k_scale_spec = TensorSpec(k_scale_shape, fp32)

    launcher = QuantBlockSparseAttnLauncher(
        mask_mode,
        block_dim,
        return_lse,
        layout_q=layout_q,
        shared_l0a=shared_l0a,
    )
    return cannbotdsl.compile(
        launcher.launch,
        TensorSpec((total_q, n1, HEAD_DIM), dtypes.bfloat16),
        TensorSpec(query_shape, fp8),
        key_spec,
        value_spec,
        TensorSpec(q_scale_shape, fp32),
        k_scale_spec,
        TensorSpec((n2,), fp32),
        TensorSpec((1,), fp32) if has_p_scale else None,
        TensorSpec((batch, n1, q_capacity, sparse_capacity), int32),
        TensorSpec((batch, n1, q_capacity), int32),
        TensorSpec((_metadata_capacity(batch, n1),), int32),
        TensorSpec((batch, table_capacity), int32),
        TensorSpec((batch,), int32),
        TensorSpec((batch + 1,), int32),
        float,
        TensorSpec((2048, 2048), dtypes.uint8) if mask_mode == 3 else None,
        TensorSpec((n1, total_q), fp32) if return_lse else None,
    )


def _metadata_capacity(batch, n1):
    return METADATA_HEADER + batch * n1 * METADATA_SECTION_SIZE + METADATA_FD_SIZE


def _validate_metadata_contract(metadata, batch, n1):
    """Validate only the host-visible ABI without reading AICPU output data."""
    expected = _metadata_capacity(batch, n1)
    message = (
        f"Metadata must be a 1D int32 tensor with {expected} elements "
        "produced by quant_block_sparse_attn_metadata"
    )
    if not isinstance(metadata, torch.Tensor):
        raise ValueError(message)
    if (
        metadata.dtype != torch.int32
        or metadata.ndim != 1
        or metadata.numel() != expected
    ):
        raise ValueError(message)


def _launch_block_dim(device):
    """Use the AICore launch quota; inactive metadata slots exit in-kernel."""
    stream = torch.npu.current_stream(device)
    block_dim = int(get_platform_info(stream=stream).cube_core_num)
    if block_dim == 0:
        block_dim = int(torch.npu.get_device_properties(device).cube_core_num)
    if not 1 <= block_dim <= METADATA_AIC_SLOTS:
        raise RuntimeError(f"Unsupported Cube launch count: {block_dim}")
    return block_dim


def _validate_kv_storage(key, value, k_descale):
    """允许独立连续张量，或同一存储中无页间 padding 的 K/V/FP32 scale 交错页。"""
    if all(tensor.is_contiguous() for tensor in (key, value, k_descale)):
        return
    pages, n2, tokens, dim = key.shape
    kv_bytes = n2 * tokens * dim
    # k_descale is FP32, so each scale element occupies 4 bytes.
    page_bytes = 2 * kv_bytes + n2 * tokens * 4
    expected_strides = (
        (key, (page_bytes, tokens * dim, dim, 1)),
        (value, (page_bytes, tokens * dim, dim, 1)),
        (k_descale, (page_bytes // 4, tokens, 1, 1)),
    )
    for tensor, stride in expected_strides:
        if tensor.stride() != stride:
            raise NotImplementedError(
                "The page/head/token strides of interleaved KV are not supported"
            )
    storage = key.untyped_storage()
    storage_ptr = storage.data_ptr()
    if any(
        tensor.untyped_storage().data_ptr() != storage_ptr
        for tensor in (value, k_descale)
    ):
        raise NotImplementedError(
            "Interleaved K/V/k_descale must share the same storage"
        )
    key_ptr = key.data_ptr()
    if (
        value.data_ptr() - key_ptr != kv_bytes
        or k_descale.data_ptr() - key_ptr != 2 * kv_bytes
    ):
        raise NotImplementedError(
            "The relative byte offsets of interleaved K/V/k_descale are not supported"
        )
    if key_ptr - storage_ptr + pages * page_bytes > storage.nbytes():
        raise ValueError(
            "Interleaved KV storage is too small to cover all physical pages"
        )


def _validate_quant_config(
    layouts, quant_mode, mask_mode, sparse_q_block_size, sparse_kv_block_size
):
    if quant_mode != 1 or mask_mode not in (0, 3):
        raise NotImplementedError(
            "Only quant_mode=1 and mask_mode in {0, 3} are supported"
        )
    layout_q, layout_kv, layout_sparse_indices, layout_out = layouts
    if layout_q not in ("TND", "NTD") or (
        layout_kv,
        layout_sparse_indices,
        layout_out,
    ) != ("PA_BNBD", "B_N_Qb_Kb", "TND"):
        raise NotImplementedError(
            "Supported layouts are TND or NTD / PA_BNBD / B_N_Qb_Kb / TND"
        )
    if sparse_q_block_size != 128 or sparse_kv_block_size != 128:
        raise ValueError("Mode1 requires sparse_q_block_size=sparse_kv_block_size=128")


def _validate_qkv_shapes(query, key, value, layout_q):
    if query.ndim != 3 or query.shape[-1] != 128 or min(query.shape[:2]) < 1:
        raise NotImplementedError(
            "Query must be TND [Q,N1,128] or NTD [N1,Q,128], with Q>=1 and N1>=1"
        )
    total_q, n1 = query.shape[:2] if layout_q == "TND" else query.shape[:2][::-1]
    query_shape = (total_q, n1, 128) if layout_q == "TND" else (n1, total_q, 128)
    if key.ndim != 4 or key.shape[2:] != (128, 128) or value.shape != key.shape:
        raise NotImplementedError("Key/value must have shape [pages,N2,128,128]")
    n2 = key.shape[1]
    if not 1 <= n2 <= 8 or n1 % n2 != 0 or not 1 <= n1 // n2 <= 16:
        raise ValueError(
            "N2 must be in [1,8], N1 must be divisible by N2, and N1/N2 must be in [1,16]"
        )
    return total_q, n1, query_shape, n2


def _validate_sparse_inputs(metadata, block_table, sparse_indices, n1):
    for name, tensor, rank in (
        ("metadata", metadata, 1),
        ("block_table", block_table, 2),
        ("sparse_indices", sparse_indices, 4),
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.int32
            or tensor.ndim != rank
        ):
            raise ValueError(
                f"Input {name} must be a {rank}D tensor with dtype torch.int32"
            )
    batch, sparse_heads, q_capacity, capacity = sparse_indices.shape
    if not 1 <= batch <= 65536:
        raise ValueError("B must be in [1,65536]")
    if sparse_heads != n1 or q_capacity < 1 or capacity < 1:
        raise ValueError(
            "The sparse_indices tensor must be [B,N1,max_Qb,Kb], with max_Qb>=1 and Kb>=1"
        )
    return batch, q_capacity, capacity


def _validate_tensor_specs(tensors):
    for name, (tensor, dtype, shape) in tensors.items():
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != dtype
            or tensor.shape != shape
        ):
            raise ValueError(
                f"Input {name} must have dtype {dtype} and shape {tuple(shape)}"
            )


def quant_block_sparse_attn(
    query,
    key,
    value,
    q_descale,
    k_descale,
    v_descale,
    p_scale,
    sparse_indices,
    sparse_seq_len,
    atten_mask,
    softmax_scale,
    sparse_q_block_size,
    sparse_kv_block_size,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    block_table=None,
    metadata=None,
    layout_kv="PA_BNBD",
    layout_q="TND",
    layout_sparse_indices="B_N_Qb_Kb",
    layout_out="TND",
    quant_mode=1,
    mask_mode=3,
    return_softmax_lse=False,
):
    """返回 (BF16 attention_out, 可选 FP32 LSE [N1,T1])；不支持的场景启动前报错。"""
    _validate_quant_config(
        (layout_q, layout_kv, layout_sparse_indices, layout_out),
        quant_mode,
        mask_mode,
        sparse_q_block_size,
        sparse_kv_block_size,
    )
    if cu_seqlens_kv is not None or seqused_q is not None:
        raise ValueError("Reserved arguments cu_seqlens_kv and seqused_q must be None")
    total_q, n1, query_shape, n2 = _validate_qkv_shapes(query, key, value, layout_q)
    if mask_mode == 3:
        if not isinstance(atten_mask, torch.Tensor) or atten_mask.dtype != torch.uint8:
            raise ValueError("When mask_mode=3, atten_mask must have dtype torch.uint8")
        if atten_mask.shape != (2048, 2048):
            raise ValueError("When mask_mode=3, atten_mask must have shape [2048,2048]")
    if not math.isfinite(float(softmax_scale)):
        raise ValueError("The softmax_scale value must be finite")
    batch, q_capacity, capacity = _validate_sparse_inputs(
        metadata,
        block_table,
        sparse_indices,
        n1,
    )
    tensors = {
        "query": (query, torch.float8_e4m3fn, query_shape),
        "key": (key, torch.float8_e4m3fn, key.shape),
        "value": (value, torch.float8_e4m3fn, key.shape),
        "q_descale": (q_descale, torch.float32, query_shape[:2]),
        "k_descale": (k_descale, torch.float32, (*key.shape[:3], 1)),
        "v_descale": (v_descale, torch.float32, (n2,)),
        "cu_seqlens_q": (cu_seqlens_q, torch.int32, (batch + 1,)),
        "seqused_kv": (seqused_kv, torch.int32, (batch,)),
    }
    _validate_tensor_specs(tensors)
    if block_table.shape[0] != batch:
        raise ValueError("The batch dimension of block_table must match sparse_indices")
    if block_table.shape[1] < capacity:
        raise ValueError(
            "The mode1 block_table length must be at least the sparse capacity"
        )
    tensors["sparse_seq_len"] = (sparse_seq_len, torch.int32, (batch, n1, q_capacity))
    if (
        not isinstance(sparse_seq_len, torch.Tensor)
        or sparse_seq_len.dtype != torch.int32
        or sparse_seq_len.shape != (batch, n1, q_capacity)
    ):
        raise ValueError(
            "The sparse_seq_len tensor must have dtype torch.int32 and shape [B,N1,max_Qb]"
        )
    _validate_metadata_contract(metadata, batch, n1)
    block_dim = _launch_block_dim(query.device)
    # 不根据运行时 Tensor 内容选择编译特化。独立 K/P L0A 路径
    # 同时支持完整宏块和尾宏块，因此作为无 Host 回读的保守路径。
    shared_l0a = False
    if p_scale is not None and (
        not isinstance(p_scale, torch.Tensor)
        or p_scale.dtype != torch.float32
        or p_scale.shape != (1,)
    ):
        raise ValueError(
            "The p_scale argument must be None or a torch.float32 tensor with shape [1]"
        )
    non_kv_tensors = [
        entry[0]
        for name, entry in tensors.items()
        if name not in ("key", "value", "k_descale")
    ] + [metadata, block_table, sparse_indices]
    if p_scale is not None:
        non_kv_tensors.append(p_scale)
    if mask_mode == 3:
        non_kv_tensors.append(atten_mask)
    if query.device.type != "npu" or any(
        t.device != query.device for t in (*non_kv_tensors, key, value, k_descale)
    ):
        raise ValueError("All execution inputs must be on the same NPU device")
    if any(not t.is_contiguous() for t in non_kv_tensors):
        raise NotImplementedError("Only contiguous inputs are supported")
    _validate_kv_storage(key, value, k_descale)
    out = torch.empty(
        (total_q, n1, query.shape[-1]), dtype=torch.bfloat16, device=query.device
    )
    softmax_lse = None
    if return_softmax_lse:
        softmax_lse = torch.empty(
            (n1, total_q), dtype=torch.float32, device=query.device
        )
    program = get_kernel(
        KernelSpec(
            mask_mode,
            block_dim,
            return_softmax_lse,
            layout_q,
            shared_l0a,
            total_q,
            n1,
            n2,
            key.shape[0],
            batch,
            q_capacity,
            capacity,
            block_table.shape[1],
            not key.is_contiguous(),
            p_scale is not None,
        )
    )
    args = (
        out,
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        p_scale,
        sparse_indices,
        sparse_seq_len,
        metadata,
        block_table,
        seqused_kv,
        cu_seqlens_q,
        float(softmax_scale),
        atten_mask if mask_mode == 3 else None,
        softmax_lse,
    )
    program(*(value for value in args if value is not None))
    return out, softmax_lse
