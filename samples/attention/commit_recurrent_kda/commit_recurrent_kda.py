# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# fmt: off

"""独立执行 Mega KDA 已接受前缀的检查点回放。"""

# 显式传递 Tensor 操作数，保持内核 ABI 和 DSL 的词法顺序。
# 硬件阶段以及同步、搬运顺序采用固定展开；
# 提取辅助函数会改变 DSL 降级过程或已测得的内核性能。

from __future__ import annotations

import functools
import sys
import threading
from types import ModuleType
import cannbotdsl
import torch
from typing import NamedTuple

from cannbotdsl import make_layout, MemLoc, Tensor, TensorSpec, dtypes
import torch._dynamo as torch_dynamo
from cannbotdsl.aot import export
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.reg import (
    UnpackMode,
    update_mask,
    vcast,
    vload,
    vload_broadcast,
    vload_unpack,
    vmadd,
    vmul,
    vstore,
)
from cannbotdsl.ops.sync import onchip_rewind
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import from_tensor, idx2crd, tile_slice

# 模型与向量计算维度
D = 128
TILE = 128
VL = 64

# 状态更新与批次调度
KDA_STATE_UPDATE_UNROLL = 4
SUPPORTED_BLOCK_NUMS = (8, 24, 28, 32)
SMALL_BATCH_HEAD_LIMIT = 16
MEDIUM_BATCH_HEAD_LIMIT = 32
SMALL_BATCH_ROW_BLOCK = 32

_compiled_commit_kernel = None


def _device_block_num(ref: torch.Tensor) -> int:
    """返回当前流的有效 AIC 核数，最多为 32。"""
    if ref.device.type not in {"npu", "privateuseone"}:
        raise ValueError("commit_recurrent_kda requires NPU inputs")
    device_index = ref.device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    stream = torch.npu.current_stream(device_index)
    properties = cannbotdsl.get_platform_info(stream=stream)
    try:
        cube_core_num = int(properties.cube_core_num)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"NPU {device_index} does not expose a valid AIC count"
        ) from error
    if cube_core_num <= 0:
        raise RuntimeError(
            f"NPU {device_index} reports invalid cube_core_num={cube_core_num}"
        )
    return min(cube_core_num, 32)


def _resolve_block_num(ref: torch.Tensor) -> int:
    """在当前流限制内选择支持的最大核数配额。"""
    available = _device_block_num(ref)
    choices = tuple(value for value in SUPPORTED_BLOCK_NUMS if value <= available)
    if not choices:
        raise ValueError(f"device exposes {available} AIC cores; at least 8 are required")
    return choices[-1]


def _commit_launch_blocks(batch_size: int, num_heads: int, aic_blocks: int) -> int:
    """使用互不重叠的检查点行块，为两个 AIV 子块提供足够的并行任务。"""
    batch_heads = batch_size * num_heads
    if batch_heads <= SMALL_BATCH_HEAD_LIMIT:
        row_block = SMALL_BATCH_ROW_BLOCK
    elif batch_heads <= MEDIUM_BATCH_HEAD_LIMIT:
        row_block = VL
    else:
        row_block = TILE
    total_items = batch_heads * (TILE // row_block)
    return min(total_items, 2 * aic_blocks)


def _commit_row_schedule(batch_size: int, num_heads: int, aic_blocks: int):
    """描述内核中互不重叠的状态行分配，供主机侧测试使用。"""
    batch_heads = batch_size * num_heads
    lane_count = _commit_launch_blocks(batch_size, num_heads, aic_blocks)
    schedule = [[] for _ in range(lane_count)]
    if batch_heads <= MEDIUM_BATCH_HEAD_LIMIT:
        row_block = (
            SMALL_BATCH_ROW_BLOCK if batch_heads <= SMALL_BATCH_HEAD_LIMIT else VL
        )
        row_blocks = TILE // row_block
        total_items = batch_heads * row_blocks
        for lane in range(lane_count):
            for work_item in range(lane, total_items, lane_count):
                head_item, row_index = divmod(work_item, row_blocks)
                schedule[lane].append((head_item, row_index * row_block, row_block))
        return tuple(tuple(lane) for lane in schedule)

    total_half_rows = 2 * batch_heads
    base_half_rows, extra_cores = divmod(total_half_rows, lane_count)
    base_full_heads = base_half_rows // 2
    extra_full_heads = (base_half_rows + 1) // 2
    extra_full_delta = extra_full_heads - base_full_heads
    full_heads = lane_count * base_full_heads + extra_cores * extra_full_delta
    for lane in range(lane_count):
        full_head_count = extra_full_heads if lane < extra_cores else base_full_heads
        prefix_extra = min(lane, extra_cores)
        full_head_start = lane * base_full_heads + prefix_extra * extra_full_delta
        for full_offset in range(full_head_count):
            schedule[lane].append((full_head_start + full_offset, 0, TILE))
        if base_half_rows % 2 == 0 and lane < extra_cores:
            schedule[lane].append((full_heads + lane // 2, (lane % 2) * VL, VL))
        if base_half_rows % 2 == 1 and lane >= extra_cores:
            half_rank = lane - extra_cores
            schedule[lane].append(
                (full_heads + half_rank // 2, (half_rank % 2) * VL, VL)
            )
    return tuple(tuple(lane) for lane in schedule)


class _CheckpointPool(NamedTuple):
    data: Tensor
    indices: Tensor
    stride0: int


@jit
def _checkpoint_head(pool, batch, head):
    slot = dtypes.int64(pool.indices[batch])
    begin = slot * pool.stride0 + head * TILE * TILE
    return pool.data[begin:begin + TILE * TILE,].view(TILE, TILE)



class ReplayPool(NamedTuple):
    """将 PA 存储展平，并保留独立的元素步长和逻辑 batch 映射。"""
    data: Tensor
    indices: Tensor
    stride0: int
    heads: int
    capacity: int


@jit
def replay_head(pool, batch, head):
    slot = dtypes.int64(pool.indices[batch])
    begin = slot * pool.stride0 + head * pool.capacity * TILE
    return pool.data[begin:begin + pool.capacity * TILE,].view(pool.capacity, TILE)


@jit
def replay_batches(pool, begin, end):
    return ReplayPool(pool.data, pool.indices[begin:end,], pool.stride0, pool.heads, pool.capacity)


class CommitRecurrentKDAStage:
    """为每个 batch/head 检查点回放已接受的 token 序列。"""

    def __init__(self):
        self.state_current = Buffer(MemLoc.UB, dtypes.float32, make_layout((TILE, TILE)))
        # 当前 head 回放已接受 token 序列时，配对调度将下一个状态块保留在 UB 中。
        # 状态块形状与 B、N、S 无关；
        # 只有调度器依据运行时形状分配任务。
        self.state_next = Buffer(MemLoc.UB, dtypes.float32, make_layout((TILE, TILE)))
        self.replay_u = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((1, TILE)), depth=2)
        self.replay_k = RingBuffer(MemLoc.UB, dtypes.float16, make_layout((1, TILE)), depth=2)
        self.replay_decay = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((1, TILE)), depth=2)
        self.u_fp32 = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, TILE)))

    @jit
    def prefetch_record(
        self,
        replay_u_gm: ReplayPool,
        replay_k_gm: ReplayPool,
        replay_decay_gm: ReplayPool,
        batch_index,
        token,
        value_head,
        row_index,
        active_rows: int,
    ):
        replay_u = self.replay_u.next_write()
        if active_rows < TILE:
            replay_u = from_tensor(replay_u, offset=0, layout=make_layout((1, active_rows)))
        mem_copy(
            replay_u,
            tile_slice(
                replay_head(replay_u_gm, batch_index, value_head),
                (1, active_rows),
                (token, row_index),
            ),
        )
        mem_copy(self.replay_k.next_write(), tile_slice(replay_head(replay_k_gm, batch_index, value_head), (1, TILE), (token, 0)))
        mem_copy(self.replay_decay.next_write(), tile_slice(replay_head(replay_decay_gm, batch_index, value_head), (1, TILE), (token, 0)))

    @jit
    def apply_current_record(self, state, active_rows: int):
        replay_u = self.replay_u.next_read()
        replay_k = self.replay_k.next_read()
        replay_decay = self.replay_decay.next_read()
        # 保留独立的 U 暂存阶段和深度为 2 的记录流水线。
        # PA 状态接口中的 U 已经是 FP32，本阶段直接复制，不做量化。
        with vf(mode="simd"):
            for segment in range((active_rows + VL - 1) // VL):
                offset = segment * VL
                segment_rows = active_rows - offset
                segment_mask, _ = update_mask(segment_rows, elem_bits=32)
                update = vload(replay_u, offset)
                vstore(self.u_fp32, offset, update, segment_mask)

        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            key_pre_packed = vload_unpack(
                replay_k, 0, unpack_mode=UnpackMode.B16_TO_B32
            )
            key_post_packed = vload_unpack(
                replay_k, VL, unpack_mode=UnpackMode.B16_TO_B32
            )
            key_pre = vcast(key_pre_packed, dtypes.float32, mask=full)
            key_post = vcast(key_post_packed, dtypes.float32, mask=full)
            decay_pre = vload(replay_decay, 0)
            decay_post = vload(replay_decay, VL)
            for row in dsl_range(0, active_rows, 1, unroll=KDA_STATE_UPDATE_UNROLL):
                offset = row * TILE
                u_row = vload_broadcast(self.u_fp32, row)
                state_pre = vload(state, offset)
                state_post = vload(state, offset + VL)
                state_pre = vmul(state_pre, decay_pre, mask=full)
                state_post = vmul(state_post, decay_post, mask=full)
                state_pre = vmadd(u_row, key_pre, state_pre, mask=full)
                state_post = vmadd(u_row, key_post, state_post, mask=full)
                vstore(state, offset, state_pre, full)
                vstore(state, offset + VL, state_post, full)

    @jit
    def execute_item(
        self,
        state_gm,
        replay_u_gm: ReplayPool,
        replay_k_gm: ReplayPool,
        replay_decay_gm: ReplayPool,
        accepted_gm: Tensor,
        item,
        batch_size,
        num_heads,
        sequence_length,
        row_index,
        active_rows: int,
    ):
        batch_index, value_head = idx2crd(item, [batch_size, num_heads])
        accepted = dtypes.int64(accepted_gm[batch_index])
        if accepted > 0:
            state = self.state_current
            if active_rows < TILE:
                state = from_tensor(self.state_current, offset=0, layout=make_layout((active_rows, TILE)))
            state_tile = tile_slice(
                _checkpoint_head(state_gm, batch_index, value_head),
                (active_rows, TILE),
                (row_index, 0),
            )
            mem_copy(state, state_tile)
            self.prefetch_record(
                replay_u_gm,
                replay_k_gm,
                replay_decay_gm,
                batch_index,
                0,
                value_head,
                row_index,
                active_rows,
            )
            for token in range(sequence_length):
                if token < accepted:
                    if token + 1 < accepted:
                        self.prefetch_record(
                            replay_u_gm,
                            replay_k_gm,
                            replay_decay_gm,
                            batch_index,
                            token + 1,
                            value_head,
                            row_index,
                            active_rows,
                        )
                    self.apply_current_record(state, active_rows)
            mem_copy(state_tile, state)

    @jit
    def execute_pair(
        self,
        state_gm,
        replay_u_gm: ReplayPool,
        replay_k_gm: ReplayPool,
        replay_decay_gm: ReplayPool,
        accepted_gm: Tensor,
        current_item,
        next_item,
        batch_size,
        num_heads,
        sequence_length,
        next_row_index,
        next_active_rows: int,
    ):
        """通过同一个 UB 双缓冲依次处理两个互不重叠的状态块。"""
        current_batch, current_head = idx2crd(current_item, [batch_size, num_heads])
        next_batch, next_head = idx2crd(next_item, [batch_size, num_heads])
        current_accepted = dtypes.int64(accepted_gm[current_batch])
        next_accepted = dtypes.int64(accepted_gm[next_batch])
        current_state_tile = tile_slice(
            _checkpoint_head(state_gm, current_batch, current_head),
            (TILE, TILE),
            (0, 0),
        )
        next_state_tile = tile_slice(
            _checkpoint_head(state_gm, next_batch, next_head),
            (next_active_rows, TILE),
            (next_row_index, 0),
        )
        next_state = self.state_next
        if next_active_rows < TILE:
            next_state = from_tensor(self.state_next, offset=0, layout=make_layout((next_active_rows, TILE)))

        if current_accepted > 0:
            mem_copy(self.state_current, current_state_tile)
            self.prefetch_record(
                replay_u_gm,
                replay_k_gm,
                replay_decay_gm,
                current_batch,
                0,
                current_head,
                0,
                TILE,
            )
            for token in range(sequence_length):
                if token < current_accepted:
                    if token + 1 < current_accepted:
                        self.prefetch_record(
                            replay_u_gm,
                            replay_k_gm,
                            replay_decay_gm,
                            current_batch,
                            token + 1,
                            current_head,
                            0,
                            TILE,
                        )
                    elif next_accepted > 0:
                        mem_copy(next_state, next_state_tile)
                        self.prefetch_record(
                            replay_u_gm,
                            replay_k_gm,
                            replay_decay_gm,
                            next_batch,
                            0,
                            next_head,
                            next_row_index,
                            next_active_rows,
                        )
                    self.apply_current_record(self.state_current, TILE)
            mem_copy(current_state_tile, self.state_current)

        if next_accepted > 0:
            if current_accepted == 0:
                mem_copy(next_state, next_state_tile)
                self.prefetch_record(
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    next_batch,
                    0,
                    next_head,
                    next_row_index,
                    next_active_rows,
                )
            for token in range(sequence_length):
                if token < next_accepted:
                    if token + 1 < next_accepted:
                        self.prefetch_record(
                            replay_u_gm,
                            replay_k_gm,
                            replay_decay_gm,
                            next_batch,
                            token + 1,
                            next_head,
                            next_row_index,
                            next_active_rows,
                        )
                    self.apply_current_record(next_state, next_active_rows)
            mem_copy(next_state_tile, next_state)

    @jit
    def execute_row_blocks(
        self,
        state_gm,
        replay_u_gm: ReplayPool,
        replay_k_gm: ReplayPool,
        replay_decay_gm: ReplayPool,
        accepted_gm: Tensor,
        row_block: int,
    ):
        batch_size = accepted_gm.shape[0]
        num_heads = replay_u_gm.heads
        sequence_length = replay_u_gm.capacity
        row_blocks = TILE // row_block
        total_items = batch_size * num_heads * row_blocks
        block_idx = get_block_idx()
        aiv_block_count = get_block_num()
        for work_item in range(block_idx, total_items, aiv_block_count):
            item = work_item // row_blocks
            row_index = work_item - item * row_blocks
            self.execute_item(
                state_gm,
                replay_u_gm,
                replay_k_gm,
                replay_decay_gm,
                accepted_gm,
                item,
                batch_size,
                num_heads,
                sequence_length,
                row_index,
                row_block,
            )

    @jit
    def execute_minimal_split(
        self,
        state_gm,
        replay_u_gm: ReplayPool,
        replay_k_gm: ReplayPool,
        replay_decay_gm: ReplayPool,
        accepted_gm: Tensor,
    ):
        batch_size = accepted_gm.shape[0]
        num_heads = replay_u_gm.heads
        sequence_length = replay_u_gm.capacity
        block_idx = get_block_idx()
        aiv_block_count = get_block_num()
        batch_heads = batch_size * num_heads
        total_half_rows = 2 * batch_heads
        base_half_rows = total_half_rows // aiv_block_count
        extra_cores = total_half_rows - base_half_rows * aiv_block_count
        base_full_heads = base_half_rows // 2
        extra_full_heads = (base_half_rows + 1) // 2
        extra_full_delta = extra_full_heads - base_full_heads
        full_head_count = base_full_heads
        if block_idx < extra_cores:
            full_head_count = extra_full_heads
        prefix_extra = block_idx if block_idx < extra_cores else extra_cores
        full_head_start = block_idx * base_full_heads + prefix_extra * extra_full_delta

        for full_offset in range(full_head_count):
            self.execute_item(
                state_gm,
                replay_u_gm,
                replay_k_gm,
                replay_decay_gm,
                accepted_gm,
                full_head_start + full_offset,
                batch_size,
                num_heads,
                sequence_length,
                0,
                TILE,
            )

        full_heads = aiv_block_count * base_full_heads + extra_cores * extra_full_delta
        if base_half_rows % 2 == 0 and block_idx < extra_cores:
            self.execute_item(
                state_gm,
                replay_u_gm,
                replay_k_gm,
                replay_decay_gm,
                accepted_gm,
                full_heads + block_idx // 2,
                batch_size,
                num_heads,
                sequence_length,
                block_idx % 2,
                VL,
            )
        if base_half_rows % 2 == 1 and block_idx >= extra_cores:
            half_rank = block_idx - extra_cores
            self.execute_item(
                state_gm,
                replay_u_gm,
                replay_k_gm,
                replay_decay_gm,
                accepted_gm,
                full_heads + half_rank // 2,
                batch_size,
                num_heads,
                sequence_length,
                half_rank % 2,
                VL,
            )

    @jit
    def execute_paired(
        self,
        state_gm,
        replay_u_gm: ReplayPool,
        replay_k_gm: ReplayPool,
        replay_decay_gm: ReplayPool,
        accepted_gm: Tensor,
    ):
        """head 数较大且为偶数时，组合完整 head 与半个 head 的任务。

        此映射推广自原有的 B16/N6 调度：AIV 并行度较低时，每个执行单元
        处理多对完整 head；并行度较高时，每个执行单元处理一个完整 head，
        再处理第二个 head 的一半。
        """
        batch_size = accepted_gm.shape[0]
        num_heads = replay_u_gm.heads
        sequence_length = replay_u_gm.capacity
        batch_heads = batch_size * num_heads
        block_idx = get_block_idx()
        aiv_block_count = get_block_num()

        # 当启动网格为每个 head 分配一个执行单元时，不再额外构造第二个任务。
        # 这是动态 B*N 较小时的尾部处理路径。
        if aiv_block_count >= batch_heads:
            if block_idx < batch_heads:
                self.execute_item(
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    accepted_gm,
                    block_idx,
                    batch_size,
                    num_heads,
                    sequence_length,
                    0,
                    TILE,
                )
        elif aiv_block_count <= batch_heads // 2:
            # 每个执行单元依次处理一对或多对完整 head。
            for pair_index in range(block_idx, batch_heads // 2, aiv_block_count):
                self.execute_pair(
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    accepted_gm,
                    2 * pair_index,
                    2 * pair_index + 1,
                    batch_size,
                    num_heads,
                    sequence_length,
                    0,
                    TILE,
                )
        elif 3 * aiv_block_count <= 2 * batch_heads:
            # 此处采用互不重叠的整行/半行拆分调度。前 pair_lanes 个执行单元
            # 各处理两个完整 head；其余执行单元各处理一个完整 head，
            # 再通过 state_next 处理剩余 head 的一半。
            pair_lanes = 2 * batch_heads - 3 * aiv_block_count
            if block_idx < pair_lanes:
                self.execute_pair(
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    accepted_gm,
                    2 * block_idx,
                    2 * block_idx + 1,
                    batch_size,
                    num_heads,
                    sequence_length,
                    0,
                    TILE,
                )
            else:
                split_rank = block_idx - pair_lanes
                remaining_lanes = aiv_block_count - pair_lanes
                current_start = 2 * pair_lanes
                self.execute_pair(
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    accepted_gm,
                    current_start + split_rank,
                    current_start + remaining_lanes + split_rank // 2,
                    batch_size,
                    num_heads,
                    sequence_length,
                    split_rank % 2,
                    VL,
                )
        else:
            # 只剩少量尾部任务时，配对处理完整 head，避免拆分状态块。
            # 这样既能保证任务不会越界，
            # 也能在 B=7、N=6 等形状下继续使用 UB 双缓冲。
            pair_lanes = batch_heads - aiv_block_count
            if block_idx < pair_lanes:
                self.execute_pair(
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    accepted_gm,
                    2 * block_idx,
                    2 * block_idx + 1,
                    batch_size,
                    num_heads,
                    sequence_length,
                    0,
                    TILE,
                )
            else:
                self.execute_item(
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    accepted_gm,
                    2 * pair_lanes + block_idx - pair_lanes,
                    batch_size,
                    num_heads,
                    sequence_length,
                    0,
                    TILE,
                )

    @jit
    def execute(
        self,
        state_gm,
        replay_u_gm: ReplayPool,
        replay_k_gm: ReplayPool,
        replay_decay_gm: ReplayPool,
        accepted_gm: Tensor,
    ):
        batch_heads = accepted_gm.shape[0] * replay_u_gm.heads
        onchip_rewind(reset_sync_id=True)
        if batch_heads <= SMALL_BATCH_HEAD_LIMIT:
            self.execute_row_blocks(
                state_gm,
                replay_u_gm,
                replay_k_gm,
                replay_decay_gm,
                accepted_gm,
                SMALL_BATCH_ROW_BLOCK,
            )
        elif batch_heads <= MEDIUM_BATCH_HEAD_LIMIT:
            self.execute_row_blocks(
                state_gm, replay_u_gm, replay_k_gm, replay_decay_gm, accepted_gm, VL
            )
        else:
            if batch_heads % 2 == 0:
                self.execute_paired(
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    accepted_gm,
                )
            else:
                self.execute_minimal_split(
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    accepted_gm,
                )


class CommitRecurrentKDAKernel:
    @kernel
    def kernel(self,checkpoint,replay_u,replay_k,replay_decay,accepted,indices,
        checkpoint_stride0,u_stride0,k_stride0,decay_stride0,heads,record_s):
        state=_CheckpointPool(checkpoint,indices,checkpoint_stride0)
        u=ReplayPool(replay_u,indices,u_stride0,heads,record_s)
        rk=ReplayPool(replay_k,indices,k_stride0,heads,record_s)
        rd=ReplayPool(replay_decay,indices,decay_stride0,heads,record_s)
        stage=CommitRecurrentKDAStage()
        stage.execute(state,u,rk,rd,accepted)

    @host
    def run(self,checkpoint,replay_u,replay_k,replay_decay,accepted,indices,
        checkpoint_stride0,u_stride0,k_stride0,decay_stride0,heads,record_s,launch_blocks):
        self.kernel[launch_blocks](checkpoint,replay_u,replay_k,replay_decay,accepted,indices,
            checkpoint_stride0,u_stride0,k_stride0,decay_stride0,heads,record_s)

def _commit_specification():
    cp=cannbotdsl.Dim('checkpoint_span',min=1)
    us=cannbotdsl.Dim('replay_u_span',min=1)
    ks=cannbotdsl.Dim('replay_k_span',min=1)
    ds=cannbotdsl.Dim('replay_decay_span',min=1)
    b=cannbotdsl.Dim('batch',min=1)
    return (TensorSpec((cp,),dtypes.float32),TensorSpec((us,),dtypes.float32),
        TensorSpec((ks,),dtypes.float16),TensorSpec((ds,),dtypes.float32),
        TensorSpec((b,),dtypes.int32),TensorSpec((b,),dtypes.int32),
        # checkpoint、U、K、decay 的 PA 步长。
        dtypes.int64, dtypes.int64, dtypes.int64, dtypes.int64,
        # head 数、记录长度 S、启动核数。
        dtypes.int64, dtypes.int64, dtypes.int64)

def _compile_commit_kernel():
    return cannbotdsl.compile(CommitRecurrentKDAKernel().run, *_commit_specification())


def _get_compiled_commit_kernel():
    global _compiled_commit_kernel
    if _compiled_commit_kernel is None:
        _compiled_commit_kernel=_compile_commit_kernel()
    return _compiled_commit_kernel


def _shared_kda_call_state():
    """在独立加载的 KDA 模块之间共享调用锁和事件状态。"""
    key = "_cannbotdsl_kda_call_state"
    state = sys.modules.get(key)
    if state is None:
        candidate = ModuleType(key)
        candidate.call_lock = threading.RLock()
        candidate.pending = {"event": None, "keep": None}
        state = sys.modules.setdefault(key, candidate)
    return state


_kda_call_state = _shared_kda_call_state()
_call_lock = _kda_call_state.call_lock
_pending = _kda_call_state.pending


class InputValidation:
    """负责参数和存储约束校验。"""

    @staticmethod
    def require(condition, message):
        if isinstance(condition, bool):
            if not condition:
                raise ValueError(message)
        else:
            torch._check(condition, lambda: message)

    @staticmethod
    def tensor_metadata(name, tensor, shape, dtype, device):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'{name} must be a torch.Tensor')
        shape_label = {
            'hidden_states': '[B,seq_len,H] or [total_tokens,H]',
            'qkv_projection_weight': '[3*P,H]',
            'decay_projection_a_weight': '[D,H]',
            'decay_projection_b_weight': '[P,D]',
            'beta_projection_weight': '[N,H]',
            'output_gate_projection_weight': '[P,H]',
            'output_norm_weight': '[D]',
            'output_projection_weight': '[H,P]',
            'state_indices': '[B]',
            'conv_num_accepted_tokens': '[B]',
            'num_accepted_tokens': '[B]',
            'A_log': '[N]',
            'dt_bias': '[N,D]',
            'cu_seqlens': '[B+1]',
            'conv1d_weight': '[4,3*P], [3,N,4,D], or [3,4,D]',
            'replay_u': '[block_num,N,S,D]',
            'replay_k': '[block_num,N,S,D]',
            'replay_decay': '[block_num,N,S,D]',
        }.get(name, str(shape))
        InputValidation.require(tensor.ndim == len(shape), f'{name} must have shape {shape_label}')
        for actual, expected in zip(tensor.shape, shape):
            if expected is not None:
                InputValidation.require(actual == expected, f'{name} must have shape {shape_label}')
        label = {torch.float16: 'FP16', torch.float32: 'FP32', torch.bfloat16: 'BF16', torch.int32: 'int32', torch.int64: 'int64'}[dtype]
        if name == 'hidden_states':
            InputValidation.require(tensor.dtype == torch.bfloat16 and tensor.is_contiguous(), 'hidden_states must be contiguous BF16')
        InputValidation.require(tensor.dtype == dtype, f'{name} must have dtype {label}')
        InputValidation.require(tensor.layout == torch.strided, f'{name} must have strided layout')
        InputValidation.require(tensor.is_contiguous(), f'{name} must be contiguous')
        InputValidation.require(tensor.device == device, f'{name} must share the input device')
        InputValidation.require(not tensor.is_neg() and (not tensor.is_conj()), f'{name} must resolve negative/conjugate view bits')

    @staticmethod
    def state_pool_metadata(name, tensor, shape, dtype, device):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'{name} must be a torch.Tensor')
        shape_label = '[block_num,L,3*P]' if name == 'conv_state' else '[block_num,N,D,D]' if name == 'checkpoint_state' else '[block_num,N,S,D]'
        InputValidation.require(tensor.ndim == len(shape), f'{name} must have shape {shape_label}')
        for actual, expected in zip(tensor.shape, shape):
            if expected is not None:
                InputValidation.require(actual == expected, f'{name} must have shape {shape_label}')
        InputValidation.require(tensor.dtype == dtype, f'{name} must have dtype {dtype}')
        InputValidation.require(tensor.layout == torch.strided and tensor.device == device, f'{name} must share input device and strided layout')
        expected = 1
        for axis in range(tensor.ndim - 1, 0, -1):
            InputValidation.require(tensor.stride(axis) == expected, f'{name} only axis 0 may be non-contiguous')
            expected = expected * tensor.shape[axis]
        InputValidation.require(tensor.stride(0) >= expected, f'{name} axis-0 stride must not overlap paged-state blocks')
        InputValidation.require(not tensor.is_neg() and (not tensor.is_conj()), f'{name} must resolve view bits')

    @staticmethod
    def _records(a,batch,heads,device):
        u=a['replay_u']
        if not isinstance(u,torch.Tensor):
            raise TypeError('replay_u must be a torch.Tensor')
        InputValidation.require(u.ndim==4,'replay_u must be [block_num,N,S,D]')
        blocks=a['checkpoint_state'].shape[0]
        InputValidation.require((blocks>0)&(batch>0)&(batch<=blocks),'batch must be in [1,block_num]')
        capacity=u.shape[2]
        InputValidation.require(capacity>0,'record S capacity must be positive')
        for name,dtype in (('replay_u',torch.float32),('replay_k',torch.float16),('replay_decay',torch.float32)):
            InputValidation.state_pool_metadata(name,a[name],(blocks,heads,capacity,D),dtype,device)
        return capacity

    @staticmethod
    def commit_metadata(a):
        checkpoint = a['checkpoint_state']
        if not isinstance(checkpoint, torch.Tensor):
            raise TypeError('checkpoint_state must be a torch.Tensor')
        InputValidation.require(checkpoint.ndim == 4, 'checkpoint_state must be [block_num,N,D,D]')
        blocks, heads = checkpoint.shape[:2]
        device = checkpoint.device
        InputValidation.require((blocks > 0) & (heads > 0), 'paged-state block count and heads must be positive')
        InputValidation.state_pool_metadata('checkpoint_state', checkpoint, (blocks, heads, D, D), torch.float32, device)
        idx = a['state_indices']
        if not isinstance(idx, torch.Tensor):
            raise TypeError('state_indices must be a torch.Tensor')
        InputValidation.require(idx.ndim == 1, 'state_indices must be [B]')
        batch = idx.shape[0]
        InputValidation.tensor_metadata('state_indices', idx, (batch,), torch.int32, device)
        InputValidation.tensor_metadata('num_accepted_tokens', a['num_accepted_tokens'], (batch,), torch.int32, device)
        InputValidation._records(a, batch, heads, device)
        InputValidation.require(not checkpoint.requires_grad, 'checkpoint_state is an inference-only writable buffer')
        return {name: value for name, value in a.items() if isinstance(value, torch.Tensor)}

    @staticmethod
    def _bounded(values, low, high, name, message=None):
        if any((v < low or v > high for v in values)):
            raise ValueError(message or f'{name} values must be in [{low},{high}]')

    @staticmethod
    def validate_commit_values(a):
        if torch.npu.is_current_stream_capturing():
            return
        indices = a['state_indices'].detach().cpu().tolist()
        InputValidation._bounded(indices, 0, a['checkpoint_state'].shape[0] - 1, 'state_indices')
        InputValidation.require(len(indices) == len(set(indices)), 'state_indices must be distinct across batch')
        InputValidation._bounded(a['num_accepted_tokens'].detach().cpu().tolist(), 0, a['replay_u'].shape[2], 'num_accepted_tokens')

    @staticmethod
    def validate_aliases(tensors, writable):
        spans = {}
        for name, tensor in tensors.items():
            InputValidation.require(tensor.device.type != 'meta', 'physical alias validation requires real tensors')
            start = tensor.data_ptr()
            size = tensor.numel() * tensor.element_size()
            if name in ('conv_state','checkpoint_state','replay_u','replay_k','replay_decay'):
                width = tensor[0].numel() * tensor.element_size()
                ranges = [(start + i * tensor.stride(0) * tensor.element_size(), start + i * tensor.stride(0) * tensor.element_size() + width) for i in range(tensor.shape[0])]
            else:
                if tensor.device.type in ('npu', 'privateuseone'):
                    import torch_npu
                    if torch_npu.get_npu_format(tensor) == 29:
                        size = int(torch_npu.get_storage_size(tensor)) * tensor.element_size()
                ranges = [(start, start + size)]
            spans[name] = ranges
        names = tuple(tensors)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                if left not in writable and right not in writable:
                    continue
                if tensors[left].device != tensors[right].device:
                    continue
                x = y = 0
                a = spans[left]
                b = spans[right]
                while x < len(a) and y < len(b):
                    if a[x][0] < b[y][1] and b[y][0] < a[x][1]:
                        raise ValueError(f'writable buffer overlap: {left} and {right}')
                    if a[x][1] <= b[y][0]:
                        x += 1
                    else:
                        y += 1


    @staticmethod
    def real_boundary(tensors, writable):
        for name, tensor in tensors.items():
            InputValidation.require(tensor.device.type in ('npu', 'privateuseone'), f'{name} requires NPU inputs')
            InputValidation.require(type(tensor) is torch.Tensor, f'{name} must be an exact Tensor; detach Parameter inputs')
        InputValidation.validate_aliases(tensors, writable)
        import torch_npu
        nz_weights = ('qkv_projection_weight', 'decay_projection_a_weight', 'decay_projection_b_weight', 'beta_projection_weight', 'output_gate_projection_weight', 'output_projection_weight')
        for name, tensor in tensors.items():
            if name not in nz_weights:
                InputValidation.require(torch_npu.get_npu_format(tensor) in (0, 2), f'{name} requires ND storage; paged-state addressing does not support blocked physical formats')
        if 'hidden_states' in tensors:
            import cannbotdsl
            for name in ('qkv_projection_weight', 'decay_projection_a_weight', 'decay_projection_b_weight', 'beta_projection_weight', 'output_gate_projection_weight', 'output_projection_weight'):
                InputValidation.require(torch_npu.get_npu_format(tensors[name]) == 29, f'{name} requires BF16 FRACTAL_NZ storage')
            ref = tensors['hidden_states']
            hardware = torch.npu.get_device_properties(ref.device)
            info = cannbotdsl.get_platform_info(stream=torch.npu.current_stream(ref.device))
            available = min(int(hardware.cube_core_num), int(info.cube_core_num), int(hardware.vector_core_num) // 2, int(info.vector_core_num) // 2)
            InputValidation.require(available >= 32 or available in (8, 16, 24, 28), f'AIC quota {available} is not supported; use 8/16/24/28/32 or a full stream with at least 32 AIC and 64 AIV')


class KDAExecutionGuard:
    """负责调用同步和异步资源生命周期保护。"""

    @staticmethod
    def serialized(function):

        @functools.wraps(function)
        def checked(*args, **kwargs):
            ref = args[0] if args else kwargs.get('hidden_states', kwargs.get('checkpoint_state'))
            if not isinstance(ref, torch.Tensor) or ref.device.type not in ('npu', 'privateuseone'):
                return function(*args, **kwargs)
            device = ref.device
            with torch.npu.device(device):
                if torch.npu.is_current_stream_capturing():
                    with _call_lock:
                        return function(*args, **kwargs)
                with _call_lock:
                    if _pending['event'] is not None:
                        _pending['event'].synchronize()
                        _pending['event'] = None
                        _pending['keep'] = None
                    result = None
                    try:
                        result = function(*args, **kwargs)
                        return result
                    finally:
                        event = torch.npu.Event()
                        event.record(torch.npu.current_stream(device))
                        _pending['event'] = event
                        _pending['keep'] = (args, kwargs, result)
        return checked


@KDAExecutionGuard.serialized
def commit_recurrent_kda(checkpoint_state:torch.Tensor,replay_u:torch.Tensor,replay_k:torch.Tensor,
    replay_decay:torch.Tensor,num_accepted_tokens:torch.Tensor,state_indices:torch.Tensor)->None:
    """将已接受的 PA 记录前缀提交到对应的检查点块。

    checkpoint_state 为 FP32，形状为 [block_num,N,D,D]。
    记录只读，形状为 [block_num,N,S,D]；U、K、decay 分别为 FP32、FP16、FP32。
    当前 D=128。各池均通过 state_indices 寻址，第 0 轴可独立包含填充间隙。
    块 0 是有效块。未选块、PA 间隙和全部记录保持不变。
    """
    a=locals().copy();tensors=InputValidation.commit_metadata(a)
    InputValidation.real_boundary(tensors,('checkpoint_state',))
    InputValidation.validate_commit_values(a)
    batch=state_indices.shape[0];heads=checkpoint_state.shape[1]
    cores=_resolve_block_num(checkpoint_state)
    lanes=_commit_launch_blocks(batch,heads,cores)
    def span(tensor):
        length=(tensor.shape[0]-1)*tensor.stride(0)+tensor[0].numel()
        return tensor.as_strided((length,),(1,))
    _get_compiled_commit_kernel()(span(checkpoint_state),span(replay_u),span(replay_k),span(replay_decay),
        num_accepted_tokens,state_indices,checkpoint_state.stride(0),replay_u.stride(0),
        replay_k.stride(0),replay_decay.stride(0),heads,replay_u.shape[2],lanes)


@export('commit_recurrent_kda')
def export_commit_recurrent_kda():
    handle=_compile_commit_kernel();handle.close()


_graph_library=torch.library.Library('cannbotdsl_commit_recurrent_kda','DEF')
_graph_library.define('commit_recurrent_kda(Tensor(a!) checkpoint_state, Tensor replay_u, Tensor replay_k, '
    'Tensor replay_decay, Tensor num_accepted_tokens, Tensor state_indices) -> ()')


@torch.library.impl(_graph_library,'commit_recurrent_kda','Meta')
def _commit_recurrent_kda_meta(checkpoint_state,replay_u,replay_k,replay_decay,num_accepted_tokens,
    state_indices):
    InputValidation.commit_metadata(locals());return None


@torch.library.impl(_graph_library,'commit_recurrent_kda','PrivateUse1')
def _commit_recurrent_kda_privateuse1(*args,**kwargs):return commit_recurrent_kda(*args,**kwargs)


commit_recurrent_kda_op=torch.ops.cannbotdsl_commit_recurrent_kda.commit_recurrent_kda
torch_dynamo.substitute_in_graph(commit_recurrent_kda,can_constant_fold_through=False)(
    lambda *args,**kwargs:commit_recurrent_kda_op(*args,**kwargs))
