# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# fmt: off

"""Standalone accepted-prefix checkpoint replay for Mega KDA ReplaySSM."""

# Tensor operands are explicit to preserve the kernel ABI and DSL lexical order.
# Hardware stages and synchronization/movement order are fixed expansions;
# extracting helpers would change DSL lowering or measured kernel performance.

from __future__ import annotations

import cannbotdsl
import torch
import torch._dynamo as torch_dynamo
from cannbotdsl import MemLoc, Tensor, TensorSpec, dtypes, make_layout
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
from cannbotdsl.tensor import idx2crd, reinterpret, tile_slice

TILE = 128
VL = 64
KDA_STATE_UPDATE_UNROLL = 4
SUPPORTED_BLOCK_NUMS = (8, 24, 28, 32)
SMALL_BATCH_HEAD_LIMIT = 16
MEDIUM_BATCH_HEAD_LIMIT = 32
SMALL_BATCH_ROW_BLOCK = 32
_COMPILED_COMMIT_KERNEL = None


def _device_block_num(ref: torch.Tensor) -> int:
    """Return the current stream's effective AIC count, capped at 32."""
    if ref.device.type not in {"npu", "privateuseone"}:
        raise ValueError("commit_recurrent_kda_replayssm requires NPU inputs")
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


def _resolve_block_num(ref: torch.Tensor | None, block_num: int | None) -> int:
    """Resolve the public AIC-equivalent quota."""
    if block_num is None:
        if ref is None:
            raise ValueError("block_num=None requires a reference NPU tensor")
        available = _device_block_num(ref)
        choices = tuple(value for value in SUPPORTED_BLOCK_NUMS if value <= available)
        if not choices:
            raise ValueError(
                f"device exposes {available} AIC cores; at least 8 are required"
            )
        return choices[-1]
    if isinstance(block_num, bool) or not isinstance(block_num, int):
        raise ValueError(
            f"block_num must be one of {SUPPORTED_BLOCK_NUMS}; got {block_num!r}"
        )
    if block_num not in SUPPORTED_BLOCK_NUMS:
        raise ValueError(
            f"block_num must be one of {SUPPORTED_BLOCK_NUMS}; got {block_num}"
        )
    if ref is not None and ref.device.type in {"npu", "privateuseone"}:
        available = _device_block_num(ref)
        if block_num > available:
            raise ValueError(
                f"block_num={block_num} exceeds the device limit {available}"
            )
    return block_num


def _commit_launch_blocks(batch_size: int, num_heads: int, aic_blocks: int) -> int:
    """Use Snapshot row blocks to expose enough work to both AIV subblocks."""
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
    """Describe the kernel's disjoint state-row assignments for host tests."""
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


class CommitRecurrentKDAReplaySSMStage:
    """Replay one accepted token stream for each batch/head checkpoint."""

    def __init__(self):
        self.state_current = Buffer(MemLoc.UB, dtypes.float32, make_layout((TILE, TILE)))
        # The paired schedule keeps the next state tile in UB while the
        # current head drains its accepted-token stream.  The tile shape is
        # independent of B, N and S; only the scheduler is runtime-shaped.
        self.state_next = Buffer(MemLoc.UB, dtypes.float32, make_layout((TILE, TILE)))
        self.replay_u = RingBuffer(MemLoc.UB, dtypes.float16, make_layout((1, TILE)), depth=2)
        self.replay_k = RingBuffer(MemLoc.UB, dtypes.float16, make_layout((1, TILE)), depth=2)
        self.replay_decay = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((1, TILE)), depth=2)
        self.u_fp32 = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, TILE)))

    @jit
    def prefetch_record(
        self,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        batch_index,
        token,
        value_head,
        row_index,
        active_rows: int,
    ):
        replay_u = self.replay_u.next_write()
        if active_rows < TILE:
            replay_u = reinterpret(replay_u, offset=0, layout=make_layout((1, active_rows)))
        mem_copy(
            replay_u,
            tile_slice(
                replay_u_gm[batch_index, token, None, None],
                (1, active_rows),
                (value_head, row_index),
            ),
        )
        mem_copy(self.replay_k.next_write(), tile_slice(replay_k_gm[batch_index, token, None, None], (1, TILE), (value_head, 0)))
        mem_copy(self.replay_decay.next_write(), tile_slice(replay_decay_gm[batch_index, token, None, None], (1, TILE), (value_head, 0)))

    @jit
    def apply_current_record(self, state, active_rows: int):
        replay_u = self.replay_u.next_read()
        replay_k = self.replay_k.next_read()
        replay_decay = self.replay_decay.next_read()
        # Keep U conversion separate: on current CANN 9.2 this is faster than
        # one VF region with an explicit vst_vld barrier.
        with vf(mode="simd"):
            for segment in range((active_rows + VL - 1) // VL):
                offset = segment * VL
                segment_rows = active_rows - offset
                segment_mask, _ = update_mask(segment_rows, elem_bits=32)
                update_packed = vload_unpack(
                    replay_u,
                    offset,
                    unpack_mode=UnpackMode.B16_TO_B32,
                )
                update = vcast(update_packed, dtypes.float32, mask=segment_mask)
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
        state_gm: Tensor,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
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
                state = reinterpret(self.state_current, offset=0, layout=make_layout((active_rows, TILE)))
            state_tile = tile_slice(
                state_gm[batch_index, value_head, None, None],
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
        state_gm: Tensor,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        accepted_gm: Tensor,
        current_item,
        next_item,
        batch_size,
        num_heads,
        sequence_length,
        next_row_index,
        next_active_rows: int,
    ):
        """Drain two disjoint state tiles through one UB double buffer."""
        current_batch, current_head = idx2crd(current_item, [batch_size, num_heads])
        next_batch, next_head = idx2crd(next_item, [batch_size, num_heads])
        current_accepted = dtypes.int64(accepted_gm[current_batch])
        next_accepted = dtypes.int64(accepted_gm[next_batch])
        current_state_tile = tile_slice(
            state_gm[current_batch, current_head, None, None],
            (TILE, TILE),
            (0, 0),
        )
        next_state_tile = tile_slice(
            state_gm[next_batch, next_head, None, None],
            (next_active_rows, TILE),
            (next_row_index, 0),
        )
        next_state = self.state_next
        if next_active_rows < TILE:
            next_state = reinterpret(self.state_next, offset=0, layout=make_layout((next_active_rows, TILE)))

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
        state_gm: Tensor,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        accepted_gm: Tensor,
        row_block: int,
    ):
        batch_size = state_gm.shape[0]
        num_heads = state_gm.shape[1]
        sequence_length = replay_u_gm.shape[1]
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
        state_gm: Tensor,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        accepted_gm: Tensor,
    ):
        batch_size = state_gm.shape[0]
        num_heads = state_gm.shape[1]
        sequence_length = replay_u_gm.shape[1]
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
        state_gm: Tensor,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        accepted_gm: Tensor,
    ):
        """Use paired full/half-head work for large even head counts.

        The mapping is the generalized form of the former B16/N6 schedule:
        low AIV grids process several full-head pairs, while larger grids use
        a full head plus one half of a second head per lane.
        """
        batch_size = state_gm.shape[0]
        num_heads = state_gm.shape[1]
        sequence_length = replay_u_gm.shape[1]
        batch_heads = batch_size * num_heads
        block_idx = get_block_idx()
        aiv_block_count = get_block_num()

        # If the launch grid has one lane per head, do not manufacture a
        # second item.  This is the tail case for small dynamic B*N values.
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
            # Every lane drains one or more full-head pairs.
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
            # This is the Snapshot-style split schedule.  `pair_lanes`
            # process two full heads; the remaining lanes process one full
            # head plus one half of the remaining head through state_next.
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
            # When only a small tail remains, pair full heads instead of
            # splitting a tile.  This keeps every item in bounds for shapes
            # such as B=7,N=6 while still using the UB double buffer.
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
        state_gm: Tensor,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        accepted_gm: Tensor,
    ):
        batch_heads = state_gm.shape[0] * state_gm.shape[1]
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


class CommitRecurrentKDAReplaySSMKernel:
    """Launch wrapper for the standalone AIV commit kernel."""

    @kernel
    def kernel(
        self,
        state_gm: Tensor,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        accepted_gm: Tensor,
    ):
        CommitRecurrentKDAReplaySSMStage().execute(
            state_gm,
            replay_u_gm,
            replay_k_gm,
            replay_decay_gm,
            accepted_gm,
        )

    @host
    def run(
        self,
        state_gm: Tensor,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        accepted_gm: Tensor,
        launch_blocks: int,
    ):
        self.kernel[launch_blocks](
            state_gm,
            replay_u_gm,
            replay_k_gm,
            replay_decay_gm,
            accepted_gm,
        )


def _compile_commit_kernel():
    batch_dim = cannbotdsl.Dim("commit_batch", min=1)
    sequence_dim = cannbotdsl.Dim("commit_sequence", min=1)
    heads_dim = cannbotdsl.Dim("commit_heads", min=1)
    return cannbotdsl.compile(
        CommitRecurrentKDAReplaySSMKernel().run,
        TensorSpec((batch_dim, heads_dim, TILE, TILE), dtypes.float32),
        TensorSpec(
            (batch_dim, sequence_dim, heads_dim, TILE),
            dtypes.float16,
        ),
        TensorSpec(
            (batch_dim, sequence_dim, heads_dim, TILE),
            dtypes.float16,
        ),
        TensorSpec(
            (batch_dim, sequence_dim, heads_dim, TILE),
            dtypes.float32,
        ),
        TensorSpec((batch_dim,), dtypes.int32),
        dtypes.int64,
    )


def _get_compiled_commit_kernel():
    global _COMPILED_COMMIT_KERNEL
    if _COMPILED_COMMIT_KERNEL is None:
        _COMPILED_COMMIT_KERNEL = _compile_commit_kernel()
    return _COMPILED_COMMIT_KERNEL


def _validate_commit_state(recurrent_state, replay_u):
    if recurrent_state.ndim != 4 or tuple(recurrent_state.shape[2:]) != (TILE, TILE):
        raise ValueError(f"recurrent_state must be [B,N,{TILE},{TILE}]")
    batch_size, num_heads = map(int, recurrent_state.shape[:2])
    if batch_size <= 0 or num_heads <= 0:
        raise ValueError("recurrent_state B and N must be positive")
    if recurrent_state.dtype != torch.float32 or not recurrent_state.is_contiguous():
        raise ValueError("recurrent_state must be contiguous FP32")
    if replay_u.ndim != 4:
        raise ValueError("replay_u must be [B,S,N,128]")
    sequence_length = int(replay_u.shape[1])
    if sequence_length <= 0:
        raise ValueError("replay_u S must be positive")
    return batch_size, num_heads, sequence_length


def _validate_commit_records(replay_u, replay_k, replay_decay, replay_shape):
    for name, tensor in (("replay_u", replay_u), ("replay_k", replay_k)):
        if tuple(tensor.shape) != replay_shape:
            raise ValueError(f"{name} must be [B,S,N,{TILE}]")
        if tensor.dtype != torch.float16 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous FP16")
    if tuple(replay_decay.shape) != replay_shape:
        raise ValueError(f"replay_decay must be [B,S,N,{TILE}]")
    if replay_decay.dtype != torch.float32 or not replay_decay.is_contiguous():
        raise ValueError("replay_decay must be contiguous FP32")


def _validate_commit_control(num_accepted_tokens, batch_size, device, records):
    if tuple(num_accepted_tokens.shape) != (batch_size,):
        raise ValueError("num_accepted_tokens must be [B]")
    if (
        num_accepted_tokens.dtype != torch.int32
        or not num_accepted_tokens.is_contiguous()
    ):
        raise ValueError("num_accepted_tokens must be contiguous int32")
    for name, tensor in records:
        if tensor.device != device:
            raise ValueError(f"{name} must share the recurrent_state device")


def commit_recurrent_kda_replayssm(
    recurrent_state: torch.Tensor,
    replay_u: torch.Tensor,
    replay_k: torch.Tensor,
    replay_decay: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    block_num: int | None = None,
) -> None:
    """Replay each accepted candidate prefix into its FP32 KDA checkpoint.

    Args:
        recurrent_state (torch.Tensor): FP32, contiguous, shape ``[B,N,D,D]``
            in batch, head, value-row, key-column order; updated in place.
        replay_u (torch.Tensor): FP16, contiguous, shape ``[B,S,N,D]``; U
            records produced by ``mega_recurrent_kda_replayssm`` and read
            without modification.
        replay_k (torch.Tensor): FP16, contiguous, shape ``[B,S,N,D]``;
            normalized K records produced by ``mega_recurrent_kda_replayssm``
            and read without modification.
        replay_decay (torch.Tensor): FP32, contiguous, shape ``[B,S,N,D]``;
            activated decay records produced by ``mega_recurrent_kda_replayssm``
            and read without modification.
        num_accepted_tokens (torch.Tensor): INT32, contiguous, shape ``[B]``;
            each value is in ``[0,S]`` and selects the prefix length committed
            for that batch item.
        block_num (int | None): Runtime AIC-equivalent quota, one of ``8``,
            ``24``, ``28``, or ``32``. ``None`` selects the largest supported
            count within the current stream quota; the AIV kernel may launch
            up to ``2*block_num`` blocks.
    Returns:
        None: Results are written to ``recurrent_state`` in place. Replay records
            and ``num_accepted_tokens`` remain unchanged.

    Notes:
        * ``B``, ``S`` and ``N`` are positive runtime dimensions; ``D = 128`` is fixed.
        * All tensors must be contiguous and on the same NPU, with the shapes and dtypes above.
        * For each batch and token ``t < num_accepted_tokens[b]``, the recurrence
          is ``state = state * replay_decay[:,t] + outer(replay_u[:,t],
          replay_k[:,t])``. A zero accepted count leaves that batch checkpoint
          unchanged.
        * Accepted-count values must satisfy the bounds above and are not copied to the host for validation.
        * One compiled callable handles dynamic B/S/N and runtime core control.
          Changing dimensions or the core quota reuses the same binary.
        * Eager calls use this host wrapper. ``torch.compile`` substitutes the
          registered dispatcher op, and the module also registers a native
          binary entry named ``commit_recurrent_kda_replayssm``.
    """
    batch_size, num_heads, sequence_length = _validate_commit_state(
        recurrent_state, replay_u
    )
    replay_shape = (batch_size, sequence_length, num_heads, TILE)
    _validate_commit_records(replay_u, replay_k, replay_decay, replay_shape)
    records = (
        ("replay_u", replay_u),
        ("replay_k", replay_k),
        ("replay_decay", replay_decay),
        ("num_accepted_tokens", num_accepted_tokens),
    )
    _validate_commit_control(
        num_accepted_tokens, batch_size, recurrent_state.device, records
    )

    aic_blocks = _resolve_block_num(recurrent_state, block_num)
    launch_blocks = _commit_launch_blocks(batch_size, num_heads, aic_blocks)
    _get_compiled_commit_kernel()(
        recurrent_state,
        replay_u,
        replay_k,
        replay_decay,
        num_accepted_tokens,
        launch_blocks,
    )


@export("commit_recurrent_kda_replayssm")
def export_commit_recurrent_kda_replayssm():
    """Export one dynamic-B/S/N commit binary for positive runtime dimensions."""
    handle = _compile_commit_kernel()
    handle.close()


_GRAPH_LIBRARY = torch.library.Library(
    "cannbotdsl_commit_recurrent_kda_replayssm", "DEF"
)
_GRAPH_LIBRARY.define(
    "commit_recurrent_kda_replayssm("
    "Tensor(a!) recurrent_state, "
    "Tensor replay_u, "
    "Tensor replay_k, "
    "Tensor replay_decay, "
    "Tensor num_accepted_tokens, "
    "int? block_num=None"
    ") -> ()"
)


@torch.library.impl(_GRAPH_LIBRARY, "commit_recurrent_kda_replayssm", "Meta")
def _commit_recurrent_kda_replayssm_meta(
    recurrent_state,
    replay_u,
    replay_k,
    replay_decay,
    num_accepted_tokens,
    block_num=None,
):
    del recurrent_state, replay_u, replay_k, replay_decay
    del num_accepted_tokens, block_num
    return None


@torch.library.impl(_GRAPH_LIBRARY, "commit_recurrent_kda_replayssm", "PrivateUse1")
def _commit_recurrent_kda_replayssm_privateuse1(
    recurrent_state,
    replay_u,
    replay_k,
    replay_decay,
    num_accepted_tokens,
    block_num=None,
):
    commit_recurrent_kda_replayssm(
        recurrent_state,
        replay_u,
        replay_k,
        replay_decay,
        num_accepted_tokens,
        block_num,
    )


commit_recurrent_kda_replayssm_op = (
    torch.ops.cannbotdsl_commit_recurrent_kda_replayssm.commit_recurrent_kda_replayssm
)

# Register the public wrapper without accessing Torch's protected module attribute.
torch_dynamo.substitute_in_graph(
    commit_recurrent_kda_replayssm, can_constant_fold_through=False
)(lambda *args, **kwargs: commit_recurrent_kda_replayssm_op(*args, **kwargs))
