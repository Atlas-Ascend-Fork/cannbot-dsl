# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Arch35 MXFP8 core partitioning and stream-ordered schedule reuse.

Reuse requires unchanged lengths and execution configuration. Advance
``lengths_epoch`` after writes invisible to Tensor._version (external kernels,
raw pointers, or graph replay). Reusing inference tensors requires an explicit epoch.
No implicit global cache retains user tensors. The state can be released after
use; record_stream protects buffers consumed asynchronously on other streams.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading

import torch
from cannbotdsl.aicpu import I64, U32, aicpu_kernel, current_raw_stream
from cannbotdsl.aicpu.toolchain import CompiledAicpuKernel, compile_aicpu_kernel

try:
    from cannbotdsl.aicpu import GmIn, GmOut
except ImportError:
    from cannbotdsl.aicpu import gm_in as GmIn, gm_out as GmOut
from .mxfp8_check import (
    MASK_MODE_RIGHT_DOWN_CAUSAL,
    SUPPORTED_MASK_MODES,
    _validate_length_vectors,
)

# Header, FA records (per section/AIC) and FD records (per section/AIV).
METADATA_STRIDE = 16
SECTION_COUNT, HAS_FD = 0, 1
FA_START_BN, FA_START_M, FA_START_S2 = 0, 1, 2
FA_END_BN, FA_END_M, FA_END_S2, FA_FIRST_SLOT = 3, 4, 5, 6
FD_BN, FD_M, FD_FIRST_SLOT, FD_PARTS, FD_ROW_BEGIN, FD_ROWS = range(6)
L2_BYTES = 96 * 1024 * 1024
FA_TOLERANCE_RATIO = 2
FD_LEAST_BLOCKS = 3
FD_TOLERANCE_BLOCKS = 10
# ABI header fields consumed as int64 words; each record reserves 16 words.
HEADER_TILE_M, HEADER_MACRO_N, HEADER_CORES, HEADER_VECTORS = 2, 3, 4, 5
AICPU_ARRAY_ARENA_BYTES = 1 << 22
ARRAY_ALIGNMENT_BYTES = 16
INT64_BYTES, UINT32_BYTES = 8, 4
UINT32_MAX = (1 << 32) - 1
INT64_MAX = (1 << 63) - 1
_CONFIG_FIELDS = frozenset(
    (
        "batch",
        "heads",
        "group",
        "head_dim",
        "tile_m",
        "macro_n",
        "sparse",
        "cores",
        "vectors",
        "sections",
    )
)
COST_M_ALIGN, COST_S2_ALIGN = 16, 64
COST_M_WEIGHT, COST_S2_WEIGHT = 6, 10


class MetadataArgs:
    q: GmIn(I64)
    kv: GmIn(I64)
    metadata: GmOut(I64)
    batch: I64
    heads: I64
    group: I64
    head_dim: I64
    tile_m: I64
    macro_n: I64
    sparse: I64
    cores: I64
    vectors: I64
    sections: I64


def get_group_size(state):
    return state.group


def get_q_seq_size_at(state, batch_idx):
    return state.q_seq[batch_idx]


def get_kv_seq_size_at(state, batch_idx):
    return state.kv_seq[batch_idx]


def get_pre_token_left_up(state, query_seq, kv_seq):
    return query_seq


def get_next_token_left_up(state, query_seq, kv_seq):
    if state.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL:
        return kv_seq - query_seq
    return kv_seq


def is_within_tolerance(limit, tolerance, value):
    # Inclusive limit check: can value be added without exceeding limit + tolerance?
    #
    # Tolerance is needed because work units rarely align with the ideal split;
    # the final unit is often partial. A strict cost <= limit can prevent progress
    # or cause severe tail imbalance; callers use tail-block cost / fa_tolerance_ratio.
    if limit + tolerance >= value:
        return 1
    return 0


def calc_cost(basic_m, basic_s2):
    # Default model: align M and S2 upward, then take a weighted sum.
    #
    # Aligning M to COST_M_ALIGN and S2 to COST_S2_ALIGN accounts for the fixed
    # transfer granularity: even a smaller tile pays the cost of a full unit.
    align_basic_m = (basic_m + COST_M_ALIGN - 1) // COST_M_ALIGN
    align_basic_s2 = (basic_s2 + COST_S2_ALIGN - 1) // COST_S2_ALIGN
    return COST_M_WEIGHT * align_basic_m + COST_S2_WEIGHT * align_basic_s2


# ---------------------------------------------------------------------------
# Grid information and section partitioning (CalcGridInfo / CalcGridInfoSection)
# ---------------------------------------------------------------------------


def calc_grid_info(state, grid):
    # Count M/S2 blocks and tail sizes per batch, and detect an all-empty grid.
    grid.is_empty = 1
    for batch_idx in range(0, state.batch_size):
        s1_size = get_q_seq_size_at(state, batch_idx)
        s2_size = get_kv_seq_size_at(state, batch_idx)
        # One M row covers group_size Q heads, so total M rows = s1 * group_size.
        m_total = s1_size * get_group_size(state)
        grid.m_block_num[batch_idx] = ceil_div(m_total, state.m_base)
        grid.m_tail_size[batch_idx] = m_total % state.m_base
        grid.s2_block_num[batch_idx] = ceil_div(s2_size, state.s2_base)
        grid.s2_tail_size[batch_idx] = s2_size % state.s2_base
        if grid.m_block_num[batch_idx] != 0 and grid.s2_block_num[batch_idx] != 0:
            grid.is_empty = 0
    return 0


def calc_grid_info_section(state, grid):
    # Split the (batch, scheduling_head) axis into sections according to L2 capacity.
    #
    # Keep the existing per-scheduling-head L2 footprint heuristic:
    # 2 * D * (visible-Q + KV). This is a partitioning estimate, not an exact
    # byte count for the FP8 inputs and BF16 output. Prefill schedules Q heads;
    # Decode schedules KV heads with grouped Q rows.
    bn2_total = state.batch_size * state.schedule_head_count
    grid.section_num = 0
    # l2_bytes == 0 means no L2 limit and no section splitting.
    if state.l2_bytes == 0:
        grid.section_bn2_idx.append(bn2_total)
        grid.section_num = 1
        return 0

    bn2_idx = 0
    token_limit = state.l2_bytes
    token_size = 0
    max_gs1_size = 0
    max_single_head_cost = 0
    head_dim = state.head_dim
    for batch_idx in range(0, state.batch_size):
        s1_size = get_q_seq_size_at(state, batch_idx)
        s2_size = get_kv_seq_size_at(state, batch_idx)
        s1_cost = s1_size * head_dim * 2
        s2v_cost = s2_size * head_dim * 2
        if state.mask_mode == MASK_MODE_RIGHT_DOWN_CAUSAL:
            s1_cost = min(s1_size, s2_size) * head_dim * 2
        single_head_cost = s1_cost + s2v_cost
        max_single_head_cost = max(max_single_head_cost, single_head_cost)
        max_gs1_size = max(max_gs1_size, get_group_size(state) * s1_size)
        for _ in range(0, state.schedule_head_count):
            # Start a section only when the next head exceeds the L2 budget.
            if token_size + single_head_cost > token_limit and token_size != 0:
                grid.section_bn2_idx.append(bn2_idx)
                grid.section_num += 1
                token_size = 0
            token_size += single_head_cost
            bn2_idx += 1
    grid.section_bn2_idx.append(bn2_total)
    grid.section_num += 1

    # Fallback to one section if M is below one base tile or one head fits its L2 share.
    if (
        max_gs1_size <= state.m_base
        or max_single_head_cost <= token_limit // state.aic_core_max
    ):
        grid.section_bn2_idx.reset(0, 0)
        grid.section_bn2_idx.append(bn2_total)
        grid.section_num = 1
    return 0


# ---------------------------------------------------------------------------
# Cost information (CalcBatchCache / CalcS2Range / CalcS1GCache / CalcBatchCost / CalcCostInfo)
# ---------------------------------------------------------------------------


def calc_batch_cache(batch_idx, state, grid, batch_cache):
    # Cache one batch's sequence lengths, window extensions, and 2 x 2 cost table.
    batch_cache.batch_idx = batch_idx
    batch_cache.s1_size = get_q_seq_size_at(state, batch_idx)
    batch_cache.s2_size = get_kv_seq_size_at(state, batch_idx)
    batch_cache.pre_token_left_up = get_pre_token_left_up(
        state, batch_cache.s1_size, batch_cache.s2_size
    )
    batch_cache.next_token_left_up = get_next_token_left_up(
        state, batch_cache.s1_size, batch_cache.s2_size
    )
    # First table index: M axis (full/tail row); second: S2 axis (full/tail block).
    # A zero tail size means the combination cannot occur; use zero cost.
    m_tail = grid.m_tail_size[batch_idx]
    s2_tail = grid.s2_tail_size[batch_idx]
    batch_cache.cost_normal_normal = calc_cost(state.m_base, state.s2_base)
    if m_tail == 0:
        batch_cache.cost_tail_normal = 0
        batch_cache.cost_tail_tail = 0
    else:
        batch_cache.cost_tail_normal = calc_cost(m_tail, state.s2_base)
        if s2_tail == 0:
            batch_cache.cost_tail_tail = 0
        else:
            batch_cache.cost_tail_tail = calc_cost(m_tail, s2_tail)
    if s2_tail == 0:
        batch_cache.cost_normal_tail = 0
    else:
        batch_cache.cost_normal_tail = calc_cost(state.m_base, s2_tail)
    return 0


def calc_s2_range(s1g_idx, state, batch_cache, s1g_cache):
    # Map an M row to its required S2 block interval [s2_start, s2_end).
    #
    # Derive the row's token interval, expand it by the mask/window to an S2
    # token interval, then convert that interval to block indices.
    s1g_cache.s2_start = 0
    s1g_cache.s2_end = 0
    # A row has no valid tokens when its sequence length is zero.
    if batch_cache.s1_size == 0 or batch_cache.s2_size == 0:
        return 0
    # No mask: the entire KV sequence is visible.
    if state.mask_mode == 0:
        s1g_cache.s2_end = ceil_div(batch_cache.s2_size, state.s2_base)
        return 0

    # 1. S1G row -> S1 token interval -> S2 token interval
    s1g_first_token = s1g_idx * state.m_base
    s1g_last_token = (
        min(s1g_first_token + state.m_base, batch_cache.s1_size * get_group_size(state))
        - 1
    )
    s1_first_token = s1g_first_token // get_group_size(state)
    s1_last_token = s1g_last_token // get_group_size(state)
    s2_first_token = s1_first_token - batch_cache.pre_token_left_up
    s2_last_token = s1_last_token + batch_cache.next_token_left_up
    # The expanded interval does not overlap KV: this row has no valid tokens.
    if (
        s2_first_token >= batch_cache.s2_size
        or s2_last_token < 0
        or s2_last_token < s2_first_token
    ):
        return 0

    # 2. Token indices -> block indices
    s2_first_token = max(s2_first_token, 0)
    s2_last_token = min(s2_last_token, batch_cache.s2_size - 1)
    s1g_cache.s2_start = s2_first_token // state.s2_base
    s1g_cache.s2_end = s2_last_token // state.s2_base + 1
    return 0


def calc_s1g_cache(s1g_idx, state, grid, batch_cache, s1g_cache):
    # Compute the S2 interval, block count, and cost for one M row.
    s1g_cache.batch_idx = batch_cache.batch_idx
    s1g_cache.s1g_idx = s1g_idx
    calc_s2_range(s1g_idx, state, batch_cache, s1g_cache)
    if (
        s1g_cache.s2_start >= s1g_cache.s2_end
        or grid.m_block_num[batch_cache.batch_idx] == 0
    ):
        s1g_cache.s1g_block = 0
        s1g_cache.s1g_cost = 0
        s1g_cache.s1g_last_block_cost = 0
        s1g_cache.s1g_normal_block_cost = 0
        return 0

    s1g_cache.s1g_block = s1g_cache.s2_end - s1g_cache.s2_start
    # Whether this row reaches the batch's final S2 block when that block is a tail.
    if (
        grid.s2_tail_size[batch_cache.batch_idx] != 0
        and s1g_cache.s2_end == grid.s2_block_num[batch_cache.batch_idx]
    ):
        cur_tail_s2_num = 1
    else:
        cur_tail_s2_num = 0
    cur_normal_s2_num = s1g_cache.s1g_block - cur_tail_s2_num
    # Use the TAIL cost-table row for the final partial M row; otherwise NORMAL.
    if (
        s1g_idx == grid.m_block_num[batch_cache.batch_idx] - 1
        and grid.m_tail_size[batch_cache.batch_idx] != 0
    ):
        s1g_cache.s1g_cost = (
            batch_cache.cost_tail_normal * cur_normal_s2_num
            + batch_cache.cost_tail_tail * cur_tail_s2_num
        )
        if cur_tail_s2_num > 0:
            s1g_cache.s1g_last_block_cost = batch_cache.cost_tail_tail
        else:
            s1g_cache.s1g_last_block_cost = batch_cache.cost_tail_normal
        s1g_cache.s1g_normal_block_cost = batch_cache.cost_tail_normal
    else:
        s1g_cache.s1g_cost = (
            batch_cache.cost_normal_normal * cur_normal_s2_num
            + batch_cache.cost_normal_tail * cur_tail_s2_num
        )
        if cur_tail_s2_num > 0:
            s1g_cache.s1g_last_block_cost = batch_cache.cost_normal_tail
        else:
            s1g_cache.s1g_last_block_cost = batch_cache.cost_normal_normal
        s1g_cache.s1g_normal_block_cost = batch_cache.cost_normal_normal
    return 0


def calc_batch_cost(batch_idx, state, grid, cost_info):
    # Sum batch cost and block count, plus the last nonempty row's tail-block cost.
    cost_info.bn2_cost_per_batch[batch_idx] = 0
    cost_info.bn2_block_per_batch[batch_idx] = 0
    cost_info.bn2_last_block_cost_per_batch[batch_idx] = 0
    if (
        get_q_seq_size_at(state, batch_idx) == 0
        or get_kv_seq_size_at(state, batch_idx) == 0
    ):
        return 0
    batch_cache = BatchCache()
    s1g_cache = S1GCache()
    calc_batch_cache(batch_idx, state, grid, batch_cache)
    for s1g_idx in range(0, grid.m_block_num[batch_idx]):
        calc_s1g_cache(s1g_idx, state, grid, batch_cache, s1g_cache)
        cost_info.bn2_cost_per_batch[batch_idx] += s1g_cache.s1g_cost
        cost_info.bn2_block_per_batch[batch_idx] += s1g_cache.s1g_block
        if s1g_cache.s1g_block > 0:
            cost_info.bn2_last_block_cost_per_batch[batch_idx] = (
                s1g_cache.s1g_last_block_cost
            )
    return 0


def calc_cost_info(state, grid, cost_info):
    # Aggregate per-batch costs and block counts into per-section totals.
    for batch_idx in range(0, state.batch_size):
        calc_batch_cost(batch_idx, state, grid, cost_info)
    cost_info.section_block_num.reset(0, grid.section_num)
    cost_info.section_cost.reset(0, grid.section_num)
    for section_idx in range(0, grid.section_num):
        bn2_start = 0
        if section_idx > 0:
            bn2_start = grid.section_bn2_idx[section_idx - 1]
        bn2_end = grid.section_bn2_idx[section_idx]
        for bn2_idx in range(bn2_start, bn2_end):
            batch_idx = floor_div_safe(bn2_idx, state.schedule_head_count, 0)
            cost_info.section_block_num[section_idx] += cost_info.bn2_block_per_batch[
                batch_idx
            ]
            cost_info.section_cost[section_idx] += cost_info.bn2_cost_per_batch[
                batch_idx
            ]
    return 0


# ---------------------------------------------------------------------------
# Greedy assignment (AssignByBatch / AssignByRow / AssignByBlock)
#
# All three stages share a protocol and run from coarse to fine granularity:
#   assign_by_batch  one flattened batch/head work unit as an indivisible unit
#   assign_by_row    one M row
#   assign_by_block  one S2 block (fd path only)
# Each step adds assigned work to assign_ctx.core_cache and subtracts it from
# the remaining work until core_cache.cost_limit + tolerance is exceeded.
# ---------------------------------------------------------------------------


def assign_by_batch(state, grid, cost_info, assign_ctx):
    if assign_ctx.is_finished != 0:
        return 0
    # bn2_cost == 0 forces progress: still advance past a batch whose cost is zero,
    # or an empty batch would leave the loop stuck at the same position.
    while (
        assign_ctx.bn2_cost == 0
        or is_within_tolerance(
            assign_ctx.core_cache.cost_limit,
            floor_div_safe(
                cost_info.bn2_last_block_cost_per_batch[assign_ctx.cur_batch_idx],
                state.fa_tolerance_ratio,
                0,
            ),
            assign_ctx.core_cache.cost + assign_ctx.bn2_cost,
        )
        != 0
    ):
        assign_ctx.core_cache.cost += assign_ctx.bn2_cost
        assign_ctx.core_cache.block += assign_ctx.bn2_block
        assign_ctx.cur_bn2_idx += 1
        # Reached the end of this section.
        if assign_ctx.cur_bn2_idx == grid.section_bn2_idx[assign_ctx.cur_section_idx]:
            assign_ctx.cur_s1g_idx = 0
            assign_ctx.cur_s2_idx = 0
            assign_ctx.is_finished = 1
            return 0
        # Advance to the next batch.
        if (
            floor_div_safe(assign_ctx.cur_bn2_idx, state.schedule_head_count, 0)
            != assign_ctx.cur_batch_idx
        ):
            assign_ctx.cur_batch_idx = floor_div_safe(
                assign_ctx.cur_bn2_idx, state.schedule_head_count, 0
            )
            calc_batch_cache(
                assign_ctx.cur_batch_idx, state, grid, assign_ctx.batch_cache
            )
        assign_ctx.bn2_cost = cost_info.bn2_cost_per_batch[assign_ctx.cur_batch_idx]
        assign_ctx.bn2_block = cost_info.bn2_block_per_batch[assign_ctx.cur_batch_idx]
        assign_ctx.cur_s1g_idx = 0
        calc_s1g_cache(
            assign_ctx.cur_s1g_idx,
            state,
            grid,
            assign_ctx.batch_cache,
            assign_ctx.s1g_cache,
        )
        assign_ctx.cur_s2_idx = assign_ctx.s1g_cache.s2_start
    return 0


def assign_by_row(state, grid, cost_info, assign_ctx):
    if assign_ctx.is_finished != 0:
        return 0
    while (
        is_within_tolerance(
            assign_ctx.core_cache.cost_limit,
            floor_div_safe(
                assign_ctx.s1g_cache.s1g_last_block_cost, state.fa_tolerance_ratio, 0
            ),
            assign_ctx.core_cache.cost + assign_ctx.s1g_cache.s1g_cost,
        )
        != 0
    ):
        assign_ctx.core_cache.cost += assign_ctx.s1g_cache.s1g_cost
        assign_ctx.core_cache.block += assign_ctx.s1g_cache.s1g_block
        # Subtract the assigned row from the batch's remaining work, clamping at zero.
        if assign_ctx.bn2_cost > assign_ctx.s1g_cache.s1g_cost:
            assign_ctx.bn2_cost = assign_ctx.bn2_cost - assign_ctx.s1g_cache.s1g_cost
        else:
            assign_ctx.bn2_cost = 0
        if assign_ctx.bn2_block > assign_ctx.s1g_cache.s1g_block:
            assign_ctx.bn2_block = assign_ctx.bn2_block - assign_ctx.s1g_cache.s1g_block
        else:
            assign_ctx.bn2_block = 0
        # Find the next nonempty row; an empty row has no cost and is handled by bn2_cost.
        while True:
            assign_ctx.cur_s1g_idx += 1
            calc_s1g_cache(
                assign_ctx.cur_s1g_idx,
                state,
                grid,
                assign_ctx.batch_cache,
                assign_ctx.s1g_cache,
            )
            if assign_ctx.s1g_cache.s1g_block != 0:
                break
        assign_ctx.cur_s2_idx = assign_ctx.s1g_cache.s2_start
    return 0


def assign_by_block(state, grid, cost_info, assign_ctx):
    if assign_ctx.is_finished != 0:
        return 0
    # Use tail-block cost for this row's last block; full-block cost otherwise.
    cur_cost = assign_ctx.s1g_cache.s1g_normal_block_cost
    if assign_ctx.cur_s2_idx == assign_ctx.s1g_cache.s2_end - 1:
        cur_cost = assign_ctx.s1g_cache.s1g_last_block_cost
    while (
        is_within_tolerance(
            assign_ctx.core_cache.cost_limit,
            floor_div_safe(cur_cost, state.fa_tolerance_ratio, 0),
            assign_ctx.core_cache.cost + cur_cost,
        )
        != 0
    ):
        assign_ctx.core_cache.cost += cur_cost
        assign_ctx.core_cache.block += 1
        assign_ctx.cur_s2_idx += 1
        # Subtract the assigned block from both batch and current-row remaining work.
        assign_ctx.bn2_cost = assign_ctx.bn2_cost - cur_cost
        assign_ctx.s1g_cache.s1g_cost = assign_ctx.s1g_cache.s1g_cost - cur_cost
        assign_ctx.bn2_block -= 1
        assign_ctx.s1g_cache.s1g_block -= 1
    return 0


# ---------------------------------------------------------------------------
# FD reduction tasks (IsNeedRecordFDInfo / RecordFDInfo / ScheduleFd)
# ---------------------------------------------------------------------------


def is_need_record_fd_info(assign_ctx, result):
    # Decide whether this core boundary needs a reduction task.
    #
    # A split rarely lands exactly on a row boundary, so defer recording until
    # the split row is complete; this avoids recording the same row twice.
    if assign_ctx.cur_core_idx == 0:
        return 0
    # The previous core ended without splitting a row; no reduction is needed.
    if assign_ctx.cur_kv_split_part <= 1:
        return 0
    # The split row is not complete yet.
    if (
        assign_ctx.cur_bn2_idx == result.bn2_end[assign_ctx.cur_core_idx - 1]
        and assign_ctx.cur_s1g_idx == result.gs1_end[assign_ctx.cur_core_idx - 1]
    ):
        return 0
    return 1


def record_fd_info(state, grid, assign_ctx, result):
    # Append a reduction task to result's FD task table with split and size.
    # The split point is the previous core's end position.
    split_batch_idx = floor_div_safe(
        result.bn2_end[assign_ctx.cur_core_idx - 1], state.schedule_head_count, 0
    )
    split_s1g_idx = result.gs1_end[assign_ctx.cur_core_idx - 1]
    s1_size = get_q_seq_size_at(state, split_batch_idx)
    # M size of the split row: m_base for a full row, remainder for the last.
    cur_fd_s1g_size = state.m_base
    if split_s1g_idx == grid.m_block_num[split_batch_idx] - 1:
        cur_fd_s1g_size = s1_size * get_group_size(state) - split_s1g_idx * state.m_base
    result.max_s2_split_num = max(result.max_s2_split_num, assign_ctx.cur_kv_split_part)
    result.fd_bn2_idx[result.fd_task_num] = result.bn2_end[assign_ctx.cur_core_idx - 1]
    result.fd_s1g_idx[result.fd_task_num] = result.gs1_end[assign_ctx.cur_core_idx - 1]
    result.fd_workspace_idx[result.fd_task_num] = assign_ctx.pre_fd_data_num
    result.fd_s2_split_num[result.fd_task_num] = assign_ctx.cur_kv_split_part
    result.fd_m_size[result.fd_task_num] = cur_fd_s1g_size
    result.fd_task_num += 1
    return 0


def schedule_fd(aiv_num, result):
    # Distribute FD reduction tasks evenly across AIV cores.
    #
    # Estimate cores per task from task load / global average load, then clamp:
    # at least one core for the task and at most free cores plus one. Round up
    # M rows per core, then derive core count again to avoid empty cores.
    if result.fd_task_num == 0:
        return 0
    total_fd_load = 0
    for i in range(0, result.fd_task_num):
        total_fd_load += result.fd_s2_split_num[i] * result.fd_m_size[i]
    empty_vec_num = aiv_num - result.fd_task_num
    average_load = ceil_div(total_fd_load, aiv_num)
    cur_core_index = 0
    for i in range(0, result.fd_task_num):
        # All cores are in use; do not subdivide further. Give this task one core.
        if empty_vec_num == 0:
            result.fd_task_idx[cur_core_index] = i
            result.fd_m_start[cur_core_index] = 0
            result.fd_m_len[cur_core_index] = result.fd_m_size[i]
            cur_core_index += 1
            continue
        cur_vec_num = result.fd_s2_split_num[i] * result.fd_m_size[i] // average_load
        cur_vec_num = max(cur_vec_num, 1)
        cur_avg_m_size = ceil_div(result.fd_m_size[i], cur_vec_num)
        # Recompute core count from rounded-up rows per core to avoid empty cores.
        cur_vec_num = ceil_div(result.fd_m_size[i], cur_avg_m_size)
        cur_vec_num = min(
            cur_vec_num, empty_vec_num + 1
        )  # Include the task's own core.
        for vid in range(0, cur_vec_num):
            result.fd_task_idx[cur_core_index] = i
            result.fd_m_start[cur_core_index] = vid * cur_avg_m_size
            if vid < cur_vec_num - 1:
                result.fd_m_len[cur_core_index] = cur_avg_m_size
            else:
                result.fd_m_len[cur_core_index] = (
                    result.fd_m_size[i] - vid * cur_avg_m_size
                )
            cur_core_index += 1
        empty_vec_num -= cur_vec_num - 1  # Free cores exclude the task's own core.
    result.used_vec_num = cur_core_index
    return 0


# ---------------------------------------------------------------------------
# Per-core candidate buffers
#
# Both candidates reuse their arrays across sections.
# ---------------------------------------------------------------------------


def alloc_result(result, aic_num, aiv_num):
    # Allocate result buffers by core count: first nine arrays by AICs, last three by AIVs.
    result.bn2_end = zeros(U32, aic_num)
    result.gs1_end = zeros(U32, aic_num)
    result.s2_end = zeros(U32, aic_num)
    result.first_fd_workspace_idx = zeros(U32, aic_num)
    result.fd_bn2_idx = zeros(U32, aic_num)
    result.fd_s1g_idx = zeros(U32, aic_num)
    result.fd_workspace_idx = zeros(U32, aic_num)
    result.fd_s2_split_num = zeros(U32, aic_num)
    result.fd_m_size = zeros(U32, aic_num)
    result.fd_task_idx = zeros(U32, aiv_num)
    result.fd_m_start = zeros(U32, aiv_num)
    result.fd_m_len = zeros(U32, aiv_num)
    return 0


def clear_result(result):
    # Clear the result for reuse by the next scheduling pass.
    #
    # Only fd_task_num, used_vec_num, and max_s2_split_num are read before overwrite.
    # Other scalars are overwritten by callers or schedule_fa. Clear all arrays
    # because all-masked or all-empty sections write only a few fields before use.
    result.max_cost = 0
    result.used_core_num = 0
    result.max_s2_split_num = 0
    result.fd_task_num = 0
    result.used_vec_num = 0
    result.bn2_end.reset(0, len(result.bn2_end))
    result.gs1_end.reset(0, len(result.gs1_end))
    result.s2_end.reset(0, len(result.s2_end))
    result.first_fd_workspace_idx.reset(0, len(result.first_fd_workspace_idx))
    result.fd_bn2_idx.reset(0, len(result.fd_bn2_idx))
    result.fd_s1g_idx.reset(0, len(result.fd_s1g_idx))
    result.fd_workspace_idx.reset(0, len(result.fd_workspace_idx))
    result.fd_s2_split_num.reset(0, len(result.fd_s2_split_num))
    result.fd_m_size.reset(0, len(result.fd_m_size))
    result.fd_task_idx.reset(0, len(result.fd_task_idx))
    result.fd_m_start.reset(0, len(result.fd_m_start))
    result.fd_m_len.reset(0, len(result.fd_m_len))
    return 0


def schedule_fa(state, grid, cost_info, config, result):
    # Run one full greedy assignment for a fixed core count into result.
    #
    # config.with_fd controls splitting rows along S2. Each core's cost_limit is
    # remaining cost / remaining cores, raised to the minimum work-unit cost
    # that core must take: a row without fd or a block with fd.
    if config.core_num == 0:
        return 0
    result.max_cost = 0
    result.used_core_num = 0
    assign_ctx = AssignCtx()
    assign_ctx.cur_section_idx = config.section_idx
    if config.section_idx == 0:
        assign_ctx.cur_batch_idx = 0
        assign_ctx.cur_bn2_idx = 0
    else:
        assign_ctx.cur_batch_idx = floor_div_safe(
            grid.section_bn2_idx[config.section_idx - 1], state.schedule_head_count, 0
        )
        assign_ctx.cur_bn2_idx = grid.section_bn2_idx[config.section_idx - 1]
    assign_ctx.cur_s1g_idx = 0
    assign_ctx.cur_core_idx = 0
    assign_ctx.cur_kv_split_part = 1
    assign_ctx.pre_fd_data_num = 0
    assign_ctx.is_finished = 0
    assign_ctx.unassigned_cost = cost_info.section_cost[config.section_idx]
    assign_ctx.bn2_cost = cost_info.bn2_cost_per_batch[assign_ctx.cur_batch_idx]
    assign_ctx.bn2_block = cost_info.bn2_block_per_batch[assign_ctx.cur_batch_idx]
    calc_batch_cache(assign_ctx.cur_batch_idx, state, grid, assign_ctx.batch_cache)
    calc_s1g_cache(
        assign_ctx.cur_s1g_idx,
        state,
        grid,
        assign_ctx.batch_cache,
        assign_ctx.s1g_cache,
    )
    assign_ctx.cur_s2_idx = assign_ctx.s1g_cache.s2_start

    for i in range(0, config.core_num):
        if assign_ctx.is_finished != 0 or assign_ctx.unassigned_cost <= 0:
            break
        assign_ctx.cur_core_idx = i
        result.first_fd_workspace_idx[i] = (
            assign_ctx.pre_fd_data_num + assign_ctx.cur_kv_split_part - 1
        )
        assign_ctx.core_cache.cost = 0
        assign_ctx.core_cache.block = 0
        assign_ctx.core_cache.cost_limit = assign_ctx.unassigned_cost // (
            config.core_num - i
        )
        if config.with_fd == 0:
            # No splitting: this core must take at least the current full row.
            assign_ctx.core_cache.cost_limit = max(
                assign_ctx.core_cache.cost_limit, assign_ctx.s1g_cache.s1g_cost
            )
        else:
            # Splitting allowed: this core must take at least the current block.
            cur_cost = assign_ctx.s1g_cache.s1g_normal_block_cost
            if assign_ctx.cur_s2_idx == assign_ctx.s1g_cache.s2_end - 1:
                cur_cost = assign_ctx.s1g_cache.s1g_last_block_cost
            assign_ctx.core_cache.cost_limit = max(
                assign_ctx.core_cache.cost_limit, cur_cost
            )
        assign_by_batch(state, grid, cost_info, assign_ctx)
        assign_by_row(state, grid, cost_info, assign_ctx)
        if config.with_fd != 0:
            assign_by_block(state, grid, cost_info, assign_ctx)
        result.bn2_end[i] = assign_ctx.cur_bn2_idx
        result.gs1_end[i] = assign_ctx.cur_s1g_idx
        result.s2_end[i] = assign_ctx.cur_s2_idx
        result.max_cost = max(result.max_cost, assign_ctx.core_cache.cost)
        assign_ctx.unassigned_cost -= assign_ctx.core_cache.cost
        if config.with_fd != 0 and is_need_record_fd_info(assign_ctx, result) != 0:
            record_fd_info(state, grid, assign_ctx, result)
            assign_ctx.pre_fd_data_num += assign_ctx.cur_kv_split_part
            assign_ctx.cur_kv_split_part = 1
        # Stopping inside a row means the row was split; increment the split count.
        if (
            assign_ctx.cur_s2_idx > assign_ctx.s1g_cache.s2_start
            and assign_ctx.cur_s2_idx <= assign_ctx.s1g_cache.s2_end
        ):
            assign_ctx.cur_kv_split_part += 1
    result.used_core_num = assign_ctx.cur_core_idx + 1
    return 0


def check_choose_with_fd(state, no_fd, with_fd):
    # Choose between the no-fd and fd schedules.
    #
    # For one section, skip fd if the slowest no-fd core is already cheap
    # (within fd_least_blocks full blocks); otherwise require fd savings above
    # fd_tolerance_blocks full blocks.
    full_block_cost = calc_cost(state.m_base, state.s2_base)
    if no_fd.max_cost <= state.fd_least_blocks * full_block_cost:
        return 0
    # Use subtraction to avoid int64 overflow.
    fd_tolerance = state.fd_tolerance_blocks * full_block_cost
    if no_fd.max_cost - fd_tolerance > with_fd.max_cost:
        return 1
    return 0


def schedule_section(state, grid, cost_info, section_idx, no_fd, with_fd):
    # Compare row-preserving and S2-splitting plans at the same core count.
    config = FaConfig()
    config.section_idx = section_idx
    config.core_num = min(state.aic_core_max, cost_info.section_block_num[section_idx])
    config.with_fd = 0
    schedule_fa(state, grid, cost_info, config, no_fd)
    config.with_fd = 1
    clear_result(with_fd)
    schedule_fa(state, grid, cost_info, config, with_fd)
    return check_choose_with_fd(state, no_fd, with_fd)


def write_fa_metadata(args, section_idx, core_idx, field_idx, value):
    # Write the FA region: metadata[head][section][core][field].
    args.metadata[
        METADATA_STRIDE
        + section_idx * args.cores * METADATA_STRIDE
        + core_idx * METADATA_STRIDE
        + field_idx
    ] = value
    return 0


def write_fd_metadata(args, section_num, section_idx, vec_idx, field_idx, value):
    # Write the FD region after FA: metadata[head][FA region][section][vector][field].
    args.metadata[
        METADATA_STRIDE
        + section_num * args.cores * METADATA_STRIDE
        + section_idx * args.vectors * METADATA_STRIDE
        + vec_idx * METADATA_STRIDE
        + field_idx
    ] = value
    return 0


def gen_section_fa(args, result, section_idx, prev_section):
    # Write one section's FA region.
    #
    # Each core records its own [start, end). The first start uses the preceding
    # core's end; core zero uses the prior section's last end (or zero).
    for i in range(0, result.used_core_num):
        if i == 0:
            write_fa_metadata(args, section_idx, i, FA_START_BN, prev_section.bn2_end)
            write_fa_metadata(args, section_idx, i, FA_START_M, prev_section.gs1_end)
            write_fa_metadata(args, section_idx, i, FA_START_S2, prev_section.s2_end)
        else:
            write_fa_metadata(args, section_idx, i, FA_START_BN, result.bn2_end[i - 1])
            write_fa_metadata(args, section_idx, i, FA_START_M, result.gs1_end[i - 1])
            write_fa_metadata(args, section_idx, i, FA_START_S2, result.s2_end[i - 1])
        write_fa_metadata(args, section_idx, i, FA_END_BN, result.bn2_end[i])
        write_fa_metadata(args, section_idx, i, FA_END_M, result.gs1_end[i])
        write_fa_metadata(args, section_idx, i, FA_END_S2, result.s2_end[i])
        write_fa_metadata(
            args,
            section_idx,
            i,
            FA_FIRST_SLOT,
            result.first_fd_workspace_idx[i],
        )
    prev_section.bn2_end = result.bn2_end[result.used_core_num - 1]
    prev_section.gs1_end = result.gs1_end[result.used_core_num - 1]
    prev_section.s2_end = result.s2_end[result.used_core_num - 1]
    return 0


def gen_section_fd(args, section_num, result, section_idx):
    # Write one section's FD region: one reduction-task record per AIV core.
    for i in range(0, result.used_vec_num):
        task = result.fd_task_idx[i]
        write_fd_metadata(
            args, section_num, section_idx, i, FD_BN, result.fd_bn2_idx[task]
        )
        write_fd_metadata(
            args, section_num, section_idx, i, FD_M, result.fd_s1g_idx[task]
        )
        write_fd_metadata(
            args,
            section_num,
            section_idx,
            i,
            FD_FIRST_SLOT,
            result.fd_workspace_idx[task],
        )
        write_fd_metadata(
            args,
            section_num,
            section_idx,
            i,
            FD_PARTS,
            result.fd_s2_split_num[task],
        )
        write_fd_metadata(
            args, section_num, section_idx, i, FD_ROW_BEGIN, result.fd_m_start[i]
        )
        write_fd_metadata(
            args, section_num, section_idx, i, FD_ROWS, result.fd_m_len[i]
        )
    return 0


def generate_metadata(args):
    # Build sectioned FA/FD records in the consumer's fixed metadata ABI.

    seq_capacity = args.batch
    section_capacity = args.sections
    state = FaState()
    state.q_seq = array(I64, seq_capacity)
    state.kv_seq = array(I64, seq_capacity)
    for b in range(args.batch):
        state.q_seq.append(args.q[b])
        state.kv_seq.append(args.kv[b])
    state.aic_core_max = args.cores
    state.aiv_core_max = args.vectors
    state.batch_size = args.batch
    state.schedule_head_count = args.heads
    state.group = args.group
    state.head_dim = args.head_dim
    state.mask_mode = args.sparse
    state.m_base = args.tile_m
    state.s2_base = args.macro_n
    state.fa_tolerance_ratio = FA_TOLERANCE_RATIO
    state.fd_tolerance_blocks = FD_TOLERANCE_BLOCKS
    state.fd_least_blocks = FD_LEAST_BLOCKS
    state.l2_bytes = L2_BYTES
    if args.sections == 1:
        state.l2_bytes = 0
    grid = GridInfo()
    grid.m_block_num = zeros(U32, seq_capacity)
    grid.s2_block_num = zeros(U32, seq_capacity)
    grid.m_tail_size = zeros(U32, seq_capacity)
    grid.s2_tail_size = zeros(U32, seq_capacity)
    grid.section_bn2_idx = array(U32, section_capacity)
    cost_info = CostInfo()
    cost_info.bn2_cost_per_batch = zeros(I64, seq_capacity)
    cost_info.bn2_block_per_batch = zeros(U32, seq_capacity)
    cost_info.bn2_last_block_cost_per_batch = zeros(I64, seq_capacity)
    cost_info.section_block_num = zeros(U32, section_capacity)
    cost_info.section_cost = zeros(I64, section_capacity)
    result = SKResult()
    alloc_result(result, args.cores, args.vectors)
    with_fd = SKResult()
    alloc_result(with_fd, args.cores, args.vectors)

    # Direct kernel callers must also provide usable tile/core geometry.
    if state.m_base == 0 or state.s2_base == 0 or state.fa_tolerance_ratio == 0:
        return 1
    if args.vectors < args.cores:
        return 1

    calc_grid_info(state, grid)
    calc_grid_info_section(state, grid)

    section_num = grid.section_num
    if grid.is_empty != 0:
        section_num = 1

    # Header, FA and FD regions are adjacent; clear every active record.
    fa_words = section_num * args.cores * METADATA_STRIDE
    fd_words = section_num * args.vectors * METADATA_STRIDE
    for i in range(0, METADATA_STRIDE + fa_words + fd_words):
        args.metadata[i] = 0

    prev_section = PrevSec()
    has_fd = 0
    if grid.is_empty != 0:
        # All batches are empty: one trivial section with one idle core.
        clear_result(result)
        result.max_cost = 0
        result.used_core_num = 1
        result.bn2_end[0] = state.batch_size * state.schedule_head_count
        gen_section_fa(args, result, 0, prev_section)
    else:
        calc_cost_info(state, grid, cost_info)
        for section_idx in range(0, section_num):
            clear_result(result)
            result.max_cost = 0
            if cost_info.section_block_num[section_idx] == 0:
                result.used_core_num = 1
                result.bn2_end[0] = grid.section_bn2_idx[section_idx]
                gen_section_fa(args, result, section_idx, prev_section)
            else:
                choose_fd = schedule_section(
                    state, grid, cost_info, section_idx, result, with_fd
                )
                if choose_fd:
                    schedule_fd(state.aiv_core_max, with_fd)
                    gen_section_fa(args, with_fd, section_idx, prev_section)
                    gen_section_fd(args, section_num, with_fd, section_idx)
                    if with_fd.used_vec_num > 0:
                        has_fd = 1
                else:
                    gen_section_fa(args, result, section_idx, prev_section)

    args.metadata[SECTION_COUNT] = section_num
    args.metadata[HAS_FD] = has_fd
    args.metadata[HEADER_TILE_M] = state.m_base
    args.metadata[HEADER_MACRO_N] = state.s2_base
    args.metadata[HEADER_CORES] = args.cores
    args.metadata[HEADER_VECTORS] = args.vectors
    return 0


# ---------------------------------------------------------------------------
# Kernel entry point
#
# Kernel-local struct declarations are registered in AST statement order:
#   * They must be inside the function; module-level classes are not registered.
#   * They must precede their first use.
#   * Dependencies must be declared first (AssignCtx depends on BatchCache).
# These classes are interpreter-only declarations; CPython never executes their bodies.
# ---------------------------------------------------------------------------


@aicpu_kernel
def fd_metadata(args: MetadataArgs):
    class FaState:
        aic_core_max: U32
        aiv_core_max: U32
        batch_size: U32
        group: U32
        schedule_head_count: U32
        head_dim: U32
        mask_mode: U32
        m_base: U32
        s2_base: U32
        fa_tolerance_ratio: U32
        fd_tolerance_blocks: U32
        fd_least_blocks: U32
        l2_bytes: I64
        q_seq: array(I64)
        kv_seq: array(I64)

    class GridInfo:
        m_block_num: array(U32)
        s2_block_num: array(U32)
        m_tail_size: array(U32)
        s2_tail_size: array(U32)
        section_bn2_idx: array(U32)
        section_num: U32
        is_empty: U32 = 1

    class CostInfo:
        bn2_cost_per_batch: array(I64)
        bn2_block_per_batch: array(U32)
        bn2_last_block_cost_per_batch: array(I64)
        section_block_num: array(U32)
        section_cost: array(I64)

    # Cost table: first index is M (full/tail row), second is S2 (full/tail block).
    class BatchCache:
        batch_idx: U32
        s1_size: U32
        s2_size: U32
        pre_token_left_up: I64
        next_token_left_up: I64
        cost_normal_normal: I64  # Full M row x full S2 block
        cost_tail_normal: I64  # Tail M row x full S2 block
        cost_normal_tail: I64  # Full M row x tail S2 block
        cost_tail_tail: I64  # Tail M row x tail S2 block

    class S1GCache:
        batch_idx: U32
        s1g_idx: U32
        s2_start: U32
        s2_end: U32
        s1g_cost: I64
        s1g_last_block_cost: I64
        s1g_block: U32
        s1g_normal_block_cost: I64

    class CoreCache:
        cost_limit: I64
        cost: I64
        block: U32

    class AssignCtx:
        cur_section_idx: U32
        cur_batch_idx: U32
        cur_bn2_idx: U32
        cur_s1g_idx: U32
        cur_s2_idx: U32
        cur_core_idx: U32
        unassigned_cost: I64
        cur_kv_split_part: U32
        pre_fd_data_num: U32
        bn2_cost: I64
        bn2_block: U32
        is_finished: U32
        batch_cache: BatchCache
        s1g_cache: S1GCache
        core_cache: CoreCache

    # with_fd distinguishes schedules that may or may not split rows along S2.
    # Both candidate policies are evaluated before selecting the section plan.
    class FaConfig:
        with_fd: U32
        core_num: U32
        section_idx: U32

    class SKResult:
        max_cost: I64
        used_core_num: U32
        max_s2_split_num: U32
        fd_task_num: U32
        used_vec_num: U32
        # Allocate the following nine arrays by AIC core count.
        bn2_end: array(U32)
        gs1_end: array(U32)
        s2_end: array(U32)
        first_fd_workspace_idx: array(U32)
        fd_bn2_idx: array(U32)
        fd_s1g_idx: array(U32)
        fd_workspace_idx: array(U32)
        fd_s2_split_num: array(U32)
        fd_m_size: array(U32)
        # Allocate the following three arrays by AIV core count.
        fd_task_idx: array(U32)
        fd_m_start: array(U32)
        fd_m_len: array(U32)

    # End position of the previous section's last core, chaining sections.
    class PrevSec:
        bn2_end: U32
        gs1_end: U32
        s2_end: U32

    return generate_metadata(args)


_METADATA_PROGRAMS = {}
_METADATA_PROGRAM_LOCK = threading.Lock()


def _metadata_source_key(sources, toolchain):
    """Fingerprint generated sources, the Python toolchain and its actual compiler."""
    home = Path(os.environ.get("ASCEND_HOME_PATH", "")).expanduser().resolve()
    compiler = home / "bin/bisheng"
    identity = json.dumps(sources, sort_keys=True).encode()
    identity += Path(toolchain.__file__).read_bytes()
    identity += str(home).encode()
    if compiler.is_file():
        info = compiler.stat()
        identity += repr(
            (str(compiler.resolve()), info.st_size, info.st_mtime_ns)
        ).encode()
    return hashlib.sha256(identity).hexdigest()


def _valid_cached_library(library, manifest, source_key):
    if not library.is_file() or not manifest.is_file():
        return False
    try:
        record = json.loads(manifest.read_text())
        return (
            isinstance(record, dict)
            and record.get("source_key") == source_key
            and record.get("sha256") == hashlib.sha256(library.read_bytes()).hexdigest()
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        # A process interrupted during a previous build must not poison reuse.
        return False


def _write_build_manifest(manifest, library, source_key):
    temporary = manifest.with_name(manifest.name + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(
                source_key=source_key,
                sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            )
        )
    )
    temporary.replace(manifest)


def _cached_metadata_program(cache_root):
    import fcntl
    import cannbotdsl.aicpu.toolchain as toolchain

    sources = fd_metadata.generate_sources(launch_mode="interface")
    source_key = _metadata_source_key(sources, toolchain)
    work = cache_root / "gqa_mxfp8_metadata" / source_key
    work.mkdir(parents=True, exist_ok=True)
    library, manifest = work / "fd_metadata.so", work / "build.json"
    with (work / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _valid_cached_library(library, manifest, source_key):
            return CompiledAicpuKernel(
                str(library),
                fd_metadata.spec,
                name="fd_metadata",
                launch_mode="interface",
            )
        program = compile_aicpu_kernel(
            fd_metadata, workdir=str(work), launch_mode="interface"
        )
        _write_build_manifest(manifest, library, source_key)
        return program


def _metadata_program():
    """Reuse the AICPU program in this process and use disk cache when configured."""
    process_id = os.getpid()
    if process_id in _METADATA_PROGRAMS:
        return _METADATA_PROGRAMS[process_id][1]
    with _METADATA_PROGRAM_LOCK:
        if process_id in _METADATA_PROGRAMS:
            return _METADATA_PROGRAMS[process_id][1]
        cache_dir = os.environ.get("CANNBOTDSL_CACHE_DIR")
        directory = None
        if cache_dir:
            program = _cached_metadata_program(Path(cache_dir).expanduser())
        else:
            directory = tempfile.TemporaryDirectory(prefix="gqa_mxfp8_metadata_")
            try:
                program = compile_aicpu_kernel(
                    fd_metadata, workdir=directory.name, launch_mode="interface"
                )
            except BaseException:
                directory.cleanup()
                raise
        # The launcher loads its library lazily; keep the directory alive.
        _METADATA_PROGRAMS[process_id] = (directory, program)
        return program


def _length_signature(tensor, epoch):
    try:
        version = tensor._version
    except RuntimeError:
        if epoch is None:
            raise ValueError(
                "inference length tensors require an explicit lengths_epoch"
            ) from None
        version = None
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        version,
    )


def _require_integer(name, value, *, minimum=1, maximum=UINT32_MAX):
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Metadata {name} must be int, got {type(value).__name__}.")
    if not minimum <= value <= maximum:
        raise ValueError(
            f"Metadata {name} must be in [{minimum}, {maximum}], got {value}."
        )


def _validate_epoch(lengths_epoch):
    if lengths_epoch is not None:
        _require_integer("lengths_epoch", lengths_epoch, minimum=0, maximum=INT64_MAX)


def _immutable_contract(contract):
    contract = tuple(contract)

    def check(value):
        if isinstance(value, torch.Tensor):
            raise TypeError(
                "Metadata contract must contain immutable metadata, not tensors."
            )
        if isinstance(value, (tuple, frozenset)):
            for item in value:
                check(item)

    check(contract)
    try:
        hash(contract)
    except TypeError:
        raise TypeError(
            "Metadata contract must contain immutable hashable values."
        ) from None
    return contract


def _validate_config(config, batch):
    config = dict(config)
    if set(config) != _CONFIG_FIELDS:
        raise ValueError(f"Metadata config requires fields {sorted(_CONFIG_FIELDS)}.")
    for name, value in config.items():
        _require_integer(name, value, minimum=0 if name == "sparse" else 1)
    if config["batch"] != batch:
        raise ValueError("Metadata batch must match the length-vector size.")
    if config["batch"] * config["heads"] > UINT32_MAX:
        raise ValueError("Metadata batch/head index must fit uint32.")
    if config["sparse"] not in SUPPORTED_MASK_MODES:
        raise ValueError(f"Metadata sparse must be in {SUPPORTED_MASK_MODES}.")
    if config["vectors"] < config["cores"]:
        raise ValueError("Metadata vectors must be at least the core count.")
    scratch = _scratch_bytes(config)
    if scratch > AICPU_ARRAY_ARENA_BYTES:
        raise ValueError(
            f"Sectioned scheduling needs {scratch} bytes of AICPU array scratch; capacity is {AICPU_ARRAY_ARENA_BYTES}."
        )
    return config


def _scratch_bytes(config):
    """Conservative 16-byte-aligned budget for generate_metadata's arrays.

    Per batch: 4 int64 + 5 uint32 arrays; per section: 1 int64 + 2 uint32.
    Two candidates each own 9 AIC-sized and 3 AIV-sized uint32 arrays.
    """

    def aligned(size):
        return (
            (size + ARRAY_ALIGNMENT_BYTES - 1)
            // ARRAY_ALIGNMENT_BYTES
            * ARRAY_ALIGNMENT_BYTES
        )

    batch, sections = config["batch"], config["sections"]
    return (
        4 * aligned(INT64_BYTES * batch)
        + 5 * aligned(UINT32_BYTES * batch)
        + aligned(INT64_BYTES * sections)
        + 2 * aligned(UINT32_BYTES * sections)
        + 2
        * (
            9 * aligned(UINT32_BYTES * config["cores"])
            + 3 * aligned(UINT32_BYTES * config["vectors"])
        )
    )


class AttentionMetadataState:
    """Caller-owned metadata cache; pass to npu_flash_atten_mxfp8_fullquant.

    Changed lengths or geometry allocate a fresh table for new readers.
    """

    ABI = 3

    def __init__(self):
        self._lock = threading.Lock()
        self._signature = None
        self.tensor = None
        self.generations = 0
        self._owners = None
        self._ready = None
        self._producer_stream = None

    def resolve(self, q, kv, config, *, contract=(), lengths_epoch=None):
        """Reuse an immutable geometry/version key and order readers by stream.

        Length values must stay within the bounds/capacities supplied to
        prepare_metadata. Advance the nonnegative integer lengths_epoch after
        writes invisible to Tensor._version. Contract entries are immutable,
        hashable host metadata; Tensor values cannot be cache-key entries.
        """
        _validate_length_vectors(q, kv)
        _validate_epoch(lengths_epoch)
        contract = _immutable_contract(contract)
        config = _validate_config(config, q.numel())
        if q.device.type == "meta":
            return torch.empty(_table_words(config), dtype=torch.int64, device=q.device)
        signature = (
            self.ABI,
            tuple(sorted(config.items())),
            contract,
            lengths_epoch,
            _length_signature(q, lengths_epoch),
            _length_signature(kv, lengths_epoch),
        )
        with torch.npu.device(q.device), self._lock:
            stream = torch.npu.current_stream(q.device)
            raw_stream = current_raw_stream(q.device.index or 0)
            if signature != self._signature:
                program = _metadata_program()
                table = torch.empty(
                    _table_words(config), dtype=torch.int64, device=q.device
                )
                program.launch(
                    raw_stream,
                    q=q.data_ptr(),
                    kv=kv.data_ptr(),
                    metadata=table.data_ptr(),
                    **config,
                )
                ready = torch.npu.Event()
                ready.record(stream)
                self.tensor, self._ready = table, ready
                self._producer_stream = raw_stream
                self._owners = (q, kv)
                self._signature = signature
                self.generations += 1
            elif raw_stream != self._producer_stream:
                stream.wait_event(self._ready)
            for tensor in (q, kv, self.tensor):
                tensor.record_stream(stream)
            return self.tensor


def max_sections(batch, heads, max_q, group, tile_m, kv_capacity, head_dim, cores):
    """Upper-bound section capacity without reading device length values."""
    for name, value in (
        ("batch", batch),
        ("heads", heads),
        ("group", group),
        ("tile_m", tile_m),
        ("head_dim", head_dim),
        ("cores", cores),
    ):
        _require_integer(name, value)
    for name, value in (("max_q", max_q), ("kv_capacity", kv_capacity)):
        _require_integer(name, value, minimum=0, maximum=INT64_MAX)
    bn = batch * heads
    if bn > UINT32_MAX:
        raise ValueError("Metadata batch/head index must fit uint32.")
    head_bytes = 2 * head_dim * (max_q + kv_capacity)
    if max_q * group <= tile_m or head_bytes <= L2_BYTES // cores:
        return 1
    # Consecutive sections jointly exceed the L2 budget; cap by flattened heads.
    return min(bn, 1 + 2 * (bn * head_bytes // L2_BYTES))


def _table_words(config):
    return METADATA_STRIDE * (
        1 + config["sections"] * (config["cores"] + config["vectors"])
    )


def prepare_metadata(
    q,
    kv,
    *,
    heads,
    group,
    max_q,
    tile_m,
    macro_n,
    sparse,
    cores,
    vectors,
    head_dim,
    kv_capacity,
    state=None,
    lengths_epoch=None,
):
    """Submit a sectioned schedule and return table plus FD slot capacity.

    heads is the scheduling-head count: Q heads for Prefill, KV heads for
    Decode; group is the number of Q heads merged into each scheduling head.
    max_q and kv_capacity must bound device lengths. The caller owns length
    values; this function only reads tensor metadata. Persistent state requires
    a nonnegative integer epoch for inference tensors and external writes.
    """
    _validate_length_vectors(q, kv)
    _validate_epoch(lengths_epoch)
    _require_integer("macro_n", macro_n)
    _require_integer("vectors", vectors)
    _require_integer("sparse", sparse, minimum=0)
    if sparse not in SUPPORTED_MASK_MODES:
        raise ValueError(f"Metadata sparse must be in {SUPPORTED_MASK_MODES}.")
    sections = max_sections(
        q.numel(), heads, max_q, group, tile_m, kv_capacity, head_dim, cores
    )
    if state is None:
        state = AttentionMetadataState()
        lengths_epoch = 0
    if not isinstance(state, AttentionMetadataState):
        raise TypeError("metadata_state must be AttentionMetadataState")
    config = dict(
        batch=q.numel(),
        heads=heads,
        group=group,
        tile_m=tile_m,
        macro_n=macro_n,
        sparse=sparse,
        cores=cores,
        vectors=vectors,
        head_dim=head_dim,
        sections=sections,
    )
    table = state.resolve(q, kv, config, lengths_epoch=lengths_epoch)
    slots = 2 * cores * sections if cores > 1 and kv_capacity > macro_n else 0
    return table, slots
