# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Flash MLA metadata AICPU operator with inlined SectionStreamK.

The module follows the single-file ``flash_attn_metadata`` architecture: the
complete load balancer runs inside ``@aicpu_kernel`` on the current NPU stream
and writes the flat metadata ABI directly in GM.  No host scheduler bridge
or host-to-device metadata copy is used.
"""

import importlib.util
import os
import sys
import tempfile
from functools import lru_cache

import torch

from cannbotdsl.aicpu import I32, I64, U32, aicpu_kernel, gm_in, gm_out

AIC_CORE_NUM = 36
AIV_CORE_NUM = 72
HEAD_METADATA_STRIDE = 16
FA_METADATA_STRIDE = 16
FD_METADATA_STRIDE = 16
HEAD_DIM_QK = 576
HEAD_DIM_V = 512
MLA_M_BASE = 96
MLA_S2_BASE = 112
COST_M_ALIGNMENT = 16
COST_S2_ALIGNMENT = 64
METADATA_ALIGNMENT_WORDS = 4096
MAX_BATCH_SIZE = 65536

# head metadata index
HEAD_SECTION_NUM_INDEX = 0
HEAD_IS_FD_INDEX = 1
HEAD_M_BASE_SIZE_INDEX = 2
HEAD_S2_BASE_SIZE_INDEX = 3
HEAD_AIC_NUM_INDEX = 4
HEAD_AIV_NUM_INDEX = 5
HEAD_OUTPUT_LAYOUT_INDEX = 6
HEAD_NEED_INIT_INDEX = 7

# Sparse mode constants follow the definitions in load_balance_common.h.
SPARSE_DEFAULT_MASK = 0
SPARSE_ALL_MASK = 1
SPARSE_LEFT_UP_CAUSAL = 2
SPARSE_RIGHT_DOWN_CAUSAL = 3
SPARSE_BAND = 4
SPARSE_BUTT = 5

UINT32_MAX = 4294967295
INT64_MAX = 9223372036854775807
INT64_MIN = -9223372036854775808

# SectionStreamKParam defaults (flash_attn_metadata_aicpu.cpp InitLoadBalanceParams)
L2_BYTE = 100663296  # 96 * 1024 * 1024
FA_TOLERANCE_RATIO = 2
FD_TOLERANCE = 10
FD_LEAST_BLOCK = 3

# Layout enum values from load_balance_common.h.
LAYOUT_BSND = 0
LAYOUT_BNSD = 1
LAYOUT_BSH = 2
LAYOUT_NBSD = 3
LAYOUT_TND = 4
LAYOUT_NTD = 5

_LAYOUT_CODES = {
    "BSND": LAYOUT_BSND,
    "BNSD": LAYOUT_BNSD,
    "BSH": LAYOUT_BSH,
    "NBSD": LAYOUT_NBSD,
    "TND": LAYOUT_TND,
    "NTD": LAYOUT_NTD,
    "PA_NZ": 6,
}

_PLATFORM_INFO_GETTER = None


def _load_platform_info_getter():
    """Resolve PR160's platform query, including old installed wheels."""
    global _PLATFORM_INFO_GETTER
    if _PLATFORM_INFO_GETTER is not None:
        return _PLATFORM_INFO_GETTER
    try:
        from cannbotdsl import get_platform_info

        _PLATFORM_INFO_GETTER = get_platform_info
        return _PLATFORM_INFO_GETTER
    except ImportError:
        pass

    repo_root = os.environ.get("CANNBOTDSL_REPO_ROOT") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    platform_path = os.path.join(repo_root, "python", "cannbotdsl", "platform.py")
    if not os.path.isfile(platform_path):
        return None
    module_name = "_cannbotdsl_platform_flash_attn_metadata"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, platform_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    _PLATFORM_INFO_GETTER = module.get_platform_info
    return _PLATFORM_INFO_GETTER


def get_effective_core_counts(stream=None):
    """Return the AIC/AIV counts available to the metadata launch stream."""
    props = torch.npu.get_device_properties(torch.npu.current_device())
    cube = int(getattr(props, "cube_core_num", 0))
    vector = int(getattr(props, "vector_core_num", 0)) or 2 * cube
    getter = _load_platform_info_getter()
    if getter is not None:
        try:
            info = getter(stream=stream)
            cube = int(info.cube_core_num) or cube
            vector = int(info.vector_core_num) or vector
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
    cube = min(cube, AIC_CORE_NUM)
    vector = min(vector, AIV_CORE_NUM)
    if cube <= 0 or vector < cube:
        raise RuntimeError(
            f"invalid effective NPU core counts: AIC={cube}, AIV={vector}"
        )
    return cube, vector


def _layout_code(layout: str) -> int:
    """Host-side layout string -> Layout enum value (BUTT=9 if unknown).

    The metadata scheduler only needs the logical BNSD geometry for the
    correctness-first PA bridge; physical page layout is consumed by flash_attn.
    """
    if layout in ("PA_BBND", "PA_BNBD"):
        layout = "BNSD"
    return _LAYOUT_CODES.get(layout, 9)


class _FlashMlaMetadataArgs:
    cu_seqlens_q: gm_in(I32)
    cache_seqlens: gm_in(I32)
    seqused_q: gm_in(I32)
    metadata: gm_out(U32)
    cu_q_len: U32
    cache_len: U32
    sq_len: U32
    num_heads_q: U32
    num_heads_kv: U32
    head_dim_qk: U32
    head_dim_v: U32
    aic_core_num: U32
    aiv_core_num: U32
    max_seqlen_q: I32
    max_seqlen_kv: I32
    mask_mode: I32
    layout_q: U32


# ---------------------------------------------------------------------------
# BaseInfo getters (base_info.h)
# ---------------------------------------------------------------------------


def get_group_size(state):
    return floor_div_safe(state.q_head_num, state.kv_head_num, 1)


def get_q_seq_size_at(state, b_idx):
    if len(state.q_seq) == 0:
        return state.q_seq_size
    return state.q_seq[b_idx]


def get_kv_seq_size_at(state, b_idx):
    if len(state.kv_seq) == 0:
        return state.kv_seq_size
    return state.kv_seq[b_idx]


def get_sparse_mode(state):
    if state.atten_mask_flag == 0:
        return SPARSE_BUTT
    if state.sparse_mode_u > SPARSE_BUTT:
        return SPARSE_BUTT
    return state.sparse_mode_u


def get_pre_token_left_up(state, query_seq, kv_seq):
    mode = get_sparse_mode(state)
    if mode == SPARSE_BAND:
        return query_seq - kv_seq + state.pre_token
    return state.pre_token


def get_next_token_left_up(state, query_seq, kv_seq):
    mode = get_sparse_mode(state)
    if (
        mode == SPARSE_DEFAULT_MASK
        or mode == SPARSE_ALL_MASK
        or mode == SPARSE_LEFT_UP_CAUSAL
    ):
        return state.next_token
    if mode == SPARSE_RIGHT_DOWN_CAUSAL:
        return kv_seq - query_seq
    if mode == SPARSE_BAND:
        return kv_seq - query_seq + state.next_token
    return state.next_token


def is_within_tolerance(limit, tolerance, value):
    return limit + tolerance >= value


# ---------------------------------------------------------------------------
# cost model (section_stream_k_impl.h CalcCost / CalcCostTable)
# ---------------------------------------------------------------------------


def calc_cost(basic_m, basic_s2):
    align_basic_m = ceil_div(basic_m, COST_M_ALIGNMENT)
    align_basic_s2 = ceil_div(basic_s2, COST_S2_ALIGNMENT)
    return 6 * align_basic_m + 10 * align_basic_s2


# ---------------------------------------------------------------------------
# grid / cost info (CalcGridInfo / CalcGridInfoSection / CalcCostInfo)
# ---------------------------------------------------------------------------


def calc_grid_info(state, grid):
    grid.is_empty = 1
    for b_idx in range(0, state.batch_size):
        s1_size = get_q_seq_size_at(state, b_idx)
        s2_size = get_kv_seq_size_at(state, b_idx)
        grid.m_base_num[b_idx] = ceil_div(s1_size * get_group_size(state), state.m_base)
        grid.m_tail[b_idx] = (s1_size * get_group_size(state)) % state.m_base
        grid.s2_base_num[b_idx] = ceil_div(s2_size, state.s2_base)
        grid.s2_tail[b_idx] = s2_size % state.s2_base
        if grid.m_base_num[b_idx] != 0 and grid.s2_base_num[b_idx] != 0:
            grid.is_empty = 0
    return 0


def calc_grid_info_section(state, grid):
    bn2_total = state.batch_size * state.kv_head_num
    grid.section_num = 0
    # L2 not set -> no section split
    if state.l2_byte == 0:
        grid.section_bn2_idx.append(bn2_total)
        grid.section_num = 1
        return 0
    bn2_idx = 0
    token_limit = state.l2_byte
    token_size = 0
    max_grouped_query_size = 0
    max_single_head_token_cost = 0
    head_dim_qk = state.head_dim_qk
    head_dim_v = state.head_dim_v
    # FP16 -> 2 bytes for both Q and KV (queryType / kvType are FP16)
    for b_idx in range(0, state.batch_size):
        s1_size = get_q_seq_size_at(state, b_idx)
        s2_size = get_kv_seq_size_at(state, b_idx)
        s1_cost = s1_size * head_dim_qk * 2 * 2
        s2v_cost = s2_size * (head_dim_qk + head_dim_v) * 2
        single_head_cost = s1_cost + s2v_cost
        max_single_head_token_cost = max(max_single_head_token_cost, single_head_cost)
        max_grouped_query_size = max(
            max_grouped_query_size, get_group_size(state) * s1_size
        )
        for n2_idx in range(0, state.kv_head_num):
            if token_size + single_head_cost > token_limit and token_size != 0:
                grid.section_bn2_idx.append(bn2_idx)
                grid.section_num += 1
                token_size = 0
            token_size += single_head_cost
            bn2_idx += 1
    grid.section_bn2_idx.append(bn2_total)
    grid.section_num += 1
    # no split when M axis is smaller than one basic block, or when the largest
    # single head fits into the per-core L2 share
    if (
        max_grouped_query_size <= state.m_base
        or max_single_head_token_cost <= token_limit // state.aic_max
    ):
        grid.section_bn2_idx.reset(0, 0)
        grid.section_bn2_idx.append(bn2_total)
        grid.section_num = 1
    return 0


def calc_batch_cache(b_idx, state, grid, bc):
    bc.b_idx = b_idx
    bc.s1_size = get_q_seq_size_at(state, b_idx)
    bc.s2_size = get_kv_seq_size_at(state, b_idx)
    bc.pre_token_left_up = get_pre_token_left_up(state, bc.s1_size, bc.s2_size)
    bc.next_token_left_up = get_next_token_left_up(state, bc.s1_size, bc.s2_size)
    bc.tc_nn = calc_cost(state.m_base, state.s2_base)
    if grid.m_tail[b_idx] == 0:
        bc.tc_tn = 0
    else:
        bc.tc_tn = calc_cost(grid.m_tail[b_idx], state.s2_base)
    if grid.s2_tail[b_idx] == 0:
        bc.tc_nt = 0
    else:
        bc.tc_nt = calc_cost(state.m_base, grid.s2_tail[b_idx])
    if grid.m_tail[b_idx] == 0 or grid.s2_tail[b_idx] == 0:
        bc.tc_tt = 0
    else:
        bc.tc_tt = calc_cost(grid.m_tail[b_idx], grid.s2_tail[b_idx])
    return 0


def calc_s2_range(grouped_query_idx, state, bc, grouped_query):
    grouped_query.s2_start = 0
    grouped_query.s2_end = 0
    # actual seq == 0
    if bc.s1_size == 0 or bc.s2_size == 0:
        return 0
    # no mask
    if get_sparse_mode(state) == SPARSE_BUTT:
        grouped_query.s2_end = ceil_div(bc.s2_size, state.s2_base)
        return 0
    # 1. s1G token range -> s1 token range -> s2 token range
    grouped_query_first_token = grouped_query_idx * state.m_base
    grouped_query_last_token = (
        min(
            grouped_query_first_token + state.m_base, bc.s1_size * get_group_size(state)
        )
        - 1
    )
    if state.is_grouped_query != 0:
        s1_first_token = grouped_query_first_token // get_group_size(state)
        s1_last_token = grouped_query_last_token // get_group_size(state)
    else:
        if (
            grouped_query_first_token // bc.s1_size
            == grouped_query_last_token // bc.s1_size
        ):
            s1_first_token = grouped_query_first_token % bc.s1_size
            s1_last_token = grouped_query_last_token % bc.s1_size
        else:
            s1_first_token = 0
            s1_last_token = bc.s1_size
    s2_first_token = s1_first_token - bc.pre_token_left_up
    s2_last_token = s1_last_token + bc.next_token_left_up
    # no valid token
    if (
        s2_first_token >= bc.s2_size
        or s2_last_token < 0
        or s2_last_token < s2_first_token
    ):
        return 0
    # 2. token index -> block index
    s2_first_token = max(s2_first_token, 0)
    s2_last_token = min(s2_last_token, bc.s2_size - 1)
    grouped_query.s2_start = s2_first_token // state.s2_base
    grouped_query.s2_end = s2_last_token // state.s2_base + 1
    return 0


def calc_grouped_query_cache(grouped_query_idx, state, grid, bc, grouped_query):
    grouped_query.b_idx = bc.b_idx
    grouped_query.grouped_query_idx = grouped_query_idx
    calc_s2_range(grouped_query_idx, state, bc, grouped_query)
    if grouped_query.s2_start >= grouped_query.s2_end or grid.m_base_num[bc.b_idx] == 0:
        grouped_query.grouped_query_block = 0
        grouped_query.grouped_query_cost = 0
        grouped_query.grouped_query_last_block_cost = 0
        grouped_query.grouped_query_normal_block_cost = 0
        return 0
    grouped_query.grouped_query_block = grouped_query.s2_end - grouped_query.s2_start
    if (
        grid.s2_tail[bc.b_idx] != 0
        and grouped_query.s2_end == grid.s2_base_num[bc.b_idx]
    ):
        cur_tail_s2_num = 1
    else:
        cur_tail_s2_num = 0
    cur_normal_s2_num = grouped_query.grouped_query_block - cur_tail_s2_num
    if (
        grouped_query_idx == grid.m_base_num[bc.b_idx] - 1
        and grid.m_tail[bc.b_idx] != 0
    ):
        grouped_query.grouped_query_cost = (
            bc.tc_tn * cur_normal_s2_num + bc.tc_tt * cur_tail_s2_num
        )
        if cur_tail_s2_num > 0:
            grouped_query.grouped_query_last_block_cost = bc.tc_tt
        else:
            grouped_query.grouped_query_last_block_cost = bc.tc_tn
        grouped_query.grouped_query_normal_block_cost = bc.tc_tn
    else:
        grouped_query.grouped_query_cost = (
            bc.tc_nn * cur_normal_s2_num + bc.tc_nt * cur_tail_s2_num
        )
        if cur_tail_s2_num > 0:
            grouped_query.grouped_query_last_block_cost = bc.tc_nt
        else:
            grouped_query.grouped_query_last_block_cost = bc.tc_nn
        grouped_query.grouped_query_normal_block_cost = bc.tc_nn
    return 0


def calc_batch_cost(b_idx, state, grid, cost):
    cost.b_n2_cost[b_idx] = 0
    cost.b_n2_block[b_idx] = 0
    cost.b_n2_last[b_idx] = 0
    if get_q_seq_size_at(state, b_idx) == 0 or get_kv_seq_size_at(state, b_idx) == 0:
        return 0
    bc = BatchCache()
    grouped_query = GroupedQueryCache()
    calc_batch_cache(b_idx, state, grid, bc)
    for grouped_query_idx in range(0, grid.m_base_num[b_idx]):
        calc_grouped_query_cache(grouped_query_idx, state, grid, bc, grouped_query)
        cost.b_n2_cost[b_idx] += grouped_query.grouped_query_cost
        cost.b_n2_block[b_idx] += grouped_query.grouped_query_block
        if grouped_query.grouped_query_block > 0:
            cost.b_n2_last[b_idx] = grouped_query.grouped_query_last_block_cost
    return 0


def calc_cost_info(state, grid, cost):
    for b_idx in range(0, state.batch_size):
        calc_batch_cost(b_idx, state, grid, cost)
    cost.section_block_num.reset(0, grid.section_num)
    cost.section_cost.reset(0, grid.section_num)
    for sec in range(0, grid.section_num):
        bn2_start = 0
        if sec > 0:
            bn2_start = grid.section_bn2_idx[sec - 1]
        bn2_end = grid.section_bn2_idx[sec]
        for bn2_idx in range(bn2_start, bn2_end):
            b_idx = floor_div_safe(bn2_idx, state.kv_head_num, 0)
            cost.section_block_num[sec] += cost.b_n2_block[b_idx]
            cost.section_cost[sec] += cost.b_n2_cost[b_idx]
    return 0


def assign_by_batch(state, grid, cost, ac):
    if ac.is_finished != 0:
        return 0
    while (
        ac.bn2_cost == 0
        or is_within_tolerance(
            ac.core_cache.cost_limit,
            floor_div_safe(cost.b_n2_last[ac.cur_b_idx], state.fa_tol_ratio, 0),
            ac.core_cache.cost + ac.bn2_cost,
        )
        != 0
    ):
        ac.core_cache.cost += ac.bn2_cost
        ac.core_cache.block += ac.bn2_block
        ac.cur_bn2_idx += 1
        # to the end of this section
        if ac.cur_bn2_idx == grid.section_bn2_idx[ac.cur_section_idx]:
            ac.current_grouped_query_idx = 0
            ac.cur_s2_idx = 0
            ac.is_finished = 1
            return 0
        # next batch
        if floor_div_safe(ac.cur_bn2_idx, state.kv_head_num, 0) != ac.cur_b_idx:
            ac.cur_b_idx = floor_div_safe(ac.cur_bn2_idx, state.kv_head_num, 0)
            calc_batch_cache(ac.cur_b_idx, state, grid, ac.batch_cache)
        ac.bn2_cost = cost.b_n2_cost[ac.cur_b_idx]
        ac.bn2_block = cost.b_n2_block[ac.cur_b_idx]
        ac.current_grouped_query_idx = 0
        calc_grouped_query_cache(
            ac.current_grouped_query_idx,
            state,
            grid,
            ac.batch_cache,
            ac.grouped_query_cache,
        )
        ac.cur_s2_idx = ac.grouped_query_cache.s2_start
    return 0


def assign_by_row(state, grid, cost, ac):
    if ac.is_finished != 0:
        return 0
    while (
        is_within_tolerance(
            ac.core_cache.cost_limit,
            floor_div_safe(
                ac.grouped_query_cache.grouped_query_last_block_cost,
                state.fa_tol_ratio,
                0,
            ),
            ac.core_cache.cost + ac.grouped_query_cache.grouped_query_cost,
        )
        != 0
    ):
        ac.core_cache.cost += ac.grouped_query_cache.grouped_query_cost
        ac.core_cache.block += ac.grouped_query_cache.grouped_query_block
        # one row assigned out of the current batch; update the remaining load
        if ac.bn2_cost > ac.grouped_query_cache.grouped_query_cost:
            ac.bn2_cost = ac.bn2_cost - ac.grouped_query_cache.grouped_query_cost
        else:
            ac.bn2_cost = 0
        if ac.bn2_block > ac.grouped_query_cache.grouped_query_block:
            ac.bn2_block = ac.bn2_block - ac.grouped_query_cache.grouped_query_block
        else:
            ac.bn2_block = 0
        # compute the next non-empty row
        while True:
            ac.current_grouped_query_idx += 1
            calc_grouped_query_cache(
                ac.current_grouped_query_idx,
                state,
                grid,
                ac.batch_cache,
                ac.grouped_query_cache,
            )
            if ac.grouped_query_cache.grouped_query_block != 0:
                break
        ac.cur_s2_idx = ac.grouped_query_cache.s2_start
    return 0


def assign_by_block(state, grid, cost, ac):
    if ac.is_finished != 0:
        return 0
    cur_cost = ac.grouped_query_cache.grouped_query_normal_block_cost
    if ac.cur_s2_idx == ac.grouped_query_cache.s2_end - 1:
        cur_cost = ac.grouped_query_cache.grouped_query_last_block_cost
    real_cost = cur_cost
    while (
        is_within_tolerance(
            ac.core_cache.cost_limit,
            floor_div_safe(real_cost, state.fa_tol_ratio, 0),
            ac.core_cache.cost + real_cost,
        )
        != 0
    ):
        ac.core_cache.cost += real_cost
        ac.core_cache.block += 1
        ac.cur_s2_idx += 1
        # one block assigned out of the current batch / row; update remainders
        ac.bn2_cost = ac.bn2_cost - real_cost
        ac.grouped_query_cache.grouped_query_cost = (
            ac.grouped_query_cache.grouped_query_cost - real_cost
        )
        ac.bn2_block -= 1
        ac.grouped_query_cache.grouped_query_block -= 1
        real_cost = cur_cost
    return 0


# ---------------------------------------------------------------------------
# FD record (IsNeedRecordFDInfo / RecordFDInfo / ScheduleFd)
# ---------------------------------------------------------------------------


def is_need_record_fd_info(ac, res):
    # the split point is unlikely to land on a row boundary; defer reduction
    # bookkeeping to the next split point and check there
    if ac.cur_core_idx == 0:
        return 0
    # no cross-core row, nothing to reduce
    if ac.cur_kv_split_part <= 1:
        return 0
    # the row to reduce is not finished yet
    if (
        ac.cur_bn2_idx == res.b_n2_end[ac.cur_core_idx - 1]
        and ac.current_grouped_query_idx == res.grouped_query_end[ac.cur_core_idx - 1]
    ):
        return 0
    return 1


def record_fd_info(state, grid, ac, res):
    # the row to reduce sits at the previous core's split point
    split_b_idx = floor_div_safe(
        res.b_n2_end[ac.cur_core_idx - 1], state.kv_head_num, 0
    )
    split_grouped_query_idx = res.grouped_query_end[ac.cur_core_idx - 1]
    s1_size = get_q_seq_size_at(state, split_b_idx)
    cur_fd_grouped_query_size = state.m_base
    if split_grouped_query_idx == grid.m_base_num[split_b_idx] - 1:
        cur_fd_grouped_query_size = (
            s1_size * get_group_size(state) - split_grouped_query_idx * state.m_base
        )
    res.max_s2_split_num = max(res.max_s2_split_num, ac.cur_kv_split_part)
    # with head reduction the split point is exactly where the previous core ended
    res.b_n2_idx[res.fd_task_num] = res.b_n2_end[ac.cur_core_idx - 1]
    res.m_idx[res.fd_task_num] = res.grouped_query_end[ac.cur_core_idx - 1]
    res.ws_idx[res.fd_task_num] = ac.pre_fd_data_num
    res.s2_split_num[res.fd_task_num] = ac.cur_kv_split_part
    res.m_size[res.fd_task_num] = cur_fd_grouped_query_size
    res.fd_task_num += 1
    return 0


def schedule_fd(aiv_num, res):
    if res.fd_task_num == 0:
        return 0
    total_fd_load = 0
    for i in range(0, res.fd_task_num):
        total_fd_load += res.s2_split_num[i] * res.m_size[i]
    empty_vec_num = aiv_num - res.fd_task_num
    average_load = ceil_div(total_fd_load, aiv_num)
    cur_core_index = 0
    # 1. global average load, ceil so no core gets 0
    # 2. per-task core count, floor, at least 1
    # 3. per-core average rows, ceil so no row count is 0
    # 4. recompute the real core count so the ceil split leaves no empty core
    for i in range(0, res.fd_task_num):
        if empty_vec_num == 0:
            res.task_idx[cur_core_index] = i
            res.m_start[cur_core_index] = 0
            res.m_len[cur_core_index] = res.m_size[i]
            cur_core_index += 1
            continue
        cur_vec_num = res.s2_split_num[i] * res.m_size[i] // average_load
        cur_vec_num = max(cur_vec_num, 1)
        cur_avg_m_size = ceil_div(res.m_size[i], cur_vec_num)
        cur_vec_num = ceil_div(res.m_size[i], cur_avg_m_size)
        cur_vec_num = min(
            cur_vec_num, empty_vec_num + 1
        )  # the fd task brings one core itself
        for vid in range(0, cur_vec_num):
            res.task_idx[cur_core_index] = i
            res.m_start[cur_core_index] = vid * cur_avg_m_size
            if vid < cur_vec_num - 1:
                res.m_len[cur_core_index] = cur_avg_m_size
            else:
                res.m_len[cur_core_index] = res.m_size[i] - vid * cur_avg_m_size
            cur_core_index += 1
        empty_vec_num -= cur_vec_num - 1  # spare cores exclude the task's own core
    res.used_vec_num = cur_core_index
    return 0


# ---------------------------------------------------------------------------
# SKResult helpers
# ---------------------------------------------------------------------------


def alloc_sk(res, aic_num, aiv_num):
    res.b_n2_end = zeros(U32, aic_num)
    res.grouped_query_end = zeros(U32, aic_num)
    res.s2_end = zeros(U32, aic_num)
    res.first_fd_ws = zeros(U32, aic_num)
    res.b_n2_idx = zeros(U32, aic_num)
    res.m_idx = zeros(U32, aic_num)
    res.ws_idx = zeros(U32, aic_num)
    res.s2_split_num = zeros(U32, aic_num)
    res.m_size = zeros(U32, aic_num)
    res.task_idx = zeros(U32, aiv_num)
    res.m_start = zeros(U32, aiv_num)
    res.m_len = zeros(U32, aiv_num)
    return 0


def _fill_array(values, value):
    for index in range(0, len(values)):
        values[index] = value
    return 0


def _copy_array(destination, source):
    for index in range(0, len(source)):
        destination[index] = source[index]
    return 0


def clear_sk(res):
    res.max_cost = INT64_MIN
    res.used_core_num = 0
    res.max_s2_split_num = 0
    res.fd_task_num = 0
    res.used_vec_num = 0
    _fill_array(res.b_n2_end, 0)
    _fill_array(res.grouped_query_end, 0)
    _fill_array(res.s2_end, 0)
    _fill_array(res.first_fd_ws, 0)
    _fill_array(res.b_n2_idx, 0)
    _fill_array(res.m_idx, 0)
    _fill_array(res.ws_idx, 0)
    _fill_array(res.s2_split_num, 0)
    _fill_array(res.m_size, 0)
    _fill_array(res.task_idx, 0)
    _fill_array(res.m_start, 0)
    _fill_array(res.m_len, 0)
    return 0


def copy_sk(dst, src):
    dst.max_cost = src.max_cost
    dst.used_core_num = src.used_core_num
    dst.max_s2_split_num = src.max_s2_split_num
    dst.fd_task_num = src.fd_task_num
    dst.used_vec_num = src.used_vec_num
    _copy_array(dst.b_n2_end, src.b_n2_end)
    _copy_array(dst.grouped_query_end, src.grouped_query_end)
    _copy_array(dst.s2_end, src.s2_end)
    _copy_array(dst.first_fd_ws, src.first_fd_ws)
    _copy_array(dst.b_n2_idx, src.b_n2_idx)
    _copy_array(dst.m_idx, src.m_idx)
    _copy_array(dst.ws_idx, src.ws_idx)
    _copy_array(dst.s2_split_num, src.s2_split_num)
    _copy_array(dst.m_size, src.m_size)
    _copy_array(dst.task_idx, src.task_idx)
    _copy_array(dst.m_start, src.m_start)
    _copy_array(dst.m_len, src.m_len)
    return 0


# ---------------------------------------------------------------------------
# section schedule (ScheduleSection / ScheduleFa / CheckChooseWithFd)
# ---------------------------------------------------------------------------


def schedule_fa(state, grid, cost, cfg, res):
    if cfg.core_num == 0:
        return 0
    res.max_cost = 0
    res.used_core_num = 0
    ac = AssignCtx()
    ac.cur_section_idx = cfg.section_idx
    if cfg.section_idx == 0:
        ac.cur_b_idx = 0
        ac.cur_bn2_idx = 0
    else:
        ac.cur_b_idx = floor_div_safe(
            grid.section_bn2_idx[cfg.section_idx - 1], state.kv_head_num, 0
        )
        ac.cur_bn2_idx = grid.section_bn2_idx[cfg.section_idx - 1]
    ac.current_grouped_query_idx = 0
    ac.cur_core_idx = 0
    ac.used_core_num = 0
    ac.cur_kv_split_part = 1
    ac.pre_fd_data_num = 0
    ac.is_finished = 0
    ac.unassigned_cost = cost.section_cost[cfg.section_idx]
    ac.bn2_cost = cost.b_n2_cost[ac.cur_b_idx]
    ac.bn2_block = cost.b_n2_block[ac.cur_b_idx]
    calc_batch_cache(ac.cur_b_idx, state, grid, ac.batch_cache)
    calc_grouped_query_cache(
        ac.current_grouped_query_idx,
        state,
        grid,
        ac.batch_cache,
        ac.grouped_query_cache,
    )
    ac.cur_s2_idx = ac.grouped_query_cache.s2_start
    for i in range(0, cfg.core_num):
        if res.max_cost > cfg.cost_limit and i > cfg.core_limit:
            return 0
        if ac.is_finished != 0 or ac.unassigned_cost <= 0:
            break
        ac.cur_core_idx = i
        res.first_fd_ws[i] = ac.pre_fd_data_num + ac.cur_kv_split_part - 1
        ac.core_cache.cost = 0
        ac.core_cache.block = 0
        ac.core_cache.cost_limit = ac.unassigned_cost // (cfg.core_num - i)
        if cfg.fd_on == 0:
            ac.core_cache.cost_limit = max(
                ac.core_cache.cost_limit, ac.grouped_query_cache.grouped_query_cost
            )
        else:
            cur_cost = ac.grouped_query_cache.grouped_query_normal_block_cost
            if ac.cur_s2_idx == ac.grouped_query_cache.s2_end - 1:
                cur_cost = ac.grouped_query_cache.grouped_query_last_block_cost
            ac.core_cache.cost_limit = max(ac.core_cache.cost_limit, cur_cost)
        assign_by_batch(state, grid, cost, ac)
        assign_by_row(state, grid, cost, ac)
        if cfg.fd_on != 0:
            assign_by_block(state, grid, cost, ac)
        res.b_n2_end[i] = ac.cur_bn2_idx
        res.grouped_query_end[i] = ac.current_grouped_query_idx
        res.s2_end[i] = ac.cur_s2_idx
        res.max_cost = max(res.max_cost, ac.core_cache.cost)
        ac.unassigned_cost -= ac.core_cache.cost
        if cfg.fd_on != 0 and is_need_record_fd_info(ac, res) != 0:
            record_fd_info(state, grid, ac, res)
            ac.pre_fd_data_num += ac.cur_kv_split_part
            ac.cur_kv_split_part = 1
        if (
            ac.cur_s2_idx > ac.grouped_query_cache.s2_start
            and ac.cur_s2_idx <= ac.grouped_query_cache.s2_end
        ):
            ac.cur_kv_split_part += 1
    res.used_core_num = ac.cur_core_idx + 1
    return 0


def check_choose_with_fd(state, section_num, no_fd, with_fd):
    if state.fd_on == 0:
        return 0
    if section_num > 1:
        return 1
    full_block_cost = calc_cost(state.m_base, state.s2_base)
    if no_fd.max_cost <= state.fd_least_block * full_block_cost:
        return 0
    fd_tolerance = state.fd_tolerance * full_block_cost
    if no_fd.max_cost - fd_tolerance > with_fd.max_cost:
        return 1
    return 0


def schedule_section(state, grid, cost, sec, res, best, best_fd, tmp):
    clear_sk(best)
    clear_sk(best_fd)
    clear_sk(tmp)
    best.max_cost = INT64_MAX
    best.used_core_num = state.aic_max
    best_fd.max_cost = INT64_MAX
    best_fd.used_core_num = state.aic_max
    fa_cfg = FaConfig()
    fa_fd_cfg = FaConfig()
    fa_cfg.fd_on = 0
    fa_cfg.section_idx = sec
    fa_cfg.core_limit = best.used_core_num
    fa_cfg.cost_limit = best.max_cost
    fa_fd_cfg.fd_on = 1
    fa_fd_cfg.section_idx = sec
    fa_fd_cfg.core_limit = best_fd.used_core_num
    fa_fd_cfg.cost_limit = best_fd.max_cost

    # Start the core search one above the integer square root of the section
    # block count, matching the reference core-range calculation.
    max_core = min(state.aic_max, cost.section_block_num[sec])
    r = 0
    while (r + 1) * (r + 1) <= cost.section_block_num[sec]:
        r += 1
    min_core = r + 1
    min_core = max(min_core, state.aic_min)
    min_core = min(min_core, max_core)

    for i in range(min_core, max_core + 1):
        fa_cfg.core_num = i
        fa_fd_cfg.core_num = i
        schedule_fa(state, grid, cost, fa_cfg, tmp)
        if tmp.max_cost < best.max_cost:
            copy_sk(best, tmp)
            fa_cfg.core_limit = best.used_core_num
            fa_cfg.cost_limit = best.max_cost
        clear_sk(tmp)
        if state.fd_on != 0:
            schedule_fa(state, grid, cost, fa_fd_cfg, tmp)
            if tmp.max_cost < best_fd.max_cost:
                copy_sk(best_fd, tmp)
                fa_fd_cfg.core_limit = best_fd.used_core_num
                fa_fd_cfg.cost_limit = best_fd.max_cost
            clear_sk(tmp)
    schedule_fd(state.aiv_max, best_fd)
    if check_choose_with_fd(state, grid.section_num, best, best_fd) != 0:
        copy_sk(res, best_fd)
    else:
        copy_sk(res, best)
    return 0


def params_init(a, state):
    state.aic_max = a.aic_core_num
    state.aic_min = a.aic_core_num
    state.aiv_max = a.aiv_core_num
    state.layout_q = a.layout_q
    state.is_grouped_query = 0
    if (
        state.layout_q == LAYOUT_TND
        or state.layout_q == LAYOUT_BSH
        or state.layout_q == LAYOUT_BSND
    ):
        state.is_grouped_query = 1

    state.batch_size = a.cache_len
    if state.batch_size == 0:
        return 1
    for i in range(0, a.cache_len):
        if a.cache_seqlens[i] < 0:
            return 1
    if a.cu_q_len > 0:
        if a.cu_seqlens_q[0] != 0:
            return 1
        for i in range(1, a.cu_q_len):
            if a.cu_seqlens_q[i] < a.cu_seqlens_q[i - 1]:
                return 1
    if a.sq_len > 0:
        for i in range(0, a.sq_len):
            if a.seqused_q[i] < 0:
                return 1

    if a.sq_len > 0:
        state.q_seq.reset(0, state.batch_size)
        for i in range(0, state.batch_size):
            state.q_seq[i] = a.seqused_q[i]
    elif a.cu_q_len > 1:
        state.q_seq.reset(0, state.batch_size)
        for i in range(0, state.batch_size):
            state.q_seq[i] = a.cu_seqlens_q[i + 1] - a.cu_seqlens_q[i]
    else:
        state.q_seq.reset(0, state.batch_size)
        for i in range(0, state.batch_size):
            state.q_seq[i] = a.max_seqlen_q

    state.kv_seq.reset(0, state.batch_size)
    for i in range(0, state.batch_size):
        state.kv_seq[i] = a.cache_seqlens[i]

    state.q_head_num = a.num_heads_q
    state.kv_head_num = a.num_heads_kv
    state.q_seq_size = a.max_seqlen_q
    state.kv_seq_size = a.max_seqlen_kv
    state.head_dim_qk = a.head_dim_qk
    state.head_dim_v = a.head_dim_v
    if a.mask_mode == 0:
        state.atten_mask_flag = 0
        state.sparse_mode_u = SPARSE_BUTT
    else:
        state.sparse_mode_u = a.mask_mode
        if a.mask_mode != SPARSE_BUTT:
            state.atten_mask_flag = 1
        else:
            state.atten_mask_flag = 0
    state.pre_token = UINT32_MAX
    state.next_token = UINT32_MAX

    state.fa_tol_ratio = FA_TOLERANCE_RATIO
    state.fd_on = 1
    state.fd_tolerance = FD_TOLERANCE
    state.fd_least_block = FD_LEAST_BLOCK
    state.m_base = MLA_M_BASE
    state.s2_base = MLA_S2_BASE
    # Match the unsigned C++ comparison.  Sentinel -1 keeps splitting on.
    if (
        a.max_seqlen_q >= 0
        and get_group_size(state) * a.max_seqlen_q <= state.m_base * 2
    ):
        state.l2_byte = 0
    else:
        state.l2_byte = L2_BYTE
    return 0


def set_fa(a, sec, core_idx, meta_idx, val):
    a.metadata[
        HEAD_METADATA_STRIDE
        + sec * a.aic_core_num * FA_METADATA_STRIDE
        + core_idx * FA_METADATA_STRIDE
        + meta_idx
    ] = val
    return 0


def set_fd(a, n_sec, sec, vec_idx, meta_idx, val):
    a.metadata[
        HEAD_METADATA_STRIDE
        + n_sec * a.aic_core_num * FA_METADATA_STRIDE
        + sec * a.aiv_core_num * FD_METADATA_STRIDE
        + vec_idx * FD_METADATA_STRIDE
        + meta_idx
    ] = val
    return 0


def gen_section_fa(a, res, sec, prev):
    # starts of core 0 chain from the previous section's last used core
    # (all-zero dummy head for section 0)
    for i in range(0, res.used_core_num):
        if i == 0:
            set_fa(a, sec, i, 0, prev.bn2_end)
            set_fa(a, sec, i, 1, prev.grouped_query_end)
            set_fa(a, sec, i, 2, prev.s2_end)
        else:
            set_fa(a, sec, i, 0, res.b_n2_end[i - 1])
            set_fa(a, sec, i, 1, res.grouped_query_end[i - 1])
            set_fa(a, sec, i, 2, res.s2_end[i - 1])
        set_fa(a, sec, i, 3, res.b_n2_end[i])
        set_fa(a, sec, i, 4, res.grouped_query_end[i])
        set_fa(a, sec, i, 5, res.s2_end[i])
        set_fa(a, sec, i, 6, res.first_fd_ws[i])
    prev.bn2_end = res.b_n2_end[res.used_core_num - 1]
    prev.grouped_query_end = res.grouped_query_end[res.used_core_num - 1]
    prev.s2_end = res.s2_end[res.used_core_num - 1]
    return 0


def gen_section_fd(a, n_sec, res, sec):
    for i in range(0, res.used_vec_num):
        t = res.task_idx[i]
        set_fd(a, n_sec, sec, i, 0, res.b_n2_idx[t])
        set_fd(a, n_sec, sec, i, 1, res.m_idx[t])
        set_fd(a, n_sec, sec, i, 2, res.ws_idx[t])
        set_fd(a, n_sec, sec, i, 3, res.s2_split_num[t])
        set_fd(a, n_sec, sec, i, 4, res.m_start[i])
        set_fd(a, n_sec, sec, i, 5, res.m_len[i])
    return 0


# ---------------------------------------------------------------------------
# kernel entry
# ---------------------------------------------------------------------------


@aicpu_kernel
def _flash_mla_metadata_kernel(a: _FlashMlaMetadataArgs):
    class FaState:
        aic_max: U32
        aic_min: U32
        aiv_max: U32
        is_grouped_query: U32
        layout_q: U32
        batch_size: U32
        q_head_num: U32
        kv_head_num: U32
        q_seq_size: U32
        kv_seq_size: U32
        head_dim_qk: U32
        head_dim_v: U32
        atten_mask_flag: U32
        sparse_mode_u: U32
        pre_token: I64
        next_token: I64
        m_base: U32
        s2_base: U32
        fa_tol_ratio: U32
        fd_on: U32
        fd_tolerance: U32
        fd_least_block: U32
        l2_byte: I64
        q_seq: array(I64)
        kv_seq: array(I64)

    class GridInfo:
        m_base_num: array(U32)
        s2_base_num: array(U32)
        m_tail: array(U32)
        s2_tail: array(U32)
        section_bn2_idx: array(U32)
        section_num: U32
        is_empty: U32 = 1

    class CostInfo:
        b_n2_cost: array(I64)
        b_n2_block: array(U32)
        b_n2_last: array(I64)
        section_block_num: array(U32)
        section_cost: array(I64)

    class BatchCache:
        b_idx: U32
        s1_size: U32
        s2_size: U32
        pre_token_left_up: I64
        next_token_left_up: I64
        tc_nn: I64
        tc_tn: I64
        tc_nt: I64
        tc_tt: I64

    class GroupedQueryCache:
        b_idx: U32
        grouped_query_idx: U32
        s2_start: U32
        s2_end: U32
        grouped_query_cost: I64
        grouped_query_last_block_cost: I64
        grouped_query_block: U32
        grouped_query_normal_block_cost: I64

    class CoreCache:
        cost_limit: I64
        cost: I64
        block: U32

    class AssignCtx:
        cur_section_idx: U32
        cur_b_idx: U32
        cur_bn2_idx: U32
        current_grouped_query_idx: U32
        cur_s2_idx: U32
        cur_core_idx: U32
        unassigned_cost: I64
        used_core_num: U32
        cur_kv_split_part: U32
        pre_fd_data_num: U32
        bn2_cost: I64
        bn2_block: U32
        is_finished: U32
        batch_cache: BatchCache
        grouped_query_cache: GroupedQueryCache
        core_cache: CoreCache

    class FaConfig:
        fd_on: U32
        core_num: U32
        core_limit: I64
        cost_limit: I64
        section_idx: U32

    class SKResult:
        max_cost: I64
        used_core_num: U32
        max_s2_split_num: U32
        fd_task_num: U32
        used_vec_num: U32
        b_n2_end: array(U32)
        grouped_query_end: array(U32)
        s2_end: array(U32)
        first_fd_ws: array(U32)
        b_n2_idx: array(U32)
        m_idx: array(U32)
        ws_idx: array(U32)
        s2_split_num: array(U32)
        m_size: array(U32)
        task_idx: array(U32)
        m_start: array(U32)
        m_len: array(U32)

    class PrevSec:
        bn2_end: U32
        grouped_query_end: U32
        s2_end: U32

    seq_cap = a.cu_q_len + a.sq_len + a.cache_len + 4
    bn2_cap = seq_cap * a.num_heads_kv + 4

    state = FaState()
    state.q_seq = array(I64, seq_cap)
    state.kv_seq = array(I64, seq_cap)
    grid = GridInfo()
    grid.m_base_num = zeros(U32, seq_cap)
    grid.s2_base_num = zeros(U32, seq_cap)
    grid.m_tail = zeros(U32, seq_cap)
    grid.s2_tail = zeros(U32, seq_cap)
    grid.section_bn2_idx = array(U32, bn2_cap)
    cost = CostInfo()
    cost.b_n2_cost = zeros(I64, seq_cap)
    cost.b_n2_block = zeros(U32, seq_cap)
    cost.b_n2_last = zeros(I64, seq_cap)
    cost.section_block_num = zeros(U32, bn2_cap)
    cost.section_cost = zeros(I64, bn2_cap)
    res = SKResult()
    alloc_sk(res, a.aic_core_num, a.aiv_core_num)
    best = SKResult()
    alloc_sk(best, a.aic_core_num, a.aiv_core_num)
    best_fd = SKResult()
    alloc_sk(best_fd, a.aic_core_num, a.aiv_core_num)
    tmp = SKResult()
    alloc_sk(tmp, a.aic_core_num, a.aiv_core_num)

    ok = params_init(a, state)
    if ok != 0:
        return 1
    # SectionStreamK::Compute / SetParam guards
    if state.m_base == 0 or state.s2_base == 0 or state.fa_tol_ratio == 0:
        return 1
    if a.aiv_core_num < a.aic_core_num:
        return 1

    calc_grid_info(state, grid)
    calc_grid_info_section(state, grid)

    n_sec = grid.section_num
    if grid.is_empty != 0:
        n_sec = 1

    # FaMetadata::Clear()
    for i in range(0, HEAD_METADATA_STRIDE):
        a.metadata[i] = 0
    fa_total = n_sec * a.aic_core_num * FA_METADATA_STRIDE
    for i in range(0, fa_total):
        a.metadata[HEAD_METADATA_STRIDE + i] = 0
    fd_base = HEAD_METADATA_STRIDE + fa_total
    fd_total = n_sec * a.aiv_core_num * FD_METADATA_STRIDE
    for i in range(0, fd_total):
        a.metadata[fd_base + i] = 0

    prev = PrevSec()
    is_fd = 0
    if grid.is_empty != 0:
        # all-empty case: single trivial section
        clear_sk(res)
        res.max_cost = 0
        res.used_core_num = 1
        res.b_n2_end[0] = state.batch_size * state.kv_head_num
        gen_section_fa(a, res, 0, prev)
    else:
        calc_cost_info(state, grid, cost)
        for sec in range(0, n_sec):
            clear_sk(res)
            res.max_cost = 0
            if cost.section_block_num[sec] == 0:
                # all-mask section: single trivial core assignment
                res.used_core_num = 1
                res.b_n2_end[0] = grid.section_bn2_idx[sec]
            else:
                schedule_section(state, grid, cost, sec, res, best, best_fd, tmp)
            gen_section_fa(a, res, sec, prev)
            if res.used_vec_num > 0:
                is_fd = 1
            gen_section_fd(a, n_sec, res, sec)

    a.metadata[HEAD_SECTION_NUM_INDEX] = n_sec
    a.metadata[HEAD_IS_FD_INDEX] = is_fd
    a.metadata[HEAD_M_BASE_SIZE_INDEX] = state.m_base
    a.metadata[HEAD_S2_BASE_SIZE_INDEX] = state.s2_base
    a.metadata[HEAD_AIC_NUM_INDEX] = a.aic_core_num
    a.metadata[HEAD_AIV_NUM_INDEX] = a.aiv_core_num
    a.metadata[HEAD_OUTPUT_LAYOUT_INDEX] = 0
    need_init = grid.is_empty
    for i in range(0, state.batch_size):
        q_storage = state.q_seq_size
        if a.cu_q_len > 1:
            q_storage = a.cu_seqlens_q[i + 1] - a.cu_seqlens_q[i]
        if state.q_seq[i] < q_storage or state.kv_seq[i] == 0:
            need_init = 1
        if a.mask_mode == SPARSE_RIGHT_DOWN_CAUSAL and state.q_seq[i] > state.kv_seq[i]:
            need_init = 1
    a.metadata[HEAD_NEED_INIT_INDEX] = need_init
    return 0


@lru_cache(maxsize=1)
def _compiled_metadata():
    from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel

    directory = tempfile.TemporaryDirectory(prefix="cannbot_flash_mla_metadata_")
    compiled = compile_aicpu_kernel(
        _flash_mla_metadata_kernel,
        workdir=directory.name,
        launch_mode="interface",
    )
    return directory, compiled


def _validate_metadata_scalars(
    num_heads_q,
    num_heads_kv,
    max_seqlen_q,
    max_seqlen_kv,
    head_dim_qk,
    head_dim_v,
    mask_mode,
    layout_q,
):
    mask_mode = 0 if mask_mode is None else mask_mode
    layout_q = "BSND" if layout_q is None else layout_q
    for name, value in (
        ("num_heads_q", num_heads_q),
        ("num_heads_kv", num_heads_kv),
        ("head_dim_qk", head_dim_qk),
        ("head_dim_v", head_dim_v),
        ("mask_mode", mask_mode),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if num_heads_q <= 0:
        raise ValueError("num_heads_q must be positive")
    if num_heads_kv != 1:
        raise ValueError("MLA requires num_heads_kv == 1")
    if head_dim_qk != HEAD_DIM_QK or head_dim_v != HEAD_DIM_V:
        raise ValueError(
            f"MLA requires head_dim_qk={HEAD_DIM_QK} and head_dim_v={HEAD_DIM_V}"
        )
    if mask_mode not in (SPARSE_DEFAULT_MASK, SPARSE_RIGHT_DOWN_CAUSAL):
        raise ValueError(
            f"mask_mode must be {SPARSE_DEFAULT_MASK} or {SPARSE_RIGHT_DOWN_CAUSAL}"
        )

    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
    for name, value in (
        ("max_seqlen_q", max_seqlen_q),
        ("max_seqlen_kv", max_seqlen_kv),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if value < -1:
            raise ValueError(f"{name} must be -1 or non-negative")

    layout_code = _layout_code(layout_q)
    if layout_code not in (LAYOUT_TND, LAYOUT_BSND, LAYOUT_BNSD):
        raise ValueError("layout_q must be one of TND, BSND, BNSD")
    return mask_mode, layout_code, max_seqlen_q, max_seqlen_kv


def _check_metadata_tensor(name, tensor, device, required=False):
    if tensor is None:
        if required:
            raise ValueError(f"{name} is required and must be non-empty")
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if tensor.numel() == 0:
        if required:
            raise ValueError(f"{name} is required and must be non-empty")
        return None
    if (
        tensor.dtype != torch.int32
        or tensor.device != device
        or tensor.ndim != 1
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be a contiguous 1D int32 tensor on {device}")
    return tensor


def flash_mla_with_kvcache_metadata(
    cache_seqlens,
    num_heads_q: int,
    num_heads_kv: int,
    cu_seqlens_q=None,
    seqused_q=None,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    head_dim_qk=HEAD_DIM_QK,
    head_dim_v=HEAD_DIM_V,
    mask_mode=0,
    layout_q="BSND",
):
    """Generate flat Flash MLA metadata directly on AICPU."""
    from cannbotdsl.aicpu import current_raw_stream

    mask_mode, layout_code, max_seqlen_q, max_seqlen_kv = _validate_metadata_scalars(
        num_heads_q,
        num_heads_kv,
        max_seqlen_q,
        max_seqlen_kv,
        head_dim_qk,
        head_dim_v,
        mask_mode,
        layout_q,
    )
    device_id = torch.npu.current_device()
    device = torch.device("npu", device_id)
    stream = torch.npu.current_stream(device_id)

    cache_seqlens = _check_metadata_tensor(
        "cache_seqlens", cache_seqlens, device, required=True
    )
    batch = cache_seqlens.numel()
    if not 0 < batch < MAX_BATCH_SIZE:
        raise ValueError(
            f"batch size derived from cache_seqlens must be in (0, {MAX_BATCH_SIZE})"
        )
    cu_seqlens_q = _check_metadata_tensor("cu_seqlens_q", cu_seqlens_q, device)
    seqused_q = _check_metadata_tensor("seqused_q", seqused_q, device)
    if layout_code == LAYOUT_TND:
        if cu_seqlens_q is None:
            raise ValueError("cu_seqlens_q is required when layout_q is TND")
    elif cu_seqlens_q is not None:
        raise ValueError("cu_seqlens_q must be absent when layout_q is not TND")
    if cu_seqlens_q is not None and cu_seqlens_q.numel() != batch + 1:
        raise ValueError("cu_seqlens_q must contain batch + 1 elements")
    if seqused_q is not None and seqused_q.numel() != batch:
        raise ValueError("seqused_q must contain batch elements")

    props = torch.npu.get_device_properties(device_id)
    platform_aic = int(props.cube_core_num)
    platform_aiv = int(props.vector_core_num)
    aic_core_num, aiv_core_num = get_effective_core_counts(stream=stream)
    max_schedule_size = (
        1 + (platform_aic + platform_aiv) * batch
    ) * HEAD_METADATA_STRIDE
    max_schedule_size = (
        (max_schedule_size + METADATA_ALIGNMENT_WORDS - 1)
        // METADATA_ALIGNMENT_WORDS
        * METADATA_ALIGNMENT_WORDS
    )
    metadata = torch.empty((max_schedule_size,), dtype=torch.int32, device=device)

    _, compiled = _compiled_metadata()
    compiled.launch(
        current_raw_stream(device_id),
        cu_seqlens_q=0 if cu_seqlens_q is None else cu_seqlens_q.data_ptr(),
        cache_seqlens=cache_seqlens.data_ptr(),
        seqused_q=0 if seqused_q is None else seqused_q.data_ptr(),
        metadata=metadata.data_ptr(),
        cu_q_len=0 if cu_seqlens_q is None else cu_seqlens_q.numel(),
        cache_len=batch,
        sq_len=0 if seqused_q is None else seqused_q.numel(),
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        aic_core_num=aic_core_num,
        aiv_core_num=aiv_core_num,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        mask_mode=mask_mode,
        layout_q=layout_code,
    )
    for tensor in (cache_seqlens, cu_seqlens_q, seqused_q):
        if tensor is not None:
            tensor.record_stream(stream)
    return metadata


__all__ = ["flash_mla_with_kvcache_metadata"]
