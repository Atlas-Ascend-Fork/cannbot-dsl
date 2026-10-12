# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""flash_attn_metadata AICPU operator -- FlashAttention core-assignment metadata generator.

Reference implementations
-------------------------
* ``ops-transformer/attention/flash_attn_metadata/op_kernel_aicpu/
  flash_attn_metadata_aicpu.cpp``            -- Prepare / ParamsInit / GenMetadata
* ``ops-transformer/attention/common/op_kernel/load_balance/``
  ``base_info.h`` / ``load_balance_common.h`` / ``section_stream_k/*``
                                             -- SectionStreamK core assignment (including multiple sections
                                                split according to L2 capacity)
* ``ops-transformer/attention/flash_attn/op_host/fa_adjust_sinner_souter.h``
                                             -- AdjustSinnerAndSouter

All code is kept in this module; no common/ layer is introduced.

Algorithm
---------
Given Q/KV sequence lengths and attention-mask settings, generate a core-assignment
table consumed directly by the FlashAttention kernel. Each AIC handles a range of
``(batch, kv_head)`` x S1 rows x S2 blocks; AIVs reduce split rows. Call order::

    init_params             attrs + sequence inputs -> FaState (m_base/s2_base tile sizes)
    calc_grid_info          M/S2 block counts and tail sizes for each batch
    calc_grid_info_section  split the (batch, kv_head) axis by L2 capacity
    calc_cost_info          costs and block counts per (batch, kv_head) and section
    schedule_section        search core counts and greedily assign work in each section
    write_fa/fd_metadata    write the assignments to the metadata buffer

Core assignment greedily packs work at three granularities for a fixed core count:
all bn2 entries in a batch, then one S1 row, then one S2 block. Each core's load
limit is remaining cost / remaining cores; ``is_within_tolerance`` permits one
tail-block cost / fa_tolerance_ratio beyond that limit to avoid tail imbalance.
The fd path splits a row along S2 across cores and reduces on AIV; ``check_choose_with_fd`` selects the better plan.

Terminology (matching the reference implementation for comparison)
------------------------------------------------------------------
===================  ==========================================================
``s1`` / ``S1``      query sequence axis (M axis)
``s2`` / ``S2``      key-value sequence axis
``m_base``           base tile size on the M axis (reference: ``mBaseSize``)
``s2_base``          base tile size on the S2 axis (reference: ``s2BaseSize``)
``bn2``              flattened ``(batch, kv_head)`` index = ``batch_idx * kv_head_num + kv_head_idx``
``s1g``              S1 group: one M-axis row of ``m_base`` elements
``fd``               final divide: partial results and reduction when a row spans cores
``aic`` / ``aiv``    cube core / vector core
===================  ==========================================================

Interface notes (kernel-direct ABI; no framework tensors on the device side)
-------------------------------------------------------------------------
* Tensor shapes are passed as scalar ``*_len`` parameters;
* Public APIs retain three layout strings; the private ABI uses the reference Layout enum;
* ``CalcCoreRange`` replaces ``std::lround(std::sqrt(n + 0.25f) + 0.5f)``
  with ``isqrt(n) + 1`` since ``floor(sqrt(n + 0.25)) == floor(sqrt(n))``;
* The metadata buffer layout matches ``flash_attn_metadata.h`` exactly:
  ``head[16]`` + ``fa[n_sec][36][16]`` + ``fd[n_sec][72][16]`` in uint32 elements;
  all three stride constants are therefore 16.

Metadata buffer capacity contract (uint32 elements)::

    (1 + batch * num_heads_kv * (AIC_CORE_NUM + AIV_CORE_NUM)) * 16

Here ``batch`` is inferred from ``len(seqused_q)`` or ``len(cu_seqlens_q) - 1``
when Q sequence input exists, and from the ``batch_size`` attr otherwise.
``section_num <= batch * num_heads_kv`` always holds.

Differences from the reference implementation (semantically equivalent simplifications)
-----------------------------------------------------------------------------------------
1. ``CalcCoreRange`` uses integer ``isqrt`` instead of floating-point sqrt.
2. The reference accepts an injected ``costFunc``; no caller in this tree injects
   one, so the default ``calc_cost`` is inlined instead of retaining a function pointer.
3. ``SectionStreamKParam::fdOn`` is always true at its only reference call site,
   and this operator ABI does not expose it, so the fd path always participates.
   To restore the switch, add an attr to args, guard ``check_choose_with_fd`` with
   ``if with_fd_enabled == 0: return 0``, and guard the fd branch in ``schedule_section``.
4. ``SectionStreamKParam::v0Cost`` is always zero in this tree, so ``+= v0_cost`` is omitted.
5. ``AssignContext::usedCoreNum`` is dead in the reference (written but never read).
6. The reference's empty-vector branch in ``BaseInfo::Get*SeqSize(batchIdx)``
   is unreachable here: ``init_params`` always fills ``q_seq`` and ``kv_seq``.
7. D256 physical tiles depend only on group size times max-Q; layouts do not gate the long-Q tile.
8. Only ``aic_core_max``, ``aic_core_min`` and ``aiv_core_max`` are read from
   ``DeviceInfo``; ``aiv_core_min`` is not ported.
9. The elementwise loops in ``clear_result`` and ``copy_result`` use DSL
   ``reset()`` and ``copy_u32()``; the generated C++ loops are equivalent.
"""

import tempfile
from functools import lru_cache
from types import SimpleNamespace

import torch

from cannbotdsl import get_platform_info

from cannbotdsl.aicpu import GmIn, GmOut, I32, I64, U32, aicpu_kernel, current_raw_stream
from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel

if __package__:
    from .flash_attn_validation import select_launch_tile_config
else:
    from flash_attn_validation import select_launch_tile_config

# ---------------------------------------------------------------------------
# Constants: metadata buffer layout (flash_attn_metadata.h)
# ---------------------------------------------------------------------------

# Hardware core limits and FA/FD record counts (one record per core)
AIC_CORE_NUM = 36
AIV_CORE_NUM = 72

# Number of uint32 words per core record: head[16], FA[core][16], FD[core][16]
HEAD_METADATA_STRIDE = 16
FA_METADATA_STRIDE = 16
FD_METADATA_STRIDE = 16

# Field indices within the head record
HEAD_SECTION_NUM_INDEX = 0
HEAD_HAS_FD_INDEX = 1
HEAD_M_BASE_INDEX = 2
HEAD_S2_BASE_INDEX = 3

# Field indices within each FA core record
FA_BN2_START_INDEX = 0
FA_S1G_START_INDEX = 1
FA_S2_START_INDEX = 2
FA_BN2_END_INDEX = 3
FA_S1G_END_INDEX = 4
FA_S2_END_INDEX = 5
FA_FIRST_FD_WORKSPACE_INDEX = 6

# Field indices within each FD core record
FD_BN2_INDEX = 0
FD_S1G_INDEX = 1
FD_WORKSPACE_INDEX = 2
FD_S2_SPLIT_NUM_INDEX = 3
FD_M_START_INDEX = 4
FD_M_LEN_INDEX = 5

# ---------------------------------------------------------------------------
# Constants: SparseMode (load_balance_common.h)
# ---------------------------------------------------------------------------

SPARSE_DEFAULT_MASK = 0
SPARSE_ALL_MASK = 1
SPARSE_LEFT_UP_CAUSAL = 2
SPARSE_RIGHT_DOWN_CAUSAL = 3
SPARSE_BAND = 4
SPARSE_BUTT = 5

MASK_MODE_NONE = 0  # Attribute mask_mode 0 means no mask

# ---------------------------------------------------------------------------
# Constants: Layout (load_balance_common.h)
# ---------------------------------------------------------------------------

LAYOUT_BSND = 0
LAYOUT_BNSD = 1
LAYOUT_BSH = 2
LAYOUT_NBSD = 3
LAYOUT_TND = 4
LAYOUT_NTD = 5
LAYOUT_PA_NZ = 6
LAYOUT_BUTT = 9  # Sentinel for an unknown layout

LAYOUT_CODE_BY_NAME = {
    "BSND": LAYOUT_BSND,
    "BNSD": LAYOUT_BNSD,
    "BSH": LAYOUT_BSH,
    "NBSD": LAYOUT_NBSD,
    "TND": LAYOUT_TND,
    "NTD": LAYOUT_NTD,
    "PA_NZ": LAYOUT_PA_NZ,
}

# ---------------------------------------------------------------------------
# Constants: SectionStreamK scheduling parameters
# Values from flash_attn_metadata_aicpu.cpp InitLoadBalanceParams
# ---------------------------------------------------------------------------

L2_BYTES = 96 * 1024 * 1024  # 96 MB; zero disables L2-based section splitting
FA_TOLERANCE_RATIO = 2  # Larger values mean smaller tolerance; see is_within_tolerance
FD_TOLERANCE_BLOCKS = 10  # Require this many full-block cost units saved to enable fd
FD_LEAST_BLOCKS = 3  # Skip fd when the slowest no-fd core costs at most this many full blocks

# ---------------------------------------------------------------------------
# Constants: default cost model (section_stream_k_impl.h CalcCost)
# ---------------------------------------------------------------------------

COST_M_ALIGN = 16  # Round the M axis up to a multiple of 16
COST_S2_ALIGN = 64  # Round the S2 axis up to a multiple of 64
COST_M_WEIGHT = 6  # M-axis weight
COST_S2_WEIGHT = 10  # S2-axis weight

# ---------------------------------------------------------------------------
# Constants: tile-size adjustment (fa_adjust_sinner_souter.h)
# ---------------------------------------------------------------------------

S_INNER_DEFAULT = 128
S_OUTER_DEFAULT = 64
S_INNER_LARGE = 256  # Use a narrow, long tile for large head_dim or wide windows
S_OUTER_SMALL = 32
HEAD_DIM_LARGE = 256  # Always use the narrow, long tile at this head_dim
HEAD_DIM_SMALL = 128  # Do not adjust for windows above this head_dim
MASK_MODE_BAND = 4  # This mode interprets windows differently; see adjust_sinner_souter
MASK_MODE_CAUSAL = 2
SHORT_QUERY_LIMIT = 64  # Enable check_qv for short Q
LONG_KV_LIMIT = 128  # Enable check_qv for long KV
WINDOW_SUM_LIMIT = 128  # Use a narrow, long tile when the combined window exceeds this

# ---------------------------------------------------------------------------
# Constants: section partitioning (section_stream_k_impl.h CalcGridInfoSection)
# ---------------------------------------------------------------------------

FP16_BYTES = 2  # Both queryType and kvType are FP16
QO_TENSOR_NUM = 2  # Move both Q and O for each row
KV_TENSOR_NUM = 2  # Move both K and V for each row
SECTION_SPLIT_TOLERANCE = 0  # Split at a strict section boundary

# ---------------------------------------------------------------------------
# Constants: special values
# ---------------------------------------------------------------------------

UINT32_MAX = 4294967295  # Unlimited pre_token / next_token value
INT64_MAX = 9223372036854775807
INT64_MIN = -9223372036854775808
MAX_SEQ_LEN_UNBOUNDED = 2147483647  # Replacement for a -1 max_seqlen_* attribute

# Number of reduction cores for a section with no FD records
FD_VEC_NUM_NONE = 0


# ---------------------------------------------------------------------------
# Host-side platform lookup and layout encoding
# ---------------------------------------------------------------------------


def get_effective_core_counts(stream=None):
    """Return the AIC/AIV counts available to the metadata launch stream."""

    props = torch.npu.get_device_properties(torch.npu.current_device())
    cube = int(getattr(props, "cube_core_num", 0))
    vector = int(getattr(props, "vector_core_num", 0)) or 2 * cube
    try:
        info = get_platform_info(stream=stream)
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


def _bounded_core_product(cores, factor, limit):
    # Cap before multiplying, matching the mainline overflow-safe bound.
    if factor >= (limit - 1) // cores + 1:
        return limit
    return cores * factor


def get_max_used_aic_cores(aic_num, batch_size, kv_heads, group_size,
                           max_seq_q, max_seq_kv, m_base, s2_base):
    # Shared by Python launch selection and the compiled AICPU scheduler.
    # Only static bounds are used; sequence tensor contents stay on device.
    if (aic_num <= 0 or batch_size <= 0 or kv_heads <= 0 or group_size <= 0
            or max_seq_q <= 0 or max_seq_kv <= 0 or m_base <= 0 or s2_base <= 0):
        return aic_num
    if max_seq_q > aic_num * m_base // group_size:
        return aic_num
    m_blocks = (max_seq_q * group_size - 1) // m_base + 1
    s2_blocks = (max_seq_kv - 1) // s2_base + 1
    cores = min(batch_size, aic_num)
    cores = _bounded_core_product(cores, kv_heads, aic_num)
    cores = _bounded_core_product(cores, m_blocks, aic_num)
    return _bounded_core_product(cores, s2_blocks, aic_num)


def get_launch_core_counts(cube, vector, batch, q_heads, kv_heads, head_dim,
                           max_seq_q, max_seq_kv, mask_mode, win_left, win_right,
                           *, tile_m_override=0, tile_n_override=0):
    """Choose launch counts from scalar attrs using the metadata tile policy."""
    ratio = vector // cube
    adjust = SimpleNamespace()
    # init_params normalizes unbounded windows before tile selection.
    left = UINT32_MAX if win_left == -1 else win_left
    right = UINT32_MAX if win_right == -1 else win_right
    adjust_sinner_souter(head_dim, max_seq_q, max_seq_kv, mask_mode, left, right, adjust)
    m_base = tile_m_override if tile_m_override else adjust.s_outer * ratio
    s2_base = tile_n_override if tile_n_override else adjust.s_inner
    launch_cube = get_max_used_aic_cores(
        cube, batch, kv_heads, q_heads // kv_heads, max_seq_q, max_seq_kv,
        m_base, s2_base,
    )
    return launch_cube, launch_cube * ratio


def _layout_code(layout: str) -> int:
    """Host-side layout string -> Layout enum value (unknown maps to LAYOUT_BUTT)."""
    return LAYOUT_CODE_BY_NAME.get(layout, LAYOUT_BUTT)


# ---------------------------------------------------------------------------
# Device-side argument structure
#
# Note: this class and kernel-local structs use different registration paths.
# collect_args_struct reads real __annotations__, so this class must stay at
# module scope. Kernel-local structs are registered by AST traversal and must
# be inside the kernel, before first use, with dependencies declared earlier.
# ---------------------------------------------------------------------------


class _FlashAttnMetadataArgs:
    # Input tensors (int32). Missing inputs use empty/None views with *_len set to zero.
    cu_seqlens_q: GmIn(I32)
    cu_seqlens_kv: GmIn(I32)
    seqused_q: GmIn(I32)
    seqused_kv: GmIn(I32)
    # Output tensor (uint32)
    metadata: GmOut(U32)
    # Input lengths
    cu_q_len: U32
    cu_kv_len: U32
    sq_len: U32
    skv_len: U32
    # Required attributes
    num_heads_q: U32
    num_heads_kv: U32
    head_dim: U32
    aic_core_num: U32
    aiv_core_num: U32
    # Optional attributes
    batch_size: I32
    max_seqlen_q: I32
    max_seqlen_kv: I32
    mask_mode: I32
    win_left: I64
    win_right: I64
    layout_q: U32
    layout_kv: U32
    layout_out: U32
    tile_m_override: U32
    tile_n_override: U32


# ---------------------------------------------------------------------------
# BaseInfo accessors (base_info.h)
#
# These read only FaState and are the algorithm's sole access to operator dimensions.
# ---------------------------------------------------------------------------


def get_group_size(state):
    # Number of Q heads per KV head (GQA group size).
    return floor_div_safe(state.q_head_num, state.kv_head_num, 1)


def get_q_seq_size_at(state, batch_idx):
    # Query sequence length for batch_idx.
    return state.q_seq[batch_idx]


def get_kv_seq_size_at(state, batch_idx):
    # KV sequence length for batch_idx.
    return state.kv_seq[batch_idx]


def get_sparse_mode(state):
    # Normalized SparseMode; no mask or out-of-range values map to SPARSE_BUTT.
    if state.mask_enabled == 0:
        return SPARSE_BUTT
    if state.sparse_mode > SPARSE_BUTT:
        return SPARSE_BUTT
    return state.sparse_mode


def get_pre_token_left_up(state, query_seq, kv_seq):
    # Extra visible tokens toward the upper left of a row (sets the S2 start).
    mode = get_sparse_mode(state)
    if mode == SPARSE_BAND:
        return query_seq - kv_seq + state.pre_token
    return state.pre_token


def get_next_token_left_up(state, query_seq, kv_seq):
    # Extra visible tokens toward the lower right of a row (sets the S2 end).
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
    # Inclusive limit check: can value be added without exceeding limit + tolerance?
    #
    # Tolerance is needed because work units rarely align with the ideal split;
    # the final unit is often partial. A strict cost <= limit can prevent progress
    # or cause severe tail imbalance; callers use tail-block cost / fa_tolerance_ratio.
    if limit + tolerance >= value:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Tile-size adjustment (fa_adjust_sinner_souter.h AdjustSinnerAndSouter)
# ---------------------------------------------------------------------------


def adjust_sinner_souter(v_head_dim, max_seq_q, max_seq_kv, mask_mode, win_left, win_right, adjust):
    # Choose base tile sizes for the M and S2 axes.
    #
    # Default: 64 x 128. Use a narrow 32 x 256 tile for large head_dim, or when
    # Q is short, KV is long, and the window is wide, covering more S2 per M row.
    if max_seq_q == -1:
        max_seq_q = MAX_SEQ_LEN_UNBOUNDED
    if max_seq_kv == -1:
        max_seq_kv = MAX_SEQ_LEN_UNBOUNDED
    adjust.s_outer = S_OUTER_DEFAULT
    adjust.s_inner = S_INNER_DEFAULT
    # A short Q with a long KV gives each M row many S2 positions to visit.
    check_qv = 0
    if max_seq_q <= SHORT_QUERY_LIMIT and max_seq_kv > LONG_KV_LIMIT:
        check_qv = 1
    if v_head_dim <= HEAD_DIM_SMALL:
        win_left_tmp = win_left
        win_right_tmp = win_right
        # A window direction inconsistent with mask_mode is inactive and counts as zero.
        if mask_mode == MASK_MODE_NONE:
            if win_left_tmp > 0:
                win_left_tmp = 0
        elif mask_mode == MASK_MODE_BAND:
            if win_right_tmp > 0:
                win_right_tmp = 0
        if (
            mask_mode != MASK_MODE_CAUSAL
            and win_left_tmp + win_right_tmp > WINDOW_SUM_LIMIT
            and check_qv != 0
        ):
            adjust.s_outer = S_OUTER_SMALL
            adjust.s_inner = S_INNER_LARGE
    if v_head_dim == HEAD_DIM_LARGE:
        adjust.s_outer = S_OUTER_SMALL
        adjust.s_inner = S_INNER_LARGE
    return 0


# ---------------------------------------------------------------------------
# Cost model (section_stream_k_impl.h CalcCost)
# ---------------------------------------------------------------------------


def calc_cost(basic_m, basic_s2):
    # Default model: align M and S2 upward, then take a weighted sum.
    #
    # Aligning M to COST_M_ALIGN and S2 to COST_S2_ALIGN accounts for the fixed
    # transfer granularity: even a smaller tile pays the cost of a full unit.
    align_basic_m = (basic_m + COST_M_ALIGN - 1) >> 4
    align_basic_s2 = (basic_s2 + COST_S2_ALIGN - 1) >> 6
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
    # Split the (batch, kv_head) axis into sections according to L2 capacity.
    #
    # One (batch, kv_head) pair loads four FP16 arrays (Q, O, K, V) into L2.
    # Accumulate per head and start a new section once the L2 limit is exceeded.
    bn2_total = state.batch_size * state.kv_head_num
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
        s1_cost = s1_size * head_dim * FP16_BYTES * QO_TENSOR_NUM
        s2v_cost = s2_size * head_dim * FP16_BYTES * KV_TENSOR_NUM
        single_head_cost = s1_cost + s2v_cost
        max_single_head_cost = max(max_single_head_cost, single_head_cost)
        max_gs1_size = max(max_gs1_size, get_group_size(state) * s1_size)
        for _ in range(0, state.kv_head_num):
            # The reference calls IsWithinTolerance(tokenLimit, 0, ...): split only on >.
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
    if get_sparse_mode(state) == SPARSE_BUTT:
        s1g_cache.s2_end = ceil_div(batch_cache.s2_size, state.s2_base)
        return 0

    # 1. S1G row -> S1 token interval -> S2 token interval
    s1g_first_token = s1g_idx * state.m_base
    s1g_last_token = (
        min(s1g_first_token + state.m_base, batch_cache.s1_size * get_group_size(state))
        - 1
    )
    if state.is_s1g != 0:
        # S1G layout: each Q head has its own row; divide by group_size for S1 index.
        s1_first_token = s1g_first_token // get_group_size(state)
        s1_last_token = s1g_last_token // get_group_size(state)
    else:
        # Packed layout: a row can cross boundaries between heads in a group.
        if s1g_first_token // batch_cache.s1_size == s1g_last_token // batch_cache.s1_size:
            s1_first_token = s1g_first_token % batch_cache.s1_size
            s1_last_token = s1g_last_token % batch_cache.s1_size
        else:
            s1_first_token = 0
            s1_last_token = batch_cache.s1_size
    s2_first_token = s1_first_token - batch_cache.pre_token_left_up
    s2_last_token = s1_last_token + batch_cache.next_token_left_up
    # The expanded interval does not overlap KV: this row has no valid tokens.
    if s2_first_token >= batch_cache.s2_size or s2_last_token < 0 or s2_last_token < s2_first_token:
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
    if s1g_cache.s2_start >= s1g_cache.s2_end or grid.m_block_num[batch_cache.batch_idx] == 0:
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
    if s1g_idx == grid.m_block_num[batch_cache.batch_idx] - 1 and grid.m_tail_size[batch_cache.batch_idx] != 0:
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
    if get_q_seq_size_at(state, batch_idx) == 0 or get_kv_seq_size_at(state, batch_idx) == 0:
        return 0
    batch_cache = BatchCache()
    s1g_cache = S1GCache()
    calc_batch_cache(batch_idx, state, grid, batch_cache)
    for s1g_idx in range(0, grid.m_block_num[batch_idx]):
        calc_s1g_cache(s1g_idx, state, grid, batch_cache, s1g_cache)
        cost_info.bn2_cost_per_batch[batch_idx] += s1g_cache.s1g_cost
        cost_info.bn2_block_per_batch[batch_idx] += s1g_cache.s1g_block
        if s1g_cache.s1g_block > 0:
            cost_info.bn2_last_block_cost_per_batch[batch_idx] = s1g_cache.s1g_last_block_cost
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
            batch_idx = floor_div_safe(bn2_idx, state.kv_head_num, 0)
            cost_info.section_block_num[section_idx] += cost_info.bn2_block_per_batch[batch_idx]
            cost_info.section_cost[section_idx] += cost_info.bn2_cost_per_batch[batch_idx]
    return 0


# ---------------------------------------------------------------------------
# Greedy assignment (AssignByBatch / AssignByRow / AssignByBlock)
#
# All three stages share a protocol and run from coarse to fine granularity:
#   assign_by_batch  all KV heads of one batch as an indivisible unit
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
    while assign_ctx.bn2_cost == 0 or is_within_tolerance(
        assign_ctx.core_cache.cost_limit,
        floor_div_safe(
            cost_info.bn2_last_block_cost_per_batch[assign_ctx.cur_batch_idx],
            state.fa_tolerance_ratio,
            0,
        ),
        assign_ctx.core_cache.cost + assign_ctx.bn2_cost,
    ) != 0:
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
        if floor_div_safe(assign_ctx.cur_bn2_idx, state.kv_head_num, 0) != assign_ctx.cur_batch_idx:
            assign_ctx.cur_batch_idx = floor_div_safe(assign_ctx.cur_bn2_idx, state.kv_head_num, 0)
            calc_batch_cache(assign_ctx.cur_batch_idx, state, grid, assign_ctx.batch_cache)
        assign_ctx.bn2_cost = cost_info.bn2_cost_per_batch[assign_ctx.cur_batch_idx]
        assign_ctx.bn2_block = cost_info.bn2_block_per_batch[assign_ctx.cur_batch_idx]
        assign_ctx.cur_s1g_idx = 0
        calc_s1g_cache(assign_ctx.cur_s1g_idx, state, grid, assign_ctx.batch_cache, assign_ctx.s1g_cache)
        assign_ctx.cur_s2_idx = assign_ctx.s1g_cache.s2_start
    return 0


def assign_by_row(state, grid, cost_info, assign_ctx):
    if assign_ctx.is_finished != 0:
        return 0
    while is_within_tolerance(
        assign_ctx.core_cache.cost_limit,
        floor_div_safe(assign_ctx.s1g_cache.s1g_last_block_cost, state.fa_tolerance_ratio, 0),
        assign_ctx.core_cache.cost + assign_ctx.s1g_cache.s1g_cost,
    ) != 0:
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
            calc_s1g_cache(assign_ctx.cur_s1g_idx, state, grid, assign_ctx.batch_cache, assign_ctx.s1g_cache)
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
    real_cost = cur_cost
    while is_within_tolerance(
        assign_ctx.core_cache.cost_limit,
        floor_div_safe(real_cost, state.fa_tolerance_ratio, 0),
        assign_ctx.core_cache.cost + real_cost,
    ) != 0:
        assign_ctx.core_cache.cost += real_cost
        assign_ctx.core_cache.block += 1
        assign_ctx.cur_s2_idx += 1
        # Subtract the assigned block from both batch and current-row remaining work.
        assign_ctx.bn2_cost = assign_ctx.bn2_cost - real_cost
        assign_ctx.s1g_cache.s1g_cost = assign_ctx.s1g_cache.s1g_cost - real_cost
        assign_ctx.bn2_block -= 1
        assign_ctx.s1g_cache.s1g_block -= 1
        real_cost = cur_cost
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
    split_batch_idx = floor_div_safe(result.bn2_end[assign_ctx.cur_core_idx - 1], state.kv_head_num, 0)
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
        cur_vec_num = min(cur_vec_num, empty_vec_num + 1)  # Include the task's own core.
        for vid in range(0, cur_vec_num):
            result.fd_task_idx[cur_core_index] = i
            result.fd_m_start[cur_core_index] = vid * cur_avg_m_size
            if vid < cur_vec_num - 1:
                result.fd_m_len[cur_core_index] = cur_avg_m_size
            else:
                result.fd_m_len[cur_core_index] = result.fd_m_size[i] - vid * cur_avg_m_size
            cur_core_index += 1
        empty_vec_num -= cur_vec_num - 1  # Free cores exclude the task's own core.
    result.used_vec_num = cur_core_index
    return 0


# ---------------------------------------------------------------------------
# Core-assignment result buffers (alloc_result / clear_result / copy_result)
#
# Keep the array-field lists in all three functions in sync.
# ---------------------------------------------------------------------------


def copy_u32(dst, src, n):
    # Copy the first n elements from src to dst.
    for i in range(0, n):
        dst[i] = src[i]
    return 0


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
    result.max_cost = INT64_MIN
    result.used_core_num = 0
    result.max_s2_split_num = 0
    result.fd_task_num = 0
    result.used_vec_num = FD_VEC_NUM_NONE
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


def copy_result(dst, src):
    # Copy the complete result; src and dst must have equal allocated capacity.
    dst.max_cost = src.max_cost
    dst.used_core_num = src.used_core_num
    dst.max_s2_split_num = src.max_s2_split_num
    dst.fd_task_num = src.fd_task_num
    dst.used_vec_num = src.used_vec_num
    copy_u32(dst.bn2_end, src.bn2_end, len(src.bn2_end))
    copy_u32(dst.gs1_end, src.gs1_end, len(src.gs1_end))
    copy_u32(dst.s2_end, src.s2_end, len(src.s2_end))
    copy_u32(dst.first_fd_workspace_idx, src.first_fd_workspace_idx, len(src.first_fd_workspace_idx))
    copy_u32(dst.fd_bn2_idx, src.fd_bn2_idx, len(src.fd_bn2_idx))
    copy_u32(dst.fd_s1g_idx, src.fd_s1g_idx, len(src.fd_s1g_idx))
    copy_u32(dst.fd_workspace_idx, src.fd_workspace_idx, len(src.fd_workspace_idx))
    copy_u32(dst.fd_s2_split_num, src.fd_s2_split_num, len(src.fd_s2_split_num))
    copy_u32(dst.fd_m_size, src.fd_m_size, len(src.fd_m_size))
    copy_u32(dst.fd_task_idx, src.fd_task_idx, len(src.fd_task_idx))
    copy_u32(dst.fd_m_start, src.fd_m_start, len(src.fd_m_start))
    copy_u32(dst.fd_m_len, src.fd_m_len, len(src.fd_m_len))
    return 0


# ---------------------------------------------------------------------------
# Section scheduling (ScheduleFa / CheckChooseWithFd / ScheduleSection)
# ---------------------------------------------------------------------------


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
            grid.section_bn2_idx[config.section_idx - 1], state.kv_head_num, 0
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
    calc_s1g_cache(assign_ctx.cur_s1g_idx, state, grid, assign_ctx.batch_cache, assign_ctx.s1g_cache)
    assign_ctx.cur_s2_idx = assign_ctx.s1g_cache.s2_start

    for i in range(0, config.core_num):
        # Prune this pass when already worse than the best and cores remain unused.
        if result.max_cost > config.cost_limit and i > config.core_limit:
            return 0
        if assign_ctx.is_finished != 0 or assign_ctx.unassigned_cost <= 0:
            break
        assign_ctx.cur_core_idx = i
        result.first_fd_workspace_idx[i] = (
            assign_ctx.pre_fd_data_num + assign_ctx.cur_kv_split_part - 1
        )
        assign_ctx.core_cache.cost = 0
        assign_ctx.core_cache.block = 0
        assign_ctx.core_cache.cost_limit = assign_ctx.unassigned_cost // (config.core_num - i)
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


def check_choose_with_fd(state, section_num, no_fd, with_fd):
    # Choose between the no-fd and fd schedules.
    #
    # For one section, skip fd if the slowest no-fd core is already cheap
    # (within fd_least_blocks full blocks); otherwise require fd savings above
    # fd_tolerance_blocks full blocks. Multiple sections choose fd to avoid idle cores.
    if section_num > 1:
        return 1
    full_block_cost = calc_cost(state.m_base, state.s2_base)
    if no_fd.max_cost <= state.fd_least_blocks * full_block_cost:
        return 0
    # Use subtraction to avoid int64 overflow.
    fd_tolerance = state.fd_tolerance_blocks * full_block_cost
    if no_fd.max_cost - fd_tolerance > with_fd.max_cost:
        return 1
    return 0


def schedule_section(state, grid, cost_info, section_idx, result, best_result_no_fd, best_result_with_fd, tmp_result):
    # Search core counts for a section, solving both no-fd and fd each time.
    #
    # Callers reuse the best-result and temporary buffers to avoid hot-path allocation.
    clear_result(best_result_no_fd)
    clear_result(best_result_with_fd)
    clear_result(tmp_result)
    best_result_no_fd.max_cost = INT64_MAX
    best_result_no_fd.used_core_num = state.aic_core_max
    best_result_with_fd.max_cost = INT64_MAX
    best_result_with_fd.used_core_num = state.aic_core_max
    fa_no_fd_config = FaConfig()
    fa_with_fd_config = FaConfig()
    fa_no_fd_config.with_fd = 0
    fa_no_fd_config.section_idx = section_idx
    fa_no_fd_config.core_limit = best_result_no_fd.used_core_num
    fa_no_fd_config.cost_limit = best_result_no_fd.max_cost
    fa_with_fd_config.with_fd = 1
    fa_with_fd_config.section_idx = section_idx
    fa_with_fd_config.core_limit = best_result_with_fd.used_core_num
    fa_with_fd_config.cost_limit = best_result_with_fd.max_cost

    # CalcCoreRange: min_core = isqrt(block_num) + 1
    # Equivalent to the reference's lround(sqrt(n + 0.25f) + 0.5f).
    max_core = min(state.aic_core_max, cost_info.section_block_num[section_idx])
    r = 0
    while (r + 1) * (r + 1) <= cost_info.section_block_num[section_idx]:
        r += 1
    min_core = r + 1
    min_core = max(min_core, state.aic_core_min)
    min_core = min(min_core, max_core)

    for i in range(min_core, max_core + 1):
        fa_no_fd_config.core_num = i
        fa_with_fd_config.core_num = i
        schedule_fa(state, grid, cost_info, fa_no_fd_config, tmp_result)
        if tmp_result.max_cost < best_result_no_fd.max_cost:
            copy_result(best_result_no_fd, tmp_result)
            # Tighten later core-count pruning thresholds using the current best result.
            fa_no_fd_config.core_limit = best_result_no_fd.used_core_num
            fa_no_fd_config.cost_limit = best_result_no_fd.max_cost
        clear_result(tmp_result)
        schedule_fa(state, grid, cost_info, fa_with_fd_config, tmp_result)
        if tmp_result.max_cost < best_result_with_fd.max_cost:
            copy_result(best_result_with_fd, tmp_result)
            fa_with_fd_config.core_limit = best_result_with_fd.used_core_num
            fa_with_fd_config.cost_limit = best_result_with_fd.max_cost
        clear_result(tmp_result)
    schedule_fd(state.aiv_core_max, best_result_with_fd)
    if check_choose_with_fd(state, grid.section_num, best_result_no_fd, best_result_with_fd) != 0:
        copy_result(result, best_result_with_fd)
    else:
        copy_result(result, best_result_no_fd)
    return 0


# ---------------------------------------------------------------------------
# Parameter initialization (flash_attn_metadata_aicpu.cpp ParamsInit)
# ---------------------------------------------------------------------------


def init_params(args, state):
    # Convert attrs and sequence inputs into FaState and choose tile sizes.
    #
    # A nonzero return means invalid input: malformed cu_seqlens or negative seqused.
    state.aic_core_max = args.aic_core_num
    state.aic_core_min = args.aic_core_num
    state.aiv_core_max = args.aiv_core_num
    state.aiv_per_aic = floor_div_safe(args.aiv_core_num, args.aic_core_num, 0)
    state.layout_q = args.layout_q
    state.layout_kv = args.layout_kv
    state.layout_out = args.layout_out
    # S1G layout: each Q head has its own M-axis row.
    state.is_s1g = 0
    if (
        state.layout_q == LAYOUT_TND
        or state.layout_q == LAYOUT_BSH
        or state.layout_q == LAYOUT_BSND
    ):
        state.is_s1g = 1

    # CheckActualQuerySeq / CheckActualKvSeq: validate raw sequence inputs.
    if args.cu_q_len > 0:
        if args.cu_seqlens_q[0] != 0:
            return 1
        for i in range(1, args.cu_q_len):
            if args.cu_seqlens_q[i] < args.cu_seqlens_q[i - 1]:
                return 1
    if args.sq_len > 0:
        for i in range(0, args.sq_len):
            if args.seqused_q[i] < 0:
                return 1
    if args.cu_kv_len > 0:
        if args.cu_seqlens_kv[0] != 0:
            return 1
        for i in range(1, args.cu_kv_len):
            if args.cu_seqlens_kv[i] < args.cu_seqlens_kv[i - 1]:
                return 1
    if args.skv_len > 0:
        for i in range(0, args.skv_len):
            if args.seqused_kv[i] < 0:
                return 1

    # Sequence priority: seqused (per-batch valid length) > cu_seqlens
    # (cumulative offsets, excluding the first) > attrs. A cu_seqlens of length
    # one yields an empty list, so batch_size/max_seqlen attrs take over.
    if args.sq_len > 0:
        state.batch_size = args.sq_len
        state.q_seq.reset(0, args.sq_len)
        for i in range(0, args.sq_len):
            state.q_seq[i] = args.seqused_q[i]
    elif args.cu_q_len > 1:
        state.batch_size = args.cu_q_len - 1
        state.q_seq.reset(0, state.batch_size)
        for i in range(0, state.batch_size):
            state.q_seq[i] = args.cu_seqlens_q[i + 1] - args.cu_seqlens_q[i]
    else:
        state.batch_size = args.batch_size
        state.q_seq.reset(args.max_seqlen_q, state.batch_size)

    if args.skv_len > 0:
        state.kv_seq.reset(0, args.skv_len)
        for i in range(0, args.skv_len):
            state.kv_seq[i] = args.seqused_kv[i]
    elif args.cu_kv_len > 1:
        n_kv = args.cu_kv_len - 1
        state.kv_seq.reset(0, n_kv)
        for i in range(0, n_kv):
            state.kv_seq[i] = args.cu_seqlens_kv[i + 1] - args.cu_seqlens_kv[i]
    else:
        state.kv_seq.reset(args.max_seqlen_kv, state.batch_size)

    # InitBaseInfo
    state.q_head_num = args.num_heads_q
    state.kv_head_num = args.num_heads_kv
    state.head_dim = args.head_dim
    if args.mask_mode == MASK_MODE_NONE:
        state.mask_enabled = 0
        state.sparse_mode = SPARSE_BUTT
    else:
        state.sparse_mode = args.mask_mode
        if args.mask_mode != SPARSE_BUTT:
            state.mask_enabled = 1
        else:
            state.mask_enabled = 0
    # win_left / win_right == -1 means unbounded in that direction.
    if args.win_left == -1:
        state.pre_token = UINT32_MAX
    else:
        state.pre_token = args.win_left
    if args.win_right == -1:
        state.next_token = UINT32_MAX
    else:
        state.next_token = args.win_right

    # InitLoadBalanceParams
    state.fa_tolerance_ratio = FA_TOLERANCE_RATIO
    state.fd_tolerance_blocks = FD_TOLERANCE_BLOCKS
    state.fd_least_blocks = FD_LEAST_BLOCKS
    state.l2_bytes = L2_BYTES

    adjust = AdjSO()
    adjust_sinner_souter(
        args.head_dim, args.max_seqlen_q, args.max_seqlen_kv,
        state.sparse_mode, state.pre_token, state.next_token, adjust,
    )
    # Scale m_base by AIV/AIC core ratio: all vector cores share one M row.
    state.m_base = adjust.s_outer * state.aiv_per_aic
    state.s2_base = adjust.s_inner
    if args.tile_m_override != 0:
        state.m_base = args.tile_m_override
    if args.tile_n_override != 0:
        state.s2_base = args.tile_n_override
    # Match the Host launch bound without changing physical metadata strides.
    state.aic_core_max = get_max_used_aic_cores(
        args.aic_core_num, state.batch_size, state.kv_head_num,
        get_group_size(state), args.max_seqlen_q, args.max_seqlen_kv,
        state.m_base, state.s2_base,
    )
    state.aic_core_min = state.aic_core_max
    state.aiv_core_max = state.aic_core_max * state.aiv_per_aic
    return 0


# ---------------------------------------------------------------------------
# Metadata output (flash_attn_metadata_aicpu.cpp GenMetadata + flash_attn_metadata.h)
# ---------------------------------------------------------------------------


def write_fa_metadata(args, section_idx, core_idx, field_idx, value):
    # Write the FA region: metadata[head][section][core][field].
    args.metadata[
        HEAD_METADATA_STRIDE + section_idx * AIC_CORE_NUM * FA_METADATA_STRIDE
        + core_idx * FA_METADATA_STRIDE + field_idx
    ] = value
    return 0


def write_fd_metadata(args, section_num, section_idx, vec_idx, field_idx, value):
    # Write the FD region after FA: metadata[head][FA region][section][vector][field].
    args.metadata[
        HEAD_METADATA_STRIDE + section_num * AIC_CORE_NUM * FA_METADATA_STRIDE
        + section_idx * AIV_CORE_NUM * FD_METADATA_STRIDE
        + vec_idx * FD_METADATA_STRIDE + field_idx
    ] = value
    return 0


def gen_section_fa(args, result, section_idx, prev_section):
    # Write one section's FA region.
    #
    # Each core records its own [start, end). The first start uses the preceding
    # core's end; core zero uses the prior section's last end (or zero).
    for i in range(0, result.used_core_num):
        if i == 0:
            write_fa_metadata(args, section_idx, i, FA_BN2_START_INDEX, prev_section.bn2_end)
            write_fa_metadata(args, section_idx, i, FA_S1G_START_INDEX, prev_section.gs1_end)
            write_fa_metadata(args, section_idx, i, FA_S2_START_INDEX, prev_section.s2_end)
        else:
            write_fa_metadata(args, section_idx, i, FA_BN2_START_INDEX, result.bn2_end[i - 1])
            write_fa_metadata(args, section_idx, i, FA_S1G_START_INDEX, result.gs1_end[i - 1])
            write_fa_metadata(args, section_idx, i, FA_S2_START_INDEX, result.s2_end[i - 1])
        write_fa_metadata(args, section_idx, i, FA_BN2_END_INDEX, result.bn2_end[i])
        write_fa_metadata(args, section_idx, i, FA_S1G_END_INDEX, result.gs1_end[i])
        write_fa_metadata(args, section_idx, i, FA_S2_END_INDEX, result.s2_end[i])
        write_fa_metadata(args, section_idx, i, FA_FIRST_FD_WORKSPACE_INDEX, result.first_fd_workspace_idx[i])
    prev_section.bn2_end = result.bn2_end[result.used_core_num - 1]
    prev_section.gs1_end = result.gs1_end[result.used_core_num - 1]
    prev_section.s2_end = result.s2_end[result.used_core_num - 1]
    return 0


def gen_section_fd(args, section_num, result, section_idx):
    # Write one section's FD region: one reduction-task record per AIV core.
    for i in range(0, result.used_vec_num):
        task = result.fd_task_idx[i]
        write_fd_metadata(args, section_num, section_idx, i, FD_BN2_INDEX, result.fd_bn2_idx[task])
        write_fd_metadata(args, section_num, section_idx, i, FD_S1G_INDEX, result.fd_s1g_idx[task])
        write_fd_metadata(args, section_num, section_idx, i, FD_WORKSPACE_INDEX, result.fd_workspace_idx[task])
        write_fd_metadata(args, section_num, section_idx, i, FD_S2_SPLIT_NUM_INDEX, result.fd_s2_split_num[task])
        write_fd_metadata(args, section_num, section_idx, i, FD_M_START_INDEX, result.fd_m_start[i])
        write_fd_metadata(args, section_num, section_idx, i, FD_M_LEN_INDEX, result.fd_m_len[i])
    return 0


def generate_metadata(args):
    # Complete SectionStreamK::Compute flow; write output into args.metadata.

    # Sequence input capacity; +4 allows for cu_seqlens first-element removal and attrs fallback.
    seq_capacity = (
        args.cu_q_len + args.sq_len + args.cu_kv_len + args.skv_len + args.batch_size + 4
    )
    bn2_capacity = seq_capacity * args.num_heads_kv + 4

    state = FaState()
    state.q_seq = array(I64, seq_capacity)
    state.kv_seq = array(I64, seq_capacity)
    grid = GridInfo()
    grid.m_block_num = zeros(U32, seq_capacity)
    grid.s2_block_num = zeros(U32, seq_capacity)
    grid.m_tail_size = zeros(U32, seq_capacity)
    grid.s2_tail_size = zeros(U32, seq_capacity)
    grid.section_bn2_idx = array(U32, bn2_capacity)
    cost_info = CostInfo()
    cost_info.bn2_cost_per_batch = zeros(I64, seq_capacity)
    cost_info.bn2_block_per_batch = zeros(U32, seq_capacity)
    cost_info.bn2_last_block_cost_per_batch = zeros(I64, seq_capacity)
    cost_info.section_block_num = zeros(U32, bn2_capacity)
    cost_info.section_cost = zeros(I64, bn2_capacity)
    # Reuse four result buffers during core-count search to avoid hot-path allocation.
    result = SKResult()
    alloc_result(result, args.aic_core_num, args.aiv_core_num)
    best_result_no_fd = SKResult()
    alloc_result(best_result_no_fd, args.aic_core_num, args.aiv_core_num)
    best_result_with_fd = SKResult()
    alloc_result(best_result_with_fd, args.aic_core_num, args.aiv_core_num)
    tmp_result = SKResult()
    alloc_result(tmp_result, args.aic_core_num, args.aiv_core_num)

    ok = init_params(args, state)
    if ok != 0:
        return 1
    # Argument guard for SectionStreamK::Compute / SetParam.
    if state.m_base == 0 or state.s2_base == 0 or state.fa_tolerance_ratio == 0:
        return 1
    if args.aiv_core_num < args.aic_core_num:
        return 1

    calc_grid_info(state, grid)
    calc_grid_info_section(state, grid)

    section_num = grid.section_num
    if grid.is_empty != 0:
        section_num = 1

    # FaMetadata::Clear(): head, FA, and FD are adjacent, so clear them together.
    fa_words = section_num * AIC_CORE_NUM * FA_METADATA_STRIDE
    fd_words = section_num * AIV_CORE_NUM * FD_METADATA_STRIDE
    for i in range(0, HEAD_METADATA_STRIDE + fa_words + fd_words):
        args.metadata[i] = 0

    prev_section = PrevSec()
    has_fd = 0
    if grid.is_empty != 0:
        # All batches are empty: one trivial section with one idle core.
        clear_result(result)
        result.max_cost = 0
        result.used_core_num = 1
        result.bn2_end[0] = state.batch_size * state.kv_head_num
        gen_section_fa(args, result, 0, prev_section)
    else:
        calc_cost_info(state, grid, cost_info)
        for section_idx in range(0, section_num):
            clear_result(result)
            result.max_cost = 0
            if cost_info.section_block_num[section_idx] == 0:
                # The section is fully masked: one core traverses all its bn2 entries.
                result.used_core_num = 1
                result.bn2_end[0] = grid.section_bn2_idx[section_idx]
            else:
                schedule_section(
                    state, grid, cost_info, section_idx, result, best_result_no_fd, best_result_with_fd, tmp_result
                )
            gen_section_fa(args, result, section_idx, prev_section)
            if result.used_vec_num > 0:
                has_fd = 1
            gen_section_fd(args, section_num, result, section_idx)

    args.metadata[HEAD_SECTION_NUM_INDEX] = section_num
    args.metadata[HEAD_HAS_FD_INDEX] = has_fd
    args.metadata[HEAD_M_BASE_INDEX] = state.m_base
    args.metadata[HEAD_S2_BASE_INDEX] = state.s2_base
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
def _flash_attn_metadata_kernel(args: _FlashAttnMetadataArgs):
    class FaState:
        aic_core_max: U32
        aic_core_min: U32
        aiv_core_max: U32
        aiv_per_aic: U32
        is_s1g: U32
        layout_q: U32
        layout_kv: U32
        layout_out: U32
        batch_size: U32
        q_head_num: U32
        kv_head_num: U32
        head_dim: U32
        mask_enabled: U32
        sparse_mode: U32
        pre_token: I64
        next_token: I64
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
    # It is not SectionStreamKParam::fdOn, which is always true in the reference.
    class FaConfig:
        with_fd: U32
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

    class AdjSO:
        s_outer: U32
        s_inner: U32

    # End position of the previous section's last core, chaining sections.
    class PrevSec:
        bn2_end: U32
        gs1_end: U32
        s2_end: U32

    return generate_metadata(args)


@lru_cache(maxsize=1)
def _compiled_metadata():

    directory = tempfile.TemporaryDirectory(prefix="cannbot_flash_attn_metadata_")
    compiled = compile_aicpu_kernel(
        _flash_attn_metadata_kernel,
        workdir=directory.name,
        launch_mode="interface",
    )
    return directory, compiled


def flash_attn_metadata(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    batch_size=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    layout_out="BSND",
):
    """Generate load-balance metadata on AICPU using the current NPU stream.

    Paged KV layouts require seqused_kv and reject cu_seqlens_kv. Pass the
    same logical layouts and sequence parameters to flash_attn.
    """
    layout_q = "BSND" if layout_q is None else layout_q
    layout_kv = "BSND" if layout_kv is None else layout_kv
    layout_out = "BSND" if layout_out is None else layout_out
    is_pa = layout_kv in ("PA_BBND", "PA_BNBD", "PA_NZ")
    if layout_q == "TND" and cu_seqlens_q is None:
        raise ValueError("cu_seqlens_q is required when layout_q is TND")
    if is_pa and seqused_kv is None:
        raise ValueError("seqused_kv is required for PA layout")
    if is_pa and cu_seqlens_kv is not None:
        raise ValueError("cu_seqlens_kv must be None for PA layout")
    if not isinstance(head_dim, int) or isinstance(head_dim, bool):
        raise TypeError("head_dim must be an integer")
    if head_dim not in (64, 128, 256):
        raise ValueError("head_dim must be one of 64, 128, 256")

    sequences = dict(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
    )
    for name, tensor in sequences.items():
        if tensor is not None and tensor.ndim != 1:
            raise ValueError(f"{name} must be a 1D tensor")
    inferred_batch = None
    if seqused_q is not None:
        inferred_batch = seqused_q.size(0)
    elif cu_seqlens_q is not None and cu_seqlens_q.size(0) > 0:
        inferred_batch = cu_seqlens_q.size(0) - 1
    if (inferred_batch is not None and batch_size is not None
            and batch_size >= 0 and batch_size != inferred_batch):
        raise ValueError("batch_size must match inferred batch size")
    if inferred_batch is not None:
        batch = inferred_batch
    elif batch_size is None or batch_size < 0:
        raise ValueError("batch_size is required when query sequence tensors are absent")
    else:
        batch = batch_size
    if num_heads_kv <= 0 or num_heads_q < num_heads_kv or num_heads_q % num_heads_kv:
        raise ValueError("num_heads_q must be a positive multiple of num_heads_kv")
    for name, tensor in sequences.items():
        expected = batch + 1 if name.startswith("cu_") else batch
        if tensor is not None and tensor.numel() != expected:
            raise ValueError(f"{name} must be a 1D tensor of length {expected}")

    device_id = torch.npu.current_device()
    device = torch.device("npu", device_id)
    stream = torch.npu.current_stream(device_id)
    for name, tensor in sequences.items():
        invalid_sequence = tensor is not None and (
            tensor.dtype != torch.int32 or tensor.device != device
            or tensor.ndim != 1 or not tensor.is_contiguous()
        )
        if invalid_sequence:
            raise ValueError(f"{name} must be a contiguous 1D int32 tensor on {device}")

    tile = select_launch_tile_config(
        head_dim, q_heads=num_heads_q, kv_heads=num_heads_kv,
        max_seqlen_q=max_seqlen_q, layout_q=layout_q,
        layout_kv=layout_kv, layout_out=layout_out, mask_mode=mask_mode,
        has_sequence_vectors=any(tensor is not None for tensor in sequences.values()),
    )
    large_d256 = head_dim == 256 and tile.tile_cube_m == 128
    _, compiled = _compiled_metadata()
    aic, aiv = get_effective_core_counts(stream=stream)
    size = ((AIC_CORE_NUM + AIV_CORE_NUM) * batch * num_heads_kv + 1) * HEAD_METADATA_STRIDE
    size = ((size + 4095) // 4096) * 4096
    metadata = torch.empty(size, dtype=torch.int32, device=device)
    compiled.launch(
        current_raw_stream(device_id),
        **{name: 0 if tensor is None else tensor.data_ptr() for name, tensor in sequences.items()},
        metadata=metadata.data_ptr(),
        cu_q_len=0 if cu_seqlens_q is None else cu_seqlens_q.numel(),
        cu_kv_len=0 if cu_seqlens_kv is None else cu_seqlens_kv.numel(),
        sq_len=0 if seqused_q is None else seqused_q.numel(),
        skv_len=0 if seqused_kv is None else seqused_kv.numel(),
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        aic_core_num=aic,
        aiv_core_num=aiv,
        batch_size=-1 if batch_size is None else batch_size,
        max_seqlen_q=-1 if max_seqlen_q is None else max_seqlen_q,
        max_seqlen_kv=-1 if max_seqlen_kv is None else max_seqlen_kv,
        mask_mode=1 if mask_mode is None else mask_mode,
        win_left=-1 if win_left is None else win_left,
        win_right=-1 if win_right is None else win_right,
        layout_q=_layout_code(layout_q),
        layout_kv=_layout_code(layout_kv),
        layout_out=_layout_code(layout_out),
        tile_m_override=tile.tile_cube_m if large_d256 else 0,
        tile_n_override=tile.tile_n if large_d256 else 0,
    )
    for tensor in sequences.values():
        if tensor is not None:
            tensor.record_stream(stream)
    return metadata
