# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MLA with Channel handoffs and explicit resident/scratch storage.

QK uses 96/112/128/112/128 reductions with resident Q96.
PV trails QK by one kv_issue_idx; KV uses three L1 slots, and QK UB uses one slot; PV has two 256-column slots.
Split-KV combine processes eight rows with a separate UB layout.
QK and PV cross-core handoffs use typed Channels with compiler-synthesized
transactions; resident state and interleaved L1 aliases remain Buffers.
"""

from collections import OrderedDict
from threading import RLock

import torch
import cannbotdsl as dsl
from math import prod
from cannbotdsl import (
    Channel,
    ChannelKind,
    MemLoc,
    Tensor,
    TensorSpec,
    dtypes,
    permute,
    host,
)
from cannbotdsl.buffer import Buffer
try:
    from cannbotdsl import ProfileSpec
except ImportError:  # 0.2.x: only the canonical submodule exports it
    from cannbotdsl.core.profiling import ProfileSpec


# Keep the kernel importable with both supported CANNBot-DSL module layouts.
try:  # 0.2.x
    from cannbotdsl.arch import get_block_idx, get_block_num, get_subblock_id
    from cannbotdsl.constexpr import const_expr
    from cannbotdsl.delay_line import DelayLineGroup
    from cannbotdsl.jit_function import jit
    from cannbotdsl.kernel_launcher import kernel
    from cannbotdsl.math import matmul
    from cannbotdsl.sync import PIPE, global_sync_all
except ImportError:  # 0.3+: submodule paths reorganised, names top-level
    from cannbotdsl import (
        get_block_idx, get_block_num, get_subblock_id, const_expr,
        DelayLineGroup, jit, kernel, matmul, PIPE, global_sync_all,
    )
from cannbotdsl.tensor import (
    reinterpret,
    make_tiler,
    tile_slice,
)
try:
    from cannbotdsl.tensor import make_copy_engine, mem_copy
except ImportError:  # 0.3+: copy engines moved to ops
    from cannbotdsl.ops import make_copy_engine, mem_copy
try:
    from cannbotdsl import raw_reg as rr
except ImportError:  # 0.3+: raw_reg moved to ops.reg, some ops renamed
    from cannbotdsl.ops import reg as _reg
    from types import SimpleNamespace as _NS
    rr = _NS(
        **{n: getattr(_reg, n) for n in dir(_reg) if not n.startswith("_")},
        vload_brc=_reg.vload_broadcast,   # renamed
        vdup_scalar=_reg.vdups,           # renamed
        vcmp_eq_scalar=_reg.veqs,         # renamed
        vln=_reg.vlog,                    # renamed
        vor=_reg.vbitwise_or,             # renamed
    )
try:
    from cannbotdsl.vf import vf
except ImportError:
    from cannbotdsl import vf

from checker import check_flash_mla_inputs
from flash_mla_with_kvcache_metadata import get_effective_core_counts

__all__ = ["flash_mla_with_kvcache"]


def _offset_view(tensor, coord):
    """Return a rank-preserving view starting at an element coordinate."""
    return tensor[coord[0]:, coord[1]:]


def _paged_kv_page(kv_cache, layout, page, page_size):
    """Select one physical KV page while preserving its outer GM stride.

    Locate the original page/head coordinates first. PA_NZ preserves the
    physical D1/token/D0 axes.
    """
    if const_expr(layout == "PA_NZ"):
        return kv_cache[page, 0, None, None, None]
    if const_expr(layout == "PA_BBND"):
        return kv_cache[page, None, 0, None].view(page_size, HEAD_DIM_QK)
    return kv_cache[page, 0, None, None].view(page_size, HEAD_DIM_QK)


def _native_row_plane(tensor, layout, physical_row, head_num, query_span, width):
    # Remove only singleton axes fixed by coordinates; keep token/head separate.
    if const_expr(layout == "TND"):
        return tensor[physical_row // head_num, None, None].view(head_num, width)
    if const_expr(layout == "BSND"):
        token = physical_row // head_num
        return tensor[token // query_span, token % query_span, None, None].view(head_num, width)
    head = physical_row // query_span
    return tensor[head // head_num, head % head_num, None, None].view(query_span, width)


def _native_lse_plane(tensor, layout, physical_row, head_num, query_span):
    if const_expr(layout == "TND"):
        return tensor[physical_row // head_num, None].view(head_num, 1)
    if const_expr(layout == "BSND"):
        token = physical_row // head_num
        return tensor[token // query_span, token % query_span, None].view(head_num, 1)
    head = physical_row // query_span
    return tensor[head // head_num, head % head_num, None].view(query_span, 1)

# Hardware geometry.
VECTOR_REGISTER_BITS = 2048
FP32_BITS = 32
FRACTAL_SIZE = 16
VL_T = VECTOR_REGISTER_BITS // FP32_BITS

# Three KVP storage slots; current QK/softmax overlaps previous PV/update.
PIPELINE_DEPTH = 3

# Kernel tile configuration.
TILE_CUBE_M = 96    # Fixed hardware M tile (L0C holds
# (64, 512) fp32 = 128K <= 256K); the launch tier is
# independent of batch/head/query counts
TILE_VEC_M = 48     # rows per AIV at M96 (split-M); runtime value is
# tile_cube_m // 2 of the selected tier
TILE_N = 112        # KV tile width (softmax column loop stays FA-shaped)
D_CHUNK = 128       # nope reduction/output segment (512 % 128 == 0, no tails)
D_ROPE = 64         # rope segment width (576 = 512 + 64)
TILE_D = 512        # head_dim_v (output width)
N_DCHUNK = 4        # 512 / D_CHUNK
QK_UB_ROW_STRIDE = 128
QK_CHUNK_WIDTHS = (96, 112, 128, 112, 128)
QK_CHUNK_OFFSETS = (0, 96, 208, 336, 448)
MAX_NZ_PITCH_COUNT = 7
PV_BAND_WIDTH = TILE_D // 2
FD_REDUCE_ROWS = 32
FD_WINDOW_ROWS = 8
FD_STATS_PER_SLOT = 2
FD_PARTIAL_CAPACITY = 192
FD_STATE_COLUMNS = 8
MASK_TEMPLATE_SIZE = 2048

TASK_KIND_MAIN = 0
TASK_KIND_PARTIAL = 1
TASK_KIND_REDUCE = 2

# FIA flat metadata ABI: head[16], FA[section][AIC][16], then
# FD[section][AIV][16].  Every record uses int32 words.
METADATA_STRIDE = 16
HEAD_SECTION_NUM = 0
HEAD_IS_FD = 1
HEAD_M_BASE_SIZE = 2
HEAD_S2_BASE_SIZE = 3
HEAD_AIC_NUM = 4
HEAD_AIV_NUM = 5
HEAD_NEED_INIT = 7
FA_BN2_START = 0
FA_M_START = 1
FA_S2_START = 2
FA_BN2_END = 3
FA_M_END = 4
FA_S2_END = 5
FA_FIRST_FD_WORKSPACE = 6
FD_BN2_IDX = 0
FD_M_IDX = 1
FD_WORKSPACE_IDX = 2
FD_WORKSPACE_NUM = 3
FD_M_START = 4
FD_M_NUM = 5
REDUCE_TILE_M = 12

# Explicit local-memory map. Named addresses make intentional aliases visible:
# mask state aliases the three online-softmax state slots, while QK/PV are
# cross-core Channels at the beginning of UB.
UB_PV_ADDR = 0
UB_QK_ADDR = 98304
UB_OUTPUT_ACCUM_ADDR = 122880
UB_FD_SUM_BASE = 163840
UB_FD_MAX_BASE = 176128
UB_FD_EXP_ADDR = 188416
UB_FD_INPUT_ADDR = 194560
UB_P_ADDR = 221184
UB_FD_OUTPUT_ACCUM_ADDR = 227328
UB_MASK_HEAD_ADDR = 233728
UB_MASK_BODY_ADDR = 239872
UB_FD_OUTPUT_ADDR = 243712
UB_SOFTMAX_SUM_BASE = 246016
UB_SOFTMAX_MAX_BASE = 246784
UB_SOFTMAX_EXP_BASE = 247552
UB_LSE_ADDR = 248320
UB_NEW_MAX_ADDR = 248576
UB_NEW_SUM_ADDR = 248832
UB_FD_MAX_STATE_ADDR = 251904
UB_FD_SUM_STATE_ADDR = 252160
UB_FD_LSE_ADDR = 252416
UB_STATE_SLOT_STRIDE = 256
UB_FD_STATE_SLOT_STRIDE = 6144

L1_Q_ADDR = 0
L1_KV_BASE_ADDR = 110592
L1_KV_SLOT_STRIDE = 136192
L1_P_OFFSET = 114688
L0A_Q96_ADDR = 0
L0A_Q128_ADDR = 18432
L0A_Q112_ADDR = 43008
L0B_SLOT_STRIDE = 32768
L0C_SLOT_STRIDE = 131072

# The kernel uses one M tile size for every batch, head count, and query
# length. Tail rows are bounded dynamically.
TILE_TIERS = (TILE_CUBE_M, )

HEAD_DIM_QK = 576
HEAD_DIM_V = 512

# PA_NZ page geometry (true source: AscendC attention common
# memcopy/gm_layout.h GmLayout<PA_NZ>::MakeLayout -- shape
# (n, d1, blockSize, d0) with d0 = 32B/element = 16, d1 = 576/16 = 36;
# strides d0 = 1 < token = 16 < d-segment = 16*BlockSize).  A token's 576
# dims are 36 sixteen-element segments, token-major inside each segment.
PA_NZ_D0 = 16                        # innermost: 16 dims, stride 1
PA_NZ_D1 = HEAD_DIM_QK // PA_NZ_D0  # 36 d-segments, stride 16*BlockSize

# Finite stand-in for negative infinity in masked row-max reductions.
# vdup_scalar literals must be finite (scalar_conversion validates), so
# the fp32 minimum serves as the max identity; any finite score beats it.
FP32_MIN = -3.4028234663852886e38
FD_LSE_SENTINEL = -1.9999999360571385e38  # reference bit pattern0xFF167699

# Split-KV execution: long-KV S2 core split and combine. FD engages only
# when the whole launch has fewer m-tiles than cube cores AND the tile's
# KV sweep spans at least FD_MIN_NTILES n-tiles (below that the split
# overhead outweighs the parallelism; every bit-exact redline and the
# hang sentinels -- n_end <= 2 -- stay on the FA path).  The split factor
# is capped so the partial slices of one launch stay within one resident
# wave, and FD_MAX_K also bounds the fd_scale buffer rows.
FD_MIN_NTILES = 4
FD_MAX_K = 32

# Events for storage with repeated reads or explicitly aliased lifetimes.
FD_GLOBAL_SYNC_FLAG_IDS = (0, 1, 2)
KV_LOAD_EVENT_BASE = 4
AIV_SYNC_OFFSET = 16


@jit
def _clamp_nonneg(value):
    """max(0, value) for a dynamic Int64."""
    return (0 if value < 0 else value)


@jit
def _clamp_range(value, kv_tile_end):
    """clamp(value, 0, kv_tile_end) for a dynamic Int64."""
    v = (0 if value < 0 else value)
    return (kv_tile_end if v > kv_tile_end else v)


@jit
def _causal_row_valid_fast(causal_base, head0, rows_per_query_token, row, actual_n,
                           gs1=False):
    """Per-row RIGHT_DOWN causal valid column count (mask_mode=3).

    Fold row (tile-local ``row``) lives in token ``t_first + (query_head_start+row)//N``
    (the (s, n) s-major fold); RIGHT_DOWN makes it see KV columns
    [0, t + (kv_length - query_length)], so within n-tile kv_tile_idx its visible prefix is
    valid_columns = clamp(t_first + (query_head_start+row)//N + delta + 1 - kv_tile_idx*128, 0, actual_n)
    with ``causal_base`` bundling t_first + delta + 1 - kv_tile_idx*128 and
    actual_n already carrying the batch_idx KV clip (_valid_kv_columns).  The token
    staircase needs a dynamic // dynamic integer division; bisheng rejects
    div of bitwidth > 32 (probe p1c_causal_map.py), so the division runs
    in i32 (values are tiny: head offsets and row indices) and extends
    back to i64 for the clamp arithmetic.  valid_columns == 0 rows (above the causal
    diagonal, kv_length < query_length) reduce to max = FP32_MIN / sum = 0 / P = 0 --
    output 0, and the lse() epilogue turns the natural -inf into the
    +inf no-attention sentinel when sum == 0 (probe: vln(0) == -inf,
    empty-mask reduce_sum == 0).
    """
    # Callers normalize head0 to [0, rows_per_query_token).  Each vector tile spans at
    # most 32 rows, so a seven-step bounded subtract computes both quotient
    # and remainder without a dynamic divide in every softmax row.
    rem = dtypes.uint32(head0) + dtypes.uint32(row)
    quotient = dtypes.uint32(0)
    period = dtypes.uint32(rows_per_query_token)
    for shift in (6, 5, 4, 3, 2, 1, 0):
        safe = period <= dtypes.uint32(0xffffffff >> shift)
        step = period * dtypes.uint32(1 << shift)
        take = (rem >= step if safe else False)
        rem = (rem - step if take else rem)
        quotient = (quotient + dtypes.uint32(1 << shift) if take else quotient)
    tok = rem if const_expr(gs1) else quotient
    return _clamp_range(causal_base + dtypes.int64(tok), actual_n)


def _causal_row_valid(causal_base, head0, rows_per_query_token, row, actual_n, gs1=False,
                      fast=False):
    """Select the causal index spelling at compile time by data layout."""
    if const_expr(fast):
        return _causal_row_valid_fast(
            causal_base, head0, rows_per_query_token, row, actual_n, gs1
        )
    if const_expr(gs1):
        tok = dtypes.int64(dtypes.int32(head0 + row) % dtypes.int32(rows_per_query_token))
    else:
        tok = dtypes.int64(dtypes.int32(head0 + row) // dtypes.int32(rows_per_query_token))
    return _clamp_range(causal_base + tok, actual_n)




# ============================================================================
# AttentionCubeStage module: QK chunk loop and PV output_band_idx loop.
# ============================================================================



from cannbotdsl.ops.sync import (
    cube_sync_intra_wait,
    vec_sync_intra_arrive,
    vec_sync_notify,
    vec_sync_wait,
    vec_sync_all,
)


class AttentionVectorStage:
    def __init__(self, tile_vec_m, tile_n, tile_d, vector_subblock_idx, input_dtype,
                 softmax_state_depth=3, should_return_lse=True, mask_mode=0, enable_split_kv_workspace=False,
                 use_padded_lse_storage=False):
        self.tile_vec_m = tile_vec_m
        self.tile_m = tile_vec_m * 2
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.vector_subblock_idx = vector_subblock_idx
        self.input_dtype = input_dtype
        self.should_return_lse = should_return_lse
        self.mask_mode = mask_mode
        self.template_mask = False
        self.mask_active = False
        # P's padded NZ allocation ends at 237824. These two 4 KiB tiles stay
        # below the first softmax state at 246016 (also for 32 AIV rows).
        self.mask_rows = dsl.make_buffer(dsl.reinterpret(dsl.UB.view(262144), dtype=dtypes.int8, shape=(
            TILE_VEC_M, QK_UB_ROW_STRIDE), offset=UB_MASK_HEAD_ADDR, data_format='nd'))
        # Static row views retain the mask DMA/VF synchronization resource.
        self.mask_row_aliases = [
            reinterpret(
                self.mask_rows, shape=(1, QK_UB_ROW_STRIDE),
                offset=row * QK_UB_ROW_STRIDE,
            )
            for row in range(TILE_VEC_M)
        ]
        self.uses_gs1_row_order = False
        self.fast_causal = False
        self.enable_split_kv_workspace = enable_split_kv_workspace
        self.use_padded_lse_storage = True
        self.output_accumulator = dsl.make_channel([dsl.reinterpret(dsl.UB.view(262144), dtype=dtypes.float32, shape=(
            tile_vec_m, TILE_D), offset=UB_OUTPUT_ACCUM_ADDR, data_format='nd')]).produce()
        self.packed_output_alias = dsl.make_buffer(dsl.reinterpret(dsl.UB.view(262144), dtype=dtypes.float32, shape=(
            tile_vec_m, TILE_D // 2), offset=UB_OUTPUT_ACCUM_ADDR, data_format='nd'))
        # Final normalization retires the FP32 accumulator and packs the same
        # storage into the 16-bit output. The owning Buffer synchronizes the
        # MTE3 writeback before the next PV update reuses this address.
        self.probability_ub = dsl.make_channel([dsl.reinterpret(
            dsl.UB.view(262144),
            dtype=input_dtype, shape=(tile_vec_m, QK_UB_ROW_STRIDE),
            offset=UB_P_ADDR, data_format='nz',
            n1_pad=(TILE_VEC_M + 1 - tile_vec_m) * FRACTAL_SIZE)]).produce()
        self.softmax_rescale_slots = [
            dsl.make_buffer(dsl.reinterpret(dsl.UB.view(262144), dtype=dtypes.float32, shape=(
                tile_vec_m, 1), offset=UB_SOFTMAX_EXP_BASE + i * UB_STATE_SLOT_STRIDE, data_format='nd'))
            for i in range(PIPELINE_DEPTH)
        ]
        # Row-wise softmax and partial-result DMA use the same state roots.
        self.softmax_sum_slots = [
            dsl.make_buffer(dsl.reinterpret(dsl.UB.view(262144), dtype=dtypes.float32, shape=(
                tile_vec_m, 1), offset=UB_SOFTMAX_SUM_BASE + i * UB_STATE_SLOT_STRIDE, data_format='nd'))
            for i in range(PIPELINE_DEPTH)
        ]
        self.softmax_max_slots = [
            dsl.make_buffer(dsl.reinterpret(dsl.UB.view(262144), dtype=dtypes.float32, shape=(
                tile_vec_m, 1), offset=UB_SOFTMAX_MAX_BASE + i * UB_STATE_SLOT_STRIDE, data_format='nd'))
            for i in range(PIPELINE_DEPTH)
        ]
        self.lse_output_ub = dsl.make_channel([dsl.reinterpret(dsl.UB.view(262144), dtype=dtypes.float32, shape=(
            tile_vec_m, 1), offset=UB_LSE_ADDR, data_format='nd')]).produce()
        self.lse_output_alias = dsl.make_buffer(dsl.reinterpret(dsl.UB.view(
            262144), dtype=dtypes.float32, shape=(tile_vec_m, 1), offset=UB_LSE_ADDR, data_format='nd'))
        # Every layout uses one 32-byte UB slot per LSE value so split
        # writebacks retain an aligned MTE3 source across token boundaries.
        self.padded_lse_buffer = dsl.make_buffer(dsl.reinterpret(dsl.UB.view(262144), dtype=dtypes.float32, shape=(
            tile_vec_m, FD_STATE_COLUMNS), offset=UB_FD_LSE_ADDR, data_format='nd'))
        self.candidate_max = dsl.make_buffer(dsl.reinterpret(dsl.UB.view(
            262144), dtype=dtypes.float32, shape=(tile_vec_m, 1), offset=UB_NEW_MAX_ADDR, data_format='nd'))
        self.candidate_sum = dsl.make_buffer(dsl.reinterpret(dsl.UB.view(
            262144), dtype=dtypes.float32, shape=(tile_vec_m, 1), offset=UB_NEW_SUM_ADDR, data_format='nd'))
        # FD is execution-disjoint from FA; it reuses retired CV buffers.

    def _nz_params(self):
        """Derive NZ fractal addressing from p_ub physical layout stride."""
        s = self.probability_ub.physical_stride
        s_n1, s_m1, s_m0, s_n0 = s[0], s[1], s[2], s[3]
        m0 = s_m1 // s_m0
        query_head_start = s_m0 // s_n0
        return m0, s_m1, s_m0, s_n1 // query_head_start


    def _softmax_fold_row(
        self,
        qk_scores_handoff,
        base,
        nz_off,
        max_brc_buf,
        row,
        ve_mask,
        vo_mask,
        b16,
        b16_full,
        full_lane_mask,
        block_stride,
    ):
        """Per-row fold: deinterleave-load compute_qk, exp(sub max), cast to fp16,
        merge even/odd halves, store into p_ub NZ slot. Returns the exp
        halves for reduce_sum."""
        mx = rr.vload_brc(max_brc_buf, row)
        ve, vo = rr.vload_deinterleave(qk_scores_handoff, base, width="b32")
        ve = rr.vexp_sub(ve, mx, mask=ve_mask)
        vo = rr.vexp_sub(vo, mx, mask=vo_mask)
        he = rr.vcast(ve, self.input_dtype, mask=ve_mask, reg_layout=rr.RegLayout.ZERO)
        ho = rr.vcast(vo, self.input_dtype, mask=vo_mask, reg_layout=rr.RegLayout.ONE)
        merged = rr.vor(he, ho, mask=b16)
        rr.vstore_strided(
            self.probability_ub,
            nz_off,
            merged,
            b16_full,
            block_stride=block_stride,
            repeat_stride=0,
        )
        return ve, vo


    def _pass_a_row(
        self,
        qk_scores_handoff,
        scale,
        sm_max_dst,
        row,
        row_stride,
        half0_mask,
        half1_mask,
        full_lane_mask,
    ):
        """Scale compute_qk row in place, compute rowmax -> sm_max_dst[row].

        The row max reduces over the VALID columns only (review_report.md
        R1). v0/v1 zero-fill their masked-element_offset lanes (exec_zeroing), so an
        unmasked max on a tail tile leaks the padding columns' 0.0 scores
        into the max; with all-negative valid scores the leaked 0 inflates
        the max and P = exp(s - 0) underflows to 0 after the fp16 cast,
        zeroing whole output rows (LSE stays right: max+ln(sum) cancels).
        Replace the masked-element_offset lanes with FP32_MIN before the reduction,
        so a fully masked half (valid_n <= 64) contributes only the
        identity and never wins the max. valid_n >= 1 keeps >= 1 finite
        lane in half 0, so the reduced max stays finite. Full tiles
        (valid_n == 128) select through to v0/v1 on every lane and keep
        the original numeric path.
        """
        base = row * row_stride
        v0 = rr.vmuls(rr.vload(qk_scores_handoff, base), scale, mask=half0_mask)
        v1 = rr.vmuls(rr.vload(qk_scores_handoff, base + VL_T), scale, mask=half1_mask)
        rr.vstore(qk_scores_handoff, base, v0, half0_mask)
        rr.vstore(qk_scores_handoff, base + VL_T, v1, half1_mask)
        neg_inf = rr.vdup_scalar(FP32_MIN, dtypes.float32, mask=full_lane_mask)
        v0m = rr.vselect(v0, neg_inf, cond_mask=half0_mask)
        v1m = rr.vselect(v1, neg_inf, cond_mask=half1_mask)
        rmax = rr.vreduce_max(rr.vmax(v0m, v1m, mask=full_lane_mask), mask=full_lane_mask)
        rr.vstore_first(sm_max_dst, row, rmax)


    @jit
    def _fold_rows(self, qk_scores_handoff, max_buf, sum_dst, rows, src_row_stride,
                   ve_mask, vo_mask, b16, b16_full, full_lane_mask,
                   m0, s_m1, s_m0, block_stride):
        """Fold loop body with caller-built (loop-invariant) masks.
        @jit so the row loop stays a traced (rolled) scf.for like the
        original _softmax_fold_loop -- a plain method cannot range() over
        an Int64 bound, and a static unroll would 32x the instruction
        footprint (vec fetch-bound risk).  The NZ params come as ARGS:
        _nz_params() divides dynamic strides, and a divsi inside the vf
        region is a hoist candidate the grouping pass rejects."""
        for row in range(rows):
            base = row * src_row_stride
            nz_off = (row // m0) * s_m1 + (row % m0) * s_m0
            ve, vo = self._softmax_fold_row(
                qk_scores_handoff, base, nz_off, max_buf, row,
                ve_mask, vo_mask, b16, b16_full, full_lane_mask, block_stride,
            )
            rsum = rr.vreduce_sum(
                rr.vadd(ve, vo, mask=full_lane_mask), mask=ve_mask
            )
            rr.vstore_first(sum_dst, row, rsum)
        # The caller publishes c1v1/v1c2 on PIPE.V/PIPE.MTE3 before any
        # consumer reloads these stores; that cross-pipe ordering replaces
        # the old per-tile trailing VST_VLD barrier.


    @jit
    def _softmax_fold_loop(
        self,
        qk_scores_handoff,
        max_buf,
        sum_dst,
        actual_n,
        rows,
        src_row_stride,
        N,
        m0,
        s_m1,
        s_m0,
        block_stride,
        causal_base,
        head0,
        rows_per_query_token,
    ):
        """Per-row fold loop: exp-sub-max into p_ub, reduce_sum into sum_dst.

        Under mask_mode=3 each row narrows its exp/cast/store masks to its
        own causal prefix valid_columns(row) (see _causal_row_valid); masked-element_offset
        lanes are zero-filled by exec_zeroing, so P carries exact zeros in
        the causally-invalid columns and the PV contraction needs no
        causal awareness."""
        for row in range(rows):
            full_lane_mask, _ = rr.update_mask(VL_T, elem_bits=32)
            b16_full, _ = rr.update_mask(N, elem_bits=16)
            if const_expr(self.mask_mode == 3):
                valid_columns = _causal_row_valid(
                    causal_base, head0, rows_per_query_token, row, actual_n, self.uses_gs1_row_order,
                    self.fast_causal
                )
            else:
                valid_columns = actual_n
            ve_mask, _ = rr.update_mask((valid_columns + 1) // 2, elem_bits=32)
            vo_mask, _ = rr.update_mask(valid_columns // 2, elem_bits=32)
            b16, _ = rr.update_mask(valid_columns, elem_bits=16)
            base = row * src_row_stride
            nz_off = (row // m0) * s_m1 + (row % m0) * s_m0
            ve, vo = self._softmax_fold_row(
                qk_scores_handoff,
                base,
                nz_off,
                max_buf,
                row,
                ve_mask,
                vo_mask,
                b16,
                b16_full,
                full_lane_mask,
                block_stride,
            )
            rsum = rr.vreduce_sum(rr.vadd(ve, vo, mask=full_lane_mask), mask=ve_mask)
            rr.vstore_first(sum_dst, row, rsum)
        # P publication and the separate state-post VF provide the required
        # ordering after this loop.


    def _softmax_rest_tail(self, sm_max, sm_sum, sm_exp, active_row_mask):
        """Online-softmax running-state update."""
        old_max = rr.vload(sm_max, 0)
        new_max = rr.vload(self.candidate_max, 0)
        se = rr.vexp_sub(old_max, new_max, mask=active_row_mask)  # exp(old-new)
        rr.vstore(sm_exp, 0, se, active_row_mask)
        rr.vstore(sm_max, 0, new_max, active_row_mask)  # max = new_max
        old_sum = rr.vload(sm_sum, 0)
        new_sum = rr.vload(self.candidate_sum, 0)
        ss = rr.vmadd(old_sum, se, new_sum, mask=active_row_mask)
        rr.vstore(sm_sum, 0, ss, active_row_mask)


    def store_probability_tile(self, probability_l1_destination, valid_query_rows):
        """p_ub -> the explicit per-AIV M tile of the shared L1 slot."""
        destination = reinterpret(
            probability_l1_destination,
            shape=make_tiler((valid_query_rows, TILE_N), alignment=(1, 1)),
        )
        mem_copy(
            destination,
            tile_slice(self.probability_ub, (self.tile_vec_m, TILE_N), (0, 0)),
            engine=make_copy_engine(split_axis=0),
            part_id=self.vector_subblock_idx,
        )




    @jit
    def compute_lse(self, softmax_max_buffer, softmax_sum_buffer):
        """lse_ub[row] = sm_max[row] + ln(sm_sum[row]) (final tile only).

        No-attention rows (sum == 0: causal rows above the diagonal, or a
        KV tile narrower than the causal prefix) carry the reference
        ClearOutput sentinel: the 3e99 "invalid" marker narrows to +inf in
        fp32.  vdup_scalar rejects non-finite literals, so +inf is produced
        at runtime as vneg(-inf) -- the natural sum==0 value is
        FP32_MIN + vln(0) = -inf, whose negation is +inf."""
        with vf(mode="simd"):
            full_lane_mask, _ = rr.update_mask(VL_T, elem_bits=32)
            for row in range(self.tile_vec_m):
                mx = rr.vload_brc(softmax_max_buffer, row)
                sm = rr.vload_brc(softmax_sum_buffer, row)
                invalid = rr.vcmp_eq_scalar(mx, FP32_MIN, mask=full_lane_mask)
                zero = rr.vdup_scalar(0.0, dtypes.float32, mask=full_lane_mask)
                sm = rr.vselect(zero, sm, cond_mask=invalid)
                lnv = rr.vln(sm, mask=full_lane_mask)
                val = rr.vadd(mx, lnv, mask=full_lane_mask)
                is_zero = rr.vcmp_eq_scalar(sm, 0.0, mask=full_lane_mask)
                val = rr.vselect(
                    rr.vneg(val, mask=full_lane_mask), val, cond_mask=is_zero
                )
                if const_expr(self.use_padded_lse_storage):
                    rr.vstore_first(self.padded_lse_buffer, row * FD_STATE_COLUMNS, val)
                else:
                    rr.vstore_first(self.lse_output_ub, row, val)


    @jit
    def _clear_inactive_rows_vf(self, rows, sm_max, sm_sum,
                                m0, s_m1, s_m0, block_stride):
        # Inlined into the consuming softmax VF before publishing P to L1.
        full_lane_mask, _ = rr.update_mask(VL_T, elem_bits=32)
        pfull, _ = rr.update_mask(self.tile_n, elem_bits=16)
        pzero = rr.vdup_scalar(0.0, self.input_dtype, mask=pfull)
        zero = rr.vdup_scalar(0.0, dtypes.float32, mask=full_lane_mask)
        neg = rr.vdup_scalar(FP32_MIN, dtypes.float32, mask=full_lane_mask)
        for row in range(rows, self.tile_vec_m):
            nz_off = (row // m0) * s_m1 + (row % m0) * s_m0
            rr.vstore_strided(self.probability_ub, nz_off, pzero, pfull,
                              block_stride=block_stride, repeat_stride=0)
            rr.vstore_first(sm_max, row, neg)
            rr.vstore_first(sm_sum, row, zero)
        rr.vmem_bar("vst_vld")


    @jit
    def clear_inactive_output_rows(self, valid_vector_rows):
        with vf(mode="simd"):
            full_lane_mask, _ = rr.update_mask(VL_T, elem_bits=32)
            zero = rr.vdup_scalar(0.0, dtypes.float32, mask=full_lane_mask)
            for row in range(valid_vector_rows, self.tile_vec_m):
                for col in tuple(range(0, self.tile_d, VL_T)):
                    rr.vstore(self.output_accumulator, row * self.tile_d + col, zero, full_lane_mask)
            rr.vmem_bar("vst_vld")


    @jit
    def load_template(self, mask_gm, start, rows, query_length, kv_length, heads, n_start, valid_n):
        """Expand the custom kernel's M96/48 compressed-mask mapping.

        The attention pipeline uses M96, but the public mask semantics are
        defined by the reference M64 tile split.  Reconstruct that split from
        each global folded row so changing the compute tile does not change
        which user mask element is consumed.
        """
        start = dtypes.int32(start)
        rows = dtypes.int32(rows)
        query_length = dtypes.int32(query_length)
        kv_length = dtypes.int32(kv_length)
        heads = dtypes.int32(heads)
        n_start = dtypes.int32(n_start)
        valid_n = dtypes.int32(valid_n)
        first = start // heads
        if const_expr(self.uses_gs1_row_order):
            first = start % query_length
        active = n_start + valid_n > first + kv_length - query_length
        if const_expr(self.uses_gs1_row_order):
            if first + rows > query_length:
                active = True
        self.mask_active = active
        self.mask_valid_n = valid_n
        self.mask_start = start
        self.mask_s1 = query_length
        self.mask_s2 = kv_length
        self.mask_heads = heads
        self.mask_n_start = n_start
        if active:
            total_rows = query_length * heads
            align_n = ((valid_n + 31) // 32) * 32
            for row in tuple(range(TILE_VEC_M)):
                if row < rows:
                    global_row = start + row
                    if const_expr(self.uses_gs1_row_order):
                        # GS1/BNSD folds (head, token).  A vector half may
                        # cross one head boundary; the reference copies the
                        # tail of that head and then wraps to token zero.
                        token_start = start % query_length
                        head_count = _clamp_range(query_length - token_start, rows)
                        in_head = row < head_count
                        group = (token_start if in_head else 0)
                        relative = (row if in_head else row - head_count)
                        copy_count = rows
                    else:
                        # SG folds (token, head).  Recover the owning M96
                        # tile and its two M48 vector halves.
                        ref_base = global_row // self.tile_m * self.tile_m
                        ref_rows = _clamp_range(
                            total_rows - ref_base, self.tile_m
                        )
                        first_count = (ref_rows + 1) // 2
                        in_first = global_row < ref_base + first_count
                        group_start = (ref_base if in_first else ref_base + first_count)
                        group_count = (first_count if in_first else ref_rows - first_count)
                        group_first = group_start // heads
                        group_last = (
                            group_start + group_count - 1
                        ) // heads
                        head_count = (group_count if group_first == group_last else (
                            heads - group_start % heads) % heads)
                        relative_row = global_row - group_start
                        in_head = relative_row < head_count
                        group = group_first + (0 if in_head else 1 if head_count > 0 else 0)
                        relative = (0 if in_head else (relative_row - head_count) // heads)
                        copy_count = group_count
                    delta = kv_length - query_length + group - n_start
                    src_row = relative + (0 if delta < 0 else delta if delta < align_n else align_n)
                    src_col = ((-delta if -delta < copy_count else copy_count) if delta < 0 else 0)
                    span = make_tiler((1, valid_n), alignment=(1, 1))
                    # The dynamic coordinate must stay on the Identity GM
                    # tensor.  Routing it through a sliced view drops the
                    # dynamic offset during lowering.
                    src = tile_slice(mask_gm, span, (src_row, src_col))
                    mem_copy(reinterpret(self.mask_row_aliases[row], shape=span), src)

    @jit
    def _template_select(self, value, row, column, full_lane_mask, apply_mask):
        if const_expr(apply_mask):
            bits = rr.vload_unpack(
                self.mask_rows, row * 128 + column,
                unpack_mode="b8_to_b32",
            )
            low_bit = rr.vdup_scalar(1, dtypes.uint32, mask=full_lane_mask)
            bits = rr.vbitwise_and(bits, low_bit, mask=full_lane_mask)
            if const_expr(not self.uses_gs1_row_order):
                global_row = (
                    dtypes.int32(self.mask_start) + dtypes.int32(row)
                )
                ref_base = global_row // self.tile_m * self.tile_m
                ref_rows = _clamp_range(
                    dtypes.int32(self.mask_s1)
                    * dtypes.int32(self.mask_heads)
                    - ref_base,
                    self.tile_m,
                )
                ref_last = ref_base + ref_rows - 1
                ref_stop = _clamp_range(
                    ref_last // dtypes.int32(self.mask_heads)
                    + dtypes.int32(self.mask_s2)
                    - dtypes.int32(self.mask_s1)
                    + 1,
                    dtypes.int32(self.mask_s2),
                )
                scheduled = dtypes.int32(self.mask_n_start) < ref_stop
                force_mask, _ = rr.update_mask(
                    (0 if scheduled else VL_T), elem_bits=32
                )
                ones = rr.vdup_scalar(1, dtypes.uint32, mask=full_lane_mask)
                bits = rr.vselect(ones, bits, cond_mask=force_mask)
            keep = rr.vcmp_eq_scalar(bits, 0, mask=full_lane_mask)
            negative = rr.vdup_scalar(FP32_MIN, dtypes.float32, mask=full_lane_mask)
            value = rr.vselect(value, negative, cond_mask=keep)
        return value

    @jit
    def _softmax_register_rows(
        self, qk_scores_handoff, scale, softmax_state_slot: int,
        is_first, per_row_causal, valid_columns, causal_base, head0,
        rows_per_query_token,
    ):
        # Select the tile path only from FP32 vector capacity, independently
        # of specific batch, query, or KV lengths.
        if qk_scores_handoff.shape[1] <= VL_T:
            self._softmax_register_narrow_rows(
                qk_scores_handoff, scale, softmax_state_slot, is_first,
                per_row_causal, valid_columns, causal_base, head0, rows_per_query_token,
            )
        else:
            self._softmax_register_wide_rows(
                qk_scores_handoff, scale, softmax_state_slot, is_first,
                per_row_causal, valid_columns, causal_base, head0, rows_per_query_token,
            )

    @jit
    def _softmax_register_narrow_rows(
        self, qk_scores_handoff, scale, softmax_state_slot: int,
        is_first, per_row_causal, valid_columns, causal_base, head0,
        rows_per_query_token,
    ):
        """Load and repack a one-register FP32 row into the same NZ layout."""
        sm_max = self.softmax_max_slots[softmax_state_slot]
        sm_sum = self.softmax_sum_slots[softmax_state_slot]
        rows, actual_n = qk_scores_handoff.shape
        row_stride = qk_scores_handoff.physical_stride[0]
        m0, s_m1, s_m0, block_stride = self._nz_params()
        if const_expr(per_row_causal):
            if const_expr(not self.uses_gs1_row_order):
                causal_base = causal_base + dtypes.int64(
                    dtypes.int32(head0) // dtypes.int32(rows_per_query_token)
                )
            head0 = dtypes.int64(
                dtypes.int32(head0) % dtypes.int32(rows_per_query_token)
            )
        with vf(mode="simd"):
            self._clear_inactive_rows_vf(
                rows, sm_max, sm_sum, m0, s_m1, s_m0, block_stride
            )
            full_mask, _ = rr.update_mask(VL_T, elem_bits=32)
            full_output_mask, _ = rr.update_mask(self.tile_n, elem_bits=16)
            negative = rr.vdup_scalar(FP32_MIN, dtypes.float32, mask=full_mask)
            zero = rr.vdup_scalar(0.0, dtypes.float32, mask=full_mask)
            zero_half = rr.vdup_scalar(0.0, self.input_dtype, mask=full_output_mask)
            causal_row_position = dtypes.int32(head0)
            causal_token_offset = dtypes.int32(0)
            causal_row_period = dtypes.int32(rows_per_query_token)
            for row in range(rows):
                if const_expr(per_row_causal):
                    # Advance quotient and remainder per row to avoid dynamic
                    # division without specializing for a head count.
                    row_token_offset = causal_token_offset
                    if const_expr(self.uses_gs1_row_order):
                        row_token_offset = causal_row_position
                    row_columns = _clamp_range(
                        causal_base + dtypes.int64(row_token_offset), actual_n
                    )
                else:
                    row_columns = valid_columns
                row_mask, _ = rr.update_mask(row_columns, elem_bits=32)
                scores = rr.vload(qk_scores_handoff, row * row_stride)
                scores = rr.vmuls(scores, scale, mask=row_mask)
                for_max = rr.vselect(scores, negative, cond_mask=row_mask)
                row_max = rr.vreduce_max(for_max, mask=full_mask)
                max_broadcast = rr.vdup(row_max, mask=full_mask)
                if const_expr(is_first):
                    rr.vstore_first(sm_max, row, row_max)
                else:
                    max_broadcast = rr.vmax(
                        rr.vload_brc(sm_max, row), max_broadcast, mask=full_mask
                    )
                    rr.vstore_first(self.candidate_max, row, max_broadcast)
                probabilities = rr.vexp_sub(scores, max_broadcast, mask=row_mask)
                probabilities = rr.vselect(probabilities, zero, cond_mask=row_mask)
                narrowed = rr.vcast(
                    probabilities, self.input_dtype, mask=full_mask,
                    reg_layout=rr.RegLayout.ZERO,
                )
                # Store one register of probabilities in the lower half and
                # zero the upper half, preserving the padded NZ row stride.
                packed, unused = rr.vdeinterleave(narrowed, zero_half)
                nz_offset = (row // m0) * s_m1 + (row % m0) * s_m0
                rr.vstore_strided(
                    self.probability_ub, nz_offset, packed, full_output_mask,
                    block_stride=block_stride, repeat_stride=0,
                )
                row_sum = rr.vreduce_sum(probabilities, mask=row_mask)
                if const_expr(is_first):
                    rr.vstore_first(sm_sum, row, row_sum)
                else:
                    rr.vstore_first(self.candidate_sum, row, row_sum)
                if const_expr(per_row_causal):
                    next_row_position = causal_row_position + dtypes.int32(1)
                    crosses_token = next_row_position >= causal_row_period
                    causal_row_position = (next_row_position - \
                                           causal_row_period if crosses_token else next_row_position)
                    causal_token_offset += (dtypes.int32(1) if crosses_token else dtypes.int32(0))
            # Make this VF output visible before publishing it; the Channel
            # still owns the cross-core handoff.
            rr.vmem_bar("vst_vld")

    @jit
    def _softmax_register_wide_rows(
        self, qk_scores_handoff, scale, softmax_state_slot: int,
        is_first, per_row_causal, valid_columns, causal_base, head0,
        rows_per_query_token,
    ):
        """Compute scaling, max, exp, sum, and P storage per row in registers."""
        sm_max = self.softmax_max_slots[softmax_state_slot]
        sm_sum = self.softmax_sum_slots[softmax_state_slot]
        rows, actual_n = qk_scores_handoff.shape
        row_stride = qk_scores_handoff.physical_stride[0]
        m0, s_m1, s_m0, block_stride = self._nz_params()
        if const_expr(per_row_causal):
            if const_expr(not self.uses_gs1_row_order):
                causal_base = causal_base + dtypes.int64(
                    dtypes.int32(head0) // dtypes.int32(rows_per_query_token)
                )
            head0 = dtypes.int64(
                dtypes.int32(head0) % dtypes.int32(rows_per_query_token)
            )
        even_count = (valid_columns + 1) // 2
        odd_count = valid_columns // 2
        with vf(mode="simd"):
            self._clear_inactive_rows_vf(
                rows, sm_max, sm_sum, m0, s_m1, s_m0, block_stride
            )
            full_mask, _ = rr.update_mask(VL_T, elem_bits=32)
            full_output_mask, _ = rr.update_mask(self.tile_n, elem_bits=16)
            negative = rr.vdup_scalar(FP32_MIN, dtypes.float32, mask=full_mask)
            causal_row_position = dtypes.int32(head0)
            causal_token_offset = dtypes.int32(0)
            causal_row_period = dtypes.int32(rows_per_query_token)
            for row in range(rows):
                if const_expr(per_row_causal):
                    # Advance quotient and remainder per row to avoid dynamic
                    # division without specializing for a head count.
                    row_token_offset = causal_token_offset
                    if const_expr(self.uses_gs1_row_order):
                        row_token_offset = causal_row_position
                    row_columns = _clamp_range(
                        causal_base + dtypes.int64(row_token_offset), actual_n
                    )
                    even_mask, _ = rr.update_mask((row_columns + 1) // 2, elem_bits=32)
                    odd_mask, _ = rr.update_mask(row_columns // 2, elem_bits=32)
                    output_mask, _ = rr.update_mask(row_columns, elem_bits=16)
                else:
                    even_mask, _ = rr.update_mask(even_count, elem_bits=32)
                    odd_mask, _ = rr.update_mask(odd_count, elem_bits=32)
                    output_mask, _ = rr.update_mask(valid_columns, elem_bits=16)
                even_scores, odd_scores = rr.vload_deinterleave(
                    qk_scores_handoff, row * row_stride, width="b32"
                )
                even_scores = rr.vmuls(even_scores, scale, mask=even_mask)
                odd_scores = rr.vmuls(odd_scores, scale, mask=odd_mask)
                even_for_max = rr.vselect(even_scores, negative, cond_mask=even_mask)
                odd_for_max = rr.vselect(odd_scores, negative, cond_mask=odd_mask)
                row_max = rr.vreduce_max(
                    rr.vmax(even_for_max, odd_for_max, mask=full_mask),
                    mask=full_mask,
                )
                max_broadcast = rr.vdup(row_max, mask=full_mask)
                if const_expr(is_first):
                    rr.vstore_first(sm_max, row, row_max)
                else:
                    max_broadcast = rr.vmax(
                        rr.vload_brc(sm_max, row), max_broadcast, mask=full_mask
                    )
                    rr.vstore_first(self.candidate_max, row, max_broadcast)
                even_exp = rr.vexp_sub(even_scores, max_broadcast, mask=even_mask)
                odd_exp = rr.vexp_sub(odd_scores, max_broadcast, mask=odd_mask)
                even_half = rr.vcast(
                    even_exp, self.input_dtype, mask=even_mask,
                    reg_layout=rr.RegLayout.ZERO,
                )
                odd_half = rr.vcast(
                    odd_exp, self.input_dtype, mask=odd_mask,
                    reg_layout=rr.RegLayout.ONE,
                )
                packed = rr.vor(even_half, odd_half, mask=output_mask)
                nz_offset = (row // m0) * s_m1 + (row % m0) * s_m0
                rr.vstore_strided(
                    self.probability_ub, nz_offset, packed, full_output_mask,
                    block_stride=block_stride, repeat_stride=0,
                )
                row_sum = rr.vreduce_sum(
                    rr.vadd(even_exp, odd_exp, mask=full_mask), mask=even_mask
                )
                if const_expr(is_first):
                    rr.vstore_first(sm_sum, row, row_sum)
                else:
                    rr.vstore_first(self.candidate_sum, row, row_sum)
                if const_expr(per_row_causal):
                    next_row_position = causal_row_position + dtypes.int32(1)
                    crosses_token = next_row_position >= causal_row_period
                    causal_row_position = (next_row_position - \
                                           causal_row_period if crosses_token else next_row_position)
                    causal_token_offset += (dtypes.int32(1) if crosses_token else dtypes.int32(0))
            # Make this VF output visible before publishing it; the Channel
            # still owns the cross-core handoff.
            rr.vmem_bar("vst_vld")

    @jit
    def _sm_first_fast_rows(self, qk_scores_handoff, scale, softmax_state_slot: int, apply_mask=False):
        if const_expr(apply_mask):
            self._sm_first_fast_rows_template(qk_scores_handoff, scale, softmax_state_slot, True)
        else:
            self._softmax_register_rows(
                qk_scores_handoff, scale, softmax_state_slot,
                True, False, qk_scores_handoff.shape[1],
                0, 0, 1,
            )

    @jit
    def _sm_first_fast_rows_template(self, qk_scores_handoff, scale, softmax_state_slot: int, apply_mask=False):
        """mask 0 / uniform causal tile, every lane visible: no select."""
        sm_max = self.softmax_max_slots[softmax_state_slot]
        sm_sum = self.softmax_sum_slots[softmax_state_slot]
        m0, s_m1, s_m0, block_stride = self._nz_params()
        with vf(mode="simd"):
            rows, _n = qk_scores_handoff.shape
            self._clear_inactive_rows_vf(rows, sm_max, sm_sum,
                                         m0, s_m1, s_m0, block_stride)
            stride = qk_scores_handoff.physical_stride[0]
            full_lane_mask, _ = rr.update_mask(VL_T, elem_bits=32)
            b16_full, _ = rr.update_mask(self.tile_n, elem_bits=16)
            for row in range(rows):
                base = row * stride
                v0 = rr.vmuls(rr.vload(qk_scores_handoff, base), scale, mask=full_lane_mask)
                v1 = rr.vmuls(
                    rr.vload(qk_scores_handoff, base + VL_T),
                    scale,
                    mask=full_lane_mask,
                )
                v0 = self._template_select(v0, row, 0, full_lane_mask, apply_mask)
                v1 = self._template_select(v1, row, VL_T, full_lane_mask, apply_mask)
                rr.vstore(qk_scores_handoff, base, v0, full_lane_mask)
                rr.vstore(qk_scores_handoff, base + VL_T, v1, full_lane_mask)
                rmax = rr.vreduce_max(
                    rr.vmax(v0, v1, mask=full_lane_mask), mask=full_lane_mask
                )
                rr.vstore_first(sm_max, row, rmax)
            rr.vmem_bar("vst_vld")
            self._fold_rows(
                qk_scores_handoff, sm_max, sm_sum, rows, stride,
                full_lane_mask, full_lane_mask, b16_full, b16_full, full_lane_mask,
                m0, s_m1, s_m0, block_stride,
            )


    @jit
    def _sm_first_masked_rows(self, qk_scores_handoff, scale, softmax_state_slot: int, valid_columns, apply_mask=False):
        if const_expr(apply_mask):
            self._sm_first_masked_rows_template(qk_scores_handoff, scale, softmax_state_slot, valid_columns, True)
        else:
            self._softmax_register_rows(
                qk_scores_handoff, scale, softmax_state_slot,
                True, False, valid_columns,
                0, 0, 1,
            )

    @jit
    def _sm_first_masked_rows_template(
            self,
            qk_scores_handoff,
            scale,
            softmax_state_slot: int,
            valid_columns,
            apply_mask=False):
        """mask 0 / uniform causal tile, valid_columns < 128 visible lanes."""
        sm_max = self.softmax_max_slots[softmax_state_slot]
        sm_sum = self.softmax_sum_slots[softmax_state_slot]
        # Compute lane counts before entering the VF region to keep invariant
        # integer division outside vector folding. The grouping pass cannot
        # hoist these non-pure operations across VF groups.
        ve_cnt = (valid_columns + 1) // 2
        vo_cnt = valid_columns // 2
        h1_cnt = _clamp_nonneg(valid_columns - VL_T)
        m0, s_m1, s_m0, block_stride = self._nz_params()
        with vf(mode="simd"):
            rows, _n = qk_scores_handoff.shape
            self._clear_inactive_rows_vf(rows, sm_max, sm_sum,
                                         m0, s_m1, s_m0, block_stride)
            stride = qk_scores_handoff.physical_stride[0]
            full_lane_mask, _ = rr.update_mask(VL_T, elem_bits=32)
            b16_full, _ = rr.update_mask(self.tile_n, elem_bits=16)
            h0m, _ = rr.update_mask(valid_columns, elem_bits=32)
            h1m, _ = rr.update_mask(h1_cnt, elem_bits=32)
            neg_inf = rr.vdup_scalar(FP32_MIN, dtypes.float32, mask=full_lane_mask)
            ve_mask, _ = rr.update_mask(ve_cnt, elem_bits=32)
            vo_mask, _ = rr.update_mask(vo_cnt, elem_bits=32)
            b16, _ = rr.update_mask(valid_columns, elem_bits=16)
            for row in range(rows):
                base = row * stride
                v0 = rr.vmuls(rr.vload(qk_scores_handoff, base), scale, mask=h0m)
                v1 = rr.vmuls(
                    rr.vload(qk_scores_handoff, base + VL_T), scale, mask=h1m
                )
                v0 = self._template_select(v0, row, 0, full_lane_mask, apply_mask)
                v1 = self._template_select(v1, row, VL_T, full_lane_mask, apply_mask)
                rr.vstore(qk_scores_handoff, base, v0, h0m)
                rr.vstore(qk_scores_handoff, base + VL_T, v1, h1m)
                v0m = rr.vselect(v0, neg_inf, cond_mask=h0m)
                v1m = rr.vselect(v1, neg_inf, cond_mask=h1m)
                rmax = rr.vreduce_max(
                    rr.vmax(v0m, v1m, mask=full_lane_mask), mask=full_lane_mask
                )
                rr.vstore_first(sm_max, row, rmax)
            rr.vmem_bar("vst_vld")
            self._fold_rows(
                qk_scores_handoff, sm_max, sm_sum, rows, stride,
                ve_mask, vo_mask, b16, b16_full, full_lane_mask,
                m0, s_m1, s_m0, block_stride,
            )


    @jit
    def _sm_rest_fast_rows(self, qk_scores_handoff, scale, softmax_state_slot: int,
                           rescale_state_slot: int, apply_mask=False):
        if const_expr(apply_mask):
            self._sm_rest_fast_rows_template(qk_scores_handoff, scale, softmax_state_slot, rescale_state_slot, True)
        else:
            self._softmax_register_rows(
                qk_scores_handoff, scale, softmax_state_slot,
                False, False, qk_scores_handoff.shape[1],
                0, 0, 1,
            )

    @jit
    def _sm_rest_fast_rows_template(self, qk_scores_handoff, scale, softmax_state_slot: int,
                                    rescale_state_slot: int, apply_mask=False):
        """softmax_rest, every lane visible."""
        sm_max = self.softmax_max_slots[softmax_state_slot]
        sm_sum = self.softmax_sum_slots[softmax_state_slot]
        sm_exp = self.softmax_rescale_slots[rescale_state_slot]
        m0, s_m1, s_m0, block_stride = self._nz_params()
        with vf(mode="simd"):
            rows, _n = qk_scores_handoff.shape
            self._clear_inactive_rows_vf(rows, sm_max, sm_sum,
                                         m0, s_m1, s_m0, block_stride)
            stride = qk_scores_handoff.physical_stride[0]
            full_lane_mask, _ = rr.update_mask(VL_T, elem_bits=32)
            b16_full, _ = rr.update_mask(self.tile_n, elem_bits=16)
            active_row_mask, _ = rr.update_mask(rows, elem_bits=32)
            for row in range(rows):
                base = row * stride
                v0 = rr.vmuls(rr.vload(qk_scores_handoff, base), scale, mask=full_lane_mask)
                v1 = rr.vmuls(
                    rr.vload(qk_scores_handoff, base + VL_T),
                    scale,
                    mask=full_lane_mask,
                )
                v0 = self._template_select(v0, row, 0, full_lane_mask, apply_mask)
                v1 = self._template_select(v1, row, VL_T, full_lane_mask, apply_mask)
                rr.vstore(qk_scores_handoff, base, v0, full_lane_mask)
                rr.vstore(qk_scores_handoff, base + VL_T, v1, full_lane_mask)
                rmax = rr.vreduce_max(
                    rr.vmax(v0, v1, mask=full_lane_mask), mask=full_lane_mask
                )
                rr.vstore_first(self.candidate_max, row, rmax)
            rr.vmem_bar("vst_vld")
            nm = rr.vmax(
                rr.vload(sm_max, 0), rr.vload(self.candidate_max, 0),
                mask=active_row_mask,
            )
            rr.vstore(self.candidate_max, 0, nm, active_row_mask)
            rr.vmem_bar("vst_vld")
            self._fold_rows(
                qk_scores_handoff, self.candidate_max, self.candidate_sum, rows, stride,
                full_lane_mask, full_lane_mask, b16_full, b16_full, full_lane_mask,
                m0, s_m1, s_m0, block_stride,
            )


    @jit
    def _sm_rest_masked_rows(self, qk_scores_handoff, scale, softmax_state_slot: int,
                             rescale_state_slot: int, valid_columns, apply_mask=False):
        if const_expr(apply_mask):
            self._sm_rest_masked_rows_template(qk_scores_handoff, scale, softmax_state_slot,
                                               rescale_state_slot, valid_columns, True)
        else:
            self._softmax_register_rows(
                qk_scores_handoff, scale, softmax_state_slot,
                False, False, valid_columns,
                0, 0, 1,
            )

    @jit
    def _sm_rest_masked_rows_template(self, qk_scores_handoff, scale, softmax_state_slot: int,
                                      rescale_state_slot: int, valid_columns, apply_mask=False):
        """softmax_rest, valid_columns < 128 visible lanes."""
        sm_max = self.softmax_max_slots[softmax_state_slot]
        sm_sum = self.softmax_sum_slots[softmax_state_slot]
        sm_exp = self.softmax_rescale_slots[rescale_state_slot]
        ve_cnt = (valid_columns + 1) // 2
        vo_cnt = valid_columns // 2
        h1_cnt = _clamp_nonneg(valid_columns - VL_T)
        m0, s_m1, s_m0, block_stride = self._nz_params()
        with vf(mode="simd"):
            rows, _n = qk_scores_handoff.shape
            self._clear_inactive_rows_vf(rows, sm_max, sm_sum,
                                         m0, s_m1, s_m0, block_stride)
            stride = qk_scores_handoff.physical_stride[0]
            full_lane_mask, _ = rr.update_mask(VL_T, elem_bits=32)
            b16_full, _ = rr.update_mask(self.tile_n, elem_bits=16)
            active_row_mask, _ = rr.update_mask(rows, elem_bits=32)
            h0m, _ = rr.update_mask(valid_columns, elem_bits=32)
            h1m, _ = rr.update_mask(h1_cnt, elem_bits=32)
            neg_inf = rr.vdup_scalar(FP32_MIN, dtypes.float32, mask=full_lane_mask)
            ve_mask, _ = rr.update_mask(ve_cnt, elem_bits=32)
            vo_mask, _ = rr.update_mask(vo_cnt, elem_bits=32)
            b16, _ = rr.update_mask(valid_columns, elem_bits=16)
            for row in range(rows):
                base = row * stride
                v0 = rr.vmuls(rr.vload(qk_scores_handoff, base), scale, mask=h0m)
                v1 = rr.vmuls(
                    rr.vload(qk_scores_handoff, base + VL_T), scale, mask=h1m
                )
                v0 = self._template_select(v0, row, 0, full_lane_mask, apply_mask)
                v1 = self._template_select(v1, row, VL_T, full_lane_mask, apply_mask)
                rr.vstore(qk_scores_handoff, base, v0, h0m)
                rr.vstore(qk_scores_handoff, base + VL_T, v1, h1m)
                v0m = rr.vselect(v0, neg_inf, cond_mask=h0m)
                v1m = rr.vselect(v1, neg_inf, cond_mask=h1m)
                rmax = rr.vreduce_max(
                    rr.vmax(v0m, v1m, mask=full_lane_mask), mask=full_lane_mask
                )
                rr.vstore_first(self.candidate_max, row, rmax)
            rr.vmem_bar("vst_vld")
            nm = rr.vmax(
                rr.vload(sm_max, 0), rr.vload(self.candidate_max, 0),
                mask=active_row_mask,
            )
            rr.vstore(self.candidate_max, 0, nm, active_row_mask)
            rr.vmem_bar("vst_vld")
            self._fold_rows(
                qk_scores_handoff, self.candidate_max, self.candidate_sum, rows, stride,
                ve_mask, vo_mask, b16, b16_full, full_lane_mask,
                m0, s_m1, s_m0, block_stride,
            )


    @jit
    def _softmax_first_pertile_rows(self, qk_scores_handoff, scale, softmax_state_slot: int,
                                    causal_base, head0, rows_per_query_token):
        self._softmax_register_rows(
            qk_scores_handoff, scale, softmax_state_slot,
            True, True, qk_scores_handoff.shape[1],
            causal_base, head0, rows_per_query_token,
        )

    @jit
    def _softmax_rest_pertile_rows(self, qk_scores_handoff, scale, softmax_state_slot: int,
                                   rescale_state_slot: int, causal_base, head0,
                                   rows_per_query_token):
        self._softmax_register_rows(
            qk_scores_handoff, scale, softmax_state_slot,
            False, True, qk_scores_handoff.shape[1],
            causal_base, head0, rows_per_query_token,
        )

    @jit
    def commit_online_softmax_state(self, softmax_state_slot: int, rescale_state_slot: int, rows):
        """Update online-softmax state after P has been published to Cube.

        This matches the reference ProcessVec1/Vec1PostProcess split: P and
        both cross-core flags become visible first, then the AIV updates the
        running max/sum and the rescale factor while Cube can start PV.
        """
        sm_max = self.softmax_max_slots[softmax_state_slot]
        sm_sum = self.softmax_sum_slots[softmax_state_slot]
        sm_exp = self.softmax_rescale_slots[rescale_state_slot]
        with vf(mode="simd"):
            active_row_mask, _ = rr.update_mask(rows, elem_bits=32)
            self._softmax_rest_tail(sm_max, sm_sum, sm_exp, active_row_mask)


    @jit
    def compute_first_softmax_tile(self, qk_scores_handoff, scale, softmax_state_slot: int, causal_base,
                                   head0, rows_per_query_token):
        """First n-tile of a new m-tile: P = softmax(S); init running max/sum.

        perf R1: JIT-level dispatch to the hoisted single-region variants
        (uniform causal tile or mask 0); non-uniform causal tiles (tier >
        N, the AIV half spans tokens) keep the per-row spelling.  All
        paths are bit-identical to the previous_output-R1 numerics.
        """
        if const_expr(self.template_mask):
            actual_n = self.mask_valid_n
            if self.mask_active:
                if actual_n >= self.tile_n:
                    self._sm_first_fast_rows(qk_scores_handoff, scale, softmax_state_slot, True)
                else:
                    self._sm_first_masked_rows(qk_scores_handoff, scale, softmax_state_slot, actual_n, True)
            else:
                if actual_n >= self.tile_n:
                    self._sm_first_fast_rows(qk_scores_handoff, scale, softmax_state_slot)
                else:
                    self._sm_first_masked_rows(qk_scores_handoff, scale, softmax_state_slot, actual_n)
        elif const_expr(self.mask_mode == 3):
            rows_c = self.tile_vec_m
            same_tok = (
                (head0 + rows_c - 1) // rows_per_query_token <= head0 // rows_per_query_token
            )
            if const_expr(self.uses_gs1_row_order):
                same_tok = False
            # A fully visible KV block has one common mask in every row,
            # including GS1 and TND tiles spanning several query tokens.
            if causal_base >= qk_scores_handoff.shape[1]:
                same_tok = True
            if same_tok:
                _rows, actual_n = qk_scores_handoff.shape
                candidate_valid_columns = _clamp_range(
                    causal_base + head0 // rows_per_query_token, actual_n
                )
                effective_valid_columns = (actual_n if candidate_valid_columns >= actual_n else candidate_valid_columns)
                if effective_valid_columns >= self.tile_n:
                    self._sm_first_fast_rows(qk_scores_handoff, scale, softmax_state_slot)
                else:
                    self._sm_first_masked_rows(
                        qk_scores_handoff, scale, softmax_state_slot, effective_valid_columns
                    )
            else:
                self._softmax_first_pertile_rows(
                    qk_scores_handoff, scale, softmax_state_slot, causal_base, head0,
                    rows_per_query_token,
                )
        else:
            _rows, actual_n = qk_scores_handoff.shape
            if actual_n >= self.tile_n:
                self._sm_first_fast_rows(qk_scores_handoff, scale, softmax_state_slot)
            else:
                self._sm_first_masked_rows(
                    qk_scores_handoff, scale, softmax_state_slot, actual_n
                )


    @jit
    def update_online_softmax(self, qk_scores_handoff, scale, softmax_state_slot: int, rescale_state_slot: int,
                              causal_base, head0, rows_per_query_token):
        """Non-first n-tile: rescale running max/sum, P = exp(S - new_max).

        perf R1: JIT-level dispatch mirroring softmax_first."""
        if const_expr(self.template_mask):
            actual_n = self.mask_valid_n
            if self.mask_active:
                if actual_n >= self.tile_n:
                    self._sm_rest_fast_rows(qk_scores_handoff, scale, softmax_state_slot, rescale_state_slot, True)
                else:
                    self._sm_rest_masked_rows(qk_scores_handoff, scale, softmax_state_slot,
                                              rescale_state_slot, actual_n, True)
            else:
                if actual_n >= self.tile_n:
                    self._sm_rest_fast_rows(qk_scores_handoff, scale, softmax_state_slot, rescale_state_slot)
                else:
                    self._sm_rest_masked_rows(qk_scores_handoff, scale, softmax_state_slot,
                                              rescale_state_slot, actual_n)
        elif const_expr(self.mask_mode == 3):
            rows_c = self.tile_vec_m
            same_tok = (
                (head0 + rows_c - 1) // rows_per_query_token <= head0 // rows_per_query_token
            )
            if const_expr(self.uses_gs1_row_order):
                same_tok = False
            # A fully visible KV block has one common mask in every row,
            # including GS1 and TND tiles spanning several query tokens.
            if causal_base >= qk_scores_handoff.shape[1]:
                same_tok = True
            if same_tok:
                _rows, actual_n = qk_scores_handoff.shape
                candidate_valid_columns = _clamp_range(
                    causal_base + head0 // rows_per_query_token, actual_n
                )
                effective_valid_columns = (actual_n if candidate_valid_columns >= actual_n else candidate_valid_columns)
                if effective_valid_columns >= self.tile_n:
                    self._sm_rest_fast_rows(
                        qk_scores_handoff, scale, softmax_state_slot, rescale_state_slot
                    )
                else:
                    self._sm_rest_masked_rows(
                        qk_scores_handoff, scale, softmax_state_slot, rescale_state_slot, effective_valid_columns
                    )
            else:
                self._softmax_rest_pertile_rows(
                    qk_scores_handoff, scale, softmax_state_slot, rescale_state_slot,
                    causal_base, head0, rows_per_query_token,
                )
        else:
            _rows, actual_n = qk_scores_handoff.shape
            if actual_n >= self.tile_n:
                self._sm_rest_fast_rows(qk_scores_handoff, scale, softmax_state_slot, rescale_state_slot)
            else:
                self._sm_rest_masked_rows(
                    qk_scores_handoff, scale, softmax_state_slot, rescale_state_slot, actual_n
                )
    @jit
    def clear_fully_masked_rows(self, softmax_max_buffer, valid_vector_rows):
        """Reference RowInvalid: clear -FLT_MAX rows in affected blocks."""
        with vf(mode="simd"):
            full_lane_mask, _ = rr.update_mask(VL_T, elem_bits=32)
            zero = rr.vdup_scalar(0.0, dtypes.float32, mask=full_lane_mask)
            for row in range(valid_vector_rows):
                mx = rr.vload_brc(softmax_max_buffer, row)
                invalid = rr.vcmp_eq_scalar(mx, FP32_MIN, mask=full_lane_mask)
                for col in tuple(range(0, self.tile_d, VL_T)):
                    element_offset = row * self.tile_d + col
                    value = rr.vload(self.output_accumulator, element_offset)
                    value = rr.vselect(zero, value, cond_mask=invalid)
                    rr.vstore(self.output_accumulator, element_offset, value, full_lane_mask)


    @jit
    def initialize_output_band(self, pv_output_handoff, rows, output_band_idx):
        with vf(mode='simd'):
            full_lane_mask, _ = rr.update_mask(VL_T, elem_bits=32)
            for row in range(rows):
                for col in tuple(range(0, PV_BAND_WIDTH, VL_T)):
                    value = rr.vload(pv_output_handoff, row * PV_BAND_WIDTH + col)
                    rr.vstore(
                        self.output_accumulator,
                        row * TILE_D + output_band_idx * PV_BAND_WIDTH + col,
                        value,
                        full_lane_mask,
                    )

    @jit
    def accumulate_output_band(self, pv_output_handoff, softmax_rescale_buffer, rows, output_band_idx):
        with vf(mode='simd'):
            full_lane_mask, _ = rr.update_mask(VL_T, elem_bits=32)
            for row in range(rows):
                rescale_factor = rr.vload_brc(softmax_rescale_buffer, row)
                for col in tuple(range(0, PV_BAND_WIDTH, VL_T)):
                    element_offset = (
                        row * TILE_D + output_band_idx * PV_BAND_WIDTH + col
                    )
                    previous = rr.vload(self.output_accumulator, element_offset)
                    current = rr.vload(
                        pv_output_handoff, row * PV_BAND_WIDTH + col
                    )
                    value = rr.vmadd(
                        previous, rescale_factor, current, mask=full_lane_mask
                    )
                    rr.vstore(
                        self.output_accumulator,
                        element_offset,
                        value,
                        full_lane_mask,
                    )

    @jit
    def finalize_output_band(
            self,
            pv_output_handoff,
            softmax_rescale_buffer,
            softmax_sum_buffer,
            rows,
            output_band_idx,
            is_first):
        """Consume the last PV output_band_idx and normalize it in the same VF pass."""
        if is_first:
            with vf(mode='simd'):
                full_lane_mask, _ = rr.update_mask(VL_T, elem_bits=32)
                one = rr.vdup_scalar(1.0, dtypes.float32, mask=full_lane_mask)
                zero = rr.vdup_scalar(0.0, dtypes.float32, mask=full_lane_mask)
                for row in range(rows):
                    denominator = rr.vload_brc(softmax_sum_buffer, row)
                    empty = rr.vcmp_eq_scalar(denominator, 0.0, mask=full_lane_mask)
                    safe = rr.vselect(one, denominator, cond_mask=empty)
                    for col in tuple(range(0, PV_BAND_WIDTH, VL_T)):
                        value = rr.vdiv(
                            rr.vload(pv_output_handoff, row * PV_BAND_WIDTH + col),
                            safe,
                            mask=full_lane_mask,
                        )
                        value = rr.vselect(zero, value, cond_mask=empty)
                        rr.vstore(
                            self.output_accumulator,
                            row * TILE_D + output_band_idx * PV_BAND_WIDTH + col,
                            value,
                            full_lane_mask,
                        )
        else:
            with vf(mode='simd'):
                full_lane_mask, _ = rr.update_mask(VL_T, elem_bits=32)
                one = rr.vdup_scalar(1.0, dtypes.float32, mask=full_lane_mask)
                zero = rr.vdup_scalar(0.0, dtypes.float32, mask=full_lane_mask)
                for row in range(rows):
                    rescale_factor = rr.vload_brc(softmax_rescale_buffer, row)
                    denominator = rr.vload_brc(softmax_sum_buffer, row)
                    empty = rr.vcmp_eq_scalar(denominator, 0.0, mask=full_lane_mask)
                    safe = rr.vselect(one, denominator, cond_mask=empty)
                    for col in tuple(range(0, PV_BAND_WIDTH, VL_T)):
                        element_offset = row * TILE_D + output_band_idx * PV_BAND_WIDTH + col
                        previous = rr.vload(self.output_accumulator, element_offset)
                        current = rr.vload(pv_output_handoff, row * PV_BAND_WIDTH + col)
                        value = rr.vmadd(previous, rescale_factor, current, mask=full_lane_mask)
                        value = rr.vdiv(value, safe, mask=full_lane_mask)
                        value = rr.vselect(zero, value, cond_mask=empty)
                        rr.vstore(self.output_accumulator, element_offset, value, full_lane_mask)

    @jit
    def update_output_band(
            self,
            pv_output_handoff,
            softmax_rescale_buffer,
            softmax_sum_buffer,
            rows,
            output_band_idx,
            is_first,
            is_last):
        if is_last:
            self.finalize_output_band(pv_output_handoff, softmax_rescale_buffer,
                                      softmax_sum_buffer, rows, output_band_idx, is_first)
        else:
            if is_first:
                self.initialize_output_band(pv_output_handoff, rows, output_band_idx)
            else:
                self.accumulate_output_band(pv_output_handoff, softmax_rescale_buffer, rows, output_band_idx)


class SplitKvReducer:
    def __init__(self, element_dtype, should_return_lse, use_padded_lse_storage=False):
        self.element_dtype = element_dtype
        self.tile_vec_m = FD_WINDOW_ROWS
        self.tile_m = TILE_CUBE_M
        self.tile_d = TILE_D
        self.should_return_lse = should_return_lse
        self.use_padded_lse_storage = True
        state_shape = (FD_PARTIAL_CAPACITY, FD_STATE_COLUMNS)
        self.partial_sum_buffers = [
            dsl.make_buffer(dsl.reinterpret(dsl.UB.view(262144), dtype=dtypes.float32, shape=state_shape,
                            offset=UB_FD_SUM_BASE + split_idx * UB_FD_STATE_SLOT_STRIDE, data_format='nd'))
            for split_idx in range(FD_STATS_PER_SLOT)
        ]
        self.partial_max_buffers = [
            dsl.make_buffer(dsl.reinterpret(dsl.UB.view(262144), dtype=dtypes.float32, shape=state_shape,
                            offset=UB_FD_MAX_BASE + split_idx * UB_FD_STATE_SLOT_STRIDE, data_format='nd'))
            for split_idx in range(FD_STATS_PER_SLOT)
        ]
        self.split_weight_buffer = dsl.make_buffer(dsl.reinterpret(dsl.UB.view(
            262144), dtype=dtypes.float32, shape=state_shape, offset=UB_FD_EXP_ADDR, data_format='nd'))
        self.partial_output_handoff = dsl.make_channel([dsl.reinterpret(
            dsl.UB.view(262144),
            dtype=dtypes.float32, shape=(FD_WINDOW_ROWS, TILE_D),
            offset=(UB_FD_INPUT_ADDR) + _slot *
            prod((FD_WINDOW_ROWS, TILE_D)) *
            (dtypes.float32.bits // 8),
            data_format='nd') for _slot in range(2)])
        self.output_accumulator = dsl.make_buffer(dsl.reinterpret(dsl.UB.view(262144), dtype=dtypes.float32, shape=(
            FD_WINDOW_ROWS, TILE_D), offset=UB_FD_OUTPUT_ACCUM_ADDR, data_format='nd'))
        self.output_ub_handoff = dsl.make_channel([dsl.reinterpret(dsl.UB.view(262144), dtype=element_dtype, shape=(
            FD_WINDOW_ROWS, TILE_D), offset=UB_FD_OUTPUT_ADDR, data_format='nd')]).produce()
        self.output_ub_alias = dsl.make_buffer(dsl.reinterpret(dsl.UB.view(262144), dtype=element_dtype, shape=(
            FD_WINDOW_ROWS, TILE_D), offset=UB_FD_OUTPUT_ADDR, data_format='nd'))
        self.softmax_max_slots = [
            dsl.make_buffer(dsl.reinterpret(dsl.UB.view(262144), dtype=dtypes.float32, shape=(
                FD_WINDOW_ROWS, 1), offset=UB_FD_MAX_STATE_ADDR, data_format='nd'))
        ]
        self.softmax_sum_slots = [
            dsl.make_buffer(dsl.reinterpret(dsl.UB.view(262144), dtype=dtypes.float32, shape=(
                FD_WINDOW_ROWS, 1), offset=UB_FD_SUM_STATE_ADDR, data_format='nd'))
        ]
        self.lse_output_ub = dsl.make_channel([dsl.reinterpret(dsl.UB.view(262144), dtype=dtypes.float32, shape=(
            FD_WINDOW_ROWS, 1), offset=UB_FD_LSE_ADDR, data_format='nd')]).produce()
        self.lse_output_alias = dsl.make_buffer(dsl.reinterpret(dsl.UB.view(
            262144), dtype=dtypes.float32, shape=(FD_WINDOW_ROWS, 1), offset=UB_FD_LSE_ADDR, data_format='nd'))
        self.padded_lse_buffer = dsl.make_buffer(dsl.reinterpret(dsl.UB.view(262144), dtype=dtypes.float32, shape=(
            FD_WINDOW_ROWS, FD_STATE_COLUMNS), offset=UB_FD_LSE_ADDR, data_format='nd'))

    @jit
    def load_stats(self, stats, first_workspace_slot, split_count, window_row_start, buffer_phase):
        for split_idx in range(split_count):
            max_rows = stats[first_workspace_slot + split_idx, 0, None, None].view(TILE_CUBE_M, 1)
            sum_rows = stats[first_workspace_slot + split_idx, 1, None, None].view(TILE_CUBE_M, 1)
            src = tile_slice(
                _offset_view(max_rows, (window_row_start, 0)), (FD_WINDOW_ROWS, 1), (0, 0)
            )
            srcsum = tile_slice(
                _offset_view(sum_rows, (window_row_start, 0)),
                (FD_WINDOW_ROWS, 1),
                (0, 0),
            )
            mem_copy(
                reinterpret(self.partial_max_buffers[buffer_phase],
                            shape=(FD_WINDOW_ROWS, 1), stride=(1, 1),
                            offset=split_idx * FD_WINDOW_ROWS * 4),
                src,
            )
            mem_copy(
                reinterpret(self.partial_sum_buffers[buffer_phase],
                            shape=(FD_WINDOW_ROWS, 1), stride=(1, 1),
                            offset=split_idx * FD_WINDOW_ROWS * 4),
                srcsum,
            )

    @jit
    def compute_split_weights(self, split_count, active_rows, window_row_offset, buffer_phase):
        partial_max_buffer = self.partial_max_buffers[buffer_phase]
        partial_sum_buffer = self.partial_sum_buffers[buffer_phase]
        with vf(mode="simd"):
            # Keep the proven scalar-addressed loads/stores: the stats
            # workspace packs one fp32 value per row, whereas the reference
            # VF first expands every row to a 32-byte block.  Treating our
            # compact rows as one aligned vector faults on some dynamic task
            # cuts.  The math still follows the reference: each split weight
            # is (exp(softmax_state_slot-M) * sum_i) / global_sum, so the later 512-D divide
            # is unnecessary.
            full_lane_mask, _ = rr.update_mask(VL_T, elem_bits=32)
            for row in range(active_rows):
                global_row_max = rr.vdup_scalar(FP32_MIN, dtypes.float32, mask=full_lane_mask)
                for split_idx in range(split_count):
                    state_offset = split_idx * FD_WINDOW_ROWS + window_row_offset + row
                    v = rr.vload_brc(partial_max_buffer, state_offset)
                    global_row_max = rr.vmax(global_row_max, v, mask=full_lane_mask)
                global_row_sum = rr.vdup_scalar(0.0, dtypes.float32, mask=full_lane_mask)
                for split_idx in range(split_count):
                    state_offset = split_idx * FD_WINDOW_ROWS + window_row_offset + row
                    v = rr.vload_brc(partial_max_buffer, state_offset)
                    partial_row_sum = rr.vload_brc(partial_sum_buffer, state_offset)
                    normalized_split_weight = rr.vexp_sub(v, global_row_max, mask=full_lane_mask)
                    weighted_row_sum = rr.vmul(normalized_split_weight, partial_row_sum, mask=full_lane_mask)
                    global_row_sum = rr.vadd(global_row_sum, weighted_row_sum, mask=full_lane_mask)
                    weight_offset = split_idx * FD_WINDOW_ROWS + row
                    rr.vstore_first(self.split_weight_buffer, weight_offset, weighted_row_sum)
                rr.vmem_bar("vst_vld")
                is_empty_row = rr.vcmp_eq_scalar(global_row_sum, 0.0, mask=full_lane_mask)
                one = rr.vdup_scalar(1.0, dtypes.float32, mask=full_lane_mask)
                zero = rr.vdup_scalar(0.0, dtypes.float32, mask=full_lane_mask)
                safe_global_row_sum = rr.vselect(one, global_row_sum, cond_mask=is_empty_row)
                for split_idx in range(split_count):
                    weight_offset = split_idx * FD_WINDOW_ROWS + row
                    weighted_row_sum = rr.vload_brc(self.split_weight_buffer, weight_offset)
                    normalized_split_weight = rr.vdiv(weighted_row_sum, safe_global_row_sum, mask=full_lane_mask)
                    normalized_split_weight = rr.vselect(zero, normalized_split_weight, cond_mask=is_empty_row)
                    rr.vstore_first(self.split_weight_buffer, weight_offset, normalized_split_weight)
                rr.vstore_first(self.softmax_max_slots[0], row, global_row_max)
                rr.vstore_first(self.softmax_sum_slots[0], row, global_row_sum)
                if const_expr(self.should_return_lse):
                    invalid = rr.vcmp_eq_scalar(global_row_max, FD_LSE_SENTINEL, mask=full_lane_mask)
                    lse_total = rr.vselect(zero, global_row_sum, cond_mask=invalid)
                    val = rr.vadd(global_row_max, rr.vln(lse_total, mask=full_lane_mask), mask=full_lane_mask)
                    val = rr.vselect(
                        rr.vneg(val, mask=full_lane_mask), val, cond_mask=invalid
                    )
                    if const_expr(self.use_padded_lse_storage):
                        rr.vstore_first(self.padded_lse_buffer, row * FD_STATE_COLUMNS, val)
                    else:
                        rr.vstore_first(self.lse_output_ub, row, val)
            rr.vmem_bar("vst_vld")

    @jit
    def load_partial_output(self, partial_output_workspace, workspace_slot,
                            window_row_start, window_row_offset, active_rows, buffer_phase):
        # Preserve the Channel's owning window and consumer offsets, but copy
        # only the rows assigned to this reducer. Every row is DMA-aligned.
        partial_output_tile = self.partial_output_handoff.produce()
        mem_copy(
            reinterpret(partial_output_tile, shape=(active_rows, TILE_D),
                        offset=window_row_offset * TILE_D * 4),
            tile_slice(
                _offset_view(partial_output_workspace[workspace_slot, None, None].view(
                    TILE_CUBE_M, TILE_D), (window_row_start + window_row_offset, 0)),
                make_tiler((active_rows, TILE_D), alignment=(1, 1)),
                (0, 0),
            ),
        )

    @jit
    def accumulate_partial_output(self, split_idx, active_rows, window_row_offset, buffer_phase):
        partial_output_tile = self.partial_output_handoff.consume()
        with vf(mode="simd"):
            full_lane_mask, _ = rr.update_mask(VL_T, elem_bits=32)
            if split_idx > 0:
                for row in range(active_rows):
                    split_weight = rr.vload_brc(self.split_weight_buffer, split_idx * FD_WINDOW_ROWS + row)
                    for col in tuple(range(0, TILE_D, VL_T)):
                        value = rr.vload(partial_output_tile, (row + window_row_offset) * TILE_D + col)
                        old = rr.vload(self.output_accumulator, row * TILE_D + col)
                        result = rr.vmadd(value, split_weight, old, mask=full_lane_mask)
                        rr.vstore(self.output_accumulator, row * TILE_D + col, result, full_lane_mask)
            else:
                for row in range(active_rows):
                    split_weight = rr.vload_brc(self.split_weight_buffer, split_idx * FD_WINDOW_ROWS + row)
                    for col in tuple(range(0, TILE_D, VL_T)):
                        value = rr.vload(partial_output_tile, (row + window_row_offset) * TILE_D + col)
                        result = rr.vmul(value, split_weight, mask=full_lane_mask)
                        rr.vstore(self.output_accumulator, row * TILE_D + col, result, full_lane_mask)
            rr.vmem_bar("vst_vld")

    def cast_combined_output_aligned(self):
        # compute_split_weights initializes every accumulator row, including tails.
        with vf(mode="simd"):
            full_lane_mask, _ = rr.update_mask(VL_T, elem_bits=32)
            for chunk in range(self.tile_vec_m * self.tile_d // VL_T):
                value = rr.vload(self.output_accumulator, chunk * VL_T)
                narrowed = rr.vcast(value, self.element_dtype, mask=full_lane_mask)
                rr.vstore_pack(
                    self.output_ub_handoff,
                    chunk * VL_T,
                    narrowed,
                    full_lane_mask,
                    pack_mode="b32_to_b16",
                )
            rr.vmem_bar("vst_vld")



class AttentionCubeStage:
    def __init__(self, max_query_rows, element_dtype, kv_page_size, kv_layout):
        self.max_query_rows = max_query_rows
        self.kv_page_size = kv_page_size
        self.kv_uses_pa_nz_storage = kv_layout == "PA_NZ"
        self.q_l1 = dsl.make_buffer(dsl.reinterpret(dsl.L1.view(524288), dtype=element_dtype, shape=(
            max_query_rows, HEAD_DIM_QK), offset=L1_Q_ADDR, data_format='nz'))
        if kv_layout == "PA_NZ":
            # GM preserves D1/token/D0. The DMA and Cube views share this
            # static NZ owning root and its synchronization identity.
            self.kv_l1_slots = [
                dsl.make_buffer(
                    dsl.reinterpret(
                        dsl.L1.view(524288),
                        dtype=element_dtype, shape=(TILE_N, HEAD_DIM_QK),
                        offset=L1_KV_BASE_ADDR + kv_pipeline_slot * L1_KV_SLOT_STRIDE, data_format='nz'))
                for kv_pipeline_slot in range(PIPELINE_DEPTH)]

        else:
            self.kv_l1_slots = [
                dsl.make_buffer(
                    dsl.reinterpret(
                        dsl.L1.view(524288),
                        dtype=element_dtype, shape=(TILE_N, HEAD_DIM_QK),
                        offset=L1_KV_BASE_ADDR + kv_pipeline_slot * L1_KV_SLOT_STRIDE, data_format='nz'))
                for kv_pipeline_slot in range(PIPELINE_DEPTH)]
        self.nd_to_nz_copy = make_copy_engine(format_transform="nd2nz")
        # PA_NZ pages and Q row fragments already carry the required
        # physical ordering. Do not let auto selection apply ND2NZ again.
        self.physical_copy = make_copy_engine(format_transform="identity")
        self.value_band_l1 = [
            reinterpret(storage, shape=(TILE_N, PV_BAND_WIDTH), data_format="nz",
                        offset=band * PV_BAND_WIDTH * TILE_N * 2)
            for storage in self.kv_l1_slots
            for band in range(2)
        ]
        # Hardware-sized tiling, independent of batch/head/query counts.
        self.q_chunk_96_l0a = dsl.make_buffer(dsl.reinterpret(dsl.L0A.view(65536), dtype=element_dtype, shape=(
            max_query_rows, QK_CHUNK_WIDTHS[0]), offset=L0A_Q96_ADDR, data_format='nz'))
        self.q_chunk_128_l0a = dsl.make_buffer(dsl.reinterpret(dsl.L0A.view(65536), dtype=element_dtype, shape=(
            max_query_rows, QK_UB_ROW_STRIDE), offset=L0A_Q128_ADDR, data_format='nz'))
        self.q_chunk_112_l0a = dsl.make_buffer(dsl.reinterpret(dsl.L0A.view(65536), dtype=element_dtype, shape=(
            max_query_rows, TILE_N), offset=L0A_Q112_ADDR, data_format='nz'))
        self.probability_first_l0a = self.q_chunk_112_l0a
        # Q128 is no longer live during PV. Load P's second chunk into that
        # resource directly: MMAD consumes an L0A root, not an qk_chunk_offset view.
        self.probability_second_l0a = self.q_chunk_128_l0a
        self.operand_b_l0_slots = [
            dsl.make_buffer(dsl.reinterpret(dsl.L0B.view(65536), dtype=element_dtype, shape=(
                QK_UB_ROW_STRIDE, QK_UB_ROW_STRIDE), offset=qk_chunk_idx * L0B_SLOT_STRIDE, data_format='nz'))
            for qk_chunk_idx in range(2)
        ]
        # Each physical L0C slot has one synchronization identity. Compact
        # views describe the transaction's M pitch for both MMAD and FIXPIPE.
        self.accumulator_l0c_slots = [
            dsl.make_buffer(dsl.reinterpret(dsl.L0C.view(262144), dtype=dtypes.float32, shape=(
                TILE_CUBE_M, PV_BAND_WIDTH), offset=kv_pipeline_slot * L0C_SLOT_STRIDE, data_format='nz'))
            for kv_pipeline_slot in range(2)
        ]
        self.fixpipe_copy = make_copy_engine(split_axis=0)
        self.transpose_copy = make_copy_engine(transpose=True)

    @jit
    def load_pa_nz_fragment(self, kv_pipeline_slot, source, _physical_pitch,
                            destination_column, column_count, _l2_cache_ctl):
        # The owning L1 storage remains typed NZ with a fixed TILE_N pitch.
        # Describe the full physical D1/token/D0 span before taking a DMA
        # window. A dynamic reinterpret of the NZ root would retain ND2NZ.
        physical = reinterpret(
            self.kv_l1_slots[kv_pipeline_slot],
            shape=(PA_NZ_D1, TILE_N * FRACTAL_SIZE), data_format="nd",
        )
        destination = _offset_view(physical, (0, destination_column * FRACTAL_SIZE))
        fragment = tile_slice(
            source,
            make_tiler((PA_NZ_D1, column_count * FRACTAL_SIZE), alignment=(1, 16)),
            (0, 0),
        )
        # Source actual bounds the DMA length; no runtime L1 pitch or
        # compile-time enumeration of fragment lengths is needed.
        mem_copy(destination, fragment, engine=self.physical_copy,
                 l2_cache_ctl=_l2_cache_ctl)

    @jit
    def compute_qk(
            self,
            kv_pipeline_slot,
            l0c_slot,
            l0b_bank_phase,
            initialize_accumulator,
            output_handoff,
            valid_kv_columns,
            valid_query_rows):
        hardware_query_rows = (16 if valid_query_rows == 1 else valid_query_rows)
        aligned_query_rows = (valid_query_rows + FRACTAL_SIZE - 1) // FRACTAL_SIZE * FRACTAL_SIZE
        accumulator_storage = reinterpret(
            self.accumulator_l0c_slots[l0c_slot],
            shape=(aligned_query_rows, QK_UB_ROW_STRIDE),
        )
        self._compute_qk_at_row_pitch(
            kv_pipeline_slot, l0c_slot, l0b_bank_phase,
            initialize_accumulator, output_handoff, valid_kv_columns,
            valid_query_rows, hardware_query_rows, accumulator_storage,
        )

    @jit
    def _compute_qk_at_row_pitch(
            self,
            kv_pipeline_slot,
            l0c_slot,
            l0b_bank_phase,
            initialize_accumulator,
            output_handoff,
            valid_kv_columns,
            valid_query_rows,
            hardware_query_rows,
            accumulator_storage):
        accumulator = reinterpret(accumulator_storage, shape=(hardware_query_rows, valid_kv_columns))
        kv_fractal_stride = TILE_N // FRACTAL_SIZE
        for qk_chunk_idx in tuple(range(5)):
            qk_chunk_width = QK_CHUNK_WIDTHS[qk_chunk_idx]
            qk_chunk_offset = QK_CHUNK_OFFSETS[qk_chunk_idx]
            l0b_slot = (l0b_bank_phase + qk_chunk_idx) % 2
            if const_expr(qk_chunk_idx == 0):
                a = reinterpret(self.q_chunk_96_l0a, shape=(hardware_query_rows, qk_chunk_width))
                if initialize_accumulator:

                    mem_copy(a, reinterpret(self.q_l1,
                             shape=(self.max_query_rows, qk_chunk_width),
                             data_format="nz", offset=qk_chunk_offset * self.max_query_rows * 2))

            else:
                if const_expr(qk_chunk_idx % 2 == 0):
                    a = reinterpret(self.q_chunk_128_l0a, shape=(hardware_query_rows, qk_chunk_width))
                else:
                    a = reinterpret(self.q_chunk_112_l0a, shape=(hardware_query_rows, qk_chunk_width))

                mem_copy(a, reinterpret(self.q_l1,
                         shape=(self.max_query_rows, qk_chunk_width),
                         data_format="nz", offset=qk_chunk_offset * self.max_query_rows * 2))

            b = reinterpret(self.operand_b_l0_slots[l0b_slot], shape=(valid_kv_columns, qk_chunk_width))
            mem_copy(b, reinterpret(self.kv_l1_slots[kv_pipeline_slot],
                     shape=(TILE_N, qk_chunk_width), data_format="nz",
                     offset=qk_chunk_offset * TILE_N * 2))

            # The copied capacity differs from the valid MMAD N. Redeclare the
            # compute view so a full-width source cannot change the packed L0B
            # tail pitch inferred by the compiler.
            matmul(accumulator,
                   reinterpret(a, shape=(hardware_query_rows, qk_chunk_width)),
                   reinterpret(b, shape=(valid_kv_columns, qk_chunk_width)),
                   init=qk_chunk_idx == 0)

        mem_copy(
            reinterpret(output_handoff, shape=(self.max_query_rows // 2, 128), stride=(128, 1)),
            accumulator_storage,
            engine=self.fixpipe_copy,
            actual=(valid_query_rows, valid_kv_columns),
        )

    @jit
    def compute_pv_band(
            self,
            kv_pipeline_slot,
            l0c_slot,
            l0b_bank_phase,
            probability_l1,
            output_handoff,
            valid_kv_columns,
            valid_query_rows,
            output_band_idx):
        hardware_query_rows = (16 if valid_query_rows == 1 else valid_query_rows)
        aligned_query_rows = (valid_query_rows + FRACTAL_SIZE - 1) // FRACTAL_SIZE * FRACTAL_SIZE
        accumulator_storage = reinterpret(
            self.accumulator_l0c_slots[l0c_slot],
            shape=(aligned_query_rows, PV_BAND_WIDTH),
        )
        self._compute_pv_band_at_row_pitch(
            kv_pipeline_slot, l0c_slot, l0b_bank_phase, probability_l1, output_handoff,
            valid_kv_columns, valid_query_rows, output_band_idx, hardware_query_rows,
            accumulator_storage,
        )

    @jit
    def _compute_pv_band_at_row_pitch(
            self,
            kv_pipeline_slot,
            l0c_slot,
            l0b_bank_phase,
            probability_l1,
            output_handoff,
            valid_kv_columns,
            valid_query_rows,
            output_band_idx,
            hardware_query_rows,
            accumulator_storage):
        accumulator = reinterpret(accumulator_storage, shape=(hardware_query_rows, 256))
        kv_fractal_stride = TILE_N // FRACTAL_SIZE
        for kv_half_idx in tuple(range(2)):
            if valid_kv_columns > kv_half_idx * 64:
                valid_half_columns = (valid_kv_columns - kv_half_idx * \
                                      64 if valid_kv_columns - kv_half_idx * 64 < 64 else 64)
                if const_expr(kv_half_idx == 0):
                    pa = reinterpret(self.probability_first_l0a, shape=(hardware_query_rows, valid_half_columns))
                else:
                    pa = reinterpret(self.probability_second_l0a, shape=(hardware_query_rows, valid_half_columns))
                l0b_slot = (l0b_bank_phase + kv_half_idx) % 2
                if output_band_idx == 0:
                    p_columns = 64 if const_expr(kv_half_idx == 0) else TILE_N - 64
                    mem_copy(pa, reinterpret(probability_l1,
                             shape=(self.max_query_rows, p_columns), data_format="nz",
                             offset=kv_half_idx * 64 * self.max_query_rows * 2))
                # Preserve the source NZ pitch; the transpose copy materializes
                # the PV operand in L0B.
                b = reinterpret(self.operand_b_l0_slots[l0b_slot], shape=(valid_half_columns, PV_BAND_WIDTH))
                value_band = self.value_band_l1[kv_pipeline_slot * 2 + output_band_idx]
                mem_copy(b, tile_slice(value_band, (64, PV_BAND_WIDTH), (kv_half_idx, 0)),
                         engine=self.transpose_copy)

                matmul(accumulator,
                       reinterpret(pa, shape=(hardware_query_rows, valid_half_columns)),
                       reinterpret(b, shape=(valid_half_columns, PV_BAND_WIDTH)),
                       init=kv_half_idx == 0)

        mem_copy(
            reinterpret(output_handoff, shape=(self.max_query_rows // 2, 256), stride=(256, 1)),
            accumulator_storage,
            engine=self.fixpipe_copy,
            actual=(valid_query_rows, 256),
        )

# Explicit profiling identity: the default would be the synthesized kernel
# function name "FlashMlaWithKvcache___call__".  The role tuples stay empty because
# two launch parameters (cu_seqlens_q, seqused_q) are optional tensors that drop
# out of the call plan when None, and launch_profile() raises KeyError for any
# named parameter missing from that plan.
#
# The class name below also sets the device symbol the profiler reports, since
# kernel_class.py synthesizes fn_name as f"{cls.__name__}_{method_name}":
#   FlashMlaWithKvcache + __call__  ->  kernel_FlashMlaWithKvcache___call___0


@kernel(profile=ProfileSpec(name="FlashMlaWithKvcache", op_type="FlashMlaWithKvcache"))
class FlashMlaWithKvcache:
    def __init__(self, input_dtype=dtypes.float16, should_return_lse=True, tile_cube_m=TILE_CUBE_M,
                 kernel_layout="TND", layout_q="TND", layout_out="TND",
                 layout_kv="PA_NZ",
                 block_dim=36, query_span=0, total_q=0,
                 mask_mode=0, enable_split_kv_workspace=False, kv_block_size=0,
                 kv_l2_cache_ctl=0, fast_causal=False, q_contiguous=True):
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_cube_m // 2
        self.input_dtype = input_dtype
        self.tile_n = TILE_N
        self.tile_d = TILE_D
        self.value_chunk_width = 128
        self.rope_dimension = 64
        self.softmax_state_depth = 3
        self.should_return_lse = should_return_lse
        self.kernel_layout = kernel_layout
        self.layout_q = layout_q
        self.layout_out = layout_out
        self.layout_kv = layout_kv
        self.q_contiguous = q_contiguous
        self.block_dim = block_dim
        self.query_span = query_span
        self.total_q = total_q
        self.mask_mode = mask_mode
        self.enable_split_kv_workspace = enable_split_kv_workspace
        self.kv_block_size = kv_block_size
        self.kv_l2_cache_ctl = kv_l2_cache_ctl
        self.block_idx = get_block_idx()
        self.vector_subblock_idx = get_subblock_id()
        self.qk_scores_handoff = dsl.make_channel([dsl.reinterpret(dsl.UB.view(262144), dtype=dtypes.float32, shape=(
            TILE_VEC_M, QK_UB_ROW_STRIDE), offset=UB_QK_ADDR, data_format='nd')], kind=ChannelKind.CrossCore).produce()
        self.pv_output_handoff = dsl.make_channel(
            [dsl.reinterpret(
                dsl.UB.view(262144),
                dtype=dtypes.float32, shape=(TILE_VEC_M, PV_BAND_WIDTH),
                offset=(UB_PV_ADDR) + _slot * prod((TILE_VEC_M, PV_BAND_WIDTH)) * (dtypes.float32.bits // 8),
                data_format='nd') for _slot in range(2)],
            kind=ChannelKind.CrossCore)
        # P reuses the retired rope region of each 136192-byte KVP slot;
        # Channel requires uniform contiguous slots, so this stays explicit.
        self.probability_l1_slots = [
            dsl.make_buffer(dsl.reinterpret(dsl.L1.view(524288), dtype=input_dtype, shape=(
                TILE_CUBE_M, TILE_N), offset=L1_KV_BASE_ADDR + i * L1_KV_SLOT_STRIDE + L1_P_OFFSET, data_format='nz'))
            for i in range(PIPELINE_DEPTH)
        ]
        self.cube_stage = AttentionCubeStage(
            TILE_CUBE_M, input_dtype, kv_block_size, layout_kv
        )
        self.vector_stage = AttentionVectorStage(
            self.tile_vec_m,
            QK_UB_ROW_STRIDE,
            TILE_D,
            self.vector_subblock_idx,
            input_dtype,
            PIPELINE_DEPTH,
            should_return_lse,
            mask_mode,
            enable_split_kv_workspace,
            layout_q == "BNSD",
        )
        self.vector_stage.uses_gs1_row_order = kernel_layout == "GS1"
        self.vector_stage.fast_causal = fast_causal
        self.split_kv_reducer = SplitKvReducer(input_dtype, should_return_lse, layout_q == "BNSD")
        # The UB scratch aliases the retired PV area and is used before
        # attention for output initialization, and after it for BNSD tails.
        self.output_init_scratch = dsl.make_buffer(dsl.reinterpret(dsl.UB.view(
            262144), dtype=input_dtype, shape=(TILE_VEC_M, HEAD_DIM_QK), offset=UB_PV_ADDR, data_format='nd'))
        self.lse_init_scratch = dsl.make_buffer(dsl.reinterpret(dsl.UB.view(
            262144), dtype=dtypes.float32, shape=(TILE_VEC_M, 1), offset=UB_LSE_ADDR, data_format='nd'))

    @jit
    def _query_span(self, batch_idx):
        if const_expr(self.layout_q == "TND"):
            return dtypes.int64(self._cu_seqlens_q[batch_idx + 1]) - dtypes.int64(self._cu_seqlens_q[batch_idx])
        return self.query_span

    @jit
    def _query_length(self, batch_idx):
        span = self._query_span(batch_idx)
        if const_expr(self._seqused_q is not None):
            used = dtypes.int64(self._seqused_q[batch_idx])
            return (used if used < span else span)
        return span

    @jit
    def _token_base(self, batch_idx):
        if const_expr(self.layout_q == "TND"):
            return dtypes.int64(self._cu_seqlens_q[batch_idx])
        return batch_idx * self._query_span(batch_idx)

    @jit
    def _fill_io_zero(self):
        with vf(mode="simd"):
            mask, _ = rr.update_mask(128, elem_bits=16)
            zero = rr.vdup_scalar(0.0, self.input_dtype, mask=mask)
            for offset in range(0, TILE_VEC_M * HEAD_DIM_QK, 128):
                rr.vstore(self.output_init_scratch, offset, zero, mask)
            rr.vmem_bar("vst_vld")

    @jit
    def _fill_lse_inf(self):
        with vf(mode="simd"):
            mask, _ = rr.update_mask(VL_T, elem_bits=32)
            zero = rr.vdup_scalar(0.0, dtypes.float32, mask=mask)
            inf = rr.vneg(rr.vln(zero, mask=mask), mask=mask)
            for row in range(TILE_VEC_M):
                rr.vstore_first(self.lse_init_scratch, row, inf)
            rr.vmem_bar("vst_vld")

    @jit
    def _clear_rows(self, tensor, width):
        worker = self.block_idx * 2 + self.vector_subblock_idx
        workers = self.block_dim * 2
        total = self.total_q * \
            self._num_query_heads if const_expr(
                self.layout_q == "TND") else tensor.shape[0] * self.query_span * self._num_query_heads
        row = worker * TILE_VEC_M
        while row < total:
            count = total - row
            if count > TILE_VEC_M:
                count = TILE_VEC_M
            done = 0
            axis_size = self.query_span if const_expr(self.layout_q == "BNSD") else self._num_query_heads
            while done < count:
                physical = row + done
                start = physical % axis_size
                chunk = min(count - done, axis_size - start)
                span = make_tiler((chunk, width), alignment=(1, 1))
                plane = _native_row_plane(tensor, self.layout_q, physical,
                                          self._num_query_heads, self.query_span, width)
                mem_copy(tile_slice(_offset_view(plane, (start, 0)), span, (0, 0)),
                         reinterpret(self.output_init_scratch, shape=span))
                done += chunk
            row += workers * TILE_VEC_M

    @jit
    def _clear_lse_rows(self, tensor):
        worker = self.block_idx * 2 + self.vector_subblock_idx
        workers = self.block_dim * 2
        total = self.total_q * \
            self._num_query_heads if const_expr(
                self.layout_q == "TND") else tensor.shape[0] * self.query_span * self._num_query_heads
        row = worker * TILE_VEC_M
        while row < total:
            count = total - row
            if count > TILE_VEC_M:
                count = TILE_VEC_M
            done = 0
            axis_size = self.query_span if const_expr(self.layout_q == "BNSD") else self._num_query_heads
            while done < count:
                physical = row + done
                start = physical % axis_size
                chunk = min(count - done, axis_size - start)
                span = make_tiler((chunk, 1), alignment=(1, 1))
                plane = _native_lse_plane(tensor, self.layout_q, physical, self._num_query_heads, self.query_span)
                mem_copy(tile_slice(_offset_view(plane, (start, 0)), span, (0, 0)),
                         reinterpret(self.lse_init_scratch, shape=span))
                done += chunk
            row += workers * TILE_VEC_M

    @jit
    def _physical_tile_row(self, batch_idx, token, head):
        """First physical Q/O/LSE row for a metadata tile."""
        if const_expr(self.layout_q == "BNSD"):
            return (batch_idx * self._num_query_heads + head) * self.query_span + token
        return token * self._num_query_heads + head

    @jit
    def _physical_row(self, batch_idx, token, head, row):
        """Map a logical metadata row to the original tensor storage."""
        if const_expr(self.layout_q == "BNSD"):
            used = self._query_length(batch_idx)
            logical = head * used + token + row
            logical32 = dtypes.int32(logical)
            used32 = dtypes.int32(used)
            return ((batch_idx * self._num_query_heads + logical32 // used32) * self.query_span
                    + logical32 % used32)
        return self._physical_tile_row(batch_idx, token, head) + row

    @jit
    def _initialize_direct_io(self):
        self._fill_io_zero()
        self._clear_rows(self._out_gm, HEAD_DIM_V)
        if const_expr(self.should_return_lse):
            self._fill_lse_inf()
            self._clear_lse_rows(self._lse_native_gm)

    @jit
    def _task_coords(self, batch_idx, query_tile_idx, kv_tile_start, kv_tile_end, workspace_slot, task_kind):
        """Expand one FIA (BN2, M, S2-range) record for the old pipeline."""
        query_length = self._query_length(batch_idx)
        local = query_tile_idx * self.tile_cube_m
        if const_expr(self.kernel_layout == "GS1"):
            query_token_idx = local % query_length
            query_head_start = local // query_length
        else:
            query_token_idx = self._token_base(batch_idx) + local // self._num_query_heads
            query_head_start = local % self._num_query_heads
        return (
            batch_idx, query_token_idx, query_head_start, 0, self.tile_cube_m, kv_tile_end - kv_tile_start,
            0, kv_tile_start, kv_tile_end, workspace_slot, task_kind,
        )


    @jit
    def _valid_kv_columns(self, batch_idx: int, kv_tile_idx: int):
        """How many KV columns of this KV tile belong to this batch.

        The packed KV tensor has no batch_idx boundary to clip against: the
        columns past a batch_idx's last key are the NEXT batch_idx's keys (real,
        finite numbers that would sail through exp() and corrupt every
        row), so the maths narrows by this count instead.
        """
        s2_b = dtypes.int64(self._cache_seqlens[batch_idx])
        return _clamp_range(s2_b - kv_tile_idx * self.tile_n, self.tile_n)


    @jit
    def _valid_query_rows(self, batch_idx, query_token_idx, query_head_start):
        query_length = self._query_length(batch_idx)
        local = (query_token_idx - self._token_base(batch_idx)) * self._num_query_heads + query_head_start
        if const_expr(self.kernel_layout == "GS1"):
            local = query_head_start * query_length + query_token_idx
        return _clamp_range(query_length * self._num_query_heads - local, self.tile_cube_m)

    @jit
    def _valid_vector_rows(self, batch_idx, query_token_idx, query_head_start):
        actual = self._valid_query_rows(batch_idx, query_token_idx, query_head_start)
        return (actual + 1 - self.vector_subblock_idx) // 2

    @jit
    def _vector_row_offset(self, batch_idx, query_token_idx, query_head_start):
        return self.vector_subblock_idx * ((self._valid_query_rows(batch_idx,
                                           query_token_idx, query_head_start) + 1) // 2)

    @jit
    def _stage_softmax_rows(self, tick: int, batch_idx, query_tile_idx, kv_tile_start, kv_tile_end,
                            workspace_slot, task_kind, kv_tile_idx: int, softmax_sequence: int):
        """V1: online softmax -> p_l1 (V2C split-M) + running max/sum/exp."""
        (
            batch_idx,
            query_token_idx,
            query_head_start,
            output_row_start,
            tile_row_capacity,
            kv_tile_count,
            kv_row_offset,
            kv_tile_start,
            kv_tile_end,
            workspace_slot,
            task_kind,
        ) = self._task_coords(
            batch_idx, query_tile_idx, kv_tile_start, kv_tile_end, workspace_slot, task_kind
        )
        valid_vector_rows = self._valid_vector_rows(batch_idx, query_token_idx, query_head_start)
        valid_n = self._valid_kv_columns(batch_idx, kv_tile_idx)
        qk_view = reinterpret(
            self.qk_scores_handoff,
            shape=(valid_vector_rows, valid_n),
            stride=(QK_UB_ROW_STRIDE, 1),
        )
        # The slice-relative first tile (kv_tile_start, 0 on FA tiles) initializes
        # the online-softmax state -- an FD slice starts cold at kv_tile_start.
        is_first = kv_tile_idx == kv_tile_start
        rescale_state_slot = (tick - 1) % PIPELINE_DEPTH
        softmax_state_slot = softmax_sequence % self.softmax_state_depth

        # RIGHT_DOWN causal is computed arithmetically on every layout (the
        # GS1 path): the 2048x2048 attn_mask is the host-generated causal
        # template of the reference interface, so per-row valid_columns from
        # _causal_row_valid reproduces the template bits exactly.  The GM
        # template staging (load_template/_template_select) is retired --
        # it also carries the only uint32 vector chain of the kernel,
        # which the current compiler build cannot lower inside this VF.

        if const_expr(self.mask_mode == 3):
            # RIGHT_DOWN causal ingredients for _causal_row_valid (B1):
            # query_token_idx is the global token under TND/BSND and the batch-local
            # s under BNSD (TILING section 3); seq_lens row 2 carries the
            # per-batch token base (0 under BNSD) so one rebase serves
            # every layout.  delta = kv_length - query_length uses the ATTENDED length
            # (seqused clip applied on the host, batch-A note #3: prefix
            # clip first, causal column clip second).  causal_base is the
            # row-0 valid count before the per-row token staircase and the
            # batch KV clip (both applied inside _causal_row_valid).
            t_first = query_token_idx - self._token_base(batch_idx)
            delta = (
                dtypes.int64(self._cache_seqlens[batch_idx]) - self._query_length(batch_idx)
            )
            causal_base = t_first + delta + 1 - kv_tile_idx * self.tile_n
            # The softmax row loop runs over THIS AIV's half of the tile
            # (split-M), so the token staircase must start at the
            # subblock's offset inside the m-tile -- subblock 1 owns the
            # TRAILING tile_row_capacity (flash_attn's local_start).  The cut follows
            # ceil(actual M / 2), including ragged and odd tail tiles.
            head0 = query_head_start + self._vector_row_offset(batch_idx, query_token_idx, query_head_start)
            rows_per_query_token = self._num_query_heads
            if const_expr(self.kernel_layout == "GS1"):
                causal_base = delta + 1 - kv_tile_idx * self.tile_n
                head0 = query_token_idx + self._vector_row_offset(batch_idx, query_token_idx, query_head_start)
                rows_per_query_token = self._query_length(batch_idx)
        else:
            # Dummies: const_expr-gated away inside the softmax methods.
            causal_base = 0
            head0 = 0
            rows_per_query_token = 1

        if is_first:
            self.vector_stage.compute_first_softmax_tile(
                qk_view, self._scale, softmax_state_slot, causal_base, head0,
                rows_per_query_token,
            )
        else:
            self.vector_stage.update_online_softmax(
                qk_view, self._scale, softmax_state_slot, rescale_state_slot,
                causal_base, head0, rows_per_query_token,
            )
        valid_query_rows = self._valid_query_rows(batch_idx, query_token_idx, query_head_start)
        self.vector_stage.store_probability_tile(
            self.probability_l1_slots[(tick - 1) % PIPELINE_DEPTH], valid_query_rows
        )


    @jit
    def _clear_invalid_output(
            self,
            softmax_state_slot,
            batch_idx,
            query_token_idx,
            query_head_start,
            valid_vector_rows):
        if const_expr(self.mask_mode == 3):
            query_length = self._query_length(batch_idx)
            kv_length = dtypes.int64(self._cache_seqlens[batch_idx])
            if query_length > kv_length:
                start = ((query_token_idx - self._token_base(batch_idx)) * self._num_query_heads
                         + query_head_start + self._vector_row_offset(batch_idx, query_token_idx, query_head_start))
                first = start // self._num_query_heads
                crosses = False
                if const_expr(self.kernel_layout == "GS1"):
                    start = query_head_start * query_length + query_token_idx + \
                        self._vector_row_offset(batch_idx, query_token_idx, query_head_start)
                    first = start % query_length
                    crosses = first + valid_vector_rows > query_length
                if first < query_length - kv_length or crosses:
                    self.vector_stage.clear_fully_masked_rows(
                        self.vector_stage.softmax_max_slots[softmax_state_slot], valid_vector_rows)

    @jit
    def _stage_update_band(
            self,
            tick,
            batch_idx,
            query_tile_idx,
            kv_tile_start,
            kv_tile_end,
            workspace_slot,
            task_kind,
            kv_tile_idx,
            softmax_sequence,
            output_band_idx):
        (
            batch_idx,
            query_token_idx,
            query_head_start,
            output_row_start,
            tile_row_capacity,
            kv_tile_count,
            kv_row_offset,
            kv_tile_start,
            kv_tile_end,
            workspace_slot,
            task_kind,
        ) = self._task_coords(
            batch_idx, query_tile_idx, kv_tile_start, kv_tile_end, workspace_slot, task_kind
        )
        valid_vector_rows = self._valid_vector_rows(batch_idx, query_token_idx, query_head_start)
        first = kv_tile_idx == kv_tile_start
        last = kv_tile_idx == kv_tile_end - 1
        softmax_state_slot = softmax_sequence % PIPELINE_DEPTH
        rescale_state_slot = (tick - PIPELINE_DEPTH) % PIPELINE_DEPTH
        if first and output_band_idx == 0 and valid_vector_rows < self.tile_vec_m:
            self.vector_stage.clear_inactive_output_rows(valid_vector_rows)
        self.vector_stage.update_output_band(
            reinterpret(
                self.pv_output_handoff.consume(),
                shape=(self.tile_vec_m, PV_BAND_WIDTH),
                stride=(PV_BAND_WIDTH, 1),
            ),
            self.vector_stage.softmax_rescale_slots[rescale_state_slot],
            self.vector_stage.softmax_sum_slots[softmax_state_slot],
            valid_vector_rows,
            output_band_idx,
            first,
            last,
        )
        if last and output_band_idx == 1:
            if const_expr(self.enable_split_kv_workspace) and task_kind == 1:
                self._store_split_kv_partial(workspace_slot, softmax_state_slot,
                                             batch_idx, query_token_idx, query_head_start)
            else:
                self._clear_invalid_output(softmax_state_slot, batch_idx, query_token_idx,
                                           query_head_start, valid_vector_rows)
                self._store_o(output_row_start, batch_idx, query_token_idx, query_head_start)
                if const_expr(self.should_return_lse):
                    self._store_lse(output_row_start, softmax_state_slot, batch_idx, query_token_idx, query_head_start)




    @jit
    def _stage_softmax(
            self,
            tick: int,
            batch_idx,
            query_tile_idx,
            kv_tile_start,
            kv_tile_end,
            workspace_slot,
            task_kind,
            kv_tile_idx: int,
            softmax_sequence: int):
        """Run the 12-row metadata softmax stage."""
        self._stage_softmax_rows(
            tick,
            batch_idx,
            query_tile_idx,
            kv_tile_start,
            kv_tile_end,
            workspace_slot,
            task_kind,
            kv_tile_idx,
            softmax_sequence)

    @jit
    def _store_o(
        self, _output_row_start: int, batch_idx: int,
        query_token_idx: int, query_head_start: int,
    ):
        with vf(mode='simd'):
            full, _ = rr.update_mask(VL_T, elem_bits=32)
            for chunk in range(self.tile_vec_m * (TILE_D // VL_T)):
                value = rr.vload(self.vector_stage.output_accumulator, chunk * VL_T)
                narrowed = rr.vcast(value, self.input_dtype, mask=full)
                rr.vstore_pack(
                    self.vector_stage.output_accumulator,
                    chunk * (VL_T // 2),
                    narrowed,
                    full,
                    pack_mode='b32_to_b16',
                )
            rr.vmem_bar('vst_vld')
        self._store_native_rows(
            self._out_gm32, self.vector_stage.output_accumulator,
            batch_idx, query_token_idx, query_head_start,
            self._vector_row_offset(batch_idx, query_token_idx, query_head_start),
            self._valid_vector_rows(batch_idx, query_token_idx, query_head_start),
            PV_BAND_WIDTH, False,
        )

    @jit
    def _store_native_rows(self, destination, source, batch, token, head,
                           logical_offset, rows, width, is_lse):
        """Write across native token/head boundaries without a global alias."""
        vec_sync_notify(PIPE.V, PIPE.MTE3, 3)
        vec_sync_wait(PIPE.V, PIPE.MTE3, 3)
        done = 0
        while done < rows:
            physical = self._physical_row(batch, token, head, logical_offset + done)
            axis_size = self.query_span if const_expr(self.layout_q == "BNSD") else self._num_query_heads
            start = physical % axis_size
            available = axis_size - start
            if const_expr(self.layout_q == "BNSD"):
                available = self._query_length(batch) - start
            count = min(rows - done, available)
            if const_expr(is_lse):
                plane = _native_lse_plane(destination, self.layout_q, physical, self._num_query_heads, self.query_span)
            else:
                plane = _native_row_plane(destination, self.layout_q, physical,
                                          self._num_query_heads, self.query_span, width)
            span = make_tiler((count, width), alignment=(1, 1))
            source_pitch = FD_STATE_COLUMNS if const_expr(is_lse) else width
            source_bytes = (
                source_pitch * 4
                if const_expr(is_lse or width == PV_BAND_WIDTH)
                else source_pitch * 2
            )
            mem_copy(tile_slice(_offset_view(plane, (start, 0)), span, (0, 0)),
                     reinterpret(source, shape=(count, width),
                                 stride=(source_pitch, 1), offset=done * source_bytes))
            done += count
        # The source keeps its owning Buffer identity. Automatic synchronization
        # orders MTE3 reads before the next vector overwrite without draining
        # the whole pipeline after every O/LSE store.


    @jit
    def _store_lse(
        self, _output_row_start: int, softmax_state_slot: int, batch_idx,
        query_token_idx, query_head_start,
    ):
        """Store LSE using the original token/head coordinates."""
        sm_max_buf = self.vector_stage.softmax_max_slots[softmax_state_slot]
        sm_sum_buf = self.vector_stage.softmax_sum_slots[softmax_state_slot]
        rows = self._valid_vector_rows(
            batch_idx, query_token_idx, query_head_start
        )
        if rows > 0:
            self.vector_stage.compute_lse(sm_max_buf, sm_sum_buf)
            source = self.vector_stage.padded_lse_buffer
            self._store_native_rows(
                self._lse_native_gm, source, batch_idx, query_token_idx,
                query_head_start,
                self._vector_row_offset(
                    batch_idx, query_token_idx, query_head_start
                ),
                rows, 1, True,
            )




    @jit
    def _store_split_kv_partial(
            self,
            workspace_slot,
            softmax_state_slot: int,
            batch_idx,
            query_token_idx,
            query_head_start):
        """FD partial finalize (task_kind 1, slice kv_tile_count): the slice's final-max
        frame unnormalized O (res_o) -> accum_ws[slot], running max/sum ->
        stats_ws[slot]. Buffer synchronization orders raw VF writes and
        protects the reused buffers until the copies complete."""
        vec = self.vector_stage
        softmax_max_buffer = vec.softmax_max_slots[softmax_state_slot]
        softmax_sum_buffer = vec.softmax_sum_slots[softmax_state_slot]
        sub = self.vector_subblock_idx
        tvm = self._valid_vector_rows(batch_idx, query_token_idx, query_head_start)
        offset = self._vector_row_offset(batch_idx, query_token_idx, query_head_start)
        tier = TILE_CUBE_M

        # accum: this AIV's (tvm, 512) tile_row_capacity of slot (slot, tier, 512).
        row0 = offset
        partial_rows = self._partial_output_workspace[workspace_slot, None, None].view(tier, self.tile_d)
        max_rows = self._softmax_stats_workspace[workspace_slot, 0, None, None].view(tier, 1)
        sum_rows = self._softmax_stats_workspace[workspace_slot, 1, None, None].view(tier, 1)
        mem_copy(
            tile_slice(
                _offset_view(partial_rows, (row0, 0)), make_tiler((tvm, self.tile_d), alignment=(1, 1)),
                (0, 0),
            ),
            reinterpret(vec.output_accumulator, shape=(tvm, self.tile_d)),
        )
        # Fix the max/sum slot first, then store this AIV's row segment.
        mrow = offset
        srow = offset
        mem_copy(
            tile_slice(_offset_view(max_rows, (mrow, 0)), make_tiler((tvm, 1), alignment=(1, 1)), (0, 0)),
            softmax_max_buffer,
        )
        mem_copy(
            tile_slice(_offset_view(sum_rows, (srow, 0)), make_tiler((tvm, 1), alignment=(1, 1)), (0, 0)),
            softmax_sum_buffer,
        )


    @jit
    def _reduce_split_kv_partials(
            self,
            batch_idx,
            query_tile_idx,
            first_workspace_slot,
            split_count,
            row_offset,
            tile_row_capacity):
        (
            batch_idx,
            query_token_idx,
            kv_tile_idx,
            output_row_start,
            _,
            _,
            kv_row_offset,
            kv_tile_start,
            kv_tile_end,
            first_workspace_slot,
            task_kind,
        ) = self._task_coords(
            batch_idx, query_tile_idx, 0, 0, first_workspace_slot, TASK_KIND_REDUCE
        )
        for window_chunk_idx in range((tile_row_capacity + 7) // 8):
            chunk_row_start = window_chunk_idx * 8
            active_rows = tile_row_capacity - chunk_row_start
            if active_rows > 8:
                active_rows = 8
            tile_row_offset = row_offset + chunk_row_start
            window_row_start = tile_row_offset
            if window_row_start > 88:
                window_row_start = 88
            window_row_offset = tile_row_offset - window_row_start
            buffer_phase = window_chunk_idx % 2
            self.split_kv_reducer.load_stats(self._softmax_stats_workspace,
                                             first_workspace_slot, split_count, window_row_start, buffer_phase)
            self.split_kv_reducer.load_partial_output(
                self._partial_output_workspace, first_workspace_slot, window_row_start,
                window_row_offset, active_rows, 0)
            if split_count > 1:
                self.split_kv_reducer.load_partial_output(
                    self._partial_output_workspace, first_workspace_slot + 1, window_row_start,
                    window_row_offset, active_rows, 1)
            self.split_kv_reducer.compute_split_weights(split_count, active_rows, window_row_offset, buffer_phase)
            for split_idx in range(split_count):
                if split_idx >= 2:
                    self.split_kv_reducer.load_partial_output(
                        self._partial_output_workspace,
                        first_workspace_slot +
                        split_idx,
                        window_row_start,
                        window_row_offset,
                        active_rows,
                        split_idx %
                        2)
                self.split_kv_reducer.accumulate_partial_output(
                    split_idx, active_rows, window_row_offset, split_idx % 2)
            self.split_kv_reducer.cast_combined_output_aligned()
            self._store_native_rows(
                self._out_gm, self.split_kv_reducer.output_ub_handoff,
                batch_idx, query_token_idx, kv_tile_idx, tile_row_offset, active_rows, HEAD_DIM_V, False,
            )
            if const_expr(self.should_return_lse):
                source = self.split_kv_reducer.padded_lse_buffer
                self._store_native_rows(
                    self._lse_native_gm, source, batch_idx, query_token_idx, kv_tile_idx,
                    tile_row_offset, active_rows, 1, True,
                )

    @jit
    def _load_query_tile(
        self, batch_idx, query_tile_idx, kv_tile_start, kv_tile_end,
        workspace_slot, task_kind,
    ):
        (
            batch_idx,
            query_token_idx,
            query_head_start,
            output_row_start,
            tile_row_capacity,
            kv_tile_count,
            kv_row_offset,
            kv_tile_start,
            kv_tile_end,
            workspace_slot,
            task_kind,
        ) = self._task_coords(
            batch_idx, query_tile_idx, kv_tile_start, kv_tile_end, workspace_slot, task_kind
        )
        rows = self._valid_query_rows(
            batch_idx, query_token_idx, query_head_start
        )
        filled = 0
        while filled < rows:
            physical = self._physical_row(
                batch_idx, query_token_idx, query_head_start, filled
            )
            axis_size = self.query_span if const_expr(self.layout_q == "BNSD") else self._num_query_heads
            start = physical % axis_size
            available = axis_size - start
            if const_expr(self.layout_q == "BNSD"):
                available = self._query_length(batch_idx) - start
            count = min(rows - filled, available)
            source = _native_row_plane(self._q_gm, self.layout_q, physical,
                                       self._num_query_heads, self.query_span, HEAD_DIM_QK)
            if start % 16 == 0 and filled % 16 == 0 and count >= 16:
                aligned_count = count // 16 * 16
                if filled % aligned_count != 0:
                    aligned_count = FRACTAL_SIZE
                for blocks in tuple(range(1, TILE_CUBE_M // FRACTAL_SIZE + 1)):
                    if aligned_count == blocks * FRACTAL_SIZE:
                        matrix = reinterpret(self.cube_stage.q_l1,
                                             shape=(TILE_CUBE_M, HEAD_DIM_QK), data_format="nz")
                        span = (blocks * FRACTAL_SIZE, HEAD_DIM_QK)
                        mem_copy(tile_slice(matrix, span, (filled // (blocks * FRACTAL_SIZE), 0)),
                                 tile_slice(source[start:, 0:], span, (0, 0)),
                                 engine=self.cube_stage.nd_to_nz_copy, l2_cache_ctl=1)
                filled += aligned_count
            else:
                # Scatter one contiguous Q row into the fixed-pitch NZ root.
                physical = reinterpret(
                    self.cube_stage.q_l1,
                    shape=(PA_NZ_D1, self.tile_cube_m * FRACTAL_SIZE),
                    data_format="nd",
                )
                destination = tile_slice(
                    _offset_view(physical, (0, filled * FRACTAL_SIZE)),
                    (PA_NZ_D1, FRACTAL_SIZE), (0, 0),
                )
                row = source[start, None].view(PA_NZ_D1, FRACTAL_SIZE)
                mem_copy(destination, row,
                         engine=self.cube_stage.physical_copy, l2_cache_ctl=1)
                filled += 1

    @jit
    def _load_kv_tile(
            self,
            batch_idx,
            query_tile_idx,
            kv_tile_start,
            kv_tile_end,
            workspace_slot,
            task_kind,
            kv_tile_idx,
            kv_issue_idx):
        (
            batch_idx,
            query_token_idx,
            query_head_start,
            output_row_start,
            tile_row_capacity,
            kv_tile_count,
            kv_row_offset,
            kv_tile_start,
            kv_tile_end,
            workspace_slot,
            task_kind,
        ) = self._task_coords(
            batch_idx, query_tile_idx, kv_tile_start, kv_tile_end, workspace_slot, task_kind
        )
        slot = kv_issue_idx % PIPELINE_DEPTH
        # Reuse is ordered after all prior L1 operand reads.
        valid_n = self._valid_kv_columns(batch_idx, kv_tile_idx)
        physical_pitch = ((valid_n + FRACTAL_SIZE - 1) // FRACTAL_SIZE) * FRACTAL_SIZE
        # Both paged-KV paths preserve the fixed TILE_N L1 row pitch on tails.
        if const_expr(self.kv_block_size != 0):
            if const_expr(self.layout_kv == "PA_BBND"):
                page_size = self.kv_block_size
                filled = 0
                while filled < physical_pitch:
                    virtual = kv_tile_idx * TILE_N + filled
                    col = virtual // page_size
                    within = virtual % page_size
                    count = (physical_pitch - filled if physical_pitch - \
                             filled < page_size - within else page_size - within)
                    page = dtypes.int64(self._block_table[batch_idx, col])
                    count = (count if count < FRACTAL_SIZE else FRACTAL_SIZE)
                    span = make_tiler(
                        (count, HEAD_DIM_QK),
                        alignment=(FRACTAL_SIZE, FRACTAL_SIZE),
                    )
                    page_view = _paged_kv_page(
                        self._kv_cache, self.layout_kv, page, page_size
                    )
                    src = tile_slice(
                        _offset_view(page_view, (within, 0)), span, (0, 0)
                    )
                    base = tile_slice(
                        self.cube_stage.kv_l1_slots[slot],
                        (FRACTAL_SIZE, HEAD_DIM_QK),
                        (filled // FRACTAL_SIZE, 0),
                    )
                    mem_copy(
                        base,
                        src,
                        engine=self.cube_stage.nd_to_nz_copy,
                        l2_cache_ctl=self.kv_l2_cache_ctl,
                    )
                    filled = filled + count
            else:
                page_size = self.kv_block_size
                filled = 0
                while filled < physical_pitch:
                    virtual = kv_tile_idx * TILE_N + filled
                    col = virtual // page_size
                    within = virtual % page_size
                    count = (physical_pitch - filled if physical_pitch - \
                             filled < page_size - within else page_size - within)
                    page = dtypes.int64(self._block_table[batch_idx, col])
                    page_view = _paged_kv_page(self._kv_cache, self.layout_kv, page, page_size)
                    page_matrix = page_view.view(PA_NZ_D1, page_size * FRACTAL_SIZE)
                    source = _offset_view(page_matrix, (0, within * FRACTAL_SIZE))
                    self.cube_stage.load_pa_nz_fragment(
                        slot,
                        source,
                        physical_pitch,
                        filled,
                        count,
                        self.kv_l2_cache_ctl,
                    )
                    filled += count
        else:
            src = tile_slice(
                _offset_view(self._kv_cache, (kv_row_offset + kv_tile_idx * TILE_N, 0)),
                (physical_pitch, HEAD_DIM_QK),
                (0, 0),
            )
            destination = tile_slice(
                self.cube_stage.kv_l1_slots[slot], (physical_pitch, HEAD_DIM_QK), (0, 0)
            )
            mem_copy(destination, src, engine=self.cube_stage.nd_to_nz_copy, l2_cache_ctl=self.kv_l2_cache_ctl)

    @jit
    def _kv_tile_limit(self, batch_idx, query_tile_idx):
        query_length = self._query_length(batch_idx)
        kv_length = dtypes.int64(self._cache_seqlens[batch_idx])
        kv_tile_count = (kv_length + self.tile_n - 1) // self.tile_n
        if const_expr(self.mask_mode == 3):
            local = query_tile_idx * self.tile_cube_m
            last = (local + self.tile_cube_m if local + self.tile_cube_m < query_length * \
                    self._num_query_heads else query_length * self._num_query_heads) - 1
            if const_expr(self.kernel_layout == "GS1"):
                first_head = local // query_length
                last_head = last // query_length
                last_token = (last % query_length if first_head == last_head else query_length)
            else:
                last_token = last // self._num_query_heads
            visible = last_token + kv_length - query_length + 1
            visible = _clamp_range(visible, kv_length)
            kv_tile_count = (visible + self.tile_n - 1) // self.tile_n
        return kv_tile_count

    @jit
    def _run_attention_range(
            self,
            batch_idx,
            query_tile_idx,
            kv_tile_idx,
            end_batch_idx,
            end_query_tile_idx,
            end_kv_tile_idx,
            first_workspace_slot):
        start_batch = dtypes.int64(batch_idx)
        start_m = dtypes.int64(query_tile_idx)
        start_n = dtypes.int64(kv_tile_idx)
        end_batch_idx = dtypes.int64(end_batch_idx)
        end_query_tile_idx = dtypes.int64(end_query_tile_idx)
        end_kv_tile_idx = dtypes.int64(end_kv_tile_idx)
        first_workspace_slot = dtypes.int64(first_workspace_slot)
        # Keep the delayed queue across query and batch boundaries. Drain it
        # only when this core finishes its assigned range.
        pending = DelayLineGroup(
            2, "batch", "query", "kv_start", "kv_end", "workspace",
            "kind", "kv", "softmax_sequence",
        )
        issue_sequence = 0
        query_sequence = 0
        l0c_sequence = 0
        l0b_sequence = 0
        for batch_idx in range(start_batch, end_batch_idx + 1):
            query_length = self._query_length(batch_idx)
            m_count = (query_length * self._num_query_heads + self.tile_cube_m - 1) // self.tile_cube_m
            m_begin = (start_m if batch_idx == start_batch else 0)
            m_end = (end_query_tile_idx + 1 if batch_idx == end_batch_idx else m_count)
            for current_m in range(m_begin, m_end):
                cap = self._kv_tile_limit(batch_idx, current_m)
                kv_tile_start = (start_n if batch_idx == start_batch and current_m == start_m else 0)
                kv_tile_end = (end_kv_tile_idx if batch_idx ==
                               end_batch_idx and current_m == end_query_tile_idx else cap)
                if kv_tile_end > kv_tile_start:
                    task_kind = TASK_KIND_MAIN
                    if kv_tile_start > 0 or kv_tile_end < cap:
                        task_kind = TASK_KIND_PARTIAL
                    workspace_slot = first_workspace_slot
                    tail_is_second = (
                        start_n > 0
                        and (batch_idx != start_batch or current_m != start_m)
                        and batch_idx == end_batch_idx and current_m == end_query_tile_idx
                        and kv_tile_end < cap
                    )
                    if tail_is_second:
                        workspace_slot += 1
                    query_sequence += 1
                    self._load_query_tile(
                        batch_idx, current_m, kv_tile_start, kv_tile_end,
                        workspace_slot, task_kind,
                    )
                    for current_kv in range(kv_tile_start, kv_tile_end):
                        pending.push(
                            batch=batch_idx, query=current_m,
                            kv_start=kv_tile_start, kv_end=kv_tile_end,
                            workspace=workspace_slot, kind=task_kind,
                            kv=current_kv, softmax_sequence=query_sequence,
                        )
                        if current_kv == kv_tile_start:
                            self._load_kv_tile(
                                batch_idx, current_m, kv_tile_start, kv_tile_end,
                                workspace_slot, task_kind, current_kv, issue_sequence,
                            )
                        if current_kv + 1 < kv_tile_end:
                            self._load_kv_tile(
                                batch_idx, current_m, kv_tile_start, kv_tile_end,
                                workspace_slot, task_kind, current_kv + 1, issue_sequence + 1,
                            )
                        self._run_qk_softmax_stage(
                            issue_sequence, l0c_sequence % 2, l0b_sequence % 2,
                            batch_idx, current_m, kv_tile_start, kv_tile_end,
                            workspace_slot, task_kind, current_kv, query_sequence,
                        )
                        l0c_sequence += 1
                        l0b_sequence += 5
                        if issue_sequence > 0:
                            self._run_pv_stage(
                                issue_sequence - 1, l0c_sequence % 2, l0b_sequence % 2,
                                pending.batch.tap(1), pending.query.tap(1),
                                pending.kv_start.tap(1), pending.kv_end.tap(1),
                                pending.workspace.tap(1), pending.kind.tap(1),
                                pending.kv.tap(1), pending.softmax_sequence.tap(1),
                            )
                            l0c_sequence += 2
                            l0b_sequence += 4
                        pending.advance()
                        issue_sequence += 1
        if issue_sequence > 0:
            # Reuse the steady-state PV/V2 path without resetting slots or
            # dropping the final query state.
            self._run_pv_stage(
                issue_sequence - 1, l0c_sequence % 2,
                (l0b_sequence + (1 if issue_sequence > 1 else 0)) % 2,
                pending.batch.tap(1), pending.query.tap(1),
                pending.kv_start.tap(1), pending.kv_end.tap(1),
                pending.workspace.tap(1), pending.kind.tap(1),
                pending.kv.tap(1), pending.softmax_sequence.tap(1),
            )

    @jit
    def _run_qk_softmax_stage(
        self, issue_sequence, l0c_slot, l0b_bank_phase, batch_idx, query_tile_idx,
        kv_tile_start, kv_tile_end, workspace_slot, task_kind, kv_tile_idx, query_sequence,
    ):
        # Softmax state follows query indices; KV/P and rescale storage rotate
        # by stream sequence. These index spaces must remain distinct.
        (
            batch_idx,
            query_token_idx,
            query_head_start,
            output_row_start,
            tile_row_capacity,
            kv_tile_count,
            kv_row_offset,
            kv_tile_start,
            kv_tile_end,
            workspace_slot,
            task_kind,
        ) = self._task_coords(
            batch_idx, query_tile_idx, kv_tile_start, kv_tile_end, workspace_slot, task_kind
        )
        self.cube_stage.compute_qk(
            issue_sequence % PIPELINE_DEPTH, l0c_slot, l0b_bank_phase,
            kv_tile_idx == kv_tile_start, self.qk_scores_handoff,
            self._valid_kv_columns(batch_idx, kv_tile_idx),
            self._valid_query_rows(batch_idx, query_token_idx, query_head_start),
        )
        self._stage_softmax(
            issue_sequence + 1, batch_idx, query_tile_idx, kv_tile_start, kv_tile_end,
            workspace_slot, task_kind, kv_tile_idx, query_sequence,
        )
        vec_sync_intra_arrive(PIPE.MTE3, KV_LOAD_EVENT_BASE + issue_sequence % PIPELINE_DEPTH)
        if kv_tile_idx != kv_tile_start:
            self.vector_stage.commit_online_softmax_state(
                query_sequence % self.softmax_state_depth,
                issue_sequence % PIPELINE_DEPTH,
                self._valid_vector_rows(batch_idx, query_token_idx, query_head_start),
            )

    @jit
    def _run_pv_stage(
            self,
            kv_issue_idx,
            l0c_slot,
            l0b_bank_phase,
            batch_idx,
            query_tile_idx,
            kv_tile_start,
            kv_tile_end,
            workspace_slot,
            task_kind,
            kv_tile_idx,
            softmax_sequence):
        cube_sync_intra_wait(PIPE.MTE1, KV_LOAD_EVENT_BASE + kv_issue_idx % PIPELINE_DEPTH)
        cube_sync_intra_wait(
            PIPE.MTE1,
            KV_LOAD_EVENT_BASE + kv_issue_idx % PIPELINE_DEPTH + AIV_SYNC_OFFSET,
        )
        (
            batch_idx,
            query_token_idx,
            query_head_start,
            output_row_start,
            tile_row_capacity,
            kv_tile_count,
            kv_row_offset,
            kv_tile_start,
            kv_tile_end,
            workspace_slot,
            task_kind,
        ) = self._task_coords(
            batch_idx, query_tile_idx, kv_tile_start, kv_tile_end, workspace_slot, task_kind
        )
        for output_band_idx in range(2):
            self.cube_stage.compute_pv_band(
                kv_issue_idx % PIPELINE_DEPTH, (l0c_slot + output_band_idx) % 2,
                (l0b_bank_phase + output_band_idx * ((self._valid_kv_columns(batch_idx, kv_tile_idx) + 63) // 64)) % 2,
                self.probability_l1_slots[kv_issue_idx % PIPELINE_DEPTH],
                self.pv_output_handoff.produce(),
                self._valid_kv_columns(batch_idx, kv_tile_idx),
                self._valid_query_rows(batch_idx, query_token_idx, query_head_start),
                output_band_idx)
            self._stage_update_band(
                kv_issue_idx + 3,
                batch_idx,
                query_tile_idx,
                kv_tile_start,
                kv_tile_end,
                workspace_slot,
                task_kind,
                kv_tile_idx,
                softmax_sequence,
                output_band_idx)

    def __call__(self, out_gm: Tensor, q_gm: Tensor, kv_gm: Tensor, lse_gm: Tensor,
                 scale: float, num_heads: int, metadata: Tensor,
                 block_table: Tensor, cache_seqlens: Tensor, cu_seqlens_q: Tensor,
                 seqused_q: Tensor, accum_ws_gm: Tensor, stats_ws_gm: Tensor):
        self._out_gm = out_gm
        self._out_gm32 = out_gm.view(dtype=dtypes.float32)
        self._q_gm = q_gm
        self._kv_cache = kv_gm
        self._lse_native_gm = lse_gm
        self._scale = scale
        self._num_query_heads = num_heads
        self._block_table = block_table
        self._cache_seqlens = cache_seqlens
        self._cu_seqlens_q = cu_seqlens_q
        self._seqused_q = seqused_q
        self._partial_output_workspace = accum_ws_gm
        self._softmax_stats_workspace = stats_ws_gm
        if metadata[HEAD_NEED_INIT] > 0:
            self._initialize_direct_io()
            global_sync_all(flag_ids=FD_GLOBAL_SYNC_FLAG_IDS)
        sections = metadata[HEAD_SECTION_NUM]
        aic = metadata[HEAD_AIC_NUM]
        aiv = metadata[HEAD_AIV_NUM]
        for section in range(sections):
            fa_base = METADATA_STRIDE + (
                section * aic + self.block_idx
            ) * METADATA_STRIDE
            self._run_attention_range(
                metadata[fa_base + FA_BN2_START],
                metadata[fa_base + FA_M_START],
                metadata[fa_base + FA_S2_START],
                metadata[fa_base + FA_BN2_END],
                metadata[fa_base + FA_M_END],
                metadata[fa_base + FA_S2_END],
                metadata[fa_base + FA_FIRST_FD_WORKSPACE],
            )
            if const_expr(self.enable_split_kv_workspace):
                if metadata[HEAD_IS_FD] > 0:
                    global_sync_all(flag_ids=FD_GLOBAL_SYNC_FLAG_IDS)
                    aiv_idx = self.block_idx * 2 + self.vector_subblock_idx
                    fd_base = (
                        METADATA_STRIDE + sections * aic * METADATA_STRIDE
                        + (section * aiv + aiv_idx) * METADATA_STRIDE
                    )
                    fd_rows = metadata[fd_base + FD_M_NUM]
                    for chunk in range(
                        (fd_rows + REDUCE_TILE_M - 1) // REDUCE_TILE_M
                    ):
                        offset = chunk * REDUCE_TILE_M
                        count = fd_rows - offset
                        if count > REDUCE_TILE_M:
                            count = REDUCE_TILE_M
                        self._reduce_split_kv_partials(
                            metadata[fd_base + FD_BN2_IDX],
                            metadata[fd_base + FD_M_IDX],
                            metadata[fd_base + FD_WORKSPACE_IDX],
                            metadata[fd_base + FD_WORKSPACE_NUM],
                            metadata[fd_base + FD_M_START] + offset,
                            count,
                        )
            if section + 1 < sections:
                global_sync_all(flag_ids=FD_GLOBAL_SYNC_FLAG_IDS)


class FlashMlaLauncher:
    """JIT-compiled kernel launch (positional binding, FA blueprint)."""

    def __init__(self, dtype, block_dim, should_return_lse, tile_cube_m=TILE_CUBE_M,
                 kernel_layout="TND", layout_q="TND", layout_out="TND",
                 layout_kv="PA_NZ", query_span=0, total_q=0,
                 mask_mode=0, enable_split_kv_workspace=False,
                 kv_block_size=0, kv_l2_cache_ctl=0, fast_causal=False,
                 q_contiguous=True):
        self.kv_block_size = kv_block_size
        self.kv_l2_cache_ctl = kv_l2_cache_ctl
        self.dtype = dtype
        self.block_dim = block_dim
        self.should_return_lse = should_return_lse
        self.tile_cube_m = tile_cube_m
        self.kernel_layout = kernel_layout
        self.layout_q = layout_q
        self.layout_out = layout_out
        self.layout_kv = layout_kv
        self.query_span = query_span
        self.total_q = total_q
        self.mask_mode = mask_mode
        self.enable_split_kv_workspace = enable_split_kv_workspace
        self.fast_causal = fast_causal
        self.q_contiguous = q_contiguous

    @host
    def launch(
        self,
        out_gm: Tensor,
        q_gm: Tensor,
        kv_gm: Tensor,
        lse_gm: Tensor,
        scale: float,
        num_heads: int,
        metadata: Tensor,
        block_table: Tensor,
        cache_seqlens: Tensor,
        cu_seqlens_q: Tensor | None,
        seqused_q: Tensor | None,
        accum_ws_gm: Tensor,
        stats_ws_gm: Tensor,
    ):
        op = FlashMlaWithKvcache(
            input_dtype=self.dtype,
            should_return_lse=self.should_return_lse,
            tile_cube_m=self.tile_cube_m,
            kernel_layout=self.kernel_layout,
            layout_q=self.layout_q,
            layout_out=self.layout_out,
            layout_kv=self.layout_kv,
            block_dim=self.block_dim,
            query_span=self.query_span,
            total_q=self.total_q,
            mask_mode=self.mask_mode,
            enable_split_kv_workspace=self.enable_split_kv_workspace,
            kv_block_size=self.kv_block_size,
            kv_l2_cache_ctl=self.kv_l2_cache_ctl,
            fast_causal=self.fast_causal,
            q_contiguous=self.q_contiguous,
        )
        op[self.block_dim](
            out_gm, q_gm, kv_gm, lse_gm, scale, num_heads, metadata, block_table,
            cache_seqlens, cu_seqlens_q, seqused_q, accum_ws_gm,
            stats_ws_gm
        )


# Host-side launch


_TORCH_TO_DSL = {
    torch.float16: dtypes.float16,
    torch.bfloat16: dtypes.bfloat16,
}


_COMPILED_KERNELS = OrderedDict()
_COMPILE_LOCK = RLock()
_COMPILE_CACHE_LIMIT = 32


def _launch_precompiled(launcher, *args):
    launch = launcher.launch
    dtype_map = {
        torch.float16: dtypes.float16,
        torch.bfloat16: dtypes.bfloat16,
        torch.float32: dtypes.float32,
        torch.int64: dtypes.int64,
        torch.int32: dtypes.int32,
        torch.int8: dtypes.int8,
    }
    contracts = []
    specs = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            if arg.dtype not in dtype_map:
                return launch(*args)
            shape = tuple(arg.shape)
            stride = tuple(arg.stride())
            contracts.append((shape, stride, arg.dtype, str(arg.device)))
            specs.append(TensorSpec(shape, dtype_map[arg.dtype], stride=stride))
        elif arg is None:
            contracts.append(None)
            specs.append(None)
        elif isinstance(arg, float):
            contracts.append(float)
            specs.append(dtypes.float32)
        elif isinstance(arg, int):
            contracts.append(int)
            specs.append(dtypes.int64)
        else:
            return launch(*args)
    key = (
        type(launcher).__module__,
        type(launcher).__qualname__,
        tuple(sorted((name, repr(value)) for name, value in vars(launcher).items())),
        tuple(contracts),
    )
    with _COMPILE_LOCK:
        compiled = _COMPILED_KERNELS.get(key)
        if compiled is None:
            compiled = dsl.compile(launch, *specs)
            _COMPILED_KERNELS[key] = compiled
            if len(_COMPILED_KERNELS) > _COMPILE_CACHE_LIMIT:
                _COMPILED_KERNELS.popitem(last=False)
        _COMPILED_KERNELS.move_to_end(key)
    return compiled(*(arg for arg in args if arg is not None))


def flash_mla_with_kvcache(
    q,
    k_cache,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=HEAD_DIM_V,
    softmax_scale=1.0,
    mask_mode=0,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="TND",
    layout_kv="PA_BBND",
    layout_out="NTD",
    return_softmax_lse=False,
):
    """Launch Flash MLA with direct Q/O/LSE storage and no fold GM buffers."""
    checked = check_flash_mla_inputs(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        attn_mask=attn_mask,
        metadata=metadata,
        head_dim_v=head_dim_v,
        mask_mode=mask_mode,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        return_softmax_lse=return_softmax_lse,
    )
    layout_q = checked.layout_q
    layout_kv = checked.layout_kv
    layout_out = checked.layout_out
    batch_size = checked.batch
    total_q = checked.total_q
    num_heads_q = checked.num_heads_q
    query_span = 0
    out_shape = (num_heads_q, total_q, HEAD_DIM_V)
    lse_shape = (num_heads_q, total_q)
    block_size = checked.block_size

    # Keep the kernel's contiguous physical row order in the public output
    # storage.  NTD is a metadata-only view of the token-major allocation.
    if layout_out == "NTD":
        out_storage = torch.empty(
            (total_q, num_heads_q, HEAD_DIM_V), dtype=q.dtype, device=q.device
        )
        attn_out = out_storage.permute(1, 0, 2)
    else:
        attn_out = torch.empty(out_shape, dtype=q.dtype, device=q.device)
        out_storage = attn_out
    out_direct = out_storage
    if return_softmax_lse:
        if layout_q == "TND":
            lse_storage = torch.empty(
                (total_q, num_heads_q), dtype=torch.float32, device=q.device
            )
            softmax_lse = lse_storage.transpose(0, 1)
        elif layout_q == "BSND":
            seqlen_q = q.shape[1]  # BSND 的 q 为 [B, S, N, D]
            lse_storage = torch.empty(
                (batch_size, seqlen_q, num_heads_q),
                dtype=torch.float32,
                device=q.device,
            )
            softmax_lse = lse_storage.permute(0, 2, 1)
        else:
            lse_storage = torch.empty(lse_shape, dtype=torch.float32, device=q.device)
            softmax_lse = lse_storage
        lse_direct = lse_storage
    else:
        # The public contract returns an empty tensor when LSE is disabled.
        # Keep a private row for the fused kernel's uniform output ABI.
        lse_direct = torch.empty((1, 1), dtype=torch.float32, device=q.device)
        softmax_lse = torch.empty(0, dtype=torch.float32, device=q.device)

    # Match the metadata scheduler's stream quota without reading device data
    # back to the host; this also keeps launch compatible with ACLGraph capture.
    block_dim, _ = get_effective_core_counts(
        stream=torch.npu.current_stream(q.device)
    )
    workspace_slots = 2 * block_dim
    accum_ws = torch.empty(
        (workspace_slots, TILE_CUBE_M, HEAD_DIM_V),
        dtype=torch.float32,
        device=q.device,
    )
    stats_ws = torch.empty(
        (workspace_slots, FD_STATS_PER_SLOT, TILE_CUBE_M, 1),
        dtype=torch.float32,
        device=q.device,
    )
    launcher = FlashMlaLauncher(
        dtype=_TORCH_TO_DSL[q.dtype],
        block_dim=block_dim,
        should_return_lse=return_softmax_lse,
        kernel_layout="GS1" if layout_q == "BNSD" else "TND",
        layout_q=layout_q,
        layout_out=layout_out,
        layout_kv=layout_kv,
        query_span=query_span,
        total_q=total_q,
        mask_mode=mask_mode,
        enable_split_kv_workspace=True,
        kv_block_size=block_size,
        kv_l2_cache_ctl=1,
        fast_causal=(layout_q == "BNSD"),
        q_contiguous=q.is_contiguous(),
    )
    scale = 1.0 if softmax_scale is None else float(softmax_scale)
    _launch_precompiled(
        launcher,
        out_direct,
        q,
        k_cache,
        lse_direct,
        scale,
        num_heads_q,
        metadata,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        seqused_q,
        accum_ws,
        stats_ws,
    )
    return attn_out, softmax_lse
