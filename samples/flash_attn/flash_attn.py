# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Channel-first software-pipeline Flash Attention kernel.

Supports full, right-down causal, and sliding-window attention.
3-stage pipeline (QK -> softmax -> PV -> update), raw-mode Vector.
"""

from functools import lru_cache

import torch

from cannbotdsl import get_platform_info

from cannbotdsl.buffer import Buffer
from cannbotdsl import ChannelKind, MemLoc, PIPE, Tensor, dtypes, permute
from cannbotdsl.ops.sync import channel_rewind
from cannbotdsl.channel import Channel
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
import cannbotdsl as cb
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_dim, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy
from cannbotdsl.ops.sync import (
    cube_sync_pipe,
    global_sync_all,
    vec_sync_all,
)
from cannbotdsl.tensor import (
    ceil_div,
    reinterpret,
    tile_slice,
    make_tiler,
)
from cannbotdsl.types.delay_line import DelayLineGroup

if __package__:
    from .flash_attn_metadata import get_launch_core_counts
    from .flash_attn_validation import (
        L0C_DEPTH,
        LSE_ROW_LANES,
        output_shapes,
        select_tile_config,
        validate_core_counts,
        validate_flash_attn_inputs,
        validate_output_init_core_counts,
        validate_workspace,
    )
else:
    from flash_attn_metadata import get_launch_core_counts
    from flash_attn_validation import (
        L0C_DEPTH,
        LSE_ROW_LANES,
        output_shapes,
        select_tile_config,
        validate_core_counts,
        validate_flash_attn_inputs,
        validate_output_init_core_counts,
        validate_workspace,
    )

# Vector register width: 2048 bits / 32 bits per fp32 element = 64 elements per vector.
VL_T = 2048 // 32

# Default ratio of vector cores to cube cores on Ascend platforms.
DEFAULT_AIV_PER_AIC = 2

# Maximum D dimension processed by one cube/vector chunk.
CUBE_D_TILE = 128

# Sentinel threshold for detecting fully masked softmax rows.
MASK_NEG_THRESHOLD = -1e29

# Pipeline depth: 3-stage (QK -> softmax -> PV -> update).
PIPELINE_DEPTH = 3

# Internal sentinel for fully masked partial rows. Combine explicitly gives
# these slots zero weight so an all-invalid row retains denominator zero and
# exports +inf through the final LSE writer.
LSE_MIN = -1e30

# Full mixed-core barrier flag for split-KV phase boundaries.
# Q GM-to-L1 ordering stays entirely on AIC and does not consume block flags.
_COMBINE_FLAG_BASE = 0

# ---- ASC split-KV metadata binary layout ----
# Native flash_attn_metadata is a flat int32 buffer with fixed physical slices:
# HEAD[16] + FA[section][36][16] + FD[section][72][16]. Fields 0:7 (FA) and 0:6
# (FD) match flash_attn_metadata.h.
HEAD_METADATA_STRIDE = 16
FA_METADATA_STRIDE = 16
FD_METADATA_STRIDE = 16

# Physical metadata strides are part of flash_attn_metadata.h's ABI. Runtime
# core limiting changes how many leading records are populated, not where a
# section starts in the flat buffer.
ASC_AIC_CORE_NUM = 36
ASC_AIV_CORE_NUM = 72

HEAD_SECTION_NUM_INDEX = 0
HEAD_IS_FD_INDEX = 1
HEAD_M_BASE_SIZE_INDEX = 2
HEAD_S2_BASE_SIZE_INDEX = 3

FA_BN2_START_INDEX = 0
FA_M_START_INDEX = 1
FA_S2_START_INDEX = 2
FA_BN2_END_INDEX = 3
FA_M_END_INDEX = 4
FA_S2_END_INDEX = 5
FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6

FD_BN2_IDX_INDEX = 0
FD_M_IDX_INDEX = 1
FD_WORKSPACE_IDX_INDEX = 2
FD_WORKSPACE_NUM_INDEX = 3
FD_M_START_INDEX = 4
FD_M_NUM_INDEX = 5


def get_effective_core_counts(stream=None):
    """Return the AIC/AIV counts available to the launch stream."""
    props = torch.npu.get_device_properties(torch.npu.current_device())
    cube = int(getattr(props, "cube_core_num", 0))
    # Some runtime versions omit vector_core_num; Ascend uses two AIVs per AIC.
    vector = int(getattr(props, "vector_core_num", 0)) or DEFAULT_AIV_PER_AIC * cube
    try:
        info = get_platform_info(stream=stream)
        cube = int(info.cube_core_num) or cube
        vector = int(info.vector_core_num) or vector
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    cube = min(cube, ASC_AIC_CORE_NUM)
    vector = min(vector, ASC_AIV_CORE_NUM)
    validate_core_counts(cube, vector)
    return cube, vector


def _present_nonempty(tensor) -> bool:
    """Match the runtime contract for an optional, non-empty tensor."""
    return tensor is not None and tensor.numel() != 0


def _need_init_output(
    *,
    has_seqused_q: bool,
    has_seqused_kv: bool,
    has_cu_seqlens_kv: bool,
    mask_mode: int,
    s1_size: int,
    s2_size: int,
    win_right: int,
) -> bool:
    """Return whether valid scheduling can leave physical output rows unwritten."""
    has_variable_lengths = (
        has_seqused_q or has_seqused_kv or has_cu_seqlens_kv
    )
    has_causal_gap = mask_mode == 3 and s1_size > s2_size
    has_window_gap = (
        mask_mode == 4
        and win_right != -1
        and s1_size - s2_size > win_right
    )

    return s1_size > 0 and (
        s2_size == 0
        or has_variable_lengths
        or has_causal_gap
        or has_window_gap
    )


@jit
def _partition_init_range(total_elements, aiv_index, aiv_count, block_elements):
    """Partition a flat GM range into disjoint block-aligned AIV slices."""
    aligned_total = ((total_elements + block_elements - 1) // block_elements) * block_elements
    chunk = ((aligned_total + aiv_count * block_elements - 1)
             // (aiv_count * block_elements)) * block_elements
    raw_start = aiv_index * chunk
    raw_end = (aiv_index + 1) * chunk
    return (
        (total_elements if raw_start > total_elements else raw_start),
        (total_elements if raw_end > total_elements else raw_end),
    )


@jit
def _init_lse_fill_count(max_tokens, aiv_count, capacity):
    """Bound the UB prefix read by every 16-element-aligned LSE copy."""
    blocks = (max_tokens + 15) // 16
    blocks_per_core = (blocks + aiv_count - 1) // aiv_count
    return min(capacity, min(max_tokens, blocks_per_core * 16))


def _stage_boundary_reset(flag_id: int):
    """Complete one split-KV phase before reusing workspace or channels."""
    global_sync_all(flag_ids=(flag_id, flag_id + 4, flag_id + 8))
    channel_rewind(reset_sync_id=False)


@jit
def _clamp_nonneg(value: int) -> int:
    """max(0, value) for dynamic Int64. Prevents negative mask counts from
    wrapping to large unsigned values in update_mask."""
    zero = value - value
    return (zero if value < 0 else value)


def _invalid_query_intervals(
    physical_q: int,
    used_q: int,
    used_kv: int,
    mask_mode: int,
    win_right: int,
):
    """Return invalid query prefix and padding suffix as half-open intervals."""
    prefix_end = used_q if used_kv == 0 else 0
    if mask_mode == 3:
        prefix_end = max(prefix_end, used_q - used_kv)
    elif mask_mode == 4 and win_right >= 0:
        prefix_end = max(prefix_end, used_q - used_kv - win_right)
    prefix_end = min(used_q, max(0, prefix_end))
    return 0, prefix_end, used_q, physical_q


@jit
def _sequence_length(cu, used, batch: int, default: int, packed):
    # Immutable launch metadata, read at the same batch index by AIC and AIV.
    if const_expr(used is not None):
        return used[batch]
    if const_expr(packed and cu is not None):
        return cu[batch + 1] - cu[batch]
    return default


@jit
def _metadata_task_m_bounds(bn2: int, bn0: int, m0: int,
                            bn1: int, m1: int, n1: int,
                            valid_m_count: int):
    """Intersect one metadata bn2 interval with its effective M extent."""
    m_begin = (m0 if bn2 == bn0 else 0)
    m_end_meta = (m1 + (1 if n1 > 0 else 0) if bn2 == bn1 else valid_m_count)
    return m_begin, min(valid_m_count, m_end_meta)


@jit
def _row_tile(gm, layout, batch, head, token, rows, capacity, width):
    if const_expr(layout == "TND"):
        span = gm[head, token:token + rows, None]
    else:
        span = gm[batch, head, token:token + rows, None]
    return tile_slice(span, (capacity, width), (0, 0))


@jit
def _merged_q_row(index: int, group: int, length: int, s1g):
    """Map one merged GxS1 row to its physical token and Q-head offset."""
    if const_expr(s1g):
        return index // group, index % group
    return index % length, index // length


@jit
def _merged_tile_coords(tile: int, max_m_rows: int, head_num_kv: int,
                        m_base: int):
    """Decode the one task identity used by native C1/V1/C2/V2."""
    m = tile % max_m_rows
    bn2 = tile // max_m_rows
    return bn2 // head_num_kv, bn2 % head_num_kv, m, m * m_base


# Extract a contiguous segment from merged S1xG rows without crossing boundaries:
# S1G fixes the token across Q heads; GS1 fixes the Q head across tokens.
@jit
def _merged_segment(gm, layout, batch: int, kv_head: int, start: int,
                     remaining: int, group: int, length: int, cu, s1g):
    token, g = _merged_q_row(start, group, length, s1g)
    head = kv_head * group + g
    if const_expr(layout == "TND"):
        token = token + cu[batch]
    if const_expr(s1g):
        count = min(remaining, group - g)
        if const_expr(layout == "TND"):
            token_major = permute(gm, (1, 0, 2))
            part = token_major[token, head:head + count, None]
        else:
            token_major = permute(gm, (0, 2, 1, 3))
            part = token_major[batch, token, head:head + count, None]
    else:
        count = min(remaining, length - start % length)
        if const_expr(layout == "TND"):
            part = gm[head, token:token + count, None]
        else:
            part = gm[batch, head, token:token + count, None]
    return tile_slice(part, (128, gm.shape[-1]), (0, 0)), count


def _nd_rows(tensor, offset, rows, width, row_stride=None):
    """Return a dynamic row window while preserving the backing slot pitch."""
    if row_stride is None:
        row_stride = width
    origin = tile_slice(tensor, (1, width), (offset, 0))
    return reinterpret(
        origin,
        shape=make_tiler((rows, width), alignment=(1, 1)),
        stride=(row_stride, 1),
    )


@jit
def _cast_nd_rows(dst, src, rows: int, width):
    """Numerically cast active ND rows without touching slot padding."""
    with vf(mode="simd"):
        full, _ = rr.update_mask(VL_T, elem_bits=32)
        for row in range(rows):
            for col_offset in tuple(range(0, width, VL_T)):
                src_offset = row * src.physical_stride[0] + col_offset
                dst_offset = row * dst.physical_stride[0] + col_offset
                if const_expr(src.dtype == dtypes.float32):
                    value = rr.vload(src, src_offset)
                    narrowed = rr.vcast(
                        value, dst.dtype, mask=full, rounding="rn"
                    )
                    rr.vstore_pack(
                        dst,
                        dst_offset,
                        narrowed,
                        full,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )
                else:
                    unpacked = rr.vload_unpack(
                        src,
                        src_offset,
                        unpack_mode=rr.UnpackMode.B16_TO_B32,
                    )
                    widened = rr.vcast(unpacked, dtypes.float32, mask=full)
                    rr.vstore(dst, dst_offset, widened, full)


@jit
def _store_o(gm, source, layout, batch: int, kv_head: int,
                        start: int, rows: int, group: int, length: int, cu, s1g):
    # G1 has identical S1G/GS1 row order; copy all token rows together.
    if const_expr(group == 1):
        s1g = False
    span = group if s1g else length
    first_offset = start % span
    segments = (ceil_div(first_offset + rows, span) if rows > 0 else 0)
    for segment in range(segments):
        offset = (0 if segment == 0 else segment * span - first_offset)
        dst, count = _merged_segment(gm, layout, batch, kv_head, start + offset,
                                     rows - offset, group, length, cu, s1g)
        mem_copy(dst, _nd_rows(source, offset, count, gm.shape[-1]))

# ---- cube side: matmul + L0C->UB store, all Channels (channel-first) ----


class Matmul:
    def __init__(self, tile_cube_m, tile_n, tile_d, dtype_16, paged_nz=False,
                 paged_strided=False):
        self.tile_cube_m = tile_cube_m
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.cube_d = min(tile_d, CUBE_D_TILE)
        self.d_loops = tile_d // self.cube_d

        self.nd2nz = make_copy_engine(format_transform="nd2nz")
        self.fixpipe = make_copy_engine(split_axis=0)

        tmp_n = max(tile_n, self.cube_d)
        if tile_d > CUBE_D_TILE:
            # D256 uses padded NZ Q slots directly. Allocating a Channel here
            # and replacing it in __call__ retains both allocations in L1.
            self.q_l1 = (
                Buffer(MemLoc.L1, (tile_cube_m, tile_d), dtype_16, n1_pad=256),
                Buffer(MemLoc.L1, (tile_cube_m, tile_d), dtype_16, n1_pad=256),
            )
        else:
            self.q_l1 = Channel(
                MemLoc.L1, shape=(tile_cube_m, tile_d), dtype=dtype_16, depth=2
            )
        kv_kind = ChannelKind.CrossCore if paged_strided else ChannelKind.SameCore
        kv_format = "nz" if paged_nz or paged_strided else None
        self.k_l1 = Channel(MemLoc.L1, shape=(tile_n, tile_d), dtype=dtype_16,
                            depth=2, kind=kv_kind, data_format=kv_format)
        self.v_l1 = Channel(MemLoc.L1, shape=(tile_n, tile_d), dtype=dtype_16,
                            depth=2, kind=kv_kind, data_format=kv_format)
        self.l0a = Channel(
            MemLoc.L0A, shape=(tile_cube_m, tmp_n), dtype=dtype_16, depth=2
        )
        # depth=2 enables ping-pong buffering between L1-to-L0B loads and
        # MMAD consumption.
        if self.cube_d == tile_n:
            # Preserve the original square-D path exactly: the same L0B
            # channel can represent both K(N,D) and V^T(D,N).
            self.qk_l0b = Channel(
                MemLoc.L0B,
                shape=(self.cube_d, tile_n),
                dtype=dtype_16,
                depth=2,
            )
            self.pv_l0b = self.qk_l0b
        else:
            # QK and PV require opposite logical L0B orientations.  Treating
            # them as one shape is only valid when N == D.
            self.qk_l0b = Channel(
                MemLoc.L0B,
                shape=(self.cube_d, tile_n),
                dtype=dtype_16,
                depth=2,
            )
            self.pv_l0b = Channel(
                MemLoc.L0B,
                shape=(tile_n, self.cube_d),
                dtype=dtype_16,
                depth=2,
            )
        self.l0c = Channel(
            MemLoc.L0C, shape=(tile_cube_m, tmp_n), dtype=dtypes.float32, depth=L0C_DEPTH
        )

    def load_k(self, gm_tensor):
        """GM -> L1 via nd2nz."""
        mem_copy(self.k_l1.produce(), gm_tensor, engine=self.nd2nz)

    def load_v(self, gm_tensor):
        """GM -> L1 via nd2nz."""
        slot = self.v_l1.produce()
        mem_copy(slot, gm_tensor, engine=self.nd2nz)
        return slot

    def _active_l0a(self, active_m, active_k):
        # Keep L1's physical pitch. Only the destination is packed to M.
        return reinterpret(self.l0a.produce(), shape=(active_m, active_k))

    def _active_l0c(self, active_m, active_n):
        return reinterpret(self.l0c.produce(), shape=(active_m, active_n))

    def _compute_l0a_view(self, active_view):
        # A tile view clips computation to the packed view's actual extent.
        # Keep the source L1 pitch and the allocation capacity unchanged.
        return tile_slice(
            active_view,
            (self.tile_cube_m, max(self.tile_n, self.cube_d)),
            (0, 0),
        )

    def compute_qk(self, active_m, q_source, valid_n):
        """QK: retain L1 pitch and bound cube work to valid KV rows."""
        k_l1 = self.k_l1.consume()
        if const_expr(self.tile_d <= self.cube_d):
            q_l0a = self._active_l0a(active_m, self.cube_d)
        s_l0c = reinterpret(
            self._active_l0c(active_m, self.tile_n),
            shape=make_tiler((active_m, valid_n), alignment=(1, 1)),
        )
        if const_expr(self.tile_d > self.cube_d):
            for d_idx in tuple(range(self.d_loops)):
                # Advance the existing two-slot L0A channel for each D chunk.
                # Loading Q[d+1] can then overlap MMA on Q[d].
                q_l0a = self._active_l0a(active_m, self.cube_d)
                q_part = tile_slice(
                    q_source, (self.tile_cube_m, self.cube_d), (0, d_idx),
                )
                # Select the D128 chunk before narrowing N so the original
                # NZ row-group pitch in L1 remains attached to the source.
                k_part = tile_slice(
                    k_l1, (self.tile_n, self.cube_d), (0, d_idx),
                )
                k_valid = reinterpret(
                    k_part,
                    shape=make_tiler((valid_n, self.cube_d), alignment=(1, 1)),
                )
                mem_copy(q_l0a, q_part)
                k_l0b = reinterpret(
                    self.qk_l0b.produce(),
                    shape=make_tiler((valid_n, self.cube_d), alignment=(1, 1)),
                )
                mem_copy(k_l0b, k_valid)
                matmul(s_l0c, self._compute_l0a_view(q_l0a), k_l0b, init=(d_idx == 0))
        else:
            mem_copy(q_l0a, q_source)
            # Retain the L1 source pitch while packing only valid KV rows.
            k_valid = reinterpret(
                k_l1,
                shape=make_tiler((valid_n, self.cube_d), alignment=(1, 1)),
            )
            k_l0b = reinterpret(
                self.qk_l0b.produce(),
                shape=make_tiler((valid_n, self.cube_d), alignment=(1, 1)),
            )
            mem_copy(k_l0b, k_valid)
            matmul(s_l0c, self._compute_l0a_view(q_l0a), k_l0b, init=True)

        return s_l0c

    def compute_pv(self, p_l1_ch, active_m, v_slot):
        """PV uses this delayed task's M and the original P storage pitch."""
        p_l0a = self._active_l0a(active_m, p_l1_ch.shape[1])
        o_l0c = self._active_l0c(active_m, self.tile_d)
        v_l0b = self.pv_l0b.produce()
        mem_copy(v_l0b, v_slot, transpose=True)
        mem_copy(p_l0a, p_l1_ch)
        matmul(o_l0c, self._compute_l0a_view(p_l0a), v_l0b, init=True)

        return o_l0c

    # Split V along D when head dimension exceeds one cube tile; each chunk
    # computes P x V and writes to its corresponding UB channel.
    def compute_pv_chunked(self, p_l1_ch, ub_channels, active_m, v_slot,
                           valid_n=None):
        """Keep D segmentation while computing each chunk with active M."""
        p_l0a = self._active_l0a(active_m, p_l1_ch.shape[1])
        # Both output-D chunks multiply the same P. Keep it resident until
        # both MMAs have read it rather than overwriting the L0A slot.
        mem_copy(p_l0a, p_l1_ch)
        for d_idx in tuple(range(self.d_loops)):
            o_l0c = self._active_l0c(active_m, self.cube_d)
            v_l0b = self.pv_l0b.produce()
            v_part = tile_slice(
                v_slot, (self.tile_n, self.cube_d), (0, d_idx),
            )
            if valid_n is not None:
                v_valid = reinterpret(
                    v_part,
                    shape=make_tiler((valid_n, self.cube_d), alignment=(1, 1)),
                )
                v_l0b = reinterpret(
                    v_l0b,
                    shape=make_tiler((valid_n, self.cube_d), alignment=(1, 1)),
                )
                mem_copy(v_l0b, v_valid, transpose=True)
            else:
                mem_copy(v_l0b, v_part, transpose=True)
            matmul(o_l0c, self._compute_l0a_view(p_l0a), v_l0b, init=True)
            mem_copy(ub_channels[d_idx].produce(), o_l0c,
                     engine=self.fixpipe,
                     actual=(active_m, self.cube_d))

    def store_s(self, ub_ch, active_m, active_n, result):
        """FIXPIPE consumes the same packed result layout as QK."""
        mem_copy(ub_ch.produce(), result,
                 engine=self.fixpipe,
                 actual=(active_m, active_n))

    def store_o(self, ub_ch, active_m, result):
        """FIXPIPE consumes the same packed result layout as PV."""
        mem_copy(ub_ch.produce(), result,
                 engine=self.fixpipe,
                 actual=(active_m, self.tile_d))


def _use_merged_window_mask(merged, s1g_mapping, mask_mode):
    # Both GS1 and S1G can use one final mask. Row mapping only selects
    # the producer: generated GS1 rows versus the S1G template loader.
    return merged and mask_mode == 4


# ---- vec side: raw-mode softmax + O accumulation, all channel-first ----
class Vector:
    def __init__(
        self,
        tile_vec_m,
        tile_n,
        tile_d,
        subblock_idx,
        mask_mode=0,
        dtype_16=dtypes.float16,
        preload_num=PIPELINE_DEPTH,
        mask_in_vector=False,
        merged_window_mask=False,
        template_window_mask=False,
    ):
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.pv_tile_d = min(tile_d, CUBE_D_TILE)
        self.subblock_idx = subblock_idx
        self.mask_mode = mask_mode
        self.dtype_16 = dtype_16
        self.merged_window_mask = merged_window_mask

        self.softmax_max_bufs = [
            Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
            for _ in range(preload_num)
        ]
        self.softmax_sum_bufs = [
            Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
            for _ in range(preload_num)
        ]
        self.softmax_exp_bufs = [
            Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
            for _ in range(preload_num)
        ]

        self.tmp_new_max = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.tmp_sum = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.res_o = Buffer(MemLoc.UB, (tile_vec_m, tile_d), dtypes.float32)

        p_n1_pad = 32 // 2  # NZ n1 alignment: 16 elements (half of 32-element n0)
        self.p_ub = Channel(
            MemLoc.UB,
            shape=(tile_vec_m, tile_n),
            dtype=self.dtype_16,
            depth=2,
            data_format="nz",
            n1_pad=p_n1_pad,
        )
        if const_expr(tile_d > CUBE_D_TILE):
            # Final cast runs after the last update. Its packed fp16 view
            # reuses the accumulator storage without a second UB allocation.
            self.o_ub = reinterpret(
                self.res_o, dtype=self.dtype_16,
                shape=(tile_vec_m, tile_d), stride=(tile_d, 1),
            )
        else:
            self.o_ub = Channel(
                MemLoc.UB, shape=(tile_vec_m, tile_d), dtype=self.dtype_16, depth=1
            ).produce()
        if const_expr(mask_mode != 0):
            if const_expr(mask_in_vector):
                # Generated and consumed by V: a local buffer needs no
                # cross-PIPE channel credit/ready events.
                self.mask_ch = Buffer(MemLoc.UB, (tile_vec_m, tile_n), dtypes.int8)
            else:
                self.mask_ch = Channel(MemLoc.UB, shape=(tile_vec_m, tile_n),
                                       dtype=dtypes.int8, depth=1).produce()
        if const_expr(mask_mode == 4 and not merged_window_mask):
            if const_expr(mask_in_vector):
                self.left_mask_ch = Buffer(MemLoc.UB, (tile_vec_m, tile_n), dtypes.int8)
            else:
                self.left_mask_ch = Channel(MemLoc.UB, shape=(tile_vec_m, tile_n),
                                            dtype=dtypes.int8, depth=1).produce()
        if const_expr(mask_in_vector and template_window_mask):
            # MTE2 writes template rows; raw VF expands and merges them into
            # the final vector-local mask. Channel supplies this cross-pipe
            # dependency without exposing transaction primitives here.
            self.template_mask_ch = Channel(
                MemLoc.UB, shape=(tile_vec_m, tile_n), dtype=dtypes.int8, depth=1
            ).produce()
            self.template_copy = make_copy_engine(kind="nddma")

    def _nz_params(self, p_ub):
        """Derive NZ fractal addressing from p_ub physical layout stride."""
        s = p_ub.physical_stride
        s_n1, s_m1, s_m0, s_n0 = s[0], s[1], s[2], s[3]
        m0 = s_m1 // s_m0
        n0 = s_m0 // s_n0
        return m0, s_m1, s_m0, s_n1 // n0

    def _softmax_fold_row(
        self,
        qk_ch,
        p_ub,
        base,
        nz_off,
        max_brc_buf,
        row,
        even_mask,
        odd_mask,
        b16,
        b16_full,
        block_stride,
    ):
        """Per-row fold: deinterleave-load qk, exp(sub max), cast to fp16,
        merge even/odd halves, store into p_ub NZ slot. Returns exp halves
        for reduce_sum."""
        mx = rr.vload_broadcast(max_brc_buf, row)
        even_exp, odd_exp = rr.vload_deinterleave(qk_ch, base, width="b32")
        even_exp = rr.vexp_sub(even_exp, mx, mask=even_mask)
        odd_exp = rr.vexp_sub(odd_exp, mx, mask=odd_mask)
        even_fp16 = rr.vcast(
            even_exp, self.dtype_16, mask=even_mask,
            reg_layout=rr.RegLayout.ZERO,
        )
        odd_fp16 = rr.vcast(
            odd_exp, self.dtype_16, mask=odd_mask,
            reg_layout=rr.RegLayout.ONE,
        )
        merged = rr.vbitwise_or(even_fp16, odd_fp16, mask=b16)
        rr.vstore_strided(
            p_ub,
            nz_off,
            merged,
            b16_full,
            block_stride=block_stride,
            repeat_stride=0,
        )
        return even_exp, odd_exp

    # First softmax pass: scale QK scores, apply masks, and record row maxima.
    # The maxima are used for exp(score - max) and normalization.
    def _pass_a_row(
        self,
        qk_ch,
        scale,
        softmax_max_dst,
        row,
        row_stride,
        vector_lanes,
        half0_mask,
        half1_mask,
        full_mask,
        apply_mask=True,
    ):
        """Scale qk row in place, compute rowmax -> softmax_max_dst[row]."""
        base = row * row_stride
        v0 = rr.vmuls(rr.vload(qk_ch, base), scale, mask=half0_mask)
        v1 = rr.vmuls(rr.vload(qk_ch, base + vector_lanes), scale, mask=half1_mask)
        if const_expr(apply_mask and self.mask_mode != 0):
            v0 = self._apply_mask(v0, self.mask_ch, base, half0_mask, True)
            v1 = self._apply_mask(v1, self.mask_ch, base + vector_lanes, half1_mask, True)
            if const_expr(self.mask_mode == 4 and not self.merged_window_mask):
                v0 = self._apply_mask(v0, self.left_mask_ch, base, half0_mask, False)
                v1 = self._apply_mask(
                    v1, self.left_mask_ch, base + vector_lanes, half1_mask, False
                )
        rr.vstore(qk_ch, base, v0, half0_mask)
        rr.vstore(qk_ch, base + vector_lanes, v1, half1_mask)
        rmax = rr.vreduce_max(rr.vmax(v0, v1, mask=full_mask), mask=full_mask)
        if const_expr(self.mask_mode != 0):
            is_all_masked = rr.vles(rmax, MASK_NEG_THRESHOLD, mask=full_mask)
            zeros = rr.vdups(0.0, dtypes.float32, mask=full_mask)
            rmax = rr.vselect(zeros, rmax, cond_mask=is_all_masked)
        rr.vstore_first(softmax_max_dst, row, rmax)

    @jit
    def _fill_mask_rows(self, first_row: int, rows: int, value: int):
        """Fill final-mask rows, including physical padding, with 0 or 1."""
        with vf(mode="simd"):
            first_row = dtypes.int64(first_row)
            full, _ = rr.update_mask(self.tile_n, elem_bits=8)
            data = rr.vdups(value, dtypes.int8, mask=full)
            for row in range(rows):
                row = dtypes.int64(row)
                rr.vstore(self.mask_ch, (first_row + row) * self.tile_n, data, full)

    @jit
    def _copy_template_rows(
        self, attn_mask, template_row: int, template_col: int,
        tokens: int, group: int, valid_n: int,
    ):
        """NDDMA contiguous template rows into G-strided UB source slots."""
        # VF mask arithmetic is i32; dynamic tile/view descriptors require i64.
        tile_tokens = dtypes.int64(tokens)
        tile_valid_n = dtypes.int64(valid_n)
        tile_row = dtypes.int64(template_row)
        tile_col = dtypes.int64(template_col)
        span = make_tiler(
            (tile_tokens, tile_valid_n), alignment=(1, 1)
        )
        source = attn_mask[
            tile_row:tile_row + tile_tokens,
            tile_col:tile_col + tile_valid_n,
        ]
        target = reinterpret(
            self.template_mask_ch, shape=span, stride=(group * self.tile_n, 1)
        )
        mem_copy(target, source, engine=self.template_copy)

    @jit
    def _expand_template_rows(
        self, first_row: int, tokens: int, rows_per_token: int,
        group: int, valid_n: int, merge_left: bool,
    ):
        """Replicate each G-strided template source row into final mask rows."""
        with vf(mode="simd"):
            first_row = dtypes.int64(first_row)
            valid, _ = rr.update_mask(valid_n, elem_bits=8)
            ones = rr.vdups(1, dtypes.int8, mask=valid)
            for token in range(tokens):
                token = dtypes.int64(token)
                template_value = rr.vload(
                    self.template_mask_ch, token * group * self.tile_n
                )
                for group_row in range(rows_per_token):
                    group_row = dtypes.int64(group_row)
                    out = (first_row + token * group + group_row) * self.tile_n
                    value = template_value
                    if const_expr(merge_left):
                        value = rr.vbitwise_xor(value, ones, mask=valid)
                        value = rr.vbitwise_or(
                            rr.vload(self.mask_ch, out), value, mask=valid
                        )
                    rr.vstore(self.mask_ch, out, value, valid)

    def _apply_mask(self, value, channel, offset, lanes, keep_zero):
        packed = rr.vload_unpack(
            channel, offset, unpack_mode=rr.UnpackMode.B8_TO_B32
        )
        # ASC's integer-to-float instruction consumes signed int32 lanes.
        packed = rr.vcast(
            rr.vreinterpret(packed, dtypes.int32),
            dtypes.float32, mask=lanes,
        )
        is_zero = rr.veqs(packed, 0, mask=lanes)
        hidden = rr.vdups(-1.7e38, dtypes.float32, mask=lanes)
        if const_expr(keep_zero):
            return rr.vselect(value, hidden, cond_mask=is_zero)
        else:
            return rr.vselect(hidden, value, cond_mask=is_zero)

    @jit
    def _softmax_fold_loop(
        self,
        qk_ch,
        p_ub,
        max_buf,
        sum_dst,
        actual_n,
        rows,
        src_row_stride,
        tile_n,
        m0,
        s_m1,
        s_m0,
        block_stride,
    ):
        """Per-row fold loop: exp-sub-max into p_ub, reduce_sum into sum_dst.
        Shared by all softmax variants."""
        for row in range(rows):
            full, _ = rr.update_mask(VL_T, elem_bits=32)
            b16_full, _ = rr.update_mask(tile_n, elem_bits=16)
            even_mask, _ = rr.update_mask((actual_n + 1) // 2, elem_bits=32)
            odd_mask, _ = rr.update_mask(actual_n // 2, elem_bits=32)
            b16, _ = rr.update_mask(actual_n, elem_bits=16)
            base = row * src_row_stride
            nz_off = (row // m0) * s_m1 + (row % m0) * s_m0
            even_exp, odd_exp = self._softmax_fold_row(
                qk_ch,
                p_ub,
                base,
                nz_off,
                max_buf,
                row,
                even_mask,
                odd_mask,
                b16,
                b16_full,
                block_stride,
            )
            rsum = rr.vreduce_sum(rr.vadd(even_exp, odd_exp, mask=full), mask=even_mask)
            rr.vstore_first(sum_dst, row, rsum)

    @jit
    def softmax_first(self, qk_ch, scale, m_axis_triple: int, apply_mask=True):
        """First n-tile of a new m-tile: P = softmax(S); init running max/sum."""
        softmax_max = self.softmax_max_bufs[m_axis_triple]
        softmax_sum = self.softmax_sum_bufs[m_axis_triple]

        p_ub = self.p_ub.produce()
        tile_n = self.tile_n
        m0, s_m1, s_m0, block_stride = self._nz_params(p_ub)
        with vf(mode="simd"):
            rows, actual_n = qk_ch.shape
            src_row_stride = qk_ch.physical_stride[0]
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                half0_mask, _ = rr.update_mask(actual_n, elem_bits=32)
                half1_mask, _ = rr.update_mask(
                    _clamp_nonneg(actual_n - VL_T), elem_bits=32
                )
                self._pass_a_row(
                    qk_ch,
                    scale,
                    softmax_max,
                    row,
                    src_row_stride,
                    VL_T,
                    half0_mask,
                    half1_mask,
                    full,
                    apply_mask,
                )
            rr.vmem_bar("vst_vld")
            self._softmax_fold_loop(
                qk_ch,
                p_ub,
                softmax_max,
                softmax_sum,
                actual_n,
                rows,
                src_row_stride,
                tile_n,
                m0,
                s_m1,
                s_m0,
                block_stride,
            )

    def _softmax_rest_tail(self, softmax_max, softmax_sum, softmax_exp, rowmask):
        """Online-softmax running-state update."""
        old_max = rr.vload(softmax_max, 0)
        new_max = rr.vload(self.tmp_new_max, 0)
        old_scale = rr.vexp_sub(old_max, new_max, mask=rowmask)  # exp(old-new)
        rr.vstore(softmax_exp, 0, old_scale, rowmask)
        rr.vstore(softmax_max, 0, new_max, rowmask)  # max = new_max
        old_sum = rr.vload(softmax_sum, 0)
        new_sum = rr.vload(self.tmp_sum, 0)
        updated_sum = rr.vmadd(old_sum, old_scale, new_sum, mask=rowmask)
        rr.vstore(softmax_sum, 0, updated_sum, rowmask)

    @jit
    def softmax_rest(self, qk_ch, scale, m_axis_triple: int, tile_triple: int, apply_mask=True):
        """Non-first n-tile: rescale running max/sum, P = exp(S - new_max)."""
        softmax_max = self.softmax_max_bufs[m_axis_triple]
        softmax_sum = self.softmax_sum_bufs[m_axis_triple]
        softmax_exp = self.softmax_exp_bufs[tile_triple]

        p_ub = self.p_ub.produce()
        tile_n = self.tile_n
        m0, s_m1, s_m0, block_stride = self._nz_params(p_ub)
        with vf(mode="simd"):
            rows, actual_n = qk_ch.shape
            src_row_stride = qk_ch.physical_stride[0]
            rowmask, _ = rr.update_mask(rows, elem_bits=32)
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                half0_mask, _ = rr.update_mask(actual_n, elem_bits=32)
                half1_mask, _ = rr.update_mask(
                    _clamp_nonneg(actual_n - VL_T), elem_bits=32
                )
                self._pass_a_row(
                    qk_ch,
                    scale,
                    self.tmp_new_max,
                    row,
                    src_row_stride,
                    VL_T,
                    half0_mask,
                    half1_mask,
                    full,
                    apply_mask,
                )
            rr.vmem_bar("vst_vld")
            nm = rr.vmax(
                rr.vload(softmax_max, 0), rr.vload(self.tmp_new_max, 0), mask=rowmask
            )
            rr.vstore(self.tmp_new_max, 0, nm, rowmask)
            rr.vmem_bar("vst_vld")
            self._softmax_fold_loop(
                qk_ch,
                p_ub,
                self.tmp_new_max,
                self.tmp_sum,
                actual_n,
                rows,
                src_row_stride,
                tile_n,
                m0,
                s_m1,
                s_m0,
                block_stride,
            )
            rr.vmem_bar("vst_vld")
            self._softmax_rest_tail(softmax_max, softmax_sum, softmax_exp, rowmask)

    @jit
    def softmax_first_unmasked(self, qk_ch, scale, m_axis_triple: int):
        self.softmax_first(qk_ch, scale, m_axis_triple, apply_mask=False)

    @jit
    def softmax_rest_unmasked(self, qk_ch, scale, m_axis_triple: int, tile_triple: int):
        self.softmax_rest(qk_ch, scale, m_axis_triple, tile_triple, apply_mask=False)

    def store_p(self, p_l1_ch, active_m, active_n):
        """p_ub -> the explicit per-AIV M tile of the shared L1 slot."""
        mem_copy(
            p_l1_ch.produce(),
            self.p_ub.consume(),
            engine=make_copy_engine(split_axis=0),
            part_id=self.subblock_idx,
            actual=(active_m, active_n),
        )

    def init_o(self, pv_ch):
        """First PV tile: res_o = P0 @ V0."""
        mem_copy(reinterpret(self.res_o, shape=(pv_ch.shape[0], self.tile_d)), pv_ch)

    @jit
    def init_o_chunk(self, pv_ch, d_offset: int):
        """Initialize one contiguous D chunk of the FP32 output accumulator."""
        with vf(mode="simd"):
            for row in range(pv_ch.shape[0]):
                src_base = row * self.pv_tile_d
                dst_base = row * self.tile_d + d_offset
                for col in tuple(range(0, self.pv_tile_d, VL_T)):
                    mask, _ = rr.update_mask(
                        self.pv_tile_d - col, elem_bits=32
                    )
                    val = rr.vload(pv_ch, src_base + col)
                    rr.vstore(self.res_o, dst_base + col, val, mask)

    @jit
    def update_o_chunk(self, pv_ch, softmax_exp_buf, d_offset: int):
        """Online-softmax update for one D chunk."""
        with vf(mode="simd"):
            for row in range(pv_ch.shape[0]):
                exp_b = rr.vload_broadcast(softmax_exp_buf, row)
                src_base = row * self.pv_tile_d
                dst_base = row * self.tile_d + d_offset
                for col in tuple(range(0, self.pv_tile_d, VL_T)):
                    mask, _ = rr.update_mask(
                        self.pv_tile_d - col, elem_bits=32
                    )
                    pre = rr.vload(self.res_o, dst_base + col)
                    cur = rr.vload(pv_ch, src_base + col)
                    val = rr.vmadd(pre, exp_b, cur, mask=mask)
                    rr.vstore(self.res_o, dst_base + col, val, mask)

    @jit
    def update_o_last_chunk(
        self, pv_ch, softmax_exp_buf, softmax_sum_buf, d_offset: int
    ):
        """Final online-softmax update and normalization for one D chunk."""
        with vf(mode="simd"):
            for row in range(pv_ch.shape[0]):
                exp_b = rr.vload_broadcast(softmax_exp_buf, row)
                sum_b = rr.vload_broadcast(softmax_sum_buf, row)
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                is_zero = rr.veqs(sum_b, 0.0, mask=full)
                one_b = rr.vdups(1.0, dtypes.float32, mask=full)
                safe_sum = rr.vselect(one_b, sum_b, cond_mask=is_zero)
                zero_b = rr.vdups(0.0, dtypes.float32, mask=full)
                src_base = row * self.pv_tile_d
                dst_base = row * self.tile_d + d_offset
                for col in tuple(range(0, self.pv_tile_d, VL_T)):
                    mask, _ = rr.update_mask(
                        self.pv_tile_d - col, elem_bits=32
                    )
                    pre = rr.vload(self.res_o, dst_base + col)
                    cur = rr.vload(pv_ch, src_base + col)
                    val = rr.vmadd(pre, exp_b, cur, mask=mask)
                    val = rr.vdiv(val, safe_sum, mask=mask)
                    val = rr.vselect(zero_b, val, cond_mask=is_zero)
                    rr.vstore(self.res_o, dst_base + col, val, mask)

    @jit
    def init_o_last_chunk(self, pv_ch, softmax_sum_buf, d_offset: int):
        """Initialize and normalize a single-KV-tile result D chunk."""
        with vf(mode="simd"):
            for row in range(pv_ch.shape[0]):
                sum_b = rr.vload_broadcast(softmax_sum_buf, row)
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                is_zero = rr.veqs(sum_b, 0.0, mask=full)
                one_b = rr.vdups(1.0, dtypes.float32, mask=full)
                safe_sum = rr.vselect(one_b, sum_b, cond_mask=is_zero)
                zero_b = rr.vdups(0.0, dtypes.float32, mask=full)
                src_base = row * self.pv_tile_d
                dst_base = row * self.tile_d + d_offset
                for col in tuple(range(0, self.pv_tile_d, VL_T)):
                    mask, _ = rr.update_mask(
                        self.pv_tile_d - col, elem_bits=32
                    )
                    val = rr.vload(pv_ch, src_base + col)
                    val = rr.vdiv(val, safe_sum, mask=mask)
                    val = rr.vselect(zero_b, val, cond_mask=is_zero)
                    rr.vstore(self.res_o, dst_base + col, val, mask)

    @jit
    def update_o(self, pv_ch, softmax_exp_buf):
        """Non-first PV tile: res_o = res_o * exp(old_max-new_max) + P_i @ V_i."""

        with vf(mode="simd"):
            for row in range(pv_ch.shape[0]):
                exp_b = rr.vload_broadcast(softmax_exp_buf, row)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    mask, _ = rr.update_mask(self.tile_d - col, elem_bits=32)
                    pre = rr.vload(self.res_o, off)
                    cur = rr.vload(pv_ch, off)
                    o = rr.vmadd(pre, exp_b, cur, mask=mask)
                    rr.vstore(self.res_o, off, o, mask)

    @jit
    def update_o_last(self, pv_ch, softmax_exp_buf, softmax_sum_buf):
        """Last PV tile: res_o = (res_o * exp + pv) / sum.

        Fuses division. Fully-masked rows (sum==0) output 0.
        """

        with vf(mode="simd"):
            for row in range(pv_ch.shape[0]):
                exp_b = rr.vload_broadcast(softmax_exp_buf, row)
                sum_b = rr.vload_broadcast(softmax_sum_buf, row)
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                is_zero = rr.veqs(sum_b, 0.0, mask=full)
                one_b = rr.vdups(1.0, dtypes.float32, mask=full)
                safe_sum = rr.vselect(one_b, sum_b, cond_mask=is_zero)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    mask, _ = rr.update_mask(self.tile_d - col, elem_bits=32)
                    pre = rr.vload(self.res_o, off)
                    cur = rr.vload(pv_ch, off)
                    o = rr.vmadd(pre, exp_b, cur, mask=mask)
                    o = rr.vdiv(o, safe_sum, mask=mask)
                    zero_b = rr.vdups(0.0, dtypes.float32, mask=full)
                    o = rr.vselect(zero_b, o, cond_mask=is_zero)
                    rr.vstore(self.res_o, off, o, mask)

    @jit
    def init_o_last(self, pv_ch, softmax_sum_buf):
        """Single-tile case: res_o = pv / sum. Fully-masked rows output 0."""

        with vf(mode="simd"):
            for row in range(pv_ch.shape[0]):
                sum_b = rr.vload_broadcast(softmax_sum_buf, row)
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                is_zero = rr.veqs(sum_b, 0.0, mask=full)
                one_b = rr.vdups(1.0, dtypes.float32, mask=full)
                safe_sum = rr.vselect(one_b, sum_b, cond_mask=is_zero)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    mask, _ = rr.update_mask(self.tile_d - col, elem_bits=32)
                    cur = rr.vload(pv_ch, off)
                    val = rr.vdiv(cur, safe_sum, mask=mask)
                    zero_b = rr.vdups(0.0, dtypes.float32, mask=full)
                    val = rr.vselect(zero_b, val, cond_mask=is_zero)
                    rr.vstore(self.res_o, off, val, mask)

    @jit
    def _finalize_div_vf(self, softmax_sum_buf, actual_vec_m):
        """res_o /= softmax_sum. Fully-masked rows (sum==0) output 0."""

        with vf(mode="simd"):
            for row in range(actual_vec_m):
                sum_b = rr.vload_broadcast(softmax_sum_buf, row)
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                is_zero = rr.veqs(sum_b, 0.0, mask=full)
                one_b = rr.vdups(1.0, dtypes.float32, mask=full)
                safe_sum = rr.vselect(one_b, sum_b, cond_mask=is_zero)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    mask, _ = rr.update_mask(self.tile_d - col, elem_bits=32)
                    val = rr.vdiv(rr.vload(self.res_o, off), safe_sum, mask=mask)
                    zero_b = rr.vdups(0.0, dtypes.float32, mask=full)
                    val = rr.vselect(zero_b, val, cond_mask=is_zero)
                    rr.vstore(self.res_o, off, val, mask)


class _ExternalLse:
    """Write final max + log(sum), preserving +inf for invalid rows."""

    # Reserve temporary UB storage and an 8-lane LSE writeback workspace.
    def __init__(self, capacity):
        self.capacity = capacity
        self.values = Channel(MemLoc.UB, shape=(capacity, 1),
                              dtype=dtypes.float32, depth=1).produce()
        self.scattered = Channel(MemLoc.UB, shape=(capacity, LSE_ROW_LANES),
                                 dtype=dtypes.float32, depth=1).produce()

    @jit
    def store_merged(self, softmax_max, softmax_sum, gm, layout, batch: int,
                      kv_head: int, start: int, rows: int, group: int,
                      length: int, total_q: int, heads: int, cu, s1g):
        # First stage: compute per-row LSE from online softmax max and sum.
        with vf(mode="simd"):
            full_mask, _ = rr.update_mask(64, elem_bits=32)
            ones_reg = rr.vdups(1.0, dtypes.float32, mask=full_mask)
            zeros_reg = rr.vdups(0.0, dtypes.float32, mask=full_mask)
            inf_reg = rr.vdiv(ones_reg, zeros_reg, mask=full_mask)
            for row in range(0, rows, LSE_ROW_LANES):
                count = min(LSE_ROW_LANES, rows - row)
                mask, _ = rr.update_mask(count * LSE_ROW_LANES, elem_bits=32)
                mx_bits = rr.vload_broadcast(
                    softmax_max, row, mode="elem2datablock")
                sum_bits = rr.vload_broadcast(
                    softmax_sum, row, mode="elem2datablock")
                mx = rr.vreinterpret(mx_bits, dtypes.float32)
                sum_value = rr.vreinterpret(sum_bits, dtypes.float32)
                invalid = rr.veqs(sum_value, 0.0, mask=mask)
                safe = rr.vselect(ones_reg, sum_value, cond_mask=invalid)
                value = rr.vadd(mx, rr.vlog(safe, mask=mask), mask=mask)
                value = rr.vselect(inf_reg, value, cond_mask=invalid)
                rr.vstore(self.scattered, row * LSE_ROW_LANES, value, mask)

        # Second stage: view one-dimensional LSE output as head/token coordinates.
        by_head = gm.view((gm.shape[0] // total_q, total_q, 1))
        # S1G rows are adjacent in output GM only when G is one.
        s1g_contiguous_rows = s1g and (group == 1)
        if const_expr(s1g):
            by_token = permute(by_head, (1, 0, 2))

        # Preserve the original stride path unless S1G output rows are adjacent.
        span = group if s1g else length
        if const_expr(s1g_contiguous_rows):
            span = length
        first = start % span
        segments = (ceil_div(first + rows, span) if rows > 0 else 0)
        for segment in range(segments):
            offset = (0 if segment == 0 else segment * span - first)
            logical = start + offset
            token, g = _merged_q_row(logical, group, length, s1g)
            count = min(rows - offset, span - logical % span)
            head = kv_head * group + g
            if const_expr(layout == "TND"):
                token = token + cu[batch]
            else:
                head = batch * heads + head
            if const_expr(s1g_contiguous_rows):
                part = by_head[head, token:token + count, None]
            elif const_expr(s1g):
                part = by_token[token, head:head + count, None]
            else:
                part = by_head[head, token:token + count, None]
            dst = tile_slice(part, (self.capacity, 1), (0, 0))
            src = _nd_rows(
                self.scattered, offset, count, 1,
                row_stride=LSE_ROW_LANES,
            )
            mem_copy(dst, src)


@cb.kernel(profile=cb.ProfileSpec(name="flash_attn", op_type="flash_attn"))
class FlashAttnKernel:
    def __init__(
        self,
        tile_cube_m,
        tile_vec_m,
        tile_n,
        tile_d,
        return_softmax_lse,
        mask_mode=0,
        dtype_16=dtypes.float16,
        layout="BNSD",
        batch_count=0,
        s1g_mapping=False,
        kv_layout="BNSD",
        output_layout="BNSD",
        pa_strided=False,
    ):
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.return_softmax_lse = return_softmax_lse
        self.mask_mode = mask_mode
        self.dtype_16 = dtype_16
        self.s1g_mapping = s1g_mapping
        self.preload_num = PIPELINE_DEPTH
        self.layout = layout
        self.kv_layout = kv_layout
        self.output_layout = output_layout
        self._batch_count = batch_count
        self._pa_strided = pa_strided
        # The singleton ND batch keeps stack_m's declared 128-row NZ pitch,
        # even when the source page segment has fewer rows.
        self.pa_page_nd2nz = make_copy_engine(
            format_transform="nd2nz", dst_nd_arrangement="stack_m",
        )
        if pa_strided:
            # Two scratch buffers shared by K and V: only one 16-row ND
            # fragment and one half-head NZ tile per AIV.
            self.pa_nd_ub = Buffer(MemLoc.UB, (16, tile_d // 2), dtype_16)
            nz_rows = 16 if tile_d > CUBE_D_TILE and tile_cube_m > 64 else tile_n
            self.pa_nz_ub = Buffer(MemLoc.UB, (tile_d // 32, nz_rows * 16), dtype_16)
            self.pa_ub_to_l1 = make_copy_engine(split_axis=1)
        # One source row at a time is the public-API fallback that covers every
        # S1G/GS1 boundary. NZ rows are 32-byte C0 units for fp16/bf16.
        self.q_row_copy = make_copy_engine(format_transform="nd2nz")

        self.qk_ub = Channel(
            MemLoc.UB,
            shape=(tile_vec_m, tile_n),
            dtype=dtypes.float32,
            depth=(4 if tile_d > CUBE_D_TILE and tile_n == CUBE_D_TILE else 2),
            kind=ChannelKind.CrossCore,
        )
        if tile_d > CUBE_D_TILE:
            if tile_n == CUBE_D_TILE:
                # QK and both PV chunks carry the same CrossCore FIFO message
                # shape. Shared identity keeps producer/consumer cursors ordered.
                self.pv_ubs = tuple(
                    self.qk_ub for _ in range(tile_d // CUBE_D_TILE)
                )
            else:
                self.pv_ubs = tuple(Channel(
                    MemLoc.UB, shape=(tile_vec_m, CUBE_D_TILE),
                    dtype=dtypes.float32, depth=2, kind=ChannelKind.CrossCore,
                ) for _ in range(tile_d // CUBE_D_TILE))
        else:
            self.pv_ub = Channel(
                MemLoc.UB, shape=(tile_vec_m, tile_d), dtype=dtypes.float32,
                depth=2, kind=ChannelKind.CrossCore,
            )
        self.p_l1 = Channel(
            MemLoc.L1,
            shape=(tile_cube_m, tile_n),
            dtype=dtype_16,
            depth=3,
            kind=ChannelKind.CrossCore,
        )

        self.block_idx = get_block_idx()
        self.subblock_idx = get_subblock_id()
        self.matmul = Matmul(tile_cube_m, tile_n, tile_d, dtype_16,
                             paged_nz=kv_layout == "PA_NZ", paged_strided=pa_strided)
        self.lse_ub = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        if return_softmax_lse:
            self.external_lse = _ExternalLse(tile_vec_m)
        # Stage each split-KV row LSE in the 8-lane format used by workspace stores.
        self.emit_lse_tail = Channel(
            MemLoc.UB, shape=(tile_vec_m, LSE_ROW_LANES),
            dtype=dtypes.float32, depth=1,
        ).produce()

    @jit
    def _load_merged_q(self, batch: int, kv_head: int, start: int, rows: int, q_l1):
        """Copy one merged S1G/GS1 Q block from GM directly into NZ L1."""
        length = self._q_length(batch)
        group = self._head_group_num
        # Without G splitting, convert the Q rows directly to NZ and copy to q_l1.
        if const_expr(group == 1):
            # One head forms a regular strided row view in every layout.
            # Keep the source stride so BSND/TND use one DMA for the Q tile.
            if const_expr(self.layout == "TND"):
                token = start + self._cu_seqlens_q[batch]
                source = self._query[kv_head, token:token + rows, None]
            else:
                source = self._query[batch, kv_head, start:start + rows, None]
            source = tile_slice(source, (self.tile_cube_m, self.tile_d), (0, 0))
            mem_copy(q_l1, source, engine=self.q_row_copy)
        elif const_expr(
            self.s1g_mapping
            and self.tile_cube_m % group == 0
            and group > 1
        ):
            # Use the bulk path when tile, start, and length align to complete G blocks.
            regular = start % group == 0 and rows % group == 0
            if regular:
                # Copy each complete G block as one source into q_l1.
                token = start // group
                if const_expr(self.layout == "TND"):
                    token = token + self._cu_seqlens_q[batch]
                    token_major = permute(self._query, (1, 0, 2))
                else:
                    token_major = permute(
                        self._query, (0, 2, 1, 3)
                    )[batch, None, None, None]
                flat_rows = token_major.view((-1, self.tile_d))
                block_step = self._head_num_q // group
                first_block = token * block_step + kv_head
                active_blocks = rows // group
                for block_count in tuple(
                    range(1, self.tile_cube_m // group + 1)
                ):
                    if active_blocks == block_count:
                        sources = []
                        for block in tuple(range(block_count)):
                            sources.append(
                                tile_slice(
                                    flat_rows,
                                    (group, self.tile_d),
                                    (first_block + block * block_step, 0),
                                )
                            )
                        if const_expr(block_count == 1):
                            mem_copy(
                                q_l1,
                                sources[0],
                                engine=self.q_row_copy,
                            )
                        else:
                            mem_copy(
                                q_l1,
                                sources,
                                engine=self.q_row_copy,
                                axis=0,
                            )
            else:
                # Fall back to segmented assembly when start or length cuts a G block.
                self._load_merged_q_segments(
                    batch, kv_head, start, rows, length, q_l1
                )
        else:
            # Assemble all other G-split layouts by segments before writing q_l1.
            self._load_merged_q_segments(
                batch, kv_head, start, rows, length, q_l1
            )

    @jit
    def _load_merged_q_segments(
        self, batch: int, kv_head: int, start: int, rows: int, length: int, q_l1
    ):
        """Fallback for tails and non-regular merged-row boundaries."""
        # Determine each segment's maximum rows: S1G splits by G, others by sequence.
        span = self._head_group_num if const_expr(self.s1g_mapping) else max(length, 1)
        first = start % span
        segments = (ceil_div(first + rows, span) if rows > 0 else 0)
        for segment in range(segments):
            local_row = (0 if segment == 0 else segment * span - first)
            src, count = _merged_segment(
                self._query,
                self.layout,
                batch,
                kv_head,
                start + local_row,
                rows - local_row,
                self._head_group_num,
                length,
                self._cu_seqlens_q,
                self.s1g_mapping,
            )
            if segments == 1:
                mem_copy(q_l1, src, engine=self.q_row_copy)
            else:
                dst = reinterpret(
                    q_l1,
                    shape=make_tiler((count, self.tile_d), alignment=(1, 1)),
                    offset=local_row * 32,
                )
                mem_copy(dst, src, engine=self.q_row_copy)

    # Check whether this AIV's Q rows fully see the KV tile and can skip masking.
    @jit
    def _merged_mask_fully_visible(self, batch: int, start: int,
                                    rows: int, n_idx: int, valid_n: int):
        """Prove every active AIV row can attend to every active KV column."""
        sub_lo = (0 if self.subblock_idx == 0 else ceil_div(rows, 2))
        count = (ceil_div(rows, 2) if self.subblock_idx == 0 else rows // 2)
        first_row = start + sub_lo
        last_row = first_row + count - 1
        q_length = max(self._q_length(batch), 1)
        if const_expr(self.s1g_mapping):
            group = max(self._head_group_num, 1)
            q_first = first_row // group
            q_last = last_row // group
        else:
            # Crossing any GS1 head boundary includes both token 0 and
            # token q_length-1; use that conservative interval.
            same_head = first_row // q_length == last_row // q_length
            q_first = (first_row % q_length if same_head else 0)
            q_last = (last_row % q_length if same_head else q_length - 1)
        shift = self._causal_offset(batch)
        key_first = n_idx * self.tile_n
        key_last = key_first + valid_n - 1
        right_ok = q_first + shift >= key_last
        left_ok = True
        if const_expr(self.mask_mode == 4):
            right_ok = (
                self._win_right < 0
                or q_first + shift + self._win_right >= key_last
            )
            left_ok = self._win_left < 0 or q_last + shift - self._win_left <= key_first
        return (1 if count > 0 and valid_n > 0 and right_ok and left_ok else 0)

    # Copy an edge template mask after G splitting and expand it to valid G rows.
    @jit
    def _copy_merged_template_edge(
        self, delta: int, first_row: int, rows_per_token: int,
        valid_n: int, merge_left: bool,
    ):
        """Copy one head/tail token template row and expand its live S1G rows."""
        group = self._head_group_num
        zero_i32 = dtypes.int32(0)
        one_i32 = dtypes.int32(1)
        source_row = (
            zero_i32 if delta < zero_i32
            else valid_n - one_i32 if delta >= valid_n
            else delta
        )
        # T[0, 1:C+1] is all one; T[C-1, 0:C] is all zero.
        source_col = (one_i32 if delta < zero_i32 else zero_i32)
        self.vector._copy_template_rows(
            self._attn_mask, source_row, source_col, 1, group, valid_n
        )
        self.vector._expand_template_rows(
            first_row, 1, rows_per_token, group, valid_n, merge_left
        )

    # S1G window bulk path: build the final mask from edge and complete G blocks.
    @jit
    def _load_merged_template_mask(
        self, batch: int, start: int, rows: int, n_idx: int, valid_n: int,
    ):
        """Mainline ProcessS1G + BAND mask producer.

        The full-token middle is copied as contiguous template rows by MTE2;
        VF expands each row to its G S1G rows.  The optional head and tail are
        the same generic decomposition as AttentionmaskCopyInForSgLayout.
        """
        sub_lo = (0 if self.subblock_idx == 0 else ceil_div(rows, 2))
        count = dtypes.int32(
            (ceil_div(rows, 2) if self.subblock_idx == 0 else rows // 2)
        )
        valid_n = dtypes.int32(valid_n)
        group = self._head_group_num
        logical = dtypes.int32(start + sub_lo)
        q0 = logical // group
        g0 = logical % group
        zero_i32 = dtypes.int32(0)
        one_i32 = dtypes.int32(1)
        head = (zero_i32 if g0 == zero_i32 else min(count, group - g0))
        remaining = count - head
        middle = remaining // group
        tail = remaining % group
        middle_q = q0 + (one_i32 if head > zero_i32 else zero_i32)
        tail_q = middle_q + middle
        shift = dtypes.int32(self._causal_offset(batch) - n_idx * self.tile_n)
        left_window = dtypes.int32(self._win_left)
        right_window = dtypes.int32(self._win_right)

        # Final-mask padding is hidden before any valid-C template writes.
        self.vector._fill_mask_rows(0, count, 1)

        if self._win_right < 0:
            self.vector._fill_mask_rows(0, count, 0)
        else:
            if head > 0:
                self._copy_merged_template_edge(
                    q0 + shift + right_window, 0, head, valid_n, False
                )
            if middle > 0:
                delta = middle_q + shift + right_window
                negative = min(middle, max(zero_i32, zero_i32 - delta))
                first_delta = max(delta, zero_i32)
                partial = min(
                    middle - negative, max(zero_i32, valid_n - first_delta)
                )
                positive = middle - negative - partial
                if partial > 0:
                    self.vector._copy_template_rows(
                        self._attn_mask, first_delta, 0, partial, group, valid_n
                    )
                    self.vector._expand_template_rows(
                        head + negative * group, partial, group, group,
                        valid_n, False,
                    )
                if positive > 0:
                    self.vector._fill_mask_rows(
                        head + (negative + partial) * group,
                        positive * group, 0,
                    )
            if tail > 0:
                self._copy_merged_template_edge(
                    tail_q + shift + right_window,
                    head + middle * group, tail, valid_n, False,
                )

        if self._win_left >= 0:
            if head > 0:
                self._copy_merged_template_edge(
                    q0 + shift - left_window - 1, 0, head, valid_n, True
                )
            if middle > 0:
                delta = middle_q + shift - left_window - 1
                negative = min(middle, max(zero_i32, zero_i32 - delta))
                first_delta = max(delta, zero_i32)
                partial = min(
                    middle - negative, max(zero_i32, valid_n - first_delta)
                )
                positive = middle - negative - partial
                if partial > 0:
                    self.vector._copy_template_rows(
                        self._attn_mask, first_delta, 0, partial, group, valid_n
                    )
                    self.vector._expand_template_rows(
                        head + negative * group, partial, group, group,
                        valid_n, True,
                    )
                if positive > 0:
                    self.vector._fill_mask_rows(
                        head + (negative + partial) * group,
                        positive * group, 1,
                    )
            if tail > 0:
                self._copy_merged_template_edge(
                    tail_q + shift - left_window - 1,
                    head + middle * group, tail, valid_n, True,
                )

    @jit
    def _load_gs1_template_mask(
        self, batch: int, start: int, rows: int, n_idx: int, valid_n: int,
    ):
        """Build a GS1 window mask from contiguous token template segments."""
        sub_lo = (0 if self.subblock_idx == 0 else ceil_div(rows, 2))
        count = dtypes.int32(
            ceil_div(rows, 2) if self.subblock_idx == 0 else rows // 2
        )
        length = max(dtypes.int32(self._q_length(batch)), 1)
        token = dtypes.int32(start + sub_lo) % length
        valid_n = dtypes.int32(valid_n)
        shift = dtypes.int32(self._causal_offset(batch) - n_idx * self.tile_n)
        zero_i32 = dtypes.int32(0)
        left_window = dtypes.int32(self._win_left)
        right_window = dtypes.int32(self._win_right)
        first_row = zero_i32

        # Hidden by default, including physical padding. Each head segment
        # then fills or copies only its own final-mask rows.
        self.vector._fill_mask_rows(0, count, 1)
        for segment in range(ceil_div(token + count, length)):
            piece_rows = min(count - first_row, length - token)
            if self._win_right < 0:
                self.vector._fill_mask_rows(first_row, piece_rows, 0)
            else:
                delta = token + shift + right_window
                negative = min(piece_rows, max(zero_i32, zero_i32 - delta))
                first_delta = max(delta, zero_i32)
                partial = min(
                    piece_rows - negative,
                    max(zero_i32, valid_n - first_delta),
                )
                positive = piece_rows - negative - partial
                if partial > 0:
                    self.vector._copy_template_rows(
                        self._attn_mask, first_delta, 0, partial, 1, valid_n
                    )
                    self.vector._expand_template_rows(
                        first_row + negative, partial, 1, 1, valid_n, False
                    )
                if positive > 0:
                    self.vector._fill_mask_rows(
                        first_row + negative + partial, positive, 0
                    )

            if self._win_left >= 0:
                delta = token + shift - left_window - 1
                negative = min(piece_rows, max(zero_i32, zero_i32 - delta))
                first_delta = max(delta, zero_i32)
                partial = min(
                    piece_rows - negative,
                    max(zero_i32, valid_n - first_delta),
                )
                positive = piece_rows - negative - partial
                if partial > 0:
                    self.vector._copy_template_rows(
                        self._attn_mask, first_delta, 0, partial, 1, valid_n
                    )
                    self.vector._expand_template_rows(
                        first_row + negative, partial, 1, 1, valid_n, True
                    )
                if positive > 0:
                    self.vector._fill_mask_rows(
                        first_row + negative + partial, positive, 1
                    )
            first_row = first_row + piece_rows
            token = zero_i32

    # Generic mask path: compute each row's visible KV range and write masks.
    @jit
    def _load_merged_mask(self, batch: int, start: int, rows: int, n_idx: int):
        sub_lo = (0 if self.subblock_idx == 0 else ceil_div(rows, 2))
        count = (ceil_div(rows, 2) if self.subblock_idx == 0 else rows // 2)
        # Keep scalar arithmetic inside VF in the supported 32-bit domain.
        length = dtypes.int32(self._q_length(batch))
        shift = dtypes.int32(self._causal_offset(batch) - n_idx * self.tile_n)
        row_start = dtypes.int32(start + sub_lo)
        row_count = dtypes.int32(count)
        left_window = dtypes.int32(self._win_left)
        right_window = dtypes.int32(self._win_right)
        # GS1 rows advance within a head. Hoist the runtime remainder
        # out of VF and carry a wrapped token, including length < count.
        gs1_length = max(length, 1)
        gs1_start = dtypes.int32(0)
        if const_expr(not self.s1g_mapping):
            gs1_start = row_start % gs1_length
        with vf(mode="simd"):
            gs1_token = gs1_start
            full, _ = rr.update_mask(self.tile_n, elem_bits=8)
            ones = rr.vdups(1, dtypes.int8, mask=full)
            for row in range(row_count):
                if const_expr(self.s1g_mapping):
                    token, _ = _merged_q_row(row_start + row, self._head_group_num,
                                            length, self.s1g_mapping)
                else:
                    token = gs1_token
                center = token + shift
                right_count = center + 1
                if const_expr(self.mask_mode == 4):
                    right_count = (
                        dtypes.int32(self.tile_n) if right_window < 0
                        else right_count + right_window
                    )
                visible, _ = rr.update_mask(min(self.tile_n, max(0, right_count)), elem_bits=8)
                right = rr.vdups(0, dtypes.int8, mask=visible, mode="merging", merge=ones)
                if const_expr(self.mask_mode == 4):
                    left_count = (dtypes.int32(0) if left_window < 0 else center - left_window)
                    hidden, _ = rr.update_mask(min(self.tile_n, max(0, left_count)), elem_bits=8)
                    if const_expr(self.vector.merged_window_mask):
                        merged = rr.vdups(1, dtypes.int8, mask=hidden,
                                         mode="merging", merge=right)
                        rr.vstore(self.vector.mask_ch, row * self.tile_n, merged, full)
                    else:
                        left = rr.vdups(0, dtypes.int8, mask=hidden,
                                       mode="merging", merge=ones)
                        rr.vstore(self.vector.mask_ch, row * self.tile_n, right, full)
                        rr.vstore(self.vector.left_mask_ch, row * self.tile_n, left, full)
                else:
                    rr.vstore(self.vector.mask_ch, row * self.tile_n, right, full)
                if const_expr(not self.s1g_mapping):
                    next_token = gs1_token + 1
                    gs1_token = (next_token if next_token < gs1_length else 0)

    @jit
    def _stage_qk(
        self, n_idx: int, batch_idx: int, kv_head: int,
        q_tile_gm: Tensor, q_source,
    ):
        """S = Q @ K^T -> qk_ub."""
        if const_expr(self.kv_layout in ("PA_BNBD", "PA_BBND", "PA_NZ")):
            self._load_pa_kv_tile(self._key, batch_idx, kv_head, n_idx)
            valid_n = min(self.tile_n, self._kv_length(batch_idx) - n_idx * self.tile_n)
        else:
            k_tile_gm = self._kv_tile(self._key, batch_idx, kv_head, n_idx)
            self.matmul.load_k(k_tile_gm)
            valid_n = k_tile_gm.shape[0]
        result = self.matmul.compute_qk(q_tile_gm.shape[0], q_source, valid_n)
        self.matmul.store_s(
            self.qk_ub, q_tile_gm.shape[0], valid_n, result
        )

    @jit
    def _stage_softmax(self, tick: int, tile_idx: int, n_idx: int, m_seq: int,
                       n_start: int = 0):
        """Normalize this fragment starting at n_start, then publish P to L1."""
        batch_idx, head_idx, m_idx, tok0, _rows, _n_end = self._tile_coords(
            tile_idx
        )

        valid_n = min(self.tile_n, self._kv_length(batch_idx) - n_idx * self.tile_n)
        actual_vec_m = (ceil_div(_rows, 2) if self.subblock_idx == 0 else _rows // 2)
        # Both layouts expose only this batch's effective KV rows.

        # First KV tile of THIS task's sweep. A whole (batch,head) tile starts
        # at 0; a split task owns only [n_start, n_end), so its running softmax
        # must (re)initialise at n_start, not a global 0. `n_start` arrives as a
        # parameter (threaded through the pipeline by the ASC walk; 0 otherwise).
        is_first = 1 if n_idx == n_start else 0
        tile_triple = (tick - 1) % 3
        m_axis_triple = m_seq % self.preload_num

        full_visible = dtypes.int32(0)
        if const_expr(self.mask_mode != 0):
            full_visible = self._merged_mask_fully_visible(batch_idx, tok0, _rows, n_idx, valid_n)
            if full_visible == 0:
                if self._template_window_mask:
                    if const_expr(self.s1g_mapping):
                        self._load_merged_template_mask(
                            batch_idx, tok0, _rows, n_idx, valid_n
                        )
                    else:
                        self._load_gs1_template_mask(
                            batch_idx, tok0, _rows, n_idx, valid_n
                        )
                else:
                    self._load_merged_mask(batch_idx, tok0, _rows, n_idx)
        qk_view = reinterpret(
            self.qk_ub.consume(), shape=(actual_vec_m, valid_n), stride=(self.tile_n, 1),
        )
        if const_expr(self.mask_mode != 0):
            # Choose the VF once per block, not dynamically for each row.
            if full_visible:
                if is_first:
                    self.vector.softmax_first_unmasked(qk_view, self._scale, m_axis_triple)
                else:
                    self.vector.softmax_rest_unmasked(
                        qk_view, self._scale, m_axis_triple, tile_triple
                    )
            else:
                if is_first:
                    self.vector.softmax_first(qk_view, self._scale, m_axis_triple)
                else:
                    self.vector.softmax_rest(
                        qk_view, self._scale, m_axis_triple, tile_triple
                    )
        else:
            if is_first:
                self.vector.softmax_first(qk_view, self._scale, m_axis_triple)
            else:
                self.vector.softmax_rest(qk_view, self._scale, m_axis_triple, tile_triple)
        self.vector.store_p(
            self.p_l1, _rows, valid_n
        )

    @jit
    def _stage_pv(self, tile_idx: int, n_idx: int):
        """P @ V -> pv_ub."""
        batch_idx, head_idx, m_idx, tok0, _rows, _n_end = self._tile_coords(
            tile_idx
        )

        kv_head = head_idx // self._head_group_num

        if const_expr(self.kv_layout in ("PA_BNBD", "PA_BBND", "PA_NZ")):
            v_slot = self._load_pa_kv_tile(self._value, batch_idx, kv_head, n_idx, True)
            valid_n = min(self.tile_n, self._kv_length(batch_idx) - n_idx * self.tile_n)
        else:
            v_tile_gm = self._kv_tile(self._value, batch_idx, kv_head, n_idx)
            v_slot = self.matmul.load_v(v_tile_gm)
            valid_n = v_tile_gm.shape[0]
        # The contraction excludes padded or neighboring-batch KV rows.
        p_l1_view = reinterpret(
            self.p_l1.consume(),
            shape=(self.tile_cube_m, valid_n),
        )
        if const_expr(self.tile_d > CUBE_D_TILE):
            if const_expr(self._pa_strided and self.tile_cube_m > 64):
                self.matmul.compute_pv_chunked(
                    p_l1_view, self.pv_ubs, _rows, v_slot, valid_n,
                )
            else:
                self.matmul.compute_pv_chunked(
                    p_l1_view, self.pv_ubs, _rows, v_slot,
                )
        else:
            result = self.matmul.compute_pv(p_l1_view, _rows, v_slot)
            self.matmul.store_o(self.pv_ub, _rows, result)

    @jit
    def _stage_update(self, tick: int, tile_idx: int, n_idx: int, m_seq: int,
                      n_start: int, n_end: int, slot: int):
        batch, head, m_idx, token, rows, _ = self._tile_coords(tile_idx)
        first = n_idx == n_start
        last = n_idx == n_end - 1
        state = m_seq % self.preload_num
        triple = (tick - 3) % 3
        softmax_sum = self.vector.softmax_sum_bufs[state]
        active_rows = (ceil_div(rows, 2) if self.subblock_idx == 0 else rows // 2)
        if const_expr(self.tile_d > CUBE_D_TILE):
            for d_idx in tuple(range(self.tile_d // CUBE_D_TILE)):
                pv = reinterpret(
                    self.pv_ubs[d_idx].consume(),
                    shape=(active_rows, CUBE_D_TILE),
                    stride=(CUBE_D_TILE, 1),
                )
                if first and last:
                    self.vector.init_o_last_chunk(pv, softmax_sum, d_idx * CUBE_D_TILE)
                elif first:
                    self.vector.init_o_chunk(pv, d_idx * CUBE_D_TILE)
                elif last:
                    self.vector.update_o_last_chunk(
                        pv, self.vector.softmax_exp_bufs[triple],
                        softmax_sum, d_idx * CUBE_D_TILE,
                    )
                else:
                    self.vector.update_o_chunk(
                        pv, self.vector.softmax_exp_bufs[triple],
                        d_idx * CUBE_D_TILE,
                    )
        else:
            pv = reinterpret(
                self.pv_ub.consume(),
                shape=(active_rows, self.tile_d),
                stride=(self.tile_d, 1),
            )
            if first and last:
                self.vector.init_o_last(pv, softmax_sum)
            elif first:
                self.vector.init_o(pv)
            elif last:
                self.vector.update_o_last(
                    pv, self.vector.softmax_exp_bufs[triple], softmax_sum
                )
            else:
                self.vector.update_o(pv, self.vector.softmax_exp_bufs[triple])
        if last:
            if slot >= 0:
                self._emit_partial(
                    slot, self.vector.softmax_max_bufs[state], softmax_sum, rows
                )
            else:
                sub_lo = (0 if self.subblock_idx == 0 else ceil_div(rows, 2))
                _cast_nd_rows(
                    reinterpret(
                        self.vector.o_ub,
                        shape=(self.tile_vec_m, self.tile_d),
                        stride=(self.tile_d, 1),
                    ),
                    self.vector.res_o,
                    active_rows,
                    self.tile_d,
                )
                source = self.vector.o_ub
                _store_o(
                    self._attn_out, source, self.layout, batch,
                    head // self._head_group_num, token + sub_lo, active_rows,
                    self._head_group_num, self._q_length(batch),
                    self._cu_seqlens_q, self.s1g_mapping)
                if const_expr(self.return_softmax_lse):
                    self.external_lse.store_merged(
                        self.vector.softmax_max_bufs[state],
                        softmax_sum, self._softmax_lse_gm,
                        self.layout, batch, head // self._head_group_num,
                        token + sub_lo, active_rows, self._head_group_num,
                        self._q_length(batch), self._seqlen_q, self._head_num_q,
                        self._cu_seqlens_q, self.s1g_mapping)

    @jit
    def _stage_emit_lse(self, w, rd):
        """emit_lse_tail[0:w] <- lse_ub[rd : rd+w], one_i32 32B row each.

        The value lands in lane 0 of each 8-lane row; the other lanes are
        garbage the combine never reads (ASC's FP32_BLOCK_ELEMENT_NUM-wide
        LSE layout)."""
        with vf(mode="simd"):
            m8, _ = rr.update_mask(LSE_ROW_LANES, elem_bits=32)
            for row in range(w):
                # vload_broadcast reads one_i32 element and broadcasts it: a plain
                # vload would pull VL lanes and run past lse_ub's (64, 1)
                # extent on the high rows.
                val = rr.vload_broadcast(self.lse_ub, rd + row)
                rr.vstore(self.emit_lse_tail, row * LSE_ROW_LANES, val, m8)

    @jit
    def _emit_partial(self, slot: int, softmax_max_buf, softmax_sum_buf, rows: int):
        """Store one_i32 head's partial result using the actual balanced row split."""
        half0 = (rows + 1) // 2
        half1 = rows // 2
        sub_cnt = (half0 if self.subblock_idx == 0 else half1)
        sub_lo = (0 if self.subblock_idx == 0 else half0)
        with vf(mode="simd"):
            m, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            mx = rr.vload(softmax_max_buf, 0)
            softmax_sum = rr.vload(softmax_sum_buf, 0)
            is_zero = rr.veqs(softmax_sum, 0.0, mask=m)
            one_b = rr.vdups(1.0, dtypes.float32, mask=m)
            safe_sum = rr.vselect(one_b, softmax_sum, cond_mask=is_zero)
            lse = rr.vadd(mx, rr.vlog(safe_sum, mask=m), mask=m)
            min_b = rr.vdups(LSE_MIN, dtypes.float32, mask=m)
            lse = rr.vselect(min_b, lse, cond_mask=is_zero)
            rr.vstore(self.lse_ub, 0, lse, m)

        _cast_nd_rows(self.vector.o_ub, self.vector.res_o, sub_cnt, self.tile_d)

        self._stage_emit_lse(self.tile_vec_m, 0)
        row_base = slot * self.tile_cube_m + sub_lo
        if sub_cnt > 0:
            mem_copy(
                self._o_ws[row_base:row_base + sub_cnt, None],
                self.vector.o_ub,
            )
            mem_copy(
                self._lse_ws[row_base:row_base + sub_cnt, None],
                self.emit_lse_tail,
            )

    @jit
    def _q_length(self, batch: int):
        return _sequence_length(
            self._cu_seqlens_q, self._seqused_q, batch, self._seqlen_q,
            self.layout == "TND",
        )

    @jit
    def _kv_length(self, batch: int):
        return _sequence_length(
            self._cu_seqlens_kv, self._seqused_kv, batch, self._seqlen_k,
            self.kv_layout == "TND",
        )

    @jit
    def _causal_offset(self, batch_idx: int):
        return self._kv_length(batch_idx) - self._q_length(batch_idx)

    @jit
    def _sparse_kv_range(self, batch: int, m: int, s2_base: int):
        s2_base = dtypes.int64(s2_base)
        q_len = dtypes.int64(self._q_length(batch))
        kv_len = dtypes.int64(self._kv_length(batch))
        zero_i64 = dtypes.int64(0)
        one_i64 = dtypes.int64(1)
        first = zero_i64
        end = ceil_div(kv_len, s2_base)
        if const_expr(self.mask_mode != 0):
            begin = dtypes.int64(m * self._m_base)
            last = min(begin + self._m_base, q_len * self._head_group_num) - one_i64
            q_first = zero_i64
            q_last = q_len
            if const_expr(self.s1g_mapping):
                q_first = begin // self._head_group_num
                q_last = last // self._head_group_num
            else:
                if q_len > zero_i64:
                    if begin // q_len == last // q_len:
                        q_first = begin % q_len
                        q_last = last % q_len
            k_first = zero_i64
            k_last = q_last + kv_len - q_len
            if const_expr(self.mask_mode == 4):
                k_first = (
                    zero_i64 if self._win_left < 0
                    else q_first + kv_len - q_len - self._win_left
                )
                k_last = (kv_len - one_i64 if self._win_right < 0 else k_last + self._win_right)
            first = max(k_first, zero_i64) // s2_base
            end = (min(k_last, kv_len - one_i64) // s2_base) + one_i64
            if q_len == zero_i64 or kv_len == zero_i64 or k_first >= kv_len or k_last < zero_i64 or k_last < k_first:
                first = zero_i64
                end = zero_i64
        return first, end

    @jit
    def _query_tile(self, rows: int, q_l1=None):
        source = self.matmul.q_l1[0] if q_l1 is None else q_l1
        return reinterpret(source, shape=(rows, self.tile_d))

    @jit
    def _load_pa_strided_d256_tile(
        self, gm, slot, batch_idx: int, kv_head: int, start, valid_n,
    ):
        """Build and copy each 16-row NZ strip before reusing UB scratch."""
        half_d = self.tile_d // 2
        d_start = self.subblock_idx * half_d
        for segment in range(self.tile_n // 16):
            copied = dtypes.int64(segment * 16)
            # Pad the final strip before copying it to the L1 destination.
            with vf(mode="simd"):
                zero_mask, _ = rr.update_mask(16, elem_bits=16)
                zero_reg = rr.vdups(0, self.dtype_16, mask=zero_mask)
                for di in tuple(range(self.tile_d // 32)):
                    for row in range(16):
                        rr.vstore(
                            self.pa_nz_ub, (di * 16 + row) * 16,
                            zero_reg, zero_mask,
                        )
            if copied < valid_n:
                token = start + copied
                page_row = token % self._pa_block_size
                take = min(16, valid_n - copied)
                page = dtypes.int64(
                    self._block_table[batch_idx, token // self._pa_block_size]
                )
                if const_expr(self.kv_layout == "PA_NZ"):
                    for di in tuple(range(self.tile_d // 32)):
                        global_di = self.subblock_idx * (self.tile_d // 32) + di
                        source = gm[
                            page, kv_head, global_di,
                            page_row:page_row + take, None,
                        ]
                        source = tile_slice(source, (16, 16), (0, 0))
                        dest = tile_slice(self.pa_nd_ub, (16, 16), (0, di))
                        mem_copy(dest, source)
                else:
                    if const_expr(self.kv_layout == "PA_BBND"):
                        page_cache = permute(gm, (0, 2, 1, 3))
                    else:
                        page_cache = gm
                    source = page_cache[
                        page, kv_head, page_row:page_row + take,
                        d_start:d_start + half_d,
                    ]
                    source = tile_slice(source, (16, half_d), (0, 0))
                    mem_copy(self.pa_nd_ub, source)
                with vf(mode="simd"):
                    load_mask, _ = rr.update_mask(16, elem_bits=16)
                    for di in tuple(range(self.tile_d // 32)):
                        for row in range(take):
                            value = rr.vload(
                                self.pa_nd_ub, row * half_d + di * 16,
                            )
                            rr.vstore(
                                self.pa_nz_ub, (di * 16 + row) * 16,
                                value, load_mask,
                            )
            nz_half = reinterpret(
                self.pa_nz_ub, shape=(16, half_d), data_format="nz",
            )
            l1_strip = tile_slice(
                slot, (16, self.tile_d), (segment, 0),
            )
            mem_copy(
                l1_strip, nz_half, engine=self.pa_ub_to_l1,
                part_id=self.subblock_idx,
            )
        return slot

    @jit
    def _load_pa_strided_tile(self, gm, batch_idx: int, kv_head: int, n_idx: int,
                              is_value=False):
        """Tile-only AIV staging for GM views with a non-unit inner stride."""
        channel = self.matmul.v_l1 if const_expr(is_value) else self.matmul.k_l1
        slot = channel.produce()
        start = dtypes.int64(n_idx * self.tile_n)
        valid_n = min(self.tile_n, self._kv_length(batch_idx) - start)
        if const_expr(self.tile_d > CUBE_D_TILE and self.tile_cube_m > 64):
            return self._load_pa_strided_d256_tile(
                gm, slot, batch_idx, kv_head, start, valid_n,
            )
        half_d = self.tile_d // 2
        d_start = self.subblock_idx * half_d
        # Both AIVs build disjoint D halves, including zero padding.
        with vf(mode="simd"):
            pa_zero_mask, _ = rr.update_mask(16, elem_bits=16)
            pa_zero_reg = rr.vdups(0, self.dtype_16, mask=pa_zero_mask)
            for di in tuple(range(self.tile_d // 32)):
                for row in range(self.tile_n):
                    rr.vstore(self.pa_nz_ub, (di * self.tile_n + row) * 16,
                              pa_zero_reg, pa_zero_mask)
        for segment in range(ceil_div(valid_n, 16)):
            copied = dtypes.int64(segment * 16)
            token = start + copied
            page_row = token % self._pa_block_size
            take = min(16, valid_n - copied)
            page = dtypes.int64(self._block_table[batch_idx, token // self._pa_block_size])
            if const_expr(self.kv_layout == "PA_NZ"):
                for di in tuple(range(self.tile_d // 32)):
                    global_di = self.subblock_idx * (self.tile_d // 32) + di
                    source = gm[page, kv_head, global_di, page_row:page_row + take, None]
                    source = tile_slice(source, (16, 16), (0, 0))
                    dest = tile_slice(self.pa_nd_ub, (16, 16), (0, di))
                    mem_copy(dest, source)
            else:
                if const_expr(self.kv_layout == "PA_BBND"):
                    page_cache = permute(gm, (0, 2, 1, 3))
                else:
                    page_cache = gm
                source = page_cache[page, kv_head, page_row:page_row + take,
                                    d_start:d_start + half_d]
                source = tile_slice(source, (16, half_d), (0, 0))
                mem_copy(self.pa_nd_ub, source)
            with vf(mode="simd"):
                pa_load_mask, _ = rr.update_mask(16, elem_bits=16)
                for di in tuple(range(self.tile_d // 32)):
                    for row in range(take):
                        value = rr.vload(self.pa_nd_ub, row * half_d + di * 16)
                        rr.vstore(self.pa_nz_ub,
                                  (di * self.tile_n + copied + row) * 16,
                                  value, pa_load_mask)
        nz_half = reinterpret(self.pa_nz_ub, shape=(self.tile_n, half_d),
                              data_format="nz")
        mem_copy(slot, nz_half, engine=self.pa_ub_to_l1, part_id=self.subblock_idx)
        return slot

    @jit
    def _load_pa_kv_tile(self, gm, batch_idx: int, kv_head: int, n_idx: int,
                         is_value=False):
        """Gather only this logical tile into one L1 slot, directly from pages."""
        if const_expr(self._pa_strided):
            return self._load_pa_strided_tile(gm, batch_idx, kv_head, n_idx, is_value)
        channel = self.matmul.v_l1 if const_expr(is_value) else self.matmul.k_l1
        slot = channel.produce()
        start = dtypes.int64(n_idx * self.tile_n)
        valid_n = min(self.tile_n, self._kv_length(batch_idx) - start)
        if valid_n < self.tile_n:
            # MMAD and L0 loads operate on aligned blocks. Never expose stale
            # V padding (including NaNs from a previously used slot).
            mem_copy(slot, self._pa_zero, engine=self.matmul.nd2nz)
            if const_expr(self.kv_layout == "PA_NZ"):
                # ND2NZ clear and raw NZ page DMA overlap the same L1 slot.
                # Finish the clear before page data overwrites its valid rows.
                cube_sync_pipe(PIPE.MTE2)
        if const_expr(self.kv_layout == "PA_NZ"):
            copied = dtypes.int64(0)
            for _segment in range(ceil_div(start % self._pa_block_size + valid_n, self._pa_block_size)):
                token = start + copied
                page_row = token % self._pa_block_size
                take = min(valid_n - copied, self._pa_block_size - page_row)
                page = dtypes.int64(self._block_table[batch_idx, token // self._pa_block_size])
                d0 = 16
                raw_slot = reinterpret(
                    slot, shape=(self.tile_d // d0, self.tile_n * d0),
                    data_format="nd",
                )
                if const_expr(gm.stride[3] == d0 and gm.stride[4] == 1):
                    page_view = gm[page, kv_head, None, None, None].view(
                        (self.tile_d // d0, self._pa_block_size * d0)
                    )
                    source = page_view[None, page_row * d0:(page_row + take) * d0]
                    dest = raw_slot[None, copied * d0:(copied + take) * d0]
                    mem_copy(dest, source)
                else:
                    # Preserve page-row padding in a non-contiguous cache.
                    strips = raw_slot.view((self.tile_d // d0, self.tile_n, d0))
                    for di in tuple(range(self.tile_d // d0)):
                        source = gm[page, kv_head, di, page_row:page_row + take, None]
                        source = tile_slice(source, (self.tile_n, d0), (0, 0))
                        dest = strips[di, copied:copied + take, None]
                        dest = tile_slice(dest, (self.tile_n, d0), (0, 0))
                        mem_copy(dest, source)
                copied += take
        else:
            # Each DMA covers the largest logical span within one page.
            # A singleton batch plus stack_m retains the parent NZ row pitch.
            copied = dtypes.int64(0)
            for _segment in range(ceil_div(start % self._pa_block_size + valid_n, self._pa_block_size)):
                token = start + copied
                page_row = token % self._pa_block_size
                take = min(valid_n - copied, self._pa_block_size - page_row)
                page = dtypes.int64(self._block_table[batch_idx, token // self._pa_block_size])
                if const_expr(self.kv_layout == "PA_BNBD"):
                    source = gm[page:page + 1, kv_head, page_row:page_row + take, None]
                else:
                    source = permute(gm, (0, 2, 1, 3))[page:page + 1, kv_head, page_row:page_row + take, None]
                source = tile_slice(
                    source, (1, self.tile_n, self.tile_d), (0, 0, 0)
                )
                # copied is 16-aligned; each row advances one 32-byte C0
                # block. Only take rows are written, and copied + take is
                # bounded by valid_n, so the fixed-pitch alias never writes
                # beyond this slot.
                dest = reinterpret(slot, shape=(self.tile_n, self.tile_d),
                                   offset=dtypes.int64(copied * 32))
                mem_copy(dest, source, engine=self.pa_page_nd2nz)
                copied += take
        return slot

    @jit
    def _kv_tile(self, gm, batch_idx: int, kv_head: int,
                 n_idx: int):
        if const_expr(self.kv_layout != "TND" and self._seqused_kv is None):
            return tile_slice(gm[batch_idx, kv_head, None, None],
                             (self.tile_n, self.tile_d), (n_idx, 0))
        token = n_idx * self.tile_n
        rows = min(self.tile_n, self._kv_length(batch_idx) - token)
        if const_expr(self.kv_layout == "TND"):
            token = token + self._cu_seqlens_kv[batch_idx]
        return _row_tile(
            gm, self.kv_layout, batch_idx, kv_head, token, rows,
            self.tile_n, self.tile_d,
        )

    @jit
    def _tile_coords(self, tile_idx: int):
        batch_idx, kv_head, m_idx, start = _merged_tile_coords(
            tile_idx, self._max_m_rows, self._head_num_kv, self._m_base,
        )
        rows = min(
            self._m_base,
            self._q_length(batch_idx) * self._head_group_num - start,
        )
        return (
            batch_idx,
            kv_head * self._head_group_num,
            m_idx,
            start,
            rows,
            ceil_div(self._kv_length(batch_idx), self.tile_n),
        )

    @jit
    def _copy_init_span(
        self, gm, base: int, count: int, aiv_index: int, aiv_count: int,
        capacity: int, source,
    ):
        """Copy one disjoint AIV partition of a contiguous output span."""
        start, end = _partition_init_range(count, aiv_index, aiv_count, 16)
        for offset in range(0, end - start, capacity):
            width = min(capacity, end - start - offset)
            mem_copy(
                gm[base + start + offset:base + start + offset + width, None],
                reinterpret(source, shape=(width, 1), stride=(1, 1)),
            )

    @jit
    def _initialize_output_tokens(
        self, out_init_gm, batch: int, begin: int, end: int,
        aiv_index: int, aiv_count: int, out_zero, fp16_capacity: int,
    ):
        """Zero one invalid token interval in the public output layout."""
        if end > begin:
            token_count = end - begin
            if const_expr(self.output_layout == "TND"):
                token_base = self._cu_seqlens_q[batch] + begin
                self._copy_init_span(
                    out_init_gm, token_base * self._head_num_q * self.tile_d,
                    token_count * self._head_num_q * self.tile_d,
                    aiv_index, aiv_count, fp16_capacity, out_zero,
                )
            elif const_expr(self.output_layout == "BSND"):
                self._copy_init_span(
                    out_init_gm,
                    (batch * self._seqlen_q + begin) * self._head_num_q * self.tile_d,
                    token_count * self._head_num_q * self.tile_d,
                    aiv_index, aiv_count, fp16_capacity, out_zero,
                )
            else:
                for head in range(self._head_num_q):
                    self._copy_init_span(
                        out_init_gm,
                        ((batch * self._head_num_q + head) * self._seqlen_q + begin)
                        * self.tile_d,
                        token_count * self.tile_d,
                        aiv_index, aiv_count, fp16_capacity, out_zero,
                    )

    @jit
    def _initialize_lse_tokens(
        self, softmax_lse_gm, batch: int, begin: int, end: int,
        aiv_index: int, aiv_count: int, fp32_capacity: int,
    ):
        """Set one invalid token interval to +inf for every Q head."""
        if end > begin:
            token_count = end - begin
            for head in range(self._head_num_q):
                if const_expr(self.output_layout == "TND"):
                    base = (
                        head * self._seqlen_q
                        + self._cu_seqlens_q[batch]
                        + begin
                    )
                else:
                    base = (
                        (batch * self._head_num_q + head) * self._seqlen_q + begin
                    )
                self._copy_init_span(
                    softmax_lse_gm, base, token_count, aiv_index, aiv_count,
                    fp32_capacity, self.vector.res_o,
                )

    @jit
    def _initialize_invalid_outputs(self, out_init_gm, softmax_lse_gm):
        """Initialize only output rows that cannot be produced by attention."""
        aiv_index = get_block_idx() * get_subblock_dim() + get_subblock_id()
        aiv_count = get_block_num() * get_subblock_dim()
        fp32_capacity = self.tile_vec_m * self.tile_d
        fp16_capacity = fp32_capacity * 2
        with vf(mode="simd"):
            for offset in range(0, fp32_capacity, VL_T):
                mask, _ = rr.update_mask(
                    min(VL_T, fp32_capacity - offset), elem_bits=32
                )
                zeros_reg = rr.vdups(0.0, dtypes.float32, mask=mask)
                rr.vstore(self.vector.res_o, offset, zeros_reg, mask=mask)
        vec_sync_all()
        out_zero = reinterpret(
            self.vector.res_o, self.dtype_16, (fp16_capacity, 1)
        )

        for batch in range(self._batch_size):
            used_q = dtypes.int64(self._q_length(batch))
            if const_expr(self.layout == "TND"):
                physical_q = (self._cu_seqlens_q[batch + 1]
                              - self._cu_seqlens_q[batch])
            else:
                physical_q = self._seqlen_q
            used_kv = dtypes.int64(self._kv_length(batch))
            prefix_end = used_q - used_q
            if used_kv == 0:
                prefix_end = used_q
            if const_expr(self.mask_mode == 3):
                prefix_end = max(prefix_end, _clamp_nonneg(used_q - used_kv))
            elif const_expr(self.mask_mode == 4):
                window_prefix = _clamp_nonneg(used_q - used_kv - self._win_right)
                prefix_end = (
                    max(prefix_end, window_prefix)
                    if self._win_right >= 0 else prefix_end
                )
            prefix_end = min(used_q, prefix_end)
            self._initialize_output_tokens(
                out_init_gm, batch, 0, prefix_end,
                aiv_index, aiv_count, out_zero, fp16_capacity,
            )
            self._initialize_output_tokens(
                out_init_gm, batch, used_q, physical_q,
                aiv_index, aiv_count, out_zero, fp16_capacity,
            )

        if const_expr(self.return_softmax_lse):
            # An invalid span contains at most _seqlen_q tokens (the total
            # physical Q length for TND). Each DMA reads the same UB prefix.
            lse_fill_count = _init_lse_fill_count(
                self._seqlen_q, aiv_count, fp32_capacity
            )
            with vf(mode="simd"):
                full_mask, _ = rr.update_mask(VL_T, elem_bits=32)
                ones_reg = rr.vdups(1.0, dtypes.float32, mask=full_mask)
                zeros_reg = rr.vdups(0.0, dtypes.float32, mask=full_mask)
                inf_reg = rr.vdiv(ones_reg, zeros_reg, mask=full_mask)
                for offset in range(0, lse_fill_count, VL_T):
                    mask, _ = rr.update_mask(
                        min(VL_T, lse_fill_count - offset), elem_bits=32
                    )
                    rr.vstore(self.vector.res_o, offset, inf_reg, mask=mask)
            vec_sync_all()
            for batch in range(self._batch_size):
                used_q = dtypes.int64(self._q_length(batch))
                if const_expr(self.layout == "TND"):
                    physical_q = (self._cu_seqlens_q[batch + 1]
                                  - self._cu_seqlens_q[batch])
                else:
                    physical_q = self._seqlen_q
                used_kv = dtypes.int64(self._kv_length(batch))
                prefix_end = used_q - used_q
                if used_kv == 0:
                    prefix_end = used_q
                if const_expr(self.mask_mode == 3):
                    prefix_end = max(prefix_end, _clamp_nonneg(used_q - used_kv))
                elif const_expr(self.mask_mode == 4):
                    window_prefix = _clamp_nonneg(
                        used_q - used_kv - self._win_right
                    )
                    prefix_end = (
                        max(prefix_end, window_prefix)
                        if self._win_right >= 0 else prefix_end
                    )
                prefix_end = min(used_q, prefix_end)
                self._initialize_lse_tokens(
                    softmax_lse_gm, batch, 0, prefix_end,
                    aiv_index, aiv_count, fp32_capacity,
                )
                self._initialize_lse_tokens(
                    softmax_lse_gm, batch, used_q, physical_q,
                    aiv_index, aiv_count, fp32_capacity,
                )

    def __call__(
        self,
        attn_out: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: float,
        block_table: Tensor,
        cu_seqlens_q: Tensor,
        cu_seqlens_kv: Tensor,
        seqused_q: Tensor,
        seqused_kv: Tensor,
        sinks: Tensor,
        win_left: int,
        win_right: int,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        softmax_lse_gm: Tensor = None,
        attn_mask: Tensor = None,
        metadata: Tensor = None,
        o_ws: Tensor = None,
        lse_ws: Tensor = None,
        out_init_gm: Tensor = None,
        need_init_output: int = 0,
        pa_zero: Tensor | None = None,
    ):
        """Merged FA kernel, channel-first, typed channels.

        Under split-KV this is TWO phases in ONE launch: a compute phase that
        writes each split task's partial O/LSE to ``o_ws``/``lse_ws``, an
        all-core barrier, then a fused combine phase that merges each output
        tile's slots via the ASC binary ``metadata`` FD descriptors.
        """
        if const_expr(self.layout == "TND"):
            # Packed tensor extents bound coordinate encoding, not per-batch lengths.
            head_num_q = query.shape[0]
            head_num_kv = (
                key.shape[0] if self.kv_layout == "TND" else key.shape[1]
            )
            batch_size = self._batch_count
            seqlen_q = query.shape[1]
            seqlen_k = key.shape[1] if self.kv_layout == "TND" else key.shape[2]
        else:
            batch_size, head_num_q, seqlen_q = query.shape[:3]
            head_num_kv, seqlen_k = key.shape[1], key.shape[2]
        if const_expr(self.kv_layout in ("PA_BNBD", "PA_BBND", "PA_NZ")):
            if const_expr(self.kv_layout == "PA_BBND"):
                head_num_kv = key.shape[2]
                self._pa_block_size = key.shape[1]
            elif const_expr(self.kv_layout == "PA_BNBD"):
                head_num_kv = key.shape[1]
                self._pa_block_size = key.shape[2]
            else:
                head_num_kv = key.shape[1]
                self._pa_block_size = key.shape[3]
            seqlen_k = block_table.shape[1] * self._pa_block_size
        head_group_num = head_num_q // head_num_kv

        self._batch_size = batch_size
        self._head_num_q = head_num_q
        self._head_num_kv = head_num_kv
        self._m_base = dtypes.int64(metadata[HEAD_M_BASE_SIZE_INDEX])
        max_q = (max_seqlen_q if max_seqlen_q > 0 else seqlen_q)
        self._max_m_rows = max(1, ceil_div(max_q * head_group_num, self._m_base))
        self._head_group_num = head_group_num
        self._seqlen_q = seqlen_q
        self._seqlen_k = seqlen_k
        self._scale = scale
        self._query = query
        self._key = key
        self._value = value
        self._attn_out = attn_out
        self._attn_mask = attn_mask
        self._block_table = block_table
        self._pa_zero = pa_zero
        self._cu_seqlens_q = cu_seqlens_q
        self._cu_seqlens_kv = cu_seqlens_kv
        self._seqused_q = seqused_q
        self._seqused_kv = seqused_kv
        self._win_left = win_left
        self._win_right = win_right
        self._softmax_lse_gm = softmax_lse_gm
        self._o_ws = o_ws
        self._lse_ws = lse_ws
        # All layouts use the same merged (bn2, m, s2) task model.  Layout
        # affects only row-to-address mapping in the direct Q loader.
        self._merged_window_mask = _use_merged_window_mask(
            True, self.s1g_mapping, self.mask_mode
        )
        self._template_window_mask = self._merged_window_mask
        self.vector = Vector(
            self.tile_vec_m,
            self.tile_n,
            self.tile_d,
            self.subblock_idx,
            self.mask_mode,
            self.dtype_16,
            mask_in_vector=True,
            merged_window_mask=self._merged_window_mask,
            template_window_mask=self._template_window_mask,
        )
        if need_init_output != 0:
            self._initialize_invalid_outputs(out_init_gm, softmax_lse_gm)
        if const_expr(seqlen_k == 0):
            return
        # Pad each NZ column group by one 16x16 block so segmented DMA
        # keeps an explicit full-buffer pitch, including dynamic row windows.
        if const_expr(self.tile_d <= CUBE_D_TILE):
            self.matmul.q_l1 = (
                Buffer(MemLoc.L1, (self.tile_cube_m, self.tile_d), self.dtype_16, n1_pad=256),
                Buffer(MemLoc.L1, (self.tile_cube_m, self.tile_d), self.dtype_16, n1_pad=256),
            )
        self._run_native(metadata)

    @jit
    def _run_native(self, metadata: Tensor):
        section_num = metadata[HEAD_SECTION_NUM_INDEX]
        s2_base = metadata[HEAD_S2_BASE_SIZE_INDEX]
        for section in range(section_num):
            base = HEAD_METADATA_STRIDE + (
                section * ASC_AIC_CORE_NUM + self.block_idx
            ) * FA_METADATA_STRIDE
            bn0 = metadata[base + FA_BN2_START_INDEX]
            m0 = metadata[base + FA_M_START_INDEX]
            n0 = metadata[base + FA_S2_START_INDEX]
            bn1 = metadata[base + FA_BN2_END_INDEX]
            m1 = metadata[base + FA_M_END_INDEX]
            n1 = metadata[base + FA_S2_END_INDEX]
            first_slot = metadata[
                base + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX
            ]
            delay_line = DelayLineGroup(
                4, "tile", "n", "m_seq", "nstart", "nend", "slot"
            )
            tick = 0
            issued = 0
            m_seq = 0
            partials = 0
            # Metadata owns one lexicographic (bn2, m, s2) range.  Each
            # iteration below creates at most one merged GxS1 task; Q-head is
            # only an address coordinate inside _load_merged_q.
            if (bn0 < bn1) or (m0 < m1) or (n0 < n1):
                bn2_limit = min(
                    bn1 + 1, self._batch_size * self._head_num_kv
                )
                for bn2 in range(bn0, bn2_limit):
                    batch = bn2 // self._head_num_kv
                    length = self._q_length(batch)
                    kv_length = self._kv_length(batch)
                    valid_m_count = ceil_div(
                        length * self._head_group_num, self._m_base
                    )
                    m_begin, m_end = _metadata_task_m_bounds(
                        bn2, bn0, m0, bn1, m1, n1, valid_m_count,
                    )
                    for m in range(m_begin, m_end):
                        tile_idx = bn2 * self._max_m_rows + m
                        batch_idx, head_idx, _m_idx, start, rows, full = (
                            self._tile_coords(tile_idx)
                        )
                        sparse_start, sparse_end = self._sparse_kv_range(
                            batch_idx, m, s2_base
                        )
                        is_first = (bn2 == bn0) and (m == m0)
                        is_last = (bn2 == bn1) and (m == m1)
                        n_start = (
                            max(sparse_start, n0) if is_first else sparse_start
                        ) * s2_base // self.tile_n
                        n_end = min(
                            full,
                            (
                                min(sparse_end, n1) if is_last else sparse_end
                            ) * s2_base // self.tile_n,
                        )
                        partial = (
                            1 if (bn0 == bn1 and m0 == m1)
                            or (is_first and n0 > sparse_start)
                            or (is_last and n1 > 0) else 0
                        )
                        native_slot = first_slot + partials
                        active = (
                            length > 0 and kv_length > 0 and rows > 0
                            and n_end > n_start
                        )
                        if active:
                            partials = partials + partial
                            slot = (native_slot if partial == 1 else -1)
                            m_seq = m_seq + 1
                            kv_head = head_idx // self._head_group_num
                            if m_seq > 2:
                                cube_sync_pipe(PIPE.MTE1)
                            if m_seq % 2:
                                self._load_merged_q(
                                    batch_idx, kv_head, start, rows, self.matmul.q_l1[0]
                                )
                            else:
                                self._load_merged_q(
                                    batch_idx, kv_head, start, rows, self.matmul.q_l1[1]
                                )
                            # Q is produced by AIC MTE2 and consumed by AIC
                            # MTE1.  Keep the dependency local to the cube core.
                            cube_sync_pipe(PIPE.MTE2)
                            q_tile = self._query_tile(rows)
                            for n_idx in range(n_start, n_end):
                                delay_line.push(
                                    tile=tile_idx,
                                    n=n_idx,
                                    m_seq=m_seq,
                                    nstart=n_start,
                                    nend=n_end,
                                    slot=slot,
                                )
                                if m_seq % 2:
                                    self._stage_qk(
                                        n_idx, batch_idx, kv_head,
                                        q_tile, self.matmul.q_l1[0],
                                    )
                                else:
                                    self._stage_qk(
                                        n_idx, batch_idx, kv_head,
                                        q_tile, self.matmul.q_l1[1],
                                    )
                                issued = issued + 1
                                if tick >= 1 and tick - 1 < issued:
                                    self._stage_softmax(
                                        tick,
                                        delay_line.tile.tap(1),
                                        delay_line.n.tap(1),
                                        delay_line.m_seq.tap(1),
                                        delay_line.nstart.tap(1),
                                    )
                                if tick >= 2 and tick - 2 < issued:
                                    self._stage_pv(
                                        delay_line.tile.tap(2), delay_line.n.tap(2)
                                    )
                                if tick >= 3 and tick - 3 < issued:
                                    self._stage_update(
                                        tick,
                                        delay_line.tile.tap(3),
                                        delay_line.n.tap(3),
                                        delay_line.m_seq.tap(3),
                                        delay_line.nstart.tap(3),
                                        delay_line.nend.tap(3),
                                        delay_line.slot.tap(3),
                                    )
                                delay_line.advance()
                                tick += 1
                            # The following Q block writes the other L1 slot.
                        m = m + 1
                for _ in range(self.preload_num):
                    if tick >= 1 and tick - 1 < issued:
                        self._stage_softmax(
                            tick,
                            delay_line.tile.tap(1),
                            delay_line.n.tap(1),
                            delay_line.m_seq.tap(1),
                            delay_line.nstart.tap(1),
                        )
                    if tick >= 2 and tick - 2 < issued:
                        self._stage_pv(delay_line.tile.tap(2), delay_line.n.tap(2))
                    if tick >= 3 and tick - 3 < issued:
                        self._stage_update(
                            tick,
                            delay_line.tile.tap(3),
                            delay_line.n.tap(3),
                            delay_line.m_seq.tap(3),
                            delay_line.nstart.tap(3),
                            delay_line.nend.tap(3),
                            delay_line.slot.tap(3),
                        )
                    delay_line.advance()
                    tick += 1

            cube_sync_pipe(PIPE.MTE1)
            if metadata[HEAD_IS_FD_INDEX] > 0:
                _stage_boundary_reset(_COMBINE_FLAG_BASE)
                combine = _CombineHelper(
                    self.tile_cube_m, self.tile_vec_m, self.tile_d,
                    self.subblock_idx, self.dtype_16, self.return_softmax_lse,
                )
                combine.run_asc(
                    self._attn_out, self._o_ws, self._lse_ws, metadata,
                    self.block_idx, section, self._head_num_q, self._head_num_kv,
                    self._seqlen_q, self._cu_seqlens_q, self._seqused_q,
                    self.layout, self.s1g_mapping, self._softmax_lse_gm)
                # The next section reuses both the partial workspace and UB.
                if section + 1 < section_num:
                    _stage_boundary_reset(_COMBINE_FLAG_BASE)


class _CombineHelper:
    """Merge native FD ranges using the same head subtiles as the compute phase."""

    def __init__(self, tile_cube_m, tile_vec_m, tile_d, subblock_idx,
                 dtype_16=dtypes.float16, return_softmax_lse=False):
        self.return_softmax_lse = return_softmax_lse
        if return_softmax_lse:
            self.external_lse = _ExternalLse(tile_vec_m)
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_vec_m
        self.tile_d = tile_d
        self.subblock_idx = subblock_idx

        # Vector is reused only for res_o (the fp32 accumulator).
        self.vector = Vector(
            tile_vec_m, tile_d, tile_d, self.subblock_idx, 0, dtype_16,
        )

        self.o_in = Channel(
            MemLoc.UB, shape=(tile_vec_m, tile_d), dtype=dtype_16, depth=2
        )
        # ASC LSE layout: 8 f32 lanes per row (32B DMA rows). Keep this in a
        # channel so the first GM->UB load is ordered before the raw-vector
        # lane-0 reads in _merge_slot.
        self.lse_in = Channel(
            MemLoc.UB, shape=(tile_vec_m, LSE_ROW_LANES), dtype=dtypes.float32, depth=2
        )
        self.run_max = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.run_den = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.old_scale_buf = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.slot_scale_buf = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)

    @jit
    def _init_state(self, active_rows: int):
        """run_max = LSE_MIN, run_den = 0, res_o = 0."""
        with vf(mode="simd"):
            m1, _ = rr.update_mask(active_rows, elem_bits=32)
            rr.vstore(self.run_max, 0,
                      rr.vdups(LSE_MIN, dtypes.float32, mask=m1), m1)
            rr.vstore(self.run_den, 0,
                      rr.vdups(0.0, dtypes.float32, mask=m1), m1)
            for row in range(active_rows):
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    mask, _ = rr.update_mask(self.tile_d - col, elem_bits=32)
                    rr.vstore(self.vector.res_o, off,
                              rr.vdups(0.0, dtypes.float32, mask=mask), mask)

    @jit
    def _merge_slot(self, active_rows: int):
        """Fold self.o_in / self.lse_in into (run_max, run_den, res_o).

        LSE handling is per-row scalar (``vload_broadcast`` + ``vstore_first``,
        the exact primitive pattern ``Vector._pass_a_row`` uses on the
        softmax row state): the 8-lane ASC LSE layout makes the lane-0
        reads strided, which the vectorized column form cannot express.
        """
        o_in_view = reinterpret(
            self.o_in.consume(),
            shape=(active_rows, self.tile_d), stride=(self.tile_d, 1),
        )
        lse_in = self.lse_in.consume()
        with vf(mode="simd"):
            for row in range(active_rows):
                m1, _ = rr.update_mask(1, elem_bits=32)
                old_max = rr.vload_broadcast(self.run_max, row)
                lse_r = rr.vload_broadcast(lse_in, row * LSE_ROW_LANES)
                slot_invalid = rr.veqs(lse_r, LSE_MIN, mask=m1)
                old_invalid = rr.veqs(old_max, LSE_MIN, mask=m1)
                new_max = rr.vmax(old_max, lse_r, mask=m1)
                old_scale = rr.vexp_sub(old_max, new_max, mask=m1)
                slot_scale = rr.vexp_sub(lse_r, new_max, mask=m1)
                zero = rr.vdups(0.0, dtypes.float32, mask=m1)
                slot_scale = rr.vselect(zero, slot_scale, cond_mask=slot_invalid)
                old_scale = rr.vselect(zero, old_scale, cond_mask=old_invalid)
                rr.vstore_first(self.run_max, row, new_max)
                old_den = rr.vload_broadcast(self.run_den, row)
                new_den = rr.vmadd(old_den, old_scale, slot_scale, mask=m1)
                rr.vstore_first(self.run_den, row, new_den)
                rr.vstore_first(self.old_scale_buf, row, old_scale)
                rr.vstore_first(self.slot_scale_buf, row, slot_scale)
            rr.vmem_bar("vst_vld")

            for row in range(active_rows):
                old_scale_vec = rr.vload_broadcast(self.old_scale_buf, row)
                slot_scale_vec = rr.vload_broadcast(self.slot_scale_buf, row)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    mask, _ = rr.update_mask(self.tile_d - col, elem_bits=32)
                    pre = rr.vload(self.vector.res_o, off)
                    packed = rr.vload_unpack(
                        o_in_view,
                        off,
                        unpack_mode=rr.UnpackMode.B16_TO_B32,
                    )
                    cur = rr.vcast(packed, dtypes.float32, mask=mask)
                    scaled = rr.vmul(cur, slot_scale_vec, mask=mask)
                    o = rr.vmadd(pre, old_scale_vec, scaled, mask=mask)
                    rr.vstore(self.vector.res_o, off, o, mask)

    @jit
    def _load_slot(self, o_ws, lse_ws, slot: int, row0: int):
        """Load one full vector tile from a workspace slot.

        The tile is contained by the slot's tile_cube_m allocation. The merge
        and store paths consume only active rows, so tail data is not observed.
        """
        base = slot * self.tile_cube_m + row0
        o_src = tile_slice(
            o_ws[base:base + self.tile_vec_m, None],
            (self.tile_vec_m, self.tile_d), (0, 0),
        )
        mem_copy(self.o_in.produce(), o_src)
        lse_src = tile_slice(
            lse_ws[base:base + self.tile_vec_m, None],
            (self.tile_vec_m, LSE_ROW_LANES), (0, 0),
        )
        mem_copy(self.lse_in.produce(), lse_src)

    @jit
    def run_asc(self, attn_out, o_ws, lse_ws, metadata, block_idx: int,
                    section: int, head_num_q: int, head_num_kv: int,
                    default_q: int, cu_q, used_q, layout, s1g, softmax_lse_gm):
        sections = dtypes.int64(metadata[HEAD_SECTION_NUM_INDEX])
        m_base = dtypes.int64(metadata[HEAD_M_BASE_SIZE_INDEX])
        base = HEAD_METADATA_STRIDE + sections * ASC_AIC_CORE_NUM * FA_METADATA_STRIDE
        base = base + (section * ASC_AIV_CORE_NUM + block_idx * 2 + self.subblock_idx) * FD_METADATA_STRIDE
        count = metadata[base + FD_M_NUM_INDEX]
        if count > 0:
            bn2 = metadata[base + FD_BN2_IDX_INDEX]
            m = metadata[base + FD_M_IDX_INDEX]
            slot = metadata[base + FD_WORKSPACE_IDX_INDEX]
            splits = metadata[base + FD_WORKSPACE_NUM_INDEX]
            begin = metadata[base + FD_M_START_INDEX]
            batch = bn2 // head_num_kv
            kv_head = bn2 % head_num_kv
            group = head_num_q // head_num_kv
            length = _sequence_length(cu_q, used_q, batch, default_q, layout == "TND")
            for offset in range(0, count, self.tile_vec_m):
                rows = min(self.tile_vec_m, count - offset)
                self._init_state(rows)
                for i in range(splits):
                    self._load_slot(
                        o_ws, lse_ws, slot + i, begin + offset
                    )
                    self._merge_slot(rows)
                self.vector._finalize_div_vf(self.run_den, rows)
                _cast_nd_rows(
                    reinterpret(self.vector.o_ub, shape=(self.tile_vec_m, self.tile_d),
                                stride=(self.tile_d, 1)),
                    self.vector.res_o,
                    rows,
                    self.tile_d,
                )
                source = self.vector.o_ub
                start = m * m_base + begin + offset
                _store_o(attn_out, source, layout, batch, kv_head, start,
                                    rows, group, length, cu_q, s1g)
                if const_expr(self.return_softmax_lse):
                    self.external_lse.store_merged(
                        self.run_max, self.run_den, softmax_lse_gm, layout, batch,
                        kv_head, start, rows, group, length, default_q, head_num_q,
                        cu_q, s1g)


def get_tile_config(D):
    """Return (tile_cube_m, tile_vec_m, tile_n, tile_d) for the given head dim."""
    if const_expr(D == 64):
        return 128, 64, 128, 64
    if const_expr(D == 128):
        return 128, 64, 128, 128
    if const_expr(D == 256):
        return 64, 32, 128, 256
    raise ValueError(f"Unsupported head dim D={D}; expected 64, 128, 256")


# ============================================================================
# 3. JIT launch
# ============================================================================


class FlashAttnLauncher:
    """JIT-compiled kernel launch with layout transforms."""

    def __init__(self, layout_q, layout_kv, layout_out, mask_mode, dtype,
                 block_dim, batch_count=0, s1g_mapping=False,
                 return_softmax_lse=False, *, tile_config=None):
        self.layout_q = layout_q
        self.layout_kv = layout_kv
        self.layout_out = layout_out
        self.mask_mode = mask_mode
        self.dtype = dtype
        self.block_dim = block_dim
        self.batch_count = batch_count
        self.s1g_mapping = s1g_mapping
        self.return_softmax_lse = return_softmax_lse
        self.tile_config = tile_config
        self._has_tile_config = tile_config is not None
        self._tile_cube_m = tile_config.tile_cube_m if tile_config is not None else 0
        self._tile_vec_m = tile_config.tile_vec_m if tile_config is not None else 0
        self._tile_n = tile_config.tile_n if tile_config is not None else 0
        self._tile_d = tile_config.head_dim if tile_config is not None else 0

    @host
    def launch(
        self,
        attn_out: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: float,
        block_table: Tensor | None = None,
        cu_seqlens_q: Tensor | None = None,
        cu_seqlens_kv: Tensor | None = None,
        seqused_q: Tensor | None = None,
        seqused_kv: Tensor | None = None,
        sinks: Tensor | None = None,
        win_left: int = 0,
        win_right: int = 0,
        max_seqlen_q: int = -1,
        max_seqlen_kv: int = -1,
        softmax_lse_gm: Tensor | None = None,
        attn_mask: Tensor | None = None,
        metadata: Tensor | None = None,
        o_ws: Tensor | None = None,
        lse_ws: Tensor | None = None,
        out_init_gm: Tensor | None = None,
        need_init_output: int = 0,
        pa_zero: Tensor | None = None,
    ):
        """JIT-compiled kernel launch with layout transforms.

        Each metadata section executes in two phases: the compute phase
        writes each split task's partial O/LSE to ``o_ws``/``lse_ws``, an
        in-kernel all-core barrier, then a fused combine phase merges each output
        tile's slots into ``attn_out`` via the ASC binary ``metadata`` FD
        descriptors.
        """
        # Keep the public layout's natural merged-row order.  A view changes
        # indexing only; it must not select a different task model.
        s1g_mapping = self.s1g_mapping
        if const_expr(self.layout_q == "BSND"):
            query = permute(query, (0, 2, 1, 3))
        if const_expr(self.layout_kv == "BSND"):
            key = permute(key, (0, 2, 1, 3))
            value = permute(value, (0, 2, 1, 3))
        if const_expr(self.layout_out == "BSND"):
            attn_out = permute(attn_out, (0, 2, 1, 3))
        # TND arrives as (T, N, D) and becomes (N, T, D) so that narrowing the
        # head leaves the token axis second-to-last, where a rank-2 tiler can
        # reach it. This is a VIEW: no bytes move, and the row pitch stays N*D.
        if const_expr(self.layout_q == "TND"):
            query = permute(query, (1, 0, 2))
        if const_expr(self.layout_kv == "TND"):
            key = permute(key, (1, 0, 2))
            value = permute(value, (1, 0, 2))
        if const_expr(self.layout_out == "TND"):
            attn_out = permute(attn_out, (1, 0, 2))

        head_dim = query.shape[-1]
        if const_expr(self._has_tile_config):
            tile_cube_m = self._tile_cube_m
            tile_vec_m = self._tile_vec_m
            tile_n = self._tile_n
            tile_d = self._tile_d
        else:
            tile_cube_m, tile_vec_m, tile_n, tile_d = get_tile_config(head_dim)
        op = FlashAttnKernel(
            tile_cube_m=tile_cube_m,
            tile_vec_m=tile_vec_m,
            tile_n=tile_n,
            tile_d=tile_d,
            return_softmax_lse=self.return_softmax_lse,
            mask_mode=self.mask_mode,
            dtype_16=self.dtype,
            layout="TND" if self.layout_q == "TND" else "BNSD",
            batch_count=self.batch_count,
            s1g_mapping=s1g_mapping,
            kv_layout=(self.layout_kv if self.layout_kv in ("PA_BNBD", "PA_BBND", "PA_NZ", "TND")
                       else "BNSD"),
            output_layout=self.layout_out,
            pa_strided=(self.layout_kv in ("PA_BNBD", "PA_BBND", "PA_NZ")
                        and (key.stride[-1] != 1 or value.stride[-1] != 1)),
        )
        op[self.block_dim](
            attn_out,
            query,
            key,
            value,
            scale,
            block_table,
            cu_seqlens_q,
            cu_seqlens_kv,
            seqused_q,
            seqused_kv,
            sinks,
            win_left,
            win_right,
            max_seqlen_q,
            max_seqlen_kv,
            softmax_lse_gm,
            attn_mask,
            metadata,
            o_ws,
            lse_ws,
            out_init_gm,
            need_init_output,
            pa_zero,
        )


# ============================================================================
# 4. Torch Interface
# ============================================================================


@lru_cache(maxsize=32)
def _pa_zero_tile(head_dim, dtype, device):
    """A fixed read-only zero tile; never scales with batch or KV length."""
    return torch.zeros((128, head_dim), dtype=dtype).to(device)


def flash_attn(
    query,
    key,
    value,
    block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="BSND",
    layout_out="BSND",
    return_softmax_lse=False,
    *,
    dtype=dtypes.float16,
    o_workspace=None,
    lse_workspace=None,
):
    """Run FlashAttention from AICPU-generated load-balance metadata.

    The public argument order mirrors ops-transformer's ``flash_attn``.  All
    layouts use this single entry and ``metadata`` is required for every call.
    """
    validated = validate_flash_attn_inputs(
        query=query, key=key, value=value, block_table=block_table,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q, seqused_kv=seqused_kv,
        attn_mask=attn_mask, metadata=metadata, mask_mode=mask_mode,
        win_left=win_left, win_right=win_right, layout_q=layout_q,
        layout_kv=layout_kv, layout_out=layout_out,
        return_softmax_lse=return_softmax_lse,
        max_seqlen_q=max_seqlen_q,
    )
    batch_size, query_length, kv_length, q_heads, kv_heads, head_dim = (
        validated.batch, validated.query_length, validated.kv_length,
        validated.q_heads, validated.kv_heads, validated.head_dim,
    )
    tile_config = validated.tile_config

    need_init_output = int(_need_init_output(
        has_seqused_q=_present_nonempty(seqused_q),
        has_seqused_kv=_present_nonempty(seqused_kv),
        has_cu_seqlens_kv=_present_nonempty(cu_seqlens_kv),
        mask_mode=mask_mode,
        s1_size=query_length,
        s2_size=kv_length,
        win_right=win_right,
    ))
    stream = torch.npu.current_stream()
    block_dim, vector_dim = get_effective_core_counts(stream=stream)
    validate_output_init_core_counts(need_init_output, block_dim, vector_dim)
    # Use the same static bound as AICPU; never read metadata or sequence data.
    block_dim, vector_dim = get_launch_core_counts(
        block_dim, vector_dim, batch_size, q_heads, kv_heads, head_dim,
        max_seqlen_q, max_seqlen_kv, mask_mode, win_left, win_right,
        tile_m_override=tile_config.tile_cube_m
        if head_dim == 256 and tile_config.tile_cube_m == 128 else 0,
        tile_n_override=tile_config.tile_n
        if head_dim == 256 and tile_config.tile_cube_m == 128 else 0,
    )
    out_shape, lse_shape = output_shapes(
        layout_q, layout_out, query, batch_size, query_length, q_heads, head_dim,
    )
    out = torch.empty(out_shape, dtype=query.dtype, device=query.device)
    softmax_lse = (
        torch.empty(lse_shape, dtype=torch.float32, device=query.device)
        if return_softmax_lse else None
    )

    tile_cube_m = tile_config.tile_cube_m
    # Each native boundary-fragment slot has one contiguous subtile per Q head.
    max_slots = 2 * ASC_AIC_CORE_NUM * (q_heads // kv_heads)
    o_workspace = o_workspace if o_workspace is not None else torch.empty(
        (max_slots + 1) * tile_cube_m, head_dim,
        dtype=query.dtype, device=query.device,
    )
    lse_workspace = lse_workspace if lse_workspace is not None else torch.empty(
        (max_slots + 1) * tile_cube_m, LSE_ROW_LANES,
        dtype=torch.float32, device=query.device,
    )

    required_rows = (max_slots + 1) * tile_cube_m
    validate_workspace(
        "o_workspace", o_workspace, required_rows, head_dim, query.dtype, query.device,
    )
    validate_workspace(
        "lse_workspace", lse_workspace, required_rows, LSE_ROW_LANES,
        torch.float32, query.device,
    )

    launcher = FlashAttnLauncher(
        layout_q,
        layout_kv,
        layout_out,
        mask_mode,
        dtype,
        block_dim,
        s1g_mapping=layout_q in ("BSND", "TND"),
        batch_count=batch_size,
        return_softmax_lse=return_softmax_lse,
        tile_config=tile_config,
    )
    launcher.launch(
        out,
        query,
        key,
        value,
        softmax_scale,
        block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        win_left=win_left,
        win_right=win_right,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        softmax_lse_gm=softmax_lse.view(-1, 1) if return_softmax_lse else None,
        attn_mask=attn_mask,
        metadata=metadata,
        o_ws=o_workspace,
        lse_ws=lse_workspace,
        out_init_gm=out.view(-1, 1),
        need_init_output=need_init_output,
        pa_zero=_pa_zero_tile(head_dim, query.dtype, query.device)
        if validated.pa_block_size is not None else None,
    )

    return out, softmax_lse
