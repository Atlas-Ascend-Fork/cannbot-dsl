# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""QLI准备、KV128流水、Token TopK和候选块选择；不读取外部Metadata。"""

from __future__ import annotations

from typing import Optional

import cannbotdsl
import torch
from cannbotdsl import Dim, MemLoc, RingBuffer, Tensor, TensorSpec, const_expr, dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import range_constexpr
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops.sync import global_sync_all, onchip_rewind, vec_sync_all
from cannbotdsl.tensor import make_layout, make_pointer, make_tensor, tile_slice
from cannbotdsl.tensor import reinterpret as _reinterpret_op


def reinterpret(tensor, dtype=None, shape=None, stride=None, offset=0, **kwargs):
    """DSL 0.8.0 takes a Layout instead of shape/stride; keep the sample's form."""
    if shape is not None:
        kwargs["layout"] = (
            make_layout(shape) if stride is None else make_layout(shape, stride=stride)
        )
    return _reinterpret_op(tensor, dtype=dtype, offset=offset, **kwargs)


QUERY_TILE_ROWS = 6

TILE_M = 32 * QUERY_TILE_ROWS

TILE_N = 256


CACHE_TOKENS = 25600

LOGICAL_D = 128

PACKED_D = LOGICAL_D // 2


N2 = 1


CANDIDATE_BLOCK_SIZE = 8


MAX_CUBE_WORKERS = 32

LD_RECORD_START = 36

METADATA_ELEMENTS = 1024

METADATA_FIELDS = 8

METADATA_ROWS = METADATA_ELEMENTS // METADATA_FIELDS


BF16_NEG_INF_BITS = 0xFF80

STREAM_TOPK_TRUNK_LEN = 16384

MAX_MERGE_PARTS = 8

MAX_PA_BLOCK_SIZE = 1024

PA_BLOCK_ALIGNMENT = 16

QK_ROW_BYTES = TILE_N * 2

CACHED_QK_UB_BYTES = TILE_M * QK_ROW_BYTES

CACHED_SCORE_UB_BYTES = (QUERY_TILE_ROWS // 2) * CACHE_TOKENS * 2

CACHED_HISTOGRAM_UB_OFFSET = CACHED_QK_UB_BYTES + CACHED_SCORE_UB_BYTES


CACHED_VECTOR_UB_OFFSET = (
    CACHED_HISTOGRAM_UB_OFFSET + QUERY_TILE_ROWS * QK_ROW_BYTES + QK_ROW_BYTES
)


KEY_STORAGE_ROW_ELEMENTS = N2 * PACKED_D

KEY_SCALE_STORAGE_ROW_ELEMENTS = N2 * 2 * 2


@jit
def _need_topk_values(metadata, return_values):
    return dtypes.int64(1 if metadata[LD_RECORD_START, 0] != 0 else return_values)


UB_CAPACITY_BYTES = 256 * 1024

UB_ALIGNMENT_BYTES = 32

TOPK_HISTOGRAM_UB_OFFSET = CACHED_HISTOGRAM_UB_OFFSET + QUERY_TILE_ROWS * QK_ROW_BYTES

TOPK_TARGET_UB_OFFSET = CACHED_VECTOR_UB_OFFSET


def _ub_buffer(memloc, shape, dtype):
    """Allocate one UB buffer through the automatic on-chip allocator.

    Addresses are reused by the compiler at every ``onchip_rewind()`` phase
    boundary, so no explicit offset is carried here any more.
    """
    return Buffer(memloc, dtype, make_layout(shape))


def _ub_channel(memloc, shape, dtype, *, depth=1):
    """Allocate a rotating UB channel through the automatic allocator."""
    return RingBuffer(memloc, dtype, make_layout(shape), depth=depth)


def _offset_view(tensor, offsets):
    """Rebase an ND input with the current DSL's zero-copy interval slicing."""
    if all(isinstance(offset, int) and offset == 0 for offset in offsets):
        return tensor
    return tensor[tuple(slice(offset, None) for offset in offsets)]


def ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _pa_axis0_storage_rows(
    tensor: torch.Tensor,
    name: str,
    expected_inner_strides: tuple[int, ...],
    storage_row_elements: int,
) -> tuple[int, int]:
    """Validate ASC's PA stride contract and expose its storage-row span.

    Only axis 0 may contain padding. The returned extent includes that
    padding without copying or repacking public input data; the fused kernel
    still selects physical pages with the original axis-0 stride.
    """
    strides = tuple(int(step) for step in tensor.stride())
    if strides[1:] != expected_inner_strides:
        raise ValueError(
            f"{name} only supports non-contiguous storage on axis 0 in PA_BBND layout"
        )
    stride0 = strides[0]
    if stride0 < 0:
        raise ValueError(f"{name} axis-0 stride must be non-negative")
    if stride0 % storage_row_elements != 0:
        raise ValueError(f"{name} axis-0 stride must preserve complete PA storage rows")
    blocks = int(tensor.shape[0])
    if blocks <= 0:
        raise ValueError(f"{name} must contain at least one physical PA block")
    storage_rows = (blocks - 1) * (stride0 // storage_row_elements) + int(
        tensor.shape[1]
    )
    return stride0, storage_rows


# Token/Candidate选择及位置排序。
CANDIDATE_LD_TRUNK_LEN = 16384


class QliRawTopKWorkspace:
    """UB scratch shared by sequential Sparse and Candidate TopK stages."""

    def __init__(self, max_trunk_len: int, max_topk: int, pipelined: bool = False):
        self.max_trunk_len = int(max_trunk_len)
        self.max_topk = int(max_topk)
        self.pipelined = bool(pipelined)
        self.topk_pad = ceil_div(self.max_topk, 256) * 256
        self.merge_len_pad = ceil_div(self.topk_pad + self.max_trunk_len, 128) * 128
        self.current_trunk = _ub_channel(
            MemLoc.UB,
            (1, self.max_trunk_len),
            dtypes.uint16,
            depth=1,
        )
        if self.pipelined:
            self.merge_key = _ub_channel(
                MemLoc.UB, (1, self.merge_len_pad), dtypes.uint16, depth=2
            )
        else:
            self.merge_key = _ub_channel(
                MemLoc.UB, (1, self.merge_len_pad), dtypes.uint16, depth=1
            )
        self.tmp_idx = _ub_buffer(
            MemLoc.UB,
            (self.merge_len_pad,),
            dtypes.uint16,
        )
        self.histogram = _ub_buffer(MemLoc.UB, (256,), dtypes.uint16)
        self.high_target_carrier = _ub_buffer(MemLoc.UB, (64,), dtypes.int32)
        self.history_key = _ub_buffer(
            MemLoc.UB,
            (2, self.topk_pad),
            dtypes.uint16,
        )
        self.next_history_key = _ub_channel(
            MemLoc.UB,
            (1, self.topk_pad),
            dtypes.uint16,
            depth=1,
        )
        self.history_idx = _ub_buffer(
            MemLoc.UB,
            (2, self.topk_pad),
            dtypes.int32,
        )
        self.next_idx = _ub_buffer(
            MemLoc.UB,
            (self.topk_pad,),
            dtypes.int32,
        )


class _RawTopKSelectorBase:
    """``QliRawTopKSelector`` 与 ``CandidateQliRawTopKSelector`` 共用的 @jit 工具。

    两者只有选择策略不同；合并、两段式选择、历史更新与空行写出完全一致，
    抽到基类后修 mask/gather/同步只需要改一处。Buffer/RingBuffer 归属与
    同步时序保持不变。
    """

    @jit
    def _build_merge(
        self,
        current_trunk,
        history_key,
        merge_key,
        current_len,
        current_len_pad,
        history_prefix=None,
    ):
        """Assemble a trunk, optionally preceded by its retained TopK."""
        if history_prefix is None:
            history_prefix = self.topk_pad
        with vf(mode="simd"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            for chunk in dsl_range((history_prefix + 127) // 128, unroll=1):
                offset = chunk * 128
                rr.vstore(
                    merge_key,
                    offset,
                    rr.vload(history_key, offset),
                    mask16,
                )
            current_chunk_count = (current_len + 127) // 128
            zero = rr.vdups(0, dtypes.uint16, mask=mask16)
            for chunk in dsl_range(current_chunk_count, unroll=1):
                source_offset = chunk * 128
                count = current_len - source_offset
                current_mask = rr.update_mask(count, elem_bits=16)[0]
                selected = rr.vselect(
                    rr.vload(current_trunk, source_offset), zero, cond_mask=current_mask
                )
                rr.vstore(merge_key, history_prefix + source_offset, selected, mask16)
            for chunk in dsl_range(
                current_chunk_count, (current_len_pad + 127) // 128, unroll=1
            ):
                rr.vstore(merge_key, history_prefix + chunk * 128, zero, mask16)
            rr.vmem_bar("vst_vld")

    @jit
    def _select_pass(
        self,
        sort_key,
        valid_len,
        history_prefix,
        current_global_base,
        tmp_idx,
        histogram,
        high_target_carrier,
        history_idx_in,
        history_idx_out,
        next_history_key,
    ):
        """Run one ASC radix pass over one trunk or history+trunk."""
        self._find_indices(sort_key, valid_len, tmp_idx, histogram, high_target_carrier)

        with vf(mode="simd"):
            # ASC FindValueOutputVFImpl keeps the compacted positions as
            # uint16 and gathers 128 values per iteration.  Do not reuse the
            # uint16->uint32 unpack used by FindRealIndexVFImpl below: that
            # unpack exposes only 64 lanes and breaks value/index pairing
            # once history is merged with another trunk.
            for chunk in dsl_range(ceil_div(self.topk, 128), unroll=1):
                count = self.topk - chunk * 128
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                pos16 = rr.vload(tmp_idx, chunk * 128)
                selected_key = rr.vgather(sort_key, pos16, mask=mask16)
                rr.vstore(
                    next_history_key,
                    chunk * 128,
                    selected_key,
                    mask16,
                )

            # ASC FindRealIndexVFImpl loads the same compacted positions with
            # DIST_UNPACK_B16, yielding 64 uint32 lanes for history/current
            # index mapping.  Keep this separate from the value gather above.
            _full32 = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                pos16 = rr.vload(tmp_idx, chunk * 64)
                pos32 = rr.vunpack(pos16, dtypes.uint32, part="lower")
                is_current = rr.vges(pos32, history_prefix, mask=mask32)
                is_history = rr.vlts(pos32, history_prefix, mask=mask32)
                old_idx = rr.vgather(history_idx_in, pos32, mask=is_history)
                current_idx = rr.vadds(
                    rr.vreinterpret(pos32, dtypes.int32),
                    current_global_base - history_prefix,
                    mask=mask32,
                )
                global_idx = rr.vselect(current_idx, old_idx, cond_mask=is_current)
                rr.vstore(history_idx_out, chunk * 64, global_idx, mask32)
            rr.vmem_bar("vst_vld")

    @jit
    def _commit_history(self, next_idx, history_idx, next_history_key, history_key):
        with vf(mode="simd"):
            copy_mask32 = rr.update_mask(64, elem_bits=32)[0]
            copy_mask16 = rr.update_mask(128, elem_bits=16)[0]
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                rr.vstore(
                    history_idx,
                    chunk * 64,
                    rr.vload(next_idx, chunk * 64),
                    copy_mask32,
                )
            for chunk in dsl_range(ceil_div(self.topk, 128), unroll=1):
                rr.vstore(
                    history_key,
                    chunk * 128,
                    rr.vload(next_history_key, chunk * 128),
                    copy_mask16,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def store_empty_row(
        self,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
    ):
        """Store the public empty-TopK representation without scanning GM."""
        idx_write = self.output_idx_stage.next_write()
        val_write = self.output_value_bits_stage.next_write()
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.mask_and(
                    rr.update_mask(count, elem_bits=16)[0],
                    rr.update_mask(64, elem_bits=16)[0],
                    exec_mask=rr.update_mask(64, elem_bits=16)[0],
                )
                rr.vstore(
                    idx_write,
                    chunk * 64,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    mask32,
                )
                rr.vstore(
                    val_write,
                    chunk * 64,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    mask16,
                )
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, self.output_idx_stage.next_read())

        mem_copy(gm_selected_value_bits, self.output_value_bits_stage.next_read())


class QliRawTopKSelector(_RawTopKSelectorBase):
    """ASC Arch35 uint16 radix TopK with in-AIV trunk/history merge.

    This is not LD/S2 core splitting.  One AIV processes a row sequentially in
    16K trunks.  The first trunk creates a TopK history; every later pass runs
    the same radix selector over ``history + current trunk`` and maps its local
    uint16 positions back to global int32 token indices.
    """

    def __init__(
        self,
        topk: int,
        workspace: QliRawTopKWorkspace,
    ):
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("ASC raw TopK requires positive topk")
        # The ASC VF works in fixed 16K trunks.  The number of complete trunks
        # and the final tail length are runtime values so one compiled kernel
        # can serve every supported S2.
        self.trunk_len = workspace.max_trunk_len
        # ASC reserves history in 256-element units before the next trunk.
        self.topk_pad = ceil_div(self.topk, 256) * 256
        self.merge_len_pad = ceil_div(self.topk_pad + self.trunk_len, 128) * 128
        self.workspace = workspace
        self.pipelined = bool(getattr(workspace, "pipelined", False))
        if (
            self.trunk_len > workspace.max_trunk_len
            or self.topk > workspace.max_topk
            or self.merge_len_pad > workspace.merge_len_pad
        ):
            raise ValueError("TopK selector exceeds the shared UB workspace")
        # Final V->MTE3 stages are selector-owned so Sparse and Candidate can
        # share computational scratch without racing each other's GM writes.
        self.output_idx_stage = _ub_channel(
            MemLoc.UB,
            (1, self.topk),
            dtypes.int32,
            depth=1,
        )
        self.output_value_bits_stage = _ub_channel(
            MemLoc.UB,
            (1, self.topk),
            dtypes.uint16,
            depth=1,
        )
        self.hist_slot0 = workspace.histogram
        self.hist_slot1 = workspace.histogram

    @jit
    def _load_merge(
        self, gm_keys, history, merge, source_offset, history_prefix, length
    ):
        copied = ((length + 127) // 128) * 128
        if not self.pipelined:
            merge_slot = merge.next_write()
            mem_copy(
                reinterpret(
                    _offset_view(merge_slot, (0, history_prefix)), shape=(1, copied)
                ),
                gm_keys[0:1, source_offset : source_offset + copied],
            )
            with vf(mode="simd"):
                mask = rr.update_mask(128, elem_bits=16)[0]
                for chunk in dsl_range(history_prefix // 128, unroll=1):
                    rr.vstore(
                        merge_slot, chunk * 128, rr.vload(history, chunk * 128), mask
                    )
                rr.vmem_bar("vst_vld")
        elif history_prefix == 0:
            # Common case: no retained history -> load the trunk straight into
            # the double-buffered merge Channel (async, no extra V copy).
            mem_copy(
                reinterpret(merge.next_write(), shape=(1, copied)),
                gm_keys[0:1, source_offset : source_offset + copied],
            )
        else:
            mem_copy(
                reinterpret(
                    self.workspace.current_trunk.next_write(), shape=(1, copied)
                ),
                gm_keys[0:1, source_offset : source_offset + copied],
            )
            self._build_merge(
                self.workspace.current_trunk.next_read(),
                history,
                merge.next_write(),
                length,
                copied,
                history_prefix,
            )

    @jit
    def _find_indices(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            h0c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            h0c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            h0c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            h0c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            full8 = rr.update_mask(256, elem_bits=8)[0]
            for block in dsl_range(valid_len // 1024, unroll=1):
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 0, width="b8"
                )
                h0c0 = rr.vhistogram_accumulate(h0c0, high, mask=full8, bin=0)
                h1c0 = rr.vhistogram_accumulate(h1c0, high, mask=full8, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 256, width="b8"
                )
                h0c1 = rr.vhistogram_accumulate(h0c1, high, mask=full8, bin=0)
                h1c1 = rr.vhistogram_accumulate(h1c1, high, mask=full8, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 512, width="b8"
                )
                h0c2 = rr.vhistogram_accumulate(h0c2, high, mask=full8, bin=0)
                h1c2 = rr.vhistogram_accumulate(h1c2, high, mask=full8, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 768, width="b8"
                )
                h0c3 = rr.vhistogram_accumulate(h0c3, high, mask=full8, bin=0)
                h1c3 = rr.vhistogram_accumulate(h1c3, high, mask=full8, bin=1)
            for chunk in dsl_range(
                (valid_len // 1024) * 4, (valid_len + 255) // 256, unroll=1
            ):
                mask8 = rr.update_mask(valid_len - chunk * 256, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                h0c0 = rr.vhistogram_accumulate(h0c0, high, mask=mask8, bin=0)
                h1c0 = rr.vhistogram_accumulate(h1c0, high, mask=mask8, bin=1)
            histogram_low_reg = rr.vadd(h0c0, h0c1, mask=full16)
            histogram_low_reg = rr.vadd(histogram_low_reg, h0c2, mask=full16)
            histogram_low_reg = rr.vadd(histogram_low_reg, h0c3, mask=full16)
            histogram_high_reg = rr.vadd(h1c0, h1c1, mask=full16)
            histogram_high_reg = rr.vadd(histogram_high_reg, h1c2, mask=full16)
            histogram_high_reg = rr.vadd(histogram_high_reg, h1c3, mask=full16)

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            bottom_k = valid_len - self.topk + 1
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = histogram_low_reg
            hist1 = histogram_high_reg
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist0, bottom_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist1, bottom_k, mask=full16),
            )
            high_target = rr.vreduce_min(
                rr.vmin(candidate0, candidate1, mask=full16), mask=full16
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=full16)
            one16 = rr.vdups(1, dtypes.uint16, mask=full16)
            bottom_k = valid_len - self.topk + 1
            high_target = rr.vdup(high_target, mask=full16)
            high_target8 = rr.vreinterpret_lanes(
                rr.vbitwise_or(
                    high_target, rr.vshl(high_target, 8, mask=full16), mask=full16
                ),
                dtypes.uint8,
            )
            prev = rr.vsub(high_target, one16, mask=full16)
            is_zero = rr.veqs(high_target, 0, mask=full16)
            prev = rr.vselect(zero16, prev, cond_mask=is_zero)
            prev_lane = rr.vbitwise_and(
                prev, rr.vdups(127, dtypes.uint16, mask=full16), mask=full16
            )
            prev0 = rr.vgather_reg(histogram_low_reg, prev_lane)
            prev1 = rr.vgather_reg(histogram_high_reg, prev_lane)
            prev_count = rr.vselect(
                prev0, prev1, cond_mask=rr.vlts(prev, 128, mask=full16)
            )
            prev_count = rr.vselect(zero16, prev_count, cond_mask=is_zero)
            next_k = rr.vsub(
                rr.vdups(bottom_k, dtypes.uint16, mask=full16),
                prev_count,
                mask=full16,
            )
            next_k_register = rr.vdup(next_k, mask=full16)

            l0c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            full8 = rr.update_mask(256, elem_bits=8)[0]
            for block in dsl_range(valid_len // 1024, unroll=1):
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 0, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
                l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 256, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c1 = rr.vhistogram_accumulate(l0c1, low, mask=eq_high, bin=0)
                l1c1 = rr.vhistogram_accumulate(l1c1, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 512, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c2 = rr.vhistogram_accumulate(l0c2, low, mask=eq_high, bin=0)
                l1c2 = rr.vhistogram_accumulate(l1c2, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 768, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c3 = rr.vhistogram_accumulate(l0c3, low, mask=eq_high, bin=0)
                l1c3 = rr.vhistogram_accumulate(l1c3, low, mask=eq_high, bin=1)
            for chunk in dsl_range(
                (valid_len // 1024) * 4, (valid_len + 255) // 256, unroll=1
            ):
                mask8 = rr.update_mask(valid_len - chunk * 256, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
                l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
            l0 = rr.vadd(l0c0, l0c1, mask=full16)
            l0 = rr.vadd(l0, l0c2, mask=full16)
            l0 = rr.vadd(l0, l0c3, mask=full16)
            l1 = rr.vadd(l1c0, l1c1, mask=full16)
            l1 = rr.vadd(l1, l1c2, mask=full16)
            l1 = rr.vadd(l1, l1c3, mask=full16)

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            next_k = next_k_register
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = l0
            hist1 = l1
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist0, next_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist1, next_k, mask=full16),
            )
            low_target = rr.vreduce_min(
                rr.vmin(candidate0, candidate1, mask=full16), mask=full16
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            low_target = rr.vdup(low_target, mask=full16)
            kth = rr.vbitwise_or(
                rr.vshl(high_target, 8, mask=full16),
                low_target,
                mask=full16,
            )
            ureg = rr.vstore_unalign_begin(tmp_idx)
            valid_chunk_count = (valid_len + 127) // 128
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.vgt(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.veq(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            rr.vstore_unalign_post(tmp_idx, 0, ureg)
            rr.vmem_bar("vst_vld")

    @jit
    def _find_indices_cached(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            histogram_low_reg = rr.vload(histogram, 0)
            histogram_high_reg = rr.vload(histogram, 128)

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            bottom_k = valid_len - self.topk + 1
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = histogram_low_reg
            hist1 = histogram_high_reg
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist0, bottom_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist1, bottom_k, mask=full16),
            )
            high_target = rr.vreduce_min(
                rr.vmin(candidate0, candidate1, mask=full16), mask=full16
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=full16)
            one16 = rr.vdups(1, dtypes.uint16, mask=full16)
            bottom_k = valid_len - self.topk + 1
            high_target = rr.vdup(high_target, mask=full16)
            high_target8 = rr.vreinterpret_lanes(
                rr.vbitwise_or(
                    high_target, rr.vshl(high_target, 8, mask=full16), mask=full16
                ),
                dtypes.uint8,
            )
            prev = rr.vsub(high_target, one16, mask=full16)
            is_zero = rr.veqs(high_target, 0, mask=full16)
            prev = rr.vselect(zero16, prev, cond_mask=is_zero)
            prev_lane = rr.vbitwise_and(
                prev, rr.vdups(127, dtypes.uint16, mask=full16), mask=full16
            )
            prev0 = rr.vgather_reg(histogram_low_reg, prev_lane)
            prev1 = rr.vgather_reg(histogram_high_reg, prev_lane)
            prev_count = rr.vselect(
                prev0, prev1, cond_mask=rr.vlts(prev, 128, mask=full16)
            )
            prev_count = rr.vselect(zero16, prev_count, cond_mask=is_zero)
            next_k = rr.vsub(
                rr.vdups(bottom_k, dtypes.uint16, mask=full16),
                prev_count,
                mask=full16,
            )
            next_k_register = rr.vdup(next_k, mask=full16)

            l0c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            full8 = rr.update_mask(256, elem_bits=8)[0]
            for block in dsl_range(valid_len // 1024, unroll=1):
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 0, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
                l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 256, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c1 = rr.vhistogram_accumulate(l0c1, low, mask=eq_high, bin=0)
                l1c1 = rr.vhistogram_accumulate(l1c1, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 512, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c2 = rr.vhistogram_accumulate(l0c2, low, mask=eq_high, bin=0)
                l1c2 = rr.vhistogram_accumulate(l1c2, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 768, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c3 = rr.vhistogram_accumulate(l0c3, low, mask=eq_high, bin=0)
                l1c3 = rr.vhistogram_accumulate(l1c3, low, mask=eq_high, bin=1)
            for chunk in dsl_range(
                (valid_len // 1024) * 4, (valid_len + 255) // 256, unroll=1
            ):
                mask8 = rr.update_mask(valid_len - chunk * 256, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
                l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
            l0 = rr.vadd(l0c0, l0c1, mask=full16)
            l0 = rr.vadd(l0, l0c2, mask=full16)
            l0 = rr.vadd(l0, l0c3, mask=full16)
            l1 = rr.vadd(l1c0, l1c1, mask=full16)
            l1 = rr.vadd(l1, l1c2, mask=full16)
            l1 = rr.vadd(l1, l1c3, mask=full16)

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            next_k = next_k_register
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = l0
            hist1 = l1
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist0, next_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist1, next_k, mask=full16),
            )
            low_target = rr.vreduce_min(
                rr.vmin(candidate0, candidate1, mask=full16), mask=full16
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            low_target = rr.vdup(low_target, mask=full16)
            kth = rr.vbitwise_or(
                rr.vshl(high_target, 8, mask=full16),
                low_target,
                mask=full16,
            )
            ureg = rr.vstore_unalign_begin(tmp_idx)
            valid_chunk_count = (valid_len + 127) // 128
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.vgt(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.veq(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            rr.vstore_unalign_post(tmp_idx, 0, ureg)
            rr.vmem_bar("vst_vld")

    @jit
    def _find_indices_dispatch(
        self,
        sort_key,
        valid_len,
        tmp_idx,
        histogram,
        high_target_carrier,
        cached_hist,
        hist_row,
    ):
        if cached_hist:
            self._find_indices_dispatch_cached(
                sort_key, valid_len, tmp_idx, histogram, high_target_carrier, hist_row
            )
        else:
            self._find_indices_dispatch_scanned(
                sort_key, valid_len, tmp_idx, histogram, high_target_carrier, hist_row
            )

    @jit
    def _find_indices_dispatch_cached(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier, hist_row
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            histogram_low_reg = rr.vdups(0, dtypes.uint16, mask=full16)
            histogram_high_reg = rr.vdups(0, dtypes.uint16, mask=full16)
            histogram_low_reg = rr.vadd(
                rr.vload(self.hist_slot0, hist_row * 512),
                rr.vload(self.hist_slot1, hist_row * 512),
                mask=full16,
            )
            histogram_high_reg = rr.vadd(
                rr.vload(self.hist_slot0, hist_row * 512 + 256),
                rr.vload(self.hist_slot1, hist_row * 512 + 256),
                mask=full16,
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            bottom_k = valid_len - self.topk + 1
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = histogram_low_reg
            hist1 = histogram_high_reg
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist0, bottom_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist1, bottom_k, mask=full16),
            )
            high_target = rr.vreduce_min(
                rr.vmin(candidate0, candidate1, mask=full16), mask=full16
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=full16)
            one16 = rr.vdups(1, dtypes.uint16, mask=full16)
            bottom_k = valid_len - self.topk + 1
            high_target = rr.vdup(high_target, mask=full16)
            high_target8 = rr.vreinterpret_lanes(
                rr.vbitwise_or(
                    high_target, rr.vshl(high_target, 8, mask=full16), mask=full16
                ),
                dtypes.uint8,
            )
            prev = rr.vsub(high_target, one16, mask=full16)
            is_zero = rr.veqs(high_target, 0, mask=full16)
            prev = rr.vselect(zero16, prev, cond_mask=is_zero)
            prev_lane = rr.vbitwise_and(
                prev, rr.vdups(127, dtypes.uint16, mask=full16), mask=full16
            )
            prev0 = rr.vgather_reg(histogram_low_reg, prev_lane)
            prev1 = rr.vgather_reg(histogram_high_reg, prev_lane)
            prev_count = rr.vselect(
                prev0, prev1, cond_mask=rr.vlts(prev, 128, mask=full16)
            )
            prev_count = rr.vselect(zero16, prev_count, cond_mask=is_zero)
            next_k = rr.vsub(
                rr.vdups(bottom_k, dtypes.uint16, mask=full16),
                prev_count,
                mask=full16,
            )
            next_k_register = rr.vdup(next_k, mask=full16)

            l0c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            full8 = rr.update_mask(256, elem_bits=8)[0]
            for block in dsl_range(valid_len // 1024, unroll=1):
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 0, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
                l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 256, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c1 = rr.vhistogram_accumulate(l0c1, low, mask=eq_high, bin=0)
                l1c1 = rr.vhistogram_accumulate(l1c1, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 512, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c2 = rr.vhistogram_accumulate(l0c2, low, mask=eq_high, bin=0)
                l1c2 = rr.vhistogram_accumulate(l1c2, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 768, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c3 = rr.vhistogram_accumulate(l0c3, low, mask=eq_high, bin=0)
                l1c3 = rr.vhistogram_accumulate(l1c3, low, mask=eq_high, bin=1)
            for chunk in dsl_range(
                (valid_len // 1024) * 4, (valid_len + 255) // 256, unroll=1
            ):
                mask8 = rr.update_mask(valid_len - chunk * 256, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
                l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
            l0 = rr.vadd(l0c0, l0c1, mask=full16)
            l0 = rr.vadd(l0, l0c2, mask=full16)
            l0 = rr.vadd(l0, l0c3, mask=full16)
            l1 = rr.vadd(l1c0, l1c1, mask=full16)
            l1 = rr.vadd(l1, l1c2, mask=full16)
            l1 = rr.vadd(l1, l1c3, mask=full16)

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            next_k = next_k_register
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = l0
            hist1 = l1
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist0, next_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist1, next_k, mask=full16),
            )
            low_target = rr.vreduce_min(
                rr.vmin(candidate0, candidate1, mask=full16), mask=full16
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            low_target = rr.vdup(low_target, mask=full16)
            kth = rr.vbitwise_or(
                rr.vshl(high_target, 8, mask=full16),
                low_target,
                mask=full16,
            )
            ureg = rr.vstore_unalign_begin(tmp_idx)
            valid_chunk_count = (valid_len + 127) // 128
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.vgt(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.veq(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            rr.vstore_unalign_post(tmp_idx, 0, ureg)
            rr.vmem_bar("vst_vld")

    @jit
    def _find_indices_dispatch_scanned(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier, hist_row
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            histogram_low_reg = rr.vdups(0, dtypes.uint16, mask=full16)
            histogram_high_reg = rr.vdups(0, dtypes.uint16, mask=full16)
            h0c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            h0c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            h0c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            h0c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            full8 = rr.update_mask(256, elem_bits=8)[0]
            for block in dsl_range(valid_len // 1024, unroll=1):
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 0, width="b8"
                )
                h0c0 = rr.vhistogram_accumulate(h0c0, high, mask=full8, bin=0)
                h1c0 = rr.vhistogram_accumulate(h1c0, high, mask=full8, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 256, width="b8"
                )
                h0c1 = rr.vhistogram_accumulate(h0c1, high, mask=full8, bin=0)
                h1c1 = rr.vhistogram_accumulate(h1c1, high, mask=full8, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 512, width="b8"
                )
                h0c2 = rr.vhistogram_accumulate(h0c2, high, mask=full8, bin=0)
                h1c2 = rr.vhistogram_accumulate(h1c2, high, mask=full8, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 768, width="b8"
                )
                h0c3 = rr.vhistogram_accumulate(h0c3, high, mask=full8, bin=0)
                h1c3 = rr.vhistogram_accumulate(h1c3, high, mask=full8, bin=1)
            for chunk in dsl_range(
                (valid_len // 1024) * 4, (valid_len + 255) // 256, unroll=1
            ):
                mask8 = rr.update_mask(valid_len - chunk * 256, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                h0c0 = rr.vhistogram_accumulate(h0c0, high, mask=mask8, bin=0)
                h1c0 = rr.vhistogram_accumulate(h1c0, high, mask=mask8, bin=1)
            histogram_low_reg = rr.vadd(h0c0, h0c1, mask=full16)
            histogram_low_reg = rr.vadd(histogram_low_reg, h0c2, mask=full16)
            histogram_low_reg = rr.vadd(histogram_low_reg, h0c3, mask=full16)
            histogram_high_reg = rr.vadd(h1c0, h1c1, mask=full16)
            histogram_high_reg = rr.vadd(histogram_high_reg, h1c2, mask=full16)
            histogram_high_reg = rr.vadd(histogram_high_reg, h1c3, mask=full16)

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            bottom_k = valid_len - self.topk + 1
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = histogram_low_reg
            hist1 = histogram_high_reg
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist0, bottom_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist1, bottom_k, mask=full16),
            )
            high_target = rr.vreduce_min(
                rr.vmin(candidate0, candidate1, mask=full16), mask=full16
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=full16)
            one16 = rr.vdups(1, dtypes.uint16, mask=full16)
            bottom_k = valid_len - self.topk + 1
            high_target = rr.vdup(high_target, mask=full16)
            high_target8 = rr.vreinterpret_lanes(
                rr.vbitwise_or(
                    high_target, rr.vshl(high_target, 8, mask=full16), mask=full16
                ),
                dtypes.uint8,
            )
            prev = rr.vsub(high_target, one16, mask=full16)
            is_zero = rr.veqs(high_target, 0, mask=full16)
            prev = rr.vselect(zero16, prev, cond_mask=is_zero)
            prev_lane = rr.vbitwise_and(
                prev, rr.vdups(127, dtypes.uint16, mask=full16), mask=full16
            )
            prev0 = rr.vgather_reg(histogram_low_reg, prev_lane)
            prev1 = rr.vgather_reg(histogram_high_reg, prev_lane)
            prev_count = rr.vselect(
                prev0, prev1, cond_mask=rr.vlts(prev, 128, mask=full16)
            )
            prev_count = rr.vselect(zero16, prev_count, cond_mask=is_zero)
            next_k = rr.vsub(
                rr.vdups(bottom_k, dtypes.uint16, mask=full16),
                prev_count,
                mask=full16,
            )
            next_k_register = rr.vdup(next_k, mask=full16)

            l0c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c1 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c2 = rr.vdups(0, dtypes.uint16, mask=full16)
            l0c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1c3 = rr.vdups(0, dtypes.uint16, mask=full16)
            full8 = rr.update_mask(256, elem_bits=8)[0]
            for block in dsl_range(valid_len // 1024, unroll=1):
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 0, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
                l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 256, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c1 = rr.vhistogram_accumulate(l0c1, low, mask=eq_high, bin=0)
                l1c1 = rr.vhistogram_accumulate(l1c1, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 512, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c2 = rr.vhistogram_accumulate(l0c2, low, mask=eq_high, bin=0)
                l1c2 = rr.vhistogram_accumulate(l1c2, low, mask=eq_high, bin=1)
                low, high = rr.vload_deinterleave(
                    sort_key, block * 1024 + 768, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=full8)
                l0c3 = rr.vhistogram_accumulate(l0c3, low, mask=eq_high, bin=0)
                l1c3 = rr.vhistogram_accumulate(l1c3, low, mask=eq_high, bin=1)
            for chunk in dsl_range(
                (valid_len // 1024) * 4, (valid_len + 255) // 256, unroll=1
            ):
                mask8 = rr.update_mask(valid_len - chunk * 256, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0c0 = rr.vhistogram_accumulate(l0c0, low, mask=eq_high, bin=0)
                l1c0 = rr.vhistogram_accumulate(l1c0, low, mask=eq_high, bin=1)
            l0 = rr.vadd(l0c0, l0c1, mask=full16)
            l0 = rr.vadd(l0, l0c2, mask=full16)
            l0 = rr.vadd(l0, l0c3, mask=full16)
            l1 = rr.vadd(l1c0, l1c1, mask=full16)
            l1 = rr.vadd(l1, l1c2, mask=full16)
            l1 = rr.vadd(l1, l1c3, mask=full16)

            full16 = rr.update_mask(128, elem_bits=16)[0]
            _lane0 = rr.update_mask(1, elem_bits=16)[0]
            next_k = next_k_register
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = l0
            hist1 = l1
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist0, next_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist1, next_k, mask=full16),
            )
            low_target = rr.vreduce_min(
                rr.vmin(candidate0, candidate1, mask=full16), mask=full16
            )

            full16 = rr.update_mask(128, elem_bits=16)[0]
            low_target = rr.vdup(low_target, mask=full16)
            kth = rr.vbitwise_or(
                rr.vshl(high_target, 8, mask=full16),
                low_target,
                mask=full16,
            )
            ureg = rr.vstore_unalign_begin(tmp_idx)
            valid_chunk_count = (valid_len + 127) // 128
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.vgt(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.veq(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            rr.vstore_unalign_post(tmp_idx, 0, ureg)
            rr.vmem_bar("vst_vld")

    @jit
    def select_row(
        self,
        gm_sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
        need_values=1,
        cached_hist=0,
        hist_row=0,
    ):
        if (
            need_values == 0
            and total_tokens >= self.topk
            and total_tokens <= self.trunk_len
        ):
            merge_key = self.workspace.merge_key
            self._load_merge(
                gm_sort_key,
                self.workspace.history_key,
                merge_key,
                dtypes.int64(0),
                dtypes.int64(0),
                total_tokens,
            )
            merge_read = merge_key.next_read()
            self._find_indices_dispatch(
                merge_read,
                total_tokens,
                self.workspace.tmp_idx,
                self.workspace.histogram,
                self.workspace.high_target_carrier,
                cached_hist,
                hist_row,
            )
            out = self.output_idx_stage.next_write()
            tmp = self.workspace.tmp_idx
            with vf(mode="simd"):
                for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                    mask = rr.update_mask(self.topk - chunk * 64, elem_bits=32)[0]
                    positions = rr.vunpack(
                        rr.vload(tmp, chunk * 64), dtypes.uint32, part="lower"
                    )
                    indices = rr.vadds(
                        rr.vreinterpret(positions, dtypes.int32), index_base, mask=mask
                    )
                    valid = rr.vlts(
                        rr.varange(chunk * 64, dtypes.uint32),
                        output_valid_count,
                        mask=mask,
                    )
                    rr.vstore(
                        out,
                        chunk * 64,
                        rr.vselect(
                            indices,
                            rr.vdups(-1, dtypes.int32, mask=mask),
                            cond_mask=valid,
                        ),
                        mask,
                    )
                rr.vmem_bar("vst_vld")
            mem_copy(gm_sparse_indices, self.output_idx_stage.next_read())
        elif (
            need_values != 0
            and total_tokens >= self.topk
            and total_tokens <= self.trunk_len
        ):
            self._select_row_single_trunk(
                gm_sort_key,
                total_tokens,
                output_valid_count,
                gm_sparse_indices,
                gm_selected_value_bits,
                index_base,
                cached_hist,
                hist_row,
            )
        else:
            self._select_row_values(
                gm_sort_key,
                total_tokens,
                output_valid_count,
                gm_sparse_indices,
                gm_selected_value_bits,
                index_base,
            )

    @jit
    def _select_row_single_trunk(
        self,
        gm_sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
        cached_hist=0,
        hist_row=0,
    ):
        """PV-C8 port: single-trunk fast path WITH values.

        When the entire input fits in one trunk (total_tokens <= 16384) and
        this is the first (and only) pass, history stays empty: skip its
        zero-init, skip the next_history_key/next_idx round-trip, and gather
        value bits directly from merge_key at the squeeze positions.
        Mathematically identical to _select_row_values for this case because
        history_prefix == 0 makes the position mapping an identity.
        """
        merge_key = self.workspace.merge_key
        tmp_idx = self.workspace.tmp_idx
        histogram = self.workspace.histogram
        high_target_carrier = self.workspace.high_target_carrier
        self._load_merge(
            gm_sort_key,
            self.workspace.history_key,
            merge_key,
            dtypes.int64(0),
            dtypes.int64(0),
            total_tokens,
        )
        merge_read = merge_key.next_read()
        self._find_indices_dispatch(
            merge_read,
            total_tokens,
            tmp_idx,
            histogram,
            high_target_carrier,
            cached_hist,
            hist_row,
        )
        out_idx = self.output_idx_stage.next_write()
        out_val = self.output_value_bits_stage.next_write()
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                pos16 = rr.vload(tmp_idx, chunk * 64)
                pos32 = rr.vunpack(pos16, dtypes.uint32, part="lower")
                keys = rr.vgather(merge_read, pos16, mask=mask16)
                positive = rr.veq(
                    rr.vbitwise_and(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                value_bits = rr.vselect(
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0xFFFF, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    cond_mask=positive,
                )
                valid32 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    output_valid_count,
                    mask=mask32,
                )
                valid16 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint16),
                    output_valid_count,
                    mask=mask16,
                )
                indices = rr.vselect(
                    rr.vadds(
                        rr.vreinterpret(pos32, dtypes.int32),
                        index_base,
                        mask=mask32,
                    ),
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=valid32,
                )
                value_bits = rr.vselect(
                    value_bits,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    cond_mask=valid16,
                )
                rr.vstore(out_idx, chunk * 64, indices, mask32)
                rr.vstore(out_val, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, self.output_idx_stage.next_read())
        mem_copy(gm_selected_value_bits, self.output_value_bits_stage.next_read())

    @jit
    def _select_row_cached_ub(
        self,
        sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
        cached_hist=0,
        hist_row=0,
    ):
        """PV-C8 port: single-trunk fast path WITH values.

        When the entire input fits in one trunk (total_tokens <= 16384) and
        this is the first (and only) pass, history stays empty: skip its
        zero-init, skip the next_history_key/next_idx round-trip, and gather
        value bits directly from merge_key at the squeeze positions.
        Mathematically identical to _select_row_values for this case because
        history_prefix == 0 makes the position mapping an identity.
        """
        merge_key = sort_key
        tmp_idx = self.cache_tmp
        histogram = self.workspace.histogram
        high_target_carrier = self.workspace.high_target_carrier
        self._find_indices_dispatch(
            merge_key,
            total_tokens,
            tmp_idx,
            histogram,
            high_target_carrier,
            cached_hist,
            hist_row,
        )
        out_idx = self.cache_output_idx_stage.next_write()
        out_val = self.cache_output_value_bits_stage.next_write()
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                pos16 = rr.vload(tmp_idx, chunk * 64)
                pos32 = rr.vunpack(pos16, dtypes.uint32, part="lower")
                keys = rr.vgather(merge_key, pos16, mask=mask16)
                positive = rr.veq(
                    rr.vbitwise_and(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                value_bits = rr.vselect(
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0xFFFF, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    cond_mask=positive,
                )
                valid32 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    output_valid_count,
                    mask=mask32,
                )
                valid16 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint16),
                    output_valid_count,
                    mask=mask16,
                )
                indices = rr.vselect(
                    rr.vadds(
                        rr.vreinterpret(pos32, dtypes.int32),
                        index_base,
                        mask=mask32,
                    ),
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=valid32,
                )
                value_bits = rr.vselect(
                    value_bits,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    cond_mask=valid16,
                )
                rr.vstore(out_idx, chunk * 64, indices, mask32)
                rr.vstore(out_val, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, self.cache_output_idx_stage.next_read())
        mem_copy(gm_selected_value_bits, self.cache_output_value_bits_stage.next_read())

    @jit
    def _select_row_values(
        self,
        gm_sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
    ):
        # Rank masks use uint16 lanes; clamp the count before narrowing.
        if output_valid_count > self.topk:
            output_valid_count = self.topk
        merge_key = self.workspace.merge_key
        _current_trunk = self.workspace.current_trunk
        tmp_idx = self.workspace.tmp_idx
        histogram = self.workspace.histogram
        high_target_carrier = self.workspace.high_target_carrier
        history_key_storage = self.workspace.history_key
        history_key = history_key_storage
        next_history_key = self.workspace.next_history_key
        history_idx_storage = self.workspace.history_idx
        history_idx = history_idx_storage
        next_idx = self.workspace.next_idx
        output_idx_stage = self.output_idx_stage.next_write()
        out_value_bits = self.output_value_bits_stage.next_write()

        # Finite signed BF16 scores have positive sortable keys; zero marks
        # invalid tokens and empty history slots.
        with vf(mode="simd"):
            init_mask16 = rr.update_mask(128, elem_bits=16)[0]
            init_mask32 = rr.update_mask(64, elem_bits=32)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=init_mask16)
            zero32 = rr.vdups(0, dtypes.int32, mask=init_mask32)
            for chunk in dsl_range(ceil_div(self.topk_pad * 2, 128), unroll=1):
                rr.vstore(history_key, chunk * 128, zero16, init_mask16)
            for chunk in dsl_range(ceil_div(self.topk_pad * 2, 64), unroll=1):
                rr.vstore(history_idx, chunk * 64, zero32, init_mask32)
            rr.vmem_bar("vst_vld")

        full_trunk_count = total_tokens // self.trunk_len
        tail_len = total_tokens % self.trunk_len
        for pass_idx in dsl_range(full_trunk_count, unroll=1):
            parity = pass_idx % 2
            history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (parity, 0)
            )
            next_history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            history_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (parity, 0)
            )
            next_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            trunk_offset = pass_idx * self.trunk_len
            history_prefix = dtypes.int64((0 if pass_idx == 0 else self.topk_pad))
            self._load_merge(
                gm_sort_key,
                history_key,
                merge_key,
                trunk_offset,
                history_prefix,
                self.trunk_len,
            )
            self._select_pass(
                merge_key.next_read(),
                history_prefix + self.trunk_len,
                history_prefix,
                trunk_offset,
                tmp_idx,
                histogram,
                high_target_carrier,
                history_idx,
                next_idx,
                next_history_key,
            )

        if tail_len > 0:
            parity = full_trunk_count % 2
            history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (parity, 0)
            )
            next_history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            history_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (parity, 0)
            )
            next_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            tail_offset = full_trunk_count * self.trunk_len
            history_prefix = dtypes.int64(self.topk_pad)
            if full_trunk_count == 0 and tail_len >= self.topk:
                history_prefix = dtypes.int64(0)
            _tail_len_pad = ((tail_len + 255) // 256) * 256
            self._load_merge(
                gm_sort_key,
                history_key,
                merge_key,
                tail_offset,
                history_prefix,
                tail_len,
            )
            self._select_pass(
                merge_key.next_read(),
                history_prefix + tail_len,
                history_prefix,
                tail_offset,
                tmp_idx,
                histogram,
                high_target_carrier,
                history_idx,
                next_idx,
                next_history_key,
            )

        final_slot = (full_trunk_count + dtypes.int64((1 if tail_len > 0 else 0))) % 2
        final_idx = tile_slice(history_idx_storage, (1, self.topk_pad), (final_slot, 0))
        history_key = tile_slice(
            history_key_storage, (1, self.topk_pad), (final_slot, 0)
        )
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.mask_and(
                    rr.update_mask(count, elem_bits=16)[0],
                    rr.update_mask(64, elem_bits=16)[0],
                    exec_mask=rr.update_mask(64, elem_bits=16)[0],
                )
                indices = rr.vadds(
                    rr.vload(final_idx, chunk * 64), index_base, mask=mask32
                )
                keys = rr.vload(history_key, chunk * 64)
                positive = rr.veq(
                    rr.vbitwise_and(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                value_bits = rr.vselect(
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0xFFFF, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    cond_mask=positive,
                )
                valid_rank32 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    output_valid_count,
                    mask=mask32,
                )
                valid_rank16 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint16),
                    output_valid_count,
                    mask=mask16,
                )
                indices = rr.vselect(
                    indices,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=valid_rank32,
                )
                value_bits = rr.vselect(
                    value_bits,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    cond_mask=valid_rank16,
                )
                rr.vstore(output_idx_stage, chunk * 64, indices, mask32)
                rr.vstore(out_value_bits, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, self.output_idx_stage.next_read())
        mem_copy(gm_selected_value_bits, self.output_value_bits_stage.next_read())


class CandidateQliRawTopKWorkspace:
    """UB scratch shared by sequential Sparse and Candidate TopK stages."""

    def __init__(
        self, max_trunk_len: int, max_topk: int, candidate_optimized: bool = False
    ):
        self.candidate_optimized = bool(candidate_optimized)
        self.max_trunk_len = int(max_trunk_len)
        self.max_topk = int(max_topk)
        self.topk_pad = ceil_div(self.max_topk, 256) * 256
        self.merge_len_pad = ceil_div(self.topk_pad + self.max_trunk_len, 128) * 128
        self.current_trunk = _ub_channel(
            MemLoc.UB,
            (1, self.max_trunk_len),
            dtypes.uint16,
            depth=1,
        )
        self.merge_key = _ub_channel(
            MemLoc.UB,
            (1, self.merge_len_pad),
            dtypes.uint16,
            depth=1,
        )
        self.tmp_idx = _ub_buffer(
            MemLoc.UB,
            (self.merge_len_pad,),
            dtypes.uint16,
        )
        self.histogram = _ub_buffer(MemLoc.UB, (256,), dtypes.uint16)
        self.high_target_carrier = _ub_buffer(MemLoc.UB, (64,), dtypes.int32)
        self.history_key = _ub_buffer(
            MemLoc.UB,
            (2, self.topk_pad),
            dtypes.uint16,
        )
        self.next_history_key = _ub_channel(
            MemLoc.UB,
            (1, self.topk_pad),
            dtypes.uint16,
            depth=1,
        )
        self.history_idx = _ub_buffer(
            MemLoc.UB,
            (2, self.topk_pad),
            dtypes.int32,
        )
        self.next_idx = _ub_buffer(
            MemLoc.UB,
            (self.topk_pad,),
            dtypes.int32,
        )


class CandidateQliRawTopKSelector(_RawTopKSelectorBase):
    """ASC Arch35 uint16 radix TopK with in-AIV trunk/history merge.

    This is not LD/S2 core splitting.  One AIV processes a row sequentially in
    16K trunks.  The first trunk creates a TopK history; every later pass runs
    the same radix selector over ``history + current trunk`` and maps its local
    uint16 positions back to global int32 token indices.
    """

    def __init__(
        self,
        topk: int,
        workspace: CandidateQliRawTopKWorkspace,
    ):
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("ASC raw TopK requires positive topk")
        # The ASC VF works in fixed 16K trunks.  The number of complete trunks
        # and the final tail length are runtime values so one compiled kernel
        # can serve every supported S2.
        self.trunk_len = CANDIDATE_LD_TRUNK_LEN
        # ASC reserves history in 256-element units before the next trunk.
        self.topk_pad = ceil_div(self.topk, 256) * 256
        self.merge_len_pad = ceil_div(self.topk_pad + CANDIDATE_LD_TRUNK_LEN, 128) * 128
        self.workspace = workspace
        if (
            CANDIDATE_LD_TRUNK_LEN > workspace.max_trunk_len
            or self.topk > workspace.max_topk
            or self.merge_len_pad > workspace.merge_len_pad
        ):
            raise ValueError("TopK selector exceeds the shared UB workspace")
        # Final V->MTE3 stages are selector-owned so Sparse and Candidate can
        # share computational scratch without racing each other's GM writes.
        self.output_idx_stage = _ub_channel(
            MemLoc.UB,
            (1, self.topk),
            dtypes.int32,
            depth=1,
        )
        self.output_value_bits_stage = _ub_channel(
            MemLoc.UB,
            (1, self.topk),
            dtypes.uint16,
            depth=1,
        )

    @jit
    def _load_merge(
        self, gm_keys, history, merge, source_offset, history_prefix, length
    ):
        copied = ((length + 127) // 128) * 128
        merge_slot = merge.next_write()
        mem_copy(
            reinterpret(
                _offset_view(merge_slot, (0, history_prefix)), shape=(1, copied)
            ),
            gm_keys[0:1, source_offset : source_offset + copied],
        )
        with vf(mode="simd"):
            mask = rr.update_mask(128, elem_bits=16)[0]
            for chunk in dsl_range(history_prefix // 128, unroll=1):
                rr.vstore(merge_slot, chunk * 128, rr.vload(history, chunk * 128), mask)
            rr.vmem_bar("vst_vld")

    @jit
    def _find_indices(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier
    ):
        if self.workspace.candidate_optimized:
            self._find_indices_optimized(
                sort_key, valid_len, tmp_idx, histogram, high_target_carrier
            )
        else:
            self._find_indices_baseline(
                sort_key, valid_len, tmp_idx, histogram, high_target_carrier
            )

    @jit
    def _find_indices_baseline(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            histogram_low_reg = rr.vdups(0, dtypes.uint16, mask=full16)
            histogram_high_reg = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range((valid_len + 255) // 256, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                _, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                histogram_low_reg = rr.vhistogram_accumulate(
                    histogram_low_reg, high, mask=mask8, bin=0
                )
                histogram_high_reg = rr.vhistogram_accumulate(
                    histogram_high_reg, high, mask=mask8, bin=1
                )
            rr.vstore(histogram, 0, histogram_low_reg, full16)
            rr.vstore(histogram, 128, histogram_high_reg, full16)
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            bottom_k = valid_len - self.topk + 1
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = rr.vload(histogram, 0)
            hist1 = rr.vload(histogram, 128)
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist0, bottom_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist1, bottom_k, mask=full16),
            )
            high_target = rr.vmin(
                rr.vreduce_min(candidate0, mask=full16),
                rr.vreduce_min(candidate1, mask=full16),
                mask=lane0,
            )
            rr.vstore(
                high_target_carrier,
                0,
                rr.vreinterpret(
                    rr.vunpack(high_target, dtypes.uint32, part="lower"),
                    dtypes.int32,
                ),
                rr.update_mask(1, elem_bits=32)[0],
            )
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=full16)
            one16 = rr.vdups(1, dtypes.uint16, mask=full16)
            bottom_k = valid_len - self.topk + 1
            high_target = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b16"),
                dtypes.uint16,
            )
            high_target8 = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b8"),
                dtypes.uint8,
            )
            prev = rr.vsub(high_target, one16, mask=full16)
            is_zero = rr.veqs(high_target, 0, mask=full16)
            prev = rr.vselect(zero16, prev, cond_mask=is_zero)
            prev_count = zero16
            for base in range_constexpr(0, 256, 128):
                hv = rr.vload(histogram, base)
                idx = rr.varange(base, dtypes.uint16)
                at_prev = rr.veq(idx, prev, mask=full16)
                picked = rr.vselect(hv, zero16, cond_mask=at_prev)
                prev_count = rr.vadd(
                    prev_count,
                    rr.vreduce_max(picked, mask=full16),
                    mask=full16,
                )
            prev_count = rr.vselect(zero16, prev_count, cond_mask=is_zero)
            next_k = rr.vsub(
                rr.vdups(bottom_k, dtypes.uint16, mask=full16),
                prev_count,
                mask=full16,
            )
            rr.vstore(tmp_idx, 0, next_k, lane0)

            l0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1 = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range((valid_len + 255) // 256, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0 = rr.vhistogram_accumulate(l0, low, mask=eq_high, bin=0)
                l1 = rr.vhistogram_accumulate(l1, low, mask=eq_high, bin=1)
            rr.vstore(histogram, 0, l0, full16)
            rr.vstore(histogram, 128, l1, full16)
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            next_k = rr.vload_broadcast(tmp_idx, 0)
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = rr.vload(histogram, 0)
            hist1 = rr.vload(histogram, 128)
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist0, next_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist1, next_k, mask=full16),
            )
            low_target = rr.vmin(
                rr.vreduce_min(candidate0, mask=full16),
                rr.vreduce_min(candidate1, mask=full16),
                mask=lane0,
            )
            rr.vstore(tmp_idx, 0, low_target, lane0)
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            high_target = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b16"),
                dtypes.uint16,
            )
            low_target = rr.vload_broadcast(tmp_idx, 0)
            kth = rr.vbitwise_or(
                rr.vshl(high_target, 8, mask=full16),
                low_target,
                mask=full16,
            )
            ureg = rr.vstore_unalign_begin(tmp_idx)
            valid_chunk_count = (valid_len + 127) // 128
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.vgt(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.veq(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            rr.vstore_unalign_post(tmp_idx, 0, ureg)
            rr.vmem_bar("vst_vld")

    @jit
    def _find_indices_optimized(
        self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier
    ):
        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            h0a = rr.vdups(0, dtypes.uint16, mask=full16)
            h1a = rr.vdups(0, dtypes.uint16, mask=full16)
            h0b = rr.vdups(0, dtypes.uint16, mask=full16)
            h1b = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range(0, (valid_len + 255) // 256, 2, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                _, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                h0a = rr.vhistogram_accumulate(h0a, high, mask=mask8, bin=0)
                h1a = rr.vhistogram_accumulate(h1a, high, mask=mask8, bin=1)
            for chunk in dsl_range(1, (valid_len + 255) // 256, 2, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                _, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                h0b = rr.vhistogram_accumulate(h0b, high, mask=mask8, bin=0)
                h1b = rr.vhistogram_accumulate(h1b, high, mask=mask8, bin=1)
            histogram_low_reg = rr.vadd(h0a, h0b, mask=full16)
            histogram_high_reg = rr.vadd(h1a, h1b, mask=full16)
            rr.vstore(histogram, 0, histogram_low_reg, full16)
            rr.vstore(histogram, 128, histogram_high_reg, full16)
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            bottom_k = valid_len - self.topk + 1
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = rr.vload(histogram, 0)
            hist1 = rr.vload(histogram, 128)
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist0, bottom_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist1, bottom_k, mask=full16),
            )
            high_target = rr.vmin(
                rr.vreduce_min(candidate0, mask=full16),
                rr.vreduce_min(candidate1, mask=full16),
                mask=lane0,
            )
            rr.vstore(
                high_target_carrier,
                0,
                rr.vreinterpret(
                    rr.vunpack(high_target, dtypes.uint32, part="lower"),
                    dtypes.int32,
                ),
                rr.update_mask(1, elem_bits=32)[0],
            )
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=full16)
            one16 = rr.vdups(1, dtypes.uint16, mask=full16)
            bottom_k = valid_len - self.topk + 1
            high_target = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b16"),
                dtypes.uint16,
            )
            high_target8 = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b8"),
                dtypes.uint8,
            )
            prev = rr.vsub(high_target, one16, mask=full16)
            is_zero = rr.veqs(high_target, 0, mask=full16)
            prev = rr.vselect(zero16, prev, cond_mask=is_zero)
            prev_count = zero16
            for base in range_constexpr(0, 256, 128):
                hv = rr.vload(histogram, base)
                idx = rr.varange(base, dtypes.uint16)
                at_prev = rr.veq(idx, prev, mask=full16)
                picked = rr.vselect(hv, zero16, cond_mask=at_prev)
                prev_count = rr.vadd(
                    prev_count,
                    rr.vreduce_max(picked, mask=full16),
                    mask=full16,
                )
            prev_count = rr.vselect(zero16, prev_count, cond_mask=is_zero)
            next_k = rr.vsub(
                rr.vdups(bottom_k, dtypes.uint16, mask=full16),
                prev_count,
                mask=full16,
            )
            rr.vstore(tmp_idx, 0, next_k, lane0)

            l0a = rr.vdups(0, dtypes.uint16, mask=full16)
            l1a = rr.vdups(0, dtypes.uint16, mask=full16)
            l0b = rr.vdups(0, dtypes.uint16, mask=full16)
            l1b = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range(0, (valid_len + 255) // 256, 2, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0a = rr.vhistogram_accumulate(l0a, low, mask=eq_high, bin=0)
                l1a = rr.vhistogram_accumulate(l1a, low, mask=eq_high, bin=1)
            for chunk in dsl_range(1, (valid_len + 255) // 256, 2, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0b = rr.vhistogram_accumulate(l0b, low, mask=eq_high, bin=0)
                l1b = rr.vhistogram_accumulate(l1b, low, mask=eq_high, bin=1)
            l0 = rr.vadd(l0a, l0b, mask=full16)
            l1 = rr.vadd(l1a, l1b, mask=full16)
            rr.vstore(histogram, 0, l0, full16)
            rr.vstore(histogram, 128, l1, full16)
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            next_k = rr.vload_broadcast(tmp_idx, 0)
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = rr.vload(histogram, 0)
            hist1 = rr.vload(histogram, 128)
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist0, next_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist1, next_k, mask=full16),
            )
            low_target = rr.vmin(
                rr.vreduce_min(candidate0, mask=full16),
                rr.vreduce_min(candidate1, mask=full16),
                mask=lane0,
            )
            rr.vstore(tmp_idx, 0, low_target, lane0)
            rr.vmem_bar("vst_vld")

        with vf(mode="simd"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            high_target = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b16"),
                dtypes.uint16,
            )
            low_target = rr.vload_broadcast(tmp_idx, 0)
            kth = rr.vbitwise_or(
                rr.vshl(high_target, 8, mask=full16),
                low_target,
                mask=full16,
            )
            ureg = rr.vstore_unalign_begin(tmp_idx)
            valid_chunk_count = (valid_len + 127) // 128
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.vgt(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.veq(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            rr.vstore_unalign_post(tmp_idx, 0, ureg)
            rr.vmem_bar("vst_vld")

    @jit
    def select_row(
        self,
        gm_sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
        need_values=1,
    ):
        if (
            need_values == 0
            and total_tokens >= self.topk
            and total_tokens <= self.trunk_len
        ):
            merge_key = self.workspace.merge_key
            self._load_merge(
                gm_sort_key,
                self.workspace.history_key,
                merge_key,
                dtypes.int64(0),
                dtypes.int64(0),
                total_tokens,
            )
            self._find_indices(
                merge_key.next_read(),
                total_tokens,
                self.workspace.tmp_idx,
                self.workspace.histogram,
                self.workspace.high_target_carrier,
            )
            out = self.output_idx_stage.next_write()
            tmp = self.workspace.tmp_idx
            with vf(mode="simd"):
                for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                    mask = rr.update_mask(self.topk - chunk * 64, elem_bits=32)[0]
                    positions = rr.vunpack(
                        rr.vload(tmp, chunk * 64), dtypes.uint32, part="lower"
                    )
                    indices = rr.vadds(
                        rr.vreinterpret(positions, dtypes.int32), index_base, mask=mask
                    )
                    valid = rr.vlts(
                        rr.varange(chunk * 64, dtypes.uint32),
                        output_valid_count,
                        mask=mask,
                    )
                    rr.vstore(
                        out,
                        chunk * 64,
                        rr.vselect(
                            indices,
                            rr.vdups(-1, dtypes.int32, mask=mask),
                            cond_mask=valid,
                        ),
                        mask,
                    )
                rr.vmem_bar("vst_vld")
            mem_copy(gm_sparse_indices, self.output_idx_stage.next_read())
        elif (
            self.workspace.candidate_optimized
            and need_values != 0
            and total_tokens >= self.topk
            and total_tokens <= self.trunk_len
        ):
            self._select_row_single_trunk(
                gm_sort_key,
                total_tokens,
                output_valid_count,
                gm_sparse_indices,
                gm_selected_value_bits,
                index_base,
            )
        else:
            self._select_row_values(
                gm_sort_key,
                total_tokens,
                output_valid_count,
                gm_sparse_indices,
                gm_selected_value_bits,
                index_base,
            )

    @jit
    def _select_row_single_trunk(
        self,
        gm_sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
    ):
        """PV-C8 port: single-trunk fast path WITH values.

        When the entire input fits in one trunk (total_tokens <= 16384) and
        this is the first (and only) pass, history stays empty: skip its
        zero-init, skip the next_history_key/next_idx round-trip, and gather
        value bits directly from merge_key at the squeeze positions.
        Mathematically identical to _select_row_values for this case because
        history_prefix == 0 makes the position mapping an identity.
        """
        merge_key = self.workspace.merge_key
        tmp_idx = self.workspace.tmp_idx
        histogram = self.workspace.histogram
        high_target_carrier = self.workspace.high_target_carrier
        self._load_merge(
            gm_sort_key,
            self.workspace.history_key,
            merge_key,
            dtypes.int64(0),
            dtypes.int64(0),
            total_tokens,
        )
        merge_read = merge_key.next_read()
        self._find_indices(
            merge_read, total_tokens, tmp_idx, histogram, high_target_carrier
        )
        out_idx = self.output_idx_stage.next_write()
        out_val = self.output_value_bits_stage.next_write()
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                pos16 = rr.vload(tmp_idx, chunk * 64)
                pos32 = rr.vunpack(pos16, dtypes.uint32, part="lower")
                keys = rr.vgather(merge_read, pos16, mask=mask16)
                positive = rr.veq(
                    rr.vbitwise_and(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                value_bits = rr.vselect(
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0xFFFF, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    cond_mask=positive,
                )
                valid32 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    output_valid_count,
                    mask=mask32,
                )
                valid16 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint16),
                    output_valid_count,
                    mask=mask16,
                )
                indices = rr.vselect(
                    rr.vadds(
                        rr.vreinterpret(pos32, dtypes.int32),
                        index_base,
                        mask=mask32,
                    ),
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=valid32,
                )
                value_bits = rr.vselect(
                    value_bits,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    cond_mask=valid16,
                )
                rr.vstore(out_idx, chunk * 64, indices, mask32)
                rr.vstore(out_val, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, self.output_idx_stage.next_read())
        mem_copy(gm_selected_value_bits, self.output_value_bits_stage.next_read())

    @jit
    def _select_row_values(
        self,
        gm_sort_key: Tensor,
        total_tokens,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
        index_base,
    ):
        # Rank masks use uint16 lanes; clamp the count before narrowing.
        if output_valid_count > self.topk:
            output_valid_count = self.topk
        merge_key = self.workspace.merge_key
        _current_trunk = self.workspace.current_trunk
        tmp_idx = self.workspace.tmp_idx
        histogram = self.workspace.histogram
        high_target_carrier = self.workspace.high_target_carrier
        history_key_storage = self.workspace.history_key
        history_key = history_key_storage
        next_history_key = self.workspace.next_history_key
        history_idx_storage = self.workspace.history_idx
        history_idx = history_idx_storage
        next_idx = self.workspace.next_idx
        output_idx_stage = self.output_idx_stage.next_write()
        out_value_bits = self.output_value_bits_stage.next_write()

        # Finite signed BF16 scores have positive sortable keys; zero marks
        # invalid tokens and empty history slots.
        with vf(mode="simd"):
            init_mask16 = rr.update_mask(128, elem_bits=16)[0]
            init_mask32 = rr.update_mask(64, elem_bits=32)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=init_mask16)
            zero32 = rr.vdups(0, dtypes.int32, mask=init_mask32)
            for chunk in dsl_range(ceil_div(self.topk_pad * 2, 128), unroll=1):
                rr.vstore(history_key, chunk * 128, zero16, init_mask16)
            for chunk in dsl_range(ceil_div(self.topk_pad * 2, 64), unroll=1):
                rr.vstore(history_idx, chunk * 64, zero32, init_mask32)
            rr.vmem_bar("vst_vld")

        full_trunk_count = total_tokens // CANDIDATE_LD_TRUNK_LEN
        tail_len = total_tokens % CANDIDATE_LD_TRUNK_LEN
        for pass_idx in dsl_range(full_trunk_count, unroll=1):
            parity = pass_idx % 2
            history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (parity, 0)
            )
            next_history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            history_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (parity, 0)
            )
            next_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            trunk_offset = pass_idx * self.trunk_len
            history_prefix = dtypes.int64((0 if pass_idx == 0 else self.topk_pad))
            self._load_merge(
                gm_sort_key,
                history_key,
                merge_key,
                trunk_offset,
                history_prefix,
                self.trunk_len,
            )
            self._select_pass(
                merge_key.next_read(),
                history_prefix + self.trunk_len,
                history_prefix,
                trunk_offset,
                tmp_idx,
                histogram,
                high_target_carrier,
                history_idx,
                next_idx,
                next_history_key,
            )

        if tail_len > 0:
            parity = full_trunk_count % 2
            history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (parity, 0)
            )
            next_history_key = tile_slice(
                history_key_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            history_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (parity, 0)
            )
            next_idx = tile_slice(
                history_idx_storage, (1, self.topk_pad), (1 - parity, 0)
            )
            tail_offset = full_trunk_count * self.trunk_len
            history_prefix = dtypes.int64(self.topk_pad)
            if full_trunk_count == 0 and tail_len >= self.topk:
                history_prefix = dtypes.int64(0)
            _tail_len_pad = ((tail_len + 255) // 256) * 256
            self._load_merge(
                gm_sort_key,
                history_key,
                merge_key,
                tail_offset,
                history_prefix,
                tail_len,
            )
            self._select_pass(
                merge_key.next_read(),
                history_prefix + tail_len,
                history_prefix,
                tail_offset,
                tmp_idx,
                histogram,
                high_target_carrier,
                history_idx,
                next_idx,
                next_history_key,
            )

        final_slot = (full_trunk_count + dtypes.int64((1 if tail_len > 0 else 0))) % 2
        final_idx = tile_slice(history_idx_storage, (1, self.topk_pad), (final_slot, 0))
        history_key = tile_slice(
            history_key_storage, (1, self.topk_pad), (final_slot, 0)
        )
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.mask_and(
                    rr.update_mask(count, elem_bits=16)[0],
                    rr.update_mask(64, elem_bits=16)[0],
                    exec_mask=rr.update_mask(64, elem_bits=16)[0],
                )
                indices = rr.vadds(
                    rr.vload(final_idx, chunk * 64), index_base, mask=mask32
                )
                keys = rr.vload(history_key, chunk * 64)
                positive = rr.veq(
                    rr.vbitwise_and(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                value_bits = rr.vselect(
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0xFFFF, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    cond_mask=positive,
                )
                valid_rank32 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    output_valid_count,
                    mask=mask32,
                )
                valid_rank16 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint16),
                    output_valid_count,
                    mask=mask16,
                )
                indices = rr.vselect(
                    indices,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=valid_rank32,
                )
                value_bits = rr.vselect(
                    value_bits,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    cond_mask=valid_rank16,
                )
                rr.vstore(output_idx_stage, chunk * 64, indices, mask32)
                rr.vstore(out_value_bits, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, self.output_idx_stage.next_read())
        mem_copy(gm_selected_value_bits, self.output_value_bits_stage.next_read())


class CandidateLdTopKSelector(CandidateQliRawTopKSelector):
    def __init__(self, topk, workspace, splits):
        super().__init__(topk, workspace)
        self.ld_length = int(splits) * self.topk
        self.ld_indices = _ub_channel(
            MemLoc.UB,
            (1, self.ld_length),
            dtypes.int32,
            depth=1,
        )
        self.ld_bits = _ub_channel(
            MemLoc.UB,
            (1, self.ld_length),
            dtypes.uint16,
            depth=1,
        )

    @jit
    def merge_ld(
        self,
        partial_indices,
        partial_bits,
        gm_sparse_indices,
        gm_selected_value_bits,
        valid_length,
        output_offset=0,
        preloaded=False,
        emit_gm=1,
    ):
        if not preloaded:
            mem_copy(
                reinterpret(self.ld_indices.next_write(), shape=(1, valid_length)),
                partial_indices[0:1, 0:valid_length],
            )
            mem_copy(
                reinterpret(self.ld_bits.next_write(), shape=(1, valid_length)),
                partial_bits[0:1, 0:valid_length],
            )
        current_trunk = self.workspace.merge_key
        trunk_write = current_trunk.next_write()
        _history_key = self.workspace.history_key
        history_idx = self.workspace.history_idx
        ld_idx_read = self.ld_indices.next_read()
        ld_bits_read = self.ld_bits.next_read()
        with vf(mode="simd"):
            mask16 = rr.update_mask(64, elem_bits=16)[0]
            mask32 = rr.update_mask(64, elem_bits=32)[0]
            # LD may merge fewer than 64 entries for a small TopK capacity.
            for chunk in dsl_range((valid_length + 63) // 64, unroll=1):
                count = min(64, valid_length - chunk * 64)
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                bits = rr.vload(ld_bits_read, chunk * 64)
                indices = rr.vload(ld_idx_read, chunk * 64)
                positive_key = rr.vbitwise_xor(
                    bits, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
                )
                negative_key = rr.vbitwise_xor(
                    bits, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16
                )
                negative = rr.veqs(
                    rr.vbitwise_and(
                        bits, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16
                    ),
                    0x8000,
                    mask=mask16,
                )
                keys = rr.vselect(negative_key, positive_key, cond_mask=negative)
                valid32 = rr.vselect(
                    rr.vdups(1, dtypes.int32, mask=mask32),
                    rr.vdups(0, dtypes.int32, mask=mask32),
                    cond_mask=rr.vges(indices, 0, mask=mask32),
                )
                valid16 = rr.vpack(
                    rr.vreinterpret(valid32, dtypes.uint32), dtypes.uint16, part="lower"
                )
                keys = rr.vselect(
                    keys,
                    rr.vdups(0, dtypes.uint16, mask=mask16),
                    cond_mask=rr.veqs(valid16, 1, mask=mask16),
                )
                rr.vstore(trunk_write, chunk * 64, keys, mask16)
            rr.vmem_bar("vst_vld")
        trunk_read = current_trunk.next_read()
        self._find_indices(
            trunk_read,
            valid_length,
            self.workspace.tmp_idx,
            self.workspace.histogram,
            self.workspace.high_target_carrier,
        )
        output_idx_stage = self.output_idx_stage.next_write()
        out_value_bits = self.output_value_bits_stage.next_write()
        output_valid_count = dtypes.int32(self.topk)
        _final_idx = history_idx
        with vf(mode="simd"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.mask_and(
                    rr.update_mask(count, elem_bits=16)[0],
                    rr.update_mask(64, elem_bits=16)[0],
                    exec_mask=rr.update_mask(64, elem_bits=16)[0],
                )
                positions = rr.vunpack(
                    rr.vload(self.workspace.tmp_idx, chunk * 64),
                    dtypes.uint32,
                    part="lower",
                )
                indices = rr.vgather(
                    ld_idx_read, rr.vreinterpret(positions, dtypes.uint32), mask=mask32
                )
                positions16 = rr.vload(self.workspace.tmp_idx, chunk * 64)
                keys = rr.vgather(trunk_read, positions16, mask=mask16)
                value_bits = rr.vgather(ld_bits_read, positions16, mask=mask16)
                _valid_rank32 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    output_valid_count,
                    mask=mask32,
                )
                _valid_rank16 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint16),
                    output_valid_count,
                    mask=mask16,
                )
                key_valid16 = rr.vselect(
                    rr.vdups(1, dtypes.uint16, mask=mask16),
                    rr.vdups(0, dtypes.uint16, mask=mask16),
                    cond_mask=rr.vges(keys, 1, mask=mask16),
                )
                key_valid32 = rr.vunpack(key_valid16, dtypes.uint32, part="lower")
                indices = rr.vselect(
                    indices,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=rr.veqs(key_valid32, 1, mask=mask32),
                )
                valid_flags32 = rr.vselect(
                    rr.vdups(1, dtypes.int32, mask=mask32),
                    rr.vdups(0, dtypes.int32, mask=mask32),
                    cond_mask=rr.vges(indices, 0, mask=mask32),
                )
                valid_flags16 = rr.vpack(
                    rr.vreinterpret(valid_flags32, dtypes.uint32),
                    dtypes.uint16,
                    part="lower",
                )
                indices = rr.vselect(
                    indices,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=rr.vges(indices, 0, mask=mask32),
                )
                value_bits = rr.vselect(
                    value_bits,
                    rr.vdups(BF16_NEG_INF_BITS, dtypes.uint16, mask=mask16),
                    cond_mask=rr.veqs(valid_flags16, 1, mask=mask16),
                )
                indices = rr.vselect(
                    rr.vadds(indices, output_offset, mask=mask32),
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=rr.veqs(valid_flags32, 1, mask=mask32),
                )
                rr.vstore(output_idx_stage, chunk * 64, indices, mask32)
                rr.vstore(out_value_bits, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")
        # Intermediate LD rounds keep the carry in the on-core output stages;
        # only the final round drains them to GM.
        if emit_gm != 0:
            mem_copy(gm_sparse_indices, self.output_idx_stage.next_read())
            mem_copy(gm_selected_value_bits, self.output_value_bits_stage.next_read())


class Sort512:
    """对 512 个 token ID 升序排序，保留关联值位和尾部无效项。"""

    def __init__(self):
        self.a = Buffer(MemLoc.UB, dtypes.int32, make_layout((512,)))
        self.b = Buffer(MemLoc.UB, dtypes.int32, make_layout((512,)))
        self.va = Buffer(MemLoc.UB, dtypes.uint32, make_layout((512,)))
        self.vb = Buffer(MemLoc.UB, dtypes.uint32, make_layout((512,)))
        self.iv = RingBuffer(MemLoc.UB, dtypes.int32, make_layout((1, 512)), depth=2)
        self.bv = RingBuffer(MemLoc.UB, dtypes.uint16, make_layout((1, 512)), depth=2)

    @jit
    def run(self, indices, bits):
        mem_copy(reinterpret(self.a, shape=(1, 512)), indices)
        bv = self.bv.next_write()
        mem_copy(bv, bits)
        val = self.bv.next_read()
        with vf(mode="simd"):
            mask = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range(8, unroll=1):
                base = chunk * 64
                ids = rr.vload(self.a, base)
                ids = rr.vselect(
                    ids,
                    rr.vdups(2147483647, dtypes.int32, mask=mask),
                    cond_mask=rr.vges(ids, 0, mask=mask),
                )
                rr.vstore(self.a, base, ids, mask)
                rr.vstore(
                    self.va,
                    base,
                    rr.vunpack(rr.vload(val, base), dtypes.uint32, part="lower"),
                    mask,
                )
            rr.vmem_bar("vst_vld")
        current = self.a
        alternate = self.b
        values = self.va
        other = self.vb
        for level in range_constexpr(1, 10):
            for step in range_constexpr(level):
                distance = 1 << level - 1 - step
                with vf(mode="simd"):
                    mask = rr.update_mask(64, elem_bits=32)[0]
                    for chunk in dsl_range(8, unroll=1):
                        base = chunk * 64
                        lane = rr.varange(base, dtypes.uint32)
                        peer_lane = rr.vbitwise_xor(
                            lane,
                            rr.vdups(distance, dtypes.uint32, mask=mask),
                            mask=mask,
                        )
                        own = rr.vload(current, base)
                        peer = rr.vgather(current, peer_lane, mask=mask)
                        asc = rr.veqs(
                            rr.vbitwise_and(
                                lane,
                                rr.vdups(1 << level, dtypes.uint32, mask=mask),
                                mask=mask,
                            ),
                            0,
                            mask=mask,
                        )
                        lower = rr.veqs(
                            rr.vbitwise_and(
                                lane,
                                rr.vdups(distance, dtypes.uint32, mask=mask),
                                mask=mask,
                            ),
                            0,
                            mask=mask,
                        )
                        ascv = rr.vselect(
                            rr.vdups(1, dtypes.uint32, mask=mask),
                            rr.vdups(0, dtypes.uint32, mask=mask),
                            cond_mask=asc,
                        )
                        lowv = rr.vselect(
                            rr.vdups(1, dtypes.uint32, mask=mask),
                            rr.vdups(0, dtypes.uint32, mask=mask),
                            cond_mask=lower,
                        )
                        take_low = rr.veq(ascv, lowv, mask=mask)
                        low = rr.vmin(own, peer, mask=mask)
                        high = rr.vmax(own, peer, mask=mask)
                        result = rr.vselect(low, high, cond_mask=take_low)
                        rr.vstore(alternate, base, result, mask)
                        rr.vstore(
                            other,
                            base,
                            rr.vselect(
                                rr.vload(values, base),
                                rr.vgather(values, peer_lane, mask=mask),
                                cond_mask=rr.veq(result, own, mask=mask),
                            ),
                            mask,
                        )
                    rr.vmem_bar("vst_vld")
                current, alternate = (alternate, current)
                values, other = (other, values)
        out = self.iv.next_write()
        valout = self.bv.next_write()
        with vf(mode="simd"):
            mask = rr.update_mask(64, elem_bits=32)[0]
            mask16 = rr.update_mask(64, elem_bits=16)[0]
            for chunk in dsl_range(8, unroll=1):
                base = chunk * 64
                ids = rr.vload(current, base)
                rr.vstore(
                    out,
                    base,
                    rr.vselect(
                        ids,
                        rr.vdups(-1, dtypes.int32, mask=mask),
                        cond_mask=rr.vlts(ids, 2147483647, mask=mask),
                    ),
                    mask,
                )
                rr.vstore(
                    valout,
                    base,
                    rr.vpack(rr.vload(values, base), dtypes.uint16, part="lower"),
                    mask16,
                )
            rr.vmem_bar("vst_vld")
        mem_copy(indices, self.iv.next_read())
        mem_copy(bits, self.bv.next_read())

    @jit
    def run_indices(self, indices):
        mem_copy(reinterpret(self.a, shape=(1, 512)), indices)
        with vf(mode="simd"):
            mask = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range(8, unroll=1):
                base = chunk * 64
                ids = rr.vload(self.a, base)
                ids = rr.vselect(
                    ids,
                    rr.vdups(2147483647, dtypes.int32, mask=mask),
                    cond_mask=rr.vges(ids, 0, mask=mask),
                )
                rr.vstore(self.a, base, ids, mask)
            rr.vmem_bar("vst_vld")
        current = self.a
        alternate = self.b
        for level in range_constexpr(1, 10):
            for step in range_constexpr(level):
                distance = 1 << level - 1 - step
                with vf(mode="simd"):
                    mask = rr.update_mask(64, elem_bits=32)[0]
                    for chunk in dsl_range(8, unroll=1):
                        base = chunk * 64
                        lane = rr.varange(base, dtypes.uint32)
                        peer_lane = rr.vbitwise_xor(
                            lane,
                            rr.vdups(distance, dtypes.uint32, mask=mask),
                            mask=mask,
                        )
                        own = rr.vload(current, base)
                        peer = rr.vgather(current, peer_lane, mask=mask)
                        asc = rr.veqs(
                            rr.vbitwise_and(
                                lane,
                                rr.vdups(1 << level, dtypes.uint32, mask=mask),
                                mask=mask,
                            ),
                            0,
                            mask=mask,
                        )
                        lower = rr.veqs(
                            rr.vbitwise_and(
                                lane,
                                rr.vdups(distance, dtypes.uint32, mask=mask),
                                mask=mask,
                            ),
                            0,
                            mask=mask,
                        )
                        ascv = rr.vselect(
                            rr.vdups(1, dtypes.uint32, mask=mask),
                            rr.vdups(0, dtypes.uint32, mask=mask),
                            cond_mask=asc,
                        )
                        lowv = rr.vselect(
                            rr.vdups(1, dtypes.uint32, mask=mask),
                            rr.vdups(0, dtypes.uint32, mask=mask),
                            cond_mask=lower,
                        )
                        take_low = rr.veq(ascv, lowv, mask=mask)
                        low = rr.vmin(own, peer, mask=mask)
                        high = rr.vmax(own, peer, mask=mask)
                        result = rr.vselect(low, high, cond_mask=take_low)
                        rr.vstore(alternate, base, result, mask)
                    rr.vmem_bar("vst_vld")
                current, alternate = (alternate, current)
        out = self.iv.next_write()
        with vf(mode="simd"):
            mask = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range(8, unroll=1):
                base = chunk * 64
                ids = rr.vload(current, base)
                rr.vstore(
                    out,
                    base,
                    rr.vselect(
                        ids,
                        rr.vdups(-1, dtypes.int32, mask=mask),
                        cond_mask=rr.vlts(ids, 2147483647, mask=mask),
                    ),
                    mask,
                )
            rr.vmem_bar("vst_vld")
        mem_copy(indices, self.iv.next_read())


# 仅描述启动ABI，不构造通用QLI设备kernel。
def _qli_specs(
    topk,
    candidate_enabled,
    candidate_topk,
    mask_mode,
    cmp_ratio,
    has_cu_seqlens_q,
    has_seqused_q,
    has_seqused_k,
    has_block_table,
    pa_block_size,
    split_count,
    n1=32,
    query_tile_rows=6,
    elide_candidate_scores=False,
):
    query_head_count = int(n1)
    query_tile_m = query_head_count * int(query_tile_rows)
    has_cmp_residual_k = int(mask_mode) == 3 and int(cmp_ratio) != 1
    topk = int(topk)
    candidate_enabled = bool(candidate_enabled)
    candidate_topk = int(candidate_topk)
    mask_mode = int(mask_mode)
    cmp_ratio = int(cmp_ratio)
    query_rows_dim = cannbotdsl.Dim("T1")
    batch_dim = cannbotdsl.Dim("B")
    block_table_width_dim = cannbotdsl.Dim("MAX_BLOCKS")
    token_capacity_dim = cannbotdsl.Dim("S2_PAD")
    candidate_capacity_dim = cannbotdsl.Dim("CANDIDATE_PAD")
    _sparse_width = cannbotdsl.Dim("SPARSE_WIDTH")
    candidate_width = cannbotdsl.Dim("CANDIDATE_WIDTH")
    partial_rows = cannbotdsl.Dim("PARTIAL_ROWS")
    _metadata_tasks = cannbotdsl.Dim("METADATA_TASKS")
    workspace_rows = MAX_CUBE_WORKERS * 2 * (query_tile_m // query_head_count)
    fake = TensorSpec
    dummy_i32_spec = fake((1,), dtypes.int32)
    cu_spec = (
        fake((batch_dim + 1,), dtypes.int32) if has_cu_seqlens_q else dummy_i32_spec
    )
    seqused_q_spec = (
        fake((batch_dim,), dtypes.int32) if has_seqused_q else dummy_i32_spec
    )
    seqused_k_spec = (
        fake((batch_dim,), dtypes.int32) if has_seqused_k else dummy_i32_spec
    )
    residual_spec = (
        fake((batch_dim,), dtypes.int32) if has_cmp_residual_k else dummy_i32_spec
    )
    block_table_spec = (
        fake((batch_dim, block_table_width_dim), dtypes.int32)
        if has_block_table
        else fake((1, 1), dtypes.int32)
    )
    candidate_key_spec = (
        fake((workspace_rows, candidate_capacity_dim), dtypes.uint16)
        if candidate_enabled
        else fake((workspace_rows, 1), dtypes.uint16)
    )
    _candidate_index_spec = (
        fake((query_rows_dim, candidate_width), dtypes.int32)
        if candidate_enabled
        else fake((1, candidate_width), dtypes.int32)
    )
    return (
        dtypes.int64,
        dtypes.int64,
        fake((query_rows_dim, query_head_count), dtypes.float32),
        fake((query_rows_dim * query_head_count, 2, 2), dtypes.float8_e8m0),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        cu_spec,
        seqused_q_spec,
        seqused_k_spec,
        residual_spec,
        block_table_spec,
        fake((workspace_rows, token_capacity_dim), dtypes.uint16),
        candidate_key_spec,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        fake((partial_rows,), dtypes.int32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        fake((METADATA_ROWS, METADATA_FIELDS), dtypes.int32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
    )


def _validate_qli_pa_sequences(
    query,
    cu_seqlens_q,
    seqused_q,
    seqused_k,
    cmp_residual_k,
    block_table,
    output_idx_offset,
    mask_mode,
    cmp_ratio,
):
    """校验 PA 分页路径的序列边界、输出偏移与页表元信息。"""
    t = int(query.shape[0])
    if t <= 0:
        raise ValueError("Q must contain at least one T1 row")
    has_cu_seqlens_q = cu_seqlens_q is not None
    batch = int(cu_seqlens_q.numel()) - 1 if has_cu_seqlens_q else 1
    if batch <= 0:
        raise ValueError("Cu_seqlens_q must have shape (B+1,)")

    def validate_metadata(name, tensor, shape):
        if tensor is None:
            return
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {tuple(shape)}")
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be int32")
        if tensor.device != query.device:
            raise ValueError(f"{name} must be on the same NPU as q")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if has_cu_seqlens_q:
        validate_metadata("cu_seqlens_q", cu_seqlens_q, (batch + 1,))
    validate_metadata("output_idx_offset", output_idx_offset, (t, 1))
    validate_metadata("seqused_q", seqused_q, (batch,))
    validate_metadata("seqused_k", seqused_k, (batch,))
    if int(mask_mode) == 3 and int(cmp_ratio) != 1:
        if cmp_residual_k is None:
            raise ValueError(
                "Cmp_residual_k is required for mask_mode=3 and cmp_ratio!=1"
            )
        validate_metadata("cmp_residual_k", cmp_residual_k, (batch,))
    elif cmp_residual_k is not None:
        raise ValueError(
            "Cmp_residual_k must be None unless mask_mode=3 and cmp_ratio!=1"
        )
    if block_table is not None:
        if block_table.ndim != 2 or int(block_table.shape[0]) != batch:
            raise ValueError("Block_table must have shape (B,max_num_blocks_per_seq)")
        if block_table.dtype != torch.int32:
            raise TypeError("Block_table must be int32")
        if block_table.device != query.device:
            raise ValueError("Block_table must be on the same NPU as q")
        if not block_table.is_contiguous():
            raise ValueError("Block_table must be contiguous")

    return t, batch, has_cu_seqlens_q


def _validate_qli_candidate(candidate_topk_blocks, candidate_block_size):
    """Candidate 关闭与开启时分别校验配套属性。"""
    if candidate_topk_blocks == -1:
        if candidate_block_size != -1:
            raise ValueError(
                "Candidate_block_size must be -1 when candidate is disabled"
            )
    elif candidate_topk_blocks <= 0 or candidate_block_size != CANDIDATE_BLOCK_SIZE:
        raise ValueError("Candidate requires positive topk_blocks and block_size=8")


def _validate_qli_compute_attributes(topk, mask_mode, cmp_ratio):
    """校验选择容量、掩码模式与压缩比例的支持范围。"""
    if int(topk) < 1 or int(topk) > 8192:
        raise ValueError("Topk must be in ASC-supported range [1,8192]")
    if int(mask_mode) not in (0, 3):
        raise ValueError("Mask_mode must be 0 or 3")
    if int(cmp_ratio) < 1 or int(cmp_ratio) > 128:
        raise ValueError("Cmp_ratio must be in [1,128]")


def prepare_qli(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    descale_q: torch.Tensor,
    descale_k: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    *,
    topk: int,
    quant_mode: int,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_k: str = "TND",
    return_value: bool = False,
    candidate_topk_blocks: int = -1,
    candidate_block_size: int = -1,
    _decode_seq: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Prepare PA views and workspace; Metadata remains an opaque ABI slot."""
    if q.ndim != 3 or tuple(q.shape[1:]) != (32, 64):
        raise ValueError("QLI requires N1=32 and logical D=128")
    if metadata is None:
        raise ValueError(
            "Metadata carrier must be provided by the fused preparation layer"
        )
    query_heads = int(q.shape[1])
    query_tile_rows = min(QUERY_TILE_ROWS, 256 // query_heads)
    tile_m = query_heads * query_tile_rows
    query, key, weights = q, k, w
    query_dequant_scale, key_dequant_scale = descale_q, descale_k
    if int(quant_mode) != 1:
        raise ValueError(
            "Quant_lightning_indexer public contract supports quant_mode=1 only"
        )
    if layout_q != "TND":
        raise ValueError("Layout_q must be TND under the current xlsx contract")
    if layout_k != "PA_BBND":
        raise ValueError("Layout_k must be PA_BBND under the current xlsx contract")
    if query.device.type != "npu" or key.device.type != "npu":
        raise ValueError("Quant_lightning_indexer requires NPU q/k tensors")
    if query.dtype != torch.uint8 or key.dtype != torch.uint8:
        raise TypeError("Public q/k storage must be uint8 packed MXFP4")
    if query.ndim != 3 or key.ndim != 4:
        raise ValueError("Q must be [T1,32,64], k must be [block_num,block_size,1,64]")
    if (
        query.shape[1] not in (32, 64)
        or key.shape[2] != N2
        or query.shape[2] != PACKED_D
        or key.shape[3] != PACKED_D
    ):
        raise ValueError("Public static axes require N1=32, N2=1 and logical D=128")
    if tuple(weights.shape) != tuple(query.shape[:2]) or weights.dtype != torch.float32:
        raise ValueError("W must be FP32 with shape (T1,N1)")
    if query_dequant_scale is None:
        raise ValueError("Descale_q is a required xlsx input")
    expected_q_scale = (*query.shape[:2], LOGICAL_D // 64, 2)
    if tuple(query_dequant_scale.shape) != expected_q_scale:
        raise ValueError("Descale_q must have corrected xlsx shape (T1,N1,D/64,2)")
    if key_dequant_scale is None:
        raise ValueError("Descale_k is a required xlsx input")
    expected = tuple(key.shape[:3]) + (2, 2)
    if tuple(key_dequant_scale.shape) != expected:
        raise ValueError("Descale_k must have xlsx shape (block_num,block_size,N2,2,2)")
    key_stride0, key_storage_rows = _pa_axis0_storage_rows(
        key,
        "k",
        (PACKED_D, PACKED_D, 1),
        KEY_STORAGE_ROW_ELEMENTS,
    )
    (
        key_dequant_scale_stride0,
        key_dequant_scale_storage_rows,
    ) = _pa_axis0_storage_rows(
        key_dequant_scale,
        "descale_k",
        (KEY_SCALE_STORAGE_ROW_ELEMENTS, 2 * 2, 2, 1),
        KEY_SCALE_STORAGE_ROW_ELEMENTS,
    )
    # CANNBotDSL's packed-FP4 host carrier is compact-only. Expose the
    # allocation span, including axis-0 gaps, as a zero-copy compact carrier;
    # the original page strides are passed separately and consumed on NPU.
    key_storage = torch.as_strided(
        key,
        (key_storage_rows, PACKED_D),
        (KEY_STORAGE_ROW_ELEMENTS, 1),
    )
    key_dequant_scale_storage = torch.as_strided(
        key_dequant_scale,
        (key_dequant_scale_storage_rows, 4),
        (KEY_SCALE_STORAGE_ROW_ELEMENTS, 1),
    )
    _validate_qli_candidate(candidate_topk_blocks, candidate_block_size)
    # Keep this as a Python/JIT specialization constant. Disabled launches
    # compile out Candidate block-max, key generation, TopK and UB resources.
    candidate_enabled = int(candidate_topk_blocks) > 0
    pa_block_size = int(key.shape[1])
    if (
        pa_block_size <= 0
        or pa_block_size > MAX_PA_BLOCK_SIZE
        or pa_block_size % PA_BLOCK_ALIGNMENT != 0
    ):
        raise ValueError("PA block_size must be a multiple of 16 in (0, 1024]")
    _validate_qli_compute_attributes(topk, mask_mode, cmp_ratio)
    if query_dequant_scale.dtype != torch.uint8:
        raise TypeError("Descale_q must use uint8 E8M0 storage")
    if key_dequant_scale.dtype != torch.uint8:
        raise TypeError("Descale_k must use uint8 E8M0 storage")
    for name, tensor in (
        ("w", weights),
        ("descale_q", query_dequant_scale),
    ):
        if tensor.device != query.device:
            raise ValueError(f"{name} must be on the same NPU as q")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not query.is_contiguous():
        raise ValueError("Q must be contiguous")

    t, batch, has_cu_seqlens_q = _validate_qli_pa_sequences(
        query,
        cu_seqlens_q,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        mask_mode,
        cmp_ratio,
    )
    has_seqused_q = seqused_q is not None
    has_seqused_k = seqused_k is not None
    has_cmp_residual_k = cmp_residual_k is not None
    has_block_table = block_table is not None
    logical_k_capacity = (
        int(block_table.shape[1]) if has_block_table else int(key.shape[0])
    ) * pa_block_size
    tokens_pad = ceil_div(logical_k_capacity, TILE_N) * TILE_N
    if tokens_pad <= 0:
        raise ValueError("K storage must expose at least one PA page")
    # Optional xlsx tensors remain optional at the public boundary.  One
    # uninitialized carrier is sufficient because the JIT specialization
    # compiles out every load when the corresponding tensor is absent.
    dummy_i32 = torch.empty((1,), dtype=torch.int32, device=query.device)
    dummy_table = torch.empty((1, 1), dtype=torch.int32, device=query.device)
    cu_kernel = cu_seqlens_q if has_cu_seqlens_q else dummy_i32
    seqused_q_kernel = seqused_q if has_seqused_q else dummy_i32
    seqused_k_kernel = seqused_k if has_seqused_k else dummy_i32
    cmp_residual_kernel = cmp_residual_k if has_cmp_residual_k else dummy_i32
    block_table_kernel = block_table if has_block_table else dummy_table

    max_query_length = int(max_seqlen_q) if int(max_seqlen_q) > 0 else t
    max_tasks_per_batch = ceil_div(max_query_length, tile_m // query_heads)
    # Preserve ABI capacity; KV128 computes actual tasks on device.
    split_count = max(1, min(MAX_MERGE_PARTS, STREAM_TOPK_TRUNK_LEN // int(topk)))
    if (
        metadata.dtype != torch.int32
        or metadata.ndim != 1
        or metadata.numel() != METADATA_ELEMENTS
        or not metadata.is_contiguous()
    ):
        raise ValueError(
            "Metadata must be contiguous int32 [1024] in ASC LI/LD boundary format"
        )
    if metadata.device != q.device:
        raise ValueError("Metadata must be on the same device as q")
    metadata = metadata.view(METADATA_ROWS, METADATA_FIELDS)
    worker_count = MAX_CUBE_WORKERS
    decode_long = _decode_seq in (6, 8, 16) and logical_k_capacity > 16384
    workspace_row_count = (
        t if decode_long else (MAX_CUBE_WORKERS * 2) * (tile_m // query_heads)
    )
    scratch_rows = (
        ceil_div(logical_k_capacity, 1024) * t if decode_long else 64 * query_tile_rows
    )
    shard_pad = tokens_pad
    # Public outputs and GM scratch are allocation-only Torch calls.  The
    # fused kernel writes every public row, including invalid/padded rows.
    sparse_indices_2d = torch.empty(
        (scratch_rows, int(topk)), dtype=torch.int32, device=query.device
    )
    sparse_values_2d = torch.empty(
        (scratch_rows, int(topk)), dtype=torch.bfloat16, device=weights.device
    )
    candidate_count = max(0, int(candidate_topk_blocks))
    candidate_indices_2d = torch.empty(
        (scratch_rows, 512 if decode_long else candidate_count),
        dtype=torch.int32,
        device=query.device,
    )
    candidate_length_1d = torch.empty(
        (t * split_count,), dtype=torch.int32, device=query.device
    )

    key_workspace = torch.empty(
        (workspace_row_count, shard_pad),
        dtype=torch.uint16,
        device=query.device,
    )
    candidate_key_workspace = torch.empty(
        (
            workspace_row_count,
            shard_pad // CANDIDATE_BLOCK_SIZE if candidate_enabled else 1,
        ),
        dtype=torch.uint16,
        device=query.device,
    )
    kernel_candidate_count = candidate_count if candidate_enabled else 1
    candidate_indices_kernel = (
        candidate_indices_2d
        if candidate_enabled
        else torch.empty((1, split_count), dtype=torch.int32, device=query.device)
    )
    candidate_value_workspace = torch.empty(
        (scratch_rows, 512 if decode_long else kernel_candidate_count),
        dtype=torch.uint16,
        device=query.device,
    )

    # GM aliases describe native packed formats without device copies.
    query_address = query.data_ptr()
    query_scale_storage = query_dequant_scale.reshape(t * query_heads, 2, 2).view(
        torch.float8_e8m0fnu
    )

    final_sparse_indices = torch.empty(
        (t, int(topk)), dtype=torch.int32, device=query.device
    )
    final_sparse_values = torch.empty(
        (t, int(topk)), dtype=torch.bfloat16, device=query.device
    )
    final_candidate_indices = torch.empty(
        (t, kernel_candidate_count), dtype=torch.int32, device=query.device
    )
    final_candidate_bits = torch.empty(
        (t, kernel_candidate_count), dtype=torch.uint16, device=query.device
    )
    elide_candidate_scores = (
        candidate_enabled
        and int(topk) == 512
        and int(candidate_topk_blocks) == 2048
        and int(candidate_block_size) == CANDIDATE_BLOCK_SIZE
        and int(mask_mode) == 0
        and t == 1
        and logical_k_capacity == 8192
        and has_seqused_k
    )
    stage = PreparedStage(
        (
            int(topk),
            candidate_enabled,
            kernel_candidate_count,
            int(mask_mode),
            int(cmp_ratio),
            has_cu_seqlens_q,
            has_seqused_q,
            has_seqused_k,
            has_block_table,
            pa_block_size,
            split_count,
            query_heads,
            query_tile_rows,
            elide_candidate_scores,
        )
    )
    # Pointer-only launch slots need explicit owners after public output aliases
    # are rebound below. Keep all scratch banks alive for prepared/Graph calls.
    sparse_indices_workspace = sparse_indices_2d
    sparse_values_workspace = sparse_values_2d
    candidate_indices_workspace = candidate_indices_kernel
    stage.args = (
        query_address,
        key_storage.data_ptr(),
        weights,
        query_scale_storage,
        key_dequant_scale_storage.data_ptr(),
        key_stride0,
        key_dequant_scale_stride0,
        cu_kernel,
        seqused_q_kernel,
        seqused_k_kernel,
        cmp_residual_kernel,
        block_table_kernel,
        key_workspace,
        candidate_key_workspace,
        sparse_indices_2d.data_ptr(),
        sparse_values_2d.view(torch.uint16).data_ptr(),
        candidate_indices_kernel.data_ptr(),
        candidate_value_workspace.data_ptr(),
        candidate_length_1d,
        worker_count,
        t,
        batch,
        logical_k_capacity,
        max_tasks_per_batch,
        key_storage_rows,
        key_dequant_scale_storage.shape[0],
        metadata,
        metadata.shape[0],
        final_sparse_indices.data_ptr(),
        final_sparse_values.view(torch.uint16).data_ptr(),
        final_candidate_indices.data_ptr(),
        final_candidate_bits.data_ptr(),
        int(return_value),
        min(64, t),
        0 if output_idx_offset is None else output_idx_offset.data_ptr(),
        key_workspace.data_ptr(),
    )

    sparse_indices_2d, sparse_values_2d = final_sparse_indices, final_sparse_values
    if candidate_enabled:
        candidate_indices_2d = final_candidate_indices
    candidate_length_1d = candidate_length_1d[:t]
    return stage.finish(
        (
            sparse_indices_2d.view(t, N2, int(topk)),
            sparse_values_2d.view(t, N2, int(topk))
            if return_value
            else sparse_values_2d.reshape(-1)[:0],
            candidate_indices_2d.view(t, N2, candidate_count)
            if candidate_enabled
            else candidate_indices_2d.reshape(-1),
            candidate_length_1d.view(t, N2)
            if candidate_enabled
            else candidate_length_1d[:0],
        ),
        locals(),
    )


class PreparedStage:
    def __init__(self, config):
        self.config = config

    def finish(self, outputs, local_values):
        self.outputs = outputs
        self.keepalive = tuple(
            value for value in local_values.values() if isinstance(value, torch.Tensor)
        )
        return self


# KV128调度、QK流水和分层归并。


def _double_outputs(selector):
    selector.output_idx_stage = _ub_channel(
        MemLoc.UB,
        (1, selector.topk),
        dtypes.int32,
        depth=2,
    )
    selector.output_value_bits_stage = _ub_channel(
        MemLoc.UB,
        (1, selector.topk),
        dtypes.uint16,
        depth=2,
    )


class RawSelector(QliRawTopKSelector):
    def __init__(self, topk, workspace):
        super().__init__(topk, workspace)
        _double_outputs(self)


class MergeSelector(CandidateLdTopKSelector):
    def __init__(self, topk, workspace, parts):
        super().__init__(topk, workspace, parts)
        _double_outputs(self)


class _ZnRing:
    """Depth-1 stand-in for ``RingBuffer`` over an explicit ZN view."""

    def __init__(self, slots):
        self._slots = tuple(slots)
        self._write = 0
        self._read = 0

    def next_write(self):
        slot = self._slots[self._write % len(self._slots)]
        self._write += 1
        return slot

    def next_read(self):
        slot = self._slots[self._read % len(self._slots)]
        self._read += 1
        return slot


class KvCube:
    def __init__(self, seq, logical_cache=True):
        self.logical_cache = logical_cache
        self.seq = seq
        self.query_rows = 6 if seq == 6 else 8
        self.tile_m = self.query_rows * 32
        self.m_tiles = (seq + self.query_rows - 1) // self.query_rows
        self.q = RingBuffer(
            MemLoc.L1,
            dtypes.fp4x2_e2m1,
            make_layout((self.tile_m, 128)),
            depth=2,
            data_format="nz",
        )
        # DSL 0.8.0 Buffer rejects data_format='zn'; keep the ZN geometry with
        # an ordinary byte allocation rebound to ZN views.
        _qs_bytes = Buffer(
            MemLoc.L1, dtypes.uint8, make_layout((self.tile_m * 4,)), data_format="nd"
        )
        self.qs = _ZnRing(
            [
                _reinterpret_op(
                    _qs_bytes,
                    dtype=dtypes.float8_e8m0,
                    layout=make_layout((self.tile_m, 4)),
                    data_format="zn",
                )
            ]
        )
        self.qa = [
            RingBuffer(
                MemLoc.L0A, dtypes.fp4x2_e2m1, make_layout((self.tile_m, 128)), depth=1
            )
            for _ in range(self.m_tiles)
        ]
        self.k = RingBuffer(
            MemLoc.L1,
            dtypes.fp4x2_e2m1,
            make_layout((128, 128)),
            depth=2,
            data_format="nz",
        )
        self.ks = RingBuffer(
            MemLoc.L1,
            dtypes.float8_e8m0,
            make_layout((4, 128)),
            depth=2,
            data_format="nz",
        )
        self.b = RingBuffer(
            MemLoc.L0B, dtypes.fp4x2_e2m1, make_layout((128, 128)), depth=2
        )
        self.c = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((self.tile_m, 128)), depth=2
        )

    @jit
    def load_q(self, q, qs):
        for mb in range_constexpr(self.m_tiles):
            mem_copy(self.q.next_write(), tile_slice(q, (self.tile_m, 128), (mb, 0)))
            mem_copy(
                self.qs.next_write(),
                tile_slice(qs, (self.tile_m, 2, 2), (mb, 0, 0)).flatten(-2, -1),
            )
            mem_copy(
                self.qa[mb].next_write(),
                self.q.next_read(),
                options=copy_options(mx_scale=self.qs.next_read()),
            )

    @jit
    def prefetch(self, k, ks, table, tile, key_stride, scale_stride):
        # The combined-cache unpack phase has already resolved the public
        # page table and published private planes in logical page order.
        if const_expr(self.logical_cache):
            page = dtypes.int64(tile)
        else:
            page = dtypes.int64(table[0, tile])
        kr = page * (key_stride // KEY_STORAGE_ROW_ELEMENTS)
        sr = page * (scale_stride // KEY_SCALE_STORAGE_ROW_ELEMENTS)
        mem_copy(self.k.next_write(), k[kr : kr + 128, None])
        mem_copy(
            self.ks.next_write(),
            ks[sr : sr + 128, None, None].flatten(-2, -1).permute(1, 0),
        )

    @jit
    def begin_k(self):
        mem_copy(
            self.b.next_write(),
            self.k.next_read(),
            options=copy_options(mx_scale=self.ks.next_read()),
        )
        return self.b.next_read()

    @jit
    def compute(self, mb, key, handoff):
        matmul(
            self.c.next_write(), self.qa[mb].next_read(), key, init=True, mx_mode=True
        )
        source = self.c.next_read()
        destination = handoff.next_write()
        mem_copy(
            destination,
            tile_slice(source, (self.tile_m // 2, 128), (0, 0)),
            options=copy_options(dst_subblock=0),
        )
        mem_copy(
            destination,
            tile_slice(source, (self.tile_m // 2, 128), (1, 0)),
            options=copy_options(dst_subblock=1),
        )


class ScoreVector:
    def __init__(self, mb, query_rows=8):
        self.mb = mb
        self.query_rows = query_rows
        self.rows = query_rows // 2
        self.weights = Buffer(MemLoc.UB, dtypes.float32, make_layout((self.rows, 32)))
        self.packed = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((self.rows, 128)))
        self.keys = RingBuffer(
            MemLoc.UB, dtypes.uint16, make_layout((self.rows, 128)), depth=2
        )

    @jit
    def load_weights(self, w):
        row = self.mb * self.query_rows + get_subblock_id() * self.rows
        mem_copy(
            self.weights, tile_slice(_offset_view(w, (row, 0)), (self.rows, 32), (0, 0))
        )
        with vf(mode="simd"):
            m32 = rr.update_mask(32, elem_bits=32)[0]
            m16 = rr.update_mask(32, elem_bits=16)[0]
            for r in range_constexpr(self.rows):
                v = rr.vcast(
                    rr.vload(self.weights, r * 32),
                    dtypes.bfloat16,
                    mask=m32,
                    reg_layout=rr.RegLayout.ZERO,
                )
                packed = rr.vpack(
                    rr.vreinterpret_lanes(v, dtypes.uint32),
                    dtypes.uint16,
                    part="lower",
                )
                rr.vstore(
                    self.packed, r * 128, rr.vreinterpret(packed, dtypes.bfloat16), m16
                )
            rr.vmem_bar("vst_vld")

    @jit
    def score(self, qk, out, tile, actual, used_q, total_q):
        key = self.keys.next_write()
        row_base = self.mb * self.query_rows + get_subblock_id() * self.rows
        for r in range_constexpr(self.rows):
            visible = max(0, min(actual, actual - used_q + row_base + r + 1))
            if row_base + r >= used_q:
                visible = 0
            count = max(0, min(128, visible - tile * 128))
            with vf(mode="simd"):
                mask = rr.update_mask(128, elem_bits=16)[0]
                lanes = rr.varange(0, dtypes.uint16)
                zero = rr.vdups(0, dtypes.uint16, mask=mask)
                w = rr.vload(self.packed, r * 128)
                acc = rr.vdups(0.0, dtypes.bfloat16, mask=mask)
                for h in dsl_range(32, unroll=4):
                    wh = rr.vgather_reg(w, rr.vdups(h, dtypes.uint16, mask=mask))
                    dot = rr.vmaxs(rr.vload(qk, (r * 32 + h) * 128), 0.0, mask=mask)
                    acc = rr.vadd(rr.vmul(dot, wh, mask=mask), acc, mask=mask)
                acc = rr.vadds(acc, 0.0, mask=mask)
                bits = rr.vreinterpret(acc, dtypes.uint16)
                neg = rr.veqs(
                    rr.vbitwise_and(
                        bits, rr.vdups(32768, dtypes.uint16, mask=mask), mask=mask
                    ),
                    32768,
                    mask=mask,
                )
                keys = rr.vselect(
                    rr.vbitwise_xor(
                        bits, rr.vdups(65535, dtypes.uint16, mask=mask), mask=mask
                    ),
                    rr.vbitwise_xor(
                        bits, rr.vdups(32768, dtypes.uint16, mask=mask), mask=mask
                    ),
                    cond_mask=neg,
                )
                valid = rr.vlts(lanes, count, mask=mask)
                rr.vstore(key, r * 128, rr.vselect(keys, zero, cond_mask=valid), mask)
                rr.vmem_bar("vst_vld")
        mem_copy(
            tile_slice(
                _offset_view(out, (row_base, tile * 128)), (self.rows, 128), (0, 0)
            ),
            self.keys.next_read(),
        )

    @jit
    def candidate_keys(self, scores, out, tile, actual, active_q):
        loaded = RingBuffer(
            MemLoc.UB, dtypes.uint16, make_layout((self.rows, 128)), depth=2
        )
        result = RingBuffer(
            MemLoc.UB, dtypes.uint16, make_layout((self.rows, 16)), depth=2
        )
        row_base = self.mb * self.query_rows + get_subblock_id() * self.rows
        mem_copy(
            loaded.next_write(),
            tile_slice(
                _offset_view(scores, (row_base, tile * 128)), (self.rows, 128), (0, 0)
            ),
        )
        source = loaded.next_read()
        dst = result.next_write()
        for r in range_constexpr(self.rows):
            visible = max(0, min(actual, actual - active_q + row_base + r + 1))
            if row_base + r >= active_q:
                visible = 0
            blocks = (visible + 7) // 8
            with vf(mode="simd"):
                mask = rr.update_mask(16, elem_bits=16)[0]
                idx = rr.vadds(
                    rr.vmuls(rr.varange(0, dtypes.uint16), 8, mask=mask),
                    r * 128,
                    mask=mask,
                )
                maximum = rr.vgather(source, idx, mask=mask)
                for lane in range_constexpr(1, 8):
                    maximum = rr.vmax(
                        maximum,
                        rr.vgather(source, rr.vadds(idx, lane, mask=mask), mask=mask),
                        mask=mask,
                    )
                block = rr.varange(0, dtypes.uint16)
                valid = rr.vlts(block, max(0, min(16, blocks - tile * 16)), mask=mask)
                maximum = rr.vselect(
                    maximum,
                    rr.vdups(0, dtypes.uint16, mask=mask),
                    cond_mask=valid,
                )
                # Keep the scalar in the local 16-lane domain before uint16
                # comparison; global block differences can exceed 65535.
                latest_lane = max(-1, min(16, blocks - 1 - tile * 16))
                latest = rr.veqs(block, latest_lane, mask=mask)
                maximum = rr.vselect(
                    rr.vdups(65408, dtypes.uint16, mask=mask),
                    maximum,
                    cond_mask=latest,
                )
                rr.vstore(dst, r * 16, maximum, mask)
                rr.vmem_bar("vst_vld")
        mem_copy(
            tile_slice(
                _offset_view(out, (row_base, tile * 16)), (self.rows, 16), (0, 0)
            ),
            result.next_read(),
        )


class Kv128:
    def __init__(self, seq, long_k=False, logical_cache=True):
        self.logical_cache = logical_cache
        self.seq = seq
        self.query_rows = 6 if seq == 6 else 8
        self.m_tiles = (seq + self.query_rows - 1) // self.query_rows
        self.long_k = long_k

    @jit
    def run(
        self,
        q,
        k,
        w,
        qs,
        ks,
        table,
        key_stride,
        scale_stride,
        used_q,
        used_k,
        scores,
        partial_idx,
        partial_bits,
        merge_idx,
        merge_bits,
        indices,
        bits,
        candidates,
        candidate_bits,
        length,
        candidate_scores,
        return_values,
        offset_address,
        capacity,
    ):
        actual = max(0, min(capacity, dtypes.int64(used_k[0])))
        active_q = max(0, min(self.seq, dtypes.int64(used_q[0])))
        tiles = (actual + 127) // 128
        cube = KvCube(self.seq, self.logical_cache)
        handoff = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((self.query_rows // 2 * 32, 128)),
            depth=2,
        )
        vectors = [ScoreVector(mb, self.query_rows) for mb in range(self.m_tiles)]
        cube.load_q(q, qs)
        for mb in range_constexpr(self.m_tiles):
            vectors[mb].load_weights(w)
        cores = dtypes.int64(get_block_num())
        first = dtypes.int64(get_block_idx())
        if first < tiles:
            cube.prefetch(k, ks, table, first, key_stride, scale_stride)
        for tile in dsl_range(first, tiles, cores, unroll=1):
            if tile + cores < tiles:
                cube.prefetch(k, ks, table, tile + cores, key_stride, scale_stride)
            key = cube.begin_k()
            for mb in range_constexpr(self.m_tiles):
                cube.compute(mb, key, handoff)
                vectors[mb].score(
                    handoff.next_read(), scores, tile, actual, active_q, self.seq
                )
                if const_expr(self.long_k):
                    if actual > 16384:
                        vec_sync_all()
                        vectors[mb].candidate_keys(
                            scores, candidate_scores, tile, actual, active_q
                        )
        global_sync_all()
        onchip_rewind()
        aiv = dtypes.int64(get_block_idx() * 2 + get_subblock_id())
        aiv_count = cores * 2
        if const_expr(self.seq == 8 and not self.long_k):
            # The host specializes long_k for capacities above 16K.  A single
            # AIV owns each short row, avoiding partial TopK publication and
            # the cross-core merge barrier.
            ws = QliRawTopKWorkspace(16384, 512)
            select = RawSelector(512, ws)
            for row in dsl_range(aiv, self.seq, aiv_count, unroll=1):
                visible = max(0, min(actual, actual - active_q + row + 1))
                if row >= active_q:
                    visible = 0
                off = dtypes.int32(0)
                if offset_address != 0:
                    off = make_tensor(
                        make_pointer(dtypes.int32, offset_address, MemLoc.GM),
                        make_layout((self.seq, 1), stride=(1, 1)),
                    )[row, 0]
                select.select_row(
                    _offset_view(scores, (row, 0)),
                    actual,
                    min(512, visible),
                    tile_slice(indices, (1, 512), (row, 0)),
                    tile_slice(bits, (1, 512), (row, 0)),
                    off,
                    return_values,
                )
        else:
            groups = (actual + 1023) // 1024
            ws = QliRawTopKWorkspace(16384, 512)
            select = RawSelector(512, ws)
            for job in dsl_range(aiv, groups * self.seq, aiv_count, unroll=1):
                row = job % self.seq
                group = job // self.seq
                span = min(1024, actual - group * 1024)
                visible = max(0, min(actual, actual - active_q + row + 1))
                if row >= active_q:
                    visible = 0
                valid = max(0, min(span, visible - group * 1024))
                select.select_row(
                    _offset_view(scores, (row, group * 1024)),
                    span,
                    valid,
                    tile_slice(partial_idx, (1, 512), (job, 0)),
                    tile_slice(partial_bits, (1, 512), (job, 0)),
                    dtypes.int32(group * 1024),
                    1,
                )
            global_sync_all()
            onchip_rewind()
            mergews = CandidateQliRawTopKWorkspace(16384, 512, candidate_optimized=True)
            merge = MergeSelector(512, mergews, 16)
            merge_groups = groups
            if const_expr(self.long_k):
                # A 1M context has 1024 initial groups: 1024 -> 128 -> 16.
                # Every contraction reads one GM bank and publishes into the other.
                # All AIVs participate in the inter-level barrier, including idle ones.
                for level in range_constexpr(2):
                    if merge_groups > 16:
                        next_groups = (merge_groups + 7) // 8
                        if const_expr(level == 0):
                            level_idx = partial_idx
                            level_bits = partial_bits
                            next_idx = merge_idx
                            next_bits = merge_bits
                        else:
                            level_idx = merge_idx
                            level_bits = merge_bits
                            next_idx = partial_idx
                            next_bits = partial_bits
                        for job in dsl_range(
                            aiv, next_groups * self.seq, aiv_count, unroll=1
                        ):
                            row = job % self.seq
                            group = job // self.seq
                            count = min(8, merge_groups - group * 8)
                            ix = merge.ld_indices.next_write()
                            bv = merge.ld_bits.next_write()
                            for g in dsl_range(count, unroll=1):
                                mem_copy(
                                    tile_slice(ix, (1, 512), (0, g)),
                                    tile_slice(
                                        level_idx,
                                        (1, 512),
                                        ((group * 8 + g) * self.seq + row, 0),
                                    ),
                                )
                                mem_copy(
                                    tile_slice(bv, (1, 512), (0, g)),
                                    tile_slice(
                                        level_bits,
                                        (1, 512),
                                        ((group * 8 + g) * self.seq + row, 0),
                                    ),
                                )
                            merge.merge_ld(
                                level_idx,
                                level_bits,
                                tile_slice(next_idx, (1, 512), (job, 0)),
                                tile_slice(next_bits, (1, 512), (job, 0)),
                                count * 512,
                                0,
                                True,
                            )
                        global_sync_all()
                        onchip_rewind()
                        merge_groups = next_groups
            source_idx = partial_idx
            source_bits = partial_bits
            if const_expr(self.long_k):
                # One level ends in the secondary bank; two end in the primary.
                if groups > 16 and groups <= 128:
                    source_idx = merge_idx
                    source_bits = merge_bits
            for row in dsl_range(aiv, self.seq, aiv_count, unroll=1):
                if merge_groups == 0:
                    # The empty-row fallback only writes the public TopK slots; a
                    # full 16K trunk would add ~100 KiB of UB to this phase and push
                    # the epoch past the 256 KiB budget.
                    QliRawTopKSelector(
                        512,
                        QliRawTopKWorkspace(128, 512),
                    ).store_empty_row(
                        tile_slice(indices, (1, 512), (row, 0)),
                        tile_slice(bits, (1, 512), (row, 0)),
                    )
                else:
                    ix = merge.ld_indices.next_write()
                    bv = merge.ld_bits.next_write()
                    for g in dsl_range(merge_groups, unroll=1):
                        mem_copy(
                            tile_slice(ix, (1, 512), (0, g)),
                            tile_slice(source_idx, (1, 512), (g * self.seq + row, 0)),
                        )
                        mem_copy(
                            tile_slice(bv, (1, 512), (0, g)),
                            tile_slice(source_bits, (1, 512), (g * self.seq + row, 0)),
                        )
                    off = dtypes.int32(0)
                    if offset_address != 0:
                        off = make_tensor(
                            make_pointer(dtypes.int32, offset_address, MemLoc.GM),
                            make_layout((self.seq, 1), stride=(1, 1)),
                        )[row, 0]
                    merge.merge_ld(
                        partial_idx,
                        partial_bits,
                        tile_slice(indices, (1, 512), (row, 0)),
                        tile_slice(bits, (1, 512), (row, 0)),
                        merge_groups * 512,
                        off,
                        True,
                    )
        vec_sync_all()
        direct = RingBuffer(MemLoc.UB, dtypes.int32, make_layout((1, 2048)), depth=1)
        cnt = RingBuffer(MemLoc.UB, dtypes.int32, make_layout((1,)), depth=2)
        for row in dsl_range(aiv, self.seq, aiv_count, unroll=1):
            visible = max(0, min(actual, actual - active_q + row + 1))
            if row >= active_q:
                visible = 0
            blocks = (visible + 7) // 8
            if blocks <= 2048:
                dst = direct.next_write()
                with vf(mode="simd"):
                    mask = rr.update_mask(64, elem_bits=32)[0]
                    for chunk in dsl_range(32, unroll=1):
                        pos = rr.varange(chunk * 64, dtypes.int32)
                        vals = rr.vselect(
                            pos,
                            rr.vdups(-1, dtypes.int32, mask=mask),
                            cond_mask=rr.vlts(pos, blocks, mask=mask),
                        )
                        rr.vstore(dst, chunk * 64, vals, mask)
                    rr.vmem_bar("vst_vld")
                mem_copy(
                    tile_slice(candidates, (1, 2048), (row, 0)), direct.next_read()
                )
                dst = cnt.next_write()
                with vf(mode="simd"):
                    mask = rr.update_mask(1, elem_bits=32)[0]
                    rr.vstore(dst, 0, rr.vdups(blocks, dtypes.int32, mask=mask), mask)
                    rr.vmem_bar("vst_vld")
                mem_copy(tile_slice(length, (1,), (row,)), cnt.next_read())
        if const_expr(self.long_k):
            vec_sync_all()
            onchip_rewind()
            candws = QliRawTopKWorkspace(16384, 2048)
            candselect = RawSelector(2048, candws)
            for row in dsl_range(aiv, self.seq, aiv_count, unroll=1):
                visible = max(0, min(actual, actual - active_q + row + 1))
                if row >= active_q:
                    visible = 0
                blocks = (visible + 7) // 8
                if blocks > 2048:
                    candselect.select_row(
                        _offset_view(candidate_scores, (row, 0)),
                        (actual + 7) // 8,
                        blocks,
                        tile_slice(candidates, (1, 2048), (row, 0)),
                        tile_slice(candidate_bits, (1, 2048), (row, 0)),
                        0,
                        1,
                    )
                    dst = cnt.next_write()
                    with vf(mode="simd"):
                        mask = rr.update_mask(1, elem_bits=32)[0]
                        rr.vstore(dst, 0, rr.vdups(2048, dtypes.int32, mask=mask), mask)
                        rr.vmem_bar("vst_vld")
                    mem_copy(tile_slice(length, (1,), (row,)), cnt.next_read())
        vec_sync_all()
        onchip_rewind()
        sorter = Sort512()
        for row in dsl_range(aiv, self.seq, aiv_count, unroll=1):
            if return_values == 0:
                sorter.run_indices(tile_slice(indices, (1, 512), (row, 0)))
            else:
                sorter.run(
                    tile_slice(indices, (1, 512), (row, 0)),
                    tile_slice(bits, (1, 512), (row, 0)),
                )


def build_runner(config, seq, long_k=False, logical_cache=True):
    specs = _qli_specs(*config)
    specs = list(specs)
    specs[12] = TensorSpec((Dim("DECODE_WS_ROWS"), Dim("S2_PAD")), dtypes.uint16)
    specs[13] = TensorSpec((Dim("DECODE_WS_ROWS"), Dim("CANDIDATE_PAD")), dtypes.uint16)
    specs = tuple(specs)
    # 从 config 取值（位置约定同 _qli_specs），保证 ABI spec 与 kernel 参数一致。
    topk = int(config[0])
    candidate_topk = int(config[2])
    query_head_count = int(config[11])
    query_tile_rows = int(config[12])

    @jit
    def QuantLightningIndexer(
        query_address,
        key_address,
        weights: Tensor,
        query_scale,
        key_scale_address,
        key_stride0,
        key_dequant_scale_stride0,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        block_table: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices_address,
        sparse_value_bits_address,
        candidate_indices_address,
        candidate_value_bits_address,
        candidate_length: Tensor,
        worker_count,
        total_q,
        batch_count,
        tokens,
        max_tasks_per_batch,
        key_rows,
        scale_rows,
        metadata: Tensor,
        task_count,
        final_sparse_indices_address,
        final_sparse_bits_address,
        final_candidate_indices_address,
        final_candidate_bits_address,
        return_values,
        merge_workers,
        output_offset_address,
        merge_metadata_address,
    ):
        final_sparse_indices = make_tensor(
            make_pointer(
                dtypes.int32, dtypes.int64(final_sparse_indices_address), MemLoc.GM
            ),
            make_layout((total_q, topk), stride=(topk, 1)),
        )
        sparse_indices = make_tensor(
            make_pointer(dtypes.int32, dtypes.int64(sparse_indices_address), MemLoc.GM),
            make_layout(
                (
                    (tokens + 1023) // 1024 * total_q
                    if long_k
                    else 64 * query_tile_rows,
                    topk,
                ),
                stride=(topk, 1),
            ),
        )
        sparse_value_bits = make_tensor(
            make_pointer(
                dtypes.uint16, dtypes.int64(sparse_value_bits_address), MemLoc.GM
            ),
            make_layout(
                (
                    (tokens + 1023) // 1024 * total_q
                    if long_k
                    else 64 * query_tile_rows,
                    topk,
                ),
                stride=(topk, 1),
            ),
        )
        final_sparse_bits = make_tensor(
            make_pointer(
                dtypes.uint16, dtypes.int64(final_sparse_bits_address), MemLoc.GM
            ),
            make_layout((total_q, topk), stride=(topk, 1)),
        )
        final_candidate_indices = make_tensor(
            make_pointer(
                dtypes.int32, dtypes.int64(final_candidate_indices_address), MemLoc.GM
            ),
            make_layout((total_q, candidate_topk), stride=(candidate_topk, 1)),
        )
        final_candidate_bits = make_tensor(
            make_pointer(
                dtypes.uint16, dtypes.int64(final_candidate_bits_address), MemLoc.GM
            ),
            make_layout((total_q, candidate_topk), stride=(candidate_topk, 1)),
        )
        candidate_indices = make_tensor(
            make_pointer(
                dtypes.int32, dtypes.int64(candidate_indices_address), MemLoc.GM
            ),
            make_layout(
                (
                    (tokens + 1023) // 1024 * total_q
                    if long_k
                    else 64 * query_tile_rows,
                    512 if long_k else candidate_topk,
                ),
                stride=(512 if long_k else candidate_topk, 1),
            ),
        )
        candidate_value_bits = make_tensor(
            make_pointer(
                dtypes.uint16, dtypes.int64(candidate_value_bits_address), MemLoc.GM
            ),
            make_layout(
                (
                    (tokens + 1023) // 1024 * total_q
                    if long_k
                    else 64 * query_tile_rows,
                    512 if long_k else candidate_topk,
                ),
                stride=(512 if long_k else candidate_topk, 1),
            ),
        )
        query = make_tensor(
            make_pointer(dtypes.fp4x2_e2m1, dtypes.int64(query_address), MemLoc.GM),
            make_layout((total_q * query_head_count, LOGICAL_D), stride=(LOGICAL_D, 1)),
        )
        key = make_tensor(
            make_pointer(dtypes.fp4x2_e2m1, dtypes.int64(key_address), MemLoc.GM),
            make_layout((key_rows, LOGICAL_D), stride=(LOGICAL_D, 1)),
        )
        key_scale = make_tensor(
            make_pointer(
                dtypes.float8_e8m0, dtypes.int64(key_scale_address), MemLoc.GM
            ),
            make_layout((scale_rows, 2, 2), stride=(4, 2, 1)),
        )
        Kv128(
            seq,
            long_k,
            logical_cache,
        ).run(
            query,
            key,
            weights,
            query_scale,
            key_scale,
            block_table,
            key_stride0,
            key_dequant_scale_stride0,
            seqused_q,
            seqused_k,
            key_out,
            sparse_indices,
            sparse_value_bits,
            candidate_indices,
            candidate_value_bits,
            final_sparse_indices,
            final_sparse_bits,
            final_candidate_indices,
            final_candidate_bits,
            candidate_length,
            candidate_key_out,
            return_values,
            output_offset_address,
            tokens,
        )

    return (QuantLightningIndexer, specs)
