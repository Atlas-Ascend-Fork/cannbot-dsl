# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stream-K (DPSK) matmul with host-side tiling, single MIX kernel.

AIC: DP tiles -> output GM, SK tiles -> workspace GM.
AIV: reduce workspace -> output GM, using GetTaskRatio()*k_splits AIVs
      (each AIV handles one M-segment of one mn-tile's reduce).

Three-phase cross-block barrier. Partial sums accumulate in registers in a raw VF.
Workspace: (m, k_splits*n_tiles*tile_n) for ragged-safe tile_slice clamp.

Optizations adopted from matmul.py:
  - L2 cache control per-matrix (l2_cache_ctl_a/b)
  - L1 4-buffer ping-pong (dynamic depth from tiling)
  - Slide window scheduling + row reversal for DP path L2 locality

Formula:  C[M,N] = A[M,K] @ B[K,N]   (fp16/bf16 inputs, fp32 accumulator)
"""

__all__ = ["matmul_streamk"]

import gc
import logging

import torch

from cannbotdsl import make_layout, dtypes, get_mem_size, get_platform_info
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.ops.arch import (
    get_block_idx,
    get_block_num,
    get_subblock_dim,
    get_subblock_id,
)
from cannbotdsl.types.dtypes import float16, float32, bfloat16
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops import reg as rr
from cannbotdsl.lang import const_expr, vf
from cannbotdsl.tensor import MemLoc
from cannbotdsl.tensor import Tensor
from cannbotdsl.tensor import idx2crd, tile_slice
from cannbotdsl.ops.sync import global_sync_all
from cannbotdsl.ops.memcpy import copy_options, mem_copy

logger = logging.getLogger(__name__)

_DTYPE_MAP = {"fp16": float16, "bf16": bfloat16, "fp32": float32}
_DTYPE_SIZE = {"fp16": 2, "bf16": 2, "fp32": 4}
_TORCH_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
_TORCH_DTYPE_TO_STR = {v: k for k, v in _TORCH_DTYPE.items()}

TASK_RATIO = 2
WINDOW_LEN = 4

# on-chip capacities from the platform (mainline convention)
_L1_SIZE = get_mem_size("l1")
_L0A_SIZE = get_mem_size("l0a")
_UB_SIZE = get_mem_size("ub")
# arch35 constant: no platform query exposes the L2 capacity
_L2_SIZE = 128 * 1024 * 1024
_DB = 2
_BASIC_L1_BUF = 4
_ALIGN_128 = 128
_BLK16 = 16
_BLK128 = 128
_BLK256 = 256
_K128B = 128
_K256B = 256
_K512B = 512
_L1_SINGLE_LIMIT = 48 * 1024


def _ceil_div(a, b):
    return (a + b - 1) // b if b else a


def _ceil_align(a, align):
    return _ceil_div(a, align) * align if align else a


def _floor_align(a, b):
    return (a // b) * b if b else a


def _global_sync():
    global_sync_all()


@jit
def _cast_tile_cols(dst, src, row):
    """Convert one tile row in 64-element column chunks."""
    cols = dst.shape[1]
    src_stride = src.physical_stride[0]
    dst_stride = dst.physical_stride[0]
    for col in range(0, cols, 64):
        mask, _ = rr.update_mask(cols - col, elem_bits=32)
        if const_expr(src.dtype == dtypes.float32):
            value = rr.vload(src, row * src_stride + col)
        else:
            value = rr.vload_unpack(
                src,
                row * src_stride + col,
                unpack_mode=rr.UnpackMode.B16_TO_B32,
            )
        converted = rr.vcast(value, dst.dtype, mask=mask)
        if const_expr(dst.dtype == dtypes.float32):
            rr.vstore(dst, row * dst_stride + col, converted, mask)
        else:
            rr.vstore_pack(
                dst,
                row * dst_stride + col,
                converted,
                mask,
                pack_mode=rr.PackMode.B32_TO_B16,
            )


@jit
def cast_tile(dst, src):
    """Convert a 2D f32/f16/bf16 tile, honoring each operand's row stride."""
    rows, cols = dst.shape
    src_stride = src.physical_stride[0]
    dst_stride = dst.physical_stride[0]
    with vf():
        for row in range(rows):
            if const_expr(isinstance(cols, int) and cols == 1):
                # Compact scalar columns need scalar addressing: successive
                # rows need not be aligned for a full register load/store.
                value = rr.vload_broadcast(src, row * src_stride)
                converted = rr.vcast(value, dst.dtype, mask=rr.full_mask())
                rr.vstore_first(dst, row * dst_stride, converted)
            else:
                _cast_tile_cols(dst, src, row)


class StreamKTiling:
    """Inline tiling for Stream-K (DPSK) matmul on Ascend950 (DAV_3510).

    Derived from CANN ops-nn matmul_v3 STREAM_K strategy. Computes base_m/base_n/base_k,
    step_ka, L1 buffer depth, L0C double buffer, L2 cache control, and core count.

    Hardware capacities are queried from the platform (get_mem_size /
    get_platform_info); only the L2 size stays an arch35 constant.
    """

    def __init__(self, m, n, k, dtype="fp16"):
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.dtype_size = _DTYPE_SIZE[dtype]
        self.aic_num = get_platform_info().cube_core_num
        self._compute()

    def _compute(self):
        self._reset_base()
        self._stream_k_tiling()
        self.l1_buffer_num = _DB
        self.db_l0c = 1
        self._l2_cache_disable()

    def _reset_base(self):
        self.base_m = _BLK256
        self.base_n = _BLK256
        self.base_k = _K128B // self.dtype_size
        self.step_ka = 1
        self.step_kb = 1
        self.depth_a1 = 2
        self.depth_b1 = 2
        self.used_core_num = self.aic_num
        self.single_core_k = self.k
        self.db_l0c = 1

    def _stream_k_tiling(self):
        k_threshold = max(8192, self.aic_num * _K256B) // self.dtype_size
        if _ceil_align(self.k, _BLK256) < k_threshold:
            return

        align_value = _BLK256
        m_cnt = _ceil_div(self.m, align_value)
        n_cnt = _ceil_div(self.n, align_value)
        total_mn = m_cnt * n_cnt

        if total_mn > self.aic_num // 2:
            if not (
                self.m % _BLK256 == 0
                and self.n % _BLK256 == 0
                and total_mn >= self.aic_num
                and total_mn % self.aic_num != 0
                and total_mn % self.aic_num <= self.aic_num // 2
            ):
                return

        if total_mn <= self.aic_num // 2:
            if self.aic_num // 3 < m_cnt < self.aic_num // 2:
                m_cnt = self.aic_num // 2
            if self.aic_num // 3 < n_cnt < self.aic_num // 2:
                n_cnt = self.aic_num // 2
            total_mn = m_cnt * n_cnt
            self.base_m = _ceil_align(_ceil_div(self.m, m_cnt), _BLK16)
            self.base_n = _ceil_align(_ceil_div(self.n, n_cnt), _BLK16)
            k_cnt = self.aic_num // total_mn
            k_align = _BLK128 // self.dtype_size
            self.single_core_k = max(
                k_align, _floor_align(_ceil_div(self.k, k_cnt), k_align)
            )
        else:
            k_cnt = self.aic_num // (total_mn % self.aic_num)
            k_align = _BLK128 // self.dtype_size
            sk_single_core_k = max(
                k_align, _floor_align(_ceil_div(self.k, k_cnt), k_align)
            )
            k_cnt = _ceil_div(self.k, sk_single_core_k)
            self.single_core_k = sk_single_core_k

        base_k_align_value = _BLK128 // self.dtype_size
        k_value_max = _floor_align(
            _L0A_SIZE // _DB // self.dtype_size // max(self.base_m, self.base_n),
            base_k_align_value,
        )
        self.base_k = min(self.single_core_k, k_value_max)

        self._cal_l1_tiling()

        if self.base_m == self.base_n and self.depth_b1 == self.depth_a1 * 2:
            self.depth_a1 *= 2
            self.depth_b1 //= 2
            self.step_kb = self.depth_b1 // _DB
            self.step_ka = self.depth_a1 // _DB

    def _cal_l1_tiling(self):
        max_step_k = min(_ceil_div(self.k, self.base_k), 8)
        k_align_unit = _K512B // self.dtype_size
        res_k_l1 = 0
        single_mte_size = 0

        for step_k in range(1, max_step_k + 1):
            cur_k_l1 = self.base_k * step_k
            a_l1_size = self.base_m * cur_k_l1 * self.dtype_size
            b_l1_size = self.base_n * cur_k_l1 * self.dtype_size
            if (a_l1_size + b_l1_size) * _DB > _L1_SIZE:
                break
            if max(a_l1_size, b_l1_size) * _DB * 2 > _L1_SIZE:
                break

            cond_no_res = res_k_l1 == 0
            cond_k_align_256b = cur_k_l1 % (_K256B // self.dtype_size) == 0
            cond_k_align = res_k_l1 % k_align_unit != 0 and (
                cond_k_align_256b
                or (not cond_k_align_256b and single_mte_size < _L1_SINGLE_LIMIT)
            )
            cond_mte_size = (
                res_k_l1 % k_align_unit == 0
                and cur_k_l1 % k_align_unit == 0
                and single_mte_size < _L1_SINGLE_LIMIT
            )
            if cond_no_res or cond_k_align or cond_mte_size:
                res_k_l1 = cur_k_l1
                single_mte_size = max(a_l1_size, b_l1_size)

        self.step_ka = res_k_l1 // self.base_k if self.base_k > 0 else 1
        self.step_kb = self.step_ka
        self.depth_a1 = self.step_ka * _DB
        self.depth_b1 = self.step_kb * _DB

    def _l2_cache_disable(self):
        self.l2_cache_disable = 0

        n_l1 = min(_ceil_align(self.n, _BLK16), self.base_n)
        step_ka_min = min(self.step_kb, self.step_ka, _BASIC_L1_BUF)
        k_l1 = self.base_k * step_ka_min

        total_size = (
            self.m * self.n * self.dtype_size
            + self.m * self.k * self.dtype_size
            + self.k * self.n * self.dtype_size
        )
        if total_size < _L2_SIZE:
            self.l2_cache_disable = 0
            return

        flag_a = k_l1 * self.dtype_size % _ALIGN_128 == 0
        flag_b = n_l1 * self.dtype_size % _ALIGN_128 == 0
        left_not_l2 = self.base_n >= self.n and flag_a
        right_not_l2 = self.base_m >= self.m and flag_b

        if left_not_l2 and right_not_l2:
            self.l2_cache_disable = 3
        elif left_not_l2:
            self.l2_cache_disable = 1
        elif right_not_l2:
            self.l2_cache_disable = 2
        else:
            self.l2_cache_disable = 0


class MatmulStreamK:
    """Stream-K (DPSK) matmul kernel with host-side tiling.

    tiling flow:
      StreamKTiling -> base_m/base_n/base_k, step_ka, db_l0c, used_core_num, l1_buffer_num, l2_cache_disable
    """

    def __init__(self, m, n, k, dtype="fp16"):
        self.m, self.n, self.k = m, n, k
        self.in_dtype = _DTYPE_MAP[dtype]

        self.tiling = StreamKTiling(m, n, k, dtype=dtype)
        self._derive_tiling()
        self._compute_tile_partition()
        self._compute_sk_partition()
        self._compute_aiv_reduce()
        self._compute_workspace()

        # kernel channels/engines, created in _init_channels at trace time
        self.a_l1 = None
        self.b_l1 = None
        self.l0a = None
        self.l0b = None
        self.l0c = None
        self.stage = None
        self.out_stage = None

    def print_config(self):
        logger.info(f"DPSK: m={self.m} n={self.n} k={self.k}")
        logger.info(
            f"  tile={self.tile_m}x{self.tile_n}x{self.tile_k} slab_k={self.slab_k} step_ka={self.step_ka}"
        )
        logger.info(
            f"  blocks={self.block_num} m_tiles={self.m_tiles} n_tiles={self.n_tiles}"
        )
        logger.info(
            f"  dp={self.dp_tiles} tail={self.tail_tiles} k_splits={self.k_splits} sk_work={self.sk_work}"
        )
        logger.info(
            f"  reduce_m={self.reduce_m} aiv_per_tile={self.aiv_per_tile} total_aiv_work={self.total_aiv_work}"
        )
        logger.info(
            f"  l1_depth={self.l1_depth} l2_ctl_a={self.l2_cache_ctl_a} l2_ctl_b={self.l2_cache_ctl_b}"
        )
        logger.info(
            f"  window={self.main_window} main_row={self.main_row} tail_window={self.tail_window}"
        )

    @jit
    def _init_ring_buffers(self):
        dt = self.in_dtype
        self.a_l1 = RingBuffer(
            MemLoc.L1,
            dt,
            make_layout((self.tile_m, self.slab_k)),
            depth=self.l1_depth,
        )
        self.b_l1 = RingBuffer(
            MemLoc.L1,
            dt,
            make_layout((self.slab_k, self.tile_n)),
            depth=self.l1_depth,
        )
        self.l0a = RingBuffer(
            MemLoc.L0A,
            dt,
            make_layout((self.tile_m, self.tile_k)),
            depth=2,
        )
        self.l0b = RingBuffer(
            MemLoc.L0B,
            dt,
            make_layout((self.tile_n, self.tile_k)),
            depth=2,
        )
        self.l0c = RingBuffer(
            MemLoc.L0C,
            float32,
            make_layout((self.tile_m, self.tile_n)),
            depth=self.l0c_depth,
        )
        self.stage = RingBuffer(
            MemLoc.UB,
            float32,
            make_layout((max(self.k_splits, 1), self.reduce_m * self.tile_n)),
            depth=1,
        ).next_write()
        self.out_stage = RingBuffer(
            MemLoc.UB,
            dt,
            make_layout((self.reduce_m, self.tile_n)),
            depth=1,
        ).next_write()

    @jit
    def _mmad_from_l1(self, a_slot, b_slot, accumulator, first_slab):
        # L1 slab -> L0 sub-blocks -> mmad; the accumulator initialises
        # only on the very first sub-block of the first slab
        for ki in range(_ceil_div(a_slot.shape[1], self.tile_k)):
            a_source = tile_slice(a_slot, (self.tile_m, self.tile_k), (0, ki))
            b_source = tile_slice(b_slot, (self.tile_k, self.tile_n), (ki, 0)).permute(
                1, 0
            )
            # M/N stay fixed across K sub-blocks; only K can have a new tail.
            a_input = self.l0a.next_write(bare=True).view(
                make_layout((a_slot.shape[0], a_source.shape[1]))
            )
            mem_copy(a_input, a_source)
            b_input = self.l0b.next_write(bare=True).view(
                make_layout((b_slot.shape[1], b_source.shape[1]))
            )
            mem_copy(b_input, b_source)
            c_view = accumulator.bare().view(
                make_layout((a_input.shape[0], b_input.shape[0])),
                dtype=float32,
                data_format="nz",
            )
            matmul(c_view, a_input, b_input, init=first_slab and (ki == 0))

    @jit
    def _dp_path(self, out_gm: Tensor, a_gm: Tensor, b_gm: Tensor):
        block_idx = get_block_idx()
        block_num = get_block_num()

        for tile_idx in range(block_idx, self.dp_tiles, block_num):
            self._process_dp_tile(tile_idx, out_gm, a_gm, b_gm)

    @jit
    def _process_dp_tile(self, tile_idx, out_gm: Tensor, a_gm: Tensor, b_gm: Tensor):
        m_idx = 0
        n_idx = 0
        if self.tail_tiles == 0 and self.dp_tiles > 0:
            # slide window + row reversal for L2 locality
            n_tiles = self.n_tiles
            main_window = self.main_window
            main_row = self.main_row
            tail_window = self.tail_window
            row_idx = tile_idx // n_tiles // main_window if main_window > 0 else 0
            if main_row > 0 and row_idx < main_row:
                m_idx = row_idx * main_window + tile_idx % main_window
                n_idx = (tile_idx // main_window) % n_tiles
            else:
                row_idx = main_row
                tail_index = tile_idx - main_row * main_window * n_tiles
                m_idx = main_row * main_window + tail_index % tail_window
                n_idx = (tail_index // tail_window) % n_tiles

            if row_idx % 2 != 0:
                n_idx = (n_tiles - 1) - n_idx
        else:
            m_idx, n_idx, _ = idx2crd(tile_idx, [self.m_tiles, self.n_tiles, 1])
        m_start = m_idx * self.tile_m
        n_start = n_idx * self.tile_n
        m_count = min(self.tile_m, self.m - m_start)
        n_count = min(self.tile_n, self.n - n_start)
        a_panel = a_gm[m_start : m_start + m_count, None]
        b_panel = b_gm[None, n_start : n_start + n_count]
        output_tile = out_gm[m_start : m_start + m_count, n_start : n_start + n_count]
        accumulator = (
            self.l0c.next_write()
            .bare()
            .view(make_layout(output_tile.shape), dtype=float32, data_format="nz")
        )
        for sk in range(self.slab_tiles):
            k_start = sk * self.slab_k
            k_count = (
                self.slab_k
                if const_expr(self.k % self.slab_k == 0)
                else min(self.slab_k, self.k - k_start)
            )
            a_slot = self.a_l1.next_write()
            b_slot = self.b_l1.next_write()
            a_slot_source = a_panel[None, k_start : k_start + k_count]
            a_slot = a_slot[tuple(map(slice, a_slot_source.shape))]
            mem_copy(
                a_slot,
                a_slot_source,
                options=copy_options(l2_cache_ctl=self.l2_cache_ctl_a),
            )
            b_slot_source = b_panel[k_start : k_start + k_count, None]
            b_slot = b_slot[tuple(map(slice, b_slot_source.shape))]
            mem_copy(
                b_slot,
                b_slot_source,
                options=copy_options(l2_cache_ctl=self.l2_cache_ctl_b),
            )
            self._mmad_from_l1(a_slot, b_slot, accumulator, sk == 0)
        mem_copy(
            output_tile,
            accumulator,
        )

    @jit
    def _sk_path(self, ws_gm: Tensor, a_gm: Tensor, b_gm: Tensor):
        block_idx = get_block_idx()
        block_num = get_block_num()

        for work_idx in range(block_idx, self.sk_work, block_num):
            tail_mn_local, k_split, _ = idx2crd(
                work_idx, [self.tail_tiles, self.k_splits, 1]
            )
            tail_mn_global = self.dp_tiles + tail_mn_local
            m_idx = tail_mn_global // self.n_tiles
            n_idx = tail_mn_global - m_idx * self.n_tiles

            slab_start = k_split * self.slabs_per
            total_slabs = _ceil_div(self.k, self.slab_k_sk)
            m_start = m_idx * self.tile_m
            n_start = n_idx * self.tile_n
            m_count = min(self.tile_m, self.m - m_start)
            n_count = min(self.tile_n, self.n - n_start)
            a_panel = a_gm[m_start : m_start + m_count, None]
            b_panel = b_gm[None, n_start : n_start + n_count]
            accumulator = (
                self.l0c.next_write()
                .bare()
                .view(make_layout((m_count, n_count)), dtype=float32, data_format="nz")
            )
            for s in range(self.slabs_per):
                sk = slab_start + s
                if sk < total_slabs:
                    k_start = sk * self.slab_k_sk
                    k_count = (
                        self.slab_k_sk
                        if const_expr(self.k % self.slab_k_sk == 0)
                        else min(self.slab_k_sk, self.k - k_start)
                    )
                    a_slot = self.a_l1.next_write()
                    b_slot = self.b_l1.next_write()
                    a_slot_source_2 = a_panel[None, k_start : k_start + k_count]
                    a_slot = a_slot[tuple(map(slice, a_slot_source_2.shape))]
                    mem_copy(
                        a_slot,
                        a_slot_source_2,
                        options=copy_options(l2_cache_ctl=self.l2_cache_ctl_a),
                    )
                    b_slot_source_2 = b_panel[k_start : k_start + k_count, None]
                    b_slot = b_slot[tuple(map(slice, b_slot_source_2.shape))]
                    mem_copy(
                        b_slot,
                        b_slot_source_2,
                        options=copy_options(l2_cache_ctl=self.l2_cache_ctl_b),
                    )
                    self._mmad_from_l1(a_slot, b_slot, accumulator, s == 0)

            ws_tile_idx = tail_mn_global * self.k_splits + k_split
            ws_start = ws_tile_idx * self.tile_m
            # MMAD writes complete 16-column NZ fragments. Preserve those
            # columns in the padded workspace so FixPipe writes aligned rows
            # instead of one narrow burst per row (notably N=1). Reduction
            # and the final store still use the live output M/N domain.
            workspace_n = _ceil_div(n_count, 16) * 16
            workspace_result = accumulator.bare().view(
                make_layout((m_count, workspace_n)), dtype=float32, data_format="nz"
            )
            ws_dst = ws_gm[ws_start : ws_start + m_count, :workspace_n]
            mem_copy(ws_dst, workspace_result)

    @jit
    def _accumulate_splits(self, stage, acc, offset, mask, slab_stride):
        running_sum = rr.vload(stage, offset)
        for ks in range(1, self.k_splits):
            value = rr.vload(stage, ks * slab_stride + offset)
            running_sum = rr.vadd(running_sum, value, mask=mask)
        rr.vstore(acc, offset, running_sum, mask)

    @jit
    def _reduce_partial_sums(self, stage, acc):
        rows, cols = acc.shape
        acc_stride = acc.physical_stride[0]
        slab_stride = self.reduce_m * self.tile_n
        with vf():
            for row in range(rows):
                for col in range(0, cols, 64):
                    mask, _ = rr.update_mask(cols - col, elem_bits=32)
                    self._accumulate_splits(
                        stage, acc, row * acc_stride + col, mask, slab_stride
                    )

    @jit
    def _aiv_reduce_path(self, out_gm: Tensor, ws_gm: Tensor):
        ws_gm_flat = ws_gm.view(
            self.total_mn * max(self.k_splits, 1),
            self.tile_m * self.tile_n,
        )
        aiv_global_idx = get_block_idx() * get_subblock_dim() + get_subblock_id()
        aiv_global_num = get_block_num() * get_subblock_dim()

        for aiv_work_idx in range(aiv_global_idx, self.total_aiv_work, aiv_global_num):
            mn_tile_idx = aiv_work_idx // self.aiv_per_tile
            m_seg = aiv_work_idx % self.aiv_per_tile

            if m_seg * self.reduce_m < self.tile_m:
                tail_mn_global = self.dp_tiles + mn_tile_idx
                m_idx = tail_mn_global // self.n_tiles
                n_idx = tail_mn_global - m_idx * self.n_tiles

                # Workspace tiles are fully allocated. The output alone has
                # an M/N tail; compute its live extents once below.
                ws_row = tail_mn_global * self.k_splits
                ws_col = m_seg * self.reduce_m * self.tile_n
                ws_count = (
                    min(self.reduce_m, self.tile_m - m_seg * self.reduce_m)
                    * self.tile_n
                )
                seg = ws_gm_flat[
                    ws_row : ws_row + self.k_splits, ws_col : ws_col + ws_count
                ]
                mem_copy(self.stage, seg)

                acc = self.stage[0:1, 0 : self.reduce_m * self.tile_n].reshape(
                    self.reduce_m, self.tile_n
                )
                if self.k_splits > 1:
                    self._reduce_partial_sums(self.stage, acc)

                cast_tile(self.out_stage, acc)
                m_start = m_idx * self.tile_m + m_seg * self.reduce_m
                n_start = n_idx * self.tile_n
                m_actual = max(
                    0,
                    min(
                        self.reduce_m,
                        min(self.tile_m - m_seg * self.reduce_m, self.m - m_start),
                    ),
                )
                n_actual = min(self.tile_n, self.n - n_start)
                out_tile = out_gm[
                    m_start : m_start + m_actual, n_start : n_start + n_actual
                ]
                out_slice = self.out_stage[0:m_actual, 0:n_actual]
                mem_copy(out_tile, out_slice)

    @kernel
    def streamk_kernel(self, out_gm: Tensor, ws_gm: Tensor, a_gm: Tensor, b_gm: Tensor):
        self._init_ring_buffers()
        if const_expr(self.dp_tiles > 0):
            self._dp_path(out_gm, a_gm, b_gm)
        if const_expr(self.sk_work > 0):
            self._sk_path(ws_gm, a_gm, b_gm)
            _global_sync()
            self._aiv_reduce_path(out_gm, ws_gm)

    @host
    def run(self, out_gm: Tensor, ws_gm: Tensor, a_gm: Tensor, b_gm: Tensor):
        self.streamk_kernel[self.block_num](out_gm, ws_gm, a_gm, b_gm)

    def _derive_tiling(self):
        t = self.tiling
        self.tile_m = t.base_m
        self.tile_n = t.base_n
        self.tile_k = t.base_k
        self.block_num = t.used_core_num
        self.step_ka = max(1, min(t.step_ka, 4))
        self.slab_k = self.tile_k * self.step_ka
        self.l0c_depth = t.db_l0c
        self.l1_depth = t.l1_buffer_num

    def _compute_tile_partition(self):
        self.m_tiles = _ceil_div(self.m, self.tile_m)
        self.n_tiles = _ceil_div(self.n, self.tile_n)
        self.k_tiles = _ceil_div(self.k, self.tile_k)
        self.slab_tiles = _ceil_div(self.k, self.slab_k) if self.slab_k <= self.k else 1
        self.total_mn = self.m_tiles * self.n_tiles

        self.tail_tiles = self.total_mn % self.block_num
        self.dp_tiles = self.total_mn - self.tail_tiles

        if 0 < self.tail_tiles and self.tail_tiles > self.block_num // 2:
            self.dp_tiles = self.total_mn
            self.tail_tiles = 0

    def _compute_sk_partition(self):
        if self.tail_tiles > 0:
            if self.total_mn <= self.block_num // 2:
                self.k_splits = self.block_num // self.total_mn
            else:
                self.k_splits = self.block_num // self.tail_tiles
            k_single = _ceil_div(self.k_tiles, self.k_splits)
            self.k_splits = _ceil_div(self.k_tiles, k_single)
            self.k_tiles_per = k_single
            self.step_ka_sk = min(self.step_ka, self.k_tiles_per)
            self.slab_k_sk = self.tile_k * self.step_ka_sk
            self.slabs_per = max(1, _ceil_div(self.k_tiles_per, self.step_ka_sk))
            total_slabs_sk = _ceil_div(self.k, self.slab_k_sk)
            self.k_splits = _ceil_div(total_slabs_sk, self.slabs_per)
            self.slab_tiles_sk = (
                (self.k_tiles_per * self.tile_k) // self.slab_k_sk
                if self.slab_k_sk > 0
                else 1
            )
            self.sk_work = self.tail_tiles * self.k_splits
        else:
            self.k_splits = 1
            self.k_tiles_per = self.k_tiles
            self.step_ka_sk = self.step_ka
            self.slab_k_sk = self.slab_k
            self.slabs_per = self.slab_tiles
            self.slab_tiles_sk = self.slab_tiles
            self.sk_work = 0

    def _compute_aiv_reduce(self):
        ideal_segs = TASK_RATIO * max(self.k_splits, 1)
        aiv_total = self.block_num * TASK_RATIO
        max_by_blocks = (
            aiv_total // max(self.tail_tiles, 1) if self.tail_tiles > 0 else 1
        )

        self.aiv_per_tile = max(1, min(ideal_segs, max_by_blocks))
        while True:
            reduce_m_candidate = _ceil_div(self.tile_m, self.aiv_per_tile)
            if 2 * reduce_m_candidate * self.tile_n * 4 <= _UB_SIZE:
                break
            if self.aiv_per_tile < max_by_blocks:
                self.aiv_per_tile += 1
            else:
                self.aiv_per_tile += 1
                if self.aiv_per_tile > self.tile_m:
                    break
        self.reduce_m = _ceil_div(self.tile_m, self.aiv_per_tile)
        max_m_seg = max(1, _ceil_div(self.m, self.reduce_m))
        self.aiv_per_tile = min(self.aiv_per_tile, max_m_seg)
        self.total_aiv_work = self.tail_tiles * self.aiv_per_tile

    def _compute_workspace(self):
        self.ws_rows = self.total_mn * max(self.k_splits, 1) * self.tile_m
        self.ws_cols = self.tile_n

        l2_dis = self.tiling.l2_cache_disable
        self.l2_cache_ctl_a = 0 if l2_dis in (1, 3) else 1
        self.l2_cache_ctl_b = 0 if l2_dis in (2, 3) else 1

        self.main_window = min(WINDOW_LEN, self.m_tiles)
        self.main_row = (
            self.m_tiles // self.main_window - 1 if self.main_window > 0 else 0
        )
        self.tail_window = self.m_tiles - self.main_row * self.main_window


def matmul_streamk(a, b, transpose_a=False, transpose_b=False):
    """Torch-facing wrapper for the Stream-K (DPSK) matmul kernel.

    The kernel computes C[M, N] = A[M, K] @ B[K, N]. This wrapper normalizes
    the input layout according to transpose flags, delegates tiling to
    StreamKTiling, and synchronously launches the JIT-compiled MIX kernel.
    """
    a_kern = a.t().contiguous() if transpose_a else a
    b_kern = b if transpose_b else b.t().contiguous()
    m, k = a_kern.shape
    n = b_kern.shape[0]
    dtype_str = (
        "fp16"
        if a.dtype == torch.float16
        else ("bf16" if a.dtype == torch.bfloat16 else "fp32")
    )
    op = MatmulStreamK(m, n, k, dtype=dtype_str)
    c = torch.zeros(m, n, dtype=a.dtype, device=a_kern.device)
    ws = torch.zeros(op.ws_rows, op.ws_cols, dtype=torch.float32, device=a_kern.device)
    op.run(c, ws, a_kern, b_kern)
    torch.npu.synchronize()
    del ws, op
    gc.collect()
    torch.npu.empty_cache()
    return c
