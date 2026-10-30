# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MX-quantized matmul with fp8 activations and fp4 weights (MXA8W4).

Structure:
  1. MixQuantTiling       — host-side tiling
  2. MixQuantKernel       — @kernel class with AIV weight prologue + AIC MX matmul
  3. matmul_mix_quant()   — torch interface

Formula:
  C[M,N] = (A[M,K] * sa) @ (B[N,K] * sb)^T + bias
  A: fp8 e4m3, per-32-group E8M0 scale sa (MXFP8 activation)
  B: packed fp4 e2m1, per-32-group E8M0 scale sb (MXFP4 weight)
  C: fp16

Weight prologue (vector core): fp4 weights cannot feed the cube directly.
Per L1 K-window, B travels GM -> UB (packed bytes) -> raw-VF nibble-unpack
+ e2m1->e4m3 conversion -> NZ-fractal scatter in UB -> L1 -> L0B. The
conversion is a pure bit permutation whose values are exactly 1/64 of the
e2m1 magnitudes (all 16 codes, subnormals included):

    nibble [s e e m] -> byte [s 0 0 e e m 0 0]   (e4m3: E=00ee, M=m00)

The 1/64 is compensated by the fixpipe's deq_scale=64 and by scaling the
bias x 2^-6 on the vector core (mirroring the official ScaleMxBias).

MX semantics: both L0 loads bind an E8M0 scale (OCP MXFP8 mmad); the bias
rides the mmad BT path (hardware applies it on the init mmad only).

K handling: every K-sized on-chip structure is a k_l1 = k_blocks_per_window*k_l0 ring,
so K is unbounded. The consume runs k_blocks_per_window L0 sub-blocks per L1 window in
runtime loops. K tails (K % 256 != 0) need NO GM padding: the channel-
actual mechanism narrows the L0 loads and the mmad n_dim to the window's
runtime width (published by the A staging's channel dst and the B piece
copy's DualParam actual), and the scale loads narrow in lockstep.

Cross-core handoff (AIV -> AIC, flash_attn/flash_kda pattern): the
converted B travels a k_l1-wide CrossCore L1 ring; AIV0/AIV1 each write
their row half of a window (DualParam split-M) and the AIC's L0B load is
the consume side. Both sides advance the ring one channel epoch per
runtime window iteration, so produce runs at most `depth` windows ahead
of consume.

GM layouts (matching the reference weight_quant_batch_matmul_mx NZ
variant):
  b:        NZ-fractal packed fp4, reference global grid [K/32, N/16,
            16, 16] — tile (k_fractal, n_group) at flat row k_fractal*n_groups_total + n_group, k-adjacent
            packed 2/byte (even k low nibble); one (32k x 16n) tile =
            16 contiguous bytes per n row
  scale_a:  (M, S) uint8 E8M0, S = align(K, 64)//32, row-major
  scale_b:  (N, S) uint8 E8M0, N-major (ScaleBDN)

The torch wrapper accepts the ND (N, K//2) packed tensor and repacks to
NZ on device (keeps the public API stable). The two AIVs split each
window's n-groups exactly in half; each converts its groups and writes
its row half of the ring slot.
"""

__all__ = ["matmul_mix_quant"]

import dataclasses
import logging
import os

import numpy as np
import torch

from cannbotdsl import make_layout, MemLoc, Tensor, dtypes, get_mem_size, vf
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx, get_subblock_id
from cannbotdsl.ops.matmul import matmul as dsl_matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy, convert
from cannbotdsl.ops.sync import cube_fill_l1_zero
from cannbotdsl.ops.reg import (
    create_mask,
    vload,
    vstore,
    vstore_interleave,
    vmuls,
    vshl,
    vshr,
    vreinterpret,
    vmem_bar,
    vbitwise_and as vand,
    vbitwise_or as vor,
    vdups as vdup_scalar,
)
from cannbotdsl.tensor import tile_slice

logger = logging.getLogger(__name__)


# host-side helpers: the framework's ceil_div is an IR Shape op that needs
# an active lowering context (it crashes from plain host code), so host
# call sites use these private versions; the module-level name stays free
# for the framework's ceil_div inside @jit code.
def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _ceil_align(a: int, b: int) -> int:
    return (a + b - 1) // b * b


# ============================================================================
# 1. Host-side tiling
# ============================================================================


class MixQuantTiling:
    """Cost-model tiling for the MXA8W4 kernel (matmul_v3-style).

    1. Candidate base blocks (baseM, baseN, baseK) filtered by the L0
       ping-pong budgets (L0 depth = 2): L0A/L0B baseM/baseN * baseK * 2
       <= 64 KB each (fp8); L0C baseM*baseN*4 <= 256 KB (fp32). baseK is
       the K window (multiple of 64; 256 = one full MX mmad block).
    2. Score by a roofline estimate per core:
         t_mem  = GM bytes per core / HBM_BW   (A re-read per N tile, B
                   NZ grid per M tile, scales, C written once)
         t_cube = 2*M*N*K / cores / CUBE_TPUT
       Under-occupancy raises per-core bytes, so small blocks emerge on
       small shapes without a special case.
    3. Tail-shrink baseM/baseN to the shape (ceil-16).
    4. L1 K window: k_l1 = k_blocks_per_window * baseK, k_blocks_per_window in {1,2,4} — the L1
       window is decoupled from the L0/mmad block so the A staging's GM
       row burst grows to k_l1 bytes.
    5. L1 ring depths from the remaining L1 budget, clamped to [2, 4] —
       a deeper ring lets the AIV conversion run further ahead.
    6. Core grid: n-major with m groups.

    Set MQ_TILING_DEBUG=1 to print the candidate table and the choice.
    """

    # arch35 platform capacities via get_mem_size (mainline convention)
    L1_BYTES = get_mem_size("l1")
    L0A_BYTES = get_mem_size("l0a")
    L0B_BYTES = get_mem_size("l0b")
    L0C_BYTES = get_mem_size("l0c")
    L0_DEPTH = 2
    # MIX_AIC launch grid: the platform-reported core count (queried
    # lazily — module import must stay NPU-free)
    aic_core_count = 0
    _platform_queried = False
    HBM_BW = 1.4  # TB/s-class weight for the roofline score
    CUBE_TPUT = 864.0  # GOPS-class weight (qbmm MXFP8 constants)
    MAX_RING_DEPTH = 4
    MAX_K_L1 = 1024  # L1 K window cap (budget-checked per candidate)
    UB_BUDGET_BYTES = 200 * 1024  # per-AIV budget for packed + b_converted

    def __init__(self, m: int, n: int, k: int):
        self.m = m
        self.n = n
        self.k = k
        self._query_platform_cores()
        self._pick_block()
        self._pick_step()
        self._pick_depths()
        self._derive()
        self._report()

    @classmethod
    def _query_platform_cores(cls):
        # lazy: module import must stay NPU-free
        if cls._platform_queried:
            return
        cls._platform_queried = True
        from cannbotdsl import get_platform_info

        cls.aic_core_count = int(get_platform_info().core_num)

    # ---- 1/2: candidate base blocks under L0 budgets, roofline pick ----
    def _fits_l0(self, mb: int, nb: int, k_l1: int) -> bool:
        if mb * k_l1 * 1 * self.L0_DEPTH > self.L0A_BYTES:
            return False
        if nb * k_l1 * 1 * self.L0_DEPTH > self.L0B_BYTES:
            return False
        # L0C single-buffered for big tiles (the official
        # IsL0Feasible carries no DB factor on C)
        if mb * nb * 4 > self.L0C_BYTES:
            return False
        return mb >= 64 and nb >= 64

    def _candidates(self):
        # 16-multiple blocks let the tile count hit the core grid exactly
        # (pow2-only sets strand 5-multiple-M shapes on part of the grid)
        out = []
        for k_l1 in (256, 128):
            for mb in (256, 192, 160, 128, 96, 64):
                out.extend(
                    (mb, nb, k_l1)
                    for nb in (256, 192, 160, 128, 96, 64)
                    if self._fits_l0(mb, nb, k_l1)
                )
        return out

    def _pick_block(self):
        best_candidate = None
        scored_rows = []
        for block_m, block_n, k_l1 in self._candidates():
            m_tile_count = _ceil_div(self.m, block_m)
            n_tile_count = _ceil_div(self.n, block_n)
            candidate_tile_count = m_tile_count * n_tile_count
            # makespan-aware effective cores: the run lasts the SLOWEST
            # core, so tiles spread over the grid leave only a few
            # tile-slots per busy core
            max_tiles_per_core = _ceil_div(candidate_tile_count, self.aic_core_count)
            effective_cores = max(1, candidate_tile_count // max_tiles_per_core)
            # re-staging traffic: A (fp8, per N tile) + B NZ grid (fp4,
            # per M tile) + scales (1B/32K both sides) + C (fp16)
            a_bytes = m_tile_count * n_tile_count * block_m * self.k
            b_bytes = m_tile_count * n_tile_count * block_n * (self.k // 2)
            scale_bytes = (
                (
                    m_tile_count * n_tile_count * block_m
                    + m_tile_count * n_tile_count * block_n
                )
                * _ceil_align(self.k, 64)
                // 32
            )
            c_bytes = self.m * self.n * 2
            t_mem = (
                (a_bytes + b_bytes + scale_bytes + c_bytes)
                / effective_cores
                / self.HBM_BW
            )
            t_cube = 2.0 * self.m * self.n * self.k / effective_cores / self.CUBE_TPUT
            # prefer the full MX window on ties (fewer ring round-trips)
            score = (max(t_mem, t_cube), -k_l1)
            scored_rows.append(
                (
                    score,
                    (block_m, block_n, k_l1),
                    candidate_tile_count,
                    effective_cores,
                    t_mem,
                    t_cube,
                )
            )
            if best_candidate is None or score < best_candidate[0]:
                best_candidate = (score, (block_m, block_n, k_l1))
        self._candidate_table = scored_rows
        self.tile_m_before_shrink, self.tile_n_before_shrink, self.k_l0 = (
            best_candidate[1]
        )
        # tail-shrink (ceil-16): small M/N never pay padded rows
        self.tile_m = _ceil_align(min(self.m, self.tile_m_before_shrink), 16)
        self.tile_n = _ceil_align(min(self.n, self.tile_n_before_shrink), 16)
        # L0C depth: double-buffered while it fits, single for big tiles
        self.l0c_depth = 2 if self.tile_m * self.tile_n * 8 <= self.L0C_BYTES else 1

    # ---- 3/2: L1 K window (k_blocks_per_window L0 blocks per ring epoch) ----
    def _pick_step(self):
        # Widen the A staging's GM row burst by carrying k_blocks_per_window
        # L0 blocks per ring epoch, so one nd2nz row covers k_l1 contiguous
        # bytes. k_blocks_per_window need not divide k_block_count: a partial
        # last L1 window is just a narrower epoch, with k_window_count given
        # by the ceiling of k_block_count over k_blocks_per_window.
        self.k_blocks_per_window = 1
        self.b_ring_single_slot = False
        k_block_count = _ceil_div(self.k, self.k_l0)
        n_groups_per_tile = self.tile_n // 16
        for step_candidate in (4, 3, 2):
            k_l1 = step_candidate * self.k_l0
            if k_l1 > self.MAX_K_L1 or step_candidate > k_block_count:
                continue
            a_slot_bytes = self.tile_m * k_l1
            b_slot_bytes = self.tile_n * k_l1
            fixed_scale_bytes = (self.tile_m + self.tile_n) * (
                k_l1 // 32
            ) + self.tile_n * 4
            packed_bytes_per_window = (k_l1 // 32) * n_groups_per_tile * 256
            if packed_bytes_per_window + self.tile_n * k_l1 > self.UB_BUDGET_BYTES:
                continue
            if 2 * a_slot_bytes + 2 * b_slot_bytes + fixed_scale_bytes <= self.L1_BYTES:
                self.k_blocks_per_window = step_candidate
                break
            if 2 * a_slot_bytes + b_slot_bytes + fixed_scale_bytes <= self.L1_BYTES:
                # asymmetric ring: keep A's double-buffered lookahead
                # (the MTE2-issue bottleneck), trade the B ring to a
                # single slot for a wider k_l1
                self.k_blocks_per_window = step_candidate
                self.b_ring_single_slot = True
                break
        self.k_l1 = self.k_blocks_per_window * self.k_l0

    # ---- 4: L1 ring depths (CalL1Tiling depthA1/depthB1 style) ----
    def _pick_depths(self):
        k_l1 = self.k_l1
        a_slot_bytes = self.tile_m * k_l1
        b_slot_bytes = self.tile_n * k_l1
        scale_a_slot_bytes = self.tile_m * (k_l1 // 32)
        scale_b_slot_bytes = self.tile_n * (k_l1 // 32)
        fixed_scale_bytes = (
            scale_a_slot_bytes + scale_b_slot_bytes + self.tile_n * 4
        )  # bias BT row
        half_l1_bytes = self.L1_BYTES // 2
        ring_depth_a = max(
            2, min(self.MAX_RING_DEPTH, half_l1_bytes // max(1, a_slot_bytes))
        )
        if self.b_ring_single_slot:
            ring_depth_b = 1
        else:
            ring_depth_b = max(
                2, min(self.MAX_RING_DEPTH, half_l1_bytes // max(1, b_slot_bytes))
            )
        # joint check: shed the deeper ring's depth on overflow
        while (
            ring_depth_a * a_slot_bytes
            + ring_depth_b * b_slot_bytes
            + fixed_scale_bytes
            > self.L1_BYTES
        ):
            if a_slot_bytes <= b_slot_bytes and ring_depth_a > 2:
                ring_depth_a -= 1
            elif ring_depth_b > 2 or (self.b_ring_single_slot and ring_depth_b > 1):
                ring_depth_b -= 1
            elif ring_depth_a > 2:
                ring_depth_a -= 1
            else:
                break
        self.ring_depth_a = ring_depth_a
        self.ring_depth_b = ring_depth_b

    # ---- derived fields (kernel/wrapper contract) ----
    def _derive(self):
        self.k_block_count = _ceil_div(self.k, self.k_l0)
        # ceil keeps full K coverage when k_blocks_per_window does not divide k_block_count
        self.k_window_count = _ceil_div(self.k_block_count, self.k_blocks_per_window)
        # repacked B grid height in k-fractals (the AIV fetch bound and
        # the host repack target): align64(k)/32; beyond-k rows are real
        # zero-filled GM
        self.k_fractals_in_grid = _ceil_align(self.k, 64) // 32
        # K-tail model: full-width window claims; the wrapper pads only the
        # E8M0 scale columns to the k_l1 grid (A columns never pad — the
        # kernel zero-fills the stale L1 tail fractals via cube_fill_l1_zero)
        self.k_fractals_per_window = self.k_l1 // 32
        self.scale_bytes_per_window = self.k_l1 // 32

        self.tile_rows = _ceil_div(self.m, self.tile_m)
        self.tile_cols = _ceil_div(self.n, self.tile_n)
        self.tile_count = self.tile_rows * self.tile_cols

        self.n_groups_per_tile = self.tile_n // 16
        self.n_groups_total = self.tile_cols * self.n_groups_per_tile
        self.packed_bytes_per_window = (
            self.k_fractals_per_window * self.n_groups_per_tile * 256
        )

        # plain core cap — the balanced contiguous assignment in the
        # kernel needs no n/m grouping factors
        self.launch_block_count = min(self.tile_count, self.aic_core_count)
        self.scale_a_grid_rows = self.tile_rows * self.tile_m
        self.scale_b_grid_rows = self.tile_cols * self.tile_n

    def _report(self):
        if os.environ.get("MQ_TILING_DEBUG") != "1":
            return
        logger.info(
            f"[tiling] M={self.m} N={self.n} K={self.k} -> "
            f"block ({self.tile_m}x{self.tile_n}, k_l0={self.k_l0}) "
            f"k_blocks_per_window={self.k_blocks_per_window} k_l1={self.k_l1} "
            f"depth a/b={self.ring_depth_a}/{self.ring_depth_b}"
            f"{' (b1)' if self.b_ring_single_slot else ''} "
            f"tiles={self.tile_rows}x{self.tile_cols} "
            f"cores={self.launch_block_count}"
        )
        for score, blk, tiles, effective_cores, t_mem, t_cube in sorted(
            self._candidate_table
        ):
            logger.info(
                f"    cand {blk}: tiles={tiles} cores={effective_cores} "
                f"t_mem={t_mem:.1f} t_cube={t_cube:.1f} "
                f"score={score[0]:.1f}"
            )


# ============================================================================
# 2. Kernel
# ============================================================================


@kernel
class MixQuantKernel:
    """AIV weight prologue + AIC MX matmul with explicit cross-core sync."""

    def __init__(self, t: MixQuantTiling, has_bias: bool):
        self.t = t
        self.has_bias = has_bias
        # consumed BIAS slot handed to the init mmad; __call__ fills it in
        self.bias_tensor = None
        # GM operands and per-window consumed slots; __call__ and
        # _process_k_window fill them in at trace time
        self.gm_a = None
        self.gm_b = None
        self.gm_sa = None
        self.gm_sb = None
        self.gm_bias = None
        self.a_window_slot = None
        self.scale_a_window_slot = None
        self.scale_b_window_slot = None
        self.b_window_slot = None
        self._init_vector_ring_buffers()
        self._init_cube_ring_buffers()

    def __call__(
        self,
        gm_a: Tensor,
        gm_b: Tensor,
        gm_sa: Tensor,
        gm_sb: Tensor,
        gm_bias: Tensor,
        gm_c: Tensor,
    ):
        t = self.t
        block_idx = get_block_idx()
        # GM operands on self: the @jit helpers read them without long
        # parameter lists
        self.gm_a = gm_a
        self.gm_b = gm_b
        self.gm_sa = gm_sa
        self.gm_sb = gm_sb
        self.gm_bias = gm_bias

        # balanced contiguous tile assignment: each block takes a
        # contiguous run of the n-major linear tile grid, run sizes
        # differing by at most one tile
        tile_total = t.tile_rows * t.tile_cols
        min_run_len = tile_total // t.launch_block_count
        extra_tiles = tile_total - min_run_len * t.launch_block_count
        run_start = block_idx * min_run_len
        run_len = min_run_len
        if block_idx < extra_tiles:
            run_start = run_start + block_idx
            run_len = run_len + 1
        else:
            run_start = run_start + extra_tiles
        for run_iter in range(run_len):
            tile_linear_idx = run_start + run_iter
            tile_n_idx = tile_linear_idx // t.tile_rows
            tile_m_idx = tile_linear_idx - tile_n_idx * t.tile_rows
            self.produce_bias(tile_n_idx)
            self.stage_tile()
            if const_expr(self.has_bias):
                self.bias_tensor = self.mmad_bias.next_read()
            # ONE accumulator slot per output tile feeds every mmad and
            # the fixpipe
            l0c_accumulator = self.l0c.next_write()
            # consume: every L1 window claims the FULL k_l1 width; the
            # partial last window's beyond-k contribution annihilates
            # (A's L1-zeroed tail x finite-converted B = 0 — see the
            # fill in _process_k_window); the static k_blocks_per_window
            # trip serves every window uniformly
            for k_window_idx in range(t.k_window_count):
                self._process_k_window(
                    k_window_idx, tile_m_idx, tile_n_idx, l0c_accumulator
                )

            # single fixpipe per output tile (unit_flag=3 handshake);
            # deq_scale=64 compensates the conversion's uniform 2^-6
            # factor (the AIV scales the bias to match, produce_bias)
            mem_copy(
                gm_c[
                    tile_m_idx * t.tile_m : (tile_m_idx + 1) * t.tile_m,
                    tile_n_idx * t.tile_n : (tile_n_idx + 1) * t.tile_n,
                ],
                convert(l0c_accumulator, gm_c.dtype, scale=64.0),
                options=copy_options(unit_flag=3),
            )

    @jit
    def _process_k_window(self, k_window_idx, tile_m_idx, tile_n_idx, l0c_accumulator):
        t = self.t
        self.stage_k_window(k_window_idx, tile_m_idx, tile_n_idx)
        self.produce_k_window(tile_n_idx, k_window_idx, t.k_l1)
        # the window's consumed slots (ONE consume per window —
        # the sub-block loop below slices them repeatedly)
        self.a_window_slot = self.a_ring.next_read()
        self.scale_a_window_slot = self.scale_a_ring.next_read().permute(1, 0)
        self.scale_b_window_slot = self.scale_b_ring.next_read()
        self.b_window_slot = self.b_ring.next_read()
        if const_expr(t.k % t.k_l1):
            # the partial last window: the staging covers the valid prefix
            # plus the nd2nz auto-fill; zero the remaining L1 k-fractals so
            # the full-width mmad's excess K annihilates against A's zeroed
            # tail. The slot's k-fractals are tile_m rows of contiguous
            # 32-byte blocks.
            if k_window_idx == t.k_window_count - 1:
                k_last_window_valid = t.k - k_window_idx * t.k_l1
                first_stale_fractal = (k_last_window_valid + 63) // 64 * 64 // 32
                cube_fill_l1_zero(
                    self.a_window_slot,
                    offset_elems=first_stale_fractal * t.tile_m * 32,
                    repeat=1,
                    blk_num=(t.k_l1 // 32 - first_stale_fractal) * t.tile_m,
                    dst_gap=0,
                )
        for sub_block_idx in range(t.k_blocks_per_window):
            self.consume_block(k_window_idx, sub_block_idx, l0c_accumulator)

    # ---- vector side: fp4 -> fp8 conversion + NZ scatter + L1 handoff ----
    @jit
    def convert_kspan(
        self,
        n_group_lo,
        n_group_count,
        fresh_fractal_count,
        converted_b_slot,
        b_packed_slot,
    ):
        """Convert the FRESH NZ tiles [0, fresh_fractal_count) x n-groups
        [n_group_lo, n_group_lo+n_group_count) of the staged window into
        the NZ cache, then ZERO-WRITE the stale tiles up to the window's
        full fractal count (t.k_fractals_per_window).

        Short names inside the hot loops (kept for line-width
        readability of the strength-reduced addressing):
          ngrps             n_groups_per_tile
          p_base / s_row    packed: first group's 256B block / per-fractal advance
          kdst              converted slot: k-fractal base (+= k_fractal_stride)
          p_addr / dst      per-tile running offsets inside a fractal
          g                 n-group loop index
          p                 one 256B packed tile (vload result)

        The packed slot holds the window's NZ tiles contiguously (tile
        (kb, g) at byte (kb*n_groups_per_tile + g)*256): one vload reads a whole
        256B tile; the nibble-unpack + e2m1->e4m3 conversion produces
        conv_lo (even-k) / conv_hi (odd-k); ONE vstore_interleave writes
        both interleaved directly into the cache at the tile's NZ
        position (the interleave output IS the NZ row-contiguous
        layout). The stale tail (beyond the fetch bound) skips the load
        and permutation and stores zeros — the full-width piece copy
        still needs finite values there, and zeros annihilate against
        A's zeroed L1 tail. The dynamic loop skeleton stays
        (constexpr-unrolled bodies trip the legacy rank check).
        """
        t = self.t
        s = converted_b_slot.physical_stride
        k_fractal_stride, n_group_stride = s[0], s[1]
        ngrps = t.n_groups_per_tile
        first_group = n_group_lo
        group_end = n_group_lo + n_group_count
        # running bases (strength reduction); the window's fractals
        # start at packed byte 0 / cache row 0
        p_base = first_group * 256
        s_row = ngrps * 256
        kdst = 0
        with vf():
            mask8 = create_mask(pattern="all", elem_bits=8)
            sign_mask = vdup_scalar(0x80, dtypes.uint8)
            exp_mant_mask = vdup_scalar(0x1C, dtypes.uint8)
            zero_f8 = vreinterpret(
                vdup_scalar(0x00, dtypes.uint8), dtypes.float8_e4m3fn
            )
            for _fractal_idx in range(fresh_fractal_count):
                # strength-reduced addressing: running accumulators
                # instead of per-tile products (the naive form was 31%
                # of AIV time on conversion-bound shapes)
                p_addr = p_base
                dst = kdst
                for _g in range(first_group, group_end):
                    p = vload(b_packed_slot, p_addr)
                    # pure-permutation e2m1 -> e4m3, values exactly 1/64
                    # of the e2m1 magnitudes (see module docstring) —
                    # both nibbles share the byte register, the masks
                    # place [s]->bit7 and [e e m]->bits 4-2 for their
                    # side, so no nibble extraction is needed
                    conv_lo = vor(
                        vand(vshl(p, 4, mask=mask8), sign_mask, mask=mask8),
                        vand(vshl(p, 2, mask=mask8), exp_mant_mask, mask=mask8),
                        mask=mask8,
                    )
                    conv_hi = vor(
                        vand(p, sign_mask, mask=mask8),
                        vand(vshr(p, 2, mask=mask8), exp_mant_mask, mask=mask8),
                        mask=mask8,
                    )
                    # the DualParam split-M copy maps cache rows [0,
                    # span) to this AIV's L1 part, so the cache holds
                    # the slice at local group (g - first_group)
                    vstore_interleave(
                        converted_b_slot,
                        dst,
                        vreinterpret(conv_lo, dtypes.float8_e4m3fn),
                        vreinterpret(conv_hi, dtypes.float8_e4m3fn),
                    )
                    p_addr = p_addr + 256
                    dst = dst + n_group_stride
                p_base = p_base + s_row
                kdst = kdst + k_fractal_stride
            for _fractal_idx in range(fresh_fractal_count, t.k_fractals_per_window):
                # stale fractal: zero-write (see the docstring)
                dst = kdst
                for _g in range(first_group, group_end):
                    vstore_interleave(converted_b_slot, dst, zero_f8, zero_f8)
                    dst = dst + n_group_stride
                kdst = kdst + k_fractal_stride
            vmem_bar("vst_vld")

    @jit
    def _fetch_packed_window(
        self, b_packed_slot, fractals_to_fetch, tile_n_idx, k_window_idx
    ):
        # GM is a kb-major grid of packed (N16, K32) fractals. Merge
        # the contiguous N groups into each DMA row, then select this tile's
        # columns. A full window uses one strided DMA; bounded tails retain
        # their per-fractal transactions before conversion zeroes stale K.
        t = self.t
        if fractals_to_fetch == t.k_fractals_per_window:
            packed_grid = self.gm_b.reshape(
                (t.k_fractals_in_grid, t.n_groups_total * 256)
            )
            packed_window = b_packed_slot.reshape(
                (t.k_fractals_per_window, t.n_groups_per_tile * 256)
            )
            first_fractal = k_window_idx * t.k_fractals_per_window
            first_group_byte = tile_n_idx * t.n_groups_per_tile * 256
            source = packed_grid[
                first_fractal : first_fractal + fractals_to_fetch,
                first_group_byte : first_group_byte + t.n_groups_per_tile * 256,
            ]
            mem_copy(packed_window[:fractals_to_fetch, None], source)
        else:
            # bounded tail window — per-fractal segments up to the grid edge
            # (tile_cols == 1 rows are contiguous, so the segment form is
            # address-identical)
            for fractal_idx in range(fractals_to_fetch):
                gm_segment = tile_slice(
                    self.gm_b,
                    (t.n_groups_per_tile, 256),
                    (
                        (k_window_idx * t.k_fractals_per_window + fractal_idx)
                        * t.tile_cols
                        + tile_n_idx,
                        0,
                    ),
                )
                mem_copy(
                    tile_slice(
                        b_packed_slot, (t.n_groups_per_tile, 256), (fractal_idx, 0)
                    ),
                    gm_segment,
                )

    @jit
    def produce_k_window(self, tile_n_idx, k_window_idx, k_window_width):
        # The two AIVs split the window's n-groups; each converts its
        # groups and writes its row half of the CrossCore ring slot
        # (every sub-block must participate — an empty piece misaligns
        # the ring's write/read FIFO)
        t = self.t
        aiv_idx = get_subblock_id()

        b_packed_slot = self.packed.next_write()
        fractals_to_fetch = t.k_fractals_per_window
        fractals_left_in_grid = t.k_fractals_in_grid - k_window_idx * fractals_to_fetch
        if fractals_left_in_grid < fractals_to_fetch:
            fractals_to_fetch = fractals_left_in_grid
        self._fetch_packed_window(
            b_packed_slot, fractals_to_fetch, tile_n_idx, k_window_idx
        )

        # group split matching the DualParam row split (ceil/floor
        # groups of 16 rows per vector sub-core)
        split_group = (t.n_groups_per_tile + 1) // 2
        groups_this_aiv = (
            split_group if aiv_idx == 0 else t.n_groups_per_tile - split_group
        )
        first_group = 0 if aiv_idx == 0 else split_group
        b_converted_slot = self.b_converted.next_write()
        self.convert_kspan(
            first_group,
            groups_this_aiv,
            fractals_to_fetch,
            b_converted_slot,
            self.packed.next_read(),
        )
        # full-width piece: the slot is written across the whole k_l1
        # (the conversion covers every window fractal; beyond-grid
        # fractals carry finite stale-converted values)
        destination = self.b_ring.next_write()[: t.tile_n, :k_window_width].partition(
            dim=0, part_id=aiv_idx, alignment=16
        )
        converted = self.b_converted.next_read()
        mem_copy(destination, converted[: destination.shape[0], : destination.shape[1]])

    # ---- cube side: A/scale staging + per-L1-window MX mmad ----
    @jit
    def stage_k_window(self, k_window_idx, tile_m_idx, tile_n_idx):
        # per-L1-window A + scale staging into the k_l1-wide rings:
        # full-window claims on every window. A's tail-window claim
        # clamps to the real k (the stale L1 tail is zeroed in
        # __call__); the scale GM is column-padded to the k_l1 grid
        t = self.t
        # Host preparation pads M/N and scales to complete tiles. Only A's
        # final K window has a live tail; the scheduler proves every origin.
        m_start = tile_m_idx * t.tile_m
        n_start = tile_n_idx * t.tile_n
        scale_start = k_window_idx * (t.scale_bytes_per_window // 2)
        scale_end = scale_start + t.scale_bytes_per_window // 2
        mem_copy(
            self.scale_a_ring.next_write().permute(1, 0),
            self.gm_sa[
                m_start : m_start + t.tile_m, scale_start:scale_end, None
            ].flatten(-2, -1),
        )
        mem_copy(
            self.scale_b_ring.next_write(),
            self.gm_sb[n_start : n_start + t.tile_n, scale_start:scale_end, None]
            .flatten(-2, -1)
            .permute(1, 0),
        )
        k_start = k_window_idx * t.k_l1
        k_count = (
            t.k_l1 if const_expr(t.k % t.k_l1 == 0) else min(t.k_l1, t.k - k_start)
        )
        mem_copy(
            self.a_ring.next_write(),
            self.gm_a[m_start : m_start + t.tile_m, k_start : k_start + k_count],
        )

    @jit
    def produce_bias(self, tile_n_idx):
        # per-tile AIV-side bias staging + MX scale: the BT bias joins
        # the 2^-6-scaled accumulator, so it carries the same factor —
        # bias x 0.015625 (=2^-6, exact in fp32) on the VECTOR core.
        # Each AIV loads the tile's full vector, scales it, and writes
        # its real HALF via a tile_slice view: the ring wants ONE write
        # transaction per sub-block, and rank-1 is what the L1->BIAS
        # consume demands.
        t = self.t
        aiv_idx = get_subblock_id()
        tile_n = t.tile_n
        if const_expr(self.has_bias):
            mem_copy(
                self.bias_ub.next_write(),
                self.gm_bias[tile_n_idx * tile_n : (tile_n_idx + 1) * tile_n,],
            )
            bias_ub_slot = self.bias_ub.next_read()
            with vf():
                mask32 = create_mask(pattern="all", elem_bits=32)
                for off in range(0, tile_n, 64):
                    bias_vec = vload(bias_ub_slot, off)
                    bias_vec = vmuls(bias_vec, 0.015625, mask=mask32)
                    vstore(bias_ub_slot, off, bias_vec, mask32)
                vmem_bar("vst_vld")
            half_len = tile_n // 2
            mem_copy(
                self.bias_ring.next_write()[
                    aiv_idx * half_len : (aiv_idx + 1) * half_len,
                ],
                bias_ub_slot[aiv_idx * half_len : (aiv_idx + 1) * half_len,],
            )

    @jit
    def stage_tile(self):
        # per-tile bias handoff: the AIC consumes the CrossCore slot
        # into BIAS memory for the first mmad's BT operand (nobias
        # kernels touch none of the bias channels)
        if const_expr(self.has_bias):
            mem_copy(self.mmad_bias.next_write(), self.bias_ring.next_read())

    @jit
    def consume_block(self, k_window_idx, sub_block_idx, l0c_accumulator):
        # L0 sub-block s (k_l0 wide) of the current L1 window, reading
        # the window's consumed slot tensors. s rides a RUNTIME loop (a
        # constexpr unroll emits k_blocks_per_window sibling matmul writers and the
        # accumulator analysis rejects the ambiguous transaction).
        # Global sub index = k_window_idx*k_blocks_per_window + sub_block_idx drives init/final; bias
        # rides the init mmad.
        t = self.t
        global_sub_idx = k_window_idx * t.k_blocks_per_window + sub_block_idx
        a_sub_block = tile_slice(
            self.a_window_slot, (t.tile_m, t.k_l0), (0, sub_block_idx)
        )
        b_sub_block = tile_slice(
            self.b_window_slot, (t.tile_n, t.k_l0), (0, sub_block_idx)
        )
        scale_a_sub_block = tile_slice(
            self.scale_a_window_slot, (t.tile_m, t.k_l0 // 32), (0, sub_block_idx)
        )
        scale_b_sub_block = tile_slice(
            self.scale_b_window_slot, (t.k_l0 // 32, t.tile_n), (sub_block_idx, 0)
        )
        # The producer initializes complete padded K windows. Keep the fixed
        # compute domain: a dynamic sub-block copy result loses that static K
        # proof and adds scalar control to every MMAD.
        mem_copy(
            self.l0a.next_write(),
            a_sub_block,
            options=copy_options(mx_scale=scale_a_sub_block),
        )
        l0a_slot = self.l0a.next_read()
        mem_copy(
            self.l0b.next_write(),
            b_sub_block,
            options=copy_options(mx_scale=scale_b_sub_block),
        )
        l0b_slot = self.l0b.next_read()
        # final flag rides the last global sub: k_block_count = ceil(k/k_l0)
        last_sub_idx = t.k_block_count - 1
        # bias rides the init mmad (init is a runtime condition in the
        # sub-block loop, so the bias operand is always attached — the
        # hardware applies it on the init mmad only)
        dsl_matmul(
            l0c_accumulator,
            l0a_slot,
            l0b_slot,
            bias=self.bias_tensor if const_expr(self.has_bias) else None,
            init=(global_sub_idx == 0),
            unit_flag=3 if global_sub_idx == last_sub_idx else 2,
            mx_mode=True,
        )

    def _init_vector_ring_buffers(self):
        t = self.t
        tile_n = t.tile_n

        # vector-side staging: ONE packed channel holds the window's NZ
        # tiles (every (32k x 16n) tile is 256B, k-block-major then
        # n-group-major) — one channel instance per AIV's own UB
        packed_window_bytes = t.packed_bytes_per_window
        # 2-D tile rows: a 4-D writer trips the legacy rank check
        self.packed = RingBuffer(
            MemLoc.UB,
            dtypes.uint8,
            make_layout((packed_window_bytes // 256, 256)),
            depth=1,
        )

        # window ring: one channel epoch per runtime L1-window iteration
        # (ring depths from the tiling's L1 budget)
        self.b_converted = RingBuffer(
            MemLoc.UB,
            dtypes.float8_e4m3fn,
            make_layout((tile_n, t.k_l1)),
            depth=1,
            data_format="nz",
        )
        self.b_ring = RingBuffer(
            MemLoc.L1,
            dtypes.float8_e4m3fn,
            make_layout((tile_n, t.k_l1)),
            depth=t.ring_depth_b,
        )
        # bias: the AIVs stage + x 2^-6 scale the raw GM bias into the
        # CrossCore slot (official ScaleMxBias placement); the AIC hands
        # it to BIAS memory (stage_tile)
        self.bias_ub = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((tile_n,)),
            depth=1,
        )
        self.bias_ring = RingBuffer(
            MemLoc.L1,
            dtypes.float32,
            make_layout((tile_n,)),
            depth=1,
            data_format="nd",
        )

    def _init_cube_ring_buffers(self):
        t = self.t
        tile_n = t.tile_n

        # cube-side staging — all K-sized state is windowed (K unbounded):
        # A and both scale channels are k_l1-wide rings
        self.a_ring = RingBuffer(
            MemLoc.L1,
            dtypes.float8_e4m3fn,
            make_layout((t.tile_m, t.k_l1)),
            depth=t.ring_depth_a,
        )
        self.scale_a_ring = RingBuffer(
            MemLoc.L1,
            dtypes.float8_e8m0,
            make_layout((t.scale_bytes_per_window, t.tile_m)),
            depth=2,
        )
        self.scale_b_ring = RingBuffer(
            MemLoc.L1,
            dtypes.float8_e8m0,
            make_layout((t.scale_bytes_per_window, tile_n)),
            depth=2,
        )
        self.mmad_bias = RingBuffer(
            MemLoc.BIAS, dtypes.float32, make_layout((tile_n,)), depth=1
        )

        # L0 operand channels sized to the consume sub-block width
        # (the tail sub-block narrows via channel actuals)
        self.l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.float8_e4m3fn,
            make_layout((t.tile_m, t.k_l0)),
            depth=2,
        )
        self.l0b = RingBuffer(
            MemLoc.L0B, dtypes.float8_e4m3fn, make_layout((tile_n, t.k_l0)), depth=2
        )
        # L0C single-buffered for big tiles ((256,256) fp32 = 256KB =
        # L0C exactly; the per-tile fixpipe serializes with the next
        # tile's first mmad)
        self.l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((t.tile_m, tile_n)),
            depth=t.l0c_depth,
        )


# ============================================================================
# 3. Torch Interface
# ============================================================================


class _MixQuantLauncher:
    """Host wrapper: instantiates the @kernel class inside a @jit boundary."""

    def __init__(self, tiling: MixQuantTiling, has_bias: bool):
        self.t = tiling
        self.has_bias = has_bias

    # NOTE: the six GM tensors must stay as separate positional launch
    # arguments — the kernel binding accepts Tensor/scalar carriers only
    # and rejects a packed struct of tensors, so the signature cannot be
    # collapsed below six parameters.
    @host
    def run(
        self,
        gm_a: Tensor,
        gm_b: Tensor,
        gm_sa: Tensor,
        gm_sb: Tensor,
        gm_bias: Tensor,
        gm_c: Tensor,
    ):
        op = MixQuantKernel(self.t, self.has_bias)
        op[self.t.launch_block_count](gm_a, gm_b, gm_sa, gm_sb, gm_bias, gm_c)


@dataclasses.dataclass
class _MixQuantPrepared:
    """Host-side prepared operands (padded/aligned) for one launch."""

    tiling: MixQuantTiling
    a: torch.Tensor
    gm_b: torch.Tensor
    gm_sa: torch.Tensor
    gm_sb: torch.Tensor
    gm_bias: torch.Tensor
    c: torch.Tensor


def matmul_mix_quant(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """MX-quantized matmul: C = (A * sa) @ (B * sb)^T + bias.

    Args:
      a:       (M, K) torch.float8_e4m3fn activations.
      b:       (N, K // 2) torch.uint8 packed fp4 e2m1 weights (even k in
               the low nibble, adjacent K packed).
      scale_a: (M, align(K, 64) // 32) torch.uint8 E8M0 activation scales.
      scale_b: (N, align(K, 64) // 32) torch.uint8 E8M0 weight scales,
               N-major (ScaleBDN).
      bias:    optional (N,) fp16/fp32 bias.

    Returns:
      c: (M, N) torch.float16.
    """
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(
            f"expected 2-D inputs, got a.dim()={a.dim()}, b.dim()={b.dim()}"
        )
    m, k = a.shape
    n = b.shape[0]
    if b.shape[1] != k // 2:
        raise ValueError(f"b has {b.shape[1]} packed columns, expected K//2={k // 2}")
    if k % 32 != 0:
        raise ValueError(f"K must be a multiple of 32 (MX group size), got K={k}")
    prepared = _prepare(a, b, scale_a, scale_b, bias)
    launcher = _MixQuantLauncher(prepared.tiling, bias is not None)
    launcher.run(
        prepared.a,
        prepared.gm_b,
        prepared.gm_sa,
        prepared.gm_sb,
        prepared.gm_bias,
        prepared.c,
    )
    return prepared.c[:m, :n]


def _repack_nd_to_nz(
    b_nd: torch.Tensor, n_groups_total: int, k: int, k_fractals_in_grid: int
) -> torch.Tensor:
    """ND -> NZ repack (the Blaze reference receives NZ directly; the
    repack keeps the ND public API). Output = the reference global
    fractal grid: element (kk, nn) -> tile (kb, g) at flat row
    fractal_idx*n_groups_total + group_idx, packed 2/byte along kk (even kk low nibble)."""
    n = b_nd.shape[0]
    # codes (n, k) -> (k_fractal, n_group, 32, 16) layout; work in
    # aligned n groups (host pads, GM tiles read only their extent)
    n_rows_padded = n_groups_total * 16
    # host-side decode (tiny buffer; NPU bitwise kernels faulted
    # intermittently on these shapes)
    b_packed_np = b_nd.detach().cpu().numpy()
    b_codes_np = np.zeros((n_rows_padded, k), dtype=np.uint8)
    b_codes_np[:n, 0::2] = b_packed_np & 0x0F
    b_codes_np[:n, 1::2] = b_packed_np >> 4
    b_codes_host = b_codes_np
    b_codes_padded = np.zeros(
        (n_groups_total * 16, k_fractals_in_grid * 32), dtype=np.uint8
    )
    b_codes_padded[
        : b_codes_host.shape[0],
        : min(b_codes_host.shape[1], k_fractals_in_grid * 32),
    ] = b_codes_host[:, : k_fractals_in_grid * 32]
    fractal_grid = b_codes_padded.reshape(
        n_groups_total, 16, k_fractals_in_grid, 32
    ).transpose(2, 0, 3, 1)
    # (fractal, group, 32k, 16n) -> global fractal-major tile stream:
    # flat row = fractal_idx*n_groups_total + group_idx (each tile 16 n-rows x 16B)
    nz_tiles_np = np.zeros(
        (k_fractals_in_grid * n_groups_total, 16, 16), dtype=np.uint8
    )
    for fractal_idx in range(k_fractals_in_grid):
        for group_idx in range(n_groups_total):
            fractal_block = fractal_grid[fractal_idx, group_idx]  # (32k, 16n)
            # pack each K pair into one byte: even K goes to the low
            # nibble, odd K to the high nibble
            nz_tiles_np[fractal_idx * n_groups_total + group_idx] = (
                fractal_block[0::2] | (fractal_block[1::2] << 4)
            ).T
    return torch.from_numpy(nz_tiles_np.reshape(-1, 256)).to(b_nd.device).contiguous()


def _prepare(a, b, scale_a, scale_b, bias) -> _MixQuantPrepared:
    """Host-side preparation: repack B to the NZ grid, pad A/scale/bias
    descriptors to the tile and k_l1 window grids. Inputs are assumed
    validated by the caller."""
    m, k = a.shape
    n = b.shape[0]
    # K is unbounded (windowed on-chip state); tail K is handled by
    # padding the A/scale columns to the k_l1 grid below
    scale_column_count = _ceil_align(k, 64) // 32
    if tuple(scale_a.shape) != (m, scale_column_count):
        raise ValueError(
            f"scale_a shape {tuple(scale_a.shape)} != ({m}, {scale_column_count})"
        )
    if tuple(scale_b.shape) != (n, scale_column_count):
        raise ValueError(
            f"scale_b shape {tuple(scale_b.shape)} != ({n}, {scale_column_count})"
        )
    if bias is not None and bias.shape[0] != n:
        raise ValueError(f"bias length {bias.shape[0]} != N={n}")

    # ONE tiling path for every shape (no tail-specialized forks)
    tiling = MixQuantTiling(m, n, k)
    # B's GM grid k-fractal count (the repack target; the AIV fetch
    # bound reads the same field): the align64 grid on every shape
    b_grid_fractal_count = tiling.k_fractals_in_grid

    gm_b = _repack_nd_to_nz(b, tiling.n_groups_total, k, b_grid_fractal_count)

    # A GM padding: rows pad to the m tile grid (a statically provable
    # row clamp at tile 0 trips the mx binding's ScaleA outer-extent
    # check; padded rows are fp8 zero feeding only sliced-away output
    # rows). Columns NEVER pad — the kernel zero-fills the tail L1 slot
    # region itself (cube_fill_l1_zero in __call__).
    k_window_aligned = _ceil_align(k, tiling.k_l1)
    a_grid_rows = tiling.tile_rows * tiling.tile_m
    if a_grid_rows != m:
        a_padded = torch.zeros(a_grid_rows, k, dtype=torch.uint8, device=a.device)
        a_padded[:m, :k] = a.view(torch.uint8)
        a = a_padded.view(torch.float8_e4m3fn)

    # the E8M0 scale descriptors pad to the tile grid AND the k_l1
    # window grid: full-width window claims read the whole slot, and a
    # stale beyond-k scale byte could be an E8M0 NaN pattern whose
    # NaN x A-zero would poison the annihilation (this copy exists for
    # the row padding anyway — the column extension is free)
    def _align_scale(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        padded = torch.zeros(rows, cols, dtype=torch.uint8, device=scale.device)
        padded[: scale.shape[0], : scale.shape[1]] = scale
        return (
            padded.reshape(rows, cols // 2, 2)
            .view(torch.int8)
            .view(torch.float8_e8m0fnu)
        )

    scale_bytes_padded = k_window_aligned // 32
    gm_sa = _align_scale(scale_a, tiling.scale_a_grid_rows, scale_bytes_padded)
    gm_sb = _align_scale(scale_b, tiling.scale_b_grid_rows, scale_bytes_padded)

    # bias descriptor aligned to the tile grid (same rationale as the
    # scales); RAW values — the 2^-6 scale rides on the AIV
    # (produce_bias), so GM is consumed as-is
    bias_f32 = (
        bias.to(torch.float32).contiguous()
        if bias is not None
        else torch.zeros(n, dtype=torch.float32, device=a.device)
    )
    gm_bias = torch.zeros(
        tiling.scale_b_grid_rows, dtype=torch.float32, device=a.device
    )
    gm_bias[:n] = bias_f32

    # output padded to the FULL tile grid: the fixpipe lowering passes
    # the L0C channel extents as nSize and mSize instead of the GM
    # clamped width, so any clamped fixpipe write corrupts the output;
    # at the grid size no claim clamps, and the slice on return is free
    c = torch.zeros(
        tiling.tile_rows * tiling.tile_m,
        tiling.tile_cols * tiling.tile_n,
        dtype=torch.float16,
        device=a.device,
    )

    return _MixQuantPrepared(
        tiling=tiling,
        a=a,
        gm_b=gm_b,
        gm_sa=gm_sa,
        gm_sb=gm_sb,
        gm_bias=gm_bias,
        c=c,
    )
