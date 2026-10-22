# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Fused ``indexer_prologue_qw`` NPU kernel (CANNBotDSL, Ascend950 / dav-3510).

One MIX_AIC_1_2 kernel, two cube paths and one vector epilogue::

    C_w  BF16  x @ ww^T  --fixpipe(deq_scale=softmax_scale)-->  GM w
    C_q  MXFP8 qr @ wqb^T  =fixpipe(dual_dst)=>  cv_ub  --VF-->  GM q, descale_q

Tiling for the frozen case ``dim=5120, q_lora=1280, N=32, D=128, Dr=64``: the
``T`` axis is split over AICs in tiles of 128 rows, ``qr`` plus its MX scale
stay resident in L1 for the whole 32-head loop, ``wqb`` streams in 80 KB K
windows, and the W path's ``x`` traffic is chopped into 32 windows retired one
per head so it never stalls the cube.  The cube roofline at T=4096 is about 52.7 us vs HBM 38.7 us.

Everything the vector unit does is raw-VF register code.  ``vcast`` has no fp4
lowering in this wheel (verified), so E2M1 nibbles are produced by integer bit
manipulation and a small gather table, then packed two-per-byte.
"""

# Read the public entry indexer_prologue_qw, then Plan, the main kernel, and
# CubeQ / CubeW / VectorQ.  Data moves GM -> L1 -> L0A/L0B -> L0C -> UB.
# Q is RoPE-quantised in UB and stored back to GM.  W either stores its result
# directly or writes partials into workspace for a later reduction.

from __future__ import annotations

__all__ = [
    "IndexerPrologueQw",
    "build_e2m1_lut",
    "plan",
    "resolve_cube_cores",
]

import os
from dataclasses import dataclass

import torch

try:
    import torch_npu
except ImportError:  # CPU-only tests import this module without an NPU stack.
    torch_npu = None

from cannbotdsl import (
    Buffer,
    Channel,
    ChannelKind,
    Dim,
    MemLoc,
    Tensor,
    TensorSpec,
    compile as dsl_compile,
    const_expr,
    dtypes,
    host,
    get_block_idx,
    get_block_num,
    get_subblock_dim,
    get_subblock_id,
    jit,
    kernel,
    cv_valid_extent,
    make_copy_engine as _dsl_make_copy_engine,
    matmul,
    mem_copy,
    range_constexpr,
    vf,
)

from cannbotdsl.aot import export
from cannbotdsl.ops.reg import (
    PackMode,
    create_mask,
    full_mask,
    varange,
    vabs,
    vadd,
    vadds,
    vbitwise_and,
    vbitwise_or,
    vbitwise_xor,
    vdups,
    veq,
    vgather,
    vgather_reg,
    vges,
    vgts,
    vload,
    vlts,
    vmadd,
    vmaxs,
    vmul,
    vne,
    vcast,
    vmem_bar,
    vpair_reduce_sum,
    vreduce_max,
    vreinterpret,
    vselect,
    vshl,
    vshr,
    vstore,
    vstore_first,
    vstore_pack,
    vsub,
)
from cannbotdsl.ops.sync import (
    PIPE,
    cube_sync_all,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    global_sync_all,
    vec_sync_all,
    vec_sync_block_wait,
)
from cannbotdsl.tensor import (
    make_tiler,
    reinterpret,
    tile_slice,
)

Float8E4M3FN = dtypes.float8_e4m3fn
Float8E8M0 = dtypes.float8_e8m0


def tile_view(tensor, tiler, coord):
    return tile_slice(tensor, tiler, coord)


def make_bounded_tiler(shape, capacity=None, alignment=None):
    del capacity
    return make_tiler(shape, alignment=alignment)


def local_slice(tensor, tiler, stride=None, offset=0):
    return reinterpret(tensor, shape=tiler, stride=stride, offset=offset)


def make_copy_engine(*args, **kwargs):
    kwargs.pop("dtype", None)
    kwargs.pop("pad_value", None)
    return _dsl_make_copy_engine(*args, **kwargs)


try:
    from device_properties import get_device_properties
except ImportError:  # packaged net/ops wheel has no samples/ on sys.path
    def get_device_properties():
        return torch.npu.get_device_properties(0)

I32 = dtypes.int32
# ``get_block_idx`` and ``get_block_num`` are int64, and a dynamic range wants
# start, stop and step in one dtype, so the runtime work count is int64 too.
I64 = dtypes.int64
U32 = dtypes.uint32
F32 = dtypes.float32
U8 = dtypes.uint8

VL = 64  # fp32 lanes in one 256-byte raw vector register
SUBBLOCKS = 2  # AIVs per block on MIX_AIC_1_2; get_subblock_dim() at runtime
MX_GROUP = 32  # elements per E8M0 scale
MX_K_ALIGN = 64  # K elements per public paired-scale group
MX_PAIR = 2
FP32_BYTES = 4
INT32_BYTES = 4
NIBBLES_PER_BYTE = 2  # two E2M1 values packed into one uint8
COS_SIN_COPIES = 2
INPUT_RANK = 2
# Frozen geometry this kernel is built for.  The host entry rejects anything else.
FROZEN_DIM = 5120
FROZEN_Q_LORA = 1280
FROZEN_N_HEADS = 32
FROZEN_D = 128
FROZEN_DR = 64
E8M0_BIAS = 127
FP4_E2M1_MAX = 6.0
FP32_EXP_SHIFT = 23
FP32_MANTISSA_MASK = 0x7FFFFF
FP32_HALF_MANTISSA = 0x400000  # significand exactly 1.5
FP32_MAG_MASK = 0x7FFFFFFF
# |y| <= 6 always, so (bits(|y|) >> LUT_SHIFT) <= 1036.
LUT_SHIFT = 20
LUT_LEN = 1088


# Channel depths, overridable while tuning so a sweep is one env var per run.
# Read at trace time, so every one of these is a compile-time constant.
def _depth(name: str, default: int) -> int:
    return int(os.environ.get(f"IPQW_{name}_DEPTH", default))


D_CV = _depth("CV", 2)
D_QUB = _depth("QUB", 2)
# How many K windows one head's ``wqb`` reduction is filled in.  A core with
# several heads has head h+1 to overlap with, so a coarse split is enough; a
# single-head core has no next head and instead wants its own windows small
# enough to keep MTE2 ahead of the mmads.  5 windows (256 elements each, the
# value the old window-size knob had converged on) measured best there.
D_WQB = _depth("WQB", 2)
D_WQB_SINGLE_HEAD = _depth("WQB_SINGLE_HEAD", 5)
# One buf id per channel slot, and the CUBE side has 32.  Everything except
# wqb_l1 and its scales is fixed-depth, hence the constant.
CUBE_BUF_IDS = 32
FIXED_CUBE_SLOTS = 16
D_L0C_Q = _depth("L0C_Q", 2)
D_L0AB_Q = _depth("L0AB_Q", 2)
D_L0B_Q = _depth("L0B_Q", D_L0AB_Q)
D_X = _depth("X", 2)
# W-path L0 is single-buffered on purpose. Giving up the second L0A buffer
# buys a wider base_k_w, and fewer MTE1 inserts beat double buffering here.
D_L0AB_W = _depth("L0AB_W", 1)

# W-path schedule inside the Q head loop.  MTE2 is one in-order pipe, so where
# the W fills sit relative to the Q fills decides whether Q's mmads wait behind
# them.  "wfirst" is the original: W fills and W mmads both ahead of Q.
W_SCHED = os.environ.get("IPQW_W_SCHED", "wfirst")

# Diagnostic-only ablations; some produce wrong numbers by construction and
# exist purely to attribute wall time to a pipe.  "" is the real kernel.
PROBE = os.environ.get("IPQW_PROBE", "")

# Where the W reduction waits. The dependency is only that every partial is
# already in GM. A barrier instead waits until the slowest core arrives, so
# its cost is the remaining work skew. The tail placement is the default:
# an early barrier sits in front of the Q head loop and nothing overlaps it.
# The four-phase barrier is slower at the tail because its first phase waits
# for the AIV, which is still retiring the last head. store_to_vec_barrier
# keeps only the two phases this dependency needs.
W_REDUCE_AT = os.environ.get("IPQW_W_REDUCE_AT", "tail")
# "lean" for the two-phase barrier below, "full" for global_sync_all's four.
W_SYNC = os.environ.get("IPQW_W_SYNC", "lean")

# L2 retention hints.  At one tile/core, caching Q-path weights in L2 is a
# wash.  At two-plus tiles/core the same hint *hurts*: T=8192 went 144 us
# (ctl=1) -> 139 us (ctl=0).  Plan.l2_q therefore defaults to 0 when the
# core owns more than one T-tile; IPQW_L2_Q still overrides.  ``x`` is
# streamed once and does not want L2.
L2_Q = int(os.environ["IPQW_L2_Q"]) if "IPQW_L2_Q" in os.environ else None
L2_W = int(os.environ.get("IPQW_L2_W", 1))
# Issue the next tile's ``qr`` load after the last Q mmad of this tile so it
# overlaps the last head's vector epilogue.  Default on when a core has more
# than one T-tile.  IPQW_PREFETCH=0/1 overrides.
PREFETCH = int(os.environ["IPQW_PREFETCH"]) if "IPQW_PREFETCH" in os.environ else None
UNIT_FLAG = int(os.environ.get("IPQW_UNIT_FLAG", 0))
# Split the head axis over leftover AICs when T-tiles alone cannot fill the
# chip.  Once there are at least as many T-tiles as cube cores the original
# T-only schedule is kept, so the 75% MFU path is unchanged.
# IPQW_HEAD_SPLIT=0 restores the old 1-core-per-T-tile launch for A/B.
HEAD_SPLIT = int(os.environ.get("IPQW_HEAD_SPLIT", 1))
# What one extra launched block costs, in units of one head GEMM's per-core
# weight traffic.  Fitted from the pinned block-count sweep in
# ``_plan_head_split``; see there for the measurements and why the head split
# has an optimum rather than wanting every core.
BLOCK_COST_HEADS = float(os.environ.get("IPQW_BLOCK_COST_HEADS", 0.146))
# Split the W path's K reduction across the cores that share a T-tile, each
# writing its partial to rows it alone owns, and sum them on the vector side.
# Letting one leader per tile own the whole reduction leaves it reading the
# entire x tile alone: at T=72 that is 1388 KB on the leader against 268 KB on
# each of the other cores, and the wall clock is the leader.
W_K_SPLIT = int(os.environ.get("IPQW_W_K_SPLIT", 1))
# Launch the whole chip even when there are fewer work items than cores, so that
# the grid stops being a specialisation dimension.  A block with no work items
# falls straight out of the work loop; see the ``used_cores`` comment in
# ``_plan_head_split`` for what that is worth measured.
GRID_FULL = int(os.environ.get("IPQW_GRID_FULL", 1))
# Where the two templates meet, counted in T-tiles rather than in cores.
#
# This being a core count is what used to put the core count in the binary. The
# boundary was ``n_tiles >= cube_cores``, and since the head split and the band
# geometry are chosen against the same number, a 28-, 32- and 36-core part each
# needed their own build of both templates -- and within a part the split moved
# with T, which is where the nine binaries came from.
#
# The tile count a T-only schedule needs before it fills a chip on its own is
# roughly the core count, so this is that value for the parts in play, frozen. A
# part with more cores switches to split-T slightly early and one with fewer runs
# split-K slightly long; that costs a little occupancy at the boundary and no
# correctness, because neither template assumes anything about the grid -- the
# work loop strides by ``get_block_num()``, which is a runtime value.
SPLIT_T_TILES = int(os.environ.get("IPQW_SPLIT_T_TILES", 25))
# The split-K head split, pinned.  ``_plan_head_split`` used to score this per
# (T, cores), and it is genuinely T-dependent -- 16 wins at T=1 and 2 wins at
# T=2048 -- but every distinct value is another binary, because the block count
# is a loop trip count, a ``k_block`` width and a band UB width.  16 is the
# decode optimum, and decode is the phase where this operator's latency is on
# the critical path; see the measured table in ``_plan_head_split`` for what the
# large end of the split-K range gives up for it.
HEAD_BLOCKS_K = int(os.environ.get("IPQW_HEAD_BLOCKS_K", 16))
# The band height of the workspace W reduction, pinned for the same reason: it is
# a UB height and a register unroll.  The narrowest legal band spreads best over
# the AIVs and 2 is what every decode shape already chose, so pinning it here
# costs the large end of split-K some DMA round trips and decode nothing.
ROWS_RED_K = int(os.environ.get("IPQW_ROWS_RED_K", 2))
# Upper bound on the launch grid, used only to bound the ``workspace`` extent.
#
# The grid cannot be a plain host integer without the core count becoming part of
# the binary: ``compute_binary_key`` hashes the verified IR, and the IR carries
# the block dim, so a 28-, 32- and 36-core export of identical source produce
# three different keys.  It also cannot exceed the physical core count, because
# the W reduction rendezvous waits on every block and a second scheduling wave
# deadlocks it.  So the grid is carried the way ``mixed_quant_sparse_flash_mla``
# carries it: the host sizes a tensor to it and the kernel reads the extent back,
# which keeps it symbolic.  ``workspace`` gets ``used_cores`` marker rows past its
# partials for that, and the Dim that bounds them has to admit any core count on
# any part or the bound itself would put the core count back in the contract.
MAX_CORES = int(os.environ.get("IPQW_MAX_CORES", 128))
# ``IPQW_T_DYN=0`` goes back to a binary per T, so the price of T being a runtime
# value can be measured rather than argued about.  See ``_plan_for``.
T_DYN = int(os.environ.get("IPQW_T_DYN", 1))
# Rows of T one core owns per work item, i.e. the M of every GEMM here.  It
# cannot go above 128: the L1 residents (``qr``, the ``x`` and ``ww`` buffers)
# reach 439 KB of the 512 KB there, and 192 would need 565 KB.  Below 128 it
# only re-reads the 5.24 MB ``wqb`` more times per row of T, which loses even
# where the round quantisation is at its worst -- 28 cores, where T=4096 is 32
# tiles over 28 cores, measures 158.3 us at 128 against 172.7 at 64 and 241.1
# at 32.  ``IPQW_BASE_M`` is that A/B.
BASE_M = int(os.environ.get("IPQW_BASE_M", 128))
# T is the only free axis, and it is ``batch * seq``, so nothing about it is
# aligned: decode contributes 1, 4, 6, 8, 12, 24 (batch times 1 or MTP's 6) and
# prefill contributes batch times 1024..8192.  Any T in this range is legal; the
# row fractal is satisfied by the tile height, not by T (see ``base_m``).
T_MIN = 1
T_MAX = 256 * 1024
# FRACTAL_NZ grid of the two weight inputs: 16 rows per fractal, and a C0 that
# is 32 bytes wide, i.e. 32 one-byte or 16 two-byte elements.
NZ_M_FRAC = 16
NZ_C0_1B = 32
NZ_C0_2B = 16

L1_BYTES = 512 * 1024
L0AB_BYTES = 64 * 1024
L0C_BYTES = 256 * 1024
UB_BYTES = 248 * 1024


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def resolve_cube_cores(cube_cores: int | None = None) -> int:
    """Cube-core count for the tile plan: a device property, not a constant.

    Every core-count decision in :class:`Plan` (head split, used cores, the
    L2/prefetch thresholds) is keyed on this.  It is resolved on the host and
    then carried as a compile-time constant so the compile-only path never
    enters the NPU runtime.  ``IPQW_CUBE_CORES`` overrides for A/B runs.
    """
    if cube_cores is None:
        forced = os.environ.get("IPQW_CUBE_CORES")
        cube_cores = (
            int(forced) if forced else get_device_properties().cube_core_num
        )
    cube_cores = int(cube_cores)
    if cube_cores <= 0:
        raise ValueError(f"cube_cores must be positive, got {cube_cores}")
    return cube_cores


def scale_l1_len(k: int) -> int:
    """Contiguous E8M0 length in L1 for a K extent (paired public layout)."""
    return ceil_div(k, MX_K_ALIGN) * MX_PAIR


# ---------------------------------------------------------------------------
# E2M1 rounding table
# ---------------------------------------------------------------------------

# Build the FP4 rounding table. The index is bits of the scaled absolute value;
# the stored code is a 3-bit magnitude. The sign is merged later, and two
# 4-bit codes are packed into one uint8.
def build_e2m1_lut() -> torch.Tensor:
    """Gather table mapping ``bits(|y|) >> 20`` to a 3-bit E2M1 magnitude code.

    ``y`` is the already-descaled value, so ``|y| <= 6``.  Index bits are the
    biased fp32 exponent times eight plus the top three mantissa bits; every
    E2M1 midpoint (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0) is exactly on such a
    bucket boundary.  The normal bucket lookup therefore selects the upper
    neighbour at a midpoint.  ``_pack_e2m1`` fixes the four exact midpoints
    whose upper neighbour is odd, completing round-to-nearest-even (RNE).
    """
    codes = torch.zeros(LUT_LEN, dtype=torch.int32)
    midpoints = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    for exponent in range(120, 133):
        for mantissa in range(8):
            index = exponent * 8 + mantissa
            if index >= LUT_LEN:
                continue
            value = 2.0 ** (exponent - E8M0_BIAS) * (1.0 + mantissa / 8.0)
            codes[index] = sum(1 for m in midpoints if value >= m)
    codes[133 * 8:] = 7
    return codes


# ---------------------------------------------------------------------------
# Host tiling
# ---------------------------------------------------------------------------

# Host-side plan: template, on-chip buffer sizes, and the work each core owns.
# T = batch * seq. Default base_m is 128. Q is split by head; W is split on K.
# This only plans. The matmul runs in indexer_prologue_qw_kernel.
class Plan:
    """Compile-time tile layout and template parameters.

    The dynamic entry sets t_dyn=True so buffer heights stay fixed and the
    live row count comes from t_live. Both split-K and split-T keep T dynamic;
    trip counts that change with T are not part of the template key.
    A Plan constructed directly still defaults t_dyn to False. Use plan() for
    the plan the public entry will actually run.
    """

    def __init__(
        self,
        t: int,
        dim: int,
        q_lora: int,
        n_heads: int,
        d: int,
        dr: int,
        cube_cores: int | None = None,
        t_dyn: bool = False,
    ):
        if t < T_MIN or t > T_MAX:
            raise ValueError(f"T must be in [{T_MIN}, {T_MAX}], got {t}")
        self.t = t
        self.t_dyn = bool(t_dyn)
        self.dim = dim
        self.q_lora = q_lora
        self.n_heads = n_heads
        self.d = d
        self.dr = dr
        self.cube_cores = resolve_cube_cores(cube_cores)

        # Sizing the tile to T is what used to make T compile-time below one
        # tile per core: ``base_m`` is the height of every L1/L0/UB buffer, so a
        # T-dependent value here fragments the small-T range into one binary per
        # T.  A dynamic plan pins it and lets the short tile be short -- which
        # costs nothing measurable, because the vector epilogue is already cut
        # to the live rows (``vec.epilogue(cv_ub, rows)``) and the wasted mmad
        # rows land in a cube that is only ~22% busy down here.
        self.base_m = (
            BASE_M
            if (self.t_dyn or t >= BASE_M)
            else max(16, ceil_div(t, 16) * 16)
        )
        self.rows_vec = self.base_m // 2
        self.n_tiles = ceil_div(t, self.base_m)
        self._plan_head_split()
        # See the L2_Q / PREFETCH comments at module level.  Both want to know
        # whether a core gets more than one T-tile, which is a runtime question
        # now, so it is answered by the template instead: heads are split exactly
        # when the T-tiles cannot fill the chip on their own, so a split-K binary
        # is the one whose cores get a single tile.  This agrees with the old
        # per-T choice everywhere except tiles 25..31, which used to be its own
        # set of binaries and which the split-T setting measured faster on.
        more_than_one_tile = self.n_head_blocks == 1
        self.l2_q = L2_Q if L2_Q is not None else (0 if more_than_one_tile else 1)
        self.prefetch = (
            PREFETCH if PREFETCH is not None else (1 if more_than_one_tile else 0)
        )
        self.bytes_used = {}
        self._init_w(dim, n_heads)
        self._init_q(q_lora, d)
        self._init_vec(d, dr)
        self._check()
        if self.t_dyn:
            self._check_t_dynamic()

    # Compile-cache key. Plans with the same geometry and on-chip layout share
    # a binary. The live row count of a dynamic T is not in this key; the
    # kernel derives the work count from t_live.
    def shape_key(self) -> tuple:
        """What this plan pins at compile time. Neither T nor the core count is.

        ``cube_cores`` used to be in here, and it had to be while the head split,
        the band height and the launch grid were all chosen against it.  All three
        are now pinned or runtime, so the same binary serves a 28-, 32- or 36-core
        part; the core count is a host-side quantity only.
        """
        return (self.dim, self.q_lora, self.n_heads, self.d, self.dr) + tuple(
            getattr(self, f) for f in self.SHAPE_KEY
        )

    def assert_budgets(self) -> None:
        l1 = (
            self.base_m * self.qr_k_l1  # qr resident
            + self.base_m * self.scale_k_len_q
            + self.n_buf_wqb * self.base_n_q * self.k_l1_q  # wqb, fully resident
            + self.n_buf_wqb * self.scale_k_l1_len_q * self.base_n_q
            + D_X * self.base_m * self.k_l1_w * 2  # x
            + D_X * self.base_n_w * self.k_l1_w * 2  # ww
        )
        l0a = (
            D_L0AB_Q * self.base_m * self.base_k_q
            + D_L0AB_W * self.base_m * self.base_k_w * 2
        )
        l0b = (
            D_L0B_Q * self.base_n_q * self.base_k_q
            + D_L0AB_W * self.base_n_w * self.base_k_w * 2
        )
        l0c = 4 * (
            D_L0C_Q * self.base_m * self.base_n_q + 2 * self.base_m * self.base_n_w
        )
        ub = (
            D_CV * self.rows_vec * self.d * FP32_BYTES  # cv_ub
            + D_QUB * self.rows_vec * (self.d // NIBBLES_PER_BYTE)  # packed q_ub
            + D_QUB * self.rows_vec * self.groups  # descale, one byte per group
            + self.amax_len * (FP32_BYTES + INT32_BYTES)  # amax fp32 + off int32
            + LUT_LEN * INT32_BYTES
            + COS_SIN_COPIES * self.rows_vec * max(self.dr, 1) * FP32_BYTES
            # one band's partials plus the summed band
            + (self.n_head_blocks + 1) * self.rows_red * self.n_heads * FP32_BYTES
        )
        for name, used, cap in (
            ("L1", l1, L1_BYTES),
            ("L0A", l0a, L0AB_BYTES),
            ("L0B", l0b, L0AB_BYTES),
            ("L0C", l0c, L0C_BYTES),
            ("UB", ub, UB_BYTES),
        ):
            if used > cap:
                raise ValueError(
                    f"{name} budget exceeded: {used} > {cap} bytes; retune Plan"
                )
        self.bytes_used = {"L1": l1, "L0A": l0a, "L0B": l0b, "L0C": l0c, "UB": ub}



    # Template choice: when ceil(T/128) < 25, split 32 heads into 16 groups of
    # 2. That also splits the W reduction of dim=5120 into 16 K slices, which
    # is why this template is called split-K. Otherwise use split-T.
    # The default boundary is T=3072/3073. It is not chosen from a
    # decode/prefill label or from the core count. The long comment below
    # keeps earlier measurements; the condition at the end of the function
    # is what actually runs.
    def _plan_head_split(self) -> None:
        """Split the head axis so the T-tiles alone do not have to fill the chip.

        Decode T is 1..192, which is one or two row tiles -- a T-only schedule
        would launch one or two cores and leave the other 30 idle, reporting the
        2-5% chip MFU the official small-T cases used to show.  The missing work
        is the other 31 heads, which are independent GEMMs, so the free axis for
        occupancy is N, not T.  Once the heads occupy the chip the shape is
        weight-load bound (``wqb`` is 5.24 MB whatever T is) and the target
        flips from MFU to MBU.

        Above ``cube_cores`` T-tiles the T axis fills the chip on its own and the
        head split is off: a core then owns a whole T-tile, keeps its ``qr`` in
        L1 for all 32 heads, and owns the W reduction outright.  That is the
        split-T template.  Measured at 28 cores and T=4096 the head split makes
        no difference at all there (157.6 .. 159.7 us over every block count from
        1 to 32, including the 4-cores-do-two-tiles imbalance), because the shape
        is MTE2-throughput bound and no split changes the bytes; so the cheapest
        schedule wins by default.

        Below that, more blocks is *not* simply better, and this is the one place
        where chasing occupancy actively costs time.  Splitting heads ``b`` ways
        launches ``b`` blocks, and every launched block pays a share of the
        the grid barrier and the band sum the workspace W reduction needs, while
        it only divides the per-core weight traffic.  Measured at 32 cores with
        the block count pinned (on the atomic-add reduction this replaced; the
        shape of the curve is what the score is fitted to, and the workspace
        reduction moved every column down by about the same 2 us):

            T     4 blk   8 blk   16 blk   32 blk
            1     16.20   11.10    9.58    12.02   us
            16    16.77   11.70   11.22    13.53
            72    20.57   14.80   13.83    15.88
            192   24.50   18.43   18.80    20.39

        so 16 launched blocks wins everywhere and 32 is 13-25% worse than that.
        Fitting those four columns to ``A + B*blocks + C*bytes_per_core`` gives
        B = 0.27 us per launched block against C = 1.8 us per head of weight
        traffic.  ``BLOCK_COST_HEADS`` is that B/C ratio: one launched block
        priced in head GEMMs.

        The other term is what a round costs beyond its head GEMMs: a core
        stages the T-tile's ``qr`` into L1 once per work item, so a round is
        ``base_m/d`` head GEMMs of weight traffic on top of ``heads_per_block`` of
        them (``qr`` is ``base_m`` rows of the same K extent that one head's
        ``wqb`` has ``d`` rows of).  Without that term the score prefers splits
        that shave a fraction off the round count and pay for it many times over
        in re-staged ``qr`` -- at 36 cores and T=2048 it would pick 32 blocks
        over 2, which measures 60.9 us against 54.3 us.

        With both terms the score reproduces the measured best block count at
        every (cores, T) pair that was swept: 28, 32 and 36 cores across
        T = 1, 16, 72, 192, 1024, 2048 and 4096.

        Lowering the per-block cost is the way to make finer splits pay.  The
        workspace reduction already took out the zero-fill and cut the barrier
        from four handshakes to two; ``BLOCK_COST_HEADS`` is still the fitted
        constant from before that, so it is now, if anything, pessimistic about
        fine splits.  Re-fit it if the reduction changes again.

        ``IPQW_HEAD_BLOCKS`` pins the count for A/B runs.
        """
        forced = int(os.environ.get("IPQW_HEAD_BLOCKS", 0)) or HEAD_BLOCKS_K
        if not HEAD_SPLIT or forced == 1 or self.n_tiles >= SPLIT_T_TILES:
            self.n_head_blocks = 1
            self.heads_per_block = self.n_heads
            self.used_cores = self._grid(self.n_tiles)
            return

        if self.n_heads % forced:
            raise ValueError(
                f"head block count {forced} does not divide n_heads="
                f"{self.n_heads}"
            )
        self.n_head_blocks = forced
        self.heads_per_block = self.n_heads // self.n_head_blocks
        self.used_cores = self._grid(self.n_tiles * self.n_head_blocks)

    def _grid(self, n_work: int) -> int:
        """How many blocks to launch for ``n_work`` work items.

        The grid is a launch parameter, so unlike the work count itself it cannot
        be a runtime value -- which makes it a specialisation dimension, and a
        bad one: taking it as ``min(cores, n_work)`` gives a separate binary for
        every work count below the core count, seven of them in T=3073..3968
        alone, all doing the same thing.  Launching the whole chip instead costs
        only the prologue of the blocks that find no work item, since the work
        loop is strided by the grid and they fall straight out of it.
        """
        if GRID_FULL:
            return self.cube_cores
        return min(self.cube_cores, n_work)

    # The fields the kernel has to know at compile time, and what each one is.
    # Nothing else about the plan reaches the kernel as a constant: the counts
    # that move with T -- ``n_tiles``, ``n_work``, ``n_bands``, ``n_bands_workspace``,
    # ``workspace_rows`` -- are all derived from the live row count at runtime.  So this
    # tuple *is* the specialisation key: two T that agree on it share a binary.
    SHAPE_KEY = (
        "base_m",         # height of every L1/L0/UB tile
        "rows_vec",       # vector-side tile height
        "n_head_blocks",  # head loop trip count, k_block, band UB width
        "heads_per_block",
        "rows_red",       # band UB height
        "red_regs",       # register unroll in the band sum
        # ``used_cores`` is deliberately absent: the grid is a symbolic
        # expression over the padded row count now, not a constant, so it is
        # neither a specialisation dimension nor a route for the core count.
        "w_workspace",           # whether the workspace path is compiled in at all
        "w_separate",
        "w_leader_only",
        "w_interleaved",
        "prefetch",
        "l2_q",
        "k_l1_w",
        "base_k_w",
        "k_l1_tiles_w",
        "qr_k_l1",
        "n_buf_wqb",
        "reduce_at",
    )

    # Dynamic T requires a fixed buffer height, and the cross-core W reduction
    # must run after every work item has finished. A barrier inside a loop
    # whose trip count differs per core can leave waits unpaired.
    def _check_t_dynamic(self) -> None:
        """Reject a dynamic-T plan that would still bake a T-derived count in.

        Two things have to hold, and neither is about which template this is.

        ``base_m`` must be the pinned tile height, because it is a buffer shape
        rather than a count -- a plan that sized it to T could not serve any
        other T.  ``__init__`` pins it under ``t_dyn``, so this is a guard on
        that, not a restriction on the caller.

        ``reduce_at`` must be "tail".  The "early" rendezvous sits inside the
        work loop and is only sound while every launched core runs that loop
        exactly once; whether that holds depends on the tile count, which is
        now a runtime value, so it cannot be decided here.
        """
        problems = []
        if self.base_m != BASE_M:
            problems.append(f"base_m={self.base_m} is sized to T (want {BASE_M})")
        if self.w_workspace and self.reduce_at != "tail":
            problems.append(
                f"reduce_at={self.reduce_at!r} needs one work item per core, "
                "which is a runtime property under t_dyn"
            )
        if problems:
            raise ValueError(
                f"T cannot be dynamic for this plan (T={self.t}): "
                + "; ".join(problems)
            )

    # Plan the W partial reduction. A band is a few rows of w; one AIV sums
    # every K slice of that band. split-K stores partials. split-T has none,
    # but still keeps marker rows that carry the launch grid.
    def _plan_reduce(self) -> None:
        """Band geometry for the AIV-side W reduction.

        A band is ``rows_red`` rows of ``w``.  One AIV pulls all
        ``n_head_blocks`` partials of its band into UB, one DMA each, and sums
        them register-wise, so a band has to be a whole number of vector
        registers wide.  Bands are handed out round-robin over every AIV on
        the chip, so the narrowest legal band spreads best; it is only grown
        when that would leave more than two bands per AIV.
        """
        if not self.w_workspace:
            # No partials to hold, so ``workspace`` is nothing but the grid marker.
            self.workspace_rows = self.used_cores
            self.rows_red = 0
            self.n_bands = 0
            self.n_bands_workspace = 0
            self.red_regs = 0
            self.reduce_at = "tail"
            return
        total_rows = self.n_tiles * self.base_m
        # Partials first, then the grid marker, so the partial indexing the
        # kernel does is untouched and the marker rows are never read.
        self.workspace_partial_rows = self.n_head_blocks * total_rows
        self.workspace_rows = self.workspace_partial_rows + self.used_cores
        legal = [
            rows
            for rows in range(1, self.base_m + 1)
            if self.base_m % rows == 0 and (rows * self.n_heads) % VL == 0
        ]
        if not legal:
            raise NotImplementedError(
                f"no band height divides base_m={self.base_m} into whole "
                f"{VL}-lane registers at N={self.n_heads}"
            )
        # Pinned rather than fitted to the AIV count: sizing it to
        # ``used_cores * SUBBLOCKS`` is a second way for the core count to reach
        # the binary, since this is a UB height and a register unroll.  See
        # ``ROWS_RED_K``.  Still checked against the legal set, because a band
        # that does not divide ``base_m`` into whole registers cannot be summed.
        if ROWS_RED_K not in legal:
            raise ValueError(
                f"ROWS_RED_K={ROWS_RED_K} is not a legal band height for "
                f"base_m={self.base_m} at N={self.n_heads}; legal: {legal}"
            )
        self.rows_red = ROWS_RED_K
        self.n_bands_workspace = total_rows // self.rows_red
        # ``workspace`` is indexed by tile, so it carries the padded tile height; only
        # the live rows of ``w`` are produced, so the band loop stops at T and
        # the last band may be short.
        self.n_bands = ceil_div(self.t, self.rows_red)
        self.red_regs = self.rows_red * self.n_heads // VL
        # The "early" rendezvous sits inside the work loop, which is only
        # sound while every launched core runs that loop exactly once -- a
        # core with two work items would arrive twice and one with none never.
        # Head splitting always lands on one item per core, so this holds
        # wherever w_workspace is on, but it is cheap to keep honest.
        self.reduce_at = (
            W_REDUCE_AT
            if (
                not self.t_dyn
                and self.n_tiles * self.n_head_blocks == self.used_cores
            )
            else "tail"
        )

    def _pick_k_l1_w(self) -> int:
        """Largest <=256 K window dividing this core's slice of the reduction.

        The slice is ``k_block``: the whole ``dim`` for a leader, or
        ``dim / n_head_blocks`` under ``w_k_split``.  A leader additionally
        needs its window count to divide ``heads_per_block`` so the fills can
        interleave under the Q heads; dim=5120 with 32 heads gives 160, one
        window per head.  A K slice is retired in one go, so it only needs the
        fattest window that fits.
        """
        forced = int(os.environ.get("IPQW_K_L1_W", 0))
        if forced:
            return forced
        best = 0
        for k in range(32, 257, 32):
            if self.k_block % k:
                continue
            if self.w_k_split or (self.k_block // k) % self.heads_per_block == 0:
                best = k
        return best if best else 128

    def _pick_base_k_w(self) -> int:
        """L0 K step for the W path: as large as L0A allows next to the Q path."""
        q_l0a = D_L0AB_Q * self.base_m * (
            int(os.environ.get("IPQW_BASE_K_Q", 0))
            or min(128, ceil_div(self.q_lora, MX_K_ALIGN) * MX_K_ALIGN)
        )
        budget = L0AB_BYTES - q_l0a
        best = 16
        for k in range(16, self.k_l1_w + 1, 16):
            if self.k_l1_w % k:
                continue
            if D_L0AB_W * self.base_m * k * 2 <= budget:
                best = k
        return best

    def _pick_k_windows_q(self) -> int:
        """How many K windows the ``wqb`` reduction is filled in per head.

        See the D_WQB comment at module level for why a single-head core wants
        more, smaller windows.  A count is legal only if it cuts ``q_lora``
        into whole L0 K steps that are also whole MX groups; the requested
        count is otherwise walked down to the nearest legal one.
        Each window gets its own L1 buffer, so the count is also capped by the
        CUBE buf-id file.  ``IPQW_K_L1_Q`` still names the window and is
        converted here.
        """
        forced_window = int(os.environ.get("IPQW_K_L1_Q", 0))
        want = (
            self.q_lora // forced_window
            if forced_window
            else (D_WQB_SINGLE_HEAD if self.heads_per_block == 1 else D_WQB)
        )
        max_tiles = (CUBE_BUF_IDS - FIXED_CUBE_SLOTS) // 2
        for tiles in range(min(want, max_tiles), 0, -1):
            window = self.q_lora // tiles
            if self.q_lora % tiles or window % self.base_k_q or window % MX_K_ALIGN:
                continue
            return tiles
        raise NotImplementedError(
            f"no legal wqb K window count for q_lora={self.q_lora} with "
            f"base_k_q={self.base_k_q}"
        )

    def _check(self) -> None:
        if self.d % VL or self.dr % VL:
            raise NotImplementedError(
                f"fused kernel needs D and Dr as multiples of {VL}, got D={self.d}, Dr={self.dr}"
            )
        if self.rope_regs != 1:
            # Dr == 64 keeps interleaved RoPE pairs inside one register, so pass 1 can read
            # the pre-RoPE partner lanes from the register it is about to
            # overwrite.  Dr > 64 would need every rope register live at once
            # before any write-back; not implemented, and Dr=64 is the frozen
            # case (and what MLA uses).
            raise NotImplementedError(
                f"interleaved RoPE is implemented for Dr == {VL} only, got Dr={self.dr}"
            )
        if self.dr > self.d:
            raise ValueError(f"Dr={self.dr} exceeds D={self.d}")
        if self.q_lora % MX_K_ALIGN:
            raise NotImplementedError(
                f"q_lora must be a multiple of {MX_K_ALIGN}, got {self.q_lora}"
            )
        if self.dim % self.k_l1_w:
            raise NotImplementedError(
                f"dim must be a multiple of {self.k_l1_w}, got {self.dim}"
            )
        # ``wqb`` and ``ww`` arrive in FRACTAL_NZ and are copied to L1 by an
        # identity engine, which cannot pad a partial fractal the way nd2nz
        # does.  Every window this kernel cuts out of them therefore has to
        # land on the fractal grid: 16 rows, and C0 columns (32 for the
        # one-byte wqb, 16 for BF16 ww).
        for name, extent, align in (
            ("D (wqb NZ row tile)", self.d, NZ_M_FRAC),
            ("k_l1_q (wqb NZ C0)", self.k_l1_q, NZ_C0_1B),
            ("N (ww NZ row tile)", self.n_heads, NZ_M_FRAC),
            ("k_l1_w (ww NZ C0)", self.k_l1_w, NZ_C0_2B),
        ):
            if extent % align:
                raise NotImplementedError(
                    f"FRACTAL_NZ inputs need {name} as a multiple of {align}, "
                    f"got {extent}"
                )
        self.assert_budgets()

    def _init_w(self, dim: int, n_heads: int) -> None:
        """W-path tile. K is split across the cores that share a T-tile."""
        self.base_n_w = n_heads
        self.w_k_split = (
            bool(W_K_SPLIT) and self.n_head_blocks > 1 and dim % self.n_head_blocks == 0
        )
        self.w_workspace = self.w_k_split
        self._plan_reduce()
        self.k_block = dim // self.n_head_blocks if self.w_k_split else dim
        self.w_leader_only = self.n_head_blocks > 1 and not self.w_k_split
        self.k_l1_w = self._pick_k_l1_w()
        self.base_k_w = self._pick_base_k_w()
        self.step_k_w = self.k_l1_w // self.base_k_w
        self.k_l1_tiles_w = ceil_div(self.k_block, self.k_l1_w)
        self.w_interleaved = self.k_l1_tiles_w % self.heads_per_block == 0
        self.w_separate = self.w_k_split or (
            self.w_leader_only and self.heads_per_block <= 8
        )
        if self.w_separate:
            self.w_per_head = 0
        elif self.w_interleaved:
            self.w_per_head = self.k_l1_tiles_w // self.heads_per_block
        else:
            self.w_per_head = 0

    def _init_q(self, q_lora: int, d: int) -> None:
        """Q-path K windows. One head keeps the full D on chip for RoPE."""
        self.base_n_q = d
        base_k = int(os.environ.get("IPQW_BASE_K_Q", 0))
        if not base_k:
            base_k = min(128, ceil_div(q_lora, MX_K_ALIGN) * MX_K_ALIGN)
        self.base_k_q = base_k
        self.k_l1_tiles_q = self._pick_k_windows_q()
        self.k_l1_q = q_lora // self.k_l1_tiles_q
        self.step_k_q = self.k_l1_q // self.base_k_q
        self.n_buf_wqb = self.k_l1_tiles_q
        self.k_l0_tiles_q = ceil_div(q_lora, self.base_k_q)
        self.scale_k_l0_len = self.base_k_q // MX_GROUP
        self.scale_k_l1_len_q = scale_l1_len(self.k_l1_q)
        self.scale_k_len_q = scale_l1_len(q_lora)
        self.qr_k_l1 = ceil_div(q_lora, MX_K_ALIGN) * MX_K_ALIGN

    def _init_vec(self, d: int, dr: int) -> None:
        """Vector epilogue register geometry."""
        self.n_reg = d // VL
        self.groups = d // MX_GROUP
        self.rope_regs = dr // VL
        self.enc_regs = ceil_div(self.rows_vec * self.groups, VL)
        self.amax_len = self.enc_regs * VL

# The plan the public entry will use, sharing the cache with _plan_for.
# Use this to read base_m and the other live parameters. A freshly built
# static Plan can disagree with the binary that will actually run.
@dataclass(frozen=True)
class Geometry:
    """Frozen axes plus the optional cube-core override. T is not part of it."""

    dim: int
    q_lora: int
    n_heads: int
    d: int
    dr: int
    cube_cores: int | None = None


def plan(t: int, geo: Geometry) -> Plan:
    """The plan that will actually serve this T.

    Goes through the same cache the host entry does, so a caller sizing its
    buffers off ``base_m`` gets the tile height the kernel will really use.
    Building a fresh ``Plan`` here would size the tile to T and hand back a
    smaller height than the binary serving that T was compiled with.
    """
    return _plan_for(t, geo).p


# ---------------------------------------------------------------------------
# Cross-core rendezvous for the W reduction
# ---------------------------------------------------------------------------

# Cross-core barrier before the W reduction. Every AIC must have stored its
# partials to GM before any AIV reads them, so a later call cannot see leftovers.
@jit
def store_to_vec_barrier():
    """All-core rendezvous for "every AIC's W partial is in GM".

    ``global_sync_all`` emits four handshakes, two of which serve
    dependencies this barrier does not have.  Its phase 1 funnels each
    block's AIVs into their own AIC, which matters when an AIV produced
    something the grid has to see; here the producers are the AICs and the
    AIVs are the consumers.  Its phase 3 is an all-AIV meeting point its own
    docstring calls redundant on a mix kernel.  What is left is the grid
    barrier over the AICs, then the handoff to each block's own AIVs.

    Flag ids match ``global_sync_all`` phases 2 and 4: the same explicit pair
    the API documents.

    Worth 0.75 us of the 1.77 us the full barrier costs at T=72.  The arrive
    goes out on FIXPIPE and is preceded by ``cube_sync_all`` because an FFTS
    arrive drains only the pipe it is issued on: the partial is a fixpipe
    store, and a peer must not see the flag before the store lands.
    """
    grid_flag, funnel_flag = 1, 2
    cube_sync_all()
    cube_sync_block_arrive(PIPE.FIXPIPE, grid_flag, mode=0)
    cube_sync_block_wait(PIPE.S, grid_flag, mode=0)
    cube_sync_block_arrive(PIPE.MTE3, funnel_flag, mode=2)
    vec_sync_block_wait(PIPE.S, funnel_flag)


# Prefetch this core's next qr tile and its scale so the copy overlaps the
# current epilogue. work is the task id; a task is one (tile, head-group) pair.
@jit
def _prefetch_qr(cube_q, qr_gm, descale_qr_gm, work, n_work):
    """Stage the next work item's ``qr`` so it rides under this tile's epilogue.

    It sits after the head loop rather than inside its last iteration, where it
    used to: the staging channels are depth 1 and the read transaction now spans
    the whole head loop, so a fill issued inside it would block on its own
    reader.  On the cube's own timeline that moves it past one
    ``drain_to_vec``, which is a fixpipe op the MTE2 fill does not wait for.
    """
    p = cube_q.p
    tile = work // p.n_head_blocks
    if const_expr(p.prefetch):
        next_work = work + get_block_num()
        if next_work < n_work:
            next_tile = next_work // p.n_head_blocks
            if next_tile != tile:
                cube_q.load_qr(qr_gm, descale_qr_gm, next_tile)


@jit
def w_barrier():
    if const_expr(W_SYNC == "lean"):
        store_to_vec_barrier()
    else:
        global_sync_all()


# ---------------------------------------------------------------------------
# Cube modules
# ---------------------------------------------------------------------------

# Q GEMM: one token tile and one head, qr @ wqb_head.T.
# qr and its scale stay in L1. wqb streams in by K window. The FP32
# accumulator lives in L0C.
class CubeQ:
    """MXFP8 ``qr @ wqb^T`` for one (T tile, head); owns its L1/L0 staging."""

    def __init__(self, p: Plan):
        self.p = p
        self.qr_l1 = Channel(
            MemLoc.L1, (p.base_m, p.qr_k_l1), Float8E4M3FN, depth=1, data_format="nz"
        )
        self.scale_qr_l1 = Channel(
            MemLoc.L1, (p.base_m, p.scale_k_len_q), Float8E8M0, depth=1, data_format="zn"
        )
        # ``wqb`` is handed over already in FRACTAL_NZ.  The GM-to-L1 identity
        # copy is a raw byte move and insists both sides share a dtype, and the
        # one-byte elements FRACTAL_NZ can describe are only the integers -- so
        # the staging channel is declared uint8 for the fill and read back
        # through an fp8 alias.
        #
        # One depth-1 channel per K window rather than one channel of that
        # depth, because the fp8 view has to be taken on the slot (see
        # gemm_head) and a slot only has a statically known byte origin when
        # its channel is depth 1.
        self.wqb_l1 = [
            Channel(
                MemLoc.L1,
                (p.base_n_q, p.k_l1_q),
                U8,
                depth=1,
                data_format="nz",
            )
            for _ in range(p.n_buf_wqb)  # == k_l1_tiles_q
        ]
        # No alias on the scales, so one rotating channel is fine here.
        self.scale_wqb_l1 = Channel(
            MemLoc.L1,
            (p.scale_k_l1_len_q, p.base_n_q),
            Float8E8M0,
            depth=p.n_buf_wqb,
            data_format="nz",
        )
        self.l0a = Channel(MemLoc.L0A, (p.base_m, p.base_k_q), Float8E4M3FN, depth=D_L0AB_Q)
        self.l0b = Channel(MemLoc.L0B, (p.base_n_q, p.base_k_q), Float8E4M3FN, depth=D_L0B_Q)
        self.l0c = Channel(MemLoc.L0C, (p.base_m, p.base_n_q), F32, depth=D_L0C_Q)

        self.eng_nd2nz = make_copy_engine(
            format_transform="nd2nz", dtype=Float8E4M3FN, pad_value=0.0
        )
        # ``wqb`` arrives in FRACTAL_NZ, so its L1 fill is a plain burst copy.
        # The ND route had to gather base_n_q separate q_lora-strided rows per
        # window; the NZ route reads whole (16, 32) fractals back to back,
        # which is what makes this the cheapest 5.24 MB on the chip.
        self.eng_nz = make_copy_engine(format_transform="identity", dtype=U8)
        self.eng_scale_a = make_copy_engine(
            format_transform="mx_scale_and", dtype=Float8E8M0, pad_value=0.0
        )
        self.eng_scale_b = make_copy_engine(
            format_transform="mx_scale_bdn", dtype=Float8E8M0, pad_value=0.0
        )
        # Hardware sends each balanced M partition to its corresponding AIV.
        self.eng_c2v = make_copy_engine(split_axis=0)

    # Copy qr and its scale for token tile into L1. Every head of this task
    # reuses them. A short last tile copies only the live rows; stores are
    # clipped to those rows as well.
    @jit
    def load_qr(self, qr_gm, descale_qr_gm, tile):
        """Stage one T-tile's ``qr`` and its scales into L1.

        The GM side is the tile view's own extent, so a short last tile reads
        only its live rows and the rest of the L1 tile keeps whatever the last
        tile through this buffer left there.  The mmads reduce those rows and
        the epilogue's store is cut to the live ones (see ``VectorQ.store``), so
        nothing reads what they produced.

        The transaction is manual only so that the read side can name the row
        height -- see ``rows_view``.
        """
        p = self.p
        mem_copy(
            self.qr_l1.produce(),
            tile_view(qr_gm, (p.base_m, p.qr_k_l1), (tile, 0)),
            engine=self.eng_nd2nz,
            l2_cache_ctl=p.l2_q,
        )
        mem_copy(
            self.scale_qr_l1.produce(),
            tile_view(
                descale_qr_gm,
                (p.base_m, p.scale_k_len_q // MX_PAIR, MX_PAIR),
                (tile, 0, 0),
            ),
            engine=self.eng_scale_a,
            l2_cache_ctl=p.l2_q,
        )

    @jit
    def wait_qr(self):
        """The staged ``qr`` tile and its scales, as one transaction pair."""
        return self.qr_l1.consume(), self.scale_qr_l1.consume()

    # Physical layout of a short tail. rows_l1 rounds the live row count up
    # to 16. Later L1 reads must use that same row pitch. Treating the tail
    # as a full 128 rows misaligns the K windows.
    @jit
    def rows_view(self, qr_slot, scale_slot, rows_l1):
        """The staged tile cut to the row height its fill actually wrote.

        A GM-to-L1 nd2nz copy of a short tile programs its NZ row stride as
        ``ceil(rows/16)*16``, so a read issued at the declared ``base_m`` walks
        at ``base_m/16`` fractals and lands on the wrong rows for every K window
        but the first -- wrong values, not a short read.  T=8229 read back 1955
        wrong ``descale_q`` bytes that way and T=4296 read back 3645; it needs a
        T past the prefetch threshold that is not a multiple of 128, and every
        prefill T in the original shape list is a multiple of 1024, which is why
        it went unnoticed.

        The layout pass infers exactly this height on its own, but only while
        one unconditional fill reaches the reads.  The prefetch adds a second
        fill site behind a runtime ``if``, the two runtime extents join to the
        declared height, and an explicit view is rejected on an implicit channel
        ("cannot project typed Channel reaching shape through explicit tile_view
        read") -- hence the manual transaction, which hands back a plain slot
        this can slice.  Measured: with the fill unconditional the inference is
        right at every T; with the prefetch on it is wrong at every T that is
        not a multiple of 128.

        The scales are cut to the same height for their row step only.  Their
        fill is ``mx_scale_and``, which does not pack to the tail, so their row
        stride is the declared one either way.
        """
        p = self.p
        qr = local_slice(
            qr_slot,
            make_bounded_tiler(
                (rows_l1, p.qr_k_l1), (p.base_m, p.qr_k_l1), (NZ_M_FRAC, 1)
            ),
            stride=(p.qr_k_l1, 1),
        )
        scale = local_slice(
            scale_slot,
            make_bounded_tiler(
                (rows_l1, p.scale_k_len_q),
                (p.base_m, p.scale_k_len_q),
                (NZ_M_FRAC, 1),
            ),
            stride=(p.scale_k_len_q, 1),
        )
        return qr, scale

    @jit
    def _fill_wqb(self, wqb_gm, descale_wqb_gm, head, kw, buf):
        """Stage K window ``kw`` of this head's ``wqb`` and its scales into L1."""
        p = self.p
        groups = p.scale_k_l1_len_q // MX_PAIR
        # Filled through the uint8 face, read back through an fp8 reinterpret
        # in gemm_head.  Both halves are written out by hand so the fill and
        # the read name the same depth-1 buffer.
        mem_copy(
            self.wqb_l1[buf].produce(),
            tile_view(wqb_gm, (p.base_n_q, p.k_l1_q), (head, kw)),
            engine=self.eng_nz,
            l2_cache_ctl=p.l2_q,
        )
        mem_copy(
            self.scale_wqb_l1.produce(),
            tile_view(descale_wqb_gm, (p.base_n_q, groups, MX_PAIR), (head, kw, 0)),
            engine=self.eng_scale_b,
            l2_cache_ctl=p.l2_q,
        )

    # Reduce the whole q_lora K of one head into one L0C slot.
    # init is true only on the first K block. The next weight window can be
    # loaded while the current one is consumed.
    @jit
    def gemm_head(self, wqb_gm, descale_wqb_gm, head, qr_rows, scale_rows):
        """Accumulate the whole q_lora reduction for one head into L0C.

        Window kw+1 is staged before window kw is consumed, so ``wqb_l1`` keeps
        MTE2 busy underneath the mmads instead of showing each window's burst
        up as a cube stall.
        """
        p = self.p
        # Select once per head: every K step accumulates into this same slot.
        l0c = self.l0c.produce()
        self._fill_wqb(wqb_gm, descale_wqb_gm, head, 0, 0)
        # Unrolled: keeps ``init`` a compile-time bool, so the hottest loop in
        # the kernel has no scf.if around its mmad.
        for kw in range_constexpr(p.k_l1_tiles_q):
            if const_expr(kw + 1 < p.k_l1_tiles_q):
                self._fill_wqb(
                    wqb_gm, descale_wqb_gm, head, kw + 1, (kw + 1) % p.n_buf_wqb
                )
            buf = kw % p.n_buf_wqb
            wqb_ready = reinterpret(
                self.wqb_l1[buf].consume(), dtype=Float8E4M3FN
            )
            scale_ready = self.scale_wqb_l1.consume()
            for kk in range_constexpr(p.step_k_q):
                k_l0 = kw * p.step_k_q + kk
                mem_copy(
                    self.l0a.produce(),
                    tile_view(qr_rows, (p.base_m, p.base_k_q), (0, k_l0)),
                    mx_scale=tile_view(
                        scale_rows, (p.base_m, p.scale_k_l0_len), (0, k_l0)
                    ),
                )
                mem_copy(
                    self.l0b.produce(),
                    tile_view(wqb_ready, (p.base_n_q, p.base_k_q), (0, kk)),
                    mx_scale=tile_view(
                        scale_ready, (p.scale_k_l0_len, p.base_n_q), (kk, 0)
                    ),
                )
                # matmul takes M from this operand view, not its last fill.
                a_rows = local_slice(
                    self.l0a.consume(), (qr_rows.shape[0], p.base_k_q)
                )
                if const_expr(UNIT_FLAG):
                    last_k = k_l0 == p.k_l0_tiles_q - 1
                    matmul(
                        l0c,
                        a_rows,
                        self.l0b.consume(),
                        init=(k_l0 == 0),
                        unit_flag=3 if last_k else 2,
                    )
                else:
                    matmul(l0c, a_rows, self.l0b.consume(), init=(k_l0 == 0))

    # Hand the Q accumulator from L0C straight to the vector UB. No GM hop.
    # rows is the live output height. rows_l1 is the accumulator layout.
    # They differ on a short tail.
    @jit
    def drain_to_vec(self, cv_ub, rows, rows_l1):
        # The accumulator is packed with the M used by matmul on this tile.
        c_rows = local_slice(self.l0c.consume(), (rows_l1, self.p.d))
        mem_copy(
            cv_ub.produce(), c_rows, engine=self.eng_c2v,
            actual=(rows, self.p.d), unit_flag=3 if UNIT_FLAG else 0,
        )
        return cv_ub.consume()


# W GEMM: x @ weight_w.T, with softmax_scale applied on the store.
# split-T writes w directly. split-K writes workspace partials and the
# vector unit sums the K slices.
class CubeW:
    """BF16 ``x @ ww^T`` scaled in fixpipe; writes GM directly, no vector hop."""

    def __init__(self, p: Plan):
        self.p = p
        bf16 = dtypes.bfloat16
        self.x_l1 = Channel(
            MemLoc.L1, (p.base_m, p.k_l1_w), bf16, depth=D_X, data_format="nz"
        )
        self.ww_l1 = Channel(
            MemLoc.L1, (p.base_n_w, p.k_l1_w), bf16, depth=D_X, data_format="nz"
        )
        self.l0a = Channel(MemLoc.L0A, (p.base_m, p.base_k_w), bf16, depth=D_L0AB_W)
        self.l0b = Channel(MemLoc.L0B, (p.base_n_w, p.base_k_w), bf16, depth=D_L0AB_W)
        self.l0c = Channel(MemLoc.L0C, (p.base_m, p.base_n_w), F32, depth=2)
        self.eng_nd2nz = make_copy_engine(
            format_transform="nd2nz", dtype=bf16, pad_value=0.0
        )
        # ``ww`` arrives in FRACTAL_NZ; ``x`` is an activation and stays ND.
        self.eng_nz = make_copy_engine(format_transform="identity", dtype=bf16)
        self.eng_fp = make_copy_engine(dtype=F32)

    # View of x that matches the L1 copy. A short tail uses rows_l1 as the
    # row pitch so later column blocks are not read from the wrong offset.
    @jit
    def x_rows(self, slot, rows_l1):
        """This tile's ``x`` in L1, at the row height its fill actually wrote.

        The GM-to-L1 nd2nz copy of a short last tile writes the live rows and
        rounds the fractal grid up: the NZ row stride it programs is
        ``ceil(rows/16)*16``, and the rows the rounding adds are the engine's
        pad value.  A read has to be issued at that same height or its own
        fractal stride is the declared ``base_m/16``, which walks past every K
        window but the first -- which is how a short tail used to corrupt the
        upper half of the head axis and why ``x`` used to be handed over grown
        to whole tiles in GM.

        The height cannot be the channel's declared shape, because only the last
        tile is short, and it cannot be inferred either: a manual produce/consume
        slot is a fresh root, so the fill's row count never reaches the read.
        (The ``qr`` staging is an implicit channel, where the layout pass does
        propagate it; see ``CubeQ.load_qr``.)  So it is passed in, and this is
        the same ``local_slice`` a dynamic-shape Channel would apply itself.
        """
        p = self.p
        return local_slice(
            slot,
            make_bounded_tiler(
                (rows_l1, p.k_l1_w), (p.base_m, p.k_l1_w), (NZ_M_FRAC, 1)
            ),
            stride=(p.k_l1_w, 1),
        )

    # Interpret L0C at the rows_l1 height the matmul used, so the store
    # reads the head columns from the right place.
    @jit
    def c_rows(self, slot, rows_l1):
        """This tile's L0C, at the row height the mmads wrote it at.

        An mmad lays C out as ``ceil(N/16)`` blocks of (M, 16), so its M is the
        stride between one head fractal and the next.  ``x_rows`` makes that M
        the rounded-up row count of the tile, which leaves the fixpipe store
        reading at the declared ``base_m`` stride and picking up the wrong half
        of the head axis -- the exact corruption the GM padding was hiding.
        """
        p = self.p
        return local_slice(
            slot,
            make_bounded_tiler(
                (rows_l1, p.base_n_w), (p.base_m, p.base_n_w), (NZ_M_FRAC, 1)
            ),
            stride=(p.base_n_w, 1),
        )

    @jit
    def gemm_head_slice(self, tile, local_head, h_block, l0c_slot, rows_l1):
        """Retire this local head's share of the K reduction into ``l0c_slot``.

        Called from inside the Q head loop so the ``x`` traffic trickles in on
        MTE2 underneath the Q path's cube work.  ``local_head`` is 0 ..
        heads_per_block-1 on this core; ``h_block`` selects which slice of
        ``ww`` / ``w`` this core owns.  ``local_head`` is a runtime index,
        hence the manual L0C transaction: with an implicit channel the
        ``scf.if`` around the ``init`` matmul reads as a second writer.
        """
        p = self.p
        self._fill_window(
            self.x_gm, self.ww_gm, tile, h_block, local_head * p.w_per_head
        )
        for j in range_constexpr(p.w_per_head):
            if const_expr(j + 1 < p.w_per_head):
                self._fill_window(
                    self.x_gm, self.ww_gm, tile, h_block, local_head * p.w_per_head + j + 1
                )
            x_ready = self.x_l1.consume()
            ww_ready = self.ww_l1.consume()
            x_rows = self.x_rows(x_ready, rows_l1)
            for kk in range_constexpr(p.step_k_w):
                mem_copy(self.l0a.produce(), tile_view(x_rows, (p.base_m, p.base_k_w), (0, kk)))
                mem_copy(self.l0b.produce(), tile_view(ww_ready, (p.base_n_w, p.base_k_w), (0, kk)))
                a_rows = local_slice(self.l0a.consume(), (rows_l1, p.base_k_w))
                first = kk == 0 and j == 0
                if const_expr(PROBE == "no_runtime_init"):
                    matmul(l0c_slot, a_rows, self.l0b.consume(), init=False)
                elif const_expr(PROBE == "no_w_mmad"):
                    pass
                else:
                    matmul(
                        l0c_slot,
                        a_rows,
                        self.l0b.consume(),
                        init=(local_head == 0) if const_expr(first) else False,
                    )

    @jit
    def _fill_window(self, x_gm, ww_gm, tile, k_base, kw):
        """Stage K window ``k_base + kw`` of ``x`` and ``ww`` into L1.

        ``k_base`` is this core's offset into the reduction, in windows: zero
        for a leader that owns the whole ``dim``, ``h_block * k_l1_tiles_w``
        under ``w_k_split``.
        """
        p = self.p
        mem_copy(
            self.x_l1.produce(),
            tile_view(x_gm, (p.base_m, p.k_l1_w), (tile, k_base + kw)),
            engine=self.eng_nd2nz,
            l2_cache_ctl=L2_W,
        )
        mem_copy(
            self.ww_l1.produce(),
            tile_view(ww_gm, (p.base_n_w, p.k_l1_w), (0, k_base + kw)),
            engine=self.eng_nz,
            l2_cache_ctl=L2_W,
        )

    @jit
    def fill(self, tile, local_head, h_block):
        """Issue this local head's W-path L1 fills; does not wait for them."""
        p = self.p
        for j in range_constexpr(p.w_per_head):
            self._fill_window(
                self.x_gm, self.ww_gm, tile, h_block, local_head * p.w_per_head + j
            )

    @jit
    def drain(self, local_head, l0c_slot, rows_l1):
        """Retire the oldest staged window pair into ``l0c_slot``."""
        p = self.p
        for j in range_constexpr(p.w_per_head):
            x_ready = self.x_l1.consume()
            ww_ready = self.ww_l1.consume()
            x_rows = self.x_rows(x_ready, rows_l1)
            for kk in range_constexpr(p.step_k_w):
                mem_copy(self.l0a.produce(), tile_view(x_rows, (p.base_m, p.base_k_w), (0, kk)))
                mem_copy(self.l0b.produce(), tile_view(ww_ready, (p.base_n_w, p.base_k_w), (0, kk)))
                a_rows = local_slice(self.l0a.consume(), (rows_l1, p.base_k_w))
                first = kk == 0 and j == 0
                matmul(
                    l0c_slot,
                    a_rows,
                    self.l0b.consume(),
                    init=(local_head == 0) if const_expr(first) else False,
                )

    # Finish this task's whole W reduction. Under split-K that is only
    # this core's K slice.
    @jit
    def gemm_full(self, tile, k_base, l0c_slot, rows_l1):
        """This core's whole K slice as one reduction; used when Q cannot hide it."""
        p = self.p
        self._fill_window(self.x_gm, self.ww_gm, tile, k_base, 0)
        for kw in range_constexpr(p.k_l1_tiles_w):
            if const_expr(kw + 1 < p.k_l1_tiles_w):
                self._fill_window(self.x_gm, self.ww_gm, tile, k_base, kw + 1)
            x_ready = self.x_l1.consume()
            ww_ready = self.ww_l1.consume()
            x_rows = self.x_rows(x_ready, rows_l1)
            for kk in range_constexpr(p.step_k_w):
                mem_copy(self.l0a.produce(), tile_view(x_rows, (p.base_m, p.base_k_w), (0, kk)))
                mem_copy(self.l0b.produce(), tile_view(ww_ready, (p.base_n_w, p.base_k_w), (0, kk)))
                a_rows = local_slice(self.l0a.consume(), (rows_l1, p.base_k_w))
                first = kw == 0 and kk == 0
                matmul(l0c_slot, a_rows, self.l0b.consume(), init=const_expr(first))

    @jit
    def store(self, tile, rows_l1):
        """Whole-reduction store, straight into ``w``. Not used under w_workspace."""
        # softmax_scale folded into the fixpipe DEQSCALE register: the W path
        # never touches the vector unit.
        p = self.p
        mem_copy(
            tile_view(self.w_gm, (p.base_m, p.base_n_w), (tile, 0)),
            self.c_rows(self.l0c.consume(), rows_l1),
            engine=self.eng_fp,
            deq_scale=self.softmax_scale,
            l2_cache_ctl=L2_W,
        )

    # Each K slice writes its W partial into a workspace region only it owns.
    # The store overwrites; it is not an atomic add, so the buffer does not
    # need to be zeroed. The vector unit sums the slices later.
    @jit
    def store_partial(self, tile, h_block, n_tiles, rows_l1):
        """Write this core's K-slice partial to the rows it alone owns.

        A plain store, not an atomic add: nothing else writes these rows, so
        they need no pre-zeroing and no barrier ahead of the store.  Scaling
        each partial by ``softmax_scale`` in the fixpipe is equivalent to
        scaling the sum because the scale is a constant.

        ``workspace`` is allocated on the padded tile grid, so the destination is
        always full height while the source is only ``rows_l1`` tall on a short
        last tile; the rows past that are left holding whatever the previous
        launch put there.  The reduction below drops them because its own output
        view is cut to the live rows of ``w``.
        """
        p = self.p
        mem_copy(
            tile_view(
                self.workspace_gm, (p.base_m, p.base_n_w), (h_block * n_tiles + tile, 0)
            ),
            self.c_rows(self.l0c.consume(), rows_l1),
            engine=self.eng_fp,
            deq_scale=self.softmax_scale,
            l2_cache_ctl=L2_W,
        )


# ---------------------------------------------------------------------------
# Vector module
# ---------------------------------------------------------------------------

# Vector epilogue. Q gets RoPE, MXFP4 quantisation in groups of 32, and
# packed FP4 stores. The same unit also sums W partials in the split-K
# template.
class VectorQ:
    """Inplace RoPE plus MXFP4 quantisation of one head tile, raw VF only."""

    def __init__(self, p: Plan):
        self.p = p
        rows = p.rows_vec
        # D is a multiple of 64, so q_ub's rows are already 32-byte aligned and
        # its declared stride is its real stride.  dsc_ub's natural shape
        # (rows, groups) is not: UB rounds the innermost axis up to 32 bytes, so
        # a 4-byte row would really be 32 bytes apart and the linear offsets
        # pass 2 writes at would land in the padding.  Declare it in the shape
        # pass 2 actually writes (one 64-byte register per row) and let ``store``
        # re-view it as (rows, groups) with an explicit packed stride.
        self.q_ub = Buffer(MemLoc.UB, (rows, p.d // 2), U8)
        self.dsc_ub = Buffer(MemLoc.UB, (p.enc_regs, VL), U8)
        # Auto-buffer-sync orders MTE2 DMA against VF reads on Buffers.
        self.amax_ub = Buffer(MemLoc.UB, (p.amax_len,), F32)
        self.off_ub = Buffer(MemLoc.UB, (p.amax_len,), I32)
        self.lut_ub = Buffer(MemLoc.UB, (LUT_LEN,), I32)
        self.cos_ub = Buffer(MemLoc.UB, (rows, max(p.dr, 1)), F32)
        self.sin_ub = Buffer(MemLoc.UB, (rows, max(p.dr, 1)), F32)
        self.eng_split = make_copy_engine(split_axis=0)
        # W reduction staging.  ``band_ub`` holds every partial of one band so
        # the sum is whole-register adds; partial ``i`` starts exactly
        # ``red_regs`` registers in, because a band's innermost extent
        # (n_heads fp32 = 128 B) is already 32-byte aligned and so the
        # declared stride is the real one.  Plain Buffers: the DMA-to-VF and
        # VF-to-DMA handoffs are both ordered by vec_sync_all, which is
        # coarse but runs at most twice per band.
        if p.w_workspace:
            self.band_ub = Buffer(
                MemLoc.UB, (p.n_head_blocks * p.rows_red, p.n_heads), F32
            )
            self.wsum_ub = Buffer(MemLoc.UB, (p.rows_red, p.n_heads), F32)

    # Hand W bands out across every AIV. Each band is summed and stored by
    # one AIV, so the writes do not collide.
    @jit
    def reduce_all(self, workspace, w_gm, n_bands, n_bands_workspace):
        """Sum the bands this AIV owns, round-robin over every AIV on chip.

        ``get_subblock_dim()`` is 2 on the vector core and 1 on the cube, so
        the cube side walks a different, and empty, band range.  Both counts are
        runtime under ``t_dyn``: a band is a fixed ``rows_red`` rows, so how many
        there are is the only thing T changes here.
        """
        aiv = get_block_idx() * get_subblock_dim() + get_subblock_id()
        for band in range(aiv, n_bands, get_block_num() * get_subblock_dim()):
            self.reduce_w(workspace, w_gm, band, n_bands_workspace)

    # Load every K-slice partial of one band, add them in UB, and store
    # the sum to w.
    @jit
    def reduce_w(self, workspace, w_gm, band, n_bands_workspace):
        """Sum every partial of one band out of UB and write it to ``w``."""
        p = self.p
        for part in range_constexpr(p.n_head_blocks):
            mem_copy(
                local_slice(
                    self.band_ub,
                    (p.rows_red, p.n_heads),
                    offset=part * p.red_regs * VL * 4,
                ),
                tile_view(
                    workspace,
                    (p.rows_red, p.n_heads),
                    (part * n_bands_workspace + band, 0),
                ),
            )
        vec_sync_all()
        with vf(mode="simd"):
            m32 = full_mask()
            for reg in range_constexpr(p.red_regs):
                acc = vload(self.band_ub, reg * VL)
                for part in range_constexpr(1, p.n_head_blocks):
                    acc = vadd(
                        acc,
                        vload(self.band_ub, (part * p.red_regs + reg) * VL),
                        mask=m32,
                    )
                vstore(self.wsum_ub, reg * VL, acc, m32)
        vec_sync_all()
        out = tile_view(w_gm, (p.rows_red, p.n_heads), (band, 0))
        mem_copy(out, local_slice(self.wsum_ub, (out.shape[0], p.n_heads)))

    @jit
    def load_tile_constants(self, lut_gm, cos_tile, sin_tile):
        """Per-T-tile prologue: DMA the LUT and the cos/sin tables into UB."""
        mem_copy(self.lut_ub, tile_view(lut_gm, (LUT_LEN,), (0,)))
        if const_expr(self.p.dr > 0):
            mem_copy(
                self.cos_ub, cos_tile, engine=self.eng_split,
                part_id=get_subblock_id(),
            )
            mem_copy(
                self.sin_ub, sin_tile, engine=self.eng_split,
                part_id=get_subblock_id(),
            )
        return self.lut_ub, self.cos_ub, self.sin_ub

    # Q epilogue, three passes: RoPE on the tail Dr channels, per-group
    # scale, then round and pack FP4. For a normal nonzero amax,
    # scale = 2**ceil(log2(amax/6)). Zero and tiny amax take other branches.
    # descale_q stores the E8M0 code of the scale that is multiplied back on
    # dequant, not its reciprocal. rows is the live row count of this AIV;
    # using the buffer capacity would walk off a short tail.
    @jit
    def epilogue(self, cv_ub, rows, lut_ub, cos_ub, sin_ub):
        """RoPE then MXFP4 quantise ``cv_ub`` into ``q_ub`` / ``dsc_ub``."""
        self._rope_amax(cv_ub, rows, cos_ub, sin_ub)
        self._e8m0_scales()
        self._pack_e2m1(cv_ub, rows, lut_ub)

    # Store packed q and the scale from UB to GM, limited to this
    # partition's live rows.
    @jit
    def store(self, q_tile, dsc_tile, rows):
        """Write back exactly ``rows`` rows, so a short T tail cannot overrun GM."""
        p = self.p
        mem_copy(
            q_tile,
            local_slice(self.q_ub, (rows, p.d // 2), stride=(p.d // 2, 1)),
            engine=self.eng_split, part_id=get_subblock_id(),
        )
        mem_copy(
            dsc_tile,
            local_slice(self.dsc_ub, (rows, p.groups), stride=(p.groups, 1)),
            engine=self.eng_split, part_id=get_subblock_id(),
        )


    @jit
    def _rope_amax(self, cv_ub, rows, cos_ub, sin_ub):
        """Inplace RoPE on the tail lanes, then per-group absolute max."""
        p = self.p
        with vf(mode="simd"):
            m32 = full_mask()
            m_lo = create_mask("h", 32)
            lane = varange(0, I32)
            one_i = vdups(1, I32, mask=m32)
            pair_idx = vreinterpret(
                vbitwise_xor(lane, one_i, mask=m32), U32
            )
            group_idx = vreinterpret(
                vbitwise_and(
                    vadds(lane, VL // 2, mask=m32),
                    vdups(VL - 1, I32, mask=m32),
                    mask=m32,
                ), U32
            )
            rope_sgn = vselect(
                vdups(1.0, F32, mask=m32),
                vdups(-1.0, F32, mask=m32),
                cond_mask=vne(vbitwise_and(lane, one_i, mask=m32), vdups(0, I32, mask=m32), mask=m32),
            )
            for r in range(rows):
                base = r * p.d
                for i in range_constexpr(p.n_reg):
                    x = vload(cv_ub, base + i * VL)
                    pe = i - (p.n_reg - p.rope_regs)
                    if const_expr(p.rope_regs > 0 and pe >= 0):
                        cosv = vload(cos_ub, r * p.dr + pe * VL)
                        sinv = vload(sin_ub, r * p.dr + pe * VL)
                        rot = vmul(
                            vreinterpret(
                                vgather_reg(vreinterpret(x, I32), pair_idx), F32
                            ),
                            rope_sgn,
                            mask=m32,
                        )
                        # vmadd(a, b, c) is the hardware a*b+c, so the addend is last.
                        x = vmadd(rot, sinv, vmul(x, cosv, mask=m32), mask=m32)
                        vstore(cv_ub, base + i * VL, x, m32)
                    abs_d = vabs(x, mask=m32)
                    abs_d_hi = vreinterpret(
                        vgather_reg(vreinterpret(abs_d, I32), group_idx), F32
                    )
                    slot = r * p.groups + 2 * i
                    vstore_first(self.amax_ub, slot, vreduce_max(abs_d, mask=m_lo))
                    vstore_first(
                        self.amax_ub, slot + 1, vreduce_max(abs_d_hi, mask=m_lo)
                    )
            # Same-region store is not ordered against a later load.
            vmem_bar("vst_vld")

    @jit
    def _e8m0_scales(self):
        """E8M0 code from the fp32 exponent. ceil(log2(amax/6)) is E-2 plus a mantissa bump."""
        p = self.p
        with vf(mode="simd"):
            m32 = full_mask()
            zero_i = vdups(0, I32, mask=m32)
            one_i = vdups(1, I32, mask=m32)
            k_bias = vdups(E8M0_BIAS, I32, mask=m32)
            mant_mask = vdups(FP32_MANTISSA_MASK, I32, mask=m32)
            for e in range(p.enc_regs):
                ab = vreinterpret(vload(self.amax_ub, e * VL), I32)
                exp_bits = vshr(ab, FP32_EXP_SHIFT, mask=m32)
                mant = vbitwise_and(ab, mant_mask, mask=m32)
                bump = vselect(
                    one_i, zero_i, cond_mask=vgts(mant, FP32_HALF_MANTISSA, mask=m32)
                )
                e8m0_exp = vadd(vadds(exp_bits, -2, mask=m32), bump, mask=m32)
                e8m0_exp = vmaxs(e8m0_exp, 1, mask=m32)
                e8m0_exp = vselect(
                    e8m0_exp, k_bias, cond_mask=vne(ab, zero_i, mask=m32)
                )
                vstore(
                    self.off_ub,
                    e * VL,
                    vshl(vsub(e8m0_exp, k_bias, mask=m32), 3, mask=m32),
                    m32,
                )
                vstore_pack(
                    self.dsc_ub, e * VL, e8m0_exp, m32, pack_mode=PackMode.B32_TO_B8
                )
            vmem_bar("vst_vld")

    @jit
    def _pack_e2m1(self, cv_ub, rows, lut_ub):
        """Gather the magnitude code, merge the sign, pack two nibbles per byte."""
        p = self.p
        with vf(mode="simd"):
            m32 = full_mask()
            m_lo = create_mask("h", 32)
            lane = varange(0, I32)
            zero_i = vdups(0, I32, mask=m32)
            k8 = vdups(8, I32, mask=m32)
            mag_mask = vdups(FP32_MAG_MASK, I32, mask=m32)
            lut_remainder_mask = vdups((1 << LUT_SHIFT) - 1, I32, mask=m32)
            one_i = vdups(1, I32, mask=m32)
            odd = vne(vbitwise_and(lane, one_i, mask=m32), zero_i, mask=m32)
            pack_w = vselect(
                vdups(16.0, F32, mask=m32), vdups(1.0, F32, mask=m32), cond_mask=odd
            )
            gsel = vshr(lane, 5, mask=m32)
            for r in range(rows):
                base = r * p.d
                gbase = vadds(gsel, r * p.groups, mask=m32)
                for i in range_constexpr(p.n_reg):
                    off = vgather(
                        self.off_ub,
                        vreinterpret(vadds(gbase, 2 * i, mask=m32), U32),
                        mask=m32,
                    )
                    xv = vload(cv_ub, base + i * VL)
                    mag = vbitwise_and(vreinterpret(xv, I32), mag_mask, mask=m32)
                    idx = vsub(vshr(mag, LUT_SHIFT, mask=m32), off, mask=m32)
                    idx = vmaxs(idx, 0, mask=m32)
                    code = vgather(lut_ub, vreinterpret(idx, U32), mask=m32)
                    # The LUT takes the upper value at every midpoint.  For
                    # RNE, the exact midpoints 0.25, 1.25, 2.5 and 5.0 must
                    # instead take the lower (even) E2M1 code.  ``idx`` is
                    # the descaled fp32 bucket, while the discarded LUT bits
                    # distinguish the exact midpoint from values just above it.
                    tie_lower = vselect(
                        one_i, zero_i,
                        cond_mask=veq(idx, vdups(1000, I32, mask=m32), mask=m32),
                    )
                    for midpoint_idx in (1018, 1026, 1034):
                        tie_lower = vselect(
                            one_i, tie_lower,
                            cond_mask=veq(
                                idx, vdups(midpoint_idx, I32, mask=m32), mask=m32
                            ),
                        )
                    tie_code = vselect(
                        vsub(code, one_i, mask=m32), code,
                        cond_mask=veq(
                            vbitwise_and(mag, lut_remainder_mask, mask=m32), zero_i,
                            mask=m32,
                        ),
                    )
                    code = vselect(
                        tie_code, code, cond_mask=vne(tie_lower, zero_i, mask=m32)
                    )
                    # Sign follows q < 0, so negative zero encodes as nibble 0.
                    neg = vlts(xv, 0.0, mask=m32)
                    sign = vselect(k8, zero_i, cond_mask=neg)
                    nib = vbitwise_or(code, sign, mask=m32)
                    nibf = vcast(nib, F32, mask=m32)
                    packed = vcast(
                        vpair_reduce_sum(vmul(nibf, pack_w, mask=m32), mask=m32),
                        I32,
                        mask=m32,
                    )
                    vstore_pack(
                        self.q_ub,
                        base // 2 + i * (VL // 2),
                        packed,
                        m_lo,
                        pack_mode=PackMode.B32_TO_B8,
                    )


# ---------------------------------------------------------------------------
# Fused kernel
# ---------------------------------------------------------------------------

class _Launch:
    """Invariant GM handles for one launch. Per-tile values stay in arguments."""


@jit
def _live_counts(p, t_live):
    """Tile and band counts for this launch. Static plans keep them constant."""
    if const_expr(p.t_dyn):
        n_tiles = (t_live + (p.base_m - 1)) // p.base_m
        n_work = n_tiles * p.n_head_blocks
        if const_expr(p.w_workspace):
            n_bands = (t_live + (p.rows_red - 1)) // p.rows_red
            n_bands_workspace = n_tiles * p.base_m // p.rows_red
        else:
            n_bands = 0
            n_bands_workspace = 0
        return n_tiles, n_work, n_bands, n_bands_workspace
    return p.n_tiles, p.n_tiles * p.n_head_blocks, p.n_bands, p.n_bands_workspace


@jit
def _w_prefix(gm, work):
    """Retire this core's W reduction before the Q head loop when W is separate."""
    p = gm.p
    cube_w = gm.cube_w
    tile = work // p.n_head_blocks
    h_block = work % p.n_head_blocks
    rows_full = tile_view(gm.w, (p.base_m, p.n_heads), (tile, 0)).shape[0]
    rows_l1 = ceil_div(rows_full, NZ_M_FRAC) * NZ_M_FRAC
    if const_expr(not (PROBE == "no_w" and p.w_separate)):
        w_slot = cube_w.l0c.produce()
    else:
        w_slot = None
    if const_expr(p.w_separate) and const_expr(PROBE != "no_w"):
        if const_expr(p.w_k_split):
            w_k_base = h_block * p.k_l1_tiles_w
        else:
            w_k_base = 0
        cube_w.gemm_full(tile, w_k_base, w_slot, rows_l1)
        if const_expr(p.w_workspace):
            cube_w.store_partial(tile, h_block, gm.n_tiles, rows_l1)
        else:
            cube_w.store(tile, rows_l1)
        w_slot = None
        if const_expr(p.w_workspace and p.reduce_at == "early" and PROBE != "no_reduce"):
            if const_expr(PROBE != "no_sync"):
                w_barrier()
            gm.vec.reduce_all(gm.workspace, gm.w, gm.n_bands, gm.n_bands_workspace)
    if const_expr(W_SCHED == "wpipe" and PROBE != "no_w" and not p.w_separate):
        cube_w.fill(tile, 0, 0)
    return w_slot


@jit
def _q_heads(gm, work, lut_ub, cos_ub, w_slot):
    """One core's Q heads, with the W K-slice interleaved when that schedule is on."""
    p = gm.p
    cube_q = gm.cube_q
    cube_w = gm.cube_w
    vec = gm.vec
    tile = work // p.n_head_blocks
    h_block = work % p.n_head_blocks
    rows_full = tile_view(gm.w, (p.base_m, p.n_heads), (tile, 0)).shape[0]
    rows_l1 = ceil_div(rows_full, NZ_M_FRAC) * NZ_M_FRAC
    rows = cv_valid_extent(
        tile_view(gm.descale_q, (p.base_m, p.groups), (tile, 0)),
        axis=0,
        part_id=gm.subblock,
    )
    sin_ub = gm.sin_ub
    qr_ready, dqr_ready = cube_q.wait_qr()
    qr_rows, dqr_rows = cube_q.rows_view(qr_ready, dqr_ready, rows_l1)
    for local_head in range(p.heads_per_block):
        head = h_block * p.heads_per_block + local_head
        if const_expr(PROBE != "no_w" and W_SCHED == "wfirst" and not p.w_separate):
            cube_w.gemm_head_slice(tile, local_head, 0, w_slot, rows_l1)
        cube_q.gemm_head(gm.wqb, gm.descale_wqb, head, qr_rows, dqr_rows)
        if const_expr(PROBE != "no_w" and W_SCHED == "wafter" and not p.w_separate):
            cube_w.gemm_head_slice(tile, local_head, 0, w_slot, rows_l1)
        elif const_expr(PROBE != "no_w" and W_SCHED == "wpipe" and not p.w_separate):
            if local_head + 1 < p.heads_per_block:
                cube_w.fill(tile, local_head + 1, 0)
            cube_w.drain(local_head, w_slot, rows_l1)
        cv_ready = cube_q.drain_to_vec(gm.cv_ub, rows_full, rows_l1)
        vec.epilogue(cv_ready, rows, lut_ub, cos_ub, sin_ub)
        vec.store(
            tile_view(gm.q, (p.base_m, p.d // 2), (tile, head)),
            tile_view(gm.descale_q, (p.base_m, p.groups), (tile, head)),
            rows,
        )
    _prefetch_qr(cube_q, gm.qr, gm.descale_qr, work, gm.n_work)
    if const_expr(PROBE != "no_w" and not p.w_separate):
        cube_w.store(tile, rows_l1)
    elif const_expr(PROBE == "no_w" and not p.w_separate):
        cube_w.l0c.consume()


@jit
def _arm(p, direct, views, t_live):
    """Stash the launch tensors. Called from the kernel so the entry stays short."""
    x, wqb, weight_w, workspace = direct
    (
        qr, q, descale_q, descale_qr, descale_wqb, lut, rope_sin, rope_cos, w,
        softmax_scale,
    ) = views
    cube_w = CubeW(p)
    cube_w.x_gm = x
    cube_w.ww_gm = weight_w
    cube_w.w_gm = w
    cube_w.workspace_gm = workspace
    cube_w.softmax_scale = softmax_scale
    gm = _Launch()
    gm.p = p
    gm.cube_q = CubeQ(p)
    gm.cube_w = cube_w
    gm.vec = VectorQ(p)
    gm.cv_ub = Channel(
        MemLoc.UB, (p.rows_vec, p.d), F32, depth=D_CV, kind=ChannelKind.CrossCore,
    )
    gm.subblock = get_subblock_id()
    gm.qr = qr
    gm.q = q
    gm.wqb = wqb
    gm.descale_qr = descale_qr
    gm.descale_wqb = descale_wqb
    gm.descale_q = descale_q
    gm.lut = lut
    gm.rope_sin = rope_sin
    gm.rope_cos = rope_cos
    gm.w = w
    gm.workspace = workspace
    n_tiles, n_work, n_bands, n_bands_workspace = _live_counts(p, t_live)
    gm.n_tiles = n_tiles
    gm.n_work = n_work
    gm.n_bands = n_bands
    gm.n_bands_workspace = n_bands_workspace
    gm.first_work = get_block_idx()
    return gm


@jit
def _one_tile(gm, work):
    """One (token tile, head group): stage qr, retire W, then the Q heads."""
    p = gm.p
    tile = work // p.n_head_blocks
    if const_expr(p.dr > 0):
        cos_tile = tile_view(gm.rope_cos, (p.base_m, p.dr), (tile, 0))
        sin_tile = tile_view(gm.rope_sin, (p.base_m, p.dr), (tile, 0))
    else:
        cos_tile = None
        sin_tile = None
    if const_expr(not p.prefetch):
        gm.cube_q.load_qr(gm.qr, gm.descale_qr, tile)
    elif work == gm.first_work:
        gm.cube_q.load_qr(gm.qr, gm.descale_qr, tile)
    lut_ub, cos_ub, sin_ub = gm.vec.load_tile_constants(gm.lut, cos_tile, sin_tile)
    gm.sin_ub = sin_ub
    h_block = work % p.n_head_blocks
    if const_expr(not p.w_leader_only) or h_block == 0:
        w_slot = _w_prefix(gm, work)
        _q_heads(gm, work, lut_ub, cos_ub, w_slot)


# Device main loop. Each work item is one (token tile, head group).
# Cores stride by the grid. The split-K W reduction runs after every
# partial has been stored.
@kernel
class IndexerPrologueQwKernel:
    def __init__(self, t, dim, q_lora, n_heads, d, dr, cube_cores, t_dyn=False):
        self.p = Plan(t, dim, q_lora, n_heads, d, dr, cube_cores, t_dyn)

    def __call__(
        self,
        x: Tensor,
        qr: Tensor,
        wqb: Tensor,
        weight_w: Tensor,
        descale_qr: Tensor,
        descale_wqb: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        lut: Tensor,
        q: Tensor,
        descale_q: Tensor,
        w: Tensor,
        workspace: Tensor,
        t_live: I64,
        softmax_scale: F32,
    ):
        gm = _arm(
            self.p,
            (x, wqb, weight_w, workspace),
            (qr, q, descale_q, descale_qr, descale_wqb, lut, rope_sin, rope_cos, w, softmax_scale),
            t_live,
        )
        for work in range(gm.first_work, gm.n_work, get_block_num()):
            _one_tile(gm, work)
        if const_expr(self.p.w_workspace and self.p.reduce_at == "tail" and PROBE != "no_reduce"):
            if const_expr(PROBE != "no_sync"):
                w_barrier()
            gm.vec.reduce_all(workspace, gm.w, gm.n_bands, gm.n_bands_workspace)



# Compile and launch wrapper around one Plan. Callers use indexer_prologue_qw().
class IndexerPrologueQw:
    """Host wrapper: validates, plans tiles, launches the fused kernel."""

    def __init__(self, t, dim, q_lora, n_heads, d, dr, cube_cores=None,
                 t_dyn=False, tile_plan=None):
        # ``tile_plan`` lets ``_plan_for`` hand over the plan it already built to read
        # the shape key off, instead of building an identical second one.
        self.p = tile_plan or Plan(t, dim, q_lora, n_heads, d, dr, cube_cores, t_dyn)

    # Recover the launch grid from the extra workspace rows, then run the kernel.
    # Those rows are a shape marker, not values the kernel reads. The grid is
    # the number of launched cores, not the number of work items.
    @host
    def run(
        self,
        x_gm: Tensor,
        qr_gm: Tensor,
        wqb_gm: Tensor,
        ww_gm: Tensor,
        descale_qr_gm: Tensor,
        descale_wqb_gm: Tensor,
        rope_sin_gm: Tensor,
        rope_cos_gm: Tensor,
        lut_gm: Tensor,
        q_gm: Tensor,
        descale_q_gm: Tensor,
        w_gm: Tensor,
        workspace_gm: Tensor,
        t_live: I64,
        softmax_scale: F32,
    ):
        p = self.p
        # The grid, read back out of the marker rows the host put on the end of
        # ``workspace``.  Symbolic, so it is not in the binary key -- a host integer here
        # is what used to make the core count a specialisation, since the block
        # dim rides along in the verified IR that key is taken over.
        #
        # It has to be the core count and not the work count: one block per work
        # item deadlocks as soon as the work outruns the chip, because the surplus
        # blocks are scheduled in a second wave while the W reduction rendezvous
        # waits on blocks that have not launched.  T=1024 is 8 tiles x 16 head
        # blocks = 128 blocks on a 64-AIV part, and it hangs to an aicore timeout.
        if const_expr(p.w_workspace):
            tiles = (x_gm.shape[0] + (p.base_m - 1)) // p.base_m
            grid = workspace_gm.shape[0] - tiles * p.n_head_blocks * p.base_m
        else:
            grid = workspace_gm.shape[0]
        IndexerPrologueQwKernel(
            p.t, p.dim, p.q_lora, p.n_heads, p.d, p.dr, p.cube_cores, p.t_dyn
        )[grid](
            x_gm, qr_gm, wqb_gm, ww_gm, descale_qr_gm, descale_wqb_gm,
            rope_sin_gm, rope_cos_gm, lut_gm, q_gm, descale_q_gm, w_gm,
            workspace_gm, t_live, softmax_scale,
        )


# ---------------------------------------------------------------------------
# NPU-only public host (FRACTAL_NZ weights; no CPU golden fallback)
# ---------------------------------------------------------------------------

MX_GROUP_SIZE = 32
# torch_npu.Format.FRACTAL_NZ; spelled out so importing this module stays free
# of torch_npu until the NPU path actually runs.
FRACTAL_NZ = 29
# T values the packaged export compiles a binary for.  The host entry point
# below plans any T; this list only bounds the AOT build.
#
# The default is the deployment set: T is batch * seq, so decode contributes
# batch x {1 accepted token, 1 + 5 MTP draft tokens} and prefill contributes
# batch x chunk length with no MTP.  Kept as a literal rather than derived from
# the verification shape module so that ``net/ops`` stays importable on its own;
# ``net/verification/test/indexer_prologue_qw_shapes.py`` generates it.
DEPLOYMENT_T = (
    1, 4, 6, 8, 12, 16, 24, 32, 48, 72, 96, 192,
    1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
)
def _export_t_values() -> tuple[int, ...]:
    raw = os.environ.get("IPQW_EXPORT_T", "")
    if not raw:
        return DEPLOYMENT_T
    values = []
    for part in raw.split(","):
        if part:
            values.append(int(part))
    return tuple(values)


EXPORT_T = _export_t_values()


def split_t_min(cube_cores: int | None = None) -> int:
    """Return BASE_M times the core count: rows that fill one T-tile per core.

    This helper does not choose the template. Dispatch is
    Plan._plan_head_split(). The default switches to split-T at the 25th
    token tile, which is T >= 3073. Do not treat this return value as that
    boundary.
    """
    return BASE_M * resolve_cube_cores(cube_cores)


# One entry per distinct compile-time shape (``Plan.shape_key``), not per T.  T
# is a runtime value in every entry, so a T nobody listed costs nothing; what
# picks an entry is the head split and the launch grid, which are buffer shapes
# and a grid size rather than counts.  See ``Plan.SHAPE_KEY``.
_PLAN_CACHE: dict[tuple, "IndexerPrologueQw"] = {}


def plan_buckets(geo, t_max=None):
    """Enumerate compile templates for dynamic T. Each item is (plan, t_lo, t_hi).

    The plan depends on the number of 128-row tiles. Tiles are walked and
    collapsed by shape_key. The default is two buckets: split-K on [1, 3072]
    and split-T on [3073, 262144]. After tuning flags change, trust the
    enumeration. Each range is the envelope of T that share one key.
    """
    dim = geo.dim
    q_lora = geo.q_lora
    n_heads = geo.n_heads
    d = geo.d
    dr = geo.dr
    cube_cores = resolve_cube_cores(geo.cube_cores)
    t_max = t_max or T_MAX
    seen: dict[tuple, list] = {}
    for tiles in range(1, ceil_div(t_max, BASE_M) + 1):
        tile_plan = Plan(
            tiles * BASE_M, dim, q_lora, n_heads, d, dr, cube_cores, t_dyn=True
        )
        entry = seen.setdefault(tile_plan.shape_key(), [tile_plan, tiles, tiles])
        entry[1] = min(entry[1], tiles)
        entry[2] = max(entry[2], tiles)
    out = []
    for tile_plan, lo, hi in seen.values():
        out.append((tile_plan, (lo - 1) * BASE_M + 1, min(hi * BASE_M, t_max)))
    out.sort(key=lambda item: item[1])
    return out


def _plan_for(t, geo):
    """Pick the plan for this T and reuse the IndexerPrologueQw for that layout.

    With T_DYN=1 the plan is dynamic and cached by shape_key, so every T on
    one template shares the object. The live row count is passed separately
    as t_live. T_DYN=0 is the tuning path: the cache key includes T and each
    T gets a static plan.
    """
    dim = geo.dim
    q_lora = geo.q_lora
    n_heads = geo.n_heads
    d = geo.d
    dr = geo.dr
    cube_cores = resolve_cube_cores(geo.cube_cores)
    if not T_DYN:
        # A/B only: one binary per T, with base_m sized to T and the grid cut
        # to the work count. This is what the operator did before T was a
        # runtime value, and it is how the cost of making it one gets priced.
        key = (t, dim, q_lora, n_heads, d, dr, cube_cores)
        op = _PLAN_CACHE.get(key)
        if op is None:
            op = IndexerPrologueQw(t, dim, q_lora, n_heads, d, dr, cube_cores)
            _PLAN_CACHE[key] = op
        return op
    probe = Plan(t, dim, q_lora, n_heads, d, dr, cube_cores, t_dyn=True)
    key = probe.shape_key()
    op = _PLAN_CACHE.get(key)
    if op is None:
        op = IndexerPrologueQw(
            t, dim, q_lora, n_heads, d, dr, cube_cores, t_dyn=True, tile_plan=probe
        )
        _PLAN_CACHE[key] = op
    return op


# Compile contract. TensorSpec fixes shape, dtype, and NZ layout. Dim marks
# an axis that changes at runtime. The public entry and the export share this
# spec so a call hits the binary that was already compiled.
def _arg_specs(geo, rows_live, workspace_rows):
    """The spec ``op.run`` is compiled against, for one bucket.

    Shared by the packaged export and the host entry deliberately. A native
    package is looked up by the digest of this spec, so a host that compiled
    against concrete row counts could never find the binary the packager built
    against Dims: the lookup misses on the contract digest and the operator
    silently re-traces on every call instead of launching the shipped .so.

    x, the two NZ weights and workspace stay individual tensors. Everything
    else is one rank-2 uint8 list; the kernel views fp32 and int32 entries back.
    """
    dim = geo.dim
    q_lora = geo.q_lora
    n_heads = geo.n_heads
    d = geo.d
    dr = geo.dr
    groups = ceil_div(d, MX_GROUP_SIZE)
    specs = []
    specs.append(TensorSpec((rows_live, dim), dtypes.bfloat16))
    specs.append(TensorSpec((rows_live, q_lora), U8))
    specs.append(TensorSpec((n_heads * d, q_lora), U8, storage_format="nz"))
    specs.append(TensorSpec((n_heads, dim), dtypes.bfloat16, storage_format="nz"))
    specs.append(TensorSpec((rows_live, ceil_div(q_lora, 64), 2), U8))
    specs.append(TensorSpec((n_heads * d, ceil_div(q_lora, 64), 2), U8))
    specs.append(TensorSpec((rows_live, dr), F32))
    specs.append(TensorSpec((rows_live, dr), F32))
    specs.append(TensorSpec((LUT_LEN,), I32))
    specs.append(TensorSpec((rows_live, n_heads * (d // 2)), U8))
    specs.append(TensorSpec((rows_live, n_heads * groups), U8))
    specs.append(TensorSpec((rows_live, n_heads), F32))
    specs.append(TensorSpec((workspace_rows, n_heads), F32))
    specs.append(I64)
    specs.append(F32)
    return specs


# Dynamic ranges for live T and workspace rows of one template. No tensor is allocated.
def _bucket_dims(index, tile_plan, t_lo, t_hi):
    """The two T-derived ``Dim``s of one bucket: live rows and workspace rows.

    ``index`` is the bucket's position in ``plan_buckets``, and it goes into the
    Dim names.  It is part of the contract digest, so the host has to number the
    buckets exactly the way the export does or the lookup misses.
    """
    tiles_lo = ceil_div(t_lo, BASE_M)
    tiles_hi = ceil_div(t_hi, BASE_M)
    # ``workspace`` is one band of rows per K slice per padded tile, so it scales with
    # the tile count too and cannot be a fixed extent either, plus the grid
    # marker rows on the end (see MAX_CORES).  The marker is why the bounds are
    # slack and why there is no ``multiple_of``: they have to admit every core
    # count from 1 to MAX_CORES, since a bound is part of the contract and a
    # bound spelled in terms of the core count would specialise on it again.
    workspace_rows = (
        Dim(
            f"WS{index}",
            min=tile_plan.n_head_blocks * tiles_lo * BASE_M + 1,
            max=tile_plan.n_head_blocks * tiles_hi * BASE_M + MAX_CORES,
        )
        if tile_plan.w_workspace
        else Dim(f"WS{index}", min=1, max=MAX_CORES)
    )
    return (
        Dim(f"T{index}", min=t_lo, max=t_hi),
        workspace_rows,
    )


# ``plan_buckets`` builds a Plan per tile count over the whole T range, so it is
# far too expensive to redo per call; the host needs it to find which bucket owns
# this T.
_BUCKET_CACHE: dict[tuple, list] = {}
# One open ProviderCallable per bucket.  Compiled against the bucket's Dims, so
# it hits the packaged binary when one is installed and is traced once when not.
_COMPILED_CACHE: dict[tuple, object] = {}


def _buckets_cached(geo):
    key = (geo.dim, geo.q_lora, geo.n_heads, geo.d, geo.dr, geo.cube_cores)
    buckets = _BUCKET_CACHE.get(key)
    if buckets is None:
        buckets = plan_buckets(geo)
        _BUCKET_CACHE[key] = buckets
    return buckets


def _run_for(op, geo):
    """Return the compiled callable for this template and cache it.

    The dynamic path compiles op.run against the bucket Dim and TensorSpec,
    the same contract the packaged export uses. Passing a concrete row count
    can miss that artifact. The static tuning path, or a launcher already
    substituted for op.run, returns op.run unchanged.
    """
    if not T_DYN:
        # A/B path: there are no buckets, the plan is specialised on T.
        return op.run
    if not hasattr(op.run, "_host_function"):
        # The export harness substitutes an already-compiled launcher for
        # ``op.run`` so it can watch which artifact each case lands on.  There is
        # nothing left to compile against in that case, and re-compiling here
        # would defeat the point of the substitution.
        return op.run
    want = op.p.shape_key()
    key = (want, geo.dim, geo.q_lora, geo.n_heads, geo.d, geo.dr, geo.cube_cores)
    fn = _COMPILED_CACHE.get(key)
    if fn is None:
        matched = None
        for index, (tile_plan, t_lo, t_hi) in enumerate(_buckets_cached(geo)):
            if tile_plan.shape_key() == want:
                matched = (index, tile_plan, t_lo, t_hi)
                break
        if matched is None:
            # No bucket owns this shape: let the @host path trace it rather than
            # refuse to run.  Reachable only if the plan policy and the bucket
            # enumeration disagree, which is a bug, not a caller error.
            return op.run
        index, tile_plan, t_lo, t_hi = matched
        fn = dsl_compile(
            op.run,
            *_arg_specs(geo, *_bucket_dims(index, tile_plan, t_lo, t_hi)),
        )
        _COMPILED_CACHE[key] = fn
    return fn


def _check_workspace(op, t, partial_rows) -> None:
    """Reject a workspace sized from another T in the same plan bucket.

    One Plan serves every T in its bucket and was built at whichever T arrived
    first.  Partials for a larger T written into a buffer allocated for a
    smaller one land outside it.
    """
    tile_plan = op.p
    if not tile_plan.w_workspace:
        return
    need = tile_plan.n_head_blocks * ceil_div(t, tile_plan.base_m) * tile_plan.base_m
    if partial_rows < need:
        raise RuntimeError(
            f"Workspace too small for T={t}: {partial_rows} partial rows "
            f"allocated, {need} needed ({tile_plan.n_head_blocks} K slices x "
            f"{ceil_div(t, tile_plan.base_m)} tiles x {tile_plan.base_m} rows)."
        )


_LUT_CACHE: dict[torch.device, torch.Tensor] = {}


def _lut_cache(device: torch.device) -> torch.Tensor:
    """E2M1 rounding table, uploaded once per device."""
    cached = _LUT_CACHE.get(device)
    if cached is None:
        cached = build_e2m1_lut().to(device)
        _LUT_CACHE[device] = cached
    return cached


def _check_storage(bound):
    """Device, dtype, rank and NZ checks. Metadata only."""
    x = bound["x"]
    qr = bound["qr"]
    wqb = bound["wqb"]
    ww = bound["ww"]
    descale_qr = bound["descale_qr"]
    descale_wqb = bound["descale_wqb"]
    rope_sin = bound["rope_sin"]
    rope_cos = bound["rope_cos"]
    inputs = (x, qr, wqb, ww, descale_qr, descale_wqb, rope_sin, rope_cos)
    if not all(tensor.device.type == "npu" for tensor in inputs):
        raise ValueError("indexer_prologue_qw is NPU-only; all tensors must be on npu")
    if any(tensor.dim() != INPUT_RANK for tensor in (x, qr, wqb, ww)):
        raise ValueError("x, qr, wqb, ww must be rank-2")
    if x.shape[0] < qr.shape[0]:
        raise ValueError(f"X has fewer rows than T: x={x.shape[0]} qr={qr.shape[0]}")
    if rope_sin.shape != rope_cos.shape:
        raise ValueError("rope_sin and rope_cos shapes must match")
    if bound["softmax_scale"] is None:
        raise ValueError("softmax_scale is required")
    _check_dtype(x, torch.bfloat16, "x")
    _check_dtype(ww, torch.bfloat16, "ww")
    _check_dtype(rope_sin, torch.float32, "rope_sin")
    _check_dtype(rope_cos, torch.float32, "rope_cos")
    if torch_npu is None:
        raise ImportError("torch_npu is required to check FRACTAL_NZ weights")
    _check_nz(wqb, "wqb")
    _check_nz(ww, "ww")
    _check_byte(qr, "qr")
    _check_byte(wqb, "wqb")
    _check_byte(descale_qr, "descale_qr")
    _check_byte(descale_wqb, "descale_wqb")


def _check_dtype(tensor, want, name):
    if tensor.dtype != want:
        raise ValueError(f"{name} must be {want}, got {tensor.dtype}")


def _check_nz(tensor, name):
    if torch_npu.get_npu_format(tensor) != FRACTAL_NZ:
        raise ValueError(
            f"{name} must be in FRACTAL_NZ; convert it once at load time "
            f"with to_nz({name}), not per step"
        )


def _check_byte(tensor, name):
    if tensor.element_size() != 1:
        raise ValueError(f"{name} must be a one-byte fp8/e8m0 dtype, got {tensor.dtype}")


def _check_prologue(bound):
    """Reject a host call the kernel cannot run."""
    x = bound["x"]
    qr = bound["qr"]
    wqb = bound["wqb"]
    ww = bound["ww"]
    rope_sin = bound["rope_sin"]
    _check_storage(bound)

    t, q_lora = qr.shape
    n_heads, dim = ww.shape
    if x.shape[1] != dim:
        raise ValueError(f"X dim {x.shape[1]} does not match ww dim {dim}")
    if wqb.shape[1] != q_lora:
        raise ValueError(f"wqb K {wqb.shape[1]} does not match qr q_lora {q_lora}")
    if n_heads == 0 or wqb.shape[0] % n_heads != 0:
        raise ValueError(
            f"wqb rows {wqb.shape[0]} must be a positive multiple of N={n_heads}"
        )
    d = wqb.shape[0] // n_heads
    if rope_sin.shape[0] != t:
        raise ValueError(
            f"RoPE tables have {rope_sin.shape[0]} rows, expected T={t}"
        )
    dr = rope_sin.shape[-1]
    frozen = (FROZEN_DIM, FROZEN_Q_LORA, FROZEN_N_HEADS, FROZEN_D, FROZEN_DR)
    got = (dim, q_lora, n_heads, d, dr)
    if got != frozen:
        raise ValueError(
            "indexer_prologue_qw only supports "
            f"dim={FROZEN_DIM}, q_lora={FROZEN_Q_LORA}, N={FROZEN_N_HEADS}, "
            f"D={FROZEN_D}, Dr={FROZEN_DR}; got dim={dim}, q_lora={q_lora}, "
            f"N={n_heads}, D={d}, Dr={dr}"
        )
    bound["t"] = t
    bound["q_lora"] = q_lora
    bound["n_heads"] = n_heads
    bound["dim"] = dim
    bound["d"] = d
    bound["dr"] = dr
    return bound



def _launch_prologue(bound):
    """Allocate outputs and launch the compiled kernel."""
    x = bound["x"]
    qr = bound["qr"]
    descale_qr = bound["descale_qr"]
    t = bound["t"]
    q_lora = bound["q_lora"]
    n_heads = bound["n_heads"]
    dim = bound["dim"]
    d = bound["d"]
    dr = bound["dr"]
    groups = ceil_div(d, MX_GROUP_SIZE)
    dev = x.device

    cube_cores = resolve_cube_cores()
    geo = Geometry(dim, q_lora, n_heads, d, dr, cube_cores)
    op = _plan_for(t, geo)
    lut = _lut_cache(dev)

    # Every T-axis input goes to the kernel at exactly ``t`` rows.  The short
    # last tile is a row-fractal problem, and it is solved where the fractal is:
    # the L1 staging pads it (see ``rows_l1`` in the kernel).  This operator
    # allocates no GM for an input.
    #
    # ``x`` is the one input a caller may hand over taller than T -- a decode
    # caller's hidden-state buffer is allocated at its own tile height -- and a
    # view is free, so the extra rows are dropped here rather than described in
    # the ABI.
    n_tiles = ceil_div(t, op.p.base_m)
    if x.shape[0] != t:
        x = x[:t]
        bound["x"] = x

    qr_in = qr.view(torch.uint8)
    dqr_in = descale_qr.reshape(t, -1, 2).view(torch.uint8)

    q_flat = torch.empty((t, n_heads * (d // 2)), dtype=torch.uint8, device=dev)
    dsc_flat = torch.empty((t, n_heads * groups), dtype=torch.uint8, device=dev)
    w_out = torch.empty((t, n_heads), dtype=torch.float32, device=dev)
    # One padded (T_tile, N) fp32 tile per K slice.  ``empty`` is enough: every
    # row is written by the core that owns it before the reduction reads it, so
    # there is nothing to initialise and no Zeros op in a captured graph.  One
    # workspace is needed.
    # Sized from this call's tile count, not the plan's: the plan was built at
    # whichever T first landed on this shape key.  Same formula as _plan_reduce.
    # The ``used_cores`` rows on the end are the grid marker the kernel reads the
    # launch width back out of; see MAX_CORES for why the grid travels this way.
    partial_rows = op.p.n_head_blocks * n_tiles * op.p.base_m if op.p.w_workspace else 0
    workspace_rows = partial_rows + op.p.used_cores
    workspace = torch.empty((workspace_rows, n_heads), dtype=torch.float32, device=dev)
    _check_workspace(op, t, partial_rows)

    return _pack_and_run(
        bound, op, geo, workspace, (qr_in, dqr_in, q_flat, dsc_flat, w_out, lut),
    )


def _pack_and_run(bound, op, geo, workspace, outs):
    """Launch the compiled kernel and copy into optional destinations."""
    qr_in, dqr_in, q_flat, dsc_flat, w_out, lut = outs
    x = bound["x"]
    wqb = bound["wqb"]
    ww = bound["ww"]
    rope_sin = bound["rope_sin"]
    rope_cos = bound["rope_cos"]
    t = bound["t"]
    n_heads = bound["n_heads"]
    d = bound["d"]
    dwqb = bound["descale_wqb"].reshape(n_heads * d, -1, 2).view(torch.uint8)
    _run_for(op, geo)(
        x.view(torch.bfloat16),
        qr_in,
        wqb.view(torch.uint8),
        ww.view(torch.bfloat16),
        dqr_in,
        dwqb,
        rope_sin,
        rope_cos,
        lut,
        q_flat,
        dsc_flat,
        w_out,
        workspace,
        t,
        float(bound["softmax_scale"]),
    )
    q_out = q_flat.view(t, n_heads, d // 2)
    dsc_out = dsc_flat.view(t, n_heads, ceil_div(d, MX_K_ALIGN), MX_PAIR)
    q = bound.get("q")
    descale_q = bound.get("descale_q")
    w = bound.get("w")
    if q is not None:
        q.copy_(q_out)
        q_out = q
    if descale_q is not None:
        descale_q.copy_(dsc_out)
        dsc_out = descale_q
    if w is not None:
        w.copy_(w_out)
        w_out = w
    return q_out, dsc_out, w_out


def indexer_prologue_qw(*tensors, **options):
    """Fused indexer prologue: MXFP8 Q GEMM + RoPE + MXFP4 quant, and BF16 W GEMM.

    Positional order is x, qr, wqb, ww, descale_qr, descale_wqb, rope_sin,
    rope_cos. The same names are accepted as keywords, plus softmax_scale and
    optional q, descale_q, w destinations.

    NPU only. The two weights ``wqb`` and ``ww`` must already be in FRACTAL_NZ
    (see :func:`to_nz`); ``x``, ``qr``, both descales and the RoPE tables are ND.

    ``T`` comes from ``qr``.  ``x`` may have more rows than that -- a caller
    whose hidden-state buffer is allocated at its own tile height can hand it
    over as it stands -- and the extra rows are dropped by a view, not read.

    No GM is allocated for any input, at any T.  A short last row tile is
    handled where the row fractal is, in L1; see ``CubeQ.rows_view``.

    Returns ``(q, descale_q, w)``. ``q`` is packed uint8 e2m1 with shape
    ``(T, N, D/2)``; ``descale_q`` is e8m0; ``w`` is fp32.
    """
    names = (
        "x", "qr", "wqb", "ww", "descale_qr", "descale_wqb", "rope_sin", "rope_cos",
    )
    if len(tensors) > len(names):
        raise TypeError("indexer_prologue_qw takes the eight input tensors")
    bound = dict(zip(names, tensors))
    bound.update(options)
    return _launch_prologue(_check_prologue(bound))




@export("indexer_prologue_qw")
def export_indexer_prologue_qw():
    """Package one artifact per template, not one artifact per T.

    plan_buckets() lists two templates by default, and each uses Dim for its
    T range. IPQW_EXPORT_T selects templates; it does not turn dynamic T into
    a static T. shape_key does not include the core count. The launch grid
    arrives through the workspace shape. resolve_cube_cores() only builds the
    host plan; it does not mean each core count needs its own template.
    """
    # Frozen geometry: hidden dim, q_lora, head count, head dim, RoPE length.
    geo = Geometry(
        FROZEN_DIM, FROZEN_Q_LORA, FROZEN_N_HEADS, FROZEN_D, FROZEN_DR,
        resolve_cube_cores(),
    )
    buckets = plan_buckets(geo)
    wanted = set()
    for t_value in EXPORT_T:
        wanted.add(_plan_for(t_value, geo).p.shape_key())
    for index, (tile_plan, t_lo, t_hi) in enumerate(buckets):
        if tile_plan.shape_key() not in wanted:
            continue
        op = IndexerPrologueQw(
            tile_plan.t, geo.dim, geo.q_lora, geo.n_heads, geo.d, geo.dr,
            geo.cube_cores, t_dyn=True, tile_plan=tile_plan,
        )
        fn = dsl_compile(
            op.run,
            *_arg_specs(geo, *_bucket_dims(index, tile_plan, t_lo, t_hi)),
        )
        fn.close()
