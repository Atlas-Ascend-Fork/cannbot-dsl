# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Q/W/K前处理：固定投影切分、模型RoPE、nearest-even量化和双cache发布。"""

from __future__ import annotations
from cannbotdsl import from_tensor

import os
from dataclasses import dataclass

import torch
from cannbotdsl import (
    MemLoc,
    RingBuffer,
    Tensor,
    TensorSpec,
    const_expr,
    cv_valid_extent,
    dtypes,
    get_mem_size,
)
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import range_constexpr
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import (
    get_block_idx,
    get_block_num,
    get_subblock_dim,
    get_subblock_id,
)
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    create_mask,
    full_mask,
    vabs,
    vadd,
    vadds,
    varange,
    vbitwise_and,
    vbitwise_or,
    vcast,
    vdeinterleave,
    vdups,
    vgather,
    vgather_reg,
    vgts,
    vinterleave,
    vload,
    vmaxs,
    vmem_bar,
    vmul,
    vne,
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
from cannbotdsl.ops.sync import vec_sync_all
from cannbotdsl.tensor import make_layout, make_pointer, make_tensor, tile_slice


def reinterpret(tensor, dtype=None, shape=None, stride=None, offset=0, **kwargs):
    """DSL 0.8.0 takes a Layout instead of shape/stride; keep the sample's form."""
    if shape is not None:
        kwargs["layout"] = (
            make_layout(shape) if stride is None else make_layout(shape, stride=stride)
        )
    return from_tensor(tensor, dtype=dtype, offset=offset, **kwargs)


# Q/W投影：专用包装器不覆盖原生K接口。
try:
    import torch_npu
except ImportError:  # CPU-only tests import this module without an NPU stack.
    torch_npu = None

Float8E4M3FN = dtypes.float8_e4m3fn

Float8E8M0 = dtypes.float8_e8m0


def _select_slot(value, *, write):
    if isinstance(value, RingBuffer):
        return value.next_write() if write else value.next_read()
    return value


def tile_view(tensor, tiler, coord):
    return tile_slice(tensor, tiler, coord)


def make_bounded_tiler(shape, capacity=None, alignment=None):
    # DSL 0.8.0 reinterpret() takes a Layout, so the bounded "tiler" degrades to
    # the plain shape tuple; capacity/alignment stay accepted for call parity.
    del capacity, alignment
    return shape


def local_slice(tensor, tiler, stride=None, offset=0):
    return from_tensor(tensor, layout=make_layout(tiler, stride=stride), offset=offset)


def qw_mem_copy(dst, src, *args, **kwargs):
    return mem_copy(
        _select_slot(dst, write=True),
        _select_slot(src, write=False),
        *args,
        **kwargs,
    )


def qw_matmul(dst, lhs, rhs, *args, **kwargs):
    return matmul(
        _select_slot(dst, write=True),
        _select_slot(lhs, write=False),
        _select_slot(rhs, write=False),
        *args,
        **kwargs,
    )


try:
    from device_properties import get_device_properties
except ImportError:  # packaged net/ops wheel has no samples/ on sys.path

    def get_device_properties():
        return torch.npu.get_device_properties(0)


I32 = dtypes.int32

I64 = dtypes.int64

U32 = dtypes.uint32

F32 = dtypes.float32

U8 = dtypes.uint8

VL = 64  # fp32 lanes in one 256-byte raw vector register


MX_GROUP = 32  # elements per E8M0 scale

MX_K_ALIGN = 64  # K elements per public paired-scale group

MX_PAIR = 2

FP32_BYTES = 4

INT32_BYTES = 4

NIBBLES_PER_BYTE = 2  # two E2M1 values packed into one uint8

COS_SIN_COPIES = 2

INPUT_RANK = 2

FROZEN_DIM = 5120

FROZEN_Q_LORA = 1280

FROZEN_N_HEADS = 32

FROZEN_D = 128

FROZEN_DR = 64

E8M0_BIAS = 127


FP32_EXP_SHIFT = 23

FP32_MANTISSA_MASK = 0x7FFFFF

FP32_HALF_MANTISSA = 0x400000  # significand exactly 1.5

FP32_MAG_MASK = 0x7FFFFFFF


def _depth(name: str, default: int) -> int:
    return int(os.environ.get(f"IPQW_{name}_DEPTH", default))


D_CV = _depth("CV", 2)

D_QUB = _depth("QUB", 2)

D_WQB = _depth("WQB", 2)

D_WQB_SINGLE_HEAD = _depth("WQB_SINGLE_HEAD", 5)

CUBE_BUF_IDS = 32

FIXED_CUBE_SLOTS = 16

D_L0C_Q = _depth("L0C_Q", 2)

D_L0AB_Q = _depth("L0AB_Q", 2)

D_L0B_Q = _depth("L0B_Q", D_L0AB_Q)

D_X = _depth("X", 2)

D_L0AB_W = _depth("L0AB_W", 1)


PROBE = os.environ.get("IPQW_PROBE", "")

W_REDUCE_AT = os.environ.get("IPQW_W_REDUCE_AT", "tail")


L2_Q = int(os.environ["IPQW_L2_Q"]) if "IPQW_L2_Q" in os.environ else None

L2_W = int(os.environ.get("IPQW_L2_W", 1))

PREFETCH = int(os.environ["IPQW_PREFETCH"]) if "IPQW_PREFETCH" in os.environ else None

UNIT_FLAG = int(os.environ.get("IPQW_UNIT_FLAG", 0))

HEAD_SPLIT = int(os.environ.get("IPQW_HEAD_SPLIT", 1))


W_K_SPLIT = int(os.environ.get("IPQW_W_K_SPLIT", 1))

GRID_FULL = int(os.environ.get("IPQW_GRID_FULL", 1))

SPLIT_T_TILES = int(os.environ.get("IPQW_SPLIT_T_TILES", 25))

HEAD_BLOCKS_K = int(os.environ.get("IPQW_HEAD_BLOCKS_K", 16))

ROWS_RED_K = int(os.environ.get("IPQW_ROWS_RED_K", 2))


BASE_M = int(os.environ.get("IPQW_BASE_M", 128))

T_MIN = 1

T_MAX = 256 * 1024

NZ_M_FRAC = 16

NZ_C0_1B = 32

NZ_C0_2B = 16

# On-chip capacities come from the DSL per-arch table, so the budget check
# follows the arch the kernel is compiled for instead of a hardcoded 950
# value.  ``ub`` is already net of the runtime reserve, so it is used as-is.
L1_BYTES = get_mem_size("l1")

# L0A and L0B are equal on every supported part; both budgets use this value.
L0AB_BYTES = get_mem_size("l0a")

L0C_BYTES = get_mem_size("l0c")

UB_BYTES = get_mem_size("ub")


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
        cube_cores = int(forced) if forced else get_device_properties().cube_core_num
    cube_cores = int(cube_cores)
    if cube_cores <= 0:
        raise ValueError(f"cube_cores must be positive, got {cube_cores}")
    return cube_cores


def scale_l1_len(k: int) -> int:
    """Contiguous E8M0 length in L1 for a K extent (paired public layout)."""
    return ceil_div(k, MX_K_ALIGN) * MX_PAIR


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
            BASE_M if (self.t_dyn or t >= BASE_M) else max(16, ceil_div(t, 16) * 16)
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
                f"head block count {forced} does not divide n_heads={self.n_heads}"
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
        "base_m",  # height of every L1/L0/UB tile
        "rows_vec",  # vector-side tile height
        "n_head_blocks",  # head loop trip count, k_block, band UB width
        "heads_per_block",
        "rows_red",  # band UB height
        "red_regs",  # register unroll in the band sum
        # ``used_cores`` is deliberately absent: the grid is a symbolic
        # expression over the padded row count now, not a constant, so it is
        # neither a specialisation dimension nor a route for the core count.
        "w_workspace",  # whether the workspace path is compiled in at all
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
            if (not self.t_dyn and self.n_tiles * self.n_head_blocks == self.used_cores)
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
        q_l0a = (
            D_L0AB_Q
            * self.base_m
            * (
                int(os.environ.get("IPQW_BASE_K_Q", 0))
                or min(128, ceil_div(self.q_lora, MX_K_ALIGN) * MX_K_ALIGN)
            )
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
            # Dr == 64 keeps rotate-half inside one register, so pass 1 can read
            # the pre-RoPE partner lanes from the register it is about to
            # overwrite.  Dr > 64 would need every rope register live at once
            # before any write-back; not implemented, and Dr=64 is the frozen
            # case (and what MLA uses).
            raise NotImplementedError(
                f"rotate-half is implemented for Dr == {VL} only, got Dr={self.dr}"
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


@dataclass(frozen=True)
class Geometry:
    """Frozen axes plus the optional cube-core override. T is not part of it."""

    dim: int
    q_lora: int
    n_heads: int
    d: int
    dr: int
    cube_cores: int | None = None


class _SingleSlot:
    """Depth-1 stand-in for ``RingBuffer`` over an explicit ZN view."""

    def __init__(self, tensor):
        self._tensor = tensor

    def next_write(self):
        return self._tensor

    def next_read(self):
        return self._tensor


class CubeQ:
    """MXFP8 ``qr @ wqb^T`` for one (T tile, head); owns its L1/L0 staging."""

    def __init__(self, p: Plan):
        self.p = p
        self.qr_l1 = RingBuffer(
            MemLoc.L1,
            Float8E4M3FN,
            make_layout((p.base_m, p.qr_k_l1)),
            depth=1,
            data_format="nz",
        )
        # DSL 0.8.0 Buffer only accepts data_format 'nd'/'nz'; the ZN scale
        # staging is an ordinary byte allocation rebound to a ZN view.
        self.scale_qr_l1 = _SingleSlot(
            from_tensor(
                Buffer(
                    MemLoc.L1,
                    dtypes.uint8,
                    make_layout((p.base_m * p.scale_k_len_q,)),
                    data_format="nd",
                ),
                dtype=Float8E8M0,
                layout=make_layout((p.base_m, p.scale_k_len_q)),
                data_format="zn",
            )
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
            RingBuffer(
                MemLoc.L1,
                U8,
                make_layout((p.base_n_q, p.k_l1_q)),
                depth=1,
                data_format="nz",
            )
            for _ in range(p.n_buf_wqb)  # == k_l1_tiles_q
        ]
        # No alias on the scales, so one rotating channel is fine here.
        self.scale_wqb_l1 = RingBuffer(
            MemLoc.L1,
            Float8E8M0,
            make_layout((p.scale_k_l1_len_q, p.base_n_q)),
            depth=p.n_buf_wqb,
            data_format="nz",
        )
        self.l0a = RingBuffer(
            MemLoc.L0A,
            Float8E4M3FN,
            make_layout((p.base_m, p.base_k_q)),
            depth=D_L0AB_Q,
        )
        self.l0b = RingBuffer(
            MemLoc.L0B,
            Float8E4M3FN,
            make_layout((p.base_n_q, p.base_k_q)),
            depth=D_L0B_Q,
        )
        self.l0c = RingBuffer(
            MemLoc.L0C, F32, make_layout((p.base_m, p.base_n_q)), depth=D_L0C_Q
        )

        # ``wqb`` arrives in FRACTAL_NZ, so its L1 fill is a plain burst copy.
        # The ND route had to gather base_n_q separate q_lora-strided rows per
        # window; the NZ route reads whole (16, 32) fractals back to back,
        # which is what makes this the cheapest 5.24 MB on the chip.

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
        qr_slot = self.qr_l1.next_write()
        qw_mem_copy(
            qr_slot,
            tile_view(qr_gm, (p.base_m, p.qr_k_l1), (tile, 0)),
            options=copy_options(l2_cache_ctl=p.l2_q),
        )
        scale_slot = self.scale_qr_l1.next_write()
        qw_mem_copy(
            scale_slot,
            tile_view(
                descale_qr_gm,
                (p.base_m, p.scale_k_len_q // MX_PAIR, MX_PAIR),
                (tile, 0, 0),
            ).flatten(-2, -1),
            options=copy_options(l2_cache_ctl=p.l2_q),
        )

    @jit
    def wait_qr(self):
        """The staged ``qr`` tile and its scales, as one transaction pair."""
        return self.qr_l1.next_read(), self.scale_qr_l1.next_read()

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
        # Filled through the uint8 face, read back through the fp8 alias in
        # gemm_head.  Both halves are written out by hand so the fill and the
        # read name the same depth-1 buffer.
        wqb_slot = self.wqb_l1[buf].next_write()
        qw_mem_copy(
            wqb_slot,
            tile_view(wqb_gm, (p.base_n_q, p.k_l1_q), (head, kw)),
            options=copy_options(l2_cache_ctl=p.l2_q),
        )
        scale_slot = self.scale_wqb_l1.next_write()
        qw_mem_copy(
            scale_slot,
            tile_view(descale_wqb_gm, (p.base_n_q, groups, MX_PAIR), (head, kw, 0))
            .flatten(-2, -1)
            .permute(1, 0),
            options=copy_options(l2_cache_ctl=p.l2_q),
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
        l0c = self.l0c.next_write()
        self._fill_wqb(wqb_gm, descale_wqb_gm, head, 0, 0)
        # Unrolled: keeps ``init`` a compile-time bool, so the hottest loop in
        # the kernel has no scf.if around its mmad.
        for kw in range_constexpr(p.k_l1_tiles_q):
            if const_expr(kw + 1 < p.k_l1_tiles_q):
                self._fill_wqb(
                    wqb_gm, descale_wqb_gm, head, kw + 1, (kw + 1) % p.n_buf_wqb
                )
            buf = kw % p.n_buf_wqb
            wqb_ready = from_tensor(self.wqb_l1[buf].next_read(), dtype=Float8E4M3FN)
            scale_ready = self.scale_wqb_l1.next_read()
            for kk in range_constexpr(p.step_k_q):
                k_l0 = kw * p.step_k_q + kk
                qw_mem_copy(
                    self.l0a,
                    tile_view(qr_rows, (p.base_m, p.base_k_q), (0, k_l0)),
                    options=copy_options(
                        mx_scale=tile_view(
                            scale_rows, (p.base_m, p.scale_k_l0_len), (0, k_l0)
                        )
                    ),
                )
                qw_mem_copy(
                    self.l0b,
                    tile_view(wqb_ready, (p.base_n_q, p.base_k_q), (0, kk)),
                    options=copy_options(
                        mx_scale=tile_view(
                            scale_ready, (p.scale_k_l0_len, p.base_n_q), (kk, 0)
                        )
                    ),
                )
                # Matmul takes M from the explicit operand view.
                a_rows = local_slice(
                    self.l0a.next_read(), (qr_rows.shape[0], p.base_k_q)
                )
                if const_expr(UNIT_FLAG):
                    last_k = k_l0 == p.k_l0_tiles_q - 1
                    qw_matmul(
                        l0c,
                        a_rows,
                        self.l0b,
                        init=k_l0 == 0,
                        unit_flag=3 if last_k else 2,
                        mx_mode=True,
                    )
                else:
                    qw_matmul(l0c, a_rows, self.l0b, init=k_l0 == 0, mx_mode=True)

    # Hand the Q accumulator from L0C straight to the vector UB. No GM hop.
    # rows is the live output height. rows_l1 is the accumulator layout.
    # They differ on a short tail.
    @jit
    def drain_to_vec(self, cv_ub, rows, rows_l1):
        # The accumulator is packed with the M used by matmul on this tile.
        c_rows = local_slice(self.l0c.next_read(), (rows_l1, self.p.d))
        qw_mem_copy(
            cv_ub.next_write(),
            c_rows[:rows, : self.p.d].partition(dim=0),
            options=copy_options(unit_flag=3 if UNIT_FLAG else 0),
        )
        return cv_ub.next_read()


class CubeW:
    """BF16 ``x @ ww^T`` scaled in fixpipe; writes GM directly, no vector hop."""

    def __init__(self, p: Plan):
        self.p = p
        bf16 = dtypes.bfloat16
        self.x_l1 = RingBuffer(
            MemLoc.L1,
            bf16,
            make_layout((p.base_m, p.k_l1_w)),
            depth=D_X,
            data_format="nz",
        )
        self.ww_l1 = RingBuffer(
            MemLoc.L1,
            bf16,
            make_layout((p.base_n_w, p.k_l1_w)),
            depth=D_X,
            data_format="nz",
        )
        self.l0a = RingBuffer(
            MemLoc.L0A, bf16, make_layout((p.base_m, p.base_k_w)), depth=D_L0AB_W
        )
        self.l0b = RingBuffer(
            MemLoc.L0B, bf16, make_layout((p.base_n_w, p.base_k_w)), depth=D_L0AB_W
        )
        self.l0c = RingBuffer(
            MemLoc.L0C, F32, make_layout((p.base_m, p.base_n_w)), depth=2
        )
        # ``ww`` arrives in FRACTAL_NZ; ``x`` is an activation and stays ND.

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
        tile is short, and it cannot be inferred either: a manual acquire/wait
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
                    self.x_gm,
                    self.ww_gm,
                    tile,
                    h_block,
                    local_head * p.w_per_head + j + 1,
                )
            x_ready = self.x_l1.next_read()
            ww_ready = self.ww_l1.next_read()
            x_rows = self.x_rows(x_ready, rows_l1)
            for kk in range_constexpr(p.step_k_w):
                qw_mem_copy(
                    self.l0a, tile_view(x_rows, (p.base_m, p.base_k_w), (0, kk))
                )
                qw_mem_copy(
                    self.l0b, tile_view(ww_ready, (p.base_n_w, p.base_k_w), (0, kk))
                )
                a_rows = local_slice(self.l0a.next_read(), (rows_l1, p.base_k_w))
                first = kk == 0 and j == 0
                if const_expr(PROBE == "no_runtime_init"):
                    qw_matmul(l0c_slot, a_rows, self.l0b, init=False)
                elif const_expr(PROBE == "no_w_mmad"):
                    pass
                else:
                    qw_matmul(
                        l0c_slot,
                        a_rows,
                        self.l0b,
                        init=local_head == 0 if const_expr(first) else False,
                    )

    @jit
    def _fill_window(self, x_gm, ww_gm, tile, k_base, kw):
        """Stage K window ``k_base + kw`` of ``x`` and ``ww`` into L1.

        ``k_base`` is this core's offset into the reduction, in windows: zero
        for a leader that owns the whole ``dim``, ``h_block * k_l1_tiles_w``
        under ``w_k_split``.
        """
        p = self.p
        x_slot = self.x_l1.next_write()
        qw_mem_copy(
            x_slot,
            tile_view(x_gm, (p.base_m, p.k_l1_w), (tile, k_base + kw)),
            options=copy_options(l2_cache_ctl=L2_W),
        )
        ww_slot = self.ww_l1.next_write()
        qw_mem_copy(
            ww_slot,
            tile_view(ww_gm, (p.base_n_w, p.k_l1_w), (0, k_base + kw)),
            options=copy_options(l2_cache_ctl=L2_W),
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
            x_ready = self.x_l1.next_read()
            ww_ready = self.ww_l1.next_read()
            x_rows = self.x_rows(x_ready, rows_l1)
            for kk in range_constexpr(p.step_k_w):
                qw_mem_copy(
                    self.l0a, tile_view(x_rows, (p.base_m, p.base_k_w), (0, kk))
                )
                qw_mem_copy(
                    self.l0b, tile_view(ww_ready, (p.base_n_w, p.base_k_w), (0, kk))
                )
                a_rows = local_slice(self.l0a.next_read(), (rows_l1, p.base_k_w))
                first = kk == 0 and j == 0
                qw_matmul(
                    l0c_slot,
                    a_rows,
                    self.l0b,
                    init=local_head == 0 if const_expr(first) else False,
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
            x_ready = self.x_l1.next_read()
            ww_ready = self.ww_l1.next_read()
            x_rows = self.x_rows(x_ready, rows_l1)
            for kk in range_constexpr(p.step_k_w):
                qw_mem_copy(
                    self.l0a, tile_view(x_rows, (p.base_m, p.base_k_w), (0, kk))
                )
                qw_mem_copy(
                    self.l0b, tile_view(ww_ready, (p.base_n_w, p.base_k_w), (0, kk))
                )
                a_rows = local_slice(self.l0a.next_read(), (rows_l1, p.base_k_w))
                first = kw == 0 and kk == 0
                qw_matmul(l0c_slot, a_rows, self.l0b, init=const_expr(first))

    @jit
    def store(self, tile, l0c_slot, rows_l1):
        """Whole-reduction store, straight into ``w``. Not used under w_workspace."""
        # softmax_scale folded into the fixpipe DEQSCALE register: the W path
        # never touches the vector unit.
        p = self.p
        ready = self.l0c.next_read()
        qw_mem_copy(
            tile_view(self.w_gm, (p.base_m, p.base_n_w), (tile, 0)),
            self.c_rows(ready, rows_l1).convert(
                tile_view(self.w_gm, (p.base_m, p.base_n_w), (tile, 0)).dtype,
                scale=self.softmax_scale,
            ),
            options=copy_options(l2_cache_ctl=L2_W),
        )

    # Each K slice writes its W partial into a workspace region only it owns.
    # The store overwrites; it is not an atomic add, so the buffer does not
    # need to be zeroed. The vector unit sums the slices later.
    @jit
    def store_partial(self, tile, h_block, n_tiles, l0c_slot, rows_l1):
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
        ready = self.l0c.next_read()
        qw_mem_copy(
            tile_view(
                self.workspace_gm, (p.base_m, p.base_n_w), (h_block * n_tiles + tile, 0)
            ),
            self.c_rows(ready, rows_l1).convert(
                tile_view(
                    self.workspace_gm,
                    (p.base_m, p.base_n_w),
                    (h_block * n_tiles + tile, 0),
                ).dtype,
                scale=self.softmax_scale,
            ),
            options=copy_options(l2_cache_ctl=L2_W),
        )


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
        self.q_ub = Buffer(MemLoc.UB, U8, make_layout((rows, p.d // 2)))
        self.dsc_ub = Buffer(MemLoc.UB, U8, make_layout((p.enc_regs, VL)))
        # Buffer access dependencies order MTE2 DMA against VF reads.
        self.amax_ub = Buffer(MemLoc.UB, F32, make_layout((p.amax_len,)))
        self.off_ub = Buffer(MemLoc.UB, I32, make_layout((p.amax_len,)))
        self.cos_ub = Buffer(MemLoc.UB, F32, make_layout((rows, max(p.dr, 1))))
        self.sin_ub = Buffer(MemLoc.UB, F32, make_layout((rows, max(p.dr, 1))))
        # W reduction staging.  ``band_ub`` holds every partial of one band so
        # the sum is whole-register adds; partial ``i`` starts exactly
        # ``red_regs`` registers in, because a band's innermost extent
        # (n_heads fp32 = 128 B) is already 32-byte aligned and so the
        # declared stride is the real one.  Plain Buffers: the DMA-to-VF and
        # VF-to-DMA handoffs are both ordered by vec_sync_all, which is
        # coarse but runs at most twice per band.
        if p.w_workspace:
            self.band_ub = Buffer(
                MemLoc.UB, F32, make_layout((p.n_head_blocks * p.rows_red, p.n_heads))
            )
            self.wsum_ub = Buffer(MemLoc.UB, F32, make_layout((p.rows_red, p.n_heads)))

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
            qw_mem_copy(
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
        qw_mem_copy(out, local_slice(self.wsum_ub, (out.shape[0], p.n_heads)))

    @jit
    def load_tile_constants(self, cos_tile, sin_tile):
        """Load the per-tile RoPE tables into UB."""
        if const_expr(self.p.dr > 0):
            qw_mem_copy(
                self.cos_ub, cos_tile.partition(dim=0, part_id=get_subblock_id())
            )
            qw_mem_copy(
                self.sin_ub, sin_tile.partition(dim=0, part_id=get_subblock_id())
            )
        return self.cos_ub, self.sin_ub

    # Q epilogue, three passes: RoPE on the tail Dr channels, per-group
    # scale, then round and pack FP4. For a normal nonzero amax,
    # scale = 2**ceil(log2(amax/6)). Zero and tiny amax take other branches.
    # descale_q stores the E8M0 code of the scale that is multiplied back on
    # dequant, not its reciprocal. rows is the live row count of this AIV;
    # using the buffer capacity would walk off a short tail.
    @jit
    def epilogue(self, cv_ub, rows, cos_ub, sin_ub):
        """RoPE then MXFP4 quantise ``cv_ub`` into ``q_ub`` / ``dsc_ub``."""
        self._rope_amax(cv_ub, rows, cos_ub, sin_ub)
        self._e8m0_scales()
        self._pack_e2m1(cv_ub, rows)

    # Store packed q and the scale from UB to GM, limited to this
    # partition's live rows.
    @jit
    def store(self, q_tile, dsc_tile, rows):
        """Write back exactly ``rows`` rows, so a short T tail cannot overrun GM."""
        p = self.p
        qw_mem_copy(
            q_tile.partition(dim=0, part_id=get_subblock_id()),
            local_slice(self.q_ub, (rows, p.d // 2), stride=(p.d // 2, 1)),
        )
        qw_mem_copy(
            dsc_tile.partition(dim=0, part_id=get_subblock_id()),
            local_slice(self.dsc_ub, (rows, p.groups), stride=(p.groups, 1)),
        )

    @jit
    def _rope_amax(self, cv_ub, rows, cos_ub, sin_ub):
        """Inplace RoPE on the tail lanes, then per-group absolute max."""
        p = self.p
        with vf(mode="simd"):
            m32 = full_mask()
            m_lo = create_mask("h", 32)
            lane = varange(0, I32)
            k_lane_wrap = vdups(VL - 1, I32, mask=m32)
            half_idx = vreinterpret(
                vbitwise_and(vadds(lane, VL // 2, mask=m32), k_lane_wrap, mask=m32), U32
            )
            for r in range(rows):
                base = r * p.d
                for i in range_constexpr(p.n_reg):
                    x = vload(cv_ub, base + i * VL)
                    pe = i - (p.n_reg - p.rope_regs)
                    if const_expr(p.rope_regs > 0 and pe >= 0):
                        cosv = vload(cos_ub, r * p.dr + pe * VL)
                        sinv = vload(sin_ub, r * p.dr + pe * VL)
                        x_even, x_odd = vdeinterleave(x, x)
                        cos_even, cos_odd = vdeinterleave(cosv, cosv)
                        sin_even, sin_odd = vdeinterleave(sinv, sinv)
                        even = vsub(
                            vmul(x_even, cos_even, mask=m_lo),
                            vmul(x_odd, sin_even, mask=m_lo),
                            mask=m_lo,
                        )
                        odd = vadd(
                            vmul(x_odd, cos_odd, mask=m_lo),
                            vmul(x_even, sin_odd, mask=m_lo),
                            mask=m_lo,
                        )
                        x, _ = vinterleave(even, odd)
                        vstore(cv_ub, base + i * VL, x, m32)
                    abs_d = vabs(x, mask=m32)
                    abs_d_hi = vreinterpret(
                        vgather_reg(vreinterpret(abs_d, I32), half_idx), F32
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
                e8m0_exp = vselect(e8m0_exp, one_i, cond_mask=vne(ab, zero_i, mask=m32))
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
    def _pack_e2m1(self, cv_ub, rows):
        """Round magnitudes to nearest-even E2M1 and pack two nibbles per byte."""
        p = self.p
        with vf(mode="simd"):
            m32 = full_mask()
            m_lo = create_mask("h", 32)
            lane = varange(0, I32)
            zero_i = vdups(0, I32, mask=m32)
            k8 = vdups(8, I32, mask=m32)
            mag_mask = vdups(FP32_MAG_MASK, I32, mask=m32)
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
                    # Power-of-two normalization also handles subnormal inputs.
                    inverse_exp = vsub(
                        vdups(E8M0_BIAS, I32, mask=m32),
                        vshr(off, 3, mask=m32),
                        mask=m32,
                    )
                    inverse = vreinterpret(
                        vshl(inverse_exp, FP32_EXP_SHIFT, mask=m32), F32
                    )
                    normalized = vmul(vreinterpret(mag, F32), inverse, mask=m32)
                    # Compare exact E2M1 midpoints.  Ties select the even code.
                    code = vdups(7, I32, mask=m32)
                    for threshold, lower_code in (
                        (5.0, 6),
                        (3.5, 5),
                        (2.5, 4),
                        (1.75, 3),
                        (1.25, 2),
                        (0.75, 1),
                        (0.25, 0),
                    ):
                        if const_expr(lower_code % 2 == 0):
                            choose_lower = rr.vles(normalized, threshold, mask=m32)
                        else:
                            choose_lower = rr.vlts(normalized, threshold, mask=m32)
                        code = vselect(
                            vdups(lower_code, I32, mask=m32),
                            code,
                            cond_mask=choose_lower,
                        )
                    sign = vbitwise_and(
                        vshr(vreinterpret(xv, I32), 28, mask=m32), k8, mask=m32
                    )
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


MX_GROUP_SIZE = 32

FRACTAL_NZ = 29


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
    specs.append(TensorSpec((rows_live, n_heads * (d // 2)), U8))
    specs.append(TensorSpec((rows_live, n_heads * groups), U8))
    specs.append(TensorSpec((rows_live, n_heads), F32))
    specs.append(TensorSpec((workspace_rows, n_heads), F32))
    specs.append(I64)
    specs.append(F32)
    return specs


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
        raise ValueError(
            f"{name} must be a one-byte fp8/e8m0 dtype, got {tensor.dtype}"
        )


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
        raise ValueError(f"RoPE tables have {rope_sin.shape[0]} rows, expected T={t}")
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


class DecodePlan(Plan):
    def __init__(self):
        super().__init__(16, 5120, 1280, 32, 256, 64, 32, False)

    def _init_q(self, q_lora, d):
        self.base_n_q = 256
        self.base_k_q = 64
        self.k_l1_tiles_q = 5
        self.k_l1_q = 256
        self.step_k_q = 4
        self.n_buf_wqb = 2
        self.k_l0_tiles_q = 20
        self.scale_k_l0_len = 2
        self.scale_k_l1_len_q = 8
        self.scale_k_len_q = 40
        self.qr_k_l1 = 1280

    def _pick_k_l1_w(self):
        return 320

    def _pick_base_k_w(self):
        return 160


class PairVector(VectorQ):
    def __init__(self, p):
        super().__init__(p)
        self.q_stage = RingBuffer(MemLoc.UB, U8, make_layout((8, 128)), depth=2)
        self.scale_stage = RingBuffer(
            MemLoc.UB, U8, make_layout((p.enc_regs, VL)), depth=2
        )

    @jit
    def epilogue(self, cv_ub, rows, cos_ub, sin_ub):
        self.q_ub = self.q_stage.next_write()
        self.dsc_ub = self.scale_stage.next_write()
        self._rope_amax(cv_ub, rows, cos_ub, sin_ub)
        self._e8m0_scales()
        self._pack_e2m1(cv_ub, rows)

    @jit
    def store(self, q_tile, dsc_tile, rows):
        qw_mem_copy(
            q_tile.partition(dim=0, part_id=get_subblock_id()),
            local_slice(self.q_stage.next_read(), (rows, 128), stride=(128, 1)),
        )
        qw_mem_copy(
            dsc_tile.partition(dim=0, part_id=get_subblock_id()),
            local_slice(self.scale_stage.next_read(), (rows, 8), stride=(8, 1)),
        )

    @jit
    def _rope_amax(self, cv_ub, rows, cos_ub, sin_ub):
        with vf(mode="simd"):
            m32 = full_mask()
            m_lo = create_mask("h", 32)
            lane = varange(0, I32)
            half_idx = vreinterpret(
                vbitwise_and(
                    vadds(lane, VL // 2, mask=m32),
                    vdups(VL - 1, I32, mask=m32),
                    mask=m32,
                ),
                U32,
            )
            for r in range(rows):
                for i in range_constexpr(4):
                    base = r * 256 + i * VL
                    x = vload(cv_ub, base)
                    if const_expr(i % 2 == 1):
                        cosv = vload(cos_ub, r * 64)
                        sinv = vload(sin_ub, r * 64)
                        xe, xo = vdeinterleave(x, x)
                        ce, co = vdeinterleave(cosv, cosv)
                        se, so = vdeinterleave(sinv, sinv)
                        even = vsub(
                            vmul(xe, ce, mask=m_lo), vmul(xo, se, mask=m_lo), mask=m_lo
                        )
                        odd = vadd(
                            vmul(xo, co, mask=m_lo), vmul(xe, so, mask=m_lo), mask=m_lo
                        )
                        x, _ = vinterleave(even, odd)
                        vstore(cv_ub, base, x, m32)
                    ab = vabs(x, mask=m32)
                    hi = vreinterpret(vgather_reg(vreinterpret(ab, I32), half_idx), F32)
                    slot = r * 8 + i * 2
                    vstore_first(self.amax_ub, slot, vreduce_max(ab, mask=m_lo))
                    vstore_first(self.amax_ub, slot + 1, vreduce_max(hi, mask=m_lo))
            vmem_bar("vst_vld")


class DoubleBufferW(CubeW):
    def __init__(self, p):
        self.p = p
        bf16 = dtypes.bfloat16
        self.x_l1 = RingBuffer(
            MemLoc.L1, bf16, make_layout((16, 320)), depth=2, data_format="nz"
        )
        self.ww_l1 = RingBuffer(
            MemLoc.L1, bf16, make_layout((32, 320)), depth=2, data_format="nz"
        )
        self.l0a = RingBuffer(MemLoc.L0A, bf16, make_layout((16, 160)), depth=2)
        self.l0b = RingBuffer(MemLoc.L0B, bf16, make_layout((32, 160)), depth=2)
        self.l0c = RingBuffer(MemLoc.L0C, F32, make_layout((16, 32)), depth=2)


class ParallelQw:
    """小 S 的 Q/W 独立分核；W 按核号顺序归约，完整 K 累计后才发布。"""

    def __init__(self):
        self.p = DecodePlan()

    @jit
    def run_q(self, qr, wqb, dqr, dwqb, sin, cos, q, dq, t, head_task=None):
        p = self.p
        cube = CubeQ(p)
        vec = PairVector(p)
        handoff = RingBuffer(MemLoc.UB, F32, make_layout((8, 256)), depth=2)
        pair = get_block_idx() if head_task is None else head_task
        rows = cv_valid_extent(
            tile_view(sin, (16, 64), (0, 0)), axis=0, part_id=get_subblock_id()
        )
        cos_ub, sin_ub = vec.load_tile_constants(
            tile_view(cos, (16, 64), (0, 0)),
            tile_view(sin, (16, 64), (0, 0)),
        )
        cube.load_qr(qr, dqr, 0)
        qr_slot, ds_slot = cube.wait_qr()
        qr_rows, ds_rows = cube.rows_view(qr_slot, ds_slot, 16)
        cube.gemm_head(wqb, dwqb, pair, qr_rows, ds_rows)
        ready = cube.drain_to_vec(handoff, t, 16)
        vec.epilogue(ready, rows, cos_ub, sin_ub)
        vec.store(
            tile_view(q, (16, 128), (0, pair)), tile_view(dq, (16, 8), (0, pair)), rows
        )

    @jit
    def run_w(self, x, ww, workspace, t, scale, partial_task=None):
        cube = DoubleBufferW(self.p)
        cube.x_gm = x
        cube.ww_gm = ww
        cube.workspace_gm = workspace
        cube.softmax_scale = scale
        part = get_block_idx() - 16 if partial_task is None else partial_task
        acc = cube.l0c.next_write()
        cube.gemm_full(0, part, acc, 16)
        cube.store_partial(0, part, 1, acc, 16)

    @jit
    def run_reduce(self, workspace, w, t):
        p = self.p
        VectorQ(p).reduce_all(
            workspace, w, ceil_div(t, p.rows_red), p.n_bands_workspace
        )


# K投影、量化及cache发布。

FP4_GROUP_SIZE = 32

FP4_MAX = 6.0

FP32_EXP_MASK = 0xFF

BASE_M = 16


NZ_FRACTAL_ALIGN = 16  # axis alignment required by raw FRACTAL_NZ copies

E2M1_VALUES_PER_BYTE = 2  # two 4-bit E2M1 codes packed per byte

ROPE_PAIR_SIZE = 2  # adjacent even/odd lanes form one rope pair


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


class KPostQuantizer:
    """Token-parallel K normalization, RoPE and quantization.

    Packed bytes use a private aligned row; no cache group ownership is
    imposed on these expensive vector calculations.
    """

    def __init__(self, *, total_rows, index_head_dim, rope_head_dim):
        self.total_rows = int(total_rows)
        self.index_head_dim = int(index_head_dim)
        self.rope_head_dim = int(rope_head_dim)
        self.padded_dim = _ceil_div(self.index_head_dim, VL) * VL

    @jit
    def _normalize(self, projected, gamma, normalized, epsilon: dtypes.float32):
        dim = self.index_head_dim
        with vf(mode="simd"):
            full_mask = rr.full_mask()
            lane0 = rr.update_mask(1, elem_bits=32)[0]
            square_sum = rr.vdups(0.0, dtypes.float32, mask=lane0)
            for segment in range(_ceil_div(dim, VL)):
                segment_offset = segment * VL
                valid = min(VL, dim - segment_offset)
                segment_mask = rr.update_mask(valid, elem_bits=32)[0]
                packed = rr.vload_unpack(
                    projected, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                )
                elem = rr.vcast(packed, dtypes.float32, mask=segment_mask)
                partial = rr.vreduce_sum(
                    rr.vmul(elem, elem, mask=segment_mask), mask=segment_mask
                )
                square_sum = rr.vadd(square_sum, partial, mask=lane0)

            mean = rr.vmuls(square_sum, 1.0 / dim, mask=lane0)
            root = rr.vsqrt(rr.vadds(mean, epsilon, mask=lane0), mask=lane0)
            inverse_rms = rr.vdiv(
                rr.vdups(1.0, dtypes.float32, mask=lane0), root, mask=lane0
            )
            inverse_rms = rr.vdup(inverse_rms, mask=full_mask)

            for segment in range(_ceil_div(dim, VL)):
                segment_offset = segment * VL
                valid = min(VL, dim - segment_offset)
                segment_mask = rr.update_mask(valid, elem_bits=32)[0]
                packed = rr.vload_unpack(
                    projected, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                )
                elem = rr.vcast(packed, dtypes.float32, mask=segment_mask)
                gamma_value = rr.vload(gamma, segment_offset)
                elem = rr.vmul(elem, inverse_rms, mask=segment_mask)
                elem = rr.vmul(elem, gamma_value, mask=segment_mask)
                elem = rr.vcast(elem, dtypes.bfloat16, mask=segment_mask)
                rr.vstore_pack(
                    normalized,
                    segment_offset,
                    elem,
                    segment_mask,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )

            rr.vmem_bar("vst_vld")

    @jit
    def _load_bf16_rope_work(self, source, work):
        with vf(mode="simd"):
            dim = self.index_head_dim
            first_segment = (dim - self.rope_head_dim) // VL
            for segment in range(first_segment, _ceil_div(dim, VL)):
                segment_offset = segment * VL
                valid = min(VL, dim - segment_offset)
                segment_mask = rr.update_mask(valid, elem_bits=32)[0]
                packed = rr.vload_unpack(
                    source, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                )
                elem = rr.vcast(packed, dtypes.float32, mask=segment_mask)
                rr.vstore(work, segment_offset, elem, segment_mask)

            rr.vmem_bar("vst_vld")

    @jit
    def _apply_rope(self, work, rope_sin, rope_cos, rotated):
        """Rotate adjacent pairs in the final ``rope_head_dim`` columns."""
        dim = self.index_head_dim
        rd = self.rope_head_dim
        rope_start = dim - rd

        with vf(mode="simd"):
            full_mask = rr.full_mask()
            for chunk in range(_ceil_div(rd, VL)):
                chunk_offset = chunk * VL
                valid = min(VL, rd - chunk_offset)
                pair_mask = rr.update_mask(valid // ROPE_PAIR_SIZE, elem_bits=32)[0]
                rope_mask = rr.update_mask(valid, elem_bits=32)[0]
                source_offset = rope_start + chunk_offset
                if const_expr(rope_start % 8 == 0):
                    x_even, x_odd = rr.vload_deinterleave(
                        work, source_offset, width="b32"
                    )
                else:
                    rr.vload_unalign_init(work, source_offset)
                    packed_rope = rr.vload_unalign(work, source_offset)
                    x_even, x_odd = rr.vdeinterleave(packed_rope, packed_rope)

                cos_even, cos_odd = rr.vload_deinterleave(
                    rope_cos, chunk_offset, width="b32"
                )
                sin_even, sin_odd = rr.vload_deinterleave(
                    rope_sin, chunk_offset, width="b32"
                )
                out_even = rr.vsub(
                    rr.vmul(x_even, cos_even, mask=pair_mask),
                    rr.vmul(x_odd, sin_even, mask=pair_mask),
                    mask=pair_mask,
                )
                out_odd = rr.vadd(
                    rr.vmul(x_odd, cos_odd, mask=pair_mask),
                    rr.vmul(x_even, sin_odd, mask=pair_mask),
                    mask=pair_mask,
                )
                interleaved, _ = rr.vinterleave(out_even, out_odd)
                rr.vstore(rotated, VL + chunk_offset, interleaved, rope_mask)
            rr.vmem_bar("vst_vld")

            # CANN 9.2 cannot lower the DSL unaligned-store helper.  Rebase
            # the compact rotated tail through an unaligned load, merge it
            # into aligned 64-lane work segments, and only issue aligned
            # stores.  The VL prefix guarantees every rebased load is in-bounds.
            first_segment = rope_start // VL
            for segment in range(first_segment, _ceil_div(dim, VL)):
                segment_start = segment * VL
                local_start = max(0, rope_start - segment_start)
                local_end = min(VL, dim - segment_start)
                end_mask = rr.update_mask(local_end, elem_bits=32)[0]
                prefix_mask = rr.update_mask(local_start, elem_bits=32)[0]
                range_mask = rr.mask_xor(end_mask, prefix_mask, exec_mask=full_mask)
                rotated_offset = VL + segment_start - rope_start
                rr.vload_unalign_init(rotated, rotated_offset)
                shifted = rr.vload_unalign(rotated, rotated_offset)
                old = rr.vload(work, segment_start)
                merged = rr.vmerge(range_mask, shifted, old)
                rr.vstore(work, segment_start, merged, end_mask)

            rr.vmem_bar("vst_vld")

    @jit
    def _pack_mxfp4(self, source, packed_output, code_scratch):
        """Pack signed E2M1 pairs and append one UE8M0 scale per 32 values."""
        dim = self.index_head_dim
        num_groups = dim // FP4_GROUP_SIZE
        value_bytes = dim // E2M1_VALUES_PER_BYTE
        with vf(mode="simd"):
            mask = rr.update_mask(FP4_GROUP_SIZE, elem_bits=32)[0]
            packed_mask = rr.update_mask(
                FP4_GROUP_SIZE // E2M1_VALUES_PER_BYTE, elem_bits=32
            )[0]
            byte_block_mask = rr.update_mask(FP4_GROUP_SIZE, elem_bits=32)[0]
            lane0 = rr.update_mask(1, elem_bits=32)[0]
            one_u32 = rr.vdups(1, dtypes.uint32, mask=mask)
            zero_u32 = rr.vdups(0, dtypes.uint32, mask=mask)
            sign_u32 = rr.vdups(8, dtypes.uint32, mask=mask)
            mantissa_mask = rr.vdups(FP32_MANTISSA_MASK, dtypes.uint32, mask=mask)
            exponent_mask = rr.vdups(FP32_EXP_MASK, dtypes.uint32, mask=mask)

            for group in range(num_groups):
                segment_offset = group * FP4_GROUP_SIZE
                unpacked = rr.vload_unpack(
                    source, segment_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                )
                elem = rr.vcast(unpacked, dtypes.float32, mask=mask)
                abs_value = rr.vabs(elem, mask=mask)
                amax = rr.vreduce_max(abs_value, mask=mask)
                safe_amax = rr.vmaxs(amax, FP4_MAX * (2.0**-126), mask=mask)
                raw_scale = rr.vmuls(safe_amax, 1.0 / FP4_MAX, mask=mask)

                scale_bits = rr.vreinterpret(raw_scale, dtypes.uint32)
                exponent = rr.vbitwise_and(
                    rr.vshr(scale_bits, FP32_EXP_SHIFT, mask=mask),
                    exponent_mask,
                    mask=mask,
                )
                mantissa = rr.vbitwise_and(scale_bits, mantissa_mask, mask=mask)
                has_mantissa = rr.vne(mantissa, zero_u32, mask=mask)
                increment = rr.vselect(one_u32, zero_u32, cond_mask=has_mantissa)
                exponent = rr.vadd(exponent, increment, mask=mask)
                pow2_bits = rr.vshl(exponent, FP32_EXP_SHIFT, mask=mask)
                scale = rr.vreinterpret(pow2_bits, dtypes.float32)
                scale = rr.vdup(scale, mask=mask)

                inverse_bits = rr.vshl(
                    rr.vsub(
                        rr.vdups(254, dtypes.uint32, mask=mask), exponent, mask=mask
                    ),
                    FP32_EXP_SHIFT,
                    mask=mask,
                )
                inverse = rr.vdup(
                    rr.vreinterpret(inverse_bits, dtypes.float32), mask=mask
                )
                scaled_abs = rr.vmul(abs_value, inverse, mask=mask)
                code = rr.vdups(7, dtypes.uint32, mask=mask)
                for threshold, level_code in (
                    (5.0, 6),
                    (3.5, 5),
                    (2.5, 4),
                    (1.75, 3),
                    (1.25, 2),
                    (0.75, 1),
                    (0.25, 0),
                ):
                    lower = rr.vdups(level_code, dtypes.uint32, mask=mask)
                    if const_expr(level_code % 2 == 0):
                        choose_lower = rr.vles(scaled_abs, threshold, mask=mask)
                    else:
                        choose_lower = rr.vlts(scaled_abs, threshold, mask=mask)
                    code = rr.vselect(lower, code, cond_mask=choose_lower)

                sign = rr.vbitwise_and(
                    rr.vshr(rr.vreinterpret(elem, dtypes.uint32), 28, mask=mask),
                    sign_u32,
                    mask=mask,
                )
                code = rr.vadd(code, sign, mask=mask)
                even_code, odd_code = rr.vdeinterleave(code, code)
                high_nibble = rr.vshl(odd_code, 4, mask=packed_mask)
                packed_code = rr.vbitwise_or(even_code, high_nibble, mask=packed_mask)
                rr.vstore(
                    code_scratch,
                    group * VL,
                    packed_code,
                    packed_mask,
                )

                exponent_u8 = rr.vcast(
                    exponent,
                    dtypes.uint8,
                    mask=lane0,
                    reg_layout=rr.RegLayout.ZERO,
                )
                rr.vstore_first(
                    packed_output,
                    value_bytes + group,
                    exponent_u8,
                )

            # Two 32-elem groups produce 16 bytes each.  Join them into one
            # aligned 32-byte store to avoid non-aligned vector writes on 950.
            prefix = rr.update_mask(
                FP4_GROUP_SIZE // E2M1_VALUES_PER_BYTE, elem_bits=32
            )[0]
            tail = rr.mask_xor(byte_block_mask, prefix, exec_mask=rr.full_mask())
            rr.vmem_bar("vst_vld")

            for group_pair in range(num_groups // E2M1_VALUES_PER_BYTE):
                first_group = 2 * group_pair
                first = rr.vload(code_scratch, first_group * VL)
                second_shifted = rr.vload(
                    code_scratch,
                    (first_group + 1) * VL - FP4_GROUP_SIZE // E2M1_VALUES_PER_BYTE,
                )
                joined = rr.vmerge(tail, second_shifted, first)
                rr.vstore_pack(
                    packed_output,
                    group_pair * FP4_GROUP_SIZE,
                    joined,
                    byte_block_mask,
                    pack_mode=rr.PackMode.B32_TO_B8,
                )
            if const_expr(num_groups % E2M1_VALUES_PER_BYTE != 0):
                last_group = num_groups - 1
                last = rr.vload(code_scratch, last_group * VL)
                rr.vstore_pack(
                    packed_output,
                    (num_groups // E2M1_VALUES_PER_BYTE) * FP4_GROUP_SIZE,
                    last,
                    packed_mask,
                    pack_mode=rr.PackMode.B32_TO_B8,
                )

            rr.vmem_bar("vst_vld")

    @jit
    def _cast_output(self, work, output):
        with vf(mode="simd"):
            dim = self.index_head_dim
            first_segment = (dim - self.rope_head_dim) // VL
            for segment in range(first_segment, _ceil_div(dim, VL)):
                segment_offset = segment * VL
                valid = min(VL, dim - segment_offset)
                segment_mask = rr.update_mask(valid, elem_bits=32)[0]
                elem = rr.vload(work, segment_offset)
                packed = rr.vcast(elem, dtypes.bfloat16, mask=segment_mask)
                rr.vstore_pack(
                    output,
                    segment_offset,
                    packed,
                    segment_mask,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )

    @jit
    def __call__(
        self,
        gm_projected: Tensor,
        gm_rope_sin: Tensor,
        gm_rope_cos: Tensor,
        gm_gamma: Tensor,
        gm_cache_index: Tensor,
        gm_tokens: Tensor,
        epsilon: dtypes.float32,
        owner_offset=0,
    ):
        dim = self.index_head_dim
        rd = self.rope_head_dim
        projected = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, self.padded_dim)), depth=2
        )
        output = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, self.padded_dim + VL))
        )
        packed_output = Buffer(
            MemLoc.UB, dtypes.uint8, make_layout((1, self.padded_dim))
        )
        packed_row = RingBuffer(MemLoc.UB, dtypes.uint8, make_layout((1, 96)), depth=2)
        code_scratch = Buffer(
            MemLoc.UB, dtypes.uint32, make_layout((dim // FP4_GROUP_SIZE, VL))
        )
        gamma = Buffer(MemLoc.UB, dtypes.float32, make_layout((self.padded_dim,)))
        work = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.padded_dim + 2 * VL,))
        )
        rope_sin = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, rd)))
        rope_cos = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, rd)))
        rotated = Buffer(MemLoc.UB, dtypes.float32, make_layout((rd + 2 * VL,)))
        mem_copy(gamma, gm_gamma)
        aiv = get_block_idx() * get_subblock_dim() + get_subblock_id() - owner_offset
        aivs = get_block_num() * get_subblock_dim()
        for row in range(aiv, gm_projected.shape[0], aivs):
            if gm_cache_index[row] != -1:
                mem_copy(
                    projected.next_write(), tile_slice(gm_projected, (1, dim), (row, 0))
                )
                mem_copy(rope_sin, tile_slice(gm_rope_sin, (1, rd), (row, 0)))
                mem_copy(rope_cos, tile_slice(gm_rope_cos, (1, rd), (row, 0)))
                self._normalize(projected.next_read(), gamma, output, epsilon)
                self._load_bf16_rope_work(output, work)
                self._apply_rope(work, rope_sin, rope_cos, rotated)
                self._cast_output(work, output)
                self._pack_mxfp4(output, packed_output, code_scratch)
                # 64 data + 4 scale bytes; the remaining 28 bytes are private
                # alignment padding and never enter the public cache.
                mem_copy(packed_row.next_write(), packed_output[0:1, 0:96])
                mem_copy(
                    tile_slice(gm_tokens, (1, 96), (row, 0)), packed_row.next_read()
                )


class SeparateKPostQuantizer(KPostQuantizer):
    """Separate-cache writer; shares all normalization/FP4 arithmetic."""

    @jit
    def __call__(
        self,
        gm_projected: Tensor,
        gm_rope_sin: Tensor,
        gm_rope_cos: Tensor,
        gm_gamma: Tensor,
        gm_cache_index: Tensor,
        gm_cache: Tensor,
        gm_scale_cache: Tensor,
        epsilon: dtypes.float32,
    ):
        dim = self.index_head_dim
        rd = self.rope_head_dim
        projected = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, self.padded_dim)), depth=2
        )
        output = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, self.padded_dim + VL))
        )
        packed = Buffer(MemLoc.UB, dtypes.uint8, make_layout((1, self.padded_dim)))
        data = RingBuffer(MemLoc.UB, dtypes.uint8, make_layout((1, 64)), depth=1)
        scale = RingBuffer(MemLoc.UB, dtypes.uint8, make_layout((1, 4)), depth=1)
        scratch = Buffer(
            MemLoc.UB, dtypes.uint32, make_layout((dim // FP4_GROUP_SIZE, VL))
        )
        gamma = Buffer(MemLoc.UB, dtypes.float32, make_layout((self.padded_dim,)))
        work = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.padded_dim + 2 * VL,))
        )
        sin = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, rd)))
        cos = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, rd)))
        rotated = Buffer(MemLoc.UB, dtypes.float32, make_layout((rd + 2 * VL,)))
        mem_copy(gamma, gm_gamma)
        aiv = get_block_idx() * get_subblock_dim() + get_subblock_id()
        aivs = get_block_num() * get_subblock_dim()
        # Physical-slot ownership preserves last-input-wins semantics.
        for row in range(gm_projected.shape[0]):
            slot = gm_cache_index[row]
            if slot >= 0 and slot % aivs == aiv:
                mem_copy(
                    projected.next_write(), tile_slice(gm_projected, (1, dim), (row, 0))
                )
                mem_copy(sin, tile_slice(gm_rope_sin, (1, rd), (row, 0)))
                mem_copy(cos, tile_slice(gm_rope_cos, (1, rd), (row, 0)))
                self._normalize(projected.next_read(), gamma, output, epsilon)
                self._load_bf16_rope_work(output, work)
                self._apply_rope(work, sin, cos, rotated)
                self._cast_output(work, output)
                self._pack_mxfp4(output, packed, scratch)
                page = slot // 128
                offset = slot % 128
                mem_copy(data.next_write(), tile_slice(packed, (1, 64), (0, 0)))
                mem_copy(
                    tile_slice(gm_cache, (1, 1, 1, 64), (page, offset, 0, 0)).view(
                        1, 64
                    ),
                    data.next_read(),
                )
                mem_copy(scale.next_write(), tile_slice(packed, (1, 4), (0, 16)))
                mem_copy(
                    tile_slice(gm_scale_cache, (1, 1, 1, 4), (page, offset, 0, 0)).view(
                        1, 4
                    ),
                    scale.next_read(),
                )


@jit
def publish_combined_cache(tokens: Tensor, indices: Tensor, cache: Tensor):
    """One owner per 8-token row; update slots in original input order."""
    aiv = get_block_idx() * get_subblock_dim() + get_subblock_id()
    aivs = get_block_num() * get_subblock_dim()
    groups = RingBuffer(MemLoc.UB, dtypes.uint8, make_layout((1, 544)), depth=2)
    rows = RingBuffer(MemLoc.UB, dtypes.uint8, make_layout((1, 96)), depth=2)
    for first in range(indices.shape[0]):
        slot = indices[first]
        if slot >= 0:
            group_id = slot // 8
            if group_id % aivs == aiv:
                earlier = 0
                for previous in range(first):
                    other = indices[previous]
                    if other >= 0 and other // 8 == group_id:
                        earlier = 1
                if earlier == 0:
                    page = slot // 128
                    group = slot % 128 // 8
                    target = groups.next_write()
                    mem_copy(
                        target,
                        tile_slice(cache, (1, 1, 544), (page, group, 0)).view(1, 544),
                    )
                    # Preserve old bytes before partial vector writes.
                    vec_sync_all()
                    for row in range(indices.shape[0]):
                        current = indices[row]
                        if current >= 0 and current // 8 == group_id:
                            mem_copy(
                                rows.next_write(), tile_slice(tokens, (1, 96), (row, 0))
                            )
                            source = rows.next_read()
                            source32 = from_tensor(
                                source, layout=make_layout((1, 24)), dtype=dtypes.uint32
                            )
                            target32 = from_tensor(
                                target,
                                layout=make_layout((1, 136)),
                                dtype=dtypes.uint32,
                            )
                            subslot = current % 8
                            with vf(mode="simd"):
                                mask16 = rr.update_mask(16, elem_bits=32)[0]
                                mask8 = rr.update_mask(8, elem_bits=32)[0]
                                data = rr.vload(source32, 0)
                                scale = rr.vdup(rr.vload(source32, 16), mask=mask8)
                                previous_scale = rr.vload(target32, 128)
                                lanes = rr.varange(0, dtypes.uint32)
                                selected = rr.veqs(lanes, subslot, mask=mask8)
                                scales = rr.vselect(
                                    scale, previous_scale, cond_mask=selected
                                )
                                rr.vstore(target32, subslot * 16, data, mask16)
                                # Vector stores keep a 32-byte-aligned base.
                                rr.vstore(target32, 128, scales, mask8)
                                rr.vmem_bar("vst_vld")
                    mem_copy(
                        tile_slice(cache, (1, 1, 544), (page, group, 0)).view(1, 544),
                        groups.next_read(),
                    )


@jit
def unpack_combined_cache(
    cache: Tensor, table: Tensor, used_k: Tensor, key_address, scale_address, capacity
):
    """Unpack active pages to logical-order private GM planes.

    All AIVs reach the caller's following global publication barrier.
    Byte copies preserve E2M1 and E8M0 exactly.
    """
    aiv = get_block_idx() * get_subblock_dim() + get_subblock_id()
    aivs = get_block_num() * get_subblock_dim()
    count = (min(max(used_k[0], 0), capacity) + 127) // 128
    key = make_tensor(
        make_pointer(dtypes.uint8, dtypes.int64(key_address), MemLoc.GM),
        make_layout((capacity // 128, 16, 512), stride=(8192, 512, 1)),
    )
    scale = make_tensor(
        make_pointer(dtypes.uint8, dtypes.int64(scale_address), MemLoc.GM),
        make_layout((capacity // 128, 16, 32), stride=(512, 32, 1)),
    )
    keys = RingBuffer(MemLoc.UB, dtypes.uint8, make_layout((16, 512)), depth=2)
    scales = RingBuffer(MemLoc.UB, dtypes.uint8, make_layout((16, 32)), depth=2)
    for logical in range(aiv, count, aivs):
        physical = dtypes.int64(table[0, logical])
        page = tile_slice(cache, (1, 16, 544), (physical, 0, 0)).view(16, 544)
        # Gather the two strided GM planes into contiguous UB buffers.  The
        # following writes are contiguous; no strided UB-to-GM copy is needed.
        mem_copy(
            keys.next_write(),
            page[None, 0:512],
        )
        mem_copy(
            scales.next_write(),
            page[None, 512:544],
        )
        mem_copy(
            tile_slice(key, (1, 16, 512), (logical, 0, 0)).view(16, 512),
            keys.next_read(),
        )
        mem_copy(
            tile_slice(scale, (1, 16, 32), (logical, 0, 0)).view(16, 32),
            scales.next_read(),
        )


def _validate_operator_geometry(latent, wk, rope_sin, storage_mode):
    """Layer 1: operator geometry (T/H/D/Dr and the storage mode)."""
    total_rows, head_dim = map(int, latent.shape)
    index_head_dim = int(wk.shape[0])
    rope_head_dim = int(rope_sin.shape[1])
    if total_rows <= 0:
        raise ValueError("T must be positive")
    if head_dim <= 0 or head_dim % NZ_FRACTAL_ALIGN != 0:
        raise ValueError(
            f"H must be a positive multiple of {NZ_FRACTAL_ALIGN} for FRACTAL_NZ wk"
        )
    if index_head_dim < FP4_GROUP_SIZE or index_head_dim % FP4_GROUP_SIZE != 0:
        raise ValueError(f"D must be a positive multiple of {FP4_GROUP_SIZE}")
    if rope_head_dim <= 0 or rope_head_dim % ROPE_PAIR_SIZE != 0:
        raise ValueError("Dr must be a positive even number")
    if rope_head_dim > index_head_dim:
        raise ValueError("Dr cannot exceed D")
    if storage_mode not in (0, 1):
        raise ValueError("storage_mode must be 0 or 1")
    return total_rows, head_dim, index_head_dim, rope_head_dim


def _validate_cache_layout(k_cache, k_scale_cache, storage_mode=1):
    """Validate the selected public cache ABI; no implicit conversion."""
    if storage_mode not in (0, 1):
        raise ValueError("storage_mode must be 0 or 1")
    if storage_mode == 0:
        if k_scale_cache is None:
            raise ValueError("storage_mode=0 requires k_scale_cache")
        for name, tensor, width in (
            ("k_cache", k_cache, 64),
            ("k_scale_cache", k_scale_cache, 4),
        ):
            if tensor.dtype != torch.uint8:
                raise TypeError(f"{name} must be UINT8")
            if (
                tensor.ndim != 4
                or tuple(tensor.shape[1:]) != (128, 1, width)
                or tensor.shape[0] < 1
            ):
                raise ValueError(f"{name} must have shape [P,128,1,{width}]")
            if (
                tensor.stride(1) != width
                or tensor.stride(2) != width
                or tensor.stride(3) != 1
                or tensor.stride(0) < 128 * width
            ):
                raise ValueError(
                    f"{name} requires contiguous page interiors and non-overlapping pages"
                )
        if (
            k_scale_cache.shape[0] != k_cache.shape[0]
            or k_scale_cache.device != k_cache.device
        ):
            raise ValueError("K and scale caches must share the page count and device")
        return
    if k_scale_cache is not None:
        raise ValueError("k_scale_cache must be None; scales are embedded in k_cache")
    if k_cache.dtype != torch.uint8:
        raise TypeError("k_cache must be UINT8")
    if k_cache.ndim != 3 or tuple(k_cache.shape[1:]) != (16, 544):
        raise ValueError("k_cache must have shape [P,16,544] (page_size=128)")
    if k_cache.shape[0] < 1:
        raise ValueError("k_cache must contain at least one physical page")
    if k_cache.stride(2) != 1 or k_cache.stride(1) != 544 or k_cache.stride(0) < 8704:
        raise ValueError(
            "k_cache requires contiguous page interiors and non-overlapping pages"
        )


class DecodeKProjection:
    """W 后接 K 投影：逻辑任务16–19 各拥有一个 N32 tile，按 K 递增累计。"""

    @jit
    def kernel(self, latent, weight, projected, block_offset=16, output_task=None):
        a = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((16, 256)),
            depth=2,
            data_format="nz",
        )
        b = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((32, 256)),
            depth=2,
            data_format="nz",
        )
        aa = RingBuffer(MemLoc.L0A, dtypes.bfloat16, make_layout((16, 128)), depth=2)
        bb = RingBuffer(MemLoc.L0B, dtypes.bfloat16, make_layout((32, 128)), depth=2)
        c = RingBuffer(MemLoc.L0C, dtypes.float32, make_layout((16, 32)), depth=1)
        owner = get_block_idx() - block_offset if output_task is None else output_task
        acc = c.next_write()
        for window in range_constexpr(2):
            mem_copy(a.next_write(), tile_slice(latent, (16, 256), (0, window)))
            mem_copy(b.next_write(), tile_slice(weight, (32, 256), (owner, window)))
            ar = a.next_read()
            br = b.next_read()
            for step in range_constexpr(2):
                mem_copy(aa.next_write(), tile_slice(ar, (16, 128), (0, step)))
                mem_copy(bb.next_write(), tile_slice(br, (32, 128), (0, step)))
                matmul(
                    acc, aa.next_read(), bb.next_read(), init=window == 0 and step == 0
                )
        # FIXPIPE uses the source domain in DSL 0.8. Clip live rows explicitly:
        # the accumulator is M16, while the public projected buffer is only T rows.
        destination = tile_slice(projected, (16, 32), (0, owner))
        mem_copy(destination, c.next_read()[: destination.shape[0], :32])
