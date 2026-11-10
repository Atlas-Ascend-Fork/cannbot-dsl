# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Batched matmul with transposed input/output addressing (aligns the A5
mainline aclnnTransposeBatchMatMul).

Math: r[b, m, n] = sum_k A[b, m, k] * Bmat[b, k, n], with the output stored
under permY=[1, 0, 2] (no materializing transpose):

    bsf = 1:            out[m, b, n]      = r[b, m, n]      # out (M, B, N)
    bsf > 1 (inner=B//bsf): out[s, m, j*N+n] = r[s*inner + j, m, n]
                                             # out (bsf, M, B*N//bsf)

All transposition semantics ride on strided GM views (getitem + tile_slice):
the A skip-read (perm_x1=[1,0,2], row pitch B*K), the B K-major permute path
(perm_x2=[0,1,2]), the WeightNZ direct fractal read (x2_format="nz") and the
C skip-write (row pitch B*N / inner*N) never copy whole tensors on the host.

Structure: TransposeBatchMatMulTiling (host tiling)
        -> TransposeBatchMatMulKernel (@kernel scheduler + @jit tile body)
        -> transpose_batch_mat_mul() (torch API, aclnn 10-param signature).

Mainline deltas (documented in README.md): cube_math_type=1 (USE_FP16) is
rejected (no DSL cube entry); type 3 (USE_HF32_HIGH_PRECISION) routes to
HF32RoundingMode.NEAREST_AWAY.

Tail-free db=1 (DROP-domain) grids run the MMAD unit_flag chain (2
while accumulating, 3 on the final K block) with the FIXPIPE consuming the
produce alias — the mainline enUnitFlag analog: the hardware auto-triggers
the FIXPIPE at the final MMAD, removing the db=1 drain serialization
(measured: -7.5% duration, mac ratio 0.83 -> 0.98). Enabled exactly by the
tiling's use_unit_flag gate (tail-free db=1 only; db=2 or M/N-tail pairings
hang or corrupt in that mode).

The K reduction is bounded and clipped at the c0 granularity (c0 =
32//dtype_bytes: 16 for fp16/bf16 — zero behavioral change — and 8 for
fp32). The nd2nz on-the-fly zero fill only covers a K tail to ceil_c0, so
a 16-granular bound/clip would read a deterministic 8-element dirty gap
for fp32 (KEEP_DTYPE and HF32 alike) whenever K%16 in [1..8]
(ceil8(K) < ceil16(K)), corrupting ~100% of the output elements. On the
K-major B side the fp32 corner carries one c0-wide tail block on the
L1->L0B permute copy (origins stay 16-aligned — the IR rejects
sub-16-aligned origins).
"""

__all__ = ["transpose_batch_mat_mul"]

import dataclasses

import torch
import torch_npu

from cannbotdsl import dtypes, get_mem_size, get_platform_info, make_layout
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.cube import (
    HF32RoundingMode,
    enable_hf32,
    set_fp32_mode,
    set_hf32_round_mode,
)
from cannbotdsl.ops.matmul import matmul as dsl_matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import MemLoc, Tensor, ceil_div, idx2crd, tile_slice


# ============================================================================
# Helpers
# ============================================================================


def _ceil_div(a: int, b: int) -> int:
    """Host-side integer ceil-div (the DSL ``ceil_div`` is kernel-side only)."""
    if b <= 0:
        raise ValueError(f"divisor must be positive; got {b}")
    return (a + b - 1) // b


def _ceil_align(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError(f"alignment divisor must be positive; got {b}")
    return (a + b - 1) // b * b


def _last_pow2(a: int) -> int:
    """Largest power of two not exceeding ``a`` (a >= 1)."""
    if a < 1:
        raise ValueError(f"last_pow2 requires a >= 1; got {a}")
    return 1 << (a.bit_length() - 1)


def _is_canonical_transpose_view(t: torch.Tensor) -> bool:
    """True iff ``t`` is a compact ``transpose(-1, -2)`` view of rank-3 storage."""
    if t.dim() != 3:
        return False
    if not (t.stride(-2) == 1 and t.stride(-1) == t.size(-2)):
        return False
    return t.stride(0) == t.size(-2) * t.size(-1) and t.is_contiguous() is False


_TORCH_DTYPE_TO_DSL = {
    torch.float16: (dtypes.float16, 16, 2),
    torch.bfloat16: (dtypes.bfloat16, 16, 2),
    torch.float32: (dtypes.float32, 8, 4),
}

_PERM_X1_VALUES = ((0, 1, 2), (1, 0, 2))
_PERM_X2_VALUES = ((0, 1, 2), (0, 2, 1))

# L0 feasibility cap on base_m*base_n (mainline baseM*baseN <= 32768 fp32 L0C
# slots) — the downshift is decided per the parallelism guardrail below
# (fp32/HF32 always keep it; fp16/bf16 only below the _R4_THETA tile count)
# — plus the pow-2 base_k fallbacks and the k_l1 staging bounds.
_BASIC_BLOCK = 16
_MN_CAP = 256
_L0C_TILE_CAP = 32768
# Guardrail multiplier: fp16/bf16 drop the L0C-cap downshift only when
# the no-downshift tile count T_nd = B*ceil(M/base_m)*ceil(N/base_n_pre)
# (pre-downshift base sizes) reaches _R4_THETA * AIC_NUM (= 96 tiles on the
# 32-core 950). Calibration over the perf matrix: every regression sits in
# the T_nd <= 64 cluster
# (+4.5%~+86%: parallel starvation below the core count, and db=1
# serialization on B-heavy short-K kernels at full occupancy) and every F1
# win in the T_nd >= 128 cluster (-5.5%~-40%); the clusters separate cleanly
# inside (64, 128], so theta in (2, 4] is decision-equivalent — theta=3
# (threshold ~ sqrt(64*128) ~= 90.5, the geometric midpoint) keeps the
# largest margin to both edges: "no-downshift tiling keeps >= 3 tiles per
# core" (2 tiles/core measured +22~24% regression, 4 tiles/core -5.5%~-40%
# win).
_R4_THETA = 3
# base_k candidates stay 16-multiples: the K-major L1 staging chunks
# its K axis into 16-wide hardware fractals, and the L1->L0B permute copy
# REJECTS any source origin that is not 16-aligned (IR verifier: "L1->L0
# source origin crosses a hardware fractal"; an 8-granular base_k was tried
# and rejected at compile time), so a c0-granular base_k is not a legal
# spelling. In the fp32 K%16 in [1..8] domain the K-major divisor filter
# therefore finds no divisor and falls back to 16, leaving one c0-wide
# (8-element) K tail on the permute copy at a 16-aligned origin — the
# residual-risk spelling this fix must carry (see the note in
# _k_reduction's K-major branch).
_BASE_K_CANDIDATES = (128, 64, 32, 16)
_MAX_STEP_K = 8
_DB_SIZE = 2
_DATA_SIZE_FP32 = 4
# Double guard on the divisible step_k pick: k_l1 must
# stay >= 128 elements (>= 256B staging rows at fp16) and >= half of the
# unguarded maximum, else the scan falls back to the plain largest step —
# K=304-class shapes would otherwise pick a degenerate 16.
_K_L1_GUARD_ELEMS = 128

_L2_CAPACITY_DEFAULT_BYTES = 128 * 1024 * 1024
_L2_BYPASS_ALIGN_ELEMS = 128

# Mainline ND2NZ inline-conversion inner-axis cap (A5), kept for parity even
# though the DSL engine itself does not enforce it.
_MAINLINE_INNER_AXIS_CAP = 65536


def _require_nd3(t: torch.Tensor, name: str) -> None:
    if t.dim() != 3:
        raise ValueError(f"{name} must be a 3-D tensor; got shape {tuple(t.shape)}")


def _require_contiguous(t: torch.Tensor, name: str) -> None:
    if not t.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous; got shape={tuple(t.shape)}, "
            f"stride={tuple(t.stride())}"
        )


# ============================================================================
# 1. Host-side tiling
# ============================================================================


class TransposeBatchMatMulTiling:
    """Host tiling for ascend950 (dav-3510), mirroring the mainline
    CalcBaseMN/DoCommonTiling upper bounds — parallelism-guardrail rules:

    - base_m/base_n = ceil_align(M/N, 16) capped at 256. The L0C cap
      base_m*base_n <= 32768 downshift is decided by the guardrail:
      fp32/HF32 keep it unconditionally (fp32/HF32 tiling structurally
      unchanged); fp16/bf16 keep it while the no-downshift tile count
      T_nd = B*ceil(M/base_m)*ceil(N/base_n) stays below _R4_THETA*AIC_NUM
      (the near-core-count band — the halved tile count either starves
      cores or exposes the db=1 serialization, both measured regressions),
      and drop it otherwise (the 256x256 tiles halve n_tiles, cutting the
      A-side GM re-read bytes by ~33%, measured wins);
    - base_k <= min(L0A, L0B)/2/dtype_size/max(base_m, base_n), c0-aligned
      (c0 = 32//dtype_bytes — 16 for fp16/bf16, 8 for fp32; the
      base_k = k_aligned shortcut may pick an 8-aligned fp32 K bound);
    - k_l1 = base_k * step_k (step_k <= 8) under a double-buffered L1 budget.
      fp16/bf16 prefer the largest feasible step whose k_l1 divides
      k_aligned = ceil_c0(K) exactly (no K segment tail) when it passes the
      double guard (k_l1 >= 128 elements and >= half of the unguarded
      maximum), else the plain largest feasible step; fp32 storage always
      takes the largest feasible step (legacy behavior);
    - l0c_db = 2 while base_m*base_n*4*2 fits L0C, else 1 (formula
      unchanged; the guardrail picks the branch through the L0C coupling
      constraint: db=2 is feasible iff base_m*base_n <= 32768 iff the
      downshift fired — a no-downshift 256x256 fp32-accumulator slot pair
      would need 512KB > the 256KB physical L0C, so DROP-domain tiles are
      db=1 by necessity and KEEP-domain/fp32/small tiles are db=2);
    - use_unit_flag = (l0c_db == 1 and m_tail_act == base_m and
      n_tail_act == base_n): arms the MMAD unit_flag chain + produce-alias
      FIXPIPE (mainline enUnitFlag analog) exactly on tail-free db=1 grids:
      K tails, N-major, bsf, NZ and perm_x1 variants all pass bitwise, while
      the db=2 and M/N-tail pairings hang or corrupt in that mode.

    The batch dim is orthogonal to the tile sizes; bsf only re-derives the
    output merge view, never the tile grid (the guardrail's B_eff is the
    logical batch count). Tail quantities (K blocks, M/N tail actual
    extents) are host-computed so every kernel-side window stays a
    compile-time constant.
    """

    L1_SIZE = get_mem_size("l1")
    L0A_SIZE = get_mem_size("l0a")
    L0B_SIZE = get_mem_size("l0b")
    L0C_SIZE = get_mem_size("l0c")

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype=torch.float16,
        *,
        kern_trans_b: bool = True,
        b: int,
    ):
        stream = torch.npu.current_stream()
        self.AIC_NUM = int(get_platform_info(stream=stream).cube_core_num)
        self.m, self.n, self.k = m, n, k
        self.b = b
        self.kern_trans_b = kern_trans_b
        props = _TORCH_DTYPE_TO_DSL.get(dtype)
        if props is None:
            raise TypeError(
                f"dtype only supports torch.float16/bfloat16/float32; got {dtype}"
            )
        self.dtype, self.c0, self.dtype_size = props
        self.is_fp32 = self.dtype == dtypes.float32
        self._compute()

    def __repr__(self):
        return (
            f"TransposeBatchMatMulTiling(M={self.m}, N={self.n}, K={self.k})\n"
            f"  base_m={self.base_m}, base_n={self.base_n}, base_k={self.base_k}\n"
            f"  k_l1={self.k_l1}, step_k={self.step_k}, l0c_db={self.l0c_db}\n"
            f"  m_tiles={self.m_tiles}, n_tiles={self.n_tiles}, "
            f"k_l1_tiles={self.k_l1_tiles}"
        )

    def _compute(self):
        ds = self.dtype_size
        self.base_m = min(_ceil_align(self.m, _BASIC_BLOCK), _MN_CAP)
        self.base_n = min(_ceil_align(self.n, _BASIC_BLOCK), _MN_CAP)
        # Parallelism guardrail. fp32/HF32 keeps the legacy L0C-cap
        # downshift unconditionally (structurally unchanged). fp16/bf16 drop
        # it only when the no-downshift tiling keeps the cube grid fed:
        # below _R4_THETA*AIC_NUM tiles lies the near-core-count band where
        # the halved tile count either starves cores (T_nd < 32) or exposes
        # the db=1 FIXPIPE-vs-mmad serialization (B-heavy short-K kernels) —
        # both measured regressions; at/above it the 256x256 tiles halve
        # n_tiles and cut the A-side GM re-read bytes by ~33%. T_nd uses
        # the pre-downshift base sizes (bsf never changes
        # the tile grid, so B_eff = B).
        need = self.base_m * self.base_n > _L0C_TILE_CAP
        t_nd = self.b * _ceil_div(self.m, self.base_m) * _ceil_div(self.n, self.base_n)
        if need and (self.is_fp32 or t_nd < _R4_THETA * self.AIC_NUM):
            self.base_n = min(self.base_n, _last_pow2(_L0C_TILE_CAP // self.base_m))

        max_k = (
            min(self.L0A_SIZE, self.L0B_SIZE)
            // _DB_SIZE
            // ds
            // max(self.base_m, self.base_n)
        )
        # The kernel runs its K loop to k_aligned = ceil_c0(K) (c0 =
        # 32//dtype_bytes: 16 for fp16/bf16 — unchanged — and 8 for fp32).
        # The GM tiles are tail-aware so the A/B reads clip to the real K,
        # and the nd2nz on-the-fly zero fill covers the K tail only to the
        # c0 boundary (the fill granularity is c0 on the staging column
        # axis, not 16) — running the loop to ceil16(K) for fp32 with
        # K%16 in [1..8] reads the deterministic dirty gap
        # [ceil8(K), ceil16(K)) out of L1 and corrupts ~100% of the
        # outputs. For the K-major B side (perm_x2=
        # [0,1,2] ND and every NZ input) base_k should divide k_aligned: a
        # partial K block on the L1->L0B transpose copy intermittently
        # corrupted scattered l0b rows on dav-3510, which is why the
        # candidate list is filtered to divisors of
        # k_aligned (16 always divides the 16-bit k_aligned). The fp32
        # K%16 in [1..8] domain has no 16-multiple divisor of its
        # 8-aligned k_aligned — it falls back to base_k=16 with one
        # c0-wide tail whose origin stays 16-aligned (sub-16-aligned
        # origins are IR-rejected). The wrapper zero-pads the K-major ND
        # B operand to ceil_c0(K), so that live tail reads real zeros.
        k_aligned = _ceil_align(self.k, self.c0)
        if self.kern_trans_b:
            if k_aligned <= max_k:
                self.base_k = k_aligned
            else:
                self.base_k = next(
                    (c for c in _BASE_K_CANDIDATES if c <= max_k), _BASIC_BLOCK
                )
        else:
            if k_aligned <= max_k:
                self.base_k = k_aligned
            else:
                self.base_k = next(
                    (
                        c
                        for c in _BASE_K_CANDIDATES
                        if c <= max_k and k_aligned % c == 0
                    ),
                    _BASIC_BLOCK,
                )
        self.k_aligned = k_aligned

        # step_k scan: track s_max (the largest L1-feasible step — the
        # legacy pick, kept unconditionally for fp32 storage) and s_div (the
        # largest feasible step whose k_l1 = base_k*s divides k_aligned
        # exactly, i.e. no K segment tail). fp16/bf16 take s_div when it
        # passes the double guard — k_l1 >= 128 elements (staging rows stay
        # >= 256B at fp16) and k_l1 >= half of the unguarded maximum (a
        # K=304-class shape would otherwise pick a degenerate 16) — else
        # fall back to s_max, and the K segment tail keeps the k_eff
        # semantics of the bounded loop.
        max_step = min(_ceil_div(k_aligned, self.base_k), _MAX_STEP_K)
        s_max = 1
        s_div = None
        for s in range(1, max_step + 1):
            a_l1 = self.base_m * self.base_k * s * ds
            b_l1 = self.base_n * self.base_k * s * ds
            if (a_l1 + b_l1) * _DB_SIZE > self.L1_SIZE:
                break
            s_max = s
            if not self.is_fp32 and k_aligned % (self.base_k * s) == 0:
                s_div = s
        if (
            not self.is_fp32
            and s_div is not None
            and self.base_k * s_div >= _K_L1_GUARD_ELEMS
            and 2 * self.base_k * s_div >= self.base_k * s_max
        ):
            step_k = s_div
        else:
            step_k = s_max
        self.step_k = step_k
        self.k_l1 = self.base_k * step_k
        # L0C ring depth (which branch fires is decided by the guardrail
        # above through the L0C coupling constraint): 2 (ping-pong across
        # tiles) while
        # the fp32 accumulator's double slot fits L0C — the KEEP domain,
        # fp32 storage and small shapes; 1 otherwise — the DROP domain's
        # 256x256 tiles, where a second fp32 slot would need 512KB > the
        # 256KB physical L0C (db=1 is a necessity, not a choice). The
        # cross-tile FIXPIPE drain vs the next tile's first mmad is then
        # serialized by the DSL auto sync barrier while the next tile's
        # MTE2/MTE1 keep flowing (measured fixpipe exposure ~0 in the DROP
        # domain, mac_ratio 0.90-0.98). That residual is NOT ~0 on
        # B-heavy short-K grids (16.6% cube idle, mac ratio 0.83 on a
        # 128x512x128 case) — the tail-free db=1 sub-domain runs the
        # unit_flag chain instead (use_unit_flag below).
        self.l0c_db = (
            _DB_SIZE
            if self.base_m * self.base_n * _DATA_SIZE_FP32 * _DB_SIZE <= self.L0C_SIZE
            else 1
        )

        self.m_tiles = _ceil_div(self.m, self.base_m)
        self.n_tiles = _ceil_div(self.n, self.base_n)
        # The K reduction runs to the c0-aligned bound (zero-fill
        # semantics; see _compute's k_aligned note). k_l1_tiles is derived
        # from it.
        self.k_l1_tiles = _ceil_div(k_aligned, self.k_l1)
        # Host-known tail actual extents (compile-time constants kernel-side).
        self.m_tail_act = self.m - (self.m_tiles - 1) * self.base_m
        self.n_tail_act = self.n - (self.n_tiles - 1) * self.base_n
        # unit_flag output-contract field: the b2 spelling — MMAD
        # unit_flag chain (2 accumulate / 3 final K
        # block) + FIXPIPE produce-alias consume — fires exactly on tail-free
        # db=1 grids. Every term is a measured prohibition: db==1 only (the
        # db=2 pairing hangs with a first-call aicore timeout, and db=2
        # already overlaps the FIXPIPE drain, no increment); M/N-tail
        # free only (the clipped-alias FIXPIPE deadlocks on M/N-tail
        # tiles with an aicore timeout, and the unclipped
        # batch_matmul-style alias silently corrupts them).
        # K tails are measured SAFE (the is_last_k_block chain folds them;
        # N-major/bsf/NZ/perm_x1 structures pass bitwise inside the gate).
        # Pure host arithmetic; boundary measured on both sides
        # (uf_boundary_grid: 14+ positive/negative shapes).
        self.use_unit_flag = (
            self.l0c_db == 1
            and self.m_tail_act == self.base_m
            and self.n_tail_act == self.base_n
        )


# ============================================================================
# 2. Kernel
# ============================================================================


@dataclasses.dataclass
class _KernelCtx:
    """Launch context bundling the wrapper's layout decisions."""

    b: int  # logical batch count
    m: int
    n: int
    k: int
    bsf: int
    kern_trans_a: bool  # x1 physical (M, B, K) -> A skip-read
    kern_trans_b: bool  # x2 physical (B, N, K) ND direct path
    nz_input: bool  # x2 is FRACTAL_NZ logical (B, K, N)
    hf32: bool  # fp32 storage with HF32 compute
    hf32_round_away: bool  # HF32RoundingMode.NEAREST_AWAY (cube_math_type=3)


class TransposeBatchMatMulKernel:
    """out[m, b, n] = A[b] @ B[b]^T with strided GM addressing.

    Operands arrive as: x1 ``[B, M, K]`` (or ``[M, B, K]`` when
    ``kern_trans_a``), x2 ``[B, N, K]`` (or ``[B, K, N]`` per
    ``kern_trans_b``; FRACTAL_NZ when ``nz_input``), out 3-D ``[M, B, N]``
    (bsf == 1) or 4-D ``[bsf, M, inner, N]`` (bsf > 1).

    GM -> L1 (MTE2, nd2nz or identity NZ->NZ, ping-pong)
      -> L0A/L0B (MTE1, double buffer; K-major B sides spell the transpose as
      ``src.permute(1, 0)``) -> MMAD (fp32 L0C) -> GM (FIXPIPE).

    M/N tails: the nd2nz zero-fill keeps the actual window correct inside the
    nominal slots, the MMAD runs at nominal extents (its m/n windows cannot
    be runtime-clipped) and the FIXPIPE source is clipped to the host-known
    actual extent while the GM destination tile is tail-aware. K tails clip
    the L1->L0 windows to ceil_c0(k_blk) (c0 = 32//dtype_bytes — the
    nd2nz fill granularity; 16 for fp16/bf16, 8 for fp32).

    On tail-free db=1 grids (tiling use_unit_flag) the MMADs carry the
    unit_flag chain (2 while accumulating, 3 on the final K block) and the
    FIXPIPE consumes the produce alias — the hardware auto-triggers the
    drain at the final MMAD (mainline enUnitFlag analog, removes the db=1
    FIXPIPE-drain serialization); every other grid keeps the auto sync
    barrier + next_read() spelling byte-for-byte.
    next_read() spelling byte-for-byte.
    """

    def __init__(self, tiling: TransposeBatchMatMulTiling, ctx: _KernelCtx):
        self.t = tiling
        self.ctx = ctx
        self.bsz = ctx.b
        self.bsf = ctx.bsf
        self.inner = ctx.b // ctx.bsf
        self.kern_trans_a = ctx.kern_trans_a
        self.kern_trans_b = ctx.kern_trans_b
        self.nz_input = ctx.nz_input
        self.hf32 = ctx.hf32
        self.hf32_round_away = ctx.hf32_round_away
        self.m_tiles = tiling.m_tiles
        self.n_tiles = tiling.n_tiles
        self.mn_tiles = self.m_tiles * self.n_tiles
        self.total_tiles = ctx.b * self.mn_tiles
        self.k_l1_tiles = tiling.k_l1_tiles
        self.used_core_num = min(self.total_tiles, tiling.AIC_NUM)
        # Tail switches are compile-time: a tail exists whenever the last
        # tile's actual extent is below the nominal size — including the
        # single-tile case where the shape itself is smaller than the
        # 16-aligned base (e.g. N=1 with base_n=16).
        self.m_tail_act = tiling.m_tail_act
        self.n_tail_act = tiling.n_tail_act
        self.has_m_tail = tiling.m_tail_act < tiling.base_m
        self.has_n_tail = tiling.n_tail_act < tiling.base_n
        # Tile visit order over the (b, m, n) grid: "bnm" keeps m
        # innermost, so one core pins its
        # (b, m) A-slice in L2 and streams B n-slices — measured 0.92x Task
        # Duration vs the n-innermost "bmn" order on GEMM-4k fp16
        # (mac_ratio 0.94 vs 0.87) and 0.95x on hf32. The mnb (0.99x vs
        # bmn) and mbn (1.23x, batch cycled fastest — no reuse survives)
        # permutations were also measured; bnm is the keeper. Semantically
        # transparent (per-tile compute is order-independent); only L2
        # locality moves.
        self.tile_order = "bnm"

        # L2 cache-hint policy (asc_load/store_l2_cache_mode: 0/1/2 are
        # cached modes with different victim priority, 4+ bypass; the DSL
        # copy control accepts 0/1/2/4). A/B are FIRST_VICTIM (0) only on the
        # compact linear no-reuse path; everywhere else (tile re-reads /
        # non-nd / unaligned) they take LAST_VICTIM (1) — the re-reads are
        # served from L2, and NOTALLOC on A/B measured 1.35x slower on
        # GEMM-4k. C is cached (1) while the whole output
        # fits L2; once it exceeds L2 the streaming write switches to
        # NOTALLOC_CLEAN (4) so it stops evicting A/B re-read lines —
        # measured 0.95x Task Duration on GEMM-4k fp16 / 0.94x hf32
        # measurements), neutral-to-negative on L2-resident outputs.
        aligned = (tiling.k % _L2_BYPASS_ALIGN_ELEMS == 0) and (
            tiling.k_l1 % _L2_BYPASS_ALIGN_ELEMS == 0
        )
        nd_path_a = not ctx.kern_trans_a
        nd_path_b = ctx.kern_trans_b and not ctx.nz_input
        reuse_a = self.n_tiles > 1
        reuse_b = self.m_tiles > 1
        self.l2_ctl_a = 1 if (reuse_a or not nd_path_a or not aligned) else 0
        self.l2_ctl_b = 1 if (reuse_b or not nd_path_b or not aligned) else 0
        self.l2_ctl_c = (
            1
            if self.bsz * tiling.m * tiling.n * tiling.dtype_size
            <= _L2_CAPACITY_DEFAULT_BYTES
            else 4
        )

        self.l1_a = self.l1_b = None
        self.l0a = self.l0b = self.l0c = None
        self.opt_a = copy_options(l2_cache_ctl=self.l2_ctl_a)
        self.opt_b = copy_options(l2_cache_ctl=self.l2_ctl_b)
        self.opt_c = copy_options(l2_cache_ctl=self.l2_ctl_c)
        # b2 unit-flag FIXPIPE options: the hardware-auto-triggered
        # FIXPIPE consumes the produce alias through a unit_flag=3 copy (a
        # ring next_read() would desync against it). Used only inside the
        # tiling-gated branch — the SAME predicate that arms the MMAD chain
        # (an options-only unit_flag without the MMAD chain is the rejected
        # b0 variant: uncharacterizable hangs, measured on tail-free small
        # geometries too). The stock branches keep opt_c untouched.
        self.opt_c_flag = (
            copy_options(unit_flag=3, l2_cache_ctl=self.l2_ctl_c)
            if self.t.use_unit_flag
            else self.opt_c
        )

    @kernel
    def tbmm_kernel(self, gm_x1: Tensor, gm_x2: Tensor, gm_out: Tensor):
        t = self.t
        if const_expr(t.is_fp32):
            if const_expr(self.hf32):
                enable_hf32()  # default NEAREST_EVEN
                if const_expr(self.hf32_round_away):
                    set_hf32_round_mode(HF32RoundingMode.NEAREST_AWAY)
            else:
                set_fp32_mode()
        self._create_buffers()
        for tile_idx in range(get_block_idx(), self.total_tiles, get_block_num()):
            self._compute_output_tile(gm_x1, gm_x2, gm_out, tile_idx)

    @jit
    def _create_buffers(self):
        """Declare the L1/L0 staging ring buffers. L1/L0A/L0B stay
        double-buffered; the L0C depth comes from the tiling
        (l0c_db follows the guardrail domain: 1 in the DROP domain —
        the fp16/bf16 256x256 tiles, where the cross-tile drain is
        serialized by the DSL auto sync barrier instead — and 2 in the
        KEEP domain / fp32 storage / small shapes)."""
        t = self.t
        self.l1_a = RingBuffer(
            MemLoc.L1,
            t.dtype,
            make_layout((t.base_m, t.k_l1)),
            depth=_DB_SIZE,
            data_format="nz",
        )
        # (base_n, k_l1) N-major stages feed l0b directly (matmul rhs is
        # (N, K)); K-major stages — perm_x2=[0,1,2] ND and every NZ input —
        # spell the transpose as src.permute(1, 0) on the L1->L0B copy.
        l1_b_shape = (t.base_n, t.k_l1) if t.kern_trans_b else (t.k_l1, t.base_n)
        self.l1_b = RingBuffer(
            MemLoc.L1,
            t.dtype,
            make_layout(l1_b_shape),
            depth=_DB_SIZE,
            data_format="nz",
        )
        self.l0a = RingBuffer(
            MemLoc.L0A, t.dtype, make_layout((t.base_m, t.base_k)), depth=2
        )
        self.l0b = RingBuffer(
            MemLoc.L0B, t.dtype, make_layout((t.base_n, t.base_k)), depth=2
        )
        self.l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((t.base_m, t.base_n)),
            depth=t.l0c_db,  # 1 in the guardrail DROP domain, else 2
        )

    @jit
    def _compute_output_tile(self, gm_x1, gm_x2, gm_out, tile_idx):
        """Decode the tile index, build the strided views, run the K
        reduction and store through the FIXPIPE for one output tile."""
        ctx = self.ctx
        if const_expr(self.bsf > 1):
            s_idx, j_idx, m_idx, n_idx = idx2crd(
                tile_idx, [self.bsf, self.inner, self.m_tiles, self.n_tiles]
            )
            b_idx = s_idx * self.inner + j_idx
            mn_idx = m_idx * self.n_tiles + n_idx
            c2d = gm_out[s_idx, None, j_idx, None].reshape(ctx.m, ctx.n)
        elif const_expr(self.tile_order == "bmn"):
            # Alternative order (n innermost).
            b_idx, m_idx, n_idx = idx2crd(
                tile_idx, [self.bsz, self.m_tiles, self.n_tiles]
            )
            mn_idx = m_idx * self.n_tiles + n_idx
            c2d = gm_out[None, b_idx, None].reshape(ctx.m, ctx.n)
        else:
            # Default "bnm": m innermost (see __init__ for the measured
            # rationale).
            b_idx, n_idx, m_idx = idx2crd(
                tile_idx, [self.bsz, self.n_tiles, self.m_tiles]
            )
            mn_idx = m_idx * self.n_tiles + n_idx
            c2d = gm_out[None, b_idx, None].reshape(ctx.m, ctx.n)

        # A view: skip-read carries the B*K row pitch (P1: the keepdim unit
        # axis must be reshaped away before tile_slice).
        if const_expr(self.kern_trans_a):
            a2d = gm_x1[None, b_idx, None].reshape(ctx.m, ctx.k)
        else:
            a2d = gm_x1[b_idx, None, None]
        # B view: batch-select keeps (K, N)/(N, K) as the trailing axes for
        # all three physical layouts (ND N-major, ND K-major, NZ).
        b2d = gm_x2[b_idx, None, None]

        c_w = self.l0c.next_write()
        # c_view threads the accumulated L0C produce alias from the
        # K reduction into the store (unit-flag FIXPIPE consumption).
        c_view = self._k_reduction(a2d, b2d, c_w, mn_idx)
        self._store_c(c2d, mn_idx, c_view)

    @jit
    def _load_ab_l1(self, a2d, b2d, k_l1_idx, mn_idx):
        """GM -> L1 copies of one k_l1 segment for both operands (mn_idx
        packs (m_idx, n_idx) for the 5-arg coding rule)."""
        m_idx = mn_idx // self.n_tiles
        n_idx = mn_idx % self.n_tiles
        t = self.t
        gm_a_tile = tile_slice(a2d, (t.base_m, t.k_l1), (m_idx, k_l1_idx))
        mem_copy(self.l1_a.next_write(), gm_a_tile, options=self.opt_a)
        if const_expr(t.kern_trans_b):
            gm_b_tile = tile_slice(b2d, (t.base_n, t.k_l1), (n_idx, k_l1_idx))
        else:
            gm_b_tile = tile_slice(b2d, (t.k_l1, t.base_n), (k_l1_idx, n_idx))
        mem_copy(self.l1_b.next_write(), gm_b_tile, options=self.opt_b)

    @jit
    def _k_reduction(self, a2d, b2d, c_view, mn_idx):
        """GM -> L1 -> L0 K reduction and MMAD accumulation for one tile
        (mn_idx packs (m_idx, n_idx); unused here, kept for call symmetry).
        Returns the accumulated L0C view so the unit-flag FIXPIPE can
        consume the produce alias."""
        t = self.t
        base_m, base_n = t.base_m, t.base_n
        # The reduction runs to the c0-aligned K bound (k_aligned =
        # ceil_c0(K)); the GM tile reads are tail-aware so the last blocks
        # consume the nd2nz zero fill, never GM bytes past the real K.
        base_k, k_l1, k_total = t.base_k, t.k_l1, t.k_aligned

        # Static (unrolled) K loops: every tail quantity stays a compile-time
        # int, so the L1->L0 windows keep static extents and the init select
        # stays Python-level.
        for k_l1_idx in range_constexpr(self.k_l1_tiles):
            self._load_ab_l1(a2d, b2d, k_l1_idx, mn_idx)
            # One read per k_l1 segment: the inner loop slices the selected
            # slot (a read per slice would rotate the cursor off the slot).
            a_l1 = self.l1_a.next_read()
            b_l1 = self.l1_b.next_read()

            k_remaining = k_total - k_l1_idx * k_l1
            k_l0_per_l1 = ceil_div(min(k_l1, k_remaining), base_k)
            for k_l0_idx in range_constexpr(k_l0_per_l1):
                # K tail clip: the GM->L1 nd2nz fill covers the K tail only
                # to the c0 boundary (ceil(k_valid/c0)*c0 columns, c0 =
                # 32//dtype_bytes — 16 for fp16/bf16, 8 for fp32;
                # measured: the previous 16-granular assumption read a
                # deterministic dirty gap for fp32 K%16 in [1..8]), so the
                # final base_k block is copied/multiplied at that
                # c0-aligned extent.
                k_blk = min(base_k, k_total - k_l1_idx * k_l1 - k_l0_idx * base_k)
                k_eff = (k_blk + t.c0 - 1) // t.c0 * t.c0

                l1_a_slice = tile_slice(a_l1, (base_m, base_k), (0, k_l0_idx))
                if k_eff != base_k:
                    l1_a_slice = tile_slice(l1_a_slice, (base_m, k_eff), (0, 0))
                a_l0 = self.l0a.next_write()
                if k_eff != base_k:
                    a_l0 = tile_slice(a_l0, (base_m, k_eff), (0, 0))
                mem_copy(a_l0, l1_a_slice)

                is_first_k_block = k_l1_idx == 0 and k_l0_idx == 0
                # unit_flag chain (mainline enUnitFlag analog, b2
                # spelling): 2 while accumulating, 3 on the final K block —
                # the hardware auto-triggers the FIXPIPE at MMAD completion,
                # removing the db=1 FIXPIPE-drain serialization (measured
                # #30 mac_r 0.83 -> 0.98). Armed by the tiling's
                # use_unit_flag — the same tail-free db=1 gate that routes
                # _store_c through the produce alias (an MMAD chain without
                # the alias FIXPIPE, or vice versa, is a rejected spelling).
                unit_flag = 0
                if const_expr(self.t.use_unit_flag):
                    is_last_k_block = (
                        k_l1_idx == self.k_l1_tiles - 1 and k_l0_idx == k_l0_per_l1 - 1
                    )
                    unit_flag = 3 if is_last_k_block else 2
                if const_expr(t.kern_trans_b):
                    l1_b_slice = tile_slice(b_l1, (base_n, base_k), (0, k_l0_idx))
                    if k_eff != base_k:
                        l1_b_slice = tile_slice(l1_b_slice, (base_n, k_eff), (0, 0))
                    b_l0 = self.l0b.next_write()
                    if k_eff != base_k:
                        b_l0 = tile_slice(b_l0, (base_n, k_eff), (0, 0))
                    mem_copy(b_l0, l1_b_slice)
                    c_view = dsl_matmul(
                        c_view,
                        a_l0,
                        b_l0,
                        init=is_first_k_block,
                        unit_flag=unit_flag,
                    )
                else:
                    # K-major B side (perm_x2=[0,1,2] ND and every NZ input).
                    # The divisor-filtered base_k selection (see the tiling)
                    # keeps every 16-bit-dtype grid tail-free here, so the
                    # nested clip stays dormant on those: a partial K block
                    # on the L1->L0B transpose copy intermittently corrupted
                    # scattered l0b rows on dav-3510 (empirical).
                    # The one live tail is the fp32 K%16 in [1..8]
                    # domain: its 8-aligned k_aligned has no 16-multiple
                    # divisor, so base_k falls back to 16 and the final
                    # block is a c0-wide (8-element) clip at a 16-aligned
                    # origin — the only legal spelling (sub-16-aligned
                    # origins are IR-rejected). This path was hammered
                    # with repeated clean golden and determinism runs
                    # across 4 shapes. The wrapper additionally zero-pads
                    # the K-major ND B operand to ceil_c0(K) (real GM
                    # zeros), so the tail rows the staging reads are
                    # deterministic even when the L1 residue is not.
                    l1_b_slice = tile_slice(b_l1, (base_k, base_n), (k_l0_idx, 0))
                    if k_eff != base_k:
                        l1_b_slice = tile_slice(l1_b_slice, (k_eff, base_n), (0, 0))
                    b_l0 = self.l0b.next_write()
                    if k_eff != base_k:
                        b_l0 = tile_slice(b_l0, (base_n, k_eff), (0, 0))
                    mem_copy(b_l0, l1_b_slice.permute(1, 0))
                    c_view = dsl_matmul(
                        c_view,
                        a_l0,
                        b_l0,
                        init=is_first_k_block,
                        unit_flag=unit_flag,
                    )
        return c_view

    @jit
    def _store_c(self, c2d, mn_idx, c_view):
        """FIXPIPE store of one tile; M/N tails clip the L0C source window to
        the host-known actual extent while the GM destination stays
        tail-aware (MMAD windows cannot be runtime-clipped, so this is the
        only necessary and sufficient clipping).
        unit-flag path: the FIXPIPE consumes the produce alias returned
        by _k_reduction (a ring next_read() would desync against the
        hardware-auto-triggered drain). Full tile — the tiling gate
        guarantees M/N-tail-free grids here (an unclipped-alias
        batch_matmul-style spelling silently corrupts tail tiles, and the
        clipped-alias spelling deadlocks with an aicore timeout — both
        are rejected)."""
        m_idx = mn_idx // self.n_tiles
        n_idx = mn_idx % self.n_tiles
        t = self.t
        dst = tile_slice(c2d, (t.base_m, t.base_n), (m_idx, n_idx))
        if const_expr(self.t.use_unit_flag):
            # b2: produce-alias FIXPIPE through the unit_flag=3 options
            # (dst is the tail-aware tile_slice — a full tile under the gate).
            mem_copy(dst, c_view, options=self.opt_c_flag)
        elif const_expr(self.has_m_tail or self.has_n_tail):
            # Runtime tile-identity branches select the static window; every
            # path consumes its L0C read entirely inside the branch (region
            # values may not cross the scf.if boundary) and takes exactly one
            # l0c read so the rotation stays even.
            if m_idx == self.m_tiles - 1:
                if n_idx == self.n_tiles - 1:
                    src = tile_slice(
                        self.l0c.next_read(),
                        (self.m_tail_act, self.n_tail_act),
                        (0, 0),
                    )
                    mem_copy(dst, src, options=self.opt_c)
                else:
                    src = tile_slice(
                        self.l0c.next_read(),
                        (self.m_tail_act, t.base_n),
                        (0, 0),
                    )
                    mem_copy(dst, src, options=self.opt_c)
            else:
                if n_idx == self.n_tiles - 1:
                    src = tile_slice(
                        self.l0c.next_read(),
                        (t.base_m, self.n_tail_act),
                        (0, 0),
                    )
                    mem_copy(dst, src, options=self.opt_c)
                else:
                    src = tile_slice(self.l0c.next_read(), (t.base_m, t.base_n), (0, 0))
                    mem_copy(dst, src, options=self.opt_c)
        else:
            mem_copy(dst, self.l0c.next_read(), options=self.opt_c)

    @host
    def run(self, gm_x1, gm_x2, gm_out):
        self.tbmm_kernel[self.used_core_num](gm_x1, gm_x2, gm_out)


# ============================================================================
# 3. Torch interface
# ============================================================================


def _validate_dtypes(x1: torch.Tensor, x2: torch.Tensor) -> None:
    if x1.dtype != x2.dtype:
        raise TypeError(
            f"x1 and x2 must have the same dtype; got {x1.dtype} vs {x2.dtype}"
        )
    if x1.dtype not in _TORCH_DTYPE_TO_DSL:
        raise TypeError(
            "only torch.float16, torch.bfloat16 and torch.float32 are "
            f"supported; got {x1.dtype}"
        )
    for name, t in (("x1", x1), ("x2", x2)):
        if t.device.type != "npu":
            raise ValueError(f"{name} must be an NPU tensor; got {t.device}")
        if t.device != x1.device:
            raise ValueError(f"all inputs must share one device; got {t.device}")


def _validate_cube_math_type(cube_math_type: int, dtype: torch.Tensor.dtype) -> str:
    """Returns the compute mode: 'default' | 'hf32' (design point d)."""
    if not isinstance(cube_math_type, int) or cube_math_type < 0 or cube_math_type > 3:
        # The mainline silently maps 4 -> 0; this interface rejects it.
        raise ValueError(
            f"cube_math_type must be in 0..3 (KEEP_DTYPE/USE_FP16/USE_HF32/"
            f"USE_HF32_HIGH_PRECISION); got {cube_math_type!r}"
        )
    if cube_math_type == 1:
        raise ValueError(
            "cube_math_type=1 (USE_FP16) is not supported: the CANNBotDSL "
            "cube API has no fp32-storage/fp16-compute entry"
        )
    if cube_math_type in (2, 3):
        if dtype != torch.float32:
            raise ValueError(
                f"cube_math_type={cube_math_type} (HF32) is only valid for "
                f"float32 inputs; got {dtype}"
            )
        return "hf32_hp" if cube_math_type == 3 else "hf32"
    return "default"


def _bridge_nz_input(x2: torch.Tensor, b: int, k: int, n: int, c0: int) -> torch.Tensor:
    """Normalize a WeightNZ operand to a logical 3-D FRACTAL_NZ tensor.

    Accepts the mainline form (a FRACTAL_NZ tensor with logical shape
    (B, K, N), zero copy) or the 5-D fractal panel (B, N1, K1, 16, c0) from
    A panel over a FRACTAL_NZ allocation re-binds zero-copy via
    ``set_``; a panel over plain ND storage is bridged into a true
    FRACTAL_NZ allocation through the logical ND form (physical-layout
    bridges only, never a semantic transpose).
    """
    fmt = torch_npu.get_npu_format(x2)
    nz_fmt = torch_npu.Format.FRACTAL_NZ
    if x2.dim() == 3:
        if fmt != nz_fmt:
            raise ValueError(
                "x2_format='nz' with a 3-D x2 requires a FRACTAL_NZ tensor "
                "(npu_format_cast(x2, 29)); use x2_format='nd' otherwise"
            )
        return x2
    if x2.dim() != 5:
        raise ValueError(
            "x2_format='nz' requires a 3-D FRACTAL_NZ tensor or the 5-D "
            f"panel (B, N1, K1, 16, c0); got shape {tuple(x2.shape)}"
        )
    n1, k1 = n // c0, k // 16
    expected = (b, n1, k1, 16, c0)
    if tuple(x2.shape) != expected:
        raise ValueError(
            f"x2 NZ panel must have shape (B, N1, K1, 16, c0) = {expected} "
            f"for K={k}, N={n}, dtype={x2.dtype}; got {tuple(x2.shape)}"
        )
    if not x2.is_contiguous() or x2.storage_offset() != 0:
        raise ValueError("x2 NZ panel must be a contiguous storage-root view")
    bridged = torch_npu.unsafe_empty_with_format(
        (b, k, n), dtype=x2.dtype, device=x2.device, acl_format=29
    )
    bridged.set_(x2.untyped_storage(), 0, (b, k, n), (k * n, n, 1))
    if torch_npu.get_npu_format(bridged) != nz_fmt:
        # Plain ND storage: the set_ rebind lost the FRACTAL_NZ tag and
        # copy_ into an internal-format tensor is unsupported on Ascend950,
        # so bridge through the logical ND form instead: unfold the panel
        # (b, n1, k1, k0, n0) back to logical (b, k, n) and let
        # npu_format_cast produce the true FRACTAL_NZ allocation. Both moves
        # are physical-layout/format bridges, never semantic transposes.
        nd_logical = x2.permute(0, 2, 3, 1, 4).reshape(b, k, n)
        return torch_npu.npu_format_cast(nd_logical, 29)
    return bridged


def _normalize_x2_nd(x2: torch.Tensor, perm_x2, b: int, n: int, k: int) -> tuple:
    """Returns (x2_kern, kern_trans_b) for an ND operand (design point b)."""
    _require_nd3(x2, "x2")
    if perm_x2 == (0, 2, 1):
        # logical (B, N, K): a contiguous (B, N, K) storage takes the direct
        # N-major path; a canonical transpose(-1,-2) view is flipped back
        # zero-copy to physical (B, K, N). The shape check covers every
        # branch — the transpose flip must not bypass it.
        if tuple(x2.shape) != (b, n, k):
            raise ValueError(
                f"x2 shape must be (B, N, K) = {(b, n, k)} for "
                f"perm_x2=(0, 2, 1); got {tuple(x2.shape)}"
            )
        if _is_canonical_transpose_view(x2):
            return x2.transpose(-1, -2), False
        if x2.is_contiguous():
            return x2, True
        raise ValueError(
            "x2 must be contiguous or a canonical transpose(-1,-2) view; "
            f"got shape={tuple(x2.shape)}, stride={tuple(x2.stride())}"
        )
    # perm_x2 == (0, 1, 2): logical (B, K, N), K-major storage only.
    if not x2.is_contiguous():
        raise ValueError(
            f"x2 must be contiguous for perm_x2=(0, 1, 2); got "
            f"shape={tuple(x2.shape)}, stride={tuple(x2.stride())}"
        )
    if tuple(x2.shape) != (b, k, n):
        raise ValueError(
            f"x2 shape must be (B, K, N) = {(b, k, n)} for perm_x2=(0, 1, 2); "
            f"got {tuple(x2.shape)}"
        )
    return x2, False


def _resolve_out(out, m: int, b: int, n: int, bsf: int, dtype, device) -> torch.Tensor:
    inner = b // bsf
    if bsf == 1:
        expected = (m, b, n)
    else:
        expected = (bsf, m, inner * n)
    if out is None:
        return torch.empty(expected, dtype=dtype, device=device)
    if tuple(out.shape) != expected:
        raise ValueError(
            f"out shape must be {expected} for bsf={bsf}; got {tuple(out.shape)}"
        )
    if out.dtype != dtype:
        raise ValueError(f"out dtype must be {dtype}; got {out.dtype}")
    if not out.is_contiguous():
        raise ValueError(f"out must be contiguous; got stride={tuple(out.stride())}")
    if out.device != device:
        raise ValueError(f"out must be on {device}; got {out.device}")
    return out


def transpose_batch_mat_mul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias=None,
    scale=None,
    perm_x1=(0, 1, 2),
    perm_x2=(0, 1, 2),
    perm_y=(1, 0, 2),
    cube_math_type: int = 0,
    batch_split_factor: int = 1,
    out=None,
    *,
    x2_format: str = "nd",
) -> torch.Tensor:
    """Batched matmul with transposed GM addressing (A5 mainline alignment).

    r[b, m, n] = sum_k A[b, m, k] * Bmat[b, k, n]; the result is stored under
    permY=[1, 0, 2] (out[m, b, n], no materializing transpose).

    Args:
      x1: 3-D ND; with perm_x1=(0,1,2) shaped (B, M, K), with perm_x1=(1,0,2)
        shaped (M, B, K) (read through a B*K row-pitch view).
      x2: 3-D ND — (B, K, N) for perm_x2=(0,1,2), (B, N, K) for
        perm_x2=(0,2,1) (or a canonical transpose(-1,-2) view thereof) — or,
        with x2_format="nz", a FRACTAL_NZ (B, K, N) tensor / the 5-D panel
        (B, N1, K1, 16, c0); NZ needs K%16==0 and N%c0==0 and only supports
        perm_x1=(1,0,2) x perm_x2=(0,1,2) x batch_split_factor=1.
      bias, scale: must be None (mainline reservation / no quant dtype range).
      perm_y: only (1, 0, 2) is supported.
      cube_math_type: 0 KEEP_DTYPE, 2/3 HF32 (float32 only; 3 uses
        NEAREST_AWAY for type 3); 1 (USE_FP16) rejected.
      batch_split_factor: 1 <= bsf <= B with B % bsf == 0.
      out: optional pre-allocated output ((M, B, N) or (bsf, M, B*N//bsf)).

    Returns ``out``: (M, B, N) for bsf=1, else (bsf, M, B*N//bsf), same dtype
    as the inputs (fp32 accumulation inside the kernel).
    """
    # ---- parameter validation ----
    if perm_x1 not in _PERM_X1_VALUES:
        raise ValueError(f"perm_x1 must be (0,1,2) or (1,0,2); got {perm_x1}")
    if perm_x2 not in _PERM_X2_VALUES:
        raise ValueError(f"perm_x2 must be (0,1,2) or (0,2,1); got {perm_x2}")
    if tuple(perm_y) != (1, 0, 2):
        raise ValueError(f"perm_y only supports (1, 0, 2); got {perm_y}")
    if x2_format not in ("nd", "nz"):
        raise ValueError(f"x2_format must be 'nd' or 'nz'; got {x2_format!r}")
    if bias is not None:
        raise ValueError("bias is reserved on the A5 mainline and unsupported")
    if scale is not None:
        raise ValueError("scale (quantized output) is out of the dtype range")
    _validate_dtypes(x1, x2)
    _require_nd3(x1, "x1")
    if not x1.is_contiguous():
        raise ValueError(
            f"x1 must be contiguous; got shape={tuple(x1.shape)}, "
            f"stride={tuple(x1.stride())}"
        )
    compute_mode = _validate_cube_math_type(cube_math_type, x1.dtype)

    # ---- logical dims ----
    if perm_x1 == (1, 0, 2):
        m, bsz, k = (int(s) for s in x1.shape)
        kern_trans_a = True
    else:
        bsz, m, k = (int(s) for s in x1.shape)
        kern_trans_a = False
    if min(m, bsz, k) < 1:
        raise ValueError(f"empty tensors are unsupported; got M={m}, B={bsz}, K={k}")
    inner_axis = bsz * k if kern_trans_a else k
    if inner_axis >= _MAINLINE_INNER_AXIS_CAP:
        raise ValueError(
            f"inner-axis limit exceeded: {'B*K' if kern_trans_a else 'K'} = "
            f"{inner_axis} must be < {_MAINLINE_INNER_AXIS_CAP} (mainline "
            "ND2NZ conversion cap)"
        )

    # ---- bsf ----
    bsf = int(batch_split_factor)
    if not (1 <= bsf <= bsz) or bsz % bsf != 0:
        raise ValueError(
            f"batch_split_factor must divide B={bsz} and be in [1, {bsz}]; "
            f"got {batch_split_factor}"
        )

    # ---- x2 layout resolution (n follows from x2's logical shape) ----
    c0 = _TORCH_DTYPE_TO_DSL[x1.dtype][1]
    nz_input = x2_format == "nz"
    kern_trans_b = False
    if nz_input:
        if perm_x1 != (1, 0, 2) or perm_x2 != (0, 1, 2) or bsf != 1:
            raise ValueError(
                "x2_format='nz' only supports perm_x1=(1,0,2), "
                "perm_x2=(0,1,2) and batch_split_factor=1 (mainline "
                "'The current attrs does not support weightNZ.')"
            )
        if x1.dtype == torch.float32:
            # fp32 NZ storage is the FRACTAL_NZ_C0_16 variant, which the DSL
            # direct-launch runtime does not accept (only format-29
            # FRACTAL_NZ, i.e. 16-bit dtypes) — measured on dav-3510.
            raise ValueError(
                "x2_format='nz' supports float16/bfloat16 only: float32 NZ "
                "storage (FRACTAL_NZ_C0_16) is not accepted by the DSL "
                "runtime"
            )
        if k % 16 != 0 or k <= 1:
            raise ValueError(
                f"x2_format='nz' requires K % 16 == 0 and K > 1; got K={k}"
            )
        if x2.dim() == 3:
            if x2.shape[0] != bsz or x2.shape[1] != k:
                raise ValueError(
                    f"x2 logical shape must be (B, K, N) = ({bsz}, {k}, N); "
                    f"got {tuple(x2.shape)}"
                )
            n = int(x2.shape[2])
        elif x2.dim() == 5:
            n = int(x2.shape[1]) * c0
            if int(x2.shape[2]) * 16 != k:
                raise ValueError(
                    f"x2 NZ panel K1*16 must equal K={k}; got K1={int(x2.shape[2])}"
                )
        else:
            raise ValueError(
                "x2_format='nz' requires a 3-D FRACTAL_NZ tensor or the "
                f"5-D panel (B, N1, K1, 16, c0); got rank {x2.dim()}"
            )
        if n % c0 != 0 or n <= 1:
            raise ValueError(
                f"x2_format='nz' requires N % c0 == 0 (c0={c0}) and N > 1; got N={n}"
            )
        x2_kern = _bridge_nz_input(x2, bsz, k, n, c0)
    else:
        if perm_x2 == (0, 2, 1):
            n = int(x2.shape[1]) if x2.dim() == 3 else 0
        else:
            n = int(x2.shape[2]) if x2.dim() == 3 else 0
        if n < 1:
            raise ValueError(f"empty tensors are unsupported; got N={n}")
        x2_kern, kern_trans_b = _normalize_x2_nd(x2, perm_x2, bsz, n, k)
        if not kern_trans_b and k % c0:
            # K-major ND B operand with a c0-misaligned K: the nd2nz
            # on-the-fly zero fill only covers the staging column axis
            # (the A operand), so the B slot's K-tail rows
            # [k, ceil_c0(k)) would carry L1 residue — stale garbage in
            # the best case, NaN-poisonable after an unlucky neighbor
            # kernel (observed as scattered NaN output columns on
            # dav-3510). Pad the GM operand with real zeros instead so
            # every tail row the L1 staging reads is deterministic.
            x2_kern = torch.nn.functional.pad(
                x2_kern, (0, 0, 0, _ceil_align(k, c0) - k)
            )

    if x2_kern.shape[0] != bsz:
        raise ValueError(f"batch mismatch: x1 B={bsz} vs x2 B={x2_kern.shape[0]}")

    # ---- output ----
    out_t = _resolve_out(out, m, bsz, n, bsf, x1.dtype, x1.device)

    # ---- tiling + kernel launch (rebuilt per shape) ----
    # b=bsz feeds the guardrail's T_nd (internal interface — the
    # tiling class is never part of the public signature).
    tiling = TransposeBatchMatMulTiling(
        m, n, k, x1.dtype, kern_trans_b=kern_trans_b, b=bsz
    )
    ctx = _KernelCtx(
        b=bsz,
        m=m,
        n=n,
        k=k,
        bsf=bsf,
        kern_trans_a=kern_trans_a,
        kern_trans_b=kern_trans_b,
        nz_input=nz_input,
        hf32=compute_mode in ("hf32", "hf32_hp"),
        hf32_round_away=compute_mode == "hf32_hp",
    )
    op = TransposeBatchMatMulKernel(tiling, ctx)
    out_arg = out_t.view(bsf, m, bsz // bsf, n) if bsf > 1 else out_t
    op.run(x1, x2_kern, out_arg)
    return out_t
