# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Non-quantized BatchMatMul with numpy-style batch broadcast (rank 2-6).

C[c_batch, M, N] = A[a_batch, M, K] @ B[b_batch, N, K]^T (+ bias),
where c_batch = broadcast(a_batch, b_batch) and size-1 batch dims repeat.

Structure: BatchMatmulTiling (host tiling) -> BatchMatmulKernel (@kernel
stride scheduler + @jit per-tile pipeline) -> batch_matmul() (torch API).
"""

__all__ = ["batch_matmul"]

import dataclasses
import os

import torch
from cannbotdsl import dtypes, get_mem_size, get_platform_info, make_layout
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.cube import enable_hf32, set_fp32_mode
from cannbotdsl.ops.matmul import matmul as dsl_matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import MemLoc, Tensor, ceil_div, tile_slice


# ============================================================================
# Helpers
# ============================================================================


_MAX_RANK = 6


def _ceil_div(a: int, b: int) -> int:
    """Host-side integer ceil-div (the DSL ``ceil_div`` is kernel-side only)."""
    if b <= 0:
        raise ValueError(f"divisor must be positive; got {b}")
    return (a + b - 1) // b


def _ceil_align(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError(f"alignment divisor must be positive; got {b}")
    return (a + b - 1) // b * b


def _prod(vals) -> int:
    out = 1
    for v in vals:
        out *= int(v)
    return out


def _is_canonical_transpose_view(b: torch.Tensor) -> bool:
    """True iff ``b`` is a compact ``transpose(-1, -2)`` view (any rank >= 2).

    Size-1 batch dims are skipped: their stride is arbitrary.
    """
    if b.dim() < 2:
        return False
    if not (b.stride(-2) == 1 and b.stride(-1) == b.size(-2)):
        return False
    expected = b.size(-2) * b.size(-1)
    for d in range(b.dim() - 3, -1, -1):
        if b.size(d) != 1:
            if b.stride(d) != expected:
                return False
            expected *= b.size(d)
    return True


def _broadcast_batch_shapes(a_batch, b_batch):
    """Numpy-style broadcast of the leading (batch) dims; returns c_batch."""
    n = max(len(a_batch), len(b_batch))
    pa = (1,) * (n - len(a_batch)) + tuple(int(d) for d in a_batch)
    pb = (1,) * (n - len(b_batch)) + tuple(int(d) for d in b_batch)
    c_batch = []
    for da, db in zip(pa, pb):
        if da == db:
            c_batch.append(da)
        elif da == 1:
            c_batch.append(db)
        elif db == 1:
            c_batch.append(da)
        else:
            raise ValueError(
                f"batch dims are not broadcastable: {tuple(a_batch)} vs "
                f"{tuple(b_batch)} (dim {da} vs {db})"
            )
    return tuple(c_batch)


def _decode_batch_terms(src_batch, c_batch):
    """Broadcast-aware source addressing for one operand.

    Returns (full, terms): ``full`` means the flat batch index equals the
    output's; otherwise ``src_idx = sum(((out_idx // div) % extent) * mul)``
    over the (div, extent, mul) terms — broadcast (size-1) dims contribute
    no term.
    """
    n = len(c_batch)
    ps = (1,) * (n - len(src_batch)) + tuple(int(d) for d in src_batch)

    divs = [1] * n
    muls = [1] * (n + 1)
    acc = 1
    for d in range(n - 1, -1, -1):
        divs[d] = acc
        acc *= c_batch[d]
        muls[d] = muls[d + 1] * ps[d]

    full = ps == tuple(c_batch)
    terms = []
    for d in range(n):
        if ps[d] != c_batch[d]:
            continue  # broadcast dim: source coordinate is always 0
        if c_batch[d] == 1:
            continue  # extent-1 output dim: coordinate is always 0
        terms.append((divs[d], c_batch[d], muls[d + 1]))
    return full, terms


def _plan_batch_broadcast(a_batch, b_batch):
    """Returns (c_batch, a_full, a_terms, b_full, b_terms); see
    ``_decode_batch_terms``."""
    c_batch = _broadcast_batch_shapes(a_batch, b_batch)
    a_full, a_terms = _decode_batch_terms(a_batch, c_batch)
    b_full, b_terms = _decode_batch_terms(b_batch, c_batch)
    return c_batch, a_full, a_terms, b_full, b_terms


_TORCH_DTYPE_TO_DSL = {
    torch.float16: (dtypes.float16, 2),
    torch.bfloat16: (dtypes.bfloat16, 2),
    torch.float32: (dtypes.float32, 4),
}


# Tile-grid arithmetic shared by the tiling and the wrapper's bias pad
# contract: _MN_CAP is the L0 feasibility cap on base_m/base_n (L0A
# double-buffer budget), _BASE_K_CANDIDATES the power-of-two fallbacks for
# base_k when K exceeds the L0A budget.
_BASIC_BLOCK = 16
_MN_CAP = 256
_BASE_K_CANDIDATES = (128, 64, 32, 16)

# L2 capacity default (dav-3510) for the C cache-hint policy — the DSL
# platform query exposes no "l2" key (get_mem_size supports bt/fb0/l0a/
# l0b/l0c/l1/ub only); _L2_BYPASS_ALIGN_ELEMS is the K/k_l1 element
# alignment the nd2nz bypass guard requires.
_L2_CAPACITY_DEFAULT_BYTES = 128 * 1024 * 1024
_L2_BYPASS_ALIGN_ELEMS = 128


def _l2_capacity_bytes() -> int:
    """L2 capacity in bytes for the C write-back cache-hint decision.

    ``BMM_L2_SIZE_BYTES`` is a bring-up/debug override for other SoCs:
    it must be a positive integer (byte count), is read at kernel
    construction time (never at import), and only affects the C copy's
    ``l2_cache_ctl`` — the A/B bypass policy never consults it.
    """
    raw = os.environ.get("BMM_L2_SIZE_BYTES")
    if raw is None:
        return _L2_CAPACITY_DEFAULT_BYTES
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"BMM_L2_SIZE_BYTES must be an integer byte count; got {raw!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"BMM_L2_SIZE_BYTES must be positive; got {value}")
    return value


# ---- torch-interface stage helpers (called by batch_matmul) ----


def _validate_inputs(a, b, hf32):
    if a.dtype != b.dtype:
        raise TypeError(f"a and b must have the same dtype, got {a.dtype} vs {b.dtype}")
    if a.dtype not in _TORCH_DTYPE_TO_DSL:
        raise TypeError(
            f"only torch.float16, torch.bfloat16 and torch.float32 are "
            f"supported; got {a.dtype}"
        )
    if hf32 and a.dtype != torch.float32:
        raise ValueError(f"hf32=True is only valid for float32 inputs; got {a.dtype}")
    if a.device != b.device:
        raise ValueError(
            f"a and b must be on the same device, got {a.device} vs {b.device}"
        )
    if a.device.type != "npu":
        raise ValueError(f"inputs must be NPU tensors; got device {a.device}")
    if not (2 <= a.dim() <= _MAX_RANK and 2 <= b.dim() <= _MAX_RANK):
        raise ValueError(
            f"a and b must have rank 2..{_MAX_RANK}, got rank {a.dim()} and {b.dim()}"
        )


def _unsqueeze2d(t):
    """Auto-unsqueeze 2-D inputs to 3-D; returns (tensor, was_2d)."""
    if t.dim() == 2:
        return t.unsqueeze(0), True
    return t, False


def _normalize_layout(t, name):
    """Flip a canonical ``transpose(-1, -2)`` view back to contiguous storage
    (zero copy); reject any other non-contiguous layout.

    Returns (t_kern, transposed_storage)."""
    if _is_canonical_transpose_view(t):
        return t.transpose(-1, -2), True
    if t.is_contiguous():
        return t, False
    raise ValueError(
        f"{name} must be contiguous or a canonical transpose(-1,-2) view; "
        f"got shape={tuple(t.shape)}, stride={tuple(t.stride())}"
    )


def _mat_dims(a, b):
    """Returns (a_batch, b_batch, M, N, K) under the ``[*, M, K]`` /
    ``[*, N, K]`` (K-last) convention; validates the K match."""
    a_batch = tuple(a.shape[:-2])
    m, k = a.shape[-2], a.shape[-1]
    b_batch = tuple(b.shape[:-2])
    n, k_b = b.shape[-2], b.shape[-1]
    if k != k_b:
        raise ValueError(f"K mismatch between a ({k}) and b ({k_b})")
    return a_batch, b_batch, m, n, k


def _normalize_bias(bias, a, c_batch, n):
    """Bias must be ``[N]`` (shared) or ``[*c_batch, N]`` (per-batch), fp32
    or the input dtype; returns (bias, bias_shared, bias_full)."""
    if bias is None:
        return None, False, False
    if bias.device != a.device:
        raise ValueError(f"bias must be on the same device as a; got {bias.device}")
    if bias.dtype not in (a.dtype, torch.float32):
        raise TypeError(
            f"bias dtype must be the input dtype or float32; got {bias.dtype}"
        )
    if bias.dim() == 1:
        if bias.shape[0] != n:
            raise ValueError(f"bias length must equal N ({n}); got {bias.shape[0]}")
        return bias, True, False
    if bias.dim() == len(c_batch) + 1:
        if tuple(bias.shape[:-1]) != c_batch or bias.shape[-1] != n:
            raise ValueError(
                f"per-batch bias shape must be {(*c_batch, n)}; got {tuple(bias.shape)}"
            )
        return bias, False, True
    raise ValueError(
        f"bias must be rank 1 [N] or rank {len(c_batch) + 1} "
        f"[*c_batch, N]; got rank {bias.dim()}"
    )


@dataclasses.dataclass
class _KernelCtx:
    """Launch context bundling the wrapper's planning outputs."""

    a_kern: torch.Tensor
    b_kern: torch.Tensor
    bias: torch.Tensor | None
    c_batch: tuple
    m: int
    n: int
    k: int
    hf32: bool
    kern_trans_a: bool
    kern_trans_b: bool
    a_full: bool
    a_terms: tuple
    b_full: bool
    b_terms: tuple
    bias_shared: bool
    bias_full: bool
    squeeze_out: bool


def _degenerate_result(ctx: _KernelCtx):
    """Degenerate shapes (M/N/batch/K == 0) never reach the kernel; returns
    None otherwise."""

    def zeros_out():
        z = torch.zeros(
            *ctx.c_batch,
            ctx.m,
            ctx.n,
            dtype=ctx.a_kern.dtype,
            device=ctx.a_kern.device,
        )
        return z[0] if ctx.squeeze_out else z

    if ctx.m == 0 or ctx.n == 0 or _prod(ctx.c_batch) == 0:
        return zeros_out()
    if ctx.k == 0:
        if ctx.bias is None:
            return zeros_out()
        if ctx.bias_shared:
            bias = ctx.bias.reshape(*((1,) * len(ctx.c_batch)), 1, ctx.n)
        else:
            bias = ctx.bias.reshape(*ctx.c_batch, 1, ctx.n)
        out = bias.expand(*ctx.c_batch, ctx.m, ctx.n).to(ctx.a_kern.dtype)
        out = out.contiguous()
        return out[0] if ctx.squeeze_out else out
    return None


def _pad_n_tail(ctx: _KernelCtx):
    """Pad a statically-short single N tile (N < base_n) up to base_n; returns
    (b_kern, bias, n_eff).

    Required when bias is present (the L1->BT chain needs equal logical
    shapes) or the dtype is 4-byte with a (K, N) b storage (nd2nz crashes
    MTE on sub-fractal rows).  Multi-tile tails lower with runtime lengths
    and need no padding.
    """
    b_kern, bias, n = ctx.b_kern, ctx.bias, ctx.n
    base_n0 = min(_ceil_align(n, _BASIC_BLOCK), _MN_CAP)
    if not (n < base_n0 and (bias is not None or ctx.a_kern.element_size() == 4)):
        return b_kern, bias, n
    n_eff = base_n0
    if ctx.kern_trans_b:
        b_kern = torch.nn.functional.pad(b_kern, (0, 0, 0, n_eff - n))
    else:
        b_kern = torch.nn.functional.pad(b_kern, (0, n_eff - n))
    if bias is not None:
        bias_pad = torch.zeros(
            *(() if ctx.bias_shared else ctx.c_batch),
            n_eff,
            dtype=bias.dtype,
            device=bias.device,
        )
        bias_pad[..., :n] = bias
        bias = bias_pad
    return b_kern, bias, n_eff


def _reshape_output(c, c_m, c_n, ctx: _KernelCtx):
    """Slice the M/N tile-grid pad back off and restore the batch dims."""
    if c_m != ctx.m:
        c = c.narrow(-2, 0, ctx.m)
    if c_n != ctx.n:
        c = c.narrow(-1, 0, ctx.n)
    if ctx.squeeze_out:
        return c[0]
    return c.reshape(*ctx.c_batch, ctx.m, ctx.n)


def _run_kernel(ctx: _KernelCtx):
    """N-tail pad, zero-copy batch flatten, tiling, kernel launch and
    output shape restore.

    C is allocated on the full (m_tiles*base_m, n_tiles*base_n) tile grid:
    the FIXPIPE writes each output tile at its nominal (base_m, base_n)
    extent, so the pad region absorbs the M/N tail overreach and is sliced
    back off afterwards.
    """
    dtype, device = ctx.a_kern.dtype, ctx.a_kern.device
    bs = _prod(ctx.c_batch)
    has_bias = ctx.bias is not None
    b_kern, bias, n_eff = _pad_n_tail(ctx)
    tiling = BatchMatmulTiling(
        ctx.m,
        n_eff,
        ctx.k,
        dtype,
        kern_trans_b=ctx.kern_trans_b,
        has_bias=has_bias,
    )
    # Zero-copy batch flatten (contiguous views, .view never copies).
    a3 = ctx.a_kern.view(_prod(ctx.a_kern.shape[:-2]), *ctx.a_kern.shape[-2:])
    b3 = b_kern.view(_prod(b_kern.shape[:-2]), *b_kern.shape[-2:])
    if has_bias:
        bias3 = bias.reshape(1, n_eff) if ctx.bias_shared else bias.reshape(bs, n_eff)
    else:
        bias3 = torch.zeros(1, tiling.base_n, dtype=torch.float32, device=device)
    c_m = tiling.m_tiles * tiling.base_m
    c_n = tiling.n_tiles * tiling.base_n
    c = torch.zeros(bs, c_m, c_n, dtype=dtype, device=device)
    op = BatchMatmulKernel(tiling, ctx, _TORCH_DTYPE_TO_DSL[bias3.dtype][0])
    op.run(a3, b3, c, bias3)
    return _reshape_output(c, c_m, c_n, ctx)


# ============================================================================
# 1. Host-side tiling
# ============================================================================


class BatchMatmulTiling:
    """Host-side tiling for ascend950 (dav-3510), mirroring the
    batch_mat_mul_v3 host tiling (ResetBaseDav3510 + CalL1TilingDefault +
    GetBaseK).

    Hardware constants are queried from the platform: L1/L0A/L0B/L0C sizes
    via get_mem_size, AIC count via get_platform_info().cube_core_num.

    The batch dim is orthogonal to the tile sizes — every batch matrix
    shares (M, N, K), so batch only multiplies the tile count.  Key
    outputs: base_m/base_n/base_k (L0 tile sizes), k_l1 (K staged per L1 pass),
    l1_buffer_num, l0c_db, m_tiles/n_tiles, usedCoreNum.
    """

    # ---- hardware properties (queried from the platform) ----
    L1_SIZE = get_mem_size("l1")
    L0A_SIZE = get_mem_size("l0a")
    L0B_SIZE = get_mem_size("l0b")
    L0C_SIZE = get_mem_size("l0c")
    AIC_NUM = get_platform_info().cube_core_num

    BASIC_BLOCK_16 = 16
    MAX_STEP_K = 8
    DB_SIZE = 2
    DATA_SIZE_FP32 = 4

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype=torch.float16,
        *,
        kern_trans_b: bool = True,
        has_bias: bool = False,
    ):
        self.m = m
        self.n = n
        self.k = k
        self.kern_trans_b = kern_trans_b
        self.has_bias = has_bias
        self.dtype, self.dtype_size = self._resolve_dtype(dtype, "dtype")
        self.is_fp32 = self.dtype == dtypes.float32
        self._compute()

    def __repr__(self):
        return (
            f"BatchMatmulTiling(M={self.m}, N={self.n}, K={self.k})\n"
            f"  base_m={self.base_m}, base_n={self.base_n}, base_k={self.base_k}\n"
            f"  k_l1={self.k_l1}, step_k={self.step_k}\n"
            f"  l1_buffer_num={self.l1_buffer_num}, l0c_db={self.l0c_db}\n"
            f"  m_tiles={self.m_tiles}, n_tiles={self.n_tiles}"
        )

    @classmethod
    def _resolve_dtype(cls, dtype, name: str) -> tuple:
        props = _TORCH_DTYPE_TO_DSL.get(dtype)
        if props is None:
            for _, dd in _TORCH_DTYPE_TO_DSL.items():
                if dd[0] == dtype:
                    props = dd
                    break
        if props is None:
            supported = ", ".join(sorted(str(d) for d in _TORCH_DTYPE_TO_DSL))
            raise TypeError(f"{name} only supports {supported}; got {dtype}")
        return props

    def _compute(self):
        """Dav-3510 basic path: base_m/base_n default 256 (16-aligned,
        shape-capped); base_k = 128B/dtype when K fits (floor-adjusted to
        L0A/L0B feasibility); k_l1 = largest base_k*step_k (step_k <= 8)
        under a double-buffered (aL1+bL1) budget.
        """
        dtype_size = self.dtype_size
        self.base_m = min(_ceil_align(self.m, self.BASIC_BLOCK_16), _MN_CAP)
        self.base_n = min(_ceil_align(self.n, self.BASIC_BLOCK_16), _MN_CAP)

        # A/B share the dtype; bound base_k by the smaller of L0A/L0B
        # (double-buffered), both queried from the platform.
        max_k = (
            min(self.L0A_SIZE, self.L0B_SIZE)
            // self.DB_SIZE
            // dtype_size
            // max(self.base_m, self.base_n)
        )
        k_align = _ceil_align(self.k, self.BASIC_BLOCK_16)
        if k_align <= max_k:
            self.base_k = k_align
        else:
            self.base_k = next(
                (c for c in _BASE_K_CANDIDATES if c <= max_k),
                self.BASIC_BLOCK_16,
            )

        max_step = min(_ceil_div(self.k, self.base_k), self.MAX_STEP_K)
        bias_l1 = (
            self.base_n * self.DATA_SIZE_FP32 * self.DB_SIZE if self.has_bias else 0
        )
        step_k = 1
        for s in range(1, max_step + 1):
            a_l1 = self.base_m * self.base_k * s * dtype_size
            b_l1 = self.base_n * self.base_k * s * dtype_size
            if (a_l1 + b_l1) * self.DB_SIZE + bias_l1 > self.L1_SIZE:
                break
            step_k = s
        self.step_k = step_k
        self.k_l1 = self.base_k * step_k
        self.l1_buffer_num = self.DB_SIZE

        # L0C depth: 2 when a double-size fp32 tile pair fits.
        if (
            self.base_m * self.base_n * self.DATA_SIZE_FP32 * self.DB_SIZE
            <= self.L0C_SIZE
        ):
            self.l0c_db = self.DB_SIZE
        else:
            self.l0c_db = 1

        self.m_tiles = _ceil_div(self.m, self.base_m)
        self.n_tiles = _ceil_div(self.n, self.base_n)


# ============================================================================
# 2. Kernel
# ============================================================================


class BatchMatmulKernel:
    """C[c_batch] = A[a_batch] @ B[b_batch]^T with linear stride scheduling.

    Operands arrive as flattened contiguous 3-D views: A ``[bsA, M, K]`` (or
    ``[bsA, K, M]`` when ``kern_trans_a``), B ``[bsB, N, K]`` (or
    ``[bsB, K, N]`` per ``kern_trans_b``).  Broadcast source addressing is
    decoded per tile from host-constant (div, extent, mul) terms, or
    ``src_idx = b_idx`` when the operand's batch shape equals c_batch.

    GM → L1 (MTE2, nd2nz, ping-pong) → L0A/L0B (MTE1, double buffer;
    transposed sides spell the transpose as ``src.permute(1, 0)``) → MMAD
    (fp32 L0C) → GM (FIXPIPE).  On ping-pong-free (l0c_db == 1) tilings the
    MMAD/FIXPIPE unit-flag mainline is enabled (the hardware auto-triggers
    the FIXPIPE on the final K block).
    """

    # Neutral (div, extent, mul) decode slot: (idx // 1) % 1 * 0 == 0.
    _NEUTRAL_TERM = (1, 1, 0)
    _MAX_BATCH_DIMS = 4  # rank 6 = 4 batch dims + 2 matrix dims

    def __init__(
        self,
        tiling: BatchMatmulTiling,
        ctx: _KernelCtx,
        bias_dtype=dtypes.float32,
    ):
        self.t = tiling
        self.bs = _prod(ctx.c_batch)
        self.a_full = ctx.a_full
        self.b_full = ctx.b_full
        self.use_hf32 = ctx.hf32
        self.kern_trans_a = ctx.kern_trans_a
        self.has_bias = ctx.bias is not None
        self.bias_dtype = bias_dtype
        self.bias_full = ctx.bias_full
        self.a0, self.a1, self.a2, self.a3 = self._pad_terms(ctx.a_terms)
        self.b0, self.b1, self.b2, self.b3 = self._pad_terms(ctx.b_terms)
        self.m_tiles = tiling.m_tiles
        self.n_tiles = tiling.n_tiles
        self.mn_tiles = self.m_tiles * self.n_tiles
        self.total_tiles = self.bs * self.mn_tiles
        self.k_l1_tiles = _ceil_div(tiling.k, tiling.k_l1)
        self.used_core_num = min(self.total_tiles, tiling.AIC_NUM)

        # L2 cache-hint policy: A/B bypass only on proven reuse and only on
        # the linear nd2nz path with 128-aligned K/k_l1; C cached iff the
        # whole batched output fits L2; bias always cached.
        aligned = (tiling.k % _L2_BYPASS_ALIGN_ELEMS == 0) and (
            tiling.k_l1 % _L2_BYPASS_ALIGN_ELEMS == 0
        )
        reuse_a = tiling.n_tiles > 1 or not ctx.a_full  # N-tile re-reads / bcast
        reuse_b = tiling.m_tiles > 1 or not ctx.b_full  # M-tile re-reads / bcast
        nd_path_a = not ctx.kern_trans_a
        nd_path_b = tiling.kern_trans_b
        self.l2_ctl_a = 1 if (reuse_a or not nd_path_a or not aligned) else 0
        self.l2_ctl_b = 1 if (reuse_b or not nd_path_b or not aligned) else 0
        self.l2_ctl_c = (
            1
            if self.bs * tiling.m * tiling.n * tiling.dtype_size <= _l2_capacity_bytes()
            else 0
        )

        # MMAD unit flag (batch_mat_mul_v3 Normal config): 2 while
        # accumulating, 3 on the final K block — the hardware then
        # auto-triggers the FIXPIPE.  Paired with the single-buffer L0C
        # specialization, so only on l0c_db == 1 tilings.
        self.use_unit_flag = tiling.l0c_db == 1

        # On-chip staging (created inside the @kernel body by
        # _create_buffers) and the per-copy option bundles.
        self.l1_a = self.l1_b = None
        self.l0a = self.l0b = self.l0c = None
        self.l1_bias = self.bias_ring = None
        self.opt_a = copy_options(l2_cache_ctl=self.l2_ctl_a)
        self.opt_b = copy_options(l2_cache_ctl=self.l2_ctl_b)
        self.opt_bias = copy_options(l2_cache_ctl=1)
        self.opt_c_flag = (
            copy_options(unit_flag=3, l2_cache_ctl=self.l2_ctl_c)
            if self.use_unit_flag
            else copy_options(l2_cache_ctl=self.l2_ctl_c)
        )

    @classmethod
    def _pad_terms(cls, terms):
        """Pad decode terms to 4 fixed slots (missing slots are neutral)."""
        terms = list(terms)
        if len(terms) > cls._MAX_BATCH_DIMS:
            raise ValueError(
                f"batch broadcast decode supports at most {cls._MAX_BATCH_DIMS} "
                f"batch dims (rank 6); got {len(terms)}"
            )
        padded = [tuple(t) for t in terms] + [cls._NEUTRAL_TERM] * (
            cls._MAX_BATCH_DIMS - len(terms)
        )
        return tuple(padded)

    @kernel
    def bmm_kernel(self, gm_a: Tensor, gm_b: Tensor, gm_c: Tensor, gm_bias: Tensor):
        t = self.t

        # Compile-time fp32 compute-mode selection (persistent cube state).
        if const_expr(t.is_fp32):
            if const_expr(self.use_hf32):
                enable_hf32()
            else:
                set_fp32_mode()

        self._create_buffers()

        for tile_idx in range(get_block_idx(), self.total_tiles, get_block_num()):
            self._compute_output_tile(gm_a, gm_b, gm_c, gm_bias, tile_idx)

    @jit
    def _create_buffers(self):
        """Declare the L1/L0/bias ring buffers."""
        t = self.t

        # Transposed sides stage K-major logical shapes in NZ L1; the L1->L0
        # copies spell the transpose as src.permute(1, 0).
        l1_b_shape = (t.base_n, t.k_l1) if t.kern_trans_b else (t.k_l1, t.base_n)
        l1_a_shape = (t.k_l1, t.base_m) if self.kern_trans_a else (t.base_m, t.k_l1)
        self.l1_a = RingBuffer(
            MemLoc.L1,
            t.dtype,
            make_layout(l1_a_shape),
            depth=t.l1_buffer_num,
            data_format="nz",
        )
        self.l1_b = RingBuffer(
            MemLoc.L1,
            t.dtype,
            make_layout(l1_b_shape),
            depth=t.l1_buffer_num,
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
            depth=t.l0c_db,
        )
        # Bias stage: L1 (nd) -> BT (fp32), consumed by the init MMAD;
        # bias-only so the buffers never eat into exact-fit no-bias L1.
        if const_expr(self.has_bias):
            self.l1_bias = RingBuffer(
                MemLoc.L1,
                self.bias_dtype,
                make_layout((t.base_n,)),
                depth=2,
                data_format="nd",
            )
            self.bias_ring = RingBuffer(
                MemLoc.BIAS,
                dtypes.float32,
                make_layout((t.base_n,)),
                depth=2,
                data_format="nd",
            )

    @jit
    def _decode_tile_batch_src(self, tile_idx):
        """Tile-index decode: (b_idx, mn_idx) plus the broadcast-resolved flat
        source indices (see _decode_batch_terms; full ⇒ source index equals
        the output batch index).  mn_idx packs (m_idx, n_idx) into one scalar
        for the 5-value coding rules (codecheck G.FNM.03/05).  Shared [N]
        bias lives in slot 0; full-batch bias by output batch."""
        b_idx = tile_idx // self.mn_tiles
        mn_idx = tile_idx % self.mn_tiles

        a_src = b_idx
        if const_expr(not self.a_full):
            a_src = (
                ((b_idx // self.a0[0]) % self.a0[1]) * self.a0[2]
                + ((b_idx // self.a1[0]) % self.a1[1]) * self.a1[2]
                + ((b_idx // self.a2[0]) % self.a2[1]) * self.a2[2]
                + ((b_idx // self.a3[0]) % self.a3[1]) * self.a3[2]
            )
        b_src = b_idx
        if const_expr(not self.b_full):
            b_src = (
                ((b_idx // self.b0[0]) % self.b0[1]) * self.b0[2]
                + ((b_idx // self.b1[0]) % self.b1[1]) * self.b1[2]
                + ((b_idx // self.b2[0]) % self.b2[1]) * self.b2[2]
                + ((b_idx // self.b3[0]) % self.b3[1]) * self.b3[2]
            )
        bias_src = 0
        if const_expr(self.bias_full):
            bias_src = b_idx
        return b_idx, mn_idx, a_src, b_src, bias_src

    @jit
    def _compute_output_tile(self, gm_a, gm_b, gm_c, gm_bias, tile_idx):
        """Bias load + K reduction + FIXPIPE for one output tile; tail tiles
        are clipped by tile_slice and zero-padded by the nd2nz engines.
        """
        decoded = self._decode_tile_batch_src(tile_idx)
        b_idx, mn_idx, a_src, b_src, bias_src = decoded
        m_idx = mn_idx // self.n_tiles
        n_idx = mn_idx % self.n_tiles

        base_m, base_n = self.t.base_m, self.t.base_n
        a2d = gm_a[a_src, None, None]
        b2d = gm_b[b_src, None, None]
        c2d = gm_c[b_idx, None, None]

        # One l0c produce per output tile: the K-loop MMADs write the slot
        # alias, the FIXPIPE consumes it once.
        c_w = self.l0c.next_write()

        # Bias: GM -> L1 -> BT ([N] load); the init MMAD folds it in once.
        if const_expr(self.has_bias):
            mem_copy(
                self.l1_bias.next_write(),
                tile_slice(gm_bias[bias_src, None], (base_n,), (n_idx,)),
                options=self.opt_bias,
            )
            mem_copy(self.bias_ring.next_write(), self.l1_bias.next_read())
            bias = self.bias_ring.next_read()
        else:
            bias = None

        # mn_idx packs (m_idx, n_idx) into one scalar purely to keep
        # _k_reduction within the repo's 5-formal-parameter coding rule
        # (codecheck G.FNM.03); it is a rule workaround, not business design.
        c_view = self._k_reduction(a2d, b2d, c_w, bias, mn_idx)

        gm_c_tile = tile_slice(c2d, (base_m, base_n), (m_idx, n_idx))
        if const_expr(self.use_unit_flag):
            # Unit-flag path: the FIXPIPE reads the produce alias; a
            # next_read() would desync against the auto-triggered FIXPIPE.
            mem_copy(gm_c_tile, c_view, options=self.opt_c_flag)
        else:
            mem_copy(gm_c_tile, self.l0c.next_read(), options=self.opt_c_flag)

    @jit
    def _load_ab_l1(self, a2d, b2d, k_l1_idx, mn_idx):
        """GM -> L1 copies of one k_l1 segment for both operands (mn_idx packs
        (m_idx, n_idx) for the 5-arg coding rule)."""
        m_idx = mn_idx // self.n_tiles
        n_idx = mn_idx % self.n_tiles
        t = self.t
        base_m, base_n = t.base_m, t.base_n
        k_l1 = t.k_l1

        if const_expr(self.kern_trans_a):
            gm_a_tile = tile_slice(a2d, (k_l1, base_m), (k_l1_idx, m_idx))
        else:
            gm_a_tile = tile_slice(a2d, (base_m, k_l1), (m_idx, k_l1_idx))
        mem_copy(self.l1_a.next_write(), gm_a_tile, options=self.opt_a)

        if const_expr(t.kern_trans_b):
            gm_b_tile = tile_slice(b2d, (base_n, k_l1), (n_idx, k_l1_idx))
        else:
            gm_b_tile = tile_slice(b2d, (k_l1, base_n), (k_l1_idx, n_idx))
        mem_copy(self.l1_b.next_write(), gm_b_tile, options=self.opt_b)

    @jit
    def _k_reduction(self, a2d, b2d, c_view, bias, mn_idx):
        """GM → L1 → L0 K reduction and MMAD accumulation for one tile
        (mn_idx packs (m_idx, n_idx) for the 5-arg coding rule).  Returns the
        accumulated L0C view for the FIXPIPE."""
        t = self.t
        base_m, base_n = t.base_m, t.base_n
        base_k, k_l1, k_total = t.base_k, t.k_l1, t.k
        last_l1_idx = self.k_l1_tiles - 1

        # Static (unrolled) K loops: every tail quantity below stays a
        # compile-time int, so the L1->L0 tile_slices keep static extents
        # and the bias/init selects stay Python-level (no runtime branch).
        for k_l1_idx in range_constexpr(self.k_l1_tiles):
            self._load_ab_l1(a2d, b2d, k_l1_idx, mn_idx)
            # One read per k_l1 segment: the inner loop slices the selected
            # slot (a read per slice would rotate the cursor off the slot).
            a_l1 = self.l1_a.next_read()
            b_l1 = self.l1_b.next_read()

            # The final k_l1 segment carries only its valid base_k blocks.
            k_remaining = k_total - k_l1_idx * k_l1
            k_l0_per_l1 = ceil_div(min(k_l1, k_remaining), base_k)
            for k_l0_idx in range_constexpr(k_l0_per_l1):
                # K tail clip: the GM->L1 nd2nz fill covers only
                # ceil(k_valid/16)*16 columns (fractal granularity), so the
                # final base_k block is copied/multiplied at that 16-aligned
                # extent instead of the nominal base_k (which would read
                # stale slot content past the fill).
                k_blk = min(base_k, k_total - k_l1_idx * k_l1 - k_l0_idx * base_k)
                k_eff = (k_blk + 15) // 16 * 16

                if const_expr(self.kern_trans_a):
                    l1_a_slice = tile_slice(a_l1, (base_k, base_m), (k_l0_idx, 0))
                    if k_eff != base_k:
                        l1_a_slice = tile_slice(l1_a_slice, (k_eff, base_m), (0, 0))
                    l1_a_slice = l1_a_slice.permute(1, 0)
                else:
                    l1_a_slice = tile_slice(a_l1, (base_m, base_k), (0, k_l0_idx))
                    if k_eff != base_k:
                        l1_a_slice = tile_slice(l1_a_slice, (base_m, k_eff), (0, 0))
                a_l0 = self.l0a.next_write()
                if k_eff != base_k:
                    a_l0 = tile_slice(a_l0, (base_m, k_eff), (0, 0))
                mem_copy(a_l0, l1_a_slice)

                if const_expr(t.kern_trans_b):
                    l1_b_slice = tile_slice(b_l1, (base_n, base_k), (0, k_l0_idx))
                    if k_eff != base_k:
                        l1_b_slice = tile_slice(l1_b_slice, (base_n, k_eff), (0, 0))
                    b_l0 = self.l0b.next_write()
                    if k_eff != base_k:
                        b_l0 = tile_slice(b_l0, (base_n, k_eff), (0, 0))
                    mem_copy(b_l0, l1_b_slice)
                else:
                    l1_b_slice = tile_slice(b_l1, (base_k, base_n), (k_l0_idx, 0))
                    if k_eff != base_k:
                        l1_b_slice = tile_slice(l1_b_slice, (k_eff, base_n), (0, 0))
                    l1_b_slice = l1_b_slice.permute(1, 0)
                    b_l0 = self.l0b.next_write()
                    if k_eff != base_k:
                        b_l0 = tile_slice(b_l0, (base_n, k_eff), (0, 0))
                    mem_copy(b_l0, l1_b_slice)

                is_first_k_block = k_l1_idx == 0 and k_l0_idx == 0
                unit_flag = 0
                if const_expr(self.use_unit_flag):
                    is_last_k_block = (
                        k_l1_idx == last_l1_idx and k_l0_idx == k_l0_per_l1 - 1
                    )
                    unit_flag = 3 if is_last_k_block else 2
                c_view = dsl_matmul(
                    c_view,
                    a_l0,
                    b_l0,
                    init=is_first_k_block,
                    bias=bias if is_first_k_block else None,
                    unit_flag=unit_flag,
                )
        return c_view

    @host
    def run(self, gm_a: Tensor, gm_b: Tensor, gm_c: Tensor, gm_bias: Tensor):
        self.bmm_kernel[self.used_core_num](gm_a, gm_b, gm_c, gm_bias)


# ============================================================================
# 3. Torch interface
# ============================================================================


def batch_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    bias=None,
    hf32: bool = False,
) -> torch.Tensor:
    """Batched matmul with numpy-style batch-dim broadcast (rank 2-6),
    ``F.linear``-style layouts: ``C[c_batch, M, N] = A[a_batch, M, K] @
    B[b_batch, N, K]^T + bias``.

    Transposition is derived from the operands' strides: contiguous tensors
    keep the fast path, canonical ``transpose(-1, -2)`` views are flipped
    back zero-copy, other layouts are rejected.

    Args:
      a: logical ``[*a_batch, M, K]``; 2-D auto-unsqueezed; contiguous or a
        canonical ``transpose(-1, -2)`` view; fp16/bf16/fp32.
      b: logical ``[*b_batch, N, K]``; same contract as ``a``.
      bias: optional ``[N]`` or ``[*c_batch, N]``, fp32 or the input dtype,
        folded into the init MMAD.
      hf32: fp32-only fast mode (TF32-tier, rel err ~3e-4).

    Returns ``c``: ``[*c_batch, M, N]`` (``[M, N]`` when both inputs were
    2-D), same dtype as the inputs (fp32 accumulation inside the kernel).
    """
    _validate_inputs(a, b, hf32)
    a, a_was_2d = _unsqueeze2d(a)
    b, b_was_2d = _unsqueeze2d(b)
    squeeze_out = a_was_2d and b_was_2d

    a_kern, a_transposed = _normalize_layout(a, "a")
    b_kern, b_transposed = _normalize_layout(b, "b")
    a_batch, b_batch, m, n, k = _mat_dims(a, b)
    c_batch, a_full, a_terms, b_full, b_terms = _plan_batch_broadcast(a_batch, b_batch)
    bias, bias_shared, bias_full = _normalize_bias(bias, a, c_batch, n)

    ctx = _KernelCtx(
        a_kern=a_kern,
        b_kern=b_kern,
        bias=bias,
        c_batch=c_batch,
        m=m,
        n=n,
        k=k,
        hf32=hf32,
        kern_trans_a=a_transposed,
        kern_trans_b=not b_transposed,
        a_full=a_full,
        a_terms=a_terms,
        b_full=b_full,
        b_terms=b_terms,
        bias_shared=bias_shared,
        bias_full=bias_full,
        squeeze_out=squeeze_out,
    )
    out = _degenerate_result(ctx)
    if out is not None:
        return out
    return _run_kernel(ctx)
