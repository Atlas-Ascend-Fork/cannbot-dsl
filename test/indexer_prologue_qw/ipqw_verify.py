# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared CPU reference, deployment matrix and precision/performance CLI.

Q: MXFP8 GEMM, tail RoPE, then MXFP4 quantization. W: BF16 GEMM and scale.
The CPU reference is independent of the NPU kernel's scale transform.
Run from the repository root with PYTHONPATH=samples:
    python test/indexer_prologue_qw/ipqw_verify.py --t 1 72 --output /tmp/ipqw.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

logger = logging.getLogger(__name__)

import torch

# CPU reference and input construction.
MX_GROUP_SIZE = 32
MX_SCALE_PAIR = 2
MX_K_ALIGN = 64
FP4_E2M1_MAX = 6.0
E8M0_BIAS = 127
FP32_SHIFT_BITS = 23
FP32_MANTISSA_MASK = 0x7FFFFF
# OCP MXFP4 E2M1 positive codebook. Index is the 3-bit magnitude field.
E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Halfway between consecutive E2M1 magnitudes: where rounding flips.
E2M1_MIDPOINTS = tuple(
    (low + high) / 2 for low, high in zip(E2M1_POS, E2M1_POS[1:])
)


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def paired_scale_groups(length: int) -> int:
    return ceil_div(length, MX_K_ALIGN)


def _decode_e8m0_scale(scale: torch.Tensor, outer_size: int, k: int) -> torch.Tensor:
    """Decode public [outer, G, 2] E8M0 bytes to one FP32 scale per 32 elements."""
    valid = ceil_div(k, MX_GROUP_SIZE)
    exponent_bytes = scale.contiguous().view(torch.uint8).reshape(outer_size, -1)
    exponents = exponent_bytes[:, :valid].to(torch.int16) - E8M0_BIAS
    return torch.pow(2.0, exponents.float())


def _fp32_scale_to_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Round FP32 scales up to the next power of two and store IEEE exponent bytes.

    Matches kv_compress_epilog: if the mantissa is nonzero, increment the exponent.
    """
    bits = scale.to(torch.float32).view(torch.int32).to(torch.int64)
    exp_bits = (bits >> FP32_SHIFT_BITS) & 0xFF
    mantissa = bits & FP32_MANTISSA_MASK
    exp_bits = torch.where(mantissa != 0, (exp_bits + 1) & 0xFF, exp_bits)
    zero = scale <= 0
    return torch.where(zero, torch.zeros_like(exp_bits), exp_bits).to(torch.uint8)


def rotate_interleave(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent RoPE pairs: ``(x0, x1) -> (-x1, x0)``."""
    if x.shape[-1] % 2:
        raise ValueError(f"interleaved RoPE needs an even width, got {x.shape[-1]}")
    pair = x.reshape(*x.shape[:-1], -1, 2)
    return torch.stack((-pair[..., 1], pair[..., 0]), dim=-1).reshape_as(x)


def apply_rope_inplace(q: torch.Tensor, rope_sin: torch.Tensor, rope_cos: torch.Tensor) -> torch.Tensor:
    """Inplace RoPE on the last Dr columns of D. q is (T, N, D)."""
    dr = int(rope_sin.shape[-1])
    if dr == 0:
        return q
    if dr % 2 != 0:
        raise ValueError(f"Dr must be even, got {dr}")
    if dr > q.shape[-1]:
        raise ValueError(f"Dr={dr} exceeds D={q.shape[-1]}")
    pe = q[..., -dr:]
    cos = rope_cos.to(torch.float32).unsqueeze(1)
    sin = rope_sin.to(torch.float32).unsqueeze(1)
    q[..., -dr:] = pe * cos + rotate_interleave(pe) * sin
    return q


def mx_matmul_fp32(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    scale_lhs: torch.Tensor,
    scale_rhs: torch.Tensor,
) -> torch.Tensor:
    """Dequantised fp32 matmul of the two MX operands."""
    m, k = lhs.shape
    n = rhs.shape[0]
    scale_a = _decode_e8m0_scale(scale_lhs, m, k).repeat_interleave(MX_GROUP_SIZE, dim=1)[:, :k]
    scale_b = _decode_e8m0_scale(scale_rhs, n, k).repeat_interleave(MX_GROUP_SIZE, dim=1)[:, :k]
    return (lhs.float() * scale_a) @ (rhs.float() * scale_b).T


def fp32_to_e2m1_nibble(values: torch.Tensor) -> torch.Tensor:
    """Round FP32 to nearest OCP E2M1 code (4-bit nibble), ties-to-even."""
    codebook = torch.tensor(E2M1_POS, dtype=torch.float32, device=values.device)
    sign = values < 0
    mag = values.abs()
    dist = (mag.unsqueeze(-1) - codebook).abs()
    min_dist = dist.min(dim=-1).values.unsqueeze(-1)
    tied = dist == min_dist
    arange = torch.arange(8, device=values.device, dtype=torch.int64)
    nearest = torch.where(tied, arange, torch.full_like(arange, 8)).min(dim=-1).values
    even = torch.where(
        tied & ((arange & 1) == 0), arange, torch.full_like(arange, 8)
    ).min(dim=-1).values
    idx = torch.where(tied.sum(dim=-1) > 1, even, nearest)
    nibble = idx.to(torch.uint8)
    nibble = torch.where(sign, nibble | 0x08, nibble)
    return nibble


def pack_e2m1_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    """Pack (..., D) nibbles into (..., D/2) bytes. D must be even."""
    if nibbles.shape[-1] % 2 != 0:
        raise ValueError(f"D must be even to pack e2m1, got {nibbles.shape[-1]}")
    low = nibbles[..., 0::2].to(torch.int16)
    high = nibbles[..., 1::2].to(torch.int16)
    return (low | (high << 4)).to(torch.uint8).contiguous()


def unpack_e2m1_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Unpack (..., D/2) bytes into (..., D) nibbles."""
    low = packed & 0x0F
    high = (packed.to(torch.int16) >> 4).to(torch.uint8)
    return torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def e2m1_nibble_to_fp32(nibbles: torch.Tensor) -> torch.Tensor:
    codebook = torch.tensor(E2M1_POS, dtype=torch.float32, device=nibbles.device)
    mag = codebook[(nibbles & 0x7).long()]
    sign = torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    return mag * sign


def mx_quant_along_last(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize (T, N, D) FP32 to MXFP4 along D.

    Returns:
        q: uint8 packed e2m1, shape (T, N, D/2)
        descale: uint8 E8M0 packed as (T, N, ceil(D/64), 2)
    """
    t, n, d = x.shape
    if d % 2 != 0:
        raise ValueError(f"D must be even for FP4 packing, got {d}")
    n_groups = ceil_div(d, MX_GROUP_SIZE)
    pad = n_groups * MX_GROUP_SIZE - d
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    grouped = x.reshape(t, n, n_groups, MX_GROUP_SIZE)
    amax = grouped.abs().amax(dim=-1).clamp_min(0.0)
    raw_scale = torch.where(
        amax > 0,
        amax / FP4_E2M1_MAX,
        torch.ones_like(amax),
    )
    e8m0 = _fp32_scale_to_e8m0(raw_scale)
    decoded = torch.pow(2.0, e8m0.to(torch.float32) - E8M0_BIAS)
    q_fp32 = grouped / decoded.unsqueeze(-1)
    nibbles = fp32_to_e2m1_nibble(q_fp32.reshape(t, n, n_groups * MX_GROUP_SIZE)[..., :d])
    q = pack_e2m1_nibbles(nibbles)

    pair_groups = paired_scale_groups(d)
    pair_elems = pair_groups * MX_SCALE_PAIR
    if n_groups < pair_elems:
        e8m0 = torch.nn.functional.pad(e8m0, (0, pair_elems - n_groups), value=E8M0_BIAS)
    descale = e8m0.reshape(t, n, pair_groups, MX_SCALE_PAIR).contiguous()
    return q, descale


@dataclass
class IndexerPrologueQwInputs:
    x: torch.Tensor
    qr: torch.Tensor
    wqb: torch.Tensor
    ww: torch.Tensor
    descale_qr: torch.Tensor
    descale_wqb: torch.Tensor
    rope_sin: torch.Tensor
    rope_cos: torch.Tensor
    softmax_scale: float


@dataclass
class IndexerPrologueQwOutputs:
    q: torch.Tensor
    descale_q: torch.Tensor
    w: torch.Tensor
    q_fp32: torch.Tensor  # after RoPE, before quant; debug / staged probes


def _dequant_mx_row(
    values: torch.Tensor, scale_bytes: torch.Tensor, k: int
) -> torch.Tensor:
    """One row of an MX operand, decoded to fp32."""
    scale = _decode_e8m0_scale(scale_bytes.reshape(1, -1), 1, k)
    scale = scale.repeat_interleave(MX_GROUP_SIZE, dim=1)[:, :k]
    return values.float().reshape(1, k) * scale


def _gemm_uncertainty(
    inputs: "IndexerPrologueQwInputs", row: int, head: int, col: int, d: int
) -> float:
    """How far apart two fp32 accumulation orders of one Q dot product can land.

    The kernel and this golden reduce the same 1280 products, so they agree on
    the exact value and disagree only on rounding.  The spread of an fp32 sum
    over ``k`` terms is bounded by ``k * eps * sum|term|`` in the worst case and
    behaves like ``sqrt(k) * eps * sum|term|`` for the blocked order L0C uses;
    the looser square-root form is used here.  What makes this safe as the basis
    for an exemption is not its tightness in ulps but its size against the E2M1
    grid: for the frozen geometry it is ~1e-5 of the quantisation step, so it can
    only ever excuse a value already sitting on a rounding boundary.
    """
    k = inputs.qr.shape[1]
    lhs = _dequant_mx_row(inputs.qr[row], inputs.descale_qr[row], k)
    rhs_row = head * d + col
    rhs = _dequant_mx_row(inputs.wqb[rhs_row], inputs.descale_wqb[rhs_row], k)
    terms = float((lhs * rhs).abs().sum())
    return math.sqrt(k) * torch.finfo(torch.float32).eps * terms


def _element_uncertainty(inputs, want, coord, width) -> float:
    """``_gemm_uncertainty`` carried through the RoPE mix for one output element."""
    row, head, col = coord
    d, dr = width
    base = _gemm_uncertainty(inputs, row, head, col, d)
    if dr == 0 or col < d - dr:
        return base
    local = col - (d - dr)
    half = dr // 2
    partner = local + half if local < half else local - half
    cos = abs(float(inputs.rope_cos[row, local]))
    sin = abs(float(inputs.rope_sin[row, local]))
    partner_unc = _gemm_uncertainty(inputs, row, head, partner + d - dr, d)
    value = abs(float(want.q_fp32[row, head, col]))
    return cos * base + sin * partner_unc + torch.finfo(torch.float32).eps * value


def q_accum_ambiguous(
    got: torch.Tensor,
    want: IndexerPrologueQwOutputs,
    inputs: IndexerPrologueQwInputs,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """Of the ``candidates`` bytes, which are E2M1 decisions fp32 cannot settle.

    ``q_mismatches``' fixed ulp window is a cheap screen, and it is a function
    of nothing -- so as T grows and the sample count grows with it, values turn
    up further from a midpoint than the window allows while still being pure
    accumulation noise (T=131072 finds one at 8 ulp).  This prices each survivor
    instead: recompute that element's own dot product, bound how far two
    summation orders of it can differ, and excuse the byte only if that interval
    straddles the midpoint between the two codes.  Only a handful of bytes ever
    reach here, so the per-element cost does not matter.
    """
    index = torch.nonzero(candidates)
    if index.numel() == 0:
        return torch.zeros_like(candidates)

    t, n, half_d = got.shape
    d = half_d * 2
    dr = int(inputs.rope_sin.shape[-1])
    got_nibbles = unpack_e2m1_nibbles(got)
    want_nibbles = unpack_e2m1_nibbles(want.q)
    scale = torch.pow(2.0, want.descale_q.reshape(t, n, -1).float() - E8M0_BIAS)

    ambiguous = torch.zeros_like(candidates)
    for row, head, byte in index.tolist():
        settled = False
        for half in (0, 1):
            col = 2 * byte + half
            code_got = int(got_nibbles[row, head, col])
            code_want = int(want_nibbles[row, head, col])
            if code_got == code_want:
                continue
            if (code_got & 0x8) != (code_want & 0x8):
                settled = True
                break
            low = min(code_got & 0x7, code_want & 0x7)
            if abs((code_got & 0x7) - (code_want & 0x7)) != 1:
                settled = True
                break
            group_scale = float(scale[row, head, col // MX_GROUP_SIZE])
            window = (
                _element_uncertainty(inputs, want, (row, head, col), (d, dr))
                / group_scale
            )
            quotient = abs(float(want.q_fp32[row, head, col])) / group_scale
            if abs(quotient - E2M1_MIDPOINTS[low]) > window:
                settled = True
                break
        ambiguous[row, head, byte] = not settled
    return ambiguous


def descale_accum_ambiguous(
    got: torch.Tensor,
    want: IndexerPrologueQwOutputs,
    inputs: IndexerPrologueQwInputs,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """Of the ``candidates`` scale bytes, which exponents fp32 cannot settle.

    ``descale_q`` is an E8M0 exponent taken from the amax of a group, so it is
    bit-exact against this golden unless amax / 6 sits on a power of two.
    When the largest magnitude in a group lands within accumulation
    noise of a 6 * 2^k boundary, the two summation orders land on either side of it
    and the exponents differ by exactly one.  T=12345 finds one such group in
    1.6 M scale bytes.

    So an off-by-one exponent is excused only when the amax it came from is
    within its own dot product's accumulation spread of the boundary; anything
    else, including any difference of more than one, still counts.
    """
    index = torch.nonzero(candidates)
    if index.numel() == 0:
        return torch.zeros_like(candidates)

    t, n = want.q_fp32.shape[0], want.q_fp32.shape[1]
    d = want.q_fp32.shape[2]
    dr = int(inputs.rope_sin.shape[-1])
    groups_per_head = got.shape[2] * got.shape[3]

    ambiguous = torch.zeros_like(candidates)
    for position in index.tolist():
        row, head = position[0], position[1]
        flat_group = position[2] * got.shape[3] + position[3]
        if abs(int(got[tuple(position)]) - int(want.descale_q[tuple(position)])) != 1:
            continue
        span = d // groups_per_head
        lo = flat_group * span
        group = want.q_fp32[row, head, lo:lo + span].abs()
        amax = float(group.max())
        if amax == 0.0:
            continue
        col = lo + int(group.argmax())
        window = _element_uncertainty(inputs, want, (row, head, col), (d, dr))
        # scale = 2**ceil(log2(amax / 6)). The boundary between the two
        # candidate exponents is six times the smaller decoded scale.
        lower_code = min(int(got[tuple(position)]), int(want.descale_q[tuple(position)]))
        boundary = FP4_E2M1_MAX * 2.0 ** (lower_code - E8M0_BIAS)
        ambiguous[tuple(position)] = abs(amax - boundary) <= window
    return ambiguous


def q_mismatches(
    got: torch.Tensor, want: IndexerPrologueQwOutputs, ulps: int = 4
) -> torch.Tensor:
    """Mask of packed ``q`` bytes that differ for a reason other than a tie.

    ``q`` is bit-exact against this golden at small T, and asserting that is
    worth keeping -- but it stops being a sound assertion as T grows, for a
    reason that is not about the kernel.  The kernel quantises its own fp32
    GEMM result, whose last bit depends on the order L0C accumulated 1280
    products in; this golden quantises torch's sum of the same products.  The
    two differ by an ulp here and there, which changes nothing until a value
    lands on an E2M1 rounding midpoint (0.25, 0.75, ... 5.0), where that ulp
    decides which way it rounds.

    How often that happens is a function of how many elements there are.
    Measured on seed 0: nothing within 2 ulp of a midpoint at T=72 or T=256,
    11 candidates at T=2048, of which 2 actually round the other way.

    So a byte is excused only when every differing nibble in it is one of:

    * a one-step, same-sign move whose golden quotient is within ``ulps``
      of a midpoint; or
    * a ``+0`` / ``-0`` pair (magnitude code 0, only the sign bit differs)
      whose golden quotient is below the first E2M1 midpoint.  T=4096 hits
      this once: a 1e-6 residual whose kernel GEMM came out the other side
      of zero.  Both sides flush to a zero nibble; the sign of underflow
      is not a contract.

    Anything else -- a sign flip on a nonzero code, a two-step jump, a
    value nowhere near a midpoint -- is a real mismatch and stays in the mask.
    """
    differs = got != want.q
    if not bool(differs.any()):
        return differs

    t, n, half_d = got.shape
    got_nibbles = unpack_e2m1_nibbles(got)
    want_nibbles = unpack_e2m1_nibbles(want.q)
    nibble_differs = got_nibbles != want_nibbles
    one_step = (
        (got_nibbles & 0x7).to(torch.int32) - (want_nibbles & 0x7).to(torch.int32)
    ).abs() == 1
    same_sign = (got_nibbles & 0x8) == (want_nibbles & 0x8)

    scale = torch.pow(2.0, want.descale_q.reshape(t, n, -1).float() - E8M0_BIAS)
    quotient = (
        want.q_fp32.reshape(t, n, scale.shape[-1], MX_GROUP_SIZE)
        / scale.unsqueeze(-1)
    ).abs().reshape(t, n, half_d * 2)
    eps = torch.finfo(torch.float32).eps
    at_midpoint = torch.zeros_like(quotient, dtype=torch.bool)
    for midpoint in E2M1_MIDPOINTS:
        at_midpoint |= (quotient - midpoint).abs() <= ulps * eps * midpoint

    signed_zero = (
        ((got_nibbles & 0x7) == 0)
        & ((want_nibbles & 0x7) == 0)
        & (quotient < E2M1_MIDPOINTS[0])
    )
    excused = (
        ~nibble_differs
        | (one_step & same_sign & at_midpoint)
        | signed_zero
    )
    # Both nibbles of a byte have to be accounted for to excuse the byte.
    return differs & ~(excused[..., 0::2] & excused[..., 1::2])


def _make_paired_scale(outer_size: int, k: int, generator: torch.Generator) -> torch.Tensor:
    scale_k_len = paired_scale_groups(k) * MX_SCALE_PAIR
    exponents = torch.randint(
        -2,
        3,
        (outer_size, scale_k_len),
        dtype=torch.int16,
        generator=generator,
    )
    return (
        (exponents + E8M0_BIAS)
        .to(torch.uint8)
        .reshape(outer_size, paired_scale_groups(k), MX_SCALE_PAIR)
        .contiguous()
    )


def make_inputs(
    t: int,
    shape: dict,
    *,
    softmax_scale: float = 1.0,
    rope_identity: bool = False,
    seed: int = 0,
) -> IndexerPrologueQwInputs:
    dim = shape["dim"]
    q_lora = shape["q_lora"]
    n_heads = shape["n_heads"]
    d = shape["d"]
    dr = shape["dr"]
    if dr > d:
        raise ValueError(f"Dr={dr} exceeds D={d}")
    if dr % 2 != 0:
        raise ValueError(f"Dr must be even, got {dr}")
    if d % 2 != 0:
        raise ValueError(f"D must be even for FP4 packing, got {d}")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    qr = (torch.randn((t, q_lora), generator=gen) * 0.5).to(torch.float8_e4m3fn)
    wqb = (torch.randn((n_heads * d, q_lora), generator=gen) * 0.5).to(torch.float8_e4m3fn)
    x = torch.randn((t, dim), generator=gen, dtype=torch.float32).to(torch.bfloat16)
    ww = torch.randn((n_heads, dim), generator=gen, dtype=torch.float32).to(torch.bfloat16)
    if rope_identity:
        rope_cos = torch.ones((t, dr), dtype=torch.float32)
        rope_sin = torch.zeros((t, dr), dtype=torch.float32)
    else:
        # RoPE uses adjacent (even, odd) pairs, so both lanes in a pair share
        # one angle/frequency.
        theta = torch.rand((t, dr // 2), generator=gen, dtype=torch.float32) * (2 * math.pi)
        theta = theta.repeat_interleave(2, dim=-1)
        rope_cos = torch.cos(theta)
        rope_sin = torch.sin(theta)
    return IndexerPrologueQwInputs(
        x=x,
        qr=qr,
        wqb=wqb,
        ww=ww,
        descale_qr=_make_paired_scale(t, q_lora, gen),
        descale_wqb=_make_paired_scale(n_heads * d, q_lora, gen),
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        softmax_scale=float(softmax_scale),
    )


def indexer_prologue_qw_golden(inputs: IndexerPrologueQwInputs) -> IndexerPrologueQwOutputs:
    t = inputs.qr.shape[0]
    n_d = inputs.wqb.shape[0]
    # Infer N, D from q output contract: wqb is (N*D, q_lora), ww is (N, dim).
    n_heads = inputs.ww.shape[0]
    if n_d % n_heads != 0:
        raise ValueError(f"wqb.shape[0]={n_d} is not divisible by N={n_heads}")
    d = n_d // n_heads

    y = mx_matmul_fp32(inputs.qr, inputs.wqb, inputs.descale_qr, inputs.descale_wqb)
    q_fp32 = y.reshape(t, n_heads, d).contiguous()
    q_fp32 = apply_rope_inplace(q_fp32, inputs.rope_sin, inputs.rope_cos)
    q, descale_q = mx_quant_along_last(q_fp32)

    w = inputs.softmax_scale * (inputs.x.float() @ inputs.ww.float().T)
    return IndexerPrologueQwOutputs(q=q, descale_q=descale_q, w=w, q_fp32=q_fp32)


def select_rows(
    inputs: IndexerPrologueQwInputs, rows: torch.Tensor
) -> IndexerPrologueQwInputs:
    """Restrict ``inputs`` to the T rows in ``rows``, keeping the weights whole.

    Every output row of this operator is a function of the same-index row of
    ``x`` / ``qr`` / the descale and RoPE tables and of the two weights; no
    reduction crosses T.  So the golden for a subset of rows is the golden of
    the subset, exactly -- which is what makes a prefill T of 262144 checkable
    at all.  The full golden would need a 262144x4096 fp32 GEMM plus a
    (T, N, D, 8) distance tensor for the E2M1 rounding.
    """
    return IndexerPrologueQwInputs(
        x=inputs.x[rows],
        qr=inputs.qr[rows],
        wqb=inputs.wqb,
        ww=inputs.ww,
        descale_qr=inputs.descale_qr[rows],
        descale_wqb=inputs.descale_wqb,
        rope_sin=inputs.rope_sin[rows],
        rope_cos=inputs.rope_cos[rows],
        softmax_scale=inputs.softmax_scale,
    )


def dequant_mx_last(q: torch.Tensor, descale: torch.Tensor, d: int) -> torch.Tensor:
    """Decode packed MXFP4 q (T,N,D/2) with paired descale back to FP32 (T,N,D)."""
    t, n, packed_d = q.shape
    if packed_d * 2 != d:
        raise ValueError(f"packed last dim {packed_d} does not match D={d}")
    nibbles = unpack_e2m1_nibbles(q)
    values = e2m1_nibble_to_fp32(nibbles)
    scale = _decode_e8m0_scale(
        descale.reshape(t * n, descale.shape[2], descale.shape[3]),
        t * n,
        d,
    ).reshape(t, n, -1)
    scale = scale.repeat_interleave(MX_GROUP_SIZE, dim=-1)[..., :d]
    return values * scale


# Deployment matrix: T = batch * seq; decode includes five MTP draft tokens.
GEOMETRY = dict(dim=5120, q_lora=1280, n_heads=32, d=128, dr=64)

MTP_DRAFT = 5


def _ints(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(int(v) for v in raw.replace(" ", "").split(",") if v)


DECODE_BATCH = _ints("IPQW_DECODE_BATCH", (1, 4, 8, 12, 16, 32))
DECODE_SEQ = _ints("IPQW_DECODE_SEQ", (1, 1 + MTP_DRAFT))
PREFILL_BATCH = _ints("IPQW_PREFILL_BATCH", (1, 4, 8, 16, 32))
PREFILL_SEQ = _ints("IPQW_PREFILL_SEQ", (1024, 2048, 4096, 8192))


def cases(phase: str) -> tuple[tuple[int, int, int], ...]:
    """``(T, batch, seq)`` for one phase, in ascending T."""
    if phase == "decode":
        batches, seqs = DECODE_BATCH, DECODE_SEQ
    elif phase == "prefill":
        batches, seqs = PREFILL_BATCH, PREFILL_SEQ
    else:
        raise ValueError(f"phase must be 'decode' or 'prefill', got {phase!r}")
    return tuple(
        sorted((b * s, b, s) for b in batches for s in seqs)
    )


def t_values(phase: str) -> tuple[int, ...]:
    """Distinct T for one phase. T is all the kernel sees."""
    return tuple(sorted({t for t, _, _ in cases(phase)}))


def origins(phase: str, t: int) -> tuple[str, ...]:
    return tuple(f"b{b}s{s}" for tt, b, s in cases(phase) if tt == t)


def all_t_values() -> tuple[int, ...]:
    return tuple(sorted(set(t_values("decode")) | set(t_values("prefill"))))


# Shared precision checks and command-line reporting.
def device_inputs(inputs):
    from indexer_prologue_qw import to_nz

    result = {}
    for name in (
        "x", "qr", "wqb", "ww", "descale_qr", "descale_wqb", "rope_sin", "rope_cos",
    ):
        result[name] = getattr(inputs, name).npu()
    for name in ("wqb", "ww"):
        result[name] = to_nz(result[name])
    result["softmax_scale"] = inputs.softmax_scale
    return result


def compare(got, want, inputs):
    q, scales, w = got
    if q.shape != want.q.shape:
        raise AssertionError(f"q shape {q.shape} != {want.q.shape}")
    if scales.shape != want.descale_q.shape:
        raise AssertionError(f"scale shape {scales.shape} != {want.descale_q.shape}")
    if w.shape != want.w.shape:
        raise AssertionError(f"w shape {w.shape} != {want.w.shape}")
    raw_q = int((q != want.q).sum())
    raw_scales = int((scales != want.descale_q).sum())
    bad_scales = scales != want.descale_q
    if bad_scales.any():
        bad_scales &= ~descale_accum_ambiguous(scales, want, inputs, bad_scales)
    if bad_scales.any():
        raise AssertionError(f"{int(bad_scales.sum())} unexplained scale bytes")
    if raw_scales:
        # An accepted scale boundary changes all 32 codes in that group.
        # Requantize the independent CPU values at that validated scale;
        # never exempt the group's Q bytes from comparison.
        decoded = torch.pow(2.0, scales.reshape(*q.shape[:2], -1).float() - 127)
        quotient = want.q_fp32 / decoded.repeat_interleave(32, dim=-1)
        want = replace(want, descale_q=scales,
                       q=pack_e2m1_nibbles(fp32_to_e2m1_nibble(quotient)))
    bad_q = q_mismatches(q, want)
    if bad_q.any():
        bad_q &= ~q_accum_ambiguous(q, want, inputs, bad_q)
    if bad_q.any():
        raise AssertionError(f"{int(bad_q.sum())} unexplained Q bytes")
    torch.testing.assert_close(w, want.w, rtol=2e-2, atol=2e-2)
    return raw_q, raw_scales, float((w - want.w).abs().max())


def check_shape(t, *, chunk_rows=1024, seed=0):
    from indexer_prologue_qw import indexer_prologue_qw

    inputs = make_inputs(t, GEOMETRY, softmax_scale=128**-0.5, seed=seed)
    got = tuple(x.cpu() for x in indexer_prologue_qw(**device_inputs(inputs)))
    if tuple(x.shape for x in got) != ((t, 32, 64), (t, 32, 2, 2), (t, 32)):
        raise AssertionError(tuple(x.shape for x in got))
    raw_q = raw_scales = 0
    max_w = 0.0
    # Every row is checked, with bounded CPU golden memory even at T=262144.
    for start in range(0, t, chunk_rows):
        stop = min(t, start + chunk_rows)
        subset = select_rows(inputs, slice(start, stop))
        metrics = compare(
            tuple(x[start:stop] for x in got), indexer_prologue_qw_golden(subset), subset,
        )
        raw_q += metrics[0]
        raw_scales += metrics[1]
        max_w = max(max_w, metrics[2])
    return dict(T=t, rows_checked=t, seed=seed, passed=True,
                q_boundary_bytes=raw_q, scale_boundary_bytes=raw_scales,
                w_max_abs_error=max_w)


def benchmark(t, *, repeats=100, rounds=10):
    """Graph device time per call; excludes compile, transfers and NZ conversion."""
    from indexer_prologue_qw import indexer_prologue_qw

    inputs = make_inputs(t, GEOMETRY, softmax_scale=128**-0.5, seed=0)
    args = device_inputs(inputs)
    for _ in range(5):
        indexer_prologue_qw(**args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = [indexer_prologue_qw(**args) for _ in range(repeats)]
    graph.replay()
    torch.npu.synchronize()
    samples = []
    for _ in range(rounds):
        start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeats)
    # Keep graph outputs alive through the timed replays.
    if outputs[-1][0].shape[0] != t:
        raise AssertionError(outputs[-1][0].shape)
    return dict(T=t, method="NPU graph event time per call", repeats=repeats,
                rounds=rounds, min_us=min(samples), median_us=statistics.median(samples),
                max_us=max(samples), samples_us=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t", type=int, nargs="+", default=None)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--perf-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("indexer_prologue_qw_validation.json"))
    args = parser.parse_args()
    import torch_npu
    from indexer_prologue_qw.indexer_prologue_qw import Geometry, plan_buckets, resolve_cube_cores

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    if args.chunk_rows <= 0:
        raise ValueError("chunk-rows must be positive")
    templates = []
    for tile_plan, lo, hi in plan_buckets(Geometry(**GEOMETRY)):
        templates.append(dict(T_min=lo, T_max=hi, head_blocks=tile_plan.n_head_blocks))
    report = dict(device=str(torch.npu.get_device_properties(0)),
                  torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                  templates=templates,
                  cube_cores=resolve_cube_cores(),
                  cases=[dict(phase=phase, batch=b, seq=s, T=t)
                         for phase in ("decode", "prefill") for t, b, s in cases(phase)],
                  precision=[], performance=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    if not args.perf_only:
        for t in args.t or all_t_values():
            logger.info("Checking all rows: T=%s", t)
            result = check_shape(t, chunk_rows=args.chunk_rows)
            report["precision"].append(result)
            logger.info("%s", json.dumps(result))
            save()
            gc.collect()
    for t in (72, 2048):
        result = benchmark(t)
        report["performance"].append(result)
        logger.info("%s", json.dumps(result))
        save()


if __name__ == "__main__":
    try:
        main()
    finally:
        from device_properties import cleanup_current_device_properties
        cleanup_current_device_properties()
