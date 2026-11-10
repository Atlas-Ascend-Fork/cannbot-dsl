# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Layered tests for transpose_batch_mat_mul.

Layout (mirrors the cannbotdsl-op-test L0-L3 layering):

  L0  compile/codegen matrix   marker=ascendc_toolchain
      AOT-compile every codegen-affecting variant and assert the .so
      artifact is produced (every compute mode on a fixed perm/bsf, the
      perm grid on fp16, one bsf=2 crossing, and the two NZ modes):
        * ND modes: fp16 / bf16 / fp32-KEEP_DTYPE / fp32-HF32;
        * NZ: fp16 / bf16 only (fp32-NZ is rejected by the DSL runtime
          on dav-3510 — FRACTAL_NZ_C0_16 is not a format-29 tensor);
        * cube_math_type=3 compiles identically to type 2 (dav-3510 has no
          NEAREST_AWAY hf32 round mode), so hf32 is one dtype-mode.

  L1  functional precision     marker=npu
      NPU result vs an independent CPU golden (torch einsum + explicit
      permute/reshape in fp32; never imports kernel code for the reference).
      Covers every perm combo x dtype pair (B alternating 1/4), the NZ
      paths (fp16/bf16, incl. the plain-ND 5-D panel format bridge), HF32
      type 2/3 and bsf in {1, 2, B, prime factor of B}.

  L2  boundary                 marker=npu
      M/N/K triple-unaligned tails (M=N=300 class), multi-segment and tail
      K, M=N=K=1, B=1 x A-skip-read, B=128, tiny/skinny shapes, out=
      in-place writes (all perms + tail sampling) and the rejection
      matrix (one case per validated rule), plus representative fp32
      K%16 in [1..8] K-tail fill-granularity cases (KEEP/HF32, both
      perm_x2 paths, and the small-K shortcut branch).

  L3  performance (optional)   markers=npu+slow
      torch-timing comparison against the bmm+transpose baseline on the
      GEMM-4k / B-heavy / long-K representative cases; records speedup
      numbers only (no optimization conclusions).

Tolerances: element-wise isclose(rtol, atol) against the dtype-cast
golden with a ptol mismatch-fraction allowance; the observed mismatch
fractions leave 3.4x-38x headroom under the gates below.

Run:  source /usr/local/Ascend/ascend-toolkit/set_env.sh
      python3 -m pytest test/matmul/transpose_batch_matmul/ -v
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "samples",
        "matmul",
        "transpose_batch_matmul",
    ),
)

import cannbotdsl  # noqa: E402
from transpose_batch_matmul import (  # noqa: E402
    TransposeBatchMatMulKernel,
    TransposeBatchMatMulTiling,
    _KernelCtx,
    transpose_batch_mat_mul,
)

_PERM_A_BMM = (0, 1, 2)
_PERM_A_TRANS = (1, 0, 2)
_PERM_B_KMAJ = (0, 1, 2)
_PERM_B_NMAJ = (0, 2, 1)


# ---------------------------------------------------------------------------
# Golden kit (CPU, independent of the NPU implementation)
# ---------------------------------------------------------------------------


def tbmm_golden(x1, x2, perm_x1, perm_x2, bsf):
    """Independent CPU reference.

    r[b, m, n] = sum_k A[b, m, k] * Bmat[b, k, n] with A = x1.permute(perm_x1)
    and Bmat = x2.permute(perm_x2); the output is stored under permY=[1, 0, 2]
    (out[m, b, n] for bsf=1; out[s, m, j*N+n] = r[s*inner+j, m, n] for bsf>1).
    fp32 upcast throughout; the caller casts to the output dtype.
    """
    a = x1.float().permute(*perm_x1)  # (B, M, K)
    bmat = x2.float().permute(*perm_x2)  # (B, K, N)
    bsz, m, _ = a.shape
    n = bmat.shape[2]
    r = torch.einsum("bmk,bkn->bmn", a, bmat)  # (B, M, N)
    if bsf == 1:
        return r.transpose(0, 1).contiguous()  # (M, B, N)
    inner = bsz // bsf
    out = torch.empty(bsf, m, inner * n, dtype=r.dtype)
    for s in range(bsf):
        for j in range(inner):
            out[s, :, j * n : (j + 1) * n] = r[s * inner + j]
    return out  # (bsf, M, inner*N)


# dtype/mode -> (rtol, ptol, atol). Observed mismatch fractions:
# fp16 <= 2.4e-4, bf16 <= 1.4e-4, fp32 <= 7.8e-5, HF32 <= 1.46e-2 —
# i.e. 4x/7x/38x/3.4x headroom under the ptol gates.
_TOL_FP16 = (1e-3, 1e-3, 1e-8)
_TOL_BF16 = (1e-3, 1e-3, 1e-8)
_TOL_FP32 = (3e-4, 3e-3, 1e-6)
_TOL_HF32 = (1e-2, 5e-2, 1e-3)


def _tolerances(dtype, cube_math_type):
    if dtype == torch.float32 and cube_math_type in (2, 3):
        return _TOL_HF32
    if dtype == torch.float32:
        return _TOL_FP32
    return _TOL_FP16 if dtype == torch.float16 else _TOL_BF16


def assert_isclose(actual, golden, tolerances, ctx):
    """Precision-tolerant comparison allowing a ptol mismatch fraction."""
    rtol, ptol, atol = tolerances
    a = actual.detach().cpu().float().reshape(-1)
    g = golden.detach().cpu().float().reshape(-1)
    if a.numel() != g.numel():
        raise AssertionError(
            f"{ctx}: output size mismatch actual={a.numel()} golden={g.numel()}"
        )
    close = torch.isclose(a, g, rtol=rtol, atol=atol, equal_nan=False)
    fail_ratio = 1.0 - close.float().mean().item()
    if bool(torch.isnan(a).any()):
        raise AssertionError(f"{ctx}: NaN in NPU output")
    if fail_ratio > ptol:
        bad = (~close).nonzero(as_tuple=False)
        first = int(bad[0]) if bad.numel() else 0
        raise AssertionError(
            f"{ctx}: fail_ratio={fail_ratio:.3e} > ptol={ptol:.3e} "
            f"(rtol={rtol:.1e}, atol={atol:.1e}); first flat index {first}: "
            f"actual={a[first].item():.6g} golden={g[first].item():.6g}"
        )


def _perm_tag(perm_x1, perm_x2):
    a = "trans" if perm_x1 == _PERM_A_TRANS else "bmm"
    b = "nmaj" if perm_x2 == _PERM_B_NMAJ else "kmaj"
    return f"{a}-{b}"


def make_nd_inputs(b, m, n, k, dtype, perm_x1, perm_x2, seed=0):
    """Deterministic physical-layout operands for the given perms.

    Returns (x1, x2_phys, x2_logical) on CPU; x2_logical is the pre-permute
    (B, K, N) tensor the golden consumes when the physical x2 is an NZ
    panel; for ND it equals x2_phys.permute(*perm_x2).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if perm_x1 == _PERM_A_TRANS:
        x1 = torch.randn(m, b, k, generator=gen, dtype=dtype)  # (M, B, K)
    else:
        x1 = torch.randn(b, m, k, generator=gen, dtype=dtype)  # (B, M, K)
    x2_logical = torch.randn(b, k, n, generator=gen, dtype=dtype)
    if perm_x2 == _PERM_B_NMAJ:
        x2_phys = x2_logical.transpose(-1, -2).contiguous()  # (B, N, K)
    else:
        x2_phys = x2_logical  # (B, K, N)
    return x1, x2_phys, x2_logical


def _run_and_check(
    x1_cpu,
    x2_npu,
    x2_golden,
    perm_x1,
    perm_x2,
    bsf,
    dtype,
    cube_math_type,
    ctx,
    out=None,
    x2_format="nd",
):
    """Common runner: launch the public wrapper, compare against golden."""
    got = transpose_batch_mat_mul(
        x1_cpu.npu(),
        x2_npu,
        perm_x1=perm_x1,
        perm_x2=perm_x2,
        cube_math_type=cube_math_type,
        batch_split_factor=bsf,
        out=out,
        x2_format=x2_format,
    )
    torch.npu.synchronize()
    ref = tbmm_golden(x1_cpu, x2_golden, perm_x1, perm_x2, bsf)
    if dtype != torch.float32:
        ref = ref.to(dtype)
    assert tuple(got.shape) == tuple(ref.shape), (
        f"{ctx}: shape {tuple(got.shape)} != golden {tuple(ref.shape)}"
    )
    assert_isclose(got, ref, _tolerances(dtype, cube_math_type), ctx)
    return got


# ---------------------------------------------------------------------------
# L0 — compile/codegen matrix (AOT, no kernel launch)
# ---------------------------------------------------------------------------

_L0_MODES = ("fp16", "bf16", "fp32", "hf32")
_L0_MODE_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "hf32": torch.float32,
}


def _l0_variants():
    """One AOT compile per codegen-affecting variant: every compute mode on
    a fixed perm/bsf, the full perm grid on fp16, one bsf=2 crossing, and
    the two NZ modes (fp32-NZ is rejected, FRACTAL_NZ requires 2-byte dtypes)."""
    variants = [
        pytest.param(_PERM_A_BMM, _PERM_B_KMAJ, 1, mode, "nd", id=f"nd-mode-{mode}")
        for mode in _L0_MODES
    ]
    variants += [
        pytest.param(
            _PERM_A_TRANS,
            perm_x2,
            1,
            "fp16",
            "nd",
            id=f"nd-{_perm_tag(_PERM_A_TRANS, perm_x2)}-fp16",
        )
        for perm_x2 in (_PERM_B_KMAJ, _PERM_B_NMAJ)
    ] + [
        pytest.param(
            _PERM_A_BMM,
            _PERM_B_NMAJ,
            1,
            "fp16",
            "nd",
            id=f"nd-{_perm_tag(_PERM_A_BMM, _PERM_B_NMAJ)}-fp16",
        ),
        pytest.param(
            _PERM_A_TRANS, _PERM_B_KMAJ, 2, "fp16", "nd", id="nd-trans-kmaj-bsf2"
        ),
        pytest.param(_PERM_A_BMM, _PERM_B_NMAJ, 2, "bf16", "nd", id="nd-bmm-nmaj-bsf2"),
    ]
    variants += [
        pytest.param(_PERM_A_TRANS, _PERM_B_KMAJ, 1, mode, "nz", id=f"nz-{mode}")
        for mode in ("fp16", "bf16")
    ]
    return variants


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize("perm_x1,perm_x2,bsf,mode,x2_format", _l0_variants())
def test_l0_compile_matrix(perm_x1, perm_x2, bsf, mode, x2_format):
    """AOT-compile one const_expr variant; assert the .so artifact exists."""
    b, m, n, k = 4, 64, 64, 64
    dtype = _L0_MODE_DTYPE[mode]
    hf32 = mode == "hf32"
    kern_trans_a = perm_x1 == _PERM_A_TRANS
    kern_trans_b = x2_format == "nd" and perm_x2 == _PERM_B_NMAJ
    nz_input = x2_format == "nz"

    tiling = TransposeBatchMatMulTiling(m, n, k, dtype, kern_trans_b=kern_trans_b, b=b)
    ctx = _KernelCtx(
        b=b,
        m=m,
        n=n,
        k=k,
        bsf=bsf,
        kern_trans_a=kern_trans_a,
        kern_trans_b=kern_trans_b,
        nz_input=nz_input,
        hf32=hf32,
        hf32_round_away=False,
    )
    op = TransposeBatchMatMulKernel(tiling, ctx)

    dsl_dtype = tiling.dtype
    x1_spec = cannbotdsl.TensorSpec((m, b, k) if kern_trans_a else (b, m, k), dsl_dtype)
    if nz_input:
        x2_spec = cannbotdsl.TensorSpec((b, k, n), dsl_dtype, storage_format="nz")
    elif kern_trans_b:
        x2_spec = cannbotdsl.TensorSpec((b, n, k), dsl_dtype)
    else:
        x2_spec = cannbotdsl.TensorSpec((b, k, n), dsl_dtype)
    if bsf == 1:
        out_spec = cannbotdsl.TensorSpec((m, b, n), dsl_dtype)
    else:
        out_spec = cannbotdsl.TensorSpec((bsf, m, b // bsf, n), dsl_dtype)

    fn = cannbotdsl.compile(op.run, x1_spec, x2_spec, out_spec)
    assert fn.so_path and os.path.exists(fn.so_path), (
        f"L0 {perm_x1}x{perm_x2} bsf={bsf} mode={mode} fmt={x2_format}: "
        f"no .so artifact at {getattr(fn, 'so_path', None)}"
    )
    fn.close()


# ---------------------------------------------------------------------------
# L1 — functional precision (NPU vs independent CPU golden)
# ---------------------------------------------------------------------------


def _l1_perm_dtype_batch():
    """Every perm x dtype pair once; the batch axis alternates b=1/b=4 so
    both sizes are exercised without the full cross product."""
    cases = []
    dtypes = (
        (torch.float16, "fp16"),
        (torch.bfloat16, "bf16"),
        (torch.float32, "fp32"),
    )
    perms = [
        (_PERM_A_BMM, _PERM_B_KMAJ),
        (_PERM_A_BMM, _PERM_B_NMAJ),
        (_PERM_A_TRANS, _PERM_B_KMAJ),
        (_PERM_A_TRANS, _PERM_B_NMAJ),
    ]
    for i, (perm_x1, perm_x2) in enumerate(perms):
        for j, (dtype, dname) in enumerate(dtypes):
            b = 1 if (i + j) % 2 == 0 else 4
            cases.append(
                pytest.param(
                    perm_x1,
                    perm_x2,
                    b,
                    dtype,
                    id=(f"l1-{_perm_tag(perm_x1, perm_x2)}-{dname}-b{b}"),
                )
            )
    return cases


@pytest.mark.npu
@pytest.mark.parametrize("perm_x1,perm_x2,b,dtype", _l1_perm_dtype_batch())
def test_l1_perm_dtype_batch(perm_x1, perm_x2, b, dtype):
    pytest.importorskip("torch_npu")
    import torch_npu  # noqa: F401, PLC0415

    m, n, k = 96, 128, 128
    ctx = f"l1 {_perm_tag(perm_x1, perm_x2)} b={b} dtype={dtype}"
    x1, x2_phys, _ = make_nd_inputs(b, m, n, k, dtype, perm_x1, perm_x2)
    _run_and_check(x1, x2_phys.npu(), x2_phys, perm_x1, perm_x2, 1, dtype, 0, ctx)


_NZ_CASES = [
    pytest.param(torch.float16, "3d", 2, 96, 128, 128, id="l1-nz-fp16-3d"),
    pytest.param(torch.float16, "panel", 2, 96, 128, 128, id="l1-nz-fp16-panel"),
    pytest.param(torch.float16, "plain", 2, 96, 128, 128, id="l1-nz-fp16-plain-bridge"),
    pytest.param(torch.bfloat16, "3d", 2, 96, 128, 128, id="l1-nz-bf16-3d"),
    pytest.param(torch.float16, "3d", 2, 96, 64, 64, id="l1-nz-fp16-k64"),
]


@pytest.mark.npu
@pytest.mark.parametrize("dtype,form,b,m,n,k", _NZ_CASES)
def test_l1_nz(dtype, form, b, m, n, k):
    """NZ weight path: 3-D FRACTAL_NZ, the 5-D panel over an NZ allocation
    and the 5-D plain-ND panel (wrapper format bridge)."""
    pytest.importorskip("torch_npu")
    import torch_npu

    c0 = 8 if dtype == torch.float32 else 16
    perm_x1, perm_x2 = _PERM_A_TRANS, _PERM_B_KMAJ
    gen = torch.Generator(device="cpu").manual_seed(0)
    x1 = torch.randn(m, b, k, generator=gen, dtype=dtype)
    x2_logical = torch.randn(b, k, n, generator=gen, dtype=dtype)
    nz = torch_npu.npu_format_cast(x2_logical.npu(), 29)
    if form == "3d":
        x2_npu = nz
    elif form == "panel":
        x2_npu = nz.view(b, n // c0, k // 16, 16, c0)
    else:  # plain: a 5-D panel over plain ND storage (format-bridge path)
        x2_npu = (
            x2_logical.reshape(b, k // 16, 16, n // c0, c0)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
            .npu()
        )
    ctx = f"l1 nz {form} dtype={dtype} b={b} m={m} n={n} k={k}"
    _run_and_check(
        x1,
        x2_npu,
        x2_logical,
        perm_x1,
        perm_x2,
        1,
        dtype,
        0,
        ctx,
        x2_format="nz",
    )


_HF32_CASES = [
    pytest.param(_PERM_A_TRANS, _PERM_B_KMAJ, 2, id="l1-hf32-t2-trans-kmaj"),
    pytest.param(_PERM_A_BMM, _PERM_B_NMAJ, 2, id="l1-hf32-t2-bmm-nmaj"),
    pytest.param(_PERM_A_BMM, _PERM_B_NMAJ, 3, id="l1-hf32-t3-bmm-nmaj-neven"),
]


@pytest.mark.npu
@pytest.mark.parametrize("perm_x1,perm_x2,cube_math_type", _HF32_CASES)
def test_l1_hf32(perm_x1, perm_x2, cube_math_type):
    """HF32 routing: type 2 and type 3 (NEAREST_EVEN fallback on dav-3510,
    numerically identical to type 2 — the toolchain has no NEAREST_AWAY)."""
    pytest.importorskip("torch_npu")
    import torch_npu  # noqa: F401

    b, m, n, k = 2, 96, 128, 128
    dtype = torch.float32
    ctx = f"l1 hf32 t{cube_math_type} {_perm_tag(perm_x1, perm_x2)} b={b} dtype={dtype}"
    x1, x2_phys, _ = make_nd_inputs(b, m, n, k, dtype, perm_x1, perm_x2)
    _run_and_check(
        x1,
        x2_phys.npu(),
        x2_phys,
        perm_x1,
        perm_x2,
        1,
        dtype,
        cube_math_type,
        ctx,
    )


_BSF_CASES = [
    pytest.param(8, 2, torch.float16, id="l1-bsf2-b8-fp16"),
    pytest.param(8, 4, torch.float16, id="l1-bsf4-b8-fp16"),
    pytest.param(8, 8, torch.float16, id="l1-bsf-b8-full-fp16"),
    pytest.param(6, 3, torch.float16, id="l1-bsf3-b6-prime-fp16"),
]


@pytest.mark.npu
@pytest.mark.parametrize("b,bsf,dtype", _BSF_CASES)
def test_l1_bsf(b, bsf, dtype):
    """batch_split_factor in {2, 4, B, prime factor of B} (bsf=1 covered by
    the perm/dtype/batch matrix)."""
    pytest.importorskip("torch_npu")
    import torch_npu  # noqa: F401

    perm_x1, perm_x2 = _PERM_A_BMM, _PERM_B_KMAJ
    m, n, k = 96, 128, 128
    ctx = f"l1 bsf={bsf} b={b} dtype={dtype}"
    x1, x2_phys, _ = make_nd_inputs(b, m, n, k, dtype, perm_x1, perm_x2)
    _run_and_check(x1, x2_phys.npu(), x2_phys, perm_x1, perm_x2, bsf, dtype, 0, ctx)


# ---------------------------------------------------------------------------
# L2 — boundary
# ---------------------------------------------------------------------------

_TAIL_CASES = [
    # M/N > 256 with non-16-aligned tails (the FIXPIPE clip path).
    pytest.param(
        1, 300, 128, 64, torch.float16, _PERM_A_BMM, _PERM_B_KMAJ, 1, id="l2-tail-m300"
    ),
    pytest.param(
        1, 128, 300, 64, torch.float16, _PERM_A_BMM, _PERM_B_KMAJ, 1, id="l2-tail-n300"
    ),
    # All three axes ragged together, across the dtype set.
    pytest.param(
        1,
        300,
        300,
        300,
        torch.float16,
        _PERM_A_BMM,
        _PERM_B_KMAJ,
        1,
        id="l2-tail-mnk300-triple",
    ),
    pytest.param(
        3,
        300,
        300,
        300,
        torch.bfloat16,
        _PERM_A_BMM,
        _PERM_B_KMAJ,
        1,
        id="l2-tail-mnk300-bf16",
    ),
    # Tails + perm_x1 swap + multi-batch A-side skip-read.
    pytest.param(
        4,
        300,
        300,
        1600,
        torch.float16,
        _PERM_A_TRANS,
        _PERM_B_KMAJ,
        1,
        id="l2-tail-mnk-b4-skipread",
    ),
    # K spanning several k_l1 segments.
    pytest.param(
        2,
        128,
        128,
        1600,
        torch.float16,
        _PERM_A_BMM,
        _PERM_B_KMAJ,
        1,
        id="l2-k1600-multiseg",
    ),
]


@pytest.mark.npu
@pytest.mark.parametrize("b,m,n,k,dtype,perm_x1,perm_x2,bsf", _TAIL_CASES)
def test_l2_tails(b, m, n, k, dtype, perm_x1, perm_x2, bsf):
    pytest.importorskip("torch_npu")
    import torch_npu  # noqa: F401

    ctx = f"l2 tail b={b} m={m} n={n} k={k} dtype={dtype} {_perm_tag(perm_x1, perm_x2)}"
    x1, x2_phys, _ = make_nd_inputs(b, m, n, k, dtype, perm_x1, perm_x2)
    _run_and_check(x1, x2_phys.npu(), x2_phys, perm_x1, perm_x2, bsf, dtype, 0, ctx)


# fp32 K-tail fill granularity: the nd2nz on-the-fly zero fill covers a
# K tail only to the c0 boundary
# (c0 = 32//dtype_bytes: 16 for fp16/bf16, 8 for fp32), so the historical
# 16-granular K bound/clip read a deterministic 8-element dirty gap
# ([ceil8(K), ceil16(K))) out of L1 whenever K%16 in [1..8] — ~100% of the
# output elements corrupted for fp32 storage (KEEP_DTYPE and HF32 alike),
# both trans_b paths. The tiling now bounds the K reduction at ceil_c0(K)
# and the kernel clips k_eff at c0, so this domain must match the golden.
# 16-bit dtypes are immune (c0 = 16) and run as controls; K=501 (K%16=5)
# additionally exercises the active zero-fill sub-domain ([K, ceil8(K))),
# and K=40/33 hit the base_k = k_aligned shortcut at an 8-aligned K bound.
_R6_KT_CASES = [
    # fp32 K tails at the c0=8 granularity: K%16 in [1..8] on both trans_b
    # paths (K=1000: %8; K=501: %5), plus HF32 and a small-K shortcut case.
    pytest.param(
        4,
        1000,
        1000,
        1000,
        torch.float32,
        _PERM_A_BMM,
        _PERM_B_KMAJ,
        0,
        7100,
        id="l2-ktail-k1000-fp32-kmaj",
    ),
    pytest.param(
        4,
        1000,
        1000,
        501,
        torch.float32,
        _PERM_A_BMM,
        _PERM_B_NMAJ,
        0,
        7601,
        id="l2-ktail-k501-fp32-nmaj",
    ),
    pytest.param(
        4,
        1000,
        1000,
        1000,
        torch.float32,
        _PERM_A_BMM,
        _PERM_B_KMAJ,
        2,
        7100,
        id="l2-ktail-k1000-hf32-kmaj",
    ),
    pytest.param(
        4,
        64,
        64,
        33,
        torch.float32,
        _PERM_A_BMM,
        _PERM_B_NMAJ,
        0,
        7033,
        id="l2-ktail-k33-fp32-nmaj",
    ),
]


@pytest.mark.npu
@pytest.mark.parametrize(
    "b,m,n,k,dtype,perm_x1,perm_x2,cube_math_type,seed", _R6_KT_CASES
)
def test_l2_ktail_c0_granularity(
    b, m, n, k, dtype, perm_x1, perm_x2, cube_math_type, seed
):
    """fp32 x K%16 in [1..8] must match the CPU golden (the nd2nz
    zero-fill granularity is c0, not 16 — see the case-list comment)."""
    pytest.importorskip("torch_npu")
    import torch_npu  # noqa: F401

    ctx = (
        f"l2 r6-ktail b={b} m={m} n={n} k={k} dtype={dtype} "
        f"{_perm_tag(perm_x1, perm_x2)} cmt={cube_math_type}"
    )
    x1, x2_phys, _ = make_nd_inputs(b, m, n, k, dtype, perm_x1, perm_x2, seed)
    _run_and_check(
        x1,
        x2_phys.npu(),
        x2_phys,
        perm_x1,
        perm_x2,
        1,
        dtype,
        cube_math_type,
        ctx,
    )


_EXTREME_CASES = [
    # Degenerate but non-empty M=1 / N=1.
    pytest.param(2, 1, 128, 128, torch.float16, _PERM_A_BMM, _PERM_B_KMAJ, id="l2-m1"),
    pytest.param(2, 96, 1, 128, torch.float16, _PERM_A_BMM, _PERM_B_NMAJ, id="l2-n1"),
    pytest.param(
        2, 96, 128, 1, torch.float16, _PERM_A_BMM, _PERM_B_KMAJ, id="l2-k1-kmaj"
    ),
    # K%16!=0 on the K-major ND path: the wrapper zero-pads x2 to ceil16(K),
    # so the L1 staging tail rows read real zeros instead of L1 residue.
    pytest.param(
        2, 96, 128, 17, torch.float16, _PERM_A_BMM, _PERM_B_KMAJ, id="l2-k17-kmaj"
    ),
    pytest.param(
        2, 96, 128, 1, torch.float16, _PERM_A_BMM, _PERM_B_NMAJ, id="l2-k1-nmaj"
    ),
    # B=1 x bsf=1 x perm_x1=[1,0,2] ((M,1,K) ~ 2-D).
    pytest.param(
        1, 96, 128, 128, torch.float16, _PERM_A_TRANS, _PERM_B_NMAJ, id="l2-b1-skipread"
    ),
    # Large B (multi-core load spread) and tiny/skinny geometries.
    pytest.param(
        128, 128, 128, 64, torch.float16, _PERM_A_BMM, _PERM_B_KMAJ, id="l2-b128"
    ),
    pytest.param(
        2, 17, 8, 16, torch.float16, _PERM_A_BMM, _PERM_B_KMAJ, id="l2-tiny-m17-n8-k16"
    ),
    pytest.param(
        2, 32, 2048, 2048, torch.float16, _PERM_A_BMM, _PERM_B_KMAJ, id="l2-skinny-m32"
    ),
]


@pytest.mark.npu
@pytest.mark.parametrize("b,m,n,k,dtype,perm_x1,perm_x2", _EXTREME_CASES)
def test_l2_extremes(b, m, n, k, dtype, perm_x1, perm_x2):
    pytest.importorskip("torch_npu")
    import torch_npu  # noqa: F401

    ctx = f"l2 extreme b={b} m={m} n={n} k={k} dtype={dtype} {_perm_tag(perm_x1, perm_x2)}"
    x1, x2_phys, _ = make_nd_inputs(b, m, n, k, dtype, perm_x1, perm_x2)
    _run_and_check(x1, x2_phys.npu(), x2_phys, perm_x1, perm_x2, 1, dtype, 0, ctx)


_OUT_CASES = [
    # out= in-place writes, all four perms (aligned).
    pytest.param(
        4, 64, 64, 64, torch.float16, _PERM_A_BMM, _PERM_B_KMAJ, 1, id="l2-out-bmm-kmaj"
    ),
    pytest.param(
        4, 64, 64, 64, torch.float16, _PERM_A_BMM, _PERM_B_NMAJ, 1, id="l2-out-bmm-nmaj"
    ),
    pytest.param(
        4,
        64,
        64,
        64,
        torch.float16,
        _PERM_A_TRANS,
        _PERM_B_KMAJ,
        1,
        id="l2-out-trans-kmaj",
    ),
    pytest.param(
        4,
        64,
        64,
        64,
        torch.float16,
        _PERM_A_TRANS,
        _PERM_B_NMAJ,
        1,
        id="l2-out-trans-nmaj",
    ),
    # Tail + perm_x1 swap + multi-segment K in one out= run.
    pytest.param(
        4,
        300,
        300,
        1600,
        torch.float16,
        _PERM_A_TRANS,
        _PERM_B_KMAJ,
        1,
        id="l2-out-tail-b4-skipread",
    ),
    # bsf>1 out shape (bsf, M, B*N//bsf).
    pytest.param(
        4,
        64,
        64,
        64,
        torch.float32,
        _PERM_A_BMM,
        _PERM_B_KMAJ,
        2,
        id="l2-out-bsf2-fp32",
    ),
]


@pytest.mark.npu
@pytest.mark.parametrize("b,m,n,k,dtype,perm_x1,perm_x2,bsf", _OUT_CASES)
def test_l2_out_inplace(b, m, n, k, dtype, perm_x1, perm_x2, bsf):
    """out= pre-allocated in-place write: the NaN-prefilled buffer must be
    fully overwritten (no stale hole) and match the golden."""
    pytest.importorskip("torch_npu")
    import torch_npu  # noqa: F401

    ctx = f"l2 out= b={b} m={m} n={n} k={k} bsf={bsf} dtype={dtype} {_perm_tag(perm_x1, perm_x2)}"
    x1, x2_phys, _ = make_nd_inputs(b, m, n, k, dtype, perm_x1, perm_x2)
    out_shape = (m, b, n) if bsf == 1 else (bsf, m, (b // bsf) * n)
    out = torch.full(out_shape, float("nan"), dtype=dtype).npu()
    got = _run_and_check(
        x1,
        x2_phys.npu(),
        x2_phys,
        perm_x1,
        perm_x2,
        bsf,
        dtype,
        0,
        ctx,
        out=out,
    )
    assert got is out, f"{ctx}: wrapper must return the provided out tensor"
    assert not bool(torch.isnan(got.cpu().float()).any()), (
        f"{ctx}: out= buffer not fully overwritten"
    )


def _rejection_cases():
    """Rejection paths: one case per validated input rule.

    Each entry: (id, builder, needle). builders run lazily on NPU tensors;
    every path must raise ValueError/TypeError containing the needle.
    """
    import torch_npu

    dev = "npu"
    f16 = torch.float16
    f32 = torch.float32

    def mk(p1, p2):
        if p1 == _PERM_A_TRANS:
            x1 = torch.randn(32, 2, 32, dtype=f16, device=dev)
        else:
            x1 = torch.randn(2, 32, 32, dtype=f16, device=dev)
        if p2 == _PERM_B_NMAJ:
            x2 = (
                torch.randn(2, 32, 32, dtype=f16, device=dev)
                .transpose(-1, -2)
                .contiguous()
            )
        else:
            x2 = torch.randn(2, 32, 32, dtype=f16, device=dev)
        return x1, x2

    x1, x2 = mk(_PERM_A_BMM, _PERM_B_KMAJ)
    nz_ok = torch_npu.npu_format_cast(
        torch.randn(2, 64, 128, dtype=f16, device=dev), 29
    )
    x1_nz = torch.randn(96, 2, 64, dtype=f16, device=dev)  # (M, B, K)

    return [
        # rank / shape
        pytest.param(
            lambda: transpose_batch_mat_mul(
                torch.randn(4, 8, dtype=f16, device=dev), x2
            ),
            "3-D",
            id="v-x1-2d",
        ),
        pytest.param(
            lambda: transpose_batch_mat_mul(
                x1, torch.randn(2, 31, 32, dtype=f16, device=dev)
            ),
            "must be",
            id="v-x2-shape",
        ),
        # perms / perm_y / x2_format
        pytest.param(
            lambda: transpose_batch_mat_mul(x1, x2, perm_x1=(2, 0, 1)),
            "perm_x1",
            id="v-perm-x1",
        ),
        pytest.param(
            lambda: transpose_batch_mat_mul(x1, x2, perm_x2=(1, 2, 0)),
            "perm_x2",
            id="v-perm-x2",
        ),
        pytest.param(
            lambda: transpose_batch_mat_mul(x1, x2, perm_y=(0, 1, 2)),
            "perm_y",
            id="v-perm-y",
        ),
        pytest.param(
            lambda: transpose_batch_mat_mul(x1, x2, x2_format="fractal"),
            "x2_format",
            id="v-x2-format",
        ),
        # dtype
        pytest.param(
            lambda: transpose_batch_mat_mul(x1, x2.to(torch.bfloat16)),
            "same dtype",
            id="v-dtype-mismatch",
        ),
        pytest.param(
            lambda: transpose_batch_mat_mul(x1.to(torch.int32), x2.to(torch.int32)),
            "float16",
            id="v-dtype-int",
        ),
        # 4. cube_math_type
        pytest.param(
            lambda: transpose_batch_mat_mul(x1, x2, cube_math_type=1),
            "USE_FP16",
            id="v-cmt-1-use-fp16",
        ),
        pytest.param(
            lambda: transpose_batch_mat_mul(x1, x2, cube_math_type=4),
            "0..3",
            id="v-cmt-4",
        ),
        pytest.param(
            lambda: transpose_batch_mat_mul(x1, x2, cube_math_type=2),
            "float32",
            id="v-cmt-hf32-fp16",
        ),
        # bias / scale
        pytest.param(
            lambda: transpose_batch_mat_mul(
                x1, x2, bias=torch.randn(32, dtype=f16, device=dev)
            ),
            "bias",
            id="v-bias",
        ),
        pytest.param(
            lambda: transpose_batch_mat_mul(x1, x2, scale=1.0), "scale", id="v-scale"
        ),
        # bsf
        pytest.param(
            lambda: transpose_batch_mat_mul(x1, x2, batch_split_factor=3),
            "batch_split_factor",
            id="v-bsf-nondiv",
        ),  # B=2, bsf=3
        pytest.param(
            lambda: transpose_batch_mat_mul(x1, x2, batch_split_factor=4),
            "batch_split_factor",
            id="v-bsf-gt-b",
        ),
        # K/batch mismatch
        pytest.param(
            lambda: transpose_batch_mat_mul(
                torch.randn(2, 32, 64, dtype=f16, device=dev),
                torch.randn(2, 32, 32, dtype=f16, device=dev),
            ),
            "must be",
            id="v-k-mismatch",
        ),
        pytest.param(
            lambda: transpose_batch_mat_mul(
                torch.randn(2, 32, 32, dtype=f16, device=dev),
                torch.randn(3, 32, 32, dtype=f16, device=dev),
            ),
            "must be",
            id="v-batch-mismatch",
        ),
        # 8. NZ rejections
        pytest.param(
            lambda: transpose_batch_mat_mul(
                torch.randn(2, 96, 64, dtype=f16, device=dev),
                nz_ok,
                perm_x1=_PERM_A_BMM,
                x2_format="nz",
            ),
            "weightNZ",
            id="v-nz-perm-x1",
        ),
        pytest.param(
            lambda: transpose_batch_mat_mul(
                torch.randn(96, 2, 50, dtype=f16, device=dev),
                torch.randn(2, 8, 3, 16, 16, dtype=f16, device=dev),
                perm_x1=_PERM_A_TRANS,
                x2_format="nz",
            ),
            "K % 16",
            id="v-nz-k-unaligned",
        ),
        pytest.param(
            lambda: transpose_batch_mat_mul(
                x1_nz,
                torch.randn(2, 8, 4, 8, 16, dtype=f16, device=dev),
                perm_x1=_PERM_A_TRANS,
                x2_format="nz",
            ),
            "panel",
            id="v-nz-bad-panel",
        ),
        pytest.param(
            lambda: transpose_batch_mat_mul(
                torch.randn(96, 2, 64, dtype=f32, device=dev),
                torch_npu.npu_format_cast(
                    torch.randn(2, 64, 128, dtype=f32, device=dev), 29
                ),
                perm_x1=_PERM_A_TRANS,
                x2_format="nz",
            ),
            "float32",
            id="v-nz-fp32",
        ),
        # out contract
        pytest.param(
            lambda: transpose_batch_mat_mul(
                x1, x2, out=torch.randn(32, 2, 31, dtype=f16, device=dev)
            ),
            "out shape",
            id="v-out-shape",
        ),
        pytest.param(
            lambda: transpose_batch_mat_mul(
                x1, x2, out=torch.empty(32, 2, 32, dtype=f32, device=dev)
            ),
            "out dtype",
            id="v-out-dtype",
        ),
        # empty tensors / inner-axis cap
        pytest.param(
            lambda: transpose_batch_mat_mul(
                torch.randn(2, 0, 32, dtype=f16, device=dev),
                torch.randn(2, 32, 32, dtype=f16, device=dev),
            ),
            "empty",
            id="v-empty",
        ),
        pytest.param(
            lambda: transpose_batch_mat_mul(
                torch.randn(32, 2048, 32, dtype=f16, device=dev),
                torch.randn(2048, 32, 32, dtype=f16, device=dev),
                perm_x1=_PERM_A_TRANS,
            ),
            "inner-axis",
            id="v-inner-axis-cap",
        ),
        # non-contiguous x1 / non-canonical x2 view
        pytest.param(
            lambda: transpose_batch_mat_mul(
                torch.randn(32, 4, 32, dtype=f16, device=dev).transpose(0, 1),
                torch.randn(4, 32, 32, dtype=f16, device=dev),
            ),
            "x1 must be contiguous",
            id="v-x1-noncontig",
        ),
    ]


@pytest.mark.npu
@pytest.mark.parametrize("builder,needle", _rejection_cases())
def test_l2_rejections(builder, needle):
    """Every rejected input must fail cleanly (host exception, no launch)."""
    pytest.importorskip("torch_npu")
    with pytest.raises((ValueError, TypeError)) as exc_info:
        builder()
    assert needle in str(exc_info.value), (
        f"rejection message mismatch: expected {needle!r} in "
        f"{str(exc_info.value).splitlines()[-1]!r}"
    )
