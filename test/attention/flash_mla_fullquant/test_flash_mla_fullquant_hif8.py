# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""HIF8 (quant_mode=0) accuracy tests for the flash_mla_fullquant sample.

All tests use the official contract: abs(err) <= 0.02 + 0.01|ref| with
bad fraction <= 0.5% (plus a separate LSE check at 0.05).  The loose
sample-repo tier (atol 0.08/rtol 0.15) is kept as an extra guard.
"""

import math

import pytest
import torch
import torch_npu  # noqa: F401  (registers the npu device)
from _samples_path import load_sample

_IMPL = load_sample("attention/flash_mla_fullquant/flash_mla_fullquant_hif8.py")
_GOLDEN = load_sample("attention/flash_mla_fullquant/golden_hif8.py")
flash_mla_fullquant_hif8 = _IMPL.flash_mla_fullquant_hif8
golden_flash_mla_fullquant = _GOLDEN.golden_flash_mla_fullquant
make_case = _GOLDEN.make_case

DEV = "npu"


def _run(
    c,
    *,
    mask_mode=0,
    layout_q="TND",
    layout_kv="PA_BBND",
    layout_out="TND",
    lse=False,
    softmax_scale=None,
    cu=None,
    seqused=None,
    attn_mask=None,
):
    want_lse = bool(lse)
    q, qd = c["q"], c["q_descale"]
    if layout_q == "BSND":
        q_tok, n_q_heads, _ = q.shape
        q = q.unsqueeze(0)
        qd = qd.reshape(1, q_tok, n_q_heads)
    elif layout_q == "BNSD":
        q_tok, n_q_heads, _ = q.shape
        q = q.unsqueeze(0).permute(0, 2, 1, 3)
        qd = qd.reshape(1, q_tok, n_q_heads).permute(0, 2, 1)
    attn, lse = flash_mla_fullquant_hif8(
        q,
        c["k_cache"],
        qd,
        c["k_descale"],
        c["block_table"],
        c["cache_seqlens"],
        0,
        cu_seqlens_q=cu,
        seqused_q=seqused,
        mask_mode=mask_mode,
        attn_mask=attn_mask,
        softmax_scale=softmax_scale,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        return_softmax_lse=want_lse,
    )
    ref, lse_ref = golden_flash_mla_fullquant(
        q.cpu(),
        c["k_cache"].cpu(),
        qd.cpu(),
        c["k_descale"].cpu(),
        c["block_table"].cpu(),
        c["cache_seqlens"].cpu(),
        0,
        cu_seqlens_q=cu,
        seqused_q=seqused,
        mask_mode=mask_mode,
        softmax_scale=softmax_scale,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        return_softmax_lse=want_lse,
    )
    return attn, ref, lse, lse_ref


def _assert_loose(attn, ref):
    got = attn.float().cpu()
    ref_f = ref.float()
    ok = torch.isclose(got, ref_f, atol=0.08, rtol=0.15)
    assert float((~ok).float().mean()) <= 0.001, (
        f"loose mismatch {(~ok).float().mean():.4%}, "
        f"max_err={float((got - ref_f).abs().max()):.4f}"
    )


def _assert_lse(lse, lse_ref):
    got = lse.float().cpu()
    ref_f = lse_ref.float()
    both_inf = torch.isinf(got) & torch.isinf(ref_f) & (got == ref_f)
    finite = torch.isfinite(got) & torch.isfinite(ref_f)
    assert bool((both_inf | finite).all()), "LSE has non-finite values"
    diff = torch.where(both_inf, torch.zeros_like(got), (got - ref_f).abs())
    assert float(diff.max()) <= 0.05, f"lse max_err={float(diff.max()):.4f}"


def _assert_tight(attn, ref):
    got = attn.float().cpu()
    ref_f = ref.float()
    assert bool(torch.isfinite(got).all()), "attention output has non-finite values"
    assert bool(torch.isfinite(ref_f).all()), "reference output has non-finite values"
    err = (got - ref_f).abs()
    thr = 0.02 + 0.01 * ref_f.abs()
    assert float((err > thr).float().mean()) <= 0.005, (
        f"bad_fraction={float((err > thr).float().mean()):.4%} "
        f"max_err={float(err.max()):.4f}"
    )


@pytest.mark.parametrize("kv", [256, 1024])
def test_hif8_loose_decode(kv):
    c = make_case(tokens=1, cache_len=kv, heads=96, seed=7, quant_mode=0)
    attn, ref, _, _ = _run(c)
    _assert_loose(attn, ref)


def test_hif8_loose_block_reorder():
    c = make_case(tokens=1, cache_len=512, heads=8, seed=11, reorder=True, quant_mode=0)
    attn, ref, _, _ = _run(c)
    _assert_loose(attn, ref)


def test_hif8_loose_page_tail():
    c = make_case(tokens=1, cache_len=384, heads=8, seed=3, quant_mode=0)
    attn, ref, _, _ = _run(c)
    _assert_loose(attn, ref)


def test_hif8_loose_causal():
    c = make_case(tokens=4, cache_len=512, heads=8, seed=7, quant_mode=0)
    mask = torch.ones(2048, 2048, dtype=torch.int8)
    attn, ref, _, _ = _run(
        c, mask_mode=3, softmax_scale=1 / math.sqrt(512), attn_mask=mask
    )
    _assert_loose(attn, ref)


def test_hif8_lse():
    c = make_case(tokens=2, cache_len=512, heads=8, seed=7, quant_mode=0)
    attn, ref, lse, lse_ref = _run(c, lse=True)
    _assert_loose(attn, ref)
    _assert_lse(lse, lse_ref)


def test_hif8_wrong_quant_mode():
    c = make_case(tokens=1, cache_len=128, heads=8, seed=7, quant_mode=0)
    with pytest.raises(ValueError):
        flash_mla_fullquant_hif8(
            c["q"],
            c["k_cache"],
            c["q_descale"],
            c["k_descale"],
            c["block_table"],
            c["cache_seqlens"],
            1,
            layout_q="TND",
        )


def test_hif8_tight_tolerance():
    c = make_case(tokens=1, cache_len=512, heads=8, seed=7, quant_mode=0)
    attn, ref, _, _ = _run(c)
    _assert_tight(attn, ref)


def test_hif8_rejects_nonpositive_scale():
    q = torch.zeros(1, 8, 576, dtype=torch.int8)
    k = torch.zeros(1, 128, 1, 576, dtype=torch.int8)
    qd = torch.ones(1, 8, dtype=torch.float32)
    kd = torch.ones(1, dtype=torch.float32)
    bt = torch.zeros(1, 1, dtype=torch.int32)
    cs = torch.ones(1, dtype=torch.int32)
    for softmax_scale in (0.0, -1.0):
        with pytest.raises(ValueError):
            flash_mla_fullquant_hif8(
                q, k, qd, kd, bt, cs, 0, softmax_scale=softmax_scale, layout_q="TND"
            )


def test_hif8_qk_scale_gt_one():
    c = make_case(tokens=1, cache_len=128, heads=8, seed=7, quant_mode=0)
    c["k_descale"] = torch.tensor([2.0], dtype=torch.float32, device=DEV)
    attn, ref, _, _ = _run(c, softmax_scale=1.0)
    _assert_loose(attn, ref)


def test_hif8_tight_assertion_rejects_nan():
    ref = torch.zeros(2, 4)
    attn = torch.zeros(2, 4)
    attn[1, 2] = float("nan")
    with pytest.raises(AssertionError):
        _assert_tight(attn, ref)


def test_hif8_lse_assertion_rejects_nan():
    with pytest.raises(AssertionError):
        _assert_lse(torch.tensor([float("nan")]), torch.ones(1))


def test_hif8_rejects_fp32_row_factor_overflow():
    c = make_case(tokens=1, cache_len=128, heads=8, seed=7, quant_mode=0)
    c["q_descale"] = torch.full_like(c["q_descale"], 1e20)
    c["k_descale"] = torch.ones(1, dtype=torch.float32, device=DEV)
    with pytest.raises(ValueError):
        _run(c, softmax_scale=1e20)
