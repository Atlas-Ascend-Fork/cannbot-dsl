# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Accuracy tests for the flash_mla_fullquant FP8 sample (real assertions).

Covers the P0 matrix: quant modes,
page reorder, page tails, causal, layouts, multi-batch, LSE on/off.
"""

import math

import pytest
import torch
import torch_npu  # noqa: F401  (registers the npu device)
from _samples_path import load_sample

_IMPL = load_sample("attention/flash_mla_fullquant/flash_mla_fullquant_fp8.py")
_GOLDEN = load_sample("attention/flash_mla_fullquant/golden_fp8.py")
_CHECK = load_sample("attention/flash_mla_fullquant/flash_mla_fullquant_checker.py")
flash_mla_fullquant_fp8 = _IMPL.flash_mla_fullquant_fp8
flash_mla_fullquant_fp8_metadata = _IMPL.flash_mla_fullquant_fp8_metadata
golden_flash_mla_fullquant = _GOLDEN.golden_flash_mla_fullquant
make_case = _GOLDEN.make_case

DEV = "npu"
Q8 = torch.float8_e4m3fn


def _assert_lse(lse, lse_ref):
    got = lse.float().cpu()
    ref_f = lse_ref.float()
    both_inf = torch.isinf(got) & torch.isinf(ref_f) & (got == ref_f)
    finite = torch.isfinite(got) & torch.isfinite(ref_f)
    assert bool((both_inf | finite).all()), "LSE has non-finite values"
    diff = torch.where(both_inf, torch.zeros_like(got), (got - ref_f).abs())
    assert float(diff.max()) <= 0.05, f"lse max_err={float(diff.max()):.4f}"


def _assert_tight(attn, ref, lse=None, lse_ref=None, want_lse=False):
    got = attn.float().cpu()
    ref_f = ref.float()
    assert bool(torch.isfinite(got).all()), "attention output has non-finite values"
    assert bool(torch.isfinite(ref_f).all()), "reference output has non-finite values"
    err = (got - ref_f).abs()
    thr = 0.02 + 0.01 * ref_f.abs()
    bad = (err > thr).float().mean()
    assert float(bad) <= 0.005, (
        f"bad_fraction={float(bad):.4%} max_err={float(err.max()):.4f}"
    )
    if want_lse:
        _assert_lse(lse, lse_ref)
    elif lse is not None:
        assert lse.numel() == 0, "LSE must be empty when disabled"


def _run_case(
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
    {"BSND": None, "BNSD": None, "TND": c["q_descale"], "TND_NTD": c["q_descale"]}
    q = c["q"]
    qd = c["q_descale"]
    want_lse = bool(lse)
    if layout_q == "BSND":
        q_tok, n_q_heads, _ = q.shape
        q = q.unsqueeze(0)
        qd = qd.reshape(1, q_tok, n_q_heads)
    elif layout_q == "BNSD":
        q_tok, n_q_heads, _ = q.shape
        q = q.unsqueeze(0).permute(0, 2, 1, 3)
        qd = qd.reshape(1, q_tok, n_q_heads).permute(0, 2, 1)
    attn, lse = flash_mla_fullquant_fp8(
        q,
        c["k_cache"],
        qd,
        c["k_descale"],
        c["block_table"],
        c["cache_seqlens"],
        c["quant_mode"],
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
        c["quant_mode"],
        cu_seqlens_q=cu,
        seqused_q=seqused,
        mask_mode=mask_mode,
        softmax_scale=softmax_scale,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        return_softmax_lse=want_lse,
    )
    _assert_tight(attn, ref, lse, lse_ref, want_lse)
    return attn


def test_p0_fp8_decode_256():
    c = make_case(tokens=1, cache_len=256, heads=96, seed=7)
    _run_case(c)


def test_p0_fp8_decode_1024():
    c = make_case(tokens=1, cache_len=1024, heads=96, seed=7)
    _run_case(c)


@pytest.mark.parametrize("kv", [4096, 16384])
def test_large_kv_fd_split(kv):
    """KV >= 4096 drives the full flash-decode split (n_slices=16).

    Everything else in this file stops at 1024 (n_slices=4), so this is the
    only coverage of the 16-way partial split + merge, including the
    1-block-per-slice edge (4096) and the multi-block case (16384).
    """
    c = make_case(tokens=1, cache_len=kv, heads=96, seed=7)
    _run_case(c, lse=True)


def test_p0_block_reorder():
    c = make_case(tokens=1, cache_len=512, heads=96, seed=11, reorder=True)
    _run_case(c)


def test_p0_sequential_pages():
    c = make_case(tokens=1, cache_len=512, heads=96, seed=5, reorder=False)
    _run_case(c)


@pytest.mark.parametrize("kv", [127, 128, 129, 384])
def test_page_tails(kv):
    c = make_case(tokens=1, cache_len=kv, heads=8, seed=3)
    _run_case(c)


@pytest.mark.parametrize("heads", [1, 8, 96, 128])
def test_q_heads(heads):
    c = make_case(tokens=1, cache_len=256, heads=heads, seed=7)
    _run_case(c)


@pytest.mark.parametrize("tokens", [1, 2, 16])
def test_q_s(tokens):
    c = make_case(tokens=tokens, cache_len=512, heads=8, seed=7)
    _run_case(c)


def test_p0_lse():
    c = make_case(tokens=2, cache_len=512, heads=8, seed=7)
    _run_case(c, lse=True)


def test_p0_lse_off_empty():
    c = make_case(tokens=1, cache_len=256, heads=8, seed=7)
    _run_case(c, lse=False)


def test_p0_causal():
    c = make_case(tokens=4, cache_len=512, heads=8, seed=7)
    mask = torch.ones(2048, 2048, dtype=torch.int8)
    _run_case(c, mask_mode=3, softmax_scale=1.0 / math.sqrt(512), attn_mask=mask)


def test_layouts_bsnd():
    c = make_case(tokens=2, cache_len=384, heads=8, seed=7)
    _run_case(c, layout_q="BSND", layout_out="BSND")


def test_layouts_bnsd_to_ntd():
    c = make_case(tokens=2, cache_len=384, heads=8, seed=7)
    _run_case(c, layout_q="BNSD", layout_out="NTD")


def test_layouts_tnd_ntd_out():
    c = make_case(tokens=2, cache_len=384, heads=8, seed=7)
    _run_case(c, layout_q="TND", layout_out="NTD")


def test_layouts_pa_bnbd():
    c = make_case(tokens=1, cache_len=384, heads=8, seed=7)
    # PA_BNBD view: (Bn, 1, Bs, 576) - same storage, different axes
    kc = c["k_cache"].permute(0, 2, 1, 3).contiguous()
    c2 = dict(c, k_cache=kc)
    _run_case(c2, layout_kv="PA_BNBD")


def test_multi_batch():
    c = make_case(tokens=1, cache_len=512, heads=8, seed=7, batch=2)
    cu = c["cu_seqlens_q"]
    _run_case(c, cu=cu)


def test_multi_batch_varlen():
    c = make_case(q_len_list=[1, 3], cache_len=384, heads=8, seed=7)
    _run_case(c, cu=c["cu_seqlens_q"], layout_q="TND")


def test_metadata_reuse():
    c = make_case(tokens=1, cache_len=512, heads=96, seed=7)
    md = flash_mla_fullquant_fp8_metadata(
        c["cache_seqlens"], 96, 1, c["quant_mode"], max_seqlen_q=1, layout_q="TND"
    )
    _run_case(c)
    # reuse path
    attn, _ = flash_mla_fullquant_fp8(
        c["q"],
        c["k_cache"],
        c["q_descale"],
        c["k_descale"],
        c["block_table"],
        c["cache_seqlens"],
        c["quant_mode"],
        metadata=md,
        layout_q="TND",
        layout_out="TND",
    )
    ref, _ = golden_flash_mla_fullquant(
        c["q"].cpu(),
        c["k_cache"].cpu(),
        c["q_descale"].cpu(),
        c["k_descale"].cpu(),
        c["block_table"].cpu(),
        c["cache_seqlens"].cpu(),
        c["quant_mode"],
        layout_q="TND",
        layout_out="TND",
    )
    _assert_tight(attn, ref)


@pytest.mark.parametrize(
    "arg",
    [
        ("bad_quant_mode", {"quant_mode": 2}),
        ("bad_mask", {"mask_mode": 1}),
        ("bad_layout_q", {"layout_q": "XYZ"}),
        ("bad_layout_out", {"layout_out": "XYZ"}),
    ],
)
def test_validation_errors(arg):
    _name, kw = arg
    c = make_case(tokens=1, cache_len=128, heads=8, seed=7)
    with pytest.raises((ValueError, TypeError)):
        flash_mla_fullquant_fp8(
            c["q"],
            c["k_cache"],
            c["q_descale"],
            c["k_descale"],
            c["block_table"],
            c["cache_seqlens"],
            c["quant_mode"],
            **kw,
        )


def test_causal_requires_mask():
    c = make_case(tokens=1, cache_len=128, heads=8, seed=7)
    with pytest.raises(ValueError):
        flash_mla_fullquant_fp8(
            c["q"],
            c["k_cache"],
            c["q_descale"],
            c["k_descale"],
            c["block_table"],
            c["cache_seqlens"],
            c["quant_mode"],
            mask_mode=3,
        )


def test_scale_contract_rejects_invalid():
    qd = torch.ones(2, 8)
    kd = torch.ones(1)
    for softmax_scale in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            _CHECK.validate_scales(qd, kd, softmax_scale)
    with pytest.raises(ValueError):
        _CHECK.validate_scales(torch.zeros(2, 8), kd, 1.0)
    with pytest.raises(ValueError):
        _CHECK.validate_scales(torch.full((2, 8), -1.0), kd, 1.0)
    with pytest.raises(ValueError):
        _CHECK.validate_scales(torch.full((2, 8), float("nan")), kd, 1.0)
    with pytest.raises(ValueError):
        _CHECK.validate_scales(qd, torch.zeros(1), 1.0)
    with pytest.raises(ValueError):
        _CHECK.validate_scales(qd, torch.tensor([float("inf")]), 1.0)
    # FP32 semantics: qk_scale itself overflows, and the per-row product
    # overflows or underflows even though the double-precision values pass.
    with pytest.raises(ValueError):
        _CHECK.validate_scales(torch.ones(2, 8), torch.full((1,), 1e20), 1e20)
    with pytest.raises(ValueError):
        _CHECK.validate_scales(torch.full((2, 8), 1e20), torch.ones(1), 1e20)
    with pytest.raises(ValueError):
        _CHECK.validate_scales(torch.full((2, 8), 1e-30), torch.full((1,), 1e-30), 1.0)


def test_scale_contract_accepts_positive():
    ss, kd, qk_scale = _CHECK.validate_scales(
        torch.ones(1, 8), torch.tensor([2.0]), 2.0
    )
    assert ss == 2.0
    assert kd == 2.0
    assert qk_scale == 4.0


def test_qk_scale_gt_one():
    # k_descale * softmax_scale = 2 exercises the scaled sentinel and
    # must not overflow the FP32 constant at build time.
    c = make_case(tokens=1, cache_len=128, heads=8, seed=7)
    c["k_descale"] = torch.tensor([2.0], dtype=torch.float32, device=DEV)
    _run_case(c, softmax_scale=1.0)


def test_multi_batch_varlen_kv():
    # Per-batch KV lengths differ; the padded page pool must address each
    # batch from its own base.
    c = make_case(q_len_list=[1, 2], kv_len_list=[256, 512], heads=8, seed=7)
    _run_case(c, cu=c["cu_seqlens_q"], layout_q="TND")


def test_tight_assertion_rejects_nan_output():
    ref = torch.zeros(2, 4)
    attn = torch.zeros(2, 4)
    attn[0, 0] = float("nan")
    with pytest.raises(AssertionError):
        _assert_tight(attn, ref)


def test_tight_assertion_rejects_nan_lse():
    attn = torch.zeros(1, 2)
    ref = torch.zeros(1, 2)
    with pytest.raises(AssertionError):
        _assert_tight(attn, ref, torch.tensor([float("nan")]), torch.ones(1), True)


def test_tight_assertion_allows_matching_inf_lse():
    attn = torch.zeros(1, 2)
    ref = torch.zeros(1, 2)
    inf = torch.tensor([float("inf")])
    _assert_tight(attn, ref, inf, inf.clone(), True)


def test_fp8_rejects_fp32_row_factor_overflow():
    c = make_case(tokens=1, cache_len=128, heads=8, seed=7)
    c["q_descale"] = torch.full_like(c["q_descale"], 1e20)
    c["k_descale"] = torch.ones(1, dtype=torch.float32, device=DEV)
    with pytest.raises(ValueError):
        _run_case(c, softmax_scale=1e20)


def test_fp8_rejects_fp32_row_factor_underflow():
    c = make_case(tokens=1, cache_len=128, heads=8, seed=7)
    c["q_descale"] = torch.full_like(c["q_descale"], 1e-30)
    c["k_descale"] = torch.full((1,), 1e-30, dtype=torch.float32, device=DEV)
    with pytest.raises(ValueError):
        _run_case(c, softmax_scale=1.0)
