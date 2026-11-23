# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""NPU precision tests for the native HIF8 FlashAttention sample (FP8 full-quant schedule)."""

import os

import pytest
import torch
from _samples_path import load_sample

npu_flash_atten_hif8_fullquant = load_sample(
    "attention/flash_attn_hif8_fullquant/flash_atten_hif8_fullquant.py"
).npu_flash_atten_hif8_fullquant
checker = load_sample("attention/flash_attn_hif8_fullquant/hif8_check.py")


def _converter():
    import torch_npu

    return torch_npu._C._cd


def _to_hif8(conv, values):
    src = values.npu()
    dst = torch.empty(src.numel(), dtype=torch.uint8, device="npu")
    conv.cast_to_fp8_noalloc(src.view(1, -1), dst.view(1, -1), conv.DType.hifloat8)
    return dst.view(src.shape).view(torch.int8)


def _from_hif8(conv, codes):
    return (
        conv.cast_from_fp8(
            codes.view(torch.uint8).view(1, -1),
            conv.DType.hifloat8,
            conv.DType.float32,
        )
        .view(codes.shape)
        .cpu()
    )


_CASES = [
    # batch, q_heads, kv_heads, q_len, kv_len, causal, p_scale, layout, dense_kv
    pytest.param(1, 1, 1, 128, 256, False, 1.0, "BNSD", None, id="b1_h1_s128x256"),
    pytest.param(1, 2, 1, 128, 256, False, 1.0, "BNSD", None, id="gqa2"),
    pytest.param(2, 2, 2, 128, 256, False, 1.0, "BNSD", None, id="b2_mha"),
    pytest.param(1, 1, 1, 128, 512, False, 1.0, "BNSD", None, id="s512"),
    pytest.param(1, 2, 1, 96, 1024, False, 1.0, "BNSD", None, id="long_tail"),
    pytest.param(1, 1, 1, 160, 384, True, 1.0, "BNSD", None, id="causal_tail"),
    pytest.param(1, 2, 1, 128, 768, True, 1.0, "BNSD", None, id="causal_gqa"),
    pytest.param(1, 2, 2, 128, 256, False, 1.0, "BSND", None, id="bsnd_lse"),
    # BSND storage holds 512 KV tokens per batch, but each batch consumes a
    # different effective length. The physical batch stride must come from
    # the padded storage, not seqused_kv[0].
    pytest.param(
        2, 1, 1, 128, 512, False, 1.0, "BSND", (256, 512), id="bsnd_varlen_kv"
    ),
    pytest.param(1, 1, 1, 64, 128, True, 1.0, "BNSD", None, id="tiny_causal"),
    pytest.param(1, 2, 1, 128, 768, False, 4.0, "BNSD", None, id="pscale4"),
    pytest.param(1, 1, 1, 256, 512, False, 1.0, "BNSD", None, id="lse_multi"),
    pytest.param(1, 2, 1, 128, 768, True, 4.0, "BNSD", None, id="causal_lse"),
    pytest.param(2, 2, 1, 128, 256, False, 1.0, "TND", None, id="tnd_varlen"),
    # Ragged per-sequence lengths: batch 1 starts mid-tile, so its tail
    # tile's writeback must not spill over batch 0's rows.
    pytest.param(2, 2, 1, 151, 768, False, 1.0, "TND", None, id="tnd_ragged"),
    pytest.param(2, 1, 1, 151, 768, True, 1.0, "TND", None, id="tnd_ragged_causal"),
    pytest.param(1, 20, 2, 256, 2048, False, 1.0, "TND", None, id="gqa10_perf_shape"),
    # Flash Decode: one Q tile, long KV, base grid leaves cores idle.
    pytest.param(1, 1, 1, 1, 8192, False, 1.0, "BNSD", None, id="fd_decode_lse"),
    pytest.param(1, 1, 1, 1, 8192, True, 1.0, "TND", None, id="fd_tnd_decode_lse"),
    pytest.param(4, 2, 1, 1, 4096, False, 1.0, "BNSD", None, id="fd_b4_gqa_lse"),
    # Combined QK scale = 2 (2.0 * 1.0 * 1.0) with q > kv causal: the leading
    # 32 rows are fully masked, so the scaled NEG_MAX sentinel saturates to
    # -inf while the invalid-row threshold must still clear those rows.
    pytest.param(1, 1, 1, 160, 128, True, 4.0, "BNSD", None, id="qk_scale_gt_one"),
]

# id -> (descale_q, descale_k, descale_v, softmax_scale)
_SCALE_OVERRIDES = {
    "qk_scale_gt_one": (2.0, 1.0, 1.0, 1.0),
}


@pytest.mark.npu
@pytest.mark.parametrize(
    "batch,q_heads,kv_heads,q_len,kv_len,causal,p_scale,layout,dense_kv",
    _CASES,
)
def test_hif8_fullquant_precision(
    request,
    batch,
    q_heads,
    kv_heads,
    q_len,
    kv_len,
    causal,
    p_scale,
    layout,
    dense_kv,
):
    pytest.importorskip("torch_npu")
    if not os.environ.get("ASCEND_HOME_PATH"):
        pytest.skip("ASCEND_HOME_PATH not set")
    if torch.npu.device_count() == 0:
        pytest.skip("no NPU device is available")
    if "Ascend950" not in torch.npu.get_device_name(0):
        pytest.skip("HIF8 attention requires Ascend950")

    conv = _converter()
    gen = torch.Generator(device="cpu").manual_seed(7)
    with_lse = "lse" in request.node.callspec.id
    if layout == "TND":
        q_lens = [q_len, max(q_len - 64, 1)]
        kv_lens = [kv_len, max(kv_len - 128, 256)]
        total_q = sum(q_lens)
        q_flat = _to_hif8(
            conv, torch.randn((total_q, q_heads, 128), generator=gen) * 0.3
        )
        k_parts, v_parts = [], []
        for length in kv_lens:
            k_parts.append(
                _to_hif8(
                    conv, torch.randn((length, kv_heads, 128), generator=gen) * 0.3
                )
            )
            v_parts.append(
                _to_hif8(
                    conv, torch.randn((length, kv_heads, 128), generator=gen) * 0.3
                )
            )
        q_in, k_in, v_in = q_flat, torch.cat(k_parts, 0), torch.cat(v_parts, 0)
        seqused_kv = torch.tensor(kv_lens, dtype=torch.int32).npu()
        seqused_q = torch.tensor(q_lens, dtype=torch.int32).npu()
    else:
        q_lens = [q_len] * batch
        kv_lens = list(dense_kv) if dense_kv is not None else [kv_len] * batch
        q = _to_hif8(
            conv, torch.randn((batch, q_heads, q_len, 128), generator=gen) * 0.3
        )
        k = _to_hif8(
            conv, torch.randn((batch, kv_heads, kv_len, 128), generator=gen) * 0.3
        )
        v = _to_hif8(
            conv, torch.randn((batch, kv_heads, kv_len, 128), generator=gen) * 0.3
        )
        if layout == "BSND":
            q_in = q.permute(0, 2, 1, 3).contiguous()
            k_in = k.permute(0, 2, 1, 3).contiguous()
            v_in = v.permute(0, 2, 1, 3).contiguous()
        else:
            q_in, k_in, v_in = q, k, v
        seqused_kv = (
            torch.tensor(kv_lens, dtype=torch.int32).npu()
            if dense_kv is not None
            else None
        )
        seqused_q = None
    dq_value, dk_value, dv_value, softmax_value = _SCALE_OVERRIDES.get(
        request.node.callspec.id, (0.5, 0.25, 2.0, 128**-0.5)
    )
    dq = torch.tensor([dq_value], dtype=torch.float32).npu()
    dk = torch.tensor([dk_value], dtype=torch.float32).npu()
    dv = torch.tensor([dv_value], dtype=torch.float32).npu()
    ps = torch.tensor([p_scale], dtype=torch.float32).npu()

    out, out_lse = npu_flash_atten_hif8_fullquant(
        q_in,
        k_in,
        v_in,
        dq,
        dk,
        dv,
        p_scale=ps if p_scale != 1.0 else None,
        softmax_scale=softmax_value,
        mask_mode=3 if causal else 0,
        layout=layout,
        seqused_kv=seqused_kv,
        seqused_q=seqused_q if layout == "TND" else None,
        return_softmax_lse=with_lse,
    )
    torch.npu.synchronize()

    if layout == "TND":
        q_golden, k_golden, v_golden = q_in, k_in, v_in
        golden_batch = len(q_lens)
    else:
        q_golden, k_golden, v_golden = q, k, v
        golden_batch = batch
    qf_raw = _from_hif8(conv, q_golden).float()
    kf_raw = _from_hif8(conv, k_golden).float()
    vf = _from_hif8(conv, v_golden).float() * dv_value
    # Kernel/AscendC scale path: max(raw) * (deqQ * deqK * softmax_scale)
    # and per-row exp((raw - raw_max) * combined_scale ...). Emulate by
    # scaling raw Q rows: q*deqQ, k*deqK keeps the same products while
    # matching the kernel's max(s)*c rounding.
    qf = qf_raw * dq_value
    kf = kf_raw * dk_value
    if layout == "TND":
        group = q_heads // kv_heads
        per_batch_scores = []
        for b in range(golden_batch):
            q_lo = sum(q_lens[:b])
            q_hi = sum(q_lens[: b + 1])
            kv_lo = sum(kv_lens[:b])
            kv_hi = sum(kv_lens[: b + 1])
            qs = qf[q_lo:q_hi]
            ks = kf[kv_lo:kv_hi]
            vs = vf[kv_lo:kv_hi]
            # GQA: Q head qh attends through KV head qh // group.
            sc_b = torch.empty(q_heads, qs.shape[0], ks.shape[0])
            for qh in range(q_heads):
                sc_b[qh] = (
                    torch.matmul(qs[:, qh], ks[:, qh // group].t()) * softmax_value
                )
            per_batch_scores.append((sc_b, vs))
    else:
        scores = torch.matmul(qf, kf.transpose(-1, -2)) * softmax_value
        if dense_kv is not None:
            # The kernel bounds each batch's KV read by seqused_kv; mask the
            # storage tail so the golden sees the same key set.
            for b in range(batch):
                visible_kv = kv_lens[b]
                if visible_kv < kv_len:
                    scores[b, :, :, visible_kv:] = float("-inf")
    if causal:
        if layout == "TND":
            # Per-sequence causal masks: each batch has its own
            # q_len/kv_len pair, hence its own diagonal offset.
            masked = []
            for b, (sc_b, vs) in enumerate(per_batch_scores):
                mask = torch.ones(sc_b.shape[-2:], dtype=torch.bool)
                sc_b = sc_b.masked_fill(
                    torch.triu(mask, diagonal=1 + kv_lens[b] - q_lens[b]),
                    float("-inf"),
                )
                masked.append((sc_b, vs))
            per_batch_scores = masked
        else:
            mask = torch.ones(scores.shape[-2:], dtype=torch.bool)
            if dense_kv is None:
                scores = scores.masked_fill(
                    torch.triu(mask, diagonal=1 + kv_len - q_len), float("-inf")
                )
            else:
                for b in range(batch):
                    scores[b] = scores[b].masked_fill(
                        torch.triu(mask, diagonal=1 + kv_lens[b] - q_lens[b]),
                        float("-inf"),
                    )
    if layout == "TND":
        group = q_heads // kv_heads
        ref_parts = []
        for sc_b, vs in per_batch_scores:
            # q > kv causal can leave fully masked rows: softmax is NaN
            # there, while the kernel zeroes the row.
            visible = torch.isfinite(sc_b).any(dim=-1, keepdim=True)
            pr = torch.softmax(sc_b, dim=-1) * float(p_scale)
            pr = torch.where(visible, pr, torch.zeros_like(pr))
            pb = torch.empty(pr.numel(), dtype=torch.uint8, device="npu")
            conv.cast_to_fp8_noalloc(
                pr.reshape(1, -1).npu(), pb.view(1, -1), conv.DType.hifloat8
            )
            pd = (
                conv.cast_from_fp8(
                    pb.view(1, -1), conv.DType.hifloat8, conv.DType.float32
                )
                .view(pr.shape)
                .cpu()
            )
            ob = torch.empty(q_heads, pd.shape[1], vs.shape[-1])
            for qh in range(q_heads):
                ob[qh] = torch.matmul(pd[qh], vs[:, qh // group])
            ref_parts.append(ob / float(p_scale))
        ref = torch.cat([part.transpose(0, 1) for part in ref_parts], dim=0)
    else:
        # q > kv causal leaves leading rows with no visible key: softmax is
        # NaN there, while the kernel zeroes the row.
        visible = torch.isfinite(scores).any(dim=-1, keepdim=True)
        probs = torch.softmax(scores, dim=-1) * float(p_scale)
        probs = torch.where(visible, probs, torch.zeros_like(probs))
        pb = torch.empty(probs.numel(), dtype=torch.uint8, device="npu")
        conv.cast_to_fp8_noalloc(
            probs.reshape(1, -1).npu(), pb.view(1, -1), conv.DType.hifloat8
        )
        p_dec = (
            conv.cast_from_fp8(pb.view(1, -1), conv.DType.hifloat8, conv.DType.float32)
            .view(probs.shape)
            .cpu()
        )
        ref = torch.matmul(p_dec, vf) / float(p_scale)
    if layout == "BSND":
        ref = ref.permute(0, 2, 1, 3)

    got = out.cpu().float()
    if layout == "TND":
        got = got[: sum(q_lens)]
    assert out.dtype == torch.bfloat16
    ok = torch.isclose(got, ref, atol=0.08, rtol=0.15)
    mismatch = float((~ok).float().mean())
    assert mismatch < 0.02, (
        f"HIF8 native mismatch {mismatch:.4%}, max_err="
        f"{float((got - ref).abs().max()):.4e}"
    )
    if with_lse:
        # AscendC contract: TND exports (N, T), dense exports (B, N, S).
        got_lse = out_lse.cpu()
        if layout == "TND":
            got_lse = got_lse[:, : sum(q_lens)]
            ref_lse = torch.cat(
                [torch.logsumexp(sc, dim=-1) for sc, _ in per_batch_scores],
                dim=1,
            )
        else:
            ref_lse = torch.logsumexp(scores, dim=-1)
        # The golden scores carry the per-sequence causal masks already, so
        # causal TND uses the same logsumexp; every LSE case keeps kv >= q, so
        # no row is fully masked (those would export +inf).
        lse_ok = torch.isclose(got_lse, ref_lse, atol=5e-2, rtol=5e-2)
        assert float((~lse_ok).float().mean()) < 0.02, (
            f"LSE mismatch {float((~lse_ok).float().mean()):.4%}"
        )


# ---------------------------------------------------------------------------
# Host-side contract, exercised without a device.
# ---------------------------------------------------------------------------


def _dense_qkv(batch=1, q_heads=2, kv_heads=1, q_len=128, kv_len=256, layout="BNSD"):
    if layout == "BNSD":
        shape_q = (batch, q_heads, q_len, 128)
        shape_kv = (batch, kv_heads, kv_len, 128)
        q = torch.zeros(shape_q, dtype=torch.int8)
        k = torch.zeros(shape_kv, dtype=torch.int8)
        v = torch.zeros(shape_kv, dtype=torch.int8)
    else:
        q = torch.zeros((batch, q_len, q_heads, 128), dtype=torch.int8)
        k = torch.zeros((batch, kv_len, kv_heads, 128), dtype=torch.int8)
        v = torch.zeros((batch, kv_len, kv_heads, 128), dtype=torch.int8)
    return q, k, v


def test_contract_accepts_dense_layouts():
    for layout in ("BNSD", "BSND"):
        q, k, v = _dense_qkv(layout=layout)
        info = checker.validate_static(q, k, v, 0, layout, None, None)
        assert info["batch"] == 1
        assert info["n_q_heads"] == 2
        assert info["kv_heads"] == 1
        assert (info["q_extent"], info["kv_extent"]) == (128, 256)


def test_contract_accepts_tnd():
    q = torch.zeros((384, 2, 128), dtype=torch.int8)
    k = torch.zeros((512, 1, 128), dtype=torch.int8)
    v = torch.zeros((512, 1, 128), dtype=torch.int8)
    lengths = torch.tensor([256, 128, 128], dtype=torch.int32)
    info = checker.validate_static(q, k, v, 3, "TND", lengths, None)
    assert info["batch"] == 3
    assert (info["q_extent"], info["kv_extent"]) == (384, 512)


def test_contract_rejects_non_int8():
    q, k, v = _dense_qkv()
    with pytest.raises(TypeError, match="int8"):
        checker.validate_static(q.float(), k, v, 0, "BNSD", None, None)


def test_contract_rejects_unknown_layout():
    q, k, v = _dense_qkv()
    with pytest.raises(ValueError, match="layout"):
        checker.validate_static(q, k, v, 0, "NZ", None, None)


def test_contract_rejects_unknown_mask():
    q, k, v = _dense_qkv()
    with pytest.raises(ValueError, match="mask_mode"):
        checker.validate_static(q, k, v, 1, "BNSD", None, None)


def test_contract_rejects_head_dim():
    q, k, v = _dense_qkv()
    with pytest.raises(ValueError, match="D=128"):
        checker.validate_static(
            q[..., :64], k[..., :64], v[..., :64], 0, "BNSD", None, None
        )


def test_contract_rejects_head_group():
    q, k, v = _dense_qkv(q_heads=3, kv_heads=2)
    with pytest.raises(ValueError, match="Nq % Nkv"):
        checker.validate_static(q, k, v, 0, "BNSD", None, None)


def test_contract_rejects_key_value_shape():
    q, k, _ = _dense_qkv()
    with pytest.raises(ValueError, match="key/value"):
        checker.validate_static(q, k, k[:, :, :128], 0, "BNSD", None, None)


def test_contract_rejects_dense_batch_mismatch():
    q, k, v = _dense_qkv(batch=2)
    with pytest.raises(ValueError, match="batch mismatch"):
        checker.validate_static(q, k[:1], v[:1], 0, "BNSD", None, None)


def test_contract_requires_tnd_lengths():
    q = torch.zeros((128, 2, 128), dtype=torch.int8)
    k = torch.zeros((256, 1, 128), dtype=torch.int8)
    with pytest.raises(ValueError, match="seqused_kv"):
        checker.validate_static(q, k, k, 0, "TND", None, None)


def test_lengths_accept_packed_with_spare_storage():
    checker.validate_lengths([128, 128], [256, 256], 384, 512, True)


def test_lengths_reject_zero():
    with pytest.raises(ValueError, match="at least 1"):
        checker.validate_lengths([128, 0], [256, 256], 256, 512, True)


def test_lengths_reject_packed_overflow():
    with pytest.raises(ValueError, match="beyond"):
        checker.validate_lengths([256, 256], [512, 512], 384, 512, True)


def test_lengths_reject_dense_overflow():
    with pytest.raises(ValueError, match="exceeds"):
        checker.validate_lengths([512], [256], 128, 256, False)


def test_lengths_reject_count_mismatch():
    with pytest.raises(ValueError, match="differ"):
        checker.validate_lengths([128, 128], [256], 256, 256, True)


def test_scales_accept_valid():
    checker.validate_scales(0.5, 0.25, 2.0, 4.0, 128**-0.5)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_scales_reject_invalid_descaled(bad):
    with pytest.raises(ValueError):
        checker.validate_scales(bad, 0.25, 2.0, 1.0, 1.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_scales_reject_invalid_p_scale(bad):
    with pytest.raises(ValueError):
        checker.validate_scales(0.5, 0.25, 2.0, bad, 1.0)


def test_scales_accept_combined_unit():
    # softmax_scale * descale_q * descale_k == 1 stays valid.
    checker.validate_scales(2.0, 1.0, 2.0, 1.0, 0.5)


@pytest.mark.parametrize(
    "descale_q,descale_k,softmax_scale",
    [
        # > 1 is supported: the sentinel saturates to -inf like AscendC.
        (2.0, 1.0, None),
        (1.0, 1.0, 1.5),
    ],
)
def test_scales_accept_combined_above_one(descale_q, descale_k, softmax_scale):
    checker.validate_scales(descale_q, descale_k, 1.0, 1.0, softmax_scale)


@pytest.mark.parametrize(
    "descale_q,descale_k,softmax_scale",
    [
        # qk_scale == 0: the masked-row sentinel and every row maximum
        # collapse to the same value, so valid rows would be erased.
        (0.5, 1.0, 0.0),
        # negative softmax_scale lands below the supported range.
        (1.0, 1.0, -0.5),
    ],
)
def test_scales_reject_combined_non_positive(descale_q, descale_k, softmax_scale):
    with pytest.raises(ValueError, match="must be positive"):
        checker.validate_scales(descale_q, descale_k, 1.0, 1.0, softmax_scale)
