# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.


from types import SimpleNamespace

import pytest
import torch

from _samples_path import load_sample

_sample = load_sample("flash_attn/flash_attn_metadata.py")
AIC_CORE_NUM = _sample.AIC_CORE_NUM
AIV_CORE_NUM = _sample.AIV_CORE_NUM
HEAD_METADATA_STRIDE = _sample.HEAD_METADATA_STRIDE
flash_attn_metadata = _sample.flash_attn_metadata


HEAD_DIM = 128


@pytest.fixture
def host_only(monkeypatch):
    def unexpected_device_access():
        pytest.fail("invalid metadata inputs must be rejected before NPU access")

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_device=unexpected_device_access),
        raising=False,
    )


@pytest.mark.parametrize("head_dim", [0, 32, 80, 192, 257])
def test_metadata_rejects_unsupported_head_dim_before_device(head_dim, host_only):
    with pytest.raises(
        ValueError, match="head_dim must be one of 64, 128, 256"
    ):
        flash_attn_metadata(
            8, 2, head_dim, batch_size=1,
            max_seqlen_q=64, max_seqlen_kv=128,
        )


@pytest.mark.parametrize("layout_kv", ["PA_BNBD", "PA_BBND", "PA_NZ"])
def test_pa_metadata_requires_seqused_kv_before_device(layout_kv, host_only):
    with pytest.raises(ValueError, match="seqused_kv is required for PA layout"):
        flash_attn_metadata(
            2, 1, 64,
            batch_size=1,
            max_seqlen_q=2,
            max_seqlen_kv=16,
            layout_q="BNSD",
            layout_kv=layout_kv,
            layout_out="BNSD",
        )


@pytest.mark.parametrize("layout_kv", ["PA_BNBD", "PA_BBND", "PA_NZ"])
def test_pa_metadata_rejects_cu_seqlens_kv_before_device(layout_kv, host_only):
    with pytest.raises(ValueError, match="cu_seqlens_kv must be None for PA layout"):
        flash_attn_metadata(
            2, 1, 64,
            cu_seqlens_kv=torch.tensor([0, 16], dtype=torch.int32),
            seqused_kv=torch.tensor([16], dtype=torch.int32),
            batch_size=1,
            max_seqlen_q=2,
            max_seqlen_kv=16,
            layout_q="BNSD",
            layout_kv=layout_kv,
            layout_out="BNSD",
        )


def test_pa_metadata_rejects_explicit_batch_size_mismatch_before_device(host_only):
    with pytest.raises(ValueError, match="batch_size must match inferred batch size"):
        flash_attn_metadata(
            2, 1, 64,
            seqused_q=torch.tensor([2], dtype=torch.int32),
            seqused_kv=torch.tensor([16], dtype=torch.int32),
            batch_size=2,
            max_seqlen_q=2,
            max_seqlen_kv=16,
            layout_q="BNSD",
            layout_kv="PA_BNBD",
            layout_out="BNSD",
        )


@pytest.mark.parametrize("head_dim", [True, 64.0, "64"])
def test_metadata_rejects_noninteger_head_dim_before_device(head_dim, host_only):
    with pytest.raises(TypeError, match="head_dim must be an integer"):
        flash_attn_metadata(2, 1, head_dim, batch_size=1)


def test_tnd_metadata_requires_cu_seqlens_q_before_device(host_only):
    with pytest.raises(ValueError, match="cu_seqlens_q is required"):
        flash_attn_metadata(2, 1, 64, batch_size=1, layout_q="TND")


@pytest.mark.parametrize("name,length", [
    ("seqused_kv", 1), ("cu_seqlens_kv", 2), ("cu_seqlens_q", 2),
])
def test_metadata_rejects_sequence_batch_mismatch_before_device(name, length, host_only):
    sequences = {
        "seqused_q": torch.tensor([2, 2], dtype=torch.int32),
        name: torch.zeros(length, dtype=torch.int32),
    }
    with pytest.raises(ValueError, match=f"{name} must be a 1D tensor of length"):
        flash_attn_metadata(2, 1, 64, batch_size=2, **sequences)


@pytest.mark.parametrize("name", ["seqused_q", "seqused_kv", "cu_seqlens_q", "cu_seqlens_kv"])
def test_metadata_rejects_scalar_sequence_before_device(name, host_only):
    with pytest.raises(ValueError, match=f"{name} must be a 1D tensor"):
        flash_attn_metadata(
            2, 1, 64, batch_size=1,
            **{name: torch.tensor(0, dtype=torch.int32)},
        )


@pytest.mark.npu
@pytest.mark.parametrize("layout", ["BSND", "BNSD", "TND"])
@pytest.mark.parametrize("mask_mode,win_left,win_right", [
    (0, -1, -1), (3, -1, -1), (4, 31, 7), (4, 4096, 7), (4, -1, 0), (4, 0, -1),
])
@pytest.mark.parametrize("batch,num_heads_q,num_heads_kv", [
    (1, 1, 1), (3, 12, 3), (5, 32, 4),
])
def test_flash_attn_metadata(
    layout, mask_mode, win_left, win_right,
    batch, num_heads_q, num_heads_kv,
):
    pytest.importorskip("torch_npu")
    q_lengths = [119 + 31 * i for i in range(batch)]
    kv_lengths = [512 + 128 * i for i in range(batch)]
    cu_q = (
        torch.tensor([0] + q_lengths, dtype=torch.int32).cumsum(0).to(torch.int32).npu()
        if layout == "TND" else None
    )
    cu_kv = (
        torch.tensor([0] + kv_lengths, dtype=torch.int32).cumsum(0).to(torch.int32).npu()
        if layout == "TND" else None
    )
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        metadata = flash_attn_metadata(
            num_heads_q, num_heads_kv, HEAD_DIM,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            batch_size=batch,
            max_seqlen_q=256,
            max_seqlen_kv=1024,
            mask_mode=mask_mode, win_left=win_left, win_right=win_right,
            layout_q=layout,
            layout_kv=layout,
            layout_out=layout,
        )
    stream.synchronize()
    assert metadata.dtype == torch.int32
    expected_elements = (
        1 + batch * num_heads_kv * (AIC_CORE_NUM + AIV_CORE_NUM)
    ) * HEAD_METADATA_STRIDE
    expected_elements = ((expected_elements + 4095) // 4096) * 4096
    assert metadata.numel() == expected_elements
    head = metadata[:16].cpu()
    assert 0 < int(head[0]) <= batch * num_heads_kv
    assert int(head[1]) in (0, 1)
    assert int(head[2]) > 0 and int(head[3]) > 0


@pytest.mark.npu
@pytest.mark.parametrize("mask_mode,win_left,win_right", [
    (0, -1, -1), (3, -1, -1), (4, 31, 7), (4, 4096, 7), (4, -1, 0), (4, 0, -1),
])
@pytest.mark.parametrize("num_heads_q,num_heads_kv", [(1, 1), (12, 3)])
def test_seqused_overrides_cu_seqlens(
    mask_mode, win_left, win_right, num_heads_q, num_heads_kv,
):
    pytest.importorskip("torch_npu")
    cu_q = torch.tensor([0, 160, 400], dtype=torch.int32, device="npu")
    cu_kv = torch.tensor([0, 512, 1536], dtype=torch.int32, device="npu")
    used_q = torch.tensor([119, 181], dtype=torch.int32, device="npu")
    used_kv = torch.tensor([400, 700], dtype=torch.int32, device="npu")
    attrs = dict(
        max_seqlen_q=256, max_seqlen_kv=1024,
        mask_mode=mask_mode, win_left=win_left, win_right=win_right,
        layout_q="TND", layout_kv="TND", layout_out="TND",
    )
    metadata = flash_attn_metadata(
        num_heads_q, num_heads_kv, HEAD_DIM,
        cu_seqlens_q=cu_q, cu_seqlens_kv=cu_kv,
        seqused_q=used_q, seqused_kv=used_kv,
        **attrs,
    )
    expected = flash_attn_metadata(
        num_heads_q, num_heads_kv, HEAD_DIM,
        cu_seqlens_q=torch.tensor([0, 119, 300], dtype=torch.int32, device="npu"),
        cu_seqlens_kv=torch.tensor([0, 400, 1100], dtype=torch.int32, device="npu"),
        **attrs,
    )
    torch.npu.synchronize()
    metadata, expected = metadata.cpu(), expected.cpu()
    assert metadata[0] == expected[0]
    # Only the sections in the header are defined; allocation padding is not.
    active = (1 + int(metadata[0]) * (36 + 72)) * 16
    assert torch.equal(metadata[:active], expected[:active])


@pytest.mark.npu
# Preserve all nine PR D128 cases; add six targeted D64/D128 cases.
@pytest.mark.parametrize("layout,mask_mode,win_left,win_right,head_dim", [
    (layout, mask, left, right, 128)
    for layout in ("BSND", "BNSD", "TND")
    for mask, left, right in ((0, -1, -1), (3, -1, -1), (4, 4096, 7))
] + [
    (layout, 0 if layout == "BSND" else 3, -1, -1, d)
    for d in (64, 128) for layout in ("BSND", "BNSD", "TND")
])
def test_split_kv_sections(layout, mask_mode, win_left, win_right, head_dim):
    pytest.importorskip("torch_npu")
    cu_q = torch.tensor([0, 17, 36, 57, 80], dtype=torch.int32, device="npu") if layout == "TND" else None
    cu_kv = torch.tensor([0, 16384, 32896, 49536, 66304], dtype=torch.int32, device="npu") if layout == "TND" else None
    metadata = flash_attn_metadata(
        32, 4, head_dim, batch_size=4,
        cu_seqlens_q=cu_q, cu_seqlens_kv=cu_kv,
        seqused_q=torch.tensor([17, 19, 21, 23], dtype=torch.int32, device="npu"),
        seqused_kv=torch.tensor([16384, 16512, 16640, 16768], dtype=torch.int32, device="npu"),
        max_seqlen_q=23, max_seqlen_kv=16768,
        mask_mode=mask_mode, win_left=win_left, win_right=win_right,
        layout_q=layout, layout_kv=layout, layout_out=layout,
    )
    head = metadata[:4].cpu()
    assert int(head[0]) >= 1
    if head_dim >= 128:
        assert int(head[0]) > 1
    assert int(head[1]) == 1
