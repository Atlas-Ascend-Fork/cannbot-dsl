# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""NPU precision tests for GQA MXFP8 prefill and decode kernels.

Formula:
    O = softmax(Q @ K^T / sqrt(D)) @ V

Q/K/V in this formula are dequantized MXFP8 inputs. The output reference also
models FP8 rounding of unnormalized P in KV subtiles; LSE uses FP32 logsumexp.

All cases use the same public test interface. ``q_scale_layout`` selects the
prefill or decode input layout; every case uses real MXFP8 block quantization.

Coverage:
  * Q scale layout: TND prefill and N2TGD decode
  * scale: per-block MXFP8
  * KV layout: contiguous and paged
  * mask: no-mask and causal modes, both with variable valid KV lengths
  * head dimension: 64, 72, 128 and 256
  * attention: MHA, GQA and MQA, including variable KV lengths
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
import subprocess
import sys

import pytest
import torch

from _samples_path import load_sample

checks = load_sample("attention/flash_attn_mxfp8_fullquant/mxfp8_check.py")

FP8_E4M3_MAX = 448.0
MX_BLOCK_ELEMENTS = 32
MX_PAIR_ELEMENTS = 64
MX_SCALE_LANES = 2
FP8_NZ_BLOCK = 32
VSCALE_NZ_BLOCK = 16

# Preserve the existing BF16 comparison policy during migration. These values
# belong to this MXFP8 test; the ordinary FP8 FA test uses different FP16 limits.
OUTPUT_RTOL = 0.0078125
OUTPUT_ATOL = 0.0001
PASS_PERCENT_LIMIT = 99.5
MAX_RELATIVE_ERROR_LIMIT = 10.0
RELATIVE_FLOOR = (1.0 / (1 << 14)) / 0.005
RELATIVE_EPSILON = 1e-10
LSE_ATOL = 2e-2
LSE_RTOL = 2e-3
DIAGNOSTIC_QUANTILE = 0.99
WORKER_TIMEOUT_SECONDS = 900
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttentionTileConfig:
    """Reference tile geometry used to model FP8 P rounding."""

    query_rows: int
    kv_tokens: int


@dataclass(frozen=True)
class AttentionTestData:
    """Generated tensors split by consumer, layout, and dtype."""

    # NPU inputs: already converted to the public kernel storage layouts.
    npu_query_tnd_fp8: torch.Tensor
    npu_key_cache_fp8: torch.Tensor
    npu_value_cache_fp8: torch.Tensor
    npu_query_scale_e8m0: torch.Tensor
    npu_key_scale_cache_e8m0: torch.Tensor
    npu_value_scale_cache_e8m0: torch.Tensor
    p_scale_fp32: torch.Tensor
    attention_mask_fp32: torch.Tensor
    block_table_i32: torch.Tensor
    kv_lengths_i64: torch.Tensor
    query_lengths_i64: torch.Tensor

    # CPU golden inputs: logical BNS tensors kept before PA/TND cache packing.
    cpu_query_bns_fp8: torch.Tensor
    cpu_key_bns_fp8: torch.Tensor
    cpu_value_bns_fp8: torch.Tensor
    cpu_query_scale_bns_e8m0: torch.Tensor
    cpu_key_scale_bns_e8m0: torch.Tensor
    cpu_value_scale_tiles_e8m0: torch.Tensor

    query_length_max: int
    query_capacity_tokens: int
    kv_capacity_tokens: int
    attention_scale_fp32: float


# ---------------------------------------------------------------------------
# Case configuration
# ---------------------------------------------------------------------------

CaseConfig = dict[str, object]

_CASE_FIELDS = (
    "name q_layout B N_q N_kv S_q S_kv D max_seqlen_q max_seqlen_kv "
    "enable_pa kv_cache_layout block_size mask_mode q_scale_layout "
    "p_scale enable_lse reverse_pages noncontiguous_pa"
).split()


def _case(*values):
    """Create one flat case row whose order is fixed by ``_CASE_FIELDS``."""
    assert len(values) == len(_CASE_FIELDS)
    return pytest.param(*values, id=values[0])


_CASES = [
    # Compact TND with a one-token tail in the last KV tile.
    _case(
        "Prefill_TND_1_4_4_65_257_128",
        "TND",
        1,
        4,
        4,
        (65,),
        (257,),
        128,
        65,
        257,
        False,
        "TND",
        512,
        0,
        "TND",
        1.0,
        False,
        False,
        False,
    ),
    # Full Prefill with a paged cache.
    _case(
        "Prefill_PABNBD_1_16_2_256_256_128_mask",
        "TND",
        1,
        16,
        2,
        (256,),
        (256,),
        128,
        256,
        256,
        True,
        "PA_BNBD",
        256,
        3,
        "TND",
        1.0,
        False,
        False,
        False,
    ),
    # Batched chunked Prefill with non-contiguous PA_BBND storage.
    _case(
        "Prefill_PABBND_4_16_4_128_512_128_mask",
        "TND",
        4,
        16,
        4,
        (128, 128, 128, 128),
        (512, 512, 512, 512),
        128,
        128,
        512,
        True,
        "PA_BBND",
        128,
        3,
        "TND",
        1.0,
        False,
        True,
        True,
    ),
    # PA_NZ Prefill with non-contiguous storage, reversed pages and LSE.
    _case(
        "Prefill_PANZ_1_4_2_65_193_128_mask",
        "TND",
        1,
        4,
        2,
        (65,),
        (193,),
        128,
        65,
        193,
        True,
        "PA_NZ",
        128,
        3,
        "TND",
        15.0,
        True,
        True,
        True,
    ),
    # D=72 Decode on compact TND KV.
    _case(
        "Decode_TND_1_4_2_4_128_72_mask",
        "TND",
        1,
        4,
        2,
        (4,),
        (128,),
        72,
        4,
        128,
        False,
        "TND",
        512,
        3,
        "N2TGD",
        1.0,
        False,
        False,
        False,
    ),
    # D=64 single-token Decode with non-contiguous PA_BNBD storage.
    _case(
        "Decode_PABNBD_1_64_4_1_1127_64_mask",
        "TND",
        1,
        64,
        4,
        (1,),
        (1127,),
        64,
        1,
        1127,
        True,
        "PA_BNBD",
        1024,
        3,
        "N2TGD",
        256.0,
        False,
        True,
        True,
    ),
    # Batched Decode with different context lengths.
    _case(
        "Decode_PABBND_2_64_4_1_3415_128",
        "TND",
        2,
        64,
        4,
        (1, 1),
        (3415, 979),
        128,
        1,
        3415,
        True,
        "PA_BBND",
        128,
        0,
        "N2TGD",
        1.0,
        False,
        False,
        False,
    ),
    # Long-context Decode with PA_NZ and MQA.
    _case(
        "Decode_PANZ_1_8_1_1_2933_128",
        "TND",
        1,
        8,
        1,
        (1,),
        (2933,),
        128,
        1,
        2933,
        True,
        "PA_NZ",
        512,
        0,
        "N2TGD",
        1.0,
        False,
        False,
        False,
    ),
    # D=256 and a batch with empty KV, with LSE enabled.
    _case(
        "Decode_PABNBD_2_1_1_1_1_256_mask",
        "TND",
        2,
        1,
        1,
        (1, 1),
        (1, 0),
        256,
        1,
        1,
        True,
        "PA_BNBD",
        64,
        3,
        "N2TGD",
        15.0,
        True,
        False,
        False,
    ),
    # Paged D=72 Prefill exercises head-dimension padding.
    _case(
        "Prefill_PABBND_1_4_2_65_321_72",
        "TND",
        1,
        4,
        2,
        (65,),
        (321,),
        72,
        65,
        321,
        True,
        "PA_BBND",
        256,
        0,
        "TND",
        1.0,
        False,
        False,
        True,
    ),
    # Batched multi-token Decode exercises nonempty D=256 work and LSE.
    _case(
        "Decode_PANZ_2_8_1_3_129_256",
        "TND",
        2,
        8,
        1,
        (3, 2),
        (129, 65),
        256,
        3,
        129,
        True,
        "PA_NZ",
        512,
        0,
        "N2TGD",
        448.0,
        True,
        False,
        False,
    ),
]

# ---------------------------------------------------------------------------
# Tile configuration helpers
# ---------------------------------------------------------------------------


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _mx_groups64(head_dim: int) -> int:
    return _ceil_div(head_dim, MX_PAIR_ELEMENTS)


def _mx_scale_cols(head_dim: int) -> int:
    return MX_SCALE_LANES * _mx_groups64(head_dim)


def _tile_config_for_head_dim(head_dim: int) -> AttentionTileConfig:
    """Keep the quantization reference independent of the implementation helpers.

    P is rounded to FP8 in KV subtiles. Changing that width can change the
    expected result, so the reference records the numerical algorithm explicitly:
    D=256 uses 128 KV tokens; D=64/72/128 use 256. Q storage reserves 128 rows
    per batch for Decode. D=72 uses the same compute width as padded D=128.
    """
    kv_tokens = {64: 256, 72: 256, 128: 256, 256: 128}
    if head_dim not in kv_tokens:
        raise ValueError(f"Unsupported reference head dimension: {head_dim}")
    return AttentionTileConfig(query_rows=128, kv_tokens=kv_tokens[head_dim])


# ---------------------------------------------------------------------------
# MXFP8 quantization helpers
# ---------------------------------------------------------------------------


def quantize_mxfp8_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last axis to MXFP8, padding to paired 32-element blocks."""
    logical_k = x.shape[-1]
    aligned_k = math.ceil(logical_k / MX_PAIR_ELEMENTS) * MX_PAIR_ELEMENTS
    if aligned_k != logical_k:
        x = torch.nn.functional.pad(x, (0, aligned_k - logical_k))
    xf = x.float().reshape(
        *x.shape[:-1], aligned_k // MX_BLOCK_ELEMENTS, MX_BLOCK_ELEMENTS
    )
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=2**-125)
    log_ratio = torch.log2(amax / FP8_E4M3_MAX)
    exponent = torch.ceil(log_ratio).clamp(min=-127, max=127).to(torch.int32)
    scale_byte = (exponent + 127).clamp(0, 255).to(torch.uint8)
    eff = torch.pow(torch.tensor(2.0), exponent.float())
    xq = (xf / eff).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    xq_fp8 = xq.to(torch.float8_e4m3fn).reshape(x.shape)[..., :logical_k]
    scales = scale_byte.squeeze(-1).reshape(
        *x.shape[:-1], aligned_k // MX_BLOCK_ELEMENTS
    )
    return xq_fp8, scales.view(torch.float8_e8m0fnu)


def dequantize_mxfp8_e4m3(
    q_fp8: torch.Tensor, scale_e8m0: torch.Tensor
) -> torch.Tensor:
    """Inverse of `quantize_mxfp8_e4m3`. Returns the logical-D FP32 tensor."""
    q_fp32 = q_fp8.float()
    logical_k = q_fp32.shape[-1]
    aligned_k = scale_e8m0.shape[-1] * MX_BLOCK_ELEMENTS
    if aligned_k != logical_k:
        q_fp32 = torch.nn.functional.pad(q_fp32, (0, aligned_k - logical_k))
    qr = q_fp32.reshape(
        *q_fp32.shape[:-1], aligned_k // MX_BLOCK_ELEMENTS, MX_BLOCK_ELEMENTS
    )
    eff = scale_e8m0.float().unsqueeze(-1)
    return (qr * eff).reshape(q_fp32.shape)[..., :logical_k]


def quantize_v_mxfp8(v_ref_fp32, tile_n=None):
    """Quantize V along the S2 token axis in 32-token MXFP8 groups.

    Return ``(v_fp8, sv_tiles)``.
    sv_tiles.shape == [B, N2, num_logical_tiles, G_V, D, 2]
    For a 256-token compute subLoop, G_V = tile_n // 64 = 4.
    PA may use a different ``tile_n`` equal to its physical page size; the
    final two lanes remain the paired-lane representation required by
    ``mx_scale_bnd``.
    """
    batch, kv_heads, padded_kv_tokens, head_dim = v_ref_fp32.shape
    if tile_n is None:
        tile_n = _tile_config_for_head_dim(head_dim).kv_tokens
    checks.check_v_quantization_extent(padded_kv_tokens, tile_n)
    num_tiles = padded_kv_tokens // tile_n
    v_scale_groups = tile_n // 64

    v_t = v_ref_fp32.transpose(-2, -1).contiguous()  # [B, N2, D, S2_padded]
    v_q_t, sv32 = quantize_mxfp8_e4m3(v_t)  # sv32: [B, N2, D, S2_padded//32]

    sv_tiles = (
        sv32.reshape(batch, kv_heads, head_dim, num_tiles, v_scale_groups, 2)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
    )  # [B, N2, num_tiles, G_V, D, 2]

    v_fp8 = v_q_t.transpose(-2, -1).contiguous()
    return v_fp8, sv_tiles


def dequantize_v_mxfp8(v_fp8, sv_tiles):
    """Inverse of quantize_v_mxfp8. Returns fp32."""
    batch, kv_heads, padded_kv_tokens, head_dim = v_fp8.shape

    sv32 = (
        sv_tiles.permute(0, 1, 4, 2, 3, 5)
        .contiguous()
        .reshape(batch, kv_heads, head_dim, padded_kv_tokens // 32)
    )
    v_q_t = v_fp8.transpose(-2, -1).contiguous()
    v_dq_t = dequantize_mxfp8_e4m3(v_q_t, sv32)
    return v_dq_t.transpose(-2, -1).contiguous()


# ---------------------------------------------------------------------------
# PA layout and storage helpers
# ---------------------------------------------------------------------------


def _fill_storage_poison(storage, poison):
    """Keep poison values as raw bytes for E8M0 storage gaps."""
    if storage.dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn):
        storage.view(torch.uint8).fill_(int(poison) & 0xFF)
    else:
        storage.fill_(poison)


def _make_pa_strided_view(tensor, mode, poison):
    """Copy a logical PA tensor into storage with page/head-axis gaps."""
    checks.check_pa_stride_mode(tensor, mode)
    if mode == checks.PAStride.CONTIGUOUS:
        return tensor

    backing_shape = list(tensor.shape)
    backing_shape[0] *= 2
    if mode == checks.PAStride.AXIS01:
        backing_shape[1] *= 2
    backing = tensor.new_empty(backing_shape)
    _fill_storage_poison(backing, poison)
    slices = [slice(None)] * tensor.ndim
    slices[0] = slice(0, None, 2)
    if mode == checks.PAStride.AXIS01:
        slices[1] = slice(0, None, 2)
    view = backing[tuple(slices)]
    checks.check_pa_strided_view(tensor, view, mode)
    view.copy_(tensor)
    return view


def _apply_noncontiguous_pa_stride(key, value, key_scale_e8m0, value_scale_e8m0, mode):
    """Apply the parsed non-contiguous PA stride after PA layout packing."""
    return (
        _make_pa_strided_view(key, mode, 1.0),
        _make_pa_strided_view(value, mode, -1.0),
        _make_pa_strided_view(key_scale_e8m0, mode, -128),
        _make_pa_strided_view(value_scale_e8m0, mode, 0),
    )


def _to_npu_preserve_pa_stride(tensor, mode, poison):
    """Transfer logical values while constructing the requested stride on NPU."""
    if mode == checks.PAStride.CONTIGUOUS:
        return tensor.npu()
    shape = tuple(int(dim) for dim in tensor.shape)
    strides = tuple(int(stride) for stride in tensor.stride())
    storage_size = 1 + sum((dim - 1) * stride for dim, stride in zip(shape, strides))
    storage = torch.empty((storage_size,), dtype=tensor.dtype, device="npu")
    _fill_storage_poison(storage, poison)
    view = torch.as_strided(storage, size=shape, stride=strides)
    if tensor.dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn):
        # torch_npu ViewCopy does not accept E8M0. Copy the encoded bytes;
        # the public tensor retains its native E8M0 dtype and exact strides.
        view.view(torch.uint8).copy_(tensor.view(torch.uint8).npu())
    else:
        view.copy_(tensor.npu())
    return view


def build_paged_kv(
    k_data,
    v_data,
    k_scale_flat,
    v_scale_tiles,
    seqused_kv_list,
    batch,
    kv_heads,
    head_dim,
    qk_scale_groups,
    block_size=128,
    kv_cache_layout="PA_BNBD",
    reverse_pages=False,
):
    """Build PA_BNBD, PA_BBND, or PA_NZ K/V and MXFP8 scale tensors.

    ``block_size`` is the PA physical page size (64/128/256/512/1024).  It is
    deliberately independent of the 256-token compute subLoop.

    Returns: (k_paged, v_paged, sk_paged, sv_paged, block_table, seqused_kv_npu)
    """
    checks.check_paged_kv_options(block_size, kv_cache_layout)
    d_blocks32 = _ceil_div(head_dim, FP8_NZ_BLOCK)
    mx_groups64 = _mx_groups64(head_dim)
    vscale_d_blocks16 = _ceil_div(head_dim, VSCALE_NZ_BLOCK)

    max_kv_tokens = max(seqused_kv_list)
    max_pages_per_batch = math.ceil(max_kv_tokens / block_size)
    num_pages = batch * max_pages_per_batch

    k_paged = torch.zeros(
        num_pages, kv_heads, block_size, head_dim, dtype=torch.float8_e4m3fn
    )
    v_paged = torch.zeros(
        num_pages, kv_heads, block_size, head_dim, dtype=torch.float8_e4m3fn
    )
    sk_paged = torch.full(
        (num_pages, kv_heads, block_size, qk_scale_groups, 2),
        1.0,
        dtype=torch.float8_e8m0fnu,
    )
    sv_paged = torch.full(
        (num_pages, kv_heads, block_size // 64, head_dim, 2),
        1.0,
        dtype=torch.float8_e8m0fnu,
    )

    # Match the ordinary FA tests: the operator consumes a real block_table
    # tensor.  Identity is the default, while reverse_pages flips page IDs
    # inside each batch's page-table row to validate non-identity mappings.
    block_table = torch.zeros(batch, max_pages_per_batch, dtype=torch.int32)
    for b in range(batch):
        n_valid = math.ceil(seqused_kv_list[b] / block_size)
        base = b * max_pages_per_batch
        page_ids = list(range(base, base + max_pages_per_batch))
        if reverse_pages:
            page_ids.reverse()
        for p in range(n_valid):
            block_table[b, p] = page_ids[p]

    for b in range(batch):
        s2_b = seqused_kv_list[b]
        n_valid_pages = math.ceil(s2_b / block_size)
        for p in range(n_valid_pages):
            phys = block_table[b, p].item()
            page_start = p * block_size
            page_end = min(page_start + block_size, s2_b)
            copy_len = page_end - page_start
            k_paged[phys, :, :copy_len, :] = k_data[b, :, page_start:page_end, :]
            v_paged[phys, :, :copy_len, :] = v_data[b, :, page_start:page_end, :]
            if k_scale_flat is not None:
                sk_paged[phys, :, :copy_len, :, :] = k_scale_flat[
                    b, :, page_start:page_end, :
                ].reshape(kv_heads, copy_len, qk_scale_groups, 2)
            if v_scale_tiles is not None:
                scale_group_start = page_start // 64
                valid_scale_groups = math.ceil(copy_len / 64)
                sv_groups = v_scale_tiles[b].reshape(kv_heads, -1, head_dim, 2)
                sv_paged[phys, :, :valid_scale_groups, :, :] = sv_groups[
:, scale_group_start: scale_group_start + valid_scale_groups
                ]  # fmt: skip

    checks.check_paged_kv_contract(
        k_paged,
        v_paged,
        sk_paged,
        sv_paged,
        block_table,
        k_data,
        v_data,
        k_scale_flat,
        v_scale_tiles,
        seqused_kv_list,
        block_size,
    )
    if kv_cache_layout == "PA_BBND":
        # AscendC PA_BnBsND physical storage:
        # K/V [Bn, Bs, N, D], K scale [Bn, Bs, N, G_QK, 2],
        # V scale [Bn, Bs/64, N, D, 2].
        k_paged = k_paged.permute(0, 2, 1, 3).contiguous()
        v_paged = v_paged.permute(0, 2, 1, 3).contiguous()
        sk_paged = sk_paged.permute(0, 2, 1, 3, 4).contiguous()
        sv_paged = sv_paged.permute(0, 2, 1, 3, 4).contiguous()
    elif kv_cache_layout == "PA_NZ":
        # AscendC MXFP8 PA_NZ physical storage:
        # K/V [Bn, N, D/32, Bs, 32],
        # K scale [Bn, N, Bs/16, D/64, 16, 2],
        # V scale [Bn, N, D/16, Bs/64, 16, 2].
        k_paged = (
            k_paged.reshape(num_pages, kv_heads, block_size, d_blocks32, 32)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        v_paged = (
            v_paged.reshape(num_pages, kv_heads, block_size, d_blocks32, 32)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        # Canonicalize scale strides even when a packed axis has size one.
        sk_paged = (
            sk_paged.reshape(num_pages, kv_heads, block_size // 16, 16, mx_groups64, 2)
            .permute(0, 1, 2, 4, 3, 5)
            .clone(memory_format=torch.contiguous_format)
        )
        sv_paged = (
            sv_paged.reshape(
                num_pages, kv_heads, block_size // 64, vscale_d_blocks16, 16, 2
            )
            .permute(0, 1, 3, 2, 4, 5)
            .clone(memory_format=torch.contiguous_format)
        )

    seqused_kv_npu = torch.tensor(seqused_kv_list, dtype=torch.int64)
    return k_paged, v_paged, sk_paged, sv_paged, block_table, seqused_kv_npu


def build_nonpaged_kv(
    k_data,
    v_data,
    k_scale_flat,
    v_scale_tiles,
    seqused_kv_list,
    kv_heads,
    head_dim,
    qk_scale_groups,
):
    """Build genuinely compact TND K/V without host-side tile padding.

    The last non-aligned tile is represented by ``tile_view``'s actual shape;
    ND/DN-to-NZ copies, MMAD and FIXPIPE must preserve that shape through the
    kernel instead of relying on synthetic GM rows.
    """
    k_tnd = torch.cat(
        [
            k_data[b, :, :length, :].permute(1, 0, 2)
            for b, length in enumerate(seqused_kv_list)
        ],
        dim=0,
    ).contiguous()
    v_tnd = torch.cat(
        [
            v_data[b, :, :length, :].permute(1, 0, 2)
            for b, length in enumerate(seqused_kv_list)
        ],
        dim=0,
    ).contiguous()

    if k_scale_flat is not None:
        sk_tnd = torch.cat(
            [
                k_scale_flat[b, :, :length, :]
                .reshape(kv_heads, length, qk_scale_groups, 2)
                .permute(1, 0, 2, 3)
                for b, length in enumerate(seqused_kv_list)
            ],
            dim=0,
        ).contiguous()
    else:
        sk_tnd = torch.full(
            (sum(seqused_kv_list), kv_heads, qk_scale_groups, 2),
            1.0,
            dtype=torch.float8_e8m0fnu,
        )

    if v_scale_tiles is not None:
        sv_tnd = torch.cat(
            [
                v_scale_tiles[b]
                .reshape(kv_heads, -1, head_dim, 2)[:, : math.ceil(length / 64)]
                .permute(1, 0, 2, 3)
                for b, length in enumerate(seqused_kv_list)
            ],
            dim=0,
        ).contiguous()
    else:
        total_scale_groups = sum(math.ceil(length / 64) for length in seqused_kv_list)
        sv_tnd = torch.full(
            (total_scale_groups, kv_heads, head_dim, 2), 1.0, dtype=torch.float8_e8m0fnu
        )

    seqused_kv_npu = torch.tensor(seqused_kv_list, dtype=torch.int64)
    return k_tnd, v_tnd, sk_tnd, sv_tnd, seqused_kv_npu


# ---------------------------------------------------------------------------
# Attention reference and comparison helpers
# ---------------------------------------------------------------------------


def build_attn_mask(
    seqused_q_list, seqused_kv_list, batch, query_capacity, mask_mode, *, tile_n
):
    """Build per-batch 3D attention mask [B, S1, S2_padded].

    0 = valid, -1e30 = masked (causal upper-triangle and/or key padding).
    """
    # The kernel consumes the score matrix in headDim-derived S2 chunks.
    # Pad the mask to that compute width so the synthetic tail columns are
    # explicitly masked instead of being copied as zero-valued mask entries.
    max_kv_tokens = max(seqused_kv_list)
    padded_kv_tokens = math.ceil(max_kv_tokens / tile_n) * tile_n
    mask = torch.zeros(batch, query_capacity, padded_kv_tokens, dtype=torch.float32)
    for b in range(batch):
        s1_b = seqused_q_list[b]
        s2_b = seqused_kv_list[b]
        mask[b, :, s2_b:] = -1e30
        mask[b, s1_b:, :] = -1e30
        if mask_mode == 3:
            # Mode 3 follows Ascend C RIGHT_DOWN_CAUSAL. Query row q may
            # attend through key q + (S_kv - S_q), rather than using the
            # left-up triangular boundary k <= q.
            q_positions = torch.arange(s1_b).unsqueeze(1)
            k_positions = torch.arange(s2_b).unsqueeze(0)
            causal = k_positions > q_positions + (s2_b - s1_b)
            mask[b, :s1_b, :s2_b][causal] = -1e30
    return mask


def online_softmax_fp8_golden(
    attention_scores,
    p_scale_fp32,
    *,
    tile_m,
    tile_n,
):
    """Run the MXFP8 online-softmax reference, including FP8 P rounding.

    Round each tile maximum upward to a multiple of ln(2). Differences between
    these maxima then produce power-of-two rescale factors representable by
    E8M0. This maximum frame affects P rounding and must precede the FP8 cast.
    The denominator accumulates exponentials before that cast.
    """
    batch, query_heads, query_tokens, kv_tokens = attention_scores.shape
    m_tiles = (query_tokens + tile_m - 1) // tile_m
    n_tiles = (kv_tokens + tile_n - 1) // tile_n
    probability_fp8 = torch.zeros_like(attention_scores)
    rescale_factors = torch.zeros(batch, query_heads, m_tiles, n_tiles, tile_m)
    final_row_sums = torch.zeros(batch, query_heads, query_tokens)

    for m_index, m_start in enumerate(range(0, query_tokens, tile_m)):
        row_count = min(tile_m, query_tokens - m_start)
        for batch_index in range(batch):
            for head_index in range(query_heads):
                running_max = torch.full((row_count,), float("-inf"))
                running_sum = torch.zeros(row_count)
                score_rows = attention_scores[
                    batch_index,
                    head_index,
                    m_start: m_start + row_count,
                    :,
                ]  # fmt: skip
                for n_index, n_start in enumerate(range(0, kv_tokens, tile_n)):
                    n_end = min(n_start + tile_n, kv_tokens)
                    score_tile = score_rows[:, n_start:n_end]
                    tile_max = score_tile.max(dim=-1).values
                    tile_max = torch.where(
                        torch.isfinite(tile_max),
                        torch.ceil(tile_max / math.log(2.0)) * math.log(2.0),
                        tile_max,
                    )
                    new_max = torch.maximum(running_max, tile_max)
                    valid_new_max = torch.isfinite(new_max)
                    safe_new_max = torch.where(
                        valid_new_max, new_max, torch.zeros_like(new_max)
                    )
                    if n_index == 0:
                        rescale = valid_new_max.float()
                    else:
                        valid_rescale = torch.isfinite(running_max) & valid_new_max
                        rescale = torch.where(
                            valid_rescale,
                            torch.exp(running_max - safe_new_max),
                            torch.zeros_like(running_max),
                        )
                    exp_values = torch.where(
                        torch.isfinite(score_tile),
                        torch.exp(score_tile - safe_new_max.unsqueeze(-1)),
                        torch.zeros_like(score_tile),
                    )
                    scaled_exp = exp_values * p_scale_fp32
                    probability_fp8[
                        batch_index,
                        head_index,
                        m_start: m_start + row_count,
                        n_start:n_end,
                    ] = scaled_exp.to(torch.float8_e4m3fn).float()  # fmt: skip
                    tile_sum = exp_values.sum(dim=-1) * p_scale_fp32
                    if n_index == 0:
                        running_sum = tile_sum
                    else:
                        running_sum = running_sum * rescale + tile_sum
                    rescale_factors[
                        batch_index, head_index, m_index, n_index, :row_count
                    ] = rescale
                    running_max = new_max
                final_row_sums[
                    batch_index,
                    head_index,
                    m_start: m_start + row_count,
                ] = running_sum  # fmt: skip
    return probability_fp8, rescale_factors, final_row_sums


def fp8_pv_golden(probability_fp8, value, *, tile_m, tile_n):
    """Compute one FP8 probability-by-value result for each KV tile."""
    batch, query_heads, query_tokens, kv_tokens = probability_fp8.shape
    head_dim = value.shape[-1]
    m_tiles = (query_tokens + tile_m - 1) // tile_m
    n_tiles = (kv_tokens + tile_n - 1) // tile_n
    pv_tiles = torch.zeros(batch, query_heads, m_tiles, n_tiles, tile_m, head_dim)
    for m_index, m_start in enumerate(range(0, query_tokens, tile_m)):
        row_count = min(tile_m, query_tokens - m_start)
        for n_index, n_start in enumerate(range(0, kv_tokens, tile_n)):
            n_end = min(n_start + tile_n, kv_tokens)
            pv_tiles[:, :, m_index, n_index, :row_count, :] = torch.matmul(
                probability_fp8[:, :, m_start: m_start + row_count, n_start:n_end],
                value[:, :, n_start:n_end, :],
            )  # fmt: skip
    return pv_tiles


def finalize_online_softmax_output_golden(
    pv_tiles,
    rescale_factors,
    final_row_sums,
    *,
    tile_m,
):
    """Rescale tiled PV partials and normalize by the final online row sum."""
    batch, query_heads, m_tiles, n_tiles, _, head_dim = pv_tiles.shape
    query_tokens = final_row_sums.shape[-1]
    output = torch.zeros(batch, query_heads, query_tokens, head_dim)
    for m_index in range(m_tiles):
        m_start = m_index * tile_m
        row_count = min(tile_m, query_tokens - m_start)
        running_output = pv_tiles[:, :, m_index, 0, :row_count, :].clone()
        for n_index in range(1, n_tiles):
            rescale = rescale_factors[:, :, m_index, n_index, :row_count].unsqueeze(-1)
            running_output = (
                running_output * rescale
                + pv_tiles[:, :, m_index, n_index, :row_count, :]
            )
        row_sums = final_row_sums[:, :, m_start: m_start + row_count].unsqueeze(-1)  # fmt: skip
        output[:, :, m_start: m_start + row_count, :] = torch.where(
            row_sums > 0,
            running_output / row_sums.clamp_min(torch.finfo(row_sums.dtype).tiny),
            torch.zeros_like(running_output),
        )  # fmt: skip
    return output.bfloat16()


def attention_error_metrics(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, float]:
    """Return diagnostics and the retained BF16 pass-percentage metrics."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Attention shape mismatch: actual={tuple(actual.shape)}, "
            f"expected={tuple(expected.shape)}"
        )
    if not torch.isfinite(expected).all():
        raise AssertionError("Attention reference contains non-finite values")
    if actual.numel() == 0:
        return {
            "nonfinite": 0.0,
            "mean_abs": 0.0,
            "p99_abs": 0.0,
            "max_abs": 0.0,
            "rms": 0.0,
            "max_ref": 0.0,
            "pass_percent": 100.0,
            "max_relative_error": 0.0,
        }
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    abs_error = (actual_fp32 - expected_fp32).abs()
    finite = torch.isfinite(actual_fp32)
    rms = expected_fp32.square().mean().sqrt().item()
    # Match ops-transformer's qfa_mxfp8_test/result_compare_method.py.
    close = torch.isclose(
        actual_fp32,
        expected_fp32,
        rtol=OUTPUT_RTOL,
        atol=OUTPUT_ATOL,
        equal_nan=True,
    )
    error = ~close
    error_count = int(error.sum().item())
    pass_percent = 100.0 * (actual.numel() - error_count) / actual.numel()
    denominator_floor = RELATIVE_FLOOR
    denominator = (
        torch.maximum(
            torch.maximum(actual_fp32.abs(), expected_fp32.abs()),
            torch.tensor(denominator_floor, dtype=torch.float32),
        )
        + RELATIVE_EPSILON
    )
    relative_error = abs_error / denominator
    max_relative_error = relative_error[error].max().item() if error_count else 0.0
    # ``torch.quantile`` rejects tensors above its internal indexing limit.
    # P99 is diagnostic only (the AscendC pass percentage below is the actual
    # precision gate), so use the exact nearest-rank order statistic instead.
    flat_abs_error = abs_error.flatten()
    p99_rank = max(1, math.ceil(DIAGNOSTIC_QUANTILE * flat_abs_error.numel()))
    p99_abs = flat_abs_error.kthvalue(p99_rank).values.item()
    return {
        "nonfinite": float((~finite).sum().item()),
        "mean_abs": abs_error.mean().item(),
        "p99_abs": p99_abs,
        "max_abs": abs_error.max().item(),
        "rms": rms,
        "max_ref": expected_fp32.abs().max().item(),
        "pass_percent": pass_percent,
        "max_relative_error": max_relative_error,
    }


def compare_attention_results(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    case_name: str,
) -> None:
    """Compare NPU output with the BF16 CPU golden using the test tolerance."""
    assert actual.dtype == torch.bfloat16, (
        f"{case_name}: expected BF16 output, got {actual.dtype}"
    )
    assert expected.dtype == torch.bfloat16, (
        f"{case_name}: expected BF16 reference, got {expected.dtype}"
    )
    metrics = attention_error_metrics(actual, expected)
    LOGGER.info(
        f"[{case_name}] mean_abs={metrics['mean_abs']:.6f}, "
        f"p99_abs={metrics['p99_abs']:.6f}, max_abs={metrics['max_abs']:.6f}, "
        f"rms={metrics['rms']:.6f}, nonfinite={int(metrics['nonfinite'])}, "
        f"pass={metrics['pass_percent']:.6f}%, "
        f"max_re={metrics['max_relative_error']:.6f}"
    )
    assert metrics["nonfinite"] == 0, f"{case_name}: output contains non-finite values"
    assert metrics["pass_percent"] >= PASS_PERCENT_LIMIT, (
        f"{case_name}: BF16 pass percentage below {PASS_PERCENT_LIMIT}%: "
        f"pass={metrics['pass_percent']:.6f}%, "
        f"max_re={metrics['max_relative_error']:.6f}"
    )
    assert metrics["max_relative_error"] < MAX_RELATIVE_ERROR_LIMIT, (
        f"{case_name}: BF16 max relative error must be < {MAX_RELATIVE_ERROR_LIMIT}: "
        f"pass={metrics['pass_percent']:.6f}%, "
        f"max_re={metrics['max_relative_error']:.6f}"
    )


# ---------------------------------------------------------------------------
# Test data, golden, and NPU execution
# ---------------------------------------------------------------------------


def _make_quantized_operands(case, query_capacity, kv_capacity, storage_tile_n):
    """Create logical CPU operands using the original Q/K/V random streams."""
    batch, query_heads, kv_heads = case["B"], case["N_q"], case["N_kv"]
    head_dim = case["D"]
    max_seqlen_kv = case["max_seqlen_kv"]
    query_lengths = case["S_q"]
    # Reproduce ops-transformer's MXFP8 functional-test inputs exactly: Q/K/V
    # Use independent fixed seeds and BF16 uniform samples in [-1, 1] as the
    # pre-quantization source.  Q/K/V are subsequently quantized to MXFP8 E4M3
    # with E8M0 block scales; the kernel output and comparison remain BF16.
    query_source_bf16 = (
        torch.rand(
            batch,
            query_heads,
            query_capacity,
            head_dim,
            dtype=torch.bfloat16,
            generator=torch.Generator().manual_seed(54),
        )
        * 2
        - 1
    )
    key_valid_bf16 = (
        torch.rand(
            batch,
            kv_heads,
            max_seqlen_kv,
            head_dim,
            dtype=torch.bfloat16,
            generator=torch.Generator().manual_seed(3),
        )
        * 2
        - 1
    )
    value_valid_bf16 = (
        torch.rand(
            batch,
            kv_heads,
            max_seqlen_kv,
            head_dim,
            dtype=torch.bfloat16,
            generator=torch.Generator().manual_seed(4),
        )
        * 2
        - 1
    )
    key_source_bf16 = torch.zeros(
        batch, kv_heads, kv_capacity, head_dim, dtype=key_valid_bf16.dtype
    )
    value_source_bf16 = torch.zeros_like(key_source_bf16)
    key_source_bf16[:, :, :max_seqlen_kv, :] = key_valid_bf16
    value_source_bf16[:, :, :max_seqlen_kv, :] = value_valid_bf16

    # Quantize each valid prefix independently and leave padded storage as
    # zero data with unit E8M0 scale.  Quantizing an all-zero padded region
    # would produce exponent byte 0 instead of the ABI's neutral 0x7f.
    cpu_query_bns_fp8 = torch.zeros_like(query_source_bf16, dtype=torch.float8_e4m3fn)
    cpu_query_scale_bns_e8m0 = torch.full(
        (
            batch,
            query_heads,
            query_capacity,
            _mx_scale_cols(head_dim),
        ),
        1.0,
        dtype=torch.float8_e8m0fnu,
    )
    for batch_index, valid_query_length in enumerate(query_lengths):
        query_valid_bf16 = query_source_bf16[
            batch_index,
            :,
            :valid_query_length,
            :,
        ]
        query_valid_fp8, query_valid_scale_e8m0 = quantize_mxfp8_e4m3(query_valid_bf16)
        cpu_query_bns_fp8[
            batch_index,
            :,
            :valid_query_length,
            :,
        ] = query_valid_fp8
        cpu_query_scale_bns_e8m0[
            batch_index,
            :,
            :valid_query_length,
            :,
        ] = query_valid_scale_e8m0
    cpu_key_bns_fp8, cpu_key_scale_bns_e8m0 = quantize_mxfp8_e4m3(key_source_bf16)
    cpu_value_bns_fp8, cpu_value_scale_tiles_e8m0 = quantize_v_mxfp8(
        value_source_bf16,
        tile_n=storage_tile_n,
    )
    return (
        cpu_query_bns_fp8,
        cpu_key_bns_fp8,
        cpu_value_bns_fp8,
        cpu_query_scale_bns_e8m0,
        cpu_key_scale_bns_e8m0,
        cpu_value_scale_tiles_e8m0,
    )


def _pack_query(case, query_capacity, cpu_query_bns_fp8, cpu_query_scale_bns_e8m0):
    """Pack compact Prefill Q or padded Decode Q and its paired E8M0 scales."""
    batch, query_heads, kv_heads = case["B"], case["N_q"], case["N_kv"]
    query_lengths = case["S_q"]
    q_scale_layout = case["q_scale_layout"]
    qk_scale_groups = _mx_groups64(case["D"])
    query_fp8 = cpu_query_bns_fp8
    # Prefill Q is compact TND. Decode follows the AscendC Q-pad contract and
    # reserves one physical headDim-derived S1 slot per batch; seqused_q identifies
    # the valid prefix in each slot.
    query_storage_fp8 = torch.cat(
        [
            query_fp8[
                b,
                :,
                : (query_capacity if q_scale_layout == "N2TGD" else length),
                :,
            ].permute(1, 0, 2)
            for b, length in enumerate(query_lengths)
        ],
        dim=0,
    ).contiguous()

    query_scale_tnd_e8m0 = torch.cat(
        [
            cpu_query_scale_bns_e8m0[
                b,
                :,
                : (query_capacity if q_scale_layout == "N2TGD" else length),
                :,
            ]
            .reshape(
                query_heads,
                query_capacity if q_scale_layout == "N2TGD" else length,
                qk_scale_groups,
                2,
            )
            .permute(1, 0, 2, 3)
            for b, length in enumerate(query_lengths)
        ],
        dim=0,
    ).contiguous()
    if q_scale_layout == "TND":
        query_scale_storage_e8m0 = query_scale_tnd_e8m0
    else:
        head_group_num = query_heads // kv_heads
        query_scale_storage_e8m0 = (
            query_scale_tnd_e8m0.reshape(
                batch * query_capacity,
                kv_heads,
                head_group_num,
                qk_scale_groups,
                2,
            )
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )

    return query_storage_fp8, query_scale_storage_e8m0


def generate_attention_test_data(
    case: CaseConfig,
    *,
    nonpaged_tile_n: int | None = None,
) -> AttentionTestData:
    """Generate deterministic CPU inputs, scales, layouts, and golden operands."""
    checks.check_test_case(case)

    batch = case["B"]
    kv_heads = case["N_kv"]
    query_lengths = list(case["S_q"])
    kv_lengths = list(case["S_kv"])
    head_dim = case["D"]
    max_seqlen_q = case["max_seqlen_q"]
    max_seqlen_kv = case["max_seqlen_kv"]
    enable_pa = case["enable_pa"]
    kv_cache_layout = case["kv_cache_layout"]
    q_scale_layout = case["q_scale_layout"]
    block_size = case["block_size"]
    mask_mode = case["mask_mode"]
    p_scale = case["p_scale"]
    tile_config = _tile_config_for_head_dim(head_dim)
    if nonpaged_tile_n is None:
        nonpaged_tile_n = tile_config.kv_tokens

    query_length = max_seqlen_q
    # Reserve physical Q capacity for quantization and Decode packing; Prefill
    # packing below still concatenates only each batch's valid Q prefix.
    if q_scale_layout == "TND" and len(set(query_lengths)) > 1:
        query_capacity = (
            math.ceil(query_length / tile_config.query_rows) * tile_config.query_rows
        )
    elif q_scale_layout == "TND":
        query_capacity = query_length
    else:
        query_capacity = tile_config.query_rows
    storage_tile_n = block_size if enable_pa else nonpaged_tile_n
    kv_capacity = math.ceil(max_seqlen_kv / storage_tile_n) * storage_tile_n
    qk_scale_groups = _mx_groups64(head_dim)

    (
        cpu_query_bns_fp8,
        cpu_key_bns_fp8,
        cpu_value_bns_fp8,
        cpu_query_scale_bns_e8m0,
        cpu_key_scale_bns_e8m0,
        cpu_value_scale_tiles_e8m0,
    ) = _make_quantized_operands(case, query_capacity, kv_capacity, storage_tile_n)
    query_storage_fp8, query_scale_storage_e8m0 = _pack_query(
        case,
        query_capacity,
        cpu_query_bns_fp8,
        cpu_query_scale_bns_e8m0,
    )
    key_fp8 = cpu_key_bns_fp8
    value_fp8 = cpu_value_bns_fp8
    if enable_pa:
        key, value, key_scale_e8m0, value_scale_e8m0, block_table, kv_lengths_tensor = (
            build_paged_kv(
                key_fp8,
                value_fp8,
                cpu_key_scale_bns_e8m0,
                cpu_value_scale_tiles_e8m0,
                kv_lengths,
                batch,
                kv_heads,
                head_dim,
                qk_scale_groups,
                block_size,
                kv_cache_layout,
                bool(case.get("reverse_pages", False)),
            )
        )
        if case.get("noncontiguous_pa", False):
            key, value, key_scale_e8m0, value_scale_e8m0 = (
                _apply_noncontiguous_pa_stride(
                    key,
                    value,
                    key_scale_e8m0,
                    value_scale_e8m0,
                    checks.pa_stride_for_case(case),
                )
            )
    else:
        key, value, key_scale_e8m0, value_scale_e8m0, kv_lengths_tensor = (
            build_nonpaged_kv(
                key_fp8,
                value_fp8,
                cpu_key_scale_bns_e8m0,
                cpu_value_scale_tiles_e8m0,
                kv_lengths,
                kv_heads,
                head_dim,
                qk_scale_groups,
            )
        )
        block_table = torch.zeros(1, dtype=torch.int32)

    attention_mask = build_attn_mask(
        query_lengths,
        kv_lengths,
        batch,
        query_capacity,
        mask_mode,
        tile_n=tile_config.kv_tokens,
    )
    return AttentionTestData(
        npu_query_tnd_fp8=query_storage_fp8,
        npu_key_cache_fp8=key,
        npu_value_cache_fp8=value,
        npu_query_scale_e8m0=query_scale_storage_e8m0,
        npu_key_scale_cache_e8m0=key_scale_e8m0,
        npu_value_scale_cache_e8m0=value_scale_e8m0,
        p_scale_fp32=torch.tensor([p_scale], dtype=torch.float32),
        attention_mask_fp32=attention_mask,
        block_table_i32=block_table,
        kv_lengths_i64=kv_lengths_tensor,
        query_lengths_i64=torch.tensor(query_lengths, dtype=torch.int64),
        cpu_query_bns_fp8=cpu_query_bns_fp8,
        cpu_key_bns_fp8=cpu_key_bns_fp8,
        cpu_value_bns_fp8=cpu_value_bns_fp8,
        cpu_query_scale_bns_e8m0=cpu_query_scale_bns_e8m0,
        cpu_key_scale_bns_e8m0=cpu_key_scale_bns_e8m0,
        cpu_value_scale_tiles_e8m0=cpu_value_scale_tiles_e8m0,
        query_length_max=query_length,
        query_capacity_tokens=query_capacity,
        kv_capacity_tokens=kv_capacity,
        attention_scale_fp32=1.0 / math.sqrt(head_dim),
    )


def generate_cpu_golden(
    case: CaseConfig,
    test_data: AttentionTestData,
    *,
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute independent attention and FP32 LSE CPU references."""
    batch = case["B"]
    query_heads = case["N_q"]
    kv_heads = case["N_kv"]
    kv_lengths = list(case["S_kv"])
    head_dim = case["D"]
    mask_mode = case["mask_mode"]
    p_scale = case["p_scale"]
    tile_config = _tile_config_for_head_dim(head_dim)
    if tile_n is None:
        tile_n = tile_config.kv_tokens

    query_lengths = list(case["S_q"])

    query_golden_fp32 = torch.zeros_like(
        test_data.cpu_query_bns_fp8,
        dtype=torch.float32,
    )
    for batch_index, valid_query_length in enumerate(query_lengths):
        query_golden_fp32[
            batch_index,
            :,
            :valid_query_length,
            :,
        ] = dequantize_mxfp8_e4m3(
            test_data.cpu_query_bns_fp8[
                batch_index,
                :,
                :valid_query_length,
                :,
            ],
            test_data.cpu_query_scale_bns_e8m0[
                batch_index,
                :,
                :valid_query_length,
                :,
            ],
        )
    key_golden_fp32 = dequantize_mxfp8_e4m3(
        test_data.cpu_key_bns_fp8,
        test_data.cpu_key_scale_bns_e8m0,
    ).repeat_interleave(
        query_heads // kv_heads,
        dim=1,
    )
    value_golden_fp32 = dequantize_v_mxfp8(
        test_data.cpu_value_bns_fp8,
        test_data.cpu_value_scale_tiles_e8m0,
    ).repeat_interleave(
        query_heads // kv_heads,
        dim=1,
    )
    attention_scores = (
        torch.matmul(
            query_golden_fp32,
            key_golden_fp32.transpose(-2, -1),
        )
        * test_data.attention_scale_fp32
    )
    for batch_index, valid_kv_length in enumerate(kv_lengths):
        valid_q_length = query_lengths[batch_index]
        attention_scores[batch_index, :, :, valid_kv_length:] = float("-inf")
        if mask_mode == 3:
            q_positions = torch.arange(valid_q_length).unsqueeze(1)
            k_positions = torch.arange(valid_kv_length).unsqueeze(0)
            causal_mask = k_positions <= (
                q_positions + valid_kv_length - valid_q_length
            )
            attention_scores[
                batch_index, :, :valid_q_length, :valid_kv_length
            ].masked_fill_(~causal_mask, float("-inf"))

    golden_tile_m = test_data.query_capacity_tokens if tile_m is None else tile_m
    probability_fp8, rescale_factors, final_row_sums = online_softmax_fp8_golden(
        attention_scores,
        p_scale,
        tile_m=golden_tile_m,
        tile_n=tile_n,
    )
    pv_tiles = fp8_pv_golden(
        probability_fp8,
        value_golden_fp32,
        tile_m=golden_tile_m,
        tile_n=tile_n,
    )
    expected = finalize_online_softmax_output_golden(
        pv_tiles,
        rescale_factors,
        final_row_sums,
        tile_m=golden_tile_m,
    )
    expected_lse = torch.full(
        (
            batch,
            query_heads,
            test_data.query_capacity_tokens,
        ),
        float("inf"),
        dtype=torch.float32,
    )
    for batch_index, valid_query_length in enumerate(query_lengths):
        valid_kv_length = kv_lengths[batch_index]
        batch_lse = torch.logsumexp(
            attention_scores[
                batch_index,
                :,
                :valid_query_length,
                :valid_kv_length,
            ],
            dim=-1,
        )
        # Match the public Flash Attention convention for fully masked rows.
        batch_lse = torch.where(
            torch.isfinite(batch_lse),
            batch_lse,
            torch.full_like(batch_lse, float("inf")),
        )
        expected_lse[
            batch_index,
            :,
            :valid_query_length,
        ] = batch_lse
    return (
        expected[:, :, : test_data.query_length_max, :],
        expected_lse[:, :, : test_data.query_length_max],
    )


def execute_npu_attention(
    case: CaseConfig,
    test_data: AttentionTestData,
    *,
    caller_device=None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Execute one kernel and return CPU attention and optional FP32 LSE."""
    batch = case["B"]
    query_heads = case["N_q"]
    head_dim = case["D"]
    mask_mode = case["mask_mode"]
    enable_pa = case["enable_pa"]
    enable_lse = case["enable_lse"]
    block_size = case["block_size"]
    q_layout = case["q_layout"]
    kv_cache_layout = case["kv_cache_layout"]

    query_npu = test_data.npu_query_tnd_fp8.npu()
    key_host = test_data.npu_key_cache_fp8
    value_host = test_data.npu_value_cache_fp8
    key_scale_host = test_data.npu_key_scale_cache_e8m0
    value_scale_host = test_data.npu_value_scale_cache_e8m0
    stride_mode = checks.pa_stride_for_case(case)
    key_npu = _to_npu_preserve_pa_stride(key_host, stride_mode, 0x38)
    value_npu = _to_npu_preserve_pa_stride(value_host, stride_mode, -0x48)
    query_scale_npu = test_data.npu_query_scale_e8m0.npu()
    key_scale_npu = _to_npu_preserve_pa_stride(key_scale_host, stride_mode, -128)
    value_scale_npu = _to_npu_preserve_pa_stride(value_scale_host, stride_mode, 0)
    if enable_pa:
        checks.check_pa_device_strides(
            (key_host, value_host, key_scale_host, value_scale_host),
            (key_npu, value_npu, key_scale_npu, value_scale_npu),
            stride_mode,
        )
    p_scale_npu = test_data.p_scale_fp32.npu()
    attention_mask_npu = test_data.attention_mask_fp32.npu()
    block_table_npu = test_data.block_table_i32.npu()
    kv_lengths_npu = test_data.kv_lengths_i64.npu()
    query_lengths_npu = test_data.query_lengths_i64.npu()
    torch.npu.synchronize()
    implementation = load_sample(
        "attention/flash_attn_mxfp8_fullquant/flash_attn_mxfp8_fullquant.py"
    )
    caller_device = query_npu.device if caller_device is None else caller_device
    with torch.npu.device(caller_device):
        caller_stream = torch.npu.current_stream(caller_device)
        output_npu, softmax_lse_npu = implementation.npu_flash_atten_mxfp8_fullquant(
            query_npu,
            key_npu,
            value_npu,
            query_scale_npu,
            key_scale_npu,
            value_scale_npu,
            p_scale_npu,
            softmax_scale=test_data.attention_scale_fp32,
            attn_mask=attention_mask_npu,
            block_table=block_table_npu,
            seqused_kv=kv_lengths_npu,
            seqused_q=query_lengths_npu,
            mask_mode=mask_mode,
            paged=enable_pa,
            block_size=block_size,
            q_layout=q_layout,
            kv_layout=kv_cache_layout,
            return_softmax_lse=enable_lse,
            max_seqlen_q=case["max_seqlen_q"],
            output_token_count=sum(case["S_q"]),
        )
        assert torch.npu.current_device() == caller_device.index
        assert torch.npu.current_stream(caller_device) == caller_stream
    torch.npu.synchronize(query_npu.device)
    assert output_npu.device == query_npu.device
    if softmax_lse_npu is not None:
        assert softmax_lse_npu.device == query_npu.device
    # Check the public compact output before unpacking; slicing below would
    # otherwise hide surplus tokens or an incorrect padded LSE allocation.
    output_tokens = sum(case["S_q"])
    assert tuple(output_npu.shape) == (output_tokens, query_heads, head_dim), (
        f"{case['name']}: invalid compact output shape {tuple(output_npu.shape)}"
    )
    assert output_npu.dtype == torch.bfloat16
    if enable_lse:
        assert softmax_lse_npu is not None
        assert tuple(softmax_lse_npu.shape) == (query_heads, output_tokens)
        assert softmax_lse_npu.dtype == torch.float32
    else:
        assert softmax_lse_npu is None
    output_tnd = output_npu.cpu()
    output_bns = torch.zeros(
        batch,
        query_heads,
        test_data.query_capacity_tokens,
        head_dim,
        dtype=torch.bfloat16,
    )
    softmax_lse_nt = softmax_lse_npu.cpu() if softmax_lse_npu is not None else None
    softmax_lse_bns = (
        torch.full(
            (batch, query_heads, test_data.query_capacity_tokens),
            float("inf"),
            dtype=torch.float32,
        )
        if enable_lse
        else None
    )
    token_start = 0
    for batch_index, query_length in enumerate(case["S_q"]):
        token_end = token_start + query_length
        output_bns[batch_index, :, :query_length, :] = output_tnd[
            token_start:token_end
        ].permute(1, 0, 2)
        if softmax_lse_bns is not None:
            softmax_lse_bns[batch_index, :, :query_length] = softmax_lse_nt[
                :, token_start:token_end
            ]
        token_start += query_length
    return (
        output_bns[:, :, : test_data.query_length_max, :],
        (
            softmax_lse_bns[:, :, : test_data.query_length_max]
            if softmax_lse_bns is not None
            else None
        ),
    )


def _run_case_in_worker(params: CaseConfig) -> None:
    """Run an exact pytest case so repository loading and fixtures also apply."""
    env = os.environ.copy()
    env["GQA_MXFP8_CASE_WORKER"] = "1"
    node_id = f"{os.path.abspath(__file__)}::test_gqa_mxfp8_precision[{params['name']}]"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", node_id, "-q"],
        env=env,
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        message = "\n".join(
            (
                "isolated NPU worker failed",
                "stdout:",
                result.stdout,
                "stderr:",
                result.stderr,
            )
        )
        pytest.fail(message)


def _run_precision_case(params: CaseConfig, *, caller_device=None) -> None:
    """Validate one GQA MXFP8 case through the common four-stage interface."""
    # A skipped child exits successfully; skip the parent before dispatch too.
    pytest.importorskip("torch_npu")
    # Reuse imports, the device context and JIT caches during normal batch runs.
    # Set GQA_MXFP8_ISOLATE_CASES=1 to debug each case in a fresh process.
    if (
        caller_device is None
        and os.environ.get("GQA_MXFP8_ISOLATE_CASES") == "1"
        and os.environ.get("GQA_MXFP8_CASE_WORKER") != "1"
    ):
        _run_case_in_worker(params)
        return
    test_data = generate_attention_test_data(params)
    expected, expected_lse = generate_cpu_golden(params, test_data)
    actual, actual_lse = execute_npu_attention(
        params, test_data, caller_device=caller_device
    )
    # Compare only the valid variable-Q prefix for each batch.
    for batch_index, valid_query_length in enumerate(params["S_q"]):
        compare_attention_results(
            actual[
                batch_index: batch_index + 1,
                :,
                :valid_query_length,
                :,
            ],
            expected[
                batch_index: batch_index + 1,
                :,
                :valid_query_length,
                :,
            ],
            case_name=f"{params['name']}/batch-{batch_index}",
        )  # fmt: skip
        if params["enable_lse"]:
            assert actual_lse is not None
            actual_batch_lse = actual_lse[
                batch_index,
                :,
                :valid_query_length,
            ]
            expected_batch_lse = expected_lse[
                batch_index,
                :,
                :valid_query_length,
            ]
            assert torch.equal(
                torch.isposinf(actual_batch_lse),
                torch.isposinf(expected_batch_lse),
            ), f"{params['name']}/batch-{batch_index}: LSE +inf mask mismatch"
            finite = torch.isfinite(expected_batch_lse)
            torch.testing.assert_close(
                actual_batch_lse[finite],
                expected_batch_lse[finite],
                atol=LSE_ATOL,
                rtol=LSE_RTOL,
                msg=(f"{params['name']}/batch-{batch_index}: LSE precision mismatch"),
            )


# ---------------------------------------------------------------------------
# Pytest setup
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _limit_cpu_threads():
    """Avoid thread-pool overhead for the small CPU reference tensors."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)


# ---------------------------------------------------------------------------
# NPU precision tests
# ---------------------------------------------------------------------------


def test_mxfp8_known_block_scales():
    """Two known blocks exercise E8M0 bias and the 32-element quantization axis."""
    values = torch.cat((torch.ones(32), torch.full((32,), FP8_E4M3_MAX)))
    quantized, scales = quantize_mxfp8_e4m3(values)
    # max=1 needs scale=2**-8; max=448 needs scale=1. E8M0 has bias 127.
    assert scales.dtype == torch.float8_e8m0fnu
    assert scales.view(torch.uint8).tolist() == [119, 127]
    torch.testing.assert_close(dequantize_mxfp8_e4m3(quantized, scales), values)


@pytest.mark.parametrize("p_scale", [1.0, 15.0, 256.0])
def test_online_reference_masked_rows(p_scale):
    """Masked rows yield zero and valid rows normalize across two KV subtiles."""
    scores = torch.full((1, 1, 2, 65), float("-inf"))
    scores[0, 0, 1, (0, 64)] = 0.0
    probabilities, rescale, sums = online_softmax_fp8_golden(
        scores, p_scale, tile_m=2, tile_n=64
    )
    partials = fp8_pv_golden(
        probabilities, torch.ones(1, 1, 65, 1), tile_m=2, tile_n=64
    )
    output = finalize_online_softmax_output_golden(partials, rescale, sums, tile_m=2)
    torch.testing.assert_close(
        output, torch.tensor([[[[0.0], [1.0]]]], dtype=torch.bfloat16)
    )
    torch.testing.assert_close(sums, torch.tensor([[[0.0, 2 * p_scale]]]))


@pytest.mark.parametrize(
    "invalid_reference, message",
    [
        (torch.ones(1, dtype=torch.bfloat16), "shape mismatch"),
        (torch.ones(2, dtype=torch.float32), "BF16 reference"),
        (torch.full((2,), float("nan"), dtype=torch.bfloat16), "non-finite"),
    ],
    ids=["broadcast_shape", "reference_dtype", "reference_nan"],
)
def test_comparison_rejects_invalid_reference(invalid_reference, message):
    """Broadcasting, dtype coercion and matching NaNs cannot hide bad references."""
    with pytest.raises(AssertionError, match=message):
        compare_attention_results(
            torch.ones(2, dtype=torch.bfloat16),
            invalid_reference,
            case_name="invalid_reference",
        )


def _scale_contract_call():
    """Build CPU inputs for public dtype validation without launching NPU work."""
    case = dict(zip(_CASE_FIELDS, _CASES[0].values))
    data = generate_attention_test_data(case)
    args = [
        data.npu_query_tnd_fp8,
        data.npu_key_cache_fp8,
        data.npu_value_cache_fp8,
        data.npu_query_scale_e8m0,
        data.npu_key_scale_cache_e8m0,
        data.npu_value_scale_cache_e8m0,
        data.p_scale_fp32,
    ]
    kwargs = dict(
        softmax_scale=data.attention_scale_fp32,
        attn_mask=data.attention_mask_fp32,
        block_table=data.block_table_i32,
        seqused_kv=data.kv_lengths_i64,
        seqused_q=data.query_lengths_i64,
        mask_mode=case["mask_mode"],
        paged=case["enable_pa"],
        block_size=case["block_size"],
        q_layout=case["q_layout"],
        kv_layout=case["kv_cache_layout"],
        return_softmax_lse=True,
        max_seqlen_q=0,
        output_token_count=sum(case["S_q"]),
    )
    return args, kwargs


def test_native_e8m0_scale_contract():
    """Native E8M0 scales and FP32 p_scale pass the public empty-bound path."""
    implementation = load_sample(
        "attention/flash_attn_mxfp8_fullquant/flash_attn_mxfp8_fullquant.py"
    )
    args, kwargs = _scale_contract_call()
    assert all(x.dtype == torch.float8_e8m0fnu for x in args[3:6])
    assert args[6].dtype == torch.float32
    output, lse = implementation.npu_flash_atten_mxfp8_fullquant(*args, **kwargs)
    assert output.dtype == torch.bfloat16 and lse.dtype == torch.float32
    assert torch.count_nonzero(output) == 0
    assert torch.isposinf(lse).all()


@pytest.mark.parametrize(
    "arg_index,name",
    [
        (0, "query"),
        (1, "key"),
        (2, "value"),
        (3, "q_scale"),
        (4, "k_scale"),
        (5, "v_scale"),
        (6, "p_scale"),
    ],
)
def test_input_contract_rejects_wrong_dtype(arg_index, name):
    """Reject int8 data and scales, and require FP32 p_scale."""
    implementation = load_sample(
        "attention/flash_attn_mxfp8_fullquant/flash_attn_mxfp8_fullquant.py"
    )
    args, kwargs = _scale_contract_call()
    args[arg_index] = (
        args[arg_index].view(torch.int8) if arg_index < 6 else args[arg_index].half()
    )
    with pytest.raises(TypeError, match=f"Argument {name} must have dtype"):
        implementation.npu_flash_atten_mxfp8_fullquant(*args, **kwargs)


@pytest.mark.parametrize(
    "case_values", [pytest.param(case.values, id=case.id) for case in _CASES]
)
@pytest.mark.npu
@pytest.mark.slow
def test_gqa_mxfp8_precision(case_values) -> None:
    _run_precision_case(dict(zip(_CASE_FIELDS, case_values)))


@pytest.mark.npu
@pytest.mark.parametrize("input_index", [0, 1], ids=["same_device", "cross_device"])
def test_mxfp8_input_stream_limits(input_index, monkeypatch):
    pytest.importorskip("torch_npu")
    if torch.npu.device_count() <= input_index:
        pytest.skip(f"requires {input_index + 1} visible NPUs")
    if not hasattr(torch.npu, "set_stream_limit"):
        pytest.skip("requires NPU Stream quota control")

    multi_device = torch.npu.device_count() > 1
    caller_device = torch.device("npu:0")
    input_device = torch.device("npu", input_index)
    implementation = load_sample(
        "attention/flash_attn_mxfp8_fullquant/flash_attn_mxfp8_fullquant.py"
    )
    case = next(case for case in _CASES if case.id == "Prefill_PABBND_1_4_2_65_321_72")
    params = dict(zip(_CASE_FIELDS, case.values))
    with torch.npu.device(input_device):
        stream = torch.npu.Stream(device=input_device)
        platform = implementation.get_platform_info(force_refresh=True, stream=stream)
    available = min(platform.cube_core_num, platform.vector_core_num // 2)
    if available < 2:
        pytest.skip("requires two distinct valid AIC/AIV Stream quotas")

    get_platform = implementation.get_platform_info
    pad = implementation._pad_d72_operands
    prepare = implementation.prepare_metadata
    launch = implementation._launch_precompiled_attention

    def check_context():
        assert torch.npu.current_device() == input_index
        assert torch.npu.current_stream(input_device) == stream

    def checked_platform(*, force_refresh, stream):
        check_context()
        assert stream == torch.npu.current_stream(input_device)
        assert force_refresh is multi_device
        return get_platform(force_refresh=force_refresh, stream=stream)

    def checked_pad(*args):
        check_context()
        assert args[-2] == cores
        return pad(*args)

    def checked_prepare(q, kv, **kwargs):
        check_context()
        assert (kwargs["cores"], kwargs["vectors"]) == (cores, 2 * cores)
        return prepare(q, kv, **kwargs)

    def checked_launch(launcher, *args):
        check_context()
        assert launcher.block_dim == cores
        return launch(launcher, *args)

    monkeypatch.setattr(implementation, "get_platform_info", checked_platform)
    monkeypatch.setattr(implementation, "_pad_d72_operands", checked_pad)
    monkeypatch.setattr(implementation, "prepare_metadata", checked_prepare)
    monkeypatch.setattr(implementation, "_launch_precompiled_attention", checked_launch)

    for cores in (available, min(8, available // 2)):
        with torch.npu.device(input_device):
            torch.npu.set_stream_limit(stream, cube_num=cores, vector_num=2 * cores)
            try:
                with torch.npu.stream(stream):
                    _run_precision_case(params, caller_device=caller_device)
            finally:
                stream.synchronize()
                torch.npu.reset_stream_limit(stream)
