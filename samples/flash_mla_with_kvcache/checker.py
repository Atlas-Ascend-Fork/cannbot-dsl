# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host-side validation for Flash MLA with KV cache."""

from dataclasses import dataclass

import torch


HEAD_DIM_QK = 576
HEAD_DIM_V = 512
MASK_TEMPLATE_SIZE = 2048
KV_BLOCK_SIZE = 128
PA_NZ_D0 = 16
PA_NZ_D1 = HEAD_DIM_QK // PA_NZ_D0
MAX_BATCH_SIZE = 65536
UNSPECIFIED_SEQLEN = -1
NO_MASK_MODE = 0
CAUSAL_MASK_MODE = 3
SUPPORTED_Q_LAYOUTS = ("TND",)
SUPPORTED_KV_LAYOUTS = ("PA_NZ", "PA_BBND")
SUPPORTED_HEAD_COUNTS = (64, 96)
SUPPORTED_MASK_MODES = (NO_MASK_MODE, CAUSAL_MASK_MODE)
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


@dataclass(frozen=True, slots=True)
class CheckedInputs:
    """Normalized values and dimensions needed by the host launcher."""

    layout_q: str
    layout_kv: str
    layout_out: str
    batch: int
    total_q: int
    num_heads_q: int
    logical_rows: int
    block_size: int


@dataclass(frozen=True, slots=True)
class InputRelations:
    """Dimensions established by relationships between public inputs."""

    batch: int


@dataclass(frozen=True, slots=True)
class FeatureSelection:
    """Normalized values derived from the supported feature set."""

    layout_q: str
    layout_kv: str
    layout_out: str
    total_q: int
    num_heads_q: int
    block_size: int


def _check_dtype(tensor, name, allowed):
    display_name = name[0].upper() + name[1:]
    if tensor.dtype not in allowed:
        if allowed == SUPPORTED_DTYPES:
            raise TypeError(
                f"{display_name} must be float16 or bfloat16; got {tensor.dtype}"
            )
        raise TypeError(f"{display_name} must be {allowed}; got {tensor.dtype}")


def _require_inner_contiguous(tensor, name):
    """Accept outer strides while keeping each innermost row DMA-compatible."""
    display_name = name[0].upper() + name[1:]
    if tensor.dim() > 0 and tensor.stride(-1) != 1:
        raise ValueError(f"{display_name} must be contiguous in its last dimension")


def _first_noncontiguous_dim(tensor):
    expected_stride = 1
    for dim in range(tensor.dim() - 1, -1, -1):
        if tensor.stride(dim) != expected_stride:
            return dim
        expected_stride *= tensor.shape[dim]
    return -1


def _normalize_layout(value, allowed, name):
    display_name = name[0].upper() + name[1:]
    if value not in allowed:
        raise ValueError(
            f"{display_name} must be one of {sorted(allowed)}; got {value!r}"
        )
    return value


def _check_required_inputs(*, metadata, block_table, cache_seqlens):
    """Check that unconditional public inputs are present."""
    if metadata is None:
        raise ValueError("Metadata is required")
    if cache_seqlens is None or block_table is None:
        raise ValueError("Cache_seqlens and block_table are required")


def _check_single_arguments(
    *,
    q,
    k_cache,
    block_table,
    cache_seqlens,
    cu_seqlens_q,
    seqused_q,
    attn_mask,
    metadata,
    head_dim_v,
    mask_mode,
    max_seqlen_q,
    max_seqlen_kv,
    layout_q,
    layout_kv,
    layout_out,
    return_softmax_lse,
):
    """Check constraints that can be decided from one argument alone."""
    max_seqlen_q = (
        UNSPECIFIED_SEQLEN if max_seqlen_q is None else max_seqlen_q
    )
    max_seqlen_kv = (
        UNSPECIFIED_SEQLEN if max_seqlen_kv is None else max_seqlen_kv
    )
    if type(max_seqlen_q) is not int:
        raise TypeError("Max_seqlen_q must be an int")
    if type(max_seqlen_kv) is not int:
        raise TypeError("Max_seqlen_kv must be an int")
    if max_seqlen_q != UNSPECIFIED_SEQLEN:
        raise ValueError(
            f"Max_seqlen_q must be {UNSPECIFIED_SEQLEN} or omitted"
        )
    if max_seqlen_kv != UNSPECIFIED_SEQLEN:
        raise ValueError(
            f"Max_seqlen_kv must be {UNSPECIFIED_SEQLEN} or omitted"
        )
    if type(head_dim_v) is not int or head_dim_v != HEAD_DIM_V:
        raise ValueError(f"Head_dim_v must be {HEAD_DIM_V}; got {head_dim_v}")
    if not isinstance(return_softmax_lse, bool):
        raise TypeError("Return_softmax_lse must be bool")
    if type(mask_mode) is not int:
        raise TypeError("Mask_mode must be an int")

    for value, name in (
        (layout_q, "layout_q"),
        (layout_kv, "layout_kv"),
        (layout_out, "layout_out"),
    ):
        if value is not None and not isinstance(value, str):
            display_name = name[0].upper() + name[1:]
            raise ValueError(f"{display_name} must be a string; got {value!r}")

    _check_dtype(q, "q", SUPPORTED_DTYPES)
    _check_dtype(k_cache, "k_cache", SUPPORTED_DTYPES)
    if block_table.dtype != torch.int32:
        raise TypeError("Block_table must be int32")
    if cache_seqlens.dtype != torch.int32:
        raise TypeError("Cache_seqlens must be int32")
    if cache_seqlens.dim() != 1:
        raise ValueError("Cache_seqlens must be one-dimensional")
    if cu_seqlens_q is not None and cu_seqlens_q.dtype != torch.int32:
        raise TypeError("Cu_seqlens_q must be int32")
    if seqused_q is not None and seqused_q.dtype != torch.int32:
        raise TypeError("Seqused_q must be int32")
    if attn_mask is not None:
        mask_shape = (MASK_TEMPLATE_SIZE, MASK_TEMPLATE_SIZE)
        if tuple(attn_mask.shape) != mask_shape:
            raise ValueError(f"Attn_mask must be {mask_shape}")
        if attn_mask.dtype != torch.int8:
            raise TypeError("Attn_mask must be int8")

    if not isinstance(metadata, torch.Tensor):
        raise ValueError(
            "Metadata tile_table must be replaced by one flat int32 tensor"
        )
    if metadata.dtype != torch.int32 or metadata.dim() != 1:
        raise ValueError("Metadata must be one flat int32 tensor")
    if metadata.numel() <= 0:
        raise ValueError("Metadata first dimension must be greater than 0")

    for tensor, name in (
        (q, "q"),
        (block_table, "block_table"),
        (cache_seqlens, "cache_seqlens"),
        (metadata, "metadata"),
        (cu_seqlens_q, "cu_seqlens_q"),
        (seqused_q, "seqused_q"),
        (attn_mask, "attn_mask"),
    ):
        if tensor is not None:
            _require_inner_contiguous(tensor, name)


def _check_cross_arguments(
    *,
    q,
    k_cache,
    block_table,
    cache_seqlens,
    cu_seqlens_q,
    seqused_q,
    attn_mask,
    mask_mode,
    layout_q,
):
    """Check relationships between two or more public inputs."""
    if k_cache.dtype != q.dtype:
        raise TypeError("Q and k_cache must have the same dtype")
    if (
        mask_mode in SUPPORTED_MASK_MODES
        and (mask_mode == NO_MASK_MODE) != (attn_mask is None)
    ):
        raise ValueError(
            f"Mask_mode = {NO_MASK_MODE} forbids attn_mask; "
            f"mask_mode = {CAUSAL_MASK_MODE} requires it"
        )

    batch = cache_seqlens.shape[0]
    if not 0 < batch < MAX_BATCH_SIZE:
        raise ValueError(f"Batch size must be in (0, {MAX_BATCH_SIZE})")
    if block_table.dim() != 2 or block_table.shape[0] != batch:
        raise ValueError("Block_table must have shape (B, max_pages)")
    if block_table.shape[1] <= 0:
        raise ValueError("Block_table max_pages must be positive")
    if layout_q == "TND":
        if cu_seqlens_q is None or cu_seqlens_q.shape != (batch + 1,):
            raise ValueError("TND requires cu_seqlens_q with shape (B+1,)")
    if seqused_q is not None and seqused_q.shape != (batch,):
        raise ValueError("Seqused_q must have shape (B,)")
    return InputRelations(batch=batch)


def _check_supported_features(
    *,
    q,
    k_cache,
    layout_q,
    layout_kv,
    layout_out,
    mask_mode,
):
    """Check limits imposed by the currently implemented feature set."""
    layout_q = _normalize_layout(layout_q, SUPPORTED_Q_LAYOUTS, "layout_q")
    layout_kv = _normalize_layout(layout_kv, SUPPORTED_KV_LAYOUTS, "layout_kv")
    layout_out = "NTD" if layout_out is None else _normalize_layout(
        layout_out, ("NTD",), "layout_out"
    )
    if mask_mode not in SUPPORTED_MASK_MODES:
        raise ValueError(
            f"Mask_mode must be {NO_MASK_MODE} or {CAUSAL_MASK_MODE}; "
            f"got {mask_mode}"
        )

    if q.dim() != 3 or q.shape[2] != HEAD_DIM_QK:
        raise ValueError(f"TND q must have shape (T, N, {HEAD_DIM_QK})")
    if q.stride(1) != HEAD_DIM_QK:
        raise ValueError(
            "TND q supports non-contiguous storage only between tokens; "
            "heads inside each token must be contiguous"
        )
    total_q, num_heads_q = q.shape[0], q.shape[1]
    if total_q <= 0:
        raise ValueError("TND q token count must be positive")
    if num_heads_q not in SUPPORTED_HEAD_COUNTS:
        supported_heads = " or ".join(map(str, SUPPORTED_HEAD_COUNTS))
        raise ValueError(f"Q head count must be {supported_heads}")

    noncontiguous_dim = _first_noncontiguous_dim(k_cache)
    max_noncontiguous_dim = 0 if layout_kv == "PA_BBND" else 1
    if noncontiguous_dim > max_noncontiguous_dim:
        allowed = (
            "dimension 0" if max_noncontiguous_dim == 0 else "dimensions 0 or 1"
        )
        raise ValueError(
            f"{layout_kv} k_cache supports non-contiguous storage only in "
            f"{allowed}; got dimension {noncontiguous_dim}"
        )

    if layout_kv == "PA_NZ":
        if (
            k_cache.dim() != 5
            or k_cache.shape[1] != 1
            or k_cache.shape[2] != PA_NZ_D1
            or k_cache.shape[4] != PA_NZ_D0
        ):
            raise ValueError(
                "PA_NZ k_cache must be "
                f"(blocks, 1, {PA_NZ_D1}, block, {PA_NZ_D0})"
            )
        block_size = k_cache.shape[3]
    else:
        if k_cache.dim() != 4 or k_cache.shape[2:] != (1, HEAD_DIM_QK):
            raise ValueError(
                "PA_BBND k_cache must be "
                f"(blocks, block, 1, {HEAD_DIM_QK})"
            )
        block_size = k_cache.shape[1]
    if k_cache.shape[0] <= 0:
        raise ValueError("K_cache block count must be positive")
    if block_size != KV_BLOCK_SIZE:
        raise ValueError(f"Paged KV block size must be {KV_BLOCK_SIZE}")

    return FeatureSelection(
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        total_q=total_q,
        num_heads_q=num_heads_q,
        block_size=block_size,
    )


def check_flash_mla_inputs(
    *,
    q,
    k_cache,
    block_table,
    cache_seqlens,
    cu_seqlens_q,
    seqused_q,
    attn_mask,
    metadata,
    head_dim_v,
    mask_mode,
    max_seqlen_q,
    max_seqlen_kv,
    layout_q,
    layout_kv,
    layout_out,
    return_softmax_lse,
):
    """Validate inputs by category and derive host launch metadata."""
    layout_q = "TND" if layout_q is None else layout_q
    _check_required_inputs(
        metadata=metadata,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
    )
    _check_single_arguments(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        attn_mask=attn_mask,
        metadata=metadata,
        head_dim_v=head_dim_v,
        mask_mode=mask_mode,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        return_softmax_lse=return_softmax_lse,
    )
    relations = _check_cross_arguments(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        attn_mask=attn_mask,
        mask_mode=mask_mode,
        layout_q=layout_q,
    )
    features = _check_supported_features(
        q=q,
        k_cache=k_cache,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        mask_mode=mask_mode,
    )

    return CheckedInputs(
        layout_q=features.layout_q,
        layout_kv=features.layout_kv,
        layout_out=features.layout_out,
        batch=relations.batch,
        total_q=features.total_q,
        num_heads_q=features.num_heads_q,
        logical_rows=features.total_q * features.num_heads_q,
        block_size=features.block_size,
    )
