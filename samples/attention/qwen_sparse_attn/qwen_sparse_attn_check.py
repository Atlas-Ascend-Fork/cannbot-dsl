# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Qwen sparse attention Python adapter validation."""

import math
from enum import IntEnum

import torch


class SparseMode(IntEnum):
    """Sparse mask mode aligned with the FusionAttention ABI."""

    RIGHT_DOWN_CAUSAL = 3


class QuantMode(IntEnum):
    """Quantization modes supported by Qwen sparse attention."""

    NO_QUANT = 0
    FP8_FULL_QUANT = 5


QSA_METADATA_VERSION = 1
QSA_PACKED_METADATA_VERSION = 2


def _as_quant_mode(value):
    try:
        return QuantMode(int(value))
    except (TypeError, ValueError):
        supported = ", ".join(f"{mode.name}={int(mode)}" for mode in QuantMode)
        raise ValueError(f"Quant_mode must be one of: {supported}") from None


def resolve_block128_mode(input_dtype, quant_mode, attention_out_dtype):
    """Validate and resolve a supported Block 128 execution mode."""
    quant_mode = _as_quant_mode(quant_mode)
    if quant_mode is QuantMode.NO_QUANT:
        if input_dtype not in (torch.bfloat16, torch.float16):
            raise TypeError("QuantMode.NO_QUANT requires BF16 or FP16 Q/K/V")
        if attention_out_dtype not in (None, input_dtype):
            raise ValueError("Regular attention_out_dtype must be None or input dtype")
        return dict(
            name="bf16" if input_dtype == torch.bfloat16 else "fp16",
            input=input_dtype,
            score=input_dtype,
            output=input_dtype,
            quant=QuantMode.NO_QUANT,
        )
    if input_dtype != torch.float8_e4m3fn:
        raise TypeError("QuantMode.FP8_FULL_QUANT requires float8_e4m3fn Q/K/V")
    if attention_out_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FP8 attention_out_dtype must be FP16 or BF16")
    return dict(
        name="fp8_fp16" if attention_out_dtype == torch.float16 else "fp8_bf16",
        input=input_dtype,
        score=attention_out_dtype,
        output=attention_out_dtype,
        quant=QuantMode.FP8_FULL_QUANT,
    )


def _validate_tensor_inputs(
    query,
    key,
    value,
    sparse_block_idx,
    sparse_block_count,
    cu_seqlens_q,
    seqused_kv,
    block_table,
    seqused_q,
):
    named = dict(
        query=query,
        key=key,
        value=value,
        sparse_block_idx=sparse_block_idx,
        sparse_block_count=sparse_block_count,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        block_table=block_table,
    )
    if seqused_q is not None:
        named["seqused_q"] = seqused_q
    if any(not isinstance(tensor, torch.Tensor) for tensor in named.values()):
        raise TypeError("All tensor arguments must be torch.Tensor")
    if len({tensor.device for tensor in named.values()}) != 1:
        raise ValueError("All tensors must be on the same device")
    if query.device.type != "npu":
        raise ValueError("Qwen sparse attention requires NPU tensors")
    return named


def _validate_common_shapes(
    query,
    key,
    value,
    sparse_block_idx,
    sparse_block_count,
    cu_seqlens_q,
    seqused_kv,
    block_table,
    seqused_q,
    n2_num,
):
    if query.ndim != 3 or query.shape[2] != 128 or query.shape[1] <= 0:
        raise ValueError("Query must be TND [T,N1,128] with N1>0")
    if value.shape != key.shape or key.shape[0] <= 0 or n2_num <= 0:
        raise ValueError("Key/value shapes must match with P,N2>0")
    if not query.is_contiguous():
        raise ValueError("Query must be contiguous TND")
    for tensor in (key, value):
        expected = 1
        for axis in (3, 2, 1):
            if tensor.shape[axis] > 1 and tensor.stride(axis) != expected:
                raise ValueError("KV must be contiguous within each page")
            expected *= tensor.shape[axis]
        if tensor.stride(0) < expected or tensor.stride(0) % expected:
            raise ValueError(
                "KV page stride must be a positive multiple of page elements"
            )
    if (
        cu_seqlens_q.ndim != 1
        or cu_seqlens_q.numel() < 2
        or cu_seqlens_q.dtype != torch.int64
    ):
        raise ValueError("Cu_seqlens_q must be int64 [B+1]")
    batch = cu_seqlens_q.numel() - 1
    if (
        block_table.ndim != 2
        or block_table.shape[0] != batch
        or block_table.shape[1] <= 0
    ):
        raise ValueError("Block_table must be [B,max_pages]")
    if seqused_kv.shape != (batch,) or (
        seqused_q is not None and seqused_q.shape != (batch,)
    ):
        raise ValueError("Seqused lengths must be [B]")
    if sparse_block_idx.ndim != 3 or sparse_block_idx.shape[:2] != (
        n2_num,
        query.shape[0],
    ):
        raise ValueError("Sparse_block_idx must be [N2,T,topK]")
    if not 1 <= sparse_block_idx.shape[2] <= 256 or sparse_block_count.shape != (
        n2_num,
        query.shape[0],
    ):
        raise ValueError("Require topK in [1,256], count shape [N2,T]")


def _validate_integer_dtypes(named):
    for name in (
        "sparse_block_idx",
        "sparse_block_count",
        "seqused_kv",
        "block_table",
        "seqused_q",
    ):
        if name in named and named[name].dtype != torch.int32:
            raise TypeError(f"{name} must be int32")


def validate_block128_inputs(
    query,
    key,
    value,
    sparse_block_idx,
    sparse_block_count,
    cu_seqlens_q,
    seqused_kv,
    block_table,
    seqused_q,
    softmax_scale,
    quant_mode=QuantMode.NO_QUANT,
    attention_out_dtype=None,
):
    """Validate all tensor inputs consumed by the Block 128 adapter."""
    named = _validate_tensor_inputs(
        query,
        key,
        value,
        sparse_block_idx,
        sparse_block_count,
        cu_seqlens_q,
        seqused_kv,
        block_table,
        seqused_q,
    )
    if key.ndim != 4 or key.shape[1] != 128 or key.shape[3] != 128:
        raise ValueError("Key must be PA_BBND [P,128,N2,128]")
    n2_num = key.shape[2]
    _validate_common_shapes(
        query,
        key,
        value,
        sparse_block_idx,
        sparse_block_count,
        cu_seqlens_q,
        seqused_kv,
        block_table,
        seqused_q,
        n2_num,
    )
    mode = resolve_block128_mode(query.dtype, quant_mode, attention_out_dtype)
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("Q/K/V must have the same dtype")
    if query.shape[1] % n2_num or query.shape[1] // n2_num > 128:
        raise ValueError("Require N1 divisible by N2 and G<=128")
    _validate_integer_dtypes(named)
    if not math.isfinite(float(softmax_scale)):
        raise ValueError("Softmax_scale must be finite")
    return mode


def validate_block128_attrs(
    block_shape,
    is_packed_gqa,
    layout_q,
    layout_kv,
    mask_mode,
    quant_mode,
    dst_type_max,
    softmax_precision,
    return_softmax_lse,
):
    if list(block_shape) != [1, 128]:
        raise ValueError("QwenSparseAttnBlock128 requires block_shape=[1, 128]")
    if is_packed_gqa is not True:
        raise ValueError("QwenSparseAttnBlock128 requires is_packed_gqa=True")
    if layout_q != "TND" or layout_kv != "PA_BBND":
        raise ValueError(
            "QwenSparseAttnBlock128 requires layout_q='TND' and layout_kv='PA_BBND'"
        )
    if int(mask_mode) != SparseMode.RIGHT_DOWN_CAUSAL:
        raise ValueError(
            "QwenSparseAttnBlock128 supports only "
            "SparseMode.RIGHT_DOWN_CAUSAL (mask_mode=3)"
        )
    _as_quant_mode(quant_mode)
    _validate_common_attrs(dst_type_max, softmax_precision, return_softmax_lse)


def _validate_common_attrs(dst_type_max, softmax_precision, return_softmax_lse):
    if float(dst_type_max) != 0.0:
        raise ValueError("Dst_type_max must be 0.0")
    if int(softmax_precision) != 1:
        raise ValueError("Ascend 950 requires softmax_precision=1")
    if bool(return_softmax_lse):
        raise ValueError("Qwen sparse attention does not support softmax LSE output")


def require_none(operator_name, **options):
    present = [name for name, value in options.items() if value is not None]
    if present:
        raise ValueError(f"{operator_name} requires None for: " + ", ".join(present))


def validate_attn_mask(attn_mask, query, mask_mode):
    if (
        not isinstance(attn_mask, torch.Tensor)
        or attn_mask.device != query.device
        or attn_mask.dtype != torch.int8
        or tuple(attn_mask.shape) != (2048, 2048)
        or not attn_mask.is_contiguous()
    ):
        raise ValueError(
            f"Mask_mode={int(mask_mode)} requires contiguous int8 attn_mask "
            "[2048,2048] on the query device"
        )


def validate_block128_metadata(metadata, query):
    if metadata is None:
        raise ValueError(
            "Metadata is required and must be produced by qwen_sparse_attn_metadata"
        )
    if not isinstance(metadata, (tuple, list)) or len(metadata) != 5:
        raise ValueError(
            "Metadata must be the complete "
            "(coreSpans, tasks, pages, counts, status) result from "
            "qwen_sparse_attn_metadata"
        )
    core_spans, tasks, pages, counts, status = metadata
    if int(status) != 0:
        raise RuntimeError(
            f"Qwen_sparse_attn_metadata failed with status {int(status)}"
        )
    if not isinstance(pages, torch.Tensor) or pages.ndim != 2 or pages.shape[0] < 2:
        raise ValueError("Metadata pages must contain a page row and mask rows")
    _validate_metadata_tensor("coreSpans", core_spans, query, torch.int64, 2)
    _validate_metadata_tensor("tasks", tasks, query, torch.int64, 5)
    _validate_metadata_tensor("pages", pages, query, torch.int32, pages.shape[0])
    _validate_metadata_counts(counts, query, 4)
    return core_spans, tasks, pages, core_spans.shape[1]


def _validate_metadata_tensor(name, tensor, query, dtype, rows):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Metadata {name} must be a torch.Tensor")
    if (
        tensor.device != query.device
        or tensor.dtype != dtype
        or tensor.ndim != 2
        or tensor.shape[0] != rows
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            f"Metadata {name} must be contiguous {dtype} [{rows}, capacity] "
            f"on {query.device}"
        )


def _validate_metadata_counts(counts, query, size):
    if (
        not isinstance(counts, torch.Tensor)
        or counts.device != query.device
        or counts.dtype != torch.int32
        or counts.ndim != 1
        or counts.numel() != size
        or not counts.is_contiguous()
    ):
        raise ValueError(
            f"Metadata counts must be contiguous int32 [{size}] on the query device"
        )
