# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host-side GQA MXFP8 checks, ordered from options to physical layouts."""

import math
import struct
import torch


# Shared public contract.
class KVLayout:
    """Plain-string KV layouts accepted by the public API and DSL tracer."""

    TND = "TND"
    PA_BNBD = "PA_BNBD"
    PA_BBND = "PA_BBND"
    PA_NZ = "PA_NZ"
    PAGED = frozenset((PA_BNBD, PA_BBND, PA_NZ))
    ALL = PAGED | {TND}


class QLayout:
    """Public Q layouts supported by the current GQA MXFP8 implementation."""

    TND = "TND"


SUPPORTED_Q_LAYOUTS = frozenset((QLayout.TND,))

MASK_MODE_NO_MASK = 0
MASK_MODE_RIGHT_DOWN_CAUSAL = 3
SUPPORTED_MASK_MODES = (
    MASK_MODE_NO_MASK,
    MASK_MODE_RIGHT_DOWN_CAUSAL,
)

HEAD_DIM_64 = 64
HEAD_DIM_72 = 72
HEAD_DIM_128 = 128
HEAD_DIM_256 = 256
SUPPORTED_HEAD_DIMS = (
    HEAD_DIM_64,
    HEAD_DIM_72,
    HEAD_DIM_128,
    HEAD_DIM_256,
)

SUPPORTED_PA_BLOCK_SIZES = (64, 128, 256, 512, 1024)
MXFP8_SCALE_PAIR_WIDTH = 2
MXFP8_GROUP_ELEMENTS = 64
PA_NZ_DATA_BLOCK = 32
PA_NZ_TOKEN_BLOCK = 16
PA_NZ_VSCALE_DIM_BLOCK = 16
# Decode places each batch at the start of its own 128-row physical Q slot.
DECODE_QUERY_ROWS = 128
_PA_TENSOR_NAMES = ("K", "V", "KScale", "VScale")


class PAStride:
    """Physical PA stride modes used by test fixtures."""

    CONTIGUOUS = "C"
    AXIS0 = "A0"
    AXIS01 = "A01"
    ALL = frozenset((CONTIGUOUS, AXIS0, AXIS01))


def _check_integer(name, value, minimum=None):
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Argument {name} must be int, got {type(value).__name__}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"Argument {name} must be at least {minimum}, got {value}.")


def _check_boolean(name, value):
    if not isinstance(value, bool):
        raise TypeError(f"Argument {name} must be bool, got {type(value).__name__}.")


def _check_softmax_scale(value):
    """Validate the Python scalar against the kernel's FP32 representation."""
    if not isinstance(value, float):
        raise TypeError("Argument softmax_scale must be a Python float.")
    try:
        fp32_value = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error):
        raise ValueError(
            "Argument softmax_scale must be finite and positive in FP32."
        ) from None
    if not math.isfinite(fp32_value) or fp32_value <= 0:
        raise ValueError("Argument softmax_scale must be finite and positive in FP32.")


# Layer 1: public scalar options and layout selection.
def check_mask_mode(mask_mode):
    _check_integer("mask_mode", mask_mode)
    if mask_mode not in SUPPORTED_MASK_MODES:
        raise ValueError(
            f"GQA MXFP8 supports mask_mode {SUPPORTED_MASK_MODES}, got {mask_mode}."
        )


def check_q_scale_rank(q_scale):
    if q_scale.ndim not in (4, 5):
        raise ValueError(
            "Argument q_scale must be rank 4 (prefill) or rank 5 (decode)."
        )


def check_q_layout(q_layout):
    """Validate the public Q layout before any shape or JIT work."""
    if not isinstance(q_layout, str):
        raise TypeError(
            f"Argument q_layout must be str, got {type(q_layout).__name__}."
        )
    if q_layout not in SUPPORTED_Q_LAYOUTS:
        raise ValueError(
            f"Unsupported GQA MXFP8 q_layout {q_layout!r}; "
            "the current implementation supports only TND."
        )
    return q_layout


def check_kv_layout(paged, kv_layout):
    """Validate an explicit KV layout and its page-attention consistency."""
    if not isinstance(kv_layout, str):
        raise TypeError(
            f"Argument kv_layout must be str, got {type(kv_layout).__name__}."
        )
    if kv_layout not in KVLayout.ALL:
        raise ValueError(
            f"Unsupported GQA MXFP8 KV layout {kv_layout!r}; expected one of "
            f"{sorted(KVLayout.ALL)}."
        )
    expected_paged = kv_layout in KVLayout.PAGED
    if paged != expected_paged:
        raise ValueError(
            f"Argument paged={paged} is inconsistent with kv_layout={kv_layout!r}; "
            f"this layout requires paged={expected_paged}."
        )
    return kv_layout


def check_head_dim_layout(head_dim, kv_layout):
    _check_integer("head_dim", head_dim, minimum=1)
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"GQA MXFP8 supports head_dim {SUPPORTED_HEAD_DIMS}, got {head_dim}."
        )
    if head_dim == HEAD_DIM_72 and kv_layout == KVLayout.PA_NZ:
        raise ValueError(f"GQA MXFP8 head_dim={HEAD_DIM_72} does not support PA_NZ.")


def _check_tensor(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Argument {name} must be a torch.Tensor.")


def _check_dtype(name, tensor, expected):
    expected_dtypes = expected if isinstance(expected, tuple) else (expected,)
    if tensor.dtype not in expected_dtypes:
        raise TypeError(
            f"Argument {name} must have dtype in {expected_dtypes}, got {tensor.dtype}."
        )


def _check_shape(name, tensor, expected):
    if tuple(tensor.shape) != tuple(expected):
        raise ValueError(
            f"Argument {name} must have shape {tuple(expected)}, "
            f"got {tuple(tensor.shape)}."
        )


def _check_same_device(query, tensors):
    for name, tensor in tensors:
        if tensor.device != query.device:
            raise ValueError(
                f"Argument {name} must be on device {query.device}, "
                f"got {tensor.device}."
            )


def _check_query_scale_shape(q_scale, query, kv_heads, groups64):
    query_tokens, query_heads, _ = query.shape
    if q_scale.ndim == 4:
        _check_shape(
            "q_scale",
            q_scale,
            (
                query_tokens,
                query_heads,
                groups64,
                MXFP8_SCALE_PAIR_WIDTH,
            ),
        )
        return
    head_group = query_heads // kv_heads
    _check_shape(
        "q_scale",
        q_scale,
        (
            kv_heads,
            query_tokens,
            head_group,
            groups64,
            MXFP8_SCALE_PAIR_WIDTH,
        ),
    )


def _check_tnd_kv_shapes(key, value, k_scale, v_scale, head_dim, groups64):
    if key.ndim != 3 or value.ndim != 3:
        raise ValueError("TND key and value must be rank-3 tensors.")
    if tuple(key.shape) != tuple(value.shape):
        raise ValueError("TND key and value must have identical shapes.")
    if key.shape[2] != head_dim:
        raise ValueError(
            f"TND K/V head dimension must be {head_dim}, got {key.shape[2]}."
        )
    tokens, kv_heads, _ = key.shape
    _check_shape(
        "k_scale",
        k_scale,
        (tokens, kv_heads, groups64, MXFP8_SCALE_PAIR_WIDTH),
    )
    if (
        v_scale.ndim != 4
        or v_scale.shape[0] <= 0
        or tuple(v_scale.shape[1:]) != (kv_heads, head_dim, MXFP8_SCALE_PAIR_WIDTH)
    ):
        raise ValueError("TND v_scale must have shape [positive_groups, N_kv, D, 2].")
    return kv_heads


def _check_pa_dense_shapes(
    key, value, k_scale, v_scale, block_size, head_dim, groups64, kv_layout
):
    """Check ordinary PA layouts; only the page-token/head axis order differs."""
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"{kv_layout} key and value must be rank-4 tensors.")
    if tuple(key.shape) != tuple(value.shape):
        raise ValueError(f"{kv_layout} key and value must have identical shapes.")
    if kv_layout == KVLayout.PA_BNBD:
        pages, kv_heads, page_size, kv_head_dim = key.shape
        expected_k_scale = (
            pages,
            kv_heads,
            block_size,
            groups64,
            MXFP8_SCALE_PAIR_WIDTH,
        )
        expected_v_scale = (
            pages,
            kv_heads,
            block_size // MXFP8_GROUP_ELEMENTS,
            head_dim,
            MXFP8_SCALE_PAIR_WIDTH,
        )
        shape_description = "[pages, N_kv, block_size, D]"
    else:
        pages, page_size, kv_heads, kv_head_dim = key.shape
        expected_k_scale = (
            pages,
            block_size,
            kv_heads,
            groups64,
            MXFP8_SCALE_PAIR_WIDTH,
        )
        expected_v_scale = (
            pages,
            block_size // MXFP8_GROUP_ELEMENTS,
            kv_heads,
            head_dim,
            MXFP8_SCALE_PAIR_WIDTH,
        )
        shape_description = "[pages, block_size, N_kv, D]"
    if page_size != block_size or kv_head_dim != head_dim:
        raise ValueError(f"{kv_layout} K/V shape must be {shape_description}.")
    _check_shape("k_scale", k_scale, expected_k_scale)
    _check_shape("v_scale", v_scale, expected_v_scale)
    return kv_heads


def _check_public_options(
    softmax_scale,
    mask_mode,
    paged,
    block_size,
    kv_layout,
    q_layout,
    return_softmax_lse,
    max_seqlen_q,
):
    """Check host scalars before tensor metadata is accessed."""
    _check_softmax_scale(softmax_scale)
    check_mask_mode(mask_mode)
    _check_boolean("paged", paged)
    _check_integer("block_size", block_size, minimum=1)
    check_kv_layout(paged, kv_layout)
    check_q_layout(q_layout)
    _check_boolean("return_softmax_lse", return_softmax_lse)
    _check_integer("max_seqlen_q", max_seqlen_q, minimum=-1)
    if paged:
        check_paged_kv_options(block_size, kv_layout)


def _check_public_tensors(
    query,
    key,
    value,
    q_scale,
    k_scale,
    v_scale,
    p_scale,
    attn_mask,
    block_table,
    seqused_kv,
    seqused_q,
):
    """Check existence, shared device and storage dtype in signature order."""
    tensors = (
        ("query", query, torch.float8_e4m3fn),
        ("key", key, torch.float8_e4m3fn),
        ("value", value, torch.float8_e4m3fn),
        ("q_scale", q_scale, torch.float8_e8m0fnu),
        ("k_scale", k_scale, torch.float8_e8m0fnu),
        ("v_scale", v_scale, torch.float8_e8m0fnu),
        ("p_scale", p_scale, torch.float32),
        ("attn_mask", attn_mask, torch.float32),
        ("block_table", block_table, torch.int32),
        ("seqused_kv", seqused_kv, torch.int64),
        ("seqused_q", seqused_q, torch.int64),
    )
    for name, tensor, _ in tensors:
        _check_tensor(name, tensor)
    _check_same_device(query, ((name, tensor) for name, tensor, _ in tensors[1:]))
    for name, tensor, dtype in tensors:
        _check_dtype(name, tensor, dtype)


def _check_query_and_sequence_shapes(
    query, q_scale, p_scale, attn_mask, seqused_kv, seqused_q, kv_layout
):
    """Return batch size after independent rank and dimension checks."""
    if query.ndim != 3:
        raise ValueError(
            f"For q_layout='TND', query must have shape [T, N_q, D], got rank {query.ndim}."
        )
    if query.shape[1] <= 0:
        raise ValueError(f"Query head count must be positive, got {query.shape[1]}.")
    check_head_dim_layout(query.shape[2], kv_layout)
    check_q_scale_rank(q_scale)
    _check_shape("p_scale", p_scale, (1,))
    if attn_mask.ndim != 3:
        raise ValueError(
            f"Argument attn_mask must be rank 3 [B, S_q, S_kv], got rank {attn_mask.ndim}."
        )
    if seqused_q.ndim != 1 or seqused_kv.ndim != 1:
        raise ValueError("Arguments seqused_q and seqused_kv must be rank-1 tensors.")
    if seqused_q.numel() <= 0 or seqused_q.numel() != seqused_kv.numel():
        raise ValueError(
            "Arguments seqused_q and seqused_kv must have the same positive length."
        )
    for name, lengths in (("seqused_q", seqused_q), ("seqused_kv", seqused_kv)):
        if not lengths.is_contiguous():
            raise ValueError(f"Argument {name} must be contiguous.")
    return seqused_q.numel()


def _check_kv_shapes(key, value, k_scale, v_scale, block_size, head_dim, kv_layout):
    groups64 = (head_dim + MXFP8_GROUP_ELEMENTS - 1) // MXFP8_GROUP_ELEMENTS
    if kv_layout == KVLayout.TND:
        return _check_tnd_kv_shapes(key, value, k_scale, v_scale, head_dim, groups64)
    if kv_layout in (KVLayout.PA_BNBD, KVLayout.PA_BBND):
        return _check_pa_dense_shapes(
            key, value, k_scale, v_scale, block_size, head_dim, groups64, kv_layout
        )
    check_pa_nz_tensors(key, value, k_scale, v_scale, block_size, head_dim)
    return key.shape[1]


def _check_attention_bounds(
    query, q_scale, attn_mask, block_table, batch_size, paged, max_seqlen_q
):
    if attn_mask.shape[0] != batch_size:
        raise ValueError(
            f"Attention mask batch size must be {batch_size}, got {attn_mask.shape[0]}."
        )
    if max_seqlen_q > attn_mask.shape[1]:
        raise ValueError(
            "Argument max_seqlen_q must not exceed the attention mask Q "
            f"extent {attn_mask.shape[1]}, got {max_seqlen_q}."
        )
    if q_scale.ndim == 5 and max_seqlen_q != 0 and query.shape[0] > 0:
        if max_seqlen_q > DECODE_QUERY_ROWS:
            raise ValueError(
                f"Decode max_seqlen_q must not exceed {DECODE_QUERY_ROWS}, got {max_seqlen_q}."
            )
        required_tokens = batch_size * DECODE_QUERY_ROWS
        # Zero-token output exits before launch. A nonempty Decode Q allocation
        # must cover every fixed batch slot, even when valid prefixes are short.
        if query.shape[0] < required_tokens:
            raise ValueError(
                f"Decode Q storage must reserve at least {required_tokens} tokens for {batch_size} batches."
            )
    if paged and (block_table.ndim != 2 or block_table.shape[0] != batch_size):
        raise ValueError(
            f"Paged block_table must have shape [B, blocks] with B={batch_size}."
        )


def check_public_inputs(
    query,
    key,
    value,
    q_scale,
    k_scale,
    v_scale,
    p_scale,
    softmax_scale,
    attn_mask,
    block_table,
    seqused_kv,
    seqused_q,
    mask_mode,
    paged,
    block_size,
    kv_layout,
    q_layout,
    return_softmax_lse,
    max_seqlen_q,
):
    """Validate host scalars and tensor metadata before allocation or tracing.

    Tensor contents are not copied to the host. The caller owns nonnegative
    sequence lengths within the supplied bounds/capacities, valid physical page
    IDs and finite positive p_scale values. Q scale rank selects Prefill (4) or
    Decode (5); Decode reserves 128 physical Q rows per batch. A maximum Q
    length of -1 is unspecified; it is never inferred from the mask.
    Successful validation returns None.
    """
    _check_public_options(
        softmax_scale,
        mask_mode,
        paged,
        block_size,
        kv_layout,
        q_layout,
        return_softmax_lse,
        max_seqlen_q,
    )
    _check_public_tensors(
        query,
        key,
        value,
        q_scale,
        k_scale,
        v_scale,
        p_scale,
        attn_mask,
        block_table,
        seqused_kv,
        seqused_q,
    )
    batch_size = _check_query_and_sequence_shapes(
        query, q_scale, p_scale, attn_mask, seqused_kv, seqused_q, kv_layout
    )
    kv_heads = _check_kv_shapes(
        key, value, k_scale, v_scale, block_size, query.shape[2], kv_layout
    )
    query_heads = query.shape[1]
    if kv_heads <= 0 or query_heads % kv_heads != 0:
        raise ValueError(
            f"Query heads {query_heads} must be divisible by positive KV heads {kv_heads}."
        )
    groups64 = (query.shape[2] + MXFP8_GROUP_ELEMENTS - 1) // MXFP8_GROUP_ELEMENTS
    _check_query_scale_shape(q_scale, query, kv_heads, groups64)
    _check_attention_bounds(
        query, q_scale, attn_mask, block_table, batch_size, paged, max_seqlen_q
    )


# Layer 2: PA storage constraints and compile-time tile compatibility.
def check_pa_tile(block_size, tile_n, paged):
    """Check that a compute tile can gather whole physical pages when needed."""
    _check_boolean("paged", paged)
    if not paged:
        return
    _check_integer("block_size", block_size, minimum=1)
    _check_integer("tile_n", tile_n, minimum=1)
    if block_size < tile_n:
        if tile_n % block_size != 0 or block_size % MXFP8_GROUP_ELEMENTS != 0:
            raise ValueError(
                "PA cross-page loads require tile_n to be divisible by block_size "
                f"and block_size to be divisible by {MXFP8_GROUP_ELEMENTS}."
            )


def check_pa_nz_tensors(key, value, k_scale, v_scale, block_size, head_dim):
    """Check physical PA_NZ data and E8M0 scale shapes for direct tile access."""
    check_paged_kv_options(block_size, KVLayout.PA_NZ)
    check_head_dim_layout(head_dim, KVLayout.PA_NZ)
    for name, tensor in (
        ("key", key),
        ("value", value),
        ("k_scale", k_scale),
        ("v_scale", v_scale),
    ):
        _check_tensor(name, tensor)
    if key.ndim != 5 or value.ndim != 5:
        raise ValueError("PA_NZ key and value must be rank-5 tensors.")
    if tuple(key.shape) != tuple(value.shape):
        raise ValueError("PA_NZ key and value must have identical shapes.")
    pages, heads, dim_blocks, page_size, inner_dim = key.shape
    expected_dim_blocks = (head_dim + PA_NZ_DATA_BLOCK - 1) // PA_NZ_DATA_BLOCK
    groups64 = (head_dim + MXFP8_GROUP_ELEMENTS - 1) // MXFP8_GROUP_ELEMENTS
    vscale_dim_blocks = (
        head_dim + PA_NZ_VSCALE_DIM_BLOCK - 1
    ) // PA_NZ_VSCALE_DIM_BLOCK
    if inner_dim != PA_NZ_DATA_BLOCK or dim_blocks != expected_dim_blocks:
        raise ValueError(
            f"PA_NZ D1/D0 must be ({expected_dim_blocks}, {PA_NZ_DATA_BLOCK}), got ({dim_blocks}, {inner_dim})."
        )
    if page_size != block_size:
        raise ValueError(
            f"PA_NZ block size mismatch: tensor has {page_size}, configured {block_size}."
        )
    expected_k_scale = (
        pages,
        heads,
        block_size // PA_NZ_TOKEN_BLOCK,
        groups64,
        PA_NZ_TOKEN_BLOCK,
        MXFP8_SCALE_PAIR_WIDTH,
    )
    expected_v_scale = (
        pages,
        heads,
        vscale_dim_blocks,
        block_size // MXFP8_GROUP_ELEMENTS,
        PA_NZ_VSCALE_DIM_BLOCK,
        MXFP8_SCALE_PAIR_WIDTH,
    )
    _check_shape("k_scale", k_scale, expected_k_scale)
    _check_shape("v_scale", v_scale, expected_v_scale)


# Layer 3: test fixture configuration and PA stride contracts.
def check_test_case(case):
    """Reject inconsistent CPU fixture configuration before allocating tensors."""
    for field in ("B", "N_q", "N_kv"):
        _check_integer(field, case[field], minimum=1)
    for field in ("max_seqlen_q", "max_seqlen_kv"):
        _check_integer(field, case[field], minimum=0)
    for field in ("enable_pa", "enable_lse", "reverse_pages", "noncontiguous_pa"):
        if field in case:
            _check_boolean(field, case[field])
    batch, query_heads, kv_heads = case["B"], case["N_q"], case["N_kv"]
    query_lengths, kv_lengths = list(case["S_q"]), list(case["S_kv"])
    max_q, max_kv = case["max_seqlen_q"], case["max_seqlen_kv"]
    check_q_layout(case["q_layout"])
    if len(query_lengths) != batch or len(kv_lengths) != batch:
        raise ValueError("S_q and S_kv must each contain B sequence lengths.")
    if query_heads % kv_heads != 0:
        raise ValueError("N_q must be divisible by positive N_kv.")
    for field, lengths in (("S_q", query_lengths), ("S_kv", kv_lengths)):
        for index, length in enumerate(lengths):
            _check_integer(f"{field}[{index}]", length, minimum=0)
    if max(query_lengths) != max_q or max(kv_lengths) != max_kv:
        raise ValueError("Fields max_seqlen_q/kv must match the maximum S_q/S_kv.")
    if not any(kv_lengths):
        raise ValueError("At least one batch must have a non-empty KV sequence.")
    if case["q_scale_layout"] not in ("TND", "N2TGD"):
        raise ValueError("Field q_scale_layout must be TND or N2TGD.")
    if case["q_scale_layout"] == "N2TGD" and max_q > DECODE_QUERY_ROWS:
        raise ValueError(f"Decode max_seqlen_q must not exceed {DECODE_QUERY_ROWS}.")
    check_mask_mode(case["mask_mode"])
    check_kv_layout(case["enable_pa"], case["kv_cache_layout"])
    if case["enable_pa"]:
        check_paged_kv_options(case["block_size"], case["kv_cache_layout"])
    check_head_dim_layout(case["D"], case["kv_cache_layout"])
    pa_stride_for_case(case)


def pa_stride_for_case(case):
    if not case.get("noncontiguous_pa", False):
        return PAStride.CONTIGUOUS
    if not case.get("enable_pa", False):
        raise ValueError("Option noncontiguous_pa=True requires a PA layout.")
    layout = case["kv_cache_layout"]
    if layout == KVLayout.PA_BBND:
        return PAStride.AXIS0
    if layout in (KVLayout.PA_BNBD, KVLayout.PA_NZ):
        return PAStride.AXIS01
    raise ValueError(f"Unsupported PA layout for non-contiguous input: {layout}.")


def noncontiguous_axes(tensor):
    """Retain singleton axes: PA head-axis stride gaps are still physical."""
    shape = tuple(int(dim) for dim in tensor.shape)
    expected = [0] * len(shape)
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        expected[axis] = stride
        stride *= shape[axis]
    return tuple(
        axis
        for axis, (got, want) in enumerate(zip(tensor.stride(), expected))
        if got != want
    )


def _stride_axes(mode):
    if not isinstance(mode, str):
        raise TypeError(f"PA stride mode must be str, got {type(mode).__name__}.")
    if mode not in PAStride.ALL:
        raise ValueError(f"Unsupported PA stride mode: {mode}.")
    return {PAStride.CONTIGUOUS: (), PAStride.AXIS0: (0,), PAStride.AXIS01: (0, 1)}[
        mode
    ]


def check_pa_stride_mode(tensor, mode):
    _check_tensor("PA tensor", tensor)
    _stride_axes(mode)
    if mode != PAStride.CONTIGUOUS and tensor.ndim < 2:
        raise ValueError("PA strided tensors must have at least two dimensions.")


def check_pa_strided_view(source, view, mode):
    expected_axes = _stride_axes(mode)
    _check_shape("PA view", view, source.shape)
    _check_dtype("PA view", view, source.dtype)
    if noncontiguous_axes(view) != expected_axes:
        raise ValueError(f"PA {mode} view did not preserve the required stride axes.")


def check_pa_device_strides(host_tensors, device_tensors, mode):
    """Check shape, dtype and exact PA strides for all four transferred inputs."""
    expected_axes = _stride_axes(mode)
    host_tensors, device_tensors = tuple(host_tensors), tuple(device_tensors)
    if len(host_tensors) != len(_PA_TENSOR_NAMES) or len(device_tensors) != len(
        _PA_TENSOR_NAMES
    ):
        raise ValueError(
            f"PA stride checks require {len(_PA_TENSOR_NAMES)} host and device tensors: {_PA_TENSOR_NAMES}."
        )
    for name, host, device in zip(_PA_TENSOR_NAMES, host_tensors, device_tensors):
        _check_tensor(name, host)
        _check_tensor(name, device)
        _check_shape(name, device, host.shape)
        _check_dtype(name, device, host.dtype)
        if tuple(device.stride()) != tuple(host.stride()):
            raise ValueError(
                f"NPU transfer changed {name} stride: host={host.stride()}, npu={device.stride()}."
            )
        if noncontiguous_axes(device) != expected_axes:
            raise ValueError(
                f"PA stride mode {mode} expected for {name}, got stride={device.stride()}."
            )


def check_v_quantization_extent(padded_tokens, tile_n):
    """Require whole 64-token scale pairs when V scales are packed into tiles."""
    _check_integer("padded_tokens", padded_tokens, minimum=0)
    _check_integer("tile_n", tile_n, minimum=1)
    if tile_n % MXFP8_GROUP_ELEMENTS != 0 or padded_tokens % tile_n != 0:
        raise ValueError(
            f"V tile_n must be a multiple of {MXFP8_GROUP_ELEMENTS} and padded_tokens "
            f"a multiple of tile_n, got {padded_tokens}, {tile_n}."
        )


def check_paged_kv_options(block_size, kv_layout):
    _check_integer("block_size", block_size, minimum=1)
    if block_size not in SUPPORTED_PA_BLOCK_SIZES:
        raise ValueError(
            f"Argument block_size must be in {SUPPORTED_PA_BLOCK_SIZES}, got {block_size}."
        )
    if not isinstance(kv_layout, str):
        raise TypeError(
            f"Argument kv_layout must be str, got {type(kv_layout).__name__}."
        )
    if kv_layout not in KVLayout.PAGED:
        raise ValueError(f"Unsupported PA layout: {kv_layout}.")


def check_paged_kv_contract(
    k_paged,
    v_paged,
    sk_paged,
    sv_paged,
    block_table,
    k_data,
    v_data,
    k_scale_flat,
    v_scale_tiles,
    kv_lengths,
    block_size,
):
    """Verify CPU fixture page IDs and gather the original logical KV prefixes."""
    _check_integer("block_size", block_size, minimum=1)
    _check_tensor("block_table", block_table)
    _check_dtype("block_table", block_table, torch.int32)
    if block_table.ndim != 2 or block_table.shape[0] != len(kv_lengths):
        raise ValueError("Fixture block_table must have one row per KV sequence.")
    pages = k_paged.shape[0]
    if any(tensor.shape[0] != pages for tensor in (v_paged, sk_paged, sv_paged)):
        raise ValueError("Paged K/V and scale tensors must have the same page count.")
    heads = k_data.shape[1]
    head_dim = k_data.shape[-1]
    groups = (head_dim + MXFP8_GROUP_ELEMENTS - 1) // MXFP8_GROUP_ELEMENTS
    for batch, length in enumerate(kv_lengths):
        _check_integer(f"kv_lengths[{batch}]", length, minimum=0)
        required_pages = (length + block_size - 1) // block_size
        if required_pages > block_table.shape[1]:
            raise ValueError(
                f"Fixture block_table row {batch} needs {required_pages} page IDs."
            )
        sv_groups = (
            v_scale_tiles[batch]
            .reshape(heads, -1, head_dim, MXFP8_SCALE_PAIR_WIDTH)
            .view(torch.uint8)
            if v_scale_tiles is not None
            else None
        )
        for logical_page in range(required_pages):
            physical_page = int(block_table[batch, logical_page])
            if not 0 <= physical_page < pages:
                raise ValueError(
                    f"Fixture page ID {physical_page} is outside [0, {pages})."
                )
            start = logical_page * block_size
            end = min(start + block_size, length)
            count = end - start
            if not torch.equal(
                k_paged[physical_page, :, :count, :], k_data[batch, :, start:end, :]
            ):
                raise AssertionError("K PA pack/unpack contract failed.")
            if not torch.equal(
                v_paged[physical_page, :, :count, :], v_data[batch, :, start:end, :]
            ):
                raise AssertionError("V PA pack/unpack contract failed.")
            if k_scale_flat is not None:
                expected_sk = (
                    k_scale_flat[batch, :, start:end, :]
                    .view(torch.uint8)
                    .reshape(heads, count, groups, MXFP8_SCALE_PAIR_WIDTH)
                )
                if not torch.equal(
                    sk_paged[physical_page, :, :count, :, :].view(torch.uint8),
                    expected_sk,
                ):
                    raise AssertionError("KScale PA pack/unpack contract failed.")
            if sv_groups is not None:
                group_start = start // MXFP8_GROUP_ELEMENTS
                group_count = (count + MXFP8_GROUP_ELEMENTS - 1) // MXFP8_GROUP_ELEMENTS
                group_end = group_start + group_count
                if not torch.equal(
                    sv_paged[physical_page, :, :group_count, :, :].view(torch.uint8),
                    sv_groups[:, group_start:group_end],
                ):
                    raise AssertionError("VScale PA pack/unpack contract failed.")


def _validate_length_vectors(q, kv):
    if not isinstance(q, torch.Tensor) or not isinstance(kv, torch.Tensor):
        raise TypeError("Metadata requires Tensor length vectors.")
    if q.dtype != torch.int64 or kv.dtype != torch.int64:
        raise ValueError("Metadata requires int64 length vectors.")
    if q.device != kv.device or q.ndim != 1 or kv.shape != q.shape:
        raise ValueError(
            "Metadata length vectors must have identical shape and device."
        )
    if not q.is_contiguous() or not kv.is_contiguous():
        raise ValueError("Metadata length vectors must be contiguous.")
    if not q.numel():
        raise ValueError("Metadata batch must be positive.")
    if q.device.type not in ("meta", "npu"):
        raise ValueError("AICPU metadata requires NPU length vectors.")


def _check_output_token_count(output_token_count, capacity):
    """Check the Host count without synchronizing the device length vector."""
    _check_integer("output_token_count", output_token_count, minimum=0)
    if output_token_count > capacity:
        raise ValueError(
            "output_token_count must equal sum(seqused_q) within Q capacity"
        )
