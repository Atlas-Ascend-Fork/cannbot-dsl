# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host-side contract checks for the native HIF8 FlashAttention sample.

The checks are pure PyTorch so they run, and are unit-tested, without an NPU.
Device-resident length vectors are validated after the caller has read their
values, so a steady-state call does not pay a second D2H copy here.
"""

import math

import torch

LAYOUTS = ("BNSD", "BSND", "TND")
MASK_NONE = 0
MASK_CAUSAL = 3
MASK_MODES = (MASK_NONE, MASK_CAUSAL)
HEAD_DIM = 128


def _check(condition, error, message):
    if not condition:
        raise error(message)


def _check_length_vector(name, tensor, batch):
    _check(tensor is not None, ValueError, f"{name} is required for this layout")
    _check(
        isinstance(tensor, torch.Tensor), TypeError, f"{name} must be a torch.Tensor"
    )
    _check(
        tensor.dtype in (torch.int32, torch.int64),
        TypeError,
        f"{name} must use int32 or int64 storage, got {tensor.dtype}",
    )
    _check(
        tensor.dim() == 1, ValueError, f"{name} must be 1-D, got rank {tensor.dim()}"
    )
    _check(
        tensor.numel() == batch,
        ValueError,
        f"{name} must hold one length per batch ({batch}), got {tensor.numel()}",
    )


def validate_static(query, key, value, mask_mode, layout, seqused_kv, seqused_q):
    """Check dtypes, layout, mask mode, head geometry and device placement.

    Returns the geometry the wrapper needs; per-batch lengths are validated
    separately once their values have been read.
    """
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        _check(
            isinstance(tensor, torch.Tensor),
            TypeError,
            f"{name} must be a torch.Tensor",
        )
        _check(
            tensor.dtype == torch.int8,
            TypeError,
            f"HIF8 {name} must use int8 storage, got {tensor.dtype}",
        )
        _check(
            tensor.shape[-1] == HEAD_DIM,
            ValueError,
            f"HIF8 requires D={HEAD_DIM} on every operand, got {tensor.shape[-1]}",
        )
    _check(layout in LAYOUTS, ValueError, f"unsupported layout {layout!r}")
    _check(
        mask_mode in MASK_MODES,
        ValueError,
        f"mask_mode must be 0 (none) or 3 (causal), got {mask_mode!r}",
    )
    _check(
        key.shape == value.shape,
        ValueError,
        f"key/value shapes differ: {tuple(key.shape)} vs {tuple(value.shape)}",
    )
    devices = {query.device, key.device, value.device}
    _check(
        len(devices) == 1,
        ValueError,
        "Q/K/V must live on one device, got " + ", ".join(sorted(map(str, devices))),
    )

    if layout == "BSND":
        _check(
            query.dim() == 4 and key.dim() == 4,
            ValueError,
            "BSND expects rank-4 (B, S, N, D) tensors",
        )
        batch, q_extent, n_q_heads, _ = query.shape
        kv_extent, kv_heads = key.shape[1], key.shape[2]
        _check(
            query.shape[0] == key.shape[0],
            ValueError,
            f"query/key batch mismatch: {query.shape[0]} vs {key.shape[0]}",
        )
    elif layout == "BNSD":
        _check(
            query.dim() == 4 and key.dim() == 4,
            ValueError,
            "BNSD expects rank-4 (B, N, S, D) tensors",
        )
        batch, n_q_heads, q_extent, _ = query.shape
        kv_heads, kv_extent = key.shape[1], key.shape[2]
        _check(
            query.shape[0] == key.shape[0],
            ValueError,
            f"query/key batch mismatch: {query.shape[0]} vs {key.shape[0]}",
        )
    else:  # TND packs every sequence into one token-major (T, N, D) stream.
        _check(
            query.dim() == 3 and key.dim() == 3,
            ValueError,
            "TND expects rank-3 (T, N, D) tensors",
        )
        n_q_heads = query.shape[1]
        kv_heads = key.shape[1]
        q_extent, kv_extent = query.shape[0], key.shape[0]
        _check(
            seqused_kv is not None,
            ValueError,
            "TND requires seqused_kv to derive batch lengths",
        )
        _check_length_vector("seqused_kv", seqused_kv, int(seqused_kv.numel()))
        if seqused_q is not None:
            _check_length_vector("seqused_q", seqused_q, int(seqused_kv.numel()))
        batch = int(seqused_kv.numel())

    _check(
        kv_heads > 0 and n_q_heads > 0,
        ValueError,
        f"head counts must be positive, got q={n_q_heads}, kv={kv_heads}",
    )
    _check(
        n_q_heads % kv_heads == 0,
        ValueError,
        f"GQA requires Nq % Nkv == 0, got Nq={n_q_heads}, Nkv={kv_heads}",
    )
    if layout != "TND" and seqused_kv is not None:
        _check_length_vector("seqused_kv", seqused_kv, batch)

    return {
        "batch": batch,
        "n_q_heads": n_q_heads,
        "kv_heads": kv_heads,
        "q_extent": q_extent,
        "kv_extent": kv_extent,
    }


def validate_lengths(q_lens, kv_lens, q_extent, kv_extent, packed):
    """Check per-batch lengths against each other and the storage extents.

    ``packed`` is true for TND, where the lengths partition one token stream
    and only their sums are bounded; dense layouts address each batch from its
    own base, so every single length must fit the extent.
    """
    _check(
        len(q_lens) == len(kv_lens),
        ValueError,
        f"query/key length vectors differ: {len(q_lens)} vs {len(kv_lens)}",
    )
    _check(len(q_lens) > 0, ValueError, "length vectors must not be empty")
    for index, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        _check(
            int(q_len) >= 1,
            ValueError,
            f"q_lens[{index}]={q_len} must be at least 1",
        )
        _check(
            int(kv_len) >= 1,
            ValueError,
            f"kv_lens[{index}]={kv_len} must be at least 1",
        )
        if not packed:
            _check(
                int(q_len) <= q_extent,
                ValueError,
                f"q_lens[{index}]={q_len} exceeds the {q_extent}-token storage",
            )
            _check(
                int(kv_len) <= kv_extent,
                ValueError,
                f"kv_lens[{index}]={kv_len} exceeds the {kv_extent}-token storage",
            )
    if packed:
        _check(
            sum(int(v) for v in q_lens) <= q_extent,
            ValueError,
            f"query lengths sum to {sum(q_lens)}, beyond the {q_extent}-token storage",
        )
        _check(
            sum(int(v) for v in kv_lens) <= kv_extent,
            ValueError,
            f"KV lengths sum to {sum(kv_lens)}, beyond the {kv_extent}-token storage",
        )


def validate_scales(descale_q, descale_k, descale_v, p_scale, softmax_scale):
    """Check the per-tensor scales that fold into the softmax constant."""
    for name, value in (
        ("descale_q", descale_q),
        ("descale_k", descale_k),
        ("descale_v", descale_v),
    ):
        _check(
            math.isfinite(value),
            ValueError,
            f"{name} must be finite, got {value}",
        )
        _check(value > 0.0, ValueError, f"{name} must be positive, got {value}")
    _check(
        math.isfinite(p_scale),
        ValueError,
        f"p_scale must be finite, got {p_scale}",
    )
    _check(p_scale > 0.0, ValueError, f"p_scale must be positive, got {p_scale}")
    if softmax_scale is not None:
        _check(
            math.isfinite(softmax_scale),
            ValueError,
            f"softmax_scale must be finite, got {softmax_scale}",
        )
    # The kernel folds the three factors into one compile-time constant and
    # uses it to scale the masked-row sentinel. A non-positive product makes
    # the sentinel and every row maximum collapse to the same value, so valid
    # rows would be erased; reject it at the entry. Positive products are
    # supported (AscendC only requires softmax_scale > 0 as well); values big
    # enough to overflow the FP32 logits are beyond any implementation.
    resolved = 1.0 if softmax_scale is None else softmax_scale
    qk_scale = resolved * descale_q * descale_k
    _check(
        qk_scale > 0.0,
        ValueError,
        f"softmax_scale * descale_q * descale_k must be positive, got {qk_scale!r}",
    )
