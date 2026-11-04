# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""QSA Indexer 的输入契约与 Tiling 规格校验。"""

from dataclasses import dataclass
from typing import Callable

import torch

try:
    from cannbotdsl import get_platform_info
except ImportError:
    get_platform_info = None


MAX_BATCH_SIZE = 65536
INT32_MAX = 2**31 - 1
UINT32_MAX = 2**32 - 1
MAX_VECTOR_CORE_MULTIPLIER = 2


@dataclass(frozen=True)
class InputInfo:
    """通过 Tensor 契约校验后可供 Tiling 使用的静态输入信息。"""

    device: torch.device
    total_query: int
    batch_size: int
    compressed_page_count: int
    max_pages: int
    metadata_numel: int


@dataclass(frozen=True)
class TilingInfo:
    """主算子 Launcher 使用的 Host Tiling 结果。"""

    total_query: int
    batch_size: int
    max_pages: int
    block_dim: int
    score_groups: int


def _require_tensor(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input {name} must be a torch.Tensor")


def validate_inputs(
    q,
    compressed_k,
    block_table,
    actual_seq,
    query_positions,
    metadata,
    *,
    q_heads,
    kv_heads,
    head_dim,
    page_size,
) -> InputInfo:
    """校验 Tensor 自身属性以及输入之间的基本一致性。"""
    for name, tensor in (
        ("q", q),
        ("compressed_k", compressed_k),
        ("block_table", block_table),
        ("actual_seq", actual_seq),
        ("query_positions", query_positions),
    ):
        _require_tensor(name, tensor)

    if metadata is None:
        raise ValueError(
            "Input metadata is required and must be produced by qsa_indexer_metadata"
        )
    _require_tensor("metadata", metadata)

    if (
        q.dtype != torch.bfloat16
        or q.ndim != 3
        or tuple(q.shape[1:])
        != (
            q_heads,
            head_dim,
        )
    ):
        raise ValueError(f"Input q must be BF16 [T,{q_heads},{head_dim}]")
    if (
        compressed_k.dtype != torch.bfloat16
        or compressed_k.ndim != 4
        or tuple(compressed_k.shape[1:]) != (page_size, kv_heads, head_dim)
    ):
        raise ValueError(
            f"Input compressed_k must be BF16 [P,{page_size},{kv_heads},{head_dim}]"
        )
    if block_table.dtype != torch.int32 or block_table.ndim != 2:
        raise ValueError("Input block_table must be INT32 [B,max_pages]")

    batch_size, max_pages = map(int, block_table.shape)
    for name, value, size in (
        ("actual_seq", actual_seq, batch_size + 1),
        ("query_positions", query_positions, q.shape[0]),
    ):
        if value.dtype != torch.int32 or value.ndim != 1 or value.numel() != size:
            raise ValueError(f"Input {name} must be a matching INT32 vector")

    if metadata.dtype != torch.int32 or metadata.ndim != 1:
        raise ValueError("Input metadata must be a matching INT32 vector")

    return InputInfo(
        device=q.device,
        total_query=int(q.shape[0]),
        batch_size=batch_size,
        compressed_page_count=int(compressed_k.shape[0]),
        max_pages=max_pages,
        metadata_numel=metadata.numel(),
    )


def get_effective_core_counts(stream=None, *, aic_capacity=36):
    """获取当前设备及执行流可用的 Matmul/Vector 核数。"""
    props = torch.npu.get_device_properties(torch.npu.current_device())
    matmul = int(props.cube_core_num)
    vector = int(getattr(props, "vector_core_num", MAX_VECTOR_CORE_MULTIPLIER * matmul))
    if get_platform_info is not None:
        info = get_platform_info(stream=stream)
        matmul = int(info.cube_core_num) or matmul
        vector = int(info.vector_core_num) or vector
    if (
        not 1 <= matmul <= aic_capacity
        or not matmul <= vector <= MAX_VECTOR_CORE_MULTIPLIER * aic_capacity
    ):
        raise RuntimeError(f"Unsupported effective core counts: {matmul}/{vector}")
    return matmul, vector


def _resolve_block_dim(device, block_dim, *, aic_capacity):
    with torch.npu.device(device):
        available, _ = get_effective_core_counts(
            torch.npu.current_stream(device), aic_capacity=aic_capacity
        )
    if block_dim is None:
        return available
    if (
        isinstance(block_dim, bool)
        or not isinstance(block_dim, int)
        or not 1 <= block_dim <= available
    ):
        raise ValueError(f"Input block_dim must be in [1,{available}]")
    return block_dim


def validate_tiling(
    inputs: InputInfo,
    block_dim,
    *,
    metadata_capacity: Callable[[int], int],
    aic_capacity,
    page_size,
    compressed_units_per_topk_chunk,
    blocks_per_topk_chunk,
) -> TilingInfo:
    """校验 Kernel 支持范围，并生成 Launcher 使用的 Tiling 信息。"""
    if not 1 <= inputs.batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"Input batch size must be in [1, {MAX_BATCH_SIZE}]")
    if inputs.total_query > INT32_MAX:
        raise ValueError("Input q length must fit INT32 sequence offsets")
    if inputs.max_pages > UINT32_MAX:
        raise ValueError("Input block_table page capacity must fit UINT32")
    if inputs.compressed_page_count > UINT32_MAX:
        raise ValueError("Input compressed_k page count must fit UINT32")
    if inputs.metadata_numel != metadata_capacity(inputs.batch_size):
        raise ValueError("Input metadata capacity does not match batch size")

    resolved_block_dim = _resolve_block_dim(
        inputs.device, block_dim, aic_capacity=aic_capacity
    )
    compressed_capacity = inputs.max_pages * page_size
    topk_chunk_count = max(
        1,
        (compressed_capacity + compressed_units_per_topk_chunk - 1)
        // compressed_units_per_topk_chunk,
    )
    return TilingInfo(
        total_query=inputs.total_query,
        batch_size=inputs.batch_size,
        max_pages=inputs.max_pages,
        block_dim=resolved_block_dim,
        score_groups=topk_chunk_count * blocks_per_topk_chunk,
    )


def validate_and_resolve(
    q,
    compressed_k,
    block_table,
    actual_seq,
    query_positions,
    metadata,
    block_dim,
    *,
    q_heads,
    kv_heads,
    head_dim,
    page_size,
    metadata_capacity,
    aic_capacity,
    compressed_units_per_topk_chunk,
    blocks_per_topk_chunk,
) -> TilingInfo:
    """依次完成输入校验和 Tiling 校验。"""
    inputs = validate_inputs(
        q,
        compressed_k,
        block_table,
        actual_seq,
        query_positions,
        metadata,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
    )
    return validate_tiling(
        inputs,
        block_dim,
        metadata_capacity=metadata_capacity,
        aic_capacity=aic_capacity,
        page_size=page_size,
        compressed_units_per_topk_chunk=compressed_units_per_topk_chunk,
        blocks_per_topk_chunk=blocks_per_topk_chunk,
    )
