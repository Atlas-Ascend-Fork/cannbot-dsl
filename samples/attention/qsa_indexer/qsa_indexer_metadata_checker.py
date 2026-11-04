# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""QSA Indexer Metadata 的输入契约与 Tiling 规格校验。"""

from dataclasses import dataclass
from typing import Callable

import torch

try:
    from cannbotdsl import get_platform_info
except ImportError:
    get_platform_info = None


MAX_BATCH_SIZE = 65536
UINT32_CAPACITY = 2**32
MAX_VECTOR_CORE_MULTIPLIER = 2


@dataclass(frozen=True)
class MetadataInputInfo:
    """通过 Tensor 契约校验后可供 Tiling 使用的静态输入信息。"""

    device_id: int
    device: torch.device
    batch_size: int
    total_query: int
    max_pages: int
    compressed_page_count: int


@dataclass(frozen=True)
class MetadataTilingInfo:
    """Metadata AICPU Launcher 使用的 Host Tiling 结果。"""

    device_id: int
    device: torch.device
    stream: object
    batch_size: int
    total_query: int
    max_pages: int
    compressed_page_count: int
    block_dim: int
    output_size: int


def _require_int32_tensor(name, tensor, ndim):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input {name} must be a torch.Tensor")
    if tensor.ndim != ndim or tensor.dtype != torch.int32:
        raise ValueError(f"Input {name} must be a {ndim}D INT32 tensor")


def validate_inputs(
    actual_seq,
    query_positions,
    block_table,
    compressed_page_count,
) -> MetadataInputInfo:
    """校验 Metadata 输入 Tensor 及 Host 标量参数。"""
    device_id = torch.npu.current_device()
    device = torch.device("npu", device_id)
    for name, tensor, ndim in (
        ("actual_seq", actual_seq, 1),
        ("query_positions", query_positions, 1),
        ("block_table", block_table, 2),
    ):
        _require_int32_tensor(name, tensor, ndim)

    batch_size, max_pages = map(int, block_table.shape)
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"Input batch size must be in [1, {MAX_BATCH_SIZE}]")
    if actual_seq.numel() != batch_size + 1:
        raise ValueError(
            "Input actual_seq and block_table batch dimensions are inconsistent"
        )
    total_query = int(query_positions.numel())
    if total_query >= UINT32_CAPACITY:
        raise ValueError("Input query_positions capacity exceeds private ABI range")
    valid_page_count_type = isinstance(compressed_page_count, int) and not isinstance(
        compressed_page_count, bool
    )
    if not valid_page_count_type or not 0 <= compressed_page_count < UINT32_CAPACITY:
        raise ValueError("Input compressed_page_count must fit nonnegative uint32")

    return MetadataInputInfo(
        device_id=device_id,
        device=device,
        batch_size=batch_size,
        total_query=total_query,
        max_pages=max_pages,
        compressed_page_count=compressed_page_count,
    )


def get_effective_core_counts(stream=None, *, aic_capacity=36):
    """获取当前设备及执行流可用的 Matmul/Vector 核数。"""
    props = torch.npu.get_device_properties(torch.npu.current_device())
    cube = int(props.cube_core_num)
    vector = int(getattr(props, "vector_core_num", MAX_VECTOR_CORE_MULTIPLIER * cube))
    if get_platform_info is not None:
        info = get_platform_info(stream=stream)
        cube = int(info.cube_core_num) or cube
        vector = int(info.vector_core_num) or vector
    if (
        not 1 <= cube <= aic_capacity
        or not cube <= vector <= MAX_VECTOR_CORE_MULTIPLIER * aic_capacity
    ):
        raise RuntimeError(f"Unsupported effective core counts: {cube}/{vector}")
    return cube, vector


def _resolve_block_dim(stream, block_dim, *, aic_capacity):
    available, _ = get_effective_core_counts(stream, aic_capacity=aic_capacity)
    if block_dim is None:
        return available
    if (
        isinstance(block_dim, bool)
        or not isinstance(block_dim, int)
        or not 1 <= block_dim <= available
    ):
        raise ValueError(f"Input block_dim must be in [1, {available}]")
    return block_dim


def validate_tiling(
    inputs: MetadataInputInfo,
    block_dim,
    *,
    metadata_capacity: Callable[[int], int],
    aic_capacity: int,
) -> MetadataTilingInfo:
    """解析核数和 Metadata 输出容量。"""
    stream = torch.npu.current_stream(inputs.device_id)
    resolved_block_dim = _resolve_block_dim(
        stream,
        block_dim,
        aic_capacity=aic_capacity,
    )
    return MetadataTilingInfo(
        device_id=inputs.device_id,
        device=inputs.device,
        stream=stream,
        batch_size=inputs.batch_size,
        total_query=inputs.total_query,
        max_pages=inputs.max_pages,
        compressed_page_count=inputs.compressed_page_count,
        block_dim=resolved_block_dim,
        output_size=metadata_capacity(inputs.batch_size),
    )


def validate_and_resolve(
    actual_seq,
    query_positions,
    block_table,
    compressed_page_count,
    block_dim,
    *,
    metadata_capacity,
    aic_capacity,
) -> MetadataTilingInfo:
    """依次完成 Metadata 输入校验和 Tiling 校验。"""
    inputs = validate_inputs(
        actual_seq,
        query_positions,
        block_table,
        compressed_page_count,
    )
    return validate_tiling(
        inputs,
        block_dim,
        metadata_capacity=metadata_capacity,
        aic_capacity=aic_capacity,
    )
