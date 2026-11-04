# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stem Indexer Metadata 的输入校验与 Host 侧 tiling 参数解析。"""

from dataclasses import dataclass

import torch

try:
    from cannbotdsl import get_platform_info
except ImportError:
    get_platform_info = None

MAX_BATCH_SIZE = 65536
UINT32_CAPACITY = 2**32


@dataclass(frozen=True)
class MetadataInputInfo:
    """完成输入校验后供 tiling 阶段使用的参数。"""

    device_id: int
    device: torch.device
    batch_size: int
    q_heads: int
    kv_heads: int
    causal: bool
    stem_block_size: int
    window_size: int
    dim_qkflat: int


@dataclass(frozen=True)
class MetadataTilingInfo:
    """Metadata Kernel 发射所需的 Host 侧参数。"""

    device_id: int
    device: torch.device
    stream: object
    batch_size: int
    q_heads: int
    kv_heads: int
    causal: bool
    stem_block_size: int
    window_size: int
    dim_qkflat: int
    block_dim: int
    output_size: int


def _require_nonnegative_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Input {name} must be a nonnegative integer")


def _require_int32_vector(name, tensor):
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.ndim != 1
        or tensor.dtype != torch.int32
    ):
        raise ValueError(f"Input {name} must be a 1D int32 tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"Input {name} must be contiguous")


def validate_inputs(
    q_seq_lens,
    kv_seq_lens,
    q_heads,
    kv_heads,
    *,
    causal,
    stem_block_size,
    window_size,
    dim_qkflat,
):
    """校验 Tensor、属性及相互约束，不读取设备 Tensor 的内容。"""
    device_id = torch.npu.current_device()
    device = torch.device("npu", device_id)
    _require_int32_vector("q_seq_lens", q_seq_lens)
    _require_int32_vector("kv_seq_lens", kv_seq_lens)

    batch_size = q_seq_lens.numel()
    if not 1 <= batch_size <= MAX_BATCH_SIZE or kv_seq_lens.numel() != batch_size:
        raise ValueError(
            f"Sequence arrays must have equal batch size in [1, {MAX_BATCH_SIZE}]"
        )

    for name, value in (
        ("q_heads", q_heads),
        ("kv_heads", kv_heads),
        ("stem_block_size", stem_block_size),
        ("window_size", window_size),
        ("dim_qkflat", dim_qkflat),
    ):
        _require_nonnegative_integer(name, value)
    if kv_heads == 0 or q_heads == 0 or q_heads % kv_heads:
        raise ValueError("Q_heads must be a positive multiple of kv_heads")
    if stem_block_size == 0 or dim_qkflat == 0 or not isinstance(causal, bool):
        raise ValueError("Invalid block size, feature dimension or causal attribute")

    return MetadataInputInfo(
        device_id=device_id,
        device=device,
        batch_size=batch_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        causal=causal,
        stem_block_size=stem_block_size,
        window_size=window_size,
        dim_qkflat=dim_qkflat,
    )


def get_effective_core_counts(stream=None, *, aic_capacity=36, aiv_capacity=72):
    """查询当前设备/流可用核数，并校验是否超出 Metadata ABI 槽位容量。"""
    props = torch.npu.get_device_properties(torch.npu.current_device())
    cube = int(props.cube_core_num)
    vector = int(getattr(props, "vector_core_num", 2 * cube))
    if get_platform_info is not None:
        info = get_platform_info(stream=stream)
        cube = int(info.cube_core_num) or cube
        vector = int(info.vector_core_num) or vector
    if not 1 <= cube <= aic_capacity or not cube <= vector <= aiv_capacity:
        raise RuntimeError(f"Unsupported effective core counts: {cube}/{vector}")
    return cube, vector


def validate_tiling(
    inputs, block_dim, *, metadata_capacity, aic_capacity, aiv_capacity
):
    """解析动态核数和 Metadata 容量，生成可直接用于 Kernel 发射的参数。"""
    output_size = metadata_capacity(inputs.batch_size, inputs.kv_heads)
    if output_size >= UINT32_CAPACITY:
        raise ValueError("Metadata capacity exceeds private ABI range")

    stream = torch.npu.current_stream(inputs.device_id)
    cube_core_num, _ = get_effective_core_counts(
        stream, aic_capacity=aic_capacity, aiv_capacity=aiv_capacity
    )
    if block_dim is not None:
        if (
            isinstance(block_dim, bool)
            or not isinstance(block_dim, int)
            or not 1 <= block_dim <= cube_core_num
        ):
            raise ValueError(f"Block_dim must be in [1, {cube_core_num}]")
        cube_core_num = block_dim

    return MetadataTilingInfo(
        device_id=inputs.device_id,
        device=inputs.device,
        stream=stream,
        batch_size=inputs.batch_size,
        q_heads=inputs.q_heads,
        kv_heads=inputs.kv_heads,
        causal=inputs.causal,
        stem_block_size=inputs.stem_block_size,
        window_size=inputs.window_size,
        dim_qkflat=inputs.dim_qkflat,
        block_dim=cube_core_num,
        output_size=output_size,
    )


def validate_and_resolve(
    q_seq_lens,
    kv_seq_lens,
    q_heads,
    kv_heads,
    *,
    causal,
    stem_block_size,
    window_size,
    dim_qkflat,
    block_dim,
    metadata_capacity,
    aic_capacity,
    aiv_capacity,
):
    """依次执行输入校验与 tiling 校验。"""
    inputs = validate_inputs(
        q_seq_lens,
        kv_seq_lens,
        q_heads,
        kv_heads,
        causal=causal,
        stem_block_size=stem_block_size,
        window_size=window_size,
        dim_qkflat=dim_qkflat,
    )
    return validate_tiling(
        inputs,
        block_dim,
        metadata_capacity=metadata_capacity,
        aic_capacity=aic_capacity,
        aiv_capacity=aiv_capacity,
    )
