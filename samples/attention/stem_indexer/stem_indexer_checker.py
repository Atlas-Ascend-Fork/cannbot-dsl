# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stem Indexer 的输入契约与 Tiling 规格校验。"""

from dataclasses import dataclass
import math
from typing import Any

import torch
from cannbotdsl import get_platform_info

MAX_BATCH_SIZE = 65536
MAX_CUBE_CORE_COUNT = 36
REQUIRED_HEAD_DIM = 2048
SUPPORTED_Q_HEADS = (32, 64)
SUPPORTED_KV_HEADS = (2, 4, 8)
METADATA_RECORDS_PER_BN_HEAD = 108
METADATA_RECORD_WIDTH = 16
METADATA_ALIGNMENT_ELEMENTS = 4096


@dataclass(frozen=True)
class StemIndexerTilingInfo:
    """Stem Indexer Launcher 和 Host 输入准备使用的 Tiling 结果。"""

    batch_size: int
    q_heads: int
    kv_heads: int
    q_blocks_max: int
    kv_blocks_max: int
    q_blocks_padded: int
    kv_blocks_padded: int
    head_dim: int
    block_dim: int
    target_device: torch.device
    attrs: Any


def _ceil_div(value: int, divisor: int) -> int:
    return value // divisor + (value % divisor != 0)


def _align_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


def _check_tensor(name, tensor, shape, dtype):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input {name} must be a torch.Tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"Input {name}.shape must be {shape}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != dtype:
        raise ValueError(f"Input {name}.dtype must be {dtype}, got {tensor.dtype}")


def _validate_attributes(attrs, attributes_type):
    if not isinstance(attrs, attributes_type):
        raise TypeError("Attrs must be FixedAttributes")
    for name, expected in (
        ("stem_block_size", 128),
        ("stem_stride", 16),
        ("initial_blocks", 4),
        ("window_size", 4),
    ):
        value = getattr(attrs, name)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(f"Input {name} must be {expected}")
    if not isinstance(attrs.causal, bool):
        raise TypeError("Causal must be bool")
    if isinstance(attrs.alpha, bool) or not isinstance(attrs.alpha, (int, float)):
        raise TypeError("Alpha must be a finite number in (0,1]")
    if not math.isfinite(attrs.alpha) or not 0.0 < attrs.alpha <= 1.0:
        raise ValueError("Alpha must be finite and in (0,1]")
    if isinstance(
        attrs.topk_score_precision, bool
    ) or attrs.topk_score_precision not in (1, 2):
        raise ValueError("Topk_score_precision must be 1 or 2")


def get_effective_core_counts(stream=None):
    """返回当前主算子发射流可用的 Cube 和 Vector 核数。"""
    props = torch.npu.get_device_properties(torch.npu.current_device())
    cube = int(getattr(props, "cube_core_num", 0))
    vector = int(getattr(props, "vector_core_num", 0)) or 2 * cube
    info = get_platform_info(stream=stream)
    cube = int(info.cube_core_num) or cube
    vector = int(info.vector_core_num) or vector
    if cube <= 0 or vector < cube:
        raise RuntimeError(
            f"Invalid effective NPU core counts: AIC={cube}, AIV={vector}"
        )
    return cube, vector


def _resolve_block_dim(target_device, block_dim):
    with torch.npu.device(target_device):
        available_cores, _ = get_effective_core_counts(
            stream=torch.npu.current_stream(target_device)
        )
    if block_dim is None:
        return available_cores
    if (
        isinstance(block_dim, bool)
        or not isinstance(block_dim, int)
        or not 1 <= block_dim <= min(MAX_CUBE_CORE_COUNT, available_cores)
    ):
        raise ValueError(
            f"Block_dim must be in [1,{min(MAX_CUBE_CORE_COUNT, available_cores)}]"
        )
    return block_dim


def _metadata_capacity(batch_size, kv_heads):
    raw = (
        1 + batch_size * kv_heads * METADATA_RECORDS_PER_BN_HEAD
    ) * METADATA_RECORD_WIDTH
    return _align_up(raw, METADATA_ALIGNMENT_ELEMENTS)


def validate_and_resolve(
    qflat,
    kflat,
    vbias,
    q_seq_lens,
    kv_seq_lens,
    num_prompt_tokens,
    metadata,
    block_dim,
    *,
    attrs,
    attributes_type,
    m_tile,
    n_tile,
) -> StemIndexerTilingInfo:
    """完成输入校验、核数解析和 Launcher Tiling 计算。"""
    _validate_attributes(attrs, attributes_type)
    for name, tensor in (
        ("qflat", qflat),
        ("kflat", kflat),
        ("vbias", vbias),
        ("q_seq_lens", q_seq_lens),
        ("kv_seq_lens", kv_seq_lens),
        ("num_prompt_tokens", num_prompt_tokens),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Input {name} must be a torch.Tensor")
    if qflat.ndim != 4 or kflat.ndim != 4 or vbias.ndim != 3:
        raise ValueError("Qflat/kflat/vbias must be 4D/4D/3D BNSD tensors")

    batch_size, q_heads, q_blocks_max, head_dim = map(int, qflat.shape)
    key_batch, kv_heads, kv_blocks_max, key_head_dim = map(int, kflat.shape)
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"Batch size must be in [1,{MAX_BATCH_SIZE}]")
    if q_blocks_max <= 0 or kv_blocks_max <= 0:
        raise ValueError("Q/K block capacities must be positive")
    if batch_size != key_batch or tuple(vbias.shape) != (
        batch_size,
        kv_heads,
        kv_blocks_max,
    ):
        raise ValueError(
            "Qflat, kflat and vbias batch/head/block dimensions are inconsistent"
        )
    if head_dim != key_head_dim:
        raise ValueError("Qflat and kflat head dimensions must match")
    if head_dim != REQUIRED_HEAD_DIM:
        raise ValueError(
            f"The current Cube launcher requires head_dim={REQUIRED_HEAD_DIM}"
        )
    if (
        q_heads not in SUPPORTED_Q_HEADS
        or kv_heads not in SUPPORTED_KV_HEADS
        or q_heads % kv_heads
    ):
        raise ValueError(
            "Q_heads must be 32/64, kv_heads must be 2/4/8, and divide q_heads"
        )
    if qflat.dtype != torch.bfloat16 or kflat.dtype != torch.bfloat16:
        raise ValueError("Qflat and kflat must use torch.bfloat16")
    if vbias.dtype != torch.float32:
        raise ValueError("Vbias must use torch.float32")

    target_device = torch.device("npu", torch.npu.current_device())
    resolved_block_dim = _resolve_block_dim(target_device, block_dim)

    for name, tensor in (
        ("q_seq_lens", q_seq_lens),
        ("kv_seq_lens", kv_seq_lens),
        ("num_prompt_tokens", num_prompt_tokens),
    ):
        _check_tensor(name, tensor, (batch_size,), torch.int32)

    if metadata is None:
        raise ValueError(
            "Metadata is required and must be produced by stem_indexer_metadata"
        )
    capacity = _metadata_capacity(batch_size, kv_heads)
    metadata_error = "Metadata must be a contiguous int32 tensor of matching capacity"
    if not isinstance(metadata, torch.Tensor):
        raise ValueError(metadata_error)
    if metadata.dtype != torch.int32 or metadata.ndim != 1:
        raise ValueError(metadata_error)
    if metadata.numel() != capacity or not metadata.is_contiguous():
        raise ValueError(metadata_error)

    # 按输入容量预留一个M64尾块，不读取设备上的实际序列长度。
    q_padded = max(
        m_tile,
        _align_up(q_blocks_max + m_tile - 1, m_tile),
    )
    kv_padded = max(n_tile, _align_up(kv_blocks_max, n_tile))

    return StemIndexerTilingInfo(
        batch_size=batch_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        q_blocks_max=q_blocks_max,
        kv_blocks_max=kv_blocks_max,
        q_blocks_padded=q_padded,
        kv_blocks_padded=kv_padded,
        head_dim=head_dim,
        block_dim=resolved_block_dim,
        target_device=target_device,
        attrs=attrs,
    )
