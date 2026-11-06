# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Metadata-only validation for query spans and two paged-KV valid lengths."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionSequenceConfig:
    tokens: int
    batches: int
    has_cu_q: bool
    has_used_q: bool
    has_used_win: bool
    has_used_cmp: bool
    paged: bool
    win_topk_per_query: bool
    cmp_topk_per_query: bool
    has_win_topk: bool
    has_cmp_topk: bool
    query_metadata_mode: str = "none"

    @property
    def extended_abi(self):
        return self.query_metadata_mode != "none"

    @property
    def signature(self):
        return tuple(self.__dict__.values())


def _tensor_version(value):
    try:
        return value._version
    except RuntimeError:  # inference-mode tensors do not track versions
        return None


@dataclass(frozen=True)
class AttentionSequenceMetadata:
    """Independent device snapshot of common Query metadata for one step.

    query_info[T,8]: batch, start, allocated length, used length, local offset,
    active and two reserved zeros. No layer-specific
    KV lengths, indices, page tables or slots are cached here.
    """

    query_info: object
    tokens: int
    batches: int
    source_versions: tuple = ()

    def validate(self, tokens, device):
        import torch

        value = self.query_info
        if (
            tokens != self.tokens
            or self.batches <= 0
            or value.dtype != torch.int32
            or value.device != device
            or tuple(value.shape) != (tokens, 8)
            or not value.is_contiguous()
        ):
            raise ValueError(
                "sequence_metadata must match T/device and contain contiguous int32 [T,8]"
            )
        for source, version in self.source_versions:
            if version is not None and _tensor_version(source) != version:
                raise ValueError(
                    "sequence_metadata is stale; regenerate after Query metadata changes"
                )


def allocate_sequence_metadata(config, device, sources):
    import torch

    value = torch.empty((config.tokens, 8), dtype=torch.int32, device=device)
    versions = tuple(
        (v, _tensor_version(v)) for v in (*sources, value) if v is not None
    )
    return AttentionSequenceMetadata(value, config.tokens, config.batches, versions)


def validate_attention_sequences(
    tokens,
    device,
    *,
    cu_seqlens_qo=None,
    seqused_q=None,
    cache_seqlens=None,
    extra_cache_seqlens=None,
    block_table=None,
    extra_block_table=None,
    topk_length=None,
    extra_topk_length=None,
    sequence_metadata=None,
    return_sequence_metadata=False,
):
    """Infer B without reading NPU tensor values or synchronizing the stream.

    Values must satisfy the documented prefix-sum/length contract. The device
    reads them on every launch, so in-place metadata changes remain visible.
    """
    import torch

    vectors = (cu_seqlens_qo, seqused_q, cache_seqlens, extra_cache_seqlens)
    if not isinstance(return_sequence_metadata, bool):
        raise ValueError("return_sequence_metadata must be bool")
    if sequence_metadata is not None:
        if not isinstance(sequence_metadata, AttentionSequenceMetadata):
            raise ValueError(
                "sequence_metadata must be an AttentionSequenceMetadata result"
            )
        sequence_metadata.validate(tokens, device)
        if cu_seqlens_qo is not None or seqused_q is not None:
            raise ValueError("sequence_metadata replaces cu_seqlens_qo and seqused_q")
    if (
        all(v is None for v in vectors)
        and sequence_metadata is None
        and not return_sequence_metadata
    ):
        return None
    batch_sizes = []
    if sequence_metadata is not None:
        batch_sizes.append(sequence_metadata.batches)
    for name, value in zip(
        ("cu_seqlens_qo", "seqused_q", "cache_seqlens", "extra_cache_seqlens"), vectors
    ):
        if value is None:
            continue
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.int32
            or value.device != device
            or value.ndim != 1
            or not value.is_contiguous()
            or value.numel() < (2 if name == "cu_seqlens_qo" else 1)
        ):
            raise ValueError(
                f"{name} must be a contiguous int32 vector on the input device"
            )
        batch_sizes.append(value.numel() - (1 if name == "cu_seqlens_qo" else 0))
    for table in (block_table, extra_block_table):
        if table is not None and table.ndim == 2:
            batch_sizes.append(int(table.shape[0]))
    batches = batch_sizes[0] if batch_sizes else 1
    if batches <= 0 or any(b != batches for b in batch_sizes):
        raise ValueError("sequence vectors and block tables must describe the same B")
    if cu_seqlens_qo is None and sequence_metadata is None and tokens % batches:
        raise ValueError(
            "unequal query spans require cu_seqlens_qo; otherwise T must be divisible by B"
        )
    paged = block_table is not None and extra_block_table is not None
    if (cache_seqlens is not None or extra_cache_seqlens is not None) and not paged:
        raise ValueError(
            "cache_seqlens/extra_cache_seqlens requires both block tables and request-local logical indices"
        )
    for name, value in (
        ("topk_length", topk_length),
        ("extra_topk_length", extra_topk_length),
    ):
        if value is not None and (
            value.ndim != 1 or value.shape[0] not in (batches, tokens)
        ):
            raise ValueError(f"{name} must be [B] or [T] with sequence metadata")
    mode = (
        "reuse"
        if sequence_metadata is not None
        else "export"
        if return_sequence_metadata
        else "none"
    )
    return AttentionSequenceConfig(
        tokens,
        batches,
        *(v is not None for v in vectors),
        paged,
        topk_length is not None and topk_length.numel() == tokens,
        extra_topk_length is not None and extra_topk_length.numel() == tokens,
        topk_length is not None,
        extra_topk_length is not None,
        mode,
    )
