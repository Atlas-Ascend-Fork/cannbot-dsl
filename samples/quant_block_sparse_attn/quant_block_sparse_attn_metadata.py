# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""AICPU metadata producer for QuantBlockSparseAttn.

The scheduling algorithm is a direct Python DSL port of the original
QuantBlockSparseAttnMetadata AICPU implementation. Tensor contents are read
only by AICPU; the host validates static contracts, allocates the output, and
launches the producer asynchronously on the current NPU stream.
"""

from functools import lru_cache
import tempfile

import torch

from cannbotdsl.aicpu import (
    gm_in,
    gm_out,
    I32,
    I64,
    U32,
    aicpu_kernel,
    current_raw_stream,
)
from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel


METADATA_HEADER = 8
METADATA_CORE_FIELDS = 8
METADATA_AIC_SLOTS = 36
METADATA_AIV_SLOTS = 72
METADATA_SECTION_SIZE = METADATA_CORE_FIELDS * METADATA_AIC_SLOTS
METADATA_FD_SIZE = METADATA_CORE_FIELDS * METADATA_AIV_SLOTS

L2_SECTION_BYTES = 120 * 1024 * 1024
BLOCK_TOLERANCE_RATIO = 2


class _MetadataArgs:
    sparse_seq_len: gm_in(I32)
    metadata: gm_out(I32)
    batch_size: U32
    num_heads_q: U32
    num_heads_kv: U32
    head_dim: U32
    q_capacity: U32
    sparse_block_size_q: U32
    sparse_block_size_k: U32
    aic_core_num: U32
    output_size: I64


def _row_block_count(a, batch, head, q_block):
    offset = (batch * a.num_heads_q + head) * a.q_capacity + q_block
    count = a.sparse_seq_len[offset]
    return count if count > 0 else 0


def _advance_to_nonempty_row(a, bn_limit, bn_block_counts, cursor):
    current_bn = cursor[0]
    current_q_block = cursor[1]
    remaining_bn_blocks = cursor[2]
    while current_bn < bn_limit:
        batch = current_bn // a.num_heads_q
        head = current_bn % a.num_heads_q
        if current_q_block == 0 and remaining_bn_blocks == 0:
            remaining_bn_blocks = bn_block_counts[current_bn]
        while (
            current_q_block < a.q_capacity
            and _row_block_count(a, batch, head, current_q_block) == 0
        ):
            current_q_block += 1
        if current_q_block < a.q_capacity:
            break
        current_bn += 1
        current_q_block = 0
        remaining_bn_blocks = 0
    cursor[0] = current_bn
    cursor[1] = current_q_block
    cursor[2] = remaining_bn_blocks


@aicpu_kernel
def _quant_block_sparse_attn_metadata_kernel(a: _MetadataArgs):
    if a.batch_size == 0 or a.num_heads_q == 0 or a.num_heads_kv == 0:
        return 1
    if a.num_heads_q % a.num_heads_kv != 0 or a.q_capacity == 0 or a.aic_core_num == 0:
        return 1
    if a.aic_core_num > METADATA_AIC_SLOTS:
        return 1

    total_bn = a.batch_size * a.num_heads_q
    bn_block_counts = zeros(I64, total_bn)
    bn_last_row_blocks = zeros(I64, total_bn)
    bn_costs = zeros(I64, total_bn)

    for bn in range(0, total_bn):
        batch = bn // a.num_heads_q
        head = bn % a.num_heads_q
        total_blocks = 0
        last_row_blocks = 0
        valid_rows = 0
        max_row_blocks = 0
        for q_block in range(0, a.q_capacity):
            row_blocks = _row_block_count(a, batch, head, q_block)
            total_blocks += row_blocks
            if row_blocks > 0:
                last_row_blocks = row_blocks
                valid_rows += 1
                max_row_blocks = max(max_row_blocks, row_blocks)
        bn_block_counts[bn] = total_blocks
        bn_last_row_blocks[bn] = last_row_blocks
        q_cost = valid_rows * a.sparse_block_size_q * a.head_dim * 2
        kv_cost = max_row_blocks * a.sparse_block_size_k * a.head_dim * 2
        bn_costs[bn] = q_cost + kv_cost

    section_starts = array(I64, total_bn + 1)
    section_ends = array(I64, total_bn + 1)
    section_blocks = array(I64, total_bn + 1)
    section_start = 0
    section_cost = 0
    block_count = 0
    for bn in range(0, total_bn):
        current_cost = bn_costs[bn]
        if section_cost != 0 and section_cost + current_cost > L2_SECTION_BYTES:
            section_starts.append(section_start)
            section_ends.append(bn)
            section_blocks.append(block_count)
            section_start = bn
            section_cost = 0
            block_count = 0
        section_cost += current_cost
        block_count += bn_block_counts[bn]
    if block_count != 0 or len(section_starts) == 0:
        section_starts.append(section_start)
        section_ends.append(total_bn)
        section_blocks.append(block_count)

    for index in range(0, a.output_size):
        a.metadata[index] = 0
    a.metadata[0] = len(section_starts)

    for section in range(0, len(section_starts)):
        bn_start = section_starts[section]
        bn_limit = section_ends[section]
        unassigned_blocks = section_blocks[section]
        section_base = METADATA_HEADER + section * METADATA_SECTION_SIZE

        if unassigned_blocks == 0:
            slot = section_base
            a.metadata[slot] = 1
            a.metadata[slot + 1] = bn_start
            a.metadata[slot + 2] = 0
            a.metadata[slot + 4] = bn_limit
            a.metadata[slot + 5] = 0
            continue

        current_bn = bn_start
        current_q_block = 0
        remaining_bn_blocks = 0
        cursor = zeros(I64, 3)
        finished = False

        for core in range(0, a.aic_core_num):
            if finished or unassigned_blocks == 0:
                break

            range_start_bn = current_bn
            range_start_q = current_q_block

            cursor[0] = current_bn
            cursor[1] = current_q_block
            cursor[2] = remaining_bn_blocks
            _advance_to_nonempty_row(a, bn_limit, bn_block_counts, cursor)
            current_bn = cursor[0]
            current_q_block = cursor[1]
            remaining_bn_blocks = cursor[2]
            if current_bn >= bn_limit:
                finished = True
                break

            remaining_cores = a.aic_core_num - core
            block_limit = ceil_div(unassigned_blocks, remaining_cores)
            batch = current_bn // a.num_heads_q
            head = current_bn % a.num_heads_q
            first_row_blocks = _row_block_count(a, batch, head, current_q_block)
            block_limit = max(block_limit, first_row_blocks)
            assigned_blocks = 0

            while current_bn < bn_limit:
                cursor[0] = current_bn
                cursor[1] = current_q_block
                cursor[2] = remaining_bn_blocks
                _advance_to_nonempty_row(a, bn_limit, bn_block_counts, cursor)
                current_bn = cursor[0]
                current_q_block = cursor[1]
                remaining_bn_blocks = cursor[2]
                if current_bn >= bn_limit:
                    break
                batch = current_bn // a.num_heads_q
                head = current_bn % a.num_heads_q
                tolerance = bn_last_row_blocks[current_bn] // BLOCK_TOLERANCE_RATIO
                if assigned_blocks + remaining_bn_blocks > block_limit + tolerance:
                    break
                assigned_blocks += remaining_bn_blocks
                current_bn += 1
                current_q_block = 0
                remaining_bn_blocks = 0

            while current_bn < bn_limit:
                cursor[0] = current_bn
                cursor[1] = current_q_block
                cursor[2] = remaining_bn_blocks
                _advance_to_nonempty_row(a, bn_limit, bn_block_counts, cursor)
                current_bn = cursor[0]
                current_q_block = cursor[1]
                remaining_bn_blocks = cursor[2]
                if current_bn >= bn_limit:
                    break
                batch = current_bn // a.num_heads_q
                head = current_bn % a.num_heads_q
                row_blocks = _row_block_count(a, batch, head, current_q_block)
                tolerance = row_blocks // BLOCK_TOLERANCE_RATIO
                if (
                    assigned_blocks > 0
                    and assigned_blocks + row_blocks > block_limit + tolerance
                ):
                    break
                assigned_blocks += row_blocks
                remaining_bn_blocks = max(remaining_bn_blocks - row_blocks, 0)
                current_q_block += 1

            slot = section_base + core * METADATA_CORE_FIELDS
            a.metadata[slot] = 1
            a.metadata[slot + 1] = range_start_bn
            a.metadata[slot + 2] = range_start_q
            a.metadata[slot + 3] = 0
            a.metadata[slot + 4] = current_bn
            a.metadata[slot + 5] = current_q_block
            a.metadata[slot + 6] = 0
            a.metadata[slot + 7] = 0
            unassigned_blocks = max(unassigned_blocks - assigned_blocks, 0)

            if current_bn >= bn_limit:
                finished = True
    return 0


def metadata_capacity(batch_size: int, num_heads_q: int) -> int:
    """Return the original QBSA metadata ABI capacity in int32 elements."""
    return (
        METADATA_HEADER
        + batch_size * num_heads_q * METADATA_SECTION_SIZE
        + METADATA_FD_SIZE
    )


def get_effective_core_counts(stream=None) -> tuple[int, int]:
    """Return the AIC/AIV quota used by the original QBSA metadata op."""
    props = torch.npu.get_device_properties(torch.npu.current_device())
    cube = int(props.cube_core_num)
    vector = int(getattr(props, "vector_core_num", 2 * cube))
    if not 1 <= cube <= METADATA_AIC_SLOTS:
        raise RuntimeError(f"Unsupported effective Cube core count: {cube}")
    if not cube <= vector <= METADATA_AIV_SLOTS:
        raise RuntimeError(f"Unsupported effective Vector core count: {vector}")
    return cube, vector


@lru_cache(maxsize=1)
def _compiled_metadata():
    directory = tempfile.TemporaryDirectory(prefix="cannbot_qbsa_metadata_")
    compiled = compile_aicpu_kernel(
        _quant_block_sparse_attn_metadata_kernel,
        workdir=directory.name,
        launch_mode="interface",
    )
    return directory, compiled


def _check_optional_sequence(name, tensor, expected_length, device):
    if tensor is None:
        return
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f"Input {name} must be a 1D int32 tensor of length {expected_length}"
        )
    if (
        tensor.dtype != torch.int32
        or tensor.ndim != 1
        or tensor.numel() != expected_length
    ):
        raise ValueError(
            f"Input {name} must be a 1D int32 tensor of length {expected_length}"
        )
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"Input {name} must be contiguous on {device}")


def _check_sparse_sequence_contract(
    sparse_seq_len, batch_size, num_heads_q, num_heads_kv
):
    message = "The sparse_seq_len tensor must be a contiguous 3D int32 tensor"
    if not isinstance(sparse_seq_len, torch.Tensor):
        raise ValueError(message)
    if (
        sparse_seq_len.dtype != torch.int32
        or sparse_seq_len.ndim != 3
        or not sparse_seq_len.is_contiguous()
    ):
        raise ValueError(message)
    if sparse_seq_len.device.type != "npu":
        raise ValueError("The sparse_seq_len tensor must be on NPU")
    inferred_batch, sparse_heads, q_capacity = sparse_seq_len.shape
    if batch_size == 0:
        batch_size = inferred_batch
    if batch_size != inferred_batch:
        raise ValueError("The batch_size argument must match sparse_seq_len.shape[0]")
    if num_heads_q != sparse_heads:
        raise ValueError("The num_heads_q argument must match sparse_seq_len.shape[1]")
    head_error = "The num_heads_q value must be a positive multiple of num_heads_kv"
    if isinstance(num_heads_q, bool) or not isinstance(num_heads_q, int):
        raise ValueError(head_error)
    if isinstance(num_heads_kv, bool) or not isinstance(num_heads_kv, int):
        raise ValueError(head_error)
    if num_heads_q <= 0 or num_heads_kv <= 0 or num_heads_q % num_heads_kv != 0:
        raise ValueError(head_error)
    return batch_size, q_capacity


def _check_metadata_shape_options(
    head_dim, sparse_block_size_q, sparse_block_size_k, quant_mode, mask_mode
):
    if head_dim != 128:
        raise NotImplementedError("Only head_dim=128 is supported")
    if sparse_block_size_q != 128 or sparse_block_size_k != 128:
        raise NotImplementedError("Only 128-token sparse blocks are supported")
    if quant_mode != 1 or mask_mode not in (0, 3):
        raise NotImplementedError("Only quant_mode=1 and mask_mode=0/3 are supported")


def _check_metadata_layout_options(layout_q, layout_kv, layout_sparse_indices):
    if layout_q not in ("TND", "NTD"):
        raise NotImplementedError("The layout_q value must be TND or NTD")
    if layout_kv != "PA_BNBD" or layout_sparse_indices != "B_N_Qb_Kb":
        raise NotImplementedError(
            "Only PA_BNBD KV and B_N_Qb_Kb sparse layouts are supported"
        )


def quant_block_sparse_attn_metadata(
    sparse_seq_len,
    num_heads_q,
    num_heads_kv,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    batch_size=0,
    sparse_block_size_q=128,
    sparse_block_size_k=128,
    quant_mode=1,
    mask_mode=3,
    layout_q="TND",
    layout_kv="PA_BNBD",
    layout_sparse_indices="B_N_Qb_Kb",
    block_dim=None,
):
    """Asynchronously generate QBSA metadata on AICPU and the current stream."""
    batch_size, q_capacity = _check_sparse_sequence_contract(
        sparse_seq_len,
        batch_size,
        num_heads_q,
        num_heads_kv,
    )
    _check_metadata_shape_options(
        head_dim,
        sparse_block_size_q,
        sparse_block_size_k,
        quant_mode,
        mask_mode,
    )
    _check_metadata_layout_options(layout_q, layout_kv, layout_sparse_indices)

    device_id = sparse_seq_len.device.index
    device = torch.device("npu", device_id)
    _check_optional_sequence("cu_seqlens_q", cu_seqlens_q, batch_size + 1, device)
    _check_optional_sequence("cu_seqlens_kv", cu_seqlens_kv, batch_size + 1, device)
    _check_optional_sequence("seqused_q", seqused_q, batch_size, device)
    _check_optional_sequence("seqused_kv", seqused_kv, batch_size, device)

    stream = torch.npu.current_stream(device_id)
    cube, _ = get_effective_core_counts(stream)
    if block_dim is not None:
        if isinstance(block_dim, bool) or not isinstance(block_dim, int):
            raise ValueError(f"The block_dim value must be in [1, {cube}]")
        if not 1 <= block_dim <= cube:
            raise ValueError(f"The block_dim value must be in [1, {cube}]")
        cube = block_dim

    output_size = metadata_capacity(batch_size, num_heads_q)
    metadata = torch.empty(output_size, dtype=torch.int32, device=device)
    _, compiled = _compiled_metadata()
    compiled.launch(
        current_raw_stream(device_id),
        sparse_seq_len=sparse_seq_len.data_ptr(),
        metadata=metadata.data_ptr(),
        batch_size=batch_size,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        q_capacity=q_capacity,
        sparse_block_size_q=sparse_block_size_q,
        sparse_block_size_k=sparse_block_size_k,
        aic_core_num=cube,
        output_size=output_size,
    )
    sparse_seq_len.record_stream(stream)
    for tensor in (cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv):
        if tensor is not None:
            tensor.record_stream(stream)
    return metadata


__all__ = [
    "get_effective_core_counts",
    "metadata_capacity",
    "quant_block_sparse_attn_metadata",
]
