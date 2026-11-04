# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""在 AICPU 上异步生成 QSA 全 K 扫描任务的调度元数据。"""

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

if __package__:
    from .qsa_indexer_metadata_checker import (
        get_effective_core_counts as get_effective_core_counts,
        validate_and_resolve,
    )
else:
    from qsa_indexer_metadata_checker import (
        get_effective_core_counts as get_effective_core_counts,
        validate_and_resolve,
    )

AIC_METADATA_CORE_CAPACITY = 36
HEAD_METADATA_STRIDE = 16
HEAD_SECTION_COUNT_INDEX = 0
CORE_METADATA_STRIDE = 16
CORE_BN_BEGIN_INDEX = 0
CORE_M_BEGIN_INDEX = 1
CORE_BN_END_INDEX = 3
CORE_M_END_INDEX = 4
SECTION_METADATA_STRIDE = AIC_METADATA_CORE_CAPACITY * CORE_METADATA_STRIDE
METADATA_ALIGNMENT_ELEMS = 4096
QUERY_TILE = 32
QUERY_HEAD_COUNT = 4
HEAD_DIM = 128
TOKEN_COMPRESSION_RATIO = 4
K_BASE_SIZE = 256
PAGE_SIZE = 256
MAX_TOPK = 512
K_COMPRESSED_UNITS_PER_TOPK_CHUNK = 4096
BF16_BYTES = 2
L2_SECTION_BUDGET_BYTES = 96 * 1024 * 1024
METADATA_SUCCESS = 0
METADATA_INVALID_INPUT = 1


# 沿用现有调度模型的相对代价，非实测耗时；原始标定依据待补充。
# 评分按 Query head 行与压缩 K 单元分组计费；TopK、读回按候选单元分组计费。
SCORING_QUERY_ROWS_PER_COST_UNIT = 16
SCORING_K_UNITS_PER_COST_UNIT = 64
TOPK_CANDIDATES_PER_COST_UNIT = 128
READBACK_CANDIDATES_PER_COST_UNIT = 128
SCORING_QUERY_COST_WEIGHT = 6
SCORING_K_COST_WEIGHT = 10
TOPK_FINAL_COST_WEIGHT = 6
TOPK_RETAIN_COST_WEIGHT = 8
VECTOR_STAGE_COST_WEIGHT = 4
VECTOR_OUTPUT_BASE_COST = 1


class _MetadataArgs:
    actual_seq: gm_in(I32)
    query_positions: gm_in(I32)
    block_table: gm_in(I32)
    metadata: gm_out(I32)
    batch_size: U32
    total_query: U32
    max_pages: U32
    compressed_page_count: U32
    core_count: U32
    output_size: U32


def _task_cost(k_compressed_unit_count):
    score_group_count = ceil_div(k_compressed_unit_count, K_BASE_SIZE)  # noqa: F821
    topk_trunk_count = ceil_div(  # noqa: F821
        k_compressed_unit_count, K_COMPRESSED_UNITS_PER_TOPK_CHUNK
    )
    scoring_cost = score_group_count * (
        SCORING_QUERY_COST_WEIGHT
        * ceil_div(QUERY_TILE * QUERY_HEAD_COUNT, SCORING_QUERY_ROWS_PER_COST_UNIT)  # noqa: F821
        + SCORING_K_COST_WEIGHT * ceil_div(K_BASE_SIZE, SCORING_K_UNITS_PER_COST_UNIT)  # noqa: F821
    )
    topk = 0
    if topk_trunk_count == 1:
        topk = TOPK_FINAL_COST_WEIGHT * ceil_div(  # noqa: F821
            K_COMPRESSED_UNITS_PER_TOPK_CHUNK, TOPK_CANDIDATES_PER_COST_UNIT
        )
    elif topk_trunk_count > 1:
        topk = TOPK_RETAIN_COST_WEIGHT * ceil_div(  # noqa: F821
            K_COMPRESSED_UNITS_PER_TOPK_CHUNK, TOPK_CANDIDATES_PER_COST_UNIT
        )
        if topk_trunk_count > 2:
            topk += (
                (topk_trunk_count - 2)
                * TOPK_RETAIN_COST_WEIGHT
                * ceil_div(  # noqa: F821
                    K_COMPRESSED_UNITS_PER_TOPK_CHUNK + MAX_TOPK,
                    TOPK_CANDIDATES_PER_COST_UNIT,
                )
            )
        topk += TOPK_FINAL_COST_WEIGHT * ceil_div(  # noqa: F821
            K_COMPRESSED_UNITS_PER_TOPK_CHUNK + MAX_TOPK, TOPK_CANDIDATES_PER_COST_UNIT
        )
    score_readback_cost = topk_trunk_count * ceil_div(  # noqa: F821
        K_COMPRESSED_UNITS_PER_TOPK_CHUNK, READBACK_CANDIDATES_PER_COST_UNIT
    )
    padding_group_count = (
        topk_trunk_count * (K_COMPRESSED_UNITS_PER_TOPK_CHUNK // K_BASE_SIZE)
        - score_group_count
    )
    return (
        scoring_cost
        + VECTOR_STAGE_COST_WEIGHT
        * (topk + score_readback_cost + VECTOR_OUTPUT_BASE_COST)
        + max(0, padding_group_count)
    )


@aicpu_kernel
def _qsa_indexer_metadata_kernel(args: _MetadataArgs):
    # 根据设备侧输入构建各请求的汇总信息及完整的 Q32 任务序列。
    offsets = zeros(I64, args.batch_size + 1)  # noqa: F821
    footprints = zeros(I64, args.batch_size)  # noqa: F821
    task_capacity = ceil_div(args.total_query, QUERY_TILE) + args.batch_size  # noqa: F821
    task_bns = zeros(I64, task_capacity)  # noqa: F821
    task_m = zeros(I64, task_capacity)  # noqa: F821
    prefix = zeros(I64, task_capacity + 1)  # noqa: F821
    boundaries = array(I64, args.batch_size + 1)  # noqa: F821

    if args.actual_seq[0] != 0 or args.actual_seq[args.batch_size] != args.total_query:
        return METADATA_INVALID_INPUT

    total_tasks = 0
    max_query_len = 0
    max_footprint = 0
    for bn in range(0, args.batch_size):
        begin = args.actual_seq[bn]
        end = args.actual_seq[bn + 1]
        if begin < 0 or end < begin or end > args.total_query:
            return METADATA_INVALID_INPUT
        query_len = end - begin
        max_query_len = max(max_query_len, query_len)
        visible = 0
        for row in range(begin, end):
            position = args.query_positions[row]
            if position < 0:
                return METADATA_INVALID_INPUT
            visible = max(visible, (position + 1) // TOKEN_COMPRESSION_RATIO)
        required_pages = ceil_div(visible, PAGE_SIZE)  # noqa: F821
        if required_pages > args.max_pages:
            return METADATA_INVALID_INPUT
        for page in range(0, required_pages):
            physical = args.block_table[bn * args.max_pages + page]
            if physical < 0 or physical >= args.compressed_page_count:
                return METADATA_INVALID_INPUT
        footprint = query_len * QUERY_HEAD_COUNT * HEAD_DIM * BF16_BYTES
        footprint += required_pages * PAGE_SIZE * HEAD_DIM * BF16_BYTES
        footprints[bn] = footprint
        max_footprint = max(max_footprint, footprint)

        m_count = ceil_div(query_len, QUERY_TILE)  # noqa: F821
        for m in range(0, m_count):
            task_begin = begin + m * QUERY_TILE
            task_end = min(task_begin + QUERY_TILE, end)
            blocks = 0
            for row in range(task_begin, task_end):
                blocks = max(
                    blocks, (args.query_positions[row] + 1) // TOKEN_COMPRESSION_RATIO
                )
            task_bns[total_tasks] = bn
            task_m[total_tasks] = m
            task_cost = _task_cost(blocks)
            prefix[total_tasks + 1] = prefix[total_tasks] + task_cost
            total_tasks += 1
        offsets[bn + 1] = total_tasks

    # 按 96 MiB 容量策略划分 section，分界点仅取 bn 边界。
    current = 0
    for bn in range(0, args.batch_size):
        footprint = footprints[bn]
        if current != 0 and current + footprint > L2_SECTION_BUDGET_BYTES:
            boundaries.append(bn)
            current = 0
        current += footprint
    boundaries.append(args.batch_size)
    if (
        max_query_len <= QUERY_TILE
        or max_footprint <= L2_SECTION_BUDGET_BYTES // args.core_count
    ):
        boundaries.reset(0, 0)
        boundaries.append(args.batch_size)

    for i in range(0, args.output_size):
        args.metadata[i] = 0
    args.metadata[HEAD_SECTION_COUNT_INDEX] = len(boundaries)

    section_start_bn = 0
    for section in range(0, len(boundaries)):
        section_end_bn = boundaries[section]
        first = offsets[section_start_bn]
        last = offsets[section_end_bn]
        task_count = last - first
        core_num = min(args.core_count, max(1, task_count))
        cursor = first
        start_bn = section_start_bn
        start_m = 0
        base = HEAD_METADATA_STRIDE + section * SECTION_METADATA_STRIDE

        for core in range(0, core_num):
            remaining_cores = core_num - core
            end_task = last
            if remaining_cores > 1:
                candidate_begin_task = cursor + 1
                candidate_end_task = last - (remaining_cores - 1)
                remaining_cost = prefix[last] - prefix[cursor]
                best = candidate_begin_task
                best_error = -1
                for candidate in range(candidate_begin_task, candidate_end_task + 1):
                    scaled = (prefix[candidate] - prefix[cursor]) * remaining_cores
                    error = scaled - remaining_cost
                    if error < 0:
                        error = -error
                    if best_error < 0 or error < best_error:
                        best = candidate
                        best_error = error
                end_task = best

            end_bn = section_end_bn
            end_m = 0
            if end_task < last:
                end_bn = task_bns[end_task]
                end_m = task_m[end_task]
            slot = base + core * CORE_METADATA_STRIDE
            args.metadata[slot + CORE_BN_BEGIN_INDEX] = start_bn
            args.metadata[slot + CORE_M_BEGIN_INDEX] = start_m
            args.metadata[slot + CORE_BN_END_INDEX] = end_bn
            args.metadata[slot + CORE_M_END_INDEX] = end_m
            cursor = end_task
            start_bn = end_bn
            start_m = end_m

        # 未参与计算的物理核写入当前 section 的排他结束坐标，表示空任务区间。
        for core in range(core_num, args.core_count):
            slot = base + core * CORE_METADATA_STRIDE
            args.metadata[slot + CORE_BN_BEGIN_INDEX] = section_end_bn
            args.metadata[slot + CORE_BN_END_INDEX] = section_end_bn
        section_start_bn = section_end_bn
    return METADATA_SUCCESS


def metadata_capacity(batch_size: int) -> int:
    raw = HEAD_METADATA_STRIDE + max(1, batch_size) * SECTION_METADATA_STRIDE
    return (
        (raw + METADATA_ALIGNMENT_ELEMS - 1)
        // METADATA_ALIGNMENT_ELEMS
        * METADATA_ALIGNMENT_ELEMS
    )


@lru_cache(maxsize=1)
def _compiled_metadata():
    directory = tempfile.TemporaryDirectory(prefix="cannbot_qsa_indexer_metadata_")
    compiled = compile_aicpu_kernel(
        _qsa_indexer_metadata_kernel, workdir=directory.name, launch_mode="interface"
    )
    return directory, compiled


def qsa_indexer_metadata(
    actual_seq,
    query_positions,
    block_table,
    compressed_page_count,
    *,
    block_dim=None,
):
    """在当前 NPU 执行流上异步生成并返回 QSA 全 K 扫描任务的调度元数据。"""
    tiling = validate_and_resolve(
        actual_seq,
        query_positions,
        block_table,
        compressed_page_count,
        block_dim,
        metadata_capacity=metadata_capacity,
        aic_capacity=AIC_METADATA_CORE_CAPACITY,
    )
    actual_seq = actual_seq.contiguous()
    query_positions = query_positions.contiguous()
    block_table = block_table.contiguous()
    _, compiled = _compiled_metadata()
    metadata = torch.empty(
        tiling.output_size,
        dtype=torch.int32,
        device=tiling.device,
    )
    compiled.launch(
        current_raw_stream(tiling.device_id),
        actual_seq=actual_seq.data_ptr(),
        query_positions=query_positions.data_ptr(),
        block_table=block_table.data_ptr(),
        metadata=metadata.data_ptr(),
        batch_size=tiling.batch_size,
        total_query=tiling.total_query,
        max_pages=tiling.max_pages,
        compressed_page_count=tiling.compressed_page_count,
        core_count=tiling.block_dim,
        output_size=tiling.output_size,
    )
    actual_seq.record_stream(tiling.stream)
    query_positions.record_stream(tiling.stream)
    block_table.record_stream(tiling.stream)
    return metadata
