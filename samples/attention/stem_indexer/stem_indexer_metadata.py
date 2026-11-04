# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""在AICPU上执行SI的SectionStreamK分核，固定M64/N256且不拆分S2。"""

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
    from .stem_indexer_metadata_checker import (
        get_effective_core_counts as get_effective_core_counts,
        validate_and_resolve,
    )
else:
    from stem_indexer_metadata_checker import (
        get_effective_core_counts as get_effective_core_counts,
        validate_and_resolve,
    )

AIC_CORE_NUM = 36
AIV_CORE_NUM = 72
HEAD_METADATA_STRIDE = 16
FA_METADATA_STRIDE = 16
L2_BYTES = 96 * 1024 * 1024
M_TASK_SIZE = 64
N_TASK_SIZE = 256
METADATA_ALIGNMENT_ELEMENTS = 4096


class _Args:
    q_seq_lens: gm_in(I32)
    kv_seq_lens: gm_in(I32)
    metadata: gm_out(I32)
    batch_size: U32
    q_heads: U32
    kv_heads: U32
    dim_qkflat: U32
    causal: U32
    stem_block_size: U32
    window_size: U32
    aic_core_num: U32
    output_size: U32


def _cost(m, n):
    return 6 * ceil_div(m, 16) + 10 * ceil_div(n, 64)  # noqa: F821


def _row(a, q, k, m, row):
    # 根据展平G×Q块的M64任务，计算因果可见的N256范围及尾块代价。
    row.blocks = 0
    row.cost = 0
    row.last = 0
    total_m = q * (a.q_heads // a.kv_heads)
    if q == 0 or k == 0 or m >= ceil_div(total_m, M_TASK_SIZE):  # noqa: F821
        return 0
    end = ceil_div(k, N_TASK_SIZE)  # noqa: F821
    if a.causal != 0:
        first = m * M_TASK_SIZE
        last = min(first + M_TASK_SIZE, total_m) - 1
        q_last = q
        if first // q == last // q:
            q_last = last % q
        # 使用有符号长度，避免q_last + k - q - window_size下溢。
        last_k = q_last + k - q - a.window_size
        if last_k < 0:
            return 0
        end = min(last_k, k - 1) // N_TASK_SIZE + 1
    row.blocks = end
    m_size = min(M_TASK_SIZE, total_m - m * M_TASK_SIZE)
    tail = 0
    if k % N_TASK_SIZE != 0 and end == ceil_div(k, N_TASK_SIZE):  # noqa: F821
        tail = 1
    row.cost = _cost(m_size, N_TASK_SIZE) * (end - tail)
    row.last = _cost(m_size, N_TASK_SIZE)
    if tail != 0:
        row.last = _cost(m_size, k % N_TASK_SIZE)
        row.cost += row.last
    return 0


@aicpu_kernel
def _stem_indexer_metadata_kernel(a: _Args):
    class Row:
        blocks: I64
        cost: I64
        last: I64

    # AICPU本地动态数组仅保存每个batch的汇总，不在Host读取长度或计算分核。
    q = zeros(I64, a.batch_size)  # noqa: F821
    k = zeros(I64, a.batch_size)  # noqa: F821
    m_count = zeros(I64, a.batch_size)  # noqa: F821
    costs = zeros(I64, a.batch_size)  # noqa: F821
    blocks = zeros(I64, a.batch_size)  # noqa: F821
    last_costs = zeros(I64, a.batch_size)  # noqa: F821
    boundaries = array(I64, a.batch_size * a.kv_heads + 1)  # noqa: F821
    row = Row()
    group = a.q_heads // a.kv_heads
    for b in range(0, a.batch_size):
        if a.q_seq_lens[b] < 0 or a.kv_seq_lens[b] < 0:
            return 1
        q[b] = ceil_div(a.q_seq_lens[b], a.stem_block_size)  # noqa: F821
        k[b] = ceil_div(a.kv_seq_lens[b], a.stem_block_size)  # noqa: F821
        m_count[b] = ceil_div(q[b] * group, M_TASK_SIZE)  # noqa: F821
        for m in range(0, m_count[b]):
            _row(a, q[b], k[b], m, row)
            costs[b] += row.cost
            blocks[b] += row.blocks
            if row.blocks > 0:
                last_costs[b] = row.last

    # 按96MiB L2预算将连续BN区间划分为多个section。
    token_size = 0
    max_m = 0
    max_single = 0
    bn = 0
    for b in range(0, a.batch_size):
        single = (q[b] + k[b]) * a.dim_qkflat * 4
        max_single = max(max_single, single)
        max_m = max(max_m, q[b] * group)
        for _ in range(0, a.kv_heads):
            if token_size != 0 and token_size + single > L2_BYTES:
                boundaries.append(bn)
                token_size = 0
            token_size += single
            bn += 1
    boundaries.append(a.batch_size * a.kv_heads)
    if max_m <= M_TASK_SIZE or max_single <= L2_BYTES // a.aic_core_num:
        boundaries.reset(0, 0)
        boundaries.append(a.batch_size * a.kv_heads)

    # 明确定义所有保留字段和padding为零，布局保持36个AIC槽和72个AIV保留槽。
    for i in range(0, a.output_size):
        a.metadata[i] = 0
    a.metadata[0] = len(boundaries)
    bn_start = 0
    for section in range(0, len(boundaries)):
        bn_end = boundaries[section]
        section_cost = 0
        section_blocks = 0
        for bn_idx in range(bn_start, bn_end):
            section_cost += costs[bn_idx // a.kv_heads]
            section_blocks += blocks[bn_idx // a.kv_heads]
        base = HEAD_METADATA_STRIDE + section * AIC_CORE_NUM * FA_METADATA_STRIDE
        if section_blocks == 0:
            # 空计算区间仍保留完整BN范围，供主算子的直出/空输出路径遍历。
            a.metadata[base] = bn_start
            a.metadata[base + 3] = bn_end
        else:
            core_num = min(a.aic_core_num, section_blocks)
            bn_idx = bn_start
            b_idx = bn_idx // a.kv_heads
            m_idx = 0
            remaining = section_cost
            batch_cost = costs[b_idx]
            finished = 0
            start_bn = bn_start
            # m表示G×S1_blocks合轴后按M64切分的任务序号。
            start_m = 0
            used = 0
            for core in range(0, core_num):
                if finished != 0 or remaining <= 0:
                    break
                _row(a, q[b_idx], k[b_idx], m_idx, row)
                limit = max(remaining // (core_num - core), row.cost)
                core_cost = 0
                # 优先接收完整的batch/KV-head剩余工作。
                while True:
                    tolerance = last_costs[b_idx] // 2
                    if batch_cost != 0 and core_cost + batch_cost > limit + tolerance:
                        break
                    core_cost += batch_cost
                    bn_idx += 1
                    if bn_idx == bn_end:
                        m_idx = 0
                        finished = 1
                        break
                    b_idx = bn_idx // a.kv_heads
                    batch_cost = costs[b_idx]
                    m_idx = 0
                # 完整BN放不下时，只按完整M64行分配；不拆分N/S2轴。
                if finished == 0:
                    while m_idx < m_count[b_idx]:
                        _row(a, q[b_idx], k[b_idx], m_idx, row)
                        if core_cost + row.cost > limit + row.last // 2:
                            break
                        core_cost += row.cost
                        batch_cost = max(0, batch_cost - row.cost)
                        m_idx += 1
                        while m_idx < m_count[b_idx]:
                            _row(a, q[b_idx], k[b_idx], m_idx, row)
                            if row.blocks > 0:
                                break
                            m_idx += 1
                slot = base + core * FA_METADATA_STRIDE
                a.metadata[slot] = start_bn
                a.metadata[slot + 1] = start_m
                a.metadata[slot + 3] = bn_idx
                a.metadata[slot + 4] = m_idx
                start_bn = bn_idx
                start_m = m_idx
                remaining -= core_cost
                used = core + 1
            # 最后一个核覆盖剩余的零代价行及BN；S2起止保持零。
            if used > 0:
                a.metadata[base + (used - 1) * FA_METADATA_STRIDE + 3] = bn_end
                a.metadata[base + (used - 1) * FA_METADATA_STRIDE + 4] = 0
        bn_start = bn_end
    return 0


def metadata_capacity(batch_size, kv_heads):
    """按固定 Metadata ABI 布局计算并对齐输出容量。"""
    raw_size = (
        1 + batch_size * kv_heads * (AIC_CORE_NUM + AIV_CORE_NUM)
    ) * HEAD_METADATA_STRIDE
    return (
        (raw_size + METADATA_ALIGNMENT_ELEMENTS - 1)
        // METADATA_ALIGNMENT_ELEMENTS
        * METADATA_ALIGNMENT_ELEMENTS
    )


@lru_cache(maxsize=1)
def _compiled_metadata():
    directory = tempfile.TemporaryDirectory(prefix="cannbot_stem_indexer_metadata_")
    compiled = compile_aicpu_kernel(
        _stem_indexer_metadata_kernel, workdir=directory.name, launch_mode="interface"
    )
    return directory, compiled


def stem_indexer_metadata(
    q_seq_lens,
    kv_seq_lens,
    q_heads,
    kv_heads,
    *,
    causal=True,
    stem_block_size=128,
    window_size=4,
    dim_qkflat=2048,
    block_dim=None,
):
    """输入当前NPU的逐batch token长度，异步返回同一流生成的int32 metadata。"""
    tiling = validate_and_resolve(
        q_seq_lens,
        kv_seq_lens,
        q_heads,
        kv_heads,
        causal=causal,
        stem_block_size=stem_block_size,
        window_size=window_size,
        dim_qkflat=dim_qkflat,
        block_dim=block_dim,
        metadata_capacity=metadata_capacity,
        aic_capacity=AIC_CORE_NUM,
        aiv_capacity=AIV_CORE_NUM,
    )
    _, compiled = _compiled_metadata()
    metadata = torch.empty(tiling.output_size, dtype=torch.int32, device=tiling.device)
    compiled.launch(
        current_raw_stream(tiling.device_id),
        q_seq_lens=q_seq_lens.data_ptr(),
        kv_seq_lens=kv_seq_lens.data_ptr(),
        metadata=metadata.data_ptr(),
        batch_size=tiling.batch_size,
        q_heads=tiling.q_heads,
        kv_heads=tiling.kv_heads,
        dim_qkflat=tiling.dim_qkflat,
        causal=int(tiling.causal),
        stem_block_size=tiling.stem_block_size,
        window_size=tiling.window_size,
        aic_core_num=tiling.block_dim,
        output_size=tiling.output_size,
    )
    q_seq_lens.record_stream(tiling.stream)
    kv_seq_lens.record_stream(tiling.stream)
    return metadata
