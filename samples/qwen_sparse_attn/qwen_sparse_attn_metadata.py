# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Qwen 稀疏注意力 AICPU metadata 算子（单文件实现）。

Block 128 入口与 ops-transformer 中其他 metadata 前置算子
（``flash_attn_metadata``、``sparse_flash_mla_metadata`` 等）保持一致：
任务切分在 NPU AICPU 上完成，主 kernel 只消费生成结果。

生产版 Block 128 模块不包含 metadata planner。描述符张量由此处 AICPU 扫描，
依次完成 causal、重复项及越界过滤，再按 L2 page 预算切分 section。任务在每个
section 内执行确定性加权 LPT 排序，但 core 的累计负载跨 section 保留：同负载
保持构建顺序，core 负载相同时选择编号最小的 core。

输出与主 kernel 消费的 ABI 完全一致：

* ``core_spans`` ``int64`` ``(2, core_capacity)``：每列保存一个 Cube core
  的首个任务下标及末后任务下标；编号不小于 ``counts[0]`` 的 core 为空区间。
* ``tasks`` ``int64`` ``(5, task_capacity)``：每个任务保存 ``qIdx``、
  ``n2Idx``、``gChunkIdx``、``page_base`` 和 ``page_count``。
* ``pages`` ``int32`` ``(2, page_capacity)``：``[0, i]`` 保存物理页号，
  ``[1, i]`` 保存该页的 causal 有效行数（1..block_size）。

当前 Block 128 每个任务只处理一个 Query token 和一个 KV head，固定使用
单 Query ABI v1。``pack_queries`` 关键字暂为兼容接口保留，不改变任务打包或
``pages`` 形状；主 kernel 因此走单 Query Q 暂存路径。FP8 该路径只写有效行，
无效行不参与 Vector softmax、PV 更新或最终输出。

容量契约如下，算子自身不分配设备内存：

    page_capacity = sum(clamp(sparse_block_count, max=topK)) + 1
    task_capacity = T * N2 * ceil(group / 32) + 1

主 kernel 使用 ``usedCoreNum = counts[0]`` 个 Cube core 启动；
``counts[1]`` 和 ``counts[2]`` 仅用于诊断。
"""

__all__ = ["qwen_sparse_attn_metadata"]

import heapq
import os
import tempfile
from functools import lru_cache

from cannbotdsl.aicpu import gm_in, gm_out, I32, I64, aicpu_kernel

# QSA Block 128 固定使用 128-token 逻辑块和 M32 Cube tile。
BLOCK_SIZE = 128
M_TILE = 32
# 解析后物理页 ID 的合法上界，对应 GM 中的 INT32 字段。
MAX_PAGE_ID = 0x7FFFFFFF
# Metadata ABI 版本，写入 ``counts[C_VERSION]`` 供诊断使用。
METADATA_VERSION = 1
PACKED_METADATA_VERSION = 2
# ``None`` 表示使用当前设备的物理 Cube core 数量。保留导出名称可维持公共关键字
# 默认值，同时避免硬编码 128。
DEFAULT_CORE_CAPACITY = None
# 调试开关：设为 1 时关闭 LPT 置换排序，保持任务构建顺序。
SKIP_LPT_SORT = 0

# Sectioned-LPT 使用 96 MiB 的 K+V 工作集预算。FP16/BF16 的一个逻辑页访问
# 占用 64 KiB，对应 1536 次 page visit；FP8 占用 32 KiB，对应 3072 次。
# section 边界只允许出现在不同的 (token, KV-head) 组之间，确保复用同一份
# K/V 页的 GQA chunks 不会被拆开。
SECTION_L2_BYTES = 96 * 1024 * 1024
SECTION_PAGE_BYTES = 2 * BLOCK_SIZE * 128 * 2
SECTION_PAGE_BUDGET_DEFAULT = SECTION_L2_BYTES // SECTION_PAGE_BYTES

# Page count dominates QK/softmax/PV work, while every task also pays for Q
# staging, state initialization, pipeline fill/drain and output materialization.
# Integer weights keep the AICPU scheduler deterministic.
TASK_FIXED_WEIGHT = 2
TASK_PAGE_WEIGHT = 1


def _default_section_page_budget(q):
    """按 K/V 元素字节数计算 96 MiB section 可容纳的 page visit 数。"""
    element_bytes = int(q.element_size())
    if element_bytes < 1:
        raise ValueError("Q element size must be positive")
    page_bytes = 2 * BLOCK_SIZE * 128 * element_bytes
    return max(1, SECTION_L2_BYTES // page_bytes)


class QsaMetadataArgs:
    """Kernel 参数结构体，字段顺序与 GM 参数顺序一致。

    AICPU interface launch 会让结构体中的每个字段单独占用一个参数槽。当按值传递的
    参数块超出寄存器预算时，kernel 入口会拒绝启动（包含 10 个指针、4 个 ``I64``
    和 13 个 ``U32`` 的结构体会报 "execute kernel param invalid"，而纯指针结构体
    可以正常启动）。因此所有标量通过小型设备张量 ``params`` 传递，结构体仅保留指针。

    ``params`` 布局如下，由 host wrapper 写入，dtype 为 ``int32``：

        0 core_capacity   1 task_capacity   2 page_capacity   3 chunk_num
        4 used_core_num_cap  5 block_size   6 kv_heads        7 sq_len
        8 cu_q_len        9 table_cols     10 topk           11 idx_tokens
       12 kv_len        13 section_page_budget

    ``counts`` 是小型 ``int32`` 结果张量：``0`` 为实际使用核数，``1`` 为写入
    任务数，``2`` 为写入页数，``3`` 为 metadata ABI 版本。
    """

    # 描述符输入
    sparse_block_idx: gm_in(I32)
    sparse_block_count: gm_in(I32)
    cu_seqlens_q: gm_in(I64)
    seqused_kv: gm_in(I32)
    block_table: gm_in(I32)
    seqused_q: gm_in(I32)
    # 标量传输与 metadata 输出
    params: gm_in(I32)
    counts: gm_out(I32)
    core_spans: gm_out(I64)
    tasks: gm_out(I64)
    pages: gm_out(I32)
    # 每个 `(token, KV head)` 的已接收页计数，以及按 `(batch, head, token)`
    # 访问顺序生成的全局页号前缀和；两者均由页扫描阶段写入、任务阶段消费。
    token_counts: gm_out(I32)
    slot_end: gm_out(I32)


# ``params`` 张量字段下标
P_CORE_CAPACITY = 0
P_TASK_CAPACITY = 1
P_PAGE_CAPACITY = 2
P_CHUNK_NUM = 3
P_USED_CORE_NUM_CAP = 4
P_BLOCK_SIZE = 5
P_KV_HEADS = 6
P_SQ_LEN = 7
P_CU_Q_LEN = 8
P_TABLE_COLS = 9
P_TOPK = 10
P_IDX_TOKENS = 11
P_KV_LEN = 12
P_SECTION_PAGE_BUDGET = 13
PARAMS_SIZE = 14

# ``counts`` 输出张量字段下标
C_USED_CORE_NUM = 0
C_ITEMS = 1
C_PAGES = 2
C_VERSION = 3
COUNTS_SIZE = 4


# ---------------------------------------------------------------------------
# kernel 入口
# ---------------------------------------------------------------------------


@aicpu_kernel
def _qwen_sparse_attn_metadata_kernel(a: QsaMetadataArgs):
    # --- 读取标量参数（设备张量，见 QsaMetadataArgs）---
    core_capacity = a.params[P_CORE_CAPACITY]
    task_capacity = a.params[P_TASK_CAPACITY]
    page_capacity = a.params[P_PAGE_CAPACITY]
    g_chunk_num = a.params[P_CHUNK_NUM]
    used_core_num_cap = a.params[P_USED_CORE_NUM_CAP]
    block_size = a.params[P_BLOCK_SIZE]
    n2_num = a.params[P_KV_HEADS]
    seqused_q_num = a.params[P_SQ_LEN]
    cu_seqlens_q_num = a.params[P_CU_Q_LEN]
    table_col_num = a.params[P_TABLE_COLS]
    topk = a.params[P_TOPK]
    q_token_num = a.params[P_IDX_TOKENS]
    section_page_budget = a.params[P_SECTION_PAGE_BUDGET]

    if task_capacity < 1 or page_capacity < 1 or core_capacity < 1:
        return 1
    if block_size != BLOCK_SIZE:
        return 1
    if n2_num < 1 or g_chunk_num < 1:
        return 1

    # --- 暂存区（AICPU 单线程，通过 arena 分配）----------------------------
    batches = cu_seqlens_q_num - 1
    # array/zeros are AICPU compiler intrinsics injected while compiling this kernel.
    task_n2_idx = array(I32, task_capacity)  # noqa: F821
    task_q_idx = array(I32, task_capacity)  # noqa: F821
    task_g_chunk_idx = array(I32, task_capacity)  # noqa: F821
    task_page_offset = array(I32, task_capacity)  # noqa: F821
    task_page_num = array(I32, task_capacity)  # noqa: F821
    task_load = array(I32, task_capacity)  # noqa: F821
    task_owner = zeros(I32, task_capacity)  # noqa: F821
    task_order = zeros(I32, task_capacity)  # noqa: F821
    section_start = zeros(I32, task_capacity + 1)  # noqa: F821
    core_load = zeros(I32, core_capacity)  # noqa: F821
    task_end_by_core = zeros(I32, core_capacity)  # noqa: F821
    task_num = 0
    page_num = 0

    # --- 页记录顺序：batch -> token -> head，与 host planner 完全一致 --------
    # 页扫描还会记录每个 (token, head) 贡献的页数，因此任务扫描无需重新推导
    # causal 过滤结果。
    for batch_idx in range(0, batches):
        q_start = a.cu_seqlens_q[batch_idx]
        q_end = a.cu_seqlens_q[batch_idx + 1]
        q_length = q_end - q_start
        if seqused_q_num > 0:
            q_length = a.seqused_q[batch_idx]
        kv_length = a.seqused_kv[batch_idx]
        causal_position_offset = kv_length - q_length
        for q_idx in range(q_start, q_end):
            local_q_idx = q_idx - q_start
            causal_position = causal_position_offset + local_q_idx
            for n2_idx in range(0, n2_num):
                page_list_offset = page_num
                if local_q_idx < q_length and kv_length > 0:
                    block_num = a.sparse_block_count[n2_idx * q_token_num + q_idx]
                    if block_num > topk:
                        block_num = topk
                    for slot in range(0, block_num):
                        accept = 0
                        logical_page_idx = a.sparse_block_idx[
                            n2_idx * q_token_num * topk + q_idx * topk + slot
                        ]
                        if logical_page_idx >= 0 and logical_page_idx < table_col_num:
                            if logical_page_idx * block_size <= causal_position:
                                accept = 1
                        if accept > 0:
                            physical_page_idx = a.block_table[
                                batch_idx * table_col_num + logical_page_idx
                            ]
                            if physical_page_idx < 0:
                                return 2
                            if physical_page_idx > MAX_PAGE_ID:
                                return 2
                            val_col = (
                                causal_position - logical_page_idx * block_size + 1
                            )
                            if val_col > block_size:
                                val_col = block_size
                            if page_num >= page_capacity:
                                return 3
                            a.pages[page_num] = physical_page_idx
                            a.pages[page_capacity + page_num] = val_col
                            page_num = page_num + 1
                a.token_counts[n2_idx * q_token_num + q_idx] = (
                    page_num - page_list_offset
                )
                # 按 (batch, head, token) 访问顺序计算全局页号前缀和；任务扫描用
                # 前缀和减去当前 token 的页数即可得到 base，无需跨 head 累加器。
                a.slot_end[
                    batch_idx * n2_num * q_token_num + n2_idx * q_token_num + q_idx
                ] = page_num

    # --- 任务顺序：batch -> head -> token，即 host planner 的 tie-break 顺序 --
    for batch_idx in range(0, batches):
        q_start = a.cu_seqlens_q[batch_idx]
        q_end = a.cu_seqlens_q[batch_idx + 1]
        q_length = q_end - q_start
        if seqused_q_num > 0:
            q_length = a.seqused_q[batch_idx]
        kv_length = a.seqused_kv[batch_idx]
        if kv_length > 0:
            for n2_idx in range(0, n2_num):
                # base 等于“截至当前 token 已接收的页数”减去“当前 token
                # 接收的页数”，与 planner 的 ``base = len(page_rows)``
                # 语义一致，且无需跨 head 累加器。
                running = 0
                if batch_idx * n2_num * q_token_num + n2_idx * q_token_num > 0:
                    running = a.slot_end[
                        batch_idx * n2_num * q_token_num + n2_idx * q_token_num - 1
                    ]
                for q_idx in range(q_start, q_end):
                    local_q_idx = q_idx - q_start
                    selected_page_num = 0
                    if local_q_idx < q_length:
                        selected_page_num = a.token_counts[n2_idx * q_token_num + q_idx]
                        running = a.slot_end[
                            batch_idx * n2_num * q_token_num
                            + n2_idx * q_token_num
                            + q_idx
                        ]
                    base = running - selected_page_num
                    for g_chunk_idx in range(0, g_chunk_num):
                        if task_num >= task_capacity:
                            return 4
                        task_n2_idx[task_num] = n2_idx
                        task_q_idx[task_num] = q_idx
                        task_g_chunk_idx[task_num] = g_chunk_idx
                        task_page_offset[task_num] = base
                        task_page_num[task_num] = selected_page_num
                        task_load[task_num] = (
                            TASK_FIXED_WEIGHT + TASK_PAGE_WEIGHT * selected_page_num
                        )
                        task_num = task_num + 1

    # 每个任务最多使用一个 Cube core；无任务时仍保留一个启动 core，确保主
    # kernel 始终收到合法的 block dim。
    used_core_num = used_core_num_cap
    if used_core_num > task_num:
        used_core_num = task_num
    if used_core_num < 1:
        used_core_num = 1

    # --- Sectioned LPT 调度 ------------------------------------------------
    # section 按原始 batch/head/token 顺序构造，以 page visit 估计 L2 工作集；
    # 仅在 token/head 边界切分，避免把共享同一页表的 GQA chunks 分开。任务在
    # section 内稳定排序，但 coreLoad 必须跨 section 累计，否则每个 section
    # 的余数都会反复落到低编号 core，形成稳定长尾。
    for i in range(0, task_num):
        task_order[i] = i
    section_num = 0
    section_traffic = 0
    section_start[0] = 0
    for i in range(0, task_num):
        at_group_boundary = 0
        if i > 0:
            if (
                task_q_idx[i] != task_q_idx[i - 1]
                or task_n2_idx[i] != task_n2_idx[i - 1]
            ):
                at_group_boundary = 1
        if (
            at_group_boundary != 0
            and section_traffic > 0
            and section_traffic + task_page_num[i] > section_page_budget
        ):
            section_num = section_num + 1
            section_start[section_num] = i
            section_traffic = 0
        section_traffic = section_traffic + task_page_num[i]
    if task_num > 0:
        section_num = section_num + 1
    section_start[section_num] = task_num

    for section_idx in range(0, section_num):
        begin = section_start[section_idx]
        end = section_start[section_idx + 1]
        if SKIP_LPT_SORT == 0:
            best = begin
            for i in range(begin, end):
                best = i
                for j in range(i + 1, end):
                    if task_load[task_order[j]] > task_load[task_order[best]]:
                        best = j
                swap = task_order[i]
                task_order[i] = task_order[best]
                task_order[best] = swap
        for schedule_idx in range(begin, end):
            best = 0
            for core_idx in range(1, used_core_num):
                if core_load[core_idx] < core_load[best]:
                    best = core_idx
            rank = task_order[schedule_idx]
            task_end_by_core[best] = task_end_by_core[best] + 1
            core_load[best] = core_load[best] + task_load[rank]
            task_owner[rank] = best

    # --- 构造 core_spans --------------------------------------------------
    cursor = 0
    for core_idx in range(0, core_capacity):
        a.core_spans[core_idx] = cursor
        if core_idx < used_core_num:
            cursor = cursor + task_end_by_core[core_idx]
        a.core_spans[core_capacity + core_idx] = cursor

    # --- 构造 tasks：遍历一次排序结果，并按所属 core 分组 ------------------
    task_write_idx = 0
    for core_idx in range(0, used_core_num):
        for schedule_idx in range(0, task_num):
            rank = task_order[schedule_idx]
            if task_owner[rank] == core_idx:
                a.tasks[task_write_idx] = task_q_idx[rank]
                a.tasks[task_capacity + task_write_idx] = task_n2_idx[rank]
                a.tasks[2 * task_capacity + task_write_idx] = task_g_chunk_idx[rank]
                a.tasks[3 * task_capacity + task_write_idx] = task_page_offset[rank]
                a.tasks[4 * task_capacity + task_write_idx] = task_page_num[rank]
                task_write_idx = task_write_idx + 1

    a.counts[C_USED_CORE_NUM] = used_core_num
    a.counts[C_ITEMS] = task_num
    a.counts[C_PAGES] = page_num
    a.counts[C_VERSION] = METADATA_VERSION
    return 0


@lru_cache(maxsize=1)
def _compiled_metadata():
    from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel

    workdir = os.environ.get("QSA_METADATA_WORKDIR") or tempfile.mkdtemp(
        prefix="cannbot_qwen_sparse_attn_metadata_"
    )
    compiled = compile_aicpu_kernel(
        _qwen_sparse_attn_metadata_kernel,
        workdir=workdir,
        launch_mode="interface",
    )
    return workdir, compiled


def metadata_capacities(
    sparse_block_idx,
    sparse_block_count,
    cu_seqlens_q,
    seqused_kv,
    q,
    block_table,
    *,
    cores=None,
    block_size=BLOCK_SIZE,
    core_capacity=DEFAULT_CORE_CAPACITY,
):
    """计算 metadata 输出和启动所需的静态容量上界。"""
    import torch

    kv_heads, idx_tokens, topk = (int(value) for value in sparse_block_idx.shape)
    group = int(q.shape[1]) // kv_heads
    chunk_num = (group + M_TILE - 1) // M_TILE
    page_capacity = int(sparse_block_idx.numel()) + 1
    task_capacity = idx_tokens * kv_heads * chunk_num + 1
    if cores is None:
        properties = torch.npu.get_device_properties(torch.npu.current_device())
        cores = int(getattr(properties, "cube_core_num", 1))
    core_capacity = int(cores if core_capacity is None else core_capacity)
    return dict(
        core_capacity=core_capacity,
        task_capacity=task_capacity,
        page_capacity=page_capacity,
        chunk_num=chunk_num,
        group=group,
        used_core_num_cap=min(int(cores), core_capacity),
    )


def _sectioned_lpt(tasks, used_core_num, section_page_budget):
    """按构建顺序切分 section，并在跨 section 累计负载上执行稳定 LPT。

    本函数同时作为 AICPU 实现的参考和多 Query host planner。section 边界只允许
    出现在不同的 ``(qIdx, n2Idx)`` 组之间，避免拆开共享 K/V 页的 GQA chunks。
    """
    tasks_by_core = [[] for _ in range(used_core_num)]
    if not tasks:
        return tasks_by_core

    sections = []
    current = []
    traffic = 0
    previous_group = None
    for order, task in enumerate(tasks):
        group = (int(task[0]), int(task[1]))
        pages = int(task[4])
        if (
            current
            and group != previous_group
            and traffic > 0
            and traffic + pages > section_page_budget
        ):
            sections.append(current)
            current = []
            traffic = 0
        current.append((order, task))
        traffic += pages
        previous_group = group
    sections.append(current)

    load_heap = [(0, core_idx) for core_idx in range(used_core_num)]
    heapq.heapify(load_heap)
    for section in sections:
        ordered = sorted(
            section,
            key=lambda item: -(TASK_FIXED_WEIGHT + TASK_PAGE_WEIGHT * int(item[1][4])),
        )
        for _, task in ordered:
            current_load, core_idx = heapq.heappop(load_heap)
            tasks_by_core[core_idx].append(task)
            task_cost = TASK_FIXED_WEIGHT + TASK_PAGE_WEIGHT * int(task[4])
            heapq.heappush(load_heap, (current_load + task_cost, core_idx))
    return tasks_by_core


def qwen_sparse_attn_metadata(
    sparse_block_idx,
    sparse_block_count,
    cu_seqlens_q,
    seqused_kv,
    q,
    block_table,
    *,
    seqused_q=None,
    cores=None,
    block_size=BLOCK_SIZE,
    core_capacity=DEFAULT_CORE_CAPACITY,
    page_capacity=None,
    task_capacity=None,
    pack_queries=None,
    section_page_budget=None,
):
    """在 AICPU 上构造 QSA 单 Query 任务切分 metadata。

    Returns ``(core_spans, tasks, pages, counts, status)``. Pass this complete
    result to ``qwen_sparse_attn_block_128``; the main operator launches from
    tensor shapes without copying metadata values back to the host. The
    ``pack_queries`` argument is retained for API compatibility but is not
    applied: ``pages`` always has two rows and the ABI version is 1.
    """
    import torch
    from cannbotdsl.aicpu import current_raw_stream

    if section_page_budget is None:
        section_page_budget = _default_section_page_budget(q)
    plan = metadata_capacities(
        sparse_block_idx,
        sparse_block_count,
        cu_seqlens_q,
        seqused_kv,
        q,
        block_table,
        cores=cores,
        block_size=block_size,
        core_capacity=core_capacity,
    )
    core_capacity = plan["core_capacity"]
    page_capacity = (
        plan["page_capacity"] if page_capacity is None else int(page_capacity)
    )
    task_capacity = (
        plan["task_capacity"] if task_capacity is None else int(task_capacity)
    )
    device_id = torch.npu.current_device()
    device = q.device
    stream = torch.npu.current_stream(device_id)
    compiled = _compiled_metadata()[1]
    core_spans = torch.empty((2, core_capacity), dtype=torch.int64, device=device)
    tasks = torch.empty((5, task_capacity), dtype=torch.int64, device=device)
    pages = torch.empty((2, page_capacity), dtype=torch.int32, device=device)
    counts = torch.empty(COUNTS_SIZE, dtype=torch.int32, device=device)
    params = torch.tensor(
        [
            core_capacity,
            task_capacity,
            page_capacity,
            plan["chunk_num"],
            plan["used_core_num_cap"],
            int(block_size),
            int(sparse_block_idx.shape[0]),
            0 if seqused_q is None else int(seqused_q.numel()),
            int(cu_seqlens_q.numel()),
            int(block_table.shape[1]),
            int(sparse_block_idx.shape[2]),
            int(sparse_block_idx.shape[1]),
            int(seqused_kv.numel()),
            section_page_budget,
        ],
        dtype=torch.int32,
        device=device,
    )
    token_counts = torch.empty(
        int(sparse_block_idx.shape[0]) * int(sparse_block_idx.shape[1]),
        dtype=torch.int32,
        device=device,
    )
    # batches * kv_heads * T，即每个访问位置对应一个前缀和单元
    slot_end = torch.empty(
        int(cu_seqlens_q.numel() - 1)
        * int(sparse_block_idx.shape[0])
        * int(sparse_block_idx.shape[1]),
        dtype=torch.int32,
        device=device,
    )
    status = compiled.launch(
        current_raw_stream(device_id),
        sparse_block_idx=sparse_block_idx.data_ptr(),
        sparse_block_count=sparse_block_count.data_ptr(),
        cu_seqlens_q=cu_seqlens_q.data_ptr(),
        seqused_kv=seqused_kv.data_ptr(),
        block_table=block_table.data_ptr(),
        seqused_q=0 if seqused_q is None else seqused_q.data_ptr(),
        params=params.data_ptr(),
        counts=counts.data_ptr(),
        core_spans=core_spans.data_ptr(),
        tasks=tasks.data_ptr(),
        pages=pages.data_ptr(),
        token_counts=token_counts.data_ptr(),
        slot_end=slot_end.data_ptr(),
    )
    for tensor in (
        sparse_block_idx,
        sparse_block_count,
        cu_seqlens_q,
        seqused_kv,
        block_table,
        seqused_q,
        q,
    ):
        if tensor is not None:
            tensor.record_stream(stream)
    return core_spans, tasks, pages, counts, status
