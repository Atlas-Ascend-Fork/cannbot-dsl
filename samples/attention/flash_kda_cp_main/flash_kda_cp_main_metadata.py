# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# fmt: off

"""FlashKDA CP main metadata producer."""

import os
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import torch
from cannbotdsl import get_platform_info
from cannbotdsl.aicpu import (
    I32,
    I64,
    U32,
    aicpu_kernel,
    current_raw_stream,
    gm_in,
    gm_out,
)
from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


MAGIC = 0x4B444132
ABI_VERSION = 2
MAX_AIC_CORES = 32
WORKSPACE_SLOTS = 861
CHUNK_SIZE = 64
# Keep the exact FlashKDA stage12w fixed record layout.  CP preprocess reuses
# the stage2 task partition for its per-head summary pass (split zero of each
# head), so no CP-only fields are appended to the shared ABI.
ROUND_FIXED_WORDS = 133
ROUND_WORDS_PER_BATCH = 9

HEADER_MAGIC = 0
HEADER_ABI_VERSION = 1
HEADER_STATUS = 2
HEADER_CORE_NUM = 3
HEADER_STAGE12_ROUND_NUM = 4
HEADER_BATCH = 5
HEADER_VALUE_HEADS = 6
HEADER_HAS_CU = 7
HEADER_WORKSPACE_SLOTS = 8
HEADER_WORDS = 9

STATUS_OK = 0
STATUS_BAD_CORE_NUM = 1
STATUS_BAD_WORKSPACE_SLOTS = 2
STATUS_BAD_FLAGS = 3
STATUS_BAD_SHAPE = 4
STATUS_BAD_CU_START = 5
STATUS_BAD_LENGTH = 6
STATUS_BAD_PADDED_LENGTH = 7
STATUS_BAD_PACKED_END = 8
STATUS_BAD_METADATA_CAPACITY = 9

MAX_ACTIVE_BATCHES = WORKSPACE_SLOTS
MAX_STAGE1_TASKS = WORKSPACE_SLOTS
MAX_STAGE2_TASKS = WORKSPACE_SLOTS * 8
FULL_CHUNK_STAGE1_COST = 4930


def _interpolate_stage1_task_cost(
    valid_row_num,
    left_row_num,
    left_cost,
    right_row_num,
    right_cost,
):
    # Fit coefficients use picoseconds; scheduler costs use nanoseconds.
    row_span = right_row_num - left_row_num
    numerator = left_cost * row_span + (valid_row_num - left_row_num) * (right_cost - left_cost)
    denominator = row_span * 1000
    return (numerator + denominator // 2) // denominator


def _stage1_task_cost(valid_row_num):
    if valid_row_num <= 0:
        return 0
    if valid_row_num <= 16:
        return _interpolate_stage1_task_cost(valid_row_num, 1, 4892380, 16, 4920438)
    if valid_row_num <= 32:
        return _interpolate_stage1_task_cost(valid_row_num, 16, 4920438, 32, 4970718)
    if valid_row_num <= 48:
        return _interpolate_stage1_task_cost(valid_row_num, 32, 4970718, 48, 4992347)
    if valid_row_num <= 63:
        return _interpolate_stage1_task_cost(valid_row_num, 48, 4992347, 63, 4952972)
    return FULL_CHUNK_STAGE1_COST


def _stage2_task_cost(candidate_idx, chunk_num):
    item_init = 352
    chunk_cost = 1839
    if candidate_idx == 1:
        item_init = 133
        chunk_cost = 1186
    elif candidate_idx == 2:
        item_init = 19
        chunk_cost = 884
    elif candidate_idx == 3:
        item_init = 65
        chunk_cost = 729
    return item_init + chunk_num * chunk_cost


def round_record_words(active_batch_num: int) -> int:
    """Return the words in one variable record."""

    active_batch_num = int(active_batch_num)
    if active_batch_num <= 0:
        raise ValueError("active_batch_num must be positive")
    return ROUND_FIXED_WORDS + ROUND_WORDS_PER_BATCH * active_batch_num


def metadata_capacity_upper_bound(
    batch: int,
    storage_length: int,
    *,
    is_packed: bool,
) -> int:
    """Return a safe capacity for every legal round decomposition."""

    batch = int(batch)
    storage_length = int(storage_length)
    if batch <= 0:
        raise ValueError("batch must be positive")
    if storage_length <= 0:
        raise ValueError("storage_length must be positive")
    storage_chunk_num = (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    if is_packed:
        chunk_entries = storage_chunk_num + batch - 1
    else:
        chunk_entries = batch * storage_chunk_num
    # One directory word and one single-batch record per possible chunk entry.
    return HEADER_WORDS + 1 + (1 + ROUND_FIXED_WORDS + ROUND_WORDS_PER_BATCH) * chunk_entries


def chunk_capacity_upper_bound(batch: int, storage_length: int) -> int:
    """Return the uniform chunk-slot capacity used by CP intermediates.

    Packed TND sequences share one allocation.  A storage length of ``T``
    tokens needs at most ``ceil(T / 64)`` chunks for one sequence; allowing
    one partial tail for each of the other ``B - 1`` sequences gives the
    common upper bound ``ceil(T / 64) + B - 1``.  CP uses this same capacity
    for BNSD/BSND so one compiled ABI covers every layout.  Metadata still
    records each sequence's exact valid chunk count, and the kernels never
    visit the unused slots.
    """

    batch = int(batch)
    storage_length = int(storage_length)
    if batch <= 0:
        raise ValueError("batch must be positive")
    if storage_length <= 0:
        raise ValueError("storage_length must be positive")
    return (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE + batch - 1


class FlashKDAMetadataArgs:
    cu_seqlens: gm_in(I32)
    metadata: gm_out(U32)
    metadata_word_capacity: U32
    batch: U32
    physical_batches: U32
    storage_length: U32
    value_heads: U32
    has_cu: U32
    is_packed: U32
    core_num: U32
    workspace_slots: U32
    max_chunk_group: U32


def _select_group_num(
    remaining_chunk_num,
    active_batch_num,
    value_heads,
    workspace_slots,
    max_chunk_group,
):
    max_remaining_chunk_num = 0
    for active_idx in range(0, active_batch_num):
        if remaining_chunk_num[active_idx] > max_remaining_chunk_num:
            max_remaining_chunk_num = remaining_chunk_num[active_idx]

    selected_group_num = 1
    lower_group_num = 1
    upper_group_num = max_remaining_chunk_num
    if upper_group_num > max_chunk_group:
        upper_group_num = max_chunk_group
    workspace_group_limit = workspace_slots // value_heads
    if upper_group_num > workspace_group_limit:
        upper_group_num = workspace_group_limit
    # Slot use grows monotonically with group size.  Select the exact largest
    # feasible group so long sequences cross fewer global Stage1/Stage2 barriers.
    while lower_group_num <= upper_group_num:
        candidate_group_num = (lower_group_num + upper_group_num) // 2
        slot_num = 0
        for active_idx in range(0, active_batch_num):
            current_chunk_num = remaining_chunk_num[active_idx]
            if current_chunk_num > candidate_group_num:
                current_chunk_num = candidate_group_num
            slot_num += value_heads * current_chunk_num
        if slot_num <= workspace_slots:
            selected_group_num = candidate_group_num
            lower_group_num = candidate_group_num + 1
        else:
            upper_group_num = candidate_group_num - 1
    return selected_group_num


def _partition_costs(
    task_costs,
    task_num,
    core_num,
    core_ranges,
    core_costs,
):
    for core_idx in range(0, MAX_AIC_CORES):
        core_ranges[core_idx * 2] = 0
        core_ranges[core_idx * 2 + 1] = 0
        core_costs[core_idx] = 0

    if task_num <= 0:
        return

    active_core_num = task_num
    if active_core_num > core_num:
        active_core_num = core_num
    remaining_cost = 0
    for task_idx in range(0, task_num):
        remaining_cost += task_costs[task_idx]

    start = 0
    for core_idx in range(0, active_core_num):
        remaining_core_num = active_core_num - core_idx
        if remaining_core_num == 1:
            end = task_num
        else:
            max_end = task_num - (remaining_core_num - 1)
            end = start + 1
            segment_cost = task_costs[start]
            while end < max_end:
                next_cost = segment_cost + task_costs[end]
                next_error = abs(next_cost * remaining_core_num - remaining_cost)
                current_error = abs(segment_cost * remaining_core_num - remaining_cost)
                if next_error <= current_error:
                    segment_cost = next_cost
                    end += 1
                else:
                    break

        segment_cost = 0
        for task_idx in range(start, end):
            segment_cost += task_costs[task_idx]
        core_ranges[core_idx * 2] = start
        core_ranges[core_idx * 2 + 1] = end
        core_costs[core_idx] = segment_cost
        remaining_cost -= segment_cost
        start = end


def _allocate_metadata_scratch(batches, group, schedule):
    batches.lengths = zeros(I64, MAX_ACTIVE_BATCHES)
    batches.token_starts = zeros(I64, MAX_ACTIVE_BATCHES)
    batches.chunks = zeros(I64, MAX_ACTIVE_BATCHES)
    batches.tail_rows = zeros(I64, MAX_ACTIVE_BATCHES)
    group.local_batches = zeros(I64, MAX_ACTIVE_BATCHES)
    group.remaining_chunks = zeros(I64, MAX_ACTIVE_BATCHES)
    group.chunks_per_batch = zeros(I64, MAX_ACTIVE_BATCHES)
    schedule.stage1_costs = zeros(I64, MAX_STAGE1_TASKS)
    schedule.stage1_ranges = zeros(I64, MAX_AIC_CORES * 2)
    schedule.stage1_core_costs = zeros(I64, MAX_AIC_CORES)
    schedule.stage2_costs = zeros(I64, MAX_STAGE2_TASKS)
    schedule.candidate_ranges = zeros(I64, MAX_AIC_CORES * 2)
    schedule.candidate_core_costs = zeros(I64, MAX_AIC_CORES)
    schedule.selected_ranges = zeros(I64, MAX_AIC_CORES * 2)


def _initialize_metadata_header(a):
    a.metadata[HEADER_MAGIC] = MAGIC
    a.metadata[HEADER_ABI_VERSION] = ABI_VERSION
    a.metadata[HEADER_STATUS] = STATUS_OK
    a.metadata[HEADER_CORE_NUM] = a.core_num
    a.metadata[HEADER_STAGE12_ROUND_NUM] = 0
    a.metadata[HEADER_BATCH] = a.batch
    a.metadata[HEADER_VALUE_HEADS] = a.value_heads
    a.metadata[HEADER_HAS_CU] = a.has_cu
    a.metadata[HEADER_WORKSPACE_SLOTS] = a.workspace_slots
    if a.metadata_word_capacity > HEADER_WORDS:
        a.metadata[HEADER_WORDS] = HEADER_WORDS + 1


def _metadata_flags_status(a):
    if a.max_chunk_group < 1 or a.max_chunk_group > 512:
        return STATUS_BAD_FLAGS
    if not (a.has_cu == 0 or a.has_cu == 1):
        return STATUS_BAD_FLAGS
    if not (a.is_packed == 0 or a.is_packed == 1):
        return STATUS_BAD_FLAGS
    if a.is_packed == 1 and a.has_cu == 0:
        return STATUS_BAD_FLAGS
    return STATUS_OK


def _metadata_argument_status(a):
    if a.core_num < 1 or a.core_num > MAX_AIC_CORES:
        return STATUS_BAD_CORE_NUM
    if a.workspace_slots != WORKSPACE_SLOTS:
        return STATUS_BAD_WORKSPACE_SLOTS
    status = _metadata_flags_status(a)
    if status != STATUS_OK:
        return status
    if a.batch < 1 or a.storage_length < 1 or a.value_heads < 1:
        return STATUS_BAD_SHAPE
    if a.value_heads > a.workspace_slots:
        return STATUS_BAD_SHAPE
    if a.is_packed == 1 and a.physical_batches != 1:
        return STATUS_BAD_SHAPE
    if a.is_packed == 0 and a.physical_batches != a.batch:
        return STATUS_BAD_SHAPE
    if a.has_cu == 1:
        if a.cu_seqlens[0] != 0:
            return STATUS_BAD_CU_START
    return STATUS_OK


def _load_batch_set(a, batches, schedule):
    for local_batch_idx in range(0, batches.count):
        batch_idx = batches.start + local_batch_idx
        token_start = 0
        valid_seq_len = a.storage_length
        if a.has_cu == 1:
            token_start = a.cu_seqlens[batch_idx]
            token_end = a.cu_seqlens[batch_idx + 1]
            valid_seq_len = token_end - token_start
        batches.lengths[local_batch_idx] = valid_seq_len
        batches.token_starts[local_batch_idx] = token_start
        if valid_seq_len <= 0:
            schedule.status = STATUS_BAD_LENGTH
            batches.chunks[local_batch_idx] = 0
            batches.tail_rows[local_batch_idx] = 0
        elif a.is_packed == 0 and valid_seq_len > a.storage_length:
            schedule.status = STATUS_BAD_PADDED_LENGTH
            batches.chunks[local_batch_idx] = 0
            batches.tail_rows[local_batch_idx] = 0
        else:
            batches.chunks[local_batch_idx] = (valid_seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
            batches.tail_rows[local_batch_idx] = ((valid_seq_len - 1) % CHUNK_SIZE) + 1


def _collect_active_batches(batches, group):
    group.count = 0
    for local_batch_idx in range(0, batches.count):
        if batches.chunks[local_batch_idx] > group.chunk_start:
            group.local_batches[group.count] = local_batch_idx
            group.remaining_chunks[group.count] = batches.chunks[local_batch_idx] - group.chunk_start
            group.count += 1
    return group.count


def _choose_chunk_group(a, group):
    group.size = _select_group_num(
        group.remaining_chunks,
        group.count,
        a.value_heads,
        a.workspace_slots,
        a.max_chunk_group,
    )
    for active_idx in range(0, group.count):
        group.chunks_per_batch[active_idx] = min(group.remaining_chunks[active_idx], group.size)


def _count_metadata_rounds(a, batches, group, schedule):
    batch_num_per_set = a.workspace_slots // a.value_heads
    batches.start = 0
    while batches.start < a.batch:
        batches.count = min(a.batch - batches.start, batch_num_per_set)
        _load_batch_set(a, batches, schedule)
        group.chunk_start = 0
        while _collect_active_batches(batches, group) > 0:
            _choose_chunk_group(a, group)
            schedule.round_count += 1
            schedule.active_entries += group.count
            group.chunk_start += group.size
        batches.start += batches.count
    if schedule.status == STATUS_OK and a.is_packed == 1:
        if a.cu_seqlens[a.batch] != a.storage_length:
            schedule.status = STATUS_BAD_PACKED_END


def _write_batch_fields(a, batches, group, round_idx, cursor):
    a.metadata[cursor] = round_idx
    a.metadata[cursor + 1] = group.count
    cursor += 2
    for active_idx in range(0, group.count):
        a.metadata[cursor + active_idx] = batches.start + group.local_batches[active_idx]
    cursor += group.count
    for active_idx in range(0, group.count):
        logical_batch_idx = batches.start + group.local_batches[active_idx]
        if a.is_packed == 1:
            a.metadata[cursor + active_idx] = 0
        else:
            a.metadata[cursor + active_idx] = logical_batch_idx
    cursor += group.count
    for active_idx in range(0, group.count):
        local_batch_idx = group.local_batches[active_idx]
        if a.is_packed == 1:
            a.metadata[cursor + active_idx] = batches.token_starts[local_batch_idx]
        else:
            a.metadata[cursor + active_idx] = 0
    cursor += group.count
    for active_idx in range(0, group.count):
        local_batch_idx = group.local_batches[active_idx]
        a.metadata[cursor + active_idx] = batches.lengths[local_batch_idx]
    cursor += group.count
    a.metadata[cursor] = group.size
    cursor += 1
    for active_idx in range(0, group.count):
        a.metadata[cursor + active_idx] = group.chunk_start
    cursor += group.count
    for active_idx in range(0, group.count):
        a.metadata[cursor + active_idx] = group.chunks_per_batch[active_idx]
    cursor += group.count

    return cursor


def _fill_stage1_costs(a, batches, group, schedule):
    stage1_task_idx = 0
    for active_idx in range(0, group.count):
        local_batch_idx = group.local_batches[active_idx]
        for _ in range(0, a.value_heads):
            for local_chunk_idx in range(0, group.chunks_per_batch[active_idx]):
                global_chunk_idx = group.chunk_start + local_chunk_idx
                stage1_cost = FULL_CHUNK_STAGE1_COST
                if (
                    global_chunk_idx + 1 == batches.chunks[local_batch_idx]
                    and batches.tail_rows[local_batch_idx] < CHUNK_SIZE
                ):
                    stage1_cost = _stage1_task_cost(batches.tail_rows[local_batch_idx])
                schedule.stage1_costs[stage1_task_idx] = stage1_cost
                stage1_task_idx += 1


def _write_stage1_partition(a, batches, group, schedule, cursor):
    stage1_task_num = 0
    a.metadata[cursor] = 0
    for active_idx in range(0, group.count):
        stage1_task_num += a.value_heads * group.chunks_per_batch[active_idx]
        a.metadata[cursor + active_idx + 1] = stage1_task_num
    cursor += group.count + 1

    _fill_stage1_costs(a, batches, group, schedule)
    _partition_costs(
        schedule.stage1_costs,
        stage1_task_num,
        a.core_num,
        schedule.stage1_ranges,
        schedule.stage1_core_costs,
    )
    for range_idx in range(0, MAX_AIC_CORES * 2):
        a.metadata[cursor + range_idx] = schedule.stage1_ranges[range_idx]
    cursor += MAX_AIC_CORES * 2

    return cursor


def _stage2_candidate_is_better(schedule, max_core_cost, total_cost, task_num, dv_splits_num):
    if schedule.selected_task_num == 0:
        return 1
    if max_core_cost != schedule.selected_peak_cost:
        return int(max_core_cost < schedule.selected_peak_cost)
    if total_cost != schedule.selected_total_cost:
        return int(total_cost < schedule.selected_total_cost)
    if task_num != schedule.selected_task_num:
        return int(task_num < schedule.selected_task_num)
    return int(dv_splits_num < schedule.selected_splits)


def _select_stage2_partition(a, group, schedule):
    schedule.selected_splits = 1
    schedule.selected_peak_cost = 0
    schedule.selected_total_cost = 0
    schedule.selected_task_num = 0
    for candidate_idx in range(0, 4):
        candidate_dv_splits_num = 1 << candidate_idx
        candidate_task_num = 0
        candidate_total_cost = 0
        for active_idx in range(0, group.count):
            task_cost = _stage2_task_cost(
                candidate_idx,
                group.chunks_per_batch[active_idx],
            )
            batch_task_num = a.value_heads * candidate_dv_splits_num
            for _ in range(0, batch_task_num):
                schedule.stage2_costs[candidate_task_num] = task_cost
                candidate_task_num += 1
                candidate_total_cost += task_cost
        _partition_costs(
            schedule.stage2_costs,
            candidate_task_num,
            a.core_num,
            schedule.candidate_ranges,
            schedule.candidate_core_costs,
        )
        candidate_max_core_cost = 0
        for core_idx in range(0, MAX_AIC_CORES):
            if schedule.candidate_core_costs[core_idx] > candidate_max_core_cost:
                candidate_max_core_cost = schedule.candidate_core_costs[core_idx]

        candidate_is_better = _stage2_candidate_is_better(
            schedule,
            candidate_max_core_cost,
            candidate_total_cost,
            candidate_task_num,
            candidate_dv_splits_num,
        )
        if candidate_is_better == 1:
            schedule.selected_splits = candidate_dv_splits_num
            schedule.selected_peak_cost = candidate_max_core_cost
            schedule.selected_total_cost = candidate_total_cost
            schedule.selected_task_num = candidate_task_num
            for range_idx in range(0, MAX_AIC_CORES * 2):
                schedule.selected_ranges[range_idx] = schedule.candidate_ranges[range_idx]


def _write_stage2_partition(a, group, schedule, cursor):
    _select_stage2_partition(a, group, schedule)
    for active_idx in range(0, group.count):
        a.metadata[cursor + active_idx] = schedule.selected_splits
    cursor += group.count
    stage2_task_num = 0
    a.metadata[cursor] = 0
    for active_idx in range(0, group.count):
        stage2_task_num += a.value_heads * schedule.selected_splits
        a.metadata[cursor + active_idx + 1] = stage2_task_num
    cursor += group.count + 1
    for range_idx in range(0, MAX_AIC_CORES * 2):
        a.metadata[cursor + range_idx] = schedule.selected_ranges[range_idx]
    cursor += MAX_AIC_CORES * 2

    return cursor


def _write_metadata_rounds(a, batches, group, schedule):
    batch_num_per_set = a.workspace_slots // a.value_heads
    batches.start = 0
    while batches.start < a.batch:
        batches.count = min(a.batch - batches.start, batch_num_per_set)
        _load_batch_set(a, batches, schedule)
        group.chunk_start = 0
        while _collect_active_batches(batches, group) > 0:
            _choose_chunk_group(a, group)
            a.metadata[HEADER_WORDS + schedule.round_index] = schedule.record_cursor
            cursor = _write_batch_fields(a, batches, group, schedule.round_index, schedule.record_cursor)
            cursor = _write_stage1_partition(a, batches, group, schedule, cursor)
            schedule.record_cursor = _write_stage2_partition(a, group, schedule, cursor)
            schedule.round_index += 1
            group.chunk_start += group.size
        batches.start += batches.count
    a.metadata[HEADER_WORDS + schedule.round_count] = schedule.record_cursor


@aicpu_kernel
def flash_kda_metadata_kernel(a: FlashKDAMetadataArgs):
    # AICPU supports annotation-only local structs, including array fields.
    class BatchSet:
        start: I64
        count: I64
        lengths: array(I64)
        token_starts: array(I64)
        chunks: array(I64)
        tail_rows: array(I64)

    class ChunkGroup:
        chunk_start: I64
        count: I64
        size: I64
        local_batches: array(I64)
        remaining_chunks: array(I64)
        chunks_per_batch: array(I64)

    class Schedule:
        status: I64
        round_count: I64
        active_entries: I64
        round_index: I64
        record_cursor: I64
        selected_splits: I64
        selected_peak_cost: I64
        selected_total_cost: I64
        selected_task_num: I64
        stage1_costs: array(I64)
        stage1_ranges: array(I64)
        stage1_core_costs: array(I64)
        stage2_costs: array(I64)
        candidate_ranges: array(I64)
        candidate_core_costs: array(I64)
        selected_ranges: array(I64)

    batches = BatchSet()
    group = ChunkGroup()
    schedule = Schedule()
    _allocate_metadata_scratch(batches, group, schedule)
    if a.metadata_word_capacity < HEADER_WORDS:
        return STATUS_BAD_METADATA_CAPACITY
    _initialize_metadata_header(a)
    schedule.status = _metadata_argument_status(a)
    if schedule.status != STATUS_OK:
        a.metadata[HEADER_STATUS] = schedule.status
        return schedule.status
    _count_metadata_rounds(a, batches, group, schedule)
    if schedule.status != STATUS_OK:
        a.metadata[HEADER_STATUS] = schedule.status
        return schedule.status
    schedule.record_cursor = HEADER_WORDS + schedule.round_count + 1
    used_words = (
        schedule.record_cursor
        + ROUND_FIXED_WORDS * schedule.round_count
        + ROUND_WORDS_PER_BATCH * schedule.active_entries
    )
    if used_words > a.metadata_word_capacity:
        a.metadata[HEADER_STATUS] = STATUS_BAD_METADATA_CAPACITY
        return STATUS_BAD_METADATA_CAPACITY
    a.metadata[HEADER_STAGE12_ROUND_NUM] = schedule.round_count
    _write_metadata_rounds(a, batches, group, schedule)
    return STATUS_OK


def _take(words: list[int], cursor: int, count: int) -> tuple[list[int], int]:
    end = cursor + count
    if end > len(words):
        raise ValueError("metadata record extends past available words")
    return words[cursor:end], end


def _decode_ranges(flat: list[int]) -> list[tuple[int, int]]:
    if len(flat) != MAX_AIC_CORES * 2:
        raise ValueError("core range block must contain 64 words")
    return [(flat[index], flat[index + 1]) for index in range(0, len(flat), 2)]


def _validate_prefix(prefix: list[int], name: str) -> None:
    if not prefix or prefix[0] != 0:
        raise ValueError(f"{name} must start at zero")
    if any(end < start for start, end in zip(prefix, prefix[1:])):
        raise ValueError(f"{name} must be nondecreasing")


def _validate_ranges(ranges: list[tuple[int, int]], total: int, name: str) -> None:
    cursor = 0
    saw_empty = False
    for start, end in ranges:
        if (start, end) == (0, 0):
            saw_empty = True
            continue
        if saw_empty:
            raise ValueError(f"{name} has work after an unused core")
        if start != cursor or end <= start or end > total:
            raise ValueError(f"{name} is not a contiguous partition")
        cursor = end
    if cursor != total:
        raise ValueError(f"{name} does not cover its task total")


def _validate_round_values(round_data: dict[str, object], batch: int) -> None:
    if round_data["group_num"] <= 0:
        raise ValueError("group_num must be positive")
    if any(index >= batch for index in round_data["batch_idx"]):
        raise ValueError("batch_idx is outside logical batch")
    if any(length <= 0 for length in round_data["valid_seq_len"]):
        raise ValueError("valid_seq_len must be positive")
    if any(count <= 0 for count in round_data["chunk_num_per_group"]):
        raise ValueError("chunk_num_per_group must be positive")
    if any(value not in (1, 2, 4, 8) for value in round_data["dv_splits_num"]):
        raise ValueError("dv_splits_num must be 1, 2, 4, or 8")


def _decode_round(
    words: list[int],
    start: int,
    end: int,
    *,
    batch: int,
    value_heads: int,
) -> dict[str, object]:
    if start + 2 > end:
        raise ValueError("metadata round record is truncated")
    cursor = start
    header_end = cursor + 2
    stage12_round_idx, active_batch_num = words[cursor:header_end]
    cursor = header_end
    if start + round_record_words(active_batch_num) != end:
        raise ValueError("metadata round record length does not match active batches")

    batch_idx, cursor = _take(words, cursor, active_batch_num)
    storage_batch_idx, cursor = _take(words, cursor, active_batch_num)
    token_start, cursor = _take(words, cursor, active_batch_num)
    valid_seq_len, cursor = _take(words, cursor, active_batch_num)
    group_num = words[cursor]
    cursor += 1
    chunk_start_per_group, cursor = _take(words, cursor, active_batch_num)
    chunk_num_per_group, cursor = _take(words, cursor, active_batch_num)
    stage1_task_prefix, cursor = _take(words, cursor, active_batch_num + 1)
    stage1_ranges_flat, cursor = _take(words, cursor, MAX_AIC_CORES * 2)
    dv_splits_num, cursor = _take(words, cursor, active_batch_num)
    stage2_task_prefix, cursor = _take(words, cursor, active_batch_num + 1)
    stage2_ranges_flat, cursor = _take(words, cursor, MAX_AIC_CORES * 2)
    if cursor != end:
        raise ValueError("metadata round decoder did not consume the full record")

    _validate_round_values(
        {
            "group_num": group_num,
            "batch_idx": batch_idx,
            "valid_seq_len": valid_seq_len,
            "chunk_num_per_group": chunk_num_per_group,
            "dv_splits_num": dv_splits_num,
        },
        batch,
    )

    _validate_prefix(stage1_task_prefix, "stage1_task_prefix")
    _validate_prefix(stage2_task_prefix, "stage2_task_prefix")
    for index, chunk_num in enumerate(chunk_num_per_group):
        if stage1_task_prefix[index + 1] - stage1_task_prefix[index] != (value_heads * chunk_num):
            raise ValueError("stage1_task_prefix does not match chunk tasks")
        if stage2_task_prefix[index + 1] - stage2_task_prefix[index] != (value_heads * dv_splits_num[index]):
            raise ValueError("stage2_task_prefix does not match DV tasks")
    if stage1_task_prefix[-1] > WORKSPACE_SLOTS:
        raise ValueError("stage1 task total exceeds workspace slots")

    stage1_core_ranges = _decode_ranges(stage1_ranges_flat)
    stage2_core_ranges = _decode_ranges(stage2_ranges_flat)
    _validate_ranges(stage1_core_ranges, stage1_task_prefix[-1], "stage1_core_ranges")
    _validate_ranges(stage2_core_ranges, stage2_task_prefix[-1], "stage2_core_ranges")
    return {
        "stage12_round_idx": stage12_round_idx,
        "active_batch_num": active_batch_num,
        "batch_idx": batch_idx,
        "storage_batch_idx": storage_batch_idx,
        "token_start": token_start,
        "valid_seq_len": valid_seq_len,
        "group_num": group_num,
        "chunk_start_per_group": chunk_start_per_group,
        "chunk_num_per_group": chunk_num_per_group,
        "stage1_task_prefix": stage1_task_prefix,
        "stage1_core_ranges": stage1_core_ranges,
        "dv_splits_num": dv_splits_num,
        "stage2_task_prefix": stage2_task_prefix,
        "stage2_core_ranges": stage2_core_ranges,
    }


def _decode_metadata_offsets(words: list[int], stage12_round_num: int) -> list[int]:
    offset_begin = HEADER_WORDS
    offset_end = offset_begin + stage12_round_num + 1
    if offset_end > len(words):
        raise ValueError("metadata offset table is truncated")
    offsets = words[offset_begin:offset_end]
    if offsets[0] < offset_end:
        raise ValueError("metadata first record offset overlaps the offset table")
    if offsets[-1] > len(words):
        raise ValueError("metadata final offset exceeds capacity")
    if any(end <= start for start, end in zip(offsets, offsets[1:])):
        raise ValueError("metadata offsets must be strictly increasing")

    return offsets


def decode_metadata(raw: Sequence[int]) -> dict[str, object]:
    """Decode and validate metadata for tests and diagnostics."""

    words = [int(value) for value in raw]
    if len(words) < HEADER_WORDS:
        raise ValueError("metadata is shorter than its header")
    if words[HEADER_MAGIC] != MAGIC:
        raise ValueError("metadata magic does not match")
    if words[HEADER_ABI_VERSION] != ABI_VERSION:
        raise ValueError("metadata ABI version does not match")

    status = words[HEADER_STATUS]
    header = {
        "magic": words[HEADER_MAGIC],
        "abi_version": words[HEADER_ABI_VERSION],
        "status": status,
        "core_num": words[HEADER_CORE_NUM],
        "stage12_round_num": words[HEADER_STAGE12_ROUND_NUM],
        "batch": words[HEADER_BATCH],
        "value_heads": words[HEADER_VALUE_HEADS],
        "has_cu": words[HEADER_HAS_CU],
        "workspace_slots": words[HEADER_WORKSPACE_SLOTS],
    }
    if status != STATUS_OK:
        header["used_words"] = HEADER_WORDS
        header["stage12_rounds"] = []
        return header
    if not 1 <= header["core_num"] <= MAX_AIC_CORES:
        raise ValueError(f"metadata core_num must be in [1, {MAX_AIC_CORES}]")
    if header["workspace_slots"] != WORKSPACE_SLOTS:
        raise ValueError(f"metadata workspace_slots must be {WORKSPACE_SLOTS}")
    if header["batch"] <= 0 or header["value_heads"] <= 0:
        raise ValueError("successful metadata requires positive shapes")

    stage12_round_num = int(header["stage12_round_num"])
    offsets = _decode_metadata_offsets(words, stage12_round_num)

    header["used_words"] = offsets[-1]
    header["stage12_rounds"] = [
        _decode_round(
            words,
            offsets[index],
            offsets[index + 1],
            batch=int(header["batch"]),
            value_heads=int(header["value_heads"]),
        )
        for index in range(stage12_round_num)
    ]
    return header


SUPPORTED_HEAD_DIM = 128
LOW_DTYPE = torch.bfloat16
HIGH_DTYPE = torch.float32


@dataclass
class _AicpuCache:
    executable: object | None = None
    workdir: tempfile.TemporaryDirectory | None = None


_AICPU_CACHE = _AicpuCache()
_AICPU_LOCK = threading.Lock()


def _device_block_num(ref: Optional[torch.Tensor] = None) -> int:
    """Return a grid-barrier-safe launch size for the current NPU stream."""
    if ref is not None and ref.device.type not in {"npu", "privateuseone"}:
        return MAX_AIC_CORES
    npu = getattr(torch, "npu", None)
    if npu is None:
        return MAX_AIC_CORES
    device_index = ref.device.index if ref is not None else None
    if device_index is None:
        try:
            device_index = npu.current_device()
        except RuntimeError:
            return MAX_AIC_CORES
    stream = npu.current_stream(device_index)
    platform_info = get_platform_info(stream=stream)
    try:
        cube_core_num = int(platform_info.cube_core_num)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"NPU {device_index} does not expose a valid cube_core_num") from exc
    if cube_core_num <= 0:
        raise RuntimeError(f"NPU {device_index} reports invalid cube_core_num={cube_core_num}")
    return min(cube_core_num, MAX_AIC_CORES)


def _get_aicpu_kernel():
    with _AICPU_LOCK:
        if _AICPU_CACHE.executable is None:
            # The executable loads its shared library lazily on the first launch.
            workdir = tempfile.TemporaryDirectory(prefix="flash_kda_metadata_aicpu_")
            try:
                compiled = compile_aicpu_kernel(
                    flash_kda_metadata_kernel, workdir=workdir.name, launch_mode="interface"
                )
            except BaseException:
                workdir.cleanup()
                raise
            _AICPU_CACHE.workdir = workdir
            _AICPU_CACHE.executable = compiled
        return _AICPU_CACHE.executable


@dataclass(frozen=True)
class _MetadataShape:
    batch: int
    value_heads: int
    key_heads: int
    head_dim: int
    physical_batches: int
    storage_length: int
    is_packed: bool


def _metadata_shape(q, v, initial_state, layout_qkv, cu_seqlens):
    if layout_qkv not in ("TND", "BNSD", "BSND"):
        raise ValueError(f"layout_qkv must be TND, BNSD, or BSND, got {layout_qkv!r}")
    if initial_state.dim() != 4:
        raise ValueError("initial_state must be rank-4")
    batch, value_heads, state_dv, state_dk = initial_state.shape
    if state_dv != SUPPORTED_HEAD_DIM or state_dk != SUPPORTED_HEAD_DIM:
        raise ValueError(f"FlashKDA only supports D={SUPPORTED_HEAD_DIM}")
    _validate_metadata_cu_seqlens(q, cu_seqlens, batch, layout_qkv == "TND")
    rank = 3 if layout_qkv == "TND" else 4
    if q.dim() != rank or v.dim() != rank:
        raise ValueError(f"{layout_qkv} q/v must be rank-{rank}")
    if layout_qkv == "TND":
        storage_length, key_heads, head_dim = q.shape
        physical_batches = 1
        expected_value_shape = (storage_length, value_heads, head_dim)
    elif layout_qkv == "BNSD":
        physical_batches, key_heads, storage_length, head_dim = q.shape
        expected_value_shape = (physical_batches, value_heads, storage_length, head_dim)
    else:
        physical_batches, storage_length, key_heads, head_dim = q.shape
        expected_value_shape = (physical_batches, storage_length, value_heads, head_dim)
    if v.shape != expected_value_shape:
        raise ValueError(f"v shape does not match {layout_qkv} layout")
    if layout_qkv != "TND" and physical_batches != batch:
        raise ValueError("padded storage batch must match initial_state")
    return _MetadataShape(
        batch, value_heads, key_heads, head_dim, physical_batches, storage_length, layout_qkv == "TND"
    )


def _validate_metadata_tensors(q, v, initial_state, shape):
    if shape.head_dim != SUPPORTED_HEAD_DIM:
        raise ValueError(f"FlashKDA only supports D={SUPPORTED_HEAD_DIM}")
    if shape.value_heads % shape.key_heads != 0:
        raise ValueError(f"GQA requires Nv % Nqk == 0, got Nv={shape.value_heads}, Nqk={shape.key_heads}")
    if shape.storage_length <= 0:
        raise ValueError("storage sequence length must be positive")
    if q.dtype != LOW_DTYPE or v.dtype != LOW_DTYPE:
        raise ValueError("q/v must be bf16")
    if initial_state.dtype != HIGH_DTYPE:
        raise ValueError("initial_state must be fp32")
    if q.device != v.device or q.device != initial_state.device:
        raise ValueError("all inputs must be on the same device")
    if not q.is_contiguous() or not v.is_contiguous() or not initial_state.is_contiguous():
        raise ValueError("all public inputs must be contiguous")


def _validate_metadata_cu_seqlens(q, cu_seqlens, batch, is_packed):
    if cu_seqlens is None:
        if is_packed:
            raise ValueError("cu_seqlens is required for TND")
        return
    if cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must be int32")
    if cu_seqlens.dim() != 1:
        raise ValueError("cu_seqlens must be rank-1")
    if cu_seqlens.shape != (batch + 1,):
        raise ValueError(f"cu_seqlens shape must be ({batch + 1},)")
    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous")
    if cu_seqlens.device != q.device:
        raise ValueError("all inputs must be on the same device")


def flash_kda_metadata(
    q: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    layout_qkv: str,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_chunk_group: int = 512,
) -> torch.Tensor:
    """Build reusable FlashKDA scheduling metadata on the current stream."""
    shape = _metadata_shape(q, v, initial_state, layout_qkv, cu_seqlens)
    _validate_metadata_tensors(q, v, initial_state, shape)
    max_chunk_group = int(max_chunk_group)
    if not 1 <= max_chunk_group <= 512:
        raise ValueError("max_chunk_group must be in [1, 512]")
    core_num = _device_block_num(q)
    metadata_capacity = metadata_capacity_upper_bound(shape.batch, shape.storage_length, is_packed=shape.is_packed)
    metadata = torch.zeros(metadata_capacity, dtype=torch.int32, device=q.device)
    aicpu = _get_aicpu_kernel()
    device_id = q.get_device() if hasattr(q, "get_device") and q.device.type != "cpu" else 0
    stream = current_raw_stream(device_id)
    aicpu.launch(
        stream,
        cu_seqlens=0 if cu_seqlens is None else cu_seqlens.data_ptr(),
        metadata=metadata.data_ptr(),
        metadata_word_capacity=metadata.numel(),
        batch=shape.batch,
        physical_batches=shape.physical_batches,
        storage_length=shape.storage_length,
        value_heads=shape.value_heads,
        has_cu=int(cu_seqlens is not None),
        is_packed=int(shape.is_packed),
        core_num=core_num,
        workspace_slots=WORKSPACE_SLOTS,
        max_chunk_group=max_chunk_group,
    )
    return metadata


def clear_metadata_caches():
    """Drop the compiled metadata executable and its compilation directory."""
    with _AICPU_LOCK:
        try:
            if _AICPU_CACHE.executable is not None:
                close = getattr(_AICPU_CACHE.executable, "close", None)
                if callable(close):
                    close()
        finally:
            _AICPU_CACHE.executable = None
            if _AICPU_CACHE.workdir is not None:
                _AICPU_CACHE.workdir.cleanup()
                _AICPU_CACHE.workdir = None


__all__ = [
    "ABI_VERSION",
    "FlashKDAMetadataArgs",
    "HEADER_ABI_VERSION",
    "HEADER_BATCH",
    "HEADER_CORE_NUM",
    "HEADER_HAS_CU",
    "HEADER_MAGIC",
    "HEADER_STAGE12_ROUND_NUM",
    "HEADER_STATUS",
    "HEADER_VALUE_HEADS",
    "HEADER_WORKSPACE_SLOTS",
    "HEADER_WORDS",
    "MAGIC",
    "MAX_AIC_CORES",
    "STATUS_BAD_CORE_NUM",
    "STATUS_BAD_CU_START",
    "STATUS_BAD_FLAGS",
    "STATUS_BAD_LENGTH",
    "STATUS_BAD_METADATA_CAPACITY",
    "STATUS_BAD_PACKED_END",
    "STATUS_BAD_PADDED_LENGTH",
    "STATUS_BAD_SHAPE",
    "STATUS_BAD_WORKSPACE_SLOTS",
    "STATUS_OK",
    "WORKSPACE_SLOTS",
    "CHUNK_SIZE",
    "clear_metadata_caches",
    "decode_metadata",
    "flash_kda_metadata",
    "flash_kda_metadata_kernel",
    "metadata_capacity_upper_bound",
    "chunk_capacity_upper_bound",
    "round_record_words",
]


def flash_kda_cp_main_metadata(
    qg: torch.Tensor,
    local_initial_state: torch.Tensor,
    layout_qkv: str,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the main-local stage12w schedule."""
    if local_initial_state.dim() != 4:
        raise ValueError("local_initial_state must be rank-4")
    batch, value_heads, state_dv, state_dk = local_initial_state.shape
    if state_dv != SUPPORTED_HEAD_DIM or state_dk != SUPPORTED_HEAD_DIM:
        raise ValueError(f"FlashKDA only supports D={SUPPORTED_HEAD_DIM}")
    if layout_qkv == "TND":
        if qg.dim() != 3 or qg.shape[1] != value_heads:
            raise ValueError("TND qg must be rank-3 with the initial_state value heads")
        if cu_seqlens is None or cu_seqlens.numel() != batch + 1:
            raise ValueError(f"TND cu_seqlens must contain {batch + 1} entries")
    elif layout_qkv in ("BNSD", "BSND"):
        if qg.dim() != 4 or qg.shape[:2] != (batch, value_heads):
            raise ValueError(f"{layout_qkv} qg must be rank-4 with matching batch and value heads")
    else:
        raise ValueError(f"unsupported layout_qkv={layout_qkv!r}")
    metadata_layout = "BNSD" if layout_qkv == "BSND" else layout_qkv
    return flash_kda_metadata(
        qg,
        qg,
        local_initial_state,
        metadata_layout,
        cu_seqlens,
        max_chunk_group=_max_chunk_group(qg, layout_qkv),
    )


# Each CP entry owns its AICPU executable and cache.
flash_kda_cp_main_metadata_kernel = flash_kda_metadata_kernel
clear_main_metadata_caches = clear_metadata_caches

_L2_BYTES = 128 * 1024 * 1024
_L2_WORKING_SET_BYTES = _L2_BYTES * 3 // 4
_CHUNK_BYTES = 4 * 64 * 128 * 2 + 64 * 64 * 2 + 128 * 4 + 64 * 128 * 2
_STATE_BOUNDARY_BYTES = 2 * 128 * 128 * 4


def _l2_group_limit(batch: int, value_heads: int) -> int:
    active_head_num = max(1, int(batch) * int(value_heads))
    state_bytes = active_head_num * _STATE_BOUNDARY_BYTES
    if state_bytes >= _L2_WORKING_SET_BYTES:
        return 1
    per_group_bytes = active_head_num * _CHUNK_BYTES
    l2_group_limit = (_L2_WORKING_SET_BYTES - state_bytes) // per_group_bytes
    workspace_group_limit = WORKSPACE_SLOTS // active_head_num
    return max(1, min(512, l2_group_limit, workspace_group_limit))


def _max_chunk_group(qg: torch.Tensor, layout_qkv: str) -> int:
    if layout_qkv == "BNSD" and qg.dim() == 4:
        if qg.shape[:2] == (1, 32) and qg.shape[2] >= 8192 and qg.shape[2] % 64 == 0:
            if _device_block_num(qg) == 32:
                return _l2_group_limit(qg.shape[0], qg.shape[1])
    return 512


__all__ += [
    "flash_kda_cp_main_metadata",
    "flash_kda_cp_main_metadata_kernel",
    "clear_main_metadata_caches",
]
