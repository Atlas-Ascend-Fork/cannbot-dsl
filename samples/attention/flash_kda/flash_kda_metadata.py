# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# fmt: off
# Keep compact call argument groups consistent with the original FlashKDA samples.

"""Cost-aware AICPU metadata producer for FlashKDA."""

__all__ = [
    "HEADER_TEMPLATE",
    "HEADER_CHUNK_TASK_NUM",
    "TEMPLATE_AUTO",
    "TEMPLATE_STAGED",
    "TEMPLATE_PERSISTENT",
    "CHUNK_TASK_WORDS",
    "CHUNK_TASK_BATCH",
    "CHUNK_TASK_STORAGE_BATCH",
    "CHUNK_TASK_TOKEN_START",
    "CHUNK_TASK_LENGTH",
    "CHUNK_TASK_VALUE_HEAD",
    "CHUNK_TASK_NEXT",
    "ABI_VERSION",
    "FlashKDAMetadataArgs",
    "HEADER_CORE_NUM",
    "HEADER_STAGE12_ROUND_NUM",
    "HEADER_STATUS",
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
    "clear_metadata_caches",
    "decode_metadata",
    "flash_kda_metadata",
    "flash_kda_metadata_kernel",
    "metadata_capacity_upper_bound",
    "round_record_words",
]

import os
import tempfile
import threading
from collections.abc import Sequence
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
ABI_VERSION = 4
MAX_AIC_CORES = 32
WORKSPACE_SLOTS = 861
CHUNK_SIZE = 64
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
HEADER_TEMPLATE = 9
HEADER_CHUNK_TASK_NUM = 10
HEADER_WORDS = 11

TEMPLATE_AUTO = 0
TEMPLATE_STAGED = 1
TEMPLATE_PERSISTENT = 2
CHUNK_TASK_WORDS = 6
CHUNK_TASK_BATCH = 0
CHUNK_TASK_STORAGE_BATCH = 1
CHUNK_TASK_TOKEN_START = 2
CHUNK_TASK_LENGTH = 3
CHUNK_TASK_VALUE_HEAD = 4
CHUNK_TASK_NEXT = 5


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


def _interpolate_stage1_task_cost(
    valid_row_num,
    left_row_num,
    left_cost,
    right_row_num,
    right_cost,
):
    # Fit coefficients use picoseconds; scheduler costs use nanoseconds.
    row_span = right_row_num - left_row_num
    numerator = left_cost * row_span + (valid_row_num - left_row_num) * (
        right_cost - left_cost
    )
    denominator = row_span * 1000
    return (numerator + denominator // 2) // denominator


def _stage1_task_cost(valid_row_num):
    cost = 0
    if valid_row_num > 0:
        if valid_row_num <= 16:
            cost = _interpolate_stage1_task_cost(valid_row_num, 1, 4892380, 16, 4920438)
        elif valid_row_num <= 32:
            cost = _interpolate_stage1_task_cost(valid_row_num, 16, 4920438, 32, 4970718)
        elif valid_row_num <= 48:
            cost = _interpolate_stage1_task_cost(valid_row_num, 32, 4970718, 48, 4992347)
        elif valid_row_num <= 63:
            cost = _interpolate_stage1_task_cost(valid_row_num, 48, 4992347, 63, 4952972)
        else:
            cost = 4930
    return cost


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
    value_heads: int = WORKSPACE_SLOTS,
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
    if not 1 <= int(value_heads) <= WORKSPACE_SLOTS:
        raise ValueError("value_heads must be in [1, 861]")
    # One directory word and one single-batch record per possible chunk entry.
    group_capacity = (
        HEADER_WORDS
        + 1
        + (1 + ROUND_FIXED_WORDS + ROUND_WORDS_PER_BATCH) * chunk_entries
    )
    # The chunk scheduler uses B temporary batch indices after the task records.
    chunk_capacity = (
        HEADER_WORDS
        + MAX_AIC_CORES
        + CHUNK_TASK_WORDS * batch * int(value_heads)
        + batch
    )
    return max(group_capacity, chunk_capacity)


class FlashKDAMetadataArgs:
    cu_seqlens: gm_in(I32)
    route_coefficients: gm_in(I64)
    calibration_enabled: U32
    layout_kind: U32
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
    template_kind: U32


def _select_group_num(
    remaining_chunk_num,
    active_batch_num,
    value_heads,
    workspace_slots,
):
    max_remaining_chunk_num = 0
    for active_idx in range(0, active_batch_num):
        if remaining_chunk_num[active_idx] > max_remaining_chunk_num:
            max_remaining_chunk_num = remaining_chunk_num[active_idx]

    selected_group_num = 1
    lower_group_num = 1
    upper_group_num = max_remaining_chunk_num
    if upper_group_num > 512:
        upper_group_num = 512
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

    if task_num > 0:
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


def _chunk_batch_precedes(cu_seqlens, left_batch, right_batch):
    left_length = cu_seqlens[left_batch + 1] - cu_seqlens[left_batch]
    right_length = cu_seqlens[right_batch + 1] - cu_seqlens[right_batch]
    left_chunks = (left_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    right_chunks = (right_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    precedes = 0
    if left_chunks > right_chunks:
        precedes = 1
    elif left_chunks == right_chunks and left_batch < right_batch:
        precedes = 1
    return precedes


def _chunk_heap_sift_down(cu_seqlens, metadata, heap_base, heap_num, root_idx):
    # A max heap orders by chunk count, then by the original batch index.
    while root_idx * 2 + 1 < heap_num:
        child_idx = root_idx * 2 + 1
        if child_idx + 1 < heap_num:
            if (
                _chunk_batch_precedes(cu_seqlens, metadata[heap_base + child_idx + 1], metadata[heap_base + child_idx])
                == 1
            ):
                child_idx += 1
        root_batch = metadata[heap_base + root_idx]
        child_batch = metadata[heap_base + child_idx]
        if _chunk_batch_precedes(cu_seqlens, child_batch, root_batch) == 0:
            break
        metadata[heap_base + root_idx] = child_batch
        metadata[heap_base + child_idx] = root_batch
        root_idx = child_idx


def _select_route_from_model(
    route_coefficients,
    layout_kind,
    batch,
    value_heads,
    core_num,
    max_chunk_num,
    peak_chunk_num,
    total_chunk_num,
):
    # This arithmetic stays in the metadata source so its AICPU image manifest
    # binds every instruction. Calibration bounds do not restrict the formula.
    model_base = layout_kind * 16
    if core_num < 1 or core_num > MAX_AIC_CORES or batch < 1:
        return TEMPLATE_STAGED
    if value_heads < 1 or value_heads > WORKSPACE_SLOTS:
        return TEMPLATE_STAGED
    if max_chunk_num < 1 or peak_chunk_num < 1 or total_chunk_num < 1:
        return TEMPLATE_STAGED
    # Check the largest feature products before multiplying, then reserve the
    # sum of absolute terms to bound every signed partial score below.
    if max_chunk_num > 9223372036854775807 // core_num // core_num:
        return TEMPLATE_STAGED
    if peak_chunk_num > 9223372036854775807 // core_num // core_num:
        return TEMPLATE_STAGED
    if total_chunk_num > 9223372036854775807 // core_num:
        return TEMPLATE_STAGED
    score = 0
    remaining_score_budget = 9223372036854775807
    for feature_idx in range(0, 8):
        feature = core_num * core_num
        if feature_idx == 0:
            feature = max_chunk_num * core_num
        elif feature_idx == 1:
            feature = max_chunk_num * core_num * core_num
        elif feature_idx == 2:
            feature = peak_chunk_num * core_num
        elif feature_idx == 3:
            feature = peak_chunk_num * core_num * core_num
        elif feature_idx == 4:
            feature = total_chunk_num
        elif feature_idx == 5:
            feature = total_chunk_num * core_num
        elif feature_idx == 6:
            feature = core_num
        coefficient = route_coefficients[model_base + feature_idx]
        if coefficient < -9223372036854775807:
            return TEMPLATE_STAGED
        coefficient_magnitude = abs(coefficient)
        if coefficient_magnitude > remaining_score_budget // feature:
            return TEMPLATE_STAGED
        score += coefficient * feature
        remaining_score_budget -= coefficient_magnitude * feature
    if score > 0:
        return TEMPLATE_PERSISTENT
    return TEMPLATE_STAGED


@aicpu_kernel
def flash_kda_metadata_kernel(a: FlashKDAMetadataArgs):
    valid_seq_len_by_batch = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU compiler intrinsic.
    token_start_by_batch = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU compiler intrinsic.
    chunk_num_by_batch = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU compiler intrinsic.
    tail_row_num_by_batch = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU compiler intrinsic.
    active_local_batch_idx = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU compiler intrinsic.
    remaining_chunk_num = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU compiler intrinsic.
    chunk_num_per_group = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU compiler intrinsic.
    stage1_task_costs = zeros(I64, MAX_STAGE1_TASKS)  # noqa: F821 - AICPU compiler intrinsic.
    stage1_core_ranges = zeros(I64, MAX_AIC_CORES * 2)  # noqa: F821 - AICPU compiler intrinsic.
    stage1_core_costs = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU compiler intrinsic.
    stage2_task_costs = zeros(I64, MAX_STAGE2_TASKS)  # noqa: F821 - AICPU compiler intrinsic.
    candidate_stage2_core_ranges = zeros(I64, MAX_AIC_CORES * 2)  # noqa: F821 - AICPU compiler intrinsic.
    candidate_stage2_core_costs = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU compiler intrinsic.
    selected_stage2_core_ranges = zeros(I64, MAX_AIC_CORES * 2)  # noqa: F821 - AICPU compiler intrinsic.
    route_core_costs = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU compiler intrinsic.
    chunk_core_costs = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU compiler intrinsic.
    chunk_core_tails = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU compiler intrinsic.

    if a.metadata_word_capacity < HEADER_WORDS:
        # AICPU treats a non-zero kernel return as a task exception (ACL 507005).
        # Diagnostic codes belong in the metadata header, not in the return value.
        return STATUS_OK

    a.metadata[HEADER_MAGIC] = MAGIC
    a.metadata[HEADER_ABI_VERSION] = ABI_VERSION
    a.metadata[HEADER_STATUS] = STATUS_OK
    a.metadata[HEADER_CORE_NUM] = a.core_num
    a.metadata[HEADER_STAGE12_ROUND_NUM] = 0
    a.metadata[HEADER_BATCH] = a.batch
    a.metadata[HEADER_VALUE_HEADS] = a.value_heads
    a.metadata[HEADER_HAS_CU] = a.has_cu
    a.metadata[HEADER_WORKSPACE_SLOTS] = a.workspace_slots
    a.metadata[HEADER_TEMPLATE] = TEMPLATE_AUTO
    a.metadata[HEADER_CHUNK_TASK_NUM] = 0
    if a.metadata_word_capacity > HEADER_WORDS:
        a.metadata[HEADER_WORDS] = HEADER_WORDS + 1

    status = STATUS_OK
    if a.core_num < 1 or a.core_num > MAX_AIC_CORES:
        status = STATUS_BAD_CORE_NUM
    elif (
        a.template_kind != TEMPLATE_AUTO
        and a.template_kind != TEMPLATE_STAGED
        and a.template_kind != TEMPLATE_PERSISTENT
    ):
        status = STATUS_BAD_FLAGS
    elif a.calibration_enabled != 0 and a.calibration_enabled != 1:
        status = STATUS_BAD_FLAGS
    elif a.layout_kind != 0 and a.layout_kind != 1:
        status = STATUS_BAD_FLAGS
    elif a.workspace_slots != WORKSPACE_SLOTS:
        status = STATUS_BAD_WORKSPACE_SLOTS
    elif not (a.has_cu == 0 or a.has_cu == 1):
        status = STATUS_BAD_FLAGS
    elif not (a.is_packed == 0 or a.is_packed == 1):
        status = STATUS_BAD_FLAGS
    elif a.is_packed == 1 and a.has_cu == 0:
        status = STATUS_BAD_FLAGS
    elif a.batch < 1 or a.storage_length < 1 or a.value_heads < 1:
        status = STATUS_BAD_SHAPE
    elif a.value_heads > a.workspace_slots:
        status = STATUS_BAD_SHAPE
    elif a.is_packed == 1 and a.physical_batches != 1:
        status = STATUS_BAD_SHAPE
    elif a.is_packed == 0 and a.physical_batches != a.batch:
        status = STATUS_BAD_SHAPE
    if status != STATUS_OK:
        a.metadata[HEADER_STATUS] = status
        return STATUS_OK

    if a.has_cu == 1:
        if a.cu_seqlens[0] != 0:
            a.metadata[HEADER_STATUS] = STATUS_BAD_CU_START
            return STATUS_OK

    # Validate all real lengths before the route or any task is published.
    uniform_chunk_num = 1
    first_chunk_num = 0
    route_max_chunk_num = 0
    route_total_chunk_num = 0
    route_peak_chunk_num = 0
    evaluate_route = 0
    if a.template_kind == TEMPLATE_AUTO and a.calibration_enabled == 1:
        evaluate_route = 1
    for batch_idx in range(0, a.batch):
        valid_seq_len = a.storage_length
        if a.has_cu == 1:
            valid_seq_len = a.cu_seqlens[batch_idx + 1] - a.cu_seqlens[batch_idx]
        if valid_seq_len <= 0:
            status = STATUS_BAD_LENGTH
        elif a.is_packed == 0 and valid_seq_len > a.storage_length:
            status = STATUS_BAD_PADDED_LENGTH
        else:
            current_chunk_num = (valid_seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
            if batch_idx == 0:
                first_chunk_num = current_chunk_num
            elif current_chunk_num != first_chunk_num:
                uniform_chunk_num = 0
            if current_chunk_num > route_max_chunk_num:
                route_max_chunk_num = current_chunk_num
            if evaluate_route == 1:
                if current_chunk_num > 9223372036854775807 // a.value_heads:
                    evaluate_route = 0
                else:
                    batch_chunk_num = a.value_heads * current_chunk_num
                    if route_total_chunk_num > 9223372036854775807 - batch_chunk_num:
                        evaluate_route = 0
                    else:
                        route_total_chunk_num += batch_chunk_num
                        # Each core cost is at most the checked total cost.
                        for value_head_idx in range(0, a.value_heads):
                            cyclic_core = (
                                batch_idx * a.value_heads + value_head_idx
                            ) % a.core_num
                            route_core_costs[cyclic_core] += current_chunk_num
    if status == STATUS_OK and a.is_packed == 1:
        if a.cu_seqlens[a.batch] != a.storage_length:
            status = STATUS_BAD_PACKED_END
    if status != STATUS_OK:
        a.metadata[HEADER_STATUS] = status
        return STATUS_OK
    selected_template = a.template_kind
    if selected_template == TEMPLATE_AUTO:
        selected_template = TEMPLATE_STAGED
        if evaluate_route == 1:
            for core_idx in range(0, a.core_num):
                if route_core_costs[core_idx] > route_peak_chunk_num:
                    route_peak_chunk_num = route_core_costs[core_idx]
            selected_template = _select_route_from_model(
                a.route_coefficients, a.layout_kind, a.batch, a.value_heads, a.core_num, route_max_chunk_num,
                route_peak_chunk_num, route_total_chunk_num,
            )
    a.metadata[HEADER_TEMPLATE] = selected_template
    if selected_template == TEMPLATE_PERSISTENT:
        task_num = a.batch * a.value_heads
        task_base = HEADER_WORDS + MAX_AIC_CORES
        used_words = task_base + CHUNK_TASK_WORDS * task_num
        heap_base = used_words
        if used_words + a.batch > a.metadata_word_capacity:
            a.metadata[HEADER_STATUS] = STATUS_BAD_METADATA_CAPACITY
            return STATUS_OK
        for core_idx in range(0, MAX_AIC_CORES):
            a.metadata[HEADER_WORDS + core_idx] = 0
        # Keep the common uniform case in its original batch/head order.
        # Ragged batches use O(B log B) LPT scheduling, with B GM scratch words
        # rather than a fixed local array or repeated scans over all batches.
        heap_num = a.batch
        if uniform_chunk_num == 0:
            for batch_idx in range(0, a.batch):
                a.metadata[heap_base + batch_idx] = batch_idx
            heap_root = a.batch // 2
            while heap_root > 0:
                heap_root -= 1
                _chunk_heap_sift_down(a.cu_seqlens, a.metadata, heap_base, heap_num, heap_root)
        task_cursor = task_base
        for scheduled_batch_idx in range(0, a.batch):
            batch_idx = scheduled_batch_idx
            if uniform_chunk_num == 0:
                batch_idx = a.metadata[heap_base]
                heap_num -= 1
                if heap_num > 0:
                    a.metadata[heap_base] = a.metadata[heap_base + heap_num]
                    _chunk_heap_sift_down(a.cu_seqlens, a.metadata, heap_base, heap_num, 0)
            storage_batch_idx = batch_idx
            token_start = 0
            valid_seq_len = a.storage_length
            if a.has_cu == 1:
                token_start = a.cu_seqlens[batch_idx]
                valid_seq_len = a.cu_seqlens[batch_idx + 1] - token_start
            if a.is_packed == 1:
                storage_batch_idx = 0
            else:
                token_start = 0
            chunk_num = (valid_seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
            for value_head_idx in range(0, a.value_heads):
                selected_core = 0
                for core_idx in range(1, a.core_num):
                    if chunk_core_costs[core_idx] < chunk_core_costs[selected_core]:
                        selected_core = core_idx
                a.metadata[task_cursor + CHUNK_TASK_BATCH] = batch_idx
                a.metadata[task_cursor + CHUNK_TASK_STORAGE_BATCH] = storage_batch_idx
                a.metadata[task_cursor + CHUNK_TASK_TOKEN_START] = token_start
                a.metadata[task_cursor + CHUNK_TASK_LENGTH] = valid_seq_len
                a.metadata[task_cursor + CHUNK_TASK_VALUE_HEAD] = value_head_idx
                a.metadata[task_cursor + CHUNK_TASK_NEXT] = 0
                previous_task = chunk_core_tails[selected_core]
                if previous_task == 0:
                    a.metadata[HEADER_WORDS + selected_core] = task_cursor
                else:
                    a.metadata[previous_task + CHUNK_TASK_NEXT] = task_cursor
                chunk_core_tails[selected_core] = task_cursor
                chunk_core_costs[selected_core] += chunk_num
                task_cursor += CHUNK_TASK_WORDS
        a.metadata[HEADER_CHUNK_TASK_NUM] = task_num
        return STATUS_OK

    batch_num_per_set = a.workspace_slots // a.value_heads
    stage12_round_num = 0
    total_active_batch_entries = 0
    batch_start = 0
    while batch_start < a.batch:
        batch_num_in_set = a.batch - batch_start
        if batch_num_in_set > batch_num_per_set:
            batch_num_in_set = batch_num_per_set
        for local_batch_idx in range(0, batch_num_in_set):
            batch_idx = batch_start + local_batch_idx
            token_start = 0
            valid_seq_len = a.storage_length
            if a.has_cu == 1:
                token_start = a.cu_seqlens[batch_idx]
                token_end = a.cu_seqlens[batch_idx + 1]
                valid_seq_len = token_end - token_start
            valid_seq_len_by_batch[local_batch_idx] = valid_seq_len
            token_start_by_batch[local_batch_idx] = token_start
            if valid_seq_len <= 0:
                status = STATUS_BAD_LENGTH
                chunk_num_by_batch[local_batch_idx] = 0
                tail_row_num_by_batch[local_batch_idx] = 0
            elif a.is_packed == 0 and valid_seq_len > a.storage_length:
                status = STATUS_BAD_PADDED_LENGTH
                chunk_num_by_batch[local_batch_idx] = 0
                tail_row_num_by_batch[local_batch_idx] = 0
            else:
                chunk_num_by_batch[local_batch_idx] = (
                    valid_seq_len + CHUNK_SIZE - 1
                ) // CHUNK_SIZE
                tail_row_num_by_batch[local_batch_idx] = (
                    (valid_seq_len - 1) % CHUNK_SIZE
                ) + 1

        chunk_start = 0
        while True:
            active_batch_num = 0
            for local_batch_idx in range(0, batch_num_in_set):
                if chunk_num_by_batch[local_batch_idx] > chunk_start:
                    active_local_batch_idx[active_batch_num] = local_batch_idx
                    remaining_chunk_num[active_batch_num] = (
                        chunk_num_by_batch[local_batch_idx] - chunk_start
                    )
                    active_batch_num += 1
            if active_batch_num == 0:
                break
            group_num = _select_group_num(remaining_chunk_num, active_batch_num, a.value_heads, a.workspace_slots)
            stage12_round_num += 1
            total_active_batch_entries += active_batch_num
            chunk_start += group_num
        batch_start += batch_num_in_set

    if status == STATUS_OK and a.is_packed == 1:
        if a.cu_seqlens[a.batch] != a.storage_length:
            status = STATUS_BAD_PACKED_END
    if status != STATUS_OK:
        a.metadata[HEADER_STATUS] = status
        return STATUS_OK

    record_begin = HEADER_WORDS + stage12_round_num + 1
    used_words = (
        record_begin
        + ROUND_FIXED_WORDS * stage12_round_num
        + ROUND_WORDS_PER_BATCH * total_active_batch_entries
    )
    if used_words > a.metadata_word_capacity:
        a.metadata[HEADER_STATUS] = STATUS_BAD_METADATA_CAPACITY
        return STATUS_OK

    a.metadata[HEADER_STAGE12_ROUND_NUM] = stage12_round_num
    record_cursor = record_begin
    global_round_idx = 0
    batch_start = 0
    while batch_start < a.batch:
        batch_num_in_set = a.batch - batch_start
        if batch_num_in_set > batch_num_per_set:
            batch_num_in_set = batch_num_per_set
        for local_batch_idx in range(0, batch_num_in_set):
            batch_idx = batch_start + local_batch_idx
            token_start = 0
            valid_seq_len = a.storage_length
            if a.has_cu == 1:
                token_start = a.cu_seqlens[batch_idx]
                token_end = a.cu_seqlens[batch_idx + 1]
                valid_seq_len = token_end - token_start
            valid_seq_len_by_batch[local_batch_idx] = valid_seq_len
            token_start_by_batch[local_batch_idx] = token_start
            chunk_num_by_batch[local_batch_idx] = (
                valid_seq_len + CHUNK_SIZE - 1
            ) // CHUNK_SIZE
            tail_row_num_by_batch[local_batch_idx] = (
                (valid_seq_len - 1) % CHUNK_SIZE
            ) + 1

        chunk_start = 0
        while True:
            active_batch_num = 0
            for local_batch_idx in range(0, batch_num_in_set):
                if chunk_num_by_batch[local_batch_idx] > chunk_start:
                    active_local_batch_idx[active_batch_num] = local_batch_idx
                    remaining_chunk_num[active_batch_num] = (
                        chunk_num_by_batch[local_batch_idx] - chunk_start
                    )
                    active_batch_num += 1
            if active_batch_num == 0:
                break

            group_num = _select_group_num(remaining_chunk_num, active_batch_num, a.value_heads, a.workspace_slots)
            for active_idx in range(0, active_batch_num):
                current_chunk_num = remaining_chunk_num[active_idx]
                if current_chunk_num > group_num:
                    current_chunk_num = group_num
                chunk_num_per_group[active_idx] = current_chunk_num

            a.metadata[HEADER_WORDS + global_round_idx] = record_cursor
            cursor = record_cursor
            a.metadata[cursor] = global_round_idx
            a.metadata[cursor + 1] = active_batch_num
            cursor += 2
            for active_idx in range(0, active_batch_num):
                a.metadata[cursor + active_idx] = (
                    batch_start + active_local_batch_idx[active_idx]
                )
            cursor += active_batch_num
            for active_idx in range(0, active_batch_num):
                logical_batch_idx = batch_start + active_local_batch_idx[active_idx]
                if a.is_packed == 1:
                    a.metadata[cursor + active_idx] = 0
                else:
                    a.metadata[cursor + active_idx] = logical_batch_idx
            cursor += active_batch_num
            for active_idx in range(0, active_batch_num):
                local_batch_idx = active_local_batch_idx[active_idx]
                if a.is_packed == 1:
                    a.metadata[cursor + active_idx] = token_start_by_batch[
                        local_batch_idx
                    ]
                else:
                    a.metadata[cursor + active_idx] = 0
            cursor += active_batch_num
            for active_idx in range(0, active_batch_num):
                local_batch_idx = active_local_batch_idx[active_idx]
                a.metadata[cursor + active_idx] = valid_seq_len_by_batch[
                    local_batch_idx
                ]
            cursor += active_batch_num
            a.metadata[cursor] = group_num
            cursor += 1
            for active_idx in range(0, active_batch_num):
                a.metadata[cursor + active_idx] = chunk_start
            cursor += active_batch_num
            for active_idx in range(0, active_batch_num):
                a.metadata[cursor + active_idx] = chunk_num_per_group[active_idx]
            cursor += active_batch_num

            stage1_task_num = 0
            a.metadata[cursor] = 0
            for active_idx in range(0, active_batch_num):
                stage1_task_num += a.value_heads * chunk_num_per_group[active_idx]
                a.metadata[cursor + active_idx + 1] = stage1_task_num
            cursor += active_batch_num + 1

            stage1_task_idx = 0
            for active_idx in range(0, active_batch_num):
                local_batch_idx = active_local_batch_idx[active_idx]
                for _ in range(0, a.value_heads):
                    for local_chunk_idx in range(0, chunk_num_per_group[active_idx]):
                        global_chunk_idx = chunk_start + local_chunk_idx
                        stage1_cost = 4930
                        if (
                            global_chunk_idx + 1 == chunk_num_by_batch[local_batch_idx]
                            and tail_row_num_by_batch[local_batch_idx] < CHUNK_SIZE
                        ):
                            stage1_cost = _stage1_task_cost(tail_row_num_by_batch[local_batch_idx])
                        stage1_task_costs[stage1_task_idx] = stage1_cost
                        stage1_task_idx += 1
            _partition_costs(stage1_task_costs, stage1_task_num, a.core_num, stage1_core_ranges, stage1_core_costs)
            for range_idx in range(0, MAX_AIC_CORES * 2):
                a.metadata[cursor + range_idx] = stage1_core_ranges[range_idx]
            cursor += MAX_AIC_CORES * 2

            selected_dv_splits_num = 1
            selected_max_core_cost = 0
            selected_total_cost = 0
            selected_task_num = 0
            selected_is_set = 0
            for candidate_idx in range(0, 4):
                candidate_dv_splits_num = 1 << candidate_idx
                candidate_task_num = 0
                candidate_total_cost = 0
                for active_idx in range(0, active_batch_num):
                    task_cost = _stage2_task_cost(candidate_idx, chunk_num_per_group[active_idx])
                    batch_task_num = a.value_heads * candidate_dv_splits_num
                    for _ in range(0, batch_task_num):
                        stage2_task_costs[candidate_task_num] = task_cost
                        candidate_task_num += 1
                        candidate_total_cost += task_cost
                _partition_costs(
                    stage2_task_costs, candidate_task_num, a.core_num, candidate_stage2_core_ranges,
                    candidate_stage2_core_costs,
                )
                candidate_max_core_cost = 0
                for core_idx in range(0, MAX_AIC_CORES):
                    if candidate_stage2_core_costs[core_idx] > candidate_max_core_cost:
                        candidate_max_core_cost = candidate_stage2_core_costs[core_idx]

                candidate_is_better = 0
                if selected_is_set == 0:
                    candidate_is_better = 1
                elif candidate_max_core_cost < selected_max_core_cost:
                    candidate_is_better = 1
                elif candidate_max_core_cost == selected_max_core_cost:
                    if candidate_total_cost < selected_total_cost:
                        candidate_is_better = 1
                    elif candidate_total_cost == selected_total_cost:
                        if candidate_task_num < selected_task_num:
                            candidate_is_better = 1
                        elif candidate_task_num == selected_task_num:
                            if candidate_dv_splits_num < selected_dv_splits_num:
                                candidate_is_better = 1
                if candidate_is_better == 1:
                    selected_is_set = 1
                    selected_dv_splits_num = candidate_dv_splits_num
                    selected_max_core_cost = candidate_max_core_cost
                    selected_total_cost = candidate_total_cost
                    selected_task_num = candidate_task_num
                    for range_idx in range(0, MAX_AIC_CORES * 2):
                        selected_stage2_core_ranges[range_idx] = (
                            candidate_stage2_core_ranges[range_idx]
                        )

            for active_idx in range(0, active_batch_num):
                a.metadata[cursor + active_idx] = selected_dv_splits_num
            cursor += active_batch_num
            stage2_task_num = 0
            a.metadata[cursor] = 0
            for active_idx in range(0, active_batch_num):
                stage2_task_num += a.value_heads * selected_dv_splits_num
                a.metadata[cursor + active_idx + 1] = stage2_task_num
            cursor += active_batch_num + 1
            for range_idx in range(0, MAX_AIC_CORES * 2):
                a.metadata[cursor + range_idx] = selected_stage2_core_ranges[range_idx]
            cursor += MAX_AIC_CORES * 2

            record_cursor = cursor
            global_round_idx += 1
            chunk_start += group_num
        batch_start += batch_num_in_set

    a.metadata[HEADER_WORDS + stage12_round_num] = record_cursor
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
    cursor += 2
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

    if group_num <= 0:
        raise ValueError("group_num must be positive")
    if any(index >= batch for index in batch_idx):
        raise ValueError("batch_idx is outside logical batch")
    if any(length <= 0 for length in valid_seq_len):
        raise ValueError("valid_seq_len must be positive")
    if any(count <= 0 for count in chunk_num_per_group):
        raise ValueError("chunk_num_per_group must be positive")
    if any(value not in (1, 2, 4, 8) for value in dv_splits_num):
        raise ValueError("dv_splits_num must be 1, 2, 4, or 8")

    _validate_prefix(stage1_task_prefix, "stage1_task_prefix")
    _validate_prefix(stage2_task_prefix, "stage2_task_prefix")
    for index, chunk_num in enumerate(chunk_num_per_group):
        if stage1_task_prefix[index + 1] - stage1_task_prefix[index] != (
            value_heads * chunk_num
        ):
            raise ValueError("stage1_task_prefix does not match chunk tasks")
        if stage2_task_prefix[index + 1] - stage2_task_prefix[index] != (
            value_heads * dv_splits_num[index]
        ):
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
        "template": {TEMPLATE_STAGED: "staged", TEMPLATE_PERSISTENT: "persistent"}.get(words[HEADER_TEMPLATE], "unknown"),
    }
    if status != STATUS_OK:
        header["used_words"] = HEADER_WORDS
        header["stage12_rounds"] = []
        return header
    if not 1 <= header["core_num"] <= MAX_AIC_CORES:
        raise ValueError("metadata core_num must be in [1, 32]")
    if header["workspace_slots"] != WORKSPACE_SLOTS:
        raise ValueError("metadata workspace_slots must be 861")
    if header["batch"] <= 0 or header["value_heads"] <= 0:
        raise ValueError("successful metadata requires positive shapes")

    if header["template"] == "unknown":
        raise ValueError("metadata template is invalid")
    if header["template"] == "persistent":
        task_num = words[HEADER_CHUNK_TASK_NUM]
        task_base = HEADER_WORDS + MAX_AIC_CORES
        used_words = task_base + CHUNK_TASK_WORDS * task_num
        if (
            header["stage12_round_num"] != 0
            or task_num != header["batch"] * header["value_heads"]
        ):
            raise ValueError("chunk metadata task count does not match its shape")
        if used_words > len(words):
            raise ValueError("chunk metadata tasks are truncated")
        seen_offsets = set()
        seen_jobs = set()
        tasks_by_core = []
        for core_idx in range(MAX_AIC_CORES):
            cursor = words[HEADER_WORDS + core_idx]
            if core_idx >= header["core_num"] and cursor:
                raise ValueError("inactive core has chunk tasks")
            tasks = []
            while cursor:
                if cursor in seen_offsets:
                    raise ValueError("chunk task chain contains a cycle or duplicate")
                if (
                    cursor < task_base
                    or cursor + CHUNK_TASK_WORDS > used_words
                    or (cursor - task_base) % CHUNK_TASK_WORDS
                ):
                    raise ValueError("chunk task pointer is out of bounds or unaligned")
                seen_offsets.add(cursor)
                batch_idx = words[cursor + CHUNK_TASK_BATCH]
                head_idx = words[cursor + CHUNK_TASK_VALUE_HEAD]
                if (
                    not 0 <= batch_idx < header["batch"]
                    or not 0 <= head_idx < header["value_heads"]
                ):
                    raise ValueError("chunk task batch/head index is out of bounds")
                if (batch_idx, head_idx) in seen_jobs:
                    raise ValueError("chunk sequence/head task is duplicated")
                seen_jobs.add((batch_idx, head_idx))
                length = words[cursor + CHUNK_TASK_LENGTH]
                storage_batch = words[cursor + CHUNK_TASK_STORAGE_BATCH]
                token_start = words[cursor + CHUNK_TASK_TOKEN_START]
                if (
                    length <= 0
                    or token_start < 0
                    or storage_batch not in (0, batch_idx)
                ):
                    raise ValueError("chunk task has invalid sequence coordinates")
                tasks.append(
                    dict(
                        batch_idx=batch_idx, storage_batch_idx=storage_batch, token_start=token_start,
                        sequence_length=length, value_head_idx=head_idx,
                    ),
                )
                cursor = words[cursor + CHUNK_TASK_NEXT]
            tasks_by_core.append(tasks)
        if len(seen_jobs) != task_num:
            raise ValueError("chunk metadata does not cover every task")
        header.update(
            used_words=used_words, stage12_rounds=[], chunk_task_num=task_num, chunk_tasks_by_core=tasks_by_core,
        )
        return header

    stage12_round_num = int(header["stage12_round_num"])
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

    header["used_words"] = offsets[-1]
    header["stage12_rounds"] = [
        _decode_round(
            words, offsets[index], offsets[index + 1], batch=int(header["batch"]),
            value_heads=int(header["value_heads"]),
        )
        for index in range(stage12_round_num)
    ]
    return header


SUPPORTED_HEAD_DIM = 128

_AICPU_COMPILED = None
_AICPU_LOCK = threading.RLock()
_AICPU_STREAMS = {}


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
    global _AICPU_COMPILED
    with _AICPU_LOCK:
        if _AICPU_COMPILED is None:
            _AICPU_COMPILED = compile_aicpu_kernel(
                flash_kda_metadata_kernel, workdir=tempfile.mkdtemp(prefix="flash_kda_metadata_"),
                launch_mode="interface", npu_arch="dav-3510",
            )
        return _AICPU_COMPILED


def flash_kda_metadata(
    q: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    layout_qkv: str,
    cu_seqlens: Optional[torch.Tensor] = None,
    *,
    template: str = "auto",
) -> torch.Tensor:
    """Submit metadata asynchronously; consume it on this same stream."""
    if template not in ("auto", "staged", "persistent"):
        raise ValueError("template must be auto, staged, or persistent")
    template_id = {
        "auto": TEMPLATE_AUTO,
        "staged": TEMPLATE_STAGED,
        "persistent": TEMPLATE_PERSISTENT,
    }[template]
    assert layout_qkv in ("TND", "BNSD", "BSND"), (
        f"layout_qkv must be TND, BNSD, or BSND, got {layout_qkv!r}"
    )
    assert initial_state.dim() == 4, "initial_state must be rank-4"
    batch, n_v, state_dv, state_dk = initial_state.shape
    assert state_dv == state_dk == SUPPORTED_HEAD_DIM, (
        f"FlashKDA only supports D={SUPPORTED_HEAD_DIM}"
    )
    if layout_qkv == "TND":
        assert cu_seqlens is not None, "cu_seqlens is required for TND"
    if cu_seqlens is not None:
        assert cu_seqlens.dtype == torch.int32, "cu_seqlens must be int32"
        assert cu_seqlens.dim() == 1, "cu_seqlens must be rank-1"
        assert cu_seqlens.shape == (batch + 1,), (
            f"cu_seqlens shape must be ({batch + 1},)"
        )
        assert cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous"

    if layout_qkv == "TND":
        assert q.dim() == v.dim() == 3, "TND q/v must be rank-3"
        storage_length, n_qk, dim = q.shape
        assert v.shape == (storage_length, n_v, dim), (
            "v shape does not match TND layout"
        )
        physical_batches = 1
        is_packed = True
    elif layout_qkv == "BNSD":
        assert q.dim() == v.dim() == 4, "BNSD q/v must be rank-4"
        physical_batches, n_qk, storage_length, dim = q.shape
        assert v.shape == (physical_batches, n_v, storage_length, dim), (
            "v shape does not match BNSD layout"
        )
        is_packed = False
    else:
        assert q.dim() == v.dim() == 4, "BSND q/v must be rank-4"
        physical_batches, storage_length, n_qk, dim = q.shape
        assert v.shape == (physical_batches, storage_length, n_v, dim), (
            "v shape does not match BSND layout"
        )
        is_packed = False

    if layout_qkv != "TND":
        assert physical_batches == batch, (
            "padded storage batch must match initial_state"
        )
    assert dim == SUPPORTED_HEAD_DIM, f"FlashKDA only supports D={SUPPORTED_HEAD_DIM}"
    assert n_v % n_qk == 0, f"GQA requires Nv % Nqk == 0, got Nv={n_v}, Nqk={n_qk}"
    assert storage_length > 0, "storage sequence length must be positive"
    assert q.dtype == v.dtype == torch.bfloat16, "q/v must be bf16"
    assert initial_state.dtype == torch.float32, "initial_state must be fp32"
    assert q.device == v.device == initial_state.device, (
        "all inputs must be on the same device"
    )
    assert q.is_contiguous() and v.is_contiguous() and initial_state.is_contiguous(), (
        "all public inputs must be contiguous"
    )
    if cu_seqlens is not None:
        assert cu_seqlens.device == q.device, "all inputs must be on the same device"

    core_num = _device_block_num(q)
    metadata_capacity = metadata_capacity_upper_bound(batch, storage_length, is_packed=is_packed, value_heads=n_v)
    # AICPU initializes every consumed word; capacity beyond its schedule is undefined.
    metadata = torch.empty(metadata_capacity, dtype=torch.int32, device=q.device)
    route_parameters, calibration_enabled = _get_route_parameters(q.device)
    device_id = q.get_device()
    with torch.npu.device(device_id), _AICPU_LOCK:
        stream = torch.npu.current_stream(device_id)
        for tensor in (cu_seqlens, route_parameters, metadata):
            if tensor is not None:
                tensor.record_stream(stream)
        _AICPU_STREAMS[(device_id, stream.npu_stream)] = stream
        try:
            aicpu = _get_aicpu_kernel()
            aicpu.launch(
                current_raw_stream(device_id), route_coefficients=0 if route_parameters is None else route_parameters.data_ptr(),
                calibration_enabled=calibration_enabled, layout_kind=0 if layout_qkv == "BNSD" else 1,
                cu_seqlens=0 if cu_seqlens is None else cu_seqlens.data_ptr(), metadata=metadata.data_ptr(),
                metadata_word_capacity=metadata.numel(), batch=batch, physical_batches=physical_batches,
                storage_length=storage_length, value_heads=n_v, has_cu=int(cu_seqlens is not None), is_packed=int(is_packed),
                core_num=core_num, workspace_slots=WORKSPACE_SLOTS, template_kind=template_id,
            )
        except Exception:
            # Retire any raw-pointer work submitted before the launch failed.
            stream.synchronize()
            raise
    return metadata


def clear_metadata_caches():
    """Wait for used streams before releasing the metadata executable."""
    global _AICPU_COMPILED
    with _AICPU_LOCK:
        for (device_id, _), stream in _AICPU_STREAMS.items():
            with torch.npu.device(device_id):
                stream.synchronize()
        _AICPU_STREAMS.clear()
        if _AICPU_COMPILED is not None:
            close = getattr(_AICPU_COMPILED, "close", None)
            if callable(close):
                close()
        _AICPU_COMPILED = None


# BNSD and BSND/TND coefficients, followed by historical calibration coverage.
_ROUTE_PARAMETERS = (
    423085,
    38413,
    -3296609,
    -32705,
    5909545,
    33970,
    807651,
    36681,
    8,
    32,
    1,
    4,
    3,
    96,
    1,
    256,
    -1057161,
    63178,
    -2467383,
    -51441,
    5132325,
    70710,
    2005962,
    -43262,
    8,
    32,
    1,
    4,
    3,
    96,
    1,
    256,
)
_ROUTE_TENSORS = {}
_ROUTE_PARAMETER_LOCK = threading.Lock()


def _get_route_parameters(device):
    device = torch.device(device)
    with torch.npu.device(device):
        stream = torch.npu.current_stream(device)
        key = (device, stream.npu_stream)
        with _ROUTE_PARAMETER_LOCK:
            tensor = _ROUTE_TENSORS.get(key)
            if tensor is None:
                tensor = torch.tensor(_ROUTE_PARAMETERS, dtype=torch.int64).to(device=device, non_blocking=True)
                _ROUTE_TENSORS[key] = tensor
        tensor.record_stream(stream)
    return tensor, 1
