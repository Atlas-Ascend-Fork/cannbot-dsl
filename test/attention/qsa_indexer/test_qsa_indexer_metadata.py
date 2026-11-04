# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""AICPU QSA metadata layout, coverage and stream checks."""

import pytest
import torch
from _samples_path import load_sample

_sample = load_sample("attention/qsa_indexer/qsa_indexer_metadata.py")
PAGE_SIZE = _sample.PAGE_SIZE
AIC_METADATA_CORE_CAPACITY = _sample.AIC_METADATA_CORE_CAPACITY
CORE_METADATA_STRIDE = _sample.CORE_METADATA_STRIDE
HEAD_METADATA_STRIDE = _sample.HEAD_METADATA_STRIDE
SECTION_METADATA_STRIDE = _sample.SECTION_METADATA_STRIDE
get_effective_core_counts = _sample.get_effective_core_counts
metadata_capacity = _sample.metadata_capacity
qsa_indexer_metadata = _sample.qsa_indexer_metadata


# 覆盖默认核数、显式裁核、多section和尾Q32任务。
QSA_METADATA_CASES = (
    pytest.param([33, 65], 4095, None, id="QI_METADATA_DEFAULT_CORES"),
    pytest.param([16 * 1024] * 6, 64 * 1024 - 1, 4, id="QI_METADATA_MULTI_SECTION"),
    pytest.param([1, 31, 32, 33, 65], 255, 7, id="QI_METADATA_Q32_TAIL"),
)


def _inputs(lengths, max_position):
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    query_positions = [i % (max_position + 1) for i in range(starts[-1])]
    visible = max(((position + 1) // 4 for position in query_positions), default=0)
    pages = max(1, (visible + PAGE_SIZE - 1) // PAGE_SIZE)
    table = torch.arange(pages, dtype=torch.int32).expand(len(lengths), -1).contiguous()
    return starts, query_positions, table, pages


def _task_index(starts, coordinate):
    request, m_index = coordinate
    return (
        sum(
            (length + 31) // 32
            for length in (starts[i + 1] - starts[i] for i in range(request))
        )
        + m_index
    )


def _check(metadata, starts, block_dim):
    data = metadata.cpu()
    assert data.dtype == torch.int32
    assert data.numel() == metadata_capacity(len(starts) - 1)
    section_count = int(data[0])
    assert 1 <= section_count <= max(1, len(starts) - 1)
    assert not torch.count_nonzero(data[1:HEAD_METADATA_STRIDE])
    previous = (0, 0)
    seen_tasks = []
    for section in range(section_count):
        base = HEAD_METADATA_STRIDE + section * SECTION_METADATA_STRIDE
        slots = data[base : base + SECTION_METADATA_STRIDE].reshape(
            AIC_METADATA_CORE_CAPACITY, CORE_METADATA_STRIDE
        )
        assert not torch.count_nonzero(slots[block_dim:])
        for slot in slots[:block_dim]:
            if not torch.count_nonzero(slot):
                continue
            start = (int(slot[0]), int(slot[1]))
            end = (int(slot[3]), int(slot[4]))
            assert start == previous
            assert start <= end <= (len(starts) - 1, 0)
            assert int(slot[2]) == 0 and not torch.count_nonzero(slot[5:])
            first, last = _task_index(starts, start), _task_index(starts, end)
            seen_tasks.extend(range(first, last))
            previous = end
    expected_tasks = sum(
        (starts[i + 1] - starts[i] + 31) // 32 for i in range(len(starts) - 1)
    )
    assert previous == (len(starts) - 1, 0)
    assert seen_tasks == list(range(expected_tasks))
    end = HEAD_METADATA_STRIDE + section_count * SECTION_METADATA_STRIDE
    assert not torch.count_nonzero(data[end:])


@pytest.mark.npu
@pytest.mark.parametrize("lengths,max_position,block_dim", QSA_METADATA_CASES)
def test_qsa_indexer_metadata_coverage(lengths, max_position, block_dim):
    pytest.importorskip("torch_npu")
    starts, query_positions, table, pages = _inputs(lengths, max_position)
    metadata = qsa_indexer_metadata(
        torch.tensor(starts, dtype=torch.int32, device="npu"),
        torch.tensor(query_positions, dtype=torch.int32, device="npu"),
        table.npu(),
        pages,
        block_dim=block_dim,
    )
    torch.npu.synchronize()
    effective_block_dim = (
        get_effective_core_counts(torch.npu.current_stream())[0]
        if block_dim is None
        else block_dim
    )
    _check(metadata, starts, effective_block_dim)


@pytest.mark.npu
def test_nondefault_stream_and_default_core_count():
    pytest.importorskip("torch_npu")
    starts, query_positions, table, pages = _inputs([65, 33], 2047)
    qstarts = torch.tensor(starts, dtype=torch.int32, device="npu")
    pos = torch.tensor(query_positions, dtype=torch.int32, device="npu")
    table = table.npu()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        block_dim, _ = get_effective_core_counts(stream)
        metadata = qsa_indexer_metadata(qstarts, pos, table, pages)
    stream.synchronize()
    _check(metadata, starts, block_dim)


@pytest.mark.npu
def test_input_validation():
    pytest.importorskip("torch_npu")
    starts, query_positions, table, pages = _inputs([32], 255)
    qstarts = torch.tensor(starts, dtype=torch.int32, device="npu")
    pos = torch.tensor(query_positions, dtype=torch.int32, device="npu")
    with pytest.raises(ValueError):
        qsa_indexer_metadata(qstarts.to(torch.int64), pos, table.npu(), pages)
    cube, _ = get_effective_core_counts(torch.npu.current_stream())
    with pytest.raises(ValueError):
        qsa_indexer_metadata(qstarts, pos, table.npu(), pages, block_dim=cube + 1)
