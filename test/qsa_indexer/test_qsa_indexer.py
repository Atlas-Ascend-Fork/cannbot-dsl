# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""QSA Indexer paper-case accuracy tests through external AICPU metadata."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pytest
import torch
from _samples_path import load_sample

_qsa = load_sample("qsa_indexer/qsa_indexer.py")
qsa_indexer = _qsa.qsa_indexer
MAX_TOPK = _qsa.MAX_TOPK
OUTPUT_LENGTH_COLUMN = _qsa.OUTPUT_LENGTH_COLUMN
qsa_indexer_metadata = load_sample(
    "qsa_indexer/qsa_indexer_metadata.py"
).qsa_indexer_metadata

TOPK_BOUNDARY_RE_TOLERANCE = 1e-3
TOPK_BOUNDARY_ABS_TOLERANCE = 2.5e-5
MAX_BOUNDARY_RE_EXCEED_RATIO = 5e-3
QUERY_LENGTH = 16 * 1024
PAGE_SIZE = 256
HEADS = 4
HEAD_DIM = 128
TOPK_ORACLE_ROWS = 8


@dataclass(frozen=True)
class QSAIndexerCase:
    context_length: int
    compressed_k_length: int
    seed: int = 20260912


# 论文中的四个Prefill场景：Q固定16K，K为r=4压缩后的长度。
QSA_INDEXER_CASES = (
    pytest.param(QSAIndexerCase(64 * 1024, 16 * 1024), id="P64K"),
    pytest.param(QSAIndexerCase(128 * 1024, 32 * 1024), id="P128K"),
    pytest.param(QSAIndexerCase(256 * 1024, 64 * 1024), id="P256K"),
    pytest.param(QSAIndexerCase(512 * 1024, 128 * 1024), id="P512K"),
)


def _relative_boundary_error(value, reference):
    if value == reference:
        return 0.0
    if not math.isfinite(value) or not math.isfinite(reference):
        return float("inf")
    error = abs(value - reference)
    if error <= TOPK_BOUNDARY_ABS_TOLERANCE:
        return 0.0
    return error / abs(reference) if reference else float("inf")


def _validate_structure(actual, query_positions):
    """Check count, causal range, uniqueness, expansion, tail and padding for every row."""
    visible = (query_positions.to(torch.int64) + 1) // 4
    tail = (query_positions.to(torch.int64) + 1) % 4
    expanded_topk = MAX_TOPK * 4
    expected_count = expanded_topk + tail
    assert torch.equal(actual[:, OUTPUT_LENGTH_COLUMN].to(torch.int64), expected_count)

    blocks = actual[:, :expanded_topk].reshape(-1, MAX_TOPK, 4)
    offsets = torch.arange(4, dtype=torch.int32).reshape(1, 1, 4)
    assert torch.equal(blocks, blocks[:, :, :1] + offsets)
    block_ids = blocks[:, :, 0] // 4
    sorted_ids = block_ids.sort(dim=1).values
    assert torch.all(sorted_ids[:, 1:] > sorted_ids[:, :-1])
    assert torch.all(block_ids >= 0)
    assert torch.all(block_ids < visible[:, None])

    for offset in range(3):
        expected = torch.where(
            tail > offset,
            visible * 4 + offset,
            torch.full_like(visible, -1),
        )
        assert torch.equal(actual[:, expanded_topk + offset].to(torch.int64), expected)
    return block_ids, visible


def _validate_sampled_topk(actual_ids, visible, q, cache, table):
    """Check eight evenly spaced rows against the BF16 UINT16 TopK boundary rule."""
    sample_rows = torch.linspace(
        0, q.shape[0] - 1, steps=TOPK_ORACLE_ROWS, dtype=torch.int64
    ).unique()
    logical_k = cache[table[0].long(), :, 0, :].reshape(-1, HEAD_DIM).float()
    queries = q[sample_rows].float()
    scores = torch.matmul(queries, logical_k.T).relu().sum(dim=1)
    scores = scores.bfloat16().float()
    columns = torch.arange(logical_k.shape[0], dtype=torch.int64)
    scores.masked_fill_(columns[None, :] >= visible[sample_rows, None], float("-inf"))
    expected = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :MAX_TOPK]

    exceeded = 0
    for sample, row in enumerate(sample_rows.tolist()):
        actual_set = set(actual_ids[row].tolist())
        expected_set = set(expected[sample].tolist())
        if actual_set == expected_set:
            continue
        boundary = min(float(scores[sample, index]) for index in expected_set)
        bad_extra = sum(
            _relative_boundary_error(float(scores[sample, index]), boundary)
            > TOPK_BOUNDARY_RE_TOLERANCE
            for index in actual_set - expected_set
        )
        bad_missing = sum(
            _relative_boundary_error(float(scores[sample, index]), boundary)
            > TOPK_BOUNDARY_RE_TOLERANCE
            for index in expected_set - actual_set
        )
        exceeded += max(bad_extra, bad_missing)
    ratio = exceeded / (sample_rows.numel() * MAX_TOPK)
    assert ratio <= MAX_BOUNDARY_RE_EXCEED_RATIO


def _run_qsa_indexer_case(case, query_length):
    pytest.importorskip("torch_npu")
    torch.manual_seed(case.seed)
    page_count = case.compressed_k_length // PAGE_SIZE
    q = torch.randn((query_length, HEADS, HEAD_DIM), dtype=torch.bfloat16)
    cache = torch.randn((page_count, PAGE_SIZE, 1, HEAD_DIM), dtype=torch.bfloat16)
    table = torch.randperm(page_count).int().reshape(1, -1)
    starts = torch.tensor([0, query_length], dtype=torch.int32)
    query_positions = torch.arange(
        case.context_length - query_length,
        case.context_length,
        dtype=torch.int32,
    )

    q_npu = q.npu()
    cache_npu = cache.npu()
    table_npu = table.npu()
    starts_npu = starts.npu()
    positions_npu = query_positions.npu()
    metadata = qsa_indexer_metadata(
        starts_npu,
        positions_npu,
        table_npu,
        compressed_page_count=page_count,
    )
    actual = qsa_indexer(
        q_npu,
        cache_npu,
        table_npu,
        starts_npu,
        positions_npu,
        metadata=metadata,
    )
    torch.npu.synchronize()

    actual_ids, visible = _validate_structure(actual.cpu(), query_positions)
    _validate_sampled_topk(actual_ids, visible, q, cache, table)


@pytest.mark.npu
@pytest.mark.parametrize("case", QSA_INDEXER_CASES)
def test_qsa_indexer(case):
    _run_qsa_indexer_case(case, QUERY_LENGTH)


@pytest.mark.npu
@pytest.mark.parametrize("compressed_length", [2304, 2560, 4096, 8192])
@pytest.mark.parametrize("query_length", [1, 17, 129])
def test_qsa_indexer_short_k(compressed_length, query_length):
    """覆盖评分写出后立即读取的短 K、Query 尾块和不同 chunk 数量。"""
    _run_qsa_indexer_case(
        QSAIndexerCase(compressed_length * 4 - 2, compressed_length, seed=410000),
        query_length,
    )
