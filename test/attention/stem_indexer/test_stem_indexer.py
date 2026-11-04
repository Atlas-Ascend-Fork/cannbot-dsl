# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Stem Indexer 内置参数化精度测试。"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import pytest
import torch
from _samples_path import load_sample

si_module = load_sample("attention/stem_indexer/stem_indexer.py")
stem_indexer = si_module.stem_indexer
FixedAttributes = si_module.FixedAttributes
MAX_TOPK = si_module.MAX_TOPK
ceil_div = si_module.ceil_div
TOPK_BUDGET_DIRECT_PROMPT_LIMIT = si_module.TOPK_BUDGET_DIRECT_PROMPT_LIMIT
TOPK_BUDGET_LONG_PROMPT_THRESHOLD = si_module.TOPK_BUDGET_LONG_PROMPT_THRESHOLD
TOPK_BUDGET_MEDIUM_PROMPT_RATIO = si_module.TOPK_BUDGET_MEDIUM_PROMPT_RATIO
TOPK_BUDGET_LONG_PROMPT_RATIO = si_module.TOPK_BUDGET_LONG_PROMPT_RATIO
TOPK_BUDGET_BASE_OFFSET = si_module.TOPK_BUDGET_BASE_OFFSET
stem_indexer_metadata = load_sample(
    "attention/stem_indexer/stem_indexer_metadata.py"
).stem_indexer_metadata


# 参考计算与数据构造
def _f32(value: float | int) -> float:
    return float(torch.tensor(value, dtype=torch.float32).item())


def calculate_topk_budget(
    *,
    q_block: int,
    q_blocks: int,
    kv_blocks: int,
    prompt_tokens: int,
    attrs: FixedAttributes,
) -> int:
    prompt_blocks = ceil_div(prompt_tokens, attrs.stem_block_size)
    if prompt_blocks < TOPK_BUDGET_DIRECT_PROMPT_LIMIT:
        start = prompt_blocks
    elif prompt_blocks < TOPK_BUDGET_LONG_PROMPT_THRESHOLD:
        start = int(
            _f32(_f32(prompt_blocks) * _f32(TOPK_BUDGET_MEDIUM_PROMPT_RATIO))
            + _f32(TOPK_BUDGET_BASE_OFFSET)
        )
    else:
        start = int(
            _f32(_f32(prompt_blocks) * _f32(TOPK_BUDGET_LONG_PROMPT_RATIO))
            + _f32(TOPK_BUDGET_BASE_OFFSET)
        )
    s1_position = q_block + kv_blocks - q_blocks
    decay_length = prompt_blocks - start
    if s1_position < start or decay_length <= 1:
        return min(max(start, 1), MAX_TOPK)
    start_f = _f32(start)
    end_f = _f32(start_f * _f32(attrs.alpha))
    ratio = _f32(_f32(s1_position - start) / _f32(decay_length - 1))
    interpolated = _f32(float(ratio) * float(_f32(end_f - start_f)) + float(start_f))
    return min(max(math.floor(interpolated), 1), start, MAX_TOPK)


def attributes_from_case(case) -> FixedAttributes:
    return FixedAttributes(
        causal=case.causal,
        stem_block_size=case.stem_block_size,
        stem_stride=case.stem_stride,
        alpha=case.alpha,
        initial_blocks=case.initial_blocks,
        window_size=case.window_size,
        topk_score_precision=case.topk_score_precision,
    )


def gen_case(
    q_lengths,
    kv_lengths,
    prompt_tokens,
    nq,
    nkv,
    d,
    *,
    causal,
    alpha,
    stem_block_size,
    stem_stride,
    initial_blocks,
    window_size,
    precision,
    seed=20260901,
    case_id="SI",
):
    """按有效长度构造 BF16 Q/K、FP32 bias 和 INT32 长度输入。"""
    batch = len(q_lengths)
    assert batch == len(kv_lengths) == len(prompt_tokens)
    q_blocks = ceil_div(max(q_lengths), stem_block_size)
    kv_blocks = ceil_div(max(kv_lengths), stem_block_size)
    case = SimpleNamespace(
        case_id=case_id,
        batch_size=batch,
        q_heads=nq,
        kv_heads=nkv,
        head_dim=d,
        q_seq_lens=tuple(q_lengths),
        kv_seq_lens=tuple(kv_lengths),
        num_prompt_tokens=tuple(prompt_tokens),
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
        causal=causal,
        alpha=alpha,
        stem_block_size=stem_block_size,
        stem_stride=stem_stride,
        initial_blocks=initial_blocks,
        window_size=window_size,
        topk_score_precision=precision,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((batch, nq, q_blocks, d), dtype=torch.bfloat16, generator=generator)
    k = torch.randn(
        (batch, nkv, kv_blocks, d), dtype=torch.bfloat16, generator=generator
    )
    bias = torch.randn(
        (batch, nkv, kv_blocks), dtype=torch.float32, generator=generator
    )
    lengths = tuple(
        torch.tensor(values, dtype=torch.int32)
        for values in (q_lengths, kv_lengths, prompt_tokens)
    )
    return case, (q, k, bias, *lengths)


TOPK_BOUNDARY_RE_TOLERANCE = 1e-3
TOPK_BOUNDARY_ABS_TOLERANCE = 2.5e-5
MAX_BOUNDARY_RE_EXCEED_RATIO = 5e-3

# 精选功能用例覆盖p1/p2、因果模式、边界长度、ragged batch及不同头数组合。
STEM_INDEXER_CASES = (
    pytest.param(
        [2049, 4097],
        [4096, 16385],
        [2048, 8192],
        32,
        4,
        2048,
        True,
        0.75,
        128,
        16,
        4,
        4,
        1,
        310030,
        id="SI_BUDGET_FP32_BOUNDARY_P1",
    ),
    pytest.param(
        [255, 255, 1023],
        [255, 255, 16385],
        [255, 128, 8192],
        64,
        2,
        2048,
        True,
        0.75,
        128,
        16,
        4,
        4,
        2,
        310143,
        id="SI_BUDGET_FP32_BOUNDARY_P2",
    ),
    pytest.param(
        [1023],
        [7040],
        [7167],
        32,
        8,
        2048,
        True,
        0.75,
        128,
        16,
        4,
        4,
        2,
        20260901,
        id="SI_ZERO_TOPK_P2",
    ),
    pytest.param(
        [1023],
        [7040],
        [7167],
        32,
        8,
        2048,
        True,
        0.75,
        128,
        16,
        4,
        4,
        1,
        20260901,
        id="SI_ZERO_TOPK_P1",
    ),
    pytest.param(
        [65536],
        [65536],
        [65536],
        64,
        8,
        2048,
        True,
        1.0,
        128,
        16,
        4,
        4,
        2,
        20260901,
        id="SI_REDLINE_000000",
    ),
    pytest.param(
        [65536],
        [65536],
        [65536],
        64,
        8,
        2048,
        False,
        1.0,
        128,
        16,
        4,
        4,
        2,
        20260901,
        id="SI_REDLINE_000001",
    ),
    pytest.param(
        [65536],
        [65536],
        [65536],
        64,
        8,
        2048,
        True,
        1.0,
        128,
        16,
        4,
        4,
        1,
        20260901,
        id="SI_REDLINE_000002",
    ),
    pytest.param(
        [3],
        [3],
        [3],
        32,
        2,
        2048,
        True,
        1.0,
        128,
        16,
        4,
        4,
        1,
        20260901,
        id="SI_REDLINE_000006",
    ),
    pytest.param(
        [52, 500],
        [52, 500],
        [128, 768],
        32,
        4,
        2048,
        True,
        0.75,
        128,
        16,
        4,
        4,
        2,
        20260901,
        id="SI_REDLINE_000008",
    ),
    pytest.param(
        [8576, 8576],
        [6144, 6144],
        [6144, 6144],
        32,
        4,
        2048,
        True,
        0.25,
        128,
        16,
        4,
        4,
        2,
        20260901,
        id="SI_REDLINE_000020",
    ),
)


# 精度判定：SI 索引/长度及 TopK 边界容差，不使用 FIA 的 FP16 输出阈值。
def _visible_blocks(case, batch: int, q_block: int) -> int:
    q_blocks = ceil_div(case.q_seq_lens[batch], case.stem_block_size)
    kv_blocks = ceil_div(case.kv_seq_lens[batch], case.stem_block_size)
    decode = (
        case.q_seq_lens[batch] == 1
        and case.num_prompt_tokens[batch] >= case.kv_seq_lens[batch]
    )
    if case.causal and not decode:
        return max(min(kv_blocks - q_blocks + q_block + 1, kv_blocks), 0)
    return kv_blocks


def _reference_topk_order(
    candidates: torch.Tensor, scores: torch.Tensor, selected_count: int
) -> torch.Tensor:
    """对齐FindIdxOutputProcessRow：先按索引输出GT项，再按索引输出EQ项。"""
    ranked = torch.argsort(scores, descending=True, stable=True)
    kth = scores[ranked[selected_count - 1]]
    greater = candidates[scores > kth]
    equal = candidates[scores == kth]
    return torch.cat((greater, equal))[:selected_count]


def _relative_boundary_error(value: float, reference: float) -> float:
    """计算精度校验使用的相对误差。"""
    if math.isnan(value) and math.isnan(reference):
        return 0.0
    if value == reference:
        return 0.0
    if not math.isfinite(value) or not math.isfinite(reference):
        return float("inf")

    absolute_error = abs(value - reference)
    if absolute_error <= TOPK_BOUNDARY_ABS_TOLERANCE:
        return 0.0
    if reference == 0.0:
        return float("inf")
    return absolute_error / abs(reference)


def _topk_mismatch_exceeded_count(
    actual_values: list[int],
    expected_values: list[int],
    scores: torch.Tensor,
    forced: set[int],
    visible: int,
    score_precision: int,
) -> int | None:
    """返回TopK边界超差数量；遇到结构性错误时返回None。"""
    actual_set = set(actual_values)
    expected_set = set(expected_values)
    if actual_set == expected_set:
        return 0
    if len(actual_set) != len(actual_values):
        return None
    if any(index < 0 or index >= visible for index in actual_values):
        return None
    if forced - actual_set:
        return None

    expected_dynamic = expected_set - forced
    actual_dynamic = actual_set - forced
    only_actual = actual_dynamic - expected_dynamic
    only_expected = expected_dynamic - actual_dynamic
    if not expected_dynamic and only_actual:
        return None
    if not only_actual:
        return 0

    compare_scores = (
        scores.to(torch.bfloat16).float() if score_precision == 2 else scores
    )
    boundary_score = min(float(compare_scores[index]) for index in expected_dynamic)
    bad_actual = sum(
        _relative_boundary_error(float(compare_scores[index]), boundary_score)
        > TOPK_BOUNDARY_RE_TOLERANCE
        for index in only_actual
    )
    bad_expected = sum(
        _relative_boundary_error(float(compare_scores[index]), boundary_score)
        > TOPK_BOUNDARY_RE_TOLERANCE
        for index in only_expected
    )
    return max(bad_actual, bad_expected)


def _expected_row(
    case, batch: int, q_block: int, scores: torch.Tensor
) -> tuple[torch.Tensor, int]:
    attrs = attributes_from_case(case)
    q_blocks = ceil_div(case.q_seq_lens[batch], case.stem_block_size)
    kv_blocks = ceil_div(case.kv_seq_lens[batch], case.stem_block_size)
    visible = _visible_blocks(case, batch, q_block)
    expected = torch.full((case.kv_blocks,), -1, dtype=torch.int32)
    if visible <= 0:
        return expected, 0

    sink_end = min(case.initial_blocks, visible)
    window_start = max(visible - case.window_size, sink_end)
    candidates = torch.arange(sink_end, window_start, dtype=torch.int64)
    budget = calculate_topk_budget(
        q_block=q_block,
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
        prompt_tokens=case.num_prompt_tokens[batch],
        attrs=attrs,
    )
    selected_count = min(budget, int(candidates.numel()))
    if selected_count:
        candidate_scores = scores[candidates]
        if case.topk_score_precision == 2:
            candidate_scores = candidate_scores.to(torch.bfloat16).float()
        ranked = _reference_topk_order(candidates, candidate_scores, selected_count)
    else:
        ranked = torch.empty(0, dtype=torch.int64)

    selected = torch.cat(
        (
            torch.arange(0, sink_end, dtype=torch.int64),
            ranked,
            torch.arange(window_start, visible, dtype=torch.int64),
        )
    )
    length = int(selected.numel())
    expected[:length] = selected.to(torch.int32)
    return expected, length


def validate_outputs(case, inputs, indices, lens) -> None:
    """采用有效前缀容差规则。形状、有效长度、-1填充、强制sink/window索引、唯一性及索引范围保持严格检查。仅允许第K个分数边界附近的动态TopK交换；超出局部边界容差的交换最多占全部有效输出元素的0.5%。"""
    got_indices = indices.detach().cpu()
    # 先在设备侧转连续，再复制长度，避免非连续视图的D2H搬运问题。
    got_lens = lens.contiguous().detach().cpu()
    q, k, bias = inputs[:3]
    scale = 1.0 / ((case.stem_block_size // case.stem_stride) ** 2)
    group_size = case.q_heads // case.kv_heads
    total_valid_topk_count = 0
    total_exceeded_difference_count = 0
    exceeded_details: list[dict[str, object]] = []

    for batch in range(case.batch_size):
        q_blocks = ceil_div(case.q_seq_lens[batch], case.stem_block_size)
        kv_blocks = ceil_div(case.kv_seq_lens[batch], case.stem_block_size)
        for kv_head in range(case.kv_heads):
            q_head_start = kv_head * group_size
            q_group = q[
                batch, q_head_start : q_head_start + group_size, :q_blocks
            ].float()
            k_group = k[batch, kv_head, :kv_blocks].float()
            scores = torch.matmul(q_group, k_group.transpose(0, 1)) * scale
            scores = scores + bias[batch, kv_head, :kv_blocks].reshape(1, 1, -1)
            scores = scores.cpu()
            for local_head in range(group_size):
                q_head = q_head_start + local_head
                for q_block in range(q_blocks):
                    expected, expected_len = _expected_row(
                        case, batch, q_block, scores[local_head, q_block]
                    )
                    actual_len = int(got_lens[batch, q_head, q_block])
                    if actual_len != expected_len:
                        row_retry = lens[batch, q_head].detach().cpu()
                        raise AssertionError(
                            f"{case.case_id} lens mismatch at "
                            f"B{batch}/H{q_head}/Q{q_block}: "
                            f"got {actual_len}, row_retry "
                            f"{int(row_retry[q_block])}, expected {expected_len}"
                        )
                    actual_row = got_indices[batch, q_head, q_block]
                    total_valid_topk_count += actual_len
                    actual_values = actual_row[:actual_len].tolist()
                    expected_values = expected[:expected_len].tolist()
                    actual_set = set(actual_values)
                    expected_set = set(expected_values)
                    if actual_set == expected_set:
                        continue

                    visible = _visible_blocks(case, batch, q_block)
                    sink_end = min(case.initial_blocks, visible)
                    window_start = max(visible - case.window_size, 0)
                    forced = set(range(sink_end)) | set(range(window_start, visible))
                    row_scores = scores[local_head, q_block]
                    exceeded_count = _topk_mismatch_exceeded_count(
                        actual_values,
                        expected_values,
                        row_scores,
                        forced,
                        visible,
                        case.topk_score_precision,
                    )
                    differing_candidates = sorted(
                        (actual_set - expected_set) | (expected_set - actual_set)
                    )
                    differing_scores = {
                        index: {
                            "fp32": float(row_scores[index]),
                            "bf16": float(row_scores[index].to(torch.bfloat16)),
                        }
                        for index in differing_candidates
                        if 0 <= index < row_scores.numel()
                    }
                    detail = {
                        "row": f"B{batch}/H{q_head}/Q{q_block}",
                        "only_actual": sorted(actual_set - expected_set),
                        "only_expected": sorted(expected_set - actual_set),
                        "differing_scores": differing_scores,
                    }
                    if exceeded_count is None:
                        raise AssertionError(
                            f"{case.case_id} structural indices mismatch: {detail}"
                        )
                    total_exceeded_difference_count += exceeded_count
                    if exceeded_count and len(exceeded_details) < 8:
                        detail["exceeded_count"] = exceeded_count
                        exceeded_details.append(detail)

        if q_blocks < case.q_blocks:
            padded_lens = got_lens[batch, :, q_blocks:]
            padded_indices = got_indices[batch, :, q_blocks:]
            if torch.count_nonzero(padded_lens):
                raise AssertionError(f"{case.case_id} padded q rows have nonzero lens")
            if torch.any(padded_indices != -1):
                raise AssertionError(
                    f"{case.case_id} padded q rows are not filled with -1"
                )

    exceeded_ratio = (
        total_exceeded_difference_count / total_valid_topk_count
        if total_valid_topk_count
        else 0.0
    )
    if exceeded_ratio > MAX_BOUNDARY_RE_EXCEED_RATIO:
        raise AssertionError(
            f"{case.case_id} TopK boundary mismatch ratio exceeds tolerance: "
            f"exceeded={total_exceeded_difference_count}, "
            f"valid_topk={total_valid_topk_count}, ratio={exceeded_ratio:.6g}, "
            f"limit={MAX_BOUNDARY_RE_EXCEED_RATIO:.6g}, details={exceeded_details}"
        )


@pytest.mark.npu
@pytest.mark.parametrize(
    "q_lengths,kv_lengths,prompt_tokens,nq,nkv,d,causal,alpha,"
    "stem_block_size,stem_stride,initial_blocks,window_size,precision,seed",
    STEM_INDEXER_CASES,
)
def test_metadata_then_stem_indexer(
    q_lengths,
    kv_lengths,
    prompt_tokens,
    nq,
    nkv,
    d,
    causal,
    alpha,
    stem_block_size,
    stem_stride,
    initial_blocks,
    window_size,
    precision,
    seed,
    request,
):
    pytest.importorskip("torch_npu")
    case_id = request.node.callspec.id
    # 不切换设备；默认由算子查询当前流可用核数，允许环境变量显式覆盖。
    block_dim_env = os.environ.get("SI_BLOCK_DIM")
    block_dim = int(block_dim_env) if block_dim_env is not None else None
    case, inputs = gen_case(
        q_lengths,
        kv_lengths,
        prompt_tokens,
        nq,
        nkv,
        d,
        causal=causal,
        alpha=alpha,
        stem_block_size=stem_block_size,
        stem_stride=stem_stride,
        initial_blocks=initial_blocks,
        window_size=window_size,
        precision=precision,
        seed=seed,
        case_id=case_id,
    )
    metadata = stem_indexer_metadata(
        inputs[3].npu().contiguous(),
        inputs[4].npu().contiguous(),
        nq,
        nkv,
        causal=causal,
        stem_block_size=stem_block_size,
        window_size=window_size,
        dim_qkflat=d,
        block_dim=block_dim,
    )
    inputs = tuple(t.npu().contiguous() for t in inputs)
    indices, lens = stem_indexer(
        *inputs,
        attrs=attributes_from_case(case),
        block_dim=block_dim,
        metadata=metadata,
    )
    try:
        torch.npu.synchronize()
        validate_outputs(case, inputs, indices, lens)
    finally:
        del indices, lens, inputs
        torch.npu.empty_cache()
