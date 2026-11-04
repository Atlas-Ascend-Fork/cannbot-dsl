# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""QuantBlockSparseAttn Mode1 functional and performance watch tests.

This module owns its case data, input generation, golden calculation, accuracy
comparison, and the six prefill performance entry points.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from quant_block_sparse_attn.quant_block_sparse_attn_metadata import (
    quant_block_sparse_attn_metadata,
)


HEAD_DIM = 128
SPARSE_BLOCK = 128
FP8_MAX = 448.0
EMPTY_LSE = -torch.finfo(torch.float32).max
SOFTMAX_MAX_SENTINEL = -3.4028234663852886e38
MASK_VALUE = -3.4e38

ATTENTION_RTOL = 0.0078125
ATTENTION_ATOL = 0.0001
LSE_RTOL = 0.005
LSE_ATOL = 0.000025
MAX_MISMATCH_RATIO = 0.005


def _generalized_case(
    *,
    layout_q,
    n1,
    n2,
    batch,
    s1,
    s2,
    cu_seqlens_q,
    seqused_kv,
    sparse_mode,
    sparse_pattern,
    block_table_pattern,
    block_num,
    max_block_per_batch,
    mask_mode,
    p_scale,
    return_lse,
    seed=3,
):
    return {
        "layout_q": layout_q,
        "n1": n1,
        "n2": n2,
        "batch": batch,
        "s1": s1,
        "s2": s2,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_kv": seqused_kv,
        "sparse_mode": sparse_mode,
        "sparse_pattern": sparse_pattern,
        "block_table_pattern": block_table_pattern,
        "block_num": block_num,
        "max_block_per_batch": max_block_per_batch,
        "mask_mode": mask_mode,
        "p_scale": p_scale,
        "softmax_scale": HEAD_DIM**-0.5,
        "return_lse": return_lse,
        "q_range": None,
        "k_range": None,
        "v_range": None,
        "seed": seed,
    }


# Complete generalized Mode1 functional watch cases.  The active smoke subset
# is selected below so temporarily disabled cases remain available for bundle
# preparation and later regression.
_ALL_GENERALIZED_CASES = [
    pytest.param(
        _generalized_case(
            layout_q="TND", n1=128, n2=8, batch=1, s1=1, s2=512,
            cu_seqlens_q=(0, 1), seqused_kv=(1,), sparse_mode="dense",
            sparse_pattern="sequential", block_table_pattern="sequential",
            block_num=4, max_block_per_batch=1, mask_mode=3, p_scale=16.0,
            return_lse=True,
        ),
        id="qbsa_fp8_precision_repro_b1_n128_s1_s2_512",
    ),
    pytest.param(
        _generalized_case(
            layout_q="TND", n1=16, n2=2, batch=2, s1=127, s2=255,
            cu_seqlens_q=(0, 81, 154), seqused_kv=(174, 159),
            sparse_mode="random", sparse_pattern="random",
            block_table_pattern="random", block_num=3,
            max_block_per_batch=4, mask_mode=3, p_scale=0.8,
            return_lse=True,
        ),
        id="gen_s127_255_b2_n16_sc2_rand_rand_grad_m1",
    ),
    pytest.param(
        _generalized_case(
            layout_q="NTD", n1=24, n2=3, batch=4, s1=129, s2=257,
            cu_seqlens_q=(0, 129, 258, 387, 516),
            seqused_kv=(257, 257, 257, 257), sparse_mode="random",
            sparse_pattern="sequential", block_table_pattern="random",
            block_num=5, max_block_per_batch=4, mask_mode=0, p_scale=0.9,
            return_lse=True,
        ),
        id="gen_s129_257_b4_n24_sc2_sequ_rand_ones_m1_ntd",
    ),
    pytest.param(
        _generalized_case(
            layout_q="TND", n1=16, n2=2, batch=2, s1=385, s2=129,
            cu_seqlens_q=(0, 385, 770), seqused_kv=(129, 129),
            sparse_mode="random", sparse_pattern="reverse",
            block_table_pattern="sequential", block_num=4,
            max_block_per_batch=3, mask_mode=0, p_scale=0.6,
            return_lse=True,
        ),
        id="gen_s385_129_b2_n16_sc1_reve_reve_grad_m1",
    ),
    pytest.param(
        _generalized_case(
            layout_q="NTD", n1=16, n2=2, batch=1, s1=669, s2=346,
            cu_seqlens_q=(0, 669), seqused_kv=(346,), sparse_mode="random",
            sparse_pattern="dense", block_table_pattern="sequential",
            block_num=6, max_block_per_batch=3, mask_mode=3, p_scale=0.9,
            return_lse=False,
        ),
        id="gen_dens_s669_346_b1_n16_sc2_dens_sequ_ones_m1_ntd",
    ),
    pytest.param(
        _generalized_case(
            layout_q="TND", n1=32, n2=4, batch=2, s1=1094, s2=1059,
            cu_seqlens_q=(0, 942, 1970), seqused_kv=(728, 903),
            sparse_mode="random", sparse_pattern="tail",
            block_table_pattern="sequential", block_num=18,
            max_block_per_batch=9, mask_mode=3, p_scale=0.6,
            return_lse=True,
        ),
        id="gen_tail_s1094_1059_b2_n32_sc6_tail_sequ_grad_m1",
    ),
    pytest.param(
        _generalized_case(
            layout_q="NTD", n1=16, n2=2, batch=2, s1=693, s2=580,
            cu_seqlens_q=(0, 693, 1386), seqused_kv=(580, 580),
            sparse_mode="random", sparse_pattern="empty_tail",
            block_table_pattern="sequential", block_num=12,
            max_block_per_batch=10, mask_mode=0, p_scale=0.9,
            return_lse=False,
        ),
        id="gen_etal_s693_580_b2_n16_sc4_etal_reve_grad_m1_ntd",
    ),
    pytest.param(
        _generalized_case(
            layout_q="TND", n1=16, n2=2, batch=2, s1=711, s2=647,
            cu_seqlens_q=(0, 711, 1422), seqused_kv=(647, 647),
            sparse_mode="random", sparse_pattern="empty",
            block_table_pattern="random", block_num=12,
            max_block_per_batch=9, mask_mode=0, p_scale=0.9,
            return_lse=True,
        ),
        id="gen_empt_s711_647_b2_n16_sc4_empt_rand_ones_m1",
    ),
    pytest.param(
        _generalized_case(
            layout_q="TND", n1=24, n2=3, batch=2, s1=600, s2=857,
            cu_seqlens_q=(0, 542, 917), seqused_kv=(502, 543),
            sparse_mode="random", sparse_pattern="causal",
            block_table_pattern="random", block_num=14,
            max_block_per_batch=10, mask_mode=0, p_scale=0.5,
            return_lse=True,
        ),
        id="gen_s600_857_b2_n24_sc6_caus_rand_grad_m1",
    ),
    pytest.param(
        _generalized_case(
            layout_q="TND", n1=16, n2=8, batch=2, s1=503, s2=301,
            cu_seqlens_q=(0, 332, 737), seqused_kv=(288, 231),
            sparse_mode="random", sparse_pattern="sequential",
            block_table_pattern="sequential", block_num=5,
            max_block_per_batch=3, mask_mode=3, p_scale=0.6,
            return_lse=True,
        ),
        id="gen_s503_301_b2_n16_sc4_sequ_sequ_grad_m1",
    ),
    pytest.param(
        _generalized_case(
            layout_q="NTD", n1=16, n2=1, batch=1, s1=499, s2=303,
            cu_seqlens_q=(0, 499), seqused_kv=(303,), sparse_mode="random",
            sparse_pattern="random", block_table_pattern="random",
            block_num=3, max_block_per_batch=5, mask_mode=0, p_scale=0.6,
            return_lse=True,
        ),
        id="gen_s499_303_b1_n16_sc8_rand_rand_grad_m1_ntd",
    ),
    pytest.param(
        _generalized_case(
            layout_q="NTD", n1=16, n2=2, batch=2, s1=1023, s2=1023,
            cu_seqlens_q=(0, 1023, 2046), seqused_kv=(1023, 1023),
            sparse_mode="random", sparse_pattern="random",
            block_table_pattern="random", block_num=22,
            max_block_per_batch=15, mask_mode=3, p_scale=0.6,
            return_lse=True,
        ),
        id="gen_s1023_1023_b2_n16_sc8_rand_rand_grad_m1_ntd",
    ),
]


_GENERALIZED_CASES_BY_ID = {
    parameter.id: parameter.values[0] for parameter in _ALL_GENERALIZED_CASES
}

# Initial functional validation subset.  The four enabled cases cover the
# precision baseline, NTD layout, tail-block assembly, and a larger randomized
# causal workload.  The measured target for this subset is 10-15 minutes.
_SMOKE_CASE_IDS = (
    "qbsa_fp8_precision_repro_b1_n128_s1_s2_512",
    "gen_s129_257_b4_n24_sc2_sequ_rand_ones_m1_ntd",
    "gen_tail_s1094_1059_b2_n32_sc6_tail_sequ_grad_m1",
    "gen_s1023_1023_b2_n16_sc8_rand_rand_grad_m1_ntd",
)
_GENERALIZED_CASES = [pytest.param(case_id, id=case_id) for case_id in _SMOKE_CASE_IDS]

_ALL_PREFILL_CASE_IDS = (
    "prefill_b1_n32_n4_s8192",
    "prefill_b8_n32_n4_s8192",
    "prefill_b1_n64_n8_s8192",
    "prefill_b8_n32_n4_s16384",
    "prefill_b2_n64_n8_s16384",
    "prefill_b4_n32_n4_s16384",
)

# Keep the performance cases available through the CLI without adding them to
# the default pytest run.
_PREFILL_CASES = []

_DEFAULT_BUNDLE_MANIFEST = (
    Path.home() / "qbsa_stage_logs/qbsa-stage42-six-case-mfu/bundles.json"
)
_DEFAULT_GENERALIZED_BUNDLE_DIR = (
    Path.home() / "qbsa_stage_logs/qbsa-generalized-watch/bundles"
)


def _random_source(
    shape, value_range, generator, amplitude_shape=None, amplitude_low=10.0,
):
    if value_range is not None:
        low, high = value_range
        return torch.empty(shape, dtype=torch.float32).uniform_(
            low, high, generator=generator,
        )
    base = torch.empty(shape, dtype=torch.float32).uniform_(
        -1.0, 1.0, generator=generator,
    )
    if amplitude_shape is None:
        amplitude_shape = shape[:-1] + (1,)
    exponent = torch.empty(amplitude_shape, dtype=torch.float32).uniform_(
        math.log10(amplitude_low), math.log10(1000.0), generator=generator,
    )
    return base * torch.pow(torch.tensor(10.0), exponent)


def _quantize_per_token_head(source):
    maximum = source.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    quant_scale = FP8_MAX / maximum
    value = torch.clamp(source * quant_scale, -FP8_MAX, FP8_MAX)
    return value.to(torch.float8_e4m3fn).contiguous(), (1.0 / quant_scale).squeeze(-1).contiguous()


def _quantize_value_per_head(source):
    maximum = source.abs().amax(dim=(0, 1, 3), keepdim=True).clamp_min(1e-8)
    quant_scale = FP8_MAX / maximum
    value = torch.clamp(source * quant_scale, -FP8_MAX, FP8_MAX)
    return value.to(torch.float8_e4m3fn).contiguous(), (1.0 / quant_scale).reshape(source.shape[2]).contiguous()


def _make_block_table(case, rng):
    # The DSL public contract requires the page-table capacity to cover the
    # sparse-list capacity.  Some generalized cases keep only the used PA
    # slots, so pad their unused table tail without changing any active index.
    table_capacity = max(
        case["max_block_per_batch"],
        math.ceil(max(case["seqused_kv"]) / SPARSE_BLOCK),
    )
    count = case["batch"] * table_capacity
    if case["block_table_pattern"] == "random":
        pages = [rng.randrange(case["block_num"]) for _ in range(count)]
    else:
        pages = [index % case["block_num"] for index in range(count)]
    return torch.tensor(pages, dtype=torch.int32).reshape(
        case["batch"], table_capacity,
    )


def _allowed_blocks(case, q_block, q_length, kv_length):
    block_count = math.ceil(kv_length / SPARSE_BLOCK)
    if case["mask_mode"] == 0:
        return list(range(block_count))
    maximum_token = (q_block + 1) * SPARSE_BLOCK - 1 + kv_length - q_length
    if maximum_token < 0:
        return []
    return list(range(min(block_count - 1, maximum_token // SPARSE_BLOCK) + 1))


def _select_blocks(blocks, count, pattern, rng):
    count = min(count, len(blocks))
    if count == 0 or pattern == "empty":
        return []
    if pattern in ("sequential", "dense", "causal", "empty_tail"):
        return blocks[:count]
    if pattern == "reverse":
        return list(reversed(blocks[-count:]))
    if pattern == "tail":
        selected = blocks[: max(0, count - 1)]
        if blocks[-1] not in selected:
            selected.append(blocks[-1])
        return selected[:count]
    if pattern == "random":
        selected = blocks[:]
        rng.shuffle(selected)
        return selected[:count]
    raise ValueError(f"unsupported sparse pattern: {pattern}")


def _fill_sparse_heads(indices, counts, batch, q_block, selected):
    for head in range(counts.shape[1]):
        counts[batch, head, q_block] = len(selected)
        if selected:
            indices[batch, head, q_block, : len(selected)] = torch.tensor(
                selected, dtype=torch.int32,
            )


def _make_sparse(case, q_lengths, rng):
    q_capacity = math.ceil(case["s1"] / SPARSE_BLOCK)
    sparse_capacity = math.ceil(max(case["seqused_kv"]) / SPARSE_BLOCK)
    indices = torch.full(
        (case["batch"], case["n1"], q_capacity, sparse_capacity),
        -1,
        dtype=torch.int32,
    )
    counts = torch.zeros(
        (case["batch"], case["n1"], q_capacity), dtype=torch.int32,
    )
    per_batch_counts = []
    for kv_length in case["seqused_kv"]:
        maximum = math.ceil(kv_length / SPARSE_BLOCK)
        count = maximum if case["sparse_mode"] == "dense" else rng.randint(0, maximum)
        per_batch_counts.append(count)
    for batch, (q_length, kv_length) in enumerate(zip(q_lengths, case["seqused_kv"])):
        real_q_blocks = math.ceil(q_length / SPARSE_BLOCK)
        for q_block in range(real_q_blocks):
            allowed = _allowed_blocks(case, q_block, q_length, kv_length)
            selected = [] if (
                case["sparse_pattern"] == "empty_tail" and q_block == real_q_blocks - 1
            ) else _select_blocks(
                allowed, per_batch_counts[batch], case["sparse_pattern"], rng,
            )
            _fill_sparse_heads(indices, counts, batch, q_block, selected)
    return indices, counts


def _pack_paged_kv(case, dense_key, dense_value, dense_k_scale, block_table):
    key = torch.zeros(
        (case["block_num"], case["n2"], SPARSE_BLOCK, HEAD_DIM),
        dtype=torch.float8_e4m3fn,
    )
    value = torch.zeros_like(key)
    k_descale = torch.zeros(
        (case["block_num"], case["n2"], SPARSE_BLOCK, 1),
        dtype=torch.float32,
    )
    for batch, kv_length in enumerate(case["seqused_kv"]):
        for logical in range(math.ceil(kv_length / SPARSE_BLOCK)):
            physical = int(block_table[batch, logical])
            begin = logical * SPARSE_BLOCK
            end = min(begin + SPARSE_BLOCK, kv_length)
            size = end - begin
            key[physical, :, :size] = dense_key[batch, begin:end].permute(1, 0, 2)
            value[physical, :, :size] = dense_value[batch, begin:end].permute(1, 0, 2)
            k_descale[physical, :, :size, 0] = dense_k_scale[batch, begin:end].T
    return key, value, k_descale


def _make_query(case, generator, total_q):
    q_shape = (
        (total_q, case["n1"], HEAD_DIM)
        if case["layout_q"] == "TND"
        else (case["n1"], total_q, HEAD_DIM)
    )
    query_source = _random_source(q_shape, case["q_range"], generator)
    return _quantize_per_token_head(query_source)


def _make_paged_kv(case, generator, rng):
    kv_capacity = max(case["seqused_kv"])
    kv_shape = (case["batch"], kv_capacity, case["n2"], HEAD_DIM)
    key_source = _random_source(
        kv_shape, case["k_range"], generator, amplitude_low=1.0,
    )
    value_source = _random_source(
        kv_shape, case["v_range"], generator,
        amplitude_shape=(case["batch"], 1, case["n2"], 1),
        amplitude_low=1.0,
    )
    dense_key, dense_k_descale = _quantize_per_token_head(key_source)
    dense_value, v_descale = _quantize_value_per_head(value_source)
    block_table = _make_block_table(case, rng)
    key, value, k_descale = _pack_paged_kv(
        case, dense_key, dense_value, dense_k_descale, block_table,
    )
    return key, value, k_descale, v_descale, block_table


def _make_inputs(case):
    rng = random.Random(case["seed"])
    generator = torch.Generator().manual_seed(case["seed"])
    q_offsets = tuple(case["cu_seqlens_q"])
    q_lengths = tuple(end - begin for begin, end in zip(q_offsets, q_offsets[1:]))
    query, q_descale = _make_query(case, generator, q_offsets[-1])
    key, value, k_descale, v_descale, block_table = _make_paged_kv(
        case, generator, rng,
    )
    sparse_indices, sparse_seq_len = _make_sparse(case, q_lengths, rng)
    cu_seqlens_q = torch.tensor(q_offsets, dtype=torch.int32)
    return {
        "query": query,
        "key": key,
        "value": value,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "p_scale": torch.tensor([case["p_scale"]], dtype=torch.float32),
        "sparse_indices": sparse_indices,
        "sparse_seq_len": sparse_seq_len,
        "atten_mask": (
            torch.triu(torch.ones((2048, 2048), dtype=torch.uint8))
            if case["mask_mode"] == 3 else None
        ),
        "softmax_scale": case["softmax_scale"],
        "sparse_q_block_size": SPARSE_BLOCK,
        "sparse_kv_block_size": SPARSE_BLOCK,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_kv": torch.tensor(case["seqused_kv"], dtype=torch.int32),
        "block_table": block_table,
        "layout_q": case["layout_q"],
        "layout_kv": "PA_BNBD",
        "layout_sparse_indices": "B_N_Qb_Kb",
        "quant_mode": 1,
        "mask_mode": case["mask_mode"],
        "return_softmax_lse": case["return_lse"],
    }


def _prefill_bundle_manifest(override=None):
    configured = override or os.environ.get("QBSA_PREFILL_BUNDLE_MANIFEST")
    return Path(configured) if configured else _DEFAULT_BUNDLE_MANIFEST


def _load_bundle(bundle_path):
    if not bundle_path.is_file():
        raise FileNotFoundError(f"QBSA case bundle not found: {bundle_path}")
    bundle = torch.load(bundle_path, weights_only=True, mmap=True)
    required = ("inputs", "expected", "expected_lse")
    missing = tuple(name for name in required if name not in bundle)
    if missing:
        raise KeyError(f"QBSA case bundle {bundle_path} is missing {missing}")
    inputs = dict(bundle["inputs"])
    # Metadata is a runtime AICPU output, never a frozen test input.
    inputs.pop("metadata", None)
    return inputs, bundle["expected"], bundle["expected_lse"]


def _load_prefill_bundle(case_id, manifest_path=None):
    manifest_path = _prefill_bundle_manifest(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"prefill bundle manifest not found: {manifest_path}; set "
            "QBSA_PREFILL_BUNDLE_MANIFEST or pass --bundle-manifest"
        )
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if case_id not in manifest:
        raise KeyError(f"prefill case {case_id!r} is absent from {manifest_path}")
    bundle_path = Path(manifest[case_id]).expanduser()
    return _load_bundle(bundle_path)


def _generalized_bundle_dir():
    configured = os.environ.get("QBSA_GENERALIZED_BUNDLE_DIR")
    return Path(configured) if configured else _DEFAULT_GENERALIZED_BUNDLE_DIR


def _load_or_create_generalized_bundle(case_id):
    """Reuse cached inputs/golden, creating the bundle only on first use."""
    case = _GENERALIZED_CASES_BY_ID[case_id]
    bundle_dir = _generalized_bundle_dir()
    bundle_path = bundle_dir / f"{case_id}.pt"
    if bundle_path.is_file():
        return _load_bundle(bundle_path)

    inputs = _make_inputs(case)
    expected, expected_lse = _golden(inputs)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "case_id": case_id,
            "inputs": inputs,
            "expected": expected,
            "expected_lse": expected_lse,
        },
        bundle_path,
    )
    return inputs, expected, expected_lse


def _scale_at(scale, layout, token, head):
    return (
        scale[token, head].reshape(-1)
        if layout == "TND"
        else scale[head, token].reshape(-1)
    )


@dataclass
class _GoldenBlock:
    inputs: dict
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    batch_idx: int
    head: int
    kv_head: int
    q_block: int
    q_begin: int
    q_length: int
    kv_length: int
    log_p_scale: float

    def __post_init__(self):
        self.local_begin = self.q_block * SPARSE_BLOCK
        self.local_end = min(self.local_begin + SPARSE_BLOCK, self.q_length)
        self.token = slice(self.q_begin + self.local_begin, self.q_begin + self.local_end)
        self.q_tile = (
            self.query[self.token, self.head]
            if self.inputs["layout_q"] == "TND"
            else self.query[self.head, self.token]
        )
        self.q_scale = _scale_at(
            self.inputs["q_descale"], self.inputs["layout_q"], self.token, self.head,
        )
        rows = self.local_end - self.local_begin
        self.running_max = torch.full((rows,), SOFTMAX_MAX_SENTINEL)
        self.running_offset = self.running_max.clone()
        self.running_sum = torch.zeros(rows)
        self.accumulator = torch.zeros((rows, HEAD_DIM))
        self.chunks = 0

    def consume(self, slot, count):
        positions = self._positions(slot, count)
        if positions is None:
            return
        scores, visible, v_tile, local_max = self._scores(positions)
        alpha, probability, new_max, new_offset, has_value = self._probabilities(
            scores, visible, local_max,
        )
        pv = probability.to(torch.float8_e4m3fn).float() @ v_tile
        v_scale = float(self.inputs["v_descale"][self.kv_head])
        if self.chunks == 0:
            self.accumulator = pv
        elif self.chunks == 1:
            self.accumulator = (
                self.accumulator * alpha[:, None] * v_scale + pv * v_scale
            )
        else:
            self.accumulator = self.accumulator * alpha[:, None] + pv * v_scale
        self.running_sum = self.running_sum * alpha + probability.sum(dim=1)
        self.running_max = torch.where(has_value, new_max, self.running_max)
        self.running_offset = torch.where(has_value, new_offset, self.running_offset)
        self.chunks += 1

    def finish(self):
        nonempty = self.running_sum > 0
        final_scale = float(self.inputs["v_descale"][self.kv_head]) if self.chunks <= 1 else 1.0
        result = torch.where(
            nonempty[:, None],
            self.accumulator * final_scale / self.running_sum[:, None],
            0.0,
        )
        lse = torch.where(
            nonempty,
            torch.log(self.running_sum) + self.running_offset,
            torch.full_like(self.running_sum, EMPTY_LSE),
        )
        return result.bfloat16(), lse

    def _positions(self, slot, count):
        logical_blocks = self.inputs["sparse_indices"][
            self.batch_idx, self.head, self.q_block, slot:min(slot + 2, count)
        ]
        positions = []
        for logical in sorted(logical_blocks.tolist()):
            begin = logical * SPARSE_BLOCK
            positions.extend(range(begin, min(begin + SPARSE_BLOCK, self.kv_length)))
        return torch.tensor(positions, dtype=torch.long) if positions else None

    def _scores(self, positions):
        physical = self.inputs["block_table"][
            self.batch_idx, positions // SPARSE_BLOCK
        ].long()
        in_page = positions % SPARSE_BLOCK
        k_tile = self.key[physical, self.kv_head, in_page]
        v_tile = self.value[physical, self.kv_head, in_page]
        k_scale = self.inputs["k_descale"][
            physical, self.kv_head, in_page, 0
        ].float()
        scores = self.q_tile @ k_tile.T
        scores *= self.q_scale[:, None] * self.inputs["softmax_scale"]
        scores *= k_scale[None, :]
        visible = torch.ones_like(scores, dtype=torch.bool)
        if self.inputs["mask_mode"] == 3:
            query_positions = torch.arange(self.local_begin, self.local_end)[:, None]
            visible &= positions[None, :] <= query_positions + self.kv_length - self.q_length
        scores = torch.where(visible, scores, torch.full_like(scores, MASK_VALUE))
        local_max = torch.where(
            visible, scores, torch.full_like(scores, SOFTMAX_MAX_SENTINEL),
        ).max(dim=1).values
        return scores, visible, v_tile, local_max

    def _probabilities(self, scores, visible, local_max):
        has_value = visible.any(dim=1)
        started = self.running_max != SOFTMAX_MAX_SENTINEL
        new_max = torch.where(
            started, torch.maximum(self.running_max, local_max), local_max,
        )
        new_max = torch.where(has_value, new_max, self.running_max)
        new_offset = new_max - self.log_p_scale
        alpha = torch.where(
            started,
            torch.exp(self.running_offset.double() - new_offset.double()).float(),
            torch.zeros_like(self.running_max),
        )
        alpha = torch.where(torch.isfinite(alpha), alpha, 0.0)
        probability = torch.exp(
            scores.double() - new_offset.double()[:, None]
        ).float()
        probability = torch.where(
            visible & has_value[:, None], probability, 0.0,
        )
        return alpha, probability, new_max, new_offset, has_value




def _golden(inputs):
    query = inputs["query"].float()
    key = inputs["key"].float()
    value = inputs["value"].float()
    batch, n1 = inputs["sparse_indices"].shape[:2]
    group_size = n1 // key.shape[1]
    total_q = int(inputs["cu_seqlens_q"][-1])
    output = torch.zeros((total_q, n1, HEAD_DIM), dtype=torch.bfloat16)
    lse_output = torch.full((n1, total_q), EMPTY_LSE, dtype=torch.float32)
    log_p_scale = math.log(float(inputs["p_scale"][0]))
    for batch_idx in range(batch):
        q_begin = int(inputs["cu_seqlens_q"][batch_idx])
        q_end = int(inputs["cu_seqlens_q"][batch_idx + 1])
        q_length = q_end - q_begin
        kv_length = int(inputs["seqused_kv"][batch_idx])
        for head in range(n1):
            for q_block in range(math.ceil(q_length / SPARSE_BLOCK)):
                count = int(inputs["sparse_seq_len"][batch_idx, head, q_block])
                if count == 0:
                    continue
                block = _GoldenBlock(
                    inputs, query, key, value, batch_idx, head, head // group_size,
                    q_block, q_begin, q_length, kv_length, log_p_scale,
                )
                for slot in range(0, count, 2):
                    block.consume(slot, count)
                result, lse = block.finish()
                output[block.token, head] = result
                lse_output[head, block.token] = lse
    return output, lse_output if inputs["return_softmax_lse"] else None


def _assert_accuracy(actual, expected, rtol, atol):
    assert actual.shape == expected.shape
    actual = actual.float()
    expected = expected.float()
    for classify in (torch.isnan, torch.isposinf, torch.isneginf):
        assert torch.equal(classify(actual), classify(expected))
    finite = torch.isfinite(expected)
    failures = ~torch.isclose(actual[finite], expected[finite], rtol=rtol, atol=atol)
    failure_count = int(failures.sum())
    ratio = failure_count / max(1, expected.numel())
    assert ratio <= MAX_MISMATCH_RATIO, (
        f"accuracy mismatch: {failure_count}/{expected.numel()} "
        f"({ratio:.6%}), rtol={rtol}, atol={atol}"
    )


def _check_result(output, lse, expected, expected_lse):
    actual = output.cpu()
    assert actual.dtype == torch.bfloat16 and torch.isfinite(actual.float()).all()
    _assert_accuracy(actual, expected, ATTENTION_RTOL, ATTENTION_ATOL)
    if expected_lse is None:
        assert lse is None
        return
    actual_lse = lse.cpu()
    empty = expected_lse == EMPTY_LSE
    assert actual_lse.dtype == torch.float32
    assert torch.equal(actual_lse == EMPTY_LSE, empty)
    _assert_accuracy(actual_lse, expected_lse, LSE_RTOL, LSE_ATOL)


def _to_device(inputs, device):
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }


def _with_aicpu_metadata(inputs, device_inputs):
    result = dict(device_inputs)
    sparse_seq_len = result["sparse_seq_len"]
    result["metadata"] = quant_block_sparse_attn_metadata(
        sparse_seq_len,
        sparse_seq_len.shape[1],
        result["key"].shape[1],
        result["query"].shape[-1],
        cu_seqlens_q=result["cu_seqlens_q"],
        seqused_kv=result["seqused_kv"],
        batch_size=sparse_seq_len.shape[0],
        sparse_block_size_q=inputs["sparse_q_block_size"],
        sparse_block_size_k=inputs["sparse_kv_block_size"],
        quant_mode=inputs["quant_mode"],
        mask_mode=inputs["mask_mode"],
        layout_q=inputs["layout_q"],
        layout_kv=inputs["layout_kv"],
        layout_sparse_indices=inputs.get(
            "layout_sparse_indices", "B_N_Qb_Kb"
        ),
    )
    return result


def _run_functional_case(case_id, device):
    from quant_block_sparse_attn.quant_block_sparse_attn import quant_block_sparse_attn

    inputs, expected, expected_lse = _load_or_create_generalized_bundle(case_id)
    device_inputs = _with_aicpu_metadata(inputs, _to_device(inputs, device))
    output, lse = quant_block_sparse_attn(**device_inputs)
    torch.npu.synchronize()
    _check_result(output, lse, expected, expected_lse)


def _run_prefill_case(case_id, device, manifest_path=None):
    from quant_block_sparse_attn.quant_block_sparse_attn import quant_block_sparse_attn

    inputs, expected, expected_lse = _load_prefill_bundle(case_id, manifest_path)
    device_inputs = _with_aicpu_metadata(inputs, _to_device(inputs, device))
    output, lse = quant_block_sparse_attn(**device_inputs)
    torch.npu.synchronize()
    _check_result(output, lse, expected, expected_lse)


@pytest.mark.npu
@pytest.mark.parametrize("case_id", _GENERALIZED_CASES)
def test_quant_block_sparse_attn(case_id):
    pytest.importorskip("torch_npu")
    device = os.environ.get("QBSA_TEST_DEVICE", "npu:0")
    torch.npu.set_device(device)
    _run_functional_case(case_id, device)


if _PREFILL_CASES:

    @pytest.mark.npu
    @pytest.mark.slow
    @pytest.mark.parametrize("case_id", _PREFILL_CASES)
    def test_quant_block_sparse_attn_prefill(case_id):
        pytest.importorskip("torch_npu")
        device = os.environ.get("QBSA_TEST_DEVICE", "npu:0")
        torch.npu.set_device(device)
        _run_prefill_case(case_id, device)


def _summary(samples):
    return {
        "count": len(samples),
        "min_us": min(samples),
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "max_us": max(samples),
    }


def _benchmark(case_id, device, warmup, repeat, manifest_path):
    from quant_block_sparse_attn.quant_block_sparse_attn import quant_block_sparse_attn

    inputs, expected, expected_lse = _load_prefill_bundle(case_id, manifest_path)
    device_inputs = _with_aicpu_metadata(inputs, _to_device(inputs, device))
    output, lse = quant_block_sparse_attn(**device_inputs)
    torch.npu.synchronize()
    _check_result(output, lse, expected, expected_lse)
    for _ in range(warmup):
        quant_block_sparse_attn(**device_inputs)
        torch.npu.synchronize()
    samples = []
    for _ in range(repeat):
        torch.npu.synchronize()
        begin = time.perf_counter_ns()
        output, lse = quant_block_sparse_attn(**device_inputs)
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns() - begin) / 1000)
    _check_result(output, lse, expected, expected_lse)
    return {
        "case_id": case_id,
        "correctness": "passed",
        "public_sync_wall": _summary(samples),
        "samples_us": samples,
    }


def main():
    output_logger = logging.getLogger("qbsa.benchmark.output")
    output_logger.setLevel(logging.INFO)
    output_logger.propagate = False
    output_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    output_logger.addHandler(handler)
    available = _ALL_PREFILL_CASE_IDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=available)
    parser.add_argument("--device", default=os.environ.get("QBSA_TEST_DEVICE", "npu:0"))
    parser.add_argument("--bundle-manifest", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        output_logger.info("\n".join(available))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        parser.error("output already exists")
    if args.warmup < 1 or args.repeat < 2:
        parser.error("warmup must be >= 1 and repeat must be >= 2")
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    report = {
        "device": args.device,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "cases": [],
    }
    for case_id in args.case or available:
        result = _benchmark(
            case_id, args.device, args.warmup, args.repeat, args.bundle_manifest,
        )
        report["cases"].append(result)
        output_logger.info("CASE_RESULT %s", json.dumps(result))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
