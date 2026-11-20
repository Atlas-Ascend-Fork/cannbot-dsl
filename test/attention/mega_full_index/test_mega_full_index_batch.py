# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Batch isolation and exact equivalence to independent B=1 executions."""

import argparse
from dataclasses import dataclass
import math
import time

import pytest
import torch

from samples.attention.mega_full_index.mega_full_index import (
    check_page_bounds,
    prepare_mega_full_index,
)

if __package__:
    from . import test_mega_full_index as reference
else:
    import test_mega_full_index as reference


ROW_INPUTS = (
    "x",
    "qr",
    "descale_qr",
    "q_rope_sin",
    "q_rope_cos",
    "latent",
    "k_rope_sin",
    "k_rope_cos",
    "output_idx_offset",
)
SHARED_INPUTS = ("wqb", "ww", "descale_wqb", "wk", "norm_weight")


@dataclass(frozen=True)
class BatchCase:
    batch: int
    seq: int
    mode: int
    kv: int = 129
    graph: bool = False
    values: bool = True
    padded: bool = False

    @property
    def name(self):
        return f"b{self.batch}-s{self.seq}-kv{self.kv}-m{self.mode}"


CASES = (
    tuple(
        BatchCase(
            batch,
            seq,
            mode,
            graph=(batch == 3 and seq == 16 and mode == 1),
            values=(mode == 1),
            padded=(seq == 6),
        )
        for batch in (2, 3, 4)
        for seq in (6, 8, 16)
        for mode in (0, 1)
    )
    + tuple(BatchCase(batch, 8, batch % 2, kv=8192) for batch in (2, 3, 4))
    + (
        BatchCase(2, 6, 1, kv=16385),
        BatchCase(3, 8, 0, kv=131073),
        BatchCase(4, 16, 1, kv=1048576),
    )
)


def _padded_cache(tensor):
    stride = (tensor.stride(0) + 64, *tensor.stride()[1:])
    span = (tensor.shape[0] - 1) * stride[0] + tensor[0].numel()
    backing = torch.full((span,), 165, device=tensor.device, dtype=tensor.dtype)
    view = torch.as_strided(backing, tensor.shape, stride)
    view.copy_(tensor)
    return view, backing


def make_batch(spec):
    """Different sequences share projection weights and a physical cache pool."""
    singles = []
    for batch in range(spec.batch):
        length = max(1, spec.kv - batch * 17)
        active_q = max(1, min(spec.seq - batch, length))
        inputs = reference.make_inputs(
            spec.seq,
            length,
            seed=31415 + batch,
            capacity=spec.kv,
            active_q=active_q,
            permute=True,
        )
        if singles:
            inputs.update({name: singles[0][name] for name in SHARED_INPUTS})
        inputs["output_idx_offset"] = (
            torch.arange(spec.seq, dtype=torch.int32, device="npu") + batch * 101
        )[:, None]
        # Exercise skipped and duplicate slots, including an entirely empty
        # sequence, while other sequences retain nonzero effective lengths.
        if batch == 1:
            inputs["cache_index"][0] = -1
            inputs["cache_index"][-1] = inputs["cache_index"][1]
        if batch == spec.batch - 1 and spec.kv == 129:
            inputs["seqused_q"].zero_()
            inputs["seqused_k"].zero_()
            inputs["cache_index"].fill_(-1)
        if spec.mode == 0:
            data, scale = reference.unpack_cache(inputs["k_cache"])
            inputs["k_cache"], inputs["k_scale_cache"] = data.npu(), scale.npu()
        singles.append(inputs)

    combined = dict(singles[0])
    for name in ROW_INPUTS:
        combined[name] = torch.cat([inputs[name] for inputs in singles])
    pages = singles[0]["k_cache"].shape[0]
    combined["k_cache"] = torch.cat([inputs["k_cache"] for inputs in singles])
    combined["k_scale_cache"] = (
        torch.cat([inputs["k_scale_cache"] for inputs in singles])
        if spec.mode == 0
        else None
    )
    combined["block_table"] = torch.cat(
        [inputs["block_table"] + batch * pages for batch, inputs in enumerate(singles)]
    )
    combined["cache_index"] = torch.cat(
        [
            torch.where(
                inputs["cache_index"] >= 0,
                inputs["cache_index"] + batch * pages * 128,
                -1,
            )
            for batch, inputs in enumerate(singles)
        ]
    )
    combined["cu_seqlens_q"] = torch.arange(
        0, (spec.batch + 1) * spec.seq, spec.seq, dtype=torch.int32, device="npu"
    )
    for name in ("seqused_q", "seqused_k"):
        combined[name] = torch.cat([inputs[name] for inputs in singles])
    padding = []
    if spec.padded:
        for name in ("k_cache", "k_scale_cache") if spec.mode == 0 else ("k_cache",):
            combined[name], backing = _padded_cache(combined[name])
            padding.append((combined[name], backing, backing.cpu().clone()))
    return singles, combined, padding


def run_batch_case(spec):
    singles, inputs, padding = make_batch(spec)
    check_page_bounds(inputs["k_cache"], inputs["block_table"], inputs["cache_index"])
    expected = []
    reference_stages = []
    for single in singles:
        initial_data = single["k_cache"].cpu().clone()
        initial_scale = (
            single["k_scale_cache"].cpu().clone() if spec.mode == 0 else None
        )
        stage = prepare_mega_full_index(
            **single,
            softmax_scale=1 / 64,
            norm_eps=1e-20,
            storage_mode=spec.mode,
            return_value=spec.values,
        )
        stage.run()
        torch.npu.synchronize()
        if spec.kv <= 16385:
            case = reference.Case(
                spec.seq, spec.kv, single, stage, initial_data, initial_scale
            )
            expected_q, expected_scale = reference.query_reference(single)
            assert torch.equal(stage.outputs[0].cpu(), expected_q), "packed Q mismatch"
            assert torch.equal(stage.outputs[1].cpu(), expected_scale), (
                "Q scale mismatch"
            )
            reference.check_query_weight(case)
            reference.check_key_cache(case)
            reference.check_qli(case)
        expected.append([value.cpu().clone() for value in stage.outputs])
        if spec.graph:
            reference_stages.append(stage)

    total = spec.batch * spec.seq
    outputs = {}
    guards = []
    for name, shape, dtype in (
        ("q", (total, 32, 64), torch.uint8),
        ("descale_q", (total, 32, 2, 2), torch.uint8),
        ("w", (total, 32), torch.float32),
    ):
        count = math.prod(shape)
        storage = torch.full((count + 8192,), 123, dtype=dtype, device="npu")
        output_end = 4096 + count
        outputs[name] = storage[4096:output_end].view(shape)
        guards.append(storage)
    stage = prepare_mega_full_index(
        **inputs,
        **outputs,
        softmax_scale=1 / 64,
        norm_eps=1e-20,
        storage_mode=spec.mode,
        return_value=spec.values,
    )
    stage.run()
    torch.npu.synchronize()
    for index in (0, 1, 2, 4, 5, 6, 7):
        actual = stage.outputs[index].cpu()
        target = torch.cat([output[index] for output in expected])
        assert torch.equal(actual, target), (
            f"{spec.name}: output {index} differs from B=1"
        )
    assert stage.outputs[3] is inputs["k_cache"]
    assert torch.equal(
        stage.outputs[3].cpu(), torch.cat([output[3] for output in expected])
    )
    if spec.mode == 0:
        assert torch.equal(
            inputs["k_scale_cache"].cpu(),
            torch.cat([i["k_scale_cache"].cpu() for i in singles]),
        )
    assert all(
        stage.outputs[index] is outputs[name]
        for index, name in enumerate(("q", "descale_q", "w"))
    )
    before = [value.cpu().clone() for value in stage.outputs]
    stage.run()
    torch.npu.synchronize()
    reference.assert_identical(before, [value.cpu() for value in stage.outputs])
    if spec.graph:
        graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        with torch.npu.graph(graph, stream=stream):
            stage.run()
        for _ in range(2):
            graph.replay()
            torch.npu.synchronize()
            reference.assert_identical(before, [value.cpu() for value in stage.outputs])
        changed = []
        for batch, single_stage in enumerate(reference_stages):
            batch_end = batch + 1
            active_q, used_k = (0, 0) if batch == 1 else (spec.seq - batch, spec.kv - 1)
            for name, value in (("seqused_q", active_q), ("seqused_k", used_k)):
                singles[batch][name].fill_(value)
                inputs[name][batch:batch_end].fill_(value)
            single_stage.run()
            torch.npu.synchronize()
            changed.append([value.cpu().clone() for value in single_stage.outputs])
        graph.replay()
        torch.npu.synchronize()
        for index in (0, 1, 2, 4, 5, 6, 7):
            assert torch.equal(
                stage.outputs[index].cpu(),
                torch.cat([output[index] for output in changed]),
            )
    for storage in guards:
        assert bool((storage[:4096].cpu() == 123).all())
        assert bool((storage[-4096:].cpu() == 123).all())
    for tensor, backing, original in padding:
        after = backing.cpu()
        for page in range(tensor.shape[0] - 1):
            begin = page * tensor.stride(0) + tensor[page].numel()
            end = begin + 64
            assert torch.equal(after[begin:end], original[begin:end])
    print("BATCH_VALIDATION_PASS", spec.name, flush=True)


@pytest.mark.npu
@pytest.mark.parametrize("spec", CASES, ids=lambda spec: spec.name)
def test_mega_full_index_batch(spec):
    run_batch_case(spec)


@pytest.mark.npu
@pytest.mark.parametrize("batch", (0, 5))
def test_unsupported_batch(batch):
    inputs = reference.make_inputs(8, 129)
    inputs["cu_seqlens_q"] = torch.zeros(batch + 1, dtype=torch.int32, device="npu")
    with pytest.raises(ValueError, match="B=1/2/3/4"):
        prepare_mega_full_index(**inputs, softmax_scale=1 / 64, norm_eps=1e-20)


@pytest.mark.npu
def test_mismatched_batch_metadata():
    _, inputs, _ = make_batch(BatchCase(2, 8, 1))
    inputs["seqused_k"] = inputs["seqused_k"][:1]
    with pytest.raises(ValueError, match="B=1/2/3/4"):
        prepare_mega_full_index(**inputs, softmax_scale=1 / 64, norm_eps=1e-20)


def test_export_batch_domains(monkeypatch):
    """Native providers carry explicit B/T domains and close every handle."""
    from samples.attention.mega_full_index import mega_full_index as entry
    from samples.attention.mega_full_index import qli

    calls, closed = [], []

    class Handle:
        def close(self):
            closed.append(self)

    def record(*args, batch_count=1):
        calls.append((batch_count, args[8], args[10]))
        return Handle()

    monkeypatch.setattr(entry, "_compile_stages", record)
    entry.export_mega_full_index()
    assert set(calls) == {
        (batch, seq, mode)
        for batch in (1, 2, 3, 4)
        for seq in (6, 8, 16)
        for mode in (0, 1)
    }
    assert len(calls) == len(closed) == 24
    config = (512, True, 2048, 3, 1, True, True, True, True, 128, 8, 32, 6, False)
    for batch in (1, 2, 3, 4):
        _, specs = qli.build_runner(config, 8, batch_count=batch)
        assert specs[2].shape == (batch * 8, 32)
        assert specs[7].shape == (batch + 1,)
        assert specs[8].shape == specs[9].shape == (batch,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[spec.name for spec in CASES], required=True)
    args = parser.parse_args()
    started = time.monotonic()
    run_batch_case(next(spec for spec in CASES if spec.name == args.case))
    print(f"ELAPSED {time.monotonic() - started:.3f}s", flush=True)
