# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Representative NPU accuracy coverage for Flash MLA with paged KV cache.

This is a compact functional subset of the historical 204-case regression
suite, restricted to the public mainline interface. Cases deliberately cross
multiple behaviors instead of repeating geometry: dtype, both supported KV
layouts, tails, ragged/empty batches, seqused padding, causal masking,
FlashDecode and metadata.
Every end-to-end case runs metadata followed by the main operator and compares
against an independent CPU reference.
"""

import math

import pytest
import torch
from _samples_path import load_sample

flash_mla_module = load_sample(
    "attention/flash_mla_with_kvcache/flash_mla_with_kvcache.py"
)
metadata_module = load_sample(
    "attention/flash_mla_with_kvcache/flash_mla_with_kvcache_metadata.py"
)


@pytest.fixture(autouse=True)
def _automatic_buffer_sync(monkeypatch):
    """Exercise the sample with compiler-generated Buffer synchronization."""
    monkeypatch.setenv("CANNBOTDSL_AUTO_BUFFER_SYNC", "1")


DEFAULT_RTOL = 0.005
BFLOAT16_RTOL = 0.0078125
DEFAULT_ATOL = 0.000025
BFLOAT16_ATOL = BFLOAT16_RTOL
FAIL_RATIO = 0.005
MAX_RELATIVE_ERROR = 10.0
RELATIVE_FLOOR = (1.0 / (1 << 14)) / 0.005
RELATIVE_EPSILON = 2e-9
FULL_D_FAILURE_RATIO = 0.8
LSE_ATOL = 0.1


def _torch_tolerances(dtype):
    if dtype == torch.float16:
        return DEFAULT_ATOL, DEFAULT_RTOL
    if dtype == torch.bfloat16:
        return BFLOAT16_ATOL, BFLOAT16_RTOL
    raise ValueError(f"unsupported precision dtype: {dtype}")


def _compare_stats(actual, golden, atol, rtol):
    if actual.shape != golden.shape:
        return {
            "shape_ok": False,
            "fail_ratio": 1.0,
            "max_rel": 0.0,
            "full_d_failure": False,
        }
    actual = actual.float()
    golden = golden.float()
    difference = (actual - golden).abs()
    exact = actual == golden
    absolute_pass = exact | (difference <= atol)
    relative_error = difference / (
        golden.abs().clamp(min=RELATIVE_FLOOR) + RELATIVE_EPSILON
    )
    bad = ~absolute_pass & ~(exact | (relative_error <= rtol))
    fail_ratio = float(bad.sum()) / difference.numel()
    needs_relative = ~absolute_pass
    max_relative = (
        relative_error[needs_relative].max().item() if needs_relative.any() else 0.0
    )
    bad_rows = bad.reshape(-1, actual.shape[-1])
    row_failure_ratio = bad_rows.float().mean(dim=1)
    full_d_failure = bool(
        bad_rows.all(dim=1).any() or (row_failure_ratio > FULL_D_FAILURE_RATIO).any()
    )
    return {
        "shape_ok": True,
        "fail_ratio": fail_ratio,
        "max_rel": max_relative,
        "full_d_failure": full_d_failure,
    }


def _precision_pass(stats):
    return (
        stats["shape_ok"]
        and not stats["full_d_failure"]
        and stats["fail_ratio"] <= FAIL_RATIO
        and stats["max_rel"] <= MAX_RELATIVE_ERROR
    )


def _case(
    case_id,
    spans,
    kv_lengths,
    *,
    heads=64,
    dtype="fp16",
    layout_q="TND",
    layout_kv="PA_NZ",
    layout_out="NTD",
    block_size=128,
    mask_mode=0,
    seqused=None,
    scale=None,
    return_lse=True,
    seed=42,
    expect_fd=False,
):
    assert len(spans) == len(kv_lengths)
    if layout_q != "TND":
        assert len(set(spans)) == 1
    cumulative = [0]
    for span in spans:
        cumulative.append(cumulative[-1] + span)
    config = {
        "B": len(spans),
        "N1": heads,
        "N2": 1,
        "S1": max(spans),
        "S2": max(kv_lengths),
        "D": 576,
        "DV": 512,
        "Dtype": dtype,
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_out": layout_q if layout_out is None else layout_out,
        "block_size": block_size,
        "mask_mode": mask_mode,
        "cache_seqlens": kv_lengths,
        "return_softmax_lse": return_lse,
        "seed": seed,
    }
    if layout_q == "TND":
        config["cu_seqlens_q"] = cumulative
    if seqused is not None:
        config["seqused_q"] = seqused
    if scale is not None:
        config["scale"] = scale
    return {
        "id": case_id,
        "source_case": "functional_subset_204",
        "config": config,
        "spans": spans,
        "effective_q": spans if seqused is None else seqused,
        "expect_fd": expect_fd,
    }


def _generate(case):
    config = case["config"]
    spans = case["spans"]
    cumulative = [0]
    for span in spans:
        cumulative.append(cumulative[-1] + span)

    heads = config["N1"]
    block_size = config["block_size"]
    kv_lengths = config["cache_seqlens"]
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[config["Dtype"]]
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    q_cpu = torch.randn((cumulative[-1], heads, 576), dtype=dtype, generator=generator)
    logical_kv = torch.randn(
        (max(sum(kv_lengths), 1), 576), dtype=dtype, generator=generator
    )

    pages_per_batch = [(length + block_size - 1) // block_size for length in kv_lengths]
    page_count = sum(pages_per_batch)
    block_table = torch.zeros(
        (len(kv_lengths), max(max(pages_per_batch), 1)), dtype=torch.int32
    )
    physical_pages = torch.zeros((page_count + 3, block_size, 576), dtype=dtype)
    permutation = torch.randperm(page_count, generator=generator).tolist()
    logical_start = 0
    page_index = 0
    for batch, length in enumerate(kv_lengths):
        for column in range(pages_per_batch[batch]):
            page = permutation[page_index]
            page_index += 1
            block_table[batch, column] = page
            count = min(block_size, length - column * block_size)
            source_start = logical_start + column * block_size
            physical_pages[page, :count] = logical_kv[
                source_start : source_start + count
            ]
        logical_start += length

    if config["layout_kv"] == "PA_NZ":
        k_cache = (
            physical_pages.reshape(page_count + 3, block_size, 36, 16)
            .permute(0, 2, 1, 3)
            .contiguous()
            .unsqueeze(1)
        )
    elif config["layout_kv"] == "PA_BNBD":
        k_cache = physical_pages.unsqueeze(1)
    else:
        k_cache = physical_pages.unsqueeze(2)

    if config["layout_q"] == "TND":
        q_layout = q_cpu
    elif config["layout_q"] == "BSND":
        q_layout = q_cpu.reshape(config["B"], spans[0], heads, 576)
    else:
        q_layout = (
            q_cpu.reshape(config["B"], spans[0], heads, 576)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
    return (
        cumulative,
        q_cpu,
        logical_kv,
        q_layout.npu(),
        k_cache.npu(),
        block_table.npu(),
    )


def _golden(case, cumulative, q_cpu, logical_kv):
    config = case["config"]
    heads = config["N1"]
    scale = config.get("scale", 1.0 / math.sqrt(576.0))
    kv_lengths = config["cache_seqlens"]
    spans = case["spans"]
    used_lengths = case["effective_q"]
    output = torch.zeros((cumulative[-1], heads, 512), dtype=torch.float32)
    lse = torch.full((cumulative[-1], heads), float("inf"), dtype=torch.float32)
    kv_start = 0
    for batch, (span, kv_length, used) in enumerate(
        zip(spans, kv_lengths, used_lengths)
    ):
        if kv_length and used:
            query = (
                q_cpu[cumulative[batch] : cumulative[batch] + used]
                .reshape(-1, 576)
                .double()
            )
            key = logical_kv[kv_start : kv_start + kv_length].double()
            output_batch = output[cumulative[batch] : cumulative[batch] + used].view(
                -1, 512
            )
            lse_batch = lse[cumulative[batch] : cumulative[batch] + used].view(-1)
            for row_start in range(0, used * heads, 64):
                row_end = min(row_start + 64, used * heads)
                scores = (query[row_start:row_end] @ key.T) * scale
                if config["mask_mode"] == 3:
                    visible = (
                        torch.arange(kv_length)[None, :]
                        <= (
                            torch.arange(row_start, row_end) // heads + kv_length - used
                        )[:, None]
                    )
                    scores.masked_fill_(~visible, -float("inf"))
                valid = torch.isfinite(scores).any(dim=1)
                if valid.any():
                    valid_scores = scores[valid]
                    output_batch[row_start:row_end][valid] = (
                        torch.softmax(valid_scores, dim=-1) @ key[:, :512]
                    ).float()
                    lse_batch[row_start:row_end][valid] = torch.logsumexp(
                        valid_scores, dim=-1
                    ).float()
        kv_start += kv_length

    if config["layout_q"] == "TND":
        if config["layout_out"] == "NTD":
            output = output.permute(1, 0, 2).contiguous()
        return output, lse.T.contiguous()

    output = output.reshape(config["B"], spans[0], heads, 512)
    lse = lse.reshape(config["B"], spans[0], heads).permute(0, 2, 1)
    if config["layout_q"] == "BNSD":
        output = output.permute(0, 2, 1, 3).contiguous()
    return output, lse.contiguous()


def _prepare_launch(case, q_npu, k_npu, block_table, cumulative):
    """Build static device inputs once and return the main-operator call."""
    config = case["config"]
    cache_seqlens = torch.tensor(
        config["cache_seqlens"], dtype=torch.int32, device="npu"
    )
    cu_seqlens_q = (
        torch.tensor(cumulative, dtype=torch.int32, device="npu")
        if config["layout_q"] == "TND"
        else None
    )
    seqused_q = (
        torch.tensor(config["seqused_q"], dtype=torch.int32, device="npu")
        if "seqused_q" in config
        else None
    )
    metadata_kwargs = {
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_q": seqused_q,
        "max_seqlen_q": max(case["spans"]),
        "max_seqlen_kv": max(config["cache_seqlens"]),
        "mask_mode": config["mask_mode"],
        "layout_q": config["layout_q"],
    }
    metadata = metadata_module.flash_mla_with_kvcache_metadata(
        cache_seqlens,
        config["N1"],
        1,
        **metadata_kwargs,
    )
    if case.get("expect_fd"):
        assert int(metadata[1].cpu().item()) == 1, (
            "case expected FlashDecode metadata, but HEAD_IS_FD is zero"
        )
    attn_mask = (
        torch.triu(torch.ones((2048, 2048), dtype=torch.int8), diagonal=1).npu()
        if config["mask_mode"] == 3
        else None
    )

    def launch():
        return flash_mla_module.flash_mla_with_kvcache(
            q_npu,
            k_npu,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            attn_mask=attn_mask,
            head_dim_v=512,
            softmax_scale=config.get("scale", 1.0 / math.sqrt(576.0)),
            layout_kv=config["layout_kv"],
            layout_out=config["layout_out"],
            return_softmax_lse=config["return_softmax_lse"],
            metadata=metadata,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            mask_mode=config["mask_mode"],
            max_seqlen_q=-1,
            max_seqlen_kv=-1,
            layout_q=config["layout_q"],
        )

    return launch


def _launch_case(case, q_npu, k_npu, block_table, cumulative):
    return _prepare_launch(case, q_npu, k_npu, block_table, cumulative)()


def _execute_launches(launch, repeats, aclgraph):
    """Run eager calls or replay one captured ACLGraph with stable tensors."""
    if aclgraph:
        # Compile and warm up outside capture, then keep captured inputs and
        # outputs alive for every replay.
        launch()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = launch()
        torch.npu.synchronize()
        outputs = []
        for _ in range(max(repeats, 2)):
            graph.replay()
            torch.npu.synchronize()
            outputs.append(tuple(value.cpu() for value in captured))
        return outputs

    outputs = []
    for _ in range(repeats):
        actual = launch()
        torch.npu.synchronize()
        outputs.append(tuple(value.cpu() for value in actual))
    return outputs


def _run_case(case, *, aclgraph=False):
    torch.npu.set_device(0)
    cumulative, q_cpu, logical_kv, q_npu, k_npu, block_table = _generate(case)
    launch = _prepare_launch(case, q_npu, k_npu, block_table, cumulative)
    first_cpu, actual = _execute_launches(launch, repeats=2, aclgraph=aclgraph)
    stable = all(torch.equal(before, after) for before, after in zip(first_cpu, actual))

    golden_output, golden_lse = _golden(case, cumulative, q_cpu, logical_kv)
    atol, rtol = _torch_tolerances(q_npu.dtype)
    output_stats = _compare_stats(actual[0], golden_output, atol, rtol)
    output_pass = _precision_pass(output_stats)
    if case["config"]["return_softmax_lse"]:
        finite = torch.isfinite(golden_lse)
        sentinel_match = torch.equal(
            torch.isposinf(actual[1]), torch.isposinf(golden_lse)
        ) and torch.equal(torch.isneginf(actual[1]), torch.isneginf(golden_lse))
        lse_error = (
            float(
                (actual[1][finite].double() - golden_lse[finite].double()).abs().max()
            )
            if finite.any()
            else 0.0
        )
        lse_pass = sentinel_match and lse_error <= LSE_ATOL
    else:
        lse_pass = True
    return {
        "repeat_stable": stable,
        "output_pass": output_pass,
        "lse_pass": lse_pass,
        "pass": stable and output_pass and lse_pass,
        "output_stats": output_stats,
    }


@pytest.fixture
def aclgraph_enabled(request):
    """Select eager or ACLGraph execution with ``-o mode=<mode>``."""
    mode = "eager"
    for option in request.config.getoption("override_ini") or ():
        name, separator, value = option.partition("=")
        if separator and name == "mode":
            mode = value
    if mode not in ("eager", "aclgraph"):
        pytest.fail("mode must be eager or aclgraph", pytrace=False)
    return mode == "aclgraph"


FUNCTION_CASES = [
    # TND + BF16 + MTP + N96 + ragged multi-batch + KV tails.
    _case(
        "tnd_mtp_ragged_bf16_n96",
        [2, 2, 2],
        [512, 300, 193],
        heads=96,
        dtype="bf16",
    ),
    # Preserve empty-KV, ragged-Q and no-LSE coverage on a supported layout.
    _case(
        "tnd_ntd_pa_nz_empty_no_lse",
        [1, 2, 1],
        [65, 0, 33],
        layout_kv="PA_NZ",
        return_lse=False,
    ),
    # Preserve seqused sentinel and page-tail coverage with mainline N64/TND.
    _case(
        "tnd_bbnd_seqused_n64",
        [2, 2],
        [130, 193],
        heads=64,
        layout_kv="PA_BBND",
        seqused=[0, 2],
    ),
    # A 128-token page crosses the 112-token compute tile boundary.
    _case(
        "tnd_bbnd_page128_tail",
        [1],
        [193],
        layout_kv="PA_BBND",
    ),
    _case(
        "tnd_bbnd_page128_bf16_tail_n96",
        [2],
        [257],
        heads=96,
        dtype="bf16",
        layout_kv="PA_BBND",
    ),
    # RIGHT_DOWN causal + long-KV split/combine threshold.
    _case(
        "tnd_causal_fd_tail",
        [2],
        [2049],
        mask_mode=3,
        expect_fd=True,
    ),
    # All batches have no KV work: metadata empty-grid and output init path.
    _case(
        "tnd_all_empty_kv",
        [1, 2],
        [0, 0],
    ),
]


@pytest.mark.npu
@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case["id"]) for case in FUNCTION_CASES],
)
def test_flash_mla_with_kvcache(case, aclgraph_enabled):
    pytest.importorskip("torch_npu")
    result = _run_case(case, aclgraph=aclgraph_enabled)
    assert result["repeat_stable"], result
    assert result["output_pass"], result
    assert result["lse_pass"], result
    assert result["pass"], result
