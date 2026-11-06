# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ruff: noqa: E402

"""Mega attention tests, CPU references, device probes and pressure profiling.

Run pytest on this file, or use the test / benchmark / benchmark-paged /
analyze-profile commands. Device-only dependencies are loaded on demand.
"""

import argparse
import ast
from contextlib import nullcontext
import csv
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import threading
from types import SimpleNamespace

SAMPLE_DIR = Path(__file__).resolve().parents[3] / "samples/attention/mega_mla"
# Select an alternate implementation before any runtime module is imported.
if __name__ == "__main__" and sys.argv[1:2] in (["benchmark"], ["benchmark-paged"]):
    _source_parser = argparse.ArgumentParser(add_help=False)
    _source_parser.add_argument("--source", type=Path, default=SAMPLE_DIR)
    _source_args, _ = _source_parser.parse_known_args(sys.argv[2:])
    SAMPLE_DIR = _source_args.source.resolve()
ROOT = SAMPLE_DIR
if str(SAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(SAMPLE_DIR))


# ============================================================================
# Profiling summary (standard-library-only command)
# ============================================================================


def analyze_profile(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--last", type=int, default=50)
    args = parser.parse_args(argv)
    for root in args.roots:
        files = sorted(Path(root).glob("**/op_summary_*.csv"))
        if not files:
            print(json.dumps(dict(root=root, error="no_op_summary")))
            continue
        rows = []
        for file in files:
            with file.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    if row.get("Task Type") == "MIX_AIC" and any(
                        prefix in row.get("Op Name", "")
                        for prefix in (
                            "mega_mla",
                            "MegaMLA",  # Historical profiling data.
                            "deepseek_v4_1_attention_mega",
                        )
                    ):
                        row["_file"] = str(file)
                        rows.append(row)
        if not rows:
            print(json.dumps(dict(root=root, error="no_mega_tasks")))
            continue
        measured = rows[-args.last :]
        durations = [float(row["Task Duration(us)"]) for row in measured]
        fastest = min(measured, key=lambda row: float(row["Task Duration(us)"]))
        sampled = max(measured, key=lambda row: float(row.get("aic_mac_ratio") or 0))
        metrics = (
            "aic_mac_ratio",
            "aic_mte1_ratio",
            "aic_mte2_ratio",
            "aic_fixpipe_ratio",
            "aic_scalar_ratio",
            "aiv_vec_ratio",
            "aiv_mte2_ratio",
            "aiv_mte3_ratio",
            "aiv_scalar_ratio",
        )
        print(
            json.dumps(
                dict(
                    root=root,
                    total_tasks=len(rows),
                    n=len(measured),
                    min_us=min(durations),
                    median_us=statistics.median(durations),
                    mean_us=statistics.mean(durations),
                    max_us=max(durations),
                    p95_us=sorted(durations)[int(0.95 * (len(durations) - 1))],
                    fastest={key: float(fastest.get(key) or 0) for key in metrics},
                    sampled={key: float(sampled.get(key) or 0) for key in metrics},
                    csv=fastest["_file"],
                )
            )
        )


if __name__ == "__main__" and sys.argv[1:2] == ["analyze-profile"]:
    analyze_profile(sys.argv[2:])
    raise SystemExit(0)

import pytest
import torch
import attention_inputs as public
from attention_geometry import (
    COMPRESSED_DECODE_STRATEGIES,
    HEAD_DIM,
    choose_compressed_cache_decode_strategy,
    select_small_k_attention_head_tile,
    validate_aic_grid,
    validate_query_heads,
)
from attention_sequence_inputs import (
    AttentionSequenceMetadata,
    validate_attention_sequences,
)
from attention_workspace import AttentionWorkspaceLayout


# ============================================================================
# Independent CPU cache, projection and attention references
# ============================================================================

D = 512
FP8_DATA, FP8_SCALE, FP8_RECORD = 512, 32, 544
FP4_DATA, FP4_SCALE, FP4_RECORD = 256, 64, 320
_FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _e8m0_scale(
    x: torch.Tensor, qmax: float, group: int
) -> tuple[torch.Tensor, torch.Tensor]:
    amax = x.float().abs().reshape(*x.shape[:-1], D // group, group).amax(-1)
    exponent = torch.ceil(torch.log2(torch.clamp_min(amax / qmax, 1e-4)))
    exponent = exponent.clamp(-127, 127).to(torch.int32)
    scale = torch.exp2(exponent.float())
    return scale, (exponent + 127).to(torch.uint8)


def _pack_page(data: torch.Tensor, scale: torch.Tensor, record: int) -> torch.Tensor:
    blocks, page = data.shape[:2]
    return torch.cat((data, scale), -1).reshape(blocks, page, 1, record)


def quantize_fp8(x: torch.Tensor) -> torch.Tensor:
    """BF16/FP32 [..., 512] -> uint8 [..., 544], page-major planes."""
    if x.ndim != 4 or x.shape[-2:] != (1, D):
        raise ValueError("expected [blocks,page,1,512]")
    y = x.squeeze(-2).float()
    s, _ = _e8m0_scale(y, 448.0, 32)
    q = (y.reshape(*y.shape[:-1], 16, 32) / s[..., None]).clamp(-448, 448)
    data = q.reshape(*y.shape).to(torch.float8_e4m3fn).view(torch.uint8)
    return _pack_page(data, s.to(torch.bfloat16).view(torch.uint8), FP8_RECORD)


def quantize_fp4(x: torch.Tensor) -> torch.Tensor:
    """BF16/FP32 [..., 512] -> packed E2M1 and BF16 scale."""
    if x.ndim != 4 or x.shape[-2:] != (1, D):
        raise ValueError("expected [blocks,page,1,512]")
    y = x.squeeze(-2).float()
    s, _ = _e8m0_scale(y, 6.0, 16)
    v = (y.reshape(*y.shape[:-1], 32, 16) / s[..., None]).reshape(*y.shape)
    mag = _FP4_VALUES.to(v.device)
    dist = (v.abs().unsqueeze(-1) - mag).abs()
    min_dist = dist.amin(-1, keepdim=True)
    even = torch.arange(8, device=v.device) % 2 == 0
    code = (
        torch.where(dist == min_dist, torch.where(even, 0, 1), 2)
        .argmin(-1)
        .to(torch.uint8)
    )
    code |= torch.signbit(v).to(torch.uint8) << 3
    data = code[..., 0::2] | (code[..., 1::2] << 4)
    return _pack_page(data, s.to(torch.bfloat16).view(torch.uint8), FP4_RECORD)


def _planes(
    cache: torch.Tensor, data_bytes: int, scale_bytes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks, page, heads, record = cache.shape
    if heads != 1 or record != data_bytes + scale_bytes or cache.dtype != torch.uint8:
        raise ValueError("cache shape/dtype mismatch")
    rows = cache.reshape(blocks, page, record)
    return rows[..., :data_bytes], rows[..., data_bytes:]


def dequantize_fp8(cache: torch.Tensor) -> torch.Tensor:
    data, scales = _planes(cache, FP8_DATA, FP8_SCALE)
    v = data.view(torch.float8_e4m3fn).float().reshape(*data.shape[:2], 16, 32)
    s = scales.contiguous().view(torch.bfloat16).float()
    return (v * s[..., None]).reshape(*data.shape[:2], 1, D).to(torch.bfloat16)


def dequantize_fp4(cache: torch.Tensor) -> torch.Tensor:
    data, scales = _planes(cache, FP4_DATA, FP4_SCALE)
    codes = torch.empty((*data.shape[:2], D), dtype=torch.uint8, device=data.device)
    codes[..., 0::2] = data & 15
    codes[..., 1::2] = data >> 4
    mag = _FP4_VALUES.to(data.device)[(codes & 7).long()]
    v = torch.where((codes & 8) != 0, -mag, mag)
    s = scales.contiguous().view(torch.bfloat16).float()
    return (
        (v.reshape(*data.shape[:2], 32, 16) * s[..., None])
        .reshape(*data.shape[:2], 1, D)
        .to(torch.bfloat16)
    )


def physical_index(block: int, offset: int, page: int) -> int:
    if block < 0 or not 0 <= offset < page:
        raise ValueError("invalid block/offset")
    return block * page + offset


def _native_cache_tensor(value, padding, offset=0, device=None):
    """Build the caller's native allocation, with sentinel page gaps and prefix."""
    pages, rows, heads, record = value.shape
    assert heads == 1 and padding >= 0 and offset >= 0
    pitch = rows * record + padding
    backing = torch.full(
        (offset + pages * pitch + 64,),
        193,
        dtype=torch.uint8,
        device=value.device if device is None else device,
    )
    view = backing.as_strided(
        (pages, rows, 1, record), (pitch, record, record, 1), offset
    )
    view.copy_(value.to(view.device))
    return view, backing


def _assert_cache_padding(view, backing):
    actual = backing.cpu()
    padding = torch.ones(actual.numel(), dtype=torch.bool)
    for page in range(view.shape[0]):
        begin = view.storage_offset() + page * view.stride(0)
        padding[begin : begin + view.shape[1] * view.shape[3]] = False
    assert torch.all(actual[padding] == 193), "cache padding or prefix was overwritten"


def valid_cmp_indices(indices, lengths=None):
    valid = indices >= 0
    if lengths is not None:
        expanded = lengths.repeat_interleave(indices.shape[0] // lengths.numel())
        valid &= torch.arange(indices.shape[1])[None, :] < expanded[:, None]
    return valid


def decode_selected_cmp(cache, indices, lengths=None):
    valid = valid_cmp_indices(indices, lengths)
    result = torch.zeros((*indices.shape, 512), dtype=torch.bfloat16)
    if not valid.any():
        return result
    records = cache.reshape(-1, 320)[indices[valid].long()]
    packed = records[:, :256].to(torch.int32)
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(-1, 512)
    magnitude = codes & 7
    # E2M1 finite magnitudes, derived from exponent/mantissa instead of a LUT.
    absolute = torch.where(
        magnitude < 2,
        magnitude.float() * 0.5,
        (1.0 + (magnitude & 1).float() * 0.5)
        * torch.exp2((magnitude >> 1).float() - 1.0),
    )
    values = torch.where((codes & 8) != 0, -absolute, absolute)
    scales = records[:, 256:].contiguous().view(torch.bfloat16).float()
    result[valid] = (
        (values.reshape(-1, 32, 16) * scales[:, :, None])
        .reshape(-1, 512)
        .to(torch.bfloat16)
    )
    return result


def _mx_round(x):
    """Independent CPU projection quantization: exponent bits, groups of 32."""
    import torch

    grouped = x.float().reshape(*x.shape[:-1], -1, 32)
    amax = grouped.abs().amax(-1)
    # For finite positive FP32, these are the biased exponent bits; zero=0.
    biased = torch.where(amax > 0, torch.floor(torch.log2(amax)) + 127, 0)
    exponent = (biased - 8).clamp_min(0) - 127
    scale = torch.exp2(exponent)
    q = (grouped / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (q.float() * scale[..., None]).reshape_as(x)


def _rope_cpu(x, cos, sin):
    import torch

    y = x.float().clone()
    even, odd = y[..., -64::2].clone(), y[..., -63::2].clone()
    y[..., -64::2] = even * cos - odd * sin
    y[..., -63::2] = even * sin + odd * cos
    return y.to(torch.bfloat16)


def _rms_reference(accumulator, gamma):
    """Round the completed FP32 projection to BF16 before FP32 RMS math."""
    import torch

    rounded = accumulator.to(torch.bfloat16).float()
    return (
        rounded * torch.rsqrt(rounded.square().mean(-1, keepdim=True) + 1e-20) * gamma
    )


def _attention_reference(q, gathered, invalid, sink):
    import torch

    score = (q.float() @ gathered.float().transpose(1, 2)).masked_fill(
        invalid[:, None, :], -1.7e38
    )
    maximum = torch.maximum(score.amax(-1, keepdim=True), sink[:, :, None])
    exp = torch.exp(score - maximum)
    prob = (
        exp / (exp.sum(-1, keepdim=True) + torch.exp(sink[:, :, None] - maximum))
    ).to(torch.bfloat16)
    return (prob.float() @ gathered.float()).to(torch.bfloat16)


# ============================================================================
# Independent CPU sequence and page-table references
# ============================================================================


def filter_sequences(
    indices_in_kvcache,
    extra_indices_in_kvcache,
    slot,
    page_size,
    win_pages,
    cmp_pages,
    *,
    cu_seqlens_qo=None,
    seqused_q=None,
    cache_seqlens=None,
    extra_cache_seqlens=None,
    block_table=None,
    extra_block_table=None,
    topk_length=None,
    extra_topk_length=None,
    compressed_page_size=None,
):
    t = slot.numel()
    if cu_seqlens_qo is not None:
        spans = cu_seqlens_qo.tolist()
        batches = len(spans) - 1
    else:
        first = next(
            v
            for v in (seqused_q, cache_seqlens, extra_cache_seqlens, block_table)
            if v is not None
        )
        batches = first.shape[0]
        spans = [b * (t // batches) for b in range(batches + 1)]
    outputs = [
        torch.full_like(indices_in_kvcache, -1),
        torch.full_like(extra_indices_in_kvcache, -1),
    ]
    safe_slot = torch.full_like(slot, -1)
    routes = (
        (
            indices_in_kvcache,
            block_table,
            cache_seqlens,
            topk_length,
            win_pages,
            page_size,
        ),
        (
            extra_indices_in_kvcache,
            extra_block_table,
            extra_cache_seqlens,
            extra_topk_length,
            cmp_pages,
            page_size if compressed_page_size is None else compressed_page_size,
        ),
    )
    for b, (start, stop) in enumerate(zip(spans, spans[1:])):
        active = stop - start if seqused_q is None else int(seqused_q[b])
        active = max(0, min(active, stop - start))
        for row in range(start, start + active):
            s = int(slot[row])
            if page_size <= s < win_pages * page_size:
                safe_slot[row] = s
            for out, (idx, table, used, topk, pages, route_page_size) in zip(
                outputs, routes
            ):
                n = (
                    idx.shape[1]
                    if topk is None
                    else int(topk[row if topk.numel() == t else b])
                )
                n = max(0, min(n, idx.shape[1]))
                capacity = (
                    pages if table is None else table.shape[1]
                ) * route_page_size
                limit = (
                    capacity if used is None else max(0, min(int(used[b]), capacity))
                )
                logical = idx[row, :n].long()
                valid = (logical >= 0) & (logical < limit)
                cols = torch.arange(n)[valid]
                logical = logical[valid]
                if table is None:
                    out[row, cols] = logical.to(torch.int32)
                else:
                    physical_pages = table[b, logical // route_page_size].long()
                    valid_pages = (physical_pages >= 0) & (physical_pages < pages)
                    out[row, cols[valid_pages]] = (
                        physical_pages[valid_pages] * route_page_size
                        + logical[valid_pages] % route_page_size
                    ).to(torch.int32)
    return *outputs, safe_slot


def query_metadata_reference(tokens, cu_seqlens_qo, seqused_q):
    spans = cu_seqlens_qo.tolist()
    result = torch.empty(tokens, 8, dtype=torch.int32)
    for batch, (start, end) in enumerate(zip(spans, spans[1:])):
        used = (
            end - start
            if seqused_q is None
            else max(0, min(int(seqused_q[batch]), end - start))
        )
        for row in range(start, end):
            result[row] = torch.tensor(
                [
                    batch,
                    start,
                    end - start,
                    used,
                    row - start,
                    int(row - start < used),
                    0,
                    0,
                ]
            )
    return result


# ============================================================================
# Shared CPU fixture
# ============================================================================


@pytest.fixture(autouse=True)
def _restore_torch_threads():
    previous = torch.get_num_threads()
    try:
        yield
    finally:
        if torch.get_num_threads() != previous:
            torch.set_num_threads(previous)


@pytest.fixture
def business(monkeypatch):
    # The CPU test cannot query NPU private formats. Everything else, including
    # tensor shape/dtype/views/allocation/alias checks, uses actual PyTorch.
    monkeypatch.setattr(public, "_validate_nz_weight", lambda *_: None)
    op = SimpleNamespace(
        mode="attn",
        rope_dim=64,
        grid=28,
        page_size=128,
        win_len=128,
        tokens=None,
        expected_k=None,
        heads=16,
    )
    t = 8
    args = dict(
        hidden_states=torch.empty(t, 5120, dtype=torch.bfloat16),
        query_norm_weight=torch.empty(1, 1280),
        kv_norm_weight=torch.empty(1, 512),
        query_down_proj_weight=torch.empty(1280, 5120, dtype=torch.float8_e4m3fn),
        kv_proj_weight=torch.empty(512, 5120, dtype=torch.float8_e4m3fn),
        query_up_proj_weight=torch.empty(8192, 1280, dtype=torch.float8_e4m3fn),
        query_down_proj_scale=torch.empty(1280, 80, 2, dtype=torch.float8_e8m0fnu),
        kv_proj_scale=torch.empty(512, 80, 2, dtype=torch.float8_e8m0fnu),
        query_up_proj_scale=torch.empty(8192, 20, 2, dtype=torch.float8_e8m0fnu),
        k_cache=torch.zeros(2, 128, 1, 544, dtype=torch.uint8),
        extra_k_cache=torch.zeros(5, 128, 1, 320, dtype=torch.uint8),
        indices_in_kvcache=torch.zeros(t, 128, dtype=torch.int32),
        extra_indices_in_kvcache=torch.zeros(t, 512, dtype=torch.int32),
        attn_sink=torch.empty(1, 16),
        rope_cos=torch.empty(t, 32),
        rope_sin=torch.empty(t, 32),
        window_cache_slots=torch.empty(t, dtype=torch.int32),
    )
    return op, args


# ============================================================================
# Memory pressure before each benchmark target
# ============================================================================


class BenchmarkCachePressure:
    """Evict target data through unrelated reads/writes; not a hardware flush."""

    def __init__(self, size_mib=256):
        if size_mib <= 0:
            raise ValueError("cache pressure size must be positive")
        self.size_mib = size_mib
        self.buffer = torch.zeros(
            size_mib * 1024 * 1024 // 4, dtype=torch.float32, device="npu"
        )
        torch.npu.synchronize()

    def apply(self):
        self.buffer.add_(1)
        # Complete pressure before the target so target task time excludes it.
        torch.npu.synchronize()


# ============================================================================
# Lazy device probes
# ============================================================================


@lru_cache(maxsize=1)
def _device_probes():
    """Load DSL probe definitions only when an AOT/NPU case needs them."""
    global \
        MemLoc, \
        Tensor, \
        dtypes, \
        tile_slice, \
        vec_sync_all, \
        Buffer, \
        host, \
        kernel, \
        get_block_idx, \
        get_block_num, \
        mem_copy, \
        vec_store_bypass, \
        MegaMlaKernel, \
        Fp4CacheDecoder, \
        CompressedCacheGather
    from cannbotdsl import MemLoc, Tensor, dtypes, tile_slice, vec_sync_all

    from cannbotdsl import make_layout
    from cannbotdsl.buffer import Buffer

    from cannbotdsl.lang import host, kernel

    from cannbotdsl.ops.arch import get_block_idx, get_block_num

    from cannbotdsl.ops.memcpy import mem_copy

    from cannbotdsl.ops.scalar import vec_store_bypass

    from mega_mla import MegaMlaKernel

    from cache_decode import Fp4CacheDecoder

    from compressed_cache_gather import CompressedCacheGather

    class CompressedCacheGatherProbe:
        def __init__(self, tokens, topk, length_stride, cache_batch_stride=0, grid=1):
            self.gather = CompressedCacheGather(
                tokens, topk, 128, 128, length_stride, cache_batch_stride
            )
            self.grid = grid

        @kernel
        def probe_compressed_cache_gather(
            self,
            cache: Tensor,
            index: Tensor,
            lengths: Tensor,
            zero: Tensor,
            output: Tensor,
        ):
            self.gather.run(
                cache,
                index,
                lengths,
                zero,
                output,
                get_block_idx(),
                get_block_num(),
                index,
            )

        @host
        def run(self, cache, index, lengths, zero, output):
            self.probe_compressed_cache_gather[1](cache, index, lengths, zero, output)

        @kernel
        def probe_paged_compressed_cache_gather(
            self,
            cache: Tensor,
            index: Tensor,
            lengths: Tensor,
            zero: Tensor,
            output: Tensor,
            block_table: Tensor,
        ):
            self.gather.run(
                cache,
                index,
                lengths,
                zero,
                output,
                get_block_idx(),
                get_block_num(),
                block_table,
            )

        @host
        def run_paged(self, cache, index, lengths, zero, output, block_table):
            self.probe_paged_compressed_cache_gather[self.grid](
                cache, index, lengths, zero, output, block_table
            )

    class WindowCacheScaleProbe:
        def __init__(self):
            self.core = MegaMlaKernel(640, 128, 128, tokens=8, grid=32)

        @kernel
        def probe_window_cache_scales(self, x: Tensor, data: Tensor, scale: Tensor):
            x_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, 64)))
            data_ub = Buffer(MemLoc.UB, dtypes.int8, make_layout((1, 64)))
            scale_ub = Buffer(MemLoc.UB, dtypes.int16, make_layout((1, 16)))
            for row in range(get_block_idx(), x.shape[0], get_block_num()):
                mem_copy(x_ub, tile_slice(x, (1, 64), (row, 0)))
                vec_sync_all()
                self.core._quantize_window_cache_chunk(x_ub, data_ub, scale_ub)
                vec_sync_all()
                mem_copy(tile_slice(data, (1, 64), (row, 0)), data_ub)
                mem_copy(
                    tile_slice(scale, (1, 2), (row, 0)),
                    tile_slice(scale_ub, (1, 2), (0, 0)),
                )
                vec_sync_all()

        @host
        def run(self, x, data, scale):
            self.probe_window_cache_scales[64](x, data, scale)

    class WindowCacheRecordProbe:
        def __init__(self):
            self.core = MegaMlaKernel(640, 128, 128, tokens=8, grid=32)

        @kernel
        def probe_window_cache_records(
            self, x: Tensor, cache_bytes: Tensor, cache_words: Tensor
        ):
            x_ub = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, 64)))
            data_ub = Buffer(MemLoc.UB, dtypes.int8, make_layout((1, 64)))
            scale_ub = Buffer(MemLoc.UB, dtypes.int16, make_layout((1, 16)))
            data_words = data_ub.reinterpret(dtypes.uint64, make_layout((1, 8)))
            scale_bytes = scale_ub.reinterpret(dtypes.int8, make_layout((1, 32)))
            for item in range(0, x.shape[0] * 8):
                row = item // 8
                chunk = item % 8
                mem_copy(x_ub, tile_slice(x, (1, 64), (row, chunk)))
                vec_sync_all()
                self.core._quantize_window_cache_chunk(x_ub, data_ub, scale_ub)
                vec_sync_all()
                for word in range(8):
                    vec_store_bypass(
                        cache_words.ptr((0, row * 68 + chunk * 8 + word)),
                        data_words[0, word],
                    )
                for byte in range(4):
                    vec_store_bypass(
                        cache_bytes.ptr((0, row * 544 + 512 + chunk * 4 + byte)),
                        scale_bytes[0, byte],
                    )
                vec_sync_all()

        @host
        def run(self, x, cache_bytes, cache_words):
            self.probe_window_cache_records[1](x, cache_bytes, cache_words)

    class CompressedCacheDecodeProbe:
        def __init__(self, unroll):
            self.decoder = Fp4CacheDecoder(unroll)

        @kernel
        def probe_compressed_cache_decode(self, src: Tensor, dst: Tensor):
            raw = Buffer(MemLoc.UB, dtypes.uint8, make_layout((1, 320)))
            out = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, 512)))
            for row in range(get_block_idx(), src.shape[0], get_block_num()):
                mem_copy(raw, tile_slice(src, (1, 320), (row, 0)))
                vec_sync_all()
                self.decoder.decode(raw, out, 0)
                vec_sync_all()
                mem_copy(tile_slice(dst, (1, 512), (row, 0)), out)
                vec_sync_all()

        @host
        def run(self, src, dst):
            self.probe_compressed_cache_decode[1](src, dst)

    class AttentionSequenceRouteProbe:
        """Exercise both route widths without projection/attention workspaces."""

        def __init__(self, config, cmp_len, grid, win_pages=4, cmp_pages=84):
            from attention_sequence_filter import AttentionSequenceFilter

            self.filter = AttentionSequenceFilter(config, 128, 128, cmp_len)
            self.grid, self.win_pages, self.cmp_pages = grid, win_pages, cmp_pages

        @kernel
        def probe_attention_sequence_routes(
            self,
            wi: Tensor,
            ci: Tensor,
            wl: Tensor,
            cl: Tensor,
            wt: Tensor,
            ct: Tensor,
            cq: Tensor,
            uq: Tensor,
            uw: Tensor,
            uc: Tensor,
            slots: Tensor,
            wo: Tensor,
            co: Tensor,
            so: Tensor,
            meta: Tensor,
        ):
            self.filter.run(
                wi,
                ci,
                wl,
                cl,
                wt,
                ct,
                cq,
                uq,
                uw,
                uc,
                slots,
                wo,
                co,
                so,
                get_block_idx(),
                get_block_num(),
                self.win_pages,
                self.cmp_pages,
                meta,
            )

        @host
        def run(self, wi, ci, wl, cl, wt, ct, cq, uq, uw, uc, slots, wo, co, so, meta):
            self.probe_attention_sequence_routes[self.grid](
                wi, ci, wl, cl, wt, ct, cq, uq, uw, uc, slots, wo, co, so, meta
            )

    return SimpleNamespace(
        CompressedCacheGatherProbe=CompressedCacheGatherProbe,
        WindowCacheScaleProbe=WindowCacheScaleProbe,
        WindowCacheRecordProbe=WindowCacheRecordProbe,
        CompressedCacheDecodeProbe=CompressedCacheDecodeProbe,
        AttentionSequenceRouteProbe=AttentionSequenceRouteProbe,
    )


# ============================================================================
# Public inputs, outputs and workspace contracts
# ============================================================================


def test_private_mapping_and_storage_aliases(business):
    op, values = business
    args, out = public.prepare_attention_arguments(op, **values)
    assert len(args) == 27
    for slot, name in enumerate(
        (
            "hidden_states",
            "query_norm_weight",
            "kv_norm_weight",
            "query_down_proj_weight",
            "kv_proj_weight",
            "query_up_proj_weight",
            "query_down_proj_scale",
            "kv_proj_scale",
            "query_up_proj_scale",
            "k_cache",
            "extra_k_cache",
            "indices_in_kvcache",
            "extra_indices_in_kvcache",
            "attn_sink",
            "rope_cos",
            "rope_sin",
            "window_cache_slots",
        )
    ):
        assert args[slot] is values[name]
    assert args[21] is out and out.shape == (8, 16, 512)
    assert args[22] is op._workspace_storage and args[22].dtype == torch.float32
    root = args[22].untyped_storage().data_ptr()
    for alias, dtype in zip(
        args[22:27],
        (
            torch.float32,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e8m0fnu,
            torch.int32,
        ),
    ):
        assert alias.dtype == dtype and alias.data_ptr() == root
        assert alias.numel() * alias.element_size() == op._workspace_layout.nbytes
    for name, shape, dtype, offset, size in op._workspace_layout.regions:
        view = op._workspace[name]
        assert view.untyped_storage().data_ptr() == root
        assert view.data_ptr() == root + offset
        assert tuple(view.shape) == shape and view.numel() * view.element_size() == size
        assert (
            offset % 512 == 0
            and offset + size <= args[22].numel() * args[22].element_size()
        )
    assert op._workspace["qr"].dtype == op._workspace["kv"].dtype == torch.float32
    assert op._workspace["zero"].shape == (1, 512)
    assert op._workspace["zero"].dtype == torch.bfloat16
    assert out.untyped_storage().data_ptr() != root


def test_reuse_and_independent_outputs(business):
    op, values = business
    first, out0 = public.prepare_attention_arguments(op, **values)
    ptrs = {name: v.data_ptr() for name, v in op._workspace.items()}
    second, out1 = public.prepare_attention_arguments(op, **values)
    assert ptrs == {name: v.data_ptr() for name, v in op._workspace.items()}
    assert out0.data_ptr() != out1.data_ptr()
    assert first[22] is second[22]
    _, result = public.prepare_attention_arguments(op, **values, out=out0)
    assert result is out0


def test_geometry_changes_reallocate(business):
    op, values = business
    public.prepare_attention_arguments(op, **values)
    old_work = op._workspace
    values["extra_indices_in_kvcache"] = torch.zeros(8, 640, dtype=torch.int32)
    args, _ = public.prepare_attention_arguments(op, **values)
    assert op._workspace is not old_work and op._workspace["kg"].shape == (8, 768, 512)
    old_work = op._workspace
    values["extra_k_cache"] = torch.zeros(6, 128, 1, 320, dtype=torch.uint8)
    args, _ = public.prepare_attention_arguments(op, **values)
    assert op._workspace is not old_work and op._workspace["cmp_dec"].shape == (
        6,
        128,
        512,
    )


@pytest.mark.parametrize(
    "name,shape,dtype",
    [
        ("rope_cos", (8, 64), torch.float32),
        ("rope_sin", (8, 64), torch.float32),
        ("rope_cos", (7, 32), torch.float32),
        ("rope_sin", (8, 32), torch.bfloat16),
        ("k_cache", (2, 128, 1, 512), torch.uint8),
        ("extra_k_cache", (5, 128, 1, 320), torch.int8),
        ("indices_in_kvcache", (8, 64), torch.int32),
        ("extra_indices_in_kvcache", (8, 511), torch.int32),
        ("attn_sink", (1, 64), torch.float32),
        ("window_cache_slots", (7,), torch.int32),
        ("query_down_proj_weight", (1280, 5120), torch.int8),
        ("kv_proj_weight", (512, 5120), torch.int8),
        ("query_up_proj_weight", (8192, 1280), torch.int8),
    ],
)
def test_bad_inputs_rejected_before_workspace(business, name, shape, dtype):
    op, values = business
    values[name] = torch.empty(shape, dtype=dtype)
    with pytest.raises(ValueError):
        public.prepare_attention_arguments(op, **values)
    assert not hasattr(op, "_workspace")


@pytest.mark.parametrize("compressed_page_size", [64, 128])
@pytest.mark.parametrize("strategy", ["predecode", "indexed"])
@pytest.mark.parametrize(
    "window_padding,compressed_padding,offset", [(0, 0, 0), (64, 96, 64), (32, 17, 128)]
)
def test_native_cache_layout_preserved(
    business, compressed_page_size, strategy, window_padding, compressed_padding, offset
):
    op, values = business
    op.cmp_decode_strategy = strategy
    win, win_backing = _native_cache_tensor(values["k_cache"], window_padding, offset)
    cmp, cmp_backing = _native_cache_tensor(
        torch.zeros(5, compressed_page_size, 1, 320, dtype=torch.uint8),
        compressed_padding,
        offset,
    )
    values["k_cache"], values["extra_k_cache"] = win, cmp
    args, _ = public.prepare_attention_arguments(op, **values)
    assert args[9] is win and args[10] is cmp
    expected = (1, 1, 512) if strategy == "indexed" else (5, compressed_page_size, 512)
    assert op._workspace["cmp_dec"].shape == expected
    _assert_cache_padding(win, win_backing)
    _assert_cache_padding(cmp, cmp_backing)


@pytest.mark.parametrize("field", ["k_cache", "extra_k_cache"])
@pytest.mark.parametrize("kind", ["overlap", "row_gap", "record_gap", "zero_rows"])
def test_native_cache_rejects_unsupported_layout_before_allocation(
    business, field, kind
):
    op, values = business
    value = values[field]
    pages, rows, _, record = value.shape
    if kind == "zero_rows":
        bad = torch.empty(pages, 0, 1, record, dtype=torch.uint8)
    else:
        stride = (
            (rows * record - record, record, record, 1)
            if kind == "overlap"
            else (
                (rows * (record + 32), record + 32, record, 1)
                if kind == "row_gap"
                else (rows * record * 2, record * 2, record * 2, 2)
            )
        )
        bad = torch.empty(
            pages * rows * (record + 32) * 2, dtype=torch.uint8
        ).as_strided(value.shape, stride)
    values[field] = bad
    with pytest.raises(ValueError):
        public.prepare_attention_arguments(op, **values)
    assert not hasattr(op, "_workspace_storage")


def test_native_compressed_page_size_switches_workspace(business):
    op, values = business
    op.cmp_decode_strategy = "predecode"
    public.prepare_attention_arguments(op, **values)
    prior = op._workspace_storage
    values["extra_k_cache"] = torch.zeros(5, 64, 1, 320, dtype=torch.uint8)
    public.prepare_attention_arguments(op, **values)
    assert op._workspace_storage is not prior
    assert op._workspace["cmp_dec"].shape == (5, 64, 512)


def test_noncontiguous_coefficients_rejected(business):
    op, values = business
    values["rope_cos"] = torch.empty(8, 64)[:, ::2]
    with pytest.raises(ValueError, match="rope_cos"):
        public.prepare_attention_arguments(op, **values)


@pytest.mark.parametrize("n", [1, 4, 8])
def test_lengths_and_overallocated_rows(business, n):
    op, values = business
    values["rope_cos"] = torch.empty(16, 32)
    values["rope_sin"] = torch.empty(16, 32)
    values["window_cache_slots"] = torch.empty(16, dtype=torch.int32)
    values["indices_in_kvcache"] = torch.zeros(16, 128, dtype=torch.int32)
    length = torch.zeros(n, dtype=torch.int32)
    args, _ = public.prepare_attention_arguments(
        op, **values, topk_length=length, extra_topk_length=length
    )
    assert args[14].shape == (16, 32) and op._workspace["qb"].shape[0] == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"topk_length": torch.zeros(3, dtype=torch.int32)},
        {"extra_topk_length": torch.zeros(8, dtype=torch.int64)},
        {
            "topk_length": torch.zeros(4, dtype=torch.int32),
            "extra_topk_length": torch.zeros(8, dtype=torch.int32),
        },
        {"out": torch.empty(8, 16, 512)},
    ],
)
def test_bad_optional_inputs_rejected(business, kwargs):
    op, values = business
    with pytest.raises(ValueError):
        public.prepare_attention_arguments(op, **values, **kwargs)
    assert not hasattr(op, "_workspace")


def test_public_signature_has_no_intermediates():
    source = (Path(public.__file__).parent / "mega_mla.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MegaMlaOp"
    )
    run = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    assert [a.arg for a in run.args.args] == [
        "self",
        "hidden_states",
        "query_norm_weight",
        "kv_norm_weight",
        "query_down_proj_weight",
        "kv_proj_weight",
        "query_up_proj_weight",
        "query_down_proj_scale",
        "kv_proj_scale",
        "query_up_proj_scale",
        "k_cache",
        "extra_k_cache",
        "indices_in_kvcache",
        "extra_indices_in_kvcache",
        "attn_sink",
        "rope_cos",
        "rope_sin",
        "window_cache_slots",
    ]
    assert run.args.vararg is None
    assert [a.arg for a in run.args.kwonlyargs] == [
        "out",
        "topk_length",
        "extra_topk_length",
        "block_table",
        "extra_block_table",
        "cu_seqlens_qo",
        "seqused_q",
        "cache_seqlens",
        "extra_cache_seqlens",
        "sequence_metadata",
        "return_sequence_metadata",
    ]
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "mega_mla"
    )
    assert [a.arg for a in function.args.args] == [a.arg for a in run.args.args[1:]]
    assert function.args.vararg is None
    assert [a.arg for a in function.args.kwonlyargs] == [
        *[a.arg for a in run.args.kwonlyargs],
        "grid",
        "cmp_decode_strategy",
    ]


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_block_tables_are_passed_without_host_conversion(business, batch):
    op, values = business
    win_table = torch.zeros(batch, 2, dtype=torch.int32)
    cmp_table = torch.zeros(batch, 4, dtype=torch.int32)
    args, _ = public.prepare_attention_arguments(
        op, **values, block_table=win_table, extra_block_table=cmp_table
    )
    assert args[17] is win_table and args[18] is cmp_table
    assert (
        args[11] is values["indices_in_kvcache"]
        and args[12] is values["extra_indices_in_kvcache"]
    )
    assert (
        public.validate_cache_block_tables(
            win_table, cmp_table, 8, values["hidden_states"].device
        )
        == 8 // batch
    )


@pytest.mark.parametrize(
    "win,cmp",
    [
        (None, torch.zeros(2, 4, dtype=torch.int32)),
        (torch.zeros(2, 2, dtype=torch.int32), None),
        (torch.zeros(3, 2, dtype=torch.int32), torch.zeros(3, 4, dtype=torch.int32)),
        (torch.zeros(2, 2, dtype=torch.int32), torch.zeros(4, 4, dtype=torch.int32)),
        (torch.zeros(2, 0, dtype=torch.int32), torch.zeros(2, 4, dtype=torch.int32)),
        (torch.zeros(0, 2, dtype=torch.int32), torch.zeros(0, 4, dtype=torch.int32)),
        (torch.zeros(2, dtype=torch.int32), torch.zeros(2, 4, dtype=torch.int32)),
        (torch.zeros(2, 2, dtype=torch.int64), torch.zeros(2, 4, dtype=torch.int32)),
        (
            torch.zeros(2, 4, dtype=torch.int32)[:, ::2],
            torch.zeros(2, 4, dtype=torch.int32),
        ),
    ],
)
def test_invalid_block_tables_rejected_before_allocation(business, win, cmp):
    op, values = business
    with pytest.raises(ValueError, match="block_table"):
        public.prepare_attention_arguments(
            op, **values, block_table=win, extra_block_table=cmp
        )
    assert not hasattr(op, "_workspace")


@pytest.mark.parametrize("d", [64, 512])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("heads", [8, 16, 64])
def test_compact_coefficients_match_duplicated_reference(d, inverse, heads):
    """Independent CPU layout check, including BF16 rounding and Q scaling."""
    torch.manual_seed(123)
    x = torch.randn(8, heads, d).to(torch.bfloat16).float()
    theta = torch.randn(8, 1, 32)
    cos, sin = theta.cos(), theta.sin()
    legacy_cos, legacy_sin = cos.repeat_interleave(2, -1), sin.repeat_interleave(2, -1)
    even, odd = x[..., -64::2], x[..., -63::2]
    sgn = 1.0 if inverse else -1.0
    compact = x.clone()
    compact[..., -64::2] = even * cos + sgn * (odd * sin)
    compact[..., -63::2] = odd * cos - sgn * (even * sin)
    legacy = x.clone()
    legacy[..., -64::2] = even * legacy_cos[..., ::2] + sgn * (
        odd * legacy_sin[..., ::2]
    )
    legacy[..., -63::2] = odd * legacy_cos[..., ::2] - sgn * (
        even * legacy_sin[..., ::2]
    )
    assert torch.equal(compact.to(torch.bfloat16), legacy.to(torch.bfloat16))
    assert torch.equal(
        (compact.to(torch.bfloat16).float() / d**0.5).to(torch.bfloat16),
        (legacy.to(torch.bfloat16).float() / d**0.5).to(torch.bfloat16),
    )
    assert torch.equal(compact[..., :-64], x[..., :-64])


def _set_heads(op, values, heads):
    op.heads = heads
    values["query_up_proj_weight"] = torch.empty(
        heads * 512, 1280, dtype=torch.float8_e4m3fn
    )
    values["query_up_proj_scale"] = torch.empty(
        heads * 512, 20, 2, dtype=torch.float8_e8m0fnu
    )
    values["attn_sink"] = torch.empty(1, heads)


@pytest.mark.parametrize("heads", [8, 16, 64])
@pytest.mark.parametrize("tokens", [1, 8, 16, 33, 64])
def test_head_dependent_inputs_and_workspaces(business, heads, tokens):
    op, values = business
    _set_heads(op, values, heads)
    for name in (
        "hidden_states",
        "indices_in_kvcache",
        "extra_indices_in_kvcache",
        "rope_cos",
        "rope_sin",
        "window_cache_slots",
    ):
        old = values[name]
        values[name] = torch.empty((tokens, *old.shape[1:]), dtype=old.dtype)
    args, out = public.prepare_attention_arguments(op, **values)
    assert out.shape == (tokens, heads, 512)
    assert out.dtype == torch.bfloat16
    assert op._workspace["qfull"].shape == (tokens, heads * 512)
    assert op._workspace["qb"].shape == (tokens, heads, 512)
    assert op._workspace["s"].shape == op._workspace["p"].shape == (tokens, heads, 640)
    assert op._workspace["qr"].shape == (tokens, 1280)
    assert op._workspace["kv"].shape == op._workspace["kvn"].shape == (tokens, 512)
    assert op._workspace["kg"].shape == (tokens, 640, 512)
    assert args[14].shape == (tokens, 32)
    expected_cmp = (1, 1, 512) if (heads, tokens) == (64, 64) else (5, 128, 512)
    assert op._workspace["cmp_dec"].shape == expected_cmp


@pytest.mark.parametrize("heads", [8, 16, 64])
@pytest.mark.parametrize(
    "name", ["query_up_proj_weight", "query_up_proj_scale", "attn_sink", "out"]
)
def test_reject_head_mismatch_before_allocation(business, heads, name):
    op, values = business
    _set_heads(op, values, heads)
    wrong = 16 if heads != 16 else 64
    shape, dtype = {
        "query_up_proj_weight": ((wrong * 512, 1280), torch.float8_e4m3fn),
        "query_up_proj_scale": ((wrong * 512, 20, 2), torch.float8_e8m0fnu),
        "attn_sink": ((1, wrong), torch.float32),
        "out": ((8, wrong, 512), torch.bfloat16),
    }[name]
    values[name] = torch.empty(shape, dtype=dtype)
    with pytest.raises(ValueError, match=name):
        public.prepare_attention_arguments(op, **values)
    assert not hasattr(op, "_workspace")


def test_workspace_identity_includes_heads(business):
    op, values = business
    first, out0 = public.prepare_attention_arguments(op, **values)
    key0 = op._workspace_key
    first_qfull = op._workspace["qfull"]
    _set_heads(op, values, 64)
    second, out1 = public.prepare_attention_arguments(op, **values)
    assert key0 != op._workspace_key
    assert first_qfull.data_ptr() != op._workspace["qfull"].data_ptr()
    assert out0.shape == (8, 16, 512) and out1.shape == (8, 64, 512)
    assert op._workspace["qfull"].shape == (8, 32768)
    ptrs = {name: v.data_ptr() for name, v in op._workspace.items()}
    public.prepare_attention_arguments(op, **values)
    assert ptrs == {name: v.data_ptr() for name, v in op._workspace.items()}


@pytest.mark.parametrize("strategy,pages", [("predecode", 5), ("indexed", 32769)])
@pytest.mark.parametrize("filter_indices", [False, True])
@pytest.mark.parametrize(
    "tokens,heads,k",
    [(1, 8, 640), (8, 16, 2176), (8, 64, 2176), (33, 16, 640), (64, 64, 640)],
)
def test_workspace_regions_are_disjoint_and_aligned(
    strategy, pages, tokens, heads, k, filter_indices
):
    from attention_workspace import AttentionWorkspaceLayout

    layout = AttentionWorkspaceLayout(
        tokens, heads, k, pages, 128, strategy, filter_indices=filter_indices
    )
    storage, views = layout.allocate("cpu")
    bound = layout.bind(*layout.arguments(storage))
    end = 0
    byte_storage = storage.view(torch.uint8)
    byte_storage.fill_(0xA5)
    for tag, (name, shape, dtype, offset, size) in enumerate(layout.regions, 1):
        assert offset >= end and offset % 512 == 0
        assert offset + size <= layout.nbytes
        assert views[name].data_ptr() == storage.data_ptr() + offset
        assert bound[name].data_ptr() == views[name].data_ptr()
        assert bound[name].dtype == views[name].dtype and bound[name].shape == shape
        views[name].view(torch.uint8).fill_(tag)
        end = offset + size
    for tag, (name, shape, dtype, offset, size) in enumerate(layout.regions, 1):
        assert torch.all(byte_storage[offset : offset + size] == tag)
    assert storage.numel() * storage.element_size() == layout.nbytes
    assert layout.nbytes % 512 == 0
    if strategy == "indexed":
        assert views["cmp_dec"].numel() == 512


def test_device_entry_has_typed_private_workspace_aliases():
    tree = ast.parse(
        (Path(public.__file__).parent / "mega_mla_kernel.py").read_text(
            encoding="utf-8"
        )
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MegaMlaKernel"
    )
    kernel = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "mega_mla"
    )
    assert [a.arg for a in kernel.args.args] == [
        "self",
        "hidden_states_gm",
        "query_norm_weight_gm",
        "kv_norm_weight_gm",
        "query_down_proj_weight_gm",
        "kv_proj_weight_gm",
        "query_up_proj_weight_gm",
        "query_down_proj_scale_gm",
        "kv_proj_scale_gm",
        "query_up_proj_scale_gm",
        "k_cache_gm",
        "extra_k_cache_gm",
        "indices_in_kvcache_gm",
        "extra_indices_in_kvcache_gm",
        "attn_sink_gm",
        "rope_cos_gm",
        "rope_sin_gm",
        "window_cache_slots_gm",
        "block_table_gm",
        "extra_block_table_gm",
        "topk_length_gm",
        "extra_topk_length_gm",
        "out_gm",
        "workspace_gm",
        "workspace_bf16_gm",
        "workspace_fp8_gm",
        "workspace_e8m0_gm",
        "workspace_int32_gm",
    ]


# ============================================================================
# Scheduling, ownership and resource budgets
# ============================================================================


GRID_COUNTS = (1, 2, 3, 7, 14, 20, 27, 28, 29, 31, 32)


@pytest.mark.parametrize("physical_cores", [28, 32])
def test_all_device_grid_counts(physical_cores):
    for grid in range(1, physical_cores + 1):
        assert validate_aic_grid(grid, physical_cores) == grid
        public.validate_attention_geometry(grid=grid)
    with pytest.raises(ValueError, match="physical AIC count"):
        validate_aic_grid(physical_cores + 1, physical_cores)


@pytest.mark.parametrize("grid", [0, -1, True, False, 1.0, 28.0, "8"])
def test_invalid_device_grid_counts(grid):
    with pytest.raises(ValueError, match="positive integer"):
        validate_aic_grid(grid, 28)


def test_overlarge_grid_rejected_before_workspace(business, monkeypatch):
    op, values = business
    op.grid = 29
    values["hidden_states"] = SimpleNamespace(
        ndim=2, shape=(8, 5120), device=SimpleNamespace(type="npu")
    )
    monkeypatch.setattr(public, "available_aic_count", lambda _: 28)
    with pytest.raises(ValueError, match="current stream AIC/AIV launch limit 28"):
        public.prepare_attention_arguments(op, **values)
    assert not hasattr(op, "_workspace_storage")


@pytest.mark.parametrize("grid", list(range(1, 33)))
@pytest.mark.parametrize("m_tiles", [1, 2])
def test_projection_tasks_cover_all_outputs(grid, m_tiles):
    tree = ast.parse((ROOT / "mega_mla_kernel.py").read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MegaMlaKernel"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_run_qa_kv_projection", "_compute_qa_kv_projection_task")
    ]
    for method in methods:
        method.decorator_list = []
    node = ast.ClassDef(
        name="Projection", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    namespace = dict(
        const_expr=bool,
        _create_projection_pipeline=lambda *a, **kw: None,
        _bind_projection_pipeline=lambda *a: None,
        BN1=128,
        BK1=256,
        KL1_1=512,
        HID=5120,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            "projection_owners",
            "exec",
        ),
        namespace,
    )
    obj = namespace["Projection"]()
    obj.grid = grid
    obj.proj = SimpleNamespace(
        p1_m=16, p1_m_tiles=m_tiles, n1_blocks=10, n3_blocks=4, k1="qa", k3="kv"
    )
    calls = []
    obj._compute_split_k_projection_tile = (
        lambda engine, out, a, b, sa, sb, mi, ni, split: calls.append(
            (engine, mi, ni, split)
        )
    )
    for block in range(grid):
        obj._run_qa_kv_projection(*([None] * 8), block)
    expected = [
        (engine, mi, ni, split)
        for engine, columns in (("qa", 10), ("kv", 4))
        for mi in range(m_tiles)
        for ni in range(columns)
        for split in range(2)
    ]
    assert sorted(calls) == sorted(expected)


def _constructor(filename, class_name, namespace=None):
    """Execute only the actual host constructor, without loading DSL/JIT code."""
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    node = ast.ClassDef(
        name=class_name, bases=[], keywords=[], body=[init], decorator_list=[]
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    ns = {
        "validate_query_heads": validate_query_heads,
        "HEAD_DIM": HEAD_DIM,
        "validate_attention_geometry": public.validate_attention_geometry,
        "select_small_k_attention_head_tile": select_small_k_attention_head_tile,
        "COMPRESSED_DECODE_STRATEGIES": COMPRESSED_DECODE_STRATEGIES,
        "get_mem_size": lambda _: 253952,
        "SOFTMAX_UB_RESERVE_BYTES": 16 * 1024,
        **(namespace or {}),
    }
    exec(compile(module, filename, "exec"), ns)
    return ns[class_name]


@pytest.fixture
def functional_entry(business, monkeypatch):
    """Exercise the actual function host path with the existing CPU validator."""
    created, synchronized = [], []
    context = [torch.device("cpu"), SimpleNamespace(npu_stream=0), 28]
    base = _constructor("mega_mla.py", "MegaMlaOp")

    class Launcher(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.append(self)

        def run(self, *inputs, **kwargs):
            args, result = public.prepare_attention_arguments(self, *inputs, **kwargs)
            self._compiled_launches.setdefault(
                (self.heads, tuple(inputs[0].shape)), object()
            )
            result.fill_(float(args[0][0, 0]))
            return (
                (result, self._sequence_metadata_result)
                if kwargs["return_sequence_metadata"]
                else result
            )

    names = {
        "_function_context",
        "_function_tensor_schema",
        "clear_caches",
        "mega_mla",
    }
    tree = ast.parse((ROOT / "mega_mla.py").read_text(encoding="utf-8"))
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ],
        type_ignores=[],
    )
    ns = dict(
        threading=threading,
        _FUNCTION_STATE=threading.local(),
        MegaMlaOp=Launcher,
        AttentionSequenceMetadata=AttentionSequenceMetadata,
        validate_query_heads=validate_query_heads,
        validate_attention_geometry=public.validate_attention_geometry,
        COMPRESSED_DECODE_STRATEGIES=COMPRESSED_DECODE_STRATEGIES,
    )
    exec(compile(ast.fix_missing_locations(module), "function_entry", "exec"), ns)
    actual_context = ns["_function_context"]
    ns["_function_context"] = lambda inputs, grid: (
        *context[:2],
        context[2] if grid is None else grid,
    )
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(device=lambda _: nullcontext()), raising=False
    )

    def stream(handle=0):
        return SimpleNamespace(
            npu_stream=handle,
            synchronize=lambda: synchronized.append((context[0], handle)),
        )

    context[1] = stream()
    return SimpleNamespace(
        call=ns["mega_mla"],
        clear=ns["clear_caches"],
        actual_context=actual_context,
        state=ns["_FUNCTION_STATE"],
        created=created,
        context=context,
        stream=stream,
        synchronized=synchronized,
        inputs=business[1],
    )


@pytest.mark.parametrize("heads", [8, 16, 64])
def test_function_entry_geometry_outputs_and_live_inputs(
    functional_entry, business, heads
):
    f = functional_entry
    _set_heads(*business, heads)
    f.inputs["hidden_states"].fill_(3)
    first = f.call(**f.inputs)
    workspace = f.created[0]._workspace_storage
    f.inputs["hidden_states"].fill_(7)
    second = f.call(**f.inputs)
    assert len(f.created) == 1 and f.created[0]._workspace_storage is workspace
    assert f.created[0].heads == heads and f.created[0].grid == 28
    assert first.shape == (8, heads, 512) and torch.all(first == 3)
    assert torch.all(second == 7) and first.data_ptr() != second.data_ptr()
    supplied = torch.empty_like(first)
    assert f.call(**f.inputs, out=supplied) is supplied and torch.all(supplied == 7)


def test_function_entry_stream_device_isolation_and_cleanup(functional_entry):
    f = functional_entry
    f.inputs["hidden_states"].zero_()
    f.call(**f.inputs)
    f.context[1] = f.stream(1)
    f.call(**f.inputs)
    assert (
        f.created[0]._workspace_storage.data_ptr()
        != f.created[1]._workspace_storage.data_ptr()
    )
    assert f.created[0]._compiled_launches is f.created[1]._compiled_launches
    f.context[0], f.context[1] = torch.device("cpu:1"), f.stream(0)
    f.call(**f.inputs)
    assert f.created[2]._compiled_launches is not f.created[0]._compiled_launches
    f.clear()
    assert len(f.synchronized) == 3 and not f.state.launchers and not f.state.programs


def test_function_entry_thread_isolation(functional_entry):
    f = functional_entry
    f.inputs["hidden_states"].zero_()
    barrier, errors = threading.Barrier(2), []

    def invoke():
        try:
            barrier.wait(timeout=10)
            f.call(**f.inputs)
        except Exception as error:
            errors.append(error)

    workers = [threading.Thread(target=invoke) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
    assert not errors and all(not worker.is_alive() for worker in workers)
    assert (
        len(f.created) == 2
        and f.created[0]._compiled_launches is not f.created[1]._compiled_launches
    )


def test_function_entry_rejects_bad_input_without_retaining_state(functional_entry):
    f = functional_entry
    with pytest.raises(ValueError, match="NPU"):
        f.actual_context(f.inputs["hidden_states"], None)
    original = f.inputs["query_norm_weight"]
    f.inputs["query_norm_weight"] = torch.empty(1, 12)
    with pytest.raises(ValueError, match="query_norm_weight"):
        f.call(**f.inputs)
    assert not f.state.launchers
    f.inputs["query_norm_weight"] = original
    f.inputs["hidden_states"].zero_()
    f.call(**f.inputs)
    assert len(f.state.launchers) == 1


@pytest.mark.parametrize(
    "name",
    [
        "mega_mla",
        "mega_mla_with_metadata",
        "MegaMLA",
        "MegaMLA_with_metadata",
        "deepseek_v4_1_attention_mega_kernel",
    ],
)
def test_profile_parser_accepts_function_kernel_names(tmp_path, capsys, name):
    path = tmp_path / "op_summary_test.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["Task Type", "Op Name", "Task Duration(us)"]
        )
        writer.writeheader()
        for op, duration in ((name, 30), ("unrelated_kernel", 900), (name, 34)):
            writer.writerow(
                {"Task Type": "MIX_AIC", "Op Name": op, "Task Duration(us)": duration}
            )
    analyze_profile([str(tmp_path), "--last", "2"])
    result = json.loads(capsys.readouterr().out)
    assert result["n"] == 2 and result["median_us"] == 32


@pytest.mark.parametrize("heads", [8, 16, 64])
def test_public_constructor_heads(heads):
    op_cls = _constructor("mega_mla.py", "MegaMlaOp")
    assert op_cls(heads=heads).heads == heads
    assert op_cls().heads == 16


@pytest.mark.parametrize("heads", [0, -8, 1, 32, 128, True, 8.0, "16", None])
def test_invalid_heads_rejected(heads):
    op_cls = _constructor("mega_mla.py", "MegaMlaOp")
    with pytest.raises(ValueError, match="heads"):
        op_cls(heads=heads)


@pytest.mark.parametrize("heads", [8, 16, 64])
def test_wqb_factory_and_n256_tiles(heads):
    tree = ast.parse((ROOT / "qkv_projection.py").read_text(encoding="utf-8"))
    mk2 = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_make_query_expansion_matmul"
    )
    ns = dict(
        validate_query_heads=validate_query_heads,
        HEAD_DIM=512,
        QLORA=1280,
        BN2=256,
        BK2=128,
        KL1_2=256,
        SKL1_2=1280,
        CUBES=32,
        Mxfp8MatmulKernel=lambda *a, **kw: SimpleNamespace(args=a, kwargs=kw),
    )
    exec(
        compile(
            ast.Module(body=[mk2], type_ignores=[]),
            "_make_query_expansion_matmul",
            "exec",
        ),
        ns,
    )
    proj_cls = _constructor(
        "qkv_projection.py",
        "QkvProjectionKernel",
        {
            **ns,
            "HID": 5120,
            "KVD": 512,
            "BN1": 128,
            "_make_qa_kv_projection_matmul": lambda *a: None,
        },
    )
    proj = proj_cls(8, heads)
    assert proj.k2.args[:3] == (8, heads * 512, 1280)
    assert proj.n2_blocks == {8: 16, 16: 32, 64: 128}[heads]
    assert proj.n1_blocks == 10 and proj.n3_blocks == 4
    for grid in range(1, 33):
        tiles = [
            i
            for core in range(grid)
            for i in range(core, proj.m_tiles * proj.n2_blocks, grid)
        ]
        assert sorted(tiles) == list(range(proj.m_tiles * proj.n2_blocks))


@pytest.mark.parametrize("heads", [8, 16, 64])
def test_qk_pv_logical_head_dimension(heads):
    qk_cls = _constructor("attention_matmul.py", "QueryKeyMatmulKernel")
    pv_cls = _constructor("attention_matmul.py", "ProbabilityValueMatmulKernel")
    assert qk_cls(heads, 640, 512).m == heads
    assert pv_cls(heads, 512, 640).m == heads


@pytest.mark.parametrize("heads", [8, 16, 64])
@pytest.mark.parametrize("grid", GRID_COUNTS)
@pytest.mark.parametrize("tokens", [1, 8, 13, 14, 16, 27, 28, 31, 32, 33, 64, 192])
def test_smallk_ownership_and_alignment(heads, grid, tokens):
    tile = select_small_k_attention_head_tile(heads, tokens, grid, 640)
    if not tile:
        assert heads != 64 or tokens <= grid
        return
    assert heads % tile == 0 and (tile // 2) % 16 == 0
    if tokens == 16 and grid == 32:
        owners = [(core // 2, core % 2) for core in range(grid)]
    else:
        owners = [
            (tok, h)
            for core in range(grid)
            for tok in range(core, tokens, grid)
            for h in range(heads // tile)
        ]
    covered = [(tok, h * tile + j) for tok, h in owners for j in range(tile)]
    assert sorted(covered) == [(tok, h) for tok in range(tokens) for h in range(heads)]


@pytest.mark.parametrize("k", [704, 768, 2176])
def test_smallk_gate_rejects_incompatible_k(k):
    assert select_small_k_attention_head_tile(64, 64, 32, k) == 0


@pytest.mark.parametrize("heads", [8, 16, 64])
@pytest.mark.parametrize("grid", GRID_COUNTS)
@pytest.mark.parametrize("tokens", list(range(1, 17)) + [27, 28, 31, 32])
def test_vector_rows_have_unique_owners(heads, grid, tokens):
    tree = ast.parse((ROOT / "mega_mla_kernel.py").read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MegaMlaKernel"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    selection = [
        n
        for n in init.body
        if isinstance(n, ast.Assign)
        and ast.unparse(n.targets[0])
        in ("softmax_workers", "softmax_row_tile", "self.softmax_batch_rows")
    ]
    state = SimpleNamespace(
        heads=heads, tokens=tokens, grid=grid, attn=SimpleNamespace(k=640)
    )
    ns = dict(
        self=state,
        heads=heads,
        tokens=tokens,
        grid=grid,
        k_attn=640,
        get_mem_size=lambda _: 253952,
        SOFTMAX_UB_RESERVE_BYTES=16 * 1024,
    )
    exec(
        compile(
            ast.Module(body=selection, type_ignores=[]), "softmax_selection", "exec"
        ),
        ns,
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_run_softmax_batched_heads"
    )
    stop = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "scores_ub"
    )
    owners = compile(
        ast.Module(body=fn.body[:stop], type_ignores=[]), "softmax_owners", "exec"
    )
    rows, counts = [], []
    for vec in range(2 * grid):
        if state.softmax_batch_rows:
            ns = dict(
                self=state, block=vec // 2, nblk=grid, get_subblock_id=lambda: vec % 2
            )
            exec(owners, ns)
            owned = [
                (ns["token"], ns["head_start"] + r) for r in range(ns["row_count"])
            ]
        else:
            owned = [
                (item // heads, item % heads)
                for item in range(vec, tokens * heads, 2 * grid)
            ]
        rows.extend(owned)
        counts.append(len(owned))
    assert sorted(rows) == [(tok, h) for tok in range(tokens) for h in range(heads)]
    assert max(counts) <= (tokens * heads + 2 * grid - 1) // (2 * grid)


@pytest.mark.parametrize("grid", GRID_COUNTS)
@pytest.mark.parametrize("tokens", list(range(1, 17)) + [27, 28, 31, 32])
def test_gather_nondivisible_token_ownership(grid, tokens):
    # Execute the ownership statements from the real JIT body on the CPU.
    tree = ast.parse((ROOT / "mega_mla_kernel.py").read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MegaMlaKernel"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_prepare_query_and_gather_kv"
    )
    begin = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "vec_id"
    )
    end = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.For) and ast.unparse(n.target) == "tok"
    )
    code = compile(
        ast.Module(body=fn.body[begin:end], type_ignores=[]), "gather_owners", "exec"
    )
    owned = []
    groups = []
    for vec in range(2 * grid):
        ns = dict(
            self=SimpleNamespace(tokens=tokens, grid=grid, dual_aiv=True),
            block=vec // 2,
            nblk=grid,
            get_subblock_id=lambda: vec % 2,
        )
        exec(code, ns)
        visits = list(range(ns["tok_begin"], ns["tok_end"], ns["tok_stride"]))
        assert len(visits) <= (tokens + 2 * grid - 1) // (2 * grid)
        groups.extend((tok, ns["parts"]) for tok in visits)
        for tok in visits:
            owned.extend((tok, row) for row in range(ns["part"], 16, ns["parts"]))
    assert sorted(owned) == [(tok, row) for tok in range(tokens) for row in range(16)]
    assert max(n for _, n in groups) - min(n for _, n in groups) <= 1


def test_no_fixed_h_global_and_cv_channel_budget():
    source = "\n".join(
        (ROOT / name).read_text(encoding="utf-8")
        for name in ("mega_mla_kernel.py", "attention_pipeline.py")
    )
    tree = ast.parse(source)
    assert not any(isinstance(n, ast.Name) and n.id == "H" for n in ast.walk(tree))
    assert not any(
        isinstance(n, ast.keyword) and n.arg == "dual_param" for n in ast.walk(tree)
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "_SmallKAttentionPipeline"
    )
    ns = {
        "D": 512,
        "make_layout": lambda shape, *args, **kw: shape,
        "_SmallKAttentionMatmul": lambda *a: SimpleNamespace(pv_n=128),
        "_SmallKAttentionVector": lambda *a: None,
        "RingBuffer": lambda *a, **kw: SimpleNamespace(**kw),
        "MemLoc": SimpleNamespace(UB=0, L1=1),
        "dtypes": SimpleNamespace(float32=0, bfloat16=1),
    }
    pipeline = _constructor("attention_pipeline.py", cls.name, ns)(640)
    assert pipeline.qk_ub.depth + pipeline.pv_ub.depth + pipeline.p_l1.depth == 8


@pytest.mark.parametrize("heads", [8, 16, 64])
@pytest.mark.parametrize("grid", GRID_COUNTS)
@pytest.mark.parametrize("tokens", [1, 6, 8, 10, 11, 12, 14, 15, 16, 64])
def test_main_constructor_propagates_head_geometry(heads, grid, tokens):
    def inert(*args, **kwargs):
        return SimpleNamespace(args=args, kwargs=kwargs)

    qk = _constructor("attention_matmul.py", "QueryKeyMatmulKernel")
    pv = _constructor("attention_matmul.py", "ProbabilityValueMatmulKernel")
    ns = dict(
        D=512,
        L2_CACHE_DISABLE=4,
        COMPRESSED_DECODE_UNROLL=1,
        COMPRESSED_DECODE_DEPTH=2,
        Fp8CacheDecoder=inert,
        Fp4CacheDecoder=inert,
        CompressedCacheGather=inert,
        PagedCacheAddressing=inert,
        QkvProjectionKernel=lambda *a: SimpleNamespace(
            args=a, k2=SimpleNamespace(base_m=64)
        ),
        DualCacheVectorAttentionKernel=inert,
        CompactRotaryEmbeddingKernel=inert,
        QueryKeyMatmulKernel=qk,
        ProbabilityValueMatmulKernel=pv,
    )
    cls = _constructor("mega_mla_kernel.py", "MegaMlaKernel", ns)
    obj = cls(640, 128, 128, heads=heads, tokens=tokens, grid=grid)
    assert obj.heads == obj.qk.m == obj.pv.m == heads
    assert obj.proj.args == (tokens, heads)
    assert obj.proj.k2.l2_cache_ctl_b == obj.proj.k2.l2_cache_ctl_scale_b == 4
    if obj.flat_attention_tiles:
        assert heads in (8, 16) and tokens < 16
        for tiles in (obj.qk.n // obj.qk.base_n, obj.pv_tiles):
            bpt = grid // tokens
            assert (tokens * tiles + grid - 1) // grid < (tiles + bpt - 1) // bpt
            work = [
                item
                for block in range(grid)
                for item in range(block, tokens * tiles, grid)
            ]
            assert sorted(work) == list(range(tokens * tiles))
    if heads in (8, 16) and grid == 28:
        assert obj.flat_attention_tiles == (tokens in (10, 15))
    assert obj.attn.args[0] == heads
    expected_cv = 0
    if heads == 64 and 2 * tokens >= grid:
        expected_cv = 32 if (tokens, grid) == (16, 32) else 64
    assert obj.cv_tile_m == expected_cv

    # New ordinary H64 schedules must preserve one writer per compressed
    # slot, including empty intervals, nondivisible T and CV fallbacks.
    indexed = cls(
        640,
        128,
        128,
        heads=heads,
        tokens=tokens,
        grid=grid,
        cmp_decode_strategy="indexed",
    )
    ordinary_small = heads in (8, 16) or (
        heads == 64 and tokens < 16 and not expected_cv
    )
    assert indexed.small_token_vector_schedule == ordinary_small
    intervals = [
        (0, indexed.cmp_p1_items),
        (indexed.cmp_p1_items, indexed.cmp_p3_items),
        (indexed.cmp_p3_items, tokens * 512),
    ]
    writers = [
        item
        for begin, end in intervals
        for vec in range(2 * grid)
        for item in range(begin + vec, end, 2 * grid)
    ]
    assert sorted(writers) == list(range(tokens * 512))
    if not ordinary_small or tokens >= 16 or grid not in (28, 32):
        assert indexed.cmp_p1_items == indexed.cmp_p3_items == 0
    else:
        assert indexed.cmp_p1_items > 0
    # Direct logical tables and length-prefix fallback cannot speculatively
    # use physical-index mirrors prepared for the seqlens-filtered path.
    logical = cls(
        640,
        128,
        128,
        heads=heads,
        tokens=tokens,
        grid=grid,
        cmp_decode_strategy="indexed",
        cache_batch_stride=1,
    )
    assert logical.cmp_p1_items == logical.cmp_p3_items == 0


@pytest.mark.parametrize(
    "filename,class_name",
    [
        ("mega_mla_kernel.py", "MegaMlaKernel"),
        ("qkv_projection.py", "QkvProjectionKernel"),
    ],
)
def test_p1_selects_l0b_double_buffer_and_p3_unchanged(filename, class_name):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    selected = []
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for call in ast.walk(fn):
            if (
                isinstance(call, ast.Call)
                and ast.unparse(call.func) == "_create_projection_pipeline"
            ):
                depth = next(
                    (k.value for k in call.keywords if k.arg == "l0b_depth"), None
                )
                if depth is not None:
                    assert isinstance(depth, ast.Constant) and depth.value == 2
                    selected.append(fn.name)
    # Main mega already selected depth=2 for WqB before this change.
    assert selected == (
        ["_run_qa_kv_projection", "_run_projection_stages"]
        if class_name == "MegaMlaKernel"
        else ["_run_qa_kv_projection"]
    )


def test_real_channel_factory_l0b_budget_and_default():
    tree = ast.parse((ROOT / "qkv_projection.py").read_text(encoding="utf-8"))
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_create_projection_pipeline"
    )
    ns = dict(
        get_scale_k_len=lambda _: 80,
        make_layout=lambda shape, *args, **kw: shape,
        MemLoc=SimpleNamespace(L1="L1", L0A="L0A", L0B="L0B", L0C="L0C"),
        Float8E4M3FN="fp8",
        Float8E8M0="e8m0",
        dtypes=SimpleNamespace(int8="i8", float32="f32"),
        RingBuffer=lambda loc, dtype, shape, **kw: SimpleNamespace(
            loc=loc, shape=shape, dtype=dtype, **kw
        ),
        Buffer=lambda loc, dtype, shape, **kw: SimpleNamespace(
            loc=loc, shape=shape, dtype=dtype, depth=1, **kw
        ),
        copy_options=lambda **kw: kw,
    )
    exec(
        compile(
            ast.Module(body=[fn], type_ignores=[]),
            "_create_projection_pipeline",
            "exec",
        ),
        ns,
    )
    for m in (16, 64):
        pipeline = ns["_create_projection_pipeline"](
            m, 128, 256, 512, 2560, l0b_depth=2
        )
        assert pipeline["l1_b"].dtype == "fp8" and pipeline["l1_b"].depth == 4
        p1 = pipeline["l0b"]
        assert (p1.loc, p1.dtype, p1.shape, p1.depth) == ("L0B", "fp8", (128, 256), 2)
        assert p1.shape[0] * p1.shape[1] * p1.depth == 64 * 1024
    p3 = ns["_create_projection_pipeline"](16, 256, 128, 256, 1280)["l0b"]
    assert p3.depth == 1 and p3.shape == (256, 128)


@pytest.mark.parametrize("width", [512, 1280])
def test_rms_reference_rounds_before_reduce_and_numerator(width):
    # BF16 halfway values deliberately distinguish RN-even from FP32-direct RMS.
    values = torch.tensor([1 + 2**-8, 1 + 3 * 2**-8, -1 - 2**-8, -1 - 3 * 2**-8])
    accumulator = values.repeat(width // 4).reshape(1, width)
    gamma = torch.linspace(0.5, 1.5, width).reshape(1, width)
    rounded = torch.tensor([1.0, 1 + 2**-6, -1.0, -1 - 2**-6])
    rounded = rounded.repeat(width // 4).reshape(1, width)
    torch.testing.assert_close(
        accumulator.to(torch.bfloat16).float(), rounded, rtol=0, atol=0
    )
    expected = (
        rounded * torch.rsqrt(rounded.square().mean(-1, keepdim=True) + 1e-20) * gamma
    )
    actual = _rms_reference(accumulator, gamma)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    direct = (
        accumulator
        * torch.rsqrt(accumulator.square().mean(-1, keepdim=True) + 1e-20)
        * gamma
    )
    assert (actual - direct).abs().max() > 1e-3


@pytest.mark.parametrize(
    "method_name,math_method_name",
    [
        ("_rms_norm_quantize_query", "_rms_norm_quantize_query"),
        ("_rms_norm_key_value_row_ub", "_rms_norm_key_value_row_vf"),
    ],
)
def test_rms_source_contract_preserves_rounded_input(method_name, math_method_name):
    tree = ast.parse((ROOT / "qkv_projection.py").read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "QkvProjectionKernel"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    bf16 = next(
        n
        for n in fn.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "h_bf16_ub"
    )
    assert isinstance(bf16.value, ast.Call) and bf16.value.func.id == "Buffer"
    assert ast.unparse(bf16.value.args[0]) == "MemLoc.UB"
    assert ast.unparse(bf16.value.args[1]) == "dtypes.bfloat16"
    assert ast.unparse(bf16.value.args[2]) == "make_layout((1, n))"
    math_fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == math_method_name
    )
    calls = [n for n in ast.walk(math_fn) if isinstance(n, ast.Call)]
    narrow = next(
        n
        for n in calls
        if ast.unparse(n.func) == "rr.vcast"
        and ast.unparse(n.args[0]) == "rr.vload(h_ub, c * VECTOR_CHUNK_ELEMENTS)"
    )
    assert ast.unparse(narrow.args[1]) == "dtypes.bfloat16"
    assert any(
        k.arg == "rounding"
        and isinstance(k.value, ast.Constant)
        and k.value.value == "rn"
        for k in narrow.keywords
    )
    unpack = next(
        n
        for n in calls
        if ast.unparse(n.func) == "rr.vload_unpack"
        and ast.unparse(n.args[0]) == "h_bf16_ub"
    )
    assert any(
        k.arg == "unpack_mode" and ast.unparse(k.value) == "rr.UnpackMode.B16_TO_B32"
        for k in unpack.keywords
    )
    xf = next(
        n
        for n in ast.walk(math_fn)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "xf"
    )
    assert ast.unparse(xf.value.func) == "rr.vcast"
    assert ast.unparse(xf.value.args[0].func) == "rr.vload_unpack"
    assert ast.unparse(xf.value.args[1]) == "dtypes.float32"
    packed = next(
        n
        for n in calls
        if ast.unparse(n.func) == "rr.vstore_pack"
        and ast.unparse(n.args[0]) == "h_bf16_ub"
    )
    barriers = [
        n.lineno
        for n in calls
        if ast.unparse(n.func) == "rr.vmem_bar" and n.args[0].value == "vst_vld"
    ]
    assert any(packed.lineno < line < unpack.lineno for line in barriers)
    assert any(
        ast.unparse(n.func) == "rr.vload" and ast.unparse(n.args[0]) == "xf_ub"
        for n in calls
    )
    # No new GM destination/copy or extra all-core synchronization in the helper.
    assert not any(ast.unparse(n.func) == "global_sync_all" for n in calls)
    assert not any(
        ast.unparse(n.func) == "mem_copy" and "h_bf16_ub" in ast.unparse(n)
        for n in calls
    )


@pytest.mark.parametrize("tile_m", [32, 64])
def test_cv_live_memory_fits_hardware(tile_m):
    allocations = []

    def channel(loc, dtype, shape, depth=1, **kwargs):
        obj = SimpleNamespace(loc=loc, shape=shape, dtype=dtype, depth=depth)
        allocations.append(obj)
        return obj

    ns = dict(
        D=512,
        make_layout=lambda shape, *args, **kw: shape,
        RingBuffer=channel,
        MemLoc=SimpleNamespace(L1="L1", L0A="L0A", L0B="L0B", L0C="L0C", UB="UB"),
        dtypes=SimpleNamespace(bfloat16=2, float32=4),
        copy_options=lambda **kw: kw,
    )
    mm = _constructor("attention_pipeline.py", "_SmallKAttentionMatmul", ns)
    _pipe = _constructor(
        "attention_pipeline.py",
        "_SmallKAttentionPipeline",
        {**ns, "_SmallKAttentionMatmul": mm, "_SmallKAttentionVector": lambda *a: None},
    )(640, tile_m=tile_m)
    import math

    for loc, limit in (
        ("L1", 512 * 1024),
        ("L0A", 64 * 1024),
        ("L0B", 64 * 1024),
        ("L0C", 256 * 1024),
    ):
        assert (
            sum(
                math.prod(a.shape) * a.dtype * a.depth
                for a in allocations
                if a.loc == loc
            )
            <= limit
        )


# ============================================================================
# Compressed-cache policy and CPU decoding
# ============================================================================


@pytest.mark.parametrize(
    "t,kc,rows,expected",
    [
        (8, 512, 640, "predecode"),
        (8, 512, 4096, "predecode"),
        (8, 512, 4224, "indexed"),
        (8, 2048, 16384, "predecode"),
        (8, 2048, 16512, "indexed"),
        (1, 512, 4096, "predecode"),
        (1, 512, 4224, "indexed"),
        (64, 512, 32768, "predecode"),
        (64, 512, 32896, "indexed"),
    ],
)
def test_policy_crossover(t, kc, rows, expected):
    assert choose_compressed_cache_decode_strategy(t, rows, kc) == expected


@pytest.mark.parametrize("policy", ["predecode", "indexed"])
def test_explicit_policy(policy):
    for rows in (128, 131072):
        assert choose_compressed_cache_decode_strategy(8, rows, 2048, policy) == policy


@pytest.mark.parametrize(
    "args",
    [
        (8, 640, 512, "invalid"),
        (0, 640, 512, "auto"),
        (8, 0, 512, "auto"),
        (8, 640, 0, "auto"),
        (True, 640, 512, "auto"),
        (8, 640, 512.0, "auto"),
    ],
)
def test_bad_policy_geometry(args):
    with pytest.raises(ValueError):
        choose_compressed_cache_decode_strategy(*args)


@pytest.mark.parametrize(
    "t,h,kc,nc",
    [
        (1, 16, 512, 24),
        (8, 16, 512, 32),
        (8, 16, 2048, 96),
        (16, 16, 512, 32),
        (64, 64, 512, 5),
    ],
)
def test_profiled_switch_is_inclusive(t, h, kc, nc):
    geometry = dict(heads=h, grid=28, page_size=128, win_len=128)
    assert (
        choose_compressed_cache_decode_strategy(t, (nc - 1) * 128, kc, **geometry)
        == "predecode"
    )
    assert (
        choose_compressed_cache_decode_strategy(t, nc * 128, kc, **geometry)
        == "indexed"
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"heads": 64},
        {"grid": 32},
        {"page_size": 64},
        {"win_len": 64},
        {"mode": "cache"},
        {"has_lengths": True},
    ],
)
def test_unprofiled_geometry_keeps_original_heuristic(changed):
    geometry = dict(heads=16, grid=28, page_size=128, win_len=128)
    geometry.update(changed)
    # Measured T8/H16/Kc2048 selects indexed here; the original rule
    # still selects predecode below 16384 rows for unprofiled contexts.
    assert (
        choose_compressed_cache_decode_strategy(8, 96 * 128, 2048, **geometry)
        == "predecode"
    )


def test_profiled_public_workspace_with_length_fallback(business):
    op, values = business
    values["extra_indices_in_kvcache"] = torch.zeros(8, 2048, dtype=torch.int32)
    values["extra_k_cache"] = torch.zeros(128, 128, 1, 320, dtype=torch.uint8)
    args, _ = public.prepare_attention_arguments(op, **values)
    assert op.selected_cmp_decode_strategy == "indexed"
    assert op._workspace["cmp_dec"].shape == (1, 1, 512)
    args, _ = public.prepare_attention_arguments(
        op, **values, extra_topk_length=torch.full((2,), 2048, dtype=torch.int32)
    )
    assert op.selected_cmp_decode_strategy == "predecode"
    assert op._workspace["cmp_dec"].shape == (128, 128, 512)
    args, _ = public.prepare_attention_arguments(op, **values)
    assert op.selected_cmp_decode_strategy == "indexed"
    assert op._workspace["cmp_dec"].shape == (1, 1, 512)


def test_auto_workspace_short_long_short(business):
    op, values = business
    op.cmp_decode_strategy = "auto"
    args, _ = public.prepare_attention_arguments(op, **values)
    assert op.selected_cmp_decode_strategy == "predecode"
    assert op._workspace["cmp_dec"].shape == (5, 128, 512)
    values["extra_k_cache"] = torch.zeros(33, 128, 1, 320, dtype=torch.uint8)
    args, _ = public.prepare_attention_arguments(op, **values)
    assert op.selected_cmp_decode_strategy == "indexed"
    assert op._workspace["cmp_dec"].shape == (1, 1, 512)
    assert op._workspace["kg"].shape == (8, 640, 512)
    values["extra_k_cache"] = torch.zeros(5, 128, 1, 320, dtype=torch.uint8)
    args, _ = public.prepare_attention_arguments(op, **values)
    assert op.selected_cmp_decode_strategy == "predecode"
    assert op._workspace["cmp_dec"].shape == (5, 128, 512)


def test_same_geometry_forced_strategy_reallocates(business):
    op, values = business
    op.cmp_decode_strategy = "predecode"
    public.prepare_attention_arguments(op, **values)
    old = op._workspace
    op.cmp_decode_strategy = "indexed"
    public.prepare_attention_arguments(op, **values)
    assert old is not op._workspace
    assert op._workspace["cmp_dec"].numel() == 512


@pytest.mark.parametrize("length_kind", ["none", "batch", "token", "zero"])
@pytest.mark.parametrize("seed", [19, 31, 47])
def test_selected_decode_matches_full_reference(length_kind, seed):
    torch.manual_seed(seed)
    pages, ps, t, kc = 5, 128, 8, 512
    cache = torch.randint(0, 256, (pages, ps, 1, 320), dtype=torch.uint8)
    scales = torch.exp2(torch.randint(-12, 13, (pages, ps, 32)).float())
    cache[..., 256:] = scales.to(torch.bfloat16).view(torch.uint8).unsqueeze(2)
    idx = torch.randint(0, pages * ps, (t, kc), dtype=torch.int32)
    idx[:, :8] = torch.tensor([0, 127, 128, 639, 128, -1, -7, 0])
    lengths = {
        "none": None,
        "batch": torch.tensor([0, 257], dtype=torch.int32),
        "token": torch.tensor([0, 1, 127, 128, 129, 257, 511, 512], dtype=torch.int32),
        "zero": torch.zeros(t, dtype=torch.int32),
    }[length_kind]
    valid = valid_cmp_indices(idx, lengths)
    # Masked positive out-of-range entries must never be dereferenced.
    idx[~valid] = pages * ps + 100
    idx[:, 5:7] = -1
    valid = valid_cmp_indices(idx, lengths)
    full = dequantize_fp4(cache).reshape(-1, 512)
    safe = torch.where(valid, idx, 0).long()
    expected = full[safe]
    expected[~valid] = 0
    actual = decode_selected_cmp(cache, idx, lengths)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# ============================================================================
# Sequence metadata and general page-table routing
# ============================================================================


def _meta():
    return dict(
        cu_seqlens_qo=torch.tensor([0, 2, 2, 5, 8], dtype=torch.int32),
        seqused_q=torch.tensor([1, 0, 2, 1], dtype=torch.int32),
        cache_seqlens=torch.tensor([129, 0, 256, 1], dtype=torch.int32),
        extra_cache_seqlens=torch.tensor([257, 0, 512, 128], dtype=torch.int32),
        block_table=torch.tensor([[1, 0], [-1, -1], [0, 1], [1, 0]], dtype=torch.int32),
        extra_block_table=torch.tensor(
            [[4, 2, 0, 1], [-1] * 4, [1, 3, 2, 4], [2, 1, 0, 4]], dtype=torch.int32
        ),
    )


def test_sequence_args_and_workspace_reuse(business, monkeypatch):
    op, values = business
    meta = _meta()
    # Metadata validation must never synchronize NPU values back to host.
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *_: pytest.fail("unexpected D2H"))
    monkeypatch.setattr(
        torch.Tensor, "tolist", lambda *_: pytest.fail("unexpected metadata read")
    )
    legacy, _ = public.prepare_attention_arguments(op, **values)
    original_layout = op._workspace_layout
    args, _ = public.prepare_attention_arguments(op, **values, **meta)
    assert len(legacy) == 27 and len(args) == 31
    assert all(
        args[27 + i] is meta[n]
        for i, n in enumerate(
            ("cu_seqlens_qo", "seqused_q", "cache_seqlens", "extra_cache_seqlens")
        )
    )
    assert op._workspace_layout.regions[:14] == original_layout.regions
    assert op._workspace["seq_win_idx"].shape == (8, 128)
    assert op._workspace["seq_cmp_idx"].shape == (8, 512)
    assert op._workspace["seq_slot"].shape == (8,)
    ptr = args[22].data_ptr()
    public.prepare_attention_arguments(op, **values, **meta)
    assert op._workspace_storage.data_ptr() == ptr
    legacy_again, _ = public.prepare_attention_arguments(op, **values)
    assert len(legacy_again) == 27 and op._sequence_config is None
    assert op._workspace_layout.regions == original_layout.regions


@pytest.mark.parametrize("spans", [[0, 1, 3, 8], [0, 0, 0, 0, 1, 1, 1, 2, 5, 8]])
def test_unequal_and_empty_batch_metadata(business, spans):
    op, values = business
    b = len(spans) - 1
    args, _ = public.prepare_attention_arguments(
        op,
        **values,
        cu_seqlens_qo=torch.tensor(spans, dtype=torch.int32),
        block_table=torch.ones(b, 2, dtype=torch.int32),
        extra_block_table=torch.ones(b, 4, dtype=torch.int32),
    )
    assert len(args) == 31 and op._sequence_config.batches == b


@pytest.mark.parametrize(
    "name", ["cu_seqlens_qo", "seqused_q", "cache_seqlens", "extra_cache_seqlens"]
)
@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(4, dtype=torch.int64),
        torch.zeros(2, 2, dtype=torch.int32),
        torch.zeros(8, dtype=torch.int32)[::2],
        torch.empty(0, dtype=torch.int32),
    ],
)
def test_invalid_sequence_metadata_rejected_before_allocation(business, name, bad):
    op, values = business
    meta = _meta()
    meta[name] = bad
    with pytest.raises(ValueError, match=name):
        public.prepare_attention_arguments(op, **values, **meta)
    assert not hasattr(op, "_workspace")


def test_batch_mismatch_and_kv_lengths_without_tables(business):
    op, values = business
    with pytest.raises(ValueError, match="same B"):
        public.prepare_attention_arguments(
            op, **values, **(_meta() | {"seqused_q": torch.zeros(3, dtype=torch.int32)})
        )
    for name in ("cache_seqlens", "extra_cache_seqlens"):
        with pytest.raises(ValueError, match="both block tables"):
            public.prepare_attention_arguments(
                op, **values, **{name: torch.ones(4, dtype=torch.int32)}
            )
    with pytest.raises(ValueError, match="cu_seqlens_qo"):
        public.prepare_attention_arguments(
            op, **values, seqused_q=torch.ones(3, dtype=torch.int32)
        )


def test_independent_topk_batch_or_query_modes(business):
    op, values = business
    public.prepare_attention_arguments(
        op,
        **values,
        **_meta(),
        topk_length=torch.ones(4, dtype=torch.int32),
        extra_topk_length=torch.ones(8, dtype=torch.int32),
    )
    assert (
        not op._sequence_config.win_topk_per_query
        and op._sequence_config.cmp_topk_per_query
    )
    with pytest.raises(ValueError, match="topk_length"):
        public.prepare_attention_arguments(
            op, **values, **_meta(), topk_length=torch.ones(2, dtype=torch.int32)
        )


def test_oracle_explicit_page_and_length_boundaries():
    # B=3, a zero-query request in the middle; Q1/Q3 padded, Q0/Q2 active.
    idx = torch.tensor([[0, 1, 2, 3, 4, -1, 2**30]] * 4, dtype=torch.int32)
    cu = torch.tensor([0, 2, 2, 4], dtype=torch.int32)
    used_q = torch.tensor([1, 0, 1], dtype=torch.int32)
    tables = torch.tensor([[2, 1], [-1, -1], [1, 2]], dtype=torch.int32)
    win, cmp, slot = filter_sequences(
        idx,
        idx,
        torch.tensor([4, 5, 6, 7], dtype=torch.int32),
        2,
        3,
        3,
        cu_seqlens_qo=cu,
        seqused_q=used_q,
        cache_seqlens=torch.tensor([3, 0, 1], dtype=torch.int32),
        extra_cache_seqlens=torch.tensor([4, 0, 4], dtype=torch.int32),
        block_table=tables,
        extra_block_table=tables,
        topk_length=torch.tensor([7, 0, 7], dtype=torch.int32),
        extra_topk_length=torch.tensor([2, 7, 3, 7], dtype=torch.int32),
    )
    assert win.tolist() == [
        [4, 5, 2, -1, -1, -1, -1],
        [-1] * 7,
        [2, -1, -1, -1, -1, -1, -1],
        [-1] * 7,
    ]
    assert cmp.tolist() == [
        [4, 5, -1, -1, -1, -1, -1],
        [-1] * 7,
        [2, 3, 4, -1, -1, -1, -1],
        [-1] * 7,
    ]
    # Q2 has an out-of-pool slot; Q1/Q3 must not scatter despite valid slots.
    assert slot.tolist() == [4, -1, -1, -1]


def test_common_query_metadata_export_and_cross_layer_reuse(business):
    op, values = business
    raw = _meta()
    args, _ = public.prepare_attention_arguments(
        op, **values, **raw, return_sequence_metadata=True
    )
    metadata = op._sequence_metadata_result
    assert len(args) == 32 and args[31] is metadata.query_info
    assert tuple(metadata.query_info.shape) == (8, 8)
    assert (
        metadata.query_info.untyped_storage().data_ptr()
        != op._workspace_storage.data_ptr()
    )
    # Simulate raw device stores, which do not bump Tensor's host version.
    metadata.query_info.data.fill_(17)
    old = metadata.query_info.clone()
    public.prepare_attention_arguments(
        op, **values, **raw, return_sequence_metadata=True
    )
    assert (
        op._sequence_metadata_result.query_info.data_ptr()
        != metadata.query_info.data_ptr()
    )
    torch.testing.assert_close(metadata.query_info, old, rtol=0, atol=0)
    # Another layer may use different indices, tables, KV lengths and page pool.
    layer = {
        name: value
        for name, value in raw.items()
        if name not in ("cu_seqlens_qo", "seqused_q")
    }
    layer["extra_block_table"] = layer["extra_block_table"].roll(1, 1)
    layer["extra_cache_seqlens"] = torch.ones(4, dtype=torch.int32)
    args, _ = public.prepare_attention_arguments(
        op, **values, **layer, sequence_metadata=metadata
    )
    assert (
        op._sequence_config.query_metadata_mode == "reuse"
        and args[31] is metadata.query_info
    )
    assert (
        args[12] is values["extra_indices_in_kvcache"]
        and args[18] is layer["extra_block_table"]
    )
    raw["seqused_q"].add_(0)
    with pytest.raises(ValueError, match="stale"):
        public.prepare_attention_arguments(
            op, **values, **layer, sequence_metadata=metadata
        )


def test_metadata_validation_has_no_d2h(business, monkeypatch):
    op, values = business
    metadata = AttentionSequenceMetadata(torch.empty(8, 8, dtype=torch.int32), 8, 4)
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *_: pytest.fail("unexpected D2H"))
    monkeypatch.setattr(
        torch.Tensor, "tolist", lambda *_: pytest.fail("unexpected host read")
    )
    public.prepare_attention_arguments(op, **values, sequence_metadata=metadata)
    with pytest.raises(ValueError, match="replaces"):
        public.prepare_attention_arguments(
            op, **values, **_meta(), sequence_metadata=metadata
        )


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize(
    "heads,tokens,paged,strategy,packed",
    [
        (16, 8, True, "indexed", True),
        (64, 8, True, "predecode", True),
        (16, 64, True, "indexed", True),
        (16, 1, False, "indexed", True),
        (16, 8, True, "indexed", False),
    ],
)
def test_sequence_mega_codegen(heads, tokens, paged, strategy, packed):
    if shutil.which("bisheng") is None:
        pytest.skip("CANN bisheng unavailable; no AOT result")
    dsl = pytest.importorskip("cannbotdsl")
    from mega_mla import (
        MegaMlaKernel,
        MegaMlaOp,
    )
    from attention_sequence_filter import AttentionSequenceFilter

    t, h, b, nc = tokens, heads, 3 if packed else 4, 42
    cu = torch.zeros(b + 1, dtype=torch.int32)
    used = torch.zeros(b, dtype=torch.int32)
    wi, ci = torch.zeros(b, 2, dtype=torch.int32), torch.zeros(b, 8, dtype=torch.int32)
    cfg = validate_attention_sequences(
        t,
        torch.device("cpu"),
        cu_seqlens_qo=cu if packed else None,
        seqused_q=used,
        cache_seqlens=used if paged else None,
        extra_cache_seqlens=used if paged else None,
        block_table=wi if paged else None,
        extra_block_table=ci if paged else None,
        topk_length=used,
        extra_topk_length=torch.zeros(t, dtype=torch.int32),
    )
    layout = AttentionWorkspaceLayout(t, h, 640, nc, 128, strategy, filter_indices=True)
    dt = dsl.dtypes
    shapes = [
        ((t, 5120), dt.bfloat16),
        ((1, 1280), dt.float32),
        ((1, 512), dt.float32),
        ((1280, 5120), dt.float8_e4m3fn),
        ((512, 5120), dt.float8_e4m3fn),
        ((h * 512, 1280), dt.float8_e4m3fn),
        ((1280, 80, 2), dt.float8_e8m0),
        ((512, 80, 2), dt.float8_e8m0),
        ((h * 512, 20, 2), dt.float8_e8m0),
        ((4, 128, 1, 544), dt.uint8),
        ((nc, 128, 1, 320), dt.uint8),
        ((t, 128), dt.int32),
        ((t, 512), dt.int32),
        ((1, h), dt.float32),
        ((t, 32), dt.float32),
        ((t, 32), dt.float32),
        ((t,), dt.int32),
        ((b, 2) if paged else (t, 128), dt.int32),
        ((b, 8) if paged else (t, 512), dt.int32),
        ((b,), dt.int32),
        ((t,), dt.int32),
        ((t, h, 512), dt.bfloat16),
        ((layout.nbytes // 4,), dt.float32),
        ((layout.nbytes // 2,), dt.bfloat16),
        ((layout.nbytes,), dt.float8_e4m3fn),
        ((layout.nbytes,), dt.float8_e8m0),
        ((layout.nbytes // 4,), dt.int32),
        ((b + 1,) if packed else (t, 128), dt.int32),
        ((b,), dt.int32),
        ((b,) if paged else (t, 128), dt.int32),
        ((b,) if paged else (t, 128), dt.int32),
    ]
    op = MegaMlaOp(grid=28, heads=h, cmp_decode_strategy=strategy)
    op.kernel = MegaMlaKernel(
        640,
        128,
        128,
        tokens=t,
        grid=28,
        heads=h,
        cmp_decode_strategy=strategy,
        workspace_layout=layout,
    )
    op.kernel.sequence_filter = AttentionSequenceFilter(cfg, 128, 128, 512)
    fn = dsl.compile(
        op._launch_mega_mla_with_seqlens,
        *(
            dsl.TensorSpec(s, d, storage_format="nz" if i in (3, 4, 5) else "nd")
            for i, (s, d) in enumerate(shapes)
        ),
    )
    try:
        assert Path(fn.so_path).is_file()
    finally:
        fn.close()


@pytest.mark.npu
@pytest.mark.parametrize(
    "columns", [8193, 16385, 1, 63, 64, 65, 84, 127, 128, 129, 257, 8191, 8192]
)
@pytest.mark.parametrize("mode", ["none", "export", "reuse"])
def test_general_page_table_routes_device(columns, mode):
    _exercise_general_page_routes(columns, mode, 8, 512)


@pytest.mark.npu
@pytest.mark.parametrize("columns,mode", [(257, "export"), (8193, "reuse")])
def test_general_page_table_routes_t64_device(columns, mode):
    _exercise_general_page_routes(columns, mode, 64, 2048)


@pytest.mark.npu
@pytest.mark.parametrize("mode", ["none", "export", "reuse"])
def test_general_page_table_sparse_segments_device(mode):
    # Selected pages occupy segments 0, 1 and 4; segments 2/3 stay empty.
    _exercise_general_page_routes(32769, mode, 8, 512, sparse=True)


def _exercise_general_page_routes(columns, mode, tokens, cmp_len, sparse=False):
    dsl = pytest.importorskip("cannbotdsl")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    AttentionSequenceRouteProbe = _device_probes().AttentionSequenceRouteProbe
    torch.npu.set_device(0)
    torch.manual_seed(133 + columns)
    capacity = columns * 128
    cu = torch.tensor([0, 2, 2, 5, tokens], dtype=torch.int32)
    uq = torch.tensor([1, 0, 2, tokens - 6], dtype=torch.int32)
    uw = torch.tensor([capacity, 0, max(0, capacity - 9), capacity], dtype=torch.int32)
    uc = uw.clone()
    wi = torch.randint(capacity, (tokens, 128), dtype=torch.int32)
    ci = torch.randint(capacity, (tokens, cmp_len), dtype=torch.int32)
    edge = torch.tensor(
        [
            0,
            capacity - 1,
            capacity,
            -1,
            -7,
            63 * 128,
            64 * 128,
            83 * 128,
            127 * 128,
            8191 * 128,
            8192 * 128,
            (columns - 1) * 128,
        ],
        dtype=torch.int32,
    )
    wi[:, : len(edge)], ci[:, : len(edge)] = edge, edge
    if sparse:
        wi = edge.repeat((tokens, (128 + len(edge) - 1) // len(edge)))[
            :, :128
        ].contiguous()
        ci = edge.repeat((tokens, (cmp_len + len(edge) - 1) // len(edge)))[
            :, :cmp_len
        ].contiguous()
    wt = torch.randint(4, (4, columns), dtype=torch.int32)
    ct = torch.randint(84, (4, columns), dtype=torch.int32)
    wt[2, -1], ct[2, -1] = -1, 84
    wl = torch.arange(tokens, dtype=torch.int32).mul(37).remainder(129)
    cl = torch.arange(tokens, dtype=torch.int32).mul(73).remainder(cmp_len + 1)
    wl[0], cl[0] = 128, cmp_len
    slots = torch.tensor(
        [128, 0, 256, 511, 512, -1, 127, 129], dtype=torch.int32
    ).repeat((tokens + 7) // 8)[:tokens]
    cpu = dict(
        cu_seqlens_qo=cu,
        seqused_q=uq,
        cache_seqlens=uw,
        extra_cache_seqlens=uc,
        block_table=wt,
        extra_block_table=ct,
        topk_length=wl,
        extra_topk_length=cl,
    )
    values = [wi, ci, wl, cl, wt, ct, cu, uq, uw, uc, slots]
    inputs = [v.npu() for v in values]
    meta = query_metadata_reference(tokens, cu, uq).npu()
    config_args = {name: value.npu() for name, value in cpu.items()}
    if mode == "reuse":
        config_args.pop("cu_seqlens_qo")
        config_args.pop("seqused_q")
        config_args["sequence_metadata"] = AttentionSequenceMetadata(meta, tokens, 4)
    config = validate_attention_sequences(
        tokens,
        inputs[0].device,
        **config_args,
        return_sequence_metadata=mode == "export",
    )
    outputs = [
        torch.empty_like(inputs[0]),
        torch.empty_like(inputs[1]),
        torch.empty_like(inputs[10]),
    ]
    arguments = inputs + outputs + [meta]
    grid = int(torch.npu.get_device_properties(0).vector_core_num)
    probe = AttentionSequenceRouteProbe(config, cmp_len, grid)
    specs = [dsl.TensorSpec(tuple(v.shape), dsl.dtypes.int32) for v in arguments]
    compiled = dsl.compile(probe.run, *specs)
    for repeat in range(3):
        if repeat == 1:
            cpu["block_table"] = wt.roll(1, 1)
            cpu["extra_block_table"] = ct.roll(1, 1)
            cpu["cache_seqlens"] = torch.tensor(
                [1, 0, capacity, 128], dtype=torch.int32
            )
            cpu["extra_cache_seqlens"] = torch.tensor(
                [capacity, 0, 1, capacity], dtype=torch.int32
            )
        if repeat == 2:
            cpu["seqused_q"] = torch.zeros(4, dtype=torch.int32)
        for index, name in (
            (4, "block_table"),
            (5, "extra_block_table"),
            (7, "seqused_q"),
            (8, "cache_seqlens"),
            (9, "extra_cache_seqlens"),
        ):
            inputs[index].copy_(cpu[name].npu())
        expected_meta = query_metadata_reference(tokens, cu, cpu["seqused_q"])
        if mode == "reuse":
            meta.copy_(expected_meta.npu())
        for output in outputs:
            output.fill_(12345)
        compiled(*arguments)
        torch.npu.synchronize()
        expected = filter_sequences(wi, ci, slots, 128, 4, 84, **cpu)
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
        if mode != "none":
            torch.testing.assert_close(meta.cpu(), expected_meta, rtol=0, atol=0)
        torch.testing.assert_close(inputs[0].cpu(), wi, rtol=0, atol=0)
        torch.testing.assert_close(inputs[1].cpu(), ci, rtol=0, atol=0)
    print("GENERAL_PAGE_ROUTE_PASS", columns, mode, tokens, cmp_len, flush=True)


@pytest.mark.npu
def test_general_page_table_int32_capacity_device():
    """A 2**31 exclusive capacity must still accept INT_MAX as an index."""
    dsl = pytest.importorskip("cannbotdsl")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    AttentionSequenceRouteProbe = _device_probes().AttentionSequenceRouteProbe
    torch.npu.set_device(0)
    columns = 1 << 24
    table = torch.zeros((1, columns), dtype=torch.int32)
    table[0, -1] = 3
    wi = torch.full((1, 128), 2147483647, dtype=torch.int32)
    wi[0, 0], wi[0, 1], wi[0, 2] = 2147483520, -1, -7
    ci = wi.clone()
    topk = torch.tensor([127], dtype=torch.int32)
    cu = torch.tensor([0, 1], dtype=torch.int32)
    used = torch.ones(1, dtype=torch.int32)
    slot = torch.tensor([128], dtype=torch.int32)
    idx, table_npu, topk_npu, cu_npu, used_npu, slot_npu = [
        v.npu() for v in (wi, table, topk, cu, used, slot)
    ]
    cfg = validate_attention_sequences(
        1,
        idx.device,
        cu_seqlens_qo=cu_npu,
        seqused_q=used_npu,
        block_table=table_npu,
        extra_block_table=table_npu,
        topk_length=topk_npu,
        extra_topk_length=topk_npu,
    )
    outputs = [torch.empty_like(idx), torch.empty_like(idx), torch.empty_like(slot_npu)]
    args = [
        idx,
        idx,
        topk_npu,
        topk_npu,
        table_npu,
        table_npu,
        cu_npu,
        used_npu,
        used_npu,
        used_npu,
        slot_npu,
        *outputs,
        torch.zeros((1, 8), dtype=torch.int32, device="npu"),
    ]
    probe = AttentionSequenceRouteProbe(
        cfg, 128, int(torch.npu.get_device_properties(0).vector_core_num)
    )
    fn = dsl.compile(
        probe.run, *[dsl.TensorSpec(tuple(v.shape), dsl.dtypes.int32) for v in args]
    )
    for page in (3, -1, 2):
        table[0, -1] = page
        table_npu[0, -1] = page
        for out in outputs:
            out.fill_(54321)
        fn(*args)
        torch.npu.synchronize()
        expected = filter_sequences(
            wi,
            ci,
            slot,
            128,
            4,
            84,
            cu_seqlens_qo=cu,
            seqused_q=used,
            block_table=table,
            extra_block_table=table,
            topk_length=topk,
            extra_topk_length=topk,
        )
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
    print("GENERAL_PAGE_INT32_CAPACITY_PASS", columns, flush=True)


# ============================================================================
# Full mega AOT and NPU regression cases
# ============================================================================


def _dsl():
    return pytest.importorskip("cannbotdsl", reason="CANNBotDSL wheel unavailable")


def _compile_mega(heads, tokens, grid, strategy="predecode", nc=5, length_stride=0):
    if shutil.which("bisheng") is None:
        pytest.skip("CANN bisheng toolchain unavailable; no AOT result")
    dsl = _dsl()
    from mega_mla import (
        MegaMlaKernel,
        MegaMlaOp,
    )

    dt, spec = dsl.dtypes, dsl.TensorSpec
    t, h, k, ps = tokens, heads, 640, 128
    from attention_workspace import AttentionWorkspaceLayout

    layout = AttentionWorkspaceLayout(t, h, k, nc, ps, strategy)
    win_lengths = (t // length_stride,) if length_stride else (t, 128)
    cmp_lengths = (t // length_stride,) if length_stride else (t, 512)
    shapes = [
        ((t, 5120), dt.bfloat16),
        ((1, 1280), dt.float32),
        ((1, 512), dt.float32),
        ((1280, 5120), dt.float8_e4m3fn),
        ((512, 5120), dt.float8_e4m3fn),
        ((h * 512, 1280), dt.float8_e4m3fn),
        ((1280, 80, 2), dt.float8_e8m0),
        ((512, 80, 2), dt.float8_e8m0),
        ((h * 512, 20, 2), dt.float8_e8m0),
        ((2, ps, 1, 544), dt.uint8),
        ((nc, ps, 1, 320), dt.uint8),
        ((t, 128), dt.int32),
        ((t, 512), dt.int32),
        ((1, h), dt.float32),
        ((t, 32), dt.float32),
        ((t, 32), dt.float32),
        ((t,), dt.int32),
        ((t, 128), dt.int32),
        ((t, 512), dt.int32),
        (win_lengths, dt.int32),
        (cmp_lengths, dt.int32),
        ((t, h, 512), dt.bfloat16),
        ((layout.nbytes // 4,), dt.float32),
        ((layout.nbytes // 2,), dt.bfloat16),
        ((layout.nbytes,), dt.float8_e4m3fn),
        ((layout.nbytes,), dt.float8_e8m0),
        ((layout.nbytes // 4,), dt.int32),
    ]
    op = MegaMlaOp(grid=grid, heads=heads, cmp_decode_strategy=strategy)
    op.kernel = MegaMlaKernel(
        k,
        ps,
        128,
        tokens=t,
        grid=grid,
        heads=h,
        win_length_stride=length_stride,
        cmp_length_stride=length_stride,
        cmp_decode_strategy=strategy,
        workspace_layout=layout,
    )
    specs = [
        spec(shape, dtype, storage_format="nz" if i in (3, 4, 5) else "nd")
        for i, (shape, dtype) in enumerate(shapes)
    ]
    fn = dsl.compile(op._launch_mega_mla, *specs)
    try:
        assert fn.so_path and Path(fn.so_path).is_file()
    finally:
        fn.close()


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize("heads", [8, 16, 64])
@pytest.mark.parametrize(
    "tokens,grid", [(1, 28), (8, 28), (8, 32), (16, 28), (16, 32), (64, 32), (100, 28)]
)
def test_full_mega_head_specialization_codegen(heads, tokens, grid):
    _compile_mega(heads, tokens, grid)


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize(
    "heads,tokens,strategy",
    [
        (8, 6, "indexed"),
        (16, 6, "indexed"),
        (16, 1, "predecode"),
        (8, 11, "indexed"),
        (16, 12, "predecode"),
    ],
)
def test_small_token_grid32_codegen(heads, tokens, strategy):
    _compile_mega(heads, tokens, 32, strategy=strategy, nc=42)


@pytest.mark.npu
@pytest.mark.parametrize("heads", [8, 16, 64])
def test_logical_m_qk_pv_device(heads):
    _dsl()
    pytest.importorskip("torch_npu", reason="torch_npu unavailable")
    import torch

    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    torch.npu.set_device(0)
    from attention_matmul import ProbabilityValueMatmulOp, QueryKeyMatmulOp

    torch.manual_seed(23)
    q = (torch.randn(heads, 512) * 0.1).to(torch.bfloat16)
    kv = (torch.randn(640, 512) * 0.1).to(torch.bfloat16)
    scores = torch.empty(heads, 640, dtype=torch.float32).npu()
    QueryKeyMatmulOp(heads, 640, 512).run(scores, q.npu(), kv.npu())
    torch.npu.synchronize()
    # BF16 operands, FP32 L0C/output. No fake heads or M16 padded operands.
    torch.testing.assert_close(
        scores.cpu(), q.float() @ kv.float().T, rtol=1e-3, atol=1e-3
    )
    p = torch.softmax(torch.randn(heads, 640), -1).to(torch.bfloat16)
    out = torch.empty(heads, 512, dtype=torch.bfloat16).npu()
    ProbabilityValueMatmulOp(heads, 512, 640, base_n=64).run(out, p.npu(), kv.npu())
    torch.npu.synchronize()
    expected = (p.float() @ kv.float()).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-2, atol=1e-3)


def _exercise_attention(
    heads,
    tokens,
    use_lengths,
    monkeypatch,
    cmp_pages=5,
    k_attn=640,
    cmp_strategy="auto",
    reuse_op=None,
    return_result=False,
    win_pages=2,
    slot_override=None,
    update_kv_gamma=False,
    cache_batch=None,
    sequence_case=False,
    sequence_cmp_table_columns=4,
    sequence_high_pages=False,
    compressed_page_size=128,
    window_page_padding=0,
    compressed_page_padding=0,
    cache_storage_offset=0,
    switch_cache_layout=False,
    grid=None,
    poison_zero=False,
):
    """K640: independent FP32 projection math, including H64/T64 CV path."""
    _dsl()
    torch_npu = pytest.importorskip("torch_npu", reason="torch_npu unavailable")
    import torch

    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    torch.npu.set_device(0)
    # Retain generated ASC separately for every specialization and test case.
    import os

    dump_root = os.environ.get("CANNBOTDSL_DUMP_DIR")
    if dump_root:
        dump_dir = Path(dump_root) / f"h{heads}_t{tokens}"
        dump_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("CANNBOTDSL_DUMP_DIR", str(dump_dir))
    from mega_mla import MegaMlaOp

    torch.manual_seed(31)
    torch.set_num_threads(4)
    physical_cores = int(torch.npu.get_device_properties(0).cube_core_num)
    grid = physical_cores if grid is None else grid
    if grid > physical_cores:
        pytest.skip(f"grid={grid} exceeds device AIC count {physical_cores}")
    t, d, k, ps = tokens, 512, k_attn, 128
    cps = compressed_page_size
    x = (torch.randn(t, 5120) * 0.1).to(torch.bfloat16)
    gfq, gfk = torch.rand(1, 1280) + 0.5, torch.rand(1, 512) + 0.5
    weights = [
        (torch.randn(n, width) * 0.05).to(torch.float8_e4m3fn)
        for n, width in ((1280, 5120), (512, 5120), (heads * d, 1280))
    ]
    scales = [
        torch.full((n, width // 64, 2), 125, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        for n, width in ((1280, 5120), (512, 5120), (heads * d, 1280))
    ]
    xq = _mx_round(x)
    qr, kv = [xq @ w.float().T * 0.25 for w in weights[:2]]
    qnorm = _rms_reference(qr, gfq)
    kvnorm = _rms_reference(kv, gfk).to(torch.bfloat16)
    qfull = (_mx_round(qnorm) @ weights[2].float().T * 0.25).to(torch.bfloat16)
    theta = torch.randn(t, 32)
    cos, sin = theta.cos(), theta.sin()
    qb = (
        _rope_cpu(qfull.reshape(t, heads, d), cos[:, None], sin[:, None]).float()
        / d**0.5
    ).to(torch.bfloat16)
    rotated_kv = _rope_cpu(kvnorm, cos, sin)
    win_cpu = quantize_fp8((torch.randn(win_pages, ps, 1, d) * 0.1).to(torch.bfloat16))
    cmp_cpu = quantize_fp4((torch.randn(cmp_pages, cps, 1, d) * 0.1).to(torch.bfloat16))
    indices_in_kvcache = torch.randint(ps, win_pages * ps, (t, 128), dtype=torch.int32)
    extra_indices_in_kvcache = torch.randint(
        0, cmp_pages * cps, (t, k - 128), dtype=torch.int32
    )
    indices_in_kvcache[:, ::7] = -1
    extra_indices_in_kvcache[:, ::11] = -1
    extra_indices_in_kvcache[:, :8] = torch.tensor(
        [0, cps - 1, cps, cmp_pages * cps - 1, cps, -1, -7, 0], dtype=torch.int32
    )
    slot = (
        torch.arange(ps, ps + t, dtype=torch.int32)
        if slot_override is None
        else slot_override.clone()
    )
    call_tables = {}
    input_win_idx, input_cmp_idx = indices_in_kvcache, extra_indices_in_kvcache
    if cache_batch is not None:
        assert t % cache_batch == 0 and win_pages > 2 * cache_batch
        assert cmp_pages > 2 * cache_batch
        query_count = t // cache_batch
        win_columns = (win_pages - 1) // cache_batch
        cmp_columns = (cmp_pages - 1) // cache_batch
        # 非连续、请求间互不重叠的页表，不能靠恒等映射或固定batch=0过关。
        win_table = (
            (torch.randperm(win_pages - 1)[: cache_batch * win_columns] + 1)
            .to(torch.int32)
            .reshape(cache_batch, win_columns)
        )
        cmp_table = (
            (torch.randperm(cmp_pages - 1)[: cache_batch * cmp_columns] + 1)
            .to(torch.int32)
            .reshape(cache_batch, cmp_columns)
        )
        input_win_idx = torch.randint(0, win_columns * ps, (t, 128), dtype=torch.int32)
        input_cmp_idx = torch.randint(
            0, cmp_columns * cps, (t, k - 128), dtype=torch.int32
        )
        for logical, columns, route_ps in (
            (input_win_idx, win_columns, ps),
            (input_cmp_idx, cmp_columns, cps),
        ):
            logical[:, ::7] = -1
            logical[:, :8] = torch.tensor(
                [
                    -1,
                    -7,
                    0,
                    route_ps - 1,
                    route_ps,
                    columns * route_ps - 1,
                    route_ps,
                    0,
                ],
                dtype=torch.int32,
            )
        # 独立CPU参考逐请求查表；设备只接收未映射的逻辑索引。
        batch_ids = torch.arange(t) // query_count

        def physical(logical, table, route_ps=ps):
            valid = logical >= 0
            safe = logical.clamp_min(0).long()
            pages = table[batch_ids[:, None], safe // route_ps].long()
            return torch.where(valid, pages * route_ps + safe % route_ps, logical).to(
                torch.int32
            )

        indices_in_kvcache = physical(input_win_idx, win_table)
        extra_indices_in_kvcache = physical(input_cmp_idx, cmp_table, cps)
        slot = (win_table[batch_ids, 0] * ps + torch.arange(t) % query_count).to(
            torch.int32
        )
        call_tables = {
            "block_table": win_table.npu(),
            "extra_block_table": cmp_table.npu(),
        }
    input_slot = slot
    if sequence_case:
        assert not use_lengths and cache_batch is None and t >= 6
        # Unequal packed requests, an empty request, inactive Q rows and mixed
        # batch/query TopK lengths. Poisoned indices must never reach page loads.
        seq_cpu = dict(
            cu_seqlens_qo=torch.tensor([0, 2, 2, 5, t], dtype=torch.int32),
            seqused_q=torch.tensor([1, 0, 2, t - 6], dtype=torch.int32),
            cache_seqlens=torch.tensor([129, 0, 256, 1], dtype=torch.int32),
            extra_cache_seqlens=torch.tensor([257, 0, 512, 127], dtype=torch.int32),
            block_table=(torch.randperm(win_pages - 1)[:8] + 1)
            .to(torch.int32)
            .reshape(4, 2),
            extra_block_table=(torch.randperm(cmp_pages - 1)[:16] + 1)
            .to(torch.int32)
            .reshape(4, 4),
            topk_length=torch.tensor([64, 0, 128, 3], dtype=torch.int32),
            extra_topk_length=(torch.arange(t, dtype=torch.int32) * 73).remainder(
                k - 128 + 1
            ),
        )
        # Exercise negative and out-of-pool page IDs, including in-place rolls.
        seq_cpu["block_table"][2, 1] = -1
        seq_cpu["extra_block_table"][2, 2] = cmp_pages
        seq_cpu["extra_block_table"][2, 3] = -1
        if sequence_cmp_table_columns > 4:
            seq_cpu["extra_block_table"] = torch.cat(
                (
                    seq_cpu["extra_block_table"],
                    torch.full(
                        (4, sequence_cmp_table_columns - 4), -1, dtype=torch.int32
                    ),
                ),
                dim=1,
            )
        if sequence_case == "fixed":
            seq_cpu.pop("cu_seqlens_qo")
        input_win_idx = torch.randint(0, 256, (t, 128), dtype=torch.int32)
        input_cmp_idx = torch.randint(0, 512, (t, k - 128), dtype=torch.int32)
        for idx in (input_win_idx, input_cmp_idx):
            idx[:, :8] = torch.tensor(
                [0, 127, 128, 255, 256, -1, -7, 2**30], dtype=torch.int32
            )
        ids = (
            torch.arange(t) // (t // 4)
            if sequence_case == "fixed"
            else torch.tensor([0, 0, 2, 2, 2] + [3] * (t - 5))
        )
        if sequence_high_pages:
            columns = sequence_cmp_table_columns
            seq_cpu["extra_block_table"] = torch.randint(
                1, cmp_pages, (4, columns), dtype=torch.int32
            )
            seq_cpu["extra_block_table"][2, -1] = cmp_pages
            seq_cpu["extra_cache_seqlens"] = torch.tensor(
                [columns * cps, 0, columns * cps - 7, columns * cps], dtype=torch.int32
            )
            input_cmp_idx = torch.randint(
                columns * cps, (t, k - 128), dtype=torch.int32
            )
            edges = torch.tensor(
                [
                    0,
                    127,
                    63 * cps,
                    64 * cps,
                    83 * cps,
                    127 * cps,
                    8191 * cps,
                    8192 * cps,
                    columns * cps - 1,
                    columns * cps,
                    -1,
                    2**30,
                ],
                dtype=torch.int32,
            )
            input_cmp_idx[:, : len(edges)] = edges
        input_slot = seq_cpu["block_table"][ids, 0] * ps + torch.arange(
            t, dtype=torch.int32
        )
        indices_in_kvcache, extra_indices_in_kvcache, slot = filter_sequences(
            input_win_idx,
            input_cmp_idx,
            input_slot,
            ps,
            win_pages,
            cmp_pages,
            compressed_page_size=cps,
            **seq_cpu,
        )
        call_tables = {name: value.npu() for name, value in seq_cpu.items()}
    sink = torch.randn(1, heads)
    golden_win = win_cpu.clone()
    valid_slot = slot >= ps
    encoded_kv = quantize_fp8(rotated_kv.reshape(1, t, 1, d)).reshape(t, 544)
    golden_win.view(-1, 544)[slot[valid_slot].long()] = encoded_kv[valid_slot]
    wd = dequantize_fp8(golden_win).reshape(-1, d)
    cd = dequantize_fp4(cmp_cpu).reshape(-1, d)
    gathered = torch.cat(
        (
            wd[indices_in_kvcache.clamp_min(0).long()],
            cd[extra_indices_in_kvcache.clamp_min(0).long()],
        ),
        1,
    )
    invalid_base = torch.cat((indices_in_kvcache < 0, extra_indices_in_kvcache < 0), 1)
    raw_gathered = gathered.clone()
    invalid = invalid_base.clone()
    call_lengths = {}
    if use_lengths:
        batch = t if use_lengths == "token" else 2
        if use_lengths == "token":
            win_lengths = torch.tensor([0, 1, 63, 128], dtype=torch.int32).repeat(
                (t + 3) // 4
            )[:t]
            cmp_lengths = torch.tensor([0, 1, 257, k - 128], dtype=torch.int32).repeat(
                (t + 3) // 4
            )[:t]
        else:
            win_lengths = torch.tensor([0, 64], dtype=torch.int32)
            cmp_lengths = torch.tensor([0, 257], dtype=torch.int32)
        win_limit = win_lengths.repeat_interleave(t // batch)
        cmp_limit = cmp_lengths.repeat_interleave(t // batch)
        invalid[:, :128] |= torch.arange(128)[None, :] >= win_limit[:, None]
        invalid[:, 128:] |= torch.arange(k - 128)[None, :] >= cmp_limit[:, None]
        call_lengths = {
            "topk_length": win_lengths.npu(),
            "extra_topk_length": cmp_lengths.npu(),
        }
    gathered[invalid] = 0
    expected = _attention_reference(qb, gathered, invalid, sink)
    print("HEAD_DEVICE_STAGE", heads, tokens, use_lengths, "pack_weights", flush=True)
    wa, wkv, wb = [
        torch_npu.npu_format_cast(w.npu(), torch_npu.Format.FRACTAL_NZ) for w in weights
    ]
    if window_page_padding or compressed_page_padding or cache_storage_offset:
        win, win_backing = _native_cache_tensor(
            win_cpu, window_page_padding, cache_storage_offset, "npu:0"
        )
        cmp, cmp_backing = _native_cache_tensor(
            cmp_cpu, compressed_page_padding, cache_storage_offset, "npu:0"
        )
    else:
        win, cmp = win_cpu.npu(), cmp_cpu.npu()
        win_backing = cmp_backing = None
    op = (
        reuse_op
        if reuse_op is not None
        else MegaMlaOp(grid=grid, heads=heads, cmp_decode_strategy=cmp_strategy)
    )
    inputs = [
        x.npu(),
        gfq.npu(),
        gfk.npu(),
        wa,
        wkv,
        wb,
        *(s.npu() for s in scales),
        win,
        cmp,
        input_win_idx.npu(),
        input_cmp_idx.npu(),
        sink.npu(),
        cos.npu(),
        sin.npu(),
        input_slot.npu(),
    ]
    print("HEAD_DEVICE_STAGE", heads, tokens, use_lengths, "first_run", flush=True)
    if poison_zero:
        public.prepare_attention_arguments(op, *inputs, **call_lengths, **call_tables)
        op._workspace["zero"].fill_(float("nan"))
    out = op.run(*inputs, **call_lengths, **call_tables)
    if poison_zero:
        assert torch.count_nonzero(op._workspace["zero"]) == 0
    torch.npu.synchronize()
    assert out.shape == (t, heads, d) and torch.isfinite(out).all()
    # Preserve a first-call snapshot, including future explicit out-buffer reuse.
    first = out.cpu().clone()
    # Scatter byte exactness also checks independent KV norm/RoPE/quantization.
    torch.testing.assert_close(win.cpu(), golden_win, rtol=0, atol=0)
    torch.testing.assert_close(op._workspace["kg"].cpu(), gathered, rtol=0, atol=0)
    torch.testing.assert_close(
        op._workspace["qfull"].cpu(), qfull, rtol=0.01, atol=0.002
    )
    torch.testing.assert_close(first, expected, rtol=0.01, atol=0.002)
    print("HEAD_DEVICE_STAGE", heads, tokens, use_lengths, "repeat_run", flush=True)
    again = op.run(*inputs, **call_lengths, **call_tables)
    torch.npu.synchronize()
    assert torch.equal(first, again.cpu()), "repeat output is not stable"
    if sequence_case:
        torch.testing.assert_close(
            op._workspace["seq_win_idx"].cpu(), indices_in_kvcache, rtol=0, atol=0
        )
        torch.testing.assert_close(
            op._workspace["seq_cmp_idx"].cpu(), extra_indices_in_kvcache, rtol=0, atol=0
        )
        torch.testing.assert_close(
            op._workspace["seq_slot"].cpu(), slot, rtol=0, atol=0
        )
        # Same addresses and compiled callable: changing device lengths to zero
        # must zero all attention outputs and leave every cache byte untouched.
        before = win.cpu().clone()
        call_tables["seqused_q"].zero_()
        empty = op.run(*inputs, **call_tables)
        torch.npu.synchronize()
        assert torch.count_nonzero(empty.cpu()) == 0
        assert torch.count_nonzero(op._workspace["kg"].cpu()) == 0
        torch.testing.assert_close(win.cpu(), before, rtol=0, atol=0)
        call_tables["seqused_q"].copy_(seq_cpu["seqused_q"].npu())
        restored = op.run(*inputs, **call_tables)
        torch.npu.synchronize()
        assert torch.equal(first, restored.cpu())
        # Prefix sums, both used-KV vectors and page tables also change in place.
        changed = {name: value.clone() for name, value in seq_cpu.items()}
        if "cu_seqlens_qo" in changed:
            changed["cu_seqlens_qo"] = torch.tensor([0, 1, 1, 4, t], dtype=torch.int32)
            changed["seqused_q"] = torch.tensor([1, 0, 2, t - 5], dtype=torch.int32)
        else:
            changed["seqused_q"] = torch.tensor([0, 1, 1, t // 4], dtype=torch.int32)
        changed["cache_seqlens"] = torch.tensor([1, 0, 128, 256], dtype=torch.int32)
        changed["extra_cache_seqlens"] = torch.tensor(
            [128, 0, 1, 512], dtype=torch.int32
        )
        for name in ("block_table", "extra_block_table"):
            changed[name] = changed[name].roll(1, dims=1)
        for name, value in changed.items():
            call_tables[name].copy_(value.npu())
        nw, nc, ns = filter_sequences(
            input_win_idx,
            input_cmp_idx,
            input_slot,
            ps,
            win_pages,
            cmp_pages,
            compressed_page_size=cps,
            **changed,
        )
        new_win = win_cpu.clone()
        selected_slots = ns >= ps
        new_win.view(-1, 544)[ns[selected_slots].long()] = encoded_kv[selected_slots]
        win.copy_(win_cpu.npu())
        new_gathered = torch.cat(
            (
                dequantize_fp8(new_win).reshape(-1, d)[nw.clamp_min(0).long()],
                cd[nc.clamp_min(0).long()],
            ),
            1,
        )
        new_invalid = torch.cat((nw < 0, nc < 0), 1)
        new_gathered[new_invalid] = 0
        changed_out = op.run(*inputs, **call_tables)
        torch.npu.synchronize()
        torch.testing.assert_close(win.cpu(), new_win, rtol=0, atol=0)
        torch.testing.assert_close(
            op._workspace["kg"].cpu(), new_gathered, rtol=0, atol=0
        )
        torch.testing.assert_close(
            changed_out.cpu(),
            _attention_reference(qb, new_gathered, new_invalid, sink),
            rtol=0.01,
            atol=0.002,
        )
        for name, value in seq_cpu.items():
            call_tables[name].copy_(value.npu())
        win.copy_(golden_win.npu())
        restored_again = op.run(*inputs, **call_tables)
        torch.npu.synchronize()
        assert torch.equal(first, restored_again.cpu())
        torch.testing.assert_close(inputs[16].cpu(), input_slot, rtol=0, atol=0)
        for name, original in seq_cpu.items():
            torch.testing.assert_close(
                call_tables[name].cpu(), original, rtol=0, atol=0
            )
    if use_lengths:
        # Alternate ABI/length specializations on the same cached operator.
        full_gathered = raw_gathered.clone()
        full_gathered[invalid_base] = 0
        print(
            "HEAD_DEVICE_STAGE", heads, tokens, use_lengths, "unmasked_run", flush=True
        )
        unmasked = op.run(*inputs, **call_tables)
        torch.npu.synchronize()
        full_expected = _attention_reference(qb, full_gathered, invalid_base, sink)
        torch.testing.assert_close(unmasked.cpu(), full_expected, rtol=0.01, atol=0.002)
        print(
            "HEAD_DEVICE_STAGE", heads, tokens, use_lengths, "masked_again", flush=True
        )
        masked_again = op.run(*inputs, **call_lengths, **call_tables)
        torch.npu.synchronize()
        assert torch.equal(first, masked_again.cpu()), (
            "cached length specialization changed"
        )
    torch.testing.assert_close(inputs[11].cpu(), input_win_idx, rtol=0, atol=0)
    torch.testing.assert_close(inputs[12].cpu(), input_cmp_idx, rtol=0, atol=0)
    if cache_batch is not None:
        # 同一实例切换逻辑/物理接口，确保编译缓存包含地址模式和batch步长。
        physical_inputs = list(inputs)
        physical_inputs[11], physical_inputs[12] = (
            indices_in_kvcache.npu(),
            extra_indices_in_kvcache.npu(),
        )
        physical_out = op.run(*physical_inputs, **call_lengths)
        torch.npu.synchronize()
        assert torch.equal(physical_out.cpu(), first)
        logical_again = op.run(*inputs, **call_lengths, **call_tables)
        torch.npu.synchronize()
        assert torch.equal(logical_again.cpu(), first)
        for name, original in (
            ("block_table", win_table),
            ("extra_block_table", cmp_table),
        ):
            torch.testing.assert_close(
                call_tables[name].cpu(), original, rtol=0, atol=0
            )
        # 页表在同一地址原地变化，必须被下一次kernel调用重新读取。
        changed_win_table = win_table.roll(1, dims=1)
        changed_cmp_table = cmp_table.roll(1, dims=1)
        changed_win_idx = physical(input_win_idx, changed_win_table)
        changed_cmp_idx = physical(input_cmp_idx, changed_cmp_table, cps)
        changed_gathered = torch.cat(
            (
                wd[changed_win_idx.clamp_min(0).long()],
                cd[changed_cmp_idx.clamp_min(0).long()],
            ),
            1,
        )
        changed_gathered[invalid] = 0
        call_tables["block_table"].copy_(changed_win_table.npu())
        call_tables["extra_block_table"].copy_(changed_cmp_table.npu())
        changed_out = op.run(*inputs, **call_lengths, **call_tables)
        torch.npu.synchronize()
        torch.testing.assert_close(
            op._workspace["kg"].cpu(), changed_gathered, rtol=0, atol=0
        )
        torch.testing.assert_close(
            changed_out.cpu(),
            _attention_reference(qb, changed_gathered, invalid, sink),
            rtol=0.01,
            atol=0.002,
        )
        call_tables["block_table"].copy_(win_table.npu())
        call_tables["extra_block_table"].copy_(cmp_table.npu())
        if use_lengths:
            # 有效前缀之外放入超出页表范围的正索引，确认先mask再查表。
            bad_win, bad_cmp = input_win_idx.clone(), input_cmp_idx.clone()
            bad_win[torch.arange(128)[None, :] >= win_limit[:, None]] = 2**29
            bad_cmp[torch.arange(k - 128)[None, :] >= cmp_limit[:, None]] = 2**29
            inputs[11].copy_(bad_win.npu())
            inputs[12].copy_(bad_cmp.npu())
            masked_out = op.run(*inputs, **call_lengths, **call_tables)
            torch.npu.synchronize()
            assert torch.equal(masked_out.cpu(), first)
            inputs[11].copy_(input_win_idx.npu())
            inputs[12].copy_(input_cmp_idx.npu())
        print(
            "PAGED_BATCH_PASS",
            "B",
            cache_batch,
            "S",
            query_count,
            "table_mutation",
            True,
            "physical_pair_exact",
            True,
            flush=True,
        )
    error = (first.float() - expected.float()).abs()
    print(
        "HEAD_DEVICE_PASS",
        heads,
        "T",
        t,
        "K",
        k,
        "grid",
        grid,
        "max_abs",
        error.max().item(),
        "mean_abs",
        error.mean().item(),
        flush=True,
    )
    torch.testing.assert_close(cmp.cpu(), cmp_cpu, rtol=0, atol=0)
    if update_kv_gamma:
        # Reuse the same cache pointers and compiled callable with changed
        # data; this catches stale reads after the DMA writer replaces bypass
        # scalar stores. The update needs no new shape specialization.
        changed_gamma = gfk * 1.25
        changed_norm = _rms_reference(kv, changed_gamma).to(torch.bfloat16)
        changed_rows = quantize_fp8(
            _rope_cpu(changed_norm, cos, sin).reshape(1, t, 1, d)
        ).reshape(t, 544)
        assert not torch.equal(changed_rows[valid_slot], encoded_kv[valid_slot])
        changed_win = golden_win.clone()
        changed_win.view(-1, 544)[slot[valid_slot].long()] = changed_rows[valid_slot]
        changed_wd = dequantize_fp8(changed_win).reshape(-1, d)
        changed_gathered = torch.cat(
            (
                changed_wd[indices_in_kvcache.clamp_min(0).long()],
                cd[extra_indices_in_kvcache.clamp_min(0).long()],
            ),
            1,
        )
        changed_gathered[invalid] = 0
        print("HEAD_DEVICE_STAGE", heads, tokens, "changed_kv_gamma", flush=True)
        inputs[2].copy_(changed_gamma.npu())
        # The attention path must leave this legacy GM diagnostic slot alone:
        # normalized KV is now handed to RoPE/quantization entirely in UB.
        legacy_norm = op._workspace["kvn"]
        legacy_norm.fill_(123.0)
        changed_out = op.run(*inputs, **call_lengths, **call_tables)
        torch.npu.synchronize()
        assert op._workspace["kvn"].data_ptr() == legacy_norm.data_ptr()
        assert torch.all(legacy_norm.cpu() == 123.0), "KV norm unexpectedly wrote GM"
        torch.testing.assert_close(win.cpu(), changed_win, rtol=0, atol=0)
        torch.testing.assert_close(
            op._workspace["kg"].cpu(), changed_gathered, rtol=0, atol=0
        )
        torch.testing.assert_close(
            changed_out.cpu(),
            _attention_reference(qb, changed_gathered, invalid, sink),
            rtol=0.01,
            atol=0.002,
        )
        print("KV_CACHE_CHANGED_INPUT_PASS", heads, tokens, flush=True)
        print("KV_CACHE_UB_HANDOFF_PASS", heads, tokens, flush=True)
    if win_backing is not None:
        _assert_cache_padding(win, win_backing)
        _assert_cache_padding(cmp, cmp_backing)
    if switch_cache_layout:
        other_win, other_win_backing = _native_cache_tensor(
            golden_win, window_page_padding + 64, cache_storage_offset, "npu:0"
        )
        other_cmp, other_cmp_backing = _native_cache_tensor(
            cmp_cpu, compressed_page_padding + 96, cache_storage_offset, "npu:0"
        )
        switched_inputs = list(inputs)
        switched_inputs[9], switched_inputs[10] = other_win, other_cmp
        programs = len(op._compiled_launches)
        switched = op.run(*switched_inputs, **call_lengths, **call_tables)
        torch.npu.synchronize()
        assert len(op._compiled_launches) == programs + 1
        assert torch.equal(switched.cpu(), first)
        torch.testing.assert_close(other_win.cpu(), golden_win, rtol=0, atol=0)
        _assert_cache_padding(other_win, other_win_backing)
        _assert_cache_padding(other_cmp, other_cmp_backing)
        restored = op.run(*inputs, **call_lengths, **call_tables)
        torch.npu.synchronize()
        assert torch.equal(restored.cpu(), first)
        assert len(op._compiled_launches) == programs + 1
    if return_result:
        return dict(
            op=op,
            output=first,
            gathered=op._workspace["kg"].cpu(),
            max_abs=error.max().item(),
            mean_abs=error.mean().item(),
            inputs=inputs,
            call_tables=call_tables,
            seq_cpu=seq_cpu if sequence_case else None,
            qb=qb,
            attn_sink=sink,
        )


@pytest.mark.npu
@pytest.mark.parametrize(
    "heads,strategy,table_columns",
    [
        (64, "indexed", 4),
        (8, "indexed", 4),
        (16, "indexed", 4),
        (64, "predecode", 4),
        pytest.param(8, "indexed", 84, id="h8-wide84"),
        pytest.param(16, "indexed", 8193, id="h16-wide8193"),
    ],
)
def test_cross_layer_query_metadata(heads, strategy, table_columns, monkeypatch):
    from mega_mla import mega_mla

    case = _exercise_attention(
        heads,
        8,
        False,
        monkeypatch,
        return_result=True,
        win_pages=9,
        cmp_pages=84 if table_columns > 4 else 17,
        sequence_case=True,
        cmp_strategy=strategy,
        sequence_cmp_table_columns=table_columns,
        sequence_high_pages=table_columns > 4,
    )
    op, inputs, raw = case["op"], case["inputs"], case["call_tables"]
    options = dict(grid=op.grid, cmp_decode_strategy=strategy)
    out, metadata = mega_mla(*inputs, **raw, **options, return_sequence_metadata=True)
    torch.npu.synchronize()
    assert torch.equal(out.cpu(), case["output"])
    expected_meta = query_metadata_reference(
        8, case["seq_cpu"]["cu_seqlens_qo"], case["seq_cpu"]["seqused_q"]
    )
    torch.testing.assert_close(metadata.query_info.cpu(), expected_meta, rtol=0, atol=0)
    original_metadata = metadata.query_info.cpu().clone()
    layer = {name: value.clone() for name, value in case["seq_cpu"].items()}
    for name in ("block_table", "extra_block_table"):
        layer[name] = layer[name].roll(1, 1)
    layer["cache_seqlens"] = torch.tensor([1, 0, 128, 256], dtype=torch.int32)
    layer["extra_cache_seqlens"] = torch.tensor(
        [128, 0, 1, 512]
        if table_columns == 4
        else [table_columns * 128, 0, table_columns * 128 - 7, table_columns * 128],
        dtype=torch.int32,
    )
    inputs[11].copy_(inputs[11].roll(1, 1))
    inputs[12].copy_(inputs[12].roll(1, 1))
    next_device = {name: value.npu() for name, value in layer.items()}
    ordinary = op.run(*inputs, **next_device).cpu().clone()
    before = inputs[9].cpu().clone()
    expected_kg = op._workspace["kg"].cpu().clone()
    later = {
        name: value
        for name, value in next_device.items()
        if name not in ("cu_seqlens_qo", "seqused_q")
    }
    reused, same_metadata = mega_mla(
        *inputs,
        **later,
        **options,
        sequence_metadata=metadata,
        return_sequence_metadata=True,
    )
    torch.npu.synchronize()
    assert same_metadata is metadata and torch.equal(ordinary, reused.cpu())
    torch.testing.assert_close(inputs[9].cpu(), before, rtol=0, atol=0)
    pw, pc, _ = filter_sequences(
        inputs[11].cpu(),
        inputs[12].cpu(),
        inputs[16].cpu(),
        128,
        9,
        84 if table_columns > 4 else 17,
        **layer,
    )
    wd = dequantize_fp8(inputs[9].cpu()).reshape(-1, 512)
    cd = dequantize_fp4(inputs[10].cpu()).reshape(-1, 512)
    gathered = torch.cat((wd[pw.clamp_min(0).long()], cd[pc.clamp_min(0).long()]), 1)
    invalid = torch.cat((pw < 0, pc < 0), 1)
    gathered[invalid] = 0
    torch.testing.assert_close(expected_kg, gathered, rtol=0, atol=0)
    golden = _attention_reference(case["qb"], gathered, invalid, case["attn_sink"])
    torch.testing.assert_close(reused.cpu(), golden, rtol=0.01, atol=0.002)
    repeat = mega_mla(*inputs, **later, **options, sequence_metadata=metadata).cpu()
    assert torch.equal(ordinary, repeat)
    torch.testing.assert_close(
        metadata.query_info.cpu(), original_metadata, rtol=0, atol=0
    )
    raw["seqused_q"].add_(0)
    with pytest.raises(ValueError, match="stale"):
        mega_mla(*inputs, **later, **options, sequence_metadata=metadata)
    print(
        "QUERY_METADATA_REUSE_PASS",
        heads,
        "max_abs",
        (reused.cpu().float() - golden.float()).abs().max().item(),
        flush=True,
    )


@pytest.mark.npu
@pytest.mark.parametrize("heads,tokens", [(8, 6), (16, 8), (64, 64)])
def test_function_entry_device(heads, tokens, monkeypatch):
    import mega_mla as mega

    case = _exercise_attention(
        heads,
        tokens,
        False,
        monkeypatch,
        cmp_strategy="indexed",
        win_pages=9,
        cmp_pages=17,
        sequence_case=True,
        return_result=True,
    )
    inputs, options = case["inputs"], case["call_tables"]
    output = mega.mega_mla(*inputs, **options, cmp_decode_strategy="indexed")
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), case["output"], rtol=0, atol=0)
    supplied = torch.empty_like(output)
    result = mega.mega_mla(
        *inputs, **options, out=supplied, cmp_decode_strategy="indexed"
    )
    torch.npu.synchronize()
    assert result is supplied and torch.equal(result.cpu(), case["output"])
    before = inputs[9].cpu().clone()
    options["seqused_q"].zero_()
    empty = mega.mega_mla(*inputs, **options, cmp_decode_strategy="indexed")
    torch.npu.synchronize()
    assert torch.count_nonzero(empty.cpu()) == 0
    torch.testing.assert_close(inputs[9].cpu(), before, rtol=0, atol=0)
    options["seqused_q"].copy_(case["seq_cpu"]["seqused_q"].npu())
    restored = mega.mega_mla(*inputs, **options, cmp_decode_strategy="indexed")
    torch.npu.synchronize()
    assert torch.equal(restored.cpu(), case["output"])
    if (heads, tokens) == (16, 8):
        # Warmed programs are shared; capture on another stream gets its own
        # workspace without compiling inside the graph capture region.
        count = sum(len(pool) for pool in mega._FUNCTION_STATE.programs.values())
        stream = torch.npu.Stream()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            mega.mega_mla(
                *inputs, **options, out=supplied, cmp_decode_strategy="indexed"
            )
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(supplied.cpu(), case["output"])
        assert (
            sum(len(pool) for pool in mega._FUNCTION_STATE.programs.values()) == count
        )
        del graph
    print("FUNCTION_ENTRY_DEVICE_PASS", heads, tokens, flush=True)


@pytest.mark.npu
@pytest.mark.parametrize(
    "heads,tokens,strategy",
    [
        (8, 8, "indexed"),
        (16, 8, "indexed"),
        (64, 8, "predecode"),
        (8, 6, "indexed"),
        (16, 6, "indexed"),
        (64, 8, "indexed"),
        (64, 6, "indexed"),
        (64, 7, "indexed"),
        (8, 10, "indexed"),
        (16, 15, "indexed"),
        (8, 64, "indexed"),
        (16, 64, "indexed"),
        (64, 64, "indexed"),
    ],
)
def test_packed_query_and_kv_sequence_lengths(heads, tokens, strategy, monkeypatch):
    _exercise_attention(
        heads,
        tokens,
        False,
        monkeypatch,
        cmp_strategy=strategy,
        win_pages=9,
        cmp_pages=17,
        sequence_case=True,
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "heads,tokens,batch,strategy,lengths",
    [
        (8, 8, 2, "indexed", "token"),
        (16, 8, 2, "predecode", True),
        (16, 8, 8, "indexed", "token"),
        (64, 8, 4, "predecode", "token"),
        (64, 8, 8, "indexed", False),
        (16, 1, 1, "indexed", False),
        (16, 16, 4, "predecode", "token"),
        (64, 64, 8, "indexed", "token"),
    ],
)
def test_paged_multi_batch_attention(
    heads, tokens, batch, strategy, lengths, monkeypatch
):
    _exercise_attention(
        heads,
        tokens,
        lengths,
        monkeypatch,
        win_pages=1 + 2 * batch,
        cmp_pages=1 + 4 * batch,
        cache_batch=batch,
        cmp_strategy=strategy,
        k_attn=640,
    )


@pytest.mark.npu
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("batch", [1, 2, 3, 6])
def test_small_token_batch_factorization(heads, batch, monkeypatch):
    _exercise_attention(
        heads,
        6,
        "token",
        monkeypatch,
        win_pages=1 + 2 * batch,
        cmp_pages=1 + 4 * batch,
        cache_batch=batch,
        cmp_strategy="predecode" if batch == 3 else "indexed",
    )


@pytest.mark.npu
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("tokens", range(1, 16))
def test_small_token_attention(heads, tokens, monkeypatch):
    """Every partial-M16 token count, including nondivisors of the AIV count."""
    _exercise_attention(
        heads,
        tokens,
        False,
        monkeypatch,
        cmp_pages=42,
        win_pages=4,
        cmp_strategy="indexed",
    )


@pytest.mark.npu
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("tokens", [1, 2])
def test_small_token_predecode(heads, tokens, monkeypatch):
    """No speculative first-batch read by owners with an empty shard."""
    _exercise_attention(
        heads,
        tokens,
        False,
        monkeypatch,
        cmp_pages=5,
        win_pages=4,
        cmp_strategy="predecode",
    )


@pytest.mark.npu
@pytest.mark.parametrize("heads", [8, 16])
def test_small_token_wider_attention(heads, monkeypatch):
    """The tile-work criterion also covers a wider attention reduction."""
    _exercise_attention(
        heads,
        10,
        False,
        monkeypatch,
        cmp_pages=42,
        win_pages=4,
        cmp_strategy="indexed",
        k_attn=2176,
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "heads,tokens,use_lengths",
    [
        (8, 8, False),
        (16, 8, False),
        (64, 8, False),
        (64, 64, False),
        (64, 4, False),
        (16, 1, False),
        (16, 16, False),
        (16, 32, False),
        (16, 64, False),
        (16, 8, True),
        (64, 64, True),
        (16, 100, False),
    ],
)
def test_public_business_device_against_cpu(heads, tokens, use_lengths, monkeypatch):
    _exercise_attention(heads, tokens, use_lengths, monkeypatch)


@pytest.mark.npu
@pytest.mark.parametrize(
    "heads,tokens,grid,strategy,sequence",
    [
        (8, 8, 1, "indexed", False),
        (16, 6, 2, "predecode", False),
        (8, 6, 3, "indexed", False),
        (16, 8, 7, "predecode", False),
        (64, 8, 7, "indexed", False),
        (16, 8, 14, "indexed", False),
        (8, 8, 20, "predecode", False),
        (16, 8, 27, "indexed", False),
        (16, 8, 28, "indexed", False),
        (64, 16, 3, "predecode", False),
        (64, 64, 14, "indexed", False),
        (16, 8, 7, "indexed", "packed"),
        (8, 8, 27, "predecode", "packed"),
    ],
)
def test_selected_grid_device_against_cpu(
    heads, tokens, grid, strategy, sequence, monkeypatch
):
    _exercise_attention(
        heads,
        tokens,
        False,
        monkeypatch,
        grid=grid,
        cmp_strategy=strategy,
        cmp_pages=42,
        win_pages=9,
        sequence_case=sequence,
        compressed_page_size=64,
        window_page_padding=64,
        compressed_page_padding=96,
        cache_storage_offset=64,
    )


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize(
    "heads,tokens,grid,strategy",
    [
        (8, 8, 1, "indexed"),
        (16, 6, 3, "predecode"),
        (64, 16, 7, "indexed"),
        (16, 8, 29, "indexed"),
        (64, 8, 31, "predecode"),
    ],
)
def test_selected_grid_codegen(heads, tokens, grid, strategy):
    _compile_mega(heads, tokens, grid, strategy=strategy, nc=42)


@pytest.mark.npu
def test_window_cache_slot_boundaries(monkeypatch):
    import torch

    # Check skipped/reserved records, adjacent records and a physical page
    # transition after the fused writer changed to one contiguous 544-B DMA.
    slots = torch.tensor([-1, 0, 127, 128, 129, 255, 256, 257], dtype=torch.int32)
    _exercise_attention(
        16,
        8,
        False,
        monkeypatch,
        win_pages=3,
        slot_override=slots,
        update_kv_gamma=True,
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "heads,tokens,page_size,strategy,window_padding,compressed_padding,batch,sequence,columns,high_pages",
    [
        (8, 8, 64, "indexed", 64, 96, None, False, 4, False),
        (16, 6, 128, "indexed", 17, 17, None, False, 4, False),
        (64, 8, 64, "predecode", 64, 96, None, False, 4, False),
        (64, 16, 64, "indexed", 64, 96, None, False, 4, False),
        (16, 8, 64, "predecode", 32, 64, 2, False, 4, False),
        (8, 8, 64, "indexed", 64, 96, None, "packed", 4, False),
        (16, 8, 64, "indexed", 32, 64, None, "packed", 84, True),
        (16, 8, 64, "indexed", 64, 96, None, "packed", 8193, True),
        (64, 64, 64, "indexed", 17, 17, None, False, 4, False),
    ],
)
def test_native_cache_device_against_cpu(
    heads,
    tokens,
    page_size,
    strategy,
    window_padding,
    compressed_padding,
    batch,
    sequence,
    columns,
    high_pages,
    monkeypatch,
):
    _exercise_attention(
        heads,
        tokens,
        batch is not None,
        monkeypatch,
        win_pages=9,
        cmp_pages=42,
        cmp_strategy=strategy,
        compressed_page_size=page_size,
        window_page_padding=window_padding,
        compressed_page_padding=compressed_padding,
        cache_storage_offset=64,
        cache_batch=batch,
        sequence_case=sequence,
        sequence_cmp_table_columns=columns,
        sequence_high_pages=high_pages,
        switch_cache_layout=not sequence and batch is None,
    )


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize("launcher", ["projection", "mxfp8"])
def test_native_standalone_launchers_codegen(launcher):
    """Python launch entries must compile as native hosts without mega imports."""
    if shutil.which("bisheng") is None:
        pytest.skip("CANN bisheng toolchain unavailable; no AOT result")
    dsl = _dsl()
    dt, spec = dsl.dtypes, dsl.TensorSpec
    f32, bf = dt.float32, dt.bfloat16
    fp8, scale = dt.float8_e4m3fn, dt.float8_e8m0
    if launcher == "mxfp8":
        from mxfp8_matmul import Mxfp8MatmulKernel

        op = Mxfp8MatmulKernel(16, 128, 128, 16, 128, 128, 128, 128, used_core_num=28)
        specs = [
            spec(shape, dtype)
            for shape, dtype in [
                ((16, 128), bf),
                ((16, 128), fp8),
                ((128, 128), fp8),
                ((16, 2, 2), scale),
                ((128, 2, 2), scale),
                ((1,), f32),
            ]
        ]
    else:
        from qkv_projection import QkvProjectionOp

        op = QkvProjectionOp(16)
        shapes = [
            ((16, 1280), f32),
            ((16, 512), f32),
            ((16, 8192), bf),
            ((16, 5120), bf),
            ((1, 1280), f32),
            ((1, 512), f32),
            ((1280, 5120), fp8),
            ((512, 5120), fp8),
            ((16, 1280), fp8),
            ((8192, 1280), fp8),
            ((16, 5120), fp8),
            ((16, 80, 2), scale),
            ((16, 1280), fp8),
            ((16, 20, 2), scale),
            ((1280, 80, 2), scale),
            ((512, 80, 2), scale),
            ((8192, 20, 2), scale),
            ((16, 512), bf),
        ]
        specs = [
            spec(shape, dtype, storage_format="nz" if i in (6, 7, 9) else "nd")
            for i, (shape, dtype) in enumerate(shapes)
        ]
    compiled = dsl.compile(op.run, *specs)
    try:
        assert compiled.so_path and Path(compiled.so_path).is_file()
    finally:
        compiled.close()


@pytest.mark.npu
@pytest.mark.parametrize(
    "rows,nz_weight",
    [
        (8, True),
        (16, False),
        (28, True),
        (28, False),
    ],
)
def test_mxfp8_projection_scales_device(rows, nz_weight):
    """Nonuniform scales distinguish MX MMAD from ordinary FP8 multiplication."""
    dsl = _dsl()
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    from mxfp8_matmul import Mxfp8MatmulKernel

    torch.manual_seed(905)
    a = (torch.randn(rows, 128) * 0.2).to(torch.float8_e4m3fn)
    b = (torch.randn(128, 128) * 0.2).to(torch.float8_e4m3fn)
    sa = torch.randint(123, 128, (rows, 2, 2), dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    sb = torch.randint(123, 128, (128, 2, 2), dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    expected = (a.float() * sa.float().reshape(rows, 4).repeat_interleave(32, -1)) @ (
        b.float() * sb.float().reshape(128, 4).repeat_interleave(32, -1)
    ).T
    weight = (
        torch_npu.npu_format_cast(b.npu(), torch_npu.Format.FRACTAL_NZ)
        if nz_weight
        else b.npu()
    )
    if nz_weight:
        import acl

        assert weight.dtype == torch.float8_e4m3fn
        assert torch_npu.get_npu_format(weight) == torch_npu.Format.FRACTAL_NZ
        assert weight.storage_offset() == 0
        # Inspect physical bytes: private NZ -> ND round trips are not the oracle.
        raw = torch.empty(b.numel(), dtype=torch.uint8, device="npu")
        canonical = (
            b.view(torch.uint8)
            .reshape(8, 16, 4, 32)
            .permute(2, 0, 1, 3)
            .contiguous()
            .flatten()
        )
        assert (
            acl.rt.memcpy(
                raw.data_ptr(), raw.numel(), weight.data_ptr(), raw.numel(), 3
            )
            == 0
        )
        torch.npu.synchronize()
        assert torch.equal(raw.cpu(), canonical)
    output_storage = torch.full(
        (rows + 16, 128), 19.0, dtype=torch.bfloat16, device="npu"
    )
    args = [
        output_storage[:rows],
        a.npu(),
        weight,
        sa.npu(),
        sb.npu(),
        torch.zeros(1, device="npu"),
    ]
    types = {
        torch.bfloat16: dsl.dtypes.bfloat16,
        torch.float8_e4m3fn: dsl.dtypes.float8_e4m3fn,
        torch.float8_e8m0fnu: dsl.dtypes.float8_e8m0,
        torch.int8: dsl.dtypes.int8,
        torch.float32: dsl.dtypes.float32,
    }
    op = Mxfp8MatmulKernel(
        rows,
        128,
        128,
        min(rows, 16),
        128,
        128,
        128,
        128,
        b_gm_nz=nz_weight,
        used_core_num=1,
    )
    fn = dsl.compile(
        op.run,
        *(
            dsl.TensorSpec(
                tuple(t.shape),
                types[t.dtype],
                storage_format="nz" if i == 2 and nz_weight else "nd",
            )
            for i, t in enumerate(args)
        ),
    )
    try:
        for _ in range(3):
            args[0].fill_(float("nan"))
            output_storage[rows:].fill_(19.0)
            fn(*args)
            torch.npu.synchronize()
            torch.testing.assert_close(
                args[0].cpu().float(), expected, rtol=0.01, atol=0.002
            )
            assert torch.all(output_storage[rows:].cpu() == 19.0), (
                "tail store overwrote guard rows"
            )
    finally:
        fn.close()


@pytest.mark.npu
@pytest.mark.parametrize(
    "heads,tokens,k,pages,use_lengths",
    [
        (16, 8, 640, 33, True),
        (16, 8, 2176, 256, True),
        (16, 16, 640, 129, "token"),
        (64, 64, 640, 257, True),
    ],
)
def test_full_predecode_indexed_pair(heads, tokens, k, pages, use_lengths, monkeypatch):
    pre = _exercise_attention(
        heads,
        tokens,
        use_lengths,
        monkeypatch,
        cmp_pages=pages,
        k_attn=k,
        cmp_strategy="predecode",
        return_result=True,
    )
    indexed = _exercise_attention(
        heads,
        tokens,
        use_lengths,
        monkeypatch,
        cmp_pages=pages,
        k_attn=k,
        cmp_strategy="indexed",
        return_result=True,
    )
    assert indexed["op"].selected_cmp_decode_strategy == "indexed"
    assert indexed["op"]._workspace["cmp_dec"].shape == (1, 1, 512)
    torch.testing.assert_close(indexed["gathered"], pre["gathered"], rtol=0, atol=0)
    torch.testing.assert_close(indexed["output"], pre["output"], rtol=0, atol=0)
    print(
        "ADAPTIVE_PAIR_PASS",
        heads,
        tokens,
        k,
        pages,
        "cpu_max_abs",
        indexed["max_abs"],
        flush=True,
    )


@pytest.mark.npu
@pytest.mark.parametrize("heads", [8, 16, 64])
def test_indexed_decode_projection_overlap(heads, monkeypatch):
    """Compare early selected decoding with independent full-cache decoding."""
    reference = _exercise_attention(
        heads,
        8,
        False,
        monkeypatch,
        cmp_pages=42,
        win_pages=4,
        cmp_strategy="predecode",
        return_result=True,
    )
    indexed = _exercise_attention(
        heads,
        8,
        False,
        monkeypatch,
        cmp_pages=42,
        win_pages=4,
        cmp_strategy="indexed",
        return_result=True,
    )
    torch.testing.assert_close(
        indexed["gathered"], reference["gathered"], rtol=0, atol=0
    )
    torch.testing.assert_close(indexed["output"], reference["output"], rtol=0, atol=0)


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize("grid", [28, 32])
@pytest.mark.parametrize("heads", [8, 16, 64])
def test_indexed_overlap_codegen(heads, grid):
    _compile_mega(heads, 8, grid, strategy="indexed", nc=42)


@pytest.mark.npu
def test_auto_short_long_short_same_object(monkeypatch):
    shared = None
    for pages, expected in [(5, "predecode"), (33, "indexed"), (5, "predecode")]:
        result = _exercise_attention(
            16,
            8,
            True,
            monkeypatch,
            cmp_pages=pages,
            reuse_op=shared,
            return_result=True,
        )
        shared = result["op"]
        assert shared.selected_cmp_decode_strategy == expected
        print("ADAPTIVE_AUTO_PASS", pages, expected, flush=True)


@pytest.mark.npu
@pytest.mark.parametrize(
    "tokens,kc,length_kind,paged",
    [
        (1, 512, "none", False),
        (8, 512, "batch", False),
        (16, 2048, "token", False),
        (8, 4096, "token", True),
        (8, 4097, "token", True),
    ],
)
def test_indexed_raw_decode_device(tokens, kc, length_kind, paged):
    dsl = pytest.importorskip("cannbotdsl")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    torch.npu.set_device(0)
    CompressedCacheGatherProbe = _device_probes().CompressedCacheGatherProbe
    torch.manual_seed(101)
    pages, ps = 33, 128
    cache = torch.randint(0, 256, (pages, ps, 1, 320), dtype=torch.uint8)
    scales = torch.exp2(torch.randint(-12, 13, (pages, ps, 32)).float())
    cache[..., 256:] = scales.to(torch.bfloat16).view(torch.uint8).unsqueeze(2)
    table = (
        torch.randperm(pages - 1).add(1).to(torch.int32).reshape(2, -1)
        if paged
        else None
    )
    logical_rows = table.shape[1] * ps if paged else pages * ps
    idx = torch.randint(0, logical_rows, (tokens, kc), dtype=torch.int32)
    idx[:, :8] = torch.tensor([0, 127, 128, logical_rows - 1, 128, -1, -7, 0])
    idx_base = idx.clone()
    if length_kind == "none":
        lengths = None
        n = 1
    elif length_kind == "batch":
        lengths = torch.tensor([0, 257], dtype=torch.int32)
        n = 2
    else:
        lengths = torch.tensor([0, 1, 257, kc], dtype=torch.int32).repeat(
            (tokens + 3) // 4
        )[:tokens]
        n = tokens
    if lengths is not None:
        expanded = lengths.repeat_interleave(tokens // n)
        masked = torch.arange(kc)[None, :] >= expanded[:, None]
        idx[masked] = logical_rows + 100  # Never dereference masked positive entries.
    batch_stride = tokens // 2 if paged else 0
    grid = int(torch.npu.get_device_properties(0).vector_core_num) if paged else 1
    probe = CompressedCacheGatherProbe(
        tokens, kc, 0 if lengths is None else tokens // n, batch_stride, grid
    )
    dt, spec = dsl.dtypes, dsl.TensorSpec
    specs = [
        spec((pages, ps, 320), dt.uint8),
        spec((tokens, kc), dt.int32),
        spec((n,), dt.int32),
        spec((1, 512), dt.bfloat16),
        spec((tokens, 128 + kc, 512), dt.bfloat16),
    ]
    if paged:
        specs.append(spec(tuple(table.shape), dt.int32))
    fn = dsl.compile(probe.run_paged if paged else probe.run, *specs)
    index_npu = idx.npu()
    length_npu = (
        torch.zeros(n, dtype=torch.int32) if lengths is None else lengths
    ).npu()
    zero = torch.zeros(1, 512, dtype=torch.bfloat16).npu()
    table_npu = table.npu() if paged else None
    try:
        for repeat in range(3):
            if repeat == 1:
                cache[..., :256] ^= 17
            if repeat == 2:
                idx = idx_base.roll(1, dims=1)
                idx[:, ::7] = -1
                if lengths is not None:
                    lengths = kc - lengths
                    expanded = lengths.repeat_interleave(tokens // n)
                    masked = torch.arange(kc)[None, :] >= expanded[:, None]
                    idx[masked] = logical_rows + 100
                    # 在同一地址更新元数据，验证每次launch均重新加载UB/标量缓存。
                    length_npu.copy_(lengths.npu())
                index_npu.copy_(idx.npu())
                if paged:
                    table = table.roll(1, dims=1)
                    table_npu.copy_(table.npu())
            output = torch.full(
                (tokens, 128 + kc, 512), 7.0, dtype=torch.bfloat16
            ).npu()
            inputs = [
                cache.reshape(pages, ps, 320).npu(),
                index_npu,
                length_npu,
                zero,
                output,
            ]
            fn(*inputs, *([table_npu] if paged else []))
            torch.npu.synchronize()
            actual = output.cpu()
            if paged:
                selected = idx.clone()
                expanded = lengths.repeat_interleave(tokens // n)
                selected[torch.arange(kc)[None, :] >= expanded[:, None]] = -1
                safe = selected.clamp_min(0).long()
                batch_id = torch.arange(tokens) // batch_stride
                mapped = table[batch_id[:, None], safe // ps].long() * ps + safe % ps
                selected = torch.where(selected >= 0, mapped, selected).to(torch.int32)
                expected = decode_selected_cmp(cache, selected, None)
            else:
                expected = decode_selected_cmp(cache, idx, lengths)
            torch.testing.assert_close(actual[:, 128:], expected, rtol=0, atol=0)
            assert torch.all(actual[:, :128] == 7.0)
        print(
            "INDEXED_RAW_DEVICE_PASS",
            tokens,
            kc,
            length_kind,
            "paged",
            paged,
            flush=True,
        )
    finally:
        fn.close()


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("strategy", ["predecode", "indexed"])
def test_full_strategy_codegen(strategy, heads):
    _compile_mega(heads, 8, 28, strategy=strategy, nc=33, length_stride=4)


@pytest.mark.npu
@pytest.mark.parametrize(
    "kind,unroll",
    [
        ("window_scales", None),
        ("window_records", None),
        ("compressed", 1),
        ("compressed", 2),
        ("compressed", 4),
    ],
)
def test_cache_record_layout(kind, unroll):
    """Byte-exact checks formerly exposed as three standalone probe scripts."""
    _dsl()
    pytest.importorskip("torch_npu", reason="torch_npu unavailable")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    torch.npu.set_device(0)
    WindowCacheScaleProbe = _device_probes().WindowCacheScaleProbe
    WindowCacheRecordProbe = _device_probes().WindowCacheRecordProbe
    CompressedCacheDecodeProbe = _device_probes().CompressedCacheDecodeProbe

    if kind == "window_scales":
        torch.manual_seed(671)
        rows = 64
        x = (torch.randn(rows, 64) * 3).to(torch.bfloat16)
        x_full = torch.zeros((1, rows, 1, 512), dtype=torch.bfloat16)
        x_full[0, :, 0, :64] = x
        expected = quantize_fp8(x_full)[0, :, 0]
        data = torch.empty((rows, 64), dtype=torch.int8, device="npu")
        scale = torch.empty((rows, 2), dtype=torch.int16, device="npu")
        WindowCacheScaleProbe().run(x.npu(), data, scale)
        torch.npu.synchronize()
        assert torch.equal(data.cpu().view(torch.uint8), expected[:, :64])
        assert torch.equal(scale.cpu().view(torch.uint8), expected[:, 512:516])
    elif kind == "window_records":
        torch.manual_seed(672)
        x = (torch.randn(2, 512) * 3).to(torch.bfloat16)
        expected = quantize_fp8(x.reshape(1, 2, 1, 512))
        cache = torch.zeros((1, 2, 1, 544), dtype=torch.uint8, device="npu")
        flat = cache.reshape(1, 2 * 544)
        WindowCacheRecordProbe().run(
            x.npu(), flat.view(torch.int8), flat.view(torch.uint64)
        )
        torch.npu.synchronize()
        assert torch.equal(cache.cpu(), expected)
    else:
        torch.manual_seed(673)
        rows = 8
        x = (torch.randn((1, rows, 1, 512)) * 2).to(torch.bfloat16)
        cache = quantize_fp4(x)
        assert cache.shape[-1] == 320
        expected = dequantize_fp4(cache).reshape(rows, 512)
        src = cache.reshape(rows, 320).npu()
        dst = torch.empty((rows, 512), dtype=torch.bfloat16, device="npu")
        CompressedCacheDecodeProbe(unroll).run(src, dst)
        torch.npu.synchronize()
        torch.testing.assert_close(
            dst.cpu().view(torch.int16), expected.view(torch.int16), atol=0, rtol=0
        )


@pytest.mark.npu
def test_sequence_large_page_table_fallback_device(monkeypatch):
    _exercise_attention(
        16,
        8,
        False,
        monkeypatch,
        cmp_pages=17,
        win_pages=9,
        cmp_strategy="indexed",
        sequence_case=True,
        sequence_cmp_table_columns=65,
    )


@pytest.mark.npu
def test_sequence_fixed_query_layout_device(monkeypatch):
    _exercise_attention(
        16,
        8,
        False,
        monkeypatch,
        cmp_pages=17,
        win_pages=9,
        cmp_strategy="indexed",
        sequence_case="fixed",
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "tokens,k,strategy,sequence",
    [
        (14, 256, "indexed", False),
        (28, 640, "indexed", True),
        (64, 640, "predecode", True),
    ],
)
def test_h64_cv_shared_kv_boundaries(tokens, k, strategy, sequence, monkeypatch):
    _exercise_attention(
        64,
        tokens,
        "token" if not sequence else False,
        monkeypatch,
        cmp_pages=42,
        k_attn=k,
        cmp_strategy=strategy,
        win_pages=9 if sequence else 4,
        sequence_case=sequence,
    )


# ============================================================================
# Single-shape pressure benchmark
# ============================================================================


def benchmark_attention(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(SAMPLE_DIR))
    parser.add_argument("--mode", choices=("proj", "cache", "attn"), default="attn")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--k", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--pressure-mib", type=int, default=256)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--snapshot")
    parser.add_argument("--compare")
    parser.add_argument("--skip-ref", action="store_true")
    cli = parser.parse_args(argv)
    import torch_npu  # Register the NPU backend after parsing --help.

    torch.npu.set_device(0)
    torch.manual_seed(31)
    torch.set_num_threads(4)
    sys.path.insert(0, cli.source)
    import cannbotdsl
    from mega_mla import (
        MegaMlaKernel,
        MegaMlaOp,
    )

    t, h, d, ps, kw = cli.tokens, cli.heads, 512, 128, 128
    k = cli.k
    nc = max(5, (k - kw + ps - 1) // ps + 1)
    x = (torch.randn(t, 5120) * 0.1).to(torch.bfloat16)
    gfq, gfk = torch.rand(1, 1280) + 0.5, torch.rand(1, 512) + 0.5
    weights = [
        (torch.randn(n, width) * 0.05).to(torch.float8_e4m3fn)
        for n, width in ((1280, 5120), (512, 5120), (h * d, 1280))
    ]
    scales = [
        torch.full((n, width // 64, 2), 125, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        for n, width in ((1280, 5120), (512, 5120), (h * d, 1280))
    ]
    theta = torch.randn(t, 32)
    cos, sin = theta.cos(), theta.sin()
    win_float = (torch.randn(2, ps, 1, d) * 0.1).to(torch.bfloat16)
    cmp_float = (torch.randn(nc, ps, 1, d) * 0.1).to(torch.bfloat16)
    indices_in_kvcache = torch.randint(ps, 2 * ps, (t, kw), dtype=torch.int32)
    extra_indices_in_kvcache = torch.randint(
        ps, nc * ps, (t, k - kw), dtype=torch.int32
    )
    indices_in_kvcache[:, ::7] = -1
    extra_indices_in_kvcache[:, ::11] = -1
    slot = torch.arange(ps, ps + t, dtype=torch.int32)
    sink = torch.randn(1, h)
    wa, wkv, wb = [
        torch_npu.npu_format_cast(w.npu(), torch_npu.Format.FRACTAL_NZ) for w in weights
    ]
    win = quantize_fp8(win_float).npu()
    cmp = quantize_fp4(cmp_float).npu()
    wr = int(win.shape[-1])
    wkc = win.view(2, ps * wr).view(torch.int8)

    from attention_workspace import AttentionWorkspaceLayout

    layout = AttentionWorkspaceLayout(t, h, k, nc, ps, "predecode")
    storage, w = layout.allocate("npu")
    qr, kv, qfull = w["qr"], w["kv"], w["qfull"]
    x8, sx, qn8, sqn = w["x8"], w["sx"], w["qn8"], w["sqn"]
    kvn, qb, kg = w["kvn"], w["qb"], w["kg"]
    score, prob, _cmp_dec = w["s"], w["p"], w["cmp_dec"]
    out = torch.empty((t, h, d), dtype=torch.bfloat16, device="npu")
    wi, ci = indices_in_kvcache.npu(), extra_indices_in_kvcache.npu()
    args = [
        x.npu(),
        gfq.npu(),
        gfk.npu(),
        wa,
        wkv,
        wb,
        *(s.npu() for s in scales),
        win,
        cmp,
        wi,
        ci,
        sink.npu(),
        cos.npu(),
        sin.npu(),
        slot.npu(),
        wi,
        ci,
        wi,
        ci,
        out,
        *layout.arguments(storage),
    ]
    assert len(args) == 27
    grid = int(torch.npu.get_device_properties(0).cube_core_num)
    init = dict(
        k=k, page_size=ps, win_len=kw, grid=grid, mode=cli.mode, tokens=t, heads=h
    )
    op = MegaMlaOp(**init)
    print(
        json.dumps(
            dict(
                stage="compile",
                dsl=cannbotdsl.__version__,
                source=cli.source,
                mode=cli.mode,
                t=t,
                h=h,
                k=k,
                grid=grid,
                win_record=wr,
                cmp_record=int(cmp.shape[-1]),
            )
        ),
        flush=True,
    )
    kernel_init = dict(
        mode=cli.mode,
        tokens=t,
        grid=grid,
        gather_batch=op.gather_batch,
        heads=h,
        workspace_layout=layout,
    )
    op.kernel = MegaMlaKernel(k, ps, kw, **kernel_init)
    launch_args = args
    dtype_map = {
        torch.float32: cannbotdsl.dtypes.float32,
        torch.bfloat16: cannbotdsl.dtypes.bfloat16,
        torch.int8: cannbotdsl.dtypes.int8,
        torch.uint8: cannbotdsl.dtypes.uint8,
        torch.uint64: cannbotdsl.dtypes.uint64,
        torch.uint32: cannbotdsl.dtypes.uint32,
        torch.int32: cannbotdsl.dtypes.int32,
        torch.float8_e4m3fn: cannbotdsl.dtypes.float8_e4m3fn,
        torch.float8_e8m0fnu: cannbotdsl.dtypes.float8_e8m0,
    }
    specs = [
        cannbotdsl.TensorSpec(
            tuple(value.shape),
            dtype_map[value.dtype],
            storage_format="nz" if i in (3, 4, 5) else "nd",
        )
        for i, value in enumerate(launch_args)
    ]
    compiled = cannbotdsl.compile(op._launch_mega_mla, *specs)

    def launch():
        return compiled(*launch_args)

    pressure = BenchmarkCachePressure(cli.pressure_mib)
    pressure.apply()
    launch()
    torch.npu.synchronize()
    for _ in range(cli.warmup):
        pressure.apply()
        launch()
    torch.npu.synchronize()
    if cli.graph:
        pressure.apply()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            launch()
        torch.npu.synchronize()
        launch = graph.replay
    for _ in range(cli.repeat):
        pressure.apply()
        launch()
    torch.npu.synchronize()
    print(
        json.dumps(
            dict(
                stage="measured",
                warmup=cli.warmup,
                repeat=cli.repeat,
                graph=cli.graph,
                pressure_mib=cli.pressure_mib,
                pressure_operation="fp32_read_write_add1",
            )
        ),
        flush=True,
    )
    # Attention/cache fusion keeps normalized KV in UB. Its private GM slot
    # remains populated only by mode=proj, so never snapshot unwritten storage.
    selected = (0, 1, 2, 10, 11, 12, 13, 18)
    if cli.mode == "proj":
        selected = (*selected, 17)
    else:
        selected = (*selected, 31, 32)
        if cli.mode == "attn":
            selected = (*selected, 36)
            if not op.kernel.cv_tile_m:
                selected = (*selected, 33, 34)
    diagnostics = {
        0: qr,
        1: kv,
        2: qfull,
        10: x8,
        11: sx,
        12: qn8,
        13: sqn,
        17: kvn,
        18: wkc,
        31: qb,
        32: kg,
        33: score,
        34: prob,
        36: out,
    }
    snapshot = {str(i): diagnostics[i].cpu() for i in selected}
    if cli.snapshot:
        torch.save(snapshot, cli.snapshot)
    if cli.compare:
        baseline = torch.load(cli.compare, weights_only=True)
        comparison = {}
        for key, value in snapshot.items():
            base = baseline[key]
            if value.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.int8):
                comparison[key] = int(
                    (value.view(torch.uint8) != base.view(torch.uint8)).sum()
                )
            else:
                comparison[key] = float((value.float() - base.float()).abs().max())
        print(json.dumps(dict(stage="comparison", deltas=comparison)), flush=True)
    if cli.skip_ref or cli.mode == "proj":
        return
    golden_win = quantize_fp8(win_float)
    # Independent CPU RMS, preserving the BF16 round before the reduction and
    # the BF16 result consumed by RoPE. No dependency on a diagnostic GM slot.
    kv_rounded = kv.cpu().to(torch.bfloat16).float()
    kv_normalized = (
        kv_rounded
        * torch.rsqrt(kv_rounded.square().mean(-1, keepdim=True) + 1e-20)
        * gfk
    ).to(torch.bfloat16)
    rotated = _rope_cpu(kv_normalized, cos, sin)
    golden_win.view(-1, wr)[slot.long()] = quantize_fp8(
        rotated.reshape(1, t, 1, d)
    ).reshape(t, wr)
    wd = dequantize_fp8(golden_win).reshape(-1, d)
    cd = dequantize_fp4(cmp.cpu()).reshape(-1, d)
    gathered = torch.cat(
        (
            wd[indices_in_kvcache.clamp_min(0).long()],
            cd[extra_indices_in_kvcache.clamp_min(0).long()],
        ),
        1,
    )
    invalid = torch.cat((indices_in_kvcache < 0, extra_indices_in_kvcache < 0), 1)
    gathered[invalid] = 0
    torch.testing.assert_close(win.cpu(), golden_win, rtol=0, atol=0)
    torch.testing.assert_close(kg.cpu(), gathered, rtol=0, atol=0)
    qgold = (
        _rope_cpu(qfull.cpu().reshape(t, h, d), cos[:, None], sin[:, None]).float()
        / d**0.5
    ).to(torch.bfloat16)
    torch.testing.assert_close(qb.cpu(), qgold, rtol=0, atol=0)
    out_error = None
    if cli.mode == "attn":
        s = (qgold.float() @ gathered.float().transpose(1, 2)).masked_fill(
            invalid[:, None, :], -1.7e38
        )
        maximum = torch.maximum(s.amax(-1, keepdim=True), sink[:, :, None])
        exp = torch.exp(s - maximum)
        p = (
            exp / (exp.sum(-1, keepdim=True) + torch.exp(sink[:, :, None] - maximum))
        ).to(torch.bfloat16)
        expected = (p.float() @ gathered.float()).to(torch.bfloat16)
        torch.testing.assert_close(out.cpu(), expected, rtol=0.01, atol=0.002)
        out_error = float((out.cpu().float() - expected.float()).abs().max())
    print(
        json.dumps(
            dict(
                stage="validation_pass",
                cache_bytes_exact=True,
                kg_exact=True,
                qb_exact=True,
                out_max_error=out_error,
            )
        ),
        flush=True,
    )


# ============================================================================
# Matched physical/logical page benchmark
# ============================================================================


def benchmark_paged_attention(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logical", action="store_true")
    parser.add_argument("--heads", type=int, nargs="+", default=[16, 64])
    parser.add_argument(
        "--capacities", type=int, nargs="+", default=[2048, 65536, 1048576]
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--ratio", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=60)
    parser.add_argument("--pressure-mib", type=int, default=256)
    cli = parser.parse_args(argv)
    import numpy as np
    import torch_npu  # Register the NPU backend after parsing --help.

    assert cli.tokens % cli.batch == 0 and cli.ratio > 0
    cli.output.mkdir(parents=True, exist_ok=False)
    generated = cli.output / "generated"
    os.environ["CANNBOTDSL_DUMP_ASCENDC"] = "1"
    os.environ["CANNBOTDSL_DUMP_DIR"] = str(generated.resolve())
    sys.path.insert(0, str(cli.source.resolve()))
    import cannbotdsl as dsl
    from mega_mla import (
        MegaMlaKernel,
        MegaMlaOp,
    )
    from attention_inputs import prepare_attention_arguments

    torch.set_num_threads(4)
    torch.npu.set_device(0)
    pressure = BenchmarkCachePressure(cli.pressure_mib)
    grid = int(torch.npu.get_device_properties(0).cube_core_num)
    t, batch, ps, d, kw, kc = cli.tokens, cli.batch, 128, 512, 128, 2048
    stride = t // batch
    dtype_map = {
        torch.float32: dsl.dtypes.float32,
        torch.bfloat16: dsl.dtypes.bfloat16,
        torch.int8: dsl.dtypes.int8,
        torch.uint8: dsl.dtypes.uint8,
        torch.uint64: dsl.dtypes.uint64,
        torch.uint32: dsl.dtypes.uint32,
        torch.int32: dsl.dtypes.int32,
        torch.float8_e4m3fn: dsl.dtypes.float8_e4m3fn,
        torch.float8_e8m0fnu: dsl.dtypes.float8_e8m0,
    }
    with (cli.output / "ledger.jsonl").open("w") as ledger:
        for heads in cli.heads:
            torch.manual_seed(31)
            x = (torch.randn(t, 5120) * 0.1).to(torch.bfloat16)
            gfq, gfk = torch.rand(1, 1280) + 0.5, torch.rand(1, d) + 0.5
            weights = [
                (torch.randn(n, width) * 0.05).to(torch.float8_e4m3fn)
                for n, width in ((1280, 5120), (d, 5120), (heads * d, 1280))
            ]
            scales = [
                torch.full((n, width // 64, 2), 125, dtype=torch.uint8).view(
                    torch.float8_e8m0fnu
                )
                for n, width in ((1280, 5120), (d, 5120), (heads * d, 1280))
            ]
            qr, kv = [_mx_round(x) @ w.float().T * 0.25 for w in weights[:2]]
            qfull = (
                _mx_round(_rms_reference(qr, gfq)) @ weights[2].float().T * 0.25
            ).to(torch.bfloat16)
            kvnorm = _rms_reference(kv, gfk).to(torch.bfloat16)
            packed = [
                torch_npu.npu_format_cast(w.npu(), torch_npu.Format.FRACTAL_NZ)
                for w in weights
            ]
            del weights
            fixed = [x.npu(), gfq.npu(), gfk.npu(), *packed, *(s.npu() for s in scales)]
            nw = 1 + 2 * batch
            win_initial = quantize_fp8(
                (torch.randn(nw, ps, 1, d) * 0.1).to(torch.bfloat16)
            )
            template = torch.cat(
                [
                    quantize_fp4(
                        (torch.randn(min(8, 257 - start), ps, 1, d) * 0.1).to(
                            torch.bfloat16
                        )
                    )
                    for start in range(0, 257, 8)
                ]
            )
            template_dec = dequantize_fp4(template).reshape(-1, d)
            max_pages = math.ceil(max(cli.capacities) / (ps * cli.ratio))
            max_nc = 1 + batch * max_pages
            pool = template.npu().repeat((math.ceil(max_nc / 257), 1, 1, 1))[:max_nc]
            compiled = {}
            nc_dim = dsl.Dim("PhysicalPages", min=1, max=max_nc)
            table_dim = dsl.Dim("LogicalPages", min=1, max=max_pages)
            for capacity in cli.capacities:
                generator = torch.Generator().manual_seed(71 + capacity)
                logical_pages = math.ceil(capacity / (ps * cli.ratio))
                nc = 1 + batch * logical_pages
                win_table = (
                    (torch.randperm(nw - 1, generator=generator) + 1)
                    .to(torch.int32)
                    .reshape(batch, 2)
                )
                cmp_table = (
                    (torch.randperm(nc - 1, generator=generator) + 1)
                    .to(torch.int32)
                    .reshape(batch, logical_pages)
                )
                wi = torch.randint(
                    0, 2 * ps, (t, kw), dtype=torch.int32, generator=generator
                )
                ci = torch.randint(
                    0,
                    capacity // cli.ratio,
                    (t, kc),
                    dtype=torch.int32,
                    generator=generator,
                )
                wi[:, ::7], ci[:, ::11] = -1, -1
                wi[:, :4] = torch.tensor([0, 127, 128, 255], dtype=torch.int32)
                ci[:, :4] = torch.tensor(
                    [0, 127, 128, capacity // cli.ratio - 1], dtype=torch.int32
                )
                batch_ids = torch.arange(t) // stride

                def physical(index, table):
                    safe = index.clamp_min(0).long()
                    return torch.where(
                        index >= 0,
                        table[batch_ids[:, None], safe // ps].long() * ps + safe % ps,
                        index,
                    ).to(torch.int32)

                wphys, cphys = physical(wi, win_table), physical(ci, cmp_table)
                slot = (win_table[batch_ids, 0] * ps + torch.arange(t) % stride).to(
                    torch.int32
                )
                lengths_w = torch.full((batch,), kw, dtype=torch.int32)
                lengths_c = torch.full(
                    (batch,), min(capacity // cli.ratio, kc), dtype=torch.int32
                )
                theta = (torch.arange(t).float() + capacity)[:, None] * torch.pow(
                    10000.0, -torch.arange(0, 64, 2).float()[None, :] / 64.0
                )
                cos, sin, sink = (
                    theta.cos(),
                    theta.sin(),
                    torch.randn(1, heads, generator=generator),
                )
                win, cmp = win_initial.npu(), pool[:nc]
                indices = (wi, ci) if cli.logical else (wphys, cphys)
                inputs = [
                    *fixed,
                    win,
                    cmp,
                    *(a.npu() for a in indices),
                    sink.npu(),
                    cos.npu(),
                    sin.npu(),
                    slot.npu(),
                ]
                op = MegaMlaOp(grid=grid, heads=heads)
                tables = dict(
                    block_table=win_table.npu(), extra_block_table=cmp_table.npu()
                )
                kwargs = dict(
                    topk_length=lengths_w.npu(), extra_topk_length=lengths_c.npu()
                )
                args, out = prepare_attention_arguments(
                    op, *inputs, **kwargs, **(tables if cli.logical else {})
                )
                # Even physical variants receive the same read-only table ABI
                # placeholders; their compiled kernels never read those slots.
                args[17], args[18] = tables["block_table"], tables["extra_block_table"]
                strategy = op.selected_cmp_decode_strategy
                compile_key = (strategy, op._workspace_layout.nbytes)
                if compile_key not in compiled:
                    owner = MegaMlaOp(grid=grid, heads=heads)
                    init = dict(
                        tokens=t,
                        grid=grid,
                        heads=heads,
                        win_length_stride=stride,
                        cmp_length_stride=stride,
                        cmp_decode_strategy=strategy,
                        workspace_layout=op._workspace_layout,
                    )
                    if cli.logical:
                        init["cache_batch_stride"] = stride
                    owner.kernel = MegaMlaKernel(kw + kc, ps, kw, **init)
                    specs = []
                    for i, a in enumerate(args):
                        shape = tuple(a.shape)
                        if i == 10:
                            shape = (nc_dim, ps, 1, 320)
                        elif i == 18:
                            shape = (batch, table_dim)
                        specs.append(
                            dsl.TensorSpec(
                                shape,
                                dtype_map[a.dtype],
                                storage_format="nz" if i in (3, 4, 5) else "nd",
                            )
                        )
                    print(
                        json.dumps(
                            dict(
                                stage="compile",
                                heads=heads,
                                strategy=strategy,
                                logical=cli.logical,
                            )
                        ),
                        flush=True,
                    )
                    compiled[compile_key] = dsl.compile(owner._launch_mega_mla, *specs)
                    shutil.copyfile(
                        generated / "launch_mega_mla.asc",
                        generated
                        / f"h{heads}_{strategy}_w{op._workspace_layout.nbytes}.asc",
                    )
                fn = compiled[compile_key]
                pressure.apply()
                fn(*args)
                torch.npu.synchronize()
                golden_win = win_initial.clone()
                encoded = quantize_fp8(
                    _rope_cpu(kvnorm, cos, sin).view(1, t, 1, d)
                ).view(t, 544)
                golden_win.view(-1, 544)[slot.long()] = encoded
                wd = dequantize_fp8(golden_win).view(-1, d)
                gathered = torch.cat(
                    (
                        wd[wphys.clamp_min(0).long()],
                        template_dec[cphys.clamp_min(0).long() % (257 * ps)],
                    ),
                    1,
                )
                invalid = torch.cat((wi < 0, ci < 0), 1)
                invalid[:, kw:] |= (
                    torch.arange(kc)[None, :]
                    >= lengths_c.repeat_interleave(stride)[:, None]
                )
                gathered[invalid] = 0
                qb = (
                    _rope_cpu(
                        qfull.view(t, heads, d), cos[:, None], sin[:, None]
                    ).float()
                    / d**0.5
                ).to(torch.bfloat16)
                expected = _attention_reference(qb, gathered, invalid, sink)
                torch.testing.assert_close(win.cpu(), golden_win, rtol=0, atol=0)
                torch.testing.assert_close(
                    op._workspace["kg"].cpu(), gathered, rtol=0, atol=0
                )
                actual = out.cpu()
                torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
                for _ in range(cli.warmup + cli.repeat):
                    pressure.apply()
                    fn(*args)
                torch.npu.synchronize()
                assert torch.equal(out.cpu(), actual), (
                    "benchmark output changed across repeats"
                )
                case_id = f"t{t}_b{batch}_h{heads}_l{capacity}_r{cli.ratio}"
                np.savez_compressed(
                    cli.output / f"{case_id}.npz",
                    win_logical=wi.numpy(),
                    cmp_logical=ci.numpy(),
                    win_physical=wphys.numpy(),
                    cmp_physical=cphys.numpy(),
                    block_table=win_table.numpy(),
                    extra_block_table=cmp_table.numpy(),
                    window_cache_slots=slot.numpy(),
                )
                row = dict(
                    case_id=case_id,
                    heads=heads,
                    tokens=t,
                    batch=batch,
                    capacity=capacity,
                    ratio=cli.ratio,
                    logical=cli.logical,
                    strategy=strategy,
                    win_shape=list(win.shape),
                    cmp_shape=list(cmp.shape),
                    win_table_shape=list(win_table.shape),
                    cmp_table_shape=list(cmp_table.shape),
                    warmup=cli.warmup,
                    repeat=cli.repeat,
                    first=1,
                    pressure_mib=cli.pressure_mib,
                    pressure_operation="fp32_read_write_add1",
                    cache_exact=True,
                    gather_exact=True,
                    max_abs=float((actual.float() - expected.float()).abs().max()),
                    output_sha256=hashlib.sha256(
                        actual.view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                    so=str(fn.so_path),
                )
                ledger.write(json.dumps(row) + "\n")
                ledger.flush()
                print(json.dumps(dict(stage="case_pass", **row)), flush=True)
            for fn in compiled.values():
                fn.close()
            del pool, fixed, packed, args, inputs, op, compiled, out, win, cmp
            torch.npu.empty_cache()
    (cli.output / "complete").touch()


# ============================================================================
# PR review regressions
# ============================================================================


# PR review regressions: scalar boundaries and cross-call workspace ownership.
@pytest.mark.parametrize("compressed_width", [64, 192, 576])
def test_review_reject_k64_only_before_workspace(business, compressed_width):
    op, values = business
    values["extra_indices_in_kvcache"] = torch.zeros(
        8, compressed_width, dtype=torch.int32
    )
    with pytest.raises(ValueError, match="128"):
        public.prepare_attention_arguments(op, **values)
    assert getattr(op, "_workspace_key", None) is None
    assert not hasattr(op, "_workspace_storage")


@pytest.mark.parametrize(
    "field,value",
    [
        ("page_size", 0),
        ("page_size", -128),
        ("page_size", 128.0),
        ("win_len", 0),
        ("win_len", 192),
        ("win_len", 128.0),
        ("grid", 28.0),
        ("grid", True),
        ("tokens", 0),
        ("tokens", 8.0),
        ("expected_k", True),
        ("expected_k", 640.0),
    ],
)
def test_review_reject_invalid_scalar_geometry(business, field, value):
    op, values = business
    setattr(op, field, value)
    with pytest.raises(ValueError):
        public.prepare_attention_arguments(op, **values)
    assert not hasattr(op, "_workspace_storage")


def test_review_window_boundary_changes_reallocate_sequence_workspace(business):
    op, values = business
    used = torch.full((2,), 4, dtype=torch.int32)
    public.prepare_attention_arguments(op, **values, seqused_q=used)
    original = op._workspace_storage
    op.win_len = 256
    values["indices_in_kvcache"] = torch.zeros(8, 256, dtype=torch.int32)
    values["extra_indices_in_kvcache"] = torch.zeros(8, 384, dtype=torch.int32)
    public.prepare_attention_arguments(op, **values, seqused_q=used)
    assert op._workspace_storage is not original
    assert op._workspace["seq_win_idx"].shape == (8, 256)
    assert op._workspace["seq_cmp_idx"].shape == (8, 384)
    stable = op._workspace_storage
    public.prepare_attention_arguments(op, **values, seqused_q=used)
    assert op._workspace_storage is stable


@pytest.mark.parametrize(
    "window_width,expected", [(64, 0), (128, 64), (192, 0), (256, 64)]
)
def test_review_cv_requires_complete_window_tiles(window_width, expected):
    assert (
        select_small_k_attention_head_tile(64, 64, 28, 640, win_len=window_width)
        == expected
    )


def _load_public_inputs_for_review():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "review_nz_inputs", ROOT / "attention_inputs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "aic,aiv,available,expected",
    [
        (28, 56, True, 28),
        (32, 16, True, 8),
        (5, 11, True, 5),
        (0, 56, True, None),
        (28, 0, True, None),
        (28, 56, False, None),
    ],
)
def test_review_current_stream_core_limits(monkeypatch, aic, aiv, available, expected):
    stream, calls = object(), []
    info = SimpleNamespace(cube_core_num=aic, vector_core_num=aiv, available=available)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            device=lambda _: nullcontext(), current_stream=lambda _: stream
        ),
        raising=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "cannbotdsl",
        SimpleNamespace(get_platform_info=lambda **kw: (calls.append(kw), info)[1]),
    )
    if expected is None:
        with pytest.raises(RuntimeError, match="current stream"):
            public.available_aic_count(torch.device("cpu"))
    else:
        assert public.available_aic_count(torch.device("cpu")) == expected
    assert calls == [dict(force_refresh=True, stream=stream)]


@pytest.mark.parametrize(
    "kind", ["root", "same_span", "offset", "larger_storage", "transpose", "nd"]
)
def test_review_nz_uses_public_storage_contract(monkeypatch, kind):
    module = _load_public_inputs_for_review()
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            Format=SimpleNamespace(FRACTAL_NZ=29),
            get_npu_format=lambda _: 0 if kind == "nd" else 29,
        ),
    )
    value = torch.empty((16, 32), dtype=torch.float8_e4m3fn)
    if kind == "same_span":
        value = value.view(16, 32)
    elif kind == "offset":
        value = torch.empty((17, 32), dtype=value.dtype)[1:]
    elif kind == "larger_storage":
        value = torch.empty((17, 32), dtype=value.dtype)[:16]
    elif kind == "transpose":
        value = value.T
    if kind in ("root", "same_span"):
        module._validate_nz_weight("weight", value)
    else:
        with pytest.raises(ValueError, match="complete E4M3 NZ allocation"):
            module._validate_nz_weight("weight", value)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, 134217728),
        ("bad", 134217728),
        ("0", 134217728),
        ("-1", 134217728),
        ("1048576", 1048576),
    ],
)
def test_review_l2_override_is_lazy_and_tolerant(monkeypatch, value, expected):
    tree = ast.parse((ROOT / "mxfp8_matmul.py").read_text(encoding="utf-8"))
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_l2_capacity_bytes"
    )
    if value is None:
        monkeypatch.delenv("QBMM_MX_L2_SIZE_BYTES", raising=False)
    else:
        monkeypatch.setenv("QBMM_MX_L2_SIZE_BYTES", value)
    namespace = dict(os=os, DEFAULT_L2_CAPACITY_BYTES=128 * 1024 * 1024)
    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), "l2_override", "exec"),
        namespace,
    )
    assert namespace["_l2_capacity_bytes"]() == expected


@pytest.mark.ascendc_toolchain
def test_review_module_import_preserves_search_path():
    pytest.importorskip("cannbotdsl")
    before = list(sys.path)
    import mega_mla as mega

    assert sys.path == before
    assert callable(mega.mega_mla)


@pytest.mark.npu
@pytest.mark.parametrize("sequence", [False, True])
def test_review_zero_record_initialized_in_kernel(monkeypatch, sequence):
    _exercise_attention(
        16,
        8,
        False,
        monkeypatch,
        cmp_pages=42,
        win_pages=17 if sequence else 4,
        cmp_strategy="indexed",
        sequence_case=sequence,
        poison_zero=True,
    )


@pytest.mark.parametrize("filename", ["mega_mla_kernel.py", "qkv_projection.py"])
@pytest.mark.parametrize("tokens", [8, 64, 100, 129])
def test_review_split_k_partial_m_transfer_extents(filename, tokens):
    """Execute the actual split-K body with checked CPU transfer endpoints."""
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_compute_split_k_projection_tile"
    )
    fn.decorator_list = []
    copies = []

    def copy(dst, src, **kwargs):
        assert tuple(dst.shape) == tuple(src.shape)
        copies.append(tuple(dst.shape))
        dst.copy_(src)

    def tile(value, size, coord):
        return value[tuple(slice(s * c, s * (c + 1)) for s, c in zip(size, coord))]

    def ring(shape, dtype=torch.int8):
        data = torch.zeros(shape, dtype=dtype)
        return SimpleNamespace(next_read=lambda: data, next_write=lambda: data)

    pm = min(64, tokens)
    pipeline = SimpleNamespace(
        l1_a=ring((pm, 512)),
        l1_b=ring((128, 512)),
        l1_scale_a=ring((pm, 80)),
        l1_scale_b=ring((80, 128)),
        l0a=ring((pm, 256)),
        l0b=ring((128, 256)),
        l0c=ring((pm, 128), torch.float32),
    )
    namespace = dict(
        const_expr=lambda x: x,
        range_constexpr=range,
        tile_view=tile,
        mem_copy=copy,
        copy_options=lambda **kwargs: kwargs,
        _as_zn_view=lambda x: x,
        matmul=lambda *args, **kwargs: None,
        BN1=128,
        HID=5120,
        BK1=256,
        KL1_1=512,
        L2_CACHE_DISABLE=4,
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), filename, "exec"), namespace)
    owner = SimpleNamespace(
        m=tokens,
        p1_m=pm,
        proj=SimpleNamespace(m=tokens, p1_m=pm, p1_m_tiles=(tokens + pm - 1) // pm),
    )
    root_out = torch.full((tokens + 1, 128), -777.0)
    x = torch.zeros((tokens, 5120), dtype=torch.int8)
    weight = torch.zeros((128, 5120), dtype=torch.int8)
    sx = torch.zeros((tokens, 80, 2), dtype=torch.int8)
    sw = torch.zeros((128, 80, 2), dtype=torch.int8)
    for mi in range((tokens + pm - 1) // pm):
        for split in (0, 1):
            namespace[fn.name](
                owner, pipeline, root_out[:-1], x, weight, sx, sw, mi, 0, split
            )
    assert torch.all(root_out[-1] == -777)
    if tokens > 64 and tokens % 64:
        assert (tokens % 64, 128) in copies
        assert (tokens % 64, 512) in copies
        assert (tokens % 64, 80) in copies


@pytest.mark.parametrize(
    "ub_bytes,expected", [(253952, 16), (251648, 16), (251647, 0), (16384, 0)]
)
def test_review_softmax_budget_uses_ub_capacity(ub_bytes, expected):
    tree = ast.parse((ROOT / "mega_mla_kernel.py").read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MegaMlaKernel"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    selection = [
        n
        for n in init.body
        if isinstance(n, ast.Assign)
        and ast.unparse(n.targets[0])
        in ("softmax_workers", "softmax_row_tile", "self.softmax_batch_rows")
    ]
    state = SimpleNamespace(heads=64)
    namespace = dict(
        self=state,
        tokens=16,
        grid=32,
        k_attn=2176,
        get_mem_size=lambda _: ub_bytes,
        SOFTMAX_UB_RESERVE_BYTES=16 * 1024,
    )
    exec(
        compile(ast.Module(body=selection, type_ignores=[]), "softmax_budget", "exec"),
        namespace,
    )
    assert state.softmax_batch_rows == expected


@pytest.mark.npu
def test_review_stream_quota_launch_guard(monkeypatch):
    _dsl()
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    if not hasattr(torch.npu, "set_stream_limit"):
        pytest.skip("torch_npu stream resource limits unavailable")
    from mega_mla import MegaMlaOp, _function_context

    torch.npu.set_device(0)
    stream = torch.npu.Stream()
    quota = min(8, public.available_aic_count(torch.device("npu", 0)))
    torch.npu.synchronize()
    torch.npu.set_stream_limit(stream, cube_num=quota, vector_num=2 * quota)
    try:
        with torch.npu.stream(stream):
            hidden = torch.empty((1, 5120), dtype=torch.bfloat16, device="npu")
            assert public.available_aic_count(hidden.device) == quota
            assert _function_context(hidden, None)[2] == quota
            oversized = MegaMlaOp(grid=quota + 1, heads=8, tokens=1)
            with pytest.raises(ValueError, match="current stream AIC/AIV launch limit"):
                public.prepare_attention_arguments(oversized, hidden, *([None] * 16))
            assert oversized._workspace_storage is None
            _exercise_attention(
                8,
                8,
                False,
                monkeypatch,
                cmp_pages=42,
                win_pages=4,
                cmp_strategy="indexed",
                grid=quota,
            )
            torch.npu.synchronize()
    finally:
        torch.npu.synchronize()
        torch.npu.reset_stream_limit(stream)


# ============================================================================
# Operator-owned Native export and packaging checks
# ============================================================================


@lru_cache(maxsize=1)
def _mega_native_builder():
    """Use the shared packager without changing its implementation or tests."""
    import importlib.util

    path = Path(__file__).resolve().parents[3] / "packaging/native/build.py"
    spec = importlib.util.spec_from_file_location("mega_mla_packaging_checks", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def _mega_mla_native_namespace(monkeypatch):
    from numbers import Integral

    builder = _mega_native_builder()
    operators, _ = builder.load_catalog()
    source = builder.source_index()[operators["mega_mla"][0]]
    sample = source.parent
    monkeypatch.syspath_prepend(str(sample))
    import attention_geometry as geometry

    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for node in functions:
        node.decorator_list = []
    namespace = dict(
        os=os,
        json=json,
        Integral=Integral,
        validate_query_heads=geometry.validate_query_heads,
        validate_attention_geometry=geometry.validate_attention_geometry,
        choose_compressed_cache_decode_strategy=geometry.choose_compressed_cache_decode_strategy,
    )
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace


def test_mega_dependencies_are_complete():
    builder = _mega_native_builder()
    operators, groups = builder.load_catalog()
    files = builder.source_index()
    sample = files["mega_mla_native.py"].parent
    assert {Path(n).name for n in operators["mega_mla"]} == {
        p.name for p in sample.glob("*.py")
    }
    assert [Path(n).name for n in operators["mega_full_index"]] == [
        "mega_full_index.py",
        "prologue.py",
        "qli.py",
    ]
    assert (
        groups["ds41"].count("mega_mla") == groups["ds41"].count("mega_full_index") == 1
    )


@pytest.mark.ascendc_toolchain
def test_new_mega_native_discovery(tmp_path):
    pytest.importorskip(
        "torch_npu", reason="collector imports need torch_npu, not an NPU launch"
    )
    from cannbotdsl.aot.collection import discover

    builder = _mega_native_builder()
    operators, _ = builder.load_catalog()
    selected = ["mega_mla", "mega_full_index"]
    sources = tmp_path / "sources"
    builder.materialize_flat_sources(sources, selected, operators)
    with discover(collectors=sources, operators=selected) as tasks:
        assert list(tasks) == selected
        assert all(callable(callback) for callback in tasks.values())


def test_mega_mla_default_profiles(monkeypatch):
    from types import SimpleNamespace

    namespace = _mega_mla_native_namespace(monkeypatch)
    monkeypatch.delenv("CANNBOTDSL_DS41_PROFILES", raising=False)
    calls, closed = [], []

    def compile_profile(profile):
        calls.append(profile)
        return SimpleNamespace(close=lambda: closed.append(profile))

    namespace["_compile_native_profile"] = compile_profile
    namespace["export_mega_mla"]()
    assert len(calls) == len(closed) == 24
    assert {(p["heads"], p["grid"], p["k"], p["batches"]) for p in calls} == {
        (h, g, k, b)
        for h in (8, 16, 64)
        for g in (28, 32)
        for k in (640, 2176)
        for b in (0, 8)
    }
    assert all(p["tokens"] == 8 for p in calls)
    assert all(
        p["cmp_decode_strategy"] == ("indexed" if p["k"] == 640 else "predecode")
        for p in calls
    )


def test_mega_mla_profile_override(monkeypatch, tmp_path):
    namespace = _mega_mla_native_namespace(monkeypatch)
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "mega_mla": [
                    {
                        "grid": 7,
                        "compressed_page_size": 64,
                        "window_page_stride": 69696,
                        "compressed_page_stride": 20576,
                    },
                    {
                        "grid": 7,
                        "compressed_page_size": 64,
                        "window_page_stride": 69696,
                        "compressed_page_stride": 20576,
                    },
                ]
            }
        )
    )
    monkeypatch.setenv("CANNBOTDSL_DS41_PROFILES", str(path))
    profiles = namespace["_native_profiles"]()
    assert len(profiles) == 1
    assert profiles[0]["grid"] == 7 and profiles[0]["compressed_page_size"] == 64
    assert profiles[0]["cmp_decode_strategy"] == "predecode"
    path.write_text(json.dumps({"qli": []}))
    assert len(namespace["_native_profiles"]()) == 24


@pytest.mark.parametrize(
    "entry",
    [
        {"unknown": 1},
        {"grid": True},
        {"grid": 0},
        {"batches": -1},
        {"k": 704},
        {"window_page_stride": 1},
        {"heads": 32},
    ],
)
def test_mega_mla_reject_invalid_export_profiles(monkeypatch, tmp_path, entry):
    namespace = _mega_mla_native_namespace(monkeypatch)
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"mega_mla": [entry]}))
    monkeypatch.setenv("CANNBOTDSL_DS41_PROFILES", str(path))
    with pytest.raises(ValueError):
        namespace["_native_profiles"]()


@pytest.mark.parametrize("batches", [0, 2])
@pytest.mark.parametrize("strategy", ["indexed", "predecode"])
def test_mega_mla_export_tensor_contract(monkeypatch, tmp_path, batches, strategy):
    from types import SimpleNamespace

    namespace = _mega_mla_native_namespace(monkeypatch)
    from attention_workspace import AttentionWorkspaceLayout
    from attention_sequence_inputs import AttentionSequenceConfig

    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "mega_mla": [
                    dict(
                        batches=batches,
                        cmp_decode_strategy=strategy,
                        compressed_page_size=64,
                        window_page_stride=69696,
                        compressed_page_stride=20576,
                    )
                ]
            }
        )
    )
    monkeypatch.setenv("CANNBOTDSL_DS41_PROFILES", str(path))

    class Owner:
        gather_batch = 32

        def __init__(self, *args, **kwargs):
            self.args, self.options = args, kwargs

        def _launch_mega_mla(self):
            pass

        def _launch_mega_mla_with_seqlens(self):
            pass

    dt = SimpleNamespace(
        **{
            name: name
            for name in (
                "bfloat16",
                "float32",
                "float8_e4m3fn",
                "float8_e8m0",
                "uint8",
                "int32",
            )
        }
    )
    namespace.update(
        MegaMlaOp=Owner,
        MegaMlaKernel=lambda *a, **kw: SimpleNamespace(args=a, options=kw),
        AttentionWorkspaceLayout=AttentionWorkspaceLayout,
        AttentionSequenceConfig=AttentionSequenceConfig,
        AttentionSequenceFilter=lambda *a, **kw: SimpleNamespace(args=a, options=kw),
        dsl=SimpleNamespace(
            dtypes=dt,
            TensorSpec=lambda shape, dtype, **kw: SimpleNamespace(
                shape=shape, dtype=dtype, **kw
            ),
            compile=lambda entry, *specs: SimpleNamespace(entry=entry, specs=specs),
        ),
    )
    provider = namespace["_compile_native_profile"](namespace["_native_profiles"]()[0])
    specs, owner = provider.specs, provider.entry.__self__
    assert len(specs) == (31 if batches else 27)
    assert specs[9].shape == (4, 128, 1, 544) and specs[9].stride == (
        69696,
        544,
        544,
        1,
    )
    assert specs[10].shape == (42, 64, 1, 320) and specs[10].stride == (
        20576,
        320,
        320,
        1,
    )
    assert [i for i, spec in enumerate(specs) if spec.storage_format == "nz"] == [
        3,
        4,
        5,
    ]
    assert specs[21].shape == (8, 16, 512)
    layout = owner.kernel.options["workspace_layout"]
    assert specs[22].shape == (layout.nbytes // 4,)
    assert owner.kernel.options["cmp_decode_strategy"] == strategy
    assert owner.kernel.options["gather_batch"] == 32
    if batches:
        assert provider.entry.__name__ == "_launch_mega_mla_with_seqlens"
        assert specs[17].shape == (2, 4) and specs[18].shape == (2, 42)
        assert [s.shape for s in specs[27:]] == [(3,), (2,), (2,), (2,)]
    else:
        assert provider.entry.__name__ == "_launch_mega_mla"
        assert specs[17].shape == (8, 128) and specs[18].shape == (8, 512)


@pytest.mark.ascendc_toolchain
def test_native_wheel_contains_mega_sources_and_resources(tmp_path, monkeypatch):
    """Build a real wheel while substituting only the device compiler."""
    import zipfile
    import cannbotdsl.aot as aot

    builder = _mega_native_builder()
    operators, _ = builder.load_catalog()
    selected = ["mega_mla", "mega_full_index"]
    source_files = {
        Path(name).name: builder.source_index()[name]
        for name in builder.selected_filenames(selected, operators)
    }
    # Avoid leaking main()'s process-wide environment into other tests.
    monkeypatch.setattr(
        builder,
        "configure_intermediate_environment",
        lambda work: work.mkdir(parents=True),
    )
    monkeypatch.setattr(sys, "path", list(sys.path))

    def compile_native(**kwargs):
        assert kwargs["operators"] == selected
        assert kwargs["target"] == "dav-3510"
        for name, source in source_files.items():
            assert (kwargs["collectors"] / name).read_bytes() == source.read_bytes()
        output = kwargs["output"]
        for operator in selected:
            variant = output / "operators" / operator / "variants" / "key"
            variant.mkdir(parents=True)
            (variant / "kernel.so").write_bytes(b"test binary")
        (output / "manifest.json").write_text("{}")
        return output

    monkeypatch.setattr(aot, "build", compile_native)
    assert (
        builder.main(
            [
                "--operators",
                *selected,
                "--output-root",
                str(tmp_path / "out"),
                "--work-root",
                str(tmp_path / "work"),
            ]
        )
        == 0
    )
    (wheel,) = (tmp_path / "out" / "operators" / "__".join(selected) / "wheels").glob(
        "*.whl"
    )
    with zipfile.ZipFile(wheel) as archive:
        for name, source in source_files.items():
            assert archive.read("ops/" + name) == source.read_bytes()
        for operator in selected:
            assert (
                archive.read(f"ops/_native/operators/{operator}/variants/key/kernel.so")
                == b"test binary"
            )
        assert archive.read("ops/_native/manifest.json") == b"{}"
        assert b"from cannbotdsl.aot import register_directory" in archive.read(
            "ops/__init__.py"
        )
        (metadata,) = (
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        assert "Requires-Dist: opkit" not in archive.read(metadata).decode()
        assert "Requires-Dist: cannbot-dsl>=0.8.0" in archive.read(metadata).decode()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "benchmark": benchmark_attention,
        "benchmark-paged": benchmark_paged_attention,
        "analyze-profile": analyze_profile,
    }
    if argv and argv[0] in commands:
        previous = torch.get_num_threads()
        try:
            return commands[argv[0]](argv[1:])
        finally:
            if torch.get_num_threads() != previous:
                torch.set_num_threads(previous)
    if argv[:1] in (["-h"], ["--help"]):
        print(
            "Usage: test_mega_mla.py [test|benchmark|benchmark-paged|analyze-profile] [arguments]"
        )
        print(
            "test: pytest options; benchmark/benchmark-paged/analyze-profile: use --help"
        )
        return 0
    if argv[:1] == ["test"]:
        argv.pop(0)
    return pytest.main([str(Path(__file__).resolve()), *argv])


if __name__ == "__main__":
    raise SystemExit(main())
