# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host validation and NPU bit-exact tests for the pure DSL GatherV2."""

import csv
import datetime
import json
import logging
import os
import sys
import tempfile
from itertools import product
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "samples" / "gather_v2"))
from gather_v2 import GatherV2Tiling, gather_v2


LOGGER = logging.getLogger(__name__)
DEVICE_TASK_TYPES = {
    "AI_VECTOR_CORE",
    "AI_CORE",
    "KERNEL_AIVEC",
    "KERNEL_AICORE",
    "MIX_AIC",
    "MIX_AIV",
}


def _word_dtype(element_bytes):
    return {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
        element_bytes
    ]


def _reference(x, indices, axis, batch_dims=0, negative=False):
    axis %= x.ndim
    if batch_dims < 0:
        batch_dims += indices.ndim
    word_dtype = _word_dtype(x.element_size())
    raw_x = x.view(word_dtype)
    inner_axis_start = axis + 1
    shape = (
        tuple(x.shape[:axis])
        + tuple(indices.shape[batch_dims:])
        + tuple(x.shape[inner_axis_start:])
    )
    output = torch.zeros(shape, dtype=word_dtype)
    if not output.numel() or not x.shape[axis]:
        return output.view(x.dtype)
    for batch in product(*(range(n) for n in x.shape[:batch_dims])):
        selected = indices[batch].long().flatten()
        if negative:
            selected = torch.where(selected < 0, selected + x.shape[axis], selected)
        valid = (selected >= 0) & (selected < x.shape[axis])
        safe = selected.clamp(0, x.shape[axis] - 1)
        chunk = torch.index_select(raw_x[batch], axis - batch_dims, safe)
        mask_shape = [1] * chunk.ndim
        mask_shape[axis - batch_dims] = selected.numel()
        chunk = torch.where(valid.reshape(mask_shape), chunk, torch.zeros_like(chunk))
        output[batch] = chunk.reshape(output[batch].shape)
    return output.view(x.dtype)


@pytest.fixture(scope="module")
def npu_ready():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    return torch.device("npu:0")


def _assert_npu_exact(x, indices, axis, device, **kwargs):
    expected = _reference(
        x,
        indices,
        axis,
        kwargs.get("batch_dims", 0),
        kwargs.get("negative_index_support", False),
    )
    actual = gather_v2(x.to(device), indices.to(device), axis, **kwargs).cpu()
    assert actual.shape == expected.shape
    dtype = _word_dtype(x.element_size())
    torch.testing.assert_close(actual.view(dtype), expected.view(dtype), rtol=0, atol=0)


def test_host_batch_shape_and_thread_cap():
    t = GatherV2Tiling(
        (2, 3, 4, 5),
        (2, 7, 9),
        -2,
        batch_dims=-2,
        core_num=50,
        thread_num=2048,
    )
    assert (t.axis, t.batch_dims) == (2, 1)
    assert t.output_shape == (2, 3, 7, 9, 5)
    assert (t.batch_size, t.outer_size, t.gather_size, t.inner_size) == (2, 3, 63, 5)
    assert t.requested_thread_num == 2048
    assert t.thread_num == 1024
    assert 1 <= t.core_num <= 50


@pytest.mark.parametrize(
    "x_shape,indices_shape,axis,batch_dims,output_shape",
    [
        ((3, 4), (), 0, 0, (4,)),
        ((3,), (), 0, 0, ()),
        ((2, 3), (2,), 1, 1, (2,)),
        ((2, 0, 5), (3,), 1, 0, (2, 3, 5)),
    ],
)
def test_host_output_shape(x_shape, indices_shape, axis, batch_dims, output_shape):
    t = GatherV2Tiling(x_shape, indices_shape, axis, batch_dims=batch_dims)
    assert t.output_shape == output_shape


@pytest.mark.npu
@pytest.mark.parametrize(
    "x,indices,axis,batch_dims,negative,thread_num",
    [
        pytest.param(
            torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
            torch.tensor([[2, 0], [1, -1]], dtype=torch.int64),
            1,
            1,
            True,
            32,
            id="batch_negative_indices",
        ),
        pytest.param(
            torch.empty((2, 0, 5), dtype=torch.int8),
            torch.tensor([0, -1, 2], dtype=torch.int64),
            1,
            0,
            False,
            32,
            id="empty_gather_axis",
        ),
    ],
)
def test_shapes_indices_and_bounds(
    npu_ready, x, indices, axis, batch_dims, negative, thread_num
):
    _assert_npu_exact(
        x,
        indices,
        axis,
        npu_ready,
        batch_dims=batch_dims,
        negative_index_support=negative,
        thread_num=thread_num,
    )


@pytest.mark.npu
@pytest.mark.parametrize("dtype", [torch.float64])
def test_64bit_high_word_and_nan_payloads(npu_ready, dtype):
    def signed_word(bits):
        return bits - (1 << 64) if bits >= (1 << 63) else bits

    if dtype is torch.int64:
        x = torch.tensor(
            [1 << 40, -(1 << 48) + 17, 0, (1 << 63) - 1, -(1 << 63)],
            dtype=torch.int64,
        )
    elif dtype is torch.float64:
        words = [
            0x7FF80000DEADBEEF,  # qNaN with payload in both 32-bit halves
            0x3FF0000000000000,  # +1.0
            0x8000000000000000,  # -0.0
            0x7FEFFFFFFFFFFFFF,  # max finite
            0xFFF8000000000001,  # negative qNaN
        ]
        x = torch.tensor([signed_word(word) for word in words], dtype=torch.int64)
        x = x.view(torch.float64)
    else:
        words = [
            0x400000003F800000,  # 1+2j, nonzero high half
            0xC080000040400000,  # 3-4j
            0x7FC123453F800000,  # imaginary NaN payload
            0x3F8000007FC54321,  # real NaN payload
            0xBF80000000000000,  # 0-1j
        ]
        x = torch.tensor([signed_word(word) for word in words], dtype=torch.int64)
        x = x.view(torch.complex64)
    x = x.reshape(5, 1)
    indices = torch.tensor([4, 2, 0, 5, 1, 3], dtype=torch.int64)
    _assert_npu_exact(x, indices, 0, npu_ready)


@pytest.mark.npu
@pytest.mark.parametrize("core_num", [2])
@pytest.mark.parametrize("index_dtype", [torch.int64])
def test_dma_simd_cache_repeated_rows_and_eviction(npu_ready, core_num, index_dtype):
    # Raw float32 words include signed zero and NaN payloads; every output
    # bit must survive UB reuse. More than three keys forces cache eviction.
    words = torch.arange(37 * 1024, dtype=torch.int32).reshape(37, 1024)
    words[:, ::11] = 0x7FC12345
    words[:, ::13] = -2147483648
    x = words.view(torch.float32)
    pattern = torch.tensor(
        [0, 1, 2, 0, 2, 1, -1, 37, -38, 36, 3, 4, 5, 0], dtype=index_dtype
    )
    indices = pattern.repeat(75)[:1043]
    _assert_npu_exact(x, indices, 0, npu_ready, core_num=core_num)


# Flat parameters follow test/flash_attn/test_flash_attn.py.
# input shape, index shape, value dtype, index dtype, axis, batch dims,
# negative-index support, number of invalid indices, reproducible seed.
_CASES = [
    pytest.param(
        (1, 135, 1, 1),
        (1, 1, 2, 199),
        torch.uint16,
        torch.int64,
        1,
        0,
        True,
        0,
        1001,
        id="gather_v2_0307",
    ),
    pytest.param(
        (30522, 1024),
        (12288,),
        torch.float32,
        torch.int32,
        0,
        0,
        False,
        5195,
        1002,
        id="gather_v2_0016",
    ),
    pytest.param(
        (572, 100),
        (50,),
        torch.bool,
        torch.int64,
        0,
        0,
        False,
        14,
        1003,
        id="gather_v2_0052",
    ),
    pytest.param(
        (128, 1, 200, 47, 163, 1),
        (128, 1, 1),
        torch.int32,
        torch.int32,
        4,
        2,
        False,
        61,
        1004,
        id="gather_v2_0343",
    ),
    pytest.param(
        (1024, 1),
        (256,),
        torch.float32,
        torch.int32,
        0,
        0,
        False,
        112,
        1005,
        id="gather_v2_0003",
    ),
    pytest.param(
        (9, 6, 5, 6, 9, 5),
        (1,),
        torch.int32,
        torch.int32,
        5,
        0,
        True,
        0,
        1006,
        id="gather_v2_test_case_0",
    ),
    pytest.param(
        (1, 1, 1, 1),
        (1,),
        torch.float16,
        torch.int64,
        -1,
        0,
        True,
        0,
        1007,
        id="gather_v2_test_case_1",
    ),
    pytest.param(
        (1, 1),
        (17,),
        torch.float32,
        torch.int32,
        -2,
        0,
        True,
        0,
        1008,
        id="gather_v2_test_case_2",
    ),
    pytest.param(
        (18, 1),
        (16,),
        torch.float32,
        torch.int64,
        -2,
        0,
        True,
        0,
        1009,
        id="gather_v2_test_case_3",
    ),
    pytest.param(
        (5, 8),
        (5,),
        torch.float16,
        torch.int32,
        1,
        0,
        True,
        0,
        1010,
        id="gather_v2_test_case_4",
    ),
]


@pytest.mark.npu
@pytest.mark.parametrize(
    "shape,index_shape,dtype,index_dtype,axis,batch_dims,negative,invalid_count,seed",
    _CASES,
)
def test_functional_precision_case(
    npu_ready,
    tmp_path,
    request,
    shape,
    index_shape,
    dtype,
    index_dtype,
    axis,
    batch_dims,
    negative,
    invalid_count,
    seed,
):
    generator = torch.Generator().manual_seed(seed)
    if dtype == torch.bool:
        x = torch.randint(0, 2, shape, generator=generator, dtype=torch.int8).bool()
    else:
        x = _make_precision_values(shape, dtype, generator)
    axis_size = shape[axis]
    indices = torch.randint(
        0, axis_size, index_shape, generator=generator, dtype=index_dtype
    )
    flat = indices.reshape(-1)
    if negative:
        flat[invalid_count::7] -= axis_size
    flat[:invalid_count:2] = axis_size
    flat[1:invalid_count:2] = -axis_size - 1
    normalized = (
        torch.where(indices < 0, indices + axis_size, indices) if negative else indices
    )
    assert int(((normalized < 0) | (normalized >= axis_size)).sum()) == invalid_count
    expected = _reference(x, indices, axis, batch_dims, negative)
    device_x = x.view(_word_dtype(x.element_size())).to(npu_ready).view(dtype)
    device_indices = indices.to(npu_ready)

    def execute():
        return gather_v2(
            device_x,
            device_indices,
            axis,
            batch_dims=batch_dims,
            negative_index_support=negative,
        )

    actual = execute().cpu()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )
    _profile_precision_case(request.node.callspec.id, execute, tmp_path)


@pytest.mark.npu
@pytest.mark.parametrize(
    "rows,inner,dtype,index_dtype,negative", [(50, 99, torch.int8, torch.int64, True)]
)
def test_short_byte_rows_tail_and_extreme_indices(
    npu_ready, rows, inner, dtype, index_dtype, negative
):
    """Cover packed tails, raw bytes, large indices and dispatch boundaries."""
    axis_size = 17
    values = torch.arange(axis_size * inner, dtype=torch.int64).reshape(
        axis_size, inner
    )
    x = ((values % 3) == 0) if dtype == torch.bool else values.to(dtype)
    limits = torch.iinfo(index_dtype)
    selected = [
        0,
        axis_size - 1,
        axis_size,
        -1,
        -axis_size,
        -axis_size - 1,
        limits.max,
        limits.min,
    ]
    if index_dtype == torch.int64:
        # Narrowing before checking would incorrectly turn these into 0/3.
        selected.extend([1 << 32, (1 << 32) + 3, -(1 << 32) + 3])
    indices = torch.tensor(
        [selected[i % len(selected)] for i in range(rows)], dtype=index_dtype
    )
    _assert_npu_exact(x, indices, 0, npu_ready, negative_index_support=negative)


def _make_precision_values(shape, dtype, generator):
    """Generate values in memory; preserve GatherV2's uint16 raw-bit inputs."""
    if dtype == torch.uint16:
        return torch.randint(-7, 8, shape, generator=generator, dtype=torch.int16).view(
            dtype
        )
    if dtype.is_floating_point:
        return (
            torch.randint(-7, 8, shape, generator=generator, dtype=torch.int32).float()
            / 8
        ).to(dtype)
    return torch.randint(-7, 8, shape, generator=generator, dtype=dtype)


def _device_durations(destination):
    """Read device durations from the available profiler CSV format."""
    candidates = list(destination.rglob("kernel_details.csv"))
    if candidates:
        evidence = max(candidates, key=lambda path: path.stat().st_mtime)
        with evidence.open(encoding="utf-8", newline="") as stream:
            records = list(csv.DictReader(stream))
        assert records, "Profiler returned no kernel samples"
        key = None
        for field in records[0]:
            if field.strip().lower() in (
                "duration(us)",
                "duration(μs)",
                "duration(µs)",
                "duration",
            ):
                key = field
                break
        assert key is not None, "Profiler returned no duration column"
        durations = [float(row[key]) for row in records]
    else:
        candidates = list(destination.rglob("op_summary*.csv"))
        assert candidates, "No NPU device profiling CSV generated"
        evidence = max(candidates, key=lambda path: path.stat().st_mtime)
        with evidence.open(encoding="utf-8", newline="") as stream:
            durations = [
                float(row["Task Duration(us)"])
                for row in csv.DictReader(stream)
                if row.get("Task Type") in DEVICE_TASK_TYPES
            ]
    assert durations and all((value > 0 for value in durations))
    return (durations, evidence)


def _profile_precision_case(name, execute, tmp_path):
    """Warm up once and collect two device executions after functional validation."""
    import torch_npu

    output_root = Path(os.environ.get("CANNBOT_CASE_RESULTS", str(tmp_path)))
    destination = output_root / name
    destination.mkdir(parents=True, exist_ok=True)
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(destination)),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        for _ in range(3):
            output = execute()
            torch.npu.synchronize()
            profiler.step()
    durations, evidence = _device_durations(destination)
    result = {
        "testcase_name": name,
        "dsl_us": sum(durations) / 2,
        "functional_validation": "PASS",
        "effective_samples": 2,
        "device_duration_samples_us": durations,
        "profiling_csv": str(evidence),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
    }
    (destination / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    LOGGER.info("%s: PASS, DSL=%.3f us", name, result["dsl_us"])


def _run_tests():
    """Run one operator script and summarize its measured cases in a temporary directory."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    script = Path(__file__).resolve()
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output = (
        Path(tempfile.gettempdir())
        / "cannbot_operator_tests"
        / (script.parent.name + "_" + timestamp)
    )
    os.environ.setdefault("CANNBOT_CASE_RESULTS", str(output))
    result_dir = Path(os.environ["CANNBOT_CASE_RESULTS"])
    result_dir.mkdir(parents=True, exist_ok=True)
    status = pytest.main([str(script), "-q", "-s", *sys.argv[1:]])
    results = []
    for case in _CASES:
        path = result_dir / case.id / "result.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
    if results:
        fields = [
            "testcase_name",
            "dsl_us",
            "functional_validation",
            "effective_samples",
        ]
        with (result_dir / "performance.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        LOGGER.info(
            "Measured %s/%s cases. Results: %s",
            len(results),
            len(_CASES),
            result_dir / "performance.csv",
        )
    return status


if __name__ == "__main__":
    raise SystemExit(_run_tests())
