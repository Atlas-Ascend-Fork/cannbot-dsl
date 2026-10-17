# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

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

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "samples" / "scatter_elements")
)
from scatter_elements import ScatterElementsTiling, scatter_elements


LOGGER = logging.getLogger(__name__)
DEVICE_TASK_TYPES = {
    "AI_VECTOR_CORE",
    "AI_CORE",
    "KERNEL_AIVEC",
    "KERNEL_AICORE",
    "MIX_AIC",
    "MIX_AIV",
}


def test_tiling_and_strides():
    tiling = ScatterElementsTiling((2, 3, 5), (2, 2, 4), (2, 3, 5), -2, "add")
    assert tiling.axis == 1
    assert tiling.data_numel == 30
    assert tiling.indices_numel == 16
    assert tiling.axis_size == 3
    assert tiling.data_strides == (15, 5, 1)
    assert tiling.index_strides == (8, 4, 1)
    assert tiling.update_strides == (15, 5, 1)
    assert len(tiling.stride_parameters) == 36
    assert tiling.stride_parameters[24:27] == (2, 2, 4)
    assert tiling.stride_parameters[32:36] == (3, 1, 3, 1)


@pytest.mark.parametrize(
    "data_shape,index_shape,update_shape,axis,reduction,match",
    [
        ((), (), (), 0, "none", "rank"),
        ((2, 3), (2,), (2, 3), 0, "none", "same rank"),
        ((2, 3), (3, 3), (3, 3), 1, "none", "data.shape"),
        ((2, 3), (2, 3), (2, 2), 1, "none", "updates.shape"),
        ((2, 3), (2, 3), (2, 3), 2, "none", "axis"),
        ((2, 3), (2, 3), (2, 3), 1, "mean", "reduction"),
        ((2, 0), (2, 1), (2, 1), 1, "none", "cannot be empty"),
    ],
)
def test_invalid_tiling(data_shape, index_shape, update_shape, axis, reduction, match):
    with pytest.raises(ValueError, match=match):
        ScatterElementsTiling(data_shape, index_shape, update_shape, axis, reduction)


@pytest.mark.parametrize("axis", [True, 1.5, "1"])
def test_invalid_axis_type(axis):
    with pytest.raises(TypeError, match="axis"):
        ScatterElementsTiling((2, 3), (2, 3), (2, 3), axis)


def test_empty_indices_is_valid():
    tiling = ScatterElementsTiling((2, 3), (0, 2), (1, 3), 1)
    assert tiling.indices_numel == 0
    assert tiling.data_numel == 6


def test_overflow():
    with pytest.raises(OverflowError, match="64-bit"):
        ScatterElementsTiling((2**62, 4), (0, 1), (0, 1), 1)


@pytest.fixture(scope="module")
def torch_npu_ready():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")


def reference(data, indices, updates, axis, reduction):
    output = data.clone()
    if not indices.numel():
        return output
    for coordinate in product(*(range(size) for size in indices.shape)):
        destination = list(coordinate)
        destination[axis] = int(indices[coordinate])
        destination = tuple(destination)
        if reduction == "none":
            output[destination] = updates[coordinate]
        elif reduction == "add":
            output[destination] = output[destination] + updates[coordinate]
        else:
            output[destination] = output[destination] * updates[coordinate]
    return output


@pytest.mark.npu
def test_none_preserves_bits(torch_npu_ready):
    bits = torch.tensor(
        [0x7FC01234, -2147483648, 0x7F800000, -8388608], dtype=torch.int32
    )
    data = bits.view(torch.float32)
    indices = torch.tensor([3, 1], dtype=torch.int64)
    update_bits = torch.tensor([0x7FA12345, -2147483648], dtype=torch.int32)
    updates = update_bits.view(torch.float32)
    actual = scatter_elements(
        data.npu(), indices.npu(), updates.npu(), 0, deterministic=True
    )
    expected = reference(data, indices, updates, 0, "none")
    assert torch.equal(actual.cpu().view(torch.int32), expected.view(torch.int32))


@pytest.mark.npu
@pytest.mark.parametrize("reduction", ["add", "mul"])
def test_int64_reduction_high_bits(torch_npu_ready, reduction):
    data = torch.tensor(
        [(1 << 63) - 1, -(1 << 63), -1, (1 << 32) - 1, 1 << 40],
        dtype=torch.int64,
    )
    indices = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=torch.int64)
    updates = torch.tensor(
        [-1, 1, (1 << 32) - 1, 2, -(1 << 63), -1, 3, 1 << 40, -1, 2],
        dtype=torch.int64,
    )
    expected = reference(data, indices, updates, 0, reduction)
    actual = scatter_elements(
        data.npu(), indices.npu(), updates.npu(), 0, reduction=reduction
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.npu
@pytest.mark.parametrize("dtype_name", ["float16"])
@pytest.mark.parametrize("index_name", ["int32"])
@pytest.mark.parametrize("reduction", ["mul"])
@pytest.mark.parametrize("deterministic", [True])
def test_fractional_negative_ordered_reduction(
    torch_npu_ready, dtype_name, index_name, reduction, deterministic
):
    """Fractional/negative payloads must not pass through integer loop state."""
    dtype = getattr(torch, dtype_name)
    data = (torch.arange(30).remainder(9).float() / 8 - 0.75).reshape(2, 3, 5).to(dtype)
    updates = (
        (torch.arange(30).remainder(7).float() / 8 - 0.5).reshape(2, 3, 5).to(dtype)
    )
    indices = torch.tensor(
        [[[1, 1, 0, 2], [1, 0, 0, 2]], [[0, 1, 2, 1], [0, 1, 2, 1]]],
        dtype=getattr(torch, index_name),
    )
    expected = reference(data, indices, updates, 1, reduction)
    actual = scatter_elements(
        data.npu(),
        indices.npu(),
        updates.npu(),
        1,
        reduction=reduction,
        deterministic=deterministic,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.npu
@pytest.mark.parametrize("index_name", ["int64"])
@pytest.mark.parametrize("deterministic", [True])
def test_axis0_float32_fractional_add(torch_npu_ready, index_name, deterministic):
    """Cover deterministic and atomic addition with repeated fractional updates."""
    shape = (10, 17, 29)
    generator = torch.Generator().manual_seed(2701)
    data = torch.randint(-32, 33, shape, generator=generator).float() / 16
    updates = torch.randint(-16, 17, shape, generator=generator).float() / 16
    indices = torch.randint(
        0, 10, shape, generator=generator, dtype=getattr(torch, index_name)
    )
    expected = reference(data, indices, updates, 0, "add")
    actual = scatter_elements(
        data.npu(),
        indices.npu(),
        updates.npu(),
        0,
        reduction="add",
        deterministic=deterministic,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


# data shape, index shape, update shape, data dtype, index dtype, axis,
# reduction and reproducible seed. Indices are generated within axis bounds.
_CASES = [
    pytest.param(
        (10, 38, 26),
        (10, 18, 26),
        (10, 28, 96),
        torch.int32,
        torch.int32,
        1,
        "none",
        2001,
        id="scatter_elements_determ_none_008",
    ),
    pytest.param(
        (1, 3, 8, 16, 5, 3, 4),
        (1, 3, 8, 16, 5, 3, 4),
        (1, 3, 8, 16, 5, 3, 4),
        torch.bfloat16,
        torch.int32,
        2,
        "none",
        2002,
        id="scatter_elements_none_027",
    ),
    pytest.param(
        (10, 2, 18, 26, 5),
        (10, 2, 18, 26, 5),
        (10, 2, 18, 26, 5),
        torch.float32,
        torch.int64,
        0,
        "add",
        2003,
        id="scatter_elements_add_014",
    ),
    pytest.param(
        (5, 9, 15, 4, 6, 6, 11),
        (1, 1, 1, 1, 1, 1, 2),
        (7, 3, 12, 2, 4, 12, 9),
        torch.float16,
        torch.int64,
        -2,
        "none",
        2004,
        id="aclnnScatter_fp16_int64_nd_infnan_random_000016",
    ),
    pytest.param(
        (672, 128),
        (672, 97),
        (672, 97),
        torch.int32,
        torch.int64,
        1,
        "none",
        2005,
        id="ScatterElements_int64_int32_ND_fuzz_000075",
    ),
]


def _check_precision_result(data, indices, updates, axis, reduction, actual):
    axis %= data.ndim
    assert bool(((indices >= 0) & (indices < data.shape[axis])).all())
    assert actual.shape == data.shape and actual.dtype == data.dtype
    if reduction == "none":
        word = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
            data.element_size()
        ]
        raw_data, raw_updates, raw_actual = [
            value.view(word) for value in (data, updates, actual)
        ]
        choices = {}
        for coordinate in product(*(range(size) for size in indices.shape)):
            destination = list(coordinate)
            destination[axis] = int(indices[coordinate])
            choices.setdefault(tuple(destination), set()).add(
                int(raw_updates[coordinate])
            )
        mask = torch.ones(data.shape, dtype=torch.bool)
        for destination, legal_values in choices.items():
            mask[destination] = False
            assert int(raw_actual[destination]) in legal_values
        assert torch.equal(raw_actual[mask], raw_data[mask]), "Untouched bits changed"
    else:
        expected = reference(data, indices, updates, axis, reduction)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.npu
@pytest.mark.parametrize(
    "shape,index_shape,update_shape,dtype,index_dtype,axis,reduction,seed", _CASES
)
def test_functional_precision_case(
    torch_npu_ready,
    tmp_path,
    request,
    shape,
    index_shape,
    update_shape,
    dtype,
    index_dtype,
    axis,
    reduction,
    seed,
):
    generator = torch.Generator().manual_seed(seed)
    data = _make_precision_values(shape, dtype, generator)
    updates = _make_precision_values(update_shape, dtype, generator)
    if dtype.is_floating_point and reduction == "none":
        for value in (data, updates):
            value.reshape(-1)[:4] = torch.tensor(
                [float("nan"), float("inf"), -float("inf"), -0.0], dtype=dtype
            )
    indices = torch.randint(
        0, shape[axis], index_shape, generator=generator, dtype=index_dtype
    )
    device = [value.npu() for value in (data, indices, updates)]

    def execute():
        return scatter_elements(*device, axis, reduction=reduction, deterministic=False)

    actual = execute().cpu()
    _check_precision_result(data, indices, updates, axis, reduction, actual)
    _profile_precision_case(request.node.callspec.id, execute, tmp_path)


def _make_precision_values(shape, dtype, generator):
    """Generate values in memory; preserve GatherV2's uint16 raw-bit inputs."""
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
