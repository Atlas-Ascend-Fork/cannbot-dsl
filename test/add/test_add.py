# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Ten functional DMA Add cases: float32/int32 across five shapes and storage offsets."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "samples/add"))
import add as _MODULE

# Fixed independent seeds make both input streams reproducible.
BASE_SEED = 20261023
INPUT2_SEED = BASE_SEED + 1
INTEGER_RANDOM_BOUND = 100
OFFSET_CASE_ELEMENTS = 1027  # Non-aligned final length after many 32-byte spans.
MULTI_TILE_ELEMENTS = 128 * 1024 + 7  # Multiple UB tiles per core, with an uneven tail.
DMA_TEST_CORES = 2  # Force enough work per core to exercise repeated UB slot reuse.
DMA_ALIGNMENT_BYTES = 32
INPUT_GUARD_ELEMENTS = 8  # Extra backing storage after the input view; excluded from its logical shape.

DTYPES = [pytest.param(torch.float32, id="fp32"), pytest.param(torch.int32, id="int32")]
CASES = [
    pytest.param((), (0, 0), None, id="scalar"),
    pytest.param((31,), (0, 0), None, id="small-tail"),
    pytest.param((3, 5, 17), (0, 0), None, id="multidimensional"),
    pytest.param((OFFSET_CASE_ELEMENTS,), (1, 7), None, id="offset-inputs"),
    pytest.param((MULTI_TILE_ELEMENTS,), (1, 7), DMA_TEST_CORES, id="multiple-tiles-offset-inputs"),
]


def make_inputs(shape, dtype):
    """Independent CPU input generation, including IEEE values and integer overflow."""
    input1_generator = torch.Generator().manual_seed(BASE_SEED)
    input2_generator = torch.Generator().manual_seed(INPUT2_SEED)
    if dtype == torch.float32:
        input1 = torch.randn(shape, generator=input1_generator)
        input2 = torch.randn(shape, generator=input2_generator)
        info = torch.finfo(dtype)
        subnormal = info.tiny * info.eps
        motif1 = [-0., 0., float("inf"), -float("inf"), float("nan"), subnormal, -subnormal, info.max]
        motif2 = [-0., -0., -float("inf"), float("inf"), 1., subnormal, subnormal, info.max]
    else:
        input1 = torch.randint(
            -INTEGER_RANDOM_BOUND, INTEGER_RANDOM_BOUND, shape, dtype=dtype, generator=input1_generator)
        input2 = torch.randint(
            -INTEGER_RANDOM_BOUND, INTEGER_RANDOM_BOUND, shape, dtype=dtype, generator=input2_generator)
        info = torch.iinfo(dtype)
        motif1 = [info.max, info.min, info.max, 1, 0, -1, 7]
        motif2 = [1, -1, info.max, info.max, 0, 1, -9]
    count = min(input1.numel(), len(motif1))
    input1.view(-1)[:count] = torch.tensor(motif1[:count], dtype=dtype)
    input2.view(-1)[:count] = torch.tensor(motif2[:count], dtype=dtype)
    return input1, input2


@pytest.fixture(scope="module")
def npu_device():
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    return torch.device("npu:0")


def upload_input(value, device, offset):
    if offset == 0:
        return value.to(device)
    backing = torch.empty(value.numel() + offset + INPUT_GUARD_ELEMENTS, dtype=value.dtype, device=device)
    result = backing[offset:offset + value.numel()].view(value.shape)
    result.copy_(value)
    assert result.is_contiguous() and result.data_ptr() % DMA_ALIGNMENT_BYTES != 0
    return result


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,input_offsets,core_num", CASES)
def test_add_functional(npu_device, dtype, shape, input_offsets, core_num):
    input1_cpu, input2_cpu = make_inputs(shape, dtype)
    expected = input1_cpu + input2_cpu
    input1 = upload_input(input1_cpu, npu_device, input_offsets[0])
    input2 = upload_input(input2_cpu, npu_device, input_offsets[1])
    output = _MODULE.add(input1, input2, core_num=core_num)
    torch.npu.synchronize()
    actual = output.cpu()
    assert output.shape == input1.shape and output.dtype == dtype and output.device == npu_device
    assert output.data_ptr() not in (input1.data_ptr(), input2.data_ptr())
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    if dtype == torch.float32:
        zeros = expected == 0
        assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))
    torch.testing.assert_close(input1.cpu(), input1_cpu, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(input2.cpu(), input2_cpu, atol=0, rtol=0, equal_nan=True)
    # Inspect the public tiling plan without accessing or patching the private compile cache.
    tiling = _MODULE.AddTiling.build(output.view(-1), core_num=core_num)
    assert tiling.element_count == expected.numel() and tiling.tile_elements > 0
    assert tiling.tile_elements % tiling.dma_alignment_elements == 0
    if core_num is not None:
        assert tiling.core_count == core_num
        assert tiling.element_count > tiling.core_count * tiling.tile_elements
