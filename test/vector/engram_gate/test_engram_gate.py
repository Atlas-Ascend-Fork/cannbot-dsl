# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch

from _samples_path import load_sample


_module = load_sample("vector/engram_gate/engram_gate.py")
EngramGate = _module.EngramGate
EngramGateKernel = _module.EngramGateKernel
engram_gate = _module.engram_gate


def _golden(x, key, value, weight, eps, clamp_value, image_mask=None):
    x_f = x.float()
    rstd = torch.rsqrt(x_f.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (x_f * weight * key).sum(-1) * rstd * x.shape[-1] ** -0.5
    gate = torch.sigmoid(
        torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot)
    )
    if image_mask is not None:
        gate = gate.masked_fill(image_mask.unsqueeze(-1), 0)
    return (x_f + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


def _inputs(t, hc, dim, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(t, hc, dim, generator=generator).to(torch.bfloat16)
    key = torch.randn(t, hc, dim, generator=generator).to(torch.bfloat16)
    value = torch.randn(t, dim, generator=generator).to(torch.bfloat16)
    weight = torch.randn(hc, dim, generator=generator, dtype=torch.float32)
    return x, key, value, weight


def test_engram_gate_compile_default_shape():
    pytest.importorskip("cannbotdsl")
    import cannbotdsl

    try:
        import torch_npu  # noqa: F401

        npu_available = torch.npu.is_available()
    except ImportError:
        npu_available = False
    if npu_available:
        max_blocks = torch.npu.get_device_properties(0).vector_core_num
    else:
        max_blocks = 8
    cannbotdsl.compile(
        EngramGate(eps=1e-6, clamp_value=1e-6, max_blocks=max_blocks).run,
        cannbotdsl.TensorSpec((2, 4, 5120), cannbotdsl.dtypes.bfloat16),
        cannbotdsl.TensorSpec((2, 4, 5120), cannbotdsl.dtypes.bfloat16),
        cannbotdsl.TensorSpec((2, 5120), cannbotdsl.dtypes.bfloat16),
        cannbotdsl.TensorSpec((4, 5120), cannbotdsl.dtypes.float32),
        cannbotdsl.TensorSpec((2, 4, 5120), cannbotdsl.dtypes.bfloat16),
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "t,mask_kind,eps,clamp_value",
    [
        pytest.param(5, "partial", 1e-6, 1e-6, id="partial_image_mask"),
    ],
)
def test_engram_gate_npu_precision(t, mask_kind, eps, clamp_value):
    pytest.importorskip("torch_npu")
    x, key, value, weight = _inputs(t, 4, 5120, 20260920 + t)
    image_mask = None
    if mask_kind == "partial":
        image_mask = torch.arange(t) % 2 == 1
    elif mask_kind == "all":
        image_mask = torch.ones(t, dtype=torch.bool)
    expected = _golden(x, key, value, weight, eps, clamp_value, image_mask)
    actual = engram_gate(
        x.npu(),
        key.npu(),
        value.npu(),
        weight.npu(),
        None if image_mask is None else image_mask.npu(),
        eps=eps,
        clamp_value=clamp_value,
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    max_error = (actual_cpu.float() - expected.float()).abs().max().item()
    assert torch.allclose(
        actual_cpu.float(), expected.float(), atol=2e-2, rtol=2e-2
    ), f"max absolute error: {max_error}"
    if image_mask is not None:
        assert torch.equal(actual_cpu[image_mask], x[image_mask])


@pytest.mark.npu
@pytest.mark.parametrize(
    "t,hc,dim,mask_kind",
    [
        pytest.param(72, 4, 5120, "none", id="model_token_count_nomask"),
    ],
)
def test_engram_gate_npu_precision_pair_pipeline(t, hc, dim, mask_kind):
    pytest.importorskip("torch_npu")
    x, key, value, weight = _inputs(t, hc, dim, 20260920 + t)
    image_mask = None
    if mask_kind == "partial":
        image_mask = torch.arange(t) % 2 == 1
    expected = _golden(x, key, value, weight, 1e-6, 1e-6, image_mask)
    actual = engram_gate(
        x.npu(),
        key.npu(),
        value.npu(),
        weight.npu(),
        None if image_mask is None else image_mask.npu(),
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    max_error = (actual_cpu.float() - expected.float()).abs().max().item()
    assert torch.allclose(
        actual_cpu.float(), expected.float(), atol=2e-2, rtol=2e-2
    ), f"max absolute error: {max_error}"
    if image_mask is not None:
        assert torch.equal(actual_cpu[image_mask], x[image_mask])


@pytest.mark.npu
@pytest.mark.parametrize(
    "t,hc,dim,mask_kind",
    [
        pytest.param(3, 4, 8449, "none", id="chunked_one_chunk_tail_one"),
    ],
)
def test_engram_gate_npu_precision_large_dim(t, hc, dim, mask_kind):
    pytest.importorskip("torch_npu")
    x, key, value, weight = _inputs(t, hc, dim, 20260923 + dim % 10000)
    image_mask = None
    if mask_kind == "partial":
        image_mask = torch.arange(t) % 2 == 1
    expected = _golden(x, key, value, weight, 1e-6, 1e-6, image_mask)
    actual = engram_gate(
        x.npu(),
        key.npu(),
        value.npu(),
        weight.npu(),
        None if image_mask is None else image_mask.npu(),
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    max_error = (actual_cpu.float() - expected.float()).abs().max().item()
    assert torch.allclose(
        actual_cpu.float(), expected.float(), atol=2e-2, rtol=2e-2
    ), f"max absolute error: {max_error}"
    if image_mask is not None:
        assert torch.equal(actual_cpu[image_mask], x[image_mask])


@pytest.mark.npu
@pytest.mark.parametrize(
    "t,hc,dim,mask_kind",
    [
        pytest.param(5, 4, 5121, "partial", id="model_dim_plus_one_partial_mask"),
    ],
)
def test_engram_gate_npu_precision_arbitrary_dim(t, hc, dim, mask_kind):
    pytest.importorskip("torch_npu")
    x, key, value, weight = _inputs(t, hc, dim, 20260920 + dim)
    image_mask = None
    if mask_kind == "partial":
        image_mask = torch.arange(t) % 2 == 1
    expected = _golden(x, key, value, weight, 1e-6, 1e-6, image_mask)
    actual = engram_gate(
        x.npu(),
        key.npu(),
        value.npu(),
        weight.npu(),
        None if image_mask is None else image_mask.npu(),
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    max_error = (actual_cpu.float() - expected.float()).abs().max().item()
    assert torch.allclose(
        actual_cpu.float(), expected.float(), atol=2e-2, rtol=2e-2
    ), f"max absolute error: {max_error}"
    if image_mask is not None:
        assert torch.equal(actual_cpu[image_mask], x[image_mask])
