# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MoonEP golden 自测。

纯 torch CPU 验证 golden 自身满足语义契约（均衡、往返一致、cu_seqlens 一致），
随仓库 CI 常跑；DSL 实现的 NPU 验收见 test_moonep_ep.py（需 torchrun 拉起 HCCL 组）。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "samples", "moonep")
)

from moonep_golden import (  # noqa: E402
    RouteShape,
    make_router_inputs,
    moonep_combine_golden,
    moonep_dispatch_golden,
    plan_redundant_experts,
)

# 形状参数：RouteShape(world_size, num_tokens, top_k, num_experts)。
# NvS = S*K；num_experts 须为 world_size 的整数倍。
_CASES = [
    pytest.param(RouteShape(2, 4, 1, 4), "balanced", 42, id="r2s4k1e4_balanced"),
    pytest.param(RouteShape(4, 16, 2, 8), "balanced", 43, id="r4s16k2e8_balanced"),
    pytest.param(RouteShape(4, 16, 2, 8), "skewed", 44, id="r4s16k2e8_skewed"),
    pytest.param(RouteShape(8, 32, 8, 32), "balanced", 45, id="r8s32k8e32_balanced"),
    pytest.param(RouteShape(8, 32, 8, 32), "skewed", 46, id="r8s32k8e32_skewed"),
    pytest.param(RouteShape(8, 8, 4, 16), "extreme", 47, id="r8s8k4e16_extreme"),
    pytest.param(RouteShape(2, 1, 1, 2), "balanced", 48, id="r2s1k1e2_minimal"),
]

_H = 32  # golden 自测用 hidden size，取小值保证 CPU 速度


def _make_case_inputs(shape, mode, seed, hidden_size=_H):
    gen = torch.Generator().manual_seed(seed + 1)
    hidden = torch.randn(
        shape.world_size,
        shape.num_tokens,
        hidden_size,
        generator=gen,
        dtype=torch.float32,
    )
    topk, rw = make_router_inputs(shape, mode=mode, seed=seed)
    return hidden, rw, topk


@pytest.mark.parametrize("shape,mode,seed", _CASES)
def test_golden_dispatch_balance(shape, mode, seed):
    """完美均衡：每个 rank 恰好接收 S*K 个 token，与路由倾斜程度无关。"""
    hidden, rw, topk = _make_case_inputs(shape, mode, seed)
    results = moonep_dispatch_golden(hidden, rw, topk, shape.num_experts)
    for r, res in enumerate(results):
        assert res["hidden"].shape == (shape.num_tokens * shape.top_k, _H), (
            f"rank {r} shape"
        )
        assert res["rw"].shape == (shape.num_tokens * shape.top_k,), (
            f"rank {r} rw shape"
        )
        assert len(res["meta"]) == shape.num_tokens * shape.top_k, f"rank {r} meta"


@pytest.mark.parametrize("shape,mode,seed", _CASES)
def test_golden_cu_seqlens_consistency(shape, mode, seed):
    """cu_seqlens：单调不降、末值等于 S*K、长度为 2*epn。"""
    _, _, topk = _make_case_inputs(shape, mode, seed)
    demand = torch.zeros(shape.num_experts, dtype=torch.int64)
    for r in range(shape.world_size):
        demand += torch.bincount(topk[r].flatten(), minlength=shape.num_experts)
    _, _, cap = plan_redundant_experts(demand, shape.num_experts, shape.world_size)
    assert cap == shape.num_tokens * shape.top_k

    hidden, rw, topk = _make_case_inputs(shape, mode, seed)
    for r, res in enumerate(
        moonep_dispatch_golden(hidden, rw, topk, shape.num_experts)
    ):
        cu = res["cu_seqlens"]
        assert cu.shape == (2 * shape.epn,), f"rank {r} cu_seqlens len"
        assert (cu[1:] >= cu[:-1]).all(), f"rank {r} cu_seqlens not monotone"
        assert int(cu[-1]) == shape.num_tokens * shape.top_k, (
            f"rank {r} cu_seqlens tail"
        )


@pytest.mark.parametrize("shape,mode,seed", _CASES)
def test_golden_roundtrip_identity(shape, mode, seed):
    """往返契约：专家为 identity 时 combine∘dispatch(x) = x * sum_k w。"""
    hidden, rw, topk = _make_case_inputs(shape, mode, seed)
    results = moonep_dispatch_golden(hidden, rw, topk, shape.num_experts)
    out = moonep_combine_golden(results, shape.num_tokens, _H)
    expected = hidden.to(torch.float32) * rw.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape,mode,seed", _CASES)
def test_golden_roundtrip_per_expert_scale(shape, mode, seed):
    """映射正确性：expert_fn 按专家缩放时，逐 (s,k) 路由关系须严格保持。"""
    hidden, rw, topk = _make_case_inputs(shape, mode, seed)
    results = moonep_dispatch_golden(hidden, rw, topk, shape.num_experts)
    out = moonep_combine_golden(
        results, shape.num_tokens, _H, expert_fn=lambda x, e: x * (e + 1)
    )
    scales = (topk.to(torch.float32) + 1.0).unsqueeze(-1)  # [R,S,K,1]
    expected = (hidden.to(torch.float32).unsqueeze(2) * rw.unsqueeze(-1) * scales).sum(
        2
    )
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
