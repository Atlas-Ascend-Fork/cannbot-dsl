# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""用户典型 shape 覆盖——从设计文档提取的生产/客户形状档。

典型档（S, H, Hp, K, E, R）：
- 小验证：S=32, H=3584, Hp=3072, K=8, E=224, R=4
- 客户典型：S=200, H=3584, Hp=3072, K=8, E=28, R=4（奇数 N 档）
- 生产中：S=512, H=3584, Hp=3072, K=8, E=28, R=4
- megamoe 样例：S=8192, H=3584, Hp=3072, K=16, E=224, R=4（DSL 容量界内）
"""

import os
import sys

import pytest
import torch
import torch_npu  # noqa: F401

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "samples", "mc2", "moonep"
    ),
)
from prefetch_utils import _init_ep_group  # noqa: E402


# MoonepDesignV3 §3.1 客户典型 shape（DSL 容量内可测变体——S×K ≤ 8192）
# 原始客户档 S=4096×K=10/16 超 DSL_PLAN_MAX_SK，按容量缩放；H/Hp 保留原值
_TYPICAL_SHAPES = [
    pytest.param(32, 3584, 3072, 8, 224, id="s32k8_small"),
    pytest.param(200, 3584, 3072, 8, 28, id="s200k8_customer"),
    pytest.param(512, 3584, 3072, 8, 28, id="s512k8_mid"),
    # MoonepDesignV3 §3.1 客户档（容量内变体）
    pytest.param(512, 7168, 3584, 8, 56, id="kimi_k3_r4"),
    pytest.param(512, 2560, 512, 8, 48, id="pangu_34b_r4"),
    pytest.param(1024, 2560, 1024, 4, 64, id="pangu_92b_r4"),
    pytest.param(1024, 5120, 1024, 4, 96, id="pangu_505b_r4"),
    pytest.param(1024, 3584, 3072, 8, 56, id="gen_std_r4"),
    pytest.param(512, 7168, 3072, 8, 56, id="gen_h7168_r4"),
    pytest.param(1024, 3584, 3072, 8, 224, id="s1024k8_boundary"),
    pytest.param(
        8192, 3584, 3072, 16, 224, id="s8192k16_megamoe", marks=pytest.mark.full
    ),
]


@pytest.mark.npu
@pytest.mark.parametrize("tokens,hidden,hp,top_k,experts", _TYPICAL_SHAPES)
@pytest.mark.parametrize("grad_dtype", ["bf16"])
def test_typical_shape_plan_prefetch_reduce(
    tokens, hidden, hp, top_k, experts, grad_dtype, moonep_groups
):
    """典型 shape 下 plan → prefetch → reduce_grad 全链功能验证。"""
    from moonep_buffer import Buffer

    rank, world = _init_ep_group()
    epn = experts // world
    weight_rows = hidden * hp  # 展平行宽

    buf = Buffer(
        tokens,
        hidden,
        hp,
        top_k,
        experts,
        world,
        group=moonep_groups,
        weight_dtype="bf16",
        grad_dtype=grad_dtype,
    )

    # plan
    torch.manual_seed(42 + rank)
    topk = torch.randint(0, experts, (tokens, top_k), dtype=torch.int32)
    tpe = torch.bincount(topk.flatten(), minlength=experts).to(torch.int32)
    plan = buf._plan(tpe.npu(), topk.npu())
    torch.npu.synchronize()

    # 均衡不变量
    assert int(plan.cu_seqlens[-1].cpu()) == tokens * top_k

    # prefetch
    def _randn_group_weights():
        # prefetch 契约：local_weights 每组 [epn, ...]
        # 本地权重——[2*epn, Wg] 全量视图（home 行+副本行）由算子内部创建并返回
        return torch.randn(epn, weight_rows, dtype=torch.bfloat16).npu()

    reals = [_randn_group_weights() for _ in range(3)]
    views, _scales = buf.prefetch_weight(
        plan=plan,
        local_gate_weight=reals[0],
        local_up_weight=reals[1],
        local_down_weight=reals[2],
    )
    torch.npu.synchronize()

    # 视图 shape 校验
    for g, v in enumerate(views):
        assert v.shape[0] == 2 * epn, f"组{g} 视图行数 {v.shape[0]} != {2 * epn}"
        assert v.shape[1] == weight_rows, (
            f"组{g} 视图列数 {v.shape[1]} != {weight_rows}"
        )

    # reduce_grad
    grads = [torch.zeros(2 * epn, weight_rows, dtype=torch.bfloat16) for _ in range(3)]
    for g in range(3):
        grads[g][:epn] = 1.0
        grads[g][epn:] = 1.5
    home = buf.reduce_grad(
        plan=plan,
        local_gate_grad=grads[0].npu(),
        local_up_grad=grads[1].npu(),
        local_down_grad=grads[2].npu(),
    )
    torch.npu.synchronize()

    # 归并结果有限
    for g, hv in enumerate(home):
        assert torch.isfinite(hv.float().cpu()).all(), f"组{g} 归并结果含 NaN/Inf"

    buf.destroy()
    print(
        f"[TYPICAL] S={tokens} H={hidden} K={top_k} E={experts} R={world}: "
        "plan+prefetch+reduce ✓"
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "tokens,top_k,experts",
    [
        (4096, 8, 224),
        (4096, 16, 224),
        (200, 8, 28),
    ],
)
def test_typical_shape_dispatch_combine(tokens, top_k, experts, moonep_groups):
    """典型 shape 下 dispatch/combine（EB 委托）功能验证。"""
    try:
        from moonep_buffer import Buffer
    except ImportError:
        pytest.skip("需 cann_ops_transformer（EB dispatch/combine）")

    rank, world = _init_ep_group()
    hidden = 3584
    buf = Buffer(
        tokens,
        hidden,
        hidden,
        top_k,
        experts,
        world,
        group=moonep_groups,
    )

    torch.manual_seed(42 + rank)
    topk = torch.randint(0, experts, (tokens, top_k), dtype=torch.int32)
    tpe = torch.bincount(topk.flatten(), minlength=experts).to(torch.int32)
    plan = buf._plan(tpe.npu(), topk.npu())

    x = torch.randn(tokens, hidden, dtype=torch.bfloat16).npu()
    weights = torch.rand(tokens, top_k, dtype=torch.float32).npu()
    recv_x, recv_w, _, plan = buf.dispatch(
        x,
        route_weights_sk=weights,
        plan=plan,
    )
    y = torch.randn(recv_x.shape[0], hidden, dtype=torch.bfloat16).npu()
    out, out_w, _ = buf.combine(
        plan=plan,
        hidden_nvsh=y,
        route_weights_nvs=recv_w,
    )
    torch.npu.synchronize()

    assert recv_x.shape[0] > 0, "dispatch 接收行数为零"
    assert out.shape == (tokens, hidden), f"combine shape {out.shape}"
    buf.destroy()
    print(f"[TYPICAL] dispatch/combine S={tokens} K={top_k} E={experts}: ✓")
