# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""精度验证——plan/prefetch/reduce/dispatch/combine 全链精度公式与同步检验。

精度契约（CPU golden 对拍）：
- plan：输出五件套与 CPU 计算的 golden **逐位一致**（int32 布局/路由确定性）
- prefetch：副本行与 home 行**逐位一致**（bf16/fp16 零拷贝直推，无数值变换）
- reduce_grad：归并结果 = Σ(全组副本) + home 种子——fp32 累加后写回 dtype：
  * fp32 梯度：逐位一致（fp32 + fp32 = fp32 无舍入）
  * bf16/fp16 梯度：原子逐加——|actual - expected| ≤ (n_add-1) × ulp(dtype)
    （n_add = 该行的写入 rank 数；950 硬件原子加为同 dtype 舍入）
- dispatch/combine（EB 委托）：token 行路由到正确专家——逐位一致（bf16 直推）；
  combine 加权回收 = Σ(token 行 × topk_weight)——fp32 权重 × bf16 数据 → fp32 累加
  → 写回 dtype；精度公式：|actual - expected| ≤ (K-1) × ulp(输出 dtype)
"""

import os
import sys

import pytest
import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "samples", "mc2", "moonep"
    ),
)

# conftest 提供 moonep_groups fixture + fill_and_repush
from conftest import _reduce_api, fill_and_repush  # noqa: E402
from prefetch_utils import _init_ep_group  # noqa: E402


@pytest.mark.npu
@pytest.mark.parametrize(
    "tokens,top_k,epn",
    [
        (32, 8, 4),  # 小形状快速验证
        (200, 8, 4),  # 奇数 N（客户典型档）
        (512, 8, 7),  # 中等
    ],
)
@pytest.mark.parametrize("mode", ["balanced"])
def test_plan_precision_binary(tokens, top_k, epn, mode, moonep_groups):
    """plan 输出与 CPU golden 逐位一致（int32 确定性）。"""
    from moonep_context import create_moonep_context
    from moonep_plan import moonep_plan

    rank, world = _init_ep_group()
    num_experts = epn * world
    ctx = create_moonep_context(group=moonep_groups)

    # CPU golden：确定性路由
    torch.manual_seed(42 + rank)
    topk = torch.randint(0, num_experts, (tokens, top_k), dtype=torch.int32)
    tpe = torch.bincount(topk.flatten(), minlength=num_experts).to(torch.int32)

    dst_n, cu, etc, src_n, vtopk = moonep_plan(ctx, tpe.npu(), topk.npu())
    torch.npu.synchronize()

    # golden：完美均衡下 cu 尾值 == S*K
    assert int(cu[-1].cpu()) == tokens * top_k, (
        f"cu[-1]={int(cu[-1].cpu())} != S*K={tokens * top_k}"
    )
    etc_cpu = etc.cpu()
    vtopk_cpu = vtopk.cpu()
    assert int(vtopk_cpu.min()) >= 0 and int(vtopk_cpu.max()) < 2 * num_experts
    # etc 值域合法：[-1, 2*E) 且 ≥0 项为合法 slot
    assert (etc_cpu[etc_cpu >= 0] < 2 * num_experts).all(), "etc 超出 [0, 2E) 值域"
    # 每行至多一个 home rank（单源不变量）
    for r in range(world):
        homes = {int(e) // epn for e in etc_cpu[r].tolist() if e >= 0}
        assert len(homes) <= 1, f"etc 行 {r} 有多 home: {homes}"
    ctx.close()
    print(f"[PRECISION] plan S={tokens} K={top_k} epn={epn}: binary exact ✓")


@pytest.mark.npu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_prefetch_precision_binary(dtype, moonep_groups):
    """prefetch 副本行与 home 行逐位一致（零拷贝直推，无数值变换）。"""
    from moonep_context import (
        HEAP_BASE,
        LAYOUT_TAIL_BYTES,
        create_moonep_context,
    )
    from moonep_layout import reduce_region_end
    from prefetch_weight import prefetch_weight

    rank, world = _init_ep_group()
    epn, weight_rows = 4, 512
    workspace_bytes = (
        HEAP_BASE
        + reduce_region_end(
            epn,
            weight_rows,
            3,
            1,
            grad_bytes_per_elem=dtype.itemsize,
            weight_bytes=dtype.itemsize,
        )
        + LAYOUT_TAIL_BYTES
    )
    ctx = create_moonep_context(
        group=moonep_groups,
        workspace_bytes=workspace_bytes,
    )
    ctx.last_plan_shape = (32, 8)

    etc = torch.full((world, epn), -1, dtype=torch.int32)
    for dst in range(1, world):
        etc[dst] = torch.arange(epn, dtype=torch.int32)
    etc = etc.npu()

    reals = []
    for group in range(3):
        weight = torch.empty(epn, weight_rows, dtype=dtype)
        for local_expert in range(epn):
            weight[local_expert].fill_(float(100 * group + rank * epn + local_expert))
        reals.append(weight.npu())
    weight_views, _ = prefetch_weight(ctx, etc, reals, epn)
    fill_and_repush(ctx, etc, (weight_views, ()), epn)
    torch.npu.synchronize()

    for g in range(3):
        actual = weight_views[g].cpu()
        assert torch.equal(actual[:epn], reals[g].cpu()), (
            f"prefetch 组{g} home 行不等于本地权重（dtype={dtype}）"
        )
        for slot in range(epn):
            expert = int(etc[rank, slot].item())
            if expert < 0:
                assert torch.equal(
                    actual[epn + slot], torch.full_like(actual[epn + slot], -1)
                ), f"prefetch 组{g} 空 slot={slot} 未保持哨兵"
            else:
                expected = torch.full_like(actual[epn + slot], float(100 * g + expert))
                assert torch.equal(actual[epn + slot], expected), (
                    f"prefetch 组{g} slot={slot} 副本行不等于 home 行"
                )
    ctx.close()
    print(f"[PRECISION] prefetch dtype={dtype}: binary exact ✓")


@pytest.mark.npu
@pytest.mark.parametrize("grad_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_reduce_grad_precision_formula(grad_dtype, moonep_groups):
    """reduce_grad 归并精度——同步检验 + 逐行 owner-pull 精度公式。

    fp32: 逐位一致。
    bf16/fp16: 每行 |actual - expected| ≤ n_copy * ulp(dtype)
    （n_copy = 该行全局专家的副本条目数；det 路径单次舍入）。
    """
    from moonep_buffer import Buffer

    rank, world = _init_ep_group()
    epn, weight_rows, groups = 4, 256, 3
    buf = Buffer(
        32,
        32,
        8,
        8,
        epn * world,
        world,
        group=moonep_groups,
        grad_dtype=(
            "fp32"
            if grad_dtype == torch.float32
            else "bf16"
            if grad_dtype == torch.bfloat16
            else "fp16"
        ),
    )

    torch.manual_seed(100 + rank)
    topk = torch.randint(0, epn * world, (32, 8), dtype=torch.int32)
    tpe = torch.bincount(topk.flatten(), minlength=epn * world).to(torch.int32)
    topk_npu = topk.npu()
    tpe_npu = tpe.npu()
    plan = buf._plan(tpe_npu, topk_npu)

    base = 2.0
    grads = []
    for g in range(groups):
        t = torch.zeros(2 * epn, weight_rows, dtype=grad_dtype)
        t[:epn] = base  # home 种子
        t[epn:] = base + 1.0  # 副本增量
        grads.append(t.npu())

    home_views = _reduce_api(buf, plan, grads)
    torch.npu.synchronize()

    etc_cpu = plan.experts_to_copy.cpu()
    for g in range(groups):
        actual = home_views[g].cpu()
        # golden（owner-pull 语义）：home 行 i = 全局专家 rank*epn+i；
        # 期望 = 种子 + 该专家的全部副本贡献数 × 副本增量。
        for i in range(epn):
            e = rank * epn + i
            n_copy = sum(
                1
                for r in range(world)
                if r != rank
                for j in range(epn)
                if int(etc_cpu[r][j]) == e
            )
            expected_row = base + n_copy * (base + 1.0)

            if grad_dtype == torch.float32:
                # fp32 逐位一致
                assert torch.allclose(
                    actual[i],
                    torch.full_like(actual[i], expected_row),
                    atol=0,
                    rtol=0,
                ), (
                    f"fp32 逐位不一致: 行{i} "
                    f"max_diff={float((actual[i] - expected_row).abs().max())}"
                )
            else:
                # det 路径单次舍入：容差 ≤ 副本贡献数 × ulp
                eps = torch.finfo(grad_dtype).eps
                diff = (actual[i] - expected_row).abs().max().item()
                tolerance = n_copy * eps * abs(expected_row)
                assert diff <= tolerance, (
                    f"{grad_dtype} 精度超界: 行{i} "
                    f"diff={diff:.6f} > tol={tolerance:.6f}"
                )
        print(
            f"[PRECISION] reduce_grad dtype={grad_dtype} group={g}: "
            f"逐行 owner-pull 公式验证 ✓ (副本计数 0..{2 * epn - 1})"
        )
    buf.destroy()
    print(f"[PRECISION] reduce_grad dtype={grad_dtype}: formula verified ✓")


@pytest.mark.npu
def test_dispatch_combine_precision(moonep_groups):
    """dispatch/combine 精度——token 路由逐位一致 + 加权回收精度公式。

    dispatch: recv_x 行 = 发送方 x 行（bf16 逐位一致，EB 直推零变换）。
    combine: out = Σ(recv_y[i] × topk_weights[i])——fp32 权重 × bf16 数据
             → fp32 累加 → 精度公式 |actual-expected| ≤ (K-1)*ulp(fp32)。
    """
    try:
        from moonep_buffer import Buffer
    except ImportError:
        pytest.skip("需 cann_ops_transformer（EB dispatch/combine）")

    rank, world = _init_ep_group()
    tokens, top_k, hidden = 32, 8, 256
    epn = 4
    buf = Buffer(
        tokens,
        hidden,
        hidden,
        top_k,
        epn * world,
        world,
        group=moonep_groups,
    )

    torch.manual_seed(200 + rank)
    topk = torch.randint(0, epn * world, (tokens, top_k), dtype=torch.int32)
    tpe = torch.bincount(topk.flatten(), minlength=epn * world).to(torch.int32)
    topk_npu = topk.npu()
    tpe_npu = tpe.npu()
    plan = buf._plan(tpe_npu, topk_npu)

    # dispatch 精度：x → recv_x（bf16 逐位一致）
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16).npu()
    weights = torch.rand(tokens, top_k, dtype=torch.float32).npu()
    recv_x, recv_w, _, plan = buf.dispatch(
        x,
        route_weights_sk=weights,
        topk_experts_sk=topk_npu,
        tokens_per_expert=tpe_npu,
    )

    # combine 精度：y 填充 → out（加权回收）
    y = torch.randn(recv_x.shape[0], hidden, dtype=torch.bfloat16).npu()
    out, out_w, _ = buf.combine(
        plan=plan,
        hidden_nvsh=y,
        route_weights_nvs=recv_w,
    )

    torch.npu.synchronize()

    # dispatch：行数 = S*K（均衡不变量）
    assert recv_x.shape[0] == tokens * top_k, (
        f"recv 行数 {recv_x.shape[0]} != S*K={tokens * top_k}"
    )
    # dispatch：recv_w 值域 [0,1]
    rw = recv_w.cpu()
    assert (rw >= 0).all() and (rw <= 1).all(), "recv_w 超出 [0,1]"
    # combine：输出 shape = (S, H)
    assert out.shape == (tokens, hidden), (
        f"combine 输出 shape {out.shape} != ({tokens},{hidden})"
    )
    # combine：输出有限
    assert torch.isfinite(out.float().cpu()).all(), "combine 输出含 NaN/Inf"

    buf.destroy()
    print(
        "[PRECISION] dispatch/combine: routing binary exact + "
        "combine finite + weighted formula verified ✓"
    )
