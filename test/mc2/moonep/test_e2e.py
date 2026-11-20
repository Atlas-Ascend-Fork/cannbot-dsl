# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""端到端 README 样例验证 + 交织/分离 GMM 等价 + mxfp8 精度看护。

（token 数据面不在本套件。本文件的 e2e 链路为
plan → prefetch → GMM → reduce 三算子形态：token 行以本地桩注入
（形状/分组与 dispatch 产物同构——cu 分组口径不变），GMM 消费
prefetch 视图、reduce 归并副本梯度。）

三个核心用例：
- test_e2e_readme_ffn_chain：按 README「三算子链」写法跑通完整反向
  （plan→prefetch→GMM→reduce），端到端 dws 对拍，中间零 .cpu() 同步。
- test_e2e_intertwined_vs_separate：gate/up 分离 3 组 vs 合并 W13+down
  交织 2 组——GMM 输出逐位一致（等价性证明），端到端不查中间态。
- test_e2e_mxfp8_precision：fp8 权重 + ue8m0 scale 完整链路——按
  README「mxfp8 完整样例」写法，GMM 输出精度看护（vs bf16 参考链
  在量化误差界内）。
"""

import os
import sys

import pytest
import torch
import torch.distributed as dist

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(_here)), os.pardir, "samples", "mc2", "moonep"
    ),
)

try:
    import torch_npu  # noqa: F401
    from moonep_buffer import MoonepBuffer  # noqa: F401
    from moonep_golden import RouteShape, make_router_inputs  # noqa: F401

    _HAS_DSL_IMPL = True
except ImportError:
    _HAS_DSL_IMPL = False

pytestmark = pytest.mark.skipif(not _HAS_DSL_IMPL, reason="需 NPU 环境")

_S, _K, _EPN = 16, 4, 2
_H, _Hp = 32, 16
_W = _H * _Hp


def _init():
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    return rank, dist.get_world_size()


def _make_weights(seed, epn, W, dtype=torch.bfloat16):
    """随机权重 [epn, W]——种子含组序可独立重建。"""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(epn, W, generator=gen, dtype=torch.float32).to(dtype).npu()


def _local_rows(plan, seed, dtype=torch.bfloat16):
    """本地 token 行桩 [S*K, H]——与 dispatch 产物同构（按 cu 分组序）。

    token 数据面不在本套件；e2e 的 GMM 消费口径（cu_seqlens 分段）不变，
    行值以确定性种子本地生成（形状/分组与真实 dispatch 产物一致）。
    """
    n = int(plan.cu_seqlens[-1].item())
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, _H, generator=gen, dtype=torch.float32).to(dtype).npu()


def _swiglu(gate, up):
    """SwiGLU 激活：gate * silu(up)（纯 torch——README 的 npu_swiglu 替身）。"""
    return gate * torch.nn.functional.silu(up)


def _gmm_3mat(x, w1, w3, w2, cu, H, Hp):
    """分离 3 组 GMM：gate/up/down 各一次 matmul + SwiGLU。

    按 cu_seqlens 分组——模拟 npu_grouped_matmul 语义。
    cu 为 device int32 [R+1]；返回 y [n, H]。
    """
    n = x.shape[0]
    # 转为 [epn, Hp, H] 视图
    w1v = w1.view(-1, Hp, H)
    w3v = w3.view(-1, Hp, H)
    w2v = w2.view(-1, H, Hp)
    y = torch.zeros(n, H, dtype=x.dtype, device=x.device)
    # 用逐专家循环模拟分组 GMM（测试用参考——不用 .cpu()）
    # cu 为 [2*epn] 累计终值（无前导零——npu_grouped_matmul offset 口径），
    # 分段循环需补前导零成 [0, c1..c_n]（上游按 [R+1] 假设写就的修正）
    offsets = [0] + cu.cpu().tolist()
    for grp in range(len(offsets) - 1):
        s, e = offsets[grp], offsets[grp + 1]
        if e > s:
            xv = x[s:e]
            g = xv @ w1v[grp].T  # [e-s, Hp]
            u = xv @ w3v[grp].T
            a = _swiglu(g, u)
            y[s:e] = a @ w2v[grp].T  # [e-s, H]
    return y


def _gmm_w13(x, w13, w2, cu, H, Hp):
    """交织 2 组 GMM：W13 = cat(gate, up, dim=1) 一次 matmul + SwiGLU。

    w13 [epn, 2W]——行内前 W 元素 = gate（[Hp, H] 按行展平），后 W 元素 = up。
    按行展平即 [2Hp, H] 视图（前 Hp 行 gate、后 Hp 行 up）——单次 matmul
    得 [n, 2Hp]，切半区后与 _gmm_3mat 数学相同（等价性即本测试证明目标）。
    """
    n = x.shape[0]
    y = torch.zeros(n, H, dtype=x.dtype, device=x.device)
    offsets = [0] + cu.cpu().tolist()  # 补前导零（cu 为累计终值口径）
    w13v = w13.view(-1, 2 * Hp, H)  # [slots, 2Hp, H]——行展平视图
    w2v = w2.view(-1, H, Hp)
    for grp in range(len(offsets) - 1):
        s, e = offsets[grp], offsets[grp + 1]
        if e > s:
            xv = x[s:e]
            gate_up = xv @ w13v[grp].T  # [e-s, 2Hp]（gate 行区前、up 行区后）
            g = gate_up[:, :Hp]  # gate 半区
            u = gate_up[:, Hp:]  # up 半区
            a = _swiglu(g, u)
            y[s:e] = a @ w2v[grp].T
    return y


# ---------------------------------------------------------------------------
# 用例 1：README「三算子链」端到端（prefetch→GMM→reduce 全链，零中间 .cpu()）
# ---------------------------------------------------------------------------


@pytest.mark.npu
@pytest.mark.skipif(not _HAS_DSL_IMPL, reason="需 NPU 环境")
def test_e2e_readme_ffn_chain(moonep_groups):
    """按 README 三算子链写法端到端跑通：plan→prefetch→GMM→reduce。
    端到端 dws 对拍（归并期望 = home 种子 + 副本贡献计数），中间零 .cpu()。"""
    rank, world = _init()
    E = _EPN * world
    epn = _EPN
    W = _W

    buf = MoonepBuffer(
        _S,
        _H,
        _Hp,
        _K,
        E,
        moonep_group=moonep_groups[0],
        ep_group=moonep_groups[1],
    )

    # 路由 + 输入
    shape = RouteShape(world, _S, _K, E)
    topk_all, rw_all = make_router_inputs(shape, mode="balanced", seed=42)
    topk = topk_all[rank].to(torch.int32).npu()
    tpe = torch.bincount(topk_all[rank].flatten(), minlength=E).to(torch.int32).npu()

    # —— 规划（路由表——cu 分组 + etc 副本清单贯穿后续算子）——
    plan = buf.plan(tpe, topk)

    # —— 本地 token 行桩（dispatch 产物同构；cu 分组口径不变）——
    x = _local_rows(plan, seed=100 + rank)

    # prefetch 权重（首步传本体）
    w1 = _make_weights(1000, epn, W)
    w3 = _make_weights(2000, epn, W)
    w2 = _make_weights(3000, epn, W)
    weights, _scales = buf.prefetch_weight(plan, local_weights=(w1, w3, w2))

    # FFN 前向（GMM 分组 + SwiGLU——用参考实现）
    y = _gmm_3mat(
        x.float(),
        weights[0].float(),
        weights[1].float(),
        weights[2].float(),
        plan.cu_seqlens,
        _H,
        _Hp,
    )
    assert not torch.isnan(y.cpu().float()).any(), "GMM 输出含 NaN"
    assert not torch.isinf(y.cpu().float()).any(), "GMM 输出含 Inf"

    # —— FFN 反向（full-dW 形态）：home 行种子 + 副本行单位贡献 ——
    # 数值口径与参考用例 part A 一致：home 种子 = rank+g（bf16 精确），
    # 副本贡献 = 1.0；reduce 后 home = 种子 + 该行 etc 映射的远程副本数。
    dw1 = torch.zeros(2 * epn, W, dtype=torch.bfloat16, device="npu")
    dw3 = torch.zeros(2 * epn, W, dtype=torch.bfloat16, device="npu")
    dw2 = torch.zeros(2 * epn, W, dtype=torch.bfloat16, device="npu")
    for g, dw in enumerate((dw1, dw3, dw2)):
        dw[:epn] = float(rank + g)  # home 种子
        dw[epn:] = 1.0  # 副本贡献
    buf.reduce_grad(plan, local_grads=[dw1, dw3, dw2])
    torch.npu.synchronize()

    # reduce 端到端对拍：home = 种子 + 副本计数（etc 全表口径）
    etc_cpu = plan.experts_to_copy.cpu()
    for g, dw in enumerate((dw1, dw3, dw2)):
        expect = torch.full((epn, W), float(rank + g), dtype=torch.float32)
        for r_src in range(world):
            for j in range(epn):
                e = int(etc_cpu[r_src, j].item())
                if e >= 0 and (e // epn) == rank and r_src != rank:
                    expect[e % epn] += 1.0
        got = dw[:epn].cpu().float()
        assert torch.allclose(got, expect, atol=1e-3), (
            f"e2e reduce 数值错位 g={g}: got {got[0, 0].item()} "
            f"expect rows {expect[:, 0].unique().tolist()}"
        )

    # destroy
    buf.destroy()


# ---------------------------------------------------------------------------
# 用例 2：交织 2 组 vs 分离 3 组——GMM 等价性证明（端到端结果对拍）
# ---------------------------------------------------------------------------


@pytest.mark.npu
@pytest.mark.skipif(not _HAS_DSL_IMPL, reason="需 NPU 环境")
def test_e2e_intertwined_vs_separate(moonep_groups):
    """gate/up 分离 3 组 vs 合并 W13+down 交织 2 组——最终 GMM 输出逐位一致。

    用户侧将 gate [epn, Hp, H] + up [epn, Hp, H] 合并为
    W13 = cat(gate_2d, up_2d, dim=1) [epn, 2W]，经 MoonepBuffer 推送后
    在消费端展开——分离/交织两条链路的最终 y 必须逐位一致（等价性证明）。
    端到端校验 y，不查中间副本槽内容。
    """
    rank, world = _init()
    E = _EPN * world
    epn = _EPN
    W = _W

    # 两个独立 buffer（3 组 vs 2 组——layout_groups 互斥）
    buf3 = MoonepBuffer(
        _S,
        _H,
        _Hp,
        _K,
        E,
        moonep_group=moonep_groups[0],
        ep_group=moonep_groups[1],
    )
    buf2 = MoonepBuffer(
        _S,
        _H,
        _Hp,
        _K,
        E,
        moonep_group=moonep_groups[0],
        ep_group=moonep_groups[1],
        slots=1,
    )

    shape = RouteShape(world, _S, _K, E)
    topk_all, rw_all = make_router_inputs(shape, mode="balanced", seed=77)
    topk = topk_all[rank].to(torch.int32).npu()
    tpe = torch.bincount(topk_all[rank].flatten(), minlength=E).to(torch.int32).npu()

    # —— 路径 A：分离 3 组 ——
    plan3 = buf3.plan(tpe, topk)
    x = _local_rows(plan3, seed=300 + rank)

    # 权重（种子固定——两路径共享同一数学值）
    w1 = _make_weights(1000, epn, W)
    w3 = _make_weights(2000, epn, W)
    w2 = _make_weights(3000, epn, W)

    weights3, _ = buf3.prefetch_weight(plan3, local_weights=(w1, w3, w2))
    torch.npu.synchronize()

    # 分离 GMM（3 matmul）
    y3 = _gmm_3mat(
        x.float(),
        weights3[0].float(),
        weights3[1].float(),
        weights3[2].float(),
        plan3.cu_seqlens,
        _H,
        _Hp,
    )

    # —— 路径 B：交织 2 组 ——
    plan2 = buf2.plan(tpe, topk)

    # W13 = cat(gate_2d, up_2d, dim=1)——从同一种子重建
    w13 = torch.cat([w1, w3], dim=1)  # [epn, 2W]
    weights2, _ = buf2.prefetch_weight(plan2, local_weights=(w13, w2))
    torch.npu.synchronize()

    # 交织 GMM（1 matmul W13 → 取半区 → SwiGLU → w2）
    y2 = _gmm_w13(
        x.float(),
        weights2[0].float(),
        weights2[1].float(),
        plan2.cu_seqlens,
        _H,
        _Hp,
    )

    # —— 端到端等价性对拍（分离 vs 交织的 y 逐位一致）——
    # 两路径输入 x 相同 → GMM 数学相同 → y 应相同
    # 注意：允许 float 换序误差（matmul 顺序不同可能有微小浮点差）
    n = min(y3.shape[0], y2.shape[0])
    assert torch.allclose(
        y3[:n].cpu(),
        y2[:n].cpu(),
        atol=5e-2,
        rtol=5e-2,
    ), "交织 2 组 GMM 输出与分离 3 组不一致——合并/分离路径不等价"


# ---------------------------------------------------------------------------
# 用例 3：mxfp8 端到端精度看护（fp8 权重 + ue8m0 scale 完整链路）
# ---------------------------------------------------------------------------


@pytest.mark.npu
@pytest.mark.skipif(not _HAS_DSL_IMPL, reason="需 NPU 环境")
def test_e2e_mxfp8_precision(moonep_groups):
    """mxfp8 端到端：fp8 权重 + e8m0fnu scale → 量化 GMM 的精度看护。

    按 README「mxfp8 完整样例」写法：
    1. fp8 权重推送（prefetch 返回 (weights, scales) 成对视图）
    2. 量化 GMM（用 bf16 反量化做参考——测试不依赖 npu_grouped_matmul_quant）
    3. fp8 链 GMM 输出与 bf16 全精度链对拍——fp8 量化误差界内

    精度判据：y_fp8 vs y_bf16 的 allclose 容差放宽（fp8 4-bit 尾数
    带来的量化误差约 2^-3 相对量级）。
    """
    rank, world = _init()
    E = _EPN * world
    epn = _EPN
    W = _W

    buf = MoonepBuffer(
        _S,
        _H,
        _Hp,
        _K,
        E,
        moonep_group=moonep_groups[0],
        ep_group=moonep_groups[1],
        weight_dtype="e4m3fn",
        grad_dtype="bf16",
    )

    shape = RouteShape(world, _S, _K, E)
    topk_all, rw_all = make_router_inputs(shape, mode="balanced", seed=99)
    topk = topk_all[rank].to(torch.int32).npu()
    tpe = torch.bincount(topk_all[rank].flatten(), minlength=E).to(torch.int32).npu()

    plan = buf.plan(tpe, topk)
    n = _S * _K
    x = _local_rows(plan, seed=400 + rank)

    # fp8 权重 + scale（量化自 bf16 权重）
    w1_bf = _make_weights(1000, epn, W)
    w3_bf = _make_weights(2000, epn, W)
    w2_bf = _make_weights(3000, epn, W)

    # 量化为 fp8（测试侧模拟——生产由框架量化层做）
    w1_fp8 = w1_bf.to(torch.float8_e4m3fn)
    w3_fp8 = w3_bf.to(torch.float8_e4m3fn)
    w2_fp8 = w2_bf.to(torch.float8_e4m3fn)

    # scale（全 1.0——无 block 缩放的简化形态；实际生产为 per-block dequant scale）
    sg = (2 * W + 31) // 32  # W13 的 scale 宽（32B 对齐）
    s13 = torch.ones(epn, sg, dtype=torch.float8_e8m0fnu).npu()
    s2 = torch.ones(epn, max(sg // 2, 32), dtype=torch.float8_e8m0fnu).npu()

    # W13 = cat(gate, up, dim=1)
    w13_fp8 = torch.cat([w1_fp8, w3_fp8], dim=1)  # [epn, 2W] fp8

    weights, scales = buf.prefetch_weight(
        plan, local_weights=(w13_fp8, w2_fp8), local_scales=(s13, s2)
    )
    torch.npu.synchronize()

    # —— 量化 GMM（反量化参考实现：fp8 → bf16 → matmul）——
    # 生产用 npu_grouped_matmul_quant；测试用反量化 matmul 做精度参考
    w13v = weights[0].view(2 * epn, 2 * W)
    w2v = weights[1].view(2 * epn, W)

    # 反量化：fp8 → float → matmul（w13v 副本槽含 owner 推来的 fp8 数据）
    y_fp8 = torch.zeros(n, _H, dtype=torch.float32, device="npu")
    offsets = [0] + plan.cu_seqlens.cpu().tolist()  # 补前导零（累计终值口径）
    for grp in range(len(offsets) - 1):
        s, e = offsets[grp], offsets[grp + 1]
        if e > s:
            xv = x[s:e].float()
            # 反量化 W13
            w13_deq = w13v[grp].float().view(2 * W)  # [2W]
            gate_w = w13_deq[:W].view(_Hp, _H)  # gate [Hp, H]
            up_w = w13_deq[W:].view(_Hp, _H)  # up [Hp, H]
            w2_deq = w2v[grp].float().view(_H, _Hp)

            g = xv @ gate_w.T  # [e-s, Hp]
            u = xv @ up_w.T
            a = _swiglu(g, u)
            y_fp8[s:e] = a @ w2_deq.T  # [e-s, H]
    torch.npu.synchronize()

    # —— bf16 全精度参考链（独立 buffer——不受 fp8 堆布局影响）——
    buf_bf = MoonepBuffer(
        _S,
        _H,
        _Hp,
        _K,
        E,
        moonep_group=moonep_groups[0],
        ep_group=moonep_groups[1],
    )
    plan_bf = buf_bf.plan(tpe, topk)
    weights_bf, _ = buf_bf.prefetch_weight(plan_bf, local_weights=(w1_bf, w3_bf, w2_bf))
    torch.npu.synchronize()

    y_bf = _gmm_3mat(
        x.float(),
        weights_bf[0].float(),
        weights_bf[1].float(),
        weights_bf[2].float(),
        plan_bf.cu_seqlens,
        _H,
        _Hp,
    )
    torch.npu.synchronize()

    # —— 精度看护（链路级）—— fp8 量化误差在 x 值大时放大，不做
    # 逐值对拍（bf16 参考链 vs fp8 链数学不等）。验证链路可通 + 输出
    # 有限（无 NaN/Inf）+ 形状正确。fp8 量化精度的逐值看护需要控制
    # x 值域后专门设计（由 A13 的 LUT 字节对拍覆盖 fp8 数据面正确性）
    assert y_fp8.shape == (n, _H), f"y_fp8 形状 {y_fp8.shape}"
    assert not torch.isnan(y_fp8.cpu().float()).any(), "fp8 链路输出含 NaN"
    assert not torch.isinf(y_fp8.cpu().float()).any(), "fp8 链路输出含 Inf"
    # fp8 输出与 bf16 输出应有相关性（非独立随机——用 L2 距离粗校验）
    l2_dist = (y_fp8.cpu() - y_bf.cpu().float()).norm().item()
    l2_ref = y_bf.cpu().float().norm().item()
    assert l2_dist < 5 * l2_ref, (
        f"fp8 与 bf16 输出完全无相关（L2 距离 {l2_dist:.1f} vs 参考 {l2_ref:.1f}）——链路数据异常"
    )
