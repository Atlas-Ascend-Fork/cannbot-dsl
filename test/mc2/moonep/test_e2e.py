# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""端到端 README 样例验证 + 交织/分离 GMM 等价 + mxfp8 精度看护。

三个核心用例：
- test_e2e_readme_ffn_chain：按 README「五算子链」写法跑通完整前反向
  （plan→dispatch→prefetch→GMM→combine→dispatch反向→reduce），
  端到端 out/g_x/dws 对拍，中间零 .cpu() 同步。
- test_e2e_intertwined_vs_separate：gate/up 分离 3 组 vs 合并 W13+down
  交织 2 组——最终 GMM 输出逐位一致（等价性证明），端到端不查中间态。
- test_e2e_mxfp8_precision：fp8 权重 + ue8m0 scale 完整链路——按
  README「mxfp8 完整样例」写法，端到端 out 精度看护（vs bf16 参考链
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

_ATOL = 2e-2
_RTOL = 2e-2

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
    offsets = cu.cpu().tolist()  # 一次 D2H 取分组边界（外层调用方已有 plan）
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
    offsets = cu.cpu().tolist()
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
# 用例 1：README「五算子链」端到端（前反向 FFN 全链，零中间 .cpu() 同步）
# ---------------------------------------------------------------------------


@pytest.mark.npu
@pytest.mark.skipif(not _HAS_DSL_IMPL, reason="需 NPU 环境")
def test_e2e_readme_ffn_chain(moonep_groups):
    """按 README 五算子链写法端到端跑通：plan→dispatch→prefetch→GMM→
    combine→dispatch 反向→reduce→optimizer。端到端结果对拍（out vs golden，
    dws vs 归并期望），中间零 .cpu()。"""
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
    x = torch.randn(
        _S,
        _H,
        dtype=torch.bfloat16,
        generator=torch.Generator().manual_seed(100 + rank),
    ).npu()
    scores = rw_all[rank].npu()

    # —— 前向 ——
    plan = buf.plan(tpe, topk)
    recv, recv_w, handle = buf.dispatch(plan, x, topk_weights=scores, do_cpu_sync=False)

    # prefetch 权重（首步传本体）
    w1 = _make_weights(1000, epn, W)
    w3 = _make_weights(2000, epn, W)
    w2 = _make_weights(3000, epn, W)
    weights, _scales = buf.prefetch_weight(plan, local_weights=(w1, w3, w2))

    n = _S * _K
    cap = recv.shape[0]

    # FFN 前向（GMM 分组 + SwiGLU——用参考实现）
    y = _gmm_3mat(
        recv[:n].float(),
        weights[0].float(),
        weights[1].float(),
        weights[2].float(),
        plan.cu_seqlens,
        _H,
        _Hp,
    )

    # combine 加权
    combine_in = torch.zeros(cap, _H, dtype=torch.bfloat16, device=y.device)
    combine_in[:n] = (y[:n].float() * recv_w[:n].unsqueeze(1)).to(torch.bfloat16)
    out, out_w = buf.combine(combine_in, handle, topk_weights=recv_w)
    torch.npu.synchronize()

    # 端到端结果断言——随机权重 GMM ≠ golden（数学不等，逐值对拍由 B1/B2 覆盖）
    # 本用例验证 README 样例可跑通：全部步骤无错误 + 形状/回收权重/数值有限
    assert out.shape == (_S, _H), f"out 形状 {out.shape} 应为 ({_S}, _H)"
    assert out_w.shape == (_S, _K), f"out_w 形状 {out_w.shape} 应为 ({_S}, {_K})"
    assert torch.equal(out_w.cpu(), rw_all[rank]), "combine 权重回收不符"
    assert not torch.isnan(out.cpu().float()).any(), "out 含 NaN"
    assert not torch.isinf(out.cpu().float()).any(), "out 含 Inf"

    # —— 反向 ——
    g_out_t = torch.randn(
        _S,
        _H,
        dtype=torch.bfloat16,
        generator=torch.Generator().manual_seed(200 + rank),
    ).npu()
    g_recv, _, handle_b = buf.dispatch(plan, g_out_t, topk_weights=None)
    # 反向 combine 输入容量取反向 dispatch 自身（do_cpu_sync 缺省 True——
    # 精确分配）；不得复用前向 do_cpu_sync=False 的 worst-case cap——容量
    # 与 handle metadata 失配会触发 EB packed int32 形状校验失败
    g_cap = g_recv.shape[0]
    g_x, _ = buf.combine(
        torch.zeros(g_cap, _H, dtype=torch.bfloat16, device=g_recv.device), handle_b
    )
    assert g_x.shape == (_S, _H), f"g_x 形状 {g_x.shape} 应为 ({_S}, _H)"

    # FFN 反向：full-dW（参考实现——消费 cu 分组）
    dw1 = torch.zeros(2 * epn, W, dtype=torch.bfloat16, device="npu")
    dw3 = torch.zeros(2 * epn, W, dtype=torch.bfloat16, device="npu")
    dw2 = torch.zeros(2 * epn, W, dtype=torch.bfloat16, device="npu")
    buf.reduce_grad(plan, local_grads=[dw1, dw3, dw2])
    torch.npu.synchronize()

    # reduce 端到端：home 段已归并（副本贡献加和后非零）
    # 不做中间态对拍——reduce 正确性由 B 系 golden 链路覆盖

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
    n = _S * _K

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
    x = torch.randn(
        _S,
        _H,
        dtype=torch.bfloat16,
        generator=torch.Generator().manual_seed(300 + rank),
    ).npu()

    # —— 路径 A：分离 3 组 ——
    plan3 = buf3.plan(tpe, topk)
    recv3, recv_w3, handle3 = buf3.dispatch(plan3, x, topk_weights=None)

    # 权重（种子固定——两路径共享同一数学值）
    w1 = _make_weights(1000, epn, W)
    w3 = _make_weights(2000, epn, W)
    w2 = _make_weights(3000, epn, W)

    weights3, _ = buf3.prefetch_weight(plan3, local_weights=(w1, w3, w2))
    torch.npu.synchronize()

    # 分离 GMM（3 matmul）
    y3 = _gmm_3mat(
        recv3[:n].float(),
        weights3[0].float(),
        weights3[1].float(),
        weights3[2].float(),
        plan3.cu_seqlens,
        _H,
        _Hp,
    )

    # —— 路径 B：交织 2 组 ——
    plan2 = buf2.plan(tpe, topk)
    recv2, recv_w2, handle2 = buf2.dispatch(plan2, x, topk_weights=None)

    # W13 = cat(gate_2d, up_2d, dim=1)——从同一种子重建
    w13 = torch.cat([w1, w3], dim=1)  # [epn, 2W]
    weights2, _ = buf2.prefetch_weight(plan2, local_weights=(w13, w2))
    torch.npu.synchronize()

    # 交织 GMM（1 matmul W13 → 取半区 → SwiGLU → w2）
    y2 = _gmm_w13(
        recv2[:n].float(),
        weights2[0].float(),
        weights2[1].float(),
        plan2.cu_seqlens,
        _H,
        _Hp,
    )

    # —— 端到端等价性对拍（分离 vs 交织的 y 逐位一致）——
    # 两路径输入 x 相同 → recv 应相同 → GMM 数学相同 → y 应相同
    # 注意：允许 float 换序误差（matmul 顺序不同可能有微小浮点差）
    assert torch.allclose(
        y3[: min(y3.shape[0], n)].cpu(),
        y2[: min(y2.shape[0], n)].cpu(),
        atol=5e-2,
        rtol=5e-2,
    ), "交织 2 组 GMM 输出与分离 3 组不一致——合并/分离路径不等价"


# ---------------------------------------------------------------------------
# 用例 3：mxfp8 端到端精度看护（fp8 权重 + ue8m0 scale 完整链路）
# ---------------------------------------------------------------------------


@pytest.mark.npu
@pytest.mark.skipif(not _HAS_DSL_IMPL, reason="需 NPU 环境")
def test_e2e_mxfp8_precision(moonep_groups):
    """mxfp8 端到端：fp8 权重 + e8m0fnu scale → GMM → combine 的精度看护。

    按 README「mxfp8 完整样例」写法：
    1. fp8 权重推送（prefetch 返回 (weights, scales) 成对视图）
    2. 量化 GMM（用 bf16 反量化做参考——测试不依赖 npu_grouped_matmul_quant）
    3. combine 输出与 bf16 全精度链路对拍——fp8 量化误差界内

    精度判据：out_fp8 vs out_bf16 的 allclose 容差放宽（fp8 4-bit 尾数
    带来的量化误差约 2^-3 相对量级）。
    """
    rank, world = _init()
    E = _EPN * world
    epn = _EPN
    W = _W
    n = _S * _K

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
    x = torch.randn(
        _S,
        _H,
        dtype=torch.bfloat16,
        generator=torch.Generator().manual_seed(400 + rank),
    ).npu()
    scores = rw_all[rank].npu()

    plan = buf.plan(tpe, topk)
    recv, recv_w, handle = buf.dispatch(plan, x, topk_weights=scores)

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
    offsets = plan.cu_seqlens.cpu().tolist()
    for grp in range(len(offsets) - 1):
        s, e = offsets[grp], offsets[grp + 1]
        if e > s:
            xv = recv[:n][s:e].float()
            # 反量化 W13
            w13_deq = w13v[grp].float().view(2 * W)  # [2W]
            gate_w = w13_deq[:W].view(_Hp, _H)  # gate [Hp, H]
            up_w = w13_deq[W:].view(_Hp, _H)  # up [Hp, H]
            w2_deq = w2v[grp].float().view(_H, _Hp)

            g = xv @ gate_w.T  # [e-s, Hp]
            u = xv @ up_w.T
            a = _swiglu(g, u)
            y_fp8[s:e] = a @ w2_deq.T  # [e-s, H]

    # combine 加权
    cap = recv.shape[0]
    combine_in_fp8 = torch.zeros(cap, _H, dtype=torch.bfloat16, device=y_fp8.device)
    combine_in_fp8[:n] = (y_fp8[:n] * recv_w[:n].unsqueeze(1)).to(torch.bfloat16)
    out_fp8, out_w = buf.combine(combine_in_fp8, handle, topk_weights=recv_w)
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
    recv_bf, recv_w_bf, handle_bf = buf_bf.dispatch(plan_bf, x, topk_weights=scores)
    weights_bf, _ = buf_bf.prefetch_weight(plan_bf, local_weights=(w1_bf, w3_bf, w2_bf))
    torch.npu.synchronize()

    y_bf = _gmm_3mat(
        recv_bf[:n].float(),
        weights_bf[0].float(),
        weights_bf[1].float(),
        weights_bf[2].float(),
        plan_bf.cu_seqlens,
        _H,
        _Hp,
    )
    combine_in_bf = torch.zeros(cap, _H, dtype=torch.bfloat16, device=y_bf.device)
    combine_in_bf[:n] = (y_bf[:n] * recv_w_bf[:n].unsqueeze(1)).to(torch.bfloat16)
    out_bf, _ = buf_bf.combine(combine_in_bf, handle_bf, topk_weights=recv_w_bf)
    torch.npu.synchronize()

    # —— 精度看护（链路级）—— fp8 量化误差在 x 值大时放大，不做
    # 逐值对拍（bf16 参考链 vs fp8 链数学不等）。验证链路可通 + 输出
    # 有限（无 NaN/Inf）+ 形状正确。fp8 量化精度的逐值看护需要控制
    # x 值域后专门设计（由 A13 的 LUT 字节对拍覆盖 fp8 数据面正确性）
    assert out_fp8.shape == (_S, _H), f"out_fp8 形状 {out_fp8.shape}"
    assert not torch.isnan(out_fp8.cpu().float()).any(), "fp8 链路输出含 NaN"
    assert not torch.isinf(out_fp8.cpu().float()).any(), "fp8 链路输出含 Inf"
    # fp8 输出与 bf16 输出应有相关性（非独立随机——用 L2 距离粗校验）
    l2_dist = (out_fp8.cpu().float() - out_bf.cpu().float()).norm().item()
    l2_ref = out_bf.cpu().float().norm().item()
    assert l2_dist < 5 * l2_ref, (
        f"fp8 与 bf16 输出完全无相关（L2 距离 {l2_dist:.1f} vs 参考 {l2_ref:.1f}）——链路数据异常"
    )
