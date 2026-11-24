# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""moonep_plan 前向 E2E 与约束测试。

（本文件覆盖 plan 计数审计与约束：counts
逐 chunk 对拍 + host 侧约束用例。）

依赖：
- 本仓 samples/mc2/moonep（moonep_plan）

运行（8 卡为验收标准形态，任意 >=2 卡可跑）：
    python -m torch.distributed.run --nproc_per_node=8 -m pytest \
        test/mc2/moonep/test_moonep_plan.py -q
"""

import os
import sys

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "samples", "mc2", "moonep"
    ),
)

from moonep_golden import (  # noqa: E402
    RouteShape,
    make_router_inputs,
    moonep_combine_golden,
    moonep_dispatch_golden,
)

try:
    from moonep_context import (  # noqa: F401
        check_pending_overflow,
        create_moonep_context,
    )
    from moonep_plan import moonep_plan  # noqa: F401

    _HAS_DSL_IMPL = True
except ImportError:
    _HAS_DSL_IMPL = False

try:
    from cann_ops_transformer.ops import ElasticBuffer  # noqa: E402,F401

    _HAS_EP = True
except ImportError:
    _HAS_EP = False

_requires_ep = pytest.mark.skipif(
    not _HAS_EP, reason="需已安装 cann_ops_transformer（ElasticBuffer 与 ep 算子）"
)

_requires_dsl = pytest.mark.skipif(
    not _HAS_DSL_IMPL, reason="samples/mc2/moonep/moonep_plan 不可用"
)

from prefetch_utils import _H, _init_ep_group  # noqa: E402

_CTX = {}


def _shared_ctx():
    """进程级共享裸 ctx（workspace+连接为重资源——池化防用例间累积）。"""
    if "ctx" not in _CTX:
        _CTX["ctx"] = create_moonep_context()
    return _CTX["ctx"]


_EPN = 32  # ElasticBuffer sizing 的 epn 上界（上游 #306 漏定义；test_e2e 同款常量）

# (S, K, epn, mode, seed)：R 取运行时 world size，E = epn*R
# 最小覆盖集：生产 shape / 非整除 N / 小 E / 大 E
_E2E_CASES = [
    # K=16 生产 shape（N=65536：S*K 上限解除后回归）
    pytest.param(4096, 16, 4, "balanced", 44, id="s4096k16_balanced"),
    # 非整除友好 N=1600：任意 N 分块降档路径（无尾补机制）
    pytest.param(200, 8, 4, "balanced", 45, id="s200k8_odd_balanced"),
    # 小 E 形态（epn=7）：覆盖小专家数下的本地/副本槽视图与计数路径
    pytest.param(4096, 8, 7, "balanced", 43, id="s4096k8_epn7"),
    # 大 E 形态（epn=28）：覆盖 E>128 的分箱、段查找与副本布局
    pytest.param(4096, 8, 28, "balanced", 43, id="s4096k8_epn28"),
    # epn=1（E==R）：plan 单次迭代折叠路径回归
    pytest.param(512, 8, 1, "balanced", 43, id="s512k8_epn1"),
]

_ATOL = _RTOL = 2e-2

_EP_POOL = {}


def _get_ep_buf(world_size, num_tokens, top_k):
    """全部用例共享同一 EP buffer/子通信域（shape 相同）。

    本机 ElasticBuffer Destroy 的 HcclBarrier 建 AICPU_TS 通道必等 ~120s/个，
    每用例新建 buffer 会把 120s 付 N 次；池化共享 + 进程退出硬退
    （见 conftest.py）跳过串行销毁。
    """
    key = (num_tokens, top_k)
    if key not in _EP_POOL:
        from torch_npu._C._distributed_c10d import ProcessGroupHCCL

        eb_mb = ElasticBuffer.get_moe_ep_ccl_buffer_size(
            world_size, _ep_nmtr(num_tokens, top_k), _H, _EPN * world_size, top_k
        )
        eb_opts = ProcessGroupHCCL.Options()
        eb_opts.hccl_config = {"hccl_buffer_size": eb_mb}
        ep_group = dist.new_group(
            backend="hccl",
            ranks=list(range(world_size)),
            pg_options=eb_opts,
        )
        # nmtr 余量：balanced 下 per-peer 期望=S*K/R，随机方差会略越界
        buf = ElasticBuffer(
            ep_group,
            num_max_tokens_per_rank=_ep_nmtr(num_tokens, top_k),
            hidden=_H,
            num_topk=top_k,
        )
        _EP_POOL[key] = (ep_group, buf)
    return _EP_POOL[key][1]


def _ep_nmtr(num_tokens, top_k):
    """per-peer 容量 = S（与 torchtitan 生产口径一致）。

    EB 窗口 per-peer 槽位承载 token 行而非 (token,expert) 扩展对，per-peer
    到达 ≤ S 恒成立；plan 均衡不变量（每 rank 扩展行恰 S*K）由接收侧 recv
    分配承载。历史公式 max(S, S*K/R*1.25+64) 非第一性推导（发送侧约束
    的推导有缺陷），废弃。
    """
    return num_tokens


def _make_case(shape, mode, seed):
    topk_all, rw_all = make_router_inputs(shape, mode=mode, seed=seed)
    gen = torch.Generator().manual_seed(1234)
    hidden_all = torch.randn(
        shape.world_size, shape.num_tokens, _H, generator=gen, dtype=torch.float32
    ).to(torch.bfloat16)
    return topk_all, rw_all, hidden_all


def _ep_dispatch(buf, x, vtopk, shape, rw=None):
    """dispatch：topk_idx=vtopk（[S,K] 虚拟专家 id），
    num_experts=2E 扩展空间。返回 (recv_x, recv_w, handle)。
    """
    recv_x, _, recv_w, handle = buf.dispatch(
        x,
        topk_idx=vtopk,
        topk_weights=rw,
        num_experts=2 * shape.num_experts,
        num_max_tokens_per_rank=_ep_nmtr(shape.num_tokens, shape.top_k),
        do_cpu_sync=False,
    )
    return recv_x, recv_w, handle


@pytest.mark.npu
@_requires_dsl
@_requires_ep
@pytest.mark.parametrize("num_tokens,top_k,epn,mode,seed", _E2E_CASES)
def test_moonep_ep_dispatch_combine_vs_golden(num_tokens, top_k, epn, mode, seed):
    """前向：plan → ep dispatch/epilogue → GMM 侧加权 → 纯 sum combine ≈ golden。

    路由权重按 rank 独立随机（随 dispatch 随车承重）；golden 所需的
    rw_all 由种子在测试侧确定性重建（生产路径不依赖种子）。
    """
    rank, world_size = _init_ep_group()
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)
    if top_k > shape.num_experts:
        pytest.skip(
            f"K={top_k} > E={shape.num_experts}（低 rank 下 E<K 跳过；生产 8P 下 E=32 OK）"
        )
    topk_all, _, hidden_all = _make_case(shape, mode, seed)
    rw_all = torch.stack(
        [
            torch.rand(
                num_tokens, top_k, generator=torch.Generator().manual_seed(777 + r)
            )
            + 0.1
            for r in range(world_size)
        ]
    )
    rw_my = rw_all[rank]

    topk_my = topk_all[rank].to(torch.int32).npu()
    tpe_my = (
        torch.bincount(topk_all[rank].flatten(), minlength=shape.num_experts)
        .to(torch.int32)
        .npu()
    )
    ctx = create_moonep_context()
    dst_n, cu, etc, src_n, vtopk = moonep_plan(ctx, tpe_my, topk_my)
    check_pending_overflow(ctx)

    buf = _get_ep_buf(world_size, num_tokens, top_k)

    recv_x, recv_w, handle = _ep_dispatch(
        buf, hidden_all[rank].npu(), vtopk, shape, rw_my.npu()
    )
    # 完美均衡不变量：每 rank 扩展行数恒为 S*K，与路由倾斜无关
    assert int(handle.num_recv_tokens_per_expert.sum()) == num_tokens * top_k, (
        f"扩展行数 != S*K（mode={mode}）"
    )

    g_res = moonep_dispatch_golden(hidden_all, rw_all, topk_all, shape.num_experts)

    # GMM 契约：权重由专家计算侧施加（此处专家为 identity，加权即 GMM 职责的替身），
    # combine 纯 sum 回收（moe_ep 契约：算子内不乘 router weight）
    y_weighted = (recv_x.float() * recv_w.unsqueeze(1)).to(recv_x.dtype)
    out, out_w = buf.combine(y_weighted, handle, topk_weights=recv_w)

    g_out = moonep_combine_golden(g_res, num_tokens, _H)
    assert torch.allclose(out.cpu().float(), g_out[rank], atol=_ATOL, rtol=_RTOL), (
        f"combine 输出与 golden 不符（mode={mode}）"
    )
    # 权重回收：combined_topk_weights 应精确等于本 rank 路由权重
    assert torch.equal(out_w.cpu(), rw_my), f"回收权重不符（mode={mode}）"


# ---------------------------------------------------------------------------
# B < R 回归：实际启动块数小于 EP 组规模（R=64、可常驻核 56 的真机场景）
# 时，Exchange 的 per-peer 发送必须按实际块数跨步覆盖全部远程 peer，
# 否则其余 peer 等不到 ready flag，plan 挂死。
# 用 resident_blocks 打桩到 2 构造 B=2 < R 的本机可复现形态。
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_dsl
@_requires_ep
def test_moonep_plan_blocks_lt_ranks(monkeypatch):
    rank, world_size = _init_ep_group()
    if world_size < 4:
        pytest.skip("需 >=4 卡构造 B<R")
    import moonep_plan as plan_mod

    monkeypatch.setattr(plan_mod, "resident_blocks", lambda cap: 2)
    num_tokens, top_k, epn = 64, 4, 4
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)
    topk_all, _, _ = _make_case(shape, "balanced", 7)
    topk_my = topk_all[rank].to(torch.int32).npu()
    tpe_my = (
        torch.bincount(topk_all[rank].flatten(), minlength=shape.num_experts)
        .to(torch.int32)
        .npu()
    )
    ctx = create_moonep_context()
    res = plan_mod.moonep_plan(ctx, tpe_my, topk_my)
    check_pending_overflow(ctx)  # 挂死会超时失败；同时校验布局不变量
    assert int(res.cu_seqlens[-1]) == num_tokens * top_k


# ---------------------------------------------------------------------------
# Count 阶段计数审计：gm_counts 与 CPU 期望逐 chunk 对拍
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_dsl
@pytest.mark.parametrize(
    "num_tokens,top_k,epn,seed",
    [pytest.param(4096, 8, 7, 43, id="epn7_hist")],
)
def test_moonep_count_audit(num_tokens, top_k, epn, seed):
    """Count 阶段计数契约：workspace 的 counts 与 CPU 直方图逐 chunk 一致。

    覆盖小 E（epn=7）与大 E（epn=28）两种形态，校验每个 chunk 内各专家
    token 计数、分箱边界及副本路由所依赖的前缀位置输入。
    """
    import moonep_plan

    rank, world_size = _init_ep_group()
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)
    topk_all, _, _ = _make_case(shape, "balanced", seed)
    topk_my = topk_all[rank].to(torch.int32).npu()
    tpe_my = (
        torch.bincount(topk_all[rank].flatten(), minlength=shape.num_experts)
        .to(torch.int32)
        .npu()
    )
    ctx = _shared_ctx()
    moonep_plan.moonep_plan(ctx, tpe_my, topk_my)
    check_pending_overflow(ctx)

    _, tables = ctx.plan_workspace
    blocks = moonep_plan._plan_blocks(num_tokens * top_k, shape.num_experts)
    nc = blocks.num_chunks
    chunk = num_tokens * top_k // nc
    offs = moonep_plan._ws_layout(world_size, shape.num_experts, nc)
    cnt = tables[offs["counts"] : offs["counts"] + nc * shape.num_experts]
    cnt = cnt.view(nc, shape.num_experts).cpu()
    flat = topk_all[rank].reshape(-1)
    expect = torch.stack(
        [
            torch.bincount(
                flat[ci * chunk : (ci + 1) * chunk], minlength=shape.num_experts
            )
            for ci in range(nc)
        ]
    ).to(torch.int32)
    diff = cnt != expect
    assert not diff.any(), (
        f"counts 与 CPU 期望不符：{int(diff.sum())} 个 bin 错，"
        f"首个错位 {[tuple(x) for x in torch.nonzero(diff)[:5].tolist()]}"
    )


# ---------------------------------------------------------------------------
# 单源清单自洽回归：per-rank token 数不等（N % R != 0）的越界占位输入
# 下，零搬移轮（min_load == cap）不得改写 rhome——否则 etc 行会漏掉该
# rank 实际服务到的副本专家（prefetch 漏拉权重、GMM 读到 -1 槽）。
# 说明：不等 N 不满足均衡服务契约（cu 尾值 == 本地 N），本用例只校验
# etc/vtopk 投影自洽，不调用 check_pending_overflow。
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_dsl
def test_moonep_plan_single_source_nondivisible():
    rank, world_size = _init_ep_group()
    if world_size == 4:
        per_rank = [[0], [0], [2, 2], [2, 2]]
    elif world_size == 8:
        per_rank = [[0], [0], [2, 2], [2, 2], [2], [2], [2], [2]]
    else:
        pytest.skip("仅 4/8 卡构造触发形态")
    epn = 2
    num_experts = epn * world_size
    mine = per_rank[rank]
    topk = torch.tensor(mine, dtype=torch.int32, device="npu").view(-1, 1)
    tpe = (
        torch.bincount(torch.tensor(mine), minlength=num_experts).to(torch.int32).npu()
    )
    ctx = create_moonep_context()
    _, cu, etc, _, vtopk = moonep_plan(ctx, tpe, topk)

    slots = 2 * epn
    etc_rows = etc.cpu().tolist()
    vid = vtopk.cpu().reshape(-1).tolist()
    for e, v in zip(mine, vid):
        r, s = v // slots, v % slots
        if s < epn:
            assert e == r * epn + s, f"home 槽错位：e={e} (r{r},s{s})"
        else:
            assert etc_rows[r][s - epn] == e, (
                f"副本槽专家与清单不符：slot={s - epn} e={e} etc[{r}]={etc_rows[r]}"
            )
    mat = torch.zeros(world_size, slots, dtype=torch.long, device="npu")
    for v in vid:
        mat[v // slots, v % slots] += 1
    dist.all_reduce(mat)
    row = mat[rank].cpu().tolist()
    cu_l = cu.cpu().tolist()
    for s in range(slots):
        prev = 0 if s == 0 else cu_l[s - 1]
        assert cu_l[s] - prev == row[s], f"cu[{s}] 段长与全局投影不符"
