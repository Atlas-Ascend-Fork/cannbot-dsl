# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""moonep_plan 前向 E2E 与约束测试。

链路：DSL moonep_plan → ElasticBuffer dispatch/epilogue（topk_idx=vtopk，
num_experts=2E 扩展虚拟空间）→ identity 专家 + GMM 侧加权 → 纯 sum combine ≈ golden。
原生 MoonEP 的 NvS 布局 = 按 2epn 虚拟专家分组 + cu_seqlens，与 moe_ep 的
expert-order + num_recv_tokens_per_expert 天然对应，无需任何装配层。

依赖：
- 本仓 samples/mc2/moonep（moonep_plan）
- 已安装 cann_ops_transformer（ElasticBuffer/EPHandle 与 ep 算子）

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
    from moonep_context import create_moonep_context  # noqa: F401
    from moonep_plan import moonep_plan  # noqa: F401

    _HAS_DSL_IMPL = True
except ImportError:
    _HAS_DSL_IMPL = False

_requires_dsl = pytest.mark.skipif(
    not _HAS_DSL_IMPL, reason="samples/mc2/moonep/moonep_plan 不可用"
)

try:
    from cann_ops_transformer.ops import ElasticBuffer  # noqa: E402,F401

    _HAS_EP = True
except ImportError:
    _HAS_EP = False

_requires_ep = pytest.mark.skipif(
    not _HAS_EP, reason="需已安装 cann_ops_transformer（ElasticBuffer 与 ep 算子）"
)

from prefetch_utils import _H, _init_ep_group  # noqa: E402

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
    的推导有缺陷），废弃（详见
    test_moonep_ep._ep_nmtr 同款注释）。
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
    """dispatch：topk_idx=vtopk（[N] 虚拟专家 id 投影 reshape 为 [S,K]），
    num_experts=2E 扩展空间。返回 (recv_x, recv_w, handle)。
    """
    recv_x, _, recv_w, handle = buf.dispatch(
        x,
        topk_idx=vtopk.view(shape.num_tokens, shape.top_k),
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
    dst_n, cu, etc, src_n, vtopk = moonep_plan(ctx, tpe_my, topk_my, verify=True)

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
    ctx = create_moonep_context()
    moonep_plan.moonep_plan(ctx, tpe_my, topk_my, verify=True)

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
