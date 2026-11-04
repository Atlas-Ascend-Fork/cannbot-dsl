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
# 有代表性的四档 shape：K=16 生产 shape / K=6+EPN=32 典型客户场景 /
# 多 chunk 多 block 落位路径 / 非整除友好小 N 分块降档路径
_E2E_CASES = [
    # K=16 生产 shape（N=65536：S*K 上限解除后回归）
    pytest.param(4096, 16, 4, "balanced", 44, id="s4096k16_balanced"),
    # 典型客户场景形状（K=6/EPN=32，8P 下 E=256）：独特分块路径
    # N=24576→nc=bd=48、E 因子结构→bdp=64（launch=64 块）——其他档位未覆盖
    pytest.param(4096, 6, 32, "balanced", 46, id="s4096k6_epn32_typical"),
    # 多 chunk/多 block 落位路径（N=32768→bd=16，对齐推荐配置量级）
    # 大 S 只用 balanced：skewed/extreme 集中路由会使接收量远超 per-peer 容量
    pytest.param(4096, 8, 4, "balanced", 43, id="s4096k8_balanced"),
    # 非整除友好 N=1600：任意 N 分块降档路径（无尾补机制）
    pytest.param(200, 8, 4, "balanced", 45, id="s200k8_odd_balanced"),
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
        ep_group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
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
    """per-peer 容量：期望 S*K/R + 泊松方差余量（按实际 world size）。

    发送侧同时要求 x.dim0 ≤ nmtr：K/R 较小时 S*K/R*1.25+64 < S，须与 S
    取 max（接入指南口径）——8P K6 形状曾因 3904 < S=4094 触发 dispatch
    tiling 拒绝（x dim0 超上限）。
    """
    r = dist.get_world_size()
    return max(num_tokens, int(num_tokens * top_k / r * 1.25) + 64)


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
    torch.npu.synchronize()

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
# 布局公式与启动块数约束（纯 host 计算，不启通信）
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_dsl
def test_moonep_layout_formulas():
    """布局公式算术自检（纯公式，无通信）。

    对称堆布局（plan 区 | 权重发布区(窗口池) | 尾）逐项对账。
    """
    from moonep_context import (
        LAYOUT_TAIL_BYTES,
        PLAN_REGION_END,
        prefetch_weight_off,
        reduce_stg_off,
    )

    epn, w_elems, num_groups, slots = 28, 3_670_016, 3, 2
    # 发布区（窗口池）：slots × groups × [2*epn, W] bf16 视图
    window_bytes = slots * num_groups * 2 * epn * w_elems * 2
    assert prefetch_weight_off() == PLAN_REGION_END, "发布区基址公式不符"
    publish_end = reduce_stg_off(epn, w_elems, num_groups, slots)
    assert publish_end == PLAN_REGION_END + window_bytes, "发布区终点公式不符"

    # E1 副本行段（dW 落点）：组 g 副本行基址 = 发布区 + g*整视图步长
    # + epn*W*elem——须落在组 g 视图内且不越发布区
    elem = 2
    view_step = 2 * epn * w_elems * elem
    for g in range(num_groups):
        rep = prefetch_weight_off() + g * view_step + epn * w_elems * elem
        assert rep >= prefetch_weight_off() + g * view_step, "副本行段基址低于组起点"
        assert (
            rep + epn * w_elems * elem <= prefetch_weight_off() + (g + 1) * view_step
        ), "副本行段越组界"
    # 堆终点 = 发布区 + 尾部对齐余量
    assert LAYOUT_TAIL_BYTES >= 32


# ---------------------------------------------------------------------------
# grid 栅栏类 kernel 的启动块数：必须 ≤ 本机有效 AIV 核数
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_dsl
def test_moonep_plan_blocks_within_core_count(monkeypatch):
    """global_sync_all 要求全部 block 同时常驻：启动块数不得超过 AIV 核数。

    回归 56 核机型上硬编码 `_MAX_PLAN_BLOCKS=64` 导致栅栏收不齐（死循环）：
    把核数分别钉成 64/56/28，检查分块与专家块的启动块数。纯 host 计算，
    不启 kernel。
    """
    import moonep_context as mc
    import moonep_plan as mp

    for cores in (64, 56, 28):
        monkeypatch.setattr(mc, "_RESIDENT_BLOCKS", cores)
        for rows in (131072, 8192, 2048):
            chunk, nc, bd = mp._plan_chunking(rows)
            bdp = mp._expert_blocks(112)
            launch = max(bd, bdp, 4)  # 4 = 典型 EP 组规模（≤ 核数）
            assert nc * chunk == rows, f"{cores} 核 rows={rows} 分块不整除"
            assert launch <= cores, (
                f"{cores} 核机型上 plan 启动 {launch} 块（bd={bd}, bdp={bdp}）："
                "栅栏要求全部 block 常驻，超核会挂死"
            )
    # 核数富余时不压低块数（64 核仍按 64 起块）
    monkeypatch.setattr(mc, "_RESIDENT_BLOCKS", 64)
    _, nc64, bd64 = mp._plan_chunking(131072)
    assert (nc64, bd64) == (64, 64), (nc64, bd64)
