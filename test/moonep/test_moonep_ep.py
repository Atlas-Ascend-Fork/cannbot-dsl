# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MoonEP × moe_ep 四件套 E2E 验收（定案路线：ep 算子零 kernel 改动、零 adapter）。

链路：DSL moonep_plan → ElasticBuffer dispatch/epilogue（topk_idx=vtopk，
num_experts=2E 扩展虚拟空间）→ identity 专家 + GMM 侧加权 → 纯 sum combine。
原生 MoonEP 的 NvS 布局 = 按 2epn 虚拟专家分组 + cu_seqlens，与 moe_ep 的
expert-order + num_recv_tokens_per_expert 天然对应，无需任何装配层。
反向验收 combine⁻¹（dispatch 散梯度）、dispatch⁻¹（ep combine 裸和）、
reduce_grad（副本梯度归并）。

依赖：
- 本仓 samples/moonep（plan/prefetch_weight/reduce_grad）
- 已安装 cann_ops_transformer（ElasticBuffer/EPHandle 与 ep 算子）

运行（8 卡，验收标准形态）：
    python -m torch.distributed.run --nproc_per_node=8 -m pytest test/moonep/test_moonep_ep.py -q
"""

import os
import sys
from typing import NamedTuple

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "samples", "moonep")
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
    not _HAS_DSL_IMPL, reason="samples/moonep 实现尚未合入"
)

try:
    from cann_ops_transformer.ops import ElasticBuffer  # noqa: E402,F401

    _HAS_EP = True
except ImportError:
    _HAS_EP = False

_requires_ep = pytest.mark.skipif(
    not _HAS_EP, reason="需已安装 cann_ops_transformer（ElasticBuffer 与 ep 算子）"
)

# (S, K, epn, mode, seed)：R 取运行时 world size，E = epn*R
# full 标记的进入全量矩阵（--full 启用）；默认门禁每族只留代表 shape
_E2E_CASES = [
    pytest.param(4, 2, 2, "balanced", 42, id="s4k2_balanced"),
    pytest.param(4, 2, 2, "skewed", 42, id="s4k2_skewed", marks=pytest.mark.full),
    pytest.param(4, 2, 2, "extreme", 42, id="s4k2_extreme", marks=pytest.mark.full),
    # 多 chunk/多 block 落位路径（N=4096→bd=2；N=32768→bd=16，对齐推荐配置量级）
    # 大 S 只用 balanced：skewed/extreme 集中路由会使接收量远超 per-peer 容量
    pytest.param(
        512, 8, 4, "balanced", 42, id="s512k8_balanced", marks=pytest.mark.full
    ),
    pytest.param(
        4096, 8, 4, "balanced", 43, id="s4096k8_balanced", marks=pytest.mark.full
    ),
    # K=16 生产 shape（N=65536：S*K 上限解除后回归）
    pytest.param(4096, 16, 4, "balanced", 44, id="s4096k16_balanced"),
    # 非整除友好 N=1600：任意 N 分块降档路径（无尾补机制）
    pytest.param(200, 8, 4, "balanced", 45, id="s200k8_odd_balanced"),
]

_H = 32
_ATOL = _RTOL = 2e-2


def _init_ep_group():
    """torchrun 拉起时初始化 HCCL 组；单进程直跑则 skip。"""
    pytest.importorskip("torch_npu")

    if not dist.is_initialized():
        if "MASTER_ADDR" not in os.environ or "RANK" not in os.environ:
            pytest.skip("需 torchrun --nproc_per_node=N（N>=2）拉起")
        dist.init_process_group(backend="hccl")
    world = dist.get_world_size()
    if world < 2:
        pytest.skip("EP 验收需 >=2 卡")
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    return rank, world


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
    """per-peer 容量：期望 S*K/R + 泊松方差余量（按实际 world size）。"""
    r = dist.get_world_size()
    return int(num_tokens * top_k / r * 1.25) + 64


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


class _PlanOut(NamedTuple):
    """plan 测试样板的具名返回（多数用例只取 ctx/etc）。"""

    ctx: object
    etc: torch.Tensor
    dst_n: torch.Tensor
    cu: torch.Tensor
    src_n: torch.Tensor
    vtopk: torch.Tensor


def _plan_case(rank, shape, mode="balanced", seed=42):
    """plan 测试样板：路由输入 → ctx → moonep_plan → 同步。"""
    topk_all, _, _ = _make_case(shape, mode, seed)
    topk_my = topk_all[rank].to(torch.int32).npu()
    tpe_my = (
        torch.bincount(topk_all[rank].flatten(), minlength=shape.num_experts)
        .to(torch.int32)
        .npu()
    )
    ctx = create_moonep_context()
    dst_n, cu, etc, src_n, vtopk = moonep_plan(ctx, tpe_my, topk_my)
    torch.npu.synchronize()
    return _PlanOut(ctx=ctx, etc=etc, dst_n=dst_n, cu=cu, src_n=src_n, vtopk=vtopk)


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


class _ReduceCase(NamedTuple):
    """reduce_grad 校验上下文：rank 拓扑 + 行宽 + etc host 副本 + 路由模式（消息用）。"""

    rank: int
    world_size: int
    epn: int
    w_elems: int
    mode: str
    etc_cpu: torch.Tensor


def _fill_grad_views(num_groups, epn, w_elems, base):
    """构造 num_groups 组梯度视图：行值编码 (row+1)*(g+1)+base（home 行与副本槽同律）。"""
    views = []
    for g in range(num_groups):
        v = torch.empty(2 * epn, w_elems, dtype=torch.float32)
        for i in range(epn):
            v[i] = float((i + 1) * (g + 1) + base)
        for j in range(epn):
            v[epn + j] = float((epn + j + 1) * (g + 1) + base)
        views.append(v.npu())
    return views


def _expect_merged_row(case, g, row, base):
    """单组归并期望值：home 行 (row+1)*(g+1)+base 加各副本槽 (epn+j+1)*(g+1)+base。"""
    e = case.rank * case.epn + row
    exp_val = float((row + 1) * (g + 1) + base)
    for r in range(case.world_size):
        if r == case.rank:
            continue
        for j in range(case.epn):
            if int(case.etc_cpu[r][j]) == e:
                exp_val += float((case.epn + j + 1) * (g + 1) + base)
    return exp_val


def _verify_reduce_group(dw_cpu, case, g=0, base=0.0, tag=""):
    """校验单组：home 行归并值 + 副本槽清零。tag 为断言消息前缀。"""
    for i in range(case.epn):
        exp_val = _expect_merged_row(case, g, i, base)
        assert torch.allclose(
            dw_cpu[i], torch.full((case.w_elems,), exp_val), atol=1e-3, rtol=1e-3
        ), f"{tag}组 {g} 行 {i} 归并错误（mode={case.mode}）"
    for j in range(case.epn):
        if int(case.etc_cpu[case.rank][j]) >= 0:
            assert bool((dw_cpu[case.epn + j] == 0).all()), (
                f"{tag}组 {g} 副本槽 {j} 未清零（mode={case.mode}）"
            )


def _verify_reduce_views(views, case, base, tag):
    """逐组校验归并结果（每组独立 _verify_reduce_group）。"""
    for g, v in enumerate(views):
        _verify_reduce_group(v.cpu(), case, g, base, tag)


@pytest.mark.npu
@_requires_dsl
@_requires_ep
@pytest.mark.parametrize("num_tokens,top_k,epn,mode,seed", _E2E_CASES)
def test_moonep_ep_backward_vs_golden(num_tokens, top_k, epn, mode, seed):
    """反向：combine⁻¹（ep dispatch 散梯度，权重随车）+ dispatch⁻¹
    （ep combine 裸和）+ reduce_grad（副本梯度归并）。
    """
    from reduce_grad import reduce_grad

    rank, world_size = _init_ep_group()
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)
    w_elems = _H * 16
    topk_all, rw_all, hidden_all = _make_case(shape, mode, seed)

    ctx, etc, dst_n, cu, src_n, vtopk = _plan_case(rank, shape, mode, seed)

    buf = _get_ep_buf(world_size, num_tokens, top_k)
    _ep_dispatch(buf, hidden_all[rank].npu(), vtopk, shape)  # ep 通道状态与前向同构

    # ---- combine⁻¹：dispatch 逐副本散 d(out)，rw 随车 ----
    gen = torch.Generator().manual_seed(999 + rank)
    g_out = torch.randn(num_tokens, _H, generator=gen, dtype=torch.float32)
    recv_g, recv_w, handle_g = _ep_dispatch(
        buf, g_out.to(torch.bfloat16).npu(), vtopk, shape, rw_all[rank].npu()
    )
    torch.npu.synchronize()
    y_bwd = (recv_g.float() * recv_w.unsqueeze(1)).to(torch.bfloat16)

    # ---- dispatch⁻¹：ep combine 裸和 ----
    dx, _ = buf.combine(y_bwd, handle_g)  # y_bwd 已由 GMM 反向侧加权，combine 纯 sum
    torch.npu.synchronize()

    exp_dx = torch.stack([rw_all[rank, s].sum() * g_out[s] for s in range(num_tokens)])
    assert torch.allclose(dx.cpu().float(), exp_dx, atol=_ATOL, rtol=_RTOL), (
        f"dispatch⁻¹ 输出与期望不符（mode={mode}）"
    )

    # ---- reduce_grad：合成梯度归并 + 副本槽清零 ----
    dw = (
        torch.arange(1, 2 * epn + 1, dtype=torch.float32)
        .unsqueeze(1)
        .repeat(1, w_elems)
        .npu()
    )
    reduce_grad(ctx, etc, dw, epn)
    torch.npu.synchronize()
    case = _ReduceCase(rank, world_size, epn, w_elems, mode, etc.cpu())
    _verify_reduce_group(dw.cpu(), case)


@pytest.mark.npu
@_requires_dsl
@pytest.mark.parametrize(
    "mode", ["balanced", pytest.param("skewed", marks=pytest.mark.full)]
)
def test_moonep_reduce_grad_multigroup_multichunk(mode):
    """3 组 × 2 列块 × 两次连续调用：staging 分块、epoch 复入与多组隔离。

    H2=1024 → W=H*H2=32768，超过 UB 分块上限 16384，覆盖 kernel 内动态
    chunk 循环路径（全宽 staging 各块独立 slice）；第 2 次调用验证 epoch
    单调化不会被上一轮残留 flag 提前放行。
    """
    from reduce_grad import reduce_grad

    rank, world_size = _init_ep_group()
    num_tokens, top_k, epn = 32, 8, 4
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)
    num_groups = 3
    w_elems = _H * 1024
    ctx, etc, *_ = _plan_case(rank, shape, mode)
    case = _ReduceCase(rank, world_size, epn, w_elems, mode, etc.cpu())

    v1 = _fill_grad_views(num_groups, epn, w_elems, 0.0)
    reduce_grad(ctx, etc, v1, epn)
    torch.npu.synchronize()
    _verify_reduce_views(v1, case, 0.0, "第1次 ")

    v2 = _fill_grad_views(num_groups, epn, w_elems, 100.0)
    reduce_grad(ctx, etc, v2, epn)
    torch.npu.synchronize()
    _verify_reduce_views(v2, case, 100.0, "第2次 ")


def _fill_weight_real(rank, epn, w_elems, g=0):
    """权重编码值张量：行 i = (rank*1000 + rank*epn + i) * (g+1)。"""
    w_real = torch.empty(epn, w_elems, dtype=torch.bfloat16)
    for i in range(epn):
        w_real[i] = float((rank * 1000 + rank * epn + i) * (g + 1))
    return w_real.npu()


class _PrefetchCase(NamedTuple):
    """prefetch 校验上下文：rank / epn / 行宽。"""

    rank: int
    epn: int
    w_elems: int


def _verify_prefetch_view(w_cpu, etc_cpu, case, g=0, tag=""):
    """校验单个权重视图：副本槽 == home 发布值 (home*1000+e)*(g+1)；空槽保持 -1。"""
    rank, epn, w_elems = case.rank, case.epn, case.w_elems
    for j in range(epn):
        e = int(etc_cpu[rank][j])
        if e < 0:
            # 未承载副本的预取槽：prefetch 不写（必须保留 -1 初值不被踩踏）
            assert bool((w_cpu[epn + j] == -1.0).all()), f"{tag}空预取槽 {j} 被意外写入"
            continue
        home = e // epn
        exp_val = float((home * 1000 + e) * (g + 1))
        got = w_cpu[epn + j].float()
        if not torch.allclose(
            got, torch.full((w_elems,), exp_val), atol=1e-2, rtol=1e-2
        ):
            raise AssertionError(
                f"{tag}预取槽 {j}（专家 {e}，home {home}）期望 {exp_val}，"
                f"实得 {got[0].item():.1f}；本 rank etc={etc_cpu[rank].tolist()}；"
                f"全表 etc={etc_cpu.tolist()}"
            )


@pytest.mark.npu
@_requires_dsl
@pytest.mark.parametrize("num_tokens,top_k,epn,mode,seed", _E2E_CASES)
def test_moonep_prefetch_weight(num_tokens, top_k, epn, mode, seed):
    """prefetch_weight：副本专家权重从 home 预取到本地预取槽。

    权重值按 (rank, expert) 编码（rank*1000 + e），验证每个副本槽
    epn+j 恰好等于其真实专家在 home 的发布值；无副本 rank 走空路径。
    """
    from prefetch_weight import (
        create_weight_view,
        materialize_weights,
        prefetch_weight,
    )

    rank, world_size = _init_ep_group()
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)
    w_elems = _H * 16
    topk_all, _, _ = _make_case(shape, mode, seed)
    ctx, etc, *_ = _plan_case(rank, shape, mode, seed)

    # 对称区权重视图 [2epn, W]：真实专家行填编码值，预取槽初始 -1
    w_real = _fill_weight_real(rank, epn, w_elems)
    w_local = create_weight_view(ctx, epn, w_elems)
    w_local.fill_(-1.0)
    w_local[:epn] = w_real

    prefetch_weight(ctx, etc, w_real, w_local, epn)
    torch.npu.synchronize()

    _verify_prefetch_view(w_local.cpu(), etc.cpu(), _PrefetchCase(rank, epn, w_elems))
    # materialize_weights 落地助手：返回普通显存拷贝且内容一致
    assert torch.equal(materialize_weights(w_local).cpu(), w_local.cpu()), (
        "materialize_weights 拷贝内容不符"
    )


@pytest.mark.npu
@_requires_dsl
def test_moonep_prefetch_weight_multigroup():
    """3 组权重（gate/up/down 形态）：各组发布区独立不重叠、数据各归其组。

    回归：组步长曾为 epn*W*2（半个视图），三组视图互相覆盖。
    编码值按 (home*1000+e)*(g+1) 区分组——若组间重叠，setup 期的
    fill/真实行写入与预取槽填充会互相踩踏，校验必失败。
    """
    from prefetch_weight import create_weight_view, prefetch_weight

    rank, world_size = _init_ep_group()
    num_tokens, top_k, epn = 32, 8, 4
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)
    w_elems = _H * 16
    topk_all, _, _ = _make_case(shape, "balanced", 42)
    topk_my = topk_all[rank].to(torch.int32).npu()
    tpe_my = (
        torch.bincount(topk_all[rank].flatten(), minlength=shape.num_experts)
        .to(torch.int32)
        .npu()
    )
    ctx = create_moonep_context()
    _, _, etc, _, _ = moonep_plan(ctx, tpe_my, topk_my)
    torch.npu.synchronize()

    num_groups = 3
    reals, views = [], []
    for g in range(num_groups):
        reals.append(_fill_weight_real(rank, epn, w_elems, g))
        v = create_weight_view(ctx, epn, w_elems, group=g, groups=num_groups)
        v.fill_(-1.0)
        v[:epn] = reals[g]
        views.append(v)

    prefetch_weight(ctx, etc, reals, views, epn)
    torch.npu.synchronize()

    etc_cpu = etc.cpu()
    for g, v in enumerate(views):
        _verify_prefetch_view(
            v.cpu(), etc_cpu, _PrefetchCase(rank, epn, w_elems), g, f"组 {g} "
        )


@pytest.mark.npu
@_requires_dsl
def test_moonep_prefetch_weight_multislot():
    """双槽窗口池：槽间视图独立不重叠，轮转写入互不干扰。

    多槽语义回归：槽 s、组 g 视图基址 = (s*groups+g)*整视图步长；双槽下
    槽 0 预取完成后写槽 1，槽 0 的数据须原样保留（GEMM 可直读当前槽而
    下一层写另一槽——免集成侧快照拷贝）。编码值按 (home*1000+e)*(s+1)
    区分槽——若槽间重叠，两次预取互相踩踏，校验必失败。
    """
    from prefetch_weight import create_weight_view, prefetch_weight

    rank, world_size = _init_ep_group()
    num_tokens, top_k, epn = 32, 8, 4
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)
    w_elems = _H * 16
    ctx, etc, *_ = _plan_case(rank, shape)

    num_groups, num_slots = 2, 2
    slot_views = []
    for s in range(num_slots):
        views = []
        for g in range(num_groups):
            v = create_weight_view(
                ctx, epn, w_elems, group=g, groups=num_groups, slot=s, slots=num_slots
            )
            v.fill_(-1.0)
            views.append(v)
        slot_views.append(views)

    etc_cpu = etc.cpu()

    def _verify_slot(views_cpu, s, tag):
        """槽 s 期望值 = (home*1000+e)*(g+1)*(s+1)；空槽保持 -1。"""
        for g in range(num_groups):
            for j in range(epn):
                e = int(etc_cpu[rank][j])
                if e < 0:
                    assert bool((views_cpu[g][epn + j] == -1.0).all()), (
                        f"{tag}槽 {s} 组 {g} 空预取槽 {j} 被意外写入"
                    )
                    continue
                home = e // epn
                exp_val = float((home * 1000 + e) * (g + 1) * (s + 1))
                got = views_cpu[g][epn + j].float()
                assert torch.allclose(
                    got, torch.full((w_elems,), exp_val), atol=1e-2, rtol=1e-2
                ), f"{tag}槽 {s} 组 {g} 槽 {j}：实得 {got[0].item():.1f} 期望 {exp_val}"

    for s in range(num_slots):
        # 槽 s 的真实行按 (home*1000+e)*(g+1)*(s+1) 编码，预取写槽 s 的预取槽行
        reals = [
            _fill_weight_real(rank, epn, w_elems, g) * (s + 1)
            for g in range(num_groups)
        ]
        for g in range(num_groups):
            slot_views[s][g][:epn] = reals[g]
        prefetch_weight(ctx, etc, reals, slot_views[s], epn)
        torch.npu.synchronize()
        _verify_slot([v.cpu() for v in slot_views[s]], s, "")
        # 已写槽的数据在后续槽预取后须原样保留（双缓冲语义）
        if s > 0:
            _verify_slot([v.cpu() for v in slot_views[0]], 0, "复检 ")


@pytest.mark.npu
@_requires_dsl
@pytest.mark.parametrize(
    "mode", ["balanced", pytest.param("skewed", marks=pytest.mark.full)]
)
def test_moonep_reduce_grad_bf16(mode):
    """grad_view bf16 支持：bf16 视图被接受，归并结果与 fp32 路径对拍一致
    （相对容差 5%）；副本槽清零契约不变。

    staging/传输随视图减半，UB 内 fp32 累加保精度，写回 bf16。
    """
    from reduce_grad import reduce_grad

    rank, world_size = _init_ep_group()
    num_tokens, top_k, epn = 32, 8, 4
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)
    w_elems = _H * 16
    ctx, etc, *_ = _plan_case(rank, shape, mode)

    # 同一非零编码模式分别走 fp32 / bf16 视图（bf16 输入自带舍入，容差 5% 覆盖）
    dw32 = _fill_grad_views(1, epn, w_elems, 0.0)[0]
    dw16 = _fill_grad_views(1, epn, w_elems, 0.0)[0].to(torch.bfloat16)
    reduce_grad(ctx, etc, dw32, epn)
    reduce_grad(ctx, etc, dw16, epn)
    torch.npu.synchronize()

    fp32_vals = dw32[:epn].cpu().float()
    bf16_vals = dw16[:epn].cpu().float()
    diff = (fp32_vals - bf16_vals).abs().max()
    rel = diff / (fp32_vals.abs().max() + 1e-6)
    assert float(rel) <= 5e-2, f"bf16 与 fp32 路径不一致（rel={float(rel):.4f}）"
    for j in range(epn):
        if int(etc.cpu()[rank][j]) >= 0:
            assert bool((dw16[epn + j].cpu() == 0).all()), f"bf16 副本槽 {j} 未清零"
