# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MoonEP plan/prefetch/reduce E2E 验收（DSL 原生三算子形态）。

（本文件覆盖：DSL moonep_plan（2E 扩展
虚拟空间路由 + cu_seqlens 分组）、prefetch_weight（跨 rank 副本预取）、
reduce_grad（副本梯度 v4 pull 归并）及布局公式单元验收。）

依赖：
- 本仓 samples/moonep（plan/prefetch_weight/reduce_grad）

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
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "samples", "mc2", "moonep"
    ),
)

from conftest import fill_and_repush  # noqa: E402

_EPN = 32  # ElasticBuffer sizing 的 epn 上界（上游 #306 漏定义；test_e2e 同款常量）
from moonep_golden import (  # noqa: E402
    RouteShape,
    make_router_inputs,
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

# (S, K, epn, mode, seed)：R 取运行时 world size，E = epn*R
# full 标记的进入全量矩阵（--full 启用）；默认门禁每族只留代表 shape
_E2E_CASES = [
    pytest.param(4, 2, 2, "balanced", 42, id="s4k2_balanced"),
    pytest.param(4, 2, 2, "skewed", 42, id="s4k2_skewed", marks=pytest.mark.full),
    pytest.param(4, 2, 2, "extreme", 42, id="s4k2_extreme", marks=pytest.mark.full),
    # 多 chunk/多 block 落位路径（N=4096→bd=2；N=32768→bd=16，对齐推荐配置量级）
    # 大 S 只用 balanced：历史排除理由"集中路由使接收量远超 per-peer 容量"
    # 基于 per-peer 扩展对模型（已证伪——EB 槽位承载 token 行，per-peer ≤ S
    # 恒成立）；nmtr=S 收口后该排除是否仍必要未验证，暂保守保留
    pytest.param(
        512, 8, 4, "balanced", 42, id="s512k8_balanced", marks=pytest.mark.full
    ),
    pytest.param(
        4096, 8, 4, "balanced", 43, id="s4096k8_balanced", marks=pytest.mark.full
    ),
    # K=16 生产 shape（N=65536：S*K 上限解除后回归）
    pytest.param(
        4096,
        16,
        4,
        "balanced",
        44,
        id="s4096k16_balanced",
        marks=pytest.mark.skip(
            reason="K=16 在 2-rank 下 E=2×epn=8<K；生产 8P 下 E=32 OK——低 rank 跳过"
        ),
    ),
    # 典型客户场景形状（K=6/EPN=32，E=128@4P/256@8P）：独特分块路径
    # N=24576→nc=bd=48、E 因子结构→bdp=64（launch=64 块）——其他档位未覆盖
    pytest.param(4096, 6, 32, "balanced", 46, id="s4096k6_epn32_typical"),
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


def _assert_cu_matches_vtopk(cu, vtopk, num_experts):
    hist = torch.bincount(vtopk.to(torch.int64), minlength=2 * num_experts)
    dist.all_reduce(hist)
    rows = cu.numel()
    rank = dist.get_rank()
    expected = torch.cumsum(hist[rank * rows : (rank + 1) * rows], dim=0).to(
        torch.int32
    )
    assert torch.equal(cu.cpu(), expected.cpu()), "cu_seqlens 与全组 vtopk 行计数不一致"


_EP_POOL = {}
_POOL_CAP = 3  # HCCL 注册限额驱动的池上限（LRU 驱逐）


def _get_ep_buf(world_size, num_tokens, top_k, num_experts):
    """全部用例共享 DSL buffer 池（同 shape 键控；UbMem 通路）。

    Buffer 构造为集体调用（含通信连接建立）——池化共享免重复
    建连；进程退出统一回收（destroy 断开连接，秒级）。
    """
    key = (num_tokens, top_k, num_experts)
    # LRU 上限：HCCL 每通信器内存注册有限额——池超限驱逐最旧 buffer
    # （destroy 断连释放注册；构造/驱逐均为集体序，全 rank 同步安全）
    while len(_EP_POOL) >= _POOL_CAP:
        _old_key = next(iter(_EP_POOL))
        _old_buf = _EP_POOL.pop(_old_key)
        _old_buf.destroy()
    if key not in _EP_POOL:
        from moonep_buffer import Buffer

        _EP_POOL[key] = Buffer(
            num_tokens,
            _H,
            _H,  # Hp=H（W 上界不参与实算——本文件走 plan/prefetch/reduce）
            top_k,
            num_experts,
            num_ep_ranks=dist.get_world_size(),
            group=dist.group.WORLD,
        )
    return _EP_POOL[key]


def _make_case(shape, mode, seed):
    topk_all, rw_all = make_router_inputs(shape, mode=mode, seed=seed)
    gen = torch.Generator().manual_seed(1234)
    hidden_all = torch.randn(
        shape.world_size, shape.num_tokens, _H, generator=gen, dtype=torch.float32
    ).to(torch.bfloat16)
    return topk_all, rw_all, hidden_all


class _PlanOut(NamedTuple):
    """plan 测试样板的具名返回（多数用例只取 ctx/etc）。"""

    ctx: object
    etc: torch.Tensor
    dst_n: torch.Tensor
    cu: torch.Tensor
    src_n: torch.Tensor
    vtopk: torch.Tensor


_CTX = {}


def _test_workspace():
    """覆盖本文件用例矩阵上界的精算 workspace（分配即所需——不再粗预留）。

    上界布局族：3 组 × 114688 宽（H×16）× 2 槽 × epn=28、bf16 梯度/权重
    ——用例矩阵（_E2E_CASES 至 114688 宽变体）全部落位其内；逐用例更
    精确的形状精算收益 <1 次窄分配，取矩阵上界一次成型。
    """
    from moonep_layout import HEAP_BASE, LAYOUT_TAIL_BYTES, reduce_region_end

    return (
        HEAP_BASE
        + reduce_region_end(28, 114688, 3, 2, grad_bytes_per_elem=2)
        + LAYOUT_TAIL_BYTES
    )


def _shared_ctx():
    """进程级共享裸 ctx（默认布局族：3 组 × 512 宽 × 1 槽）。"""
    if "ctx" not in _CTX:
        _CTX["ctx"] = create_moonep_context(workspace_bytes=_test_workspace())
    return _CTX["ctx"]


def _fresh_ctx():
    """布局变体用例专用裸 ctx（用后 close）。

    layout_groups/组宽/槽位在 ctx 上首调冻结——跨族共享会触发布局
    守卫（d5/d6：2 组/32768 宽/双槽用例连环被拦）。专用 ctx 用后
    即关，防 HCCL 注册累积（HcclCommMemReg 限额，池上限 3 的同源约束）。
    """
    return create_moonep_context(workspace_bytes=_test_workspace())


def _assert_one_source_etc(etc_cpu, epn):
    """plan 契约：每个 destination rank 的副本槽至多来自一个 home rank。"""
    for rank, row in enumerate(etc_cpu.tolist()):
        homes = {int(e) // epn for e in row if int(e) >= 0}
        assert len(homes) <= 1, (
            f"etc[{rank}] 同时来自多个 home rank：{sorted(homes)}（一卡一源契约被破坏）"
        )


def _plan_case(rank, shape, mode="balanced", seed=42, ctx=None):
    """plan 测试样板：路由输入 → ctx（缺省共享族；变体用例传专用 ctx）
    → moonep_plan → 同步。"""
    topk_all, _, _ = _make_case(shape, mode, seed)
    topk_my = topk_all[rank].to(torch.int32).npu()
    tpe_my = (
        torch.bincount(topk_all[rank].flatten(), minlength=shape.num_experts)
        .to(torch.int32)
        .npu()
    )
    ctx = ctx if ctx is not None else _shared_ctx()
    dst_n, cu, etc, src_n, vtopk = moonep_plan(ctx, tpe_my, topk_my)
    torch.npu.synchronize()
    etc_cpu = etc.cpu()
    _assert_one_source_etc(etc_cpu, etc_cpu.shape[1])
    return _PlanOut(ctx=ctx, etc=etc, dst_n=dst_n, cu=cu, src_n=src_n, vtopk=vtopk)


class _ReduceCase(NamedTuple):
    """reduce_grad 校验上下文：rank 拓扑 + 行宽 + etc host 副本 + 路由模式（消息用）。"""

    rank: int
    world_size: int
    epn: int
    w_elems: int
    mode: str
    etc_cpu: torch.Tensor


def _full_dw_tensors(num_groups, epn, w_elems, base, grad_dtype=torch.float32):
    """full-dW 契约：[2*epn, W] 张量（GMM 反传原生形态）。

    home 行 i（[0, epn)）值 (i+1)*(g+1)+base（种子）；副本行 epn+j 值
    (epn+j+1)*(g+1)+base（dW 尾段——reduce 内部入槽，owner 远端读槽
    原子累加进 home）。归并期望公式 _expect_merged_row 不变。
    """
    grads = []
    for g in range(num_groups):
        t = torch.empty(2 * epn, w_elems, dtype=grad_dtype)
        for i in range(2 * epn):
            t[i] = float((i + 1) * (g + 1) + base)
        grads.append(t.npu())
    return grads


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


def _verify_reduce_group(grad_cpu, case, g=0, base=0.0, tag=""):
    """校验单组（full-dW 契约）：home 段归并值 + 副本段保持调用方原值。"""
    _assert_one_source_etc(case.etc_cpu, case.epn)
    epn = case.epn
    for i in range(epn):
        exp_val = _expect_merged_row(case, g, i, base)
        assert torch.allclose(
            grad_cpu[i], torch.full((case.w_elems,), exp_val), atol=1e-3, rtol=1e-3
        ), f"{tag}组 {g} 行 {i} 归并错误（mode={case.mode}）"
    for j in range(epn):
        exp_tail = float((epn + j + 1) * (g + 1) + base)
        assert torch.allclose(
            grad_cpu[epn + j].float(),
            torch.full((case.w_elems,), exp_tail),
            atol=1e-2,
            rtol=1e-2,
        ), (
            f"{tag}组 {g} 副本行 {j} 被改写（full-dW 契约：副本段只读，"
            f"应保持调用方写入值 {exp_tail}）"
        )


def _verify_reduce_views(grads, case, base, tag):
    """逐组校验归并结果（每组独立 _verify_reduce_group）。"""
    for g, t in enumerate(grads):
        _verify_reduce_group(t.cpu(), case, g, base, tag)


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
    # 32768 宽变体：专用 ctx（组宽首调冻结——共享族会拦截；用后即关
    # 防 HCCL 注册累积）
    ctx, etc, *_ = _plan_case(rank, shape, mode, ctx=_fresh_ctx())
    case = _ReduceCase(rank, world_size, epn, w_elems, mode, etc.cpu())

    grads1 = _full_dw_tensors(num_groups, epn, w_elems, 0.0)
    reduce_grad(ctx, etc, grads1, epn)
    torch.npu.synchronize()
    _verify_reduce_views(grads1, case, 0.0, "第1次 ")

    # 第 2 次调用：full-dW 重填（尾段=槽源重写），验证 epoch 复入 + 槽值覆盖语义
    grads2 = _full_dw_tensors(num_groups, epn, w_elems, 100.0)
    reduce_grad(ctx, etc, grads2, epn)
    torch.npu.synchronize()
    _verify_reduce_views(grads2, case, 100.0, "第2次 ")
    ctx.close()


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
    _assert_one_source_etc(etc_cpu, case.epn)
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
    """prefetch_weight：副本专家权重从 home 预取到本地预取槽（3 组形态）。

    权重值按 (rank, expert, 组) 编码（(rank*1000 + e)*(g+1)），验证每个
    副本槽 epn+j 恰好等于其真实专家在 home 的发布值；无副本 rank 走空
    路径。参数化形状矩阵（s4k2/s4096k6/s200k8 等）承载 etc 矩形覆盖。
    """
    from prefetch_weight import prefetch_weight

    rank, world_size = _init_ep_group()
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)
    w_elems = _H * 16
    topk_all, _, _ = _make_case(shape, mode, seed)
    ctx, etc, *_ = _plan_case(rank, shape, mode, seed)

    # prefetch 直传 reals（2/3 组自适应——视图按各组行宽创建并返回）
    num_groups = 3
    reals = [_fill_weight_real(rank, epn, w_elems, g) for g in range(num_groups)]
    views_w, views_s = prefetch_weight(ctx, etc, reals, epn)
    # 空槽哨兵（库内 fill 已移除）：fill → barrier → 复推
    fill_and_repush(ctx, etc, (views_w, views_s), epn)

    etc_cpu = etc.cpu()
    for g, v in enumerate(views_w):
        _verify_prefetch_view(v.cpu(), etc_cpu, _PrefetchCase(rank, epn, w_elems), g)
    # home 行同步（功能校验，非精度对比）：视图前 epn 行必须由 prefetch 写入
    # real_weights；期望值按 (rank, 行号, 组) 编码公式独立计算，不引用 reals。
    for g, v in enumerate(views_w):
        exp_home = (
            torch.tensor(
                [float((rank * 1000 + rank * epn + i) * (g + 1)) for i in range(epn)],
                dtype=torch.bfloat16,
            )
            .unsqueeze(1)
            .expand(epn, w_elems)
        )
        assert torch.equal(v.cpu()[:epn], exp_home), (
            f"组 {g} home 行未被 prefetch 写入（预期 = real_weights 的 "
            "(rank*1000+expert)*(g+1) 编码）"
        )


@pytest.mark.npu
@_requires_dsl
def test_moonep_one_source_full_slot_fanout():
    """满副本槽 + 单 home 多目的 fan-out + home 多 sender 梯度归并。"""
    from prefetch_weight import prefetch_weight
    from reduce_grad import reduce_grad

    rank, world_size = _init_ep_group()
    num_tokens, top_k, epn = 32, 8, 4
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)
    w_elems = _H * 16
    ctx, *_ = _plan_case(rank, shape, "balanced", 2026)

    etc_cpu = torch.full((world_size, epn), -1, dtype=torch.int32)
    for dst in range(1, world_size):
        etc_cpu[dst] = torch.arange(epn, dtype=torch.int32)
    _assert_one_source_etc(etc_cpu, epn)
    etc = etc_cpu.npu()

    w_reals = [_fill_weight_real(rank, epn, w_elems, g) for g in range(3)]
    w_views_w, w_views_s = prefetch_weight(ctx, etc, w_reals, epn)
    # 空槽哨兵（库内 fill 已移除）：fill → barrier → 复推
    fill_and_repush(ctx, etc, (w_views_w, w_views_s), epn)
    for g, v in enumerate(w_views_w):
        _verify_prefetch_view(
            v.cpu(), etc_cpu, _PrefetchCase(rank, epn, w_elems), g, "full-slot "
        )

    grads = _full_dw_tensors(3, epn, w_elems, 0.0)
    reduce_grad(ctx, etc, grads, epn)
    torch.npu.synchronize()
    case = _ReduceCase(rank, world_size, epn, w_elems, "full-slot", etc_cpu)
    _verify_reduce_views(grads, case, 0.0, "full-slot ")


@pytest.mark.npu
@_requires_dsl
def test_moonep_prefetch_weight_multigroup():
    """3 组权重（gate/up/down 形态）：各组发布区独立不重叠、数据各归其组。

    回归：组步长曾为 epn*W*2（半个视图），三组视图互相覆盖。
    编码值按 (home*1000+e)*(g+1) 区分组——若组间重叠，setup 期的
    fill/真实行写入与预取槽填充会互相踩踏，校验必失败。
    """
    from prefetch_weight import prefetch_weight

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
    ctx = _shared_ctx()
    _, _, etc, _, _ = moonep_plan(ctx, tpe_my, topk_my)
    torch.npu.synchronize()

    num_groups = 3
    reals = [_fill_weight_real(rank, epn, w_elems, g) for g in range(num_groups)]
    views_w, views_s = prefetch_weight(ctx, etc, reals, epn)
    # 空槽哨兵（库内 fill 已移除）：fill → barrier → 复推
    fill_and_repush(ctx, etc, (views_w, views_s), epn)

    etc_cpu = etc.cpu()
    for g, v in enumerate(views_w):
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
    from prefetch_weight import prefetch_weight

    rank, world_size = _init_ep_group()
    num_tokens, top_k, epn = 32, 8, 4
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)
    w_elems = _H * 16
    # 双槽变体：专用 ctx（window_slots 首调冻结——共享族会拦截）
    ctx, etc, *_ = _plan_case(rank, shape, ctx=_fresh_ctx())

    num_groups, num_slots = 3, 2
    slot_views = []  # prefetch 返回视图按槽缓存（首调创建）

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
        slot_w, _slot_s = prefetch_weight(ctx, etc, reals, epn, slot=s, slots=num_slots)
        slot_views.append(slot_w)
        # 空槽哨兵（库内 fill 已移除）：fill → barrier → 同槽复推
        fill_and_repush(ctx, etc, (slot_w, _slot_s), epn, slot=s, slots=num_slots)
        _verify_slot([v.cpu() for v in slot_w], s, "")
        # 已写槽的数据在后续槽预取后须原样保留（双缓冲语义）
        if s > 0:
            _verify_slot([v.cpu() for v in slot_views[0]], 0, "复检 ")
    ctx.close()


@pytest.mark.full
@pytest.mark.npu
@_requires_dsl
def test_moonep_prefetch_weight_dualslot():
    """双槽窗口池（slot=1/slots=2）+ 推荐零拷贝形态。

    槽 s 的组 g 基址 = prefetch_weight_off + (s*groups+g)*整视图步长；只写
    槽 1 的预取槽，槽 1 的真实行与空槽语义与单槽一致。real_weights 直接
    取视图 home 行切片（参数建在视图上的推荐形态）——覆盖 sync_transport
    的同址零拷贝跳过路径。
    """
    from prefetch_weight import prefetch_weight

    rank, world_size = _init_ep_group()
    num_tokens, top_k, epn = 64, 8, 4
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)
    w_elems = _H * 16
    num_groups = 3
    topk_all, _ = make_router_inputs(shape, mode="balanced", seed=7)
    topk_my = topk_all[rank].to(torch.int32).npu()
    tpe_my = (
        torch.bincount(topk_all[rank].flatten(), minlength=shape.num_experts)
        .to(torch.int32)
        .npu()
    )
    ctx = _fresh_ctx()  # 双槽变体：专用 ctx（window_slots 首调冻结）
    _, _, etc, _, _ = moonep_plan(ctx, tpe_my, topk_my)
    torch.npu.synchronize()

    # 槽 0 首建（显式 real——视图创建 + home 拷贝 + 推送）
    reals0 = [_fill_weight_real(rank, epn, w_elems, g) for g in range(num_groups)]
    prefetch_weight(ctx, etc, reals0, epn, slot=0, slots=2)
    # 槽 1：显式 real 建视图 + 推送；随后以 home 行切片为零拷贝 real 复推
    # （data_ptr 同址——sync_transport 跳过，覆盖推荐零拷贝形态）
    reals1 = [_fill_weight_real(rank, epn, w_elems, g) for g in range(num_groups)]
    views1_w, _views1_s = prefetch_weight(ctx, etc, reals1, epn, slot=1, slots=2)
    # 空槽哨兵（库内 fill 已移除）：fill → barrier → 零拷贝复推（home 行
    # 切片——同址跳过，即本用例原本要覆盖的推荐零拷贝形态）
    fill_and_repush(ctx, etc, (views1_w, _views1_s), epn, slot=1, slots=2)
    etc_cpu = etc.cpu()
    for g, view in enumerate(views1_w):
        _verify_prefetch_view(
            view.cpu(), etc_cpu, _PrefetchCase(rank, epn, w_elems), g, f"双槽 组 {g} "
        )
    ctx.close()


@pytest.mark.npu
@_requires_dsl
@pytest.mark.parametrize(
    "mode", ["balanced", pytest.param("skewed", marks=pytest.mark.full)]
)
def test_moonep_reduce_grad_bf16(mode):
    """grad_view bf16 支持：bf16 视图被接受，归并结果与 fp32 路径对拍一致
    （相对容差 5%）；副本槽不清零契约不变（保持 FFN 写入值）。

    staging/传输随视图减半，UB 内 fp32 累加保精度，写回 bf16。
    """
    from reduce_grad import reduce_grad

    rank, world_size = _init_ep_group()
    num_tokens, top_k, epn = 32, 8, 4
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)
    w_elems = _H * 16
    # 2 组变体：专用 ctx（layout_groups 首调冻结——共享族会拦截）
    ctx, etc, *_ = _plan_case(rank, shape, mode, ctx=_fresh_ctx())

    # 同一非零编码模式分别走 fp32 / bf16（bf16 输入自带舍入，容差 5% 覆盖）。
    # dtype 从张量解析：fp32 轮先固化结果再跑 bf16 轮（同槽基址顺序复用）
    grads32 = _full_dw_tensors(2, epn, w_elems, 0.0, grad_dtype=torch.float32)
    reduce_grad(ctx, etc, grads32, epn)
    torch.npu.synchronize()
    fp32_vals = grads32[0][:epn].cpu().float()

    grads16 = _full_dw_tensors(2, epn, w_elems, 0.0, grad_dtype=torch.bfloat16)
    reduce_grad(ctx, etc, grads16, epn)
    torch.npu.synchronize()
    bf16_vals = grads16[0][:epn].cpu().float()
    diff = (fp32_vals - bf16_vals).abs().max()
    rel = diff / (fp32_vals.abs().max() + 1e-6)
    assert float(rel) <= 5e-2, f"bf16 与 fp32 路径不一致（rel={float(rel):.4f}）"
    for j in range(epn):
        exp_tail = float((epn + j + 1) + 0.0)
        assert torch.allclose(
            grads16[0][epn + j].cpu().float(),
            torch.full((w_elems,), exp_tail),
            atol=1e-2,
        ), f"bf16 副本行 {j} 被改写（full-dW 只读契约）"
    ctx.close()


@pytest.mark.npu
@_requires_dsl
@pytest.mark.parametrize("mode", ["balanced"])
def test_moonep_reduce_grad_fp16(mode):
    """grad_view fp16 支持：fp16 视图被接受，归并结果与 fp32 路径对拍一致
    （相对容差 5%——fp16 尾数 10 位与 bf16 同容差面）；副本槽不清零契约
    不变。16 位路径与 bf16 同构（B16↔B32 unpack/pack，dtype 三选一）。
    """
    from reduce_grad import reduce_grad

    rank, world_size = _init_ep_group()
    num_tokens, top_k, epn = 32, 8, 4
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)
    w_elems = _H * 16
    ctx, etc, *_ = _plan_case(rank, shape, mode, ctx=_fresh_ctx())

    grads32 = _full_dw_tensors(2, epn, w_elems, 0.0, grad_dtype=torch.float32)
    reduce_grad(ctx, etc, grads32, epn)
    torch.npu.synchronize()
    fp32_vals = grads32[0][:epn].cpu().float()

    grads_h = _full_dw_tensors(2, epn, w_elems, 0.0, grad_dtype=torch.float16)
    reduce_grad(ctx, etc, grads_h, epn)
    torch.npu.synchronize()
    fp16_vals = grads_h[0][:epn].cpu().float()
    diff = (fp32_vals - fp16_vals).abs().max()
    rel = diff / (fp32_vals.abs().max() + 1e-6)
    assert float(rel) <= 5e-2, f"fp16 与 fp32 路径不一致（rel={float(rel):.4f}）"
    for j in range(epn):
        exp_tail = float((epn + j + 1) + 0.0)
        assert torch.allclose(
            grads_h[0][epn + j].cpu().float(),
            torch.full((w_elems,), exp_tail),
            atol=2e-2,
        ), f"fp16 副本行 {j} 被改写（full-dW 只读契约）"
    ctx.close()


@pytest.mark.npu
@_requires_dsl
def test_moonep_precision_pipeline_fp32_then_bf16():
    """三算子数值漏斗：plan 落点 → prefetch 副本槽 → reduce 归并（fp32 紧接 bf16）。

    精度口径：每个被测张量都与独立计算的期望值比较（路由表 + 编码公式），
    不拿算子输入当参照，也不放宽容差。
      - prefetch 副本槽逐元素 == bf16(编码值)，空槽仍为 -1；
      - reduce home 行逐元素 == 副本和（整数编码，bf16 可精确表示，故用逐位
        相等而非 allclose），副本槽保持 FFN 写入值（v4 契约：不清零）；
      - 同一输入的 fp32 / bf16 两次归并必须逐位一致——覆盖"先 fp32 后 bf16
        复用同一 staging/flags"的脏数据场景。

    不参与对比的部分（显式跳过，不做假断言）：视图 home 行（前 epn 行）是
    `sync_transport` 从 real_weights 复制来的输入侧数据，不是算子计算结果；
    其写入正确性由 `test_moonep_prefetch_weight*` 的独立断言覆盖。
    """
    from prefetch_weight import prefetch_weight
    from reduce_grad import reduce_grad

    rank, world_size = _init_ep_group()
    num_tokens, top_k, epn = 256, 8, 8
    num_groups = 3
    w_elems = _H * 16
    num_rows = num_tokens * top_k
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)

    topk_all, _ = make_router_inputs(shape, mode="balanced", seed=2026)
    topk_my = topk_all[rank].to(torch.int32).npu()
    tpe_my = (
        torch.bincount(topk_all[rank].flatten(), minlength=shape.num_experts)
        .to(torch.int32)
        .npu()
    )

    ctx = _fresh_ctx()  # 2 组变体：专用 ctx
    _, cu, etc, _, vtopk = moonep_plan(ctx, tpe_my, topk_my)
    torch.npu.synchronize()

    # ---- plan：均衡不变量 + 落点承载（索引类输出，无容差可言）----
    assert int(cu[-1].item()) == num_rows, (
        f"cu 尾值 {int(cu[-1].item())} != N={num_rows}（副本槽溢出或贪心失衡）"
    )
    etc_cpu = etc.cpu()
    v_cpu = vtopk.cpu()
    for r in range(world_size):
        remote_homes = {int(e) // epn for e in etc_cpu[r].tolist() if int(e) >= 0}
        assert len(remote_homes) <= 1, (
            f"rank {r} 的副本槽来自多个 home rank：{sorted(remote_homes)}"
        )
    assert int(v_cpu.min()) >= 0 and int(v_cpu.max()) < 2 * shape.num_experts, (
        f"vtopk 越界：[{int(v_cpu.min())},{int(v_cpu.max())}) 应为 [0,{2 * shape.num_experts})"
    )
    _assert_cu_matches_vtopk(cu, vtopk, shape.num_experts)
    for i, (v, e) in enumerate(zip(v_cpu.tolist(), topk_all[rank].flatten().tolist())):
        d, slot = v // (2 * epn), v % (2 * epn)
        if slot < epn:
            assert d == e // epn and slot == e % epn, (
                f"token {i} 的 home 编码错误：expert={e}, vtopk={v}"
            )
        else:
            assert int(etc_cpu[d, slot - epn]) == e, (
                f"token {i} 的副本槽编码错误：expert={e}, vtopk={v}, "
                f"etc[{d},{slot - epn}]={int(etc_cpu[d, slot - epn])}"
            )
    # etc 语义护栏（下面 prefetch 期望值依赖它）：副本表只记别人的专家
    for r in range(world_size):
        for j in range(epn):
            e = int(etc_cpu[r][j])
            assert e == -1 or e // epn != r, (
                f"etc[{r}][{j}]={e} 是本 rank 的 home 专家，不应出现在副本表"
            )
    # 覆盖面护栏：没有副本时下面的"期望值"会退化成平凡比较，此时必须失败而非空过
    total_copies = sum(1 for row in etc_cpu.tolist() for e in row if int(e) >= 0)
    assert total_copies > 0, (
        "本用例未产生任何副本槽：prefetch/reduce 的副本路径未被覆盖"
    )
    # 本 rank 槽位/行的期望校验条数（与路由表独立对齐，防止空循环假装通过）
    local_slots = [int(e) for e in etc_cpu[rank].tolist() if int(e) >= 0]
    my_experts_with_copies = {
        i
        for i in range(epn)
        if any(
            int(etc_cpu[r][j]) == rank * epn + i
            for r in range(world_size)
            if r != rank
            for j in range(epn)
        )
    }

    # ---- prefetch：副本槽逐元素精确比对（期望值由编码公式独立给出）----
    reals = [_fill_weight_real(rank, epn, w_elems, g) for g in range(num_groups)]
    views_w, views_s = prefetch_weight(ctx, etc, reals, epn)
    # 空槽哨兵（库内不做 fill）：fill → barrier → 复推
    fill_and_repush(ctx, etc, (views_w, views_s), epn)
    checked_slots = 0
    for g, view in enumerate(views_w):
        w_cpu = view.cpu()
        # 只比对副本槽（后 epn 行）；home 行是输入侧数据，按约定跳过
        for j in range(epn):
            e = int(etc_cpu[rank][j])
            if e < 0:
                assert bool((w_cpu[epn + j] == -1.0).all()), (
                    f"组 {g} 空预取槽 {j} 被意外写入"
                )
                continue
            home = e // epn
            exp = torch.tensor(float((home * 1000 + e) * (g + 1)), dtype=torch.bfloat16)
            assert torch.equal(w_cpu[epn + j], exp.expand(w_elems)), (
                f"组 {g} 预取槽 {j}（专家 {e}，home {home}）期望 {exp.item()}，"
                f"实得 {w_cpu[epn + j][0].item()}"
            )
            checked_slots += 1
    assert checked_slots == len(local_slots) * num_groups, (
        f"副本槽校验条数 {checked_slots} != {len(local_slots) * num_groups}"
        f"（本 rank 副本 {len(local_slots)} 条 × {num_groups} 组）"
    )

    # ---- reduce：fp32 / bf16 归并逐位对拍 ----
    base = 0.0
    case = _ReduceCase(rank, world_size, epn, w_elems, "balanced", etc_cpu)
    nontrivial = {"rows_with_copies": 0}
    # 精度前提：归并期望值必须能被 bf16 精确表示，否则"逐位相等"不成立
    # （用例规模变化时在此显式失败，而不是给出误导性的不匹配）
    for g in range(num_groups):
        for i in range(epn):
            v = _expect_merged_row(case, g, i, base)
            assert float(torch.tensor(v, dtype=torch.bfloat16)) == v, (
                f"组 {g} 行 {i} 期望值 {v} 无法被 bf16 精确表示："
                "请缩小编码基数或行数，逐位比较不成立"
            )

    def _check_reduce_strict(grads_g, dtype, tag):
        for g, dw in enumerate(grads_g):
            dw_cpu = dw.cpu()
            for i in range(epn):
                exp_val = _expect_merged_row(case, g, i, base)
                if exp_val != float((i + 1) * (g + 1) + base):
                    nontrivial["rows_with_copies"] += 1
                exp = torch.tensor(exp_val, dtype=dtype).float()
                got = dw_cpu[i].float()
                if not torch.equal(got, exp.expand(w_elems)):
                    raise AssertionError(
                        f"{tag} 组 {g} 行 {i} 归并值不符：期望 {exp.item()}，"
                        f"实得 {got[0].item()}（max|Δ|={(got - exp).abs().max().item()}）"
                    )
            for j in range(epn):
                exp_tail = torch.tensor(
                    float((epn + j + 1) * (g + 1) + base), dtype=dtype
                ).float()
                assert torch.allclose(
                    dw_cpu[epn + j].float(), exp_tail.expand(w_elems), atol=1e-2
                ), (
                    f"{tag} 组 {g} 副本行 {j} 被改写（full-dW 契约：副本段只读，"
                    f"应保持调用方写入值 {exp_tail.item()}）"
                )

    grad32_h = _full_dw_tensors(
        num_groups, epn, w_elems, base, grad_dtype=torch.float32
    )
    reduce_grad(ctx, etc, grad32_h, epn)
    torch.npu.synchronize()
    _check_reduce_strict(grad32_h, torch.float32, "fp32")
    assert nontrivial["rows_with_copies"] == len(my_experts_with_copies) * num_groups, (
        f"归并校验覆盖的副本行数 {nontrivial['rows_with_copies']} != "
        f"{len(my_experts_with_copies) * num_groups}"
        f"（本 rank {len(my_experts_with_copies)} 个专家有副本 × {num_groups} 组）"
        "：期望值计算与路由表不一致"
    )

    grad16_h = _full_dw_tensors(
        num_groups, epn, w_elems, base, grad_dtype=torch.bfloat16
    )
    reduce_grad(ctx, etc, grad16_h, epn)
    torch.npu.synchronize()
    _check_reduce_strict(grad16_h, torch.bfloat16, "bf16")
    for g in range(num_groups):
        assert torch.equal(
            grad32_h[g][:epn].float().cpu(), grad16_h[g][:epn].float().cpu()
        ), f"组 {g} fp32 与 bf16 归并结果不一致（跨 dtype 复用脏数据）"
    ctx.close()


@pytest.mark.npu
@_requires_dsl
def test_moonep_layout_formulas():
    """布局公式算术自检（纯公式，无通信）。

    v4 布局（plan 区 | 发布区(窗口池) | 尾）逐项对账——无独立梯度槽区：
    副本 dW 与预取权重同址（权重视图副本行段 [epn, 2*epn)）。
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
    # 变宽组（交织型）：(2W, W) 与等宽 3 组字节相同（2W + W = 3W）
    assert reduce_stg_off(epn, widths=(2 * w_elems, w_elems), slots=slots) == (
        reduce_stg_off(epn, w_elems, num_groups=3, slots=slots)
    ), "交织 2 组与等宽 3 组发布区终点应一致"


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
            bdp = mp._pfx_block_count(112)
            launch = max(bd, bdp, 4)  # 4 = 典型 EP 组规模（≤ 核数）
            assert nc * chunk == rows, f"{cores} 核 rows={rows} 分块不整除"
            assert launch <= cores, (
                f"{cores} 核机型上 plan 启动 {launch} 块（bd={bd}, bdp={bdp}）："
                "栅栏要求全部 block 常驻，超核会挂死"
            )
    # 核数富余时不压低启动块数（64 核仍 64 块起跑）；nc 逻辑分块受
    # _VF_SAFE_CHUNK=1024 上限约束可超核数（ebbccf6：chunk=2048 时 bisheng
    # vf 栈 6720B > 6144B 上限编译必败，压 chunk 至 1024、块内 chunk 循环
    # 动态迭代消化多出的逻辑分块）
    monkeypatch.setattr(mc, "_RESIDENT_BLOCKS", 64)
    chunk64, nc64, bd64 = mp._plan_chunking(131072)
    assert chunk64 == 1024, chunk64  # vf 栈安全 cap 生效
    assert (nc64, bd64) == (128, 64), (nc64, bd64)


@pytest.mark.npu
@_requires_dsl
@pytest.mark.full
def test_moonep_general_spec_8192_h7168_k16_epn28():
    """经典规格泛化用例：S=8192、H=7168、K=16、EPN=28（E=epn*R，4 卡即 112）。

    一次串起三个算子并各自校验：
      - plan：cu 尾值 == S*K（完美均衡不变量）；每个 token 的 vtopk 落点
      （dst_rank = vtopk // 2epn）确实承载该 token 的专家（home 或副本）；
      - prefetch：3 组权重视图的副本槽 == home 发布值，空槽保持 -1；
      - reduce_grad：3 组梯度按 home 归并正确、本 rank 预取槽清零。

    资源：权重视图 3 组 × [2epn, H×16] bf16 ≈ 38MB，reduce staging
    ≈ 51MB（HCCL 对称堆），需 HCCL_BUFFSIZE 足够（本仓跑法设 4096）。
    """
    from prefetch_weight import prefetch_weight
    from reduce_grad import reduce_grad

    rank, world_size = _init_ep_group()
    num_tokens, top_k, epn, hidden = 8192, 16, 28, 7168
    num_groups = 3
    w_elems = hidden * 16  # 每组行宽 H×H'（H'=16），三组 gate/up/down 同宽
    num_rows = num_tokens * top_k
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)

    topk_all, _ = make_router_inputs(shape, mode="balanced", seed=2026)
    topk_my = topk_all[rank].to(torch.int32).npu()
    tpe_my = (
        torch.bincount(topk_all[rank].flatten(), minlength=shape.num_experts)
        .to(torch.int32)
        .npu()
    )

    ctx = _fresh_ctx()  # 114688 宽变体：专用 ctx（组宽首调冻结）
    from moonep_context import HEAP_BASE, reduce_stg_off

    need = HEAP_BASE + reduce_stg_off(epn, w_elems, num_groups=num_groups)
    avail = ctx.workspace_bytes  # workspace 取代 HCCL 对称堆
    if need > avail:
        pytest.skip(
            f"workspace 不足：本用例需 {need / 2**20:.0f}MiB，"
            f"可用 {avail / 2**20:.0f}MiB（按形状精算传 workspace_bytes 再跑）"
        )

    _, cu, etc, _, vtopk = moonep_plan(ctx, tpe_my, topk_my)
    torch.npu.synchronize()

    # ---- plan：完美均衡 + 落点承载校验 ----
    assert int(cu[-1].item()) == num_rows, (
        f"cu 尾值 {int(cu[-1].item())} != N={num_rows}（副本槽溢出或贪心失衡）"
    )
    etc_cpu = etc.cpu()
    v_cpu = vtopk.cpu()
    for r in range(world_size):
        remote_homes = {int(e) // epn for e in etc_cpu[r].tolist() if int(e) >= 0}
        assert len(remote_homes) <= 1, (
            f"rank {r} 的副本槽来自多个 home rank：{sorted(remote_homes)}"
        )
    assert int(v_cpu.min()) >= 0 and int(v_cpu.max()) < 2 * shape.num_experts, (
        f"vtopk 越界：min={int(v_cpu.min())} max={int(v_cpu.max())} "
        f"（合法范围 [0, 2E)={2 * shape.num_experts}）"
    )
    _assert_cu_matches_vtopk(cu, vtopk, shape.num_experts)
    for i, (v, e) in enumerate(zip(v_cpu.tolist(), topk_all[rank].flatten().tolist())):
        d, slot = v // (2 * epn), v % (2 * epn)
        if slot < epn:
            assert d == e // epn and slot == e % epn, (
                f"token {i} 的 home 编码错误：expert={e}, vtopk={v}"
            )
        else:
            assert int(etc_cpu[d, slot - epn]) == e, (
                f"token {i} 的副本槽编码错误：expert={e}, vtopk={v}, "
                f"etc[{d},{slot - epn}]={int(etc_cpu[d, slot - epn])}"
            )

    # ---- prefetch：3 组权重 ----
    reals = [_fill_weight_real(rank, epn, w_elems, g) for g in range(num_groups)]
    views_w, views_s = prefetch_weight(ctx, etc, reals, epn)
    # 空槽哨兵（库内 fill 已移除）：fill → barrier → 复推
    fill_and_repush(ctx, etc, (views_w, views_s), epn)
    case = _PrefetchCase(rank, epn, w_elems)
    for g, view in enumerate(views_w):
        _verify_prefetch_view(view.cpu(), etc_cpu, case, g, f"spec 组 {g} ")

    # ---- reduce_grad：3 组梯度 ----
    grads = _full_dw_tensors(num_groups, epn, w_elems, 0.0)
    reduce_grad(ctx, etc, grads, epn)
    torch.npu.synchronize()
    _verify_reduce_views(
        grads,
        _ReduceCase(rank, world_size, epn, w_elems, "balanced", etc_cpu),
        0.0,
        "spec ",
    )
    ctx.close()


# ---------------------------------------------------------------------------
# async_finish=True：侧流并发路径——host 不被末尾屏障阻塞
# ---------------------------------------------------------------------------


def _side_stream_load(stream, iters=200):
    """在指定流上挂一段长负载（~数百 ms），用于判别 host 是否被算子阻塞。"""
    with torch.npu.stream(stream):
        a = torch.randn(2048, 2048, device="npu")
        for _ in range(iters):
            a = a @ a  # 自赋值：值会溢出到 inf，但 FLOPs 不变（纯负载）
    return a


@pytest.mark.npu
@_requires_dsl
def test_moonep_prefetch_overlap_async():
    """prefetch async_finish=True：侧流发射后 host 立即返回，事件汇合后数据正确。

    侧流先挂长负载再发射 prefetch：若末尾 barrier 仍做 host 同步，调用会
    阻塞到侧流全部完成（负载 + 直推，数百 ms）；async 路径应在负载仍在
    执行时即返回（side.query() 为 False 是 host 未阻塞的直接证据）。
    """
    from prefetch_weight import prefetch_weight

    rank, world_size = _init_ep_group()
    num_tokens, top_k, epn = 64, 8, 4
    shape = RouteShape(world_size, num_tokens, top_k, epn * world_size)
    w_elems = _H * 16
    num_groups = 3
    ctx, etc, *_ = _plan_case(rank, shape, "balanced", seed=21)

    reals = [_fill_weight_real(rank, epn, w_elems, g) for g in range(num_groups)]

    # 热身（阻塞形态，同 spec）：预热编译/HCCL/清单缓存——否则首拍 host 开销
    # （编译秒级）会比侧流负载还长，污染"host 未阻塞"判据；视图随热身创建
    views_w, views_s = prefetch_weight(ctx, etc, reals, epn)
    for view in views_w:
        view.fill_(-1.0)  # 清掉热身数据：async 调用必须重新填槽，校验才有意义
    # 空槽哨兵时序（库内 fill 已移除——bcdf4d8）：全组 fill 先于 async
    # push（fill 在本 rank 默认流、push 在侧流，跨 rank 须显式对齐）
    from moonep_context import sym_barrier

    sym_barrier(ctx)

    side = torch.npu.Stream()
    _side_stream_load(side)
    with torch.npu.stream(side):
        _views_a, ev = prefetch_weight(ctx, etc, reals, epn, async_finish=True)
    assert ev is not None, "async_finish=True 应返回发起流完成事件"
    still_busy = not side.query()
    assert still_busy, (
        "prefetch 返回时侧流前置负载应仍在执行——host 被末尾屏障阻塞的证据"
        "（async 路径不应等待直推完成）"
    )
    # 主流汇合（本 rank 发射完成）+ 跨 rank 对齐后再校验：本地事件只保证
    # 本 rank 的 push 已发出，副本槽是对端写入的——消费对端数据前须全组
    # 对齐（async_finish 调用方契约；生产形态由下一轮迭代 barrier 承担）。
    # 顺序：先 wait_event（主流排在侧流 push 之后），再 sym_barrier（flag 与
    # 数据同通道保序，flag 到达 ⟹ 对端发来的数据已落地）
    torch.npu.current_stream().wait_event(ev)
    from moonep_context import sym_barrier

    sym_barrier(ctx)
    torch.npu.synchronize()
    case = _PrefetchCase(rank, epn, w_elems)
    etc_cpu = etc.cpu()
    for g, view in enumerate(views_w):
        _verify_prefetch_view(view.cpu(), etc_cpu, case, g, f"async 组 {g} ")
    exp_home = (
        torch.tensor(
            [float(rank * 1000 + rank * epn + i) for i in range(epn)],
            dtype=torch.bfloat16,
        )
        .unsqueeze(1)
        .expand(epn, w_elems)
    )
    assert torch.equal(views_w[0].cpu()[:epn], exp_home), "async home 行未写入"


# reduce_grad 的 async_finish kwarg 极性与 finish_op 对齐
# （True=异步 Event / False=阻塞），侧流并发语义由发射序
# 纪律承担（sync_transport_stream + sym_barrier）。
