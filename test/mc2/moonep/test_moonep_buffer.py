# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MoonepBuffer 用例矩阵：基础功能 + EB dispatch/combine 增量（设计 §6.3）。

运行（4 卡）：
    torchrun --nproc_per_node=4 -m pytest test/moonep/test_moonep_buffer.py -q
    （或脚本形态：torchrun --nproc_per_node=4 test/moonep/test_moonep_buffer.py）
单进程 pytest 收集自动跳过（CI 安全——双守卫：DSL/EB import 失败或非 torchrun
拉起均 skip，与 test_moonep_mem.py / test_moonep_ep.py 同款思想）。

方法学：
- 数值对拍用编码值法（行值编码 home*1000+e、期望合并值公式、空槽 -1 保持，
  test_moonep_ep.py 同源）+ golden 对拍（moonep_golden 的 dispatch/combine，
  B1/B2/B4 的 EB 链路）；数据流按 PR #261 后的 pull 协议建模。
- 副本槽契约（已知事实）：pull 协议下 owner 对槽位只读远拉、不清零——断言
  "副本槽保持写入值"而非"清零"。
- ⚠️ EB 在场禁用 dist.all_reduce（EI0006：EB URMA 与集合通信混用的已知风险，
  mem 用例 --verify 注记同源）：A2 的 cu/vtopk 跨 rank 对拍改为本地全表落点
  校验（etc 为 [R, epn] 全表，vtopk→(dst, slot) 逐 token 核对），端到端路由
  正确性由 B1/B4 的 golden 对拍兜底。

组织（对齐 test_moonep_ep._EP_POOL 思想 + EB 平台行为约束）：
- _BUFFER_POOL 按 (S,K,E,W,groups,slots,grad_dtype,hidden) 键控共享 buffer
  （构造为集体调用，避免每用例建子组/堆累积）；
- 专用 buffer（destroy 类）入 _DEDICATED 保活列表——EB 以
  explicitly_destroy=False 构造，GC 析构会触发 C++ Destroy 的 AICPU_TS 等待
  （conftest 实证 ~120s/个）；引用保活到进程退出 + conftest 硬退跳过 GC。
- 全部 EB@WORLD：moe 通信窗按 (group, tag) 进程级共享池复用、容量取首建值
  （EB csrc 语义），故本文件全部 dispatch 形状一致（同 S/K/E/hidden/nmtr），
  A11 的大形状 buffer 只构造不 dispatch（不触窗注册）。
"""

import os
import subprocess
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

from moonep_golden import (  # noqa: E402
    RouteShape,
    make_router_inputs,
    moonep_combine_golden,
    moonep_dispatch_golden,
)

from conftest import fill_empty_slots  # noqa: E402

try:
    import torch_npu  # noqa: F401
    from moonep_buffer import MoonepBuffer  # noqa: F401
    from moonep_context import (  # noqa: F401
        HEAP_BASE,
        LAYOUT_TAIL_BYTES,
        reduce_region_end,
    )

    _HAS_DSL_IMPL = True
except Exception:  # CI 无 NPU / 无 DSL / 无 EB whl 时收集不报错（skip 而非 error）
    _HAS_DSL_IMPL = False

try:
    from cann_ops_transformer.ops import ElasticBuffer  # noqa: F401,E402

    _HAS_EP = True
except ImportError:
    _HAS_EP = False

_requires_train_env = pytest.mark.skipif(
    not _HAS_DSL_IMPL
    or not _HAS_EP
    or "RANK" not in os.environ
    or "WORLD_SIZE" not in os.environ
    or os.environ["WORLD_SIZE"] == "1",
    reason="需 torch.distributed.run 多卡拉起且已装 cann_ops_transformer"
    "（单进程 pytest 收集形态跳过）",
)

# 主门禁形状：W = _H*_Hp = 512（W%64 与 chunk 可分性满足）；E = epn*R 运行时定；
# _H 为 EB hidden（dispatch/combine 载荷宽，与 W 相互独立）
_S, _K, _EPN = 32, 8, 4
_H, _Hp = 32, 16
_W = _H * _Hp
_NUM_GROUPS = 3
_ATOL = _RTOL = 2e-2
_SAMPLES_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "samples", "mc2", "moonep"
    )
)


# ---------------------------------------------------------------------------
# 环境与池化
# ---------------------------------------------------------------------------


def _init_group():
    """torchrun 拉起时初始化 HCCL 组；单进程直跑则 skip（ep 用例同款）。"""
    if not dist.is_initialized():
        if "MASTER_ADDR" not in os.environ or "RANK" not in os.environ:
            pytest.skip("需 torchrun --nproc_per_node=N（N>=2）拉起")
        dist.init_process_group(backend="hccl")
    world = dist.get_world_size()
    if world < 2:
        pytest.skip("MoonepBuffer 用例需 >= 2 卡")
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    return rank, world


_BUFFER_POOL = {}  # (S, K, E, W, groups, slots, grad_dtype, hidden) 键控共享


_DEDICATED = []  # 专用 buffer 保活（防 GC 中途触发 EB 析构——conftest 120s 经验）


def _fill_and_repush(buf, plan, views):
    """空槽哨兵协议（库内不做 fill——测试侧负责）：
    fill 副本段 → 跨 rank barrier → home 行切片复推。

    复推重写命中槽、空槽保持哨兵；home 行切片 data_ptr 同址 ⇒ sync
    零拷贝跳过。scale 组同规则随行（未量化 scales=() 时 local_scales=None）。
    barrier 必要性：复推写对端副本槽，须保证对端 fill 已完成。
    """
    fill_empty_slots(buf.epn, views)
    buf.barrier()
    weights, scales = views
    buf.prefetch_weight(
        plan,
        local_weights=[v[: buf.epn] for v in weights],
        local_scales=[s[: buf.epn] for s in scales] if scales else None,
    )
    torch.npu.synchronize()


def _get_buffer(moonep_groups, slots=1, grad_dtype="bf16"):
    """构造（或复用）池化 MoonepBuffer——全 rank 同序命中同键。"""
    world = dist.get_world_size()
    E = _EPN * world
    key = (_S, _K, E, _W, _NUM_GROUPS, slots, str(grad_dtype), _H)
    if key not in _BUFFER_POOL:
        moonep_group, ep_group = moonep_groups
        _BUFFER_POOL[key] = MoonepBuffer(
            _S,
            _H,
            _Hp,
            _K,
            E,
            moonep_group=moonep_group,
            ep_group=ep_group,
            slots=slots,
            grad_dtype=grad_dtype,
        )
    return _BUFFER_POOL[key]


def _dedicated_buffer(moonep_groups, **kwargs):
    """专用小 buffer（destroy 类用例）——入保活列表防 GC 析构。

    形状缺省主门禁档；E/expert 两维等均可经 kwargs 覆盖（后写胜出）。
    """
    world = dist.get_world_size()
    moonep_group, ep_group = moonep_groups
    args = dict(
        E=_EPN * world,
        Hp=_Hp,
    )
    args.update(kwargs)
    buf = MoonepBuffer(
        _S,
        _H,
        args.pop("Hp"),
        _K,
        args.pop("E"),
        moonep_group=moonep_group,
        ep_group=ep_group,
        **args,
    )
    _DEDICATED.append(buf)
    return buf


def _local_route(topk_all, rank, num_experts):
    """golden 路由全表 → 本 rank 的 (topk, tpe) int32 NPU 张量。"""
    topk = topk_all[rank].to(torch.int32).npu()
    tpe = (
        torch.bincount(topk_all[rank].flatten(), minlength=num_experts)
        .to(torch.int32)
        .npu()
    )
    return topk, tpe


def _routing(seed):
    """种子路由 → 本 rank 的 (topk, tpe)（A 系用例入口）。"""
    world = dist.get_world_size()
    shape = RouteShape(world, _S, _K, _EPN * world)
    topk_all, _ = make_router_inputs(shape, mode="balanced", seed=seed)
    return _local_route(topk_all, dist.get_rank(), shape.num_experts)


def _make_case(shape, seed):
    """B 系用例输入：golden 路由全表 + 独立种子 hidden 全表（ep 用例同款）。"""
    topk_all, rw_all = make_router_inputs(shape, mode="balanced", seed=seed)
    gen = torch.Generator().manual_seed(1234)
    hidden_all = torch.randn(
        shape.world_size, shape.num_tokens, _H, generator=gen, dtype=torch.float32
    ).to(torch.bfloat16)
    return topk_all, rw_all, hidden_all


# ---------------------------------------------------------------------------
# 编码值助手（test_moonep_ep.py 方法学的小副本，随 pull 协议适配）
# ---------------------------------------------------------------------------


class _PrefetchCase(NamedTuple):
    """prefetch 校验上下文：rank / epn / 行宽。"""

    rank: int
    epn: int
    w_elems: int


def _assert_one_source_etc(etc_cpu, epn):
    """plan 契约：每个 destination rank 的副本槽至多来自一个 home rank。"""
    for row in etc_cpu.tolist():
        homes = {int(e) // epn for e in row if int(e) >= 0}
        assert len(homes) <= 1, f"etc 行 {row} 同时来自多个 home rank：{sorted(homes)}"


def _check_vtopk_placement_local(vtopk, topk, etc_cpu, epn):
    """vtopk 落点本地全表校验（替代 all_reduce 对拍——EB 在场规避 EI0006）。

    etc 为 [R, epn] 全表（plan 的 Exchange 阶段交换所得，各 rank 一致），
    逐 token 核对 vtopk→(dst_rank, slot) 与路由表/副本清单的对应：
    home 槽（slot < epn）须指向专家的 home rank；副本槽须命中 etc[d] 的清单。
    """
    v_cpu = vtopk.cpu().reshape(-1).tolist()
    t_cpu = topk.cpu().reshape(-1).tolist()
    assert len(v_cpu) == len(t_cpu)
    for i, (v, e) in enumerate(zip(v_cpu, t_cpu)):
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


def _fill_weight_real(rank, epn, w_elems, g=0, scale=1):
    """权重编码值张量：行 i = (rank*1000 + rank*epn + i) * (g+1) * scale。"""
    w_real = torch.empty(epn, w_elems, dtype=torch.bfloat16)
    for i in range(epn):
        w_real[i] = float((rank * 1000 + rank * epn + i) * (g + 1) * scale)
    return w_real.npu()


def _exp_home_rows(rank, epn, w_elems, g=0, scale=1):
    """home 行期望（独立按编码公式计算，不引用填充张量本身）。"""
    vals = [float((rank * 1000 + rank * epn + i) * (g + 1) * scale) for i in range(epn)]
    return (
        torch.tensor(vals, dtype=torch.bfloat16)
        .unsqueeze(1)
        .expand(epn, w_elems)
        .contiguous()
    )


def _verify_prefetch_view(w_cpu, etc_cpu, case, g=0, scale=1, tag=""):
    """副本槽 == home 发布值 (home*1000+e)*(g+1)*scale；空槽保持 -1。"""
    _assert_one_source_etc(etc_cpu, case.epn)
    rank, epn, w_elems = case.rank, case.epn, case.w_elems
    for j in range(epn):
        e = int(etc_cpu[rank][j])
        if e < 0:
            assert bool((w_cpu[epn + j] == -1.0).all()), f"{tag}空预取槽 {j} 被意外写入"
            continue
        home = e // epn
        exp_val = float((home * 1000 + e) * (g + 1) * scale)
        got = w_cpu[epn + j].float()
        assert torch.allclose(
            got, torch.full((w_elems,), exp_val), atol=1e-2, rtol=1e-2
        ), (
            f"{tag}预取槽 {j}（专家 {e}，home {home}）期望 {exp_val}，实得 {got[0].item():.1f}"
        )


def _full_dw_grads(epn, d0, d1, g, base, dtype):
    """full-dW 张量 [2*epn, d0, d1]：home 行 i = (i+1)*(g+1)+base；
    副本行 epn+j = (epn+j+1)*(g+1)+base（行序与 prefetch 视图一致）。"""
    t = torch.empty(2 * epn, d0, d1, dtype=dtype, device="npu")
    for i in range(2 * epn):
        t[i] = float((i + 1) * (g + 1) + base)
    return t


def _expect_merged_row(rank, world_size, epn, etc_cpu, g, row, base):
    """单组归并期望：home 种子 + Σ 各 rank 副本行贡献（编码同 _full_dw_grads 尾段）。"""
    e = rank * epn + row
    exp_val = float((row + 1) * (g + 1) + base)
    for r in range(world_size):
        if r == rank:
            continue
        for j in range(epn):
            if int(etc_cpu[r][j]) == e:
                exp_val += float((epn + j + 1) * (g + 1) + base)
    return exp_val


def _verify_merged(t, rank, world_size, epn, etc_cpu, g, base, w_elems, tag=""):
    """reduce 后逐行校验 home 段归并值（编码基数保证 bf16 可精确表示）。"""
    m = t[:epn].reshape(epn, w_elems).cpu().float()
    for i in range(epn):
        exp_val = _expect_merged_row(rank, world_size, epn, etc_cpu, g, i, base)
        assert torch.allclose(
            m[i], torch.full((w_elems,), exp_val), atol=1e-3, rtol=1e-3
        ), f"{tag}行 {i} 归并值期望 {exp_val}，实得 {m[i][0].item():.1f}"


def _verify_replica_intact(grads, epn, base, tag=""):
    """full-dW 尾段保持校验：reduce 只读副本行（入槽后归并远端贡献进 home 段）。

    已知事实（当前实现实证）：pull 协议对输入副本行不清零（对称堆槽是
    独立落点，非输入张量本体），断言实际契约而非历史语义。
    """
    for g, t in enumerate(grads):
        tail = t[epn:].reshape(epn, -1).cpu().float()
        for j in range(epn):
            exp_val = float((epn + j + 1) * (g + 1) + base)
            assert torch.allclose(
                tail[j], torch.full((tail.shape[1],), exp_val), atol=1e-3, rtol=1e-3
            ), f"{tag}组 {g} 副本行 {j} 被 reduce 破坏（应只读）"


# ---------------------------------------------------------------------------
# A2 plan 元组兼容 + 落点
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_train_env
def test_a2_plan_tuple_compat_and_guards(moonep_groups):
    """A2：五件套解包 + 属性访问双形态；cu 尾值均衡不变量；vtopk 落点承载
    （本地全表校验——EB 在场禁 all_reduce，端到端由 B1/B4 golden 兜底）；
    异形 tpe/topk 在进入 kernel 前被构造期一致性守卫拒绝。"""
    _init_group()
    buf = _get_buffer(moonep_groups)
    topk, tpe = _routing(42)
    plan = buf.plan(tpe, topk)
    # 元组解包兼容（NamedTuple 即 tuple）+ 属性访问双形态
    dst_n, cu, etc, src_n, vtopk = plan
    assert plan.dst_n is dst_n and plan.src_n is src_n
    assert plan.cu_seqlens is cu and plan.experts_to_copy is etc
    assert plan.vtopk is vtopk
    torch.npu.synchronize()
    # 均衡不变量：cu 尾值 == S*K（plan 返回路径零同步，此处测试侧同步后取值）
    assert int(cu[-1].item()) == _S * _K
    etc_cpu = etc.cpu()
    _assert_one_source_etc(etc_cpu, buf.epn)
    # vtopk 落点承载：值域 [0, 2E) + 本地全表逐 token 核对
    E = buf.epn * buf.rank_size
    v_cpu = vtopk.cpu()
    assert int(v_cpu.min()) >= 0 and int(v_cpu.max()) < 2 * E
    _check_vtopk_placement_local(vtopk, topk, etc_cpu, buf.epn)
    # 覆盖面护栏：balanced 路由须产生副本（否则副本路径未被覆盖）
    assert int((etc_cpu >= 0).sum()) > 0, "balanced 路由未产生副本槽：副本路径未被覆盖"
    # 构造期一致性守卫：异形入参拒绝（先于 kernel 发射，无集合调用）
    with pytest.raises(ValueError):
        buf.plan(tpe[: E - 1].clone(), topk)
    with pytest.raises(ValueError):
        buf.plan(tpe, topk[: _S - 1])


# ---------------------------------------------------------------------------
# A3/A4 prefetch：零拷贝端到端 / 显式 real
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_train_env
def test_a3_prefetch_zero_copy_e2e(moonep_groups):
    """A3：参数直接初始化进视图 home 行 → prefetch(real=None) →
    副本槽 == home 编码值、空槽保持 -1、home 行等于参数（零拷贝路径）。"""
    rank, _world = _init_group()
    buf = _get_buffer(moonep_groups)
    topk, tpe = _routing(42)
    plan = buf.plan(tpe, topk)
    etc_cpu = plan.experts_to_copy.cpu()
    assert int((etc_cpu >= 0).sum()) > 0, "balanced 路由未产生副本槽：副本路径未被覆盖"
    # 首调显式传 real：视图按各组行宽创建并缓存（A4 路径），返回供零拷贝复用
    reals = [
        _fill_weight_real(rank, buf.epn, buf.H * buf.Hp, g) for g in range(_NUM_GROUPS)
    ]
    views_w, views_s = buf.prefetch_weight(plan, local_weights=reals)
    # 空槽哨兵（库内 fill 已移除）：fill → barrier → 复推后空槽保持 -1
    _fill_and_repush(buf, plan, (views_w, views_s))
    # 零拷贝轮：参数更新直接写视图 home 行（data_ptr 与 real 切片同址），
    # prefetch(plan) 的 real=None 取 home 行切片——sync_transport 同址跳过
    for g, v in enumerate(views_w):
        v[: buf.epn].copy_(_fill_weight_real(rank, buf.epn, buf.H * buf.Hp, g))
    # C7+C8 契约：local_weights 必选——零拷贝语义 = 传视图 home 行切片本体
    # （data_ptr 与 home 段同址 ⇒ sync_transport 同址跳过，自动零拷贝）
    home_rows = [v[: buf.epn] for v in views_w]
    buf.prefetch_weight(plan, local_weights=home_rows)
    torch.npu.synchronize()
    case = _PrefetchCase(rank, buf.epn, buf.H * buf.Hp)
    for g, v in enumerate(views_w):
        _verify_prefetch_view(v.cpu(), etc_cpu, case, g, tag=f"组 {g} ")
        exp_home = _exp_home_rows(rank, buf.epn, buf.H * buf.Hp, g)
        assert torch.equal(v.cpu()[: buf.epn], exp_home), f"组 {g} home 行未被保留"


@pytest.mark.npu
@_requires_train_env
def test_a4_prefetch_explicit_real(moonep_groups):
    """A4：独立 real_weights 张量路径——home 行被 sync_transport 写入 real，
    副本槽对拍（拷贝路径）。"""
    rank, _world = _init_group()
    buf = _get_buffer(moonep_groups)
    topk, tpe = _routing(42)
    plan = buf.plan(tpe, topk)
    etc_cpu = plan.experts_to_copy.cpu()
    reals = [
        _fill_weight_real(rank, buf.epn, buf.H * buf.Hp, g) for g in range(_NUM_GROUPS)
    ]
    views_w, views_s = buf.prefetch_weight(
        plan, local_weights=reals
    )  # home 行被写入 real
    # 空槽哨兵（库内 fill 已移除）：fill → barrier → 复推后空槽保持 -1
    _fill_and_repush(buf, plan, (views_w, views_s))
    case = _PrefetchCase(rank, buf.epn, buf.H * buf.Hp)
    for g, v in enumerate(views_w):
        _verify_prefetch_view(v.cpu(), etc_cpu, case, g, tag=f"显式real 组 {g} ")
        exp_home = _exp_home_rows(rank, buf.epn, buf.H * buf.Hp, g)
        assert torch.equal(v.cpu()[: buf.epn], exp_home), f"组 {g} home 行未被写入 real"


# ---------------------------------------------------------------------------
# A5 reduce 编码值对拍（pull 协议）
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_train_env
def test_a5_reduce_encoded_multigroup(moonep_groups):
    """A5：3 组 home 种子 + 副本槽编码 dW → reduce → 逐组逐行归并对拍；
    [epn,F,D]/[epn,D,F] 混合形态验证内部 view(epn, W)；副本槽保持（pull 契约）。

    编码基数保证合并值 < 256（bf16 精确表示域），atol/rtol 1e-3 下逐位成立。
    """
    rank, world = _init_group()
    buf = _get_buffer(moonep_groups)
    topk, tpe = _routing(42)
    plan = buf.plan(tpe, topk)
    etc_cpu = plan.experts_to_copy.cpu()
    assert int((etc_cpu >= 0).sum()) > 0, "未产生副本槽：reduce 副本路径未被覆盖"
    base = 0.0
    # full-dW：g1/g3 用 [2*epn,F,D]、g2 用 [2*epn,D,F]（mem run() 同款形状对；
    # 前 epn 行 home 种子、后 epn 行副本 dW——副本入槽由 reduce 内部完成）
    g1 = _full_dw_grads(buf.epn, _Hp, _H, 0, base, buf.grad_dtype)
    g3 = _full_dw_grads(buf.epn, _Hp, _H, 1, base, buf.grad_dtype)
    g2 = _full_dw_grads(buf.epn, _H, _Hp, 2, base, buf.grad_dtype)
    buf.reduce_grad(
        plan, local_grads=[g1, g3, g2]
    )  # 门/上/下组序（组数从张量解析；dtype 经 E5 校验）
    torch.npu.synchronize()
    for gi, t in enumerate((g1, g3, g2)):
        _verify_merged(
            t,
            rank,
            world,
            buf.epn,
            etc_cpu,
            gi,
            base,
            buf.H * buf.Hp,
            tag=f"组 {gi} ",
        )
    _verify_replica_intact((g1, g3, g2), buf.epn, base)


# ---------------------------------------------------------------------------
# A6 视图零拷贝 / 布局
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_train_env
def test_a6_view_zero_copy_and_layout(moonep_groups):
    """A6：weight_view 缓存访问器幂等（同 data_ptr）；
    组间按整视图步长推进；slots=2 槽间不重叠；group/slot 越界守卫。"""
    rank, _world = _init_group()
    buf = _get_buffer(moonep_groups)
    topk, tpe = _routing(42)
    plan = buf.plan(tpe, topk)
    reals = [
        _fill_weight_real(rank, buf.epn, buf.H * buf.Hp, g) for g in range(_NUM_GROUPS)
    ]
    views_w, views_s = buf.prefetch_weight(plan, local_weights=reals)
    # 缓存访问器幂等（同 data_ptr）+ 视图数 = 组数（运行期定型）
    assert buf.weight_view(0).data_ptr() == buf.weight_view(0).data_ptr()
    assert len(views_w) == _NUM_GROUPS and not views_s
    # 组间步长：同槽相邻组视图按整视图步长推进（组间不重叠）
    va, vb = views_w[0], views_w[1]
    view_bytes = va.numel() * va.element_size()
    assert vb.data_ptr() - va.data_ptr() == view_bytes
    # slots=2 构造档：槽 s 视图基址 = 发布区 + s*组数*整视图步长
    buf2 = _get_buffer(moonep_groups, slots=2)
    topk2, tpe2 = _routing(42)
    plan2 = buf2.plan(tpe2, topk2)
    reals2 = [
        _fill_weight_real(rank, buf2.epn, buf2.H * buf2.Hp, g)
        for g in range(_NUM_GROUPS)
    ]
    # sync 返回 (weights, scales) 二元组——取 slot0 的组 0 权重视图
    v_slot0 = buf2.prefetch_weight(plan2, local_weights=reals2, slot=0)
    v_slot1 = buf2.prefetch_weight(plan2, local_weights=reals2, slot=1)
    s0, s1 = v_slot0[0][0], v_slot1[0][0]
    assert buf2.slots == 2
    assert (
        s1.data_ptr() - s0.data_ptr() == _NUM_GROUPS * view_bytes
        and (s1.data_ptr() - s0.data_ptr()) % view_bytes == 0
    ), "双槽视图槽间重叠或步长不符"
    # 越界守卫（组/槽）
    with pytest.raises(ValueError):
        buf.weight_view(_NUM_GROUPS)
    with pytest.raises(ValueError):
        buf.weight_view(0, slot=buf.slots)
    # 未 prefetch 的槽位访问 → 引导性报错（视图随首调创建）：buf2 双槽
    # 两槽均已预取，改用专用 buffer 只预取 slot 0
    buf3 = MoonepBuffer(
        _S,
        _H,
        _Hp,
        _K,
        _EPN * _world,
        moonep_group=moonep_groups[0],
        ep_group=moonep_groups[1],
        slots=2,
    )
    _DEDICATED.append(buf3)
    topk3, tpe3 = _routing(42)
    plan3 = buf3.plan(tpe3, topk3)
    reals3 = [
        _fill_weight_real(rank, buf3.epn, buf3.H * buf3.Hp, g)
        for g in range(_NUM_GROUPS)
    ]
    buf3.prefetch_weight(plan3, local_weights=reals3, slot=0)
    with pytest.raises(RuntimeError):
        buf3.weight_view(0, slot=1)


# ---------------------------------------------------------------------------
# A7 destroy 幂等
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_train_env
def test_a7_destroy_idempotent_and_reject(moonep_groups):
    """A7：destroy() 幂等；destroyed 置位；后续 plan/视图/prefetch/reduce/
    dispatch/combine/barrier 均 RuntimeError；纯值 property 仍可读。"""
    rank, _world = _init_group()
    # 专用小 buffer（不入池——destroy 不殃及后续用例；入 _DEDICATED 保活防 GC）
    buf = _dedicated_buffer(moonep_groups)
    topk, tpe = _routing(42)
    plan = buf.plan(tpe, topk)
    assert not buf.destroyed
    buf.destroy()
    buf.destroy()  # 幂等：二次不抛
    assert buf.destroyed
    with pytest.raises(RuntimeError):
        buf.plan(tpe, topk)
    with pytest.raises(RuntimeError):
        buf.weight_view(0)
    with pytest.raises((RuntimeError, TypeError)):
        # C7+C8: local_weights 必选——None 时 TypeError（先于 destroy 检查）
        buf.prefetch_weight(plan)
    with pytest.raises(RuntimeError):
        buf.reduce_grad(
            plan,
            local_grads=[torch.zeros(2 * _EPN, _W, dtype=torch.bfloat16)] * _NUM_GROUPS,
        )
    # dispatch/combine 同经 _require_alive 拒绝（EB 虽保活，逻辑资源已关）
    x = torch.zeros(_S, _H, dtype=torch.bfloat16, device="npu")
    with pytest.raises(RuntimeError):
        buf.dispatch(plan, x)
    with pytest.raises(RuntimeError):
        buf.combine(x, None)
    with pytest.raises(RuntimeError):
        buf.barrier()
    with pytest.raises(RuntimeError):
        _ = buf.heap_bytes
    # 纯值 property 在 destroy 后仍可读（构造期定型量）
    assert buf.epn == _EPN and buf.H * buf.Hp == _W
    assert buf.slots == 1
    assert buf.heap_required_bytes > 0 and buf.heap_config_mb >= 2
    assert buf.grad_dtype == torch.bfloat16 and buf.rank == rank
    assert buf.H == _H and buf.nmtr >= _S


# ---------------------------------------------------------------------------
# A8 连续两轮 epoch 复入
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_train_env
def test_a8_two_round_reentry(moonep_groups):
    """A8：同一 buffer 两轮 plan→prefetch→reduce（不同 seed 路由与编码基数）——
    第二轮数据正确、无残留 flag 提前放行（epoch 单调）、staging/scratch 复用不互踩。"""
    rank, world = _init_group()
    buf = _get_buffer(moonep_groups)
    for rnd, (seed, base, scale) in enumerate(((101, 0.0, 1), (202, 8.0, 2))):
        topk, tpe = _routing(seed)
        plan = buf.plan(tpe, topk)
        etc_cpu = plan.experts_to_copy.cpu()
        # —— prefetch 轮（编码按轮倍率区分——上一轮残留数据必被检出；
        #     首轮显式 real，次轮 real=None 零拷贝——两种入参形态都覆盖）——
        if rnd == 0:
            reals = [
                _fill_weight_real(
                    rank,
                    buf.epn,
                    buf.H * buf.Hp,
                    g,
                    scale,
                )
                for g in range(_NUM_GROUPS)
            ]
            views_w, views_s = buf.prefetch_weight(plan, local_weights=reals)
            # 空槽哨兵（库内 fill 已移除）：fill → barrier → 复推
            _fill_and_repush(buf, plan, (views_w, views_s))
        else:
            # 零拷贝轮：先重置空槽语义（清上轮残留——旧版每轮 fill(-1)
            # 等价；etc 变化后旧副本槽数据会被误判），再写 home 行（参数
            # 更新），最后直推——顺序不可颠倒（先推后写会推送旧值）
            for g, v in enumerate(views_w):
                v.fill_(-1.0)
                v[: buf.epn].copy_(
                    _fill_weight_real(
                        rank,
                        buf.epn,
                        buf.H * buf.Hp,
                        g,
                        scale,
                    )
                )
            # 多轮空槽重置由测试侧显式
            # 控制。barrier 必须在 fill 之后、prefetch 之前：对端 push 写入
            # 本 rank 副本槽若先于本 rank fill 会把推送数据抹掉。
            buf.barrier()
            # C7+C8 契约：零拷贝轮传视图 home 行切片本体（data_ptr 同址
            # ⇒ sync 跳过——若传陈旧 reals 会把旧值覆写进已更新的 home 行）
            home_rows = [v[: buf.epn] for v in views_w]
            views_w, views_s = buf.prefetch_weight(plan, local_weights=home_rows)
        # —— reduce 轮（编码基数随轮变化；full-dW 直入）——
        g1 = _full_dw_grads(buf.epn, _Hp, _H, 0, base, buf.grad_dtype)
        g3 = _full_dw_grads(buf.epn, _Hp, _H, 1, base, buf.grad_dtype)
        g2 = _full_dw_grads(buf.epn, _H, _Hp, 2, base, buf.grad_dtype)
        buf.reduce_grad(plan, local_grads=[g1, g3, g2])
        torch.npu.synchronize()
        # —— 双算子校验 ——
        case = _PrefetchCase(rank, buf.epn, buf.H * buf.Hp)
        for g, v in enumerate(views_w):
            _verify_prefetch_view(
                v.cpu(), etc_cpu, case, g, scale, tag=f"第{rnd + 1}轮 "
            )
        for gi, t in enumerate((g1, g3, g2)):
            _verify_merged(
                t,
                rank,
                world,
                buf.epn,
                etc_cpu,
                gi,
                base,
                buf.H * buf.Hp,
                tag=f"第{rnd + 1}轮 组{gi} ",
            )


# ---------------------------------------------------------------------------
# A9 async_finish 透传
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_train_env
def test_a9_async_finish_events(moonep_groups):
    """A9：prefetch/reduce async_finish=True 返回 Event 非 None；wait_event +
    barrier 汇合后数据正确（槽写入是对端行为——跨 rank 对齐后校验；
    不做侧流时序断言，透传语义由既有 overlap 用例保底）。"""
    rank, world = _init_group()
    buf = _get_buffer(moonep_groups)
    topk, tpe = _routing(42)
    plan = buf.plan(tpe, topk)
    etc_cpu = plan.experts_to_copy.cpu()
    # —— prefetch async（首调显式 real；返回 (views, event)）——
    reals = [
        _fill_weight_real(rank, buf.epn, buf.H * buf.Hp, g) for g in range(_NUM_GROUPS)
    ]
    (views_w, views_s), ev = buf.prefetch_weight(
        plan, async_finish=True, local_weights=reals
    )
    assert ev is not None and isinstance(ev, torch.npu.Event)
    torch.npu.current_stream().wait_event(ev)
    buf.barrier()  # async 调用方契约：跨 rank 对齐后再校验副本槽
    torch.npu.synchronize()
    # 空槽哨兵（库内 fill 已移除）：fill → barrier → 同步复推（async 契约
    # 已由首次调用覆盖，复推走 sync 路径不影响语义）
    _fill_and_repush(buf, plan, (views_w, views_s))
    case = _PrefetchCase(rank, buf.epn, buf.H * buf.Hp)
    for g, v in enumerate(views_w):
        _verify_prefetch_view(v.cpu(), etc_cpu, case, g, tag=f"async 组 {g} ")
    # —— reduce async（full-dW 直入）——
    base = 0.0
    g1 = _full_dw_grads(buf.epn, _Hp, _H, 0, base, buf.grad_dtype)
    g3 = _full_dw_grads(buf.epn, _Hp, _H, 1, base, buf.grad_dtype)
    g2 = _full_dw_grads(buf.epn, _H, _Hp, 2, base, buf.grad_dtype)
    _hv, ev2 = buf.reduce_grad(plan, async_finish=True, local_grads=[g1, g3, g2])
    assert ev2 is not None and isinstance(ev2, torch.npu.Event)
    torch.npu.current_stream().wait_event(ev2)
    buf.barrier()
    torch.npu.synchronize()
    for gi, t in enumerate((g1, g3, g2)):
        _verify_merged(
            t,
            rank,
            world,
            buf.epn,
            etc_cpu,
            gi,
            base,
            buf.H * buf.Hp,
            tag=f"async 组 {gi} ",
        )


# ---------------------------------------------------------------------------
# A12 交织 2 组形态（3 矩阵 → 2 组，核心新特性）
# ---------------------------------------------------------------------------


def _full_dw_2d(epn, w, g, base, dtype):
    """full-dW 二维张量 [2*epn, w]：行 i = (i+1)*(g+1)+base（A12 专用）。"""
    t = torch.empty(2 * epn, w, dtype=dtype, device="npu")
    for i in range(2 * epn):
        t[i] = float((i + 1) * (g + 1) + base)
    return t


@pytest.mark.npu
@_requires_train_env
def test_a12_intertwined_2group(moonep_groups):
    """A12：交织 2 组——用户把 3 个权重矩阵合并成 2 个传入。

    ``W13 = cat(gate, up, dim=1)``（每专家一行 2W，行内 gate 前 up 后，
    一次 GMM 算完免消费端 concat）+ down（W）：prefetch/reduce 的组数
    （2 vs 3）与各组行宽全部从张量解析，堆布局 (2W, W) 总字节与 3 组
    等宽（2W+W=3W）一致。prefetch 返回 (view13, view_down)；reduce 收
    [2*epn,2W]+[2*epn,W] full-dW，归并结果与 3 组公式逐组同律对拍。
    传输对 W13 行内容不感知（不透明搬运——行内 gate/up 语义属调用方）。
    """
    rank, world = _init_group()
    # 专用 buffer：2 组结构定型 ctx.layout_groups（与池化 3 组互斥），
    # 入 _DEDICATED 保活防 GC 中途触发 EB 析构
    buf = MoonepBuffer(
        _S,
        _H,
        _Hp,
        _K,
        _EPN * world,
        moonep_group=moonep_groups[0],
        ep_group=moonep_groups[1],
        slots=1,
    )
    _DEDICATED.append(buf)
    topk, tpe = _routing(42)
    plan = buf.plan(tpe, topk)
    etc_cpu = plan.experts_to_copy.cpu()
    assert int((etc_cpu >= 0).sum()) > 0, "balanced 路由未产生副本槽"
    W = buf.H * buf.Hp

    # —— prefetch 2 组交织：组数/行宽从张量解析（(2W, W)）——
    w13 = _fill_weight_real(rank, buf.epn, 2 * W, 0)  # [epn, 2W] 交织行
    w2 = _fill_weight_real(rank, buf.epn, W, 1)  # [epn, W]
    views_w, views_s = buf.prefetch_weight(plan, local_weights=(w13, w2))
    torch.npu.synchronize()
    assert len(views_w) == 2 and not views_s, (
        f"交织 2 组 bf16 应返回 2 权重视图、无 scale，got {len(views_w)}/{len(views_s)}"
    )
    assert views_w[0].shape == (2 * buf.epn, 2 * W), f"view13 形状 {views_w[0].shape}"
    assert views_w[1].shape == (2 * buf.epn, W), f"view_down 形状 {views_w[1].shape}"
    # 视图按槽区前缀和紧密排布：view_down 紧随 view13（2*epn*2W*2 字节后）
    stride13 = views_w[0].numel() * views_w[0].element_size()
    assert views_w[1].data_ptr() - views_w[0].data_ptr() == stride13, (
        "交织 2 组视图未按前缀和排布"
    )
    # 空槽哨兵（库内 fill 已移除）：fill → barrier → 复推
    _fill_and_repush(buf, plan, (views_w, views_s))
    # 副本槽对拍（逐视图按其行宽——编码 ×(g+1) 区分组）
    _verify_prefetch_view(
        views_w[0].cpu(),
        etc_cpu,
        _PrefetchCase(rank, buf.epn, 2 * W),
        0,
        tag="w13 ",
    )
    _verify_prefetch_view(
        views_w[1].cpu(), etc_cpu, _PrefetchCase(rank, buf.epn, W), 1, tag="down "
    )
    # home 行保留
    assert torch.equal(
        views_w[0].cpu()[: buf.epn], _exp_home_rows(rank, buf.epn, 2 * W, 0)
    ), "w13 home 行未被写入"
    assert torch.equal(
        views_w[1].cpu()[: buf.epn], _exp_home_rows(rank, buf.epn, W, 1)
    ), "down home 行未被写入"

    # —— reduce 2 组 full-dW（[2*epn,2W] + [2*epn,W]）——
    base = 0.0
    g13 = _full_dw_2d(buf.epn, 2 * W, 0, base, buf.grad_dtype)
    g2 = _full_dw_2d(buf.epn, W, 1, base, buf.grad_dtype)
    buf.reduce_grad(plan, local_grads=[g13, g2])
    torch.npu.synchronize()
    _verify_merged(g13, rank, world, buf.epn, etc_cpu, 0, base, 2 * W, tag="w13 ")
    _verify_merged(g2, rank, world, buf.epn, etc_cpu, 1, base, W, tag="down ")
    _verify_replica_intact((g13, g2), buf.epn, base, tag="A12 ")


def _fill_intertwined_real(rank, epn, W, g=0):
    """W13 交织差分编码：gate 半区 = 编码 ×2，up 半区 = 编码 ×3。

    整行常值编码使行内半区交换不可见（A12 覆盖盲区）；差分编码使
    传输若意外交换前后半区（gate↔up）即被检出。
    """
    w13 = torch.empty(epn, 2 * W, dtype=torch.bfloat16)
    for i in range(epn):
        base_val = float((rank * 1000 + rank * epn + i) * (g + 1))
        w13[i, :W] = base_val * 2  # gate 半区（前 W 列）
        w13[i, W:] = base_val * 3  # up 半区（后 W 列）
    return w13.npu()


def _verify_intertwined_view(w13_cpu, etc_cpu, case, W, g=0, tag=""):
    """交织视图行内半区对拍：前 W 列 == gate 编码 ×2、后 W 列 == up 编码 ×3。

    逐槽逐半区校验——若传输交换了行内 gate/up 半区或行内 tile 边界
    错位，差分编码必被检出（整行常值则不可见）。
    """
    _assert_one_source_etc(etc_cpu, case.epn)
    rank, epn = case.rank, case.epn
    for j in range(epn):
        e = int(etc_cpu[rank][j])
        if e < 0:
            assert bool((w13_cpu[epn + j] == -1.0).all()), (
                f"{tag}空预取槽 {j} 被意外写入"
            )
            continue
        home = e // epn
        base_val = float((home * 1000 + e) * (g + 1))
        got = w13_cpu[epn + j]
        # gate 半区（前 W 列 == 编码 ×2）
        assert torch.allclose(
            got[:W].float(), torch.full((W,), base_val * 2), atol=1e-2, rtol=1e-2
        ), (
            f"{tag}槽 {j}（专家 {e}，home {home}）gate 半区期望 {base_val * 2}"
            f"，实得 {got[0].item():.1f}"
        )
        # up 半区（后 W 列 == 编码 ×3）
        assert torch.allclose(
            got[W:].float(), torch.full((W,), base_val * 3), atol=1e-2, rtol=1e-2
        ), (
            f"{tag}槽 {j}（专家 {e}，home {home}）up 半区期望 {base_val * 3}"
            f"，实得 {got[W].item():.1f}"
        )


@pytest.mark.npu
@_requires_train_env
def test_a12b_intertwined_half_gate_up(moonep_groups):
    """A12b：交织行内 gate/up 半区差分对拍——关闭 A12 的行内盲区。

    A12 用整行常值编码，行内 gate↔up 半区交换不可见（2W 字节全同一
    个数）。本用例在 W13 行内做差分编码：gate 半区（前 W 列）= 编码
    ×2，up 半区（后 W 列）= 编码 ×3——传输若意外交换前后半区或行内
    tile 边界错位，差分必被检出。同时校验 home 行半区一致。
    """
    rank, world = _init_group()
    buf = MoonepBuffer(
        _S,
        _H,
        _Hp,
        _K,
        _EPN * world,
        moonep_group=moonep_groups[0],
        ep_group=moonep_groups[1],
        slots=1,
    )
    _DEDICATED.append(buf)
    topk, tpe = _routing(43)
    plan = buf.plan(tpe, topk)
    etc_cpu = plan.experts_to_copy.cpu()
    assert int((etc_cpu >= 0).sum()) > 0, "balanced 路由未产生副本槽"
    W = buf.H * buf.Hp
    epn = buf.epn

    # prefetch 交织差分编码
    w13 = _fill_intertwined_real(rank, epn, W, 0)
    w2 = _fill_weight_real(rank, epn, W, 1)
    views_w, views_s = buf.prefetch_weight(plan, local_weights=(w13, w2))
    torch.npu.synchronize()
    assert views_w[0].shape == (2 * epn, 2 * W)

    # 空槽哨兵 + 复推
    _fill_and_repush(buf, plan, (views_w, views_s))

    # 行内半区差分对拍（核心新增验证）
    case = _PrefetchCase(rank, epn, 2 * W)
    _verify_intertwined_view(views_w[0].cpu(), etc_cpu, case, W, 0, tag="w13 半区 ")
    # w2 组用常值对拍（无交织——不受半区盲区影响）
    _verify_prefetch_view(
        views_w[1].cpu(), etc_cpu, _PrefetchCase(rank, epn, W), 1, tag="w2 "
    )

    # home 行半区一致（gate 前 / up 后）
    home_w13 = _fill_intertwined_real(rank, epn, W, 0)
    got_home = views_w[0][:epn].cpu()
    assert torch.allclose(
        got_home[:, :W].float(), home_w13.cpu()[:, :W].float(), atol=1e-2, rtol=1e-2
    ), "home 行 gate 半区不一致"
    assert torch.allclose(
        got_home[:, W:].float(), home_w13.cpu()[:, W:].float(), atol=1e-2, rtol=1e-2
    ), "home 行 up 半区不一致"


# ---------------------------------------------------------------------------
# A11 旋钮映射（slots / grad_dtype）
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_train_env
def test_a11_knob_mapping(moonep_groups):
    """A11：slots=2 堆宽翻倍对账（A6 复用）；grad_dtype='fp32' 档复跑 A5 路径
    ——覆盖 mem run() 的 --slots/--grad-dtype 两旋钮。"""
    rank, world = _init_group()
    # —— slots=2 构造档：堆需求增量 == 一倍窗口宽度（发布区随槽翻倍，槽区不变）——
    buf1 = _get_buffer(moonep_groups, slots=1)
    buf2 = _get_buffer(moonep_groups, slots=2)
    assert buf2.slots == 2
    window_bytes = _NUM_GROUPS * 2 * buf1.epn * buf1.H * buf1.Hp * 2
    assert buf2.heap_required_bytes - buf1.heap_required_bytes == window_bytes, (
        "slots=2 堆增量 != 一倍发布窗口宽度"
    )
    # —— grad_dtype='fp32' 档：A5 同款流程（fp32 槽区/种子/归并，逐位精确）——
    buf32 = _get_buffer(moonep_groups, grad_dtype="fp32")
    assert buf32.grad_dtype == torch.float32
    topk, tpe = _routing(42)
    plan = buf32.plan(tpe, topk)
    etc_cpu = plan.experts_to_copy.cpu()
    base = 0.0
    g1 = _full_dw_grads(buf32.epn, _Hp, _H, 0, base, torch.float32)
    g3 = _full_dw_grads(buf32.epn, _Hp, _H, 1, base, torch.float32)
    g2 = _full_dw_grads(buf32.epn, _H, _Hp, 2, base, torch.float32)
    buf32.reduce_grad(plan, local_grads=[g1, g3, g2])
    torch.npu.synchronize()
    for gi, t in enumerate((g1, g3, g2)):
        _verify_merged(
            t,
            rank,
            world,
            buf32.epn,
            etc_cpu,
            gi,
            base,
            buf32.H * buf32.Hp,
            tag=f"fp32 组 {gi} ",
        )


# ---------------------------------------------------------------------------
# B0 import 失败守卫（EB 依赖显性化的运行期证据）
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_train_env
def test_b0_import_failure_guidance(tmp_path):
    """B0：cann_ops_transformer 缺失时 import moonep_buffer 报错文案含
    "安装指引"要素（stub 触发真实 except ImportError 分支）。"""
    _init_group()
    stub = tmp_path / "cann_ops_transformer.py"
    stub.write_text('raise ImportError("simulated missing")\n', encoding="utf-8")
    # PYTHONPATH 置 stub 于前（优先于真实安装路径与 samples 目录）
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(tmp_path), _SAMPLES_DIR, os.environ.get("PYTHONPATH")) if p
    )
    r = subprocess.run(
        [sys.executable, "-c", "import moonep_buffer"],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode != 0, f"stub 导入意外成功：\n{r.stdout}\n{r.stderr}"
    err = r.stderr
    assert "cann_ops_transformer" in err, f"报错未指明缺失依赖：\n{err}"
    assert "安装" in err, f"报错缺安装指引：\n{err}"


# ---------------------------------------------------------------------------
# B1 dispatch/combine 前向 vs golden
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_train_env
def test_b1_dispatch_combine_forward_vs_golden(moonep_groups):
    """B1：前向 plan→dispatch→GMM 侧加权→combine ≈ golden（identity 专家，
    权重施加为 GMM 职责替身）；nmtr 推导断言；recv 扩展行数 == S*K；
    combine 权重回收（out_w 精确等于本 rank 路由权重）。"""
    rank, world = _init_group()
    buf = _get_buffer(moonep_groups)
    # —— nmtr 断言（= S，发送侧 per-peer 上界；运行时口径）——
    assert buf.nmtr == _S  # = S；1.25 余量公式已废弃（review 11-05）
    assert buf.H == _H
    shape = RouteShape(world, _S, _K, _EPN * world)
    topk_all, rw_all, hidden_all = _make_case(shape, 42)
    topk, tpe = _local_route(topk_all, rank, shape.num_experts)
    plan = buf.plan(tpe, topk)
    x = hidden_all[rank].npu()
    scores = rw_all[rank].npu()
    recv_x, recv_w, handle = buf.dispatch(plan, x, scores)  # do_cpu_sync=True 缺省
    torch.npu.synchronize()
    n = _S * _K
    # 扩展行数 == S*K（plan 均衡不变量在 EB 侧的体现；host 元数据零同步）
    assert int(handle.num_recv_tokens_per_expert.sum()) == n
    cap = recv_x.shape[0]
    assert cap >= n
    # GMM 侧加权 + cap 零填充契约（do_cpu_sync=True 下 cap 即精确行数，惯用法原样）
    y_w = (recv_x[:n].float() * recv_w[:n].unsqueeze(1)).to(recv_x.dtype)
    combine_in = torch.zeros(cap, _H, dtype=recv_x.dtype, device=recv_x.device)
    combine_in[:n] = y_w
    out, out_w = buf.combine(combine_in, handle, topk_weights=recv_w)
    torch.npu.synchronize()
    g_res = moonep_dispatch_golden(hidden_all, rw_all, topk_all, shape.num_experts)
    g_out = moonep_combine_golden(g_res, _S, _H)
    assert torch.allclose(out.cpu().float(), g_out[rank], atol=_ATOL, rtol=_RTOL), (
        "combine 输出与 golden 不符"
    )
    # combine 权重回收：精确等于本 rank 路由权重（fp32 逐位）
    assert torch.equal(out_w.cpu(), rw_all[rank]), "回收权重不符"


# ---------------------------------------------------------------------------
# B2 反向链路 vs golden
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_train_env
def test_b2_backward_chain_vs_golden(moonep_groups):
    """B2：反向链路——combine⁻¹（dispatch 散梯度：rw 随车 / scores=None 复用
    前向权重两形态）+ dispatch⁻¹（combine 裸和）对拍 golden 公式；
    + A5 方法 reduce 对拍（EB 在场用编码值法，规避 EI0006 的 all_reduce）。"""
    rank, world = _init_group()
    buf = _get_buffer(moonep_groups)
    shape = RouteShape(world, _S, _K, _EPN * world)
    topk_all, rw_all, hidden_all = _make_case(shape, 42)
    topk, tpe = _local_route(topk_all, rank, shape.num_experts)
    plan = buf.plan(tpe, topk)
    n = _S * _K
    # 前向 dispatch 一轮：取得 recv_w（mem 形态反向的权重复用源）
    _recv_x, recv_w, _handle = buf.dispatch(
        plan, hidden_all[rank].npu(), rw_all[rank].npu()
    )
    torch.npu.synchronize()
    gen = torch.Generator().manual_seed(999 + rank)
    g_out = torch.randn(_S, _H, generator=gen, dtype=torch.float32)
    exp_dx = torch.stack([rw_all[rank, s].sum() * g_out[s] for s in range(_S)])
    # —— combine⁻¹ 形态 1（ep 实证形态：rw 随车）——
    g_recv, g_rw, handle_g = buf.dispatch(
        plan, g_out.to(torch.bfloat16).npu(), rw_all[rank].npu()
    )
    torch.npu.synchronize()
    y_bwd = (g_recv[:n].float() * g_rw[:n].unsqueeze(1)).to(torch.bfloat16)
    cap_g = g_recv.shape[0]
    gin = torch.zeros(cap_g, _H, dtype=torch.bfloat16, device=g_recv.device)
    gin[:n] = y_bwd
    dx, _ = buf.combine(gin, handle_g)  # 裸和（topk_weights=None）
    torch.npu.synchronize()
    assert torch.allclose(dx.cpu().float(), exp_dx, atol=_ATOL, rtol=_RTOL), (
        "dispatch⁻¹（散梯度随车加权形态）输出与期望不符"
    )
    # —— combine⁻¹ 形态 2（mem 实证形态：scores=None，复用前向 recv_w 加权）——
    g_recv2, w2, handle_g2 = buf.dispatch(
        plan, g_out.to(torch.bfloat16).npu(), topk_weights=None
    )
    assert w2 is None, "topk_weights=None 时 EB 不随车权重（recv_w 应为 None）"
    torch.npu.synchronize()
    cap2 = g_recv2.shape[0]
    y2 = (g_recv2[:n].float() * recv_w[:n].unsqueeze(1)).to(torch.bfloat16)
    gin2 = torch.zeros(cap2, _H, dtype=torch.bfloat16, device=g_recv2.device)
    gin2[:n] = y2
    dx2, _ = buf.combine(gin2, handle_g2)
    torch.npu.synchronize()
    assert torch.allclose(dx2.cpu().float(), exp_dx, atol=_ATOL, rtol=_RTOL), (
        "dispatch⁻¹（topk_weights=None 复用前向权重形态）输出与期望不符"
    )
    # —— reduce 对拍（EB 在场——编码值法，无集合通信；full-dW 直入）——
    etc_cpu = plan.experts_to_copy.cpu()
    base = 0.0
    g1 = _full_dw_grads(buf.epn, _Hp, _H, 0, base, buf.grad_dtype)
    g3 = _full_dw_grads(buf.epn, _Hp, _H, 1, base, buf.grad_dtype)
    g2t = _full_dw_grads(buf.epn, _H, _Hp, 2, base, buf.grad_dtype)
    buf.reduce_grad(plan, local_grads=[g1, g3, g2t])
    torch.npu.synchronize()
    for gi, t in enumerate((g1, g3, g2t)):
        _verify_merged(
            t,
            rank,
            world,
            buf.epn,
            etc_cpu,
            gi,
            base,
            buf.H * buf.Hp,
            tag=f"B2 组{gi} ",
        )
    _verify_replica_intact((g1, g3, g2t), buf.epn, base, tag="B2 ")


# ---------------------------------------------------------------------------
# B3 destroy 的 EB 销毁策略
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_train_env
def test_b3_destroy_eb_default_keepalive(moonep_groups):
    """B3（缺省路径）：destroy() 后 dispatch 拒绝；EB 引用仍存活（池化保活——
    防 GC 中途析构）。release_eb=True 慢路径见 full 标记用例。"""
    _init_group()
    buf = _dedicated_buffer(moonep_groups)
    topk, tpe = _routing(42)
    plan = buf.plan(tpe, topk)
    buf.destroy()
    assert buf.destroyed
    x = torch.zeros(_S, _H, dtype=torch.bfloat16, device="npu")
    with pytest.raises(RuntimeError):
        buf.dispatch(plan, x)
    # 白盒断言（池化保活契约的运行期证据）：逻辑 destroy 后本对象仍持有 EB
    # 引用且未销毁（_runtime 非 None = 平台"未销毁"判定口径）
    assert buf._eb is not None  # noqa: SLF001
    assert getattr(buf._eb, "_runtime", None) is not None, (
        "缺省 destroy 不应销毁 EB（release_eb=False 池化保活）"
    )


@pytest.mark.slow
@pytest.mark.npu
@_requires_train_env
def test_b3_destroy_release_eb_slowpath(moonep_groups):
    """B3（opt-in 慢路径，slow 标记——-m "not slow" 可解筛）：release_eb=True
    显式销毁 EB 走通——先真实 dispatch 一次（通道建立后的销毁才是慢路径
    实证），接受 ~120s（conftest 的 AICPU_TS 等待经验）。"""
    _init_group()
    buf = _dedicated_buffer(moonep_groups)
    topk, tpe = _routing(42)
    plan = buf.plan(tpe, topk)
    x = torch.zeros(_S, _H, dtype=torch.bfloat16, device="npu")
    recv_x, _recv_w, _handle = buf.dispatch(plan, x)
    torch.npu.synchronize()
    assert recv_x.shape[0] >= _S * _K
    buf.destroy(release_eb=True)
    assert buf.destroyed
    # 平台约定：destroy 后 _runtime 置 None（check_no_shared_elastic_buffer
    # 对"已 destroy 实例"的同款判定口径）
    assert getattr(buf._eb, "_runtime", None) is None, "release_eb=True 应已销毁 EB"
    x2 = torch.zeros(_S, _H, dtype=torch.bfloat16, device="npu")
    with pytest.raises(RuntimeError):
        buf.dispatch(plan, x2)


# ---------------------------------------------------------------------------
# B4 do_cpu_sync=False 透传 + nmtr/容量对拍
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_train_env
def test_b4_do_cpu_sync_false_and_nmtr_capacity(moonep_groups):
    """B4：do_cpu_sync=False 透传（ep 实证路径回归：worst-case 容量 recv 直喂
    combine）+ nmtr 推导与堆容量对拍（构造期公式逐项核对——A1 口径的
    B 侧承载）。"""
    rank, world = _init_group()
    buf = _get_buffer(moonep_groups)
    # —— nmtr 推导与堆容量（构造期口径）——
    assert buf.nmtr == _S  # = S；1.25 余量公式已废弃（review 11-05）
    assert buf.nmtr >= _S
    assert buf.heap_bytes >= buf.heap_required_bytes > 0
    assert 2 * buf.heap_config_mb * 2**20 >= buf.heap_required_bytes
    grad_elem = 2 if buf.grad_dtype == torch.bfloat16 else 4
    need = (
        HEAP_BASE
        + reduce_region_end(  # 堆按 3W 总宽上限预留（组数运行期自适应）
            world,
            buf.epn,
            buf.H * buf.Hp,
            _NUM_GROUPS,
            buf.slots,
            grad_bytes_per_elem=grad_elem,
        )
        + LAYOUT_TAIL_BYTES
    )
    assert buf.heap_required_bytes == need, "heap_required_bytes 与手算公式不符"
    # —— do_cpu_sync=False 透传一轮（ep 实证形态：全容量张量直喂 combine）——
    shape = RouteShape(world, _S, _K, _EPN * world)
    topk_all, rw_all, hidden_all = _make_case(shape, 42)
    topk, tpe = _local_route(topk_all, rank, shape.num_experts)
    plan = buf.plan(tpe, topk)
    recv_x, recv_w, handle = buf.dispatch(
        plan, hidden_all[rank].npu(), rw_all[rank].npu(), do_cpu_sync=False
    )
    torch.npu.synchronize()
    n = _S * _K
    assert int(handle.num_recv_tokens_per_expert.sum()) == n
    # 免同步路径：recv 为 worst-case 容量，recv 全量加权后直喂 combine（ep 同款）
    y_w = (recv_x.float() * recv_w.unsqueeze(1)).to(recv_x.dtype)
    out, out_w = buf.combine(y_w, handle, topk_weights=recv_w)
    torch.npu.synchronize()
    g_res = moonep_dispatch_golden(hidden_all, rw_all, topk_all, shape.num_experts)
    g_out = moonep_combine_golden(g_res, _S, _H)
    assert torch.allclose(out.cpu().float(), g_out[rank], atol=_ATOL, rtol=_RTOL), (
        "do_cpu_sync=False 路径 combine 输出与 golden 不符"
    )
    assert torch.equal(out_w.cpu(), rw_all[rank]), "do_cpu_sync=False 回收权重不符"


# test_b5（pin_recv_rows 精确容量）已删：公开 pin API 经评审移除
# （容量/免同步语义由 test_b4_do_cpu_sync_false_and_nmtr_capacity 覆盖）


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


# ---------------------------------------------------------------------------
# A13 mxfp8（uint8 + ue8m0 scale）形态 C——R3/R2 完整形态（v2）
# ---------------------------------------------------------------------------

# mxfp8 专用形状：W = 64×16 = 1024（交织 (2W, W) 的自然 Sg = (64, 32)——
# 恰为 32B 对齐，无需 pad；主门禁 W=512 的自然 Sg=16 须 pad 到 32，本组
# 形状避开 pad 以聚焦数据面验证——pad 形态由 E3 负例覆盖契约侧）
_QUANT_HP = 32  # W = _H × _QUANT_HP = 1024（Sg (64,32) 自然 32B 对齐）
_QUANT_W = _H * _QUANT_HP
_QUANT_POOL = {}  # (world,) 键控共享 mxfp8 buffer


def _get_quant_buffer(moonep_groups):
    """构造（或复用）mxfp8 专用 buffer（weight_dtype=e4m3fn）。"""
    world = dist.get_world_size()
    key = (world,)
    if key not in _QUANT_POOL:
        _QUANT_POOL[key] = MoonepBuffer(
            _S,
            _H,
            _QUANT_HP,
            _K,
            _EPN * world,
            moonep_group=moonep_groups[0],
            ep_group=moonep_groups[1],
            weight_dtype="e4m3fn",
        )
    return _QUANT_POOL[key]


def _fill_fp8_real(rank, epn, w_elems, g=0):
    """e4m3fn 权重编码张量：行 i 常值 (rank*29 + i*13 + g*71) % 256。

    uint8 常值经 ``.to(torch.float8_e4m3fn)`` 数值化（dtype 终态）——
    对拍期望经同式 LUT（_e4m3_lut）重建，编码位确定性一致。
    """
    t = torch.empty(epn, w_elems, dtype=torch.uint8)
    for i in range(epn):
        t[i] = (rank * 29 + i * 13 + g * 71) % 256
    return t.to(torch.float8_e4m3fn).npu()


def _fill_scale_real(rank, epn, sg, g=0):
    """ue8m0 scale 编码张量：行 i 常值 (rank*17 + i*31 + g*97) % 256。"""
    t = torch.empty(epn, sg, dtype=torch.uint8)
    for i in range(epn):
        t[i] = (rank * 17 + i * 31 + g * 97) % 256
    return t.to(torch.float8_e8m0fnu).npu()


_E4M3_LUT = None
_E8M0_LUT = None


def _e4m3_lut():
    """uint8 值域 → e4m3fn 编码字节 LUT（惰性一次构建；对拍与填充同式）。"""
    global _E4M3_LUT
    if _E4M3_LUT is None:
        _E4M3_LUT = (
            torch.arange(256, dtype=torch.uint8)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
    return _E4M3_LUT


def _e8m0_lut():
    """uint8 值域 → e8m0fnu 编码字节 LUT（惰性一次构建）。"""
    global _E8M0_LUT
    if _E8M0_LUT is None:
        _E8M0_LUT = (
            torch.arange(256, dtype=torch.uint8)
            .to(torch.float8_e8m0fnu)
            .view(torch.uint8)
        )
    return _E8M0_LUT


def _verify_fp8_view(v_cpu, etc_cpu, case, g, encode, lut, tag=""):
    """1B 视图字节对拍：副本槽 == owner 编码字节、空槽保持 0xFF。

    对拍按 ``view(torch.uint8)`` 字节进行（fp8 数值比较对 NaN 编码位
    不可判等；字节语义 = 数据面真实搬运语义）。
    """
    _assert_one_source_etc(etc_cpu, case.epn)
    rank, epn = case.rank, case.epn
    for j in range(epn):
        e = int(etc_cpu[rank][j])
        if e < 0:
            assert bool((v_cpu[epn + j].view(torch.uint8) == 255).all()), (
                f"{tag}空预取槽 {j} 被意外写入"
            )
            continue
        home = e // epn
        exp_val = encode(home, e - home * epn, g)
        got = v_cpu[epn + j].view(torch.uint8)
        assert bool((got == lut[exp_val]).all()), (
            f"{tag}预取槽 {j}（专家 {e}，home {home}）期望字节 "
            f"{int(lut[exp_val])}，实得 {int(got[0].item())}"
        )


@pytest.mark.npu
@_requires_train_env
def test_a13_mxfp8_intertwined_scale(moonep_groups):
    """A13：形态 C——e4m3fn 权重 + ue8m0 scale 组端到端（交织 2 组）。

    ``weight_dtype=e4m3fn ⟹ 必传 local_scales``：权重 (W13, down)
    fp8 + scale (s13 [epn, 64], s2 [epn, 32])——scale 与权重共用全部
    控制面（同一张清单、同一次 launch、同一 barrier、同 per-expert DMA）；
    视图成对返回（weights 组 float8_e4m3fn + scales 组
    float8_e8m0fnu——dtype 终态）。随后 bf16
    full-dW reduce（grad_dtype 缺省 bf16——梯度与权重 dtype 无关）覆盖
    scale 感知的梯度槽区基址（发布区含 scale 后移）。
    """
    rank, world = _init_group()
    buf = _get_quant_buffer(moonep_groups)
    topk, tpe = _routing(57)
    plan = buf.plan(tpe, topk)
    etc_cpu = plan.experts_to_copy.cpu()
    assert int((etc_cpu >= 0).sum()) > 0, "balanced 路由未产生副本槽"
    epn, W = buf.epn, _QUANT_W
    sg13, sg2 = 2 * W // 32, W // 32  # (64, 32)——均 32B 对齐

    # —— prefetch：fp8 权重 + scale 组（E2：fp8 ⟺ 必传 scales）——
    w13 = _fill_fp8_real(rank, epn, 2 * W, 0)
    w2 = _fill_fp8_real(rank, epn, W, 1)
    s13 = _fill_scale_real(rank, epn, sg13, 0)
    s2 = _fill_scale_real(rank, epn, sg2, 1)
    views_w, views_s = buf.prefetch_weight(
        plan, local_weights=(w13, w2), local_scales=(s13, s2)
    )
    torch.npu.synchronize()

    # —— 返回形态：(weights, scales) 二元组两族（weights e4m3fn + scales e8m0fnu）——
    assert len(views_w) == 2 and len(views_s) == 2
    assert views_w[0].dtype == torch.float8_e4m3fn, (
        "mxfp8 权重视图须为 float8_e4m3fn（dtype 终态）"
    )
    assert views_s[0].dtype == torch.float8_e8m0fnu, (
        "mxfp8 scale 视图须为 float8_e8m0fnu（OCP MX 标准）"
    )
    assert views_w[0].shape == (2 * epn, 2 * W)
    assert views_s[0].shape == (2 * epn, sg13)
    assert views_s[1].shape == (2 * epn, sg2)
    # scale 视图置于同 slot 窗口内权重组之后（与权重视图同槽轮转）：
    # s13 基址 = w13 基址 + 2*epn*2W（字节）；s2 紧随 s13
    w13_bytes = views_w[0].numel()  # e4m3fn 1B：元素数 = 字节数
    w2_bytes = views_w[1].numel()
    assert views_s[0].data_ptr() - views_w[0].data_ptr() == (w13_bytes + w2_bytes), (
        "scale 视图未按窗口内前缀和置于权重组之后"
    )

    # 空槽哨兵（库内 fill 已移除）：fill → barrier → 复推（含 scale 组）
    _fill_and_repush(buf, plan, (views_w, views_s))
    # —— 数据面对拍（字节逐位；home 行 + 副本槽 + 空槽 0xFF）——
    case = _PrefetchCase(rank, epn, 2 * W)
    _verify_fp8_view(
        views_w[0].cpu(), etc_cpu, case, 0, _u8_row_val, _e4m3_lut(), "w13 "
    )
    _verify_fp8_view(
        views_w[1].cpu(), etc_cpu, case, 1, _u8_row_val, _e4m3_lut(), "w2 "
    )
    _verify_fp8_view(
        views_s[0].cpu(), etc_cpu, case, 0, _scale_row_val, _e8m0_lut(), "s13 "
    )
    _verify_fp8_view(
        views_s[1].cpu(), etc_cpu, case, 1, _scale_row_val, _e8m0_lut(), "s2 "
    )
    # home 行由 sync_transport 写入（字节对拍——与填充同式 LUT）
    for g, v in enumerate(views_w):
        assert bool(
            (
                v[:epn].cpu().view(torch.uint8)
                == _fill_fp8_real(rank, epn, v.shape[1], g).cpu().view(torch.uint8)
            ).all()
        ), f"fp8 组 {g} home 行未被同步写入"

    # —— bf16 full-dW reduce（梯度槽区基址含 scale 区域——离线对拍已验证
    #     grad_slot_off == 权重+scale 窗口步长；此为上板实证）——
    base = 0.0
    g13 = _full_dw_2d(epn, 2 * W, 0, base, torch.bfloat16)
    g2 = _full_dw_2d(epn, W, 1, base, torch.bfloat16)
    buf.reduce_grad(plan, local_grads=[g13, g2])
    torch.npu.synchronize()
    _verify_merged(g13, rank, world, epn, etc_cpu, 0, base, 2 * W, "mxfp8 w13 ")
    _verify_merged(g2, rank, world, epn, etc_cpu, 1, base, W, "mxfp8 w2 ")

    # —— 零拷贝复推：home 行切片直推（含 scale home 行——C8 同址跳过）——
    home_w = [v[:epn] for v in views_w]
    home_s = [v[:epn] for v in views_s]
    views2_w, views2_s = buf.prefetch_weight(
        plan, local_weights=home_w, local_scales=home_s
    )
    torch.npu.synchronize()
    _verify_fp8_view(
        views2_s[1].cpu(), etc_cpu, case, 1, _scale_row_val, _e8m0_lut(), "zc s2 "
    )


def _u8_row_val(rank, i, g):
    """fp8 权重行编码原值（与 _fill_fp8_real 同式——编码后经 LUT 换字节）。"""
    return (rank * 29 + i * 13 + g * 71) % 256


def _scale_row_val(rank, i, g):
    """ue8m0 scale 行编码（与 _fill_scale_real 同式）。"""
    return (rank * 17 + i * 31 + g * 97) % 256


@pytest.mark.npu
@_requires_train_env
def test_e2_fp8_requires_scales(moonep_groups):
    """E2 负例：fp8 权重（e4m3fn）而未传 local_scales → ValueError。"""
    _init_group()
    buf = _get_quant_buffer(moonep_groups)
    topk, tpe = _routing(57)
    plan = buf.plan(tpe, topk)
    rank = dist.get_rank()
    epn, W = buf.epn, _QUANT_W
    w13 = _fill_fp8_real(rank, epn, 2 * W, 0)
    w2 = _fill_fp8_real(rank, epn, W, 1)
    with pytest.raises(ValueError, match="local_scales"):
        buf.prefetch_weight(plan, local_weights=(w13, w2))


@pytest.mark.npu
@_requires_train_env
def test_e2_bf16_rejects_scales(moonep_groups):
    """E2 负例：bf16 权重传 local_scales → ValueError（scales ⟺ fp8 双向）。"""
    _init_group()
    buf = _get_buffer(moonep_groups)
    topk, tpe = _routing(42)
    plan = buf.plan(tpe, topk)
    rank = dist.get_rank()
    W = _H * _Hp
    reals = [_fill_weight_real(rank, buf.epn, W, g) for g in range(_NUM_GROUPS)]
    sg = (_H * _Hp) // 32 or 32  # 16 → 至少 32；E2 先于 E3 报错
    scales = [
        torch.full((buf.epn, sg), 3, dtype=torch.uint8).to(torch.float8_e8m0fnu).npu()
        for _ in range(_NUM_GROUPS)
    ]
    with pytest.raises(ValueError, match="fp8"):
        buf.prefetch_weight(plan, local_weights=reals, local_scales=scales)


@pytest.mark.npu
@_requires_train_env
def test_e2_scale_dtype_gate(moonep_groups):
    """E2 scale dtype 负例：非 float8_e8m0fnu（如 uint8）→ TypeError。"""
    _init_group()
    buf = _get_quant_buffer(moonep_groups)
    topk, tpe = _routing(57)
    plan = buf.plan(tpe, topk)
    rank = dist.get_rank()
    epn, W = buf.epn, _QUANT_W
    w13 = _fill_fp8_real(rank, epn, 2 * W, 0)
    w2 = _fill_fp8_real(rank, epn, W, 1)
    sg13 = 2 * W // 32
    s13 = _fill_scale_real(rank, epn, sg13, 0)
    # uint8 scale：dtype 终态后不再合法（MX 标准要求 e8m0fnu 原生编码）
    s2_u8 = _fill_scale_real(rank, epn, sg13, 1).view(torch.uint8)
    with pytest.raises(TypeError, match="e8m0fnu"):
        buf.prefetch_weight(plan, local_weights=(w13, w2), local_scales=(s13, s2_u8))


@pytest.mark.npu
@_requires_train_env
def test_e3_scale_align(moonep_groups):
    """E3 负例：scale 行宽非 32B 倍数 → ValueError（附 pad 建议）。"""
    _init_group()
    buf = _get_quant_buffer(moonep_groups)
    topk, tpe = _routing(57)
    plan = buf.plan(tpe, topk)
    rank = dist.get_rank()
    epn, W = buf.epn, _QUANT_W
    w13 = _fill_fp8_real(rank, epn, 2 * W, 0)
    w2 = _fill_fp8_real(rank, epn, W, 1)
    s13 = _fill_scale_real(rank, epn, 2 * W // 32, 0)
    s_bad = _fill_scale_real(rank, epn, 16, 1)  # 16B：非 32B 倍数
    with pytest.raises(ValueError, match="32"):
        buf.prefetch_weight(plan, local_weights=(w13, w2), local_scales=(s13, s_bad))


@pytest.mark.npu
@_requires_train_env
def test_e1_weight_dtype_mismatch(moonep_groups):
    """E1 负例：local_weights dtype 与构造 weight_dtype 不符 → TypeError。"""
    _init_group()
    buf = _get_quant_buffer(moonep_groups)
    topk, tpe = _routing(57)
    plan = buf.plan(tpe, topk)
    epn, W = buf.epn, _QUANT_W
    w13 = _fill_weight_real(dist.get_rank(), epn, 2 * W, 0)  # bf16 → 违约
    w2 = _fill_fp8_real(dist.get_rank(), epn, W, 1)
    with pytest.raises(TypeError, match="dtype"):
        buf.prefetch_weight(plan, local_weights=(w13, w2))


@pytest.mark.npu
@_requires_train_env
def test_e1_uint8_retired(moonep_groups):
    """E1 负例：weight_dtype=uint8 已退役（dtype 终态）→ TypeError。"""
    _init_group()
    world = dist.get_world_size()
    with pytest.raises(TypeError, match="uint8"):
        MoonepBuffer(
            _S,
            _H,
            _QUANT_HP,
            _K,
            _EPN * world,
            moonep_group=moonep_groups[0],
            ep_group=moonep_groups[1],
            weight_dtype="uint8",
        )


@pytest.mark.npu
@_requires_train_env
def test_e5m2_whitelist_and_e2(moonep_groups):
    """e5m2 白名单接受（dtype 守卫通过）+ E2 正确触发。

    e5m2 专用 buffer 构造成功（白名单接受）→ e5m2 权重不传 scales 触发
    E2 ValueError；补一次 e5m2 + e8m0fnu scales 的最小 e2e 推送对拍
    （kernel 侧 1B 字节语义与 e4m3fn 同路径，此处覆盖 host 侧映射）。"""
    rank, world = _init_group()
    buf = MoonepBuffer(
        _S,
        _H,
        _QUANT_HP,
        _K,
        _EPN * world,
        moonep_group=moonep_groups[0],
        ep_group=moonep_groups[1],
        weight_dtype="e5m2",
    )
    _DEDICATED.append(buf)
    assert buf.weight_dtype == torch.float8_e5m2
    topk, tpe = _routing(57)
    plan = buf.plan(tpe, topk)
    etc_cpu = plan.experts_to_copy.cpu()
    epn, W = buf.epn, _QUANT_W
    sg13, sg2 = 2 * W // 32, W // 32

    # —— e5m2 编码张量（uint8 常值数值化，与 e4m3fn 同式）——
    t13 = torch.empty(epn, 2 * W, dtype=torch.uint8)
    t2 = torch.empty(epn, W, dtype=torch.uint8)
    for i in range(epn):
        t13[i] = (rank * 29 + i * 13) % 256
        t2[i] = (rank * 29 + i * 13 + 71) % 256
    w13 = t13.to(torch.float8_e5m2).npu()
    w2 = t2.to(torch.float8_e5m2).npu()
    s13 = _fill_scale_real(rank, epn, sg13, 0)
    s2 = _fill_scale_real(rank, epn, sg2, 1)

    # E2：e5m2 不传 scales → ValueError
    with pytest.raises(ValueError, match="local_scales"):
        buf.prefetch_weight(plan, local_weights=(w13, w2))

    # 最小 e2e：e5m2 + e8m0fnu scales 推送对拍（字节 LUT 同式）
    views_w, views_s = buf.prefetch_weight(
        plan, local_weights=(w13, w2), local_scales=(s13, s2)
    )
    # 空槽哨兵（库内 fill 已移除）：fill → barrier → 复推（含 scale 组）
    _fill_and_repush(buf, plan, (views_w, views_s))
    assert views_w[0].dtype == torch.float8_e5m2
    e5m2_lut = (
        torch.arange(256, dtype=torch.uint8).to(torch.float8_e5m2).view(torch.uint8)
    )
    case = _PrefetchCase(rank, epn, 2 * W)
    _verify_fp8_view(
        views_w[0].cpu(), etc_cpu, case, 0, _u8_row_val, e5m2_lut, "e5m2 w13 "
    )
    _verify_fp8_view(
        views_s[1].cpu(), etc_cpu, case, 1, _scale_row_val, _e8m0_lut(), "e5m2 s2 "
    )


# ---------------------------------------------------------------------------
# A14（T4 覆盖加固 N5）：prefetch 行内元素级对拍——整行常值编码使行内
# 错位/tile 边界错位结构性不可见（kimi 以逐元素随机对拍为此），本用例
# 以逐元素随机权重关闭该盲区（bf16 字节直推应逐位一致）
# ---------------------------------------------------------------------------


def _randn_weight(rank, epn, w_elems):
    """逐元素随机权重 [epn, w] bf16（种子含 rank——CPU 期望可独立重建）。"""
    gen = torch.Generator().manual_seed(4321 + rank)
    return (
        torch.randn(epn, w_elems, generator=gen, dtype=torch.float32)
        .to(torch.bfloat16)
        .npu()
    )


@pytest.mark.npu
@_requires_train_env
def test_a14_prefetch_randn_elementwise(moonep_groups):
    """A14：逐元素随机权重的推送对拍——槽行 = owner 对应行逐位一致。"""
    rank, world = _init_group()
    buf = _get_buffer(moonep_groups)
    topk, tpe = _routing(67)
    plan = buf.plan(tpe, topk)
    etc_cpu = plan.experts_to_copy.cpu()
    epn = buf.epn

    reals = [_randn_weight(rank, epn, _W) for _ in range(_NUM_GROUPS)]
    views_w, views_s = buf.prefetch_weight(plan, local_weights=reals)
    # 空槽哨兵（库内 fill 已移除）：fill → barrier → 复推
    _fill_and_repush(buf, plan, (views_w, views_s))

    for g, v in enumerate(views_w):
        v_cpu = v.cpu()
        # home 行：本地 real 逐位
        gen = torch.Generator().manual_seed(4321 + rank)
        local = torch.randn(epn, _W, generator=gen, dtype=torch.float32).to(
            torch.bfloat16
        )
        assert torch.equal(v_cpu[:epn], local), f"A14 组 {g}: home 行元素级失配"
        # 副本槽行：owner 的对应行逐位（直推字节拷贝，无算术）
        for j in range(epn):
            e = int(etc_cpu[rank][j])
            if e < 0:
                assert bool((v_cpu[epn + j] == -1.0).all()), (
                    f"A14 组 {g} 空槽 {j} 被写入"
                )
                continue
            home = e // epn
            gen_h = torch.Generator().manual_seed(4321 + home)
            owner_w = torch.randn(epn, _W, generator=gen_h, dtype=torch.float32).to(
                torch.bfloat16
            )
            assert torch.equal(v_cpu[epn + j], owner_w[e - home * epn]), (
                f"A14 组 {g} 槽 {j}（专家 {e}，home {home}）元素级失配"
            )
