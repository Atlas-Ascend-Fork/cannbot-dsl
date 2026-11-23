# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""reduce_grad 语义回归套件（v2 §7：防退步 + E5 负例 + 容量 fail-fast）。

运行（4 卡起）：
    torchrun --nproc_per_node=4 -m pytest test/mc2/moonep/test_reduce_grad.py -q

覆盖（设计 moonep-api-v2-views-design-20261106.md §7）：
- 变宽组 2/3 × bf16 归并语义回归（home 行 = 种子 + Σ 副本贡献，逐位对拍）；
- dtype-E5 负例：local_grads dtype 与构造 grad_dtype 定型值不符 → TypeError；
- 容量 fail-fast：注入小堆 moonep_group → 发布区越界 RuntimeError（秒级
  明确错误，防运行中 507899 静默越界）；
- fp32 口径：R5 硬门禁（元素级诊断无可复现缺陷，直接硬校验）。

方法学与 test_moonep_buffer.py 同源（编码值法 + 等存量化期望公式）；本
文件聚焦 reduce_grad 单算子契约，不重复 buffer 套件的端到端链路。
"""

import os
import sys

import pytest

from conftest import _reduce_api  # noqa: E402
import torch
import torch.distributed as dist

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "samples", "mc2", "moonep"
    ),
)

try:
    import torch_npu  # noqa: F401
    from moonep_buffer import Buffer  # noqa: F401

    _HAS_DSL_IMPL = True
except Exception:  # CI 无 NPU / 无 DSL 时收集不报错（skip 而非 error）
    _HAS_DSL_IMPL = False


_requires_train_env = pytest.mark.skipif(
    not _HAS_DSL_IMPL
    or "RANK" not in os.environ
    or "WORLD_SIZE" not in os.environ
    or os.environ["WORLD_SIZE"] == "1",
    reason="需 torch.distributed.run 多卡拉起（单进程 pytest 收集形态跳过）",
)

_S, _K, _EPN = 32, 8, 4
_H, _Hp = 32, 16
_W = _H * _Hp
_H = 32
_NUM_GROUPS = 3

_R_POOL = {}  # (world, grad_dtype) 键控共享


def _init_group():
    if not dist.is_initialized():
        if "MASTER_ADDR" not in os.environ or "RANK" not in os.environ:
            pytest.skip("需 torchrun --nproc_per_node=N（N>=2）拉起")
        dist.init_process_group(backend="hccl")
    world = dist.get_world_size()
    if world < 2:
        pytest.skip("reduce_grad 用例需 >= 2 卡")
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    return rank, world


def _routing(seed, epn=None):
    """均衡路由 → (topk, tpe)（与 test_moonep_buffer 同法；epn 缺省模块值）。"""
    from moonep_golden import RouteShape, make_router_inputs

    world = dist.get_world_size()
    shape = RouteShape(world, _S, _K, (epn if epn is not None else _EPN) * world)
    topk_all, _ = make_router_inputs(shape, mode="balanced", seed=seed)
    topk = topk_all[dist.get_rank()].to(torch.int32).npu()
    tpe = (
        torch.bincount(topk_all[dist.get_rank()].flatten(), minlength=shape.num_experts)
        .to(torch.int32)
        .npu()
    )
    return topk, tpe


def _get_buffer(moonep_groups, grad_dtype="bf16"):
    """构造（或复用）池化 buffer（组型 3 等宽，本套件主门禁档）。"""
    world = dist.get_world_size()
    key = (world, str(grad_dtype))
    if key not in _R_POOL:
        _R_POOL[key] = Buffer(
            _S,
            _H,
            _Hp,
            _K,
            _EPN * world,
            num_ep_ranks=dist.get_world_size(),
            group=moonep_groups,
            grad_dtype=grad_dtype,
        )
    return _R_POOL[key]


def _full_dw(epn, w, g, base, dtype):
    """full-dW 张量 [2*epn, w]：行 i = (i+1)*(g+1)+base（行序同 prefetch 视图）。"""
    t = torch.empty(2 * epn, w, dtype=dtype, device="npu")
    for i in range(2 * epn):
        t[i] = float((i + 1) * (g + 1) + base)
    return t


def _expect_merged_row(rank, world_size, epn, etc_cpu, g, row, base):
    """归并期望：home 种子 + Σ 各 rank 副本行贡献。"""
    e = rank * epn + row
    exp_val = float((row + 1) * (g + 1) + base)
    for r in range(world_size):
        if r == rank:
            continue
        for j in range(epn):
            if int(etc_cpu[r][j]) == e:
                exp_val += float((epn + j + 1) * (g + 1) + base)
    return exp_val


def _verify_merged(t, rank, world, epn, etc_cpu, g, base, w, tag=""):
    """reduce 后逐行校验 home 段归并值。"""
    m = t[:epn].reshape(epn, w).cpu().float()
    for i in range(epn):
        exp_val = _expect_merged_row(rank, world, epn, etc_cpu, g, i, base)
        assert torch.allclose(m[i], torch.full((w,), exp_val), atol=1e-3, rtol=1e-3), (
            f"{tag}行 {i} 归并值期望 {exp_val}，实得 {m[i][0].item():.1f}"
        )


@pytest.mark.npu
@_requires_train_env
def test_r1_reduce_3group_bf16(moonep_groups):
    """R1：3 组等宽 bf16 归并语义回归（home = 种子 + Σ 副本贡献）。"""
    rank, world = _init_group()
    buf = _get_buffer(moonep_groups, "bf16")
    topk, tpe = _routing(11)
    plan = buf._plan(tpe, topk)
    etc_cpu = plan.experts_to_copy.cpu()
    assert int((etc_cpu >= 0).sum()) > 0
    base = 0.0
    grads = [_full_dw(buf.epn, _W, g, base, torch.bfloat16) for g in range(_NUM_GROUPS)]
    home_views = _reduce_api(buf, plan, local_grads=grads)  # sync：单值
    assert len(home_views) == _NUM_GROUPS
    torch.npu.synchronize()
    for g, t in enumerate(grads):
        _verify_merged(t, rank, world, buf.epn, etc_cpu, g, base, _W, f"R1 组{g} ")


@pytest.mark.npu
@_requires_train_env
def test_r2_reduce_2group_intertwined_bf16(moonep_groups):
    """R2：交织 2 组 (2W, W) 变宽组归并——一次调用、逐组独立前缀和槽区。

    专用 buffer：2 组结构定型 ctx.layout_groups（与池化 3 组互斥——组数
    首调定型守卫 E4）；reduce-only 直入（无 prefetch，layout 由 reduce
    首调定型）。
    """
    rank, world = _init_group()
    buf = Buffer(
        _S,
        _H,
        _Hp,
        _K,
        _EPN * world,
        num_ep_ranks=dist.get_world_size(),
        group=moonep_groups,
        grad_dtype="bf16",
    )
    topk, tpe = _routing(12)
    plan = buf._plan(tpe, topk)
    etc_cpu = plan.experts_to_copy.cpu()
    base = 1.0
    g13 = _full_dw(buf.epn, 2 * _W, 0, base, torch.bfloat16)
    g2 = _full_dw(buf.epn, _W, 1, base, torch.bfloat16)
    home_views = _reduce_api(buf, plan, local_grads=[g13, g2])
    assert len(home_views) == 2
    torch.npu.synchronize()
    _verify_merged(g13, rank, world, buf.epn, etc_cpu, 0, base, 2 * _W, "R2 w13 ")
    _verify_merged(g2, rank, world, buf.epn, etc_cpu, 1, base, _W, "R2 w2 ")


@pytest.mark.npu
@_requires_train_env
def test_r9_reduce_rotated_peer_blocks(moonep_groups, monkeypatch):
    """R9：块数封顶常驻核数后的 peer 轮转路径（issue #14 B1/B2）。

    R>64 时读/屏障块不再等于 R-1，而是封顶后在块内轮转多个 peer；本机只
    有 4 卡，把 resident_blocks 压到 2 即可在 R=4 复现同一路径（UBMEM 直累
    n_read=2/per=2），归并语义须与 block-per-peer 逐位一致。
    """
    rank, world = _init_group()
    import reduce_grad as rg_mod

    monkeypatch.setattr(rg_mod, "resident_blocks", lambda cap: 2)
    buf = _get_buffer(moonep_groups, "bf16")
    topk, tpe = _routing(19)
    plan = buf._plan(tpe, topk)
    etc_cpu = plan.experts_to_copy.cpu()
    assert int((etc_cpu >= 0).sum()) > 0
    base = 2.0
    grads = [_full_dw(buf.epn, _W, g, base, torch.bfloat16) for g in range(_NUM_GROUPS)]
    home_views = _reduce_api(buf, plan, local_grads=grads)
    assert len(home_views) == _NUM_GROUPS
    # 确实按 clamp 后的块数编译（否则本用例退化为 block-per-peer，白跑）
    specs = list(rg_mod._UBMEM_PROGRAMS)  # 直累程序缓存
    assert any(spec.blocks == 2 for spec in specs), "未走到块数 clamp 的轮转路径"
    torch.npu.synchronize()
    for g, t in enumerate(grads):
        _verify_merged(t, rank, world, buf.epn, etc_cpu, g, base, _W, f"R9 组{g} ")


@pytest.mark.npu
@_requires_train_env
def test_r3_e5_grad_dtype_mismatch(moonep_groups):
    """R3（E5 负例）：local_grads dtype ≠ 构造 grad_dtype 定型值 → TypeError。

    构造 grad_dtype 缺省 bf16；传 fp32 full-dW 须显式 grad_dtype="fp32"
    重建（梯度槽区 fp32 = bf16 的 2×，堆在建组时刻已定死——双真值源
    只能首调报错，不能静默扩容）。
    """
    _init_group()
    buf = _get_buffer(moonep_groups, "bf16")
    topk, tpe = _routing(13)
    plan = buf._plan(tpe, topk)
    grads = [_full_dw(buf.epn, _W, g, 0.0, torch.float32) for g in range(_NUM_GROUPS)]
    with pytest.raises(TypeError, match="grad_dtype"):
        _reduce_api(buf, plan, local_grads=grads)


@pytest.mark.npu
@_requires_train_env
def test_r4_capacity_failfast():
    """R4（容量 fail-fast）：小 workspace 裸 ctx → 接收区越界 RuntimeError。

    Buffer 构造按形状精算 workspace，不使用 EB/对称堆（
    heap_required_bytes == heap_bytes 恒成立），旧"注入小 hccl_buffer_size
    组"的注入面不复存在。等价 fail-fast：显式 workspace_bytes 仅过 plan
    区（构造检查）的裸 ctx 上调 reduce_grad——check_heap_capacity 在程序
    发射前秒级报错（需要/可用，防运行中静默越界）。
    """
    _init_group()
    dist.get_world_size()
    from moonep_context import create_moonep_context
    from moonep_layout import HEAP_BASE, PLAN_REGION_END
    from moonep_plan import moonep_plan
    from reduce_grad import reduce_grad

    # 仅过 plan 区（+64B 裕度）、不过权重发布区+梯度接收区（reduce 拦截）
    ctx = create_moonep_context(
        group=dist.group.WORLD,
        workspace_bytes=HEAP_BASE + PLAN_REGION_END + 64,
    )
    # 首调 plan 定型 last_plan_shape（reduce 的调用序守卫；plan 交换区在
    # 容量内——首版直接调 reduce 被该守卫拦截，A5 d5）
    topk, tpe = _routing(15)
    _, _, etc, _, _ = moonep_plan(ctx, tpe, topk)
    grads = [
        torch.zeros(2 * _EPN, _W, dtype=torch.bfloat16, device="npu")
        for _ in range(_NUM_GROUPS)
    ]

    with pytest.raises(RuntimeError, match="超出 workspace"):
        reduce_grad(ctx, etc, grads, _EPN)
    ctx.close()


@pytest.mark.full
@pytest.mark.npu
@_requires_train_env
def test_r5_reduce_fp32(moonep_groups):
    """R5：fp32 口径归并（full 门硬校验）。

    fp32 元素级探针（randn+常值双实验）无可复现缺陷，硬门禁直接校验。
    """
    rank, world = _init_group()
    buf = Buffer(
        _S,
        _H,
        _Hp,
        _K,
        _EPN * world,
        num_ep_ranks=dist.get_world_size(),
        group=moonep_groups,
        grad_dtype="fp32",
    )
    topk, tpe = _routing(15)
    plan = buf._plan(tpe, topk)
    etc_cpu = plan.experts_to_copy.cpu()
    base = 0.0
    grads = [_full_dw(buf.epn, _W, g, base, torch.float32) for g in range(_NUM_GROUPS)]
    _reduce_api(buf, plan, local_grads=grads)
    torch.npu.synchronize()
    for g, t in enumerate(grads):
        _verify_merged(t, rank, world, buf.epn, etc_cpu, g, base, _W, f"R5 组{g} ")


# ---------------------------------------------------------------------------
# R6-R8（T4 覆盖加固 P0：行内元素级数据面 + 空态/非对称/状态机——
# 盲区根因：既有对拍全部整行常值编码，行内元素错位/tile 边界错位/
# 原子宽度错位结构性不可见；kimi 仓以逐元素随机数据对拍正是为此）
# ---------------------------------------------------------------------------


def _randn_full_dw(rank, epn, w, g, dtype):
    """逐元素随机 full-dW [2*epn, w]（种子含 rank/g——CPU 可独立重建）。

    行内元素级：任意行内错位/原子宽度撕裂都会显形（整行常值则对称
    掩盖——同行同值，加错位置仍是同值）。
    """
    gen = torch.Generator().manual_seed(1234 + rank * 100 + g)
    return torch.randn(2 * epn, w, generator=gen, dtype=torch.float32).to(dtype).npu()


def _expect_randn_merged(world, epn, w, g, dtype, etc_cpu, rank):
    """CPU 重建期望：本 rank 种子 + Σ 他人副本行（序无关和的精确基准）。

    贡献映射须与 kernel/pack 语义一致（与常值档 ``_expect_merged_row`` 同
    模型）：``etc[r][j] = e`` 表示 peer r 的副本槽 j 承载专家 e——归并
    即 ``exp[e % epn] += dw_r[epn + j]``（对端槽 j 的梯度 = 其 full-dW 行
    epn+j；owner home 行 = e % epn）。两版历史误构造：首版误加全组种子
    （999ab6d 修）；第二版假设"槽 j 恒承载本地索引 j 的专家"（``e % epn
    == j`` 门 + ``exp[j] += dw[epn + j]``）——plan 的槽位是紧凑排布
    （_fill_rep_copy 按专家号升序从 0 压缩填槽，非 e%epn 对位；一专家可
    多 rank 副本），v24 8P routing-21 下 etc[1][0]=15 / etc[3,6,7][0]=17 /
    etc[2][0]=31 全部落 j=0，rank 3/4/7 的行 3/1/3 期望漏加（got 实为
    正确——8P 探针 kernel 语义逐 rank 对拍全过）。"""
    gen_me = torch.Generator().manual_seed(1234 + rank * 100 + g)
    exp = (
        torch.randn(2 * epn, w, generator=gen_me, dtype=torch.float32)[:epn]
        .to(torch.float64)
        .clone()
    )
    for r in range(world):
        if r == rank:
            continue
        gen = torch.Generator().manual_seed(1234 + r * 100 + g)
        dw = torch.randn(2 * epn, w, generator=gen, dtype=torch.float32).to(
            torch.float64
        )
        for j in range(epn):
            e = int(etc_cpu[r][j])
            if e >= 0 and e // epn == rank:
                exp[e % epn] += dw[epn + j]
    return exp.to(dtype).cpu()


@pytest.mark.npu
@_requires_train_env
def test_r6_reduce_randn_fp32_bf16(moonep_groups):
    """R6（N1）：逐元素随机 fp32/bf16 双档归并对拍——U5 首要复现器。

    fp32 档若 0.8 原子路径按 2B 假设 lowering（F1）或原子宽度撕裂，
    行内元素级错位必然显形；bf16 档容差 rel 5%（原子串行化舍入）。
    副本段只读契约同步校验。

    rank 分歧告警：本用例双 dtype 循环中途 assert 失败会让
    失败 rank 提前进入 R7、存活 rank 留在 bf16 档——两组 plan/barrier 的
    进程内 epoch 计数在同值上碰撞（plan _EPOCH 与 _BAR_EPOCH 均为进程级
    全局计数），同一堆内交换互穿：需求矩阵混行 → 混合 etc、槽区数据互
    读（R7 "期望 8.0 实得 2.5" 即 3.0+对端 bf16 randn）→ 存活 rank 挂死
    于后续 poll。诊断时只信首个失败；级联失败的形态（常值档实得非整值）
    是分歧标志。pytest 断言不退出进程，跨 rank 收敛类用例的分歧本就
    未定义——修期望后分歧源消失。
    """
    rank, world = _init_group()
    etc_cpu = None
    for dtype, tol in (
        (torch.float32, dict(atol=1e-4, rtol=1e-5)),
        (torch.bfloat16, dict(atol=2e-2, rtol=5e-2)),
    ):
        # 池化（按 (world, grad_dtype) 键）：直构双 buffer 时第 5 个子组
        # 的 get_hccl_comm_name 报 HCCL 设备错 4（进程内组/EB 累积资源
        # 面——v17 mixed pfw 同族）；池化后 fp32 仅增 1 组、bf16 复用 R1 池
        gdtype = "fp32" if dtype == torch.float32 else "bf16"
        buf = _get_buffer(moonep_groups, gdtype)
        topk, tpe = _routing(21)
        plan = buf._plan(tpe, topk)
        etc_cpu = plan.experts_to_copy.cpu()
        grads = [
            _randn_full_dw(rank, buf.epn, _W, g, dtype) for g in range(_NUM_GROUPS)
        ]
        _reduce_api(buf, plan, local_grads=grads)
        torch.npu.synchronize()
        for g, t in enumerate(grads):
            exp = _expect_randn_merged(world, buf.epn, _W, g, dtype, etc_cpu, rank)
            got = t[: buf.epn].reshape(buf.epn, _W).cpu().to(torch.float64)
            assert torch.allclose(got, exp.to(torch.float64), **tol), (
                f"R6 {gdtype} 组 {g}: 元素级归并失配 "
                f"max_diff={(got - exp).abs().max().item():.3e}"
            )
            # 副本段只读契约
            gen = torch.Generator().manual_seed(1234 + rank * 100 + g)
            rep_exp = (
                torch.randn(2 * buf.epn, _W, generator=gen, dtype=torch.float32)
                .to(dtype)[buf.epn :]
                .cpu()
            )
            assert torch.allclose(
                t[buf.epn :].reshape(buf.epn, _W).cpu().to(torch.float64),
                rep_exp.to(torch.float64),
                **tol,
            ), f"R6 {gdtype} 组 {g}: 副本段被 reduce 破坏（应只读）"


def _local_only_routing():
    """全 rank 只选本地专家的路由 → plan 无跨 rank 需求（etc 恒 -1）。"""
    world = dist.get_world_size()
    rank = dist.get_rank()
    gen = torch.Generator().manual_seed(77)
    topk = torch.empty(_S, _K, dtype=torch.int64)
    for k in range(_K):
        topk[:, k] = torch.randint(rank * _EPN, (rank + 1) * _EPN, (_S,), generator=gen)
    tpe = torch.bincount(topk.flatten(), minlength=_EPN * world)
    return topk.to(torch.int32).npu(), tpe.to(torch.int32).npu()


@pytest.mark.npu
@_requires_train_env
def test_r7_reduce_empty_and_asymmetric(moonep_groups):
    """R7（N2/N3）：空 plan 不挂死/零触碰 + 三连发 active→empty→active 状态机。

    空 plan（etc 恒 -1）：reduce 应零通信零触碰（home 仅本地种子）；
    中夹空轮后 epoch/barrier 自复位——第三轮 active 归并与首轮独立一致。
    """
    rank, world = _init_group()
    buf = _get_buffer(moonep_groups, "bf16")
    # 轮 1：active（正常路由）
    topk, tpe = _routing(31)
    plan = buf._plan(tpe, topk)
    etc_active = plan.experts_to_copy.cpu()
    g1 = [_full_dw(buf.epn, _W, g, 0.0, torch.bfloat16) for g in range(3)]
    _reduce_api(buf, plan, local_grads=g1)
    torch.npu.synchronize()
    _verify_merged(g1[0], rank, world, buf.epn, etc_active, 0, 0.0, _W, "R7 轮1 ")

    # 轮 2：空 plan（全 rank 只选本地专家 → etc 恒 -1）
    topk_e, tpe_e = _local_only_routing()
    plan_e = buf._plan(tpe_e, topk_e)
    etc_empty = plan_e.experts_to_copy.cpu()
    assert not bool((etc_empty >= 0).any()), "本地路由不应产生副本清单"
    g2 = [_full_dw(buf.epn, _W, g, 5.0, torch.bfloat16) for g in range(3)]
    _reduce_api(buf, plan_e, local_grads=g2)
    torch.npu.synchronize()
    # 零触碰：home = 纯本地种子（_full_dw 行变编码 (i+1)·(g+1)+base）
    for g, t in enumerate(g2):
        m = t[: buf.epn].reshape(buf.epn, _W).cpu().float()
        exp = (
            torch.tensor(
                [float((i + 1) * (g + 1) + 5.0) for i in range(buf.epn)],
                dtype=torch.float32,
            )
            .unsqueeze(1)
            .expand(buf.epn, _W)
        )
        assert torch.allclose(m, exp, atol=1e-3, rtol=1e-3), (
            f"R7 空轮组 {g}: home 期望种子行 {(g + 1) + 5.0}..，"
            f"实得 {m[0][0].item():.2f}"
        )

    # 轮 3：active 复位（独立归并，base=7 区分）
    topk3, tpe3 = _routing(31)
    plan3 = buf._plan(tpe3, topk3)
    assert bool((plan3.experts_to_copy.cpu() >= 0).any()) or world == 1
    g3 = [_full_dw(buf.epn, _W, g, 7.0, torch.bfloat16) for g in range(3)]
    _reduce_api(buf, plan3, local_grads=g3)
    torch.npu.synchronize()
    _verify_merged(g3[0], rank, world, buf.epn, etc_active, 0, 7.0, _W, "R7 轮3 ")


@pytest.mark.npu
@_requires_train_env
def test_r8_reduce_multichunk_column_position(moonep_groups):
    """R8：多 chunk 列位置回归（W=32768 > chunk 16384，n_c=2）。

    行内列位置编码（半区异值）：远读须带列块偏移 c——若漏加（历史缺陷：
    每个列块都并入远端行首 chunk），后半区会归并到错误值。行恒值填法
    检测不了本缺陷，故本用例列值随半区分。
    """
    rank, world = _init_group()
    epn = _EPN
    w = 32768  # bf16 chunk=16384 → n_c=2
    half = w // 2

    buf = Buffer(
        _S,
        _H,
        w // _H,  # Hp=1024 → W=32768（专用 buffer——形状键独立）
        _K,
        _EPN * world,
        num_ep_ranks=dist.get_world_size(),
        group=moonep_groups,
        grad_dtype="bf16",
    )
    try:
        topk, tpe = _routing(23)
        plan = buf._plan(tpe, topk)
        etc_cpu = plan.experts_to_copy.cpu()
        assert int((etc_cpu >= 0).sum()) > 0

        def _row_val(i, g):
            return float((i + 1) * (g + 1))

        # full-dW：行 i 的前半区 = row_val(i)，后半区 = row_val(i)+8（值域
        # ≤ ~130——bf16 整数精确域内；+8 使列半区可判而总和不失精确）
        grads = []
        for g in range(_NUM_GROUPS):
            t = torch.empty(2 * epn, w, dtype=torch.bfloat16, device="npu")
            for i in range(2 * epn):
                t[i, :half] = _row_val(i, g)
                t[i, half:] = _row_val(i, g) + 8.0
            grads.append(t)

        _reduce_api(buf, plan, local_grads=grads)  # sync：单值——归并原位
        torch.npu.synchronize()

        for g, t in enumerate(grads):
            m = t[:epn].cpu().float()
            for i in range(epn):
                # 期望 = home 种子 + Σ 各 rank 副本贡献（同 _expect_merged_row
                # 的 etc 语义），按半区独立计算
                e = rank * epn + i
                exp_lo = _row_val(i, g)
                exp_hi = _row_val(i, g) + 8.0
                for r in range(world):
                    if r == rank:
                        continue
                    for j in range(epn):
                        if int(etc_cpu[r][j]) == e:
                            exp_lo += _row_val(epn + j, g)
                            exp_hi += _row_val(epn + j, g) + 8.0
                got_lo = m[i, 0].item()
                got_hi = m[i, half].item()
                assert abs(got_lo - exp_lo) < 1e-2, (
                    f"R8 g={g} 行{i} 前半区：期望 {exp_lo}，实得 {got_lo}"
                )
                assert abs(got_hi - exp_hi) < 1e-2, (
                    f"R8 g={g} 行{i} 后半区：期望 {exp_hi}，实得 {got_hi}"
                    "——远读缺列块偏移（首 chunk 并入每个列块）的典型症状"
                )
    finally:
        buf.destroy()
