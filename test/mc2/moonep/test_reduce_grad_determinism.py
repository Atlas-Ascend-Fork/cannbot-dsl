# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""reduce_grad 确定性归并验收套件（确定性归并——唯一累加路径）。

逐位验收方法论：
- D1 逐位钉序：randn 数据 + host 参照（fp32 串行加法重放 kernel 相同
  加法序列：seed → peer 升序 → 终轮一次舍入）→ torch.equal 逐位对比；
  连发 3 次同输入，输出两两 torch.equal（run-to-run 确定性直接证据）。
- D2 精确数据逐位：整值编码（bf16 精确可加）→ torch.equal。
- D3 副本区只读不变量：reduce 前后 workspace 副本区逐位不变
  （协议无"消费清零"——写侧归 host 入槽，kernel 只读）。
- D5 生命周期：单 buffer 三轮（active/empty/active）——空轮不卡死、
  epoch/屏障自复位。
- D6 跨 rank 聚合：断言经 all_reduce(MIN)——单 rank 失败全组失败。

运行（4 卡起，逐文件独立进程）：
    确定性归并为唯一累加路径，无需环境开关。
    torchrun --nproc_per_node=4 -m pytest test/mc2/moonep/test_reduce_grad_determinism.py -q
"""

import os
import sys

import pytest
import torch
import torch.distributed as dist

from conftest import _reduce_api

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

_S, _K, _EPN = 32, 8, 4
_W = 512  # VL=64 的倍数（尾块越界防护）；≤16K → 单 chunk/行


# ---------------------------------------------------------------------------
# 数据生成（各 rank 逐位可再生——同构节点同种子同比特）
# ---------------------------------------------------------------------------


def _randn_grads(rank, world, epn, widths, dtype, seed):
    """全组 full-dW 张量组（每 rank 一份 [2*epn, Wg]×组——本 rank 供种子/
    副本，对端供参照重放）。Philox 按 rank 隔离种子，逐位可再生。"""
    groups = []
    for g, w in enumerate(widths):
        gen = torch.Generator().manual_seed(30_000 + 1_000 * int(seed) + 8 * rank + g)
        t = torch.randn((2 * epn, w), generator=gen, dtype=torch.float32)
        groups.append(t.to(dtype).npu())
    return groups


def _local_grads_of(all_grads, rank):
    """本 rank 的输入组（原位归并——先 clone 防跨轮污染）。"""
    return [t.clone() for t in all_grads[rank]]


def _expected_home(all_grads, rank, world, epn, etc_cpu, g, i, w):
    """host 参照：重放 kernel 的加法序列（seed → peer 升序 → 一次舍入）。

    与 kernel 逐位同构：bf16→fp32 无损展开、fp32 定序累加（RN）、
    终轮一次 .to(dtype)（RN）——任何顺序漂移都会逐位暴露。
    """
    e = rank * epn + i
    seed = all_grads[rank][g][i]
    acc = seed.to(torch.float32)
    for r in range(world):
        if r == rank:
            continue
        for j in range(epn):
            if int(etc_cpu[r][j]) == e:
                contrib = all_grads[r][g][epn + j]
                acc = acc + contrib.to(torch.float32)
    return acc.to(seed.dtype)


def _assert_all_ranks(ok, label):
    """跨 rank 断言聚合（MIN）：单 rank 失败即全组失败。"""
    if not ok:
        print(f"[rank {dist.get_rank()}] FAIL: {label}")
    flag = torch.tensor([int(bool(ok))], dtype=torch.int32, device="npu")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    assert int(flag.item()) == 1, label


def _routing(seed):
    """均衡路由 → (topk, tpe)（与 test_reduce_grad 同法）。"""
    from moonep_golden import RouteShape, make_router_inputs

    world = dist.get_world_size()
    shape = RouteShape(world, _S, _K, _EPN * world)
    topk_all, _ = make_router_inputs(shape, mode="balanced", seed=seed)
    topk = topk_all[dist.get_rank()].to(torch.int32).npu()
    tpe = (
        torch.bincount(topk_all[dist.get_rank()].flatten(), minlength=shape.num_experts)
        .to(torch.int32)
        .npu()
    )
    return topk, tpe


def _local_only_routing():
    """全 rank 只选本地专家 → plan 无跨 rank 需求（etc 恒 -1）。"""
    world = dist.get_world_size()
    rank = dist.get_rank()
    gen = torch.Generator().manual_seed(77)
    topk = torch.empty(_S, _K, dtype=torch.int64)
    for k in range(_K):
        topk[:, k] = torch.randint(rank * _EPN, (rank + 1) * _EPN, (_S,), generator=gen)
    tpe = torch.bincount(topk.flatten(), minlength=_EPN * world)
    return topk.to(torch.int32).npu(), tpe.to(torch.int32).npu()


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------


@pytest.mark.npu
@_requires_train_env
@pytest.mark.parametrize(
    "dtype_name,torch_dtype",
    [
        ("fp32", torch.float32),
        ("bf16", torch.bfloat16),
        ("fp16", torch.float16),
    ],
    ids=["fp32", "bf16", "fp16"],
)
def test_d1_bitwise_pinned(dtype_name, torch_dtype, moonep_groups):
    """D1 核心验收：randn + 参照逐位 + 连发 3 次 run-to-run 逐位。

    randn 数据 fp32 加法逐次舍入——逐位对比把加法顺序钉死到
    seed → peer 升序；三次重复发射的输出两两 torch.equal 是
    run-to-run 确定性的直接证据。
    """
    rank, world = _init_group()
    buf = _get_buffer(moonep_groups, dtype_name)
    epn = buf.epn
    topk, tpe = _routing(41)
    plan = buf._plan(tpe, topk)
    etc_cpu = plan.experts_to_copy.cpu()
    assert int((etc_cpu >= 0).any()) > 0

    all_grads = [
        _randn_grads(r, world, buf.epn, (_W,) * 3, torch_dtype, seed=41)
        for r in range(world)
    ]
    outs = []
    for launch in range(3):
        grads = _local_grads_of(all_grads, rank)
        _reduce_api(buf, plan, grads)
        torch.npu.synchronize()
        outs.append([t[:epn].clone() for t in grads])
        if launch == 0:
            # 参照逐位（首轮即验——后续轮验 run-to-run 稳定）
            for g in range(3):
                for i in range(buf.epn):
                    exp = _expected_home(
                        all_grads, rank, world, buf.epn, etc_cpu, g, i, _W
                    )
                    got = outs[0][g][i].cpu()
                    _assert_all_ranks(
                        torch.equal(got, exp.cpu()),
                        f"D1 dtype={dtype_name} 组{g} 行{i} 逐位不符"
                        f"（max_diff={float((got.float() - exp.cpu().float()).abs().max())}）",
                    )
    # run-to-run：三轮输出两两逐位
    for launch in (1, 2):
        for g in range(3):
            _assert_all_ranks(
                torch.equal(outs[launch][g].cpu(), outs[0][g].cpu()),
                f"D1 run-to-run 第{launch}轮 组{g} 输出与首轮逐位不符",
            )
    if rank == 0:
        print(f"[DET] D1 dtype={dtype_name}: bitwise + run-to-run x3 PASS")
    buf.destroy()


@pytest.mark.npu
@_requires_train_env
def test_d1b_multichunk(moonep_groups):
    """D1 变体：多 chunk 行（W=32768 → chunk 16K × 2）——tile 循环 + 环
    滞后窗跨 chunk 正确性。
    """
    rank, world = _init_group()
    weight_rows = 32768
    buf = Buffer(
        _S,
        64,
        weight_rows,
        _K,
        _EPN * world,
        world,
        group=moonep_groups,
        grad_dtype="bf16",
    )
    epn = buf.epn
    topk, tpe = _routing(43)
    plan = buf._plan(tpe, topk)
    etc_cpu = plan.experts_to_copy.cpu()
    all_grads = [
        _randn_grads(r, world, buf.epn, (weight_rows,) * 2, torch.bfloat16, seed=43)
        for r in range(world)
    ]
    grads = _local_grads_of(all_grads, rank)
    _reduce_api(buf, plan, grads)
    torch.npu.synchronize()
    for g in range(2):
        for i in range(buf.epn):
            exp = _expected_home(
                all_grads, rank, world, buf.epn, etc_cpu, g, i, weight_rows
            )
            got = grads[g][:epn][i].cpu()
            _assert_all_ranks(torch.equal(got, exp.cpu()), f"D1b 组{g} 行{i} 逐位不符")
    if rank == 0:
        print(f"[DET] D1b multichunk W={weight_rows}: bitwise PASS")
    buf.destroy()


@pytest.mark.npu
@_requires_train_env
@pytest.mark.parametrize("dtype_name", ["fp32", "bf16", "fp16"])
def test_d2_encoded_exact(dtype_name, moonep_groups):
    """D2：整值编码（16 位下精确可加，和 ≤ 2^8）→ torch.equal 逐位。"""
    rank, world = _init_group()
    torch_dtype = _dt_of(dtype_name)
    buf = _get_buffer(moonep_groups, dtype_name)
    epn = buf.epn
    topk, tpe = _routing(45)
    plan = buf._plan(tpe, topk)
    etc_cpu = plan.experts_to_copy.cpu()

    # 行编码：行 r 的 home 行 i = 2 + (i+1)（小整数，7 次加仍精确）
    groups = []
    for g in range(3):
        t = torch.zeros(2 * buf.epn, _W, dtype=torch_dtype)
        for r_i in range(2 * buf.epn):
            t[r_i] = float(2 + (r_i + 1))
        groups.append(t.npu())
    _reduce_api(buf, plan, groups)
    torch.npu.synchronize()
    for g in range(3):
        for i in range(buf.epn):
            e = rank * buf.epn + i
            exp_val = float(2 + (i + 1))
            for r in range(world):
                if r == rank:
                    continue
                for j in range(buf.epn):
                    if int(etc_cpu[r][j]) == e:
                        exp_val += float(2 + (buf.epn + j + 1))
            got = groups[g][:epn][i].cpu()
            exp = torch.full((_W,), exp_val, dtype=torch_dtype)
            _assert_all_ranks(
                torch.equal(got, exp),
                f"D2 dtype={dtype_name} 组{g} 行{i}: 期望 {exp_val} 实得 {got[0].item()}",
            )
    if rank == 0:
        print(f"[DET] D2 dtype={dtype_name}: encoded exact PASS")
    buf.destroy()


@pytest.mark.npu
@_requires_train_env
def test_d3_workspace_readonly(moonep_groups):
    """D3：活跃副本槽只读不变量——reduce 后本 rank 已发布槽与入槽内容
    逐位一致（kernel 只读对端；未参与计划的槽不进入数据面）。
    """
    rank, world = _init_group()
    buf = _get_buffer(moonep_groups, "bf16")
    topk, tpe = _routing(47)
    plan = buf._plan(tpe, topk)
    all_grads = [
        _randn_grads(r, world, buf.epn, (_W,) * 3, torch.bfloat16, seed=47)
        for r in range(world)
    ]
    grads = _local_grads_of(all_grads, rank)
    _reduce_api(buf, plan, grads)
    torch.npu.synchronize()
    epn = buf.epn
    ws_views = getattr(buf._ctx, "_pull_ws_copy_views", None)
    ok = ws_views is not None
    if ok:
        active_rows = (
            plan.experts_to_copy[rank].ge(0).nonzero(as_tuple=False).flatten().cpu()
        )
        if int(active_rows.numel()):
            for g in range(3):
                actual = ws_views[1][g].cpu()[active_rows]
                expected = grads[g][epn:].cpu()[active_rows]
                ok = ok and torch.equal(actual, expected)
    _assert_all_ranks(ok, "D3 活跃副本槽被 reduce 改写（只读不变量破坏）")
    if rank == 0:
        print("[DET] D3 workspace readonly invariant PASS")
    buf.destroy()


@pytest.mark.npu
@_requires_train_env
def test_d5_lifecycle_rounds(moonep_groups):
    """D5 生命周期：单 buffer 三轮（active → empty → active）——空轮
    零贡献不挂死，epoch/屏障自复位，第三轮逐位独立正确。
    """
    rank, world = _init_group()
    buf = _get_buffer(moonep_groups, "bf16")
    epn = buf.epn
    # 轮 1：active
    topk, tpe = _routing(51)
    plan = buf._plan(tpe, topk)
    etc_1 = plan.experts_to_copy.cpu()
    grads1 = _local_grads_of(_randn_grads_set(51, world, buf.epn), rank)
    _reduce_api(buf, plan, grads1)
    torch.npu.synchronize()
    _verify_against_reference(
        grads1,
        all_grads=None,
        rank=rank,
        world=world,
        buf=buf,
        etc_cpu=etc_1,
        seed=51,
        tag="D5 轮1",
    )
    # 轮 2：empty（本地路由）——home = 纯种子（逐位）
    topk_e, tpe_e = _local_only_routing()
    plan_e = buf._plan(tpe_e, topk_e)
    etc_e = plan_e.experts_to_copy.cpu()
    assert not bool((etc_e >= 0).any()), "本地路由不应产生副本清单"
    grads2 = _local_grads_of(_randn_grads_set(52, world, buf.epn), rank)
    seed_snap = [t[:epn].clone() for t in grads2]
    _reduce_api(buf, plan_e, grads2)
    torch.npu.synchronize()
    for g in range(3):
        _assert_all_ranks(
            torch.equal(grads2[g][:epn].cpu(), seed_snap[g].cpu()),
            f"D5 空轮组{g}: home 应逐位等于种子（零贡献零触碰）",
        )
    # 轮 3：active 复位
    topk3, tpe3 = _routing(53)
    plan3 = buf._plan(tpe3, topk3)
    etc_3 = plan3.experts_to_copy.cpu()
    grads3 = _local_grads_of(_randn_grads_set(53, world, buf.epn), rank)
    _reduce_api(buf, plan3, grads3)
    torch.npu.synchronize()
    _verify_against_reference(
        grads3,
        all_grads=None,
        rank=rank,
        world=world,
        buf=buf,
        etc_cpu=etc_3,
        seed=53,
        tag="D5 轮3",
    )
    if rank == 0:
        print("[DET] D5 lifecycle active/empty/active PASS")
    buf.destroy()


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


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


_BUF_POOL = {}


def _dt_of(name):
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def _get_buffer(moonep_groups, dtype_name):
    """池化 buffer（组型 3 等宽；dtype 键控共享——梯度槽区按定型 dtype 预留）。"""
    world = dist.get_world_size()
    key = (world, str(dtype_name))
    buf = _BUF_POOL.get(key)
    if buf is None or buf.destroyed:
        _BUF_POOL[key] = Buffer(
            _S,
            64,
            _W,
            _K,
            _EPN * world,
            world,
            group=moonep_groups,
            grad_dtype=dtype_name,
        )
    return _BUF_POOL[key]


def _randn_grads_set(seed, world, epn):
    """全组 grads（D5 三轮复用同一生成器形态）。"""
    return [
        _randn_grads(r, world, epn, (_W,) * 3, torch.bfloat16, seed=seed)
        for r in range(world)
    ]


def _verify_against_reference(grads, all_grads, rank, world, buf, etc_cpu, seed, tag):
    """逐行对照 host 参照（all_grads=None 时按 seed 重放全组）。"""
    epn = buf.epn
    if all_grads is None:
        all_grads = _randn_grads_set(seed, world, epn)
    for g in range(3):
        for i in range(epn):
            exp = _expected_home(all_grads, rank, world, epn, etc_cpu, g, i, _W)
            got = grads[g][:epn][i].cpu()
            _assert_all_ranks(
                torch.equal(got, exp.cpu()), f"{tag} 组{g} 行{i} 逐位不符"
            )
