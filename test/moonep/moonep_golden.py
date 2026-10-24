# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MoonEP golden 参考实现（纯 torch，单进程模拟整个 EP 组）。

语义契约（与具体放置算法无关）：
- dispatch：每个 rank 恰好接收 S*K 个真实 token（完美均衡），输出按专家分组；
- combine ：out[r, s] = sum_k w[r, s, k] * expert_fn(hidden[r, s])，恢复 token 主序。

golden 内置一个贪心冗余专家规划器，仅用于生成合法布局；与实现侧规划器允许不同
（实现侧另有"每个目的 rank 至多引入一个远程 home group"等工程约束，golden 不建模）。
NvS 的 padding 属于实现细节，golden 输出不 padded，NvS = S*K。
"""

from typing import NamedTuple

import torch


class RouteShape(NamedTuple):
    """路由形状：EP 组规模 + 单 rank 路由参数。"""

    world_size: int  # EP rank 数
    num_tokens: int  # 单 rank token 数
    top_k: int  # 每 token 路由专家数
    num_experts: int  # EP 组专家总数

    @property
    def epn(self) -> int:
        """每 rank 真实本地专家数。"""
        return self.num_experts // self.world_size


def _home(e, epn):
    return e // epn


def plan_redundant_experts(demand, num_experts, world_size):
    """按全局需求做贪心均衡放置。

    Args:
        demand: [num_experts] int64，每个专家的全局 token 数。
        num_experts: EP 组专家总数。
        world_size: EP rank 数。

    Returns:
        copies: dict e -> [rank, ...]，home 在前，冗余副本按加入顺序在后。
        rem: dict (e, rank) -> int，专家 e 放置在 rank 上的 token 数。
        cap: 每 rank 负载上限，等于 S*K。
    """
    epn = num_experts // world_size
    total = int(demand.sum())
    if total % world_size != 0:
        raise ValueError(f"total tokens {total} not divisible by R={world_size}")
    cap = total // world_size

    loads = [0] * world_size
    copies = {}
    rem = {}
    for e in range(num_experts):
        d = int(demand[e])
        if d == 0:
            continue
        h = _home(e, epn)
        copies[e] = [h]
        rem[(e, h)] = d
        loads[h] += d

    # 修正超载：把超载 rank 上剩余最多的专家的 token 搬到最空闲 rank（新建冗余副本）。
    # 每轮至少搬运 1 个 token，总量有限，必收敛；总和恒为 R*cap，无超载即全均衡。
    while max(loads) > cap:
        src = max(range(world_size), key=lambda r: loads[r])
        dst = min(range(world_size), key=lambda r: loads[r])
        excess = loads[src] - cap
        room = cap - loads[dst]
        if room <= 0:
            raise ValueError("infeasible: overloaded rank but no rank has room")
        e = max(
            (e for e in copies if rem.get((e, src), 0) > 0),
            key=lambda e: rem.get((e, src), 0),
        )
        move = min(excess, room, rem.get((e, src), 0))
        if dst not in copies[e]:
            copies[e].append(dst)
        rem[(e, src)] -= move
        rem[(e, dst)] = rem.get((e, dst), 0) + move
        loads[src] -= move
        loads[dst] += move

    if not all(load == cap for load in loads):
        raise ValueError(f"unbalanced loads: {loads}")
    return copies, rem, cap


def _take_slot(slots, idx):
    """推进到下一个有余量的槽位，返回新下标。"""
    while slots[idx][1] == 0:
        idx += 1
    return idx


def _assign_rank_tokens(expert, r, topk_all, slots, idx, assigned):
    """把 expert 在 rank r 上的全部 (s, k) 来源填入 slots，返回推进后的槽位下标。"""
    num_tokens, top_k = topk_all.shape[1], topk_all.shape[2]
    for s in range(num_tokens):
        for k in range(top_k):
            if int(topk_all[r, s, k]) == expert:
                idx = _take_slot(slots, idx)
                dst, left = slots[idx]
                assigned[(expert, dst)].append((r, s, k))
                slots[idx] = (dst, left - 1)
    return idx


def _assign_expert_tokens(expert, topk_all, slots, assigned):
    """把 expert 的全部 (src_rank, s, k) 来源按序填入 slots 指定的 rank。"""
    world_size = topk_all.shape[0]
    idx = 0
    for r in range(world_size):
        idx = _assign_rank_tokens(expert, r, topk_all, slots, idx, assigned)


def _assign_tokens(topk_all, copies, rem):
    """每个专家的 token 副本按 (src_rank, s, k) 顺序依次填入 copies[e] 各 rank，
    各 rank 可接收量由 rem[(e, rank)] 限定。返回 (e, rank) -> [(src_rank, s, k)]。
    """
    assigned = {(e, r): [] for (e, r) in rem}
    for e, homes in copies.items():
        slots = [(r, rem.get((e, r), 0)) for r in homes]
        _assign_expert_tokens(e, topk_all, slots, assigned)
    return assigned


def _stack_view(view_rows, hidden_all, route_weights_all, hidden_size):
    """把一个 rank 的视图行 (expert, entries) 列表堆叠为 hidden/rw/cu_seqlens/meta。"""
    rows, rw_rows, meta = [], [], []
    cu_seqlens = []
    for e, entries in view_rows:
        for src_r, s, k in entries:
            rows.append(hidden_all[src_r, s])
            rw_rows.append(route_weights_all[src_r, s, k])
            meta.append((src_r, s, k, e))
        cu_seqlens.append(len(rows))
    hidden = (
        torch.stack(rows)
        if rows
        else torch.empty(0, hidden_size, dtype=hidden_all.dtype)
    )
    rw = (
        torch.stack(rw_rows).to(torch.float32)
        if rw_rows
        else torch.empty(0, dtype=torch.float32)
    )
    return {
        "hidden": hidden,
        "rw": rw,
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int64),
        "meta": meta,
    }


def _rank_view_rows(rank, epn, copies, assigned):
    """单 rank 的视图行规划：本地专家升序，随后预取槽，空行补 None 至 2*epn。"""
    view_rows = []
    for e in range(rank * epn, (rank + 1) * epn):
        view_rows.append((e, assigned.get((e, rank), [])))
    prefetch = [e for e in copies if rank in copies[e][1:]]
    for e in prefetch:
        view_rows.append((e, assigned[(e, rank)]))
    while len(view_rows) < 2 * epn:
        view_rows.append((None, []))
    return view_rows


def moonep_dispatch_golden(hidden_all, route_weights_all, topk_all, num_experts):
    """dispatch golden：单进程模拟 R 个 rank 的分发结果。

    Args:
        hidden_all: [R, S, H] fp32/bf16，各 rank 输入 token。
        route_weights_all: [R, S, K] fp32，路由权重。
        topk_all: [R, S, K] int64/int32，top-k 专家索引（每行互不相同）。
        num_experts: EP 组专家总数。

    Returns:
        list[dict]，长度 R，每项含：
            hidden: [S*K, H] 按视图行（本地专家升序，随后预取槽）分组的 token；
            rw: [S*K] fp32，与 hidden 行对齐的路由权重；
            cu_seqlens: [2*epn] int64，各视图行的 token 结束偏移（空行重复上一值）；
            meta: list[(src_rank, s, k, e)]，每行的来源，供 combine golden 使用。
    """
    world_size, num_tokens, hidden_size = hidden_all.shape
    top_k = topk_all.shape[2]
    epn = num_experts // world_size

    demand = torch.zeros(num_experts, dtype=torch.int64)
    for r in range(world_size):
        demand += torch.bincount(
            topk_all[r].flatten().to(torch.int64), minlength=num_experts
        )

    copies, rem, cap = plan_redundant_experts(demand, num_experts, world_size)
    if cap != num_tokens * top_k:
        raise ValueError(f"cap={cap} != S*K={num_tokens * top_k}")

    assigned = _assign_tokens(topk_all, copies, rem)

    results = []
    for r in range(world_size):
        view_rows = _rank_view_rows(r, epn, copies, assigned)
        results.append(
            _stack_view(view_rows, hidden_all, route_weights_all, hidden_size)
        )
    return results


def _identity_expert(x, _e):
    """默认专家计算：identity（combine 纯加权归并）。"""
    return x


def moonep_combine_golden(dispatch_results, num_tokens, hidden_size, expert_fn=None):
    """combine golden：把 dispatch 结果按路由权重加权归并回 token 主序。

    Args:
        dispatch_results: moonep_dispatch_golden 的返回值。
        num_tokens/hidden_size: 形状参数（rank 数取自 dispatch_results 长度）。
        expert_fn: 专家计算，默认 identity；签名为 fn(hidden_row, expert_id) -> row。

    Returns:
        [R, S, H] fp32，out[r, s] = sum_k w[r, s, k] * expert_fn(hidden[r, s], e)。
    """
    if expert_fn is None:
        expert_fn = _identity_expert
    world_size = len(dispatch_results)
    out = torch.zeros(world_size, num_tokens, hidden_size, dtype=torch.float32)
    for r in range(world_size):
        res = dispatch_results[r]
        for row, (src_r, s, k, e) in enumerate(res["meta"]):
            out[src_r, s] += res["rw"][row] * expert_fn(
                res["hidden"][row].to(torch.float32), e
            )
    return out


def _gen_topk(shape, mode, gen):
    """按 mode 生成 [R, S, K] topk：extreme 全相同 / skewed 集中前 1/4 / balanced 均匀随机。"""
    world_size, num_tokens, top_k, num_experts = shape
    if mode == "extreme":
        return (
            torch.arange(top_k, dtype=torch.int64)
            .view(1, 1, top_k)
            .expand(world_size, num_tokens, top_k)
            .contiguous()
        )
    topk = torch.empty(world_size, num_tokens, top_k, dtype=torch.int64)
    for r in range(world_size):
        for s in range(num_tokens):
            if mode == "skewed":
                pool = max(top_k, num_experts // 4)
                perm = torch.randperm(pool, generator=gen)[:top_k]
            else:
                perm = torch.randperm(num_experts, generator=gen)[:top_k]
            topk[r, s] = perm
    return topk


def make_router_inputs(shape, mode="balanced", seed=42):
    """构造路由输入。shape 为 RouteShape；mode: balanced / skewed / extreme。"""
    if shape.top_k > shape.num_experts:
        raise ValueError(f"K={shape.top_k} > E={shape.num_experts}")
    gen = torch.Generator().manual_seed(seed)
    topk = _gen_topk(shape, mode, gen)
    rw = (
        torch.rand(
            shape.world_size,
            shape.num_tokens,
            shape.top_k,
            generator=gen,
            dtype=torch.float32,
        )
        + 0.01
    )
    return topk, rw
