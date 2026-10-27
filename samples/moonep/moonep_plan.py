# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""moonep_plan：MoonEP 规划算子。

输入路由计数与 topk，产出 plan 五件套（纯张量，跨调用、跨前后向复用）：
  dst_n[N]               【零占位】vtopk-only 快路径下不计算（原语义：目的地编码
                         rank*NvS+slot；恢复见 cached 路线的 kernel 内产出方案）
  cu_seqlens[2*epn]      本 rank 接收区各视图行累计结束偏移（本地专家升序 + 预取槽，空行重复前值）
  experts_to_copy[R,epn] 各 rank 承载的远程专家副本清单（-1 填充）
  src_n[N]               【零占位】vtopk-only 快路径下不计算（原语义：接收槽反查表）
  vtopk[N]               每个 (s,k) 的虚拟专家 id 投影（扩展空间 2E，供 ep dispatch 路由）

形状约束：E<=1024、R<=64、E%R==0；S*K 不设上限且任意 N 原生整除
（topk host 侧直拷 workspace；分块粒度降档到 N 的因子，无尾补机制）。

算法（与 test/moonep/moonep_golden.py 的规划器同语义）：
  整段（_MoonepPlanComputeKernel，一次 launch；阶段间 global_sync_all 栅栏）：
    阶段0  A Exchange（仅 block 0）：跨卡交换各 rank 的 tokens_per_expert 直方图
           （put + flag + 轮询）；本 rank topk 由 wrapper 直接以调用方张量的
           扁平视图喂给后续阶段（kernel 零 UB 中转）
    阶段1  B0 GreedyInit（bi<bdp，按专家切）：dem → rem/order 初态 + 各 block
           局部 loads 行  |  Cnt（bi<bd，按 chunk 切）：分块 per-expert token 数
           ——两者互不依赖，同阶段并行
    阶段2  B Greedy（仅 block 0）：loads 汇总 + 运行时 cap + 贪心修正 +
           cu_seqlens（含副本槽溢出写保护）
    阶段3  Tab（bi<R，每 block 一个 rank）：jrep 与 experts_to_copy（含溢出写保护）
    阶段4  Pfx（bi<bdp，按专家切）：pcnt chunk 前缀/dpre/ord 落位查表 +
           快路径 vid 表（无远程副本的专家 vid 恒定）
    阶段5  Asg（bi<bd，按 chunk 切）：段倒计时落位——快路径专家查 vid 表 O(1)，
           其余只在段边界重定位（pos 越过段尾才扫 ord 表），与逐 token 串行扫描
           语义逐位等价，产出 vtopk

Kernel split rationale（历史）：早期把通信（xfer）与重 vf 区直接混在一个
kernel 里会撞 DSL 编译器 SIGSEGV，所以曾经拆成 7 次 launch。现在的结构是
“xfer 只出现在阶段 0 的 block 0 上、且紧跟一个 global_sync_all 栅栏”，
实测可以编过并正确运行，plan 因此收敛到单次 launch。

约束备忘：通信只在单 block 内；不用 load_bypass；UB 标量点读写只在 vf(simt) 区域；
大循环用 cannbotdsl.range 动态循环；tile 写出用单行 tile + 运行时坐标
（多行 2D tile + 运行时首坐标会静默产出垃圾，DSL #60）。
"""

import threading
from typing import NamedTuple

import torch

try:
    import cannbotdsl
    from cannbotdsl import TensorSpec, dtypes
    from cannbotdsl.buffer import Buffer
    from cannbotdsl.lang.constexpr import const_expr, range_constexpr
    from cannbotdsl.lang.host import host
    from cannbotdsl.lang.jit import jit
    from cannbotdsl.lang.kernel import kernel
    from cannbotdsl.lang.vf import vf
    from cannbotdsl.ops.arch import get_block_idx
    from cannbotdsl.ops.distributed import (
        CommContext,
        SymBuffer,
        TransferRole,
        create_transfer_context,
    )
    from cannbotdsl.ops.memcpy import mem_copy
    from cannbotdsl.ops.sync import global_sync_all
    from cannbotdsl.tensor import MemLoc, tile_slice
    from cannbotdsl.types.datastruct import StructRef

    _HAS_DSL = True
except ImportError:
    # CPU 无 DSL 环境：仅本文件尾部 plan→EPHandle 翻译段（纯 torch）可用，
    # kernel/host 定义退化为占位（不会被调用）。
    _HAS_DSL = False

    def kernel(cls):
        return cls

    def host(fn):
        return fn

    def jit(fn):
        return fn

    class _StubStructRef:
        _sentinel = None

        def __getitem__(self, item):
            del item
            return self._sentinel

    StructRef = _StubStructRef()

    class _StubDtypes:  # host 函数签名注解（dtypes.int32）占位
        _sentinel = None

        def __getattr__(self, name):
            del name
            return self._sentinel

    dtypes = _StubDtypes()

    class CommContext:
        pass


from moonep_context import (
    DEM_ALL_OFF,
    MAX_E,
    MAX_R,
    READY_OFF,
    busy_wait_flag,
    check_no_shared_elastic_buffer,
    record_transport_event,
    resident_blocks,
    sync_transport_stream,
)

# 规划区在对称堆内的布局（heap_base=4096 起算的字节偏移，32B 对齐，常量见 moonep_context）
_EVENT_ID = 7
_EPOCH = 0  # 进程内单调调用序号（epoch flag）
_ME = 0  # 组内 rank（编译锁内写入，trace 期烘焙为常量）
_R = 1  # 组内 rank_size（同 shape 不同 rank 须进编译缓存键，见 _get_plan_program）

# 单次启动的 block 数经验上限（核数再多也不加块：块数收益递减）。
# 融合 kernel 用 global_sync_all 做栅栏，要求全部 block 同时常驻——实际块数
# 必须由 `resident_blocks()` 按本机有效 AIV 核数取 min，**不能硬编码 64**：
# 56 核机型上排 64 块会被调度器分时复用，栅栏永远收不齐而挂死。
_MAX_PLAN_BLOCKS = 64
# 分块粒度下限（token/块）：再小则每 chunk 的 O(E) 状态重建开销占比过高
_CHUNK_MIN = 128

# 贪心 argmax/argmin 初值（比任何可能的 token 数都大；int32 安全）
_GREEDY_NEG_INF = -(1 << 30)
_GREEDY_POS_INF = 1 << 30


# ---------------------------------------------------------------------------
# Kernel A：跨卡需求交换 + 全局汇总（单 block 通信）
# ---------------------------------------------------------------------------
@kernel
class _MoonepPlanComputeKernel:
    """plan 计算段单 kernel：阶段间用 global_sync_all() 做栅栏。

    原来 6 个 kernel（GreedyInit/Greedy/Tables/Count/Pfx/Assign）各一次 launch，
    下发成本按 “每次 ≈0.09ms + 0.028ms×参数个数” 线性累计；合并后只剩一次。

    阶段（块角色由 block_idx 决定；所有 block 都参与每个栅栏）：
      1. GreedyInit（bi<bdp，按专家切）  |  Count（bi<bd，按 chunk 切）——互不依赖
      2. Greedy（仅 block 0：串行贪心 + cu）
      3. Tables（bi<R，每 block 一个 rank） | Pfx（bi<bdp，按专家切）——同依赖
      4. Assign（bi<bd，按 chunk 切）
    """

    def __init__(
        self, rank, rank_size, num_expert, n_tokens, n_chunks, block_dim, init_blocks
    ):
        self._me = int(rank)
        self._r = int(rank_size)
        self._e = int(num_expert)
        self._epn = self._e // self._r
        self._chunk = int(n_tokens) // int(n_chunks)
        self._nc = int(n_chunks)
        self._bd = int(block_dim)
        self._bdp = int(init_blocks)
        self._ws = _ws_layout(self._r, self._e, self._nc, self._bdp)

    def __call__(
        self,
        gm_tpe,
        gm_flag,
        gm_dem_by_rank,
        gm_ws,
        gm_topk,
        gm_cu,
        gm_etc,
        gm_vtopk,
        ctx: StructRef[CommContext],
    ):
        r, e, nc, bdp = self._r, self._e, self._nc, self._bdp
        o = self._ws
        # 扁平 workspace 按编译期偏移切视图（10 个表 → program 1 个参数；
        # dem_by_rank 单独传：Exchange 的 peer 寻址需要原始指针）
        gm_rem = gm_ws[o["rem"] : o["rem"] + e * r,].view(e, r)
        gm_order = gm_ws[o["order"] : o["order"] + e * r,].view(e, r)
        gm_loads_part = gm_ws[o["loads_part"] : o["loads_part"] + bdp * r,].view(bdp, r)
        gm_counts = gm_ws[o["counts"] : o["counts"] + nc * e,].view(nc, e)
        gm_pcnt = gm_ws[o["pcnt"] : o["pcnt"] + e * nc,].view(e, nc)
        gm_dpre = gm_ws[o["dpre"] : o["dpre"] + e,]
        gm_vid = gm_ws[o["vid"] : o["vid"] + e,]
        gm_ord_rank = gm_ws[o["ord_rank"] : o["ord_rank"] + e * (r + 1),].view(e, r + 1)
        gm_ord_end = gm_ws[o["ord_end"] : o["ord_end"] + e * (r + 1),].view(e, r + 1)
        gm_jrep = gm_ws[o["jrep"] : o["jrep"] + r * e,].view(r, e)
        bi = get_block_idx()

        # ---- 阶段 0：跨卡需求交换（仅 block 0；put+flag+轮询） ----
        if bi == dtypes.int64(0):
            self._exchange(ctx, gm_tpe, gm_flag, gm_dem_by_rank)

        global_sync_all()

        # ---- 阶段 1：需求初始化（按专家） + 分块计数（按 chunk），互不依赖 ----
        if bi < dtypes.int64(self._bdp):
            self._greedy_init(bi, gm_dem_by_rank, gm_rem, gm_order, gm_loads_part)
        if bi < dtypes.int64(self._bd):
            self._count(bi, gm_topk, gm_counts)

        global_sync_all()

        # ---- 阶段 2：贪心修正 + cu（单 block 串行） ----
        if bi == dtypes.int64(0):
            self._greedy(gm_rem, gm_order, gm_loads_part, gm_cu)

        global_sync_all()

        # ---- 阶段 3：jrep/etc 表（每 block 一个 rank）∥ 前缀/查表（按专家） ----
        # 两者都只依赖阶段 2 的 order/counts，写不同输出，同阶段并行
        if bi < dtypes.int64(self._r):
            self._tables(bi, gm_order, gm_jrep, gm_etc)
        if bi < dtypes.int64(self._bdp):
            self._pfx(
                bi,
                gm_dem_by_rank,
                gm_counts,
                gm_rem,
                gm_order,
                gm_pcnt,
                gm_dpre,
                gm_vid,
                gm_ord_rank,
                gm_ord_end,
            )

        global_sync_all()

        # ---- 阶段 4：token 落位（按 chunk） ----
        if bi < dtypes.int64(self._bd):
            self._assign(
                bi,
                gm_topk,
                gm_pcnt,
                gm_dpre,
                gm_ord_rank,
                gm_ord_end,
                gm_jrep,
                gm_vid,
                gm_vtopk,
            )

    # ---------------- 阶段 0：Exchange（单 block 通信） ----------------
    @jit
    def _exchange(self, ctx, gm_tpe, gm_flag, gm_dem_by_rank):
        world_size, num_experts, me = self._r, self._e, self._me
        dem_all = SymBuffer(
            ctx,
            shape=(MAX_R, MAX_E),
            dtype=dtypes.int32,
            rank_size=world_size,
            base_offset=DEM_ALL_OFF,
        )
        ready = SymBuffer(
            ctx,
            shape=(MAX_R,),
            dtype=dtypes.int32,
            rank_size=world_size,
            base_offset=READY_OFF,
        )
        ub_xfer = Buffer(MemLoc.UB, (512,), dtypes.uint8)
        xfer = create_transfer_context(ctx, ub_xfer, role=TransferRole.VEC)

        for p in range_constexpr(world_size):
            if const_expr(p != me):
                xfer.put(
                    p, tile_slice(dem_all, (1, num_experts), (me, 0)).peer(p), gm_tpe
                )
        for p in range_constexpr(world_size):
            if const_expr(p != me):
                xfer.put(p, tile_slice(ready, (1,), (me,)).peer(p), gm_flag)
        for p in range_constexpr(world_size):
            if const_expr(p != me):
                xfer.flush(p)
        ub_tpe = Buffer(MemLoc.UB, (1, num_experts), dtypes.int32)
        mem_copy(ub_tpe, gm_tpe)
        mem_copy(tile_slice(dem_all, (1, num_experts), (me, 0)), ub_tpe)
        for r in range_constexpr(world_size):
            if const_expr(r != me):
                busy_wait_flag(tile_slice(ready, (1,), (r,)), gm_flag, _EVENT_ID)
        ub_dem = Buffer(MemLoc.UB, (world_size, num_experts), dtypes.int32)
        mem_copy(ub_dem, tile_slice(dem_all, (world_size, num_experts), (0, 0)))
        mem_copy(gm_dem_by_rank, ub_dem)

    # ---------------- 阶段 1a：GreedyInit ----------------
    @jit
    def _greedy_init(self, bi, gm_dem_by_rank, gm_rem, gm_order, gm_loads_part):
        world_size, num_experts = self._r, self._e
        e_per = num_experts // self._bdp
        e_lo = bi * e_per

        ub_dem = Buffer(MemLoc.UB, (world_size, num_experts), dtypes.int32)
        mem_copy(ub_dem, gm_dem_by_rank)
        ub_rem_row = Buffer(MemLoc.UB, (1, world_size), dtypes.int32)
        ub_order_row = Buffer(MemLoc.UB, (1, world_size), dtypes.int32)
        ub_part = Buffer(MemLoc.UB, (world_size,), dtypes.int32)

        with vf(mode="simt", thread=1):
            for r in range_constexpr(world_size):
                ub_part[dtypes.int64(r)] = dtypes.int32(0)
        for k in cannbotdsl.range(e_per):
            e = e_lo + k
            self._init_expert_row(e, ub_dem, ub_rem_row, ub_order_row, ub_part)
            mem_copy(tile_slice(gm_rem, (1, world_size), (e, 0)), ub_rem_row)
            mem_copy(tile_slice(gm_order, (1, world_size), (e, 0)), ub_order_row)
        mem_copy(tile_slice(gm_loads_part, (1, world_size), (bi, 0)), ub_part)

    @jit
    def _init_expert_row(self, e, ub_dem, ub_rem_row, ub_order_row, ub_part):
        """单专家：全局需求求和 → home 放置 + 负载累计。"""
        epn = self._epn
        with vf(mode="simt", thread=1):
            total = dtypes.int32(0)
            for r in range_constexpr(self._r):
                total = total + ub_dem[dtypes.int64(r), dtypes.int64(e)]
            h = e // epn
            for r in range_constexpr(self._r):
                ub_rem_row[dtypes.int64(0), dtypes.int64(r)] = dtypes.int32(0)
                ub_order_row[dtypes.int64(0), dtypes.int64(r)] = dtypes.int32(0)
            ub_rem_row[dtypes.int64(0), h] = total
            ub_order_row[dtypes.int64(0), h] = dtypes.int32(1)
            ub_part[dtypes.int64(h)] = ub_part[dtypes.int64(h)] + total

    # ---------------- 阶段 1b：Count ----------------
    @jit
    def _count(self, bi, gm_topk, gm_counts):
        num_experts, chunk, nc, bd = self._e, self._chunk, self._nc, self._bd
        ub_topk = Buffer(MemLoc.UB, (chunk,), dtypes.int32)
        ub_cnt = Buffer(MemLoc.UB, (num_experts,), dtypes.int32)

        for k in cannbotdsl.range(nc // bd):
            ci = bi + k * bd
            with vf(mode="simt", thread=1):
                for e in cannbotdsl.range(num_experts):
                    ub_cnt[dtypes.int64(e)] = dtypes.int32(0)
            mem_copy(ub_topk, tile_slice(gm_topk, (chunk,), (ci,)))
            with vf(mode="simt", thread=1):
                for i in cannbotdsl.range(chunk):
                    e = ub_topk[dtypes.int64(i)]
                    ub_cnt[e] = ub_cnt[e] + dtypes.int32(1)
            mem_copy(tile_slice(gm_counts, (1, num_experts), (ci, 0)), ub_cnt)

    # ---------------- 阶段 2：Greedy ----------------
    @jit
    def _greedy(self, gm_rem, gm_order, gm_loads_part, gm_cu):
        world_size, num_experts = self._r, self._e
        ub_rem = Buffer(MemLoc.UB, (num_experts, world_size), dtypes.int32)
        mem_copy(ub_rem, gm_rem)
        ub_order = Buffer(MemLoc.UB, (num_experts, world_size), dtypes.int32)
        mem_copy(ub_order, gm_order)
        ub_loads = Buffer(MemLoc.UB, (world_size,), dtypes.int32)
        ub_part_all = Buffer(MemLoc.UB, (self._bdp, world_size), dtypes.int32)
        mem_copy(ub_part_all, gm_loads_part)
        ub_sizes = Buffer(MemLoc.UB, (2 * self._epn,), dtypes.int32)
        ub_cu = Buffer(MemLoc.UB, (2 * self._epn,), dtypes.int32)

        with vf(mode="simt", thread=1):
            for r in range_constexpr(world_size):
                acc = dtypes.int32(0)
                for b in range_constexpr(self._bdp):
                    acc = acc + ub_part_all[dtypes.int64(b), dtypes.int64(r)]
                ub_loads[dtypes.int64(r)] = acc
            cap = dtypes.int32(0)
            for r in range_constexpr(world_size):
                cap = cap + ub_loads[dtypes.int64(r)]
            cap = cap // world_size
            for _it in cannbotdsl.range(num_experts):
                if self._loads_max(ub_loads) > cap:
                    self._greedy_round(ub_rem, ub_order, ub_loads, cap)
            self._fill_sizes_cu(ub_rem, ub_order, ub_sizes, ub_cu)
        mem_copy(gm_rem, ub_rem)
        mem_copy(gm_order, ub_order)
        mem_copy(gm_cu, ub_cu)

    @jit
    def _loads_max(self, ub_loads):
        max_load = dtypes.int32(_GREEDY_NEG_INF)
        for r in range_constexpr(self._r):
            if ub_loads[dtypes.int64(r)] > max_load:
                max_load = ub_loads[dtypes.int64(r)]
        return max_load

    @jit
    def _loads_argmax(self, ub_loads):
        src_r = dtypes.int32(0)
        best = dtypes.int32(_GREEDY_NEG_INF)
        for r in range_constexpr(self._r):
            if ub_loads[dtypes.int64(r)] > best:
                best = ub_loads[dtypes.int64(r)]
                src_r = dtypes.int32(r)
        return src_r

    @jit
    def _loads_argmin(self, ub_loads):
        dst_r = dtypes.int32(0)
        best = dtypes.int32(_GREEDY_POS_INF)
        for r in range_constexpr(self._r):
            if ub_loads[dtypes.int64(r)] < best:
                best = ub_loads[dtypes.int64(r)]
                dst_r = dtypes.int32(r)
        return dst_r

    @jit
    def _max_rem_expert(self, ub_rem, src_r):
        e_best = dtypes.int32(-1)
        rem_best = dtypes.int32(0)
        for e in cannbotdsl.range(self._e):
            rem_er = ub_rem[dtypes.int64(e), src_r]
            if rem_er > rem_best:
                rem_best = rem_er
                e_best = dtypes.int32(e)
        return e_best, rem_best

    @jit
    def _greedy_round(self, ub_rem, ub_order, ub_loads, cap):
        src_r = self._loads_argmax(ub_loads)
        dst_r = self._loads_argmin(ub_loads)
        e_best, rem_best = self._max_rem_expert(ub_rem, src_r)
        move = ub_loads[src_r] - cap
        room = cap - ub_loads[dst_r]
        if room < move:
            move = room
        if rem_best < move:
            move = rem_best
        if ub_order[e_best, dst_r] == dtypes.int32(0):
            n_copy = dtypes.int32(0)
            for r in range_constexpr(self._r):
                if ub_order[e_best, dtypes.int64(r)] > dtypes.int32(0):
                    n_copy = n_copy + dtypes.int32(1)
            ub_order[e_best, dst_r] = n_copy + dtypes.int32(1)
        ub_rem[e_best, src_r] = ub_rem[e_best, src_r] - move
        ub_rem[e_best, dst_r] = ub_rem[e_best, dst_r] + move
        ub_loads[src_r] = ub_loads[src_r] - move
        ub_loads[dst_r] = ub_loads[dst_r] + move

    @jit
    def _fill_sizes_cu(self, ub_rem, ub_order, ub_sizes, ub_cu):
        me = self._me
        epn = self._epn
        for i in range_constexpr(epn):
            ub_sizes[dtypes.int64(i)] = ub_rem[
                dtypes.int64(me * epn + i), dtypes.int64(me)
            ]
        for i in range_constexpr(epn):
            ub_sizes[dtypes.int64(epn + i)] = dtypes.int32(0)
        rep_cnt = dtypes.int32(0)
        for e in cannbotdsl.range(self._e):
            o = ub_order[dtypes.int64(e), dtypes.int64(me)]
            if o >= dtypes.int32(2):
                if rep_cnt < dtypes.int32(epn):
                    ub_sizes[dtypes.int32(epn) + rep_cnt] = ub_rem[
                        dtypes.int64(e), dtypes.int64(me)
                    ]
                rep_cnt = rep_cnt + dtypes.int32(1)
        acc = dtypes.int32(0)
        for i in range_constexpr(2 * epn):
            acc = acc + ub_sizes[dtypes.int64(i)]
            ub_cu[dtypes.int64(i)] = acc

    # ---------------- 阶段 3：Tables ----------------
    @jit
    def _tables(self, r, gm_order, gm_jrep, gm_etc):
        num_experts, epn = self._e, self._epn
        ub_order = Buffer(MemLoc.UB, (num_experts, self._r), dtypes.int32)
        mem_copy(ub_order, gm_order)
        ub_jrep_row = Buffer(MemLoc.UB, (1, num_experts), dtypes.int32)
        ub_etc_row = Buffer(MemLoc.UB, (1, epn), dtypes.int32)

        with vf(mode="simt", thread=1):
            self._fill_jrep_etc(ub_order, ub_jrep_row, ub_etc_row, r, epn)
        mem_copy(tile_slice(gm_jrep, (1, num_experts), (r, 0)), ub_jrep_row)
        mem_copy(tile_slice(gm_etc, (1, epn), (r, 0)), ub_etc_row)

    @jit
    def _fill_jrep_etc(self, ub_order, ub_jrep_row, ub_etc_row, r, epn):
        jc = dtypes.int32(0)
        for e in cannbotdsl.range(self._e):
            ub_jrep_row[dtypes.int64(0), dtypes.int64(e)] = jc
            if ub_order[dtypes.int64(e), dtypes.int64(r)] >= dtypes.int32(2):
                jc = jc + dtypes.int32(1)
        for j in range_constexpr(epn):
            ub_etc_row[dtypes.int64(0), dtypes.int64(j)] = dtypes.int32(-1)
        rc = dtypes.int32(0)
        for e in cannbotdsl.range(self._e):
            o = ub_order[dtypes.int64(e), dtypes.int64(r)]
            if o >= dtypes.int32(2):
                if rc < dtypes.int32(epn):
                    ub_etc_row[dtypes.int64(0), rc] = dtypes.int32(e)
                rc = rc + dtypes.int32(1)

    # ---------------- 阶段 4：Pfx ----------------
    @jit
    def _pfx(
        self,
        bi,
        gm_dem_by_rank,
        gm_counts,
        gm_rem,
        gm_order,
        gm_pcnt,
        gm_dpre,
        gm_vid,
        gm_ord_rank,
        gm_ord_end,
    ):
        world_size, num_experts = self._r, self._e
        nc = self._nc
        e_per = num_experts // self._bdp
        e_lo = bi * e_per

        ub_dem = Buffer(MemLoc.UB, (world_size, num_experts), dtypes.int32)
        mem_copy(ub_dem, gm_dem_by_rank)
        ub_counts = Buffer(MemLoc.UB, (nc, num_experts), dtypes.int32)
        mem_copy(ub_counts, gm_counts)
        ub_rem = Buffer(MemLoc.UB, (num_experts, world_size), dtypes.int32)
        mem_copy(ub_rem, gm_rem)
        ub_order = Buffer(MemLoc.UB, (num_experts, world_size), dtypes.int32)
        mem_copy(ub_order, gm_order)

        pcnt_row = Buffer(MemLoc.UB, (1, nc), dtypes.int32)
        dpre_row = Buffer(MemLoc.UB, (1,), dtypes.int32)
        vid_row = Buffer(MemLoc.UB, (1,), dtypes.int32)
        ord_rank_row = Buffer(MemLoc.UB, (1, world_size + 1), dtypes.int32)
        ord_end_row = Buffer(MemLoc.UB, (1, world_size + 1), dtypes.int32)

        for k in cannbotdsl.range(e_per):
            e = e_lo + k
            self._build_row(
                e,
                ub_dem,
                ub_counts,
                ub_rem,
                ub_order,
                pcnt_row,
                dpre_row,
                vid_row,
                ord_rank_row,
                ord_end_row,
            )
            mem_copy(tile_slice(gm_pcnt, (1, nc), (e, 0)), pcnt_row)
            mem_copy(tile_slice(gm_dpre, (1,), (e,)), dpre_row)
            mem_copy(tile_slice(gm_vid, (1,), (e,)), vid_row)
            mem_copy(tile_slice(gm_ord_rank, (1, world_size + 1), (e, 0)), ord_rank_row)
            mem_copy(tile_slice(gm_ord_end, (1, world_size + 1), (e, 0)), ord_end_row)

    @jit
    def _build_row(
        self,
        e,
        ub_dem,
        ub_counts,
        ub_rem,
        ub_order,
        pcnt_row,
        dpre_row,
        vid_row,
        ord_rank_row,
        ord_end_row,
    ):
        world_size = self._r
        me = self._me
        nc = self._nc
        epn = self._e // self._r
        with vf(mode="simt", thread=1):
            acc_c = dtypes.int32(0)
            for c in range_constexpr(nc):
                pcnt_row[dtypes.int64(0), dtypes.int64(c)] = acc_c
                acc_c = acc_c + ub_counts[dtypes.int64(c), dtypes.int64(e)]
            acc_p = dtypes.int32(0)
            for r in range_constexpr(world_size):
                if r < me:
                    acc_p = acc_p + ub_dem[dtypes.int64(r), dtypes.int64(e)]
            dpre_row[dtypes.int64(0)] = acc_p
            hh = e // epn
            fast = dtypes.int32(1)
            for r in range_constexpr(world_size):
                if dtypes.int64(r) != hh:
                    if ub_rem[dtypes.int64(e), dtypes.int64(r)] > dtypes.int32(0):
                        fast = dtypes.int32(0)
            if fast != dtypes.int32(0):
                vid_row[dtypes.int64(0)] = dtypes.int32(hh * (2 * epn) + (e % epn))
            else:
                vid_row[dtypes.int64(0)] = dtypes.int32(-1)
            for o in range_constexpr(world_size + 1):
                ord_rank_row[dtypes.int64(0), dtypes.int64(o)] = dtypes.int32(-1)
            for r in range_constexpr(world_size):
                o = ub_order[dtypes.int64(e), dtypes.int64(r)]
                if o >= dtypes.int32(1):
                    ord_rank_row[dtypes.int64(0), dtypes.int64(o - 1)] = dtypes.int32(r)
            acc_o = dtypes.int32(0)
            for o in range_constexpr(world_size + 1):
                rnk = ord_rank_row[dtypes.int64(0), dtypes.int64(o)]
                if rnk >= dtypes.int32(0):
                    acc_o = acc_o + ub_rem[dtypes.int64(e), dtypes.int64(rnk)]
                ord_end_row[dtypes.int64(0), dtypes.int64(o)] = acc_o

    # ---------------- 阶段 5：Assign ----------------
    @jit
    def _assign(
        self,
        bi,
        gm_topk,
        gm_pcnt,
        gm_dpre,
        gm_ord_rank,
        gm_ord_end,
        gm_jrep,
        gm_vid,
        gm_vtopk,
    ):
        world_size, num_experts = self._r, self._e
        nc, bd = self._nc, self._bd

        self._pcnt = Buffer(MemLoc.UB, (num_experts, nc), dtypes.int32)
        mem_copy(self._pcnt, gm_pcnt)
        self._dpre = Buffer(MemLoc.UB, (num_experts,), dtypes.int32)
        mem_copy(self._dpre, gm_dpre)
        self._ord_rank = Buffer(MemLoc.UB, (num_experts, world_size + 1), dtypes.int32)
        mem_copy(self._ord_rank, gm_ord_rank)
        self._ord_end = Buffer(MemLoc.UB, (num_experts, world_size + 1), dtypes.int32)
        mem_copy(self._ord_end, gm_ord_end)
        self._jrep = Buffer(MemLoc.UB, (world_size, num_experts), dtypes.int32)
        mem_copy(self._jrep, gm_jrep)
        self._vid = Buffer(MemLoc.UB, (num_experts,), dtypes.int32)
        mem_copy(self._vid, gm_vid)

        for k in cannbotdsl.range(nc // bd):
            self._assign_chunk(bi + k * bd, gm_topk, gm_vtopk)

    @jit
    def _assign_chunk(self, ci, gm_topk, gm_vtopk):
        num_experts = self._e
        chunk = self._chunk
        ub_topk = Buffer(MemLoc.UB, (chunk,), dtypes.int32)
        ub_vtopk = Buffer(MemLoc.UB, (chunk,), dtypes.int32)
        ub_pos = Buffer(MemLoc.UB, (num_experts,), dtypes.int32)
        ub_send = Buffer(MemLoc.UB, (num_experts,), dtypes.int32)
        ub_cvid = Buffer(MemLoc.UB, (num_experts,), dtypes.int32)

        with vf(mode="simt", thread=1):
            for e in cannbotdsl.range(num_experts):
                if self._vid[dtypes.int64(e)] < dtypes.int32(0):
                    ub_pos[dtypes.int64(e)] = (
                        self._dpre[dtypes.int64(e)]
                        + self._pcnt[dtypes.int64(e), dtypes.int64(ci)]
                    )
                    ub_send[dtypes.int64(e)] = dtypes.int32(0)
        mem_copy(ub_topk, tile_slice(gm_topk, (chunk,), (ci,)))
        with vf(mode="simt", thread=1):
            for i in cannbotdsl.range(chunk):
                e = ub_topk[dtypes.int64(i)]
                fv = self._vid[dtypes.int64(e)]
                if fv >= dtypes.int32(0):
                    ub_vtopk[dtypes.int64(i)] = fv
                else:
                    pos = ub_pos[dtypes.int64(e)]
                    if pos >= ub_send[dtypes.int64(e)]:
                        vid, seg_idx = self._relocate(e, pos)
                        ub_send[dtypes.int64(e)] = self._ord_end[
                            dtypes.int64(e), seg_idx
                        ]
                        ub_cvid[dtypes.int64(e)] = vid
                    ub_vtopk[dtypes.int64(i)] = ub_cvid[dtypes.int64(e)]
                    ub_pos[dtypes.int64(e)] = pos + dtypes.int32(1)
        mem_copy(tile_slice(gm_vtopk, (chunk,), (ci,)), ub_vtopk)

    @jit
    def _relocate(self, e, pos):
        world_size, epn = self._r, self._epn
        dst_rank = dtypes.int32(-1)
        ord_idx = dtypes.int32(1)
        for o in range_constexpr(world_size + 1):
            if (dst_rank < 0) and (
                pos < self._ord_end[dtypes.int64(e), dtypes.int64(o)]
            ):
                dst_rank = self._ord_rank[dtypes.int64(e), dtypes.int64(o)]
                ord_idx = o + 1
        seg_idx = ord_idx - dtypes.int32(1)
        vid = dst_rank * dtypes.int32(2 * epn) + (e % dtypes.int32(epn))
        if ord_idx >= dtypes.int32(2):
            vid = (
                dst_rank * dtypes.int32(2 * epn)
                + dtypes.int32(epn)
                + self._jrep[dtypes.int64(dst_rank), dtypes.int64(e)]
            )
        return vid, seg_idx


# ---------------------------------------------------------------------------
# host 入口
# ---------------------------------------------------------------------------


def _plan_chunking(num_rows):
    """分块：block 数（=nc=N/chunk）尽量逼近核数上限（块内 token 循环是串行
    串行依赖链，实测耗时随块数单调下降）；chunk 向上取 2 的幂后降档到整除因子
    （保证 nc*chunk==N，无尾块）；下限 _CHUNK_MIN 防状态重建占比过高。
    """
    cap = resident_blocks(_MAX_PLAN_BLOCKS)
    want = max(_CHUNK_MIN, -(-num_rows // cap))
    chunk = 1 << (want - 1).bit_length() if want > 1 else 1
    while chunk > 1 and num_rows % chunk != 0:
        chunk //= 2
    nc = num_rows // chunk
    bd = min(nc, cap)
    while bd > 1 and nc % bd != 0:
        bd -= 1
    return chunk, nc, bd


def _expert_blocks(num_experts):
    """GreedyInit/Pfx 的 block 数：E 的可整除因子，且不超过本机可常驻核数。"""
    bd = min(num_experts, resident_blocks(_MAX_PLAN_BLOCKS))
    while bd > 1 and num_experts % bd != 0:
        bd -= 1
    return bd


class _PlanBlocks(NamedTuple):
    """plan 分块全量（chunk/nc/bd 为 token 侧，bdp 为专家侧）。"""

    chunk: int
    nc: int
    bd: int
    bdp: int


def _plan_blocks(num_rows, num_experts):
    """分块与专家块一次算全（traced host / workspace 分配 / 编译锁内共用一份）。

    不做 lru_cache：块数回归测试 monkeypatch _RESIDENT_BLOCKS，缓存会把
    钉住的核数焊死。
    """
    chunk, nc, bd = _plan_chunking(num_rows)
    return _PlanBlocks(chunk, nc, bd, _expert_blocks(num_experts))


@host
def _moonep_plan_host(
    gm_tpe,
    gm_flag,
    gm_dem_by_rank,
    gm_ws,
    gm_topk,
    gm_cu,
    gm_etc,
    gm_vtopk,
    ctx: StructRef[CommContext],
):
    num_experts = gm_tpe.shape[0]
    world_size = _R  # trace 期烘焙的组内 rank/rank_size（traced 标量不可 int 转换）
    me = _ME
    num_rows = gm_topk.shape[0]  # 本 rank topk 扁平视图，直接作 Cnt/Asg 输入
    blocks = _plan_blocks(num_rows, num_experts)
    # plan 全部 7 个 kernel 合成 1 个 launch：阶段 0 跨卡交换（block 0）→
    # 栅栏 → 6 个纯计算阶段；workspace 打包后 program 参数 17→9。
    _MoonepPlanComputeKernel(
        me, world_size, num_experts, num_rows, blocks.nc, blocks.bd, blocks.bdp
    )[max(blocks.bd, blocks.bdp, world_size)](
        gm_tpe,
        gm_flag,
        gm_dem_by_rank,
        gm_ws,
        gm_topk,
        gm_cu,
        gm_etc,
        gm_vtopk,
        ctx,
    )


_PLAN_PROGRAMS = {}
_PLAN_LOCK = threading.Lock()

# 零占位张量（dst_n/src_n）进程级缓存：全零、只读契约、kernel 不写——
# 免每步 2 次 alloc+H2D；调用方不得写入（接口文档：零占位不计算）
_PLACEHOLDER_CACHE = {}


class _PlanDims(NamedTuple):
    """plan 形状键：单 rank token 数 / top-k / 专家总数 / EP rank 数。"""

    num_tokens: int
    top_k: int
    num_experts: int
    world_size: int


# workspace 打包：11 个中间表放进一个扁平 int32 张量，program 只收 1 个参数
# （host 下发成本按 “≈0.09ms + 0.024ms×参数个数” 线性计，17 参数≈0.45ms/次）。
# 每段按 16 元素（64B）对齐；各表由 kernel 全量覆写，同 shape 缓存复用安全。
_WS_ALIGN = 16


def _ws_layout(rank_size, num_experts, nc, bdp):
    """返回 {表名: 元素偏移} 与 total 元素数（编译期常量，host/kernel 共用）。"""
    r, e = rank_size, num_experts
    sizes = (
        ("rem", e * r),
        ("order", e * r),
        ("loads_part", bdp * r),
        ("counts", nc * e),
        ("pcnt", e * nc),
        ("dpre", e),
        ("vid", e),
        ("ord_rank", e * (r + 1)),
        ("ord_end", e * (r + 1)),
        ("jrep", r * e),
    )
    offs = {}
    cur = 0
    for name, n in sizes:
        cur = (cur + _WS_ALIGN - 1) // _WS_ALIGN * _WS_ALIGN
        offs[name] = cur
        cur += n
    offs["total"] = (cur + _WS_ALIGN - 1) // _WS_ALIGN * _WS_ALIGN
    return offs


def _zero_placeholder(num_rows):
    t = _PLACEHOLDER_CACHE.get(num_rows)
    if t is None:
        t = torch.zeros(num_rows, dtype=torch.int32, device="npu")
        _PLACEHOLDER_CACHE[num_rows] = t
    return t


def _alloc_plan_tables(dims, num_rows, nc, bdp):
    """plan workspace 分配：(dem_by_rank, 打包扁平张量)。

    dem_by_rank 单独一块：Exchange 的 peer 寻址要从张量取原始地址，
    interval slice 的 provenance 不支持 extract_pointer；其余 10 张表按
    _ws_layout 的编译期偏移放在同一块里，program 只收 1 个参数。
    """
    i32 = torch.int32
    total = _ws_layout(dims.world_size, dims.num_experts, nc, bdp)["total"]
    return (
        torch.zeros(dims.world_size, dims.num_experts, dtype=i32, device="npu"),
        torch.zeros(total, dtype=i32, device="npu"),
    )


def _plan_tables(context, dims, num_rows):
    """按 dims 单槽缓存的 plan workspace（各 kernel 全量覆写，复用安全）。"""
    ws = context.plan_workspace
    if ws is not None and ws[0] == dims:
        return ws[1]
    blocks = _plan_blocks(num_rows, dims.num_experts)
    tables = _alloc_plan_tables(dims, num_rows, blocks.nc, blocks.bdp)
    context.plan_workspace = (dims, tables)
    return tables


def _get_plan_program(num_tokens, top_k, num_experts, world_size, me):
    """按 (S,K,E,R,me) 键控的编译产物缓存（me 进键：rank 经模块级 _ME/_R 烘焙，
    同 shape 不同 rank 不能复用同一 program）。

    _ME/_R 的烘焙与编译在锁内原子完成（多线程并发首次编译不同
    EP 组时，烘焙窗口互污会产出错误 program）。
    """
    # 块数进键：bd/bdp 是 kernel 编译期常量（角色分派），核数不同的机器
    # 不能复用同一 program
    key = (
        num_tokens,
        top_k,
        num_experts,
        world_size,
        me,
        resident_blocks(_MAX_PLAN_BLOCKS),
    )
    with _PLAN_LOCK:
        prog = _PLAN_PROGRAMS.get(key)
        if prog is not None:
            return prog
        num_rows = num_tokens * top_k
        blocks = _plan_blocks(num_rows, num_experts)
        ws_total = _ws_layout(world_size, num_experts, blocks.nc, blocks.bdp)["total"]
        global _ME, _R
        _ME, _R = me, world_size
        epn = num_experts // world_size
        i32 = dtypes.int32
        prog = cannbotdsl.compile(
            _moonep_plan_host,
            TensorSpec((num_experts,), i32),  # tokens_per_expert
            TensorSpec((1,), i32),  # epoch flag
            TensorSpec((world_size, num_experts), i32),  # dem_by_rank
            TensorSpec((ws_total,), i32),  # 打包 workspace
            TensorSpec((num_rows,), i32),  # topk 扁平视图
            TensorSpec((2 * epn,), i32),  # cu_seqlens
            TensorSpec((world_size, epn), i32),  # experts_to_copy
            TensorSpec((num_rows,), i32),  # vtopk
            StructRef[CommContext],
        )
        _PLAN_PROGRAMS[key] = prog
        return prog


def _launch_plan(context, dims, ws, gm_flag, topk_flat, io_tensors):
    """编译产物 lookup + 七 kernel 一次 launch（io_tensors = tpe/cu/etc/vtopk）。

    topk_flat 直接是本 rank 的 [N] 视图（Count/Assign 只读本 rank 行），
    不再先拷进 workspace——那次 copy_ 在 NPU 上是 0.8ms 的慢路径。
    """
    tokens_per_expert, cu_seqlens, experts_to_copy, vtopk = io_tensors
    _get_plan_program(
        dims.num_tokens, dims.top_k, dims.num_experts, dims.world_size, context.rank
    )(
        tokens_per_expert,
        gm_flag,
        ws[0],  # dem_by_rank
        ws[1],  # 打包 workspace
        topk_flat,
        cu_seqlens,
        experts_to_copy,
        vtopk,
        context.ctx_tensor,
    )


def _stash_replica_overflow(context, cu_seqlens, num_rows):
    """暂存溢出检查载荷（plan 返回路径零同步）。

    尾部 D2H（.cpu()/.item()）会把 host 卡到流上全部在飞工作完成，阻塞
    紧随的 dispatch 发射（多层训练实测 941ms 级 host 阻塞）。
    不变量检查延迟到 prefetch/reduce 尾部执行（check_pending_overflow，
    sym_barrier 后微秒级）——无溢出路径语义不变。
    """
    context.pending_overflow = (cu_seqlens, num_rows)


def moonep_plan(context, tokens_per_expert, topk_experts_sk):
    """规划入口。返回 (dst_n, cu_seqlens, experts_to_copy, src_n, vtopk)。

    vtopk-only 快路径：C kernel 只产出 vtopk/cu_seqlens/experts_to_copy，
    dst_n/src_n 为零占位（ep dispatch 以 vtopk 直喂路由，无需二者）。

    vtopk: [N] int32，虚拟专家 id（扩展空间 E'=2E，home(v)=v//2epn 自动指向承载 rank），
    供 ops-transformer moe_ep 四件套路由改写使用。

    Args:
        context: MoonepContext（见 moonep_context.py）
        tokens_per_expert: [E] torch.int32 NPU 张量，本 rank 各专家 token 计数
        topk_experts_sk: [S, K] torch.int32 NPU 张量，本 rank top-k 专家索引

    返回路径零同步（尾部 D2H 会阻塞紧随的 dispatch 发射）；副本槽溢出
    不变量检测延迟到 prefetch/reduce 尾部执行（写前保护仍即时生效），
    无溢出路径语义不变：链路内算子未报错 ⟹ cu_seqlens[-1] == num_rows。
    """

    if not _HAS_DSL:
        raise ImportError(
            "moonep_plan kernel 需要 cannbotdsl（NPU 环境）；CPU 环境请仅运行 "
            "golden/纯 torch 测试"
        )

    if tokens_per_expert.dtype != torch.int32 or topk_experts_sk.dtype != torch.int32:
        raise TypeError(
            f"tpe/topk 须为 int32（bincount/topk 默认 int64，需 .to(torch.int32)）："
            f"got {tokens_per_expert.dtype} / {topk_experts_sk.dtype}"
        )
    if not context.eb_scan_done:  # 通信域分离契约复查：覆盖 ctx 先建、EB 后建
        check_no_shared_elastic_buffer(context.group_name)
        context.eb_scan_done = True
    sync_transport_stream(context)  # 跨流护栏（握手状态机要求调用按序）
    world_size = context.rank_size
    # 阶段 3 的 Tables 每个 block 处理一个 rank：组规模不能超过可常驻块数，
    # 否则 global_sync_all 栅栏收不齐（先报错，别等到挂死）
    nblk = resident_blocks(_MAX_PLAN_BLOCKS)
    if world_size > nblk:
        raise RuntimeError(
            f"EP 组规模 {world_size} 超过本机可常驻 AIV block 数 {nblk}："
            "plan 用 global_sync_all 做阶段栅栏，要求全部 block 同时常驻"
        )
    num_tokens, top_k = topk_experts_sk.shape
    num_experts = tokens_per_expert.shape[0]
    if num_experts % world_size != 0:
        raise ValueError(f"E={num_experts} 不能被 R={world_size} 整除")
    if num_experts > MAX_E:
        raise ValueError(f"E={num_experts} 超过 v1 上限 MAX_E={MAX_E}")
    epn = num_experts // world_size
    num_rows = num_tokens * top_k  # 任意 N 原生整除（分块降档，无尾补机制）
    dims = _PlanDims(num_tokens, top_k, num_experts, world_size)

    dst_n = _zero_placeholder(num_rows)
    cu_seqlens = torch.zeros(2 * epn, dtype=torch.int32, device="npu")
    experts_to_copy = torch.zeros(world_size, epn, dtype=torch.int32, device="npu")
    src_n = _zero_placeholder(num_rows)
    vtopk = torch.zeros(num_rows, dtype=torch.int32, device="npu")
    global _EPOCH
    _EPOCH += 1
    # _ME/_R 的烘焙挪入 _get_plan_program 锁内（与编译原子，防多线程
    # 首次编译窗口互污）；wrapper 只管 epoch
    gm_flag = torch.full(
        (1,), _EPOCH, dtype=torch.int32, device="npu"
    )  # epoch 单调 flag
    ws = _plan_tables(context, dims, num_rows)
    # Cnt/Asg 只消费本 rank 的行：直接把 [S,K] 展平视图当输入，省掉一次
    # 往 workspace 的整行 copy_（NPU 上该 copy_ 实测是 ~0.8ms 的慢路径）
    _launch_plan(
        context,
        dims,
        ws,
        gm_flag,
        topk_experts_sk.reshape(-1),
        (tokens_per_expert, cu_seqlens, experts_to_copy, vtopk),
    )
    record_transport_event(context)  # 跨流护栏 fork 点（只等我方算子，不等主流尾部）
    # 溢出检查延迟执行（返回路径零同步，不阻塞 dispatch 发射）
    _stash_replica_overflow(context, cu_seqlens, num_rows)
    context.last_plan_shape = (num_tokens, top_k)  # 供 prefetch/reduce 的调用顺序守卫
    return dst_n, cu_seqlens, experts_to_copy, src_n, vtopk
