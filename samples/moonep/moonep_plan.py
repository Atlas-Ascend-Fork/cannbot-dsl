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
  A（_MoonepPlanExchangeKernel，单 block 通信）：跨卡交换各 rank 的 tokens_per_expert
      直方图（put + flag + 轮询）；本 rank topk 行由 wrapper 在启动前 host 侧
      直拷 workspace（kernel 零 UB 中转）
  B0（_MoonepPlanGreedyInitKernel，多 block 按专家切）：dem → rem/order 初态 +
      各 block 局部 loads 行（免原子：独立行，B 汇总）
  B（_MoonepPlanGreedyKernel，单 block）：loads 汇总 + 运行时 cap + 贪心修正 +
      cu_seqlens（含副本槽溢出写保护）
  Tab（_MoonepPlanTablesKernel，多 block 每块一个 rank）：jrep（布局 (R,E)，每块写连续行）
      与 experts_to_copy（含溢出写保护）
  Cnt（_MoonepPlanCountKernel，多 block）：分块统计本 rank 各 chunk 的 per-expert token 数
  Pfx（_MoonepPlanPfxKernel，多 block 按专家切）：pcnt chunk 前缀/dpre/ord 落位查表 +
      快路径 vid 表（无远程副本的专家 vid 恒定）
  Asg（_MoonepPlanAssignKernel，多 block）：段倒计时落位——快路径专家查 vid 表 O(1)，
      其余只在段边界重定位（pos 越过段尾才扫 ord 表），与逐 token 串行扫描
      语义逐位等价，产出 vtopk

Kernel split rationale: communication primitives mixed with complex scalar
control flow / vf regions in one kernel exceed DSL compiler limits; pure
compute kernels (GreedyInit/Greedy/Tables/Cnt/Pfx/Asg) have no such issue.

约束备忘：通信只在单 block 内；不用 load_bypass；UB 标量点读写只在 vf(simt) 区域；
大循环用 cannbotdsl.range 动态循环；tile 写出用单行 tile + 运行时坐标
（多行 2D tile + 运行时首坐标会静默产出垃圾数据，编译器已知限制）。
"""

import threading
from typing import NamedTuple

import torch

try:
    import cannbotdsl
    from cannbotdsl import dtypes
    from cannbotdsl import TensorSpec
    from cannbotdsl.types.datastruct import StructRef
    from cannbotdsl.buffer import Buffer
    from cannbotdsl.tensor import MemLoc, tile_slice
    from cannbotdsl.lang.constexpr import const_expr, range_constexpr
    from cannbotdsl.lang.host import host
    from cannbotdsl.lang.jit import jit
    from cannbotdsl.lang.kernel import kernel
    from cannbotdsl.lang.vf import vf
    from cannbotdsl.ops.memcpy import mem_copy
    from cannbotdsl.ops.arch import get_block_idx
    from cannbotdsl.ops.distributed import (
        CommContext,
        create_transfer_context,
        SymBuffer,
        TransferRole,
    )

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
    MAX_E,
    MAX_R,
    DEM_ALL_OFF,
    READY_OFF,
    busy_wait_flag,
    check_no_shared_elastic_buffer,
    sync_transport_stream,
)

# 规划区在对称堆内的布局（heap_base=4096 起算的字节偏移，32B 对齐，常量见 moonep_context）
_EVENT_ID = 7
_EPOCH = 0  # 进程内单调调用序号（epoch flag）
_ME = 0  # 组内 rank（编译锁内写入，trace 期烘焙为常量）
_R = 1  # 组内 rank_size（同 shape 不同 rank 须进编译缓存键，见 _get_plan_program）

# 单次启动的 block 数上限：设备 28 核，取 32 让核数以内一轮跑完并留冗余档
_MAX_PLAN_BLOCKS = 32
# 分块粒度下限（token/块）：再小则每 chunk 的 O(E) 状态重建开销占比过高
_CHUNK_MIN = 128

# 贪心 argmax/argmin 初值（比任何可能的 token 数都大；int32 安全）
_GREEDY_NEG_INF = -(1 << 30)
_GREEDY_POS_INF = 1 << 30


# ---------------------------------------------------------------------------
# Kernel A：跨卡需求交换 + 全局汇总（单 block 通信）
# ---------------------------------------------------------------------------
@kernel
class _MoonepPlanExchangeKernel:
    def __init__(self, rank, rank_size, num_expert):
        self._me = int(rank)
        self._r = int(rank_size)
        self._e = int(num_expert)

    def __call__(
        self,
        gm_tpe,
        gm_flag,
        gm_dem_by_rank,
        ctx: StructRef[CommContext],
    ):
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

        # vtopk-only：只需交换 tpe 直方图（R×E，8KB 级）——Cnt/Asg 仅读本
        # rank 的 topk 行（gm_topk_all[me]），peers 的 topk 行无人消费，原
        # 1MB/rank 的 topk 行交换与逐行拷出为死代码（对照 AscendC 版规划
        # kernel 的交换语义确认后删除）
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
        # 本 rank 的 topk 行不落 gm_topk_all——改由 wrapper 在 program 启动前
        # host 侧 copy_（topk_all 为本地 workspace，仅 [me] 行被 Cnt/Asg 消费；
        # 原 kernel 内整行 UB 中转在 N=65536 时超 UB 256KB 上限）
        # 轮询全部 peer 的 tpe 到达（flag 语义：该 peer 的 tpe 已就位）
        for r in range_constexpr(world_size):
            if const_expr(r != me):
                busy_wait_flag(tile_slice(ready, (1,), (r,)), gm_flag, _EVENT_ID)
        # dem 表拷出（全 R 行，须在全部 poll 之后）
        ub_dem = Buffer(MemLoc.UB, (world_size, num_experts), dtypes.int32)
        mem_copy(ub_dem, tile_slice(dem_all, (world_size, num_experts), (0, 0)))
        mem_copy(gm_dem_by_rank, ub_dem)


# ---------------------------------------------------------------------------
# Kernel B0：贪心初始化（多 block，按专家切）
#   dem → rem/order 初态 + 各 block 局部 loads 行（独立行写出，免原子免清零，
#   由 B 汇总）；整表载入 + 单行 tile 逐行写回（多行 2D tile 运行时坐标不可靠）
# ---------------------------------------------------------------------------
@kernel
class _MoonepPlanGreedyInitKernel:
    def __init__(self, rank_size, num_expert, block_dim):
        self._r = int(rank_size)
        self._e = int(num_expert)
        self._bd = int(block_dim)
        self._epn = self._e // self._r

    def __call__(self, gm_dem_by_rank, gm_rem, gm_order, gm_loads_part):
        world_size, num_experts, bd = self._r, self._e, self._bd
        bi = get_block_idx()
        e_per = num_experts // bd
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
            self._init_and_store_row(
                e_lo + k, ub_dem, gm_rem, gm_order, ub_rem_row, ub_order_row, ub_part
            )
        # 本 block 的局部负载写独立行（Greedy 汇总；免原子 DMA、免启动前清零）
        mem_copy(tile_slice(gm_loads_part, (1, world_size), (bi, 0)), ub_part)

    @jit
    def _init_and_store_row(
        self, e, ub_dem, gm_rem, gm_order, ub_rem_row, ub_order_row, ub_part
    ):
        """单专家：初态构建 + 单行 tile 写回（多行 2D tile 运行时坐标不可靠）。"""
        world_size = self._r
        self._init_expert_row(e, ub_dem, ub_rem_row, ub_order_row, ub_part)
        mem_copy(tile_slice(gm_rem, (1, world_size), (e, 0)), ub_rem_row)
        mem_copy(tile_slice(gm_order, (1, world_size), (e, 0)), ub_order_row)

    @jit
    def _init_expert_row(self, e, ub_dem, ub_rem_row, ub_order_row, ub_part):
        """单专家：全局需求求和 → home 放置 + 负载累计。

        非 home 单元必须显式清零：B2 的副本累加（rem[e][dst] += move）会读到
        未初始化 UB 的垃圾，且全表写回会把垃圾带进 GM。
        """
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


# ---------------------------------------------------------------------------
# Kernel B：贪心修正 + cu_seqlens（纯计算，单 block）
#   输出 rem[E,R] / order[E,R]（order：0 不承载；1 home；2+ 第几副本）+ cu
# ---------------------------------------------------------------------------
@kernel
class _MoonepPlanGreedyKernel:
    def __init__(self, rank, rank_size, num_expert, n_init_blocks):
        self._me = int(rank)
        self._r = int(rank_size)
        self._e = int(num_expert)
        self._epn = self._e // self._r
        self._nib = int(n_init_blocks)

    def __call__(self, gm_rem, gm_order, gm_loads_part, gm_cu):
        world_size, num_experts = self._r, self._e
        ub_rem = Buffer(MemLoc.UB, (num_experts, world_size), dtypes.int32)
        mem_copy(ub_rem, gm_rem)
        ub_order = Buffer(MemLoc.UB, (num_experts, world_size), dtypes.int32)
        mem_copy(ub_order, gm_order)
        ub_loads = Buffer(MemLoc.UB, (world_size,), dtypes.int32)
        # 各 init block 的局部 loads 一次整表载入后汇总（constexpr 循环内含
        # mem_copy 会撞 DSL 编译器复杂度上限——改单次整表搬运 + vf 内累加）
        ub_part_all = Buffer(MemLoc.UB, (self._nib, world_size), dtypes.int32)
        mem_copy(ub_part_all, gm_loads_part)
        ub_sizes = Buffer(MemLoc.UB, (2 * self._epn,), dtypes.int32)
        ub_cu = Buffer(MemLoc.UB, (2 * self._epn,), dtypes.int32)

        with vf(mode="simt", thread=1):
            # 各 init block 的局部 loads 汇总（在同一 vf 区内完成——拆成独立
            # vf 区与 B2 串列会撞编译器；本 kernel 全程单 vf 区，同旧版结构）
            for r in range_constexpr(world_size):
                acc = dtypes.int32(0)
                for b in range_constexpr(self._nib):
                    acc = acc + ub_part_all[dtypes.int64(b), dtypes.int64(r)]
                ub_loads[dtypes.int64(r)] = acc
            # 容量上限 cap = total/R：运行时从 dem 求和推导（与 golden 同语义；
            # 烘焙 N 会被 host 侧尾补污染——pad 槽不进 tpe，dem 总和=真实需求）
            cap = dtypes.int32(0)
            for r in range_constexpr(world_size):
                cap = cap + ub_loads[dtypes.int64(r)]
            cap = cap // world_size
            # B2 贪心修正（bounded for + guard；每轮至少修掉 1 个超载 token，轮数 < E 必收敛）
            for _it in cannbotdsl.range(num_experts):
                if self._loads_max(ub_loads) > cap:
                    self._greedy_round(ub_rem, ub_order, ub_loads, cap)
            self._fill_sizes_cu(ub_rem, ub_order, ub_sizes, ub_cu)
        mem_copy(gm_rem, ub_rem)
        mem_copy(gm_order, ub_order)
        mem_copy(gm_cu, ub_cu)

    @jit
    def _loads_max(self, ub_loads):
        """当前各 rank 负载的最大值。"""
        max_load = dtypes.int32(_GREEDY_NEG_INF)
        for r in range_constexpr(self._r):
            if ub_loads[dtypes.int64(r)] > max_load:
                max_load = ub_loads[dtypes.int64(r)]
        return max_load

    @jit
    def _loads_argmax(self, ub_loads):
        """负载最高的 rank 下标。"""
        src_r = dtypes.int32(0)
        best = dtypes.int32(_GREEDY_NEG_INF)
        for r in range_constexpr(self._r):
            if ub_loads[dtypes.int64(r)] > best:
                best = ub_loads[dtypes.int64(r)]
                src_r = dtypes.int32(r)
        return src_r

    @jit
    def _loads_argmin(self, ub_loads):
        """负载最低的 rank 下标。"""
        dst_r = dtypes.int32(0)
        best = dtypes.int32(_GREEDY_POS_INF)
        for r in range_constexpr(self._r):
            if ub_loads[dtypes.int64(r)] < best:
                best = ub_loads[dtypes.int64(r)]
                dst_r = dtypes.int32(r)
        return dst_r

    @jit
    def _max_rem_expert(self, ub_rem, src_r):
        """src_r 上剩余 token 最多的专家 (e_best, rem_best)。"""
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
        """单轮贪心修正：最超载 rank 上剩余最多的专家搬 token 到最空闲 rank。"""
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
        """本 rank 各视图行 token 数（本地专家 + 副本槽）与 cu 前缀和。"""
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
                # 写前保护：副本数超 epn 槽时不落盘（防 UB 越界写），
                # cu 尾值随之 < N——wrapper 以完美均衡不变量检测并报错
                if rep_cnt < dtypes.int32(epn):
                    ub_sizes[dtypes.int32(epn) + rep_cnt] = ub_rem[
                        dtypes.int64(e), dtypes.int64(me)
                    ]
                rep_cnt = rep_cnt + dtypes.int32(1)
        acc = dtypes.int32(0)
        for i in range_constexpr(2 * epn):
            acc = acc + ub_sizes[dtypes.int64(i)]
            ub_cu[dtypes.int64(i)] = acc


# ---------------------------------------------------------------------------
# Kernel Tables：jrep / etc（多 block，每块一个 rank；各 rank 的扫描互相独立）
#   jrep 布局 (R,E)：每块只写自己 rank 的连续一行；etc 含超槽写前保护
# ---------------------------------------------------------------------------
@kernel
class _MoonepPlanTablesKernel:
    def __init__(self, rank_size, num_expert):
        self._r = int(rank_size)
        self._e = int(num_expert)
        self._epn = self._e // self._r

    def __call__(self, gm_order, gm_jrep, gm_etc):
        num_experts, epn = self._e, self._epn
        r = get_block_idx()  # 块号即 rank
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
        """jrep[e]：专家 e 在 rank r 上是第几副本（按 e 升序累计 order>=2）；
        etc 行：本 rank 承载的副本专家清单（-1 填充，超槽写前保护）。
        """
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
                if rc < dtypes.int32(epn):  # 写前保护（超槽不落盘，wrapper 检测并报错）
                    ub_etc_row[dtypes.int64(0), rc] = dtypes.int32(e)
                rc = rc + dtypes.int32(1)


# ---------------------------------------------------------------------------
# Kernel Cnt：分块计数（多 block，纯计算）
#   每 block 以步长 bd 认领 chunk，统计各 chunk 的 per-expert token 数 → counts[nc,E]
#   （供 Pfx 的 pcnt 前缀与 Assign 的跨块全局序）
# ---------------------------------------------------------------------------
@kernel
class _MoonepPlanCountKernel:
    def __init__(self, rank, num_expert, n_tokens, n_chunks, block_dim):
        self._me = int(rank)
        self._e = int(num_expert)
        self._chunk = int(n_tokens) // int(n_chunks)
        self._nc = int(n_chunks)
        self._bd = int(block_dim)

    def __call__(self, gm_topk_all, gm_counts):
        num_experts, me, chunk, nc, bd = (
            self._e,
            self._me,
            self._chunk,
            self._nc,
            self._bd,
        )
        bi = get_block_idx()
        ub_topk = Buffer(MemLoc.UB, (chunk,), dtypes.int32)
        ub_cnt = Buffer(MemLoc.UB, (num_experts,), dtypes.int32)

        # nc % bd == 0（host 保证），每 block 恰有 nc//bd 个 chunk，无需动态 guard
        for k in cannbotdsl.range(nc // bd):
            ci = bi + k * bd  # 运行时 tile 坐标（多 block 探针已验证）
            with vf(mode="simt", thread=1):
                for e in cannbotdsl.range(num_experts):
                    ub_cnt[dtypes.int64(e)] = dtypes.int32(0)
            mem_copy(ub_topk, tile_slice(gm_topk_all, (1, chunk), (me, ci)))
            with vf(mode="simt", thread=1):
                for i in cannbotdsl.range(chunk):
                    e = ub_topk[dtypes.int64(i)]
                    ub_cnt[e] = ub_cnt[e] + dtypes.int32(1)
            mem_copy(tile_slice(gm_counts, (1, num_experts), (ci, 0)), ub_cnt)


# ---------------------------------------------------------------------------
# Kernel Pfx（多 block，按专家切）：pcnt[E,nc] + dpre[E] + 快路径 vid[E] + ord 段表
# ---------------------------------------------------------------------------
@kernel
class _MoonepPlanPfxKernel:
    """多 block：按 e 切片并行构建 pcnt/dpre/ord 查表 + 快路径 vid。

    动态行循环（cannbotdsl.range）+ 单行 tile 写出——Cnt/Asg 同款已验证模式；
    pcnt 布局 [E, nc]（每 e 行 = 该专家在各 chunk 的起始全局序，行写出连续）。
    vid[e] >= 0 表示该专家无远程副本（vid 恒定，Asg 快路径 O(1) 查表）。
    """

    def __init__(self, rank, rank_size, num_expert, n_chunks, block_dim):
        self._me = int(rank)
        self._r = int(rank_size)
        self._e = int(num_expert)
        self._nc = int(n_chunks)
        self._bd = int(block_dim)
        self._pcnt_row = None  # __call__ 内分配（DSL trace 边界）
        self._dpre_row = None  # __call__ 内分配（DSL trace 边界）
        self._vid_row = None  # __call__ 内分配（DSL trace 边界）
        self._ord_rank_row = None  # __call__ 内分配（DSL trace 边界）
        self._ord_end_row = None  # __call__ 内分配（DSL trace 边界）

    def __call__(
        self,
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
        bi = get_block_idx()
        e_per = num_experts // self._bd
        e_lo = bi * e_per

        ub_dem = Buffer(MemLoc.UB, (world_size, num_experts), dtypes.int32)
        mem_copy(ub_dem, gm_dem_by_rank)
        ub_counts = Buffer(MemLoc.UB, (nc, num_experts), dtypes.int32)
        mem_copy(ub_counts, gm_counts)
        ub_rem = Buffer(MemLoc.UB, (num_experts, world_size), dtypes.int32)
        mem_copy(ub_rem, gm_rem)
        ub_order = Buffer(MemLoc.UB, (num_experts, world_size), dtypes.int32)
        mem_copy(ub_order, gm_order)

        self._pcnt_row = Buffer(MemLoc.UB, (1, nc), dtypes.int32)
        self._dpre_row = Buffer(MemLoc.UB, (1,), dtypes.int32)
        self._vid_row = Buffer(MemLoc.UB, (1,), dtypes.int32)
        self._ord_rank_row = Buffer(MemLoc.UB, (1, world_size + 1), dtypes.int32)
        self._ord_end_row = Buffer(MemLoc.UB, (1, world_size + 1), dtypes.int32)

        for k in cannbotdsl.range(e_per):
            e = e_lo + k
            self._build_row(e, ub_dem, ub_counts, ub_rem, ub_order)
            mem_copy(tile_slice(gm_pcnt, (1, nc), (e, 0)), self._pcnt_row)
            mem_copy(tile_slice(gm_dpre, (1,), (e,)), self._dpre_row)
            mem_copy(tile_slice(gm_vid, (1,), (e,)), self._vid_row)
            mem_copy(
                tile_slice(gm_ord_rank, (1, world_size + 1), (e, 0)), self._ord_rank_row
            )
            mem_copy(
                tile_slice(gm_ord_end, (1, world_size + 1), (e, 0)), self._ord_end_row
            )

    @jit
    def _fast_vid(self, e, hh, ub_rem):
        """快路径 vid：无远程副本（仅 home 承载）的专家 vid 恒定。"""
        world_size = self._r
        epn = self._e // self._r
        fast = dtypes.int32(1)
        for r in range_constexpr(world_size):
            if dtypes.int64(r) != hh:
                if ub_rem[dtypes.int64(e), dtypes.int64(r)] > dtypes.int32(0):
                    fast = dtypes.int32(0)
        vid = dtypes.int32(-1)
        if fast != dtypes.int32(0):
            vid = dtypes.int32(hh * (2 * epn) + (e % dtypes.int32(epn)))
        return vid

    @jit
    def _build_row(self, e, ub_dem, ub_counts, ub_rem, ub_order):
        """单专家的 pcnt 前缀 / dpre / 快路径 vid / ord 查表行构建（vf simt 单线程）。"""
        world_size = self._r
        me = self._me
        nc = self._nc
        epn = self._e // self._r
        with vf(mode="simt", thread=1):
            # pcnt 行：chunk c 前累计
            acc_c = dtypes.int32(0)
            for c in range_constexpr(nc):
                self._pcnt_row[dtypes.int64(0), dtypes.int64(c)] = acc_c
                acc_c = acc_c + ub_counts[dtypes.int64(c), dtypes.int64(e)]
            # dem 前缀
            acc_p = dtypes.int32(0)
            for r in range_constexpr(world_size):
                if r < me:
                    acc_p = acc_p + ub_dem[dtypes.int64(r), dtypes.int64(e)]
            self._dpre_row[dtypes.int64(0)] = acc_p
            hh = e // epn
            self._vid_row[dtypes.int64(0)] = self._fast_vid(e, hh, ub_rem)
            # order → rank/结束边界
            for o in range_constexpr(world_size + 1):
                self._ord_rank_row[dtypes.int64(0), dtypes.int64(o)] = dtypes.int32(-1)
            for r in range_constexpr(world_size):
                o = ub_order[dtypes.int64(e), dtypes.int64(r)]
                if o >= dtypes.int32(1):
                    self._ord_rank_row[dtypes.int64(0), dtypes.int64(o - 1)] = (
                        dtypes.int32(r)
                    )
            acc_o = dtypes.int32(0)
            for o in range_constexpr(world_size + 1):
                rnk = self._ord_rank_row[dtypes.int64(0), dtypes.int64(o)]
                if rnk >= dtypes.int32(0):
                    acc_o = acc_o + ub_rem[dtypes.int64(e), dtypes.int64(rnk)]
                self._ord_end_row[dtypes.int64(0), dtypes.int64(o)] = acc_o


# ---------------------------------------------------------------------------
# Kernel Asg：token 落位（多 block，段倒计时）
#   每 block 以步长 bd 认领 chunk；快路径专家（vid>=0）O(1) 查表，其余专家
#   维护 (pos, 段尾, 当前段 vid) 状态，只在 pos 越过段尾时重定位扫 ord 表——
#   与逐 token「pos = dpre + pcnt[chunk] + 局部计数」串行扫描语义逐位等价。
#   段初态按 chunk 惰性重建（段尾置 0 → 该专家在本 chunk 的首个 token 补全）。
# ---------------------------------------------------------------------------
@kernel
class _MoonepPlanAssignKernel:
    def __init__(self, rank, rank_size, num_expert, n_tokens, n_chunks, block_dim):
        self._me = int(rank)
        self._r = int(rank_size)
        self._e = int(num_expert)
        self._epn = self._e // self._r
        self._chunk = int(n_tokens) // int(n_chunks)
        self._nc = int(n_chunks)
        self._bd = int(block_dim)
        self._pcnt = None  # __call__ 内分配（DSL trace 边界）
        self._dpre = None  # __call__ 内分配（DSL trace 边界）
        self._ord_rank = None  # __call__ 内分配（DSL trace 边界）
        self._ord_end = None  # __call__ 内分配（DSL trace 边界）
        self._jrep = None  # __call__ 内分配（DSL trace 边界）
        self._vid = None  # __call__ 内分配（DSL trace 边界）

    def __call__(
        self,
        gm_topk_all,
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
        bi = get_block_idx()

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
            self._assign_chunk(bi + k * bd, gm_topk_all, gm_vtopk)

    @jit
    def _assign_chunk(self, ci, gm_topk_all, gm_vtopk):
        """单 chunk 段倒计时落位，产出 vtopk 块。"""
        num_experts = self._e
        me = self._me
        chunk = self._chunk
        ub_topk = Buffer(MemLoc.UB, (chunk,), dtypes.int32)
        ub_vtopk = Buffer(MemLoc.UB, (chunk,), dtypes.int32)
        ub_pos = Buffer(MemLoc.UB, (num_experts,), dtypes.int32)  # 当前全局序
        ub_send = Buffer(MemLoc.UB, (num_experts,), dtypes.int32)  # 当前段尾
        ub_cvid = Buffer(MemLoc.UB, (num_experts,), dtypes.int32)  # 当前段 vid

        with vf(mode="simt", thread=1):
            self._init_segment_state(ci, num_experts, ub_pos, ub_send)
        mem_copy(ub_topk, tile_slice(gm_topk_all, (1, chunk), (me, ci)))
        with vf(mode="simt", thread=1):
            for i in cannbotdsl.range(chunk):
                e = ub_topk[dtypes.int64(i)]
                fv = self._vid[dtypes.int64(e)]
                if fv >= dtypes.int32(0):
                    ub_vtopk[dtypes.int64(i)] = fv
                else:
                    self._slow_place(e, i, ub_pos, ub_send, ub_cvid, ub_vtopk)
        mem_copy(tile_slice(gm_vtopk, (chunk,), (ci,)), ub_vtopk)

    @jit
    def _init_segment_state(self, ci, num_experts, ub_pos, ub_send):
        """惰性初态：快路径专家（vid>=0）不读状态，跳过初始化；
        段尾置 0 → 该专家在本 chunk 的首个 token 必重定位。"""
        for e in cannbotdsl.range(num_experts):
            if self._vid[dtypes.int64(e)] < dtypes.int32(0):
                ub_pos[dtypes.int64(e)] = (
                    self._dpre[dtypes.int64(e)]
                    + self._pcnt[dtypes.int64(e), dtypes.int64(ci)]
                )
                ub_send[dtypes.int64(e)] = dtypes.int32(0)

    @jit
    def _slow_place(self, e, i, ub_pos, ub_send, ub_cvid, ub_vtopk):
        """慢路径落位：pos 越过段尾时重定位（_relocate），再写 vtopk 并推进 pos。"""
        pos = ub_pos[dtypes.int64(e)]
        if pos >= ub_send[dtypes.int64(e)]:
            vid, seg_idx = self._relocate(e, pos)
            ub_send[dtypes.int64(e)] = self._ord_end[dtypes.int64(e), seg_idx]
            ub_cvid[dtypes.int64(e)] = vid
        ub_vtopk[dtypes.int64(i)] = ub_cvid[dtypes.int64(e)]
        ub_pos[dtypes.int64(e)] = pos + dtypes.int32(1)

    @jit
    def _relocate(self, e, pos):
        """pos 越过段尾时重定位：扫 ord 表找首个满足序位，返回 (新段 vid, 段下标）。"""
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
    依赖链，实测耗时随块数单调下降）；chunk 向上取 2 的幂后降档到整除因子
    （保证 nc*chunk==N，无尾块）；下限 _CHUNK_MIN 防状态重建占比过高。
    """
    want = max(_CHUNK_MIN, -(-num_rows // _MAX_PLAN_BLOCKS))
    chunk = 1 << (want - 1).bit_length() if want > 1 else 1
    while chunk > 1 and num_rows % chunk != 0:
        chunk //= 2
    nc = num_rows // chunk
    bd = min(nc, _MAX_PLAN_BLOCKS)
    while bd > 1 and nc % bd != 0:
        bd -= 1
    return chunk, nc, bd


def _expert_blocks(num_experts):
    """GreedyInit/Pfx 的 block 数：E 的 ≤32 因子。"""
    bd = min(num_experts, _MAX_PLAN_BLOCKS)
    while bd > 1 and num_experts % bd != 0:
        bd -= 1
    return bd


@host
def _moonep_plan_host(
    gm_tpe,
    gm_flag,
    gm_dem_by_rank,
    gm_topk_all,
    gm_rem,
    gm_order,
    gm_loads_part,
    gm_cu,
    gm_etc,
    gm_counts,
    gm_pcnt,
    gm_dpre,
    gm_vid,
    gm_ord_rank,
    gm_ord_end,
    gm_jrep,
    gm_vtopk,
    ctx: StructRef[CommContext],
):
    num_experts = gm_tpe.shape[0]
    world_size = _R  # trace 期烘焙的组内 rank/rank_size（traced 标量不可 int 转换）
    me = _ME
    num_rows = gm_topk_all.shape[1]  # topk 本体已由 wrapper host 侧拷入 [me] 行
    chunk, nc, bd = _plan_chunking(num_rows)
    bdp = _expert_blocks(num_experts)
    _MoonepPlanExchangeKernel(me, world_size, num_experts)[1](
        gm_tpe, gm_flag, gm_dem_by_rank, ctx
    )
    _MoonepPlanGreedyInitKernel(world_size, num_experts, bdp)[bdp](
        gm_dem_by_rank, gm_rem, gm_order, gm_loads_part
    )
    _MoonepPlanGreedyKernel(me, world_size, num_experts, bdp)[1](
        gm_rem, gm_order, gm_loads_part, gm_cu
    )
    _MoonepPlanTablesKernel(world_size, num_experts)[world_size](
        gm_order, gm_jrep, gm_etc
    )
    _MoonepPlanCountKernel(me, num_experts, num_rows, nc, bd)[bd](
        gm_topk_all, gm_counts
    )
    _MoonepPlanPfxKernel(me, world_size, num_experts, nc, bdp)[bdp](
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
    _MoonepPlanAssignKernel(me, world_size, num_experts, num_rows, nc, bd)[bd](
        gm_topk_all,
        gm_pcnt,
        gm_dpre,
        gm_ord_rank,
        gm_ord_end,
        gm_jrep,
        gm_vid,
        gm_vtopk,
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


class _PlanTables(NamedTuple):
    """plan 中间 workspace 表（各 kernel 全量覆写，同 shape 缓存复用安全）。"""

    dem_by_rank: torch.Tensor
    topk_all: torch.Tensor
    counts: torch.Tensor
    rem: torch.Tensor
    order: torch.Tensor
    loads_part: torch.Tensor
    pcnt: torch.Tensor
    dpre: torch.Tensor
    vid: torch.Tensor
    ord_rank: torch.Tensor
    ord_end: torch.Tensor
    jrep: torch.Tensor


def _zero_placeholder(num_rows):
    t = _PLACEHOLDER_CACHE.get(num_rows)
    if t is None:
        t = torch.zeros(num_rows, dtype=torch.int32, device="npu")
        _PLACEHOLDER_CACHE[num_rows] = t
    return t


def _alloc_plan_tables(dims, num_rows, nc, bdp):
    """plan workspace 全量分配（topk_all 尾补槽恒零 = 专家 0 哑元）。"""
    i32 = torch.int32
    return _PlanTables(
        dem_by_rank=torch.zeros(
            dims.world_size, dims.num_experts, dtype=i32, device="npu"
        ),
        topk_all=torch.zeros(dims.world_size, num_rows, dtype=i32, device="npu"),
        counts=torch.zeros(nc, dims.num_experts, dtype=i32, device="npu"),
        rem=torch.zeros(dims.num_experts, dims.world_size, dtype=i32, device="npu"),
        order=torch.zeros(dims.num_experts, dims.world_size, dtype=i32, device="npu"),
        loads_part=torch.zeros(bdp, dims.world_size, dtype=i32, device="npu"),
        pcnt=torch.zeros(dims.num_experts, nc, dtype=i32, device="npu"),
        dpre=torch.zeros(dims.num_experts, dtype=i32, device="npu"),
        vid=torch.zeros(dims.num_experts, dtype=i32, device="npu"),
        ord_rank=torch.zeros(
            dims.num_experts, dims.world_size + 1, dtype=i32, device="npu"
        ),
        ord_end=torch.zeros(
            dims.num_experts, dims.world_size + 1, dtype=i32, device="npu"
        ),
        jrep=torch.zeros(dims.world_size, dims.num_experts, dtype=i32, device="npu"),
    )


def _plan_tables(context, dims, num_rows):
    """按 dims 单槽缓存的 plan workspace（各 kernel 全量覆写，复用安全）。

    流序不变式（跨调用复用缓冲的写入纪律）：复用的设备缓冲只允许 kernel
    全量覆写或设备侧 copy_ 写入，禁止 host 侧写（torch.tensor(..., device="npu")
    之外的同名 H2D 直写/aclrtMemcpy 落到复用缓冲）——host 写会与上一轮仍在读
    该缓冲的 kernel 竞争，表现为 exchange/轮询永久等待（多层调用实测可复现）。
    epoch 类小标量须每次新分配（torch.full）而非复用缓冲回写。
    """
    ws = context.plan_workspace
    if ws is not None and ws[0] == dims:
        return ws[1]
    _, nc, _ = _plan_chunking(num_rows)
    tables = _alloc_plan_tables(dims, num_rows, nc, _expert_blocks(dims.num_experts))
    context.plan_workspace = (dims, tables)
    return tables


def _get_plan_program(num_tokens, top_k, num_experts, world_size, me):
    """按 (S,K,E,R,me) 键控的编译产物缓存（me 进键：rank 经模块级 _ME/_R 烘焙，
    同 shape 不同 rank 不能复用同一 program）。

    _ME/_R 的烘焙与编译在锁内原子完成（多线程并发首次编译不同
    EP 组时，烘焙窗口互污会产出错误 program）。
    """
    key = (num_tokens, top_k, num_experts, world_size, me)
    with _PLAN_LOCK:
        prog = _PLAN_PROGRAMS.get(key)
        if prog is not None:
            return prog
        num_rows = num_tokens * top_k
        _, nc, _ = _plan_chunking(num_rows)
        bdp = _expert_blocks(num_experts)
        global _ME, _R
        _ME, _R = me, world_size
        epn = num_experts // world_size
        i32 = dtypes.int32
        prog = cannbotdsl.compile(
            _moonep_plan_host,
            TensorSpec((num_experts,), i32),
            TensorSpec((1,), i32),
            TensorSpec((world_size, num_experts), i32),
            TensorSpec((world_size, num_rows), i32),
            TensorSpec((num_experts, world_size), i32),
            TensorSpec((num_experts, world_size), i32),
            TensorSpec((bdp, world_size), i32),
            TensorSpec((2 * epn,), i32),
            TensorSpec((world_size, epn), i32),
            TensorSpec((nc, num_experts), i32),
            TensorSpec((num_experts, nc), i32),
            TensorSpec((num_experts,), i32),
            TensorSpec((num_experts,), i32),
            TensorSpec((num_experts, world_size + 1), i32),
            TensorSpec((num_experts, world_size + 1), i32),
            TensorSpec((world_size, num_experts), i32),
            TensorSpec((num_rows,), i32),
            StructRef[CommContext],
        )
        _PLAN_PROGRAMS[key] = prog
        return prog


def _launch_plan(context, dims, ws, gm_flag, io_tensors):
    """编译产物 lookup + 七 kernel 一次 launch（io_tensors = tpe/cu/etc/vtopk）。"""
    tokens_per_expert, cu_seqlens, experts_to_copy, vtopk = io_tensors
    _get_plan_program(
        dims.num_tokens, dims.top_k, dims.num_experts, dims.world_size, context.rank
    )(
        tokens_per_expert,
        gm_flag,
        ws.dem_by_rank,
        ws.topk_all,
        ws.rem,
        ws.order,
        ws.loads_part,
        cu_seqlens,
        experts_to_copy,
        ws.counts,
        ws.pcnt,
        ws.dpre,
        ws.vid,
        ws.ord_rank,
        ws.ord_end,
        ws.jrep,
        vtopk,
        context.ctx_tensor,
    )


def _stash_replica_overflow(context, cu_seqlens, num_rows):
    """暂存溢出检查载荷（plan 返回路径零同步）。

    尾部 D2H（.cpu()/.item()）会把 host 卡到流上全部在飞工作完成，阻塞
    紧随的 dispatch 发射（多层训练实测 941ms 级 host 阻塞）。
    不变量检查延迟到 prefetch/reduce 尾部执行（check_pending_overflow，
    sym_barrier 后流已排空，.item() 微秒级）——无溢出路径语义不变。
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
    # topk 本体 host 侧直拷进 workspace 的 [me] 行（Cnt/Asg 只消费该行；
    # 必须设备侧 copy_（流序）：同 stream 顺序保证 program 的 Cnt kernel
    # 读到新值且不与上一轮 kernel 竞争，kernel 内零 UB 中转，N 不再受
    # 256KB UB 上限约束）
    ws.topk_all[context.rank].copy_(topk_experts_sk.reshape(-1))
    _launch_plan(
        context,
        dims,
        ws,
        gm_flag,
        (tokens_per_expert, cu_seqlens, experts_to_copy, vtopk),
    )
    # 溢出检查延迟执行（返回路径零同步，不阻塞 dispatch 发射）
    _stash_replica_overflow(context, cu_seqlens, num_rows)
    context.last_plan_shape = (num_tokens, top_k)  # 供 prefetch/reduce 的调用顺序守卫
    return dst_n, cu_seqlens, experts_to_copy, src_n, vtopk
