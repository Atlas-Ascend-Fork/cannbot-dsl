# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""moonep_plan：MoonEP 规划算子。

输入路由计数与 topk，输出 PlanResult（五件套，NamedTuple，可按 5-tuple 解包）：

    dst_n[N]                零占位；vtopk-only 快路径下不计算
    cu_seqlens[2*epn]       本 rank 接收区各视图行累计结束偏移
                            （本地专家升序 + 预取槽，空行重复前值）
    experts_to_copy[R,epn]  各 rank 承载的远程专家副本清单（-1 填充）
    src_n[N]                零占位；vtopk-only 快路径下不计算
    vtopk[N]                每个 (s,k) 的虚拟专家 id（扩展空间 2E）

形状约束：E <= 1024、R <= 64、E % R == 0；S*K 不限（仅受显存与时间约束），
且任意 N 原生整除（topk 直接用调用方 [S,K] 扁平视图；分块粒度降档到 N 的
因子，无尾补机制）。

算法（与 test/mc2/moonep/moonep_golden.py 同语义；一次 launch、两次
global_sync_all）：

    阶段 0  Exchange（block 0）：跨卡交换 tokens_per_expert（put + flag + 轮询）
            Count（token_blocks 并行，每 block 串多个 chunk）：per-expert 计数
            GreedyAll（block 0）：demand 矩阵上做 cap 均衡贪心（warp 归约
            argmax/argmin，一卡一源修正），同时生成快路径 vid 表
    阶段 1  Tables（block 内轮询 R 行）：按唯一远程 home 生成压缩 jrep 与
            experts_to_copy
            Pfx（按专家切）：慢专家的 chunk 前缀表与段查找表
    阶段 2  Assign（token_blocks 并行，每 block 串多个 chunk）：快路径 vgather
            查 vid；慢路径下标压缩后按原顺序处理，段查找二分 O(log R)，
            只在段边界重定位，与逐 token 串行扫描逐位等价
"""

import threading
from typing import NamedTuple

import torch

import cannbotdsl
from cannbotdsl import TensorSpec, dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import simt
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.distributed import (
    CommContext,
    SymBuffer,
    TransferRole,
    create_transfer_context,
)
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.reg import (
    full_mask,
    vadd,
    varange,
    vbitwise_or,
    vdups,
    veqs,
    vgather,
    vhistogram_frequency,
    vlts,
    vload,
    vload_deinterleave,
    vunpack,
    vreinterpret,
    vsqueeze_and_storeunalign,
    vsqueeze_and_storeunalign_finalize,
    vsqueeze_and_storeunalign_init,
    vstore,
    vstore_unalign_begin,
    vstorealign_squeeze_status,
)
from cannbotdsl.ops.sync import global_sync_all, vec_sync_all
from cannbotdsl.tensor import MemLoc, make_layout, tile_slice
from cannbotdsl.types.datastruct import StructRef

from moonep_context import (
    DEM_ALL_OFF,
    MAX_E,
    MAX_R,
    READY_OFF,
    busy_wait_flag,
    check_no_shared_elastic_buffer,
    check_pending_overflow,
    record_transport_event,
    resident_blocks,
    sync_transport_stream,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
_MAX_PLAN_BLOCKS = 64  # 单次启动 block 上限，实际取 min(AIV 核数, cap)
_CHUNK_MIN = 128  # chunk 下限（token/块），再小则每块 O(E) 重建占比过高

# 向量粒度（2048-bit 向量）
_I32_LANES = 64  # int32 lanes（Assign 快路径步长）
_HISTOGRAM_U8_LANES = 256  # uint8 lanes（Count 向量路径步长）
_HISTOGRAM_BINS_PER_VL = 128  # uint16 直方图 bins/向量
_HISTOGRAM_MAX_CHUNK = 1 << 16
_HISTOGRAM_SENTINEL = -(1 << 24)  # b3=0xFF 的 int32 表示
_WARP_SIZE = 32
_PLAN_INIT_THREADS = _WARP_SIZE
_XFER_UB_BYTES = 512

# 贪心的极值哨兵
_GREEDY_NEG_INF = -(1 << 30)
_GREEDY_POS_INF = 1 << 30

# 段序 order：0=非服务 rank，1=home，>=2=副本（按创建顺序）
_ORDER_UNSERVED = 0
_ORDER_HOME = 1
_ORDER_FIRST_REPLICA = 2
_VID_SLOW = -1  # 慢专家标记（需走 pcnt/ord 段查找）

_INT32_BYTES = 4  # squeeze 状态字节数 → int32 计数
_SLOTS_PER_RANK = 2  # 每 rank 两组槽位：本地专家 + 预取副本
_WS_ALIGN = 16  # workspace 每段按 16 元素（64B）对齐


# ---------------------------------------------------------------------------
# 纯整数原语
# ---------------------------------------------------------------------------
def ceil_div(x, y):
    """向上取整除法。"""
    return (x + y - 1) // y


def align_up(x, alignment):
    """向上对齐到 alignment 的整数倍。"""
    return ceil_div(x, alignment) * alignment


def ceil_pow2(x):
    """不小于 x 的最小 2 的幂。"""
    return 1 << max(x - 1, 0).bit_length()


# ---------------------------------------------------------------------------
# 运行期状态
# ---------------------------------------------------------------------------
_EVENT_ID = 7
_EPOCH = 0  # 进程内单调调用序号（epoch flag）
_R = 1  # 组内 rank_size（编译锁内写入，参与编译缓存键）


# ---------------------------------------------------------------------------
# plan 单 kernel：跨卡交换、规划与 token 落位
# ---------------------------------------------------------------------------
@kernel
class _PlanKernel:
    """plan 单 kernel：跨卡交换 → 贪心均衡 → 表生成 → token 落位。

    单 launch 多阶段融合；阶段间 global_sync_all() 做 grid 级屏障，块角色由
    block_idx 决定。阶段划分见模块 docstring。
    """

    def __init__(
        self,
        rank_size,
        num_experts,
        num_rows,
        num_chunks,
        token_blocks,
        pfx_blocks,
    ):
        """绑定编译期形状参数。"""
        self._r = int(rank_size)
        self._e = int(num_experts)
        self._epn = self._e // self._r
        self._chunk = int(num_rows) // int(num_chunks)
        self._num_chunks = int(num_chunks)
        self._token_blocks = int(token_blocks)
        self._pfx_blocks = int(pfx_blocks)

    def __call__(
        self,
        gm_tpe,
        gm_flag,
        gm_ws,
        gm_topk,
        gm_io,
        ctx: StructRef[CommContext],
    ):
        """按 block 角色分派三个阶段。"""
        r, e = self._r, self._e
        epn = self._epn
        views = _bind_plan_views(gm_ws, r, e, self._num_chunks)
        # 输出三件套共享一个 io 张量：cu [2epn] | etc [R,epn] | vtopk [N]
        cu_elems = _SLOTS_PER_RANK * epn
        io_base = cu_elems + r * epn
        gm_cu = gm_io[0:cu_elems,]
        gm_etc = gm_io[cu_elems:io_base,].view(r, epn)
        gm_vtopk = gm_io[io_base : io_base + self._chunk * self._num_chunks,]
        block_id = get_block_idx()
        ub_demand = Buffer(MemLoc.UB, dtypes.int32, make_layout((r, e)))

        if block_id == dtypes.int64(0):
            self._exchange(ctx, gm_tpe, gm_flag, views.dem, ub_demand)
        if block_id < dtypes.int64(self._token_blocks):
            self._count(block_id, gm_topk, views.counts)
        if block_id == dtypes.int64(0):
            self._greedy_all(ctx, ub_demand, views, gm_cu)

        global_sync_all()

        # ---- 阶段 1：Tables / Pfx（仅依赖阶段 0 的 order/counts，可并行） ----
        num_blocks = get_block_num()
        self._tables_loop(block_id, num_blocks, views, gm_etc)
        if block_id < dtypes.int64(self._pfx_blocks):
            self._pfx(ctx, block_id, views)

        global_sync_all()

        # ---- 阶段 2：token 落位（按 chunk） ----
        if block_id < dtypes.int64(self._token_blocks):
            self._assign(block_id, gm_topk, gm_vtopk, views)

    # ---- 阶段 0：Exchange（单 block 通信） ----
    @jit
    def _exchange(self, ctx, gm_tpe, gm_flag, gm_dem_by_rank, ub_demand):
        """单 block 跨卡交换 tokens_per_expert，落到 dem 矩阵。"""
        world_size, num_experts = self._r, self._e
        my_rank = dtypes.int32(ctx.rankId)  # 运行时从 CommContext 取（rank 不进编译键）
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
        ub_xfer = Buffer(MemLoc.UB, dtypes.uint8, make_layout((_XFER_UB_BYTES,)))
        xfer = create_transfer_context(ctx, ub_xfer, role=TransferRole.VEC)

        for peer in cannbotdsl.range(world_size):
            if peer != my_rank:
                xfer.put(
                    peer,
                    tile_slice(dem_all, (1, num_experts), (my_rank, 0))
                    .peer(peer)
                    .view((1, num_experts)),
                    gm_tpe,
                )
                xfer.put(
                    peer,
                    tile_slice(ready, (1,), (my_rank,)).peer(peer).view((1,)),
                    gm_flag,
                )
                xfer.flush(peer)
        ub_tpe = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, num_experts)))
        mem_copy(ub_tpe, gm_tpe.view(1, num_experts))
        mem_copy(tile_slice(dem_all, (1, num_experts), (my_rank, 0)), ub_tpe)
        for r in cannbotdsl.range(world_size):
            if r != my_rank:
                busy_wait_flag(tile_slice(ready, (1,), (r,)), gm_flag, _EVENT_ID)
        mem_copy(ub_demand, tile_slice(dem_all, (world_size, num_experts), (0, 0)))
        mem_copy(gm_dem_by_rank, ub_demand)

    # ---- 阶段 0：Count ----
    @jit
    def _count(self, block_id, gm_topk, gm_counts):
        """按 chunk 并行统计 per-expert token 数。"""
        num_experts, chunk, num_chunks, token_blocks = (
            self._e,
            self._chunk,
            self._num_chunks,
            self._token_blocks,
        )
        num_hist_groups = ceil_div(num_experts, _HISTOGRAM_U8_LANES)
        use_vec = chunk % _HISTOGRAM_U8_LANES == 0 and chunk < _HISTOGRAM_MAX_CHUNK

        for k in cannbotdsl.range(num_chunks // token_blocks):
            chunk_id = block_id + k * token_blocks
            if const_expr(use_vec):
                self._count_vec(chunk_id, num_hist_groups, gm_topk, gm_counts)
            else:
                self._count_scalar(chunk_id, gm_topk, gm_counts)

    @jit
    def _count_vec(self, chunk_id, num_hist_groups, gm_topk, gm_counts):
        """向量直方图路径（uint8 打包 + 双 bin）。"""
        num_experts, chunk = self._e, self._chunk
        ub_topk = Buffer(MemLoc.UB, dtypes.int32, make_layout((chunk,)))
        ub_hist = Buffer(
            MemLoc.UB,
            dtypes.int32,
            make_layout((num_hist_groups * _HISTOGRAM_U8_LANES,)),
        )
        mem_copy(ub_topk, tile_slice(gm_topk, (chunk,), (chunk_id,)))
        with vf(mode="simd"):
            bmask = full_mask(32)
            sentinel = vdups(
                dtypes.int32(_HISTOGRAM_SENTINEL), dtypes.int32, mask=bmask
            )
            for boff in range_constexpr(0, chunk, _I32_LANES):
                bv = vload(ub_topk, boff)
                vstore(
                    ub_topk,
                    boff,
                    vbitwise_or(bv, sentinel, mask=bmask),
                    mask=bmask,
                )
        with vf(mode="simd"):
            # full_mask 参数为元素位宽（8=uint8, 16=uint16, 32=int32）
            mask8 = full_mask(8)
            mask16 = full_mask(16)
            mask32 = full_mask(32)
            for group in range_constexpr(num_hist_groups):
                acc0 = vdups(0, dtypes.uint16, mask=mask16)
                acc1 = vdups(0, dtypes.uint16, mask=mask16)
                for off in range_constexpr(0, chunk, _HISTOGRAM_U8_LANES):
                    low0, group0 = vload_deinterleave(ub_topk, off, width="b8")
                    low1, group1 = vload_deinterleave(
                        ub_topk, off + _HISTOGRAM_BINS_PER_VL, width="b8"
                    )
                    mask0 = veqs(group0, group, mask=mask8)
                    mask1 = veqs(group1, group, mask=mask8)
                    acc0 = vadd(
                        acc0,
                        vhistogram_frequency(low0, mask=mask0, bin=0),
                        mask=mask16,
                    )
                    acc0 = vadd(
                        acc0,
                        vhistogram_frequency(low1, mask=mask1, bin=0),
                        mask=mask16,
                    )
                    acc1 = vadd(
                        acc1,
                        vhistogram_frequency(low0, mask=mask0, bin=1),
                        mask=mask16,
                    )
                    acc1 = vadd(
                        acc1,
                        vhistogram_frequency(low1, mask=mask1, bin=1),
                        mask=mask16,
                    )
                base = group * _HISTOGRAM_U8_LANES
                lo = vreinterpret(
                    vunpack(acc0, dtypes.uint32, part="lower"), dtypes.int32
                )
                vstore(ub_hist, base, lo, mask32)
                hi = vreinterpret(
                    vunpack(acc0, dtypes.uint32, part="upper"), dtypes.int32
                )
                vstore(ub_hist, base + _I32_LANES, hi, mask32)
                lo1 = vreinterpret(
                    vunpack(acc1, dtypes.uint32, part="lower"), dtypes.int32
                )
                vstore(ub_hist, base + _HISTOGRAM_BINS_PER_VL, lo1, mask32)
                hi1 = vreinterpret(
                    vunpack(acc1, dtypes.uint32, part="upper"), dtypes.int32
                )
                vstore(
                    ub_hist,
                    base + _HISTOGRAM_BINS_PER_VL + _I32_LANES,
                    hi1,
                    mask32,
                )
        mem_copy(
            tile_slice(gm_counts, (1, num_experts), (chunk_id, 0)),
            ub_hist.view(1, num_experts),
        )

    @jit
    def _count_scalar(self, chunk_id, gm_topk, gm_counts):
        """标量回退路径（chunk 非 256 倍数或过大时）。"""
        num_experts, chunk = self._e, self._chunk
        ub_topk = Buffer(MemLoc.UB, dtypes.int32, make_layout((chunk,)))
        ub_cnt = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))
        with vf(mode="simt", thread=1):
            for e in cannbotdsl.range(num_experts):
                ub_cnt[dtypes.int64(e)] = dtypes.int32(0)
        mem_copy(ub_topk, tile_slice(gm_topk, (chunk,), (chunk_id,)))
        with vf(mode="simt", thread=1):
            for i in cannbotdsl.range(chunk):
                e = ub_topk[dtypes.int64(i)]
                ub_cnt[e] = ub_cnt[e] + dtypes.int32(1)
        mem_copy(
            tile_slice(gm_counts, (1, num_experts), (chunk_id, 0)),
            ub_cnt.view(1, num_experts),
        )

    @jit
    def _greedy_all(self, ctx, ub_demand, views, gm_cu):
        """demand 矩阵 → rem/order 分配 + cu + 快路径 vid。"""
        world_size, num_experts = self._r, self._e
        epn = self._epn
        ub_rem = ub_demand.view(num_experts, world_size)
        ub_order = Buffer(
            MemLoc.UB, dtypes.int32, make_layout((num_experts, world_size))
        )
        ub_loads = Buffer(MemLoc.UB, dtypes.int32, make_layout((world_size,)))
        ub_rhome = Buffer(MemLoc.UB, dtypes.int32, make_layout((world_size,)))
        ub_expert_total = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))
        # cu 与 vid 生命周期重叠，不能别名复用（否则 vid 写进 cu）
        ub_cu = Buffer(MemLoc.UB, dtypes.int32, make_layout((_SLOTS_PER_RANK * epn,)))
        # ctx.rankId 须在 vf scope 外读取（datastruct_ref_load 不允许在 vf 内）
        my_rank = dtypes.int32(ctx.rankId)

        with vf(mode="simt", thread=_PLAN_INIT_THREADS):
            thread_id = simt.thread_idx()[0]
            num_threads = simt.thread_dim()[0]
            for e in cannbotdsl.range(thread_id, num_experts, num_threads):
                total = dtypes.int32(0)
                for r in cannbotdsl.range(world_size):
                    total = total + ub_demand[dtypes.int64(r), dtypes.int64(e)]
                ub_expert_total[dtypes.int64(e)] = total
            simt.syncthreads()
            # demand 已全部读出，ub_demand 转置复用为 rem，控制 UB 峰值
            for e in cannbotdsl.range(thread_id, num_experts, num_threads):
                home_rank = e // epn
                for r in cannbotdsl.range(world_size):
                    ub_rem[dtypes.int64(e), dtypes.int64(r)] = dtypes.int32(0)
                    ub_order[dtypes.int64(e), dtypes.int64(r)] = _ORDER_UNSERVED
                ub_rem[dtypes.int64(e), home_rank] = ub_expert_total[dtypes.int64(e)]
                ub_order[dtypes.int64(e), home_rank] = _ORDER_HOME
            simt.syncthreads()
            for r in cannbotdsl.range(thread_id, world_size, num_threads):
                acc = dtypes.int32(0)
                for e in cannbotdsl.range(num_experts):
                    acc = acc + ub_rem[dtypes.int64(e), dtypes.int64(r)]
                ub_loads[dtypes.int64(r)] = acc
                ub_rhome[dtypes.int64(r)] = dtypes.int32(-1)
            simt.syncthreads()
            if thread_id < _WARP_SIZE:
                lane = simt.lane_idx()
                cap = dtypes.int32(0)
                for j in range_constexpr((world_size + _WARP_SIZE - 1) // _WARP_SIZE):
                    k = lane + j * _WARP_SIZE
                    v = dtypes.int32(0)
                    if k < world_size:
                        v = ub_loads[dtypes.int64(k)]
                    cap = cap + simt.reduce_add(v)
                cap = cap // world_size
                for _ in cannbotdsl.range(world_size):
                    src_r, max_load = self._warp_extreme_idx(
                        ub_loads, world_size, lane, True
                    )
                    if max_load > cap:
                        dst_r, min_load = self._warp_extreme_idx(
                            ub_loads, world_size, lane, False
                        )
                        moved = self._greedy_round_warp(
                            ub_rem,
                            ub_order,
                            ub_rhome,
                            src_r,
                            dst_r,
                            min_load,
                            cap,
                            lane,
                        )
                        if lane == 0:
                            ub_loads[dtypes.int64(src_r)] = (
                                ub_loads[dtypes.int64(src_r)] - moved
                            )
                            ub_loads[dtypes.int64(dst_r)] = cap
                        simt.syncthreads()
                if lane == 0:
                    self._fill_cu_seqlens(my_rank, ub_rem, ub_order, ub_rhome, ub_cu)
            simt.syncthreads()
            for e in cannbotdsl.range(thread_id, num_experts, num_threads):
                home_rank = e // epn
                fast = dtypes.int32(1)
                for r in cannbotdsl.range(world_size):
                    if dtypes.int64(r) != home_rank:
                        if ub_rem[dtypes.int64(e), dtypes.int64(r)] > dtypes.int32(0):
                            fast = dtypes.int32(0)
                if fast != dtypes.int32(0):
                    ub_expert_total[dtypes.int64(e)] = dtypes.int32(
                        home_rank * (_SLOTS_PER_RANK * epn) + (e % epn)
                    )
                else:
                    ub_expert_total[dtypes.int64(e)] = _VID_SLOW
            simt.syncthreads()
        mem_copy(views.rem, ub_rem)
        mem_copy(views.order, ub_order)
        mem_copy(views.rhome, ub_rhome)
        mem_copy(views.vid, ub_expert_total)
        mem_copy(gm_cu, tile_slice(ub_cu, (_SLOTS_PER_RANK * epn,), (0,)))

    @jit
    def _warp_extreme_idx(self, ub_values, num_values, lane, maximize):
        """warp 归约极值，并列取最小下标（maximize=True/False 对应 max/min）。"""
        best = dtypes.int32(_GREEDY_NEG_INF if maximize else _GREEDY_POS_INF)
        idx = dtypes.int32(0)
        invalid = dtypes.int32(_GREEDY_NEG_INF if maximize else _GREEDY_POS_INF)
        for j in range_constexpr((num_values + _WARP_SIZE - 1) // _WARP_SIZE):
            k = lane + j * _WARP_SIZE
            v = invalid
            if k < num_values:
                v = ub_values[dtypes.int64(k)]
            if const_expr(maximize):
                round_best = simt.reduce_max(v)
            else:
                round_best = simt.reduce_min(v)
            candidate = dtypes.int32(num_values)
            if (k < num_values) and (v == round_best):
                candidate = k
            round_idx = simt.reduce_min(candidate)
            if const_expr(maximize):
                better = round_best > best
            else:
                better = round_best < best
            if better:
                best = round_best
                idx = round_idx
        return idx, best

    @jit
    def _warp_max_rem_idx(self, ub_rem, src_r, num_local_experts, lane):
        """最大正 rem 的最小专家下标；全零时返回 (-1, 0)。"""
        expert_start = src_r * self._epn
        best = dtypes.int32(0)
        idx = dtypes.int32(-1)
        for j in range_constexpr((num_local_experts + _WARP_SIZE - 1) // _WARP_SIZE):
            k = lane + j * _WARP_SIZE
            v = dtypes.int32(0)
            if k < num_local_experts:
                v = ub_rem[dtypes.int64(expert_start + k), dtypes.int64(src_r)]
            round_best = simt.reduce_max(v)
            candidate = dtypes.int32(num_local_experts)
            if (k < num_local_experts) and (v == round_best) and (round_best > 0):
                candidate = k
            round_idx = simt.reduce_min(candidate)
            if round_best > best:
                best = round_best
                idx = round_idx
        return idx, best

    @jit
    def _greedy_round_warp(
        self, ub_rem, ub_order, ub_rhome, src_r, dst_r, min_load, cap, lane
    ):
        """一轮组间搬移：warp 选专家，lane 0 写回共享表。"""
        epn = self._epn
        move = cap - min_load
        moved = dtypes.int32(0)
        if lane == 0:
            ub_rhome[dtypes.int64(dst_r)] = src_r
        for _ in cannbotdsl.range(epn):
            if move > dtypes.int32(0):
                e_local, rem_best = self._warp_max_rem_idx(ub_rem, src_r, epn, lane)
                if rem_best > dtypes.int32(0):
                    take = rem_best
                    if move < take:
                        take = move
                    if lane == 0:
                        e_best = src_r * epn + e_local
                        if ub_order[dtypes.int64(e_best), dtypes.int64(dst_r)] == (
                            _ORDER_UNSERVED
                        ):
                            num_copies = dtypes.int32(0)
                            for r in cannbotdsl.range(self._r):
                                if ub_order[
                                    dtypes.int64(e_best), dtypes.int64(r)
                                ] > dtypes.int32(0):
                                    num_copies = num_copies + dtypes.int32(1)
                            ub_order[dtypes.int64(e_best), dtypes.int64(dst_r)] = (
                                num_copies + dtypes.int32(1)
                            )
                        ub_rem[dtypes.int64(e_best), dtypes.int64(src_r)] = (
                            ub_rem[dtypes.int64(e_best), dtypes.int64(src_r)] - take
                        )
                        ub_rem[dtypes.int64(e_best), dtypes.int64(dst_r)] = (
                            ub_rem[dtypes.int64(e_best), dtypes.int64(dst_r)] + take
                        )
                    move = move - take
                    moved = moved + take
                    # lane 0 的 rem/order 写回须在下一次 warp 扫描前可见
                    simt.syncthreads()
        return moved

    @jit
    def _fill_cu_seqlens(self, my_rank, ub_rem, ub_order, ub_rhome, ub_cu):
        """[2*epn] 槽位行数前缀和 → cu_seqlens。副本槽按源 rank 专家升序压缩填充。"""
        epn = self._epn
        home = ub_rhome[dtypes.int64(my_rank)]
        for i in range_constexpr(epn):
            ub_cu[dtypes.int64(i)] = ub_rem[
                dtypes.int64(my_rank * epn + i), dtypes.int64(my_rank)
            ]
        for i in range_constexpr(epn):
            ub_cu[dtypes.int64(epn + i)] = dtypes.int32(0)
        if home >= dtypes.int32(0):
            rep_cnt = dtypes.int32(0)
            for i in cannbotdsl.range(epn):
                e = home * epn + i
                if (
                    ub_order[dtypes.int64(e), dtypes.int64(my_rank)]
                    >= _ORDER_FIRST_REPLICA
                ):
                    ub_cu[dtypes.int64(epn) + rep_cnt] = ub_rem[
                        dtypes.int64(e), dtypes.int64(my_rank)
                    ]
                    rep_cnt = rep_cnt + dtypes.int32(1)
        acc = dtypes.int32(0)
        for i in range_constexpr(_SLOTS_PER_RANK * epn):
            acc = acc + ub_cu[dtypes.int64(i)]
            ub_cu[dtypes.int64(i)] = acc

    # ---- 阶段 1：Tables ----
    @jit
    def _tables_loop(self, block_id, num_blocks, views, gm_etc):
        """当前 block 轮询处理 rank block_id, block_id+num_blocks, ...；R 可大于 block 数。"""
        for r in cannbotdsl.range(block_id, self._r, num_blocks):
            self._tables(r, views, gm_etc)

    def _tables(self, r, views, gm_etc):
        """生成 rank r 的 jrep 与 experts_to_copy 行。"""
        epn = self._epn
        ub_jrep_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, epn)))
        ub_etc_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, epn)))

        with vf(mode="simt", thread=1):
            self._fill_jrep_etc(
                views.order, views.rhome, ub_jrep_row, ub_etc_row, r, epn
            )
        mem_copy(tile_slice(views.jrep, (1, epn), (r, 0)), ub_jrep_row)
        mem_copy(tile_slice(gm_etc, (1, epn), (r, 0)), ub_etc_row)

    @jit
    def _fill_jrep_etc(self, gm_order, gm_rhome, ub_jrep_row, ub_etc_row, r, epn):
        """压缩副本映射：jrep 行与 etc 行。"""
        for j in range_constexpr(epn):
            ub_jrep_row[dtypes.int64(0), dtypes.int64(j)] = dtypes.int32(0)
            ub_etc_row[dtypes.int64(0), dtypes.int64(j)] = dtypes.int32(-1)
        home = simt.l2cache_load(gm_rhome.ptr((dtypes.int64(r),)))
        if home >= dtypes.int32(0):
            copy_idx = dtypes.int32(0)
            for i in cannbotdsl.range(epn):
                e = home * epn + i
                seg = simt.l2cache_load(
                    gm_order.ptr((dtypes.int64(e), dtypes.int64(r)))
                )
                if seg >= _ORDER_FIRST_REPLICA:
                    ub_jrep_row[dtypes.int64(0), dtypes.int64(i)] = copy_idx
                    ub_etc_row[dtypes.int64(0), copy_idx] = dtypes.int32(e)
                    copy_idx = copy_idx + dtypes.int32(1)

    # ---- 阶段 1：Pfx ----
    @jit
    def _pfx(self, ctx, block_id, views):
        """慢专家的 chunk 前缀位置表与段查找表。"""
        world_size, num_experts = self._r, self._e
        num_chunks = self._num_chunks
        experts_per_block = num_experts // self._pfx_blocks
        expert_start = block_id * experts_per_block

        pcnt_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, num_chunks)))
        ord_rank_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, world_size)))
        ord_end_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, world_size)))

        for k in cannbotdsl.range(experts_per_block):
            e = expert_start + k
            self._build_pfx_row(ctx, e, views, pcnt_row, ord_rank_row, ord_end_row)
            mem_copy(tile_slice(views.pcnt, (1, num_chunks), (e, 0)), pcnt_row)
            mem_copy(tile_slice(views.ord_rank, (1, world_size), (e, 0)), ord_rank_row)
            mem_copy(tile_slice(views.ord_end, (1, world_size), (e, 0)), ord_end_row)

    @jit
    def _build_pfx_row(self, ctx, e, views, pcnt_row, ord_rank_row, ord_end_row):
        """单个专家的 pcnt/ord_rank/ord_end 行。"""
        world_size = self._r
        my_rank = dtypes.int32(ctx.rankId)  # 运行时取（self-serve 偏好序）
        num_chunks = self._num_chunks
        gm_dem_by_rank, gm_counts = views.dem, views.counts
        gm_rem, gm_order, gm_vid = views.rem, views.order, views.vid
        with vf(mode="simt", thread=1):
            # fast expert 早退：Assign 快路径直读 vid，不消费本行 pcnt/ord
            vid = simt.l2cache_load(gm_vid.ptr((dtypes.int64(e),)))
            if vid < dtypes.int32(0):
                # 段序表先行：偏好序依赖 home/副本的段排列
                for seg in cannbotdsl.range(world_size):
                    ord_rank_row[dtypes.int64(0), dtypes.int64(seg)] = dtypes.int32(-1)
                for r in cannbotdsl.range(world_size):
                    seg = simt.l2cache_load(
                        gm_order.ptr((dtypes.int64(e), dtypes.int64(r)))
                    )
                    if seg >= _ORDER_HOME:
                        ord_rank_row[dtypes.int64(0), dtypes.int64(seg - 1)] = (
                            dtypes.int32(r)
                        )
                # 偏好序 [home, 副本按段序, 非服务 rank]，pcnt 取其前序需求之和
                prefix_base = dtypes.int32(0)
                seen_self = dtypes.int32(0)
                for seg in cannbotdsl.range(world_size):
                    r = ord_rank_row[dtypes.int64(0), dtypes.int64(seg)]
                    if r >= dtypes.int32(0) and seen_self == dtypes.int32(0):
                        if r == my_rank:
                            seen_self = dtypes.int32(1)
                        else:
                            prefix_base = prefix_base + simt.l2cache_load(
                                gm_dem_by_rank.ptr((dtypes.int64(r), dtypes.int64(e)))
                            )
                for r in cannbotdsl.range(world_size):
                    if seen_self == dtypes.int32(0):
                        if r == my_rank:
                            seen_self = dtypes.int32(1)
                        else:
                            seg = simt.l2cache_load(
                                gm_order.ptr((dtypes.int64(e), dtypes.int64(r)))
                            )
                            if seg == _ORDER_UNSERVED:
                                prefix_base = prefix_base + simt.l2cache_load(
                                    gm_dem_by_rank.ptr(
                                        (dtypes.int64(r), dtypes.int64(e))
                                    )
                                )
                chunk_prefix = dtypes.int32(0)
                for c in range_constexpr(num_chunks):
                    pcnt_row[dtypes.int64(0), dtypes.int64(c)] = (
                        prefix_base + chunk_prefix
                    )
                    chunk_prefix = chunk_prefix + simt.l2cache_load(
                        gm_counts.ptr((dtypes.int64(c), dtypes.int64(e)))
                    )
                seg_end_acc = dtypes.int32(0)
                for seg in cannbotdsl.range(world_size):
                    rnk = ord_rank_row[dtypes.int64(0), dtypes.int64(seg)]
                    if rnk >= dtypes.int32(0):
                        seg_end_acc = seg_end_acc + simt.l2cache_load(
                            gm_rem.ptr((dtypes.int64(e), dtypes.int64(rnk)))
                        )
                    ord_end_row[dtypes.int64(0), dtypes.int64(seg)] = seg_end_acc

    # ---- 阶段 2：Assign ----
    @jit
    def _assign(self, block_id, gm_topk, gm_vtopk, views):
        """token 落位：按 chunk 并行处理快/慢路径。"""
        world_size, num_experts = self._r, self._e
        num_chunks, token_blocks = self._num_chunks, self._token_blocks

        # buf_id 预算：ord_end 是二分热表留 UB；ord_rank/jrep 每轮只读一次，
        # 走 L2 直读省 2 个 buf_id。
        self._ord_end = Buffer(
            MemLoc.UB, dtypes.int32, make_layout((num_experts, world_size))
        )
        mem_copy(self._ord_end, views.ord_end)
        self._vid = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))
        mem_copy(self._vid, views.vid)

        for k in cannbotdsl.range(num_chunks // token_blocks):
            self._assign_chunk(block_id + k * token_blocks, gm_topk, gm_vtopk, views)

    @jit
    def _assign_chunk(self, chunk_id, gm_topk, gm_vtopk, views):
        """单 chunk 落位：vgather 快路径 + 段查找慢路径。"""
        num_experts = self._e
        chunk = self._chunk
        gm_pcnt, gm_ord_rank, gm_jrep = views.pcnt, views.ord_rank, views.jrep
        ub_topk = Buffer(MemLoc.UB, dtypes.int32, make_layout((chunk,)))
        ub_vtopk = Buffer(MemLoc.UB, dtypes.int32, make_layout((chunk,)))
        ub_slow_idx = Buffer(MemLoc.UB, dtypes.int32, make_layout((chunk,)))
        ub_num_slow = Buffer(MemLoc.UB, dtypes.int32, make_layout((1,)))
        ub_pos = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))
        ub_seg_end = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))
        ub_cur_vid = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))

        with vf(mode="simt", thread=1):
            for e in cannbotdsl.range(num_experts):
                if self._vid[dtypes.int64(e)] < dtypes.int32(0):
                    ub_pos[dtypes.int64(e)] = simt.l2cache_load(
                        gm_pcnt.ptr((dtypes.int64(e), dtypes.int64(chunk_id)))
                    )
                    ub_seg_end[dtypes.int64(e)] = dtypes.int32(0)
        mem_copy(ub_topk, tile_slice(gm_topk, (chunk,), (chunk_id,)))
        vec_end = chunk - chunk % _I32_LANES
        if const_expr(vec_end > 0):
            with vf(mode="simd"):
                mask = full_mask(32)  # int32 全 lane
                cursor = vstore_unalign_begin(ub_slow_idx)
                for off in range_constexpr(0, vec_end, _I32_LANES):
                    indices = vreinterpret(vload(ub_topk, off), dtypes.uint32)
                    values = vgather(self._vid, indices, mask=mask)
                    vstore(ub_vtopk, off, values, mask)
                    slow = vlts(values, 0, mask=mask)
                    positions = varange(off, dtypes.int32)
                    squeezed = vsqueeze_and_storeunalign_init(positions, mask=slow)
                    vsqueeze_and_storeunalign(ub_slow_idx, 0, squeezed, cursor)
                vsqueeze_and_storeunalign_finalize(ub_slow_idx, 0, cursor)
                vstorealign_squeeze_status(ub_num_slow, 0)
            vec_sync_all()
        with vf(mode="simt", thread=1):
            # squeeze 状态字节数 → int32 计数
            if const_expr(vec_end > 0):
                num_slow = ub_num_slow[dtypes.int64(0)] // _INT32_BYTES
            else:
                num_slow = dtypes.int32(0)
            for i in range_constexpr(vec_end, chunk):
                e = ub_topk[dtypes.int64(i)]
                cur_vid = self._vid[dtypes.int64(e)]
                ub_vtopk[dtypes.int64(i)] = cur_vid
                if cur_vid < dtypes.int32(0):
                    ub_slow_idx[dtypes.int64(num_slow)] = dtypes.int32(i)
                    num_slow = num_slow + dtypes.int32(1)
            for j in cannbotdsl.range(num_slow):
                i = ub_slow_idx[dtypes.int64(j)]
                e = ub_topk[dtypes.int64(i)]
                pos = ub_pos[dtypes.int64(e)]
                if pos >= ub_seg_end[dtypes.int64(e)]:
                    vid, seg_idx = self._relocate(e, pos, gm_ord_rank, gm_jrep)
                    ub_seg_end[dtypes.int64(e)] = self._ord_end[
                        dtypes.int64(e), seg_idx
                    ]
                    ub_cur_vid[dtypes.int64(e)] = vid
                ub_vtopk[dtypes.int64(i)] = ub_cur_vid[dtypes.int64(e)]
                ub_pos[dtypes.int64(e)] = pos + dtypes.int32(1)
        mem_copy(tile_slice(gm_vtopk, (chunk,), (chunk_id,)), ub_vtopk)

    @jit
    def _relocate(self, e, pos, gm_ord_rank, gm_jrep):
        """二分定位 pos 所在段，返回 (vid, seg_idx)。"""
        world_size, epn = self._r, self._epn
        # ord_end[e,:] 非降累计——二分找首个 pos < end 的段
        lo = dtypes.int32(0)
        span = dtypes.int32(world_size)
        for _ in range_constexpr(world_size.bit_length()):
            half = span // dtypes.int32(2)
            mid = lo + half
            end_mid = self._ord_end[dtypes.int64(e), dtypes.int64(mid)]
            if pos < end_mid:
                span = half
            else:
                lo = mid + dtypes.int32(1)
                span = span - half - dtypes.int32(1)
        seg_idx = lo
        dst_rank = simt.l2cache_load(
            gm_ord_rank.ptr((dtypes.int64(e), dtypes.int64(seg_idx)))
        )
        vid = dst_rank * dtypes.int32(_SLOTS_PER_RANK * epn) + (e % dtypes.int32(epn))
        if seg_idx >= dtypes.int32(1):
            vid = (
                dst_rank * dtypes.int32(_SLOTS_PER_RANK * epn)
                + dtypes.int32(epn)
                + simt.l2cache_load(
                    gm_jrep.ptr((dtypes.int64(dst_rank), dtypes.int64(e % epn)))
                )
            )
        return vid, seg_idx


# ---------------------------------------------------------------------------
# host 入口
# ---------------------------------------------------------------------------


def _plan_chunking(num_rows):
    """分块：chunk 取 2 的幂且整除 N，下限 _CHUNK_MIN。"""
    cap = resident_blocks(_MAX_PLAN_BLOCKS)
    target_chunk = max(_CHUNK_MIN, ceil_div(num_rows, cap))
    chunk = ceil_pow2(target_chunk)
    while chunk > 1 and num_rows % chunk != 0:
        chunk //= 2
    num_chunks = num_rows // chunk
    token_blocks = min(num_chunks, cap)
    while token_blocks > 1 and num_chunks % token_blocks != 0:
        token_blocks -= 1
    return chunk, num_chunks, token_blocks


def _pfx_block_count(num_experts):
    """Pfx 的 block 数：E 的可整除因子，且不超过本机可常驻核数。"""
    block_count = min(num_experts, resident_blocks(_MAX_PLAN_BLOCKS))
    while block_count > 1 and num_experts % block_count != 0:
        block_count -= 1
    return block_count


class _PlanBlocks(NamedTuple):
    """plan 分块：chunk 总数、token block 数与 Pfx block 数。"""

    num_chunks: int
    token_blocks: int
    pfx_blocks: int


def _plan_blocks(num_rows, num_experts):
    """分块与专家块一次算全；不做 lru_cache（块数依赖运行期核数）。"""
    _, num_chunks, token_blocks = _plan_chunking(num_rows)
    return _PlanBlocks(num_chunks, token_blocks, _pfx_block_count(num_experts))


@host
def _moonep_plan_host(
    gm_tpe,
    gm_flag,
    gm_ws,
    gm_topk,
    gm_io,
    ctx: StructRef[CommContext],
):
    """单 launch 发射：三个阶段一次提交。"""
    num_experts = gm_tpe.shape[0]
    world_size = _R  # trace 期烘焙；rank 运行时由 ctx.rankId 取
    num_rows = gm_topk.shape[0]  # 本 rank topk 扁平视图，直接作 Cnt/Asg 输入
    blocks = _plan_blocks(num_rows, num_experts)
    _PlanKernel(
        world_size,
        num_experts,
        num_rows,
        blocks.num_chunks,
        blocks.token_blocks,
        blocks.pfx_blocks,
    )[
        min(
            max(blocks.token_blocks, blocks.pfx_blocks),
            resident_blocks(_MAX_PLAN_BLOCKS),
        )
    ](
        gm_tpe,
        gm_flag,
        gm_ws,
        gm_topk,
        gm_io,
        ctx,
    )


_PLAN_PROGRAMS = {}
_PLAN_LOCK = threading.Lock()

# dst_n/src_n 零占位：只读、kernel 不写，进程级缓存省 alloc+H2D。
_PLACEHOLDER_CACHE = {}

# 预生成 1..cap 的 epoch 值池，取独立标量视图，省每轮 torch.full。
_EPOCH_POOLS = []
_EPOCH_POOL_CAP = 0
_EPOCH_POOL_MIN_CAP = 1 << 16


def _epoch_flag(epoch):
    """取 epoch 对应的独立标量视图（预生成值池）。"""
    global _EPOCH_POOL_CAP
    if not _EPOCH_POOLS or epoch > _EPOCH_POOL_CAP:
        _EPOCH_POOL_CAP = max(_EPOCH_POOL_MIN_CAP, 1 << (epoch - 1).bit_length())
        _EPOCH_POOLS.append(
            torch.arange(1, _EPOCH_POOL_CAP + 1, dtype=torch.int32, device="npu")
        )
    return _EPOCH_POOLS[-1][epoch - 1 : epoch]


class _PlanDims(NamedTuple):
    """plan 形状键：单 rank token 数 / top-k / 专家总数 / EP rank 数。"""

    num_tokens: int
    top_k: int
    num_experts: int
    world_size: int


# workspace 布局：(表名, 维度名)；元素数为各维之积。
_WS_SPECS = (
    ("dem", ("r", "e")),
    ("rem", ("e", "r")),
    ("order", ("e", "r")),
    ("counts", ("num_chunks", "e")),
    ("pcnt", ("e", "num_chunks")),
    ("vid", ("e",)),
    ("ord_rank", ("e", "r")),
    ("ord_end", ("e", "r")),
    ("jrep", ("r", "epn")),
    ("rhome", ("r",)),
)


def _ws_tables(rank_size, num_experts, num_chunks):
    """(name, offset, count, shape) 序列 + 总元素数（布局的唯一来源）。"""
    dims = {
        "r": rank_size,
        "e": num_experts,
        "num_chunks": num_chunks,
        "epn": num_experts // rank_size,
    }
    out = []
    cur = 0
    for name, dim_names in _WS_SPECS:
        shape = tuple(dims[d] for d in dim_names)
        count = 1
        for dim in shape:
            count *= dim
        cur = align_up(cur, _WS_ALIGN)
        out.append((name, cur, count, shape))
        cur += count
    return tuple(out), align_up(cur, _WS_ALIGN)


def _ws_layout(rank_size, num_experts, num_chunks):
    """workspace 布局：{表名: 元素偏移} + total（host 分配与测试使用）。"""
    tables, total = _ws_tables(rank_size, num_experts, num_chunks)
    offs = {name: off for name, off, _, _ in tables}
    offs["total"] = total
    return offs


class _PlanViews(NamedTuple):
    """plan 中间表在 workspace 上的具名视图（kernel 一次绑定、按需传参）。"""

    dem: object
    rem: object
    order: object
    counts: object
    pcnt: object
    vid: object
    ord_rank: object
    ord_end: object
    jrep: object
    rhome: object


def _bind_plan_views(gm_ws, rank_size, num_experts, num_chunks):
    """按 _WS_SPECS 把扁平 workspace 绑成具名视图（布局与切片同一来源）。"""
    tables, _ = _ws_tables(rank_size, num_experts, num_chunks)
    bound = {}
    for name, off, count, shape in tables:
        seg = gm_ws[off : off + count,]
        bound[name] = seg.view(*shape) if len(shape) > 1 else seg
    return _PlanViews(
        dem=bound["dem"],
        rem=bound["rem"],
        order=bound["order"],
        counts=bound["counts"],
        pcnt=bound["pcnt"],
        vid=bound["vid"],
        ord_rank=bound["ord_rank"],
        ord_end=bound["ord_end"],
        jrep=bound["jrep"],
        rhome=bound["rhome"],
    )


def _zero_placeholder(num_rows):
    """进程级缓存的零占位张量（只读）。"""
    t = _PLACEHOLDER_CACHE.get(num_rows)
    if t is None:
        t = torch.zeros(num_rows, dtype=torch.int32, device="npu")
        _PLACEHOLDER_CACHE[num_rows] = t
    return t


def _get_plan_program(num_tokens, top_k, num_experts, world_size):
    """按 (S,K,E,R,块数) 键控的编译产物缓存。"""
    # 块数也是编译期常量（角色分派），核数不同的机器不能复用同一 program
    key = (
        num_tokens,
        top_k,
        num_experts,
        world_size,
        resident_blocks(_MAX_PLAN_BLOCKS),
    )
    with _PLAN_LOCK:
        prog = _PLAN_PROGRAMS.get(key)
        if prog is not None:
            return prog
        num_rows = num_tokens * top_k
        blocks = _plan_blocks(num_rows, num_experts)
        ws_total = _ws_layout(world_size, num_experts, blocks.num_chunks)["total"]
        global _R
        _R = world_size
        epn = num_experts // world_size
        i32 = dtypes.int32
        prog = cannbotdsl.compile(
            _moonep_plan_host,
            TensorSpec((num_experts,), i32),  # tokens_per_expert
            TensorSpec((1,), i32),  # epoch flag
            TensorSpec((ws_total,), i32),  # dem + 打包 workspace
            TensorSpec((num_rows,), i32),  # topk 扁平视图
            TensorSpec(
                (_SLOTS_PER_RANK * epn + world_size * epn + num_rows,), i32
            ),  # 合并输出
            StructRef[CommContext],
        )
        _PLAN_PROGRAMS[key] = prog
        return prog


class PlanResult(NamedTuple):
    """plan 五件套（NamedTuple，可按 5-tuple 解包）。"""

    dst_n: torch.Tensor
    cu_seqlens: torch.Tensor
    experts_to_copy: torch.Tensor
    src_n: torch.Tensor
    vtopk: torch.Tensor


def _validate_plan_result(result, world_size, epn, num_rows):
    """校验五件套的形状/dtype 契约（仅 verify/验收路径调用，避免 host 开销）。"""
    expected = (
        ("dst_n", result.dst_n, (num_rows,)),
        ("cu_seqlens", result.cu_seqlens, (_SLOTS_PER_RANK * epn,)),
        ("experts_to_copy", result.experts_to_copy, (world_size, epn)),
        ("src_n", result.src_n, (num_rows,)),
        ("vtopk", result.vtopk, (num_rows,)),
    )
    for name, tensor, shape in expected:
        assert tuple(tensor.shape) == shape, (
            f"{name} shape {tuple(tensor.shape)} != {shape}"
        )
        assert tensor.dtype == torch.int32, f"{name} dtype {tensor.dtype} != int32"


def moonep_plan(context, tokens_per_expert, topk_experts_sk, *, verify=False):
    """规划入口。

    Args:
        context: MoonEP 通信上下文（见 moonep_context.py）。
        tokens_per_expert: [E] int32，本 rank 各专家的 token 计数。
        topk_experts_sk: [S, K] int32，本 rank 的 top-k 专家索引。
        verify: True 时在返回前同步校验布局不变量；生产路径保持 False。

    Returns:
        PlanResult（NamedTuple，可按 5-tuple 解包）：
        dst_n[N] / src_n[N] 为零占位；cu_seqlens[2*epn]、experts_to_copy[R,epn]
        与 vtopk[N] 由 kernel 全量产出。

    Notes:
        - 返回路径零同步：溢出与布局不变量延迟到下一个 moonep 算子尾部校验。
        - 形状约束：E <= 1024、R <= 64、E % R == 0；S*K 任意 N 原生整除。
        - 规划结果可跨调用、跨前反向复用。
    """

    if tokens_per_expert.dtype != torch.int32 or topk_experts_sk.dtype != torch.int32:
        raise TypeError(
            f"tpe/topk 须为 int32（bincount/topk 默认 int64，需 .to(torch.int32)）："
            f"got {tokens_per_expert.dtype} / {topk_experts_sk.dtype}"
        )
    if not context.eb_scan_done:  # 通信域分离契约复查：覆盖 ctx 先建、EB 后建
        check_no_shared_elastic_buffer(context.group_name)
        context.eb_scan_done = True
    cur_stream = sync_transport_stream(context)  # 跨流护栏（握手状态机要求调用按序）
    world_size = context.rank_size
    # 硬约束：R ≤ MAX_R=64（dem/rhome 表维度）
    if world_size > MAX_R:
        raise RuntimeError(
            f"EP 组规模 {world_size} 超过 MAX_R={MAX_R}（dem/rhome 表维度上限）；"
            "128P 需扩 MAX_R 并做表 GM 分块/CSR"
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
    # 三个输出共享一次分配；kernel 对 cu/etc/vtopk 均全量覆写，empty 即可
    cu_elems = _SLOTS_PER_RANK * epn
    etc_elems = world_size * epn
    io = torch.empty(cu_elems + etc_elems + num_rows, dtype=torch.int32, device="npu")
    cu_seqlens = io[:cu_elems]
    experts_to_copy = io[cu_elems : cu_elems + etc_elems].view(world_size, epn)
    src_n = _zero_placeholder(num_rows)
    vtopk = io[cu_elems + etc_elems :]
    global _EPOCH
    _EPOCH += 1
    # _R 的烘焙在 _get_plan_program 锁内（与编译原子）；wrapper 只管 epoch
    gm_flag = _epoch_flag(_EPOCH)
    # workspace 单槽缓存：同 shape 复用，shape 变化时按新布局重分配。
    ws_entry = context.plan_workspace
    if ws_entry is None or ws_entry[0] != dims:
        blocks = _plan_blocks(num_rows, dims.num_experts)
        ws_total = _ws_layout(dims.world_size, dims.num_experts, blocks.num_chunks)[
            "total"
        ]
        tables = torch.zeros(ws_total, dtype=torch.int32, device="npu")
        ws_entry = (dims, tables)
        context.plan_workspace = ws_entry
    tables = ws_entry[1]
    # Cnt/Asg 只消费本 rank 的行，直接用 [S,K] 展平视图省一次整行 copy_。
    _get_plan_program(dims.num_tokens, dims.top_k, dims.num_experts, dims.world_size)(
        tokens_per_expert,
        gm_flag,
        tables,
        topk_experts_sk.reshape(-1),
        io,
        context.ctx_tensor,
    )
    # 跨流护栏 fork 点：只等我方算子，不等主流尾部
    record_transport_event(context, stream=cur_stream)
    # 溢出检查延迟执行（返回路径零同步，不阻塞 dispatch 发射）
    context.pending_overflow = (cu_seqlens, num_rows)
    context.last_plan_shape = (num_tokens, top_k)  # 供 prefetch/reduce 的调用顺序守卫
    if verify:
        # 立即同步校验，坏布局在消费前熔断；代价一次流同步，生产勿开
        check_pending_overflow(context)

    result = PlanResult(dst_n, cu_seqlens, experts_to_copy, src_n, vtopk)
    if verify:
        _validate_plan_result(result, world_size, epn, num_rows)
    return result


# ---------------------------------------------------------------------------
# AOT 收集器（离线构建期预编译入口；源码直跑不受影响）
# ---------------------------------------------------------------------------
try:
    from cannbotdsl import aot
except ImportError:
    aot = None

if aot is not None and hasattr(aot, "export"):

    @aot.export("moonep_plan")
    def export_moonep_plan():
        """预编译推荐配置（S=8192,K=16,E=224,R=8）。"""
        _get_plan_program(8192, 16, 224, 8)
