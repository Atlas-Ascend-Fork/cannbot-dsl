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
                          rank*NvS+slot；恢复需 kernel 内重建落位表）
  cu_seqlens[2*epn]      本 rank 接收区各视图行累计结束偏移（本地专家升序 + 预取槽，空行重复前值）
  experts_to_copy[R,epn] 各 rank 承载的远程专家副本清单（-1 填充）
  src_n[N]               【零占位】vtopk-only 快路径下不计算（原语义：接收槽反查表）
  vtopk[N]               每个 (s,k) 的虚拟专家 id 投影（扩展空间 2E，供 ep dispatch 路由）

形状约束：E<=1024、R<=64、E%R==0；S*K 不设上限且任意 N 原生整除
（topk 直接使用调用方 [S,K] 扁平视图；分块粒度降档到 N 的因子，无尾补机制）。

算法（与 test/mc2/moonep/moonep_golden.py 的规划器同语义）：
  整段（_PlanKernel，一次 launch、两次 global_sync_all）：
    阶段0  Exchange（仅 block 0）：跨卡交换 tokens_per_expert（put + flag + 轮询）
           Count（bi<bd，按 chunk 切）：分块 per-expert token 数
           GreedyAll（仅 block 0）：demand 矩阵上运行 cap 均衡贪心
           （warp 归约 argmax/argmin，一卡一源修正），同时生成快路径 vid 表
           （无远程副本的专家 vid 恒定，供 Assign 快路径 vgather 直读）

    阶段1  Tables（bi<R，每 block 一个 rank）：按唯一远程 home 生成压缩 jrep
           与 experts_to_copy
           Pfx（按专家切）：慢专家的 chunk 前缀位置表与段查找表（vid 已在
           GreedyAll 生成；fast expert 早退；位置拼接用 self-serve 偏好序）
    阶段2  Assign（bi<bd，按 chunk 切）：快路径经 vgather 查 vid 表；慢路径下标
           压缩后按原顺序处理，段查找二分（O(log R)），只在段边界重定位，
           与逐 token 串行扫描语义逐位等价
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
    check_pending_overflow,
    record_transport_event,
    resident_blocks,
    sync_transport_stream,
)

# 规划区在对称堆内的布局（heap_base=4096 起算的字节偏移，32B 对齐）
_EVENT_ID = 7
_EPOCH = 0  # 进程内单调调用序号（epoch flag）
_R = 1  # 组内 rank_size（编译锁内写入，须进编译缓存键）

# 单次启动的 block 数经验上限。融合 kernel 用 global_sync_all 做 grid 级同步屏障，要求
# 全部 block 同时常驻——实际块数由 `resident_blocks()` 按本机有效 AIV 核数
# 取 min，不能硬编码 64（56 核机型上 64 块会分时复用，同步屏障收不齐挂死）。
_MAX_PLAN_BLOCKS = 64
# 分块粒度下限（token/块）：再小则每 chunk 的 O(E) 状态重建占比过高
_CHUNK_MIN = 128

# 2048-bit 向量宽度下的各元素粒度
_I32_LANES = 64  # int32 lanes（Assign 快路径步长）
_HISTOGRAM_BINS_PER_VL = 128  # uint16 直方图 bins/向量
_HISTOGRAM_U8_LANES = 256  # uint8 lanes（Count 向量路径步长）
_HISTOGRAM_MAX_CHUNK = 1 << 16
# Count 向量路径哨兵：vload_deinterleave(width="b8") 返回 (偶字节, 奇字节) 平面，
# int32 输入下 low={b0,b2}、group={b1,b3} 交织——b2/b3 全 0 时伪 lane 混过
# group 过滤，把每值一个"零"灌进 bin 0（实测每 chunk 恰好 +chunk）。
# 给输入 OR 上 b3=0xFF：伪 lane 的 group 变 0xFF 永不匹配，真 lane 的 group
# 仍为 b1（= id>>8），直方图逐 lane 掩码将其精确滤除。
_HISTOGRAM_SENTINEL = -(1 << 24)  # 0xFF<<24 的 int32 有符号表示（b3=0xFF）
_WARP_SIZE = 32
_PLAN_INIT_THREADS = _WARP_SIZE
_XFER_UB_BYTES = 512

_GREEDY_NEG_INF = -(1 << 30)
_GREEDY_POS_INF = 1 << 30

# 段序 order：0=非服务 rank，1=home，>=2=副本（按创建顺序）。
_ORDER_UNSERVED = 0
_ORDER_HOME = 1
_ORDER_FIRST_REPLICA = 2
_VID_SLOW = -1  # 慢专家标记（需走 pcnt/ord 段查找）
_INT32_BYTES = 4  # squeeze 状态字节数 → int32 计数换算


# ---------------------------------------------------------------------------
# plan 单 kernel：跨卡交换、规划与 token 落位
# ---------------------------------------------------------------------------
@kernel
class _PlanKernel:
    """plan 单 kernel：跨卡交换 → 贪心均衡 → 表生成 → token 落位。

    单 launch 多阶段融合，阶段间用 global_sync_all() 做 grid 级同步屏障，块角色由 block_idx 决定。

    阶段（所有 block 都参与每个 grid 级同步屏障）：
      0. Exchange(bi=0) | Count(bi<token_blocks) | GreedyAll(bi=0，UB)
      1. Tables(bi<R) | Pfx(bi<pfx_blocks，按专家切)
      2. Assign(bi<token_blocks，按 chunk 切)
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
        self._r = int(rank_size)
        self._e = int(num_experts)
        self._epn = self._e // self._r
        self._chunk = int(num_rows) // int(num_chunks)
        self._num_chunks = int(num_chunks)
        self._token_blocks = int(token_blocks)
        self._pfx_blocks = int(pfx_blocks)
        self._ws = _ws_layout(self._r, self._e, self._num_chunks)

    def __call__(
        self,
        gm_tpe,
        gm_flag,
        gm_ws,
        gm_topk,
        gm_io,
        ctx: StructRef[CommContext],
    ):
        r, e, nc = self._r, self._e, self._num_chunks
        o = self._ws
        epn = self._epn
        # 中间表共享一个扁平 workspace（dem 为 [R,E] 交换矩阵，其余为 [E,R] 等）
        gm_dem_by_rank = gm_ws[o["dem"] : o["dem"] + e * r,].view(r, e)
        gm_rem = gm_ws[o["rem"] : o["rem"] + e * r,].view(e, r)
        gm_order = gm_ws[o["order"] : o["order"] + e * r,].view(e, r)
        gm_counts = gm_ws[o["counts"] : o["counts"] + nc * e,].view(nc, e)
        gm_pcnt = gm_ws[o["pcnt"] : o["pcnt"] + e * nc,].view(e, nc)
        gm_vid = gm_ws[o["vid"] : o["vid"] + e,]
        gm_ord_rank = gm_ws[o["ord_rank"] : o["ord_rank"] + e * r,].view(e, r)
        gm_ord_end = gm_ws[o["ord_end"] : o["ord_end"] + e * r,].view(e, r)
        gm_jrep = gm_ws[o["jrep"] : o["jrep"] + e,].view(r, self._epn)
        gm_rhome = gm_ws[o["rhome"] : o["rhome"] + r,]
        # 输出三件套共享一个 io 张量：cu [2epn] | etc [R,epn] | vtopk [N]
        io_base = (r + 2) * epn
        gm_cu = gm_io[0 : 2 * epn,]
        gm_etc = gm_io[2 * epn : io_base,].view(r, epn)
        gm_vtopk = gm_io[io_base : io_base + self._chunk * nc,]
        bi = get_block_idx()
        ub_demand = Buffer(MemLoc.UB, dtypes.int32, make_layout((r, e)))

        if bi == dtypes.int64(0):
            self._exchange(ctx, gm_tpe, gm_flag, gm_dem_by_rank, ub_demand)
        if bi < dtypes.int64(self._token_blocks):
            self._count(bi, gm_topk, gm_counts)
        if bi == dtypes.int64(0):
            self._greedy_all(ctx, ub_demand, gm_rem, gm_order, gm_rhome, gm_vid, gm_cu)

        global_sync_all()

        # ---- 阶段 1：jrep/etc 表（block 内循环多 rank）∥ 前缀/查表（按专家） ----
        # 两者都只依赖阶段 0 的 order/counts，写不同输出，同阶段并行。
        # Tables 改为 block 内循环（原生 MoonEP 形态：range(pid, R, num_blocks)），
        # R 可超 block 数——128P 时不再要求 block ≥ R。
        nb = get_block_num()
        self._tables_loop(bi, nb, gm_order, gm_rhome, gm_jrep, gm_etc)
        if bi < dtypes.int64(self._pfx_blocks):
            self._pfx(
                ctx,
                bi,
                gm_dem_by_rank,
                gm_counts,
                gm_rem,
                gm_order,
                gm_pcnt,
                gm_vid,
                gm_ord_rank,
                gm_ord_end,
            )

        global_sync_all()

        # ---- 阶段 2：token 落位（按 chunk） ----
        if bi < dtypes.int64(self._token_blocks):
            self._assign(
                bi,
                gm_topk,
                gm_pcnt,
                gm_ord_rank,
                gm_ord_end,
                gm_jrep,
                gm_vid,
                gm_vtopk,
            )

    # ---------------- 阶段 0：Exchange（单 block 通信） ----------------
    @jit
    def _exchange(self, ctx, gm_tpe, gm_flag, gm_dem_by_rank, ub_demand):
        world_size, num_experts = self._r, self._e
        me = dtypes.int32(ctx.rankId)  # 运行时从 CommContext 取（rank 不进编译键）
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

        for p in cannbotdsl.range(world_size):
            if p != me:
                xfer.put(
                    p,
                    tile_slice(dem_all, (1, num_experts), (me, 0))
                    .peer(p)
                    .view((1, num_experts)),
                    gm_tpe,
                )
                xfer.put(p, tile_slice(ready, (1,), (me,)).peer(p).view((1,)), gm_flag)
                xfer.flush(p)
        ub_tpe = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, num_experts)))
        mem_copy(ub_tpe, gm_tpe.view(1, num_experts))
        mem_copy(tile_slice(dem_all, (1, num_experts), (me, 0)), ub_tpe)
        for r in cannbotdsl.range(world_size):
            if r != me:
                busy_wait_flag(tile_slice(ready, (1,), (r,)), gm_flag, _EVENT_ID)
        mem_copy(ub_demand, tile_slice(dem_all, (world_size, num_experts), (0, 0)))
        mem_copy(gm_dem_by_rank, ub_demand)

    # ---------------- 阶段 0：Count ----------------
    @jit
    def _count(self, bi, gm_topk, gm_counts):
        num_experts, chunk, nc, bd = (
            self._e,
            self._chunk,
            self._num_chunks,
            self._token_blocks,
        )
        ub_topk = Buffer(MemLoc.UB, dtypes.int32, make_layout((chunk,)))
        num_hist_groups = (num_experts + _HISTOGRAM_U8_LANES - 1) // _HISTOGRAM_U8_LANES
        use_vec = chunk % _HISTOGRAM_U8_LANES == 0 and chunk < _HISTOGRAM_MAX_CHUNK

        for k in cannbotdsl.range(nc // bd):
            ci = bi + k * bd
            if const_expr(use_vec):
                ub_hist = Buffer(
                    MemLoc.UB,
                    dtypes.int32,
                    make_layout((num_hist_groups * _HISTOGRAM_U8_LANES,)),
                )
                mem_copy(ub_topk, tile_slice(gm_topk, (chunk,), (ci,)))
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
                    tile_slice(gm_counts, (1, num_experts), (ci, 0)),
                    ub_hist.view(1, num_experts),
                )
            else:
                ub_cnt = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))
                with vf(mode="simt", thread=1):
                    for e in cannbotdsl.range(num_experts):
                        ub_cnt[dtypes.int64(e)] = dtypes.int32(0)
                mem_copy(ub_topk, tile_slice(gm_topk, (chunk,), (ci,)))
                with vf(mode="simt", thread=1):
                    for i in cannbotdsl.range(chunk):
                        e = ub_topk[dtypes.int64(i)]
                        ub_cnt[e] = ub_cnt[e] + dtypes.int32(1)
                mem_copy(
                    tile_slice(gm_counts, (1, num_experts), (ci, 0)),
                    ub_cnt.view(1, num_experts),
                )

    @jit
    def _greedy_all(self, ctx, ub_demand, gm_rem, gm_order, gm_rhome, gm_vid, gm_cu):
        """贪心均衡：demand 矩阵 → rem/order 服务分配 + cu + 快路径 vid。

        ub_demand 初始为 [R,E] 需求矩阵；行/列转置后（view 为 [E,R]）复用为
        rem（各专家由各 rank 服务的行数），避免额外分配——两段生命周期不重叠。
        ub_expert_total 先暂存 per-expert 总需求（用于 rem 初始化），后被
        覆写为 vid 值（快路径时为正、慢路径时为 -1），同样不重叠。
        """
        world_size, num_experts = self._r, self._e
        epn = self._epn
        ub_rem = ub_demand.view(num_experts, world_size)
        ub_order = Buffer(
            MemLoc.UB, dtypes.int32, make_layout((num_experts, world_size))
        )
        ub_loads = Buffer(MemLoc.UB, dtypes.int32, make_layout((world_size,)))
        ub_rhome = Buffer(MemLoc.UB, dtypes.int32, make_layout((world_size,)))
        ub_expert_total = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))
        # cu 与 vid 生命周期重叠（cu 要活到尾部 mem_copy，vid 在其前覆写），
        # 不能别名复用同一 buffer，否则 vid 会写进 cu
        ub_sizes = Buffer(MemLoc.UB, dtypes.int32, make_layout((2 * epn,)))
        # ctx.rankId 须在 vf scope 外读取（datastruct_ref_load 不允许在 vf 内）
        me_id = dtypes.int32(ctx.rankId)

        with vf(mode="simt", thread=_PLAN_INIT_THREADS):
            tid = simt.thread_idx()[0]
            nt = simt.thread_dim()[0]
            for e in cannbotdsl.range(tid, num_experts, nt):
                total = dtypes.int32(0)
                for r in cannbotdsl.range(world_size):
                    total = total + ub_demand[dtypes.int64(r), dtypes.int64(e)]
                ub_expert_total[dtypes.int64(e)] = total
            simt.syncthreads()
            # demand 已全部读出，ub_demand 转置复用为 rem，控制 UB 峰值
            for e in cannbotdsl.range(tid, num_experts, nt):
                home_rank = e // epn
                for r in cannbotdsl.range(world_size):
                    ub_rem[dtypes.int64(e), dtypes.int64(r)] = dtypes.int32(0)
                    ub_order[dtypes.int64(e), dtypes.int64(r)] = _ORDER_UNSERVED
                ub_rem[dtypes.int64(e), home_rank] = ub_expert_total[dtypes.int64(e)]
                ub_order[dtypes.int64(e), home_rank] = _ORDER_HOME
            simt.syncthreads()
            for r in cannbotdsl.range(tid, world_size, nt):
                acc = dtypes.int32(0)
                for e in cannbotdsl.range(num_experts):
                    acc = acc + ub_rem[dtypes.int64(e), dtypes.int64(r)]
                ub_loads[dtypes.int64(r)] = acc
                ub_rhome[dtypes.int64(r)] = dtypes.int32(-1)
            simt.syncthreads()
            if tid < _WARP_SIZE:
                lane = simt.lane_idx()
                cap = dtypes.int32(0)
                for j in range_constexpr((world_size + _WARP_SIZE - 1) // _WARP_SIZE):
                    k = lane + j * _WARP_SIZE
                    v = dtypes.int32(0)
                    if k < world_size:
                        v = ub_loads[dtypes.int64(k)]
                    cap = cap + simt.reduce_add(v)
                cap = cap // world_size
                for _it in cannbotdsl.range(world_size):
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
                    self._fill_sizes_cu(me_id, ub_rem, ub_order, ub_rhome, ub_sizes)
            simt.syncthreads()
            for e in cannbotdsl.range(tid, num_experts, nt):
                home_rank = e // epn
                fast = dtypes.int32(1)
                for r in cannbotdsl.range(world_size):
                    if dtypes.int64(r) != home_rank:
                        if ub_rem[dtypes.int64(e), dtypes.int64(r)] > dtypes.int32(0):
                            fast = dtypes.int32(0)
                if fast != dtypes.int32(0):
                    ub_expert_total[dtypes.int64(e)] = dtypes.int32(
                        home_rank * (2 * epn) + (e % epn)
                    )
                else:
                    ub_expert_total[dtypes.int64(e)] = _VID_SLOW
            simt.syncthreads()
        mem_copy(gm_rem, ub_rem)
        mem_copy(gm_order, ub_order)
        mem_copy(gm_rhome, ub_rhome)
        mem_copy(gm_vid, ub_expert_total)
        mem_copy(gm_cu, tile_slice(ub_sizes, (2 * epn,), (0,)))

    @jit
    def _warp_extreme_idx(self, ub_values, n, lane, maximize):
        """warp 归约极值，并列取最小下标（maximize=True/False 对应 max/min）。"""
        best = dtypes.int32(_GREEDY_NEG_INF if maximize else _GREEDY_POS_INF)
        idx = dtypes.int32(0)
        invalid = dtypes.int32(_GREEDY_NEG_INF if maximize else _GREEDY_POS_INF)
        for j in range_constexpr((n + _WARP_SIZE - 1) // _WARP_SIZE):
            k = lane + j * _WARP_SIZE
            v = invalid
            if k < n:
                v = ub_values[dtypes.int64(k)]
            if const_expr(maximize):
                round_best = simt.reduce_max(v)
            else:
                round_best = simt.reduce_min(v)
            candidate = dtypes.int32(n)
            if (k < n) and (v == round_best):
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
    def _warp_max_rem_idx(self, ub_rem, src_r, n, lane):
        """最大正 rem 的最小专家下标；全零时返回 (-1, 0)。"""
        e_lo = src_r * self._epn
        best = dtypes.int32(0)
        idx = dtypes.int32(-1)
        for j in range_constexpr((n + _WARP_SIZE - 1) // _WARP_SIZE):
            k = lane + j * _WARP_SIZE
            v = dtypes.int32(0)
            if k < n:
                v = ub_rem[dtypes.int64(e_lo + k), dtypes.int64(src_r)]
            round_best = simt.reduce_max(v)
            candidate = dtypes.int32(n)
            if (k < n) and (v == round_best) and (round_best > 0):
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
                            n_copy = dtypes.int32(0)
                            for r in cannbotdsl.range(self._r):
                                if ub_order[
                                    dtypes.int64(e_best), dtypes.int64(r)
                                ] > dtypes.int32(0):
                                    n_copy = n_copy + dtypes.int32(1)
                            ub_order[dtypes.int64(e_best), dtypes.int64(dst_r)] = (
                                n_copy + dtypes.int32(1)
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
    def _fill_sizes_cu(self, me_id, ub_rem, ub_order, ub_rhome, ub_sizes):
        """[2*epn] 槽位行数前缀和 → cu_seqlens。副本槽按源 rank 专家升序压缩填充。"""
        me = me_id  # vf scope 外读取的 ctx.rankId（传入参数）
        epn = self._epn
        home = ub_rhome[dtypes.int64(me)]
        for i in range_constexpr(epn):
            ub_sizes[dtypes.int64(i)] = ub_rem[
                dtypes.int64(me * epn + i), dtypes.int64(me)
            ]
        for i in range_constexpr(epn):
            ub_sizes[dtypes.int64(epn + i)] = dtypes.int32(0)
        if home >= dtypes.int32(0):
            rep_cnt = dtypes.int32(0)
            for i in cannbotdsl.range(epn):
                e = home * epn + i
                if ub_order[dtypes.int64(e), dtypes.int64(me)] >= _ORDER_FIRST_REPLICA:
                    ub_sizes[dtypes.int64(epn) + rep_cnt] = ub_rem[
                        dtypes.int64(e), dtypes.int64(me)
                    ]
                    rep_cnt = rep_cnt + dtypes.int32(1)
        acc = dtypes.int32(0)
        for i in range_constexpr(2 * epn):
            acc = acc + ub_sizes[dtypes.int64(i)]
            ub_sizes[dtypes.int64(i)] = acc

    # ---------------- 阶段 1：Tables ----------------
    @jit
    def _tables_loop(self, bi, nb, gm_order, gm_rhome, gm_jrep, gm_etc):
        """block bi 循环处理 rank bi, bi+nb, bi+2*nb...——R 可超 block 数。"""
        for r in cannbotdsl.range(bi, self._r, nb):
            self._tables(r, gm_order, gm_rhome, gm_jrep, gm_etc)

    def _tables(self, r, gm_order, gm_rhome, gm_jrep, gm_etc):
        epn = self._epn
        ub_jrep_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, epn)))
        ub_etc_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, epn)))

        with vf(mode="simt", thread=1):
            self._fill_jrep_etc(gm_order, gm_rhome, ub_jrep_row, ub_etc_row, r, epn)
        mem_copy(tile_slice(gm_jrep, (1, epn), (r, 0)), ub_jrep_row)
        mem_copy(tile_slice(gm_etc, (1, epn), (r, 0)), ub_etc_row)

    @jit
    def _fill_jrep_etc(self, gm_order, gm_rhome, ub_jrep_row, ub_etc_row, r, epn):
        for j in range_constexpr(epn):
            ub_jrep_row[dtypes.int64(0), dtypes.int64(j)] = dtypes.int32(0)
            ub_etc_row[dtypes.int64(0), dtypes.int64(j)] = dtypes.int32(-1)
        home = simt.l2cache_load(gm_rhome.ptr((dtypes.int64(r),)))
        if home >= dtypes.int32(0):
            rc = dtypes.int32(0)
            for i in cannbotdsl.range(epn):
                e = home * epn + i
                o = simt.l2cache_load(gm_order.ptr((dtypes.int64(e), dtypes.int64(r))))
                if o >= _ORDER_FIRST_REPLICA:
                    ub_jrep_row[dtypes.int64(0), dtypes.int64(i)] = rc
                    ub_etc_row[dtypes.int64(0), rc] = dtypes.int32(e)
                    rc = rc + dtypes.int32(1)

    # ---------------- 阶段 1：Pfx ----------------
    @jit
    def _pfx(
        self,
        ctx,
        bi,
        gm_dem_by_rank,
        gm_counts,
        gm_rem,
        gm_order,
        gm_pcnt,
        gm_vid,
        gm_ord_rank,
        gm_ord_end,
    ):
        world_size, num_experts = self._r, self._e
        nc = self._num_chunks
        e_per = num_experts // self._pfx_blocks
        e_lo = bi * e_per

        pcnt_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, nc)))
        ord_rank_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, world_size)))
        ord_end_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, world_size)))

        for k in cannbotdsl.range(e_per):
            e = e_lo + k
            self._build_pfx_row(
                ctx,
                e,
                gm_dem_by_rank,
                gm_counts,
                gm_rem,
                gm_order,
                gm_vid,
                pcnt_row,
                ord_rank_row,
                ord_end_row,
            )
            mem_copy(tile_slice(gm_pcnt, (1, nc), (e, 0)), pcnt_row)
            mem_copy(tile_slice(gm_ord_rank, (1, world_size), (e, 0)), ord_rank_row)
            mem_copy(tile_slice(gm_ord_end, (1, world_size), (e, 0)), ord_end_row)

    @jit
    def _build_pfx_row(
        self,
        ctx,
        e,
        gm_dem_by_rank,
        gm_counts,
        gm_rem,
        gm_order,
        gm_vid,
        pcnt_row,
        ord_rank_row,
        ord_end_row,
    ):
        world_size = self._r
        me = dtypes.int32(ctx.rankId)  # 运行时取（self-serve 偏好序）
        nc = self._num_chunks
        with vf(mode="simt", thread=1):
            # fast expert（无远程副本）早退：Assign 快路径直读 vid，不消费
            # 本行 pcnt/ord；写回的陈旧行值无害（消费侧只在 vid<0 时读行）
            vid = simt.l2cache_load(gm_vid.ptr((dtypes.int64(e),)))
            if vid < dtypes.int32(0):
                # 段序表先行（偏好序需要 home/副本 host 的段排列）
                for o in cannbotdsl.range(world_size):
                    ord_rank_row[dtypes.int64(0), dtypes.int64(o)] = dtypes.int32(-1)
                for r in cannbotdsl.range(world_size):
                    o = simt.l2cache_load(
                        gm_order.ptr((dtypes.int64(e), dtypes.int64(r)))
                    )
                    if o >= _ORDER_HOME:
                        ord_rank_row[dtypes.int64(0), dtypes.int64(o - 1)] = (
                            dtypes.int32(r)
                        )
                # pcnt 基址 = 拼接序中排在本 rank 之前的需求总和。
                # 偏好序 [home, 副本 host 按段序, 非服务 rank]：serving rank
                # 自己的 token 优先占自己段（本地消化，满倾斜 75%→0% 跨卡）。
                # 段容量/均衡不变量不变；流量单调不增。
                acc_p = dtypes.int32(0)
                seen_me = dtypes.int32(0)
                for o in cannbotdsl.range(world_size):
                    r = ord_rank_row[dtypes.int64(0), dtypes.int64(o)]
                    if r >= dtypes.int32(0) and seen_me == dtypes.int32(0):
                        if r == me:
                            seen_me = dtypes.int32(1)
                        else:
                            acc_p = acc_p + simt.l2cache_load(
                                gm_dem_by_rank.ptr((dtypes.int64(r), dtypes.int64(e)))
                            )
                for r in cannbotdsl.range(world_size):
                    if seen_me == dtypes.int32(0):
                        if r == me:
                            seen_me = dtypes.int32(1)
                        else:
                            o = simt.l2cache_load(
                                gm_order.ptr((dtypes.int64(e), dtypes.int64(r)))
                            )
                            if o == _ORDER_UNSERVED:
                                acc_p = acc_p + simt.l2cache_load(
                                    gm_dem_by_rank.ptr(
                                        (dtypes.int64(r), dtypes.int64(e))
                                    )
                                )
                acc_c = dtypes.int32(0)
                for c in range_constexpr(nc):
                    pcnt_row[dtypes.int64(0), dtypes.int64(c)] = acc_p + acc_c
                    acc_c = acc_c + simt.l2cache_load(
                        gm_counts.ptr((dtypes.int64(c), dtypes.int64(e)))
                    )
                acc_o = dtypes.int32(0)
                for o in cannbotdsl.range(world_size):
                    rnk = ord_rank_row[dtypes.int64(0), dtypes.int64(o)]
                    if rnk >= dtypes.int32(0):
                        acc_o = acc_o + simt.l2cache_load(
                            gm_rem.ptr((dtypes.int64(e), dtypes.int64(rnk)))
                        )
                    ord_end_row[dtypes.int64(0), dtypes.int64(o)] = acc_o

    # ---------------- 阶段 2：Assign ----------------
    @jit
    def _assign(
        self,
        bi,
        gm_topk,
        gm_pcnt,
        gm_ord_rank,
        gm_ord_end,
        gm_jrep,
        gm_vid,
        gm_vtopk,
    ):
        world_size, num_experts = self._r, self._e
        nc, bd = self._num_chunks, self._token_blocks

        # buf_id 预算（VEC 上限 32）：ord_end 是二分热表留 UB；
        # ord_rank/jrep 每次 relocate 只读一次，走 L2 直读省 2 个 buf_id
        self._ord_end = Buffer(
            MemLoc.UB, dtypes.int32, make_layout((num_experts, world_size))
        )
        mem_copy(self._ord_end, gm_ord_end)
        self._vid = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))
        mem_copy(self._vid, gm_vid)

        for k in cannbotdsl.range(nc // bd):
            self._assign_chunk(
                bi + k * bd, gm_topk, gm_pcnt, gm_vtopk, gm_ord_rank, gm_jrep
            )

    @jit
    def _assign_chunk(self, ci, gm_topk, gm_pcnt, gm_vtopk, gm_ord_rank, gm_jrep):
        num_experts = self._e
        chunk = self._chunk
        ub_topk = Buffer(MemLoc.UB, dtypes.int32, make_layout((chunk,)))
        ub_vtopk = Buffer(MemLoc.UB, dtypes.int32, make_layout((chunk,)))
        ub_slow = Buffer(MemLoc.UB, dtypes.int32, make_layout((chunk,)))
        ub_nslow = Buffer(MemLoc.UB, dtypes.int32, make_layout((1,)))
        ub_pos = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))
        ub_seg_end = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))
        ub_cur_vid = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))

        with vf(mode="simt", thread=1):
            for e in cannbotdsl.range(num_experts):
                if self._vid[dtypes.int64(e)] < dtypes.int32(0):
                    ub_pos[dtypes.int64(e)] = simt.l2cache_load(
                        gm_pcnt.ptr((dtypes.int64(e), dtypes.int64(ci)))
                    )
                    ub_seg_end[dtypes.int64(e)] = dtypes.int32(0)
        mem_copy(ub_topk, tile_slice(gm_topk, (chunk,), (ci,)))
        vec_end = chunk - chunk % _I32_LANES
        if const_expr(vec_end > 0):
            with vf(mode="simd"):
                mask = full_mask(32)  # int32 全 lane
                cursor = vstore_unalign_begin(ub_slow)
                for off in range_constexpr(0, vec_end, _I32_LANES):
                    indices = vreinterpret(vload(ub_topk, off), dtypes.uint32)
                    values = vgather(self._vid, indices, mask=mask)
                    vstore(ub_vtopk, off, values, mask)
                    slow = vlts(values, 0, mask=mask)
                    positions = varange(off, dtypes.int32)
                    squeezed = vsqueeze_and_storeunalign_init(positions, mask=slow)
                    vsqueeze_and_storeunalign(ub_slow, 0, squeezed, cursor)
                vsqueeze_and_storeunalign_finalize(ub_slow, 0, cursor)
                vstorealign_squeeze_status(ub_nslow, 0)
            vec_sync_all()
        with vf(mode="simt", thread=1):
            if const_expr(vec_end > 0):
                nslow = (
                    ub_nslow[dtypes.int64(0)] // _INT32_BYTES
                )  # squeeze 状态字节数→int32 计数
            else:
                nslow = dtypes.int32(0)
            for i in range_constexpr(vec_end, chunk):
                e = ub_topk[dtypes.int64(i)]
                fv = self._vid[dtypes.int64(e)]
                ub_vtopk[dtypes.int64(i)] = fv
                if fv < dtypes.int32(0):
                    ub_slow[dtypes.int64(nslow)] = dtypes.int32(i)
                    nslow = nslow + dtypes.int32(1)
            for j in cannbotdsl.range(nslow):
                i = ub_slow[dtypes.int64(j)]
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
        mem_copy(tile_slice(gm_vtopk, (chunk,), (ci,)), ub_vtopk)

    @jit
    def _relocate(self, e, pos, gm_ord_rank, gm_jrep):
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
        vid = dst_rank * dtypes.int32(2 * epn) + (e % dtypes.int32(epn))
        if seg_idx >= dtypes.int32(1):
            vid = (
                dst_rank * dtypes.int32(2 * epn)
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
    """分块：chunk 数尽量逼近核数上限；chunk 取 2 的幂并降档到 N 的整除因子
    （保证 chunk 数 × chunk == N，无尾块）；下限 _CHUNK_MIN 防止每块状态重建
    占比过高。
    """
    cap = resident_blocks(_MAX_PLAN_BLOCKS)
    want = max(_CHUNK_MIN, -(-num_rows // cap))
    chunk = 1 << (want - 1).bit_length() if want > 1 else 1
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
    """分块与专家块一次算全（traced host / workspace 分配 / 编译锁内共用一份）。

    不做 lru_cache：块数回归测试 monkeypatch _RESIDENT_BLOCKS，缓存会把
    钉住的核数焊死。
    """
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
    num_experts = gm_tpe.shape[0]
    world_size = (
        _R  # trace 期烘焙的组内 rank_size（rank 不再烘焙——ctx.rankId 运行时取）
    )
    num_rows = gm_topk.shape[0]  # 本 rank topk 扁平视图，直接作 Cnt/Asg 输入
    blocks = _plan_blocks(num_rows, num_experts)
    # 单 launch：Exchange/Count/GreedyAll → Tables/Pfx → Assign，两次 grid 级同步屏障；
    # dem 折入 workspace、输出合并。
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

# 零占位张量（dst_n/src_n）进程级缓存：全零、只读契约、kernel 不写——
# 免每步 2 次 alloc+H2D；调用方不得写入（接口文档：零占位不计算）
_PLACEHOLDER_CACHE = {}

# epoch 值池：预生成 1..cap，调用时取当前值的独立标量视图。各轮地址不同，
# 不存在复用缓冲的流序竞争，同时消除每轮 torch.full 的 alloc+H2D 开销。
_EPOCH_POOLS = []
_EPOCH_POOL_CAP = 0
_EPOCH_POOL_MIN_CAP = 1 << 16


def _epoch_flag(epoch):
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


# workspace 打包：中间表放进一个扁平 int32 张量，program 只收 1 个参数。
# 每段按 16 元素（64B）对齐；各表由 kernel 全量覆写，同 shape 缓存复用安全
# （Pfx 的 fast expert 行写回陈旧值，但消费侧只在 vid<0 时读 pcnt/ord 行）。
_WS_ALIGN = 16


def _ws_layout(rank_size, num_experts, nc):
    """workspace 布局：{表名: 元素偏移} + total（host/kernel 共用的编译期常量）。"""
    r, e = rank_size, num_experts
    sizes = (
        ("dem", e * r),
        ("rem", e * r),
        ("order", e * r),
        ("counts", nc * e),
        ("pcnt", e * nc),
        ("vid", e),
        ("ord_rank", e * r),
        ("ord_end", e * r),
        ("jrep", e),
        ("rhome", r),
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


def _get_plan_program(num_tokens, top_k, num_experts, world_size):
    """按 (S,K,E,R,块数) 键控的编译产物缓存。

    rank 不再进键：kernel 内通过 ctx.rankId 运行时获取，
    同 shape 所有 rank 共用同一 program——R=32 从 32 次编译降到 1 次。
    R 经模块级 _R 烘焙（shape 维度，非 rank 身份），仍须进键。
    """
    # 块数进键：token/Pfx block 数是 kernel 编译期常量（角色分派），核数不同的机器
    # 不能复用同一 program
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
            TensorSpec((2 * epn + world_size * epn + num_rows,), i32),  # 合并输出
            StructRef[CommContext],
        )
        _PLAN_PROGRAMS[key] = prog
        return prog


def moonep_plan(context, tokens_per_expert, topk_experts_sk, *, verify=False):
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
    verify=True 时改为立即同步校验（调试/验收口径，见函数体注释）。
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
    cur_stream = sync_transport_stream(context)  # 跨流护栏（握手状态机要求调用按序）
    world_size = context.rank_size
    # Tables 已改为 block 内循环（原生 MoonEP 形态），R 不再受 AIV 核驻数限制。
    # 剩余硬约束：R ≤ MAX_R=64（dem/rhome 表维度）；128P 需扩 MAX_R 并做 GM 分块。
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
    io = torch.empty(
        2 * epn + world_size * epn + num_rows, dtype=torch.int32, device="npu"
    )
    cu_seqlens = io[: 2 * epn]
    experts_to_copy = io[2 * epn : 2 * epn + world_size * epn].view(world_size, epn)
    src_n = _zero_placeholder(num_rows)
    vtopk = io[2 * epn + world_size * epn :]
    global _EPOCH
    _EPOCH += 1
    # _R 的烘焙在 _get_plan_program 锁内（与编译原子）；wrapper 只管 epoch
    gm_flag = _epoch_flag(_EPOCH)  # 预生成值池中的独立标量视图
    # plan workspace 单槽缓存：dem 与中间表打包扁平张量，各 kernel 全量覆写，
    # 同 shape 复用；shape 变化时按新布局重新分配。
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
    # Cnt/Asg 只消费本 rank 的行：直接把 [S,K] 展平视图当输入，省掉一次
    # 往 workspace 的整行 copy_（NPU 上该 copy_ 实测是 ~0.8ms 的慢路径）
    _get_plan_program(dims.num_tokens, dims.top_k, dims.num_experts, dims.world_size)(
        tokens_per_expert,
        gm_flag,
        tables,
        topk_experts_sk.reshape(-1),
        io,
        context.ctx_tensor,
    )
    record_transport_event(
        context, stream=cur_stream
    )  # 跨流护栏 fork 点（只等我方算子，不等主流尾部）
    # 溢出检查延迟执行（返回路径零同步，不阻塞 dispatch 发射）
    context.pending_overflow = (cu_seqlens, num_rows)
    context.last_plan_shape = (num_tokens, top_k)  # 供 prefetch/reduce 的调用顺序守卫
    if verify:
        # 调试/验收口径：立即同步读回尾值，不变量破坏当场熔断——先于任何
        # 下游消费算子（消除"坏布局先被消费一次才报警"的窗口）。代价是
        # 一次流同步（流水线下等待在飞工作），生产路径勿开。
        check_pending_overflow(context)
    return dst_n, cu_seqlens, experts_to_copy, src_n, vtopk


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
        """预编译推荐配置（S=8192,K=16,E=224,R=8）的 plan 程序。

        rank 不进键——一份程序全 rank 共用。其余 shape 运行时 JIT 编译。"""
        _get_plan_program(8192, 16, 224, 8)
