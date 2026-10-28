# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""reduce_grad：副本专家权重梯度跨 rank 归并回 home rank 并清零预取槽（训练反向）。

机制（单 kernel 两段 + 栅栏 + kernel 内动态 chunk 循环 + 清单运行时化）：
  发送段（每 block 一个目的 peer）：把该 home 的副本梯度行整行 put 到其
    staging[me][j]（单次大块传输，无 host 分块循环），同通道 flag 保序通知，
    最后 flush 自己的通道。
  —— global_sync_all() ——
  归并段（每 block 认领 home 行 i % nblk == bi）：先轮询该行全部 sender 的
    flag（数据就位），再按 chunk 动态循环在 UB 累加（vadd）后做**单次**
    mem_copy(atomic_add=True)（staging 免清零：发送侧 put 全量覆写后才置 flag）；
    同时按条目跨 block 分派把本地预取槽梯度清零（槽位复用契约，必须在所有
    put 完成之后）。

清单为定长数组 + 运行时计数（send: epn 槽 + cnt + 唯一 home；recv:
epn×(R-1) 定步长 + 逐行 cnt）——编译键只含形状/布局，**与路由结构彻底解耦**
（随机路由每步变化也不会触发重编译；早期版本把 recv_groups 结构烘进编译键，
随机路由下约两成步长触发 6.6s 重编译）。

staging 为全宽 (world_size, epn, W) 且组间复用同一块（组间自旋 barrier 串行，
占用与组数解耦：3 组生产配置 45GB→15GB）：各 chunk 有独立 slice，无块间
复用竞态——单次 program launch 服务全部列块（无 host 循环）。
布局超出 HCCL 对称缓冲区时显式报错（防越界静默错算）。
"""

import threading
from typing import NamedTuple

import cannbotdsl
import torch
from cannbotdsl import TensorSpec, dtypes, get_block_idx, get_block_num
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.distributed import (
    CommContext,
    SymBuffer,
    TransferRole,
    create_transfer_context,
)
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.reg import (
    full_mask,
    update_mask,
    vadd,
    vcast,
    vdups,
    vload,
    vload_unpack,
    vstore,
    vstore_pack,
)
from cannbotdsl.ops.reg.reg_defs import PackMode, UnpackMode
from cannbotdsl.ops.sync import global_sync_all
from cannbotdsl.tensor import MemLoc, tile_slice
from cannbotdsl.types.datastruct import StructRef
from moonep_context import (
    HEAP_BASE,
    LAYOUT_TAIL_BYTES,
    busy_wait_flag,
    check_heap_capacity,
    finish_op,
    normalize_group_list,
    pick_weight_chunk,
    reduce_stg_off,
    resident_blocks,
    scalar_read_fence_post,
    sym_barrier,
    scalar_read_fence_pre,
    sync_transport_stream,
)

_EVENT_ID = 6
_REDUCE_CHUNK_LIMIT = 16384  # 归并分块粒度上限（元素/块，动态循环用）
# 发送侧清零用的 UB 缓冲（合并 kernel 后要与 recv 的三块 chunk 缓冲共存：
# 3×64KB + 16KB < 256KB；清零不是带宽关键路径，小块多迭代即可）
_ZERO_CHUNK = 4096
_EPOCH = 0  # 进程内单调调用序号（epoch flag）
VL = 64


def _reduce_sym_bufs(ctx, world_size, epn, w_elems, stg_off):
    """staging (world_size, epn, w_elems) + flags (world_size, epn) int32 对称区视图。

    staging dtype 跟随 grad 视图（bf16 时传输/占用减半，UB 内仍 fp32 累加保精度；
    容量公式同步按 dtype 计算，flags 区固定在 staging 之前，全 rank 公式一致）。
    """
    stg = SymBuffer(
        ctx,
        shape=(world_size, epn, w_elems),
        dtype=dtypes.bfloat16 if _GRAD_BF16 else dtypes.float32,
        rank_size=world_size,
        base_offset=stg_off,
    )
    flags = SymBuffer(
        ctx,
        shape=(world_size, epn),
        dtype=dtypes.int32,
        rank_size=world_size,
        base_offset=_FLAGS_OFF,
    )
    return stg, flags


@kernel
class _MoonepReduceGradKernel:
    """reduce_grad 单 kernel：发送（put+flag+清槽）→ 栅栏 → home 侧轮询+归并。

    原为两次 launch（send / recv 分开），合并后省一次下发，两段之间用
    global_sync_all() 代替 stream 顺序；xfer 只出现在发送段、与重 vf 的归并段
    用栅栏隔离（与 plan 的 Exchange 同款结构，实测可编过）。

    发送侧：副本梯度整行 put 到 home staging + flag 通知 + 本地预取槽清零。
    清单运行时化：gm_sj 为 epn 定长数组（前 cnt 项有效），gm_scnt 为
    运行时计数，gm_shome 标记唯一发送 home——编译键不再含清单结构。
    """

    def __init__(self, rank, rank_size, epn, w_elems, stg_off, chunk):
        self._me = int(rank)
        self._r = int(rank_size)
        self._epn = int(epn)
        self._w = int(w_elems)
        self._stg_off = int(stg_off)
        self._chunk = int(chunk)

    def __call__(
        self,
        gm_rm,
        gm_grad,
        gm_flag,
        ctx: StructRef[CommContext],
    ):
        # 清单打包成单张 gm_rm（host 下发按参数个数线性收费）
        gm_sj, gm_scnt, gm_shome, gm_rs, gm_rj, gm_rcnt = _manifest_views(
            gm_rm, self._epn, self._r
        )
        blk = get_block_idx()
        # 发送段按目的 peer 分块（每个 peer 一个 block，flush 只碰自己的通道——
        # 同一 peer 被多个 block 并发 flush 会挂，早期 8 块按条目分派就这样挂过）
        if blk < dtypes.int64(self._r):
            self._send(ctx, gm_sj, gm_scnt, gm_shome, gm_grad, gm_flag, blk)
        global_sync_all()
        # 清本地预取槽必须在所有 put 完成之后（数据源是这些槽行）
        self._clear_slots(gm_sj, gm_scnt, gm_grad, blk)
        self._recv(ctx, gm_rs, gm_rj, gm_rcnt, gm_grad, gm_flag)

    @jit
    def _send(self, ctx, gm_sj, gm_scnt, gm_shome, gm_grad, gm_flag, blk):
        world_size, epn = self._r, self._epn
        me = self._me
        w_elems = self._w

        stg, flags = _reduce_sym_bufs(ctx, world_size, epn, w_elems, self._stg_off)

        ub_xfer = Buffer(MemLoc.UB, (512,), dtypes.uint8)
        xfer = create_transfer_context(ctx, ub_xfer, role=TransferRole.VEC)

        scalar_read_fence_pre(_EVENT_ID)  # 与 busy_wait_flag 同款完整握手
        ub_scnt = Buffer(MemLoc.UB, (1,), dtypes.int32)
        ub_shome = Buffer(MemLoc.UB, (1,), dtypes.int32)
        mem_copy(ub_scnt, gm_scnt)
        mem_copy(ub_shome, gm_shome)
        ub_sj = Buffer(MemLoc.UB, (epn,), dtypes.int32)
        mem_copy(ub_sj, gm_sj)
        scalar_read_fence_post(_EVENT_ID)  # 标量读前等清单拷贝落地
        cnt_send = ub_scnt[dtypes.int64(0)]  # 运行时标量

        if cnt_send > dtypes.int32(0):
            # 一卡一源：本 rank 的全部副本槽只发往唯一 home block。
            if dtypes.int32(blk) == ub_shome[dtypes.int64(0)]:
                for i in cannbotdsl.range(cnt_send):
                    j = ub_sj[i]
                    xfer.put(
                        dtypes.int64(blk),
                        tile_slice(stg, (1, 1, w_elems), (me, j, 0)).peer(
                            dtypes.int64(blk)
                        ),
                        tile_slice(gm_grad, (1, w_elems), (epn + j, 0)),
                    )
                for i in cannbotdsl.range(cnt_send):
                    j = ub_sj[i]
                    xfer.put(
                        dtypes.int64(blk),
                        tile_slice(flags, (1, 1), (me, j)).peer(dtypes.int64(blk)),
                        gm_flag,
                    )
                xfer.flush(dtypes.int64(blk))

    @jit
    def _clear_slots(self, gm_sj, gm_scnt, gm_grad, blk):
        """清零本地预取槽梯度行（栅栏之后，按条目跨 block 分派）。"""
        epn = self._epn
        w_elems = self._w
        chunk = self._chunk
        grad_dtype = dtypes.bfloat16 if _GRAD_BF16 else dtypes.float32
        ub_zero = Buffer(MemLoc.UB, (1, _ZERO_CHUNK), grad_dtype)
        with vf(mode="simd"):
            if _GRAD_BF16:
                seg16 = update_mask(VL, elem_bits=16)[0]
                vz = vdups(0.0, grad_dtype, mask=seg16)
                for off in range_constexpr(0, _ZERO_CHUNK, VL):
                    vstore(ub_zero, off, vz, seg16)
            else:
                full = full_mask()
                vz = vdups(0.0, grad_dtype, mask=full)
                for off in range_constexpr(0, _ZERO_CHUNK, VL):
                    vstore(ub_zero, off, vz, full)

        ub_scnt = Buffer(MemLoc.UB, (1,), dtypes.int32)
        mem_copy(ub_scnt, gm_scnt)
        ub_sj = Buffer(MemLoc.UB, (epn,), dtypes.int32)
        mem_copy(ub_sj, gm_sj)
        scalar_read_fence_post(_EVENT_ID)
        cnt_send = ub_scnt[dtypes.int64(0)]
        zc = _ZERO_CHUNK if chunk % _ZERO_CHUNK == 0 else chunk
        z_cols = w_elems // zc
        nblk = dtypes.int64(get_block_num())
        b = dtypes.int64(blk)
        for i in cannbotdsl.range(b, dtypes.int64(epn), nblk):
            if i < dtypes.int64(cnt_send):
                j = ub_sj[i]
                for c in cannbotdsl.range(z_cols):
                    mem_copy(
                        tile_slice(gm_grad, (1, zc), (epn + j, c)),
                        tile_slice(ub_zero, (1, zc), (0, 0)),
                    )

    @jit
    def _recv(self, ctx, gm_rs, gm_rj, gm_rcnt, gm_grad, gm_flag):
        """home 侧：逐行轮询 flag → 行内 staging UB 累加 → 每 chunk 单次 atomic。

        全宽 staging (world_size, epn, W)：各 chunk 有独立 slice，接收侧无需等发送侧
        "上一块消费完"——一次轮询全行 flag 后，chunk 循环内独立搬运/累加。
        全部循环为运行时界动态循环（cannbotdsl.range），清单经定长数组 + 逐行
        计数传入——编译键与路由结构解耦（每 shape 只编一次）。

        轮询的 flag/epoch UB 缓冲在入口分配一次、全行共享（VEC buf_id 预算）。
        """

        world_size, epn = self._r, self._epn
        w_elems = self._w
        chunk = self._chunk
        n_chunks = w_elems // chunk
        stride = world_size - 1

        stg, flags = _reduce_sym_bufs(ctx, world_size, epn, w_elems, self._stg_off)

        # staging 免清零：发送侧 put 全量覆写该行后才置 flag
        scalar_read_fence_pre(_EVENT_ID)  # 同款握手
        self._rcnt = Buffer(MemLoc.UB, (epn,), dtypes.int32)
        mem_copy(self._rcnt, gm_rcnt)
        self._rs = Buffer(MemLoc.UB, (epn * stride,), dtypes.int32)
        mem_copy(self._rs, gm_rs)
        self._rj = Buffer(MemLoc.UB, (epn * stride,), dtypes.int32)
        mem_copy(self._rj, gm_rj)
        scalar_read_fence_post(_EVENT_ID)  # rcnt/rs/rj 读取前等拷贝落地
        grad_dtype = dtypes.bfloat16 if _GRAD_BF16 else dtypes.float32
        self._acc = Buffer(
            MemLoc.UB, (1, chunk), dtypes.float32
        )  # 恒 fp32（累加精度契约）
        self._tmp = Buffer(MemLoc.UB, (1, chunk), grad_dtype)  # 跟随 staging
        self._cur = Buffer(MemLoc.UB, (1, chunk), grad_dtype)  # bf16 写回读-改-写用
        self._flag = Buffer(MemLoc.UB, (1,), dtypes.int32)
        self._epoch = Buffer(MemLoc.UB, (1,), dtypes.int32)
        blk = dtypes.int64(get_block_idx())
        nblk = dtypes.int64(get_block_num())
        for i in cannbotdsl.range(blk, dtypes.int64(epn), nblk):
            cnt_i = self._rcnt[i]
            if cnt_i > dtypes.int32(0):
                self._poll_row(i, cnt_i, flags, gm_flag)
                self._accumulate_row(i, cnt_i, stg, gm_grad, n_chunks)

    @jit
    def _poll_row(self, i, cnt_i, flags, gm_flag):
        """先轮询行内全部 sender 的 flag（数据全部就位后再开始搬运）。

        exact=True（== 精确匹配）：staging flag 槽在 WORLD 组共享堆上跨用例
        残留（前置 prefetch/多槽用例的权重编码位型当 int32 为大正数），>=
        会被残留提前放行而读到在途旧数据。reduce 有组间 barrier 充当读完成
        回执，== 无被 peer 下一轮覆盖的死锁风险（详见
        `moonep_context.busy_wait_flag`）。
        """
        stride = self._r - 1
        for s in cannbotdsl.range(cnt_i):
            view = tile_slice(
                flags,
                (1, 1),
                (self._rs[i * stride + s], self._rj[i * stride + s]),
            )
            busy_wait_flag(
                view, gm_flag, _EVENT_ID, self._flag, self._epoch, exact=True
            )

    @jit
    def _accumulate_row(self, i, cnt_i, stg, gm_grad, n_chunks):
        """chunk 动态循环：全宽 staging 各块独立 slice，无需块间同步。

        bf16 视图时：staging 读为 bf16（传输减半），UB 内 cvt→fp32 累加保精度；
        写回走读-改-写（recv 独占 home 行，无需 atomic）——atomic_add 仅 fp32 路径。
        """
        stride = self._r - 1
        chunk = self._chunk
        for c in cannbotdsl.range(n_chunks):
            self._load_first_sender(i, c, chunk, stg)
            for s in cannbotdsl.range(1, cnt_i):
                mem_copy(
                    self._tmp,
                    tile_slice(
                        stg,
                        (1, 1, chunk),
                        (
                            self._rs[i * stride + s],
                            self._rj[i * stride + s],
                            c,
                        ),
                    ),
                )
                with vf(mode="simd"):
                    full = full_mask()
                    seg = update_mask(VL, elem_bits=32)[0]  # bf16 载体按 32 位粒度
                    for off in cannbotdsl.range(0, chunk, VL):
                        va = vload(self._acc, off)
                        if _GRAD_BF16:
                            vb = vload_unpack(
                                self._tmp, off, unpack_mode=UnpackMode.B16_TO_B32
                            )
                            vb = vcast(vb, dtypes.float32, mask=seg)
                            vstore(self._acc, off, vadd(va, vb, mask=seg), mask=seg)
                        else:
                            vb = vload(self._tmp, off)
                            vstore(self._acc, off, vadd(va, vb, mask=full), mask=full)
            self._writeback_row(i, c, chunk, gm_grad)

    @jit
    def _load_first_sender(self, i, c, chunk, stg):
        """行内首 sender 的 chunk 装入 acc（bf16 staging 时 cvt 到 fp32）。"""
        if _GRAD_BF16:
            mem_copy(
                self._tmp,
                tile_slice(
                    stg,
                    (1, 1, chunk),
                    (self._rs[i * (self._r - 1)], self._rj[i * (self._r - 1)], c),
                ),
            )
            with vf(mode="simd"):
                seg = update_mask(VL, elem_bits=32)[0]
                for off in cannbotdsl.range(0, chunk, VL):
                    vb = vload_unpack(self._tmp, off, unpack_mode=UnpackMode.B16_TO_B32)
                    vstore(
                        self._acc, off, vcast(vb, dtypes.float32, mask=seg), mask=seg
                    )
        else:
            mem_copy(
                self._acc,
                tile_slice(
                    stg,
                    (1, 1, chunk),
                    (self._rs[i * (self._r - 1)], self._rj[i * (self._r - 1)], c),
                ),
            )

    @jit
    def _writeback_row(self, i, c, chunk, gm_grad):
        """归并结果写回 home 行：fp32 路径单次 atomic_add；bf16 路径读-改-写
        （读 home 行 bf16 → fp32 与 acc 相加 → cvt 回 bf16 写回）。"""
        if _GRAD_BF16:
            mem_copy(
                self._cur,
                tile_slice(gm_grad, (1, chunk), (i, c)),
            )
            with vf(mode="simd"):
                seg = update_mask(VL, elem_bits=32)[0]
                for off in cannbotdsl.range(0, chunk, VL):
                    va = vload(self._acc, off)
                    vc = vload_unpack(self._cur, off, unpack_mode=UnpackMode.B16_TO_B32)
                    vc = vcast(vc, dtypes.float32, mask=seg)
                    vsum = vadd(va, vc, mask=seg)
                    vstore_pack(
                        self._cur,
                        off,
                        vcast(vsum, dtypes.bfloat16, mask=seg),
                        seg,
                        pack_mode=PackMode.B32_TO_B16,
                    )
            mem_copy(
                tile_slice(gm_grad, (1, chunk), (i, c)),
                self._cur,
            )
        else:
            mem_copy(
                tile_slice(gm_grad, (1, chunk), (i, c)),
                self._acc,
                atomic_add=True,
            )


_STG_OFF = 0  # wrapper 在编译锁内写入（trace 期模块级常量）
_FLAGS_OFF = 0  # flags 区基址：窗口池之后、staging 之前（位置不随 grad dtype 变）
_ME = 0  # 组内 rank（编译锁内写入）
_R = 1  # 组内 rank_size（均须进 program 键）
_CHUNK = 1  # UB 分块粒度（编译锁内写入，进 program 键）
_GRAD_BF16 = False  # grad 视图 dtype（编译锁内写入，进 program 键）
_BLOCKS = 28  # 实际启动块数（编译锁内写入，进 program 键）
REDUCE_BLOCKS = 28  # 归并阶段 block 数经验上限（epn 上界；实际取 min(核数, 上限)）


@host
def _reduce_grad_host(
    gm_rm,
    gm_grad,
    gm_flag,
    ctx: StructRef[CommContext],
):
    epn2, w_elems = gm_grad.shape
    # 单 kernel：发送（按目的 peer 分块）→ global_sync_all 栅栏 → 清槽 + 归并
    # （按 home 行分块）。xfer 只在发送段、且与重 vf 段用栅栏隔离，编译/运行
    # 均正常（与 plan 的 Exchange 同款结构）。
    _MoonepReduceGradKernel(
        _ME,
        _R,
        epn2 // 2,
        w_elems,
        _STG_OFF,
        _CHUNK,
    )[_BLOCKS](
        gm_rm,
        gm_grad,
        gm_flag,
        ctx,
    )


@kernel
class _MoonepReduceManifestKernel:
    """设备侧生成发送/接收清单（替代 host 侧 etc 扫描）。

    单 block 扫描 etc [R, epn]：
    - 发送：本 rank 承载的副本槽 j → (唯一 home=e//epn, j)，前 scnt 项有效；
    - 接收：本 rank 真实专家 e=me*epn+i 的远端副本来源 → 行 i 占
      [i*(R-1), i*(R-1)+rcnt[i])，行号即 home 行。
    输出定长数组 + 运行时计数（编译键与路由结构解耦）；host 侧零回读。
    """

    def __init__(self, rank, rank_size, epn):
        self._me = int(rank)
        self._r = int(rank_size)
        self._epn = int(epn)

    def __call__(self, gm_etc, gm_rm):
        world_size, epn, me = self._r, self._epn, self._me
        gm_sj, gm_scnt, gm_shome, gm_rs, gm_rj, gm_rcnt = _manifest_views(
            gm_rm, epn, world_size
        )
        stride = world_size - 1
        ub_etc = Buffer(MemLoc.UB, (world_size, epn), dtypes.int32)
        mem_copy(ub_etc, gm_etc)
        ub_sj = Buffer(MemLoc.UB, (epn,), dtypes.int32)
        ub_scnt = Buffer(MemLoc.UB, (1,), dtypes.int32)
        ub_shome = Buffer(MemLoc.UB, (1,), dtypes.int32)
        ub_rs = Buffer(MemLoc.UB, (epn * stride,), dtypes.int32)
        ub_rj = Buffer(MemLoc.UB, (epn * stride,), dtypes.int32)
        ub_rcnt = Buffer(MemLoc.UB, (epn,), dtypes.int32)

        with vf(mode="simt", thread=1):
            ub_scnt[dtypes.int64(0)] = dtypes.int32(0)
            ub_shome[dtypes.int64(0)] = dtypes.int32(-1)
            for j in cannbotdsl.range(epn):
                e = ub_etc[dtypes.int64(me), dtypes.int64(j)]
                if e >= dtypes.int32(0):
                    c = ub_scnt[dtypes.int64(0)]
                    ub_sj[c] = dtypes.int32(j)
                    if c == dtypes.int32(0):
                        ub_shome[dtypes.int64(0)] = e // dtypes.int32(epn)
                    ub_scnt[dtypes.int64(0)] = c + dtypes.int32(1)
            for k in cannbotdsl.range(epn * stride):
                ub_rs[dtypes.int64(k)] = dtypes.int32(-1)
                ub_rj[dtypes.int64(k)] = dtypes.int32(-1)
            for i in cannbotdsl.range(epn):
                ub_rcnt[dtypes.int64(i)] = dtypes.int32(0)
            for r in range_constexpr(world_size):
                if const_expr(r != me):
                    self._scan_peer_row(r, ub_etc, ub_rs, ub_rj, ub_rcnt)
        mem_copy(gm_sj, ub_sj)
        mem_copy(gm_scnt, ub_scnt)
        mem_copy(gm_shome, ub_shome)
        mem_copy(gm_rs, ub_rs)
        mem_copy(gm_rj, ub_rj)
        mem_copy(gm_rcnt, ub_rcnt)

    @jit
    def _scan_peer_row(self, r, ub_etc, ub_rs, ub_rj, ub_rcnt):
        """单 peer 的 etc 行扫描：命中本地专家区间的条目登记为该专家的副本来源。"""
        epn, me = self._epn, self._me
        stride = self._r - 1
        for j in cannbotdsl.range(epn):
            e = ub_etc[dtypes.int64(r), dtypes.int64(j)]
            if e >= dtypes.int32(me * epn):
                if e < dtypes.int32((me + 1) * epn):
                    i = e - dtypes.int32(me * epn)
                    rc = ub_rcnt[i]
                    ub_rs[i * stride + rc] = dtypes.int32(r)
                    ub_rj[i * stride + rc] = dtypes.int32(j)
                    ub_rcnt[i] = rc + dtypes.int32(1)


@host
def _reduce_manifest_host(gm_etc, gm_rm):
    world_size, epn = gm_etc.shape
    _MoonepReduceManifestKernel(_ME, world_size, epn)[1](gm_etc, gm_rm)


_REDUCE_MANIFEST_PROGRAMS = {}


def _get_reduce_manifest_program(epn, world_size, me):
    """清单 program：(epn, rank_size, me) 键控（_ME 烘焙与编译在锁内原子完成）。"""
    key = (epn, world_size, me)
    with _REDUCE_LOCK:
        prog = _REDUCE_MANIFEST_PROGRAMS.get(key)
        if prog is None:
            global _ME
            _ME = me
            i32 = dtypes.int32
            prog = cannbotdsl.compile(
                _reduce_manifest_host,
                TensorSpec((world_size, epn), i32),
                TensorSpec((_manifest_layout(epn, world_size)["total"],), i32),
            )
            _REDUCE_MANIFEST_PROGRAMS[key] = prog
    return prog


def _manifest_layout(epn, world_size):
    """清单缓冲区段的元素偏移（编译期常量，host/kernel 共用）。

    段序：sj[epn] | scnt[1] | shome[1] | rs/rj[epn·(R-1)] | rcnt[epn]
    打包成一块，kernel 只收 1 个参数。shome 为唯一发送 home，无副本写 -1。
    """
    stride = world_size - 1
    n_rs = epn * stride
    return {
        "sj": 0,
        "scnt": epn,
        "shome": epn + 1,
        "rs": epn + 2,
        "rj": epn + 2 + n_rs,
        "rcnt": epn + 2 + 2 * n_rs,
        "total": 2 * epn + 2 + 2 * n_rs,
    }


def _manifest_views(gm_rm, epn, world_size):
    """清单各段的切片视图（元组索引形式 t[a:b,]——DSL kernel 内切片约束）。"""
    off = _manifest_layout(epn, world_size)
    n_rs = epn * (world_size - 1)
    return (
        gm_rm[off["sj"] : off["sj"] + epn,],
        gm_rm[off["scnt"] : off["scnt"] + 1,],
        gm_rm[off["shome"] : off["shome"] + 1,],
        gm_rm[off["rs"] : off["rs"] + n_rs,],
        gm_rm[off["rj"] : off["rj"] + n_rs,],
        gm_rm[off["rcnt"] : off["rcnt"] + epn,],
    )


def _build_reduce_manifest(context, experts_to_copy, epn):
    """发送/接收清单（设备侧 kernel 生成，按 etc 身份单槽缓存）。

    定长数组 + 运行时计数（编译键与路由解耦）：
    - 发送：sj 为 epn 定长（前 cnt 项有效），scnt 为计数标量，shome 为
      唯一 home（无副本 -1）；
    - 接收：rs/rj 为 epn×(R-1) 定步长（行 i 的 sender 占
      [i*stride, i*stride+rcnt[i])），rcnt 为逐行计数；行号即 home 行。
    各段打包成单块（_manifest_layout），kernel 只收 1 个参数。
    清单由 _MoonepReduceManifestKernel 在设备侧从 etc 生成——host 侧零回读/
    零 Python 扫描/零 H2D（流序写入纪律：清单只经同流 kernel 全量覆写）。
    返回打包后的 NPU 张量。
    """
    cache = context.reduce_manifest
    if cache is not None and cache[0] is experts_to_copy:
        return cache[1]
    gm_rm = torch.empty(
        _manifest_layout(epn, context.rank_size)["total"],
        dtype=torch.int32,
        device="npu",
    )
    _get_reduce_manifest_program(epn, context.rank_size, context.rank)(
        experts_to_copy, gm_rm
    )
    context.reduce_manifest = (experts_to_copy, gm_rm)
    return gm_rm


class _ReduceSpec(NamedTuple):
    """reduce program 编译键：形状/布局/rank 拓扑/分块粒度——**不含路由清单**
    （清单为定长数组 + 运行时计数，结构变化不重编译）。

    stg_off/flags_off/me/rank_size 必须进键：host 函数经模块级变量烘焙，不同 plan
    配置、组数或 rank 不能复用同一 program。chunk 是 UB 分块粒度。
    """

    rows2: int
    w_elems: int
    chunk: int
    stg_off: int
    flags_off: int
    me: int
    rank_size: int
    grad_bf16: bool = False  # grad 视图 dtype（bf16 时 staging/传输减半）
    blocks: int = 28  # 启动块数 = max(R, min(核数, REDUCE_BLOCKS))；grid 栅栏要求常驻


_REDUCE_PROGRAMS = {}
_REDUCE_LOCK = threading.Lock()


def _get_reduce_program(spec):
    """_ReduceSpec 键控的编译产物缓存（模块级烘焙与编译在锁内原子完成）。"""
    with _REDUCE_LOCK:
        prog = _REDUCE_PROGRAMS.get(spec)
        if prog is not None:
            return prog
        global _STG_OFF, _FLAGS_OFF, _ME, _R, _CHUNK, _GRAD_BF16, _BLOCKS
        _STG_OFF = spec.stg_off
        _FLAGS_OFF = spec.flags_off
        _ME, _R = spec.me, spec.rank_size
        _CHUNK = spec.chunk
        _GRAD_BF16 = spec.grad_bf16
        _BLOCKS = spec.blocks
        epn = spec.rows2 // 2
        i32 = dtypes.int32
        prog = cannbotdsl.compile(
            _reduce_grad_host,
            TensorSpec((_manifest_layout(epn, spec.rank_size)["total"],), i32),
            TensorSpec(
                (spec.rows2, spec.w_elems),
                dtypes.bfloat16 if spec.grad_bf16 else dtypes.float32,
            ),
            TensorSpec((1,), i32),
            StructRef[CommContext],
        )
        _REDUCE_PROGRAMS[spec] = prog
        return prog


def _normalize_grad_views(context, grad_view):
    """归一化为 list 并校验组数守卫/行宽一致/dtype/调用序，返回 (grad_view, w_elems, grad_bf16)。"""
    grad_view, w_elems = normalize_group_list(context, grad_view, "reduce_grad")
    grad_dtype = grad_view[0].dtype
    if grad_dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            f"grad_view 须为 fp32 或 bf16（bf16 时 staging/传输减半、UB 内 fp32 "
            f"累加保精度、写回 bf16），got {grad_dtype}"
        )
    if any(g.dtype != grad_dtype for g in grad_view):
        raise TypeError("多梯度组的 dtype 必须一致")
    return grad_view, w_elems, grad_dtype == torch.bfloat16


def _check_stg_capacity(context, epn, w_elems, num_groups, grad_bf16=False):
    """staging 布局容量自检：超 HCCL 对称缓冲区显式报错（防越界静默错算）。

    布局（A1 修订）：窗口池 | flags 区（固定宽，32B 对齐）| staging（按 dtype
    收缩）| 尾。flags 位置与 staging dtype 无关——否则 bf16 的 flags 会踩上
    上次 fp32 staging 的残留数据（浮点位型当 epoch 读出提前放行，见
    `moonep_context.reduce_region_end`）。staging 组间复用：全组共享同一块
    （组间以自旋 barrier 串行），3 组生产配置 staging 占用 45GB→15GB。

    返回 (stg_off, flags_off)。
    """
    world_size = context.rank_size
    # staging 位于多槽窗口池之后：槽位数由首个 create_weight_view 定型
    slots = context.window_slots or 1
    flags_off = reduce_stg_off(epn, w_elems, num_groups=num_groups, slots=slots)
    stg_off = (flags_off + world_size * epn * 4 + 31) & ~31  # 32B 对齐
    elem_bytes = 2 if grad_bf16 else 4
    group_stg_bytes = world_size * epn * w_elems * elem_bytes  # 单组全宽 staging
    # 容量前置校验：相对缓冲区起点的总占用含 HEAP_BASE
    need_bytes = HEAP_BASE + stg_off + group_stg_bytes + LAYOUT_TAIL_BYTES
    check_heap_capacity(context, need_bytes, "reduce_grad staging 布局")
    return stg_off, flags_off


def reduce_grad(context, experts_to_copy, grad_view, epn, *, async_finish=False):
    """副本专家权重梯度归并回 home 并清零预取槽。

    Args:
        context: MoonepContext
        experts_to_copy: [world_size, epn] int32，moonep_plan 输出
            （每行副本槽至多来自一个远程 home rank）
        grad_view: [2*epn, W] fp32 或 bf16 本地梯度视图（bf16 时 staging/传输
            减半、UB 内 fp32 累加保精度、写回 bf16），或多组视图 list/tuple
            （各组行宽 W 一致；与 prefetch_weight 的组数需一致）。
            多组时共享一次清单与屏障，各组独立 staging 区与 epoch flag
        epn: 每 rank 真实本地专家数
        async_finish: 完成语义开关（缺省 False：独立 sym_barrier + host 级
            流等待，返回即全组本轮 staging 读写全部完成）。
            **侧流并发调用传 True**：组间 barrier 改为只发射 kernel 不做
            host 同步（跨 rank staging 保序由 barrier kernel 的自旋协议在
            设备序上保证），末尾屏障跳过，返回发起流上的 NPU 事件——调用方
            `wait_event` 汇合，跨 rank 的 staging/槽位复用安全由调用方在合适
            位置（如下一轮迭代的 barrier）负责；plan 延迟溢出检查顺延到
            下一个我方算子尾部。

    语义：home 的 grad_view[e_local] += Σ_{r 托管 e} 副本梯度；副本槽梯度清零。
    清单按 etc 身份单槽缓存于 context（同 plan 免回读/重建），且为定长数组 +
    运行时计数——路由结构变化不触发重编译。

    staging 为全宽 (world_size, epn, W) 且组间复用同一块（组间自旋 barrier
    串行）：各 chunk 有独立 slice，无块间竞态——单次 program launch 服务全部
    列块（无 host 循环）。布局超出 HCCL 对称缓冲区时显式报错。
    """
    grad_view, w_elems, grad_bf16 = _normalize_grad_views(context, grad_view)
    sync_transport_stream(context)  # 跨流护栏
    if context.rank_size == 1:
        return None  # 单 rank 无副本可归并（etc 恒全 -1；定步长数组退化 0 宽）

    # 启动块数：发送段每个 peer 一个 block（≥R），归并段按 home 行分派。
    # kernel 用 global_sync_all 栅栏 → 块数不得超过本机可常驻 AIV 核数。
    nblk = resident_blocks(REDUCE_BLOCKS)
    if context.rank_size > nblk:
        raise RuntimeError(
            f"EP 组规模 {context.rank_size} 超过本机可常驻 AIV block 数 {nblk}："
            "reduce_grad 的发送段要求每个 peer 一个常驻 block"
        )
    blocks = max(context.rank_size, nblk)

    # UB 分块粒度（kernel 内动态 chunk 循环用；不影响 staging 宽度）
    chunk = pick_weight_chunk(w_elems, _REDUCE_CHUNK_LIMIT)
    stg_off, flags_off = _check_stg_capacity(
        context, epn, w_elems, len(grad_view), grad_bf16=grad_bf16
    )
    gm_rm = _build_reduce_manifest(context, experts_to_copy, epn)

    global _EPOCH
    gm_flag = torch.empty(1, dtype=torch.int32, device=grad_view[0].device)
    for gi, grad_g in enumerate(grad_view):
        if gi > 0:
            # staging 组间复用的配套：前一组的 staging 读写全组完成后才覆盖
            # （flag epoch 跨组单调，残值不会提前放行）。不内联进 kernel：
            # xfer 出现在重 vf 归并段之后会撞编译器复杂度上限（冷编译病态
            # 慢），独立 barrier launch 是已验证形态。async_finish=True 时
            # 只发射不 host 同步：跨 rank staging 覆写安全由 barrier kernel
            # 自旋协议在设备序保证，后继组 kernel 排在其后
            sym_barrier(context, sync=not async_finish)
        # 每组独立数据 epoch flag：跨组/跨次调用都不会被上一轮残留 flag 提前放行
        _EPOCH += 1
        gm_flag.fill_(_EPOCH)
        # 单次 program launch：kernel 内 发送（按 peer 分块）→ 栅栏 → 清槽+归并
        spec = _ReduceSpec(
            rows2=grad_g.shape[0],
            w_elems=w_elems,
            chunk=chunk,
            stg_off=stg_off,
            flags_off=flags_off,
            me=context.rank,
            rank_size=context.rank_size,
            grad_bf16=grad_bf16,
            blocks=blocks,
        )
        _get_reduce_program(spec)(
            gm_rm,
            grad_g,
            gm_flag,
            context.ctx_tensor,
        )
    # 尾段契约见 finish_op（阻塞=独立 barrier+流级等待+溢出检查；async=返回事件）
    return finish_op(context, async_finish)
