# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""reduce_grad：副本专家权重梯度跨 rank 归并回 home rank 并清零预取槽（训练反向）。

机制（两段式归并 + kernel 内动态 chunk 循环 + 清单运行时化）：
  发送侧：对每个托管副本，把其梯度行整行 put 到 home 对称区的 staging[me][j]
    （单次大块传输，无 host 分块循环），同通道 flag 保序通知；
    随后把本地预取槽梯度按 chunk 动态循环清零（槽位复用契约）。
  home 侧：逐行处理本 rank 真实专家，先轮询该行全部 sender 的 flag（数据就位），
    再按 chunk 动态循环在 UB 累加（vadd）后做**单次** mem_copy(atomic_add=True)
    （staging 免清零：发送侧 put 全量覆写后才置 flag）。

清单为定长数组 + 运行时计数（send: epn 槽 + cnt；recv: epn×(R-1) 定步长 +
逐行 cnt），由 _MoonepReduceManifestKernel 在设备侧从 etc 生成（host 侧零回读/
零 Python 扫描/零 H2D）——编译键只含形状/布局，**与路由结构彻底解耦**
（随机路由每步变化也不会触发重编译；早期版本把 recv_groups 结构烘进
编译键，随机路由下约两成步长触发 6.6s 重编译）。

staging 为全宽 (world_size, epn, W) 且组间复用同一块（组间由 kernel 尾段融合
barrier 保序串行，
占用与组数解耦：3 组生产配置 45GB→15GB）：各 chunk 有独立 slice，无块间
复用竞态——单次 program launch 服务全部列块（无 host 循环）。
布局超出 HCCL 对称缓冲区时显式报错（防越界静默错算）。
"""

import threading
from typing import NamedTuple

import torch

from cannbotdsl import TensorSpec, dtypes

import cannbotdsl
from cannbotdsl.types.datastruct import StructRef
from cannbotdsl.buffer import Buffer
from cannbotdsl.tensor import MemLoc, tile_slice
from cannbotdsl.lang.vf import vf
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel

# 归并算子的寄存器原语组（bf16 unpack/pack 与 fp32 累加链）
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
from cannbotdsl.ops.distributed import (
    CommContext,
    create_transfer_context,
    SymBuffer,
    TransferRole,
)

from moonep_context import (
    HEAP_BASE,
    barrier_tail,
    busy_wait_flag,
    check_heap_capacity,
    check_pending_overflow,
    next_barrier_epoch,
    pick_weight_chunk,
    reduce_stg_off,
    scalar_read_fence_post,
    scalar_read_fence_pre,
    sync_transport_stream,
)

_EVENT_ID = 6
_REDUCE_CHUNK_LIMIT = 16384  # 归并分块粒度上限（fp32 元素/块，动态循环用）
_EPOCH = 0  # 进程内单调调用序号（epoch flag）
VL = 64


def _reduce_sym_bufs(ctx, world_size, epn, w_elems, stg_off):
    """staging (world_size, epn, w_elems) + flags (world_size, epn) int32 对称区视图。

    staging dtype 跟随 grad 视图（bf16 时传输/占用减半，UB 内仍 fp32 累加保精度；
    布局偏移恒按 fp32 口径计算，bf16 只占半个包络，全 rank 公式一致）。
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
class _MoonepReduceSendKernel:
    """发送侧：副本梯度整行 put 到 home staging + flag 通知 + 本地预取槽清零。

    清单运行时化：gm_sh/gm_sj 为 epn 定长数组（前 cnt 项有效），cnt 为运行时
    标量（gm_scnt）——编译键不再含清单结构。
    """

    def __init__(self, rank, rank_size, epn, w_elems, stg_off, chunk):
        self._me = int(rank)
        self._r = int(rank_size)
        self._epn = int(epn)
        self._w = int(w_elems)
        self._stg_off = int(stg_off)
        self._chunk = int(chunk)

    def __call__(
        self, gm_sh, gm_sj, gm_scnt, gm_grad, gm_flag, ctx: StructRef[CommContext]
    ):
        world_size, epn = self._r, self._epn
        me = self._me
        w_elems = self._w
        chunk = self._chunk

        stg, flags = _reduce_sym_bufs(ctx, world_size, epn, w_elems, self._stg_off)

        ub_xfer = Buffer(MemLoc.UB, (512,), dtypes.uint8)
        xfer = create_transfer_context(ctx, ub_xfer, role=TransferRole.VEC)

        # 清零缓冲（发送侧清预取槽梯度行，按 chunk 动态循环；dtype 跟随视图）
        grad_dtype = dtypes.bfloat16 if _GRAD_BF16 else dtypes.float32
        ub_zero = self._make_zero_buf(grad_dtype, chunk)

        cnt_send, ub_sh, ub_sj = self._load_send_manifest(gm_scnt, gm_sh, gm_sj)

        if cnt_send > dtypes.int32(0):
            n_chunks = w_elems // chunk
            # 先全部 issue 整行数据 put（单次大块传输，无 host 分块循环），
            # 再补 flag，最后统一 flush：不同 peer 的传输在各自通道并行
            for i in cannbotdsl.range(cnt_send):
                h = dtypes.int64(ub_sh[i])
                j = ub_sj[i]
                xfer.put(
                    h,
                    tile_slice(stg, (1, 1, w_elems), (me, j, 0)).peer(h),
                    tile_slice(gm_grad, (1, w_elems), (epn + j, 0)),
                )
            for i in cannbotdsl.range(cnt_send):
                h = dtypes.int64(ub_sh[i])
                j = ub_sj[i]
                xfer.put(h, tile_slice(flags, (1, 1), (me, j)).peer(h), gm_flag)
            for i in cannbotdsl.range(cnt_send):
                xfer.flush(dtypes.int64(ub_sh[i]))
            # 清零本地预取槽梯度行（动态 chunk 循环，避免 constexpr 展开编译爆炸）
            self._zero_prefetch_slots(gm_grad, ub_sj, ub_zero, cnt_send, n_chunks)

    @jit
    def _make_zero_buf(self, grad_dtype, chunk):
        """清零缓冲制作（bf16 掩码按 16 位粒度——full_mask 的 32 位粒度在
        16 位数据上只写一半）。"""
        ub_zero = Buffer(MemLoc.UB, (1, chunk), grad_dtype)
        with vf(mode="simd"):
            if _GRAD_BF16:
                seg16 = update_mask(VL, elem_bits=16)[0]
                vz = vdups(0.0, grad_dtype, mask=seg16)
                for off in range_constexpr(0, chunk, VL):
                    vstore(ub_zero, off, vz, seg16)
            else:
                full = full_mask()
                vz = vdups(0.0, grad_dtype, mask=full)
                for off in range_constexpr(0, chunk, VL):
                    vstore(ub_zero, off, vz, full)
        return ub_zero

    @jit
    def _load_send_manifest(self, gm_scnt, gm_sh, gm_sj):
        """清单 GM→UB 拷贝 + 完整握手（与 busy_wait_flag 同款护栏）。"""
        epn = self._epn
        scalar_read_fence_pre(_EVENT_ID)
        ub_scnt = Buffer(MemLoc.UB, (1,), dtypes.int32)
        mem_copy(ub_scnt, gm_scnt)
        ub_sh = Buffer(MemLoc.UB, (epn,), dtypes.int32)
        ub_sj = Buffer(MemLoc.UB, (epn,), dtypes.int32)
        mem_copy(ub_sh, gm_sh)
        mem_copy(ub_sj, gm_sj)
        scalar_read_fence_post(_EVENT_ID)
        return ub_scnt[dtypes.int64(0)], ub_sh, ub_sj

    @jit
    def _zero_prefetch_slots(self, gm_grad, ub_sj, ub_zero, cnt_send, n_chunks):
        """清零本地预取槽梯度行（动态 chunk 循环）。"""
        epn = self._epn
        chunk = self._chunk
        for i in cannbotdsl.range(cnt_send):
            j = ub_sj[i]
            for c in cannbotdsl.range(n_chunks):
                mem_copy(
                    tile_slice(gm_grad, (1, chunk), (epn + j, c)),
                    tile_slice(ub_zero, (1, chunk), (0, 0)),
                )


@kernel
class _MoonepReduceRecvKernel:
    """home 侧：逐行轮询 flag → 行内 staging UB 累加 → 每 chunk 单次 atomic。

    全宽 staging (world_size, epn, W)：各 chunk 有独立 slice，接收侧无需等发送侧
    "上一块消费完"——一次轮询全行 flag 后，chunk 循环内独立搬运/累加。
    全部循环为运行时界动态循环（cannbotdsl.range），清单经定长数组 + 逐行
    计数传入——编译键与路由结构解耦（每 shape 只编一次）。

    轮询的 flag/epoch UB 缓冲在 __call__ 入口分配一次、全行共享（VEC buf_id 预算）。
    """

    def __init__(self, rank, rank_size, epn, w_elems, stg_off, chunk):
        self._me = int(rank)
        self._r = int(rank_size)
        self._epn = int(epn)
        self._w = int(w_elems)
        self._stg_off = int(stg_off)
        self._chunk = int(chunk)
        self._rcnt = None  # __call__ 内分配（DSL trace 边界）
        self._rs = None  # __call__ 内分配（DSL trace 边界）
        self._rj = None  # __call__ 内分配（DSL trace 边界）
        self._acc = None  # __call__ 内分配（DSL trace 边界）
        self._tmp = None  # __call__ 内分配（DSL trace 边界）
        self._cur = None  # __call__ 内分配（DSL trace 边界）
        self._flag = None  # __call__ 内分配（DSL trace 边界）
        self._epoch = None  # __call__ 内分配（DSL trace 边界）

    def __call__(
        self,
        gm_rs,
        gm_rj,
        gm_rcnt,
        gm_grad,
        gm_flag,
        gm_bar_epoch,
        ctx: StructRef[CommContext],
    ):
        world_size, epn = self._r, self._epn
        w_elems = self._w
        chunk = self._chunk
        n_chunks = w_elems // chunk

        stg, flags = _reduce_sym_bufs(ctx, world_size, epn, w_elems, self._stg_off)

        # staging 免清零：发送侧 put 全量覆写该行后才置 flag
        self._load_recv_manifest(gm_rcnt, gm_rs, gm_rj)
        grad_dtype = dtypes.bfloat16 if _GRAD_BF16 else dtypes.float32
        self._acc = Buffer(
            MemLoc.UB, (1, chunk), dtypes.float32
        )  # 恒 fp32（累加精度契约）
        self._tmp = Buffer(MemLoc.UB, (1, chunk), grad_dtype)  # 跟随 staging
        self._cur = Buffer(MemLoc.UB, (1, chunk), grad_dtype)  # bf16 写回读-改-写用
        self._flag = Buffer(MemLoc.UB, (1,), dtypes.int32)
        self._epoch = Buffer(MemLoc.UB, (1,), dtypes.int32)
        self._process_rows(epn, flags, gm_flag, stg, gm_grad, n_chunks)
        self._barrier_epilogue(ctx, gm_bar_epoch, world_size)

    @jit
    def _process_rows(self, epn, flags, gm_flag, stg, gm_grad, n_chunks):
        """逐行处理：轮询行内 sender flag 就位后累加写回。"""
        for i in cannbotdsl.range(epn):
            cnt_i = self._rcnt[i]
            if cnt_i > dtypes.int32(0):
                self._poll_row(i, cnt_i, flags, gm_flag)
                self._accumulate_row(i, cnt_i, stg, gm_grad, n_chunks)

    @jit
    def _barrier_epilogue(self, ctx, gm_bar_epoch, world_size):
        """融合 barrier 尾段：本 kernel 完成即全组本组 staging 读写全部完成
        （组间复用同一块 staging 的串行保序 + 返回保证；与独立 sym_barrier
        同 flag 协议，省一次发射器 launch ~550µs）。"""
        ub_xfer = Buffer(MemLoc.UB, (512,), dtypes.uint8)
        xfer = create_transfer_context(ctx, ub_xfer, role=TransferRole.VEC)
        barrier_tail(
            xfer, ctx, gm_bar_epoch, self._me, world_size, self._flag, self._epoch
        )

    @jit
    def _load_recv_manifest(self, gm_rcnt, gm_rs, gm_rj):
        """接收清单 GM→UB 拷贝 + 完整握手（rcnt/rs/rj 读取前等拷贝落地）。"""
        epn = self._epn
        stride = self._r - 1
        scalar_read_fence_pre(_EVENT_ID)
        self._rcnt = Buffer(MemLoc.UB, (epn,), dtypes.int32)
        mem_copy(self._rcnt, gm_rcnt)
        self._rs = Buffer(MemLoc.UB, (epn * stride,), dtypes.int32)
        mem_copy(self._rs, gm_rs)
        self._rj = Buffer(MemLoc.UB, (epn * stride,), dtypes.int32)
        mem_copy(self._rj, gm_rj)
        scalar_read_fence_post(_EVENT_ID)

    @jit
    def _poll_row(self, i, cnt_i, flags, gm_flag):
        """先轮询行内全部 sender 的 flag（数据全部就位后再开始搬运）。"""
        stride = self._r - 1
        for s in cannbotdsl.range(cnt_i):
            view = tile_slice(
                flags,
                (1, 1),
                (self._rs[i * stride + s], self._rj[i * stride + s]),
            )
            busy_wait_flag(view, gm_flag, _EVENT_ID, self._flag, self._epoch)

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
                self._acc_merge(chunk)
            self._writeback_row(i, c, chunk, gm_grad)

    @jit
    def _acc_merge(self, chunk):
        """acc ← acc + tmp（bf16 载体按 32 位粒度，unpack/cvt 到 fp32 累加保精度）。"""
        with vf(mode="simd"):
            full = full_mask()
            seg = update_mask(VL, elem_bits=32)[0]
            for off in cannbotdsl.range(0, chunk, VL):
                va = vload(self._acc, off)
                if _GRAD_BF16:
                    vb = vload_unpack(self._tmp, off, unpack_mode=UnpackMode.B16_TO_B32)
                    vb = vcast(vb, dtypes.float32, mask=seg)
                    vstore(self._acc, off, vadd(va, vb, mask=seg), mask=seg)
                else:
                    vb = vload(self._tmp, off)
                    vstore(self._acc, off, vadd(va, vb, mask=full), mask=full)

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
_FLAGS_OFF = 0  # flags 区基址：所有组 staging 之后（多组时不能与下一组 staging 重叠）
_ME = 0  # 组内 rank（编译锁内写入）
_R = 1  # 组内 rank_size（均须进 program 键）
_CHUNK = 1  # UB 分块粒度（编译锁内写入，进 program 键）
_GRAD_BF16 = False  # grad 视图 dtype（编译锁内写入，进 program 键）


@host
def _reduce_grad_host(
    gm_send_home,
    gm_send_slot,
    gm_send_cnt,
    gm_recv_sender,
    gm_recv_slot,
    gm_recv_rcnt,
    gm_grad,
    gm_flag,
    gm_bar_epoch,
    ctx: StructRef[CommContext],
):
    epn2, w_elems = gm_grad.shape
    # 通信 kernel（put+flag+清槽）与计算 kernel（轮询+两段式归并）拆分：
    # xfer 通信原语与重 vf 区混在同一 kernel 会撞 DSL 编译器 SIGSEGV
    # （与 plan 的多 kernel 拆分同因）。流上顺序执行，无死锁：recv 只等 send。
    _MoonepReduceSendKernel(
        _ME,
        _R,
        epn2 // 2,
        w_elems,
        _STG_OFF,
        _CHUNK,
    )[1](gm_send_home, gm_send_slot, gm_send_cnt, gm_grad, gm_flag, ctx)
    _MoonepReduceRecvKernel(
        _ME,
        _R,
        epn2 // 2,
        w_elems,
        _STG_OFF,
        _CHUNK,
    )[1](
        gm_recv_sender, gm_recv_slot, gm_recv_rcnt, gm_grad, gm_flag, gm_bar_epoch, ctx
    )


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


_REDUCE_PROGRAMS = {}
_REDUCE_LOCK = threading.Lock()


def _get_reduce_program(spec):
    """_ReduceSpec 键控的编译产物缓存（模块级烘焙与编译在锁内原子完成）。"""
    with _REDUCE_LOCK:
        prog = _REDUCE_PROGRAMS.get(spec)
        if prog is not None:
            return prog
        global _STG_OFF, _FLAGS_OFF, _ME, _R, _CHUNK, _GRAD_BF16
        _STG_OFF = spec.stg_off
        _FLAGS_OFF = spec.flags_off
        _ME, _R = spec.me, spec.rank_size
        _CHUNK = spec.chunk
        _GRAD_BF16 = spec.grad_bf16
        epn = spec.rows2 // 2
        stride = spec.rank_size - 1
        i32 = dtypes.int32
        prog = cannbotdsl.compile(
            _reduce_grad_host,
            TensorSpec((epn,), i32),
            TensorSpec((epn,), i32),
            TensorSpec((1,), i32),
            TensorSpec((epn * stride,), i32),
            TensorSpec((epn * stride,), i32),
            TensorSpec((epn,), i32),
            TensorSpec(
                (spec.rows2, spec.w_elems),
                dtypes.bfloat16 if spec.grad_bf16 else dtypes.float32,
            ),
            TensorSpec((1,), i32),
            TensorSpec((1,), i32),
            StructRef[CommContext],
        )
        _REDUCE_PROGRAMS[spec] = prog
        return prog


@kernel
class _MoonepReduceManifestKernel:
    """设备侧生成发送/接收清单（替代 host 侧 etc 扫描）。

    单 block 扫描 etc [R, epn]：
    - 发送：本 rank 承载的副本槽 j → (home=e//epn, j)，前 scnt 项有效；
    - 接收：本 rank 真实专家 e=me*epn+i 的远端副本来源 → 行 i 占
      [i*(R-1), i*(R-1)+rcnt[i])，行号即 home 行。
    输出定长数组 + 运行时计数（编译键与路由结构解耦）；host 侧零回读。
    """

    def __init__(self, rank, rank_size, epn):
        self._me = int(rank)
        self._r = int(rank_size)
        self._epn = int(epn)

    def __call__(self, gm_etc, gm_sh, gm_sj, gm_scnt, gm_rs, gm_rj, gm_rcnt):
        world_size, epn, me = self._r, self._epn, self._me
        stride = world_size - 1
        ub_etc = Buffer(MemLoc.UB, (world_size, epn), dtypes.int32)
        mem_copy(ub_etc, gm_etc)
        ub_sh = Buffer(MemLoc.UB, (epn,), dtypes.int32)
        ub_sj = Buffer(MemLoc.UB, (epn,), dtypes.int32)
        ub_scnt = Buffer(MemLoc.UB, (1,), dtypes.int32)
        ub_rs = Buffer(MemLoc.UB, (epn * stride,), dtypes.int32)
        ub_rj = Buffer(MemLoc.UB, (epn * stride,), dtypes.int32)
        ub_rcnt = Buffer(MemLoc.UB, (epn,), dtypes.int32)

        with vf(mode="simt", thread=1):
            # 发送清单：etc 行即本 rank 的副本槽序（计数器驻留 UB）
            ub_scnt[dtypes.int64(0)] = dtypes.int32(0)
            for j in cannbotdsl.range(epn):
                e = ub_etc[dtypes.int64(me), dtypes.int64(j)]
                if e >= dtypes.int32(0):
                    c = ub_scnt[dtypes.int64(0)]
                    ub_sh[c] = e // dtypes.int32(epn)
                    ub_sj[c] = dtypes.int32(j)
                    ub_scnt[dtypes.int64(0)] = c + dtypes.int32(1)
            # 接收清单：无效区置 -1（仅前 rcnt 项被读，便于异常时定位）
            for k in cannbotdsl.range(epn * stride):
                ub_rs[dtypes.int64(k)] = dtypes.int32(-1)
                ub_rj[dtypes.int64(k)] = dtypes.int32(-1)
            for i in cannbotdsl.range(epn):
                ub_rcnt[dtypes.int64(i)] = dtypes.int32(0)
            for r in range_constexpr(world_size):
                if const_expr(r != me):
                    self._scan_peer_row(r, ub_etc, ub_rs, ub_rj, ub_rcnt)
        mem_copy(gm_sh, ub_sh)
        mem_copy(gm_sj, ub_sj)
        mem_copy(gm_scnt, ub_scnt)
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
def _reduce_manifest_host(gm_etc, gm_sh, gm_sj, gm_scnt, gm_rs, gm_rj, gm_rcnt):
    world_size, epn = gm_etc.shape
    _MoonepReduceManifestKernel(_ME, world_size, epn)[1](
        gm_etc, gm_sh, gm_sj, gm_scnt, gm_rs, gm_rj, gm_rcnt
    )


_REDUCE_MANIFEST_PROGRAMS = {}


def _get_reduce_manifest_program(epn, world_size, me):
    """清单 program：(epn, rank_size, me) 键控（_ME 烘焙与编译在 _REDUCE_LOCK
    内原子完成——与 reduce program 共用锁，防烘焙窗口互污）。
    """
    key = (epn, world_size, me)
    with _REDUCE_LOCK:
        prog = _REDUCE_MANIFEST_PROGRAMS.get(key)
        if prog is None:
            global _ME
            _ME = me
            i32 = dtypes.int32
            stride = world_size - 1
            prog = cannbotdsl.compile(
                _reduce_manifest_host,
                TensorSpec((world_size, epn), i32),
                TensorSpec((epn,), i32),
                TensorSpec((epn,), i32),
                TensorSpec((1,), i32),
                TensorSpec((epn * stride,), i32),
                TensorSpec((epn * stride,), i32),
                TensorSpec((epn,), i32),
            )
            _REDUCE_MANIFEST_PROGRAMS[key] = prog
    return prog


def _build_reduce_manifest(context, experts_to_copy, epn):
    """发送/接收清单（设备侧 kernel 生成，按 etc 身份单槽缓存）。

    定长数组 + 运行时计数（编译键与路由解耦）：
    - 发送：gm_sh/gm_sj 为 epn 定长（前 cnt 项有效），gm_scnt 为计数标量；
    - 接收：gm_rs/gm_rj 为 epn×(R-1) 定步长（行 i 的 sender 占
      [i*stride, i*stride+rcnt[i])），gm_rcnt 为逐行计数；行号即 home 行。
    清单由 _MoonepReduceManifestKernel 在设备侧从 etc 生成——host 侧零回读/
    零 Python 扫描/零 H2D（流序写入纪律：清单只经同流 kernel 全量覆写）。
    返回 (gm_sh, gm_sj, gm_scnt, gm_rs, gm_rj, gm_rcnt)，均为 NPU 张量。
    """
    cache = context.reduce_manifest
    if cache is not None and cache[0] is experts_to_copy:
        return cache[1]
    me = context.rank
    world_size = context.rank_size
    stride = world_size - 1
    i32 = torch.int32
    manifest = (
        torch.zeros(epn, dtype=i32, device="npu"),
        torch.zeros(epn, dtype=i32, device="npu"),
        torch.zeros(1, dtype=i32, device="npu"),
        torch.full((epn * stride,), -1, dtype=i32, device="npu"),
        torch.full((epn * stride,), -1, dtype=i32, device="npu"),
        torch.zeros(epn, dtype=i32, device="npu"),
    )
    _get_reduce_manifest_program(epn, world_size, me)(experts_to_copy, *manifest)
    context.reduce_manifest = (experts_to_copy, manifest)
    return manifest


def _normalize_grad_views(context, grad_view):
    """归一化为 list 并校验组数守卫/行宽一致/dtype/调用序，返回 (grad_view, w_elems, grad_bf16)。"""
    if isinstance(grad_view, torch.Tensor):
        grad_view = [grad_view]
    if not grad_view:
        raise ValueError("grad_view 至少一组（可为单 Tensor 或多组 list/tuple）")
    if context.layout_groups is None:
        context.layout_groups = len(grad_view)
    elif context.layout_groups != len(grad_view):
        raise RuntimeError(
            f"prefetch/reduce 组数不一致（已有 {context.layout_groups}，本次 {len(grad_view)}），"
            "会导致发布/staging 区重叠"
        )
    w_elems = grad_view[0].shape[1]
    if any(g.shape[1] != w_elems for g in grad_view):
        raise ValueError("多梯度组的行宽 W 必须一致")
    grad_dtype = grad_view[0].dtype
    if grad_dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            f"grad_view 须为 fp32 或 bf16（bf16 时 staging/传输减半、UB 内 fp32 "
            f"累加保精度、写回 bf16），got {grad_dtype}"
        )
    if any(g.dtype != grad_dtype for g in grad_view):
        raise TypeError("多梯度组的 dtype 必须一致")
    if context.last_plan_shape is None:
        raise RuntimeError("reduce_grad 需先调用 moonep_plan（依赖其形状与窗口布局）")
    return grad_view, w_elems, grad_dtype == torch.bfloat16


def _check_stg_capacity(context, epn, w_elems, num_groups):
    """staging 布局容量自检：超 HCCL 对称缓冲区显式报错（防越界静默错算）。

    返回 (base_stg_off, group_stg_bytes, flags_off)。
    """
    world_size = context.rank_size
    # staging 位于多槽窗口池之后：槽位数由首个 create_weight_view 定型
    slots = context.window_slots or 1
    base_stg_off = reduce_stg_off(epn, w_elems, num_groups=num_groups, slots=slots)
    group_stg_bytes = world_size * epn * w_elems * 4  # 单组全宽 staging 字节数（fp32）
    # staging 组间复用：全组共享同一块（组间由 kernel 尾段融合 barrier 串行），flags 紧随
    # 单组 staging 之后——3 组生产配置 staging 占用 45GB→15GB
    flags_off = base_stg_off + group_stg_bytes
    # 容量前置校验：相对缓冲区起点的总占用含 HEAP_BASE
    need_bytes = HEAP_BASE + flags_off + world_size * epn * 4
    check_heap_capacity(context, need_bytes, "reduce_grad staging 布局")
    return base_stg_off, group_stg_bytes, flags_off


def reduce_grad(context, experts_to_copy, grad_view, epn):
    """副本专家权重梯度归并回 home 并清零预取槽。

    Args:
        context: MoonepContext
        experts_to_copy: [world_size, epn] int32，moonep_plan 输出
        grad_view: [2*epn, W] fp32 或 bf16 本地梯度视图（bf16 时 staging/传输
            减半、UB 内 fp32 累加保精度、写回 bf16），或多组视图 list/tuple
            （各组行宽 W 一致；与 prefetch_weight 的组数需一致）。
            多组时共享一次清单与屏障，各组独立 staging 区与 epoch flag
        epn: 每 rank 真实本地专家数

    语义：home 的 grad_view[e_local] += Σ_{r 托管 e} 副本梯度；副本槽梯度清零。
    清单按 etc 身份单槽缓存于 context（设备侧 kernel 生成），且为定长数组 +
    运行时计数——路由结构变化不触发重编译。

    staging 为全宽 (world_size, epn, W) 且组间复用同一块（组间自旋 barrier
    串行）：各 chunk 有独立 slice，无块间竞态——单次 program launch 服务全部
    列块（无 host 循环）。布局超出 HCCL 对称缓冲区时显式报错。
    """
    grad_view, w_elems, grad_bf16 = _normalize_grad_views(context, grad_view)
    sync_transport_stream(context)  # 跨流护栏
    if context.rank_size == 1:
        return  # 单 rank 无副本可归并（etc 恒全 -1；定步长数组退化 0 宽）

    # UB 分块粒度（kernel 内动态 chunk 循环用；不影响 staging 宽度）
    chunk = pick_weight_chunk(w_elems, _REDUCE_CHUNK_LIMIT)
    base_stg_off, group_stg_bytes, flags_off = _check_stg_capacity(
        context, epn, w_elems, len(grad_view)
    )
    gm_sh, gm_sj, gm_scnt, gm_rs, gm_rj, gm_rcnt = _build_reduce_manifest(
        context, experts_to_copy, epn
    )

    global _EPOCH
    gm_flag = torch.empty(1, dtype=torch.int32, device=grad_view[0].device)
    for grad_g in grad_view:  # 组索引不再需要（barrier 融合后无逐组 host 逻辑）
        # 每组独立数据 epoch flag：跨组/跨次调用都不会被上一轮残留 flag 提前放行
        _EPOCH += 1
        gm_flag.fill_(_EPOCH)
        # 融合 barrier 尾段（recv kernel 尾）：本组 program 完成即全组本组
        # staging 读写完成——组间复用同一块 staging 的串行保序由同流 kernel
        # 顺序 + barrier 尾段天然保证，无需独立 barrier launch 与 host 同步
        gm_bar_epoch = torch.full(
            (1,), next_barrier_epoch(), dtype=torch.int32, device="npu"
        )
        # 单次 program launch：kernel 内动态 chunk 循环处理全部列块，
        # 全宽 staging（组间复用同一块）各块独立 slice，无 host 循环
        spec = _ReduceSpec(
            rows2=grad_g.shape[0],
            w_elems=w_elems,
            chunk=chunk,
            stg_off=base_stg_off,
            flags_off=flags_off,
            me=context.rank,
            rank_size=context.rank_size,
            grad_bf16=grad_bf16,
        )
        _get_reduce_program(spec)(
            gm_sh,
            gm_sj,
            gm_scnt,
            gm_rs,
            gm_rj,
            gm_rcnt,
            grad_g,
            gm_flag,
            gm_bar_epoch,
            context.ctx_tensor,
        )
    # 返回保证：发起流级等待（融合 barrier 在各组 kernel 尾段，流排空即全组完成）
    torch.npu.current_stream().synchronize()
    # plan 延迟溢出检查（此处流已排空，.item() 微秒级）
    check_pending_overflow(context)
