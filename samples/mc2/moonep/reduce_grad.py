# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""reduce_grad：副本专家权重梯度跨 rank 归并回 home rank（训练反向）。

**owner 直累协议（0.8.3 新通信 API，仅 UbMem 映射读——URMA 两段式
已随传输退役删除，2026-11-16 用户指令）**，数据流：

  FFN 反向产出 full-dW ``t[2*epn, Wg]``（前 epn 行 home 种子、后 epn 行
  副本 dW）→ 发送侧 host 把副本行拷入**本 rank** workspace 副本区
  ``[epn, Wg]``（一次 GM 拷贝，恒定不随 R 增长）→ 单 kernel
  （2/3 组一次 launch，block-per-peer）：
  ① 头部屏障（per-peer flag 握手：对端 kernel 启动即其流序在前的
     host 副本区写入已完成——保护对端上轮远读与本轮覆写不重叠）
  ② owner 逐 peer **映射远读**副本区（etc_inv 逆映射行命中）→ UB
     流水环（issue #16 depth=2：读 k+1 ∥ 累加 k）→ ``mem_copy(
     atomic_add=True)`` 拷贝引擎原子直累 home 行
  ③ 尾部 sym_barrier（下一轮 host 覆写安全）

  精度注记：bf16 逐加舍入（测试整值精确；生产梯度累积可接受——
  fp32 梯度路径为精确档）。

本模块要求：E%R==0；grad_view 为 [2*epn, W]（home 段 + 副本段，行序与
prefetch 视图一致）。
"""

import os
import threading
from typing import NamedTuple

import cannbotdsl
import cannbotdsl as cbd
import torch
from cannbotdsl import copy_options
from cannbotdsl import TensorSpec, dtypes
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.tensor import MemLoc, make_layout, make_pointer, make_tensor, tile_slice
from cannbotdsl.types.datastruct import StructRef

from moonep_context import (
    BAR_FLAGS_OFF,
    HEAP_BASE,
    LAYOUT_TAIL_BYTES,
    check_heap_capacity,
    finish_op,
    grad_slot_off,
    next_barrier_epoch,
    normalize_group_list,
    pick_weight_chunk,
    resident_blocks,
    sync_transport_stream,
)
from moonep_context import sym_barrier
from moonep_urma import (
    MoonepUrmaCtx,
    connection_from_address,
    spin_u32_ge,
    tensor_address,
    ub_flag_write,
    xfer_open,
)

# VL：fp32 归并步进（vf 寄存器 256B//4 = 64 @950PR——A5 实证）。
# 检视意见建议 get_vf_len() 动态获取——实测不可行：该 API 需 IR
# location（登录节点 ValueError "No current Location"——模块导入期/
# host 侧均不可调用；range_constexpr 亦要求编译期常量）。保留常量 +
# 本注释作为 won't-fix 依据。
VL = 64
_MERGE_FENCE_ID = 6  # etc 扫描 MTE2→S 栅栏事件（kernel 内独立无冲突）

_MAX_GRAD_GROUPS = 3  # 梯度组槽上限（kernel 入口固定 3 张量槽，ngroups 裁剪）

_R = 1  # 组内 rank_size（编译锁内写入，进 program 键）
_BLOCKS = 28  # 实际启动块数（编译锁内写入，进 program 键）
_REDUCE_LOCK = threading.Lock()

# merge UB 预算：ub_row（grad dtype chunk）+ ub_acc（fp32 chunk）≤ ~96KB
_CHUNK_UB_LIMIT_BF16 = 16384
_CHUNK_UB_LIMIT_FP32 = 8192
_HEADER_BARRIER_CAP = 64  # 头部屏障/读块上限（950PR vector_core_num 满额）


def peer_block_split(rank_size, blocks, reserve_merge=False):
    """头部屏障写者块与读块的划分（R 与 AIV 核数解耦，issue #14 B1/B2）。

    ``blocks`` 由 host clamp 到常驻核数（resident_blocks）——所有块都在首波，
    屏障写者不会再被分波调度推到第二波。写者块按 ``rank_size-1`` 轮转分摊
    peer（``reserve_merge=True`` 供归并同拍形态让出读块——当前直累路径
    不用，保留给组合形态）。
    返回 ``(n_writer, per_writer, n_read, per_read)``。
    """
    n_rem = max(int(rank_size) - 1, 0)
    n_writer = min(n_rem, int(blocks))
    per_writer = (n_rem + n_writer - 1) // n_writer if n_writer else 0
    n_read = n_writer
    if reserve_merge and n_writer >= int(blocks):
        n_read = max(1, int(blocks) // 2)
    per_read = (n_rem + n_read - 1) // n_read if n_read else 0
    return n_writer, per_writer, n_read, per_read


@jit
def _barrier_flags_put(ctx, blk, me64, n_peer, n_writer, per_writer, gm_epoch, epoch):
    """头部屏障写者块：把 my flag 写到本块分管的全部 peer（轮转分配）。

    先写完分管 peer 再进自旋（调用方在循环外等全部 peer）——写者块全在
    首波，写先于等，避免分波后互相等待。ub_flag_write 首尾 fence（MTE3
    同通道保序），无需额外 drain。
    """
    for slot in range_constexpr(int(per_writer)):
        p_slot = blk + dtypes.int64(slot * int(n_writer))
        if p_slot < dtypes.int64(n_peer):
            peer_idx = dtypes.uint32(dtypes.uint64(p_slot))
            peerh = xfer_open(ctx, peer_idx)
            ub_flag_write(
                peerh,
                dtypes.uint64(HEAP_BASE + BAR_FLAGS_OFF) + me64 * dtypes.uint64(4),
                epoch,
                tensor_address(gm_epoch),
            )


def _normalize_grad_views(context, local_grads, epn):
    """归一化 full-dW 组列表并推导 (groups, widths, dtype)。

    契约：每组 ``[2*epn, Wg]`` 或任意尾维 ``[2*epn, ...]``（连续——内部
    展平为 [2*epn, Wg]）；行序与 prefetch 视图一致（前 epn 行 home 种子、
    后 epn 行副本 dW）。组数 2 或 3、各组行宽可不同；返回
    (flat_views, widths, grad_bf16, grad_fp16)。
    """
    local_grads, widths = normalize_group_list(context, local_grads, "reduce_grad")
    grad_dtype = local_grads[0].dtype
    if grad_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError(
            f"local_grads 须为 fp32/bf16/fp16（16 位时传输减半、归并期 "
            f"fp32 累加保精度、写回原 dtype），got {grad_dtype}"
        )
    frozen = getattr(context, "grad_dtype", None)
    if frozen is not None and frozen != grad_dtype:
        raise TypeError(
            f"local_grads dtype 与构造定型值不符（定型 {frozen}，本次 {grad_dtype}）"
            "——梯度接收区按定型 dtype 预留（fp32 为 bf16 的 2×），"
            "不可运行期漂移；请以 grad_dtype=... 重建或转换 local_grads"
        )
    flats = []
    for g, t in enumerate(local_grads):
        if t.dtype != grad_dtype:
            raise TypeError("多梯度组的 dtype 必须一致")
        if int(t.shape[0]) != 2 * int(epn):
            raise ValueError(
                f"第 {g} 组行数须为 2*epn={2 * int(epn)}（full-dW 形态：前 epn 行 "
                f"home 种子、后 epn 行副本 dW——与 prefetch 权重视图同序），"
                f"实得 {tuple(t.shape)}"
            )
        if not t.is_contiguous():
            raise ValueError(f"第 {g} 组须连续（展平前置）")
        flats.append(t.reshape(2 * int(epn), widths[g]))
    return flats, widths, grad_dtype == torch.bfloat16, grad_dtype == torch.float16


def _grad_region_params(context):
    """(scale_widths, weight_bytes)——梯度接收区基址的两个布局参数。"""
    scale_widths = getattr(context, "layout_scale_widths", None) or None
    weight_dtype = getattr(context, "weight_dtype", None)
    if weight_dtype is None:
        weight_bytes = 2
    else:
        weight_bytes = int(weight_dtype.itemsize)  # bf16/fp16=2，fp8=1
    return scale_widths, weight_bytes


class _ReduceSpec(NamedTuple):
    """reduce 直累 kernel 编译键（副本区偏移 copy_off）。"""

    epn: int
    sizes: tuple  # 3 槽行宽
    elem_bytes: tuple  # 3 槽要素字节宽
    ngroups: int
    copy_off: int  # 副本区基址（[epn, Wg]——无 R 维度）
    chunks: tuple  # 逐组归并 chunk
    rank_size: int
    grad_bf16: bool
    grad_fp16: bool  # fp16 梯度（16 位路径与 bf16 同构，dtype 三选一）
    blocks: int
    split: int = 8  # 行分片读块数/链（=1 退化 block-per-peer；进键）
    depth: int = 3  # 流水环槽深（≥2；进键——UB 占用 = depth×chunk）


# ---------------------------------------------------------------------------
# reduce 直累 kernel（仅 UbMem——URMA 两段式已随传输退役删除）
# ---------------------------------------------------------------------------
@kernel
class _UbmemReduceAccKernel:
    """直累 kernel（master 0.8 设计回归，0.8.3 API 适配——用户指令
    2026-11-15：UbMem 映射直读底层即 URMA 同链，master 实证 359.4）。

    读块按 peer 轮转：块 b 分管 peer b, b+n_writer, ...（``peer_block_split``
    ——R>64 时块数封顶常驻核数，与链路结构解耦）；逐行 etc_inv 直读标量
    （j≥0 = 该 peer 承载行 i 副本）；逐 chunk：映射远读 tile → UB 暂存 →
    ``mem_copy(atomic_add=True)`` 直累 home 行（**归并零 vf——拷贝引擎
    原子加，硬件串行化任意 sender 数正确**）。home 种子由调用侧 local_grads
    前段即位（原子加叠加上去——无种子搬运）。精度注记：bf16 逐加舍入
    （测试整值精确；生产梯度累积可接受——fp32 梯度为精确档）。
    """

    def __init__(
        self,
        rank_size,
        epn,
        sizes,
        elem_bytes,
        ngroups,
        copy_off,
        chunks,
        blocks,
        grad_bf16,
        grad_fp16=False,
        row_split=1,
        depth=2,
    ):
        self._r = int(rank_size)
        self._epn = int(epn)
        self._sizes = tuple(int(s) for s in sizes)
        self._elem_bytes = tuple(int(b) for b in elem_bytes)
        self._ngroups = int(ngroups)
        self._copy_off = int(copy_off)
        self._row_bytes = tuple(s * b for s, b in zip(self._sizes, self._elem_bytes))
        self._chunks = tuple(int(c) for c in chunks)
        self._n_tiles = tuple(
            self._epn * (self._sizes[g] // self._chunks[g]) for g in range(3)
        )
        self._blocks = int(blocks)
        self._row_split = int(row_split)
        self._depth = int(depth)
        # 读块布局（R 与块数解耦 + 行分片提在飞读数/链）：
        #   块 b ∈ [0, n_read)：peer 槽 = b % n_rem（轮转覆盖全 peer），
        #   行分片 = b // n_rem（行 i 以 n_slices 步进归属本块——同行
        #   仅一个 (peer, 行) 写者，跨块同行并发由 atomic_add 硬件串行化）。
        #   split=1 或块数不足时 n_slices=1/per_read>1——退化为原
        #   block-per-peer 全行形态（含 R>64 封顶轮转）。
        #   flag 写者块独立取前 n_writer 个（peer 轮转分管）。
        n_rem = max(self._r - 1, 0)
        self._n_writer = min(n_rem, self._blocks)
        self._per_writer = (
            (n_rem + self._n_writer - 1) // self._n_writer if self._n_writer else 0
        )
        n_read = min(n_rem * self._row_split, self._blocks)
        if n_read > n_rem:
            # 纯分片域：裁齐到 n_rem 整数倍——保证每 peer 恰 k 个分片块
            # （非整倍时尾 cohort 部分覆盖 → 该 peer 缺行——R=16/b=64 仿真实证）
            n_read = (n_read // n_rem) * n_rem
        self._n_read = n_read
        self._per_read = (
            (n_rem + self._n_read - 1) // self._n_read if self._n_read else 0
        )
        self._n_slices = (self._n_read + n_rem - 1) // n_rem if n_rem else 1
        self._grad_bf16 = bool(grad_bf16)
        self._grad_fp16 = bool(grad_fp16)
        group_off, gbase = [], int(copy_off)
        for g in range(3):
            group_off.append(gbase)
            gbase += self._epn * self._row_bytes[g]
        self._group_off = tuple(group_off)

    def __call__(
        self,
        k0,
        k1,
        k2,
        gm_etc_inv,
        gm_epoch,
        ctx: StructRef[MoonepUrmaCtx],
    ):
        rank_size, _epn, ngroups = self._r, self._epn, self._ngroups
        reals = (k0, k1, k2)
        epn_rt = dtypes.int64(self._epn)
        me = dtypes.int64(ctx.rankId)
        me64 = dtypes.uint64(ctx.rankId)
        epoch = dtypes.uint32(gm_epoch[dtypes.int64(0)])
        grad_dt = (
            dtypes.float16
            if self._grad_fp16
            else (dtypes.bfloat16 if self._grad_bf16 else dtypes.float32)
        )
        elem = 2 if const_expr(_ACC_BF16 or _ACC_FP16) else 4
        max_chunk = max(self._chunks)
        # grad dtype 流水环（issue #16）：读 k+1（MTE2 入环槽）∥ 累加 k
        # 出上一槽（MTE3 atomic）——换单槽 ub_tmp 的 WAW 串行；槽深
        # depth（≥2）越高在读腿上的在飞 chunk 越多（WAR 间隔 = depth-1）
        ub_pipe = RingBuffer(
            MemLoc.UB, grad_dt, make_layout((max_chunk,)), depth=self._depth
        )
        blk = dtypes.int64(get_block_idx())
        if blk < dtypes.int64(self._n_read):
            # —— ① 头部屏障写者：my flag 写到本块分管的全部 peer
            #     （写者块 ≤ 常驻核数，先写完再等，不会分波互等；行分片
            #     模式下写者块 ⊂ 读块，其余读块只等不写）——
            if blk < dtypes.int64(self._n_writer):
                _barrier_flags_put(
                    ctx,
                    blk,
                    me64,
                    rank_size - 1,
                    self._n_writer,
                    self._per_writer,
                    gm_epoch,
                    epoch,
                )
            connection = connection_from_address(dtypes.uint64(ctx.connection))
            inv_base = tensor_address(gm_etc_inv)
            # 行分片：本块分管行 i = slice_id, slice_id+n_slices, ...
            # （split=1 时 n_slices=1、slice_id=0——与全行形态逐位一致）
            n_rem_rt = dtypes.int64(rank_size - 1)
            slice_id = blk // n_rem_rt
            i_start = slice_id
            i_step = dtypes.int64(self._n_slices)
            for read_slot in range_constexpr(int(self._per_read)):
                p_compact = blk + dtypes.int64(read_slot * int(self._n_read))
                if self._row_split > 1 and self._per_read == 1:
                    # 纯分片模式（n_read ≥ n_rem）：块号取模落 peer 槽——
                    # 每 (peer, 行) 恰一个块。per_read>1 的封顶轮转形态
                    # 不取模（模回绕会在 stride 形态重复覆盖 peer——
                    # R=128 封顶仿真实证 double write），沿用原 stride+守卫。
                    p_compact = p_compact - (p_compact // n_rem_rt) * n_rem_rt
                if p_compact < dtypes.int64(rank_size - 1):
                    peer_idx = dtypes.uint32(dtypes.uint64(p_compact))
                    p_actual = p_compact
                    if p_compact >= me:
                        p_actual = p_compact + dtypes.int64(1)
                    # —— ② 等 peer p 的 flag（副本入槽可见）——
                    spin_u32_ge(
                        ctx,
                        dtypes.uint64(HEAP_BASE + BAR_FLAGS_OFF)
                        + dtypes.uint64(p_actual) * dtypes.uint64(4),
                        epoch,
                    )
                    # —— 映射远读句柄（peer p 的副本区视图）——
                    peer = connection.get_peer(peer_idx)
                    mapped = dtypes.uint64(peer.get_mapped_addr(0))
                    # —— ③ 逐行（本块分片）：etc_inv 直读 j → 逐组逐 chunk
                    #     远读+原子直累 ——
                    for i in cannbotdsl.range(i_start, epn_rt, i_step):
                        j_t = make_tensor(
                            make_pointer(
                                dtypes.int32,
                                inv_base
                                + dtypes.uint64(i * rank_size + p_actual)
                                * dtypes.uint64(4),
                                MemLoc.GM,
                            ),
                            make_layout((1,)),
                        )
                        j = dtypes.int64(cbd.scalar.load_bypass(j_t.ptr(0)))
                        if j >= dtypes.int64(0):
                            for g in range_constexpr(_MAX_GRAD_GROUPS):
                                if g < ngroups:
                                    chunk = self._chunks[g]
                                    n_c = self._sizes[g] // chunk
                                    row_base = dtypes.uint64(
                                        HEAP_BASE
                                        + self._group_off[g]
                                        + j * self._row_bytes[g]
                                    )
                                    # 流水（issue #16）：chunk k 读入环当前槽
                                    # （MTE2）∥ 从上一槽原子累加 k 出去（MTE3）
                                    # ——两腿不同 UB 槽 = 无 WAW = MTE 通道重叠。
                                    # prologue：chunk 0 入首槽（累加在主循环消费）
                                    slot = ub_pipe.next_write()
                                    remote_src = make_tensor(
                                        make_pointer(
                                            grad_dt,
                                            dtypes.int64(
                                                mapped
                                                + row_base
                                                + dtypes.uint64(0 * chunk * elem)
                                            ),
                                            MemLoc.GM,
                                        ),
                                        make_layout((chunk,)),
                                    )
                                    mem_copy(
                                        tile_slice(slot, (chunk,), (0,)), remote_src
                                    )
                                    for c in cannbotdsl.range(
                                        dtypes.int64(0),
                                        dtypes.int64(n_c),
                                        dtypes.int64(1),
                                    ):
                                        slot_next = ub_pipe.next_write()
                                        # 预读 chunk k+1 入下一槽（末块无 k+1）
                                        c1 = c + dtypes.int64(1)
                                        if c1 < dtypes.int64(n_c):
                                            remote_src = make_tensor(
                                                make_pointer(
                                                    grad_dt,
                                                    dtypes.int64(
                                                        mapped
                                                        + row_base
                                                        + dtypes.uint64(
                                                            c1 * chunk * elem
                                                        )
                                                    ),
                                                    MemLoc.GM,
                                                ),
                                                make_layout((chunk,)),
                                            )
                                            mem_copy(
                                                tile_slice(slot_next, (chunk,), (0,)),
                                                remote_src,
                                            )
                                        # 累加 chunk k 出当前槽（MTE3 atomic）
                                        mem_copy(
                                            tile_slice(
                                                reals[g], (chunk,), (i * n_c + c,)
                                            ),
                                            tile_slice(slot, (chunk,), (0,)),
                                            copy_options(atomic_add=True),
                                        )
                                        slot = slot_next


# ---------------------------------------------------------------------------
# reduce host entry + program cache
# ---------------------------------------------------------------------------
_UBMEM_PROGRAMS = {}
_ACC_LOCK = threading.Lock()  # 名留史（原 pull/acc 共锁）
# ★ 编译烘焙走模块级全局（_get_acc_program 锁内赋值）——§5.7 烘焙铁律：
#   声明行 + 赋值行成对改，漏 global 声明则局部遮蔽、按缺省值 trace
_ACC_R = 4
_ACC_EPN = 1
_ACC_SIZES = (1,)
_ACC_ELEM = (2,)
_ACC_NGROUPS = 2
_ACC_COPY_OFF = 0
_ACC_CHUNKS = (1,)
_ACC_BF16 = True
_ACC_FP16 = False  # fp16 梯度（16 位路径同 bf16——dtype 三选一）
_ACC_BLOCKS = 8
_ACC_SPLIT = 8  # 行分片读块数/链（>1 提每链在飞读数；1=block-per-peer）
_ACC_DEPTH = 3  # 流水环槽深（UB = depth×max_chunk；2=issue#16 基线）


@host
def _reduce_ubmem_host(
    k0,
    k1,
    k2,
    gm_etc_inv,
    gm_epoch,
    ctx: StructRef[MoonepUrmaCtx],
):
    """reduce host entry（仅 UbMem）：单 kernel——block-per-peer 逐行
    映射远读 + 拷贝引擎原子直累。"""
    _UbmemReduceAccKernel(
        _ACC_R,
        _ACC_EPN,
        _ACC_SIZES,
        _ACC_ELEM,
        _ACC_NGROUPS,
        _ACC_COPY_OFF,
        _ACC_CHUNKS,
        _ACC_BLOCKS,
        _ACC_BF16,
        _ACC_FP16,
        row_split=_ACC_SPLIT,
        depth=_ACC_DEPTH,
    )[_ACC_BLOCKS](k0, k1, k2, gm_etc_inv, gm_epoch, ctx)


def _get_acc_program(spec):
    """直累 program 缓存（_ReduceSpec 键控——锁内原子烘焙）。"""
    with _ACC_LOCK:
        prog = _UBMEM_PROGRAMS.get(spec)
        if prog is not None:
            return prog
        global _ACC_R, _ACC_EPN, _ACC_SIZES, _ACC_ELEM
        global _ACC_NGROUPS, _ACC_COPY_OFF, _ACC_CHUNKS
        global _ACC_BF16, _ACC_BLOCKS, _ACC_FP16
        global _ACC_SPLIT, _ACC_DEPTH
        _ACC_R = spec.rank_size
        _ACC_EPN = spec.epn
        _ACC_SIZES = spec.sizes
        _ACC_ELEM = spec.elem_bytes
        _ACC_NGROUPS = spec.ngroups
        _ACC_COPY_OFF = spec.copy_off
        _ACC_CHUNKS = spec.chunks
        _ACC_BF16 = spec.grad_bf16
        _ACC_FP16 = spec.grad_fp16
        _ACC_BLOCKS = spec.blocks
        _ACC_SPLIT = spec.split
        _ACC_DEPTH = spec.depth
        i32 = dtypes.int32
        dt = (
            dtypes.float16
            if spec.grad_fp16
            else (dtypes.bfloat16 if spec.grad_bf16 else dtypes.float32)
        )
        specs = [TensorSpec((2 * spec.epn * s,), dt) for s in spec.sizes]
        specs.append(TensorSpec((spec.epn, spec.rank_size), i32))
        specs.append(TensorSpec((1,), i32))  # epoch
        specs.append(StructRef[MoonepUrmaCtx])
        prog = cannbotdsl.compile(_reduce_ubmem_host, *specs)
        _UBMEM_PROGRAMS[spec] = prog
        return prog


def _copy_grads_to_ws(
    context, local_grads, epn, widths, grad_bf16, grad_fp16, copy_off
):
    """发送侧：副本行 local_grads[g][epn:] 拷入 workspace 副本区。

    副本区 [epn, Wg] × 组（grad_slot_off 后，紧随权重发布区——
    每 rank 只放自己的副本行，owner 远读。布局与 push 接收区共享
    基址公式但无 R 维度——内存恒定 [epn, Wg]×组。

    dst 视图 1 槽缓存（同 shape/dtype/copy_off 复用——每拍
    narrow/view/view × 组的 aten 派发是 host 段 ~70 ops/拍的实证
    削减项之一；源侧 local_grads 每拍新张量不可缓存）。
    """
    dt = (
        torch.float16 if grad_fp16 else (torch.bfloat16 if grad_bf16 else torch.float32)
    )
    elem = 2 if (grad_bf16 or grad_fp16) else 4
    ws = context.workspace
    key = (
        tuple(int(w) for w in widths),
        bool(grad_bf16),
        bool(grad_fp16),
        int(copy_off),
        int(epn),
    )
    cached = getattr(context, "_pull_ws_copy_views", None)
    if cached is None or cached[0] != key:
        views = []
        base = HEAP_BASE + int(copy_off)
        for w in widths:
            region = epn * w * elem
            views.append(ws.narrow(0, base, region).view(dt).view(epn, w))
            base += region
        cached = (key, views)
        context._pull_ws_copy_views = cached
    for g, w in enumerate(widths):
        cached[1][g].copy_(local_grads[g][epn:])


def reduce_grad(
    context,
    experts_to_copy,
    local_grads,
    epn,
    *,
    slot=0,
    slots=None,
    async_finish=False,
):
    """owner 直累 reduce_grad：副本梯度归并回 home（full-dW 直入形态）。

    ``local_grads`` 为 2 或 3 组 ``[2*epn, Wg]`` 张量——GMM 对全量
    （home+副本）权重视图反传的原生形态，行序与 prefetch 视图一致（前
    epn 行 home 种子、后 epn 行副本 dW）。组数/各组行宽从张量解析；
    dtype 须与构造定型值一致（交织型传 ``[2*epn, 2W] + [2*epn, W]``）。

    梯度搬运（owner 直累协议）：发送侧 host 拷贝副本行入 workspace 副本区
    ``[epn, Wg]``（恒定 ~19.3MB，不随 R 增长）；owner 侧单 kernel 映射
    远读各 peer 副本区 → UB 流水环 → 拷贝引擎原子直累 home 段（原位）。

    Args:
        local_grads: list/tuple，每组 ``[2*epn, Wg]`` 连续张量（fp32/bf16，
                     全组同 dtype，与构造 grad_dtype 定型值一致）。
        async_finish: False（缺省）阻塞到全组归并完成（返回 event=None）；
                     True 返回发起流 torch.npu.Event（wait_event 汇合后读
                     归并结果）。
        slot: 窗口池槽位（与 prefetch 同族约定——reduce 侧仅校验一致性，
            归并落点与槽位无关）。
        slots: 窗口池层数（None = context.window_slots 定型值；显式给出时
            须与定型值一致，否则 ValueError）。

    Returns:
        变长返回：``async_finish=False``（缺省）只返回
        ``home_views`` 单组视图元组；``async_finish=True`` 返回
        ``(home_views, event)``。``home_views`` 为各组归并结果的
        ``[epn, Wg]`` 视图（与输入张量同存储——原位归并的直接读柄，归并
        完成后即全量 dW）；``event`` 同步模式为 None、异步为发起流事件。
    """
    # 归一化 + 行数/宽度/dtype/调用序守卫（组数与行宽从张量解析）。
    local_grads, widths, grad_bf16, grad_fp16 = _normalize_grad_views(
        context, local_grads, epn
    )

    local_grads[0].dtype
    elem_bytes = 2 if (grad_bf16 or grad_fp16) else 4
    # 跨流护栏：本算子与前序通信算子共用 epoch 握手机制，须按发起序
    # 执行（同流为快路径零开销）
    sync_transport_stream(context)

    if context.rank_size == 1:
        # 单 rank 无副本可归并（etc 恒全 -1）——home 段已是全量 dW；
        # 变长返回契约：sync 只返回 home_views；async 返回
        # (home_views, event=None)——无集合操作，事件立即可汇合
        home_views = tuple(t[:epn] for t in local_grads)
        if async_finish:
            return home_views, None
        return home_views

    # —— 窗口池约定校验（与 prefetch 同族；归并落点与槽位无关）——
    if slots is None:
        slots = context.window_slots or 1
    elif context.window_slots is not None and int(slots) != int(context.window_slots):
        raise ValueError(
            f"reduce_grad slots={slots} 与窗口池定型层数 {context.window_slots} 不一致"
        )

    # —— 副本区布局（紧随权重发布区；[epn, Wg] × 组——pull 恒定，不随 R 增长）——
    scale_widths, weight_bytes = _grad_region_params(context)
    copy_off = grad_slot_off(
        epn,
        widths=widths,
        scale_widths=scale_widths,
        weight_bytes=weight_bytes,
        slots=slots,
    )
    rank_size = context.rank_size
    copy_bytes = epn * sum(widths) * elem_bytes
    check_heap_capacity(
        context,
        HEAP_BASE + copy_off + copy_bytes + LAYOUT_TAIL_BYTES,
        "pull 梯度副本区",
    )

    # —— 槽视图一致性守卫（pull_slots_used 契约沿用）——
    used = context.pull_slots_used
    if used is None:
        context.pull_slots_used = slots
    elif used != slots:
        raise ValueError(
            f"梯度槽视图 slots={slots} 与既有 slots={used}"
            " 不一致——副本区基址会错位（请保持全链 slots 一致）"
        )

    # —— owner 直累数据面（发送侧拷入 + owner 映射远读直累）——
    # 发送侧：host 把副本行拷入 workspace 副本区（一次 GM 拷贝 ~19MB < 1ms）
    # 归并 kernel 只读 etc 映射行（无未映射行入和——pull 天然无陈旧行问题）
    _copy_grads_to_ws(context, local_grads, epn, widths, grad_bf16, grad_fp16, copy_off)
    # 归并 kernel：owner 远读 peer 副本 + vf 流式归并
    gm_epoch = next_barrier_epoch(context)
    # 块数封顶常驻核数（950PR 64 块）——R>64 时不随 R 增长：屏障写者与读块
    # 都按 peer 轮转（peer_block_split），避免写者被分波调度推到第二波互等
    # （issue #14 B1/B2）。归并 tile 条带同样取此块数。
    blocks = resident_blocks(_HEADER_BARRIER_CAP)
    chunk_limit = (
        _CHUNK_UB_LIMIT_BF16 if (grad_bf16 or grad_fp16) else _CHUNK_UB_LIMIT_FP32
    )
    chunks = tuple(pick_weight_chunk(w, chunk_limit) for w in widths)
    # 行宽须为 VL=64 倍数（归并 tile 步进=VL——非倍数时尾 tile 越界 UB；
    # MoonepBuffer 构造期已查，裸 API 路径在此 fail-fast）
    for w in widths:
        if w % VL:
            raise ValueError(f"Wg={w} 须为 {VL} 的倍数（归并 VL 尾块越界防护）")
    ngroups = len(local_grads)
    # 3 槽：真实组前缀 + pad 尾置（缺省槽复用末组——kernel 不读取）
    sizes = tuple(widths) + (widths[-1],) * (_MAX_GRAD_GROUPS - ngroups)
    # 行分片/环深旋钮（缺省生产值；MOONEP_ACC_SPLIT/MOONEP_ACC_DEPTH
    # 供 bench 矩阵实验——spec 进编译键，环境变量不直接进 kernel）
    acc_split = int(os.environ.get("MOONEP_ACC_SPLIT", "8"))
    acc_depth = int(os.environ.get("MOONEP_ACC_DEPTH", "3"))
    spec = _ReduceSpec(
        epn=epn,
        sizes=sizes,
        elem_bytes=(elem_bytes,) * _MAX_GRAD_GROUPS,
        ngroups=ngroups,
        copy_off=copy_off,
        chunks=chunks + (chunks[-1],) * (_MAX_GRAD_GROUPS - ngroups),
        rank_size=rank_size,
        grad_bf16=grad_bf16,
        grad_fp16=grad_fp16,
        blocks=blocks,
        split=acc_split,
        depth=acc_depth,
    )
    # 3 槽实参（缺省槽复用末组张量——kernel 不读取）
    reals3 = tuple(local_grads) + (local_grads[-1],) * (_MAX_GRAD_GROUPS - ngroups)
    # 逆映射 [epn, R]：etc_inv[i, p] = j（peer p 副本区承载本 rank 行 i
    # 的槽位；-1 = 无映射）。device 侧向量化构建（无 host 同步）——kernel
    # 每 tile 一次 R×4B 行拷贝即得全部源，消逐 peer 的 etc 拷贝 + fence
    # + epn 次标量扫描
    # 1 槽 etc_inv 记忆化（同 etc 张量同版本 → 同内容，纯 host 派生
    # 可复用——多 micro-batch/梯度累积循环同 etc 复用实证形态；槽持
    # etc 引用防 ptr/id 复用误撞）
    inv_cache = getattr(context, "_pull_etc_inv_cache", None)
    inv_key = (id(experts_to_copy), experts_to_copy._version, epn)
    if inv_cache is not None and inv_cache[0] == inv_key:
        etc_inv = inv_cache[1]
    else:
        dev = experts_to_copy.device
        targets = context.rank * epn + torch.arange(epn, device=dev, dtype=torch.int32)
        eq = experts_to_copy.unsqueeze(-1) == targets  # [R, epn(j), epn(i)]
        has = eq.any(dim=1)  # [R, epn(i)]
        # argmax 恒返 int64——须显式转回 int32（TensorSpec 声明 int32，
        # int64 张量按 4B 步长解读 = 静默错值 + 野偏移远读）
        j_idx = eq.to(torch.int32).argmax(dim=1).to(torch.int32)  # [R, epn(i)]
        etc_inv = (
            torch.where(has, j_idx, torch.tensor(-1, dtype=torch.int32, device=dev))
            .t()
            .contiguous()
        )  # [epn, R]
        context._pull_etc_inv_cache = (
            inv_key,
            etc_inv,
            experts_to_copy,
        )
    # 直累路径（UbMem 映射读 + 拷贝引擎原子加——master 0.8 设计回归；
    # URMA 两段式 SQE 路径已随传输退役删除）
    _get_acc_program(spec)(
        *(t.reshape(-1) for t in reals3),
        etc_inv,
        gm_epoch,
        context.ctx_tensor,
    )
    # 尾部屏障（入队形态）：保证全组 pull kernel 完成后下一轮的 host 侧
    # 副本区拷贝才入队（_copy_grads_to_ws 在 kernel 前执行——若对端上轮
    # 远读未完即被覆写，会归并到新旧混合行）。async 模式由返回事件一并
    # 覆盖（barrier kernel 与 pull 同流序）。
    sym_barrier(context, sync=False)

    # 变长返回契约：sync 只返回 home_views 单值；async 二元组。
    event = finish_op(context, async_finish)
    home_views = tuple(t[:epn] for t in local_grads)
    if async_finish:
        return home_views, event
    return home_views
