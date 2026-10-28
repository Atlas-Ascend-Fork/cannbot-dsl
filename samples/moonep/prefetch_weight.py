# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""prefetch_weight: push replica expert weights from home to consumer ranks.

Direct-push single-phase: home uses xfer.put to write each replica row into
the consumer's symmetric prefetch slot. A trailing barrier guards against
slot-reuse races. Supports multiple weight groups (gate/up/down) in a single
call.

副本清单由 kernel 内向量扫描 etc 生成（UB 内 64-lane 比较，不落 GM）：
host 侧零回读/零 Python 扫描/零 H2D，编译键与路由结构解耦。
"""

import threading
from typing import NamedTuple

import torch

import cannbotdsl
from cannbotdsl import dtypes
from cannbotdsl import TensorSpec
from cannbotdsl.types.datastruct import StructRef
from cannbotdsl.buffer import Buffer
from cannbotdsl.tensor import MemLoc, tile_slice
from cannbotdsl.lang.constexpr import range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.distributed import (
    CommContext,
    create_transfer_context,
    SymBuffer,
    TransferRole,
)
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.reg import (
    mask_and,
    update_mask,
    vadds,
    vdups,
    vges,
    vload,
    vlts,
    vselect,
    vstore,
)
from cannbotdsl.ops.sync import PIPE, vec_sync_notify, vec_sync_wait

from moonep_context import (
    HEAP_BASE,
    check_heap_capacity,
    finish_op,
    hccl_buffer_base,
    normalize_group_list,
    prefetch_weight_off,
    reduce_stg_off,
    resident_blocks,
    sync_transport_stream,
)
from npu_blob import npu_from_blob

_MAX_PUSH_BLOCKS = 64  # 块数经验上限；实际 = min(R-1, resident_blocks(cap))
_PUSH_VL = 64  # int32 向量寄存器 lane 数（256B / 4B）
_PUSH_EVENT_ID = 4  # V→S 同步 event（plan=7、reduce=6、barrier=5）


def _epn_pad(epn):
    """etc 行按 VL 向上取整，末块 vload 不越界（余 lane 由 exec mask 屏蔽）。"""
    return ((int(epn) + _PUSH_VL - 1) // _PUSH_VL) * _PUSH_VL


def _push_layout(rank_size):
    """返回 (块数, 每块负责的 peer 数)。

    - 块数不得超过本机可同时常驻的 AIV 数（`resident_blocks`，含流的控核
      配额）——R-1 可能大于核数，不能直接按 R-1 起块；
    - peer 按 round-robin 分给各 block，**每个 peer 仍只由一个 block 写/
      flush**（同一 peer 通道被多个 block 并发访问会挂，R=8+大 W 的 aivec
      271 即此）。
    """
    n_rem = max(rank_size - 1, 0)
    nblk = min(max(n_rem, 1), resident_blocks(_MAX_PUSH_BLOCKS))
    per_blk = (n_rem + nblk - 1) // nblk if n_rem > 0 else 1
    return nblk, per_blk


@kernel
class _MoonepPrefetchPushKernel:
    """home 把有副本需求的真实专家行直推到各消费 rank 的预取槽。

    block 与目的 peer 是“多对一分配、一对一访问”：peer 按 round-robin 分给
    `min(R-1, resident_blocks())` 个 block，**每个 peer 通道只由一个 block
    写和 flush**。早期“条目按 block 取模分派”会让多个 block 并发 put+flush
    同一 peer，R=8 + 大 W 实测触发 aivec 271（块数压到 1 可规避）；而直接
    按 R-1 起块又会在 R 大于本机 AIV 数时超核——块数必须由 resident_blocks
    约束，块数上限内的多 peer 由同一 block 串行处理。

    副本清单在 kernel 内算，且**用向量比较代替逐元素标量扫描**：整行 etc
    mem_copy 进 UB 后，64 lane 并行判 `lo <= e < hi`，命中写本地行号、未命中
    写 -1（vselect），清单直接留在 UB（不落 GM）；随后标量按 lane 顺序读
    UB 并发 put。UB 读前用 V→S 的 notify/wait（set_flag/wait_flag）只等这条
    vstore 落地，不用全 pipe 的 vec_sync_all。
    """

    def __init__(self, rank, rank_size, epn, w_elems, weight_offs, nblk, per_block):
        self._me = int(rank)
        self._r = int(rank_size)
        self._epn = int(epn)
        self._w = int(w_elems)
        self._offs = tuple(int(o) for o in weight_offs)
        self._num_g = len(self._offs)
        self._nblk = int(nblk)
        self._per_block = int(per_block)
        self._epn_pad = _epn_pad(self._epn)

    def __call__(
        self,
        kr_real0,
        kr_real1,
        kr_real2,
        gm_etc,
        ktx: StructRef[CommContext],
    ):
        # 参数名与 plan/reduce host 刻意错开（位置绑定，名不作数）
        world_size, epn, w_elems, me = self._r, self._epn, self._w, self._me
        reals = (kr_real0, kr_real1, kr_real2)
        wsyms = []
        for g in range_constexpr(self._num_g):
            wsyms.append(
                SymBuffer(
                    ktx,
                    shape=(2 * epn, w_elems),
                    dtype=dtypes.bfloat16,
                    rank_size=world_size,
                    base_offset=self._offs[g],
                )
            )
        ub_xfer = Buffer(MemLoc.UB, (512,), dtypes.uint8)
        xfer = create_transfer_context(ktx, ub_xfer, role=TransferRole.VEC)

        blk = dtypes.int64(get_block_idx())
        n_rem = dtypes.int64(world_size - 1)
        # 每个 block 负责 per_block 个 peer（round-robin；slot 维编译期展开）
        for peer_slot in range_constexpr(self._per_block):
            peer_id = blk + dtypes.int64(peer_slot * self._nblk)
            if peer_id < n_rem:
                # peer 序号去掉自己：>= me 的顺延一位
                peer = (
                    peer_id if peer_id < dtypes.int64(me) else peer_id + dtypes.int64(1)
                )
                # 每个 peer 独立一对 UB 缓冲，避免跨 slot 的读写冒险
                ub_etc = Buffer(MemLoc.UB, (1, self._epn_pad), dtypes.int32)
                ub_row = Buffer(MemLoc.UB, (1, self._epn_pad), dtypes.int32)
                self._scan_peer(gm_etc, peer, ub_etc, ub_row)

                # 按 lane 顺序读清单：命中 lane 的本地行号非负，未命中为 -1
                for i in cannbotdsl.range(dtypes.int64(epn)):
                    row = dtypes.int64(ub_row[dtypes.int64(0), dtypes.int64(i)])
                    if row >= dtypes.int64(0):
                        for g in range_constexpr(self._num_g):
                            xfer.put(
                                peer,
                                tile_slice(wsyms[g], (1, w_elems), (epn + i, 0)).peer(
                                    peer
                                ),
                                tile_slice(reals[g], (1, w_elems), (row, 0)),
                            )
                xfer.flush(peer)

    @jit
    def _scan_peer(self, gm_etc, peer, ub_etc, ub_row):
        """向量扫描 etc[peer, :]：命中本 rank 专家区间的 lane 写本地行号，否则 -1。

        64 lane 并行比较，结果留在 UB；V→S 的 notify/wait 保证随后的标量读
        能看到 vstore 的结果。余 lane（epn 不足一个 VL）由 exec mask 屏蔽。
        """
        epn, pad, vl = self._epn, self._epn_pad, _PUSH_VL
        lo_e, hi_e = self._me * epn, (self._me + 1) * epn
        mem_copy(
            tile_slice(ub_etc, (1, epn), (0, 0)),
            tile_slice(gm_etc, (1, epn), (peer, 0)),
        )
        with vf(mode="simd"):
            minus_one = vdups(dtypes.int32(-1), dtypes.int32)
            for off in range_constexpr(0, pad, vl):
                lanes = update_mask(min(vl, epn - off), elem_bits=32)[0]
                expert = vload(ub_etc, off)
                in_range = mask_and(
                    vges(expert, lo_e, mask=lanes),
                    vlts(expert, hi_e, mask=lanes),
                    exec_mask=lanes,
                )
                row = vadds(expert, -lo_e, mask=lanes)
                vstore(
                    ub_row,
                    off,
                    vselect(row, minus_one, cond_mask=in_range),
                    mask=lanes,
                )
        # V→S：vstore 完成后，标量单元才能读 ub_row
        vec_sync_notify(PIPE.V, PIPE.S, _PUSH_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.S, _PUSH_EVENT_ID)


class _PushConfig(NamedTuple):
    """直推 program 的编译期配置（同时用作 program 缓存键）。"""

    me: int
    rank_size: int
    epn: int
    w_elems: int
    weight_offs: tuple
    nblk: int
    per_block: int


def _make_push_host(cfg):
    """用闭包携带编译期常量，生成该配置专属的 @host launch 函数。

    不再用模块级全局变量烘焙：不同配置各自拥有独立函数对象/闭包，并发编译
    不会互相污染，也不需要"锁内改全局再 trace"的窗口。
    """

    @host
    def _push_host(gm_real0, gm_real1, gm_real2, gm_etc, ctx: StructRef[CommContext]):
        _MoonepPrefetchPushKernel(
            cfg.me,
            cfg.rank_size,
            cfg.epn,
            cfg.w_elems,
            cfg.weight_offs,
            cfg.nblk,
            cfg.per_block,
        )[cfg.nblk](gm_real0, gm_real1, gm_real2, gm_etc, ctx)

    return _push_host


_PUSH_PROGRAMS = {}
_PUSH_LOCK = threading.Lock()


def _get_push_program(epn, w_elems, weight_offs, rank_size, me):
    """按 `_PushConfig` 键控的直推 program——不含路由结构（清单由 kernel 内
    扫 etc 表生成，路由变化不重编译）。
    """
    nblk, per_blk = _push_layout(rank_size)
    cfg = _PushConfig(me, rank_size, epn, w_elems, tuple(weight_offs), nblk, per_blk)
    with _PUSH_LOCK:
        prog = _PUSH_PROGRAMS.get(cfg)
        if prog is None:
            prog = cannbotdsl.compile(
                _make_push_host(cfg),
                TensorSpec((epn, w_elems), dtypes.bfloat16),
                TensorSpec((epn, w_elems), dtypes.bfloat16),
                TensorSpec((epn, w_elems), dtypes.bfloat16),
                TensorSpec((rank_size, epn), dtypes.int32),
                StructRef[CommContext],
            )
            _PUSH_PROGRAMS[cfg] = prog
    return prog


def materialize_weights(local_weights):
    """对称窗权重视图 → 普通显存拷贝。

    npu_grouped_matmul 的 host 路径对 HCCL 对称窗支撑的非连续视图（如 .mT）
    会 segfault；进 GMM 前先用本函数把视图落地为普通显存张量。
    """
    if isinstance(local_weights, torch.Tensor):
        return local_weights.contiguous()
    return [w.contiguous() for w in local_weights]


def create_weight_view(context, epn, w_elems, *, group=0, groups=1, slot=0, slots=1):
    """在对称缓冲区上创建 [2*epn, W] bf16 权重视图（真实行 + 预取槽同址）。

    home 直推要求预取槽位于对称区，因此权重视图必须经本函数创建，
    不要再 `torch.empty(...).npu()`。同 plan 的多组权重（gate/up/down）
    用 group=0..groups-1 分别创建，组内布局与 prefetch_weight 的直推寻址一致。

    多槽窗口池（slot=0..slots-1）：槽 s 的组 g 视图基址 =
    prefetch_weight_off + (s*groups + g)*整视图步长。双槽轮转（层 L 用
    L%2 槽）使 GEMM 可直接消费当前槽视图、下一层 prefetch 写另一槽，
    免集成侧快照拷贝（单槽下并发覆写实测 SIGSEGV）。槽位数由首个调用
    定型，后续调用须一致；缺省 slot=0/slots=1 与单槽布局字节一致。
    """
    if context.last_plan_shape is None:
        raise RuntimeError(
            "create_weight_view 需先调用 moonep_plan（依赖其形状与窗口布局）"
        )
    if group < 0 or group >= groups:
        raise ValueError(f"group={group} 超出组数 {groups}")
    if slot < 0 or slot >= slots:
        raise ValueError(f"slot={slot} 超出槽位数 {slots}")
    if context.window_slots is None:
        context.window_slots = slots
    elif context.window_slots != slots:
        raise RuntimeError(
            f"窗口槽位数不一致（已有 {context.window_slots}，本次 {slots}）"
        )
    if context.layout_groups is None:
        context.layout_groups = groups
    elif context.layout_groups != groups:
        raise RuntimeError(
            f"窗口组数不一致（已有 {context.layout_groups}，本次 {groups}）"
        )
    # 视图缓存：同一块对称堆上的非拥有句柄，重复创建直接复用——省掉每步的
    # HCCL 缓冲查询（C API）、容量校验与 npu_from_blob 扩展调用
    key = (int(epn), int(w_elems), int(group), int(groups), int(slot), int(slots))
    cached = context.weight_views.get(key)
    if cached is not None:
        return cached
    # 容量前置校验：整个多槽窗口池（staging 区由 reduce_grad 自身校验，
    # 不殃及 prefetch-only 使用形态）
    check_heap_capacity(
        context,
        HEAP_BASE + reduce_stg_off(epn, w_elems, num_groups=groups, slots=slots),
        "对称堆布局（plan 交换区 + 多槽权重发布区）",
    )
    view_bytes = 2 * epn * w_elems * 2  # 组步长 = 完整 [2epn, W] bf16 视图字节数
    weight_off = prefetch_weight_off() + (slot * groups + group) * view_bytes
    addr = hccl_buffer_base(context) + HEAP_BASE + weight_off
    view = npu_from_blob(addr, (2 * epn, w_elems), "bf16")
    context.weight_views[key] = view
    return view


def _normalize_weight_groups(context, real_weights, local_weights):
    """归一化为 list 并校验组数/行宽/调用序，返回 (real_weights, local_weights, w_elems)。"""
    if isinstance(real_weights, torch.Tensor):
        real_weights = [real_weights]
    local_weights, w_elems = normalize_group_list(
        context, local_weights, "prefetch_weight"
    )
    if len(real_weights) != len(local_weights):
        raise ValueError("real_weights 与 local_weights 组数必须一致且至少一组")
    if any(w.shape[1] != w_elems for w in real_weights):
        raise ValueError("real_weights 行宽须与视图一致（直推数据源，W=H*H'）")
    return real_weights, local_weights, w_elems


def _pad_to_three(items):
    """不足 3 组时以末组占位（kernel 按 3 组编译，多出的槽位用编译期展开消除）。"""
    padded = list(items)
    while len(padded) < 3:
        padded.append(padded[-1])
    return padded


def sync_transport(context, real_weights, local_weights, epn):
    """home 行同步：real_weights 的本 rank 真实专家行写入视图前 epn 行。

    调用方不再需要自己做 `local_weights[g][:epn] = real_weights[g]`——
    本函数在直推前完成 home 行写入，GEMM 侧读视图前 epn 行即为当前参数。
    real_weights 与视图 home 行相同地址时自动跳过（参数直接建在视图上的
    推荐形态，零拷贝：免掉每步每组 epn*W*2 字节的整块 D2D）。
    """
    for g, (real_w, view_w) in enumerate(zip(real_weights, local_weights)):
        if real_w.data_ptr() == view_w[:epn].data_ptr():
            continue
        view_w[:epn] = real_w


def _publish_offsets(context, local_weights, epn, w_elems):
    """校验视图落在多槽窗口区内且按整视图步长对齐，返回各组发布区偏移。

    偏移是相对对称堆基址的字节数（`SymBuffer.base_offset` 口径）；槽位信息
    由视图地址自身携带，kernel 无需额外参数。
    """
    buf_base = hccl_buffer_base(context) + HEAP_BASE
    view_bytes = 2 * epn * w_elems * 2  # 组步长 = 完整 [2epn, W] bf16 视图字节数
    slots = context.window_slots or 1
    groups = context.layout_groups or len(local_weights)
    window_bytes = slots * groups * view_bytes
    offs = []
    for g, w_g in enumerate(local_weights):
        publish_off = w_g.data_ptr() - buf_base
        rel = publish_off - prefetch_weight_off()
        if rel < 0 or rel >= window_bytes or rel % view_bytes != 0:
            raise ValueError(
                "local_weights 必须是 create_weight_view 创建的对称区视图"
                f"（第 {g} 组相对偏移 {rel} 不在窗口区 [0, {window_bytes}) 内"
                " 或未按整视图步长对齐）"
            )
        offs.append(publish_off)
    return offs


def prefetch_weight(
    context, experts_to_copy, real_weights, local_weights, epn, *, async_finish=False
):
    """home 直推预取本 rank 托管的远程专家权重（单阶段，无 publish/get）。

    Args:
        context: MoonepContext
        experts_to_copy: [R, epn] int32，moonep_plan 输出（-1 填充；
            每行副本槽至多来自一个远程 home rank）
        real_weights: 普通 NPU 张量 [epn, W] bf16（或多组 list/tuple），
            真实本地专家权重（当前参数值）。本函数会先把它写入视图前 epn 行（home 行），
            再直推副本——调用方无需自己做 home 行写回。
        local_weights: `create_weight_view` 创建的 [2*epn, W] bf16 对称区权重视图
            （或多组 list/tuple，组数与 real_weights 一致）。
            前 epn 行为真实本地专家（本函数写入），后 epn 行为预取槽（本函数填充）。
        epn: 每 rank 真实本地专家数
        async_finish: 完成语义开关（缺省 False：末尾独立 sym_barrier +
            host 级流等待，返回即全组本轮直推完成，连续调用无槽位竞态）。
            **侧流并发调用传 True**：跳过末尾屏障及其 host 同步，返回
            发起流上的 NPU 事件——调用方 `wait_event` 汇合，跨 rank 槽位
            复用安全由调用方在合适位置（如下一轮迭代的 barrier）负责；
            plan 延迟溢出检查顺延到下一个我方算子尾部。

    说明：
    - home 仅推送有副本需求的行（规划不产生自承载副本：home 的 order 恒为 1）。
    - 目的 peer 按 round-robin 分给 min(R-1, AIV 数) 个 block，每 peer
      只由一个 block 写/flush。
    """
    real_weights, local_weights, w_elems = _normalize_weight_groups(
        context, real_weights, local_weights
    )
    sync_transport_stream(context)  # 跨流护栏
    sync_transport(context, real_weights, local_weights, epn)
    # 视图地址推导各组发布区偏移（槽位由视图携带），并校验窗口区对齐
    publish_offs = _publish_offsets(context, local_weights, epn, w_elems)

    # 多组融合发射：每批最多 3 组（dsl 发射器 host 开销 ~550µs/launch，
    # 独立逐组发射会在 ctx 流上拖出设备空窗）；不足 3 组以末组占位
    # （编译期组数展开消除未用分支）。批次间同流串行（清单在 UB 内，随
    # kernel 生命周期结束，无跨批复用）
    for b0 in range(0, len(local_weights), 3):
        batch_rw = _pad_to_three(real_weights[b0 : b0 + 3])
        batch_offs = tuple(publish_offs[b0 : b0 + 3])
        _get_push_program(epn, w_elems, batch_offs, context.rank_size, context.rank)(
            batch_rw[0],
            batch_rw[1],
            batch_rw[2],
            experts_to_copy,
            context.ctx_tensor,
        )
    # 尾段契约见 finish_op（阻塞=独立 barrier+流级等待+溢出检查；async=返回事件）
    return finish_op(context, async_finish)
