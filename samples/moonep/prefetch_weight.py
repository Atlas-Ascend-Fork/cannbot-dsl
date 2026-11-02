# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""prefetch_weight：home 把副本专家权重直推到各消费 rank 的对称预取槽。

设计方案：

- 输入固定为 3 组权重（gate/up/down），一次 launch 完成三组直推；组数不做
  参数化，也不做 padding。传入非 3 组时由元组解包自然报错。
- 权重视图必须由 create_weight_view 创建：前 epn 行是真实本地专家，后 epn
  行是预取槽，视图位于 HCCL 对称堆的发布区。调用前调用方需已执行
  moonep_plan 并据此创建视图。
- 副本清单不落 GM、不做 H2D：kernel 把 etc[peer] 整行搬进 UB，按设备侧
  get_vf_len() 查到的 VL 换算 lane 数（int32 下 64 lane），向量比较
  （lo <= e < hi）生成本地行号，命中写行号、未命中写 -1，随后按 lane 顺序
  读清单发起 put + flush；尾块由 mask_counter 屏蔽。
- block 与 peer 是多对一分配、一对一访问：peer 按 round-robin 分给
  min(R-1, resident_blocks()) 个 block，每个 peer 通道只由一个 block 写和
  flush，避免多 block 并发访问同一通道。
- 收尾默认阻塞：独立 sym_barrier 等待全组完成后再返回，连续调用无槽位竞态；
  async_finish=True 时只发射 barrier 并返回发起流事件，由调用方 wait_event
  汇合。
"""

import threading
from typing import NamedTuple

import torch

import cannbotdsl
from cannbotdsl import dtypes, make_layout
from cannbotdsl import TensorSpec
from cannbotdsl.buffer import Buffer
from cannbotdsl.types.datastruct import StructRef
from cannbotdsl.tensor import MemLoc, tile_slice
from cannbotdsl.lang.constexpr import range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx, get_vf_len
from cannbotdsl.ops.distributed import (
    CommContext,
    create_transfer_context,
    SymBuffer,
    TransferRole,
)
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.reg import (
    mask_and,
    mask_counter,
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
    prefetch_weight_off,
    reduce_stg_off,
    resident_blocks,
    sync_transport_stream,
)
from npu_blob import npu_from_blob

_MAX_PUSH_BLOCKS = 64
_PUSH_EVENT_ID = 4


def _push_layout(rank_size):
    n_rem = max(rank_size - 1, 0)
    nblk = min(max(n_rem, 1), resident_blocks(_MAX_PUSH_BLOCKS))
    per_blk = (n_rem + nblk - 1) // nblk if n_rem > 0 else 1
    return nblk, per_blk


@kernel
class _MoonepPrefetchPushKernel:
    def __init__(self, rank, rank_size, epn, w_elems, weight_offs, nblk, per_block):
        self._me = int(rank)
        self._r = int(rank_size)
        self._epn = int(epn)
        self._w = int(w_elems)
        self._offs = tuple(int(o) for o in weight_offs)
        self._num_g = len(self._offs)
        self._nblk = int(nblk)
        self._per_block = int(per_block)

    def __call__(
        self,
        kr_real0,
        kr_real1,
        kr_real2,
        gm_etc,
        ktx: StructRef[CommContext],
    ):
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
        ub_xfer = Buffer(MemLoc.UB, dtypes.uint8, make_layout((512,)))
        xfer = create_transfer_context(ktx, ub_xfer, role=TransferRole.VEC)

        blk = dtypes.int64(get_block_idx())
        n_rem = dtypes.int64(world_size - 1)
        for peer_slot in range_constexpr(self._per_block):
            peer_id = blk + dtypes.int64(peer_slot * self._nblk)
            if peer_id < n_rem:
                peer = (
                    peer_id if peer_id < dtypes.int64(me) else peer_id + dtypes.int64(1)
                )
                ub_etc = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, self._epn)))
                ub_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, self._epn)))
                self._scan_peer(gm_etc, peer, ub_etc, ub_row)

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
        epn = self._epn
        # 矢量寄存器 VL 由设备侧接口查询（字节），按 int32 位宽换算 lane 数。
        vl = get_vf_len() * dtypes.int64(8) // dtypes.int64(dtypes.int32.bits)
        # 步长固定为 1、按 VL 换算偏移，保持 VF 硬件循环可识别
        n_chunks = (dtypes.int64(epn) + vl - 1) // vl
        lo_e, hi_e = self._me * epn, (self._me + 1) * epn
        mem_copy(
            tile_slice(ub_etc, (1, epn), (0, 0)),
            tile_slice(gm_etc, (1, epn), (peer, 0)),
        )
        with vf(mode="simd"):
            minus_one = vdups(dtypes.int32(-1), dtypes.int32)
            remaining = mask_counter(epn)
            for chunk in cannbotdsl.range(n_chunks):
                off = chunk * vl
                lanes, remaining = update_mask(remaining, elem_bits=32)
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
        vec_sync_notify(PIPE.V, PIPE.S, _PUSH_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.S, _PUSH_EVENT_ID)


class _PushConfig(NamedTuple):
    me: int
    rank_size: int
    epn: int
    w_elems: int
    weight_offs: tuple
    nblk: int
    per_block: int


def _make_push_host(cfg):
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
    if isinstance(local_weights, torch.Tensor):
        return local_weights.contiguous()
    return [w.contiguous() for w in local_weights]


def create_weight_view(context, epn, w_elems, *, group=0, groups=1, slot=0, slots=1):
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
    key = (int(epn), int(w_elems), int(group), int(groups), int(slot), int(slots))
    cached = context.weight_views.get(key)
    if cached is not None:
        return cached
    check_heap_capacity(
        context,
        HEAP_BASE + reduce_stg_off(epn, w_elems, num_groups=groups, slots=slots),
        "对称堆布局（plan 交换区 + 多槽权重发布区）",
    )
    view_bytes = 2 * epn * w_elems * 2
    weight_off = prefetch_weight_off() + (slot * groups + group) * view_bytes
    addr = hccl_buffer_base(context) + HEAP_BASE + weight_off
    view = npu_from_blob(addr, (2 * epn, w_elems), "bf16")
    context.weight_views[key] = view
    return view


def sync_transport(context, real_weights, local_weights, epn):
    for real_w, view_w in zip(real_weights, local_weights):
        if real_w.data_ptr() == view_w[:epn].data_ptr():
            continue
        view_w[:epn] = real_w


def _publish_offsets(context, local_weights):
    buf_base = hccl_buffer_base(context) + HEAP_BASE
    return [w.data_ptr() - buf_base for w in local_weights]


def prefetch_weight(
    context, experts_to_copy, real_weights, local_weights, epn, *, async_finish=False
):
    rw0, rw1, rw2 = real_weights
    lw0, lw1, lw2 = local_weights
    w_elems = lw0.shape[1]
    sync_transport_stream(context)
    sync_transport(context, (rw0, rw1, rw2), (lw0, lw1, lw2), epn)
    publish_offs = _publish_offsets(context, (lw0, lw1, lw2))
    _get_push_program(epn, w_elems, publish_offs, context.rank_size, context.rank)(
        rw0,
        rw1,
        rw2,
        experts_to_copy,
        context.ctx_tensor,
    )
    return finish_op(context, async_finish)
