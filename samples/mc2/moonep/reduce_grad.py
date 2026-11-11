# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""reduce_grad：副本专家权重梯度跨 rank 归并回 home rank（训练反向）。

pull 协议（kernel 直累）：
  FFN 反向预写 grad_view[i] home 种子（本地零通信）→
  block p 向量扫 etc[p, :]（SIMD 清单，host 零 D2H/H2D）→
  命中 lane：批量 xfer.get 拉整行 → scratch 行 p →
  flush → 逐 chunk 原子加把远端贡献并进 grad_view[i]。
  通道拓扑 block-per-peer（每队列单提交者）；bf16/fp32 统一
  mem_copy(atomic_add=True) 原子累加（硬件串行化，任意副本 sender
  数正确——无写独占约束）。

本模块要求：E%R==0；grad_view 为 [epn, W]（home 段）。副本 dW 由
FFN 反向直写权重视图副本行 [epn, 2*epn)（发布区——远程可寻址）。
"""

import os
import threading
from typing import NamedTuple

import cannbotdsl
import torch
from cannbotdsl import TensorSpec, dtypes, get_block_idx
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import range_constexpr
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
from cannbotdsl.ops.memcpy import copy_options, mem_copy
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

# MEMSTAT 显存对账打点（MEMSTAT=1 开启，默认关——上游 opt-in）

from cannbotdsl.ops.sync import PIPE, vec_sync_notify, vec_sync_wait
from cannbotdsl.tensor import MemLoc, make_layout, tile_slice
from cannbotdsl.types.datastruct import StructRef
from moonep_context import (
    _BAR_EVENT_ID,
    BAR_FLAGS_OFF,
    HEAP_BASE,
    LAYOUT_TAIL_BYTES,
    MAX_R,
    busy_wait_flag,
    check_heap_capacity,
    finish_op,
    grad_slot_off,
    hccl_buffer_base,
    next_barrier_epoch,
    normalize_group_list,
    pick_weight_chunk,
    sync_transport_stream,
)

VL = 64

_R = 1  # 组内 rank_size（均须进 program 键）
_GRAD_BF16 = False  # grad 视图 dtype（编译锁内写入，进 program 键）
_BLOCKS = 28  # 实际启动块数（编译锁内写入，进 program 键）
_REDUCE_LOCK = threading.Lock()


def _normalize_grad_views(context, local_grads, epn):
    """归一化 full-dW 组列表并推导 (groups, widths, dtype)。

    契约：每组 ``[2*epn, Wg]`` 或任意尾维 ``[2*epn, ...]``（连续——内部
    展平为 [2*epn, Wg]，GMM 对全量权重视图反传的原生形态 [2*epn, F, D]
    直入免调用方 reshape；行序与 prefetch 视图一致：前 epn 行 home、后
    epn 行副本）。组数 2 或 3、各组行宽可不同（交织型 W13 宽 2W、down
    宽 W）；组数/行宽从张量解析。返回 (flat_views, widths, grad_bf16)——
    flat_views 为各组 [2*epn, Wg] 视图（与原张量同存储，归并原位生效）。

    dtype 守卫：dtype 须与**构造定型值** context.grad_dtype 一致
    （MoonepBuffer 构造注入，缺省 bf16）——梯度槽区字节宽在堆布局里按定型
    值预留（fp32 = bf16 的 2×，建组时刻堆已定死），漂移只能首调报错暴露，
    不能静默扩容。**裸模块 API（无 MoonepBuffer）不冻结**：ctx.grad_dtype
    为 None 时维持历史契约——dtype 每调用从张量解析（同 ctx 双 dtype 混用
    合法，#308 精度管线实证形态；梯度槽视图缓存键含 dtype、逐 dtype 独立
    容量校验）。
    """
    local_grads, widths = normalize_group_list(context, local_grads, "reduce_grad")
    grad_dtype = local_grads[0].dtype
    if grad_dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            f"local_grads 须为 fp32 或 bf16（bf16 时 staging/传输减半、UB 内 fp32 "
            f"累加保精度、写回 bf16），got {grad_dtype}"
        )
    frozen = getattr(context, "grad_dtype", None)
    if frozen is not None and frozen != grad_dtype:
        raise TypeError(
            f"local_grads dtype 与构造定型值不符（定型 {frozen}，本次 {grad_dtype}）"
            "——梯度槽区按定型 dtype 预留（fp32 为 bf16 的 2×），"
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
    return flats, widths, grad_dtype == torch.bfloat16


# ===========================================================================
# pull 协议 reduce_grad：owner 远端读对端副本行 + kernel 内原子直累
#
# 数据流：FFN 反向预写 grad_view[i] home 种子（本地零通信）→
#   头部屏障（per-peer flag 握手）→
#   block p 对每个（peer p 条目）：批量拉整行 → scratch 行 p → flush →
#   逐 chunk 原子加把远端贡献并进 grad_view[i]。
#
# 约束与依据：
#   - 队列单提交者：block p 的命令/flush 只进 channelHandles[p]
#   - scratch [R, W]：block p 落地行 p（行独占，无互踩）
#   - 同 kernel 消费：批 get → flush → 消费（排序已实证可行）
#   - C2 写并发：bf16/fp32 统一 mem_copy(atomic_add=True)（硬件串行化）
# ===========================================================================


class _PullSpec(NamedTuple):
    """pull kernel 编译键：形状/源偏移/分块/dtype/rank 拓扑（不含本 rank
    序号——运行时取自 CommContext，多 rank 共享编译产物）。"""

    epn: int
    w_elems: int
    chunk: int
    slot_off: int
    rank_size: int
    grad_bf16: bool
    blocks: int
    dual: int = 0
    seg_k: int = 0
    scratch_cols: int = 0  # 0 = W（非双缓冲）；>0 = 2*k*chunk（双缓冲缩减）


def _grad_region_params(context):
    """(scale_widths, weight_bytes)——梯度槽区基址的两个布局参数。

    槽区紧跟权重发布区（含 scale 组）：scale_widths 取首调定型的
    ``context.layout_scale_widths``（prefetch 冻结；无 scale = None）；
    weight_bytes 取定型的 ``context.weight_dtype``（bf16 2B / fp8
    （e4m3fn/e5m2）1B；未定型（None）取 2B 保守口径——实际区域更小，
    偏移过大只留安全间隙不重叠）。
    """
    scale_widths = getattr(context, "layout_scale_widths", None) or None
    weight_dtype = getattr(context, "weight_dtype", None)
    if weight_dtype is None or weight_dtype == torch.bfloat16:
        weight_bytes = 2
    else:
        weight_bytes = 1  # fp8（e4m3fn/e5m2）——dtype 终态
    return scale_widths, weight_bytes


def _create_grad_slot_views(
    context, epn, widths, *, slots=None, grad_dtype=torch.bfloat16
):
    """按 widths 创建（或复用）全组梯度槽视图，返回同序元组（内部接口）。

    副本 dW 的对称堆落点（独立于权重发布区——prefetch 与 reduce 内存解耦）。
    组 g 的槽视图 ``[epn, widths[g]]``，槽内按 widths 前缀和紧密排布；
    dtype 即槽区字节宽（bf16 2B / fp32 4B）。基址 = 权重发布区之后（含
    scale 组——见 _grad_region_params，与 prefetch 窗口步长同式）。仅供
    reduce_grad 内部调用（副本行入槽是 reduce 数据流的内部步骤，不经调用方）。
    """
    if context.last_plan_shape is None:
        raise RuntimeError("_create_grad_slot_views 需先调用 moonep_plan")
    if slots is None:
        slots = context.window_slots or 1
    used = context.pull_slots_used
    if used is None:
        context.pull_slots_used = slots
    elif used != slots:
        raise ValueError(
            f"梯度槽视图 slots={slots} 与既有 slots={used}"
            " 不一致——槽区基址会错位（请保持全链 slots 一致）"
        )
    elem_bytes = 2 if grad_dtype == torch.bfloat16 else 4
    widths = tuple(int(w) for w in widths)
    # 键须含布局参数（scale_widths/weight_bytes）：混序调用
    # reduce→prefetch(scale)→reduce 时梯度槽基址随冻结值后移——键不含
    # 则旧偏移缓存被复用（copy_ 写旧偏移、pull kernel 读新偏移，数据错位）
    scale_widths, weight_bytes = _grad_region_params(context)
    key = (
        "grad_slots",
        int(epn),
        widths,
        int(slots),
        str(grad_dtype),
        tuple(scale_widths) if scale_widths else (),
        int(weight_bytes),
    )
    cached = getattr(context, "pull_slot_views", None)
    if cached is not None and getattr(context, "_grad_slot_key", None) == key:
        return cached
    scale_widths, weight_bytes = _grad_region_params(context)
    slot_region = sum(widths) * epn * elem_bytes
    need = (
        HEAP_BASE
        + grad_slot_off(
            epn,
            widths=widths,
            scale_widths=scale_widths,
            weight_bytes=weight_bytes,
            slots=slots,
        )
        + slot_region
        + LAYOUT_TAIL_BYTES
    )
    check_heap_capacity(context, need, "pull 梯度槽区")
    from npu_blob import npu_from_blob

    view_off = grad_slot_off(
        epn,
        widths=widths,
        scale_widths=scale_widths,
        weight_bytes=weight_bytes,
        slots=slots,
    )
    addr = hccl_buffer_base(context) + HEAP_BASE + view_off
    views = []
    for w in widths:
        views.append(
            npu_from_blob(
                addr, (epn, w), "bf16" if grad_dtype == torch.bfloat16 else "fp32"
            )
        )
        addr += epn * w * elem_bytes
    views = tuple(views)
    context.pull_slot_views = views
    context._grad_slot_key = key
    return views


_PULL_PROGRAMS = {}
# ★ 编译烘焙走模块级全局（_R/_GRAD_BF16/_BLOCKS）——共用 _REDUCE_LOCK
# 保证并发编译不互相污染烘焙常量
_PULL_LOCK = _REDUCE_LOCK
_PULL_EPN = 1
_PULL_W = 1
_PULL_CHUNK = 1
_PULL_FENCE_ID = 6  # 清单 V→S 栅栏事件（SIMD 扫描后标量读；kernel 内独立无冲突）
_PULL_VL = 64  # int32 向量寄存器 lane 数（256B / 4B）

# 分块上限：对齐 push 的 _REDUCE_CHUNK_LIMIT——UB 预算
# （acc fp32 64KB + tmp 32KB ≈ 96KB < 248KB；原子路径无 cur）


@kernel
class _MoonepReducePullAccKernel:
    """pull 协议 kernel 直累：头部屏障 + block-per-peer 拉取 + 同 kernel 消费 + 原子加。

    数据流：
      FFN 反向预写 grad_g[i] 的 home 种子（本地零通信）→
      头部屏障（per-peer flag 握手——我方 kernel 启动即入槽
      host copy_（流序在前）已完成；等对端 flag 即等其入槽可见）→
      block p 向量扫 etc[p, :]（SIMD 64-lane 比较生成 UB 清单，消 host
      D2H）→ 命中 home 区间的 lane 逐条：批量拉整行 → scratch 行 p →
      flush → 逐 chunk 原子加把远端贡献并进 grad_g[i]。

    约束与依据：
      - 队列单提交者：block p 的命令/flush 只进 channelHandles[p]
      - scratch 扁平 [R*cols]（行 p 起于 p*cols）：block p 落地行 p（行独占，
        无互踩）
      - 同 kernel 消费：批 get → flush → 消费（排序已实证可行）
      - C2 写并发：bf16/fp32 统一 mem_copy(atomic_add=True)——硬件
        串行化，任意 rcnt 正确（无写独占约束、无 max_rcnt 守卫需求）。
      - xfer.get 的 dst/src 均须过 0.8 distributed 传输校验：视图链只认
        tile_slice（_SUPPORTED_VIEW_OPS），连续性要求"至多一个动态
        extent（最内维）"，且 extract_pointer 要求视图根有 concrete
        storage alias——strided_slice 视图（无论实参根还是 peer 根）在
        lowering 期均无 storage alias，不可用于传输。两侧统一 1-D 扁平 +
        1-D tile：dst 为 scratch 实参 [R*cols]（行 p 起于 p*cols），src 为
        grad_slots 1-D SymBuffer。**tile_slice 的 coord 是 tile 序号而非
        元素偏移**（0.8 tensor.py docstring：origin = coord × tiler ×
        stride；distributed walker 的 _view_origin_elements 同款实现）：
        行 j 列 c 的 tile 序号 = j*(W//chunk) + c，scratch 行 p 起于
        p*(cols//chunk)——单动态 extent + tile 链 + 实参/make_pointer 根，
        全部合规，寻址与 0.7 的 2-D (1, chunk)@(j, c) 逐字节等价。
        另 mem_copy 要求两侧 view rank 一致——pull 链路（scratch 暂存、
        gm_grad 原子目标）统一 1-D。
    累加机器 = push kernel _load_first_sender/_writeback_row 同款。
    """

    def __init__(self, rank_size, epn, w_elems, slot_off, chunk, cols, dual=0, seg_k=0):
        self._r = int(rank_size)
        self._epn = int(epn)
        self._w = int(w_elems)
        self._slot_off = int(slot_off)
        self._chunk = int(chunk)
        self._cols = int(cols)
        self._dual = int(dual)
        self._seg_k = int(seg_k)
        # 1-D tile 的 coord 是 tile 序号（origin = coord × tiler × stride），
        # 预换算行起始 tile 序号：scratch 行 p 起于 p*(cols//chunk)、行 j 的
        # 副本槽/梯度行起于 j*(W//chunk)。W 与 cols 均为 chunk 整数倍
        # （pick_weight_chunk 保证），除法无余数。
        self._w_t = self._w // self._chunk
        self._cols_t = self._cols // self._chunk
        # etc 行按 VL 向上取整，末块 vload 不越界（余 lane 由 exec mask 屏蔽）
        self._epn_pad = ((self._epn + _PULL_VL - 1) // _PULL_VL) * _PULL_VL

    def __call__(
        self, gm_etc, gm_scratch, gm_grad, gm_epoch, ctx: StructRef[CommContext]
    ):
        self._pull_seg(gm_etc, gm_scratch, gm_grad, gm_epoch, ctx)

    @jit
    def _pull_seg(self, gm_etc, gm_scratch, gm_grad, gm_epoch, ctx):
        epn = self._epn
        w_elems = self._w
        R = self._r
        chunk = self._chunk
        n_chunks = w_elems // chunk
        grad_dtype = dtypes.bfloat16 if _GRAD_BF16 else dtypes.float32

        # 副本行段 SymBuffer（对端源）：1-D 扁平 [epn*W]——distributed
        # 传输的视图链 walker 只认 tile_slice，2-D tile + 动态行列坐标
        # 会产出双动态 extent（tail-aware 钳制）被连续性校验拒绝；
        # 1-D tile 单动态 extent 合规。coord 是 tile 序号（元素偏移须除以
        # chunk）：行 j 列 c → j*(W//chunk) + c，与 0.7 的
        # 2-D (1, chunk)@(j, c) 寻址逐字节等价。
        grad_slots = SymBuffer(
            ctx,
            shape=(epn * w_elems,),
            dtype=grad_dtype,
            rank_size=R,
            base_offset=self._slot_off,
        )

        ub_xfer = Buffer(MemLoc.UB, dtypes.uint8, make_layout((512,)))
        xfer = create_transfer_context(ctx, ub_xfer, role=TransferRole.VEC)

        # 原子源暂存（UB，grad_dtype——与目标同 dtype）
        # 1-D 暂存：与 scratch/gm_grad 的 1-D tile 传输、rank 一致
        self._tmp = Buffer(MemLoc.UB, grad_dtype, make_layout((chunk,)))

        # 本 rank 序号运行时取自 CommContext（编译键不含 rank——多 rank
        # 共享同一份编译产物）
        me = dtypes.int32(ctx.rankId)

        # 头部屏障（原独立 sym_barrier kernel 融入本
        # kernel，省一次 kernel 发射）：block p 先向 p 推我方到达 flag、
        # 再自旋等 p 的到达。我方 kernel 启动即我方 ① 入槽写入（流序在
        # 前的 host copy_）已完成；对端 flag 到达即其入槽写入已可见。
        # 同一 reduce 轮的逐组多次发射共用同一 epoch——首组真等、后续组
        # flag 已 ≥ epoch 即时通过（幂等 put 无害）。epoch 与 push 内联
        # 屏障/独立 sym_barrier 共用同一单调计数器，交错调用不串轮。
        flags = SymBuffer(
            ctx,
            shape=(MAX_R,),
            dtype=dtypes.int32,
            rank_size=R,
            base_offset=BAR_FLAGS_OFF,
        )
        ub_flag = Buffer(MemLoc.UB, dtypes.int32, make_layout((1,)))
        ub_epoch = Buffer(MemLoc.UB, dtypes.int32, make_layout((1,)))

        # block-per-peer：block p 独占队列[p]
        blk = get_block_idx()
        if blk < dtypes.int32(R) and blk != me:
            p = blk
            me64 = dtypes.int64(me)
            xfer.put(p, tile_slice(flags, (1,), (me64,)).peer(p), gm_epoch)
            xfer.flush(p)
            busy_wait_flag(
                tile_slice(flags, (1,), (dtypes.int64(p),)),
                gm_epoch,
                _BAR_EVENT_ID,
                ub_flag,
                ub_epoch,
            )
            # Phase 0：向量扫 etc[p, :] → UB 清单（消 host D2H）。
            # lane j 命中本 rank home 区间时 ub_row[j] = home 行号，否则 -1
            ub_etc = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, self._epn_pad)))
            ub_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, self._epn_pad)))
            self._scan_peer(gm_etc, p, me, ub_etc, ub_row)
            # 标量按 lane 顺序消费：命中 lane 的 home 行 i = ub_row[j]、
            # 对端副本行段行号 = j
            for j in cannbotdsl.range(dtypes.int64(epn)):
                i = dtypes.int64(ub_row[dtypes.int64(0), dtypes.int64(j)])
                if i >= dtypes.int64(0):
                    if self._dual and self._seg_k > 0 and self._seg_k < n_chunks:
                        # ── 双缓冲流水路径（scratch 缩减形态）──
                        # scratch [R, 2*k*chunk]：buf_A = chunk 0..k-1，
                        # buf_B = chunk k..2k-1（无间隙紧凑布局）
                        k = self._seg_k
                        n_seg = (n_chunks + k - 1) // k
                        half = k  # buf_B 起始 chunk（紧凑，非 n_chunks/2）

                        # 灌入段 0 → buf_A
                        for c in cannbotdsl.range(0, k):
                            xfer.get(
                                p,
                                tile_slice(
                                    gm_scratch,
                                    (chunk,),
                                    (p * self._cols_t + c,),
                                ),
                                tile_slice(
                                    grad_slots, (chunk,), (j * self._w_t + c,)
                                ).peer(p),
                            )
                        xfer.flush(p)

                        # 流水主循环：段 1..n_seg-1
                        for seg in cannbotdsl.range(1, n_seg):
                            seg_lo = seg * k
                            seg_hi = seg_lo + k
                            if seg_hi > n_chunks:
                                seg_hi = n_chunks
                            # 段奇 → buf_B（half 起），段偶 → buf_A（0 起）
                            # seg - (seg//2)*2 = seg%2
                            odd = seg - (seg // 2) * 2
                            cur_off = odd * half
                            prev_off = half - odd * half

                            # 提交当前段到对侧（非阻塞，DMA 后台跑）
                            for c in cannbotdsl.range(seg_lo, seg_hi):
                                xfer.get(
                                    p,
                                    tile_slice(
                                        gm_scratch,
                                        (chunk,),
                                        (p * self._cols_t + (cur_off + c - seg_lo),),
                                    ),
                                    tile_slice(
                                        grad_slots, (chunk,), (j * self._w_t + c,)
                                    ).peer(p),
                                )

                            # 消费前段（计算引擎——与 DMA 并行）
                            prev_lo = (seg - 1) * k
                            prev_hi = prev_lo + k
                            if prev_hi > n_chunks:
                                prev_hi = n_chunks
                            for c in cannbotdsl.range(prev_lo, prev_hi):
                                self._consume_tile(
                                    gm_scratch, gm_grad, p, prev_off + c - prev_lo, i, c
                                )

                            # 等当前段到达（均衡时瞬时返回）
                            xfer.flush(p)

                        # 消费最后段
                        last_lo = (n_seg - 1) * k
                        last_odd = (n_seg - 1) - ((n_seg - 1) // 2) * 2
                        last_off = last_odd * half
                        for c in cannbotdsl.range(last_lo, n_chunks):
                            self._consume_tile(
                                gm_scratch, gm_grad, p, last_off + c - last_lo, i, c
                            )
                    else:
                        # ① 批量拉取整行 → scratch 行 p（原路径）
                        for c in cannbotdsl.range(0, n_chunks):
                            xfer.get(
                                p,
                                tile_slice(
                                    gm_scratch,
                                    (chunk,),
                                    (p * self._cols_t + c,),
                                ),
                                tile_slice(
                                    grad_slots, (chunk,), (j * self._w_t + c,)
                                ).peer(p),
                            )
                        xfer.flush(p)
                        # ② 同 kernel 消费 + 原子加直累
                        for c in cannbotdsl.range(0, n_chunks):
                            self._load_scratch_tile(gm_scratch, p, c)
                            self._add_into_row(gm_grad, i, c)

    @jit
    def _scan_peer(self, gm_etc, p, me, ub_etc, ub_row):
        """向量扫描 etc[p, :]：命中本 rank home 区间的 lane 写 home 行号，否则 -1。

        64 lane 并行比较，结果留在 UB；V→S 的 notify/wait 保证随后的标量读
        能看到 vstore 的结果。余 lane（epn 不足一个 VL）由 exec mask 屏蔽。
        """
        epn, pad, vl = self._epn, self._epn_pad, _PULL_VL
        lo_e = me * dtypes.int32(epn)
        hi_e = lo_e + dtypes.int32(epn)
        mem_copy(
            tile_slice(ub_etc, (1, epn), (0, 0)),
            tile_slice(gm_etc, (1, epn), (p, 0)),
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
        vec_sync_notify(PIPE.V, PIPE.S, _PULL_FENCE_ID)
        vec_sync_wait(PIPE.V, PIPE.S, _PULL_FENCE_ID)

    @jit
    def _load_scratch_tile(self, gm_scratch, p, c):
        """scratch 行 p 的 chunk 装入 UB 暂存（原子源须在片上）。"""
        mem_copy(
            self._tmp,
            tile_slice(gm_scratch, (self._chunk,), (p * self._cols_t + c,)),
        )

    @jit
    def _consume_tile(self, gm_scratch, gm_grad, p, buf_c, i, grad_c):
        """从 scratch 行 p 的 buf_c 位置读 chunk，原子加进 gm_grad 行 i 的 grad_c 位置。"""
        mem_copy(
            self._tmp,
            tile_slice(
                gm_scratch,
                (self._chunk,),
                (p * self._cols_t + buf_c,),
            ),
        )
        mem_copy(
            tile_slice(gm_grad, (self._chunk,), (i * self._w_t + grad_c,)),
            self._tmp,
            copy_options(atomic_add=True),
        )

    @jit
    def _add_into_row(self, gm_grad, i, c):
        """UB 暂存原子加进 home 行：硬件串行化，任意 sender 数正确。"""
        mem_copy(
            tile_slice(gm_grad, (self._chunk,), (i * self._w_t + c,)),
            self._tmp,
            copy_options(atomic_add=True),
        )


@host
def _reduce_pull_acc_host(
    gm_etc, gm_scratch, gm_grad, gm_epoch, ctx: StructRef[CommContext]
):
    """直累 host entry：gm_scratch=[R*cols]（行 p 起于 p*cols）；gm_grad=[epn*W]（行 i 起于 i*W）原子；
    gm_epoch=本轮屏障 epoch（头部内联屏障）。"""
    _MoonepReducePullAccKernel(
        _R,
        _PULL_EPN,
        _PULL_W,
        _GRAD_SLOT_OFF,
        _PULL_CHUNK,
        _PULL_COLS,
        _PULL_DUAL,
        _PULL_SEG_K,
    )[_BLOCKS](gm_etc, gm_scratch, gm_grad, gm_epoch, ctx)


_GRAD_SLOT_OFF = 0  # 副本行段源基址（编译锁内写入，进 program 键）
_PULL_COLS = 0  # scratch 扁平行步长（编译锁内写入，进 program 键）
_PULL_DUAL = 0  # 双缓冲流水开关（编译锁内写入，进 program 键）
_PULL_SEG_K = 0  # 流水段 chunk 数（编译锁内写入，进 program 键）


def _get_pull_program(spec):
    """_PullSpec 键控的 pull program 缓存（模块级烘焙与编译在锁内原子完成）。"""
    with _PULL_LOCK:
        prog = _PULL_PROGRAMS.get(spec)
        if prog is not None:
            return prog
        global _GRAD_SLOT_OFF, _R, _GRAD_BF16, _BLOCKS
        global _PULL_EPN, _PULL_W, _PULL_CHUNK
        global _PULL_DUAL, _PULL_SEG_K, _PULL_COLS
        _GRAD_SLOT_OFF = spec.slot_off
        _R = spec.rank_size
        _GRAD_BF16 = spec.grad_bf16
        _BLOCKS = spec.blocks
        _PULL_EPN, _PULL_W = spec.epn, spec.w_elems
        _PULL_CHUNK = spec.chunk
        _PULL_DUAL = spec.dual
        _PULL_SEG_K = spec.seg_k
        i32 = dtypes.int32
        dt = dtypes.bfloat16 if spec.grad_bf16 else dtypes.float32
        sc_cols = spec.scratch_cols if spec.scratch_cols > 0 else spec.w_elems
        _PULL_COLS = sc_cols
        prog = cannbotdsl.compile(
            _reduce_pull_acc_host,
            TensorSpec((spec.rank_size, spec.epn), i32),
            # scratch/gm_grad 扁平 1-D：distributed 传输要求 dst 为 tile 链 +
            # 单动态 extent，mem_copy 又要求两侧 rank 一致（详见类 docstring），
            # 2-D 行列双坐标做不到，pull 链路统一 1-D
            TensorSpec((spec.rank_size * sc_cols,), dt),
            TensorSpec((spec.epn * spec.w_elems,), dt),
            TensorSpec((1,), i32),  # 屏障 epoch（头部内联屏障）
            StructRef[CommContext],
        )
        _PULL_PROGRAMS[spec] = prog
        return prog


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
    """pull 协议 reduce_grad：副本梯度归并回 home（full-dW 直入形态）。

    ``local_grads`` 为 2 或 3 组 ``[2*epn, Wg]`` 张量——GMM 对全量
    （home+副本）权重视图反传的原生形态，行序与 prefetch 视图一致（前
    epn 行 home、后 epn 行副本）。组数/各组行宽从张量解析；dtype 须与
    构造定型值一致（交织型传 ``[2*epn, 2W] + [2*epn, W]`` 两组即可）。

    梯度搬运四步（数据流，prefetch_weight 不搬任何梯度——全部在本算子内）：

    ① 入槽：副本行 ``t[epn:]`` 拷入对称堆梯度槽 ``[epn, Wg]``（本地写己方
       堆分片，零通信；槽视图内部创建，不对外暴露）
    ② 屏障（pull kernel 头部 per-peer flag 握手）：全组所有 rank 的
       入槽写入互可见
    ③ 拉取归并：owner 逐组远拉每个持副本 rank 的槽行（xfer.get → scratch
       → UB 暂存）→ mem_copy(atomic_add=True) 原子加进本地 dw home 行
       ``t[:epn]``（方向与 prefetch 相反：prefetch 是 owner push 权重到
       消费方，reduce 是 owner pull 梯度回属主——同一张 experts_to_copy
       清单，两个方向各用一次；搬运者双重身份：对他 rank home 专家的副本
       是 holder（①入槽供他人拉）、对自己的 home 专家是 owner（③拉取归并））
    ④ 收尾：finish_op（async_finish=False 阻塞 / True 返回发起流 Event）

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
    # 归一化 + 行数/宽度/dtype/调用序守卫（组数与行宽从张量解析；dtype）。
    # C15：单张量 isinstance 包装已删（只通向报错的死分支）——裸 Tensor
    # 由 normalize_group_list 真身包装后触发组数守卫的清晰报错。
    local_grads, widths, grad_bf16 = _normalize_grad_views(context, local_grads, epn)

    grad_dtype = local_grads[0].dtype
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
    # —— ① 副本行入槽（内部数据流：full-dW 尾段 → 对称堆梯度槽）——
    slot_views = _create_grad_slot_views(
        context, epn, widths, slots=slots, grad_dtype=grad_dtype
    )
    for t, sv in zip(local_grads, slot_views):
        sv.copy_(t[epn:])

    # block-per-peer：每 block 独占一个 peer 的通道（无 grid 栅栏，块数=R 即可）
    blocks = context.rank_size
    elem_bytes = 2 if grad_bf16 else 4
    R = context.rank_size
    # 源 = 对称堆独立梯度槽区（v3 终态；槽内按 widths 前缀和；基址含
    # scale 组区域——与 prefetch 窗口步长同式）
    scale_widths, weight_bytes = _grad_region_params(context)
    slot_off = grad_slot_off(
        epn,
        widths=widths,
        scale_widths=scale_widths,
        weight_bytes=weight_bytes,
        slots=slots,
    )

    # —— ② 跨 rank 屏障在 pull kernel 头部——对端 kernel 启动即其入槽
    #     写入（流序在前的 host copy_）已完成，头部 flag 握手即等价原屏障
    #     的全组到达语义。同一轮逐组发射共用本 epoch：首组真等、后续组
    #     flag 已 ≥ epoch 即时通过。与上方跨流护栏职责不同：那是本进程
    #     流序，这是全组到达语义。
    gm_epoch = next_barrier_epoch(context)

    # scratch 常驻（键控缓存——零池 churn；block p 落地行 p）
    # 双缓冲缩减：scratch [R, 2*k*chunk]（k=5, chunk=65536 → 32P 仅 40 MiB）
    # k=5 足以容纳一个流水段（5×128KB=640KB per buf，远大于 DMA 单笔开销）
    # 性能无损：DMA 是瓶颈，scratch 大小不影响 DMA 速度
    # 从 ctx 读 Buffer 构造参数（裸 API 时 ctx 未设——env 缺省回退）
    dual = getattr(context, "pull_dual", None)
    if dual is None:
        dual = int(os.environ.get("MOONEP_PULL_DUAL", "1"))
    seg_k = getattr(context, "pull_seg_k", None)
    if seg_k is None:
        seg_k = int(os.environ.get("MOONEP_PULL_K", "5"))
    max_w = max(widths)
    chunk = pick_weight_chunk(max_w, 65536 if grad_bf16 else 16384)
    if dual and seg_k > 0 and seg_k * chunk < max_w:
        scratch_cols = 2 * seg_k * chunk
    else:
        scratch_cols = max_w
    skey = (R, scratch_cols, grad_dtype)
    scache = context.pull_scratch
    if scache is None or scache[0] != skey:
        gm_scratch = torch.empty(
            R * scratch_cols, dtype=grad_dtype, device=local_grads[0].device
        )
        context.pull_scratch = (skey, gm_scratch)
    else:
        gm_scratch = scache[1]

    # —— ③ 逐组拉取 + 同 kernel 原子直累（home 段归并后已是全量）——
    # 每组独立 spec：w_elems=Wg（变宽）、槽基址按 widths 前缀和推进
    group_off = 0
    for gi, (grad_g, wg) in enumerate(zip(local_grads, widths)):
        if wg % VL:
            raise ValueError(f"第 {gi} 组 W={wg} 非 {VL} 的倍数——尾块越界 UB 缓冲")
        spec = _PullSpec(
            epn=epn,
            w_elems=wg,
            chunk=pick_weight_chunk(wg, 65536 if grad_bf16 else 16384),
            slot_off=slot_off + group_off,
            rank_size=R,
            grad_bf16=grad_bf16,
            blocks=blocks,
            dual=dual,
            seg_k=seg_k,
            scratch_cols=scratch_cols,
        )
        _get_pull_program(spec)(
            experts_to_copy,
            gm_scratch,
            grad_g[:epn].reshape(-1),
            gm_epoch,
            context.ctx_tensor,
        )
        group_off += epn * wg * elem_bytes

    event = finish_op(context, async_finish)
    home_views = tuple(t[:epn] for t in local_grads)
    # H1/C14b 变长返回契约：sync 只返回 home_views 单值；async 二元组。
    if async_finish:
        return home_views, event
    return home_views
