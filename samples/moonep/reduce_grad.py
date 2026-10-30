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

# MEMSTAT 显存对账打点（MEMSTAT=1 开启，默认关——上游 opt-in）

from cannbotdsl.ops.sync import PIPE, vec_sync_notify, vec_sync_wait
from cannbotdsl.tensor import MemLoc, tile_slice
from cannbotdsl.types.datastruct import StructRef
from moonep_context import (
    HEAP_BASE,
    LAYOUT_TAIL_BYTES,
    check_heap_capacity,
    finish_op,
    grad_slot_off,
    normalize_group_list,
    pick_weight_chunk,
    sym_barrier,
    sync_transport_stream,
)

VL = 64

_R = 1  # 组内 rank_size（均须进 program 键）
_GRAD_BF16 = False  # grad 视图 dtype（编译锁内写入，进 program 键）
_BLOCKS = 28  # 实际启动块数（编译锁内写入，进 program 键）
_REDUCE_PROGRAMS = {}
_REDUCE_LOCK = threading.Lock()


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


# ===========================================================================
# pull 协议 reduce_grad：owner 远端读对端副本行 + kernel 内原子直累
#
# 数据流：FFN 反向预写 grad_view[i] home 种子（本地零通信）→
#   block p 对每个（peer p 条目）：批量拉整行 → scratch 行 p → flush →
#   逐 chunk 原子加把远端贡献并进 grad_view[i]。
#
# 约束与依据（2026-10-27 探针链定论）：
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


def create_grad_slot_view(
    context, epn, w_elems, *, group=0, groups=1, slots=None, grad_dtype=torch.bfloat16
):
    """在对称堆的梯度槽区创建 [epn, W] 梯度槽视图（pull 协议的数据落点）。

    与 create_weight_view 对称。FFN 反向把副本专家的 dW 直写到此槽
    （本地写，零通信）；reduce_grad 的 owner 远端读此槽累加。
    布局：组 g 的槽基址 = grad_slot_off() + g * epn * W * elem。
    """
    if context.last_plan_shape is None:
        raise RuntimeError("create_grad_slot_view 需先调用 moonep_plan")
    if group < 0 or group >= groups:
        raise ValueError(f"group={group} 超出组数 {groups}")
    if slots is None:
        slots = context.window_slots or 1
    used = context.pull_slots_used
    if used is None:
        context.pull_slots_used = slots
    elif used != slots:
        raise ValueError(
            f"create_grad_slot_view slots={slots} 与既有槽视图 slots={used}"
            " 不一致——槽区基址会错位（请保持全链 slots 一致）"
        )
    elem_bytes = 2 if grad_dtype == torch.bfloat16 else 4
    slot_bytes = groups * epn * w_elems * elem_bytes
    need = (
        HEAP_BASE
        + grad_slot_off(epn, w_elems, num_groups=groups, slots=slots)
        + slot_bytes
        + LAYOUT_TAIL_BYTES
    )
    check_heap_capacity(context, need, "pull 梯度槽区")
    from moonep_context import hccl_buffer_base
    from npu_blob import npu_from_blob

    view_off = grad_slot_off(epn, w_elems, num_groups=groups, slots=slots)
    group_off = view_off + group * epn * w_elems * elem_bytes
    addr = hccl_buffer_base(context) + HEAP_BASE + group_off
    return npu_from_blob(
        addr, (epn, w_elems), "bf16" if grad_dtype == torch.bfloat16 else "fp32"
    )


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
_PULL_ACC_LIMIT = int(os.environ.get("MOONEP_PULL_CHUNK", "16384"))


@kernel
class _MoonepReducePullAccKernel:
    """pull 协议 kernel 直累：block-per-peer 拉取 + 同 kernel 消费 + 原子加。

    数据流：
      FFN 反向预写 grad_g[i] 的 home 种子（本地零通信）→
      block p 向量扫 etc[p, :]（SIMD 64-lane 比较生成 UB 清单，消 host
      D2H）→ 命中 home 区间的 lane 逐条：批量拉整行 → scratch 行 p →
      flush → 逐 chunk 原子加把远端贡献并进 grad_g[i]。

    约束与依据：
      - 队列单提交者：block p 的命令/flush 只进 channelHandles[p]
      - scratch [R, W]：block p 落地行 p（行独占，无互踩）
      - 同 kernel 消费：批 get → flush → 消费（排序已实证可行）
      - C2 写并发：bf16/fp32 统一 mem_copy(atomic_add=True)——硬件
        串行化，任意 rcnt 正确（无写独占约束、无 max_rcnt 守卫需求）。
    累加机器 = push kernel _load_first_sender/_writeback_row 同款。
    """

    def __init__(self, rank_size, epn, w_elems, slot_off, chunk, dual=0, seg_k=0):
        self._r = int(rank_size)
        self._epn = int(epn)
        self._w = int(w_elems)
        self._slot_off = int(slot_off)
        self._chunk = int(chunk)
        self._dual = int(dual)
        self._seg_k = int(seg_k)
        # etc 行按 VL 向上取整，末块 vload 不越界（余 lane 由 exec mask 屏蔽）
        self._epn_pad = ((self._epn + _PULL_VL - 1) // _PULL_VL) * _PULL_VL

    def __call__(self, gm_etc, gm_scratch, gm_grad, ctx: StructRef[CommContext]):
        self._pull_seg(gm_etc, gm_scratch, gm_grad, ctx)

    @jit
    def _pull_seg(self, gm_etc, gm_scratch, gm_grad, ctx):
        epn = self._epn
        w_elems = self._w
        R = self._r
        chunk = self._chunk
        n_chunks = w_elems // chunk
        grad_dtype = dtypes.bfloat16 if _GRAD_BF16 else dtypes.float32

        # 副本行段 SymBuffer（对端源）：[epn, W]
        grad_slots = SymBuffer(
            ctx,
            shape=(epn, w_elems),
            dtype=grad_dtype,
            rank_size=R,
            base_offset=self._slot_off,
        )

        ub_xfer = Buffer(MemLoc.UB, (512,), dtypes.uint8)
        xfer = create_transfer_context(ctx, ub_xfer, role=TransferRole.VEC)

        # 原子源暂存（UB，grad_dtype——与目标同 dtype）
        self._tmp = Buffer(MemLoc.UB, (1, chunk), grad_dtype)

        # 本 rank 序号运行时取自 CommContext（编译键不含 rank——多 rank
        # 共享同一份编译产物）
        me = dtypes.int32(ctx.rankId)

        # block-per-peer：block p 独占队列[p]
        blk = get_block_idx()
        if blk < dtypes.int32(R) and blk != me:
            p = blk
            # Phase 0：向量扫 etc[p, :] → UB 清单（消 host D2H）。
            # lane j 命中本 rank home 区间时 ub_row[j] = home 行号，否则 -1
            ub_etc = Buffer(MemLoc.UB, (1, self._epn_pad), dtypes.int32)
            ub_row = Buffer(MemLoc.UB, (1, self._epn_pad), dtypes.int32)
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
                                tile_slice(gm_scratch, (1, chunk), (p, c)),
                                tile_slice(grad_slots, (1, chunk), (j, c)).peer(p),
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
                                        (1, chunk),
                                        (p, cur_off + c - seg_lo),
                                    ),
                                    tile_slice(grad_slots, (1, chunk), (j, c)).peer(p),
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
                                tile_slice(gm_scratch, (1, chunk), (p, c)),
                                tile_slice(grad_slots, (1, chunk), (j, c)).peer(p),
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
        mem_copy(self._tmp, tile_slice(gm_scratch, (1, self._chunk), (p, c)))

    @jit
    def _consume_tile(self, gm_scratch, gm_grad, p, buf_c, i, grad_c):
        """从 scratch 行 p 的 buf_c 位置读 chunk，原子加进 gm_grad 行 i 的 grad_c 位置。"""
        mem_copy(
            self._tmp,
            tile_slice(gm_scratch, (1, self._chunk), (p, buf_c)),
        )
        mem_copy(
            tile_slice(gm_grad, (1, self._chunk), (i, grad_c)),
            self._tmp,
            atomic_add=True,
        )

    @jit
    def _add_into_row(self, gm_grad, i, c):
        """UB 暂存原子加进 home 行：硬件串行化，任意 sender 数正确。"""
        mem_copy(
            tile_slice(gm_grad, (1, self._chunk), (i, c)),
            self._tmp,
            atomic_add=True,
        )


@host
def _reduce_pull_acc_host(gm_etc, gm_scratch, gm_grad, ctx: StructRef[CommContext]):
    """直累 host entry：gm_scratch=[R, W]（block p 落行 p）；gm_grad=[epn, W] 原子。"""
    _MoonepReducePullAccKernel(
        _R,
        _PULL_EPN,
        _PULL_W,
        _GRAD_SLOT_OFF,
        _PULL_CHUNK,
        _PULL_DUAL,
        _PULL_SEG_K,
    )[_BLOCKS](gm_etc, gm_scratch, gm_grad, ctx)


_GRAD_SLOT_OFF = 0  # 副本行段源基址（编译锁内写入，进 program 键）
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
        global _PULL_DUAL, _PULL_SEG_K
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
        prog = cannbotdsl.compile(
            _reduce_pull_acc_host,
            TensorSpec((spec.rank_size, spec.epn), i32),
            TensorSpec((spec.rank_size, sc_cols), dt),
            TensorSpec((spec.epn, spec.w_elems), dt),
            StructRef[CommContext],
        )
        _PULL_PROGRAMS[spec] = prog
        return prog


def reduce_grad(context, experts_to_copy, grad_view, epn, *, barrier=True):
    """pull 协议 reduce_grad：owner 远端读对端梯度槽归并进 home 梯度。

    副本 dW 由 FFN 反向直写 create_grad_slot_view 创建的对称堆梯度槽
    （独立于权重发布区——prefetch 与 reduce 内存解耦，支持异步流水）。
    本函数 owner 远端读对端梯度槽原子累加进 grad_view。

    接口与原 push 签名一致（无新增参数）：梯度槽视图经 context 共享
    （调用方 create_grad_slot_view 创建后存入 context.pull_slot_views，
    FFN 反向与 reduce 均从 context 读取）。

    Args:
        grad_view: [epn, W] 本地梯度（home 段——FFN 反向预写种子，
                   reduce 后为全量）。
    """
    if isinstance(grad_view, torch.Tensor):
        grad_view = [grad_view]
    # 归一化 + 行宽/dtype/调用序守卫
    grad_view, W, grad_bf16 = _normalize_grad_views(context, grad_view)

    grad_dtype = grad_view[0].dtype
    # 跨流护栏：本算子与前序通信算子共用 epoch 握手机制，须按发起序
    # 执行（同流为快路径零开销）
    sync_transport_stream(context)

    if context.rank_size == 1:
        return None  # 单 rank 无副本可归并（etc 恒全 -1）

    # block-per-peer：每 block 独占一个 peer 的通道（无 grid 栅栏，块数=R 即可）
    blocks = context.rank_size
    elem_bytes = 2 if grad_bf16 else 4
    R = context.rank_size
    # UB 预算：原子源暂存（grad_dtype）
    # bf16 默认 65536（4× 少 WQE → 有效带宽 7→11.7 GB/s，瓶颈 rank −25%）；
    # fp32 必须 16384（65536×4B = 256KB 超 248KB UB 上限）
    chunk = pick_weight_chunk(W, 65536 if grad_bf16 else 16384)

    # 源 = 对称堆独立梯度槽区（v3 终态）
    slot_off = grad_slot_off(
        epn, W, num_groups=len(grad_view), slots=context.window_slots or 1
    )

    # 行宽须为搬运通道的整数倍（尾块越界防护）
    if W % VL:
        raise ValueError(f"W={W} 非 {VL} 的倍数——尾块越界 UB 缓冲")

    # 跨 rank 屏障：确保所有 rank 的 FFN 反向写入副本行完成并可见
    # （与上方跨流护栏职责不同：那是本进程流序，这是全组到达语义）
    sym_barrier(context)

    # scratch 常驻（键控缓存——零池 churn；block p 落地行 p）
    # 双缓冲缩减：scratch [R, 2*k*chunk]（k=5, chunk=65536 → 32P 仅 40 MiB）
    # k=5 足以容纳一个流水段（5×128KB=640KB per buf，远大于 DMA 单笔开销）
    # 性能无损：DMA 是瓶颈，scratch 大小不影响 DMA 速度
    dual = int(os.environ.get("MOONEP_PULL_DUAL", "1"))
    seg_k = int(os.environ.get("MOONEP_PULL_K", "5"))
    if dual and seg_k > 0 and seg_k * chunk < W:
        scratch_cols = 2 * seg_k * chunk
    else:
        scratch_cols = W
    skey = (R, scratch_cols, grad_dtype)
    scache = context.pull_scratch
    if scache is None or scache[0] != skey:
        gm_scratch = torch.empty(
            R, scratch_cols, dtype=grad_dtype, device=grad_view[0].device
        )
        context.pull_scratch = (skey, gm_scratch)
    else:
        gm_scratch = scache[1]

    # 每组：kernel 拉取 + 同 kernel 原子直累（grad_g 归并后已是全量）
    for gi, grad_g in enumerate(grad_view):
        spec = _PullSpec(
            epn=epn,
            w_elems=W,
            chunk=chunk,
            slot_off=slot_off + gi * epn * W * elem_bytes,
            rank_size=R,
            grad_bf16=grad_bf16,
            blocks=blocks,
            dual=dual,
            seg_k=seg_k,
            scratch_cols=scratch_cols,
        )
        _get_pull_program(spec)(experts_to_copy, gm_scratch, grad_g, context.ctx_tensor)

    return finish_op(context, barrier)
