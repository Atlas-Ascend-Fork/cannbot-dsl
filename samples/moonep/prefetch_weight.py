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

副本清单由 kernel 内逐 block 扫 etc 生成（GM 点读写与 xfer 共存，每 block
独占一个目的 rank）：host 侧零回读/零 Python 扫描/零 H2D，编译键与路由结构
解耦（流序写入纪律：清单只经同流 kernel 全量覆写，免疫 host 写竞态）。
"""

import threading

import torch

import cannbotdsl
from cannbotdsl import dtypes
from cannbotdsl import TensorSpec
from cannbotdsl.types.datastruct import StructRef
from cannbotdsl.buffer import Buffer
from cannbotdsl.tensor import MemLoc, tile_slice
from cannbotdsl.lang.constexpr import range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.distributed import (
    CommContext,
    SymBuffer,
    TransferRole,
    create_transfer_context,
)

from moonep_context import (
    HEAP_BASE,
    check_heap_capacity,
    finish_op,
    hccl_buffer_base,
    normalize_group_list,
    prefetch_weight_off,
    reduce_stg_off,
    sync_transport_stream,
)
from npu_blob import npu_from_blob


@kernel
class _MoonepPrefetchPushKernel:
    """home 把有副本需求的真实专家行直推到各消费 rank 的预取槽。

    每 block 独占一个目的 rank，扫描该目的 rank 的 etc 行并筛选本 rank 的
    home 专家；同一 peer 的 put/flush 只由一个 block 发起，避免多 block 并发
    写同一通道。副本清单直接点读 GM，无需 SIMT VF 区，可与 xfer 共存；put
    循环保持扁平，所有 put 完成后只对活跃 peer flush 一次。
    """

    def __init__(self, rank, rank_size, epn, w_elems, weight_offs):
        self._me = int(rank)
        self._r = int(rank_size)
        self._epn = int(epn)
        self._w = int(w_elems)
        self._offs = tuple(int(o) for o in weight_offs)
        self._num_g = len(self._offs)

    def __call__(
        self,
        kr_real0,
        kr_real1,
        kr_real2,
        gm_etc,
        gm_mf,
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

        # 每 block 的清单段：slot[epn] | row[epn] | cnt；目的 rank 即 block id。
        dst_r = dtypes.int64(get_block_idx())
        base = dst_r * dtypes.int64(2 * epn + 1)
        c = dtypes.int32(0)
        for i in cannbotdsl.range(dtypes.int64(epn)):
            e = gm_etc[dst_r, i]
            if e >= dtypes.int32(me * epn):
                if e < dtypes.int32((me + 1) * epn):
                    gm_mf[base + dtypes.int64(c)] = dtypes.int32(epn) + dtypes.int32(i)
                    gm_mf[base + dtypes.int64(epn) + dtypes.int64(c)] = (
                        e - dtypes.int32(me * epn)
                    )
                    c = c + dtypes.int32(1)
        gm_mf[base + dtypes.int64(2 * epn)] = c
        for k in cannbotdsl.range(dtypes.int64(c)):
            slot = dtypes.int64(gm_mf[base + k])
            row = dtypes.int64(gm_mf[base + dtypes.int64(epn) + k])
            for g in range_constexpr(self._num_g):
                xfer.put(
                    dst_r,
                    tile_slice(wsyms[g], (1, w_elems), (slot, 0)).peer(dst_r),
                    tile_slice(reals[g], (1, w_elems), (row, 0)),
                )
        if c > dtypes.int32(0):
            xfer.flush(dst_r)


_WEIGHT_OFFS = ()  # wrapper 在编译锁内写入（trace 期模块级常量：各组发布区偏移）
_R = 1  # 组内 rank_size（wrapper 在编译前写入，同 shape 不同 rank_size 须进键）
_ME = 0  # 组内 rank（清单寻址依赖，进编译键）


@host
def _push_host(
    gm_real0, gm_real1, gm_real2, gm_etc, gm_mf, ctx: StructRef[CommContext]
):
    epn, w_elems = gm_real0.shape
    _MoonepPrefetchPushKernel(_ME, _R, epn, w_elems, _WEIGHT_OFFS)[_R](
        gm_real0, gm_real1, gm_real2, gm_etc, gm_mf, ctx
    )


_PUSH_PROGRAMS = {}
_PUSH_LOCK = threading.Lock()


def _get_push_program(epn, w_elems, weight_offs, rank_size, me):
    """(epn, W, 组偏移元组, rank_size, me) 键控的融合直推 program——不含路由结构
    （清单由 kernel 内扫 etc 表按 block 分段生成，路由变化不重编译）。

    _WEIGHT_OFFS/_R 的烘焙与编译在锁内原子完成（多线程并发首次编译不同
    组偏移时，烘焙窗口互污会产出错误 program）。
    """
    key = (epn, w_elems, tuple(weight_offs), rank_size, me)
    with _PUSH_LOCK:
        prog = _PUSH_PROGRAMS.get(key)
        if prog is None:
            global _WEIGHT_OFFS, _R, _ME
            _WEIGHT_OFFS, _R, _ME = tuple(weight_offs), rank_size, me
            mf_elems = rank_size * (2 * epn + 1)
            prog = cannbotdsl.compile(
                _push_host,
                TensorSpec((epn, w_elems), dtypes.bfloat16),
                TensorSpec((epn, w_elems), dtypes.bfloat16),
                TensorSpec((epn, w_elems), dtypes.bfloat16),
                TensorSpec((rank_size, epn), dtypes.int32),
                TensorSpec((mf_elems,), dtypes.int32),
                StructRef[CommContext],
            )
            _PUSH_PROGRAMS[key] = prog
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


def _manifest_buf(context, epn, world_size):
    """设备端清单：每个目的 rank 一段 slot[epn] | row[epn] | cnt。"""
    key = (world_size, epn)
    cached = context.prefetch_manifest
    if cached is not None and cached[0] == key:
        return cached[1]
    buf = torch.empty(world_size * (2 * epn + 1), dtype=torch.int32, device="npu")
    context.prefetch_manifest = (key, buf)
    return buf


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


def _clamp3(items):
    """按需以末元素补足 3 个（不足 3 组以末组占位，const_expr 消除未用分支）。"""
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


def _check_view_addrs(context, local_weights, epn, w_elems):
    """校验各组视图地址落在多槽窗口区内且按整视图步长对齐（防普通张量误传）。

    返回各组相对对称堆基址的发布区偏移（prefetch 以视图地址推导直推
    偏移——槽位信息由视图自身携带，无需额外参数）。
    """
    buf_base = hccl_buffer_base(context) + HEAP_BASE
    view_bytes = 2 * epn * w_elems * 2  # 组步长 = 完整 [2epn, W] bf16 视图字节数
    slots = context.window_slots or 1
    groups = context.layout_groups or len(local_weights)
    window_bytes = slots * groups * view_bytes
    offs = []
    for g, w_g in enumerate(local_weights):
        rel = w_g.data_ptr() - buf_base - prefetch_weight_off()
        if rel < 0 or rel >= window_bytes or rel % view_bytes != 0:
            raise ValueError(
                "local_weights 必须是 create_weight_view 创建的对称区视图"
                f"（第 {g} 组相对偏移 {rel} 不在窗口区 [0, {window_bytes}) 内"
                " 或未按整视图步长对齐）"
            )
        offs.append(rel + prefetch_weight_off())
    return offs


def prefetch_weight(
    context, experts_to_copy, real_weights, local_weights, epn, *, barrier=True
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
        barrier: 末尾完成屏障（缺省 True 保持原语义：独立 sym_barrier +
            host 级流等待，返回即全组本轮直推完成，连续调用无槽位竞态）。
            **侧流并发调用传 False（async_finish）**：跳过末尾屏障及其
            host 同步，返回发起流上的 NPU 事件——调用方 `wait_event` 汇合，
            跨 rank 槽位复用安全由调用方在合适位置（如下一轮迭代的
            barrier）负责；plan 延迟溢出检查顺延到下一个我方算子尾部。

    说明：
    - home 仅推送有副本需求的行（规划不产生自承载副本：home 的 order 恒为 1）。
    - 每 block 独占一个目的 rank，同一 peer 的所有权重行由同一 block 推送。
    """
    real_weights, local_weights, w_elems = _normalize_weight_groups(
        context, real_weights, local_weights
    )
    sync_transport_stream(context)  # 跨流护栏
    sync_transport(context, real_weights, local_weights, epn)
    # 视图地址推导各组发布区偏移（槽位由视图携带），并校验窗口区对齐
    group_offs = _check_view_addrs(context, local_weights, epn, w_elems)

    gm_mf = _manifest_buf(context, epn, context.rank_size)
    # 多组融合发射：每批最多 3 组（dsl 发射器 host 开销 ~550µs/launch，
    # 独立逐组发射会在 ctx 流上拖出设备空窗）；不足 3 组以首组占位
    # （const_expr 消除未用分支）。批次间同流串行（kernel 顺序执行，
    # gm_mf 复用安全）
    for b0 in range(0, len(local_weights), 3):
        batch_rw = _clamp3(real_weights[b0 : b0 + 3])
        batch_offs = tuple(group_offs[b0 : b0 + 3])
        _get_push_program(epn, w_elems, batch_offs, context.rank_size, context.rank)(
            batch_rw[0],
            batch_rw[1],
            batch_rw[2],
            experts_to_copy,
            gm_mf,
            context.ctx_tensor,
        )
    # 尾段契约见 finish_op（阻塞=独立 barrier+流级等待+溢出检查；async=返回事件）
    return finish_op(context, barrier)
