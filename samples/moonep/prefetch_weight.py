# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""prefetch_weight: push replica expert weights from home to consumer ranks.

Direct-push single-phase: home uses xfer.put to write each replica row into
the consumer's symmetric prefetch slot. Self-hosted replicas use local index
copy. A trailing barrier guards against slot-reuse races. Supports multiple
weight groups (gate/up/down) in a single call.

清单由设备侧 kernel 从 etc 生成（定长数组 + 运行时计数）：host 侧零回读/
零 Python 扫描/零 H2D，编译键与路由结构解耦（流序写入纪律：清单只经
同流 kernel 全量覆写，免疫 host 写竞态）。
"""

import threading

import torch

import cannbotdsl
from cannbotdsl import dtypes
from cannbotdsl import TensorSpec
from cannbotdsl.types.datastruct import StructRef
from cannbotdsl.buffer import Buffer
from cannbotdsl.tensor import MemLoc, tile_slice
from cannbotdsl.lang.vf import vf
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.distributed import (
    CommContext,
    create_transfer_context,
    SymBuffer,
    TransferRole,
)

from moonep_context import (
    HEAP_BASE,
    barrier_tail,
    check_heap_capacity,
    check_pending_overflow,
    hccl_buffer_base,
    next_barrier_epoch,
    prefetch_weight_off,
    reduce_stg_off,
    scalar_read_fence_post,
    scalar_read_fence_pre,
    sync_transport_stream,
)
from npu_blob import npu_from_blob

_EVENT_ID = 4  # 清单标量读护栏事件（plan=7、reduce=6、barrier=5、prefetch=4）


@kernel
class _MoonepPrefetchPushKernel:
    """home 把有副本需求的真实专家行直推到各消费 rank 的预取槽。

    多组融合发射：单次 launch 服务最多 3 个权重组（组偏移烘焙，trace 期
    const_expr 展开逐组 SymBuffer/put）——dsl 发射器 host 开销 ~550µs/launch
    （profiling 实测），3 组独立发射会在 ctx 流上拖出设备空窗。
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
        kr_dst_rank,
        kr_dst_slot,
        kr_src_row,
        kr_cnt,
        kr_bar_epoch,
        ktx: StructRef[CommContext],
    ):
        # 参数名与 _push_host 刻意错开（位置绑定，名不作数），避免签名段重复误报
        world_size = self._r
        ub_xfer = Buffer(MemLoc.UB, (512,), dtypes.uint8)
        xfer = create_transfer_context(ktx, ub_xfer, role=TransferRole.VEC)

        cnt, ub_dst_rank, ub_dst_slot, ub_src_row = self._load_manifest(
            kr_dst_rank, kr_dst_slot, kr_src_row, kr_cnt
        )

        self._push_groups(
            ktx,
            xfer,
            cnt,
            ub_dst_rank,
            ub_dst_slot,
            ub_src_row,
            kr_real0,
            kr_real1,
            kr_real2,
        )
        self._flush_all(xfer, cnt, ub_dst_rank)
        self._barrier_epilogue(ktx, xfer, kr_bar_epoch, world_size)

    @jit
    def _push_groups(
        self,
        ctx,
        xfer,
        cnt,
        ub_dst_rank,
        ub_dst_slot,
        ub_src_row,
        gm_real0,
        gm_real1,
        gm_real2,
    ):
        """逐组直推（const_expr 展开）：清单条目逐条 put 到对端预取槽。"""
        world_size, epn, w_elems = self._r, self._epn, self._w
        reals = (gm_real0, gm_real1, gm_real2)
        for g in range_constexpr(self._num_g):
            off = self._offs[g]
            gm_real = reals[g]
            w_sym = SymBuffer(
                ctx,
                shape=(2 * epn, w_elems),
                dtype=dtypes.bfloat16,
                rank_size=world_size,
                base_offset=off,
            )
            for i in cannbotdsl.range(cnt):
                dst_r = dtypes.int64(ub_dst_rank[i])
                slot = ub_dst_slot[i]
                row = ub_src_row[i]
                w_tile = tile_slice(w_sym, (1, w_elems), (slot, 0))
                s_tile = tile_slice(gm_real, (1, w_elems), (row, 0))
                xfer.put(dst_r, w_tile.peer(dst_r), s_tile)

    @jit
    def _flush_all(self, xfer, cnt, ub_dst_rank):
        """统一 flush：全部组的 put 入队后按 peer 一次性冲刷（UDMA 各通道并行）。"""
        for i in cannbotdsl.range(cnt):
            xfer.flush(dtypes.int64(ub_dst_rank[i]))

    @jit
    def _barrier_epilogue(self, ctx, xfer, gm_bar_epoch, world_size):
        """融合 barrier 尾段：本 kernel 完成即全组本轮直推完成——语义与
        独立 sym_barrier 等价（省独立 launch 的发射器 host 开销）。"""
        ub_bar_flag = Buffer(MemLoc.UB, (1,), dtypes.int32)
        ub_bar_epoch = Buffer(MemLoc.UB, (1,), dtypes.int32)
        barrier_tail(
            xfer, ctx, gm_bar_epoch, self._me, world_size, ub_bar_flag, ub_bar_epoch
        )

    @jit
    def _load_manifest(self, gm_dst_rank, gm_dst_slot, gm_src_row, gm_cnt):
        """清单 GM→UB 拷贝 + 完整握手（cnt/坐标读到未落地 UB 会静默跳过直推）。

        清单由 _MoonepPrefetchPushManifestKernel 在设备侧生成（定长数组 +
        运行时计数）：编译键与路由结构解耦，host 侧零回读/零 Python 扫描。
        """
        max_cnt = gm_dst_rank.shape[0]
        scalar_read_fence_pre(_EVENT_ID)
        ub_cnt = Buffer(MemLoc.UB, (1,), dtypes.int32)
        ub_dst_rank = Buffer(MemLoc.UB, (max_cnt,), dtypes.int32)
        ub_dst_slot = Buffer(MemLoc.UB, (max_cnt,), dtypes.int32)
        ub_src_row = Buffer(MemLoc.UB, (max_cnt,), dtypes.int32)
        mem_copy(ub_cnt, gm_cnt)
        mem_copy(ub_dst_rank, gm_dst_rank)
        mem_copy(ub_dst_slot, gm_dst_slot)
        mem_copy(ub_src_row, gm_src_row)
        scalar_read_fence_post(_EVENT_ID)
        return ub_cnt[dtypes.int64(0)], ub_dst_rank, ub_dst_slot, ub_src_row


_WEIGHT_OFFS = ()  # wrapper 在编译锁内写入（trace 期模块级常量：各组发布区偏移）
_R = 1  # 组内 rank_size（wrapper 在编译前写入，同 shape 不同 rank_size 须进键）
_ME = 0  # 组内 rank（清单寻址依赖，进清单 program 编译键）


@kernel
class _MoonepPrefetchPushManifestKernel:
    """设备侧生成 home 直推清单（替代 host 侧 etc 扫描）。

    扫描其他 rank 的 etc 行 [R-1, epn]，命中本 rank 本地专家区间
    [me*epn, (me+1)*epn) 的条目即一条直推 (dst_rank=r, dst_slot=epn+j,
    src_row=e-me*epn)。输出定长数组（epn*(R-1)）+ 运行时计数——
    编译键与路由解耦，host 侧零回读/零 Python 扫描。
    """

    def __init__(self, rank, rank_size, epn):
        self._me = int(rank)
        self._r = int(rank_size)
        self._epn = int(epn)

    def __call__(self, gm_etc, gm_dst_rank, gm_dst_slot, gm_src_row, gm_cnt):
        world_size, epn, me = self._r, self._epn, self._me
        max_cnt = gm_dst_rank.shape[0]
        ub_etc = Buffer(MemLoc.UB, (world_size, epn), dtypes.int32)
        mem_copy(ub_etc, gm_etc)
        ub_dst_rank = Buffer(MemLoc.UB, (max_cnt,), dtypes.int32)
        ub_dst_slot = Buffer(MemLoc.UB, (max_cnt,), dtypes.int32)
        ub_src_row = Buffer(MemLoc.UB, (max_cnt,), dtypes.int32)
        ub_cnt = Buffer(MemLoc.UB, (1,), dtypes.int32)

        with vf(mode="simt", thread=1):
            # 计数器驻留 UB（跨展开的 rank 段保持，避免跨循环携带标量）
            ub_cnt[dtypes.int64(0)] = dtypes.int32(0)
            for r in range_constexpr(world_size):
                if const_expr(r != me):
                    self._scan_rank_row(
                        r, ub_etc, ub_dst_rank, ub_dst_slot, ub_src_row, ub_cnt
                    )
        mem_copy(gm_dst_rank, ub_dst_rank)
        mem_copy(gm_dst_slot, ub_dst_slot)
        mem_copy(gm_src_row, ub_src_row)
        mem_copy(gm_cnt, ub_cnt)

    @jit
    def _scan_rank_row(self, r, ub_etc, ub_dst_rank, ub_dst_slot, ub_src_row, ub_cnt):
        """单 peer 的 etc 行扫描：命中本地专家区间的条目追加进清单。"""
        epn, me = self._epn, self._me
        for j in cannbotdsl.range(epn):
            e = ub_etc[dtypes.int64(r), dtypes.int64(j)]
            if e >= dtypes.int32(me * epn):
                if e < dtypes.int32((me + 1) * epn):
                    c = ub_cnt[dtypes.int64(0)]
                    ub_dst_rank[c] = dtypes.int32(r)
                    ub_dst_slot[c] = dtypes.int32(epn) + dtypes.int32(j)
                    ub_src_row[c] = e - dtypes.int32(me * epn)
                    ub_cnt[dtypes.int64(0)] = c + dtypes.int32(1)


@host
def _push_manifest_host(gm_etc, gm_m_dst, gm_m_slot, gm_m_row, gm_m_cnt):
    """清单 program 入口（参数名与 push host 刻意不同，避免签名段重复误报）。"""
    world_size, epn = gm_etc.shape
    _MoonepPrefetchPushManifestKernel(_ME, world_size, epn)[1](
        gm_etc, gm_m_dst, gm_m_slot, gm_m_row, gm_m_cnt
    )


@host
def _push_host(
    gm_real0,
    gm_real1,
    gm_real2,
    gm_dst_rank,
    gm_dst_slot,
    gm_src_row,
    gm_cnt,
    gm_bar_epoch,
    ctx: StructRef[CommContext],
):
    epn2, w_elems = gm_real0.shape
    _MoonepPrefetchPushKernel(_ME, _R, epn2, w_elems, _WEIGHT_OFFS)[1](
        gm_real0,
        gm_real1,
        gm_real2,
        gm_dst_rank,
        gm_dst_slot,
        gm_src_row,
        gm_cnt,
        gm_bar_epoch,
        ctx,
    )


_PUSH_PROGRAMS = {}
_PUSH_MANIFEST_PROGRAMS = {}
_PUSH_LOCK = threading.Lock()


def _get_push_manifest_program(epn, world_size, me):
    """清单 program：(epn, rank_size, me) 键控（_ME 烘焙与编译在锁内原子完成）。"""
    key = (epn, world_size, me)
    with _PUSH_LOCK:
        prog = _PUSH_MANIFEST_PROGRAMS.get(key)
        if prog is None:
            global _ME
            _ME = me
            i32 = dtypes.int32
            max_cnt = epn * (world_size - 1)
            prog = cannbotdsl.compile(
                _push_manifest_host,
                TensorSpec((world_size, epn), i32),
                TensorSpec((max_cnt,), i32),
                TensorSpec((max_cnt,), i32),
                TensorSpec((max_cnt,), i32),
                TensorSpec((1,), i32),
            )
            _PUSH_MANIFEST_PROGRAMS[key] = prog
    return prog


def _get_push_program(epn, w_elems, weight_offs, rank_size, me):
    """(epn, W, 组偏移元组, rank_size) 键控的融合直推 program——不含路由结构
    （清单为 epn*(R-1) 定长数组 + 运行时计数，路由变化不重编译）。

    _WEIGHT_OFFS/_R 的烘焙与编译在锁内原子完成（多线程并发首次编译不同
    组偏移时，烘焙窗口互污会产出错误 program）。
    """
    key = (epn, w_elems, tuple(weight_offs), rank_size, me)
    with _PUSH_LOCK:
        prog = _PUSH_PROGRAMS.get(key)
        if prog is None:
            global _WEIGHT_OFFS, _R, _ME
            _WEIGHT_OFFS, _R, _ME = tuple(weight_offs), rank_size, me
            max_cnt = epn * (rank_size - 1)
            prog = cannbotdsl.compile(
                _push_host,
                TensorSpec((epn, w_elems), dtypes.bfloat16),
                TensorSpec((epn, w_elems), dtypes.bfloat16),
                TensorSpec((epn, w_elems), dtypes.bfloat16),
                TensorSpec((max_cnt,), dtypes.int32),
                TensorSpec((max_cnt,), dtypes.int32),
                TensorSpec((max_cnt,), dtypes.int32),
                TensorSpec((1,), dtypes.int32),
                TensorSpec((1,), dtypes.int32),
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
    return npu_from_blob(addr, (2 * epn, w_elems), "bf16")


def _build_push_manifest(context, experts_to_copy, epn):
    """home 视角的直推清单（设备侧 kernel 生成，按 etc 身份单槽缓存）。

    覆盖本 rank 每个真实专家在其他 rank 的全部副本（规划不产生自承载副本：
    home 的 order 恒为 1，不会进自己的 etc）。清单由 _MoonepPrefetchPushManifestKernel
    在设备侧从 etc 生成——host 侧零回读/零 Python 扫描/零 H2D（流序写入
    纪律：清单只经同流 kernel 全量覆写，免疫 host 写竞态）。
    """
    cache = context.prefetch_manifest
    if cache is not None and cache[0] is experts_to_copy:
        return cache[1]
    i32 = torch.int32
    max_cnt = epn * (context.rank_size - 1)
    manifest = (
        torch.zeros(max_cnt, dtype=i32, device="npu"),
        torch.zeros(max_cnt, dtype=i32, device="npu"),
        torch.zeros(max_cnt, dtype=i32, device="npu"),
        torch.zeros(1, dtype=i32, device="npu"),
    )
    _get_push_manifest_program(epn, context.rank_size, context.rank)(
        experts_to_copy, *manifest
    )
    context.prefetch_manifest = (experts_to_copy, manifest)
    return manifest


def _normalize_weight_groups(context, real_weights, local_weights):
    """归一化为 list 并校验组数/行宽/调用序，返回 (real_weights, local_weights, w_elems)。"""
    if isinstance(real_weights, torch.Tensor):
        real_weights = [real_weights]
    if isinstance(local_weights, torch.Tensor):
        local_weights = [local_weights]
    if not local_weights or len(real_weights) != len(local_weights):
        raise ValueError("real_weights 与 local_weights 组数必须一致且至少一组")
    if context.layout_groups is None:
        context.layout_groups = len(local_weights)
    elif context.layout_groups != len(local_weights):
        raise RuntimeError(
            f"prefetch/reduce 组数不一致（已有 {context.layout_groups}，本次 {len(local_weights)}），"
            "会导致发布/staging 区重叠"
        )
    w_elems = local_weights[0].shape[1]
    if any(w.shape[1] != w_elems for w in local_weights) or any(
        w.shape[1] != w_elems for w in real_weights
    ):
        raise ValueError("多权重组的行宽 W 必须一致（gate/up/down 同为 H*H'）")
    if context.last_plan_shape is None:
        raise RuntimeError(
            "prefetch_weight 需先调用 moonep_plan（依赖其形状与窗口布局）"
        )
    return real_weights, local_weights, w_elems


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


def prefetch_weight(context, experts_to_copy, real_weights, local_weights, epn):
    """home 直推预取本 rank 托管的远程专家权重（单阶段，无 publish/get）。

    Args:
        context: MoonepContext
        experts_to_copy: [R, epn] int32，moonep_plan 输出（-1 填充）
        real_weights: 普通 NPU 张量 [epn, W] bf16（或多组 list/tuple），
            真实本地专家权重，作为直推数据源（DSL 直接读它，不碰对称区视图）。
        local_weights: `create_weight_view` 创建的 [2*epn, W] bf16 对称区权重视图
            （或多组 list/tuple，组数与 real_weights 一致）。
            前 epn 行为真实本地专家（调用前已填好），后 epn 行为预取槽（本函数填充）。
        epn: 每 rank 真实本地专家数

    说明：
    - home 仅推送有副本需求的行（规划不产生自承载副本：home 的 order 恒为 1）。
    - 末尾完成屏障（融合在 kernel 尾段）：本轮直推全部完成后才返回，
      连续多次调用（host 分块）无槽位竞态。
    """
    real_weights, local_weights, w_elems = _normalize_weight_groups(
        context, real_weights, local_weights
    )
    sync_transport_stream(context)  # 跨流护栏
    # 视图地址推导各组发布区偏移（槽位由视图携带），并校验窗口区对齐
    group_offs = _check_view_addrs(context, local_weights, epn, w_elems)

    dst_rank, dst_slot, src_row, gm_cnt = _build_push_manifest(
        context, experts_to_copy, epn
    )
    # 融合 barrier 尾段：kernel 完成即全组本轮直推完成（epoch 跨算子单调，
    # 与独立 sym_barrier 同 flag 协议；省一次发射器 launch ~550µs）
    gm_bar_epoch = torch.full(
        (1,), next_barrier_epoch(), dtype=torch.int32, device="npu"
    )
    # 多组融合发射：每批最多 3 组（dsl 发射器 host 开销 ~550µs/launch，
    # 独立逐组发射会在 ctx 流上拖出设备空窗）；不足 3 组以首组占位
    # （const_expr 消除未用分支）。批次间同流串行，barrier 尾段天然保序
    for b0 in range(0, len(local_weights), 3):
        b_hi = b0 + 3
        batch_rw = real_weights[b0:b_hi]
        batch_offs = tuple(group_offs[b0:b_hi])
        _get_push_program(epn, w_elems, batch_offs, context.rank_size, context.rank)(
            batch_rw[0],
            batch_rw[min(1, len(batch_rw) - 1)],
            batch_rw[min(2, len(batch_rw) - 1)],
            dst_rank,
            dst_slot,
            src_row,
            gm_cnt,
            gm_bar_epoch,
            context.ctx_tensor,
        )
    # 返回保证：发起流级等待（融合 barrier 在 kernel 尾段，流排空即全组完成）
    torch.npu.current_stream().synchronize()
    # plan 延迟溢出检查（此处流已排空，.item() 微秒级）
    check_pending_overflow(context)
