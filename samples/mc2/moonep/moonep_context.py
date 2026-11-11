# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Communication contexts and symmetric-memory layout for MoonEP weight prefetch.

Resolves torch.distributed process groups into device-side CommContext tensors.
Provides capacity checks, stream ordering, workspace storage and rank synchronization barriers."""

from __future__ import annotations

import threading
import warnings

import torch
import torch_npu

import cannbotdsl
from cannbotdsl import dtypes, make_layout
from cannbotdsl import TensorSpec
from cannbotdsl.buffer import Buffer
from cannbotdsl.types.datastruct import StructRef
from cannbotdsl.tensor import MemLoc, tile_slice
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.sync import PIPE, vec_busy_wait, vec_sync_notify, vec_sync_wait
from cannbotdsl.ops.distributed import (
    CommContext,
    create_transfer_context,
    SymBuffer,
    TransferRole,
    get_comm_context,
)
from cannbotdsl.ops.distributed.context import hccl

# ---------------------------------------------------------------------------
# moonep_layout 为布局契约唯一真源
# （常量 + 区域公式 + LCM），moonep_topology 为 rank graph 层选择补丁
# （host 侧建链驱动——见其模块 docstring）。此处重导出全部符号，
# 既有 import 面（from moonep_context import HEAP_BASE, ...）零变更。
# ---------------------------------------------------------------------------
from moonep_layout import (  # noqa: F401
    BAR_FLAGS_OFF,
    DEM_ALL_OFF,
    HEAP_BASE,
    LAYOUT_TAIL_BYTES,
    MAX_E,
    MAX_R,
    PLAN_REGION_END,
    READY_OFF,
    ROWS_PER_EXPERT,
    SCALE_ALIGN_BYTES,
    WEIGHT_BYTES,
    WEIGHT_DTYPE,
    XFER_WORKSPACE_BYTES,
    grad_slot_off,
    lcm_slot_stride,
    prefetch_weight_off,
    reduce_region_end,
    reduce_stg_off,
    slot_stride_bytes,
)
from moonep_topology import (  # noqa: F401
    _patch_rank_graph_layer_selection,
    _select_rank_graph_links_by_protocol,
)


_RESIDENT_BLOCKS = None  # 本机有效 AIV 核数（首次查询后缓存）
_FALLBACK_BLOCKS = 16  # 平台信息不可用时的保守值（小核数机型也不会卡死在同步屏障）


def _device_ready() -> bool:
    """torch 侧是否已选定 NPU 设备（不触发惰性初始化）。

    `_npu_getDeviceWithoutSet()` 只读当前设备号：未选设备返回 -1，且不建
    流、不初始化设备（torch_npu 的 device 上下文管理器就是用它保存现场）。
    """
    return torch_npu._C._npu_getDeviceWithoutSet() >= 0


def _vector_core_num(ready: bool) -> int:
    """AIV 核数；设备未就绪时走不碰设备的静态规格查询。

    `get_platform_info()` 缺省会取 torch 当前流以叠管控核配额；torch_npu
    尚未选设备时，这一步会做设备初始化——卡被占用/不可用时是分钟级重试，
    被误当成"编译慢"。编译期只需要一个 block 上界，静态 SoC 规格足够：
    传 `stream=False` 跳过当前流查询（拿不到 handle 时不叠配额），
    完全不触发 NPU 惰性初始化。
    """
    if ready:
        return int(cannbotdsl.get_platform_info().vector_core_num)
    return int(cannbotdsl.get_platform_info(stream=False).vector_core_num)


def resident_blocks(cap: int = 64) -> int:
    """本机可同时常驻的 AIV block 数（含 grid 级同步屏障 kernel 的硬上限）。

    `global_sync_all()` 要求**全部 block 同时常驻**：block_dim 超过本机有效
    AIV 核数（例如 56 核机型上排 64 块）时，调度器只能分时复用，grid 级同步屏障永远
    收不齐 → 死循环挂死。核数取 `get_platform_info().vector_core_num`
    （设备已初始化时含流的控核配额）；查询报错直接抛出，只有核数为 0 时
    退保守值而不是猜硬件规格。设备未初始化时只读静态 SoC 规格，不触发
    设备初始化；该结果**不缓存**，运行期设备就绪后再按流配额复核，保证
    实启块数不超过配额。

    cap 是"核数再多也不加块"的经验上限（块数收益递减）；返回值 ≤ cap。
    进程内只在设备已初始化时缓存，保证同一进程内编译期与运行期的 block
    数一致——二者不一致会让 kernel 的角色分派与实启块数错位（不一致时
    按新配置重新编译一次，不会用旧网格启动）。
    """
    global _RESIDENT_BLOCKS
    if _RESIDENT_BLOCKS is not None:
        return min(cap, _RESIDENT_BLOCKS)
    ready = _device_ready()
    cores = _vector_core_num(ready)
    if cores <= 0:
        warnings.warn(
            f"无法获取 AIV 核数，block 数退保守值 {_FALLBACK_BLOCKS}"
            "（使用保守的并行度）",
            RuntimeWarning,
            stacklevel=2,
        )
        _RESIDENT_BLOCKS = _FALLBACK_BLOCKS
    elif ready:
        _RESIDENT_BLOCKS = cores
    else:
        # 设备未初始化：静态规格不缓存，运行期初始化后再按配额复核
        return min(cap, cores)
    return min(cap, _RESIDENT_BLOCKS)


def finish_op(context, async_finish=False):
    """完成权重预取的同步收尾。

    同步模式等待全组完成，记录发起流事件并检查布局容量状态，返回 None。
    异步模式记录并返回发起流事件；调用方通过 wait_event 汇合后复用缓冲区。"""
    if not async_finish:
        sym_barrier(context)
        record_transport_event(context)  # 跨流护栏 fork 点（只等我方算子）
        check_pending_overflow(context)
        return None
    ev = torch.npu.Event()
    ev.record(torch.npu.current_stream())
    record_transport_event(context)
    return ev


def normalize_group_list(context, tensors, op_name, *, min_groups=2, max_groups=3):
    """将 Tensor、list 或 tuple 归一化为列表，并检查组数、形状和布局元数据。

    返回 ``(tensors, widths)``：``widths`` 为各组行宽元组（元素数）——
    支持变宽组（交织型 W13 宽 2W、down 宽 W）。同一上下文内的组数由
    首次调用确定（layout_groups 守卫防发布区/槽区错位）。
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    if not tensors:
        raise ValueError(f"{op_name} 组列表为空（须 {min_groups}~{max_groups} 组）")
    if not (min_groups <= len(tensors) <= max_groups):
        raise ValueError(
            f"{op_name} 支持 {min_groups} 组或 {max_groups} 组（kernel 入口按组数"
            f"固定），实得 {len(tensors)} 组"
        )
    if context.layout_groups is None:
        context.layout_groups = len(tensors)
    elif context.layout_groups != len(tensors):
        raise RuntimeError(
            f"权重组数不一致（已有 {context.layout_groups}，本次 {len(tensors)}），"
            "会导致发布/staging 区重叠"
        )
    widths = []
    for g, t in enumerate(tensors):
        if t.dim() < 2 or t.shape[0] == 0:
            raise ValueError(
                f"{op_name} 第 {g} 组须为 (行, ...) >= 2 维，实得 {tuple(t.shape)}"
            )
        # 行宽 = 尾维展平宽（N-D 形态如 [行, F, D]——调用方免 reshape）；
        # 行宽 0（如 [行, 0] 张量）须入口拦截——下游 LCM 公式会除零
        w = int(t.numel() // t.shape[0])
        if w <= 0:
            raise ValueError(
                f"{op_name} 第 {g} 组行宽须为正（尾维展平宽），实得 {w}"
                f"（张量形状 {tuple(t.shape)}）"
            )
        widths.append(w)
    # 组宽守卫（E4）：与首调定型值一致——梯度槽区基址按实际 widths 计算，
    # 宽度漂移会使槽区起点落进权重发布区（发布/槽区重叠互踩）
    widths = tuple(widths)
    if context.layout_widths is None:
        context.layout_widths = widths
    elif widths != context.layout_widths:
        raise RuntimeError(
            f"{op_name} 组宽不一致（已有 {context.layout_widths}，本次 {widths}）"
            "——宽度变化会使梯度槽区基址错位、发布/槽区重叠"
        )
    if context.last_plan_shape is None:
        raise RuntimeError(
            f"{op_name} 需要先设置 context.last_plan_shape（输入形状与窗口布局）"
        )
    return tensors, widths


def pick_weight_chunk(w_elems: int, ub_limit_elems: int) -> int:
    """UB 分块粒度：整行单块（W<=ub_limit）或最大 2 的幂因子（保证整除、无尾块）。

    DSL 的 tile_slice 拒绝静态部分宽度尾块与超尺寸 tile，因此分块必须
    满足 W % chunk == 0。W=H*Hp（H=3584=2^9*7）必能命中；W 非 512 倍数且超
    上限时抛 ValueError。
    """
    if w_elems <= ub_limit_elems:
        return w_elems
    chunk = ub_limit_elems
    while chunk > 512 and w_elems % chunk != 0:
        chunk //= 2
    if w_elems % chunk != 0:
        raise ValueError(
            f"W={w_elems} 在 UB 上限 {ub_limit_elems} 下无 2 的幂因子整除方案"
        )
    return chunk


class MoonepContext:
    """EP 组级通信上下文句柄：封装设备侧 CommContext 张量与组信息。

    Attributes:
        ctx_tensor: torch.uint8 NPU 张量，CommContext 的设备侧字节（传给算子的
            StructRef[CommContext] 入参）。
        rank: 本 rank 在 EP 组内的 id。
        rank_size: EP 组 rank 数。
    """

    def __init__(
        self, group_name: str, *, op_name: str = "moonep", rank=None, rank_size=None
    ):
        self.group_name = group_name
        self.ctx_tensor = get_comm_context(CommContext, group_name, op_name=op_name)
        import torch.distributed as dist

        # rank/rank_size 缺省回退 WORLD；EP 为子通信域时由 create_moonep_context(group=...) 显式传入
        self.rank = dist.get_rank() if rank is None else rank
        self.rank_size = dist.get_world_size() if rank_size is None else rank_size
        if self.rank_size > 64:
            raise ValueError(
                f"moonep 暂支持 EP 组 rank 数 <= 64（COMM_MAX_RANK_NUM），当前 {self.rank_size}"
            )
        # 单槽清单缓存：(etc 张量强引用, 清单 NPU 张量组)。同一 plan 的 etc 不变时，
        # reduce 重复调用免清单重建（清单由设备侧 kernel 从 etc 生成）；新 etc
        # （新 plan）自动重建。（内部状态，外部只读，勿改）
        self.reduce_manifest = None
        # pull per-peer 清单缓存：(etc 身份, gm_rm, max_rcnt)——peer=block idx。
        self.pull_manifest = None
        # pull 直累 scratch 常驻缓存：((R, W, dtype), gm_scratch [R, W])。
        self.pull_scratch = None
        # 槽视图已用的 slots 值（一致性守卫——create_grad_slot_view 写入）。
        self.pull_slots_used = None
        # plan 中间 workspace 缓存：((S,K,E,R), (dem/topk_all/rem/order))。
        # A/B kernel 对四表全量覆写，同 shape 重复调用免分配。（内部状态，勿改）
        self.plan_workspace = None
        # 跨流安全护栏状态：首个算子调用的发起流；跨流调用时
        # 由 sync_transport_stream 做事件 fork/join 串行化（跨流时等
        # transport_event=上一个我方算子的完成点，而非旧流尾部）
        self.transport_stream = None
        self.transport_event = None
        # 调用者设置为 (S, K)，用于描述输入形状和窗口布局
        self.last_plan_shape = None
        # rank 同步屏障作用域：EP 子组时由 create_moonep_context(group=...) 设置，缺省 WORLD
        self.group = None
        # 多权重组布局守卫：同一上下文的组数须一致，避免发布区与工作区重叠
        # （内部状态，勿改）
        self.layout_groups = None
        # 组结构守卫：各组行宽/scale 行宽首调定型、此后不可变
        # ——梯度槽区基址按实际 widths 计算，宽度漂移会使槽区起点落进发布区
        # （内部状态，勿改）
        self.layout_widths = None
        self.layout_scale_widths = None
        # 权重 dtype 定型值：MoonepBuffer 构造传入（weight_dtype）；
        # 裸 API 首次 prefetch 冻结。dtype 进编译键与视图 dtype，不可漂移
        # （内部状态，勿改）
        self.weight_dtype = None
        # grad dtype 定型值：MoonepBuffer 构造传入（grad_dtype，
        # 缺省 bf16）；裸 API 首次 reduce 冻结。local_grads 的 dtype 校验基
        # （内部状态，勿改）
        self.grad_dtype = None
        # 多槽窗口池层数：首个 create_weight_view 定型（缺省 1 = 单槽），
        # 后续调用须一致；reduce 容量校验按它推导（内部状态，勿改）
        self.window_slots = None
        # 通信域分离契约：ctx 创建时已校验一次；首个算子调用再复查一次，
        # 覆盖"ctx 先建、ElasticBuffer 后建"的顺序（内部状态，勿改）
        self.eb_scan_done = False
        # plan 的延迟溢出检查载荷 (cu_seqlens, num_rows)：plan 返回路径零同步，
        # 布局状态在预取完成后检查（sym_barrier 后流已排空，
        # .item() 微秒级）执行（内部状态，勿改）
        self.pending_overflow = None
        # HCCL 对称缓冲区（base, size）：每次 hccl.get_comm_handle +
        # get_hccl_buffer 都是 C API 调用，而值在 comm 生命周期内不变——
        # 缓存后 create_weight_view / prefetch / reduce 的每步 host 路径省掉它们
        # （内部状态，勿改）
        self.hccl_buf = None
        # 权重视图缓存：{(epn, W, group, groups, slot, slots): tensor}。视图只是
        # 同一块对称堆上的非拥有句柄，重复 create_weight_view 直接复用（免 HCCL
        # API、免容量校验、免 npu_from_blob 扩展调用）（内部状态，勿改）
        self.weight_views = {}
        # rank 同步屏障 epoch 张量（复用，避免每步 torch.full 分配）（内部状态，勿改）
        self.bar_epoch_t = None


def _hccl_buf(context):
    """(base, size) 单次查询后缓存在 context 上（comm 生命周期内不变）。"""
    cached = context.hccl_buf
    if cached is None:
        comm = hccl.get_comm_handle(context.group_name)
        cached = hccl.get_hccl_buffer(comm)
        context.hccl_buf = cached
    return cached


def hccl_buffer_base(context):
    """本 rank HCCL 对称缓冲区基址（SymBuffer 的 commBufferAddrs[me]）。"""
    return _hccl_buf(context)[0]


def hccl_buffer_size(context):
    """本 rank HCCL 对称缓冲区字节数。"""
    return _hccl_buf(context)[1]


def check_heap_capacity(context, need_bytes, what):
    """对称堆容量前置校验：不足即报所需/现有，防运行中 507899
    静默越界（"driver internal error" 不指向根因，排查代价极高）。

    need_bytes 为相对 HCCL 缓冲区起点的总占用（含 HEAP_BASE）。
    """
    heap_bytes = hccl_buffer_size(context)
    if need_bytes > heap_bytes:
        raise RuntimeError(
            f"{what}超出 HCCL 对称缓冲区：需要 {need_bytes} B "
            f"({need_bytes / 2**20:.1f} MiB)，可用 {heap_bytes} B "
            f"({heap_bytes / 2**20:.1f} MiB)。请调大 HCCL_BUFFSIZE。"
        )


def check_pending_overflow(context):
    """检查上下文中待验证的布局容量状态；没有待验证状态时直接返回。"""
    pending = context.pending_overflow
    if pending is None:
        return
    context.pending_overflow = None
    cu_seqlens, num_rows = pending
    cu_tail = int(cu_seqlens[-1].item())
    if cu_tail != num_rows:
        raise RuntimeError(
            f"副本槽溢出：极端路由下单 rank 副本数超过 epn={cu_seqlens.shape[0] // 2}"
            f" 个预留槽（cu_seqlens 尾值={cu_tail} != N={num_rows}）。"
            "请检查专家清单与布局元数据。"
        )


def check_no_shared_elastic_buffer(group_name):
    """通信域分离契约校验：同 HCCL 通信器上存在存活 ElasticBuffer 即报错。

    ElasticBuffer（moe_ep dispatch/combine）与 moonep 算子上下文共用通信器
    即共用同一块 HCCL 对称堆：双方各自圈地导致 flag/staging 互相踩踏，表现
    为 dispatch AIV 自旋死锁（~600s 后 507899 vector core timeout）。本校验
    把挂死变为创建期/首次算子调用时的秒级明确错误。EB 须与
    create_moonep_context 各用独立通信器（dist.new_group）。
    """
    import gc

    try:
        from cann_ops_transformer.ops import ElasticBuffer
    except ImportError:
        return
    for obj in gc.get_objects():
        if not isinstance(obj, ElasticBuffer):
            continue
        if getattr(obj, "_runtime", None) is None:  # 已 destroy 的实例不拦截
            continue
        if getattr(obj, "_group_name", None) == group_name:
            raise RuntimeError(
                "ElasticBuffer 与 moonep 算子上下文共用同一 HCCL 通信器"
                f"（{group_name}）：双方在同一块对称堆上各自圈地会互相踩踏，"
                "dispatch 将自旋死锁（507899）。请为 ElasticBuffer 与 "
                "create_moonep_context 各建独立通信器（dist.new_group）。"
            )


def busy_wait_flag(view, gm_epoch, event_id, ub_flag=None, ub_epoch=None, exact=False):
    """轮询 view 处 flag 就位（缺省 >= gm_epoch；exact=True 时 == gm_epoch）。

    该握手原语用于同步标志。
    ub_flag/ub_epoch 缺省现配；recv 侧按组轮询时由调用方共享传入，
    避免每实例一对 UB 缓冲耗尽 VEC buf_id。

    两种匹配语义的分工：
    - **>=（缺省，rank 同步屏障 / plan ready 必须用）**：容忍"peer 已进入下一轮、
      用 n+1 覆盖了自己槽位"——这两类收齐协议没有"对方已读走本轮 flag"的
      回执，等待方可能读到超前值，改 == 会死锁。
    - **==（exact=True，仅 reduce staging flag）**：WORLD 组对称堆在同进程
      多用例间复用，flag 槽位的**残留数据**（前置 prefetch/多槽用例写入的
      权重编码，bf16 位型当 int32 读数 ~1e9 大正数）在 >= 下被当作"已就位"，
      recv 首轮即放行、读到在途的旧 staging。reduce 有组间 rank 同步屏障充当"读
      完成回执"，== 无被下一轮覆盖的死锁风险；残留恰等于当轮 epoch 需 32 位
      巧合，概率可忽略。
    """
    if ub_flag is None:
        ub_flag = Buffer(MemLoc.UB, dtypes.int32, make_layout((1,)))
        ub_epoch = Buffer(MemLoc.UB, dtypes.int32, make_layout((1,)))

    def _check():
        vec_sync_notify(PIPE.S, PIPE.MTE2, event_id)
        vec_sync_wait(PIPE.S, PIPE.MTE2, event_id)
        mem_copy(ub_flag, view)
        mem_copy(ub_epoch, gm_epoch)
        vec_sync_notify(PIPE.MTE2, PIPE.S, event_id)
        vec_sync_wait(PIPE.MTE2, PIPE.S, event_id)
        if exact:
            return ub_flag[dtypes.int64(0)] == ub_epoch[dtypes.int64(0)]
        return ub_flag[dtypes.int64(0)] >= ub_epoch[dtypes.int64(0)]

    vec_busy_wait(_check)


# ---------------------------------------------------------------------------
# 对称堆上的 rank 自旋同步屏障
# ---------------------------------------------------------------------------
# sym_barrier 独立实现的依据：①流级设备序保证强于
# host 侧集合通信（kernel 返回即全组本轮此前流上工作完成）；②无 host 往返；
# ③epoch 单调防陈旧 flag 提前放行。用与数据面相同的 put/flag 通道做 rank
# 对齐：flag 与数据同通道保序，不经过 HCCL 集合通信路径。
_BAR_EVENT_ID = 5  # plan=7、reduce=6、barrier=5
_BAR_EPOCH = 0  # 跨算子共享的单调 rank 同步屏障 epoch（各次调用共用一套 flag）
_BAR_PROGRAMS = {}
_BAR_LOCK = threading.Lock()


@kernel
class _MoonepBarrierKernel:
    """各 rank 把本轮 epoch 直推全部 peer 的 flag 槽，再自旋收齐（单 block）。"""

    def __init__(self, rank, rank_size):
        self._me = int(rank)
        self._r = int(rank_size)

    def __call__(self, gm_epoch, ctx: StructRef[CommContext]):
        world_size, me = self._r, self._me
        flags = SymBuffer(
            ctx,
            shape=(MAX_R,),
            dtype=dtypes.int32,
            rank_size=world_size,
            base_offset=BAR_FLAGS_OFF,
        )
        ub_xfer = Buffer(MemLoc.UB, dtypes.uint8, make_layout((512,)))
        xfer = create_transfer_context(ctx, ub_xfer, role=TransferRole.VEC)
        for r in range_constexpr(world_size):
            if const_expr(r != me):
                xfer.put(r, tile_slice(flags, (1,), (me,)).peer(r), gm_epoch)
        for r in range_constexpr(world_size):
            if const_expr(r != me):
                xfer.flush(r)
        # 共享一对轮询缓冲：range_constexpr 逐 peer 展开时缺省现配会让每个
        # busy_wait_flag 实例各占一对 UB 缓冲——R=8 时 7 对耗尽 VEC buf_id
        # （aivec 271 "scalar access internal buffer out of bounds"；R=4 的
        # 3 对未触界）。与 reduce _poll_row 的共享形态同源。
        ub_flag = Buffer(MemLoc.UB, dtypes.int32, make_layout((1,)))
        ub_epoch = Buffer(MemLoc.UB, dtypes.int32, make_layout((1,)))
        for r in range_constexpr(world_size):
            if const_expr(r != me):
                busy_wait_flag(
                    tile_slice(flags, (1,), (r,)),
                    gm_epoch,
                    _BAR_EVENT_ID,
                    ub_flag,
                    ub_epoch,
                )


def _make_barrier_host(me, world_size):
    """闭包工厂：rank/rank_size 进闭包参数（无 global 烘焙）。"""

    @host
    def _barrier_host(gm_epoch, ctx: StructRef[CommContext]):
        _MoonepBarrierKernel(me, world_size)[1](gm_epoch, ctx)

    return _barrier_host


def record_transport_event(context, stream=None):
    """各算子 wrapper 发射后记录完成事件（跨流护栏的 fork 点）。

    后续跨流调用只等待本事件（上一个我方算子的完成点），而非发起流尾部
    （主流尾部可能排着 dispatch 等非我方工作，等尾部会把双流重叠串行掉）。
    事件对象复用（每步新建 torch.npu.Event() 有 host 分配开销；单线程
    按序"先等后录"，重录语义安全）。stream 缺省取当前流；跨流调用方
    传入 sync_transport_stream 返回的发起流，保证事件录在算子实际发射的流上。
    """
    ev = context.transport_event
    if ev is None:
        ev = context.transport_event = torch.npu.Event()
    ev.record(stream if stream is not None else torch.npu.current_stream())


def _get_barrier_program(world_size, me):
    """(world_size, me) 键控的 rank 同步屏障 program（闭包参数化，无 global）。"""
    key = (world_size, me)
    with _BAR_LOCK:
        prog = _BAR_PROGRAMS.get(key)
        if prog is None:
            prog = cannbotdsl.compile(
                _make_barrier_host(me, world_size),
                TensorSpec((1,), dtypes.int32),
                StructRef[CommContext],
            )
            _BAR_PROGRAMS[key] = prog
    return prog


def next_barrier_epoch(context):
    """推进本轮 barrier epoch 并写回设备张量（1 元素 int32），返回该张量。

    独立 barrier kernel 与内联在算子里的收尾 barrier 共用同一个 epoch 计数，
    保证两种形态交错调用时 flag 不会串轮。
    """
    global _BAR_EPOCH
    _BAR_EPOCH += 1
    gm_epoch = context.bar_epoch_t
    if gm_epoch is None:
        gm_epoch = torch.empty(1, dtype=torch.int32, device="npu")
        context.bar_epoch_t = gm_epoch
    gm_epoch.fill_(_BAR_EPOCH)  # 复用张量：省掉每步一次分配
    return gm_epoch


def sym_barrier(context, *, sync=True):
    """对称堆自旋 barrier：dist.barrier 的流级强化替代（非因后者不可用）。

    语义与同流 dist.barrier 等价且更强：返回时全组 rank 本轮此前流上工作全部
    完成（kernel 内自旋收齐各 peer 的 epoch flag，flag 与数据同通道保序）。
    epoch 跨算子单调递增，陈旧 flag 不会提前放行。

    sync=False 为"只发射"形态：仅入队 rank 同步屏障 kernel、不做 host 同步——
    跨 rank 保序由 kernel 自身的自旋协议保证（后继同流 kernel 在设备序上
    排在它之后），host 立即返回。供 async_finish 路径的组间 staging 复用
    同步屏障使用：staging 覆写的跨 rank 安全是设备序问题，不需要 host 等待。
    """
    gm_epoch = next_barrier_epoch(context)
    _get_barrier_program(context.rank_size, context.rank)(gm_epoch, context.ctx_tensor)
    if not sync:
        return
    # 等待范围为发起流而非整卡：rank 同步屏障 kernel 在发起流上，等本流完成即意味着
    # 全组本轮握手收齐。全设备 synchronize 会把无关流上的在飞工作也等进来，
    # 封死跨层/跨流流水（多层训练里每层 reduce/组间串行都走本 rank 同步屏障）。
    torch.npu.current_stream().synchronize()


def sync_transport_stream(context):
    """跨流安全护栏：算子握手状态机（epoch flag 轮询 + staging
    槽位复用）要求调用严格按发起序执行。

    - 首次调用：记录发起流；
    - 同流调用：快路径，仅一次比较，零开销；
    - 跨流调用（激活重算等 autograd 场景）：在旧流上记录事件、新流等待——
      事件 fork/join 把调用序列串行化，握手语义保持成立（替代框架侧钉流）。

    覆盖 moonep 各算子（plan / prefetch_weight）；dispatch/combine 为 moe_ep
    原生算子，不在本护栏内——框架侧仍需同流发起或重算回放。返回发起流，
    供 wrapper 在算子实际发射的流上记录完成事件。
    """
    cur = torch.npu.current_stream()
    ts = context.transport_stream
    if ts is None or ts == cur:
        context.transport_stream = cur
        return cur
    # 跨流：等上一个我方算子的完成事件（而非旧流尾部——主流尾部可能排着
    # dispatch 等非我方工作，等尾部会把双流重叠串行掉）
    if context.transport_event is not None:
        cur.wait_event(context.transport_event)
    context.transport_stream = cur
    return cur


def _hccl_comm_name(group, rank):
    """由 ProcessGroup 解析 HCCL comm name。

    torch 未公开 backend 访问接口（get_hccl_comm_name 只在 backend 对象上），
    全仓仅此一处触达保护成员。
    """
    # torch 未公开 backend 对象访问接口（get_backend 只返回名字枚举，拿不到
    # ProcessGroupHCCL 实例）；经属性名动态获取，语义与直接点取完全一致
    backend = getattr(group, "_get_backend")(torch.device("npu"))
    return backend.get_hccl_comm_name(rank)


def create_moonep_context(
    group_name: str | None = None, *, group=None
) -> MoonepContext:
    """创建 moonep 通信上下文。

    group_name 缺省时用 torch.distributed 默认组；EP 为子通信域时传 ``group``
    （ProcessGroup），rank/rank_size 与 HCCL comm name 均按组内取值。
    """
    import torch.distributed as dist

    if group is not None:
        rank = dist.get_rank(group)
        rank_size = dist.get_world_size(group)
        if group_name is None:
            group_name = _hccl_comm_name(group, rank)
        ctx = MoonepContext(group_name, rank=rank, rank_size=rank_size)
        ctx.group = group
        check_no_shared_elastic_buffer(group_name)
        check_heap_capacity(ctx, HEAP_BASE + PLAN_REGION_END, "plan 交换区")
        return ctx
    if group_name is None:
        group_name = _hccl_comm_name(dist.group.WORLD, dist.get_rank())
    ctx = MoonepContext(group_name)
    check_no_shared_elastic_buffer(group_name)
    check_heap_capacity(ctx, HEAP_BASE + PLAN_REGION_END, "plan 交换区")
    return ctx
