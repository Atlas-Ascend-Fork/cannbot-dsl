# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared infrastructure for the five MoonEP operators.

Wraps cannbotdsl get_comm_context: resolves the HCCL communicator for a
torch.distributed process group into a device-side CommContext tensor.
Also hosts: cross-stream guard, host-side etc cache, workspace caching,
symmetric-heap capacity pre-check, the spin barrier used as the ops'
completion fence, and the scalar-read fences for runtime manifests.
"""

from __future__ import annotations

import threading

import torch

import cannbotdsl
from cannbotdsl import dtypes
from cannbotdsl import TensorSpec
from cannbotdsl.types.datastruct import StructRef
from cannbotdsl.buffer import Buffer
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

# 对称堆内布局（字节偏移，均以 heap_base=4096 为基址，32B 对齐）：
#   [0, PLAN_END)                规划交换区（demand/topk 表等 + barrier flag）
#   [PUB_OFF, ...)               prefetch 权重发布区（(epn, W) bf16）
#   [STG_OFF, ...)               reduce_grad 梯度 staging 区（(R, epn, W) fp32）
# 各区域基址由下列函数按运行时形状计算，所有 rank 公式一致 → 结果一致。
HEAP_BASE = 4096  # 对称堆基址（与 cannbotdsl SymBuffer 的默认 heap_base 一致）
MAX_E = 1024  # v1 支持的专家总数上限
MAX_R = 64  # COMM_MAX_RANK_NUM
DEM_ALL_OFF = 0  # (MAX_R, MAX_E) int32：各 rank 本地计数交换区（256KB）
READY_OFF = MAX_R * MAX_E * 4  # (MAX_R,) int32：交换完成标志（紧随 dem 区；topk
# 交换区已随死交换删除+host 侧直拷移除，不再占位）
BAR_FLAGS_OFF = READY_OFF + MAX_R * 4  # (MAX_R,) int32：自旋 barrier 的 epoch flag
LAYOUT_TAIL_BYTES = 4096  # 区域尾部对齐余量（32B 对齐要求下取整页）
PLAN_REGION_END = BAR_FLAGS_OFF + MAX_R * 4 + LAYOUT_TAIL_BYTES


def prefetch_weight_off() -> int:
    """prefetch 权重发布区基址（紧随 plan 交换区）。"""
    return PLAN_REGION_END


def reduce_stg_off(epn: int, w_elems: int, num_groups: int = 1, slots: int = 1) -> int:
    """reduce_grad staging 区基址（发布区按 slots×num_groups 预留）。

    每组发布区是完整的 [2*epn, W] bf16 权重视图（真实行 + 预取槽同址），
    组步长必须按整视图 2*epn*W*2 字节计——按 epn*W*2 计会使组间重叠一半
    （组 g+1 起点压在组 g 的预取槽区正中）。slots 为多槽窗口池层数
    （槽 s、组 g 的视图基址 = prefetch_weight_off + (s*num_groups+g)*步长）：
    双槽轮转使 GEMM 可直读当前槽视图，下一层 prefetch 写另一槽，免快照拷贝。
    """
    return prefetch_weight_off() + slots * num_groups * 2 * epn * w_elems * 2


def reduce_region_end(
    world_size: int, epn: int, w_elems: int, num_groups: int = 1, slots: int = 1
) -> int:
    """Symmetric heap upper bound (for capacity self-check).

    staging 组间复用：与 num_groups 无关（组间串行共享同一块）。
    """
    return (
        reduce_stg_off(epn, w_elems, num_groups, slots)
        + world_size * epn * w_elems * 4
        + world_size * epn * 4
        + LAYOUT_TAIL_BYTES
    )


def pick_weight_chunk(w_elems: int, ub_limit_elems: int) -> int:
    """UB 分块粒度：整行单块（W<=ub_limit）或最大 2 的幂因子（保证整除、无尾块）。

    DSL 的 tile_slice 拒绝静态部分宽度尾块与超尺寸 tile，因此分块必须
    满足 W % chunk == 0。W=H*CH2（H=7168=2^10*7）必能命中；W 非 512 倍数且超
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
        # prefetch/reduce 重复调用免清单重建（清单由设备侧 kernel 从 etc 生成）；
        # 新 etc（新 plan）自动重建。（内部状态，外部只读，勿改）
        self.prefetch_manifest = None
        self.reduce_manifest = None
        # plan 中间 workspace 缓存：((S,K,E,R), (dem/topk_all/rem/order))。
        # A/B kernel 对四表全量覆写，同 shape 重复调用免分配。（内部状态，勿改）
        self.plan_workspace = None
        # 跨流安全护栏状态：首个算子调用的发起流；跨流调用时
        # 由 sync_transport_stream 做事件 fork/join 串行化
        self.transport_stream = None
        # 由 moonep_plan 在规划后填充为 (S, K)，prefetch/reduce 推导窗口布局依赖它
        self.last_plan_shape = None
        # barrier 作用域：EP 子组时由 create_moonep_context(group=...) 设置，缺省 WORLD
        self.group = None
        # 多权重组布局守卫：prefetch/reduce 组数必须一致，否则发布/staging 区会重叠
        # （内部状态，勿改）
        self.layout_groups = None
        # 多槽窗口池层数：首个 create_weight_view 定型（缺省 1 = 单槽），
        # 后续调用须一致；reduce 容量校验按它推导（内部状态，勿改）
        self.window_slots = None
        # 通信域分离契约：ctx 创建时已校验一次；首个算子调用再复查一次，
        # 覆盖"ctx 先建、ElasticBuffer 后建"的顺序（内部状态，勿改）
        self.eb_scan_done = False
        # plan 的延迟溢出检查载荷 (cu_seqlens, num_rows)：plan 返回路径零同步，
        # 不变量检查延迟到 prefetch/reduce 尾部（sym_barrier 后流已排空，
        # .item() 微秒级）执行（内部状态，勿改）
        self.pending_overflow = None


def hccl_buffer_base(context):
    """本 rank HCCL 对称缓冲区基址（SymBuffer 的 commBufferAddrs[me]）。"""
    comm = hccl.get_comm_handle(context.group_name)
    buf, _ = hccl.get_hccl_buffer(comm)
    return buf


def hccl_buffer_size(context):
    """本 rank HCCL 对称缓冲区字节数。"""
    comm = hccl.get_comm_handle(context.group_name)
    _, size = hccl.get_hccl_buffer(comm)
    return size


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
    """延迟执行的 plan 溢出检查（plan 返回路径零同步的代价：报警点后移）。

    plan 发射后不排空流——尾部 D2H（.cpu()/.item()）会把 host 卡到流上
    全部在飞工作完成，阻塞紧随的 dispatch 发射（多层训练实测）。
    溢出不变量改由本函数检查：prefetch/reduce 尾部调用（sym_barrier 已
    同步发起流，.item() 微秒级）。无溢出路径语义不变——
    **链路内算子未报错 ⟹ 所有已执行 plan 的 cu_seqlens 尾值 == num_rows**
    （写前保护仍保证不越界写；极端倾斜的报错从 plan 内移到同层 prefetch
    /reduce 尾部，约晚一个算子边界）。
    """
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
            "当前版本拒绝继续，请反馈算子侧（槽位感知贪心在规划中）"
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


def scalar_read_fence_pre(event_id):
    """清单加载护栏·前置（S→MTE2）：与 busy_wait_flag 同款完整握手的前半。"""
    vec_sync_notify(PIPE.S, PIPE.MTE2, event_id)
    vec_sync_wait(PIPE.S, PIPE.MTE2, event_id)


def scalar_read_fence_post(event_id):
    """清单加载护栏·后置（MTE2→S）：标量读/坐标读前等拷贝落地。

    以运行时标量驱动控制流（cannbotdsl.range 界/运行时 if/下标/tile 坐标）
    前必须按 pre→拷贝→post 完整握手——否则标量单元可能读到 mem_copy 尚未
    落地的 UB 内容（非确定性错；实测 cnt=0 致整轮直推静默跳过）。
    """
    vec_sync_notify(PIPE.MTE2, PIPE.S, event_id)
    vec_sync_wait(PIPE.MTE2, PIPE.S, event_id)


def busy_wait_flag(view, gm_epoch, event_id, ub_flag=None, ub_epoch=None):
    """轮询 view 处 flag >= gm_epoch（epoch 单调化阈值，进程内复入不被陈旧 flag 误唤醒）。

    plan 的 ready flag 与 reduce_grad 的 staging flag 共用本握手原语。
    ub_flag/ub_epoch 缺省现配；recv 侧按组轮询时由调用方共享传入，
    避免每实例一对 UB 缓冲耗尽 VEC buf_id。
    """
    if ub_flag is None:
        ub_flag = Buffer(MemLoc.UB, (1,), dtypes.int32)
        ub_epoch = Buffer(MemLoc.UB, (1,), dtypes.int32)

    def _check():
        vec_sync_notify(PIPE.S, PIPE.MTE2, event_id)
        vec_sync_wait(PIPE.S, PIPE.MTE2, event_id)
        mem_copy(ub_flag, view)
        mem_copy(ub_epoch, gm_epoch)
        vec_sync_notify(PIPE.MTE2, PIPE.S, event_id)
        vec_sync_wait(PIPE.MTE2, PIPE.S, event_id)
        return ub_flag[dtypes.int64(0)] >= ub_epoch[dtypes.int64(0)]

    vec_busy_wait(_check)


# ---------------------------------------------------------------------------
# 对称堆自旋 barrier
# ---------------------------------------------------------------------------
# dist.barrier 落到 HCCL SoleMesh 4B allreduce，高并发通信负载下存在对称
# 死锁风险（多 rank 永久挂起）。此处用与数据面相同的 put/flag 通道做 rank
# 对齐：flag 与数据同通道保序，kernel 返回即全组本轮流上工作全部完成，
# 不经过 HCCL 集合通信路径。
_BAR_EVENT_ID = 5  # plan=7、reduce=6、barrier=5
_BAR_EPOCH = 0  # 跨算子共享的单调 barrier epoch（prefetch/reduce 共用一套 flag）
_BAR_PROGRAMS = {}
_BAR_LOCK = threading.Lock()
_BAR_ME = 0  # 组内 rank（编译锁内写入，trace 期烘焙）
_BAR_R = 1  # 组内 rank_size（进编译缓存键）


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
        ub_xfer = Buffer(MemLoc.UB, (512,), dtypes.uint8)
        xfer = create_transfer_context(ctx, ub_xfer, role=TransferRole.VEC)
        for r in range_constexpr(world_size):
            if const_expr(r != me):
                xfer.put(r, tile_slice(flags, (1,), (me,)).peer(r), gm_epoch)
        for r in range_constexpr(world_size):
            if const_expr(r != me):
                xfer.flush(r)
        for r in range_constexpr(world_size):
            if const_expr(r != me):
                busy_wait_flag(tile_slice(flags, (1,), (r,)), gm_epoch, _BAR_EVENT_ID)


@host
def _barrier_host(gm_epoch, ctx: StructRef[CommContext]):
    _MoonepBarrierKernel(_BAR_ME, _BAR_R)[1](gm_epoch, ctx)


def barrier_tail(xfer, ctx, gm_epoch, me, world_size, ub_flag, ub_epoch):
    """kernel 内联的 barrier 尾段：flag 广播 + 收齐（与 _MoonepBarrierKernel 同协议）。

    供算子 kernel 融合发射使用（省一次独立 barrier launch，~550µs/次的发射器
    host 开销）。语义与独立 sym_barrier 等价：本 kernel 完成即全组各 peer 的
    epoch flag 到达（flag 与数据同通道保序——peer 的 flag 到达意味着其发往
    本 rank 的数据已落地）。ub_flag/ub_epoch 由调用方在 kernel 入口分配一次
    传入（VEC buf_id 预算）。
    """
    flags = SymBuffer(
        ctx,
        shape=(MAX_R,),
        dtype=dtypes.int32,
        rank_size=world_size,
        base_offset=BAR_FLAGS_OFF,
    )
    for r in range_constexpr(world_size):
        if const_expr(r != me):
            xfer.put(r, tile_slice(flags, (1,), (me,)).peer(r), gm_epoch)
    for r in range_constexpr(world_size):
        if const_expr(r != me):
            xfer.flush(r)
    for r in range_constexpr(world_size):
        if const_expr(r != me):
            busy_wait_flag(
                tile_slice(flags, (1,), (r,)),
                gm_epoch,
                _BAR_EVENT_ID,
                ub_flag,
                ub_epoch,
            )


def next_barrier_epoch():
    """取下一个 barrier epoch（跨算子单调；融合发射的 kernel 尾段共用本计数器）。"""
    global _BAR_EPOCH
    _BAR_EPOCH += 1
    return _BAR_EPOCH


def _get_barrier_program(world_size, me):
    """(world_size, me) 键控的 barrier program（烘焙与编译在锁内原子完成）。"""
    key = (world_size, me)
    with _BAR_LOCK:
        prog = _BAR_PROGRAMS.get(key)
        if prog is None:
            global _BAR_ME, _BAR_R
            _BAR_ME, _BAR_R = me, world_size
            prog = cannbotdsl.compile(
                _barrier_host,
                TensorSpec((1,), dtypes.int32),
                StructRef[CommContext],
            )
            _BAR_PROGRAMS[key] = prog
    return prog


def sym_barrier(context):
    """对称堆自旋 barrier：替代 dist.barrier（绕开 HCCL SoleMesh 集合通信死锁）。

    语义与同流 dist.barrier 等价且更强：返回时全组 rank 本轮此前流上工作全部
    完成（kernel 内自旋收齐各 peer 的 epoch flag，flag 与数据同通道保序）。
    epoch 跨算子单调递增，陈旧 flag 不会提前放行。
    """
    global _BAR_EPOCH
    _BAR_EPOCH += 1
    gm_epoch = torch.full((1,), _BAR_EPOCH, dtype=torch.int32, device="npu")
    _get_barrier_program(context.rank_size, context.rank)(gm_epoch, context.ctx_tensor)
    # 等待范围为发起流而非整卡：barrier kernel 在发起流上，等本流完成即意味着
    # 全组本轮握手收齐。全设备 synchronize 会把无关流上的在飞工作也等进来，
    # 封死跨层/跨流流水（多层训练里每层 reduce/组间串行都走本 barrier）。
    torch.npu.current_stream().synchronize()


def sync_transport_stream(context):
    """跨流安全护栏：算子握手状态机（epoch flag 轮询 + staging
    槽位复用）要求调用严格按发起序执行。

    - 首次调用：记录发起流；
    - 同流调用：快路径，仅一次比较，零开销；
    - 跨流调用（激活重算等 autograd 场景）：在旧流上记录事件、新流等待——
      事件 fork/join 把调用序列串行化，握手语义保持成立（替代框架侧钉流）。

    覆盖范围：moonep 三算子（plan/prefetch/reduce）。dispatch/combine 为
    moe_ep 原生算子，不在本护栏内——框架侧仍需同流发起或重算回放。
    """
    cur = torch.npu.current_stream()
    ts = context.transport_stream
    if ts is None:
        context.transport_stream = cur
        return
    if ts == cur:
        return
    ev = torch.npu.Event()
    ev.record(ts)
    cur.wait_event(ev)
    context.transport_stream = cur


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
