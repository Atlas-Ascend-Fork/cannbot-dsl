# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""prefetch_weight：home 把副本专家权重（+ mxfp8 scale）直推到各消费 rank。

- 输入为 2 组或 3 组权重，**按同序**传入 local_weights；每组行宽直接由
  ``local_weights[g]`` 尾维展平推断（N-D ``[epn, F, D]`` 直入），一次
  launch 推完全部组；kernel 是单一入口（固定 6 槽 = 3 权重 + 3 scale，
  ngroups 编译期裁剪）。
- **mxfp8 数据面（双向硬契约：scales ⟺ fp8）**：fp8 权重
  （e4m3fn/e5m2）时必须同序传 ``local_scales``（每组 ``[epn, ...]``
  连续 float8_e8m0fnu，行宽 Sg = 每专家 scale 字节数）；bf16 权重不传
  也不可传 scale。scale 与权重共用
  全部控制面（同一张 experts_to_copy 清单、同一次 SIMD 扫描、同一 slot
  轮转、同一逐 peer barrier、同 per-expert 连续 DMA）；行/列存储方向不进
  契约（"每专家连续字节行"抽象吸收，消费端量化 GEMM 按同布局读视图）。
  Sg 须满足 ``Sg % SCALE_ALIGN_BYTES == 0``（E3，不足 pad 到 32B 倍数）。
  dtype 终态：权重 dtype 三选（bf16/e4m3fn/e5m2，uint8 退役）、
  scale dtype 固定 float8_e8m0fnu（OCP MX 标准）；kernel 侧 1B 家族
  统一以 uint8 视图入 TensorSpec（字节语义不变）。
- 权重视图由 prefetch_weight 按 local_weights 的形状创建（并在同
  (sizes, slot, slots) 上复用）后以 ``(weights, scales)`` 裸二元组返回
  ：
  前 epn 行是真实本地专家，后 epn 行是预取槽，视图位于 HCCL 对称堆的
  发布区（scale 组跟同 slot 窗口内权重组之后前缀和排布）。调用方设置
  context.last_plan_shape 后调用。
- 发布窗口是一维 uint8 的**扁平**对称堆视图：每组以“本组一行”为瓦片切分，
  组偏移的**行号**由编译期 sizes 在 kernel 内算出（tile_slice 的坐标是瓦片
  序号，不是元素偏移），运行期只传 slot 序号；各组行宽可以不同，组间按
  sizes 前缀和紧密排布、无需补齐（权重字节宽进编译键：bf16 2B / fp8 1B）。
- 控制面在算子内：rank 从 ``CommContext`` 读取（不在编译键里，同组全 rank
  共用一份产物）、偏移在 kernel 内计算、收尾 barrier 内联在推送 kernel 中
  （逐 peer 通道：推本轮 epoch flag 后自旋收齐，flag 与数据同通道保序）。
- 副本清单不落 GM、不做 H2D：kernel 把 etc[peer] 整行搬进 UB，按设备侧
  get_vf_len() 查到的 VL 换算 lane 数（int32 下 64 lane），向量比较
  （lo <= e < hi）生成本地行号，命中写行号、未命中写 -1，随后按 lane 顺序
  读清单发起 put + flush；尾块由 mask_counter 屏蔽。
- block 与 peer 是多对一分配、一对一访问：peer 按 round-robin 分给
  min(R-1, resident_blocks()) 个 block，每个 peer 通道只由一个 block 写和
  flush，避免多 block 并发访问同一通道。
- 同步模式在 kernel 返回后等发起流，保证全组本轮数据落地；async_finish=True
  时只记录发起流事件、不等待（跨 rank 保序仍由 kernel 内 barrier 保证），由
  调用方 wait_event 汇合。

典型用法：

- 三组（gate/up/down）bf16：``(weights, scales), ev = prefetch_weight(ctx,
  etc, (gate, up, down), epn)``——``weights`` 三组 ``[2*epn, W]``、
  ``scales == ()``；
- 交织型：训练侧把 W_gate 与 W_up 融合成 W13 = [gate; up]（每专家一行 2W，
  gate 在前），一次 GMM 算完交给 SITU，省去消费端 concat。传
  ``(W13, down)``，视图按两者的行宽 (2W, W) 自动创建；每个专家的 W13 以
  一条 2W 连续 DMA 搬完，两组槽区总字节数与三组布局相同（2W + W = 3W）。
- mxfp8（e4m3fn/e5m2 + ue8m0 scale）：``prefetch_weight(ctx, etc, (w13_fp8,
  w2_fp8), epn, local_scales=(s13, s2))``——scale 视图 ``[2*epn, Sg]``
  float8_e8m0fnu 与权重视图
  同构（前 epn 行 home、后 epn 行预取槽），量化 GEMM 按
  ``(weights, scales)`` 成对消费。
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
    BAR_FLAGS_OFF,
    HEAP_BASE,
    MAX_R,
    ROWS_PER_EXPERT,
    SCALE_ALIGN_BYTES,
    XFER_WORKSPACE_BYTES,
    _BAR_EVENT_ID,
    busy_wait_flag,
    check_heap_capacity,
    next_barrier_epoch,
    finish_op,
    hccl_buffer_base,
    lcm_slot_stride,
    prefetch_weight_off,
    reduce_stg_off,
    resident_blocks,
    slot_stride_bytes,
    sync_transport_stream,
)
from npu_blob import npu_from_blob

_MAX_PUSH_BLOCKS = 64
_PUSH_EVENT_ID = 4
_MAX_WEIGHT_GROUPS = 3  # 权重组槽上限（2 组时第 3 槽不参与推送）
_MAX_SCALE_GROUPS = 3  # scale 组槽上限（fp8 形态每组权重配一个 scale 组）
_MAX_GROUPS = _MAX_WEIGHT_GROUPS + _MAX_SCALE_GROUPS  # kernel 入口固定 6 张量槽
_MIN_GROUPS = 2  # 支持的权重组数下限
_WEIGHT_RANK = 2  # 权重/视图张量都是二维 (行, 行宽)
_BITS_PER_BYTE = 8

_WEIGHT_DTYPES = (
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)  # E1：权重 dtype 白名单终态（bf16 / e4m3fn / e5m2，uint8 退役）


def _is_fp8_weight(dtype):
    """权重 dtype 是否 fp8（mxfp8 数据面——E2 双向契约的判定基）。"""
    return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def _push_layout(rank_size):
    n_rem = max(rank_size - 1, 0)
    nblk = min(max(n_rem, 1), resident_blocks(_MAX_PUSH_BLOCKS))
    per_blk = (n_rem + nblk - 1) // nblk if n_rem > 0 else 1
    return nblk, per_blk


@kernel
class _MoonepPrefetchPushKernel:
    """按组推送：窗口是一维 uint8 对称堆视图，每组以“本组一行”为瓦片。

    控制面在算子内：rank 取自 ``CommContext``（对应 HCCL 的 GetRankId），组
    起始行号与槽步长由编译期 ``sizes`` 计算，收尾 barrier 内联在推送 kernel
    里（flag 与数据同通道保序）。单一入口固定 6 个张量槽（3 权重 + 3
    scale），``ngroups``（权重组数 + scale 组数，2..6）在 trace 阶段裁剪实际
    推送组数；缺组槽由 host 填入重复值、不读取（现状 2 组 pad 逻辑的自然
    延拓）。

    组宽 ``sizes``（元素数）与逐组要素字节宽 ``elem_bytes``（权重 bf16 2B /
    fp8 1B；scale 恒 1B）均为编译期常量（tile_slice 的瓦片尺寸必须静态）；
    tile_slice 的坐标是瓦片序号，因此组基址用“本组一行字节宽”为单位的行号
    表示。各槽区按 (sizes × elem_bytes) 字节前缀和紧密排布，组宽可以不同
    （交织型 W13=2W、down=W；scale 组 Sg 与权重组 Wg 独立）。
    """

    def __init__(
        self,
        rank_size,
        epn,
        sizes,
        elem_bytes,
        ngroups,
        window_bytes,
        nblk,
        per_block,
    ):
        self._r = int(rank_size)
        self._epn = int(epn)
        # 定长 6（缺省槽复用末组宽度/字节宽——该槽不参与推送）
        self._sizes = tuple(int(s) for s in sizes)
        self._elem_bytes = tuple(int(b) for b in elem_bytes)
        self._ngroups = int(ngroups)
        # 一行字节数（逐组：权重 sizes×w_bytes / scale sizes×1B）
        self._row_bytes = tuple(s * b for s, b in zip(self._sizes, self._elem_bytes))
        # 组起始行号 / 槽步长：sizes/elem_bytes 是编译期常量，这里一并在
        # 编译期算好，不再由 host 传偏移（host 只传运行期的 slot 序号）。
        # 组基址按**字节**前缀和（权重与 scale 混排时字节宽不同）；槽步长
        # 走 slot_stride_bytes 的 LCM 对齐公式（与 host 视图/容量公式唯一
        # 真源——步长须被每个组的行宽整除，dst 行号寻址才合法）。
        actual = self._row_bytes[: self._ngroups]
        slot_bytes = lcm_slot_stride(self._epn, actual)
        group_rows, slot_rows = [], []
        base_bytes = 0
        for g in range(self._ngroups):
            row_bytes = self._row_bytes[g]
            if base_bytes % row_bytes or slot_bytes % row_bytes:
                raise ValueError(
                    f"第 {g} 组偏移 {base_bytes}B / 槽步长 {slot_bytes}B "
                    f"未按行宽 {row_bytes}B 对齐"
                )
            group_rows.append(base_bytes // row_bytes)
            slot_rows.append(slot_bytes // row_bytes)
            base_bytes += row_bytes * ROWS_PER_EXPERT * self._epn
        self._group_rows = tuple(group_rows)
        self._slot_rows = tuple(slot_rows)
        self._window_bytes = int(window_bytes)
        self._nblk = int(nblk)
        self._per_block = int(per_block)

    def __call__(
        self,
        k0,
        k1,
        k2,
        k3,
        k4,
        k5,
        gm_etc,
        slot,
        gm_epoch,
        ktx: StructRef[CommContext],
    ):
        world_size, epn, ngroups = self._r, self._epn, self._ngroups
        sizes, row_bytes = self._sizes, self._row_bytes
        # 槽序：真实组前缀（权重 n_w 个 + scale n_s 个）+ pad 尾置——
        # 前缀推送 g < ngroups 恰覆盖真实组（分族各归各位的交错排布会让
        # pad 权重插进前缀：n_w=2 时被推的是 (w0, w1, w_pad, s0)——w_pad
        # 写入 scale 区、末 scale 组从未推送，A13 探针实证）
        reals = (k0, k1, k2, k3, k4, k5)
        # rank 信息来自算子内的 CommContext（仿 hccl_.GetRankId），不进编译键：
        # 同一份程序被全组 rank 共用。
        me = dtypes.int64(ktx.rankId)
        me32 = dtypes.int32(ktx.rankId)
        wsym = SymBuffer(
            ktx,
            shape=(self._window_bytes,),
            dtype=dtypes.uint8,
            rank_size=world_size,
            base_offset=prefetch_weight_off(),
        )
        flags = SymBuffer(
            ktx,
            shape=(MAX_R,),
            dtype=dtypes.int32,
            rank_size=world_size,
            base_offset=BAR_FLAGS_OFF,
        )
        ub_xfer = Buffer(MemLoc.UB, dtypes.uint8, make_layout((XFER_WORKSPACE_BYTES,)))
        xfer = create_transfer_context(ktx, ub_xfer, role=TransferRole.VEC)
        # 轮询缓冲共享一对（逐 peer 展开时现配会耗尽 VEC buf_id）
        ub_flag = Buffer(MemLoc.UB, dtypes.int32, make_layout((1,)))
        ub_epoch = Buffer(MemLoc.UB, dtypes.int32, make_layout((1,)))

        blk = dtypes.int64(get_block_idx())
        n_rem = dtypes.int64(world_size - 1)
        epn64 = dtypes.int64(epn)
        for peer_slot in range_constexpr(self._per_block):
            peer_id = blk + dtypes.int64(peer_slot * self._nblk)
            if peer_id < n_rem:
                peer = peer_id if peer_id < me else peer_id + dtypes.int64(1)
                ub_etc = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, self._epn)))
                ub_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, self._epn)))
                self._scan_peer(gm_etc, peer, ub_etc, ub_row, me32)

                for i in cannbotdsl.range(epn64):
                    row = dtypes.int64(ub_row[dtypes.int64(0), i])
                    if row >= dtypes.int64(0):
                        # 预取槽行号 = 组起始行号 + slot × 槽步长 + epn + i。
                        # 目标瓦片按 uint8 计（= 本行字节数），源瓦片按源 dtype
                        # 元素计；_put 只从源推导 nbytes，单位不同但字节数一致
                        # （权重 bf16 2B/fp8 1B、scale ue8m0 1B——逐组 elem_bytes
                        # 进编译键保证两侧字节宽恒一致）。
                        slot_row = epn64 + i
                        for g in range_constexpr(_MAX_GROUPS):
                            if g < ngroups:  # 编译期裁剪：缺组槽不发射
                                dst = (
                                    dtypes.int64(self._group_rows[g])
                                    + slot * dtypes.int64(self._slot_rows[g])
                                    + slot_row
                                )
                                xfer.put(
                                    peer,
                                    tile_slice(wsym, (row_bytes[g],), (dst,)).peer(
                                        peer
                                    ),
                                    tile_slice(reals[g], (1, sizes[g]), (row, 0)),
                                )
                xfer.flush(peer)
                # 算子内收尾 barrier（逐 peer 通道）：先把本轮 epoch flag 直推
                # 该 peer，再自旋等对方的 flag。flag 与数据同通道保序，peer 的
                # flag 到达即其发往本 rank 的数据已落地；每个 peer 只由所属
                # block 写和等，避免多 block 并发同一通道。
                xfer.put(peer, tile_slice(flags, (1,), (me,)).peer(peer), gm_epoch)
                xfer.flush(peer)
                busy_wait_flag(
                    tile_slice(flags, (1,), (peer,)),
                    gm_epoch,
                    _BAR_EVENT_ID,
                    ub_flag,
                    ub_epoch,
                )

    @jit
    def _scan_peer(self, gm_etc, peer, ub_etc, ub_row, me32):
        epn = self._epn
        # 矢量寄存器 VL 由设备侧接口查询（字节），按 int32 位宽换算 lane 数。
        vl = (
            get_vf_len()
            * dtypes.int64(_BITS_PER_BYTE)
            // dtypes.int64(dtypes.int32.bits)
        )
        # 步长固定为 1、按 VL 换算偏移，保持 VF 硬件循环可识别
        n_chunks = (dtypes.int64(epn) + vl - 1) // vl
        lo_e = me32 * dtypes.int32(epn)  # 本 rank 负责的专家区间 [lo, hi)
        hi_e = lo_e + dtypes.int32(epn)
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
    rank_size: int
    epn: int
    sizes: tuple  # 6 槽元素宽（权重 3 + scale 3，缺省槽复用末组）
    elem_bytes: tuple  # 6 槽逐组要素字节宽（bf16 2B / fp8·scale 1B）
    ngroups: int  # 实际推送组数（权重组数 + scale 组数）
    window_bytes: int
    nblk: int
    per_block: int


def _make_push_host(cfg):
    @host
    def _push_host(
        k0,
        k1,
        k2,
        k3,
        k4,
        k5,
        gm_etc,
        slot: int,
        gm_epoch,
        ctx: StructRef[CommContext],
    ):
        _MoonepPrefetchPushKernel(
            cfg.rank_size,
            cfg.epn,
            cfg.sizes,
            cfg.elem_bytes,
            cfg.ngroups,
            cfg.window_bytes,
            cfg.nblk,
            cfg.per_block,
        )[cfg.nblk](k0, k1, k2, k3, k4, k5, gm_etc, slot, gm_epoch, ctx)

    return _push_host


_PUSH_PROGRAMS = {}
_PUSH_LOCK = threading.Lock()


def _get_push_program(
    epn,
    weight_sizes,
    scale_sizes,
    weight_bytes,
    window_bytes,
    rank_size,
):
    """按 (epn, sizes, elem_bytes, ngroups, window_bytes, rank_size) 编译/复用。

    sizes/elem_bytes 全进编译键：tile_slice 的瓦片尺寸必须静态、
    TensorSpec 的 dtype/stride 须与实参一致（bf16/uint8 逐槽可异——权重
    槽与 scale 槽 dtype 不同；1B 家族在 kernel 侧统一 uint8 视图，字节
    语义不变——dtype 终态/U7 路径）；rank 不进键——kernel 从 CommContext
    读取，同一组内全 rank 共用一份产物。

    6 槽布局为**真实组前缀 + pad 尾置**：槽 0..ngroups-1 是真实组（权重
    n_w 个在前、scale n_s 个随后——与 host 视图布局同序），槽 ngroups..5
    是 pad（权重 pad 复用末位权重、scale pad 复用末位 scale；无 scale 时
    pad 全用末位权重）。kernel 的前缀推送 ``g < ngroups`` 因此恰覆盖真实
    组。**分族各归各位的交错排布（[W0, W1, W_pad, S0, ...]）会让 pad 权重
    插进前缀**：n_w=2 时被推的是 (w0, w1, w_pad, s0)——w_pad 写入 scale
    区、末 scale 组从未推送（A13 探针实证：scale 槽行保持 255 尾置初值）。
    """
    n_w = len(weight_sizes)
    n_s = len(scale_sizes)
    ngroups = n_w + n_s
    real_sizes = tuple(weight_sizes) + tuple(scale_sizes)
    real_elem = (weight_bytes,) * n_w + (1,) * n_s
    if n_s:
        pad_sizes = (weight_sizes[-1],) * (_MAX_WEIGHT_GROUPS - n_w) + (
            scale_sizes[-1],
        ) * (_MAX_SCALE_GROUPS - n_s)
        pad_elem = (weight_bytes,) * (_MAX_WEIGHT_GROUPS - n_w) + (1,) * (
            _MAX_SCALE_GROUPS - n_s
        )
    else:
        # 无 scale：pad 全部复用末组权重（v1 行为等价）
        pad_sizes = (weight_sizes[-1],) * (_MAX_GROUPS - n_w)
        pad_elem = (weight_bytes,) * (_MAX_GROUPS - n_w)
    sizes = real_sizes + pad_sizes
    elem6 = real_elem + pad_elem
    nblk, per_blk = _push_layout(rank_size)
    cfg = _PushConfig(
        rank_size,
        epn,
        tuple(sizes),
        tuple(elem6),
        ngroups,
        int(window_bytes),
        nblk,
        per_blk,
    )
    with _PUSH_LOCK:
        prog = _PUSH_PROGRAMS.get(cfg)
        if prog is None:
            # 逐槽 dtype：按 elem_bytes（bf16 2B / 1B 家族 uint8——权重
            # fp8 与 scale ue8m0 在 kernel 侧均为 1B 字节搬运，统一 uint8
            # 收敛；host 侧以 view(torch.uint8) 传入）
            specs = [
                TensorSpec((epn, s), _dsl_dtype(b))
                for s, b in zip(cfg.sizes, cfg.elem_bytes)
            ]
            specs.append(TensorSpec((rank_size, epn), dtypes.int32))
            specs.append(dtypes.int64)  # slot
            specs.append(TensorSpec((1,), dtypes.int32))  # barrier epoch
            specs.append(StructRef[CommContext])
            prog = cannbotdsl.compile(_make_push_host(cfg), *specs)
            _PUSH_PROGRAMS[cfg] = prog
    return prog


def _dsl_dtype(elem_bytes):
    """要素字节宽 → DSL dtype（2B=bf16、1B=uint8——kernel 侧 1B 家族
    统一字节语义：fp8 权重与 ue8m0 scale 均以 uint8 视图入 TensorSpec）。"""
    if elem_bytes == 2:
        return dtypes.bfloat16
    return dtypes.uint8


def _torch_dtype_str(dtype):
    """torch dtype → npu_from_blob 的 dtype 字符串。"""
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.uint8:
        return "uint8"
    if dtype == torch.float8_e4m3fn:
        return "e4m3fn"
    if dtype == torch.float8_e5m2:
        return "e5m2"
    if dtype == torch.float8_e8m0fnu:
        return "e8m0fnu"
    if dtype == torch.float32:
        return "fp32"
    raise TypeError(f"不支持的视图 dtype {dtype}")


def _as_sizes(sizes):
    """归一化每组件宽（元素数），要求为正。"""
    sizes = tuple(int(s) for s in sizes)
    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError(f"每组行宽必须为正，实得 {sizes}")
    return sizes


def _freeze_layout(context, widths, scale_widths, *, what):
    """组结构定型守卫（E4）：组数与各组行宽首调定型，此后不可变。

    宽度也须一致：梯度槽区基址按实际 widths 计算（紧跟权重发布区），
    widths 缩小会使槽区起点落进发布区内部——发布/槽区重叠、数据互踩。
    weight/scale/grad 三侧共用本守卫（layout_widths 全链一份）。
    """
    if context.layout_groups is None:
        context.layout_groups = len(widths)
    elif context.layout_groups != len(widths):
        raise RuntimeError(
            f"窗口组数不一致（已有 {context.layout_groups}，本次 {len(widths)}），"
            "会导致发布区重叠"
        )
    if context.layout_widths is None:
        context.layout_widths = tuple(int(w) for w in widths)
    elif tuple(int(w) for w in widths) != context.layout_widths:
        raise RuntimeError(
            f"窗口组宽不一致（已有 {context.layout_widths}，本次 {tuple(widths)}）"
            f"（{what}）：宽度变化会使梯度槽区基址错位、发布/槽区重叠"
        )
    scale_widths = tuple(int(s) for s in scale_widths) if scale_widths else ()
    if context.layout_scale_widths is None:
        context.layout_scale_widths = scale_widths
    elif scale_widths != context.layout_scale_widths:
        raise RuntimeError(
            f"scale 组宽不一致（已有 {context.layout_scale_widths}，"
            f"本次 {scale_widths}）（{what}）：scale 随 slot 窗口轮转，"
            "宽度变化会使窗口步长错位"
        )


def _create_weight_views(context, epn, sizes, scale_sizes, *, slot=0, slots=1, dtype):
    """按 sizes 顺序创建（或复用）N 组权重视图，返回同序元组（内部接口）。

    组 g 的视图形状 ``[2*epn, sizes[g]]``（dtype），槽内布局：权重组按
    sizes 前缀和在前、scale 组紧随其后（前缀和见 _create_scale_views）；
    槽步长按 (ΣWg×w_bytes + ΣSg)×2*epn 计（scale 在同窗口内随权重轮转）。
    前 epn 行是真实本地专家、后 epn 行是预取槽。仅供 prefetch_weight
    内部使用。
    """
    sizes = _as_sizes(sizes)
    scale_sizes = tuple(int(s) for s in scale_sizes) if scale_sizes else ()
    weight_bytes = 2 if dtype == torch.bfloat16 else 1
    _freeze_layout(context, sizes, scale_sizes, what="prefetch 权重视图")
    if context.last_plan_shape is None:
        raise RuntimeError(
            "创建权重视图前请设置 context.last_plan_shape（输入形状与窗口布局）"
        )
    if slot < 0 or slot >= slots:
        raise ValueError(f"slot={slot} 超出槽位数 {slots}")
    if context.window_slots is None:
        context.window_slots = slots
    elif context.window_slots != slots:
        raise RuntimeError(
            f"窗口槽位数不一致（已有 {context.window_slots}，本次 {slots}）"
        )
    key = ("views", int(epn), sizes, int(slot), int(slots), str(dtype))
    cached = context.weight_views.get(key)
    if cached is not None:
        return cached
    check_heap_capacity(
        context,
        HEAP_BASE
        + reduce_stg_off(
            epn,
            slots=slots,
            widths=sizes,
            scale_widths=scale_sizes or None,
            weight_bytes=weight_bytes,
        ),
        "对称堆布局（多组权重发布区）",
    )
    slot_bytes = slot_stride_bytes(epn, sizes, scale_sizes or None, weight_bytes)
    addr = (
        hccl_buffer_base(context)
        + HEAP_BASE
        + prefetch_weight_off()
        + slot * slot_bytes
    )
    views = []
    for w in sizes:
        views.append(
            npu_from_blob(addr, (ROWS_PER_EXPERT * epn, w), _torch_dtype_str(dtype))
        )
        addr += ROWS_PER_EXPERT * epn * w * weight_bytes
    views = tuple(views)
    context.weight_views[key] = views
    return views


def _create_scale_views(
    context, epn, weight_sizes, scale_sizes, *, slot=0, slots=1, weight_bytes=2
):
    """按 scale_sizes 顺序创建（或复用）scale 视图（内部接口，e8m0fnu）。

    组 g 的视图 ``[2*epn, scale_sizes[g]]`` float8_e8m0fnu（OCP MX 标准
    ue8m0 编码，1B）；放置：同 slot 窗口内、
    跟在该槽全部权重组之后（字节前缀和）——scale 必须与权重同槽轮转
    （L 层 GEMM 读 slot s 的 (weights, scales) 对、L+1 层 prefetch 写
    slot s+1；若 scale 放全局单窗口会在 L 层还在读时被覆写）。行序与
    权重视图同构：前 epn 行 home、后 epn 行预取槽。
    """
    scale_sizes = _as_sizes(scale_sizes)
    _freeze_layout(context, weight_sizes, scale_sizes, what="prefetch scale 视图")
    weight_bytes_total = sum(int(w) for w in weight_sizes) * weight_bytes
    key = ("scale_views", int(epn), scale_sizes, int(slot), int(slots))
    cached = context.weight_views.get(key)
    if cached is not None:
        return cached
    slot_bytes = slot_stride_bytes(epn, weight_sizes, scale_sizes, weight_bytes)
    addr = (
        hccl_buffer_base(context)
        + HEAP_BASE
        + prefetch_weight_off()
        + slot * slot_bytes
        + weight_bytes_total * ROWS_PER_EXPERT * epn
    )
    views = []
    for s in scale_sizes:
        views.append(npu_from_blob(addr, (ROWS_PER_EXPERT * epn, s), "e8m0fnu"))
        addr += ROWS_PER_EXPERT * epn * s
    views = tuple(views)
    context.weight_views[key] = views
    return views


def _sync_transport(context, local_weights, views, epn):
    """home 行灌入（真实参数 → 视图 home 段；同址零拷贝跳过；内部步骤）。

    1B 家族（fp8 权重 / ue8m0 scale）经 ``view(torch.uint8)`` 按位拷——
    字节语义与后端实现无关（fp8 原生 copy 依赖后端算子覆盖度）；bf16
    直拷。
    """
    for real_w, view_w in zip(local_weights, views):
        if real_w.data_ptr() == view_w[:epn].data_ptr():
            continue
        if real_w.dtype.itemsize == 1:
            view_w[:epn].view(torch.uint8).copy_(real_w.view(torch.uint8))
        else:
            view_w[:epn] = real_w


def _window_bytes(context, epn, sizes, scale_sizes, weight_bytes):
    """发布窗口总字节数（slots × LCM 对齐步长——与 kernel SymBuffer 同式）。"""
    slots = context.window_slots or 1
    return slots * slot_stride_bytes(epn, sizes, scale_sizes or None, weight_bytes)


def _normalize_groups(local_weights, epn, context=None):
    """归一化组列表并按形状推断每组行宽（任意尾维 N-D，内部展平）。

    每组 ``[epn, Wg]`` 或任意尾维 ``[epn, ...]``（如 ``[epn, F, D]``——GMM
    权重的自然形态；行宽 = 尾维展平宽 numel/epn）；组数限 2 或 3（kernel
    入口按组数固定）。返回 (flat_views, sizes, weight_dtype)——flat 与原
    张量同存储。

    dtype 守卫：全组同 dtype 且 ∈ (bf16, e4m3fn, e5m2)；context.weight_dtype
    （MoonepBuffer 构造定型；裸 API 首调冻结）不一致即报错——dtype 进
    编译键与视图 dtype，运行期漂移会使 push 产物与视图错位。
    """
    if isinstance(local_weights, torch.Tensor):
        # 单张量无成功路径（组数恒 1 ∉ {2,3}）——显式报错防
        # tuple(tensor) 行迭代歧义（原包装是只通向报错的死分支）。
        raise TypeError(
            "local_weights 须为 2/3 组 list/tuple（每组 [epn, ...]）——不支持单张量"
        )
    local_weights = tuple(local_weights)
    if len(local_weights) not in (_MIN_GROUPS, _MAX_WEIGHT_GROUPS):
        raise ValueError(
            f"prefetch 支持 2 组或 3 组权重（kernel 入口按组数固定），"
            f"实得 {len(local_weights)} 组"
        )
    flats, sizes = [], []
    dtype = None
    for g, rw in enumerate(local_weights):
        if rw.dim() < 2 or int(rw.shape[0]) != int(epn):
            raise ValueError(
                f"第 {g} 组权重形状应为 (epn, ...) >= 2 维，实得 {tuple(rw.shape)}"
            )
        if not rw.is_contiguous():
            raise ValueError(f"第 {g} 组须连续（展平前置）")
        if dtype is None:
            dtype = rw.dtype
        elif rw.dtype != dtype:
            raise TypeError(
                f"多权重组的 dtype 必须一致（第 {g} 组 {rw.dtype} != {dtype}）"
            )
        w = int(rw.numel() // int(epn))
        flats.append(rw.reshape(int(epn), w))
        sizes.append(w)
    if dtype not in _WEIGHT_DTYPES:
        raise TypeError(
            "权重 dtype 仅支持 bf16 / float8_e4m3fn / float8_e5m2"
            f"（dtype 终态，uint8 已退役），got {dtype}"
        )
    if context is not None:
        frozen = getattr(context, "weight_dtype", None)
        if frozen is None:
            context.weight_dtype = dtype  # 裸 API：首调冻结
        elif frozen != dtype:
            raise TypeError(
                f"权重 dtype 与定型值不符（构造/首调 {frozen}，本次 {dtype}）"
                "——dtype 进编译键与视图，不可运行期漂移；请以一致的 dtype "
                "重建或转换权重"
            )
    return tuple(flats), _as_sizes(sizes), dtype


def _normalize_scales(local_scales, epn, num_weight_groups, weight_dtype):
    """归一化 scale 组列表 → (flat_views, scale_sizes)；E2/E3 守卫。

    双向硬契约（scales ⟺ fp8）：fp8 权重（e4m3fn/e5m2）⇒ 必须给出
    local_scales（mxfp8 必有 scale）；给出 scales ⇒ 权重须为 fp8
    （bf16 不传也不可传）。每组 ``[epn, ...]`` 连续 float8_e8m0fnu
    （OCP MX 标准 ue8m0 编码，1B；N-D 展平，行宽 = 每专家 scale 字节数
    Sg）；对齐约束：Sg 须为 SCALE_ALIGN_BYTES（32B）倍数。
    """
    if local_scales is None:
        if _is_fp8_weight(weight_dtype):
            raise ValueError(
                "mxfp8 契约：fp8 权重（e4m3fn/e5m2）⇒ 必须传 local_scales"
                "（ue8m0 block scale 与权重同推；bf16 形态不传 scale）"
            )
        return (), ()
    if not _is_fp8_weight(weight_dtype):
        raise ValueError(
            "mxfp8 契约：local_scales 仅支持 fp8 权重"
            f"（e4m3fn/e5m2），实得权重 dtype {weight_dtype}"
        )
    if isinstance(local_scales, torch.Tensor):
        # 单 Tensor 无成功路径（组数恒 1 ≠ num_weight_groups）
        raise TypeError(
            "local_scales 须为与权重组同长度的 list/tuple"
            f"（每组 [epn, ...] float8_e8m0fnu，共 {num_weight_groups} 组）"
            "——不支持单张量"
        )
    local_scales = tuple(local_scales)
    if len(local_scales) != num_weight_groups:
        raise ValueError(
            f"local_scales 长度须等于权重组数 {num_weight_groups}（同组同序"
            f"——量化 GEMM 按 (weights, scales) 成对消费），"
            f"实得 {len(local_scales)} 组"
        )
    flats, sizes = [], []
    for g, sc in enumerate(local_scales):
        if sc.dim() < 2 or int(sc.shape[0]) != int(epn):
            raise ValueError(
                f"第 {g} 组 scale 形状应为 (epn, ...) >= 2 维，实得 {tuple(sc.shape)}"
            )
        if not sc.is_contiguous():
            raise ValueError(f"第 {g} 组 scale 须连续（展平前置）")
        if sc.dtype != torch.float8_e8m0fnu:
            raise TypeError(
                f"第 {g} 组 scale 须为 float8_e8m0fnu（MX 标准 ue8m0 编码）"
                f"，实得 {sc.dtype}"
            )
        s = int(sc.numel() // int(epn))
        if s <= 0 or s % SCALE_ALIGN_BYTES:
            padded = (
                (s + SCALE_ALIGN_BYTES - 1) // SCALE_ALIGN_BYTES * SCALE_ALIGN_BYTES
            )
            raise ValueError(
                f"第 {g} 组 scale 行宽 Sg={s} 须为正且是 {SCALE_ALIGN_BYTES}B"
                f" 倍数（对齐约束）；请 pad 到 {padded}B"
                f"（消费端按 padded 行宽解）"
            )
        flats.append(sc.reshape(int(epn), s))
        sizes.append(s)
    return tuple(flats), tuple(sizes)


def prefetch_weight(
    context,
    experts_to_copy,
    local_weights,
    epn,
    *,
    local_scales=None,
    slot=0,
    slots=1,
    async_finish=False,
):
    """按同序推送 N 组权重（+ 可选 scale 组），返回 ``((weights, scales), event)``。

    每组行宽由 ``local_weights[g]`` 尾维展平推断；视图按 ``(sizes, slot,
    slots)`` 缓存在 context 上，重复调用直接复用。交织型 W13+down 的 W13
    每专家一行连续 2W（gate 在前、up 在后），一条 DMA 搬完，消费端无需再
    concat。mxfp8 形态（fp8 权重）必须同序传 ``local_scales``，scale
    与权重共用全部控制面一次 launch 推完（行/列方向不感知——每专家连续
    字节行抽象）。

    Returns:
        **变长返回**：``async_finish=False``（缺省）只返回
        ``(weights, scales)`` 二元组单值（weights = 各组 ``[2*epn, Wg]``
        视图元组；scales = 各组 ``[2*epn, Sg]`` float8_e8m0fnu 视图元组
        ，未量化时 ``()``——裸二元组，无包装类）；
        ``async_finish=True`` 返回 ``((weights, scales), event)`` 二元组
        ——``event`` 为发起流完成事件（wait_event 汇合后复用缓冲区）。
    """
    flats, sizes, weight_dtype = _normalize_groups(local_weights, epn, context)
    scale_flats, scale_sizes = _normalize_scales(
        local_scales, epn, len(flats), weight_dtype
    )
    weight_bytes = 2 if weight_dtype == torch.bfloat16 else 1
    views = _create_weight_views(
        context,
        epn,
        sizes,
        scale_sizes,
        slot=slot,
        slots=slots,
        dtype=weight_dtype,
    )
    scale_views = (
        _create_scale_views(
            context,
            epn,
            sizes,
            scale_sizes,
            slot=slot,
            slots=slots,
            weight_bytes=weight_bytes,
        )
        if scale_sizes
        else ()
    )
    sync_transport_stream(context)
    _sync_transport(context, flats, views, epn)
    if scale_flats:
        _sync_transport(context, scale_flats, scale_views, epn)
    # 视图创建期一次性哨兵 fill；算子热路径不做每步 fill+屏障。多轮复用的
    # 空槽重置由调用方组合原语（副本段 fill + barrier）显式控制；GEMM 消费
    # 侧只读 cu_seqlens 覆盖段。
    # 入口固定 6 槽：真实组前缀（权重 + scale）+ pad 尾置（复用末组张量、
    # 不发射——与 _get_push_program 的槽序约定一致；pad 不得插入前缀）。
    # kernel 侧 1B 家族（fp8 权重 / ue8m0 scale）统一 uint8 视图入
    # TensorSpec（dtype 终态/U7 路径——字节语义不变，kernel dtype 与
    # bf16/uint8 路径全同）
    n_w = len(flats)
    n_s = len(scale_flats)
    if weight_bytes == 1:
        flats_k = tuple(f.view(torch.uint8) for f in flats)
    else:
        flats_k = flats
    if n_s:
        scale_flats_k = tuple(s.view(torch.uint8) for s in scale_flats)
        reals6 = (*flats_k, *scale_flats_k) + (
            (flats_k[-1],) * (_MAX_WEIGHT_GROUPS - n_w)
            + (scale_flats_k[-1],) * (_MAX_SCALE_GROUPS - n_s)
        )
    else:
        # 无 scale：pad 全部复用末组权重张量（v1 行为等价）
        reals6 = (*flats_k, *flats_k[-1:] * (_MAX_GROUPS - n_w))
    gm_epoch = next_barrier_epoch(context)
    _get_push_program(
        epn,
        sizes,
        scale_sizes,
        weight_bytes,
        _window_bytes(context, epn, sizes, scale_sizes, weight_bytes),
        context.rank_size,
    )(
        *reals6,
        experts_to_copy,
        int(slot),
        gm_epoch,
        context.ctx_tensor,
    )
    event = finish_op(context, async_finish)
    # 变长返回契约：sync 只返回 (weights, scales) 二元组单值；
    # async_finish=True 才返回 ((weights, scales), event) 二元组。
    if async_finish:
        return (views, scale_views), event
    return views, scale_views
