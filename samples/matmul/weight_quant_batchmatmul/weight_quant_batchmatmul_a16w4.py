# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Ascend950 A16W4 ``weight_quant_batchmatmul`` operator.

Structure:
  1. Host-side tiling (bandwidth-cost block solver with device-aware grid)
  2. MIX kernel (AIC:AIV = 1:2): AIV dequant pipeline + AIC MMAD pipeline
     joined by a cross-core L1 zn RingBuffer
  3. Torch-facing ``npu_weight_quant_batchmatmul`` wrapper

Formula: y[M,N] = x[M,K] @ dequant(w[K,N]), where
dequant(w[k,n]) = E2M1(w[k,n]) * 2^(E8M0(scale[k//32,n]) - 127)
(x fp16, w MXFP4 in WeightNZ layout, scale E8M0, y fp16)

Data flow (MIX, two AIV split the N half-panels):
  AIV x2: GM w(WeightNZ fp4 panels) + GM scale(e8m0) --MTE2--> UB ring
          (depth 4) --V--> dequant (vreinterpret_lanes(fp4x2) + 4xvcast
          + vgather_reg/vselect fractal reorder to zn order + e2b scale
          broadcast xvmul) --MTE3--> L1 zn cross-core RingBuffer
          (flat nd allocation + reinterpret "zn" views, depth 2; READY/FREE
          barriers and AIV attribution are derived by the framework counter
          system from tensor accesses)
  AIC: x nd2nz -> L1 ring (depth 2) -> L0A/L0B (depth 1) -> 2xMMAD
       cross-part accumulation -> L0C (depth 1) -> FixPipe -> GM

Tail handling: K/N are zero-padded on host up to the tiling geometry
multiples (see ``_pad_inputs``) so every tile and part runs the full-width
path; the wrapper crops the output back to (M, N).
"""

from cannbotdsl import from_tensor

__all__ = ["npu_weight_quant_batchmatmul"]

import os
import threading

import numpy as np
import torch

from cannbotdsl import (
    MemLoc,
    Tensor,
    copy_options,
    dtypes,
    get_block_idx,
    get_block_num,
    get_subblock_id,
    host,
    idx2crd,
    jit,
    kernel,
    make_layout,
    matmul,
    mem_copy,
    range_constexpr,
    tile_slice,
    vf,
)
from cannbotdsl.buffer import Buffer
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.reg import (
    PackMode,
    RegLayout,
    UnpackMode,
    create_mask,
    vcast,
    vdups,
    vgather_reg,
    vload,
    vload_broadcast,
    vload_unpack,
    vreinterpret,
    vreinterpret_lanes,
    vselect,
    vshl,
    vstore,
    vstore_pack,
    vmem_bar,
    vlt,
    vmul,
    varange,
)


# Integer alignment helper shared by WqbmmTiling and the torch interface
def ceil_div(a: int, b: int) -> int:
    """Integer ceil division used for tile-count calculations."""
    if b == 0:
        return a
    return (a + b - 1) // b


def ceil_align(a: int, b: int) -> int:
    """Round up to the nearest multiple of the alignment."""
    if b == 0:
        return a
    return (a + b - 1) // b * b


# ============================================================================
# 1. Host-side Tiling
# ============================================================================

# 默认几何常量（冻结值，TILING.md §6；实际几何由带宽代价模型选块+投影
# 按 shape 决定，仅作无解回退与预算基准）
BM = 128  # 输出 tile M（每 AIC）—— vec/cube 比仅取决于 1/BM
BN = 128  # 输出 tile N（= 份的 N 全宽）
KB = 256  # 每份 K（UB 份 ≡ L1 份 ≡ L0 事务）
NH = BN // 2  # 每 AIV 的 N 半宽
N1H = NH // 16  # 每 AIV 半区 16 列 panel 数
K1S = KB // 16  # 每份 k 分形数
SG = KB // 32  # 每份 scale 组数
KB8 = KB * 8  # 每份每 panel 字节数
FRACTAL_BYTES = 512  # 16×16 fp16 分形
SLOT_M1_BLOCK = (NH // 16) * FRACTAL_BYTES  # zn slot 内每 m1 块字节数
MAX_GRID = 32  # 超配上限（设备探测失败时的回退值）

# grid 择优：tiles ≤ 64（短 kernel）取设备物理核数——超配块的 co-schedule
# 调度量子在短 kernel 上占主导；tiles > 64 取 MAX_GRID=32 超配——用
# co-schedule 换 tile 均衡，避免大 shape 在物理核数下的双峰慢模式。
_SMALL_GRID_TILES = 64

# 片上预算（950PR，字节）
L0A_BUDGET = 64 * 1024
L0B_BUDGET = 64 * 1024
L0C_BUDGET = 256 * 1024
L1_BUDGET = 512 * 1024
UB_BUDGET = 248 * 1024


# ----------------------------------------------------------------------------
# 基本块 tiling 求解器（带宽代价模型 + cube-bound 选块，A16W4-MX-NZ 配置）
# 流程: 基本块候选枚举（带宽比过滤）→ 满核 5 案例整除/折半选块 →
#       cube-bound 比较序排序 → L1 stepKa/stepKb 枚举定 K 份 → 非满核回退。
# 近似: L1 候选取 kBL1 最大合法解；回退档按带宽比排序 —— 仅影响 stepKb
#       细选，不影响 baseM/baseN 选块。
# ----------------------------------------------------------------------------
_BB_BLOCK_CUBE = 16
_BB_BASE_MN_LIMIT = 256 * 256
_BB_BASE_MK_LIMIT = 256 * 64
_BB_BASE_BLOCK_MAX = 512
_BB_A16W4_START_M = 48  # BASE_BLOCK_MIN(96)//2
_BB_TAIL_BW_MAX = 10000.0
_BB_A16W4_MAX_BL1_NZ = 96 * 1024
_BB_A16W4_MAX_BUB_NZ = 24 * 1024
_BB_PLAT = dict(
    l1=512 * 1024, l0a=64 * 1024, l0b=64 * 1024, freq=1.6, hbm=1.6, l2=5.4
)  # 950PR / V100 口径
_BB_A_BYTE, _BB_B_BYTE = 2.0, 0.5


def _bb_mte2_bw(bm, bn, md, nd):
    """给定分核与基本块的综合 MTE2 带宽估计（HBM/L2 两级模型）。"""
    den = ((md * bm) * _BB_A_BYTE + (nd * bn) * _BB_B_BYTE) / _BB_PLAT["hbm"] + (
        (md * nd - md) * bm * _BB_A_BYTE + bn * (md * nd - nd) * _BB_B_BYTE
    ) / _BB_PLAT["l2"]
    if den <= 0:
        return 0.0
    return (md * nd * (bm * _BB_A_BYTE + _BB_B_BYTE * bn)) / den * (md * nd / 32)


def _bb_bw_ratio(bm, bn, md, nd):
    """带宽比 = MTE2 带宽 / cube-bound 最低带宽（≥1 即达标）。"""
    lo = (
        _BB_BLOCK_CUBE**3
        * _BB_PLAT["freq"]
        * 0.001
        * md
        * nd
        * (_BB_A_BYTE / bn + _BB_B_BYTE / bm)
    )
    return _bb_mte2_bw(bm, bn, md, nd) / lo if lo > 0 else 0.0


def _bb_get_base_k(bm, bn, k):
    """GetBaseK（groupSize=32 ≤ baseK 的 weightNz 分支）。"""
    bk = max(_BB_BASE_MK_LIMIT // max(bm, bn), 32)
    bk = (bk // 32) * 32
    bk = bk // 32 * 32
    while _BB_A16W4_MAX_BUB_NZ % 32 == 0 and bk > 32 and _BB_A16W4_MAX_BUB_NZ % bk > 0:
        bk -= 32
    return min((ceil_align(k, _BB_BLOCK_CUBE) // 32) * 32, bk)


def _bb_candidates(md):
    """给定分核的基本块候选集（带宽比 > 1 过滤；baseM 升/baseN 降序）。"""
    nd = 32 // md
    row = []
    for bm in range(_BB_A16W4_START_M, _BB_BASE_BLOCK_MAX + 1, _BB_BLOCK_CUBE):
        start_n = min(
            _BB_BASE_BLOCK_MAX,
            (_BB_BASE_MN_LIMIT // bm) // _BB_BLOCK_CUBE * _BB_BLOCK_CUBE,
        )
        for bn in range(start_n, _BB_BLOCK_CUBE - 1, -_BB_BLOCK_CUBE):
            if _bb_bw_ratio(bm, bn, md, nd) > 1.0:
                row.append((bm, bn))
    return row


def _bb_invalid_nz(bn, k_bl1, single_n, n_size):
    """NZ 布局合法性：BL1 4-buffer 上限 + N 尾块须可拆分给两个 vector。"""
    if ceil_align(bn, 32) * k_bl1 * _BB_A_BYTE > _BB_A16W4_MAX_BL1_NZ:
        return True
    n_bub = ceil_align(ceil_div(bn, 2), _BB_BLOCK_CUBE)
    n_tail = single_n % bn
    tail_block = n_size % single_n % bn
    if 0 < n_tail <= n_bub:
        return True
    if 0 < tail_block <= n_bub and tail_block < n_size:
        return True
    return False


def _bb_l1_tiling(bm, bn, bk, single_m, single_n, k, n_size):
    """枚举 stepKa/stepKb（L1 容量/对齐约束下取 kBL1 最大的合法解）。"""
    step_k_max = ceil_div(k, bk)
    step_ka_max = min(ceil_div(step_k_max, 2), _BB_PLAT["l1"] // (bm * bk * 2))
    step_ka_min = (
        min(max(ceil_div(128, bk * 2), 1), step_ka_max) if k * 16 > 128 * 8 else 1
    )
    best = None
    step_ka = max(step_ka_min, 1)
    while step_ka <= step_ka_max:
        free = _BB_PLAT["l1"] - bm * bk * step_ka * _BB_A_BYTE
        step_kb_max = min(ceil_div(step_k_max, 2), free // (bn * bk * _BB_A_BYTE))
        step_kb = 1
        while step_kb <= step_kb_max:
            k_bl1 = step_kb * bk
            ka2 = step_ka
            if ceil_div(single_m, bm) == 1 and ceil_div(step_k_max, step_ka) <= 4:
                ka2 = step_k_max
            ok = k_bl1 % 32 == 0 or 32 % k_bl1 == 0
            if (
                ka2 < step_k_max
                and step_kb < step_k_max
                and ka2 % step_kb > 0
                and step_kb % ka2 > 0
            ):
                ok = False
            if ok and not _bb_invalid_nz(bn, k_bl1, single_n, n_size):
                if best is None or k_bl1 > best[1]:
                    best = (step_kb, k_bl1)
            step_kb += 1
        step_ka += step_ka_min
    return best


def _bb_half(v, max_base):
    while v > max_base:
        v = ((v // _BB_BLOCK_CUBE + 1) // 2) * _BB_BLOCK_CUBE
    return v


def _bb_cube_bound_solutions(row, single_m, single_n, md, nd, k):
    """整除/折半 5 案例 → cube-bound 候选解集。"""
    # 端点（baseM 升/baseN 降序）：row[0]=(minBaseM, maxBaseN)，row[-1]=(maxBaseM, minBaseN)
    if not row or single_m < row[0][0] or single_n < row[-1][1]:
        return []
    hm = _bb_half(single_m, row[-1][0])
    hn = _bb_half(single_n, row[0][1])
    out = []
    for bm, bn in row:
        tail = None
        if single_m % bm == 0 and single_n % bn == 0:
            tail = _BB_TAIL_BW_MAX + 1
        elif single_m % bm == 0 and single_n >= bn:
            tail = _bb_bw_ratio(bm, single_n % bn, md, nd)
        elif single_n % bn == 0 and single_m >= bm:
            tail = _bb_bw_ratio(single_m % bm, bn, md, nd)
        elif hm % bm == 0 and single_n >= bn:
            tail = _bb_bw_ratio(bm, single_n % bn, md, nd)
        elif hn % bn == 0 and single_m >= bm:
            tail = _bb_bw_ratio(single_m % bm, bn, md, nd)
        if tail is not None:
            out.append(
                (
                    bm,
                    bn,
                    _bb_get_base_k(bm, bn, k),
                    tail,
                    _bb_bw_ratio(bm, bn, md, nd),
                    md,
                    nd,
                )
            )
    return out


def _bb_cmp_cube(a, b):
    """cube-bound 解比较序：满核优先 → 无尾块优先 → 带宽比/尾块带宽比。"""
    if a[5] * a[6] != b[5] * b[6]:
        return b[5] * b[6] - a[5] * a[6]
    ta, tb = a[3], b[3]
    if ta > _BB_TAIL_BW_MAX and tb < _BB_TAIL_BW_MAX:
        return -1
    if ta < _BB_TAIL_BW_MAX and tb > _BB_TAIL_BW_MAX:
        return 1
    if ta > _BB_TAIL_BW_MAX and tb > _BB_TAIL_BW_MAX:
        return -1 if a[4] > b[4] else (1 if a[4] < b[4] else 0)
    return -1 if ta > tb else (1 if ta < tb else 0)


def _bb_solve(m, n, k):
    """求解入口：满核枚举优先，无解时回退非满核（0.8 核剪枝）。

    返回 dict(base_m, base_n, k_bl1) 或 None（无解）。回退档排序以带宽比
    近似。
    """
    from functools import cmp_to_key

    cands = []
    for i in range(6):
        md, nd = 1 << i, 32 >> i
        sm = ceil_align(ceil_div(m, md), _BB_BLOCK_CUBE)
        sn = ceil_align(ceil_div(n, nd), _BB_BLOCK_CUBE)
        if ceil_div(m, sm) != md or ceil_div(n, sn) != nd:
            continue
        cands += _bb_cube_bound_solutions(_bb_candidates(md), sm, sn, md, nd, k)
    cands.sort(key=cmp_to_key(_bb_cmp_cube))
    for bm, bn, bk, _, _, md, nd in cands:
        sm = ceil_align(ceil_div(m, md), _BB_BLOCK_CUBE)
        sn = ceil_align(ceil_div(n, nd), _BB_BLOCK_CUBE)
        l1 = _bb_l1_tiling(bm, bn, bk, sm, sn, k, n)
        if l1:
            return dict(base_m=bm, base_n=bn, k_bl1=l1[1])
    # ---- 回退：非满核（0.8 核剪枝 + 带宽比近似排序）----
    fallback = []
    for md in range(min(ceil_div(m, _BB_BLOCK_CUBE), 32), 0, -1):
        for nd in range(min(ceil_div(n, _BB_BLOCK_CUBE), 32 // md), 0, -1):
            if fallback and md * nd < 0.8 * 32:
                continue
            sm = ceil_align(ceil_div(m, md), _BB_BLOCK_CUBE)
            sn = ceil_align(ceil_div(n, nd), _BB_BLOCK_CUBE)
            if ceil_div(m, sm) != md or ceil_div(n, sn) != nd:
                continue
            for bm in range(_BB_BLOCK_CUBE, sm + 1, _BB_BLOCK_CUBE):
                bn_max = min(
                    _BB_BASE_BLOCK_MAX,
                    (_BB_BASE_MN_LIMIT // bm) // _BB_BLOCK_CUBE * _BB_BLOCK_CUBE,
                    sn,
                )
                for bn in range(bn_max, _BB_BLOCK_CUBE - 1, -_BB_BLOCK_CUBE):
                    if bm * bn > _BB_BASE_MN_LIMIT:
                        continue
                    bk = _bb_get_base_k(bm, bn, k)
                    l1 = _bb_l1_tiling(bm, bn, bk, sm, sn, k, n)
                    if l1:
                        fallback.append((bm, bn, l1[1], _bb_bw_ratio(bm, bn, md, nd)))
    if fallback:
        fallback.sort(key=lambda x: -x[3])
        bm, bn, k_bl1, _ = fallback[0]
        return dict(base_m=bm, base_n=bn, k_bl1=k_bl1)
    return None


def _device_max_grid() -> int:
    """运行设备的 MIX 核数（= cube 核数；探测失败回退 MAX_GRID=32）。

    写死 32 在核数更少的设备上会让尾块超配共享核（grid 超出物理
    MIX 单元，满配大 shape 串行化损失）。MIX 1 AIC : 2 AIV 配比由
    硬件固定，grid 直接取 cube 数。
    """
    npu = getattr(torch, "npu", None)
    if npu is None:
        return MAX_GRID
    try:
        props = npu.get_device_properties(npu.current_device())
        cube = int(getattr(props, "cube_core_num", 0))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return MAX_GRID
    return cube if cube > 0 else MAX_GRID


class WqbmmTiling:
    """Host-side tiling computation.

    tiling flow:
      ResolveGeometry → CalTileNum → CalKParts → CalNTail → Finalize

    Key outputs consumed by the kernel:
      bm, bn, kb, nh, n1h, k1s, sg, kb8 — tile 几何（带宽代价模型选块
        +投影，落于已验证几何集 {128×128×256, 256×128×128}）
      m_tiles, n_tiles, total_tiles  — 输出 tile 网格（idx2crd：N 最快）
      num_parts, full_parts          — K 份计数（满份 / 尾份切分）
      kb_tail, sg_tail               — 尾份真实 K / 尾份 scale 组数
      h1_always                      — N 尾免护栏标志
      used_core_num                  — launch grid
    """

    def __init__(
        self, m: int, n: int, k: int, *, bm: int = None, bn: int = None, kb: int = None
    ):
        """Store shape metadata and immediately derive all tiling fields.

        Args:
          m/n/k: 逻辑矩阵维度 y[M,N] = x[M,K] @ w[K,N]（K ≡ 0 mod 32，由
            torch 接口校验）。
          bm/bn/kb: tile 几何显式覆盖（需三者齐给；缺省走带宽代价
            模型选块+投影，供测试注入已验证几何）。
        """
        self.m = m
        self.n = n
        self.k = k
        # Tiling outputs; all of them are derived by _compute() below.
        self.m_tiles = None
        self.n_tiles = None
        self.total_tiles = None
        self.num_parts = None
        self.full_parts = None
        self.kb_tail = None
        self.sg_tail = None
        self.h1_always = None
        self.used_core_num = None
        self._resolve_geometry(bm, bn, kb)
        self._compute()

    @staticmethod
    def _project_geometry(m: int, n: int, k: int):
        """求解器块型 → 本实现已验证几何集投影。

        bm=128（贴主线偏好）在本实现协议下比 bm=256 慢：M tile × 2 →
        FIXPIPE/L1 开销 × 2，大于 part 数减半的节省；主线偏好薄 M 依赖
        其 BUB 级无协议开销流水，本实现不具备。两档投影已是 tile×part
        开销最优。投影规则：保块面积（baseM*baseN）——面积 ≥ 32768 →
        (256,128,128)，否则 (128,128,256)。kBL1 不透传（份大小受 L0
        单事务约束）。
        """
        res = _bb_solve(m, n, k)
        if res is None:
            return (BM, BN, KB)
        area = res["base_m"] * res["base_n"]
        if area >= 32768:
            g = (256, 128, 128)
        else:
            g = (BM, BN, KB)
        return g

    def _resolve_geometry(self, bm, bn, kb):
        """Select and validate the tile geometry against on-chip budgets."""
        if bm is None and bn is None and kb is None:
            g_bm, g_bn, g_kb = self._project_geometry(self.m, self.n, self.k)
        elif bm is None or bn is None or kb is None:
            raise ValueError("geometry override requires all of bm/bn/kb")
        else:
            g_bm, g_bn, g_kb = bm, bn, kb
        nh = g_bn // 2
        self.bm, self.bn, self.kb = g_bm, g_bn, g_kb
        self.nh = nh
        self.n1h = nh // 16
        self.k1s = g_kb // 16
        self.sg = g_kb // 32
        self.kb8 = g_kb * 8
        # ---- 合法性与片上预算（L0A/L0B/L0C/L1/UB，950PR）----
        if g_bm % 16 or g_bn % 32 or g_kb % 32:
            raise ValueError(
                f"geometry bm%16/bn%32/kb%32 required, got ({g_bm},{g_bn},{g_kb})"
            )
        if (self.sg * nh) % 128:
            raise ValueError(f"sg*nh must be multiple of 128, got {self.sg}*{nh}")
        if g_bm * g_kb * 2 > L0A_BUDGET:
            raise ValueError(
                f"L0A overflow: {g_bm}x{g_kb} fp16 = {g_bm * g_kb * 2}B > {L0A_BUDGET}B"
            )
        if g_bn * g_kb * 2 > L0B_BUDGET:
            raise ValueError(
                f"L0B overflow: {g_bn}x{g_kb} fp16 = {g_bn * g_kb * 2}B > {L0B_BUDGET}B"
            )
        if g_bm * g_bn * 4 > L0C_BUDGET:
            raise ValueError(
                f"L0C overflow: {g_bm}x{g_bn} fp32 = {g_bm * g_bn * 4}B > {L0C_BUDGET}B"
            )
        l1_use = 2 * (2 * g_kb * nh * 2) + 2 * (g_bm * g_kb * 2)
        if l1_use > L1_BUDGET:
            raise ValueError(f"L1 overflow: {l1_use}B > {L1_BUDGET}B")
        ub_use = (
            4 * (g_kb * nh * 2 + self.n1h * self.kb8 + self.sg * nh)
            + self.sg * nh * 2
            + 512 * 2
            + 512 * 2
        )
        if ub_use > UB_BUDGET:
            raise ValueError(f"UB overflow: {ub_use}B > {UB_BUDGET}B")

    def _compute(self):
        """Run the host-side tiling pipeline in dependency order.

        输出 tile 网格先行，随后 K 份切分与 N 尾标志，最后汇总 launch grid。
        """
        self._cal_tile_num()
        self._cal_k_parts()
        self._cal_n_tail()
        self._finalize()

    def _cal_tile_num(self):
        """Compute the output tile grid over the selected bm×bn geometry."""
        self.m_tiles = ceil_div(self.m, self.bm)
        self.n_tiles = ceil_div(self.n, self.bn)
        self.total_tiles = self.m_tiles * self.n_tiles

    def _cal_k_parts(self):
        """Compute K 份切分：满份数 / 尾份真实 K / 尾份 scale 组数."""
        self.num_parts = ceil_div(self.k, self.kb)
        self.kb_tail = self.k % self.kb if self.k % self.kb else self.kb
        self.full_parts = (
            self.num_parts if self.k % self.kb == 0 else self.num_parts - 1
        )
        self.sg_tail = self.kb_tail // 32

    def _cal_n_tail(self):
        """Compute the N-tail guard flag.

        N 尾 ∈ (nh, bn) 或整 bn：AIV1/AIC-h1 的原点恒在界内（actual
        extent 裁剪即可）→ 免运行时护栏。
        """
        n_tail = self.n % self.bn
        self.h1_always = (n_tail == 0) or (n_tail > self.nh)

    def _finalize(self):
        """Compute used_core_num（设备感知 + tile 规模择优，见 _SMALL_GRID_TILES 注释）.

        WQBMM_MAX_GRID=N（N>0）：绝对覆盖（实验档/保守回退，可高于设备核数，
        如 950DT 上强制 32）。
        """
        dev = _device_max_grid()
        if self.total_tiles <= _SMALL_GRID_TILES:
            grid = min(dev, self.total_tiles)
        else:
            grid = min(MAX_GRID, self.total_tiles)
        forced = int(os.environ.get("WQBMM_MAX_GRID", "0") or 0)
        if forced > 0:
            grid = min(forced, self.total_tiles)
        self.used_core_num = grid


# ============================================================================
# 2. Block Scheduler and Kernel
# ============================================================================

# 跨核同步：L1 zn handoff 走跨核 RingBuffer，READY/FREE 屏障与 AIV
# 子块归因由框架 counter 系统从张量访问自动推导。


class WqbmmKernel:
    """y = x @ dequant(w)：AIV 反量化流水 + AIC matmul 流水（L1 zn handoff）.

    编排层：跨核 handoff（单条 (2·kb, nh) L1 zn 跨核 RingBuffer，两 AIV
    各写一行块半区）在此声明；AIV 环/常量 UB 与 AIC L1/L0 通道按使用侧
    就近代码段声明。
    """

    def __init__(self, tiling: WqbmmTiling):
        """Pre-compute loop bounds and scheduling metadata for the kernel.

        The tiling object contains the per-shape tile/part counts. This
        constructor converts them into the trace-time branch flags and
        tail widths consumed by the device kernel.
        """
        self.t = tiling
        # K 尾 trace-time 分支（尾份 epilogue 是否生成）
        self.has_k_tail = (tiling.k % tiling.kb) != 0
        # MMAD unit flag（对齐主线 batch_mat_mul_v3 Normal config）：累加中 2、
        # 最终 K 块 3——硬件自动触发 FIXPIPE，省一轮 M_FIX。开启条件对齐主线
        # dbL0C == 1（单缓冲 L0C 特化）；本实现 l0c0/l0c1 双通道恒 depth=1，
        # 条件恒成立。
        self.use_unit_flag = True
        # B 权重 L2 策略（对齐主线 SetBL2Cache 语义，blaze
        # kernel_qbmm_mx_without_batch.h / quant_batch_matmul_mx 同款）：
        # B 跨 M tile 复用（base_m < m）时保持 NORMAL cache；仅当 M 单 tile
        # 覆盖（base_m >= m，即 m_tiles == 1）且 N 流 cacheline 对齐时 bypass
        # （ctl=4）。A/scale 对齐主线不带 hint（硬件默认 NORMAL）。
        # wqbmm 的 w GM 行距 = k*8 B，k%32==0 约束下恒 128B 对齐。
        self.disable_w_l2 = (tiling.bm >= tiling.m) and (tiling.k * 8 % 128 == 0)

    # Device kernel entry. It maps output tiles to MIX blocks (1 AIC + 2 AIV):
    # AIV 侧按份搬运 fp4/scale 并反量化到 L1 zn slot，AIC 侧 x nd2nz 入
    # L1/L0 后做跨份 MMAD 累加，最后 FIXPIPE 写回 GM。
    @kernel
    def wqbmm_kernel(
        self, gm_x: Tensor, gm_w: Tensor, gm_sc: Tensor, gm_idx: Tensor, gm_y: Tensor
    ):
        t = self.t
        sub = get_subblock_id()

        # ---------------- AIV 私有 UB（几何取自 tiling）----------------
        # fp4/scale 输入环 depth4（一次搬运 4 份、MTE2 领先不阻塞）。
        # fp4/scale 份都必须 2D：尾份部分产出按 slot 行布局放置（panel p
        # 落在 p*kb8、scale 组 g 落在 g*nh）；一维 flat 会把部分 extent 压平
        # 导致行错位。
        ub_wfp4 = RingBuffer(
            MemLoc.UB, dtypes.uint8, make_layout((t.n1h, t.kb8)), depth=4
        )
        ub_scin = RingBuffer(
            MemLoc.UB, dtypes.uint8, make_layout((t.sg, t.nh)), depth=4
        )
        # fp16 zn 输出环 depth4（depth2 的 slot 复用有竞态，d4 消除）。
        # UB 存储只允许 nd/nz 物理，zn 语义移到视图层——flat nd 分配，
        # _vf_quant 按物理 fractal 偏移直写（字节序即 zn），读侧经
        # reinterpret(data_format="zn") 视图供 MTE3 消费。
        ub_wzn = RingBuffer(
            MemLoc.UB, dtypes.float16, make_layout((t.kb * t.nh,)), depth=4
        )
        # V-pipe 程序序顺序复用 / 常驻（Buffer：无同步语义）
        ub_scf16 = Buffer(MemLoc.UB, dtypes.float16, make_layout((t.sg * t.nh,)))
        idx_n = 2048  # vcast 16 组转置索引表
        ub_idx = Buffer(
            MemLoc.UB, dtypes.uint16, make_layout((idx_n,))
        )  # 转置索引（vcast 模式含双表）

        # ---------------- 共享 L1：权重 zn 双 slot（行块拆半区）----------------
        # 跨核 RingBuffer，READY/FREE 屏障与 AIV 子块归由框架 counter
        # 系统自动推导。框架 counter 模型要求每个跨核环由块内两个 AIV
        # 共同 produce（AIC 侧端点等待 {id, id+16} 双归因 arrive）⇒ 两
        # AIV produce 同一 slot、经 reinterpret 显式 typed view 各写自己
        # 的 KB 行块（每半区为连续 32KB 整 zn 矩阵——M/N 分形 tile 会触发
        # "tiled M/N UB->L1 requires an NZ destination" 守卫）。K 尾裁剪
        # 移到 AIC 侧 consume 后的行前缀视图。
        # L1 存储只允许 nd/nz 物理——flat nd 分配（zn 网格 k1-major，
        # K 向行块恰为字节前缀/后缀，半区 byte-offset 拆分语义不变），读写
        # 两侧均经 reinterpret(data_format="zn") 视图。
        l1_w = RingBuffer(
            MemLoc.L1,
            dtypes.float16,
            make_layout((2 * t.kb * t.nh,)),
            depth=2,
            data_format="nd",
        )

        # ---------------- AIC 侧通道 ----------------
        l1_a = RingBuffer(
            MemLoc.L1,
            dtypes.float16,
            make_layout((t.bm, t.kb)),
            depth=2,
            data_format="nz",
        )
        # l0a/l0b depth1：每份 1 produce + 1 consume + 1 次整份 K=kb 的
        # MMAD（该几何下 L0A/L0B 各 64KB 精确满）。L0 K 步拆分（子 tile
        # consume）会触发通道 FIFO 账本歧义（动态循环 + gather-VF + 多块
        # 组合下 m-tile 折叠）；depth1 串行化为已知 perf 让步。
        l0a = RingBuffer(MemLoc.L0A, dtypes.float16, make_layout((t.bm, t.kb)), depth=1)
        l0b0 = RingBuffer(
            MemLoc.L0B, dtypes.float16, make_layout((t.nh, t.kb)), depth=1
        )
        l0b1 = RingBuffer(
            MemLoc.L0B, dtypes.float16, make_layout((t.nh, t.kb)), depth=1
        )
        # l0c 拆两通道：单通道双 matmul 写触发 multiple-accumulator-writers
        l0c0 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((t.bm, t.nh)), depth=1
        )
        l0c1 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((t.bm, t.nh)), depth=1
        )

        # ---------------- 一次性初始化 ----------------
        mem_copy(ub_idx, gm_idx)
        # scale f16 全量置 1.0：尾份垃圾组 / N 尾垃圾 panel 的 scale 读数
        # 保持有限值，杜绝 inf(scale)×0(A 补零行)=NaN 污染有效输出。
        self._vf_init_scale(ub_scf16)

        full_parts = t.full_parts
        has_k_tail = self.has_k_tail

        block_idx = get_block_idx()
        block_num = get_block_num()

        # L1 zn slot 轮转由跨核 RingBuffer 的 next_write/next_read 游标管理
        # （跨 tile 连续滚动，替代原 pslot 手动配对）。
        for tile_idx in range(block_idx, t.total_tiles, block_num):
            m_idx, n_idx = idx2crd(tile_idx, [t.m_tiles, t.n_tiles])  # N 最快
            # 第二 N 半区是否有有效列（仅 y-h1 写出目标选择；l0b1/l0c1 全程
            # 无条件运行——短 tile 的垃圾列有限且经 h0 区覆写，不出界）
            h1v = (n_idx * t.bn + t.nh) < t.n
            # 本 AIV 半区是否有有效列（AIV0 恒真；AIV1 = h1v）——GM 载入护栏
            active = (n_idx * t.bn + sub * t.nh) < t.n

            # l0c 累加事务：显式 next_write → 循环内 matmul 写 slot alias →
            # FIXPIPE 读（隐式协议下循环内多个 matmul 写同一
            # 通道会被判 multiple accumulator writers）
            c0w = l0c0.next_write()
            c1w = l0c1.next_write()

            # ============ 满份（动态循环，每份单次整份 MMAD）============
            # full_parts == 0（K ≤ kb）时不 trace 循环体：零 trip 循环的 consume
            # tile 仍会被静态 extent 检查（通道按尾份 (bm, kb_real) 规划 →
            # 满份 tile 裁剪到 0 被拒）。
            if full_parts > 0:
                for j in range(full_parts):
                    # ---- AIV：反量化份 j（满份 n_grp=sg）----
                    self._aiv_part(
                        gm_w,
                        gm_sc,
                        ub_wfp4,
                        ub_scin,
                        ub_wzn,
                        ub_idx,
                        ub_scf16,
                        l1_w,
                        n_idx,
                        j,
                        t.sg,
                        active,
                        sub,
                    )
                    # ---- AIC：消费份 j（整槽 1:1:1）----
                    a_l1 = l1_a.next_write()
                    mem_copy(a_l1, tile_slice(gm_x, (t.bm, t.kb), (m_idx, j)))
                    a_l0 = l0a.next_write()
                    mem_copy(a_l0, l1_a.next_read())
                    w_slot = l1_w.next_read()
                    b0_l0 = l0b0.next_write()
                    mem_copy(
                        b0_l0,
                        from_tensor(
                            w_slot,
                            layout=make_layout((t.kb, t.nh)),
                            offset=0,
                            data_format="zn",
                        ).permute(1, 0),
                    )
                    b1_l0 = l0b1.next_write()
                    mem_copy(
                        b1_l0,
                        from_tensor(
                            w_slot,
                            layout=make_layout((t.kb, t.nh)),
                            offset=t.kb * t.nh * 2,
                            data_format="zn",
                        ).permute(1, 0),
                    )
                    # unit_flag：累加中 2；无尾份时本份即最终块 → 3（硬件
                    # auto 触发 FIXPIPE）
                    uf = 3 if (j == full_parts - 1 and not has_k_tail) else 2
                    matmul(c0w, a_l0, b0_l0, init=(j == 0), unit_flag=uf)
                    matmul(c1w, a_l0, b1_l0, init=(j == 0), unit_flag=uf)

            # ============ 尾份 epilogue（trace-time 静态特化）============
            # kb_real = K mod kb：A 侧 tile_slice 裁剪、B 侧取 zn slot 行
            # 前缀视图（reinterpret）。
            # host 侧 K 补零到 kb 倍数后本分支恒不 trace（公共接口下
            # has_k_tail 恒 False）。直接实例化 WqbmmKernel 且 K%kb ≠ 0 时
            # 本路径会消费 L0 陈旧尾部（extent 不再传播到 MMAD）——须走
            # npu_weight_quant_batchmatmul 的补零入口。
            if has_k_tail:
                j_last = t.num_parts - 1
                # ---- AIV：反量化尾份（n_grp=sg_tail）----
                self._aiv_part(
                    gm_w,
                    gm_sc,
                    ub_wfp4,
                    ub_scin,
                    ub_wzn,
                    ub_idx,
                    ub_scf16,
                    l1_w,
                    n_idx,
                    j_last,
                    t.sg_tail,
                    active,
                    sub,
                )
                # ---- AIC：消费尾份（单步；A/B 侧操作数均按 kb_real 裁剪
                # 后进 MMAD）----
                a_l1 = l1_a.next_write()
                mem_copy(a_l1, tile_slice(gm_x, (t.bm, t.kb), (m_idx, j_last)))
                a_l0 = l0a.next_write()
                mem_copy(a_l0, tile_slice(l1_a.next_read(), (t.bm, t.kb_tail), (0, 0)))
                w_slot = l1_w.next_read()
                b0_l0 = l0b0.next_write()
                mem_copy(
                    b0_l0,
                    from_tensor(
                        w_slot,
                        layout=make_layout((t.kb_tail, t.nh)),
                        offset=0,
                        data_format="zn",
                    ).permute(1, 0),
                )
                b1_l0 = l0b1.next_write()
                mem_copy(
                    b1_l0,
                    from_tensor(
                        w_slot,
                        layout=make_layout((t.kb_tail, t.nh)),
                        offset=t.kb * t.nh * 2,
                        data_format="zn",
                    ).permute(1, 0),
                )
                matmul(c0w, a_l0, b0_l0, init=(full_parts == 0), unit_flag=3)
                matmul(c1w, a_l0, b1_l0, init=(full_parts == 0), unit_flag=3)

            # ============ AIC：y 写出（FIXPIPE，actual extent 裁剪尾块）============
            # 先写 h1（真实列 → 2n+1；短 tile 垃圾 → 写进 h0 区后被覆写），
            # 再写 h0（恒有效列）。unit_flag 路径：FIXPIPE 读 next_write alias
            # （next_read() 会与硬件 auto-FIXPIPE 游标脱同步——batch_matmul
            # 实证），src 即 c1w/c0w。
            if t.h1_always:
                mem_copy(
                    tile_slice(gm_y, (t.bm, t.nh), (m_idx, n_idx * 2 + 1)),
                    c1w,
                    options=copy_options(unit_flag=3),
                )
            else:
                if h1v:
                    mem_copy(
                        tile_slice(gm_y, (t.bm, t.nh), (m_idx, n_idx * 2 + 1)),
                        c1w,
                        options=copy_options(unit_flag=3),
                    )
                else:
                    # 短 tile：h1 无有效列（原点越界）→ 垃圾写进 h0 tile 区
                    mem_copy(
                        tile_slice(gm_y, (t.bm, t.nh), (m_idx, n_idx * 2)),
                        c1w,
                        options=copy_options(unit_flag=3),
                    )
            mem_copy(
                tile_slice(gm_y, (t.bm, t.nh), (m_idx, n_idx * 2)),
                c0w,
                options=copy_options(unit_flag=3),
            )

    @jit
    def _aiv_part(
        self,
        gm_w,
        gm_sc,
        ub_wfp4,
        ub_scin,
        ub_wzn,
        ub_idx,
        ub_scf16,
        l1_w,
        n_idx,
        j,
        n_grp,
        active,
        sub,
    ):
        """AIV 单份流水段（满份/尾份共用；j 可为 SSA 或 Python int）。

        MTE2 载入 fp4+scale 份 → V 反量化 → MTE3 写 L1 zn CrossCore slot 的
        本 AIV batch 半区（两 AIV produce 同一 slot、写 (2, KB, NH) 的不同
        batch——框架 counter 模型要求每通道两 AIV 共同 produce）。核内
        MTE2→V→MTE3 顺序与跨核 READY/FREE 均由框架从张量访问推导。
        """
        t = self.t
        w_l2 = 4 if self.disable_w_l2 else 0
        pin = ub_wfp4.next_write()
        if t.h1_always:
            # coord 是 tile 索引（R2）：(2n+sub, j) —— 尾块 actual extent 自动裁剪
            mem_copy(
                pin,
                tile_slice(gm_w, (t.n1h, t.kb8), (n_idx * 2 + sub, j)),
                options=copy_options(l2_cache_ctl=w_l2),
            )
        else:
            if active:
                mem_copy(
                    pin,
                    tile_slice(gm_w, (t.n1h, t.kb8), (n_idx * 2 + sub, j)),
                    options=copy_options(l2_cache_ctl=w_l2),
                )
            else:
                # 短 tile 的 AIV1：自身原点越界 → 改载 AIV0 坐标的真实权重
                # （本半区 = AIV0 半区的重复内容；越界列在 y 写出时被
                # actual extent 裁掉，不出界）
                mem_copy(
                    pin,
                    tile_slice(gm_w, (t.n1h, t.kb8), (n_idx * 2, j)),
                    options=copy_options(l2_cache_ctl=w_l2),
                )
        sin = ub_scin.next_write()
        if t.h1_always:
            mem_copy(sin, tile_slice(gm_sc, (t.sg, t.nh), (j, n_idx * 2 + sub)))
        else:
            if active:
                mem_copy(sin, tile_slice(gm_sc, (t.sg, t.nh), (j, n_idx * 2 + sub)))
            else:
                mem_copy(sin, tile_slice(gm_sc, (t.sg, t.nh), (j, n_idx * 2)))

        # V：反量化（scale conv 只转 n_grp 组 → 垃圾组保持 1.0/前份有限值）
        fp4r = ub_wfp4.next_read()
        scr = ub_scin.next_read()
        znw = ub_wzn.next_write()
        self._vf_quant(fp4r, scr, ub_idx, ub_scf16, znw, n_grp)

        # MTE3：UB zn 半区 → L1 zn slot 的本 AIV 行块（zn→zn 恒等）
        znr = from_tensor(
            ub_wzn.next_read(), layout=make_layout((t.kb, t.nh)), data_format="zn"
        )
        slot = l1_w.next_write()
        if sub == 0:
            mem_copy(
                from_tensor(
                    slot, layout=make_layout((t.kb, t.nh)), offset=0, data_format="zn"
                ),
                znr,
            )
        else:
            mem_copy(
                from_tensor(
                    slot,
                    layout=make_layout((t.kb, t.nh)),
                    offset=t.kb * t.nh * 2,
                    data_format="zn",
                ),
                znr,
            )

    @jit
    def _vf_init_scale(self, ub_scf16):
        """启动初始化 scale f16 = 1.0（尾组 NaN 防护）。"""
        with vf(mode="simd"):
            m16 = create_mask("all", elem_bits=16)
            one = vdups(1.0, dtypes.float16, mask=m16)
            for off in range_constexpr(self.t.sg * self.t.nh // 128):
                vstore(ub_scf16, off * 128, one, m16)
            vmem_bar("vst_vld")

    @jit
    def _vf_quant(self, ub_fp4, ub_scin, ub_idx, ub_scf16, ub_zn, n_grp: int):
        """份反量化 VF（WeightNZ 输入唯一解码链）。

        ① scale conv：e8m0 → f16（e<<23 f32 位构造，n_grp 组；垃圾组保持
           1.0 初值/前份有限值 → K/N 尾 NaN 路径封死）
        ② fp4 解码（WeightNZ panel）：vload 256B + vreinterpret_lanes(fp4x2)
           + 4×vcast（位相 → bf16 → f16）——硬件 cast 路径
        ③ 分形段重排：vgather_reg + vselect 三级选择树，P 位相 → zn 序
           （n0-major，L1 zn 消费端布局）
        ④ e2b scale 广播 ×vmul（分形 lane 每 16 lane 一列）
        ⑤ zn 物理平铺直写（plain vstore 按物理 fractal 寻址）

        满份与尾份同链（尾份垃圾组由 _vf_init_scale 的 1.0 防护封死）。
        fp4 panel 按 slot 行布局寻址（panel p 落在 p*KB8——produce 按
        actual extent 列裁剪后行放置）。k1 越过 kb_real/16 的分形读的
        是本行陈旧字节——有限 fp4 垃圾 × scale 防护，且尾份 zn 行不被
        AIC 消费，无害。
        """
        t = self.t
        with vf(mode="simd"):
            m16 = create_mask("all", elem_bits=16)
            m32 = create_mask("all", elem_bits=32)
            # ---- ① scale conv ----
            for part in range_constexpr(n_grp):
                idx32 = vload_unpack(
                    ub_scin, part * t.nh, unpack_mode=UnpackMode.B8_TO_B32
                )
                bits = vshl(idx32, 23, mask=m32)  # f32 位型 2^(e-127)
                f32 = vreinterpret(bits, dtypes.float32)
                f16 = vcast(f32, dtypes.float16, mask=m32)
                vstore_pack(
                    ub_scf16, part * t.nh, f16, m32, pack_mode=PackMode.B32_TO_B16
                )
            vmem_bar("vst_vld")
            ar = varange(0, dtypes.uint16)
            # ============ vcast 解码链（WeightNZ panel 唯一路径）============
            # 每 256B（2 个 k1 分形）：vload + vreinterpret_lanes(fp4x2)
            # + 4xvcast(P0..P3->bf16) + 4xvcast(->f16) + 分形段 gather
            # 重排到 zn 序（P 位相 -> n0-major，索引含 +64X 分形偏移）
            m8 = create_mask("all", elem_bits=8)
            it_0 = vload(ub_idx, 0)
            it_1 = vload(ub_idx, 128)
            it_2 = vload(ub_idx, 256)
            it_3 = vload(ub_idx, 384)
            it_4 = vload(ub_idx, 512)
            it_5 = vload(ub_idx, 640)
            it_6 = vload(ub_idx, 768)
            it_7 = vload(ub_idx, 896)
            it_8 = vload(ub_idx, 1024)
            it_9 = vload(ub_idx, 1152)
            it_a = vload(ub_idx, 1280)
            it_b = vload(ub_idx, 1408)
            it_c = vload(ub_idx, 1536)
            it_d = vload(ub_idx, 1664)
            it_e = vload(ub_idx, 1792)
            it_f = vload(ub_idx, 1920)
            # 段选择 mask：p(i) = i//32（阈值 32/64/96），o0/o1 共用
            m_lo = vlt(vdups(31, dtypes.uint16, mask=m16), ar, mask=m16)
            m_mid = vlt(vdups(63, dtypes.uint16, mask=m16), ar, mask=m16)
            m_hi = vlt(vdups(95, dtypes.uint16, mask=m16), ar, mask=m16)
            for fb in range_constexpr((t.k1s // 2) * t.n1h):
                k1b = (fb // t.n1h) * 2
                n1h = fb % t.n1h
                byte_off = n1h * t.kb8 + k1b * 128
                src = vload(ub_fp4, byte_off)
                fp4 = vreinterpret_lanes(src, dtypes.fp4x2_e2m1)
                p0b = vcast(fp4, dtypes.bfloat16, mask=m8, reg_layout=RegLayout.ZERO)
                p1b = vcast(fp4, dtypes.bfloat16, mask=m8, reg_layout=RegLayout.ONE)
                p2b = vcast(fp4, dtypes.bfloat16, mask=m8, reg_layout=RegLayout.TWO)
                p3b = vcast(fp4, dtypes.bfloat16, mask=m8, reg_layout=RegLayout.THREE)
                pf0 = vcast(p0b, dtypes.float16, mask=m16)
                pf1 = vcast(p1b, dtypes.float16, mask=m16)
                pf2 = vcast(p2b, dtypes.float16, mask=m16)
                pf3 = vcast(p3b, dtypes.float16, mask=m16)
                # ---- 分形 A (k1b + 0) ----
                g00 = vgather_reg(pf0, it_0)
                g01 = vgather_reg(pf1, it_1)
                g02 = vgather_reg(pf2, it_2)
                g03 = vgather_reg(pf3, it_3)
                g10 = vgather_reg(pf0, it_4)
                g11 = vgather_reg(pf1, it_5)
                g12 = vgather_reg(pf2, it_6)
                g13 = vgather_reg(pf3, it_7)
                t0 = vselect(g01, g00, cond_mask=m_lo)
                t1 = vselect(g03, g02, cond_mask=m_hi)
                o0 = vselect(t1, t0, cond_mask=m_mid)
                u0 = vselect(g11, g10, cond_mask=m_lo)
                u1 = vselect(g13, g12, cond_mask=m_hi)
                o1 = vselect(u1, u0, cond_mask=m_mid)
                g_ = k1b // 2
                sv0 = vreinterpret(
                    vload_broadcast(ub_scf16, g_ * t.nh + n1h * 16, mode="datablock"),
                    dtypes.float16,
                )
                sv1 = vreinterpret(
                    vload_broadcast(
                        ub_scf16, g_ * t.nh + n1h * 16 + 8, mode="datablock"
                    ),
                    dtypes.float16,
                )
                o0 = vmul(o0, sv0, mask=m16)
                o1 = vmul(o1, sv1, mask=m16)
                zn_off = k1b * (t.n1h * 256) + n1h * 256
                vstore(ub_zn, zn_off, o0, m16)
                vstore(ub_zn, zn_off + 128, o1, m16)
                # ---- 分形 B (k1b + 1) ----（预生成 +64 表 it_8..it_f）
                h00 = vgather_reg(pf0, it_8)
                h01 = vgather_reg(pf1, it_9)
                h02 = vgather_reg(pf2, it_a)
                h03 = vgather_reg(pf3, it_b)
                h10 = vgather_reg(pf0, it_c)
                h11 = vgather_reg(pf1, it_d)
                h12 = vgather_reg(pf2, it_e)
                h13 = vgather_reg(pf3, it_f)
                v0 = vselect(h01, h00, cond_mask=m_lo)
                v1 = vselect(h03, h02, cond_mask=m_hi)
                o0 = vselect(v1, v0, cond_mask=m_mid)
                w0 = vselect(h11, h10, cond_mask=m_lo)
                w1 = vselect(h13, h12, cond_mask=m_hi)
                o1 = vselect(w1, w0, cond_mask=m_mid)
                g_ = (k1b + 1) // 2
                sv0 = vreinterpret(
                    vload_broadcast(ub_scf16, g_ * t.nh + n1h * 16, mode="datablock"),
                    dtypes.float16,
                )
                sv1 = vreinterpret(
                    vload_broadcast(
                        ub_scf16, g_ * t.nh + n1h * 16 + 8, mode="datablock"
                    ),
                    dtypes.float16,
                )
                o0 = vmul(o0, sv0, mask=m16)
                o1 = vmul(o1, sv1, mask=m16)
                zn_off = (k1b + 1) * (t.n1h * 256) + n1h * 256
                vstore(ub_zn, zn_off, o0, m16)
                vstore(ub_zn, zn_off + 128, o1, m16)

    # Host wrapper used by the Python interface. The bracket syntax launches the
    # kernel with the grid selected by host-side tiling.
    # 按位置绑定（api-ref #17）：signature 顺序 = x, w, sc, idx, y
    # 0.6.2 起 Python 入口须 @host（@jit 为 inline helper，仅限 @host/@kernel
    # 内调用）；@kernel 内部的 @jit 方法不受影响。
    @host
    def run(
        self, gm_x: Tensor, gm_w: Tensor, gm_sc: Tensor, gm_idx: Tensor, gm_y: Tensor
    ):
        self.wqbmm_kernel[self.t.used_core_num](gm_x, gm_w, gm_sc, gm_idx, gm_y)


# ============================================================================
# 3. Torch Interface
# ============================================================================

_const_lock = threading.Lock()
_const_cache: dict = {}


def _vcast_transpose_indices() -> np.ndarray:
    """vcast P 序 -> zn 序的 gather 索引（8×128 u16）。

    P_p lane l(=2j+c) = byte(4j+p) 的 (c? hi : lo) nibble。zn 目标
    o[i]（n0=i//16, k0=i%16）的源：byte b = 8*k0 + n0//2（分形内），
    p(i) = (n0//2) % 4 = i//32，lane = 64*X + 4*k0 + (n0%2) (+2 for o1)。
    分形 X∈{0,1} 由 +64X 偏移。非 p(i) 段的索引填 0（select 树保证
    只取正确分支）。
    """
    idx = np.zeros((16, 128), dtype=np.uint16)
    for i in range(128):
        p, k0, c = i // 32, i % 16, (i // 16) % 2
        idx[p][i] = 4 * k0 + c
        idx[4 + p][i] = 4 * k0 + c + 2
        idx[8 + p][i] = 64 + 4 * k0 + c  # 分形 B（+64）
        idx[12 + p][i] = 64 + 4 * k0 + c + 2
    return idx.reshape(-1)


def _host_constants(device):
    """vcast 转置索引（2048 u16）常驻 GM 张量，进程内缓存。"""
    with _const_lock:
        key = str(device)
        if key not in _const_cache:
            idx = torch.from_numpy(_vcast_transpose_indices())
            _const_cache[key] = idx.to(device)
        return _const_cache[key]


def _check_x(x: torch.Tensor) -> tuple:
    """校验激活 x：fp16 / 2-D (M,K) / 连续 / K 为 MX 组宽，返回 (M, K)。"""
    if x.dtype != torch.float16:
        raise ValueError(f"x must be fp16, got {x.dtype}")
    if x.dim() != 2:
        raise ValueError(f"x must be 2-D (M, K), got {x.dim()}-D")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous (v1 无 stride 透传兜底)")
    M, K = x.shape
    if K % 32 != 0:
        raise ValueError(f"K must be a multiple of 32 (MX group), got K={K}")
    if K < 32:
        raise ValueError(f"K must be >= 32, got K={K}")
    return M, K


def _check_antiquant_scale(antiquant_scale: torch.Tensor, K: int) -> tuple:
    """校验 antiquant_scale：2-D (K//32, N) / 连续 / E8M0 或 u8 载体，返回 (N, uint8 视图)。"""
    if antiquant_scale.dim() != 2:
        raise ValueError(
            f"antiquant_scale must be 2-D (K//32, N), got {antiquant_scale.dim()}-D"
        )
    if not antiquant_scale.is_contiguous():
        raise ValueError("antiquant_scale must be contiguous")
    if antiquant_scale.shape[0] != K // 32:
        raise ValueError(
            f"antiquant_scale shape {tuple(antiquant_scale.shape)} inconsistent "
            f"with K={K} (expect rows={K // 32})"
        )
    N = antiquant_scale.shape[1]
    if N < 1:
        raise ValueError("N must be >= 1")
    if antiquant_scale.dtype == torch.float8_e8m0fnu:
        sc_u8 = antiquant_scale.view(torch.uint8)
    elif antiquant_scale.dtype in (torch.uint8, torch.int8):
        sc_u8 = (
            antiquant_scale.view(torch.uint8)
            if antiquant_scale.dtype == torch.int8
            else antiquant_scale
        )
    else:
        raise ValueError(
            f"antiquant_scale must be float8_e8m0fnu/uint8, got {antiquant_scale.dtype}"
        )
    return N, sc_u8


def _check_weight(weight: torch.Tensor, M: int, N: int, K: int) -> torch.Tensor:
    """校验 weight：int8/uint8 载体 / 2-D (⌈N/16⌉, K·8) WeightNZ / 连续，返回 uint8 视图。"""
    if weight.dtype not in (torch.int8, torch.uint8):
        raise ValueError(f"weight must be int8/uint8 carrier, got {weight.dtype}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D (ceil(N/16), K*8), got {weight.dim()}-D")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous (物理布局即语义)")
    expect_shape = (ceil_div(N, 16), K * 8)
    if tuple(weight.shape) != expect_shape:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} != expected "
            f"{expect_shape} for (M={M}, N={N}, K={K})"
        )
    return weight.view(torch.uint8) if weight.dtype == torch.int8 else weight


def _check_device(
    x: torch.Tensor, weight: torch.Tensor, antiquant_scale: torch.Tensor
) -> None:
    """校验 device：x 在 NPU，antiquant_scale/weight 与 x 同 device。"""
    if x.device.type != "npu":
        raise ValueError(f"x must be on NPU, got {x.device}")
    if antiquant_scale.device != x.device or weight.device != x.device:
        raise ValueError(
            f"antiquant_scale/weight must be on the same device as x ({x.device}), "
            f"got antiquant_scale on {antiquant_scale.device}, weight on {weight.device}"
        )


def _pad_inputs(
    x: torch.Tensor,
    w_u8: torch.Tensor,
    sc_u8: torch.Tensor,
    M: int,
    N: int,
    K: int,
    n_pad: int,
    k_pad: int,
    tiling: "WqbmmTiling",
) -> tuple:
    """Host 侧补零到 tiling 几何倍数。

    - K 尾（full_parts=0 / kb_tail < kb）：小份拷入大 L0 slot 后，extent
      不再传播到 MMAD——MMAD 消费 L0A/L0B 陈旧尾部（未初始化 L0 位模
      式 → 跨进程随机错误 / NaN）；
    - N 尾（n_tail ∈ (0, nh)，末 tile AIV0 半区部分有效）：末 tile 的
      部分列 FIXPIPE 会丢失其它 tile 的 h0 写出。
    统一补零到满宽后所有 tile/份走满几何路径（对齐 shape 零开销、与
    已验证路径完全一致），输出按 (M, N) 裁剪返回。

    M 尾（M % bm ≠ 0）：sub-bm M 的 L0A/L0C 事务与 MMAD/FIXPIPE 组合
    存在正确性缺陷（每半区 col 16-63 错位），补齐后 A 事务恒满 BM。
    补零行 × 任意权重 = 0，不影响有效行。
    K 尾：x 补零列 + w 补零字节（E2M1(0) = +0.0）+ scale 补零行
    （e8m0 0 → f16 0）——三重零保护，补零 K 列贡献恒 0。
    N 尾：w 补零 panel 行 + scale 补零列（scale=0）→ 补齐列 B=0，
    列独立性下无害（y 裁剪丢弃）。注意 w 的 panel 行距 = K·8，K 补齐
    时行距变化，须整体重建（不可原地右补）。
    """
    m_pad = tiling.m_tiles * tiling.bm
    # x: (M, K) -> (m_pad, k_pad)
    if m_pad != M or k_pad != K:
        x_use = torch.zeros(m_pad, k_pad, dtype=torch.float16, device=x.device)
        x_use[:M, :K] = x
    else:
        x_use = x
    # w: (ceil(N/16), K*8) -> (ceil(n_pad/16), k_pad*8)
    p_pad = ceil_div(n_pad, 16)
    if p_pad != w_u8.shape[0] or k_pad != K:
        w_use = torch.zeros(p_pad, k_pad * 8, dtype=w_u8.dtype, device=w_u8.device)
        w_use[: w_u8.shape[0], : K * 8] = w_u8
    else:
        w_use = w_u8
    # scale: (K//32, N) -> (k_pad//32, n_pad)
    if k_pad != K or n_pad != N:
        sc_use = torch.zeros(k_pad // 32, n_pad, dtype=sc_u8.dtype, device=sc_u8.device)
        sc_use[: K // 32, :N] = sc_u8
    else:
        sc_use = sc_u8
    return x_use, w_use, sc_use


def npu_weight_quant_batchmatmul(
    x: torch.Tensor,  # fp16 (M, K) ND 连续
    weight: torch.Tensor,  # int8/uint8 载体，WeightNZ 布局 (⌈N/16⌉, K·8)
    antiquant_scale: torch.Tensor,  # float8_e8m0fnu 或 uint8/int8 (K//32, N) ND 连续
) -> torch.Tensor:
    """y[M,N] = x @ (fp4→fp16(w) × e8m0→fp16(antiquant_scale))，FP16 输出。

    仅暴露 kernel 消费的三个张量入参（antiquant-MX A16W4 路径：无 offset/
    bias/quant-out）；输出在内部分配并返回。weight 仅支持 WeightNZ 布局
    （(⌈N/16⌉, K·8) 物理排布，lo nibble = 偶数列）。补零 pad 的 K/N 维
    会在内部裁剪，返回 (M, N) 切片视图（pad 时非连续）。

    约束：K ≡ 0 (mod 32)；x/weight/antiquant_scale 标准连续
    ND；非连续或 dtype 不符 → ValueError（不做 .contiguous() 兜底）。
    """
    M, K = _check_x(x)
    N, sc_u8 = _check_antiquant_scale(antiquant_scale, K)
    w_u8 = _check_weight(weight, M, N, K)
    _check_device(x, weight, antiquant_scale)
    # 几何按原始 shape 求解（保持带宽代价模型的选块口径），随后把 K/N
    # 补齐到几何倍数——满份 × 满 tile 路径（0.8 适配，见 _pad_inputs）。
    geom = WqbmmTiling(M, N, K)
    n_pad = ceil_div(N, geom.bn) * geom.bn
    k_pad = ceil_div(K, geom.kb) * geom.kb
    if n_pad == N and k_pad == K:
        tiling = geom
    else:
        tiling = WqbmmTiling(M, n_pad, k_pad, bm=geom.bm, bn=geom.bn, kb=geom.kb)
    x_use, w_use, sc_use = _pad_inputs(x, w_u8, sc_u8, M, N, K, n_pad, k_pad, tiling)
    out = torch.empty(
        tiling.m_tiles * tiling.bm, n_pad, dtype=torch.float16, device=x.device
    )
    idx_t = _host_constants(x.device)
    op = WqbmmKernel(tiling)
    op.run(x_use, w_use, sc_use, idx_t, out)
    return out[:M, :N]
