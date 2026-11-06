# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""支持 token 数泛化的 MXFP8 attention prologue。

小 T 对 QA/KV 使用 split-K，在归一化前合并 FP32 部分和；大 T 使用有界的
token 分块，QB 按 token 分块和 head 分配任务。权重预先转换为 FRACTAL_NZ，
跨核数据通过 GM 和生产者发布同步传递。DSL 根据每次调用的 T 进行编译特化。
FixPipe 使用补齐的物理尾块，公开输出和 cache 更新仍使用逻辑 T。
"""

from __future__ import annotations

import ctypes
import dataclasses
import math
import numbers
import threading
from collections import OrderedDict

import torch
import torch_npu

import cannbotdsl
import cannbotdsl.reg as rr
from cannbotdsl import (
    Buffer,
    TensorSpec,
    const_expr,
    MemLoc,
    dtypes,
    get_block_idx,
    get_mem_size,
    get_platform_info,
    get_subblock_id,
    host,
    jit,
    kernel,
    make_layout,
    matmul,
    mem_copy,
    range_constexpr,
    reinterpret,
    tile_slice,
    vf,
)
from cannbotdsl.ops.memcpy import copy_options
from cannbotdsl.ring_buffer import RingBuffer

from cannbotdsl.ops.sync import (
    PIPE,
    cube_sync_all,
    vec_sync_all,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    vec_sync_block_arrive,
    vec_sync_block_wait,
)

FLOAT8_E4M3FN = dtypes.float8_e4m3fn
FLOAT8_E8M0 = dtypes.float8_e8m0

# MXFP8 编码常量，与独立 golden 保持一致。
MX_GROUP = 32  # 每个 E8M0 scale 对应的元素数。
MX_PAIR = 2  # 公开布局中相邻 scale 组的配对数。
E8M0_BIAS = 127
E4M3_MAX = 448.0  # E4M3FN 的最大有限值。
QUANT_EXP_HEADROOM = 8  # 为 E4M3 尾数预留动态范围。
FP32_EXP_SHIFT = 23
FP32_EXP_BIAS = 127

VL = 64  # 每个向量寄存器的 FP32 lane 数。
UB_ALIGN = 32  # UB 向量访问的字节对齐要求。
FRACTAL_N = 32  # FP8 内层输出轴宽度。
FRACTAL_M = 16  # NZ 内层行数。
FIXPIPE_ACCUMULATE = 2  # 继续累加尚未完成的 L0C 分块。
FIXPIPE_FINAL = 3  # 将完整 L0C 分块发布给 FixPipe。
L0_DEPTH = 2  # L0 操作数的缓冲深度。
# NZ 权重网格：每个 fractal 包含 16 个输出行，归约方向占 32 字节。
NZ_M_FRAC = 16
NZ_C0_1B = 32
W_BUF = 2  # K 窗口权重预取的缓冲槽数。

FRACTAL_NZ = 29
ND_FORMATS = (0, 2)
MX_PAIR_WIDTH = MX_GROUP * MX_PAIR
FP32_BYTES = 4
L0C_RESERVED_BYTES = 16 * 1024
ACL_MEMCPY_DEVICE_TO_HOST = 2
MAX_COMPILED_KERNELS = 32
# 驻留 QB panel 包含四个 N64 分块和一个 K320 窗口。
PANEL_OUTPUT_TILES = 4
PANEL_TILE_OUTPUT_WIDTH = 64
PANEL_OUTPUT_WIDTH = PANEL_OUTPUT_TILES * PANEL_TILE_OUTPUT_WIDTH
PANEL_REDUCTION_WIDTH = 320
PANEL_REDUCTION_WINDOWS = 4
PANEL_RANK_WIDTH = PANEL_REDUCTION_WIDTH * PANEL_REDUCTION_WINDOWS
PANEL_N_FRACTALS = PANEL_OUTPUT_WIDTH // NZ_M_FRAC
PANEL_K_FRACTALS = PANEL_REDUCTION_WIDTH // NZ_C0_1B
PANEL_BYTES = PANEL_OUTPUT_WIDTH * PANEL_REDUCTION_WIDTH
# 驻留特化几何：(hidden_size, rank_size, head_count×head_size, head_size, rope_width)。
RESIDENT_GEOMETRY = (5120, PANEL_RANK_WIDTH, 32768, 512, 64)

# ---- 硬件执行模型 ----
AIV_PER_BLOCK = 2  # 每个 Cube block 绑定的 Vector subblock 数（AIV 全局编号 = block_idx*2 + subblock_id）。
FIXPIPE_DUAL_DST = 2  # FixPipe 双目的地沿行/列的对半切分数。
HEAD_SLICES = (
    AIV_PER_BLOCK * FIXPIPE_DUAL_DST
)  # head 特征轴被双目的地×双 subblock 切成的份数。
ZN_FRACTAL_BYTES = NZ_M_FRAC * MX_PAIR  # E8M0 zn 分形（16 行 × 2 组）的字节数。

# ---- 分块搜索边界 ----
TOKEN_TILE_ALIGN = NZ_M_FRAC  # token 分块行按 NZ 网格（16 行一个 fractal）对齐。
K_L1_MAX = 512  # L1 激活/权重 K 窗口宽度上限。
BK_MAX = 128  # L0 归约分块宽度上限。
BM_A1_MAX = 128  # split_k 模板 token 分块行数上限。
BM_SPLIT_T_MAX = 96  # split_t 模板 token 分块行数上限（缩小 A1/A2 为 B 保留容量）。

# ---- 模板与 split-K 选择 ----
DECODE_TOKEN_LIMIT = 256  # decode（split_k）与 prefill（split_t）模板的 token 分界。
DECODE_SPLIT_K_TOKENS = 128  # 超过该 token 数的 decode 固定使用 4 路 split-K。
SPLIT_K_DECODE = 4  # token 较多 decode 的 K 分块数。
LONG_TOKEN_LIMIT = 4096  # 启用 20 路 split-K 的 prefill token 阈值。

# ---- 驻留（L1 复用）几何 ----
RESIDENT_BN = 64  # 驻留路径的输出列分块（N64）。
K_WINDOW_RESIDENT = 320  # 驻留路径预取的 K 窗口宽度。
RESIDENT_K_WINDOWS = (  # 驻留权重预取的 K 窗口轮转槽数（1280 / 320 = 4）。
    PANEL_RANK_WIDTH // K_WINDOW_RESIDENT
)
RESIDENT_HEAD_PANELS = 2  # 驻留 panel 打包下每 head 切成的半区数（512 / 256）。
BN_B_FULL_HEAD = 512  # 覆盖完整 head 的 QB 输出列分块。

# ---- A2 归约分块 ----
BM_A2_PREFILL = 4  # split_t 且无 split-K 时 A2 的 token 分块行数。
A2_BUF_DEPTH_PREFILL = 1  # split_t 模板的 A2 Channel 深度。
A2_BUF_DEPTH_DECODE = 2  # split_k 模板的 A2 Channel 深度。


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def _align(value: int, alignment: int) -> int:
    """将 value 向上对齐到 alignment 的整数倍。"""
    return _ceil_div(value, alignment) * alignment


def _largest_divisor(value: int, cap: int, step: int) -> int:
    """返回不超过 cap、且为 step 整数倍的最大因子。"""
    best = 0
    candidate = step
    while candidate <= cap:
        if value % candidate == 0:
            best = candidate
        candidate += step
    if best == 0:
        raise ValueError(
            f"No divisor of {value} is at most {cap} and divisible by {step}"
        )
    return best


# host 侧分块与编译期常量
def _resident_groups(owners, n_count, cores, weight_bytes, activation_bytes):
    """host 侧负载模型：最小化最繁忙核的搬运量。保留原有分组候选，兼容不能整除任务数的核数配置。该模型仅用于调度选择，不直接预测设备执行耗时。"""
    old = min(n_count, _ceil_div(cores, owners))
    candidates = {old} | {g for g in range(1, n_count + 1) if n_count % g == 0}

    def critical_path(groups):
        costs = [0] * cores
        for job in range(owners * groups):
            tiles = _ceil_div(n_count - job % groups, groups)
            costs[job % cores] += tiles * weight_bytes + activation_bytes
        return max(costs)

    return min(candidates, key=lambda g: (critical_path(g), g))


@dataclasses.dataclass(frozen=True)
class AttnPrologueTiling:
    """一个形状和设备组合的静态分块，所有字段均参与编译特化。Cube 核数由平台接口提供。"""

    token_count: int
    hidden_size: int
    rank_size: int
    head_count: int
    head_size: int
    rope_width: int
    block_dim: int
    template: str
    split_k: int
    # A1 将 split-K、token 和输出列分块的 FP32 部分和写入交换区。
    # A2 在生产者发布数据后归约部分和，并逐 token 归一化。
    bm_a1: int
    bn_a1: int
    bk_a1: int
    k_l1_a1: int
    bm_a2: int
    bm_b: int
    bn_b: int
    bk_b: int
    k_l1_b: int
    # 策略标志是编译期常量，也是编译缓存键的一部分。
    a1_window: bool = False
    b_resident: bool = False
    l1_reuse: bool = False

    @property
    def k_l1_x_a1(self) -> int:
        """激活的 L1 K 窗口宽度；L0A 计算块仍使用 bk_a1。"""
        if self.a1_window:
            return self.k_l1_a1
        return self.bk_a1

    @property
    def qr_groups(self) -> int:
        """每行 rank 特征的量化组数，也是公开 scale 的每行字节数。"""
        return self.rank_size // MX_GROUP

    @property
    def exp_stride(self) -> int:
        """指数暂存的行跨度，以 int32 元素计，按 UB_ALIGN 字节对齐。"""
        return _ceil_div(self.qr_groups, UB_ALIGN // 4) * (UB_ALIGN // 4)

    @property
    def scale_stride(self) -> int:
        """scale 的物理行跨度，以字节计；公开输出仅保留 qr_groups 列。"""
        return _ceil_div(self.qr_groups, UB_ALIGN) * UB_ALIGN

    @property
    def m_tiles_a1(self) -> int:
        """投影阶段沿 token 轴的分块数量。"""
        return _ceil_div(self.token_count, self.bm_a1)

    @property
    def n_tiles_qa(self) -> int:
        return self.rank_size // self.bn_a1

    @property
    def n_tiles_kv(self) -> int:
        return self.head_size // self.bn_a1

    @property
    def tiles_a1_qa(self) -> int:
        return self.m_tiles_a1 * self.n_tiles_qa * self.split_k

    @property
    def tiles_a1_kv(self) -> int:
        return self.m_tiles_a1 * self.n_tiles_kv * self.split_k

    @property
    def tiles_a2(self) -> int:
        """归约阶段的分块数量，每块分配给一个 Vector 核。"""
        return _ceil_div(self.token_count, self.bm_a2)

    @property
    def b_cols(self) -> int:
        """每个 Vector 核负责的输出特征列数（FixPipe 沿特征轴二分）。"""
        if self.b_resident:
            return self.bn_b // FIXPIPE_DUAL_DST
        return self.head_size - self.head_size // FIXPIPE_DUAL_DST

    @property
    def use_branch_pipeline(self) -> bool:
        """限制拆分 QA/KV 任务队列所引入的额外 Cube 波次。

        拆分可使 KV Cube 与 QR Vector 重叠，但额外 Cube 波次不得超过 5%。
        该选择只依赖任务几何和核数，不依赖 token 长度白名单。
        """
        shared = _ceil_div(self.tiles_a1_qa + self.tiles_a1_kv, self.block_dim)
        separate = _ceil_div(self.tiles_a1_qa, self.block_dim) + _ceil_div(
            self.tiles_a1_kv, self.block_dim
        )
        return 100 * (separate - shared) <= 5 * shared

    @property
    def min_b_iterations(self) -> int:
        """每核 QB 任务数的下界，用于省略不需要的 KV 尾部循环。"""
        if self.b_resident:
            head_tiles = self.head_count * (self.head_size // self.bn_b)
            return self.owners_b * (head_tiles // self.block_dim)
        return self.tiles_b // self.block_dim

    @property
    def owners_b(self) -> int:
        return _ceil_div(self.token_count, self.bm_b)

    @property
    def tiles_b(self) -> int:
        if self.b_resident:
            return self.owners_b * self.head_count * (self.head_size // self.bn_b)
        return self.owners_b * self.head_count

    @property
    def resident_tiles(self):
        return max(self.hidden_size // self.split_k, self.rank_size) // self.bk_a1

    @property
    def scale_k_tiles(self):
        qa_reduction_tiles = self.hidden_size // self.split_k // self.bk_a1
        qb_reduction_tiles = self.rank_size // self.bk_b
        # 4：scale 暂存最多合并的 K 分块数（可整除时启用）。
        return (
            4 if qa_reduction_tiles % 4 == 0 and qb_reduction_tiles % 4 == 0 else W_BUF
        )

    @property
    def resident_scale_tiles(self):
        return self.resident_tiles // self.scale_k_tiles

    @property
    def groups_a(self):
        k = self.hidden_size // self.split_k
        return _resident_groups(
            self.m_tiles_a1 * self.split_k,
            self.n_tiles_qa + self.n_tiles_kv,
            self.block_dim,
            self.bn_a1 * k,
            self.bm_a1 * k,
        )

    @property
    def groups_b(self):
        return _resident_groups(
            self.owners_b,
            self.head_count,
            self.block_dim,
            self.head_size * self.rank_size,
            self.bm_b * self.rank_size,
        )

    # 分块规划
    @classmethod
    def plan(
        cls,
        token_count,
        hidden_size,
        rank_size,
        head_count,
        head_size,
        rope_width,
        l1_reuse_eligible=False,
    ) -> AttnPrologueTiling:
        """为已通过公共校验的输入选择静态分块。调用方负责公共几何约束和 L1 复用资格；本方法按容量及 NZ 网格检查候选。"""
        cores = get_platform_info().cube_core_num
        # ---- 阶段 1：驻留规划（L1 复用资格时优先） ----
        if l1_reuse_eligible:
            # 使用对齐的完整物理分块平衡 token 任务。候选由实际容量决定，
            # 不依赖具体机器的核数或型号。
            # 96：驻留 token 分块行数上限。
            for limit in range(
                min(96, _align(token_count, TOKEN_TILE_ALIGN)),
                TOKEN_TILE_ALIGN - 1,
                -TOKEN_TILE_ALIGN,
            ):
                owners = _ceil_div(token_count, limit)
                token_tile_rows = _align(
                    _ceil_div(token_count, owners), TOKEN_TILE_ALIGN
                )
                tiling = cls(
                    token_count,
                    hidden_size,
                    rank_size,
                    head_count,
                    head_size,
                    rope_width,
                    cores,
                    "split_k",
                    8,  # 8：L1 复用流水线沿 K 的固定分块数。
                    token_tile_rows,
                    RESIDENT_BN,
                    K_WINDOW_RESIDENT,
                    K_WINDOW_RESIDENT,
                    1,
                    token_tile_rows,
                    RESIDENT_BN,
                    K_WINDOW_RESIDENT,
                    K_WINDOW_RESIDENT,
                    l1_reuse=True,
                )
                try:
                    tiling._check_capacity(
                        get_mem_size("l1"),
                        get_mem_size("ub"),
                        get_mem_size("l0c") - L0C_RESERVED_BYTES,
                        get_mem_size("l0b") // L0_DEPTH,
                    )
                    tiling._check_nz_grid()
                    return tiling
                except ValueError:
                    pass
            # 驻留分块容量不足时，退回通用规划器。

        # ---- 阶段 2：通用容量预算 ----
        l0b_cap = get_mem_size("l0b") // L0_DEPTH
        l0c_cap = get_mem_size("l0c")
        l1_cap = get_mem_size("l1")
        ub_cap = get_mem_size("ub")
        # 为 FixPipe 的 unit flag 窗口预留 16 KiB L0C 空间。
        l0c_budget = l0c_cap - L0C_RESERVED_BYTES

        def pick_k(reduction: int, output_tile_cols: int) -> tuple[int, int]:
            # A1 与 B 的静态缓冲共存，各使用一半 L0B 预算。
            k_l1 = _largest_divisor(reduction, min(K_L1_MAX, reduction), MX_PAIR_WIDTH)
            reduction_tile_cols = _largest_divisor(
                k_l1, min(BK_MAX, k_l1, l0b_cap // 2 // output_tile_cols), MX_PAIR_WIDTH
            )
            return k_l1, reduction_tile_cols

        # ---- 阶段 3：模板与 split-K 选择 ----
        # 按 token 数选择模板，再按设备容量调整具体分块。
        template = "split_k" if token_count <= DECODE_TOKEN_LIMIT else "split_t"
        # 20：大 T 沿 K 的分块数。
        split_k = 20 if template == "split_t" and token_count >= LONG_TOKEN_LIMIT else 1
        if template == "split_k":
            if token_count > DECODE_SPLIT_K_TOKENS:
                split_k = SPLIT_K_DECODE
            else:
                # (2, 4, 8)：小 T decode 的候选 K 分块数。
                for candidate in (2, 4, 8):
                    if (
                        hidden_size % (candidate * MX_PAIR_WIDTH) == 0
                        and head_size // MX_PAIR_WIDTH * candidate >= cores
                    ):
                        split_k = candidate
                        # 在保证并行度的同时，限制 FP32 部分和的归约长度。
                        # 640：split-K 单分块归约长度上限。
                        if hidden_size // candidate <= 640:
                            break

        # ---- 阶段 4：A1 输出列与 K 窗口 ----
        # A1 任务由 token 分块、输出列分块和 K 分块组成，NZ 列保持对齐。
        # 64/128：bn_a1 搜索上限（split_k / split_t）。
        bn_a1 = _largest_divisor(
            math.gcd(rank_size, head_size),
            64 if template == "split_k" else 128,
            FRACTAL_N,
        )
        k_l1_a1, bk_a1 = pick_k(hidden_size // split_k, bn_a1)

        # A2 使用有界的 2 的幂次分块，使偏移可整除补齐后的交换区行数。
        bm_a2 = 1 if token_count <= DECODE_TOKEN_LIMIT or split_k > 1 else BM_A2_PREFILL

        # ---- 阶段 5：QB 输出列分块三分支 ----
        # QB 并行处理相互独立的 token 分块与 head 任务。
        if template == "split_t" and head_size % BN_B_FULL_HEAD == 0:
            # 完整 head 分块使 B 阶段的 QR/scale 读取次数减半。
            # L0B 使用单槽，使 A1 与 B 的合计占用保持在 64 KiB 内。
            bn_b, k_l1_b, bk_b = BN_B_FULL_HEAD, K_WINDOW_RESIDENT, MX_PAIR_WIDTH
        elif template == "split_k" and (
            # 8/96：驻留半头资格的 decode token 数区间（单 token 亦可）。
            token_count == 1 or 8 <= token_count <= 96
        ):
            # 先选择完整 head 的保底分块，再由容量检查决定是否使用驻留半头。
            # 256：decode 半头的 K1 窗口宽度。
            bn_b, k_l1_b, bk_b = BN_B_FULL_HEAD, 256, MX_PAIR_WIDTH
        else:
            # 256：通用路径 bn_b 搜索上限。
            bn_b = _largest_divisor(head_size, 256, FRACTAL_N)
            k_l1_b, bk_b = pick_k(rank_size, bn_b)
        # ---- 阶段 6：行分块容量回退搜索 ----
        # 所有 token 分块均为 16 的倍数，容量限制不依赖 T。
        # 输入尾块按物理分块补齐，输出与 cache 写回仍使用逻辑 T。
        # A1 不使用 UB，B 则在 UB 中保存完整 FixPipe 结果及其 BF16 输出。
        # 两个阶段独立选择行分块：容量不足时先缩小 UB 占用较大的 B，
        # 在 L1/L0 容量允许时保留更大的 A1 分块，避免重复加载权重。
        rounded_tokens = max(TOKEN_TILE_ALIGN, _align(token_count, TOKEN_TILE_ALIGN))
        if template == "split_t":
            # 平衡三个阶段的片上容量：缩小 A1/A2 可为 B 保留 96 行，
            # 减少主要开销 Wqb 扫描的重复次数。
            bm_a1 = min(BM_SPLIT_T_MAX, rounded_tokens)
            bm_b = min(BM_SPLIT_T_MAX, rounded_tokens)
        else:
            bm_a1 = min(BM_A1_MAX, rounded_tokens)
            bm_b = min(BM_A1_MAX, rounded_tokens)
        while True:
            tiling = cls(
                token_count=token_count,
                hidden_size=hidden_size,
                rank_size=rank_size,
                head_count=head_count,
                head_size=head_size,
                rope_width=rope_width,
                template=template,
                split_k=split_k,
                # 启动网格使用平台提供的 Cube 核数。
                block_dim=cores,
                bm_a1=bm_a1,
                bn_a1=bn_a1,
                bk_a1=bk_a1,
                k_l1_a1=k_l1_a1,
                bm_a2=bm_a2,
                bm_b=bm_b,
                bn_b=bn_b,
                bk_b=bk_b,
                k_l1_b=k_l1_b,
            )
            try:
                tiling._check_capacity(l1_cap, ub_cap, l0c_budget, l0b_cap)
            except ValueError:
                if bm_b > 16:
                    bm_b = max(16, (bm_b // 2 // TOKEN_TILE_ALIGN) * TOKEN_TILE_ALIGN)
                    continue
                if bm_a1 == 16:
                    raise
                bm_a1 = max(16, (bm_a1 // 2 // TOKEN_TILE_ALIGN) * TOKEN_TILE_ALIGN)
                continue
            break

        # ---- 阶段 7：decode 列分块增强与专属分块 ----
        # decode 保持基础 token 分块不变，再在实际 L1/L0 容量内增大 QA/KV
        # 共用的输出列分块。更大的输出块可减少对同一激活块的重复读取。
        # 该搜索位于基础行分块选择之后，避免增大输出块时隐式缩小行块。
        if template == "split_k":
            for candidate_bn in range(
                math.gcd(rank_size, head_size), FRACTAL_N - 1, -FRACTAL_N
            ):
                if rank_size % candidate_bn or head_size % candidate_bn:
                    continue
                try:
                    candidate_k_l1, candidate_bk = pick_k(
                        hidden_size // split_k, candidate_bn
                    )
                    candidate = dataclasses.replace(
                        tiling,
                        bn_a1=candidate_bn,
                        bk_a1=candidate_bk,
                        k_l1_a1=candidate_k_l1,
                    )
                    candidate._check_capacity(l1_cap, ub_cap, l0c_budget, l0b_cap)
                    candidate._check_nz_grid()
                except (ValueError, NotImplementedError):
                    continue
                tiling = candidate
                break

        if template == "split_k" and token_count > DECODE_SPLIT_K_TOKENS:
            candidate_k_l1 = _largest_divisor(
                hidden_size // split_k,
                min(K_L1_MAX, hidden_size // split_k),
                MX_PAIR_WIDTH,
            )
            # 112/256：token>128 decode 的专属行/列分块。
            candidate = dataclasses.replace(
                tiling,
                bm_a1=112,
                bn_a1=256,
                bk_a1=MX_PAIR_WIDTH,
                k_l1_a1=candidate_k_l1,
            )
            candidate._check_capacity(l1_cap, ub_cap, l0c_budget, l0b_cap)
            candidate._check_nz_grid()
            tiling = candidate

        # ---- 阶段 8：搬运窗口选择 ----
        tiling = tiling._select_transfer_tiling(l1_cap, ub_cap, l0c_budget, l0b_cap)
        tiling._check_nz_grid()
        return tiling

    def _select_transfer_tiling(self, l1_cap, ub_cap, l0c_budget, l0b_cap):
        """选择搬运窗口，不通过缩小投影的 token 或输出分块换取空间。驻留 QB 跨 head 复用 QR 和 RoPE 行，并通过双槽预取权重。容量不足时退回基础布局。"""

        def fits(candidate):
            try:
                candidate._check_capacity(l1_cap, ub_cap, l0c_budget, l0b_cap)
                candidate._check_nz_grid()
                return True
            except (ValueError, NotImplementedError):
                return False

        base = self
        # decode 减少独立 K 部分和以降低 A2 搬运量，同时保留足够的
        # token、输出列和 K 分块任务并行度。
        if (
            self.template == "split_k"
            and self.token_count <= DECODE_SPLIT_K_TOKENS
            and self.hidden_size % (SPLIT_K_DECODE * MX_PAIR_WIDTH) == 0
        ):
            k_a = _largest_divisor(
                self.hidden_size // SPLIT_K_DECODE,
                min(K_L1_MAX, self.hidden_size // SPLIT_K_DECODE),
                MX_PAIR_WIDTH,
            )
            base = dataclasses.replace(
                base,
                split_k=SPLIT_K_DECODE,
                k_l1_a1=k_a,
                bk_a1=MX_PAIR_WIDTH,
                # token>32 时 A2 用 2 行分块。
                bm_a2=2 if self.token_count > 32 else 1,
            )
        # token 分块足够多时，无需再用二十个 K 分块提供并行度。
        if (
            self.template == "split_t"
            and self.token_count >= LONG_TOKEN_LIMIT
            and self.hidden_size % MX_PAIR_WIDTH == 0
        ):
            k_a = _largest_divisor(
                self.hidden_size, min(K_L1_MAX, self.hidden_size), MX_PAIR_WIDTH
            )
            base = dataclasses.replace(
                base,
                split_k=1,
                k_l1_a1=k_a,
                bk_a1=_largest_divisor(
                    k_a, min(BK_MAX, k_a, l0b_cap // 2 // base.bn_a1), MX_PAIR_WIDTH
                ),
                bm_a2=BM_A2_PREFILL,
            )
        # 每个 head 拆成两个列块，两个 Vector 半区都包含完整的 VL 分组。
        if (
            self.head_size % (HEAD_SLICES * VL) == 0
            and self.head_size // HEAD_SLICES >= self.rope_width
            and self.head_count * FIXPIPE_DUAL_DST >= self.block_dim
        ):
            k_window = _largest_divisor(
                self.rank_size, min(K_WINDOW_RESIDENT, self.rank_size), MX_PAIR_WIDTH
            )
            tried = set()
            # 在任务归属数相同的情况下，选择最小的对齐 token 分块。
            # 128：驻留 bm 搜索上限。
            for bm_limit in range(
                min(128, _align(self.token_count, TOKEN_TILE_ALIGN)),
                TOKEN_TILE_ALIGN - 1,
                -TOKEN_TILE_ALIGN,
            ):
                owners = _ceil_div(self.token_count, bm_limit)
                token_tile_rows = _align(
                    _ceil_div(self.token_count, owners), TOKEN_TILE_ALIGN
                )
                if token_tile_rows in tried:
                    continue
                tried.add(token_tile_rows)
                for a_window in (True, False):
                    candidate = dataclasses.replace(
                        base,
                        b_resident=True,
                        a1_window=a_window,
                        bn_b=self.head_size // FIXPIPE_DUAL_DST,
                        bk_b=MX_PAIR_WIDTH,
                        k_l1_b=k_window,
                        bm_b=token_tile_rows,
                    )
                    if fits(candidate):
                        return candidate
        candidate = dataclasses.replace(base, a1_window=True)
        if fits(candidate):
            return candidate
        return base if fits(base) else self

    def _check_nz_grid(self) -> None:
        """要求完整的 NZ fractal：输出行和归约列均按网格对齐。"""
        for name, reduction, window, block in (
            ("A1", self.hidden_size // self.split_k, self.k_l1_a1, self.bk_a1),
            ("B", self.rank_size, self.k_l1_b, self.bk_b),
        ):
            if reduction % window or window % block:
                raise ValueError(
                    f"Invalid {name} tiling: K={reduction}, L1={window}, L0={block} must divide exactly"
                )
        for name, extent, align in (
            ("bn_a1 (QA/KV NZ output rows)", self.bn_a1, NZ_M_FRAC),
            ("k_l1_a1 (QA/KV NZ reduction columns)", self.k_l1_a1, NZ_C0_1B),
            ("bn_b (QB NZ output rows)", self.bn_b, NZ_M_FRAC),
            ("k_l1_b (QB NZ reduction columns)", self.k_l1_b, NZ_C0_1B),
            ("rank_size (QA output width)", self.rank_size, NZ_M_FRAC),
            ("head_size (KV output width)", self.head_size, NZ_M_FRAC),
        ):
            if extent % align:
                raise NotImplementedError(
                    f"NZ weights require {name} divisible by {align}, got {extent}"
                )

    def _check_capacity(self, l1_cap, ub_cap, l0c_budget, l0b_cap) -> None:
        """检查所有同时存在的片上缓冲区是否超过设备容量。"""
        rank_width, head_width = self.rank_size, self.head_size
        # ---- 阶段 1：驻留路径容量检查（l1_reuse） ----
        if self.l1_reuse:
            # A1 与 B 复用同一组 L1/L0 Channel；容量还要计入预取的 B panel、
            # A2 同时加载的全部 split-K 部分和以及 Channel 标识数。
            token_tile_rows, output_tile_cols, reduction_tile_cols = (
                self.bm_a1,
                self.bn_a1,
                self.bk_a1,
            )
            # 驻留 QB scale：每行字节按 16 对齐，行数按 32 对齐。
            scale_bytes = (
                _ceil_div(rank_width // MX_GROUP, 16)
                * 16
                * _ceil_div(head_width, 32)
                * 32
            )
            ub_a2 = (
                (self.split_k + 1) * (rank_width + head_width) * FP32_BYTES
                + (rank_width + head_width) * FP32_BYTES
                + 2 * VL * FP32_BYTES
                + 2 * rank_width
                + 2 * max(self.exp_stride, VL) * FP32_BYTES
                + 2 * self.scale_stride
                + 2 * head_width
                + VL * FP32_BYTES
            )
            # 6 = FP32 累加 + BF16 输出的每元素字节数。
            ub_b = (
                2
                * (token_tile_rows // FIXPIPE_DUAL_DST)
                * (head_width // FIXPIPE_DUAL_DST)
                * 6
                + 2 * (token_tile_rows // FIXPIPE_DUAL_DST) * VL * FP32_BYTES
                + VL * FP32_BYTES
            )
            checks = [
                # 每 MX_GROUP 个 payload 字节另需一个 scale 字节。
                (
                    "shared L1",
                    (self.resident_tiles * token_tile_rows + 2 * output_tile_cols)
                    * reduction_tile_cols
                    * (MX_GROUP + 1)
                    // MX_GROUP
                    + scale_bytes
                    # 驻留权重 scale 的 zn 分形展开（16=行 fractal）暂按 16 行网格估算。
                    + 16 * output_tile_cols * reduction_tile_cols,
                    l1_cap,
                ),
                (
                    "shared L0A",
                    L0_DEPTH * token_tile_rows * reduction_tile_cols,
                    get_mem_size("l0a"),
                ),
                (
                    "shared L0B",
                    L0_DEPTH * output_tile_cols * reduction_tile_cols,
                    l0b_cap * L0_DEPTH,
                ),
                (
                    "L0C",
                    token_tile_rows * (output_tile_cols + head_width) * FP32_BYTES,
                    l0c_budget,
                ),
                ("UB", ub_a2 + ub_b, ub_cap),
                (
                    "Cube channel IDs",
                    self.resident_tiles
                    + self.resident_scale_tiles
                    + 2 * W_BUF
                    + 2 * L0_DEPTH
                    # +4/+8：early panel、FixPipe 目的等保留 Channel 预算修正项。
                    + 4
                    + 8,
                    # 32：单 kernel 可用的 Channel 标识上限。
                    32,
                ),
            ]
            for name, used, cap in checks:
                if used > cap:
                    raise ValueError(f"Buffer {name}: {used} exceeds {cap}")
            return

        # ---- 阶段 2：L1 字节建模 ----
        def l1_buffer_bytes(
            token_tile_rows, output_tile_cols, k_a, k_l1, weight_buffers=W_BUF
        ):
            # 激活的 K 宽度可为一个 L0 分块或完整 L1 窗口。
            # 权重 payload 使用单槽，以保证 FP8 别名的起始地址在编译期可知。
            return L0_DEPTH * (
                token_tile_rows * k_a + token_tile_rows * (k_a // MX_GROUP)
            ) + weight_buffers * (
                output_tile_cols * k_l1 + (k_l1 // MX_GROUP) * output_tile_cols
            )

        # ---- 阶段 3：L1/L0B/L0A/L0C 检查表 ----
        checks = [
            # 各阶段缓冲在同一个 kernel 内静态共存，同步不会释放其空间。
            (
                "L1 (stage A1 + B)",
                l1_buffer_bytes(self.bm_a1, self.bn_a1, self.k_l1_x_a1, self.k_l1_a1)
                + (
                    self.bm_b * (rank_width + rank_width // MX_GROUP)
                    + 2
                    * (self.bn_b * self.k_l1_b + (self.k_l1_b // MX_GROUP) * self.bn_b)
                    if self.b_resident
                    else l1_buffer_bytes(self.bm_b, self.bn_b, self.bk_b, self.k_l1_b)
                ),
                l1_cap,
            ),
            (
                "L0B (stage A1 + B)",
                self.bn_a1 * self.bk_a1 * L0_DEPTH
                + self.bn_b
                * self.bk_b
                * (1 if self.bn_b == BN_B_FULL_HEAD else L0_DEPTH),
                l0b_cap * L0_DEPTH,
            ),
            (
                "L0A (stage A1 + B)",
                (self.bm_a1 * self.bk_a1 + self.bm_b * self.bk_b) * L0_DEPTH,
                l0b_cap * L0_DEPTH,
            ),
            # 为 FixPipe 的 unit flag 保留余量，不能占满全部 L0C。
            (
                "L0C (stage A1 + B)",
                self.bm_a1 * self.bn_a1 * FP32_BYTES
                + self.bm_b
                * (self.bn_b if self.b_resident else head_width)
                * FP32_BYTES,
                l0c_budget,
            ),
        ]
        # ---- 阶段 4：A2/B UB 建模与总检查 ----
        m2 = self.bm_a2
        a2_depth = (
            A2_BUF_DEPTH_PREFILL if self.template == "split_t" else A2_BUF_DEPTH_DECODE
        )
        # 归约只由 Vector 执行，每个 DMA 分块包含 bm_a2 个 token 行。
        ub_a2 = (
            a2_depth * m2 * rank_width * FP32_BYTES  # FP32 QA 交换区行。
            + a2_depth * m2 * head_width * FP32_BYTES  # KV 行。
            + rank_width * FP32_BYTES
            + head_width * FP32_BYTES  # 归一化权重 gamma。
            + 2 * (m2 * VL) * FP32_BYTES  # RoPE 正弦和余弦表。
            + a2_depth * m2 * rank_width  # QR payload 字节。
            + (m2 + 1) * max(self.exp_stride, VL) * FP32_BYTES
            + a2_depth * m2 * self.scale_stride
            + 2 * head_width
            + VL * FP32_BYTES
        )  # 双缓冲 KV 字节及 RoPE 暂存区。
        if self.split_k > 1:
            # QA 部分和、Kahan 补偿及 KV 部分和同时占用 UB。
            ub_a2 += m2 * (2 * rank_width + head_width) * FP32_BYTES
        # 投影结果从 L0C 直接写入 GM；QB 沿特征轴分给两个 Vector 核。
        ub_b = (
            self.bm_b * self.b_cols * FP32_BYTES  # FixPipe 目的缓冲区。
            + self.bm_b * self.b_cols * 2  # BF16 输出缓冲区。
            + 2 * self.bm_b * VL * FP32_BYTES  # RoPE 正弦和余弦表。
            + VL * FP32_BYTES
        )  # RoPE 暂存区。
        checks += [("UB (stage A2 + B)", ub_a2 + ub_b, ub_cap)]
        for name, used, cap in checks:
            if used > cap:
                raise ValueError(
                    f"Buffer {name} requires {used} bytes, exceeding {cap}; reduce its token/output tile"
                )


def _pick_bn_a1(rank: int, head: int, hidden: int, tokens: int, cores: int) -> int:
    """选择使最繁忙调度波次搬运量最小的输出轴分块。每个 QA/KV 波次读取自己的权重切片及完整的激活分块。"""
    best, best_cost = None, None
    limit = math.gcd(rank, head)
    for output_tile_cols in range(NZ_M_FRAC, limit + 1, NZ_M_FRAC):
        if rank % output_tile_cols or head % output_tile_cols:
            continue
        waves = _ceil_div(rank // output_tile_cols, cores) + _ceil_div(
            head // output_tile_cols, cores
        )
        cost = waves * (output_tile_cols * hidden + tokens * hidden)
        if best_cost is None or cost < best_cost:
            best, best_cost = output_tile_cols, cost
    if best is None:
        raise ValueError(
            f"No 16-aligned output tile divides both R={rank} and D={head}"
        )
    return best


def _balance_m(cap: int, total: int, predicate) -> int:
    """选择使两个 Vector 核都能获得非空尾块的 token 分块。
    FixPipe 在静态分块中点切分；第二个目的地为空时不会触发 Channel 标志，
    因此尾块行数必须超过该中点。单行分块使用框架支持的单目的地路径。"""

    def safe(token_tile_rows: int) -> bool:
        if token_tile_rows == 1:
            return True
        tail = total - (_ceil_div(total, token_tile_rows) - 1) * token_tile_rows
        return tail > token_tile_rows - token_tile_rows // 2

    for token_tile_rows in range(
        _halve_until_tile_fits(min(cap, total), predicate), 0, -1
    ):
        if predicate(token_tile_rows) and safe(token_tile_rows):
            return token_tile_rows
    raise ValueError("No token tile fits capacity without an empty Vector partition")


def _halve_until_tile_fits(start: int, predicate) -> int:
    """逐次将 token 分块减半，直到满足容量检查条件。"""
    value = start
    while value > 1 and not predicate(value):
        value //= 2
    if not predicate(value):
        raise ValueError("Even a one-row token tile exceeds capacity")
    return value


@kernel
class AttnPrologueKernel:
    """通过一次 kernel 启动完成整个 attention prologue。"""

    def __init__(self, tiling: AttnPrologueTiling):
        self.tiling = tiling

    # kernel 入口
    def __call__(
        self,
        gm_x,
        gm_dsx,
        gm_wqa,
        gm_dswqa,
        gm_wqb,
        gm_dswqb,
        gm_wkv,
        gm_dswkv,
        gm_gamma_qr,
        gm_gamma_kv,
        gm_sin,
        gm_cos,
        gm_index,
        gm_cache,
        gm_q,
        gm_qr,
        gm_dqr2d,
        gm_dqr3d,
        gm_ws_qa,
        gm_ws_kv,
        norm_eps: dtypes.float32,
    ):
        # Vector 归约前发布 Cube 部分和，Cube 执行 QB 前发布 Vector 的 QR。
        if const_expr(self.tiling.l1_reuse):
            self._shared_pipeline(
                gm_x,
                gm_dsx,
                gm_wqa,
                gm_dswqa,
                gm_wqb,
                gm_dswqb,
                gm_wkv,
                gm_dswkv,
                gm_gamma_qr,
                gm_gamma_kv,
                gm_sin,
                gm_cos,
                gm_index,
                gm_cache,
                gm_q,
                gm_qr,
                gm_dqr2d,
                gm_dqr3d,
                gm_ws_qa,
                gm_ws_kv,
                norm_eps,
            )
        elif const_expr(self.tiling.use_branch_pipeline):
            self._a1_branch(
                gm_x,
                gm_dsx,
                gm_wqa,
                gm_dswqa,
                gm_wkv,
                gm_dswkv,
                gm_ws_qa,
                gm_ws_kv,
                gm_gamma_qr,
                gm_index,
                gm_qr,
                gm_dqr2d,
                norm_eps,
            )
            # Vector 完成 QA 后处理时预取第一个 QB 权重窗口。
            # KV 后处理与 QB 结果消费交错执行。
            if const_expr(self.tiling.b_resident):
                prefetched_qb_weights = RingBuffer(
                    MemLoc.L1,
                    dtypes.int8,
                    make_layout((self.tiling.bn_b, self.tiling.k_l1_b)),
                    depth=1,
                    data_format="nz",
                )
                prefetched_qb_scales = RingBuffer(
                    MemLoc.L1,
                    FLOAT8_E8M0,
                    make_layout((self.tiling.k_l1_b // MX_GROUP, self.tiling.bn_b)),
                    depth=2,
                    data_format="nz",
                )
                self._fill_w_b(
                    prefetched_qb_weights,
                    prefetched_qb_scales,
                    gm_wqb,
                    gm_dswqb,
                    get_block_idx(),
                    0,
                    self.tiling.k_l1_b // MX_PAIR_WIDTH,
                )
                self._stage_b(
                    gm_qr,
                    gm_dqr3d,
                    gm_wqb,
                    gm_dswqb,
                    gm_sin,
                    gm_cos,
                    gm_q,
                    prefetched_qb_weights,
                    prefetched_qb_scales,
                    gm_gamma_kv,
                    gm_index,
                    gm_cache,
                    gm_ws_kv,
                    norm_eps,
                )
            else:
                self._stage_b(
                    gm_qr,
                    gm_dqr3d,
                    gm_wqb,
                    gm_dswqb,
                    gm_sin,
                    gm_cos,
                    gm_q,
                    None,
                    None,
                    gm_gamma_kv,
                    gm_index,
                    gm_cache,
                    gm_ws_kv,
                    norm_eps,
                )
        else:
            self._project_qa_kv(
                gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_wkv, gm_dswkv, gm_ws_qa, gm_ws_kv
            )
            self._publish_a1()
            # Vector 完成 QA 后处理时预取第一个 QB 权重窗口。
            # KV 后处理与 QB 结果消费交错执行。
            if const_expr(self.tiling.b_resident):
                prefetched_qb_weights = RingBuffer(
                    MemLoc.L1,
                    dtypes.int8,
                    make_layout((self.tiling.bn_b, self.tiling.k_l1_b)),
                    depth=1,
                    data_format="nz",
                )
                prefetched_qb_scales = RingBuffer(
                    MemLoc.L1,
                    FLOAT8_E8M0,
                    make_layout((self.tiling.k_l1_b // MX_GROUP, self.tiling.bn_b)),
                    depth=2,
                    data_format="nz",
                )
                self._fill_w_b(
                    prefetched_qb_weights,
                    prefetched_qb_scales,
                    gm_wqb,
                    gm_dswqb,
                    get_block_idx(),
                    0,
                    self.tiling.k_l1_b // MX_PAIR_WIDTH,
                )
            self._reduce_normalize_a(
                gm_gamma_qr,
                gm_gamma_kv,
                gm_sin,
                gm_cos,
                gm_index,
                gm_cache,
                gm_qr,
                gm_dqr2d,
                gm_ws_qa,
                gm_ws_kv,
                norm_eps,
            )
            if const_expr(self.tiling.b_resident):
                self._stage_b(
                    gm_qr,
                    gm_dqr3d,
                    gm_wqb,
                    gm_dswqb,
                    gm_sin,
                    gm_cos,
                    gm_q,
                    prefetched_qb_weights,
                    prefetched_qb_scales,
                    gm_gamma_kv,
                    gm_index,
                    gm_cache,
                    gm_ws_kv,
                    norm_eps,
                )
            else:
                self._stage_b(
                    gm_qr,
                    gm_dqr3d,
                    gm_wqb,
                    gm_dswqb,
                    gm_sin,
                    gm_cos,
                    gm_q,
                    None,
                    None,
                    gm_gamma_kv,
                    gm_index,
                    gm_cache,
                    gm_ws_kv,
                    norm_eps,
                )

    @jit
    def _shared_pipeline(
        self,
        gm_x,
        gm_dsx,
        gm_wqa,
        gm_dswqa,
        gm_wqb,
        gm_dswqb,
        gm_wkv,
        gm_dswkv,
        gm_gamma_qr,
        gm_gamma_kv,
        gm_sin,
        gm_cos,
        gm_index,
        gm_cache,
        gm_q,
        gm_qr,
        gm_dqr2d,
        gm_dqr3d,
        gm_ws_qa,
        gm_ws_kv,
        norm_eps: dtypes.float32,
    ):
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (
            tiling.bm_a1,
            tiling.bn_a1,
            tiling.bk_a1,
        )
        panels_per_head = tiling.head_size // (PANEL_OUTPUT_TILES * output_tile_cols)
        l1_x = [
            RingBuffer(
                MemLoc.L1,
                dtypes.float8_e4m3fn,
                make_layout((token_tile_rows, reduction_tile_cols)),
                depth=1,
                data_format="nz",
            )
            for _ in range(tiling.resident_tiles)
        ]
        # E8M0 scale 的 zn 分形（(M1, G1, 16, 2)、每分形 32 字节）不能用
        # 每次选槽时用 reinterpret 绑定 zn 视图。
        l1_sx = [
            RingBuffer(
                MemLoc.L1,
                dtypes.uint8,
                make_layout(
                    (
                        _ceil_div(token_tile_rows, NZ_M_FRAC)
                        * _ceil_div(
                            tiling.scale_k_tiles * reduction_tile_cols // MX_GROUP,
                            MX_PAIR,
                        )
                        * ZN_FRACTAL_BYTES,
                    )
                ),
                depth=1,
                data_format="nd",
            )
            for _ in range(tiling.resident_scale_tiles)
        ]
        l1_w = [
            RingBuffer(
                MemLoc.L1,
                dtypes.int8,
                make_layout((output_tile_cols, reduction_tile_cols)),
                depth=1,
                data_format="nz",
            )
            for _ in range(W_BUF)
        ]
        l1_sw = RingBuffer(
            MemLoc.L1,
            dtypes.float8_e8m0,
            make_layout((reduction_tile_cols // MX_GROUP, output_tile_cols)),
            depth=W_BUF,
            data_format="nz",
        )
        l0a = [
            RingBuffer(
                MemLoc.L0A,
                dtypes.float8_e4m3fn,
                make_layout((token_tile_rows, reduction_tile_cols)),
                depth=1,
            )
            for _ in range(L0_DEPTH)
        ]
        l0b = RingBuffer(
            MemLoc.L0B,
            dtypes.float8_e4m3fn,
            make_layout((output_tile_cols, reduction_tile_cols)),
            depth=L0_DEPTH,
        )
        l0c_a = Buffer(
            MemLoc.L0C, dtypes.float32, make_layout((token_tile_rows, output_tile_cols))
        )
        # A 与 B 复用相同的物理 Channel，因此此部分容量按较大者计算。
        self._resident_a_joint(
            l1_x,
            l1_sx,
            l1_w,
            l1_sw,
            l0a,
            l0b,
            l0c_a,
            gm_x,
            gm_dsx,
            gm_wqa,
            gm_dswqa,
            gm_ws_qa,
            gm_wkv,
            gm_dswkv,
            gm_ws_kv,
        )
        # 发布 A1 交换区后，可在 AIV 归约期间加载 B panel。
        early_s = [
            RingBuffer(
                MemLoc.L1,
                dtypes.float8_e8m0,
                make_layout((reduction_tile_cols // MX_GROUP, tiling.head_size)),
                depth=1,
                data_format="nz",
            )
            for _ in range(RESIDENT_K_WINDOWS)
        ]
        early_w = [
            RingBuffer(
                MemLoc.L1,
                dtypes.int8,
                make_layout(
                    (PANEL_OUTPUT_TILES * output_tile_cols, reduction_tile_cols)
                ),
                depth=1,
                data_format="nz",
            )
            for _ in range(RESIDENT_K_WINDOWS)
        ]
        cube_sync_all()
        vec_sync_all()
        vec_sync_block_arrive(PIPE.MTE3, 0, mode=2)
        cube_sync_block_wait(PIPE.S, 0, mode=2)
        cube_sync_block_arrive(PIPE.FIXPIPE, 1, mode=0)
        if get_block_idx() < tiling.owners_b * tiling.groups_b:
            head = get_block_idx() % tiling.groups_b
            mem_copy(
                early_s[0].next_write(),
                tile_slice(
                    gm_dswqb,
                    (tiling.head_size, reduction_tile_cols // MX_PAIR_WIDTH, MX_PAIR),
                    (head, 0, 0),
                )
                .permute(1, 2, 0)
                .flatten(0, 1),
            )
            self._load_b_panel(
                early_w[0].next_write(),
                gm_wqb,
                head * panels_per_head,
                0,
            )
        cube_sync_block_wait(PIPE.S, 1, mode=0)
        vec_sync_block_arrive(PIPE.MTE3, 1, mode=0)
        vec_sync_block_wait(PIPE.S, 1, mode=0)
        cube_sync_block_arrive(PIPE.MTE3, 2, mode=2)
        vec_sync_block_wait(PIPE.S, 2, mode=2)
        if get_block_idx() < tiling.owners_b * tiling.groups_b:
            for u in range_constexpr(1, RESIDENT_K_WINDOWS):
                mem_copy(
                    early_s[u].next_write(),
                    tile_slice(
                        gm_dswqb,
                        (
                            tiling.head_size,
                            reduction_tile_cols // MX_PAIR_WIDTH,
                            MX_PAIR,
                        ),
                        (get_block_idx() % tiling.groups_b, u, 0),
                    )
                    .permute(1, 2, 0)
                    .flatten(0, 1),
                )
                self._load_b_panel(
                    early_w[u].next_write(),
                    gm_wqb,
                    get_block_idx() % tiling.groups_b * panels_per_head,
                    u,
                )
        self._reduce_normalize_a(
            gm_gamma_qr,
            gm_gamma_kv,
            gm_sin,
            gm_cos,
            gm_index,
            gm_cache,
            gm_qr,
            gm_dqr2d,
            gm_ws_qa,
            gm_ws_kv,
            norm_eps,
        )
        self._stage_b_panels(
            l1_x,
            l1_sx,
            l0a,
            l0b,
            gm_qr,
            gm_dqr3d,
            gm_wqb,
            gm_dswqb,
            gm_sin,
            gm_cos,
            gm_q,
            early_w,
            early_s,
        )

    @jit
    def _resident_a_joint(
        self,
        l1_x,
        l1_sx,
        l1_w,
        l1_sw,
        l0a,
        l0b,
        l0c,
        gm_x,
        gm_dsx,
        gm_wqa,
        gm_dswqa,
        gm_ws_qa,
        gm_wkv,
        gm_dswkv,
        gm_ws_kv,
    ):
        tiling = self.tiling
        token_tile_rows, reduction_tile_cols = tiling.bm_a1, tiling.bk_a1
        nk = tiling.hidden_size // tiling.split_k // reduction_tile_cols
        # 同一个激活块服务于 QA 和 KV 的多个输出列任务。
        groups = tiling.groups_a
        n_count = tiling.n_tiles_qa + tiling.n_tiles_kv
        for job in range(
            get_block_idx(),
            tiling.m_tiles_a1 * tiling.split_k * groups,
            tiling.block_dim,
        ):
            token_tile_idx = job // (tiling.split_k * groups)
            split = job // groups % tiling.split_k
            group = job % groups
            held_x = []
            held_s = []
            for reduction_tile_idx in range_constexpr(nk):
                activation_l1_slot = l1_x[reduction_tile_idx].next_write()
                mem_copy(
                    activation_l1_slot,
                    tile_slice(
                        gm_x,
                        (token_tile_rows, reduction_tile_cols),
                        (token_tile_idx, split * nk + reduction_tile_idx),
                    ),
                )
                held_x.append(l1_x[reduction_tile_idx].next_read())
            for scale_tile_idx in range_constexpr(nk // tiling.scale_k_tiles):
                activation_scale_slot = reinterpret(
                    l1_sx[scale_tile_idx].next_write(),
                    dtype=FLOAT8_E8M0,
                    layout=make_layout(
                        (
                            token_tile_rows,
                            tiling.scale_k_tiles * reduction_tile_cols // MX_GROUP,
                        )
                    ),
                    data_format="zn",
                )
                mem_copy(
                    activation_scale_slot,
                    tile_slice(
                        gm_dsx,
                        (
                            token_tile_rows,
                            tiling.scale_k_tiles * reduction_tile_cols // MX_PAIR_WIDTH,
                            MX_PAIR,
                        ),
                        (
                            token_tile_idx,
                            split * (nk // tiling.scale_k_tiles) + scale_tile_idx,
                            0,
                        ),
                    ).flatten(1, 2),
                )
                held_s.append(
                    reinterpret(
                        l1_sx[scale_tile_idx].next_read(),
                        dtype=FLOAT8_E8M0,
                        layout=make_layout(
                            (
                                token_tile_rows,
                                tiling.scale_k_tiles * reduction_tile_cols // MX_GROUP,
                            )
                        ),
                        data_format="zn",
                    )
                )
            held_a = []
            for reduction_tile_idx in range_constexpr(nk):
                activation_l0_slot = l0a[reduction_tile_idx].next_write()
                mem_copy(
                    activation_l0_slot,
                    held_x[reduction_tile_idx],
                    copy_options(
                        mx_scale=tile_slice(
                            held_s[reduction_tile_idx // tiling.scale_k_tiles],
                            (token_tile_rows, reduction_tile_cols // MX_GROUP),
                            (0, reduction_tile_idx % tiling.scale_k_tiles),
                        )
                    ),
                )
                held_a.append(l0a[reduction_tile_idx].next_read())
            for output_tile_idx in range(group, n_count, groups):
                if output_tile_idx < tiling.n_tiles_qa:
                    self._resident_a_project(
                        l1_w,
                        l1_sw,
                        l0b,
                        l0c,
                        held_a,
                        gm_wqa,
                        gm_dswqa,
                        gm_ws_qa,
                        output_tile_idx,
                        token_tile_idx,
                        split,
                    )
                else:
                    self._resident_a_project(
                        l1_w,
                        l1_sw,
                        l0b,
                        l0c,
                        held_a,
                        gm_wkv,
                        gm_dswkv,
                        gm_ws_kv,
                        output_tile_idx - tiling.n_tiles_qa,
                        token_tile_idx,
                        split,
                    )

    @jit
    def _resident_a_project(
        self,
        l1_w,
        l1_sw,
        l0b,
        l0c,
        held_a,
        gm_w,
        gm_dsw,
        gm_ws,
        output_tile_idx,
        token_tile_idx,
        split,
    ):
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (
            tiling.bm_a1,
            tiling.bn_a1,
            tiling.bk_a1,
        )
        nk = tiling.hidden_size // tiling.split_k // reduction_tile_cols
        for pair in range(_ceil_div(nk, W_BUF)):
            for u in range_constexpr(min(W_BUF, nk)):
                reduction_tile_idx = pair * W_BUF + u
                self._fill_w_a(
                    l1_w[u],
                    l1_sw,
                    gm_w,
                    gm_dsw,
                    output_tile_idx,
                    split * nk + reduction_tile_idx,
                    reduction_tile_cols // MX_PAIR_WIDTH,
                )
            for u in range_constexpr(min(W_BUF, nk)):
                reduction_tile_idx = pair * W_BUF + u
                weight_l1_tile = l1_w[u].next_read().reinterpret(dtypes.float8_e4m3fn)
                weight_scale_tile = l1_sw.next_read()
                mem_copy(
                    l0b.next_write(),
                    weight_l1_tile,
                    copy_options(mx_scale=weight_scale_tile),
                )
                weight_l0_tile = l0b.next_read()
                matmul(l0c, held_a[u], weight_l0_tile, init=u == 0, mx_mode=True)
        #  mem_copy 以源域为准：将 L0C 源逻辑域重绑定到目的实际域，
        # 尾块裁剪生效且物理布局保持不变。
        ws_tile = tile_slice(
            gm_ws,
            (token_tile_rows, output_tile_cols),
            (split * tiling.m_tiles_a1 + token_tile_idx, output_tile_idx),
        )
        mem_copy(
            ws_tile,
            reinterpret(
                l0c,
                layout=make_layout(ws_tile.shape),
                physical_layout=l0c.physical_layout,
            ),
        )

    # 驻留路径保留既有的从左到右求和顺序；通用 QA 路径使用补偿求和。
    @jit
    def _acc_all_partials(self, dst, src, width):
        with vf(mode="simd"):
            # 1. 读取各特征向量，并累加 split-K 部分和。
            mask = rr.full_mask()
            for feature_vector_idx in range(width // VL):
                value = rr.vload(src, feature_vector_idx * VL)
                for part in range(1, self.tiling.split_k):
                    value = rr.vadd(
                        value,
                        rr.vload(
                            src, part * src.physical_stride[0] + feature_vector_idx * VL
                        ),
                        mask=mask,
                    )
                rr.vstore(dst, feature_vector_idx * VL, value, mask)

    @jit
    def _load_b_panel(self, dst, gm_weight, n_group, k_panel):
        tiling = self.tiling
        panel_bytes = PANEL_OUTPUT_TILES * tiling.bn_b * tiling.bk_b
        raw_dst = reinterpret(
            dst, layout=make_layout((1, panel_bytes)), data_format="nd"
        )
        mem_copy(
            raw_dst,
            tile_slice(
                gm_weight,
                (1, panel_bytes),
                (n_group * (tiling.rank_size // tiling.bk_b) + k_panel, 0),
            ),
        )

    @jit
    def _stage_b_panels(
        self,
        l1_x,
        l1_sx,
        l0a,
        l0b,
        gm_qr,
        gm_dqr,
        gm_w,
        gm_dsw,
        gm_sin,
        gm_cos,
        gm_q,
        early_w,
        early_s,
    ):
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (
            tiling.bm_b,
            tiling.bn_b,
            tiling.bk_b,
        )
        panels_per_head = tiling.head_size // (PANEL_OUTPUT_TILES * output_tile_cols)
        l0c = Buffer(
            MemLoc.L0C, dtypes.float32, make_layout((token_tile_rows, tiling.head_size))
        )
        half, feature_cols = (
            token_tile_rows // FIXPIPE_DUAL_DST,
            tiling.head_size // FIXPIPE_DUAL_DST,
        )
        ub_acc = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((half, feature_cols)),
            depth=2,
        )
        ub_out = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((half, feature_cols)), depth=2
        )
        ub_tail = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, VL)))
        ub_sin = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((half, VL)), depth=1)
        ub_cos = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((half, VL)), depth=1)
        groups = tiling.groups_b
        for job in range(get_block_idx(), tiling.owners_b * groups, tiling.block_dim):
            owner = job // groups
            group = job % groups
            # 首个任务已在 A2 期间预取；后续 token 分块需先完成前一个 head
            # 循环，再重新填充所有 panel Channel。
            if job >= tiling.block_dim:
                for u in range_constexpr(RESIDENT_K_WINDOWS):
                    mem_copy(
                        early_s[u].next_write(),
                        tile_slice(
                            gm_dsw,
                            (
                                tiling.head_size,
                                reduction_tile_cols // MX_PAIR_WIDTH,
                                MX_PAIR,
                            ),
                            (group, u, 0),
                        )
                        .permute(1, 2, 0)
                        .flatten(0, 1),
                    )
                    self._load_b_panel(
                        early_w[u].next_write(),
                        gm_w,
                        group * panels_per_head,
                        u,
                    )
            sin_slot = ub_sin.next_write()
            mem_copy(
                sin_slot,
                tile_slice(
                    gm_sin, (half, VL), (owner * AIV_PER_BLOCK + get_subblock_id(), 0)
                ),
            )
            cos_slot = ub_cos.next_write()
            mem_copy(
                cos_slot,
                tile_slice(
                    gm_cos, (half, VL), (owner * AIV_PER_BLOCK + get_subblock_id(), 0)
                ),
            )
            sin_tab = ub_sin.next_read()
            cos_tab = ub_cos.next_read()
            self._resident_b_cube(
                l1_x,
                l1_sx,
                l0a,
                l0b,
                l0c,
                ub_acc,
                gm_qr,
                gm_dqr,
                gm_w,
                gm_dsw,
                owner,
                group,
                groups,
                ub_out,
                ub_tail,
                sin_tab,
                cos_tab,
                gm_sin,
                gm_cos,
                gm_q,
                early_w,
                early_s,
            )

    @jit
    def _resident_b_cube(
        self,
        l1_x,
        l1_sx,
        l0a,
        l0b,
        l0c,
        ub_acc,
        gm_qr,
        gm_dqr,
        gm_w,
        gm_dsw,
        owner,
        group,
        groups,
        ub_out,
        ub_tail,
        ub_sin,
        ub_cos,
        gm_sin,
        gm_cos,
        gm_q,
        early_w,
        early_s,
    ):
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (
            tiling.bm_b,
            tiling.bn_b,
            tiling.bk_b,
        )
        nk = tiling.rank_size // reduction_tile_cols
        panels_per_head = tiling.head_size // (PANEL_OUTPUT_TILES * output_tile_cols)
        l1_head_scale = early_s
        l1_b_wide = early_w
        held_x = []
        held_s = []
        for k in range_constexpr(nk):
            qr_payload_slot = l1_x[k].next_write()
            mem_copy(
                qr_payload_slot,
                tile_slice(gm_qr, (token_tile_rows, reduction_tile_cols), (owner, k)),
            )
            held_x.append(l1_x[k].next_read())
        for sk in range_constexpr(nk // tiling.scale_k_tiles):
            qr_scale_slot = reinterpret(
                l1_sx[sk].next_write(),
                dtype=FLOAT8_E8M0,
                layout=make_layout(
                    (
                        token_tile_rows,
                        tiling.scale_k_tiles * reduction_tile_cols // MX_GROUP,
                    )
                ),
                data_format="zn",
            )
            mem_copy(
                qr_scale_slot,
                tile_slice(
                    gm_dqr,
                    (
                        token_tile_rows,
                        tiling.scale_k_tiles * reduction_tile_cols // MX_PAIR_WIDTH,
                        MX_PAIR,
                    ),
                    (owner, sk, 0),
                ).flatten(1, 2),
            )
            held_s.append(
                reinterpret(
                    l1_sx[sk].next_read(),
                    dtype=FLOAT8_E8M0,
                    layout=make_layout(
                        (
                            token_tile_rows,
                            tiling.scale_k_tiles * reduction_tile_cols // MX_GROUP,
                        )
                    ),
                    data_format="zn",
                )
            )
        for head in range(group, tiling.head_count, groups):
            for panel in range_constexpr(panels_per_head * nk):
                head_scale = l1_head_scale[panel % nk].next_read()
                wide_w = (
                    l1_b_wide[panel % RESIDENT_K_WINDOWS]
                    .next_read()
                    .reinterpret(dtypes.float8_e4m3fn)
                )
                self._b_panel_l0_reuse(
                    wide_w,
                    head_scale,
                    l0a,
                    l0b,
                    l0c,
                    held_x,
                    held_s,
                    panel // nk,
                    panel % nk,
                )
                if const_expr(panel + RESIDENT_K_WINDOWS < panels_per_head * nk):
                    self._load_b_panel(
                        l1_b_wide[panel % RESIDENT_K_WINDOWS].next_write(),
                        gm_w,
                        head * panels_per_head + (panel + RESIDENT_K_WINDOWS) // nk,
                        (panel + RESIDENT_K_WINDOWS) % nk,
                    )
                elif head + groups < tiling.head_count:
                    mem_copy(
                        l1_head_scale[panel % nk].next_write(),
                        tile_slice(
                            gm_dsw,
                            (
                                tiling.head_size,
                                reduction_tile_cols // MX_PAIR_WIDTH,
                                MX_PAIR,
                            ),
                            (head + groups, panel % nk, 0),
                        )
                        .permute(1, 2, 0)
                        .flatten(0, 1),
                    )
                    self._load_b_panel(
                        l1_b_wide[panel % RESIDENT_K_WINDOWS].next_write(),
                        gm_w,
                        (head + groups) * panels_per_head,
                        (panel + RESIDENT_K_WINDOWS) % nk,
                    )
                if const_expr(panel % nk == nk - 1):
                    qb_accumulator_slot = ub_acc.next_write()
                    # FixPipe 双目的地（split-M）：L0C 源 partition(dim=0) 将
                    # 行数对半分给两个 Vector 核。
                    mem_copy(
                        qb_accumulator_slot,
                        reinterpret(
                            tile_slice(
                                l0c,
                                (
                                    token_tile_rows,
                                    PANEL_OUTPUT_TILES * output_tile_cols,
                                ),
                                (0, panel // nk),
                            ),
                            layout=make_layout(
                                (token_tile_rows, PANEL_OUTPUT_TILES * output_tile_cols)
                            ),
                        ).partition(dim=0),
                        copy_options(unit_flag=FIXPIPE_FINAL),
                    )
                    self._b_vec(
                        ub_acc,
                        ub_out,
                        ub_tail,
                        ub_sin,
                        ub_cos,
                        gm_sin,
                        gm_cos,
                        gm_q,
                        head * tiling.owners_b + owner,
                        panel // nk,
                    )

    @jit
    def _b_panel_l0_reuse(
        self,
        wide_w,
        head_scale,
        l0a,
        l0b,
        l0c,
        held_x,
        held_s,
        n_group,
        reduction_tile_idx,
    ):
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (
            tiling.bm_b,
            tiling.bn_b,
            tiling.bk_b,
        )
        nk = tiling.rank_size // reduction_tile_cols
        # 一个 QR K320 分块服务于打包 panel 中的四个 N64 切片。
        for reduction_buffer_idx in range_constexpr(L0_DEPTH):
            if reduction_tile_idx % L0_DEPTH == reduction_buffer_idx:
                activation_l0_slot = l0a[reduction_buffer_idx].next_write()
                for resident_reduction_idx in range_constexpr(nk):
                    if reduction_tile_idx == resident_reduction_idx:
                        mem_copy(
                            activation_l0_slot,
                            held_x[resident_reduction_idx],
                            copy_options(
                                mx_scale=tile_slice(
                                    held_s[
                                        resident_reduction_idx // tiling.scale_k_tiles
                                    ],
                                    (token_tile_rows, reduction_tile_cols // MX_GROUP),
                                    (0, resident_reduction_idx % tiling.scale_k_tiles),
                                )
                            ),
                        )
                held_a = l0a[reduction_buffer_idx].next_read()
                for n_sub in range(PANEL_OUTPUT_TILES):
                    output_tile_idx = n_group * PANEL_OUTPUT_TILES + n_sub
                    weight_l1_tile = tile_slice(
                        wide_w, (output_tile_cols, reduction_tile_cols), (n_sub, 0)
                    )
                    weight_scale_tile = tile_slice(
                        head_scale,
                        (reduction_tile_cols // MX_GROUP, output_tile_cols),
                        (0, output_tile_idx),
                    )
                    mem_copy(
                        l0b.next_write(),
                        weight_l1_tile,
                        copy_options(mx_scale=weight_scale_tile),
                    )
                    matmul(
                        tile_slice(
                            l0c,
                            (token_tile_rows, output_tile_cols),
                            (0, output_tile_idx),
                        ),
                        held_a,
                        l0b.next_read(),
                        init=reduction_tile_idx == 0,
                        unit_flag=(
                            FIXPIPE_FINAL
                            if reduction_tile_idx + 1 == nk
                            else FIXPIPE_ACCUMULATE
                        ),
                        mx_mode=True,
                    )

    @jit
    def _publish_a1(self):
        # A1 只有 Cube 生产者：排空 Cube 流水线并完成 Cube 全核汇合，
        # 再释放配对的 AIV，无需额外的 AIV→AIC 汇合。
        cube_sync_all()
        cube_sync_block_arrive(PIPE.FIXPIPE, 0, mode=0)
        cube_sync_block_wait(PIPE.S, 0, mode=0)
        cube_sync_block_arrive(PIPE.FIXPIPE, 1, mode=2)
        vec_sync_block_wait(PIPE.S, 1, mode=2)

    @jit
    def _publish_qr(self):
        # 所有 QR/scale 行均由 AIV 产生：完成 DMA 和 AIV 全核汇合后，
        # 释放配对 AIC；AIV 随后可继续处理 KV。
        vec_sync_all()
        vec_sync_block_arrive(PIPE.MTE3, 0, mode=0)
        vec_sync_block_wait(PIPE.S, 0, mode=0)
        vec_sync_block_arrive(PIPE.MTE3, 1, mode=2)
        cube_sync_block_wait(PIPE.S, 1, mode=2)

    @jit
    def _a1_branch(
        self,
        gm_x,
        gm_dsx,
        gm_wqa,
        gm_dswqa,
        gm_wkv,
        gm_dswkv,
        gm_ws_qa,
        gm_ws_kv,
        gm_gamma_qr,
        gm_index,
        gm_qr,
        gm_dqr2d,
        norm_eps,
    ):
        """先发布 QA，再使 KV 投影与 QR 归一化重叠执行。"""
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols, k_l1 = (
            tiling.bm_a1,
            tiling.bn_a1,
            tiling.bk_a1,
            tiling.k_l1_a1,
        )
        scale_l0 = reduction_tile_cols // MX_GROUP
        scale_l1 = k_l1 // MX_GROUP
        groups_l1 = k_l1 // MX_PAIR_WIDTH

        # 物理 token 分块保持对齐，每次搬入完整的 K 窗口。
        l1_x = RingBuffer(
            MemLoc.L1,
            FLOAT8_E4M3FN,
            make_layout((token_tile_rows, tiling.k_l1_x_a1)),
            depth=2,
            data_format="nz",
        )
        # E8M0 scale 的 zn 分形不能用 data_format="zn" 直接分配（ 仅接受
        # nd/nz）：以 uint8 原始字节承载，每次选槽时用 reinterpret 绑定 zn 视图。
        l1_sx = RingBuffer(
            MemLoc.L1,
            dtypes.uint8,
            make_layout(
                (
                    _ceil_div(token_tile_rows, NZ_M_FRAC)
                    * _ceil_div(tiling.k_l1_x_a1 // MX_GROUP, MX_PAIR)
                    * ZN_FRACTAL_BYTES,
                )
            ),
            depth=2,
            data_format="nd",
        )
        # 权重输入已经是 NZ，以整数字节搬入 L1，在进入 L0B 时重解释为 FP8。
        # FP8 别名要求编译期已知的起始地址，因此各权重缓冲使用单槽 Channel。
        l1_w = [
            RingBuffer(
                MemLoc.L1,
                dtypes.int8,
                make_layout((output_tile_cols, k_l1)),
                depth=1,
                data_format="nz",
            )
            for _ in range(W_BUF)
        ]
        l1_sw = RingBuffer(
            MemLoc.L1,
            FLOAT8_E8M0,
            make_layout((scale_l1, output_tile_cols)),
            depth=W_BUF,
            data_format="nz",
        )
        l0a = RingBuffer(
            MemLoc.L0A,
            FLOAT8_E4M3FN,
            make_layout((token_tile_rows, reduction_tile_cols)),
            depth=L0_DEPTH,
        )
        l0b = RingBuffer(
            MemLoc.L0B,
            FLOAT8_E4M3FN,
            make_layout((output_tile_cols, reduction_tile_cols)),
            depth=L0_DEPTH,
        )
        # 在一个 token×输出列的 L0C 分块中累加所分配的 K 分块。
        l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((token_tile_rows, output_tile_cols)),
            depth=1,
        )

        # 阶段 1：将所有 QA 部分和发布给 QR 消费者。
        for qa_projection_task_idx in range(
            get_block_idx(), tiling.tiles_a1_qa, tiling.block_dim
        ):
            self._a1_tile(
                l1_x,
                l1_sx,
                l1_w,
                l1_sw,
                l0a,
                l0b,
                l0c,
                gm_x,
                gm_dsx,
                gm_wqa,
                gm_dswqa,
                gm_ws_qa,
                qa_projection_task_idx,
                tiling.n_tiles_qa,
                scale_l0,
                groups_l1,
            )
        self._publish_a1()
        # 阶段 2：独立执行 Cube KV 生产和 Vector QR 消费。
        self._a2_branch(
            gm_gamma_qr,
            gm_index,
            gm_qr,
            gm_dqr2d,
            gm_ws_qa,
            gm_ws_kv,
            norm_eps,
            l1_x,
            l1_sx,
            l1_w,
            l1_sw,
            l0a,
            l0b,
            l0c,
            gm_x,
            gm_dsx,
            gm_wkv,
            gm_dswkv,
            scale_l0,
            groups_l1,
        )

    @jit
    def _a2_branch(
        self,
        gm_gamma_qr,
        gm_index,
        gm_qr,
        gm_dqr2d,
        gm_ws_qa,
        gm_ws_kv,
        norm_eps,
        l1_x,
        l1_sx,
        l1_w,
        l1_sw,
        l0a,
        l0b,
        l0c,
        gm_x,
        gm_dsx,
        gm_wkv,
        gm_dswkv,
        scale_l0,
        groups_l1,
    ):
        """Cube 计算 KV 投影，Vector 按 token 分块归约并量化 QA。"""
        tiling = self.tiling
        token_tile_rows, rank = tiling.bm_a2, tiling.rank_size

        # DMA 与 Vector 之间的依赖由 Channel 承载，Buffer 仅用于 Vector 暂存。
        # prefill 使用单槽 Channel，在增大 token 分块时限制 UB 占用。
        a2_depth = (
            A2_BUF_DEPTH_PREFILL
            if tiling.template == "split_t"
            else A2_BUF_DEPTH_DECODE
        )
        ub_qa = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((token_tile_rows, rank)),
            depth=a2_depth,
        )
        ub_gamma_qr = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, rank)), depth=1
        )
        ub_qr = RingBuffer(
            MemLoc.UB,
            dtypes.uint8,
            make_layout((token_tile_rows, rank)),
            depth=a2_depth,
        )
        # 指数暂存仅在 VF 区域间传递，多分配一行以容纳完整向量读取。
        ub_exp = Buffer(
            MemLoc.UB,
            dtypes.int32,
            make_layout((token_tile_rows + 1, max(tiling.exp_stride, VL))),
        )
        # Channel 不会隐式保证每行的字节对齐，需要显式指定 scale 行跨度。
        # DMA 对外只暴露有效的量化组列。
        ub_scale = RingBuffer(
            MemLoc.UB,
            dtypes.uint8,
            make_layout((token_tile_rows, tiling.scale_stride)),
            depth=a2_depth,
        )
        # 循环不变量 gamma 只加载一次，同时保留 DMA→Vector 的 Channel 握手。
        g_slot = ub_gamma_qr.next_write()
        mem_copy(g_slot, tile_slice(gm_gamma_qr, (1, rank), (0, 0)))
        gamma_qr = ub_gamma_qr.next_read()

        if const_expr(tiling.split_k > 1):
            qa_sum = Buffer(
                MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, rank))
            )
            qa_comp = Buffer(
                MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, rank))
            )
        # KV 投影与 QA 数据互不依赖，可与后续 VF 工作重叠。
        for kv_slot in range(get_block_idx(), tiling.tiles_a1_kv, tiling.block_dim):
            self._a1_tile(
                l1_x,
                l1_sx,
                l1_w,
                l1_sw,
                l0a,
                l0b,
                l0c,
                gm_x,
                gm_dsx,
                gm_wkv,
                gm_dswkv,
                gm_ws_kv,
                kv_slot,
                tiling.n_tiles_kv,
                scale_l0,
                groups_l1,
            )
        aiv = get_block_idx() * AIV_PER_BLOCK + get_subblock_id()
        for token_tile_idx in range(
            aiv, tiling.tiles_a2, tiling.block_dim * AIV_PER_BLOCK
        ):
            rows = tile_slice(
                gm_index, (token_tile_rows, 1), (token_tile_idx, 0)
            ).shape[0]
            # QR 分支：沿 rank 轴执行 RMSNorm、动态 MXFP8 量化并打包 scale。
            # 与 _reduce_normalize_a 中的对应循环语义相同（分支流水线路径），修改需同步。
            qa_slot = ub_qa.next_write()
            mem_copy(
                qa_slot,
                tile_slice(gm_ws_qa, (token_tile_rows, rank), (token_tile_idx, 0)),
            )
            qa = ub_qa.next_read()
            qr_slot = ub_qr.next_write()
            if const_expr(tiling.split_k > 1):
                self._acc_partial_qa(qa_sum, qa_comp, qa, rows, rank, True)
                for part in range(1, tiling.split_k):
                    qa_slot = ub_qa.next_write()
                    mem_copy(
                        qa_slot,
                        tile_slice(
                            gm_ws_qa,
                            (token_tile_rows, rank),
                            (
                                part
                                * tiling.m_tiles_a1
                                * tiling.bm_a1
                                // token_tile_rows
                                + token_tile_idx,
                                0,
                            ),
                        ),
                    )
                    qa_part = ub_qa.next_read()
                    self._acc_partial_qa(qa_sum, qa_comp, qa_part, rows, rank, False)
                self._quantize_qr(qa_sum, gamma_qr, qr_slot, ub_exp, rows, norm_eps)
            else:
                self._quantize_qr(qa, gamma_qr, qr_slot, ub_exp, rows, norm_eps)
            scale_slot = ub_scale.next_write()
            self._pack_scale(ub_exp, scale_slot, rows)
            mem_copy(
                tile_slice(gm_qr, (token_tile_rows, rank), (token_tile_idx, 0)),
                ub_qr.next_read(),
            )
            mem_copy(
                tile_slice(
                    gm_dqr2d, (token_tile_rows, tiling.qr_groups), (token_tile_idx, 0)
                ),
                reinterpret(
                    ub_scale.next_read(),
                    layout=make_layout(
                        (token_tile_rows, tiling.qr_groups),
                        stride=(tiling.scale_stride, 1),
                    ),
                ),
            )

        # 在 Cube 消费前发布全部 QR/scale 行。KV 部分和需要对 Vector 可见，
        # QR 行需要对 Cube 可见；分别保留两条生产者到消费者的同步依赖。
        self._publish_a1()
        self._publish_qr()

    @jit
    def _service_kv(
        self,
        gm_gamma_kv,
        gm_sin,
        gm_cos,
        gm_index,
        gm_cache,
        gm_ws_kv,
        norm_eps,
        token_tile_idx,
    ):
        tiling = self.tiling
        token_tile_rows, head = tiling.bm_a2, tiling.head_size
        if token_tile_idx < tiling.tiles_a2:
            a2_depth = (
                A2_BUF_DEPTH_PREFILL
                if tiling.template == "split_t"
                else A2_BUF_DEPTH_DECODE
            )
            ub_kv = RingBuffer(
                MemLoc.UB,
                dtypes.float32,
                make_layout((token_tile_rows, head)),
                depth=a2_depth,
            )
            ub_gamma_kv = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, head)), depth=1
            )
            ub_sin = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, VL)), depth=1
            )
            ub_cos = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, VL)), depth=1
            )
            ub_kvb = RingBuffer(
                MemLoc.UB, dtypes.uint8, make_layout((1, head)), depth=2
            )
            ub_tail = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, VL)))
            gk_slot = ub_gamma_kv.next_write()
            mem_copy(gk_slot, tile_slice(gm_gamma_kv, (1, head), (0, 0)))
            gamma_kv = ub_gamma_kv.next_read()
            if const_expr(tiling.split_k > 1):
                kv_sum = Buffer(
                    MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, head))
                )
            rows = tile_slice(
                gm_index, (token_tile_rows, 1), (token_tile_idx, 0)
            ).shape[0]
            # KV 分支：沿 head 特征轴执行 RMSNorm、RoPE 和固定 scale 的 E4M3FN 转换。
            # 与 _reduce_normalize_a 中的对应循环共享 KV 分支语义，修改需同步。
            kv_slot = ub_kv.next_write()
            mem_copy(
                kv_slot,
                tile_slice(gm_ws_kv, (token_tile_rows, head), (token_tile_idx, 0)),
            )
            sin_slot = ub_sin.next_write()
            mem_copy(
                sin_slot, tile_slice(gm_sin, (token_tile_rows, VL), (token_tile_idx, 0))
            )
            cos_slot = ub_cos.next_write()
            mem_copy(
                cos_slot, tile_slice(gm_cos, (token_tile_rows, VL), (token_tile_idx, 0))
            )
            kv = ub_kv.next_read()
            if const_expr(tiling.split_k > 1):
                self._acc_partial(kv_sum, kv, rows, head, True)
                for part in range(1, tiling.split_k):
                    kv_slot = ub_kv.next_write()
                    mem_copy(
                        kv_slot,
                        tile_slice(
                            gm_ws_kv,
                            (token_tile_rows, head),
                            (
                                part
                                * tiling.m_tiles_a1
                                * tiling.bm_a1
                                // token_tile_rows
                                + token_tile_idx,
                                0,
                            ),
                        ),
                    )
                    kv_part = ub_kv.next_read()
                    self._acc_partial(kv_sum, kv_part, rows, head, False)
            sin_tab = ub_sin.next_read()
            cos_tab = ub_cos.next_read()
            # 逐行散写到当前 token 的 cache 索引位置。
            for j in range(rows):
                kvb_slot = ub_kvb.next_write()
                if const_expr(tiling.split_k > 1):
                    self._finish_kv_row(
                        kv_sum,
                        gamma_kv,
                        sin_tab,
                        cos_tab,
                        kvb_slot,
                        ub_tail,
                        j,
                        j,
                        norm_eps,
                    )
                else:
                    self._finish_kv_row(
                        kv,
                        gamma_kv,
                        sin_tab,
                        cos_tab,
                        kvb_slot,
                        ub_tail,
                        j,
                        j,
                        norm_eps,
                    )
                mem_copy(
                    tile_slice(
                        gm_cache,
                        (1, head),
                        (gm_index[token_tile_idx * token_tile_rows + j, 0], 0),
                    ),
                    ub_kvb.next_read(),
                )

    # A1 计算独立的 Cube 部分和，A2 按 token 归约并执行后处理。
    @jit
    def _project_qa_kv(
        self, gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_wkv, gm_dswkv, gm_ws_qa, gm_ws_kv
    ):
        """独立计算 QA/KV 输出分块，将 FP32 部分和写入交换区。"""
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols, k_l1 = (
            tiling.bm_a1,
            tiling.bn_a1,
            tiling.bk_a1,
            tiling.k_l1_a1,
        )
        scale_l0 = reduction_tile_cols // MX_GROUP
        scale_l1 = k_l1 // MX_GROUP
        groups_l1 = k_l1 // MX_PAIR_WIDTH

        # token 分块按 NZ 网格补齐，每次搬运完整 K 窗口。
        # 缓冲分配与 _a1_branch 保持一致，修改需同步。
        l1_x = RingBuffer(
            MemLoc.L1,
            FLOAT8_E4M3FN,
            make_layout((token_tile_rows, tiling.k_l1_x_a1)),
            depth=2,
            data_format="nz",
        )
        # E8M0 scale 的 zn 分形不能用 data_format="zn" 直接分配（ 仅接受
        # nd/nz）：以 uint8 原始字节承载，每次选槽时用 reinterpret 绑定 zn 视图。
        l1_sx = RingBuffer(
            MemLoc.L1,
            dtypes.uint8,
            make_layout(
                (
                    _ceil_div(token_tile_rows, NZ_M_FRAC)
                    * _ceil_div(tiling.k_l1_x_a1 // MX_GROUP, MX_PAIR)
                    * ZN_FRACTAL_BYTES,
                )
            ),
            depth=2,
            data_format="nd",
        )
        # 权重输入已经是 NZ，以整数字节搬入 L1，在进入 L0B 时重解释为 FP8。
        # FP8 别名要求编译期已知的起始地址，因此各权重缓冲使用单槽 Channel。
        l1_w = [
            RingBuffer(
                MemLoc.L1,
                dtypes.int8,
                make_layout((output_tile_cols, k_l1)),
                depth=1,
                data_format="nz",
            )
            for _ in range(W_BUF)
        ]
        l1_sw = RingBuffer(
            MemLoc.L1,
            FLOAT8_E8M0,
            make_layout((scale_l1, output_tile_cols)),
            depth=W_BUF,
            data_format="nz",
        )
        l0a = RingBuffer(
            MemLoc.L0A,
            FLOAT8_E4M3FN,
            make_layout((token_tile_rows, reduction_tile_cols)),
            depth=L0_DEPTH,
        )
        l0b = RingBuffer(
            MemLoc.L0B,
            FLOAT8_E4M3FN,
            make_layout((output_tile_cols, reduction_tile_cols)),
            depth=L0_DEPTH,
        )
        # 在一个 token×输出列的 L0C 分块中累加所分配的 K 分块。
        l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((token_tile_rows, output_tile_cols)),
            depth=1,
        )

        # QA 与 KV 共用任务队列，并使用相同的权重 Channel 轮转规则。
        for projection_task_idx in range(
            get_block_idx(), tiling.tiles_a1_qa + tiling.tiles_a1_kv, tiling.block_dim
        ):
            if projection_task_idx < tiling.tiles_a1_qa:
                self._a1_tile(
                    l1_x,
                    l1_sx,
                    l1_w,
                    l1_sw,
                    l0a,
                    l0b,
                    l0c,
                    gm_x,
                    gm_dsx,
                    gm_wqa,
                    gm_dswqa,
                    gm_ws_qa,
                    projection_task_idx,
                    tiling.n_tiles_qa,
                    scale_l0,
                    groups_l1,
                )
            else:
                self._a1_tile(
                    l1_x,
                    l1_sx,
                    l1_w,
                    l1_sw,
                    l0a,
                    l0b,
                    l0c,
                    gm_x,
                    gm_dsx,
                    gm_wkv,
                    gm_dswkv,
                    gm_ws_kv,
                    projection_task_idx - tiling.tiles_a1_qa,
                    tiling.n_tiles_kv,
                    scale_l0,
                    groups_l1,
                )

    @jit
    def _a1_tile(
        self,
        l1_a,
        l1_sa,
        l1_b,
        l1_sb,
        l0a,
        l0b,
        l0c,
        gm_a,
        gm_dsa,
        gm_b,
        gm_dsb,
        gm_ws,
        projection_task_idx,
        n_count,
        scale_l0,
        groups_l1,
    ):
        """累加一个 K 分块，并将其独立的 FP32 部分和写出。"""
        tiling = self.tiling
        token_tile_rows, output_tile_cols, k_l1 = (
            tiling.bm_a1,
            tiling.bn_a1,
            tiling.k_l1_a1,
        )
        n_kw = tiling.hidden_size // tiling.split_k // k_l1
        logical_slot = projection_task_idx // tiling.split_k
        split = projection_task_idx % tiling.split_k
        m = logical_slot // n_count
        n = logical_slot % n_count
        start_kw = split * n_kw
        accumulator = l0c.next_write()
        # 成对展开 K 窗口，使每个单槽权重 Channel 在一次循环体中只轮转一次。
        if const_expr(n_kw % W_BUF == 0):
            for kwp in range(n_kw // W_BUF):
                kw0 = start_kw + kwp * W_BUF
                for u in range_constexpr(W_BUF):
                    self._fill_w_a(l1_b[u], l1_sb, gm_b, gm_dsb, n, kw0 + u, groups_l1)
                for u in range_constexpr(W_BUF):
                    self._mm_a(
                        l1_a,
                        l1_sa,
                        l1_b[u],
                        l1_sb,
                        l0a,
                        l0b,
                        accumulator,
                        gm_a,
                        gm_dsa,
                        m,
                        kw0 + u,
                        start_kw,
                        scale_l0,
                    )
        else:
            # 窗口数量为奇数时退回单槽，不再重叠预取。
            for kw in range(start_kw, start_kw + n_kw):
                self._fill_w_a(l1_b[0], l1_sb, gm_b, gm_dsb, n, kw, groups_l1)
                self._mm_a(
                    l1_a,
                    l1_sa,
                    l1_b[0],
                    l1_sb,
                    l0a,
                    l0b,
                    accumulator,
                    gm_a,
                    gm_dsa,
                    m,
                    kw,
                    start_kw,
                    scale_l0,
                )
        # L0C 直接写入 GM，不经过 Vector 暂存。
        #  mem_copy 以源域为准：将 L0C 源逻辑域重绑定到目的实际域，
        # 尾块裁剪生效且物理布局保持不变。
        ws_tile = tile_slice(
            gm_ws,
            (token_tile_rows, output_tile_cols),
            (split * tiling.m_tiles_a1 + m, n),
        )
        mem_copy(
            ws_tile,
            reinterpret(
                accumulator,
                layout=make_layout(ws_tile.shape),
                physical_layout=accumulator.physical_layout,
            ),
        )

    @jit
    def _mm_a(
        self,
        l1_a,
        l1_sa,
        l1_b,
        l1_sb,
        l0a,
        l0b,
        l0c,
        gm_a,
        gm_dsa,
        m,
        kw,
        start_kw,
        scale_l0,
    ):
        """将一个权重窗口中的所有 L0 归约分块累加到 L0C。"""
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols, k_l1 = (
            tiling.bm_a1,
            tiling.bn_a1,
            tiling.bk_a1,
            tiling.k_l1_a1,
        )
        groups_l0 = reduction_tile_cols // MX_PAIR_WIDTH
        sa_layout = make_layout((token_tile_rows, tiling.k_l1_x_a1 // MX_GROUP))
        w_ready = l1_b.next_read().reinterpret(dtype=FLOAT8_E4M3FN)
        s_ready = l1_sb.next_read()
        if const_expr(tiling.a1_window):
            mem_copy(
                l1_a.next_write(),
                tile_slice(gm_a, (token_tile_rows, k_l1), (m, kw)),
            )
            mem_copy(
                reinterpret(
                    l1_sa.next_write(),
                    dtype=FLOAT8_E8M0,
                    layout=sa_layout,
                    data_format="zn",
                ),
                tile_slice(
                    gm_dsa,
                    (token_tile_rows, k_l1 // MX_PAIR_WIDTH, MX_PAIR),
                    (m, kw, 0),
                ).flatten(1, 2),
            )
            a_ready = l1_a.next_read()
            sa_ready = reinterpret(
                l1_sa.next_read(),
                dtype=FLOAT8_E8M0,
                layout=sa_layout,
                data_format="zn",
            )
        for kl0 in range(k_l1 // reduction_tile_cols):
            kb = kw * (k_l1 // reduction_tile_cols) + kl0
            # 两个 L0 操作数都需要 mx_scale 元数据，matmul 才会使用原生 MXFP8 指令。
            if const_expr(tiling.a1_window):
                mem_copy(
                    l0a.next_write(),
                    tile_slice(
                        a_ready, (token_tile_rows, reduction_tile_cols), (0, kl0)
                    ),
                    copy_options(
                        mx_scale=tile_slice(
                            sa_ready, (token_tile_rows, scale_l0), (0, kl0)
                        )
                    ),
                )
            else:
                mem_copy(
                    l1_a.next_write(),
                    tile_slice(gm_a, (token_tile_rows, reduction_tile_cols), (m, kb)),
                )
                mem_copy(
                    reinterpret(
                        l1_sa.next_write(),
                        dtype=FLOAT8_E8M0,
                        layout=sa_layout,
                        data_format="zn",
                    ),
                    tile_slice(
                        gm_dsa, (token_tile_rows, groups_l0, MX_PAIR), (m, kb, 0)
                    ).flatten(1, 2),
                )
                mem_copy(
                    l0a.next_write(),
                    l1_a.next_read(),
                    copy_options(
                        mx_scale=reinterpret(
                            l1_sa.next_read(),
                            dtype=FLOAT8_E8M0,
                            layout=sa_layout,
                            data_format="zn",
                        )
                    ),
                )
            if const_expr(k_l1 == reduction_tile_cols):
                mem_copy(l0b.next_write(), w_ready, copy_options(mx_scale=s_ready))
            else:
                mem_copy(
                    l0b.next_write(),
                    tile_slice(
                        w_ready, (output_tile_cols, reduction_tile_cols), (0, kl0)
                    ),
                    copy_options(
                        mx_scale=tile_slice(
                            s_ready, (scale_l0, output_tile_cols), (kl0, 0)
                        )
                    ),
                )
            matmul(
                l0c,
                l0a.next_read(),
                l0b.next_read(),
                init=(kb == start_kw * (k_l1 // reduction_tile_cols)),
                mx_mode=True,
            )

    @jit
    def _fill_w_a(self, l1_b, l1_sb, gm_b, gm_dsb, n, kw, groups_l1):
        """将一个 NZ 权重窗口及其 ND scale 装入 L1。
        scale 的 [输出列, 归约组对, pair] 布局由 permute+flatten 显式
        转置展开为 (G, N) 后拷入 nz 通道；payload 字节直接搬运。"""
        tiling = self.tiling
        output_tile_cols, k_l1 = tiling.bn_a1, tiling.k_l1_a1
        w_slot = l1_b.next_write()
        mem_copy(w_slot, tile_slice(gm_b, (output_tile_cols, k_l1), (n, kw)))
        s_slot = l1_sb.next_write()
        mem_copy(
            s_slot,
            tile_slice(gm_dsb, (output_tile_cols, groups_l1, MX_PAIR), (n, kw, 0))
            .permute(1, 2, 0)
            .flatten(0, 1),
        )

    @jit
    def _reduce_normalize_a(
        self,
        gm_gamma_qr,
        gm_gamma_kv,
        gm_sin,
        gm_cos,
        gm_index,
        gm_cache,
        gm_qr,
        gm_dqr2d,
        gm_ws_qa,
        gm_ws_kv,
        norm_eps,
    ):
        """归约 QA 并生成 QR；L1 复用路径还在此完成 KV 后处理。每个 token 分块只归一个 Vector 核，避免 split-M 空尾块。"""
        tiling = self.tiling
        token_tile_rows, rank, head = tiling.bm_a2, tiling.rank_size, tiling.head_size

        # DMA 与 Vector 之间的依赖由 Channel 承载，Buffer 仅用于 Vector 暂存。
        # prefill 使用单槽 Channel，在增大 token 分块时限制 UB 占用。
        a2_depth = (
            A2_BUF_DEPTH_PREFILL
            if tiling.template == "split_t"
            else A2_BUF_DEPTH_DECODE
        )
        if const_expr(tiling.l1_reuse):
            ub_qa = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((tiling.split_k, rank)), depth=1
            )
            ub_kv = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((tiling.split_k, head)), depth=1
            )
        else:
            ub_qa = RingBuffer(
                MemLoc.UB,
                dtypes.float32,
                make_layout((token_tile_rows, rank)),
                depth=a2_depth,
            )
        ub_gamma_qr = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, rank)), depth=1
        )
        # RoPE 表的每行补齐到 VL，保证完整向量读取不越界。
        if const_expr(tiling.l1_reuse):
            ub_gamma_kv = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, head)), depth=1
            )
            ub_sin = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, VL)), depth=1
            )
            ub_cos = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, VL)), depth=1
            )
        ub_qr = RingBuffer(
            MemLoc.UB,
            dtypes.uint8,
            make_layout((token_tile_rows, rank)),
            depth=a2_depth,
        )
        # 指数暂存仅在 VF 区域间传递，多分配一行以容纳完整向量读取。
        ub_exp = Buffer(
            MemLoc.UB,
            dtypes.int32,
            make_layout((token_tile_rows + 1, max(tiling.exp_stride, VL))),
        )
        # 显式指定对齐的 scale 行跨度，DMA 只暴露公开的有效列。
        ub_scale = RingBuffer(
            MemLoc.UB,
            dtypes.uint8,
            make_layout((token_tile_rows, tiling.scale_stride)),
            depth=a2_depth,
        )
        # 使用单行 Channel 逐条散写 KV cache；
        # 在 FP32 暂存区拼接对齐的 KV 尾部向量。
        if const_expr(tiling.l1_reuse):
            ub_kvb = RingBuffer(
                MemLoc.UB, dtypes.uint8, make_layout((1, head)), depth=2
            )
            ub_tail = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, VL)))

        # 循环不变量 gamma 只加载一次，同时保留 DMA→Vector 的 Channel 握手。
        g_slot = ub_gamma_qr.next_write()
        mem_copy(g_slot, tile_slice(gm_gamma_qr, (1, rank), (0, 0)))
        if const_expr(tiling.l1_reuse):
            gk_slot = ub_gamma_kv.next_write()
            mem_copy(gk_slot, tile_slice(gm_gamma_kv, (1, head), (0, 0)))
        gamma_qr = ub_gamma_qr.next_read()
        if const_expr(tiling.l1_reuse):
            gamma_kv = ub_gamma_kv.next_read()

        qa_sum = None
        kv_sum = None
        qa_comp = None
        if const_expr(tiling.split_k > 1):
            qa_sum = Buffer(
                MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, rank))
            )
            if const_expr(not tiling.l1_reuse):
                qa_comp = Buffer(
                    MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, rank))
                )
            if const_expr(tiling.l1_reuse):
                kv_sum = Buffer(
                    MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, head))
                )
        aiv = get_block_idx() * AIV_PER_BLOCK + get_subblock_id()
        for token_tile_idx in range(
            aiv, tiling.tiles_a2, tiling.block_dim * AIV_PER_BLOCK
        ):
            rows = tile_slice(
                gm_index, (token_tile_rows, 1), (token_tile_idx, 0)
            ).shape[0]
            # QR 分支：沿 rank 轴执行 RMSNorm、动态 MXFP8 量化并打包 scale。
            # 与 _a2_branch 中的对应循环语义相同（通用路径），修改需同步。
            qa_slot = ub_qa.next_write()
            if const_expr(tiling.l1_reuse):
                partials = gm_ws_qa.view(
                    tiling.split_k, tiling.m_tiles_a1 * tiling.bm_a1 * rank
                )
                mem_copy(
                    qa_slot,
                    tile_slice(partials, (tiling.split_k, rank), (0, token_tile_idx)),
                )
            else:
                mem_copy(
                    qa_slot,
                    tile_slice(gm_ws_qa, (token_tile_rows, rank), (token_tile_idx, 0)),
                )
            qa = ub_qa.next_read()
            qr_slot = ub_qr.next_write()
            if const_expr(tiling.l1_reuse):
                self._acc_all_partials(qa_sum, qa, rank)
                self._quantize_qr(qa_sum, gamma_qr, qr_slot, ub_exp, rows, norm_eps)
            elif const_expr(tiling.split_k > 1):
                self._acc_partial_qa(qa_sum, qa_comp, qa, rows, rank, True)
                for part in range(1, tiling.split_k):
                    qa_slot = ub_qa.next_write()
                    mem_copy(
                        qa_slot,
                        tile_slice(
                            gm_ws_qa,
                            (token_tile_rows, rank),
                            (
                                part
                                * tiling.m_tiles_a1
                                * tiling.bm_a1
                                // token_tile_rows
                                + token_tile_idx,
                                0,
                            ),
                        ),
                    )
                    qa_part = ub_qa.next_read()
                    self._acc_partial_qa(qa_sum, qa_comp, qa_part, rows, rank, False)
                self._quantize_qr(qa_sum, gamma_qr, qr_slot, ub_exp, rows, norm_eps)
            else:
                self._quantize_qr(qa, gamma_qr, qr_slot, ub_exp, rows, norm_eps)
            scale_slot = ub_scale.next_write()
            self._pack_scale(ub_exp, scale_slot, rows)
            mem_copy(
                tile_slice(gm_qr, (token_tile_rows, rank), (token_tile_idx, 0)),
                ub_qr.next_read(),
            )
            mem_copy(
                tile_slice(
                    gm_dqr2d, (token_tile_rows, tiling.qr_groups), (token_tile_idx, 0)
                ),
                reinterpret(
                    ub_scale.next_read(),
                    layout=make_layout(
                        (token_tile_rows, tiling.qr_groups),
                        stride=(tiling.scale_stride, 1),
                    ),
                ),
            )

        # 在 Cube 消费前发布全部 QR/scale 行；各角色参与一次同步，
        # AIV 随后处理 KV，AIC 可进入 B 阶段。
        self._publish_qr()
        if const_expr(tiling.l1_reuse):
            for token_tile_idx in range(
                aiv, tiling.tiles_a2, tiling.block_dim * AIV_PER_BLOCK
            ):
                rows = tile_slice(
                    gm_index, (token_tile_rows, 1), (token_tile_idx, 0)
                ).shape[0]
                # KV 分支：沿 head 特征轴执行 RMSNorm、RoPE 和固定 scale 的 E4M3FN 转换。
                # 与 _service_kv 共享 KV 分支语义，修改需同步。
                kv_slot = ub_kv.next_write()
                partials = gm_ws_kv.view(
                    tiling.split_k, tiling.m_tiles_a1 * tiling.bm_a1 * head
                )
                mem_copy(
                    kv_slot,
                    tile_slice(partials, (tiling.split_k, head), (0, token_tile_idx)),
                )
                sin_slot = ub_sin.next_write()
                mem_copy(
                    sin_slot,
                    tile_slice(gm_sin, (token_tile_rows, VL), (token_tile_idx, 0)),
                )
                cos_slot = ub_cos.next_write()
                mem_copy(
                    cos_slot,
                    tile_slice(gm_cos, (token_tile_rows, VL), (token_tile_idx, 0)),
                )
                kv = ub_kv.next_read()
                self._acc_all_partials(kv_sum, kv, head)
                sin_tab = ub_sin.next_read()
                cos_tab = ub_cos.next_read()
                # 逐行散写到当前 token 的 cache 索引位置。
                for j in range(rows):
                    kvb_slot = ub_kvb.next_write()
                    if const_expr(tiling.split_k > 1):
                        self._finish_kv_row(
                            kv_sum,
                            gamma_kv,
                            sin_tab,
                            cos_tab,
                            kvb_slot,
                            ub_tail,
                            j,
                            j,
                            norm_eps,
                        )
                    else:
                        self._finish_kv_row(
                            kv,
                            gamma_kv,
                            sin_tab,
                            cos_tab,
                            kvb_slot,
                            ub_tail,
                            j,
                            j,
                            norm_eps,
                        )
                    mem_copy(
                        tile_slice(
                            gm_cache,
                            (1, head),
                            (gm_index[token_tile_idx * token_tile_rows + j, 0], 0),
                        ),
                        ub_kvb.next_read(),
                    )

    # A 阶段的 Vector 运算

    @jit
    def _acc_partial(self, dst, src, rows, width, first):
        with vf(mode="simd"):
            # 1. 读取各特征向量，并累加 split-K 部分和。
            mask = rr.full_mask()
            for token_row in cannbotdsl.range(rows):
                for feature_vector_idx in range(width // VL):
                    dst_feature_offset = (
                        token_row * dst.physical_stride[0] + feature_vector_idx * VL
                    )
                    partial_values = rr.vload(
                        src,
                        token_row * src.physical_stride[0] + feature_vector_idx * VL,
                    )
                    if const_expr(not first):
                        partial_values = rr.vadd(
                            rr.vload(dst, dst_feature_offset), partial_values, mask=mask
                        )
                    rr.vstore(dst, dst_feature_offset, partial_values, mask)

    @jit
    def _acc_partial_qa(self, dst, compensation, src, rows, width, first):
        """使用补偿求和合并 split-K QA，保留抵消后的低位残差。"""
        with vf(mode="simd"):
            # 1. 读取各特征向量，并累加 split-K 部分和。
            mask = rr.full_mask()
            for token_row in cannbotdsl.range(rows):
                for feature_vector_idx in range(width // VL):
                    dst_feature_offset = (
                        token_row * dst.physical_stride[0] + feature_vector_idx * VL
                    )
                    partial_values = rr.vload(
                        src,
                        token_row * src.physical_stride[0] + feature_vector_idx * VL,
                    )
                    # 2. 初始化部分和，或对下一个部分和执行补偿求和。
                    if const_expr(first):
                        rr.vstore(dst, dst_feature_offset, partial_values, mask)
                        rr.vstore(
                            compensation,
                            dst_feature_offset,
                            rr.vdups(0.0, dtypes.float32, mask=mask),
                            mask,
                        )
                    else:
                        corrected = rr.vsub(
                            partial_values,
                            rr.vload(compensation, dst_feature_offset),
                            mask=mask,
                        )
                        prior = rr.vload(dst, dst_feature_offset)
                        total = rr.vadd(prior, corrected, mask=mask)
                        lost = rr.vsub(
                            rr.vsub(total, prior, mask=mask), corrected, mask=mask
                        )
                        # 3. 同时保存累加结果和丢失的低位残差。
                        rr.vstore(dst, dst_feature_offset, total, mask)
                        rr.vstore(compensation, dst_feature_offset, lost, mask)

    @jit
    def _quantize_qr(self, ub_qa, ub_gamma, ub_qr, ub_exp, rows, norm_eps):
        """沿 rank 轴执行 RMSNorm，并按 MX_GROUP 动态量化为 MXFP8。
        共享指数为 max(floor(log2(amax)) - 8, -127)。
        归一化后的分组值保留在寄存器中，避免 UB 写后读冒险；
        读取源张量时使用包含 FixPipe 填充的物理行跨度。"""
        tiling = self.tiling
        rank = tiling.rank_size
        inv_rank = 1.0 / rank
        # 按物理 UB 行跨度寻址，包含 FixPipe 插入的填充。
        src_pitch = ub_qa.physical_stride[0]
        qr_pitch = ub_qr.physical_stride[0]
        with vf(mode="simd"):
            full = rr.full_mask()
            group_mask = rr.update_mask(MX_GROUP, elem_bits=32)[0]
            one = rr.vdups(1.0, dtypes.float32, mask=full)
            for token_row in cannbotdsl.range(rows):
                qa_row_offset = token_row * src_pitch
                # 1. 将平方和归约到第 0 个 lane，再广播到整行。
                square_sum_accumulator = rr.vdups(0.0, dtypes.float32, mask=full)
                for rank_vector_idx in range(rank // VL):
                    qa_values = rr.vload(ub_qa, qa_row_offset + rank_vector_idx * VL)
                    squared_values = rr.vmul(qa_values, qa_values, mask=full)
                    square_sum_accumulator = rr.vadd(
                        square_sum_accumulator,
                        rr.vreduce_sum(squared_values, mask=full),
                        mask=full,
                    )
                square_sum = rr.vdup(square_sum_accumulator, mask=full)
                mean_square = rr.vadds(
                    rr.vmuls(square_sum, inv_rank, mask=full), norm_eps, mask=full
                )
                rstd = rr.vdiv(one, rr.vsqrt(mean_square, mask=full), mask=full)

                # 2. 逐个 rank 分组归一化，并确定共享量化 scale。
                for rank_group_idx in range(rank // MX_GROUP):
                    rank_group_offset = rank_group_idx * MX_GROUP
                    normalized_values = rr.vmul(
                        rr.vload(ub_qa, qa_row_offset + rank_group_offset),
                        rstd,
                        mask=group_mask,
                    )
                    normalized_values = rr.vmul(
                        normalized_values,
                        rr.vload(ub_gamma, rank_group_offset),
                        mask=group_mask,
                    )
                    # sum 和 max 归约都只写第 0 个 lane，逐元素使用前必须广播。
                    amax = rr.vdup(
                        rr.vreduce_max(
                            rr.vabs(normalized_values, mask=group_mask), mask=group_mask
                        ),
                        mask=group_mask,
                    )
                    shared_exponent = self._shared_exponent(amax, group_mask)
                    payload = self._to_e4m3(
                        rr.vmul(
                            normalized_values,
                            self._reciprocal(shared_exponent, group_mask),
                            mask=group_mask,
                        ),
                        group_mask,
                    )
                    # 3. 写出 payload 字节及编码后的分组指数。
                    rr.vstore_pack(
                        ub_qr,
                        token_row * qr_pitch + rank_group_offset,
                        payload,
                        group_mask,
                        pack_mode=rr.PackMode.B32_TO_B8,
                    )
                    rr.vstore_first(
                        ub_exp,
                        token_row * ub_exp.physical_stride[0] + rank_group_idx,
                        rr.vadds(shared_exponent, E8M0_BIAS, mask=group_mask),
                    )

    @jit
    def _shared_exponent(self, amax, mask):
        """提取 FP32 指数字段，并为 E4M3 尾数预留动态范围。
        当 amax 为零或次正规数时，共享指数钳位到 -127，编码后的 scale 字节为零。
        amax 为零时 payload 为零；次正规数使用最小 scale，仍可能得到非零 payload。"""
        ebits = rr.vshr(rr.vreinterpret(amax, dtypes.int32), FP32_EXP_SHIFT, mask=mask)
        return rr.vmaxs(
            rr.vadds(ebits, -(FP32_EXP_BIAS + QUANT_EXP_HEADROOM), mask=mask),
            -E8M0_BIAS,
            mask=mask,
        )

    @jit
    def _reciprocal(self, exponent, mask):
        """直接利用 FP32 指数字段构造 2**(-exponent)。有限 FP32 amax 对应的量化指数不超过 119，因此倒数保持为正规数，无需通过除法计算。"""
        field = rr.vadds(rr.vneg(exponent, mask=mask), FP32_EXP_BIAS, mask=mask)
        return rr.vreinterpret(
            rr.vshl(field, FP32_EXP_SHIFT, mask=mask), dtypes.float32
        )

    @jit
    def _to_e4m3(self, value, mask):
        """先钳位到 E4M3FN 有限值域，再按最近偶数舍入转换。显式钳位保证饱和语义；前序运算的舍入可能使 FP8 中点附近的结果与 FP32 golden 不逐字节相同。"""
        clamped = rr.vmins(rr.vmaxs(value, -E4M3_MAX, mask=mask), E4M3_MAX, mask=mask)
        return rr.vcast(
            clamped,
            dtypes.float8_e4m3fn,
            mask=mask,
            rounding=rr.RoundingMode.RN,
            reg_layout=rr.RegLayout.ZERO,
        )

    @jit
    def _pack_scale(self, ub_exp, ub_scale, rows):
        """将 int32 指数打包为 E8M0 字节，并使用对齐的向量写入。独立的 VF 区域读取 _quantize_qr 写出的指数；B32_TO_B8 直接保留低字节，不执行数值转换。"""
        tiling = self.tiling
        exp_pitch = ub_exp.physical_stride[0]
        scale_pitch = ub_scale.physical_stride[0]
        with vf(mode="simd"):
            # 1. 读取 int32 编码指数；2. 将低字节打包写出。
            for token_row in cannbotdsl.range(rows):
                for scale_vector_idx in range(_ceil_div(tiling.qr_groups, VL)):
                    active_scale_lanes = min(
                        VL, tiling.qr_groups - scale_vector_idx * VL
                    )
                    mask = rr.update_mask(active_scale_lanes, elem_bits=32)[0]
                    rr.vstore_pack(
                        ub_scale,
                        token_row * scale_pitch + scale_vector_idx * VL,
                        rr.vload(ub_exp, token_row * exp_pitch + scale_vector_idx * VL),
                        mask,
                        pack_mode=rr.PackMode.B32_TO_B8,
                    )

    @jit
    def _finish_kv_row(
        self,
        ub_kv,
        ub_gamma,
        ub_sin,
        ub_cos,
        ub_kvb,
        ub_tail,
        token_row,
        table_row,
        norm_eps,
    ):
        """沿 head 特征轴归一化 KV，执行尾部 RoPE，再量化。
        KV 使用固定 scale=1。RoPE 特征对在寄存器中归一化，最后一个向量
        先在 FP32 暂存区拼接，再对齐写出字节。每行单独搬到其索引指定的 cache 位置。"""
        tiling = self.tiling
        head_width, rope_width = tiling.head_size, tiling.rope_width
        rope_half_width = rope_width // 2
        inv_head = 1.0 / head_width
        src_pitch = ub_kv.physical_stride[0]
        with vf(mode="simd"):
            full = rr.full_mask()
            m_half = rr.update_mask(rope_half_width, elem_bits=32)[0]
            one = rr.vdups(1.0, dtypes.float32, mask=full)
            kv_row_offset = token_row * src_pitch
            payload_offset = 0
            # 1. 计算沿 head 特征轴的 RMS 归一化因子。
            square_sum_accumulator = rr.vdups(0.0, dtypes.float32, mask=full)
            for head_vector_idx in range(head_width // VL):
                kv_values = rr.vload(ub_kv, kv_row_offset + head_vector_idx * VL)
                squared_values = rr.vmul(kv_values, kv_values, mask=full)
                square_sum_accumulator = rr.vadd(
                    square_sum_accumulator,
                    rr.vreduce_sum(squared_values, mask=full),
                    mask=full,
                )
            square_sum = rr.vdup(square_sum_accumulator, mask=full)
            mean_square = rr.vadds(
                rr.vmuls(square_sum, inv_head, mask=full), norm_eps, mask=full
            )
            rstd = rr.vdiv(one, rr.vsqrt(mean_square, mask=full), mask=full)

            # 2. 对 RoPE 之前的完整特征向量归一化并量化。
            for head_vector_idx in range(head_width // VL - 1):
                normed = rr.vmul(
                    rr.vload(ub_kv, kv_row_offset + head_vector_idx * VL),
                    rstd,
                    mask=full,
                )
                normed = rr.vmul(
                    normed, rr.vload(ub_gamma, head_vector_idx * VL), mask=full
                )
                rr.vstore_pack(
                    ub_kvb,
                    payload_offset + head_vector_idx * VL,
                    self._to_e4m3(normed, full),
                    full,
                    pack_mode=rr.PackMode.B32_TO_B8,
                )

            # 3. 在 FP32 暂存区拼接最后一个向量。字节打包后 RoPE 后半区
            # 可能不对齐，因此拼接完成后一次性打包整个向量。
            normed = rr.vmul(
                rr.vload(ub_kv, kv_row_offset + head_width - VL), rstd, mask=full
            )
            normed = rr.vmul(normed, rr.vload(ub_gamma, head_width - VL), mask=full)
            rr.vstore(ub_tail, 0, normed, full)
            rope_first_half, rope_second_half = self._rope_pair(
                ub_kv,
                kv_row_offset + head_width - rope_width,
                ub_gamma,
                head_width - rope_width,
                ub_sin,
                ub_cos,
                table_row,
                rstd,
                m_half,
            )
            rr.vstore(ub_tail, VL - rope_width, rope_first_half, m_half)
            rr.vstore(
                ub_tail, VL - rope_width + rope_half_width, rope_second_half, m_half
            )
            # 此屏障必须保留：紧接着的读取依赖上方刚写入的尾部数据。
            rr.vmem_bar("vst_vld")
            rr.vstore_pack(
                ub_kvb,
                payload_offset + head_width - VL,
                self._to_e4m3(rr.vload(ub_tail, 0), full),
                full,
                pack_mode=rr.PackMode.B32_TO_B8,
            )

    @jit
    def _rope_pair(
        self,
        src,
        src_feature_offset,
        gamma,
        gamma_feature_offset,
        ub_sin,
        ub_cos,
        table_row,
        rstd,
        mask,
    ):
        """将交错排列的特征对转换为 RoPE 输出的前后两个半区。
        first[i] = even[i] * cos[i] + odd[i] * sin[i]
        second[i] = even[i] * sin[half+i] + odd[i] * cos[half+i]
        符号由输入位置表携带，与 golden 的 INTERLEAVE_HALF 约定一致。
        rstd 为 None 时跳过归一化，用于 QB 分支。"""
        tiling = self.tiling
        rope_width = tiling.rope_width
        rope_half_width = rope_width // 2
        even, odd = rr.vload_deinterleave(src, src_feature_offset, width="b32")
        if const_expr(gamma is not None):
            g_even, g_odd = rr.vload_deinterleave(
                gamma, gamma_feature_offset, width="b32"
            )
            even = rr.vmul(rr.vmul(even, g_even, mask=mask), rstd, mask=mask)
            odd = rr.vmul(rr.vmul(odd, g_odd, mask=mask), rstd, mask=mask)
        # RoPE 表按补齐后的物理行跨度寻址。
        pitch = ub_sin.physical_stride[0]
        cos_first_half = rr.vload(ub_cos, table_row * pitch)
        sin_first_half = rr.vload(ub_sin, table_row * pitch)
        sin_second_half = rr.vload(ub_sin, table_row * pitch + rope_half_width)
        cos_second_half = rr.vload(ub_cos, table_row * pitch + rope_half_width)
        # 分别执行乘法和加法，保持既有的舍入顺序。
        rope_first_half = rr.vadd(
            rr.vmul(even, cos_first_half, mask=mask),
            rr.vmul(odd, sin_first_half, mask=mask),
            mask=mask,
        )
        rope_second_half = rr.vadd(
            rr.vmul(even, sin_second_half, mask=mask),
            rr.vmul(odd, cos_second_half, mask=mask),
            mask=mask,
        )
        return rope_first_half, rope_second_half

    # B 阶段

    @jit
    def _stage_b(
        self,
        gm_qr,
        gm_dqr3d,
        gm_wqb,
        gm_dswqb,
        gm_sin,
        gm_cos,
        gm_q,
        prefetched_qb_weights,
        prefetched_qb_scales,
        gm_gamma_kv,
        gm_index,
        gm_cache,
        gm_ws_kv,
        norm_eps,
    ):
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols, k_l1 = (
            tiling.bm_b,
            tiling.bn_b,
            tiling.bk_b,
            tiling.k_l1_b,
        )
        rank, head = tiling.rank_size, tiling.head_size
        panels_per_head = head // output_tile_cols
        scale_l1 = k_l1 // MX_GROUP
        scale_l0 = reduction_tile_cols // MX_GROUP
        groups_l1 = k_l1 // MX_PAIR_WIDTH

        # ---- 阶段 1：缓冲分配（b_resident 二分） ----
        # 驻留 QB 跨 head 复用一个 QR token 分块，其他路径沿 K 轴双缓冲。
        if const_expr(tiling.b_resident):
            # 每个 token 分块对应一次 Channel 轮转，所有 head 任务共用该视图。
            # 切换到下一个分块前，Channel 复用会插入 MTE1→MTE2 依赖。
            l1_a = RingBuffer(
                MemLoc.L1,
                FLOAT8_E4M3FN,
                make_layout((token_tile_rows, rank)),
                depth=1,
                data_format="nz",
            )
            # E8M0 scale 的 zn 分形不能用 data_format="zn" 直接分配（ 仅
            # 接受 nd/nz）：以 uint8 原始字节承载，选槽时 reinterpret 成 zn 视图。
            l1_sa = RingBuffer(
                MemLoc.L1,
                dtypes.uint8,
                make_layout(
                    (
                        _ceil_div(token_tile_rows, NZ_M_FRAC)
                        * _ceil_div(rank // MX_GROUP, MX_PAIR)
                        * ZN_FRACTAL_BYTES,
                    )
                ),
                depth=1,
                data_format="nd",
            )
        else:
            l1_a = RingBuffer(
                MemLoc.L1,
                FLOAT8_E4M3FN,
                make_layout((token_tile_rows, reduction_tile_cols)),
                depth=2,
                data_format="nz",
            )
            l1_sa = RingBuffer(
                MemLoc.L1,
                dtypes.uint8,
                make_layout(
                    (
                        _ceil_div(token_tile_rows, NZ_M_FRAC)
                        * _ceil_div(scale_l0, MX_PAIR)
                        * ZN_FRACTAL_BYTES,
                    )
                ),
                depth=2,
                data_format="nd",
            )
        # 与投影阶段相同，NZ 权重使用整数字节载体和 FP8 别名。
        if const_expr(tiling.b_resident):
            l1_b = [
                prefetched_qb_weights,
                RingBuffer(
                    MemLoc.L1,
                    dtypes.int8,
                    make_layout((output_tile_cols, k_l1)),
                    depth=1,
                    data_format="nz",
                ),
            ]
            l1_sb = prefetched_qb_scales
        else:
            l1_b = [
                RingBuffer(
                    MemLoc.L1,
                    dtypes.int8,
                    make_layout((output_tile_cols, k_l1)),
                    depth=1,
                    data_format="nz",
                )
                for _ in range(W_BUF)
            ]
            l1_sb = RingBuffer(
                MemLoc.L1,
                FLOAT8_E8M0,
                make_layout((scale_l1, output_tile_cols)),
                depth=W_BUF,
                data_format="nz",
            )
        l0a = RingBuffer(
            MemLoc.L0A,
            FLOAT8_E4M3FN,
            make_layout((token_tile_rows, reduction_tile_cols)),
            depth=L0_DEPTH,
        )
        l0b = RingBuffer(
            MemLoc.L0B,
            FLOAT8_E4M3FN,
            make_layout((output_tile_cols, reduction_tile_cols)),
            depth=1 if tiling.bn_b == BN_B_FULL_HEAD else L0_DEPTH,
        )
        l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout(
                (token_tile_rows, output_tile_cols if tiling.b_resident else head)
            ),
            depth=1,
        )
        # 沿特征轴二分，每个 Vector 核获得全部行和一半列。
        feature_cols = tiling.b_cols
        ub_acc = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((token_tile_rows, feature_cols)),
            depth=1,
        )
        # 输出 Channel 使用单槽，使 QB 的 UB 占用不超过容量。
        ub_out = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((token_tile_rows, feature_cols)),
            depth=1,
        )
        ub_tail = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, VL)))
        # RoPE 行补齐到 VL，特征轴放在最内层以保证读取对齐。
        ub_sin = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, VL)), depth=1
        )
        ub_cos = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, VL)), depth=1
        )

        # ---- 阶段 2：驻留主循环；阶段 3：通用任务循环 ----
        # ND 激活在 kernel 内转换为 NZ；预打包的 NZ 权重直接搬运。
        subblock = get_subblock_id()
        if const_expr(tiling.b_resident):
            # 各核完成当前 token 分块的所有 head 任务后再前进。
            # 每核对该分块的 QR/scale 和 RoPE 表只加载一次。
            for owner in range(tiling.owners_b):
                mem_copy(
                    l1_a.next_write(),
                    tile_slice(gm_qr, (token_tile_rows, rank), (owner, 0)),
                )
                mem_copy(
                    reinterpret(
                        l1_sa.next_write(),
                        dtype=FLOAT8_E8M0,
                        layout=make_layout((token_tile_rows, rank // MX_GROUP)),
                        data_format="zn",
                    ),
                    tile_slice(
                        gm_dqr3d,
                        (token_tile_rows, rank // MX_PAIR_WIDTH, MX_PAIR),
                        (owner, 0, 0),
                    ).flatten(1, 2),
                )
                a_ready = l1_a.next_read()
                sa_ready = reinterpret(
                    l1_sa.next_read(),
                    dtype=FLOAT8_E8M0,
                    layout=make_layout((token_tile_rows, rank // MX_GROUP)),
                    data_format="zn",
                )
                mem_copy(
                    ub_sin.next_write(),
                    tile_slice(gm_sin, (token_tile_rows, VL), (owner, 0)),
                )
                mem_copy(
                    ub_cos.next_write(),
                    tile_slice(gm_cos, (token_tile_rows, VL), (owner, 0)),
                )
                sin_tab = ub_sin.next_read()
                cos_tab = ub_cos.next_read()
                for head_tile in range(
                    get_block_idx(),
                    tiling.head_count * panels_per_head,
                    tiling.block_dim,
                ):
                    head_token_task_idx = head_tile * tiling.owners_b + owner
                    self._b_cube(
                        a_ready,
                        sa_ready,
                        l1_b,
                        l1_sb,
                        l0a,
                        l0b,
                        l0c,
                        ub_acc,
                        gm_qr,
                        gm_dqr3d,
                        gm_wqb,
                        gm_dswqb,
                        gm_q,
                        head_token_task_idx,
                        scale_l0,
                        groups_l1,
                    )
                    # 驻留 B 优先遍历 token 分块；owners_b 大于 1 时，
                    # 本地迭代号不能直接用任务编号除以核数计算。
                    b_step = owner * _ceil_div(
                        tiling.head_count * panels_per_head - get_block_idx(),
                        tiling.block_dim,
                    )
                    b_step += head_tile // tiling.block_dim
                    self._service_kv(
                        gm_gamma_kv,
                        gm_sin,
                        gm_cos,
                        gm_index,
                        gm_cache,
                        gm_ws_kv,
                        norm_eps,
                        get_block_idx() * AIV_PER_BLOCK
                        + subblock
                        + b_step * (tiling.block_dim * AIV_PER_BLOCK),
                    )
                    self._b_vec_reuse(
                        ub_acc,
                        ub_out,
                        ub_tail,
                        sin_tab,
                        cos_tab,
                        gm_q,
                        head_token_task_idx,
                        subblock,
                    )
        else:
            for head_token_task_idx in range(
                get_block_idx(), tiling.tiles_b, tiling.block_dim
            ):
                if head_token_task_idx < tiling.tiles_b:
                    self._b_cube(
                        l1_a,
                        l1_sa,
                        l1_b,
                        l1_sb,
                        l0a,
                        l0b,
                        l0c,
                        ub_acc,
                        gm_qr,
                        gm_dqr3d,
                        gm_wqb,
                        gm_dswqb,
                        gm_q,
                        head_token_task_idx,
                        scale_l0,
                        groups_l1,
                    )
                    self._service_kv(
                        gm_gamma_kv,
                        gm_sin,
                        gm_cos,
                        gm_index,
                        gm_cache,
                        gm_ws_kv,
                        norm_eps,
                        get_block_idx() * AIV_PER_BLOCK
                        + subblock
                        + (head_token_task_idx // tiling.block_dim)
                        * (tiling.block_dim * AIV_PER_BLOCK),
                    )
                    self._b_vec(
                        ub_acc,
                        ub_out,
                        ub_tail,
                        ub_sin,
                        ub_cos,
                        gm_sin,
                        gm_cos,
                        gm_q,
                        head_token_task_idx,
                        subblock,
                    )

        # ---- 阶段 4：剩余 KV 尾部补齐 ----
        # 各核的 B 任务数量可能不同，包括没有 B 任务的核在内，
        # 每个剩余 KV 分块都必须且只处理一次。
        if const_expr(
            tiling.tiles_a2
            > tiling.min_b_iterations * (tiling.block_dim * AIV_PER_BLOCK)
        ):
            if const_expr(tiling.b_resident):
                b_count = tiling.owners_b * _ceil_div(
                    tiling.head_count * panels_per_head - get_block_idx(),
                    tiling.block_dim,
                )
            else:
                b_count = _ceil_div(tiling.tiles_b - get_block_idx(), tiling.block_dim)
            for kv_slot in range(
                get_block_idx() * AIV_PER_BLOCK
                + subblock
                + b_count * (tiling.block_dim * AIV_PER_BLOCK),
                tiling.tiles_a2,
                tiling.block_dim * AIV_PER_BLOCK,
            ):
                self._service_kv(
                    gm_gamma_kv,
                    gm_sin,
                    gm_cos,
                    gm_index,
                    gm_cache,
                    gm_ws_kv,
                    norm_eps,
                    kv_slot,
                )

    @jit
    def _b_cube(
        self,
        l1_a,
        l1_sa,
        l1_b,
        l1_sb,
        l0a,
        l0b,
        l0c,
        ub_acc,
        gm_qr,
        gm_dqr3d,
        gm_wqb,
        gm_dswqb,
        gm_q,
        head_token_task_idx,
        scale_l0,
        groups_l1,
    ):
        """Cube 使用公开的 QR payload/scale 完成投影，并通过 FixPipe 发布结果。"""
        # ---- 阶段 1：任务解析与 L0C 累加器 ----
        tiling = self.tiling
        output_tile_cols, k_l1 = tiling.bn_b, tiling.k_l1_b
        rank, head = tiling.rank_size, tiling.head_size
        n_tiles = 1 if tiling.b_resident else head // output_tile_cols
        owner = head_token_task_idx % tiling.owners_b
        head_tile_idx = head_token_task_idx // tiling.owners_b
        n_kw = rank // k_l1
        accumulator = l0c.next_write()
        for n in range(n_tiles):
            # 公开 QR scale 已经具有 mx_scale_and 所需的
            # [token, 归约组对, pair] 布局。
            wrow = head_tile_idx * n_tiles + n
            # ---- 阶段 2：驻留双窗口轮转；阶段 3/4：偶数窗口成对展开与奇数窗口回退 ----
            # 成对展开窗口，使每个单槽权重 Channel 在一次循环体中只轮转一次。
            if const_expr(tiling.b_resident):
                # 当前权重窗口驻留在 L1 时，由 MTE2 加载下一窗口。
                # 首个窗口已在 __call__ 的 A2 期间预取。
                if head_token_task_idx != get_block_idx() * tiling.owners_b:
                    self._fill_w_b(l1_b[0], l1_sb, gm_wqb, gm_dswqb, wrow, 0, groups_l1)
                if const_expr(n_kw > 1):
                    self._fill_w_b(l1_b[1], l1_sb, gm_wqb, gm_dswqb, wrow, 1, groups_l1)
                for kwp in range(n_kw // W_BUF):
                    kw0 = kwp * W_BUF
                    self._mm_window_b(
                        l1_a,
                        l1_sa,
                        l1_b[0],
                        l1_sb,
                        l0a,
                        l0b,
                        accumulator,
                        gm_qr,
                        gm_dqr3d,
                        owner,
                        n,
                        kw0,
                        scale_l0,
                    )
                    if kw0 + W_BUF < n_kw:
                        self._fill_w_b(
                            l1_b[0],
                            l1_sb,
                            gm_wqb,
                            gm_dswqb,
                            wrow,
                            kw0 + W_BUF,
                            groups_l1,
                        )
                    self._mm_window_b(
                        l1_a,
                        l1_sa,
                        l1_b[1],
                        l1_sb,
                        l0a,
                        l0b,
                        accumulator,
                        gm_qr,
                        gm_dqr3d,
                        owner,
                        n,
                        kw0 + 1,
                        scale_l0,
                    )
                    if kw0 + W_BUF + 1 < n_kw:
                        self._fill_w_b(
                            l1_b[1],
                            l1_sb,
                            gm_wqb,
                            gm_dswqb,
                            wrow,
                            kw0 + W_BUF + 1,
                            groups_l1,
                        )
                if const_expr(n_kw % W_BUF != 0):
                    self._mm_window_b(
                        l1_a,
                        l1_sa,
                        l1_b[0],
                        l1_sb,
                        l0a,
                        l0b,
                        accumulator,
                        gm_qr,
                        gm_dqr3d,
                        owner,
                        n,
                        n_kw - 1,
                        scale_l0,
                    )
            elif const_expr(n_kw % W_BUF == 0):
                for kwp in range(n_kw // W_BUF):
                    kw0 = kwp * W_BUF
                    for u in range_constexpr(W_BUF):
                        self._fill_w_b(
                            l1_b[u],
                            l1_sb,
                            gm_wqb,
                            gm_dswqb,
                            wrow,
                            kw0 + u,
                            groups_l1,
                        )
                    for u in range_constexpr(W_BUF):
                        self._mm_window_b(
                            l1_a,
                            l1_sa,
                            l1_b[u],
                            l1_sb,
                            l0a,
                            l0b,
                            accumulator,
                            gm_qr,
                            gm_dqr3d,
                            owner,
                            n,
                            kw0 + u,
                            scale_l0,
                        )
            else:
                for kw in range(n_kw):
                    self._fill_w_b(
                        l1_b[0],
                        l1_sb,
                        gm_wqb,
                        gm_dswqb,
                        wrow,
                        kw,
                        groups_l1,
                    )
                    self._mm_window_b(
                        l1_a,
                        l1_sa,
                        l1_b[0],
                        l1_sb,
                        l0a,
                        l0b,
                        accumulator,
                        gm_qr,
                        gm_dqr3d,
                        owner,
                        n,
                        kw,
                        scale_l0,
                    )

        # ---- 阶段 5：FixPipe 双目的地发布 ----
        acc_slot = ub_acc.next_write()
        # FixPipe 双目的地（split-N）：L0C 源 partition(dim=1) 将特征列对半
        # 分给两个 Vector 核。
        mem_copy(
            acc_slot,
            accumulator.partition(dim=1),
            copy_options(unit_flag=FIXPIPE_FINAL),
        )

    @jit
    def _fill_w_b(self, l1_b, l1_sb, gm_wqb, gm_dswqb, wrow, kw, groups_l1):
        """加载一个 QB 权重窗口及其 scale，规则同 _fill_w_a；
        两函数分别绑定 A1/B 阶段的 tiling 字段，修改需同步。"""
        tiling = self.tiling
        output_tile_cols, k_l1 = tiling.bn_b, tiling.k_l1_b
        w_slot = l1_b.next_write()
        mem_copy(
            w_slot,
            tile_slice(gm_wqb, (output_tile_cols, k_l1), (wrow, kw)),
        )
        s_slot = l1_sb.next_write()
        mem_copy(
            s_slot,
            tile_slice(gm_dswqb, (output_tile_cols, groups_l1, MX_PAIR), (wrow, kw, 0))
            .permute(1, 2, 0)
            .flatten(0, 1),
        )

    @jit
    def _mm_window_b(
        self,
        l1_a,
        l1_sa,
        l1_b,
        l1_sb,
        l0a,
        l0b,
        l0c,
        gm_qr,
        gm_dqr3d,
        owner,
        n,
        kw,
        scale_l0,
    ):
        """使用 L0 归约分块累加一个 QB 权重窗口。"""
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols, k_l1 = (
            tiling.bm_b,
            tiling.bn_b,
            tiling.bk_b,
            tiling.k_l1_b,
        )
        k_tiles = tiling.rank_size // reduction_tile_cols
        w_ready = l1_b.next_read().reinterpret(dtype=FLOAT8_E4M3FN)
        s_ready = l1_sb.next_read()
        for kl0 in range(k_l1 // reduction_tile_cols):
            kb = kw * (k_l1 // reduction_tile_cols) + kl0
            if const_expr(tiling.b_resident):
                mem_copy(
                    l0a.next_write(),
                    tile_slice(l1_a, (token_tile_rows, reduction_tile_cols), (0, kb)),
                    copy_options(
                        mx_scale=tile_slice(l1_sa, (token_tile_rows, scale_l0), (0, kb))
                    ),
                )
            else:
                mem_copy(
                    l1_a.next_write(),
                    tile_slice(
                        gm_qr, (token_tile_rows, reduction_tile_cols), (owner, kb)
                    ),
                )
                mem_copy(
                    reinterpret(
                        l1_sa.next_write(),
                        dtype=FLOAT8_E8M0,
                        layout=make_layout((token_tile_rows, scale_l0)),
                        data_format="zn",
                    ),
                    tile_slice(
                        gm_dqr3d,
                        (
                            token_tile_rows,
                            reduction_tile_cols // MX_PAIR_WIDTH,
                            MX_PAIR,
                        ),
                        (owner, kb, 0),
                    ).flatten(1, 2),
                )
                mem_copy(
                    l0a.next_write(),
                    l1_a.next_read(),
                    copy_options(
                        mx_scale=reinterpret(
                            l1_sa.next_read(),
                            dtype=FLOAT8_E8M0,
                            layout=make_layout((token_tile_rows, scale_l0)),
                            data_format="zn",
                        )
                    ),
                )
            if const_expr(k_l1 == reduction_tile_cols):
                mem_copy(l0b.next_write(), w_ready, copy_options(mx_scale=s_ready))
            else:
                mem_copy(
                    l0b.next_write(),
                    tile_slice(
                        w_ready, (output_tile_cols, reduction_tile_cols), (0, kl0)
                    ),
                    copy_options(
                        mx_scale=tile_slice(
                            s_ready, (scale_l0, output_tile_cols), (kl0, 0)
                        )
                    ),
                )
            matmul(
                tile_slice(l0c, (token_tile_rows, output_tile_cols), (0, n)),
                l0a.next_read(),
                l0b.next_read(),
                init=(kb == 0),
                unit_flag=FIXPIPE_FINAL if kb + 1 == k_tiles else FIXPIPE_ACCUMULATE,
                mx_mode=True,
            )

    def _b_dst(self, gm_q, head_token_task_idx, subblock):
        """返回当前 Vector 核在 Q 中对应的 token 和特征切片。一个任务负责完整 head 或驻留半头；目的地特征块编号为2 * (task // owners_b) + subblock。"""
        tiling = self.tiling
        if const_expr(tiling.l1_reuse):
            dst = tile_slice(
                gm_q,
                (
                    tiling.bm_b // FIXPIPE_DUAL_DST,
                    tiling.head_size // FIXPIPE_DUAL_DST,
                ),
                (
                    FIXPIPE_DUAL_DST * (head_token_task_idx % tiling.owners_b)
                    + get_subblock_id(),
                    FIXPIPE_DUAL_DST * (head_token_task_idx // tiling.owners_b)
                    + subblock,
                ),
            )
        else:
            dst = tile_slice(
                gm_q,
                (tiling.bm_b, tiling.b_cols),
                (
                    head_token_task_idx % tiling.owners_b,
                    (head_token_task_idx // tiling.owners_b) * FIXPIPE_DUAL_DST
                    + subblock,
                ),
            )
        return dst

    @jit
    def _b_vec(
        self,
        ub_acc,
        ub_out,
        ub_tail,
        ub_sin,
        ub_cos,
        gm_sin,
        gm_cos,
        gm_q,
        head_token_task_idx,
        subblock,
    ):
        """对当前 Vector 核的 Q 切片执行尾部 RoPE，并转换为 BF16。"""
        tiling = self.tiling
        token_tile_rows = tiling.bm_b
        owner = head_token_task_idx % tiling.owners_b
        # 两个特征切片使用同一 token 对应的 RoPE 表行。
        q_half = self._b_dst(gm_q, head_token_task_idx, subblock)
        rows = q_half.shape[0]
        if const_expr(tiling.l1_reuse):
            q_accumulator = ub_acc.next_read()
            sin_tab, cos_tab = ub_sin, ub_cos
        else:
            sin_slot = ub_sin.next_write()
            mem_copy(sin_slot, tile_slice(gm_sin, (token_tile_rows, VL), (owner, 0)))
            cos_slot = ub_cos.next_write()
            mem_copy(cos_slot, tile_slice(gm_cos, (token_tile_rows, VL), (owner, 0)))
            q_accumulator = ub_acc.next_read()
            sin_tab = ub_sin.next_read()
            cos_tab = ub_cos.next_read()
        out_slot = ub_out.next_write()
        self._finish_q(
            q_accumulator,
            sin_tab,
            cos_tab,
            out_slot,
            ub_tail,
            rows,
            subblock,
            head_token_task_idx // tiling.owners_b,
        )
        #  mem_copy 以源域为准：非补齐 Q 输出的尾块需把 UB 源显式
        # 绑定到目的实际域，避免越界写满物理行。
        mem_copy(q_half, ub_out.next_read()[: q_half.shape[0], : q_half.shape[1]])

    @jit
    def _b_vec_reuse(
        self,
        ub_acc,
        ub_out,
        ub_tail,
        sin_tab,
        cos_tab,
        gm_q,
        head_token_task_idx,
        subblock,
    ):
        """复用当前 token 分块已加载到 UB 的 RoPE 表。"""
        tiling = self.tiling
        q_half = self._b_dst(gm_q, head_token_task_idx, subblock)
        rows = q_half.shape[0]
        q_accumulator = ub_acc.next_read()
        out_slot = ub_out.next_write()
        head_tile = head_token_task_idx // tiling.owners_b
        # RESIDENT_HEAD_PANELS：每 head 的 panel 半区数，仅后半半区（且第二目的地）含 RoPE。
        if head_tile % RESIDENT_HEAD_PANELS == 1 and subblock == 1:
            self._finish_q(
                q_accumulator,
                sin_tab,
                cos_tab,
                out_slot,
                ub_tail,
                rows,
                subblock,
                head_tile,
            )
        else:
            self._finish_q_plain(q_accumulator, out_slot, rows)
        # 同 _b_vec：尾块按目的实际域裁剪 UB 源。
        mem_copy(q_half, ub_out.next_read()[: q_half.shape[0], : q_half.shape[1]])

    @jit
    def _finish_q_plain(self, ub_acc, ub_out, rows):
        """对 head 中不涉及 RoPE 的三个四分块直接转换为 BF16。"""
        feature_cols = self.tiling.b_cols
        src_pitch = ub_acc.physical_stride[0]
        out_pitch = ub_out.physical_stride[0]
        with vf(mode="simd"):
            # 1. 读取 Q 向量；2. 舍入并打包为 BF16。
            full = rr.full_mask()
            for token_row in cannbotdsl.range(rows):
                q_row_offset = token_row * src_pitch
                bf16_row_offset = token_row * out_pitch
                for feature_vector_idx in range(feature_cols // VL):
                    rr.vstore_pack(
                        ub_out,
                        bf16_row_offset + feature_vector_idx * VL,
                        self._to_bf16(
                            rr.vload(ub_acc, q_row_offset + feature_vector_idx * VL),
                            full,
                        ),
                        full,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )

    @jit
    def _finish_q(
        self, ub_acc, ub_sin, ub_cos, ub_out, ub_tail, rows, subblock, head_tile
    ):
        """执行每个 head 的尾部 RoPE 并写出 BF16，不执行 RMSNorm。
        只有包含逻辑 head 尾部的 Vector 切片执行 RoPE；
        使用掩码控制，避免把 Channel 握手切入动态分支。"""
        tiling = self.tiling
        feature_cols, rope_width = tiling.b_cols, tiling.rope_width
        rope_half_width = rope_width // 2
        # head 较窄时，最后一个特征向量的宽度可能小于 VL。
        tail = min(VL, feature_cols)
        src_pitch = ub_acc.physical_stride[0]
        out_pitch = ub_out.physical_stride[0]
        if const_expr(
            not tiling.l1_reuse
            and tiling.template == "split_k"
            # 72：固定形状特判，此时 RoPE 尾部仅位于逻辑 head 后半区。
            and tiling.token_count == 72
        ):
            # 只有逻辑 head 的后半区包含 RoPE 尾部。
            rope_lanes = (
                0
                if head_tile % RESIDENT_HEAD_PANELS == 0
                else (0 if subblock == 0 else rope_half_width)
            )
        else:
            rope_lanes = 0 if subblock == 0 else rope_half_width
        with vf(mode="simd"):
            full = rr.full_mask()
            m_tail = rr.update_mask(tail, elem_bits=32)[0]
            m_half = rr.update_mask(rope_lanes, elem_bits=32)[0]
            # 1. 将前缀特征转换为 BF16。
            for token_row in cannbotdsl.range(rows):
                q_row_offset = token_row * src_pitch
                bf16_row_offset = token_row * out_pitch
                for feature_vector_idx in range((feature_cols - tail) // VL):
                    rr.vstore_pack(
                        ub_out,
                        bf16_row_offset + feature_vector_idx * VL,
                        self._to_bf16(
                            rr.vload(ub_acc, q_row_offset + feature_vector_idx * VL),
                            full,
                        ),
                        full,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )
                # 直接写入两个半区不满足对齐时，先拼接完整尾部向量。
                # 第一个切片的 RoPE 掩码为空，保留原始值。
                # 2. 执行尾部 RoPE；3. 打包并对齐写出 BF16。
                if const_expr(rope_width % UB_ALIGN == 0 and not tiling.l1_reuse):
                    # BF16 目的地址按 32 字节对齐，可直接写入互不重叠的前缀和
                    # 两个 RoPE 半区，避免经过 UB 暂存区。
                    prefix_mask = rr.update_mask(tail - 2 * rope_lanes, elem_bits=32)[0]
                    rr.vstore_pack(
                        ub_out,
                        bf16_row_offset + feature_cols - tail,
                        self._to_bf16(
                            rr.vload(ub_acc, q_row_offset + feature_cols - tail),
                            prefix_mask,
                        ),
                        prefix_mask,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )
                    rope_first_half, rope_second_half = self._rope_pair(
                        ub_acc,
                        q_row_offset + feature_cols - rope_width,
                        None,
                        0,
                        ub_sin,
                        ub_cos,
                        token_row,
                        None,
                        m_half,
                    )
                    rr.vstore_pack(
                        ub_out,
                        bf16_row_offset + feature_cols - rope_width,
                        self._to_bf16(rope_first_half, m_half),
                        m_half,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )
                    rr.vstore_pack(
                        ub_out,
                        bf16_row_offset + feature_cols - rope_half_width,
                        self._to_bf16(rope_second_half, m_half),
                        m_half,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )
                else:
                    rr.vstore(
                        ub_tail,
                        0,
                        rr.vload(ub_acc, q_row_offset + feature_cols - tail),
                        m_tail,
                    )
                    rope_first_half, rope_second_half = self._rope_pair(
                        ub_acc,
                        q_row_offset + feature_cols - rope_width,
                        None,
                        0,
                        ub_sin,
                        ub_cos,
                        token_row,
                        None,
                        m_half,
                    )
                    rr.vstore(ub_tail, tail - rope_width, rope_first_half, m_half)
                    rr.vstore(
                        ub_tail,
                        tail - rope_width + rope_half_width,
                        rope_second_half,
                        m_half,
                    )
                    rr.vmem_bar("vst_vld")
                    rr.vstore_pack(
                        ub_out,
                        bf16_row_offset + feature_cols - tail,
                        self._to_bf16(rr.vload(ub_tail, 0), m_tail),
                        m_tail,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )

    @jit
    def _to_bf16(self, value, mask):
        return rr.vcast(value, dtypes.bfloat16, mask=mask, rounding=rr.RoundingMode.RN)


# host 侧入口

# torch dtype 到 DSL dtype 的静态映射（TensorSpec 构造用）。
_TORCH_TO_DSL_DTYPE = {
    torch.int8: dtypes.int8,
    torch.uint8: dtypes.uint8,
    torch.int64: dtypes.int64,
    torch.float32: dtypes.float32,
    torch.bfloat16: dtypes.bfloat16,
}

_COMPILED_KERNELS = OrderedDict()
_COMPILE_LOCK = threading.Lock()


class _Launcher:
    def __init__(self, tiling: AttnPrologueTiling):
        self.tiling = tiling

    @host
    def run(
        self,
        gm_x,
        gm_dsx,
        gm_wqa,
        gm_dswqa,
        gm_wqb,
        gm_dswqb,
        gm_wkv,
        gm_dswkv,
        gm_gamma_qr,
        gm_gamma_kv,
        gm_sin,
        gm_cos,
        gm_index,
        gm_cache,
        gm_q,
        gm_qr,
        gm_dqr2d,
        gm_dqr3d,
        gm_ws_qa,
        gm_ws_kv,
        norm_eps: dtypes.float32,
    ):
        op = AttnPrologueKernel(self.tiling)
        op[self.tiling.block_dim](
            gm_x,
            gm_dsx,
            gm_wqa,
            gm_dswqa,
            gm_wqb,
            gm_dswqb,
            gm_wkv,
            gm_dswkv,
            gm_gamma_qr,
            gm_gamma_kv,
            gm_sin,
            gm_cos,
            gm_index,
            gm_cache,
            gm_q,
            gm_qr,
            gm_dqr2d,
            gm_dqr3d,
            gm_ws_qa,
            gm_ws_kv,
            norm_eps,
        )

    def compile_cached(self, args):
        """复用静态分块和张量契约相同的编译产物。

        norm_eps 保持为运行时 FP32 参数。缓存键和 TensorSpec 都必须包含 NZ
        信息，仅凭逻辑形状无法区分权重布局。缓存项不保存输入张量或交换区缓冲。
        """
        contracts, specs = [], []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                shape, stride = tuple(arg.shape), tuple(arg.stride())
                storage_format = (
                    "nz" if torch_npu.get_npu_format(arg) == FRACTAL_NZ else "nd"
                )
                contracts.append(
                    (shape, stride, arg.dtype, str(arg.device), storage_format)
                )
                if arg.dtype not in _TORCH_TO_DSL_DTYPE:
                    raise KeyError(arg.dtype)
                specs.append(
                    TensorSpec(
                        shape,
                        _TORCH_TO_DSL_DTYPE[arg.dtype],
                        stride=stride,
                        storage_format=storage_format,
                    )
                )
            else:
                contracts.append(dtypes.float32)
                specs.append(dtypes.float32)
        key = (self.tiling, tuple(contracts))
        with _COMPILE_LOCK:
            compiled = _COMPILED_KERNELS.get(key)
            if compiled is None:
                compiled = cannbotdsl.compile(self.run, *specs)
                _COMPILED_KERNELS[key] = compiled
                if len(_COMPILED_KERNELS) > MAX_COMPILED_KERNELS:
                    # 被淘汰的编译产物仍可能由 Prepared 对象持有，不能提前关闭。
                    _COMPILED_KERNELS.popitem(last=False)
            _COMPILED_KERNELS.move_to_end(key)
        return compiled


def _as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """以 int8 视图传递 FP8 原始字节，不转换数值，并保留 NZ 存储标记。"""
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        return tensor.view(torch.int8)
    return tensor


def to_nz(weight: torch.Tensor) -> torch.Tensor:
    """在加载静态权重时，将 ND 权重一次性转换为 FRACTAL_NZ。"""
    if weight.device.type != "npu":
        raise ValueError("Weight passed to to_nz must be an NPU tensor")
    return torch_npu.npu_format_cast(weight, FRACTAL_NZ)


def _require_nz(name: str, weight: torch.Tensor) -> torch.Tensor:
    fmt = torch_npu.get_npu_format(weight)
    if fmt != FRACTAL_NZ:
        raise ValueError(
            f"Weight {name} must use FRACTAL_NZ, got format={fmt}; "
            f"convert it once with to_nz({name}) during weight loading"
        )
    return weight


def _rope_table(table):
    """将每个 token 的位置表行从 rope_width 补齐至 VL，保证向量读取不越界。"""
    rows, width = table.shape
    if width == VL:
        return table.contiguous()
    padded = torch.zeros((rows, VL), dtype=table.dtype, device=table.device)
    padded[:, :width] = table
    return padded


def prepare_attn_prologue(
    x,
    wqa,
    wqb,
    wkv,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkv,
    norm_weight_qr,
    norm_weight_kv,
    rope_sin,
    rope_cos,
    cache_index,
    kv_cache,
    *,
    norm_eps: float = 1e-6,
):
    """校验输入，准备分块、缓冲区及可复用的启动参数。

    run() 返回 (q, qr, descale_qr)，并原地更新 kv_cache。
    payload 使用 E4M3FN，输入 scale 使用 E8M0；权重采用 NZ，其余输入采用 ND。
    核数由平台接口提供，分块方案自动选择。
    """
    _validate_attn_prologue_inputs(
        x,
        wqa,
        wqb,
        wkv,
        descale_x,
        descale_wqa,
        descale_wqb,
        descale_wkv,
        norm_weight_qr,
        norm_weight_kv,
        rope_sin,
        rope_cos,
        cache_index,
        kv_cache,
        norm_eps=norm_eps,
    )
    token_count, hidden_size = x.shape
    rank_size = wqa.shape[0]
    head_size = wkv.shape[0]
    head_count = wqb.shape[0] // head_size
    rope_width = rope_sin.shape[1]
    l1_reuse_eligible = _is_l1_reuse_eligible(x, wqa, wqb, wkv, rope_sin)
    tiling = AttnPrologueTiling.plan(
        token_count,
        hidden_size,
        rank_size,
        head_count,
        head_size,
        rope_width,
        l1_reuse_eligible=l1_reuse_eligible,
    )
    l1_reuse = tiling.l1_reuse
    device = x.device
    qb_operand = _pack_b_weight_panels(wqb) if l1_reuse else _as_bytes(wqb)
    # kernel 输入和尾行填充使用 E8M0 字节视图，不执行数值转换。
    descale_x = _as_bytes(descale_x)
    descale_wqa = _as_bytes(descale_wqa)
    descale_wqb = _as_bytes(descale_wqb)
    descale_wkv = _as_bytes(descale_wkv)
    # FP32 交换区跨核传递投影部分和，以便沿完整行归约。
    work_rows = tiling.m_tiles_a1 * tiling.bm_a1
    ws_qa = torch.empty(
        (work_rows * tiling.split_k, rank_size), dtype=torch.float32, device=device
    )
    ws_kv = torch.empty(
        (work_rows * tiling.split_k, head_size), dtype=torch.float32, device=device
    )
    pad_rows = tiling.owners_b * tiling.bm_b
    q_buf = torch.empty(
        (pad_rows if l1_reuse else token_count, head_count, head_size),
        dtype=torch.bfloat16,
        device=device,
    )
    q = q_buf[:token_count]
    if l1_reuse and pad_rows > token_count:
        # 即使 T=1，也用完整的物理行保持两个 FixPipe 目的地有效。
        # 公开 Q 裁剪回 T 行，填充行不参与 RMSNorm。
        rope_sin = torch.nn.functional.pad(rope_sin, (0, 0, 0, pad_rows - token_count))
        rope_cos = torch.nn.functional.pad(rope_cos, (0, 0, 0, pad_rows - token_count))
    # QB 输入行补齐为完整 FixPipe 分块，公开输出仅暴露逻辑 token 前缀。
    qr_buf = torch.empty((pad_rows, rank_size), dtype=torch.int8, device=device)
    dqr_buf = torch.empty(
        (pad_rows, _ceil_div(rank_size, MX_PAIR_WIDTH), MX_PAIR),
        dtype=torch.int8,
        device=device,
    )
    if pad_rows > token_count:
        # 确定性初始化填充行；matmul 各行独立，不影响有效输出。
        qr_buf[token_count:].zero_()
        dqr_buf[token_count:].zero_()
    qr = qr_buf[:token_count]
    descale_qr = dqr_buf[:token_count]

    if work_rows > token_count:
        x_padded = torch.zeros(
            (work_rows, hidden_size), dtype=torch.int8, device=device
        )
        x_padded[:token_count].copy_(_as_bytes(x))
        dsx_padded = torch.zeros(
            (work_rows, hidden_size // MX_PAIR_WIDTH, MX_PAIR),
            dtype=descale_x.dtype,
            device=device,
        )
        dsx_padded[:token_count].copy_(descale_x)
        x, descale_x = x_padded, dsx_padded
    run_args = (
        # payload 按字节传入，权重保留 NZ、scale 保留 ND，计算类型由 Channel 指定。
        _as_bytes(x),
        descale_x,
        _as_bytes(wqa),
        descale_wqa,
        qb_operand,
        descale_wqb,
        _as_bytes(wkv),
        descale_wkv,
        norm_weight_qr.reshape(1, rank_size),
        norm_weight_kv.reshape(1, head_size),
        _rope_table(rope_sin),
        _rope_table(rope_cos),
        cache_index.reshape(token_count, 1),
        kv_cache.view(-1, head_size),
        q_buf.view(-1, head_count * head_size),
        qr_buf,
        dqr_buf.view(pad_rows, tiling.qr_groups),
        dqr_buf,
        ws_qa,
        ws_kv,
        float(norm_eps),
    )
    return PreparedAttnPrologue(
        tiling,
        run_args,
        (q, qr.view(torch.float8_e4m3fn), descale_qr),
        (ws_qa.numel() + ws_kv.numel()) * FP32_BYTES,
    )


class PreparedAttnPrologue:
    def __init__(self, tiling, args, outputs, workspace_bytes):
        self.tiling, self.args, self.outputs = tiling, args, outputs
        self.workspace_bytes = workspace_bytes
        self.launcher = _Launcher(tiling)
        self._compiled = None

    def run(self):
        if self._compiled is None:
            self._compiled = self.launcher.compile_cached(self.args)
        self._compiled(*self.args)
        return self.outputs


def attn_prologue(*args, **kwargs):
    return prepare_attn_prologue(*args, **kwargs).run()


def _pack_b_weight_panels(weight: torch.Tensor) -> torch.Tensor:
    """一次性打包静态 QB 权重，不计入重复执行 kernel 的耗时。
    输入为逻辑形状 [N,1280] 的 int8/FP8 权重，支持 ND 或 NZ；
    输出为 ND int8 [N/256*4,81920]，每行按 [K1=10,N1=16,N0=16,K0=32] 排列。
    也可在模型加载时先打包 CPU ND 权重，再一次性传到设备。
    权重改变后必须重新打包，此处不隐式缓存权重。
    当前环境的 int8 NZ→ND 格式转换不能保持权重数值，因此在 CPU 上解析 NZ
    原始字节；这次 D2H/H2D 往返仅发生在准备阶段，不进入重复执行路径。"""
    if (
        weight.ndim != 2
        or weight.shape[1] != PANEL_RANK_WIDTH
        or weight.shape[0] % PANEL_OUTPUT_WIDTH
    ):
        raise ValueError(
            f"Panel packing requires [N,{PANEL_RANK_WIDTH}], N divisible by {PANEL_OUTPUT_WIDTH}"
        )
    if weight.dtype not in (torch.int8, torch.float8_e4m3fn):
        raise TypeError("Panel packing requires int8 or float8_e4m3fn")
    raw = _as_bytes(weight)
    if raw.device.type == "npu":
        device = raw.device
        if torch_npu.get_npu_format(raw) == FRACTAL_NZ:
            output_width, reduction_width = raw.shape
            if (
                raw.storage_offset() != 0
                or raw.untyped_storage().nbytes() != output_width * reduction_width
            ):
                raise ValueError(
                    "NZ packing requires an unpadded, whole weight allocation"
                )
            host_weight_bytes = torch.empty(
                output_width * reduction_width, dtype=torch.int8
            )
            copy_device_to_host = ctypes.CDLL("libascendcl.so").aclrtMemcpy
            copy_device_to_host.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            copy_device_to_host.restype = ctypes.c_int
            with torch.npu.device(device):
                torch.npu.synchronize(device)
                copy_status = copy_device_to_host(
                    host_weight_bytes.data_ptr(),
                    output_width * reduction_width,
                    raw.data_ptr(),
                    output_width * reduction_width,
                    ACL_MEMCPY_DEVICE_TO_HOST,
                )
                if copy_status:
                    raise RuntimeError(f"Weight D2H copy failed: {copy_status}")
            return (
                host_weight_bytes.view(
                    PANEL_REDUCTION_WINDOWS,
                    PANEL_K_FRACTALS,
                    output_width // PANEL_OUTPUT_WIDTH,
                    PANEL_N_FRACTALS,
                    NZ_M_FRAC,
                    NZ_C0_1B,
                )
                .permute(2, 0, 1, 3, 4, 5)
                .contiguous()
                .view(
                    output_width // PANEL_OUTPUT_WIDTH * PANEL_REDUCTION_WINDOWS,
                    PANEL_BYTES,
                )
                .to(device)
            )
        if torch_npu.get_npu_format(raw) not in ND_FORMATS:
            raise ValueError("Weight packing supports ND or NZ storage")
        return _pack_b_weight_panels(raw.cpu()).to(device)
    output_width = raw.shape[0]
    return (
        raw.contiguous()
        .view(
            output_width // PANEL_OUTPUT_WIDTH,
            PANEL_N_FRACTALS,
            NZ_M_FRAC,
            PANEL_REDUCTION_WINDOWS,
            PANEL_K_FRACTALS,
            NZ_C0_1B,
        )
        .permute(0, 3, 4, 1, 2, 5)
        .contiguous()
        .view(output_width // PANEL_OUTPUT_WIDTH * PANEL_REDUCTION_WINDOWS, PANEL_BYTES)
    )


def _validate_norm_eps(norm_eps):
    """第 1 层：先检查host 标量，再访问设备张量元数据。"""
    if isinstance(norm_eps, bool) or not isinstance(norm_eps, numbers.Real):
        raise TypeError("Parameter norm_eps must be a host real scalar")
    if not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError("Parameter norm_eps must be finite and positive")


def _validate_tensor_metadata(tensors):
    """第 2 层：检查公共张量类型、设备一致性及连续性。"""
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Input {name} must be a torch.Tensor")
    device = tensors["x"].device
    if device.type != "npu":
        raise ValueError("Input x must be an NPU tensor")
    for name, tensor in tensors.items():
        if tensor.device != device:
            raise ValueError(f"Input {name} must be on the same NPU as x")
        if not tensor.is_contiguous():
            raise ValueError(f"Input {name} must be contiguous")


def _require_dtype(name, tensor, dtype):
    if tensor.dtype != dtype:
        raise TypeError(f"Input {name} must have dtype {dtype}, got {tensor.dtype}")


def _require_shape(name, tensor, shape):
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"Input {name} must have shape {shape}, got {tuple(tensor.shape)}"
        )


def _validate_projection_inputs(x, wqa, wqb, wkv):
    """第 3a 层：检查 payload 元数据及三个投影的关联轴。"""
    for name, tensor in (("x", x), ("wqa", wqa), ("wqb", wqb), ("wkv", wkv)):
        if tensor.ndim != 2 or any(size <= 0 for size in tensor.shape):
            raise ValueError(f"Input {name} must be a nonempty matrix")
        _require_dtype(name, tensor, torch.float8_e4m3fn)
    _, hidden_size = x.shape
    rank_size, head_size = wqa.shape[0], wkv.shape[0]
    for name, width in (
        ("hidden_size", hidden_size),
        ("rank_size", rank_size),
        ("head_size", head_size),
    ):
        if width % MX_PAIR_WIDTH:
            raise ValueError(
                f"Dimension {name} must be divisible by {MX_PAIR_WIDTH}, got {width}"
            )
    if wqa.shape[1] != hidden_size or wkv.shape[1] != hidden_size:
        raise ValueError("Weights wqa and wkv must reduce over the input hidden axis")
    if wqb.shape[1] != rank_size:
        raise ValueError("Weight wqb must reduce over the QA rank axis")
    if wqb.shape[0] % head_size:
        raise ValueError("Weight wqb output width must be divisible by head_size")


def _validate_scale_inputs(tensors):
    """第 3b 层：每 MX_GROUP 个 payload 元素对应一个 E8M0 scale。"""
    for payload_name in ("x", "wqa", "wqb", "wkv"):
        scale_name = "descale_" + payload_name
        scale = tensors[scale_name]
        rows, reduction_width = tensors[payload_name].shape
        _require_dtype(scale_name, scale, torch.float8_e8m0fnu)
        _require_shape(
            scale_name, scale, (rows, reduction_width // MX_PAIR_WIDTH, MX_PAIR)
        )


def _validate_norm_inputs(norm_weight_qr, norm_weight_kv, rank_size, head_size):
    """第 3c 层：RMSNorm 权重必须覆盖对应的完整归约轴。"""
    for name, tensor, width in (
        ("norm_weight_qr", norm_weight_qr, rank_size),
        ("norm_weight_kv", norm_weight_kv, head_size),
    ):
        _require_shape(name, tensor, (width,))
        _require_dtype(name, tensor, torch.float32)


def _validate_rope_inputs(rope_sin, rope_cos, token_count, head_size):
    """第 3d 层：检查 RoPE 行数及尾部特征对齐约束。"""
    if rope_sin.ndim != 2:
        raise ValueError("RoPE tables must be 2-D with shape [T, Dr]")
    if (
        rope_sin.shape[0] != token_count
        or rope_sin.shape[1] <= 0
        or rope_cos.shape != rope_sin.shape
    ):
        raise ValueError("RoPE tables must share shape [T, Dr] with Dr > 0")
    rope_width = rope_sin.shape[1]
    if rope_width % 2 or rope_width > head_size:
        raise ValueError("RoPE width must be even and no larger than head_size")
    if rope_width % FRACTAL_M or rope_width > VL:
        # 两个 FP32 半区的起点都需要按 UB_ALIGN 字节对齐。
        raise NotImplementedError(
            f"RoPE width must be divisible by {FRACTAL_M} and at most {VL}, got {rope_width}"
        )
    if head_size // FIXPIPE_DUAL_DST < rope_width:
        raise NotImplementedError(
            "RoPE tail must fit in the second FixPipe feature slice"
        )
    _require_dtype("rope_sin", rope_sin, torch.float32)
    _require_dtype("rope_cos", rope_cos, torch.float32)


def _validate_cache_inputs(cache_index, kv_cache, token_count, head_size):
    """第 3e 层：检查 cache 存储；索引范围及唯一性由调用方保证。"""
    _require_shape("cache_index", cache_index, (token_count,))
    _require_dtype("cache_index", cache_index, torch.int64)
    if kv_cache.ndim != 4:
        raise ValueError("Cache kv_cache must have shape [P, BS, 1, D] with P, BS > 0")
    if (
        tuple(kv_cache.shape[2:]) != (1, head_size)
        or kv_cache.shape[0] <= 0
        or kv_cache.shape[1] <= 0
    ):
        raise ValueError("Cache kv_cache must have shape [P, BS, 1, D] with P, BS > 0")
    _require_dtype("kv_cache", kv_cache, torch.uint8)


def _validate_storage_layouts(tensors):
    """第 4 层：逻辑元数据检查通过后，再检查物理存储布局。"""
    for name, tensor in tensors.items():
        if name in ("wqa", "wqb", "wkv"):
            _require_nz(name, tensor)
        elif torch_npu.get_npu_format(tensor) not in ND_FORMATS:
            raise ValueError(f"Input {name} must use ND storage (format 0 or 2)")


def _validate_attn_prologue_inputs(
    x,
    wqa,
    wqb,
    wkv,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkv,
    norm_weight_qr,
    norm_weight_kv,
    rope_sin,
    rope_cos,
    cache_index,
    kv_cache,
    *,
    norm_eps,
):
    """按层次和计算模块校验输入，不读取设备张量值。分块容量及 NZ 网格检查仍由 AttnPrologueTiling 负责。cache_index 的取值范围和唯一性由调用方保证。"""
    _validate_norm_eps(norm_eps)
    tensors = dict(
        x=x,
        wqa=wqa,
        wqb=wqb,
        wkv=wkv,
        descale_x=descale_x,
        descale_wqa=descale_wqa,
        descale_wqb=descale_wqb,
        descale_wkv=descale_wkv,
        norm_weight_qr=norm_weight_qr,
        norm_weight_kv=norm_weight_kv,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        cache_index=cache_index,
        kv_cache=kv_cache,
    )
    _validate_tensor_metadata(tensors)
    _validate_projection_inputs(x, wqa, wqb, wkv)
    _validate_scale_inputs(tensors)
    _validate_norm_inputs(norm_weight_qr, norm_weight_kv, wqa.shape[0], wkv.shape[0])
    _validate_rope_inputs(rope_sin, rope_cos, x.shape[0], wkv.shape[0])
    _validate_cache_inputs(cache_index, kv_cache, x.shape[0], wkv.shape[0])
    _validate_storage_layouts(tensors)


def _is_l1_reuse_eligible(x, wqa, wqb, wkv, rope_sin) -> bool:
    """公共输入校验通过后，判断是否具备 L1 复用资格。"""
    tokens, hidden = x.shape
    if (
        not 1 <= tokens <= DECODE_TOKEN_LIMIT
        or (hidden, wqa.shape[0], wqb.shape[0], wkv.shape[0], rope_sin.shape[1])
        != RESIDENT_GEOMETRY
    ):
        return False
    # 驻留打包要求完整且没有额外填充的 NZ 权重存储。
    return wqb.storage_offset() == 0 and wqb.untyped_storage().nbytes() == wqb.numel()
