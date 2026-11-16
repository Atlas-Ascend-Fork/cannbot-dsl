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

import dataclasses
import math


from cannbotdsl import (
    dtypes,
    get_mem_size,
    get_platform_info,
)


FLOAT8_E4M3FN = dtypes.float8_e4m3fn
FLOAT8_E8M0 = dtypes.float8_e8m0

# MXFP8 编码常量，与独立 golden 保持一致。
MX_GROUP = 32  # 每个 E8M0 scale 对应的元素数。
MX_PAIR = 2  # 公开布局中相邻 scale 组的配对数。
E8M0_BIAS = 127
E4M3_MAX = 448.0  # E4M3FN 的最大有限值。
# KV cache 每 32 个 payload 元素后跟一个小端 BF16 scale。
KV_SCALE_BYTES = 2
# 支持的 Q head 数；wqb.shape[0] // head_size 必须落在该集合内。
SUPPORTED_HEAD_COUNTS = (16, 64)
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
BF16_BYTES = 2
L0C_RESERVED_BYTES = 16 * 1024
L1_ALIGN_RESERVE_BYTES = 4 * 1024  # 精确分配之外的 L1 对齐余量。

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
# head_count 支持 64（32768）与 16（8192），两者共用同一套驻留流水；其余几何回退 split-K。
RESIDENT_GEOMETRY = (5120, PANEL_RANK_WIDTH, 32768, 512, 64)
RESIDENT_GEOMETRIES = (RESIDENT_GEOMETRY, (5120, PANEL_RANK_WIDTH, 8192, 512, 64))

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

# ---- 动态 T 范围模板 ----
# decode/L1 复用路径按 range_max 规划一次编译契约，kernel 内再以
# cache_index 的实际行数决定 token 循环次数，避免每个 T 单独生成 kernel。
TOKEN_RANGE_UPPERS = (16, 32, 48, 64, 80, 96, 192, 256)


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def _token_range_upper(token_count: int) -> int | None:
    """返回覆盖 token_count 的模板上界；超出最大范围时返回 None（逐 T 特化）。"""
    for upper in TOKEN_RANGE_UPPERS:
        if token_count <= upper:
            return upper
    return None


def _align(value: int, alignment: int) -> int:
    """将 value 向上对齐到 alignment 的整数倍。"""
    return _ceil_div(value, alignment) * alignment


def _kv_row_bytes(head_size: int) -> int:
    """KV cache 单行字节数：payload 后紧跟每组的 BF16 scale。"""
    return head_size + (head_size // MX_GROUP) * KV_SCALE_BYTES


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
    def groups_b_pair(self):
        """twin 路径的 B 分组：job=(head group)，两个 token 分块内层，按单 owner 求解。"""
        return _resident_groups(
            1, self.head_count, self.block_dim, self.head_size * self.rank_size, 0
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
                    4,  # 4：L1 复用流水线沿 K 的固定分块数，A1 按 W_BUF 槽成对流水。
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
            (
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
                + 2 * _kv_row_bytes(head_width)
                + 2 * VL * FP32_BYTES
                + A2_BUF_DEPTH_DECODE * rank_width * BF16_BYTES  # BF16 QR 输出行。
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
            # round3：twin 路径把两个 token 分块的激活/QR 同时驻留，B 预取深度降为 2；
            # 容量改成与 kernel 实际分配逐一对应的精确求和（含 4 KiB 对齐余量）。
            twin_blocks = 2 if self.owners_b == 2 else 1
            resident_slots = self.resident_tiles * twin_blocks
            prefetch_windows = 2 if twin_blocks == 2 else RESIDENT_K_WINDOWS
            scale_slot_bytes = (
                _ceil_div(token_tile_rows, NZ_M_FRAC)
                * _ceil_div(
                    self.scale_k_tiles * reduction_tile_cols // MX_GROUP, MX_PAIR
                )
                * ZN_FRACTAL_BYTES
            )
            l1_used = (
                resident_slots * token_tile_rows * reduction_tile_cols
                + twin_blocks * self.resident_scale_tiles * scale_slot_bytes
                + W_BUF * output_tile_cols * reduction_tile_cols
                + (reduction_tile_cols // MX_GROUP) * output_tile_cols * W_BUF
                + RESIDENT_K_WINDOWS * (reduction_tile_cols // MX_GROUP) * head_width
                + prefetch_windows
                * (PANEL_OUTPUT_TILES * output_tile_cols)
                * reduction_tile_cols
            ) + L1_ALIGN_RESERVE_BYTES
            checks = [
                # 每 MX_GROUP 个 payload 字节另需一个 scale 字节。
                ("shared L1", l1_used, l1_cap),
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
                    (
                        twin_blocks
                        * token_tile_rows
                        * (PANEL_OUTPUT_TILES * output_tile_cols)
                        + twin_blocks * token_tile_rows * output_tile_cols
                        if twin_blocks == 2
                        else token_tile_rows * (output_tile_cols + head_width)
                    )
                    * FP32_BYTES,
                    l0c_budget,
                ),
                ("UB", ub_a2 + ub_b + _twin_ub_extra(self), ub_cap),
                (
                    "Cube channel IDs",
                    resident_slots
                    + twin_blocks * self.resident_scale_tiles
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
            + 2 * _kv_row_bytes(head_width)
            + 2 * VL * FP32_BYTES
            + a2_depth * m2 * rank_width * BF16_BYTES  # BF16 QR 输出行。
            + VL * FP32_BYTES  # KV scale 位打包暂存区。
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


def _twin_ub_extra(tiling) -> int:
    """twin 路径为两个 token 分块各驻留一组 sin/cos 表，深度 2 而非 1。"""
    if tiling.owners_b != 2:
        return 0
    half = tiling.bm_b // FIXPIPE_DUAL_DST
    return 2 * half * VL * FP32_BYTES
