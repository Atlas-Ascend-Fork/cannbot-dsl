# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FP8 全量化 GQA 融合推理注意力算子。

核心计算为：

    S = (Q_fp8 @ K_fp8^T) * deq_q * deq_k / sqrt(D)
    m = row_max - ln(quant_scale_p)
    P = cast_fp8(exp(S - m))
    O = (P @ V_fp8) * deq_v / sum(exp(S - m))

算子按 KV 维分块执行在线 Softmax，支持 GQA、PageAttention 缓存以及
``BNSD``/``TND`` 两种 Q/O 布局。公开入口为 :func:`flash_attn_fp8_fullquant`。
"""

from cannbotdsl import from_tensor

__all__ = ["flash_attn_fp8_fullquant"]

import math
import threading

from cannbotdsl.lang.host import host
import torch
import cannbotdsl

from cannbotdsl import (
    const_expr,
    dtypes,
    get_block_idx,
    get_subblock_id,
    jit,
    kernel,
    matmul,
    vf,
    MemLoc,
    RegLayout,
    RingBuffer,
    Tensor,
)
from cannbotdsl.buffer import Buffer
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.tensor import (
    ceil_div,
    make_layout,
    make_tiler,
    tile_slice,
)
from cannbotdsl.types.dtypes import float8_e4m3fn as Float8E4M3FN
from cannbotdsl.reg import (
    update_mask,
    vadd,
    vadds,
    varange,
    vcast,
    vle as vcmp_le,
    vdiv,
    vdups as vdup_scalar,
    vexp_sub,
    vgather_reg,
    vload,
    vload_broadcast as vload_brc,
    vload_deinterleave,
    vmadd,
    vmax,
    vmem_bar,
    vmul,
    vbitwise_or as vor,
    vselect,
    vstore,
    vstore_strided,
)

if __package__:
    from .flash_attn_fp8_fullquant_checker import (
        KEY_SCALE_BUFFER_SLOTS,
        validate_and_resolve,
    )
else:
    from flash_attn_fp8_fullquant_checker import (
        KEY_SCALE_BUFFER_SLOTS,
        validate_and_resolve,
    )

# 基础分块参数：一个 Cube 任务处理 128 行 Q 和 256 行 KV；两个 AIV
# 各处理其中 64 行 Q。PAGE_BLOCK_ROWS=128 是组成宽 KV tile 的物理缓存子块行数。
M_BASE_SIZE = 128  # 每个 Q tile 的行数
M_BASE_SIZE_PER_AIV = M_BASE_SIZE // 2  # 每个 AIV 处理的 Q 行数
S2_BASE_SIZE = 256  # 每次在线 Softmax 扫描的 KV 行数
D = 128  # head_dim
PAGE_BLOCK_ROWS = 128  # 单个 PageAttention 物理块的行数

# Vector 两阶段展开度：Pass A 每次处理 4 行，Pass B 每次处理 8 行。
PASS_A_UNROLL = 4

# P-NZ 的一个物理分片覆盖 32 个 KV 元素；两个相邻分片组成一个
# 64 元素宽的 L1 tile 区域。
P_NZ_FRAGMENT_WIDTH = 32
P_NZ_FRAGMENTS_PER_L1_TILE_ROW = 2

KEY_DEQUANT_SCALE_PADDING = 8
# K scale 保留 4 份副本，使连续的 4 次广播落到不同 bank。
KEY_DEQUANT_SCALE_COPIES = 4

# MM2 比当前 Softmax 落后两个 KV block，因此 P/corr 环形槽至少为 3。
SKEW_LAG = 2

# 当前 KV 块全部有效，无需屏蔽。
MASK_NONE = 0
# 按每个 Q 行各自的因果可见边界屏蔽未来的 KV 行。
MASK_CAUSAL = 1
# 关闭因果掩码时，屏蔽最后一个 KV 块中超出实际 S2 长度的填充行。
MASK_KV_TAIL = 2

# RingBuffer 槽位深度配置。Q、L0A/B 和 P-NZ 使用双 Buffer；K/V 共用四槽 L1；
# L0C 独立分配：MM1 单槽，MM2 双槽，共占 256 KB。
# P-L1 为适配两拍延迟使用三槽；K scale 使用两个独立 UB 槽 ping-pong。
DEPTHS = {
    "q_l1": 2,
    "kv_l1": 4,
    "p_l1": 3,
    "l0a": 2,
    "l0b": 2,
    "l0c_mm1": 1,
    "l0c_mm2": 2,
    "s_ub": 2,
    "pv_ub": 1,
    "p_nz": 2,
    "query_dequant_scale_ub": 1,
    "key_dequant_scale_ub": KEY_SCALE_BUFFER_SLOTS,
    "value_dequant_scale_ub": 1,
    "output_fp16_ub": 1,
}


def nz_block_stride(s2_base_size=S2_BASE_SIZE):
    """返回 P 散写的 DataBlock 步长。"""
    return (s2_base_size // 4) | 1


NZ_N1_PADDING = 32


def nz_gather_index(m_base_size_per_aiv=M_BASE_SIZE_PER_AIV):
    """生成带 padding 的 P-NZ 打包索引。"""
    return [i + (j << 2) for i in range(4) for j in range(m_base_size_per_aiv)]


KEY_SCALE_ROWS = 4

FP32_VECTOR_LANES = 64
NEG_MAX = -3.4028234663852886e38


def _query_row_tile(gm, layout, batch_idx, head_idx, query_token_start, rows, width):
    """按布局返回 Q 行 tile。"""
    alignment = (16, 32) if isinstance(rows, int) and width == D else (1, 1)
    if layout != "TND":
        base = gm[
            batch_idx, head_idx, query_token_start : query_token_start + rows, :width
        ]
        return tile_slice(base, make_tiler((rows, width), alignment=alignment), (0, 0))

    base = gm[head_idx, query_token_start : query_token_start + rows, :width]
    return tile_slice(base, make_tiler((rows, width), alignment=alignment), (0, 0))


class Matmul:
    """FlashAttention 的 Cube 数据搬运、MM1/MM2 与 Fixpipe。"""

    def __init__(self, m_base_size, s2_base_size, d, block_size, depths):
        self.m_base_size, self.s2_base_size, self.d = m_base_size, s2_base_size, d
        self.block_size = block_size
        self.cache_blocks_per_kv_tile = s2_base_size // block_size
        self.q_l1 = RingBuffer(
            MemLoc.L1, Float8E4M3FN, make_layout((m_base_size, d)), depth=depths["q_l1"]
        )
        self.kv_l1 = RingBuffer(
            MemLoc.L1,
            Float8E4M3FN,
            make_layout((s2_base_size, d)),
            depth=depths["kv_l1"],
        )
        self.l0a = RingBuffer(
            MemLoc.L0A,
            Float8E4M3FN,
            make_layout((s2_base_size, d)),
            depth=depths["l0a"],
        )
        self.l0b = RingBuffer(
            MemLoc.L0B,
            Float8E4M3FN,
            make_layout((s2_base_size, d)),
            depth=depths["l0b"],
        )
        self.l0c_mm1 = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((s2_base_size, m_base_size)),
            depth=depths["l0c_mm1"],
        )
        self.l0c_mm2 = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((m_base_size, d)),
            depth=depths["l0c_mm2"],
        )

    @jit
    def load_q(self, q_tile):
        q_slot = self.q_l1.next_write()
        mem_copy(q_slot, q_tile)
        return q_slot

    @jit
    def load_k(self, gm_key_cache, gm_block_table, batch_idx, kv_head, kv_block_idx):
        slot = self.kv_l1.next_write()
        for cache_block_offset in tuple(range(self.cache_blocks_per_kv_tile)):
            physical_cache_block_index = gm_block_table[
                batch_idx,
                kv_block_idx * self.cache_blocks_per_kv_tile + cache_block_offset,
            ]
            mem_copy(
                tile_slice(slot, (self.block_size, self.d), (cache_block_offset, 0)),
                tile_slice(
                    gm_key_cache[physical_cache_block_index, kv_head, None, None],
                    (self.block_size, self.d),
                    (0, 0),
                ),
            )

    @jit
    def compute_qk(self, score_ub, q_slot):
        key_l0a = self.l0a.next_write()
        # 复用较大 L0 槽位时，显式 NZ 布局按本次矩阵形状重建紧凑物理跨度。
        query_l0b = from_tensor(
            self.l0b.next_write(),
            Float8E4M3FN,
            make_layout((self.m_base_size, self.d)),
            data_format="nz",
        )
        score_l0c = self.l0c_mm1.next_write()
        mem_copy(key_l0a, self.kv_l1.next_read())
        mem_copy(query_l0b, q_slot)
        matmul(score_l0c, key_l0a, query_l0b, init=True)
        mem_copy(
            score_ub.next_write(),
            score_l0c.partition(dim=1),
        )

    @jit
    def load_v(self, gm_value_cache, gm_block_table, batch_idx, kv_head, kv_block_idx):
        slot = self.kv_l1.next_write()
        for cache_block_offset in tuple(range(self.cache_blocks_per_kv_tile)):
            physical_cache_block_index = gm_block_table[
                batch_idx,
                kv_block_idx * self.cache_blocks_per_kv_tile + cache_block_offset,
            ]
            mem_copy(
                tile_slice(slot, (self.block_size, self.d), (cache_block_offset, 0)),
                tile_slice(
                    gm_value_cache[physical_cache_block_index, kv_head, None, None],
                    (self.block_size, self.d),
                    (0, 0),
                ),
            )

    @jit
    def compute_pv(self, p_l1, output_ub):
        # P/V 的逻辑转置与目标 L0 形状一起声明，搬运按两端视图选择格式转换。
        probability_l0a = from_tensor(
            self.l0a.next_write(),
            Float8E4M3FN,
            make_layout((self.m_base_size, self.s2_base_size)),
            data_format="nz",
        )
        value_l0b = from_tensor(
            self.l0b.next_write(),
            Float8E4M3FN,
            make_layout((self.d, self.s2_base_size)),
            data_format="nz",
        )
        output_l0c = self.l0c_mm2.next_write()
        mem_copy(probability_l0a, p_l1.next_read().permute(1, 0))
        mem_copy(value_l0b, self.kv_l1.next_read().permute(1, 0))
        matmul(output_l0c, probability_l0a, value_l0b, init=True)
        mem_copy(
            output_ub.next_write(),
            output_l0c.partition(dim=0),
        )


class Vector:
    """FlashAttention 的在线 Softmax、P 打包、输出累加与归一化。"""

    def __init__(self, m_base_size_per_aiv, s2_base_size, d, depths):
        self.m_base_size_per_aiv, self.s2_base_size, self.d = (
            m_base_size_per_aiv,
            s2_base_size,
            d,
        )
        self.p_nz = RingBuffer(
            MemLoc.UB,
            Float8E4M3FN,
            make_layout((m_base_size_per_aiv, s2_base_size)),
            depth=depths["p_nz"],
            data_format="nz",
            physical_layout=make_layout(
                (s2_base_size // 32, m_base_size_per_aiv // 16, 16, 32),
                stride=(m_base_size_per_aiv * 32 + NZ_N1_PADDING, 16 * 32, 32, 1),
            ),
        )
        self.query_dequant_scale_ub = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((m_base_size_per_aiv, 1)),
            depth=depths["query_dequant_scale_ub"],
        )
        self.query_dequant_scale_buffer = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((m_base_size_per_aiv, 1))
        )
        self.key_dequant_scale_stride = s2_base_size + KEY_DEQUANT_SCALE_PADDING
        self.key_dequant_scale_ub_slots = [
            RingBuffer(
                MemLoc.UB,
                dtypes.float32,
                make_layout((KEY_DEQUANT_SCALE_COPIES, self.key_dequant_scale_stride)),
                depth=1,
            ).next_write()
            for _ in range(KEY_SCALE_BUFFER_SLOTS)
        ]
        self.key_dequant_scale_byte_views = [
            from_tensor(
                slot,
                dtypes.uint8,
                make_layout(
                    (KEY_DEQUANT_SCALE_COPIES, self.key_dequant_scale_stride * 4)
                ),
            )
            for slot in self.key_dequant_scale_ub_slots
        ]
        self.value_dequant_scale_ub = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((1, 1)),
            depth=depths["value_dequant_scale_ub"],
        )
        self.nz_gather_indices = nz_gather_index(m_base_size_per_aiv)
        self.nz_gather_indices_ub = Buffer(
            MemLoc.UB, dtypes.uint8, make_layout((1, len(self.nz_gather_indices)))
        )
        self.nz_gather_indices_slot = RingBuffer(
            MemLoc.UB,
            dtypes.uint8,
            make_layout((1, len(self.nz_gather_indices))),
            depth=1,
        ).next_write()
        self.output_fp16_ub = RingBuffer(
            MemLoc.UB,
            dtypes.float16,
            make_layout((m_base_size_per_aiv, d)),
            depth=depths["output_fp16_ub"],
        )
        self.acc = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((m_base_size_per_aiv, d))
        )
        self.row_m = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((m_base_size_per_aiv, 1))
        )
        self.row_l = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((m_base_size_per_aiv, 1))
        )
        self.corr_buffers = [
            Buffer(MemLoc.UB, dtypes.float32, make_layout((m_base_size_per_aiv, 1)))
            for _ in range(SKEW_LAG + 1)
        ]
        self.p_nz_fragment_width = P_NZ_FRAGMENT_WIDTH
        self.nz_group_rows = s2_base_size // 4
        self.nz_block_stride_value = nz_block_stride(s2_base_size)

    @jit
    def initialize(self):
        """初始化当前 Q tile 的 Softmax 状态和输出累加器。"""
        query_dequant_scale_slot = self.query_dequant_scale_ub.next_read()
        with vf(mode="simd"):
            row_mask, _ = update_mask(self.m_base_size_per_aiv, elem_bits=32)
            # 复用初始化 VF 完成 dq 的一次自动读取，供整个 S2 循环使用。
            vstore(
                self.query_dequant_scale_buffer,
                0,
                vload(query_dequant_scale_slot, 0),
                row_mask,
            )
            vstore(
                self.row_m,
                0,
                vdup_scalar(NEG_MAX, dtypes.float32, mask=row_mask),
                row_mask,
            )
            vstore(
                self.row_l, 0, vdup_scalar(0.0, dtypes.float32, mask=row_mask), row_mask
            )

        with vf(mode="simd"):
            mf, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
            zero = vdup_scalar(0.0, dtypes.float32, mask=mf)
            for row in range(self.m_base_size_per_aiv):
                for col in tuple(range(0, self.d, FP32_VECTOR_LANES)):
                    vstore(self.acc, row * self.d + col, zero, mf)

    @jit
    def softmax(
        self,
        score_ub,
        first_query_visible_limit,
        pipeline_slot,
        kv_block_mask_type,
        k_scale_slot,
        log_quant_scale_p,
    ):
        """对一个 KV 块执行反量化、mask 与在线 Softmax。

        递推公式为：

        ``m_new=max(m_old, max(S_j)-ln(quant_scale_p))``；
        ``corr=exp(m_old-m_new)``；
        ``P_j=cast_fp8(exp(S_j-m_new))``；
        ``l_new=l_old*corr+sum(exp(S_j-m_new))``。

        ``l`` 累加的是 Cast 前的 FP32 指数和，P 则以 FP8 NZ 形式供 MM2。
        """
        p_slot = self.p_nz.next_write()
        corr_buffer = self.corr_buffers[pipeline_slot]
        key_dequant_scale_ub = self.key_dequant_scale_ub_slots[
            int(const_expr(k_scale_slot))
        ]
        with vf(mode="simd"):
            fp32_mask, _ = update_mask(self.m_base_size_per_aiv, elem_bits=32)

            # Pass B 每个打包组包含四个 64-lane FP32 Vector，分别 Cast 到
            # FP8 字节位 0～3，再 OR 成 256 个 FP8 元素，因此 mask 覆盖四组 Q 行。
            fp8_mask, _ = update_mask(
                PASS_A_UNROLL * self.m_base_size_per_aiv, elem_bits=8
            )

            nz_gather_indices = vload(self.nz_gather_indices_ub, 0)

            query_dequant_scale_vector = vload(self.query_dequant_scale_buffer, 0)
            negative_max = vdup_scalar(NEG_MAX, dtypes.float32, mask=fp32_mask)

            visible_limit_vectors = []
            if const_expr(kv_block_mask_type != MASK_NONE):
                if const_expr(kv_block_mask_type == MASK_CAUSAL):
                    visible_limit_vector = vadd(
                        vdup_scalar(
                            first_query_visible_limit, dtypes.float32, mask=fp32_mask
                        ),
                        varange(0, dtypes.float32),
                        mask=fp32_mask,
                    )
                else:
                    visible_limit_vector = vdup_scalar(
                        first_query_visible_limit, dtypes.float32, mask=fp32_mask
                    )

                visible_limit_vectors.append(visible_limit_vector)
                # 本轮连续处理 4 个 KV 行，但比较时共用第一个行号 kv_group_start。
                # 将后 3 个边界分别减 1、2、3，等价于判断
                # kv_group_start+1、kv_group_start+2、kv_group_start+3
                # 是否仍未超过原始可见边界。
                visible_limit_vectors.append(
                    vadds(visible_limit_vector, -1.0, mask=fp32_mask)
                )
                visible_limit_vectors.append(
                    vadds(visible_limit_vector, -2.0, mask=fp32_mask)
                )
                visible_limit_vectors.append(
                    vadds(visible_limit_vector, -3.0, mask=fp32_mask)
                )

            kv_group_start = vdup_scalar(0.0, dtypes.float32, mask=fp32_mask)

            block_max = negative_max
            for kv_row_start in range(0, self.s2_base_size, PASS_A_UNROLL):
                raw_score_0 = vload(
                    score_ub, (kv_row_start + 0) * self.m_base_size_per_aiv
                )
                raw_score_1 = vload(
                    score_ub, (kv_row_start + 1) * self.m_base_size_per_aiv
                )
                raw_score_2 = vload(
                    score_ub, (kv_row_start + 2) * self.m_base_size_per_aiv
                )
                raw_score_3 = vload(
                    score_ub, (kv_row_start + 3) * self.m_base_size_per_aiv
                )

                dequantized_score_0 = vmul(
                    raw_score_0, query_dequant_scale_vector, mask=fp32_mask
                )
                dequantized_score_1 = vmul(
                    raw_score_1, query_dequant_scale_vector, mask=fp32_mask
                )
                dequantized_score_2 = vmul(
                    raw_score_2, query_dequant_scale_vector, mask=fp32_mask
                )
                dequantized_score_3 = vmul(
                    raw_score_3, query_dequant_scale_vector, mask=fp32_mask
                )

                # K scale UB 保存 4 份相同副本。前面的 0～3 选择不同副本，
                # 让连续四次广播落到不同 UB bank；末尾的 +0～+3 选择
                # 当前 Pass A 展开的四个连续 KV 行。
                key_scale_0 = vload_brc(
                    key_dequant_scale_ub,
                    0 * self.key_dequant_scale_stride + kv_row_start + 0,
                )
                key_scale_1 = vload_brc(
                    key_dequant_scale_ub,
                    1 * self.key_dequant_scale_stride + kv_row_start + 1,
                )
                key_scale_2 = vload_brc(
                    key_dequant_scale_ub,
                    2 * self.key_dequant_scale_stride + kv_row_start + 2,
                )
                key_scale_3 = vload_brc(
                    key_dequant_scale_ub,
                    3 * self.key_dequant_scale_stride + kv_row_start + 3,
                )
                dequantized_score_0 = vmul(
                    dequantized_score_0, key_scale_0, mask=fp32_mask
                )
                dequantized_score_1 = vmul(
                    dequantized_score_1, key_scale_1, mask=fp32_mask
                )
                dequantized_score_2 = vmul(
                    dequantized_score_2, key_scale_2, mask=fp32_mask
                )
                dequantized_score_3 = vmul(
                    dequantized_score_3, key_scale_3, mask=fp32_mask
                )

                if const_expr(kv_block_mask_type != MASK_NONE):
                    dequantized_score_0 = vselect(
                        dequantized_score_0,
                        negative_max,
                        cond_mask=vcmp_le(
                            kv_group_start, visible_limit_vectors[0], mask=fp32_mask
                        ),
                    )
                    dequantized_score_1 = vselect(
                        dequantized_score_1,
                        negative_max,
                        cond_mask=vcmp_le(
                            kv_group_start, visible_limit_vectors[1], mask=fp32_mask
                        ),
                    )
                    dequantized_score_2 = vselect(
                        dequantized_score_2,
                        negative_max,
                        cond_mask=vcmp_le(
                            kv_group_start, visible_limit_vectors[2], mask=fp32_mask
                        ),
                    )
                    dequantized_score_3 = vselect(
                        dequantized_score_3,
                        negative_max,
                        cond_mask=vcmp_le(
                            kv_group_start, visible_limit_vectors[3], mask=fp32_mask
                        ),
                    )
                    kv_group_start = vadds(
                        kv_group_start, float(PASS_A_UNROLL), mask=fp32_mask
                    )
                vstore(
                    score_ub,
                    (kv_row_start + 0) * self.m_base_size_per_aiv,
                    dequantized_score_0,
                    fp32_mask,
                )
                vstore(
                    score_ub,
                    (kv_row_start + 1) * self.m_base_size_per_aiv,
                    dequantized_score_1,
                    fp32_mask,
                )
                vstore(
                    score_ub,
                    (kv_row_start + 2) * self.m_base_size_per_aiv,
                    dequantized_score_2,
                    fp32_mask,
                )
                vstore(
                    score_ub,
                    (kv_row_start + 3) * self.m_base_size_per_aiv,
                    dequantized_score_3,
                    fp32_mask,
                )

                pair_max_01 = vmax(
                    dequantized_score_0, dequantized_score_1, mask=fp32_mask
                )
                pair_max_23 = vmax(
                    dequantized_score_2, dequantized_score_3, mask=fp32_mask
                )
                four_row_max = vmax(pair_max_01, pair_max_23, mask=fp32_mask)
                block_max = vmax(block_max, four_row_max, mask=fp32_mask)

            # 当前 KV 块在每个 Q 行上的最大 score，形状为 [64]。
            adjusted_block_max = vadds(block_max, -log_quant_scale_p, mask=fp32_mask)
            old_block_max = vload(self.row_m, 0)
            new_block_max = vmax(old_block_max, adjusted_block_max, mask=fp32_mask)
            current_block_corr = vexp_sub(old_block_max, new_block_max, mask=fp32_mask)
            vstore(self.row_m, 0, new_block_max, fp32_mask)

            # 保存当前 KV 块的修正系数；两拍后同一块完成 MM2 时，
            # 用它将历史输出分子 acc 修正到新的 Softmax 指数基准。
            vstore(corr_buffer, 0, current_block_corr, fp32_mask)

            # 同一 VF 内 Pass A 写回 score_ub，Pass B 随即重读，必须保证写后读顺序。
            vmem_bar("vst_vld")

            zero = vdup_scalar(0.0, dtypes.float32, mask=fp32_mask)
            partial_sum_0, partial_sum_1 = zero, zero
            partial_sum_2, partial_sum_3 = zero, zero
            partial_sum_4, partial_sum_5 = zero, zero
            partial_sum_6, partial_sum_7 = zero, zero

            for nz_group_row in range(0, self.nz_group_rows, 2):
                score_row_0 = vload(
                    score_ub,
                    (nz_group_row + 0 * self.nz_group_rows) * self.m_base_size_per_aiv,
                )
                score_row_1 = vload(
                    score_ub,
                    (nz_group_row + 1 * self.nz_group_rows) * self.m_base_size_per_aiv,
                )
                score_row_2 = vload(
                    score_ub,
                    (nz_group_row + 2 * self.nz_group_rows) * self.m_base_size_per_aiv,
                )
                score_row_3 = vload(
                    score_ub,
                    (nz_group_row + 3 * self.nz_group_rows) * self.m_base_size_per_aiv,
                )
                score_row_4 = vload(
                    score_ub,
                    (nz_group_row + 0 * self.nz_group_rows + 1)
                    * self.m_base_size_per_aiv,
                )
                score_row_5 = vload(
                    score_ub,
                    (nz_group_row + 1 * self.nz_group_rows + 1)
                    * self.m_base_size_per_aiv,
                )
                score_row_6 = vload(
                    score_ub,
                    (nz_group_row + 2 * self.nz_group_rows + 1)
                    * self.m_base_size_per_aiv,
                )
                score_row_7 = vload(
                    score_ub,
                    (nz_group_row + 3 * self.nz_group_rows + 1)
                    * self.m_base_size_per_aiv,
                )
                exp_score_0 = vexp_sub(score_row_0, new_block_max, mask=fp32_mask)
                exp_score_1 = vexp_sub(score_row_1, new_block_max, mask=fp32_mask)
                exp_score_2 = vexp_sub(score_row_2, new_block_max, mask=fp32_mask)
                exp_score_3 = vexp_sub(score_row_3, new_block_max, mask=fp32_mask)
                exp_score_4 = vexp_sub(score_row_4, new_block_max, mask=fp32_mask)
                exp_score_5 = vexp_sub(score_row_5, new_block_max, mask=fp32_mask)
                exp_score_6 = vexp_sub(score_row_6, new_block_max, mask=fp32_mask)
                exp_score_7 = vexp_sub(score_row_7, new_block_max, mask=fp32_mask)
                partial_sum_0 = vadd(partial_sum_0, exp_score_0, mask=fp32_mask)
                partial_sum_1 = vadd(partial_sum_1, exp_score_1, mask=fp32_mask)
                partial_sum_2 = vadd(partial_sum_2, exp_score_2, mask=fp32_mask)
                partial_sum_3 = vadd(partial_sum_3, exp_score_3, mask=fp32_mask)
                partial_sum_4 = vadd(partial_sum_4, exp_score_4, mask=fp32_mask)
                partial_sum_5 = vadd(partial_sum_5, exp_score_5, mask=fp32_mask)
                partial_sum_6 = vadd(partial_sum_6, exp_score_6, mask=fp32_mask)
                partial_sum_7 = vadd(partial_sum_7, exp_score_7, mask=fp32_mask)

                fp8_exp_0 = vcast(
                    exp_score_0,
                    Float8E4M3FN,
                    mask=fp32_mask,
                    saturate=True,
                    reg_layout=RegLayout.ZERO,
                )
                fp8_exp_1 = vcast(
                    exp_score_1,
                    Float8E4M3FN,
                    mask=fp32_mask,
                    saturate=True,
                    reg_layout=RegLayout.ONE,
                )
                fp8_exp_2 = vcast(
                    exp_score_2,
                    Float8E4M3FN,
                    mask=fp32_mask,
                    saturate=True,
                    reg_layout=RegLayout.TWO,
                )
                fp8_exp_3 = vcast(
                    exp_score_3,
                    Float8E4M3FN,
                    mask=fp32_mask,
                    saturate=True,
                    reg_layout=RegLayout.THREE,
                )
                packed_exp_group_0 = vor(
                    vor(fp8_exp_0, fp8_exp_1, mask=fp8_mask),
                    vor(fp8_exp_2, fp8_exp_3, mask=fp8_mask),
                    mask=fp8_mask,
                )

                vstore_strided(
                    p_slot,
                    (nz_group_row + 0) * self.p_nz_fragment_width,
                    vgather_reg(packed_exp_group_0, nz_gather_indices),
                    fp8_mask,
                    block_stride=self.nz_block_stride_value,
                    repeat_stride=0,
                )

                fp8_exp_4 = vcast(
                    exp_score_4,
                    Float8E4M3FN,
                    mask=fp32_mask,
                    saturate=True,
                    reg_layout=RegLayout.ZERO,
                )
                fp8_exp_5 = vcast(
                    exp_score_5,
                    Float8E4M3FN,
                    mask=fp32_mask,
                    saturate=True,
                    reg_layout=RegLayout.ONE,
                )
                fp8_exp_6 = vcast(
                    exp_score_6,
                    Float8E4M3FN,
                    mask=fp32_mask,
                    saturate=True,
                    reg_layout=RegLayout.TWO,
                )
                fp8_exp_7 = vcast(
                    exp_score_7,
                    Float8E4M3FN,
                    mask=fp32_mask,
                    saturate=True,
                    reg_layout=RegLayout.THREE,
                )
                packed_exp_group_1 = vor(
                    vor(fp8_exp_4, fp8_exp_5, mask=fp8_mask),
                    vor(fp8_exp_6, fp8_exp_7, mask=fp8_mask),
                    mask=fp8_mask,
                )
                vstore_strided(
                    p_slot,
                    (nz_group_row + 1) * self.p_nz_fragment_width,
                    vgather_reg(packed_exp_group_1, nz_gather_indices),
                    fp8_mask,
                    block_stride=self.nz_block_stride_value,
                    repeat_stride=0,
                )

            current_block_sum = vadd(
                vadd(
                    vadd(partial_sum_0, partial_sum_1, mask=fp32_mask),
                    vadd(partial_sum_2, partial_sum_3, mask=fp32_mask),
                    mask=fp32_mask,
                ),
                vadd(
                    vadd(partial_sum_4, partial_sum_5, mask=fp32_mask),
                    vadd(partial_sum_6, partial_sum_7, mask=fp32_mask),
                    mask=fp32_mask,
                ),
                mask=fp32_mask,
            )
            vstore(
                self.row_l,
                0,
                vadd(
                    vmul(vload(self.row_l, 0), current_block_corr, mask=fp32_mask),
                    current_block_sum,
                    mask=fp32_mask,
                ),
                fp32_mask,
            )

    @jit
    def store_p(self, p_l1, vector_subblock_index):
        """将全部带 padding 的 P-NZ 分片重排写入同一个 L1 槽位。"""
        p_source = self.p_nz.next_read()
        p_destination = p_l1.next_write()
        fragment_shape = (self.m_base_size_per_aiv, self.p_nz_fragment_width)
        fragment_count = self.s2_base_size // self.p_nz_fragment_width
        for fragment_index in tuple(range(fragment_count)):
            mem_copy(
                tile_slice(
                    p_destination,
                    fragment_shape,
                    (
                        fragment_index // P_NZ_FRAGMENTS_PER_L1_TILE_ROW,
                        vector_subblock_index * P_NZ_FRAGMENTS_PER_L1_TILE_ROW
                        + fragment_index % P_NZ_FRAGMENTS_PER_L1_TILE_ROW,
                    ),
                ),
                tile_slice(p_source, fragment_shape, (0, fragment_index)),
            )

    @jit
    def update_output_acc(self, pv_ub, pipeline_slot):
        """按 ``acc=acc*corr+PV`` 更新在线输出累加值。"""
        corr_buffer = self.corr_buffers[pipeline_slot]
        with vf(mode="simd"):
            for query_row_index in range(self.m_base_size_per_aiv):
                corr_broadcast = vload_brc(corr_buffer, query_row_index)
                for d_offset in tuple(range(0, self.d, FP32_VECTOR_LANES)):
                    fp32_mask, _ = update_mask(self.d - d_offset, elem_bits=32)
                    output_offset = query_row_index * self.d + d_offset
                    vstore(
                        self.acc,
                        output_offset,
                        vmadd(
                            vload(self.acc, output_offset),
                            corr_broadcast,
                            vload(pv_ub, output_offset),
                            mask=fp32_mask,
                        ),
                        fp32_mask,
                    )

    @jit
    def finalize_output(
        self,
        gm_output,
        gm_value_dequant_scale,
        layout,
        vector_subblock_index,
        batch_index,
        query_head_index,
        query_token_start,
        key_value_head_index,
        valid_query_rows,
    ):
        """按 ``O=acc*deq_v/row_l`` 归一化，Cast 为 FP16 后写回 GM。"""
        value_dequant_scale_slot = self.value_dequant_scale_ub.next_write()
        output_slot = self.output_fp16_ub.next_write()
        mem_copy(
            value_dequant_scale_slot,
            tile_slice(
                gm_value_dequant_scale[key_value_head_index, None, None], (1, 1), (0, 0)
            ),
        )
        with vf(mode="simd"):
            fp32_mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
            fp16_mask, _ = update_mask(2 * FP32_VECTOR_LANES, elem_bits=16)
            for query_row_index in range(self.m_base_size_per_aiv):
                normalization_denominator = vdiv(
                    vload_brc(self.row_l, query_row_index),
                    vload_brc(value_dequant_scale_slot, 0),
                    mask=fp32_mask,
                )
                row_offset = query_row_index * self.d
                even_acc, odd_acc = vload_deinterleave(
                    self.acc, row_offset, width="b32"
                )
                even_acc = vdiv(even_acc, normalization_denominator, mask=fp32_mask)
                odd_acc = vdiv(odd_acc, normalization_denominator, mask=fp32_mask)
                even_fp16 = vcast(
                    even_acc, dtypes.float16, mask=fp32_mask, reg_layout=RegLayout.ZERO
                )
                odd_fp16 = vcast(
                    odd_acc, dtypes.float16, mask=fp32_mask, reg_layout=RegLayout.ONE
                )
                vstore(
                    output_slot,
                    row_offset,
                    vor(even_fp16, odd_fp16, mask=fp16_mask),
                    fp16_mask,
                )

        # 两端都限定有效行；空 AIV 不写回，避免尾 tile 覆盖下一段 TND 序列。
        if valid_query_rows > 0:
            mem_copy(
                _query_row_tile(
                    gm_output,
                    layout,
                    batch_index,
                    query_head_index,
                    query_token_start
                    + vector_subblock_index * self.m_base_size_per_aiv,
                    valid_query_rows,
                    self.d,
                ),
                output_slot[:valid_query_rows, : self.d],
            )


@kernel
# 设备侧 Cube/Vector 流水
class FlashAttnFp8FullQuantKernel:
    """FP8 全量化 GQA 的设备侧 Cube/Vector 流水。

    每个 Q tile 只搬入一次 Q/deq_q。随后对每个 256 行 KV block 执行：

    1. K 搬入 L1，MM1 计算 ``S_raw = K @ Q^T``；
    2. Vector 完成反量化、因果 mask、在线 Softmax，并把 P 打包为 FP8 NZ；
    3. P 搬入 L1，MM2 计算 ``PV = P^T @ V``；
    4. 用 ``acc = acc*corr + PV`` 更新输出，其中
       ``corr = exp(m_old-m_new)``。

    MM2 落后 Softmax 两拍，扫描结束后统一排空流水并归一化写回 FP16。
    """

    def __init__(
        self,
        m_base_size=M_BASE_SIZE,
        s2_base_size=S2_BASE_SIZE,
        d=D,
        block_size=None,
        depths=None,
        mask_tail=1,
        layout="BNSD",
    ):
        self.m_base_size, self.s2_base_size, self.d = m_base_size, s2_base_size, d
        self.m_base_size_per_aiv = m_base_size_per_aiv = m_base_size // 2

        self.block_size = bs = s2_base_size if block_size is None else block_size
        self.cache_blocks_per_kv_tile = s2_base_size // bs

        self.layout = layout

        self.tok_base_mul = 1 if layout == "TND" else 0

        self.mask_tail = mask_tail

        dep = dict(DEPTHS if depths is None else depths)

        self.vector_subblock_index = get_subblock_id()
        self.kernel_block_index = get_block_idx()
        self.vector = Vector(m_base_size_per_aiv, s2_base_size, d, dep)

        self.matmul = Matmul(m_base_size, s2_base_size, d, bs, dep)

        self.p_l1 = RingBuffer(
            MemLoc.L1,
            Float8E4M3FN,
            make_layout((s2_base_size, m_base_size)),
            depth=dep["p_l1"],
        )

        self.s_ub = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((s2_base_size, m_base_size_per_aiv)),
            depth=dep["s_ub"],
        )
        self.pv_ub = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((m_base_size_per_aiv, d)),
            depth=dep["pv_ub"],
        )

    def __call__(
        self,
        gm_query: Tensor,
        gm_key_cache: Tensor,
        gm_value_cache: Tensor,
        gm_block_table: Tensor,
        gm_query_dequant_scale: Tensor,
        gm_value_dequant_scale: Tensor,
        gm_core_task_ranges: Tensor,
        gm_query_sequence_boundaries: Tensor,
        gm_key_value_lengths: Tensor,
        gm_nz_gather_indices: Tensor,
        gm_output: Tensor,
        batch: int,
        n_q_heads: int,
        head_group: int,
        gm_log_quant_scale_p: Tensor,
    ):
        """执行当前核负责的 Q tile，并沿 S2 方向完成融合 Attention 流水。"""
        log_quant_scale_p = gm_log_quant_scale_p[0, 0]

        # NZ gather 索引在整个 Kernel 生命周期内固定，每个核只需搬入 UB 一次。
        mem_copy(
            self.vector.nz_gather_indices_slot,
            gm_nz_gather_indices,
        )
        with vf(mode="simd"):
            idx_mask, _ = update_mask(len(self.vector.nz_gather_indices), elem_bits=8)
            vstore(
                self.vector.nz_gather_indices_ub,
                0,
                vload(self.vector.nz_gather_indices_slot, 0),
                idx_mask,
            )

        # Metadata 为每个核分配一段连续逻辑任务；每个任务对应一个 Q tile 和一个 Q head。
        tile_begin = gm_core_task_ranges[0, self.kernel_block_index]
        tile_end = gm_core_task_ranges[1, self.kernel_block_index]
        # MM2 比 MM1/Softmax 落后 SKEW_LAG 拍，因此需要 SKEW_LAG+1 个轮转槽。
        pipeline_slot_count = SKEW_LAG + 1
        for tile_idx in range(tile_begin, tile_end):
            # 将全局线性任务号反解为 batch、Q head 和 batch 内的 Q tile。
            # 前面各 batch 累计的 Q tile 数。
            accumulated_task_count = 0
            # 当前判断所属的 batch，默认第 0 个。
            batch_idx = 0
            # 该 batch 之前共有多少个 Q tile。
            batch_task_start = 0
            # 前一个 Q 累计长度边界，通常初始为 0。
            previous_q_boundary = gm_query_sequence_boundaries[0, 0]
            for candidate_batch in range(1, batch):
                current_q_boundary = gm_query_sequence_boundaries[0, candidate_batch]
                accumulated_task_count = accumulated_task_count + n_q_heads * ceil_div(
                    current_q_boundary - previous_q_boundary, self.m_base_size
                )
                belongs_to_later_batch = tile_idx >= accumulated_task_count
                batch_idx = candidate_batch if belongs_to_later_batch else batch_idx
                batch_task_start = (
                    accumulated_task_count
                    if belongs_to_later_batch
                    else batch_task_start
                )
                previous_q_boundary = current_q_boundary

            # batch 确定后，再按“head 内连续 Q tile”的排列方式解码任务坐标。
            task_index_in_batch = tile_idx - batch_task_start
            batch_query_token_start = gm_query_sequence_boundaries[0, batch_idx]
            query_length = (
                (gm_query_sequence_boundaries[0, batch_idx + 1])
                - batch_query_token_start
            )
            query_tiles_per_head = ceil_div(query_length, self.m_base_size)
            head_idx = task_index_in_batch // query_tiles_per_head
            q_tile_idx = task_index_in_batch - head_idx * query_tiles_per_head

            # TND 使用 batch 的累计 token 起点；BNSD 通过 tok_base_mul 保持原布局寻址。
            query_token_start = (
                batch_query_token_start * self.tok_base_mul
                + q_tile_idx * self.m_base_size
            )

            key_value_length = gm_key_value_lengths[0, batch_idx]
            # 右下 causal 对齐时，KV 比 Q 多出的前缀长度需要加入可见边界。
            causal_offset = key_value_length - query_length

            rows = min(self.m_base_size, query_length - q_tile_idx * self.m_base_size)
            kv_head = head_idx // head_group

            # off 扫描全部有效 KV 块；on 只扫描当前 Q tile 能够看到的 KV 范围。
            kv_block_capacity = ceil_div(key_value_length, self.s2_base_size)
            if const_expr(self.mask_tail == 0):
                kv_block_count = kv_block_capacity
            else:
                kv_block_count = min(
                    kv_block_capacity,
                    ceil_div(
                        (q_tile_idx + 1) * self.m_base_size + causal_offset,
                        self.s2_base_size,
                    ),
                )

            # Q 和 Q scale 在整个 S2 扫描期间保持不变，每个 Q tile 只搬入一次。
            q_slot = self._load_query_and_scale(
                gm_query, gm_query_dequant_scale, batch_idx, head_idx, query_token_start
            )
            # 清零在线 Softmax 状态和输出累加器，开始处理新的 Q tile。
            self.vector.initialize()
            if const_expr(self.mask_tail == 0):
                # off 关闭因果 mask，但最后一个 KV 块仍需屏蔽长度外 padding。
                mask_begin = kv_block_count - 1
            else:
                mask_begin = kv_block_count - self.mask_tail

            # 主流水每拍为当前 KV 块生产 S/P；从第三拍开始，同时消费两拍前的 P/V。
            # 根据 KV 块位置选择普通、causal 或 KV 尾块 mask，三类路径共用同一发射循环。
            for kv_block_idx in range(kv_block_count):
                # K scale 使用两个 UB 槽按 KV 块奇偶交替，避免覆盖仍在使用的数据。
                k_scale_slot = kv_block_idx % KEY_SCALE_BUFFER_SLOTS
                if k_scale_slot == 0:
                    self._stage_kscale(
                        gm_block_table,
                        gm_key_cache,
                        gm_value_dequant_scale,
                        batch_idx,
                        kv_head,
                        kv_block_idx,
                        0,
                    )
                else:
                    self._stage_kscale(
                        gm_block_table,
                        gm_key_cache,
                        gm_value_dequant_scale,
                        batch_idx,
                        kv_head,
                        kv_block_idx,
                        1,
                    )
                self._stage_qk_auto(
                    gm_block_table,
                    gm_key_cache,
                    batch_idx,
                    kv_head,
                    kv_block_idx,
                    q_slot,
                )
                if kv_block_idx >= mask_begin:
                    if const_expr(self.mask_tail == 0):
                        if k_scale_slot == 0:
                            self._stage_softmax_p(
                                kv_block_idx,
                                q_tile_idx,
                                causal_offset,
                                key_value_length,
                                kv_block_idx % pipeline_slot_count,
                                MASK_KV_TAIL,
                                0,
                                log_quant_scale_p,
                            )
                        else:
                            self._stage_softmax_p(
                                kv_block_idx,
                                q_tile_idx,
                                causal_offset,
                                key_value_length,
                                kv_block_idx % pipeline_slot_count,
                                MASK_KV_TAIL,
                                1,
                                log_quant_scale_p,
                            )
                    else:
                        if k_scale_slot == 0:
                            self._stage_softmax_p(
                                kv_block_idx,
                                q_tile_idx,
                                causal_offset,
                                key_value_length,
                                kv_block_idx % pipeline_slot_count,
                                MASK_CAUSAL,
                                0,
                                log_quant_scale_p,
                            )
                        else:
                            self._stage_softmax_p(
                                kv_block_idx,
                                q_tile_idx,
                                causal_offset,
                                key_value_length,
                                kv_block_idx % pipeline_slot_count,
                                MASK_CAUSAL,
                                1,
                                log_quant_scale_p,
                            )
                else:
                    if k_scale_slot == 0:
                        self._stage_softmax_p(
                            kv_block_idx,
                            q_tile_idx,
                            causal_offset,
                            key_value_length,
                            kv_block_idx % pipeline_slot_count,
                            MASK_NONE,
                            0,
                            log_quant_scale_p,
                        )
                    else:
                        self._stage_softmax_p(
                            kv_block_idx,
                            q_tile_idx,
                            causal_offset,
                            key_value_length,
                            kv_block_idx % pipeline_slot_count,
                            MASK_NONE,
                            1,
                            log_quant_scale_p,
                        )

                # 等当前块与待消费块拉开 SKEW_LAG 拍后，执行对应的 MM2 和在线输出累加。
                if kv_block_idx >= SKEW_LAG:
                    consume_block_idx = kv_block_idx - SKEW_LAG
                    self._consume(
                        gm_block_table,
                        gm_value_cache,
                        batch_idx,
                        consume_block_idx,
                        kv_head,
                        consume_block_idx % pipeline_slot_count,
                    )

            # 主循环结束时仍有最多 SKEW_LAG 个 P/V 未消费，在此统一排空流水。
            for drain in tuple(range(SKEW_LAG)):
                consume_block_idx = kv_block_count + drain - SKEW_LAG
                if consume_block_idx >= 0:
                    self._consume(
                        gm_block_table,
                        gm_value_cache,
                        batch_idx,
                        consume_block_idx,
                        kv_head,
                        consume_block_idx % pipeline_slot_count,
                    )

            # 使用在线 Softmax 分母归一化 acc，并按当前 AIV 实际负责的有效行数写回 GM。
            self.vector.finalize_output(
                gm_output,
                gm_value_dequant_scale,
                self.layout,
                self.vector_subblock_index,
                batch_idx,
                head_idx,
                query_token_start,
                kv_head,
                max(
                    0,
                    min(
                        rows - self.vector_subblock_index * self.m_base_size_per_aiv,
                        self.m_base_size_per_aiv,
                    ),
                ),
            )

    @jit
    def _consume(
        self,
        gm_block_table,
        gm_value_cache,
        batch_idx,
        kv_block_idx,
        kv_head,
        pipeline_slot,
    ):
        """执行 ``PV=P^T@V``，并按 ``acc=acc*corr+PV`` 更新在线累加值。"""
        self.matmul.load_v(
            gm_value_cache, gm_block_table, batch_idx, kv_head, kv_block_idx
        )
        self.matmul.compute_pv(self.p_l1, self.pv_ub)
        self.vector.update_output_acc(self.pv_ub.next_read(), pipeline_slot)

    @jit
    def _load_query_and_scale(
        self, gm_query, gm_query_dequant_scale, batch_idx, head_idx, query_token_start
    ):
        """搬入整个 S2 扫描复用的 Q 与 deq_q，每个 Q tile 仅执行一次。"""
        query_dequant_scale_tile = _query_row_tile(
            gm_query_dequant_scale,
            self.layout,
            batch_idx,
            head_idx,
            query_token_start + self.vector_subblock_index * self.m_base_size_per_aiv,
            self.m_base_size_per_aiv,
            1,
        )
        query_tile = _query_row_tile(
            gm_query,
            self.layout,
            batch_idx,
            head_idx,
            query_token_start,
            self.m_base_size,
            self.d,
        )
        mem_copy(
            self.vector.query_dequant_scale_ub.next_write(),
            query_dequant_scale_tile,
        )
        return self.matmul.load_q(query_tile)

    @jit
    def _stage_kscale(
        self,
        gm_block_table,
        gm_key_cache,
        gm_value_dequant_scale,
        batch_idx,
        kv_head,
        kv_block_idx,
        k_scale_slot,
    ):
        """将 K scale 搬入双 Buffer，并复制 4 份供 Pass A 无 bank 冲突广播。"""
        k_scale_slot = int(const_expr(k_scale_slot))
        key_dequant_scale_ub = self.vector.key_dequant_scale_ub_slots[k_scale_slot]
        key_dequant_scale_byte_view = self.vector.key_dequant_scale_byte_views[
            k_scale_slot
        ]
        mem_copy(
            tile_slice(key_dequant_scale_ub, (1, 1), (0, self.s2_base_size)),
            tile_slice(gm_value_dequant_scale[kv_head, None, None], (1, 1), (0, 0)),
        )
        for cache_block_offset in tuple(range(self.cache_blocks_per_kv_tile)):
            physical_cache_block_index = gm_block_table[
                batch_idx,
                kv_block_idx * self.cache_blocks_per_kv_tile + cache_block_offset,
            ]

            # cache 中连续的 4 行字节与 UB 单行字节窗口按一维域对应。
            for copy_index in tuple(range(KEY_DEQUANT_SCALE_COPIES)):
                mem_copy(
                    tile_slice(
                        key_dequant_scale_byte_view,
                        (1, KEY_SCALE_ROWS * self.d),
                        (copy_index, cache_block_offset),
                    ).flatten(),
                    tile_slice(
                        gm_key_cache[physical_cache_block_index, kv_head, None, None],
                        (KEY_SCALE_ROWS, self.d),
                        (self.block_size // KEY_SCALE_ROWS, 0),
                    ).flatten(),
                )

    @jit
    def _stage_k(self, gm_block_table, gm_key_cache, batch_idx, kv_head, kv_block_idx):
        """将两个 ``128x128`` PageAttention 块写入同一 ``256x128`` K-L1 槽位。"""
        self.matmul.load_k(
            gm_key_cache, gm_block_table, batch_idx, kv_head, kv_block_idx
        )

    @jit
    def _stage_qk_auto(
        self, gm_block_table, gm_key_cache, batch_idx, kv_head, kv_block_idx, q_slot
    ):
        """单一 S2 发射点：搬入 K 并执行 MM1。"""
        self._stage_k(gm_block_table, gm_key_cache, batch_idx, kv_head, kv_block_idx)
        self.matmul.compute_qk(self.s_ub, q_slot)

    @jit
    def _stage_softmax_p(
        self,
        kv_block_idx,
        q_tile_idx,
        causal_offset,
        s2_length,
        pipeline_slot,
        kv_block_mask_type,
        k_scale_slot,
        log_quant_scale_p,
    ):
        """使用提前搬入的 K scale，执行 Softmax 并生产 P-L1。"""
        # 将完整序列中的可见边界转换成当前 KV 块内部的局部行号，供 Vector 比较生成 mask。
        if const_expr(kv_block_mask_type == MASK_KV_TAIL):
            visible_limit = s2_length - kv_block_idx * self.s2_base_size - 1
        else:
            visible_limit = (
                q_tile_idx * self.m_base_size
                + causal_offset
                - kv_block_idx * self.s2_base_size
                + self.vector_subblock_index * self.m_base_size_per_aiv
            )
        self.vector.softmax(
            self.s_ub.next_read(),
            visible_limit,
            pipeline_slot,
            kv_block_mask_type,
            k_scale_slot,
            log_quant_scale_p,
        )
        self.vector.store_p(self.p_l1, self.vector_subblock_index)


# Host 侧算子启动封装
class FlashAttnFp8FullQuantLauncher:
    """保存编译配置并启动融合 Kernel。"""

    def __init__(
        self,
        m_base_size,
        s2_base_size,
        d,
        block_size,
        depths=None,
        mask_tail=0,
        layout="BNSD",
    ):
        self.m_base_size = m_base_size
        self.s2_base_size = s2_base_size
        self.d = d
        self.block_size = block_size
        self.layout = layout
        self.mask_tail = mask_tail
        self.depths = dict(DEPTHS if depths is None else depths)

    @host
    def launch(
        self,
        gm_output,
        gm_query,
        gm_key_cache,
        gm_value_cache,
        gm_block_table,
        gm_query_dequant_scale,
        gm_value_dequant_scale,
        gm_core_task_ranges,
        gm_query_sequence_boundaries,
        gm_key_value_lengths,
        gm_nz_gather_indices,
        batch: int,
        n_q_heads: int,
        head_group: int,
        block_dim: int,
        gm_log_quant_scale_p: Tensor,
    ):
        op = FlashAttnFp8FullQuantKernel(
            self.m_base_size,
            self.s2_base_size,
            self.d,
            self.block_size,
            self.depths,
            self.mask_tail,
            self.layout,
        )
        op[block_dim](
            gm_query,
            gm_key_cache,
            gm_value_cache,
            gm_block_table,
            gm_query_dequant_scale,
            gm_value_dequant_scale,
            gm_core_task_ranges,
            gm_query_sequence_boundaries,
            gm_key_value_lengths,
            gm_nz_gather_indices,
            gm_output,
            batch,
            n_q_heads,
            head_group,
            gm_log_quant_scale_p,
        )


_COMPILED_KERNELS = {}
_COMPILED_KERNELS_LOCK = threading.Lock()


def _dynamic_tensor_specs(layout):
    """构造一个布局内跨 shape 复用的动态 AOT 调用契约。"""
    batch = cannbotdsl.Dim("batch", min=1)
    query_heads = cannbotdsl.Dim("query_heads", min=1)
    key_value_heads = cannbotdsl.Dim("key_value_heads", min=1)
    physical_cache_blocks = cannbotdsl.Dim("physical_cache_blocks", min=1)
    block_table_width = cannbotdsl.Dim("block_table_width", min=1)
    cube_core_count = cannbotdsl.Dim("cube_core_count", min=1, max=64)
    query_boundary_width = cannbotdsl.Dim("query_boundary_width", min=2)
    cache_block_stride = cannbotdsl.Dim("cache_block_stride", min=1)
    tensor_spec = cannbotdsl.TensorSpec

    if layout == "TND":
        output_tokens = cannbotdsl.Dim("output_tokens", min=1)
        query_storage_tokens = cannbotdsl.Dim("query_storage_tokens", min=1)
        packed_head_stride = cannbotdsl.Dim("packed_head_stride", min=D)
        output_spec = tensor_spec(
            (query_heads, output_tokens, D),
            dtypes.float16,
            stride=(D, packed_head_stride, 1),
        )
        query_spec = tensor_spec(
            (query_heads, query_storage_tokens, D),
            dtypes.uint8,
            stride=(D, packed_head_stride, 1),
        )
        query_scale_spec = tensor_spec(
            (query_heads, query_storage_tokens, 1),
            dtypes.float32,
            stride=(query_storage_tokens, 1, 1),
        )
    else:
        query_tokens = cannbotdsl.Dim("query_tokens", min=1)
        batch_query_stride = cannbotdsl.Dim("batch_query_stride", min=D)
        head_query_stride = cannbotdsl.Dim("head_query_stride", min=D)
        batch_scale_stride = cannbotdsl.Dim("batch_scale_stride", min=1)
        head_scale_stride = cannbotdsl.Dim("head_scale_stride", min=1)
        output_spec = tensor_spec(
            (batch, query_heads, query_tokens, D),
            dtypes.float16,
            stride=(batch_query_stride, head_query_stride, D, 1),
        )
        query_spec = tensor_spec(
            (batch, query_heads, query_tokens, D),
            dtypes.uint8,
            stride=(batch_query_stride, head_query_stride, D, 1),
        )
        query_scale_spec = tensor_spec(
            (batch, query_heads, query_tokens, 1),
            dtypes.float32,
            stride=(batch_scale_stride, head_scale_stride, 1, 1),
        )

    cache_spec = tensor_spec(
        (physical_cache_blocks, key_value_heads, PAGE_BLOCK_ROWS + KEY_SCALE_ROWS, D),
        dtypes.uint8,
        stride=(cache_block_stride, (PAGE_BLOCK_ROWS + KEY_SCALE_ROWS) * D, D, 1),
    )
    block_table_spec = tensor_spec(
        (batch, block_table_width), dtypes.int64, stride=(block_table_width, 1)
    )
    value_scale_spec = tensor_spec((key_value_heads, 1, 1), dtypes.float32)
    core_task_ranges_spec = tensor_spec(
        (2, cube_core_count), dtypes.int64, stride=(cube_core_count, 1)
    )
    query_boundaries_spec = tensor_spec(
        (1, query_boundary_width), dtypes.int64, stride=(query_boundary_width, 1)
    )
    key_value_lengths_spec = tensor_spec((1, batch), dtypes.int64, stride=(batch, 1))
    gather_index_count = len(nz_gather_index(M_BASE_SIZE_PER_AIV))
    gather_indices_spec = tensor_spec((1, gather_index_count), dtypes.uint8)

    return (
        output_spec,
        query_spec,
        cache_spec,
        cache_spec,
        block_table_spec,
        query_scale_spec,
        value_scale_spec,
        core_task_ranges_spec,
        query_boundaries_spec,
        key_value_lengths_spec,
        gather_indices_spec,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        tensor_spec((1, 1), dtypes.float32),
    )


def _get_compiled_kernel(tiling):
    """按布局和 ``mask_tail`` 复用动态 shape 的编译产物。"""
    key = (tiling.layout, tiling.mask_tail)
    with _COMPILED_KERNELS_LOCK:
        compiled = _COMPILED_KERNELS.get(key)
        if compiled is not None:
            return compiled

        launcher = FlashAttnFp8FullQuantLauncher(
            M_BASE_SIZE,
            S2_BASE_SIZE,
            D,
            PAGE_BLOCK_ROWS,
            depths=DEPTHS,
            mask_tail=tiling.mask_tail,
            layout=tiling.layout,
        )
        compiled = cannbotdsl.compile(
            launcher.launch, *_dynamic_tensor_specs(tiling.layout)
        )
        _COMPILED_KERNELS[key] = compiled
        return compiled


def _prepare_inputs(
    q_fp8,
    k_cache_u8,
    v_cache_u8,
    block_table,
    deq_q,
    deq_v,
    s1,
    s2,
    n_kv_heads,
    *,
    scale,
    block_dim,
    mask_mode,
    layout,
    actual_seq,
    actual_seq_kv,
    quant_scale_p,
):
    """准备融合 Kernel 使用的 Tensor、输出和 Host Tiling 参数。"""
    tiling = validate_and_resolve(
        q_fp8,
        k_cache_u8,
        v_cache_u8,
        block_table,
        deq_q,
        deq_v,
        s1,
        s2,
        n_kv_heads,
        scale=scale,
        block_dim=block_dim,
        mask_mode=mask_mode,
        layout=layout,
        actual_seq=actual_seq,
        actual_seq_kv=actual_seq_kv,
        quant_scale_p=quant_scale_p,
        m_base_size=M_BASE_SIZE,
        s2_base_size=S2_BASE_SIZE,
        page_block_rows=PAGE_BLOCK_ROWS,
        depths=DEPTHS,
    )

    cumulative_query_lengths = tiling.cumulative_query_lengths
    query_padding_rows = tiling.query_padding_rows
    query_device = q_fp8.view(torch.uint8).npu().contiguous()
    if cumulative_query_lengths is not None:
        total_query = cumulative_query_lengths[-1]
        query_device = query_device.reshape(
            total_query,
            tiling.n_q_heads * tiling.head_dim,
        )
        if query_padding_rows:
            query_device = torch.cat(
                [
                    query_device,
                    torch.zeros(
                        query_padding_rows,
                        tiling.n_q_heads * tiling.head_dim,
                        dtype=query_device.dtype,
                        device=query_device.device,
                    ),
                ],
                dim=0,
            )
        query_device = query_device.reshape(
            total_query + query_padding_rows,
            tiling.n_q_heads,
            tiling.head_dim,
        ).permute(1, 0, 2)

    key_cache_device = k_cache_u8.npu().contiguous()
    value_cache_device = v_cache_u8.npu().contiguous()
    block_table_device = tiling.widened_block_table.to(query_device.device).contiguous()
    core_task_ranges_device = tiling.core_task_ranges.to(
        query_device.device
    ).contiguous()
    query_sequence_boundaries_device = tiling.query_sequence_boundaries.to(
        query_device.device
    ).contiguous()
    key_value_lengths_device = tiling.key_value_length_table.to(
        query_device.device
    ).contiguous()

    query_dequant_scale_device = (
        deq_q.reshape(
            tiling.n_q_heads,
            cumulative_query_lengths[-1],
            1,
        ).float()
        if cumulative_query_lengths is not None
        else deq_q.reshape(
            tiling.batch_size,
            tiling.n_q_heads,
            s1,
            1,
        ).float()
    )
    if query_padding_rows:
        query_dequant_scale_device = torch.cat(
            [
                query_dequant_scale_device,
                torch.zeros(
                    tiling.n_q_heads,
                    query_padding_rows,
                    1,
                    dtype=query_dequant_scale_device.dtype,
                    device=query_dequant_scale_device.device,
                ),
            ],
            dim=1,
        )
    query_dequant_scale_device = query_dequant_scale_device * tiling.scale
    query_dequant_scale_device = query_dequant_scale_device.npu().contiguous()
    value_dequant_scale_device = (
        deq_v.reshape(tiling.n_kv_heads, 1, 1).float().npu().contiguous()
    )

    nz_gather_indices = nz_gather_index(M_BASE_SIZE // 2)
    nz_gather_indices_device = (
        torch.tensor(
            nz_gather_indices,
            dtype=torch.uint8,
        )
        .reshape(1, len(nz_gather_indices))
        .npu()
        .contiguous()
    )
    if isinstance(tiling.quant_scale_p, torch.Tensor):
        log_quant_scale_p_device = torch.log(
            tiling.quant_scale_p.to(query_device.device).reshape(1, 1)
        ).contiguous()
    else:
        log_quant_scale_p_device = torch.tensor(
            [[math.log(tiling.quant_scale_p)]],
            dtype=torch.float32,
            device=query_device.device,
        )

    output_storage = (
        torch.empty(
            cumulative_query_lengths[-1],
            tiling.n_q_heads,
            tiling.head_dim,
            dtype=torch.float16,
            device=query_device.device,
        )
        if cumulative_query_lengths is not None
        else torch.empty(
            tiling.batch_size,
            tiling.n_q_heads,
            s1,
            tiling.head_dim,
            dtype=torch.float16,
            device=query_device.device,
        )
    )
    output_kernel_view = (
        output_storage.permute(1, 0, 2)
        if cumulative_query_lengths is not None
        else output_storage
    )
    launcher = _get_compiled_kernel(tiling)
    call_args = (
        output_kernel_view,
        query_device,
        key_cache_device,
        value_cache_device,
        block_table_device,
        query_dequant_scale_device,
        value_dequant_scale_device,
        core_task_ranges_device,
        query_sequence_boundaries_device,
        key_value_lengths_device,
        nz_gather_indices_device,
        tiling.batch_size,
        tiling.n_q_heads,
        tiling.n_q_heads // tiling.n_kv_heads,
        tiling.block_dim,
        log_quant_scale_p_device,
    )
    return launcher, call_args, output_storage


def flash_attn_fp8_fullquant(
    q_fp8,
    k_cache_u8,
    v_cache_u8,
    block_table,
    deq_q,
    deq_v,
    s1,
    s2,
    n_kv_heads,
    *,
    scale=None,
    block_dim=None,
    mask_mode="on",
    layout="BNSD",
    actual_seq=None,
    actual_seq_kv=None,
    quant_scale_p=1.0,
):
    """执行 FP8 全量化 GQA，返回 NPU 上的 FP16 张量。"""
    launcher, call_args, output = _prepare_inputs(
        q_fp8,
        k_cache_u8,
        v_cache_u8,
        block_table,
        deq_q,
        deq_v,
        s1,
        s2,
        n_kv_heads,
        scale=scale,
        block_dim=block_dim,
        mask_mode=mask_mode,
        layout=layout,
        actual_seq=actual_seq,
        actual_seq_kv=actual_seq_kv,
        quant_scale_p=quant_scale_p,
    )
    launcher(*call_args)
    return output
