# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""attn_epilogue 的 kernel 选择、切分与片上容量计算。"""

# fmt: off

from bisect import bisect_right
from math import gcd

import cannbotdsl as cbd
import torch
import torch_npu

if __package__:
    from .attn_epilogue_kernel import (
        AttnEpilogueKernel, FP32_BYTES, FP4_TILE_BYTES, FP4_TILE_N, FP4_VALUES_PER_BYTE, GROUP_INPUT_DIM,
        HEADS_PER_GROUP, MIX_VECTOR_PARTS, MM1_OUTPUT_DIM, MX_SCALE_GROUP_SIZE, OUTPUT_DIM, QUANT_CHUNK_ELEMENTS,
        QUANT_RECIPROCAL_ELEMS_PER_BYTE, ROPE_DIM, UB_ALIGNMENT, _align_up, _ceil_div,
    )
else:
    from attn_epilogue_kernel import (
        AttnEpilogueKernel, FP32_BYTES, FP4_TILE_BYTES, FP4_TILE_N, FP4_VALUES_PER_BYTE, GROUP_INPUT_DIM,
        HEADS_PER_GROUP, MIX_VECTOR_PARTS, MM1_OUTPUT_DIM, MX_SCALE_GROUP_SIZE, OUTPUT_DIM, QUANT_CHUNK_ELEMENTS,
        QUANT_RECIPROCAL_ELEMS_PER_BYTE, ROPE_DIM, UB_ALIGNMENT, _align_up, _ceil_div,
    )

BASE_K = 128
BF16_BYTES = cbd.dtypes.bfloat16.bytes_for(1)
GM_ROW_ALIGNMENT = 16

# FP4 WOA 的 Torch shape 与 GM 搬运使用的字节网格。
FP4_NZ_FORMAT = 50  # FRACTAL_NZ_C0_16
FP4_NZ_SHAPE = (MM1_OUTPUT_DIM, GROUP_INPUT_DIM // FP4_VALUES_PER_BYTE)
FP4_NZ_BYTE_SHAPE = (GROUP_INPUT_DIM // MX_SCALE_GROUP_SIZE, MM1_OUTPUT_DIM // FP4_TILE_N * FP4_TILE_BYTES)


class Tiling:
    """T=1–256 选最合适的 N 条带和 K 窗口，其余使用固定 M/N/K 分块。"""

    _MAX_TUNED_T = 256
    _DEFAULT_M, _DEFAULT_N, _DEFAULT_K = 64, 128, 512
    _DEFAULT_ROPE_ROWS = 4
    _BASE_N1, _BASE_N2 = (64, 128, 256), (32, 64, 128, 160, 256)
    _K_WINDOWS = (0, BASE_K, 256, 512, 1024, 2048, 4096, 8192)
    # MM1 条带数及对齐后的宽度。末条须有有效列。
    _MM1_STRIPE_N = {1: 1024, 2: 512, 3: 352, 4: 256, 5: 224, 6: 192, 7: 160, 8: 128, 11: 96, 16: 64, 32: 32}

    @staticmethod
    def select_kernel(shapes, dtypes, formats, block_num):
        """返回 kernel 与切分方案。"""
        t, heads, _ = shapes[0]
        ng = heads // HEADS_PER_GROUP
        if t <= 0 or block_num <= 0:
            raise ValueError("core count and T must be positive")
        if ng not in (1, 2, 8):
            raise ValueError("local groups must be 1, 2 or 8")
        nd, nz = torch_npu.Format.ND, torch_npu.Format.FRACTAL_NZ
        if dtypes[0] != torch.bfloat16 or formats[0] != nd:
            raise ValueError("activation must use torch.bfloat16 ND")
        if dtypes[2] != torch.float8_e4m3fn or formats[2] not in (nd, nz):
            raise ValueError("WOB must use torch.float8_e4m3fn with ND or FRACTAL_NZ")
        mm1_fp4 = dtypes[1] == torch.uint8
        if mm1_fp4:
            if formats[1] != FP4_NZ_FORMAT or shapes[1] != (ng, *FP4_NZ_SHAPE):
                raise ValueError("FP4 WOA requires uint8 FRACTAL_NZ_C0_16")
        elif dtypes[1] != torch.float8_e4m3fn or formats[1] not in (nd, nz):
            raise ValueError("WOA must use MXFP8 with ND or NZ, or uint8 packed MXFP4 with NZ")
        weight_nz = formats[2] == nz
        if (formats[1] != nd) != weight_nz:
            raise ValueError("WOA and WOB must both use ND or both use NZ")

        shape = dict(
            t=t, ng=ng, block_num=block_num, m_align16=_align_up(t, GM_ROW_ALIGNMENT),
            mm1_fp4=mm1_fp4, weight_nz=weight_nz,
        )
        # 一个 UB buffer 存打包权重，展开后的 FP8 权重占两倍空间。
        weight_slots = 1 + FP4_VALUES_PER_BYTE if mm1_fp4 else 0
        plan = Tiling._n_split_tiling(shape, weight_slots) if t <= Tiling._MAX_TUNED_T else None
        if plan is None:
            n, k = Tiling._DEFAULT_N, Tiling._DEFAULT_K
            plan = dict(
                mm1_bm=min(shape["m_align16"], Tiling._DEFAULT_M), mm1_bns=(n,) * ng,
                mm1_l1_k=k, mm2_bn=n, mm2_l1_k=k, mm1_tiles_per_block=0,
                rope_rows=Tiling._DEFAULT_ROPE_ROWS, rope_cols=GROUP_INPUT_DIM, rope_offset=0,
            )

        bm, mm1_bns = plan["mm1_bm"], plan.pop("mm1_bns")
        m_tiles = _ceil_div(t, bm)
        mm1_l1_k, mm2_bn, mm2_l1_k = plan["mm1_l1_k"], plan["mm2_bn"], plan["mm2_l1_k"]
        mm1_tiles_per_block = plan["mm1_tiles_per_block"]
        mm1_tile_offsets = [0]
        for bn in mm1_bns:
            mm1_tile_offsets.append(mm1_tile_offsets[-1] + m_tiles * _ceil_div(MM1_OUTPUT_DIM, bn))
        first_groups = [i for i, bn in enumerate(mm1_bns) if i == 0 or bn != mm1_bns[i - 1]]
        widths = tuple(mm1_bns[i] for i in first_groups)
        # 每种 BN 静态编译一条流水。调优解连续分配，默认解沿用按核轮转。
        if mm1_tiles_per_block:
            block_edges = [mm1_tile_offsets[i] // mm1_tiles_per_block for i in (*first_groups, ng)]
        else:
            block_edges = [0, min(block_num, mm1_tile_offsets[-1])]
        mm1_block_ranges = tuple(
            (group, start, end, _ceil_div(MM1_OUTPUT_DIM, bn))
            for group, start, end, bn in zip(first_groups, block_edges, block_edges[1:], widths)
        )

        bk_by_bn = {}
        for bn in (*widths, mm2_bn):
            bk = BASE_K
            while 2 * bn * bk > cbd.get_mem_size("l0b") and bk > MX_SCALE_GROUP_SIZE:
                bk //= 2
            bk_by_bn[bn] = bk
        max_bn = max(widths)
        c_depth = 1 if mm1_tiles_per_block == 1 else 2
        quant_y_offset = max_bn * mm1_l1_k * weight_slots // (MIX_VECTOR_PARTS * FP4_VALUES_PER_BYTE)
        quant_y_recv_rows = bm // MIX_VECTOR_PARTS
        return AttnEpilogueKernel, dict(
            **shape, **plan, m_tiles=m_tiles,
            mm1_tile_offsets=tuple(mm1_tile_offsets), mm1_block_ranges=mm1_block_ranges,
            mm1_bn_bk_map=tuple((bn, bk_by_bn[bn]) for bn in widths),
            mm1_l0c_num=tuple(min(c_depth, cbd.get_mem_size("l0c") // (bm * bn * FP32_BYTES)) for bn in widths),
            mm1_l1_slots=(2, 2),
            quant_y_offset=quant_y_offset, quant_y_recv_rows=quant_y_recv_rows,
            quant_y_rows_map=tuple(Tiling._quant_y_rows(quant_y_recv_rows, bn, quant_y_offset) for bn in widths),
            mm2_bk=bk_by_bn[mm2_bn], mm2_n_tiles=_ceil_div(OUTPUT_DIM, mm2_bn),
            mm2_l1_slots=Tiling._l1_slots(bm, mm2_bn, mm2_l1_k, window_scales=shape["weight_nz"]),
        )

    @staticmethod
    def _n_split_tiling(shape, weight_slots):
        """优先均衡 MM1 的核间工作量，再确定两级 K 窗口和 RoPE 批次。"""
        m, ng, t, blocks = shape["m_align16"], shape["ng"], shape["t"], shape["block_num"]
        mm1_fp4 = shape["mm1_fp4"]
        vector_parts = MIX_VECTOR_PARTS * blocks
        if m * BASE_K * 2 > cbd.get_mem_size("l0a") or vector_parts % ng:
            return None

        max_n = min(cbd.get_mem_size("l0c") // (m * FP32_BYTES), cbd.get_mem_size("l0b") // (2 * MX_SCALE_GROUP_SIZE))
        candidates = []
        for waves in ((1, 2, 4) if blocks % ng else (1, 2, 3, 4)):
            tiles, narrow_groups = divmod(waves * blocks, ng)
            wide_n = Tiling._MM1_STRIPE_N.get(tiles, 0)
            narrow_n = Tiling._MM1_STRIPE_N.get(tiles + bool(narrow_groups), 0)
            if 0 < wide_n <= max_n and narrow_n:
                l1_k = Tiling._k_window(m, wide_n, GROUP_INPUT_DIM, weight_slots=weight_slots)
                if l1_k:
                    score = Tiling._tile_cost(m, wide_n, GROUP_INPUT_DIM, waves, mm1_fp4)
                    candidates.append((score, wide_n, narrow_n, narrow_groups, waves, l1_k))

        _, wide_n, narrow_n, narrow_groups, waves, mm1_l1_k = min(
            candidates, key=lambda item: item[0], default=(0, 0, 0, 0, 0, 0),
        )
        mm2_bn = _align_up(_ceil_div(OUTPUT_DIM, blocks), MX_SCALE_GROUP_SIZE)
        if not waves or mm2_bn > max_n:
            base_n1, base_n2 = Tiling._base_widths(shape)
            if mm2_bn > max_n:
                mm2_bn = base_n2
            if not waves:
                if base_n1 > max_n:
                    return None
                wide_n = base_n1
                mm1_l1_k = Tiling._k_window(m, wide_n, GROUP_INPUT_DIM, weight_slots=weight_slots)
        mm2_l1_k = Tiling._k_window(m, mm2_bn, ng * MM1_OUTPUT_DIM, mm2_nz=shape["weight_nz"])
        if not mm1_l1_k or not mm2_l1_k:
            return None

        rows_per_lane = _ceil_div(t, vector_parts // ng)
        activation_bytes = 12928  # 单行 BF16/FP8 激活、scale 和倒数
        rope_table_bytes = 2 * ROPE_DIM * FP32_BYTES
        rope_cols = GROUP_INPUT_DIM
        if not mm1_fp4 and waves and rows_per_lane >= cbd.get_mem_size("ub") // (activation_bytes + rope_table_bytes):
            rope_cols //= 2
            activation_bytes //= 2
        per_row = activation_bytes + rope_table_bytes
        rope_rows = _ceil_div(rows_per_lane, _ceil_div(rows_per_lane, cbd.get_mem_size("ub") // per_row))
        packed_bytes = wide_n * mm1_l1_k // (MIX_VECTOR_PARTS * FP4_VALUES_PER_BYTE)
        rope_offset = packed_bytes if mm1_fp4 and packed_bytes + rope_rows * per_row <= cbd.get_mem_size("ub") else 0
        return dict(
            mm1_bm=m, mm1_bns=(narrow_n,) * narrow_groups + (wide_n,) * (ng - narrow_groups),
            mm1_l1_k=mm1_l1_k, mm1_tiles_per_block=waves, mm2_bn=mm2_bn, mm2_l1_k=mm2_l1_k,
            rope_rows=rope_rows, rope_cols=rope_cols, rope_offset=rope_offset,
        )

    @staticmethod
    def _l1_slots(bm, bn, l1_k, *, window_scales):
        if window_scales and 2 * (bm + bn) * (l1_k + l1_k // MX_SCALE_GROUP_SIZE) > cbd.get_mem_size("l1"):
            return 1, 2
        return 2, 2

    @staticmethod
    def _quant_y_rows(recv_rows, bn, reserve):
        """取满足 UB 预算且量化次数最少的一批行数及接收槽的实际行数。"""
        align = QUANT_CHUNK_ELEMENTS // gcd(bn, QUANT_CHUNK_ELEMENTS)
        choices = []
        for rows in range(align, _align_up(recv_rows, align) + 1, align):
            count = rows * bn
            ub_rows = _align_up(recv_rows, rows)
            size = ub_rows * bn * FP32_BYTES + count * (BF16_BYTES + 1)
            size += _align_up(count // MX_SCALE_GROUP_SIZE + QUANT_CHUNK_ELEMENTS, UB_ALIGNMENT)
            size += rows * UB_ALIGNMENT + count // QUANT_RECIPROCAL_ELEMS_PER_BYTE
            if reserve + size <= cbd.get_mem_size("ub"):
                choices.append((_ceil_div(recv_rows, rows), ub_rows, -rows))
        _, ub_rows, negative_rows = min(choices)
        return -negative_rows, ub_rows

    @staticmethod
    def _base_widths(shape):
        """条带无法均衡时，按两级投影的成本选择可轮转的基础 N。"""
        t, ng, blocks = shape["t"], shape["ng"], shape["block_num"]
        mm1_fp4 = shape["mm1_fp4"]
        k2 = ng * MM1_OUTPUT_DIM
        max_k = max(GROUP_INPUT_DIM, k2)
        max_m = min(shape["m_align16"], cbd.get_mem_size("l0a") // (2 * BASE_K))
        candidates = []
        for m in range(GM_ROW_ALIGNMENT, max_m + 1, GM_ROW_ALIGNMENT):
            m_tiles = _ceil_div(t, m)
            max_n = min(
                cbd.get_mem_size("l0b") // (2 * BASE_K), cbd.get_mem_size("l0c") // (m * FP32_BYTES),
                cbd.get_mem_size("l1") // (4 * BASE_K + max_k // MX_SCALE_GROUP_SIZE) - m,
            )
            if max_n < Tiling._BASE_N1[0]:
                break
            stages = []
            for groups, width, k, widths, fp4 in (
                (ng, MM1_OUTPUT_DIM, GROUP_INPUT_DIM, Tiling._BASE_N1, mm1_fp4),
                (1, OUTPUT_DIM, k2, Tiling._BASE_N2, False),
            ):
                scores = []
                for n in widths[:bisect_right(widths, max_n)]:
                    waves = _ceil_div(groups * m_tiles * (width // n), blocks)
                    scores.append((Tiling._tile_cost(m, n, k, waves, fp4), 0 if mm1_fp4 else waves, n))
                stages.append(min(scores))
            (cost1, _, n1), (cost2, _, n2) = stages
            candidates.append((cost1 + cost2, -m, n1, n2))
        return min(candidates)[2:]

    @staticmethod
    def _tile_cost(m, n, k, waves, fp4):
        weight_cost = 192 if fp4 else 256
        return waves * k * (2 * m * n + 256 * m + weight_cost * n)

    @staticmethod
    def _k_window(m, n, k, *, weight_slots=None, mm2_nz=False):
        """按 L1 容量选 K，MM1 同时给 QuantY 和可选 FP4 权重留出 UB。"""
        if mm2_nz:
            # FP8 NZ 的 scale 随窗口搬运，A 单槽、B 双槽。
            limit = cbd.get_mem_size("l1") * MX_SCALE_GROUP_SIZE // ((m + 2 * n) * (MX_SCALE_GROUP_SIZE + 1))
        else:
            limit = (cbd.get_mem_size("l1") // (m + n) - k // MX_SCALE_GROUP_SIZE) // 2
        if weight_slots is not None:
            rows = QUANT_CHUNK_ELEMENTS // gcd(n, QUANT_CHUNK_ELEMENTS)
            count = rows * n
            quant_bytes = (m // MIX_VECTOR_PARTS) * n * FP32_BYTES + rows * UB_ALIGNMENT
            quant_bytes += count * (BF16_BYTES + 1) + count // QUANT_RECIPROCAL_ELEMS_PER_BYTE
            quant_bytes += _align_up(count // MX_SCALE_GROUP_SIZE + QUANT_CHUNK_ELEMENTS, UB_ALIGNMENT)
            remaining = cbd.get_mem_size("ub") - quant_bytes
            if remaining < 0:
                return 0
            if weight_slots:
                weight_stride = n * weight_slots // (MIX_VECTOR_PARTS * FP4_VALUES_PER_BYTE)
                limit = min(limit, remaining // weight_stride)
        return Tiling._K_WINDOWS[bisect_right(Tiling._K_WINDOWS, max(0, min(k, limit))) - 1]
