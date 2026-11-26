# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""attn_epilogue 的设备流水与共享规格。

计算顺序为 inverse RoPE → BF16 → QuantA → MM1(FP32) → BF16 → QuantY → MM2 → BF16。
"""

# fmt: off

import cannbotdsl as cbd
from cannbotdsl.ops import reg
from cannbotdsl.ops.reg.cast import RoundingMode
from cannbotdsl.ops.sync import cube_sync_notify, cube_sync_wait, vec_sync_notify, vec_sync_wait


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _align_up(value, alignment):
    return _ceil_div(value, alignment) * alignment


# 算子规格
HEADS_PER_GROUP = 8
HEAD_DIM = 512
ROPE_DIM = 64
GROUP_INPUT_DIM = HEADS_PER_GROUP * HEAD_DIM
MM1_OUTPUT_DIM = 1024
OUTPUT_DIM = 5120

# 数据编码与权重布局
MX_SCALE_GROUP_SIZE = 32
MX_SCALE_PAIR_SIZE = 2 * MX_SCALE_GROUP_SIZE
FP32_BYTES = cbd.dtypes.float32.bytes_for(1)
FP4_VALUES_PER_BYTE = cbd.dtypes.uint8.bits // cbd.dtypes.fp4x2_e2m1.bits
FP4_TILE_N = 16
FP4_TILE_BYTES = FP4_TILE_N * MX_SCALE_GROUP_SIZE // FP4_VALUES_PER_BYTE
FP4_ACC_SCALE = 64.0
FP8_CTRL_BIT = 1 << 48
WEIGHT_L2_DEFAULT, WEIGHT_L2_BYPASS = 0, 4

# 片上容量与对齐
# UB 根视图需要完整物理容量，get_mem_size("ub") 用于 tiling 的容量预算。
UB_PHYSICAL_BYTES = 256 * 1024
UB_ALIGNMENT = 32
MIX_VECTOR_PARTS = 2  # 一个 MIX block 中的 AIV 分片数

# 激活发布的跨 AIV 标志与 AIV→AIC 标志
QUANT_A_VEC_FLAG, QUANT_A_CUBE_FLAG = 11, 3
QUANT_Y_VEC_FLAG, QUANT_Y_CUBE_FLAG = 13, 7
STORAGE_REUSE_EVENT = 0
MM1_FIXPIPE_EVENT = 1

# 向量量化的处理粒度和临时空间
QUANT_CHUNK_ELEMENTS = 256
QUANT_RECIPROCAL_ELEMS_PER_BYTE = 8


class AttnEpilogueKernel:
    """按行块和 N 条带分配投影任务，阶段之间复用同一组片上通道。"""

    def __init__(self, tiling):
        self.t, self.ng, self.block_num = tiling["t"], tiling["ng"], tiling["block_num"]
        self.m_align16, self.m_tiles = tiling["m_align16"], tiling["m_tiles"]
        self.rope_rows, self.rope_cols = tiling["rope_rows"], tiling["rope_cols"]
        self.rope_offset = tiling["rope_offset"]
        self.mm1_fp4, self.weight_nz = tiling["mm1_fp4"], tiling["weight_nz"]
        self.mm_b_l2_cache_ctl = WEIGHT_L2_DEFAULT if self.mm1_fp4 else WEIGHT_L2_BYPASS
        self.mm1_bm, self.mm1_l1_k = tiling["mm1_bm"], tiling["mm1_l1_k"]
        self.mm1_bn_bk_map, self.mm1_tile_offsets = tiling["mm1_bn_bk_map"], tiling["mm1_tile_offsets"]
        self.mm1_block_ranges, self.mm1_tiles_per_block = tiling["mm1_block_ranges"], tiling["mm1_tiles_per_block"]
        self.mm1_l1_slots, self.mm1_l0c_num = tiling["mm1_l1_slots"], tiling["mm1_l0c_num"]
        self.quant_y_offset, self.quant_y_rows_map = tiling["quant_y_offset"], tiling["quant_y_rows_map"]
        self.quant_y_recv_rows = tiling["quant_y_recv_rows"]
        self.mm2_bn, self.mm2_bk, self.mm2_l1_k = tiling["mm2_bn"], tiling["mm2_bk"], tiling["mm2_l1_k"]
        self.mm2_l1_slots = tiling["mm2_l1_slots"]
        self.mm2_n_tiles = tiling["mm2_n_tiles"]

    @cbd.host
    def __call__(
        self, x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos, gm_aq, gm_asc, gm_yq, gm_ysc, gm_out,
    ):
        self.run[self.block_num](
            x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos, gm_aq, gm_asc, gm_yq, gm_ysc, gm_out,
        )

    @cbd.kernel
    def run(
        self, x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos, gm_aq, gm_asc, gm_yq, gm_ysc, gm_out,
    ):
        saved_ctrl = cbd.get_ctrl()
        cbd.set_ctrl(saved_ctrl & ~FP8_CTRL_BIT)

        self.l1_tensor = cbd.L1.view(cbd.get_mem_size("l1"))
        self.l0a_tensor = cbd.L0A.view(cbd.get_mem_size("l0a"))
        self.l0b_tensor = cbd.L0B.view(cbd.get_mem_size("l0b"))
        self.l0c_tensor = cbd.L0C.view(cbd.get_mem_size("l0c"))
        self.ub_tensor = cbd.UB.view(UB_PHYSICAL_BYTES)

        self._mm_make_prefetch_buffers(
            self.mm1_bn_bk_map, MM1_OUTPUT_DIM, GROUP_INPUT_DIM, self.mm1_l1_k,
            self.mm1_fp4, self.mm1_l1_slots,
        )
        self._mm1_prefetch_weights(woa, descale_woa)
        self._rope_run_with_quant_a(x, rope_sin, rope_cos, gm_aq, gm_asc, woa)
        self._mm_make_compute_buffers(self.mm1_l0c_num)
        if cbd.const_expr(self.mm1_fp4):
            # QuantA 写回完成后，FP4 搬入和解包才能覆盖 UB。
            vec_sync_notify(cbd.PIPE.MTE3, cbd.PIPE.MTE2, event_id=STORAGE_REUSE_EVENT)
            vec_sync_notify(cbd.PIPE.MTE3, cbd.PIPE.V, event_id=STORAGE_REUSE_EVENT)
            vec_sync_wait(cbd.PIPE.MTE3, cbd.PIPE.MTE2, event_id=STORAGE_REUSE_EVENT)
            vec_sync_wait(cbd.PIPE.MTE3, cbd.PIPE.V, event_id=STORAGE_REUSE_EVENT)
        if cbd.const_expr(self.rope_cols == GROUP_INPUT_DIM):
            self._quant_publish_activation(QUANT_A_VEC_FLAG, QUANT_A_CUBE_FLAG)
        self._mm1_run_with_quant_y(gm_aq, gm_asc, woa, descale_woa, gm_yq, gm_ysc)

        # MM1 取完 L1 后，MM2 才能向同一地址预取权重。
        cube_sync_notify(cbd.PIPE.MTE1, cbd.PIPE.MTE2, event_id=STORAGE_REUSE_EVENT)
        if cbd.const_expr(self.mm1_fp4):
            # FP4 MM1 的末次 Fixpipe 完成后再发 NZ WOB 预取，保持原版交接顺序。
            cube_sync_notify(cbd.PIPE.FIXPIPE, cbd.PIPE.MTE2, event_id=MM1_FIXPIPE_EVENT)
        self._mm_make_prefetch_buffers(
            ((self.mm2_bn, self.mm2_bk),), OUTPUT_DIM, self.ng * MM1_OUTPUT_DIM, self.mm2_l1_k,
            False, self.mm2_l1_slots, is_mm2=True,
        )
        cube_sync_wait(cbd.PIPE.MTE1, cbd.PIPE.MTE2, event_id=STORAGE_REUSE_EVENT)
        if cbd.const_expr(self.mm1_fp4):
            cube_sync_wait(cbd.PIPE.FIXPIPE, cbd.PIPE.MTE2, event_id=MM1_FIXPIPE_EVENT)
        self._mm2_prefetch_weights(wob, descale_wob)
        self._mm_make_compute_buffers((1,))
        self._quant_publish_activation(QUANT_Y_VEC_FLAG, QUANT_Y_CUBE_FLAG)
        self._mm2_run(gm_yq, gm_ysc, wob, descale_wob, gm_out)

        cbd.set_ctrl(saved_ctrl)

    @cbd.jit
    def _mm1_prefetch_weights(self, woa, descale_woa):
        """定位本核首个 tile，预取其 scale 和可直搬的 FP8 权重。"""
        block_idx = cbd.get_block_idx()
        self.mm1_first_tile = block_idx * max(self.mm1_tiles_per_block, 1)
        ng_idx, row, col = 0, 0, 0
        for bn_idx in cbd.range_constexpr(len(self.mm1_block_ranges)):
            _, block_begin, block_end, _ = self.mm1_block_ranges[bn_idx]
            if block_idx >= block_begin and block_idx < block_end:
                ng_idx, row, col = self._mm1_get_tile_position(bn_idx, self.mm1_first_tile)
                if cbd.const_expr(not self.mm1_fp4):
                    self._mm_load_b_to_l1(woa, bn_idx, ng_idx, col, 0)
                self._mm_load_b_scale_to_l1(descale_woa, bn_idx, ng_idx, col, 0)
        self.mm1_first_position = ng_idx, row, col

    @cbd.jit
    def _mm1_get_tile_position(self, bn_idx, tile_idx):
        """从全局 tile 编号定位组及块在 M、N 上的起点。"""
        first_ng, _, _, n_tiles = self.mm1_block_ranges[bn_idx]
        group_tile_offset = tile_idx - self.mm1_tile_offsets[first_ng]
        ng_idx = first_ng + group_tile_offset // (self.m_tiles * n_tiles)
        row = group_tile_offset // n_tiles % self.m_tiles * self.mm1_bm
        return ng_idx, row, group_tile_offset % n_tiles * self.mm1_bn_bk_map[bn_idx][0]

    @cbd.jit
    def _rope_run_with_quant_a(self, x, rope_sin, rope_cos, gm_aq, gm_asc, woa):
        """按 AIV 分批执行 RoPE，原位舍入到 BF16 后量化。"""
        if cbd.const_expr(self.rope_offset > 0):
            self._mm1_make_fp4_packed_buffer()
        (
            self.rope_ub_o, self.quant_a_ub_fp8, self.quant_a_ub_scale,
            self.rope_ub_cos, self.rope_ub_sin, self.quant_a_ub_reciprocal,
        ) = self._make_ub_buffers(self.rope_offset, (
            ((self.rope_rows, self.rope_cols), cbd.dtypes.bfloat16, 1),
            ((self.rope_rows, self.rope_cols), cbd.dtypes.uint8, 1),
            ((self.rope_rows, self.rope_cols // MX_SCALE_GROUP_SIZE), cbd.dtypes.uint8, 1),
            ((self.rope_rows, ROPE_DIM), cbd.dtypes.float32, 1), ((self.rope_rows, ROPE_DIM), cbd.dtypes.float32, 1),
            ((1, self.rope_rows * self.rope_cols // (2 * QUANT_RECIPROCAL_ELEMS_PER_BYTE)), cbd.dtypes.uint16, 0),
        ))
        x = x.view(self.t, self.ng * GROUP_INPUT_DIM)
        aiv_idx = cbd.get_block_idx() * MIX_VECTOR_PARTS + cbd.get_subblock_id()
        aiv_num = self.block_num * MIX_VECTOR_PARTS
        if cbd.const_expr(aiv_num % self.ng):
            # 核数不能均分到各组时，在所有 AIV 上轮转 (ng_idx, row) 块。
            row_tiles = cbd.ceil_div(self.t, self.rope_rows)
            for rope_row_tile_idx in cbd.range(aiv_idx, self.ng * row_tiles, aiv_num):
                ng_idx, row = rope_row_tile_idx // row_tiles, rope_row_tile_idx % row_tiles * self.rope_rows
                valid_rows = min(self.rope_rows, self.t - row)
                self._rope_load_rows(x, rope_sin, rope_cos, row, valid_rows, ng_idx * GROUP_INPUT_DIM)
                self._rope_quant_a_compute_store_tile(gm_aq, gm_asc, ng_idx, row, valid_rows, 0)
            return
        ng_idx = aiv_idx % self.ng
        row_step = aiv_num // self.ng * self.rope_rows
        first_row = aiv_idx // self.ng * self.rope_rows
        for rope_k_tile_idx in cbd.range(GROUP_INPUT_DIM // self.rope_cols):
            col = ng_idx * GROUP_INPUT_DIM + rope_k_tile_idx * self.rope_cols
            row_begin = first_row
            if cbd.const_expr(self.rope_offset > 0):
                valid_rows = min(self.rope_rows, self.t - first_row)
                if first_row < self.t:
                    self._rope_load_rows(x, rope_sin, rope_cos, first_row, valid_rows, col)
                mm1_ng_idx, _, mm1_col = self.mm1_first_position
                block_idx = cbd.get_block_idx()
                for bn_idx in cbd.range_constexpr(len(self.mm1_block_ranges)):
                    _, block_begin, block_end, _ = self.mm1_block_ranges[bn_idx]
                    if block_idx >= block_begin and block_idx < block_end:
                        self._mm1_load_fp4_b_to_ub(woa, mm1_ng_idx, mm1_col, self.mm1_bn_bk_map[bn_idx][0], 0)
                if first_row < self.t:
                    self._rope_quant_a_compute_store_tile(gm_aq, gm_asc, ng_idx, first_row, valid_rows, rope_k_tile_idx)
                row_begin = first_row + row_step
            for row in cbd.range(row_begin, self.t, row_step):
                valid_rows = min(self.rope_rows, self.t - row)
                self._rope_load_rows(x, rope_sin, rope_cos, row, valid_rows, col)
                self._rope_quant_a_compute_store_tile(gm_aq, gm_asc, ng_idx, row, valid_rows, rope_k_tile_idx)
            if cbd.const_expr(self.rope_cols < GROUP_INPUT_DIM):
                self._quant_publish_activation(
                    QUANT_A_VEC_FLAG + rope_k_tile_idx, QUANT_A_CUBE_FLAG + rope_k_tile_idx, wait_on_cube=False,
                )

    @cbd.jit
    def _rope_load_rows(self, x, rope_sin, rope_cos, row, valid_rows, col):
        row_end = row + valid_rows
        cbd.mem_copy(self.rope_ub_o.next_write()[:valid_rows, None], x[row:row_end, col:col + self.rope_cols])
        cbd.mem_copy(self.rope_ub_cos.next_write()[:valid_rows, None], rope_cos[row:row_end, None])
        cbd.mem_copy(self.rope_ub_sin.next_write()[:valid_rows, None], rope_sin[row:row_end, None])

    @cbd.jit
    def _rope_quant_a_compute_store_tile(self, gm_aq, gm_asc, ng_idx, row, valid_rows, rope_k_tile_idx):
        ub_o = self.rope_ub_o.next_read()
        self._rope_apply_inverse(
            ub_o, self.rope_ub_cos.next_read(), self.rope_ub_sin.next_read(),
            self.rope_rows, self.rope_cols // HEAD_DIM,
        )
        self._mx_quantize_bf16_to_fp8(
            ub_o, self.quant_a_ub_fp8.next_write(), self.quant_a_ub_scale.next_write(), self.quant_a_ub_reciprocal,
            self.rope_rows * self.rope_cols,
        )
        gm_row = ng_idx * self.m_align16 + row
        col = rope_k_tile_idx * self.rope_cols
        col_end = col + self.rope_cols
        cbd.mem_copy(gm_aq[gm_row:gm_row + valid_rows, col:col_end], self.quant_a_ub_fp8.next_read()[:valid_rows, None])
        cbd.mem_copy(
            gm_asc[gm_row:gm_row + valid_rows, col // MX_SCALE_GROUP_SIZE:col_end // MX_SCALE_GROUP_SIZE],
            self.quant_a_ub_scale.next_read()[:valid_rows, None],
        )

    @cbd.jit
    def _rope_apply_inverse(self, src, cos, sin, rows, heads):
        """在 BF16 输入槽中更新每个 head 的 RoPE 尾部。"""
        with cbd.vf(mode="simd"):
            full = reg.full_mask()
            lane = reg.varange(0, cbd.dtypes.uint32)
            one = reg.vdups(1, cbd.dtypes.uint32, mask=full)
            even_bits = reg.vbitwise_and(lane, one, mask=full)
            even_mask = reg.veqs(even_bits, 0, mask=full)
            partner = reg.vbitwise_xor(lane, one, mask=full)
            even_lane = reg.vsub(lane, even_bits, mask=full)
            for row in cbd.range(0, rows, 1):
                cr = reg.vgather_reg(reg.vload(cos, row * ROPE_DIM), even_lane)
                sr = reg.vgather_reg(reg.vload(sin, row * ROPE_DIM), even_lane)
                for head in cbd.range(0, heads, 1):
                    base = (row * heads + head + 1) * HEAD_DIM - ROPE_DIM
                    raw = reg.vload_unpack(src, base, unpack_mode=reg.UnpackMode.B16_TO_B32)
                    x = reg.vcast(raw, cbd.dtypes.float32, mask=full)
                    xc = reg.vmul(x, cr, mask=full)
                    xs = reg.vmul(reg.vgather_reg(x, partner), sr, mask=full)
                    even = reg.vadd(xc, xs, mask=full)
                    odd = reg.vsub(xc, xs, mask=full)
                    rotated = reg.vselect(even, odd, cond_mask=even_mask)
                    rounded = reg.vcast(rotated, cbd.dtypes.bfloat16, mask=full, rounding=RoundingMode.RN)
                    reg.vstore_pack(src, base, rounded, full, pack_mode=reg.PackMode.B32_TO_B16)
            reg.vmem_bar(mode="vst_vld")

    @cbd.jit
    def _mm1_run_with_quant_y(self, gm_aq, gm_asc, woa, descale_woa, gm_yq, gm_ysc):
        """计算本核的 MM1 tile，并把 L0C 分片交给 QuantY。"""
        block_idx = cbd.get_block_idx()
        gm_aq = gm_aq.view(dtype=cbd.dtypes.float8_e4m3fn)
        gm_asc = gm_asc.view(dtype=cbd.dtypes.float8_e8m0)
        for bn_idx in cbd.range_constexpr(len(self.mm1_block_ranges)):
            _, block_begin, block_end, _ = self.mm1_block_ranges[bn_idx]
            if block_idx >= block_begin and block_idx < block_end:
                if cbd.const_expr(self.mm1_tiles_per_block):
                    tile_end, tile_step = self.mm1_first_tile + self.mm1_tiles_per_block, 1
                else:
                    tile_end, tile_step = self.mm1_tile_offsets[-1], self.block_num
                bn, _ = self.mm1_bn_bk_map[bn_idx]
                quant_y_rows, quant_y_recv_buffer_rows = self.quant_y_rows_map[bn_idx]
                quant_y_elems = quant_y_rows * bn
                (
                    self.quant_y_ub_fp32, self.quant_y_ub_bf16, self.quant_y_ub_fp8,
                    self.quant_y_ub_scale, self.quant_y_ub_scale_padded, self.quant_y_ub_reciprocal,
                ) = self._make_ub_buffers(self.quant_y_offset, (
                    ((quant_y_recv_buffer_rows, bn), cbd.dtypes.float32, 1),
                    ((1, quant_y_elems), cbd.dtypes.bfloat16, 0),
                    ((1, quant_y_elems), cbd.dtypes.uint8, 0),
                    ((1, quant_y_elems // MX_SCALE_GROUP_SIZE + QUANT_CHUNK_ELEMENTS), cbd.dtypes.uint8, 0),
                    ((quant_y_rows, UB_ALIGNMENT), cbd.dtypes.uint8, 0),
                    ((1, quant_y_elems // (2 * QUANT_RECIPROCAL_ELEMS_PER_BYTE)), cbd.dtypes.uint16, 0),
                ))
                for tile_idx in cbd.range(self.mm1_first_tile, tile_end, tile_step):
                    ng_idx, row, col = self.mm1_first_position
                    is_first_tile_on_core = True
                    if cbd.const_expr(
                        self.mm1_tiles_per_block > 1
                        or (self.mm1_tiles_per_block == 0 and self.mm1_tile_offsets[-1] > self.block_num)
                    ):
                        is_first_tile_on_core = tile_idx == self.mm1_first_tile
                        if not is_first_tile_on_core:
                            ng_idx, row, col = self._mm1_get_tile_position(bn_idx, tile_idx)
                    gm_row = ng_idx * self.m_align16 + row
                    gm_row_end = gm_row + min(self.mm1_bm, self.t - row)
                    l0c = self.mm_l0c[bn_idx].next_write()
                    self._mm_run_tile(
                        gm_aq[gm_row:gm_row_end, None], woa, gm_asc[gm_row:gm_row_end, None], descale_woa,
                        l0c, bn_idx, ng_idx, col, is_first_tile_on_core,
                    )
                    self._quant_y_run_tile(
                        l0c, gm_yq, gm_ysc, quant_y_rows, bn, row,
                        ng_idx * MM1_OUTPUT_DIM + col, min(bn, MM1_OUTPUT_DIM - col),
                    )

    @cbd.jit
    def _quant_y_run_tile(self, l0c, gm_yq, gm_ysc, quant_y_rows, bn, row, col, valid_cols):
        ub_y = self.quant_y_ub_fp32.next_write()
        if cbd.const_expr(ub_y.shape[0] != self.quant_y_recv_rows):
            ub_y = ub_y[:self.quant_y_recv_rows, None]
        cbd.mem_copy(ub_y, l0c.partition(dim=0), options=cbd.copy_options(unit_flag=3))
        ub_y = self.quant_y_ub_fp32.next_read()
        row += cbd.get_subblock_id() * self.quant_y_recv_rows
        valid_rows = max(0, min(self.quant_y_recv_rows, self.t - row))
        # 空分片也要消费 Fixpipe buffer。量化临时区按行块复用。
        if cbd.const_expr(quant_y_rows >= self.quant_y_recv_rows):
            self._quant_y_compute_store_rows(ub_y, gm_yq, gm_ysc, quant_y_rows, bn, 0, row, valid_rows, col, valid_cols)
        else:
            for row_offset in cbd.range(0, max(valid_rows, 1), quant_y_rows):
                self._quant_y_compute_store_rows(
                    ub_y, gm_yq, gm_ysc, quant_y_rows, bn, row_offset * bn, row + row_offset,
                    max(0, min(quant_y_rows, valid_rows - row_offset)), col, valid_cols,
                )

    @cbd.jit
    def _quant_y_compute_store_rows(
        self, ub_y, gm_yq, gm_ysc, quant_y_rows, bn, y_offset, row, valid_rows, col, valid_cols,
    ):
        """舍入并量化一个行块，把有效列及对应 scale 一起写回。"""
        count = cbd.ceil_div(max(valid_rows, 1) * bn, QUANT_CHUNK_ELEMENTS) * QUANT_CHUNK_ELEMENTS
        with cbd.vf(mode="simd"):
            mask = reg.full_mask()
            for i in cbd.range(0, count, 64):
                value = reg.vload(ub_y, y_offset + i)
                if cbd.const_expr(self.mm1_fp4):
                    value = reg.vmuls(value, FP4_ACC_SCALE, mask=mask)
                rounded = reg.vcast(value, cbd.dtypes.bfloat16, mask=mask, rounding=RoundingMode.RN)
                reg.vstore_pack(self.quant_y_ub_bf16, i, rounded, mask, pack_mode=reg.PackMode.B32_TO_B16)
        self._mx_quantize_bf16_to_fp8(
            self.quant_y_ub_bf16, self.quant_y_ub_fp8, self.quant_y_ub_scale, self.quant_y_ub_reciprocal, count,
        )
        scale_cols = bn // MX_SCALE_GROUP_SIZE
        with cbd.vf(mode="simd"):
            mask, _ = reg.update_mask(scale_cols, elem_bits=8)
            for ub_row in cbd.range(valid_rows):
                scale_offset = ub_row * scale_cols
                reg.vload_unalign_init(self.quant_y_ub_scale, scale_offset)
                scale = reg.vload_unalign(self.quant_y_ub_scale, scale_offset)
                reg.vstore(self.quant_y_ub_scale_padded, ub_row * UB_ALIGNMENT, scale, mask)
        if valid_rows > 0:
            row_end, col_end = row + valid_rows, col + valid_cols
            cbd.mem_copy(
                gm_yq[row:row_end, col:col_end], self.quant_y_ub_fp8.view(quant_y_rows, bn)[:valid_rows, :valid_cols],
            )
            cbd.mem_copy(
                gm_ysc[row:row_end, col // MX_SCALE_GROUP_SIZE:col_end // MX_SCALE_GROUP_SIZE],
                self.quant_y_ub_scale_padded[:valid_rows, :valid_cols // MX_SCALE_GROUP_SIZE],
            )

    @cbd.jit
    def _mm2_prefetch_weights(self, wob, descale_wob):
        """在 Y 发布前预取本核首块的 FP8 权重和 scale。"""
        self.mm2_first_tile = cbd.get_block_idx()
        if cbd.const_expr(self.m_tiles == 1):
            self.mm2_first_col = self.mm2_first_tile * self.mm2_bn
        else:
            self.mm2_first_col = self.mm2_first_tile % self.mm2_n_tiles * self.mm2_bn
        if self.mm2_first_tile < self.m_tiles * self.mm2_n_tiles:
            self._mm_load_b_to_l1(wob, 0, 0, self.mm2_first_col, 0)
            self._mm_load_b_scale_to_l1(descale_wob, 0, 0, self.mm2_first_col, 0)

    @cbd.jit
    def _mm2_run(self, gm_yq, gm_ysc, wob, descale_wob, gm_out):
        gm_yq = gm_yq.view(dtype=cbd.dtypes.float8_e4m3fn)
        gm_ysc = gm_ysc.view(dtype=cbd.dtypes.float8_e8m0)
        mm2_tiles = self.m_tiles * self.mm2_n_tiles
        for tile_idx in cbd.range(self.mm2_first_tile, mm2_tiles, self.block_num):
            if cbd.const_expr(self.m_tiles == 1):
                row, row_end = 0, self.t
            else:
                row = tile_idx // self.mm2_n_tiles * self.mm1_bm
                row_end = min(row + self.mm1_bm, self.t)
            col, is_first_tile_on_core = self.mm2_first_col, True
            if cbd.const_expr(mm2_tiles > self.block_num):
                is_first_tile_on_core = tile_idx == self.mm2_first_tile
                if not is_first_tile_on_core:
                    col = tile_idx % self.mm2_n_tiles * self.mm2_bn
            col_end = min(col + self.mm2_bn, OUTPUT_DIM)
            out_tile = gm_out[row:row_end, col:col_end]
            l0c = cbd.from_tensor(self.mm_l0c[0].next_write(), layout=cbd.make_layout(out_tile.shape))
            self._mm_run_tile(
                gm_yq[row:row_end, None], wob, gm_ysc[row:row_end, None], descale_wob,
                l0c, 0, 0, col, is_first_tile_on_core,
            )
            cbd.mem_copy(out_tile, l0c, options=cbd.copy_options(unit_flag=3))

    @cbd.jit
    def _mm_run_tile(self, gm_a, gm_b, gm_a_scale, gm_b_scale, l0c, bn_idx, ng_idx, col, is_first_tile_on_core):
        """准备首块和 scale，再按 L1 K 块推进搬运与 MMAD。"""
        l1_b_scale = None
        if cbd.const_expr(not self.mm_scale_per_l1_k):
            if not is_first_tile_on_core:
                self._mm_load_b_scale_to_l1(gm_b_scale, bn_idx, ng_idx, col, 0)
            l1_b_scale = self.mm_l1_b_scales[bn_idx].next_read()

        if cbd.const_expr(not self.is_mm2 and self.rope_cols < self.mm_k):
            l1_k_tiles_per_rope_tile = self.rope_cols // self.mm_l1_k
            for rope_k_tile_idx in cbd.range(self.mm_k // self.rope_cols):
                if is_first_tile_on_core:
                    cbd.cube_sync_block_wait(cbd.PIPE.MTE2, QUANT_A_CUBE_FLAG + rope_k_tile_idx)
                self._mm_load_a_scale_to_l1(gm_a_scale, rope_k_tile_idx * self.rope_cols)
                rope_a_scale = self.mm_l1_a_scales.next_read().permute(1, 0)
                self._mm_run_l1_k_tile(
                    gm_a, gm_b, gm_a_scale, gm_b_scale, l0c, bn_idx, ng_idx, col, rope_a_scale, l1_b_scale,
                    is_first_tile_on_core, rope_k_tile_idx, l1_k_tiles_per_rope_tile,
                )
            return

        l1_a_scale = None
        if cbd.const_expr(not self.mm_scale_per_l1_k):
            self._mm_load_a_scale_to_l1(gm_a_scale, 0)
            l1_a_scale = self.mm_l1_a_scales.next_read().permute(1, 0)
        # FP4 打包权重只有 RoPE 留出空间时才提前搬入。
        if cbd.const_expr(self.mm_b_fp4 and self.rope_offset == 0):
            first_b_l1_tile_prefetched = False
        else:
            first_b_l1_tile_prefetched = is_first_tile_on_core

        self._mm_load_a_l1_tile(gm_a, gm_a_scale, 0)
        self._mm_prepare_b_l1_tile(gm_b, gm_b_scale, bn_idx, ng_idx, col, 0, first_b_l1_tile_prefetched)
        self._mm_run_l1_k_tile(
            gm_a, gm_b, gm_a_scale, gm_b_scale, l0c, bn_idx, ng_idx, col, l1_a_scale, l1_b_scale,
            first_b_l1_tile_prefetched, 0, self.mm_k // self.mm_l1_k,
        )

    @cbd.jit
    def _mm_run_l1_k_tile(
        self, gm_a, gm_b, gm_a_scale, gm_b_scale, l0c, bn_idx, ng_idx, col, l1_a_scale, l1_b_scale,
        first_b_l1_tile_prefetched, rope_k_tile_idx, l1_k_tile_count,
    ):
        """消费一个连续的 K 分段，保留各路径的原有搬运顺序。"""
        stream_rope = not self.is_mm2 and self.rope_cols < self.mm_k
        for local_l1_k_tile_idx in cbd.range(l1_k_tile_count):
            l1_k_tile_idx = rope_k_tile_idx * l1_k_tile_count + local_l1_k_tile_idx
            next_l1_k_tile_idx = l1_k_tile_idx + 1
            if cbd.const_expr(stream_rope):
                self._mm_load_a_l1_tile(gm_a, gm_a_scale, l1_k_tile_idx * self.mm_l1_k)
                self._mm_prepare_b_l1_tile(
                    gm_b, gm_b_scale, bn_idx, ng_idx, col, l1_k_tile_idx, first_b_l1_tile_prefetched,
                )
            elif next_l1_k_tile_idx < l1_k_tile_count:
                if cbd.const_expr(self.mm_l1_a_slots > 1 and not self.mm_scale_per_l1_k):
                    self._mm_load_a_l1_tile(gm_a, gm_a_scale, next_l1_k_tile_idx * self.mm_l1_k)
                # B 使用双 buffer，可以在当前 K 块的 MMAD 前搬入下一块。
                self._mm_prepare_b_l1_tile(
                    gm_b, gm_b_scale, bn_idx, ng_idx, col, next_l1_k_tile_idx, first_b_l1_tile_prefetched,
                )
                if cbd.const_expr(self.mm_l1_a_slots > 1 and self.mm_scale_per_l1_k):
                    self._mm_load_a_l1_tile(gm_a, gm_a_scale, next_l1_k_tile_idx * self.mm_l1_k)
            self._mm_compute_l1_k_tile(l0c, bn_idx, l1_a_scale, l1_b_scale, l1_k_tile_idx, local_l1_k_tile_idx)
            if cbd.const_expr(not stream_rope and self.mm_l1_a_slots == 1):
                if next_l1_k_tile_idx < l1_k_tile_count:
                    self._mm_load_a_l1_tile(gm_a, gm_a_scale, next_l1_k_tile_idx * self.mm_l1_k)

    @cbd.jit
    def _mm_compute_l1_k_tile(self, l0c, bn_idx, l1_a_scale, l1_b_scale, l1_k_tile_idx, a_scale_l1_tile_idx):
        bn, bk = self.mm_bn_bk_map[bn_idx]
        # MM2 NZ 的 scale buffer 每次只存当前 L1 K 块。
        b_scale_l1_tile_idx = 0 if self.mm_scale_per_l1_k else l1_k_tile_idx
        l1_a = self.mm_l1_a.next_read()
        l1_b = self.mm_l1_b[bn_idx].next_read()
        if cbd.const_expr(self.mm_scale_per_l1_k):
            l1_a_scale = self.mm_l1_a_scales.next_read().permute(1, 0)
            l1_b_scale = self.mm_l1_b_scales[bn_idx].next_read()
            a_scale_l1_tile_idx = 0

        l1_b_kn = self.weight_nz and not self.mm_b_fp4
        if cbd.const_expr(l1_b_kn):
            l1_b = cbd.from_tensor(l1_b, cbd.dtypes.float8_e4m3fn, l1_b.layout, data_format="nz")
        # MM1 的 Fixpipe 将完整 L0C 等分给两个 AIV，MM2 按有效 M/N 搬运。
        valid_rows, valid_cols = l0c.shape if cbd.const_expr(self.is_mm2) else (self.mm1_bm, bn)
        l0_k_tile_count = self.mm_l1_k // bk
        scale_step = bk // MX_SCALE_GROUP_SIZE
        for l0_k_tile_idx in cbd.range(l0_k_tile_count, unroll_full=self.is_mm2 or self.rope_cols >= self.mm_k):
            a_scale_k_offset = (a_scale_l1_tile_idx * l0_k_tile_count + l0_k_tile_idx) * scale_step
            b_scale_k_offset = (b_scale_l1_tile_idx * l0_k_tile_count + l0_k_tile_idx) * scale_step
            # MX load 需要视图显式描述当前 K tile 的物理分形范围。
            l1_a_tile = cbd.tile_slice(l1_a, (self.mm1_bm, bk), (0, l0_k_tile_idx))
            if cbd.const_expr(self.is_mm2):
                l1_a_tile = l1_a_tile[:valid_rows, None]
                l1_a_scale_tile = l1_a_scale[:valid_rows, a_scale_k_offset:a_scale_k_offset + scale_step]
            else:
                l1_a_scale_tile = l1_a_scale[None, a_scale_k_offset:a_scale_k_offset + scale_step]
            if cbd.const_expr(l1_b_kn):
                l1_b_tile = cbd.tile_slice(l1_b, (bk, bn), (l0_k_tile_idx, 0))[None, :valid_cols].permute(1, 0)
            else:
                l1_b_tile = cbd.tile_slice(l1_b, (bn, bk), (0, l0_k_tile_idx))[:valid_cols, None]
            l1_b_scale_tile = l1_b_scale[b_scale_k_offset:b_scale_k_offset + scale_step, :valid_cols]
            l0a, l0b = self.mm_l0a[bn_idx].next_write(), self.mm_l0b[bn_idx].next_write()
            if cbd.const_expr(self.is_mm2):
                l0a = cbd.from_tensor(l0a, layout=cbd.make_layout(l1_a_tile.shape))
                l0b = cbd.from_tensor(l0b, layout=cbd.make_layout(l1_b_tile.shape))
            cbd.mem_copy(l0a, l1_a_tile, options=cbd.copy_options(mx_scale=l1_a_scale_tile))
            cbd.mem_copy(l0b, l1_b_tile, options=cbd.copy_options(mx_scale=l1_b_scale_tile))
            last = (l1_k_tile_idx + 1) * self.mm_l1_k == self.mm_k and l0_k_tile_idx + 1 == l0_k_tile_count
            cbd.matmul(
                l0c, l0a, l0b, init=l1_k_tile_idx == 0 and l0_k_tile_idx == 0,
                unit_flag=3 if last else 2, mx_mode=True,
            )

    @cbd.jit
    def _mm_load_a_l1_tile(self, gm_a, gm_a_scale, gm_k_offset):
        cbd.mem_copy(
            self.mm_l1_a.next_write()[:gm_a.shape[0], None], gm_a[None, gm_k_offset:gm_k_offset + self.mm_l1_k],
        )
        if cbd.const_expr(self.mm_scale_per_l1_k):
            self._mm_load_a_scale_to_l1(gm_a_scale, gm_k_offset)

    @cbd.jit
    def _mm_prepare_b_l1_tile(self, gm_b, gm_b_scale, bn_idx, ng_idx, col, l1_k_tile_idx, first_b_l1_tile_prefetched):
        gm_k_offset = l1_k_tile_idx * self.mm_l1_k
        if l1_k_tile_idx != 0 or not first_b_l1_tile_prefetched:
            if cbd.const_expr(self.mm_b_fp4):
                self._mm1_load_fp4_b_to_ub(gm_b, ng_idx, col, self.mm_bn_bk_map[bn_idx][0], gm_k_offset)
            else:
                self._mm_load_b_to_l1(gm_b, bn_idx, ng_idx, col, gm_k_offset)
                if cbd.const_expr(self.mm_scale_per_l1_k):
                    self._mm_load_b_scale_to_l1(gm_b_scale, bn_idx, ng_idx, col, gm_k_offset)
        if cbd.const_expr(self.mm_b_fp4):
            self._mm1_unpack_fp4_b_to_l1(bn_idx)

    @cbd.jit
    def _mm_load_a_scale_to_l1(self, gm_a_scale, gm_k_offset):
        gm_scale_k_offset = gm_k_offset // MX_SCALE_GROUP_SIZE
        l1_a_scale = self.mm_l1_a_scales.next_write().permute(1, 0)
        cbd.mem_copy(
            l1_a_scale[:gm_a_scale.shape[0], None],
            gm_a_scale[None, gm_scale_k_offset:gm_scale_k_offset + l1_a_scale.shape[1]],
        )

    @cbd.jit
    def _mm_load_b_to_l1(self, gm_b, bn_idx, ng_idx, col, gm_k_offset):
        bn = self.mm_bn_bk_map[bn_idx][0]
        valid_cols = min(bn, self.mm_n - col)
        l1_b = self.mm_l1_b[bn_idx].next_write()
        options = cbd.copy_options(l2_cache_ctl=self.mm_b_l2_cache_ctl)
        if cbd.const_expr(self.weight_nz):
            if cbd.const_expr(len(gm_b.shape) == 3):
                gm_b = cbd.tile_slice(gm_b, (1, self.mm_k, self.mm_n), (ng_idx, 0, 0))
            gm_b_tile = cbd.tile_slice(gm_b, (self.mm_l1_k, bn), (gm_k_offset // self.mm_l1_k, col // bn))
            l1_b_tile, gm_b_tile = l1_b[None, :valid_cols], gm_b_tile[None, :valid_cols]
        else:
            row = ng_idx * self.mm_n + col
            b_rows = self.mm_n * (gm_b.shape[0] if len(gm_b.shape) == 3 else 1)
            gm_b_tile = gm_b.view(b_rows, self.mm_k)[row:row + valid_cols, gm_k_offset:gm_k_offset + self.mm_l1_k]
            l1_b_tile = l1_b[:valid_cols, None]
        cbd.mem_copy(l1_b_tile, gm_b_tile, options=options)

    @cbd.jit
    def _mm_load_b_scale_to_l1(self, gm_b_scale, bn_idx, ng_idx, col, gm_k_offset):
        bn = self.mm_bn_bk_map[bn_idx][0]
        valid_cols = min(bn, self.mm_n - col)
        scale_rows = self.mm_l1_b_scales[bn_idx].shape[0]
        gm_b_scale = gm_b_scale.view(dtype=cbd.dtypes.float8_e8m0)
        if cbd.const_expr(self.weight_nz and not self.mm_b_fp4):
            if cbd.const_expr(len(gm_b_scale.shape) == 4):
                gm_b_scale = gm_b_scale[ng_idx:ng_idx + 1, None, None, None].view(
                    self.mm_k // MX_SCALE_PAIR_SIZE, self.mm_n, 2,
                )
            gm_scale_k_offset = gm_k_offset // MX_SCALE_PAIR_SIZE
            gm_b_scale_tile = (
                gm_b_scale[gm_scale_k_offset:gm_scale_k_offset + scale_rows // 2, col:col + valid_cols, None]
                .permute(0, 2, 1).reshape(scale_rows, valid_cols)
            )
        else:
            row = ng_idx * self.mm_n + col
            gm_scale_k_offset = gm_k_offset // MX_SCALE_GROUP_SIZE
            b_scale_rows = self.mm_n * (gm_b_scale.shape[0] if len(gm_b_scale.shape) == 4 else 1)
            gm_b_scale = gm_b_scale.view(b_scale_rows, self.mm_k // MX_SCALE_GROUP_SIZE)
            gm_b_scale_tile = gm_b_scale[
                row:row + valid_cols, gm_scale_k_offset:gm_scale_k_offset + scale_rows,
            ].permute(1, 0)
        cbd.mem_copy(self.mm_l1_b_scales[bn_idx].next_write()[None, :valid_cols], gm_b_scale_tile)

    @cbd.jit
    def _mm1_load_fp4_b_to_ub(self, gm_b, ng_idx, col, bn, gm_k_offset):
        """每个 AIV 搬入半个 N 块的 FP4 字节，清零尾块的空分形。"""
        n_fractals = bn // (MIX_VECTOR_PARTS * FP4_TILE_N)
        k_fractals = self.mm_l1_k // MX_SCALE_GROUP_SIZE
        n_fractal = col // FP4_TILE_N + cbd.get_subblock_id() * n_fractals
        k_fractal = gm_k_offset // MX_SCALE_GROUP_SIZE
        valid_n_fractals = max(0, min(n_fractals, MM1_OUTPUT_DIM // FP4_TILE_N - n_fractal))
        ub_packed = self.mm1_fp4_packed_ub.next_write()
        if cbd.const_expr(MM1_OUTPUT_DIM % bn):
            if valid_n_fractals < n_fractals:
                with cbd.vf():
                    mask = reg.create_mask("all", elem_bits=8)
                    zero = reg.vdups(0, cbd.dtypes.uint8)
                    for byte_offset in cbd.range(0, n_fractals * k_fractals * FP4_TILE_BYTES, QUANT_CHUNK_ELEMENTS):
                        reg.vstore(ub_packed, byte_offset, zero, mask)
                    reg.vmem_bar("vst_vld")
        if valid_n_fractals > 0:
            gm_b_group = gm_b[ng_idx:ng_idx + 1, None, None].view(gm_b.shape[1:])
            ub_packed_nz = cbd.from_tensor(
                ub_packed, cbd.dtypes.uint8, cbd.make_layout((k_fractals, n_fractals * FP4_TILE_BYTES)),
                data_format="nd",
            )
            cbd.mem_copy(
                ub_packed_nz[None, :valid_n_fractals * FP4_TILE_BYTES],
                gm_b_group[k_fractal:k_fractal + k_fractals,
                           n_fractal * FP4_TILE_BYTES:(n_fractal + valid_n_fractals) * FP4_TILE_BYTES],
            )

    @cbd.jit
    def _mm1_unpack_fp4_fractal(self, ub_packed, ub_expanded, packed_fractal, nz_fractal, mask, sign, magnitude):
        byte = reg.vload(ub_packed, packed_fractal * FP4_TILE_BYTES)
        # 重排两个 E2M1 半字节，保留投影结束时的 64 倍补偿。
        lo_bits = reg.vbitwise_or(
            reg.vbitwise_and(reg.vshl(byte, 4, mask=mask), sign, mask=mask),
            reg.vbitwise_and(reg.vshl(byte, 2, mask=mask), magnitude, mask=mask), mask=mask,
        )
        hi_bits = reg.vbitwise_or(
            reg.vbitwise_and(byte, sign, mask=mask),
            reg.vbitwise_and(reg.vshr(byte, 2, mask=mask), magnitude, mask=mask), mask=mask,
        )
        lo = reg.vreinterpret(lo_bits, cbd.dtypes.float8_e4m3fn)
        hi = reg.vreinterpret(hi_bits, cbd.dtypes.float8_e4m3fn)
        reg.vstore_interleave(ub_expanded, nz_fractal * 2 * FP4_TILE_BYTES, lo, hi)

    @cbd.jit
    def _mm1_unpack_fp4_b_to_l1(self, bn_idx):
        """将 FP4 字节网格逐分形展开成 L1 所需的 FP8 NZ。"""
        bn = self.mm_bn_bk_map[bn_idx][0]
        aiv_bn = bn // MIX_VECTOR_PARTS
        n_fractals = aiv_bn // FP4_TILE_N
        max_n_fractals = max(self.mm_bn_bk_map)[0] // (MIX_VECTOR_PARTS * FP4_TILE_N)
        k_fractals = self.mm_l1_k // MX_SCALE_GROUP_SIZE
        ub_packed = self.mm1_fp4_packed_ub.next_read()
        ub_expanded = self.mm1_fp4_expanded_ub.next_write()
        with cbd.vf():
            mask = reg.create_mask("all", elem_bits=8)
            sign = reg.vdups(0x80, cbd.dtypes.uint8)
            magnitude = reg.vdups(0x1C, cbd.dtypes.uint8)
            if cbd.const_expr(n_fractals != max_n_fractals):
                # 窄条带源和目标的 N 步长不同，按 K、N 两级循环避免逐分形求余。
                for k_fractal in cbd.range(k_fractals):
                    for n_fractal in cbd.range(n_fractals):
                        self._mm1_unpack_fp4_fractal(
                            ub_packed, ub_expanded, k_fractal * n_fractals + n_fractal,
                            k_fractal * max_n_fractals + n_fractal, mask, sign, magnitude,
                        )
            else:
                for fractal in cbd.range(n_fractals * k_fractals):
                    self._mm1_unpack_fp4_fractal(ub_packed, ub_expanded, fractal, fractal, mask, sign, magnitude)
            reg.vmem_bar(mode="vst_vld")
        ub_expanded = self.mm1_fp4_expanded_ub.next_read()
        l1_b = self.mm_l1_b[bn_idx].next_write()[:bn, None].partition(
            dim=0, part_id=cbd.get_subblock_id(), alignment=FP4_TILE_N,
        )
        cbd.mem_copy(l1_b, ub_expanded[:aiv_bn, None])

    @cbd.jit
    def _mm_make_prefetch_buffers(self, bn_bk_map, n, k, l1_k, weight_fp4, l1_slots, *, is_mm2=False):
        """创建预取使用的 L1 B 和 B scale buffer，并确定 A scale 布局。"""
        # MM1/MM2 进入此方法时绑定当前规格，后续使用同一片物理 L1。
        self.mm_bn_bk_map, self.mm_n, self.mm_k, self.mm_l1_k = bn_bk_map, n, k, l1_k
        self.mm_b_fp4, self.is_mm2 = weight_fp4, is_mm2
        self.mm_scale_per_l1_k = is_mm2 and self.weight_nz
        self.mm_l1_a_slots, self.mm_l1_b_slots = l1_slots

        if cbd.const_expr(self.mm_scale_per_l1_k):
            a_scale_k = b_scale_k = l1_k
            self.mm_l1_a_scale_slots, l1_b_scale_slots = l1_slots
        else:
            a_scale_k, b_scale_k = (k if is_mm2 else self.rope_cols), k
            self.mm_l1_a_scale_slots, l1_b_scale_slots = 1, 1
        self.mm_l1_a_scale_shape = (a_scale_k // MX_SCALE_GROUP_SIZE, self.mm1_bm)
        l1_b_scale_rows = b_scale_k // MX_SCALE_GROUP_SIZE

        # L1 的 buffer 地址依次为 A、B、A scale、B scale。
        max_bn = max(bn_bk_map)[0]
        self.mm_l1_b_offset = self.mm_l1_a_slots * self.mm1_bm * l1_k
        self.mm_l1_a_scale_offset = self.mm_l1_b_offset + self.mm_l1_b_slots * max_bn * l1_k
        l1_b_scale_offset = self.mm_l1_a_scale_offset + (
            self.mm_l1_a_scale_slots * self.mm_l1_a_scale_shape[0] * self.mm_l1_a_scale_shape[1]
        )

        if cbd.const_expr(not weight_fp4):
            l1_b = []
            for bn, _ in bn_bk_map:
                l1_b_layout = cbd.make_layout((l1_k, bn) if self.weight_nz else (bn, l1_k))
                l1_b_dtype = cbd.dtypes.uint8 if self.weight_nz else cbd.dtypes.float8_e4m3fn
                l1_b.append(cbd.make_ring_buffer([
                    cbd.from_tensor(
                        self.l1_tensor, l1_b_dtype, l1_b_layout,
                        offset=self.mm_l1_b_offset + slot * bn * l1_k, data_format="nz",
                    )
                    for slot in range(self.mm_l1_b_slots)
                ]))
            self.mm_l1_b = tuple(l1_b)
        self.mm_l1_b_scales = tuple(cbd.make_ring_buffer([
            cbd.from_tensor(
                self.l1_tensor, cbd.dtypes.float8_e8m0, cbd.make_layout((l1_b_scale_rows, bn)),
                offset=l1_b_scale_offset + slot * l1_b_scale_rows * bn, data_format="nz",
            )
            for slot in range(l1_b_scale_slots)
        ]) for bn, _ in bn_bk_map)

    @cbd.jit
    def _mm_make_compute_buffers(self, l0c_num):
        """创建 A、A scale、L0A/B/C 和 FP4 展开使用的 buffer。"""
        if cbd.const_expr(self.mm_b_fp4):
            if cbd.const_expr(self.rope_offset == 0):
                self._mm1_make_fp4_packed_buffer()
            max_bn = max(self.mm_bn_bk_map)[0]
            half_n = max_bn // MIX_VECTOR_PARTS
            self.mm1_fp4_expanded_ub = cbd.make_ring_buffer([
                cbd.from_tensor(
                    self.ub_tensor, cbd.dtypes.float8_e4m3fn, cbd.make_layout((half_n, self.mm_l1_k)),
                    offset=self.mm1_fp4_packed_ub.depth * half_n * self.mm_l1_k // FP4_VALUES_PER_BYTE,
                    data_format="nz",
                ),
            ])
            self.mm_l1_b = (cbd.make_ring_buffer([
                cbd.from_tensor(
                    self.l1_tensor, cbd.dtypes.float8_e4m3fn, cbd.make_layout((max_bn, self.mm_l1_k)),
                    offset=self.mm_l1_b_offset + slot * max_bn * self.mm_l1_k, data_format="nz",
                )
                for slot in range(self.mm_l1_b_slots)
            ]),) * len(self.mm_bn_bk_map)

        self.mm_l1_a = cbd.make_ring_buffer([
            cbd.from_tensor(
                self.l1_tensor, cbd.dtypes.float8_e4m3fn, cbd.make_layout((self.mm1_bm, self.mm_l1_k)),
                offset=slot * self.mm1_bm * self.mm_l1_k, data_format="nz",
            )
            for slot in range(self.mm_l1_a_slots)
        ])
        l1_a_scale_slot_bytes = self.mm_l1_a_scale_shape[0] * self.mm_l1_a_scale_shape[1]
        self.mm_l1_a_scales = cbd.make_ring_buffer([
            cbd.from_tensor(
                self.l1_tensor, cbd.dtypes.float8_e8m0, cbd.make_layout(self.mm_l1_a_scale_shape),
                offset=self.mm_l1_a_scale_offset + slot * l1_a_scale_slot_bytes, data_format="nz",
            )
            for slot in range(self.mm_l1_a_scale_slots)
        ])
        self.mm_l0a = tuple(cbd.make_ring_buffer([
            cbd.from_tensor(
                self.l0a_tensor, cbd.dtypes.float8_e4m3fn, cbd.make_layout((self.mm1_bm, bk)),
                offset=slot * self.mm1_bm * bk, data_format="nz",
            )
            for slot in range(2)
        ]) for _, bk in self.mm_bn_bk_map)
        self.mm_l0b = tuple(cbd.make_ring_buffer([
            cbd.from_tensor(self.l0b_tensor, cbd.dtypes.float8_e4m3fn, cbd.make_layout((bn, bk)),
                            offset=slot * bn * bk, data_format="nz")
            for slot in range(2)
        ]) for bn, bk in self.mm_bn_bk_map)
        self.mm_l0c = tuple(cbd.make_ring_buffer([
            cbd.from_tensor(
                self.l0c_tensor, cbd.dtypes.float32, cbd.make_layout((self.mm1_bm, bn)),
                offset=slot * self.mm1_bm * bn * FP32_BYTES, data_format="nz",
            )
            for slot in range(count)
        ]) for (bn, _), count in zip(self.mm_bn_bk_map, l0c_num))

    @cbd.jit
    def _mm1_make_fp4_packed_buffer(self):
        """创建 FP4 权重搬入使用的 UB buffer。"""
        shape = (max(self.mm_bn_bk_map)[0] // (MIX_VECTOR_PARTS * FP4_TILE_N),
                 self.mm_l1_k * FP4_TILE_N // FP4_VALUES_PER_BYTE)
        slot_bytes = shape[0] * shape[1]
        self.mm1_fp4_packed_ub = cbd.make_ring_buffer([self.ub_tensor[:slot_bytes,].view(shape)])

    def _make_ub_buffers(self, offset, specs):
        """在给定 UB 偏移处连续创建缓冲，depth=0 表示普通 Buffer。"""
        buffers = []
        for shape, dtype, depth in specs:
            offset = _align_up(offset, UB_ALIGNMENT)
            slot_bytes = dtype.bytes_for(shape[0] * shape[1])
            slots = [
                self.ub_tensor[offset + slot * slot_bytes:offset + (slot + 1) * slot_bytes,].view(dtype).view(shape)
                for slot in range(depth or 1)
            ]
            buffers.append(cbd.make_ring_buffer(slots) if depth else cbd.make_buffer(slots[0]))
            offset += len(slots) * slot_bytes
        return tuple(buffers)

    @cbd.jit
    def _quant_publish_activation(self, ready_flag, cube_flag, *, wait_on_cube=True):
        """所有 AIV 写回后发布激活，流式 MM1 在消费对应 K 块时等待。"""
        cbd.vec_sync_block_arrive(cbd.PIPE.MTE3, ready_flag, mode=0)
        cbd.vec_sync_block_wait(cbd.PIPE.MTE3, ready_flag, mode=0)
        cbd.vec_sync_block_arrive(cbd.PIPE.MTE3, cube_flag)
        if cbd.const_expr(wait_on_cube):
            cbd.cube_sync_block_wait(cbd.PIPE.MTE2, cube_flag)

    @cbd.jit
    def _mx_quantize_bf16_to_fp8(self, source, output, scales, reciprocal, count):
        """量化 BF16，并复用一次指数归约得到 scale 和倒数。"""
        with cbd.vf(mode="simd"):
            mask_b16 = reg.create_mask("all", elem_bits=16)
            mask_b32 = reg.create_mask("all", elem_bits=32)
            scale_mask = reg.create_mask("vl16", elem_bits=8)
            reciprocal_mask = reg.create_mask("vl16", elem_bits=16)
            exponent_mask = reg.vdups(0x7F80, cbd.dtypes.uint16, mask=mask_b16)
            min_exponent = reg.vdups(8, cbd.dtypes.uint16, mask=mask_b16)
            reciprocal_bias = reg.vdups(262, cbd.dtypes.uint16, mask=mask_b16)
            lane = reg.varange(0, cbd.dtypes.uint8)
            odd_bit = reg.vdups(1, cbd.dtypes.uint8, mask=scale_mask)
            even_mask = reg.veqs(reg.vbitwise_and(lane, odd_bit, mask=scale_mask), 0, mask=scale_mask)
            scale_store = reg.vstore_unalign_begin(scales)
            for chunk in cbd.range(0, count // QUANT_CHUNK_ELEMENTS, 1):
                x0, x1 = reg.vload_deinterleave(source, chunk * QUANT_CHUNK_ELEMENTS)
                even_exponent = reg.vbitwise_and(reg.vreinterpret(x0, cbd.dtypes.uint16), exponent_mask, mask=mask_b16)
                odd_exponent = reg.vbitwise_and(reg.vreinterpret(x1, cbd.dtypes.uint16), exponent_mask, mask=mask_b16)
                exponent = reg.vmax(even_exponent, odd_exponent, mask=mask_b16)
                exponent = reg.vreduce_max_datablock(exponent, mask=mask_b16)
                exponent = reg.vmaxs(reg.vshr(exponent, 7, mask=mask_b16), 8, mask=mask_b16)
                scale_bytes = reg.vcast(
                    reg.vsub(exponent, min_exponent, mask=mask_b16), cbd.dtypes.uint8,
                    mask=mask_b16, reg_layout=cbd.RegLayout.ZERO,
                )
                packed = reg.vsqueeze_and_storeunalign_init(scale_bytes, mask=even_mask)
                reg.vsqueeze_and_storeunalign(scales, 0, packed, scale_store)
                reciprocal_bits = reg.vshl(reg.vsub(reciprocal_bias, exponent, mask=mask_b16), 7, mask=mask_b16)
                reciprocal_bits = reg.vselect(
                    reg.vdups(0, cbd.dtypes.uint16, mask=mask_b16), reciprocal_bits,
                    cond_mask=reg.veqs(exponent, 8, mask=mask_b16),
                )
                reciprocal_pair, _ = reg.vinterleave(reciprocal_bits, reciprocal_bits)
                reg.vstore(reciprocal, chunk * 16, reciprocal_pair, reciprocal_mask)
            reg.vsqueeze_and_storeunalign_finalize(scales, 0, scale_store)
            reg.vmem_bar(mode="vst_vld")

            for chunk in cbd.range(0, count // QUANT_CHUNK_ELEMENTS, 1):
                for half in cbd.range_constexpr(2):
                    elem_offset = chunk * QUANT_CHUNK_ELEMENTS + half * 128
                    x = reg.vload(source, elem_offset)
                    factor = reg.vload_broadcast(reciprocal, chunk * 16 + half * 8, mode="datablock")
                    scaled = reg.vmul(x, reg.vreinterpret(factor, cbd.dtypes.bfloat16), mask=mask_b16)
                    even = reg.vcast(scaled, cbd.dtypes.float32, mask=mask_b16, reg_layout=cbd.RegLayout.ZERO)
                    odd = reg.vcast(scaled, cbd.dtypes.float32, mask=mask_b16, reg_layout=cbd.RegLayout.ONE)
                    v0, v1 = reg.vinterleave(even, odd)
                    fp8_lo = reg.vcast(
                        v0, cbd.dtypes.float8_e4m3fn, mask=mask_b32, reg_layout=cbd.RegLayout.ZERO, saturate=True,
                    )
                    fp8_hi = reg.vcast(
                        v1, cbd.dtypes.float8_e4m3fn, mask=mask_b32, reg_layout=cbd.RegLayout.ZERO, saturate=True,
                    )
                    reg.vstore_pack(output, elem_offset, fp8_lo, mask_b32, pack_mode=reg.PackMode.B32_TO_B8)
                    reg.vstore_pack(output, elem_offset + 64, fp8_hi, mask_b32, pack_mode=reg.PackMode.B32_TO_B8)
            reg.vmem_bar(mode="vst_vld")
