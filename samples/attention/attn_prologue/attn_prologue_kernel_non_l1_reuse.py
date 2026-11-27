# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""NonL1Reuse AttnPrologue 的公共实现及四种单继承流水组合。"""

from __future__ import annotations

# fmt: off

import cannbotdsl as cbd
import cannbotdsl.reg as reg


from .attn_prologue_kernel_common import AttnPrologueCommon
from .attn_prologue_tiling import (A2_BUF_DEPTH_DECODE, A2_BUF_DEPTH_PREFILL, AIV_PER_BLOCK, BN_B_FULL_HEAD,
                                   FIXPIPE_ACCUMULATE, FIXPIPE_DUAL_DST, FIXPIPE_FINAL, FLOAT8_E4M3FN, FLOAT8_E8M0,
                                   L0_DEPTH, MX_GROUP, MX_PAIR, MX_PAIR_WIDTH, NZ_M_FRAC, RESIDENT_HEAD_PANELS,
                                   UB_ALIGN, VL, W_BUF, ZN_FRACTAL_BYTES, ceil_div, kv_row_bytes)


class NonL1ReuseKernelBase(AttnPrologueCommon):
    """提供 Branch/Generic 的 A 阶段实现和两种 B 策略共用的权重加载、KV/Q 后处理。"""

    @cbd.jit
    def _publish_a1(self):
        """发布 A1 Cube 写出的 QA/KV FP32 部分和，使配对 Vector 核可安全读取。

        FIXPIPE 通知确保本核 A1 写出完成，再进行 Cube 全核汇合并释放 AIV；mode=0 处理全核
        生产者握手，mode=2 处理跨核配对等待。该函数只建立可见性和先后关系，不搬运数据，也不要求
        Vector 在发布后立刻停止后续独立工作。
        """
        # A1 只有 Cube 生产者：FIXPIPE 通知确保写出完成，随后完成 Cube 全核汇合，
        # 再释放配对的 AIV，无需额外的 AIV→AIC 汇合。
        cbd.cube_sync_block_arrive(cbd.PIPE.FIXPIPE, 0, mode=0)
        cbd.cube_sync_block_wait(cbd.PIPE.FIXPIPE, 0, mode=0)
        cbd.cube_sync_block_arrive(cbd.PIPE.FIXPIPE, 1, mode=2)
        cbd.vec_sync_block_wait(cbd.PIPE.MTE2, 1, mode=2)

    def _alloc_a1_buffers(self):
        """为分支和通用 A1 投影统一分配 L1/L0 缓冲并返回布局参数。

        激活 payload/scale 使用双槽 Channel，权重 payload/scale 均使用 W_BUF 深 Channel；L0A/L0B 按 L0_DEPTH
        流水，L0C 保存单个 FP32 输出 tile。返回值还包括
        scale_l0、scale_l1、groups_l1，分别描述 L0 组数、L1 组数和 GM scale 的 pair 分组宽度，
        供 _a1_tile、_mm_a、_fill_w_a 共享，避免各路径重复计算布局。
        """
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols, k_l1 = (tiling.bm_a1, tiling.bn_a1, tiling.bk_a1,
                                                                        tiling.k_l1_a1)
        scale_l0 = reduction_tile_cols // MX_GROUP
        scale_l1 = k_l1 // MX_GROUP
        groups_l1 = k_l1 // MX_PAIR_WIDTH
        l1_x = cbd.RingBuffer(cbd.MemLoc.L1, FLOAT8_E4M3FN,
                              cbd.make_layout((token_tile_rows, tiling.k_l1_x_a1)), depth=2, data_format="nz")
        l1_sx = cbd.RingBuffer(cbd.MemLoc.L1, cbd.dtypes.uint8,
                               cbd.make_layout((ceil_div(token_tile_rows,
                                                         NZ_M_FRAC) * ceil_div(tiling.k_l1_x_a1 // MX_GROUP,
                                                                               MX_PAIR) * ZN_FRACTAL_BYTES, )), depth=2,
                               data_format="nd")
        l1_w = cbd.RingBuffer(cbd.MemLoc.L1, cbd.dtypes.int8,
                              cbd.make_layout((output_tile_cols, k_l1)), depth=W_BUF, data_format="nz")
        l1_sw = cbd.RingBuffer(cbd.MemLoc.L1, FLOAT8_E8M0,
                               cbd.make_layout((scale_l1, output_tile_cols)), depth=W_BUF, data_format="nz")
        l0a = cbd.RingBuffer(cbd.MemLoc.L0A, FLOAT8_E4M3FN,
                             cbd.make_layout((token_tile_rows, reduction_tile_cols)), depth=L0_DEPTH)
        l0b = cbd.RingBuffer(cbd.MemLoc.L0B, FLOAT8_E4M3FN,
                             cbd.make_layout((output_tile_cols, reduction_tile_cols)), depth=L0_DEPTH)
        l0c = cbd.RingBuffer(cbd.MemLoc.L0C, cbd.dtypes.float32,
                             cbd.make_layout((token_tile_rows, output_tile_cols)), depth=1)
        return scale_l0, scale_l1, groups_l1, l1_x, l1_sx, l1_w, l1_sw, l0a, l0b, l0c

    @cbd.jit
    def _service_kv(self, gm_gamma_kv, gm_sin, gm_cos, gm_index, gm_cache, gm_ws_kv, norm_eps, token_tile_idx):
        """在 B 阶段任务间隙处理一个 KV token 分块，实现 Cube QB 与 Vector KV 重叠。

        对仍在实际 token 范围内的分块按需分配 UB、加载 gamma/RoPE 表，并调用 _kv_tile 完成 split-K
        归约、RMSNorm、RoPE、FP8 量化及 cache 写回；超出实际范围的调度槽直接返回。该函数由 _stage_b
        穿插调用并负责尾部补齐，保证每个 KV 分块恰好处理一次，同时避免为无效槽分配和执行计算。
        """
        tiling = self.tiling
        token_tile_rows, head = tiling.bm_a2, tiling.head_size
        tokens = gm_index.shape[0]
        actual_tiles = (tokens + token_tile_rows - 1) // token_tile_rows
        if token_tile_idx < actual_tiles:
            a2_depth = (A2_BUF_DEPTH_PREFILL if tiling.template == "split_t" else A2_BUF_DEPTH_DECODE)
            ub_kv = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32,
                                   cbd.make_layout((token_tile_rows, head)), depth=a2_depth)
            ub_gamma_kv = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((1, head)), depth=1)
            ub_sin = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((token_tile_rows, VL)), depth=1)
            ub_cos = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((token_tile_rows, VL)), depth=1)
            ub_kvb = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.uint8, cbd.make_layout((1, kv_row_bytes(head))), depth=2)
            ub_tail = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((1, 2 * VL)))
            ub_kv_bits = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.int32, cbd.make_layout((VL,)))
            cbd.mem_copy(ub_gamma_kv.next_write(), cbd.tile_slice(gm_gamma_kv, (1, head), (0, 0)))
            gamma_kv = ub_gamma_kv.next_read()
            kv_sum = None
            if cbd.const_expr(tiling.split_k > 1):
                kv_sum = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((token_tile_rows, head)))
            rows = cbd.tile_slice(gm_index, (token_tile_rows, 1), (token_tile_idx, 0)).shape[0]
            # KV 分支：沿 head 特征轴执行 RMSNorm、RoPE 和 per-32 BF16 scale 的 E4M3FN 转换。
            # 与 _reduce_normalize_a 中的对应循环共享 KV 分支语义，修改需同步。
            self._kv_tile(gm_ws_kv, gm_sin, gm_cos, gm_index, gm_cache, ub_kv, ub_sin, ub_cos, ub_kvb, ub_tail,
                          ub_kv_bits, gamma_kv, kv_sum, token_tile_idx, rows, norm_eps)

    @cbd.jit
    def _a1_tile(self, l1_a, l1_sa, l1_b, l1_sb, l0a, l0b, l0c, gm_a, gm_dsa, gm_b, gm_dsb, gm_ws, projection_task_idx,
                 n_count, scale_l0, groups_l1):
        """计算一个 A1 投影任务的单个 split-K 部分和。

        projection_task_idx 被解码为 token 行块 m、输出列块 n 和 split；随后遍历该 split 的 K 窗口，
        用 _fill_w_a 装载权重、用 _mm_a 在 L0C 中累加 MXFP8 matmul。窗口数支持 W_BUF 成对预取，
        不能整除时退回单窗口顺序执行；最后把 L0C 的 FP32 结果按 GM 目的逻辑形状写回交换区，
        由 A2 阶段再跨 split 归约。
        """
        tiling = self.tiling
        token_tile_rows, output_tile_cols, k_l1 = (tiling.bm_a1, tiling.bn_a1, tiling.k_l1_a1)
        n_kw = tiling.hidden_size // tiling.split_k // k_l1
        logical_slot = projection_task_idx // tiling.split_k
        split = projection_task_idx % tiling.split_k
        m = logical_slot // n_count
        n = logical_slot % n_count
        start_kw = split * n_kw
        accumulator = l0c.next_write()
        # 成对展开 K 窗口，使 depth-W_BUF 权重 Channel 在一次循环体中按窗口轮转。
        if cbd.const_expr(n_kw % W_BUF == 0):
            for kwp in range(n_kw // W_BUF):
                kw0 = start_kw + kwp * W_BUF
                for u in cbd.range_constexpr(W_BUF):
                    self._fill_w_a(l1_b, l1_sb, gm_b, gm_dsb, n, kw0 + u, groups_l1)
                for u in cbd.range_constexpr(W_BUF):
                    self._mm_a(l1_a, l1_sa, l1_b, l1_sb, l0a, l0b, accumulator, gm_a, gm_dsa, m, kw0 + u, start_kw,
                               scale_l0)
        else:
            # 窗口数量为奇数时退回单窗口顺序执行，不再重叠预取。
            for kw in range(start_kw, start_kw + n_kw):
                self._fill_w_a(l1_b, l1_sb, gm_b, gm_dsb, n, kw, groups_l1)
                self._mm_a(l1_a, l1_sa, l1_b, l1_sb, l0a, l0b, accumulator, gm_a, gm_dsa, m, kw, start_kw, scale_l0)
        # L0C 直接写入 GM，不经过 Vector 暂存。
        #  mem_copy 以源域为准：将 L0C 源逻辑域重绑定到目的实际域，
        # 尾块裁剪生效且物理布局保持不变。
        ws_tile = cbd.tile_slice(gm_ws, (token_tile_rows, output_tile_cols), (split * tiling.m_tiles_a1 + m, n))
        cbd.mem_copy(ws_tile, cbd.from_tensor(accumulator, layout=cbd.make_layout(ws_tile.shape),
                                              physical_layout=accumulator.physical_layout))

    @cbd.jit
    def _mm_a(self, l1_a, l1_sa, l1_b, l1_sb, l0a, l0b, l0c, gm_a, gm_dsa, m, kw, start_kw, scale_l0):
        """完成 A1 一个 L1 K 窗口内全部 L0 分块的 MXFP8 矩阵乘累加。

        根据 a1_window 配置，激活既可整窗预装 L1 后切片，也可逐 L0 K 块从 GM 加载；两条方式都为
        L0A/L0B 搬运附加 mx_scale，使 matmul 使用原生 MXFP8 语义。权重窗口在函数入口读取一次，
        按 kl0 切片复用；kb 决定 L0C 是否初始化，最终部分和保留在 L0C 供后续窗口继续累加。
        """
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols, k_l1 = (tiling.bm_a1, tiling.bn_a1, tiling.bk_a1,
                                                                        tiling.k_l1_a1)
        groups_l0 = reduction_tile_cols // MX_PAIR_WIDTH
        sa_layout = cbd.make_layout((token_tile_rows, tiling.k_l1_x_a1 // MX_GROUP))
        w_ready = cbd.from_tensor(l1_b.next_read(), dtype=FLOAT8_E4M3FN)
        s_ready = l1_sb.next_read()
        if cbd.const_expr(tiling.a1_window):
            cbd.mem_copy(l1_a.next_write(), cbd.tile_slice(gm_a, (token_tile_rows, k_l1), (m, kw)))
            cbd.mem_copy(cbd.from_tensor(l1_sa.next_write(), dtype=FLOAT8_E8M0, layout=sa_layout, data_format="zn"),
                         cbd.tile_slice(gm_dsa, (token_tile_rows, k_l1 // MX_PAIR_WIDTH, MX_PAIR), (m, kw,
                                                                                                    0)).flatten(1, 2))
            a_ready = l1_a.next_read()
            sa_ready = cbd.from_tensor(l1_sa.next_read(), dtype=FLOAT8_E8M0, layout=sa_layout, data_format="zn")
        for kl0 in range(k_l1 // reduction_tile_cols):
            kb = kw * (k_l1 // reduction_tile_cols) + kl0
            # 两个 L0 操作数都需要 mx_scale 元数据，matmul 才会使用原生 MXFP8 指令。
            if cbd.const_expr(tiling.a1_window):
                activation_scale_tile = cbd.tile_slice(sa_ready, (token_tile_rows, scale_l0), (0, kl0))
                cbd.mem_copy(l0a.next_write(),
                             cbd.tile_slice(a_ready, (token_tile_rows, reduction_tile_cols), (0, kl0)),
                             cbd.copy_options(mx_scale=activation_scale_tile))
            else:
                cbd.mem_copy(l1_a.next_write(), cbd.tile_slice(gm_a, (token_tile_rows, reduction_tile_cols), (m, kb)))
                cbd.mem_copy(cbd.from_tensor(l1_sa.next_write(), dtype=FLOAT8_E8M0, layout=sa_layout, data_format="zn"),
                             cbd.tile_slice(gm_dsa, (token_tile_rows, groups_l0, MX_PAIR), (m, kb, 0)).flatten(1, 2))
                cbd.mem_copy(l0a.next_write(),
                             l1_a.next_read(),
                             cbd.copy_options(mx_scale=cbd.from_tensor(l1_sa.next_read(), dtype=FLOAT8_E8M0,
                                                                       layout=sa_layout, data_format="zn")))
            if cbd.const_expr(k_l1 == reduction_tile_cols):
                cbd.mem_copy(l0b.next_write(), w_ready, cbd.copy_options(mx_scale=s_ready))
            else:
                weight_scale_tile = cbd.tile_slice(s_ready, (scale_l0, output_tile_cols), (kl0, 0))
                cbd.mem_copy(l0b.next_write(),
                             cbd.tile_slice(w_ready, (output_tile_cols, reduction_tile_cols), (0, kl0)),
                             cbd.copy_options(mx_scale=weight_scale_tile))
            cbd.matmul(l0c,
                       l0a.next_read(),
                       l0b.next_read(), init=(kb == start_kw * (k_l1 // reduction_tile_cols)), mx_mode=True)

    @cbd.jit
    def _qr_tile(self, gm_ws_qa, gm_qr, gm_qr_bf16, gm_dqr2d, ub_qa, ub_qr, ub_qr_bf16, ub_exp, ub_scale, gamma_qr,
                 qa_sum, qa_comp, token_tile_idx, rows, norm_eps):
        """逐份归约非驻留路径的 QA split，量化并写出 QR、BF16 QR 及 scale。"""
        tiling = self.tiling
        token_tile_rows, rank = tiling.bm_a2, tiling.rank_size
        cbd.mem_copy(ub_qa.next_write(), cbd.tile_slice(gm_ws_qa, (token_tile_rows, rank), (token_tile_idx, 0)))
        qa = ub_qa.next_read()
        qr_slot = ub_qr.next_write()
        qr_bf16_slot = ub_qr_bf16.next_write()
        if cbd.const_expr(tiling.split_k > 1):
            self._acc_partial_qa(qa_sum, qa_comp, qa, rows, rank, True)
            for part in range(1, tiling.split_k):
                cbd.mem_copy(ub_qa.next_write(),
                             cbd.tile_slice(gm_ws_qa, (token_tile_rows, rank),
                                            (part * tiling.m_tiles_a1 * tiling.bm_a1
                                             // token_tile_rows + token_tile_idx,
                                             0)))
                qa_part = ub_qa.next_read()
                self._acc_partial_qa(qa_sum, qa_comp, qa_part, rows, rank, False)
            self._quantize_qr(qa_sum, gamma_qr, qr_slot, ub_qr_bf16, qr_bf16_slot, ub_exp, rows, norm_eps)
        else:
            self._quantize_qr(qa, gamma_qr, qr_slot, ub_qr_bf16, qr_bf16_slot, ub_exp, rows, norm_eps)
        scale_slot = ub_scale.next_write()
        self._pack_scale(ub_exp, scale_slot, rows)
        cbd.mem_copy(cbd.tile_slice(gm_qr, (token_tile_rows, rank), (token_tile_idx, 0)), ub_qr.next_read())
        cbd.mem_copy(cbd.tile_slice(gm_qr_bf16, (token_tile_rows, rank), (token_tile_idx, 0)), ub_qr_bf16.next_read())
        cbd.mem_copy(cbd.tile_slice(gm_dqr2d, (token_tile_rows, tiling.qr_groups), (token_tile_idx, 0)),
                     cbd.from_tensor(ub_scale.next_read(),
                                     layout=cbd.make_layout((token_tile_rows, tiling.qr_groups),
                                                            stride=(tiling.scale_stride, 1))))

    @cbd.jit
    def _kv_tile(self, gm_ws_kv, gm_sin, gm_cos, gm_index, gm_cache, ub_kv, ub_sin, ub_cos, ub_kvb, ub_tail, ub_kv_bits,
                 gamma_kv, kv_sum, token_tile_idx, rows, norm_eps):
        """逐份归约非驻留路径的 KV split，并逐 token 完成归一化、RoPE、量化和 cache 写回。"""
        tiling = self.tiling
        token_tile_rows, head = tiling.bm_a2, tiling.head_size
        cbd.mem_copy(ub_kv.next_write(), cbd.tile_slice(gm_ws_kv, (token_tile_rows, head), (token_tile_idx, 0)))
        cbd.mem_copy(ub_sin.next_write(), cbd.tile_slice(gm_sin, (token_tile_rows, VL), (token_tile_idx, 0)))
        cbd.mem_copy(ub_cos.next_write(), cbd.tile_slice(gm_cos, (token_tile_rows, VL), (token_tile_idx, 0)))
        kv = ub_kv.next_read()
        if cbd.const_expr(tiling.split_k > 1):
            self._acc_partial(kv_sum, kv, rows, head, True)
            for part in range(1, tiling.split_k):
                cbd.mem_copy(ub_kv.next_write(),
                             cbd.tile_slice(gm_ws_kv, (token_tile_rows, head),
                                            (part * tiling.m_tiles_a1 * tiling.bm_a1
                                             // token_tile_rows + token_tile_idx,
                                             0)))
                kv_part = ub_kv.next_read()
                self._acc_partial(kv_sum, kv_part, rows, head, False)
        sin_tab = ub_sin.next_read()
        cos_tab = ub_cos.next_read()
        for j in cbd.range(rows):
            kvb_slot = ub_kvb.next_write()
            if cbd.const_expr(tiling.split_k > 1):
                self._finish_kv_row(kv_sum, gamma_kv, sin_tab, cos_tab, kvb_slot, ub_tail, ub_kv_bits, j, j, norm_eps)
            else:
                self._finish_kv_row(kv, gamma_kv, sin_tab, cos_tab, kvb_slot, ub_tail, ub_kv_bits, j, j, norm_eps)
            slot = gm_index[token_tile_idx * token_tile_rows + j, 0]
            if slot >= 0:
                if slot < gm_cache.shape[0]:
                    cbd.mem_copy(cbd.tile_slice(gm_cache, (1, kv_row_bytes(head)), (slot, 0)), ub_kvb.next_read())

    @cbd.jit
    def _fill_w_b(self, l1_b, l1_sb, gm_wqb, gm_dswqb, wrow, kw, groups_l1):
        """加载一个 QB 权重窗口及对应 MX scale，布局和搬运规则与 _fill_w_a 对称。

        wrow/kw 选择输出权重行和 rank K 窗口，FP8 payload 按 NZ 预打包格式复制到 L1；scale 从
        (N, 组对, MX_PAIR) 转置展开为 (G, N)。本函数使用 B 阶段的 bn_b/k_l1_b，而 A1 版本使用
        bn_a1/k_l1_a1，两者保持相同数据语义但缓冲尺寸和任务坐标独立。
        """
        tiling = self.tiling
        output_tile_cols, k_l1 = tiling.bn_b, tiling.k_l1_b
        cbd.mem_copy(l1_b.next_write(), cbd.tile_slice(gm_wqb, (output_tile_cols, k_l1), (wrow, kw)))
        scale_tile = cbd.tile_slice(gm_dswqb, (output_tile_cols, groups_l1, MX_PAIR), (wrow, kw, 0))
        cbd.mem_copy(l1_sb.next_write(), scale_tile.permute(1, 2, 0).flatten(0, 1))

    @cbd.jit
    def _b_dst(self, gm_q, head_token_task_idx, subblock):
        """按 NonL1Reuse 的 head/token 布局计算当前 Vector 核的 GM Q 写回切片。"""
        tiling = self.tiling
        return cbd.tile_slice(gm_q, (tiling.bm_b, tiling.b_cols),
                              (head_token_task_idx % tiling.owners_b,
                               (head_token_task_idx // tiling.owners_b) * FIXPIPE_DUAL_DST + subblock))

    @cbd.jit
    def _finish_q_plain(self, ub_acc, ub_out, rows):
        """把不含 RoPE 特征的 Q FP32 切片直接转换为 BF16。

        按 rows 遍历有效 token，并在 b_cols 特征轴上按 VL 向量读取，使用各缓冲的物理行跨度寻址；
        _to_bf16 负责最近偶数舍入，vstore_pack 完成 FP32 到 16 位的打包。该快捷路径不加载位置表、
        不拼接尾部向量，用于驻留 head 中三个与 RoPE 无关的四分块。
        """
        feature_cols = self.tiling.b_cols
        src_pitch = ub_acc.physical_layout.stride[0]
        out_pitch = ub_out.physical_layout.stride[0]
        with cbd.vf(mode="simd"):
            # 1. 读取 Q 向量；2. 舍入并打包为 BF16。
            full = reg.full_mask()
            for token_row in cbd.range(rows):
                q_row_offset = token_row * src_pitch
                bf16_row_offset = token_row * out_pitch
                for feature_vector_idx in range(feature_cols // VL):
                    reg.vstore_pack(ub_out, bf16_row_offset + feature_vector_idx * VL,
                                   self._to_bf16(reg.vload(ub_acc, q_row_offset + feature_vector_idx * VL), full), full,
                                   pack_mode=reg.PackMode.B32_TO_B16)

    @cbd.jit
    def _finish_q(self, ub_acc, ub_sin, ub_cos, ub_out, ub_tail, rows, subblock, head_tile):
        """按 NonL1Reuse 布局完成 Q 前缀转换和 head 尾部 RoPE，并选择直接写回或 UB 拼接。"""
        tiling = self.tiling
        feature_cols, rope_width = tiling.b_cols, tiling.rope_width
        rope_width // 2
        # head 较窄时，最后一个特征向量的宽度可能小于 VL。
        tail = min(VL, feature_cols)
        src_pitch = ub_acc.physical_layout.stride[0]
        out_pitch = ub_out.physical_layout.stride[0]
        # interleave 输出由拥有 head 尾部的第二个 subblock 整段写出；另一个
        # subblock 只写前缀，避免两个 subblock 写同一段尾部。
        if cbd.const_expr(tiling.template == "split_k"
                          # 72：固定形状特判，此时 RoPE 尾部仅位于逻辑 head 后半区。
                          and tiling.token_count == 72):
            rope_lanes = (0 if head_tile % RESIDENT_HEAD_PANELS == 0 else (0 if subblock == 0 else rope_width))
        else:
            rope_lanes = 0 if subblock == 0 else rope_width
        with cbd.vf(mode="simd"):
            full = reg.full_mask()
            m_tail = reg.update_mask(tail, elem_bits=32)[0]
            m_rope = reg.update_mask(rope_lanes, elem_bits=32)[0]
            m_pair = reg.update_mask(rope_width // 2, elem_bits=32)[0]
            # 1. 将前缀特征转换为 BF16。
            for token_row in cbd.range(rows):
                q_row_offset = token_row * src_pitch
                bf16_row_offset = token_row * out_pitch
                for feature_vector_idx in range((feature_cols - tail) // VL):
                    reg.vstore_pack(ub_out, bf16_row_offset + feature_vector_idx * VL,
                                   self._to_bf16(reg.vload(ub_acc, q_row_offset + feature_vector_idx * VL), full), full,
                                   pack_mode=reg.PackMode.B32_TO_B16)
                # 直接写入两个半区不满足对齐时，先拼接完整尾部向量。
                # 第一个切片的 RoPE 掩码为空，保留原始值。
                # 2. 执行尾部 RoPE；3. 打包并对齐写出 BF16。
                if cbd.const_expr(rope_width % UB_ALIGN == 0):
                    # BF16 目的地址按 32 字节对齐，可直接写入互不重叠的前缀和
                    # 两个 RoPE 半区，避免经过 UB 暂存区。
                    prefix_mask = reg.update_mask(tail - rope_lanes, elem_bits=32)[0]
                    reg.vstore_pack(ub_out, bf16_row_offset + feature_cols - tail,
                                   self._to_bf16(reg.vload(ub_acc, q_row_offset + feature_cols - tail), prefix_mask),
                                   prefix_mask, pack_mode=reg.PackMode.B32_TO_B16)
                    rope_vec = self._rope_pair(ub_acc, q_row_offset + feature_cols - rope_width, None, 0, ub_sin,
                                               ub_cos, token_row, None, m_pair)
                    reg.vstore_pack(ub_out, bf16_row_offset + feature_cols - rope_width,
                                   self._to_bf16(rope_vec, m_rope), m_rope, pack_mode=reg.PackMode.B32_TO_B16)
                else:
                    reg.vstore(ub_tail, 0, reg.vload(ub_acc, q_row_offset + feature_cols - tail), m_tail)
                    rope_vec = self._rope_pair(ub_acc, q_row_offset + feature_cols - rope_width, None, 0, ub_sin,
                                               ub_cos, token_row, None, m_pair)
                    reg.vstore(ub_tail, tail - rope_width, rope_vec, m_rope)
                    reg.vmem_bar("vst_vld")
                    reg.vstore_pack(ub_out, bf16_row_offset + feature_cols - tail,
                                   self._to_bf16(reg.vload(ub_tail, 0), m_tail), m_tail,
                                   pack_mode=reg.PackMode.B32_TO_B16)

    @cbd.jit
    def _a1_branch(self, gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_wkv, gm_dswkv, gm_ws_qa, gm_ws_kv, gm_gamma_qr, gm_index,
                   gm_qr, gm_dqr2d, gm_qr_bf16, norm_eps):
        """执行分支流水的 A1/A2 前半段：先完成 QA，再让 KV 与 QR 后处理并行。

        所有核先按任务编号计算 QA 的 split-K FP32 部分和，并通过 _publish_a1 提前释放 QR 消费者；
        随后进入 _a2_branch，由 Cube 继续生成 KV 部分和，同时 Vector 归约、RMSNorm 并量化 QA。
        该顺序解除 QR 对 KV 投影完成的不必要依赖，是本路径区别于通用共享流水的核心。
        """
        scale_l0, scale_l1, groups_l1, l1_x, l1_sx, l1_w, l1_sw, l0a, l0b, l0c = (self._alloc_a1_buffers())
        tiling = self.tiling

        for qa_projection_task_idx in range(cbd.get_block_idx(), tiling.tiles_a1_qa, tiling.block_dim):
            self._a1_tile(l1_x, l1_sx, l1_w, l1_sw, l0a, l0b, l0c, gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_ws_qa,
                          qa_projection_task_idx, tiling.n_tiles_qa, scale_l0, groups_l1)
        self._publish_a1()
        self._a2_branch(gm_gamma_qr, gm_index, gm_qr, gm_dqr2d, gm_qr_bf16, gm_ws_qa, gm_ws_kv, norm_eps, l1_x, l1_sx,
                        l1_w, l1_sw, l0a, l0b, l0c, gm_x, gm_dsx, gm_wkv, gm_dswkv, scale_l0, groups_l1)

    @cbd.jit
    def _a2_branch(self, gm_gamma_qr, gm_index, gm_qr, gm_dqr2d, gm_qr_bf16, gm_ws_qa, gm_ws_kv, norm_eps, l1_x, l1_sx,
                   l1_w, l1_sw, l0a, l0b, l0c, gm_x, gm_dsx, gm_wkv, gm_dswkv, scale_l0, groups_l1):
        """在分支流水中并行推进 Cube KV 投影与 Vector QA 后处理。

        Cube 按 kv_slot 计算 KV FP32 部分和；Vector 同时按 token_tile_idx 调 _qr_tile，合并 QA
        split-K、生成 FP8/BF16 QR 和 E8M0 scale。两条角色路径互不等待对方逐块完成；循环结束后分别
        发布 KV 部分和与 QR，使后续 B 阶段可消费 QR，并使 KV 后处理能读取完整交换区。
        """
        tiling = self.tiling
        token_tile_rows, rank = tiling.bm_a2, tiling.rank_size
        a2_depth = (A2_BUF_DEPTH_PREFILL if tiling.template == "split_t" else A2_BUF_DEPTH_DECODE)
        ub_qa = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32,
                               cbd.make_layout((token_tile_rows, rank)), depth=a2_depth)
        ub_gamma_qr = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((1, rank)), depth=1)
        ub_qr = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.uint8,
                               cbd.make_layout((token_tile_rows, rank)), depth=a2_depth)
        ub_qr_bf16 = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.bfloat16,
                                    cbd.make_layout((token_tile_rows, rank)), depth=a2_depth)
        ub_exp = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.int32,
                            cbd.make_layout((token_tile_rows + 1, max(tiling.exp_stride, VL))))
        ub_scale = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.uint8,
                                  cbd.make_layout((token_tile_rows, tiling.scale_stride)), depth=a2_depth)
        cbd.mem_copy(ub_gamma_qr.next_write(), cbd.tile_slice(gm_gamma_qr, (1, rank), (0, 0)))
        gamma_qr = ub_gamma_qr.next_read()

        qa_sum = qa_comp = None
        if cbd.const_expr(tiling.split_k > 1):
            qa_sum = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((token_tile_rows, rank)))
            qa_comp = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((token_tile_rows, rank)))
        for kv_slot in range(cbd.get_block_idx(), tiling.tiles_a1_kv, tiling.block_dim):
            self._a1_tile(l1_x, l1_sx, l1_w, l1_sw, l0a, l0b, l0c, gm_x, gm_dsx, gm_wkv, gm_dswkv, gm_ws_kv, kv_slot,
                          tiling.n_tiles_kv, scale_l0, groups_l1)
        aiv = cbd.get_block_idx() * AIV_PER_BLOCK + cbd.get_subblock_id()
        tokens = gm_index.shape[0]
        actual_tiles = (tokens + token_tile_rows - 1) // token_tile_rows
        for token_tile_idx in range(aiv, actual_tiles, tiling.block_dim * AIV_PER_BLOCK):
            rows = cbd.tile_slice(gm_index, (token_tile_rows, 1), (token_tile_idx, 0)).shape[0]
            self._qr_tile(gm_ws_qa, gm_qr, gm_qr_bf16, gm_dqr2d, ub_qa, ub_qr, ub_qr_bf16, ub_exp, ub_scale, gamma_qr,
                          qa_sum, qa_comp, token_tile_idx, rows, norm_eps)

        self._publish_a1()
        self._publish_qr()

    @cbd.jit
    def _project_qa_kv(self, gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_wkv, gm_dswkv, gm_ws_qa, gm_ws_kv):
        """在通用 A1 阶段计算全部 QA/KV 投影，并输出独立 FP32 部分和。

        QA 与 KV 被拼成一个跨核任务队列，前段使用 WQA、后段使用 WKV；每个任务解析为 token、输出列
        和 split-K 分块后调用 _a1_tile。激活与权重的 L1/L0 流水由共享缓冲承载，本函数不做归约、
        RMSNorm 或量化，只保证 gm_ws_qa/gm_ws_kv 中的部分和完整写出。
        """
        scale_l0, scale_l1, groups_l1, l1_x, l1_sx, l1_w, l1_sw, l0a, l0b, l0c = (self._alloc_a1_buffers())
        tiling = self.tiling
        for projection_task_idx in range(cbd.get_block_idx(), tiling.tiles_a1_qa + tiling.tiles_a1_kv,
                                         tiling.block_dim):
            if projection_task_idx < tiling.tiles_a1_qa:
                self._a1_tile(l1_x, l1_sx, l1_w, l1_sw, l0a, l0b, l0c, gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_ws_qa,
                              projection_task_idx, tiling.n_tiles_qa, scale_l0, groups_l1)
            else:
                self._a1_tile(l1_x, l1_sx, l1_w, l1_sw, l0a, l0b, l0c, gm_x, gm_dsx, gm_wkv, gm_dswkv, gm_ws_kv,
                              projection_task_idx - tiling.tiles_a1_qa, tiling.n_tiles_kv, scale_l0, groups_l1)

    @cbd.jit
    def _reduce_normalize_qr(self, gm_gamma_qr, gm_index, gm_qr, gm_dqr2d, gm_ws_qa, norm_eps, gm_qr_bf16):
        """完成通用路径的 QR 归约量化；发布由调用方执行，KV 后处理由 _stage_b 穿插执行。"""
        tiling = self.tiling
        token_tile_rows, rank = tiling.bm_a2, tiling.rank_size
        a2_depth = (A2_BUF_DEPTH_PREFILL if tiling.template == "split_t" else A2_BUF_DEPTH_DECODE)
        ub_qa = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32,
                               cbd.make_layout((token_tile_rows, rank)), depth=a2_depth)
        ub_gamma_qr = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((1, rank)), depth=1)
        ub_qr = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.uint8,
                               cbd.make_layout((token_tile_rows, rank)), depth=a2_depth)
        ub_qr_bf16 = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.bfloat16,
                                    cbd.make_layout((token_tile_rows, rank)), depth=a2_depth)
        ub_exp = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.int32,
                            cbd.make_layout((token_tile_rows + 1, max(tiling.exp_stride, VL))))
        ub_scale = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.uint8,
                                  cbd.make_layout((token_tile_rows, tiling.scale_stride)), depth=a2_depth)

        cbd.mem_copy(ub_gamma_qr.next_write(), cbd.tile_slice(gm_gamma_qr, (1, rank), (0, 0)))
        gamma_qr = ub_gamma_qr.next_read()
        qa_sum = qa_comp = None
        if cbd.const_expr(tiling.split_k > 1):
            qa_sum = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((token_tile_rows, rank)))
            qa_comp = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((token_tile_rows, rank)))
        aiv = cbd.get_block_idx() * AIV_PER_BLOCK + cbd.get_subblock_id()
        tokens = gm_index.shape[0]
        actual_tiles = (tokens + token_tile_rows - 1) // token_tile_rows
        for token_tile_idx in range(aiv, actual_tiles, tiling.block_dim * AIV_PER_BLOCK):
            rows = cbd.tile_slice(gm_index, (token_tile_rows, 1), (token_tile_idx, 0)).shape[0]
            self._qr_tile(gm_ws_qa, gm_qr, gm_qr_bf16, gm_dqr2d, ub_qa, ub_qr, ub_qr_bf16, ub_exp, ub_scale, gamma_qr,
                          qa_sum, qa_comp, token_tile_idx, rows, norm_eps)


class BResidentKernelBase(NonL1ReuseKernelBase):
    """完整 QR token 分块驻留 L1，并跨 head 复用 QR、scale 和 RoPE 表。"""

    @cbd.jit
    def _prefetch_qb_weights(self, gm_wqb, gm_dswqb):
        """在 A 阶段后处理期间预取本核首个 QB 权重窗口及 scale。

        返回 depth-W_BUF 的 L1 payload Channel 和已预取首窗口的 scale Channel，供 _stage_b 直接消费，
        将权重搬运隐藏在 QR/KV 的 Vector 工作之后；payload 的其余槽位由 B 阶段按游标轮转填充。
        """
        weights = cbd.RingBuffer(cbd.MemLoc.L1, cbd.dtypes.int8,
                                 cbd.make_layout((self.tiling.bn_b, self.tiling.k_l1_b)), depth=W_BUF, data_format="nz")
        scales = cbd.RingBuffer(cbd.MemLoc.L1, FLOAT8_E8M0,
                                cbd.make_layout((self.tiling.k_l1_b // MX_GROUP, self.tiling.bn_b)), depth=2,
                                data_format="nz")
        self._fill_w_b(weights, scales, gm_wqb, gm_dswqb,
                       cbd.get_block_idx(), 0, self.tiling.k_l1_b // MX_PAIR_WIDTH)
        return weights, scales

    @cbd.jit
    def _stage_b(self, gm_qr, gm_dqr3d, gm_wqb, gm_dswqb, gm_sin, gm_cos, gm_q, prefetched_qb_weights,
                 prefetched_qb_scales, gm_gamma_kv, gm_index, gm_cache, gm_ws_kv, norm_eps):
        """复用驻留 QR 执行 QB 投影，并交错完成 Q 后处理和剩余 KV 后处理。

        每个 token 分块的完整 QR/scale 和 RoPE 表只加载一次，跨全部 head panel 复用；QB 权重采用
        双窗口轮转。每个 B 任务后插入一次 KV 后处理，最后按实际任务数补齐剩余 KV 分块。
        """
        tiling = self.tiling
        tokens = gm_index.shape[0]
        actual_tiles = (tokens + tiling.bm_a2 - 1) // tiling.bm_a2
        token_tile_rows, output_tile_cols, reduction_tile_cols, k_l1 = (tiling.bm_b, tiling.bn_b, tiling.bk_b,
                                                                        tiling.k_l1_b)
        rank, head = tiling.rank_size, tiling.head_size
        panels_per_head = head // output_tile_cols
        scale_l0 = reduction_tile_cols // MX_GROUP
        groups_l1 = k_l1 // MX_PAIR_WIDTH

        # 每个 token 分块对应一次 Channel 轮转，所有 head 任务共用该视图。
        l1_a = cbd.RingBuffer(cbd.MemLoc.L1, FLOAT8_E4M3FN,
                              cbd.make_layout((token_tile_rows, rank)), depth=1, data_format="nz")
        # E8M0 scale 以 uint8 原始字节承载，选槽时通过 from_tensor 创建 zn 视图。
        l1_sa = cbd.RingBuffer(cbd.MemLoc.L1, cbd.dtypes.uint8,
                               cbd.make_layout((ceil_div(token_tile_rows, NZ_M_FRAC)
                                                * ceil_div(rank // MX_GROUP, MX_PAIR) * ZN_FRACTAL_BYTES,)),
                               depth=1, data_format="nd")
        l1_b = prefetched_qb_weights
        l1_sb = prefetched_qb_scales
        l0a = cbd.RingBuffer(cbd.MemLoc.L0A, FLOAT8_E4M3FN,
                             cbd.make_layout((token_tile_rows, reduction_tile_cols)), depth=L0_DEPTH)
        l0b = cbd.RingBuffer(cbd.MemLoc.L0B, FLOAT8_E4M3FN,
                             cbd.make_layout((output_tile_cols, reduction_tile_cols)),
                             depth=1 if tiling.bn_b == BN_B_FULL_HEAD else L0_DEPTH)
        l0c = cbd.RingBuffer(cbd.MemLoc.L0C, cbd.dtypes.float32,
                             cbd.make_layout((token_tile_rows, output_tile_cols)), depth=1)
        feature_cols = tiling.b_cols
        ub_acc = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32,
                                cbd.make_layout((token_tile_rows, feature_cols)), depth=1)
        ub_out = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.bfloat16,
                                cbd.make_layout((token_tile_rows, feature_cols)), depth=1)
        ub_tail = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((1, VL)))
        ub_sin = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((token_tile_rows, VL)), depth=1)
        ub_cos = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((token_tile_rows, VL)), depth=1)

        # 各核完成当前 token 分块的所有 head 任务后再前进，每块只加载一次 QR 和 RoPE 表。
        subblock = cbd.get_subblock_id()
        for owner in range(tiling.owners_b):
            cbd.mem_copy(l1_a.next_write(), cbd.tile_slice(gm_qr, (token_tile_rows, rank), (owner, 0)))
            cbd.mem_copy(cbd.from_tensor(l1_sa.next_write(), dtype=FLOAT8_E8M0,
                                         layout=cbd.make_layout((token_tile_rows, rank // MX_GROUP)),
                                         data_format="zn"),
                         cbd.tile_slice(gm_dqr3d, (token_tile_rows, rank // MX_PAIR_WIDTH, MX_PAIR),
                                        (owner, 0, 0)).flatten(1, 2))
            a_ready = l1_a.next_read()
            sa_ready = cbd.from_tensor(l1_sa.next_read(), dtype=FLOAT8_E8M0,
                                       layout=cbd.make_layout((token_tile_rows, rank // MX_GROUP)), data_format="zn")
            cbd.mem_copy(ub_sin.next_write(), cbd.tile_slice(gm_sin, (token_tile_rows, VL), (owner, 0)))
            cbd.mem_copy(ub_cos.next_write(), cbd.tile_slice(gm_cos, (token_tile_rows, VL), (owner, 0)))
            sin_tab = ub_sin.next_read()
            cos_tab = ub_cos.next_read()
            for head_tile in range(cbd.get_block_idx(), tiling.head_count * panels_per_head, tiling.block_dim):
                head_token_task_idx = head_tile * tiling.owners_b + owner
                self._b_cube(a_ready, sa_ready, l1_b, l1_sb, l0a, l0b, l0c, ub_acc, gm_wqb, gm_dswqb,
                             head_token_task_idx, scale_l0, groups_l1)
                # owners_b 大于 1 时，本地迭代号不能直接用任务编号除以核数计算。
                b_step = owner * ceil_div(tiling.head_count * panels_per_head - cbd.get_block_idx(), tiling.block_dim)
                b_step += head_tile // tiling.block_dim
                self._service_kv(gm_gamma_kv, gm_sin, gm_cos, gm_index, gm_cache, gm_ws_kv, norm_eps,
                                 cbd.get_block_idx() * AIV_PER_BLOCK + subblock
                                 + b_step * (tiling.block_dim * AIV_PER_BLOCK))
                self._b_vec_reuse(ub_acc, ub_out, ub_tail, sin_tab, cos_tab, gm_q, head_token_task_idx, subblock)

        if cbd.const_expr(tiling.tiles_a2 > tiling.min_b_iterations * (tiling.block_dim * AIV_PER_BLOCK)):
            b_count = tiling.owners_b * ceil_div(tiling.head_count * panels_per_head - cbd.get_block_idx(),
                                                 tiling.block_dim)
            for kv_slot in range(cbd.get_block_idx() * AIV_PER_BLOCK + subblock
                                 + b_count * (tiling.block_dim * AIV_PER_BLOCK), actual_tiles,
                                 tiling.block_dim * AIV_PER_BLOCK):
                self._service_kv(gm_gamma_kv, gm_sin, gm_cos, gm_index, gm_cache, gm_ws_kv, norm_eps, kv_slot)

    @cbd.jit
    def _b_cube(self, l1_a, l1_sa, l1_b, l1_sb, l0a, l0b, l0c, ub_acc, gm_wqb, gm_dswqb, head_token_task_idx,
                scale_l0, groups_l1):
        """使用双窗口 QB 权重和驻留 QR 完成一个 head/token 任务的 MXFP8 投影。

        任务编号解析为 token owner 和权重行；计算当前窗口时预取后续窗口，全部 K 块累加到 L0C。
        最后通过 FixPipe split-N 将 FP32 结果分发给两个 Vector 核，不执行 RoPE 或 BF16 转换。
        """
        tiling = self.tiling
        owner = head_token_task_idx % tiling.owners_b
        wrow = head_token_task_idx // tiling.owners_b
        n_kw = tiling.rank_size // tiling.k_l1_b
        accumulator = l0c.next_write()

        # 当前权重窗口驻留在 L1 时，由 MTE2 加载下一窗口；首个窗口已在 A2 期间预取。
        if head_token_task_idx != cbd.get_block_idx() * tiling.owners_b:
            self._fill_w_b(l1_b, l1_sb, gm_wqb, gm_dswqb, wrow, 0, groups_l1)
        if cbd.const_expr(n_kw > 1):
            self._fill_w_b(l1_b, l1_sb, gm_wqb, gm_dswqb, wrow, 1, groups_l1)
        for kwp in range(n_kw // W_BUF):
            kw0 = kwp * W_BUF
            self._mm_window_b(l1_a, l1_sa, l1_b, l1_sb, l0a, l0b, accumulator, owner, kw0, scale_l0)
            if kw0 + W_BUF < n_kw:
                self._fill_w_b(l1_b, l1_sb, gm_wqb, gm_dswqb, wrow, kw0 + W_BUF, groups_l1)
            self._mm_window_b(l1_a, l1_sa, l1_b, l1_sb, l0a, l0b, accumulator, owner, kw0 + 1, scale_l0)
            if kw0 + W_BUF + 1 < n_kw:
                self._fill_w_b(l1_b, l1_sb, gm_wqb, gm_dswqb, wrow, kw0 + W_BUF + 1, groups_l1)
        if cbd.const_expr(n_kw % W_BUF != 0):
            self._mm_window_b(l1_a, l1_sa, l1_b, l1_sb, l0a, l0b, accumulator, owner, n_kw - 1, scale_l0)

        cbd.mem_copy(ub_acc.next_write(), accumulator.partition(dim=1), cbd.copy_options(unit_flag=FIXPIPE_FINAL))

    @cbd.jit
    def _mm_window_b(self, l1_a, l1_sa, l1_b, l1_sb, l0a, l0b, l0c, owner, kw, scale_l0):
        """从完整驻留 QR 中切出 K 块，完成一个 L1 权重窗口内的全部 L0 matmul。

        QR payload/scale 直接从驻留 L1 视图切片到 L0A，权重及 scale 按 kl0 选择 L0B 子块。
        首 K 块初始化 L0C，中间块累加，末块设置 FIXPIPE_FINAL。
        """
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols, k_l1 = (tiling.bm_b, tiling.bn_b, tiling.bk_b,
                                                                        tiling.k_l1_b)
        k_tiles = tiling.rank_size // reduction_tile_cols
        w_ready = cbd.from_tensor(l1_b.next_read(), dtype=FLOAT8_E4M3FN)
        s_ready = l1_sb.next_read()
        for kl0 in range(k_l1 // reduction_tile_cols):
            kb = kw * (k_l1 // reduction_tile_cols) + kl0
            activation_scale_tile = cbd.tile_slice(l1_sa, (token_tile_rows, scale_l0), (0, kb))
            cbd.mem_copy(l0a.next_write(),
                         cbd.tile_slice(l1_a, (token_tile_rows, reduction_tile_cols), (0, kb)),
                         cbd.copy_options(mx_scale=activation_scale_tile))
            if cbd.const_expr(k_l1 == reduction_tile_cols):
                cbd.mem_copy(l0b.next_write(), w_ready, cbd.copy_options(mx_scale=s_ready))
            else:
                weight_scale_tile = cbd.tile_slice(s_ready, (scale_l0, output_tile_cols), (kl0, 0))
                cbd.mem_copy(l0b.next_write(),
                             cbd.tile_slice(w_ready, (output_tile_cols, reduction_tile_cols), (0, kl0)),
                             cbd.copy_options(mx_scale=weight_scale_tile))
            cbd.matmul(l0c, l0a.next_read(), l0b.next_read(), init=(kb == 0),
                       unit_flag=FIXPIPE_FINAL if kb + 1 == k_tiles else FIXPIPE_ACCUMULATE, mx_mode=True)

    @cbd.jit
    def _b_vec_reuse(self, ub_acc, ub_out, ub_tail, sin_tab, cos_tab, gm_q, head_token_task_idx, subblock):
        """复用驻留 token 分块的 sin/cos 表，完成一个 QB panel 的 BF16 Q 写回。

        只有 head 后半个 panel 且 subblock == 1 的切片包含逻辑 RoPE 尾部，调用 _finish_q；
        其余三个四分块直接调用 _finish_q_plain。目的写回按有效行列裁剪，避免访问补齐区域。
        """
        tiling = self.tiling
        q_half = self._b_dst(gm_q, head_token_task_idx, subblock)
        rows = q_half.shape[0]
        q_accumulator = ub_acc.next_read()
        out_slot = ub_out.next_write()
        head_tile = head_token_task_idx // tiling.owners_b
        # 只有 head 后半个 panel 且 subblock == 1 的切片包含逻辑 RoPE 尾部。
        if head_tile % RESIDENT_HEAD_PANELS == 1 and subblock == 1:
            self._finish_q(q_accumulator, sin_tab, cos_tab, out_slot, ub_tail, rows, subblock, head_tile)
        else:
            self._finish_q_plain(q_accumulator, out_slot, rows)
        cbd.mem_copy(q_half, ub_out.next_read()[: q_half.shape[0], : q_half.shape[1]])


class BStreamingKernelBase(NonL1ReuseKernelBase):
    """沿 K 轴双缓冲搬运 QR，并为每个 B 任务加载 RoPE 表。"""

    @cbd.jit
    def _stage_b(self, gm_qr, gm_dqr3d, gm_wqb, gm_dswqb, gm_sin, gm_cos, gm_q, gm_gamma_kv, gm_index, gm_cache,
                 gm_ws_kv, norm_eps):
        """以 K 轴双缓冲执行 QB 投影，并交错完成 Q 后处理和剩余 KV 后处理。

        QR/scale 按 L0 K 块从 GM 搬到 L1，QB 权重使用 W_BUF 个窗口预取；每个 head/token 任务
        独立加载 RoPE 表。B 任务之间穿插 KV 后处理，末尾按实际任务数补齐剩余 KV 分块。
        """
        tiling = self.tiling
        tokens = gm_index.shape[0]
        actual_tiles = (tokens + tiling.bm_a2 - 1) // tiling.bm_a2
        token_tile_rows, output_tile_cols, reduction_tile_cols, k_l1 = (tiling.bm_b, tiling.bn_b, tiling.bk_b,
                                                                        tiling.k_l1_b)
        _rank, head = tiling.rank_size, tiling.head_size
        scale_l1 = k_l1 // MX_GROUP
        scale_l0 = reduction_tile_cols // MX_GROUP
        groups_l1 = k_l1 // MX_PAIR_WIDTH

        l1_a = cbd.RingBuffer(cbd.MemLoc.L1, FLOAT8_E4M3FN,
                              cbd.make_layout((token_tile_rows, reduction_tile_cols)), depth=2, data_format="nz")
        l1_sa = cbd.RingBuffer(cbd.MemLoc.L1, cbd.dtypes.uint8,
                               cbd.make_layout((ceil_div(token_tile_rows, NZ_M_FRAC)
                                                * ceil_div(scale_l0, MX_PAIR) * ZN_FRACTAL_BYTES,)),
                               depth=2, data_format="nd")
        l1_b = cbd.RingBuffer(cbd.MemLoc.L1, cbd.dtypes.int8,
                              cbd.make_layout((output_tile_cols, k_l1)), depth=W_BUF, data_format="nz")
        l1_sb = cbd.RingBuffer(cbd.MemLoc.L1, FLOAT8_E8M0,
                               cbd.make_layout((scale_l1, output_tile_cols)), depth=W_BUF, data_format="nz")
        l0a = cbd.RingBuffer(cbd.MemLoc.L0A, FLOAT8_E4M3FN,
                             cbd.make_layout((token_tile_rows, reduction_tile_cols)), depth=L0_DEPTH)
        l0b = cbd.RingBuffer(cbd.MemLoc.L0B, FLOAT8_E4M3FN,
                             cbd.make_layout((output_tile_cols, reduction_tile_cols)),
                             depth=1 if tiling.bn_b == BN_B_FULL_HEAD else L0_DEPTH)
        l0c = cbd.RingBuffer(cbd.MemLoc.L0C, cbd.dtypes.float32,
                             cbd.make_layout((token_tile_rows, head)), depth=1)
        feature_cols = tiling.b_cols
        ub_acc = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32,
                                cbd.make_layout((token_tile_rows, feature_cols)), depth=1)
        ub_out = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.bfloat16,
                                cbd.make_layout((token_tile_rows, feature_cols)), depth=1)
        ub_tail = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((1, VL)))
        ub_sin = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((token_tile_rows, VL)), depth=1)
        ub_cos = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((token_tile_rows, VL)), depth=1)

        subblock = cbd.get_subblock_id()
        for head_token_task_idx in range(cbd.get_block_idx(), tiling.tiles_b, tiling.block_dim):
            self._b_cube(l1_a, l1_sa, l1_b, l1_sb, l0a, l0b, l0c, ub_acc, gm_qr, gm_dqr3d, gm_wqb, gm_dswqb,
                         head_token_task_idx, scale_l0, groups_l1)
            self._service_kv(gm_gamma_kv, gm_sin, gm_cos, gm_index, gm_cache, gm_ws_kv, norm_eps,
                             cbd.get_block_idx() * AIV_PER_BLOCK + subblock
                             + (head_token_task_idx // tiling.block_dim) * (tiling.block_dim * AIV_PER_BLOCK))
            self._b_vec(ub_acc, ub_out, ub_tail, ub_sin, ub_cos, gm_sin, gm_cos, gm_q, head_token_task_idx, subblock)

        if cbd.const_expr(tiling.tiles_a2 > tiling.min_b_iterations * (tiling.block_dim * AIV_PER_BLOCK)):
            b_count = ceil_div(tiling.tiles_b - cbd.get_block_idx(), tiling.block_dim)
            for kv_slot in range(cbd.get_block_idx() * AIV_PER_BLOCK + subblock
                                 + b_count * (tiling.block_dim * AIV_PER_BLOCK), actual_tiles,
                                 tiling.block_dim * AIV_PER_BLOCK):
                self._service_kv(gm_gamma_kv, gm_sin, gm_cos, gm_index, gm_cache, gm_ws_kv, norm_eps, kv_slot)

    @cbd.jit
    def _b_cube(self, l1_a, l1_sa, l1_b, l1_sb, l0a, l0b, l0c, ub_acc, gm_qr, gm_dqr3d, gm_wqb, gm_dswqb,
                head_token_task_idx, scale_l0, groups_l1):
        """按 W_BUF 成对预取或顺序回退，完成一个 head/token 任务的 QB MXFP8 投影。

        每个任务遍历 head 的输出列块，沿 rank 加载 QB 权重/scale 并累加到完整 head 的 L0C。
        全部 K/N 计算后通过 FixPipe split-N 发布 FP32 结果，不在本函数中执行 RoPE 或 BF16 转换。
        """
        tiling = self.tiling
        output_tile_cols, k_l1 = tiling.bn_b, tiling.k_l1_b
        rank, head = tiling.rank_size, tiling.head_size
        n_tiles = head // output_tile_cols
        owner = head_token_task_idx % tiling.owners_b
        head_tile_idx = head_token_task_idx // tiling.owners_b
        n_kw = rank // k_l1
        accumulator = l0c.next_write()
        for n in range(n_tiles):
            wrow = head_tile_idx * n_tiles + n
            if cbd.const_expr(n_kw % W_BUF == 0):
                for kwp in range(n_kw // W_BUF):
                    kw0 = kwp * W_BUF
                    for u in cbd.range_constexpr(W_BUF):
                        self._fill_w_b(l1_b, l1_sb, gm_wqb, gm_dswqb, wrow, kw0 + u, groups_l1)
                    for u in cbd.range_constexpr(W_BUF):
                        self._mm_window_b(l1_a, l1_sa, l1_b, l1_sb, l0a, l0b, accumulator, gm_qr, gm_dqr3d,
                                          owner, n, kw0 + u, scale_l0)
            else:
                for kw in range(n_kw):
                    self._fill_w_b(l1_b, l1_sb, gm_wqb, gm_dswqb, wrow, kw, groups_l1)
                    self._mm_window_b(l1_a, l1_sa, l1_b, l1_sb, l0a, l0b, accumulator, gm_qr, gm_dqr3d, owner, n,
                                      kw, scale_l0)

        cbd.mem_copy(ub_acc.next_write(), accumulator.partition(dim=1), cbd.copy_options(unit_flag=FIXPIPE_FINAL))

    @cbd.jit
    def _mm_window_b(self, l1_a, l1_sa, l1_b, l1_sb, l0a, l0b, l0c, gm_qr, gm_dqr3d, owner, n, kw, scale_l0):
        """逐块从 GM 加载 QR/scale，完成一个 L1 权重窗口内的全部 L0 matmul。

        每个 K 块经 L1 双缓冲送入 L0A，权重及 scale 按 kl0 选择 L0B 子块；n 选择完整 head
        L0C 的输出列切片。首 K 块初始化累加器，中间块累加，末块设置 FIXPIPE_FINAL。
        """
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols, k_l1 = (tiling.bm_b, tiling.bn_b, tiling.bk_b,
                                                                        tiling.k_l1_b)
        k_tiles = tiling.rank_size // reduction_tile_cols
        w_ready = cbd.from_tensor(l1_b.next_read(), dtype=FLOAT8_E4M3FN)
        s_ready = l1_sb.next_read()
        for kl0 in range(k_l1 // reduction_tile_cols):
            kb = kw * (k_l1 // reduction_tile_cols) + kl0
            cbd.mem_copy(l1_a.next_write(),
                         cbd.tile_slice(gm_qr, (token_tile_rows, reduction_tile_cols), (owner, kb)))
            cbd.mem_copy(cbd.from_tensor(l1_sa.next_write(), dtype=FLOAT8_E8M0,
                                         layout=cbd.make_layout((token_tile_rows, scale_l0)), data_format="zn"),
                         cbd.tile_slice(gm_dqr3d, (token_tile_rows, reduction_tile_cols // MX_PAIR_WIDTH, MX_PAIR),
                                        (owner, kb, 0)).flatten(1, 2))
            cbd.mem_copy(l0a.next_write(),
                         l1_a.next_read(),
                         cbd.copy_options(mx_scale=cbd.from_tensor(l1_sa.next_read(), dtype=FLOAT8_E8M0,
                                                                   layout=cbd.make_layout((token_tile_rows, scale_l0)),
                                                                   data_format="zn")))
            if cbd.const_expr(k_l1 == reduction_tile_cols):
                cbd.mem_copy(l0b.next_write(), w_ready, cbd.copy_options(mx_scale=s_ready))
            else:
                weight_scale_tile = cbd.tile_slice(s_ready, (scale_l0, output_tile_cols), (kl0, 0))
                cbd.mem_copy(l0b.next_write(),
                             cbd.tile_slice(w_ready, (output_tile_cols, reduction_tile_cols), (0, kl0)),
                             cbd.copy_options(mx_scale=weight_scale_tile))
            cbd.matmul(cbd.tile_slice(l0c, (token_tile_rows, output_tile_cols), (0, n)),
                       l0a.next_read(), l0b.next_read(), init=(kb == 0),
                       unit_flag=FIXPIPE_FINAL if kb + 1 == k_tiles else FIXPIPE_ACCUMULATE, mx_mode=True)

    @cbd.jit
    def _b_vec(self, ub_acc, ub_out, ub_tail, ub_sin, ub_cos, gm_sin, gm_cos, gm_q, head_token_task_idx, subblock):
        """加载当前 token 的 RoPE 表，完成 Streaming 路径的 Q 后处理和 BF16 写回。

        两个 Vector 半区复用当前 owner 的 sin/cos 表，_finish_q 完成 FP32 前缀转换和尾部 RoPE；
        写回 GM 前按目的实际域裁剪 UB 源，避免尾块访问补齐行。
        """
        tiling = self.tiling
        token_tile_rows = tiling.bm_b
        owner = head_token_task_idx % tiling.owners_b
        q_half = self._b_dst(gm_q, head_token_task_idx, subblock)
        rows = q_half.shape[0]
        cbd.mem_copy(ub_sin.next_write(), cbd.tile_slice(gm_sin, (token_tile_rows, VL), (owner, 0)))
        cbd.mem_copy(ub_cos.next_write(), cbd.tile_slice(gm_cos, (token_tile_rows, VL), (owner, 0)))
        q_accumulator = ub_acc.next_read()
        sin_tab = ub_sin.next_read()
        cos_tab = ub_cos.next_read()
        out_slot = ub_out.next_write()
        self._finish_q(q_accumulator, sin_tab, cos_tab, out_slot, ub_tail, rows, subblock,
                       head_token_task_idx // tiling.owners_b)
        # 尾块按目的实际域裁剪 UB 源，避免越界写满物理行。
        cbd.mem_copy(q_half, ub_out.next_read()[: q_half.shape[0], : q_half.shape[1]])


@cbd.kernel
class BranchBResidentAttnPrologueKernel(BResidentKernelBase):
    """提前发布 QA，并使用 QR 驻留的 B 阶段流水。"""

    def __call__(self, gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_wqb, gm_dswqb, gm_wkv, gm_dswkv, gm_gamma_qr, gm_gamma_kv,
                 gm_sin, gm_cos, gm_index, gm_cache, gm_q, gm_qr, gm_dqr2d, gm_dqr3d, gm_ws_qa, gm_ws_kv,
                 norm_eps: cbd.dtypes.float32, gm_qr_bf16):
        """执行 QA 提前发布、QR 驻留的 Branch 流水。

        先完成并发布 QA，再并行推进 KV 投影与 QR 后处理；随后预取首个 QB 权重窗口。
        B 阶段复用完整 QR/scale 和 RoPE 表，跨 head 执行 QB，并交错完成 Q 写回和 KV 后处理。
        """
        self._a1_branch(gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_wkv, gm_dswkv, gm_ws_qa, gm_ws_kv, gm_gamma_qr, gm_index,
                        gm_qr, gm_dqr2d, gm_qr_bf16, norm_eps)
        prefetched_qb_weights, prefetched_qb_scales = self._prefetch_qb_weights(gm_wqb, gm_dswqb)
        self._stage_b(gm_qr, gm_dqr3d, gm_wqb, gm_dswqb, gm_sin, gm_cos, gm_q, prefetched_qb_weights,
                      prefetched_qb_scales, gm_gamma_kv, gm_index, gm_cache, gm_ws_kv, norm_eps)


@cbd.kernel
class GenericBResidentAttnPrologueKernel(BResidentKernelBase):
    """使用通用 A 阶段和 QR 驻留的 B 阶段流水。"""

    def __call__(self, gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_wqb, gm_dswqb, gm_wkv, gm_dswkv, gm_gamma_qr, gm_gamma_kv,
                 gm_sin, gm_cos, gm_index, gm_cache, gm_q, gm_qr, gm_dqr2d, gm_dqr3d, gm_ws_qa, gm_ws_kv,
                 norm_eps: cbd.dtypes.float32, gm_qr_bf16):
        """执行通用 A 阶段和 QR 驻留的 B 阶段流水。

        先通过共享任务队列完成 QA/KV 投影并发布工作区，再预取首个 QB 权重窗口，归约量化并发布 QR。
        B 阶段复用完整 QR/scale 和 RoPE 表，跨 head 执行 QB，并交错完成 Q 写回和 KV 后处理。
        """
        self._project_qa_kv(gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_wkv, gm_dswkv, gm_ws_qa, gm_ws_kv)
        self._publish_a1()
        prefetched_qb_weights, prefetched_qb_scales = self._prefetch_qb_weights(gm_wqb, gm_dswqb)
        self._reduce_normalize_qr(gm_gamma_qr, gm_index, gm_qr, gm_dqr2d, gm_ws_qa, norm_eps, gm_qr_bf16)
        self._publish_qr()
        self._stage_b(gm_qr, gm_dqr3d, gm_wqb, gm_dswqb, gm_sin, gm_cos, gm_q, prefetched_qb_weights,
                      prefetched_qb_scales, gm_gamma_kv, gm_index, gm_cache, gm_ws_kv, norm_eps)


@cbd.kernel
class BranchBStreamingAttnPrologueKernel(BStreamingKernelBase):
    """提前发布 QA，并使用 K 轴双缓冲的 B 阶段流水。"""

    def __call__(self, gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_wqb, gm_dswqb, gm_wkv, gm_dswkv, gm_gamma_qr, gm_gamma_kv,
                 gm_sin, gm_cos, gm_index, gm_cache, gm_q, gm_qr, gm_dqr2d, gm_dqr3d, gm_ws_qa, gm_ws_kv,
                 norm_eps: cbd.dtypes.float32, gm_qr_bf16):
        """执行 QA 提前发布、K 轴双缓冲的 Branch 流水。

        先完成并发布 QA，再并行推进 KV 投影与 QR 后处理；B 阶段沿 K 轴双缓冲加载 QR/scale，
        逐任务加载 RoPE 表，并交错完成 QB 投影、Q 写回和 KV 后处理。
        """
        self._a1_branch(gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_wkv, gm_dswkv, gm_ws_qa, gm_ws_kv, gm_gamma_qr, gm_index,
                        gm_qr, gm_dqr2d, gm_qr_bf16, norm_eps)
        self._stage_b(gm_qr, gm_dqr3d, gm_wqb, gm_dswqb, gm_sin, gm_cos, gm_q, gm_gamma_kv, gm_index, gm_cache,
                      gm_ws_kv, norm_eps)


@cbd.kernel
class GenericBStreamingAttnPrologueKernel(BStreamingKernelBase):
    """使用通用 A 阶段和 K 轴双缓冲的 B 阶段流水。"""

    def __call__(self, gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_wqb, gm_dswqb, gm_wkv, gm_dswkv, gm_gamma_qr, gm_gamma_kv,
                 gm_sin, gm_cos, gm_index, gm_cache, gm_q, gm_qr, gm_dqr2d, gm_dqr3d, gm_ws_qa, gm_ws_kv,
                 norm_eps: cbd.dtypes.float32, gm_qr_bf16):
        """执行通用 A 阶段和 K 轴双缓冲的 B 阶段流水。

        先通过共享任务队列完成 QA/KV 投影并发布工作区，再归约量化并发布 QR；B 阶段沿 K 轴
        双缓冲加载 QR/scale，逐任务加载 RoPE 表，并交错完成 QB 投影、Q 写回和 KV 后处理。
        """
        self._project_qa_kv(gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_wkv, gm_dswkv, gm_ws_qa, gm_ws_kv)
        self._publish_a1()
        self._reduce_normalize_qr(gm_gamma_qr, gm_index, gm_qr, gm_dqr2d, gm_ws_qa, norm_eps, gm_qr_bf16)
        self._publish_qr()
        self._stage_b(gm_qr, gm_dqr3d, gm_wqb, gm_dswqb, gm_sin, gm_cos, gm_q, gm_gamma_kv, gm_index, gm_cache,
                      gm_ws_kv, norm_eps)
