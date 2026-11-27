# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""三条 AttnPrologue 流水共用的路径无关实现。"""

from __future__ import annotations

# fmt: off

import cannbotdsl as cbd
import cannbotdsl.reg as reg


from .attn_prologue_tiling import (AttnPrologueTiling, E4M3_MAX, E8M0_BIAS, FP32_EXP_BIAS, FP32_EXP_SHIFT, MX_GROUP,
                                   MX_PAIR, VL, ceil_div)


class AttnPrologueCommon:
    """保存静态 tiling，并提供三条流水共用的归约、量化、RoPE 和同步原语。"""

    def __init__(self, tiling: AttnPrologueTiling):
        """保存一次编译特化所需的完整 tiling 配置。

        tiling 决定各阶段的分块尺寸、任务划分、缓冲深度和流水路径；kernel 内所有方法共享该对象，
        不在设备侧重新推导 host 已确定的静态参数。
        """
        self.tiling = tiling

    @cbd.jit
    def _acc_all_partials(self, dst, src, width):
        """在 L1 驻留路径中一次读入并累加全部 split-K 部分和。

        src 按 split 维连续存放，每个 split 的物理行跨度由 src.physical_layout.stride[0] 给出。
        本函数以 VL 宽度沿 width 特征轴向量加载，保持固定的从左到右加法顺序，将 FP32 总和写入 dst；
        只处理当前单行 token，不执行 RMSNorm、量化或尾行裁剪。
        """
        with cbd.vf(mode="simd"):
            # 1. 读取各特征向量，并累加 split-K 部分和。
            mask = reg.full_mask()
            for feature_vector_idx in range(width // VL):
                value = reg.vload(src, feature_vector_idx * VL)
                for part in range(1, self.tiling.split_k):
                    value = reg.vadd(value,
                                    reg.vload(src, part * src.physical_layout.stride[0] + feature_vector_idx * VL),
                                    mask=mask)
                reg.vstore(dst, feature_vector_idx * VL, value, mask)

    @cbd.jit
    def _publish_qr(self):
        """发布 AIV 已写出的全部 QR payload 和 scale，允许配对 AIC 开始 QB。

        Vector 到 MTE3 的片上依赖由框架处理，MTE3 通知保证 QR 的 DMA 写出完成后再进行跨核同步。
        发布点位于全部 QR 行之后而非逐 token 发布，因此 QB 不会读取尚未生成的 scale；AIV 返回后
        可继续处理与 QR 无关的 KV，实现 KV 后处理与 QB Cube 重叠。
        """
        # 所有 QR/scale 行均由 AIV 产生：完成 DMA 和 AIV 全核汇合后，
        # 释放配对 AIC；AIV 随后可继续处理 KV。
        cbd.vec_sync_block_arrive(cbd.PIPE.MTE3, 0, mode=0)
        cbd.vec_sync_block_wait(cbd.PIPE.MTE3, 0, mode=0)
        cbd.vec_sync_block_arrive(cbd.PIPE.MTE3, 3, mode=2)
        cbd.cube_sync_block_wait(cbd.PIPE.MTE2, 3, mode=2)

    @cbd.jit
    def _fill_w_a(self, l1_b, l1_sb, gm_b, gm_dsb, n, kw, groups_l1):
        """将一个 A1 权重 K 窗口及 MX scale 装入 L1，并转换为硬件需要的布局。

        FP8 权重 payload 按预打包 NZ 窗口直接搬运；GM scale 原始形状为 (N, K 组对, MX_PAIR)，
        通过 permute(1, 2, 0) 和 flatten 转为 matmul 需要的 (G, N) 视图。n/kw 选择输出列块和
        K 窗口，groups_l1 给出 scale 的组对宽度；函数只负责装载，不推进 L0C 累加。
        """
        tiling = self.tiling
        output_tile_cols, k_l1 = tiling.bn_a1, tiling.k_l1_a1
        w_slot = l1_b.next_write()
        cbd.mem_copy(w_slot, cbd.tile_slice(gm_b, (output_tile_cols, k_l1), (n, kw)))
        s_slot = l1_sb.next_write()
        scale_tile = cbd.tile_slice(gm_dsb, (output_tile_cols, groups_l1, MX_PAIR), (n, kw, 0))
        cbd.mem_copy(s_slot, scale_tile.permute(1, 2, 0).flatten(0, 1))

    @cbd.jit
    def _acc_partial(self, dst, src, rows, width, first):
        """按普通 FP32 加法合并一份通用路径的 split-K 部分和。

        first=True 时以 src 初始化 dst，否则将 src 加到已有 dst；按 rows 和 VL 特征向量逐块处理，
        使用各张量真实物理行跨度寻址。该函数供 KV 等不要求补偿误差的归约使用，只更新内存中的
        部分和，不归一化、不量化，也不读取无效尾行。
        """
        with cbd.vf(mode="simd"):
            # 1. 读取各特征向量，并累加 split-K 部分和。
            mask = reg.full_mask()
            for token_row in cbd.range(rows):
                for feature_vector_idx in range(width // VL):
                    dst_feature_offset = (token_row * dst.physical_layout.stride[0] + feature_vector_idx * VL)
                    partial_values = reg.vload(src, token_row * src.physical_layout.stride[0] + feature_vector_idx * VL)
                    if cbd.const_expr(not first):
                        partial_values = reg.vadd(reg.vload(dst, dst_feature_offset), partial_values, mask=mask)
                    reg.vstore(dst, dst_feature_offset, partial_values, mask)

    @cbd.jit
    def _acc_partial_qa(self, dst, compensation, src, rows, width, first):
        """以补偿求和合并一份 QA split-K 部分和，降低 FP32 累加舍入误差。

        首次调用保存部分和并将 compensation 清零；后续调用先扣除历史低位残差，再把修正值加入
        当前总和，并反推出本次加法丢失的低位写回 compensation。处理范围为 rows × width，
        dst/compensation 使用同一物理行跨度；该函数只负责数值归约，QR 的归一化和量化由调用方完成。
        """
        with cbd.vf(mode="simd"):
            # 1. 读取各特征向量，并累加 split-K 部分和。
            mask = reg.full_mask()
            for token_row in cbd.range(rows):
                for feature_vector_idx in range(width // VL):
                    dst_feature_offset = (token_row * dst.physical_layout.stride[0] + feature_vector_idx * VL)
                    partial_values = reg.vload(src, token_row * src.physical_layout.stride[0] + feature_vector_idx * VL)
                    # 2. 初始化部分和，或对下一个部分和执行补偿求和。
                    if cbd.const_expr(first):
                        reg.vstore(dst, dst_feature_offset, partial_values, mask)
                        reg.vstore(compensation, dst_feature_offset,
                                   reg.vdups(0.0, cbd.dtypes.float32, mask=mask), mask)
                    else:
                        corrected = reg.vsub(partial_values, reg.vload(compensation, dst_feature_offset), mask=mask)
                        prior = reg.vload(dst, dst_feature_offset)
                        total = reg.vadd(prior, corrected, mask=mask)
                        lost = reg.vsub(reg.vsub(total, prior, mask=mask), corrected, mask=mask)
                        # 3. 同时保存累加结果和丢失的低位残差。
                        reg.vstore(dst, dst_feature_offset, total, mask)
                        reg.vstore(compensation, dst_feature_offset, lost, mask)

    @cbd.jit
    def _quantize_qr(self, ub_qa, ub_gamma, ub_qr, ub_qr_bf16, ub_qr_bf16_slot, ub_exp, rows, norm_eps):
        """对一个 QA token 分块执行 RMSNorm，并生成 BF16 QR 与 MXFP8 payload/指数。

        先沿 rank 计算每行平方均值和 rstd，再按 32 元素组应用 gamma。每组保留 FP32 归一化值用于
        量化，同时写出量化前 BF16 QR；根据 amax 计算共享 2 的幂 scale，将缩放值饱和舍入为 E4M3FN，
        payload 打包到 ub_qr，指数加 E8M0_BIAS 后暂存 ub_exp。rows 限定有效行，物理 pitch 包含补齐。
        """
        tiling = self.tiling
        rank = tiling.rank_size
        inv_rank = 1.0 / rank
        # 按物理 UB 行跨度寻址，包含 FixPipe 插入的填充。
        src_pitch = ub_qa.physical_layout.stride[0]
        qr_pitch = ub_qr.physical_layout.stride[0]
        bf16_pitch = ub_qr_bf16.physical_layout.stride[0]
        with cbd.vf(mode="simd"):
            full = reg.full_mask()
            group_mask = reg.update_mask(MX_GROUP, elem_bits=32)[0]
            one = reg.vdups(1.0, cbd.dtypes.float32, mask=full)
            for token_row in cbd.range(rows):
                qa_row_offset = token_row * src_pitch
                # 1. 将平方和归约到第 0 个 lane，再广播到整行。
                square_sum_accumulator = reg.vdups(0.0, cbd.dtypes.float32, mask=full)
                for rank_vector_idx in range(rank // VL):
                    qa_values = reg.vload(ub_qa, qa_row_offset + rank_vector_idx * VL)
                    squared_values = reg.vmul(qa_values, qa_values, mask=full)
                    square_sum_accumulator = reg.vadd(square_sum_accumulator,
                                                     reg.vreduce_sum(squared_values, mask=full), mask=full)
                square_sum = reg.vdup(square_sum_accumulator, mask=full)
                mean_square = reg.vadds(reg.vmuls(square_sum, inv_rank, mask=full), norm_eps, mask=full)
                rstd = reg.vdiv(one, reg.vsqrt(mean_square, mask=full), mask=full)

                # 2. 逐个 rank 分组归一化，并确定共享量化 scale。
                for rank_group_idx in range(rank // MX_GROUP):
                    rank_group_offset = rank_group_idx * MX_GROUP
                    normalized_values = reg.vmul(reg.vload(ub_qa, qa_row_offset + rank_group_offset), rstd,
                                                mask=group_mask)
                    normalized_values = reg.vmul(normalized_values,
                                                reg.vload(ub_gamma, rank_group_offset), mask=group_mask)
                    # 公开 QR 是量化前的 BF16；内部 QB 的 MXFP8 量化继续使用 FP32
                    # 归一化值（接受与原小算子链路的数值差异，由网络精度对比验收）。
                    reg.vstore_pack(ub_qr_bf16_slot, token_row * bf16_pitch + rank_group_offset,
                                   reg.vcast(normalized_values, cbd.dtypes.bfloat16, mask=group_mask), group_mask,
                                   pack_mode=reg.PackMode.B32_TO_B16)
                    # sum 和 max 归约都只写第 0 个 lane，逐元素使用前必须广播。
                    amax = reg.vdup(reg.vreduce_max(reg.vabs(normalized_values, mask=group_mask), mask=group_mask),
                                   mask=group_mask)
                    shared_exponent = self._shared_exponent(amax, group_mask)
                    payload = self._to_e4m3(reg.vmul(normalized_values,
                                                    self._reciprocal(shared_exponent, group_mask), mask=group_mask),
                                            group_mask)
                    # 3. 写出 payload 字节及编码后的分组指数。
                    reg.vstore_pack(ub_qr, token_row * qr_pitch + rank_group_offset, payload, group_mask,
                                   pack_mode=reg.PackMode.B32_TO_B8)
                    reg.vstore_first(ub_exp, token_row * ub_exp.physical_layout.stride[0] + rank_group_idx,
                                    reg.vadds(shared_exponent, E8M0_BIAS, mask=group_mask))

    @cbd.jit
    def _shared_exponent(self, amax, mask):
        """HF 口径：scale 为不小于 max(amax,1e-4)/448 的最小 2 的幂。

        用 FP32 指数/尾数位向上取整（不依赖 log2 边界）；返回无偏指数 e，
        编码字节为 e+127。amax 为零时下限 1e-4 保证 scale 非零、payload 为 0。
        """
        floored = reg.vmaxs(amax, 1e-4, mask=mask)
        s0 = reg.vmuls(floored, 1.0 / E4M3_MAX, mask=mask)
        bits = reg.vreinterpret(s0, cbd.dtypes.int32)
        # (bits + 0x7FFFFF) >> 23：尾数非零时进位到指数，等价于向上取整到 2 的幂。
        biased = reg.vshr(reg.vadds(bits, 0x007FFFFF, mask=mask), FP32_EXP_SHIFT, mask=mask)
        exponent = reg.vmin(reg.vadds(biased, -FP32_EXP_BIAS, mask=mask),
                           reg.vdups(127, cbd.dtypes.int32, mask=mask), mask=mask)
        return reg.vmaxs(exponent, -E8M0_BIAS, mask=mask)

    @cbd.jit
    def _reciprocal(self, exponent, mask):
        """由量化共享指数直接构造 FP32 倒数 2**(-exponent)。

        对整数指数取负并加 FP32_EXP_BIAS，再左移到 FP32 指数字段并重新解释为 float32，避免一次
        向量除法。输入来自 _shared_exponent 的有界指数，因此结果为正规有限数；mask 保持与调用方
        当前 MX 分组一致，返回值用于把归一化值转换到 E4M3FN 的量化数值域。
        """
        field = reg.vadds(reg.vneg(exponent, mask=mask), FP32_EXP_BIAS, mask=mask)
        return reg.vreinterpret(reg.vshl(field, FP32_EXP_SHIFT, mask=mask), cbd.dtypes.float32)

    @cbd.jit
    def _to_e4m3(self, value, mask):
        """把 FP32 向量转换为饱和、最近偶数舍入的 E4M3FN payload。

        先将输入钳位到 [-E4M3_MAX, E4M3_MAX]，再调用 vcast 执行 RN 转换，避免有限值域外输入产生
        非预期表示；RegLayout.ZERO 指定目标寄存器布局，mask 限定当前有效分组。该函数同时服务 QR
        和 KV 量化，只返回寄存器中的 FP8 值，具体字节打包由调用方的 vstore_pack 完成。
        """
        clamped = reg.vmins(reg.vmaxs(value, -E4M3_MAX, mask=mask), E4M3_MAX, mask=mask)
        return reg.vcast(clamped, cbd.dtypes.float8_e4m3fn, mask=mask, rounding=reg.RoundingMode.RN,
                        reg_layout=reg.RegLayout.ZERO)

    @cbd.jit
    def _pack_scale(self, ub_exp, ub_scale, rows):
        """把 _quantize_qr 暂存的 int32 有偏指数打包为公开 E8M0 scale 字节。

        按有效 token 行和 qr_groups 逐向量读取 ub_exp，B32_TO_B8 仅保留每个 int32 的低字节，
        不再做数值转换或加偏置。输出使用 ub_scale 的对齐物理行跨度，尾向量通过 mask 限制有效列；
        分离该步骤可让量化主循环保留寄存器值，并用独立 VF 区域完成对齐写回。
        """
        tiling = self.tiling
        exp_pitch = ub_exp.physical_layout.stride[0]
        scale_pitch = ub_scale.physical_layout.stride[0]
        with cbd.vf(mode="simd"):
            # 1. 读取 int32 编码指数；2. 将低字节打包写出。
            for token_row in cbd.range(rows):
                for scale_vector_idx in range(ceil_div(tiling.qr_groups, VL)):
                    active_scale_lanes = min(VL, tiling.qr_groups - scale_vector_idx * VL)
                    mask = reg.update_mask(active_scale_lanes, elem_bits=32)[0]
                    reg.vstore_pack(ub_scale, token_row * scale_pitch + scale_vector_idx * VL,
                                   reg.vload(ub_exp, token_row * exp_pitch + scale_vector_idx * VL), mask,
                                   pack_mode=reg.PackMode.B32_TO_B8)

    @cbd.jit
    def _quantize_kv_group(self, value, ub_kvb, payload_offset, ub_bits, group_index, mask):
        """量化 KV 的一个 32 元素组，分别产出 E4M3FN payload 和 BF16 scale。

        根据组内 FP32 amax 求不小于 max(amax, 1e-4)/448 的最小 2 的幂 scale，利用 FP32 位域
        向上取整并截取高 16 位保存 BF16 scale。随后用实际 scale 反量化输入、饱和转换为 E4M3FN，
        payload 写入 ub_kvb 的指定偏移，scale 位值写入 ub_bits 的 group_index，供整行末尾统一拼接。
        """
        reg.vdups(E4M3_MAX, cbd.dtypes.float32, mask=mask)
        amax = reg.vdup(reg.vreduce_max(reg.vabs(value, mask=mask), mask=mask), mask=mask)
        # 网络原有 round_scale=True 口径：a=max(amax,1e-4)，scale 为不小于
        # a/448 的最小 2 的幂；用 FP32 指数/尾数位向上取整，避免 log2 边界差异。
        floored = reg.vmaxs(amax, 1e-4, mask=mask)
        s0 = reg.vmuls(floored, 1.0 / E4M3_MAX, mask=mask)
        s0_bits = reg.vreinterpret(s0, cbd.dtypes.int32)
        exponent = reg.vshr(reg.vadds(s0_bits, 0x007FFFFF, mask=mask), 23, mask=mask)
        exponent = reg.vmin(exponent, reg.vdups(254, cbd.dtypes.int32, mask=mask), mask=mask)
        scale_bits = reg.vshl(exponent, 23, mask=mask)
        bf16_bits = reg.vshr(scale_bits, 16, mask=mask)
        reg.vstore_first(ub_bits, group_index, bf16_bits)
        scale_used = reg.vreinterpret(scale_bits, cbd.dtypes.float32)
        quotient = reg.vdiv(value, scale_used, mask=mask)
        quotient = reg.vselect(quotient,
                              reg.vdups(0.0, cbd.dtypes.float32, mask=mask),
                              cond_mask=reg.vnes(scale_used, 0.0, mask=mask))
        reg.vstore_pack(ub_kvb, payload_offset, self._to_e4m3(quotient, mask), mask, pack_mode=reg.PackMode.B32_TO_B8)

    @cbd.jit
    def _finish_kv_row(self, ub_kv, ub_gamma, ub_sin, ub_cos, ub_kvb, ub_tail, ub_bits, token_row, table_row, norm_eps):
        """沿 head 特征轴归一化 KV，执行尾部 RoPE，再按 32 元素组量化。
        每组使用独立的 BF16 scale；payload 与 scale 成对写入 544 字节行。
        RoPE 之前的组就地量化，尾部 RoPE 之后的组从 FP32 暂存区量化。
        每行单独搬到其索引指定的 cache 位置。"""
        tiling = self.tiling
        head_width, rope_width = tiling.head_size, tiling.rope_width
        rope_half_width = rope_width // 2
        inv_head = 1.0 / head_width
        src_pitch = ub_kv.physical_layout.stride[0]
        group_count = head_width // MX_GROUP
        # RoPE 可能与最后一个 32 元素组部分相交；该组必须整体留到旋转后量化。
        first_rope_group = (head_width - rope_width) // MX_GROUP
        tail_group_offset = (head_width - VL) // MX_GROUP
        with cbd.vf(mode="simd"):
            full = reg.full_mask()
            m_pair = reg.update_mask(rope_half_width, elem_bits=32)[0]
            m_rope = reg.update_mask(rope_width, elem_bits=32)[0]
            group_mask = reg.update_mask(MX_GROUP, elem_bits=32)[0]
            one = reg.vdups(1.0, cbd.dtypes.float32, mask=full)
            kv_row_offset = token_row * src_pitch
            # 1. 计算沿 head 特征轴的 RMS 归一化因子。
            square_sum_accumulator = reg.vdups(0.0, cbd.dtypes.float32, mask=full)
            for head_vector_idx in range(head_width // VL):
                kv_values = reg.vload(ub_kv, kv_row_offset + head_vector_idx * VL)
                squared_values = reg.vmul(kv_values, kv_values, mask=full)
                square_sum_accumulator = reg.vadd(square_sum_accumulator,
                                                 reg.vreduce_sum(squared_values, mask=full), mask=full)
            square_sum = reg.vdup(square_sum_accumulator, mask=full)
            mean_square = reg.vadds(reg.vmuls(square_sum, inv_head, mask=full), norm_eps, mask=full)
            rstd = reg.vdiv(one, reg.vsqrt(mean_square, mask=full), mask=full)

            if cbd.const_expr(rope_width % MX_GROUP == 0):
                pre_groups = group_count - rope_width // MX_GROUP
                post_group_start = pre_groups
                rope_tail_offset = 0
            else:
                pre_groups = first_rope_group
                post_group_start = first_rope_group
                rope_tail_offset = VL - rope_width
            post_groups = group_count - post_group_start

            # 2. RoPE 之前的组：逐 32 元素归一化、求 BF16 scale 并量化。
            for group_index in range(pre_groups):
                normed = reg.vmul(reg.vload(ub_kv, kv_row_offset + group_index * MX_GROUP), rstd, mask=group_mask)
                normed = reg.vmul(normed, reg.vload(ub_gamma, group_index * MX_GROUP), mask=group_mask)
                self._quantize_kv_group(normed, ub_kvb, group_index * MX_GROUP, ub_bits, group_index, group_mask)

            # 3. 在 FP32 暂存区拼接最后一个向量。字节打包后 RoPE 后半区
            # 可能不对齐，因此拼接完成后一次性打包整个向量。
            normed = reg.vmul(reg.vload(ub_kv, kv_row_offset + head_width - VL), rstd, mask=full)
            normed = reg.vmul(normed, reg.vload(ub_gamma, head_width - VL), mask=full)
            reg.vstore(ub_tail, 0, normed, full)
            rope_vec = self._rope_pair(ub_kv, kv_row_offset + head_width - rope_width, ub_gamma,
                                       head_width - rope_width, ub_sin, ub_cos, table_row, rstd, m_pair)
            reg.vstore(ub_tail, rope_tail_offset, rope_vec, m_rope)
            # 此屏障必须保留：紧接着的读取依赖上方刚写入的尾部数据。
            reg.vmem_bar("vst_vld")

            # 4. RoPE 之后的组：从暂存区按 32 元素量化。
            if cbd.const_expr(rope_width % MX_GROUP == 0):
                for local_group in range(post_groups):
                    group_index = post_group_start + local_group
                    self._quantize_kv_group(reg.vload(ub_tail, local_group * MX_GROUP), ub_kvb, group_index * MX_GROUP,
                                            ub_bits, group_index, group_mask)
            else:
                for group_index in range(post_group_start, group_count):
                    self._quantize_kv_group(reg.vload(ub_tail, (group_index - tail_group_offset) * MX_GROUP), ub_kvb,
                                            group_index * MX_GROUP, ub_bits, group_index, group_mask)

            # 5. 16 个 BF16 scale 以小端连续写入 payload 之后的 scale 区。
            reg.vmem_bar("vst_vld")
            scale_view = cbd.from_tensor(ub_kvb, dtype=cbd.dtypes.uint16,
                                         layout=cbd.make_layout((1, group_count)), offset=head_width)
            scale_mask = reg.update_mask(group_count, elem_bits=16)[0]
            reg.vstore(scale_view, 0,
                      reg.vreinterpret(reg.vpack(reg.vload(ub_bits, 0), cbd.dtypes.uint16), cbd.dtypes.bfloat16),
                      scale_mask)

    @cbd.jit
    def _rope_pair(self, src, src_feature_offset, gamma, gamma_feature_offset, ub_sin, ub_cos, table_row, rstd, mask):
        """按网络现有的 interleave 约定执行尾部 RoPE，输出保持交错排列。
        out_even = x_even * cos_even - x_odd * sin_even
        out_odd  = x_odd * cos_odd + x_even * sin_odd
        位置表为全宽 [T,Dr]，按 (2i, 2i+1) 取 cos/sin；不依赖相邻表项相等。
        rstd 为 None 时跳过归一化，用于 QB 分支。"""
        even, odd = reg.vload_deinterleave(src, src_feature_offset, width="b32")
        if cbd.const_expr(gamma is not None):
            g_even, g_odd = reg.vload_deinterleave(gamma, gamma_feature_offset, width="b32")
            even = reg.vmul(reg.vmul(even, g_even, mask=mask), rstd, mask=mask)
            odd = reg.vmul(reg.vmul(odd, g_odd, mask=mask), rstd, mask=mask)
        # RoPE 表按补齐后的物理行跨度寻址。
        pitch = ub_sin.physical_layout.stride[0]
        cos_even, cos_odd = reg.vload_deinterleave(ub_cos, table_row * pitch, width="b32")
        sin_even, sin_odd = reg.vload_deinterleave(ub_sin, table_row * pitch, width="b32")
        out_even = reg.vsub(reg.vmul(even, cos_even, mask=mask), reg.vmul(odd, sin_even, mask=mask), mask=mask)
        out_odd = reg.vadd(reg.vmul(odd, cos_odd, mask=mask), reg.vmul(even, sin_odd, mask=mask), mask=mask)
        interleaved, _ = reg.vinterleave(out_even, out_odd)
        return interleaved

    @cbd.jit
    def _to_bf16(self, value, mask):
        """将 FP32 向量按最近偶数舍入转换为 BF16，并保留调用方的有效 lane 掩码。

        该轻量封装统一 Q 输出路径的舍入模式；结果仍位于向量寄存器中，由调用方通过 vstore_pack
        完成 32 位到 16 位的内存打包，不在此函数内写 UB 或 GM。
        """
        return reg.vcast(value, cbd.dtypes.bfloat16, mask=mask, rounding=reg.RoundingMode.RN)
