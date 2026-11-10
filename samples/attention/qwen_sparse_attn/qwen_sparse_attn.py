# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""基于 128-token 分块的 Qwen 稀疏注意力。

负载均衡 metadata 由 ``qwen_sparse_attn_metadata`` 在 AICPU 上显式生成，
本 NPU 主 kernel 只负责消费。公共 Python 适配层暴露受支持的 DSL 契约，
不支持的选项会被明确拒绝。
"""

from cannbotdsl import from_tensor

__all__ = [
    "QuantMode",
    "SparseMode",
    "launch_qwen_sparse_attn_block_128_bf16",
    "launch_qwen_sparse_attn_block_128_fp16",
    "launch_qwen_sparse_attn_block_128_fp8",
    "get_kernel",
    "qwen_sparse_attn_block_128",
    "qwen_sparse_attn",
]

from threading import Lock

import cannbotdsl as cb
import torch
from cannbotdsl import (
    Constexpr,
    MemLoc,
    Tensor,
    const_expr,
    dtypes,
    get_block_idx,
    get_subblock_id,
    jit,
    matmul,
    mem_copy,
    tile_slice,
    vec_sync_all,
    vf,
)
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl import make_layout
from cannbotdsl.lang.host import host

if __package__:
    from .qwen_sparse_attn_check import (
        QuantMode,
        SparseMode,
        require_none,
        validate_attn_mask,
        validate_block128_attrs,
        validate_block128_inputs,
        validate_block128_metadata,
    )
else:
    from qwen_sparse_attn_check import (
        QuantMode,
        SparseMode,
        require_none,
        validate_attn_mask,
        validate_block128_attrs,
        validate_block128_inputs,
        validate_block128_metadata,
    )

_COMPILED_KERNEL_LOCK = Lock()
_COMPILED_KERNELS = {}


def _offset_view(tensor, origin):
    """使用原生张量切片替代已移除的 ``rebase_view`` API。"""
    return tensor[tuple(slice(start, None) for start in origin)]


def _bf16_row_tnd(tensor, t_idx, n1_idx, dim):
    return tile_slice(
        _offset_view(tensor[t_idx, n1_idx, None], (0, 0, 0)),
        (1, dim),
        (0, 0),
    )


class QueryTail:
    def __init__(self, dtype, dim):
        self.dim = dim
        # 无法组成完整 M32 Q tile 的 GQA group 使用这些回退缓冲区；完整 M32
        # group 通过 Cube MTE2 直接搬运，不经过这些缓冲区。
        self.row = RingBuffer(MemLoc.UB, dtype, make_layout((1, dim)), depth=2)
        self.q = Buffer(MemLoc.UB, dtype, make_layout((16, dim)), data_format="nz")

    @jit
    def clear_q_tail(self, dst):
        strides = dst.physical_layout.stride
        rows_per_nz_block = strides[1] // strides[2]
        block_stride = strides[0] // (strides[2] // strides[3])
        with vf(mode="simd"):
            mask, _ = rr.update_mask(128, elem_bits=16)
            zero = rr.vdups(0.0, dst.dtype, mask=mask)
            for row in range(dst.shape[0]):
                row_offset = (row // rows_per_nz_block) * strides[1] + (
                    row % rows_per_nz_block
                ) * strides[2]
                for part in range(self.dim // 128):
                    rr.vstore_strided(
                        dst,
                        row_offset + part * 8 * strides[0],
                        zero,
                        mask,
                        block_stride=block_stride,
                    )

    @jit
    def scatter(self, dst, src, row0):
        strides = dst.physical_layout.stride
        rows_per_nz_block = strides[1] // strides[2]
        block_stride = strides[0] // (strides[2] // strides[3])
        with vf(mode="simd"):
            mask, _ = rr.update_mask(128, elem_bits=16)
            for row in range(src.shape[0]):
                dst_row = row0 + row
                row_offset = (dst_row // rows_per_nz_block) * strides[1] + (
                    dst_row % rows_per_nz_block
                ) * strides[2]
                for part in range(self.dim // 128):
                    value = rr.vload(src, row * self.dim + part * 128)
                    rr.vstore_strided(
                        dst,
                        row_offset + part * 8 * strides[0],
                        value,
                        mask,
                        block_stride=block_stride,
                    )

    @jit
    def load_q_tail_tnd(
        self,
        target,
        q,
        t_idx,
        n2_idx,
        g_chunk_idx,
        group,
        vec_idx,
        logical_row_base,
        active_rows,
    ):
        """将当前 vector core 的有效行载入本地 16 行暂存 tile。"""
        vec_sync_all()
        dst = self.q
        self.clear_q_tail(dst)
        for row in range(active_rows):
            g_idx = g_chunk_idx * 32 + logical_row_base + row
            n1_idx = n2_idx * group + g_idx
            mem_copy(self.row.next_write(), _bf16_row_tnd(q, t_idx, n1_idx, self.dim))
            self.scatter(dst, self.row.next_read(), row)
        vec_sync_all()
        mem_copy(target.next_write().partition(dim=0, part_id=vec_idx), self.q)

    @jit
    def load_q_packed_tnd(
        self,
        target,
        q,
        q_idx,
        q_token_num,
        n2_idx,
        group,
        vec_idx,
        logical_row_base,
        active_rows,
    ):
        """按 `Query x GQA head` 顺序把任意 G 的有效行填入 M32。"""
        vec_sync_all()
        self.clear_q_tail(self.q)
        for row in range(active_rows):
            packed_row = logical_row_base + row
            q_slot = packed_row // group
            g_idx = packed_row % group
            if q_slot < q_token_num:
                mem_copy(
                    self.row.next_write(),
                    _bf16_row_tnd(q, q_idx + q_slot, n2_idx * group + g_idx, self.dim),
                )
                self.scatter(self.q, self.row.next_read(), row)
        vec_sync_all()
        mem_copy(target.next_write().partition(dim=0, part_id=vec_idx), self.q)


class Fp8QueryTail:
    """将不完整的 GQA group 打包为 FP8 NZ（C0=32）。"""

    def __init__(self, dim):
        self.dim = dim
        self.row_bytes = RingBuffer(
            MemLoc.UB, dtypes.int8, make_layout((1, dim)), depth=2
        )
        self.q_bytes = Buffer(
            MemLoc.UB, dtypes.int8, make_layout((16, dim)), data_format="nz"
        )
        self.q = from_tensor(self.q_bytes, dtype=dtypes.float8_e4m3fn)

    @jit
    def clear(self):
        """清零完整的 `(16,128)` FP8 Q 暂存缓冲区。

        多 Query 打包路径保留清零；单 Query 尾块只消费有效输出行，无需清零
        未写入的 Q 行。
        """
        with vf(mode="simd"):
            mask, _ = rr.update_mask(256, elem_bits=8)
            zero = rr.vdups(0, dtypes.int8, mask=mask)
            for part in range((16 * self.dim) // 256):
                rr.vstore(self.q_bytes, part * 256, zero, mask)

    @jit
    def scatter(self, src, row):
        strides = self.q_bytes.physical_layout.stride
        rows_per_nz_block = strides[1] // strides[2]
        block_stride = strides[0] // (strides[2] // strides[3])
        with vf(mode="simd"):
            mask, _ = rr.update_mask(128, elem_bits=8)
            row_offset = (row // rows_per_nz_block) * strides[1] + (
                row % rows_per_nz_block
            ) * strides[2]
            value = rr.vload(src, 0)
            rr.vstore_strided(
                self.q_bytes, row_offset, value, mask, block_stride=block_stride
            )

    @jit
    def load_q_tail_tnd(
        self,
        target,
        q,
        t_idx,
        n2_idx,
        g_chunk_idx,
        group,
        vec_idx,
        logical_row_base,
        active_rows,
    ):
        """将当前 vector core 的有效行载入本地 16 行暂存 tile。"""
        vec_sync_all()
        # QK may read the unused physical rows, but subsequent Vector stages
        # consume only active_rows and do not propagate those rows to output.
        for row in range(active_rows):
            g_idx = g_chunk_idx * 32 + logical_row_base + row
            n1_idx = n2_idx * group + g_idx
            row_write = from_tensor(
                self.row_bytes.next_write(), dtype=dtypes.float8_e4m3fn
            )
            mem_copy(row_write, _bf16_row_tnd(q, t_idx, n1_idx, self.dim))
            row_read = from_tensor(
                self.row_bytes.next_read(), dtype=dtypes.float8_e4m3fn
            )
            self.scatter(row_read, row)
        vec_sync_all()
        mem_copy(target.next_write().partition(dim=0, part_id=vec_idx), self.q)

    @jit
    def load_q_packed_tnd(
        self,
        target,
        q,
        q_idx,
        q_token_num,
        n2_idx,
        group,
        vec_idx,
        logical_row_base,
        active_rows,
    ):
        """按 `Query x GQA head` 顺序把任意 G 的有效行填入 M32。"""
        vec_sync_all()
        self.clear()
        for row in range(active_rows):
            packed_row = logical_row_base + row
            q_slot = packed_row // group
            g_idx = packed_row % group
            if q_slot < q_token_num:
                row_write = from_tensor(
                    self.row_bytes.next_write(), dtype=dtypes.float8_e4m3fn
                )
                mem_copy(
                    row_write,
                    _bf16_row_tnd(q, q_idx + q_slot, n2_idx * group + g_idx, self.dim),
                )
                row_read = from_tensor(
                    self.row_bytes.next_read(), dtype=dtypes.float8_e4m3fn
                )
                self.scatter(row_read, row)
        vec_sync_all()
        mem_copy(target.next_write().partition(dim=0, part_id=vec_idx), self.q)


class QwenSparseAttnBlock128Vector:
    def __init__(self, score_dtype, output_dtype, dim, fp8=False):
        self.dim = dim
        self.dtype = score_dtype
        self.output_dtype = output_dtype
        self.fp8 = fp8
        self.half_softmax = score_dtype == dtypes.float16
        self.m = Buffer(MemLoc.UB, dtypes.float32, make_layout((16, 8)))
        self.l = Buffer(MemLoc.UB, dtypes.float32, make_layout((16, 8)))
        self.new_max = Buffer(MemLoc.UB, dtypes.float32, make_layout((16, 8)))
        self.new_sum = Buffer(MemLoc.UB, dtypes.float32, make_layout((16, 8)))
        # Softmax 比对应的 PV 更新提前两个宏阶段执行，因此每个在线重缩放系数都要
        # 保留到对应 PV 更新消费完成。
        self.alpha = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((16, 8)), depth=3
        )
        self.u = Buffer(MemLoc.UB, dtypes.float32, make_layout((16, dim)))
        # sparse_mode=3 使用外部 2048x2048 上三角模板。MTE2 每次搬入一行，
        # Vector 再把该行扩展到同一 Query 对应的所有 head 行。
        self.mask_template = RingBuffer(
            MemLoc.UB, dtypes.int8, make_layout((1, 128)), depth=2
        )
        self.mask = Buffer(MemLoc.UB, dtypes.int8, make_layout((16, 128)))
        # raw-VF writes P and MTE3 consumes it one stage later.  Keeping P as
        # a depth-2 Channel lets the compiler insert the V<->MTE3 transactions
        # and overlap the next softmax with the previous UB->L1 copy.
        self.p_storage = RingBuffer(
            MemLoc.UB,
            dtypes.int8 if fp8 else score_dtype,
            make_layout((16, 128)),
            depth=2,
            data_format="nz",
        )
        self.p_dtype = dtypes.float8_e4m3fn if fp8 else score_dtype
        self.out = Buffer(MemLoc.UB, output_dtype, make_layout((16, dim)))

    @jit
    def load_mask(self, attn_mask, val_col):
        """从 2048x2048 causal 模板搬入当前 128 列对应的一行。"""
        base = min(127, max(-1, val_col - 1))
        nonnegative = base >= 0
        row_offset = base if nonnegative else 0
        col_offset = 0 if nonnegative else 0 - base
        row_end = row_offset + 1
        col_end = col_offset + 128
        source = tile_slice(
            attn_mask[row_offset:row_end, col_offset:col_end],
            (1, 128),
            (0, 0),
        )
        mem_copy(self.mask_template.next_write(), source)

    @jit
    def expand_mask(self, first_row, row_num):
        """把一个 Query 的模板行复制到该 Query 的所有 head 行。"""
        template = self.mask_template.next_read()
        with vf(mode="simd"):
            full, _ = rr.update_mask(128, elem_bits=8)
            value = rr.vload(template, 0)
            for row in range(row_num):
                rr.vstore(self.mask, (first_row + row) * 128, value, full)

    def apply_mask(self, value, offset, lanes):
        packed = rr.vload_unpack(self.mask, offset, unpack_mode=rr.UnpackMode.B8_TO_B32)
        packed = rr.vcast(
            rr.vreinterpret(packed, dtypes.int32), dtypes.float32, mask=lanes
        )
        visible = rr.veqs(packed, 0, mask=lanes)
        hidden = rr.vdups(-3.0e38, dtypes.float32, mask=lanes)
        return rr.vselect(value, hidden, cond_mask=visible)

    @jit
    def _softmax_fold_loop(self, p, probability, max_buf, sum_dst, active_rows):
        """以指定最大值执行 exp、reduce_sum，并把 P 写入 L1 前置 UB。"""
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            full16, _ = rr.update_mask(128, elem_bits=16)
            for row in range(active_rows):
                row_offset = row * p.physical_layout.stride[0]
                row_max = rr.vload_broadcast(max_buf, row * 8)
                first_half = rr.vexp_sub(rr.vload(p, row_offset), row_max, mask=full)
                second_half = rr.vexp_sub(
                    rr.vload(p, row_offset + 64), row_max, mask=full
                )
                self._store_probability_row(
                    probability, first_half, second_half, row, full, full16
                )
                row_sum = rr.vreduce_sum(
                    rr.vadd(first_half, second_half, mask=full), mask=full
                )
                rr.vstore_first(sum_dst, row * 8, row_sum)

    @jit
    def _softmax_rest_tail(self, alpha_out, active_rows):
        """更新在线 softmax 状态并发布 V2 所需的重缩放系数。"""
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            for row in range(active_rows):
                old_max = rr.vload_broadcast(self.m, row * 8)
                next_max = rr.vload_broadcast(self.new_max, row * 8)
                alpha = rr.vexp_sub(old_max, next_max, mask=full)
                old_sum = rr.vload_broadcast(self.l, row * 8)
                block_sum = rr.vload_broadcast(self.new_sum, row * 8)
                next_sum = rr.vmadd(old_sum, alpha, block_sum, mask=full)
                rr.vstore_first(alpha_out, row * 8, alpha)
                rr.vstore_first(self.m, row * 8, next_max)
                rr.vstore_first(self.l, row * 8, next_sum)

    @jit
    def softmax_first(self, p, probability, scale, active_rows):
        """首个 KV 页：建立 max/sum，不读取旧状态，也不产生 alpha。"""
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            for row in range(active_rows):
                self._pass_a_row(p, scale, self.m, row, full)
        self._softmax_fold_loop(p, probability, self.m, self.l, active_rows)

    @jit
    def softmax_rest(self, p, probability, alpha_out, scale, active_rows):
        """后续 KV 页：先合并 max，再 fold，最后更新在线状态。"""
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            for row in range(active_rows):
                self._pass_a_row(p, scale, self.new_max, row, full)
            rr.vmem_bar("vst_vld")
            for row in range(active_rows):
                merged_max = rr.vmax(
                    rr.vload_broadcast(self.m, row * 8),
                    rr.vload_broadcast(self.new_max, row * 8),
                    mask=full,
                )
                rr.vstore_first(self.new_max, row * 8, merged_max)
        self._softmax_fold_loop(p, probability, self.new_max, self.new_sum, active_rows)
        self._softmax_rest_tail(alpha_out, active_rows)

    @jit
    def _softmax_fold_loop_fp8(self, p, probability, max_buf, sum_dst, active_rows):
        """FP8 P 物化版本的 fold loop；运行和仍保持未缩放 FP32。"""
        strides = probability.physical_layout.stride
        rows_per_nz_block = strides[1] // strides[2]
        block_stride = strides[0] // (strides[2] // strides[3])
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            full8, _ = rr.update_mask(256, elem_bits=8)
            store8, _ = rr.update_mask(128, elem_bits=8)
            for row in range(active_rows):
                row_offset = row * p.physical_layout.stride[0]
                row_max = rr.vload_broadcast(max_buf, row * 8)
                first_half = rr.vexp_sub(rr.vload(p, row_offset), row_max, mask=full)
                second_half = rr.vexp_sub(
                    rr.vload(p, row_offset + 64), row_max, mask=full
                )
                row_sum = rr.vreduce_sum(
                    rr.vadd(first_half, second_half, mask=full), mask=full
                )
                rr.vstore_first(sum_dst, row * 8, row_sum)
                even, odd = rr.vdeinterleave(
                    rr.vmuls(first_half, 448.0, mask=full),
                    rr.vmuls(second_half, 448.0, mask=full),
                )
                first_fp8 = rr.vcast(
                    even,
                    dtypes.float8_e4m3fn,
                    mask=full,
                    saturate=True,
                    reg_layout=rr.RegLayout.ZERO,
                )
                second_fp8 = rr.vcast(
                    odd,
                    dtypes.float8_e4m3fn,
                    mask=full,
                    saturate=True,
                    reg_layout=rr.RegLayout.TWO,
                )
                packed = rr.vbitwise_or(first_fp8, second_fp8, mask=full8)
                packed, _ = rr.vdeinterleave(packed, packed)
                output_offset = (row // rows_per_nz_block) * strides[1] + (
                    row % rows_per_nz_block
                ) * strides[2]
                rr.vstore_strided(
                    probability,
                    output_offset,
                    packed,
                    store8,
                    block_stride=block_stride,
                )

    @jit
    def softmax_fp8_first(self, p, probability, scale, active_rows):
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            for row in range(active_rows):
                self._pass_a_row_fp8(p, scale, self.m, row, full)
        self._softmax_fold_loop_fp8(p, probability, self.m, self.l, active_rows)

    @jit
    def softmax_fp8_rest(self, p, probability, alpha_out, scale, active_rows):
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            for row in range(active_rows):
                self._pass_a_row_fp8(p, scale, self.new_max, row, full)
            rr.vmem_bar("vst_vld")
            for row in range(active_rows):
                merged_max = rr.vmax(
                    rr.vload_broadcast(self.m, row * 8),
                    rr.vload_broadcast(self.new_max, row * 8),
                    mask=full,
                )
                rr.vstore_first(self.new_max, row * 8, merged_max)
        self._softmax_fold_loop_fp8(
            p, probability, self.new_max, self.new_sum, active_rows
        )
        self._softmax_rest_tail(alpha_out, active_rows)

    @jit
    def update_first(self, pv, part_idx, active_rows):
        """首个 KV 页直接建立 O 累加器，不读取旧状态和 alpha。"""
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            for row in range(active_rows):
                for half in range(2):
                    off = row * self.dim + part_idx * 128 + half * 64
                    part = rr.vload(pv, row * pv.physical_layout.stride[0] + half * 64)
                    rr.vstore(self.u, off, part, full)

    @jit
    def update_rest(
        self,
        pv,
        part_idx,
        alpha_ready,
        active_rows,
    ):
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            for row in range(active_rows):
                alpha = rr.vload_broadcast(alpha_ready, row * 8)
                for half in range(2):
                    off = row * self.dim + part_idx * 128 + half * 64
                    part = rr.vload(pv, row * pv.physical_layout.stride[0] + half * 64)
                    updated = rr.vmadd(rr.vload(self.u, off), alpha, part, mask=full)
                    rr.vstore(self.u, off, updated, full)

    @jit
    def finalize(self, active_rows):
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            full16, _ = rr.update_mask(128, elem_bits=16)
            for row in range(active_rows):
                denom = rr.vload_broadcast(self.l, row * 8)
                for part in range(self.dim // 128):
                    off = row * self.dim + part * 128
                    first_half = rr.vdiv(rr.vload(self.u, off), denom, mask=full)
                    second_half = rr.vdiv(rr.vload(self.u, off + 64), denom, mask=full)
                    first_half = rr.vcast(
                        first_half,
                        self.output_dtype,
                        mask=full,
                        reg_layout=rr.RegLayout.ZERO,
                    )
                    second_half = rr.vcast(
                        second_half,
                        self.output_dtype,
                        mask=full,
                        reg_layout=rr.RegLayout.ZERO,
                    )
                    packed, _ = rr.vdeinterleave(first_half, second_half)
                    rr.vstore(self.out, off, packed, full16)

    @jit
    def finalize_fp8(self, active_rows):
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            full16, _ = rr.update_mask(128, elem_bits=16)
            for row in range(active_rows):
                denom = rr.vload_broadcast(self.l, row * 8)
                for part in range(self.dim // 128):
                    off = row * self.dim + part * 128
                    first_half = rr.vmuls(
                        rr.vdiv(rr.vload(self.u, off), denom, mask=full),
                        1.0 / 448.0,
                        mask=full,
                    )
                    second_half = rr.vmuls(
                        rr.vdiv(rr.vload(self.u, off + 64), denom, mask=full),
                        1.0 / 448.0,
                        mask=full,
                    )
                    first_half = rr.vcast(
                        first_half,
                        self.output_dtype,
                        mask=full,
                        reg_layout=rr.RegLayout.ZERO,
                    )
                    second_half = rr.vcast(
                        second_half,
                        self.output_dtype,
                        mask=full,
                        reg_layout=rr.RegLayout.ZERO,
                    )
                    packed, _ = rr.vdeinterleave(first_half, second_half)
                    rr.vstore(self.out, off, packed, full16)

    @jit
    def finalize_empty(self, active_rows):
        """在没有 KV 行时物化数学定义上的零结果。"""
        with vf(mode="simd"):
            full16, _ = rr.update_mask(128, elem_bits=16)
            zero = rr.vdups(0.0, self.output_dtype, mask=full16)
            for row in range(active_rows):
                for part in range(self.dim // 128):
                    rr.vstore(self.out, row * self.dim + part * 128, zero, full16)

    def _pass_a_row(self, p, scale, max_dst, row, full):
        """缩放并应用 mask，同时把当前行最大值写入指定状态。"""
        row_offset = row * p.physical_layout.stride[0]
        full16, _ = rr.update_mask(128, elem_bits=16)
        raw_first_half = rr.vcast(
            rr.vload(p, row_offset),
            self.dtype,
            mask=full,
            reg_layout=rr.RegLayout.ZERO,
        )
        raw_second_half = rr.vcast(
            rr.vload(p, row_offset + 64),
            self.dtype,
            mask=full,
            reg_layout=rr.RegLayout.ZERO,
        )
        first_half = rr.vcast(
            raw_first_half, dtypes.float32, mask=full16, reg_layout=rr.RegLayout.ZERO
        )
        second_half = rr.vcast(
            raw_second_half, dtypes.float32, mask=full16, reg_layout=rr.RegLayout.ZERO
        )
        first_half = rr.vcast(
            rr.vmuls(first_half, scale, mask=full),
            self.dtype,
            mask=full,
            reg_layout=rr.RegLayout.ZERO,
        )
        second_half = rr.vcast(
            rr.vmuls(second_half, scale, mask=full),
            self.dtype,
            mask=full,
            reg_layout=rr.RegLayout.ZERO,
        )
        first_half = rr.vcast(
            first_half, dtypes.float32, mask=full16, reg_layout=rr.RegLayout.ZERO
        )
        second_half = rr.vcast(
            second_half, dtypes.float32, mask=full16, reg_layout=rr.RegLayout.ZERO
        )
        first_half = self.apply_mask(first_half, row * 128, full)
        second_half = self.apply_mask(second_half, row * 128 + 64, full)
        rr.vstore(p, row_offset, first_half, full)
        rr.vstore(p, row_offset + 64, second_half, full)
        row_max = rr.vreduce_max(rr.vmax(first_half, second_half, mask=full), mask=full)
        all_masked = rr.vles(row_max, -1.0e30, mask=full)
        zero = rr.vdups(0.0, dtypes.float32, mask=full)
        row_max = rr.vselect(zero, row_max, cond_mask=all_masked)
        rr.vstore_first(max_dst, row * 8, row_max)

    def _pass_a_row_fp8(self, p, scale, max_dst, row, full):
        row_offset = row * p.physical_layout.stride[0]
        first_half = rr.vmuls(rr.vload(p, row_offset), scale, mask=full)
        second_half = rr.vmuls(rr.vload(p, row_offset + 64), scale, mask=full)
        first_half = self.apply_mask(first_half, row * 128, full)
        second_half = self.apply_mask(second_half, row * 128 + 64, full)
        rr.vstore(p, row_offset, first_half, full)
        rr.vstore(p, row_offset + 64, second_half, full)
        row_max = rr.vreduce_max(rr.vmax(first_half, second_half, mask=full), mask=full)
        all_masked = rr.vles(row_max, -1.0e30, mask=full)
        zero = rr.vdups(0.0, dtypes.float32, mask=full)
        row_max = rr.vselect(zero, row_max, cond_mask=all_masked)
        rr.vstore_first(max_dst, row * 8, row_max)

    def _store_probability_row(
        self, probability, first_half, second_half, row, full, full16
    ):
        strides = probability.physical_layout.stride
        rows_per_nz_block = strides[1] // strides[2]
        block_stride = strides[0] // (strides[2] // strides[3])
        first_half = rr.vcast(
            first_half, self.dtype, mask=full, reg_layout=rr.RegLayout.ZERO
        )
        second_half = rr.vcast(
            second_half, self.dtype, mask=full, reg_layout=rr.RegLayout.ZERO
        )
        packed, _ = rr.vdeinterleave(first_half, second_half)
        output_offset = (row // rows_per_nz_block) * strides[1] + (
            row % rows_per_nz_block
        ) * strides[2]
        rr.vstore_strided(
            probability, output_offset, packed, full16, block_stride=block_stride
        )


class QwenSparseAttnBlock128Cube:
    def __init__(self, dim, q_channel, input_dtype):
        self.dim = dim
        self.q = q_channel
        self.k = RingBuffer(
            MemLoc.L1, input_dtype, make_layout((128, dim)), depth=3, data_format="nz"
        )
        self.v = RingBuffer(
            MemLoc.L1, input_dtype, make_layout((128, dim)), depth=3, data_format="nz"
        )
        # 阶段内共享 L0 槽位：QK 按 128 宽分块归约 D，PV 每次输出 128 个维度。
        self.a = RingBuffer(MemLoc.L0A, input_dtype, make_layout((32, 128)), depth=2)
        self.b = RingBuffer(MemLoc.L0B, input_dtype, make_layout((128, 128)), depth=2)
        self.c = RingBuffer(MemLoc.L0C, dtypes.float32, make_layout((32, 128)), depth=2)

    @jit
    def load_q_tnd(self, q, t_idx, n2_idx, g_chunk_idx, group):
        """将一个完整连续的 TND M32 head 分块从 GM 搬运到 L1。"""
        first_n1_idx = n2_idx * group + g_chunk_idx * 32
        src = tile_slice(
            _offset_view(q[t_idx, None, None], (0, first_n1_idx, 0)),
            (32, self.dim),
            (0, 0),
        )
        mem_copy(self.q.next_write(), src)

    @jit
    def load_page(self, target, cache, page_idx, n2_idx):
        """加载完整的 paged `[128,D]` ND tile，行 stride 保留全部 KV head。"""
        src = tile_slice(
            cache[page_idx, n2_idx, None, None],
            (128, self.dim),
            (0, 0),
        )
        mem_copy(target.next_write(), src)

    @jit
    def compute_qk(self, q_tile):
        k_tile = self.k.next_read()
        result = self.c.next_write()
        for part in range(self.dim // 128):
            mem_copy(self.a.next_write(), tile_slice(q_tile, (32, 128), (0, part)))
            mem_copy(self.b.next_write(), tile_slice(k_tile, (128, 128), (0, part)))
            matmul(
                result,
                self.a.next_read(),
                self.b.next_read(),
                init=part == 0,
            )

    @jit
    def compute_pv_block(self, p, v_tile, part):
        mem_copy(self.a.next_write(), p)
        mem_copy(
            self.b.next_write(), tile_slice(v_tile, (128, 128), (0, part)).permute(1, 0)
        )
        matmul(
            self.c.next_write(),
            self.a.next_read(),
            self.b.next_read(),
            init=True,
        )

    @jit
    def store(self, target):
        mem_copy(target.next_write(), self.c.next_read().partition(dim=0))


@cb.kernel(profile=cb.ProfileSpec(name="Qwen_Sparse_Attn", op_type="Qwen_Sparse_Attn"))
class QwenSparseAttnBlock128:
    def __init__(
        self,
        group,
        input_dtype,
        score_dtype,
        output_dtype,
        fp8=False,
        queries_per_tile=1,
    ):
        self.group = group
        self.fp8 = fp8
        self.queries_per_tile = queries_per_tile
        self.q_l1 = RingBuffer(
            MemLoc.L1,
            input_dtype,
            make_layout((32, 128)),
            depth=2 if queries_per_tile == 1 and group % 32 != 0 else 1,
            data_format="nz",
        )
        self.cube = QwenSparseAttnBlock128Cube(128, self.q_l1, input_dtype)
        self.vec = QwenSparseAttnBlock128Vector(score_dtype, output_dtype, 128, fp8)
        self.query_tail = Fp8QueryTail(128) if fp8 else QueryTail(input_dtype, 128)
        self.scores = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((16, 128)), depth=3
        )
        self.pv = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((16, 128)), depth=2)
        p_dtype = dtypes.float8_e4m3fn if fp8 else score_dtype
        self.p = RingBuffer(
            MemLoc.L1, p_dtype, make_layout((32, 128)), depth=2, data_format="nz"
        )

    def __call__(self, out, q, k, v, attn_mask, metadata, tasks, pages, scale):
        self._run(out, q, k, v, attn_mask, metadata, tasks, pages, scale)

    @jit
    def _stage_qk(self, q_tile, k, page_idx, n2_idx):
        self.cube.load_page(self.cube.k, k, page_idx, n2_idx)
        self.cube.compute_qk(q_tile)
        self.cube.store(self.scores)

    @jit
    def _stage_softmax(
        self,
        v,
        attn_mask,
        pages,
        page_row_idx,
        page_idx,
        n2_idx,
        scale,
        vec_idx,
        logical_row_base,
        active_rows,
        is_first,
    ):
        # 提前加载 V，使 MTE2 搬运与当前页的 Vector softmax 重叠。
        self.cube.load_page(self.cube.v, v, page_idx, n2_idx)
        # 一个 M32 tile 可以容纳多个 Query；每个 Query 的 mask 模板行按
        # 当前 vector core 实际持有的交集行数扩展。G=8 时自然得到 4 个 Query，
        # 但算法不依赖 4 或 8 这两个常量。
        for q_slot in tuple(range(self.queries_per_tile)):
            query_first_row = q_slot * self.group
            query_last_row = query_first_row + self.group
            vec_first_row = logical_row_base
            vec_last_row = logical_row_base + active_rows
            first_row = max(query_first_row, vec_first_row)
            last_row = min(query_last_row, vec_last_row)
            row_num = max(0, last_row - first_row)
            if row_num > 0:
                val_col = dtypes.int64(pages[q_slot + 1, page_row_idx])
                self.vec.load_mask(attn_mask, val_col)
                self.vec.expand_mask(first_row - logical_row_base, row_num)
        scores = self.scores.next_read()
        probability = from_tensor(
            self.vec.p_storage.next_write(), dtype=self.vec.p_dtype
        )
        if const_expr(self.fp8):
            if is_first:
                self.vec.softmax_fp8_first(scores, probability, scale, active_rows)
            else:
                self.vec.softmax_fp8_rest(
                    scores,
                    probability,
                    self.vec.alpha.next_write(),
                    scale,
                    active_rows,
                )
        else:
            if is_first:
                self.vec.softmax_first(scores, probability, scale, active_rows)
            else:
                self.vec.softmax_rest(
                    scores,
                    probability,
                    self.vec.alpha.next_write(),
                    scale,
                    active_rows,
                )
        probability_ready = from_tensor(
            self.vec.p_storage.next_read(), dtype=self.vec.p_dtype
        )
        mem_copy(
            self.p.next_write().partition(dim=0, part_id=vec_idx), probability_ready
        )

    @jit
    def _stage_pv(self):
        self.cube.compute_pv_block(self.p.next_read(), self.cube.v.next_read(), 0)
        self.cube.store(self.pv)

    @jit
    def _stage_update(self, active_rows, is_first):
        pv = self.pv.next_read()
        if is_first:
            self.vec.update_first(pv, 0, active_rows)
        else:
            self.vec.update_rest(pv, 0, self.vec.alpha.next_read(), active_rows)

    @jit
    def store_output(
        self,
        out,
        q_idx,
        n2_idx,
        g_chunk_idx,
        logical_row_base,
        active_rows,
        q_token_num,
        vec_idx,
        has_values,
    ):
        if has_values:
            if const_expr(self.fp8):
                self.vec.finalize_fp8(active_rows)
            else:
                self.vec.finalize(active_rows)
        else:
            self.vec.finalize_empty(active_rows)
        vec_sync_all()
        if const_expr(self.queries_per_tile > 1):
            for row in range(active_rows):
                packed_row = logical_row_base + row
                q_slot = packed_row // self.group
                g_idx = packed_row % self.group
                if q_slot < q_token_num:
                    mem_copy(
                        tile_slice(
                            _offset_view(
                                out[q_idx + q_slot, n2_idx * self.group + g_idx, None],
                                (0, 0, 0),
                            ),
                            (1, 128),
                            (0, 0),
                        ),
                        tile_slice(self.vec.out, (1, 128), (row, 0)),
                    )
        else:
            for row in range(active_rows):
                g_idx = g_chunk_idx * 32 + logical_row_base + row
                if g_idx < self.group:
                    n1_idx = n2_idx * self.group + g_idx
                    mem_copy(
                        tile_slice(
                            _offset_view(out[q_idx, n1_idx, None], (0, 0, 0)),
                            (1, 128),
                            (0, 0),
                        ),
                        tile_slice(self.vec.out, (1, 128), (row, 0)),
                    )

    @jit
    def _prefetch_next_q(self, q, tasks, task_idx, task_end, vec_idx):
        """Produce the next single-query tail Q tile into the second L1 slot."""
        if task_idx + 1 < task_end:
            next_q_idx = dtypes.int64(tasks[0, task_idx + 1])
            next_n2_idx = dtypes.int64(tasks[1, task_idx + 1])
            next_g_chunk_idx = dtypes.int64(tasks[2, task_idx + 1])
            next_page_num = dtypes.int64(tasks[4, task_idx + 1])
            next_remaining_rows = self.group - next_g_chunk_idx * 32
            next_chunk_rows = (
                32
                if next_remaining_rows > 32
                else (next_remaining_rows if next_remaining_rows > 0 else 0)
            )
            next_first_vec_rows = (next_chunk_rows + 1) // 2
            next_second_vec_rows = next_chunk_rows // 2
            next_active_rows = (
                next_first_vec_rows if vec_idx == 0 else next_second_vec_rows
            )
            next_logical_row_base = 0 if vec_idx == 0 else next_first_vec_rows
            if next_page_num > 0:
                self.query_tail.load_q_tail_tnd(
                    self.q_l1,
                    q,
                    next_q_idx,
                    next_n2_idx,
                    next_g_chunk_idx,
                    self.group,
                    vec_idx,
                    next_logical_row_base,
                    next_active_rows,
                )

    @jit
    def _run(
        self,
        out: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor,
        metadata: Tensor,
        tasks: Tensor,
        pages: Tensor,
        scale: float,
    ):
        block_idx = get_block_idx()
        vec_idx = get_subblock_id()
        task_start = dtypes.int64(metadata[0, block_idx])
        task_end = dtypes.int64(metadata[1, block_idx])
        for task_idx in range(task_start, task_end):
            q_idx = dtypes.int64(tasks[0, task_idx])
            n2_idx = dtypes.int64(tasks[1, task_idx])
            task_field = dtypes.int64(tasks[2, task_idx])
            page_list_offset = dtypes.int64(tasks[3, task_idx])
            page_num = dtypes.int64(tasks[4, task_idx])
            # 物理 M32 目标始终按 16+16 划分，但将有效逻辑行均匀映射到两个半区。
            # 加载 Q 和写回最终输出使用相同的 logicalRowBase，因此 QK/softmax/PV
            # 观察到的只是行置换；完整 M32 分块仍保持原始 16+16 映射。
            if const_expr(self.queries_per_tile > 1):
                q_token_num = task_field
                g_chunk_idx = 0
                tile_rows = q_token_num * self.group
                first_vec_rows = (tile_rows + 1) // 2
                second_vec_rows = tile_rows // 2
                active_rows = first_vec_rows if vec_idx == 0 else second_vec_rows
                logical_row_base = 0 if vec_idx == 0 else first_vec_rows
            else:
                q_token_num = 1
                g_chunk_idx = task_field
                remaining_rows = self.group - g_chunk_idx * 32
                chunk_rows = (
                    32
                    if remaining_rows > 32
                    else (remaining_rows if remaining_rows > 0 else 0)
                )
                first_vec_rows = (chunk_rows + 1) // 2
                second_vec_rows = chunk_rows // 2
                active_rows = first_vec_rows if vec_idx == 0 else second_vec_rows
                logical_row_base = 0 if vec_idx == 0 else first_vec_rows
            if page_num > 0:
                if const_expr(self.queries_per_tile > 1):
                    self.query_tail.load_q_packed_tnd(
                        self.q_l1,
                        q,
                        q_idx,
                        q_token_num,
                        n2_idx,
                        self.group,
                        vec_idx,
                        logical_row_base,
                        active_rows,
                    )
                elif const_expr(self.group % 32 == 0):
                    self.cube.load_q_tnd(q, q_idx, n2_idx, g_chunk_idx, self.group)
                else:
                    # Single-query tail Q tiles are prefetched one task ahead.
                    # Only the first task is loaded on demand here.
                    if task_idx == task_start:
                        self.query_tail.load_q_tail_tnd(
                            self.q_l1,
                            q,
                            q_idx,
                            n2_idx,
                            g_chunk_idx,
                            self.group,
                            vec_idx,
                            logical_row_base,
                            active_rows,
                        )
                q_tile = self.q_l1.next_read()
                # The second Channel slot carries the next single-query tail
                # tile across the task boundary without explicit notify/wait.
                if const_expr(self.queries_per_tile == 1 and self.group % 32 != 0):
                    self._prefetch_next_q(q, tasks, task_idx, task_end, vec_idx)
                for step in range(page_num + 3):
                    if step < page_num:
                        page_idx = dtypes.int64(pages[0, page_list_offset + step])
                        self._stage_qk(q_tile, k, page_idx, n2_idx)
                    if step >= 1 and step < page_num + 1:
                        page_row_idx = page_list_offset + step - 1
                        page_idx = dtypes.int64(pages[0, page_row_idx])
                        self._stage_softmax(
                            v,
                            attn_mask,
                            pages,
                            page_row_idx,
                            page_idx,
                            n2_idx,
                            scale,
                            vec_idx,
                            logical_row_base,
                            active_rows,
                            step == 1,
                        )
                    if step >= 2 and step < page_num + 2:
                        self._stage_pv()
                    if step >= 3:
                        self._stage_update(active_rows, step == 3)
            elif const_expr(self.queries_per_tile == 1 and self.group % 32 != 0):
                # No Q slot exists for an empty task.  It still prepares the
                # following non-empty task so FIFO production stays aligned.
                self._prefetch_next_q(q, tasks, task_idx, task_end, vec_idx)
            self.store_output(
                out,
                q_idx,
                n2_idx,
                g_chunk_idx,
                logical_row_base,
                active_rows,
                q_token_num,
                vec_idx,
                page_num > 0,
            )


@host
def launch_qwen_sparse_attn_block_128_bf16(
    out: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor,
    metadata: Tensor,
    tasks: Tensor,
    pages: Tensor,
    scale: float,
    group: Constexpr[int],
    queries_per_tile: Constexpr[int],
    used_core_num: Constexpr[int],
):
    QwenSparseAttnBlock128(
        group,
        dtypes.bfloat16,
        dtypes.bfloat16,
        dtypes.bfloat16,
        False,
        queries_per_tile,
    )[used_core_num](out, q, k, v, attn_mask, metadata, tasks, pages, scale)


@host
def launch_qwen_sparse_attn_block_128_fp16(
    out: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor,
    metadata: Tensor,
    tasks: Tensor,
    pages: Tensor,
    scale: float,
    group: Constexpr[int],
    queries_per_tile: Constexpr[int],
    used_core_num: Constexpr[int],
):
    QwenSparseAttnBlock128(
        group,
        dtypes.float16,
        dtypes.float16,
        dtypes.float16,
        False,
        queries_per_tile,
    )[used_core_num](out, q, k, v, attn_mask, metadata, tasks, pages, scale)


@host
def _launch_qwen_sparse_attn_block_128_fp8_fp16(
    out: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor,
    metadata: Tensor,
    tasks: Tensor,
    pages: Tensor,
    scale: float,
    group: Constexpr[int],
    queries_per_tile: Constexpr[int],
    used_core_num: Constexpr[int],
):
    QwenSparseAttnBlock128(
        group,
        dtypes.float8_e4m3fn,
        dtypes.float16,
        dtypes.float16,
        True,
        queries_per_tile,
    )[used_core_num](out, q, k, v, attn_mask, metadata, tasks, pages, scale)


@host
def _launch_qwen_sparse_attn_block_128_fp8_bf16(
    out: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor,
    metadata: Tensor,
    tasks: Tensor,
    pages: Tensor,
    scale: float,
    group: Constexpr[int],
    queries_per_tile: Constexpr[int],
    used_core_num: Constexpr[int],
):
    QwenSparseAttnBlock128(
        group,
        dtypes.float8_e4m3fn,
        dtypes.bfloat16,
        dtypes.bfloat16,
        True,
        queries_per_tile,
    )[used_core_num](out, q, k, v, attn_mask, metadata, tasks, pages, scale)


def launch_qwen_sparse_attn_block_128_fp8(output_dtype):
    """根据输出 dtype 选择内部 FP8 特化实现。"""
    if output_dtype == torch.float16:
        return _launch_qwen_sparse_attn_block_128_fp8_fp16
    if output_dtype == torch.bfloat16:
        return _launch_qwen_sparse_attn_block_128_fp8_bf16
    raise ValueError("FP8 output dtype must be FP16 or BF16")


def get_kernel(input_dtype, output_dtype, group, queries_per_tile, used_core_num):
    """Cache compiled executables by the required compile-time specialization.

    Token count, page count, head counts and metadata capacities remain
    symbolic in every template. The three integer launch parameters are
    Constexpr values and therefore must distinguish compiled kernels.
    """
    key = (input_dtype, output_dtype, group, queries_per_tile, used_core_num)
    with _COMPILED_KERNEL_LOCK:
        compiled = _COMPILED_KERNELS.get(key)
        if compiled is not None:
            return compiled
        return _compile_qwen_sparse_attn_locked(*key, key)


def _compile_qwen_sparse_attn_locked(
    input_dtype, output_dtype, group, queries_per_tile, used_core_num, key
):
    # The public wrapper passes torch.dtype values; opkit previously made this
    # conversion inside its cache decorator.
    torch_to_dsl = {
        torch.bfloat16: dtypes.bfloat16,
        torch.float16: dtypes.float16,
        torch.float8_e4m3fn: dtypes.float8_e4m3fn,
    }
    input_dsl = torch_to_dsl.get(input_dtype)
    output_dsl = torch_to_dsl.get(output_dtype)
    if input_dsl is None or output_dsl is None:
        raise TypeError("Unsupported Qwen sparse attention input/output dtype")
    tokens = cb.Dim("T", min=1, max=16384)
    query_heads = cb.Dim("N1", min=1, max=128 * 128)
    pages = cb.Dim("P", min=1)
    kv_heads = cb.Dim("N2", min=1, max=128)
    core_capacity = cb.Dim("C", min=used_core_num, max=64)
    task_capacity = cb.Dim("A", min=1)
    page_capacity = cb.Dim("L", min=1)
    cache_shape = (pages, kv_heads, 128, 128)
    cache_stride = (kv_heads * 128 * 128, 128, kv_heads * 128, 1)
    spec = cb.TensorSpec

    if input_dtype == torch.bfloat16:
        launch = launch_qwen_sparse_attn_block_128_bf16
    elif input_dtype == torch.float16:
        launch = launch_qwen_sparse_attn_block_128_fp16
    elif output_dtype == torch.float16:
        launch = _launch_qwen_sparse_attn_block_128_fp8_fp16
    elif output_dtype == torch.bfloat16:
        launch = _launch_qwen_sparse_attn_block_128_fp8_bf16
    else:
        raise ValueError("FP8 output dtype must be FP16 or BF16")
    compiled = cb.compile(
        launch,
        spec((tokens, query_heads, 128), output_dsl),
        spec((tokens, query_heads, 128), input_dsl),
        spec(cache_shape, input_dsl, stride=cache_stride),
        spec(cache_shape, input_dsl, stride=cache_stride),
        spec((2048, 2048), dtypes.int8),
        spec((2, core_capacity), dtypes.int64),
        spec((5, task_capacity), dtypes.int64),
        spec((queries_per_tile + 1, page_capacity), dtypes.int32),
        dtypes.float32,
        group,
        queries_per_tile,
        used_core_num,
    )
    # A failed compilation is not cached and can be retried on the next call.
    _COMPILED_KERNELS[key] = compiled
    return compiled


def _pa_bbnd_kernel_view(cache):
    """不复制 PA_BBND 存储，仅暴露 `[P,N2,BS,D]` stride 视图。"""
    return cache.permute(0, 2, 1, 3)


def _round_low_scalar(value, dtype):
    """与 score 特化路径的低精度 host 标量转换保持一致。"""
    return float(torch.tensor(float(value), dtype=torch.float32).to(dtype).float())


def qwen_sparse_attn_block_128(
    q,
    k,
    v,
    sparse_block_idx,
    sparse_block_count,
    block_shape,
    *,
    attn_mask=None,
    q_dequant_scale=None,
    k_dequant_scale=None,
    v_dequant_scale=None,
    p_quant_scale=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    block_table=None,
    metadata=None,
    is_packed_gqa=True,
    layout_q="TND",
    layout_kv="PA_BBND",
    softmax_scale=0.0,
    mask_mode=SparseMode.RIGHT_DOWN_CAUSAL,
    quant_mode=QuantMode.NO_QUANT,
    dst_type_max=0.0,
    softmax_precision=1,
    return_softmax_lse=False,
    attention_out_dtype=None,
):
    """支持 BF16/FP16/FP8 的 128-token 分块 Qwen 稀疏注意力。

    与官方 torch 扩展一致，返回 ``(attention_out, softmaxLse)``。由于 v1
    不支持 LSE，第二个张量始终为空的 FP32 NPU 张量。无效或未来位置的稀疏 ID
    会被跳过，重复 ID 保留其出现次数。
    """
    if not isinstance(q, torch.Tensor):
        raise TypeError("Q must be a torch.Tensor")
    validate_block128_attrs(
        block_shape,
        is_packed_gqa,
        layout_q,
        layout_kv,
        mask_mode,
        quant_mode,
        dst_type_max,
        softmax_precision,
        return_softmax_lse,
    )
    require_none(
        "QwenSparseAttnBlock128",
        q_dequant_scale=q_dequant_scale,
        k_dequant_scale=k_dequant_scale,
        v_dequant_scale=v_dequant_scale,
        p_quant_scale=p_quant_scale,
        cu_seqlens_kv=cu_seqlens_kv,
    )
    if cu_seqlens_q is None or seqused_kv is None or block_table is None:
        raise ValueError("Cu_seqlens_q, seqused_kv and block_table are required")
    mode = validate_block128_inputs(
        q,
        k,
        v,
        sparse_block_idx,
        sparse_block_count,
        cu_seqlens_q,
        seqused_kv,
        block_table,
        seqused_q,
        softmax_scale,
        quant_mode,
        attention_out_dtype,
    )
    validate_attn_mask(attn_mask, q, mask_mode)
    core_spans, tasks, pages, used_core_num = validate_block128_metadata(metadata, q)
    out = torch.empty(q.shape, dtype=mode["output"], device=q.device)
    softmax_lse = torch.empty((0,), dtype=torch.float32, device=q.device)
    if q.shape[0] == 0:
        return out, softmax_lse
    effective_scale = 128**-0.5 if float(softmax_scale) == 0.0 else float(softmax_scale)
    effective_scale = _round_low_scalar(effective_scale, mode["score"])
    # 当前 DSL ND2NZ 引擎在 head 轴位于 BS 与 D 之间时不会正确处理 PA_BBND
    # 行 stride。permute 仅创建 stride 视图，保持公共 PA_BBND 存储不变，
    # 不发生 host/device 数据复制。
    k_view = _pa_bbnd_kernel_view(k)
    v_view = _pa_bbnd_kernel_view(v)
    group = q.shape[1] // k.shape[2]
    queries_per_tile = pages.shape[0] - 1
    compiled = get_kernel(
        mode["input"], mode["output"], group, queries_per_tile, used_core_num
    )
    compiled(
        out,
        q,
        k_view,
        v_view,
        attn_mask,
        core_spans.to(q.device),
        tasks.to(q.device),
        pages.to(q.device),
        effective_scale,
    )
    return out, softmax_lse


# Public entry point under the sample's generic name.
qwen_sparse_attn = qwen_sparse_attn_block_128
