# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# fmt: off

"""Independent packed-QKV fused recurrent KDA decode operator.

The public input is one contiguous ``mixedqkv`` tensor in ``[Q(P) | K(P) |
V(P)]`` order with shape ``[B, S, 3 * P]``, where ``P = head_num *
head_dim``.  The current implementation requires ``head_dim_qk = head_dim_v
= 128``.  It does not import or call the base operator.
"""

# Public dispatcher signatures and device entry points must keep their ABI.
# Hardware stages and synchronization/movement order are fixed expansions;
# extracting helpers would change DSL lowering or measured kernel performance.

from __future__ import annotations

import threading
from typing import NamedTuple, Optional, Tuple

import cannbotdsl
import torch
from cannbotdsl import MemLoc, Tensor, dtypes, make_layout
from cannbotdsl.aot import export
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    update_mask,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdup,
    vdups,
    vexp,
    vload,
    vload_broadcast,
    vload_unpack,
    vmadd,
    vmul,
    vmuls,
    vneg,
    vreduce_sum,
    vsqrt,
    vstore,
    vstore_first,
    vstore_pack,
    vsub,
)
from cannbotdsl.ops.sync import onchip_rewind
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import idx2crd, reinterpret, tile_slice

SUPPORTED_HEAD_DIM = 128
VL = 64
D_SEGMENTS = SUPPORTED_HEAD_DIM // VL
DEFAULT_BLOCK_NUM = 56
MAX_AIV_CORES = 64
MAX_VALUE_HEADS = 96
SMALL_BATCH_HEAD_LIMIT = 16
MEDIUM_BATCH_HEAD_LIMIT = 32
SMALL_ROW_BLOCK = 32
MEDIUM_ROW_BLOCK = VL
_COMPILED_KERNELS: dict[object, object] = {}
_COMPILED_KERNEL_LOCK = threading.Lock()


@jit
def _cast_bf16_to_fp32_tile(dst, src):
    rows, cols = dst.shape
    src_stride = src.physical_stride[0]
    dst_stride = dst.physical_stride[0]
    with vf(mode="simd"):
        for row in range(rows):
            for col in range(0, cols, VL):
                mask, _ = update_mask(cols - col, elem_bits=32)
                value = vload_unpack(
                    src,
                    row * src_stride + col,
                    unpack_mode=UnpackMode.B16_TO_B32,
                )
                converted = vcast(value, dtypes.float32, mask=mask)
                vstore(dst, row * dst_stride + col, converted, mask)


@jit
def _cast_fp32_to_bf16_tile(dst, src):
    rows, cols = dst.shape
    src_stride = src.physical_stride[0]
    dst_stride = dst.physical_stride[0]
    with vf(mode="simd"):
        for row in range(rows):
            for col in range(0, cols, VL):
                mask, _ = update_mask(cols - col, elem_bits=32)
                value = vload(src, row * src_stride + col)
                converted = vcast(value, dtypes.bfloat16, mask=mask)
                vstore_pack(
                    dst,
                    row * dst_stride + col,
                    converted,
                    mask,
                    pack_mode=PackMode.B32_TO_B16,
                )


def _sequence_spec(dtype, physical_batches, heads, length, width, stride_prefix):
    stride_0 = cannbotdsl.Dim(f"{stride_prefix}_S0")
    stride_1 = cannbotdsl.Dim(f"{stride_prefix}_S1")
    stride_2 = cannbotdsl.Dim(f"{stride_prefix}_S2")
    return cannbotdsl.TensorSpec(
        (physical_batches, heads, length, width),
        dtype,
        stride=(stride_0, stride_1, stride_2, 1),
    )


def get_row_block_config(batch_heads: int, _seq_len: int) -> int:
    if batch_heads <= SMALL_BATCH_HEAD_LIMIT:
        return SMALL_ROW_BLOCK
    if batch_heads <= MEDIUM_BATCH_HEAD_LIMIT:
        return MEDIUM_ROW_BLOCK
    return SUPPORTED_HEAD_DIM


def _get_load_balance_config(
    batch_heads: int,
    sequence_length: int,
    vector_core_num: int,
) -> tuple[int, int]:
    row_block = get_row_block_config(batch_heads, sequence_length)
    total_items = batch_heads * (SUPPORTED_HEAD_DIM // row_block)
    core_num = min(vector_core_num, MAX_AIV_CORES, total_items)
    return row_block, core_num


def _require(condition: bool, message: str) -> None:
    """Raise the same validation error in optimized and regular Python."""
    if not condition:
        raise AssertionError(message)


def _device_block_num(ref: torch.Tensor) -> int:
    if ref.device.type not in {"npu", "privateuseone"}:
        return DEFAULT_BLOCK_NUM
    npu = getattr(torch, "npu", None)
    if npu is None:
        return DEFAULT_BLOCK_NUM
    device_index = ref.device.index
    if device_index is None:
        device_index = npu.current_device()
    stream = npu.current_stream(device_index)
    try:
        vector_core_num = int(
            cannbotdsl.get_platform_info(stream=stream).vector_core_num
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"NPU {device_index} does not expose a valid vector_core_num"
        ) from error
    if vector_core_num <= 0:
        raise RuntimeError(
            f"NPU {device_index} reports invalid vector_core_num={vector_core_num}"
        )
    return vector_core_num


class FusedRecurrentKDASnapshotVector:
    def __init__(
        self, row_block: int, dk: int, num_heads: int, state_dtype=dtypes.bfloat16
    ):
        self.row_block = row_block
        self.dk = dk
        self.state_is_bf16 = state_dtype == dtypes.bfloat16
        value_width = max(row_block, VL)
        state_depth = 1 if row_block == SUPPORTED_HEAD_DIM else 2
        self.state = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((row_block, dk)), depth=state_depth)
        self.key = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, dk)))
        self.query = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, dk)))
        self.decay_exp = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, dk)))
        self.ub_value = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, value_width)))
        self.ub_beta = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, VL)))
        self.out = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, row_block)))
        if self.state_is_bf16:
            self.state_bf16 = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((row_block, dk)), depth=2)
            self.state_bf16_output = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((row_block, dk)), depth=2)
        else:
            self.state_snapshot = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((row_block, dk)), depth=2)
        self.value = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, max(row_block, VL))), depth=2)
        self.output = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, row_block)), depth=2)
        self.qk_norm_sums = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, D_SEGMENTS * 2)))
        self.state_key_sums = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, value_width)))
        self.delta_row = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, value_width)))
        self.raw_query = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, dk)), depth=2)
        self.raw_key = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, dk)), depth=2)
        self.raw_g = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, dk)), depth=2)
        self.raw_beta = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, VL)), depth=2)
        self.dt_bias = Buffer(MemLoc.UB, dtypes.float32, make_layout((num_heads, dk)))
        self.a_log = Buffer(MemLoc.UB, dtypes.float32, make_layout((num_heads,)))
        # Produced lazily to preserve the channel epoch and buffer lifetime.
        self.state_tensor: object
        self.snapshot_tensor: object

    def load_state(self, gm_state_tile, active_rows=None):
        if active_rows is None:
            active_rows = self.row_block
        self.state_tensor = self.state.next_write()
        state = self.state_tensor
        if active_rows < self.row_block:
            state = reinterpret(state, offset=0, layout=make_layout((active_rows, self.dk)))
        if self.state_is_bf16:
            state_bf16 = self.state_bf16.next_write()
            mem_copy(state_bf16, gm_state_tile)
            _cast_bf16_to_fp32_tile(state, state_bf16)
        else:
            mem_copy(state, gm_state_tile)

    def load_qk(self, raw_key_gm, raw_query_gm):
        mem_copy(self.raw_query.next_write(), raw_query_gm)
        mem_copy(self.raw_key.next_write(), raw_key_gm)

    def prefetch_verify_inputs(
        self,
        raw_key_gm,
        raw_query_gm,
        raw_g_gm,
        value_gm,
        raw_beta_gm,
    ):
        """Issue one Verify token's Q/K/G/BETA/V loads into depth-2 channels."""
        self.load_qk(raw_key_gm, raw_query_gm)
        self.load_g_beta_value(raw_g_gm, value_gm, raw_beta_gm)

    @jit
    def normalize_qk(self, scale_value: float):
        raw_query = self.raw_query.next_read()
        raw_key = self.raw_key.next_read()
        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            query_raw_pre = vload_unpack(
                raw_query, 0, unpack_mode=UnpackMode.B16_TO_B32
            )
            query_raw_post = vload_unpack(
                raw_query, VL, unpack_mode=UnpackMode.B16_TO_B32
            )
            key_raw_pre = vload_unpack(raw_key, 0, unpack_mode=UnpackMode.B16_TO_B32)
            key_raw_post = vload_unpack(raw_key, VL, unpack_mode=UnpackMode.B16_TO_B32)
            query_pre = vcast(query_raw_pre, dtypes.float32, mask=full)
            query_post = vcast(query_raw_post, dtypes.float32, mask=full)
            key_pre = vcast(key_raw_pre, dtypes.float32, mask=full)
            key_post = vcast(key_raw_post, dtypes.float32, mask=full)
            query_square_pre = vmul(query_pre, query_pre, mask=full)
            query_square_post = vmul(query_post, query_post, mask=full)
            key_square_pre = vmul(key_pre, key_pre, mask=full)
            key_square_post = vmul(key_post, key_post, mask=full)
            query_sum_pre = vreduce_sum(query_square_pre, mask=full)
            query_sum_post = vreduce_sum(query_square_post, mask=full)
            key_sum_pre = vreduce_sum(key_square_pre, mask=full)
            key_sum_post = vreduce_sum(key_square_post, mask=full)
            vstore(self.query, 0, query_pre, full)
            vstore(self.query, VL, query_post, full)
            vstore(self.key, 0, key_pre, full)
            vstore(self.key, VL, key_post, full)
            vstore_first(self.qk_norm_sums, 0, query_sum_pre)
            vstore_first(self.qk_norm_sums, 1, query_sum_post)
            vstore_first(self.qk_norm_sums, D_SEGMENTS, key_sum_pre)
            vstore_first(self.qk_norm_sums, D_SEGMENTS + 1, key_sum_post)

        with vf(mode="simd"):
            lane0, _ = update_mask(1, elem_bits=32)
            query_sum_pre = vload_broadcast(self.qk_norm_sums, 0)
            query_sum_post = vload_broadcast(self.qk_norm_sums, 1)
            key_sum_pre = vload_broadcast(self.qk_norm_sums, D_SEGMENTS)
            key_sum_post = vload_broadcast(self.qk_norm_sums, D_SEGMENTS + 1)
            query_sum = vadd(query_sum_pre, query_sum_post, mask=lane0)
            key_sum = vadd(key_sum_pre, key_sum_post, mask=lane0)
            query_sum_eps = vadds(query_sum, 1e-6, mask=lane0)
            key_sum_eps = vadds(key_sum, 1e-6, mask=lane0)
            query_root = vsqrt(query_sum_eps, mask=lane0)
            key_root = vsqrt(key_sum_eps, mask=lane0)
            vstore_first(self.qk_norm_sums, 0, query_root)
            vstore_first(self.qk_norm_sums, 1, key_root)

        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            query_root = vload_broadcast(self.qk_norm_sums, 0)
            key_root = vload_broadcast(self.qk_norm_sums, 1)
            query_denom = vdup(query_root, mask=full)
            key_denom = vdup(key_root, mask=full)
            query_pre = vload(self.query, 0)
            query_post = vload(self.query, VL)
            key_pre = vload(self.key, 0)
            key_post = vload(self.key, VL)
            normalized_query_pre = vdiv(query_pre, query_denom, mask=full)
            normalized_query_post = vdiv(query_post, query_denom, mask=full)
            normalized_key_pre = vdiv(key_pre, key_denom, mask=full)
            normalized_key_post = vdiv(key_post, key_denom, mask=full)
            scaled_query_pre = vmuls(normalized_query_pre, scale_value, mask=full)
            scaled_query_post = vmuls(normalized_query_post, scale_value, mask=full)
            vstore(self.query, 0, scaled_query_pre, full)
            vstore(self.query, VL, scaled_query_post, full)
            vstore(self.key, 0, normalized_key_pre, full)
            vstore(self.key, VL, normalized_key_post, full)

    def load_gate_params(self, a_log_gm, dt_bias_gm, value_heads=None):
        if value_heads is None:
            mem_copy(self.a_log, a_log_gm)
            mem_copy(self.dt_bias, dt_bias_gm)
        else:
            mem_copy(self.a_log, a_log_gm)
            mem_copy(reinterpret(self.dt_bias, offset=0, layout=make_layout((value_heads, self.dk))), dt_bias_gm)

    def load_g_beta_value(self, raw_g_gm, value_gm, raw_beta_gm):
        mem_copy(self.raw_g.next_write(), raw_g_gm)
        mem_copy(self.raw_beta.next_write(), raw_beta_gm)
        mem_copy(self.value.next_write(), value_gm)

    @jit
    def activate_g_beta(self, lower_bound: float, value_head, active_rows=None):
        if active_rows is None:
            active_rows = self.row_block
        raw_g = self.raw_g.next_read()
        raw_beta = self.raw_beta.next_read()
        value = self.value.next_read()
        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            one = vdups(1.0, dtypes.float32)
            a_log = vload_broadcast(self.a_log, value_head)
            alpha = vexp(a_log, mask=full)
            negative_alpha = vneg(alpha, mask=full)
            raw_gate_pre = vload_unpack(raw_g, 0, unpack_mode=UnpackMode.B16_TO_B32)
            raw_gate_post = vload_unpack(raw_g, VL, unpack_mode=UnpackMode.B16_TO_B32)
            gate_value_pre = vcast(raw_gate_pre, dtypes.float32, mask=full)
            gate_value_post = vcast(raw_gate_post, dtypes.float32, mask=full)
            dt_bias_offset = value_head * self.dk
            dt_bias_pre = vload(self.dt_bias, dt_bias_offset)
            dt_bias_post = vload(self.dt_bias, dt_bias_offset + VL)
            gate_input_pre = vadd(gate_value_pre, dt_bias_pre, mask=full)
            gate_input_post = vadd(gate_value_post, dt_bias_post, mask=full)
            gate_pre = vmul(negative_alpha, gate_input_pre, mask=full)
            gate_post = vmul(negative_alpha, gate_input_post, mask=full)
            gate_exp_pre = vexp(gate_pre, mask=full)
            gate_exp_post = vexp(gate_post, mask=full)
            gate_denom_pre = vadds(gate_exp_pre, 1.0, mask=full)
            gate_denom_post = vadds(gate_exp_post, 1.0, mask=full)
            gate_sigmoid_pre = vdiv(one, gate_denom_pre, mask=full)
            gate_sigmoid_post = vdiv(one, gate_denom_post, mask=full)
            activated_gate_pre = vmuls(gate_sigmoid_pre, lower_bound, mask=full)
            activated_gate_post = vmuls(gate_sigmoid_post, lower_bound, mask=full)
            # Cache final decay while activation values are live so the hot
            # recurrence path does not evaluate exp again.
            decay_exp_pre = vexp(activated_gate_pre, mask=full)
            decay_exp_post = vexp(activated_gate_post, mask=full)
            vstore(self.decay_exp, 0, decay_exp_pre, full)
            vstore(self.decay_exp, VL, decay_exp_post, full)

            raw_beta_value = vload_unpack(
                raw_beta, 0, unpack_mode=UnpackMode.B16_TO_B32
            )
            beta_logit = vcast(raw_beta_value, dtypes.float32, mask=full)
            beta_neg_logit = vneg(beta_logit, mask=full)
            beta_exp = vexp(beta_neg_logit, mask=full)
            beta_denom = vadds(beta_exp, 1.0, mask=full)
            activated_beta = vdiv(one, beta_denom, mask=full)
            vstore(self.ub_beta, 0, activated_beta, full)
            value_raw = vload_unpack(value, 0, unpack_mode=UnpackMode.B16_TO_B32)
            value_f32 = vcast(value_raw, dtypes.float32, mask=full)
            vstore(self.ub_value, 0, value_f32, full)
            if active_rows > VL:
                value_raw_post = vload_unpack(
                    value, VL, unpack_mode=UnpackMode.B16_TO_B32
                )
                value_f32_post = vcast(value_raw_post, dtypes.float32, mask=full)
                vstore(self.ub_value, VL, value_f32_post, full)

    @jit
    def recur_step(self, write_snapshot: bool = True, active_rows=None):
        row_block, dk = self.row_block, self.dk
        if active_rows is not None:
            row_block = active_rows
        state, out = self.state_tensor, self.out
        if write_snapshot and not self.state_is_bf16:
            self.snapshot_tensor = self.state_snapshot.next_write()
        key, query = self.key, self.query
        state_key_sums, delta_row = self.state_key_sums, self.delta_row

        with vf(mode="simd"):
            mask, _ = update_mask(VL, elem_bits=32)
            decay_pre = vload(self.decay_exp, 0)
            decay_post = vload(self.decay_exp, VL)
            key_pre = vload(key, 0)
            key_post = vload(key, VL)
            decay_key_pre = vmul(decay_pre, key_pre, mask=mask)
            decay_key_post = vmul(decay_post, key_post, mask=mask)
            for dv in dsl_range(0, row_block, 1, unroll=4):
                offset_pre = dv * dk
                offset_post = offset_pre + VL
                state_pre = vload(state, offset_pre)
                state_post = vload(state, offset_post)
                product_pre = vmul(state_pre, decay_key_pre, mask=mask)
                product = vmadd(state_post, decay_key_post, product_pre, mask=mask)
                state_key_sum = vreduce_sum(product, mask=mask)
                vstore_first(state_key_sums, dv, state_key_sum)

        with vf(mode="simd"):
            mask, _ = update_mask(VL, elem_bits=32)
            beta_brc = vload_broadcast(self.ub_beta, 0)
            state_key_vec = vload(state_key_sums, 0)
            value_vec = vload(self.ub_value, 0)
            residual_vec = vsub(value_vec, state_key_vec, mask=mask)
            delta_vec = vmul(residual_vec, beta_brc, mask=mask)
            vstore(delta_row, 0, delta_vec, mask)
            if row_block > VL:
                state_key_vec_post = vload(state_key_sums, VL)
                value_vec_post = vload(self.ub_value, VL)
                residual_vec_post = vsub(value_vec_post, state_key_vec_post, mask=mask)
                delta_vec_post = vmul(residual_vec_post, beta_brc, mask=mask)
                vstore(delta_row, VL, delta_vec_post, mask)

        with vf(mode="simd"):
            mask, _ = update_mask(VL, elem_bits=32)
            decay_pre = vload(self.decay_exp, 0)
            decay_post = vload(self.decay_exp, VL)
            key_pre = vload(key, 0)
            key_post = vload(key, VL)
            query_pre = vload(query, 0)
            query_post = vload(query, VL)
            for dv in dsl_range(0, row_block, 1, unroll=1):
                offset_pre = dv * dk
                offset_post = offset_pre + VL
                state_pre = vload(state, offset_pre)
                state_post = vload(state, offset_post)
                delta = vload_broadcast(delta_row, dv)
                delta_key_pre = vmul(delta, key_pre, mask=mask)
                delta_key_post = vmul(delta, key_post, mask=mask)
                state_new_pre = vmadd(state_pre, decay_pre, delta_key_pre, mask=mask)
                state_new_post = vmadd(
                    state_post, decay_post, delta_key_post, mask=mask
                )
                vstore(state, offset_pre, state_new_pre, mask)
                vstore(state, offset_post, state_new_post, mask)
                if write_snapshot and not self.state_is_bf16:
                    vstore(self.snapshot_tensor, offset_pre, state_new_pre, mask)
                    vstore(self.snapshot_tensor, offset_post, state_new_post, mask)
                output_pre = vmul(state_new_pre, query_pre, mask=mask)
                output = vmadd(state_new_post, query_post, output_pre, mask=mask)
                output_sum = vreduce_sum(output, mask=mask)
                vstore_first(out, dv, output_sum)

    def store_output(self, gm_out_tile, active_rows=None):
        if active_rows is None:
            active_rows = self.row_block
        output = self.output.next_write()
        out = self.out
        if active_rows < self.row_block:
            output = reinterpret(output, offset=0, layout=make_layout((1, active_rows)))
        _cast_fp32_to_bf16_tile(output, out)
        mem_copy(gm_out_tile, output)

    def store_state(self, gm_state_tile, active_rows=None):
        if active_rows is None:
            active_rows = self.row_block
        if self.state_is_bf16:
            state_bf16_output = self.state_bf16_output.next_write()
            state = self.state_tensor
            if active_rows < self.row_block:
                state_bf16_output = reinterpret(state_bf16_output, offset=0, layout=make_layout((active_rows, self.dk)))
            _cast_fp32_to_bf16_tile(state_bf16_output, state)
            mem_copy(gm_state_tile, state_bf16_output)
        else:
            state_snapshot = self.state_snapshot.next_read()
            mem_copy(gm_state_tile, state_snapshot[:active_rows, :self.dk])


@jit
def _run_recurrent_item(
    vector,
    gm_mixedqkv,
    gm_g,
    gm_beta,
    gm_out,
    gm_initial_state,
    gm_final_state,
    gm_state_indices,
    gm_accepted_tokens,
    gm_cu_seqlens,
    scale,
    lower_bound,
    layout_mode,
    batch_size,
    sequence_length,
    value_heads_dim,
    item,
    *,
    active_rows: int,
    row_index,
):
    head_dim = SUPPORTED_HEAD_DIM
    batch_index, value_head = idx2crd(item, [batch_size, value_heads_dim])
    storage_batch_index = batch_index
    token_start = batch_index * sequence_length
    token_end = token_start + sequence_length
    if layout_mode == 1:
        storage_batch_index = 0
        token_start = gm_cu_seqlens[batch_index]
        token_end = gm_cu_seqlens[batch_index + 1]

    accepted = gm_accepted_tokens[batch_index]
    initial_token_index = token_start + accepted - 1 if accepted > 0 else token_start
    initial_state_index = gm_state_indices[initial_token_index]
    vector.load_state(
        tile_slice(
            gm_initial_state[initial_state_index, value_head, None, None],
            (active_rows, head_dim),
            (row_index, 0),
        ),
        active_rows=active_rows,
    )

    first_storage_token = 0
    if layout_mode == 1:
        first_storage_token = token_start
    vector.prefetch_verify_inputs(
        tile_slice(
            gm_mixedqkv[storage_batch_index, value_heads_dim + value_head, None, None],
            (1, head_dim),
            (first_storage_token, 0),
        ),
        tile_slice(
            gm_mixedqkv[storage_batch_index, value_head, None, None],
            (1, head_dim),
            (first_storage_token, 0),
        ),
        tile_slice(
            gm_g[storage_batch_index, value_head, None, None],
            (1, head_dim),
            (first_storage_token, 0),
        ),
        tile_slice(
            gm_mixedqkv[
                storage_batch_index, 2 * value_heads_dim + value_head, None, None
            ],
            (1, active_rows),
            (first_storage_token, row_index),
        ),
        tile_slice(
            gm_beta[storage_batch_index, value_head, None, None],
            (1, 1),
            (first_storage_token, 0),
        ),
    )

    for token_index in range(token_start, token_end, 1):
        storage_token_index = token_index - token_start
        if layout_mode == 1:
            storage_token_index = token_index
        vector.normalize_qk(scale)
        vector.activate_g_beta(lower_bound, value_head, active_rows=active_rows)
        if token_index + 1 < token_end:
            next_token_index = token_index + 1
            next_storage_token = next_token_index - token_start
            if layout_mode == 1:
                next_storage_token = next_token_index
            vector.prefetch_verify_inputs(
                tile_slice(
                    gm_mixedqkv[
                        storage_batch_index, value_heads_dim + value_head, None, None
                    ],
                    (1, head_dim),
                    (next_storage_token, 0),
                ),
                tile_slice(
                    gm_mixedqkv[storage_batch_index, value_head, None, None],
                    (1, head_dim),
                    (next_storage_token, 0),
                ),
                tile_slice(
                    gm_g[storage_batch_index, value_head, None, None],
                    (1, head_dim),
                    (next_storage_token, 0),
                ),
                tile_slice(
                    gm_mixedqkv[
                        storage_batch_index,
                        2 * value_heads_dim + value_head,
                        None,
                        None,
                    ],
                    (1, active_rows),
                    (next_storage_token, row_index),
                ),
                tile_slice(
                    gm_beta[storage_batch_index, value_head, None, None],
                    (1, 1),
                    (next_storage_token, 0),
                ),
            )
        vector.recur_step(active_rows=active_rows)
        vector.store_output(
            tile_slice(
                gm_out[storage_batch_index, value_head, None, None],
                (1, active_rows),
                (storage_token_index, row_index),
            ),
            active_rows=active_rows,
        )
        state_index = gm_state_indices[token_index]
        vector.store_state(
            tile_slice(
                gm_final_state[state_index, value_head, None, None],
                (active_rows, head_dim),
                (row_index, 0),
            ),
            active_rows=active_rows,
        )


@jit
def _run_minimal_split_recurrent_body(
    gm_mixedqkv,
    gm_g,
    gm_beta,
    gm_a_log,
    gm_dt_bias,
    gm_out,
    gm_initial_state,
    gm_final_state,
    gm_state_indices,
    gm_accepted_tokens,
    gm_cu_seqlens,
    scale,
    lower_bound,
    layout_mode,
    batch_size,
    sequence_length,
    *,
    state_dtype,
):
    value_heads_dim = gm_g.shape[1]
    vector = FusedRecurrentKDASnapshotVector(
        SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM, MAX_VALUE_HEADS, state_dtype=state_dtype
    )
    vector.load_gate_params(gm_a_log, gm_dt_bias, value_heads_dim)

    core = get_block_idx()
    core_num = get_block_num()
    batch_heads = batch_size * value_heads_dim
    total_half_rows = 2 * batch_heads
    base_half_rows = total_half_rows // core_num
    extra_cores = total_half_rows - base_half_rows * core_num
    base_full_heads = base_half_rows // 2
    extra_full_heads = (base_half_rows + 1) // 2
    extra_full_delta = extra_full_heads - base_full_heads
    full_head_count = base_full_heads
    if core < extra_cores:
        full_head_count = extra_full_heads
    prefix_extra = core if core < extra_cores else extra_cores
    full_head_start = core * base_full_heads + prefix_extra * extra_full_delta

    # Balance 64-row units first, then pair adjacent units into complete heads.
    # Only cores with an odd unit count receive half of a split head.  This
    # gives the minimum possible number of split heads for the balanced load.
    for full_offset in range(full_head_count):
        _run_recurrent_item(
            vector,
            gm_mixedqkv,
            gm_g,
            gm_beta,
            gm_out,
            gm_initial_state,
            gm_final_state,
            gm_state_indices,
            gm_accepted_tokens,
            gm_cu_seqlens,
            scale,
            lower_bound,
            layout_mode,
            batch_size,
            sequence_length,
            value_heads_dim,
            full_head_start + full_offset,
            active_rows=SUPPORTED_HEAD_DIM,
            row_index=0,
        )

    full_heads = core_num * base_full_heads + extra_cores * extra_full_delta
    if base_half_rows % 2 == 0 and core < extra_cores:
        _run_recurrent_item(
            vector,
            gm_mixedqkv,
            gm_g,
            gm_beta,
            gm_out,
            gm_initial_state,
            gm_final_state,
            gm_state_indices,
            gm_accepted_tokens,
            gm_cu_seqlens,
            scale,
            lower_bound,
            layout_mode,
            batch_size,
            sequence_length,
            value_heads_dim,
            full_heads + core // 2,
            active_rows=VL,
            row_index=core % 2,
        )
    if base_half_rows % 2 == 1 and core >= extra_cores:
        half_rank = core - extra_cores
        _run_recurrent_item(
            vector,
            gm_mixedqkv,
            gm_g,
            gm_beta,
            gm_out,
            gm_initial_state,
            gm_final_state,
            gm_state_indices,
            gm_accepted_tokens,
            gm_cu_seqlens,
            scale,
            lower_bound,
            layout_mode,
            batch_size,
            sequence_length,
            value_heads_dim,
            full_heads + half_rank // 2,
            active_rows=VL,
            row_index=half_rank % 2,
        )


@jit
def _run_recurrent_body(
    gm_mixedqkv,
    gm_g,
    gm_beta,
    gm_a_log,
    gm_dt_bias,
    gm_out,
    gm_initial_state,
    gm_final_state,
    gm_state_indices,
    gm_accepted_tokens,
    gm_cu_seqlens,
    scale,
    lower_bound,
    layout_mode,
    batch_size,
    sequence_length,
    *,
    row_block: int,
    state_dtype,
):
    head_dim = SUPPORTED_HEAD_DIM
    num_row_blocks = head_dim // row_block
    value_heads_dim = gm_g.shape[1]
    vector = FusedRecurrentKDASnapshotVector(
        row_block, head_dim, MAX_VALUE_HEADS, state_dtype=state_dtype
    )
    total_items = batch_size * value_heads_dim * num_row_blocks
    logical_core_num = get_block_num()
    if logical_core_num > total_items:
        logical_core_num = total_items
    items_per_core = (total_items + logical_core_num - 1) // logical_core_num
    item_start = get_block_idx() * items_per_core
    item_end = item_start + items_per_core
    if item_end > total_items:
        item_end = total_items

    if get_block_idx() < logical_core_num:
        vector.load_gate_params(gm_a_log, gm_dt_bias, value_heads_dim)
        for item in range(item_start, item_end):
            head_item = item // num_row_blocks
            row_index = item - head_item * num_row_blocks
            _run_recurrent_item(
                vector,
                gm_mixedqkv,
                gm_g,
                gm_beta,
                gm_out,
                gm_initial_state,
                gm_final_state,
                gm_state_indices,
                gm_accepted_tokens,
                gm_cu_seqlens,
                scale,
                lower_bound,
                layout_mode,
                batch_size,
                sequence_length,
                value_heads_dim,
                head_item,
                active_rows=row_block,
                row_index=row_index,
            )


def _rewind_row_block_branch() -> None:
    onchip_rewind(reset_sync_id=True)


@kernel
class FusedRecurrentKDASnapshotKernel:
    def __init__(self, state_dtype=dtypes.float32):
        self.state_dtype = state_dtype

    def __call__(
        self,
        gm_mixedqkv: Tensor,
        gm_g: Tensor,
        gm_beta: Tensor,
        gm_a_log: Tensor,
        gm_dt_bias: Tensor,
        gm_out: Tensor,
        gm_initial_state: Tensor,
        gm_final_state: Tensor,
        gm_state_indices: Tensor,
        gm_accepted_tokens: Tensor,
        gm_cu_seqlens: Tensor,
        scale: float,
        lower_bound: float,
        layout_mode: int,
        batch_size: int,
        sequence_length: int,
    ):
        value_heads_dim = gm_g.shape[1]
        if batch_size * value_heads_dim > MEDIUM_BATCH_HEAD_LIMIT:
            _rewind_row_block_branch()
            _run_minimal_split_recurrent_body(
                gm_mixedqkv,
                gm_g,
                gm_beta,
                gm_a_log,
                gm_dt_bias,
                gm_out,
                gm_initial_state,
                gm_final_state,
                gm_state_indices,
                gm_accepted_tokens,
                gm_cu_seqlens,
                scale,
                lower_bound,
                layout_mode,
                batch_size,
                sequence_length,
                state_dtype=self.state_dtype,
            )
        elif batch_size * value_heads_dim <= SMALL_BATCH_HEAD_LIMIT:
            _rewind_row_block_branch()
            _run_recurrent_body(
                gm_mixedqkv,
                gm_g,
                gm_beta,
                gm_a_log,
                gm_dt_bias,
                gm_out,
                gm_initial_state,
                gm_final_state,
                gm_state_indices,
                gm_accepted_tokens,
                gm_cu_seqlens,
                scale,
                lower_bound,
                layout_mode,
                batch_size,
                sequence_length,
                row_block=SMALL_ROW_BLOCK,
                state_dtype=self.state_dtype,
            )
        else:
            _rewind_row_block_branch()
            _run_recurrent_body(
                gm_mixedqkv,
                gm_g,
                gm_beta,
                gm_a_log,
                gm_dt_bias,
                gm_out,
                gm_initial_state,
                gm_final_state,
                gm_state_indices,
                gm_accepted_tokens,
                gm_cu_seqlens,
                scale,
                lower_bound,
                layout_mode,
                batch_size,
                sequence_length,
                row_block=MEDIUM_ROW_BLOCK,
                state_dtype=self.state_dtype,
            )


class FusedRecurrentKDASnapshot:
    """One dynamic Snapshot provider for one state dtype."""

    def __init__(self, state_dtype):
        self.state_dtype = state_dtype

    @host
    def run(
        self,
        mixedqkv: Tensor,
        raw_g: Tensor,
        raw_beta: Tensor,
        a_log: Tensor,
        dt_bias: Tensor,
        out: Tensor,
        initial_state: Tensor,
        final_state: Tensor,
        state_indices: Tensor,
        accepted_tokens: Tensor,
        cu_seqlens: Tensor,
        scale: float,
        lower_bound: float,
        layout_mode: int,
        batch_size: int,
        sequence_length: int,
        core_num: int,
    ):
        op = FusedRecurrentKDASnapshotKernel(self.state_dtype)
        op[core_num](
            mixedqkv,
            raw_g,
            raw_beta,
            a_log,
            dt_bias,
            out,
            initial_state,
            final_state,
            state_indices,
            accepted_tokens,
            cu_seqlens,
            scale,
            lower_bound,
            layout_mode,
            batch_size,
            sequence_length,
        )


class _SnapshotCompileSpecs(NamedTuple):
    mixedqkv: object
    raw_g: object
    raw_beta: object
    a_log: object
    dt_bias: object
    out: object
    initial_state: object
    final_state: object
    state_indices: object
    accepted_tokens: object
    cu_seqlens: object
    scale: object
    lower_bound: object
    layout_mode: object
    batch_size: object
    sequence_length: object
    core_num: object


def _snapshot_compile_specs(state_dtype):
    physical_batches = cannbotdsl.Dim("PB")
    storage_length = cannbotdsl.Dim("L")
    packed_heads = cannbotdsl.Dim("N3")
    value_heads = cannbotdsl.Dim("N", min=1, max=MAX_VALUE_HEADS)
    state_pool = cannbotdsl.Dim("POOL")
    token_count = cannbotdsl.Dim("TOKENS")
    batch = cannbotdsl.Dim("B")
    cu_size = cannbotdsl.Dim("CU")
    tensor_spec = cannbotdsl.TensorSpec
    sequence_args = (physical_batches, value_heads, storage_length)
    mixed_spec = _sequence_spec(
        dtypes.bfloat16,
        physical_batches,
        packed_heads,
        storage_length,
        SUPPORTED_HEAD_DIM,
        "MIXED",
    )
    g_spec = _sequence_spec(dtypes.bfloat16, *sequence_args, SUPPORTED_HEAD_DIM, "G")
    beta_spec = _sequence_spec(dtypes.bfloat16, *sequence_args, 1, "BETA")
    out_spec = _sequence_spec(
        dtypes.bfloat16, *sequence_args, SUPPORTED_HEAD_DIM, "OUT"
    )
    state_spec = tensor_spec(
        (state_pool, value_heads, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), state_dtype
    )
    return _SnapshotCompileSpecs(
        mixed_spec,
        g_spec,
        beta_spec,
        tensor_spec((value_heads,), dtypes.float32),
        tensor_spec((value_heads, SUPPORTED_HEAD_DIM), dtypes.float32),
        out_spec,
        state_spec,
        state_spec,
        tensor_spec((token_count,), dtypes.int32),
        tensor_spec((batch,), dtypes.int32),
        tensor_spec((cu_size,), dtypes.int32),
        dtypes.float32,
        dtypes.float32,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
    )


def _get_compiled_kernel(state_dtype):
    with _COMPILED_KERNEL_LOCK:
        compiled = _COMPILED_KERNELS.get(state_dtype)
        if compiled is not None:
            return compiled
        compiled = cannbotdsl.compile(
            FusedRecurrentKDASnapshot(state_dtype).run,
            *_snapshot_compile_specs(state_dtype),
        )
        _COMPILED_KERNELS[state_dtype] = compiled
        return compiled


class _SnapshotLayout(NamedTuple):
    batch: int
    storage_length: int
    token_count: int
    mixed: torch.Tensor
    gate: torch.Tensor
    beta: torch.Tensor
    output: torch.Tensor
    output_logical: torch.Tensor
    mode: int


def _validate_snapshot_common(mixedqkv, state, beta, gate, a_log, dt_bias, lower):
    _require(state.dim() == 4, "state must be rank-4")
    _require(
        state.dtype in (torch.bfloat16, torch.float32),
        f"state must be bf16 or fp32, got {state.dtype}",
    )
    _require(state.is_contiguous(), "state must be contiguous")
    state_pool, num_heads, state_dv, state_dk = map(int, state.shape)
    _require(
        state_dv == state_dk == SUPPORTED_HEAD_DIM,
        f"state must have trailing shape ({SUPPORTED_HEAD_DIM}, {SUPPORTED_HEAD_DIM})",
    )
    _require(
        1 <= num_heads <= MAX_VALUE_HEADS,
        f"number of heads must be in [1, {MAX_VALUE_HEADS}], got {num_heads}",
    )
    dtype_checks = (
        (
            mixedqkv.dtype == torch.bfloat16,
            f"mixedqkv must be bf16, got {mixedqkv.dtype}",
        ),
        (gate.dtype == torch.bfloat16, f"g must be bf16, got {gate.dtype}"),
        (beta.dtype == torch.bfloat16, f"beta must be bf16, got {beta.dtype}"),
        (a_log.dtype == torch.float32, f"A_log must be fp32, got {a_log.dtype}"),
        (dt_bias.dtype == torch.float32, f"dt_bias must be fp32, got {dt_bias.dtype}"),
    )
    for condition, message in dtype_checks:
        _require(condition, message)
    _require(
        tuple(a_log.shape) == (num_heads,), f"A_log must have shape ({num_heads},)"
    )
    _require(
        tuple(dt_bias.shape) == (num_heads, SUPPORTED_HEAD_DIM),
        f"dt_bias must have shape ({num_heads}, {SUPPORTED_HEAD_DIM})",
    )
    _require(-5.0 <= float(lower) <= 0.0, "lower_bound must be in [-5, 0]")
    tensors = (mixedqkv, state, beta, gate, a_log, dt_bias)
    _require(
        all(tensor.device == state.device for tensor in tensors),
        "all inputs must share one device",
    )
    _require(
        all(tensor.is_contiguous() for tensor in tensors),
        "all public tensor inputs must be contiguous",
    )
    return state_pool, num_heads


def _prepare_bsnd_layout(mixedqkv, gate, beta, state, num_heads):
    _require(mixedqkv.dim() == 3, "BSND mixedqkv must be rank-3")
    batch, storage_length, packed_width = map(int, mixedqkv.shape)
    _require(batch > 0 and storage_length > 0, "BSND B and S must be positive")
    expected_width = 3 * num_heads * SUPPORTED_HEAD_DIM
    _require(
        packed_width == expected_width,
        f"mixedqkv last dimension must be {expected_width}",
    )
    _require(
        tuple(gate.shape) == (batch, storage_length, num_heads, SUPPORTED_HEAD_DIM),
        f"g must have shape [B,S,N,{SUPPORTED_HEAD_DIM}]",
    )
    _require(
        tuple(beta.shape) == (batch, storage_length, num_heads, 1),
        "beta must have shape [B,S,N,1]",
    )
    token_count = batch * storage_length
    mixed = mixedqkv.view(
        batch, storage_length, 3 * num_heads, SUPPORTED_HEAD_DIM
    ).transpose(1, 2)
    output = torch.empty(
        batch,
        storage_length,
        num_heads,
        SUPPORTED_HEAD_DIM,
        dtype=torch.bfloat16,
        device=state.device,
    )
    return _SnapshotLayout(
        batch,
        storage_length,
        token_count,
        mixed,
        gate.transpose(1, 2),
        beta.transpose(1, 2),
        output,
        output.transpose(1, 2),
        0,
    )


def _prepare_bnsd_layout(mixedqkv, gate, beta, state, num_heads):
    _require(mixedqkv.dim() == 4, "BNSD mixedqkv must be rank-4")
    batch, packed_heads, storage_length, dim = map(int, mixedqkv.shape)
    _require(batch > 0 and storage_length > 0, "BNSD B and S must be positive")
    expected_heads = 3 * num_heads
    _require(
        packed_heads == expected_heads and dim == SUPPORTED_HEAD_DIM,
        f"BNSD mixedqkv must have shape [B,{expected_heads},S,{SUPPORTED_HEAD_DIM}]",
    )
    _require(
        tuple(gate.shape) == (batch, num_heads, storage_length, SUPPORTED_HEAD_DIM),
        f"g must have shape [B,N,S,{SUPPORTED_HEAD_DIM}]",
    )
    _require(
        tuple(beta.shape) == (batch, num_heads, storage_length, 1),
        "beta must have shape [B,N,S,1]",
    )
    output = torch.empty(
        batch,
        num_heads,
        storage_length,
        SUPPORTED_HEAD_DIM,
        dtype=torch.bfloat16,
        device=state.device,
    )
    return _SnapshotLayout(
        batch,
        storage_length,
        batch * storage_length,
        mixedqkv,
        gate,
        beta,
        output,
        output,
        0,
    )


def _validate_tnd_inputs(mixedqkv, gate, beta, state, num_heads, cu_seqlens):
    _require(mixedqkv.dim() == 2, "TND mixedqkv must be rank-2")
    token_count, packed_width = map(int, mixedqkv.shape)
    _require(token_count > 0, "TND T must be positive")
    expected_width = 3 * num_heads * SUPPORTED_HEAD_DIM
    _require(
        packed_width == expected_width,
        f"mixedqkv last dimension must be {expected_width}",
    )
    _require(
        tuple(gate.shape) == (token_count, num_heads, SUPPORTED_HEAD_DIM),
        f"g must have shape [T,N,{SUPPORTED_HEAD_DIM}]",
    )
    _require(
        tuple(beta.shape) == (token_count, num_heads, 1), "beta must have shape [T,N,1]"
    )
    _require(cu_seqlens is not None, "cu_seqlens is required for TND")
    _require(cu_seqlens.dtype == torch.int32, "cu_seqlens must be int32")
    _require(
        cu_seqlens.dim() == 1 and cu_seqlens.numel() >= 2,
        "cu_seqlens must be rank-1 with B+1 entries",
    )
    _require(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous")
    _require(
        cu_seqlens.device == state.device, "cu_seqlens must share the input device"
    )
    batch = int(cu_seqlens.numel()) - 1
    if cu_seqlens.device.type == "cpu":
        _require(int(cu_seqlens[0]) == 0, "cu_seqlens must start at 0")
        _require(int(cu_seqlens[-1]) == token_count, "cu_seqlens must end at T")
        _require(
            bool(torch.all(cu_seqlens[1:] > cu_seqlens[:-1])),
            "cu_seqlens must be strictly increasing",
        )
    return batch, token_count


def _prepare_tnd_layout(mixedqkv, gate, beta, state, num_heads, cu_seqlens):
    batch, token_count = _validate_tnd_inputs(
        mixedqkv, gate, beta, state, num_heads, cu_seqlens
    )
    mixed = (
        mixedqkv.view(token_count, 3 * num_heads, SUPPORTED_HEAD_DIM)
        .permute(1, 0, 2)
        .unsqueeze(0)
    )
    gate_logical = gate.permute(1, 0, 2).unsqueeze(0)
    beta_logical = beta.permute(1, 0, 2).unsqueeze(0)
    output = torch.empty(
        token_count,
        num_heads,
        SUPPORTED_HEAD_DIM,
        dtype=torch.bfloat16,
        device=state.device,
    )
    return _SnapshotLayout(
        batch,
        token_count,
        token_count,
        mixed,
        gate_logical,
        beta_logical,
        output,
        output.permute(1, 0, 2).unsqueeze(0),
        1,
    )


def _prepare_snapshot_layout(
    layout, mixedqkv, gate, beta, state, num_heads, cu_seqlens
):
    if layout == "BSND":
        return _prepare_bsnd_layout(mixedqkv, gate, beta, state, num_heads)
    if layout == "BNSD":
        return _prepare_bnsd_layout(mixedqkv, gate, beta, state, num_heads)
    return _prepare_tnd_layout(mixedqkv, gate, beta, state, num_heads, cu_seqlens)


def _prepare_state_indices(indices, token_count, state):
    if indices is None:
        return torch.arange(token_count, dtype=torch.int32, device=state.device)
    _require(
        indices.dtype == torch.int32,
        f"ssm_state_indices must be int32, got {indices.dtype}",
    )
    _require(
        tuple(indices.shape) == (token_count,),
        f"ssm_state_indices must have shape ({token_count},)",
    )
    _require(indices.is_contiguous(), "ssm_state_indices must be contiguous")
    _require(
        indices.device == state.device, "ssm_state_indices must share the input device"
    )
    return indices


def _prepare_accepted_tokens(accepted, batch, state):
    if accepted is None:
        return torch.ones(batch, dtype=torch.int32, device=state.device)
    _require(
        accepted.dtype == torch.int32,
        f"num_accepted_tokens must be int32, got {accepted.dtype}",
    )
    _require(
        tuple(accepted.shape) == (batch,),
        f"num_accepted_tokens must have shape ({batch},)",
    )
    _require(accepted.is_contiguous(), "num_accepted_tokens must be contiguous")
    _require(
        accepted.device == state.device,
        "num_accepted_tokens must share the input device",
    )
    return accepted


def _launch_snapshot(
    layout,
    state,
    a_log,
    dt_bias,
    mixedqkv,
    scale,
    lower_bound,
    state_indices,
    accepted_tokens,
    cu_launch,
):
    scale_value = float(scale) if scale is not None else SUPPORTED_HEAD_DIM**-0.5
    _, core_num = _get_load_balance_config(
        layout.batch * int(state.shape[1]),
        layout.storage_length,
        _device_block_num(mixedqkv),
    )
    state_dtype = dtypes.bfloat16 if state.dtype == torch.bfloat16 else dtypes.float32
    _get_compiled_kernel(state_dtype)(
        layout.mixed,
        layout.gate,
        layout.beta,
        a_log,
        dt_bias,
        layout.output_logical,
        state,
        state,
        state_indices,
        accepted_tokens,
        cu_launch,
        scale_value,
        float(lower_bound),
        layout.mode,
        layout.batch,
        layout.storage_length,
        core_num,
    )


def fused_recurrent_kda_functional(
    mixedqkv: torch.Tensor,
    state: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float],
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = -5.0,
    layout_qkv: str = "BSND",
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run Snapshot with dense BSND/BNSD or packed TND/cu storage."""
    _require(
        layout_qkv in ("BSND", "BNSD", "TND"),
        "layout_qkv must be BSND, BNSD, or TND",
    )
    state_pool, num_heads = _validate_snapshot_common(
        mixedqkv, state, beta, g, A_log, dt_bias, lower_bound
    )

    layout = _prepare_snapshot_layout(
        layout_qkv, mixedqkv, g, beta, state, num_heads, cu_seqlens
    )
    _require(
        state_pool >= layout.token_count,
        f"state pool must provide at least {layout.token_count} slots",
    )
    state_indices = _prepare_state_indices(ssm_state_indices, layout.token_count, state)
    accepted_tokens = _prepare_accepted_tokens(num_accepted_tokens, layout.batch, state)
    if layout.mode == 0:
        cu_launch = state_indices[:1]
    else:
        cu_launch = cu_seqlens

    _launch_snapshot(
        layout,
        state,
        A_log,
        dt_bias,
        mixedqkv,
        scale,
        lower_bound,
        state_indices,
        accepted_tokens,
        cu_launch,
    )
    return state, layout.output


def fused_recurrent_kda(
    mixedqkv: torch.Tensor,
    state: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float],
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = -5.0,
    layout_qkv: str = "BSND",
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, output = fused_recurrent_kda_functional(
        mixedqkv,
        state,
        beta,
        g,
        scale,
        A_log,
        dt_bias,
        lower_bound,
        layout_qkv,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
    )
    return state, output


_GRAPH_LIBRARY = torch.library.Library("cannbotdsl_fused_recurrent_kda", "DEF")
_GRAPH_LIBRARY.define(
    "fused_recurrent_kda("
    "Tensor mixedqkv, Tensor(a!) state, Tensor beta, Tensor g, "
    "float scale, Tensor A_log, Tensor dt_bias, float lower_bound, str layout_qkv, "
    "Tensor? cu_seqlens=None, Tensor? ssm_state_indices=None, "
    "Tensor? num_accepted_tokens=None) -> Tensor"
)
_GRAPH_LIBRARY.define(
    "fused_recurrent_kda_functional("
    "Tensor mixedqkv, Tensor state, Tensor beta, Tensor g, "
    "float scale, Tensor A_log, Tensor dt_bias, float lower_bound, str layout_qkv, "
    "Tensor? cu_seqlens=None, Tensor? ssm_state_indices=None, "
    "Tensor? num_accepted_tokens=None) -> (Tensor, Tensor)"
)


@torch.library.impl(_GRAPH_LIBRARY, "fused_recurrent_kda", "Meta")
def _fused_recurrent_kda_meta(
    mixedqkv,
    state,
    beta,
    g,
    scale,
    A_log,
    dt_bias,
    lower_bound,
    layout_qkv,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
):
    del mixedqkv, state, beta, scale, A_log, dt_bias
    del lower_bound, cu_seqlens, ssm_state_indices, num_accepted_tokens, layout_qkv
    return torch.empty_like(g, device="meta")


@torch.library.impl(_GRAPH_LIBRARY, "fused_recurrent_kda_functional", "Meta")
def _fused_recurrent_kda_functional_meta(
    mixedqkv,
    state,
    beta,
    g,
    scale,
    A_log,
    dt_bias,
    lower_bound,
    layout_qkv,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
):
    del mixedqkv, beta, scale, A_log, dt_bias
    del lower_bound, cu_seqlens, ssm_state_indices, num_accepted_tokens, layout_qkv
    return torch.empty_like(state, device="meta"), torch.empty_like(g, device="meta")


@torch.library.impl(_GRAPH_LIBRARY, "fused_recurrent_kda", "PrivateUse1")
def _fused_recurrent_kda_privateuse1(
    mixedqkv,
    state,
    beta,
    g,
    scale,
    A_log,
    dt_bias,
    lower_bound,
    layout_qkv,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
):
    _, output = fused_recurrent_kda(
        mixedqkv,
        state,
        beta,
        g,
        scale,
        A_log,
        dt_bias,
        lower_bound,
        layout_qkv,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
    )
    return output


@torch.library.impl(_GRAPH_LIBRARY, "fused_recurrent_kda_functional", "PrivateUse1")
def _fused_recurrent_kda_functional_privateuse1(
    mixedqkv,
    state,
    beta,
    g,
    scale,
    A_log,
    dt_bias,
    lower_bound,
    layout_qkv,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
):
    _, output = fused_recurrent_kda_functional(
        mixedqkv,
        state,
        beta,
        g,
        scale,
        A_log,
        dt_bias,
        lower_bound,
        layout_qkv,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
    )
    return output, state


fused_recurrent_kda_op = torch.ops.cannbotdsl_fused_recurrent_kda.fused_recurrent_kda
fused_recurrent_kda_functional_op = (
    torch.ops.cannbotdsl_fused_recurrent_kda.fused_recurrent_kda_functional
)


@export("fused_recurrent_kda")
def export_fused_recurrent_kda():
    """Export Snapshot variants for both supported recurrent-state dtypes."""
    try:
        for state_dtype in (dtypes.bfloat16, dtypes.float32):
            _get_compiled_kernel(state_dtype)
    finally:
        with _COMPILED_KERNEL_LOCK:
            compiled = list(_COMPILED_KERNELS.values())
            _COMPILED_KERNELS.clear()
        for kernel_handle in compiled:
            kernel_handle.close()


# Old network code calls the public host function directly. Under Dynamo
# (torch.compile, fullgraph) that host body is untraceable, so during
# tracing the call is substituted with the registered dispatcher op;
# eager calls keep taking the host path unchanged.
getattr(torch, "_dynamo").substitute_in_graph(
    fused_recurrent_kda, can_constant_fold_through=False
)(lambda *args, **kwargs: fused_recurrent_kda_op(*args, **kwargs))
