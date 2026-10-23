# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# fmt: off
# Keep compact call argument groups consistent with the original FlashKDA samples.


"""Persistent FlashKDA training forward with resident FP32 state and saved backward inputs."""

from typing import Optional, Tuple

from cannbotdsl import PIPE, MemLoc, Tensor, dtypes
from cannbotdsl.aot import export
from cannbotdsl.buffer import Buffer
from cannbotdsl.channel import Channel as RingBuffer
from cannbotdsl.lang import vf
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.jit import jit
from cannbotdsl.ops.arch import get_block_idx, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    update_mask,
    vabs,
    vadd,
    vadds,
    varange,
    vbitwise_and,
    vcast,
    vdiv,
    vdups,
    veq,
    vexp,
    vexp_sub,
    vgather,
    vload,
    vload_broadcast,
    vload_unpack,
    vlog,
    vmadd,
    vmaxs,
    vmem_bar,
    vmul,
    vmuls,
    vpack,
    vreduce_sum,
    vreinterpret,
    vreinterpret_lanes,
    vshl,
    vsqrt,
    vstore,
    vstore_first,
    vstore_pack,
    vstore_strided,
)
from cannbotdsl.ops.reg import vselect as vselect_raw
from cannbotdsl.ops.sync import (
    channel_rewind,
    cube_fill_l1_zero,
    cube_sync_all,
    cube_sync_pipe,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
    vec_sync_notify,
    vec_sync_wait,
)
from cannbotdsl.tensor import reinterpret, tile_slice
import torch

from .flash_kda_fwd_metadata import (
    HEADER_WORDS,
    CHUNK_TASK_BATCH,
    CHUNK_TASK_LENGTH,
    CHUNK_TASK_NEXT,
    CHUNK_TASK_STORAGE_BATCH,
    CHUNK_TASK_TOKEN_START,
    CHUNK_TASK_VALUE_HEAD,
)
from .flash_kda_fwd_staged import GATE_MODE_FUSED_WITH_DT_BIAS


CHUNK_SIZE = 64
HALF_CHUNK_SIZE = CHUNK_SIZE // 2
SUPPORTED_HEAD_DIM = 128
D_HALF = SUPPORTED_HEAD_DIM // 2
Q_DECAYED_OPERAND_INDEX = 0
K_RESTORED_OPERAND_INDEX = 1
K_DECAYED_BETA_OPERAND_INDEX = 2
V_BETA_OPERAND_INDEX = 3
DIAGONAL_BLOCK_SIZE = 16
DIAGONAL_BLOCK_COUNT = CHUNK_SIZE // DIAGONAL_BLOCK_SIZE
FRACTAL_ELEMENTS = DIAGONAL_BLOCK_SIZE * DIAGONAL_BLOCK_SIZE


def cast_and_pack_bf16(value, lane_mask):
    value_bf16 = vcast(value, dtypes.bfloat16, mask=lane_mask)
    packed_value = vpack(vreinterpret_lanes(value_bf16, dtypes.uint32), dtypes.uint16)
    return vreinterpret(packed_value, dtypes.bfloat16)


@jit
def load_sequence_tile(dst, src, tile_idx, column_idx, sequence_length, has_tail, cache_policy=0):
    """Load one ND tile; explicitly zero invalid rows of a partial sequence."""
    rows, cols = dst.shape
    source_tile = tile_slice(src, (rows, cols), (tile_idx, column_idx))
    if const_expr(has_tail):
        first_row = tile_idx * rows
        if first_row + rows <= sequence_length:
            mem_copy(dst, source_tile, l2_cache_ctl=cache_policy)
        else:
            with vf(mode="simd"):
                for offset in range(0, rows * cols, D_HALF):
                    lane_mask, _ = update_mask(rows * cols - offset, elem_bits=32)
                    zero_fp32 = vdups(0.0, dtypes.float32, mask=lane_mask)
                    zero_bf16 = vcast(zero_fp32, dtypes.bfloat16, mask=lane_mask)
                    vstore_pack(dst, offset, zero_bf16, lane_mask, pack_mode=PackMode.B32_TO_B16)
                vmem_bar("vst_vld")
            if first_row < sequence_length:
                valid_rows = sequence_length - first_row
                mem_copy(dst[:valid_rows, None], source_tile[:valid_rows, None], l2_cache_ctl=cache_policy)
    else:
        mem_copy(dst, source_tile, l2_cache_ctl=cache_policy)


@jit
def pack_score_operand_nz(destination, source):
    with vf(mode="simd"):
        lane_mask, _ = update_mask(D_HALF, elem_bits=32)
        low_mask, _ = update_mask(D_HALF, elem_bits=16)
        for row in dsl_range(0, DIAGONAL_BLOCK_SIZE, 1):
            values = vcast(
                vload_unpack(source, row * D_HALF, unpack_mode=UnpackMode.B16_TO_B32), dtypes.float32, mask=lane_mask,
            )
            packed = cast_and_pack_bf16(values, lane_mask)
            vstore_strided(destination, row * 16, packed, low_mask, block_stride=destination.physical_stride[0] // 16)


class L2NormAndActivate:
    """Exchange Q/K L2 norms, activate gate/beta, and publish recurrent operands."""

    def __init__(self, subblock_idx):
        self.subblock_idx = subblock_idx
        self.ub_q = [
            Buffer(MemLoc.UB, (HALF_CHUNK_SIZE, D_HALF), dtypes.bfloat16)
            for _ in range(2)
        ]
        self.ub_k = [
            Buffer(MemLoc.UB, (HALF_CHUNK_SIZE, D_HALF), dtypes.bfloat16)
            for _ in range(2)
        ]
        self.ub_g_raw = Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16)
        self.ub_v_raw = Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16)
        self.ub_beta_raw = Buffer(MemLoc.UB, (CHUNK_SIZE, 1), dtypes.bfloat16)
        self.ub_dt_bias = RingBuffer(MemLoc.UB, (D_HALF,), dtypes.float32, depth=1).produce()
        self.ub_neg_exp_a = RingBuffer(MemLoc.UB, (1,), dtypes.float32, depth=1).produce()
        self.ub_g_cumsum = Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.float32)
        self.ub_q_norm = Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16)
        self.ub_k_norm = Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16)
        self.ub_beta = Buffer(MemLoc.UB, (CHUNK_SIZE, 1), dtypes.float32)
        self.ub_qk_norms = Buffer(MemLoc.UB, (2, CHUNK_SIZE), dtypes.float32)
        self.ub_inverse_prefix = Buffer(MemLoc.UB, (11, 32), dtypes.float32)
        self.ub_peer_qk_partials = Buffer(MemLoc.UB, (2 * CHUNK_SIZE,), dtypes.float32)
        self.ub_operand_staging = RingBuffer(
            MemLoc.UB, (18 * 16, D_HALF), dtypes.bfloat16, depth=1, data_format="nz", n1_pad=16,
        ).produce()
        self.ub_recurrent_operands_nz = reinterpret(
            self.ub_operand_staging, shape=(4 * CHUNK_SIZE, D_HALF), data_format="nz", n1_pad=16,
        )
        self.l1_recurrent_operands = RingBuffer(
            MemLoc.L1, (4 * CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=2,
        )

        # Bound by the task dispatcher before the corresponding stage runs.
        self.operand_tiles = None
        self.scale = None
        self.lower_bound = None
        self.use_qk_l2norm_in_kernel = None
        self.has_dt_bias = None
        self.use_gate_in_kernel = None
        self.has_lower_bound = None
        self.use_beta_sigmoid_in_kernel = None
        self.allow_neg_eigval = None
        self.save_qg = None
        self.use_stable_scores = None
        self.cache_writer = None
        self.saved_qg_nd = None
        self.saved_kg_nd = None

    @jit
    def load_head(self, a_log, dt_bias, value_head_idx):
        mem_copy(self.ub_neg_exp_a, tile_slice(a_log, (1,), (value_head_idx,)))
        if self.has_dt_bias:
            mem_copy(self.ub_dt_bias, tile_slice(dt_bias[value_head_idx, None], (D_HALF,), (self.subblock_idx,)))
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            exp_a = vexp(vload_broadcast(self.ub_neg_exp_a, 0), mask=lane_mask)
            vstore_first(self.ub_neg_exp_a, 0, vmuls(exp_a, -1.0, mask=lane_mask))

    @jit
    def load_chunk_inputs(self, q, k, v, g, beta, chunk_idx):
        vec_sync_wait(PIPE.V, PIPE.MTE2, 6)
        for panel_idx in (0, 1):
            if chunk_idx == 0:
                load_sequence_tile(
                    self.ub_q[panel_idx], q, chunk_idx * 2 + panel_idx, self.subblock_idx, self.sequence_length,
                    self.has_tail,
                )
                load_sequence_tile(
                    self.ub_k[panel_idx], k, chunk_idx * 2 + panel_idx, self.subblock_idx, self.sequence_length,
                    self.has_tail,
                )
        vec_sync_notify(PIPE.MTE2, PIPE.V, 6)
        load_sequence_tile(
            self.ub_g_raw, g, chunk_idx, self.subblock_idx, self.sequence_length, self.has_tail, 4,
        )
        load_sequence_tile(self.ub_beta_raw, beta, chunk_idx, 0, self.sequence_length, self.has_tail)
        vec_sync_notify(PIPE.MTE2, PIPE.V, 3)
        load_sequence_tile(
            self.ub_v_raw, v, chunk_idx, self.subblock_idx, self.sequence_length, self.has_tail, 4,
        )

    def prefetch_qk_front(self, q, k, chunk_idx):
        load_sequence_tile(self.ub_q[0], q, chunk_idx * 2, self.subblock_idx, self.sequence_length, self.has_tail)
        load_sequence_tile(self.ub_k[0], k, chunk_idx * 2, self.subblock_idx, self.sequence_length, self.has_tail)

    def prefetch_qk_back(self, q, k, chunk_idx):
        load_sequence_tile(self.ub_q[1], q, chunk_idx * 2 + 1, self.subblock_idx, self.sequence_length, self.has_tail)
        load_sequence_tile(self.ub_k[1], k, chunk_idx * 2 + 1, self.subblock_idx, self.sequence_length, self.has_tail)

    @jit
    def compute_qk_squared_norms(self):
        vec_sync_wait(PIPE.MTE2, PIPE.V, 6)
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            for half_idx in (0, 1):
                q_half, k_half = (self.ub_q[half_idx], self.ub_k[half_idx])
                for row in dsl_range(0, HALF_CHUNK_SIZE, 1, unroll=4):
                    q_value = vcast(
                        vload_unpack(q_half, row * D_HALF, unpack_mode=UnpackMode.B16_TO_B32), dtypes.float32,
                        mask=lane_mask,
                    )
                    k_value = vcast(
                        vload_unpack(k_half, row * D_HALF, unpack_mode=UnpackMode.B16_TO_B32), dtypes.float32,
                        mask=lane_mask,
                    )
                    vstore_first(
                        self.ub_qk_norms, half_idx * HALF_CHUNK_SIZE + row,
                        vreduce_sum(vmul(q_value, q_value, mask=lane_mask), mask=lane_mask),
                    )
                    vstore_first(
                        self.ub_qk_norms, CHUNK_SIZE + half_idx * HALF_CHUNK_SIZE + row,
                        vreduce_sum(vmul(k_value, k_value, mask=lane_mask), mask=lane_mask),
                    )
            vmem_bar("vst_vld")

    def exchange_qk_squared_norms(self, exchange, block_idx):
        vec_sync_notify(PIPE.V, PIPE.MTE3, 4)
        vec_sync_wait(PIPE.V, PIPE.MTE3, 4)
        mem_copy(
            exchange[block_idx, self.subblock_idx, None, None],
            reinterpret(self.ub_qk_norms, shape=(2, CHUNK_SIZE), data_format="nd"),
        )
        vec_sync_block_arrive(PIPE.MTE3, 10, mode=1)
        vec_sync_block_wait(PIPE.MTE2, 10, mode=1)
        mem_copy(
            reinterpret(self.ub_peer_qk_partials, shape=(2, CHUNK_SIZE), data_format="nd"),
            exchange[block_idx, 1 - self.subblock_idx, None, None],
        )
        vec_sync_notify(PIPE.MTE2, PIPE.V, 5)

    def compute_norms_and_activations(self, bound, exchange, block_idx, chunk_idx):
        self.compute_qk_squared_norms()
        self.exchange_qk_squared_norms(exchange, block_idx)
        self.activate_gate_and_beta(bound, chunk_idx)
        self.compute_qk_normalization_scales()

    @jit
    def activate_gate_and_beta(self, bound, chunk_idx):
        vec_sync_wait(PIPE.MTE2, PIPE.V, 3)
        valid_rows = CHUNK_SIZE
        if const_expr(self.has_tail):
            valid_rows = self.sequence_length - chunk_idx * CHUNK_SIZE
            if valid_rows > CHUNK_SIZE:
                valid_rows = CHUNK_SIZE
        if self.use_gate_in_kernel != 0:
            if self.has_lower_bound != 0:
                self._vf_select_dt_bias_mode(bound, valid_rows, 2)
            else:
                self._vf_select_dt_bias_mode(bound, valid_rows, 1)
        else:
            self._vf_select_beta_mode(bound, valid_rows, 0, False)

    @jit
    def compute_qk_normalization_scales(self):
        vec_sync_wait(PIPE.MTE2, PIPE.V, 5)
        vec_sync_block_arrive(PIPE.MTE2, 11, mode=1)
        vec_sync_block_wait(PIPE.MTE3, 11, mode=1)
        if self.use_qk_l2norm_in_kernel != 0:
            self._vf_write_qk_normalization_scales(True)
        else:
            self._vf_write_qk_normalization_scales(False)

    @jit
    def prepare_recurrent_operands(self, scale, chunk_decay, chunk_idx):
        if self.save_qg != 0:
            self._vf_pack_recurrent_operands(scale, chunk_decay, True)
        else:
            self._vf_pack_recurrent_operands(scale, chunk_decay, False)
        mem_copy(
            self.l1_recurrent_operands.produce(), self.ub_recurrent_operands_nz, engine=make_copy_engine(split_axis=1),
            part_id=self.subblock_idx,
        )
        rows = self.cache_writer.valid_rows(chunk_idx)
        if self.save_qg != 0:
            mem_copy(
                tile_slice(self.cache_writer.qg, (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))[:rows, None],
                self.saved_qg_nd[:rows, None], l2_cache_ctl=0,
            )
        if self.save_qg != 0:
            mem_copy(
                tile_slice(self.cache_writer.kg, (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))[:rows, None],
                self.saved_kg_nd[:rows, None], l2_cache_ctl=0,
            )

    def acquire_recurrent_operands(self):
        operands_l1_read = self.l1_recurrent_operands.consume()
        self.operand_tiles = [
            tile_slice(operands_l1_read, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (operand_idx, 0))
            for operand_idx in range(4)
        ]

    @jit
    def _vf_select_dt_bias_mode(self, bound, valid_rows, gate_mode):
        if self.has_dt_bias:
            self._vf_select_beta_mode(bound, valid_rows, gate_mode, True)
        else:
            self._vf_select_beta_mode(bound, valid_rows, gate_mode, False)

    @jit
    def _vf_select_beta_mode(self, bound, valid_rows, gate_mode, has_dt_bias):
        if self.use_beta_sigmoid_in_kernel != 0:
            if self.allow_neg_eigval != 0:
                self._vf_activate_gate_beta(bound, valid_rows, gate_mode, 2, has_dt_bias)
            else:
                self._vf_activate_gate_beta(bound, valid_rows, gate_mode, 1, has_dt_bias)
        else:
            self._vf_activate_gate_beta(bound, valid_rows, gate_mode, 0, has_dt_bias)

    @jit
    def _vf_activate_gate_beta(
        self, bound, valid_rows, gate_mode, beta_mode, has_dt_bias
    ):
        raw_gate = self.ub_g_raw
        raw_beta = self.ub_beta_raw
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            beta_mask, _ = update_mask(CHUNK_SIZE, elem_bits=32)
            one = vdups(1.0, dtypes.float32, mask=lane_mask)
            negative_exp_a = vdups(0.0, dtypes.float32, mask=lane_mask)
            dt_bias = vdups(0.0, dtypes.float32, mask=lane_mask)
            if const_expr(gate_mode != 0):
                negative_exp_a = vload_broadcast(self.ub_neg_exp_a, 0)
                if const_expr(has_dt_bias):
                    dt_bias = vload(self.ub_dt_bias, 0)
            prefix = vdups(0.0, dtypes.float32, mask=lane_mask)
            beta_value = vcast(
                vload_unpack(raw_beta, 0, unpack_mode=UnpackMode.B16_TO_B32), dtypes.float32, mask=beta_mask,
            )
            activated_beta = beta_value
            if const_expr(beta_mode != 0):
                activated_beta = vdiv(
                    one, vadds(vexp(vmuls(beta_value, -1.0, mask=beta_mask), mask=beta_mask), 1.0, mask=beta_mask),
                    mask=beta_mask,
                )
                if const_expr(beta_mode == 2):
                    activated_beta = vmuls(activated_beta, 2.0, mask=beta_mask)
            if const_expr(self.has_tail):
                valid_beta_mask, _ = update_mask(valid_rows, elem_bits=32)
                activated_beta = vselect_raw(
                    activated_beta, vdups(0.0, dtypes.float32, mask=beta_mask), cond_mask=valid_beta_mask,
                )
            vstore(self.ub_beta, 0, activated_beta, beta_mask)
            for row in dsl_range(0, valid_rows, 1, unroll=2):
                gate_value = vcast(
                    vload_unpack(raw_gate, row * D_HALF, unpack_mode=UnpackMode.B16_TO_B32), dtypes.float32,
                    mask=lane_mask,
                )
                gate = gate_value
                if const_expr(gate_mode != 0):
                    biased_gate = vadd(gate_value, dt_bias, mask=lane_mask)
                    if const_expr(gate_mode == 2):
                        logit = vmul(negative_exp_a, biased_gate, mask=lane_mask)
                        gate = vdiv(
                            vdups(bound, dtypes.float32, mask=lane_mask),
                            vadds(vexp(logit, mask=lane_mask), 1.0, mask=lane_mask), mask=lane_mask,
                        )
                    else:
                        softplus_tail = vlog(
                            vadds(
                                vexp(vmuls(vabs(biased_gate, mask=lane_mask), -1.0, mask=lane_mask), mask=lane_mask),
                                1.0, mask=lane_mask,
                            ),
                            mask=lane_mask,
                        )
                        softplus = vadd(vmaxs(biased_gate, 0.0, mask=lane_mask), softplus_tail, mask=lane_mask)
                        gate = vmul(negative_exp_a, softplus, mask=lane_mask)
                prefix = vadd(prefix, gate, mask=lane_mask)
                vstore(self.ub_g_cumsum, row * D_HALF, prefix, lane_mask)
            if const_expr(self.has_tail):
                for row in dsl_range(valid_rows, CHUNK_SIZE, 1, unroll=2):
                    vstore(self.ub_g_cumsum, row * D_HALF, prefix, lane_mask)
            vmem_bar("vst_vld")

    @jit
    def _vf_write_qk_normalization_scales(self, normalize):
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            one = vdups(1.0, dtypes.float32, mask=lane_mask)
            for qk_idx in (0, 1):
                squared_norm = vadd(
                    vload(self.ub_qk_norms, qk_idx * CHUNK_SIZE), vload(self.ub_peer_qk_partials, qk_idx * CHUNK_SIZE),
                    mask=lane_mask,
                )
                inverse_norm = one
                if const_expr(normalize):
                    inverse_norm = vdiv(
                        one, vsqrt(vadds(squared_norm, 1e-06, mask=lane_mask), mask=lane_mask), mask=lane_mask,
                    )
                vstore(self.ub_qk_norms, qk_idx * CHUNK_SIZE, inverse_norm, lane_mask)
            vmem_bar("vst_vld")

    @jit
    def _vf_pack_recurrent_operands(self, scale, chunk_decay, save_inputs):
        raw_value = self.ub_v_raw
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            low_mask, _ = update_mask(D_HALF, elem_bits=16)
            last_gate_cumsum = vload(self.ub_g_cumsum, (CHUNK_SIZE - 1) * D_HALF)
            vstore(chunk_decay, 0, vexp(last_gate_cumsum, mask=lane_mask), lane_mask)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=2):
                element_offset = row * D_HALF
                gate_prefix_sum = vload(self.ub_g_cumsum, element_offset)
                decay_scale = vexp(gate_prefix_sum, mask=lane_mask)
                restore_scale = vexp_sub(last_gate_cumsum, gate_prefix_sum, mask=lane_mask)
                q_value = vcast(
                    vload_unpack(self.ub_q_norm, element_offset, unpack_mode=UnpackMode.B16_TO_B32), dtypes.float32,
                    mask=lane_mask,
                )
                k_value = vcast(
                    vload_unpack(self.ub_k_norm, element_offset, unpack_mode=UnpackMode.B16_TO_B32), dtypes.float32,
                    mask=lane_mask,
                )
                v_value = vcast(
                    vload_unpack(raw_value, element_offset, unpack_mode=UnpackMode.B16_TO_B32), dtypes.float32,
                    mask=lane_mask,
                )
                raw_beta = vload_broadcast(self.ub_beta, row)
                qg_value = vmul(q_value, decay_scale, mask=lane_mask)
                kg_value = vmul(k_value, restore_scale, mask=lane_mask)
                if const_expr(save_inputs):
                    vstore_pack(
                        self.saved_qg_nd, element_offset, vcast(qg_value, dtypes.bfloat16, mask=lane_mask), lane_mask,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                if const_expr(save_inputs):
                    vstore_pack(
                        self.saved_kg_nd, element_offset, vcast(kg_value, dtypes.bfloat16, mask=lane_mask), lane_mask,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                operand_values = (
                    vmuls(qg_value, scale, mask=lane_mask),
                    kg_value,
                    vmul(vmul(k_value, decay_scale, mask=lane_mask), raw_beta, mask=lane_mask),
                    vmul(v_value, raw_beta, mask=lane_mask),
                )
                for operand_idx in (0, 1, 2, 3):
                    operand_bf16 = cast_and_pack_bf16(operand_values[operand_idx], lane_mask)
                    vstore_strided(
                        self.ub_recurrent_operands_nz, (operand_idx * CHUNK_SIZE + row) * 16, operand_bf16, low_mask,
                        block_stride=self.ub_recurrent_operands_nz.physical_stride[0] // 16,
                    )


class RecurrentStateUpdate:
    """Retain FP32 state in UB and publish BF16 snapshots for Cube products."""

    def __init__(self, subblock_idx):
        self.subblock_idx = subblock_idx
        self.ub_state_fp32 = Buffer(MemLoc.UB, (SUPPORTED_HEAD_DIM, D_HALF), dtypes.float32)
        self.ub_state_delta = RingBuffer(MemLoc.UB, (SUPPORTED_HEAD_DIM, D_HALF), dtypes.float32, depth=1).produce()
        self.ub_state_nz = RingBuffer(
            MemLoc.UB, (SUPPORTED_HEAD_DIM, D_HALF), dtypes.bfloat16, depth=1, data_format="nz", n1_pad=16,
        ).produce()
        self.state_l1 = RingBuffer(MemLoc.L1, (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=2)
        self.gamma_ub = RingBuffer(MemLoc.UB, (D_HALF,), dtypes.float32, depth=2)
        self.u_l1 = RingBuffer(MemLoc.L1, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=1)


    def load(self, gm_state_rows):
        mem_copy(self.ub_state_fp32, tile_slice(gm_state_rows, (SUPPORTED_HEAD_DIM, D_HALF), (0, self.subblock_idx)))
        vec_sync_notify(PIPE.MTE2, PIPE.V, 7)
        vec_sync_wait(PIPE.MTE2, PIPE.V, 7)
        self.cast_state_snapshot_to_bf16_nz()
        self.publish_state_snapshot()

    @jit
    def cast_state_snapshot_to_bf16_nz(self):
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            low_mask, _ = update_mask(D_HALF, elem_bits=16)
            for row in dsl_range(0, SUPPORTED_HEAD_DIM, 1, unroll=4):
                state_bf16 = cast_and_pack_bf16(vload(self.ub_state_fp32, row * D_HALF), lane_mask)
                vstore_strided(
                    self.ub_state_nz, row * 16, state_bf16, low_mask,
                    block_stride=self.ub_state_nz.physical_stride[0] // 16,
                )

    def publish_state_snapshot(self):
        mem_copy(
            self.state_l1.produce(), self.ub_state_nz, engine=make_copy_engine(split_axis=1), part_id=self.subblock_idx,
        )

    @jit
    def update_recurrent_state(self):
        chunk_decay = self.gamma_ub.consume()
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            gamma_c = vload(chunk_decay, 0)
            low_mask, _ = update_mask(D_HALF, elem_bits=16)
            for row in dsl_range(0, SUPPORTED_HEAD_DIM, 1, unroll=4):
                state_value = vload(self.ub_state_fp32, row * D_HALF)
                state_delta = vload(self.ub_state_delta, row * D_HALF)
                state_next = vmadd(state_value, gamma_c, state_delta, mask=lane_mask)
                vstore(self.ub_state_fp32, row * D_HALF, state_next, lane_mask)
                state_bf16 = cast_and_pack_bf16(state_next, lane_mask)
                vstore_strided(
                    self.ub_state_nz, row * 16, state_bf16, low_mask,
                    block_stride=self.ub_state_nz.physical_stride[0] // 16,
                )

    def store(self, gm_state_rows):
        vec_sync_notify(PIPE.V, PIPE.MTE3, 7)
        vec_sync_wait(PIPE.V, PIPE.MTE3, 7)
        mem_copy(tile_slice(gm_state_rows, (SUPPORTED_HEAD_DIM, D_HALF), (0, self.subblock_idx)), self.ub_state_fp32)



class ChunkMatmul:
    """Compute chunk scores, compose the resident inverse, and update output/state products."""

    def __init__(self):
        self.tile_inv_l1 = RingBuffer(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1).produce()
        self.packed_inv64_l1 = RingBuffer(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1).produce()
        self.inverse_product_l1 = RingBuffer(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1).produce()
        self.negative_kkt_l1 = RingBuffer(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1).produce()
        self.packed_identity_l1 = RingBuffer(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1).produce()
        self.zero_l1 = Buffer(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16)
        self.resident_inv_l1 = RingBuffer(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.bfloat16, depth=1).produce()
        self.l0a_recurrent = RingBuffer(MemLoc.L0A, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=2)
        self.l0b_recurrent = RingBuffer(MemLoc.L0B, (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=2)
        self.l0c_state = RingBuffer(
            MemLoc.L0C, (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), dtypes.float32, depth=1,
        ).produce()
        self.l0c_o = RingBuffer(MemLoc.L0C, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.float32, depth=1).produce()
        self.l0c_w = reinterpret(self.l0c_state, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), data_format="nz")
        self.l0a_scores = RingBuffer(MemLoc.L0A, (HALF_CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=2)
        self.raw_l0a_db = RingBuffer(MemLoc.L0A, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=2)
        self.raw_l0c_main = reinterpret(self.l0c_state, shape=(CHUNK_SIZE, CHUNK_SIZE), data_format="nz")
        self.raw_l0c_aux = reinterpret(self.l0c_o, shape=(CHUNK_SIZE, CHUNK_SIZE), data_format="nz")
        self.l0c_kkt = RingBuffer(MemLoc.L0C, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float32, depth=1).produce()
        self.l0c_qkt = RingBuffer(MemLoc.L0C, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float32, depth=1).produce()
        self.w_l1 = RingBuffer(MemLoc.L1, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=1)
        self.raw_l0b_pending = []
        # Scores, inverse and recurrent products have disjoint operand lifetimes.
        # Reuse the two 32 KiB recurrent slots for all L0B operands.
        l0b_cache_0 = self.l0b_recurrent.produce()
        l0b_cache_1 = self.l0b_recurrent.produce()
        k_transpose_panels = []
        panel_layout = (
            (0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (1, 4), (1, 5),
        )
        for slot, panel in panel_layout:
            operand = reinterpret(
                l0b_cache_0 if slot == 0 else l0b_cache_1,
                shape=(DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM),
                offset=panel * DIAGONAL_BLOCK_SIZE * SUPPORTED_HEAD_DIM * 2, data_format="nz",
            )
            k_transpose_panels.append(operand)
        self.k_transpose_l0b_cache = tuple(k_transpose_panels)

        # Bound by the task dispatcher before the corresponding stage runs.
        self.save_backward_inputs = None
        self.saved_w = None
        self.saved_u = None
        self.saved_v_new = None

    @staticmethod
    def l0_matrix_block_offset_elems(row_block: int, col_block: int):
        return (row_block * DIAGONAL_BLOCK_COUNT + col_block) * FRACTAL_ELEMENTS

    @staticmethod
    def c64_diag_nz_offset_elems(diag_block: int):
        return diag_block * (DIAGONAL_BLOCK_COUNT + 1) * FRACTAL_ELEMENTS

    @staticmethod
    def zero_fp16_l1(l1):
        cube_fill_l1_zero(l1, repeat=256, blk_num=1, dst_gap=0)

    @staticmethod
    def publish_inverse_product_fp16_l1(dst_ring_buffer, c_tile):
        mem_copy(dst_ring_buffer, c_tile, unit_flag=3)

    @staticmethod
    def publish_resident_inverse_bf16_l1(dst_ring_buffer, c_tile):
        mem_copy(dst_ring_buffer, c_tile, unit_flag=3)

    @staticmethod
    def _packed64_block_offset_elems(row_block: int, col_block: int):
        return (
            col_block * CHUNK_SIZE * DIAGONAL_BLOCK_SIZE
            + row_block * DIAGONAL_BLOCK_SIZE * DIAGONAL_BLOCK_SIZE
        )

    def compose_chunk_inverse64(self, identity):
        negative = self.negative_kkt_l1
        diagonal = self.tile_inv_l1
        self.issue_odd_lower_even_inv_full64(negative, diagonal)
        self.begin_mmad_raw_l0ab_db_packed64(self.raw_l0c_aux, init_c=True, unit_flag=3)
        self.publish_inverse_product_fp16_l1(self.inverse_product_l1, self.raw_l0c_aux)
        left = self.raw_l0a_db.produce()
        right = reinterpret(
            self.l0b_recurrent.produce(), shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM),
            dtype=dtypes.float16, data_format="nz",
        )
        self.raw_l0b_pending.append(reinterpret(right, shape=(CHUNK_SIZE, CHUNK_SIZE), data_format="nz"))
        mem_copy(left, self.zero_l1)
        for block in range_constexpr(DIAGONAL_BLOCK_COUNT):
            source = reinterpret(
                diagonal, shape=(DIAGONAL_BLOCK_SIZE, DIAGONAL_BLOCK_SIZE),
                offset=self._packed64_block_offset_elems(block, block) * 2,
            )
            target = reinterpret(
                left, shape=(DIAGONAL_BLOCK_SIZE, DIAGONAL_BLOCK_SIZE),
                offset=self.l0_matrix_block_offset_elems(block, block) * 2,
            )
            mem_copy(target, source)
        mem_copy(reinterpret(right, shape=(CHUNK_SIZE, CHUNK_SIZE), data_format="nz"), identity, transpose=True)
        self._load_packed16_pair_to_full64_l0b_trans_with_gap(
            self.inverse_product_l1, right, dst_offset_elems=4 * 256,
            dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals(),
        )
        self.begin_mmad_raw_l0ab_db_packed64(self.raw_l0c_main, init_c=True, unit_flag=3)
        self.publish_inverse_product_fp16_l1(self.packed_inv64_l1, self.raw_l0c_main)
        self.compose_odd_even_lower32_to64_accum_full64(negative, self.packed_inv64_l1)

    @jit
    def compute_stable_scores(self, scores):
        left_k = self.l0a_scores.produce()
        left_q = self.l0a_scores.produce()
        for row_block in range_constexpr(DIAGONAL_BLOCK_COUNT):
            scores.build_per_d_left_k_and_decay(
                scores.ub_k_norm, scores.ub_g_cumsum, row_block, scores.l1_per_d_operand,
            )
            mem_copy(
                reinterpret(left_k, shape=(DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM)), scores.l1_per_d_operand.consume(),
            )
            scores.build_per_d_left_q_from_decay(scores.ub_q_norm, row_block, scores.l1_per_d_operand)
            mem_copy(
                reinterpret(left_q, shape=(DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM)), scores.l1_per_d_operand.consume(),
            )
            for col_block in range_constexpr(row_block):
                scores.build_per_d_right(
                    scores.ub_k_norm, scores.ub_g_cumsum, row_block, col_block, scores.l1_per_d_operand,
                )
                right = reinterpret(self.l0b_recurrent.produce(), shape=(16, SUPPORTED_HEAD_DIM), data_format="nz")
                mem_copy(right, scores.l1_per_d_operand.consume())
                matmul(
                    tile_slice(self.l0c_kkt, (DIAGONAL_BLOCK_SIZE, DIAGONAL_BLOCK_SIZE), (row_block, col_block)),
                    reinterpret(left_k, shape=(DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM)), right, init=True,
                )
                matmul(
                    tile_slice(self.l0c_qkt, (DIAGONAL_BLOCK_SIZE, DIAGONAL_BLOCK_SIZE), (row_block, col_block)),
                    reinterpret(left_q, shape=(DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM)), right, init=True,
                )
            scores.prepare_safe_diagonal_left()
            for active_row in dsl_range(0, DIAGONAL_BLOCK_SIZE, 1):
                scores.update_safe_diagonal_left(
                    scores.ub_k_norm, scores.ub_q_norm, row_block, active_row, scores.l1_per_d_operand,
                )
                mem_copy(
                    reinterpret(left_k, shape=(DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM)),
                    scores.l1_per_d_operand.consume(),
                )
                scores._write_per_d_full_d_handoff(scores.l1_per_d_operand, scores.ub_safe_left_q)
                mem_copy(
                    reinterpret(left_q, shape=(DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM)),
                    scores.l1_per_d_operand.consume(),
                )
                scores.build_safe_diagonal_right(
                    scores.ub_k_norm, scores.ub_g_cumsum, row_block, active_row, scores.l1_per_d_operand,
                )
                right = reinterpret(self.l0b_recurrent.produce(), shape=(16, SUPPORTED_HEAD_DIM), data_format="nz")
                mem_copy(right, scores.l1_per_d_operand.consume())
                matmul(
                    tile_slice(self.l0c_kkt, (DIAGONAL_BLOCK_SIZE, DIAGONAL_BLOCK_SIZE), (row_block, row_block)),
                    reinterpret(left_k, shape=(DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM)), right, init=active_row == 0,
                )
                matmul(
                    tile_slice(self.l0c_qkt, (DIAGONAL_BLOCK_SIZE, DIAGONAL_BLOCK_SIZE), (row_block, row_block)),
                    reinterpret(left_q, shape=(DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM)), right, init=active_row == 0,
                )
        mem_copy(scores.ub_kkt, self.l0c_kkt, engine=make_copy_engine(split_axis=0))
        mem_copy(scores.ub_mqk, self.l0c_qkt, engine=make_copy_engine(split_axis=0))

    @jit
    def compute_kkt_and_qkt(self, scores_and_inverse):
        if const_expr(scores_and_inverse.use_stable_scores):
            self.compute_stable_scores(scores_and_inverse)
        else:
            main = scores_and_inverse.l1_score_operands.consume()
            tail = tile_slice(main, (HALF_CHUNK_SIZE, SUPPORTED_HEAD_DIM), (8, 0))
            for kind in range_constexpr(2):
                destination = self.l0c_kkt if kind == 0 else self.l0c_qkt
                left = reinterpret(
                    self.l0a_scores.produce(), shape=(DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM), data_format="nz",
                )
                mem_copy(left, tile_slice(main, (DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM), (kind, 0)))
                next_left = left
                for row_block in range_constexpr(DIAGONAL_BLOCK_COUNT):
                    if const_expr(row_block > 0):
                        left = next_left
                    for col_block in range_constexpr(row_block + 1):
                        index = row_block * (row_block + 1) // 2 + col_block
                        right = self.k_transpose_l0b_cache[index]
                        if const_expr(kind == 0):
                            if const_expr(index == 0 or index == 2):
                                for cache_index in range_constexpr(index, index + 2):
                                    mem_copy(
                                        self.k_transpose_l0b_cache[cache_index],
                                        tile_slice(
                                            main, (DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM), (8 + cache_index, 0),
                                        ),
                                    )
                            elif const_expr(index == 4 or index == 5):
                                mem_copy(
                                    right, tile_slice(main, (DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM), (8 + index, 0)),
                                )
                            elif const_expr(index == 6):
                                for cache_index in range_constexpr(6, 10):
                                    if const_expr(cache_index < 8):
                                        mem_copy(
                                            self.k_transpose_l0b_cache[cache_index],
                                            tile_slice(
                                                main, (DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM,), (8 + cache_index, 0),
                                            ),
                                        )
                                    else:
                                        mem_copy(
                                            self.k_transpose_l0b_cache[cache_index],
                                            tile_slice(
                                                tail, (DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM,), (cache_index - 8, 0),
                                            ),
                                        )
                        matmul(
                            tile_slice(destination, (DIAGONAL_BLOCK_SIZE, DIAGONAL_BLOCK_SIZE), (row_block, col_block)),
                            left, right, init=True,
                        )
                        if const_expr(row_block + 1 < DIAGONAL_BLOCK_COUNT and col_block == 0):
                            next_left = reinterpret(
                                self.l0a_scores.produce(), shape=(DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM),
                                data_format="nz",
                            )
                            mem_copy(
                                next_left,
                                tile_slice(
                                    main, (DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM), (2 * (row_block + 1) + kind, 0),
                                ),
                            )
                if const_expr(kind == 0):
                    mem_copy(scores_and_inverse.ub_kkt, self.l0c_kkt, engine=make_copy_engine(split_axis=0))
                else:
                    mem_copy(scores_and_inverse.ub_mqk, self.l0c_qkt, engine=make_copy_engine(split_axis=0))

    @jit
    def compute_u_pre_and_w(self, preprocessing, recurrent_state):
        inverse_l0a = self._load_resident_inv_to_l0a()
        k_beta_l0b = self._load_beta_to_l0b(preprocessing.operand_tiles[K_DECAYED_BETA_OPERAND_INDEX])
        matmul(self.l0c_w, reinterpret(inverse_l0a, shape=(CHUNK_SIZE, CHUNK_SIZE), offset=0), k_beta_l0b, init=True)
        if self.save_backward_inputs != 0:
            mem_copy(
                tile_slice(
                    self.saved_w, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (self.chunk_idx, 0),
                )[: self.valid_rows, None],
                self.l0c_w, l2_cache_ctl=4,
            )
        # Saved W stays positive; FIXPIPE publishes -W for U_pre + (-W) @ S.
        mem_copy(self.w_l1.produce(), self.l0c_w, deq_scale=-1.0)
        v_beta_l0b = self._load_beta_to_l0b(preprocessing.operand_tiles[V_BETA_OPERAND_INDEX])
        matmul(self.l0c_w, reinterpret(inverse_l0a, shape=(CHUNK_SIZE, CHUNK_SIZE), offset=0), v_beta_l0b, init=True)
        if self.save_backward_inputs != 0:
            mem_copy(
                tile_slice(
                    self.saved_u, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (self.chunk_idx, 0),
                )[: self.valid_rows, None],
                self.l0c_w, l2_cache_ctl=4,
            )

    @jit
    def compute_w_state_and_q_state(self, preprocessing, recurrent_state):
        state_l0b = self.l0b_recurrent.produce()
        mem_copy(state_l0b, recurrent_state.state_l1.consume())
        w_l0a = self.l0a_recurrent.produce()
        mem_copy(w_l0a, self.w_l1.consume())
        q_l0a = self.l0a_recurrent.produce()
        mem_copy(q_l0a, preprocessing.operand_tiles[Q_DECAYED_OPERAND_INDEX])
        matmul(self.l0c_w, w_l0a, state_l0b, init=False)
        matmul(self.l0c_o, q_l0a, state_l0b, init=True)

        mem_copy(recurrent_state.u_l1.produce(), self.l0c_w)
        if self.save_backward_inputs != 0:
            mem_copy(
                tile_slice(
                    self.saved_v_new, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (self.chunk_idx, 0),
                )[:self.valid_rows, None],
                self.l0c_w, l2_cache_ctl=4,
            )

    @jit
    def compute_output_and_state_delta(
        self, preprocessing, recurrent_state, scores_and_inverse, out, chunk_idx
    ):
        k_panel = reinterpret(self.l0b_recurrent.produce(), shape=(D_HALF, SUPPORTED_HEAD_DIM))
        mem_copy(k_panel, preprocessing.operand_tiles[K_RESTORED_OPERAND_INDEX], transpose=True)
        causal_scores_l0a = reinterpret(self.l0a_recurrent.produce(), shape=(CHUNK_SIZE, CHUNK_SIZE))
        mem_copy(causal_scores_l0a, scores_and_inverse.mqk_l1.consume())
        u = recurrent_state.u_l1.consume()
        u_l0a = reinterpret(
            self.l0a_recurrent.produce(), shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), data_format="zn",
        )
        mem_copy(u_l0a, u, transpose=True)
        matmul(self.l0c_state, u_l0a, k_panel, init=True)
        u_l0b = reinterpret(self.l0b_recurrent.produce(), shape=(D_HALF, SUPPORTED_HEAD_DIM))
        mem_copy(u_l0b, u, transpose=True)
        matmul(self.l0c_o, causal_scores_l0a, u_l0b, init=False)
        mem_copy(recurrent_state.ub_state_delta, self.l0c_state, engine=make_copy_engine(split_axis=1))
        output_tile = tile_slice(out, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk_idx, 0))
        if const_expr(self.has_tail):
            valid_rows = self.sequence_length - chunk_idx * CHUNK_SIZE
            if valid_rows > CHUNK_SIZE:
                valid_rows = CHUNK_SIZE
            mem_copy(output_tile[:valid_rows, None], self.l0c_o, l2_cache_ctl=4)
        else:
            mem_copy(output_tile, self.l0c_o, l2_cache_ctl=4)

    def c64_odd_even_pair_dst_gap_fractals(self):
        return self.c64_diag_nz_offset_elems(2) // FRACTAL_ELEMENTS - 1

    def zero_packed64_l0_operands(self, l0a_slot, l0b_slot):
        mem_copy(l0a_slot, self.zero_l1)
        mem_copy(l0b_slot, self.zero_l1)

    def begin_mmad_raw_l0ab_db_packed64(self, c_tile, *, init_c: bool, unit_flag: int):
        """发起 full64 raw MMAD，显式指定 init/accumulate。"""
        l0a_read = self.raw_l0a_db.consume()
        l0b_read = self.raw_l0b_pending.pop(0)
        matmul(c_tile, l0a_read, l0b_read, init=init_c, unit_flag=unit_flag)

    def prepare_packed64_identity_l1(self, gm_identity16):
        """构造 packed 64x64 identity。"""
        identity_write = self.packed_identity_l1
        self.zero_fp16_l1(identity_write)
        cube_sync_pipe(PIPE.MTE2)
        for diagonal_tile in range_constexpr(DIAGONAL_BLOCK_COUNT):
            mem_copy(
                tile_slice(identity_write, (16, 16), (diagonal_tile, diagonal_tile)),
                tile_slice(gm_identity16, (16, 16), (0, 0)),
            )
        return self.packed_identity_l1

    def issue_odd_lower_even_inv_full64(self, scratch, packed_inv):
        l0a_write = self.raw_l0a_db.produce()
        l0b_write = reinterpret(
            self.l0b_recurrent.produce(), shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM),
            dtype=dtypes.float16, data_format="nz",
        )
        self.raw_l0b_pending.append(reinterpret(l0b_write, shape=(CHUNK_SIZE, CHUNK_SIZE), data_format="nz"))
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        self._load_packed16_pair_to_full64_l0a_with_gap(
            scratch, l0a_write, src0_row_block=1, src0_col_block=0, src1_row_block=3, src1_col_block=2,
            dst_offset_elems=DIAGONAL_BLOCK_COUNT * FRACTAL_ELEMENTS,
            dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals(),
        )
        self._load_packed16_pair_to_full64_l0b_trans_with_gap(
            packed_inv, l0b_write, dst_offset_elems=self.c64_diag_nz_offset_elems(0),
            dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals(),
        )

    def issue_odd32_lower_even32_inv_full64(self, scratch, packed_inv):
        l0a_write = self.raw_l0a_db.produce()
        l0b_write = reinterpret(
            self.l0b_recurrent.produce(), shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM),
            dtype=dtypes.float16, data_format="nz",
        )
        self.raw_l0b_pending.append(reinterpret(l0b_write, shape=(CHUNK_SIZE, CHUNK_SIZE), data_format="nz"))
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        mem_copy(l0a_write, scratch)
        self._load_packed32_block_to_full64_l0b(
            packed_inv, l0b_write, src_row=0, src_col=0, dst_row=0, dst_col=0, transpose=True,
        )

    def prepare_odd32_inv_odd32_tmp_full64(self, packed_inv):
        l0a_write = self.raw_l0a_db.produce()
        l0b_write = reinterpret(
            self.l0b_recurrent.produce(), shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM),
            dtype=dtypes.float16, data_format="nz",
        )
        self.raw_l0b_pending.append(reinterpret(l0b_write, shape=(CHUNK_SIZE, CHUNK_SIZE), data_format="nz"))
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        mem_copy(l0a_write, packed_inv)
        return l0b_write

    def complete_odd32_inv_odd32_tmp_full64(self, tmp64, l0b_write):
        self._load_packed32_block_to_full64_l0b(
            tmp64, l0b_write, src_row=HALF_CHUNK_SIZE, src_col=0, dst_row=HALF_CHUNK_SIZE, dst_col=0, transpose=True,
        )

    def compose_odd_even_lower32_to64_accum_full64(self, scratch, packed_inv):
        """组合 64x64 左下 32x32 lower 逆矩并生成 BF16 resident inverse。"""
        self.issue_odd32_lower_even32_inv_full64(scratch, packed_inv)
        second_l0b = self.prepare_odd32_inv_odd32_tmp_full64(packed_inv)
        c_tmp = self.raw_l0c_aux
        self.begin_mmad_raw_l0ab_db_packed64(c_tmp, init_c=True, unit_flag=3)
        c_tmp_read = self.raw_l0c_aux
        self.publish_inverse_product_fp16_l1(self.inverse_product_l1, c_tmp_read)
        tmp64 = self.inverse_product_l1
        self.complete_odd32_inv_odd32_tmp_full64(tmp64, second_l0b)
        c_inv = self.raw_l0c_main
        self.begin_mmad_raw_l0ab_db_packed64(c_inv, init_c=False, unit_flag=3)
        c_inv_read = self.raw_l0c_main
        self.publish_resident_inverse_bf16_l1(self.resident_inv_l1, c_inv_read)

    def _load_beta_to_l0b(self, beta_read):
        l0b_write = reinterpret(self.l0b_recurrent.produce(), shape=(D_HALF, SUPPORTED_HEAD_DIM))
        mem_copy(l0b_write, beta_read, transpose=True)
        return l0b_write

    def _raw_l0b_block(self, l0b_slot, row_block: int, col_block: int):
        return reinterpret(
            l0b_slot, shape=(DIAGONAL_BLOCK_SIZE, DIAGONAL_BLOCK_SIZE),
            offset=self.l0_matrix_block_offset_elems(row_block, col_block) * 2,
        )

    def _load_l1_block_to_l0b_block(
        self, l1_matrix, l0b_slot, row_block: int, col_block: int,
    ):
        """把一个 packed L1 16x16 fractal 转置装入 full64 L0B 的指定块位置。"""
        l0_tile = self._raw_l0b_block(l0b_slot, row_block, col_block)
        l1_tile = reinterpret(
            l1_matrix,
            shape=(DIAGONAL_BLOCK_SIZE, DIAGONAL_BLOCK_SIZE),
            offset=(col_block * CHUNK_SIZE * DIAGONAL_BLOCK_SIZE
                    + row_block * DIAGONAL_BLOCK_SIZE * DIAGONAL_BLOCK_SIZE)
            * 2,
        )
        mem_copy(l0_tile, l1_tile, transpose=True)

    def _load_packed32_block_to_full64_l0b(
        self,
        l1_matrix,
        l0b_slot,
        *,
        src_row: int,
        src_col: int,
        dst_row: int,
        dst_col: int,
        src_packed_size: int = CHUNK_SIZE,
        transpose: bool = True,
    ):
        for row_offset in (0, DIAGONAL_BLOCK_SIZE):
            l1_offset = (src_col // DIAGONAL_BLOCK_SIZE) * src_packed_size * DIAGONAL_BLOCK_SIZE
            l1_offset += (src_row + row_offset) * DIAGONAL_BLOCK_SIZE + (src_col % DIAGONAL_BLOCK_SIZE)
            l0_offset = self.l0_matrix_block_offset_elems(
                (dst_row + row_offset) // DIAGONAL_BLOCK_SIZE, dst_col // DIAGONAL_BLOCK_SIZE
            )
            l0_tile = reinterpret(l0b_slot, shape=(DIAGONAL_BLOCK_SIZE, HALF_CHUNK_SIZE), offset=l0_offset * 2)
            l1_tile = reinterpret(l1_matrix, shape=(DIAGONAL_BLOCK_SIZE, HALF_CHUNK_SIZE), offset=l1_offset * 2)
            mem_copy(l0_tile, l1_tile, transpose=transpose)

    def _load_packed16_pair_to_full64_l0a_with_gap(
        self,
        l1_matrix,
        l0a_slot,
        *,
        src0_row_block: int,
        src0_col_block: int,
        src1_row_block: int,
        src1_col_block: int,
        dst_offset_elems: int,
        dst_gap_fractals: int,
    ):
        """把一对 packed16 L1 block 装入 full64 L0A，目标块之间保留 gap。"""
        dst0_block = dst_offset_elems // FRACTAL_ELEMENTS
        dst1_block = dst0_block + dst_gap_fractals + 1
        for src_row_block, src_col_block, dst_block in (
            (src0_row_block, src0_col_block, dst0_block),
            (src1_row_block, src1_col_block, dst1_block),
        ):
            src_offset = self._packed64_block_offset_elems(src_row_block, src_col_block)
            # 3510 L0A 的 16x16 单块地址仍需补偿一个 full64 M span。
            dst_offset = (dst_block - (DIAGONAL_BLOCK_COUNT - 1)) * FRACTAL_ELEMENTS
            l0_tile = reinterpret(l0a_slot, shape=(DIAGONAL_BLOCK_SIZE, DIAGONAL_BLOCK_SIZE), offset=dst_offset * 2)
            l1_tile = reinterpret(l1_matrix, shape=(DIAGONAL_BLOCK_SIZE, DIAGONAL_BLOCK_SIZE), offset=src_offset * 2)
            mem_copy(l0_tile, l1_tile)

    def _load_packed16_pair_to_full64_l0b_trans_with_gap(
        self, l1_matrix, l0b_slot, *, dst_offset_elems: int, dst_gap_fractals: int
    ):
        """把一对 packed16 L1 block 转置装入 full64 L0B，目标块之间保留 gap。"""
        dst0_block = dst_offset_elems // FRACTAL_ELEMENTS
        dst1_block = dst0_block + dst_gap_fractals + 1
        self._load_l1_block_to_l0b_block(
            l1_matrix, l0b_slot, dst0_block // DIAGONAL_BLOCK_COUNT, dst0_block % DIAGONAL_BLOCK_COUNT,
        )
        self._load_l1_block_to_l0b_block(
            l1_matrix, l0b_slot, dst1_block // DIAGONAL_BLOCK_COUNT, dst1_block % DIAGONAL_BLOCK_COUNT,
        )

    def _load_resident_inv_to_l0a(self):
        resident_inv = self.resident_inv_l1
        l0a_write = self.l0a_recurrent.produce()
        mem_copy(reinterpret(l0a_write, shape=(CHUNK_SIZE, CHUNK_SIZE), offset=0), resident_inv)
        for block in range_constexpr(DIAGONAL_BLOCK_COUNT):
            source_column = block % 2 + 1
            source = reinterpret(
                self.tile_inv_l1, shape=(DIAGONAL_BLOCK_SIZE, DIAGONAL_BLOCK_SIZE),
                offset=self._packed64_block_offset_elems(block, source_column) * 2, dtype=dtypes.bfloat16,
            )
            target = reinterpret(
                l0a_write, shape=(DIAGONAL_BLOCK_SIZE, DIAGONAL_BLOCK_SIZE),
                offset=self.l0_matrix_block_offset_elems(block, block) * 2,
            )
            mem_copy(target, source)
        return l0a_write


class GateWeightedScoresAndDiagonalInverse:
    """Prepare relative-decay score panels and solve four diagonal 16x16 blocks."""

    def __init__(self, subblock_idx, preprocessing):
        self.subblock_idx = subblock_idx
        self.lower_bound = preprocessing.lower_bound
        self.ub_negative_kkt = reinterpret(
            preprocessing.ub_q_norm, dtype=dtypes.float32, shape=(32, 64), data_format="nd",
        )
        self.use_stable_scores = preprocessing.use_stable_scores
        score_rows = (
            7 * HALF_CHUNK_SIZE
            if self.use_stable_scores
            else 4 * CHUNK_SIZE + HALF_CHUNK_SIZE
        )
        self.ub_score_operands_nz = reinterpret(
            preprocessing.ub_operand_staging, shape=(score_rows, D_HALF), data_format="nz", n1_pad=16,
        )
        self.l1_score_operands = RingBuffer(MemLoc.L1, (score_rows, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=1)
        self.ub_kkt = RingBuffer(MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.float32, depth=1).produce()
        self.ub_mqk = RingBuffer(MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.float32, depth=1).produce()
        self.ub_negative_kkt_nz = reinterpret(
            preprocessing.ub_recurrent_operands_nz, dtype=dtypes.float16, shape=(HALF_CHUNK_SIZE, CHUNK_SIZE),
            data_format="nz", n1_pad=16,
        )
        self.ub_mqk_nz = reinterpret(
            preprocessing.ub_recurrent_operands_nz, shape=(HALF_CHUNK_SIZE, CHUNK_SIZE), offset=33 * 16 * 4 * 2,
            data_format="nz", n1_pad=16,
        )
        self.mqk_l1 = RingBuffer(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.bfloat16, depth=1)
        self.ub_diag_inv_nz = RingBuffer(
            MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1, data_format="nz", n1_pad=16,
        ).produce()
        self.ub_inverse_prefix = preprocessing.ub_inverse_prefix
        self.ub_k_norm = preprocessing.ub_k_norm
        self.ub_q_norm = preprocessing.ub_q_norm
        self.ub_g_cumsum = preprocessing.ub_g_cumsum
        self.scale_value = preprocessing.scale
        if self.use_stable_scores:
            self.ub_qk_exchange = reinterpret(
                preprocessing.ub_k[1], shape=(DIAGONAL_BLOCK_SIZE, D_HALF), dtype=dtypes.float32, data_format="nd",
            )
            self.ub_per_d_nd = reinterpret(
                preprocessing.ub_operand_staging, shape=(DIAGONAL_BLOCK_SIZE, D_HALF), data_format="nd",
            )
            self.ub_per_d_nz = reinterpret(
                preprocessing.ub_operand_staging, shape=(DIAGONAL_BLOCK_SIZE, D_HALF),
                offset=DIAGONAL_BLOCK_SIZE * D_HALF * 2, data_format="nz", n1_pad=16,
            )
            self.ub_safe_left_k = reinterpret(
                preprocessing.ub_q[0], shape=(DIAGONAL_BLOCK_SIZE, D_HALF), data_format="nd",
            )
            self.ub_safe_left_q = reinterpret(
                preprocessing.ub_q[1], shape=(DIAGONAL_BLOCK_SIZE, D_HALF), data_format="nd",
            )
            self.ub_safe_right_k = reinterpret(
                preprocessing.ub_k[0], shape=(DIAGONAL_BLOCK_SIZE, D_HALF), data_format="nd",
            )
            self.l1_per_d_operand = RingBuffer(
                MemLoc.L1, (DIAGONAL_BLOCK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=1,
            )

        # Bound by the task dispatcher before the corresponding stage runs.
        self.causal_qkt_nd = None

    @jit
    def prepare_score_operands(self, preprocessing, q, k, chunk_idx, chunk_num):
        self.prepare_score_panel(preprocessing, 0)
        if const_expr(not self.use_stable_scores):
            if chunk_idx + 1 < chunk_num:
                preprocessing.prefetch_qk_front(q, k, chunk_idx + 1)
        self.prepare_score_panel(preprocessing, 1)
        if const_expr(not self.use_stable_scores):
            score_slot = self.l1_score_operands.produce()
            mem_copy(
                score_slot, self.ub_score_operands_nz, engine=make_copy_engine(split_axis=1), part_id=self.subblock_idx,
            )
        if const_expr(not self.use_stable_scores):
            if chunk_idx + 1 < chunk_num:
                preprocessing.prefetch_qk_back(q, k, chunk_idx + 1)

    @jit
    def prepare_score_panel(self, preprocessing, panel_idx):
        """Normalize raw Q/K and pack bounded score operands using 16-row references."""
        q_panel, k_panel = (
            preprocessing.ub_q[panel_idx],
            preprocessing.ub_k[panel_idx],
        )
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            low_mask, _ = update_mask(D_HALF, elem_bits=16)
            for qk_half_idx in range_constexpr(2):
                col_block = panel_idx * 2 + qk_half_idx
                for row in dsl_range(0, DIAGONAL_BLOCK_SIZE, 1, unroll=2):
                    panel_row = qk_half_idx * DIAGONAL_BLOCK_SIZE + row
                    logical_row = panel_idx * HALF_CHUNK_SIZE + panel_row
                    element_offset = logical_row * D_HALF
                    q_norm_scale = vload_broadcast(preprocessing.ub_qk_norms, logical_row)
                    k_norm_scale = vload_broadcast(preprocessing.ub_qk_norms, CHUNK_SIZE + logical_row)
                    raw_q = vcast(
                        vload_unpack(q_panel, panel_row * D_HALF, unpack_mode=UnpackMode.B16_TO_B32), dtypes.float32,
                        mask=lane_mask,
                    )
                    raw_k = vcast(
                        vload_unpack(k_panel, panel_row * D_HALF, unpack_mode=UnpackMode.B16_TO_B32), dtypes.float32,
                        mask=lane_mask,
                    )
                    q_normalized = vcast(vmul(raw_q, q_norm_scale, mask=lane_mask), dtypes.bfloat16, mask=lane_mask)
                    k_normalized = vcast(vmul(raw_k, k_norm_scale, mask=lane_mask), dtypes.bfloat16, mask=lane_mask)
                    vstore_pack(
                        preprocessing.ub_q_norm, element_offset, q_normalized, lane_mask, pack_mode=PackMode.B32_TO_B16,
                    )
                    vstore_pack(
                        preprocessing.ub_k_norm, element_offset, k_normalized, lane_mask, pack_mode=PackMode.B32_TO_B16,
                    )
                    q_value = vcast(q_normalized, dtypes.float32, mask=lane_mask)
                    k_value = vcast(k_normalized, dtypes.float32, mask=lane_mask)
                    if const_expr(not self.use_stable_scores):
                        gate = vload(preprocessing.ub_g_cumsum, element_offset)
                        reference = vload(preprocessing.ub_g_cumsum, col_block * DIAGONAL_BLOCK_SIZE * D_HALF)
                        decay = vexp_sub(gate, reference, mask=lane_mask)
                        self.store_score_operand16(
                            cast_and_pack_bf16(vmul(k_value, decay, mask=lane_mask), lane_mask), 2 * col_block, row,
                            low_mask,
                        )
                        self.store_score_operand16(
                            cast_and_pack_bf16(
                                vmuls(vmul(q_value, decay, mask=lane_mask), preprocessing.scale, mask=lane_mask),
                                lane_mask,
                            ),
                            2 * col_block + 1, row, low_mask,
                        )
                        for row_block in range_constexpr(DIAGONAL_BLOCK_COUNT):
                            if const_expr(row_block >= col_block):
                                right_reference = vload(
                                    preprocessing.ub_g_cumsum, row_block * DIAGONAL_BLOCK_SIZE * D_HALF,
                                )
                                right_decay = vexp_sub(right_reference, gate, mask=lane_mask)
                                right_tile = (
                                    8 + row_block * (row_block + 1) // 2 + col_block
                                )
                                self.store_score_operand16(
                                    cast_and_pack_bf16(vmul(k_value, right_decay, mask=lane_mask), lane_mask),
                                    right_tile, row, low_mask,
                                )

    def store_score_operand16(self, value, tile, row, lane_mask):
        vstore_strided(
            self.ub_score_operands_nz, (tile * DIAGONAL_BLOCK_SIZE + row) * 16, value, lane_mask,
            block_stride=self.ub_score_operands_nz.physical_stride[0] // 16,
        )

    @jit
    def compute_diagonal_inverse16(self, matmul_stage, chunk_idx):
        if chunk_idx == 0:
            with vf(mode="simd"):
                inverse_fp32, _ = update_mask(CHUNK_SIZE, elem_bits=32)
                low, _ = update_mask(CHUNK_SIZE, elem_bits=16)
                zero = vdups(0.0, dtypes.float32, mask=inverse_fp32)
                inverse_fp16 = vcast(zero, dtypes.float16, mask=inverse_fp32)
                packed_inverse = vreinterpret(
                    vpack(vreinterpret_lanes(inverse_fp16, dtypes.uint32), dtypes.uint16), dtypes.float16,
                )
                for row in dsl_range(0, HALF_CHUNK_SIZE, 1, unroll=4):
                    vstore_strided(
                        self.ub_diag_inv_nz, row * 16, packed_inverse, low,
                        block_stride=self.ub_diag_inv_nz.physical_stride[0] // 16,
                    )
                vmem_bar("vst_vld")
        for start, count in ((0, 11), (11, 5)):
            with vf(mode="simd"):
                self.solve_diagonal_inverse_rows(start, count)
                vmem_bar("vst_vld")
        mem_copy(
            matmul_stage.tile_inv_l1, self.ub_diag_inv_nz, engine=make_copy_engine(split_axis=0),
            part_id=self.subblock_idx,
        )

    @jit
    def solve_diagonal_inverse_rows(self, start, count):
        """Solve consecutive rows of each 16x16 diagonal inverse with ascending FP32 FMA."""
        lane_mask, _ = update_mask(32, elem_bits=32)
        mask_b16, _ = update_mask(32, elem_bits=16)
        zero = vdups(0.0, dtypes.float32, mask=lane_mask)
        one = vdups(1.0, dtypes.float32, mask=lane_mask)
        lane_idx = varange(0, dtypes.uint32)
        block_idx = vbitwise_and(lane_idx, vdups(16, dtypes.uint32), mask=lane_mask)
        lane_in_block = vbitwise_and(lane_idx, vdups(15, dtypes.uint32), mask=lane_mask)
        block_base = vadd(vshl(block_idx, 6, mask=lane_mask), block_idx, mask=lane_mask)
        block_base = vadds(block_base, self.subblock_idx * 32, mask=lane_mask)
        inverse_rows = []
        for row in range_constexpr(start, start + count):
            diagonal = veq(lane_in_block, vdups(row, dtypes.uint32), mask=lane_mask)
            inverse_rows.append(vselect_raw(one, zero, cond_mask=diagonal))
        if const_expr(start > 0):
            row_acc0 = inverse_rows[0]
            row_acc1 = inverse_rows[1]
            row_acc2 = inverse_rows[2]
            row_acc3 = inverse_rows[3]
            row_acc4 = inverse_rows[4]
            for col in dsl_range(0, start, 1, unroll=1):
                prefix_inverse = vload(self.ub_inverse_prefix, col * 32)
                indices = vadds(block_base, (start + 0) * 64 + col, mask=lane_mask)
                value = vgather(self.ub_negative_kkt, indices, mask=lane_mask)
                row_acc0 = vmadd(value, prefix_inverse, row_acc0, mask=lane_mask)
                indices = vadds(block_base, (start + 1) * 64 + col, mask=lane_mask)
                value = vgather(self.ub_negative_kkt, indices, mask=lane_mask)
                row_acc1 = vmadd(value, prefix_inverse, row_acc1, mask=lane_mask)
                indices = vadds(block_base, (start + 2) * 64 + col, mask=lane_mask)
                value = vgather(self.ub_negative_kkt, indices, mask=lane_mask)
                row_acc2 = vmadd(value, prefix_inverse, row_acc2, mask=lane_mask)
                indices = vadds(block_base, (start + 3) * 64 + col, mask=lane_mask)
                value = vgather(self.ub_negative_kkt, indices, mask=lane_mask)
                row_acc3 = vmadd(value, prefix_inverse, row_acc3, mask=lane_mask)
                indices = vadds(block_base, (start + 4) * 64 + col, mask=lane_mask)
                value = vgather(self.ub_negative_kkt, indices, mask=lane_mask)
                row_acc4 = vmadd(value, prefix_inverse, row_acc4, mask=lane_mask)
            inverse_rows = [row_acc0, row_acc1, row_acc2, row_acc3, row_acc4]
        for row in range_constexpr(start, start + count):
            acc = inverse_rows[row - start]
            for col in range_constexpr(start, row):
                indices = vadds(block_base, row * 64 + col, mask=lane_mask)
                value = vgather(self.ub_negative_kkt, indices, mask=lane_mask)
                acc = vmadd(value, inverse_rows[col - start], acc, mask=lane_mask)
            inverse_rows[row - start] = acc
            if const_expr(start == 0):
                vstore(self.ub_inverse_prefix, row * 32, acc, lane_mask)
            inverse_fp16 = vcast(acc, dtypes.float16, mask=lane_mask)
            packed_inverse = vreinterpret(
                vpack(vreinterpret_lanes(inverse_fp16, dtypes.uint32), dtypes.uint16), dtypes.float16,
            )
            vstore_strided(
                self.ub_diag_inv_nz, self.subblock_idx * 2 * self.ub_diag_inv_nz.physical_stride[0] + row * 16,
                packed_inverse, mask_b16, block_stride=16 + self.ub_diag_inv_nz.physical_stride[0] // 16,
            )
            diagonal_final_bf16 = cast_and_pack_bf16(acc, lane_mask)
            diagonal_final_bits = vreinterpret(diagonal_final_bf16, dtypes.float16)
            vstore_strided(
                self.ub_diag_inv_nz, self.ub_diag_inv_nz.physical_stride[0] + row * 16, diagonal_final_bits, mask_b16,
                block_stride=16 + self.ub_diag_inv_nz.physical_stride[0] // 16,
            )

    @jit
    def apply_beta_and_mask_kkt(self, preprocessing, matmul_stage):
        with vf(mode="simd"):
            lane_mask, _ = update_mask(CHUNK_SIZE, elem_bits=32)
            low, _ = update_mask(CHUNK_SIZE, elem_bits=16)
            zero = vdups(0.0, dtypes.float32, mask=lane_mask)
            for row in dsl_range(0, HALF_CHUNK_SIZE, 1, unroll=4):
                row_offset = self.subblock_idx * HALF_CHUNK_SIZE + row
                strict_lower_mask, _ = update_mask(row_offset, elem_bits=32)
                raw_beta = vload_broadcast(preprocessing.ub_beta, row_offset)
                kkt_value = vmul(vload(self.ub_kkt, row * CHUNK_SIZE), raw_beta, mask=lane_mask)
                kkt_value = vselect_raw(kkt_value, zero, cond_mask=strict_lower_mask)
                negative_kkt = vmuls(kkt_value, -1.0, mask=lane_mask)
                kkt_fp16 = vcast(negative_kkt, dtypes.float16, mask=lane_mask)
                vstore(
                    self.ub_negative_kkt, row * CHUNK_SIZE, vcast(kkt_fp16, dtypes.float32, mask=lane_mask), lane_mask,
                )
                packed_kkt = vreinterpret(
                    vpack(vreinterpret_lanes(kkt_fp16, dtypes.uint32), dtypes.uint16), dtypes.float16,
                )
                vstore_strided(
                    self.ub_negative_kkt_nz, row * 16, packed_kkt, low,
                    block_stride=self.ub_negative_kkt_nz.physical_stride[0] // 16,
                )
            vmem_bar("vst_vld")
        mem_copy(
            matmul_stage.negative_kkt_l1, self.ub_negative_kkt_nz, engine=make_copy_engine(split_axis=0),
            part_id=self.subblock_idx,
        )

    @jit
    def mask_and_publish_causal_qkt(self):
        with vf(mode="simd"):
            lane_mask, _ = update_mask(CHUNK_SIZE, elem_bits=32)
            low_mask, _ = update_mask(CHUNK_SIZE, elem_bits=16)
            zero = vdups(0.0, dtypes.float32, mask=lane_mask)
            for row in dsl_range(0, HALF_CHUNK_SIZE, 1, unroll=4):
                row_offset = self.subblock_idx * HALF_CHUNK_SIZE + row
                causal_mask, _ = update_mask(row_offset + 1, elem_bits=32)
                value_mask = vselect_raw(vload(self.ub_mqk, row * CHUNK_SIZE), zero, cond_mask=causal_mask)
                mqk_bf16 = vcast(value_mask, dtypes.bfloat16, mask=lane_mask)
                mqk_value = vreinterpret(
                    vpack(vreinterpret_lanes(mqk_bf16, dtypes.uint32), dtypes.uint16), dtypes.bfloat16,
                )
                vstore_pack(self.causal_qkt_nd, row * CHUNK_SIZE, mqk_bf16, lane_mask, pack_mode=PackMode.B32_TO_B16)
                vstore_strided(
                    self.ub_mqk_nz, row * 16, mqk_value, low_mask, block_stride=self.ub_mqk_nz.physical_stride[0] // 16,
                )
        mem_copy(
            self.mqk_l1.produce(), self.ub_mqk_nz, engine=make_copy_engine(split_axis=0), part_id=self.subblock_idx,
        )

    @jit
    def build_per_d_left_k_and_decay(
        self, normalized_k_dhalf, gate_cumsum_dhalf, row_block: int, left_k_l1_operand
    ):
        k_read = tile_slice(normalized_k_dhalf, (DIAGONAL_BLOCK_SIZE, D_HALF), (row_block, 0))
        decay_scratch = tile_slice(self.ub_qk_exchange, (DIAGONAL_BLOCK_SIZE, D_HALF), (0, 0))
        ref_offset = row_block * DIAGONAL_BLOCK_SIZE * D_HALF
        nd_write = self.ub_per_d_nd
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            ref = vload(gate_cumsum_dhalf, ref_offset)
            for row in dsl_range(0, DIAGONAL_BLOCK_SIZE, 1, unroll=2):
                offset = row * D_HALF
                g_offset = (row_block * DIAGONAL_BLOCK_SIZE + row) * D_HALF
                unpacked_k = vload_unpack(k_read, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k = vcast(unpacked_k, dtypes.float32, mask=lane_mask)
                gate_cumsum = vload(gate_cumsum_dhalf, g_offset)
                decay = vexp_sub(gate_cumsum, ref, mask=lane_mask)
                vstore(decay_scratch, offset, decay, lane_mask)
                k_decayed = vmul(k, decay, mask=lane_mask)
                k_decayed_bf16 = vcast(k_decayed, dtypes.bfloat16, mask=lane_mask)
                vstore_pack(nd_write, offset, k_decayed_bf16, lane_mask, pack_mode="b32_to_b16")
            vmem_bar("vst_vld")
        nd_read = self.ub_per_d_nd
        self._write_per_d_full_d_handoff(left_k_l1_operand, nd_read)

    @jit
    def build_per_d_left_q_from_decay(
        self, normalized_q_dhalf, row_block: int, left_q_l1_operand
    ):
        q_read = tile_slice(normalized_q_dhalf, (DIAGONAL_BLOCK_SIZE, D_HALF), (row_block, 0))
        decay_scratch = tile_slice(self.ub_qk_exchange, (DIAGONAL_BLOCK_SIZE, D_HALF), (0, 0))
        nd_write = self.ub_per_d_nd
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            for row in dsl_range(0, DIAGONAL_BLOCK_SIZE, 1, unroll=2):
                offset = row * D_HALF
                unpacked_q = vload_unpack(q_read, offset, unpack_mode=UnpackMode.B16_TO_B32)
                q = vcast(unpacked_q, dtypes.float32, mask=lane_mask)
                decay = vload(decay_scratch, offset)
                q_decayed = vmul(q, decay, mask=lane_mask)
                scaled_q = vmuls(q_decayed, self.scale_value, mask=lane_mask)
                scaled_q_bf16 = vcast(scaled_q, dtypes.bfloat16, mask=lane_mask)
                vstore_pack(nd_write, offset, scaled_q_bf16, lane_mask, pack_mode="b32_to_b16")
        nd_read = self.ub_per_d_nd
        self._write_per_d_full_d_handoff(left_q_l1_operand, nd_read)

    @jit
    def build_per_d_right(
        self,
        normalized_k_dhalf,
        gate_cumsum_dhalf,
        row_block: int,
        col_block: int,
        right_k_l1_operand,
    ):
        k_read = tile_slice(normalized_k_dhalf, (DIAGONAL_BLOCK_SIZE, D_HALF), (col_block, 0))
        ref_offset = row_block * DIAGONAL_BLOCK_SIZE * D_HALF
        nd_write = self.ub_per_d_nd
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            ref = vload(gate_cumsum_dhalf, ref_offset)
            for row in dsl_range(0, DIAGONAL_BLOCK_SIZE, 1, unroll=2):
                offset = row * D_HALF
                g_offset = (col_block * DIAGONAL_BLOCK_SIZE + row) * D_HALF
                unpacked_k = vload_unpack(k_read, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k = vcast(unpacked_k, dtypes.float32, mask=lane_mask)
                gate_cumsum = vload(gate_cumsum_dhalf, g_offset)
                decay = vexp_sub(ref, gate_cumsum, mask=lane_mask)
                restored_k = vmul(k, decay, mask=lane_mask)
                restored_k_bf16 = vcast(restored_k, dtypes.bfloat16, mask=lane_mask)
                vstore_pack(nd_write, offset, restored_k_bf16, lane_mask, pack_mode="b32_to_b16")
        nd_read = self.ub_per_d_nd
        self._write_per_d_full_d_handoff(right_k_l1_operand, nd_read)

    @jit
    def prepare_safe_diagonal_left(self):
        """Initialize both persistent row masks once per diagonal block."""
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            zero = vcast(vdups(0.0, dtypes.float32, mask=lane_mask), dtypes.bfloat16, mask=lane_mask)
            for row in dsl_range(0, DIAGONAL_BLOCK_SIZE, 1):
                offset = row * D_HALF
                vstore_pack(self.ub_safe_left_k, offset, zero, lane_mask, pack_mode="b32_to_b16")
                vstore_pack(self.ub_safe_left_q, offset, zero, lane_mask, pack_mode="b32_to_b16")
                vstore_pack(self.ub_safe_right_k, offset, zero, lane_mask, pack_mode="b32_to_b16")
            vmem_bar("vst_vld")

    @jit
    def update_safe_diagonal_left(
        self,
        normalized_k_dhalf,
        normalized_q_dhalf,
        row_block: int,
        active_row,
        left_l1_operand,
    ):
        if active_row > 0:
            self._vf_update_safe_diagonal_rows(normalized_k_dhalf, normalized_q_dhalf, row_block, active_row, True)
        else:
            self._vf_update_safe_diagonal_rows(normalized_k_dhalf, normalized_q_dhalf, row_block, active_row, False)
        self._write_per_d_full_d_handoff(left_l1_operand, self.ub_safe_left_k)

    @jit
    def build_safe_diagonal_right(
        self,
        normalized_k_dhalf,
        gate_cumsum_dhalf,
        row_block: int,
        active_row,
        right_k_l1_operand,
    ):
        source = tile_slice(normalized_k_dhalf, (DIAGONAL_BLOCK_SIZE, D_HALF), (row_block, 0))
        nd_write = self.ub_safe_right_k
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            ref = vload(gate_cumsum_dhalf, (row_block * DIAGONAL_BLOCK_SIZE + active_row) * D_HALF)
            for row in dsl_range(0, active_row + 1, 1):
                offset = row * D_HALF
                unpacked_k = vload_unpack(source, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k = vcast(unpacked_k, dtypes.float32, mask=lane_mask)
                gate = vload(gate_cumsum_dhalf, (row_block * DIAGONAL_BLOCK_SIZE + row) * D_HALF)
                decay = vexp_sub(ref, gate, mask=lane_mask)
                decayed_k = vmul(k, decay, mask=lane_mask)
                decayed_bf16 = vcast(decayed_k, dtypes.bfloat16, mask=lane_mask)
                vstore_pack(nd_write, offset, decayed_bf16, lane_mask, pack_mode="b32_to_b16")
            vmem_bar("vst_vld")
        self._write_per_d_full_d_handoff(right_k_l1_operand, nd_write)

    @jit
    def solve_diagonal_blocks(self, reference_low2):
        for start, count in ((0, 11), (11, 5)):
            with vf(mode="simd"):
                self.solve_diagonal_inverse_rows(start, count, reference_low2)
                vmem_bar("vst_vld")

    @jit
    def _write_per_d_full_d_handoff(self, full_ring_buffer, nd_slot):
        nz_slot = self.ub_per_d_nz
        pack_score_operand_nz(nz_slot, nd_slot)
        mem_copy(full_ring_buffer.produce(), nz_slot, engine=make_copy_engine(split_axis=1), part_id=self.subblock_idx)

    @jit
    def _vf_update_safe_diagonal_rows(
        self,
        normalized_k_dhalf,
        normalized_q_dhalf,
        row_block,
        active_row,
        clear_previous,
    ):
        k_source = tile_slice(normalized_k_dhalf, (DIAGONAL_BLOCK_SIZE, D_HALF), (row_block, 0))
        q_source = tile_slice(normalized_q_dhalf, (DIAGONAL_BLOCK_SIZE, D_HALF), (row_block, 0))
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            if const_expr(clear_previous):
                zero = vcast(vdups(0.0, dtypes.float32, mask=lane_mask), dtypes.bfloat16, mask=lane_mask)
                old_offset = (active_row - 1) * D_HALF
                vstore_pack(self.ub_safe_left_k, old_offset, zero, lane_mask, pack_mode="b32_to_b16")
                vstore_pack(self.ub_safe_left_q, old_offset, zero, lane_mask, pack_mode="b32_to_b16")
            offset = active_row * D_HALF
            unpacked_k = vload_unpack(k_source, offset, unpack_mode=UnpackMode.B16_TO_B32)
            unpacked_q = vload_unpack(q_source, offset, unpack_mode=UnpackMode.B16_TO_B32)
            k = vcast(unpacked_k, dtypes.float32, mask=lane_mask)
            q = vcast(unpacked_q, dtypes.float32, mask=lane_mask)
            scaled_q = vmuls(q, self.scale_value, mask=lane_mask)
            vstore_pack(
                self.ub_safe_left_k, offset, vcast(k, dtypes.bfloat16, mask=lane_mask), lane_mask,
                pack_mode="b32_to_b16",
            )
            vstore_pack(
                self.ub_safe_left_q, offset, vcast(scaled_q, dtypes.bfloat16, mask=lane_mask), lane_mask,
                pack_mode="b32_to_b16",
            )
            vmem_bar("vst_vld")


@jit
def run_persistent_training_sequence(
    preprocessing,
    recurrent_state,
    matmul_stage,
    scores_and_inverse,
    q,
    k,
    v,
    g,
    beta,
    state_in,
    state_out,
    out,
    a_log,
    bias,
    identity,
    norm_exchange,
    block_idx,
    value_head_idx,
    sequence_length,
    scale,
    lower_bound,
    has_tail,
    caches,
):
    """Keep one head's state resident while processing every 64-token chunk."""
    preprocessing.has_tail = has_tail
    preprocessing.sequence_length = sequence_length
    matmul_stage.has_tail = has_tail
    matmul_stage.sequence_length = sequence_length
    chunk_num = (sequence_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    if preprocessing.use_gate_in_kernel != 0:
        preprocessing.load_head(a_log, bias, value_head_idx)
    recurrent_state.load(state_in)
    vec_sync_notify(PIPE.V, PIPE.MTE2, 6)
    preprocessing.load_chunk_inputs(q, k, v, g, beta, 0)
    preprocessing.compute_norms_and_activations(lower_bound, norm_exchange, block_idx, 0)
    scores_and_inverse.prepare_score_operands(preprocessing, q, k, 0, chunk_num)
    caches.store_preprocessed(preprocessing, 0)
    preprocessing.prepare_recurrent_operands(scale, recurrent_state.gamma_ub.produce(), 0)
    for chunk_idx in range(chunk_num):
        caches.store_entry_state(recurrent_state, chunk_idx)
        preprocessing.acquire_recurrent_operands()
        if chunk_idx + 1 < chunk_num:
            vec_sync_notify(PIPE.V, PIPE.MTE2, 6)
            preprocessing.load_chunk_inputs(q, k, v, g, beta, chunk_idx + 1)
        matmul_stage.compute_kkt_and_qkt(scores_and_inverse)
        scores_and_inverse.apply_beta_and_mask_kkt(preprocessing, matmul_stage)
        scores_and_inverse.compute_diagonal_inverse16(matmul_stage, chunk_idx)
        if chunk_idx + 1 < chunk_num:
            if const_expr(scores_and_inverse.use_stable_scores):
                preprocessing.prefetch_qk_front(q, k, chunk_idx + 1)
                preprocessing.prefetch_qk_back(q, k, chunk_idx + 1)
            preprocessing.compute_norms_and_activations(lower_bound, norm_exchange, block_idx, chunk_idx + 1)
            scores_and_inverse.prepare_score_operands(preprocessing, q, k, chunk_idx + 1, chunk_num)
            caches.store_preprocessed(preprocessing, chunk_idx + 1)
        scores_and_inverse.mask_and_publish_causal_qkt()
        caches.store_causal_qkt(scores_and_inverse, chunk_idx)
        matmul_stage.compose_chunk_inverse64(identity)
        caches.store_inverse(matmul_stage, scores_and_inverse, chunk_idx)
        matmul_stage.chunk_idx = chunk_idx
        matmul_stage.valid_rows = caches.valid_rows(chunk_idx)
        matmul_stage.compute_u_pre_and_w(preprocessing, recurrent_state)
        matmul_stage.compute_w_state_and_q_state(preprocessing, recurrent_state)
        matmul_stage.compute_output_and_state_delta(preprocessing, recurrent_state, scores_and_inverse, out, chunk_idx)
        if chunk_idx + 1 < chunk_num:
            preprocessing.prepare_recurrent_operands(scale, recurrent_state.gamma_ub.produce(), chunk_idx + 1)
        recurrent_state.update_recurrent_state()
        if chunk_idx + 1 < chunk_num:
            recurrent_state.publish_state_snapshot()
    recurrent_state.store(state_out)


class TrainingCacheWriter:
    """Save public backward inputs through one shared, bounded UB conversion tile."""

    def __init__(self, subblock_idx):
        self.subblock_idx = subblock_idx
        self.ub_cast_bf16 = Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16)

        # Bound by the task dispatcher before the corresponding stage runs.
        self.save_backward_inputs = None
        self.save_intermediate_state = None
        self.sequence_length = None
        self.write_qk = None
        self.q_norm = None
        self.k_norm = None
        self.q_rstd = None
        self.k_rstd = None
        self.beta_activated = None
        self.qg = None
        self.kg = None
        self.g_cumsum = None
        self.v_new = None
        self.Aqk = None
        self.Akk = None
        self.h = None

    @jit
    def valid_rows(self, chunk_idx):
        rows = self.sequence_length - chunk_idx * CHUNK_SIZE
        if rows > CHUNK_SIZE:
            rows = CHUNK_SIZE
        return rows

    @jit
    def store_preprocessed(self, preprocessing, chunk_idx):
        if self.save_backward_inputs != 0:
            rows = self.valid_rows(chunk_idx)
            if self.write_qk:
                mem_copy(
                    tile_slice(self.q_norm, (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))[:rows, None],
                    preprocessing.ub_q_norm[:rows, None], l2_cache_ctl=0,
                )
                mem_copy(
                    tile_slice(self.k_norm, (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))[:rows, None],
                    preprocessing.ub_k_norm[:rows, None], l2_cache_ctl=0,
                )
                if self.subblock_idx == 0:
                    mem_copy(
                        tile_slice(self.q_rstd, (CHUNK_SIZE, 1), (chunk_idx, 0))[:rows, :1],
                        reinterpret(preprocessing.ub_qk_norms, shape=(CHUNK_SIZE, 1), data_format="nd"), l2_cache_ctl=4,
                    )
                    mem_copy(
                        tile_slice(self.k_rstd, (CHUNK_SIZE, 1), (chunk_idx, 0))[:rows, :1],
                        reinterpret(
                            preprocessing.ub_qk_norms, shape=(CHUNK_SIZE, 1), offset=CHUNK_SIZE * 4, data_format="nd",
                        ),
                        l2_cache_ctl=4,
                    )
            if self.subblock_idx == 0:
                mem_copy(
                    tile_slice(self.beta_activated, (CHUNK_SIZE, 1), (chunk_idx, 0))[:rows, :1], preprocessing.ub_beta,
                    l2_cache_ctl=4,
                )
            mem_copy(
                tile_slice(self.g_cumsum, (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))[:rows, None],
                preprocessing.ub_g_cumsum[:rows, None], l2_cache_ctl=4,
            )

    @jit
    def store_entry_state(self, recurrent_state, chunk_idx):
        if self.save_intermediate_state != 0:
            mem_copy(
                tile_slice(self.h[0, 0, chunk_idx, None, None], (SUPPORTED_HEAD_DIM, D_HALF), (0, self.subblock_idx)),
                recurrent_state.ub_state_fp32, l2_cache_ctl=4,
            )

    @jit
    def store_causal_qkt(self, scores, chunk_idx):
        rows = self.valid_rows(chunk_idx) - self.subblock_idx * HALF_CHUNK_SIZE
        if rows > HALF_CHUNK_SIZE:
            rows = HALF_CHUNK_SIZE
        if rows > 0:
            mem_copy(
                tile_slice(
                    self.Aqk[0, 0, chunk_idx, None, None], (HALF_CHUNK_SIZE, CHUNK_SIZE), (self.subblock_idx, 0),
                )[:rows, None],
                self.ub_cast_bf16[:rows, None], l2_cache_ctl=4,
            )

    @jit
    def store_inverse(self, matmul_stage, scores, chunk_idx):
        """Assemble final BF16 cache in existing UB, then publish each valid half once."""
        inverse = self.Akk[0, 0, chunk_idx, None, None]
        valid_rows = self.valid_rows(chunk_idx)
        mem_copy(scores.ub_kkt, matmul_stage.raw_l0c_main, engine=make_copy_engine(split_axis=0))
        conversion = reinterpret(self.ub_cast_bf16, shape=(HALF_CHUNK_SIZE, CHUNK_SIZE))
        with vf(mode="simd"):
            lane_mask, _ = update_mask(CHUNK_SIZE, elem_bits=32)
            for row in dsl_range(0, HALF_CHUNK_SIZE, 1, unroll=4):
                value = vcast(vload(scores.ub_kkt, row * CHUNK_SIZE), dtypes.bfloat16, mask=lane_mask)
                vstore_pack(conversion, row * CHUNK_SIZE, value, lane_mask, pack_mode=PackMode.B32_TO_B16)
        diagonal = reinterpret(scores.ub_diag_inv_nz, dtype=dtypes.bfloat16)
        for local_block in range_constexpr(2):
            with vf(mode="simd"):
                lane_mask, _ = update_mask(DIAGONAL_BLOCK_SIZE, elem_bits=32)
                for row in dsl_range(0, DIAGONAL_BLOCK_SIZE, 1):
                    source_offset = (
                        (local_block + 1) * scores.ub_diag_inv_nz.physical_stride[0]
                        + local_block * FRACTAL_ELEMENTS
                        + row * DIAGONAL_BLOCK_SIZE
                    )
                    value = vload_unpack(diagonal, source_offset, unpack_mode=UnpackMode.B16_TO_B32)
                    target_offset = (
                        local_block * DIAGONAL_BLOCK_SIZE + row
                    ) * CHUNK_SIZE + (
                        self.subblock_idx * 2 + local_block
                    ) * DIAGONAL_BLOCK_SIZE
                    vstore_pack(conversion, target_offset, value, lane_mask, pack_mode=PackMode.B32_TO_B16)
        rows = valid_rows - self.subblock_idx * HALF_CHUNK_SIZE
        if rows > HALF_CHUNK_SIZE:
            rows = HALF_CHUNK_SIZE
        if rows > 0:
            mem_copy(
                tile_slice(inverse, (HALF_CHUNK_SIZE, CHUNK_SIZE), (self.subblock_idx, 0))[:rows, None],
                conversion[:rows, None], l2_cache_ctl=4,
            )



@jit
def run_persistent_flash_kda_fwd(
    gm_K: Tensor,
    gm_V: Tensor,
    gm_Q: Tensor,
    gm_beta: Tensor,
    gm_g: Tensor,
    gm_qk_exchange: Tensor,
    gm_A_log: Tensor,
    gm_dt_bias: Tensor,
    gm_kg: Tensor,
    gm_gamma: Tensor,
    gm_q_decayed: Tensor,
    gm_u: Tensor,
    gm_saved_w: Tensor,
    gm_qg: Tensor,
    gm_g_cumsum: Tensor,
    gm_v_new: Tensor,
    gm_h: Tensor,
    gm_O: Tensor,
    gm_state_in: Tensor,
    gm_state_out: Tensor,
    gm_identity16: Tensor,
    gm_q_norm: Tensor,
    gm_k_norm: Tensor,
    gm_q_rstd: Tensor,
    gm_k_rstd: Tensor,
    gm_beta_activated: Tensor,
    gm_Aqk: Tensor,
    gm_Akk: Tensor,
    gm_metadata: Tensor,
    scale_value: float,
    lower_bound: float,
    use_qk_l2norm_in_kernel: int,
    use_gate_in_kernel: int,
    use_beta_sigmoid_in_kernel: int,
    allow_neg_eigval: int,
    has_lower_bound: int,
    state_is_compact: int,
    save_qg: int,
    save_g_cumsum: int,
    save_intermediate_state: int,
):
    if use_gate_in_kernel * has_lower_bound != 0:
        channel_rewind(reset_sync_id=True)
        _run_persistent_training_tasks(
            gm_K, gm_V, gm_Q, gm_beta, gm_g, gm_qk_exchange, gm_A_log, gm_dt_bias, gm_kg, gm_gamma, gm_q_decayed, gm_u,
            gm_saved_w, gm_qg, gm_g_cumsum, gm_v_new, gm_h, gm_O, gm_state_in, gm_state_out, gm_identity16, gm_q_norm,
            gm_k_norm, gm_q_rstd, gm_k_rstd, gm_beta_activated, gm_Aqk, gm_Akk, gm_metadata, scale_value, lower_bound,
            use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval, has_lower_bound,
            state_is_compact, save_qg, save_g_cumsum, save_intermediate_state, False,
        )
    else:
        channel_rewind(reset_sync_id=True)
        _run_persistent_training_tasks(
            gm_K, gm_V, gm_Q, gm_beta, gm_g, gm_qk_exchange, gm_A_log, gm_dt_bias, gm_kg, gm_gamma, gm_q_decayed, gm_u,
            gm_saved_w, gm_qg, gm_g_cumsum, gm_v_new, gm_h, gm_O, gm_state_in, gm_state_out, gm_identity16, gm_q_norm,
            gm_k_norm, gm_q_rstd, gm_k_rstd, gm_beta_activated, gm_Aqk, gm_Akk, gm_metadata, scale_value, lower_bound,
            use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval, has_lower_bound,
            state_is_compact, save_qg, save_g_cumsum, save_intermediate_state, True,
        )


@jit
def _run_persistent_training_tasks(
    gm_K: Tensor,
    gm_V: Tensor,
    gm_Q: Tensor,
    gm_beta: Tensor,
    gm_g: Tensor,
    gm_qk_exchange: Tensor,
    gm_A_log: Tensor,
    gm_dt_bias: Tensor,
    gm_kg: Tensor,
    gm_gamma: Tensor,
    gm_q_decayed: Tensor,
    gm_u: Tensor,
    gm_saved_w: Tensor,
    gm_qg: Tensor,
    gm_g_cumsum: Tensor,
    gm_v_new: Tensor,
    gm_h: Tensor,
    gm_O: Tensor,
    gm_state_in: Tensor,
    gm_state_out: Tensor,
    gm_identity16: Tensor,
    gm_q_norm: Tensor,
    gm_k_norm: Tensor,
    gm_q_rstd: Tensor,
    gm_k_rstd: Tensor,
    gm_beta_activated: Tensor,
    gm_Aqk: Tensor,
    gm_Akk: Tensor,
    gm_metadata: Tensor,
    scale_value: float,
    lower_bound: float,
    use_qk_l2norm_in_kernel: int,
    use_gate_in_kernel: int,
    use_beta_sigmoid_in_kernel: int,
    allow_neg_eigval: int,
    has_lower_bound: int,
    state_is_compact: int,
    save_qg: int,
    save_g_cumsum: int,
    save_intermediate_state: int,
    stable_scores,
):
    block_idx = get_block_idx()
    subblock_idx = get_subblock_id()
    head_group_size = gm_V.shape[1] // gm_Q.shape[1]
    preprocessing = L2NormAndActivate(subblock_idx)
    preprocessing.scale = scale_value
    preprocessing.lower_bound = lower_bound
    preprocessing.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
    preprocessing.has_dt_bias = use_gate_in_kernel == GATE_MODE_FUSED_WITH_DT_BIAS
    if const_expr(not stable_scores):
        preprocessing.use_gate_in_kernel = 1
        preprocessing.has_lower_bound = 1
    else:
        preprocessing.use_gate_in_kernel = use_gate_in_kernel
        preprocessing.has_lower_bound = has_lower_bound
    preprocessing.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
    preprocessing.allow_neg_eigval = allow_neg_eigval
    preprocessing.save_qg = save_qg
    preprocessing.use_stable_scores = stable_scores
    recurrent_state = RecurrentStateUpdate(subblock_idx)
    matmul_stage = ChunkMatmul()
    scores = GateWeightedScoresAndDiagonalInverse(subblock_idx, preprocessing)
    caches = TrainingCacheWriter(subblock_idx)
    scores.causal_qkt_nd = caches.ub_cast_bf16
    caches.save_backward_inputs = save_qg
    caches.save_intermediate_state = save_intermediate_state
    matmul_stage.save_backward_inputs = save_qg
    preprocessing.cache_writer = caches
    preprocessing.saved_qg_nd = caches.ub_cast_bf16
    preprocessing.saved_kg_nd = preprocessing.ub_v_raw
    identity = matmul_stage.prepare_packed64_identity_l1(gm_identity16)
    matmul_stage.zero_fp16_l1(matmul_stage.zero_l1)
    task = gm_metadata[HEADER_WORDS + block_idx]
    while task != 0:
        batch_idx = gm_metadata[task + CHUNK_TASK_BATCH]
        storage_batch_idx = gm_metadata[task + CHUNK_TASK_STORAGE_BATCH]
        token_start = gm_metadata[task + CHUNK_TASK_TOKEN_START]
        sequence_length = dtypes.int64(gm_metadata[task + CHUNK_TASK_LENGTH])
        head_idx = gm_metadata[task + CHUNK_TASK_VALUE_HEAD]
        qk_head_idx = head_idx // head_group_size
        token_stop = token_start + sequence_length
        q = gm_Q[storage_batch_idx, qk_head_idx, token_start:token_stop, None]
        k = gm_K[storage_batch_idx, qk_head_idx, token_start:token_stop, None]
        v = gm_V[storage_batch_idx, head_idx, token_start:token_stop, None]
        gate = gm_g[storage_batch_idx, head_idx, token_start:token_stop, None]
        beta = gm_beta[storage_batch_idx, head_idx, token_start:token_stop, None]
        out = gm_O[storage_batch_idx, head_idx, token_start:token_stop, None]
        caches.sequence_length = sequence_length
        caches.write_qk = (
            head_idx % head_group_size == 0 and use_qk_l2norm_in_kernel != 0
        )
        caches.q_norm = gm_q_norm[
            storage_batch_idx, qk_head_idx, token_start:token_stop, None
        ]
        caches.k_norm = gm_k_norm[
            storage_batch_idx, qk_head_idx, token_start:token_stop, None
        ]
        caches.q_rstd = gm_q_rstd[
            storage_batch_idx, qk_head_idx, token_start:token_stop, None
        ]
        caches.k_rstd = gm_k_rstd[
            storage_batch_idx, qk_head_idx, token_start:token_stop, None
        ]
        caches.beta_activated = gm_beta_activated[
            storage_batch_idx, head_idx, token_start:token_stop, None
        ]
        caches.qg = gm_qg[storage_batch_idx, head_idx, token_start:token_stop, None]
        caches.kg = gm_kg[storage_batch_idx, head_idx, token_start:token_stop, None]
        caches.g_cumsum = gm_g_cumsum[
            storage_batch_idx, head_idx, token_start:token_stop, None
        ]
        caches.v_new = gm_v_new[
            storage_batch_idx, head_idx, token_start:token_stop, None
        ]
        chunk_storage_base = 0
        if state_is_compact != 0:
            chunk_storage_base = token_start // CHUNK_SIZE + batch_idx
        caches.Aqk = gm_Aqk[
            storage_batch_idx, head_idx, chunk_storage_base:, None, None
        ]
        caches.Akk = gm_Akk[
            storage_batch_idx, head_idx, chunk_storage_base:, None, None
        ]
        state_batch_idx = batch_idx
        state_chunk_base = chunk_storage_base
        if state_is_compact != 0:
            state_batch_idx = 0
        if save_intermediate_state == 0:
            state_batch_idx = 0
            state_chunk_base = 0
        caches.h = gm_h[state_batch_idx, head_idx, state_chunk_base:, None, None]
        matmul_stage.saved_w = gm_saved_w[
            storage_batch_idx, head_idx, token_start:token_stop, None
        ]
        matmul_stage.saved_u = gm_u[
            storage_batch_idx, head_idx, token_start:token_stop, None
        ]
        matmul_stage.saved_v_new = caches.v_new
        initial_state = gm_state_in[batch_idx, head_idx, None, None]
        final_state = gm_state_out[batch_idx, head_idx, None, None]
        if sequence_length % CHUNK_SIZE == 0:
            run_persistent_training_sequence(
                preprocessing, recurrent_state, matmul_stage, scores, q, k, v, gate, beta, initial_state, final_state,
                out, gm_A_log, gm_dt_bias, identity, gm_qk_exchange, block_idx, head_idx, sequence_length, scale_value,
                lower_bound, False, caches,
            )
        else:
            run_persistent_training_sequence(
                preprocessing, recurrent_state, matmul_stage, scores, q, k, v, gate, beta, initial_state, final_state,
                out, gm_A_log, gm_dt_bias, identity, gm_qk_exchange, block_idx, head_idx, sequence_length, scale_value,
                lower_bound, True, caches,
            )
        task = gm_metadata[task + CHUNK_TASK_NEXT]
    cube_sync_all()
    vec_sync_all()


def persistent_flash_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = SUPPORTED_HEAD_DIM ** (-0.5),
    initial_state: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
    layout_qkv: str = "TND",
    metadata: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    disable_recompute: bool = True,
    return_intermediate_state: bool = False,
    allow_neg_eigval: bool = False,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, ...]:
    from .flash_kda_fwd import _flash_kda_fwd_impl
    return _flash_kda_fwd_impl(
        q, k, v, g, beta, scale, initial_state, A_log, dt_bias, lower_bound, layout_qkv, metadata,
        use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, disable_recompute,
        return_intermediate_state, allow_neg_eigval, output_final_state, internal_template=2,
    )


@export("persistent_flash_kda_fwd")
def export_persistent_flash_kda_fwd():
    """Collect the direct template compile request without launching a Provider."""
    from .flash_kda_fwd import _get_compiled_kernel
    from .flash_kda_fwd import clear_caches
    try:
        _get_compiled_kernel(2)
    finally:
        clear_caches()


_GRAPH_LIBRARY = torch.library.Library("cannbotdsl_flash_kda_fwd", "FRAGMENT")
_GRAPH_LIBRARY.define(
    "persistent_flash_kda_fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float? "
    "scale, Tensor initial_state, Tensor? a_log, Tensor? dt_bias, float? lower_bound, str"
    " layout_qkv, Tensor metadata, bool use_qk_l2norm_in_kernel=True, bool "
    "use_gate_in_kernel=True, bool use_beta_sigmoid_in_kernel=True, bool "
    "disable_recompute=True, bool return_intermediate_state=False, bool "
    "allow_neg_eigval=False, bool output_final_state=False) -> (Tensor, Tensor?, Tensor?,"
    " Tensor?, Tensor?, Tensor?, Tensor?, Tensor, Tensor, Tensor?, Tensor?, Tensor?, "
    "Tensor?, Tensor?, Tensor?, Tensor?)",
)


@torch.library.impl(_GRAPH_LIBRARY, "persistent_flash_kda_fwd", "Meta")
def _persistent_flash_kda_fwd_meta(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    a_log,
    dt_bias,
    lower_bound,
    layout_qkv,
    metadata,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    disable_recompute=True,
    return_intermediate_state=False,
    allow_neg_eigval=False,
    output_final_state=False,
):
    del g, scale, dt_bias, lower_bound, metadata
    if initial_state.dtype != torch.float32:
        raise AssertionError("initial_state must be fp32")
    if use_gate_in_kernel:
        if a_log is None:
            raise AssertionError("A_log is required when use_gate_in_kernel=True")
    if allow_neg_eigval:
        if not use_beta_sigmoid_in_kernel:
            raise AssertionError("allow_neg_eigval=True requires use_beta_sigmoid_in_kernel=True")
    storage_length = (
        q.shape[0] if layout_qkv == "TND" else q.shape[2 if layout_qkv == "BNSD" else 1]
    )
    chunk_count = (
        _tnd_state_chunk_capacity(storage_length, initial_state.shape[0])
        if layout_qkv == "TND"
        else (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    )
    if layout_qkv == "TND":
        q_bnsd_shape = (1, q.shape[1], storage_length, q.shape[2])
        v_bnsd_shape = (1, v.shape[1], storage_length, v.shape[2])
        q_public_shape = q.shape
        v_public_shape = v.shape
        state_public_shape = (
            _tnd_state_chunk_capacity(storage_length, initial_state.shape[0]),
            initial_state.shape[1],
            SUPPORTED_HEAD_DIM,
            SUPPORTED_HEAD_DIM,
        )
        chunk_matrix_shape = (chunk_count, v.shape[1], CHUNK_SIZE, CHUNK_SIZE)
    elif layout_qkv == "BNSD":
        q_bnsd_shape = q.shape
        v_bnsd_shape = v.shape
        q_public_shape = q.shape
        v_public_shape = v.shape
        state_public_shape = (
            initial_state.shape[0],
            initial_state.shape[1],
            chunk_count,
            SUPPORTED_HEAD_DIM,
            SUPPORTED_HEAD_DIM,
        )
        chunk_matrix_shape = (
            q.shape[0],
            v.shape[1],
            chunk_count,
            CHUNK_SIZE,
            CHUNK_SIZE,
        )
    else:
        q_bnsd_shape = (q.shape[0], q.shape[2], q.shape[1], q.shape[3])
        v_bnsd_shape = (v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        q_public_shape = q_bnsd_shape
        v_public_shape = v_bnsd_shape
        state_public_shape = (
            initial_state.shape[0],
            initial_state.shape[1],
            chunk_count,
            SUPPORTED_HEAD_DIM,
            SUPPORTED_HEAD_DIM,
        )
        chunk_matrix_shape = (
            q.shape[0],
            v.shape[2],
            chunk_count,
            CHUNK_SIZE,
            CHUNK_SIZE,
        )
    return (
        torch.empty_like(v, device="meta"),
        torch.empty_like(initial_state, device="meta") if output_final_state else None,
        (
            torch.empty(q_public_shape, dtype=torch.bfloat16, device="meta")
            if use_qk_l2norm_in_kernel and disable_recompute
            else None
        ),
        (
            torch.empty(q_public_shape, dtype=torch.bfloat16, device="meta")
            if use_qk_l2norm_in_kernel and disable_recompute
            else None
        ),
        (
            torch.empty(q_public_shape[:-1], dtype=torch.float32, device="meta")
            if use_qk_l2norm_in_kernel and disable_recompute
            else None
        ),
        (
            torch.empty(q_public_shape[:-1], dtype=torch.float32, device="meta")
            if use_qk_l2norm_in_kernel and disable_recompute
            else None
        ),
        (
            torch.empty(v_public_shape[:-1], dtype=torch.float32, device="meta")
            if use_beta_sigmoid_in_kernel and disable_recompute
            else None
        ),
        torch.empty(chunk_matrix_shape, dtype=torch.bfloat16, device="meta"),
        torch.empty(chunk_matrix_shape, dtype=torch.bfloat16, device="meta"),
        (
            torch.empty(v_public_shape, dtype=torch.bfloat16, device="meta")
            if disable_recompute
            else None
        ),
        (
            torch.empty(v_public_shape, dtype=torch.bfloat16, device="meta")
            if disable_recompute
            else None
        ),
        (
            torch.empty(v_public_shape, dtype=torch.bfloat16, device="meta")
            if disable_recompute
            else None
        ),
        (
            torch.empty(v_public_shape, dtype=torch.bfloat16, device="meta")
            if disable_recompute
            else None
        ),
        (
            torch.empty(v_public_shape, dtype=torch.bfloat16, device="meta")
            if disable_recompute
            else None
        ),
        (
            torch.empty(v_public_shape, dtype=torch.float32, device="meta")
            if disable_recompute
            else None
        ),
        (
            torch.empty(state_public_shape, dtype=torch.float32, device="meta")
            if return_intermediate_state
            else None
        ),
    )


@torch.library.impl(_GRAPH_LIBRARY, "persistent_flash_kda_fwd", "PrivateUse1")
def _persistent_flash_kda_fwd_privateuse1(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    a_log,
    dt_bias,
    lower_bound,
    layout_qkv,
    metadata,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    disable_recompute=True,
    return_intermediate_state=False,
    allow_neg_eigval=False,
    output_final_state=False,
):
    return persistent_flash_kda_fwd(
        q, k, v, g, beta, scale, initial_state, a_log, dt_bias, lower_bound, layout_qkv, metadata,
        use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, disable_recompute,
        return_intermediate_state, allow_neg_eigval, output_final_state,
    )


persistent_flash_kda_fwd_op = torch.ops.cannbotdsl_flash_kda_fwd.persistent_flash_kda_fwd


@torch._dynamo.substitute_in_graph(persistent_flash_kda_fwd, can_constant_fold_through=False)
def _persistent_flash_kda_fwd_graph(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = SUPPORTED_HEAD_DIM ** (-0.5),
    initial_state: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
    layout_qkv: str = "TND",
    metadata: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    disable_recompute: bool = True,
    return_intermediate_state: bool = False,
    allow_neg_eigval: bool = False,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, ...]:
    return persistent_flash_kda_fwd_op(
        q, k, v, g, beta, scale, initial_state, A_log, dt_bias, lower_bound, layout_qkv, metadata,
        use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, disable_recompute,
        return_intermediate_state, allow_neg_eigval, output_final_state,
    )


def _tnd_state_chunk_capacity(storage_length: int, batch: int) -> int:
    return (int(storage_length) + CHUNK_SIZE - 1) // CHUNK_SIZE + int(batch)
