# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Packed-QKV decode with a read-only checkpoint and Mega-compatible replay records."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import cannbotdsl
import torch
from cannbotdsl import MemLoc, Tensor, TensorSpec, dtypes
from cannbotdsl.aot import export
from cannbotdsl.buffer import Buffer
from cannbotdsl.channel import Channel
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.sync import channel_rewind
from cannbotdsl.ops.reg import (
    PackMode, UnpackMode, update_mask, vadd, vadds, vcast, vdiv, vdup,
    vdups, vexp, vload, vload_broadcast, vload_unpack, vmul, vmuls, vneg,
    vreduce_sum, vsqrt, vstore, vstore_first, vstore_pack, vsub,
)
from cannbotdsl.tensor import idx2crd, reinterpret, tile_slice

TILE = SUPPORTED_HEAD_DIM = 128
VL = 64
MAX_VALUE_HEADS = 96
SUPPORTED_BLOCK_NUMS = (8, 24, 28, 32)
SMALL_BATCH_HEAD_LIMIT = 16
MEDIUM_BATCH_HEAD_LIMIT = 32
SMALL_BATCH_ROW_BLOCK = SMALL_ROW_BLOCK = 32
MEDIUM_ROW_BLOCK = 64
KDA_STATE_KEY_UNROLL = 4
KDA_STATE_UPDATE_UNROLL = 1
_COMPILED_KERNEL = None
_COMPILED_KERNEL_LOCK = threading.Lock()


def _device_block_num(ref: torch.Tensor) -> int:
    """Return the current stream's effective AIC count, capped at 32."""
    if ref.device.type not in {"npu", "privateuseone"}:
        raise ValueError("fused_recurrent_kda_replayssm requires NPU inputs")
    device_index = ref.device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    stream = torch.npu.current_stream(device_index)
    properties = cannbotdsl.get_platform_info(stream=stream)
    try:
        cube_core_num = int(properties.cube_core_num)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(f"NPU {device_index} does not expose a valid AIC count") from error
    if cube_core_num <= 0:
        raise RuntimeError(f"NPU {device_index} reports invalid cube_core_num={cube_core_num}")
    return min(cube_core_num, 32)


def _resolve_block_num(ref: torch.Tensor | None, block_num: int | None) -> int:
    """Resolve the public AIC-equivalent quota."""
    if block_num is None:
        if ref is None:
            raise ValueError("block_num=None requires a reference NPU tensor")
        available = _device_block_num(ref)
        choices = tuple(value for value in SUPPORTED_BLOCK_NUMS if value <= available)
        if not choices:
            raise ValueError(f"device exposes {available} AIC cores; at least 8 are required")
        return choices[-1]
    if isinstance(block_num, bool) or not isinstance(block_num, int):
        raise ValueError(f"block_num must be one of {SUPPORTED_BLOCK_NUMS}; got {block_num!r}")
    if block_num not in SUPPORTED_BLOCK_NUMS:
        raise ValueError(f"block_num must be one of {SUPPORTED_BLOCK_NUMS}; got {block_num}")
    if ref is not None and ref.device.type in {"npu", "privateuseone"}:
        available = _device_block_num(ref)
        if block_num > available:
            raise ValueError(f"block_num={block_num} exceeds the device limit {available}")
    return block_num


def _launch_blocks(batch_size: int, num_heads: int, aic_blocks: int) -> int:
    """Use Snapshot row blocks to expose enough work to both AIV subblocks."""
    batch_heads = batch_size * num_heads
    if batch_heads <= SMALL_BATCH_HEAD_LIMIT:
        row_block = SMALL_BATCH_ROW_BLOCK
    elif batch_heads <= MEDIUM_BATCH_HEAD_LIMIT:
        row_block = VL
    else:
        row_block = TILE
    total_items = batch_heads * (TILE // row_block)
    return min(total_items, 2 * aic_blocks)


class FusedRecurrentKDAReplaySSMVector:
    """Replay-record delta-rule recurrence for one complete value head."""

    def __init__(
        self,
        row_block: int,
        num_heads: int,
        state_key_unroll: int = KDA_STATE_KEY_UNROLL,
        state_update_unroll: int = KDA_STATE_UPDATE_UNROLL,
    ):
        self.row_block = row_block
        value_width = max(row_block, VL)
        self.num_heads = int(num_heads)
        self.state_key_unroll = int(state_key_unroll)
        self.state_update_unroll = int(state_update_unroll)
        self.state = Channel(MemLoc.UB, (self.row_block, TILE), dtypes.float32, depth=1)
        self.state_tensor = None
        self.key = Buffer(MemLoc.UB, (1, TILE), dtypes.float32)
        self.query = Buffer(MemLoc.UB, (1, TILE), dtypes.float32)
        self.decay_exp = Buffer(MemLoc.UB, (1, TILE), dtypes.float32)
        self.ub_value = Buffer(MemLoc.UB, (1, value_width), dtypes.float32)
        self.ub_beta = Buffer(MemLoc.UB, (1, VL), dtypes.float32)
        self.out = Buffer(MemLoc.UB, (1, value_width), dtypes.float32)
        self.output = Channel(MemLoc.UB, (1, value_width), dtypes.bfloat16, depth=2)
        self.value = Channel(MemLoc.UB, (1, value_width), dtypes.bfloat16, depth=2)
        self.state_key_sums = Buffer(MemLoc.UB, (1, value_width), dtypes.float32)
        self.delta_row = Buffer(MemLoc.UB, (1, value_width), dtypes.float32)
        # vLLM ReplaySSM PR #47576 stores GDN d/k records as FP16 when the
        # model activations are BF16.  Arithmetic remains FP32 in this stage.
        self.replay_u = Channel(MemLoc.UB, (1, value_width), dtypes.float16, depth=1)
        self.replay_k = Channel(MemLoc.UB, (1, TILE), dtypes.float16, depth=1)
        self.replay_decay = Channel(MemLoc.UB, (1, TILE), dtypes.float32, depth=1)
        self.raw_query = Channel(MemLoc.UB, (1, TILE), dtypes.bfloat16, depth=2)
        self.raw_key = Channel(MemLoc.UB, (1, TILE), dtypes.bfloat16, depth=2)
        self.raw_g = Channel(MemLoc.UB, (1, TILE), dtypes.bfloat16, depth=2)
        self.raw_beta = Channel(MemLoc.UB, (1, VL), dtypes.bfloat16, depth=2)
        self.dt_bias = Buffer(MemLoc.UB, (self.num_heads, TILE), dtypes.float32)
        self.a_log = Buffer(MemLoc.UB, (self.num_heads,), dtypes.float32)

    def load_gate_params(self, a_log_gm, dt_bias_gm, value_heads):
        mem_copy(self.a_log, a_log_gm)
        mem_copy(self.dt_bias, dt_bias_gm)

    def load_state(self, state_gm, active_rows: int):
        self.state_tensor = reinterpret(self.state.produce(), shape=(active_rows, TILE), offset=0)
        mem_copy(self.state_tensor, state_gm, l2_cache_ctl=2)

    def prefetch_inputs(self, raw_key_gm, raw_query_gm, raw_g_gm, value_gm, raw_beta_gm, active_rows: int):
        mem_copy(self.raw_query.produce(), raw_query_gm)
        mem_copy(self.raw_key.produce(), raw_key_gm)
        mem_copy(self.raw_g.produce(), raw_g_gm)
        mem_copy(tile_slice(self.raw_beta.produce(), (1, 1), (0, 0)), raw_beta_gm)
        mem_copy(reinterpret(self.value.produce(), shape=(1, active_rows), offset=0), value_gm)

    @jit
    def normalize_qk(self, scale_value: float):
        raw_query = self.raw_query.consume()
        raw_key = self.raw_key.consume()
        replay_k = self.replay_k.produce()
        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            query_pre_bf16 = vload_unpack(raw_query, 0, unpack_mode=UnpackMode.B16_TO_B32)
            query_pre = vcast(query_pre_bf16, dtypes.float32, mask=full)
            query_post_bf16 = vload_unpack(raw_query, VL, unpack_mode=UnpackMode.B16_TO_B32)
            query_post = vcast(query_post_bf16, dtypes.float32, mask=full)
            key_pre_bf16 = vload_unpack(raw_key, 0, unpack_mode=UnpackMode.B16_TO_B32)
            key_pre = vcast(key_pre_bf16, dtypes.float32, mask=full)
            key_post_bf16 = vload_unpack(raw_key, VL, unpack_mode=UnpackMode.B16_TO_B32)
            key_post = vcast(key_post_bf16, dtypes.float32, mask=full)
            query_pre_square = vmul(query_pre, query_pre, mask=full)
            query_pre_sum = vreduce_sum(query_pre_square, mask=full)
            query_post_square = vmul(query_post, query_post, mask=full)
            query_post_sum = vreduce_sum(query_post_square, mask=full)
            key_pre_square = vmul(key_pre, key_pre, mask=full)
            key_pre_sum = vreduce_sum(key_pre_square, mask=full)
            key_post_square = vmul(key_post, key_post, mask=full)
            key_post_sum = vreduce_sum(key_post_square, mask=full)
            lane0, _ = update_mask(1, elem_bits=32)
            query_sum = vadd(query_pre_sum, query_post_sum, mask=lane0)
            key_sum = vadd(key_pre_sum, key_post_sum, mask=lane0)
            query_sum_eps = vadds(query_sum, 1e-6, mask=lane0)
            query_norm = vsqrt(query_sum_eps, mask=lane0)
            key_sum_eps = vadds(key_sum, 1e-6, mask=lane0)
            key_norm = vsqrt(key_sum_eps, mask=lane0)
            full, _ = update_mask(VL, elem_bits=32)
            query_denom = vdup(query_norm, mask=full)
            key_denom = vdup(key_norm, mask=full)
            query_pre = vdiv(query_pre, query_denom, mask=full)
            query_pre = vmuls(query_pre, scale_value, mask=full)
            vstore(self.query, 0, query_pre, full)
            query_post = vdiv(query_post, query_denom, mask=full)
            query_post = vmuls(query_post, scale_value, mask=full)
            vstore(self.query, VL, query_post, full)
            key_pre = vdiv(key_pre, key_denom, mask=full)
            vstore(self.key, 0, key_pre, full)
            key_pre_fp16 = vcast(key_pre, dtypes.float16, mask=full)
            vstore_pack(
                replay_k,
                0,
                key_pre_fp16,
                full,
                pack_mode=PackMode.B32_TO_B16,
            )
            key_post = vdiv(key_post, key_denom, mask=full)
            vstore(self.key, VL, key_post, full)
            key_post_fp16 = vcast(key_post, dtypes.float16, mask=full)
            vstore_pack(
                replay_k,
                VL,
                key_post_fp16,
                full,
                pack_mode=PackMode.B32_TO_B16,
            )

    @jit
    def activate_g_beta(self, lower_bound: float, value_head, active_rows: int):
        raw_g = self.raw_g.consume()
        raw_beta = self.raw_beta.consume()
        value_slot = self.value.consume()
        replay_decay = self.replay_decay.produce()
        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            one = vdups(1.0, dtypes.float32)
            alpha = vload_broadcast(self.a_log, value_head)
            alpha = vexp(alpha, mask=full)
            negative_alpha = vneg(alpha, mask=full)
            gate_offset = value_head * TILE
            raw_gate_pre_bf16 = vload_unpack(raw_g, 0, unpack_mode=UnpackMode.B16_TO_B32)
            raw_gate_pre = vcast(raw_gate_pre_bf16, dtypes.float32, mask=full)
            raw_gate_post_bf16 = vload_unpack(raw_g, VL, unpack_mode=UnpackMode.B16_TO_B32)
            raw_gate_post = vcast(raw_gate_post_bf16, dtypes.float32, mask=full)
            gate_bias_pre = vload(self.dt_bias, gate_offset)
            gate_pre = vadd(raw_gate_pre, gate_bias_pre, mask=full)
            gate_pre = vmul(negative_alpha, gate_pre, mask=full)
            gate_bias_post = vload(self.dt_bias, gate_offset + VL)
            gate_post = vadd(raw_gate_post, gate_bias_post, mask=full)
            gate_post = vmul(negative_alpha, gate_post, mask=full)
            activated_pre = vexp(gate_pre, mask=full)
            activated_pre = vadds(activated_pre, 1.0, mask=full)
            activated_pre = vdiv(one, activated_pre, mask=full)
            activated_pre = vmuls(activated_pre, lower_bound, mask=full)
            activated_post = vexp(gate_post, mask=full)
            activated_post = vadds(activated_post, 1.0, mask=full)
            activated_post = vdiv(one, activated_post, mask=full)
            activated_post = vmuls(activated_post, lower_bound, mask=full)
            decay_exp_pre = vexp(activated_pre, mask=full)
            vstore(self.decay_exp, 0, decay_exp_pre, full)
            vstore(replay_decay, 0, decay_exp_pre, full)
            decay_exp_post = vexp(activated_post, mask=full)
            vstore(self.decay_exp, VL, decay_exp_post, full)
            vstore(replay_decay, VL, decay_exp_post, full)

            beta_logit_bf16 = vload_unpack(raw_beta, 0, unpack_mode=UnpackMode.B16_TO_B32)
            beta_logit = vcast(beta_logit_bf16, dtypes.float32, mask=full)
            beta = vneg(beta_logit, mask=full)
            beta = vexp(beta, mask=full)
            beta = vadds(beta, 1.0, mask=full)
            beta = vdiv(one, beta, mask=full)
            vstore(self.ub_beta, 0, beta, full)
            value_mask, _ = update_mask(min(active_rows, VL), elem_bits=32)
            value_bf16 = vload_unpack(value_slot, 0, unpack_mode=UnpackMode.B16_TO_B32)
            value = vcast(value_bf16, dtypes.float32, mask=value_mask)
            vstore(self.ub_value, 0, value, value_mask)
            if active_rows > VL:
                value_post_bf16 = vload_unpack(value_slot, VL, unpack_mode=UnpackMode.B16_TO_B32)
                value_post = vcast(value_post_bf16, dtypes.float32, mask=full)
                vstore(self.ub_value, VL, value_post, full)

    @jit
    def compute_delta(self, active_rows: int):
        replay_u = self.replay_u.produce()
        with vf(mode="simd"):
            mask, _ = update_mask(VL, elem_bits=32)
            decay_pre = vload(self.decay_exp, 0)
            decay_post = vload(self.decay_exp, VL)
            key_pre = vload(self.key, 0)
            key_post = vload(self.key, VL)
            decay_key_pre = vmul(decay_pre, key_pre, mask=mask)
            decay_key_post = vmul(decay_post, key_post, mask=mask)
            for dv in dsl_range(0, active_rows, 1, unroll=self.state_key_unroll):
                offset = dv * TILE
                state_key_pre = vload(self.state_tensor, offset)
                state_key_post = vload(self.state_tensor, offset + VL)
                state_key = vmul(state_key_pre, decay_key_pre, mask=mask)
                state_key_post = vmul(state_key_post, decay_key_post, mask=mask)
                state_key = vadd(state_key, state_key_post, mask=mask)
                state_key_sum = vreduce_sum(state_key, mask=mask)
                vstore_first(self.state_key_sums, dv, state_key_sum)
        with vf(mode="simd"):
            mask, _ = update_mask(min(active_rows, VL), elem_bits=32)
            beta = vload_broadcast(self.ub_beta, 0)
            value_pre = vload(self.ub_value, 0)
            state_key_sum_pre = vload(self.state_key_sums, 0)
            residual_pre = vsub(value_pre, state_key_sum_pre, mask=mask)
            delta_pre = vmul(residual_pre, beta, mask=mask)
            vstore(self.delta_row, 0, delta_pre, mask)
            delta_pre_fp16 = vcast(delta_pre, dtypes.float16, mask=mask)
            vstore_pack(
                replay_u,
                0,
                delta_pre_fp16,
                mask,
                pack_mode=PackMode.B32_TO_B16,
            )
            if active_rows > VL:
                value_post = vload(self.ub_value, VL)
                state_key_sum_post = vload(self.state_key_sums, VL)
                residual_post = vsub(value_post, state_key_sum_post, mask=mask)
                delta_post = vmul(residual_post, beta, mask=mask)
                vstore(self.delta_row, VL, delta_post, mask)
                delta_post_fp16 = vcast(delta_post, dtypes.float16, mask=mask)
                vstore_pack(
                    replay_u,
                    VL,
                    delta_post_fp16,
                    mask,
                    pack_mode=PackMode.B32_TO_B16,
                )

    @jit
    def update_state(self, active_rows: int):
        with vf(mode="simd"):
            mask, _ = update_mask(VL, elem_bits=32)
            decay_pre = vload(self.decay_exp, 0)
            decay_post = vload(self.decay_exp, VL)
            key_pre = vload(self.key, 0)
            key_post = vload(self.key, VL)
            query_pre = vload(self.query, 0)
            query_post = vload(self.query, VL)
            for dv in dsl_range(0, active_rows, 1, unroll=self.state_update_unroll):
                offset = dv * TILE
                delta = vload_broadcast(self.delta_row, dv)
                state_pre = vload(self.state_tensor, offset)
                update_pre = vmul(delta, key_pre, mask=mask)
                state_pre = vmul(state_pre, decay_pre, mask=mask)
                state_pre = vadd(state_pre, update_pre, mask=mask)
                state_post = vload(self.state_tensor, offset + VL)
                update_post = vmul(delta, key_post, mask=mask)
                state_post = vmul(state_post, decay_post, mask=mask)
                state_post = vadd(state_post, update_post, mask=mask)
                vstore(self.state_tensor, offset, state_pre, mask)
                vstore(self.state_tensor, offset + VL, state_post, mask)
                output_pre = vmul(state_pre, query_pre, mask=mask)
                output_post = vmul(state_post, query_post, mask=mask)
                output = vadd(output_pre, output_post, mask=mask)
                output = vreduce_sum(output, mask=mask)
                vstore_first(self.out, dv, output)

    @jit
    def publish_replay_k(self, replay_k_gm):
        """Publish the FP16 key produced by normalization."""
        mem_copy(replay_k_gm, self.replay_k.consume())

    @jit
    def publish_replay_decay(self, replay_decay_gm):
        """Publish the FP32 decay produced by gate activation."""
        mem_copy(replay_decay_gm, self.replay_decay.consume())

    @jit
    def publish_replay_u(self, replay_u_gm, active_rows: int):
        """Publish each lane's disjoint FP16 U rows."""
        mem_copy(replay_u_gm, reinterpret(self.replay_u.consume(), shape=(1, active_rows), offset=0))

    @jit
    def store_output(self, out_gm, active_rows: int):
        output = self.output.produce()
        with vf(mode="simd"):
            for segment in range((active_rows + VL - 1) // VL):
                offset = segment * VL
                mask, _ = update_mask(active_rows - offset, elem_bits=32)
                value = vcast(vload(self.out, offset), dtypes.bfloat16, mask=mask)
                vstore_pack(output, offset, value, mask, pack_mode=PackMode.B32_TO_B16)
        mem_copy(out_gm, reinterpret(output, shape=(1, active_rows), offset=0))


@jit
def _prefetch_token(vector, gm_mixedqkv, gm_g, gm_beta, batch, head, heads,
                    token, row_index, active_rows: int):
    vector.prefetch_inputs(
        tile_slice(gm_mixedqkv[batch, heads + head, None, None], (1, TILE), (token, 0)),
        tile_slice(gm_mixedqkv[batch, head, None, None], (1, TILE), (token, 0)),
        tile_slice(gm_g[batch, head, None, None], (1, TILE), (token, 0)),
        tile_slice(gm_mixedqkv[batch, 2 * heads + head, None, None], (1, active_rows), (token, row_index)),
        tile_slice(gm_beta[batch, head, None, None], (1, 1), (token, 0)), active_rows,
    )


@jit
def _run_recurrent_item(vector, gm_mixedqkv, gm_g, gm_beta, gm_out,
                        gm_state, gm_replay_u, gm_replay_k, gm_replay_decay,
                        gm_cu_seqlens, scale, lower_bound, layout_mode,
                        batch_size, sequence_length, value_heads_dim, item,
                        *, active_rows: int, row_index):
    batch_index, value_head = idx2crd(item, [batch_size, value_heads_dim])
    storage_batch = batch_index
    token_start = 0
    token_end = sequence_length
    if layout_mode == 1:
        storage_batch = 0
        token_start = gm_cu_seqlens[batch_index]
        token_end = gm_cu_seqlens[batch_index + 1]
    vector.load_state(
        tile_slice(gm_state[batch_index, value_head, None, None],
                   (active_rows, TILE), (row_index, 0)), active_rows)
    _prefetch_token(vector, gm_mixedqkv, gm_g, gm_beta, storage_batch,
                    value_head, value_heads_dim, token_start, row_index, active_rows)
    for token in range(token_start, token_end):
        replay_token = token - token_start
        vector.normalize_qk(scale)
        # Split state rows share K/decay; exactly one owner writes each record.
        if row_index == 0:
            vector.publish_replay_k(tile_slice(
                gm_replay_k[batch_index, replay_token, None, None],
                (1, TILE), (value_head, 0)))
        vector.activate_g_beta(lower_bound, value_head, active_rows)
        if row_index == 0:
            vector.publish_replay_decay(tile_slice(
                gm_replay_decay[batch_index, replay_token, None, None],
                (1, TILE), (value_head, 0)))
        if token + 1 < token_end:
            _prefetch_token(vector, gm_mixedqkv, gm_g, gm_beta, storage_batch,
                            value_head, value_heads_dim, token + 1, row_index, active_rows)
        vector.compute_delta(active_rows)
        vector.publish_replay_u(tile_slice(
            gm_replay_u[batch_index, replay_token, None, None],
            (1, active_rows), (value_head, row_index)), active_rows)
        vector.update_state(active_rows)
        vector.store_output(tile_slice(
            gm_out[storage_batch, value_head, None, None],
            (1, active_rows), (token, row_index)), active_rows)


@jit
def _run_minimal_split_recurrent_body(
        gm_mixedqkv, gm_g, gm_beta, gm_a_log, gm_dt_bias, gm_out,
        gm_state, gm_replay_u, gm_replay_k,
        gm_replay_decay, gm_cu_seqlens, scale, lower_bound, layout_mode,
        batch_size, sequence_length):
    value_heads_dim = gm_g.shape[1]
    vector = FusedRecurrentKDAReplaySSMVector(SUPPORTED_HEAD_DIM, MAX_VALUE_HEADS)
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
            vector, gm_mixedqkv, gm_g, gm_beta, gm_out, gm_state,
            gm_replay_u, gm_replay_k, gm_replay_decay,
            gm_cu_seqlens, scale, lower_bound, layout_mode, batch_size,
            sequence_length, value_heads_dim,
            full_head_start + full_offset,
            active_rows=SUPPORTED_HEAD_DIM, row_index=0)

    full_heads = core_num * base_full_heads + extra_cores * extra_full_delta
    if base_half_rows % 2 == 0 and core < extra_cores:
        _run_recurrent_item(
            vector, gm_mixedqkv, gm_g, gm_beta, gm_out, gm_state,
            gm_replay_u, gm_replay_k, gm_replay_decay,
            gm_cu_seqlens, scale, lower_bound, layout_mode, batch_size,
            sequence_length, value_heads_dim,
            full_heads + core // 2,
            active_rows=VL, row_index=core % 2)
    if base_half_rows % 2 == 1 and core >= extra_cores:
        half_rank = core - extra_cores
        _run_recurrent_item(
            vector, gm_mixedqkv, gm_g, gm_beta, gm_out, gm_state,
            gm_replay_u, gm_replay_k, gm_replay_decay,
            gm_cu_seqlens, scale, lower_bound, layout_mode, batch_size,
            sequence_length, value_heads_dim,
            full_heads + half_rank // 2,
            active_rows=VL, row_index=half_rank % 2)


@jit
def _run_recurrent_body(gm_mixedqkv, gm_g, gm_beta, gm_a_log, gm_dt_bias,
                        gm_out, gm_state, gm_replay_u,
                        gm_replay_k, gm_replay_decay, gm_cu_seqlens,
                        scale, lower_bound, layout_mode, batch_size,
                        sequence_length, *, row_block: int):
    head_dim = SUPPORTED_HEAD_DIM
    num_row_blocks = head_dim // row_block
    value_heads_dim = gm_g.shape[1]
    vector = FusedRecurrentKDAReplaySSMVector(row_block, MAX_VALUE_HEADS)
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
                vector, gm_mixedqkv, gm_g, gm_beta, gm_out,
                gm_state, gm_replay_u, gm_replay_k,
                gm_replay_decay, gm_cu_seqlens, scale, lower_bound,
                layout_mode, batch_size, sequence_length, value_heads_dim,
                head_item, active_rows=row_block, row_index=row_index)


@kernel
class FusedRecurrentKDAReplaySSMKernel:
    def __call__(self, mixed: Tensor, gate: Tensor, beta: Tensor,
                 a_log: Tensor, bias: Tensor, out: Tensor, state: Tensor,
                 replay_u: Tensor, replay_k: Tensor, replay_decay: Tensor,
                 cu: Tensor, scale: float, lower_bound: float, layout: int,
                 batch: int, length: int):
        batch_heads = batch * gate.shape[1]
        if batch_heads > MEDIUM_BATCH_HEAD_LIMIT:
            channel_rewind(reset_sync_id=True)
            _run_minimal_split_recurrent_body(
                mixed, gate, beta, a_log, bias, out, state, replay_u,
                replay_k, replay_decay, cu, scale, lower_bound, layout,
                batch, length)
        elif batch_heads <= SMALL_BATCH_HEAD_LIMIT:
            channel_rewind(reset_sync_id=True)
            _run_recurrent_body(
                mixed, gate, beta, a_log, bias, out, state, replay_u,
                replay_k, replay_decay, cu, scale, lower_bound, layout,
                batch, length, row_block=SMALL_ROW_BLOCK)
        else:
            channel_rewind(reset_sync_id=True)
            _run_recurrent_body(
                mixed, gate, beta, a_log, bias, out, state, replay_u,
                replay_k, replay_decay, cu, scale, lower_bound, layout,
                batch, length, row_block=MEDIUM_ROW_BLOCK)


class FusedRecurrentKDAReplaySSM:
    @host
    def run(self, mixed: Tensor, gate: Tensor, beta: Tensor,
            a_log: Tensor, bias: Tensor, out: Tensor, state: Tensor,
            replay_u: Tensor, replay_k: Tensor, replay_decay: Tensor,
            cu: Tensor, scale: float, lower_bound: float, layout: int,
            batch: int, length: int, cores: int):
        op = FusedRecurrentKDAReplaySSMKernel()
        op[cores](mixed, gate, beta, a_log, bias, out, state,
                  replay_u, replay_k, replay_decay, cu, scale,
                  lower_bound, layout, batch, length)


def _sequence_spec(dtype, shape, prefix):
    return TensorSpec(shape, dtype, stride=(*(cannbotdsl.Dim(f"{prefix}_S{i}", min=1) for i in range(3)), 1))


def _get_compiled_kernel():
    global _COMPILED_KERNEL
    with _COMPILED_KERNEL_LOCK:
        if _COMPILED_KERNEL is None:
            pb, length, packed_heads, heads, batch, capacity, cu = (
                cannbotdsl.Dim(name, min=1) for name in ("PB", "L", "N3", "N", "B", "RS", "CU")
            )
            heads = cannbotdsl.Dim("N", min=1, max=MAX_VALUE_HEADS)

            def records(dtype):
                return TensorSpec((batch, capacity, heads, TILE), dtype)

            _COMPILED_KERNEL = cannbotdsl.compile(
                FusedRecurrentKDAReplaySSM().run,
                _sequence_spec(dtypes.bfloat16, (pb, packed_heads, length, TILE), "MIXED"),
                _sequence_spec(dtypes.bfloat16, (pb, heads, length, TILE), "G"),
                _sequence_spec(dtypes.bfloat16, (pb, heads, length, 1), "BETA"),
                TensorSpec((heads,), dtypes.float32),
                TensorSpec((heads, TILE), dtypes.float32),
                _sequence_spec(dtypes.bfloat16, (pb, heads, length, TILE), "OUT"),
                TensorSpec((batch, heads, TILE, TILE), dtypes.float32),
                records(dtypes.float16),
                records(dtypes.float16),
                records(dtypes.float32),
                TensorSpec((cu,), dtypes.int32),
                dtypes.float32,
                dtypes.float32,
                dtypes.int64,
                dtypes.int64,
                dtypes.int64,
                dtypes.int64,
            )
        return _COMPILED_KERNEL


def _check_tensor(name, tensor, dtype, shape=None):
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


@dataclass(frozen=True)
class _ReplayInputs:
    """Host-side tensors for validation and layout preparation."""

    mixedqkv: torch.Tensor
    recurrent_state: torch.Tensor
    beta: torch.Tensor
    g: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    replay_u: torch.Tensor
    replay_k: torch.Tensor
    replay_decay: torch.Tensor
    cu_seqlens: torch.Tensor | None


def _validate_inputs(inputs, lower_bound):
    mixedqkv = inputs.mixedqkv
    recurrent_state = inputs.recurrent_state
    beta = inputs.beta
    g = inputs.g
    a_log = inputs.a_log
    dt_bias = inputs.dt_bias
    replay_u = inputs.replay_u
    replay_k = inputs.replay_k
    replay_decay = inputs.replay_decay
    if recurrent_state.ndim != 4:
        raise ValueError("recurrent_state must be [B,N,128,128]")
    batch, heads, dv, dk = map(int, recurrent_state.shape)
    if batch <= 0 or not 1 <= heads <= MAX_VALUE_HEADS or (dv, dk) != (TILE, TILE):
        raise ValueError("recurrent_state requires B>0, 1<=N<=96 and D=128")
    _check_tensor("recurrent_state", recurrent_state, torch.float32)
    _check_tensor("mixedqkv", mixedqkv, torch.bfloat16)
    _check_tensor("g", g, torch.bfloat16)
    _check_tensor("beta", beta, torch.bfloat16)
    _check_tensor("A_log", a_log, torch.float32, (heads,))
    _check_tensor("dt_bias", dt_bias, torch.float32, (heads, TILE))
    if replay_u.ndim != 4 or int(replay_u.shape[1]) <= 0:
        raise ValueError("replay_u must be [B,S,N,128] with S>0")
    capacity = int(replay_u.shape[1])
    for name, tensor, dtype in (
        ("replay_u", replay_u, torch.float16),
        ("replay_k", replay_k, torch.float16),
        ("replay_decay", replay_decay, torch.float32),
    ):
        _check_tensor(name, tensor, dtype, (batch, capacity, heads, TILE))
    if not -5.0 <= float(lower_bound) <= 0.0:
        raise ValueError("lower_bound must be in [-5,0]")
    return batch, heads


def _prepare_layout(inputs, layout_qkv):
    mixedqkv = inputs.mixedqkv
    g = inputs.g
    beta = inputs.beta
    cu_seqlens = inputs.cu_seqlens
    batch, heads = map(int, inputs.recurrent_state.shape[:2])
    capacity = int(inputs.replay_u.shape[1])
    if layout_qkv == "BSND":
        if mixedqkv.ndim != 3 or int(mixedqkv.shape[0]) != batch or int(mixedqkv.shape[2]) != 3 * heads * TILE:
            raise ValueError("BSND mixedqkv must be [B,S,3*N*128]")
        length = int(mixedqkv.shape[1])
        _check_tensor("g", g, torch.bfloat16, (batch, length, heads, TILE))
        _check_tensor("beta", beta, torch.bfloat16, (batch, length, heads, 1))
        mixed = mixedqkv.view(batch, length, 3 * heads, TILE).transpose(1, 2)
        gate, logits = g.transpose(1, 2), beta.transpose(1, 2)
        layout = 0
    elif layout_qkv == "BNSD":
        if mixedqkv.ndim != 4 or (int(mixedqkv.shape[0]), int(mixedqkv.shape[1]), int(mixedqkv.shape[3])) != (
            batch,
            3 * heads,
            TILE,
        ):
            raise ValueError("BNSD mixedqkv must be [B,3*N,S,128]")
        length = int(mixedqkv.shape[2])
        _check_tensor("g", g, torch.bfloat16, (batch, heads, length, TILE))
        _check_tensor("beta", beta, torch.bfloat16, (batch, heads, length, 1))
        mixed, gate, logits = mixedqkv, g, beta
        layout = 0
    elif layout_qkv == "TND":
        if mixedqkv.ndim != 2 or int(mixedqkv.shape[1]) != 3 * heads * TILE:
            raise ValueError("TND mixedqkv must be [T,3*N*128]")
        length = int(mixedqkv.shape[0])
        _check_tensor("g", g, torch.bfloat16, (length, heads, TILE))
        _check_tensor("beta", beta, torch.bfloat16, (length, heads, 1))
        if cu_seqlens is None:
            raise ValueError("cu_seqlens is required for TND")
        _check_tensor("cu_seqlens", cu_seqlens, torch.int32, (batch + 1,))
        if length > batch * capacity:
            raise ValueError("replay_u capacity is too small for TND tokens")
        mixed = mixedqkv.view(length, 3 * heads, TILE).permute(1, 0, 2).unsqueeze(0)
        gate, logits = g.permute(1, 0, 2).unsqueeze(0), beta.permute(1, 0, 2).unsqueeze(0)
        layout = 1
    else:
        raise ValueError("layout_qkv must be BSND, BNSD or TND")
    if length <= 0 or (layout == 0 and length > capacity):
        raise ValueError("replay_u capacity must cover the positive sequence length")
    return mixed, gate, logits, layout, length


def fused_recurrent_kda_replayssm(
    mixedqkv: torch.Tensor,
    recurrent_state: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float | None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = -5.0,
    layout_qkv: str = "BSND",
    *,
    replay_u: torch.Tensor,
    replay_k: torch.Tensor,
    replay_decay: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    block_num: int | None = None,
) -> torch.Tensor:
    """Evaluate candidates without changing their FP32 checkpoint.

    Args:
        mixedqkv (torch.Tensor): Contiguous BF16 packed Q/K/V in BSND,
            BNSD or TND layout, with equal head counts and D=128.
        recurrent_state (torch.Tensor): Read-only FP32 [B,N,128,128] checkpoint.
        beta (torch.Tensor): BF16 raw sigmoid logits, with trailing width 1.
        g (torch.Tensor): BF16 raw per-channel decay logits.
        scale (float | None): Query scale; None uses 128**-0.5.
        A_log (torch.Tensor): FP32 per-head log scale [N].
        dt_bias (torch.Tensor): FP32 gate bias [N,128].
        lower_bound (float): Activated gate lower bound in [-5,0].
        layout_qkv (str): BSND, BNSD or packed TND.
        replay_u (torch.Tensor): Mutated FP16 [B,S,N,128] correction records.
        replay_k (torch.Tensor): Mutated FP16 [B,S,N,128] normalized keys.
        replay_decay (torch.Tensor): Mutated FP32 [B,S,N,128] decay records.
        cu_seqlens (torch.Tensor | None): TND INT32 [B+1] token prefix sums.
        block_num (int | None): AIC-equivalent quota 8/24/28/32, or auto.

    Returns:
        torch.Tensor: BF16 candidate outputs with the same layout/shape as g.

    Notes:
        Call commit_recurrent_kda_replayssm with the previous accepted prefix
        before this verify overwrites the records. The working recurrence is
        FP32; only the exported U/K records are rounded to FP16. TND records
        use per-batch capacity S and leave the unused suffix unchanged.
        TND prefix sums must strictly increase from 0 to the packed token count,
        with every sequence length in [1,S]; the caller supplies valid NPU values.
    """
    a_log = A_log
    inputs = _ReplayInputs(
        mixedqkv,
        recurrent_state,
        beta,
        g,
        a_log,
        dt_bias,
        replay_u,
        replay_k,
        replay_decay,
        cu_seqlens,
    )
    batch, heads = _validate_inputs(inputs, lower_bound)
    scale_value = TILE**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale_value):
        raise ValueError("scale must be finite")
    mixed, gate, logits, layout, length = _prepare_layout(inputs, layout_qkv)
    tensors = (mixedqkv, recurrent_state, beta, g, a_log, dt_bias, replay_u, replay_k, replay_decay)
    if cu_seqlens is not None:
        tensors += (cu_seqlens,)
    if any(tensor.device != recurrent_state.device for tensor in tensors):
        raise ValueError("all inputs must share the recurrent_state device")
    cores = _resolve_block_num(mixedqkv, block_num)
    # A caller-supplied INT32 tensor supplies an unused dense-layout placeholder.
    cu = cu_seqlens if layout == 1 else replay_decay.view(torch.int32).reshape(-1)[:1]
    out = torch.empty(tuple(g.shape), dtype=torch.bfloat16, device=g.device)
    logical_out = out.transpose(1, 2) if layout_qkv == "BSND" else out
    if layout == 1:
        logical_out = out.permute(1, 0, 2).unsqueeze(0)
    _get_compiled_kernel()(
        mixed,
        gate,
        logits,
        a_log,
        dt_bias,
        logical_out,
        recurrent_state,
        replay_u,
        replay_k,
        replay_decay,
        cu,
        scale_value,
        float(lower_bound),
        layout,
        batch,
        length,
        _launch_blocks(batch, heads, cores),
    )
    return out


_GRAPH_LIBRARY = torch.library.Library("cannbotdsl_fused_recurrent_kda_replayssm", "DEF")
_GRAPH_LIBRARY.define(
    "fused_recurrent_kda_replayssm("
    "Tensor mixedqkv, Tensor recurrent_state, Tensor beta, Tensor g, "
    "float? scale, Tensor A_log, Tensor dt_bias, float lower_bound, str layout_qkv, "
    "Tensor(a!) replay_u, Tensor(b!) replay_k, Tensor(c!) replay_decay, "
    "Tensor? cu_seqlens=None, int? block_num=None) -> Tensor"
)


@torch.library.impl(_GRAPH_LIBRARY, "fused_recurrent_kda_replayssm", "Meta")
def _replayssm_meta(
    mixedqkv,
    recurrent_state,
    beta,
    g,
    scale,
    A_log,
    dt_bias,
    lower_bound,
    layout_qkv,
    replay_u,
    replay_k,
    replay_decay,
    cu_seqlens=None,
    block_num=None,
):
    return torch.empty_like(g, device="meta")


@torch.library.impl(_GRAPH_LIBRARY, "fused_recurrent_kda_replayssm", "PrivateUse1")
def _replayssm_dispatch(
    mixedqkv,
    recurrent_state,
    beta,
    g,
    scale,
    A_log,
    dt_bias,
    lower_bound,
    layout_qkv,
    replay_u,
    replay_k,
    replay_decay,
    cu_seqlens=None,
    block_num=None,
):
    return fused_recurrent_kda_replayssm(
        mixedqkv,
        recurrent_state,
        beta,
        g,
        scale,
        A_log,
        dt_bias,
        lower_bound,
        layout_qkv,
        replay_u=replay_u,
        replay_k=replay_k,
        replay_decay=replay_decay,
        cu_seqlens=cu_seqlens,
        block_num=block_num,
    )


fused_recurrent_kda_replayssm_op = torch.ops.cannbotdsl_fused_recurrent_kda_replayssm.fused_recurrent_kda_replayssm


@export("fused_recurrent_kda_replayssm")
def export_fused_recurrent_kda_replayssm():
    global _COMPILED_KERNEL
    try:
        _get_compiled_kernel()
    finally:
        with _COMPILED_KERNEL_LOCK:
            if _COMPILED_KERNEL is not None:
                _COMPILED_KERNEL.close()
                _COMPILED_KERNEL = None


@torch.compiler.substitute_in_graph(fused_recurrent_kda_replayssm, can_constant_fold_through=False)
def _replayssm_graph(
    mixedqkv,
    recurrent_state,
    beta,
    g,
    scale,
    A_log,
    dt_bias,
    lower_bound=-5.0,
    layout_qkv="BSND",
    *,
    replay_u,
    replay_k,
    replay_decay,
    cu_seqlens=None,
    block_num=None,
):
    return fused_recurrent_kda_replayssm_op(
        mixedqkv,
        recurrent_state,
        beta,
        g,
        scale,
        A_log,
        dt_bias,
        lower_bound,
        layout_qkv,
        replay_u,
        replay_k,
        replay_decay,
        cu_seqlens,
        block_num,
    )
