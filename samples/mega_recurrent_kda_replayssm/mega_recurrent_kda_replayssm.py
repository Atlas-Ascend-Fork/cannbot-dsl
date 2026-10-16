# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""One mixed Mega KDA kernel with a stream-controlled 8/16/24/28/32 grid.

get_block_num() selects the schedule inside the kernel. The host compiles
once, then derives the launch grid from get_platform_info() on the current
stream. The public B=16,32,...,512 contract uses runtime batch groups; S is fixed at 8.
Each grid retains its established output-projection tile/buffer configuration.
"""

# Tensor operands are explicit to preserve the kernel ABI and DSL lexical order.
# Hardware stages and synchronization/movement order are fixed expansions;
# extracting helpers would change DSL lowering or measured kernel performance.

from __future__ import annotations

import math
from typing import NamedTuple

import cannbotdsl
import torch
from cannbotdsl import (
    ChannelKind,
    MemLoc,
    PIPE,
    Tensor,
    TensorSpec,
    channel_rewind,
    dtypes,
)
from cannbotdsl.aot import export
from cannbotdsl.buffer import Buffer
from cannbotdsl.channel import Channel
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy
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
    vmem_bar,
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
from cannbotdsl.ops.sync import (
    cube_sync_all,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    global_sync_all,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
    vec_sync_notify,
    vec_sync_wait,
)
from cannbotdsl.tensor import (
    ceil_div,
    idx2crd,
    make_tiler,
    reinterpret,
    tile_slice,
)

TILE = 128
VL = 64
D_SEGMENTS = TILE // VL
CONV_KERNEL = 4
BATCH_TILE = 16
MAX_BATCH_GROUPS = 32
BETA_TILE = 16
HIDDEN_SIZE = 7168
SEQ_LEN = 8
OPROJ_KC = 256
OPROJ_KS = 64
PROJECTION_KC = 256
PROJECTION_KS = 128
CONV_RAW_V_TO_MTE3_EVENT = 2
KDA_STATE_KEY_UNROLL = 4
KDA_STATE_UPDATE_UNROLL = 2
_COMPILED_LAUNCH = None


def _device_block_num(ref: torch.Tensor) -> int:
    """Select the largest supported grid within the current stream quota."""
    if ref.device.type not in {"npu", "privateuseone"}:
        raise ValueError("mega_recurrent_kda_replayssm requires NPU inputs")
    device_index = ref.device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    stream = torch.npu.current_stream(device_index)
    properties = cannbotdsl.get_platform_info(stream=stream)
    try:
        cube_core_num = int(properties.cube_core_num)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"NPU {device_index} does not expose a valid AIC count"
        ) from error
    if cube_core_num < 8:
        raise ValueError(
            f"current stream exposes {cube_core_num} AIC cores; at least 8 are required"
        )
    return max(cores for cores in (8, 16, 24, 28, 32) if cores <= cube_core_num)


class _CompiledLaunch:
    __slots__ = ("fn", "workspace", "workspace_views", "norm_exchange")

    def __init__(
        self,
        fn,
        workspace: torch.Tensor,
        workspace_views: tuple[torch.Tensor, ...],
        norm_exchange: torch.Tensor,
    ):
        self.fn = fn
        self.workspace = workspace
        self.workspace_views = workspace_views
        self.norm_exchange = norm_exchange


def publish_vector_grid_to_cube(vector_flag: int, cube_flag: int, ack_flag: int):
    """Acknowledge slot receipt before AIV flag reuse, then project concurrently."""
    vec_sync_block_arrive(PIPE.MTE3, vector_flag, mode=2)
    cube_sync_block_wait(PIPE.S, vector_flag, mode=2)
    cube_sync_block_arrive(PIPE.MTE3, ack_flag, mode=2)
    vec_sync_block_wait(PIPE.S, ack_flag, mode=2)
    cube_sync_block_arrive(PIPE.FIXPIPE, cube_flag, mode=0)
    cube_sync_block_wait(PIPE.S, cube_flag, mode=0)


def publish_batch_group_vector_grid_to_cube(vector_flag: int, cube_flag: int):
    """Publish ordered AIV MTE3 writes to every AIC without a return broadcast."""
    vec_sync_block_arrive(PIPE.MTE3, vector_flag, mode=2)
    cube_sync_block_wait(PIPE.S, vector_flag, mode=2)
    cube_sync_block_arrive(PIPE.FIXPIPE, cube_flag, mode=0)
    cube_sync_block_wait(PIPE.S, cube_flag, mode=0)


class ProjectionStage:
    """Cube projection over all BSH rows and paired-AIV publication."""

    def __init__(
        self,
        row_count: int = TILE,
        input_l2_cache_ctl: int = 0,
        weight_l2_cache_ctl: int = 0,
        tile_width: int = TILE,
        row_capacity: int = 128,
    ):
        self.row_count = row_count
        self.row_capacity = int(row_capacity)
        self.tile_width = int(tile_width)
        self.row_block = self.row_count // 2
        self.input_l2_cache_ctl = int(input_l2_cache_ctl)
        self.weight_l2_cache_ctl = int(weight_l2_cache_ctl)
        self.a_l1 = Channel(
            MemLoc.L1,
            (self.row_count, PROJECTION_KC),
            dtypes.bfloat16,
            depth=2,
            capacity=(self.row_capacity, PROJECTION_KC),
        )
        if self.tile_width != BETA_TILE:
            self.b_l1 = Channel(
                MemLoc.L1,
                (self.tile_width, PROJECTION_KC),
                dtypes.bfloat16,
                depth=2,
                data_format="nz",
            )
        self.l0a = Channel(
            MemLoc.L0A,
            (self.row_count, PROJECTION_KS),
            dtypes.bfloat16,
            depth=2,
            capacity=(self.row_capacity, PROJECTION_KS),
        )
        self.l0b = Channel(
            MemLoc.L0B, (self.tile_width, PROJECTION_KS), dtypes.bfloat16, depth=2
        )
        self.l0c = Channel(
            MemLoc.L0C,
            (self.row_count, self.tile_width),
            dtypes.float32,
            depth=2,
            capacity=(self.row_capacity, self.tile_width),
        )
        self.projection_half = Channel(
            MemLoc.UB,
            (self.row_block, self.tile_width),
            dtypes.float32,
            depth=1,
            kind=ChannelKind.CrossCore,
            capacity=(self.row_capacity // 2, self.tile_width),
        ).produce()
        self.projection_bf16 = Buffer(
            MemLoc.UB,
            (self.row_block, self.tile_width),
            dtypes.bfloat16,
            capacity=(self.row_capacity // 2, self.tile_width),
        )
        self.nd2nz = make_copy_engine(format_transform="nd2nz")
        self.fixpipe = make_copy_engine(split_axis=0)
        # Acquired lazily so FIXPIPE and vector stages keep their original order.
        self.projection_fp32: object

    @jit
    def matmul_hidden(
        self,
        hidden_gm: Tensor,
        weight_gm: Tensor,
        output_tile,
        k_tiles,
    ):
        """Compute ``hidden @ weight.T`` from model-layout ``[out,in]``."""
        if const_expr(self.tile_width == BETA_TILE):
            # Beta's six logical rows require the padded whole-root NZ path.
            beta_l1 = Buffer(
                MemLoc.L1,
                dtype=weight_gm.dtype,
                capacity=(BETA_TILE, HIDDEN_SIZE),
                layout=weight_gm.layout,
                physical_layout=weight_gm.physical_layout,
                data_format="nz",
            )
            mem_copy(beta_l1, weight_gm, l2_cache_ctl=self.weight_l2_cache_ctl)
        accumulator = self.l0c.produce()
        for kc_idx in range(k_tiles * TILE // PROJECTION_KC):
            a_l1_slot = self.a_l1.produce()
            mem_copy(
                a_l1_slot,
                tile_slice(hidden_gm, (self.row_capacity, PROJECTION_KC), (0, kc_idx)),
                l2_cache_ctl=self.input_l2_cache_ctl,
                engine=self.nd2nz,
            )
            if const_expr(self.tile_width == BETA_TILE):
                self._accumulate_panel(
                    accumulator,
                    a_l1_slot,
                    beta_l1,
                    kc_idx * (PROJECTION_KC // PROJECTION_KS),
                    kc_idx == 0,
                )
            else:
                b_l1_slot = self.b_l1.produce()
                mem_copy(
                    b_l1_slot,
                    tile_slice(
                        weight_gm,
                        (self.tile_width, PROJECTION_KC),
                        (output_tile, kc_idx),
                    ),
                    l2_cache_ctl=self.weight_l2_cache_ctl,
                )
                self._accumulate_panel(
                    accumulator, a_l1_slot, b_l1_slot, 0, kc_idx == 0
                )

    @jit
    def _accumulate_panel(self, accumulator, a_l1_slot, b_l1_slot, first_k, init):
        for ks_idx in range(PROJECTION_KC // PROJECTION_KS):
            l0a_slot = self.l0a.produce()
            l0b_slot = self.l0b.produce()
            mem_copy(
                l0a_slot,
                tile_slice(a_l1_slot, (self.row_capacity, PROJECTION_KS), (0, ks_idx)),
            )
            mem_copy(
                l0b_slot,
                tile_slice(
                    b_l1_slot, (self.tile_width, PROJECTION_KS), (0, first_k + ks_idx)
                ),
            )
            matmul(accumulator, l0a_slot, l0b_slot, init=init and ks_idx == 0)

    @jit
    def matmul_resident(
        self,
        resident_l1,
        weight_l1,
        resident_copy_engine=None,
    ):
        """Compute one K=128 projection from a resident BF16 L1 tile."""
        accumulator = self.l0c.produce()
        for ks_idx in range(TILE // PROJECTION_KS):
            l0a_slot = self.l0a.produce()
            mem_copy(
                l0a_slot,
                tile_slice(
                    resident_l1, (self.row_capacity, PROJECTION_KS), (0, ks_idx)
                ),
                engine=resident_copy_engine,
            )
            l0b_slot = self.l0b.produce()
            mem_copy(
                l0b_slot,
                tile_slice(weight_l1, (self.tile_width, PROJECTION_KS), (0, ks_idx)),
            )
            matmul(accumulator, l0a_slot, l0b_slot, init=ks_idx == 0)

    @jit
    def acquire_result(self):
        """Send L0C to both paired AIVs and hold the FP32 half tile."""
        mem_copy(
            self.projection_half,
            reinterpret(self.l0c.consume(), shape=(self.row_count, self.tile_width)),
            engine=self.fixpipe,
        )
        self.projection_fp32 = self.projection_half

    @jit
    def cast_result(self):
        """Send L0C to both paired AIVs and cast each AIV half."""
        self.acquire_result()
        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            for segment in range(self.row_block * self.tile_width // VL):
                offset = segment * VL
                projected = vload(self.projection_fp32, offset)
                projected_bf16 = vcast(projected, dtypes.bfloat16, mask=full)
                vstore_pack(
                    self.projection_bf16,
                    offset,
                    projected_bf16,
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )
        vec_sync_all()

    @jit
    def project(self, hidden_gm: Tensor, weight_gm: Tensor, n_idx, k_tiles):
        """Compute one output tile and cast its AIV half."""
        self.matmul_hidden(hidden_gm, weight_gm, n_idx, k_tiles)
        self.cast_result()

    def publish(self, workspace_tile, subblock_idx):
        out_half = tile_slice(
            workspace_tile,
            make_tiler((self.row_block, self.tile_width), alignment=(1, 1)),
            (subblock_idx, 0),
        )
        mem_copy(out_half, self.projection_bf16)

    def publish_source(self, workspace_tile, source, subblock_idx):
        out_half = tile_slice(
            workspace_tile,
            make_tiler((self.row_block, self.tile_width), alignment=(1, 1)),
            (subblock_idx, 0),
        )
        mem_copy(out_half, source)


class N384ProjectionStage:
    """M128xN384 projection that reuses each hidden panel for three outputs."""

    def __init__(self, input_l2_cache_ctl: int = 0, weight_l2_cache_ctl: int = 0):
        self.input_l2_cache_ctl = int(input_l2_cache_ctl)
        self.weight_l2_cache_ctl = int(weight_l2_cache_ctl)
        # A and B ping-pong consume the complete 512-KiB L1 budget:
        # 2 * M128xK256 plus 2 * N384xK256 BF16.
        self.a_l1 = Channel(MemLoc.L1, (TILE, PROJECTION_KC), dtypes.bfloat16, depth=2)
        self.b_l1 = Channel(
            MemLoc.L1,
            (3 * TILE, PROJECTION_KC),
            dtypes.bfloat16,
            depth=2,
            data_format="nz",
        )
        self.l0a = Channel(MemLoc.L0A, (TILE, PROJECTION_KS), dtypes.bfloat16, depth=2)
        self.l0b = Channel(MemLoc.L0B, (TILE, PROJECTION_KS), dtypes.bfloat16, depth=2)
        self.l0c0 = Channel(MemLoc.L0C, (TILE, TILE), dtypes.float32, depth=1).produce()
        self.l0c1 = Channel(MemLoc.L0C, (TILE, TILE), dtypes.float32, depth=1).produce()
        self.l0c2 = Channel(MemLoc.L0C, (TILE, TILE), dtypes.float32, depth=1).produce()
        # Independent channels let the final panel's MMAD overlap the prior
        # panel's FIXPIPE/AIV drain while preserving explicit 0.5.1 epochs.
        self.projection_half0 = Channel(
            MemLoc.UB,
            (TILE // 2, TILE),
            dtypes.float32,
            depth=1,
            kind=ChannelKind.CrossCore,
        ).produce()
        self.projection_half1 = Channel(
            MemLoc.UB,
            (TILE // 2, TILE),
            dtypes.float32,
            depth=1,
            kind=ChannelKind.CrossCore,
        ).produce()
        self.projection_half2 = Channel(
            MemLoc.UB,
            (TILE // 2, TILE),
            dtypes.float32,
            depth=1,
            kind=ChannelKind.CrossCore,
        ).produce()
        self.projection_bf16 = Buffer(MemLoc.UB, (TILE // 2, TILE), dtypes.bfloat16)
        self.nd2nz = make_copy_engine(format_transform="nd2nz")
        self.fixpipe = make_copy_engine(split_axis=0)
        # Selected lazily from one of the three resident projection buffers.
        self.projection_fp32: object

    @jit
    def matmul_hidden3(self, hidden_gm, weight_gm, m_group, n_macro, k_tiles):
        for kc_idx in range(k_tiles * TILE // PROJECTION_KC):
            a_l1_slot = self.a_l1.produce()
            mem_copy(
                a_l1_slot,
                tile_slice(hidden_gm, (TILE, PROJECTION_KC), (m_group, kc_idx)),
                engine=self.nd2nz,
                l2_cache_ctl=self.input_l2_cache_ctl,
            )
            b_l1_slot = self.b_l1.produce()
            mem_copy(
                b_l1_slot,
                tile_slice(weight_gm, (3 * TILE, PROJECTION_KC), (n_macro, kc_idx)),
                l2_cache_ctl=self.weight_l2_cache_ctl,
            )
            for ks_idx in range(PROJECTION_KC // PROJECTION_KS):
                l0a_slot = self.l0a.produce()
                mem_copy(
                    l0a_slot, tile_slice(a_l1_slot, (TILE, PROJECTION_KS), (0, ks_idx))
                )
                l0b_slot = self.l0b.produce()
                mem_copy(
                    l0b_slot, tile_slice(b_l1_slot, (TILE, PROJECTION_KS), (0, ks_idx))
                )
                matmul(self.l0c0, l0a_slot, l0b_slot, init=kc_idx == 0 and ks_idx == 0)
                if (
                    kc_idx == k_tiles * TILE // PROJECTION_KC - 1
                    and ks_idx == PROJECTION_KC // PROJECTION_KS - 1
                ):
                    mem_copy(
                        self.projection_half0,
                        reinterpret(self.l0c0, shape=(TILE, TILE)),
                        engine=self.fixpipe,
                    )
                l0b_slot = self.l0b.produce()
                mem_copy(
                    l0b_slot, tile_slice(b_l1_slot, (TILE, PROJECTION_KS), (1, ks_idx))
                )
                matmul(self.l0c1, l0a_slot, l0b_slot, init=kc_idx == 0 and ks_idx == 0)
                if (
                    kc_idx == k_tiles * TILE // PROJECTION_KC - 1
                    and ks_idx == PROJECTION_KC // PROJECTION_KS - 1
                ):
                    mem_copy(
                        self.projection_half1,
                        reinterpret(self.l0c1, shape=(TILE, TILE)),
                        engine=self.fixpipe,
                    )
                l0b_slot = self.l0b.produce()
                mem_copy(
                    l0b_slot, tile_slice(b_l1_slot, (TILE, PROJECTION_KS), (2, ks_idx))
                )
                matmul(self.l0c2, l0a_slot, l0b_slot, init=kc_idx == 0 and ks_idx == 0)
                if (
                    kc_idx == k_tiles * TILE // PROJECTION_KC - 1
                    and ks_idx == PROJECTION_KC // PROJECTION_KS - 1
                ):
                    mem_copy(
                        self.projection_half2,
                        reinterpret(self.l0c2, shape=(TILE, TILE)),
                        engine=self.fixpipe,
                    )

    @jit
    def acquire_result0(self):
        self.projection_fp32 = self.projection_half0

    @jit
    def acquire_result1(self):
        self.projection_fp32 = self.projection_half1

    @jit
    def acquire_result2(self):
        self.projection_fp32 = self.projection_half2

    @jit
    def cast_result(self):
        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            for segment in range((TILE // 2) * TILE // VL):
                offset = segment * VL
                value = vcast(
                    vload(self.projection_fp32, offset), dtypes.bfloat16, mask=full
                )
                vstore_pack(
                    self.projection_bf16,
                    offset,
                    value,
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )

    @staticmethod
    def publish(workspace_gm, m_group, output_head, subblock_idx, source):
        whole_tile = tile_slice(workspace_gm, (TILE, TILE), (m_group, output_head))
        out_half = tile_slice(
            whole_tile,
            make_tiler((TILE // 2, TILE), alignment=(1, 1)),
            (subblock_idx, 0),
        )
        mem_copy(out_half, source)


class GateDecayProjectionStage:
    """Share one hidden K128 walk across gate heads 3..5 and decay Fa."""

    def __init__(self, input_l2_cache_ctl: int = 0, weight_l2_cache_ctl: int = 0):
        self.input_l2_cache_ctl = int(input_l2_cache_ctl)
        self.weight_l2_cache_ctl = int(weight_l2_cache_ctl)
        self.a_l1 = Channel(MemLoc.L1, (TILE, TILE), dtypes.bfloat16, depth=2)
        self.g_l1 = Channel(
            MemLoc.L1, (3 * TILE, TILE), dtypes.bfloat16, depth=2, data_format="nz"
        )
        self.fa_weight_l1 = Channel(
            MemLoc.L1, (TILE, TILE), dtypes.bfloat16, depth=2, data_format="nz"
        )
        self.l0a = Channel(MemLoc.L0A, (TILE, TILE), dtypes.bfloat16, depth=2)
        self.l0b = Channel(MemLoc.L0B, (TILE, TILE), dtypes.bfloat16, depth=2)
        self.l0c0 = Channel(MemLoc.L0C, (TILE, TILE), dtypes.float32, depth=1).produce()
        self.l0c1 = Channel(MemLoc.L0C, (TILE, TILE), dtypes.float32, depth=1).produce()
        self.l0c2 = Channel(MemLoc.L0C, (TILE, TILE), dtypes.float32, depth=1).produce()
        self.l0c_fa = Channel(
            MemLoc.L0C, (TILE, TILE), dtypes.float32, depth=1
        ).produce()
        self.fa_l1 = Channel(
            MemLoc.L1, (TILE, TILE), dtypes.bfloat16, depth=1
        ).produce()
        self.fb_l1 = Channel(
            MemLoc.L1, (TILE, TILE), dtypes.bfloat16, depth=1, data_format="nz"
        ).produce()
        self.nd2nz = make_copy_engine(format_transform="nd2nz")
        self.fixpipe = make_copy_engine()
        self.fa_copy_engine = make_copy_engine()

    @jit
    def matmul_hidden3_fa(
        self,
        hidden_gm,
        gate_weight_gm,
        fa_weight_gm,
        m_group,
        k_tiles,
    ):
        for k_idx in range(k_tiles):
            a_l1_slot = self.a_l1.produce()
            mem_copy(
                a_l1_slot,
                tile_slice(hidden_gm, (TILE, TILE), (m_group, k_idx)),
                engine=self.nd2nz,
                l2_cache_ctl=self.input_l2_cache_ctl,
            )
            g_l1_slot = self.g_l1.produce()
            mem_copy(
                g_l1_slot,
                tile_slice(gate_weight_gm, (3 * TILE, TILE), (1, k_idx)),
                l2_cache_ctl=self.weight_l2_cache_ctl,
            )
            fa_weight_l1_slot = self.fa_weight_l1.produce()
            mem_copy(
                fa_weight_l1_slot,
                tile_slice(fa_weight_gm, (TILE, TILE), (0, k_idx)),
                l2_cache_ctl=self.weight_l2_cache_ctl,
            )
            l0a_slot = self.l0a.produce()
            mem_copy(l0a_slot, a_l1_slot)
            l0b_slot = self.l0b.produce()
            mem_copy(l0b_slot, tile_slice(g_l1_slot, (TILE, TILE), (0, 0)))
            matmul(self.l0c0, l0a_slot, l0b_slot, init=k_idx == 0)
            l0b_slot = self.l0b.produce()
            mem_copy(l0b_slot, tile_slice(g_l1_slot, (TILE, TILE), (1, 0)))
            matmul(self.l0c1, l0a_slot, l0b_slot, init=k_idx == 0)
            l0b_slot = self.l0b.produce()
            mem_copy(l0b_slot, tile_slice(g_l1_slot, (TILE, TILE), (2, 0)))
            matmul(self.l0c2, l0a_slot, l0b_slot, init=k_idx == 0)
            l0b_slot = self.l0b.produce()
            mem_copy(l0b_slot, fa_weight_l1_slot)
            matmul(self.l0c_fa, l0a_slot, l0b_slot, init=k_idx == 0)

    @jit
    def _publish_l0c(
        self,
        workspace_gm,
        source_l0c,
        m_group,
        output_head,
        subblock_idx,
    ):
        whole_tile = tile_slice(workspace_gm, (TILE, TILE), (m_group, output_head))
        mem_copy(whole_tile, source_l0c, engine=self.fixpipe)

    @jit
    def publish_gate(self, output_gate_workspace_gm, m_group, subblock_idx):
        self._publish_l0c(output_gate_workspace_gm, self.l0c0, m_group, 3, subblock_idx)
        self._publish_l0c(output_gate_workspace_gm, self.l0c1, m_group, 4, subblock_idx)
        self._publish_l0c(output_gate_workspace_gm, self.l0c2, m_group, 5, subblock_idx)

    @jit
    def project_fb(self, decay_workspace_gm, fb_weight_gm, m_group, subblock_idx):
        mem_copy(self.fa_l1, reinterpret(self.l0c_fa, shape=(TILE, TILE)))
        for output_head in range(6):
            mem_copy(
                self.fb_l1,
                tile_slice(fb_weight_gm, (TILE, TILE), (output_head, 0)),
                l2_cache_ctl=self.weight_l2_cache_ctl,
            )
            l0a_slot = self.l0a.produce()
            mem_copy(l0a_slot, self.fa_l1, engine=self.fa_copy_engine)
            l0b_slot = self.l0b.produce()
            mem_copy(l0b_slot, self.fb_l1)
            matmul(self.l0c_fa, l0a_slot, l0b_slot, init=True)
            self._publish_l0c(
                decay_workspace_gm, self.l0c_fa, m_group, output_head, subblock_idx
            )


class LowRankDecayStage:
    """Keep ``f_a`` in L1 and execute six dependent ``f_b`` tiles."""

    def __init__(
        self,
        projection_stage: ProjectionStage,
        num_heads: int,
    ):
        self.projection_stage = projection_stage
        self.num_heads = int(num_heads)
        self.fa_l1 = Channel(
            MemLoc.L1,
            (self.projection_stage.row_count, TILE),
            dtypes.bfloat16,
            depth=1,
            capacity=(self.projection_stage.row_capacity, TILE),
        ).produce()
        self.fb_l1 = Channel(
            MemLoc.L1, (TILE, TILE), dtypes.bfloat16, depth=1, data_format="nz"
        ).produce()
        # FIXPIPE stores Fa with the L1 capacity stride, unlike packed GM loads.
        self.fa_copy_engine = make_copy_engine()

    @jit
    def project(
        self,
        decay_workspace_gm: Tensor,
        hidden_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        k_tiles,
        subblock_idx,
    ):
        """Compute ``f_b(f_a(hidden))`` without a GM intermediate."""
        self.projection_stage.matmul_hidden(hidden_gm, fa_weight_gm, 0, k_tiles)
        fa_result = reinterpret(
            self.projection_stage.l0c.consume(),
            shape=(self.projection_stage.row_count, TILE),
        )
        mem_copy(self.fa_l1, fa_result)
        for output_head in range(self.num_heads):
            mem_copy(
                self.fb_l1,
                tile_slice(fb_weight_gm, (TILE, TILE), (output_head, 0)),
                l2_cache_ctl=self.projection_stage.weight_l2_cache_ctl,
            )
            self.projection_stage.matmul_resident(
                self.fa_l1, self.fb_l1, self.fa_copy_engine
            )
            self.projection_stage.cast_result()
            self.projection_stage.publish(
                tile_slice(
                    decay_workspace_gm,
                    (self.projection_stage.row_capacity, TILE),
                    (0, output_head),
                ),
                subblock_idx,
            )
            # The six heads reuse one BF16 staging buffer.  Do not let the
            # next Vector cast overwrite it before this head's MTE3 publish
            # has consumed the data.
            vec_sync_all()


class ConvSiluStage:
    """K=4 depthwise causal Conv1D and SiLU in the same AIV pipeline."""

    def __init__(self, seq_len: int, row_block: int, row_capacity: int = 64):
        self.seq_len = seq_len
        self.row_block = row_block
        self.window = Buffer(MemLoc.UB, (CONV_KERNEL, TILE), dtypes.bfloat16)
        # Keep the three panel slots in a flat 2-D allocation.  tile_slice uses
        # 2-D tile coordinates throughout this kernel, so flattening the slot
        # dimension avoids relying on rank-reducing view semantics.
        self.weight = Buffer(MemLoc.UB, (3 * CONV_KERNEL, TILE), dtypes.bfloat16)
        self.output_bf16 = Buffer(
            MemLoc.UB,
            (self.row_block, TILE),
            dtypes.bfloat16,
            capacity=(row_capacity, TILE),
        )
        self.raw_bf16 = Buffer(
            MemLoc.UB,
            (self.row_block, TILE),
            dtypes.bfloat16,
            capacity=(row_capacity, TILE),
        )
        self.weight_ch = Channel(
            MemLoc.UB, (CONV_KERNEL, TILE), dtypes.bfloat16, depth=1
        ).produce()

    @jit
    def install_history(self, history_ub):
        mem_copy(tile_slice(self.window, (3, TILE), (0, 0)), history_ub)
        vec_sync_all()

    @jit
    def load_weight(self, weight_gm):
        mem_copy(self.weight_ch, weight_gm)
        mem_copy(tile_slice(self.weight, (CONV_KERNEL, TILE), (0, 0)), self.weight_ch)
        vec_sync_all()

    @jit
    def convolve_sequence(self, local_batch: int, raw_projection, weight_ub):
        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            for segment in range(TILE // VL):
                off = segment * VL
                x0_bf16 = vload_unpack(
                    self.window, off, unpack_mode=UnpackMode.B16_TO_B32
                )
                x0 = vcast(x0_bf16, dtypes.float32, mask=full)
                x1_bf16 = vload_unpack(
                    self.window, TILE + off, unpack_mode=UnpackMode.B16_TO_B32
                )
                x1 = vcast(x1_bf16, dtypes.float32, mask=full)
                x2_bf16 = vload_unpack(
                    self.window, 2 * TILE + off, unpack_mode=UnpackMode.B16_TO_B32
                )
                x2 = vcast(x2_bf16, dtypes.float32, mask=full)
                w0_bf16 = vload_unpack(
                    weight_ub, off, unpack_mode=UnpackMode.B16_TO_B32
                )
                w0 = vcast(w0_bf16, dtypes.float32, mask=full)
                w1_bf16 = vload_unpack(
                    weight_ub, TILE + off, unpack_mode=UnpackMode.B16_TO_B32
                )
                w1 = vcast(w1_bf16, dtypes.float32, mask=full)
                w2_bf16 = vload_unpack(
                    weight_ub, 2 * TILE + off, unpack_mode=UnpackMode.B16_TO_B32
                )
                w2 = vcast(w2_bf16, dtypes.float32, mask=full)
                w3_bf16 = vload_unpack(
                    weight_ub, 3 * TILE + off, unpack_mode=UnpackMode.B16_TO_B32
                )
                w3 = vcast(w3_bf16, dtypes.float32, mask=full)
                for token in range(self.seq_len):
                    local_row = local_batch * self.seq_len + token
                    raw_offset = local_row * TILE
                    x3 = vload(raw_projection, raw_offset + off)
                    x3_bf16 = vcast(x3, dtypes.bfloat16, mask=full)
                    product1 = vmul(x1, w1, mask=full)
                    product01 = vmadd(x0, w0, product1, mask=full)
                    product3 = vmul(x3, w3, mask=full)
                    product23 = vmadd(x2, w2, product3, mask=full)
                    acc = vadd(product01, product23, mask=full)
                    negative_acc = vneg(acc, mask=full)
                    exp_value = vexp(negative_acc, mask=full)
                    denominator = vadds(exp_value, 1.0, mask=full)
                    activated = vdiv(acc, denominator, mask=full)
                    activated_bf16 = vcast(activated, dtypes.bfloat16, mask=full)
                    output_offset = raw_offset + off
                    vstore_pack(
                        self.output_bf16,
                        output_offset,
                        activated_bf16,
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                    vstore_pack(
                        self.raw_bf16,
                        output_offset,
                        x3_bf16,
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                    x0 = x1
                    x1 = x2
                    x2 = vcast(x3_bf16, dtypes.float32, mask=full)
                final_x0 = vcast(x0, dtypes.bfloat16, mask=full)
                final_x1 = vcast(x1, dtypes.bfloat16, mask=full)
                final_x2 = vcast(x2, dtypes.bfloat16, mask=full)
                vstore_pack(
                    self.window, off, final_x0, full, pack_mode=PackMode.B32_TO_B16
                )
                vstore_pack(
                    self.window,
                    TILE + off,
                    final_x1,
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    self.window,
                    2 * TILE + off,
                    final_x2,
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )
            vmem_bar("vst_vld")

    @jit
    def zero_sequence(self, local_batch: int):
        with vf(mode="simd"):
            full_bf16, _ = update_mask(TILE, elem_bits=16)
            zero = vdups(0.0, dtypes.bfloat16, mask=full_bf16)
            for token in range(self.seq_len):
                local_row = local_batch * self.seq_len + token
                vstore(self.output_bf16, local_row * TILE, zero, full_bf16)
            vmem_bar("vst_vld")


class RecurrentKDAStage:
    """Replay-record delta-rule recurrence for one complete value head."""

    def __init__(
        self,
        row_block: int,
        num_heads: int,
        state_key_unroll: int = KDA_STATE_KEY_UNROLL,
        state_update_unroll: int = KDA_STATE_UPDATE_UNROLL,
    ):
        self.row_block = row_block
        self.num_heads = int(num_heads)
        self.state_key_unroll = int(state_key_unroll)
        self.state_update_unroll = int(state_update_unroll)
        self.state = Channel(
            MemLoc.UB, (self.row_block, TILE), dtypes.float32, depth=1
        ).produce()
        self.key = Buffer(MemLoc.UB, (1, TILE), dtypes.float32)
        self.query = Buffer(MemLoc.UB, (1, TILE), dtypes.float32)
        self.decay_exp = Buffer(MemLoc.UB, (1, TILE), dtypes.float32)
        self.ub_value = Buffer(MemLoc.UB, (1, self.row_block), dtypes.float32)
        self.ub_beta = Buffer(MemLoc.UB, (1, VL), dtypes.float32)
        self.out = Buffer(MemLoc.UB, (1, self.row_block), dtypes.float32)
        self.value = Channel(MemLoc.UB, (1, self.row_block), dtypes.bfloat16, depth=2)
        self.state_key_sums = Buffer(MemLoc.UB, (1, self.row_block), dtypes.float32)
        self.delta_row = Buffer(MemLoc.UB, (1, self.row_block), dtypes.float32)
        # vLLM ReplaySSM PR #47576 stores GDN d/k records as FP16 when the
        # model activations are BF16.  Arithmetic remains FP32 in this stage.
        self.replay_u = Channel(MemLoc.UB, (1, self.row_block), dtypes.float16, depth=1)
        self.replay_k = Channel(MemLoc.UB, (1, TILE), dtypes.float16, depth=1)
        self.replay_decay = Channel(MemLoc.UB, (1, TILE), dtypes.float32, depth=1)
        self.raw_query = Channel(MemLoc.UB, (1, TILE), dtypes.bfloat16, depth=2)
        self.raw_key = Channel(MemLoc.UB, (1, TILE), dtypes.bfloat16, depth=2)
        self.raw_g = Channel(MemLoc.UB, (1, TILE), dtypes.bfloat16, depth=2)
        self.raw_beta = Channel(MemLoc.UB, (1, VL), dtypes.bfloat16, depth=2)
        self.dt_bias = Buffer(MemLoc.UB, (self.num_heads, TILE), dtypes.float32)
        self.a_log = Buffer(MemLoc.UB, (self.num_heads,), dtypes.float32)

    def load_gate_params(self, a_log_gm, dt_bias_gm):
        mem_copy(self.a_log, a_log_gm)
        mem_copy(self.dt_bias, dt_bias_gm)

    def load_state(self, state_gm):
        mem_copy(self.state, state_gm, l2_cache_ctl=2)

    def prefetch_inputs(
        self, raw_key_gm, raw_query_gm, raw_g_gm, value_gm, raw_beta_gm
    ):
        mem_copy(self.raw_query.produce(), raw_query_gm)
        mem_copy(self.raw_key.produce(), raw_key_gm)
        mem_copy(self.raw_g.produce(), raw_g_gm)
        mem_copy(tile_slice(self.raw_beta.produce(), (1, 1), (0, 0)), raw_beta_gm)
        mem_copy(self.value.produce(), value_gm)

    @jit
    def normalize_qk(self, scale_value: float):
        raw_query = self.raw_query.consume()
        raw_key = self.raw_key.consume()
        replay_k = self.replay_k.produce()
        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            query_pre_bf16 = vload_unpack(
                raw_query, 0, unpack_mode=UnpackMode.B16_TO_B32
            )
            query_pre = vcast(query_pre_bf16, dtypes.float32, mask=full)
            query_post_bf16 = vload_unpack(
                raw_query, VL, unpack_mode=UnpackMode.B16_TO_B32
            )
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
    def activate_g_beta(self, lower_bound: float, value_head):
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
            raw_gate_pre_bf16 = vload_unpack(
                raw_g, 0, unpack_mode=UnpackMode.B16_TO_B32
            )
            raw_gate_pre = vcast(raw_gate_pre_bf16, dtypes.float32, mask=full)
            raw_gate_post_bf16 = vload_unpack(
                raw_g, VL, unpack_mode=UnpackMode.B16_TO_B32
            )
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

            beta_logit_bf16 = vload_unpack(
                raw_beta, 0, unpack_mode=UnpackMode.B16_TO_B32
            )
            beta_logit = vcast(beta_logit_bf16, dtypes.float32, mask=full)
            beta = vneg(beta_logit, mask=full)
            beta = vexp(beta, mask=full)
            beta = vadds(beta, 1.0, mask=full)
            beta = vdiv(one, beta, mask=full)
            vstore(self.ub_beta, 0, beta, full)
            value_bf16 = vload_unpack(value_slot, 0, unpack_mode=UnpackMode.B16_TO_B32)
            value = vcast(value_bf16, dtypes.float32, mask=full)
            vstore(self.ub_value, 0, value, full)
            value_post_bf16 = vload_unpack(
                value_slot, VL, unpack_mode=UnpackMode.B16_TO_B32
            )
            value_post = vcast(value_post_bf16, dtypes.float32, mask=full)
            vstore(self.ub_value, VL, value_post, full)

    @jit
    def compute_delta(self):
        replay_u = self.replay_u.produce()
        with vf(mode="simd"):
            mask, _ = update_mask(VL, elem_bits=32)
            decay_pre = vload(self.decay_exp, 0)
            decay_post = vload(self.decay_exp, VL)
            key_pre = vload(self.key, 0)
            key_post = vload(self.key, VL)
            decay_key_pre = vmul(decay_pre, key_pre, mask=mask)
            decay_key_post = vmul(decay_post, key_post, mask=mask)
            for dv in dsl_range(0, self.row_block, 1, unroll=self.state_key_unroll):
                offset = dv * TILE
                state_key_pre = vload(self.state, offset)
                state_key_post = vload(self.state, offset + VL)
                state_key = vmul(state_key_pre, decay_key_pre, mask=mask)
                state_key_post = vmul(state_key_post, decay_key_post, mask=mask)
                state_key = vadd(state_key, state_key_post, mask=mask)
                state_key_sum = vreduce_sum(state_key, mask=mask)
                vstore_first(self.state_key_sums, dv, state_key_sum)
        with vf(mode="simd"):
            mask, _ = update_mask(VL, elem_bits=32)
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
    def update_state(self):
        with vf(mode="simd"):
            mask, _ = update_mask(VL, elem_bits=32)
            decay_pre = vload(self.decay_exp, 0)
            decay_post = vload(self.decay_exp, VL)
            key_pre = vload(self.key, 0)
            key_post = vload(self.key, VL)
            query_pre = vload(self.query, 0)
            query_post = vload(self.query, VL)
            for dv in dsl_range(0, self.row_block, 1, unroll=self.state_update_unroll):
                offset = dv * TILE
                delta = vload_broadcast(self.delta_row, dv)
                state_pre = vload(self.state, offset)
                update_pre = vmul(delta, key_pre, mask=mask)
                state_pre = vmul(state_pre, decay_pre, mask=mask)
                state_pre = vadd(state_pre, update_pre, mask=mask)
                state_post = vload(self.state, offset + VL)
                update_post = vmul(delta, key_post, mask=mask)
                state_post = vmul(state_post, decay_post, mask=mask)
                state_post = vadd(state_post, update_post, mask=mask)
                vstore(self.state, offset, state_pre, mask)
                vstore(self.state, offset + VL, state_post, mask)
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
    def publish_replay_u(self, replay_u_gm):
        """Publish the FP16 correction produced by delta computation."""
        mem_copy(replay_u_gm, self.replay_u.consume())


class OutputGateStage:
    """RMSNorm and sigmoid output gate for one complete value head."""

    def __init__(self):
        self.gamma_fp32 = Buffer(MemLoc.UB, (TILE,), dtypes.float32)
        self.gate_input = Channel(MemLoc.UB, (1, TILE), dtypes.bfloat16, depth=2)
        self.output = Channel(MemLoc.UB, (1, TILE), dtypes.bfloat16, depth=2)
        self.raw_fp32 = Buffer(MemLoc.UB, (1, TILE), dtypes.float32)
        self.local_sum = Buffer(MemLoc.UB, (1, 8), dtypes.float32)

    @jit
    def load_gamma(self, output_norm_weight_gm: Tensor):
        mem_copy(self.gamma_fp32, output_norm_weight_gm)

    def issue_gate(self, gate_gm):
        mem_copy(self.gate_input.produce(), gate_gm)

    @jit
    def prepare_local_sum(self, recurrent_output):
        """Keep the complete recurrent output in FP32 and reduce its RMS sum."""
        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            eight, _ = update_mask(8, elem_bits=32)
            zero = vdups(0.0, dtypes.float32)
            vstore(self.local_sum, 0, zero, eight)
            raw_pre = vload(recurrent_output, 0)
            vstore(self.raw_fp32, 0, raw_pre, full)
            square_pre = vmul(raw_pre, raw_pre, mask=full)
            sum_pre = vreduce_sum(square_pre, mask=full)
            vstore_first(self.local_sum, 0, sum_pre)
            raw_post = vload(recurrent_output, VL)
            vstore(self.raw_fp32, VL, raw_post, full)
            square_post = vmul(raw_post, raw_post, mask=full)
            sum_post = vreduce_sum(square_post, mask=full)
            vstore_first(self.local_sum, 1, sum_post)

    @jit
    def apply_and_store(self, output_gm, rms_norm_eps: float):
        """Apply FP32 RMSNorm and sigmoid, then publish final BF16."""
        gate_input_slot = self.gate_input.consume()
        output_slot = self.output.produce()
        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            one = vdups(1.0, dtypes.float32)
            sum_pre = vload_broadcast(self.local_sum, 0)
            sum_post = vload_broadcast(self.local_sum, 1)
            total_sum = vadd(sum_pre, sum_post, mask=full)
            mean_square = vmuls(total_sum, 1.0 / TILE, mask=full)
            mean_square = vadds(mean_square, rms_norm_eps, mask=full)
            root_mean_square = vsqrt(mean_square, mask=full)
            inverse_rms = vdiv(one, root_mean_square, mask=full)
            for segment in range(D_SEGMENTS):
                offset = segment * VL
                raw = vload(self.raw_fp32, offset)
                gamma = vload(self.gamma_fp32, offset)
                normalized = vmul(raw, inverse_rms, mask=full)
                normalized = vmul(normalized, gamma, mask=full)
                gate_bf16 = vload_unpack(
                    gate_input_slot, offset, unpack_mode=UnpackMode.B16_TO_B32
                )
                gate = vcast(gate_bf16, dtypes.float32, mask=full)
                activated_gate = vneg(gate, mask=full)
                activated_gate = vexp(activated_gate, mask=full)
                activated_gate = vadds(activated_gate, 1.0, mask=full)
                activated_gate = vdiv(one, activated_gate, mask=full)
                final_output = vmul(normalized, activated_gate, mask=full)
                final_bf16 = vcast(final_output, dtypes.bfloat16, mask=full)
                vstore_pack(
                    output_slot, offset, final_bf16, full, pack_mode=PackMode.B32_TO_B16
                )
        mem_copy(output_gm, output_slot)


@jit
def prefetch_recurrent_item(
    recurrent_stage,
    output_gate_stage,
    q_workspace_gm,
    k_workspace_gm,
    v_workspace_gm,
    decay_workspace_gm,
    output_gate_workspace_gm,
    beta_workspace_gm,
    item,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    row_block: int,
):
    """Issue token-0 inputs for one complete batch/head item."""
    batch_index, value_head = idx2crd(item, [batch_size, num_heads])
    base = batch_index * seq_len
    recurrent_stage.prefetch_inputs(
        tile_slice(k_workspace_gm, (1, TILE), (base, value_head)),
        tile_slice(q_workspace_gm, (1, TILE), (base, value_head)),
        tile_slice(decay_workspace_gm, (1, TILE), (base, value_head)),
        tile_slice(v_workspace_gm, (1, row_block), (base, value_head)),
        tile_slice(beta_workspace_gm, (1, 1), (base, value_head)),
    )
    output_gate_stage.issue_gate(
        tile_slice(output_gate_workspace_gm, (1, row_block), (base, value_head))
    )


@jit
def prefetch_recurrent_state(
    recurrent_stage,
    state_gm,
    item,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    row_block: int,
):
    """Issue the initial-state load for one complete batch/head item."""
    batch_index, value_head = idx2crd(item, [batch_size, num_heads])
    recurrent_stage.load_state(
        tile_slice(
            state_gm[batch_index, value_head, None, None],
            (row_block, TILE),
            (0, 0),
        )
    )


@jit
def run_recurrent_wave(
    recurrent_stage,
    output_gate_stage,
    q_workspace_gm,
    k_workspace_gm,
    v_workspace_gm,
    decay_workspace_gm,
    output_gate_workspace_gm,
    beta_workspace_gm,
    state_gm,
    replay_u_gm,
    replay_k_gm,
    replay_decay_gm,
    item_start,
    item_end,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    row_block: int,
    scale_value: float,
    lower_bound: float,
    rms_norm_eps: float,
    first_inputs_prefetched: bool = False,
    first_state_prefetched: bool = False,
    next_item_stride: int = 0,
):
    """Run a wave, optionally carrying token-0 inputs across item boundaries."""
    for item in range(item_start, item_end):
        batch_index, value_head = idx2crd(item, [batch_size, num_heads])
        base = batch_index * seq_len
        if not const_expr(first_inputs_prefetched):
            prefetch_recurrent_item(
                recurrent_stage,
                output_gate_stage,
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                decay_workspace_gm,
                output_gate_workspace_gm,
                beta_workspace_gm,
                item,
                batch_size,
                seq_len,
                num_heads,
                row_block,
            )
        if not const_expr(first_state_prefetched):
            recurrent_stage.load_state(
                tile_slice(
                    state_gm[batch_index, value_head, None, None],
                    (row_block, TILE),
                    (0, 0),
                )
            )
        for token in range(seq_len):
            recurrent_stage.normalize_qk(scale_value)
            recurrent_stage.publish_replay_k(
                tile_slice(
                    replay_k_gm[batch_index, token, None, None],
                    (1, TILE),
                    (value_head, 0),
                )
            )
            recurrent_stage.activate_g_beta(lower_bound, value_head)
            recurrent_stage.publish_replay_decay(
                tile_slice(
                    replay_decay_gm[batch_index, token, None, None],
                    (1, TILE),
                    (value_head, 0),
                )
            )
            if token + 1 < seq_len:
                next_row = base + token + 1
                recurrent_stage.prefetch_inputs(
                    tile_slice(k_workspace_gm, (1, TILE), (next_row, value_head)),
                    tile_slice(q_workspace_gm, (1, TILE), (next_row, value_head)),
                    tile_slice(decay_workspace_gm, (1, TILE), (next_row, value_head)),
                    tile_slice(v_workspace_gm, (1, row_block), (next_row, value_head)),
                    tile_slice(beta_workspace_gm, (1, 1), (next_row, value_head)),
                )
                output_gate_stage.issue_gate(
                    tile_slice(
                        output_gate_workspace_gm, (1, row_block), (next_row, value_head)
                    )
                )
            recurrent_stage.compute_delta()
            recurrent_stage.publish_replay_u(
                tile_slice(
                    replay_u_gm[batch_index, token, None, None],
                    (1, row_block),
                    (value_head, 0),
                )
            )
            recurrent_stage.update_state()
            output_gate_stage.prepare_local_sum(recurrent_stage.out)
            output_gate_stage.apply_and_store(
                tile_slice(q_workspace_gm, (1, row_block), (base + token, value_head)),
                rms_norm_eps,
            )
            if const_expr(next_item_stride > 0):
                if token + 1 == seq_len:
                    next_item = item + next_item_stride
                    if next_item < batch_size * num_heads:
                        # All depth-2 input channels have released the current
                        # token, so fill their alternate slots while snapshot,
                        # ACK, and Oproj control work is still outstanding.
                        prefetch_recurrent_item(
                            recurrent_stage,
                            output_gate_stage,
                            q_workspace_gm,
                            k_workspace_gm,
                            v_workspace_gm,
                            decay_workspace_gm,
                            output_gate_workspace_gm,
                            beta_workspace_gm,
                            next_item,
                            batch_size,
                            seq_len,
                            num_heads,
                            row_block,
                        )
                        prefetch_recurrent_state(
                            recurrent_stage,
                            state_gm,
                            next_item,
                            batch_size,
                            seq_len,
                            num_heads,
                            row_block,
                        )


class OutputProjectionStage:
    """Shared output projection with compile-time tile mapping and buffer sizes."""

    def __init__(
        self,
        tile_width: int = 128,
        row_count: int = TILE,
        input_l2_cache_ctl: int = 0,
        weight_l2_cache_ctl: int = 0,
        store_l2_cache_ctl: int = 0,
        tiles_per_core: int = 2,
        *,
        kc: int = OPROJ_KC,
        resident_input: bool = False,
        resident_weight: bool = False,
        tile_stride: int = 1,
        l0c_depth: int = 1,
        row_capacity: int = 64,
    ):
        self.tiles_per_core = int(tiles_per_core)
        self.tile_width = int(tile_width)
        self.row_count = row_count
        self.input_l2_cache_ctl = int(input_l2_cache_ctl)
        self.weight_l2_cache_ctl = int(weight_l2_cache_ctl)
        self.store_l2_cache_ctl = int(store_l2_cache_ctl)
        self.kc = int(kc)
        self.resident_input = resident_input
        self.resident_weight = resident_weight
        self.tile_stride = int(tile_stride)
        self.row_capacity = int(row_capacity)
        # N224/N256 use KS64 so two L0B slots remain within capacity.
        self.ks = OPROJ_KS
        if self.resident_weight:
            self.resident_weight_rows = self.tile_width * self.tiles_per_core
            self.b_l1 = Buffer(
                MemLoc.L1,
                (self.resident_weight_rows, 6 * TILE),
                dtypes.bfloat16,
                data_format="nz",
            )
        else:
            self.b_l1 = Channel(
                MemLoc.L1,
                (self.tile_width, self.kc),
                dtypes.bfloat16,
                depth=2,
                data_format="nz",
            )
        self.a_l1 = Channel(
            MemLoc.L1,
            (self.row_count, 768 if self.resident_input else self.kc),
            dtypes.bfloat16,
            depth=1 if self.resident_input else 2,
            capacity=(self.row_capacity, 768 if self.resident_input else self.kc),
        )
        self.l0a = Channel(
            MemLoc.L0A,
            (self.row_count, self.ks),
            dtypes.bfloat16,
            depth=2,
            capacity=(self.row_capacity, self.ks),
        )
        self.l0b = Channel(
            MemLoc.L0B, (self.tile_width, self.ks), dtypes.bfloat16, depth=2
        )
        self.l0c = Channel(
            MemLoc.L0C,
            (self.row_count, self.tile_width),
            dtypes.float32,
            depth=l0c_depth,
            capacity=(self.row_capacity, self.tile_width),
        )
        self.nd2nz = make_copy_engine(format_transform="nd2nz")
        self.fixpipe = make_copy_engine()

    @jit
    def prefetch_weight(self, weight_gm: Tensor):
        """Keep this AIC's complete output-width-by-K768 Weight-NZ shard in L1."""
        if const_expr(self.resident_weight):
            resident_shards = ceil_div(weight_gm.shape[0], self.resident_weight_rows)
            if get_block_idx() < resident_shards:
                mem_copy(
                    self.b_l1,
                    tile_slice(
                        weight_gm,
                        (self.resident_weight_rows, 6 * TILE),
                        (get_block_idx(), 0),
                    ),
                    l2_cache_ctl=self.weight_l2_cache_ctl,
                )

    @jit
    def project(
        self,
        out_gm: Tensor,
        gated_workspace_gm: Tensor,
        weight_gm: Tensor,
        k_tiles: int,
        row_tile: int,
    ):
        """Compute each AIC's assigned output tiles."""
        tile_count = ceil_div(out_gm.shape[1], self.tile_width)
        wave_start = row_tile * self.row_count
        wave_end = wave_start + self.row_count
        gated_wave = gated_workspace_gm[wave_start:wave_end, None]
        out_wave = out_gm[wave_start:wave_end, None]
        if const_expr(self.resident_input):
            resident_a = self.a_l1.produce()
            mem_copy(
                resident_a,
                tile_slice(gated_wave, (self.row_capacity, 768), (0, 0)),
                l2_cache_ctl=self.input_l2_cache_ctl,
                engine=self.nd2nz,
            )
        first_tile = get_block_idx() * self.tiles_per_core
        for local_tile in range(self.tiles_per_core):
            if const_expr(self.resident_input):
                output_tile = get_block_idx() * self.tiles_per_core + local_tile
            elif const_expr(self.tile_stride != 1):
                output_tile = get_block_idx() + local_tile * self.tile_stride
            else:
                output_tile = first_tile + local_tile
            if output_tile < tile_count:
                accumulator = self.l0c.produce()
                for kc_idx in range(k_tiles * TILE // self.kc):
                    if const_expr(self.resident_input):
                        a_panel = tile_slice(
                            resident_a, (self.row_capacity, self.kc), (0, kc_idx)
                        )
                    else:
                        a_panel = self.a_l1.produce()
                        mem_copy(
                            a_panel,
                            tile_slice(
                                gated_wave, (self.row_capacity, self.kc), (0, kc_idx)
                            ),
                            engine=self.nd2nz,
                            l2_cache_ctl=self.input_l2_cache_ctl,
                        )
                    if const_expr(self.resident_weight):
                        b_panel = tile_slice(
                            self.b_l1, (self.tile_width, self.kc), (local_tile, kc_idx)
                        )
                    else:
                        b_panel = self.b_l1.produce()
                        mem_copy(
                            b_panel,
                            tile_slice(
                                weight_gm,
                                (self.tile_width, self.kc),
                                (output_tile, kc_idx),
                            ),
                            l2_cache_ctl=self.weight_l2_cache_ctl,
                        )
                    for ks_idx in range(self.kc // self.ks):
                        l0a_slot = self.l0a.produce()
                        mem_copy(
                            l0a_slot,
                            tile_slice(
                                a_panel, (self.row_capacity, self.ks), (0, ks_idx)
                            ),
                        )
                        l0b_slot = self.l0b.produce()
                        mem_copy(
                            l0b_slot,
                            tile_slice(
                                b_panel, (self.tile_width, self.ks), (0, ks_idx)
                            ),
                        )
                        matmul(
                            accumulator,
                            l0a_slot,
                            l0b_slot,
                            init=kc_idx == 0 and ks_idx == 0,
                        )
                mem_copy(
                    tile_slice(
                        out_wave, (self.row_capacity, self.tile_width), (0, output_tile)
                    ),
                    accumulator,
                    l2_cache_ctl=self.store_l2_cache_ctl,
                    engine=self.fixpipe,
                )


class BetaZSharedStage:
    """Share one hidden KC load between the last gate head and Beta."""

    def __init__(self, rows, input_l2_cache_ctl=0, weight_l2_cache_ctl=0):
        self.input_l2_cache_ctl = int(input_l2_cache_ctl)
        self.weight_l2_cache_ctl = int(weight_l2_cache_ctl)
        self.rows = rows
        self.half = rows // 2
        self.a = Channel(
            MemLoc.L1, (rows, 256), dtypes.bfloat16, depth=2, capacity=(128, 256)
        )
        self.z = Channel(
            MemLoc.L1, (128, 256), dtypes.bfloat16, depth=2, data_format="nz"
        )
        self.la = Channel(
            MemLoc.L0A, (rows, 128), dtypes.bfloat16, depth=2, capacity=(128, 128)
        )
        self.lz = Channel(MemLoc.L0B, (128, 128), dtypes.bfloat16, depth=1).produce()
        self.lb = Channel(MemLoc.L0B, (16, 128), dtypes.bfloat16, depth=2)
        self.cz = Channel(
            MemLoc.L0C, (rows, 128), dtypes.float32, depth=1, capacity=(128, 128)
        ).produce()
        self.cb = Channel(
            MemLoc.L0C, (rows, 16), dtypes.float32, depth=1, capacity=(128, 16)
        ).produce()
        self.uz = Channel(
            MemLoc.UB,
            (self.half, 128),
            dtypes.float32,
            depth=1,
            kind=ChannelKind.CrossCore,
            capacity=(64, 128),
        ).produce()
        self.ub = Channel(
            MemLoc.UB,
            (self.half, 16),
            dtypes.float32,
            depth=1,
            kind=ChannelKind.CrossCore,
            capacity=(64, 16),
        ).produce()
        self.zbf = Buffer(
            MemLoc.UB, (self.half, 128), dtypes.bfloat16, capacity=(64, 128)
        )
        self.bbf = Buffer(
            MemLoc.UB, (self.half, 16), dtypes.bfloat16, capacity=(64, 16)
        )
        self.nd = make_copy_engine(format_transform="nd2nz")
        self.fix = make_copy_engine(split_axis=0)

    @jit
    def project(self, hidden_gm, z_weight_gm, beta_weight_gm, z_gm, beta_gm, subblock):
        beta_l1 = Buffer(
            MemLoc.L1,
            dtype=beta_weight_gm.dtype,
            capacity=(BETA_TILE, HIDDEN_SIZE),
            layout=beta_weight_gm.layout,
            physical_layout=beta_weight_gm.physical_layout,
            data_format="nz",
        )
        mem_copy(beta_l1, beta_weight_gm, l2_cache_ctl=self.weight_l2_cache_ctl)
        for kc in range(HIDDEN_SIZE // PROJECTION_KC):
            a_slot = self.a.produce()
            mem_copy(
                a_slot,
                tile_slice(hidden_gm, (128, 256), (0, kc)),
                engine=self.nd,
                l2_cache_ctl=self.input_l2_cache_ctl,
            )
            z_slot = self.z.produce()
            mem_copy(
                z_slot,
                tile_slice(z_weight_gm, (128, 256), (5, kc)),
                l2_cache_ctl=self.weight_l2_cache_ctl,
            )
            for ks in range(2):
                la_slot = self.la.produce()
                mem_copy(la_slot, tile_slice(a_slot, (128, 128), (0, ks)))
                mem_copy(self.lz, tile_slice(z_slot, (128, 128), (0, ks)))
                matmul(self.cz, la_slot, self.lz, init=kc == 0 and ks == 0)
                lb_slot = self.lb.produce()
                mem_copy(lb_slot, tile_slice(beta_l1, (16, 128), (0, kc * 2 + ks)))
                matmul(self.cb, la_slot, lb_slot, init=kc == 0 and ks == 0)
        mem_copy(self.uz, reinterpret(self.cz, shape=(self.rows, 128)), engine=self.fix)
        mem_copy(self.ub, reinterpret(self.cb, shape=(self.rows, 16)), engine=self.fix)
        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            for segment in range(self.half * 128 // VL):
                offset = segment * VL
                value = vcast(vload(self.uz, offset), dtypes.bfloat16, mask=full)
                vstore_pack(
                    self.zbf, offset, value, full, pack_mode=PackMode.B32_TO_B16
                )
            for segment in range(self.half * 16 // VL):
                offset = segment * VL
                value = vcast(vload(self.ub, offset), dtypes.bfloat16, mask=full)
                vstore_pack(
                    self.bbf, offset, value, full, pack_mode=PackMode.B32_TO_B16
                )
        vec_sync_all()
        mem_copy(
            tile_slice(z_gm, (128, 128), (0, 0)),
            self.zbf,
            engine=self.fix,
            part_id=subblock,
            actual=(self.rows, 128),
        )
        mem_copy(
            tile_slice(beta_gm, (128, 16), (0, 0)),
            self.bbf,
            engine=self.fix,
            part_id=subblock,
            actual=(self.rows, 16),
        )
        vec_sync_all()
        cube_sync_all()


class _GridSchedule:
    """Compile-time branch configuration with one shared KDA wave schedule."""

    def __init__(self, block_num: int, hidden_size: int, num_heads: int):
        # Five grid schedules are embedded in one kernel. These constants only
        # describe branch-local tiles; runtime get_block_num() selects a branch.
        self.block_num = block_num
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.batch_tile = BATCH_TILE
        self.qkv_block_count = 3 * num_heads
        self.output_gate_block_count = num_heads
        self.beta_block = self.qkv_block_count + num_heads
        self.low_rank_block = self.beta_block + 1
        self.recurrent_wave_count = 2
        self.output_tiles_per_core = {8: 7, 16: 2, 24: 2, 28: 1, 32: 1}[block_num]
        self.output_tile_width = {8: 128, 16: 256, 24: 256, 28: 256, 32: 256}[block_num]
        # Keep output weights resident when the shard and activation buffers fit.
        activation_channel_bytes = 2 * 64 * OPROJ_KC * 2
        self.output_weight_resident = (
            self.output_tile_width * self.output_tiles_per_core * num_heads * TILE * 2
            + activation_channel_bytes
            <= cannbotdsl.get_mem_size("l1")
        )
        self.projection_input_l2_cache_ctl = 0
        self.projection_weight_l2_cache_ctl = 1
        # These values are specialized by each schedule branch before use.
        self.batch_groups: object
        self.seq_len: object
        self.row_count: object
        self.projection_row_block: object
        self.recurrent_wave_rows: object

    @jit
    def _run_qkvg_task(
        self,
        projection_stage: ProjectionStage,
        conv_stage: ConvSiluStage,
        hidden_gm: Tensor,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        qkv_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        conv_state_gm: Tensor,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        projection_kind,
        projection_head,
        batch_start,
        batch_count,
        subblock_idx,
        k_tiles,
    ):
        """Run one complete QKV+Conv or gate projection task."""
        if projection_kind == 0:
            projection_stage.matmul_hidden(
                hidden_gm, qkv_weight_gm, projection_head, k_tiles
            )
            batches_per_subblock = batch_count // 2
            selected_history = Buffer(
                MemLoc.UB,
                (batches_per_subblock * (CONV_KERNEL - 1), TILE),
                dtypes.bfloat16,
            )
            for candidate_head in range(3 * self.num_heads):
                if projection_head == candidate_head:
                    projection = candidate_head // self.num_heads
                    value_head = candidate_head - projection * self.num_heads
                    conv_stage.load_weight(
                        tile_slice(
                            conv_weight_gm[projection, value_head, None, None],
                            (CONV_KERNEL, TILE),
                            (0, 0),
                        )
                    )
                    for local_batch in range(batches_per_subblock):
                        batch_index = (
                            batch_start
                            + subblock_idx * batches_per_subblock
                            + local_batch
                        )
                        cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                        conv_accepted = dtypes.int64(conv_accepted_gm[batch_index])
                        history_offset = conv_accepted - 1 if conv_accepted > 0 else 0
                        if cache_index != 0:
                            state_view = conv_state_gm[cache_index, None, None]
                            for history_row in range(CONV_KERNEL - 1):
                                mem_copy(
                                    tile_slice(
                                        selected_history,
                                        (1, TILE),
                                        (
                                            local_batch * (CONV_KERNEL - 1)
                                            + history_row,
                                            0,
                                        ),
                                    ),
                                    tile_slice(
                                        state_view,
                                        (1, TILE),
                                        (history_offset + history_row, candidate_head),
                                    ),
                                )
            projection_stage.acquire_result()

            for candidate_head in range(3 * self.num_heads):
                if projection_head == candidate_head:
                    for local_batch in range(batches_per_subblock):
                        batch_index = (
                            batch_start
                            + subblock_idx * batches_per_subblock
                            + local_batch
                        )
                        cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                        if cache_index == 0:
                            conv_stage.zero_sequence(local_batch)
                        else:
                            history_view = tile_slice(
                                selected_history,
                                (CONV_KERNEL - 1, TILE),
                                (local_batch, 0),
                            )
                            conv_stage.install_history(history_view)
                            state_view = conv_state_gm[cache_index, None, None]
                            for cache_row in range(2):
                                mem_copy(
                                    tile_slice(
                                        state_view,
                                        (1, TILE),
                                        (cache_row, candidate_head),
                                    ),
                                    tile_slice(
                                        history_view, (1, TILE), (cache_row + 1, 0)
                                    ),
                                )
                            conv_stage.convolve_sequence(
                                local_batch,
                                projection_stage.projection_fp32,
                                tile_slice(
                                    conv_stage.weight, (CONV_KERNEL, TILE), (0, 0)
                                ),
                            )
                    vec_sync_notify(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                    vec_sync_wait(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                    for local_batch in range(batches_per_subblock):
                        batch_index = (
                            batch_start
                            + subblock_idx * batches_per_subblock
                            + local_batch
                        )
                        cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                        if cache_index != 0:
                            state_view = conv_state_gm[cache_index, None, None]
                            for cache_token in range(self.seq_len):
                                mem_copy(
                                    tile_slice(
                                        state_view,
                                        (1, TILE),
                                        (cache_token + 2, candidate_head),
                                    ),
                                    tile_slice(
                                        conv_stage.raw_bf16,
                                        (1, TILE),
                                        (local_batch * self.seq_len + cache_token, 0),
                                    ),
                                )
            vec_sync_all()
            if projection_head < self.num_heads:
                projection_stage.publish_source(
                    tile_slice(
                        q_workspace_gm,
                        (projection_stage.row_capacity, TILE),
                        (0, projection_head),
                    ),
                    conv_stage.output_bf16,
                    subblock_idx,
                )
            elif projection_head < 2 * self.num_heads:
                projection_stage.publish_source(
                    tile_slice(
                        k_workspace_gm,
                        (projection_stage.row_capacity, TILE),
                        (0, projection_head - self.num_heads),
                    ),
                    conv_stage.output_bf16,
                    subblock_idx,
                )
            else:
                projection_stage.publish_source(
                    tile_slice(
                        v_workspace_gm,
                        (projection_stage.row_capacity, TILE),
                        (0, projection_head - 2 * self.num_heads),
                    ),
                    conv_stage.output_bf16,
                    subblock_idx,
                )
        else:
            projection_stage.project(
                hidden_gm, output_gate_weight_gm, projection_head, k_tiles
            )
            projection_stage.publish(
                tile_slice(
                    output_gate_workspace_gm,
                    (projection_stage.row_capacity, TILE),
                    (0, projection_head),
                ),
                subblock_idx,
            )
        # A following task can reuse Conv/output BF16 staging immediately.
        # Finish this AIV's MTE3 reads before its next Vector writes.
        vec_sync_all()

    @jit
    def _run_frontend_main_task(
        self,
        projection_stage: ProjectionStage,
        conv_stage: ConvSiluStage,
        low_rank_stage: LowRankDecayStage,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        conv_state_gm: Tensor,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        projection_kind,
        projection_head,
        batch_start,
        batch_count,
        subblock_idx,
        k_tiles,
    ):
        rows = projection_stage.row_capacity
        row_tile = batch_start // batch_count
        hidden_tile = make_tiler((rows, self.hidden_size), alignment=(16, 16))
        qkv_tile = make_tiler((rows, self.num_heads * TILE), alignment=(16, 16))
        if projection_kind < 2:
            self._run_qkvg_task(
                projection_stage,
                conv_stage,
                tile_slice(hidden_gm, hidden_tile, (row_tile, 0)),
                tile_slice(q_workspace_gm, qkv_tile, (row_tile, 0)),
                tile_slice(k_workspace_gm, qkv_tile, (row_tile, 0)),
                tile_slice(v_workspace_gm, qkv_tile, (row_tile, 0)),
                tile_slice(output_gate_workspace_gm, qkv_tile, (row_tile, 0)),
                qkv_weight_gm,
                output_gate_weight_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
                projection_kind,
                projection_head,
                batch_start,
                batch_count,
                subblock_idx,
                k_tiles,
            )
        else:
            low_rank_stage.project(
                tile_slice(decay_workspace_gm, qkv_tile, (row_tile, 0)),
                tile_slice(hidden_gm, hidden_tile, (row_tile, 0)),
                fa_weight_gm,
                fb_weight_gm,
                k_tiles,
                subblock_idx,
            )

    @jit
    def _prefetch_n384_panel(
        self,
        conv_stage: ConvSiluStage,
        selected_history,
        conv_state_gm: Tensor,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        n_macro,
        batch_base,
        batches_per_subblock,
        panel,
    ):
        """Fetch one panel's Conv metadata while its projection is in flight."""
        logical_n = n_macro * 3 + panel
        if logical_n < 3 * self.num_heads:
            projection = logical_n // self.num_heads
            value_head = logical_n - projection * self.num_heads
            mem_copy(
                tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (panel, 0)),
                tile_slice(
                    conv_weight_gm[projection, value_head, None, None],
                    (CONV_KERNEL, TILE),
                    (0, 0),
                ),
            )
            for local_batch in range(batches_per_subblock):
                batch_index = batch_base + local_batch
                cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                conv_accepted = dtypes.int64(conv_accepted_gm[batch_index])
                history_offset = conv_accepted - 1 if conv_accepted > 0 else 0
                if cache_index != 0:
                    state_view = conv_state_gm[cache_index, None, None]
                    # fmt: off
                    mem_copy(
                        tile_slice(
                            selected_history, (CONV_KERNEL - 1, TILE), (local_batch, 0)
                        ),
                        tile_slice(
                            state_view[
                                None,
                                history_offset:history_offset + CONV_KERNEL - 1,
                                None,
                            ],
                            (CONV_KERNEL - 1, TILE),
                            (0, logical_n),
                        ),
                    )
                    # fmt: on

    @jit
    def _consume_n384_panel(
        self,
        projection_stage: N384ProjectionStage,
        conv_stage: ConvSiluStage,
        selected_history,
        weight_ub,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        conv_state_gm: Tensor,
        conv_state_indices_gm: Tensor,
        m_group,
        n_macro,
        batch_base,
        batches_per_subblock,
        subblock_idx,
        panel,
    ):
        """Consume one resident N128 accumulator while Cube holds the other two."""
        logical_n = n_macro * 3 + panel
        if logical_n < 3 * self.num_heads:
            projection = logical_n // self.num_heads
            value_head = logical_n - projection * self.num_heads
            for local_batch in range(batches_per_subblock):
                batch_index = batch_base + local_batch
                cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                if cache_index == 0:
                    conv_stage.zero_sequence(local_batch)
                else:
                    history_view = tile_slice(
                        selected_history, (CONV_KERNEL - 1, TILE), (local_batch, 0)
                    )
                    conv_stage.install_history(history_view)
                    state_view = conv_state_gm[cache_index, None, None]
                    mem_copy(
                        tile_slice(
                            state_view[None, 0:2, None], (2, TILE), (0, logical_n)
                        ),
                        tile_slice(history_view[1:3, None], (2, TILE), (0, 0)),
                    )
                    conv_stage.convolve_sequence(
                        local_batch, projection_stage.projection_fp32, weight_ub
                    )
                    vec_sync_notify(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                    vec_sync_wait(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                    # Publish the eight accepted tokens with one strided 2-D
                    # MTE3 command instead of eight single-row commands.
                    # fmt: off
                    mem_copy(
                        tile_slice(
                            state_view[None, 2:2 + self.seq_len, None],
                            (self.seq_len, TILE),
                            (0, logical_n),
                        ),
                        tile_slice(
                            conv_stage.raw_bf16, (self.seq_len, TILE), (local_batch, 0)
                        ),
                    )
                    # fmt: on
            vec_sync_all()
            if projection == 0:
                projection_stage.publish(
                    q_workspace_gm,
                    m_group,
                    value_head,
                    subblock_idx,
                    conv_stage.output_bf16,
                )
            elif projection == 1:
                projection_stage.publish(
                    k_workspace_gm,
                    m_group,
                    value_head,
                    subblock_idx,
                    conv_stage.output_bf16,
                )
            else:
                projection_stage.publish(
                    v_workspace_gm,
                    m_group,
                    value_head,
                    subblock_idx,
                    conv_stage.output_bf16,
                )
        else:
            gate_head = logical_n - 3 * self.num_heads
            projection_stage.cast_result()
            vec_sync_all()
            projection_stage.publish(
                output_gate_workspace_gm,
                m_group,
                gate_head,
                subblock_idx,
                projection_stage.projection_bf16,
            )
        vec_sync_all()

    @jit
    def _consume_n384_triplet(
        self,
        projection_stage: N384ProjectionStage,
        conv_stage: ConvSiluStage,
        selected_history,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        conv_state_gm: Tensor,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        m_group,
        n_macro,
        batch_base,
        batches_per_subblock,
        subblock_idx,
    ):
        history0 = tile_slice(
            selected_history, (batches_per_subblock * (CONV_KERNEL - 1), TILE), (0, 0)
        )
        history1 = tile_slice(
            selected_history, (batches_per_subblock * (CONV_KERNEL - 1), TILE), (1, 0)
        )
        history2 = tile_slice(
            selected_history, (batches_per_subblock * (CONV_KERNEL - 1), TILE), (2, 0)
        )
        weight0 = tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (0, 0))
        weight1 = tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (1, 0))
        weight2 = tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (2, 0))
        self._prefetch_n384_panel(
            conv_stage,
            history0,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            n_macro,
            batch_base,
            batches_per_subblock,
            0,
        )
        self._prefetch_n384_panel(
            conv_stage,
            history1,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            n_macro,
            batch_base,
            batches_per_subblock,
            1,
        )
        self._prefetch_n384_panel(
            conv_stage,
            history2,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            n_macro,
            batch_base,
            batches_per_subblock,
            2,
        )
        # All metadata DMA runs while AIV waits for the long N384 Cube body.
        vec_sync_all()
        projection_stage.acquire_result0()
        self._consume_n384_panel(
            projection_stage,
            conv_stage,
            history0,
            weight0,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            output_gate_workspace_gm,
            conv_state_gm,
            conv_state_indices_gm,
            m_group,
            n_macro,
            batch_base,
            batches_per_subblock,
            subblock_idx,
            0,
        )
        projection_stage.acquire_result1()
        self._consume_n384_panel(
            projection_stage,
            conv_stage,
            history1,
            weight1,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            output_gate_workspace_gm,
            conv_state_gm,
            conv_state_indices_gm,
            m_group,
            n_macro,
            batch_base,
            batches_per_subblock,
            subblock_idx,
            1,
        )
        projection_stage.acquire_result2()
        self._consume_n384_panel(
            projection_stage,
            conv_stage,
            history2,
            weight2,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            output_gate_workspace_gm,
            conv_state_gm,
            conv_state_indices_gm,
            m_group,
            n_macro,
            batch_base,
            batches_per_subblock,
            subblock_idx,
            2,
        )

    @jit
    def _frontend_adaptive_n384(
        self,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        conv_state_gm: Tensor,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
    ):
        """Balance every group's eight M128xN384 tasks across the launch grid."""
        block_idx = get_block_idx()
        block_num = self.block_num
        subblock_idx = get_subblock_id()
        k_tiles = ceil_div(hidden_gm.shape[1], TILE)
        task_count = self.batch_groups * 8
        batches_per_subblock = BATCH_TILE // 2

        # N-major order preserves the proven B64 mapping. The runtime grid
        # stride gives both 28- and 32-AIC launches a balanced task count.
        for task_idx in range(block_idx, task_count, block_num):
            n_macro = task_idx // self.batch_groups
            m_group = task_idx - n_macro * self.batch_groups
            batch_base = m_group * BATCH_TILE + subblock_idx * batches_per_subblock
            channel_rewind(reset_sync_id=True)
            if n_macro == 7 and task_count <= block_num:
                projection_stage = GateDecayProjectionStage(
                    self.projection_input_l2_cache_ctl,
                    self.projection_weight_l2_cache_ctl,
                )
                projection_stage.matmul_hidden3_fa(
                    hidden_gm, output_gate_weight_gm, fa_weight_gm, m_group, k_tiles
                )
                projection_stage.publish_gate(
                    output_gate_workspace_gm, m_group, subblock_idx
                )
                projection_stage.project_fb(
                    decay_workspace_gm, fb_weight_gm, m_group, subblock_idx
                )
            else:
                projection_stage = N384ProjectionStage(
                    self.projection_input_l2_cache_ctl,
                    self.projection_weight_l2_cache_ctl,
                )
                conv_stage = ConvSiluStage(self.seq_len, TILE // 2)
                selected_history = Buffer(
                    MemLoc.UB,
                    (3 * batches_per_subblock * (CONV_KERNEL - 1), TILE),
                    dtypes.bfloat16,
                )
                if n_macro < 6:
                    projection_stage.matmul_hidden3(
                        hidden_gm, qkv_weight_gm, m_group, n_macro, k_tiles
                    )
                else:
                    projection_stage.matmul_hidden3(
                        hidden_gm, output_gate_weight_gm, m_group, n_macro - 6, k_tiles
                    )
                self._consume_n384_triplet(
                    projection_stage,
                    conv_stage,
                    selected_history,
                    q_workspace_gm,
                    k_workspace_gm,
                    v_workspace_gm,
                    output_gate_workspace_gm,
                    conv_state_gm,
                    conv_state_indices_gm,
                    conv_accepted_gm,
                    conv_weight_gm,
                    m_group,
                    n_macro,
                    batch_base,
                    batches_per_subblock,
                    subblock_idx,
                )

    @jit
    def _frontend_adaptive_auxiliary(
        self,
        decay_workspace_gm: Tensor,
        beta_workspace_gm: Tensor,
        hidden_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        beta_weight_gm: Tensor,
    ):
        """Place Beta and multi-wave decay tasks on the lightest AICs."""
        block_idx = get_block_idx()
        block_num = self.block_num
        subblock_idx = get_subblock_id()
        task_count = self.batch_groups * 8
        extra_task_cores = task_count % block_num
        aux_start = block_num - 8 if extra_task_cores == 0 else extra_task_cores
        aux_slot = (block_idx - aux_start + block_num) % block_num
        auxiliary_count = (
            self.batch_groups if task_count <= block_num else 2 * self.batch_groups
        )
        k_tiles = ceil_div(hidden_gm.shape[1], TILE)
        for auxiliary_idx in range(aux_slot, auxiliary_count, block_num):
            channel_rewind(reset_sync_id=True)
            if auxiliary_idx < self.batch_groups:
                m_group = auxiliary_idx
                beta_stage = ProjectionStage(
                    row_count=TILE,
                    row_capacity=TILE,
                    tile_width=BETA_TILE,
                    input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
                    weight_l2_cache_ctl=self.projection_weight_l2_cache_ctl,
                )
                beta_stage.project(
                    tile_slice(hidden_gm, (TILE, self.hidden_size), (m_group, 0)),
                    beta_weight_gm,
                    0,
                    k_tiles,
                )
                beta_stage.publish(
                    tile_slice(beta_workspace_gm, (TILE, BETA_TILE), (m_group, 0)),
                    subblock_idx,
                )
                vec_sync_all()
            else:
                m_group = auxiliary_idx - self.batch_groups
                projection_stage = ProjectionStage(
                    row_count=TILE,
                    row_capacity=TILE,
                    input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
                    weight_l2_cache_ctl=self.projection_weight_l2_cache_ctl,
                )
                low_rank_stage = LowRankDecayStage(projection_stage, self.num_heads)
                low_rank_stage.project(
                    tile_slice(
                        decay_workspace_gm, (TILE, self.num_heads * TILE), (m_group, 0)
                    ),
                    tile_slice(hidden_gm, (TILE, self.hidden_size), (m_group, 0)),
                    fa_weight_gm,
                    fb_weight_gm,
                    k_tiles,
                    subblock_idx,
                )

    @jit
    def _frontend_tail(
        self,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        beta_workspace_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        beta_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        replay_decay_gm: Tensor,
        conv_state_gm: Tensor,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
    ):
        """Exact host-model task order; cores progress without a grid barrier."""
        self.seq_len = SEQ_LEN
        self.row_count = 128
        self.projection_row_block = 64
        block_idx = get_block_idx()
        subblock_idx = get_subblock_id()
        groups = replay_decay_gm.shape[0] // 16
        k_tiles = ceil_div(hidden_gm.shape[1], TILE)
        frontend_weight_l2_cache_ctl = 1
        full_qkv_count = 32 if groups == 2 else groups * 18
        full_gate_count = 0 if groups == 2 else 7 if groups == 3 else 20
        full_stage = ProjectionStage(
            row_count=128,
            row_capacity=128,
            input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
            weight_l2_cache_ctl=frontend_weight_l2_cache_ctl,
        )
        full_conv = ConvSiluStage(8, 64, row_capacity=64)
        full_decay = LowRankDecayStage(full_stage, self.num_heads)
        for local_round in range(groups - 1):
            ordinal = local_round * 32 + block_idx
            projection_kind = 0
            logical = ordinal
            if ordinal >= full_qkv_count:
                projection_kind = 1
                logical = ordinal - full_qkv_count
                if logical >= full_gate_count:
                    projection_kind = 2
                    logical = logical - full_gate_count
            projection_head = logical // groups
            group = logical % groups
            self._run_frontend_main_task(
                full_stage,
                full_conv,
                full_decay,
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                decay_workspace_gm,
                output_gate_workspace_gm,
                hidden_gm,
                qkv_weight_gm,
                fa_weight_gm,
                fb_weight_gm,
                output_gate_weight_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
                projection_kind,
                projection_head,
                group * 16,
                16,
                subblock_idx,
                k_tiles,
            )

        # M128 consumes the L0C arena. Drain local pipelines before reusing
        # its addresses/IDs for M64; LowerChannel drains the paired-AIV
        # CrossCore lifetime. These are per-core operations, not grid barriers.
        cube_sync_all()
        vec_sync_all()
        channel_rewind(reset_sync_id=True)
        if groups == 2 or (groups == 3 and block_idx < 12):
            half_stage = ProjectionStage(
                row_count=64,
                row_capacity=64,
                input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
                weight_l2_cache_ctl=frontend_weight_l2_cache_ctl,
            )
            half_conv = ConvSiluStage(8, 32, row_capacity=32)
            half_decay = LowRankDecayStage(half_stage, self.num_heads)
            projection_kind = 1
            logical = block_idx // 2 + 7
            if groups == 2:
                if block_idx < 8:
                    projection_kind = 0
                    logical = block_idx // 2 + 32
                elif block_idx < 28:
                    logical = (block_idx - 8) // 2
                else:
                    projection_kind = 2
                    logical = (block_idx - 28) // 2
            projection_head = logical // groups
            group = logical % groups
            batch_start = group * 16 + (block_idx % 2) * 8
            self._run_frontend_main_task(
                half_stage,
                half_conv,
                half_decay,
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                decay_workspace_gm,
                output_gate_workspace_gm,
                hidden_gm,
                qkv_weight_gm,
                fa_weight_gm,
                fb_weight_gm,
                output_gate_weight_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
                projection_kind,
                projection_head,
                batch_start,
                8,
                subblock_idx,
                k_tiles,
            )
            cube_sync_all()
            vec_sync_all()

        channel_rewind(reset_sync_id=True)
        quarter_core = 0 if groups == 2 else 12 if groups == 3 else 8
        quarter_count = 8 if groups == 2 else 20 if groups == 3 else 16
        if block_idx >= quarter_core and block_idx < quarter_core + quarter_count:
            quarter_stage = ProjectionStage(
                row_count=32,
                row_capacity=32,
                input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
                weight_l2_cache_ctl=frontend_weight_l2_cache_ctl,
            )
            piece = block_idx - quarter_core
            first_gate = 10 if groups == 2 else 13 if groups == 3 else 20
            logical = first_gate + piece // 4
            projection_head = logical // groups
            group = logical % groups
            row_tile = group * 4 + piece % 4
            quarter_stage.project(
                tile_slice(hidden_gm, (32, self.hidden_size), (row_tile, 0)),
                output_gate_weight_gm,
                projection_head,
                k_tiles,
            )
            quarter_stage.publish(
                tile_slice(
                    output_gate_workspace_gm, (32, TILE), (row_tile, projection_head)
                ),
                subblock_idx,
            )
            cube_sync_all()
            vec_sync_all()

        channel_rewind(reset_sync_id=True)
        beta_group = -1
        if groups == 2:
            if block_idx >= 8 and block_idx < 10:
                beta_group = block_idx - 8
        elif groups == 3:
            if block_idx >= 12 and block_idx <= 20 and block_idx % 4 == 0:
                beta_group = (block_idx - 12) // 4
        else:
            if block_idx >= 24 and block_idx < 28:
                beta_group = block_idx - 24
        if beta_group >= 0:
            beta_stage = ProjectionStage(
                row_count=128,
                row_capacity=128,
                tile_width=BETA_TILE,
                input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
                weight_l2_cache_ctl=frontend_weight_l2_cache_ctl,
            )
            beta_stage.project(
                tile_slice(hidden_gm, (128, self.hidden_size), (beta_group, 0)),
                beta_weight_gm,
                0,
                k_tiles,
            )
            beta_stage.publish(
                tile_slice(beta_workspace_gm, (128, BETA_TILE), (beta_group, 0)),
                subblock_idx,
            )

    @jit
    def _frontend_flat_m128(
        self,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        beta_workspace_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        beta_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        replay_decay_gm: Tensor,
        conv_state_gm: Tensor,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
    ):
        self.seq_len = SEQ_LEN
        self.row_count = 128
        self.projection_row_block = 64
        block_idx = get_block_idx()
        subblock_idx = get_subblock_id()
        block_num = self.block_num
        k_tiles = ceil_div(hidden_gm.shape[1], TILE)
        groups = replay_decay_gm.shape[0] // BATCH_TILE
        task_count = groups * 24
        round_count = ceil_div(task_count, block_num)
        hidden_tile = make_tiler((128, self.hidden_size), alignment=(16, 16))
        qkv_tile = make_tiler((128, self.num_heads * TILE), alignment=(16, 16))
        beta_tile = make_tiler((128, BETA_TILE), alignment=(16, 16))

        projection_stage = ProjectionStage(
            row_count=128,
            input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
            weight_l2_cache_ctl=self.projection_weight_l2_cache_ctl,
            row_capacity=128,
        )
        conv_stage = ConvSiluStage(self.seq_len, 64, row_capacity=64)
        for local_round in range(round_count):
            task = local_round * block_num + block_idx
            if task < task_count:
                group = task // 24
                projection = task - group * 24
                batch_start = group * 16
                self._run_qkvg_task(
                    projection_stage,
                    conv_stage,
                    tile_slice(hidden_gm, hidden_tile, (group, 0)),
                    tile_slice(q_workspace_gm, qkv_tile, (group, 0)),
                    tile_slice(k_workspace_gm, qkv_tile, (group, 0)),
                    tile_slice(v_workspace_gm, qkv_tile, (group, 0)),
                    tile_slice(output_gate_workspace_gm, qkv_tile, (group, 0)),
                    qkv_weight_gm,
                    output_gate_weight_gm,
                    conv_state_gm,
                    conv_state_indices_gm,
                    conv_accepted_gm,
                    conv_weight_gm,
                    projection_kind=(0 if projection < 18 else 1),
                    projection_head=(
                        projection if projection < 18 else projection - 18
                    ),
                    batch_start=batch_start,
                    batch_count=16,
                    subblock_idx=subblock_idx,
                    k_tiles=k_tiles,
                )

        cube_sync_all()
        channel_rewind(reset_sync_id=True)
        aux_kind = -1
        aux_group = 0
        tail_tasks = task_count % get_block_num()
        aux_start = 2 if tail_tasks == 0 else tail_tasks + 2
        beta_gap = groups if tail_tasks == 0 else 0
        beta_start = aux_start + groups + beta_gap
        if block_idx >= aux_start and block_idx < aux_start + groups:
            aux_kind = 0
            aux_group = block_idx - aux_start
        elif block_idx >= beta_start and block_idx < beta_start + groups:
            aux_kind = 1
            aux_group = block_idx - beta_start

        if aux_kind == 0:
            auxiliary_stage = ProjectionStage(
                row_count=128,
                row_capacity=128,
                input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
                weight_l2_cache_ctl=self.projection_weight_l2_cache_ctl,
            )
            low_rank_stage = LowRankDecayStage(auxiliary_stage, self.num_heads)
            low_rank_stage.project(
                tile_slice(decay_workspace_gm, qkv_tile, (aux_group, 0)),
                tile_slice(hidden_gm, hidden_tile, (aux_group, 0)),
                fa_weight_gm,
                fb_weight_gm,
                k_tiles,
                subblock_idx,
            )
        elif aux_kind == 1:
            beta_stage = ProjectionStage(
                row_count=128,
                row_capacity=128,
                tile_width=BETA_TILE,
                input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
                weight_l2_cache_ctl=self.projection_weight_l2_cache_ctl,
            )
            beta_stage.project(
                tile_slice(hidden_gm, hidden_tile, (aux_group, 0)),
                beta_weight_gm,
                0,
                k_tiles,
            )
            beta_stage.publish(
                tile_slice(beta_workspace_gm, beta_tile, (aux_group, 0)), subblock_idx
            )

    @jit
    def _frontend_24_single_group(
        self,
        q_workspace_gm,
        k_workspace_gm,
        v_workspace_gm,
        decay_workspace_gm,
        output_gate_workspace_gm,
        beta_workspace_gm,
        fa_workspace_gm,
        norm_exchange_gm,
        hidden_gm,
        qkv_weight_gm,
        fa_weight_gm,
        fb_weight_gm,
        beta_weight_gm,
        output_gate_weight_gm,
        conv_state_gm,
        conv_state_indices_gm,
        conv_accepted_gm,
        conv_weight_gm,
    ):
        block_idx = get_block_idx()
        subblock_idx = get_subblock_id()
        k_tiles = ceil_div(hidden_gm.shape[1], TILE)
        if block_idx < 18:
            vec_sync_block_arrive(PIPE.MTE3, 0, mode=0)
            vec_sync_block_arrive(PIPE.MTE3, 3, mode=0)
            vec_sync_block_arrive(PIPE.MTE3, 4, mode=0)
            projection_stage = ProjectionStage(
                self.row_count,
                self.projection_input_l2_cache_ctl,
                self.projection_weight_l2_cache_ctl,
            )
            conv_stage = ConvSiluStage(self.seq_len, self.projection_row_block)
            self._run_qkvg_task(
                projection_stage,
                conv_stage,
                hidden_gm,
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                output_gate_workspace_gm,
                qkv_weight_gm,
                output_gate_weight_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
                0,
                block_idx,
                0,
                BATCH_TILE,
                subblock_idx,
                k_tiles,
            )
            vec_sync_block_wait(PIPE.S, 0, mode=0)
            vec_sync_block_wait(PIPE.S, 3, mode=0)
            vec_sync_block_wait(PIPE.S, 4, mode=0)
        else:
            channel_rewind(reset_sync_id=True)
            if block_idx < 23:
                vec_sync_block_arrive(PIPE.MTE3, 0, mode=0)
                vec_sync_block_arrive(PIPE.MTE3, 4, mode=0)
                vec_sync_block_wait(PIPE.S, 0, mode=0)
                vec_sync_block_arrive(PIPE.MTE3, 1, mode=2)
                cube_sync_block_wait(PIPE.S, 1, mode=2)
                projection_stage = ProjectionStage(
                    self.row_count,
                    self.projection_input_l2_cache_ctl,
                    self.projection_weight_l2_cache_ctl,
                )
                first_kc = (
                    (block_idx - 18) * 6
                    if block_idx < 21
                    else 18 + (block_idx - 21) * 5
                )
                kc_count = 6 if block_idx < 21 else 5
                accumulator = projection_stage.l0c.produce()
                for local_kc in range(kc_count):
                    kc = first_kc + local_kc
                    a_slot = projection_stage.a_l1.produce()
                    b_slot = projection_stage.b_l1.produce()
                    mem_copy(
                        a_slot,
                        tile_slice(hidden_gm, (128, 256), (0, kc)),
                        engine=projection_stage.nd2nz,
                        l2_cache_ctl=self.projection_input_l2_cache_ctl,
                    )
                    mem_copy(
                        b_slot,
                        tile_slice(fa_weight_gm, (128, 256), (0, kc)),
                        l2_cache_ctl=self.projection_weight_l2_cache_ctl,
                    )
                    for ks in range(2):
                        a_input = projection_stage.l0a.produce()
                        b_input = projection_stage.l0b.produce()
                        mem_copy(a_input, tile_slice(a_slot, (128, 128), (0, ks)))
                        mem_copy(b_input, tile_slice(b_slot, (128, 128), (0, ks)))
                        matmul(
                            accumulator,
                            a_input,
                            b_input,
                            init=local_kc == 0 and ks == 0,
                        )
                # Advance the L0C read cursor before reusing this channel for gate/Fb.
                mem_copy(
                    norm_exchange_gm,
                    projection_stage.l0c.consume(),
                    engine=make_copy_engine(),
                    atomic_add=True,
                )
                cube_sync_block_arrive(PIPE.FIXPIPE, 2, mode=2)
                vec_sync_block_wait(PIPE.S, 2, mode=2)
                vec_sync_block_arrive(PIPE.MTE3, 3, mode=0)
                vec_sync_block_wait(PIPE.S, 4, mode=0)
                vec_sync_block_arrive(PIPE.MTE3, 5, mode=2)
                projection_stage.project(
                    hidden_gm, output_gate_weight_gm, block_idx - 18, k_tiles
                )
                projection_stage.publish(
                    tile_slice(
                        output_gate_workspace_gm, (128, 128), (0, block_idx - 18)
                    ),
                    subblock_idx,
                )
                vec_sync_all()
                cube_sync_block_wait(PIPE.S, 5, mode=2)
                f_l1 = Channel(
                    MemLoc.L1,
                    (self.row_count, 128),
                    dtypes.bfloat16,
                    depth=1,
                    capacity=(128, 128),
                ).produce()
                fb_l1 = Channel(
                    MemLoc.L1, (128, 128), dtypes.bfloat16, depth=1, data_format="nz"
                ).produce()
                mem_copy(f_l1, fa_workspace_gm, engine=projection_stage.nd2nz)
                mem_copy(
                    fb_l1, tile_slice(fb_weight_gm, (128, 128), (block_idx - 18, 0))
                )
                projection_stage.matmul_resident(f_l1, fb_l1)
                projection_stage.cast_result()
                projection_stage.publish(
                    tile_slice(decay_workspace_gm, (128, 128), (0, block_idx - 18)),
                    subblock_idx,
                )
                vec_sync_all()
                vec_sync_block_wait(PIPE.S, 3, mode=0)
            else:
                acc_half = Buffer(
                    MemLoc.UB,
                    (self.projection_row_block, 128),
                    dtypes.float32,
                    capacity=(64, 128),
                )
                f_half = Buffer(
                    MemLoc.UB,
                    (self.projection_row_block, 128),
                    dtypes.bfloat16,
                    capacity=(64, 128),
                )
                with vf(mode="simd"):
                    full, _ = update_mask(VL, elem_bits=32)
                    zero = vdups(0.0, dtypes.float32, mask=full)
                    for segment in range(self.projection_row_block * 128 // VL):
                        vstore(acc_half, segment * VL, zero, full)
                vec_sync_all()
                mem_copy(
                    tile_slice(
                        norm_exchange_gm,
                        make_tiler((self.projection_row_block, 128), alignment=(1, 1)),
                        (subblock_idx, 0),
                    ),
                    acc_half,
                )
                vec_sync_block_arrive(PIPE.MTE3, 0, mode=0)
                vec_sync_block_arrive(PIPE.MTE3, 3, mode=0)
                vec_sync_block_wait(PIPE.S, 3, mode=0)
                mem_copy(
                    acc_half,
                    tile_slice(
                        norm_exchange_gm,
                        make_tiler((self.projection_row_block, 128), alignment=(1, 1)),
                        (subblock_idx, 0),
                    ),
                )
                vec_sync_all()
                with vf(mode="simd"):
                    full, _ = update_mask(VL, elem_bits=32)
                    for segment in range(self.projection_row_block * 128 // VL):
                        offset = segment * VL
                        value = vcast(
                            vload(acc_half, offset), dtypes.bfloat16, mask=full
                        )
                        vstore_pack(
                            f_half, offset, value, full, pack_mode=PackMode.B32_TO_B16
                        )
                vec_sync_all()
                mem_copy(
                    tile_slice(
                        fa_workspace_gm,
                        make_tiler((self.projection_row_block, 128), alignment=(1, 1)),
                        (subblock_idx, 0),
                    ),
                    f_half,
                )
                vec_sync_block_arrive(PIPE.MTE3, 4, mode=0)
                vec_sync_block_wait(PIPE.S, 4, mode=0)
                vec_sync_block_arrive(PIPE.MTE3, 5, mode=2)
                fused = BetaZSharedStage(
                    self.row_count,
                    self.projection_input_l2_cache_ctl,
                    self.projection_weight_l2_cache_ctl,
                )
                fused.project(
                    hidden_gm,
                    output_gate_weight_gm,
                    beta_weight_gm,
                    tile_slice(output_gate_workspace_gm, (128, 128), (0, 5)),
                    beta_workspace_gm,
                    subblock_idx,
                )
                cube_sync_block_wait(PIPE.S, 5, mode=2)
                channel_rewind(reset_sync_id=True)
                projection_stage = ProjectionStage(
                    self.row_count,
                    self.projection_input_l2_cache_ctl,
                    self.projection_weight_l2_cache_ctl,
                )
                f_l1 = Channel(
                    MemLoc.L1,
                    (self.row_count, 128),
                    dtypes.bfloat16,
                    depth=1,
                    capacity=(128, 128),
                ).produce()
                fb_l1 = Channel(
                    MemLoc.L1, (128, 128), dtypes.bfloat16, depth=1, data_format="nz"
                ).produce()
                mem_copy(f_l1, fa_workspace_gm, engine=projection_stage.nd2nz)
                mem_copy(fb_l1, tile_slice(fb_weight_gm, (128, 128), (5, 0)))
                projection_stage.matmul_resident(f_l1, fb_l1)
                projection_stage.cast_result()
                projection_stage.publish(
                    tile_slice(decay_workspace_gm, (128, 128), (0, 5)), subblock_idx
                )
                vec_sync_all()
                vec_sync_block_wait(PIPE.S, 0, mode=0)

    @jit
    def _frontend_large(
        self,
        out_gm: Tensor,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        beta_workspace_gm: Tensor,
        fa_workspace_gm: Tensor,
        norm_exchange_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        beta_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        output_norm_weight_gm: Tensor,
        o_proj_weight_gm: Tensor,
        a_log_gm: Tensor,
        dt_bias_gm: Tensor,
        state_gm: Tensor,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        conv_state_gm: Tensor,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        scale_value: float,
        lower_bound: float,
        rms_norm_eps: float,
    ):
        self.row_count = hidden_gm.shape[0]
        self.projection_row_block = self.row_count // 2
        self.recurrent_wave_rows = self.row_count // self.recurrent_wave_count
        batches_per_subblock = self.batch_tile // 2
        block_idx = get_block_idx()
        block_num = get_block_num()
        subblock_idx = get_subblock_id()
        n_tiles = self.qkv_block_count
        k_tiles = ceil_div(hidden_gm.shape[1], TILE)

        if const_expr(self.block_num == 32):
            # Each group handles nine QKV tiles, three gates, and beta or decay.
            group_idx = block_idx // 16
            local_idx = block_idx % 16
            block_idx = (
                group_idx * 9 + local_idx
                if local_idx < 9
                else 18 + group_idx * 3 + local_idx - 9
                if local_idx < 12
                else 24 + group_idx
                if local_idx == 12
                else 26 + group_idx * 3 + local_idx - 13
            )

        if block_idx == self.beta_block:
            beta_stage = ProjectionStage(
                self.row_count,
                self.projection_input_l2_cache_ctl,
                self.projection_weight_l2_cache_ctl,
                tile_width=BETA_TILE,
            )
            beta_stage.project(hidden_gm, beta_weight_gm, 0, k_tiles)
            beta_stage.publish(beta_workspace_gm, subblock_idx)
        else:
            # Beta and the other projection cores use disjoint control paths
            # and can reuse the same on-chip arena and channel identifiers.
            channel_rewind(reset_sync_id=True)
            projection_stage = ProjectionStage(
                self.row_count,
                self.projection_input_l2_cache_ctl,
                self.projection_weight_l2_cache_ctl,
            )
            low_rank_stage = LowRankDecayStage(projection_stage, self.num_heads)
            conv_stage = ConvSiluStage(self.seq_len, self.projection_row_block)
            selected_history = Buffer(
                MemLoc.UB,
                (batches_per_subblock * (CONV_KERNEL - 1), TILE),
                dtypes.bfloat16,
            )
            for n_idx in range(block_idx, n_tiles, block_num):
                projection_stage.matmul_hidden(hidden_gm, qkv_weight_gm, n_idx, k_tiles)
                for projection_head in range(3 * self.num_heads):
                    if n_idx == projection_head:
                        projection = projection_head // self.num_heads
                        value_head = projection_head - projection * self.num_heads
                        channel_tile = projection * self.num_heads + value_head
                        conv_stage.load_weight(
                            tile_slice(
                                conv_weight_gm[projection, value_head, None, None],
                                (CONV_KERNEL, TILE),
                                (0, 0),
                            )
                        )
                        for local_batch in range(batches_per_subblock):
                            batch_index = (
                                subblock_idx * batches_per_subblock + local_batch
                            )
                            cache_index = dtypes.int64(
                                conv_state_indices_gm[batch_index]
                            )
                            conv_accepted = dtypes.int64(conv_accepted_gm[batch_index])
                            history_offset = (
                                conv_accepted - 1 if conv_accepted > 0 else 0
                            )
                            if cache_index != 0:
                                state_view = conv_state_gm[cache_index, None, None]
                                for history_row in range(CONV_KERNEL - 1):
                                    mem_copy(
                                        tile_slice(
                                            selected_history,
                                            (1, TILE),
                                            (
                                                local_batch * (CONV_KERNEL - 1)
                                                + history_row,
                                                0,
                                            ),
                                        ),
                                        tile_slice(
                                            state_view,
                                            (1, TILE),
                                            (
                                                history_offset + history_row,
                                                channel_tile,
                                            ),
                                        ),
                                    )
                projection_stage.acquire_result()

                for projection_head in range(3 * self.num_heads):
                    if n_idx == projection_head:
                        projection = projection_head // self.num_heads
                        for local_batch in range(batches_per_subblock):
                            batch_index = (
                                subblock_idx * batches_per_subblock + local_batch
                            )
                            cache_index = dtypes.int64(
                                conv_state_indices_gm[batch_index]
                            )
                            if cache_index == 0:
                                conv_stage.zero_sequence(local_batch)
                            else:
                                history_view = tile_slice(
                                    selected_history,
                                    (CONV_KERNEL - 1, TILE),
                                    (local_batch, 0),
                                )
                                conv_stage.install_history(history_view)
                                state_view = conv_state_gm[cache_index, None, None]
                                for cache_row in range(2):
                                    mem_copy(
                                        tile_slice(
                                            state_view,
                                            (1, TILE),
                                            (cache_row, projection_head),
                                        ),
                                        tile_slice(
                                            history_view, (1, TILE), (cache_row + 1, 0)
                                        ),
                                    )
                                conv_stage.convolve_sequence(
                                    local_batch,
                                    projection_stage.projection_fp32,
                                    tile_slice(
                                        conv_stage.weight, (CONV_KERNEL, TILE), (0, 0)
                                    ),
                                )
                        # Raw VF stores must finish before MTE3 reads the Conv
                        # snapshots. One drain covers every batch because each
                        # batch owns disjoint raw_bf16 rows.
                        vec_sync_notify(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                        vec_sync_wait(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                        for local_batch in range(batches_per_subblock):
                            batch_index = (
                                subblock_idx * batches_per_subblock + local_batch
                            )
                            cache_index = dtypes.int64(
                                conv_state_indices_gm[batch_index]
                            )
                            if cache_index != 0:
                                state_view = conv_state_gm[cache_index, None, None]
                                for cache_token in range(self.seq_len):
                                    mem_copy(
                                        tile_slice(
                                            state_view,
                                            (1, TILE),
                                            (cache_token + 2, projection_head),
                                        ),
                                        tile_slice(
                                            conv_stage.raw_bf16,
                                            (1, TILE),
                                            (
                                                local_batch * self.seq_len
                                                + cache_token,
                                                0,
                                            ),
                                        ),
                                    )
                vec_sync_all()
                if n_idx < self.num_heads:
                    projection_stage.publish_source(
                        tile_slice(q_workspace_gm, (128, TILE), (0, n_idx)),
                        conv_stage.output_bf16,
                        subblock_idx,
                    )
                elif n_idx < 2 * self.num_heads:
                    projection_stage.publish_source(
                        tile_slice(
                            k_workspace_gm, (128, TILE), (0, n_idx - self.num_heads)
                        ),
                        conv_stage.output_bf16,
                        subblock_idx,
                    )
                else:
                    projection_stage.publish_source(
                        tile_slice(
                            v_workspace_gm, (128, TILE), (0, n_idx - 2 * self.num_heads)
                        ),
                        conv_stage.output_bf16,
                        subblock_idx,
                    )

            if block_idx >= self.qkv_block_count and block_idx < self.beta_block:
                output_head = block_idx - self.qkv_block_count
                projection_stage.project(
                    hidden_gm, output_gate_weight_gm, output_head, k_tiles
                )
                projection_stage.publish(
                    tile_slice(output_gate_workspace_gm, (128, TILE), (0, output_head)),
                    subblock_idx,
                )

            if block_idx == self.low_rank_block:
                low_rank_stage.project(
                    decay_workspace_gm,
                    hidden_gm,
                    fa_weight_gm,
                    fb_weight_gm,
                    k_tiles,
                    subblock_idx,
                )

    @jit
    def _execute_compact_s8(
        self,
        out_gm: Tensor,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        beta_workspace_gm: Tensor,
        fa_workspace_gm: Tensor,
        norm_exchange_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        beta_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        output_norm_weight_gm: Tensor,
        o_proj_weight_gm: Tensor,
        a_log_gm: Tensor,
        dt_bias_gm: Tensor,
        state_gm: Tensor,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        conv_state_gm: Tensor,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        scale_value: float,
        lower_bound: float,
        rms_norm_eps: float,
    ):
        self.batch_groups = replay_decay_gm.shape[0] // BATCH_TILE
        self.seq_len = SEQ_LEN
        self.row_count = replay_decay_gm.shape[0] * self.seq_len
        self.projection_row_block = self.row_count // 2
        recurrent_items = replay_decay_gm.shape[0] * self.num_heads
        recurrent_lanes = min(48, 2 * self.block_num)
        recurrent_wave_count = ceil_div(recurrent_items, recurrent_lanes)
        self.recurrent_wave_rows = self.row_count // recurrent_wave_count
        if const_expr(self.block_num == 24):
            self._frontend_24_single_group(
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                decay_workspace_gm,
                output_gate_workspace_gm,
                beta_workspace_gm,
                fa_workspace_gm,
                norm_exchange_gm,
                hidden_gm,
                qkv_weight_gm,
                fa_weight_gm,
                fb_weight_gm,
                beta_weight_gm,
                output_gate_weight_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
            )
        else:
            self._frontend_large(
                out_gm,
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                decay_workspace_gm,
                output_gate_workspace_gm,
                beta_workspace_gm,
                fa_workspace_gm,
                norm_exchange_gm,
                hidden_gm,
                qkv_weight_gm,
                fa_weight_gm,
                fb_weight_gm,
                beta_weight_gm,
                output_gate_weight_gm,
                output_norm_weight_gm,
                o_proj_weight_gm,
                a_log_gm,
                dt_bias_gm,
                state_gm,
                replay_u_gm,
                replay_k_gm,
                replay_decay_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
                scale_value,
                lower_bound,
                rms_norm_eps,
            )
        global_sync_all()
        channel_rewind(reset_sync_id=True)

        state_row_block = TILE
        block_idx = get_block_idx()
        subblock_idx = get_subblock_id()
        if const_expr(self.block_num == 32):
            group_idx = block_idx // 16
            local_idx = block_idx % 16
            block_idx = (
                group_idx * 12 + local_idx
                if local_idx < 12
                else 24 + group_idx * 4 + local_idx - 12
            )
        recurrent_active = block_idx < recurrent_lanes // 2
        recurrent_stage = RecurrentKDAStage(
            state_row_block,
            self.num_heads,
            state_key_unroll=KDA_STATE_KEY_UNROLL,
            state_update_unroll=KDA_STATE_UPDATE_UNROLL,
        )
        output_gate_stage = OutputGateStage()
        output_projection_stage = OutputProjectionStage(
            self.output_tile_width,
            self.recurrent_wave_rows,
            tiles_per_core=self.output_tiles_per_core,
            resident_input=False,
            resident_weight=self.output_weight_resident,
            tile_stride=24 if self.block_num == 24 else 1,
            l0c_depth=2 if self.block_num == 24 else 1,
        )
        if const_expr(self.output_weight_resident):
            output_projection_stage.prefetch_weight(o_proj_weight_gm)
        if recurrent_active:
            recurrent_stage.load_gate_params(a_log_gm, dt_bias_gm)
            output_gate_stage.load_gamma(output_norm_weight_gm)
        for wave in range(recurrent_wave_count):
            # Compact grids share 48 jobs per wave; surplus AICs only project.
            item_start = wave * recurrent_lanes + block_idx * 2 + subblock_idx
            item_count = 1
            if recurrent_active:
                run_recurrent_wave(
                    recurrent_stage,
                    output_gate_stage,
                    q_workspace_gm,
                    k_workspace_gm,
                    v_workspace_gm,
                    decay_workspace_gm,
                    output_gate_workspace_gm,
                    beta_workspace_gm,
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    item_start,
                    item_start + item_count,
                    replay_decay_gm.shape[0],
                    self.seq_len,
                    self.num_heads,
                    state_row_block,
                    scale_value,
                    lower_bound,
                    rms_norm_eps,
                )
            publish_batch_group_vector_grid_to_cube(3 + wave * 2, 4 + wave * 2)
            output_projection_stage.project(
                out_gm, q_workspace_gm, o_proj_weight_gm, self.num_heads, wave
            )

    @jit
    def _run_backend_stream(
        self,
        out_gm: Tensor,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        beta_workspace_gm: Tensor,
        fa_workspace_gm: Tensor,
        norm_exchange_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        beta_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        output_norm_weight_gm: Tensor,
        o_proj_weight_gm: Tensor,
        a_log_gm: Tensor,
        dt_bias_gm: Tensor,
        state_gm: Tensor,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        conv_state_gm: Tensor,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        scale_value: float,
        lower_bound: float,
        rms_norm_eps: float,
    ):
        batch_size = replay_decay_gm.shape[0]
        total = batch_size * 6
        slot_capacity = 2 * get_block_num()
        slot_count = ceil_div(total, slot_capacity)
        chunk_rows = 8 * self.seq_len
        block_idx = get_block_idx()
        subblock_idx = get_subblock_id()
        recurrent_stage = RecurrentKDAStage(
            TILE,
            self.num_heads,
            state_key_unroll=KDA_STATE_KEY_UNROLL,
            state_update_unroll=KDA_STATE_UPDATE_UNROLL,
        )
        output_gate_stage = OutputGateStage()
        if const_expr(self.block_num == 8):
            output_projection_stage = OutputProjectionStage(
                self.output_tile_width,
                chunk_rows,
                kc=384,
                tiles_per_core=self.output_tiles_per_core,
                row_capacity=64,
                resident_input=True,
                resident_weight=False,
                tile_stride=1,
                l0c_depth=2,
            )
        elif const_expr(self.block_num == 16 or self.block_num == 24):
            output_projection_stage = OutputProjectionStage(
                self.output_tile_width,
                chunk_rows,
                kc=384,
                tiles_per_core=self.output_tiles_per_core,
                row_capacity=64,
                resident_input=False,
                resident_weight=False,
                tile_stride=self.block_num,
                l0c_depth=2,
            )
        else:
            output_projection_stage = OutputProjectionStage(
                self.output_tile_width,
                chunk_rows,
                kc=384,
                tiles_per_core=self.output_tiles_per_core,
                row_capacity=64,
                resident_input=False,
                resident_weight=True,
                tile_stride=1,
                l0c_depth=1,
            )
        output_projection_stage.prefetch_weight(o_proj_weight_gm)
        recurrent_stage.load_gate_params(a_log_gm, dt_bias_gm)
        output_gate_stage.load_gamma(output_norm_weight_gm)
        first_item = block_idx * 2 + subblock_idx
        if first_item < total:
            prefetch_recurrent_item(
                recurrent_stage,
                output_gate_stage,
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                decay_workspace_gm,
                output_gate_workspace_gm,
                beta_workspace_gm,
                first_item,
                batch_size,
                self.seq_len,
                6,
                TILE,
            )
            prefetch_recurrent_state(
                recurrent_stage,
                state_gm,
                first_item,
                batch_size,
                self.seq_len,
                6,
                TILE,
            )
        for slot in range(slot_count):
            # Each AIV owns a complete batch/head, including every D128 row.
            item = slot * slot_capacity + block_idx * 2 + subblock_idx
            if item < total:
                run_recurrent_wave(
                    recurrent_stage,
                    output_gate_stage,
                    q_workspace_gm,
                    k_workspace_gm,
                    v_workspace_gm,
                    decay_workspace_gm,
                    output_gate_workspace_gm,
                    beta_workspace_gm,
                    state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    item,
                    item + 1,
                    batch_size,
                    self.seq_len,
                    6,
                    TILE,
                    scale_value,
                    lower_bound,
                    rms_norm_eps,
                    first_inputs_prefetched=True,
                    first_state_prefetched=True,
                    next_item_stride=2 * self.block_num,
                )
            completed_before = (
                slot * slot_capacity if slot * slot_capacity < total else total
            )
            completed_after = (
                (slot + 1) * slot_capacity
                if (slot + 1) * slot_capacity < total
                else total
            )
            ready_before = completed_before // 48
            ready_after = completed_after // 48
            publish_vector_grid_to_cube(3, 4, ack_flag=5)
            for chunk in range(ready_before, ready_after):
                output_projection_stage.project(
                    out_gm, q_workspace_gm, o_proj_weight_gm, k_tiles=6, row_tile=chunk
                )
            channel_rewind(reset_sync_id=True)

    @jit
    def _execute_task_pool_s8(
        self,
        out_gm: Tensor,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        beta_workspace_gm: Tensor,
        fa_workspace_gm: Tensor,
        norm_exchange_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        beta_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        output_norm_weight_gm: Tensor,
        o_proj_weight_gm: Tensor,
        a_log_gm: Tensor,
        dt_bias_gm: Tensor,
        state_gm: Tensor,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        conv_state_gm: Tensor,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        scale_value: float,
        lower_bound: float,
        rms_norm_eps: float,
    ):
        self.batch_groups = replay_decay_gm.shape[0] // BATCH_TILE
        self.seq_len = SEQ_LEN
        self.row_count = 128
        self.projection_row_block = 64
        self.recurrent_wave_rows = 64
        # An N384 task packs three N128 panels; select it once its eight tasks
        # per batch group keep at least three quarters of the launch grid busy.
        n384_task_count = self.batch_groups * 8
        use_n384 = n384_task_count * 4 >= self.block_num * 3
        use_32core_g2_tail = self.batch_groups == 2 and self.block_num == 32

        if use_n384:
            self._frontend_adaptive_n384(
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                decay_workspace_gm,
                output_gate_workspace_gm,
                hidden_gm,
                qkv_weight_gm,
                fa_weight_gm,
                fb_weight_gm,
                output_gate_weight_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
            )
            # Cube must drain before its channel identities are rewound for
            # auxiliary tasks. Vector has no remaining producer/consumer here.
            cube_sync_all()
            channel_rewind(reset_sync_id=True)
            self._frontend_adaptive_auxiliary(
                decay_workspace_gm,
                beta_workspace_gm,
                hidden_gm,
                fa_weight_gm,
                fb_weight_gm,
                beta_weight_gm,
            )
        elif use_32core_g2_tail:
            self._frontend_tail(
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                decay_workspace_gm,
                output_gate_workspace_gm,
                beta_workspace_gm,
                hidden_gm,
                qkv_weight_gm,
                fa_weight_gm,
                fb_weight_gm,
                beta_weight_gm,
                output_gate_weight_gm,
                replay_decay_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
            )
        else:
            self._frontend_flat_m128(
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                decay_workspace_gm,
                output_gate_workspace_gm,
                beta_workspace_gm,
                hidden_gm,
                qkv_weight_gm,
                fa_weight_gm,
                fb_weight_gm,
                beta_weight_gm,
                output_gate_weight_gm,
                replay_decay_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
            )
        global_sync_all()
        channel_rewind(reset_sync_id=True)

        self._run_backend_stream(
            out_gm,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            decay_workspace_gm,
            output_gate_workspace_gm,
            beta_workspace_gm,
            fa_workspace_gm,
            norm_exchange_gm,
            hidden_gm,
            qkv_weight_gm,
            fa_weight_gm,
            fb_weight_gm,
            beta_weight_gm,
            output_gate_weight_gm,
            output_norm_weight_gm,
            o_proj_weight_gm,
            a_log_gm,
            dt_bias_gm,
            state_gm,
            replay_u_gm,
            replay_k_gm,
            replay_decay_gm,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            scale_value,
            lower_bound,
            rms_norm_eps,
        )

    @jit
    def execute(
        self,
        out_gm: Tensor,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        beta_workspace_gm: Tensor,
        fa_workspace_gm: Tensor,
        norm_exchange_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        beta_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        output_norm_weight_gm: Tensor,
        o_proj_weight_gm: Tensor,
        a_log_gm: Tensor,
        dt_bias_gm: Tensor,
        state_gm: Tensor,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        conv_state_gm: Tensor,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        scale_value: float,
        lower_bound: float,
        rms_norm_eps: float,
    ):
        execution_args = (
            out_gm,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            decay_workspace_gm,
            output_gate_workspace_gm,
            beta_workspace_gm,
            fa_workspace_gm,
            norm_exchange_gm,
            hidden_gm,
            qkv_weight_gm,
            fa_weight_gm,
            fb_weight_gm,
            beta_weight_gm,
            output_gate_weight_gm,
            output_norm_weight_gm,
            o_proj_weight_gm,
            a_log_gm,
            dt_bias_gm,
            state_gm,
            replay_u_gm,
            replay_k_gm,
            replay_decay_gm,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            scale_value,
            lower_bound,
            rms_norm_eps,
        )
        batch_groups = replay_decay_gm.shape[0] // BATCH_TILE
        frontend_panel_tasks = batch_groups * (
            self.qkv_block_count + self.output_gate_block_count
        )
        # A B16 group needs 24 frontend panels, so 8/16-core grids always
        # use the task pool. Keep their unreachable compact paths out of IR.
        if const_expr(self.block_num >= 24) and frontend_panel_tasks <= self.block_num:
            self._execute_compact_s8(*execution_args)
        else:
            self._execute_task_pool_s8(*execution_args)


class MegaRecurrentKDAReplaySSMKernel:
    """One compiled S8 kernel with a runtime launch grid and batch size."""

    def __init__(self, *, hidden_size: int, num_heads: int):
        if hidden_size != HIDDEN_SIZE or num_heads != 6:
            raise ValueError(f"requires H={HIDDEN_SIZE} and N=6")
        self.workspace_region_widths = (num_heads * TILE,) * 5 + (BETA_TILE, TILE)
        self.workspace_width = sum(self.workspace_region_widths)
        self.grid_8 = _GridSchedule(8, hidden_size, num_heads)
        self.grid_16 = _GridSchedule(16, hidden_size, num_heads)
        self.grid_24 = _GridSchedule(24, hidden_size, num_heads)
        self.grid_28 = _GridSchedule(28, hidden_size, num_heads)
        self.grid_32 = _GridSchedule(32, hidden_size, num_heads)

    @kernel(
        profile=cannbotdsl.ProfileSpec(
            name="mega_recurrent_kda_replayssm",
            op_type="MegaRecurrentKDAReplaySSM",
            inputs=(
                "hidden_gm",
                "qkv_weight_gm",
                "fa_weight_gm",
                "fb_weight_gm",
                "beta_weight_gm",
                "output_gate_weight_gm",
                "output_norm_weight_gm",
                "o_proj_weight_gm",
                "a_log_gm",
                "dt_bias_gm",
                "state_gm",
                "conv_state_indices_gm",
                "conv_accepted_gm",
                "conv_weight_gm",
            ),
            outputs=("out_gm", "replay_u_gm", "replay_k_gm", "replay_decay_gm"),
            inouts=("conv_state_gm",),
        )
    )
    def kernel(
        self,
        out_gm: Tensor,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        beta_workspace_gm: Tensor,
        fa_workspace_gm: Tensor,
        norm_exchange_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        beta_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        output_norm_weight_gm: Tensor,
        o_proj_weight_gm: Tensor,
        a_log_gm: Tensor,
        dt_bias_gm: Tensor,
        state_gm: Tensor,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        conv_state_gm: Tensor,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        scale_value: float,
        lower_bound: float,
        rms_norm_eps: float,
    ):
        block_num = get_block_num()
        execution_args = (
            out_gm,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            decay_workspace_gm,
            output_gate_workspace_gm,
            beta_workspace_gm,
            fa_workspace_gm,
            norm_exchange_gm,
            hidden_gm,
            qkv_weight_gm,
            fa_weight_gm,
            fb_weight_gm,
            beta_weight_gm,
            output_gate_weight_gm,
            output_norm_weight_gm,
            o_proj_weight_gm,
            a_log_gm,
            dt_bias_gm,
            state_gm,
            replay_u_gm,
            replay_k_gm,
            replay_decay_gm,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            scale_value,
            lower_bound,
            rms_norm_eps,
        )
        if block_num == 8:
            self.grid_8.execute(*execution_args)
        elif block_num == 16:
            self.grid_16.execute(*execution_args)
        elif block_num == 24:
            self.grid_24.execute(*execution_args)
        elif block_num == 28:
            self.grid_28.execute(*execution_args)
        elif block_num == 32:
            self.grid_32.execute(*execution_args)

    @host
    def run(
        self,
        out_gm: Tensor,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        beta_workspace_gm: Tensor,
        fa_workspace_gm: Tensor,
        norm_exchange_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        beta_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        output_norm_weight_gm: Tensor,
        o_proj_weight_gm: Tensor,
        a_log_gm: Tensor,
        dt_bias_gm: Tensor,
        state_gm: Tensor,
        replay_u_gm: Tensor,
        replay_k_gm: Tensor,
        replay_decay_gm: Tensor,
        conv_state_gm: Tensor,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        scale_value: float,
        lower_bound: float,
        rms_norm_eps: float,
        launch_cores: int,
    ):
        self.kernel[launch_cores](
            out_gm,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            decay_workspace_gm,
            output_gate_workspace_gm,
            beta_workspace_gm,
            fa_workspace_gm,
            norm_exchange_gm,
            hidden_gm,
            qkv_weight_gm,
            fa_weight_gm,
            fb_weight_gm,
            beta_weight_gm,
            output_gate_weight_gm,
            output_norm_weight_gm,
            o_proj_weight_gm,
            a_log_gm,
            dt_bias_gm,
            state_gm,
            replay_u_gm,
            replay_k_gm,
            replay_decay_gm,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            scale_value,
            lower_bound,
            rms_norm_eps,
        )


def _tensor_spec(dtype, tensor: torch.Tensor, shape=None, *, storage_format="nd"):
    return cannbotdsl.TensorSpec(
        tuple(tensor.shape) if shape is None else shape,
        dtype,
        storage_format=storage_format,
    )


def _pack_conv_weights(conv1d_weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    if tuple(conv1d_weight.shape) == (3, num_heads, CONV_KERNEL, TILE):
        return conv1d_weight
    if tuple(conv1d_weight.shape) == (3, CONV_KERNEL, TILE):
        return (
            conv1d_weight[:, None, :, :]
            .expand(3, num_heads, CONV_KERNEL, TILE)
            .contiguous()
        )
    if tuple(conv1d_weight.shape) == (CONV_KERNEL, 3 * num_heads * TILE):
        return (
            conv1d_weight.view(CONV_KERNEL, 3, num_heads, TILE)
            .permute(1, 2, 0, 3)
            .contiguous()
        )
    raise ValueError("conv1d_weight must be [3,N,4,128], [3,4,128], or [4,3*N*128]")


def _validate_runtime_scalars(
    scale: float,
    lower_bound: float,
    rms_norm_eps: float,
) -> tuple[float, float, float]:
    scale_value = float(scale)
    lower_bound_value = float(lower_bound)
    rms_norm_eps_value = float(rms_norm_eps)
    if not math.isfinite(scale_value):
        raise ValueError("scale must be finite")
    if not -5.0 <= lower_bound_value <= 0.0:
        raise ValueError("lower_bound must be in [-5,0]")
    if not math.isfinite(rms_norm_eps_value) or rms_norm_eps_value <= 0.0:
        raise ValueError("rms_norm_eps must be finite and positive")
    return scale_value, lower_bound_value, rms_norm_eps_value


def _validate_replay_hidden(hidden_states, beta_weight):
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must be [B,S,H]")
    batch_size, seq_len, hidden_size = map(int, hidden_states.shape)
    invalid_batch = (
        not BATCH_TILE <= batch_size <= BATCH_TILE * MAX_BATCH_GROUPS
        or batch_size % BATCH_TILE
    )
    if invalid_batch or hidden_size != HIDDEN_SIZE or seq_len != SEQ_LEN:
        raise ValueError(
            "mega_recurrent_kda_replayssm requires B to be a multiple of 16 in "
            f"[16, 512], S={SEQ_LEN}, and H={HIDDEN_SIZE}; got B={batch_size}, "
            f"S={seq_len}, H={hidden_size}"
        )
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous BF16")
    if beta_weight.ndim != 2 or int(beta_weight.shape[1]) != hidden_size:
        raise ValueError("beta_projection_weight must be [N,H]")
    num_heads = int(beta_weight.shape[0])
    return batch_size, seq_len, hidden_size, num_heads


def _validate_replay_weight_shapes(
    hidden_size,
    num_heads,
    qkv_weight,
    decay_a_weight,
    decay_b_weight,
    output_gate_weight,
    output_norm_weight,
    output_weight,
):
    shape_checks = (
        (
            tuple(qkv_weight.shape) == (3 * num_heads * TILE, hidden_size),
            "qkv_projection_weight must be [3*N*128,H]",
        ),
        (
            tuple(decay_a_weight.shape) == (TILE, hidden_size),
            "decay_projection_a_weight must be [128,H]",
        ),
        (
            tuple(decay_b_weight.shape) == (num_heads * TILE, TILE),
            "decay_projection_b_weight must be [N*128,128]",
        ),
        (
            tuple(output_gate_weight.shape) == (num_heads * TILE, hidden_size),
            "output_gate_projection_weight must be [N*128,H]",
        ),
        (
            tuple(output_norm_weight.shape) == (TILE,),
            "output_norm_weight must be [128]",
        ),
        (
            tuple(output_weight.shape) == (hidden_size, num_heads * TILE),
            "output_projection_weight must be [H,N*128]",
        ),
    )
    for condition, message in shape_checks:
        if not condition:
            raise ValueError(message)


def _validate_replay_weight_properties(hidden_states, bf16_tensors, output_norm_weight):
    for name, tensor in bf16_tensors:
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be BF16")
        if tensor.device != hidden_states.device:
            raise ValueError(f"{name} must be on the hidden_states device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if output_norm_weight.dtype != torch.float32:
        raise ValueError("output_norm_weight must be FP32")
    if output_norm_weight.device != hidden_states.device:
        raise ValueError("output_norm_weight must be on the hidden_states device")
    if not output_norm_weight.is_contiguous():
        raise ValueError("output_norm_weight must be contiguous")


def _validate_replay_weights(
    hidden_states,
    hidden_size,
    num_heads,
    qkv_weight,
    decay_a_weight,
    decay_b_weight,
    beta_weight,
    output_gate_weight,
    output_norm_weight,
    output_weight,
    conv_weight,
):
    _validate_replay_weight_shapes(
        hidden_size,
        num_heads,
        qkv_weight,
        decay_a_weight,
        decay_b_weight,
        output_gate_weight,
        output_norm_weight,
        output_weight,
    )
    if num_heads != 6:
        raise ValueError(
            f"mega_recurrent_kda_replayssm requires N=6; got N={num_heads}"
        )
    bf16_tensors = (
        ("qkv_projection_weight", qkv_weight),
        ("decay_projection_a_weight", decay_a_weight),
        ("decay_projection_b_weight", decay_b_weight),
        ("beta_projection_weight", beta_weight),
        ("output_gate_projection_weight", output_gate_weight),
        ("output_projection_weight", output_weight),
        ("conv1d_weight", conv_weight),
    )
    _validate_replay_weight_properties(hidden_states, bf16_tensors, output_norm_weight)


def _validate_replay_state_shapes(
    batch_size,
    seq_len,
    num_heads,
    conv_state,
    recurrent_state,
):
    conv_shape_error = (
        conv_state.ndim != 3
        or int(conv_state.shape[0]) < 1
        or int(conv_state.shape[1]) < seq_len + CONV_KERNEL - 2
        or int(conv_state.shape[2]) != 3 * num_heads * TILE
    )
    if conv_shape_error:
        raise ValueError(
            "conv_state must be [pool,state_length,3*N*128] with state_length >= S+2"
        )
    if conv_state.dtype != torch.bfloat16 or not conv_state.is_contiguous():
        raise ValueError("conv_state must be contiguous BF16")
    if tuple(recurrent_state.shape) != (batch_size, num_heads, TILE, TILE):
        raise ValueError("recurrent_state must be [B,N,128,128]")
    if recurrent_state.dtype != torch.float32 or not recurrent_state.is_contiguous():
        raise ValueError("recurrent_state must be contiguous FP32")


def _validate_replay_buffers(replay_buffers, replay_shape):
    replay_u, replay_k, replay_decay = replay_buffers
    for name, tensor in (("replay_u", replay_u), ("replay_k", replay_k)):
        if tuple(tensor.shape) != replay_shape:
            raise ValueError(f"{name} must be [B,S,N,128]")
        if tensor.dtype != torch.float16 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous FP16")
    if tuple(replay_decay.shape) != replay_shape:
        raise ValueError("replay_decay must be [B,S,N,128]")
    if replay_decay.dtype != torch.float32 or not replay_decay.is_contiguous():
        raise ValueError("replay_decay must be contiguous FP32")


def _validate_replay_state(
    hidden_states,
    batch_size,
    seq_len,
    num_heads,
    conv_state,
    recurrent_state,
    replay_buffers,
    a_log,
    dt_bias,
):
    _validate_replay_state_shapes(
        batch_size, seq_len, num_heads, conv_state, recurrent_state
    )
    replay_shape = (batch_size, seq_len, num_heads, TILE)
    _validate_replay_buffers(replay_buffers, replay_shape)
    replay_u, replay_k, replay_decay = replay_buffers
    if tuple(a_log.shape) != (num_heads,) or a_log.dtype != torch.float32:
        raise ValueError("a_log must be FP32 [N]")
    if tuple(dt_bias.shape) != (num_heads, TILE) or dt_bias.dtype != torch.float32:
        raise ValueError("dt_bias must be FP32 [N,128]")
    state_tensors = (
        conv_state,
        recurrent_state,
        replay_u,
        replay_k,
        replay_decay,
        a_log,
        dt_bias,
    )
    if any(tensor.device != hidden_states.device for tensor in state_tensors):
        raise ValueError("all inputs must share one device")
    if any(not tensor.is_contiguous() for tensor in (a_log, dt_bias)):
        raise ValueError("a_log and dt_bias must be contiguous")


def _validate_replay_indices(device, batch_size, index_tensors):
    for name, tensor in index_tensors:
        if tuple(tensor.shape) != (batch_size,):
            raise ValueError(f"{name} must be [B]")
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be int32")
        if tensor.device != device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on the input device")


def _replay_workspace_views(workspace, row_count, widths):
    regions = torch.split(workspace, [row_count * width for width in widths])
    views = []
    for region, width in zip(regions, widths):
        views.append(region.view(row_count, width))
    return tuple(views)


class _ReplayWeightSpecs(NamedTuple):
    qkv: object
    decay_a: object
    decay_b: object
    beta: object
    output_gate: object
    output_norm: object
    output_weight: object


class _ReplayStateSpecs(NamedTuple):
    a_log: object
    dt_bias: object
    recurrent_state: object
    replay_u: object
    replay_k: object
    replay_decay: object


def _replay_weight_specs(weights):
    qkv, decay_a, decay_b, beta, output_gate, output_norm, output_weight = weights
    return _ReplayWeightSpecs(
        _tensor_spec(dtypes.bfloat16, qkv, storage_format="nz"),
        _tensor_spec(dtypes.bfloat16, decay_a, storage_format="nz"),
        _tensor_spec(dtypes.bfloat16, decay_b, storage_format="nz"),
        _tensor_spec(dtypes.bfloat16, beta, storage_format="nz"),
        _tensor_spec(dtypes.bfloat16, output_gate, storage_format="nz"),
        _tensor_spec(dtypes.float32, output_norm),
        _tensor_spec(dtypes.bfloat16, output_weight, storage_format="nz"),
    )


def _replay_state_specs(batch_dim, state_inputs):
    a_log, dt_bias, recurrent_state, replay_u, replay_k, replay_decay = state_inputs
    state_shape = (batch_dim, int(recurrent_state.shape[1]), TILE, TILE)
    replay_shape = (
        batch_dim,
        int(replay_u.shape[1]),
        int(replay_u.shape[2]),
        TILE,
    )
    return _ReplayStateSpecs(
        _tensor_spec(dtypes.float32, a_log),
        _tensor_spec(dtypes.float32, dt_bias),
        _tensor_spec(dtypes.float32, recurrent_state, state_shape),
        _tensor_spec(dtypes.float16, replay_u, replay_shape),
        _tensor_spec(dtypes.float16, replay_k, replay_shape),
        _tensor_spec(dtypes.float32, replay_decay, replay_shape),
    )


def _replay_conv_specs(batch_dim, conv_dims, conv_inputs):
    conv_pool_dim, conv_tokens_dim = conv_dims
    conv_state, conv_indices, conv_accepted, packed_conv = conv_inputs
    return (
        _tensor_spec(
            dtypes.bfloat16,
            conv_state,
            (conv_pool_dim, conv_tokens_dim, int(conv_state.shape[2])),
        ),
        _tensor_spec(dtypes.int32, conv_indices, (batch_dim,)),
        _tensor_spec(dtypes.int32, conv_accepted, (batch_dim,)),
        _tensor_spec(dtypes.bfloat16, packed_conv),
    )


def _replay_compile_specs(
    dims,
    flat_output,
    workspace_views,
    norm_exchange,
    flat_hidden,
    weights,
    state_inputs,
    conv_inputs,
):
    rows_dim, batch_dim, conv_pool_dim, conv_tokens_dim = dims
    specs = [
        _tensor_spec(
            dtypes.bfloat16, flat_output, (rows_dim, int(flat_output.shape[1]))
        )
    ]
    specs.extend(
        _tensor_spec(dtypes.bfloat16, region, (rows_dim, int(region.shape[1])))
        for region in workspace_views
    )
    group_rows = cannbotdsl.Dim("group_rows", min=16, max=128, multiple_of=16)
    specs.extend(
        (
            _tensor_spec(dtypes.float32, norm_exchange, (group_rows, TILE)),
            _tensor_spec(
                dtypes.bfloat16, flat_hidden, (rows_dim, int(flat_hidden.shape[1]))
            ),
            *_replay_weight_specs(weights),
            *_replay_state_specs(batch_dim, state_inputs),
            *_replay_conv_specs(
                batch_dim, (conv_pool_dim, conv_tokens_dim), conv_inputs
            ),
            dtypes.float32,
            dtypes.float32,
            dtypes.float32,
            dtypes.int64,
        )
    )
    return tuple(specs)


def _create_replay_launch(flat_output, flat_hidden, compile_inputs, shape):
    hidden_size, num_heads, row_count, seq_len, device = shape
    op = MegaRecurrentKDAReplaySSMKernel(hidden_size=hidden_size, num_heads=num_heads)
    workspace = torch.empty(
        row_count * op.workspace_width, dtype=torch.bfloat16, device=device
    )
    workspace_views = _replay_workspace_views(
        workspace, row_count, op.workspace_region_widths
    )
    norm_exchange = torch.empty(
        (BATCH_TILE * seq_len, TILE), dtype=torch.float32, device=device
    )
    dims = (
        cannbotdsl.Dim("rows", min=16, max=4096, multiple_of=16),
        cannbotdsl.Dim("batch", min=16, max=512, multiple_of=16),
        cannbotdsl.Dim("conv_pool"),
        cannbotdsl.Dim("conv_tokens", min=3),
    )
    weights, state_inputs, conv_inputs = compile_inputs
    specs = _replay_compile_specs(
        dims,
        flat_output,
        workspace_views,
        norm_exchange,
        flat_hidden,
        weights,
        state_inputs,
        conv_inputs,
    )
    fn = cannbotdsl.compile(op.run, *specs)
    return _CompiledLaunch(fn, workspace, workspace_views, norm_exchange)


def _resize_replay_launch(launch, row_count, seq_len, device):
    widths = tuple(int(region.shape[1]) for region in launch.workspace_views)
    launch.workspace = torch.empty(
        row_count * sum(widths), dtype=torch.bfloat16, device=device
    )
    launch.workspace_views = _replay_workspace_views(
        launch.workspace, row_count, widths
    )
    launch.norm_exchange = torch.empty(
        BATCH_TILE * seq_len, TILE, dtype=torch.float32, device=device
    )


def _invoke_replay_launch(
    launch, flat_output, flat_hidden, weights, state_inputs, conv_inputs, scalars
):
    conv_state, conv_indices, conv_accepted, packed_conv = conv_inputs
    scale_value, lower_bound_value, rms_norm_eps_value, launch_cores = scalars
    launch.fn(
        flat_output,
        *launch.workspace_views,
        launch.norm_exchange,
        flat_hidden,
        *weights,
        *state_inputs,
        conv_state,
        conv_indices,
        conv_accepted,
        packed_conv,
        scale_value,
        lower_bound_value,
        rms_norm_eps_value,
        launch_cores,
    )


def _run_replay_launch(
    hidden_states, shape, weights, state_inputs, conv_inputs, scalars
):
    batch_size, seq_len, hidden_size, num_heads = shape
    device = hidden_states.device
    conv_state, conv_indices, conv_accepted, conv_weight = conv_inputs
    packed_conv = _pack_conv_weights(conv_weight, num_heads)
    packed_conv_inputs = (conv_state, conv_indices, conv_accepted, packed_conv)
    row_count = batch_size * seq_len
    flat_hidden = hidden_states.view(row_count, hidden_size)
    flat_output = torch.empty(
        row_count, hidden_size, dtype=torch.bfloat16, device=device
    )
    compile_inputs = (weights, state_inputs, packed_conv_inputs)
    global _COMPILED_LAUNCH
    launch = _COMPILED_LAUNCH
    if launch is None:
        launch = _create_replay_launch(
            flat_output,
            flat_hidden,
            compile_inputs,
            (hidden_size, num_heads, row_count, seq_len, device),
        )
        _COMPILED_LAUNCH = launch
    elif (
        launch.workspace.device != device
        or launch.workspace_views[0].shape[0] != row_count
        or launch.norm_exchange.shape[0] != BATCH_TILE * seq_len
    ):
        _resize_replay_launch(launch, row_count, seq_len, device)
    _invoke_replay_launch(
        launch,
        flat_output,
        flat_hidden,
        weights,
        state_inputs,
        packed_conv_inputs,
        scalars,
    )
    return flat_output.view(batch_size, seq_len, hidden_size)


class _ReplayCall(NamedTuple):
    hidden_states: torch.Tensor
    qkv_weight: torch.Tensor
    decay_a_weight: torch.Tensor
    decay_b_weight: torch.Tensor
    beta_weight: torch.Tensor
    output_gate_weight: torch.Tensor
    output_norm_weight: torch.Tensor
    output_weight: torch.Tensor
    conv_weight: torch.Tensor
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor
    replay_u: torch.Tensor
    replay_k: torch.Tensor
    replay_decay: torch.Tensor
    conv_indices: torch.Tensor
    conv_accepted: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    scale: float
    lower_bound: float
    rms_norm_eps: float


def _validate_replay_call(call):
    shape = _validate_replay_hidden(call.hidden_states, call.beta_weight)
    batch_size, seq_len, hidden_size, num_heads = shape
    _validate_replay_weights(
        call.hidden_states,
        hidden_size,
        num_heads,
        call.qkv_weight,
        call.decay_a_weight,
        call.decay_b_weight,
        call.beta_weight,
        call.output_gate_weight,
        call.output_norm_weight,
        call.output_weight,
        call.conv_weight,
    )
    _validate_replay_state(
        call.hidden_states,
        batch_size,
        seq_len,
        num_heads,
        call.conv_state,
        call.recurrent_state,
        (call.replay_u, call.replay_k, call.replay_decay),
        call.a_log,
        call.dt_bias,
    )
    scalar_values = _validate_runtime_scalars(
        call.scale, call.lower_bound, call.rms_norm_eps
    )
    index_tensors = (
        ("conv_state_indices", call.conv_indices),
        ("conv_num_accepted_tokens", call.conv_accepted),
    )
    _validate_replay_indices(call.hidden_states.device, batch_size, index_tensors)
    launch_cores = _device_block_num(call.hidden_states)
    return shape, (*scalar_values, launch_cores)


def _replay_call_groups(call):
    weights = (
        call.qkv_weight,
        call.decay_a_weight,
        call.decay_b_weight,
        call.beta_weight,
        call.output_gate_weight,
        call.output_norm_weight,
        call.output_weight,
    )
    state_inputs = (
        call.a_log,
        call.dt_bias,
        call.recurrent_state,
        call.replay_u,
        call.replay_k,
        call.replay_decay,
    )
    conv_inputs = (
        call.conv_state,
        call.conv_indices,
        call.conv_accepted,
        call.conv_weight,
    )
    return weights, state_inputs, conv_inputs


def _execute_replay_call(call):
    shape, scalars = _validate_replay_call(call)
    weights, state_inputs, conv_inputs = _replay_call_groups(call)
    return _run_replay_launch(
        call.hidden_states, shape, weights, state_inputs, conv_inputs, scalars
    )


def mega_recurrent_kda_replayssm(
    hidden_states: torch.Tensor,
    qkv_projection_weight: torch.Tensor,
    decay_projection_a_weight: torch.Tensor,
    decay_projection_b_weight: torch.Tensor,
    beta_projection_weight: torch.Tensor,
    output_gate_projection_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
    output_projection_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    replay_u: torch.Tensor,
    replay_k: torch.Tensor,
    replay_decay: torch.Tensor,
    conv_state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    rms_norm_eps: float = 1e-5,
) -> torch.Tensor:
    """Run fused Mega KDA verification and save records for a later commit.

    Args:
        hidden_states (torch.Tensor): BF16, contiguous, shape ``[B,8,H]``;
            ``B`` is a multiple of 16 in ``[16,512]`` and ``H = 7168``.
        qkv_projection_weight (torch.Tensor): BF16 FRACTAL_NZ allocation-root
            tensor, shape ``[3*P,H] = [2304,7168]``, with output channels in
            ``[Q(P) | K(P) | V(P)]`` order.
        decay_projection_a_weight (torch.Tensor): BF16 FRACTAL_NZ allocation-root
            tensor, shape ``[D,H] = [128,7168]``; first decay projection, whose
            intermediate result is rounded to BF16.
        decay_projection_b_weight (torch.Tensor): BF16 FRACTAL_NZ allocation-root
            tensor, shape ``[P,D] = [768,128]``; second decay projection.
        beta_projection_weight (torch.Tensor): BF16 FRACTAL_NZ allocation-root
            tensor, shape ``[N,H] = [6,7168]``; produces one beta logit per head.
        output_gate_projection_weight (torch.Tensor): BF16 FRACTAL_NZ
            allocation-root tensor, shape ``[P,H] = [768,7168]``; produces the
            output gate logits.
        output_norm_weight (torch.Tensor): FP32, contiguous, shape ``[D] = [128]``; RMSNorm weight shared by all heads.
        output_projection_weight (torch.Tensor): BF16 FRACTAL_NZ allocation-root
            tensor, shape ``[H,P] = [7168,768]``; maps the gated head outputs
            back to hidden channels.
        conv1d_weight (torch.Tensor): BF16, contiguous, preferably shape
            ``[3,N,4,D] = [3,6,4,128]`` in Q/K/V, head, tap, channel order; also
            accepts ``[4,3*P]`` or head-shared ``[3,4,D]``, which are converted
            on every call without caching.
        conv_state (torch.Tensor): BF16, contiguous, shape ``[C,L,3*P]`` with
            ``C >= 1`` and ``L >= 10``; pooled Conv history updated in place,
            with slot 0 reserved as an immutable null slot.
        recurrent_state (torch.Tensor): FP32, contiguous, shape ``[B,N,D,D]``
            in batch, head, value-row, key-column order; read-only checkpoint
            used to evaluate every candidate token.
        replay_u (torch.Tensor): FP16, contiguous, shape ``[B,8,N,D]``; output
            buffer overwritten with the U record for every candidate token.
        replay_k (torch.Tensor): FP16, contiguous, shape ``[B,8,N,D]``; output
            buffer overwritten with the normalized K record for every candidate
            token.
        replay_decay (torch.Tensor): FP32, contiguous, shape ``[B,8,N,D]``;
            output buffer overwritten with the activated decay record for every
            candidate token.
        conv_state_indices (torch.Tensor): INT32, contiguous, shape ``[B]``;
            values in ``[0,C)`` select Conv slots, and 0 skips Conv state
            reads/writes and produces zero Conv output.
        conv_num_accepted_tokens (torch.Tensor): INT32, contiguous, shape
            ``[B]``; counts in ``[0,8]`` select Conv history at
            ``offset = max(count-1,0)``; each non-null slot requires
            ``offset+3 <= L``.
        a_log (torch.Tensor): FP32, contiguous, shape ``[N] = [6]``; per-head logarithmic decay parameters.
        dt_bias (torch.Tensor): FP32, contiguous, shape ``[N,D] = [6,128]``; per-channel decay bias.
        scale (float): Required finite query scale, passed as a runtime FP32
            scalar; use ``128 ** -0.5`` for the usual scaling.
        lower_bound (float): Required decay gate lower bound in ``[-5,0]``, passed as a runtime FP32 scalar.
        rms_norm_eps (float): Finite positive output RMSNorm epsilon, defaulting
            to ``1e-5`` and passed as a runtime FP32 scalar; Q/K normalization
            keeps epsilon 1e-6.
    Returns:
        torch.Tensor: Contiguous BF16 output with shape ``[B,8,H]`` on the input
            device; ``conv_state``, ``replay_u``, ``replay_k``, and
            ``replay_decay`` are updated in place, while ``recurrent_state``
            remains unchanged.

    Notes:
        * ``N = 6``, ``D = 128``, ``P = N*D = 768``, and Conv kernel size is 4; only BSH input layout is supported.
        * All tensors must be contiguous and on the same NPU, with the shapes
          and dtypes above; the six projection weights use FRACTAL_NZ storage
          with output-channel-first logical layout. Pass allocation-root exact
          tensors; detach Parameter weights before calling.
        * All eight proposal tokens are evaluated and written to the replay
          buffers. Sampling happens after this call; use
          ``commit_recurrent_kda_replayssm`` to apply only the accepted prefix.
        * A non-null Conv slot receives the last two selected history rows at
          ``[0:2]`` and raw projected QKV at ``[2:10]``; other rows and
          unselected slots remain unchanged. Nonzero Conv indices must be
          distinct across batch items.
        * Index/count values must satisfy the bounds above and are not copied to the host for validation.
        * The mixed kernel processes all runtime B16 groups directly. Changing B
          resizes scratch storage and reuses the compiled callable.
        * One compiled callable is retained without a cache key; scratch storage
          is reallocated when its device or shape changes. Overlapping calls
          require external serialization.
        * Eager calls use this host wrapper. ``torch.compile`` substitutes the
          registered dispatcher op, and the module also registers a native
          binary entry named ``mega_recurrent_kda_replayssm``.
    """
    call = _ReplayCall(
        hidden_states,
        qkv_projection_weight,
        decay_projection_a_weight,
        decay_projection_b_weight,
        beta_projection_weight,
        output_gate_projection_weight,
        output_norm_weight,
        output_projection_weight,
        conv1d_weight,
        conv_state,
        recurrent_state,
        replay_u,
        replay_k,
        replay_decay,
        conv_state_indices,
        conv_num_accepted_tokens,
        a_log,
        dt_bias,
        scale,
        lower_bound,
        rms_norm_eps,
    )
    return _execute_replay_call(call)


# ---------------------------------------------------------------------------
# Native export + graph-mode registration.
# ---------------------------------------------------------------------------
@export("mega_recurrent_kda_replayssm")
def export_mega_recurrent_kda_replayssm():
    """Export the S=8 megakernel served by the 8/16/24/28/32-core schedules."""
    op = MegaRecurrentKDAReplaySSMKernel(hidden_size=HIDDEN_SIZE, num_heads=6)
    bf16 = dtypes.bfloat16
    fp16 = dtypes.float16
    fp32 = dtypes.float32
    rows_dim = cannbotdsl.Dim("rows", min=16, max=4096, multiple_of=16)
    batch_dim = cannbotdsl.Dim("batch", min=16, max=512, multiple_of=16)
    conv_pool_dim = cannbotdsl.Dim("conv_pool")
    conv_tokens_dim = cannbotdsl.Dim("conv_tokens", min=3)
    group_rows_dim = cannbotdsl.Dim("group_rows", min=16, max=128, multiple_of=16)
    handles = []
    try:
        handles.append(
            cannbotdsl.compile(
                op.run,
                TensorSpec((rows_dim, HIDDEN_SIZE), bf16),
                *(
                    TensorSpec((rows_dim, width), bf16)
                    for width in op.workspace_region_widths
                ),
                TensorSpec((group_rows_dim, 128), fp32),
                TensorSpec((rows_dim, HIDDEN_SIZE), bf16),
                TensorSpec((2304, HIDDEN_SIZE), bf16, storage_format="nz"),
                TensorSpec((128, HIDDEN_SIZE), bf16, storage_format="nz"),
                TensorSpec((768, 128), bf16, storage_format="nz"),
                TensorSpec((6, HIDDEN_SIZE), bf16, storage_format="nz"),
                TensorSpec((768, HIDDEN_SIZE), bf16, storage_format="nz"),
                TensorSpec((128,), fp32),
                TensorSpec((HIDDEN_SIZE, 768), bf16, storage_format="nz"),
                TensorSpec((6,), fp32),
                TensorSpec((6, 128), fp32),
                TensorSpec((batch_dim, 6, 128, 128), fp32),
                TensorSpec((batch_dim, SEQ_LEN, 6, 128), fp16),
                TensorSpec((batch_dim, SEQ_LEN, 6, 128), fp16),
                TensorSpec((batch_dim, SEQ_LEN, 6, 128), fp32),
                TensorSpec((conv_pool_dim, conv_tokens_dim, 2304), bf16),
                TensorSpec((batch_dim,), dtypes.int32),
                TensorSpec((batch_dim,), dtypes.int32),
                TensorSpec((3, 6, 4, 128), bf16),
                fp32,
                fp32,
                fp32,
                dtypes.int64,
            )
        )
    finally:
        for handle in handles:
            handle.close()


_GRAPH_LIBRARY = torch.library.Library("cannbotdsl_mega_recurrent_kda_replayssm", "DEF")
_GRAPH_LIBRARY.define(
    "mega_recurrent_kda_replayssm("
    "Tensor hidden_states, "
    "Tensor qkv_projection_weight, "
    "Tensor decay_projection_a_weight, "
    "Tensor decay_projection_b_weight, "
    "Tensor beta_projection_weight, "
    "Tensor output_gate_projection_weight, "
    "Tensor output_norm_weight, "
    "Tensor output_projection_weight, "
    "Tensor conv1d_weight, "
    "Tensor(a!) conv_state, "
    "Tensor recurrent_state, "
    "Tensor(b!) replay_u, "
    "Tensor(c!) replay_k, "
    "Tensor(d!) replay_decay, "
    "Tensor conv_state_indices, "
    "Tensor conv_num_accepted_tokens, "
    "Tensor a_log, "
    "Tensor dt_bias, "
    "float scale, "
    "float lower_bound, "
    "float rms_norm_eps=1e-5"
    ") -> Tensor"
)


@torch.library.impl(_GRAPH_LIBRARY, "mega_recurrent_kda_replayssm", "Meta")
def _mega_recurrent_kda_replayssm_meta(hidden_states, *_args, **_kwargs):
    batch_size = int(hidden_states.shape[0])
    if (
        not BATCH_TILE <= batch_size <= BATCH_TILE * MAX_BATCH_GROUPS
        or batch_size % BATCH_TILE
    ):
        raise ValueError(
            "mega_recurrent_kda_replayssm requires B to be a multiple of 16 in [16, 512]"
        )
    return torch.empty_like(hidden_states, device="meta")


@torch.library.impl(_GRAPH_LIBRARY, "mega_recurrent_kda_replayssm", "PrivateUse1")
def _mega_recurrent_kda_replayssm_privateuse1(
    hidden_states,
    qkv_projection_weight,
    decay_projection_a_weight,
    decay_projection_b_weight,
    beta_projection_weight,
    output_gate_projection_weight,
    output_norm_weight,
    output_projection_weight,
    conv1d_weight,
    conv_state,
    recurrent_state,
    replay_u,
    replay_k,
    replay_decay,
    conv_state_indices,
    conv_num_accepted_tokens,
    a_log,
    dt_bias,
    scale,
    lower_bound,
    rms_norm_eps=1e-5,
):
    return mega_recurrent_kda_replayssm(
        hidden_states,
        qkv_projection_weight,
        decay_projection_a_weight,
        decay_projection_b_weight,
        beta_projection_weight,
        output_gate_projection_weight,
        output_norm_weight,
        output_projection_weight,
        conv1d_weight,
        conv_state,
        recurrent_state,
        replay_u,
        replay_k,
        replay_decay,
        conv_state_indices,
        conv_num_accepted_tokens,
        a_log,
        dt_bias,
        scale,
        lower_bound,
        rms_norm_eps,
    )


mega_recurrent_kda_replayssm_op = (
    torch.ops.cannbotdsl_mega_recurrent_kda_replayssm.mega_recurrent_kda_replayssm
)


# Old network code calls the public host function directly. Under Dynamo
# (torch.compile, fullgraph) that host body is untraceable, so during
# tracing the call is substituted with the registered dispatcher op;
# eager calls keep taking the host path unchanged.
getattr(torch, "_dynamo").substitute_in_graph(
    mega_recurrent_kda_replayssm, can_constant_fold_through=False
)(lambda *args, **kwargs: mega_recurrent_kda_replayssm_op(*args, **kwargs))
