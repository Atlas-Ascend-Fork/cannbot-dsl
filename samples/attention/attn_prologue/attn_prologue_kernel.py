# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""支持 token 数泛化的 MXFP8 attention prologue。

小 T 对 QA/KV 使用 split-K，在归一化前合并 FP32 部分和；大 T 使用有界的
token 分块，QB 按 token 分块和 head 分配任务。权重预先转换为 FRACTAL_NZ，
跨核数据通过 GM 和生产者发布同步传递。DSL 根据每次调用的 T 进行编译特化。
FixPipe 使用补齐的物理尾块，公开输出和 cache 更新仍使用逻辑 T。
"""

from __future__ import annotations
from cannbotdsl import from_tensor

import threading
from collections import OrderedDict

import torch

import cannbotdsl
import cannbotdsl.reg as rr
from cannbotdsl import (
    Buffer,
    const_expr,
    MemLoc,
    dtypes,
    get_block_idx,
    get_subblock_id,
    jit,
    kernel,
    make_layout,
    matmul,
    mem_copy,
    range_constexpr,
    tile_slice,
    vf,
)
from cannbotdsl.ops.memcpy import copy_options
from cannbotdsl.ring_buffer import RingBuffer

from cannbotdsl.ops.sync import (
    PIPE,
    cube_sync_all,
    vec_sync_all,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    vec_sync_block_arrive,
    vec_sync_block_wait,
)


from .attn_prologue_tiling import (
    A2_BUF_DEPTH_DECODE,
    A2_BUF_DEPTH_PREFILL,
    AIV_PER_BLOCK,
    AttnPrologueTiling,
    BN_B_FULL_HEAD,
    E4M3_MAX,
    E8M0_BIAS,
    FIXPIPE_ACCUMULATE,
    FIXPIPE_DUAL_DST,
    FIXPIPE_FINAL,
    FLOAT8_E4M3FN,
    FLOAT8_E8M0,
    FP32_EXP_BIAS,
    FP32_EXP_SHIFT,
    L0_DEPTH,
    MX_GROUP,
    MX_PAIR,
    MX_PAIR_WIDTH,
    NZ_M_FRAC,
    PANEL_OUTPUT_TILES,
    RESIDENT_HEAD_PANELS,
    RESIDENT_K_WINDOWS,
    UB_ALIGN,
    VL,
    W_BUF,
    ZN_FRACTAL_BYTES,
    _ceil_div,
    _kv_row_bytes,
)


@kernel
class AttnPrologueKernel:
    """通过一次 kernel 启动完成整个 attention prologue。

    方法按调度归属分三组，公共部分集中在共享组：
    - 共享：_qr_tile / _kv_tile / _quantize_qr / _quantize_kv_group /
      _finish_kv_row / _finish_q / _rope_pair / _pack_scale / _shared_exponent /
      _to_e4m3 / _fill_w_a / _fill_w_b / _acc_* / _b_vec / _publish_qr。
    - decode（l1_reuse）：_shared_pipeline / _resident_a_joint* /
      _resident_a_project* / _stage_b_panels* / _resident_b_cube* /
      _b_panel_l0_reuse / _load_b_panel。
    - prefill（split_t 与通用路径）：_a1_branch / _a2_branch / _project_qa_kv /
      _a1_tile / _mm_a / _stage_b / _b_cube / _mm_window_b / _service_kv。
    """

    def __init__(self, tiling: AttnPrologueTiling):
        self.tiling = tiling

    # kernel 入口
    def __call__(
        self,
        gm_x,
        gm_dsx,
        gm_wqa,
        gm_dswqa,
        gm_wqb,
        gm_dswqb,
        gm_wkv,
        gm_dswkv,
        gm_gamma_qr,
        gm_gamma_kv,
        gm_sin,
        gm_cos,
        gm_index,
        gm_cache,
        gm_q,
        gm_qr,
        gm_dqr2d,
        gm_dqr3d,
        gm_ws_qa,
        gm_ws_kv,
        norm_eps: dtypes.float32,
        gm_qr_bf16,
    ):
        # Vector 归约前发布 Cube 部分和，Cube 执行 QB 前发布 Vector 的 QR。
        if const_expr(self.tiling.l1_reuse):
            self._shared_pipeline(
                gm_x,
                gm_dsx,
                gm_wqa,
                gm_dswqa,
                gm_wqb,
                gm_dswqb,
                gm_wkv,
                gm_dswkv,
                gm_gamma_qr,
                gm_gamma_kv,
                gm_sin,
                gm_cos,
                gm_index,
                gm_cache,
                gm_q,
                gm_qr,
                gm_dqr2d,
                gm_dqr3d,
                gm_ws_qa,
                gm_ws_kv,
                norm_eps,
                gm_qr_bf16,
            )
        elif const_expr(self.tiling.use_branch_pipeline):
            self._a1_branch(
                gm_x,
                gm_dsx,
                gm_wqa,
                gm_dswqa,
                gm_wkv,
                gm_dswkv,
                gm_ws_qa,
                gm_ws_kv,
                gm_gamma_qr,
                gm_index,
                gm_qr,
                gm_dqr2d,
                gm_qr_bf16,
                norm_eps,
            )
            # Vector 完成 QA 后处理时预取第一个 QB 权重窗口。
            # KV 后处理与 QB 结果消费交错执行。
            prefetched_qb_weights, prefetched_qb_scales = self._prefetch_qb_weights(
                gm_wqb, gm_dswqb
            )
            self._stage_b(
                gm_qr,
                gm_dqr3d,
                gm_wqb,
                gm_dswqb,
                gm_sin,
                gm_cos,
                gm_q,
                prefetched_qb_weights,
                prefetched_qb_scales,
                gm_gamma_kv,
                gm_index,
                gm_cache,
                gm_ws_kv,
                norm_eps,
            )
        else:
            self._project_qa_kv(
                gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_wkv, gm_dswkv, gm_ws_qa, gm_ws_kv
            )
            self._publish_a1()
            # Vector 完成 QA 后处理时预取第一个 QB 权重窗口。
            prefetched_qb_weights, prefetched_qb_scales = self._prefetch_qb_weights(
                gm_wqb, gm_dswqb
            )
            self._reduce_normalize_a(
                gm_gamma_qr,
                gm_gamma_kv,
                gm_sin,
                gm_cos,
                gm_index,
                gm_cache,
                gm_qr,
                gm_dqr2d,
                gm_ws_qa,
                gm_ws_kv,
                norm_eps,
                gm_qr_bf16,
            )
            self._stage_b(
                gm_qr,
                gm_dqr3d,
                gm_wqb,
                gm_dswqb,
                gm_sin,
                gm_cos,
                gm_q,
                prefetched_qb_weights,
                prefetched_qb_scales,
                gm_gamma_kv,
                gm_index,
                gm_cache,
                gm_ws_kv,
                norm_eps,
            )

    def _prefetch_qb_weights(self, gm_wqb, gm_dswqb):
        """预取第一个 QB 权重窗口；非驻留路径返回空占位。"""
        if const_expr(self.tiling.b_resident):
            weights = RingBuffer(
                MemLoc.L1,
                dtypes.int8,
                make_layout((self.tiling.bn_b, self.tiling.k_l1_b)),
                depth=1,
                data_format="nz",
            )
            scales = RingBuffer(
                MemLoc.L1,
                FLOAT8_E8M0,
                make_layout((self.tiling.k_l1_b // MX_GROUP, self.tiling.bn_b)),
                depth=2,
                data_format="nz",
            )
            self._fill_w_b(
                weights,
                scales,
                gm_wqb,
                gm_dswqb,
                get_block_idx(),
                0,
                self.tiling.k_l1_b // MX_PAIR_WIDTH,
            )
            return weights, scales
        return None, None

    @jit
    def _shared_pipeline(
        self,
        gm_x,
        gm_dsx,
        gm_wqa,
        gm_dswqa,
        gm_wqb,
        gm_dswqb,
        gm_wkv,
        gm_dswkv,
        gm_gamma_qr,
        gm_gamma_kv,
        gm_sin,
        gm_cos,
        gm_index,
        gm_cache,
        gm_q,
        gm_qr,
        gm_dqr2d,
        gm_dqr3d,
        gm_ws_qa,
        gm_ws_kv,
        norm_eps: dtypes.float32,
        gm_qr_bf16,
    ):
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (
            tiling.bm_a1,
            tiling.bn_a1,
            tiling.bk_a1,
        )
        panels_per_head = tiling.head_size // (PANEL_OUTPUT_TILES * output_tile_cols)
        # round3：两个 token 分块共用一次静态权重装载（twin）。
        twin_blocks = 2 if tiling.owners_b == 2 else 1
        prefetch_windows = 2 if twin_blocks == 2 else RESIDENT_K_WINDOWS
        seed_windows = 1 if twin_blocks == 2 else RESIDENT_K_WINDOWS
        b_groups = tiling.groups_b_pair if twin_blocks == 2 else tiling.groups_b
        l1_x = [
            RingBuffer(
                MemLoc.L1,
                dtypes.float8_e4m3fn,
                make_layout((token_tile_rows, reduction_tile_cols)),
                depth=1,
                data_format="nz",
            )
            for _ in range(tiling.resident_tiles * twin_blocks)
        ]
        # E8M0 scale 的 zn 分形（(M1, G1, 16, 2)、每分形 32 字节）不能用
        # 每次选槽时用 from_tensor 绑定 zn 视图。
        l1_sx = [
            RingBuffer(
                MemLoc.L1,
                dtypes.uint8,
                make_layout(
                    (
                        _ceil_div(token_tile_rows, NZ_M_FRAC)
                        * _ceil_div(
                            tiling.scale_k_tiles * reduction_tile_cols // MX_GROUP,
                            MX_PAIR,
                        )
                        * ZN_FRACTAL_BYTES,
                    )
                ),
                depth=1,
                data_format="nd",
            )
            for _ in range(tiling.resident_scale_tiles * twin_blocks)
        ]
        l1_w = [
            RingBuffer(
                MemLoc.L1,
                dtypes.int8,
                make_layout((output_tile_cols, reduction_tile_cols)),
                depth=1,
                data_format="nz",
            )
            for _ in range(W_BUF)
        ]
        l1_sw = RingBuffer(
            MemLoc.L1,
            dtypes.float8_e8m0,
            make_layout((reduction_tile_cols // MX_GROUP, output_tile_cols)),
            depth=W_BUF,
            data_format="nz",
        )
        l0a = [
            RingBuffer(
                MemLoc.L0A,
                dtypes.float8_e4m3fn,
                make_layout((token_tile_rows, reduction_tile_cols)),
                depth=1,
            )
            for _ in range(L0_DEPTH)
        ]
        l0b = RingBuffer(
            MemLoc.L0B,
            dtypes.float8_e4m3fn,
            make_layout((output_tile_cols, reduction_tile_cols)),
            depth=L0_DEPTH,
        )
        l0c_a = [
            Buffer(
                MemLoc.L0C,
                dtypes.float32,
                make_layout((token_tile_rows, output_tile_cols)),
            )
            for _ in range(twin_blocks)
        ]
        # A 与 B 复用相同的物理 Channel，因此此部分容量按较大者计算。
        if const_expr(twin_blocks == 2):
            self._resident_a_joint_twin(
                l1_x,
                l1_sx,
                l1_w,
                l1_sw,
                l0a,
                l0b,
                l0c_a,
                gm_x,
                gm_dsx,
                gm_wqa,
                gm_dswqa,
                gm_ws_qa,
                gm_wkv,
                gm_dswkv,
                gm_ws_kv,
            )
        else:
            self._resident_a_joint(
                l1_x,
                l1_sx,
                l1_w,
                l1_sw,
                l0a,
                l0b,
                l0c_a[0],
                gm_x,
                gm_dsx,
                gm_wqa,
                gm_dswqa,
                gm_ws_qa,
                gm_wkv,
                gm_dswkv,
                gm_ws_kv,
            )
        # 发布 A1 交换区后，可在 AIV 归约期间加载 B panel。
        early_s = [
            RingBuffer(
                MemLoc.L1,
                dtypes.float8_e8m0,
                make_layout((reduction_tile_cols // MX_GROUP, tiling.head_size)),
                depth=1,
                data_format="nz",
            )
            for _ in range(RESIDENT_K_WINDOWS)
        ]
        early_w = [
            RingBuffer(
                MemLoc.L1,
                dtypes.int8,
                make_layout(
                    (PANEL_OUTPUT_TILES * output_tile_cols, reduction_tile_cols)
                ),
                depth=1,
                data_format="nz",
            )
            for _ in range(prefetch_windows)
        ]
        cube_sync_all()
        vec_sync_all()
        vec_sync_block_arrive(PIPE.MTE3, 0, mode=2)
        cube_sync_block_wait(PIPE.S, 0, mode=2)
        cube_sync_block_arrive(PIPE.FIXPIPE, 1, mode=0)
        if get_block_idx() < tiling.owners_b * b_groups:
            if const_expr(twin_blocks == 2):
                seed_task = get_block_idx()
                seed_head = seed_task // panels_per_head
                mem_copy(
                    early_s[0].next_write(),
                    tile_slice(
                        gm_dswqb,
                        (
                            tiling.head_size,
                            reduction_tile_cols // MX_PAIR_WIDTH,
                            MX_PAIR,
                        ),
                        (seed_head, 0, 0),
                    )
                    .permute(1, 2, 0)
                    .flatten(0, 1),
                )
                self._load_b_panel(early_w[0].next_write(), gm_wqb, seed_task, 0)
            else:
                seed_unit = get_block_idx() % b_groups
                head = seed_unit // panels_per_head
                mem_copy(
                    early_s[0].next_write(),
                    tile_slice(
                        gm_dswqb,
                        (
                            tiling.head_size,
                            reduction_tile_cols // MX_PAIR_WIDTH,
                            MX_PAIR,
                        ),
                        (head, 0, 0),
                    )
                    .permute(1, 2, 0)
                    .flatten(0, 1),
                )
                self._load_b_panel(
                    early_w[0].next_write(),
                    gm_wqb,
                    seed_unit,
                    0,
                )
        cube_sync_block_wait(PIPE.S, 1, mode=0)
        vec_sync_block_arrive(PIPE.MTE3, 1, mode=0)
        vec_sync_block_wait(PIPE.S, 1, mode=0)
        cube_sync_block_arrive(PIPE.MTE3, 2, mode=2)
        vec_sync_block_wait(PIPE.S, 2, mode=2)
        if get_block_idx() < tiling.owners_b * b_groups:
            for u in range_constexpr(1, seed_windows):
                mem_copy(
                    early_s[u].next_write(),
                    tile_slice(
                        gm_dswqb,
                        (
                            tiling.head_size,
                            reduction_tile_cols // MX_PAIR_WIDTH,
                            MX_PAIR,
                        ),
                        ((get_block_idx() % b_groups) // panels_per_head, u, 0),
                    )
                    .permute(1, 2, 0)
                    .flatten(0, 1),
                )
                self._load_b_panel(
                    early_w[u].next_write(),
                    gm_wqb,
                    get_block_idx() % b_groups,
                    u,
                )
        self._reduce_normalize_a(
            gm_gamma_qr,
            gm_gamma_kv,
            gm_sin,
            gm_cos,
            gm_index,
            gm_cache,
            gm_qr,
            gm_dqr2d,
            gm_ws_qa,
            gm_ws_kv,
            norm_eps,
            gm_qr_bf16,
        )
        if const_expr(twin_blocks == 2):
            self._stage_b_panels_twin(
                l1_x,
                l1_sx,
                l0a,
                l0b,
                gm_qr,
                gm_dqr3d,
                gm_wqb,
                gm_dswqb,
                gm_sin,
                gm_cos,
                gm_q,
                early_w,
                early_s,
            )
        else:
            self._stage_b_panels(
                l1_x,
                l1_sx,
                l0a,
                l0b,
                gm_qr,
                gm_dqr3d,
                gm_wqb,
                gm_dswqb,
                gm_sin,
                gm_cos,
                gm_q,
                early_w,
                early_s,
            )

    def _load_activation_block(
        self, l1_x, l1_sx, gm_x, gm_dsx, token_tile_idx, split, x_offset, s_offset
    ):
        """加载一个 token 分块的激活 payload/scale，返回 L1 读视图。"""
        tiling = self.tiling
        token_tile_rows, reduction_tile_cols = tiling.bm_a1, tiling.bk_a1
        nk = tiling.hidden_size // tiling.split_k // reduction_tile_cols
        tile_x = []
        for reduction_tile_idx in range_constexpr(nk):
            activation_l1_slot = l1_x[x_offset + reduction_tile_idx].next_write()
            mem_copy(
                activation_l1_slot,
                tile_slice(
                    gm_x,
                    (token_tile_rows, reduction_tile_cols),
                    (token_tile_idx, split * nk + reduction_tile_idx),
                ),
            )
            tile_x.append(l1_x[x_offset + reduction_tile_idx].next_read())
        tile_s = []
        for scale_tile_idx in range_constexpr(nk // tiling.scale_k_tiles):
            scale_slot = l1_sx[s_offset + scale_tile_idx].next_write()
            activation_scale_slot = from_tensor(
                scale_slot,
                dtype=FLOAT8_E8M0,
                layout=make_layout(
                    (
                        token_tile_rows,
                        tiling.scale_k_tiles * reduction_tile_cols // MX_GROUP,
                    )
                ),
                data_format="zn",
            )
            mem_copy(
                activation_scale_slot,
                tile_slice(
                    gm_dsx,
                    (
                        token_tile_rows,
                        tiling.scale_k_tiles * reduction_tile_cols // MX_PAIR_WIDTH,
                        MX_PAIR,
                    ),
                    (
                        token_tile_idx,
                        split * (nk // tiling.scale_k_tiles) + scale_tile_idx,
                        0,
                    ),
                ).flatten(1, 2),
            )
            tile_s.append(
                from_tensor(
                    scale_slot,
                    dtype=FLOAT8_E8M0,
                    layout=make_layout(
                        (
                            token_tile_rows,
                            tiling.scale_k_tiles * reduction_tile_cols // MX_GROUP,
                        )
                    ),
                    data_format="zn",
                )
            )
        return tile_x, tile_s

    @jit
    def _resident_a_joint(
        self,
        l1_x,
        l1_sx,
        l1_w,
        l1_sw,
        l0a,
        l0b,
        l0c,
        gm_x,
        gm_dsx,
        gm_wqa,
        gm_dswqa,
        gm_ws_qa,
        gm_wkv,
        gm_dswkv,
        gm_ws_kv,
    ):
        tiling = self.tiling
        # 同一个激活块服务于 QA 和 KV 的多个输出列任务。
        groups = tiling.groups_a
        n_count = tiling.n_tiles_qa + tiling.n_tiles_kv
        for job in range(
            get_block_idx(),
            tiling.m_tiles_a1 * tiling.split_k * groups,
            tiling.block_dim,
        ):
            token_tile_idx = job // (tiling.split_k * groups)
            split = job // groups % tiling.split_k
            group = job % groups
            held_x, held_s = self._load_activation_block(
                l1_x, l1_sx, gm_x, gm_dsx, token_tile_idx, split, 0, 0
            )
            for output_tile_idx in range(group, n_count, groups):
                if output_tile_idx < tiling.n_tiles_qa:
                    self._resident_a_project(
                        l1_w,
                        l1_sw,
                        l0a,
                        l0b,
                        l0c,
                        held_x,
                        held_s,
                        gm_wqa,
                        gm_dswqa,
                        gm_ws_qa,
                        output_tile_idx,
                        token_tile_idx,
                        split,
                    )
                else:
                    self._resident_a_project(
                        l1_w,
                        l1_sw,
                        l0a,
                        l0b,
                        l0c,
                        held_x,
                        held_s,
                        gm_wkv,
                        gm_dswkv,
                        gm_ws_kv,
                        output_tile_idx - tiling.n_tiles_qa,
                        token_tile_idx,
                        split,
                    )

    @jit
    def _resident_a_project(
        self,
        l1_w,
        l1_sw,
        l0a,
        l0b,
        l0c,
        held_x,
        held_s,
        gm_w,
        gm_dsw,
        gm_ws,
        output_tile_idx,
        token_tile_idx,
        split,
    ):
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (
            tiling.bm_a1,
            tiling.bn_a1,
            tiling.bk_a1,
        )
        nk = tiling.hidden_size // tiling.split_k // reduction_tile_cols
        for pair in range(_ceil_div(nk, W_BUF)):
            for u in range_constexpr(min(W_BUF, nk)):
                reduction_tile_idx = pair * W_BUF + u
                mem_copy(
                    l0a[u].next_write(),
                    held_x[reduction_tile_idx],
                    copy_options(
                        mx_scale=tile_slice(
                            held_s[reduction_tile_idx // tiling.scale_k_tiles],
                            (token_tile_rows, reduction_tile_cols // MX_GROUP),
                            (0, reduction_tile_idx % tiling.scale_k_tiles),
                        )
                    ),
                )
                self._fill_w_a(
                    l1_w[u],
                    l1_sw,
                    gm_w,
                    gm_dsw,
                    output_tile_idx,
                    split * nk + reduction_tile_idx,
                    reduction_tile_cols // MX_PAIR_WIDTH,
                )
            for u in range_constexpr(min(W_BUF, nk)):
                reduction_tile_idx = pair * W_BUF + u
                weight_l1_tile = from_tensor(l1_w[u].next_read(), dtypes.float8_e4m3fn)
                weight_scale_tile = l1_sw.next_read()
                mem_copy(
                    l0b.next_write(),
                    weight_l1_tile,
                    copy_options(mx_scale=weight_scale_tile),
                )
                weight_l0_tile = l0b.next_read()
                matmul(
                    l0c,
                    l0a[u].next_read(),
                    weight_l0_tile,
                    init=reduction_tile_idx == 0,
                    mx_mode=True,
                )
        #  mem_copy 以源域为准：将 L0C 源逻辑域重绑定到目的实际域，
        # 尾块裁剪生效且物理布局保持不变。
        ws_tile = tile_slice(
            gm_ws,
            (token_tile_rows, output_tile_cols),
            (split * tiling.m_tiles_a1 + token_tile_idx, output_tile_idx),
        )
        mem_copy(
            ws_tile,
            from_tensor(
                l0c,
                layout=make_layout(ws_tile.shape),
                physical_layout=l0c.physical_layout,
            ),
        )

    @jit
    def _resident_a_joint_twin(
        self,
        l1_x,
        l1_sx,
        l1_w,
        l1_sw,
        l0a,
        l0b,
        l0c,
        gm_x,
        gm_dsx,
        gm_wqa,
        gm_dswqa,
        gm_ws_qa,
        gm_wkv,
        gm_dswkv,
        gm_ws_kv,
    ):
        """两个 token 分块共用一次静态权重装载：job=(split, group)，token 分块内层。"""
        tiling = self.tiling
        nk = tiling.hidden_size // tiling.split_k // tiling.bk_a1
        groups = tiling.groups_a
        n_count = tiling.n_tiles_qa + tiling.n_tiles_kv
        for job in range(get_block_idx(), tiling.split_k * groups, tiling.block_dim):
            split = job // groups
            group = job % groups
            held_x = []
            held_s = []
            for tt in range_constexpr(2):
                tile_x, tile_s = self._load_activation_block(
                    l1_x,
                    l1_sx,
                    gm_x,
                    gm_dsx,
                    tt,
                    split,
                    tt * nk,
                    tt * tiling.resident_scale_tiles,
                )
                held_x.append(tile_x)
                held_s.append(tile_s)
            for output_tile_idx in range(group, n_count, groups):
                if output_tile_idx < tiling.n_tiles_qa:
                    self._resident_a_project_twin(
                        l1_w,
                        l1_sw,
                        l0a,
                        l0b,
                        l0c,
                        held_x,
                        held_s,
                        gm_wqa,
                        gm_dswqa,
                        gm_ws_qa,
                        output_tile_idx,
                        split,
                    )
                else:
                    self._resident_a_project_twin(
                        l1_w,
                        l1_sw,
                        l0a,
                        l0b,
                        l0c,
                        held_x,
                        held_s,
                        gm_wkv,
                        gm_dswkv,
                        gm_ws_kv,
                        output_tile_idx - tiling.n_tiles_qa,
                        split,
                    )

    @jit
    def _resident_a_project_twin(
        self,
        l1_w,
        l1_sw,
        l0a,
        l0b,
        l0c,
        held_x,
        held_s,
        gm_w,
        gm_dsw,
        gm_ws,
        output_tile_idx,
        split,
    ):
        """每个 K 窗口权重只装一次，对两个 token 分块各做一次 matmul。"""
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (
            tiling.bm_a1,
            tiling.bn_a1,
            tiling.bk_a1,
        )
        nk = tiling.hidden_size // tiling.split_k // reduction_tile_cols
        for pair in range(_ceil_div(nk, W_BUF)):
            for u in range_constexpr(min(W_BUF, nk)):
                reduction_tile_idx = pair * W_BUF + u
                self._fill_w_a(
                    l1_w[u],
                    l1_sw,
                    gm_w,
                    gm_dsw,
                    output_tile_idx,
                    split * nk + reduction_tile_idx,
                    reduction_tile_cols // MX_PAIR_WIDTH,
                )
            for u in range_constexpr(min(W_BUF, nk)):
                reduction_tile_idx = pair * W_BUF + u
                weight_l1_tile = from_tensor(l1_w[u].next_read(), dtypes.float8_e4m3fn)
                weight_scale_tile = l1_sw.next_read()
                mem_copy(
                    l0b.next_write(),
                    weight_l1_tile,
                    copy_options(mx_scale=weight_scale_tile),
                )
                weight_l0_tile = l0b.next_read()
                for tt in range_constexpr(2):
                    mem_copy(
                        l0a[tt].next_write(),
                        held_x[tt][reduction_tile_idx],
                        copy_options(
                            mx_scale=tile_slice(
                                held_s[tt][reduction_tile_idx // tiling.scale_k_tiles],
                                (token_tile_rows, reduction_tile_cols // MX_GROUP),
                                (0, reduction_tile_idx % tiling.scale_k_tiles),
                            )
                        ),
                    )
                    matmul(
                        l0c[tt],
                        l0a[tt].next_read(),
                        weight_l0_tile,
                        init=reduction_tile_idx == 0,
                        mx_mode=True,
                    )
        for tt in range_constexpr(2):
            ws_tile = tile_slice(
                gm_ws,
                (token_tile_rows, output_tile_cols),
                (split * tiling.m_tiles_a1 + tt, output_tile_idx),
            )
            mem_copy(
                ws_tile,
                from_tensor(
                    l0c[tt],
                    layout=make_layout(ws_tile.shape),
                    physical_layout=l0c[tt].physical_layout,
                ),
            )

    # 驻留路径保留既有的从左到右求和顺序；通用 QA 路径使用补偿求和。
    @jit
    def _acc_all_partials(self, dst, src, width):
        with vf(mode="simd"):
            # 1. 读取各特征向量，并累加 split-K 部分和。
            mask = rr.full_mask()
            for feature_vector_idx in range(width // VL):
                value = rr.vload(src, feature_vector_idx * VL)
                for part in range(1, self.tiling.split_k):
                    value = rr.vadd(
                        value,
                        rr.vload(
                            src,
                            part * src.physical_layout.stride[0]
                            + feature_vector_idx * VL,
                        ),
                        mask=mask,
                    )
                rr.vstore(dst, feature_vector_idx * VL, value, mask)

    @jit
    def _load_b_panel(self, dst, gm_weight, n_group, k_panel):
        tiling = self.tiling
        panel_bytes = PANEL_OUTPUT_TILES * tiling.bn_b * tiling.bk_b
        raw_dst = from_tensor(
            dst, layout=make_layout((1, panel_bytes)), data_format="nd"
        )
        mem_copy(
            raw_dst,
            tile_slice(
                gm_weight,
                (1, panel_bytes),
                (n_group * (tiling.rank_size // tiling.bk_b) + k_panel, 0),
            ),
        )

    @jit
    def _stage_b_panels(
        self,
        l1_x,
        l1_sx,
        l0a,
        l0b,
        gm_qr,
        gm_dqr,
        gm_w,
        gm_dsw,
        gm_sin,
        gm_cos,
        gm_q,
        early_w,
        early_s,
    ):
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (
            tiling.bm_b,
            tiling.bn_b,
            tiling.bk_b,
        )
        panels_per_head = tiling.head_size // (PANEL_OUTPUT_TILES * output_tile_cols)
        l0c = Buffer(
            MemLoc.L0C, dtypes.float32, make_layout((token_tile_rows, tiling.head_size))
        )
        half, feature_cols = (
            token_tile_rows // FIXPIPE_DUAL_DST,
            tiling.head_size // FIXPIPE_DUAL_DST,
        )
        ub_acc = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((half, feature_cols)),
            depth=2,
        )
        ub_out = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((half, feature_cols)), depth=2
        )
        ub_tail = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, VL)))
        ub_sin = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((half, VL)), depth=1)
        ub_cos = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((half, VL)), depth=1)
        groups = tiling.groups_b
        for job in range(get_block_idx(), tiling.owners_b * groups, tiling.block_dim):
            owner = job // groups
            group = job % groups
            # 首个任务已在 A2 期间预取；后续 token 分块需先完成前一个 head
            # 循环，再重新填充所有 panel Channel。
            if job >= tiling.block_dim:
                for u in range_constexpr(RESIDENT_K_WINDOWS):
                    mem_copy(
                        early_s[u].next_write(),
                        tile_slice(
                            gm_dsw,
                            (
                                tiling.head_size,
                                reduction_tile_cols // MX_PAIR_WIDTH,
                                MX_PAIR,
                            ),
                            (group // panels_per_head, u, 0),
                        )
                        .permute(1, 2, 0)
                        .flatten(0, 1),
                    )
                    self._load_b_panel(
                        early_w[u].next_write(),
                        gm_w,
                        group,
                        u,
                    )
            sin_slot = ub_sin.next_write()
            mem_copy(
                sin_slot,
                tile_slice(
                    gm_sin, (half, VL), (owner * AIV_PER_BLOCK + get_subblock_id(), 0)
                ),
            )
            cos_slot = ub_cos.next_write()
            mem_copy(
                cos_slot,
                tile_slice(
                    gm_cos, (half, VL), (owner * AIV_PER_BLOCK + get_subblock_id(), 0)
                ),
            )
            sin_tab = ub_sin.next_read()
            cos_tab = ub_cos.next_read()
            self._resident_b_cube(
                l1_x,
                l1_sx,
                l0a,
                l0b,
                l0c,
                ub_acc,
                gm_qr,
                gm_dqr,
                gm_w,
                gm_dsw,
                owner,
                group,
                groups,
                ub_out,
                ub_tail,
                sin_tab,
                cos_tab,
                gm_sin,
                gm_cos,
                gm_q,
                early_w,
                early_s,
            )

    def _load_qr_partials(self, l1_x, l1_sx, gm_qr, gm_dqr, owner, x_offset, s_offset):
        """加载一个 token 分块的 QR payload/scale，返回 L1 读视图。"""
        tiling = self.tiling
        token_tile_rows, reduction_tile_cols = tiling.bm_b, tiling.bk_b
        nk = tiling.rank_size // reduction_tile_cols
        held_x = []
        held_s = []
        for k in range_constexpr(nk):
            qr_payload_slot = l1_x[x_offset + k].next_write()
            mem_copy(
                qr_payload_slot,
                tile_slice(gm_qr, (token_tile_rows, reduction_tile_cols), (owner, k)),
            )
            held_x.append(l1_x[x_offset + k].next_read())
        for sk in range_constexpr(nk // tiling.scale_k_tiles):
            scale_slot = l1_sx[s_offset + sk].next_write()
            qr_scale_slot = from_tensor(
                scale_slot,
                dtype=FLOAT8_E8M0,
                layout=make_layout(
                    (
                        token_tile_rows,
                        tiling.scale_k_tiles * reduction_tile_cols // MX_GROUP,
                    )
                ),
                data_format="zn",
            )
            mem_copy(
                qr_scale_slot,
                tile_slice(
                    gm_dqr,
                    (
                        token_tile_rows,
                        tiling.scale_k_tiles * reduction_tile_cols // MX_PAIR_WIDTH,
                        MX_PAIR,
                    ),
                    (owner, sk, 0),
                ).flatten(1, 2),
            )
            held_s.append(
                from_tensor(
                    scale_slot,
                    dtype=FLOAT8_E8M0,
                    layout=make_layout(
                        (
                            token_tile_rows,
                            tiling.scale_k_tiles * reduction_tile_cols // MX_GROUP,
                        )
                    ),
                    data_format="zn",
                )
            )
        return held_x, held_s

    @jit
    def _resident_b_cube(
        self,
        l1_x,
        l1_sx,
        l0a,
        l0b,
        l0c,
        ub_acc,
        gm_qr,
        gm_dqr,
        gm_w,
        gm_dsw,
        owner,
        group,
        groups,
        ub_out,
        ub_tail,
        ub_sin,
        ub_cos,
        gm_sin,
        gm_cos,
        gm_q,
        early_w,
        early_s,
    ):
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (
            tiling.bm_b,
            tiling.bn_b,
            tiling.bk_b,
        )
        nk = tiling.rank_size // reduction_tile_cols
        panels_per_head = tiling.head_size // (PANEL_OUTPUT_TILES * output_tile_cols)
        l1_head_scale = early_s
        l1_b_wide = early_w
        held_x, held_s = self._load_qr_partials(l1_x, l1_sx, gm_qr, gm_dqr, owner, 0, 0)
        # round6：非 twin B 的搬运任务粒度从整 head 改为 (head, half) 半头任务，
        # 使 64 head / 28 核 的 3:2 不均衡压到 5:4 半头（critic 核搬运 -16.7%）。
        unit_total = tiling.head_count * panels_per_head
        for unit in range(group, unit_total, groups):
            head = unit // panels_per_head
            half = unit % panels_per_head
            for panel in range_constexpr(nk):
                head_scale = l1_head_scale[panel % nk].next_read()
                wide_w = from_tensor(
                    l1_b_wide[panel % RESIDENT_K_WINDOWS].next_read(),
                    dtypes.float8_e4m3fn,
                )
                self._b_panel_l0_reuse(
                    wide_w,
                    head_scale,
                    l0a,
                    l0b,
                    l0c,
                    held_x,
                    held_s,
                    half,
                    panel,
                )
                next_unit = unit + groups
                if next_unit < unit_total:
                    mem_copy(
                        l1_head_scale[panel % nk].next_write(),
                        tile_slice(
                            gm_dsw,
                            (
                                tiling.head_size,
                                reduction_tile_cols // MX_PAIR_WIDTH,
                                MX_PAIR,
                            ),
                            (next_unit // panels_per_head, panel, 0),
                        )
                        .permute(1, 2, 0)
                        .flatten(0, 1),
                    )
                    self._load_b_panel(
                        l1_b_wide[panel % RESIDENT_K_WINDOWS].next_write(),
                        gm_w,
                        next_unit,
                        panel,
                    )
                if const_expr(panel + 1 == nk):
                    qb_accumulator_slot = ub_acc.next_write()
                    # FixPipe 双目的地（split-M）：L0C 源 partition(dim=0) 将
                    # 行数对半分给两个 Vector 核。
                    mem_copy(
                        qb_accumulator_slot,
                        from_tensor(
                            tile_slice(
                                l0c,
                                (
                                    token_tile_rows,
                                    PANEL_OUTPUT_TILES * output_tile_cols,
                                ),
                                (0, half),
                            ),
                            layout=make_layout(
                                (
                                    token_tile_rows,
                                    PANEL_OUTPUT_TILES * output_tile_cols,
                                )
                            ),
                        ).partition(dim=0),
                        copy_options(unit_flag=FIXPIPE_FINAL),
                    )
                    self._b_vec(
                        ub_acc,
                        ub_out,
                        ub_tail,
                        ub_sin,
                        ub_cos,
                        gm_sin,
                        gm_cos,
                        gm_q,
                        head * tiling.owners_b + owner,
                        half,
                    )

    @jit
    def _stage_b_panels_twin(
        self,
        l1_x,
        l1_sx,
        l0a,
        l0b,
        gm_qr,
        gm_dqr,
        gm_w,
        gm_dsw,
        gm_sin,
        gm_cos,
        gm_q,
        early_w,
        early_s,
    ):
        """两个 token 分块共用一次 QB panel 装载：job=(head group)，token 分块内层。"""
        tiling = self.tiling
        token_tile_rows, output_tile_cols, _reduction_tile_cols = (
            tiling.bm_b,
            tiling.bn_b,
            tiling.bk_b,
        )
        panels_per_head = tiling.head_size // (PANEL_OUTPUT_TILES * output_tile_cols)
        l0c = [
            Buffer(
                MemLoc.L0C,
                dtypes.float32,
                make_layout((token_tile_rows, PANEL_OUTPUT_TILES * output_tile_cols)),
            )
            for _ in range(2)
        ]
        half, feature_cols = (
            token_tile_rows // FIXPIPE_DUAL_DST,
            tiling.head_size // FIXPIPE_DUAL_DST,
        )
        ub_acc = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((half, feature_cols)),
            depth=2,
        )
        ub_out = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((half, feature_cols)), depth=2
        )
        ub_tail = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, VL)))
        ub_sin = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((half, VL)), depth=2)
        ub_cos = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((half, VL)), depth=2)
        # round4：twin B 任务粒度改为 (head, half) 的 128 个 half-head task，按 28 核轮转，
        # 把 8 核 3 head / 20 核 2 head 压到 16 核 5 half / 12 核 4 half（tail 由 8 核变 16 核）。
        task_total = tiling.head_count * panels_per_head
        for owner in range_constexpr(2):
            mem_copy(
                ub_sin.next_write(),
                tile_slice(
                    gm_sin,
                    (half, VL),
                    (owner * AIV_PER_BLOCK + get_subblock_id(), 0),
                ),
            )
            mem_copy(
                ub_cos.next_write(),
                tile_slice(
                    gm_cos,
                    (half, VL),
                    (owner * AIV_PER_BLOCK + get_subblock_id(), 0),
                ),
            )
        sin_tab = [ub_sin.next_read(), ub_sin.next_read()]
        cos_tab = [ub_cos.next_read(), ub_cos.next_read()]
        self._resident_b_cube_twin(
            l1_x,
            l1_sx,
            l0a,
            l0b,
            l0c,
            ub_acc,
            gm_qr,
            gm_dqr,
            gm_w,
            gm_dsw,
            gm_sin,
            gm_cos,
            task_total,
            ub_out,
            ub_tail,
            sin_tab,
            cos_tab,
            gm_q,
            early_w,
            early_s,
        )

    @jit
    def _resident_b_cube_twin(
        self,
        l1_x,
        l1_sx,
        l0a,
        l0b,
        l0c,
        ub_acc,
        gm_qr,
        gm_dqr,
        gm_w,
        gm_dsw,
        gm_sin,
        gm_cos,
        task_total,
        ub_out,
        ub_tail,
        sin_tab,
        cos_tab,
        gm_q,
        early_w,
        early_s,
    ):
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (
            tiling.bm_b,
            tiling.bn_b,
            tiling.bk_b,
        )
        nk = tiling.rank_size // reduction_tile_cols
        panels_per_head = tiling.head_size // (PANEL_OUTPUT_TILES * output_tile_cols)
        panels_per_head * nk
        held_x = []
        held_s = []
        for owner in range_constexpr(2):
            tile_x, tile_s = self._load_qr_partials(
                l1_x,
                l1_sx,
                gm_qr,
                gm_dqr,
                owner,
                owner * nk,
                owner * tiling.resident_scale_tiles,
            )
            held_x.append(tile_x)
            held_s.append(tile_s)
        for task in range(get_block_idx(), task_total, tiling.block_dim):
            head = task // panels_per_head
            half = task % panels_per_head
            for reduction_tile_idx in range_constexpr(nk):
                head_scale = early_s[reduction_tile_idx].next_read()
                wide_w = from_tensor(
                    early_w[reduction_tile_idx % 2].next_read(),
                    dtypes.float8_e4m3fn,
                )
                # 一次 QR K 窗口装载服务两个 token 分块。
                for owner in range_constexpr(2):
                    mem_copy(
                        l0a[owner].next_write(),
                        held_x[owner][reduction_tile_idx],
                        copy_options(
                            mx_scale=tile_slice(
                                held_s[owner][
                                    reduction_tile_idx // tiling.scale_k_tiles
                                ],
                                (token_tile_rows, reduction_tile_cols // MX_GROUP),
                                (0, reduction_tile_idx % tiling.scale_k_tiles),
                            )
                        ),
                    )
                held_a = [l0a[0].next_read(), l0a[1].next_read()]
                for n_sub in range_constexpr(PANEL_OUTPUT_TILES):
                    output_tile_idx = half * PANEL_OUTPUT_TILES + n_sub
                    weight_l1_tile = tile_slice(
                        wide_w,
                        (output_tile_cols, reduction_tile_cols),
                        (n_sub, 0),
                    )
                    weight_scale_tile = tile_slice(
                        head_scale,
                        (reduction_tile_cols // MX_GROUP, output_tile_cols),
                        (0, output_tile_idx),
                    )
                    mem_copy(
                        l0b.next_write(),
                        weight_l1_tile,
                        copy_options(mx_scale=weight_scale_tile),
                    )
                    weight_l0_tile = l0b.next_read()
                    for owner in range_constexpr(2):
                        matmul(
                            tile_slice(
                                l0c[owner],
                                (token_tile_rows, output_tile_cols),
                                (0, n_sub),
                            ),
                            held_a[owner],
                            weight_l0_tile,
                            init=reduction_tile_idx == 0,
                            unit_flag=(
                                FIXPIPE_FINAL
                                if reduction_tile_idx + 1 == nk
                                else FIXPIPE_ACCUMULATE
                            ),
                            mx_mode=True,
                        )
                if const_expr(reduction_tile_idx + 1 == nk):
                    for owner in range_constexpr(2):
                        qb_accumulator_slot = ub_acc.next_write()
                        mem_copy(
                            qb_accumulator_slot,
                            from_tensor(
                                tile_slice(
                                    l0c[owner],
                                    (
                                        token_tile_rows,
                                        PANEL_OUTPUT_TILES * output_tile_cols,
                                    ),
                                    (0, 0),
                                ),
                                layout=make_layout(
                                    (
                                        token_tile_rows,
                                        PANEL_OUTPUT_TILES * output_tile_cols,
                                    )
                                ),
                            ).partition(dim=0),
                            copy_options(unit_flag=FIXPIPE_FINAL),
                        )
                        self._b_vec(
                            ub_acc,
                            ub_out,
                            ub_tail,
                            sin_tab[owner],
                            cos_tab[owner],
                            gm_sin,
                            gm_cos,
                            gm_q,
                            head * tiling.owners_b + owner,
                            half,
                        )
                next_task = task
                next_kw = reduction_tile_idx + 1
                if const_expr(reduction_tile_idx + 1 == nk):
                    next_task = task + tiling.block_dim
                    next_kw = 0
                if next_task < task_total:
                    next_head = next_task // panels_per_head
                    mem_copy(
                        early_s[next_kw].next_write(),
                        tile_slice(
                            gm_dsw,
                            (
                                tiling.head_size,
                                reduction_tile_cols // MX_PAIR_WIDTH,
                                MX_PAIR,
                            ),
                            (next_head, next_kw, 0),
                        )
                        .permute(1, 2, 0)
                        .flatten(0, 1),
                    )
                    self._load_b_panel(
                        early_w[next_kw % 2].next_write(),
                        gm_w,
                        next_task,
                        next_kw,
                    )

    @jit
    def _b_panel_l0_reuse(
        self,
        wide_w,
        head_scale,
        l0a,
        l0b,
        l0c,
        held_x,
        held_s,
        n_group,
        reduction_tile_idx,
    ):
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (
            tiling.bm_b,
            tiling.bn_b,
            tiling.bk_b,
        )
        nk = tiling.rank_size // reduction_tile_cols
        # 一个 QR K320 分块服务于打包 panel 中的四个 N64 切片。
        for reduction_buffer_idx in range_constexpr(L0_DEPTH):
            if reduction_tile_idx % L0_DEPTH == reduction_buffer_idx:
                activation_l0_slot = l0a[reduction_buffer_idx].next_write()
                for resident_reduction_idx in range_constexpr(nk):
                    if reduction_tile_idx == resident_reduction_idx:
                        mem_copy(
                            activation_l0_slot,
                            held_x[resident_reduction_idx],
                            copy_options(
                                mx_scale=tile_slice(
                                    held_s[
                                        resident_reduction_idx // tiling.scale_k_tiles
                                    ],
                                    (token_tile_rows, reduction_tile_cols // MX_GROUP),
                                    (0, resident_reduction_idx % tiling.scale_k_tiles),
                                )
                            ),
                        )
                held_a = l0a[reduction_buffer_idx].next_read()
                for n_sub in range(PANEL_OUTPUT_TILES):
                    output_tile_idx = n_group * PANEL_OUTPUT_TILES + n_sub
                    weight_l1_tile = tile_slice(
                        wide_w, (output_tile_cols, reduction_tile_cols), (n_sub, 0)
                    )
                    weight_scale_tile = tile_slice(
                        head_scale,
                        (reduction_tile_cols // MX_GROUP, output_tile_cols),
                        (0, output_tile_idx),
                    )
                    mem_copy(
                        l0b.next_write(),
                        weight_l1_tile,
                        copy_options(mx_scale=weight_scale_tile),
                    )
                    matmul(
                        tile_slice(
                            l0c,
                            (token_tile_rows, output_tile_cols),
                            (0, output_tile_idx),
                        ),
                        held_a,
                        l0b.next_read(),
                        init=reduction_tile_idx == 0,
                        unit_flag=(
                            FIXPIPE_FINAL
                            if reduction_tile_idx + 1 == nk
                            else FIXPIPE_ACCUMULATE
                        ),
                        mx_mode=True,
                    )

    @jit
    def _publish_a1(self):
        # A1 只有 Cube 生产者：排空 Cube 流水线并完成 Cube 全核汇合，
        # 再释放配对的 AIV，无需额外的 AIV→AIC 汇合。
        cube_sync_all()
        cube_sync_block_arrive(PIPE.FIXPIPE, 0, mode=0)
        cube_sync_block_wait(PIPE.S, 0, mode=0)
        cube_sync_block_arrive(PIPE.FIXPIPE, 1, mode=2)
        vec_sync_block_wait(PIPE.S, 1, mode=2)

    @jit
    def _publish_qr(self):
        # 所有 QR/scale 行均由 AIV 产生：完成 DMA 和 AIV 全核汇合后，
        # 释放配对 AIC；AIV 随后可继续处理 KV。
        vec_sync_all()
        vec_sync_block_arrive(PIPE.MTE3, 0, mode=0)
        vec_sync_block_wait(PIPE.S, 0, mode=0)
        vec_sync_block_arrive(PIPE.MTE3, 1, mode=2)
        cube_sync_block_wait(PIPE.S, 1, mode=2)

    def _alloc_a1_buffers(self):
        """分配 A1 阶段共享的 L1/L0 缓冲，供分支与通用流水线复用。"""
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols, k_l1 = (
            tiling.bm_a1,
            tiling.bn_a1,
            tiling.bk_a1,
            tiling.k_l1_a1,
        )
        scale_l0 = reduction_tile_cols // MX_GROUP
        scale_l1 = k_l1 // MX_GROUP
        groups_l1 = k_l1 // MX_PAIR_WIDTH
        l1_x = RingBuffer(
            MemLoc.L1,
            FLOAT8_E4M3FN,
            make_layout((token_tile_rows, tiling.k_l1_x_a1)),
            depth=2,
            data_format="nz",
        )
        l1_sx = RingBuffer(
            MemLoc.L1,
            dtypes.uint8,
            make_layout(
                (
                    _ceil_div(token_tile_rows, NZ_M_FRAC)
                    * _ceil_div(tiling.k_l1_x_a1 // MX_GROUP, MX_PAIR)
                    * ZN_FRACTAL_BYTES,
                )
            ),
            depth=2,
            data_format="nd",
        )
        l1_w = [
            RingBuffer(
                MemLoc.L1,
                dtypes.int8,
                make_layout((output_tile_cols, k_l1)),
                depth=1,
                data_format="nz",
            )
            for _ in range(W_BUF)
        ]
        l1_sw = RingBuffer(
            MemLoc.L1,
            FLOAT8_E8M0,
            make_layout((scale_l1, output_tile_cols)),
            depth=W_BUF,
            data_format="nz",
        )
        l0a = RingBuffer(
            MemLoc.L0A,
            FLOAT8_E4M3FN,
            make_layout((token_tile_rows, reduction_tile_cols)),
            depth=L0_DEPTH,
        )
        l0b = RingBuffer(
            MemLoc.L0B,
            FLOAT8_E4M3FN,
            make_layout((output_tile_cols, reduction_tile_cols)),
            depth=L0_DEPTH,
        )
        l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((token_tile_rows, output_tile_cols)),
            depth=1,
        )
        return scale_l0, scale_l1, groups_l1, l1_x, l1_sx, l1_w, l1_sw, l0a, l0b, l0c

    @jit
    def _a1_branch(
        self,
        gm_x,
        gm_dsx,
        gm_wqa,
        gm_dswqa,
        gm_wkv,
        gm_dswkv,
        gm_ws_qa,
        gm_ws_kv,
        gm_gamma_qr,
        gm_index,
        gm_qr,
        gm_dqr2d,
        gm_qr_bf16,
        norm_eps,
    ):
        """先发布 QA，再使 KV 投影与 QR 归一化重叠执行。"""
        scale_l0, scale_l1, groups_l1, l1_x, l1_sx, l1_w, l1_sw, l0a, l0b, l0c = (
            self._alloc_a1_buffers()
        )
        tiling = self.tiling

        # 阶段 1：将所有 QA 部分和发布给 QR 消费者。
        for qa_projection_task_idx in range(
            get_block_idx(), tiling.tiles_a1_qa, tiling.block_dim
        ):
            self._a1_tile(
                l1_x,
                l1_sx,
                l1_w,
                l1_sw,
                l0a,
                l0b,
                l0c,
                gm_x,
                gm_dsx,
                gm_wqa,
                gm_dswqa,
                gm_ws_qa,
                qa_projection_task_idx,
                tiling.n_tiles_qa,
                scale_l0,
                groups_l1,
            )
        self._publish_a1()
        # 阶段 2：独立执行 Cube KV 生产和 Vector QR 消费。
        self._a2_branch(
            gm_gamma_qr,
            gm_index,
            gm_qr,
            gm_dqr2d,
            gm_qr_bf16,
            gm_ws_qa,
            gm_ws_kv,
            norm_eps,
            l1_x,
            l1_sx,
            l1_w,
            l1_sw,
            l0a,
            l0b,
            l0c,
            gm_x,
            gm_dsx,
            gm_wkv,
            gm_dswkv,
            scale_l0,
            groups_l1,
        )

    @jit
    def _a2_branch(
        self,
        gm_gamma_qr,
        gm_index,
        gm_qr,
        gm_dqr2d,
        gm_qr_bf16,
        gm_ws_qa,
        gm_ws_kv,
        norm_eps,
        l1_x,
        l1_sx,
        l1_w,
        l1_sw,
        l0a,
        l0b,
        l0c,
        gm_x,
        gm_dsx,
        gm_wkv,
        gm_dswkv,
        scale_l0,
        groups_l1,
    ):
        """Cube 计算 KV 投影，Vector 按 token 分块归约并量化 QA。"""
        tiling = self.tiling
        token_tile_rows, rank = tiling.bm_a2, tiling.rank_size

        # DMA 与 Vector 之间的依赖由 Channel 承载，Buffer 仅用于 Vector 暂存。
        # prefill 使用单槽 Channel，在增大 token 分块时限制 UB 占用。
        a2_depth = (
            A2_BUF_DEPTH_PREFILL
            if tiling.template == "split_t"
            else A2_BUF_DEPTH_DECODE
        )
        ub_qa = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((token_tile_rows, rank)),
            depth=a2_depth,
        )
        ub_gamma_qr = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, rank)), depth=1
        )
        ub_qr = RingBuffer(
            MemLoc.UB,
            dtypes.uint8,
            make_layout((token_tile_rows, rank)),
            depth=a2_depth,
        )
        # 量化前的 BF16 QR 输出；与 FP8 payload 同步写出，供下游算子消费。
        ub_qr_bf16 = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((token_tile_rows, rank)),
            depth=a2_depth,
        )
        # 指数暂存仅在 VF 区域间传递，多分配一行以容纳完整向量读取。
        ub_exp = Buffer(
            MemLoc.UB,
            dtypes.int32,
            make_layout((token_tile_rows + 1, max(tiling.exp_stride, VL))),
        )
        # Channel 不会隐式保证每行的字节对齐，需要显式指定 scale 行跨度。
        # DMA 对外只暴露有效的量化组列。
        ub_scale = RingBuffer(
            MemLoc.UB,
            dtypes.uint8,
            make_layout((token_tile_rows, tiling.scale_stride)),
            depth=a2_depth,
        )
        # 循环不变量 gamma 只加载一次，同时保留 DMA→Vector 的 Channel 握手。
        g_slot = ub_gamma_qr.next_write()
        mem_copy(g_slot, tile_slice(gm_gamma_qr, (1, rank), (0, 0)))
        gamma_qr = ub_gamma_qr.next_read()

        qa_sum = qa_comp = None
        if const_expr(tiling.split_k > 1):
            qa_sum = Buffer(
                MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, rank))
            )
            qa_comp = Buffer(
                MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, rank))
            )
        # KV 投影与 QA 数据互不依赖，可与后续 VF 工作重叠。
        for kv_slot in range(get_block_idx(), tiling.tiles_a1_kv, tiling.block_dim):
            self._a1_tile(
                l1_x,
                l1_sx,
                l1_w,
                l1_sw,
                l0a,
                l0b,
                l0c,
                gm_x,
                gm_dsx,
                gm_wkv,
                gm_dswkv,
                gm_ws_kv,
                kv_slot,
                tiling.n_tiles_kv,
                scale_l0,
                groups_l1,
            )
        aiv = get_block_idx() * AIV_PER_BLOCK + get_subblock_id()
        # 实际 T 由运行时输入决定；填充行不进入归约与 cache 写回。
        tokens = gm_index.shape[0]
        actual_tiles = (tokens + token_tile_rows - 1) // token_tile_rows
        for token_tile_idx in range(
            aiv, actual_tiles, tiling.block_dim * AIV_PER_BLOCK
        ):
            rows = tile_slice(
                gm_index, (token_tile_rows, 1), (token_tile_idx, 0)
            ).shape[0]
            self._qr_tile(
                gm_ws_qa,
                gm_qr,
                gm_qr_bf16,
                gm_dqr2d,
                ub_qa,
                ub_qr,
                ub_qr_bf16,
                ub_exp,
                ub_scale,
                gamma_qr,
                qa_sum,
                qa_comp,
                token_tile_idx,
                rows,
                norm_eps,
            )

        # 在 Cube 消费前发布全部 QR/scale 行。KV 部分和需要对 Vector 可见，
        # QR 行需要对 Cube 可见；分别保留两条生产者到消费者的同步依赖。
        self._publish_a1()
        self._publish_qr()

    @jit
    def _service_kv(
        self,
        gm_gamma_kv,
        gm_sin,
        gm_cos,
        gm_index,
        gm_cache,
        gm_ws_kv,
        norm_eps,
        token_tile_idx,
    ):
        tiling = self.tiling
        token_tile_rows, head = tiling.bm_a2, tiling.head_size
        tokens = gm_index.shape[0]
        actual_tiles = (tokens + token_tile_rows - 1) // token_tile_rows
        if token_tile_idx < actual_tiles:
            a2_depth = (
                A2_BUF_DEPTH_PREFILL
                if tiling.template == "split_t"
                else A2_BUF_DEPTH_DECODE
            )
            ub_kv = RingBuffer(
                MemLoc.UB,
                dtypes.float32,
                make_layout((token_tile_rows, head)),
                depth=a2_depth,
            )
            ub_gamma_kv = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, head)), depth=1
            )
            ub_sin = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, VL)), depth=1
            )
            ub_cos = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, VL)), depth=1
            )
            ub_kvb = RingBuffer(
                MemLoc.UB, dtypes.uint8, make_layout((1, _kv_row_bytes(head))), depth=2
            )
            ub_tail = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 2 * VL)))
            ub_kv_bits = Buffer(MemLoc.UB, dtypes.int32, make_layout((VL,)))
            gk_slot = ub_gamma_kv.next_write()
            mem_copy(gk_slot, tile_slice(gm_gamma_kv, (1, head), (0, 0)))
            gamma_kv = ub_gamma_kv.next_read()
            kv_sum = None
            if const_expr(tiling.split_k > 1):
                kv_sum = Buffer(
                    MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, head))
                )
            rows = tile_slice(
                gm_index, (token_tile_rows, 1), (token_tile_idx, 0)
            ).shape[0]
            # KV 分支：沿 head 特征轴执行 RMSNorm、RoPE 和 per-32 BF16 scale 的 E4M3FN 转换。
            # 与 _reduce_normalize_a 中的对应循环共享 KV 分支语义，修改需同步。
            self._kv_tile(
                gm_ws_kv,
                gm_sin,
                gm_cos,
                gm_index,
                gm_cache,
                ub_kv,
                ub_sin,
                ub_cos,
                ub_kvb,
                ub_tail,
                ub_kv_bits,
                gamma_kv,
                kv_sum,
                token_tile_idx,
                rows,
                norm_eps,
            )

    # A1 计算独立的 Cube 部分和，A2 按 token 归约并执行后处理。
    @jit
    def _project_qa_kv(
        self, gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_wkv, gm_dswkv, gm_ws_qa, gm_ws_kv
    ):
        """独立计算 QA/KV 输出分块，将 FP32 部分和写入交换区。"""
        scale_l0, scale_l1, groups_l1, l1_x, l1_sx, l1_w, l1_sw, l0a, l0b, l0c = (
            self._alloc_a1_buffers()
        )
        tiling = self.tiling

        # QA 与 KV 共用任务队列，并使用相同的权重 Channel 轮转规则。
        for projection_task_idx in range(
            get_block_idx(), tiling.tiles_a1_qa + tiling.tiles_a1_kv, tiling.block_dim
        ):
            if projection_task_idx < tiling.tiles_a1_qa:
                self._a1_tile(
                    l1_x,
                    l1_sx,
                    l1_w,
                    l1_sw,
                    l0a,
                    l0b,
                    l0c,
                    gm_x,
                    gm_dsx,
                    gm_wqa,
                    gm_dswqa,
                    gm_ws_qa,
                    projection_task_idx,
                    tiling.n_tiles_qa,
                    scale_l0,
                    groups_l1,
                )
            else:
                self._a1_tile(
                    l1_x,
                    l1_sx,
                    l1_w,
                    l1_sw,
                    l0a,
                    l0b,
                    l0c,
                    gm_x,
                    gm_dsx,
                    gm_wkv,
                    gm_dswkv,
                    gm_ws_kv,
                    projection_task_idx - tiling.tiles_a1_qa,
                    tiling.n_tiles_kv,
                    scale_l0,
                    groups_l1,
                )

    @jit
    def _a1_tile(
        self,
        l1_a,
        l1_sa,
        l1_b,
        l1_sb,
        l0a,
        l0b,
        l0c,
        gm_a,
        gm_dsa,
        gm_b,
        gm_dsb,
        gm_ws,
        projection_task_idx,
        n_count,
        scale_l0,
        groups_l1,
    ):
        """累加一个 K 分块，并将其独立的 FP32 部分和写出。"""
        tiling = self.tiling
        token_tile_rows, output_tile_cols, k_l1 = (
            tiling.bm_a1,
            tiling.bn_a1,
            tiling.k_l1_a1,
        )
        n_kw = tiling.hidden_size // tiling.split_k // k_l1
        logical_slot = projection_task_idx // tiling.split_k
        split = projection_task_idx % tiling.split_k
        m = logical_slot // n_count
        n = logical_slot % n_count
        start_kw = split * n_kw
        accumulator = l0c.next_write()
        # 成对展开 K 窗口，使每个单槽权重 Channel 在一次循环体中只轮转一次。
        if const_expr(n_kw % W_BUF == 0):
            for kwp in range(n_kw // W_BUF):
                kw0 = start_kw + kwp * W_BUF
                for u in range_constexpr(W_BUF):
                    self._fill_w_a(l1_b[u], l1_sb, gm_b, gm_dsb, n, kw0 + u, groups_l1)
                for u in range_constexpr(W_BUF):
                    self._mm_a(
                        l1_a,
                        l1_sa,
                        l1_b[u],
                        l1_sb,
                        l0a,
                        l0b,
                        accumulator,
                        gm_a,
                        gm_dsa,
                        m,
                        kw0 + u,
                        start_kw,
                        scale_l0,
                    )
        else:
            # 窗口数量为奇数时退回单槽，不再重叠预取。
            for kw in range(start_kw, start_kw + n_kw):
                self._fill_w_a(l1_b[0], l1_sb, gm_b, gm_dsb, n, kw, groups_l1)
                self._mm_a(
                    l1_a,
                    l1_sa,
                    l1_b[0],
                    l1_sb,
                    l0a,
                    l0b,
                    accumulator,
                    gm_a,
                    gm_dsa,
                    m,
                    kw,
                    start_kw,
                    scale_l0,
                )
        # L0C 直接写入 GM，不经过 Vector 暂存。
        #  mem_copy 以源域为准：将 L0C 源逻辑域重绑定到目的实际域，
        # 尾块裁剪生效且物理布局保持不变。
        ws_tile = tile_slice(
            gm_ws,
            (token_tile_rows, output_tile_cols),
            (split * tiling.m_tiles_a1 + m, n),
        )
        mem_copy(
            ws_tile,
            from_tensor(
                accumulator,
                layout=make_layout(ws_tile.shape),
                physical_layout=accumulator.physical_layout,
            ),
        )

    @jit
    def _mm_a(
        self,
        l1_a,
        l1_sa,
        l1_b,
        l1_sb,
        l0a,
        l0b,
        l0c,
        gm_a,
        gm_dsa,
        m,
        kw,
        start_kw,
        scale_l0,
    ):
        """将一个权重窗口中的所有 L0 归约分块累加到 L0C。"""
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols, k_l1 = (
            tiling.bm_a1,
            tiling.bn_a1,
            tiling.bk_a1,
            tiling.k_l1_a1,
        )
        groups_l0 = reduction_tile_cols // MX_PAIR_WIDTH
        sa_layout = make_layout((token_tile_rows, tiling.k_l1_x_a1 // MX_GROUP))
        w_ready = from_tensor(l1_b.next_read(), dtype=FLOAT8_E4M3FN)
        s_ready = l1_sb.next_read()
        if const_expr(tiling.a1_window):
            mem_copy(
                l1_a.next_write(),
                tile_slice(gm_a, (token_tile_rows, k_l1), (m, kw)),
            )
            mem_copy(
                from_tensor(
                    l1_sa.next_write(),
                    dtype=FLOAT8_E8M0,
                    layout=sa_layout,
                    data_format="zn",
                ),
                tile_slice(
                    gm_dsa,
                    (token_tile_rows, k_l1 // MX_PAIR_WIDTH, MX_PAIR),
                    (m, kw, 0),
                ).flatten(1, 2),
            )
            a_ready = l1_a.next_read()
            sa_ready = from_tensor(
                l1_sa.next_read(),
                dtype=FLOAT8_E8M0,
                layout=sa_layout,
                data_format="zn",
            )
        for kl0 in range(k_l1 // reduction_tile_cols):
            kb = kw * (k_l1 // reduction_tile_cols) + kl0
            # 两个 L0 操作数都需要 mx_scale 元数据，matmul 才会使用原生 MXFP8 指令。
            if const_expr(tiling.a1_window):
                mem_copy(
                    l0a.next_write(),
                    tile_slice(
                        a_ready, (token_tile_rows, reduction_tile_cols), (0, kl0)
                    ),
                    copy_options(
                        mx_scale=tile_slice(
                            sa_ready, (token_tile_rows, scale_l0), (0, kl0)
                        )
                    ),
                )
            else:
                mem_copy(
                    l1_a.next_write(),
                    tile_slice(gm_a, (token_tile_rows, reduction_tile_cols), (m, kb)),
                )
                mem_copy(
                    from_tensor(
                        l1_sa.next_write(),
                        dtype=FLOAT8_E8M0,
                        layout=sa_layout,
                        data_format="zn",
                    ),
                    tile_slice(
                        gm_dsa, (token_tile_rows, groups_l0, MX_PAIR), (m, kb, 0)
                    ).flatten(1, 2),
                )
                mem_copy(
                    l0a.next_write(),
                    l1_a.next_read(),
                    copy_options(
                        mx_scale=from_tensor(
                            l1_sa.next_read(),
                            dtype=FLOAT8_E8M0,
                            layout=sa_layout,
                            data_format="zn",
                        )
                    ),
                )
            if const_expr(k_l1 == reduction_tile_cols):
                mem_copy(l0b.next_write(), w_ready, copy_options(mx_scale=s_ready))
            else:
                mem_copy(
                    l0b.next_write(),
                    tile_slice(
                        w_ready, (output_tile_cols, reduction_tile_cols), (0, kl0)
                    ),
                    copy_options(
                        mx_scale=tile_slice(
                            s_ready, (scale_l0, output_tile_cols), (kl0, 0)
                        )
                    ),
                )
            matmul(
                l0c,
                l0a.next_read(),
                l0b.next_read(),
                init=(kb == start_kw * (k_l1 // reduction_tile_cols)),
                mx_mode=True,
            )

    @jit
    def _fill_w_a(self, l1_b, l1_sb, gm_b, gm_dsb, n, kw, groups_l1):
        """将一个 NZ 权重窗口及其 ND scale 装入 L1。
        scale 的 [输出列, 归约组对, pair] 布局由 permute+flatten 显式
        转置展开为 (G, N) 后拷入 nz 通道；payload 字节直接搬运。"""
        tiling = self.tiling
        output_tile_cols, k_l1 = tiling.bn_a1, tiling.k_l1_a1
        w_slot = l1_b.next_write()
        mem_copy(w_slot, tile_slice(gm_b, (output_tile_cols, k_l1), (n, kw)))
        s_slot = l1_sb.next_write()
        mem_copy(
            s_slot,
            tile_slice(gm_dsb, (output_tile_cols, groups_l1, MX_PAIR), (n, kw, 0))
            .permute(1, 2, 0)
            .flatten(0, 1),
        )

    def _qr_tile(
        self,
        gm_ws_qa,
        gm_qr,
        gm_qr_bf16,
        gm_dqr2d,
        ub_qa,
        ub_qr,
        ub_qr_bf16,
        ub_exp,
        ub_scale,
        gamma_qr,
        qa_sum,
        qa_comp,
        token_tile_idx,
        rows,
        norm_eps,
    ):
        """归约一个 token 分块的 QA 部分和，量化并写出 QR/scale。"""
        tiling = self.tiling
        token_tile_rows, rank = tiling.bm_a2, tiling.rank_size
        qa_slot = ub_qa.next_write()
        if const_expr(tiling.l1_reuse):
            partials = gm_ws_qa.view(
                tiling.split_k, tiling.m_tiles_a1 * tiling.bm_a1 * rank
            )
            mem_copy(
                qa_slot,
                tile_slice(partials, (tiling.split_k, rank), (0, token_tile_idx)),
            )
        else:
            mem_copy(
                qa_slot,
                tile_slice(gm_ws_qa, (token_tile_rows, rank), (token_tile_idx, 0)),
            )
        qa = ub_qa.next_read()
        qr_slot = ub_qr.next_write()
        qr_bf16_slot = ub_qr_bf16.next_write()
        if const_expr(tiling.l1_reuse):
            self._acc_all_partials(qa_sum, qa, rank)
            self._quantize_qr(
                qa_sum,
                gamma_qr,
                qr_slot,
                ub_qr_bf16,
                qr_bf16_slot,
                ub_exp,
                rows,
                norm_eps,
            )
        elif const_expr(tiling.split_k > 1):
            self._acc_partial_qa(qa_sum, qa_comp, qa, rows, rank, True)
            for part in range(1, tiling.split_k):
                qa_slot = ub_qa.next_write()
                mem_copy(
                    qa_slot,
                    tile_slice(
                        gm_ws_qa,
                        (token_tile_rows, rank),
                        (
                            part * tiling.m_tiles_a1 * tiling.bm_a1 // token_tile_rows
                            + token_tile_idx,
                            0,
                        ),
                    ),
                )
                qa_part = ub_qa.next_read()
                self._acc_partial_qa(qa_sum, qa_comp, qa_part, rows, rank, False)
            self._quantize_qr(
                qa_sum,
                gamma_qr,
                qr_slot,
                ub_qr_bf16,
                qr_bf16_slot,
                ub_exp,
                rows,
                norm_eps,
            )
        else:
            self._quantize_qr(
                qa, gamma_qr, qr_slot, ub_qr_bf16, qr_bf16_slot, ub_exp, rows, norm_eps
            )
        scale_slot = ub_scale.next_write()
        self._pack_scale(ub_exp, scale_slot, rows)
        mem_copy(
            tile_slice(gm_qr, (token_tile_rows, rank), (token_tile_idx, 0)),
            ub_qr.next_read(),
        )
        mem_copy(
            tile_slice(gm_qr_bf16, (token_tile_rows, rank), (token_tile_idx, 0)),
            ub_qr_bf16.next_read(),
        )
        mem_copy(
            tile_slice(
                gm_dqr2d, (token_tile_rows, tiling.qr_groups), (token_tile_idx, 0)
            ),
            from_tensor(
                ub_scale.next_read(),
                layout=make_layout(
                    (token_tile_rows, tiling.qr_groups),
                    stride=(tiling.scale_stride, 1),
                ),
            ),
        )

    @jit
    def _reduce_normalize_a(
        self,
        gm_gamma_qr,
        gm_gamma_kv,
        gm_sin,
        gm_cos,
        gm_index,
        gm_cache,
        gm_qr,
        gm_dqr2d,
        gm_ws_qa,
        gm_ws_kv,
        norm_eps,
        gm_qr_bf16,
    ):
        """归约 QA 并生成 QR；L1 复用路径还在此完成 KV 后处理。每个 token 分块只归一个 Vector 核，避免 split-M 空尾块。"""
        tiling = self.tiling
        token_tile_rows, rank, head = tiling.bm_a2, tiling.rank_size, tiling.head_size

        # DMA 与 Vector 之间的依赖由 Channel 承载，Buffer 仅用于 Vector 暂存。
        # prefill 使用单槽 Channel，在增大 token 分块时限制 UB 占用。
        a2_depth = (
            A2_BUF_DEPTH_PREFILL
            if tiling.template == "split_t"
            else A2_BUF_DEPTH_DECODE
        )
        if const_expr(tiling.l1_reuse):
            ub_qa = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((tiling.split_k, rank)), depth=1
            )
            ub_kv = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((tiling.split_k, head)), depth=1
            )
        else:
            ub_qa = RingBuffer(
                MemLoc.UB,
                dtypes.float32,
                make_layout((token_tile_rows, rank)),
                depth=a2_depth,
            )
        ub_gamma_qr = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, rank)), depth=1
        )
        # RoPE 表的每行补齐到 VL，保证完整向量读取不越界。
        if const_expr(tiling.l1_reuse):
            ub_gamma_kv = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, head)), depth=1
            )
            ub_sin = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, VL)), depth=1
            )
            ub_cos = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, VL)), depth=1
            )
        ub_qr = RingBuffer(
            MemLoc.UB,
            dtypes.uint8,
            make_layout((token_tile_rows, rank)),
            depth=a2_depth,
        )
        # 量化前的 BF16 QR 输出；与 FP8 payload 同步写出，供下游算子消费。
        ub_qr_bf16 = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((token_tile_rows, rank)),
            depth=a2_depth,
        )
        # 指数暂存仅在 VF 区域间传递，多分配一行以容纳完整向量读取。
        ub_exp = Buffer(
            MemLoc.UB,
            dtypes.int32,
            make_layout((token_tile_rows + 1, max(tiling.exp_stride, VL))),
        )
        # 显式指定对齐的 scale 行跨度，DMA 只暴露公开的有效列。
        ub_scale = RingBuffer(
            MemLoc.UB,
            dtypes.uint8,
            make_layout((token_tile_rows, tiling.scale_stride)),
            depth=a2_depth,
        )
        # 使用单行 Channel 逐条散写 KV cache；
        # 在 FP32 暂存区拼接对齐的 KV 尾部向量。
        if const_expr(tiling.l1_reuse):
            ub_kvb = RingBuffer(
                MemLoc.UB, dtypes.uint8, make_layout((1, _kv_row_bytes(head))), depth=2
            )
            ub_tail = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 2 * VL)))
            ub_kv_bits = Buffer(MemLoc.UB, dtypes.int32, make_layout((VL,)))

        # 循环不变量 gamma 只加载一次，同时保留 DMA→Vector 的 Channel 握手。
        g_slot = ub_gamma_qr.next_write()
        mem_copy(g_slot, tile_slice(gm_gamma_qr, (1, rank), (0, 0)))
        if const_expr(tiling.l1_reuse):
            gk_slot = ub_gamma_kv.next_write()
            mem_copy(gk_slot, tile_slice(gm_gamma_kv, (1, head), (0, 0)))
        gamma_qr = ub_gamma_qr.next_read()
        if const_expr(tiling.l1_reuse):
            gamma_kv = ub_gamma_kv.next_read()

        qa_sum = None
        kv_sum = None
        qa_comp = None
        if const_expr(tiling.split_k > 1):
            qa_sum = Buffer(
                MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, rank))
            )
            if const_expr(not tiling.l1_reuse):
                qa_comp = Buffer(
                    MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, rank))
                )
            if const_expr(tiling.l1_reuse):
                kv_sum = Buffer(
                    MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, head))
                )
        aiv = get_block_idx() * AIV_PER_BLOCK + get_subblock_id()
        # 实际 T 由运行时输入决定；填充行不进入归约与 cache 写回。
        tokens = gm_index.shape[0]
        actual_tiles = (tokens + token_tile_rows - 1) // token_tile_rows
        for token_tile_idx in range(
            aiv, actual_tiles, tiling.block_dim * AIV_PER_BLOCK
        ):
            if const_expr(tiling.l1_reuse):
                # 驻留路径 bm_a2 恒为 1，且循环上界已裁到实际 T。
                rows = token_tile_rows
            else:
                rows = tile_slice(
                    gm_index, (token_tile_rows, 1), (token_tile_idx, 0)
                ).shape[0]
            self._qr_tile(
                gm_ws_qa,
                gm_qr,
                gm_qr_bf16,
                gm_dqr2d,
                ub_qa,
                ub_qr,
                ub_qr_bf16,
                ub_exp,
                ub_scale,
                gamma_qr,
                qa_sum,
                qa_comp,
                token_tile_idx,
                rows,
                norm_eps,
            )

        # 在 Cube 消费前发布全部 QR/scale 行；各角色参与一次同步，
        # AIV 随后处理 KV，AIC 可进入 B 阶段。
        self._publish_qr()
        if const_expr(tiling.l1_reuse):
            for token_tile_idx in range(
                aiv, actual_tiles, tiling.block_dim * AIV_PER_BLOCK
            ):
                # 驻留路径 bm_a2 恒为 1，且循环上界已裁到实际 T。
                rows = token_tile_rows
                # KV 分支：沿 head 特征轴执行 RMSNorm、RoPE 和 per-32 BF16 scale 的 E4M3FN 转换。
                # 与 _service_kv 共享 KV 分支语义，修改需同步。
                self._kv_tile(
                    gm_ws_kv,
                    gm_sin,
                    gm_cos,
                    gm_index,
                    gm_cache,
                    ub_kv,
                    ub_sin,
                    ub_cos,
                    ub_kvb,
                    ub_tail,
                    ub_kv_bits,
                    gamma_kv,
                    kv_sum,
                    token_tile_idx,
                    rows,
                    norm_eps,
                )

    @jit
    def _kv_tile(
        self,
        gm_ws_kv,
        gm_sin,
        gm_cos,
        gm_index,
        gm_cache,
        ub_kv,
        ub_sin,
        ub_cos,
        ub_kvb,
        ub_tail,
        ub_kv_bits,
        gamma_kv,
        kv_sum,
        token_tile_idx,
        rows,
        norm_eps,
    ):
        """归约一个 token 分块的 KV 部分和，逐行归一化并写回 cache。"""
        tiling = self.tiling
        token_tile_rows, head = tiling.bm_a2, tiling.head_size
        kv_slot = ub_kv.next_write()
        if const_expr(tiling.l1_reuse):
            partials = gm_ws_kv.view(
                tiling.split_k, tiling.m_tiles_a1 * tiling.bm_a1 * head
            )
            mem_copy(
                kv_slot,
                tile_slice(partials, (tiling.split_k, head), (0, token_tile_idx)),
            )
        else:
            mem_copy(
                kv_slot,
                tile_slice(gm_ws_kv, (token_tile_rows, head), (token_tile_idx, 0)),
            )
        sin_slot = ub_sin.next_write()
        mem_copy(
            sin_slot, tile_slice(gm_sin, (token_tile_rows, VL), (token_tile_idx, 0))
        )
        cos_slot = ub_cos.next_write()
        mem_copy(
            cos_slot, tile_slice(gm_cos, (token_tile_rows, VL), (token_tile_idx, 0))
        )
        kv = ub_kv.next_read()
        if const_expr(tiling.l1_reuse):
            self._acc_all_partials(kv_sum, kv, head)
        elif const_expr(tiling.split_k > 1):
            self._acc_partial(kv_sum, kv, rows, head, True)
            for part in range(1, tiling.split_k):
                kv_slot = ub_kv.next_write()
                mem_copy(
                    kv_slot,
                    tile_slice(
                        gm_ws_kv,
                        (token_tile_rows, head),
                        (
                            part * tiling.m_tiles_a1 * tiling.bm_a1 // token_tile_rows
                            + token_tile_idx,
                            0,
                        ),
                    ),
                )
                kv_part = ub_kv.next_read()
                self._acc_partial(kv_sum, kv_part, rows, head, False)
        sin_tab = ub_sin.next_read()
        cos_tab = ub_cos.next_read()
        for j in cannbotdsl.range(rows):
            kvb_slot = ub_kvb.next_write()
            if const_expr(tiling.split_k > 1):
                self._finish_kv_row(
                    kv_sum,
                    gamma_kv,
                    sin_tab,
                    cos_tab,
                    kvb_slot,
                    ub_tail,
                    ub_kv_bits,
                    j,
                    j,
                    norm_eps,
                )
            else:
                self._finish_kv_row(
                    kv,
                    gamma_kv,
                    sin_tab,
                    cos_tab,
                    kvb_slot,
                    ub_tail,
                    ub_kv_bits,
                    j,
                    j,
                    norm_eps,
                )
            slot = gm_index[token_tile_idx * token_tile_rows + j, 0]
            if slot >= 0:
                if slot < gm_cache.shape[0]:
                    mem_copy(
                        tile_slice(gm_cache, (1, _kv_row_bytes(head)), (slot, 0)),
                        ub_kvb.next_read(),
                    )

    # A 阶段的 Vector 运算

    @jit
    def _acc_partial(self, dst, src, rows, width, first):
        with vf(mode="simd"):
            # 1. 读取各特征向量，并累加 split-K 部分和。
            mask = rr.full_mask()
            for token_row in cannbotdsl.range(rows):
                for feature_vector_idx in range(width // VL):
                    dst_feature_offset = (
                        token_row * dst.physical_layout.stride[0]
                        + feature_vector_idx * VL
                    )
                    partial_values = rr.vload(
                        src,
                        token_row * src.physical_layout.stride[0]
                        + feature_vector_idx * VL,
                    )
                    if const_expr(not first):
                        partial_values = rr.vadd(
                            rr.vload(dst, dst_feature_offset), partial_values, mask=mask
                        )
                    rr.vstore(dst, dst_feature_offset, partial_values, mask)

    @jit
    def _acc_partial_qa(self, dst, compensation, src, rows, width, first):
        """使用补偿求和合并 split-K QA，保留抵消后的低位残差。"""
        with vf(mode="simd"):
            # 1. 读取各特征向量，并累加 split-K 部分和。
            mask = rr.full_mask()
            for token_row in cannbotdsl.range(rows):
                for feature_vector_idx in range(width // VL):
                    dst_feature_offset = (
                        token_row * dst.physical_layout.stride[0]
                        + feature_vector_idx * VL
                    )
                    partial_values = rr.vload(
                        src,
                        token_row * src.physical_layout.stride[0]
                        + feature_vector_idx * VL,
                    )
                    # 2. 初始化部分和，或对下一个部分和执行补偿求和。
                    if const_expr(first):
                        rr.vstore(dst, dst_feature_offset, partial_values, mask)
                        rr.vstore(
                            compensation,
                            dst_feature_offset,
                            rr.vdups(0.0, dtypes.float32, mask=mask),
                            mask,
                        )
                    else:
                        corrected = rr.vsub(
                            partial_values,
                            rr.vload(compensation, dst_feature_offset),
                            mask=mask,
                        )
                        prior = rr.vload(dst, dst_feature_offset)
                        total = rr.vadd(prior, corrected, mask=mask)
                        lost = rr.vsub(
                            rr.vsub(total, prior, mask=mask), corrected, mask=mask
                        )
                        # 3. 同时保存累加结果和丢失的低位残差。
                        rr.vstore(dst, dst_feature_offset, total, mask)
                        rr.vstore(compensation, dst_feature_offset, lost, mask)

    @jit
    def _quantize_qr(
        self,
        ub_qa,
        ub_gamma,
        ub_qr,
        ub_qr_bf16,
        ub_qr_bf16_slot,
        ub_exp,
        rows,
        norm_eps,
    ):
        """沿 rank 轴执行 RMSNorm，并按 MX_GROUP 动态量化为 MXFP8。
        共享指数为 max(floor(log2(amax)) - 8, -127)。
        归一化后的分组值保留在寄存器中，避免 UB 写后读冒险；
        读取源张量时使用包含 FixPipe 填充的物理行跨度。"""
        tiling = self.tiling
        rank = tiling.rank_size
        inv_rank = 1.0 / rank
        # 按物理 UB 行跨度寻址，包含 FixPipe 插入的填充。
        src_pitch = ub_qa.physical_layout.stride[0]
        qr_pitch = ub_qr.physical_layout.stride[0]
        bf16_pitch = ub_qr_bf16.physical_layout.stride[0]
        with vf(mode="simd"):
            full = rr.full_mask()
            group_mask = rr.update_mask(MX_GROUP, elem_bits=32)[0]
            one = rr.vdups(1.0, dtypes.float32, mask=full)
            for token_row in cannbotdsl.range(rows):
                qa_row_offset = token_row * src_pitch
                # 1. 将平方和归约到第 0 个 lane，再广播到整行。
                square_sum_accumulator = rr.vdups(0.0, dtypes.float32, mask=full)
                for rank_vector_idx in range(rank // VL):
                    qa_values = rr.vload(ub_qa, qa_row_offset + rank_vector_idx * VL)
                    squared_values = rr.vmul(qa_values, qa_values, mask=full)
                    square_sum_accumulator = rr.vadd(
                        square_sum_accumulator,
                        rr.vreduce_sum(squared_values, mask=full),
                        mask=full,
                    )
                square_sum = rr.vdup(square_sum_accumulator, mask=full)
                mean_square = rr.vadds(
                    rr.vmuls(square_sum, inv_rank, mask=full), norm_eps, mask=full
                )
                rstd = rr.vdiv(one, rr.vsqrt(mean_square, mask=full), mask=full)

                # 2. 逐个 rank 分组归一化，并确定共享量化 scale。
                for rank_group_idx in range(rank // MX_GROUP):
                    rank_group_offset = rank_group_idx * MX_GROUP
                    normalized_values = rr.vmul(
                        rr.vload(ub_qa, qa_row_offset + rank_group_offset),
                        rstd,
                        mask=group_mask,
                    )
                    normalized_values = rr.vmul(
                        normalized_values,
                        rr.vload(ub_gamma, rank_group_offset),
                        mask=group_mask,
                    )
                    # 公开 QR 是量化前的 BF16；内部 QB 的 MXFP8 量化继续使用 FP32
                    # 归一化值（接受与原小算子链路的数值差异，由网络精度对比验收）。
                    rr.vstore_pack(
                        ub_qr_bf16_slot,
                        token_row * bf16_pitch + rank_group_offset,
                        rr.vcast(normalized_values, dtypes.bfloat16, mask=group_mask),
                        group_mask,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )
                    # sum 和 max 归约都只写第 0 个 lane，逐元素使用前必须广播。
                    amax = rr.vdup(
                        rr.vreduce_max(
                            rr.vabs(normalized_values, mask=group_mask), mask=group_mask
                        ),
                        mask=group_mask,
                    )
                    shared_exponent = self._shared_exponent(amax, group_mask)
                    payload = self._to_e4m3(
                        rr.vmul(
                            normalized_values,
                            self._reciprocal(shared_exponent, group_mask),
                            mask=group_mask,
                        ),
                        group_mask,
                    )
                    # 3. 写出 payload 字节及编码后的分组指数。
                    rr.vstore_pack(
                        ub_qr,
                        token_row * qr_pitch + rank_group_offset,
                        payload,
                        group_mask,
                        pack_mode=rr.PackMode.B32_TO_B8,
                    )
                    rr.vstore_first(
                        ub_exp,
                        token_row * ub_exp.physical_layout.stride[0] + rank_group_idx,
                        rr.vadds(shared_exponent, E8M0_BIAS, mask=group_mask),
                    )

    @jit
    def _shared_exponent(self, amax, mask):
        """HF 口径：scale 为不小于 max(amax,1e-4)/448 的最小 2 的幂。

        用 FP32 指数/尾数位向上取整（不依赖 log2 边界）；返回无偏指数 e，
        编码字节为 e+127。amax 为零时下限 1e-4 保证 scale 非零、payload 为 0。
        """
        floored = rr.vmaxs(amax, 1e-4, mask=mask)
        s0 = rr.vmuls(floored, 1.0 / E4M3_MAX, mask=mask)
        bits = rr.vreinterpret(s0, dtypes.int32)
        # (bits + 0x7FFFFF) >> 23：尾数非零时进位到指数，等价于向上取整到 2 的幂。
        biased = rr.vshr(
            rr.vadds(bits, 0x007FFFFF, mask=mask), FP32_EXP_SHIFT, mask=mask
        )
        exponent = rr.vmin(
            rr.vadds(biased, -FP32_EXP_BIAS, mask=mask),
            rr.vdups(127, dtypes.int32, mask=mask),
            mask=mask,
        )
        return rr.vmaxs(exponent, -E8M0_BIAS, mask=mask)

    @jit
    def _reciprocal(self, exponent, mask):
        """直接利用 FP32 指数字段构造 2**(-exponent)。有限 FP32 amax 对应的量化指数不超过 119，因此倒数保持为正规数，无需通过除法计算。"""
        field = rr.vadds(rr.vneg(exponent, mask=mask), FP32_EXP_BIAS, mask=mask)
        return rr.vreinterpret(
            rr.vshl(field, FP32_EXP_SHIFT, mask=mask), dtypes.float32
        )

    @jit
    def _to_e4m3(self, value, mask):
        """先钳位到 E4M3FN 有限值域，再按最近偶数舍入转换。显式钳位保证饱和语义；前序运算的舍入可能使 FP8 中点附近的结果与 FP32 golden 不逐字节相同。"""
        clamped = rr.vmins(rr.vmaxs(value, -E4M3_MAX, mask=mask), E4M3_MAX, mask=mask)
        return rr.vcast(
            clamped,
            dtypes.float8_e4m3fn,
            mask=mask,
            rounding=rr.RoundingMode.RN,
            reg_layout=rr.RegLayout.ZERO,
        )

    @jit
    def _pack_scale(self, ub_exp, ub_scale, rows):
        """将 int32 指数打包为 E8M0 字节，并使用对齐的向量写入。独立的 VF 区域读取 _quantize_qr 写出的指数；B32_TO_B8 直接保留低字节，不执行数值转换。"""
        tiling = self.tiling
        exp_pitch = ub_exp.physical_layout.stride[0]
        scale_pitch = ub_scale.physical_layout.stride[0]
        with vf(mode="simd"):
            # 1. 读取 int32 编码指数；2. 将低字节打包写出。
            for token_row in cannbotdsl.range(rows):
                for scale_vector_idx in range(_ceil_div(tiling.qr_groups, VL)):
                    active_scale_lanes = min(
                        VL, tiling.qr_groups - scale_vector_idx * VL
                    )
                    mask = rr.update_mask(active_scale_lanes, elem_bits=32)[0]
                    rr.vstore_pack(
                        ub_scale,
                        token_row * scale_pitch + scale_vector_idx * VL,
                        rr.vload(ub_exp, token_row * exp_pitch + scale_vector_idx * VL),
                        mask,
                        pack_mode=rr.PackMode.B32_TO_B8,
                    )

    @jit
    def _quantize_kv_group(
        self, value, ub_kvb, payload_offset, ub_bits, group_index, mask
    ):
        """按 32 元素组量化 KV：FP32 amax 求 BF16 scale，再用存储值反量化。"""
        rr.vdups(E4M3_MAX, dtypes.float32, mask=mask)
        amax = rr.vdup(rr.vreduce_max(rr.vabs(value, mask=mask), mask=mask), mask=mask)
        # 网络原有 round_scale=True 口径：a=max(amax,1e-4)，scale 为不小于
        # a/448 的最小 2 的幂；用 FP32 指数/尾数位向上取整，避免 log2 边界差异。
        floored = rr.vmaxs(amax, 1e-4, mask=mask)
        s0 = rr.vmuls(floored, 1.0 / E4M3_MAX, mask=mask)
        s0_bits = rr.vreinterpret(s0, dtypes.int32)
        exponent = rr.vshr(rr.vadds(s0_bits, 0x007FFFFF, mask=mask), 23, mask=mask)
        exponent = rr.vmin(exponent, rr.vdups(254, dtypes.int32, mask=mask), mask=mask)
        scale_bits = rr.vshl(exponent, 23, mask=mask)
        bf16_bits = rr.vshr(scale_bits, 16, mask=mask)
        rr.vstore_first(ub_bits, group_index, bf16_bits)
        scale_used = rr.vreinterpret(scale_bits, dtypes.float32)
        quotient = rr.vdiv(value, scale_used, mask=mask)
        quotient = rr.vselect(
            quotient,
            rr.vdups(0.0, dtypes.float32, mask=mask),
            cond_mask=rr.vnes(scale_used, 0.0, mask=mask),
        )
        rr.vstore_pack(
            ub_kvb,
            payload_offset,
            self._to_e4m3(quotient, mask),
            mask,
            pack_mode=rr.PackMode.B32_TO_B8,
        )

    @jit
    def _finish_kv_row(
        self,
        ub_kv,
        ub_gamma,
        ub_sin,
        ub_cos,
        ub_kvb,
        ub_tail,
        ub_bits,
        token_row,
        table_row,
        norm_eps,
    ):
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
        with vf(mode="simd"):
            full = rr.full_mask()
            m_pair = rr.update_mask(rope_half_width, elem_bits=32)[0]
            m_rope = rr.update_mask(rope_width, elem_bits=32)[0]
            group_mask = rr.update_mask(MX_GROUP, elem_bits=32)[0]
            one = rr.vdups(1.0, dtypes.float32, mask=full)
            kv_row_offset = token_row * src_pitch
            # 1. 计算沿 head 特征轴的 RMS 归一化因子。
            square_sum_accumulator = rr.vdups(0.0, dtypes.float32, mask=full)
            for head_vector_idx in range(head_width // VL):
                kv_values = rr.vload(ub_kv, kv_row_offset + head_vector_idx * VL)
                squared_values = rr.vmul(kv_values, kv_values, mask=full)
                square_sum_accumulator = rr.vadd(
                    square_sum_accumulator,
                    rr.vreduce_sum(squared_values, mask=full),
                    mask=full,
                )
            square_sum = rr.vdup(square_sum_accumulator, mask=full)
            mean_square = rr.vadds(
                rr.vmuls(square_sum, inv_head, mask=full), norm_eps, mask=full
            )
            rstd = rr.vdiv(one, rr.vsqrt(mean_square, mask=full), mask=full)

            if const_expr(rope_width % MX_GROUP == 0):
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
                normed = rr.vmul(
                    rr.vload(ub_kv, kv_row_offset + group_index * MX_GROUP),
                    rstd,
                    mask=group_mask,
                )
                normed = rr.vmul(
                    normed,
                    rr.vload(ub_gamma, group_index * MX_GROUP),
                    mask=group_mask,
                )
                self._quantize_kv_group(
                    normed,
                    ub_kvb,
                    group_index * MX_GROUP,
                    ub_bits,
                    group_index,
                    group_mask,
                )

            # 3. 在 FP32 暂存区拼接最后一个向量。字节打包后 RoPE 后半区
            # 可能不对齐，因此拼接完成后一次性打包整个向量。
            normed = rr.vmul(
                rr.vload(ub_kv, kv_row_offset + head_width - VL), rstd, mask=full
            )
            normed = rr.vmul(normed, rr.vload(ub_gamma, head_width - VL), mask=full)
            rr.vstore(ub_tail, 0, normed, full)
            rope_vec = self._rope_pair(
                ub_kv,
                kv_row_offset + head_width - rope_width,
                ub_gamma,
                head_width - rope_width,
                ub_sin,
                ub_cos,
                table_row,
                rstd,
                m_pair,
            )
            rr.vstore(ub_tail, rope_tail_offset, rope_vec, m_rope)
            # 此屏障必须保留：紧接着的读取依赖上方刚写入的尾部数据。
            rr.vmem_bar("vst_vld")

            # 4. RoPE 之后的组：从暂存区按 32 元素量化。
            if const_expr(rope_width % MX_GROUP == 0):
                for local_group in range(post_groups):
                    group_index = post_group_start + local_group
                    self._quantize_kv_group(
                        rr.vload(ub_tail, local_group * MX_GROUP),
                        ub_kvb,
                        group_index * MX_GROUP,
                        ub_bits,
                        group_index,
                        group_mask,
                    )
            else:
                for group_index in range(post_group_start, group_count):
                    self._quantize_kv_group(
                        rr.vload(
                            ub_tail,
                            (group_index - tail_group_offset) * MX_GROUP,
                        ),
                        ub_kvb,
                        group_index * MX_GROUP,
                        ub_bits,
                        group_index,
                        group_mask,
                    )

            # 5. 16 个 BF16 scale 以小端连续写入 payload 之后的 scale 区。
            rr.vmem_bar("vst_vld")
            scale_view = from_tensor(
                ub_kvb,
                dtype=dtypes.uint16,
                layout=make_layout((1, group_count)),
                offset=head_width,
            )
            scale_mask = rr.update_mask(group_count, elem_bits=16)[0]
            rr.vstore(
                scale_view,
                0,
                rr.vreinterpret(
                    rr.vpack(rr.vload(ub_bits, 0), dtypes.uint16), dtypes.bfloat16
                ),
                scale_mask,
            )

    @jit
    def _rope_pair(
        self,
        src,
        src_feature_offset,
        gamma,
        gamma_feature_offset,
        ub_sin,
        ub_cos,
        table_row,
        rstd,
        mask,
    ):
        """按网络现有的 interleave 约定执行尾部 RoPE，输出保持交错排列。
        out_even = x_even * cos_even - x_odd * sin_even
        out_odd  = x_odd * cos_odd + x_even * sin_odd
        位置表为全宽 [T,Dr]，按 (2i, 2i+1) 取 cos/sin；不依赖相邻表项相等。
        rstd 为 None 时跳过归一化，用于 QB 分支。"""
        even, odd = rr.vload_deinterleave(src, src_feature_offset, width="b32")
        if const_expr(gamma is not None):
            g_even, g_odd = rr.vload_deinterleave(
                gamma, gamma_feature_offset, width="b32"
            )
            even = rr.vmul(rr.vmul(even, g_even, mask=mask), rstd, mask=mask)
            odd = rr.vmul(rr.vmul(odd, g_odd, mask=mask), rstd, mask=mask)
        # RoPE 表按补齐后的物理行跨度寻址。
        pitch = ub_sin.physical_layout.stride[0]
        cos_even, cos_odd = rr.vload_deinterleave(
            ub_cos, table_row * pitch, width="b32"
        )
        sin_even, sin_odd = rr.vload_deinterleave(
            ub_sin, table_row * pitch, width="b32"
        )
        out_even = rr.vsub(
            rr.vmul(even, cos_even, mask=mask),
            rr.vmul(odd, sin_even, mask=mask),
            mask=mask,
        )
        out_odd = rr.vadd(
            rr.vmul(odd, cos_odd, mask=mask),
            rr.vmul(even, sin_odd, mask=mask),
            mask=mask,
        )
        interleaved, _ = rr.vinterleave(out_even, out_odd)
        return interleaved

    # B 阶段

    @jit
    def _stage_b(
        self,
        gm_qr,
        gm_dqr3d,
        gm_wqb,
        gm_dswqb,
        gm_sin,
        gm_cos,
        gm_q,
        prefetched_qb_weights,
        prefetched_qb_scales,
        gm_gamma_kv,
        gm_index,
        gm_cache,
        gm_ws_kv,
        norm_eps,
    ):
        tiling = self.tiling
        tokens = gm_index.shape[0]
        actual_tiles = (tokens + tiling.bm_a2 - 1) // tiling.bm_a2
        token_tile_rows, output_tile_cols, reduction_tile_cols, k_l1 = (
            tiling.bm_b,
            tiling.bn_b,
            tiling.bk_b,
            tiling.k_l1_b,
        )
        rank, head = tiling.rank_size, tiling.head_size
        panels_per_head = head // output_tile_cols
        scale_l1 = k_l1 // MX_GROUP
        scale_l0 = reduction_tile_cols // MX_GROUP
        groups_l1 = k_l1 // MX_PAIR_WIDTH

        # ---- 阶段 1：缓冲分配（b_resident 二分） ----
        # 驻留 QB 跨 head 复用一个 QR token 分块，其他路径沿 K 轴双缓冲。
        if const_expr(tiling.b_resident):
            # 每个 token 分块对应一次 Channel 轮转，所有 head 任务共用该视图。
            # 切换到下一个分块前，Channel 复用会插入 MTE1→MTE2 依赖。
            l1_a = RingBuffer(
                MemLoc.L1,
                FLOAT8_E4M3FN,
                make_layout((token_tile_rows, rank)),
                depth=1,
                data_format="nz",
            )
            # E8M0 scale 的 zn 分形不能用 data_format="zn" 直接分配（ 仅
            # 接受 nd/nz）：以 uint8 原始字节承载，选槽时 from_tensor 成 zn 视图。
            l1_sa = RingBuffer(
                MemLoc.L1,
                dtypes.uint8,
                make_layout(
                    (
                        _ceil_div(token_tile_rows, NZ_M_FRAC)
                        * _ceil_div(rank // MX_GROUP, MX_PAIR)
                        * ZN_FRACTAL_BYTES,
                    )
                ),
                depth=1,
                data_format="nd",
            )
        else:
            l1_a = RingBuffer(
                MemLoc.L1,
                FLOAT8_E4M3FN,
                make_layout((token_tile_rows, reduction_tile_cols)),
                depth=2,
                data_format="nz",
            )
            l1_sa = RingBuffer(
                MemLoc.L1,
                dtypes.uint8,
                make_layout(
                    (
                        _ceil_div(token_tile_rows, NZ_M_FRAC)
                        * _ceil_div(scale_l0, MX_PAIR)
                        * ZN_FRACTAL_BYTES,
                    )
                ),
                depth=2,
                data_format="nd",
            )
        # 与投影阶段相同，NZ 权重使用整数字节载体和 FP8 别名。
        if const_expr(tiling.b_resident):
            l1_b = [
                prefetched_qb_weights,
                RingBuffer(
                    MemLoc.L1,
                    dtypes.int8,
                    make_layout((output_tile_cols, k_l1)),
                    depth=1,
                    data_format="nz",
                ),
            ]
            l1_sb = prefetched_qb_scales
        else:
            l1_b = [
                RingBuffer(
                    MemLoc.L1,
                    dtypes.int8,
                    make_layout((output_tile_cols, k_l1)),
                    depth=1,
                    data_format="nz",
                )
                for _ in range(W_BUF)
            ]
            l1_sb = RingBuffer(
                MemLoc.L1,
                FLOAT8_E8M0,
                make_layout((scale_l1, output_tile_cols)),
                depth=W_BUF,
                data_format="nz",
            )
        l0a = RingBuffer(
            MemLoc.L0A,
            FLOAT8_E4M3FN,
            make_layout((token_tile_rows, reduction_tile_cols)),
            depth=L0_DEPTH,
        )
        l0b = RingBuffer(
            MemLoc.L0B,
            FLOAT8_E4M3FN,
            make_layout((output_tile_cols, reduction_tile_cols)),
            depth=1 if tiling.bn_b == BN_B_FULL_HEAD else L0_DEPTH,
        )
        l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout(
                (token_tile_rows, output_tile_cols if tiling.b_resident else head)
            ),
            depth=1,
        )
        # 沿特征轴二分，每个 Vector 核获得全部行和一半列。
        feature_cols = tiling.b_cols
        ub_acc = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((token_tile_rows, feature_cols)),
            depth=1,
        )
        # 输出 Channel 使用单槽，使 QB 的 UB 占用不超过容量。
        ub_out = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((token_tile_rows, feature_cols)),
            depth=1,
        )
        ub_tail = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, VL)))
        # RoPE 行补齐到 VL，特征轴放在最内层以保证读取对齐。
        ub_sin = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, VL)), depth=1
        )
        ub_cos = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((token_tile_rows, VL)), depth=1
        )

        # ---- 阶段 2：驻留主循环；阶段 3：通用任务循环 ----
        # ND 激活在 kernel 内转换为 NZ；预打包的 NZ 权重直接搬运。
        subblock = get_subblock_id()
        if const_expr(tiling.b_resident):
            # 各核完成当前 token 分块的所有 head 任务后再前进。
            # 每核对该分块的 QR/scale 和 RoPE 表只加载一次。
            for owner in range(tiling.owners_b):
                mem_copy(
                    l1_a.next_write(),
                    tile_slice(gm_qr, (token_tile_rows, rank), (owner, 0)),
                )
                mem_copy(
                    from_tensor(
                        l1_sa.next_write(),
                        dtype=FLOAT8_E8M0,
                        layout=make_layout((token_tile_rows, rank // MX_GROUP)),
                        data_format="zn",
                    ),
                    tile_slice(
                        gm_dqr3d,
                        (token_tile_rows, rank // MX_PAIR_WIDTH, MX_PAIR),
                        (owner, 0, 0),
                    ).flatten(1, 2),
                )
                a_ready = l1_a.next_read()
                sa_ready = from_tensor(
                    l1_sa.next_read(),
                    dtype=FLOAT8_E8M0,
                    layout=make_layout((token_tile_rows, rank // MX_GROUP)),
                    data_format="zn",
                )
                mem_copy(
                    ub_sin.next_write(),
                    tile_slice(gm_sin, (token_tile_rows, VL), (owner, 0)),
                )
                mem_copy(
                    ub_cos.next_write(),
                    tile_slice(gm_cos, (token_tile_rows, VL), (owner, 0)),
                )
                sin_tab = ub_sin.next_read()
                cos_tab = ub_cos.next_read()
                for head_tile in range(
                    get_block_idx(),
                    tiling.head_count * panels_per_head,
                    tiling.block_dim,
                ):
                    head_token_task_idx = head_tile * tiling.owners_b + owner
                    self._b_cube(
                        a_ready,
                        sa_ready,
                        l1_b,
                        l1_sb,
                        l0a,
                        l0b,
                        l0c,
                        ub_acc,
                        gm_qr,
                        gm_dqr3d,
                        gm_wqb,
                        gm_dswqb,
                        gm_q,
                        head_token_task_idx,
                        scale_l0,
                        groups_l1,
                    )
                    # 驻留 B 优先遍历 token 分块；owners_b 大于 1 时，
                    # 本地迭代号不能直接用任务编号除以核数计算。
                    b_step = owner * _ceil_div(
                        tiling.head_count * panels_per_head - get_block_idx(),
                        tiling.block_dim,
                    )
                    b_step += head_tile // tiling.block_dim
                    self._service_kv(
                        gm_gamma_kv,
                        gm_sin,
                        gm_cos,
                        gm_index,
                        gm_cache,
                        gm_ws_kv,
                        norm_eps,
                        get_block_idx() * AIV_PER_BLOCK
                        + subblock
                        + b_step * (tiling.block_dim * AIV_PER_BLOCK),
                    )
                    self._b_vec_reuse(
                        ub_acc,
                        ub_out,
                        ub_tail,
                        sin_tab,
                        cos_tab,
                        gm_q,
                        head_token_task_idx,
                        subblock,
                    )
        else:
            for head_token_task_idx in range(
                get_block_idx(), tiling.tiles_b, tiling.block_dim
            ):
                if head_token_task_idx < tiling.tiles_b:
                    self._b_cube(
                        l1_a,
                        l1_sa,
                        l1_b,
                        l1_sb,
                        l0a,
                        l0b,
                        l0c,
                        ub_acc,
                        gm_qr,
                        gm_dqr3d,
                        gm_wqb,
                        gm_dswqb,
                        gm_q,
                        head_token_task_idx,
                        scale_l0,
                        groups_l1,
                    )
                    self._service_kv(
                        gm_gamma_kv,
                        gm_sin,
                        gm_cos,
                        gm_index,
                        gm_cache,
                        gm_ws_kv,
                        norm_eps,
                        get_block_idx() * AIV_PER_BLOCK
                        + subblock
                        + (head_token_task_idx // tiling.block_dim)
                        * (tiling.block_dim * AIV_PER_BLOCK),
                    )
                    self._b_vec(
                        ub_acc,
                        ub_out,
                        ub_tail,
                        ub_sin,
                        ub_cos,
                        gm_sin,
                        gm_cos,
                        gm_q,
                        head_token_task_idx,
                        subblock,
                    )

        # ---- 阶段 4：剩余 KV 尾部补齐 ----
        # 各核的 B 任务数量可能不同，包括没有 B 任务的核在内，
        # 每个剩余 KV 分块都必须且只处理一次。
        if const_expr(
            tiling.tiles_a2
            > tiling.min_b_iterations * (tiling.block_dim * AIV_PER_BLOCK)
        ):
            if const_expr(tiling.b_resident):
                b_count = tiling.owners_b * _ceil_div(
                    tiling.head_count * panels_per_head - get_block_idx(),
                    tiling.block_dim,
                )
            else:
                b_count = _ceil_div(tiling.tiles_b - get_block_idx(), tiling.block_dim)
            for kv_slot in range(
                get_block_idx() * AIV_PER_BLOCK
                + subblock
                + b_count * (tiling.block_dim * AIV_PER_BLOCK),
                actual_tiles,
                tiling.block_dim * AIV_PER_BLOCK,
            ):
                self._service_kv(
                    gm_gamma_kv,
                    gm_sin,
                    gm_cos,
                    gm_index,
                    gm_cache,
                    gm_ws_kv,
                    norm_eps,
                    kv_slot,
                )

    @jit
    def _b_cube(
        self,
        l1_a,
        l1_sa,
        l1_b,
        l1_sb,
        l0a,
        l0b,
        l0c,
        ub_acc,
        gm_qr,
        gm_dqr3d,
        gm_wqb,
        gm_dswqb,
        gm_q,
        head_token_task_idx,
        scale_l0,
        groups_l1,
    ):
        """Cube 使用公开的 QR payload/scale 完成投影，并通过 FixPipe 发布结果。"""
        # ---- 阶段 1：任务解析与 L0C 累加器 ----
        tiling = self.tiling
        output_tile_cols, k_l1 = tiling.bn_b, tiling.k_l1_b
        rank, head = tiling.rank_size, tiling.head_size
        n_tiles = 1 if tiling.b_resident else head // output_tile_cols
        owner = head_token_task_idx % tiling.owners_b
        head_tile_idx = head_token_task_idx // tiling.owners_b
        n_kw = rank // k_l1
        accumulator = l0c.next_write()
        for n in range(n_tiles):
            # 公开 QR scale 已经具有 mx_scale_and 所需的
            # [token, 归约组对, pair] 布局。
            wrow = head_tile_idx * n_tiles + n
            # ---- 阶段 2：驻留双窗口轮转；阶段 3/4：偶数窗口成对展开与奇数窗口回退 ----
            # 成对展开窗口，使每个单槽权重 Channel 在一次循环体中只轮转一次。
            if const_expr(tiling.b_resident):
                # 当前权重窗口驻留在 L1 时，由 MTE2 加载下一窗口。
                # 首个窗口已在 __call__ 的 A2 期间预取。
                if head_token_task_idx != get_block_idx() * tiling.owners_b:
                    self._fill_w_b(l1_b[0], l1_sb, gm_wqb, gm_dswqb, wrow, 0, groups_l1)
                if const_expr(n_kw > 1):
                    self._fill_w_b(l1_b[1], l1_sb, gm_wqb, gm_dswqb, wrow, 1, groups_l1)
                for kwp in range(n_kw // W_BUF):
                    kw0 = kwp * W_BUF
                    self._mm_window_b(
                        l1_a,
                        l1_sa,
                        l1_b[0],
                        l1_sb,
                        l0a,
                        l0b,
                        accumulator,
                        gm_qr,
                        gm_dqr3d,
                        owner,
                        n,
                        kw0,
                        scale_l0,
                    )
                    if kw0 + W_BUF < n_kw:
                        self._fill_w_b(
                            l1_b[0],
                            l1_sb,
                            gm_wqb,
                            gm_dswqb,
                            wrow,
                            kw0 + W_BUF,
                            groups_l1,
                        )
                    self._mm_window_b(
                        l1_a,
                        l1_sa,
                        l1_b[1],
                        l1_sb,
                        l0a,
                        l0b,
                        accumulator,
                        gm_qr,
                        gm_dqr3d,
                        owner,
                        n,
                        kw0 + 1,
                        scale_l0,
                    )
                    if kw0 + W_BUF + 1 < n_kw:
                        self._fill_w_b(
                            l1_b[1],
                            l1_sb,
                            gm_wqb,
                            gm_dswqb,
                            wrow,
                            kw0 + W_BUF + 1,
                            groups_l1,
                        )
                if const_expr(n_kw % W_BUF != 0):
                    self._mm_window_b(
                        l1_a,
                        l1_sa,
                        l1_b[0],
                        l1_sb,
                        l0a,
                        l0b,
                        accumulator,
                        gm_qr,
                        gm_dqr3d,
                        owner,
                        n,
                        n_kw - 1,
                        scale_l0,
                    )
            elif const_expr(n_kw % W_BUF == 0):
                for kwp in range(n_kw // W_BUF):
                    kw0 = kwp * W_BUF
                    for u in range_constexpr(W_BUF):
                        self._fill_w_b(
                            l1_b[u],
                            l1_sb,
                            gm_wqb,
                            gm_dswqb,
                            wrow,
                            kw0 + u,
                            groups_l1,
                        )
                    for u in range_constexpr(W_BUF):
                        self._mm_window_b(
                            l1_a,
                            l1_sa,
                            l1_b[u],
                            l1_sb,
                            l0a,
                            l0b,
                            accumulator,
                            gm_qr,
                            gm_dqr3d,
                            owner,
                            n,
                            kw0 + u,
                            scale_l0,
                        )
            else:
                for kw in range(n_kw):
                    self._fill_w_b(
                        l1_b[0],
                        l1_sb,
                        gm_wqb,
                        gm_dswqb,
                        wrow,
                        kw,
                        groups_l1,
                    )
                    self._mm_window_b(
                        l1_a,
                        l1_sa,
                        l1_b[0],
                        l1_sb,
                        l0a,
                        l0b,
                        accumulator,
                        gm_qr,
                        gm_dqr3d,
                        owner,
                        n,
                        kw,
                        scale_l0,
                    )

        # ---- 阶段 5：FixPipe 双目的地发布 ----
        acc_slot = ub_acc.next_write()
        # FixPipe 双目的地（split-N）：L0C 源 partition(dim=1) 将特征列对半
        # 分给两个 Vector 核。
        mem_copy(
            acc_slot,
            accumulator.partition(dim=1),
            copy_options(unit_flag=FIXPIPE_FINAL),
        )

    @jit
    def _fill_w_b(self, l1_b, l1_sb, gm_wqb, gm_dswqb, wrow, kw, groups_l1):
        """加载一个 QB 权重窗口及其 scale，规则同 _fill_w_a；
        两函数分别绑定 A1/B 阶段的 tiling 字段，修改需同步。"""
        tiling = self.tiling
        output_tile_cols, k_l1 = tiling.bn_b, tiling.k_l1_b
        w_slot = l1_b.next_write()
        mem_copy(
            w_slot,
            tile_slice(gm_wqb, (output_tile_cols, k_l1), (wrow, kw)),
        )
        s_slot = l1_sb.next_write()
        mem_copy(
            s_slot,
            tile_slice(gm_dswqb, (output_tile_cols, groups_l1, MX_PAIR), (wrow, kw, 0))
            .permute(1, 2, 0)
            .flatten(0, 1),
        )

    @jit
    def _mm_window_b(
        self,
        l1_a,
        l1_sa,
        l1_b,
        l1_sb,
        l0a,
        l0b,
        l0c,
        gm_qr,
        gm_dqr3d,
        owner,
        n,
        kw,
        scale_l0,
    ):
        """使用 L0 归约分块累加一个 QB 权重窗口。"""
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols, k_l1 = (
            tiling.bm_b,
            tiling.bn_b,
            tiling.bk_b,
            tiling.k_l1_b,
        )
        k_tiles = tiling.rank_size // reduction_tile_cols
        w_ready = from_tensor(l1_b.next_read(), dtype=FLOAT8_E4M3FN)
        s_ready = l1_sb.next_read()
        for kl0 in range(k_l1 // reduction_tile_cols):
            kb = kw * (k_l1 // reduction_tile_cols) + kl0
            if const_expr(tiling.b_resident):
                mem_copy(
                    l0a.next_write(),
                    tile_slice(l1_a, (token_tile_rows, reduction_tile_cols), (0, kb)),
                    copy_options(
                        mx_scale=tile_slice(l1_sa, (token_tile_rows, scale_l0), (0, kb))
                    ),
                )
            else:
                mem_copy(
                    l1_a.next_write(),
                    tile_slice(
                        gm_qr, (token_tile_rows, reduction_tile_cols), (owner, kb)
                    ),
                )
                mem_copy(
                    from_tensor(
                        l1_sa.next_write(),
                        dtype=FLOAT8_E8M0,
                        layout=make_layout((token_tile_rows, scale_l0)),
                        data_format="zn",
                    ),
                    tile_slice(
                        gm_dqr3d,
                        (
                            token_tile_rows,
                            reduction_tile_cols // MX_PAIR_WIDTH,
                            MX_PAIR,
                        ),
                        (owner, kb, 0),
                    ).flatten(1, 2),
                )
                mem_copy(
                    l0a.next_write(),
                    l1_a.next_read(),
                    copy_options(
                        mx_scale=from_tensor(
                            l1_sa.next_read(),
                            dtype=FLOAT8_E8M0,
                            layout=make_layout((token_tile_rows, scale_l0)),
                            data_format="zn",
                        )
                    ),
                )
            if const_expr(k_l1 == reduction_tile_cols):
                mem_copy(l0b.next_write(), w_ready, copy_options(mx_scale=s_ready))
            else:
                mem_copy(
                    l0b.next_write(),
                    tile_slice(
                        w_ready, (output_tile_cols, reduction_tile_cols), (0, kl0)
                    ),
                    copy_options(
                        mx_scale=tile_slice(
                            s_ready, (scale_l0, output_tile_cols), (kl0, 0)
                        )
                    ),
                )
            matmul(
                tile_slice(l0c, (token_tile_rows, output_tile_cols), (0, n)),
                l0a.next_read(),
                l0b.next_read(),
                init=(kb == 0),
                unit_flag=FIXPIPE_FINAL if kb + 1 == k_tiles else FIXPIPE_ACCUMULATE,
                mx_mode=True,
            )

    def _b_dst(self, gm_q, head_token_task_idx, subblock):
        """返回当前 Vector 核在 Q 中对应的 token 和特征切片。一个任务负责完整 head 或驻留半头；目的地特征块编号为2 * (task // owners_b) + subblock。"""
        tiling = self.tiling
        if const_expr(tiling.l1_reuse):
            dst = tile_slice(
                gm_q,
                (
                    tiling.bm_b // FIXPIPE_DUAL_DST,
                    tiling.head_size // FIXPIPE_DUAL_DST,
                ),
                (
                    FIXPIPE_DUAL_DST * (head_token_task_idx % tiling.owners_b)
                    + get_subblock_id(),
                    FIXPIPE_DUAL_DST * (head_token_task_idx // tiling.owners_b)
                    + subblock,
                ),
            )
        else:
            dst = tile_slice(
                gm_q,
                (tiling.bm_b, tiling.b_cols),
                (
                    head_token_task_idx % tiling.owners_b,
                    (head_token_task_idx // tiling.owners_b) * FIXPIPE_DUAL_DST
                    + subblock,
                ),
            )
        return dst

    @jit
    def _b_vec(
        self,
        ub_acc,
        ub_out,
        ub_tail,
        ub_sin,
        ub_cos,
        gm_sin,
        gm_cos,
        gm_q,
        head_token_task_idx,
        subblock,
    ):
        """对当前 Vector 核的 Q 切片执行尾部 RoPE，并转换为 BF16。"""
        tiling = self.tiling
        token_tile_rows = tiling.bm_b
        owner = head_token_task_idx % tiling.owners_b
        # 两个特征切片使用同一 token 对应的 RoPE 表行。
        q_half = self._b_dst(gm_q, head_token_task_idx, subblock)
        rows = q_half.shape[0]
        if const_expr(tiling.l1_reuse):
            q_accumulator = ub_acc.next_read()
            sin_tab, cos_tab = ub_sin, ub_cos
        else:
            sin_slot = ub_sin.next_write()
            mem_copy(sin_slot, tile_slice(gm_sin, (token_tile_rows, VL), (owner, 0)))
            cos_slot = ub_cos.next_write()
            mem_copy(cos_slot, tile_slice(gm_cos, (token_tile_rows, VL), (owner, 0)))
            q_accumulator = ub_acc.next_read()
            sin_tab = ub_sin.next_read()
            cos_tab = ub_cos.next_read()
        out_slot = ub_out.next_write()
        self._finish_q(
            q_accumulator,
            sin_tab,
            cos_tab,
            out_slot,
            ub_tail,
            rows,
            subblock,
            head_token_task_idx // tiling.owners_b,
        )
        #  mem_copy 以源域为准：非补齐 Q 输出的尾块需把 UB 源显式
        # 绑定到目的实际域，避免越界写满物理行。
        mem_copy(q_half, ub_out.next_read()[: q_half.shape[0], : q_half.shape[1]])

    @jit
    def _b_vec_reuse(
        self,
        ub_acc,
        ub_out,
        ub_tail,
        sin_tab,
        cos_tab,
        gm_q,
        head_token_task_idx,
        subblock,
    ):
        """复用当前 token 分块已加载到 UB 的 RoPE 表。"""
        tiling = self.tiling
        q_half = self._b_dst(gm_q, head_token_task_idx, subblock)
        rows = q_half.shape[0]
        q_accumulator = ub_acc.next_read()
        out_slot = ub_out.next_write()
        head_tile = head_token_task_idx // tiling.owners_b
        # RESIDENT_HEAD_PANELS：每 head 的 panel 半区数，仅后半半区（且第二目的地）含 RoPE。
        if head_tile % RESIDENT_HEAD_PANELS == 1 and subblock == 1:
            self._finish_q(
                q_accumulator,
                sin_tab,
                cos_tab,
                out_slot,
                ub_tail,
                rows,
                subblock,
                head_tile,
            )
        else:
            self._finish_q_plain(q_accumulator, out_slot, rows)
        # 同 _b_vec：尾块按目的实际域裁剪 UB 源。
        mem_copy(q_half, ub_out.next_read()[: q_half.shape[0], : q_half.shape[1]])

    @jit
    def _finish_q_plain(self, ub_acc, ub_out, rows):
        """对 head 中不涉及 RoPE 的三个四分块直接转换为 BF16。"""
        feature_cols = self.tiling.b_cols
        src_pitch = ub_acc.physical_layout.stride[0]
        out_pitch = ub_out.physical_layout.stride[0]
        with vf(mode="simd"):
            # 1. 读取 Q 向量；2. 舍入并打包为 BF16。
            full = rr.full_mask()
            for token_row in cannbotdsl.range(rows):
                q_row_offset = token_row * src_pitch
                bf16_row_offset = token_row * out_pitch
                for feature_vector_idx in range(feature_cols // VL):
                    rr.vstore_pack(
                        ub_out,
                        bf16_row_offset + feature_vector_idx * VL,
                        self._to_bf16(
                            rr.vload(ub_acc, q_row_offset + feature_vector_idx * VL),
                            full,
                        ),
                        full,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )

    @jit
    def _finish_q(
        self, ub_acc, ub_sin, ub_cos, ub_out, ub_tail, rows, subblock, head_tile
    ):
        """执行每个 head 的尾部 RoPE 并写出 BF16，不执行 RMSNorm。
        只有包含逻辑 head 尾部的 Vector 切片执行 RoPE；
        使用掩码控制，避免把 Channel 握手切入动态分支。"""
        tiling = self.tiling
        feature_cols, rope_width = tiling.b_cols, tiling.rope_width
        rope_width // 2
        # head 较窄时，最后一个特征向量的宽度可能小于 VL。
        tail = min(VL, feature_cols)
        src_pitch = ub_acc.physical_layout.stride[0]
        out_pitch = ub_out.physical_layout.stride[0]
        # interleave 输出由拥有 head 尾部的第二个 subblock 整段写出；另一个
        # subblock 只写前缀，避免两个 subblock 写同一段尾部。
        if const_expr(
            not tiling.l1_reuse
            and tiling.template == "split_k"
            # 72：固定形状特判，此时 RoPE 尾部仅位于逻辑 head 后半区。
            and tiling.token_count == 72
        ):
            rope_lanes = (
                0
                if head_tile % RESIDENT_HEAD_PANELS == 0
                else (0 if subblock == 0 else rope_width)
            )
        else:
            rope_lanes = 0 if subblock == 0 else rope_width
        with vf(mode="simd"):
            full = rr.full_mask()
            m_tail = rr.update_mask(tail, elem_bits=32)[0]
            m_rope = rr.update_mask(rope_lanes, elem_bits=32)[0]
            m_pair = rr.update_mask(rope_width // 2, elem_bits=32)[0]
            # 1. 将前缀特征转换为 BF16。
            for token_row in cannbotdsl.range(rows):
                q_row_offset = token_row * src_pitch
                bf16_row_offset = token_row * out_pitch
                for feature_vector_idx in range((feature_cols - tail) // VL):
                    rr.vstore_pack(
                        ub_out,
                        bf16_row_offset + feature_vector_idx * VL,
                        self._to_bf16(
                            rr.vload(ub_acc, q_row_offset + feature_vector_idx * VL),
                            full,
                        ),
                        full,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )
                # 直接写入两个半区不满足对齐时，先拼接完整尾部向量。
                # 第一个切片的 RoPE 掩码为空，保留原始值。
                # 2. 执行尾部 RoPE；3. 打包并对齐写出 BF16。
                if const_expr(rope_width % UB_ALIGN == 0 and not tiling.l1_reuse):
                    # BF16 目的地址按 32 字节对齐，可直接写入互不重叠的前缀和
                    # 两个 RoPE 半区，避免经过 UB 暂存区。
                    prefix_mask = rr.update_mask(tail - rope_lanes, elem_bits=32)[0]
                    rr.vstore_pack(
                        ub_out,
                        bf16_row_offset + feature_cols - tail,
                        self._to_bf16(
                            rr.vload(ub_acc, q_row_offset + feature_cols - tail),
                            prefix_mask,
                        ),
                        prefix_mask,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )
                    rope_vec = self._rope_pair(
                        ub_acc,
                        q_row_offset + feature_cols - rope_width,
                        None,
                        0,
                        ub_sin,
                        ub_cos,
                        token_row,
                        None,
                        m_pair,
                    )
                    rr.vstore_pack(
                        ub_out,
                        bf16_row_offset + feature_cols - rope_width,
                        self._to_bf16(rope_vec, m_rope),
                        m_rope,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )
                else:
                    rr.vstore(
                        ub_tail,
                        0,
                        rr.vload(ub_acc, q_row_offset + feature_cols - tail),
                        m_tail,
                    )
                    rope_vec = self._rope_pair(
                        ub_acc,
                        q_row_offset + feature_cols - rope_width,
                        None,
                        0,
                        ub_sin,
                        ub_cos,
                        token_row,
                        None,
                        m_pair,
                    )
                    rr.vstore(ub_tail, tail - rope_width, rope_vec, m_rope)
                    rr.vmem_bar("vst_vld")
                    rr.vstore_pack(
                        ub_out,
                        bf16_row_offset + feature_cols - tail,
                        self._to_bf16(rr.vload(ub_tail, 0), m_tail),
                        m_tail,
                        pack_mode=rr.PackMode.B32_TO_B16,
                    )

    @jit
    def _to_bf16(self, value, mask):
        return rr.vcast(value, dtypes.bfloat16, mask=mask, rounding=rr.RoundingMode.RN)


# host 侧入口

# torch dtype 到 DSL dtype 的静态映射（TensorSpec 构造用）。
_TORCH_TO_DSL_DTYPE = {
    torch.int8: dtypes.int8,
    torch.uint8: dtypes.uint8,
    torch.int64: dtypes.int64,
    torch.float32: dtypes.float32,
    torch.bfloat16: dtypes.bfloat16,
}

_COMPILED_KERNELS = OrderedDict()
_COMPILE_LOCK = threading.Lock()


# run 参数中 cache_index / kv_cache 的位置；动态模板只在这两个参数上使用 Dim。
_CACHE_INDEX_ARG = 12
_KV_CACHE_ARG = 13
