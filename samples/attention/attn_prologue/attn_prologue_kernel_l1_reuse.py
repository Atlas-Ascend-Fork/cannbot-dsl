# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""L1 驻留 AttnPrologue 流水及其专用后处理实现。"""

from __future__ import annotations

# fmt: off

import cannbotdsl as cbd
import cannbotdsl.reg as reg


from .attn_prologue_kernel_common import AttnPrologueCommon
from .attn_prologue_tiling import (A2_BUF_DEPTH_DECODE, AIV_PER_BLOCK, FIXPIPE_ACCUMULATE, FIXPIPE_DUAL_DST,
                                   FIXPIPE_FINAL, FLOAT8_E8M0, L0_DEPTH, MX_GROUP, MX_PAIR, MX_PAIR_WIDTH, NZ_M_FRAC,
                                   PANEL_OUTPUT_TILES, RESIDENT_K_WINDOWS, VL, W_BUF, ZN_FRACTAL_BYTES, ceil_div,
                                   kv_row_bytes)


@cbd.kernel
class L1ReuseAttnPrologueKernel(AttnPrologueCommon):
    """在共享驻留流水中完成 A1、A2 和 B 阶段。"""

    def __call__(self, gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_wqb, gm_dswqb, gm_wkv, gm_dswkv, gm_gamma_qr, gm_gamma_kv,
                 gm_sin, gm_cos, gm_index, gm_cache, gm_q, gm_qr, gm_dqr2d, gm_dqr3d, gm_ws_qa, gm_ws_kv,
                 norm_eps: cbd.dtypes.float32, gm_qr_bf16):
        """执行一次共享调度的 L1 驻留流水，完成 A1、A2 和 B 阶段。

        激活、MX scale、QA/KV 权重及部分和缓冲驻留 L1/L0，由 _resident_a_joint* 产出 QA/KV 的 FP32 部分和；
        随后在同步交换区的同时预取 QB panel，并由 _reduce_normalize_a 完成和发布 QR/scale。最后进入
        _stage_b_panels*，交错执行 QB Cube、Q 写回和延后的 KV 后处理；owners_b == 2 时使用 twin 模式，
        让两个 token 分块复用同一份静态权重和 QB panel。
        """
        tiling = self.tiling
        output_tile_cols = tiling.bn_a1
        panels_per_head = tiling.head_size // (PANEL_OUTPUT_TILES * output_tile_cols)
        # round3：两个 token 分块共用一次静态权重装载（twin）。
        twin_blocks = 2 if tiling.owners_b == 2 else 1
        prefetch_windows = 2 if twin_blocks == 2 else RESIDENT_K_WINDOWS
        seed_windows = 1 if twin_blocks == 2 else RESIDENT_K_WINDOWS
        b_groups = tiling.groups_b_pair if twin_blocks == 2 else tiling.groups_b
        l1_x, l1_sx, l1_w, l1_sw, l0a, l0b, l0c_a = self._alloc_shared_buffers(twin_blocks)
        # A 与 B 复用相同的物理 Channel，因此此部分容量按较大者计算。
        if cbd.const_expr(twin_blocks == 2):
            self._resident_a_joint_twin(l1_x, l1_sx, l1_w, l1_sw, l0a, l0b, l0c_a, gm_x, gm_dsx, gm_wqa, gm_dswqa,
                                        gm_ws_qa, gm_wkv, gm_dswkv, gm_ws_kv)
        else:
            self._resident_a_joint(l1_x, l1_sx, l1_w, l1_sw, l0a, l0b, l0c_a, gm_x, gm_dsx, gm_wqa, gm_dswqa,
                                   gm_ws_qa, gm_wkv, gm_dswkv, gm_ws_kv)
        # 发布 A1 交换区后，可在 AIV 归约期间加载 B panel。
        early_w, early_s = self._alloc_early_b_buffers(prefetch_windows)
        if cbd.get_block_idx() < tiling.owners_b * b_groups:
            if cbd.const_expr(twin_blocks == 2):
                seed_task = cbd.get_block_idx()
            else:
                seed_task = cbd.get_block_idx() % b_groups
            self._load_early_b_window(early_w, early_s, gm_wqb, gm_dswqb, seed_task,
                                      seed_task // panels_per_head, 0)
        self._publish_a1()
        if cbd.get_block_idx() < tiling.owners_b * b_groups:
            for u in cbd.range_constexpr(1, seed_windows):
                seed_task = cbd.get_block_idx() % b_groups
                self._load_early_b_window(early_w, early_s, gm_wqb, gm_dswqb, seed_task,
                                          seed_task // panels_per_head, u)
        self._reduce_normalize_a(gm_gamma_qr, gm_gamma_kv, gm_sin, gm_cos, gm_index, gm_cache, gm_qr, gm_dqr2d,
                                 gm_ws_qa, gm_ws_kv, norm_eps, gm_qr_bf16)
        if cbd.const_expr(twin_blocks == 2):
            self._stage_b_panels_twin(l1_x, l1_sx, l0a, l0b, gm_qr, gm_dqr3d, gm_wqb, gm_dswqb, gm_sin, gm_cos, gm_q,
                                      early_w, early_s)
        else:
            self._stage_b_panels(l1_x, l1_sx, l0a, l0b, gm_qr, gm_dqr3d, gm_wqb, gm_dswqb, gm_sin, gm_cos, gm_q,
                                 early_w, early_s)

    def _alloc_shared_buffers(self, twin_blocks):
        """分配 A/B 复用的激活、scale、权重和 L0 缓冲，保持 twin 槽位布局。"""
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (tiling.bm_a1, tiling.bn_a1, tiling.bk_a1)
        l1_x = cbd.RingBuffer(cbd.MemLoc.L1, cbd.dtypes.float8_e4m3fn,
                              cbd.make_layout((token_tile_rows, reduction_tile_cols)),
                              depth=tiling.resident_tiles * twin_blocks, data_format="nz")
        # E8M0 scale 的 zn 分形（(M1, G1, 16, 2)、每分形 32 字节）不能用
        # 每次选槽时用 from_tensor 绑定 zn 视图。
        l1_sx = cbd.RingBuffer(cbd.MemLoc.L1, cbd.dtypes.uint8,
                               cbd.make_layout((ceil_div(token_tile_rows, NZ_M_FRAC)
                                                * ceil_div(tiling.scale_k_tiles * reduction_tile_cols // MX_GROUP,
                                                           MX_PAIR) * ZN_FRACTAL_BYTES,)),
                               depth=tiling.resident_scale_tiles * twin_blocks, data_format="nd")
        l1_w = cbd.RingBuffer(cbd.MemLoc.L1, cbd.dtypes.int8,
                              cbd.make_layout((output_tile_cols, reduction_tile_cols)), depth=W_BUF, data_format="nz")
        l1_sw = cbd.RingBuffer(cbd.MemLoc.L1, cbd.dtypes.float8_e8m0,
                               cbd.make_layout((reduction_tile_cols // MX_GROUP, output_tile_cols)), depth=W_BUF,
                               data_format="nz")
        l0a = cbd.RingBuffer(cbd.MemLoc.L0A, cbd.dtypes.float8_e4m3fn,
                             cbd.make_layout((token_tile_rows, reduction_tile_cols)), depth=L0_DEPTH)
        l0b = cbd.RingBuffer(cbd.MemLoc.L0B, cbd.dtypes.float8_e4m3fn,
                             cbd.make_layout((output_tile_cols, reduction_tile_cols)), depth=L0_DEPTH)
        l0c_a = cbd.RingBuffer(cbd.MemLoc.L0C, cbd.dtypes.float32,
                               cbd.make_layout((token_tile_rows, output_tile_cols)), depth=twin_blocks)
        return l1_x, l1_sx, l1_w, l1_sw, l0a, l0b, l0c_a

    def _alloc_early_b_buffers(self, prefetch_windows):
        """分配 A2 期间预取的 QB payload/scale 缓冲。"""
        tiling = self.tiling
        output_tile_cols, reduction_tile_cols = tiling.bn_a1, tiling.bk_a1
        early_s = cbd.RingBuffer(cbd.MemLoc.L1, cbd.dtypes.float8_e8m0,
                                 cbd.make_layout((reduction_tile_cols // MX_GROUP, tiling.head_size)),
                                 depth=RESIDENT_K_WINDOWS, data_format="nz")
        early_w = cbd.RingBuffer(cbd.MemLoc.L1, cbd.dtypes.int8,
                                 cbd.make_layout((PANEL_OUTPUT_TILES * output_tile_cols, reduction_tile_cols)),
                                 depth=prefetch_windows, data_format="nz")
        return early_w, early_s

    @cbd.jit
    def _load_early_b_window(self, early_w, early_s, gm_wqb, gm_dswqb, task, head, window):
        """加载一个 early QB panel 及对应 scale，task/head/window 决定预打包坐标。"""
        tiling = self.tiling
        reduction_tile_cols = tiling.bk_a1
        scale_tile = cbd.tile_slice(gm_dswqb, (tiling.head_size, reduction_tile_cols // MX_PAIR_WIDTH, MX_PAIR),
                                    (head, window, 0))
        cbd.mem_copy(early_s.next_write(), scale_tile.permute(1, 2, 0).flatten(0, 1))
        self._load_b_panel(early_w.next_write(), gm_wqb, task, window)

    @cbd.jit
    def _publish_a1(self):
        """发布 A1 交换区并释放配对 AIV，使 A2 归约与 early B 加载重叠。"""
        cbd.cube_sync_block_arrive(cbd.PIPE.FIXPIPE, 1, mode=0)
        cbd.cube_sync_block_wait(cbd.PIPE.FIXPIPE, 1, mode=0)
        cbd.cube_sync_block_arrive(cbd.PIPE.FIXPIPE, 2, mode=2)
        cbd.vec_sync_block_wait(cbd.PIPE.MTE2, 2, mode=2)

    @cbd.jit
    def _load_activation_block(self, l1_x, l1_sx, gm_x, gm_dsx, token_tile_idx, split):
        """把一个 split-K 激活分块的 FP8 payload 和 E8M0 scale 装入 L1。

        l1_x/l1_sx 通过 depth-N 游标依次选择 K 窗口槽位，twin 的两个 token 分块连续占用槽位。
        payload 按 NZ tile 搬运，scale 先以原始字节保存，再绑定为 ZN 视图；返回的 tile_x/tile_s
        是后续投影可直接读取的 L1 视图列表。
        """
        tiling = self.tiling
        token_tile_rows, reduction_tile_cols = tiling.bm_a1, tiling.bk_a1
        nk = tiling.hidden_size // tiling.split_k // reduction_tile_cols
        tile_x = []
        for reduction_tile_idx in cbd.range_constexpr(nk):
            cbd.mem_copy(l1_x.next_write(),
                         cbd.tile_slice(gm_x, (token_tile_rows, reduction_tile_cols),
                                        (token_tile_idx, split * nk + reduction_tile_idx)))
            tile_x.append(l1_x.next_read())
        tile_s = []
        for scale_tile_idx in cbd.range_constexpr(nk // tiling.scale_k_tiles):
            activation_scale_slot = cbd.from_tensor(l1_sx.next_write(), dtype=FLOAT8_E8M0,
                                                    layout=cbd.make_layout((token_tile_rows,
                                                                            tiling.scale_k_tiles
                                                                            * reduction_tile_cols // MX_GROUP)),
                                                    data_format="zn")
            cbd.mem_copy(activation_scale_slot,
                         cbd.tile_slice(gm_dsx, (token_tile_rows,
                                                 tiling.scale_k_tiles * reduction_tile_cols // MX_PAIR_WIDTH, MX_PAIR),
                                        (token_tile_idx, split * (nk // tiling.scale_k_tiles) + scale_tile_idx,
                                         0)).flatten(1, 2))
            tile_s.append(activation_scale_slot)
        return tile_x, tile_s

    @cbd.jit
    def _resident_a_joint(self, l1_x, l1_sx, l1_w, l1_sw, l0a, l0b, l0c, gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_ws_qa,
                          gm_wkv, gm_dswkv, gm_ws_kv):
        """在非 twin 的 L1 驻留路径中联合调度 QA 与 KV 的 A1 投影任务。

        每个 job 对应一个 (token 分块, split-K, 输出组)，激活只加载一次，随后供该组内多个 QA/KV
        输出列块复用。任务按 block_idx 跨核均分；_resident_a_project 完成具体 MXFP8 矩阵乘，
        QA 和 KV 的部分和分别写入 gm_ws_qa/gm_ws_kv，暂不做归约或归一化。
        """
        tiling = self.tiling
        # 同一个激活块服务于 QA 和 KV 的多个输出列任务。
        groups = tiling.groups_a
        n_count = tiling.n_tiles_qa + tiling.n_tiles_kv
        for job in range(cbd.get_block_idx(), tiling.m_tiles_a1 * tiling.split_k * groups, tiling.block_dim):
            token_tile_idx = job // (tiling.split_k * groups)
            split = job // groups % tiling.split_k
            group = job % groups
            held_x, held_s = self._load_activation_block(l1_x, l1_sx, gm_x, gm_dsx, token_tile_idx, split)
            for output_tile_idx in range(group, n_count, groups):
                if output_tile_idx < tiling.n_tiles_qa:
                    self._resident_a_project(l1_w, l1_sw, l0a, l0b, l0c, held_x, held_s, gm_wqa, gm_dswqa, gm_ws_qa,
                                             output_tile_idx, token_tile_idx, split)
                else:
                    self._resident_a_project(l1_w, l1_sw, l0a, l0b, l0c, held_x, held_s, gm_wkv, gm_dswkv, gm_ws_kv,
                                             output_tile_idx - tiling.n_tiles_qa, token_tile_idx, split)

    @cbd.jit
    def _resident_a_project(self, l1_w, l1_sw, l0a, l0b, l0c, held_x, held_s, gm_w, gm_dsw, gm_ws, output_tile_idx,
                            token_tile_idx, split):
        """对已驻留激活完成一个输出列块的 split-K MXFP8 投影。

        按 W_BUF 成对装入权重窗口，将激活和 scale 搬入 L0A、权重和 scale 搬入 L0B，
        在 L0C 中按固定 K 顺序累加所有归约分块。首个分块初始化累加器，其余分块增量累加；
        最终把当前 token/split/output tile 的 FP32 部分和直接写入 GM 交换区，尾块按逻辑域裁剪。
        """
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (tiling.bm_a1, tiling.bn_a1, tiling.bk_a1)
        nk = tiling.hidden_size // tiling.split_k // reduction_tile_cols
        accumulator = l0c.next_write()
        for pair in range(ceil_div(nk, W_BUF)):
            for u in cbd.range_constexpr(min(W_BUF, nk)):
                reduction_tile_idx = pair * W_BUF + u
                activation_scale_tile = cbd.tile_slice(held_s[reduction_tile_idx // tiling.scale_k_tiles],
                                                       (token_tile_rows, reduction_tile_cols // MX_GROUP),
                                                       (0, reduction_tile_idx % tiling.scale_k_tiles))
                cbd.mem_copy(l0a.next_write(), held_x[reduction_tile_idx],
                             cbd.copy_options(mx_scale=activation_scale_tile))
                self._fill_w_a(l1_w, l1_sw, gm_w, gm_dsw, output_tile_idx, split * nk + reduction_tile_idx,
                               reduction_tile_cols // MX_PAIR_WIDTH)
            for u in cbd.range_constexpr(min(W_BUF, nk)):
                reduction_tile_idx = pair * W_BUF + u
                weight_l1_tile = cbd.from_tensor(l1_w.next_read(), cbd.dtypes.float8_e4m3fn)
                weight_scale_tile = l1_sw.next_read()
                cbd.mem_copy(l0b.next_write(), weight_l1_tile, cbd.copy_options(mx_scale=weight_scale_tile))
                weight_l0_tile = l0b.next_read()
                cbd.matmul(accumulator, l0a.next_read(), weight_l0_tile, init=reduction_tile_idx == 0, mx_mode=True)
        #  mem_copy 以源域为准：将 L0C 源逻辑域重绑定到目的实际域，
        # 尾块裁剪生效且物理布局保持不变。
        ws_tile = cbd.tile_slice(gm_ws, (token_tile_rows, output_tile_cols),
                                 (split * tiling.m_tiles_a1 + token_tile_idx, output_tile_idx))
        cbd.mem_copy(ws_tile, cbd.from_tensor(accumulator, layout=cbd.make_layout(ws_tile.shape),
                                              physical_layout=accumulator.physical_layout))

    @cbd.jit
    def _resident_a_joint_twin(self, l1_x, l1_sx, l1_w, l1_sw, l0a, l0b, l0c, gm_x, gm_dsx, gm_wqa, gm_dswqa, gm_ws_qa,
                               gm_wkv, gm_dswkv, gm_ws_kv):
        """联合调度 twin 驻留路径的 QA/KV 投影，让两个 token 分块共享静态权重。

        job 只枚举 (split-K, 输出组)，两个 token 分块在 job 内一次性加载并分别持有 L1 读视图。
        随后按输出列块调用 _resident_a_project_twin，同一份权重依次服务两个 L0C 累加器，
        减少重复权重搬运；QA/KV 仍分别写入独立的部分和交换区。
        """
        tiling = self.tiling
        groups = tiling.groups_a
        n_count = tiling.n_tiles_qa + tiling.n_tiles_kv
        for job in range(cbd.get_block_idx(), tiling.split_k * groups, tiling.block_dim):
            split = job // groups
            group = job % groups
            held_x = []
            held_s = []
            for tt in cbd.range_constexpr(2):
                tile_x, tile_s = self._load_activation_block(l1_x, l1_sx, gm_x, gm_dsx, tt, split)
                held_x.append(tile_x)
                held_s.append(tile_s)
            for output_tile_idx in range(group, n_count, groups):
                if output_tile_idx < tiling.n_tiles_qa:
                    self._resident_a_project_twin(l1_w, l1_sw, l0a, l0b, l0c, held_x, held_s, gm_wqa, gm_dswqa,
                                                  gm_ws_qa, output_tile_idx, split)
                else:
                    self._resident_a_project_twin(l1_w, l1_sw, l0a, l0b, l0c, held_x, held_s, gm_wkv, gm_dswkv,
                                                  gm_ws_kv, output_tile_idx - tiling.n_tiles_qa, split)

    @cbd.jit
    def _resident_a_project_twin(self, l1_w, l1_sw, l0a, l0b, l0c, held_x, held_s, gm_w, gm_dsw, gm_ws, output_tile_idx,
                                 split):
        """用同一权重窗口完成两个 token 分块的 A1 MXFP8 投影。

        每个 K 窗口只执行一次权重及 scale 的 L1/L0B 装载；随后分别为两个 token 分块搬运 L0A
        并调用 matmul，结果保存在两个独立 L0C 中。K 轴全部累加后，将两个 FP32 部分和分别写回
        gm_ws 中对应的 split/token/output tile，从而把权重搬运成本摊薄到两次矩阵乘。
        """
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (tiling.bm_a1, tiling.bn_a1, tiling.bk_a1)
        nk = tiling.hidden_size // tiling.split_k // reduction_tile_cols
        held_c = [l0c.next_write(), l0c.next_write()]
        for pair in range(ceil_div(nk, W_BUF)):
            for u in cbd.range_constexpr(min(W_BUF, nk)):
                reduction_tile_idx = pair * W_BUF + u
                self._fill_w_a(l1_w, l1_sw, gm_w, gm_dsw, output_tile_idx, split * nk + reduction_tile_idx,
                               reduction_tile_cols // MX_PAIR_WIDTH)
            for u in cbd.range_constexpr(min(W_BUF, nk)):
                reduction_tile_idx = pair * W_BUF + u
                weight_l1_tile = cbd.from_tensor(l1_w.next_read(), cbd.dtypes.float8_e4m3fn)
                weight_scale_tile = l1_sw.next_read()
                cbd.mem_copy(l0b.next_write(), weight_l1_tile, cbd.copy_options(mx_scale=weight_scale_tile))
                weight_l0_tile = l0b.next_read()
                for tt in cbd.range_constexpr(2):
                    activation_scale_tile = cbd.tile_slice(held_s[tt][reduction_tile_idx // tiling.scale_k_tiles],
                                                           (token_tile_rows, reduction_tile_cols // MX_GROUP),
                                                           (0, reduction_tile_idx % tiling.scale_k_tiles))
                    cbd.mem_copy(l0a.next_write(), held_x[tt][reduction_tile_idx],
                                 cbd.copy_options(mx_scale=activation_scale_tile))
                    cbd.matmul(held_c[tt], l0a.next_read(), weight_l0_tile, init=reduction_tile_idx == 0, mx_mode=True)
        for tt in cbd.range_constexpr(2):
            ws_tile = cbd.tile_slice(gm_ws, (token_tile_rows, output_tile_cols), (split * tiling.m_tiles_a1 + tt,
                                                                                  output_tile_idx))
            cbd.mem_copy(ws_tile, cbd.from_tensor(held_c[tt], layout=cbd.make_layout(ws_tile.shape),
                                                  physical_layout=held_c[tt].physical_layout))

    @cbd.jit
    def _load_b_panel(self, dst, gm_weight, n_group, k_panel):
        """加载 QB 的一个宽权重 panel，覆盖连续多个 N 输出 tile。

        n_group 指定 head 内的 panel 任务，k_panel 指定归约窗口；函数根据 rank_size/bk_b
        将二维任务坐标换算为预打包权重中的行号，把 panel_bytes 原始字节直接搬到 L1。
        目的暂以 ND 一维字节视图承接搬运，调用方再重绑定为 FP8 NZ 视图并切分各个 N 子块。
        """
        tiling = self.tiling
        panel_bytes = PANEL_OUTPUT_TILES * tiling.bn_b * tiling.bk_b
        cbd.mem_copy(cbd.from_tensor(dst, layout=cbd.make_layout((1, panel_bytes)), data_format="nd"),
                     cbd.tile_slice(gm_weight, (1, panel_bytes), (n_group * (tiling.rank_size // tiling.bk_b) + k_panel,
                                                                  0)))

    @cbd.jit
    def _stage_b_panels(self, l1_x, l1_sx, l0a, l0b, gm_qr, gm_dqr, gm_w, gm_dsw, gm_sin, gm_cos, gm_q, early_w,
                        early_s):
        """驱动非 twin 驻留路径的 QB panel 流水和 Q 输出后处理。

        为当前 owner/token 分块准备 L0C、FP32/BF16 UB 以及 RoPE 表，按 (owner, group) 分配任务。
        首个 panel 使用 A2 期间预取的 early_w/early_s，后续任务在进入 _resident_b_cube 前重新填充；
        具体 Cube 累加、后续 panel 预取、FixPipe 分发和 Vector 写回由被调用函数协同完成。
        """
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (tiling.bm_b, tiling.bn_b, tiling.bk_b)
        panels_per_head = tiling.head_size // (PANEL_OUTPUT_TILES * output_tile_cols)
        l0c = cbd.Buffer(cbd.MemLoc.L0C, cbd.dtypes.float32, cbd.make_layout((token_tile_rows, tiling.head_size)))
        half, feature_cols = (token_tile_rows // FIXPIPE_DUAL_DST, tiling.head_size // FIXPIPE_DUAL_DST)
        ub_acc = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((half, feature_cols)), depth=2)
        ub_out = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.bfloat16, cbd.make_layout((half, feature_cols)), depth=2)
        ub_tail = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((1, VL)))
        ub_sin = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((half, VL)), depth=1)
        ub_cos = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((half, VL)), depth=1)
        groups = tiling.groups_b
        for job in range(cbd.get_block_idx(), tiling.owners_b * groups, tiling.block_dim):
            owner = job // groups
            group = job % groups
            # 首个任务已在 A2 期间预取；后续 token 分块需先完成前一个 head
            # 循环，再重新填充所有 panel Channel。
            if job >= tiling.block_dim:
                for u in cbd.range_constexpr(RESIDENT_K_WINDOWS):
                    scale_tile = cbd.tile_slice(gm_dsw, (tiling.head_size,
                                                         reduction_tile_cols // MX_PAIR_WIDTH, MX_PAIR),
                                                (group // panels_per_head, u, 0))
                    cbd.mem_copy(early_s.next_write(), scale_tile.permute(1, 2, 0).flatten(0, 1))
                    self._load_b_panel(early_w.next_write(), gm_w, group, u)
            cbd.mem_copy(ub_sin.next_write(),
                         cbd.tile_slice(gm_sin, (half, VL), (owner * AIV_PER_BLOCK + cbd.get_subblock_id(), 0)))
            cbd.mem_copy(ub_cos.next_write(),
                         cbd.tile_slice(gm_cos, (half, VL), (owner * AIV_PER_BLOCK + cbd.get_subblock_id(), 0)))
            sin_tab = ub_sin.next_read()
            cos_tab = ub_cos.next_read()
            self._resident_b_cube(l1_x, l1_sx, l0a, l0b, l0c, ub_acc, gm_qr, gm_dqr, gm_w, gm_dsw, owner, group, groups,
                                  ub_out, ub_tail, sin_tab, cos_tab, gm_sin, gm_cos, gm_q, early_w, early_s)

    @cbd.jit
    def _load_qr_partials(self, l1_x, l1_sx, gm_qr, gm_dqr, owner):
        """把一个 QB owner 的 QR token 分块完整预装入 L1。

        l1_x/l1_sx 通过 depth-N 游标依次持有全部 K 窗口，scale 以 ZN 视图供 mx_scale 使用；
        twin 的两个 owner 连续占用槽位。返回 held_x/held_s 列表后，同一 QR 分块可被多个 head/panel
        复用，避免在 B 阶段重复访问 GM。
        """
        tiling = self.tiling
        token_tile_rows, reduction_tile_cols = tiling.bm_b, tiling.bk_b
        nk = tiling.rank_size // reduction_tile_cols
        held_x = []
        held_s = []
        for k in cbd.range_constexpr(nk):
            cbd.mem_copy(l1_x.next_write(), cbd.tile_slice(gm_qr, (token_tile_rows, reduction_tile_cols), (owner, k)))
            held_x.append(l1_x.next_read())
        for sk in cbd.range_constexpr(nk // tiling.scale_k_tiles):
            qr_scale_slot = cbd.from_tensor(l1_sx.next_write(), dtype=FLOAT8_E8M0,
                                            layout=cbd.make_layout((token_tile_rows,
                                                                    tiling.scale_k_tiles
                                                                    * reduction_tile_cols // MX_GROUP)),
                                            data_format="zn")
            cbd.mem_copy(qr_scale_slot,
                         cbd.tile_slice(gm_dqr, (token_tile_rows,
                                                 tiling.scale_k_tiles * reduction_tile_cols // MX_PAIR_WIDTH, MX_PAIR),
                                        (owner, sk, 0)).flatten(1, 2))
            held_s.append(qr_scale_slot)
        return held_x, held_s

    @cbd.jit
    def _resident_b_cube(self, l1_x, l1_sx, l0a, l0b, l0c, ub_acc, gm_qr, gm_dqr, gm_w, gm_dsw, owner, group, groups,
                         ub_out, ub_tail, ub_sin, ub_cos, gm_sin, gm_cos, gm_q, early_w, early_s):
        """执行非 twin 驻留 B 阶段的一个 owner/group 全部 Cube 与 Vector 工作。

        QR payload/scale 在本函数开始时一次装入 L1，随后按 (head, half-head) 遍历输出 panel。
        每个 K panel 调用 _b_panel_l0_reuse 累加到完整 head 的 L0C，并在消费当前权重的窗口中预取
        下一 unit 的权重；K 归约结束后经 FixPipe split-M 分发给两个 Vector 核，由 _b_vec 完成 RoPE
        和 BF16 Q 写回。该粒度用于改善多 head 在有限核数上的负载均衡。
        """
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (tiling.bm_b, tiling.bn_b, tiling.bk_b)
        nk = tiling.rank_size // reduction_tile_cols
        panels_per_head = tiling.head_size // (PANEL_OUTPUT_TILES * output_tile_cols)
        l1_head_scale = early_s
        l1_b_wide = early_w
        held_x, held_s = self._load_qr_partials(l1_x, l1_sx, gm_qr, gm_dqr, owner)
        # round6：非 twin B 的搬运任务粒度从整 head 改为 (head, half) 半头任务，
        # 使 64 head / 28 核 的 3:2 不均衡压到 5:4 半头（critic 核搬运 -16.7%）。
        unit_total = tiling.head_count * panels_per_head
        for unit in range(group, unit_total, groups):
            head = unit // panels_per_head
            half = unit % panels_per_head
            for panel in cbd.range_constexpr(nk):
                head_scale = l1_head_scale.next_read()
                wide_w = cbd.from_tensor(l1_b_wide.next_read(), cbd.dtypes.float8_e4m3fn)
                self._b_panel_l0_reuse(wide_w, head_scale, l0a, l0b, l0c, held_x, held_s, half, panel)
                next_unit = unit + groups
                if next_unit < unit_total:
                    next_head = next_unit // panels_per_head
                    scale_tile = cbd.tile_slice(gm_dsw, (tiling.head_size,
                                                         reduction_tile_cols // MX_PAIR_WIDTH, MX_PAIR),
                                                (next_head, panel, 0))
                    cbd.mem_copy(l1_head_scale.next_write(), scale_tile.permute(1, 2, 0).flatten(0, 1))
                    self._load_b_panel(l1_b_wide.next_write(), gm_w, next_unit, panel)
                if cbd.const_expr(panel + 1 == nk):
                    # FixPipe 双目的地（split-M）：L0C 源 partition(dim=0) 将
                    # 行数对半分给两个 Vector 核。
                    cbd.mem_copy(ub_acc.next_write(),
                                 cbd.from_tensor(cbd.tile_slice(l0c, (token_tile_rows,
                                                                      PANEL_OUTPUT_TILES * output_tile_cols),
                                                                (0, half)),
                                                 layout=cbd.make_layout((token_tile_rows, PANEL_OUTPUT_TILES
                                                                         * output_tile_cols))).partition(dim=0),
                                 cbd.copy_options(unit_flag=FIXPIPE_FINAL))
                    self._b_vec(ub_acc, ub_out, ub_tail, ub_sin, ub_cos, gm_q, head * tiling.owners_b + owner, half)

    @cbd.jit
    def _stage_b_panels_twin(self, l1_x, l1_sx, l0a, l0b, gm_qr, gm_dqr, gm_w, gm_dsw, gm_sin, gm_cos, gm_q, early_w,
                             early_s):
        """为 twin 驻留 B 阶段准备两个 token 分块的共享缓冲和 RoPE 表。

        两个 owner 共用 QB panel 任务队列，但分别持有 L0C、RoPE sin/cos 视图及最终输出位置。
        本函数按 head/half-head 建立任务总数，并一次性加载两个 token 分块的 RoPE 表，随后把主要
        Cube/Vector 流水交给 _resident_b_cube_twin；half-head 粒度用于减小核间任务数量差异。
        """
        tiling = self.tiling
        token_tile_rows, output_tile_cols, _reduction_tile_cols = (tiling.bm_b, tiling.bn_b, tiling.bk_b)
        panels_per_head = tiling.head_size // (PANEL_OUTPUT_TILES * output_tile_cols)
        l0c = [cbd.Buffer(cbd.MemLoc.L0C, cbd.dtypes.float32,
                          cbd.make_layout((token_tile_rows, PANEL_OUTPUT_TILES * output_tile_cols))) for _ in range(2)]
        half, feature_cols = (token_tile_rows // FIXPIPE_DUAL_DST, tiling.head_size // FIXPIPE_DUAL_DST)
        ub_acc = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((half, feature_cols)), depth=2)
        ub_out = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.bfloat16, cbd.make_layout((half, feature_cols)), depth=2)
        ub_tail = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((1, VL)))
        ub_sin = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((half, VL)), depth=2)
        ub_cos = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((half, VL)), depth=2)
        # round4：twin B 任务粒度改为 (head, half) 的 128 个 half-head task，按 28 核轮转，
        # 把 8 核 3 head / 20 核 2 head 压到 16 核 5 half / 12 核 4 half（tail 由 8 核变 16 核）。
        task_total = tiling.head_count * panels_per_head
        for owner in cbd.range_constexpr(2):
            cbd.mem_copy(ub_sin.next_write(),
                         cbd.tile_slice(gm_sin, (half, VL), (owner * AIV_PER_BLOCK + cbd.get_subblock_id(), 0)))
            cbd.mem_copy(ub_cos.next_write(),
                         cbd.tile_slice(gm_cos, (half, VL), (owner * AIV_PER_BLOCK + cbd.get_subblock_id(), 0)))
        sin_tab = [ub_sin.next_read(), ub_sin.next_read()]
        cos_tab = [ub_cos.next_read(), ub_cos.next_read()]
        self._resident_b_cube_twin(l1_x, l1_sx, l0a, l0b, l0c, ub_acc, gm_qr, gm_dqr, gm_w, gm_dsw, gm_sin, gm_cos,
                                   task_total, ub_out, ub_tail, sin_tab, cos_tab, gm_q, early_w, early_s)

    @cbd.jit
    def _resident_b_cube_twin(self, l1_x, l1_sx, l0a, l0b, l0c, ub_acc, gm_qr, gm_dqr, gm_w, gm_dsw, gm_sin, gm_cos,
                              task_total, ub_out, ub_tail, sin_tab, cos_tab, gm_q, early_w, early_s):
        """在 twin 驻留路径中让两个 token 分块共享 QB 权重和 panel 搬运。

        开始时将两个 owner 的全部 QR K 窗口驻留 L1；每个 (head, half) 任务内，同一权重窗口依次
        与两个 token 的 L0A 做 MXFP8 matmul，累加到独立 L0C。K 轴完成后分别通过 FixPipe split-M
        和 _b_vec 输出 BF16 Q，同时预取下一 K 窗口或下一 head 任务的 payload/scale，保持 panel
        装载与矩阵流水重叠。
        """
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (tiling.bm_b, tiling.bn_b, tiling.bk_b)
        nk = tiling.rank_size // reduction_tile_cols
        panels_per_head = tiling.head_size // (PANEL_OUTPUT_TILES * output_tile_cols)
        panels_per_head * nk
        held_x = []
        held_s = []
        for owner in cbd.range_constexpr(2):
            tile_x, tile_s = self._load_qr_partials(l1_x, l1_sx, gm_qr, gm_dqr, owner)
            held_x.append(tile_x)
            held_s.append(tile_s)
        for task in range(cbd.get_block_idx(), task_total, tiling.block_dim):
            head = task // panels_per_head
            half = task % panels_per_head
            for reduction_tile_idx in cbd.range_constexpr(nk):
                head_scale = early_s.next_read()
                wide_w = cbd.from_tensor(early_w.next_read(), cbd.dtypes.float8_e4m3fn)
                # 一次 QR K 窗口装载服务两个 token 分块。
                for owner in cbd.range_constexpr(2):
                    activation_scale_tile = cbd.tile_slice(held_s[owner][reduction_tile_idx // tiling.scale_k_tiles],
                                                           (token_tile_rows, reduction_tile_cols // MX_GROUP),
                                                           (0, reduction_tile_idx % tiling.scale_k_tiles))
                    cbd.mem_copy(l0a.next_write(), held_x[owner][reduction_tile_idx],
                                 cbd.copy_options(mx_scale=activation_scale_tile))
                held_a = [l0a.next_read(), l0a.next_read()]
                for n_sub in cbd.range_constexpr(PANEL_OUTPUT_TILES):
                    output_tile_idx = half * PANEL_OUTPUT_TILES + n_sub
                    weight_l1_tile = cbd.tile_slice(wide_w, (output_tile_cols, reduction_tile_cols), (n_sub, 0))
                    weight_scale_tile = cbd.tile_slice(head_scale, (reduction_tile_cols // MX_GROUP, output_tile_cols),
                                                       (0, output_tile_idx))
                    cbd.mem_copy(l0b.next_write(), weight_l1_tile, cbd.copy_options(mx_scale=weight_scale_tile))
                    weight_l0_tile = l0b.next_read()
                    for owner in cbd.range_constexpr(2):
                        cbd.matmul(cbd.tile_slice(l0c[owner], (token_tile_rows, output_tile_cols), (0, n_sub)),
                                   held_a[owner], weight_l0_tile, init=reduction_tile_idx == 0,
                                   unit_flag=(FIXPIPE_FINAL if reduction_tile_idx + 1 == nk else FIXPIPE_ACCUMULATE),
                                   mx_mode=True)
                if cbd.const_expr(reduction_tile_idx + 1 == nk):
                    for owner in cbd.range_constexpr(2):
                        cbd.mem_copy(ub_acc.next_write(),
                                     cbd.from_tensor(cbd.tile_slice(l0c[owner], (token_tile_rows,
                                                                                 PANEL_OUTPUT_TILES * output_tile_cols),
                                                                    (0, 0)),
                                                     layout=cbd.make_layout((token_tile_rows, PANEL_OUTPUT_TILES
                                                                             * output_tile_cols))).partition(dim=0),
                                     cbd.copy_options(unit_flag=FIXPIPE_FINAL))
                        self._b_vec(ub_acc, ub_out, ub_tail, sin_tab[owner], cos_tab[owner], gm_q,
                                    head * tiling.owners_b + owner, half)
                next_task = task
                next_kw = reduction_tile_idx + 1
                if cbd.const_expr(reduction_tile_idx + 1 == nk):
                    next_task = task + tiling.block_dim
                    next_kw = 0
                if next_task < task_total:
                    next_head = next_task // panels_per_head
                    scale_tile = cbd.tile_slice(gm_dsw,
                                                (tiling.head_size, reduction_tile_cols // MX_PAIR_WIDTH, MX_PAIR),
                                                (next_head, next_kw, 0))
                    cbd.mem_copy(early_s.next_write(), scale_tile.permute(1, 2, 0).flatten(0, 1))
                    self._load_b_panel(early_w.next_write(), gm_w, next_task, next_kw)

    @cbd.jit
    def _b_panel_l0_reuse(self, wide_w, head_scale, l0a, l0b, l0c, held_x, held_s, n_group, reduction_tile_idx):
        """复用一次 L0A 激活装载，计算 QB 宽 panel 中连续多个 N 子块。

        reduction_tile_idx 选择已驻留的 QR K 窗口，l0a 游标选择 L0A 槽位，wide_w/head_scale 是同窗口的宽权重
        和整 head scale。函数对 panel 内每个输出子块搬运对应 L0B/scale，并累加到完整 head 的
        L0C 列切片；首 K 块初始化、末 K 块标记 FixPipe 完成，中间块只累加。
        """
        tiling = self.tiling
        token_tile_rows, output_tile_cols, reduction_tile_cols = (tiling.bm_b, tiling.bn_b, tiling.bk_b)
        nk = tiling.rank_size // reduction_tile_cols
        # 一个 QR K320 分块服务于打包 panel 中的四个 N64 切片。
        activation_l0_slot = l0a.next_write()
        for resident_reduction_idx in cbd.range_constexpr(nk):
            if reduction_tile_idx == resident_reduction_idx:
                activation_scale_tile = cbd.tile_slice(held_s[resident_reduction_idx // tiling.scale_k_tiles],
                                                       (token_tile_rows, reduction_tile_cols // MX_GROUP),
                                                       (0, resident_reduction_idx % tiling.scale_k_tiles))
                cbd.mem_copy(activation_l0_slot, held_x[resident_reduction_idx],
                             cbd.copy_options(mx_scale=activation_scale_tile))
        held_a = l0a.next_read()
        for n_sub in range(PANEL_OUTPUT_TILES):
            output_tile_idx = n_group * PANEL_OUTPUT_TILES + n_sub
            weight_l1_tile = cbd.tile_slice(wide_w, (output_tile_cols, reduction_tile_cols), (n_sub, 0))
            weight_scale_tile = cbd.tile_slice(head_scale, (reduction_tile_cols // MX_GROUP, output_tile_cols),
                                               (0, output_tile_idx))
            cbd.mem_copy(l0b.next_write(), weight_l1_tile, cbd.copy_options(mx_scale=weight_scale_tile))
            cbd.matmul(cbd.tile_slice(l0c, (token_tile_rows, output_tile_cols), (0, output_tile_idx)), held_a,
                       l0b.next_read(), init=reduction_tile_idx == 0,
                       unit_flag=(FIXPIPE_FINAL if reduction_tile_idx + 1 == nk else FIXPIPE_ACCUMULATE),
                       mx_mode=True)

    @cbd.jit
    def _qr_tile(self, gm_ws_qa, gm_qr, gm_qr_bf16, gm_dqr2d, ub_qa, ub_qr, ub_qr_bf16, ub_exp, ub_scale, gamma_qr,
                 qa_sum, token_tile_idx, rows, norm_eps):
        """一次加载驻留路径的全部 QA split，顺序归约后量化并写出 QR 及 scale。"""
        tiling = self.tiling
        token_tile_rows, rank = tiling.bm_a2, tiling.rank_size
        partials = gm_ws_qa.view(tiling.split_k, tiling.m_tiles_a1 * tiling.bm_a1 * rank)
        cbd.mem_copy(ub_qa.next_write(), cbd.tile_slice(partials, (tiling.split_k, rank), (0, token_tile_idx)))
        qa = ub_qa.next_read()
        qr_slot = ub_qr.next_write()
        qr_bf16_slot = ub_qr_bf16.next_write()
        self._acc_all_partials(qa_sum, qa, rank)
        self._quantize_qr(qa_sum, gamma_qr, qr_slot, ub_qr_bf16, qr_bf16_slot, ub_exp, rows, norm_eps)
        scale_slot = ub_scale.next_write()
        self._pack_scale(ub_exp, scale_slot, rows)
        cbd.mem_copy(cbd.tile_slice(gm_qr, (token_tile_rows, rank), (token_tile_idx, 0)), ub_qr.next_read())
        cbd.mem_copy(cbd.tile_slice(gm_qr_bf16, (token_tile_rows, rank), (token_tile_idx, 0)), ub_qr_bf16.next_read())
        cbd.mem_copy(cbd.tile_slice(gm_dqr2d, (token_tile_rows, tiling.qr_groups), (token_tile_idx, 0)),
                     cbd.from_tensor(ub_scale.next_read(),
                                     layout=cbd.make_layout((token_tile_rows, tiling.qr_groups),
                                                            stride=(tiling.scale_stride, 1))))

    @cbd.jit
    def _reduce_normalize_a(self, gm_gamma_qr, gm_gamma_kv, gm_sin, gm_cos, gm_index, gm_cache, gm_qr, gm_dqr2d,
                            gm_ws_qa, gm_ws_kv, norm_eps, gm_qr_bf16):
        """完成驻留路径的 QR 归约量化与发布，随后处理 KV 并写回 cache。"""
        tiling = self.tiling
        token_tile_rows, rank, head = tiling.bm_a2, tiling.rank_size, tiling.head_size

        a2_depth = A2_BUF_DEPTH_DECODE
        ub_qa = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((tiling.split_k, rank)), depth=1)
        ub_kv = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((tiling.split_k, head)), depth=1)
        ub_gamma_qr = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((1, rank)), depth=1)
        ub_gamma_kv = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((1, head)), depth=1)
        ub_sin = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((token_tile_rows, VL)), depth=1)
        ub_cos = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((token_tile_rows, VL)), depth=1)
        ub_qr = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.uint8,
                               cbd.make_layout((token_tile_rows, rank)), depth=a2_depth)
        # 量化前的 BF16 QR 输出；与 FP8 payload 同步写出，供下游算子消费。
        ub_qr_bf16 = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.bfloat16,
                                    cbd.make_layout((token_tile_rows, rank)), depth=a2_depth)
        # 指数暂存仅在 VF 区域间传递，多分配一行以容纳完整向量读取。
        ub_exp = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.int32,
                            cbd.make_layout((token_tile_rows + 1,
                                             max(tiling.exp_stride, VL))))
        # 显式指定对齐的 scale 行跨度，DMA 只暴露公开的有效列。
        ub_scale = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.uint8,
                                  cbd.make_layout((token_tile_rows, tiling.scale_stride)), depth=a2_depth)
        ub_kvb = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.uint8, cbd.make_layout((1, kv_row_bytes(head))), depth=2)
        ub_tail = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((1, 2 * VL)))
        ub_kv_bits = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.int32, cbd.make_layout((VL,)))

        cbd.mem_copy(ub_gamma_qr.next_write(), cbd.tile_slice(gm_gamma_qr, (1, rank), (0, 0)))
        cbd.mem_copy(ub_gamma_kv.next_write(), cbd.tile_slice(gm_gamma_kv, (1, head), (0, 0)))
        gamma_qr = ub_gamma_qr.next_read()
        gamma_kv = ub_gamma_kv.next_read()
        qa_sum = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((token_tile_rows, rank)))
        kv_sum = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((token_tile_rows, head)))
        aiv = cbd.get_block_idx() * AIV_PER_BLOCK + cbd.get_subblock_id()
        # 实际 T 由运行时输入决定；填充行不进入归约与 cache 写回。
        tokens = gm_index.shape[0]
        actual_tiles = (tokens + token_tile_rows - 1) // token_tile_rows
        for token_tile_idx in range(aiv, actual_tiles, tiling.block_dim * AIV_PER_BLOCK):
            self._qr_tile(gm_ws_qa, gm_qr, gm_qr_bf16, gm_dqr2d, ub_qa, ub_qr, ub_qr_bf16, ub_exp, ub_scale, gamma_qr,
                          qa_sum, token_tile_idx, token_tile_rows, norm_eps)

        self._publish_qr()
        for token_tile_idx in range(aiv, actual_tiles, tiling.block_dim * AIV_PER_BLOCK):
            self._kv_tile(gm_ws_kv, gm_sin, gm_cos, gm_index, gm_cache, ub_kv, ub_sin, ub_cos, ub_kvb, ub_tail,
                          ub_kv_bits, gamma_kv, kv_sum, token_tile_idx, token_tile_rows, norm_eps)

    @cbd.jit
    def _kv_tile(self, gm_ws_kv, gm_sin, gm_cos, gm_index, gm_cache, ub_kv, ub_sin, ub_cos, ub_kvb, ub_tail, ub_kv_bits,
                 gamma_kv, kv_sum, token_tile_idx, rows, norm_eps):
        """一次归约驻留路径的全部 KV split，并逐 token 完成归一化、RoPE、量化和 cache 写回。"""
        tiling = self.tiling
        token_tile_rows, head = tiling.bm_a2, tiling.head_size
        partials = gm_ws_kv.view(tiling.split_k, tiling.m_tiles_a1 * tiling.bm_a1 * head)
        cbd.mem_copy(ub_kv.next_write(), cbd.tile_slice(partials, (tiling.split_k, head), (0, token_tile_idx)))
        cbd.mem_copy(ub_sin.next_write(), cbd.tile_slice(gm_sin, (token_tile_rows, VL), (token_tile_idx, 0)))
        cbd.mem_copy(ub_cos.next_write(), cbd.tile_slice(gm_cos, (token_tile_rows, VL), (token_tile_idx, 0)))
        kv = ub_kv.next_read()
        self._acc_all_partials(kv_sum, kv, head)
        sin_tab = ub_sin.next_read()
        cos_tab = ub_cos.next_read()
        for j in cbd.range(rows):
            kvb_slot = ub_kvb.next_write()
            self._finish_kv_row(kv_sum, gamma_kv, sin_tab, cos_tab, kvb_slot, ub_tail, ub_kv_bits, j, j, norm_eps)
            slot = gm_index[token_tile_idx * token_tile_rows + j, 0]
            if slot >= 0:
                if slot < gm_cache.shape[0]:
                    cbd.mem_copy(cbd.tile_slice(gm_cache, (1, kv_row_bytes(head)), (slot, 0)), ub_kvb.next_read())

    @cbd.jit
    def _b_dst(self, gm_q, head_token_task_idx, subblock):
        """按驻留路径的 split-M 布局计算当前 Vector 核的 GM Q 写回切片。"""
        tiling = self.tiling
        return cbd.tile_slice(gm_q, (tiling.bm_b // FIXPIPE_DUAL_DST, tiling.head_size // FIXPIPE_DUAL_DST),
                              (FIXPIPE_DUAL_DST * (head_token_task_idx % tiling.owners_b) + cbd.get_subblock_id(),
                               FIXPIPE_DUAL_DST * (head_token_task_idx // tiling.owners_b) + subblock))

    @cbd.jit
    def _b_vec(self, ub_acc, ub_out, ub_tail, sin_tab, cos_tab, gm_q, head_token_task_idx, subblock):
        """复用调用方预加载的 RoPE 表，完成驻留路径的 Q 后处理和 BF16 写回。"""
        tiling = self.tiling
        q_half = self._b_dst(gm_q, head_token_task_idx, subblock)
        rows = q_half.shape[0]
        q_accumulator = ub_acc.next_read()
        out_slot = ub_out.next_write()
        self._finish_q(q_accumulator, sin_tab, cos_tab, out_slot, ub_tail, rows, subblock,
                       head_token_task_idx // tiling.owners_b)
        #  mem_copy 以源域为准：非补齐 Q 输出的尾块需把 UB 源显式
        # 绑定到目的实际域，避免越界写满物理行。
        cbd.mem_copy(q_half, ub_out.next_read()[: q_half.shape[0], : q_half.shape[1]])

    @cbd.jit
    def _finish_q(self, ub_acc, ub_sin, ub_cos, ub_out, ub_tail, rows, subblock, head_tile):
        """按驻留布局完成 Q 前缀转换和 head 尾部 RoPE，经 UB 拼接后打包为 BF16。"""
        tiling = self.tiling
        feature_cols, rope_width = tiling.b_cols, tiling.rope_width
        rope_width // 2
        # head 较窄时，最后一个特征向量的宽度可能小于 VL。
        tail = min(VL, feature_cols)
        src_pitch = ub_acc.physical_layout.stride[0]
        out_pitch = ub_out.physical_layout.stride[0]
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
                reg.vstore(ub_tail, 0, reg.vload(ub_acc, q_row_offset + feature_cols - tail), m_tail)
                rope_vec = self._rope_pair(ub_acc, q_row_offset + feature_cols - rope_width, None, 0, ub_sin,
                                           ub_cos, token_row, None, m_pair)
                reg.vstore(ub_tail, tail - rope_width, rope_vec, m_rope)
                reg.vmem_bar("vst_vld")
                reg.vstore_pack(ub_out, bf16_row_offset + feature_cols - tail,
                               self._to_bf16(reg.vload(ub_tail, 0), m_tail), m_tail,
                               pack_mode=reg.PackMode.B32_TO_B16)
