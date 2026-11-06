# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Front quantization, projections, normalization, workspace and weight preparation."""

import cannbotdsl as cb
import torch
import torch_npu
from cannbotdsl import dtypes as dt
from cannbotdsl import simt
from cannbotdsl.ops import reg as rr

from .reindex_matmul import MatmulKernel, MatmulTiling, dsl_matmul
from .reindex_mx import (
    MX_K_ALIGN,
    MX_SCALE_PAIR,
    MemLoc,
    QbmmMxKernel,
    Tensor,
    _fp8_nz_transposed_view,
    ceil_align,
    const_expr,
    copy_options,
    dtypes,
    get_block_idx,
    get_qbmm_mx_tile_config,
    jit,
    mem_copy,
    tile_slice,
)
from .reindex_runtime import (
    FRACTAL_NZ,
    HIDDEN_SIZE,
    INPUT_QUANT_TILE,
    INPUT_SCALE_PAIRS,
    MX_GROUP_SIZE,
    NUM_HEADS,
    Q_LORA_RANK,
    QUERY_DIM,
    QUERY_SCALE_PAIRS,
    SCALE_DTYPE,
    VECTOR_SUBBLOCKS,
    Buffer,
    RingBuffer,
    projection_core_counts,
    quantize_fp8_group,
)

NORM_VECTOR_LANES = 64


class QuantOnlyPhase(MatmulKernel):
    def __init__(self, tiling, cube_cores):
        super().__init__(tiling)
        self.vector_blocks = cube_cores * VECTOR_SUBBLOCKS

    @cb.jit
    def phase(
        self, gm_a: Tensor, gm_b: Tensor, gm_c: Tensor, y: Tensor, scales: Tensor
    ):
        # Stage GM traffic through UB; SIMT only touches local storage.
        inputs = RingBuffer(cb.MemLoc.UB, (INPUT_QUANT_TILE,), dt.bfloat16, depth=1)
        quantized = RingBuffer(cb.MemLoc.UB, (INPUT_QUANT_TILE,), dt.uint8, depth=1)
        scale_channel = RingBuffer(
            cb.MemLoc.UB, (INPUT_QUANT_TILE // MX_GROUP_SIZE,), dt.uint8, depth=1
        )
        for chunk in range(
            cb.get_block_idx() * VECTOR_SUBBLOCKS + cb.get_subblock_id(),
            self.t.m * (HIDDEN_SIZE // INPUT_QUANT_TILE),
            self.vector_blocks,
        ):
            cb.mem_copy(
                inputs.next_write(),
                cb.tile_slice(gm_a.flatten(), (INPUT_QUANT_TILE,), (chunk,)),
            )
            values = inputs.next_read()
            output = quantized.next_write()
            group_scales = scale_channel.next_write()
            with cb.vf(mode="simt", thread=INPUT_QUANT_TILE):
                tid = dt.int32(simt.thread_idx()[0])
                i = tid
                value = dt.float32(values[i])
                code, exponent = quantize_fp8_group(value)
                output[i] = code
                if tid % 32 == 0:
                    group_scales[i // 32] = dt.uint8(exponent)
            cb.mem_copy(
                cb.tile_slice(y, (INPUT_QUANT_TILE,), (chunk,)), quantized.next_read()
            )
            cb.mem_copy(
                cb.tile_slice(scales, (32,), (chunk,)), scale_channel.next_read()
            )


class NormPhase:
    def __init__(self, rows, norm_eps, softmax_scale, cube_cores):
        self.vector_blocks = cube_cores * VECTOR_SUBBLOCKS
        self.rows = rows
        self.norm_eps = norm_eps
        self.softmax_scale = softmax_scale
        self.norm_dim = Q_LORA_RANK
        self.norm_chunks = self.norm_dim // NORM_VECTOR_LANES
        self.norm_tail = self.norm_dim % NORM_VECTOR_LANES
        # Unmasked unpack loads may read the final full register inside UB.
        self.norm_padded = (self.norm_chunks + bool(self.norm_tail)) * NORM_VECTOR_LANES

    @cb.jit
    def phase(
        self,
        x: cb.Tensor,
        g: cb.Tensor,
        y: cb.Tensor,
        out_q: cb.Tensor,
        scales: cb.Tensor,
        weight_raw: cb.Tensor,
        weights: cb.Tensor,
    ):
        ub_x = RingBuffer(cb.MemLoc.UB, (1, self.norm_padded), dt.bfloat16, depth=1)
        ub_g = RingBuffer(cb.MemLoc.UB, (1, self.norm_padded), dt.bfloat16, depth=1)
        ub_y = RingBuffer(cb.MemLoc.UB, (1, self.norm_padded), dt.bfloat16, depth=1)
        ub_s = Buffer(cb.MemLoc.UB, (NORM_VECTOR_LANES,), dt.float32)
        for row in range(
            cb.get_block_idx() * VECTOR_SUBBLOCKS + cb.get_subblock_id(),
            self.rows,
            self.vector_blocks,
        ):
            if cb.const_expr(self.norm_tail != 0):
                cb.mem_copy(
                    ub_x.next_write().view(self.norm_padded)[: self.norm_dim,],
                    cb.tile_slice(x, (self.norm_dim,), (row,)),
                )
                cb.mem_copy(
                    ub_g.next_write().view(self.norm_padded)[: self.norm_dim,], g
                )
            else:
                cb.mem_copy(
                    ub_x.next_write().flatten(),
                    cb.tile_slice(x, (Q_LORA_RANK,), (row,)),
                )
                cb.mem_copy(ub_g.next_write().flatten(), g)
            xx = ub_x.next_read()
            gg = ub_g.next_read()
            yy = ub_y.next_write()
            tmp = ub_s
            self._normalize(xx, gg, yy, tmp)
            normed = ub_y.next_read()
            if cb.const_expr(self.norm_tail != 0):
                cb.mem_copy(
                    cb.tile_slice(y, (self.norm_dim,), (row,)),
                    normed.view(self.norm_padded)[: self.norm_dim,],
                )
            else:
                cb.mem_copy(cb.tile_slice(y, (Q_LORA_RANK,), (row,)), normed.flatten())

            with cb.vf(mode="simt", thread=256):
                tid = dt.int32(simt.thread_idx()[0])
                if tid < 32:
                    weights[row, tid] = simt.cast(
                        weight_raw[row, tid] * dt.float32(self.softmax_scale),
                        dtype=dt.bfloat16,
                    )
                for part in cb.range_constexpr(5):
                    i = tid + part * 256
                    value = dt.float32(normed[0, i])
                    code, exponent = quantize_fp8_group(value)
                    out_q[row * Q_LORA_RANK + i] = code
                    if tid % 32 == 0:
                        scales[row * 40 + i // 32] = dt.uint8(exponent)

    @cb.jit
    def _normalize(self, xx: cb.Tensor, gg: cb.Tensor, yy: cb.Tensor, tmp: cb.Tensor):
        """RMSNorm over norm_dim BF16 values in safely padded UB buffers."""
        with cb.vf(mode="simd"):
            mask = rr.full_mask()
            acc = rr.vdups(0.0, dt.float32, mask=mask)
            for j in range(self.norm_chunks):
                z = rr.vcast(
                    rr.vload_unpack(
                        xx, j * NORM_VECTOR_LANES, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dt.float32,
                    mask=mask,
                )
                acc = rr.vadd(acc, rr.vmul(z, z, mask=mask), mask=mask)
            if cb.const_expr(self.norm_tail != 0):
                tail_mask = rr.update_mask(self.norm_tail, elem_bits=32)[0]
                offset = self.norm_chunks * NORM_VECTOR_LANES
                z = rr.vcast(
                    rr.vload_unpack(xx, offset, unpack_mode=rr.UnpackMode.B16_TO_B32),
                    dt.float32,
                    mask=tail_mask,
                )
                updated = rr.vadd(acc, rr.vmul(z, z, mask=tail_mask), mask=tail_mask)
                # Keep full-chunk partial sums in the inactive tail lanes.
                acc = rr.vselect(updated, acc, cond_mask=tail_mask)
            total = rr.vreduce_sum(acc, mask=mask)
            rr.vstore_first(tmp, 0, total)
            rr.vmem_bar("vst_vld")
            total = rr.vload_broadcast(tmp, 0)
            var = rr.vadds(
                rr.vmuls(total, 1.0 / self.norm_dim, mask=mask),
                self.norm_eps,
                mask=mask,
            )
            one = rr.vdups(1.0, dt.float32, mask=mask)
            rs = rr.vsqrt(rr.vdiv(one, var, mask=mask), mask=mask)
            for j in range(self.norm_chunks):
                z = rr.vcast(
                    rr.vload_unpack(
                        xx, j * NORM_VECTOR_LANES, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dt.float32,
                    mask=mask,
                )
                gamma = rr.vcast(
                    rr.vload_unpack(
                        gg, j * NORM_VECTOR_LANES, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dt.float32,
                    mask=mask,
                )
                z = rr.vmul(rr.vmul(z, rs, mask=mask), gamma, mask=mask)
                z = rr.vcast(z, dt.bfloat16, mask=mask)
                rr.vstore_pack(
                    yy, j * NORM_VECTOR_LANES, z, mask, pack_mode=rr.PackMode.B32_TO_B16
                )
            if cb.const_expr(self.norm_tail != 0):
                tail_mask = rr.update_mask(self.norm_tail, elem_bits=32)[0]
                offset = self.norm_chunks * NORM_VECTOR_LANES
                z = rr.vcast(
                    rr.vload_unpack(xx, offset, unpack_mode=rr.UnpackMode.B16_TO_B32),
                    dt.float32,
                    mask=tail_mask,
                )
                gamma = rr.vcast(
                    rr.vload_unpack(gg, offset, unpack_mode=rr.UnpackMode.B16_TO_B32),
                    dt.float32,
                    mask=tail_mask,
                )
                z = rr.vmul(rr.vmul(z, rs, mask=tail_mask), gamma, mask=tail_mask)
                z = rr.vcast(z, dt.bfloat16, mask=tail_mask)
                rr.vstore_pack(
                    yy, offset, z, tail_mask, pack_mode=rr.PackMode.B32_TO_B16
                )


class WqaPhase(QbmmMxKernel):
    @jit
    def phase(
        self,
        out_gm: Tensor,
        a_gm: Tensor,
        b_gm: Tensor,
        scale_a_gm: Tensor,
        scale_b_gm: Tensor,
        bias_gm: Tensor,
    ):
        """Assign output tiles to AICs with AL1_FULL_LOAD or ASW snake order."""
        # E8M0 typed views preserve the existing byte-storage input ABI.
        scale_a_gm = scale_a_gm.view(dtype=dtypes.float8_e8m0)
        scale_b_gm = scale_b_gm.view(dtype=dtypes.float8_e8m0)
        if const_expr(self.prepacked_b):
            b_gm = _fp8_nz_transposed_view(b_gm, self.n, self.k)
        full_load = self.full_load
        transpose_a, transpose_b = self.transpose_a, self.transpose_b
        base_m, base_n, base_k = self.base_m, self.base_n, self.base_k
        k_l1, l1_buffers, l0c_buffers = self.k_l1, self.l1_buffers, self.l0c_buffers
        a_dtype, b_dtype = self.a_dtype, self.b_dtype

        # L1/L0 ring buffers — created inside kernel context and
        # kept as locals: SSA values must not ride the `self` receiver across
        # the @jit helper boundary (ST101), so they are passed as explicit
        # arguments to _compute_output_tile instead.
        a_k = ceil_align(self.k, MX_K_ALIGN) if full_load else k_l1
        a_depth = 1 if full_load else l1_buffers
        scale_a_l1_len = self.scale_k_len if full_load else self.scale_k_l1_len
        a_scale_depth = 1 if full_load else 2
        l1_a = RingBuffer(
            MemLoc.L1,
            (a_k, base_m) if transpose_a else (base_m, a_k),
            a_dtype,
            depth=a_depth,
            data_format="nz",
        )
        if const_expr(self.prepacked_b):
            l1_b = RingBuffer(
                MemLoc.L1,
                (k_l1, base_n),
                dtypes.int8,
                depth=l1_buffers,
                data_format="nz",
            )
        else:
            l1_b = RingBuffer(
                MemLoc.L1,
                (base_n, k_l1) if transpose_b else (k_l1, base_n),
                dtypes.int8,
                depth=l1_buffers,
                data_format="nz",
            )
        l1_scale_a = RingBuffer(
            MemLoc.L1,
            (base_m, scale_a_l1_len),
            dtypes.float8_e8m0,
            depth=a_scale_depth,
            data_format="zn",
        )
        l1_scale_b = RingBuffer(
            MemLoc.L1,
            (self.scale_k_l1_len, base_n),
            dtypes.float8_e8m0,
            depth=2,
            data_format="nz",
        )
        if const_expr(a_dtype == dtypes.fp4x2_e2m1):
            # Packed FP4: transpose lowering doubles the packed outer extent
            # (two nibbles per byte) and swaps the fractal orientation.
            l0a = RingBuffer(
                MemLoc.L0A,
                (base_m * 2, base_k) if transpose_a else (base_m, base_k),
                a_dtype,
                depth=2,
                data_format="zn" if transpose_a else "nz",
            )
        else:
            l0a = RingBuffer(MemLoc.L0A, (base_m, base_k), a_dtype, depth=2)
        if const_expr(b_dtype == dtypes.fp4x2_e2m1):
            l0b = RingBuffer(
                MemLoc.L0B,
                (base_n, base_k) if transpose_b else (base_n * 2, base_k),
                b_dtype,
                depth=2,
                data_format="nz" if transpose_b else "zn",
            )
        else:
            l0b = RingBuffer(MemLoc.L0B, (base_n, base_k), b_dtype, depth=2)
        l0c = RingBuffer(
            MemLoc.L0C, (base_m, base_n), dtypes.float32, depth=l0c_buffers
        )
        l1_bias = None
        l1_bt = None

        if const_expr(self.has_bias):
            l1_bias = RingBuffer(
                MemLoc.L1, (base_n,), dtypes.float32, depth=2, data_format="nd"
            )
            l1_bt = RingBuffer(
                MemLoc.BIAS, (base_n,), dtypes.float32, depth=2, data_format="nd"
            )

        # Multi-core scheduling: full_load assigns fixed M tiles across cores;
        # normal mode uses ASW snake ordering with 4-row sliding window.
        block_idx = get_block_idx()

        if const_expr(self.full_load):
            # Keep A/ScaleA for one M tile resident and reuse them across N tiles.
            full_m_idx = block_idx % self.block_scheduler.m_tiles
            n_lane = block_idx // self.block_scheduler.m_tiles
            n_stride = self.core_count // self.block_scheduler.m_tiles
            if const_expr(self.transpose_a):
                a_tile = tile_slice(
                    a_gm, (ceil_align(self.k, MX_K_ALIGN), self.base_m), (0, full_m_idx)
                )
            else:
                a_tile = tile_slice(
                    a_gm, (self.base_m, ceil_align(self.k, MX_K_ALIGN)), (full_m_idx, 0)
                )
            mem_copy(
                l1_a.next_write(),
                a_tile,
                options=copy_options(l2_cache_ctl=self.l2_cache_ctl_a),
            )
            if const_expr(self.transpose_a):
                scale_a_tile = tile_slice(
                    scale_a_gm,
                    (self.scale_k_len // MX_SCALE_PAIR, self.base_m, MX_SCALE_PAIR),
                    (0, full_m_idx, 0),
                )
            else:
                scale_a_tile = tile_slice(
                    scale_a_gm,
                    (self.base_m, self.scale_k_len // MX_SCALE_PAIR, MX_SCALE_PAIR),
                    (full_m_idx, 0, 0),
                )
            mem_copy(
                l1_scale_a.next_write(),
                scale_a_tile.permute((1, 0, 2)).flatten(-2, -1)
                if self.transpose_a
                else scale_a_tile.flatten(-2, -1),
                options=copy_options(l2_cache_ctl=1),
            )
            for full_n_idx in range(n_lane, self.block_scheduler.n_tiles, n_stride):
                self._compute_output_tile(
                    out_gm,
                    a_gm,
                    b_gm,
                    scale_a_gm,
                    scale_b_gm,
                    full_m_idx,
                    full_n_idx,
                    bias_gm,
                    l1_a,
                    l1_b,
                    l1_scale_a,
                    l1_scale_b,
                    l0a,
                    l0b,
                    l0c,
                    l1_bias,
                    l1_bt,
                )
        else:
            for task in range(
                block_idx,
                self.block_scheduler.m_tiles * self.block_scheduler.n_tiles,
                self.core_count,
            ):
                m_idx = task // self.block_scheduler.n_tiles
                n_idx = task % self.block_scheduler.n_tiles
                tail_split = False
                sub_m_idx = 0
                sub_n_idx = 0
                self._compute_output_tile(
                    out_gm,
                    a_gm,
                    b_gm,
                    scale_a_gm,
                    scale_b_gm,
                    m_idx,
                    n_idx,
                    bias_gm,
                    l1_a,
                    l1_b,
                    l1_scale_a,
                    l1_scale_b,
                    l0a,
                    l0b,
                    l0c,
                    l1_bias,
                    l1_bt,
                    tail_split,
                    sub_m_idx,
                    sub_n_idx,
                )


class SpareNzWeightPhase(MatmulKernel):
    def __init__(self, tiling, first_core, core_count):
        super().__init__(tiling)
        self.first_core = first_core
        self.core_count = core_count

    @cb.jit
    def phase(
        self, gm_a: Tensor, gm_b: Tensor, gm_c: Tensor, y: Tensor, scales: Tensor
    ):
        t = self.t
        l1_a = RingBuffer(
            MemLoc.L1, shape=(t.base_m, t.k_l1), dtype=t.a_dtype, depth=t.l1_buffer_num
        )
        l1_b = RingBuffer(
            MemLoc.L1,
            shape=(t.base_n, t.k_l1),
            dtype=t.b_dtype,
            depth=t.l1_buffer_num,
            data_format="nz",
        )
        l0a = RingBuffer(
            MemLoc.L0A, shape=(t.base_m, t.base_k), dtype=t.a_dtype, depth=2
        )
        l0b = RingBuffer(
            MemLoc.L0B, shape=(t.base_n, t.base_k), dtype=t.b_dtype, depth=2
        )
        l0c = RingBuffer(
            MemLoc.L0C, shape=(t.base_m, t.base_n), dtype=dtypes.float32, depth=t.l0c_db
        )

        n_tiles = self.n_tiles
        main_window = self.main_window
        main_row = self.main_row
        tail_window = self.tail_window
        total_tiles = self.total_tiles
        k_l1_tiles = self.k_l1_tiles
        k_l0_per_l1 = self.k_l0_per_l1

        block_idx = get_block_idx() - self.first_core
        block_num = self.core_count

        for tile_idx in range(block_idx, total_tiles, block_num):
            m_idx = 0
            n_idx = 0
            row_idx = tile_idx // n_tiles // main_window
            if row_idx < main_row:
                m_idx = row_idx * main_window + tile_idx % main_window
                n_idx = (tile_idx // main_window) % n_tiles
            else:
                row_idx = main_row
                tail_index = tile_idx - main_row * main_window * n_tiles
                m_idx = main_row * main_window + tail_index % tail_window
                n_idx = (tail_index // tail_window) % n_tiles

            n_idx = (n_tiles - 1 - n_idx) if (row_idx % 2 != 0) else n_idx

            l0c_acc = l0c.next_write()
            for k_l1_idx in range(k_l1_tiles):
                gm_a_tile = tile_slice(gm_a, (t.base_m, t.k_l1), (m_idx, k_l1_idx))
                gm_b_tile = tile_slice(gm_b, (t.base_n, t.k_l1), (n_idx, k_l1_idx))
                mem_copy(
                    l1_a.next_write(),
                    gm_a_tile,
                    options=copy_options(l2_cache_ctl=self.l2_cache_ctl_a),
                )
                l1_a_tensor = l1_a.next_read()
                mem_copy(
                    l1_b.next_write(),
                    gm_b_tile,
                    options=copy_options(l2_cache_ctl=self.l2_cache_ctl_b),
                )
                l1_b_tensor = l1_b.next_read()

                for k_l0_idx in range(k_l0_per_l1):
                    l1_a_slice = tile_slice(
                        l1_a_tensor, (t.base_m, t.base_k), (0, k_l0_idx)
                    )
                    l1_b_slice = tile_slice(
                        l1_b_tensor, (t.base_n, t.base_k), (0, k_l0_idx)
                    )
                    mem_copy(l0a.next_write(), l1_a_slice)
                    l0a_tensor = l0a.next_read()
                    mem_copy(l0b.next_write(), l1_b_slice)
                    l0b_tensor = l0b.next_read()
                    global_k = k_l1_idx * k_l0_per_l1 + k_l0_idx
                    dsl_matmul(l0c_acc, l0a_tensor, l0b_tensor, init=(global_k == 0))
            # Fixpipe stores the source extent. Trim the L0C tile to the
            # logical output so a partial M tile cannot overwrite adjacent GM.
            m_start = m_idx * t.base_m
            n_start = n_idx * t.base_n
            current_m = min(t.base_m, t.m - m_start)
            current_n = min(t.base_n, t.n - n_start)
            gm_c_tile = gm_c[
                m_start : m_start + current_m, n_start : n_start + current_n
            ]
            mem_copy(gm_c_tile, l0c_acc[:current_m, :current_n])


class FrontPhases:
    def __init__(self, rows, norm_eps, softmax_scale, cube_cores):
        query_cores, weight_cores = projection_core_counts(cube_cores)
        self.rows = rows
        # Small-M tile geometry remains fixed; the number of M tiles is variable.
        wt = MatmulTiling(
            1,
            NUM_HEADS,
            HIDDEN_SIZE,
            a_dtype=torch.bfloat16,
            b_dtype=torch.bfloat16,
            c_dtype=torch.float32,
        )
        wt.m = rows
        # Raw NZ DMA requires static, full tiles; 1024 divides K=5120.
        wt.k_l1 = 1024
        self.quant = QuantOnlyPhase(wt, cube_cores)
        self.weight = SpareNzWeightPhase(wt, query_cores, weight_cores)
        qt = get_qbmm_mx_tile_config(
            1, Q_LORA_RANK, HIDDEN_SIZE, transpose_a=False, transpose_b=False
        )
        qt = qt._replace(
            full_load=False, m_base_tail_split_count=1, n_base_tail_split_count=1
        )
        self.query = WqaPhase(rows, Q_LORA_RANK, HIDDEN_SIZE, qt, transpose_b=False)
        self.query.core_count = query_cores
        self.query.l2_cache_ctl_b = 0
        self.norm = NormPhase(rows, norm_eps, softmax_scale, cube_cores)


class FrontWorkspace:
    """Caller-owned storage; reuse on one stream, separately per concurrent call."""

    def __init__(self, rows, device):
        if not isinstance(rows, int) or rows < 1:
            raise ValueError("T must be positive")
        self.rows, self.device = rows, torch.device(device)

        def alloc(shape, dtype=torch.bfloat16):
            return torch.empty(shape, dtype=dtype, device=device)

        self.xq = alloc((rows, HIDDEN_SIZE), torch.uint8)
        self.xs = alloc((rows, INPUT_SCALE_PAIRS, 2), torch.uint8)
        self.qa = alloc((rows, Q_LORA_RANK))
        self.qr = alloc((rows, Q_LORA_RANK))
        self.rq = alloc((rows, Q_LORA_RANK), torch.uint8)
        self.rs = alloc((rows, QUERY_SCALE_PAIRS, 2), torch.uint8)
        self.w = alloc((rows, NUM_HEADS))
        self.w_raw = alloc((rows, NUM_HEADS), torch.float32)
        self.bias = alloc((0,), torch.float32)


def prepare_weight_nz(ww):
    """Convert unscaled NPU BF16 WW[32,5120] to FRACTAL_NZ (format 29).

    Logical shape and values are preserved. Run once at weight load time;
    already-prepared NZ weights can be passed directly to indexer_front.
    """
    if tuple(ww.shape) != (NUM_HEADS, HIDDEN_SIZE) or ww.dtype != torch.bfloat16:
        raise ValueError("ww must be BF16 [32,5120]")
    if ww.device.type != "npu":
        raise ValueError("ww must be on NPU")
    return torch_npu.npu_format_cast(ww.contiguous(), FRACTAL_NZ)


def prepare_wqa_nz(weight_nk, scale_ngp):
    """ND WQA[N,K]/scale[N,K/64,2] -> NZ WQA[K,N]/ND scale[K/64,N,2]."""
    if tuple(weight_nk.shape) != (Q_LORA_RANK, HIDDEN_SIZE) or tuple(
        scale_ngp.shape
    ) != (
        Q_LORA_RANK,
        INPUT_SCALE_PAIRS,
        2,
    ):
        raise ValueError("expected WQA [1280,5120] and scale [1280,80,2]")
    # Keep FP8 dtype during format conversion; byte-typed NZ roundtrips in
    # this torch_npu build do not reliably preserve arbitrary bit patterns.
    weight_kn = weight_nk.view(torch.float8_e4m3fn).transpose(0, 1).contiguous()
    return (
        torch_npu.npu_format_cast(weight_kn, FRACTAL_NZ),
        scale_ngp.view(torch.uint8).transpose(0, 1).contiguous().view(SCALE_DTYPE),
    )


def prepare_wqb_nz(weight_nk, scale_ngp):
    """ND WQB[N,K] -> NZ WQB[N,K]; ND scale[N,K/64,2] stays unchanged."""
    if tuple(weight_nk.shape) != (QUERY_DIM, Q_LORA_RANK) or tuple(scale_ngp.shape) != (
        QUERY_DIM,
        QUERY_SCALE_PAIRS,
        2,
    ):
        raise ValueError("expected WQB [4096,1280] and scale [4096,20,2]")
    return (
        torch_npu.npu_format_cast(
            weight_nk.view(torch.float8_e4m3fn).contiguous(), FRACTAL_NZ
        ),
        scale_ngp.view(torch.uint8).contiguous().view(SCALE_DTYPE),
    )
