# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""T-tiled WQ_B + RoPE + FP4 QDQ; L0C -> UB, no Q/RoPE GM intermediates."""

import cannbotdsl as cb
from cannbotdsl import simt
from cannbotdsl._mlir.dialects import cannir

from .reindex_runtime import Q_LORA_RANK, QUERY_DIM, ceil_mx_scale, encode_fp4_e2m1

try:
    from cannbotdsl.core.ir_transport import unwrap_operand
except ImportError:
    from cannbotdsl.ir_transport import unwrap_operand
from cannbotdsl.tensor import make_layout, make_tensor

from .reindex_mx import (
    MX_K_ALIGN,
    MX_SCALE_PAIR,
    MemLoc,
    QbmmMxKernel,
    QbmmMxTiling,
    RingBuffer,
    Tensor,
    ceil_align,
    ceil_div,
    const_expr,
    copy_options,
    dtypes,
    get_block_idx,
    jit,
    matmul,
    mem_copy,
    tile_slice,
)


def _fp8_nz_transposed_view(gm_weight, output_size, input_size):
    """Describe packed [N/32,K,32] bytes as logical W.T[K,N]."""
    typed = gm_weight.view(dtype=dtypes.int8)
    pointer = cannir.extract_pointer(unwrap_operand(typed))
    return make_tensor(
        pointer,
        make_layout((input_size, output_size)),
        physical_layout=make_layout((output_size // 32, input_size // 16, 16, 32)),
        data_format="nz",
    )


QUERY_TILE_M = 16
QUERY_TILE_N = 128
QUERY_TILE_K = 256


def get_query_tile_config(cube_cores):
    """Preserve the validated Query epilogue geometry independently of quota.

    Each M tile has 32 N tiles. TailQueryPhase distributes these with a stride
    of cube_cores, so a 28-core launch processes the remaining four in round 2.
    Full-K residency and disabled tail splitting match the validated path.
    """
    if not isinstance(cube_cores, int) or cube_cores < 1:
        raise ValueError("cube_cores must be a positive integer")
    return QbmmMxTiling(
        base_m=QUERY_TILE_M,
        base_n=QUERY_TILE_N,
        base_k=QUERY_TILE_K,
        k_l1=Q_LORA_RANK,
        scale_k_l1=Q_LORA_RANK,
        l1_buffers=1,
        l0c_buffers=1,
        used_core_num=min(cube_cores, QUERY_DIM // QUERY_TILE_N),
        full_load=True,
        m_tail_tile=1,
        n_tail_tile=1,
        m_base_tail_split_count=1,
        n_base_tail_split_count=1,
        m_tail_main=0,
        n_tail_main=0,
    )


class TailQueryPhase(QbmmMxKernel):
    def __init__(self, m, n, k, tiling, cube_cores):
        super().__init__(m, n, k, tiling)
        self.core_count = cube_cores
        if (n, k) != (QUERY_DIM, Q_LORA_RANK) or (
            self.base_m,
            self.base_n,
            self.base_k,
        ) != (
            QUERY_TILE_M,
            QUERY_TILE_N,
            QUERY_TILE_K,
        ):
            raise ValueError(
                "WqbRopeQdq requires N4096 K1280 baseM16 baseN128 baseK256"
            )
        self.l2_cache_ctl_b = 0
        self.prepacked_b = False

    @jit
    def _compute_output_tile(
        self,
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
        cv,
        outub,
        freq,
        scales,
        tail_split=False,
        sub_m_idx=0,
        sub_n_idx=0,
    ):
        """Compute one output tile with the shared L0C and FixPipe path."""
        # E8M0 typed views preserve the existing byte-storage input ABI.
        scale_a_gm = scale_a_gm.view(dtype=dtypes.float8_e8m0)
        scale_b_gm = scale_b_gm.view(dtype=dtypes.float8_e8m0)
        m_start, n_start, current_m, current_n = self.block_scheduler.get_block(
            m_idx,
            n_idx,
            tail_split,
            sub_m_idx,
            sub_n_idx,
        )
        m_end = m_start + current_m
        n_end = n_start + current_n
        # One L0C slot per output tile: the whole K-accumulation and the
        # FixPipe read below share it; the write cursor advances once per
        # tile so depth>1 rotates slots across tiles (FixPipe overlap).
        l0c_slot = l0c.next_write()

        bias = None
        if const_expr(self.has_bias):
            bias_tile = bias_gm[n_start:n_end,]
            mem_copy(l1_bias.next_write(), bias_tile)
            mem_copy(l1_bt.next_write(), l1_bias.next_read())
            bias = l1_bt.next_read()

        # Each scaleKL1 window covers scale_k_l1_factor data kL1 windows.
        for scale_l1_idx in range(self.scale_l1_tiles):
            scale_group_count = self.scale_k_l1_len // MX_SCALE_PAIR
            scale_group_start = scale_l1_idx * scale_group_count
            scale_group_end = min(
                scale_group_start + scale_group_count,
                ceil_div(self.k, MX_K_ALIGN),
            )
            # Copy ScaleA from GM to L1 unless it is resident in full-load mode.
            if const_expr(not self.full_load):
                if const_expr(self.transpose_a):
                    scale_a_gm_tile = scale_a_gm[
                        scale_group_start:scale_group_end,
                        m_start:m_end,
                        0:MX_SCALE_PAIR,
                    ]
                    mem_copy(
                        l1_scale_a.next_write(),
                        scale_a_gm_tile.permute((1, 0, 2)).flatten(-2, -1)
                        if self.transpose_a
                        else scale_a_gm_tile.flatten(-2, -1),
                        options=copy_options(l2_cache_ctl=1),
                    )
                else:
                    scale_a_gm_tile = scale_a_gm[
                        m_start:m_end,
                        scale_group_start:scale_group_end,
                        0:MX_SCALE_PAIR,
                    ]
                    mem_copy(
                        l1_scale_a.next_write(),
                        scale_a_gm_tile.permute((1, 0, 2)).flatten(-2, -1)
                        if self.transpose_a
                        else scale_a_gm_tile.flatten(-2, -1),
                        options=copy_options(l2_cache_ctl=1),
                    )
            # Scale window is written once above and read by every kL0 step
            # below: select the read slot once per scale window so the
            # read cursor tracks the write cursor (depth rotation).
            scale_a_l1_rd = l1_scale_a.next_read()
            # ScaleB GM→L1
            if const_expr(self.transpose_b):
                scale_b_gm_tile = scale_b_gm[
                    n_start:n_end,
                    scale_group_start:scale_group_end,
                    0:MX_SCALE_PAIR,
                ]
                mem_copy(
                    l1_scale_b.next_write(),
                    scale_b_gm_tile.flatten(-2, -1).permute(1, 0)
                    if self.transpose_b
                    else scale_b_gm_tile.permute((0, 2, 1)).flatten(0, 1),
                    options=copy_options(l2_cache_ctl=1),
                )
            else:
                scale_b_gm_tile = scale_b_gm[
                    scale_group_start:scale_group_end,
                    n_start:n_end,
                    0:MX_SCALE_PAIR,
                ]
                mem_copy(
                    l1_scale_b.next_write(),
                    scale_b_gm_tile.flatten(-2, -1).permute(1, 0)
                    if self.transpose_b
                    else scale_b_gm_tile.permute((0, 2, 1)).flatten(0, 1),
                    options=copy_options(l2_cache_ctl=1),
                )
            scale_b_l1_rd = l1_scale_b.next_read()

            # Process only valid data windows in the final scaleKL1 window.
            current_scale_k_l1_factor = min(
                self.scale_k_l1_factor,
                self.k_l1_tiles - scale_l1_idx * self.scale_k_l1_factor,
            )
            for k_l1_idx_in_scale in range(current_scale_k_l1_factor):
                global_k_l1_idx = (
                    scale_l1_idx * self.scale_k_l1_factor + k_l1_idx_in_scale
                )
                k_start = global_k_l1_idx * self.k_l1
                k_end = min(k_start + self.k_l1, self.k)
                # Copy A/B from GM to L1; full-load mode copies only B here.
                if const_expr(not self.full_load):
                    if const_expr(self.transpose_a):
                        mem_copy(
                            l1_a.next_write(),
                            a_gm[k_start:k_end, m_start:m_end],
                            options=copy_options(l2_cache_ctl=self.l2_cache_ctl_a),
                        )
                    else:
                        mem_copy(
                            l1_a.next_write(),
                            a_gm[m_start:m_end, k_start:k_end],
                            options=copy_options(l2_cache_ctl=self.l2_cache_ctl_a),
                        )
                if const_expr(self.prepacked_b):
                    mem_copy(
                        l1_b.next_write(),
                        tile_slice(
                            b_gm, (self.k_l1, self.base_n), (global_k_l1_idx, n_idx)
                        ),
                        options=copy_options(l2_cache_ctl=self.l2_cache_ctl_b),
                    )
                elif const_expr(self.transpose_b):
                    mem_copy(
                        l1_b.next_write(),
                        tile_slice(
                            b_gm, (self.base_n, self.k_l1), (n_idx, global_k_l1_idx)
                        ),
                        options=copy_options(l2_cache_ctl=self.l2_cache_ctl_b),
                    )
                else:
                    mem_copy(
                        l1_b.next_write(),
                        b_gm[k_start:k_end, n_start:n_end],
                        options=copy_options(l2_cache_ctl=self.l2_cache_ctl_b),
                    )
                # Same kL1 window for the L1→L0A/L0B reads below: select
                # once per window (full_load keeps A resident in slot 0).
                a_l1_rd = l1_a.next_read()
                b_l1_rd = l1_b.next_read().reinterpret(dtype=self.b_dtype)

                # The final kL1 window contains only its valid baseK blocks.
                current_k_l1 = k_end - k_start
                current_step_k = ceil_div(current_k_l1, self.base_k)
                # Move each baseK block to L0A/L0B and accumulate with MX MMAD.
                for k_l0_idx in range(current_step_k):
                    global_k_l0_idx = global_k_l1_idx * self.step_k + k_l0_idx
                    scale_l0_idx_in_l1 = k_l1_idx_in_scale * self.step_k + k_l0_idx
                    # Full-load A uses a global K index; streamed A uses a local one.
                    if const_expr(self.full_load):
                        a_k_l0_idx = global_k_l0_idx
                        scale_a_k_l0_idx = global_k_l0_idx
                    else:
                        a_k_l0_idx = k_l0_idx
                        scale_a_k_l0_idx = scale_l0_idx_in_l1
                    # L1→L0A + MX ScaleA
                    if const_expr(self.transpose_a):
                        a_l1_tile = tile_slice(
                            a_l1_rd, (self.base_k, self.base_m), (a_k_l0_idx, 0)
                        )
                    else:
                        a_l1_tile = tile_slice(
                            a_l1_rd, (self.base_m, self.base_k), (0, a_k_l0_idx)
                        )
                    scale_a_l1_tile = tile_slice(
                        scale_a_l1_rd,
                        (self.base_m, self.scale_k_l0_len),
                        (0, scale_a_k_l0_idx),
                    )
                    mem_copy(
                        l0a.next_write(),
                        a_l1_tile.permute(1, 0) if self.transpose_a else a_l1_tile,
                        options=copy_options(mx_scale=scale_a_l1_tile),
                    )
                    # L1→L0B + MX ScaleB
                    if const_expr(self.prepacked_b):
                        b_l1_tile = tile_slice(
                            b_l1_rd.reinterpret(dtype=dtypes.float8_e4m3fn),
                            (self.base_k, self.base_n),
                            (k_l0_idx, 0),
                        )
                    elif const_expr(self.transpose_b):
                        b_l1_tile = tile_slice(
                            b_l1_rd, (self.base_n, self.base_k), (0, k_l0_idx)
                        )
                    else:
                        b_l1_tile = tile_slice(
                            b_l1_rd, (self.base_k, self.base_n), (k_l0_idx, 0)
                        )
                    scale_b_l1_tile = tile_slice(
                        scale_b_l1_rd,
                        (self.scale_k_l0_len, self.base_n),
                        (scale_l0_idx_in_l1, 0),
                    )
                    mem_copy(
                        l0b.next_write(),
                        b_l1_tile.permute(1, 0)
                        if self.prepacked_b or not self.transpose_b
                        else b_l1_tile,
                        options=copy_options(mx_scale=scale_b_l1_tile),
                    )
                    # MX MMAD
                    is_final_acc = global_k_l0_idx + 1 == self.k_l0_tiles
                    matmul(
                        l0c_slot,
                        l0a.next_read(),
                        l0b.next_read(),
                        init=(global_k_l0_idx == 0),
                        bias=bias,
                        unit_flag=3 if is_final_acc else 2,
                        mx_mode=True,
                    )

        # Cube accumulator goes directly to both Vector subcores, split in N.
        # Each active SIMT lane packs an adjacent FP4 pair.  This keeps the
        # query in the same packed MXFP4 + E8M0 representation consumed by the
        # repository QSLI Cube path, without a BF16-QDQ GM round trip.
        mem_copy(
            cv.next_write(),
            l0c_slot.partition(dim=1),
            options=copy_options(unit_flag=3),
        )
        values = cv.next_read()
        sub = cb.get_subblock_id()
        dest = outub.next_write()
        with cb.vf(mode="simt", thread=1024):
            tid = dtypes.int32(simt.thread_idx()[0])
            row = tid // (self.base_n // 2)
            col = tid % (self.base_n // 2)
            if row < current_m and col % 2 == 0:
                absolute = (
                    dtypes.int32(n_start) + dtypes.int32(sub) * (self.base_n // 2) + col
                )
                dim = absolute % 128
                val0 = dtypes.float32(
                    simt.cast(values[row, col], dtype=dtypes.bfloat16)
                )
                val1 = dtypes.float32(
                    simt.cast(values[row, col + 1], dtype=dtypes.bfloat16)
                )
                if dim >= 64:
                    pair = (dim - 64) // 2
                    cosine = freq[row * 64 + pair * 2]
                    sine = freq[row * 64 + pair * 2 + 1]
                    raw0 = val0
                    raw1 = val1
                    val0 = raw0 * cosine - raw1 * sine
                    val1 = raw0 * sine + raw1 * cosine
                value0 = dtypes.float32(simt.cast(val0, dtype=dtypes.bfloat16))
                value1 = dtypes.float32(simt.cast(val1, dtype=dtypes.bfloat16))
                pair_max = simt.fmax(simt.abs(value0), simt.abs(value1))
                amax = simt.reduce_max(pair_max)
                scaled = simt.fmax(
                    amax, dtypes.float32(6.0 * 2.0**-126)
                ) * dtypes.float32(1.0 / 6.0)
                exponent, scale = ceil_mx_scale(scaled)
                z0 = value0 / scale
                code0 = encode_fp4_e2m1(z0)
                z1 = value1 / scale
                code1 = encode_fp4_e2m1(z1)
                dest[row, col // 2] = code0 + code1 * dtypes.uint8(16)
                if absolute % 32 == 0:
                    scales[m_start + row, absolute // 32] = dtypes.uint8(exponent)
        mem_copy(
            out_gm[
                m_start:m_end, n_start // 2 + sub * 32 : n_start // 2 + sub * 32 + 32
            ],
            outub.next_read()[:current_m, :32],
        )

    # Device kernel entry. It maps output tiles to AIC cores, stages A/B/Scale
    # from GM to L1 and then L0A/L0B, performs K-sliced MX MMAD accumulation
    # in L0C, and writes the completed tile back to GM via FixPipe.
    @jit
    def phase(
        self,
        out_gm: Tensor,
        a_gm: Tensor,
        b_gm: Tensor,
        scale_a_gm: Tensor,
        scale_b_gm: Tensor,
        bias_gm: Tensor,
        freq: Tensor,
        scales: Tensor,
    ):
        """Assign output tiles to AICs with AL1_FULL_LOAD or ASW snake order."""
        if const_expr(self.prepacked_b):
            b_gm = _fp8_nz_transposed_view(b_gm, 4096, self.k)
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

        outub = RingBuffer(MemLoc.UB, (16, 32), dtypes.uint8, depth=1)
        freq_channel = RingBuffer(MemLoc.UB, (1024,), dtypes.float32, depth=1)
        cv = RingBuffer(MemLoc.UB, (base_m, base_n // 2), dtypes.float32, depth=1)

        # Multi-core scheduling: full_load assigns fixed M tiles across cores;
        # normal mode uses ASW snake ordering with 4-row sliding window.
        block_idx = get_block_idx()

        for full_m_idx in range(self.block_scheduler.m_tiles):
            n_lane = block_idx
            n_stride = self.core_count
            mem_copy(
                freq_channel.next_write(),
                freq[
                    full_m_idx * 1024 : min(
                        (full_m_idx + 1) * 1024, out_gm.shape[0] * 64
                    ),
                ],
            )
            freq_ub = freq_channel.next_read()
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
                    cv,
                    outub,
                    freq_ub,
                    scales,
                )
