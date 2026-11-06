# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""BF16 Cube QK and PV matrix products for sparse attention.

QK computes scores as Q @ K.T; PV computes the output as P @ V.
Both share pipeline allocation while retaining their own operand layouts,
flat/token-owner schedules and final MMAD/FIXPIPE dependencies.
"""

from __future__ import annotations

from cannbotdsl import make_layout, MemLoc, dtypes, Tensor, tile_slice as tile_view
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.buffer import Buffer


def _allocate_pipeline_buffer(memloc, shape, dtype, *, depth):
    return RingBuffer(memloc, dtype, make_layout(shape), depth=depth)


class QueryKeyMatmulKernel:
    """S[m,n] = Q[m,k] @ K[n,k]^T with BF16 operands and FP32 scores."""

    def __init__(self, m: int, n: int, k: int, base_n: int = 64, base_k: int = 128):
        if m <= 0 or (m != 8 and m % 16) or n % base_n or k % base_k:
            raise ValueError("M=8 or positive multiple of 16, N of base_n, K of base_k")
        self.m, self.n, self.k = m, n, k
        self.base_n, self.base_k = base_n, base_k

    @jit
    def compute_flat_tiles(self, s_gm, q_gm, k_gm, block, nblk, tokens):
        """Flat dispatch over (token, N-tile) so all blocks share the work.

        ``s_gm``/``q_gm``/``k_gm`` are ``[T, m, n]`` / ``[T, m, k]`` /
        ``[T, n, k]``; ``tokens=1`` with 2-D inputs is also supported.
        Flatten T and the row dimension before a 2-D tile_slice: it addresses
        trailing dimensions and clips tiles to that domain in DSL 0.8.
        """
        m, n, k = self.m, self.n, self.k
        base_n, base_k = self.base_n, self.base_k
        # ping-pong depths follow samples/matmul/matmul.py: L1 A/B depth>1 and
        # L0A/L0B depth=2 so MTE2 (GM->L1), MTE1 (L1->L0) and MMAD overlap
        # instead of serializing every K slice.
        q_l1 = _allocate_pipeline_buffer(MemLoc.L1, (m, k), dtypes.bfloat16, depth=2)
        k_l1 = _allocate_pipeline_buffer(
            MemLoc.L1, (base_n, base_k), dtypes.bfloat16, depth=2
        )
        l0a = _allocate_pipeline_buffer(
            MemLoc.L0A, (m, base_k), dtypes.bfloat16, depth=2
        )
        l0b = _allocate_pipeline_buffer(
            MemLoc.L0B, (base_n, base_k), dtypes.bfloat16, depth=2
        )
        l0c = _allocate_pipeline_buffer(
            MemLoc.L0C, (m, base_n), dtypes.float32, depth=1
        )
        k_tiles = k // base_k
        n_tiles = n // base_n
        for it in range(block, tokens * n_tiles, nblk):
            tok = it // n_tiles
            nt = it % n_tiles
            q_t = tile_view(q_gm.view(tokens * m, k), (m, k), (tok, 0))
            k_t = tile_view(k_gm.view(tokens * n, k), (n, k), (tok, 0))
            s_t = tile_view(s_gm.view(tokens * m, n), (m, n), (tok, 0))
            mem_copy(q_l1.next_write(), q_t[0:m, 0:k])
            q_l1_slot = q_l1.next_read()
            l0c_slot = l0c.next_write()
            for kt in range(k_tiles):
                mem_copy(l0a.next_write(), tile_view(q_l1_slot, (m, base_k), (0, kt)))
                mem_copy(
                    k_l1.next_write(),
                    k_t[
                        nt * base_n : (nt + 1) * base_n, kt * base_k : (kt + 1) * base_k
                    ],
                )
                mem_copy(l0b.next_write(), k_l1.next_read())
                matmul(
                    l0c_slot,
                    l0a.next_read(),
                    l0b.next_read(),
                    init=(kt == 0),
                    unit_flag=3 if kt + 1 == k_tiles else 2,
                )
            mem_copy(
                s_t[0:m, nt * base_n : (nt + 1) * base_n],
                l0c.next_read(),
                options=copy_options(unit_flag=3),
            )

    @jit
    def compute_owned_tiles(self, s_gm, q_gm, k_gm, block, tokens, bpt, nt_per):
        """Token-local owner schedule with compile-time ``bpt`` / ``nt_per``.

        Block ``b`` owns token ``b // bpt`` and the N-tile run
        ``[sub*nt_per, min((sub+1)*nt_per, n_tiles))`` with ``sub = b % bpt``.
        Q is loaded to L1 once per owned token and reused across its N-tiles
        (the flat interleave reloaded Q once per (token, N-tile)).

        ``bpt``/``nt_per`` MUST be Python ints so the tile loops stay
        compile-time; only the ``sub == sid`` compare is dynamic.
        """
        m, n, k = self.m, self.n, self.k
        base_n, base_k = self.base_n, self.base_k
        q_l1 = _allocate_pipeline_buffer(MemLoc.L1, (m, k), dtypes.bfloat16, depth=2)
        k_l1 = _allocate_pipeline_buffer(
            MemLoc.L1, (base_n, base_k), dtypes.bfloat16, depth=2
        )
        l0a = _allocate_pipeline_buffer(
            MemLoc.L0A, (m, base_k), dtypes.bfloat16, depth=2
        )
        l0b = _allocate_pipeline_buffer(
            MemLoc.L0B, (base_n, base_k), dtypes.bfloat16, depth=2
        )
        l0c = _allocate_pipeline_buffer(
            MemLoc.L0C, (m, base_n), dtypes.float32, depth=1
        )
        k_tiles = k // base_k
        n_tiles = n // base_n
        tok = block // bpt
        sub = block % bpt
        if tok < tokens:
            q_t = tile_view(q_gm.view(tokens * m, k), (m, k), (tok, 0))
            k_t = tile_view(k_gm.view(tokens * n, k), (n, k), (tok, 0))
            s_t = tile_view(s_gm.view(tokens * m, n), (m, n), (tok, 0))
            mem_copy(q_l1.next_write(), q_t[0:m, 0:k])
            q_l1_slot = q_l1.next_read()
            for sid in range(bpt):
                if sub == sid:
                    for nt in range(sid * nt_per, sid * nt_per + nt_per):
                        if nt < n_tiles:
                            l0c_slot = l0c.next_write()
                            for kt in range(k_tiles):
                                mem_copy(
                                    l0a.next_write(),
                                    tile_view(q_l1_slot, (m, base_k), (0, kt)),
                                )
                                mem_copy(
                                    k_l1.next_write(),
                                    k_t[
                                        nt * base_n : (nt + 1) * base_n,
                                        kt * base_k : (kt + 1) * base_k,
                                    ],
                                )
                                mem_copy(l0b.next_write(), k_l1.next_read())
                                matmul(
                                    l0c_slot,
                                    l0a.next_read(),
                                    l0b.next_read(),
                                    init=(kt == 0),
                                    unit_flag=3 if kt + 1 == k_tiles else 2,
                                )
                            mem_copy(
                                s_t[0:m, nt * base_n : (nt + 1) * base_n],
                                l0c.next_read(),
                                options=copy_options(unit_flag=3),
                            )

    @kernel
    def query_key_matmul_kernel(self, s_gm: Tensor, q_gm: Tensor, k_gm: Tensor):
        self.compute_flat_tiles(s_gm, q_gm, k_gm, get_block_idx(), get_block_num(), 1)


class QueryKeyMatmulOp:
    def __init__(self, m, n, k, base_n=64, base_k=128):
        self.kernel = QueryKeyMatmulKernel(m, n, k, base_n, base_k)

    @host
    def run(self, s_gm, q_gm, k_gm):
        self.kernel.query_key_matmul_kernel[self.kernel.n // self.kernel.base_n](
            s_gm, q_gm, k_gm
        )


class ProbabilityValueMatmulKernel:
    def __init__(self, m: int, n: int, k: int, base_n: int = 128, base_k: int = 128):
        if m <= 0 or (m != 8 and m % 16) or n % base_n or k % base_k:
            raise ValueError("M=8 or positive multiple of 16, N of base_n, K of base_k")
        self.m, self.n, self.k = m, n, k
        self.base_n, self.base_k = base_n, base_k

    @jit
    def _compute_output_feature_tile(self, out_gm, p_gm, v_gm, nt):
        m, _n, k = self.m, self.n, self.k
        base_n, base_k = self.base_n, self.base_k
        p_l1 = _allocate_pipeline_buffer(
            MemLoc.L1, (m, base_k), dtypes.bfloat16, depth=2
        )
        v_l1 = _allocate_pipeline_buffer(
            MemLoc.L1, (base_k, base_n), dtypes.bfloat16, depth=2
        )
        l0a = _allocate_pipeline_buffer(
            MemLoc.L0A, (m, base_k), dtypes.bfloat16, depth=2
        )
        l0b_shape = (base_n, base_k)
        l0b = _allocate_pipeline_buffer(MemLoc.L0B, l0b_shape, dtypes.bfloat16, depth=2)
        l0c = _allocate_pipeline_buffer(
            MemLoc.L0C, (m, base_n), dtypes.float32, depth=1
        )
        k_tiles = k // base_k
        l0c_slot = l0c.next_write()
        for kt in range(k_tiles):
            mem_copy(p_l1.next_write(), p_gm[0:m, kt * base_k : (kt + 1) * base_k])
            mem_copy(l0a.next_write(), p_l1.next_read())
            mem_copy(
                v_l1.next_write(),
                v_gm[kt * base_k : (kt + 1) * base_k, nt * base_n : (nt + 1) * base_n],
            )
            mem_copy(l0b.next_write(), (v_l1.next_read()).permute((1, 0)))
            matmul(
                l0c_slot,
                l0a.next_read(),
                l0b.next_read(),
                init=(kt == 0),
                unit_flag=3 if kt + 1 == k_tiles else 2,
            )
        mem_copy(
            out_gm[0:m, nt * base_n : (nt + 1) * base_n],
            l0c.next_read(),
            options=copy_options(unit_flag=3),
        )

    @jit
    def compute_flat_tiles(self, out_gm, p_gm, v_gm, block, nblk, tokens):
        """Flat dispatch over (token, N-tile); channels allocated once.

        ``out_gm``/``p_gm``/``v_gm`` are ``[T, m, n]`` / ``[T, m, k]`` /
        ``[T, k, n]`` (2-D with ``tokens=1`` also works).
        Flatten T and the row dimension before taking tail-aware 2-D tiles.
        """
        m, n, k = self.m, self.n, self.k
        base_n, base_k = self.base_n, self.base_k
        p_l1 = _allocate_pipeline_buffer(
            MemLoc.L1, (m, base_k), dtypes.bfloat16, depth=2
        )
        v_l1 = _allocate_pipeline_buffer(
            MemLoc.L1, (base_k, base_n), dtypes.bfloat16, depth=2
        )
        l0a = _allocate_pipeline_buffer(
            MemLoc.L0A, (m, base_k), dtypes.bfloat16, depth=2
        )
        l0b_shape = (base_n, base_k)
        l0b = _allocate_pipeline_buffer(MemLoc.L0B, l0b_shape, dtypes.bfloat16, depth=2)
        # This is one mutable reduction accumulator, reused after FIXPIPE.
        # Buffer preserves the dependency across flat output iterations;
        # a depth-1 RingBuffer stalled on the T64/H16 multi-token schedule.
        l0c = Buffer(MemLoc.L0C, dtypes.float32, make_layout((m, base_n)))
        k_tiles = k // base_k
        n_tiles = n // base_n
        for it in range(block, tokens * n_tiles, nblk):
            tok = it // n_tiles
            nt = it % n_tiles
            out_t = tile_view(out_gm.view(tokens * m, n), (m, n), (tok, 0))
            p_t = tile_view(p_gm.view(tokens * m, k), (m, k), (tok, 0))
            v_t = tile_view(v_gm.view(tokens * k, n), (k, n), (tok, 0))
            l0c_slot = l0c
            for kt in range(k_tiles):
                mem_copy(p_l1.next_write(), p_t[0:m, kt * base_k : (kt + 1) * base_k])
                mem_copy(l0a.next_write(), p_l1.next_read())
                mem_copy(
                    v_l1.next_write(),
                    v_t[
                        kt * base_k : (kt + 1) * base_k, nt * base_n : (nt + 1) * base_n
                    ],
                )
                mem_copy(l0b.next_write(), (v_l1.next_read()).permute((1, 0)))
                matmul(
                    l0c_slot,
                    l0a.next_read(),
                    l0b.next_read(),
                    init=(kt == 0),
                    unit_flag=3 if kt + 1 == k_tiles else 2,
                )
            mem_copy(
                out_t[0:m, nt * base_n : (nt + 1) * base_n],
                l0c,
                options=copy_options(unit_flag=3),
            )

    @jit
    def compute_owned_tiles(self, out_gm, p_gm, v_gm, block, tokens, bpt, nt_per):
        """Token-local owner schedule reusing P across the owned N-tile run.

        P[m, k] does not depend on the output feature tile, so loading it once
        per owned token (instead of once per (token, N-tile)) removes the
        repeated P GM->L1 traffic of :meth:`compute_flat_tiles`.  Mirrors
        ``QueryKeyMatmulKernel.compute_owned_tiles``: ``tok = block // bpt`` and the N-tile run
        ``[sub*nt_per, min((sub+1)*nt_per, n_tiles))`` with ``sub = block % bpt``.

        ``bpt``/``nt_per`` MUST be Python ints so the tile loops stay
        compile-time.
        """
        m, n, k = self.m, self.n, self.k
        base_n, base_k = self.base_n, self.base_k
        p_l1 = _allocate_pipeline_buffer(MemLoc.L1, (m, k), dtypes.bfloat16, depth=1)
        v_l1 = _allocate_pipeline_buffer(
            MemLoc.L1, (base_k, base_n), dtypes.bfloat16, depth=2
        )
        l0a = _allocate_pipeline_buffer(
            MemLoc.L0A, (m, base_k), dtypes.bfloat16, depth=2
        )
        l0b_shape = (base_n, base_k)
        l0b = _allocate_pipeline_buffer(MemLoc.L0B, l0b_shape, dtypes.bfloat16, depth=2)
        l0c = _allocate_pipeline_buffer(
            MemLoc.L0C, (m, base_n), dtypes.float32, depth=1
        )
        k_tiles = k // base_k
        n_tiles = n // base_n
        tok = block // bpt
        sub = block % bpt
        if tok < tokens:
            p_t = tile_view(p_gm.view(tokens * m, k), (m, k), (tok, 0))
            v_t = tile_view(v_gm.view(tokens * k, n), (k, n), (tok, 0))
            out_t = tile_view(out_gm.view(tokens * m, n), (m, n), (tok, 0))
            mem_copy(p_l1.next_write(), p_t[0:m, 0:k])
            p_l1_slot = p_l1.next_read()
            for sid in range(bpt):
                if sub == sid:
                    for nt in range(sid * nt_per, sid * nt_per + nt_per):
                        if nt < n_tiles:
                            l0c_slot = l0c.next_write()
                            for kt in range(k_tiles):
                                mem_copy(
                                    l0a.next_write(),
                                    tile_view(p_l1_slot, (m, base_k), (0, kt)),
                                )
                                mem_copy(
                                    v_l1.next_write(),
                                    v_t[
                                        kt * base_k : (kt + 1) * base_k,
                                        nt * base_n : (nt + 1) * base_n,
                                    ],
                                )
                                mem_copy(
                                    l0b.next_write(), (v_l1.next_read()).permute((1, 0))
                                )
                                matmul(
                                    l0c_slot,
                                    l0a.next_read(),
                                    l0b.next_read(),
                                    init=(kt == 0),
                                    unit_flag=3 if kt + 1 == k_tiles else 2,
                                )
                            mem_copy(
                                out_t[0:m, nt * base_n : (nt + 1) * base_n],
                                l0c.next_read(),
                                options=copy_options(unit_flag=3),
                            )

    @kernel
    def probability_value_matmul_kernel(
        self, out_gm: Tensor, p_gm: Tensor, v_gm: Tensor
    ):
        n_tiles = self.n // self.base_n
        for nt in range(n_tiles):
            self._compute_output_feature_tile(out_gm, p_gm, v_gm, nt)


class ProbabilityValueMatmulOp:
    def __init__(self, m, n, k, base_n=128, base_k=128):
        self.kernel = ProbabilityValueMatmulKernel(m, n, k, base_n, base_k)

    @host
    def run(self, out_gm, p_gm, v_gm):
        self.kernel.probability_value_matmul_kernel[1](out_gm, p_gm, v_gm)
