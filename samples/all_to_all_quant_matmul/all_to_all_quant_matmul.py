# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from cannbotdsl import (
    Buffer,
    MemLoc,
    StructRef,
    Tensor,
    dtypes,
    get_mem_size,
    host,
    jit,
    kernel,
    mem_copy,
    make_copy_engine,
    get_block_idx,
    get_subblock_id,
)
from cannbotdsl.channel import Channel
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.vf import vf
from cannbotdsl._mlir.dialects import cannir
from cannbotdsl.ops.distributed import (
    CommContext,
    SymBuffer,
    TransferRole,
    create_transfer_context,
)
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.sync import (
    PIPE,
    cube_sync_block_wait,
    dcci_entire_out,
    dci,
    vec_busy_wait,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_notify,
    vec_sync_pipe,
    vec_sync_wait,
    vec_sync_block_wait,
)
from cannbotdsl.core.ir_transport import unwrap_operand
from cannbotdsl.ops.distributed.distributed import (
    _addr_from_parts,
    _view_origin_elements,
)
from cannbotdsl.tensor import make_layout, make_pointer, make_tensor, tile_slice

MX_GROUP_SIZE = 64
SCALE_C0 = 2
COMM_WORKSPACE_SIZE = 512
PAYLOAD_BASE = 4096
ZN_FRACTAL_N = 16
COMM_MODES = ("urma", "ub")
OUT_DTYPES = ("bf16", "fp16", "fp32")
QUANT_DTYPES = ("e4m3", "e5m2", "fp4")

_MAX_HEAD_CNT = 14
_AIV_EPOCH_COMPLETE_FLAG = 14
_ACK_EVENT_ID = 2
_AIC_READY_FLAG = 9
_SYNC_META_BASE = 4096


class AllToAllQuantMatmul:
    def __init__(
        self,
        m,
        k,
        n,
        rank_size,
        comm_mode="urma",
        base_m=128,
        base_n=128,
        base_k=64,
        head_m=None,
        used_core_num=8,
        has_bias=False,
        out_dtype="bf16",
        a_dtype="e4m3",
        b_dtype="e4m3",
    ):
        self.m = m
        self.k = k
        self.n = n
        self.rank_size = rank_size
        self.comm_mode = comm_mode
        self.out_dtype = out_dtype
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.is_fp4 = a_dtype == "fp4"
        self.base_m = base_m
        self.base_n = base_n
        self.base_k = base_k
        self.head_m = head_m
        self.used_core_num = used_core_num
        self.has_bias = has_bias
        if self.head_m is None:
            self.head_m = self._default_head_m()
        self.ka = self.k // self.rank_size
        self.ka_bytes = self.ka // 2 if self.is_fp4 else self.ka
        self.scale_ka = (self.ka + MX_GROUP_SIZE - 1) // MX_GROUP_SIZE * SCALE_C0
        self.n_padded = (self.n + self.base_n - 1) // self.base_n * self.base_n
        self.m_padded = (self.m + self.base_m - 1) // self.base_m * self.base_m
        self.head_cnt = (self.m + self.head_m - 1) // self.head_m
        self._check_input()

    def aot_compile_specs(self):
        """AOT 参数规格：CommContext ×2 + a/scale_a/b/scale_b 字节流 + y + bias。"""
        from cannbotdsl import TensorSpec

        out_dtype = {
            "bf16": dtypes.bfloat16,
            "fp16": dtypes.float16,
            "fp32": dtypes.float32,
        }[self.out_dtype]
        return [
            StructRef[CommContext],
            StructRef[CommContext],
            TensorSpec((self.m_padded, self.ka_bytes), dtypes.uint8),
            TensorSpec((self.m_padded, self.scale_ka), dtypes.uint8),
            TensorSpec(
                (self.k, self.n_padded // 2)
                if self.is_fp4
                else (self.k, self.n_padded),
                dtypes.uint8,
            ),
            TensorSpec(
                (self.k // MX_GROUP_SIZE, self.n_padded, SCALE_C0), dtypes.uint8
            ),
            TensorSpec((self.m, self.n), out_dtype),
            TensorSpec((self.n_padded,) if self.has_bias else (0,), dtypes.float32),
        ]

    def required_comm_buffer_bytes(self):
        payload = (
            self.head_cnt
            * self.rank_size
            * self.head_m
            * (self.ka_bytes + self.scale_ka)
        )
        ack_offset = (payload + 31) // 32 * 32
        return PAYLOAD_BASE + ack_offset + self.head_cnt * self.rank_size * 4

    @host
    def run(
        self,
        data_ctx: StructRef[CommContext],
        barrier_ctx: StructRef[CommContext],
        a_gm: Tensor,
        scale_a_gm: Tensor,
        b_gm: Tensor,
        scale_b_gm: Tensor,
        y_gm: Tensor,
        bias_gm: Tensor,
    ):
        self._check_tensor_shapes((a_gm, scale_a_gm, b_gm, scale_b_gm, y_gm, bias_gm))
        alltoall_mm_kernel = _AllToAllQuantMatmulKernel(self)
        alltoall_mm_kernel[self.used_core_num](
            data_ctx,
            barrier_ctx,
            a_gm,
            scale_a_gm,
            b_gm,
            scale_b_gm,
            y_gm,
            bias_gm,
        )

    def _check_input(self):
        if self.comm_mode not in COMM_MODES:
            raise ValueError(
                f"comm_mode={self.comm_mode!r} must be one of {COMM_MODES}"
            )
        if self.out_dtype not in OUT_DTYPES:
            raise ValueError(
                f"out_dtype={self.out_dtype!r} must be one of {OUT_DTYPES}"
            )
        for name, dt in (("a_dtype", self.a_dtype), ("b_dtype", self.b_dtype)):
            if dt not in QUANT_DTYPES:
                raise ValueError(f"{name}={dt!r} must be one of {QUANT_DTYPES}")
        if (self.a_dtype == "fp4") != (self.b_dtype == "fp4"):
            raise ValueError("fp4 requires both a_dtype and b_dtype to be fp4")
        if self.a_dtype == "fp4" and self.n % 2 != 0:
            raise ValueError("fp4 packing requires an even n")
        if self.base_k % MX_GROUP_SIZE != 0:
            raise ValueError(
                f"base_k={self.base_k} must be a multiple of {MX_GROUP_SIZE} "
                "(the MX group size)"
            )
        if self.base_n % ZN_FRACTAL_N != 0:
            raise ValueError(
                f"base_n={self.base_n} must be a multiple of {ZN_FRACTAL_N} "
                "(the ZN fractal width)"
            )
        if self.k % (self.rank_size * self.base_k) != 0:
            raise ValueError(
                f"k={self.k} must be a multiple of rank_size*base_k="
                f"{self.rank_size * self.base_k}: K-tail blocks are not "
                "supported, each rank's K shard ka=k/rank_size must be "
                "base_k-aligned"
            )
        if self.head_m % self.base_m != 0:
            raise ValueError(
                f"head_m={self.head_m} must be a multiple of base_m="
                f"{self.base_m}: each epoch's tile grid reads base_m-row "
                "tiles inside the head_m-row share, a partial last tile "
                "would read into the next epoch's rows"
            )
        if self.head_cnt > _MAX_HEAD_CNT:
            raise ValueError(
                "full-depth all-to-all supports at most "
                f"{_MAX_HEAD_CNT} head slices (head_cnt); reduce head_m "
                "or add a payload-reuse protocol"
            )
        if self.used_core_num < self.rank_size:
            raise ValueError(
                "used_core_num must be at least rank_size: the UBMEM "
                "team barrier has one participating AIV block per rank"
            )

    def _check_tensor_shapes(self, tensors):
        a_gm, scale_a_gm, b_gm, scale_b_gm, y_gm, bias_gm = tensors
        b_shape = (
            (self.k, self.n_padded // 2) if self.is_fp4 else (self.k, self.n_padded)
        )
        expected = {
            "a": ((self.m_padded, self.ka_bytes), a_gm),
            "scale_a": ((self.m_padded, self.scale_ka), scale_a_gm),
            "b": (b_shape, b_gm),
            "scale_b": (
                (self.k // MX_GROUP_SIZE, self.n_padded, SCALE_C0),
                scale_b_gm,
            ),
            "y": ((self.m, self.n), y_gm),
            "bias": ((self.n_padded,) if self.has_bias else (0,), bias_gm),
        }
        for name, (want, tensor) in expected.items():
            got = tuple(tensor.shape)
            if got != want:
                raise ValueError(
                    f"{name} shape {got} does not match the AOT contract "
                    f"{want} (see aot_compile_specs)"
                )

    def _default_head_m(self):
        n_tiles = (self.n + self.base_n - 1) // self.base_n
        m_tiles_per_epoch = max(
            (self.used_core_num + n_tiles - 1) // n_tiles,
            (self.m + 4 * self.base_m - 1) // (4 * self.base_m),
        )
        return m_tiles_per_epoch * self.base_m


@kernel
class _AllToAllQuantMatmulKernel:
    """设备侧 MIX kernel：AIV 通信流 + AIC matmul 流。"""

    def __init__(self, op):
        """设备侧参数落位与通信/计算资源配置。"""
        self.m = op.m
        self.n = op.n
        self.rank_size = op.rank_size
        self.ka = op.ka
        self.scale_ka = op.scale_ka
        self.base_m = op.base_m
        self.base_n = op.base_n
        self.base_k = op.base_k
        self.head_m = op.head_m
        self.used_core_num = op.used_core_num
        self.n_padded = op.n_padded
        self.has_bias = op.has_bias
        self.comm_mode = op.comm_mode
        self.head_cnt = op.head_cnt
        self.is_fp4 = op.is_fp4
        self.ka_bytes = op.ka_bytes
        dtype_map = {
            "e4m3": dtypes.float8_e4m3fn,
            "e5m2": dtypes.float8_e5m2,
            "fp4": dtypes.fp4x2_e2m1,
        }
        a_dt = dtype_map.get(op.a_dtype)
        b_dt = dtype_map.get(op.b_dtype)
        if a_dt is None or b_dt is None:
            raise ValueError(f"a_dtype/b_dtype must be one of {tuple(dtype_map)}")
        self.a_dtype = a_dt
        self.b_dtype = b_dt
        self.data_ctx = self.a_gm = self.scale_a_gm = None
        self.b_gm = self.scale_b_gm = self.y_gm = self.bias_gm = None
        self.rank_id = self.workers = None
        self.sym_a = self.sym_scale_a = self.sym_ack = self.sync_meta = None
        self.b_view = self.a_src = self.scale_a_src = None
        self.scale_a_l1_read = self.scale_b_l1_read = None
        self.a_l1_read = self.b_l1_read = None
        self.launch_counter_ub = self.launch_one_ub = self.ack_read_ub = None
        self.comm_workspace_ub = None
        self.ub_stage_rows_a = self.ub_stage_rows_scale_a = None
        self.ub_stage_a = self.ub_stage_scale_a = None
        self.nd2nz = self.identity = self.mx_scale_a = None
        self.mx_scale_b = self.fixpipe = None
        self.bias_l1 = self.bias_bt = self.bias_nd = None
        self.a_l1 = self.b_l1 = self.scale_a_l1 = self.scale_b_l1 = None
        self.l0a = self.l0b = self.l0c = None
        self._init_comm_config()
        self._init_matmul_config()

    def __call__(
        self,
        data_ctx: StructRef[CommContext],
        barrier_ctx: StructRef[CommContext],
        a: Tensor,
        sa: Tensor,
        b: Tensor,
        sb: Tensor,
        y: Tensor,
        bias: Tensor,
    ):
        self.data_ctx = data_ctx
        self.a_gm = a
        self.scale_a_gm = sa
        self.b_gm = b
        self.scale_b_gm = sb
        self.y_gm = y
        self.bias_gm = bias
        self.rank_id = dtypes.int64(data_ctx.rankId)
        self.workers = dtypes.int64(self.used_core_num)
        (
            self.sym_a,
            self.sym_scale_a,
            self.sym_ack,
            self.sync_meta,
        ) = self._build_symbuffer(data_ctx, barrier_ctx)
        self.b_view = self._build_b_view(b)
        self._comm_process()
        self._matmul_process()

    @jit
    def _barrier_all_ack(self, sym_ack, epoch, rank_id):
        if const_expr(self.comm_mode == "ub"):
            vec_sync_pipe(PIPE.MTE3)
        for src in range(self.rank_size):
            if dtypes.int64(src) != rank_id:
                self._poll_ack(sym_ack, epoch, src)

    @jit
    def _build_b_view(self, b_gm):
        b_addr = cannir.extract_address(cannir.extract_pointer(unwrap_operand(b_gm)))
        if const_expr(self.is_fp4):
            return make_tensor(
                make_pointer(dtypes.fp4x2_e2m1, b_addr, MemLoc.GM),
                make_layout((self.rank_size * self.ka, self.n_padded)),
            )
        return make_tensor(
            make_pointer(self.b_dtype, b_addr, MemLoc.GM),
            make_layout((self.rank_size * self.ka, self.n_padded)),
            data_format="zn",
        )

    @jit
    def _build_local_epoch_view(self, src_gm, epoch, cols, dtype):
        epoch_addr = _addr_from_parts(
            _view_origin_elements(unwrap_operand(src_gm))[0],
            [dtypes.int64(epoch) * dtypes.int64(self.head_m * cols)],
        )
        return make_tensor(
            make_pointer(dtype, epoch_addr, MemLoc.GM),
            make_layout((self.head_m, cols)),
        )

    @jit
    def _build_symbuffer(self, data_ctx, barrier_ctx):
        sym_a = SymBuffer(
            data_ctx,
            shape=(self.head_cnt * self.rank_size * self.head_m, self.ka_bytes),
            dtype=dtypes.uint8 if self.is_fp4 else self.a_dtype,
            rank_size=self.rank_size,
            heap_base=PAYLOAD_BASE,
        )
        sym_scale_a = SymBuffer(
            data_ctx,
            shape=(
                self.head_cnt * self.rank_size * self.head_m,
                self.scale_ka,
            ),
            dtype=dtypes.uint8,
            rank_size=self.rank_size,
            heap_base=PAYLOAD_BASE,
            base_offset=(self.head_cnt * self.rank_size * self.head_m * self.ka_bytes),
        )
        ack_offset = (
            (
                self.head_cnt
                * self.rank_size
                * self.head_m
                * (self.ka_bytes + self.scale_ka)
                + 31
            )
            // 32
            * 32
        )
        sym_ack = SymBuffer(
            data_ctx,
            shape=(self.head_cnt, self.rank_size),
            dtype=dtypes.int32,
            rank_size=self.rank_size,
            heap_base=PAYLOAD_BASE,
            base_offset=ack_offset,
        )
        sync_meta = SymBuffer(
            barrier_ctx,
            shape=(1, 8),
            dtype=dtypes.int32,
            rank_size=self.rank_size,
            heap_base=_SYNC_META_BASE,
        )
        return sym_a, sym_scale_a, sym_ack, sync_meta

    @jit
    def _comm_process(self):
        if get_subblock_id() == 0 and dtypes.int64(get_block_idx()) == dtypes.int64(0):
            with vf(mode="simt", thread=1):
                self.launch_one_ub[dtypes.int64(0), dtypes.int64(0)] = dtypes.int32(1)
            mem_copy(
                tile_slice(self.sync_meta, (1, 1), (0, 0)),
                tile_slice(self.launch_one_ub, (1, 1), (0, 0)),
                atomic_add=True,
            )
            vec_sync_pipe(PIPE.MTE3)
        vec_sync_all()
        vec_sync_block_arrive(PIPE.MTE3, _AIV_EPOCH_COMPLETE_FLAG, mode=0)
        vec_sync_block_wait(PIPE.S, _AIV_EPOCH_COMPLETE_FLAG, mode=0)
        if get_subblock_id() == 0:
            self._read_launch_counter(self.sync_meta)
        for epoch in range(self.head_cnt):
            if get_subblock_id() == 0:
                self._comm_run(epoch)
            self._wait_comm_done()
            self._notify_comm_done()
        dcci_entire_out()
        dci()

    @jit
    def _comm_run(self, epoch):
        if dtypes.int64(get_block_idx()) < dtypes.int64(self.rank_size):
            trans_ctx = self._make_trans_ctx()
            for block_idx in range(
                get_block_idx(), dtypes.int64(self.rank_size), self.workers
            ):
                self._push_to_peer(trans_ctx, epoch, block_idx)
            self._barrier_all_ack(self.sym_ack, epoch, self.rank_id)

    @jit
    def _epoch_share_views(self, epoch):
        a_share = tile_slice(self.a_gm, (self.head_m, self.ka_bytes), (epoch, 0))
        scale_share = tile_slice(
            self.scale_a_gm, (self.head_m, self.scale_ka), (epoch, 0)
        )
        inbox_row = dtypes.int64(epoch) * dtypes.int64(self.rank_size) + self.rank_id
        a_slot = tile_slice(self.sym_a, (self.head_m, self.ka_bytes), (inbox_row, 0))
        scale_slot = tile_slice(
            self.sym_scale_a, (self.head_m, self.scale_ka), (inbox_row, 0)
        )
        return a_share, scale_share, a_slot, scale_slot

    @jit
    def _load_rank_scales(self, src_rank, epoch_m_tile, n_tile):
        scale_a_tile = tile_slice(
            self.scale_a_src, (self.base_m, self.scale_ka), (epoch_m_tile, 0)
        )
        scale_a_l1_write = self.scale_a_l1.produce()
        mem_copy(scale_a_l1_write, scale_a_tile, engine=self.mx_scale_a)
        self.scale_a_l1_read = self.scale_a_l1.consume()
        scale_b_tile = tile_slice(
            self.scale_b_gm,
            (self.scale_pairs, self.base_n, SCALE_C0),
            (src_rank, n_tile, 0),
        )
        scale_b_l1_write = self.scale_b_l1.produce()
        mem_copy(scale_b_l1_write, scale_b_tile, engine=self.mx_scale_b)
        self.scale_b_l1_read = self.scale_b_l1.consume()

    @jit
    def _make_trans_ctx(self):
        if const_expr(self.comm_mode == "urma"):
            return create_transfer_context(
                self.data_ctx, self.comm_workspace_ub, role=TransferRole.VEC
            )
        return None

    @jit
    def _matmul_k_iter(self, src_rank, epoch_m_tile, n_tile, l0c_acc, bias_bt):
        for k_block in range(self.l1_k_blocks):
            k_tile_start = k_block * self.step_k
            global_k_block = src_rank * dtypes.int64(self.l1_k_blocks) + dtypes.int64(
                k_block
            )
            self._stage_l1(global_k_block, k_block, n_tile, epoch_m_tile)
            for k_sub in range(self.step_k):
                self._stage_l0_mm(
                    k_sub, k_tile_start + k_sub, src_rank, l0c_acc, bias_bt
                )

    @jit
    def _matmul_process(self):
        for epoch in range(self.head_cnt):
            cube_sync_block_wait(PIPE.MTE2, _AIC_READY_FLAG, mode=2)
            valid_rows = min(self.head_m, self.m - epoch * self.head_m)
            valid_m_tiles = (valid_rows + self.base_m - 1) // self.base_m
            tiles_per_epoch = self.m_tiles_per_epoch * self.n_tiles
            for tile_idx in range(
                dtypes.int64(get_block_idx()),
                dtypes.int64(tiles_per_epoch),
                self.workers,
            ):
                epoch_m_tile = tile_idx % self.m_tiles_per_epoch
                if dtypes.int64(epoch_m_tile) < valid_m_tiles:
                    self._matmul_tile(
                        epoch, epoch_m_tile, tile_idx // self.m_tiles_per_epoch
                    )

    @jit
    def _matmul_rank_iter(self, epoch, epoch_m_tile, n_tile, l0c_acc, bias_bt):
        for r in range(self.rank_size):
            src_rank = dtypes.int64(r)
            self._select_a_source(epoch, src_rank)
            self._load_rank_scales(src_rank, epoch_m_tile, n_tile)
            self._matmul_k_iter(src_rank, epoch_m_tile, n_tile, l0c_acc, bias_bt)

    @jit
    def _matmul_tile(self, epoch, epoch_m_tile, n_tile):
        bias_bt = None
        if const_expr(self.has_bias):
            bias_tile = tile_slice(self.bias_gm, (self.base_n,), (n_tile,))
            mem_copy(self.bias_l1.produce(), bias_tile, engine=self.bias_nd)
            mem_copy(self.bias_bt.produce(), self.bias_l1.consume())
            bias_bt = self.bias_bt.consume()
        l0c_acc = self.l0c.produce()
        self._matmul_rank_iter(epoch, epoch_m_tile, n_tile, l0c_acc, bias_bt)
        l0c_result = self.l0c.consume()
        global_m_tile = epoch * self.m_tiles_per_epoch + epoch_m_tile
        out_tile = tile_slice(
            self.y_gm, (self.base_m, self.base_n), (global_m_tile, n_tile)
        )
        mem_copy(out_tile, l0c_result, engine=self.fixpipe)

    @jit
    def _notify_comm_done(self):
        vec_sync_block_arrive(PIPE.MTE3, _AIC_READY_FLAG, mode=2)

    @jit
    def _push_share_ub(self, epoch, block_idx):
        valid_rows = min(self.head_m, self.m - epoch * self.head_m)
        a_share, scale_share, a_slot, scale_slot = self._epoch_share_views(epoch)
        self._ub_push_share_a(a_share, a_slot, valid_rows, block_idx)
        self._ub_push_share_scale_a(scale_share, scale_slot, valid_rows, block_idx)
        mem_copy(
            tile_slice(self.sym_ack, (1, 1), (epoch, self.rank_id)).peer(block_idx),
            tile_slice(self.launch_counter_ub, (1, 1), (0, 0)),
            l2_cache_ctl=4,
        )

    @jit
    def _push_share_urma(self, trans_ctx, epoch, block_idx):
        a_share, scale_share, a_slot, scale_slot = self._epoch_share_views(epoch)
        trans_ctx.put(block_idx, a_slot.peer(block_idx), a_share)
        trans_ctx.put(block_idx, scale_slot.peer(block_idx), scale_share)
        trans_ctx.put(
            block_idx,
            tile_slice(self.sym_ack, (1, 1), (epoch, self.rank_id)).peer(block_idx),
            tile_slice(self.sync_meta, (1, 1), (0, 0)),
        )
        trans_ctx.flush(block_idx)

    @jit
    def _push_to_peer(self, trans_ctx, epoch, block_idx):
        if dtypes.int64(block_idx) != self.rank_id:
            if const_expr(self.comm_mode == "urma"):
                self._push_share_urma(trans_ctx, epoch, block_idx)
            else:
                self._push_share_ub(epoch, block_idx)

    @jit
    def _read_launch_counter(self, sync_meta):
        counter_view = tile_slice(sync_meta, (1, 1), (0, 0))
        read_view = tile_slice(self.launch_counter_ub, (1, 1), (0, 0))
        vec_sync_notify(PIPE.S, PIPE.MTE2, _ACK_EVENT_ID)
        vec_sync_wait(PIPE.S, PIPE.MTE2, _ACK_EVENT_ID)
        mem_copy(read_view, counter_view, l2_cache_ctl=4)
        vec_sync_notify(PIPE.MTE2, PIPE.S, _ACK_EVENT_ID)
        vec_sync_wait(PIPE.MTE2, PIPE.S, _ACK_EVENT_ID)

    @jit
    def _reinterpret_fp4(self, byte_view, rows, cols):
        addr = cannir.extract_address(cannir.extract_pointer(unwrap_operand(byte_view)))
        return make_tensor(
            make_pointer(dtypes.fp4x2_e2m1, addr, MemLoc.GM),
            make_layout((rows, cols)),
        )

    @jit
    def _select_a_source(self, epoch, src_rank):
        is_local = src_rank == self.rank_id
        a_local = self._build_local_epoch_view(
            self.a_gm,
            epoch,
            self.ka_bytes,
            dtypes.uint8 if self.is_fp4 else self.a_dtype,
        )
        scale_a_local = self._build_local_epoch_view(
            self.scale_a_gm, epoch, self.scale_ka, dtypes.uint8
        )
        inbox_row = dtypes.int64(epoch) * dtypes.int64(self.rank_size) + src_rank
        a_remote = tile_slice(self.sym_a, (self.head_m, self.ka_bytes), (inbox_row, 0))
        scale_a_remote = tile_slice(
            self.sym_scale_a,
            (self.head_m, self.scale_ka),
            (inbox_row, 0),
        )
        if const_expr(self.is_fp4):
            a_local = self._reinterpret_fp4(a_local, self.head_m, self.ka)
            a_remote = self._reinterpret_fp4(a_remote, self.head_m, self.ka)
        self.a_src = a_local if is_local else a_remote
        self.scale_a_src = scale_a_local if is_local else scale_a_remote

    @jit
    def _stage_l0_mm(self, k_sub, k_tile, src_rank, l0c_acc, bias_bt):
        scale_a_k_tile = tile_slice(
            self.scale_a_l1_read,
            (self.base_m, self.scale_base_k),
            (0, k_tile),
        )
        scale_b_k_tile = tile_slice(
            self.scale_b_l1_read,
            (self.scale_base_k, self.base_n),
            (k_tile, 0),
        )
        l0a_write = self.l0a.produce()
        mem_copy(
            l0a_write,
            tile_slice(self.a_l1_read, (self.base_m, self.base_k), (0, k_sub)),
            mx_scale=scale_a_k_tile,
        )
        l0b_write = self.l0b.produce()
        if const_expr(self.is_fp4):
            mem_copy(
                l0b_write,
                tile_slice(self.b_l1_read, (self.base_k, self.base_n), (k_sub, 0)),
                mx_scale=scale_b_k_tile,
                transpose=True,
            )
        else:
            mem_copy(
                l0b_write,
                tile_slice(self.b_l1_read, (self.base_k, self.base_n), (k_sub, 0)),
                mx_scale=scale_b_k_tile,
            )
        init = src_rank + dtypes.int64(k_tile) == dtypes.int64(0)
        matmul(
            l0c_acc,
            self.l0a.consume(),
            self.l0b.consume(),
            init=init,
            bias=bias_bt,
        )

    @jit
    def _stage_l1(self, global_k_block, k_block, n_tile, epoch_m_tile):
        b_k_block = tile_slice(
            self.b_view,
            (self.base_k * self.step_k, self.base_n),
            (global_k_block, n_tile),
        )
        b_l1_write = self.b_l1.produce()
        if const_expr(self.is_fp4):
            mem_copy(b_l1_write, b_k_block, engine=self.nd2nz)
        else:
            mem_copy(b_l1_write, b_k_block, engine=self.identity)
        self.b_l1_read = self.b_l1.consume()
        a_k_block = tile_slice(
            self.a_src,
            (self.base_m, self.base_k * self.step_k),
            (epoch_m_tile, k_block),
        )
        a_l1_write = self.a_l1.produce()
        mem_copy(a_l1_write, a_k_block, engine=self.nd2nz)
        self.a_l1_read = self.a_l1.consume()

    @jit
    def _ub_push_share_a(self, share, slot, valid_rows, block_idx):
        chunks = (valid_rows + self.ub_stage_rows_a - 1) // self.ub_stage_rows_a
        for chunk in range(chunks):
            n_rows = min(
                self.ub_stage_rows_a,
                valid_rows - chunk * self.ub_stage_rows_a,
            )
            stage_write = self.ub_stage_a.produce()
            mem_copy(
                stage_write,
                tile_slice(share, (self.ub_stage_rows_a, self.ka_bytes), (chunk, 0)),
            )
            stage_read = self.ub_stage_a.consume()
            row_lo = chunk * self.ub_stage_rows_a
            row_hi = row_lo + n_rows
            peer_view = slot.peer(block_idx)
            mem_copy(peer_view[row_lo:row_hi, None], stage_read[0:n_rows, None])

    @jit
    def _ub_push_share_scale_a(self, share, slot, valid_rows, block_idx):
        chunks = (
            valid_rows + self.ub_stage_rows_scale_a - 1
        ) // self.ub_stage_rows_scale_a
        for chunk in range(chunks):
            n_rows = min(
                self.ub_stage_rows_scale_a,
                valid_rows - chunk * self.ub_stage_rows_scale_a,
            )
            stage_write = self.ub_stage_scale_a.produce()
            mem_copy(
                stage_write,
                tile_slice(
                    share,
                    (self.ub_stage_rows_scale_a, self.scale_ka),
                    (chunk, 0),
                ),
            )
            stage_read = self.ub_stage_scale_a.consume()
            row_lo = chunk * self.ub_stage_rows_scale_a
            row_hi = row_lo + n_rows
            peer_view = slot.peer(block_idx)
            mem_copy(peer_view[row_lo:row_hi, None], stage_read[0:n_rows, None])

    @jit
    def _wait_comm_done(self):
        vec_sync_all()
        vec_sync_block_arrive(PIPE.MTE3, _AIV_EPOCH_COMPLETE_FLAG, mode=0)
        vec_sync_block_wait(PIPE.S, _AIV_EPOCH_COMPLETE_FLAG, mode=0)

    def _init_comm_config(self):
        self.launch_counter_ub = Buffer(MemLoc.UB, shape=(1, 1), dtype=dtypes.int32)
        self.launch_one_ub = Buffer(MemLoc.UB, shape=(1, 1), dtype=dtypes.int32)
        self.ack_read_ub = Buffer(MemLoc.UB, shape=(1, 1), dtype=dtypes.int32)
        if self.comm_mode == "urma":
            self.comm_workspace_ub = Buffer(
                MemLoc.UB, shape=(COMM_WORKSPACE_SIZE,), dtype=dtypes.uint8
            )
        else:
            stage_budget = get_mem_size("ub") // 2
            self.ub_stage_rows_a = self._ub_stage_rows(self.ka_bytes, stage_budget)
            self.ub_stage_rows_scale_a = self._ub_stage_rows(
                self.scale_ka, stage_budget
            )
            self.ub_stage_a = Channel(
                MemLoc.UB,
                shape=(self.ub_stage_rows_a, self.ka_bytes),
                dtype=dtypes.uint8,
                depth=1,
            )
            self.ub_stage_scale_a = Channel(
                MemLoc.UB,
                shape=(self.ub_stage_rows_scale_a, self.scale_ka),
                dtype=dtypes.uint8,
                depth=1,
            )

    def _init_copy_engines(self):
        self.nd2nz = make_copy_engine(format_transform="nd2nz")
        self.identity = make_copy_engine(format_transform="identity")
        self.mx_scale_a = make_copy_engine(format_transform="mx_scale_a")
        self.mx_scale_b = make_copy_engine(format_transform="mx_scale_bnd")
        self.fixpipe = make_copy_engine()
        if self.has_bias:
            self.bias_l1 = Channel(
                MemLoc.L1, (self.base_n,), dtypes.float32, depth=2, data_format="nd"
            )
            self.bias_bt = Channel(
                MemLoc.BIAS, (self.base_n,), dtypes.float32, depth=2, data_format="nd"
            )
            self.bias_nd = make_copy_engine(format_transform="identity")

    def _init_matmul_channels(self, l1_k):
        self.a_l1 = Channel(
            MemLoc.L1,
            shape=(self.base_m, l1_k),
            dtype=self.a_dtype,
            depth=2,
            **({"data_format": "nz"} if self.is_fp4 else {}),
        )
        self.b_l1 = Channel(
            MemLoc.L1,
            shape=(l1_k, self.base_n),
            dtype=self.b_dtype,
            depth=2,
            data_format="nz" if self.is_fp4 else "zn",
        )
        self.scale_a_l1 = Channel(
            MemLoc.L1,
            shape=(self.base_m, self.scale_ka),
            dtype=dtypes.float8_e8m0,
            depth=2,
            data_format="zn",
        )
        self.scale_b_l1 = Channel(
            MemLoc.L1,
            shape=(self.scale_ka, self.base_n),
            dtype=dtypes.float8_e8m0,
            depth=2,
            data_format="nz",
        )
        self.l0a = Channel(
            MemLoc.L0A,
            shape=(self.base_m, self.base_k),
            dtype=self.a_dtype,
            depth=2,
            **({"data_format": "nz"} if self.is_fp4 else {}),
        )
        l0b_shape = (
            (self.base_n * 2, self.base_k)
            if self.is_fp4
            else (self.base_k, self.base_n)
        )
        self.l0b = Channel(
            MemLoc.L0B,
            shape=l0b_shape,
            dtype=self.b_dtype,
            depth=2,
            **({"data_format": "zn"} if self.is_fp4 else {}),
        )
        self.l0c = Channel(
            MemLoc.L0C, shape=(self.base_m, self.base_n), dtype=dtypes.float32, depth=1
        )

    def _init_matmul_config(self):
        """AIC 计算侧资源配置：tiling 派生 + L1/L0 通道 + 搬运引擎。"""
        self.n_tiles = (self.n + self.base_n - 1) // self.base_n
        self.m_tiles_per_epoch = (self.head_m + self.base_m - 1) // self.base_m
        self.k_tiles = self.ka // self.base_k
        self.step_k = 2 if self.k_tiles % 2 == 0 else 1
        self.l1_k_blocks = self.k_tiles // self.step_k
        self.scale_base_k = (
            (self.base_k + MX_GROUP_SIZE - 1) // MX_GROUP_SIZE * SCALE_C0
        )
        self.scale_pairs = self.scale_ka // SCALE_C0
        self._init_matmul_channels(self.base_k * self.step_k)
        self._init_copy_engines()

    def _poll_ack(self, sym_ack, epoch, src):
        ack_view = tile_slice(sym_ack, (1, 1), (epoch, src))
        ack_read = tile_slice(self.ack_read_ub, (1, 1), (0, 0))

        def _ack_ready():
            vec_sync_notify(PIPE.S, PIPE.MTE2, _ACK_EVENT_ID)
            vec_sync_wait(PIPE.S, PIPE.MTE2, _ACK_EVENT_ID)
            mem_copy(ack_read, ack_view, l2_cache_ctl=4)
            vec_sync_notify(PIPE.MTE2, PIPE.S, _ACK_EVENT_ID)
            vec_sync_wait(PIPE.MTE2, PIPE.S, _ACK_EVENT_ID)
            return (
                self.ack_read_ub[dtypes.int64(0), dtypes.int64(0)]
                == self.launch_counter_ub[dtypes.int64(0), dtypes.int64(0)]
            )

        vec_busy_wait(_ack_ready)

    def _ub_stage_rows(self, cols, budget):
        """UB 分块行数：head_m/预算/base_m 约束下取 base_m 的最大因子。"""
        rows = max(1, min(self.head_m, budget // cols, self.base_m))
        while rows > 1 and self.base_m % rows != 0:
            rows -= 1
        return rows
