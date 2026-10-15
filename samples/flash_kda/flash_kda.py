# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FlashKDA AICore consumer for BNSD, BSND, and packed TND storage."""

import dataclasses
from functools import lru_cache, wraps
import os
from typing import NamedTuple, Optional, Tuple

import cannbotdsl
import torch
from cannbotdsl import aot, dtypes, get_platform_info
from cannbotdsl.lang.host import host
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.buffer import Buffer
from cannbotdsl.channel import Channel
from cannbotdsl.ops.sync import channel_rewind
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.types.delay_line import DelayLineGroup
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops import reg as reg_ops
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    full_mask,
    mask_xor,
    update_mask,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdups,
    vexp,
    vexp_sub,
    vload,
    vload_broadcast,
    vload_unpack,
    vmem_bar,
    vmul,
    vmuls,
    vreduce_sum,
    vstore,
    vstore_first,
    vstore_pack,
    vsqrt,
)
from cannbotdsl.ops.reg import vselect as vselect_raw
from cannbotdsl.ops.sync import (
    cube_fill_l1_zero,
    cube_sync_all,
    cube_sync_pipe,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
    vec_sync_notify,
    vec_sync_wait,
)
from cannbotdsl.tensor import idx2crd, reinterpret, tile_slice
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy, CopyBlockMapping
from cannbotdsl import ChannelKind, MemLoc, PIPE, Tensor
from cannbotdsl.lang import vf


os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

compile_cache = getattr(aot, "compile_cache", lru_cache(maxsize=1))

CHUNK_SIZE = 64
SUPPORTED_HEAD_DIM = 128
HALF_CHUNK_SIZE = CHUNK_SIZE // 2
D_HALF = SUPPORTED_HEAD_DIM // 2
LOW_DTYPE = torch.bfloat16
HIGH_DTYPE = torch.float32

# Ascend 9.2 CAPI raw encodings, not ordinal cache-policy indices.
_L2_CACHE_LAST_USE = 1
_L2_CACHE_DISABLE = 4


@jit
def cast_tile(dst, src):
    rows, cols = dst.shape
    src_stride = src.physical_stride[0]
    dst_stride = dst.physical_stride[0]
    with vf(mode="simd"):
        for row in range(rows):
            if const_expr(isinstance(cols, int) and cols == 1):
                value = vload_broadcast(src, row * src_stride)
                converted = vcast(value, dst.dtype, mask=full_mask())
                vstore_first(dst, row * dst_stride, converted)
            else:
                for col in range(0, cols, 64):
                    mask, _ = update_mask(cols - col, elem_bits=32)
                    if const_expr(src.dtype == dtypes.float32):
                        value = vload(src, row * src_stride + col)
                    else:
                        value = vload_unpack(src, row * src_stride + col, unpack_mode=UnpackMode.B16_TO_B32)
                    converted = vcast(value, dst.dtype, mask=mask)
                    if const_expr(dst.dtype == dtypes.float32):
                        vstore(dst, row * dst_stride + col, converted, mask)
                    else:
                        vstore_pack(dst, row * dst_stride + col, converted, mask, pack_mode=PackMode.B32_TO_B16)


@jit
def prefix_sum_rows(dst, src):
    rows, cols = dst.shape
    src_stride = src.physical_stride[0]
    dst_stride = dst.physical_stride[0]
    with vf(mode="simd"):
        for col in range(0, cols, 64):
            mask, _ = update_mask(cols - col, elem_bits=32)
            prefix = vload(src, col)
            vstore(dst, col, prefix, mask)
            for row in range(1, rows):
                row_value = vload(src, row * src_stride + col)
                prefix = vadd(prefix, row_value, mask=mask)
                vstore(dst, row * dst_stride + col, prefix, mask)


def _valid_rows_view(full_tile, valid_rows, cols: int):
    """Clip a rebased packed-TND tile to this logical sequence."""
    return full_tile[:valid_rows, :cols]


def _o_head_chunk_tile(gm_O, seq_idx, chunk_idx, dv_base, dv_idx, valid_rows):
    """Return one logical BNSD output chunk while preserving physical strides."""
    nv = gm_O.shape[1]
    batch_idx = seq_idx // nv
    head_idx = seq_idx - batch_idx * nv
    full_chunk = tile_slice(
        gm_O[batch_idx, head_idx, None, None],
        (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
        (chunk_idx, 0),
    )
    column_tile = tile_slice(full_chunk, (CHUNK_SIZE, dv_base), (0, dv_idx))
    return _valid_rows_view(column_tile, valid_rows, dv_base)


@jit
def nd2nz_tile(dst, src, elem_bits: int = 16):
    rows, cols = src.shape
    src_stride = src.physical_stride[0]
    dst_stride = dst.physical_stride[0]
    n0 = 256 // elem_bits
    block_stride = dst_stride * elem_bits // 256
    with vf(mode="simd"):
        for row in range(rows):
            remaining = reg_ops.mask_counter(cols)
            for col in range(0, cols, 2048 // elem_bits):
                mask, remaining = reg_ops.update_mask(remaining, elem_bits=elem_bits)
                values = reg_ops.vload(src, row * src_stride + col)
                offset = row * n0 + (col // n0) * dst_stride
                reg_ops.vstore_strided(dst, offset, values, mask,
                              block_stride=block_stride,
                              repeat_stride=0)



class _Stage1Constants(NamedTuple):
    identity16: torch.Tensor


_STAGE1_CONSTANTS_CACHE: dict[tuple[str, int | None, torch.dtype], _Stage1Constants] = {}


@dataclasses.dataclass
class StageOneWorkspace:
    Q_decayed: torch.Tensor
    Mqk: torch.Tensor
    K_restored: torch.Tensor
    gamma_C: torch.Tensor
    U_pre: torch.Tensor
    W: torch.Tensor
    identity16: torch.Tensor


def _empty_like_device(
    shape: tuple[int, ...],
    *,
    ref: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=ref.device)


def allocate_qk_exchange_workspace(
    *, active_block_num: int, ref: torch.Tensor,
) -> torch.Tensor:
    """Allocate two Q/K partial sums in each AIV exchange direction."""
    assert active_block_num > 0, "active_block_num must be positive"
    return _empty_like_device(
        (active_block_num, 2, 2, CHUNK_SIZE), ref=ref, dtype=HIGH_DTYPE,
    )


def qk_exchange_slices(
    workspace: torch.Tensor, *, block_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return this block's two AIV-to-AIV exchange directions."""
    assert workspace.dtype == HIGH_DTYPE
    assert workspace.dim() == 4 and tuple(workspace.shape[1:]) == (2, 2, CHUNK_SIZE)
    assert 0 <= block_idx < workspace.shape[0], "block_idx is outside the exchange workspace"
    return workspace[block_idx, 0], workspace[block_idx, 1]


def _resolved_device_key(ref: torch.Tensor) -> tuple[str, int | None]:
    device = ref.device
    device_type = getattr(device, "type", "cpu")
    device_index = getattr(device, "index", None)
    if device_type in {"npu", "privateuseone"} and device_index is None:
        npu = getattr(torch, "npu", None)
        current_device = getattr(npu, "current_device", None) if npu is not None else None
        if current_device is not None:
            device_index = int(current_device())
    return device_type, device_index


def _stage1_constants_for(
    ref: torch.Tensor,
    *,
    identity_dtype: torch.dtype = torch.float16,
) -> _Stage1Constants:
    device_type, device_index = _resolved_device_key(ref)
    key = (device_type, device_index, identity_dtype)
    cached = _STAGE1_CONSTANTS_CACHE.get(key)
    if cached is not None:
        return cached

    constants = _Stage1Constants(
        identity16=torch.eye(16, dtype=identity_dtype, device=ref.device).contiguous(),
    )
    _STAGE1_CONSTANTS_CACHE[key] = constants
    return constants


def allocate_stage1_workspace(
    batch: int,
    heads: int,
    eff_group: int,
    dim: int,
    *,
    ref: torch.Tensor,
) -> StageOneWorkspace:
    # Scratch is reused for each chunk group rather than spanning the sequence.
    bn_chunks = batch * heads * eff_group
    constants = _stage1_constants_for(ref)
    return StageOneWorkspace(
        Q_decayed=_empty_like_device((bn_chunks * CHUNK_SIZE, dim), ref=ref, dtype=LOW_DTYPE),
        Mqk=_empty_like_device((bn_chunks * CHUNK_SIZE, CHUNK_SIZE), ref=ref, dtype=LOW_DTYPE),
        K_restored=_empty_like_device((bn_chunks * CHUNK_SIZE, dim), ref=ref, dtype=LOW_DTYPE),
        gamma_C=_empty_like_device((bn_chunks * dim, 1), ref=ref, dtype=HIGH_DTYPE),
        U_pre=_empty_like_device((bn_chunks * CHUNK_SIZE, dim), ref=ref, dtype=LOW_DTYPE),
        W=_empty_like_device((bn_chunks * CHUNK_SIZE, dim), ref=ref, dtype=LOW_DTYPE),
        identity16=constants.identity16,
    )


ELEM_BYTES = 4
VL = 64  # 一个 256B 向量寄存器的 f32/b32 lane 数（raw VF 步长）；此处恰等于 CHUNK_SIZE，一行 = 一寄存器

NEUMANN_BLOCK_SIZE = 16
NEUMANN_DIAG_BLOCK_NUM = 4
NEUMANN_FRACTAL_ELEMS = 256                                                     # 一个 16x16 块的元素数。
NEUMANN_DIAG_SRC_STRIDE = NEUMANN_BLOCK_SIZE * CHUNK_SIZE + NEUMANN_BLOCK_SIZE  # ND 源矩阵相邻对角块起点间隔。
NEUMANN_DIAG_DST_STRIDE = (NEUMANN_DIAG_BLOCK_NUM + 1) * NEUMANN_FRACTAL_ELEMS  # packed/NZ 目标相邻对角块起点间隔。


class StageOneMatmul:
    """StageOne Cube 侧矩阵乘计算。"""

    def __init__(self):
        # Neumann packed64 求逆工作区。
        self.tile_power_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1).produce()
        self.packed_inv64_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1).produce()
        self.tile_inv_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1).produce()
        self.scratch_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1).produce()
        self.neumann_power_handoff_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1, kind=ChannelKind.CrossCore).produce()
        self.packed_identity_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1).produce()
        self.zero_l1 = Buffer(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16)
        self.resident_inv_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.bfloat16, depth=1).produce()
        # L0 — double-buffer + Neumann16
        self.l0a_db = Channel(MemLoc.L0A, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=2)
        self.l0b_db = Channel(MemLoc.L0B, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=2)
        self.l0c_db = Channel(MemLoc.L0C, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.float32, depth=2)
        self.per_d_left_k_l0a = Channel(MemLoc.L0A, (NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=1).produce()
        self.per_d_left_q_l0a = Channel(MemLoc.L0A, (NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=1).produce()
        self.per_d_l0b_db = Channel(MemLoc.L0B, (NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=2)
        self.raw_l0a_db = Channel(MemLoc.L0A, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=2)
        self.raw_l0b_db = Channel(MemLoc.L0B, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=2)
        self.raw_l0c_main = Channel(MemLoc.L0C, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float32, depth=1).produce()
        self.raw_l0c_aux = Channel(MemLoc.L0C, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float32, depth=1).produce()
        # 引擎
        self.gm2l1_identity16_diag = make_copy_engine(format_transform='nd2nz', block_mapping=CopyBlockMapping((16, 16), repeats=NEUMANN_DIAG_BLOCK_NUM, src_step=(0, 0), dst_step=(16, 16)))
        self.packed64_diag_l12l0 = make_copy_engine(block_mapping=CopyBlockMapping((16, 16), repeats=NEUMANN_DIAG_BLOCK_NUM, src_step=(16, 16), dst_step=(16, 16)))
        self.fixpipe_engine = make_copy_engine(split_axis=0)
        self.fixpipe_l0c2packed_l1 = make_copy_engine()

    def load_k_decayed(self, k_decayed_l1):
        mem_copy(self.l0a_db.produce(), k_decayed_l1)

    def load_q_decayed(self, q_decayed_l1):
        mem_copy(self.l0a_db.produce(), q_decayed_l1)

    def load_k_inv(self, k_inv_l1):
        mem_copy(self.l0b_db.produce(), k_inv_l1)

    def mm_kkt(self):
        """KKT = K_decayed @ K_inv^T；K_inv 的 L0B slot 留给 Mqk 复用。"""
        k_inv = self.l0b_db.consume()
        matmul(self.l0c_db.produce(), self.l0a_db.consume(), k_inv, init=True)
        return k_inv

    def mm_mqk(self, k_inv_l0b):
        """Mqk = Q_decayed @ K_inv^T，消费 mm_kkt 保留的 K_inv slot。"""
        matmul(self.l0c_db.produce(), self.l0a_db.consume(), k_inv_l0b, init=True)

    def load_per_d_left_k(self, l1_channel):
        mem_copy(self.per_d_left_k_l0a, l1_channel.consume())

    def load_per_d_left_q(self, l1_channel):
        mem_copy(self.per_d_left_q_l0a, l1_channel.consume())

    def load_per_d_right_k(self, l1_channel):
        right = self.per_d_l0b_db.produce()
        mem_copy(right, l1_channel.consume())
        return right

    def mm_per_d_kkt(self, row_block: int, col_block: int, right):
        kkt_tile = tile_slice(
            self.raw_l0c_main,
            (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE),
            (row_block, col_block),
        )
        matmul(kkt_tile, self.per_d_left_k_l0a, right, init=True)

    def mm_per_d_mqk(self, row_block: int, col_block: int, right):
        mqk_tile = tile_slice(
            self.raw_l0c_aux,
            (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE),
            (row_block, col_block),
        )
        matmul(mqk_tile, self.per_d_left_q_l0a, right, init=True)

    def finish_per_d_kkt(self, ub_kkt):
        mem_copy(
            ub_kkt,
            self.raw_l0c_main,
            engine=self.fixpipe_engine,
        )

    def finish_per_d_mqk(self, ub_mqk):
        mem_copy(
            ub_mqk.produce(),
            self.raw_l0c_aux,
            engine=self.fixpipe_engine,
        )

    def store_l0c_to_gm(self, gm_tile):
        mem_copy(gm_tile, self.l0c_db.consume())

    def packed64_block(self, l1_matrix, row_block: int, col_block: int):
        # l1_matrix 是 64x64 NZ compact 矩阵。逻辑 16x16 子块按 packed64 block offset 定位。
        return None, l1_matrix, row_block, col_block

    def _packed64_block_offset_elems(self, row_block: int, col_block: int):
        return (col_block * CHUNK_SIZE * NEUMANN_BLOCK_SIZE
                + row_block * NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE)

    def l0_matrix_block_offset_elems(self, row_block: int, col_block: int):
        return (row_block * NEUMANN_DIAG_BLOCK_NUM + col_block) * NEUMANN_FRACTAL_ELEMS

    def c64_diag_nz_offset_elems(self, diag_block: int):
        return diag_block * (NEUMANN_DIAG_BLOCK_NUM + 1) * NEUMANN_FRACTAL_ELEMS

    def c64_odd_even_pair_dst_gap_fractals(self):
        return (self.c64_diag_nz_offset_elems(2) // NEUMANN_FRACTAL_ELEMS) - 1

    def _raw_l0a_block(self, l0a_slot, row_block: int, col_block: int):
        return reinterpret(l0a_slot, shape=(NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE),
                            offset=self.l0_matrix_block_offset_elems(row_block, col_block) * 2,)

    def _raw_l0b_block(self, l0b_slot, row_block: int, col_block: int):
        return reinterpret(l0b_slot, shape=(NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE),
                            offset=self.l0_matrix_block_offset_elems(row_block, col_block) * 2,)

    def _load_l1_block_to_l0a_block(
        self, l1_block, l0a_slot, row_block: int, col_block: int,
    ):
        """把一个 packed L1 16x16 fractal 装入 full64 L0A 的指定块位置。"""
        _, l1_parent, _, _ = l1_block
        l0_tile = self._raw_l0a_block(l0a_slot, row_block, col_block)
        l1_tile = reinterpret(
            l1_parent,
            shape=(NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE),
            offset=(col_block * CHUNK_SIZE * NEUMANN_BLOCK_SIZE + row_block * NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE)
            * 2,
        )
        mem_copy(l0_tile, l1_tile)

    def _load_l1_block_to_l0b_block(
        self, l1_block, l0b_slot, row_block: int, col_block: int,
    ):
        """把一个 packed L1 16x16 fractal 转置装入 full64 L0B 的指定块位置。"""
        _, l1_parent, _, _ = l1_block
        l0_tile = self._raw_l0b_block(l0b_slot, row_block, col_block)
        l1_tile = reinterpret(
            l1_parent,
            shape=(NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE),
            offset=(col_block * CHUNK_SIZE * NEUMANN_BLOCK_SIZE + row_block * NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE)
            * 2,
        )
        mem_copy(l0_tile, l1_tile, transpose=True)

    def _load_packed32_block_to_full64_l0b_raw(
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
        for row_offset in (0, NEUMANN_BLOCK_SIZE):
            l1_offset = (src_col // NEUMANN_BLOCK_SIZE) * src_packed_size * NEUMANN_BLOCK_SIZE
            l1_offset += (src_row + row_offset) * NEUMANN_BLOCK_SIZE + (src_col % NEUMANN_BLOCK_SIZE)
            l0_offset = self.l0_matrix_block_offset_elems(
                (dst_row + row_offset) // NEUMANN_BLOCK_SIZE, dst_col // NEUMANN_BLOCK_SIZE
            )
            l0_tile = reinterpret(l0b_slot, shape=(NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE), offset=l0_offset * 2)
            l1_tile = reinterpret(l1_matrix, shape=(NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE), offset=l1_offset * 2)
            mem_copy(l0_tile, l1_tile, transpose=transpose)

    def _load_packed32_block_to_full64_l0a_raw(
        self,
        l1_matrix,
        l0a_slot,
        *,
        src_row: int,
        src_col: int,
        dst_row: int,
        dst_col: int,
        src_packed_size: int = CHUNK_SIZE,
        transpose: bool = False,
    ):
        """把 packed32 L1 block 装入 full64 L0A。"""
        for row_offset in (0, NEUMANN_BLOCK_SIZE):
            l1_offset = (src_col // NEUMANN_BLOCK_SIZE) * src_packed_size * NEUMANN_BLOCK_SIZE
            l1_offset += (src_row + row_offset) * NEUMANN_BLOCK_SIZE + (src_col % NEUMANN_BLOCK_SIZE)
            l0_offset = self.l0_matrix_block_offset_elems(
                (dst_row + row_offset) // NEUMANN_BLOCK_SIZE, dst_col // NEUMANN_BLOCK_SIZE
            )
            l0_offset -= 6 * NEUMANN_FRACTAL_ELEMS
            l0_tile = reinterpret(l0a_slot, shape=(NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE), offset=l0_offset * 2)
            l1_tile = reinterpret(l1_matrix, shape=(NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE), offset=l1_offset * 2)
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
        dst0_block = dst_offset_elems // NEUMANN_FRACTAL_ELEMS
        dst1_block = dst0_block + dst_gap_fractals + 1
        for src_row_block, src_col_block, dst_block in (
            (src0_row_block, src0_col_block, dst0_block),
            (src1_row_block, src1_col_block, dst1_block),
        ):
            src_offset = self._packed64_block_offset_elems(src_row_block, src_col_block)
            # 3510 L0A 的 16x16 单块地址仍需补偿一个 full64 M span。
            dst_offset = (dst_block - (NEUMANN_DIAG_BLOCK_NUM - 1)) * NEUMANN_FRACTAL_ELEMS
            l0_tile = reinterpret(l0a_slot, shape=(NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), offset=dst_offset * 2)
            l1_tile = reinterpret(l1_matrix, shape=(NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), offset=src_offset * 2)
            mem_copy(l0_tile, l1_tile)

    def _load_packed16_pair_to_full64_l0b_trans_with_gap(
        self,
        l1_matrix,
        l0b_slot,
        *,
        src0_row_block: int,
        src0_col_block: int,
        src1_row_block: int,
        src1_col_block: int,
        dst_offset_elems: int,
        dst_gap_fractals: int,
    ):
        """把一对 packed16 L1 block 转置装入 full64 L0B，目标块之间保留 gap。"""
        dst0_block = dst_offset_elems // NEUMANN_FRACTAL_ELEMS
        dst1_block = dst0_block + dst_gap_fractals + 1
        self._load_l1_block_to_l0b_block(
            self.packed64_block(l1_matrix, src0_row_block, src0_col_block),
            l0b_slot,
            dst0_block // NEUMANN_DIAG_BLOCK_NUM,
            dst0_block % NEUMANN_DIAG_BLOCK_NUM,
        )
        self._load_l1_block_to_l0b_block(
            self.packed64_block(l1_matrix, src1_row_block, src1_col_block),
            l0b_slot,
            dst1_block // NEUMANN_DIAG_BLOCK_NUM,
            dst1_block % NEUMANN_DIAG_BLOCK_NUM,
        )

    def zero_half_l1(self, l1):
        cube_fill_l1_zero(l1, repeat=256, blk_num=1, dst_gap=0)

    def zero_packed64_l0_operands(self, l0a_slot, l0b_slot):
        mem_copy(l0a_slot, self.zero_l1)
        mem_copy(l0b_slot, self.zero_l1)

    def load_packed64_diag_only_operands_to_l0(
        self, left_l1, right_l1, l0a_slot, l0b_slot, *, clear_l0: bool = True,
    ):
        if clear_l0:
            self.zero_packed64_l0_operands(l0a_slot, l0b_slot)
        mem_copy(
            l0a_slot,
            left_l1,
            engine=self.packed64_diag_l12l0,
        )
        mem_copy(l0b_slot, right_l1, engine=make_copy_engine(block_mapping=CopyBlockMapping((16, 16), repeats=NEUMANN_DIAG_BLOCK_NUM, src_step=(16, 16), dst_step=(16, 16)), transpose=True))

    def issue_packed64_diag_only_operands_to_l0_db(
        self, left_l1, right_l1, *, clear_l0: bool = True,
    ):
        l0a_write = self.raw_l0a_db.produce()
        l0b_write = self.raw_l0b_db.produce()
        self.load_packed64_diag_only_operands_to_l0(left_l1, right_l1, l0a_write, l0b_write, clear_l0=clear_l0)

    def begin_mmad_raw_l0ab_db_packed64(
        self, c_tile, *, init_c: bool, unit_flag: int,
    ):
        """发起 full64 raw MMAD，显式指定 init/accumulate。"""
        l0a_read = self.raw_l0a_db.consume()
        l0b_read = self.raw_l0b_db.consume()
        matmul(c_tile, l0a_read, l0b_read, init=init_c, unit_flag=unit_flag)

    def raw_mmad_packed64_diag_only(
        self, left_l1, right_l1, c_tile, *, init_c: bool, unit_flag: int, clear_l0: bool = True
    ):
        self.issue_packed64_diag_only_operands_to_l0_db(left_l1, right_l1, clear_l0=clear_l0)
        self.begin_mmad_raw_l0ab_db_packed64(c_tile, init_c=init_c, unit_flag=unit_flag)

    def raw_mmad_packed64_diag_only_db_pair(
        self,
        left_l10,
        right_l10,
        left_l11,
        right_l11,
        c_tile,
        *,
        init_c0: bool,
        unit_flag0: int,
        init_c1: bool,
        unit_flag1: int,
        clear_l0: bool = True,
    ):
        self.issue_packed64_diag_only_operands_to_l0_db(left_l10, right_l10, clear_l0=clear_l0)
        self.issue_packed64_diag_only_operands_to_l0_db(left_l11, right_l11, clear_l0=clear_l0)
        self.begin_mmad_raw_l0ab_db_packed64(c_tile, init_c=init_c0, unit_flag=unit_flag0)
        self.begin_mmad_raw_l0ab_db_packed64(c_tile, init_c=init_c1, unit_flag=unit_flag1)

    def fixpipe_packed64_full_to_half_l1(self, dst_channel, c_tile,):
        mem_copy(dst_channel, c_tile, engine=self.fixpipe_l0c2packed_l1, unit_flag=3)

    def fixpipe_packed64_full_to_bf16_l1(
        self,
        dst_channel,
        c_tile,
    ):
        mem_copy(dst_channel, c_tile, engine=self.fixpipe_l0c2packed_l1, unit_flag=3)

    def half_l1_power_packed64_diag_only_on_c(
        self, left_l1, right_l1, dst_channel, *, clear_l0: bool,
    ):
        c_power = self.raw_l0c_aux
        self.raw_mmad_packed64_diag_only(left_l1, right_l1, c_power, init_c=True, unit_flag=3, clear_l0=clear_l0)
        c_power_read = self.raw_l0c_aux
        self.fixpipe_packed64_full_to_half_l1(dst_channel, c_power_read)

    def prepare_packed64_identity_l1(self, gm_identity16):
        """构造 packed 64x64 identity。"""
        identity_write = self.packed_identity_l1
        self.zero_half_l1(identity_write)
        cube_sync_pipe(PIPE.MTE2)
        mem_copy(identity_write, gm_identity16, engine=self.gm2l1_identity16_diag)
        return self.packed_identity_l1

    def issue_negative_t_identity_operands(self, negative_t, identity):
        l0a_write = self.raw_l0a_db.produce()
        l0b_write = self.raw_l0b_db.produce()
        mem_copy(l0a_write, negative_t)
        mem_copy(l0b_write, identity)

    def issue_identity_identity_operands(self, identity):
        l0a_write = self.raw_l0a_db.produce()
        l0b_write = self.raw_l0b_db.produce()
        mem_copy(l0a_write, identity)
        mem_copy(l0b_write, identity)

    def prepare_inv_init_from_negative_t_cube(self, negative_t, identity):
        """Cube 侧在 L0C 构造 scratch_l1 = (-T) @ I + I @ I。"""
        self.issue_negative_t_identity_operands(negative_t, identity)
        self.issue_identity_identity_operands(identity)
        c_inv_init = self.raw_l0c_main
        self.begin_mmad_raw_l0ab_db_packed64(c_inv_init, init_c=True, unit_flag=2)
        self.begin_mmad_raw_l0ab_db_packed64(c_inv_init, init_c=False, unit_flag=3)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.scratch_l1, c_inv_read)
        return self.scratch_l1

    def load_packed64_diagonal_inputs_from_l1_handoff(
        self, power_handoff_l1, identity,
    ):
        """消费 AIV `-T` handoff，并在 Cube 侧构造 `I + (-T)`。"""
        negative_t = power_handoff_l1
        scratch = self.prepare_inv_init_from_negative_t_cube(negative_t, identity)
        return negative_t, identity, scratch

    def neumann_diag_power2_update(self, negative_t, identity, scratch):
        self.half_l1_power_packed64_diag_only_on_c(negative_t, negative_t, self.tile_inv_l1, clear_l0=True)
        power2 = self.tile_inv_l1

        c_inv = self.raw_l0c_main
        self.raw_mmad_packed64_diag_only_db_pair(scratch, identity, scratch, power2, c_inv, init_c0=True,
                                                  unit_flag0=2, init_c1=False, unit_flag1=3, clear_l0=False,)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return power2, self.packed_inv64_l1

    def neumann_diag_power4_update(self, power2, packed_inv):
        self.half_l1_power_packed64_diag_only_on_c(power2, power2, self.tile_power_l1, clear_l0=False)
        power4 = self.tile_power_l1

        c_inv = self.raw_l0c_main
        self.raw_mmad_packed64_diag_only(packed_inv, power4, c_inv, init_c=False, unit_flag=3, clear_l0=False)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return power4, self.packed_inv64_l1

    def neumann_diag_power8_update(self, power4, packed_inv):
        self.half_l1_power_packed64_diag_only_on_c(power4, power4, self.tile_inv_l1, clear_l0=False)
        power8 = self.tile_inv_l1

        c_inv = self.raw_l0c_main
        self.raw_mmad_packed64_diag_only(packed_inv, power8, c_inv, init_c=False, unit_flag=3, clear_l0=False)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return self.packed_inv64_l1

    def issue_odd_lower_even_inv_full64(self, scratch, packed_inv):
        l0a_write = self.raw_l0a_db.produce()
        l0b_write = self.raw_l0b_db.produce()
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        self._load_packed16_pair_to_full64_l0a_with_gap(
            scratch,
            l0a_write,
            src0_row_block=1,
            src0_col_block=0,
            src1_row_block=3,
            src1_col_block=2,
            dst_offset_elems=NEUMANN_DIAG_BLOCK_NUM * NEUMANN_FRACTAL_ELEMS,
            dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals(),
        )
        self._load_packed16_pair_to_full64_l0b_trans_with_gap(
            packed_inv,
            l0b_write,
            src0_row_block=0,
            src0_col_block=0,
            src1_row_block=2,
            src1_col_block=2,
            dst_offset_elems=self.c64_diag_nz_offset_elems(0),
            dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals(),
        )

    def issue_odd_inv_odd_tmp_full64(
        self, packed_inv, tmp64,
    ):
        l0a_write = self.raw_l0a_db.produce()
        l0b_write = self.raw_l0b_db.produce()
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        self._load_l1_block_to_l0a_block(self.packed64_block(packed_inv, 1, 1), l0a_write, 1, 1)
        self._load_l1_block_to_l0a_block(self.packed64_block(packed_inv, 3, 3), l0a_write, 3, 3)
        self._load_packed16_pair_to_full64_l0b_trans_with_gap(
            tmp64,
            l0b_write,
            src0_row_block=1,
            src0_col_block=0,
            src1_row_block=3,
            src1_col_block=2,
            dst_offset_elems=NEUMANN_DIAG_BLOCK_NUM * NEUMANN_FRACTAL_ELEMS,
            dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals(),
        )

    def compose_odd_even_lower16_to32_accum_full64(
        self, scratch, packed_inv,
    ):
        """组合两个 32x32 对角子块内部的 lower 逆矩。"""
        self.issue_odd_lower_even_inv_full64(scratch, packed_inv)
        c_tmp = self.raw_l0c_aux
        self.begin_mmad_raw_l0ab_db_packed64(c_tmp, init_c=True, unit_flag=3)
        c_tmp_read = self.raw_l0c_aux
        self.fixpipe_packed64_full_to_half_l1(self.tile_inv_l1, c_tmp_read)
        tmp64 = self.tile_inv_l1

        self.issue_odd_inv_odd_tmp_full64(packed_inv, tmp64)
        c_inv = self.raw_l0c_main
        self.begin_mmad_raw_l0ab_db_packed64(c_inv, init_c=False, unit_flag=3)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return self.packed_inv64_l1

    def issue_odd32_lower_even32_inv_full64(self, scratch, packed_inv):
        l0a_write = self.raw_l0a_db.produce()
        l0b_write = self.raw_l0b_db.produce()
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        mem_copy(l0a_write, scratch)
        self._load_packed32_block_to_full64_l0b_raw(packed_inv, l0b_write, src_row=0, src_col=0, dst_row=0, dst_col=0,
                                                     transpose=True,)

    def prepare_odd32_inv_odd32_tmp_full64(self, packed_inv):
        l0a_write = self.raw_l0a_db.produce()
        l0b_write = self.raw_l0b_db.produce()
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        mem_copy(l0a_write, packed_inv)
        return l0a_write, l0b_write

    def complete_odd32_inv_odd32_tmp_full64(
        self, tmp64, l0a_write, l0b_write,
    ):
        self._load_packed32_block_to_full64_l0b_raw(tmp64, l0b_write, src_row=HALF_CHUNK_SIZE, src_col=0,
                                                     dst_row=HALF_CHUNK_SIZE, dst_col=0, transpose=True,)

    def compose_odd_even_lower32_to64_accum_full64(
        self, scratch, packed_inv,
    ):
        """组合 64x64 左下 32x32 lower 逆矩并生成 BF16 resident inverse。"""
        self.issue_odd32_lower_even32_inv_full64(scratch, packed_inv)
        second_l0a, second_l0b = (
            self.prepare_odd32_inv_odd32_tmp_full64(packed_inv)
        )
        c_tmp = self.raw_l0c_aux
        self.begin_mmad_raw_l0ab_db_packed64(c_tmp, init_c=True, unit_flag=3)
        c_tmp_read = self.raw_l0c_aux
        self.fixpipe_packed64_full_to_half_l1(self.tile_inv_l1, c_tmp_read)
        tmp64 = self.tile_inv_l1

        self.complete_odd32_inv_odd32_tmp_full64(tmp64, second_l0a, second_l0b)
        c_inv = self.raw_l0c_main
        self.begin_mmad_raw_l0ab_db_packed64(c_inv, init_c=False, unit_flag=3)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_bf16_l1(self.resident_inv_l1, c_inv_read)

    def _load_resident_inv_to_l0a(self):
        resident_inv = self.resident_inv_l1
        l0a_write = self.l0a_db.produce()
        mem_copy(reinterpret(l0a_write, shape=(CHUNK_SIZE, CHUNK_SIZE), offset=0), resident_inv)
        return l0a_write

    def _load_beta_to_l0b(self, beta_channel):
        beta_read = beta_channel
        l0b_write = self.l0b_db.produce()
        mem_copy(l0b_write, beta_read, transpose=True)
        return l0b_write

    def _mm_resident_inv_beta(self, l0a_resident, l0b_beta):
        l0c_write = self.l0c_db.produce()
        matmul(l0c_write, reinterpret(l0a_resident, shape=(CHUNK_SIZE, CHUNK_SIZE), offset=0), l0b_beta, init=True)

    def compute_u_pre_and_w_from_resident_inv_beta(
        self,
        gm_U_pre_scratch,
        gm_W_scratch,
        linear_idx,
        v_beta_l1,
        k_decayed_beta_l1,
    ):
        # β·K 先产出，故先算 W；W 的 MMAD/store 与后续 β·V handoff 重叠。
        l0a_resident = self._load_resident_inv_to_l0a()

        l0b_beta = self._load_beta_to_l0b(k_decayed_beta_l1)
        self._mm_resident_inv_beta(l0a_resident, l0b_beta)
        gm_W_chunk = tile_slice(gm_W_scratch, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (linear_idx, 0))
        self.store_l0c_to_gm(gm_W_chunk)

        l0b_beta = self._load_beta_to_l0b(v_beta_l1)
        self._mm_resident_inv_beta(l0a_resident, l0b_beta)
        u_pre_part = tile_slice(gm_U_pre_scratch, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (linear_idx, 0))
        self.store_l0c_to_gm(u_pre_part)


@jit
def _copy_padded_normalized_k(dst, src):
    """Snapshot all initialized rows, including the zero-padded tail.

    Raw stores initialize padding without widening the earlier DMA's valid
    extent. An explicit full-tile view includes padding needed by delayed work.
    """
    full_tile = reinterpret(src, shape=(CHUNK_SIZE, D_HALF), stride=(D_HALF, 1), offset=0)
    mem_copy(dst, full_tile)


class StageOneVector:
    """Vector 侧标量、mask、搬运和状态更新封装。"""

    def __init__(self, *, scale_value, subblock_idx):
        self.scale_value = scale_value
        self.subblock_idx = subblock_idx
        self.ub_q = [Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16) for _ in range(2)]
        self.ub_k = [Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16) for _ in range(2)]
        self.ub_qk_exchange = Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.float32)
        self.ub_g_raw = Channel(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16, depth=1).produce()
        self.ub_gate_activated = Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.float32)
        self.ub_dt_bias = Channel(MemLoc.UB, (D_HALF,), dtypes.float32, depth=1).produce()
        self.ub_alpha = Channel(MemLoc.UB, (1,), dtypes.float32, depth=1).produce()
        self.ub_beta_raw = Channel(MemLoc.UB, (CHUNK_SIZE, 1), dtypes.bfloat16, depth=1).produce()
        self.ub_beta = Buffer(MemLoc.UB, (CHUNK_SIZE, 1), dtypes.float32)
        self.ub_v_raw = Channel(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16, depth=1).produce()
        self.ub_gamma_c = Channel(MemLoc.UB, (1, D_HALF), dtypes.float32, depth=1).produce()
        self.ub_cs_snapshot = [Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.float32) for _ in range(2)]
        self.ub_mqk = Channel(
            MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.float32, depth=2, kind=ChannelKind.CrossCore
        )
        self.ub_bf16_nz = Channel(
            MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16, depth=2, data_format="nz", n1_pad=16
        )
        self.ub_cast_bf16 = Channel(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16, depth=1, data_format="nd").produce()
        self.ub_kkt_fp16 = Buffer(MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.float16)
        self.ub_mqk_bf16 = Channel(MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.bfloat16, depth=1, data_format="nd").produce()
        self.ub_raw_k = [Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16) for _ in range(2)]
        self.ub_kkt = Channel(MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.float32, depth=1, kind=ChannelKind.CrossCore).produce()
        self.ub_kkt_nz = Channel(MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1, data_format="nz", n1_pad=16).produce()
        self.ub_per_d_nd = Channel(MemLoc.UB, (NEUMANN_BLOCK_SIZE, D_HALF), dtypes.bfloat16, depth=2)
        self.ub_per_d_nz = Channel(
            MemLoc.UB, (NEUMANN_BLOCK_SIZE, D_HALF), dtypes.bfloat16, depth=2, data_format="nz", n1_pad=16
        )
        # One FIFO carries target-scoped left K/Q and column-scoped right K.
        # Its three original L1 slots let AIV prepare left Q while AIC consumes
        # the first right K; Cube keeps left K/Q resident in distinct L0A roots.
        self.l1_per_d_operand = Channel(
            MemLoc.L1, (NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=3, kind=ChannelKind.CrossCore
        )
        self.l1_v_beta = Channel(
            MemLoc.L1, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=1,
            kind=ChannelKind.CrossCore,
        ).produce()
        self.l1_k_decayed_beta_for_w = Channel(MemLoc.L1, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16,
                                                depth=1, kind=ChannelKind.CrossCore,).produce()
        self.ub2l1 = make_copy_engine(format_transform='nd2nz')
        self.per_d_ub2l1 = make_copy_engine(format_transform='nd2nz')
        self.fp16_ub2l1 = make_copy_engine(format_transform='nd2nz')

    def scratch_half_tile(self, gm_scratch, block_idx, cols: int):
        return tile_slice(gm_scratch, (HALF_CHUNK_SIZE, cols), (block_idx * 2 + self.subblock_idx, 0))

    def scratch_dhalf_tile(self, gm_scratch, block_idx):
        return tile_slice(gm_scratch, (CHUNK_SIZE, D_HALF), (block_idx, self.subblock_idx))

    @jit
    def publish_qk_partials(self, exchange, outgoing_partials):
        """Publish this AIV's Q/K partial sums before unrelated gate work."""
        vec_sync_notify(PIPE.V, PIPE.MTE3, QK_V_TO_MTE3_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.MTE3, QK_V_TO_MTE3_EVENT_ID)
        mem_copy(exchange, outgoing_partials)
        vec_sync_block_arrive(PIPE.MTE3, QK_PUBLISH_FLAG_ID, mode=1)

    @jit
    def issue_qk_partials(self, exchange, incoming_partials):
        """Issue the peer's Q/K partial sums before independent gate work."""
        vec_sync_block_wait(PIPE.MTE2, QK_PUBLISH_FLAG_ID, mode=1)
        mem_copy(incoming_partials, exchange)
        vec_sync_notify(PIPE.MTE2, PIPE.V, QK_MTE2_TO_V_EVENT_ID)

    @jit
    def finish_issued_qk_partials(self):
        """Wait for peer partial sums and complete the cross-AIV handshake."""
        vec_sync_wait(PIPE.MTE2, PIPE.V, QK_MTE2_TO_V_EVENT_ID)
        vec_sync_block_arrive(PIPE.MTE2, QK_CONSUME_FLAG_ID, mode=1)
        vec_sync_block_wait(PIPE.MTE3, QK_CONSUME_FLAG_ID, mode=1)

    @jit
    def compute_qk_partial_sums(self, q_half, k_half, partials):
        """Reduce this AIV's Q/K halves to one FP32 sum per token row."""
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                q_unpacked = vload_unpack(q_half, offset, unpack_mode=UnpackMode.B16_TO_B32)
                q_value = vcast(q_unpacked, dtypes.float32, mask=lane_mask)
                k_unpacked = vload_unpack(k_half, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k_value = vcast(k_unpacked, dtypes.float32, mask=lane_mask)
                q_square = vmul(q_value, q_value, mask=lane_mask)
                q_partial = vreduce_sum(q_square, mask=lane_mask)
                k_square = vmul(k_value, k_value, mask=lane_mask)
                k_partial = vreduce_sum(k_square, mask=lane_mask)
                vstore_first(partials, row, q_partial)
                vstore_first(partials, CHUNK_SIZE + row, k_partial)
            vmem_bar("vst_vld")

    @jit
    def compute_qk_inverse_vectors(self, local_partials, peer_partials, inverse_norms):
        """Compute all 64 Q/K inverse norms as two vector operations."""
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            one = vdups(1.0, dtypes.float32, mask=lane_mask)
            q_local = vload(local_partials, 0)
            q_peer = vload(peer_partials, 0)
            q_square_sum = vadd(q_local, q_peer, mask=lane_mask)
            q_shifted = vadds(q_square_sum, 1.0e-6, mask=lane_mask)
            q_root = vsqrt(q_shifted, mask=lane_mask)
            q_inverse = vdiv(one, q_root, mask=lane_mask)
            vstore(inverse_norms, 0, q_inverse, lane_mask)
            k_local = vload(local_partials, CHUNK_SIZE)
            k_peer = vload(peer_partials, CHUNK_SIZE)
            k_square_sum = vadd(k_local, k_peer, mask=lane_mask)
            k_shifted = vadds(k_square_sum, 1.0e-6, mask=lane_mask)
            k_root = vsqrt(k_shifted, mask=lane_mask)
            k_inverse = vdiv(one, k_root, mask=lane_mask)
            vstore(inverse_norms, CHUNK_SIZE, k_inverse, lane_mask)
            vmem_bar("vst_vld")

    @jit
    def normalize_local_qk_from_inverse_vectors(self, q_half, k_half, inverse_norms):
        """Normalize both local BF16 halves from packed Q/K inverse vectors."""
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                q_unpacked = vload_unpack(q_half, offset, unpack_mode=UnpackMode.B16_TO_B32)
                q_value = vcast(q_unpacked, dtypes.float32, mask=lane_mask)
                q_inverse = vload_broadcast(inverse_norms, row)
                q_normalized = vmul(q_value, q_inverse, mask=lane_mask)
                q_bf16 = vcast(q_normalized, dtypes.bfloat16, mask=lane_mask)
                vstore_pack(q_half, offset, q_bf16, lane_mask, pack_mode=PackMode.B32_TO_B16)
                k_unpacked = vload_unpack(k_half, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k_value = vcast(k_unpacked, dtypes.float32, mask=lane_mask)
                k_inverse = vload_broadcast(inverse_norms, CHUNK_SIZE + row)
                k_normalized = vmul(k_value, k_inverse, mask=lane_mask)
                k_bf16 = vcast(k_normalized, dtypes.bfloat16, mask=lane_mask)
                vstore_pack(k_half, offset, k_bf16, lane_mask, pack_mode=PackMode.B32_TO_B16)
            vmem_bar("vst_vld")

    @jit
    def prefetch_qk_local_halves(
        self, gm_Q, gm_K, batch_idx, qk_head_idx, chunk_idx, valid_rows, buffer_parity: int,
    ):
        """Prefetch this AIV's Q/K halves without publishing an event yet."""
        q_local_half = tile_slice(gm_Q[batch_idx, qk_head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))
        k_local_half = tile_slice(gm_K[batch_idx, qk_head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))
        q_local_half = _valid_rows_view(q_local_half, valid_rows, D_HALF)
        k_local_half = _valid_rows_view(k_local_half, valid_rows, D_HALF)
        mem_copy(self.ub_q[buffer_parity], q_local_half)
        mem_copy(self.ub_k[buffer_parity], k_local_half)

    @jit
    def notify_qk_local_halves_ready(self):
        if self.subblock_idx == 0:
            vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        else:
            vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_SUBBLOCK1_EVENT_ID)

    @jit
    def issue_qk_local_halves(
        self, gm_Q, gm_K, batch_idx, qk_head_idx, chunk_idx, valid_rows, buffer_parity: int,
    ):
        self.prefetch_qk_local_halves(
            gm_Q, gm_K, batch_idx, qk_head_idx, chunk_idx, valid_rows, buffer_parity,
        )
        self.notify_qk_local_halves_ready()

    @jit
    def finish_and_publish_qk_partials(
        self, gm_qk_exchange, block_idx, valid_rows, buffer_parity: int,
    ):
        """Reduce the local halves and publish both partial-sum vectors."""
        if self.subblock_idx == 0:
            vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        else:
            vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_SUBBLOCK1_EVENT_ID)
        q_half = self.ub_q[buffer_parity]
        k_half = self.ub_k[buffer_parity]
        if valid_rows < CHUNK_SIZE:
            self._fill_bf16_invalid_rows(q_half, valid_rows=valid_rows, cols=D_HALF, value=0.0)
            self._fill_bf16_invalid_rows(k_half, valid_rows=valid_rows, cols=D_HALF, value=0.0)
        local_partials = reinterpret(self.ub_qk_exchange, shape=(2, CHUNK_SIZE), offset=0)
        self.compute_qk_partial_sums(q_half, k_half, local_partials)
        if self.subblock_idx == 0:
            self.publish_qk_partials(
                gm_qk_exchange[block_idx, 0, None, None], local_partials,
            )
        else:
            self.publish_qk_partials(
                gm_qk_exchange[block_idx, 1, None, None], local_partials,
            )

    @jit
    def issue_peer_qk_partials(self, gm_qk_exchange, block_idx):
        """Start receiving both partial-sum vectors from the peer AIV."""
        incoming_partials = reinterpret(
            self.ub_qk_exchange,
            shape=(2, CHUNK_SIZE),
            offset=2 * CHUNK_SIZE * ELEM_BYTES,
        )
        if self.subblock_idx == 0:
            self.issue_qk_partials(
                gm_qk_exchange[block_idx, 1, None, None], incoming_partials,
            )
        else:
            self.issue_qk_partials(
                gm_qk_exchange[block_idx, 0, None, None], incoming_partials,
            )

    @jit
    def finish_qk_partial_exchange(self, buffer_parity: int):
        """Finish the one-round exchange and normalize both local halves."""
        self.finish_issued_qk_partials()
        local_partials = self.ub_qk_exchange
        peer_partials = reinterpret(
            self.ub_qk_exchange,
            shape=(2, CHUNK_SIZE),
            offset=2 * CHUNK_SIZE * ELEM_BYTES,
        )
        inverse_norms = reinterpret(
            self.ub_qk_exchange,
            shape=(2, CHUNK_SIZE),
            offset=4 * CHUNK_SIZE * ELEM_BYTES,
        )
        q_half = self.ub_q[buffer_parity]
        k_half = self.ub_k[buffer_parity]
        self.compute_qk_inverse_vectors(local_partials, peer_partials, inverse_norms)
        self.normalize_local_qk_from_inverse_vectors(q_half, k_half, inverse_norms)
        return k_half, q_half

    @jit
    def _fill_bf16_invalid_rows(self, ub_tile, *, valid_rows, cols: int, value: float):
        """用真实 BF16 常量初始化 [valid_rows, 64) 的无效行。"""
        with vf(mode="simd"):
            lane_mask, _ = update_mask(cols, elem_bits=16)
            fill_bf16 = vdups(value, dtypes.bfloat16, mask=lane_mask)
            for row in dsl_range(valid_rows, CHUNK_SIZE, 1, unroll=4):
                vstore(ub_tile, row * cols, fill_bf16, lane_mask)
            vmem_bar("vst_vld")

    @jit
    def issue_raw_gate(self, gm_g, gm_a_log, gm_dt_bias, batch_idx, head_idx, chunk_idx, valid_rows):
        """Issue raw gate inputs before Q/K normalization and exchange."""
        g_chunk = tile_slice(gm_g[batch_idx, head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))
        g_chunk = _valid_rows_view(g_chunk, valid_rows, D_HALF)
        bias_half = tile_slice(gm_dt_bias[head_idx, None], (D_HALF,), (self.subblock_idx,))
        alpha = tile_slice(gm_a_log, (1,), (head_idx,))
        mem_copy(self.ub_g_raw, g_chunk)
        mem_copy(self.ub_dt_bias, bias_half)
        mem_copy(self.ub_alpha, alpha)
        vec_sync_notify(PIPE.MTE2, PIPE.V, GATE_MTE2_TO_V_EVENT_ID)

    @jit
    def activate_issued_gate(self, lower_bound, valid_rows, buffer_parity: int):
        """Wait for issued gate inputs, activate them, then take cumsum."""
        vec_sync_wait(PIPE.MTE2, PIPE.V, GATE_MTE2_TO_V_EVENT_ID)
        g_slot = self.ub_g_raw
        rows_to_activate = CHUNK_SIZE
        if valid_rows < CHUNK_SIZE:
            # gate 只计算有效行；将 [valid_rows, 64) 清零后送入 cumsum，
            # 使无效行成为尾块的身份行。
            with vf(mode="simd"):
                lane_mask, _ = update_mask(D_HALF, elem_bits=32)
                zero = vdups(0.0, dtypes.float32, mask=lane_mask)
                for row in dsl_range(valid_rows, CHUNK_SIZE, 1, unroll=4):
                    vstore(self.ub_gate_activated, row * D_HALF, zero, lane_mask)
                vmem_bar("vst_vld")
            rows_to_activate = valid_rows
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            bound = vdups(lower_bound, dtypes.float32, mask=lane_mask)
            one = vdups(1.0, dtypes.float32, mask=lane_mask)
            alpha = vload_broadcast(self.ub_alpha, 0)
            alpha_exp = vexp(alpha, mask=lane_mask)
            neg_exp_a = vmuls(alpha_exp, -1.0, mask=lane_mask)
            for row in dsl_range(0, rows_to_activate, 1, unroll=4):
                offset = row * D_HALF
                unpacked_g = vload_unpack(g_slot, offset, unpack_mode=UnpackMode.B16_TO_B32)
                raw_g = vcast(unpacked_g, dtypes.float32, mask=lane_mask)
                dt_bias = vload(self.ub_dt_bias, 0)
                biased_g = vadd(raw_g, dt_bias, mask=lane_mask)
                gate_logit = vmul(neg_exp_a, biased_g, mask=lane_mask)
                exp_gate_logit = vexp(gate_logit, mask=lane_mask)
                gate_denominator = vadd(one, exp_gate_logit, mask=lane_mask)
                gate = vdiv(bound, gate_denominator, mask=lane_mask)
                vstore(self.ub_gate_activated, offset, gate, lane_mask)
            vmem_bar("vst_vld")
        vec_sync_all()
        # The delayed task consumes its parity slot before the current task
        # reaches this point, so cumsum can write the persistent snapshot
        # directly instead of copying a second 4096-element FP32 tile.
        gate_cumsum_snapshot = self.ub_cs_snapshot[buffer_parity]
        prefix_sum_rows(gate_cumsum_snapshot, self.ub_gate_activated)
        return gate_cumsum_snapshot

    @jit
    def issue_raw_beta(self, gm_beta, batch_idx, head_idx, chunk_idx, valid_rows):
        """Issue one raw BF16 beta-logit chunk before per-D work."""
        beta_chunk = tile_slice(gm_beta[batch_idx, head_idx, None, None], (CHUNK_SIZE, 1), (chunk_idx, 0))
        beta_chunk = _valid_rows_view(beta_chunk, valid_rows, 1)
        mem_copy(self.ub_beta_raw, beta_chunk)
        vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)

    @jit
    def activate_issued_beta(self, valid_rows):
        """Wait for issued beta logits and apply sigmoid."""
        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        beta_slot = self.ub_beta_raw
        with vf(mode="simd"):
            lane_mask, _ = update_mask(VL, elem_bits=32)
            one = vdups(1.0, dtypes.float32, mask=lane_mask)
            unpacked_beta = vload_unpack(beta_slot, 0, unpack_mode=UnpackMode.B16_TO_B32)
            raw_beta = vcast(unpacked_beta, dtypes.float32, mask=lane_mask)
            neg_beta = vmuls(raw_beta, -1.0, mask=lane_mask)
            exp_neg_beta = vexp(neg_beta, mask=lane_mask)
            sigmoid_denominator = vadd(one, exp_neg_beta, mask=lane_mask)
            beta = vdiv(one, sigmoid_denominator, mask=lane_mask)
            if valid_rows < CHUNK_SIZE:
                zero = vdups(0.0, dtypes.float32, mask=lane_mask)
                valid_mask, _ = update_mask(valid_rows, elem_bits=32)
                invalid_mask = mask_xor(lane_mask, valid_mask, exec_mask=lane_mask)
                vstore(self.ub_beta, 0, beta, valid_mask)
                vstore(self.ub_beta, 0, zero, invalid_mask)
            else:
                vstore(self.ub_beta, 0, beta, lane_mask)
            vmem_bar("vst_vld")
        return self.ub_beta

    @jit
    def apply_beta_and_finish_t_for_neumann(self, beta_slot, power_handoff_l1, subblock_base: int, buffer_parity: int):
        row_base = subblock_base * HALF_CHUNK_SIZE   # 本半块全局行起点(0 或 32)
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            zero_reg = vdups(0.0, dtypes.float32, mask=row_mask)
            for row in dsl_range(0, HALF_CHUNK_SIZE, 1, unroll=4):
                row_off = row * CHUNK_SIZE
                t_true = vload(self.ub_kkt, row_off)
                beta_brc = vload_broadcast(beta_slot, row_base + row)        # β[row] 广播到整行(逐行标量)
                t_beta = vmul(t_true, beta_brc, mask=row_mask)               # T[row,:] *= β[row]
                # 严格下三角：全局行 i 保留前 i 列(对角也清)；update_mask(i) 的前 i 个 lane = 保留谓词。
                tril_mask, _ = update_mask(row_base + row, elem_bits=32)
                t_tril = vselect_raw(t_beta, zero_reg, cond_mask=tril_mask)   # lane<i 取 t_beta，其余取 0
                neg_t = vmuls(t_tril, -1.0, mask=row_mask)                    # -T
                neg_t_fp16 = vcast(neg_t, dtypes.float16, mask=row_mask)            # f32 -> fp16
                vstore_pack(self.ub_kkt_fp16, row_off, neg_t_fp16, row_mask, pack_mode="b32_to_b16")
        # 先写完 ND tile，再转换为 NZ；同一 V pipe 的程序顺序保证读到本轮数据。
        nd2nz_tile(self.ub_kkt_nz, self.ub_kkt_fp16)
        self._store_neumann_tmp_to_handoff_l1(power_handoff_l1, subblock_base)

    @jit
    def _finish_beta_weighted_c3_operands(
        self, v_slot, beta_slot, raw_k_snapshot, gate_cumsum_snapshot, valid_rows,
    ):
        cast_write = self.ub_cast_bf16
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                beta = vload_broadcast(beta_slot, row)
                neg_beta = vmuls(beta, -1.0, mask=row_mask)
                cs_value = vload(gate_cumsum_snapshot, offset)
                raw_k = vload_unpack(raw_k_snapshot, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k_value = vcast(raw_k, dtypes.float32, mask=row_mask)
                decay = vexp(cs_value, mask=row_mask)
                k_decayed = vmul(k_value, decay, mask=row_mask)
                k_decayed_beta = vmul(k_decayed, neg_beta, mask=row_mask)
                k_decayed_beta_bf16 = vcast(k_decayed_beta, dtypes.bfloat16, mask=row_mask)
                vstore_pack(cast_write, offset, k_decayed_beta_bf16, row_mask, pack_mode="b32_to_b16")
        cast_read = self.ub_cast_bf16
        nd2nz_tile(self.ub_bf16_nz.produce(), cast_read)
        self._store_dhalf_nz_ub_to_full_l1(self.l1_k_decayed_beta_for_w, self.ub_bf16_nz.consume())

        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_V_STAGING_EVENT_ID)
        if valid_rows < CHUNK_SIZE:
            self._fill_bf16_invalid_rows(v_slot, valid_rows=valid_rows, cols=D_HALF, value=0.0)

        cast_write = self.ub_cast_bf16
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                beta = vload_broadcast(beta_slot, row)
                raw_v = vload_unpack(v_slot, offset, unpack_mode=UnpackMode.B16_TO_B32)
                v_value = vcast(raw_v, dtypes.float32, mask=row_mask)
                v_beta = vmul(v_value, beta, mask=row_mask)
                v_beta_bf16 = vcast(v_beta, dtypes.bfloat16, mask=row_mask)
                vstore_pack(cast_write, offset, v_beta_bf16, row_mask, pack_mode="b32_to_b16")
        cast_read = self.ub_cast_bf16
        nd2nz_tile(self.ub_bf16_nz.produce(), cast_read)
        self._store_dhalf_nz_ub_to_full_l1(self.l1_v_beta, self.ub_bf16_nz.consume())

    @jit
    def issue_beta_weighted_v(self, gm_V, batch_idx, head_idx, chunk_idx, valid_rows):
        """Issue V one task before its C3 vector transform consumes it."""
        v_half = tile_slice(gm_V[batch_idx, head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))
        v_half = _valid_rows_view(v_half, valid_rows, D_HALF)
        mem_copy(self.ub_v_raw, v_half)
        vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_V_STAGING_EVENT_ID)

    @jit
    def prepare_beta_weighted_c3_operands(self, beta_slot, buffer_parity: int, valid_rows):
        """Construct C3 operands after the prior-task V prefetch is ready."""
        raw_k_snapshot = self.ub_raw_k[buffer_parity]
        gate_cumsum_snapshot = self.ub_cs_snapshot[buffer_parity]
        self._finish_beta_weighted_c3_operands(
            self.ub_v_raw, beta_slot, raw_k_snapshot, gate_cumsum_snapshot, valid_rows,
        )

    @jit
    def prepare_normalized_chunk_inputs(
        self, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf, buffer_parity: int
    ):
        raw_k_snapshot = self.ub_raw_k[buffer_parity]
        gate_cumsum_tile = reinterpret(gate_cumsum_dhalf, shape=(CHUNK_SIZE, D_HALF), offset=0)

        _copy_padded_normalized_k(raw_k_snapshot, normalized_k_dhalf)

        cast_write = self.ub_cast_bf16
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                gamma = vload(gate_cumsum_tile, offset)
                exp_value = vexp(gamma, mask=row_mask)
                raw_q = vload_unpack(normalized_q_dhalf, offset, unpack_mode=UnpackMode.B16_TO_B32)
                q_value = vcast(raw_q, dtypes.float32, mask=row_mask)
                q_decayed = vmul(q_value, exp_value, mask=row_mask)
                q_decayed = vmuls(q_decayed, self.scale_value, mask=row_mask)
                q_decayed_bf16 = vcast(q_decayed, dtypes.bfloat16, mask=row_mask)
                vstore_pack(cast_write, offset, q_decayed_bf16, row_mask, pack_mode="b32_to_b16")
        return self.ub_cast_bf16, gate_cumsum_dhalf, normalized_k_dhalf, normalized_q_dhalf

    def _store_dhalf_nz_ub_to_full_l1(self, l1_dst, ub_nz_src):
        mem_copy(l1_dst, ub_nz_src, engine=make_copy_engine(split_axis=1), part_id=self.subblock_idx)

    def _store_neumann_tmp_to_handoff_l1(self, l1_dst, subblock_base: int):
        mem_copy(l1_dst, self.ub_kkt_nz, engine=make_copy_engine(split_axis=0), part_id=subblock_base)

    @jit
    def _store_masked_mqk_to_bf16_gm(self, gm_tile, buffer_parity: int):
        # raw：Mqk(32×64) 取含对角下三角(casual: update_mask(全局行 i+1) 谓词+vselect 零寄存器) → cast bf16 → GM(nd)。2 行 unroll。
        mqk_slot = self.ub_mqk.consume()
        bf16_write = self.ub_mqk_bf16
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            zero_reg = vdups(0.0, dtypes.float32, mask=row_mask)
            for row in dsl_range(0, HALF_CHUNK_SIZE, 1, unroll=4):
                off = row * CHUNK_SIZE
                keep, _ = update_mask(self.subblock_idx * HALF_CHUNK_SIZE + row + 1, elem_bits=32)
                mqk = vload(mqk_slot, off)
                mqk_tril = vselect_raw(mqk, zero_reg, cond_mask=keep)   # 保留 lane<=i（含对角）
                mqk_bf16 = vcast(mqk_tril, dtypes.bfloat16, mask=row_mask)
                vstore_pack(bf16_write, off, mqk_bf16, row_mask, pack_mode="b32_to_b16")
        mem_copy(gm_tile, self.ub_mqk_bf16)

    @jit
    def finish_mqk(self, gm_mqk_scratch, block_idx, subblock_base: int, buffer_parity: int):
        mqk_part = self.scratch_half_tile(gm_mqk_scratch, block_idx, CHUNK_SIZE)
        self._store_masked_mqk_to_bf16_gm(mqk_part, buffer_parity)
        return reinterpret(
            self.ub_cs_snapshot[buffer_parity],
            shape=(1, D_HALF),
            offset=(CHUNK_SIZE - 1) * D_HALF * ELEM_BYTES,
        )

    def store_q_decayed_scratch(self, gm_Q_decayed_scratch, block_idx, q_decayed_slot):
        # Q_decayed 的 bf16 已在 prepare_chunk 的 Q_decay vf 里 cast 好、留在 ub_cast_bf16(buf9)；
        # 本方法紧跟 prepare_chunk、中间无人碰 buf9 → 直接把它搬到 GM，省一次 f32→bf16 cast。
        Q_decayed_slot_part = self.scratch_dhalf_tile(gm_Q_decayed_scratch, block_idx)
        mem_copy(Q_decayed_slot_part, q_decayed_slot)

    @jit
    def store_restored_state_inputs(
        self, gm_K_restored_scratch, gm_gamma_C_scratch, block_idx, gamma_c_slot, buffer_parity: int,
    ):
        raw_k_snapshot = self.ub_raw_k[buffer_parity]
        gate_cumsum_snapshot = self.ub_cs_snapshot[buffer_parity]
        K_restored_dhalf = self.scratch_dhalf_tile(gm_K_restored_scratch, block_idx)
        cast_write = self.ub_cast_bf16
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            cs_last = vload(gamma_c_slot, 0)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                raw_k = vload_unpack(raw_k_snapshot, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k_value = vcast(raw_k, dtypes.float32, mask=row_mask)
                gate_cumsum = vload(gate_cumsum_snapshot, offset)
                restore_scale = vexp_sub(cs_last, gate_cumsum, mask=row_mask)
                restored = vmul(k_value, restore_scale, mask=row_mask)
                restored_bf16 = vcast(restored, dtypes.bfloat16, mask=row_mask)
                vstore_pack(cast_write, offset, restored_bf16, row_mask, pack_mode="b32_to_b16")
        cast_read = self.ub_cast_bf16
        mem_copy(K_restored_dhalf, cast_read)

        # gamma_C can underflow to zero, which is the correct state-decay value; it is never multiplied by K_inv.
        gamma_c_write = self.ub_gamma_c
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            gamma_c = vload(gamma_c_slot, 0)
            gamma_c_exp = vexp(gamma_c, mask=row_mask)
            vstore(gamma_c_write, 0, gamma_c_exp, row_mask)
        gamma_c_gm = tile_slice(gm_gamma_C_scratch, (D_HALF, 1), (block_idx * 2 + self.subblock_idx, 0))
        gamma_c_read = self.ub_gamma_c
        mem_copy(gamma_c_gm, gamma_c_read)

    @jit
    def _write_per_d_full_d_handoff(self, full_channel, nd_slot):
        nz_slot = self.ub_per_d_nz.produce()
        nd2nz_tile(nz_slot, nd_slot)
        mem_copy(full_channel.produce(), nz_slot, engine=make_copy_engine(split_axis=1), part_id=self.subblock_idx)

    @jit
    def build_per_d_left_k_and_decay(self, normalized_k_dhalf, gate_cumsum_dhalf, row_block: int, left_k_l1_operand):
        k_read = tile_slice(normalized_k_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), (row_block, 0))
        decay_scratch = tile_slice(self.ub_qk_exchange, (NEUMANN_BLOCK_SIZE, D_HALF), (0, 0))
        ref_offset = row_block * NEUMANN_BLOCK_SIZE * D_HALF

        nd_write = self.ub_per_d_nd.produce()
        with vf(mode="simd"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            ref = vload(gate_cumsum_dhalf, ref_offset)
            for row in dsl_range(0, NEUMANN_BLOCK_SIZE, 1, unroll=2):
                offset = row * D_HALF
                g_offset = (row_block * NEUMANN_BLOCK_SIZE + row) * D_HALF
                unpacked_k = vload_unpack(k_read, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k = vcast(unpacked_k, dtypes.float32, mask=mask)
                gate_cumsum = vload(gate_cumsum_dhalf, g_offset)
                decay = vexp_sub(gate_cumsum, ref, mask=mask)
                vstore(decay_scratch, offset, decay, mask)
                k_decayed = vmul(k, decay, mask=mask)
                k_decayed_bf16 = vcast(k_decayed, dtypes.bfloat16, mask=mask)
                vstore_pack(nd_write, offset, k_decayed_bf16, mask, pack_mode="b32_to_b16")
            vmem_bar("vst_vld")
        nd_read = self.ub_per_d_nd.consume()
        self._write_per_d_full_d_handoff(left_k_l1_operand, nd_read)

    @jit
    def build_per_d_left_q_from_decay(self, normalized_q_dhalf, row_block: int, left_q_l1_operand):
        q_read = tile_slice(normalized_q_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), (row_block, 0))
        decay_scratch = tile_slice(self.ub_qk_exchange, (NEUMANN_BLOCK_SIZE, D_HALF), (0, 0))
        nd_write = self.ub_per_d_nd.produce()
        with vf(mode="simd"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            for row in dsl_range(0, NEUMANN_BLOCK_SIZE, 1, unroll=2):
                offset = row * D_HALF
                unpacked_q = vload_unpack(q_read, offset, unpack_mode=UnpackMode.B16_TO_B32)
                q = vcast(unpacked_q, dtypes.float32, mask=mask)
                decay = vload(decay_scratch, offset)
                q_decayed = vmul(q, decay, mask=mask)
                scaled_q = vmuls(q_decayed, self.scale_value, mask=mask)
                scaled_q_bf16 = vcast(scaled_q, dtypes.bfloat16, mask=mask)
                vstore_pack(nd_write, offset, scaled_q_bf16, mask, pack_mode="b32_to_b16")
        nd_read = self.ub_per_d_nd.consume()
        self._write_per_d_full_d_handoff(left_q_l1_operand, nd_read)

    @jit
    def build_per_d_right(
        self, normalized_k_dhalf, gate_cumsum_dhalf, row_block: int, col_block: int, right_k_l1_operand
    ):
        k_read = tile_slice(normalized_k_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), (col_block, 0))
        ref_offset = row_block * NEUMANN_BLOCK_SIZE * D_HALF

        nd_write = self.ub_per_d_nd.produce()
        with vf(mode="simd"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            ref = vload(gate_cumsum_dhalf, ref_offset)
            for row in dsl_range(0, NEUMANN_BLOCK_SIZE, 1, unroll=2):
                offset = row * D_HALF
                g_offset = (col_block * NEUMANN_BLOCK_SIZE + row) * D_HALF
                unpacked_k = vload_unpack(k_read, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k = vcast(unpacked_k, dtypes.float32, mask=mask)
                gate_cumsum = vload(gate_cumsum_dhalf, g_offset)
                decay = vexp_sub(ref, gate_cumsum, mask=mask)
                restored_k = vmul(k, decay, mask=mask)
                restored_k_bf16 = vcast(restored_k, dtypes.bfloat16, mask=mask)
                vstore_pack(nd_write, offset, restored_k_bf16, mask, pack_mode="b32_to_b16")
        nd_read = self.ub_per_d_nd.consume()
        self._write_per_d_full_d_handoff(right_k_l1_operand, nd_read)


@jit
def _run_stage1_body(
    tail_group,
    tail_rows,
    gm_K,
    gm_V,
    gm_Q,
    gm_beta_brc,
    gm_g,
    gm_qk_exchange,
    gm_A_log,
    gm_dt_bias,
    gm_q_decayed,
    gm_Mqk,
    gm_k_restored,
    gm_gamma,
    gm_U_pre,
    gm_w,
    gm_identity16,
    workspace_task_base: int,  # Global Stage1 task-slot prefix (not byte/row offset).
    batch: int,          # 兼容位(不再使用；BN 合轴后 head 由 head_base/head_count 表达)
    head_count: int,
    chunks_total: int,
    seq_len: int,
    head_base: int,
    group_start: int,
    group_count: int,
    eff_group: int,
    active_core_num: int,
    task_start: int,
    task_end: int,
    scale_value: float,
    lower_bound: float,
):
    """Chunk Kimi Delta Attention Stage1。

    Stage1 对每个 chunk 计算 K_inv、K_decayed、Q_decayed 和 beta 加权的 KKT 下三角矩阵，
    使用 Neumann 展开求解块内三角系统，生成 Mqk、U_pre、W、K_restored 和 gamma_C。

    计算 [head_base, head_base+head_count) 这些(平铺后)head 一组 chunk 的 StageOne workspace。
    输入保持原始 (B, N, S, D)；用 idx2crd 把平铺 head 反解回 (batch, head) 再 2D 索引（免 reshape）。"""
    subblock_idx = get_subblock_id()
    block_idx = get_block_idx()
    batch_count = gm_K.shape[0]
    value_head_count = gm_V.shape[1]
    key_head_count = gm_K.shape[1]
    heads_per_key_head = value_head_count // key_head_count


    if block_idx < active_core_num:
        # 下三角由 apply_beta / _store_masked_mqk 的 update_mask 谓词生成。
        # matmul + vector 提到循环外(对齐 stage2)。identity 在循环外构造并复用。
        matmul = StageOneMatmul()
        vector = StageOneVector(
            scale_value=scale_value,
            subblock_idx=subblock_idx,
        )
        task_delay_line = DelayLineGroup(2, "idx", "valid_rows")
        # The packed identity is immutable after construction.  Produce it
        # once before the task loop and let the Channel resolver keep its
        # read transaction across all delayed-task consumers.
        identity = matmul.prepare_packed64_identity_l1(gm_identity16)
        # All packed64 L0 clears read this immutable zero tile.
        matmul.zero_half_l1(matmul.zero_l1)
        pipeline_tick = 0
        issued_task_count = 0
        qk_halves_prefetched = False
        for local_task_index in range(task_start, task_end):
            current_local_head_index = local_task_index // group_count
            current_group_chunk_index = (
                local_task_index - current_local_head_index * group_count
            )
            current_chunk_index = group_start + current_group_chunk_index
            current_global_head_index = head_base + current_local_head_index
            current_batch_index, current_value_head_index = idx2crd(
                current_global_head_index, [batch_count, value_head_count]
            )
            current_query_key_head_index = (
                current_value_head_index // heads_per_key_head
            )
            current_valid_rows = CHUNK_SIZE
            if tail_group:
                if current_group_chunk_index + 1 == group_count:
                    current_valid_rows = tail_rows
            buffer_parity = local_task_index % 2

            # Publish current-task MTE2 work before the prior task's vector/cube tail.
            vector.issue_raw_gate(
                gm_g,
                gm_A_log,
                gm_dt_bias,
                current_batch_index,
                current_value_head_index,
                current_chunk_index,
                current_valid_rows,
            )
            if not qk_halves_prefetched:
                vector.issue_qk_local_halves(
                    gm_Q, gm_K, current_batch_index, current_query_key_head_index,
                    current_chunk_index, current_valid_rows, buffer_parity,
                )
            else:
                vector.notify_qk_local_halves_ready()

            if pipeline_tick >= 1 and pipeline_tick - 1 < issued_task_count:
                delayed_task_index = task_delay_line.idx.tap(1)
                local_head_index = delayed_task_index // group_count
                group_chunk_index = (
                    delayed_task_index - local_head_index * group_count
                )
                chunk_index = group_start + group_chunk_index
                global_head_index = head_base + local_head_index
                batch_index, value_head_index = idx2crd(global_head_index, [batch_count, value_head_count])
                buffer_parity = delayed_task_index % 2
                delayed_valid_rows = task_delay_line.valid_rows.tap(1)
                vector.apply_beta_and_finish_t_for_neumann(
                    vector.ub_beta, matmul.neumann_power_handoff_l1, subblock_idx, buffer_parity
                )
                vector.prepare_beta_weighted_c3_operands(vector.ub_beta, buffer_parity, delayed_valid_rows)
                negative_lower_matrix, identity_matrix, neumann_scratch = (
                    matmul.load_packed64_diagonal_inputs_from_l1_handoff(matmul.neumann_power_handoff_l1, identity)
                )
                second_power, packed_inverse = matmul.neumann_diag_power2_update(
                    negative_lower_matrix, identity_matrix, neumann_scratch
                )
                fourth_power, packed_inverse = matmul.neumann_diag_power4_update(second_power, packed_inverse)
                packed_inverse = matmul.neumann_diag_power8_update(fourth_power, packed_inverse)
                packed_inverse = matmul.compose_odd_even_lower16_to32_accum_full64(neumann_scratch, packed_inverse)
                matmul.compose_odd_even_lower32_to64_accum_full64(neumann_scratch, packed_inverse)

            vector.issue_beta_weighted_v(
                gm_V, current_batch_index, current_value_head_index, current_chunk_index, current_valid_rows
            )
            task_delay_line.push(idx=local_task_index, valid_rows=current_valid_rows)
            local_head_index = current_local_head_index
            group_chunk_index = current_group_chunk_index
            chunk_index = current_chunk_index
            global_head_index = current_global_head_index
            batch_index = current_batch_index
            value_head_index = current_value_head_index
            workspace_task_index = (
                workspace_task_base + global_head_index * eff_group + group_chunk_index
            )
            buffer_parity = local_task_index % 2
            vector.finish_and_publish_qk_partials(
                gm_qk_exchange, block_idx, current_valid_rows, buffer_parity,
            )
            vector.issue_peer_qk_partials(gm_qk_exchange, block_idx)
            gate_cumsum_dhalf = vector.activate_issued_gate(lower_bound, current_valid_rows, buffer_parity)
            normalized_k_dhalf, normalized_q_dhalf = vector.finish_qk_partial_exchange(buffer_parity)
            vector.issue_raw_beta(gm_beta_brc, batch_index, value_head_index, chunk_index, current_valid_rows)
            decayed_q_dhalf, gate_cumsum_dhalf, normalized_k_dhalf, normalized_q_dhalf = (
                vector.prepare_normalized_chunk_inputs(
                    normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf, buffer_parity
                )
            )
            vector.store_q_decayed_scratch(gm_q_decayed, workspace_task_index, decayed_q_dhalf)

            # Current normalized halves remain live for C1. Prefetch the next
            # task into the other parity while C1 consumes the current parity.
            if local_task_index + 1 < task_end:
                next_local_task_index = local_task_index + 1
                next_local_head_index = next_local_task_index // group_count
                next_group_chunk_index = next_local_task_index - next_local_head_index * group_count
                next_chunk_index = group_start + next_group_chunk_index
                next_global_head_index = head_base + next_local_head_index
                next_batch_index, next_value_head_index = idx2crd(
                    next_global_head_index, [batch_count, value_head_count]
                )
                next_query_key_head_index = next_value_head_index // heads_per_key_head
                next_valid_rows = CHUNK_SIZE
                if tail_group:
                    if next_group_chunk_index + 1 == group_count:
                        next_valid_rows = tail_rows
                next_buffer_parity = next_local_task_index % 2
                vector.prefetch_qk_local_halves(
                    gm_Q, gm_K, next_batch_index, next_query_key_head_index,
                    next_chunk_index, next_valid_rows, next_buffer_parity,
                )
                qk_halves_prefetched = True
            else:
                qk_halves_prefetched = False

            # Stage1 per-D tiles are causal: only col_block <= row_block contributes.
            # Skip the six upper-triangle (KKT, Mqk) pairs while preserving producer order.
            vector.build_per_d_left_k_and_decay(normalized_k_dhalf, gate_cumsum_dhalf, 0, vector.l1_per_d_operand)
            matmul.load_per_d_left_k(vector.l1_per_d_operand)
            vector.build_per_d_left_q_from_decay(normalized_q_dhalf, 0, vector.l1_per_d_operand)
            matmul.load_per_d_left_q(vector.l1_per_d_operand)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 0, 0, vector.l1_per_d_operand)
            right = matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(0, 0, right)
            matmul.mm_per_d_mqk(0, 0, right)

            vector.build_per_d_left_k_and_decay(normalized_k_dhalf, gate_cumsum_dhalf, 1, vector.l1_per_d_operand)
            matmul.load_per_d_left_k(vector.l1_per_d_operand)
            vector.build_per_d_left_q_from_decay(normalized_q_dhalf, 1, vector.l1_per_d_operand)
            matmul.load_per_d_left_q(vector.l1_per_d_operand)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 1, 0, vector.l1_per_d_operand)
            right = matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(1, 0, right)
            matmul.mm_per_d_mqk(1, 0, right)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 1, 1, vector.l1_per_d_operand)
            right = matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(1, 1, right)
            matmul.mm_per_d_mqk(1, 1, right)

            vector.build_per_d_left_k_and_decay(normalized_k_dhalf, gate_cumsum_dhalf, 2, vector.l1_per_d_operand)
            matmul.load_per_d_left_k(vector.l1_per_d_operand)
            vector.build_per_d_left_q_from_decay(normalized_q_dhalf, 2, vector.l1_per_d_operand)
            matmul.load_per_d_left_q(vector.l1_per_d_operand)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 2, 0, vector.l1_per_d_operand)
            right = matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(2, 0, right)
            matmul.mm_per_d_mqk(2, 0, right)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 2, 1, vector.l1_per_d_operand)
            right = matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(2, 1, right)
            matmul.mm_per_d_mqk(2, 1, right)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 2, 2, vector.l1_per_d_operand)
            right = matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(2, 2, right)
            matmul.mm_per_d_mqk(2, 2, right)

            vector.build_per_d_left_k_and_decay(normalized_k_dhalf, gate_cumsum_dhalf, 3, vector.l1_per_d_operand)
            matmul.load_per_d_left_k(vector.l1_per_d_operand)
            vector.build_per_d_left_q_from_decay(normalized_q_dhalf, 3, vector.l1_per_d_operand)
            matmul.load_per_d_left_q(vector.l1_per_d_operand)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 3, 0, vector.l1_per_d_operand)
            right = matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(3, 0, right)
            matmul.mm_per_d_mqk(3, 0, right)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 3, 1, vector.l1_per_d_operand)
            right = matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(3, 1, right)
            matmul.mm_per_d_mqk(3, 1, right)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 3, 2, vector.l1_per_d_operand)
            right = matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(3, 2, right)
            matmul.mm_per_d_mqk(3, 2, right)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 3, 3, vector.l1_per_d_operand)
            right = matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(3, 3, right)
            matmul.mm_per_d_mqk(3, 3, right)
            matmul.finish_per_d_kkt(vector.ub_kkt)

            vector.activate_issued_beta(current_valid_rows)
            matmul.finish_per_d_mqk(vector.ub_mqk)

            issued_task_count = issued_task_count + 1
            if pipeline_tick >= 1 and pipeline_tick - 1 < issued_task_count:
                delayed_task_index = task_delay_line.idx.tap(1)
                local_head_index = delayed_task_index // group_count
                group_chunk_index = (
                    delayed_task_index - local_head_index * group_count
                )
                workspace_task_index = (
                    head_base + local_head_index
                ) * eff_group + group_chunk_index + workspace_task_base
                buffer_parity = delayed_task_index % 2
                log_gamma_c_dhalf = vector.finish_mqk(gm_Mqk, workspace_task_index, subblock_idx, buffer_parity)
                vector.store_restored_state_inputs(
                    gm_k_restored, gm_gamma, workspace_task_index, log_gamma_c_dhalf, buffer_parity
                )
                matmul.compute_u_pre_and_w_from_resident_inv_beta(
                    gm_U_pre, gm_w, workspace_task_index, vector.l1_v_beta, vector.l1_k_decayed_beta_for_w
                )
            task_delay_line.advance()
            pipeline_tick = pipeline_tick + 1

        if pipeline_tick >= 1 and pipeline_tick - 1 < issued_task_count:
            delayed_task_index = task_delay_line.idx.tap(1)
            local_head_index = delayed_task_index // group_count
            group_chunk_index = delayed_task_index - local_head_index * group_count
            chunk_index = group_start + group_chunk_index
            global_head_index = head_base + local_head_index
            batch_index, value_head_index = idx2crd(global_head_index, [batch_count, value_head_count])
            workspace_task_index = (
                workspace_task_base + global_head_index * eff_group + group_chunk_index
            )
            buffer_parity = delayed_task_index % 2
            delayed_valid_rows = task_delay_line.valid_rows.tap(1)
            vector.apply_beta_and_finish_t_for_neumann(
                vector.ub_beta, matmul.neumann_power_handoff_l1, subblock_idx, buffer_parity
            )
            vector.prepare_beta_weighted_c3_operands(vector.ub_beta, buffer_parity, delayed_valid_rows)
            negative_lower_matrix, identity_matrix, neumann_scratch = (
                matmul.load_packed64_diagonal_inputs_from_l1_handoff(matmul.neumann_power_handoff_l1, identity)
            )
            second_power, packed_inverse = matmul.neumann_diag_power2_update(
                negative_lower_matrix, identity_matrix, neumann_scratch
            )
            fourth_power, packed_inverse = matmul.neumann_diag_power4_update(second_power, packed_inverse)
            packed_inverse = matmul.neumann_diag_power8_update(fourth_power, packed_inverse)
            packed_inverse = matmul.compose_odd_even_lower16_to32_accum_full64(neumann_scratch, packed_inverse)
            matmul.compose_odd_even_lower32_to64_accum_full64(neumann_scratch, packed_inverse)
            log_gamma_c_dhalf = vector.finish_mqk(gm_Mqk, workspace_task_index, subblock_idx, buffer_parity)
            vector.store_restored_state_inputs(
                gm_k_restored, gm_gamma, workspace_task_index, log_gamma_c_dhalf, buffer_parity
            )
            matmul.compute_u_pre_and_w_from_resident_inv_beta(
                gm_U_pre, gm_w, workspace_task_index, vector.l1_v_beta, vector.l1_k_decayed_beta_for_w
            )


class StageTwoMatmul:

    def __init__(self, *, dv_base: int):
        self.dv_base = dv_base
        self.l0a_chunk = Channel(MemLoc.L0A, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=2)
        self.l0a_kr_t = Channel(MemLoc.L0A, shape=(CHUNK_SIZE, dv_base), dtype=dtypes.bfloat16, depth=1, data_format="zn").produce()
        self.l0a_mqk = Channel(MemLoc.L0A, shape=(CHUNK_SIZE, CHUNK_SIZE), dtype=dtypes.bfloat16, depth=1).produce()
        self.l0b_u = Channel(MemLoc.L0B, shape=(CHUNK_SIZE, dv_base), dtype=dtypes.bfloat16, depth=1).produce()
        self.l0b_state = Channel(MemLoc.L0B, shape=(dv_base, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=1).produce()
        self.l0b_kr = Channel(MemLoc.L0B, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=1).produce()
        self.l0c_u = Channel(MemLoc.L0C, shape=(CHUNK_SIZE, dv_base), dtype=dtypes.float32, depth=1).produce()
        self.l0c_o = Channel(MemLoc.L0C, shape=(CHUNK_SIZE, dv_base), dtype=dtypes.float32, depth=1).produce()
        self.l0c_state = Channel(MemLoc.L0C, shape=(dv_base, SUPPORTED_HEAD_DIM), dtype=dtypes.float32, depth=1).produce()
        self.gm2l1 = make_copy_engine(format_transform='nd2nz')
        self.fixpipe_state = make_copy_engine(split_axis=1)
        self.fixpipe_u = make_copy_engine(split_axis=0)

    def load_k_cumdecay(self, k_cumdecay_l1, gm_tile):
        mem_copy(k_cumdecay_l1.produce(), gm_tile, engine=self.gm2l1)

    def load_q_decayed(self, q_decayed_l1, gm_tile):
        mem_copy(q_decayed_l1.produce(), gm_tile, engine=self.gm2l1)

    def load_k_restored(self, k_restored_l1, gm_tile):
        mem_copy(k_restored_l1.produce(), gm_tile, l2_cache_ctl=_L2_CACHE_LAST_USE, engine=self.gm2l1)

    def load_mqk(self, mqk_l1, gm_tile):
        mem_copy(mqk_l1.produce(), gm_tile, l2_cache_ctl=_L2_CACHE_LAST_USE, engine=self.gm2l1)

    def load_state(self, state_l1):
        mem_copy(self.l0b_state, state_l1.consume())

    def compute_w_state(self, w_l1):
        mem_copy(self.l0a_chunk.produce(), w_l1.consume())
        matmul(self.l0c_u, self.l0a_chunk.consume(), self.l0b_state, init=True)

    def compute_q_state(self, q_decayed_l1):
        mem_copy(self.l0a_chunk.produce(), q_decayed_l1.consume())
        matmul(self.l0c_o, self.l0a_chunk.consume(), self.l0b_state, init=True)

    def load_k_restored_operand(self, k_restored_l1):
        mem_copy(self.l0b_kr, k_restored_l1.consume(), transpose=True)

    def load_mqk_operand(self, mqk_l1):
        mem_copy(self.l0a_mqk, mqk_l1.consume())

    def update_state(self, u_slot):
        u_src = reinterpret(u_slot, shape=(CHUNK_SIZE, self.dv_base), offset=0)
        mem_copy(self.l0a_kr_t, u_src, transpose=True)
        matmul(self.l0c_state, self.l0a_kr_t, self.l0b_kr, init=True)

    def accumulate_mqk_u(self, u_slot):
        u_tile = reinterpret(u_slot, shape=(CHUNK_SIZE, self.dv_base), offset=0)
        mem_copy(self.l0b_u, u_tile, transpose=True)
        matmul(self.l0c_o, self.l0a_mqk, self.l0b_u, init=False)

    def store_u_delta(self, ub):
        mem_copy(ub, tile_slice(self.l0c_u, (CHUNK_SIZE, self.dv_base), (0, 0)), engine=self.fixpipe_u)

    def store_output(self, gm_tile):
        """FIXPIPE 按目的 GM Access.actual_rows 写入逻辑尾块。"""
        mem_copy(
            gm_tile,
            self.l0c_o,
            l2_cache_ctl=_L2_CACHE_DISABLE,
        )

    def store_state_delta(self, ub):
        mem_copy(ub, self.l0c_state, engine=self.fixpipe_state)


class StageTwoVector:
    def __init__(self, *, u_row_block: int, dv_base: int):
        self.dv_base = dv_base
        self.u_row_block = u_row_block
        self.half_dk = D_HALF
        self.ub_state_fp32 = Buffer(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.float32)
        self.tmp_state_bf16 = Buffer(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.bfloat16)
        self.tmp_u_bf16 = Buffer(MemLoc.UB, (u_row_block, dv_base), dtypes.bfloat16)
        self.state_nd = Channel(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.bfloat16, depth=1).produce()
        self.state_nz = Channel(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.bfloat16, depth=1, data_format="nz", n1_pad=16).produce()
        self.u_nz = Channel(MemLoc.UB, (u_row_block, dv_base), dtypes.bfloat16, depth=1, data_format="nz", n1_pad=16).produce()
        self.state_in = Channel(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.float32, depth=1).produce()
        self.state_out = Channel(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.float32, depth=1).produce()
        self.u_in = Channel(MemLoc.UB, (u_row_block, dv_base), dtypes.bfloat16, depth=2)
        self.ub2l1 = make_copy_engine(format_transform='nd2nz')

    def load_gamma_col(self, gamma_ub, gamma_col):
        slot = gamma_ub.produce()
        mem_copy(slot, gamma_col)

    def _produce_state_l1(self, state_l1, subblock_idx):
        ub_state = self.ub_state_fp32
        nd_slot = self.state_nd
        cast_tile(nd_slot, ub_state)
        nd_read = self.state_nd
        nz_slot = self.state_nz
        nd2nz_tile(nz_slot, nd_read)
        nz_read = self.state_nz
        slot = state_l1.produce()
        state_l1_rows = tile_slice(slot, (self.dv_base, self.half_dk), (0, subblock_idx))
        mem_copy(state_l1_rows, nz_read)

    def load_initial_state_and_write_l1(self, current_state, state_l1, subblock_idx):
        in_slot = self.state_in
        mem_copy(in_slot, current_state)
        in_read = self.state_in
        ub_state = self.ub_state_fp32
        mem_copy(ub_state, in_read)
        self._produce_state_l1(state_l1, subblock_idx)

    def write_state_l1_snapshot(self, state_l1, subblock_idx):
        """Publish the accumulated state for the next Cube step."""
        self._produce_state_l1(state_l1, subblock_idx)

    @jit
    def scale_and_add_delta(self, tmp_delta, gamma_ub):
        """Fuse state decay and delta accumulation into one full-state VF pass."""
        gamma_read = gamma_ub.consume()
        ub_gamma = gamma_read
        ub_state = self.ub_state_fp32
        delta = tmp_delta
        with vf(mode="simd"):
            mask, _ = update_mask(VL, elem_bits=32)
            gamma_vec = vload(ub_gamma, 0)
            for base in dsl_range(0, self.dv_base * self.half_dk, VL, unroll=4):
                state_v = vload(ub_state, base)
                delta_v = vload(delta, base)
                state_scaled = vmul(state_v, gamma_vec, mask=mask)
                state_next = vadd(state_scaled, delta_v, mask=mask)
                vstore(ub_state, base, state_next, mask)

    def issue_u_rows(self, gm_u_tile, subblock_idx):
        gm_u_rows = tile_slice(gm_u_tile, (self.u_row_block, self.dv_base), (subblock_idx, 0))
        mem_copy(self.u_in.produce(), gm_u_rows)

    @jit
    def finish_u_delta_to_l1(self, u_l1, tmp_u_delta, subblock_idx):
        u_l1_rows = tile_slice(u_l1.produce(), (self.u_row_block, self.dv_base), (subblock_idx, 0))

        # Consume the MTE2 channel directly, without an identity UB pass.
        u_in_read = self.u_in.consume()
        delta = tmp_u_delta
        with vf(mode="simd"):
            mask, _ = update_mask(VL, elem_bits=32)
            for base in range(0, self.u_row_block * self.dv_base, 2 * VL):
                u_raw_pre = vload_unpack(u_in_read, base, unpack_mode=UnpackMode.B16_TO_B32)
                u_raw_post = vload_unpack(u_in_read, base + VL, unpack_mode=UnpackMode.B16_TO_B32)
                u_fp32_pre = vcast(u_raw_pre, dtypes.float32, mask=mask)
                u_fp32_post = vcast(u_raw_post, dtypes.float32, mask=mask)
                delta_pre = vload(delta, base)
                delta_post = vload(delta, base + VL)
                u_next_pre = vadd(u_fp32_pre, delta_pre, mask=mask)
                u_next_post = vadd(u_fp32_post, delta_post, mask=mask)
                u_bf16_pre = vcast(u_next_pre, dtypes.bfloat16, mask=mask)
                u_bf16_post = vcast(u_next_post, dtypes.bfloat16, mask=mask)
                vstore_pack(self.tmp_u_bf16, base, u_bf16_pre, mask, pack_mode=PackMode.B32_TO_B16)
                vstore_pack(self.tmp_u_bf16, base + VL, u_bf16_post, mask, pack_mode=PackMode.B32_TO_B16)

        u_nz_slot = self.u_nz
        nd2nz_tile(u_nz_slot, self.tmp_u_bf16)
        u_nz_read = self.u_nz
        mem_copy(u_l1_rows, u_nz_read)

    def store_state_to_gm(self, gm_state_rows):
        """末态 ub_state_fp32 → GM（FP32）。累加在 V pipe、写 GM 在 MTE3，跨 pipe，经 Channel 交接。"""
        ub_state = self.ub_state_fp32
        out_slot = self.state_out
        mem_copy(out_slot, ub_state)
        out_read = self.state_out
        mem_copy(gm_state_rows, out_read)


@jit
def _run_stage2_body(
    gm_k_restored,
    gm_gamma,
    gm_q_decayed,
    gm_w,
    gm_U,
    gm_O,
    gm_state_in,
    gm_state_out,
    gm_Mqk,
    *,
    workspace_task_base: int,
    dv_base: int,
    head_base: int,
    head_count: int,
    chunks_total: int,
    seq_len: int,
    group_start: int,
    group_count: int,
    eff_group: int,
    logical_core_num: int,
    item_start: int,
    item_end: int,
):

    block_idx = get_block_idx()
    subblock_idx = get_subblock_id()
    num_dv_base = SUPPORTED_HEAD_DIM // dv_base
    half_dk = D_HALF
    u_row_block = CHUNK_SIZE // 2

    q_l1 = Channel(MemLoc.L1, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=2)
    w_l1 = Channel(MemLoc.L1, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=2)
    k_restored_l1 = Channel(
        MemLoc.L1, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=2, data_format="nz"
    )
    mqk_l1 = Channel(MemLoc.L1, shape=(CHUNK_SIZE, CHUNK_SIZE), dtype=dtypes.bfloat16, depth=2)
    state_l1 = Channel(
        MemLoc.L1,
        shape=(dv_base, SUPPORTED_HEAD_DIM),
        dtype=dtypes.bfloat16,
        depth=2,
        data_format="nz",
        kind=ChannelKind.CrossCore,
    )
    u_l1 = Channel(
        MemLoc.L1, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=2, kind=ChannelKind.CrossCore
    )
    s2_matmul = StageTwoMatmul(dv_base=dv_base)
    gamma_ub = Channel(MemLoc.UB, shape=(half_dk, 1), dtype=dtypes.float32, depth=2)
    tmp_delta = Channel(MemLoc.UB, shape=(dv_base, half_dk), dtype=dtypes.float32, depth=1, kind=ChannelKind.CrossCore).produce()
    tmp_u_delta = Channel(MemLoc.UB, shape=(u_row_block, dv_base), dtype=dtypes.float32, depth=1, kind=ChannelKind.CrossCore).produce()

    s2_vector = StageTwoVector(u_row_block=u_row_block, dv_base=dv_base)


    if block_idx < logical_core_num:
        for item in range(item_start, item_end):
            head_id, dv_idx = idx2crd(item, [head_count, num_dv_base])
            seq_idx = head_base + head_id
            write_state_ref = gm_state_out[0, seq_idx, None, None]

            first_linear = workspace_task_base + seq_idx * eff_group
            first_w_chunk = tile_slice(gm_w, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (first_linear, 0))
            first_q_chunk = tile_slice(gm_q_decayed, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (first_linear, 0))
            first_k_restored_chunk = tile_slice(gm_k_restored, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (first_linear, 0))
            first_mqk_tile = tile_slice(gm_Mqk, (CHUNK_SIZE, CHUNK_SIZE), (first_linear, 0))
            first_gamma_col_full = tile_slice(gm_gamma, (SUPPORTED_HEAD_DIM, 1), (first_linear, 0))
            first_gamma_col = tile_slice(first_gamma_col_full, (half_dk, 1), (subblock_idx, 0))
            first_u_tile_full = tile_slice(gm_U, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (first_linear, 0))
            first_u_tile = tile_slice(first_u_tile_full, (CHUNK_SIZE, dv_base), (0, dv_idx))
            s2_matmul.load_k_cumdecay(w_l1, first_w_chunk)
            s2_matmul.load_q_decayed(q_l1, first_q_chunk)
            s2_matmul.load_k_restored(k_restored_l1, first_k_restored_chunk)
            s2_matmul.load_mqk(mqk_l1, first_mqk_tile)
            if group_start == 0:
                in_state_ref = gm_state_in[0, seq_idx, None, None]
                first_state_rows = tile_slice(in_state_ref, (dv_base, half_dk), (dv_idx, subblock_idx))
                s2_vector.load_initial_state_and_write_l1(first_state_rows, state_l1, subblock_idx)
            else:
                first_state_rows = tile_slice(write_state_ref, (dv_base, half_dk), (dv_idx, subblock_idx))
                s2_vector.load_initial_state_and_write_l1(first_state_rows, state_l1, subblock_idx)
            s2_vector.load_gamma_col(gamma_ub, first_gamma_col)
            s2_vector.issue_u_rows(first_u_tile, subblock_idx)

            for chunk_id in range(0, group_count):
                k = group_start + chunk_id
                linear = workspace_task_base + seq_idx * eff_group + chunk_id
                valid_rows = CHUNK_SIZE
                if k + 1 == chunks_total:
                    valid_rows = seq_len - k * CHUNK_SIZE
                o_chunk = _o_head_chunk_tile(gm_O, seq_idx, k, dv_base, dv_idx, valid_rows)
                s2_matmul.load_state(state_l1)
                s2_matmul.compute_w_state(w_l1)
                s2_matmul.compute_q_state(q_l1)
                s2_matmul.load_k_restored_operand(k_restored_l1)
                s2_matmul.load_mqk_operand(mqk_l1)
                if chunk_id + 1 < group_count:
                    next_linear = linear + 1
                    next_q_chunk = tile_slice(gm_q_decayed, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (next_linear, 0))
                    next_w_chunk = tile_slice(gm_w, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (next_linear, 0))
                    next_gamma_col_full = tile_slice(gm_gamma, (SUPPORTED_HEAD_DIM, 1), (next_linear, 0))
                    next_gamma_col = tile_slice(next_gamma_col_full, (half_dk, 1), (subblock_idx, 0))
                    s2_matmul.load_q_decayed(q_l1, next_q_chunk)
                    s2_matmul.load_k_cumdecay(w_l1, next_w_chunk)
                    s2_vector.load_gamma_col(gamma_ub, next_gamma_col)
                    next_u_tile_full = tile_slice(gm_U, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (next_linear, 0))
                    next_u_tile = tile_slice(next_u_tile_full, (CHUNK_SIZE, dv_base), (0, dv_idx))
                    s2_vector.issue_u_rows(next_u_tile, subblock_idx)
                s2_matmul.store_u_delta(tmp_u_delta)

                s2_vector.finish_u_delta_to_l1(u_l1, tmp_u_delta, subblock_idx)

                u_slot = u_l1.consume()
                s2_matmul.update_state(u_slot)
                s2_matmul.store_state_delta(tmp_delta)

                s2_matmul.accumulate_mqk_u(u_slot)
                if chunk_id + 1 < group_count:
                    next_linear = linear + 1
                    next_k_restored_chunk = tile_slice(gm_k_restored, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (next_linear, 0))
                    next_mqk_tile = tile_slice(gm_Mqk, (CHUNK_SIZE, CHUNK_SIZE), (next_linear, 0))
                    s2_matmul.load_k_restored(k_restored_l1, next_k_restored_chunk)
                    s2_matmul.load_mqk(mqk_l1, next_mqk_tile)
                s2_matmul.store_output(o_chunk)

                s2_vector.scale_and_add_delta(tmp_delta, gamma_ub)
                if chunk_id + 1 < group_count:
                    s2_vector.write_state_l1_snapshot(state_l1, subblock_idx)
                else:
                    state_rows = tile_slice(write_state_ref, (dv_base, half_dk), (dv_idx, subblock_idx))
                    s2_vector.store_state_to_gm(state_rows)


# mode=0 below is a whole-grid barrier. The metadata operator limits the launch
# to the resident cube-core count so every block can reach the barrier.
STAGE1_BOUNDARY_FLAG_ID = 0
STAGE2_BOUNDARY_FLAG_ID = 1
_STAGE_BOUNDARY_AIC_FLAG_OFFSET = 4
_STAGE_BOUNDARY_AIC_AIV_FLAG_OFFSET = 8
QK_PUBLISH_FLAG_ID = 10
QK_CONSUME_FLAG_ID = 11
MTE2_TO_V_EVENT_ID = 0
MTE2_TO_V_V_STAGING_EVENT_ID = 1
MTE2_TO_V_SUBBLOCK1_EVENT_ID = 4
QK_V_TO_MTE3_EVENT_ID = 2
QK_MTE2_TO_V_EVENT_ID = 3
GATE_MTE2_TO_V_EVENT_ID = 5


def _rewind_buffers():
    channel_rewind(reset_sync_id=False)


def _stage_boundary_reset(flag_id: int):
    cube_sync_all()
    vec_sync_all()

    aiv_ready_flag_id = flag_id
    aic_ready_flag_id = flag_id + _STAGE_BOUNDARY_AIC_FLAG_OFFSET
    aic_aiv_ready_flag_id = flag_id + _STAGE_BOUNDARY_AIC_AIV_FLAG_OFFSET

    vec_sync_block_arrive(PIPE.MTE3, aiv_ready_flag_id, mode=2)
    cube_sync_block_wait(PIPE.S, aiv_ready_flag_id, mode=2)

    cube_sync_block_arrive(PIPE.FIXPIPE, aic_ready_flag_id, mode=0)
    cube_sync_block_wait(PIPE.S, aic_ready_flag_id, mode=0)

    cube_sync_block_arrive(PIPE.MTE3, aic_aiv_ready_flag_id, mode=2)
    vec_sync_block_wait(PIPE.S, aic_aiv_ready_flag_id, mode=2)
    _rewind_buffers()


# ---------------------------------------------------------------------------
# FlashKDA AICore consumer and public wrapper.
# ---------------------------------------------------------------------------
# The AICore consumer keeps the metadata ABI fields it reads locally.
MAX_AIC_CORES = 32
WORKSPACE_SLOTS = 861
HEADER_STATUS = 2
HEADER_CORE_NUM = 3
HEADER_STAGE12_ROUND_NUM = 4
HEADER_WORDS = 9
STATUS_OK = 0


def _device_block_num(ref: Optional[torch.Tensor] = None) -> int:
    """Return a grid-barrier-safe launch size for the current NPU stream."""
    if ref is not None and ref.device.type not in {"npu", "privateuseone"}:
        return MAX_AIC_CORES
    npu = getattr(torch, "npu", None)
    if npu is None:
        return MAX_AIC_CORES
    device_index = ref.device.index if ref is not None else None
    if device_index is None:
        try:
            device_index = npu.current_device()
        except RuntimeError:
            return MAX_AIC_CORES
    stream = npu.current_stream(device_index)
    platform_info = get_platform_info(stream=stream)
    try:
        cube_core_num = int(platform_info.cube_core_num)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"NPU {device_index} does not expose a valid cube_core_num") from exc
    if cube_core_num <= 0:
        raise RuntimeError(f"NPU {device_index} reports invalid cube_core_num={cube_core_num}")
    return min(cube_core_num, MAX_AIC_CORES)


def _rewind_mutually_exclusive_stage2_branch() -> None:
    """Declare each Stage2 alternative's explicit allocation boundary."""
    channel_rewind(reset_sync_id=True)


@jit
def _run_stage2_round(
    gm_k_restored,
    gm_gamma,
    gm_q_decayed,
    gm_w,
    gm_u,
    gm_o,
    gm_state_in,
    gm_state_out,
    gm_mqk,
    gm_metadata,
    *,
    dv_base: int,
    value_heads,
    active_batch_num,
    batch_idx_base,
    storage_batch_idx_base,
    token_start_base,
    valid_seq_len_base,
    chunk_start_base,
    chunk_num_base,
    stage1_task_prefix_base,
    stage2_task_prefix_base,
    stage2_core_start,
    stage2_core_end,
    logical_core_num,
):
    for active_batch_idx in range(0, active_batch_num):
        prefix_start = gm_metadata[stage2_task_prefix_base + active_batch_idx]
        prefix_end = gm_metadata[stage2_task_prefix_base + active_batch_idx + 1]
        intersection_start = stage2_core_start
        if intersection_start < prefix_start:
            intersection_start = prefix_start
        intersection_end = stage2_core_end
        if intersection_end > prefix_end:
            intersection_end = prefix_end
        item_start = 0
        item_end = 0
        if intersection_start < intersection_end:
            item_start = intersection_start - prefix_start
            item_end = intersection_end - prefix_start

        logical_batch = gm_metadata[batch_idx_base + active_batch_idx]
        storage_batch = gm_metadata[storage_batch_idx_base + active_batch_idx]
        token_start = gm_metadata[token_start_base + active_batch_idx]
        seq_len = dtypes.int64(gm_metadata[valid_seq_len_base + active_batch_idx])
        group_start = gm_metadata[chunk_start_base + active_batch_idx]
        chunk_num_per_group = gm_metadata[chunk_num_base + active_batch_idx]
        chunks_total = (seq_len + (CHUNK_SIZE - 1)) // CHUNK_SIZE

        token_stop = token_start + seq_len
        o = gm_o[storage_batch, None, token_start:token_stop, None]
        state_in = gm_state_in[logical_batch, None, None, None]
        state_out = gm_state_out[logical_batch, None, None, None]
        workspace_slot_start = gm_metadata[stage1_task_prefix_base + active_batch_idx]
        _run_stage2_body(
            gm_k_restored,
            gm_gamma,
            gm_q_decayed,
            gm_w,
            gm_u,
            o,
            state_in,
            state_out,
            gm_mqk,
            workspace_task_base=workspace_slot_start,
            dv_base=dv_base,
            head_base=0,
            head_count=value_heads,
            chunks_total=chunks_total,
            seq_len=seq_len,
            group_start=group_start,
            group_count=chunk_num_per_group,
            eff_group=chunk_num_per_group,
            logical_core_num=logical_core_num,
            item_start=item_start,
            item_end=item_end,
        )


@jit
def _run_stage2_dynamic(
    gm_k_restored,
    gm_gamma,
    gm_q_decayed,
    gm_w,
    gm_u,
    gm_o,
    gm_state_in,
    gm_state_out,
    gm_mqk,
    gm_metadata,
    *,
    dv_splits_num,
    value_heads,
    active_batch_num,
    batch_idx_base,
    storage_batch_idx_base,
    token_start_base,
    valid_seq_len_base,
    chunk_start_base,
    chunk_num_base,
    stage1_task_prefix_base,
    stage2_task_prefix_base,
    stage2_core_start,
    stage2_core_end,
    logical_core_num,
):
    if dv_splits_num == 1:
        _rewind_mutually_exclusive_stage2_branch()
        _run_stage2_round(
            gm_k_restored,
            gm_gamma,
            gm_q_decayed,
            gm_w,
            gm_u,
            gm_o,
            gm_state_in,
            gm_state_out,
            gm_mqk,
            gm_metadata,
            dv_base=128,
            value_heads=value_heads,
            active_batch_num=active_batch_num,
            batch_idx_base=batch_idx_base,
            storage_batch_idx_base=storage_batch_idx_base,
            token_start_base=token_start_base,
            valid_seq_len_base=valid_seq_len_base,
            chunk_start_base=chunk_start_base,
            chunk_num_base=chunk_num_base,
            stage1_task_prefix_base=stage1_task_prefix_base,
            stage2_task_prefix_base=stage2_task_prefix_base,
            stage2_core_start=stage2_core_start,
            stage2_core_end=stage2_core_end,
            logical_core_num=logical_core_num,
        )
    elif dv_splits_num == 2:
        _rewind_mutually_exclusive_stage2_branch()
        _run_stage2_round(
            gm_k_restored,
            gm_gamma,
            gm_q_decayed,
            gm_w,
            gm_u,
            gm_o,
            gm_state_in,
            gm_state_out,
            gm_mqk,
            gm_metadata,
            dv_base=64,
            value_heads=value_heads,
            active_batch_num=active_batch_num,
            batch_idx_base=batch_idx_base,
            storage_batch_idx_base=storage_batch_idx_base,
            token_start_base=token_start_base,
            valid_seq_len_base=valid_seq_len_base,
            chunk_start_base=chunk_start_base,
            chunk_num_base=chunk_num_base,
            stage1_task_prefix_base=stage1_task_prefix_base,
            stage2_task_prefix_base=stage2_task_prefix_base,
            stage2_core_start=stage2_core_start,
            stage2_core_end=stage2_core_end,
            logical_core_num=logical_core_num,
        )
    elif dv_splits_num == 4:
        _rewind_mutually_exclusive_stage2_branch()
        _run_stage2_round(
            gm_k_restored,
            gm_gamma,
            gm_q_decayed,
            gm_w,
            gm_u,
            gm_o,
            gm_state_in,
            gm_state_out,
            gm_mqk,
            gm_metadata,
            dv_base=32,
            value_heads=value_heads,
            active_batch_num=active_batch_num,
            batch_idx_base=batch_idx_base,
            storage_batch_idx_base=storage_batch_idx_base,
            token_start_base=token_start_base,
            valid_seq_len_base=valid_seq_len_base,
            chunk_start_base=chunk_start_base,
            chunk_num_base=chunk_num_base,
            stage1_task_prefix_base=stage1_task_prefix_base,
            stage2_task_prefix_base=stage2_task_prefix_base,
            stage2_core_start=stage2_core_start,
            stage2_core_end=stage2_core_end,
            logical_core_num=logical_core_num,
        )
    else:
        _rewind_mutually_exclusive_stage2_branch()
        _run_stage2_round(
            gm_k_restored,
            gm_gamma,
            gm_q_decayed,
            gm_w,
            gm_u,
            gm_o,
            gm_state_in,
            gm_state_out,
            gm_mqk,
            gm_metadata,
            dv_base=16,
            value_heads=value_heads,
            active_batch_num=active_batch_num,
            batch_idx_base=batch_idx_base,
            storage_batch_idx_base=storage_batch_idx_base,
            token_start_base=token_start_base,
            valid_seq_len_base=valid_seq_len_base,
            chunk_start_base=chunk_start_base,
            chunk_num_base=chunk_num_base,
            stage1_task_prefix_base=stage1_task_prefix_base,
            stage2_task_prefix_base=stage2_task_prefix_base,
            stage2_core_start=stage2_core_start,
            stage2_core_end=stage2_core_end,
            logical_core_num=logical_core_num,
        )


@kernel(profile=cannbotdsl.ProfileSpec(
    name="flash_kda",
    op_type="FlashKDA",
    inputs=(
        "gm_K", "gm_V", "gm_Q", "gm_beta", "gm_g", "gm_A_log",
        "gm_dt_bias", "gm_state_in", "gm_metadata",
    ),
    outputs=("gm_O",),
    inouts=("gm_state_out",),
))
class flash_kda_kernel:
    """Consume variable cross-batch metadata with the launched AIC grid."""

    def __call__(
        self,
        gm_K: Tensor,
        gm_V: Tensor,
        gm_Q: Tensor,
        gm_beta: Tensor,
        gm_g: Tensor,
        gm_qk_exchange: Tensor,
        gm_A_log: Tensor,
        gm_dt_bias: Tensor,
        gm_k_restored: Tensor,
        gm_gamma: Tensor,
        gm_q_decayed: Tensor,
        gm_w: Tensor,
        gm_Mqk: Tensor,
        gm_U: Tensor,
        gm_O: Tensor,
        gm_state_in: Tensor,
        gm_state_out: Tensor,
        gm_identity16: Tensor,
        gm_metadata: Tensor,
        scale_value: float,
        lower_bound: float,
    ):
        block_idx = get_block_idx()
        block_num = get_block_num()
        metadata_status = gm_metadata[HEADER_STATUS]
        metadata_core_num = gm_metadata[HEADER_CORE_NUM]
        if metadata_status == STATUS_OK and metadata_core_num == block_num:
            stage12_round_num = gm_metadata[HEADER_STAGE12_ROUND_NUM]
            for global_round_idx in range(0, stage12_round_num):
                record = gm_metadata[HEADER_WORDS + global_round_idx]
                active_batch_num = gm_metadata[record + 1]
                batch_idx_base = record + 2
                storage_batch_idx_base = batch_idx_base + active_batch_num
                token_start_base = storage_batch_idx_base + active_batch_num
                valid_seq_len_base = token_start_base + active_batch_num
                group_num_index = valid_seq_len_base + active_batch_num
                chunk_start_base = group_num_index + 1
                chunk_num_base = chunk_start_base + active_batch_num
                stage1_task_prefix_base = chunk_num_base + active_batch_num
                stage1_core_ranges_base = stage1_task_prefix_base + active_batch_num + 1
                dv_splits_num_base = stage1_core_ranges_base + (MAX_AIC_CORES * 2)
                stage2_task_prefix_base = dv_splits_num_base + active_batch_num
                stage2_core_ranges_base = stage2_task_prefix_base + active_batch_num + 1

                stage1_core_range_word = stage1_core_ranges_base + block_idx * 2
                stage1_core_start = gm_metadata[stage1_core_range_word]
                stage1_core_end = gm_metadata[stage1_core_range_word + 1]
                for active_batch_idx in range(0, active_batch_num):
                    prefix_start = gm_metadata[stage1_task_prefix_base + active_batch_idx]
                    prefix_end = gm_metadata[stage1_task_prefix_base + active_batch_idx + 1]
                    intersection_start = stage1_core_start
                    if intersection_start < prefix_start:
                        intersection_start = prefix_start
                    intersection_end = stage1_core_end
                    if intersection_end > prefix_end:
                        intersection_end = prefix_end
                    task_start = 0
                    task_end = 0
                    if intersection_start < intersection_end:
                        task_start = intersection_start - prefix_start
                        task_end = intersection_end - prefix_start

                    storage_batch = gm_metadata[storage_batch_idx_base + active_batch_idx]
                    token_start = gm_metadata[token_start_base + active_batch_idx]
                    seq_len = dtypes.int64(gm_metadata[valid_seq_len_base + active_batch_idx])
                    group_start = gm_metadata[chunk_start_base + active_batch_idx]
                    chunk_num_per_group = gm_metadata[chunk_num_base + active_batch_idx]
                    chunks_total = (seq_len + (CHUNK_SIZE - 1)) // CHUNK_SIZE
                    tail_rows = seq_len - (chunks_total - 1) * CHUNK_SIZE
                    tail_group = group_start + chunk_num_per_group == chunks_total

                    token_stop = token_start + seq_len
                    k = gm_K[storage_batch, None, token_start:token_stop, None]
                    v = gm_V[storage_batch, None, token_start:token_stop, None]
                    q = gm_Q[storage_batch, None, token_start:token_stop, None]
                    beta = gm_beta[storage_batch, None, token_start:token_stop, None]
                    gate = gm_g[storage_batch, None, token_start:token_stop, None]
                    value_heads = v.shape[1]
                    _run_stage1_body(
                        tail_group,
                        tail_rows,
                        k,
                        v,
                        q,
                        beta,
                        gate,
                        gm_qk_exchange,
                        gm_A_log,
                        gm_dt_bias,
                        gm_q_decayed,
                        gm_Mqk,
                        gm_k_restored,
                        gm_gamma,
                        gm_U,
                        gm_w,
                        gm_identity16,
                        prefix_start,
                        value_heads,
                        value_heads,
                        chunks_total,
                        seq_len,
                        0,
                        group_start,
                        chunk_num_per_group,
                        chunk_num_per_group,
                        block_num,
                        task_start,
                        task_end,
                        scale_value,
                        lower_bound,
                    )

                _stage_boundary_reset(STAGE1_BOUNDARY_FLAG_ID)

                stage2_core_range_word = stage2_core_ranges_base + block_idx * 2
                stage2_core_start = gm_metadata[stage2_core_range_word]
                stage2_core_end = gm_metadata[stage2_core_range_word + 1]
                round_dv_splits_num = gm_metadata[dv_splits_num_base]
                _run_stage2_dynamic(
                    gm_k_restored,
                    gm_gamma,
                    gm_q_decayed,
                    gm_w,
                    gm_U,
                    gm_O,
                    gm_state_in,
                    gm_state_out,
                    gm_Mqk,
                    gm_metadata,
                    dv_splits_num=round_dv_splits_num,
                    value_heads=gm_V.shape[1],
                    active_batch_num=active_batch_num,
                    batch_idx_base=batch_idx_base,
                    storage_batch_idx_base=storage_batch_idx_base,
                    token_start_base=token_start_base,
                    valid_seq_len_base=valid_seq_len_base,
                    chunk_start_base=chunk_start_base,
                    chunk_num_base=chunk_num_base,
                    stage1_task_prefix_base=stage1_task_prefix_base,
                    stage2_task_prefix_base=stage2_task_prefix_base,
                    stage2_core_start=stage2_core_start,
                    stage2_core_end=stage2_core_end,
                    logical_core_num=block_num,
                )

                if global_round_idx + 1 < stage12_round_num:
                    _stage_boundary_reset(STAGE2_BOUNDARY_FLAG_ID)


class FlashKDA:
    """Compiled AICore entry used by :func:`flash_kda`."""

    @host
    def run(
        self,
        K: Tensor,
        V: Tensor,
        Q: Tensor,
        beta: Tensor,
        g: Tensor,
        qk_exchange: Tensor,
        A_log: Tensor,
        dt_bias: Tensor,
        kr: Tensor,
        gamma: Tensor,
        qd: Tensor,
        w: Tensor,
        mqk: Tensor,
        u: Tensor,
        out: Tensor,
        state_in: Tensor,
        state_out: Tensor,
        identity: Tensor,
        metadata: Tensor,
        scale: float,
        lower_bound: float,
        core_num: int,
    ):
        op = flash_kda_kernel()
        op[core_num](
            K,
            V,
            Q,
            beta,
            g,
            qk_exchange,
            A_log,
            dt_bias,
            kr,
            gamma,
            qd,
            w,
            mqk,
            u,
            out,
            state_in,
            state_out,
            identity,
            metadata,
            scale,
            lower_bound,
        )


def _sequence_spec(dtype, physical_batches, length, heads, width, stride_prefix):
    stride_0 = cannbotdsl.Dim(f"{stride_prefix}_S0")
    stride_1 = cannbotdsl.Dim(f"{stride_prefix}_S1")
    stride_2 = cannbotdsl.Dim(f"{stride_prefix}_S2")
    return cannbotdsl.TensorSpec(
        (physical_batches, heads, length, width), dtype, stride=(stride_0, stride_1, stride_2, 1)
    )


@compile_cache
def _compile_flash_kda():
    physical_batches = cannbotdsl.Dim("P")
    storage_length = cannbotdsl.Dim("L")
    output_length = cannbotdsl.Dim("OL")
    batch = cannbotdsl.Dim("B")
    n_qk = cannbotdsl.Dim("NQK")
    n_v = cannbotdsl.Dim("NV")
    metadata_capacity = cannbotdsl.Dim("META")
    core_num = cannbotdsl.Dim("CORE_NUM", min=1, max=MAX_AIC_CORES)
    scratch_tokens = WORKSPACE_SLOTS * CHUNK_SIZE
    gamma_rows = WORKSPACE_SLOTS * SUPPORTED_HEAD_DIM
    fake = cannbotdsl.TensorSpec
    qk_spec = _sequence_spec(
        dtypes.bfloat16, physical_batches, storage_length, n_qk, SUPPORTED_HEAD_DIM, "QK"
    )
    v_spec = _sequence_spec(
        dtypes.bfloat16, physical_batches, storage_length, n_v, SUPPORTED_HEAD_DIM, "V"
    )
    beta_spec = _sequence_spec(
        dtypes.bfloat16, physical_batches, storage_length, n_v, 1, "BETA"
    )
    output_spec = _sequence_spec(
        dtypes.bfloat16, physical_batches, output_length, n_v, SUPPORTED_HEAD_DIM, "OUT"
    )
    state_spec = fake((batch, n_v, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), dtypes.float32)
    return cannbotdsl.compile(
        FlashKDA().run,
        qk_spec,
        v_spec,
        qk_spec,
        beta_spec,
        v_spec,
        fake((core_num, 2, 2, CHUNK_SIZE), dtypes.float32),
        fake((n_v,), dtypes.float32),
        fake((n_v, SUPPORTED_HEAD_DIM), dtypes.float32),
        fake((scratch_tokens, SUPPORTED_HEAD_DIM), dtypes.bfloat16),
        fake((gamma_rows, 1), dtypes.float32),
        fake((scratch_tokens, SUPPORTED_HEAD_DIM), dtypes.bfloat16),
        fake((scratch_tokens, SUPPORTED_HEAD_DIM), dtypes.bfloat16),
        fake((scratch_tokens, CHUNK_SIZE), dtypes.bfloat16),
        fake((scratch_tokens, SUPPORTED_HEAD_DIM), dtypes.bfloat16),
        output_spec,
        state_spec,
        state_spec,
        fake((16, 16), dtypes.float16),
        fake((metadata_capacity,), dtypes.int32),
        dtypes.float32,
        dtypes.float32,
        dtypes.int64,
    )


def _validate_metadata(metadata, ref):
    assert metadata is not None, "metadata is required"
    assert metadata.dtype == torch.int32, "metadata must be int32"
    assert metadata.dim() == 1, "metadata must be rank-1"
    assert metadata.is_contiguous(), "metadata must be contiguous"
    assert metadata.device == ref.device, "metadata must be on the same device as q"


def flash_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    layout_qkv: str,
    metadata: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FlashKDA forward (Flash Kimi Delta Attention).

    Args:
        q (torch.Tensor): Query, bf16, BNSD ``[B, Nqk, S, D]``, BSND ``[B, S, Nqk, D]``, or TND ``[T, Nqk, D]``.
        k (torch.Tensor): Key, bf16, with the same shape and layout as ``q``.
        v (torch.Tensor): Value, bf16, BNSD ``[B, Nv, S, D]``, BSND ``[B, S, Nv, D]``, or TND ``[T, Nv, D]``.
        g (torch.Tensor): Raw gate input, bf16, with the same shape and layout as ``v``.
        beta (torch.Tensor): Raw beta logits, bf16, BNSD ``[B, Nv, S]``, BSND ``[B, S, Nv]``, or TND ``[T, Nv]``.
        scale (float): Query-key scaling factor.
        initial_state (torch.Tensor): Initial recurrent state, fp32, shape ``[B, Nv, D, D]``.
        A_log (torch.Tensor): Per-value-head decay coefficient, fp32, shape ``[Nv]``.
        dt_bias (torch.Tensor): Per-value-head bias, fp32, shape ``[Nv, D]``.
        lower_bound (float): Gate lower bound in ``[-5, 0]``.
        layout_qkv (str): Input layout: ``"BNSD"``, ``"BSND"``, or ``"TND"``.
        metadata (torch.Tensor): Reusable int32 scheduling metadata returned by ``flash_kda_metadata()``.

    Returns:
        out (torch.Tensor): Output in the same physical layout as ``v``. Padding rows are undefined.
        final_state (torch.Tensor): Final recurrent state, fp32, shape ``[B, Nv, D, D]``.

    Notes:
        * Currently requires ``Dk = Dv = 128``.
        * All tensor inputs must be contiguous, on the same device, and have the dtypes listed above.
        * ``Nv <= 96``.
        * ``Nv % Nqk == 0``.
        * ``lower_bound`` is in ``[-5, 0]``.
        * Internal output storage is padded to a multiple of 64 rows for kernel computation.
    """
    assert layout_qkv in ("TND", "BNSD", "BSND"), f"layout_qkv must be TND, BNSD, or BSND, got {layout_qkv!r}"
    _validate_metadata(metadata, q)
    assert initial_state.dim() == 4, "initial_state must be rank-4"
    batch, n_v, state_dv, state_dk = initial_state.shape
    assert state_dv == state_dk == SUPPORTED_HEAD_DIM, f"FlashKDA only supports D={SUPPORTED_HEAD_DIM}"

    if layout_qkv == "TND":
        assert q.dim() == k.dim() == v.dim() == g.dim() == 3, "TND q/k/v/g must be rank-3"
        storage_length, n_qk, dim = q.shape
        assert k.shape == q.shape, "q/k shape does not match TND layout"
        assert v.shape == g.shape == (storage_length, n_v, dim), "v/g shape does not match TND layout"
        assert beta.dim() == 2 and beta.shape == (storage_length, n_v), "beta must be [T, Nv] for TND"
        q_logical = torch.as_strided(
            q, (1, n_qk, storage_length, dim), (storage_length * n_qk * dim, dim, n_qk * dim, 1)
        )
        k_logical = torch.as_strided(
            k, (1, n_qk, storage_length, dim), (storage_length * n_qk * dim, dim, n_qk * dim, 1)
        )
        v_logical = torch.as_strided(v, (1, n_v, storage_length, dim), (storage_length * n_v * dim, dim, n_v * dim, 1))
        g_logical = torch.as_strided(g, (1, n_v, storage_length, dim), (storage_length * n_v * dim, dim, n_v * dim, 1))
        beta_logical = torch.as_strided(beta, (1, n_v, storage_length, 1), (storage_length * n_v, 1, n_v, 1))
        padded_length = ((storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE
        out_stage_storage = torch.empty(padded_length, n_v, dim, dtype=LOW_DTYPE, device=q.device)
        out = out_stage_storage[:storage_length]
        out_stage = torch.as_strided(
            out_stage_storage, (1, n_v, padded_length, dim), (padded_length * n_v * dim, dim, n_v * dim, 1)
        )
    else:
        assert q.dim() == k.dim() == v.dim() == g.dim() == 4, "BNSD/BSND q/k/v/g must be rank-4"
        if layout_qkv == "BNSD":
            physical_batches, n_qk, storage_length, dim = q.shape
            assert k.shape == q.shape, "q/k shape does not match BNSD layout"
            assert v.shape == g.shape == (physical_batches, n_v, storage_length, dim), (
                "v/g shape does not match BNSD layout"
            )
            assert beta.dim() == 3 and beta.shape == (physical_batches, n_v, storage_length), (
                "beta must be [B, Nv, S] for BNSD"
            )
            q_logical, k_logical, v_logical, g_logical = q, k, v, g
            beta_logical = beta.unsqueeze(-1)
        else:
            physical_batches, storage_length, n_qk, dim = q.shape
            assert k.shape == q.shape, "q/k shape does not match BSND layout"
            assert v.shape == g.shape == (physical_batches, storage_length, n_v, dim), (
                "v/g shape does not match BSND layout"
            )
            assert beta.dim() == 3 and beta.shape == (physical_batches, storage_length, n_v), (
                "beta must be [B, S, Nv] for BSND"
            )
            q_logical = q.transpose(1, 2)
            k_logical = k.transpose(1, 2)
            v_logical = v.transpose(1, 2)
            g_logical = g.transpose(1, 2)
            beta_logical = beta.unsqueeze(-1).transpose(1, 2)
        assert physical_batches == batch, "padded storage batch must match initial_state"
        padded_length = ((storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE
        if layout_qkv == "BNSD":
            out_stage_storage = torch.empty(batch, n_v, padded_length, dim, dtype=LOW_DTYPE, device=q.device)
            out = out_stage_storage[:, :, :storage_length, :]
            out_stage = out_stage_storage
        else:
            out_stage_storage = torch.empty(batch, padded_length, n_v, dim, dtype=LOW_DTYPE, device=q.device)
            out = out_stage_storage[:, :storage_length, :, :]
            out_stage = out_stage_storage.transpose(1, 2)

    assert dim == SUPPORTED_HEAD_DIM, f"FlashKDA only supports D={SUPPORTED_HEAD_DIM}"
    assert n_v % n_qk == 0, f"GQA requires Nv % Nqk == 0, got Nv={n_v}, Nqk={n_qk}"
    assert storage_length > 0, "storage sequence length must be positive"
    assert q.dtype == k.dtype == v.dtype == g.dtype == beta.dtype == LOW_DTYPE, "q/k/v/g/beta must be bf16"
    assert initial_state.dtype == HIGH_DTYPE, "initial_state must be fp32"
    assert A_log.dtype == HIGH_DTYPE and A_log.shape == (n_v,), f"A_log must be fp32 with shape ({n_v},)"
    assert dt_bias.dtype == HIGH_DTYPE and dt_bias.shape == (n_v, dim), (
        f"dt_bias must be fp32 with shape ({n_v}, {dim})"
    )
    assert -5.0 <= float(lower_bound) <= 0.0, "lower_bound must be in [-5, 0]"
    tensors = (q, k, v, g, beta, initial_state, A_log, dt_bias)
    assert all(tensor.device == q.device for tensor in tensors), "all inputs must be on the same device"
    assert all(tensor.is_contiguous() for tensor in tensors), "all public inputs must be contiguous"

    core_num = _device_block_num(q)
    workspace = allocate_stage1_workspace(1, 1, WORKSPACE_SLOTS, SUPPORTED_HEAD_DIM, ref=q)
    qk_exchange = allocate_qk_exchange_workspace(active_block_num=core_num, ref=q)
    final_state = torch.empty(initial_state.shape, dtype=initial_state.dtype, device=initial_state.device)
    fn = _compile_flash_kda()
    fn(
        k_logical,
        v_logical,
        q_logical,
        beta_logical,
        g_logical,
        qk_exchange,
        A_log,
        dt_bias,
        workspace.K_restored,
        workspace.gamma_C,
        workspace.Q_decayed,
        workspace.W,
        workspace.Mqk,
        workspace.U_pre,
        out_stage,
        initial_state,
        final_state,
        workspace.identity16,
        metadata,
        float(scale),
        float(lower_bound),
        core_num,
    )
    return out, final_state


def clear_caches():
    _compile_flash_kda.cache_clear()


__all__ = ["FlashKDA", "clear_caches", "flash_kda", "flash_kda_kernel"]


@aot.export("flash_kda")
def export_flash_kda():
    _compile_flash_kda()


_GRAPH_LIBRARY = torch.library.Library("cannbotdsl_flash_kda", "DEF")
_GRAPH_LIBRARY.define(
    "flash_kda("
    "Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float scale, "
    "Tensor initial_state, Tensor a_log, Tensor dt_bias, float lower_bound, "
    "str layout_qkv, Tensor metadata"
    ") -> (Tensor, Tensor)"
)


@torch.library.impl(_GRAPH_LIBRARY, "flash_kda", "Meta")
def _flash_kda_meta(q, k, v, g, beta, scale, initial_state, a_log, dt_bias,
                    lower_bound, layout_qkv, metadata):
    assert layout_qkv in ("TND", "BNSD", "BSND"), f"layout_qkv must be TND, BNSD, or BSND, got {layout_qkv!r}"
    _validate_metadata(metadata, q)
    assert initial_state.dtype == HIGH_DTYPE, "initial_state must be float32"
    assert v.dtype == LOW_DTYPE, "v must be bfloat16"
    del k, g, beta, scale, a_log, dt_bias, lower_bound
    # Match the host's padded output views so graph consumers see the real strides.
    sequence_axis = {"TND": 0, "BNSD": 2, "BSND": 1}[layout_qkv]
    output_shape = list(v.shape)
    sequence_length = output_shape[sequence_axis]
    output_shape[sequence_axis] = ((sequence_length + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE
    output_storage = torch.empty(output_shape, dtype=v.dtype, device="meta")
    out = output_storage.narrow(sequence_axis, 0, sequence_length)
    return out, torch.empty_like(initial_state, device="meta")


@torch.library.impl(_GRAPH_LIBRARY, "flash_kda", "PrivateUse1")
def _flash_kda_privateuse1(q, k, v, g, beta, scale, initial_state, a_log,
                           dt_bias, lower_bound, layout_qkv, metadata):
    return flash_kda(q, k, v, g, beta, scale, initial_state, a_log, dt_bias,
                     lower_bound, layout_qkv, metadata)


flash_kda_op = torch.ops.cannbotdsl_flash_kda.flash_kda


@wraps(flash_kda)
def _flash_kda_graph(*args, **kwargs):
    # Keep the public keyword spelling compatible with the dispatcher schema.
    if "A_log" in kwargs:
        kwargs["a_log"] = kwargs.pop("A_log")
    return flash_kda_op(*args, **kwargs)


# Dynamo captures the functional dispatcher call; the operator allocates both outputs.
torch._dynamo.substitute_in_graph(
    flash_kda, can_constant_fold_through=False
)(_flash_kda_graph)
