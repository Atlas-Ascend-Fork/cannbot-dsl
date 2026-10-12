# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host-side Flash Attention checks and their supporting calculations."""

from dataclasses import dataclass

import torch

LSE_ROW_LANES = 8
L0C_DEPTH = 4
_PA_LAYOUT_NAMES = frozenset(("PA_BBND", "PA_BNBD", "PA_NZ"))


@dataclass(frozen=True)
class TileConfig:
    name: str
    head_dim: int
    tile_cube_m: int
    tile_vec_m: int
    tile_n: int
    cube_d: int
    d_chunks: int


_TILE_CONFIGS = {
    64: TileConfig("D64_M128_N128", 64, 128, 64, 128, 64, 1),
    128: TileConfig("D128_M128_N128", 128, 128, 64, 128, 128, 1),
    256: TileConfig("D256_M64_N128", 256, 64, 32, 128, 128, 2),
}


def select_tile_config(head_dim):
    """Return the physical tile configuration implemented for head_dim."""
    if not isinstance(head_dim, int) or isinstance(head_dim, bool):
        raise TypeError("Invalid head_dim: must be an integer")
    if head_dim not in (64, 128, 256):
        raise ValueError(
            f"Unsupported head dim D={head_dim}; supported values are 64, 128, 256"
        )
    return _TILE_CONFIGS[head_dim]


_D256_LONG_Q_TILE = TileConfig("D256_M128_N128", 256, 128, 64, 128, 128, 2)


def select_launch_tile_config(
    head_dim, *, q_heads, kv_heads, max_seqlen_q, layout_q, layout_kv,
    layout_out, mask_mode, has_sequence_vectors=False,
):
    """Use G*max-Q for D256; layouts and masks do not change the tile."""
    default = select_tile_config(head_dim)
    if head_dim != 256:
        return default
    max_q = (2**31 - 1) if max_seqlen_q == -1 else max_seqlen_q
    return default if (q_heads // kv_heads) * max_q < 64 else _D256_LONG_Q_TILE


@dataclass(frozen=True)
class ResourcePlan:
    config_name: str
    l0a_bytes: int
    l0b_bytes: int
    l0c_bytes: int
    l1_bytes: int
    ub_bytes: int
    workspace_row_bytes: int
    d_segment_width: int
    d_segments: int


def estimate_resource_plan(config, *, grouped, window_mask, return_softmax_lse, split_mode,
                           pa_strided=False):
    """Count local-memory channels for a physical tile."""
    # Retain the legacy keyword for existing host-side callers.
    del grouped
    elem16, elem32 = 2, 4
    m, vm, n, d, c = (
        config.tile_cube_m, config.tile_vec_m, config.tile_n,
        config.head_dim, config.cube_d,
    )
    l0_width = max(n, c)
    l0a = 2 * m * l0_width * elem16
    l0c = L0C_DEPTH * m * l0_width * elem32
    l0b = 2 * c * n * elem16 * (1 if c == n else 2)
    q_l1_bytes = 2 * m * d * elem16
    if d > 128:
        # The allocator reserves the full NZ pitch for every D/16 group.
        # This includes the final 256-element pad after the occupied span.
        q_slot_bytes = (m * 16 + 256) * (d // 16) * elem16
        q_slot_bytes = ((q_slot_bytes + 511) // 512) * 512
        q_l1_bytes = 2 * q_slot_bytes
    l1 = q_l1_bytes + 4 * n * d * elem16 + 3 * m * n * elem16
    qk_depth = 4 if d > 128 and n == c else 2
    qk_ub_bytes = qk_depth * vm * n * elem32
    # D256/N128 PV chunks reuse the QK CrossCore FIFO allocation.
    pv_ub_bytes = (
        0 if d > 128 and n == c
        else 2 * config.d_chunks * vm * c * elem32
    )
    p_nz = 2 * vm * n * elem16
    # D256 casts into a packed fp16 view of the fp32 accumulator.
    output = vm * d * (elem32 if d > 128 else elem32 + elem16)
    softmax = 11 * vm * elem32
    masks = vm * n * (2 if window_mask else 1)
    lse = vm * elem32 + vm * LSE_ROW_LANES * elem32 if return_softmax_lse or split_mode else 0
    ub = qk_ub_bytes + pv_ub_bytes + p_nz + output + softmax + masks + lse
    if pa_strided:
        ub += 32 * d if d > 128 and m > 64 else 144 * d
    workspace_row = d * elem16 + LSE_ROW_LANES * elem32 if split_mode else 0
    return ResourcePlan(
        config.name, l0a, l0b, l0c, l1, ub, workspace_row, c, config.d_chunks,
    )


def validate_resource_plan(plan):
    """Reject a tile whose logical allocation exceeds a local pool."""
    capacities = {
        "L0A": 64 * 1024, "L0B": 64 * 1024, "L0C": 256 * 1024,
        "L1": 512 * 1024, "UB": 256 * 1024,
    }
    for pool, used in (
        ("L0A", plan.l0a_bytes), ("L0B", plan.l0b_bytes),
        ("L0C", plan.l0c_bytes), ("L1", plan.l1_bytes), ("UB", plan.ub_bytes),
    ):
        if used > capacities[pool]:
            raise ValueError(
                f"Invalid {plan.config_name}: {pool} needs {used} bytes,"
                f"capacity={capacities[pool]}"
            )


@dataclass(frozen=True)
class ValidatedInputs:
    batch: int
    query_length: int
    kv_length: int
    q_heads: int
    kv_heads: int
    head_dim: int
    tile_config: TileConfig
    pa_block_size: int | None


def _parse_pa_kv_shape(key, value, layout):
    """Return PA page count, KV heads, block size, and logical head dimension."""
    rank = 5 if layout == "PA_NZ" else 4
    if key.ndim != rank or key.shape != value.shape:
        raise ValueError(f"{layout} key and value must have the same {rank}D shape")
    if layout == "PA_BNBD":
        _, heads, block_size, dim = key.shape
    elif layout == "PA_BBND":
        _, block_size, heads, dim = key.shape
    else:
        _, heads, d1, block_size, d0 = key.shape
        if d0 != 32 // key.element_size():
            raise ValueError("Invalid PA_NZ last dimension must fill one 32-byte block")
        dim = d1 * d0
    if block_size % 16 or not 16 <= block_size <= 1024:
        raise ValueError("Invalid PA block_size: must be a 16-aligned value in [16, 1024]")
    if dim not in (64, 128, 256):
        raise ValueError("Invalid PA head dimension must be 64, 128 or 256")
    return int(heads), int(block_size), int(dim)


def _validate_pa_inputs(
    query, key, value, block_table, seqused_kv, layout_q, layout_kv,
    cu_seqlens_q, cu_seqlens_kv,
):
    if block_table is None:
        raise ValueError("Invalid block_table: is required for PA layout")
    if seqused_kv is None:
        raise ValueError("Invalid seqused_kv: is required for PA layout")
    if cu_seqlens_kv is not None:
        raise ValueError("Invalid cu_seqlens_kv: must be None for PA layout")
    if block_table.device != query.device or block_table.dtype != torch.int32:
        raise ValueError("Invalid block_table: must be an int32 tensor on the query device")
    if block_table.ndim != 2:
        raise ValueError("Invalid block_table: must be a 2D tensor")
    if not block_table.is_contiguous():
        raise ValueError("Invalid block_table and seqused_kv: must be contiguous")
    if layout_q != "TND" and cu_seqlens_q is not None:
        raise ValueError("Invalid cu_seqlens_q: must be None unless layout_q is TND")
    return _parse_pa_kv_shape(key, value, layout_kv)


def _validate_launch_metadata(query, metadata, mask_mode, win_left, win_right):
    if metadata is None:
        raise ValueError(
            "Invalid metadata: is required and must be produced by flash_attn_metadata"
        )
    if metadata.dtype != torch.int32 or metadata.device != query.device:
        raise ValueError("Invalid metadata: must be an int32 tensor on the query device")
    if mask_mode not in (0, 3, 4):
        raise ValueError("Invalid mask_mode: must be 0, 3, or 4")
    if win_left < -1 or win_right < -1:
        raise ValueError("Invalid window sizes: must be >= -1")
    if mask_mode != 4 and (win_left != -1 or win_right != -1):
        raise ValueError("Invalid window sizes: must be -1 for full/causal attention")


def _validate_attn_mask(query, attn_mask, mask_mode):
    if mask_mode == 0:
        if attn_mask is not None:
            raise ValueError("Invalid attn_mask: must be None when mask_mode=0")
    else:
        invalid_mask = (
            attn_mask is None or attn_mask.dtype != torch.int8
            or attn_mask.device != query.device
            or tuple(attn_mask.shape) != (2048, 2048)
            or not attn_mask.is_contiguous()
        )
        if invalid_mask:
            raise ValueError(
                "Invalid attn_mask: must be a contiguous NPU int8 [2048, 2048] triangle"
            )


def _query_dimensions(query, layout_q, layout_out, cu_seqlens_q):
    if layout_q == "TND":
        if layout_out != "TND":
            raise ValueError("Invalid TND query: requires TND output")
        if cu_seqlens_q is None:
            raise ValueError("Invalid TND: requires cu_seqlens_q")
        batch = cu_seqlens_q.numel() - 1
        if batch <= 0:
            raise ValueError("Invalid cu_seqlens_q: must describe at least one batch")
        query_length, q_heads, head_dim = query.shape
    elif layout_q == "BSND":
        batch, query_length, q_heads, head_dim = query.shape
    elif layout_q == "BNSD":
        batch, q_heads, query_length, head_dim = query.shape
    else:
        raise ValueError(f"Unsupported query layout: {layout_q!r}")
    return batch, query_length, q_heads, head_dim


def _dense_kv_dimensions(key, layout_q, layout_kv, vectors, batch):
    if layout_q == "TND":
        if layout_kv == "TND":
            if vectors["cu_seqlens_kv"] is None:
                raise ValueError("Invalid TND key/value: requires cu_seqlens_kv")
            kv_length, kv_heads, _ = key.shape
        elif layout_kv == "BNSD":
            if vectors["seqused_kv"] is None or key.shape[0] != batch:
                raise ValueError(
                    "TND query with BNSD key/value requires seqused_kv and matching batch size"
                )
            kv_heads, kv_length = key.shape[1], key.shape[2]
        else:
            raise ValueError("Invalid TND query: requires TND or BNSD key/value")
    elif layout_kv == "BSND":
        kv_heads, kv_length = key.shape[2], key.shape[1]
    elif layout_kv == "BNSD":
        kv_heads, kv_length = key.shape[1], key.shape[2]
    else:
        raise ValueError(f"Unsupported key/value layout: {layout_kv!r}")
    return kv_heads, kv_length


def _validate_sequence_vectors(query, batch, vectors):
    for name, tensor in vectors.items():
        if tensor is None:
            continue
        expected = batch + 1 if name.startswith("cu_") else batch
        invalid_vector = (
            tensor.device != query.device or tensor.dtype != torch.int32
            or tensor.ndim != 1 or tensor.numel() != expected
            or not tensor.is_contiguous()
        )
        if invalid_vector:
            raise ValueError(
                f"Invalid {name}: must be a contiguous NPU int32 vector"
                f" of length {expected}"
            )


def _validate_dense_dimensions(key, value, head_dim, kv_heads, q_heads):
    invalid_heads = (
        head_dim not in (64, 128, 256) or kv_heads <= 0
        or q_heads < kv_heads or q_heads % kv_heads
    )
    if invalid_heads:
        raise ValueError("Invalid flash_attn: requires D=64/128/256 and integral GQA")
    if key.shape[-1] != head_dim or value.shape[-1] != head_dim:
        raise ValueError("Invalid Q/K/V head dimensions must match")


def validate_flash_attn_inputs(
    *, query, key, value, block_table, cu_seqlens_q, cu_seqlens_kv,
    seqused_q, seqused_kv, attn_mask, metadata, mask_mode, win_left,
    win_right, layout_q, layout_kv, layout_out, return_softmax_lse,
    max_seqlen_q=-1,
):
    """Validate existing host launch requirements and return derived dimensions."""
    _validate_launch_metadata(query, metadata, mask_mode, win_left, win_right)
    _validate_attn_mask(query, attn_mask, mask_mode)

    is_pa = layout_kv in _PA_LAYOUT_NAMES
    if not is_pa and block_table is not None:
        raise ValueError("Invalid block_table: is only valid with a PA layout")
    batch, query_length, q_heads, head_dim = _query_dimensions(
        query, layout_q, layout_out, cu_seqlens_q,
    )

    vectors = dict(
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q, seqused_kv=seqused_kv,
    )
    pa_block_size = None
    if is_pa:
        kv_heads, pa_block_size, pa_dim = _validate_pa_inputs(
            query, key, value, block_table, seqused_kv, layout_q, layout_kv,
            cu_seqlens_q, cu_seqlens_kv,
        )
        invalid_pa_dimensions = (
            pa_dim != head_dim or kv_heads <= 0
            or q_heads < kv_heads or q_heads % kv_heads
        )
        if invalid_pa_dimensions:
            raise ValueError("Invalid PA KV heads and head dimension must form integral GQA")
        if block_table.shape[0] != batch:
            raise ValueError("Invalid block_table batch size must match query batch size")
        kv_length = block_table.shape[1] * pa_block_size
    else:
        kv_heads, kv_length = _dense_kv_dimensions(
            key, layout_q, layout_kv, vectors, batch,
        )

    _validate_sequence_vectors(query, batch, vectors)
    if not is_pa:
        _validate_dense_dimensions(key, value, head_dim, kv_heads, q_heads)
    tile_config = select_launch_tile_config(
        head_dim, q_heads=q_heads, kv_heads=kv_heads,
        max_seqlen_q=max_seqlen_q, layout_q=layout_q,
        layout_kv=layout_kv, layout_out=layout_out, mask_mode=mask_mode,
        has_sequence_vectors=any(tensor is not None for tensor in vectors.values()),
    )
    validate_resource_plan(estimate_resource_plan(
        tile_config, grouped=True, window_mask=mask_mode == 4,
        return_softmax_lse=return_softmax_lse, split_mode=True,
        pa_strided=is_pa and (key.stride(-1) != 1 or value.stride(-1) != 1),
    ))
    return ValidatedInputs(
        batch, query_length, kv_length, q_heads, kv_heads, head_dim,
        tile_config, pa_block_size,
    )


def validate_core_counts(cube, vector):
    if cube <= 0 or vector < cube:
        raise RuntimeError(f"Invalid effective NPU core counts: AIC={cube}, AIV={vector}")


def validate_output_init_core_counts(need_init_output, cube, vector):
    if need_init_output and vector < 2 * cube:
        raise RuntimeError(
            "Invalid output initialization: requires one resident AIV wave: "
            f"AIC={cube}, AIV={vector}"
        )


def output_shapes(layout_q, layout_out, query, batch, query_length, q_heads, head_dim):
    """Validate output layout and return output and LSE shapes."""
    if layout_out == "TND":
        out_shape = tuple(query.shape)
    elif layout_out == "BSND":
        out_shape = (batch, query_length, q_heads, head_dim)
    elif layout_out == "BNSD":
        out_shape = (batch, q_heads, query_length, head_dim)
    else:
        raise ValueError(f"Unsupported output layout: {layout_out!r}")
    lse_shape = (
        (q_heads, query_length) if layout_q == "TND"
        else (batch, q_heads, query_length)
    )
    return out_shape, lse_shape


def validate_workspace(name, workspace, required_rows, width, expected_dtype, device):
    """Check an allocated or caller-provided workspace against the launch shape."""
    invalid_workspace = (
        workspace.ndim != 2 or workspace.shape[1] != width
        or workspace.shape[0] < required_rows
        or workspace.dtype != expected_dtype
        or workspace.device != device or not workspace.is_contiguous()
    )
    if invalid_workspace:
        raise ValueError(
            f"{name} must be contiguous on {device}, dtype={expected_dtype}, "
            f"shape>=({required_rows}, {width})"
        )
