# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host-only business-input validation and private variable-head workspace.

The device ABI remains private. Common Query metadata can be exported/reused.
Weight packing is deliberately outside run; import torch_npu only for NZ checks.
"""

if __package__:
    from .attention_geometry import (
        HEAD_DIM,
        choose_compressed_cache_decode_strategy,
        validate_attention_geometry,
        validate_cache_layout,
        validate_query_heads,
    )
else:
    from attention_geometry import (
        HEAD_DIM,
        choose_compressed_cache_decode_strategy,
        validate_attention_geometry,
        validate_cache_layout,
        validate_query_heads,
    )
if __package__:
    from .attention_workspace import AttentionWorkspaceLayout
else:
    from attention_workspace import AttentionWorkspaceLayout
if __package__:
    from .attention_sequence_inputs import (
        validate_attention_sequences,
        allocate_sequence_metadata,
    )
else:
    from attention_sequence_inputs import (
        validate_attention_sequences,
        allocate_sequence_metadata,
    )


def _validate_tensor(
    name, value, shape, dtype, device, *, cover_t=False, contiguous=True
):
    actual = tuple(value.shape)
    shape_ok = actual == shape
    if cover_t:
        shape_ok = (
            len(actual) == len(shape)
            and actual[0] >= shape[0]
            and actual[1:] == shape[1:]
        )
    if (
        not shape_ok
        or value.dtype != dtype
        or value.device != device
        or (contiguous and not value.is_contiguous())
    ):
        prefix = "at least T rows, " if cover_t else ""
        raise ValueError(
            f"{name} must be {prefix}{shape}, {dtype}, on {device}"
            + (", contiguous" if contiguous else "")
        )


def _validate_nz_weight(name, weight):
    import torch_npu

    # These projection shapes have no NZ padding. Validate the complete logical
    # allocation using public APIs; an identical full-storage alias is safe.
    if (
        torch_npu.get_npu_format(weight) != torch_npu.Format.FRACTAL_NZ
        or weight.storage_offset() != 0
        or not weight.is_contiguous()
        or weight.untyped_storage().nbytes() != weight.numel() * weight.element_size()
    ):
        raise ValueError(
            f"{name} must span a complete E4M3 NZ allocation with zero offset; "
            "call torch_npu.npu_format_cast(weight, torch_npu.Format.FRACTAL_NZ) "
            "once before run"
        )


def available_aic_count(device):
    """Query this device's current stream, including both AIC and AIV limits."""
    import torch
    import cannbotdsl as dsl

    with torch.npu.device(device):
        info = dsl.get_platform_info(
            force_refresh=True, stream=torch.npu.current_stream(device)
        )
    aic, aiv = int(info.cube_core_num), int(info.vector_core_num)
    if not info.available or aic <= 0 or aiv < 2:
        raise RuntimeError(
            "cannot query positive AIC/AIV counts for the current stream"
        )
    return min(aic, aiv // 2)


def validate_cache_block_tables(
    block_table, extra_block_table, tokens, device, *, sequence_batches=None
):
    """Validate metadata only; return S for logical indices, zero for physical."""
    import torch

    if block_table is None and extra_block_table is None:
        return 0
    if block_table is None or extra_block_table is None:
        raise ValueError("block_table and extra_block_table must be supplied together")
    for name, table in (
        ("block_table", block_table),
        ("extra_block_table", extra_block_table),
    ):
        if (
            table.ndim != 2
            or table.shape[0] <= 0
            or table.shape[1] <= 0
            or (sequence_batches is None and tokens % table.shape[0])
        ):
            raise ValueError(
                f"{name} must be [B,pages], B>0, pages>0, T divisible by B"
            )
        _validate_tensor(name, table, tuple(table.shape), torch.int32, device)
    if block_table.shape[0] != extra_block_table.shape[0]:
        raise ValueError("block_table and extra_block_table must use the same B")
    if sequence_batches is not None and block_table.shape[0] != sequence_batches:
        raise ValueError("block tables must match the sequence metadata B")
    return tokens // int(block_table.shape[0])


def prepare_attention_arguments(
    op,
    hidden_states,
    query_norm_weight,
    kv_norm_weight,
    query_down_proj_weight,
    kv_proj_weight,
    query_up_proj_weight,
    query_down_proj_scale,
    kv_proj_scale,
    query_up_proj_scale,
    k_cache,
    extra_k_cache,
    indices_in_kvcache,
    extra_indices_in_kvcache,
    attn_sink,
    rope_cos,
    rope_sin,
    window_cache_slots,
    *,
    out=None,
    topk_length=None,
    extra_topk_length=None,
    block_table=None,
    extra_block_table=None,
    cu_seqlens_qo=None,
    seqused_q=None,
    cache_seqlens=None,
    extra_cache_seqlens=None,
    sequence_metadata=None,
    return_sequence_metadata=False,
):
    """Validate before allocation and map business tensors to the private ABI.

    This helper is host-testable on CPU with a mocked NZ-format query. The public
    operator additionally requires NPU inputs. It never copies cache contents.
    """
    import torch

    if op.mode != "attn" or op.rope_dim != 64:
        raise ValueError("public entry requires attn, rope_dim=64")
    heads = validate_query_heads(op.heads)
    validate_attention_geometry(
        k=op.expected_k,
        page_size=op.page_size,
        win_len=op.win_len,
        grid=op.grid,
        tokens=op.tokens,
    )
    q_width = heads * HEAD_DIM
    if hidden_states.ndim != 2 or hidden_states.shape[0] <= 0:
        raise ValueError("hidden_states must have shape [T,5120], T > 0")
    t, dev = int(hidden_states.shape[0]), hidden_states.device
    if dev.type == "npu":
        limit = available_aic_count(dev)
        if op.grid > limit:
            raise ValueError(
                f"grid={op.grid} exceeds the current stream AIC/AIV launch limit {limit}"
            )
    _validate_tensor("hidden_states", hidden_states, (t, 5120), torch.bfloat16, dev)
    if op.tokens is not None and t != op.tokens:
        raise ValueError(f"input T={t}, configured T={op.tokens}")
    for name, value, n in (
        ("query_norm_weight", query_norm_weight, 1280),
        ("kv_norm_weight", kv_norm_weight, 512),
    ):
        _validate_tensor(name, value, (1, n), torch.float32, dev)
    for name, value, shape in (
        ("query_down_proj_weight", query_down_proj_weight, (1280, 5120)),
        ("kv_proj_weight", kv_proj_weight, (512, 5120)),
        ("query_up_proj_weight", query_up_proj_weight, (q_width, 1280)),
    ):
        _validate_tensor(name, value, shape, torch.float8_e4m3fn, dev, contiguous=False)
        _validate_nz_weight(name, value)
    for name, value, shape in (
        ("query_down_proj_scale", query_down_proj_scale, (1280, 80, 2)),
        ("kv_proj_scale", kv_proj_scale, (512, 80, 2)),
        ("query_up_proj_scale", query_up_proj_scale, (q_width, 20, 2)),
    ):
        _validate_tensor(name, value, shape, torch.float8_e8m0fnu, dev)
    for name, value, width in (
        ("k_cache", k_cache, 544),
        ("extra_k_cache", extra_k_cache, 320),
    ):
        validate_cache_layout(
            name, value, width, page_size=op.page_size if name == "k_cache" else None
        )
        _validate_tensor(
            name, value, tuple(value.shape), torch.uint8, dev, contiguous=False
        )
    compressed_page_size = int(extra_k_cache.shape[1])
    if (
        indices_in_kvcache.ndim != 2
        or extra_indices_in_kvcache.ndim != 2
        or indices_in_kvcache.shape[1] != op.win_len
        or extra_indices_in_kvcache.shape[1] <= 0
    ):
        raise ValueError(
            "indices_in_kvcache/extra_indices_in_kvcache must be [>=T,Kw]/[>=T,Kc], Kw=win_len"
        )
    kc = int(extra_indices_in_kvcache.shape[1])
    _validate_tensor(
        "indices_in_kvcache",
        indices_in_kvcache,
        (t, op.win_len),
        torch.int32,
        dev,
        cover_t=True,
    )
    _validate_tensor(
        "extra_indices_in_kvcache",
        extra_indices_in_kvcache,
        (t, kc),
        torch.int32,
        dev,
        cover_t=True,
    )
    k = op.win_len + kc
    if k % 128 or (op.expected_k is not None and k != op.expected_k):
        raise ValueError("index K must be divisible by 128 and match configured K")
    _validate_tensor("attn_sink", attn_sink, (1, heads), torch.float32, dev)
    for name, value in (("rope_cos", rope_cos), ("rope_sin", rope_sin)):
        _validate_tensor(name, value, (t, 32), torch.float32, dev, cover_t=True)
    _validate_tensor(
        "window_cache_slots", window_cache_slots, (t,), torch.int32, dev, cover_t=True
    )
    sequence = validate_attention_sequences(
        t,
        dev,
        cu_seqlens_qo=cu_seqlens_qo,
        seqused_q=seqused_q,
        cache_seqlens=cache_seqlens,
        extra_cache_seqlens=extra_cache_seqlens,
        block_table=block_table,
        extra_block_table=extra_block_table,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        sequence_metadata=sequence_metadata,
        return_sequence_metadata=return_sequence_metadata,
    )
    validate_cache_block_tables(
        block_table,
        extra_block_table,
        t,
        dev,
        sequence_batches=None if sequence is None else sequence.batches,
    )
    for name, length in (
        ("topk_length", topk_length),
        ("extra_topk_length", extra_topk_length),
    ):
        if length is not None:
            if (
                length.ndim != 1
                or length.shape[0] <= 0
                or (sequence is None and t % length.shape[0])
            ):
                raise ValueError(f"{name} must be [B] or [T], with T divisible by B")
            _validate_tensor(name, length, tuple(length.shape), torch.int32, dev)
    if (
        sequence is None
        and topk_length is not None
        and extra_topk_length is not None
        and topk_length.shape != extra_topk_length.shape
    ):
        raise ValueError("topk_length and extra_topk_length must use the same shape")
    if out is not None:
        _validate_tensor("out", out, (t, heads, HEAD_DIM), torch.bfloat16, dev)

    selected = choose_compressed_cache_decode_strategy(
        t,
        int(extra_k_cache.shape[0]) * compressed_page_size,
        kc,
        getattr(op, "cmp_decode_strategy", "auto"),
        heads=heads,
        grid=op.grid,
        page_size=compressed_page_size,
        win_len=op.win_len,
        mode=op.mode,
        has_lengths=(
            sequence is not None
            or topk_length is not None
            or extra_topk_length is not None
        ),
    )
    op.selected_cmp_decode_strategy = selected
    op._sequence_config = sequence
    key = (
        t,
        heads,
        k,
        int(extra_k_cache.shape[0]),
        op.page_size,
        compressed_page_size,
        op.win_len,
        selected,
        dev,
        sequence is not None,
    )
    if getattr(op, "_workspace_key", None) != key:
        layout = AttentionWorkspaceLayout(
            t,
            heads,
            k,
            int(extra_k_cache.shape[0]),
            op.page_size,
            selected,
            filter_indices=sequence is not None,
            win_len=op.win_len,
            compressed_page_size=compressed_page_size,
        )
        op._workspace_storage, op._workspace = layout.allocate(dev)
        op._workspace_layout, op._workspace_key = layout, key
    if out is None:
        # Do not return a workspace-owned buffer: later calls must not silently
        # overwrite previously returned attention outputs.
        out = torch.empty((t, heads, HEAD_DIM), dtype=torch.bfloat16, device=dev)
    # Page tables and absent lengths use read-only index placeholders.
    # Specialization guards keep the corresponding device accesses disabled.
    wbt = indices_in_kvcache if block_table is None else block_table
    cbt = extra_indices_in_kvcache if extra_block_table is None else extra_block_table
    wl = indices_in_kvcache if topk_length is None else topk_length
    cl = extra_indices_in_kvcache if extra_topk_length is None else extra_topk_length
    args = [
        hidden_states,
        query_norm_weight,
        kv_norm_weight,
        query_down_proj_weight,
        kv_proj_weight,
        query_up_proj_weight,
        query_down_proj_scale,
        kv_proj_scale,
        query_up_proj_scale,
        k_cache,
        extra_k_cache,
        indices_in_kvcache,
        extra_indices_in_kvcache,
        attn_sink,
        rope_cos,
        rope_sin,
        window_cache_slots,
        wbt,
        cbt,
        wl,
        cl,
        out,
        *op._workspace_layout.arguments(op._workspace_storage),
    ]
    if sequence is not None:
        args += [
            indices_in_kvcache if value is None else value
            for value in (cu_seqlens_qo, seqused_q, cache_seqlens, extra_cache_seqlens)
        ]
        if sequence.extended_abi:
            metadata = sequence_metadata
            if sequence.query_metadata_mode == "export":
                sources = [cu_seqlens_qo, seqused_q]
                metadata = allocate_sequence_metadata(sequence, dev, sources)
            op._sequence_metadata_result = metadata
            args += [metadata.query_info]
        else:
            op._sequence_metadata_result = None
    else:
        op._sequence_metadata_result = None
    return args, out
