# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Public mega_mla API, input preparation and per-stream compiled launchers."""

from __future__ import annotations

import threading

import cannbotdsl as _dsl
from cannbotdsl import dtypes
from cannbotdsl.lang.host import host

if __package__:
    from .attention_geometry import (
        COMPRESSED_DECODE_STRATEGIES,
        choose_compressed_cache_decode_strategy,
        validate_attention_geometry,
        validate_query_heads,
    )
    from .attention_inputs import prepare_attention_arguments, available_aic_count
    from .attention_sequence_filter import AttentionSequenceFilter
    from .attention_sequence_inputs import AttentionSequenceMetadata
    from .mega_mla_kernel import MegaMlaKernel
else:
    from attention_geometry import (
        COMPRESSED_DECODE_STRATEGIES,
        choose_compressed_cache_decode_strategy,
        validate_attention_geometry,
        validate_query_heads,
    )
    from attention_inputs import prepare_attention_arguments, available_aic_count
    from attention_sequence_filter import AttentionSequenceFilter
    from attention_sequence_inputs import AttentionSequenceMetadata
    from mega_mla_kernel import MegaMlaKernel


class MegaMlaOp:
    MODEL_ATTENTION_K = 2176  # 128 window + 2048 compressed, model default

    def __init__(
        self,
        k=None,
        page_size=128,
        win_len=128,
        grid=32,
        mode="attn",
        rope_dim=64,
        tokens=None,
        gather_batch=32,
        *,
        heads=16,
        cmp_decode_strategy="auto",
    ):
        self.heads = validate_query_heads(heads)
        validate_attention_geometry(
            k=k, page_size=page_size, win_len=win_len, grid=grid, tokens=tokens
        )
        self.expected_k = k
        self.page_size = page_size
        self.win_len = win_len
        self.grid = grid
        self.mode = mode
        self.rope_dim = rope_dim
        self.tokens = tokens
        self.gather_batch = gather_batch
        if cmp_decode_strategy not in COMPRESSED_DECODE_STRATEGIES:
            raise ValueError("cmp_decode_strategy must be auto, predecode or indexed")
        self.cmp_decode_strategy = cmp_decode_strategy
        self.selected_cmp_decode_strategy = None
        self.kernel = None
        self._workspace_key = None
        self._workspace = None
        self._workspace_storage = None
        self._workspace_layout = None
        self._sequence_config = None
        self._compiled_launches = {}

    def run(
        self,
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
        """Return BF16 attention [T,heads,512], updating k_cache in place.

        return_sequence_metadata=True also returns an independent device
        snapshot. Later layers pass sequence_metadata instead of cu/used_q;
        KV lengths, indices, tables and slots remain layer-specific.
        rope_cos/rope_sin contain 32 independent FP32 coefficients for 64 RoPE features.
        Paired block tables [B,pages] select logical indices. cu_seqlens_qo
        optionally partitions packed T into unequal query spans; otherwise
        T=B*S and token t uses row t//S. seqused_q limits each span's active
        prefix; cache_seqlens/extra_cache_seqlens limit request-local cache indices.
        TopK lengths limit index columns independently of those KV lengths.
        Without tables indices remain physical (cache lengths are then rejected).
        window_cache_slots always contains physical window-cache write locations.
        Workspace storage is reused; an instance must not run concurrently on
        different streams. Warm up before graph capture or device profiling.
        """
        if hidden_states.device.type != "npu":
            raise ValueError("public run requires NPU tensors")
        args, result = prepare_attention_arguments(
            self,
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
            out=out,
            topk_length=topk_length,
            extra_topk_length=extra_topk_length,
            block_table=block_table,
            extra_block_table=extra_block_table,
            cu_seqlens_qo=cu_seqlens_qo,
            seqused_q=seqused_q,
            cache_seqlens=cache_seqlens,
            extra_cache_seqlens=extra_cache_seqlens,
            sequence_metadata=sequence_metadata,
            return_sequence_metadata=return_sequence_metadata,
        )
        self._compile_and_run(
            *args,
            topk_length=topk_length,
            extra_topk_length=extra_topk_length,
            sequence_config=self._sequence_config,
            cache_batch_stride=(
                0
                if block_table is None or self._sequence_config is not None
                else int(hidden_states.shape[0]) // int(block_table.shape[0])
            ),
        )
        if return_sequence_metadata:
            return result, self._sequence_metadata_result
        return result

    def _compile_and_run(
        self,
        *args,
        topk_length=None,
        extra_topk_length=None,
        cache_batch_stride=0,
        sequence_config=None,
    ):
        """Specialize K/T from separate window/compressed index tensors.

        ``k`` is optional (a common model configuration is 2176); a supplied value is
        checked against the sum of runtime index widths rather than silently used.
        A fresh kernel object also avoids carrying JIT SSA state across calls.
        Optional int32 NPU lengths have shape [B]. With T=B*S, consecutive
        S queries share one length, matching FlashMLA decode's batch contract.
        Each absent length defaults to that route's index capacity.
        """
        expected_args = (
            27
            if sequence_config is None
            else 32
            if sequence_config.extended_abi
            else 31
        )
        if len(args) != expected_args:
            raise ValueError(
                f"expected {expected_args} tensors (business, output, workspace), got {len(args)}"
            )
        if len(args[11].shape) != 2 or len(args[12].shape) != 2:
            raise ValueError(
                "indices_in_kvcache and extra_indices_in_kvcache must have shape [T, K_route]"
            )
        if int(args[11].shape[-1]) != self.win_len:
            raise ValueError(
                f"indices_in_kvcache width must equal win_len={self.win_len}"
            )
        runtime_k = int(args[11].shape[-1]) + int(args[12].shape[-1])
        runtime_t = int(args[0].shape[0])
        if int(args[11].shape[0]) < runtime_t or int(args[12].shape[0]) < runtime_t:
            raise ValueError("both index tensors must cover input T")
        if self.expected_k is not None and runtime_k != self.expected_k:
            raise ValueError(f"index K={runtime_k}, configured K={self.expected_k}")
        if self.tokens is not None and runtime_t != self.tokens:
            raise ValueError(f"input T={runtime_t}, configured T={self.tokens}")
        if runtime_k <= self.win_len or runtime_k % 128:
            raise ValueError("K must exceed window length and be divisible by 128")
        import torch

        strides = []
        for name, length in (
            ("topk_length", topk_length),
            ("extra_topk_length", extra_topk_length),
        ):
            if sequence_config is not None or length is None:
                strides.append(0)
            else:
                if (
                    length.dtype != torch.int32
                    or length.ndim != 1
                    or length.shape[0] <= 0
                    or runtime_t % length.shape[0]
                    or not length.is_contiguous()
                    or length.device != args[0].device
                ):
                    raise ValueError(
                        f"{name} must be contiguous int32 [B] on the input device, with T divisible by B"
                    )
                strides.append(runtime_t // int(length.shape[0]))
        if (
            sequence_config is None
            and topk_length is not None
            and extra_topk_length is not None
            and topk_length.shape != extra_topk_length.shape
        ):
            raise ValueError("topk_length and extra_topk_length must use the same B")
        # Compile a fresh receiver only for a new ABI/configuration. Reusing
        # the compiled callable avoids tracing receiver fields that now carry SSA.
        launch_args = list(args)
        compressed_page_size = int(args[10].shape[1])
        schema = tuple(
            (
                tuple(value.shape),
                value.dtype,
                tuple(value.stride()) if i in (9, 10) else None,
            )
            for i, value in enumerate(launch_args)
        )
        selected = choose_compressed_cache_decode_strategy(
            runtime_t,
            int(args[10].shape[0]) * compressed_page_size,
            runtime_k - self.win_len,
            self.cmp_decode_strategy,
            heads=self.heads,
            grid=self.grid,
            page_size=compressed_page_size,
            win_len=self.win_len,
            mode=self.mode,
            has_lengths=(sequence_config is not None or any(strides)),
        )
        self.selected_cmp_decode_strategy = selected
        key = (
            runtime_t,
            runtime_k,
            self.heads,
            self.page_size,
            self.win_len,
            self.grid,
            self.mode,
            self.rope_dim,
            self.gather_batch,
            tuple(strides),
            selected,
            cache_batch_stride,
            schema,
            None if sequence_config is None else sequence_config.signature,
        )
        compiled = self._compiled_launches.get(key)
        if compiled is None:
            self.kernel = MegaMlaKernel(
                runtime_k,
                self.page_size,
                self.win_len,
                mode=self.mode,
                rope_dim=self.rope_dim,
                tokens=runtime_t,
                gather_batch=self.gather_batch,
                grid=self.grid,
                win_length_stride=strides[0],
                cmp_length_stride=strides[1],
                heads=self.heads,
                cmp_decode_strategy=selected,
                cache_batch_stride=cache_batch_stride,
                workspace_layout=self._workspace_layout,
                compressed_page_size=compressed_page_size,
            )
            if sequence_config is not None:
                self.kernel.sequence_filter = AttentionSequenceFilter(
                    sequence_config,
                    self.page_size,
                    self.win_len,
                    runtime_k - self.win_len,
                    compressed_page_size=compressed_page_size,
                )
            dtype_map = {
                torch.float32: dtypes.float32,
                torch.bfloat16: dtypes.bfloat16,
                torch.int8: dtypes.int8,
                torch.uint8: dtypes.uint8,
                torch.uint64: dtypes.uint64,
                torch.uint32: dtypes.uint32,
                torch.int32: dtypes.int32,
                torch.float8_e4m3fn: dtypes.float8_e4m3fn,
                torch.float8_e8m0fnu: dtypes.float8_e8m0,
            }
            specs = [
                _dsl.TensorSpec(
                    tuple(value.shape),
                    dtype_map[value.dtype],
                    stride=tuple(value.stride()) if i in (9, 10) else None,
                    storage_format="nz" if i in (3, 4, 5) else "nd",
                )
                for i, value in enumerate(launch_args)
            ]
            # A new specialization also needs a fresh host receiver: a prior
            # trace may have placed SSA fields on the original @host object.
            launch_owner = MegaMlaOp(
                runtime_k,
                self.page_size,
                self.win_len,
                self.grid,
                self.mode,
                self.rope_dim,
                runtime_t,
                self.gather_batch,
                heads=self.heads,
                cmp_decode_strategy=selected,
            )
            launch_owner.kernel = self.kernel
            entry = (
                launch_owner._launch_mega_mla
                if sequence_config is None
                else launch_owner._launch_mega_mla_with_metadata
                if sequence_config.extended_abi
                else launch_owner._launch_mega_mla_with_seqlens
            )
            compiled = _dsl.compile(entry, *specs)
            self._compiled_launches[key] = compiled
        return compiled(*launch_args)

    @host
    def _launch_mega_mla(
        self,
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
        block_table,
        extra_block_table,
        topk_length,
        extra_topk_length,
        out,
        workspace,
        workspace_bf16,
        workspace_fp8,
        workspace_e8m0,
        workspace_int32,
    ):
        self.kernel.mega_mla[self.grid](
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
            block_table,
            extra_block_table,
            topk_length,
            extra_topk_length,
            out,
            workspace,
            workspace_bf16,
            workspace_fp8,
            workspace_e8m0,
            workspace_int32,
        )

    @host
    def _launch_mega_mla_with_seqlens(
        self,
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
        block_table,
        extra_block_table,
        topk_length,
        extra_topk_length,
        out,
        workspace,
        workspace_bf16,
        workspace_fp8,
        workspace_e8m0,
        workspace_int32,
        cu_seqlens_qo,
        seqused_q,
        cache_seqlens,
        extra_cache_seqlens,
    ):
        self.kernel.mega_mla_with_seqlens[self.grid](
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
            block_table,
            extra_block_table,
            topk_length,
            extra_topk_length,
            out,
            workspace,
            workspace_bf16,
            workspace_fp8,
            workspace_e8m0,
            workspace_int32,
            cu_seqlens_qo,
            seqused_q,
            cache_seqlens,
            extra_cache_seqlens,
        )

    @host
    def _launch_mega_mla_with_metadata(
        self,
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
        block_table,
        extra_block_table,
        topk_length,
        extra_topk_length,
        out,
        workspace,
        workspace_bf16,
        workspace_fp8,
        workspace_e8m0,
        workspace_int32,
        cu_seqlens_qo,
        seqused_q,
        cache_seqlens,
        extra_cache_seqlens,
        query_metadata,
    ):
        self.kernel.mega_mla_with_metadata[self.grid](
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
            block_table,
            extra_block_table,
            topk_length,
            extra_topk_length,
            out,
            workspace,
            workspace_bf16,
            workspace_fp8,
            workspace_e8m0,
            workspace_int32,
            cu_seqlens_qo,
            seqused_q,
            cache_seqlens,
            extra_cache_seqlens,
            query_metadata,
        )


# Compatibility alias for existing four-operator wheel callers.
MegaL4CubeOp = MegaMlaOp


_FUNCTION_STATE = threading.local()


def _function_context(hidden_states, grid):
    import torch

    if hidden_states.device.type != "npu":
        raise ValueError("mega_mla requires NPU tensors")
    stream = torch.npu.current_stream(hidden_states.device)
    if grid is None:
        grid = available_aic_count(hidden_states.device)
    return hidden_states.device, stream, grid


def _function_tensor_schema(tensors):
    return tuple(
        None
        if value is None
        else (tuple(value.shape), value.dtype, tuple(value.stride()))
        for value in tensors
    )


def clear_caches():
    """Release this thread's function launchers after their streams finish.

    Dispose captured graphs using these launchers before clearing the cache.
    """
    launchers = getattr(_FUNCTION_STATE, "launchers", None)
    if launchers:
        streams = {key[:2]: stream for key, (stream, _) in launchers.items()}
        for stream in streams.values():
            stream.synchronize()
        launchers.clear()
    programs = getattr(_FUNCTION_STATE, "programs", None)
    if programs is not None:
        programs.clear()


def mega_mla(
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
    grid=None,
    cmp_decode_strategy="auto",
):
    """Return BF16 [T,H,512] attention and update the window cache in place.

    Infer H, page size and index widths from the input tensors. An omitted grid
    uses the current stream's AIC/AIV pairing limit. Other input/metadata contracts match
    MegaMlaOp.run. Private workspaces and compiled launchers are
    cached per thread/device/stream and tensor schema; values and addresses are
    read anew on every call. Each call without out returns independent storage.
    Warm up fixed schemas before graph capture or device profiling.
    """
    import torch

    inputs = (
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
    )
    if any(not isinstance(value, torch.Tensor) for value in inputs):
        raise ValueError("all business inputs must be torch tensors")
    for name, value, rank in (
        ("hidden_states", hidden_states, 2),
        ("k_cache", k_cache, 4),
        ("extra_k_cache", extra_k_cache, 4),
        ("indices_in_kvcache", indices_in_kvcache, 2),
        ("extra_indices_in_kvcache", extra_indices_in_kvcache, 2),
        ("attn_sink", attn_sink, 2),
    ):
        if value.ndim != rank:
            raise ValueError(f"{name} must have rank {rank}")
    heads = validate_query_heads(int(attn_sink.shape[1]))
    if cmp_decode_strategy not in COMPRESSED_DECODE_STRATEGIES:
        raise ValueError("cmp_decode_strategy must be auto, predecode or indexed")
    if not isinstance(return_sequence_metadata, bool):
        raise ValueError("return_sequence_metadata must be bool")
    if sequence_metadata is not None and not isinstance(
        sequence_metadata, AttentionSequenceMetadata
    ):
        raise ValueError(
            "sequence_metadata must be an AttentionSequenceMetadata result"
        )
    options = dict(
        out=out,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        block_table=block_table,
        extra_block_table=extra_block_table,
        cu_seqlens_qo=cu_seqlens_qo,
        seqused_q=seqused_q,
        cache_seqlens=cache_seqlens,
        extra_cache_seqlens=extra_cache_seqlens,
    )
    for name, value in options.items():
        if value is not None and not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a torch tensor or None")
    device, stream, grid = _function_context(hidden_states, grid)
    page_size, win_len = int(k_cache.shape[1]), int(indices_in_kvcache.shape[1])
    validate_attention_geometry(
        k=win_len + int(extra_indices_in_kvcache.shape[1]),
        page_size=page_size,
        win_len=win_len,
        grid=grid,
        tokens=int(hidden_states.shape[0]),
    )
    key = (
        device,
        int(stream.npu_stream),
        grid,
        cmp_decode_strategy,
        return_sequence_metadata,
        None
        if sequence_metadata is None
        else (sequence_metadata.tokens, sequence_metadata.batches),
        _function_tensor_schema((*inputs, *options.values())),
    )
    launchers = getattr(_FUNCTION_STATE, "launchers", None)
    if launchers is None:
        launchers = _FUNCTION_STATE.launchers = {}
    cached = launchers.get(key)
    launcher = (
        cached[1]
        if cached is not None
        else MegaMlaOp(
            page_size=page_size,
            win_len=win_len,
            grid=grid,
            heads=heads,
            cmp_decode_strategy=cmp_decode_strategy,
        )
    )
    if cached is None:
        programs = getattr(_FUNCTION_STATE, "programs", None)
        if programs is None:
            programs = _FUNCTION_STATE.programs = {}
        # Programs bind runtime tensor arguments; only workspace storage is
        # stream-local. A warmed specialization can also launch during capture
        # on another stream without compiling inside the capture region.
        launcher._compiled_launches = programs.setdefault(device, {})
    with torch.npu.device(device):
        result = launcher.run(
            *inputs,
            **options,
            sequence_metadata=sequence_metadata,
            return_sequence_metadata=return_sequence_metadata,
        )
    if cached is None:
        # Keep the stream alive with its private workspace and compiled programs.
        launchers[key] = (stream, launcher)
    return result
