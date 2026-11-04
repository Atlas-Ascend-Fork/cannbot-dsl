# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# fmt: off
# Keep compact call argument groups consistent with the original FlashKDA samples.

"""FlashKDA public entry with metadata-selected staged/persistent device templates."""

__all__ = ["FlashKDA", "clear_caches", "flash_kda", "flash_kda_kernel"]

import math
from functools import lru_cache
from typing import Optional, Tuple

import cannbotdsl
import torch
from cannbotdsl import Tensor, aot, dtypes, get_platform_info
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_num
from cannbotdsl.ops.sync import onchip_rewind

from .flash_kda_metadata import (
    ABI_VERSION,
    HEADER_ABI_VERSION,
    HEADER_CORE_NUM,
    HEADER_MAGIC,
    HEADER_STATUS,
    HEADER_TEMPLATE,
    HEADER_WORDS,
    MAGIC,
    MAX_AIC_CORES,
    STATUS_OK,
    TEMPLATE_PERSISTENT,
    TEMPLATE_STAGED,
    WORKSPACE_SLOTS,
)
from . import flash_kda_staged as _staged_template
from .flash_kda_staged import (
    allocate_qk_exchange_workspace,
    allocate_stage1_workspace,
    run_staged_flash_kda,
    _compile_flash_kda as compile_staged_kernel,
    _device_block_num as staged_device_block_num,
)
from .flash_kda_persistent import (
    run_persistent_flash_kda,
    _compile_flash_kda as compile_persistent_kernel,
    _device_block_num as persistent_device_block_num,
    _identity_matrix16 as persistent_identity_matrix16,
)


CHUNK_SIZE = 64
SUPPORTED_HEAD_DIM = 128
compile_cache = getattr(aot, "compile_cache", lru_cache(maxsize=1))


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


@kernel(
    profile=cannbotdsl.ProfileSpec(
        name="flash_kda", op_type="FlashKDA",
        inputs=("gm_K", "gm_V", "gm_Q", "gm_beta", "gm_g", "gm_A_log", "gm_dt_bias", "gm_state_in", "gm_metadata",),
        outputs=("gm_O",), inouts=("gm_state_out",),
    ),
)
class flash_kda_kernel:
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
        block_num = get_block_num()
        if (
            gm_metadata[HEADER_MAGIC] == MAGIC
            and gm_metadata[HEADER_ABI_VERSION] == ABI_VERSION
            and (gm_metadata[HEADER_STATUS] == STATUS_OK)
            and (gm_metadata[HEADER_CORE_NUM] == block_num)
            and (
                gm_metadata[HEADER_TEMPLATE] == TEMPLATE_STAGED
                or gm_metadata[HEADER_TEMPLATE] == TEMPLATE_PERSISTENT
            )
        ):
            selected_template = gm_metadata[HEADER_TEMPLATE]
            if selected_template == TEMPLATE_STAGED:
                onchip_rewind(reset_sync_id=True)
                run_staged_flash_kda(
                    gm_K, gm_V, gm_Q, gm_beta, gm_g, gm_qk_exchange, gm_A_log, gm_dt_bias, gm_k_restored, gm_gamma,
                    gm_q_decayed, gm_w, gm_Mqk, gm_U, gm_O, gm_state_in, gm_state_out, gm_identity16, gm_metadata,
                    scale_value, lower_bound,
                )
            else:
                onchip_rewind(reset_sync_id=True)
                run_persistent_flash_kda(
                    gm_Q, gm_K, gm_V, gm_g, gm_beta, gm_state_in, gm_A_log, gm_dt_bias, gm_identity16, gm_qk_exchange,
                    gm_O, gm_state_out, gm_metadata, scale_value, lower_bound,
                )


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
            K, V, Q, beta, g, qk_exchange, A_log, dt_bias, kr, gamma, qd, w, mqk, u, out, state_in, state_out, identity,
            metadata, scale, lower_bound,
        )


def _sequence_spec(dtype, physical_batches, length, heads, width, stride_prefix):
    stride_0 = cannbotdsl.Dim(f"{stride_prefix}_S0")
    stride_1 = cannbotdsl.Dim(f"{stride_prefix}_S1")
    stride_2 = cannbotdsl.Dim(f"{stride_prefix}_S2")
    return cannbotdsl.TensorSpec(
        (physical_batches, heads, length, width), dtype, stride=(stride_0, stride_1, stride_2, 1),
    )


def _compile_staged_entry(entry):
    """Share the tensor contract while compiling each named AICore entry separately."""
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
    qk_spec = _sequence_spec(dtypes.bfloat16, physical_batches, storage_length, n_qk, SUPPORTED_HEAD_DIM, "QK")
    v_spec = _sequence_spec(dtypes.bfloat16, physical_batches, storage_length, n_v, SUPPORTED_HEAD_DIM, "V")
    beta_spec = _sequence_spec(dtypes.bfloat16, physical_batches, storage_length, n_v, 1, "BETA")
    output_spec = _sequence_spec(dtypes.bfloat16, physical_batches, output_length, n_v, SUPPORTED_HEAD_DIM, "OUT")
    state_spec = fake((batch, n_v, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), dtypes.float32)
    return cannbotdsl.compile(
        entry().run, qk_spec, v_spec, qk_spec, beta_spec, v_spec, fake((core_num, 2, 2, CHUNK_SIZE), dtypes.float32),
        fake((n_v,), dtypes.float32), fake((n_v, SUPPORTED_HEAD_DIM), dtypes.float32),
        fake((scratch_tokens, SUPPORTED_HEAD_DIM), dtypes.bfloat16), fake((gamma_rows, 1), dtypes.float32),
        fake((scratch_tokens, SUPPORTED_HEAD_DIM), dtypes.bfloat16),
        fake((scratch_tokens, SUPPORTED_HEAD_DIM), dtypes.bfloat16),
        fake((scratch_tokens, CHUNK_SIZE), dtypes.bfloat16),
        fake((scratch_tokens, SUPPORTED_HEAD_DIM), dtypes.bfloat16), output_spec, state_spec, state_spec,
        fake((16, 16), dtypes.float16), fake((metadata_capacity,), dtypes.int32), dtypes.float32, dtypes.float32,
        dtypes.int64,
    )


@compile_cache
def _compile_flash_kda():
    return _compile_staged_entry(FlashKDA)


def _validate_metadata(metadata, ref):
    assert metadata is not None, "metadata is required"
    assert metadata.dtype == torch.int32, "metadata must be int32"
    assert metadata.dim() == 1, "metadata must be rank-1"
    assert metadata.is_contiguous(), "metadata must be contiguous"
    assert metadata.device == ref.device, "metadata must be on the same device as q"


def _flash_kda_impl(
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
    *,
    internal_template: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate and prepare all inference layouts, then launch the selected native entry."""
    assert layout_qkv in ("TND", "BNSD", "BSND"), (
        f"layout_qkv must be TND, BNSD, or BSND, got {layout_qkv!r}"
    )
    assert initial_state is not None, "initial_state is required"
    assert initial_state.dim() == 4, "initial_state must be rank-4"
    batch, n_v, state_dv, state_dk = initial_state.shape
    assert state_dv == state_dk == SUPPORTED_HEAD_DIM, (
        f"FlashKDA only supports D={SUPPORTED_HEAD_DIM}"
    )

    if layout_qkv == "TND":
        assert q.dim() == k.dim() == v.dim() == g.dim() == 3, (
            "TND q/k/v/g must be rank-3"
        )
        storage_length, n_qk, dim = q.shape
        assert k.shape == q.shape, "q/k shape does not match TND layout"
        assert v.shape == g.shape == (storage_length, n_v, dim), (
            "v/g shape does not match TND layout"
        )
        assert beta.dim() == 2 and beta.shape == (storage_length, n_v), (
            "beta must be [T, Nv] for TND"
        )
        q_logical = torch.as_strided(
            q, (1, n_qk, storage_length, dim), (storage_length * n_qk * dim, dim, n_qk * dim, 1),
        )
        k_logical = torch.as_strided(
            k, (1, n_qk, storage_length, dim), (storage_length * n_qk * dim, dim, n_qk * dim, 1),
        )
        v_logical = torch.as_strided(v, (1, n_v, storage_length, dim), (storage_length * n_v * dim, dim, n_v * dim, 1))
        g_logical = torch.as_strided(g, (1, n_v, storage_length, dim), (storage_length * n_v * dim, dim, n_v * dim, 1))
        beta_logical = torch.as_strided(beta, (1, n_v, storage_length, 1), (storage_length * n_v, 1, n_v, 1))
        padded_length = ((storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE
        out_stage_storage = torch.empty(padded_length, n_v, dim, dtype=torch.bfloat16, device=q.device)
        out = out_stage_storage[:storage_length]
        out_stage = torch.as_strided(
            out_stage_storage, (1, n_v, padded_length, dim), (padded_length * n_v * dim, dim, n_v * dim, 1),
        )
    else:
        assert q.dim() == k.dim() == v.dim() == g.dim() == 4, (
            "BNSD/BSND q/k/v/g must be rank-4"
        )
        if layout_qkv == "BNSD":
            physical_batches, n_qk, storage_length, dim = q.shape
            assert k.shape == q.shape, "q/k shape does not match BNSD layout"
            assert v.shape == g.shape == (physical_batches, n_v, storage_length, dim), (
                "v/g shape does not match BNSD layout"
            )
            assert beta.dim() == 3 and beta.shape == (
                physical_batches,
                n_v,
                storage_length,
            ), "beta must be [B, Nv, S] for BNSD"
            q_logical, k_logical, v_logical, g_logical = q, k, v, g
            beta_logical = beta.unsqueeze(-1)
        else:
            physical_batches, storage_length, n_qk, dim = q.shape
            assert k.shape == q.shape, "q/k shape does not match BSND layout"
            assert v.shape == g.shape == (physical_batches, storage_length, n_v, dim), (
                "v/g shape does not match BSND layout"
            )
            assert beta.dim() == 3 and beta.shape == (
                physical_batches,
                storage_length,
                n_v,
            ), "beta must be [B, S, Nv] for BSND"
            q_logical = q.transpose(1, 2)
            k_logical = k.transpose(1, 2)
            v_logical = v.transpose(1, 2)
            g_logical = g.transpose(1, 2)
            beta_logical = beta.unsqueeze(-1).transpose(1, 2)
        assert physical_batches == batch, (
            "padded storage batch must match initial_state"
        )
        padded_length = ((storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE
        if layout_qkv == "BNSD":
            out_stage_storage = torch.empty(batch, n_v, padded_length, dim, dtype=torch.bfloat16, device=q.device)
            out = out_stage_storage[:, :, :storage_length, :]
            out_stage = out_stage_storage
        else:
            out_stage_storage = torch.empty(batch, padded_length, n_v, dim, dtype=torch.bfloat16, device=q.device)
            out = out_stage_storage[:, :storage_length, :, :]
            out_stage = out_stage_storage.transpose(1, 2)

    assert dim == SUPPORTED_HEAD_DIM, f"FlashKDA only supports D={SUPPORTED_HEAD_DIM}"
    assert n_qk > 0 and n_v > 0, (
        f"head counts must be positive, got Nv={n_v}, Nqk={n_qk}"
    )
    assert n_v % n_qk == 0, f"GQA requires Nv % Nqk == 0, got Nv={n_v}, Nqk={n_qk}"
    assert storage_length > 0, "storage sequence length must be positive"
    assert q.dtype == k.dtype == v.dtype == g.dtype == beta.dtype == torch.bfloat16, (
        "q/k/v/g/beta must be bf16"
    )
    assert initial_state.dtype == torch.float32, "initial_state must be fp32"
    assert A_log.dtype == torch.float32 and A_log.shape == (n_v,), (
        f"A_log must be fp32 with shape ({n_v},)"
    )
    assert dt_bias.dtype == torch.float32 and dt_bias.shape == (n_v, dim), (
        f"dt_bias must be fp32 with shape ({n_v}, {dim})"
    )
    assert math.isfinite(float(scale)), "scale must be finite"
    assert -5.0 <= float(lower_bound) <= 0.0, "lower_bound must be in [-5, 0]"
    tensors = (q, k, v, g, beta, initial_state, A_log, dt_bias)
    assert all(tensor.device == q.device for tensor in tensors), (
        "all inputs must be on the same device"
    )
    assert all(tensor.is_contiguous() for tensor in tensors), (
        "all public inputs must be contiguous"
    )
    assert metadata is not None, "metadata is required"
    assert metadata.dtype == torch.int32, "metadata must be int32"
    assert metadata.dim() == 1, "metadata must be rank-1"
    assert metadata.numel() >= HEADER_WORDS, (
        f"metadata must contain at least {HEADER_WORDS} header words"
    )
    assert metadata.is_contiguous(), "metadata must be contiguous"
    assert metadata.device == q.device, "all inputs must be on the same device"

    # Direct exports keep their own compile cache and ABI. Unified dispatch remains
    # inside the AICore kernel and never reads metadata back to the host.
    assert internal_template in (0, TEMPLATE_STAGED, TEMPLATE_PERSISTENT), (
        "invalid internal template"
    )
    if internal_template == TEMPLATE_STAGED:
        core_num = staged_device_block_num(q)
        compile_entry = compile_staged_kernel
    elif internal_template == TEMPLATE_PERSISTENT:
        core_num = persistent_device_block_num(q)
        compile_entry = compile_persistent_kernel
    else:
        core_num = _device_block_num(q)
        compile_entry = _compile_flash_kda
    qk_exchange = allocate_qk_exchange_workspace(active_block_num=core_num, ref=q)
    final_state = torch.empty(initial_state.shape, dtype=initial_state.dtype, device=initial_state.device)
    if internal_template == TEMPLATE_PERSISTENT:
        # Standalone persistent takes q/k/v and no staged scratch. Its C08 contract
        # differs from the B991 unified/staged entry even when both execute persistent.
        identity = persistent_identity_matrix16(str(q.device))
        fn = compile_entry()
        fn(
            q_logical, k_logical, v_logical, g_logical, beta_logical, initial_state, A_log, dt_bias, identity,
            qk_exchange, out_stage, final_state, metadata, float(scale), float(lower_bound), core_num,
        )
    else:
        workspace = allocate_stage1_workspace(1, 1, WORKSPACE_SLOTS, SUPPORTED_HEAD_DIM, ref=q)
        fn = compile_entry()
        fn(
            k_logical, v_logical, q_logical, beta_logical, g_logical, qk_exchange, A_log, dt_bias, workspace.K_restored,
            workspace.gamma_C, workspace.Q_decayed, workspace.W, workspace.Mqk, workspace.U_pre, out_stage,
            initial_state, final_state, workspace.identity16, metadata, float(scale), float(lower_bound), core_num,
        )
    return out, final_state


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
        scale (float): Finite query-key scaling factor.
        initial_state (torch.Tensor): Initial recurrent state, fp32, shape ``[B, Nv, D, D]``.
        A_log (torch.Tensor): Per-value-head decay coefficient, fp32, shape ``[Nv]``.
        dt_bias (torch.Tensor): Per-value-head bias, fp32, shape ``[Nv, D]``.
        lower_bound (float): Gate lower bound in ``[-5, 0]``.
        layout_qkv (str): Input layout: ``"BNSD"``, ``"BSND"``, or ``"TND"``.
        metadata (torch.Tensor): Reusable rank-1 int32 scheduling metadata returned by
            ``flash_kda_metadata()``, including its complete header.

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
    _validate_metadata(metadata, q)
    return _flash_kda_impl(q, k, v, g, beta, scale, initial_state, A_log, dt_bias, lower_bound, layout_qkv, metadata)


def clear_caches():
    _compile_flash_kda.cache_clear()


@aot.export("flash_kda")
def export_flash_kda():
    _compile_flash_kda()


_GRAPH_LIBRARY = torch.library.Library("cannbotdsl_flash_kda", "DEF")
_GRAPH_LIBRARY.define(
    "flash_kda(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float scale, Tensor "
    "initial_state, Tensor a_log, Tensor dt_bias, float lower_bound, str layout_qkv, "
    "Tensor metadata) -> (Tensor, Tensor)",
)


def _flash_kda_meta_impl(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    a_log,
    dt_bias,
    lower_bound,
    layout_qkv,
    metadata,
):
    assert layout_qkv in ("TND", "BNSD", "BSND"), (
        "layout_qkv must be TND, BNSD, or BSND"
    )
    _validate_metadata(metadata, q)
    assert initial_state.dtype == torch.float32, "initial_state must be float32"
    assert v.dtype == torch.bfloat16, "v must be bfloat16"
    sequence_axis = 2 if layout_qkv == "BNSD" else 1 if layout_qkv == "BSND" else 0
    storage_length = v.shape[sequence_axis]
    padded_shape = list(v.shape)
    padded_shape[sequence_axis] = (
        (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE * CHUNK_SIZE
    )
    storage = torch.empty(padded_shape, dtype=v.dtype, device="meta")
    if layout_qkv == "BNSD":
        out = storage[:, :, :storage_length, :]
    elif layout_qkv == "BSND":
        out = storage[:, :storage_length, :, :]
    else:
        out = storage[:storage_length]
    return (
        out,
        torch.empty(initial_state.shape, dtype=initial_state.dtype, device="meta"),
    )


@torch.library.impl(_GRAPH_LIBRARY, "flash_kda", "Meta")
def _flash_kda_meta(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    a_log,
    dt_bias,
    lower_bound,
    layout_qkv,
    metadata,
):
    return _flash_kda_meta_impl(
        q, k, v, g, beta, scale, initial_state, a_log, dt_bias, lower_bound, layout_qkv, metadata,
    )


@torch.library.impl(_GRAPH_LIBRARY, "flash_kda", "PrivateUse1")
def _flash_kda_privateuse1(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    a_log,
    dt_bias,
    lower_bound,
    layout_qkv,
    metadata,
):
    return flash_kda(q, k, v, g, beta, scale, initial_state, a_log, dt_bias, lower_bound, layout_qkv, metadata)


flash_kda_op = torch.ops.cannbotdsl_flash_kda.flash_kda


@torch._dynamo.substitute_in_graph(flash_kda, can_constant_fold_through=False)
def _flash_kda_graph(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    A_log,
    dt_bias,
    lower_bound,
    layout_qkv,
    metadata,
):
    return flash_kda_op(q, k, v, g, beta, scale, initial_state, A_log, dt_bias, lower_bound, layout_qkv, metadata)


def __getattr__(name):
    """Preserve the original module helpers after extracting the staged template."""
    return getattr(_staged_template, name)
