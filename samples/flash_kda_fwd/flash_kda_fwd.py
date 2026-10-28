# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# fmt: off
# Keep compact call argument groups consistent with the original FlashKDA samples.


"""Training forward entry with metadata-selected staged/persistent device templates."""

import math
import os
import threading
from typing import Optional, Tuple

import cannbotdsl
from cannbotdsl import Tensor, dtypes
from cannbotdsl.aot import export
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_num
from cannbotdsl.ops.sync import channel_rewind
import torch

from .flash_kda_fwd_metadata import (
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
    _device_block_num,
)
from .flash_kda_fwd_staged import (
    CHUNK_SIZE,
    GATE_MODE_FUSED_WITH_DT_BIAS,
    GATE_MODE_FUSED_WITHOUT_DT_BIAS,
    GATE_MODE_PREPROCESSED,
    SUPPORTED_HEAD_DIM,
    allocate_qk_exchange_workspace,
    allocate_stage1_workspace,
    run_staged_flash_kda_fwd,
    _tnd_state_chunk_capacity,
    _public_bnsd_tensor,
    _public_bnsd_scalar,
    _public_chunk_tensor,
    _public_chunk_state,
)
from .flash_kda_fwd_persistent import run_persistent_flash_kda_fwd


def compile_environment():
    """Use the framework's complete factory environment identity plus PID."""
    environment = tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith("CANNBOTDSL_") or name == "ASCEND_HOME_PATH"
        )
    )
    return os.getpid(), environment


@kernel(
    profile=cannbotdsl.ProfileSpec(
        name="flash_kda_fwd", op_type="FlashKDAFwd",
        inputs=("gm_K", "gm_V", "gm_Q", "gm_beta", "gm_g", "gm_A_log", "gm_dt_bias", "gm_metadata",),
        outputs=(
            "gm_O",
            "gm_state_out",
            "gm_Aqk",
            "gm_Akk",
            "gm_saved_w",
            "gm_u",
            "gm_qg",
            "gm_kg",
            "gm_v_new",
            "gm_g_cumsum",
            "gm_h",
        ),
    ),
)
class flash_kda_fwd_kernel:
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
        gm_kg: Tensor,
        gm_gamma: Tensor,
        gm_q_decayed: Tensor,
        gm_u: Tensor,
        gm_saved_w: Tensor,
        gm_qg: Tensor,
        gm_g_cumsum: Tensor,
        gm_v_new: Tensor,
        gm_h: Tensor,
        gm_O: Tensor,
        gm_state_in: Tensor,
        gm_state_out: Tensor,
        gm_identity16: Tensor,
        gm_q_norm: Tensor,
        gm_k_norm: Tensor,
        gm_q_rstd: Tensor,
        gm_k_rstd: Tensor,
        gm_beta_activated: Tensor,
        gm_Aqk: Tensor,
        gm_Akk: Tensor,
        gm_metadata: Tensor,
        scale_value: float,
        lower_bound: float,
        use_qk_l2norm_in_kernel: int,
        use_gate_in_kernel: int,
        use_beta_sigmoid_in_kernel: int,
        allow_neg_eigval: int,
        has_lower_bound: int,
        state_is_compact: int,
        save_qg: int,
        save_g_cumsum: int,
        save_intermediate_state: int,
    ):
        _block_num = get_block_num()
        if (
            gm_metadata[HEADER_MAGIC] == MAGIC
            and gm_metadata[HEADER_ABI_VERSION] == ABI_VERSION
            and (gm_metadata[HEADER_STATUS] == STATUS_OK)
            and (gm_metadata[HEADER_CORE_NUM] == get_block_num())
            and (
                gm_metadata[HEADER_TEMPLATE] == TEMPLATE_STAGED
                or gm_metadata[HEADER_TEMPLATE] == TEMPLATE_PERSISTENT
            )
        ):
            template = gm_metadata[HEADER_TEMPLATE]
            if template == TEMPLATE_STAGED:
                channel_rewind(reset_sync_id=True)
                run_staged_flash_kda_fwd(
                    gm_K, gm_V, gm_Q, gm_beta, gm_g, gm_qk_exchange, gm_A_log, gm_dt_bias, gm_kg, gm_gamma,
                    gm_q_decayed, gm_u, gm_saved_w, gm_qg, gm_g_cumsum, gm_v_new, gm_h, gm_O, gm_state_in, gm_state_out,
                    gm_identity16, gm_q_norm, gm_k_norm, gm_q_rstd, gm_k_rstd, gm_beta_activated, gm_Aqk, gm_Akk,
                    gm_metadata, scale_value, lower_bound, use_qk_l2norm_in_kernel, use_gate_in_kernel,
                    use_beta_sigmoid_in_kernel, allow_neg_eigval, has_lower_bound, state_is_compact, save_qg,
                    save_g_cumsum, save_intermediate_state,
                )
            elif template == TEMPLATE_PERSISTENT:
                channel_rewind(reset_sync_id=True)
                run_persistent_flash_kda_fwd(
                    gm_K, gm_V, gm_Q, gm_beta, gm_g, gm_qk_exchange, gm_A_log, gm_dt_bias, gm_kg, gm_gamma,
                    gm_q_decayed, gm_u, gm_saved_w, gm_qg, gm_g_cumsum, gm_v_new, gm_h, gm_O, gm_state_in, gm_state_out,
                    gm_identity16, gm_q_norm, gm_k_norm, gm_q_rstd, gm_k_rstd, gm_beta_activated, gm_Aqk, gm_Akk,
                    gm_metadata, scale_value, lower_bound, use_qk_l2norm_in_kernel, use_gate_in_kernel,
                    use_beta_sigmoid_in_kernel, allow_neg_eigval, has_lower_bound, state_is_compact, save_qg,
                    save_g_cumsum, save_intermediate_state,
                )


@kernel(
    profile=cannbotdsl.ProfileSpec(
        name="staged_flash_kda_fwd", op_type="FlashKDAFwd",
        inputs=("gm_K", "gm_V", "gm_Q", "gm_beta", "gm_g", "gm_A_log", "gm_dt_bias", "gm_metadata",),
        outputs=(
            "gm_O",
            "gm_state_out",
            "gm_Aqk",
            "gm_Akk",
            "gm_saved_w",
            "gm_u",
            "gm_qg",
            "gm_kg",
            "gm_v_new",
            "gm_g_cumsum",
            "gm_h",
        ),
    ),
)
class staged_flash_kda_fwd_kernel:
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
        gm_kg: Tensor,
        gm_gamma: Tensor,
        gm_q_decayed: Tensor,
        gm_u: Tensor,
        gm_saved_w: Tensor,
        gm_qg: Tensor,
        gm_g_cumsum: Tensor,
        gm_v_new: Tensor,
        gm_h: Tensor,
        gm_O: Tensor,
        gm_state_in: Tensor,
        gm_state_out: Tensor,
        gm_identity16: Tensor,
        gm_q_norm: Tensor,
        gm_k_norm: Tensor,
        gm_q_rstd: Tensor,
        gm_k_rstd: Tensor,
        gm_beta_activated: Tensor,
        gm_Aqk: Tensor,
        gm_Akk: Tensor,
        gm_metadata: Tensor,
        scale_value: float,
        lower_bound: float,
        use_qk_l2norm_in_kernel: int,
        use_gate_in_kernel: int,
        use_beta_sigmoid_in_kernel: int,
        allow_neg_eigval: int,
        has_lower_bound: int,
        state_is_compact: int,
        save_qg: int,
        save_g_cumsum: int,
        save_intermediate_state: int,
    ):
        if (
            gm_metadata[HEADER_MAGIC] == MAGIC
            and gm_metadata[HEADER_ABI_VERSION] == ABI_VERSION
            and (gm_metadata[HEADER_STATUS] == STATUS_OK)
            and (gm_metadata[HEADER_CORE_NUM] == get_block_num())
            and (gm_metadata[HEADER_TEMPLATE] == TEMPLATE_STAGED)
        ):
            channel_rewind(reset_sync_id=True)
            run_staged_flash_kda_fwd(
                gm_K, gm_V, gm_Q, gm_beta, gm_g, gm_qk_exchange, gm_A_log, gm_dt_bias, gm_kg, gm_gamma, gm_q_decayed,
                gm_u, gm_saved_w, gm_qg, gm_g_cumsum, gm_v_new, gm_h, gm_O, gm_state_in, gm_state_out, gm_identity16,
                gm_q_norm, gm_k_norm, gm_q_rstd, gm_k_rstd, gm_beta_activated, gm_Aqk, gm_Akk, gm_metadata, scale_value,
                lower_bound, use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval,
                has_lower_bound, state_is_compact, save_qg, save_g_cumsum, save_intermediate_state,
            )


@kernel(
    profile=cannbotdsl.ProfileSpec(
        name="persistent_flash_kda_fwd", op_type="FlashKDAFwd",
        inputs=("gm_K", "gm_V", "gm_Q", "gm_beta", "gm_g", "gm_A_log", "gm_dt_bias", "gm_metadata",),
        outputs=(
            "gm_O",
            "gm_state_out",
            "gm_Aqk",
            "gm_Akk",
            "gm_saved_w",
            "gm_u",
            "gm_qg",
            "gm_kg",
            "gm_v_new",
            "gm_g_cumsum",
            "gm_h",
        ),
    ),
)
class persistent_flash_kda_fwd_kernel:
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
        gm_kg: Tensor,
        gm_gamma: Tensor,
        gm_q_decayed: Tensor,
        gm_u: Tensor,
        gm_saved_w: Tensor,
        gm_qg: Tensor,
        gm_g_cumsum: Tensor,
        gm_v_new: Tensor,
        gm_h: Tensor,
        gm_O: Tensor,
        gm_state_in: Tensor,
        gm_state_out: Tensor,
        gm_identity16: Tensor,
        gm_q_norm: Tensor,
        gm_k_norm: Tensor,
        gm_q_rstd: Tensor,
        gm_k_rstd: Tensor,
        gm_beta_activated: Tensor,
        gm_Aqk: Tensor,
        gm_Akk: Tensor,
        gm_metadata: Tensor,
        scale_value: float,
        lower_bound: float,
        use_qk_l2norm_in_kernel: int,
        use_gate_in_kernel: int,
        use_beta_sigmoid_in_kernel: int,
        allow_neg_eigval: int,
        has_lower_bound: int,
        state_is_compact: int,
        save_qg: int,
        save_g_cumsum: int,
        save_intermediate_state: int,
    ):
        if (
            gm_metadata[HEADER_MAGIC] == MAGIC
            and gm_metadata[HEADER_ABI_VERSION] == ABI_VERSION
            and (gm_metadata[HEADER_STATUS] == STATUS_OK)
            and (gm_metadata[HEADER_CORE_NUM] == get_block_num())
            and (gm_metadata[HEADER_TEMPLATE] == TEMPLATE_PERSISTENT)
        ):
            channel_rewind(reset_sync_id=True)
            run_persistent_flash_kda_fwd(
                gm_K, gm_V, gm_Q, gm_beta, gm_g, gm_qk_exchange, gm_A_log, gm_dt_bias, gm_kg, gm_gamma, gm_q_decayed,
                gm_u, gm_saved_w, gm_qg, gm_g_cumsum, gm_v_new, gm_h, gm_O, gm_state_in, gm_state_out, gm_identity16,
                gm_q_norm, gm_k_norm, gm_q_rstd, gm_k_rstd, gm_beta_activated, gm_Aqk, gm_Akk, gm_metadata, scale_value,
                lower_bound, use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval,
                has_lower_bound, state_is_compact, save_qg, save_g_cumsum, save_intermediate_state,
            )


class FlashKDAFwd:
    """Compiled AICore entry used by :func:`flash_kda_fwd`."""

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
        kg: Tensor,
        gamma: Tensor,
        qd: Tensor,
        u: Tensor,
        saved_w: Tensor,
        qg: Tensor,
        g_cumsum: Tensor,
        v_new: Tensor,
        h: Tensor,
        out: Tensor,
        state_in: Tensor,
        state_out: Tensor,
        identity: Tensor,
        q_norm: Tensor,
        k_norm: Tensor,
        q_rstd: Tensor,
        k_rstd: Tensor,
        beta_activated: Tensor,
        Aqk: Tensor,
        Akk: Tensor,
        metadata: Tensor,
        scale: float,
        lower_bound: float,
        use_qk_l2norm_in_kernel: int,
        use_gate_in_kernel: int,
        use_beta_sigmoid_in_kernel: int,
        allow_neg_eigval: int,
        has_lower_bound: int,
        state_is_compact: int,
        save_qg: int,
        save_g_cumsum: int,
        save_intermediate_state: int,
        core_num: int,
    ):
        op = flash_kda_fwd_kernel()
        op[core_num](
            K, V, Q, beta, g, qk_exchange, A_log, dt_bias, kg, gamma, qd, u, saved_w, qg, g_cumsum, v_new, h, out,
            state_in, state_out, identity, q_norm, k_norm, q_rstd, k_rstd, beta_activated, Aqk, Akk, metadata, scale,
            lower_bound, use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval,
            has_lower_bound, state_is_compact, save_qg, save_g_cumsum, save_intermediate_state,
        )


class StagedFlashKDAFwd:
    """Compiled AICore entry used by :func:`flash_kda_fwd`."""

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
        kg: Tensor,
        gamma: Tensor,
        qd: Tensor,
        u: Tensor,
        saved_w: Tensor,
        qg: Tensor,
        g_cumsum: Tensor,
        v_new: Tensor,
        h: Tensor,
        out: Tensor,
        state_in: Tensor,
        state_out: Tensor,
        identity: Tensor,
        q_norm: Tensor,
        k_norm: Tensor,
        q_rstd: Tensor,
        k_rstd: Tensor,
        beta_activated: Tensor,
        Aqk: Tensor,
        Akk: Tensor,
        metadata: Tensor,
        scale: float,
        lower_bound: float,
        use_qk_l2norm_in_kernel: int,
        use_gate_in_kernel: int,
        use_beta_sigmoid_in_kernel: int,
        allow_neg_eigval: int,
        has_lower_bound: int,
        state_is_compact: int,
        save_qg: int,
        save_g_cumsum: int,
        save_intermediate_state: int,
        core_num: int,
    ):
        op = staged_flash_kda_fwd_kernel()
        op[core_num](
            K, V, Q, beta, g, qk_exchange, A_log, dt_bias, kg, gamma, qd, u, saved_w, qg, g_cumsum, v_new, h, out,
            state_in, state_out, identity, q_norm, k_norm, q_rstd, k_rstd, beta_activated, Aqk, Akk, metadata, scale,
            lower_bound, use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval,
            has_lower_bound, state_is_compact, save_qg, save_g_cumsum, save_intermediate_state,
        )


class PersistentFlashKDAFwd:
    """Compiled AICore entry used by :func:`flash_kda_fwd`."""

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
        kg: Tensor,
        gamma: Tensor,
        qd: Tensor,
        u: Tensor,
        saved_w: Tensor,
        qg: Tensor,
        g_cumsum: Tensor,
        v_new: Tensor,
        h: Tensor,
        out: Tensor,
        state_in: Tensor,
        state_out: Tensor,
        identity: Tensor,
        q_norm: Tensor,
        k_norm: Tensor,
        q_rstd: Tensor,
        k_rstd: Tensor,
        beta_activated: Tensor,
        Aqk: Tensor,
        Akk: Tensor,
        metadata: Tensor,
        scale: float,
        lower_bound: float,
        use_qk_l2norm_in_kernel: int,
        use_gate_in_kernel: int,
        use_beta_sigmoid_in_kernel: int,
        allow_neg_eigval: int,
        has_lower_bound: int,
        state_is_compact: int,
        save_qg: int,
        save_g_cumsum: int,
        save_intermediate_state: int,
        core_num: int,
    ):
        op = persistent_flash_kda_fwd_kernel()
        op[core_num](
            K, V, Q, beta, g, qk_exchange, A_log, dt_bias, kg, gamma, qd, u, saved_w, qg, g_cumsum, v_new, h, out,
            state_in, state_out, identity, q_norm, k_norm, q_rstd, k_rstd, beta_activated, Aqk, Akk, metadata, scale,
            lower_bound, use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval,
            has_lower_bound, state_is_compact, save_qg, save_g_cumsum, save_intermediate_state,
        )


_COMPILED_KERNEL = {}
_COMPILED_KERNEL_LOCK = threading.Lock()
_COMPILED_KERNEL_ENVIRONMENT = None


def _sequence_spec(dtype, physical_batches, length, heads, width, stride_prefix):
    minimum_stride = 0 if stride_prefix in ("RSTD", "ACT", "G_CUMSUM") else 1
    stride_0 = cannbotdsl.Dim(f"{stride_prefix}_S0", min=minimum_stride)
    stride_1 = cannbotdsl.Dim(f"{stride_prefix}_S1", min=minimum_stride)
    stride_2 = cannbotdsl.Dim(f"{stride_prefix}_S2", min=minimum_stride)
    return cannbotdsl.TensorSpec(
        (physical_batches, heads, length, width), dtype, stride=(stride_0, stride_1, stride_2, 1),
    )


def _chunk_cache_spec(dtype, batches, heads, chunks, width, stride_prefix):
    outer_stride = tuple(cannbotdsl.Dim(f"{stride_prefix}_S{i}", min=1) for i in range(3))
    return cannbotdsl.TensorSpec(
        (batches, heads, chunks, width, width), dtype, stride=(*outer_stride, width, 1),
    )


def _get_compiled_kernel(template=0):
    global _COMPILED_KERNEL, _COMPILED_KERNEL_ENVIRONMENT, _COMPILED_KERNEL_LOCK
    from cannbotdsl.aot.runtime import COLLECTOR

    current_collection = COLLECTOR.get
    environment = compile_environment()
    if (
        _COMPILED_KERNEL_ENVIRONMENT is not None
        and _COMPILED_KERNEL_ENVIRONMENT[0] != environment[0]
    ):
        _COMPILED_KERNEL = {}
        _COMPILED_KERNEL_LOCK = threading.Lock()
    with _COMPILED_KERNEL_LOCK:
        if _COMPILED_KERNEL_ENVIRONMENT != environment:
            for cached in _COMPILED_KERNEL.values():
                cached.close()
            _COMPILED_KERNEL.clear()
            _COMPILED_KERNEL_ENVIRONMENT = environment
        collecting = current_collection() is not None
        if (
            not collecting
            and template in _COMPILED_KERNEL
            and not _COMPILED_KERNEL[template].closed
        ):
            return _COMPILED_KERNEL[template]
        physical_batches = cannbotdsl.Dim("P")
        storage_length = cannbotdsl.Dim("L")
        storage_chunks = cannbotdsl.Dim("CHUNKS")
        output_length = cannbotdsl.Dim("OL")
        batch = cannbotdsl.Dim("B")
        state_batches = cannbotdsl.Dim("STATE_B")
        state_chunks = cannbotdsl.Dim("STATE_CHUNKS")
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
        rstd_spec = _sequence_spec(dtypes.float32, physical_batches, storage_length, n_qk, 1, "RSTD")
        activated_spec = _sequence_spec(dtypes.float32, physical_batches, storage_length, n_v, 1, "ACT")
        matrix_spec = _chunk_cache_spec(dtypes.bfloat16, physical_batches, n_v, storage_chunks, CHUNK_SIZE, "MATRIX")
        gate_cumsum_spec = _sequence_spec(
            dtypes.float32, physical_batches, storage_length, n_v, SUPPORTED_HEAD_DIM, "G_CUMSUM",
        )
        h_spec = _chunk_cache_spec(dtypes.float32, state_batches, n_v, state_chunks, SUPPORTED_HEAD_DIM, "H")
        runner = (FlashKDAFwd, StagedFlashKDAFwd, PersistentFlashKDAFwd)[template]()
        compiled_kernel = cannbotdsl.compile(
            runner.run, qk_spec, v_spec, qk_spec, beta_spec, v_spec, fake((core_num, 2, 2, CHUNK_SIZE), dtypes.float32),
            fake((n_v,), dtypes.float32), fake((n_v, SUPPORTED_HEAD_DIM), dtypes.float32), v_spec,
            fake((gamma_rows, 1), dtypes.float32), fake((scratch_tokens, SUPPORTED_HEAD_DIM), dtypes.bfloat16), v_spec,
            v_spec, v_spec, gate_cumsum_spec, v_spec, h_spec, output_spec, state_spec, state_spec,
            fake((16, 16), dtypes.float16), qk_spec, qk_spec, rstd_spec, rstd_spec, activated_spec, matrix_spec,
            matrix_spec, fake((metadata_capacity,), dtypes.int32), dtypes.float32, dtypes.float32, dtypes.int64,
            dtypes.int64, dtypes.int64, dtypes.int64, dtypes.int64, dtypes.int64, dtypes.int64, dtypes.int64,
            dtypes.int64, dtypes.int64,
        )
        _COMPILED_KERNEL[template] = compiled_kernel
        return compiled_kernel


def _flash_kda_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = SUPPORTED_HEAD_DIM ** (-0.5),
    initial_state: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
    layout_qkv: str = "TND",
    metadata: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    disable_recompute: bool = True,
    return_intermediate_state: bool = False,
    allow_neg_eigval: bool = False,
    output_final_state: bool = False,
    *,
    internal_template: int = 0,
) -> Tuple[torch.Tensor, ...]:
    """FlashKDAFwd forward (Flash Kimi Delta Attention).

    Args:
        q (torch.Tensor): Query, bf16, BNSD ``[B, Nqk, S, D]``, BSND ``[B, S, Nqk, D]``, or TND ``[T, Nqk, D]``.
        k (torch.Tensor): Key, bf16, with the same shape and layout as ``q``.
        v (torch.Tensor): Value, bf16, BNSD ``[B, Nv, S, D]``, BSND ``[B, S, Nv, D]``, or TND ``[T, Nv, D]``.
        g (torch.Tensor): Gate input, bf16 or fp32, with the shape and layout of ``v``.
            With gate fusion this contains raw logits; otherwise it contains log decay.
        beta (torch.Tensor): Beta input, bf16 or fp32, BNSD ``[B, Nv, S]``, BSND ``[B, S, Nv]``, or TND ``[T, Nv]``.
            With sigmoid fusion this contains logits; otherwise it contains activated beta.
        scale (float or None): Query-key scaling factor. ``None`` uses
            ``1 / sqrt(headdim_qk)``.
        initial_state (torch.Tensor): Initial recurrent state, fp32, shape ``[B, Nv, D, D]``.
        A_log (torch.Tensor or None): Per-value-head decay coefficient, fp32, shape ``[Nv]``;
            required when the gate is activated in the kernel.
        dt_bias (torch.Tensor or None): Per-value-head bias, fp32, shape ``[Nv, D]``;
            None means zero when the gate is activated in the kernel.
        lower_bound (float or None): Optional gate lower bound in ``[-5, 0]``;
            it is used only when ``use_gate_in_kernel=True``. None selects softplus.
        layout_qkv (str): Input layout: ``"BNSD"``, ``"BSND"``, or ``"TND"``.
        metadata (torch.Tensor): Reusable int32 scheduling metadata returned by ``flash_kda_fwd_metadata()``.

    Returns:
        out (torch.Tensor): Output in the same physical layout as ``v``. Padding rows are undefined.
        final_state (torch.Tensor or None): Final recurrent state when
            ``output_final_state=True``.
        q_norm, k_norm (torch.Tensor or None): BF16 normalized inputs, saved when
            ``disable_recompute`` and ``use_qk_l2norm_in_kernel`` are both True.
            BNSD/BSND inputs produce BNSD caches; TND produces TND caches.
        q_rstd, k_rstd (torch.Tensor or None): FP32 inverse norms saved with q_norm/k_norm.
        beta_activated (torch.Tensor or None): FP32 beta saved when
            ``disable_recompute`` and ``use_beta_sigmoid_in_kernel`` are both True;
            BNSD/BSND inputs produce BNSD caches and TND produces a TND cache.
        Aqk, Akk (torch.Tensor): BF16 causal QK and inverse rows in chunk-major public layout.
            Padded layouts are ``[B, Nv, num_chunks, 64, 64]`` and TND is
            ``[num_chunks, Nv, 64, 64]``.
            Metadata-invalid tokens are undefined in all sequence outputs.
        w, u, qg, kg, v_new (torch.Tensor or None): BF16 backward caches returned when
            ``disable_recompute=True``.
        g_cumsum (torch.Tensor or None): FP32 cumulative log decay for the real
            sequence length, returned when ``disable_recompute=True``. BNSD/BSND
            inputs use ``[B, Nv, S, D]`` and TND uses ``[T, Nv, D]``.
        h (torch.Tensor or None): FP32 chunk-entry states; returned only when
            ``return_intermediate_state=True``.

    Notes:
        * Currently requires ``Dk = Dv = 128``.
        * Consumed tensor inputs must be contiguous, on the same device, and have the dtypes listed above.
        * FP32 gate/beta inputs are rounded to BF16 before AICore execution.
        * ``1 <= Nqk <= Nv <= WORKSPACE_SLOTS``.
        * ``Nv % Nqk == 0``.
        * ``lower_bound`` is optional and, when provided with gate fusion, is in ``[-5, 0]``.
        * Internal output storage is padded to a multiple of 64 rows for kernel computation.
    """
    if initial_state is None:
        raise AssertionError("initial_state is required")
    if metadata is None:
        raise AssertionError("metadata is required")
    if layout_qkv not in ("TND", "BNSD", "BSND"):
        raise AssertionError(f"layout_qkv must be TND, BNSD, or BSND, got {layout_qkv!r}")
    if initial_state.dim() != 4:
        raise AssertionError("initial_state must be rank-4")
    batch, n_v, state_dv, state_dk = initial_state.shape
    if not state_dv == state_dk == SUPPORTED_HEAD_DIM:
        raise AssertionError(f"FlashKDAFwd only supports D={SUPPORTED_HEAD_DIM}")
    if layout_qkv == "TND":
        if not q.dim() == k.dim() == v.dim() == g.dim() == 3:
            raise AssertionError("TND q/k/v/g must be rank-3")
        physical_batches = 1
        storage_length, n_qk, dim = q.shape
        if k.shape != q.shape:
            raise AssertionError("q/k shape does not match TND layout")
        if not v.shape == g.shape == (storage_length, n_v, dim):
            raise AssertionError("v/g shape does not match TND layout")
        if not (beta.dim() == 2 and beta.shape == (storage_length, n_v)):
            raise AssertionError("beta must be [T, Nv] for TND")
        q_logical = torch.as_strided(
            q, (1, n_qk, storage_length, dim), (storage_length * n_qk * dim, dim, n_qk * dim, 1),
        )
        k_logical = torch.as_strided(
            k, (1, n_qk, storage_length, dim), (storage_length * n_qk * dim, dim, n_qk * dim, 1),
        )
        v_logical = torch.as_strided(v, (1, n_v, storage_length, dim), (storage_length * n_v * dim, dim, n_v * dim, 1))
        g_logical = torch.as_strided(g, (1, n_v, storage_length, dim), (storage_length * n_v * dim, dim, n_v * dim, 1))
        beta_logical = torch.as_strided(beta, (1, n_v, storage_length, 1), (storage_length * n_v, 1, n_v, 1))
        padded_length = (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE * CHUNK_SIZE
        out_stage_storage = torch.empty(padded_length, n_v, dim, dtype=torch.bfloat16, device=q.device)
        out = out_stage_storage[:storage_length]
        out_stage = torch.as_strided(
            out_stage_storage, (1, n_v, padded_length, dim), (padded_length * n_v * dim, dim, n_v * dim, 1),
        )
    else:
        if not q.dim() == k.dim() == v.dim() == g.dim() == 4:
            raise AssertionError("BNSD/BSND q/k/v/g must be rank-4")
        if layout_qkv == "BNSD":
            physical_batches, n_qk, storage_length, dim = q.shape
            if k.shape != q.shape:
                raise AssertionError("q/k shape does not match BNSD layout")
            if not v.shape == g.shape == (physical_batches, n_v, storage_length, dim):
                raise AssertionError("v/g shape does not match BNSD layout")
            if not (
                beta.dim() == 3
                and beta.shape == (physical_batches, n_v, storage_length)
            ):
                raise AssertionError("beta must be [B, Nv, S] for BNSD")
            q_logical, k_logical, v_logical, g_logical = (q, k, v, g)
            beta_logical = beta.unsqueeze(-1)
        else:
            physical_batches, storage_length, n_qk, dim = q.shape
            if k.shape != q.shape:
                raise AssertionError("q/k shape does not match BSND layout")
            if not v.shape == g.shape == (physical_batches, storage_length, n_v, dim):
                raise AssertionError("v/g shape does not match BSND layout")
            if not (
                beta.dim() == 3
                and beta.shape == (physical_batches, storage_length, n_v)
            ):
                raise AssertionError("beta must be [B, S, Nv] for BSND")
            q_logical = q.transpose(1, 2)
            k_logical = k.transpose(1, 2)
            v_logical = v.transpose(1, 2)
            g_logical = g.transpose(1, 2)
            beta_logical = beta.unsqueeze(-1).transpose(1, 2)
        if physical_batches != batch:
            raise AssertionError("padded storage batch must match initial_state")
        padded_length = (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE * CHUNK_SIZE
        if layout_qkv == "BNSD":
            out_stage_storage = torch.empty(batch, n_v, padded_length, dim, dtype=torch.bfloat16, device=q.device)
            out = out_stage_storage[:, :, :storage_length, :]
            out_stage = out_stage_storage
        else:
            out_stage_storage = torch.empty(batch, padded_length, n_v, dim, dtype=torch.bfloat16, device=q.device)
            out = out_stage_storage[:, :storage_length, :, :]
            out_stage = out_stage_storage.transpose(1, 2)
    if dim != SUPPORTED_HEAD_DIM:
        raise AssertionError(f"FlashKDAFwd only supports D={SUPPORTED_HEAD_DIM}")
    if scale is None:
        scale = dim ** (-0.5)
    if not math.isfinite(float(scale)):
        raise AssertionError("scale must be finite")
    if batch < 1:
        raise AssertionError("logical batch must be positive")
    if not 1 <= n_qk <= n_v <= WORKSPACE_SLOTS:
        raise AssertionError("heads must satisfy 1 <= Nqk <= Nv <= WORKSPACE_SLOTS")
    if n_v % n_qk != 0:
        raise AssertionError(f"GQA requires Nv % Nqk == 0, got Nv={n_v}, Nqk={n_qk}")
    if storage_length <= 0:
        raise AssertionError("storage sequence length must be positive")
    if not q.dtype == k.dtype == v.dtype == torch.bfloat16:
        raise AssertionError("q/k/v must be bf16")
    if g.dtype not in (torch.bfloat16, torch.float32):
        raise AssertionError("g must be bf16 or fp32")
    if beta.dtype not in (torch.bfloat16, torch.float32):
        raise AssertionError("beta must be bf16 or fp32")
    if initial_state.dtype != torch.float32:
        raise AssertionError("initial_state must be fp32")
    if use_gate_in_kernel:
        if A_log is None:
            raise AssertionError("A_log is required when use_gate_in_kernel=True")
        if not (A_log.dtype == torch.float32 and A_log.shape == (n_v,)):
            raise AssertionError(f"A_log must be fp32 with shape ({n_v},)")
        if dt_bias is not None:
            if not (dt_bias.dtype == torch.float32 and dt_bias.shape == (n_v, dim)):
                raise AssertionError(f"dt_bias must be fp32 with shape ({n_v}, {dim})")
    if use_gate_in_kernel and lower_bound is not None:
        if not -5.0 <= float(lower_bound) <= 0.0:
            raise AssertionError("lower_bound must be None or in [-5, 0]")
    if allow_neg_eigval:
        if not use_beta_sigmoid_in_kernel:
            raise AssertionError("allow_neg_eigval=True requires use_beta_sigmoid_in_kernel=True")
    if g.dtype != torch.bfloat16:
        g = g.to(dtype=torch.bfloat16)
        if layout_qkv == "TND":
            g_logical = torch.as_strided(
                g, (1, n_v, storage_length, dim), (storage_length * n_v * dim, dim, n_v * dim, 1),
            )
        elif layout_qkv == "BNSD":
            g_logical = g
        else:
            g_logical = g.transpose(1, 2)
    if beta.dtype != torch.bfloat16:
        beta = beta.to(dtype=torch.bfloat16)
        if layout_qkv == "TND":
            beta_logical = torch.as_strided(beta, (1, n_v, storage_length, 1), (storage_length * n_v, 1, n_v, 1))
        elif layout_qkv == "BNSD":
            beta_logical = beta.unsqueeze(-1)
        else:
            beta_logical = beta.unsqueeze(-1).transpose(1, 2)
    tensors = (q, k, v, g, beta, initial_state)
    if use_gate_in_kernel:
        tensors += (A_log,)
        if dt_bias is not None:
            tensors += (dt_bias,)
    if not all((tensor.device == q.device for tensor in tensors)):
        raise AssertionError("all inputs must be on the same device")
    if not all((tensor.is_contiguous() for tensor in tensors)):
        raise AssertionError("all public inputs must be contiguous")
    if metadata.dtype != torch.int32:
        raise AssertionError("metadata must be int32")
    if metadata.dim() != 1:
        raise AssertionError("metadata must be rank-1")
    if not metadata.is_contiguous():
        raise AssertionError("metadata must be contiguous")
    if metadata.numel() < HEADER_WORDS:
        raise AssertionError("metadata must contain the complete scheduling header")
    if metadata.device != q.device:
        raise AssertionError("all inputs must be on the same device")
    gate_input_mode = GATE_MODE_PREPROCESSED
    if use_gate_in_kernel:
        gate_input_mode = GATE_MODE_FUSED_WITH_DT_BIAS
        if dt_bias is None:
            gate_input_mode = GATE_MODE_FUSED_WITHOUT_DT_BIAS
            # The absent-bias path never reads this ABI-compatible placeholder.
            dt_bias = initial_state.view(-1)[: n_v * dim].view(n_v, dim)
    else:
        state_flat = initial_state.view(-1)
        A_log = state_flat[:n_v]
        dt_bias = state_flat[: n_v * dim].view(n_v, dim)

    def empty_with_layout(ref, shape=None, dtype=None):
        shape = ref.shape if shape is None else shape
        if internal_template == 2 and not disable_recompute and dtype == torch.float32:
            return torch.as_strided(initial_state, shape, (0,) * len(shape))
        stride = ref.stride()[: len(shape)]
        return torch.empty_strided(shape, stride, dtype=ref.dtype if dtype is None else dtype, device=q.device)

    def empty_inverse_norm(ref):
        shape = ref.shape[:-1]
        if internal_template == 2 and not disable_recompute:
            return torch.as_strided(initial_state, shape, (0,) * len(shape))
        if layout_qkv == "BNSD":
            return torch.empty(shape, dtype=torch.float32, device=q.device)
        # Keep sequence-major scalar storage for BSND/TND without the removed D stride.
        storage = torch.empty((shape[0], shape[2], shape[1]), dtype=torch.float32, device=q.device)
        return storage.transpose(1, 2)

    allocate_backward_inputs = disable_recompute or internal_template != 2
    if use_qk_l2norm_in_kernel and allocate_backward_inputs:
        q_norm = empty_with_layout(q_logical, dtype=torch.bfloat16)
        k_norm = empty_with_layout(k_logical, dtype=torch.bfloat16)
        q_rstd = empty_inverse_norm(q_logical)
        k_rstd = empty_inverse_norm(k_logical)
    else:
        q_norm, k_norm = (q_logical, k_logical)
        q_rstd = empty_inverse_norm(q_logical)
        k_rstd = q_rstd
    beta_activated = empty_with_layout(beta_logical, shape=beta_logical.shape[:-1], dtype=torch.float32)
    chunk_count = (
        _tnd_state_chunk_capacity(storage_length, initial_state.shape[0])
        if layout_qkv == "TND"
        else (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    )
    chunk_matrix_shape = (physical_batches, n_v, chunk_count, CHUNK_SIZE, CHUNK_SIZE)
    g_cumsum_shape = (physical_batches, n_v, storage_length, dim)
    def empty_chunk_cache():
        if layout_qkv == "TND":
            storage = torch.empty((chunk_count, n_v, CHUNK_SIZE, CHUNK_SIZE), dtype=torch.bfloat16, device=q.device)
            return storage.permute(1, 0, 2, 3).unsqueeze(0)
        return torch.empty(chunk_matrix_shape, dtype=torch.bfloat16, device=q.device)

    Aqk = empty_chunk_cache()
    Akk = empty_chunk_cache()
    w = (
        empty_with_layout(v_logical, dtype=torch.bfloat16)
        if allocate_backward_inputs
        else v_logical
    )
    u = (
        empty_with_layout(v_logical, dtype=torch.bfloat16)
        if allocate_backward_inputs
        else v_logical
    )
    qg = (
        empty_with_layout(v_logical, dtype=torch.bfloat16)
        if disable_recompute
        else v_logical
    )
    kg = (
        empty_with_layout(v_logical, dtype=torch.bfloat16)
        if allocate_backward_inputs
        else v_logical
    )
    v_new = (
        empty_with_layout(v_logical, dtype=torch.bfloat16)
        if allocate_backward_inputs
        else v_logical
    )
    if not allocate_backward_inputs:
        g_cumsum = torch.as_strided(initial_state, g_cumsum_shape, (0,) * (len(g_cumsum_shape) - 1) + (1,))
    elif layout_qkv == "TND":
        storage = torch.empty((storage_length, n_v, dim), dtype=torch.float32, device=q.device)
        g_cumsum = storage.transpose(0, 1).unsqueeze(0)
    else:
        g_cumsum = torch.empty(g_cumsum_shape, dtype=torch.float32, device=q.device)
    state_is_compact = int(layout_qkv == "TND")
    save_qg = int(disable_recompute)
    save_g_cumsum = int(disable_recompute or not use_gate_in_kernel)
    if internal_template == 2 and not return_intermediate_state:
        h = initial_state[:1].unsqueeze(2)
    elif state_is_compact:
        state_chunks = _tnd_state_chunk_capacity(storage_length, batch)
        storage = torch.empty(
            (state_chunks, n_v, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), dtype=torch.float32, device=q.device,
        )
        h = storage.permute(1, 0, 2, 3).unsqueeze(0)
    else:
        h = torch.empty(
            (batch, n_v, (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM,),
            dtype=torch.float32, device=q.device,
        )
    core_num = _device_block_num(q)
    workspace = allocate_stage1_workspace(1, 1, WORKSPACE_SLOTS, SUPPORTED_HEAD_DIM, ref=q)
    qk_exchange = allocate_qk_exchange_workspace(active_block_num=core_num, ref=q)
    final_state = torch.empty(initial_state.shape, dtype=initial_state.dtype, device=initial_state.device)
    fn = _get_compiled_kernel(internal_template)
    fn(
        k_logical, v_logical, q_logical, beta_logical, g_logical, qk_exchange, A_log, dt_bias, kg, workspace.gamma_C,
        workspace.Q_decayed, u, w, qg, g_cumsum, v_new, h, out_stage, initial_state, final_state, workspace.identity16,
        q_norm, k_norm, q_rstd.unsqueeze(-1), k_rstd.unsqueeze(-1), beta_activated.unsqueeze(-1), Aqk, Akk, metadata,
        float(scale), float(-5.0 if lower_bound is None else lower_bound), int(use_qk_l2norm_in_kernel),
        gate_input_mode, int(use_beta_sigmoid_in_kernel), int(allow_neg_eigval), int(lower_bound is not None),
        state_is_compact, save_qg, save_g_cumsum, int(return_intermediate_state), core_num,
    )
    q_norm_public = (
        _public_bnsd_tensor(q_norm, layout_qkv)
        if use_qk_l2norm_in_kernel and disable_recompute
        else None
    )
    k_norm_public = (
        _public_bnsd_tensor(k_norm, layout_qkv)
        if use_qk_l2norm_in_kernel and disable_recompute
        else None
    )
    q_rstd_public = (
        _public_bnsd_scalar(q_rstd, layout_qkv)
        if use_qk_l2norm_in_kernel and disable_recompute
        else None
    )
    k_rstd_public = (
        _public_bnsd_scalar(k_rstd, layout_qkv)
        if use_qk_l2norm_in_kernel and disable_recompute
        else None
    )
    beta_public = (
        _public_bnsd_scalar(beta_activated, layout_qkv)
        if use_beta_sigmoid_in_kernel and disable_recompute
        else None
    )
    token_outputs = (
        tuple((_public_bnsd_tensor(value, layout_qkv) for value in (w, u, qg, kg, v_new)))
        if disable_recompute
        else (None,) * 5
    )
    w_public, u_public, qg_public, kg_public, v_new_public = token_outputs
    g_cumsum_public = (
        _public_bnsd_tensor(g_cumsum, layout_qkv) if disable_recompute else None
    )
    h_public = _public_chunk_state(h, layout_qkv) if return_intermediate_state else None
    return (
        out,
        final_state if output_final_state else None,
        q_norm_public,
        k_norm_public,
        q_rstd_public,
        k_rstd_public,
        beta_public,
        _public_chunk_tensor(Aqk, layout_qkv, CHUNK_SIZE),
        _public_chunk_tensor(Akk, layout_qkv, CHUNK_SIZE),
        w_public if disable_recompute else None,
        u_public if disable_recompute else None,
        qg_public if disable_recompute else None,
        kg_public if disable_recompute else None,
        v_new_public if disable_recompute else None,
        g_cumsum_public if disable_recompute else None,
        h_public if return_intermediate_state else None,
    )


def flash_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = SUPPORTED_HEAD_DIM ** (-0.5),
    initial_state: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
    layout_qkv: str = "TND",
    metadata: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    disable_recompute: bool = True,
    return_intermediate_state: bool = False,
    allow_neg_eigval: bool = False,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Run training forward using the metadata-selected device template."""
    return _flash_kda_fwd_impl(
        q, k, v, g, beta, scale, initial_state, A_log, dt_bias, lower_bound, layout_qkv, metadata,
        use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, disable_recompute,
        return_intermediate_state, allow_neg_eigval, output_final_state,
    )


def clear_caches():
    global _COMPILED_KERNEL, _COMPILED_KERNEL_ENVIRONMENT, _COMPILED_KERNEL_LOCK
    environment = compile_environment()
    if (
        _COMPILED_KERNEL_ENVIRONMENT is not None
        and _COMPILED_KERNEL_ENVIRONMENT[0] != environment[0]
    ):
        _COMPILED_KERNEL = {}
        _COMPILED_KERNEL_LOCK = threading.Lock()
        _COMPILED_KERNEL_ENVIRONMENT = environment
    with _COMPILED_KERNEL_LOCK:
        for compiled_kernel in _COMPILED_KERNEL.values():
            compiled_kernel.close()
        _COMPILED_KERNEL.clear()


@export("flash_kda_fwd")
def export_flash_kda_fwd():
    """Collect the training forward compile request without running a Provider."""
    try:
        _get_compiled_kernel()
    finally:
        clear_caches()


_GRAPH_LIBRARY = torch.library.Library("cannbotdsl_flash_kda_fwd", "DEF")
_GRAPH_LIBRARY.define(
    "flash_kda_fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float? scale, "
    "Tensor initial_state, Tensor? a_log, Tensor? dt_bias, float? lower_bound, str "
    "layout_qkv, Tensor metadata, bool use_qk_l2norm_in_kernel=True, bool "
    "use_gate_in_kernel=True, bool use_beta_sigmoid_in_kernel=True, bool "
    "disable_recompute=True, bool return_intermediate_state=False, bool "
    "allow_neg_eigval=False, bool output_final_state=False) -> (Tensor, Tensor?, Tensor?,"
    " Tensor?, Tensor?, Tensor?, Tensor?, Tensor, Tensor, Tensor?, Tensor?, Tensor?, "
    "Tensor?, Tensor?, Tensor?, Tensor?)",
)


@torch.library.impl(_GRAPH_LIBRARY, "flash_kda_fwd", "Meta")
def _flash_kda_fwd_meta(
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
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    disable_recompute=True,
    return_intermediate_state=False,
    allow_neg_eigval=False,
    output_final_state=False,
):
    del g, scale, dt_bias, lower_bound, metadata
    if initial_state.dtype != torch.float32:
        raise AssertionError("initial_state must be fp32")
    if use_gate_in_kernel:
        if a_log is None:
            raise AssertionError("A_log is required when use_gate_in_kernel=True")
    if allow_neg_eigval:
        if not use_beta_sigmoid_in_kernel:
            raise AssertionError("allow_neg_eigval=True requires use_beta_sigmoid_in_kernel=True")
    storage_length = (
        q.shape[0] if layout_qkv == "TND" else q.shape[2 if layout_qkv == "BNSD" else 1]
    )
    chunk_count = (
        _tnd_state_chunk_capacity(storage_length, initial_state.shape[0])
        if layout_qkv == "TND"
        else (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    )
    if layout_qkv == "TND":
        q_bnsd_shape = (1, q.shape[1], storage_length, q.shape[2])
        v_bnsd_shape = (1, v.shape[1], storage_length, v.shape[2])
        q_public_shape = q.shape
        v_public_shape = v.shape
        state_public_shape = (
            _tnd_state_chunk_capacity(storage_length, initial_state.shape[0]),
            initial_state.shape[1],
            SUPPORTED_HEAD_DIM,
            SUPPORTED_HEAD_DIM,
        )
        chunk_matrix_shape = (chunk_count, v.shape[1], CHUNK_SIZE, CHUNK_SIZE)
    elif layout_qkv == "BNSD":
        q_bnsd_shape = q.shape
        v_bnsd_shape = v.shape
        q_public_shape = q.shape
        v_public_shape = v.shape
        state_public_shape = (
            initial_state.shape[0],
            initial_state.shape[1],
            chunk_count,
            SUPPORTED_HEAD_DIM,
            SUPPORTED_HEAD_DIM,
        )
        chunk_matrix_shape = (
            q.shape[0],
            v.shape[1],
            chunk_count,
            CHUNK_SIZE,
            CHUNK_SIZE,
        )
    else:
        q_bnsd_shape = (q.shape[0], q.shape[2], q.shape[1], q.shape[3])
        v_bnsd_shape = (v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        q_public_shape = q_bnsd_shape
        v_public_shape = v_bnsd_shape
        state_public_shape = (
            initial_state.shape[0],
            initial_state.shape[1],
            chunk_count,
            SUPPORTED_HEAD_DIM,
            SUPPORTED_HEAD_DIM,
        )
        chunk_matrix_shape = (
            q.shape[0],
            v.shape[2],
            chunk_count,
            CHUNK_SIZE,
            CHUNK_SIZE,
        )
    return (
        torch.empty_like(v, device="meta"),
        torch.empty_like(initial_state, device="meta") if output_final_state else None,
        torch.empty(q_public_shape, dtype=torch.bfloat16, device="meta")
        if use_qk_l2norm_in_kernel and disable_recompute
        else None,
        torch.empty(q_public_shape, dtype=torch.bfloat16, device="meta")
        if use_qk_l2norm_in_kernel and disable_recompute
        else None,
        torch.empty(q_public_shape[:-1], dtype=torch.float32, device="meta")
        if use_qk_l2norm_in_kernel and disable_recompute
        else None,
        torch.empty(q_public_shape[:-1], dtype=torch.float32, device="meta")
        if use_qk_l2norm_in_kernel and disable_recompute
        else None,
        torch.empty(v_public_shape[:-1], dtype=torch.float32, device="meta")
        if use_beta_sigmoid_in_kernel and disable_recompute
        else None,
        torch.empty(chunk_matrix_shape, dtype=torch.bfloat16, device="meta"),
        torch.empty(chunk_matrix_shape, dtype=torch.bfloat16, device="meta"),
        torch.empty(v_public_shape, dtype=torch.bfloat16, device="meta")
        if disable_recompute
        else None,
        torch.empty(v_public_shape, dtype=torch.bfloat16, device="meta")
        if disable_recompute
        else None,
        torch.empty(v_public_shape, dtype=torch.bfloat16, device="meta")
        if disable_recompute
        else None,
        torch.empty(v_public_shape, dtype=torch.bfloat16, device="meta")
        if disable_recompute
        else None,
        torch.empty(v_public_shape, dtype=torch.bfloat16, device="meta")
        if disable_recompute
        else None,
        torch.empty(v_public_shape, dtype=torch.float32, device="meta")
        if disable_recompute
        else None,
        torch.empty(state_public_shape, dtype=torch.float32, device="meta")
        if return_intermediate_state
        else None,
    )


@torch.library.impl(_GRAPH_LIBRARY, "flash_kda_fwd", "PrivateUse1")
def _flash_kda_fwd_privateuse1(
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
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    disable_recompute=True,
    return_intermediate_state=False,
    allow_neg_eigval=False,
    output_final_state=False,
):
    return flash_kda_fwd(
        q, k, v, g, beta, scale, initial_state, a_log, dt_bias, lower_bound, layout_qkv, metadata,
        use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, disable_recompute,
        return_intermediate_state, allow_neg_eigval, output_final_state,
    )


flash_kda_fwd_op = torch.ops.cannbotdsl_flash_kda_fwd.flash_kda_fwd


# Old network code calls the public host function directly. Under Dynamo
# (torch.compile, fullgraph) that host body is untraceable, so during
# tracing the call is substituted with the registered dispatcher op;
# eager calls keep taking the host path unchanged.


@torch._dynamo.substitute_in_graph(flash_kda_fwd, can_constant_fold_through=False)
def _flash_kda_fwd_graph(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = SUPPORTED_HEAD_DIM ** (-0.5),
    initial_state: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
    layout_qkv: str = "TND",
    metadata: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    disable_recompute: bool = True,
    return_intermediate_state: bool = False,
    allow_neg_eigval: bool = False,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, ...]:
    return flash_kda_fwd_op(
        q, k, v, g, beta, scale, initial_state, A_log, dt_bias, lower_bound, layout_qkv, metadata,
        use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, disable_recompute,
        return_intermediate_state, allow_neg_eigval, output_final_state,
    )
