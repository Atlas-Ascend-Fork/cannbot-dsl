# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""第 20 层 Indexer 的公开接口、准备对象与单 kernel 融合启动。

输入准备仅做校验、分配与零拷贝视图。公开入口默认combined PA cache，并保留分离K/scale模式，B1/S8、16 使用设备有效长度进行 KV128 调度，Metadata 仅占位。设备参数先排列 Tensor，再排列标量，公开接口顺序保持不变。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch_npu
from cannbotdsl import Dim, Tensor, TensorSpec, const_expr, dtypes, get_platform_info
from cannbotdsl import compile as dsl_compile
from cannbotdsl.aot import export
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import (
    get_block_idx,
    get_block_num,
    get_subblock_dim,
    get_subblock_id,
)
from cannbotdsl.ops.sync import cube_sync_all, global_sync_all, onchip_rewind

from . import prologue, qli
from .prologue import DecodeKProjection, ParallelQw

_COMPILED = {}


def _current_core_count(device):
    """Usable 1-AIC/2-AIV Mix blocks on the input device's current stream."""
    index = device.index
    if index is None:
        index = torch.npu.current_device()
    with torch.npu.device(index):
        info = get_platform_info(stream=torch.npu.current_stream(index))
    cores = min(int(info.cube_core_num), int(info.vector_core_num) // 2)
    if not info.available or cores < 1:
        raise RuntimeError("Indexer requires available AIC/AIV platform information")
    return cores


def _prepare_qw(
    x,
    qr,
    wqb,
    ww,
    descale_qr,
    descale_wqb,
    rope_sin,
    rope_cos,
    softmax_scale,
    q=None,
    descale_q=None,
    w=None,
):
    bound = dict(
        x=x,
        qr=qr,
        wqb=wqb,
        ww=ww,
        descale_qr=descale_qr,
        descale_wqb=descale_wqb,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        softmax_scale=softmax_scale,
        q=q,
        descale_q=descale_q,
        w=w,
    )
    prologue._check_prologue(bound)
    t = bound["t"]
    geo = prologue.Geometry(
        bound["dim"],
        bound["q_lora"],
        bound["n_heads"],
        bound["d"],
        bound["dr"],
        prologue.resolve_cube_cores(),
    )
    op = ParallelQw()
    if x.shape[0] != t:
        x = x[:t]
    q_shape = (t, 32, 64)
    d_shape = (t, 32, 2, 2)
    w_shape = (t, 32)
    for name, dest, shape, dtype in (
        ("q", q, q_shape, torch.uint8),
        ("descale_q", descale_q, d_shape, torch.uint8),
        ("w", w, w_shape, torch.float32),
    ):
        if dest is not None and (
            tuple(dest.shape) != shape
            or dest.dtype != dtype
            or dest.device != x.device
            or (not dest.is_contiguous())
        ):
            raise ValueError(f"{name} destination must be contiguous {shape} {dtype}")
    q_out = (
        q if q is not None else torch.empty(q_shape, dtype=torch.uint8, device=x.device)
    )
    d_out = (
        descale_q
        if descale_q is not None
        else torch.empty(
            d_shape,
            dtype=torch.uint8,
            device=x.device,
        )
    )
    w_out = (
        w
        if w is not None
        else torch.empty(w_shape, dtype=torch.float32, device=x.device)
    )
    # 行数由 Plan 推导：n_head_blocks * n_tiles * base_m + used_cores。
    workspace = torch.empty(
        (op.p.workspace_rows, op.p.n_heads), dtype=torch.float32, device=x.device
    )
    scale_cols = prologue.ceil_div(geo.q_lora, 64)
    if tuple(descale_qr.shape) != (t, scale_cols, 2):
        raise ValueError(
            f"descale_qr must be {(t, scale_cols, 2)}, got {tuple(descale_qr.shape)}"
        )
    if tuple(descale_wqb.shape) != (geo.n_heads * geo.d, scale_cols, 2):
        raise ValueError(
            f"descale_wqb must be {(geo.n_heads * geo.d, scale_cols, 2)}, "
            f"got {tuple(descale_wqb.shape)}"
        )
    args = (
        x.view(torch.bfloat16),
        qr.view(torch.uint8),
        wqb.view(torch.uint8),
        ww.view(torch.bfloat16),
        descale_qr.reshape(t, -1, 2).view(torch.uint8),
        descale_wqb.reshape(4096, -1, 2).view(torch.uint8),
        rope_sin,
        rope_cos,
        q_out.view(t, 2048),
        d_out.view(t, 128),
        w_out,
        workspace,
        t,
        float(softmax_scale),
    )
    return ((q_out, d_out, w_out), args, op, geo)


def _prepare_k(
    latent,
    wk,
    norm_weight,
    rope_sin,
    rope_cos,
    k_cache,
    k_scale_cache,
    cache_index,
    norm_eps,
    storage_mode,
):
    total, h, d, dr = prologue._validate_operator_geometry(
        latent, wk, rope_sin, storage_mode
    )
    if (h, d, dr) != (512, 128, 64):
        raise ValueError("Layer 20 requires H=512, D=128, Dr=64")
    prologue._validate_cache_layout(k_cache, k_scale_cache, storage_mode)
    tensors = (latent, wk, norm_weight, rope_sin, rope_cos, k_cache, cache_index)
    if not all(t.device.type == "npu" for t in tensors):
        raise ValueError("mega_full_index is NPU-only; all K inputs must be on npu")
    if len({t.device for t in tensors}) != 1:
        raise ValueError("K-side inputs must be on the same device")
    if latent.dtype != torch.bfloat16 or wk.dtype != torch.bfloat16:
        raise ValueError("latent and wk must be bfloat16")
    if norm_weight.dtype != torch.float32 or tuple(norm_weight.shape) != (d,):
        raise ValueError(f"norm_weight must be float32 [{d}]")
    if rope_sin.dtype != torch.float32 or rope_cos.dtype != torch.float32:
        raise ValueError("k_rope_sin and k_rope_cos must be float32")
    if tuple(rope_sin.shape) != tuple(rope_cos.shape):
        raise ValueError("k_rope_sin and k_rope_cos shapes must match")
    if torch_npu.get_npu_format(wk) != prologue.FRACTAL_NZ:
        raise ValueError("wk must be FRACTAL_NZ")
    if cache_index.dtype != torch.int64 or tuple(cache_index.shape) != (total,):
        raise ValueError("cache_index must be int64 [Tk]")
    projected = torch.empty((total, 128), dtype=torch.bfloat16, device=latent.device)
    tokens = (
        torch.empty((total, 96), dtype=torch.uint8, device=latent.device)
        if storage_mode == 1
        else k_scale_cache
    )
    args = (
        latent,
        wk,
        norm_weight,
        rope_sin,
        rope_cos,
        cache_index,
        k_cache,
        tokens,
        projected,
        float(norm_eps),
    )
    partials = torch.empty((256, 32), dtype=torch.float32, device=latent.device)
    return (args, (total, h, d, dr), projected, partials)


def _k_specs(storage_mode=1):
    tk = Dim("TK", min=1)
    blocks = Dim("CACHE_BLOCKS", min=1)
    s0 = Dim("CACHE_S0")
    cache_spec = (
        TensorSpec((blocks, 16, 544), dtypes.uint8, stride=(s0, 544, 1))
        if storage_mode == 1
        else TensorSpec((blocks, 128, 1, 64), dtypes.uint8, stride=(s0, 64, 64, 1))
    )
    auxiliary_spec = (
        TensorSpec((tk, 96), dtypes.uint8)
        if storage_mode == 1
        else TensorSpec(
            (blocks, 128, 1, 4), dtypes.uint8, stride=(Dim("SCALE_S0"), 4, 4, 1)
        )
    )
    return (
        TensorSpec((tk, 512), dtypes.bfloat16),
        TensorSpec((128, 512), dtypes.bfloat16, storage_format="nz"),
        TensorSpec((128,), dtypes.float32),
        TensorSpec((tk, 64), dtypes.float32),
        TensorSpec((tk, 64), dtypes.float32),
        TensorSpec((tk,), dtypes.int64),
        cache_spec,
        auxiliary_spec,
        TensorSpec((tk, 128), dtypes.bfloat16),
        dtypes.float32,
    )


def _build(config, plan, geo, kgeo, storage_mode=1):
    t, _, d, dr = kgeo
    post_type = (
        prologue.KPostQuantizer
        if storage_mode == 1
        else prologue.SeparateKPostQuantizer
    )
    post = post_type(total_rows=t, index_head_dim=d, rope_head_dim=dr)
    return (ParallelQw(), DecodeKProjection(), post, None, qli._qli_specs(*config))


def check_page_bounds(k_cache, block_table=None, cache_index=None):
    """接入侧越界校验：页号与 slot 必须落在物理 cache 范围内。

    会各触发一次 Device-to-Host 回读，因此**不要**放进 ``run`` 或公开入口的
    热路径。``mega_full_index()`` 与 ``prepare_mega_full_index()`` 只校验
    shape/dtype/stride 等元信息，不读取张量内容，也不影响 Graph 捕获；
    调用方在首次接入时调用本函数确认一次即可。

    ``block_table`` 的页号必须位于 ``[0, P)``；``cache_index`` 的每个 slot
    必须为 ``-1``（跳过）或位于 ``[0, P*128)``，``P`` 为物理 cache 页数。
    """
    pages = (
        int(block_table.shape[1]) if block_table is not None else int(k_cache.shape[0])
    )
    if block_table is not None and block_table.numel():
        page_ids = block_table.cpu()
        if int(page_ids.min()) < 0 or int(page_ids.max()) >= pages:
            raise ValueError(f"block_table page ids must be in [0, {pages})")
    if cache_index is not None and cache_index.numel():
        slots = cache_index.cpu()
        bad = ((slots < 0) & (slots != -1)) | (slots >= pages * 128)
        if bool(bad.any()):
            raise ValueError(f"cache_index slots must be -1 or in [0, {pages * 128})")


def prepare_mega_full_index(
    x,
    qr,
    wqb,
    ww,
    descale_qr,
    descale_wqb,
    q_rope_sin,
    q_rope_cos,
    latent,
    wk,
    norm_weight,
    k_rope_sin,
    k_rope_cos,
    k_cache,
    k_scale_cache,
    cache_index,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    *,
    softmax_scale,
    norm_eps,
    storage_mode=1,
    topk=512,
    quant_mode=1,
    max_seqlen_q=-1,
    mask_mode=3,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="PA_BBND",
    return_value=False,
    candidate_topk_blocks=2048,
    candidate_block_size=8,
    q=None,
    descale_q=None,
    w=None,
):
    """Prepare a reusable graph-call object from 24 public Tensor slots.

    QW destinations are optional keyword slots, matching the upstream
    Prologue QW API. No input format conversion or device copy occurs.
    """
    if float(softmax_scale) != 1.0 / 64.0:
        raise ValueError("Layer 20 requires softmax_scale=1/64")
    if float(norm_eps) != 1e-20:
        raise ValueError("Layer 20 requires norm_eps=1e-20")
    if (mask_mode, cmp_ratio) != (3, 1):
        raise ValueError(
            "Layer 20 requires right-bottom causal mask_mode=3 and cmp_ratio=1"
        )
    if storage_mode not in (0, 1):
        raise ValueError("storage_mode must be 0 or 1")
    prologue._validate_cache_layout(k_cache, k_scale_cache, storage_mode)
    fast_decode = (
        qr.shape[0]
        in (
            6,
            8,
            16,
        )
        and block_table is not None
        and ((k_cache.shape[1] * 8 if storage_mode == 1 else k_cache.shape[1]) == 128)
        and (cu_seqlens_q is not None)
        and (cu_seqlens_q.numel() == 2)
        and (seqused_q is not None)
        and (seqused_q.numel() == 1)
        and (seqused_k is not None)
        and (seqused_k.numel() == 1)
        and (cmp_residual_k is None)
    )
    if not fast_decode:
        raise ValueError("Indexer currently requires B=1, S=6/8/16 and page_size=128")
    qw_outputs, qw_args, plan, geo = _prepare_qw(
        x,
        qr,
        wqb,
        ww,
        descale_qr,
        descale_wqb,
        q_rope_sin,
        q_rope_cos,
        softmax_scale,
        q,
        descale_q,
        w,
    )
    k_args, kgeo, projected, kpartials = _prepare_k(
        latent,
        wk,
        norm_weight,
        k_rope_sin,
        k_rope_cos,
        k_cache,
        k_scale_cache,
        cache_index,
        norm_eps,
        storage_mode,
    )
    if kgeo[0] != qw_outputs[0].shape[0]:
        raise ValueError("QW and K must contain the same token rows")
    capacity_pages = (
        block_table.shape[1] if block_table is not None else k_cache.shape[0]
    )
    max_capacity = 1024 * 1024 if fast_decode else 128 * 1024
    if capacity_pages * 128 > max_capacity:
        raise ValueError(f"Logical KV capacity exceeds supported limit {max_capacity}")
    if (
        topk,
        quant_mode,
        layout_q,
        layout_k,
        candidate_topk_blocks,
        candidate_block_size,
    ) != (512, 1, "TND", "PA_BBND", 2048, 8):
        raise ValueError("Unsupported layer-20 QLI configuration")
    if storage_mode == 1:
        key_staging = torch.empty(
            (capacity_pages, 128, 1, 64), dtype=torch.uint8, device=x.device
        )
        scale_staging = torch.empty(
            (capacity_pages, 128, 1, 4), dtype=torch.uint8, device=x.device
        )
    else:
        key_staging, scale_staging = k_cache, k_scale_cache
    scale_qli = torch.as_strided(
        scale_staging,
        (scale_staging.shape[0], 128, 1, 2, 2),
        (scale_staging.stride(0), 4, 4, 2, 1),
    )
    if metadata is None:
        if not fast_decode:
            raise ValueError("Metadata is required for the general B/S path")
        metadata = torch.empty((1024,), dtype=torch.int32, device=x.device)
    if (
        metadata.dtype != torch.int32
        or tuple(metadata.shape) != (1024,)
        or (not metadata.is_contiguous())
        or (metadata.device != x.device)
    ):
        raise ValueError("metadata must be contiguous int32 [1024] on the input NPU")
    qli_stage = qli.prepare_qli(
        qw_outputs[0],
        key_staging,
        qw_outputs[2],
        qw_outputs[1],
        scale_qli,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        metadata,
        topk=topk,
        quant_mode=quant_mode,
        max_seqlen_q=max_seqlen_q,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        return_value=return_value,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        _decode_seq=qr.shape[0] if fast_decode else 0,
    )
    return PreparedFusion(
        qw_args,
        k_args,
        qli_stage,
        plan,
        geo,
        kgeo,
        qw_outputs,
        k_cache,
        projected,
        scale_qli,
        kpartials,
        fast_decode,
        key_staging,
        scale_staging,
        storage_mode,
    )


@dataclass(frozen=True)
class FusionLaunchArguments:
    """Named host groups; packing preserves the existing flat launch ABI."""

    qw: tuple
    key: tuple
    qli: tuple
    key_partials: torch.Tensor

    def pack(self):
        return (*self.qw, *self.key, *self.qli, self.key_partials)


@dataclass(frozen=True)
class FusionWorkspaces:
    """Named views retained for the lifetime of the prepared launch."""

    query_partials: torch.Tensor
    projected_key: torch.Tensor
    key_partials: torch.Tensor
    score_keys: torch.Tensor
    candidate_keys: torch.Tensor
    metadata: torch.Tensor
    quantized_key_rows: torch.Tensor
    key_staging: torch.Tensor
    scale_staging: torch.Tensor


@dataclass(frozen=True)
class FusionOutputs:
    """Public output order and cache alias are kept in one place."""

    query: torch.Tensor
    query_scale: torch.Tensor
    weight: torch.Tensor
    key_cache: torch.Tensor
    indices: torch.Tensor
    values: torch.Tensor
    candidate_indices: torch.Tensor
    candidate_length: torch.Tensor

    def pack(self):
        return tuple((getattr(self, name) for name in self.__dataclass_fields__))


class PreparedFusion:
    """持有输入引用、输出别名及 workspace，并按配置缓存融合二进制。"""

    def __init__(
        self,
        qw_args,
        k_args,
        qli_stage,
        plan,
        geo,
        kgeo,
        qw_outputs,
        k_cache,
        projected,
        scale_qli,
        kpartials,
        fast_decode,
        key_staging,
        scale_staging,
        storage_mode=1,
    ):
        self.storage_mode = storage_mode
        self.fast_decode = fast_decode
        self.qw_args = qw_args
        self.k_args = k_args
        self.qli_stage = qli_stage
        self.plan = plan
        self.geo = geo
        self.kgeo = kgeo
        self.keepalive = (
            projected,
            scale_qli,
            kpartials,
            key_staging,
            scale_staging,
            qli_stage.keepalive,
        )
        self.public_outputs = FusionOutputs(*qw_outputs, k_cache, *qli_stage.outputs)
        self.outputs = self.public_outputs.pack()
        self.launch_arguments = FusionLaunchArguments(
            tuple(qw_args),
            tuple(k_args),
            tuple(qli_stage.args),
            kpartials,
        )
        self.args = self.launch_arguments.pack()
        self.workspaces = FusionWorkspaces(
            qw_args[11],
            projected,
            kpartials,
            qli_stage.args[12],
            qli_stage.args[13],
            qli_stage.args[26],
            k_args[7] if storage_mode == 1 else None,
            key_staging,
            scale_staging,
        )

    def run(self):
        cores = _current_core_count(self.qw_args[0].device)
        if not self.fast_decode and cores != 32:
            raise RuntimeError(
                "The general Metadata path currently requires 32 Mix blocks"
            )
        # Worker count is a runtime scalar. Re-read it even for a prepared
        # object: stream quotas may change after preparation.
        args = list(self.args)
        args[len(self.qw_args) + len(self.k_args) + 19] = cores
        key = (
            self.storage_mode,
            self.plan.p.shape_key(),
            self.kgeo[1:],
            self.kgeo[0] if self.fast_decode else 0,
            self.qli_stage.config,
            self.qli_stage.args[22] > 16384,
        )
        compiled = _COMPILED.get(key)
        if compiled is None:
            qw_device, k_projection, k_post, qli_device, qli_specs = _build(
                self.qli_stage.config,
                self.plan,
                self.geo,
                self.kgeo,
                self.storage_mode,
            )
            compiled = _compile_stages(
                qw_device,
                k_projection,
                k_post,
                qli_device,
                qli_specs,
                self.geo,
                self.kgeo[0] == 1,
                self.qli_stage.config,
                self.kgeo[0] if self.fast_decode else 0,
                self.qli_stage.args[22] > 16384,
                self.storage_mode,
            )
            _COMPILED[key] = compiled
        compiled(*args)
        return self.outputs


def _compile_stages(
    qw_device,
    k_projection,
    k_post,
    qli_device,
    qli_specs,
    geo,
    splitk_decode,
    qli_config,
    fast_seq=0,
    long_k=False,
    storage_mode=1,
):
    if fast_seq:
        qw_device = ParallelQw()
        k_projection = DecodeKProjection()
        qli_device, qli_specs = qli.build_runner(
            qli_config, fast_seq, long_k, logical_cache=storage_mode == 1
        )

    @prologue.jit
    def prologue_task(
        task,
        qwx,
        qwqr,
        qwwqb,
        qwww,
        qwdqr,
        qwdwqb,
        qwrs,
        qwrc,
        qwq,
        qwdq,
        qwws,
        qwt,
        qwscale,
        klatent,
        kwk,
        kprojected,
    ):
        if task < 16:
            qw_device.run_q(
                qwqr,
                qwwqb,
                qwdqr,
                qwdwqb,
                qwrs,
                qwrc,
                qwq,
                qwdq,
                qwt,
                task,
            )
        elif task < 32:
            qw_device.run_w(qwx, qwww, qwws, qwt, qwscale, task - 16)
            if task < 20:
                cube_sync_all()
                onchip_rewind()
                k_projection.kernel(klatent, kwk, kprojected, 16, task - 16)

    @kernel
    def mega_full_index(
        qwx: Tensor,
        qwqr: Tensor,
        qwwqb: Tensor,
        qwww: Tensor,
        qwdqr: Tensor,
        qwdwqb: Tensor,
        qwrs: Tensor,
        qwrc: Tensor,
        qwq: Tensor,
        qwdq: Tensor,
        qww: Tensor,
        qwws: Tensor,
        klatent: Tensor,
        kwk: Tensor,
        knorm: Tensor,
        krs: Tensor,
        krc: Tensor,
        kindex: Tensor,
        kcache: Tensor,
        ktokens: Tensor,
        kprojected: Tensor,
        lw: Tensor,
        lqscale: Tensor,
        lcu: Tensor,
        lseqq: Tensor,
        lseqk: Tensor,
        lresidual: Tensor,
        ltable: Tensor,
        lkeyout: Tensor,
        lcandidatekeyout: Tensor,
        lcandlength: Tensor,
        lmetadata: Tensor,
        kpartials: Tensor,
        qwt,
        qwscale,
        keps,
        lquery,
        lkey,
        lkscale,
        lkeystride,
        lscalestride,
        lpartialidx,
        lpartialbits,
        lpartialcandidx,
        lpartialcandbits,
        lworkers,
        ltotalq,
        lbatch,
        ltokens,
        lmaxqtask,
        lkeyrows,
        lscalerows,
        ltaskcount,
        lfinalidx,
        lfinalbits,
        lfinalcandidx,
        lfinalcandbits,
        lreturn,
        lmerge,
        loffset,
        lmergeaddr,
    ):
        if const_expr(fast_seq != 0):
            cores = get_block_num()
            # Warm-up round uses the same task body without a loop-carried
            # task index. The remaining rounds use the identical dispatcher.
            prologue_task(
                get_block_idx(),
                qwx,
                qwqr,
                qwwqb,
                qwww,
                qwdqr,
                qwdwqb,
                qwrs,
                qwrc,
                qwq,
                qwdq,
                qwws,
                qwt,
                qwscale,
                klatent,
                kwk,
                kprojected,
            )
            global_sync_all()
            onchip_rewind()
            for task_round in qli.dsl_range(1, (32 + cores - 1) // cores, unroll=1):
                prologue_task(
                    get_block_idx() + task_round * cores,
                    qwx,
                    qwqr,
                    qwwqb,
                    qwww,
                    qwdqr,
                    qwdwqb,
                    qwrs,
                    qwrc,
                    qwq,
                    qwdq,
                    qwws,
                    qwt,
                    qwscale,
                    klatent,
                    kwk,
                    kprojected,
                )
                global_sync_all()
                onchip_rewind()
        elif get_block_idx() < 16:
            qw_device.run_main(
                qwx,
                qwqr,
                qwwqb,
                qwww,
                qwdqr,
                qwdwqb,
                qwrs,
                qwrc,
                qwq,
                qwdq,
                qww,
                qwws,
                qwt,
                qwscale,
            )
        if const_expr(storage_mode == 1 and fast_seq != 0):
            vector_id = get_block_idx() * get_subblock_dim() + get_subblock_id()
            vector_count = get_block_num() * get_subblock_dim()
            w_jobs = (fast_seq + qw_device.p.rows_red - 1) // qw_device.p.rows_red
            if vector_count >= w_jobs + fast_seq:
                if vector_id < w_jobs:
                    qw_device.run_reduce(qwws, qww, qwt)
                elif vector_id < w_jobs + fast_seq:
                    k_post(kprojected, krs, krc, knorm, kindex, ktokens, keps, w_jobs)
            else:
                qw_device.run_reduce(qwws, qww, qwt)
                k_post(kprojected, krs, krc, knorm, kindex, ktokens, keps)
            global_sync_all()
        else:
            qw_device.run_reduce(qwws, qww, qwt)
        if const_expr(storage_mode == 1):
            if const_expr(fast_seq == 0):
                k_post(kprojected, krs, krc, knorm, kindex, ktokens, keps)
                global_sync_all()
            onchip_rewind()
            prologue.publish_combined_cache(ktokens, kindex, kcache)
            global_sync_all()
            onchip_rewind()
            prologue.unpack_combined_cache(
                kcache, ltable, lseqk, lkey, lkscale, ltokens
            )
            global_sync_all()
            onchip_rewind()
        else:
            k_post(kprojected, krs, krc, knorm, kindex, kcache, ktokens, keps)
            global_sync_all()
            onchip_rewind()
        qli_device(
            lquery,
            lkey,
            lw,
            lqscale,
            lkscale,
            lkeystride,
            lscalestride,
            lcu,
            lseqq,
            lseqk,
            lresidual,
            ltable,
            lkeyout,
            lcandidatekeyout,
            lpartialidx,
            lpartialbits,
            lpartialcandidx,
            lpartialcandbits,
            lcandlength,
            lworkers,
            ltotalq,
            lbatch,
            ltokens,
            lmaxqtask,
            lkeyrows,
            lscalerows,
            lmetadata,
            ltaskcount,
            lfinalidx,
            lfinalbits,
            lfinalcandidx,
            lfinalcandbits,
            lreturn,
            lmerge,
            loffset,
            lmergeaddr,
        )

    @host
    def launch(
        qwx: Tensor,
        qwqr: Tensor,
        qwwqb: Tensor,
        qwww: Tensor,
        qwdqr: Tensor,
        qwdwqb: Tensor,
        qwrs: Tensor,
        qwrc: Tensor,
        qwq: Tensor,
        qwdq: Tensor,
        qww: Tensor,
        qwws: Tensor,
        qwt,
        qwscale,
        klatent: Tensor,
        kwk: Tensor,
        knorm: Tensor,
        krs: Tensor,
        krc: Tensor,
        kindex: Tensor,
        kcache: Tensor,
        ktokens: Tensor,
        kprojected: Tensor,
        keps,
        lquery,
        lkey,
        lw: Tensor,
        lqscale: Tensor,
        lkscale,
        lkeystride,
        lscalestride,
        lcu: Tensor,
        lseqq: Tensor,
        lseqk: Tensor,
        lresidual: Tensor,
        ltable: Tensor,
        lkeyout: Tensor,
        lcandidatekeyout: Tensor,
        lpartialidx,
        lpartialbits,
        lpartialcandidx,
        lpartialcandbits,
        lcandlength: Tensor,
        lworkers,
        ltotalq,
        lbatch,
        ltokens,
        lmaxqtask,
        lkeyrows,
        lscalerows,
        lmetadata: Tensor,
        ltaskcount,
        lfinalidx,
        lfinalbits,
        lfinalcandidx,
        lfinalcandbits,
        lreturn,
        lmerge,
        loffset,
        lmergeaddr,
        kpartials: Tensor,
    ):
        mega_full_index[lworkers](
            qwx,
            qwqr,
            qwwqb,
            qwww,
            qwdqr,
            qwdwqb,
            qwrs,
            qwrc,
            qwq,
            qwdq,
            qww,
            qwws,
            klatent,
            kwk,
            knorm,
            krs,
            krc,
            kindex,
            kcache,
            ktokens,
            kprojected,
            lw,
            lqscale,
            lcu,
            lseqq,
            lseqk,
            lresidual,
            ltable,
            lkeyout,
            lcandidatekeyout,
            lcandlength,
            lmetadata,
            kpartials,
            qwt,
            qwscale,
            keps,
            lquery,
            lkey,
            lkscale,
            lkeystride,
            lscalestride,
            lpartialidx,
            lpartialbits,
            lpartialcandidx,
            lpartialcandbits,
            lworkers,
            ltotalq,
            lbatch,
            ltokens,
            lmaxqtask,
            lkeyrows,
            lscalerows,
            ltaskcount,
            lfinalidx,
            lfinalbits,
            lfinalcandidx,
            lfinalcandbits,
            lreturn,
            lmerge,
            loffset,
            lmergeaddr,
        )

    specs = (
        *prologue._arg_specs(geo, Dim("T1", min=1), Dim("QW_WS", min=1)),
        *_k_specs(storage_mode),
        *qli_specs,
        TensorSpec((256, 32), dtypes.float32),
    )
    assert len(specs) == 61
    return dsl_compile(launch, *specs)


def mega_full_index(
    x,
    qr,
    wqb,
    ww,
    descale_qr,
    descale_wqb,
    q_rope_sin,
    q_rope_cos,
    latent,
    wk,
    norm_weight,
    k_rope_sin,
    k_rope_cos,
    k_cache,
    k_scale_cache,
    cache_index,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    *,
    softmax_scale=1.0 / 64.0,
    norm_eps=1e-20,
    storage_mode=1,
    topk=512,
    quant_mode=1,
    max_seqlen_q=-1,
    mask_mode=3,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="PA_BBND",
    return_value=False,
    candidate_topk_blocks=2048,
    candidate_block_size=8,
    q=None,
    descale_q=None,
    w=None,
):
    """Return (q, descale_q, w, k_cache, indices, values, candidates, length)."""
    return prepare_mega_full_index(
        x,
        qr,
        wqb,
        ww,
        descale_qr,
        descale_wqb,
        q_rope_sin,
        q_rope_cos,
        latent,
        wk,
        norm_weight,
        k_rope_sin,
        k_rope_cos,
        k_cache,
        k_scale_cache,
        cache_index,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        metadata,
        softmax_scale=softmax_scale,
        norm_eps=norm_eps,
        storage_mode=storage_mode,
        topk=topk,
        quant_mode=quant_mode,
        max_seqlen_q=max_seqlen_q,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        return_value=return_value,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        q=q,
        descale_q=descale_q,
        w=w,
    ).run()


@export("mega_full_index")
def export_mega_full_index():
    """Export S6/S8/S16 providers without creating tensors or running NPU work.

    Page count, page strides and effective KV length remain runtime axes.
    All providers use the same compile factory as PreparedFusion.run().
    Export both cache ABIs for the S6/S8/S16 short-K configurations.
    """
    # 字段顺序与 qli._qli_specs 一致：topk, candidate_enabled, candidate_topk,
    # mask_mode, cmp_ratio, has_cu_seqlens_q, has_seqused_q, has_seqused_k,
    # has_block_table, pa_block_size, split_count, n1, query_tile_rows,
    # elide_candidate_scores.
    config = (512, True, 2048, 3, 1, True, True, True, True, 128, 8, 32, 6, False)
    geometry = prologue.Geometry(5120, 1280, 32, 128, 64, 32)
    handles = []
    try:
        for storage_mode in (1, 0):
            for seq in (6, 8, 16):
                plan = ParallelQw()
                key_geometry = (seq, 512, 128, 64)
                stages = _build(config, plan, geometry, key_geometry, storage_mode)
                handles.append(
                    _compile_stages(
                        *stages,
                        geometry,
                        False,
                        config,
                        seq,
                        False,
                        storage_mode,
                    )
                )
    finally:
        for handle in handles:
            handle.close()
