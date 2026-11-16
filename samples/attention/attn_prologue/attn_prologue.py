# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""支持 token 数泛化的 MXFP8 attention prologue。

小 T 对 QA/KV 使用 split-K，在归一化前合并 FP32 部分和；大 T 使用有界的
token 分块，QB 按 token 分块和 head 分配任务。权重预先转换为 FRACTAL_NZ，
跨核数据通过 GM 和生产者发布同步传递。DSL 根据每次调用的 T 进行编译特化。
FixPipe 使用补齐的物理尾块，公开输出和 cache 更新仍使用逻辑 T。
"""

from __future__ import annotations

import ctypes
import math
import numbers

import torch
import torch_npu

import cannbotdsl
from cannbotdsl import (
    TensorSpec,
    dtypes,
    host,
)


from .attn_prologue_tiling import (
    ACL_MEMCPY_DEVICE_TO_HOST,
    AttnPrologueTiling,
    DECODE_TOKEN_LIMIT,
    E4M3_MAX,
    E8M0_BIAS,
    FIXPIPE_DUAL_DST,
    FP32_BYTES,
    FP32_EXP_BIAS,
    FRACTAL_M,
    FRACTAL_NZ,
    MAX_COMPILED_KERNELS,
    MX_GROUP,
    MX_PAIR,
    MX_PAIR_WIDTH,
    ND_FORMATS,
    NZ_C0_1B,
    NZ_M_FRAC,
    PANEL_BYTES,
    PANEL_K_FRACTALS,
    PANEL_N_FRACTALS,
    PANEL_OUTPUT_WIDTH,
    PANEL_RANK_WIDTH,
    PANEL_REDUCTION_WINDOWS,
    RESIDENT_GEOMETRIES,
    RESIDENT_GEOMETRY,
    SUPPORTED_HEAD_COUNTS,
    TOKEN_RANGE_UPPERS,
    VL,
    _ceil_div,
    _kv_row_bytes,
    _token_range_upper,
)

from .attn_prologue_kernel import (
    AttnPrologueKernel,
    _CACHE_INDEX_ARG,
    _COMPILED_KERNELS,
    _COMPILE_LOCK,
    _KV_CACHE_ARG,
    _TORCH_TO_DSL_DTYPE,
)


def _dynamic_index_spec(tiling):
    """按模板上界约束实际 T 的 cache_index 契约，不枚举具体 T。"""
    return TensorSpec(
        (cannbotdsl.Dim("TOKENS", 1, tiling.token_count, 1), 1), dtypes.int64
    )


def _dynamic_cache_spec(tiling):
    """KV cache row 数随调用方的容量变化，行维必须动态化才能跨 T 复用。"""
    return TensorSpec(
        (cannbotdsl.Dim("CACHE_ROWS", 1), _kv_row_bytes(tiling.head_size)),
        dtypes.uint8,
    )


class _Launcher:
    def __init__(self, tiling: AttnPrologueTiling, dynamic_tokens: bool = False):
        self.tiling = tiling
        self.dynamic_tokens = dynamic_tokens

    @host
    def run(
        self,
        gm_x,
        gm_dsx,
        gm_wqa,
        gm_dswqa,
        gm_wqb,
        gm_dswqb,
        gm_wkv,
        gm_dswkv,
        gm_gamma_qr,
        gm_gamma_kv,
        gm_sin,
        gm_cos,
        gm_index,
        gm_cache,
        gm_q,
        gm_qr,
        gm_dqr2d,
        gm_dqr3d,
        gm_ws_qa,
        gm_ws_kv,
        norm_eps: dtypes.float32,
        gm_qr_bf16,
    ):
        op = AttnPrologueKernel(self.tiling)
        op[self.tiling.block_dim](
            gm_x,
            gm_dsx,
            gm_wqa,
            gm_dswqa,
            gm_wqb,
            gm_dswqb,
            gm_wkv,
            gm_dswkv,
            gm_gamma_qr,
            gm_gamma_kv,
            gm_sin,
            gm_cos,
            gm_index,
            gm_cache,
            gm_q,
            gm_qr,
            gm_dqr2d,
            gm_dqr3d,
            gm_ws_qa,
            gm_ws_kv,
            norm_eps,
            gm_qr_bf16,
        )

    def compile_cached(self, args):
        """复用静态分块和张量契约相同的编译产物。

        norm_eps 保持为运行时 FP32 参数。缓存键和 TensorSpec 都必须包含 NZ
        信息，仅凭逻辑形状无法区分权重布局。缓存项不保存输入张量或交换区缓冲。
        """
        contracts, specs = [], []
        for position, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                if self.dynamic_tokens and position == _CACHE_INDEX_ARG:
                    # 实际 T 由 cache_index 的行数在运行时决定；同一 range 内
                    # 所有 T 共用 (tiling, 其余张量契约) 这一份编译产物。
                    contracts.append(("dynT", self.tiling.token_count))
                    specs.append(_dynamic_index_spec(self.tiling))
                    continue
                if self.dynamic_tokens and position == _KV_CACHE_ARG:
                    # cache 容量由模型固定，但测试与不同调用方会给出不同的
                    # slot 数；行维动态化后同一 range 不会因容量变化重编译。
                    contracts.append(("dynCACHE", _kv_row_bytes(self.tiling.head_size)))
                    specs.append(_dynamic_cache_spec(self.tiling))
                    continue
                shape, stride = tuple(arg.shape), tuple(arg.stride())
                storage_format = (
                    "nz" if torch_npu.get_npu_format(arg) == FRACTAL_NZ else "nd"
                )
                contracts.append(
                    (shape, stride, arg.dtype, str(arg.device), storage_format)
                )
                if arg.dtype not in _TORCH_TO_DSL_DTYPE:
                    raise KeyError(arg.dtype)
                specs.append(
                    TensorSpec(
                        shape,
                        _TORCH_TO_DSL_DTYPE[arg.dtype],
                        stride=stride,
                        storage_format=storage_format,
                    )
                )
            else:
                contracts.append(dtypes.float32)
                specs.append(dtypes.float32)
        key = (self.tiling, tuple(contracts))
        with _COMPILE_LOCK:
            compiled = _COMPILED_KERNELS.get(key)
            if compiled is None:
                compiled = cannbotdsl.compile(self.run, *specs)
                _COMPILED_KERNELS[key] = compiled
                if len(_COMPILED_KERNELS) > MAX_COMPILED_KERNELS:
                    # 被淘汰的编译产物仍可能由 Prepared 对象持有，不能提前关闭。
                    _COMPILED_KERNELS.popitem(last=False)
            _COMPILED_KERNELS.move_to_end(key)
        return compiled


def _as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """以 int8 视图传递 FP8 原始字节，不转换数值，并保留 NZ 存储标记。"""
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        return tensor.view(torch.int8)
    return tensor


def to_nz(weight: torch.Tensor) -> torch.Tensor:
    """在加载静态权重时，将 ND 权重一次性转换为 FRACTAL_NZ。"""
    if weight.device.type != "npu":
        raise ValueError("Weight passed to to_nz must be an NPU tensor")
    return torch_npu.npu_format_cast(weight, FRACTAL_NZ)


def _require_nz(name: str, weight: torch.Tensor) -> torch.Tensor:
    fmt = torch_npu.get_npu_format(weight)
    if fmt != FRACTAL_NZ:
        raise ValueError(
            f"Weight {name} must use FRACTAL_NZ, got format={fmt}; "
            f"convert it once with to_nz({name}) during weight loading"
        )
    return weight


def _rope_table(table):
    """将每个 token 的位置表行从 rope_width 补齐至 VL，保证向量读取不越界。"""
    rows, width = table.shape
    if width == VL:
        return table.contiguous()
    padded = torch.zeros((rows, VL), dtype=table.dtype, device=table.device)
    padded[:, :width] = table
    return padded


def prepare_attn_prologue(
    x,
    wqa,
    wqb,
    wkv,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkv,
    norm_weight_qr,
    norm_weight_kv,
    rope_sin,
    rope_cos,
    cache_index,
    kv_cache,
    *,
    norm_eps: float = 1e-6,
):
    """校验输入，准备分块、缓冲区及可复用的启动参数。

    run() 返回 (q, qr, descale_qr)，并原地更新 kv_cache。
    payload 使用 E4M3FN，输入 scale 使用 E8M0；权重采用 NZ，其余输入采用 ND。
    kv_cache 每行为 [D 字节 payload][D/32 个 BF16 scale]，D=512 时共 544 字节。
    核数由平台接口提供，分块方案自动选择。
    """
    _validate_attn_prologue_inputs(
        x,
        wqa,
        wqb,
        wkv,
        descale_x,
        descale_wqa,
        descale_wqb,
        descale_wkv,
        norm_weight_qr,
        norm_weight_kv,
        rope_sin,
        rope_cos,
        cache_index,
        kv_cache,
        norm_eps=norm_eps,
    )
    token_count, hidden_size = x.shape
    rank_size = wqa.shape[0]
    head_size = wkv.shape[0]
    head_count = wqb.shape[0] // head_size
    rope_width = rope_sin.shape[1]
    l1_reuse_eligible = _is_l1_reuse_eligible(x, wqa, wqb, wkv, rope_sin)
    # 只有 L1 复用 decode 路径按 T 范围复用编译模板；容量不足退回通用路径或
    # T>192 的 split_t 路径时保持逐 T 特化。
    range_upper = _token_range_upper(token_count) if l1_reuse_eligible else None
    tiling = AttnPrologueTiling.plan(
        token_count if range_upper is None else range_upper,
        hidden_size,
        rank_size,
        head_count,
        head_size,
        rope_width,
        l1_reuse_eligible=l1_reuse_eligible,
    )
    if range_upper is not None and not tiling.l1_reuse:
        range_upper = None
        tiling = AttnPrologueTiling.plan(
            token_count,
            hidden_size,
            rank_size,
            head_count,
            head_size,
            rope_width,
            l1_reuse_eligible=False,
        )
    dynamic_tokens = range_upper is not None
    l1_reuse = tiling.l1_reuse
    device = x.device
    qb_operand = _pack_b_weight_panels(wqb) if l1_reuse else _as_bytes(wqb)
    # kernel 输入和尾行填充使用 E8M0 字节视图，不执行数值转换。
    descale_x = _as_bytes(descale_x)
    descale_wqa = _as_bytes(descale_wqa)
    descale_wqb = _as_bytes(descale_wqb)
    descale_wkv = _as_bytes(descale_wkv)
    # FP32 交换区跨核传递投影部分和，以便沿完整行归约。
    work_rows = tiling.m_tiles_a1 * tiling.bm_a1
    ws_qa = torch.empty(
        (work_rows * tiling.split_k, rank_size), dtype=torch.float32, device=device
    )
    ws_kv = torch.empty(
        (work_rows * tiling.split_k, head_size), dtype=torch.float32, device=device
    )
    pad_rows = tiling.owners_b * tiling.bm_b
    q_buf = torch.empty(
        (pad_rows if l1_reuse else token_count, head_count, head_size),
        dtype=torch.bfloat16,
        device=device,
    )
    q = q_buf[:token_count]
    # 位置表、cache_index 与 x/scale 都使用 prepared 自有的补齐缓冲；
    # 这样每步 update 可以原地刷新数据，而不是复用首次调用的拷贝。
    table_rows = max(pad_rows, token_count)
    sin_buf = torch.zeros((table_rows, VL), dtype=rope_sin.dtype, device=device)
    cos_buf = torch.zeros((table_rows, VL), dtype=rope_cos.dtype, device=device)
    sin_buf[:token_count].copy_(_rope_table(rope_sin)[:token_count])
    cos_buf[:token_count].copy_(_rope_table(rope_cos)[:token_count])
    index_buf = torch.full((table_rows, 1), -1, dtype=torch.int64, device=device)
    index_buf[:token_count, 0] = cache_index.reshape(token_count)
    # QB 输入行补齐为完整 FixPipe 分块，公开输出仅暴露逻辑 token 前缀。
    qr_buf = torch.empty((pad_rows, rank_size), dtype=torch.int8, device=device)
    dqr_buf = torch.empty(
        (pad_rows, _ceil_div(rank_size, MX_PAIR_WIDTH), MX_PAIR),
        dtype=torch.int8,
        device=device,
    )
    if pad_rows > token_count:
        # 确定性初始化填充行；matmul 各行独立，不影响有效输出。
        qr_buf[token_count:].zero_()
        dqr_buf[token_count:].zero_()
    qr_bf16_buf = torch.empty(
        (pad_rows, rank_size), dtype=torch.bfloat16, device=device
    )
    qr = qr_bf16_buf[:token_count]
    descale_qr = dqr_buf[:token_count]

    x_padded = torch.zeros((work_rows, hidden_size), dtype=torch.int8, device=device)
    x_padded[:token_count].copy_(_as_bytes(x))
    dsx_padded = torch.zeros(
        (work_rows, hidden_size // MX_PAIR_WIDTH, MX_PAIR),
        dtype=descale_x.dtype,
        device=device,
    )
    dsx_padded[:token_count].copy_(descale_x)
    x, descale_x = x_padded, dsx_padded
    run_args = (
        # payload 按字节传入，权重保留 NZ、scale 保留 ND，计算类型由 Channel 指定。
        x_padded,
        dsx_padded,
        _as_bytes(wqa),
        descale_wqa,
        qb_operand,
        descale_wqb,
        _as_bytes(wkv),
        descale_wkv,
        norm_weight_qr.reshape(1, rank_size),
        norm_weight_kv.reshape(1, head_size),
        sin_buf,
        cos_buf,
        index_buf[:token_count],
        kv_cache.view(-1, _kv_row_bytes(head_size)),
        q_buf.view(-1, head_count * head_size),
        qr_buf,
        dqr_buf.view(pad_rows, tiling.qr_groups),
        dqr_buf,
        ws_qa,
        ws_kv,
        float(norm_eps),
        qr_bf16_buf,
    )
    prepared = PreparedAttnPrologue(
        tiling,
        run_args,
        (q, qr, descale_qr),
        (ws_qa.numel() + ws_kv.numel()) * FP32_BYTES,
        dynamic_tokens,
    )
    prepared._bind_step_inputs(
        x=x_padded,
        descale_x=dsx_padded,
        sin=sin_buf,
        cos=cos_buf,
        index=index_buf,
        q=q_buf,
        qr=qr_bf16_buf,
        dqr=dqr_buf,
        tokens=token_count,
        work_rows=work_rows,
        table_rows=table_rows,
        hidden=hidden_size,
        rank=rank_size,
    )
    return prepared


class PreparedAttnPrologue:
    """可复用的 prepared 对象；权重/分块/编译产物只准备一次。

    ``update_inputs`` 用于每步刷新 x、descale_x、位置表与 cache_index；它直接
    改写 kernel 读取的补齐缓冲，避免图回放或连续多步沿用 capture 时的数据。
    """

    def __init__(self, tiling, args, outputs, workspace_bytes, dynamic_tokens=False):
        self.tiling, self.args, self.outputs = tiling, args, outputs
        self.workspace_bytes = workspace_bytes
        self.launcher = _Launcher(tiling, dynamic_tokens)
        self.dynamic_tokens = dynamic_tokens
        self._compiled = None
        self._step = None

    def _bind_step_inputs(
        self,
        *,
        x,
        descale_x,
        sin,
        cos,
        index,
        q,
        qr,
        dqr,
        tokens,
        work_rows,
        table_rows,
        hidden,
        rank,
    ):
        self._step = dict(
            x=x,
            descale_x=descale_x,
            sin=sin,
            cos=cos,
            index=index,
            q=q,
            qr=qr,
            dqr=dqr,
            tokens=tokens,
            work_rows=work_rows,
            table_rows=table_rows,
            hidden=hidden,
            rank=rank,
        )
        self._refresh_outputs()

    def _refresh_outputs(self):
        s = self._step
        t = s["tokens"]
        self.outputs = (s["q"][:t], s["qr"][:t], s["dqr"][:t])

    def update_inputs(
        self,
        *,
        x=None,
        descale_x=None,
        rope_sin=None,
        rope_cos=None,
        cache_index=None,
        norm_eps=None,
    ):
        """原地刷新每步输入；未提供的项保持不变。返回 self。"""
        s = self._step
        if s is None:
            raise RuntimeError("PreparedAttnPrologue has no bound step inputs")
        tokens = s["tokens"]
        if x is not None:
            tokens = x.shape[0]
            if x.shape[1] != s["hidden"]:
                raise ValueError("x feature width changed")
        if cache_index is not None and cache_index.shape[0] != tokens:
            raise ValueError("cache_index length must match x rows")
        if not self.dynamic_tokens and tokens != s["tokens"]:
            raise ValueError(
                "token count change requires a dynamic-T prepared template"
            )
        if tokens > s["work_rows"]:
            raise ValueError(
                f"token count {tokens} exceeds prepared capacity {s['work_rows']}"
            )
        if x is not None:
            s["x"].zero_()
            s["x"][:tokens].copy_(_as_bytes(x))
        if descale_x is not None:
            s["descale_x"].zero_()
            s["descale_x"][:tokens].copy_(_as_bytes(descale_x))
        if rope_sin is not None:
            s["sin"].zero_()
            s["sin"][:tokens].copy_(_rope_table(rope_sin)[:tokens])
        if rope_cos is not None:
            s["cos"].zero_()
            s["cos"][:tokens].copy_(_rope_table(rope_cos)[:tokens])
        if cache_index is not None:
            s["index"].fill_(-1)
            s["index"][:tokens, 0] = cache_index.reshape(tokens)
        s["tokens"] = tokens
        args = list(self.args)
        args[0] = s["x"]
        args[1] = s["descale_x"]
        args[10] = s["sin"]
        args[11] = s["cos"]
        args[12] = s["index"][:tokens]
        if norm_eps is not None:
            args[20] = float(norm_eps)
        self.args = tuple(args)
        self._refresh_outputs()
        return self

    def run(self):
        if self._compiled is None:
            self._compiled = self.launcher.compile_cached(self.args)
        self._compiled(*self.args)
        return self.outputs


def attn_prologue(*args, **kwargs):
    return prepare_attn_prologue(*args, **kwargs).run()


# --------------------------------------------------------------------------
# 网络侧封装：现有 prolog（Attention.compute_win_cache）的替换入口
# --------------------------------------------------------------------------
# 输入契约：
#   x         : BF16 [T, H]，每步由本封装量化成 MXFP8(group=32)。
#   kv_cache  : UINT8，原地更新的 window KV cache（行宽 D + D/32*2）。
#   attn_metadata:
#       "cos_sin"            : {"win": (cos, sin), "comp": (cos, sin)}，
#                              两个张量均为 FP32 [T,1,1,Dr]（可 reshape [T,Dr]）。
#       "cache_slot_mapping" : {"win_kv": INT64 [T] 或 [T,1]，允许 -1 / 越界}。
#       "attention_type"     : 可选，"win" 取 cos_sin["win"]，否则取 "comp"。
#       "norm_eps"           : 可选，覆盖构造时的 norm_eps。
# 权重（逻辑 [K,N]，E4M3 MXFP8）与 weight_scale（[K/64,N,2]，uint8 E8M0 字节）
# 在构造时识别 ND/NZ 并一次性转置/预打包；gamma 为 BF16，构造时转 FP32。
# 返回 (win_kv, qr, q)：win_kv 即原地更新后的 kv_cache；qr 为量化前 BF16；
# q 为 RoPE 后 BF16。


class NetworkAttnPrologue:
    def __init__(
        self,
        wqa,
        wkv,
        wqb,
        descale_wqa,
        descale_wkv,
        descale_wqb,
        norm_weight_qr,
        norm_weight_kv,
        *,
        attention_type: str = "win",
        weight_nz: bool = True,
        norm_eps: float = 1e-20,
    ):
        self.attention_type = attention_type
        self.norm_eps = float(norm_eps)
        self.weight_nz = bool(weight_nz)
        self._static = dict(
            wqa=self._network_weight(wqa, descale_wqa, "wqa"),
            wkv=self._network_weight(wkv, descale_wkv, "wkv"),
            wqb=self._network_weight(wqb, descale_wqb, "wqb"),
        )
        self.gamma_qr = _network_gamma(norm_weight_qr, "norm_weight_qr")
        self.gamma_kv = _network_gamma(norm_weight_kv, "norm_weight_kv")
        self._prepared = None
        self._cache = None
        self._tokens = None

    def _network_weight(self, weight, descale, name):
        if weight.dim() != 2 or descale.dim() != 3:
            raise ValueError(f"{name}: weight must be [K,N] and scale [K/64,N,2]")
        reduction, output = weight.shape
        if reduction % MX_PAIR_WIDTH or descale.shape != (
            reduction // MX_PAIR_WIDTH,
            output,
            MX_PAIR,
        ):
            raise ValueError(f"{name}: unsupported weight/scale layout")
        if descale.dtype != torch.uint8:
            raise TypeError(f"{name}: weight_scale must be uint8 E8M0 bytes")
        # 物理格式可能是 ND 或 NZ；统一转回逻辑 ND，再转置成算子内部的 [N,K]。
        if torch_npu.get_npu_format(weight) == FRACTAL_NZ:
            weight = torch_npu.npu_format_cast(weight, 0)
        weight_nk = weight.transpose(0, 1).contiguous()
        if self.weight_nz:
            weight_nk = to_nz(weight_nk)
        scale_nk = descale.transpose(0, 1).contiguous().view(torch.float8_e8m0fnu)
        return weight_nk, scale_nk

    def _rope_tables(self, attn_metadata, tokens):
        key = "win" if self.attention_type == "win" else "comp"
        table = attn_metadata["cos_sin"][key]
        if not isinstance(table, (tuple, list)) or len(table) != 2:
            raise ValueError("cos_sin entries must be (cos, sin) tuples")
        cos, sin = table
        if cos.shape[0] != tokens or sin.shape[0] != tokens:
            raise ValueError("RoPE table token length differs from x")
        return cos.reshape(tokens, -1).contiguous(), sin.reshape(
            tokens, -1
        ).contiguous()

    def _rebind(self, tokens, kv_cache, eps, cos, sin, cache_index, payload, scales):
        self._prepared = prepare_attn_prologue(
            payload,
            self._static["wqa"][0],
            self._static["wqb"][0],
            self._static["wkv"][0],
            scales,
            self._static["wqa"][1],
            self._static["wqb"][1],
            self._static["wkv"][1],
            self.gamma_qr,
            self.gamma_kv,
            rope_sin=sin,
            rope_cos=cos,
            cache_index=cache_index,
            kv_cache=kv_cache,
            norm_eps=eps,
        )
        self._cache = kv_cache
        self._tokens = tokens

    def compute_win_cache(
        self,
        x,
        kv_cache,
        attn_metadata,
        *,
        attention_type=None,
        norm_eps=None,
    ):
        tokens = x.shape[0]
        if x.dtype != torch.bfloat16 or x.dim() != 2:
            raise TypeError("x must be BF16 [T,H]")
        # 网络传入的是 KVCache 对象，算法侧只接收其中的 win_kv Tensor。
        if not isinstance(kv_cache, torch.Tensor):
            if not hasattr(kv_cache, "win_kv"):
                raise TypeError("kv_cache must be a Tensor or expose .win_kv")
            kv_cache = kv_cache.win_kv
        payload, scales = _quantize_mxfp8_x(x)
        # 当前层必须显式给出 attention_type，否则 win/comp 位置表会取错。
        attention_type = attention_type or attn_metadata.get(
            "attention_type", self.attention_type
        )
        self.attention_type = attention_type
        cos, sin = self._rope_tables(attn_metadata, tokens)
        cache_index = attn_metadata["cache_slot_mapping"]["win_kv"].reshape(tokens)
        if cache_index.dtype != torch.int64:
            raise TypeError("cache_slot_mapping must be INT64")
        eps = float(
            norm_eps
            if norm_eps is not None
            else attn_metadata.get("norm_eps", self.norm_eps)
        )
        non_dynamic_t_change = (
            self._prepared is not None
            and not self._prepared.dynamic_tokens
            and tokens != self._tokens
        )
        if (
            self._prepared is None
            or self._cache is not kv_cache
            or tokens > self._prepared._step["work_rows"]
            or non_dynamic_t_change
        ):
            self._rebind(tokens, kv_cache, eps, cos, sin, cache_index, payload, scales)
        else:
            self._prepared.update_inputs(
                x=payload,
                descale_x=scales,
                rope_sin=sin,
                rope_cos=cos,
                cache_index=cache_index,
                norm_eps=eps,
            )
        q, qr, _ = self._prepared.run()
        return kv_cache, qr, q


def _network_gamma(gamma, name):
    if gamma.dtype == torch.bfloat16:
        return gamma.float()
    if gamma.dtype == torch.float32:
        return gamma
    raise TypeError(f"{name} must be BF16 or FP32")


def _quantize_mxfp8_x(x):
    """每步把 BF16 x 量化为 MXFP8（group=32）与 paired E8M0 字节。

    与现有 Linear 内部的量化口径一致：共享指数 e=floor(log2(amax))-8，
    clamp 到 [-127,127]；零组 scale 字节为 0。该转换独立于主 kernel。
    """
    tokens, hidden = x.shape
    if hidden % MX_PAIR_WIDTH:
        raise ValueError("x feature width must be a multiple of 64")
    grouped = x.float().reshape(tokens, hidden // MX_GROUP, MX_GROUP)
    amax = grouped.abs().amax(dim=-1)
    # HF round_scale：a=max(amax,1e-4)，scale 为不小于 a/448 的最小 2 的幂。
    floored = torch.clamp_min(amax, 1e-4)
    s0 = floored * (1.0 / E4M3_MAX)
    bits = s0.view(torch.int32)
    exponent = ((bits + 0x007FFFFF) >> 23) - FP32_EXP_BIAS
    exponent = exponent.clamp(-E8M0_BIAS, 127)
    scale = torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=x.device),
        exponent.to(torch.float32),
    )
    payload = (
        (grouped / scale[..., None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    )
    encoded = (exponent + E8M0_BIAS).to(torch.uint8)
    return (
        payload.reshape(tokens, hidden),
        encoded.view(tokens, hidden // MX_PAIR_WIDTH, MX_PAIR).view(
            torch.float8_e8m0fnu
        ),
    )


def _artifact_specs(tiling: AttnPrologueTiling):
    """AOT 编译契约，与 :meth:`_Launcher.compile_cached` 的动态模板一致。

    只有 cache_index 与 kv_cache 行数动态；其余张量按模板上界固定。
    """
    hidden, rank, head = tiling.hidden_size, tiling.rank_size, tiling.head_size
    head_count = tiling.head_count
    work_rows = tiling.m_tiles_a1 * tiling.bm_a1
    pad_rows = tiling.owners_b * tiling.bm_b
    scale_cols = hidden // MX_PAIR_WIDTH
    packed_b_rows = head_count * head // PANEL_OUTPUT_WIDTH * PANEL_REDUCTION_WINDOWS
    return [
        TensorSpec((work_rows, hidden), dtypes.int8),
        TensorSpec((work_rows, scale_cols, MX_PAIR), dtypes.int8),
        TensorSpec((rank, hidden), dtypes.int8, storage_format="nz"),
        TensorSpec((rank, scale_cols, MX_PAIR), dtypes.int8),
        TensorSpec((packed_b_rows, PANEL_BYTES), dtypes.int8),
        TensorSpec((head_count * head, rank // MX_PAIR_WIDTH, MX_PAIR), dtypes.int8),
        TensorSpec((head, hidden), dtypes.int8, storage_format="nz"),
        TensorSpec((head, scale_cols, MX_PAIR), dtypes.int8),
        TensorSpec((1, rank), dtypes.float32),
        TensorSpec((1, head), dtypes.float32),
        TensorSpec((pad_rows, VL), dtypes.float32),
        TensorSpec((pad_rows, VL), dtypes.float32),
        _dynamic_index_spec(tiling),
        _dynamic_cache_spec(tiling),
        TensorSpec((pad_rows, head_count * head), dtypes.bfloat16),
        TensorSpec((pad_rows, rank), dtypes.int8),
        TensorSpec((pad_rows, tiling.qr_groups), dtypes.int8),
        TensorSpec((pad_rows, _ceil_div(rank, MX_PAIR_WIDTH), MX_PAIR), dtypes.int8),
        TensorSpec((work_rows * tiling.split_k, rank), dtypes.float32),
        TensorSpec((work_rows * tiling.split_k, head), dtypes.float32),
        dtypes.float32,
        TensorSpec((pad_rows, rank), dtypes.bfloat16),
    ]


@cannbotdsl.aot.export("attn_prologue")
def export_attn_prologue():
    """为 7 个 T 范围模板 × N∈{16,64} 收集 AOT 编译请求，不启动 Provider。

    T 使用 ``Dim("TOKENS", 1, range_max, 1)`` 的运行时契约，不枚举具体 T；
    kv_cache 行维同样动态，避免容量差异把同一 range 拆成多个产物。
    """
    geometry = RESIDENT_GEOMETRY
    handles = []
    try:
        for head_count in SUPPORTED_HEAD_COUNTS:
            for upper in TOKEN_RANGE_UPPERS:
                tiling = AttnPrologueTiling.plan(
                    upper,
                    geometry[0],
                    geometry[1],
                    head_count,
                    geometry[3],
                    geometry[4],
                    l1_reuse_eligible=True,
                )
                launcher = _Launcher(tiling, dynamic_tokens=True)
                handles.append(
                    cannbotdsl.compile(launcher.run, *_artifact_specs(tiling))
                )
    finally:
        for handle in handles:
            handle.close()


def _pack_b_weight_panels(weight: torch.Tensor) -> torch.Tensor:
    """一次性打包静态 QB 权重，不计入重复执行 kernel 的耗时。
    输入为逻辑形状 [N,1280] 的 int8/FP8 权重，支持 ND 或 NZ；
    输出为 ND int8 [N/256*4,81920]，每行按 [K1=10,N1=16,N0=16,K0=32] 排列。
    也可在模型加载时先打包 CPU ND 权重，再一次性传到设备。
    权重改变后必须重新打包，此处不隐式缓存权重。
    当前环境的 int8 NZ→ND 格式转换不能保持权重数值，因此在 CPU 上解析 NZ
    原始字节；这次 D2H/H2D 往返仅发生在准备阶段，不进入重复执行路径。"""
    if (
        weight.ndim != 2
        or weight.shape[1] != PANEL_RANK_WIDTH
        or weight.shape[0] % PANEL_OUTPUT_WIDTH
    ):
        raise ValueError(
            f"Panel packing requires [N,{PANEL_RANK_WIDTH}], N divisible by {PANEL_OUTPUT_WIDTH}"
        )
    if weight.dtype not in (torch.int8, torch.float8_e4m3fn):
        raise TypeError("Panel packing requires int8 or float8_e4m3fn")
    raw = _as_bytes(weight)
    if raw.device.type == "npu":
        device = raw.device
        if torch_npu.get_npu_format(raw) == FRACTAL_NZ:
            output_width, reduction_width = raw.shape
            if (
                raw.storage_offset() != 0
                or raw.untyped_storage().nbytes() != output_width * reduction_width
            ):
                raise ValueError(
                    "NZ packing requires an unpadded, whole weight allocation"
                )
            host_weight_bytes = torch.empty(
                output_width * reduction_width, dtype=torch.int8
            )
            copy_device_to_host = ctypes.CDLL("libascendcl.so").aclrtMemcpy
            copy_device_to_host.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            copy_device_to_host.restype = ctypes.c_int
            with torch.npu.device(device):
                torch.npu.synchronize(device)
                copy_status = copy_device_to_host(
                    host_weight_bytes.data_ptr(),
                    output_width * reduction_width,
                    raw.data_ptr(),
                    output_width * reduction_width,
                    ACL_MEMCPY_DEVICE_TO_HOST,
                )
                if copy_status:
                    raise RuntimeError(f"Weight D2H copy failed: {copy_status}")
            return (
                host_weight_bytes.view(
                    PANEL_REDUCTION_WINDOWS,
                    PANEL_K_FRACTALS,
                    output_width // PANEL_OUTPUT_WIDTH,
                    PANEL_N_FRACTALS,
                    NZ_M_FRAC,
                    NZ_C0_1B,
                )
                .permute(2, 0, 1, 3, 4, 5)
                .contiguous()
                .view(
                    output_width // PANEL_OUTPUT_WIDTH * PANEL_REDUCTION_WINDOWS,
                    PANEL_BYTES,
                )
                .to(device)
            )
        if torch_npu.get_npu_format(raw) not in ND_FORMATS:
            raise ValueError("Weight packing supports ND or NZ storage")
        return _pack_b_weight_panels(raw.cpu()).to(device)
    output_width = raw.shape[0]
    return (
        raw.contiguous()
        .view(
            output_width // PANEL_OUTPUT_WIDTH,
            PANEL_N_FRACTALS,
            NZ_M_FRAC,
            PANEL_REDUCTION_WINDOWS,
            PANEL_K_FRACTALS,
            NZ_C0_1B,
        )
        .permute(0, 3, 4, 1, 2, 5)
        .contiguous()
        .view(output_width // PANEL_OUTPUT_WIDTH * PANEL_REDUCTION_WINDOWS, PANEL_BYTES)
    )


def _validate_norm_eps(norm_eps):
    """第 1 层：先检查host 标量，再访问设备张量元数据。"""
    if isinstance(norm_eps, bool) or not isinstance(norm_eps, numbers.Real):
        raise TypeError("Parameter norm_eps must be a host real scalar")
    if not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError("Parameter norm_eps must be finite and positive")


def _validate_tensor_metadata(tensors):
    """第 2 层：检查公共张量类型、设备一致性及连续性。"""
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Input {name} must be a torch.Tensor")
    device = tensors["x"].device
    if device.type != "npu":
        raise ValueError("Input x must be an NPU tensor")
    for name, tensor in tensors.items():
        if tensor.device != device:
            raise ValueError(f"Input {name} must be on the same NPU as x")
        if not tensor.is_contiguous():
            raise ValueError(f"Input {name} must be contiguous")


def _require_dtype(name, tensor, dtype):
    if tensor.dtype != dtype:
        raise TypeError(f"Input {name} must have dtype {dtype}, got {tensor.dtype}")


def _require_shape(name, tensor, shape):
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"Input {name} must have shape {shape}, got {tuple(tensor.shape)}"
        )


def _validate_projection_inputs(x, wqa, wqb, wkv):
    """第 3a 层：检查 payload 元数据及三个投影的关联轴。"""
    for name, tensor in (("x", x), ("wqa", wqa), ("wqb", wqb), ("wkv", wkv)):
        if tensor.ndim != 2 or any(size <= 0 for size in tensor.shape):
            raise ValueError(f"Input {name} must be a nonempty matrix")
        _require_dtype(name, tensor, torch.float8_e4m3fn)
    _, hidden_size = x.shape
    rank_size, head_size = wqa.shape[0], wkv.shape[0]
    for name, width in (
        ("hidden_size", hidden_size),
        ("rank_size", rank_size),
        ("head_size", head_size),
    ):
        if width % MX_PAIR_WIDTH:
            raise ValueError(
                f"Dimension {name} must be divisible by {MX_PAIR_WIDTH}, got {width}"
            )
    if wqa.shape[1] != hidden_size or wkv.shape[1] != hidden_size:
        raise ValueError("Weights wqa and wkv must reduce over the input hidden axis")
    if wqb.shape[1] != rank_size:
        raise ValueError("Weight wqb must reduce over the QA rank axis")
    if wqb.shape[0] % head_size:
        raise ValueError("Weight wqb output width must be divisible by head_size")
    head_count = wqb.shape[0] // head_size
    if head_count not in SUPPORTED_HEAD_COUNTS:
        raise ValueError(
            f"Q head count must be one of {SUPPORTED_HEAD_COUNTS}, got {head_count}"
        )


def _validate_scale_inputs(tensors):
    """第 3b 层：每 MX_GROUP 个 payload 元素对应一个 E8M0 scale。"""
    for payload_name in ("x", "wqa", "wqb", "wkv"):
        scale_name = "descale_" + payload_name
        scale = tensors[scale_name]
        rows, reduction_width = tensors[payload_name].shape
        _require_dtype(scale_name, scale, torch.float8_e8m0fnu)
        _require_shape(
            scale_name, scale, (rows, reduction_width // MX_PAIR_WIDTH, MX_PAIR)
        )


def _validate_norm_inputs(norm_weight_qr, norm_weight_kv, rank_size, head_size):
    """第 3c 层：RMSNorm 权重必须覆盖对应的完整归约轴。"""
    for name, tensor, width in (
        ("norm_weight_qr", norm_weight_qr, rank_size),
        ("norm_weight_kv", norm_weight_kv, head_size),
    ):
        _require_shape(name, tensor, (width,))
        _require_dtype(name, tensor, torch.float32)


def _validate_rope_inputs(rope_sin, rope_cos, token_count, head_size):
    """第 3d 层：检查 RoPE 行数及尾部特征对齐约束。"""
    if rope_sin.ndim != 2:
        raise ValueError("RoPE tables must be 2-D with shape [T, Dr]")
    if (
        rope_sin.shape[0] != token_count
        or rope_sin.shape[1] <= 0
        or rope_cos.shape != rope_sin.shape
    ):
        raise ValueError("RoPE tables must share shape [T, Dr] with Dr > 0")
    rope_width = rope_sin.shape[1]
    if rope_width % 2 or rope_width > head_size:
        raise ValueError("RoPE width must be even and no larger than head_size")
    if rope_width % FRACTAL_M or rope_width > VL:
        # 两个 FP32 半区的起点都需要按 UB_ALIGN 字节对齐。
        raise NotImplementedError(
            f"RoPE width must be divisible by {FRACTAL_M} and at most {VL}, got {rope_width}"
        )
    if head_size // FIXPIPE_DUAL_DST < rope_width:
        raise NotImplementedError(
            "RoPE tail must fit in the second FixPipe feature slice"
        )
    _require_dtype("rope_sin", rope_sin, torch.float32)
    _require_dtype("rope_cos", rope_cos, torch.float32)


def _validate_cache_inputs(cache_index, kv_cache, token_count, head_size):
    """第 3e 层：检查 cache 存储；索引范围及唯一性由调用方保证。"""
    _require_shape("cache_index", cache_index, (token_count,))
    _require_dtype("cache_index", cache_index, torch.int64)
    row_bytes = _kv_row_bytes(head_size)
    if kv_cache.ndim != 4:
        raise ValueError(
            f"Cache kv_cache must have shape [P, BS, 1, {row_bytes}] with P, BS > 0"
        )
    if (
        tuple(kv_cache.shape[2:]) != (1, row_bytes)
        or kv_cache.shape[0] <= 0
        or kv_cache.shape[1] <= 0
    ):
        raise ValueError(
            f"Cache kv_cache must have shape [P, BS, 1, {row_bytes}] with P, BS > 0"
        )
    _require_dtype("kv_cache", kv_cache, torch.uint8)


def _validate_storage_layouts(tensors):
    """第 4 层：逻辑元数据检查通过后，再检查物理存储布局。"""
    for name, tensor in tensors.items():
        if name in ("wqa", "wqb", "wkv"):
            _require_nz(name, tensor)
        elif torch_npu.get_npu_format(tensor) not in ND_FORMATS:
            raise ValueError(f"Input {name} must use ND storage (format 0 or 2)")


def _validate_attn_prologue_inputs(
    x,
    wqa,
    wqb,
    wkv,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkv,
    norm_weight_qr,
    norm_weight_kv,
    rope_sin,
    rope_cos,
    cache_index,
    kv_cache,
    *,
    norm_eps,
):
    """按层次和计算模块校验输入，不读取设备张量值。分块容量及 NZ 网格检查仍由 AttnPrologueTiling 负责。cache_index 的取值范围和唯一性由调用方保证。"""
    _validate_norm_eps(norm_eps)
    tensors = dict(
        x=x,
        wqa=wqa,
        wqb=wqb,
        wkv=wkv,
        descale_x=descale_x,
        descale_wqa=descale_wqa,
        descale_wqb=descale_wqb,
        descale_wkv=descale_wkv,
        norm_weight_qr=norm_weight_qr,
        norm_weight_kv=norm_weight_kv,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        cache_index=cache_index,
        kv_cache=kv_cache,
    )
    _validate_tensor_metadata(tensors)
    _validate_projection_inputs(x, wqa, wqb, wkv)
    _validate_scale_inputs(tensors)
    _validate_norm_inputs(norm_weight_qr, norm_weight_kv, wqa.shape[0], wkv.shape[0])
    _validate_rope_inputs(rope_sin, rope_cos, x.shape[0], wkv.shape[0])
    _validate_cache_inputs(cache_index, kv_cache, x.shape[0], wkv.shape[0])
    _validate_storage_layouts(tensors)


def _is_l1_reuse_eligible(x, wqa, wqb, wkv, rope_sin) -> bool:
    """公共输入校验通过后，判断是否具备 L1 复用资格。"""
    tokens, hidden = x.shape
    if (
        not 1 <= tokens <= DECODE_TOKEN_LIMIT
        or (hidden, wqa.shape[0], wqb.shape[0], wkv.shape[0], rope_sin.shape[1])
        not in RESIDENT_GEOMETRIES
    ):
        return False
    # 驻留打包要求完整且没有额外填充的 NZ 权重存储。
    return wqb.storage_offset() == 0 and wqb.untyped_storage().nbytes() == wqb.numel()
