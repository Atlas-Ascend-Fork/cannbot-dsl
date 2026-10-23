# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# fmt: off
# Keep compact call argument groups consistent with the original FlashKDA samples.

"""Three-way FlashKDA precision: CPU FP32, NPU-rounding simulation, and DSL."""

import dataclasses
import os

import pytest
import torch
import torch.nn.functional as F
from _samples_path import load_sample

CHUNK_SIZE = 64
REFERENCE_BLOCK_SIZE = 16
LOW_DTYPE = torch.bfloat16
SUPPORTED_HEAD_DIM = 128


# Independent PyTorch references from the original FlashKDA example:
# cpu_chunk uses FP32; npu_chunk models current BF16/FP16 handoffs on CPU,
# including FP32 recurrent-state and output accumulators.
@dataclasses.dataclass
class KdaInputs:
    k: torch.Tensor  # (B, Nk, S, D) bf16
    v: torch.Tensor  # (B, Nv, S, D) bf16
    q: torch.Tensor  # (B, Nk, S, D) bf16
    beta_brc: torch.Tensor  # (B, Nv, S, 1) raw BF16 logits
    g: torch.Tensor  # (B, Nv, S, D) raw BF16 logits
    initial_state: torch.Tensor  # (B, Nv, Dk, Dv) fp32
    scale_value: float
    a_log: torch.Tensor  # (Nv,) fp32
    dt_bias: torch.Tensor  # (Nv, D) fp32
    lower_bound: float


def _chunk_major(x: torch.Tensor) -> torch.Tensor:
    batch, heads, seq_len, dim = x.shape
    chunks = seq_len // CHUNK_SIZE
    return x.reshape(batch, heads, chunks, CHUNK_SIZE, dim).contiguous()


def make_inputs_generalized(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    *,
    nk: int | None = None,
    seed: int = 0,
    lower_bound: float = -1.0,
    v_scale: float = 1.0,
    b_spread: float = 1.0,
    init_scale: float = 0.1,
) -> KdaInputs:
    """Build BF16 Q/K/V and raw gate/beta logits with a seeded CPU generator."""
    if dim != SUPPORTED_HEAD_DIM:
        raise ValueError(f"flash_kda 当前只支持 D={SUPPORTED_HEAD_DIM}，收到 D={dim}")
    nk = heads if nk is None else nk
    if heads % nk != 0:
        raise ValueError(f"GQA requires Nv % Nk == 0，收到 Nv={heads}, Nk={nk}")

    gen = torch.Generator(device="cpu").manual_seed(seed)
    k = F.normalize(torch.randn(batch, nk, seq_len, dim, generator=gen), p=2, dim=-1)
    q = F.normalize(torch.randn(batch, nk, seq_len, dim, generator=gen), p=2, dim=-1)
    v = torch.randn(batch, heads, seq_len, dim, generator=gen) * float(v_scale)
    g = torch.randn(batch, heads, seq_len, dim, generator=gen).to(LOW_DTYPE)
    beta = (torch.randn(batch, heads, seq_len, 1, generator=gen) * float(b_spread)).to(LOW_DTYPE)
    initial_state = torch.rand(batch, heads, dim, dim, generator=gen) * float(init_scale)

    return KdaInputs(
        k=k.to(LOW_DTYPE), v=v.to(LOW_DTYPE), q=q.to(LOW_DTYPE), beta_brc=beta, g=g, initial_state=initial_state,
        scale_value=dim**-0.5, a_log=torch.linspace(-1.0, 0.5, heads, dtype=torch.float32),
        dt_bias=torch.randn(heads, dim, generator=gen, dtype=torch.float32), lower_bound=float(lower_bound),
    )


@dataclasses.dataclass
class _StageOneGolden:
    gamma: torch.Tensor
    k_decayed_beta: torch.Tensor
    k_decayed: torch.Tensor
    q_decayed: torch.Tensor
    t: torch.Tensor
    u_pre: torch.Tensor
    w: torch.Tensor
    mqk: torch.Tensor
    k_restored: torch.Tensor
    gamma_c: torch.Tensor


@dataclasses.dataclass
class _StageTwoGolden:
    u: torch.Tensor
    o_state: torch.Tensor
    s: torch.Tensor


@dataclasses.dataclass
class _StageThreeGolden:
    out: torch.Tensor


@dataclasses.dataclass
class _ChainGolden:
    stage1: _StageOneGolden
    stage2: _StageTwoGolden
    stage3: _StageThreeGolden


def _to_fp16_to_fp32(x: torch.Tensor) -> torch.Tensor:
    """近似 Stage1 FP16 packed/NZ L1 交接。"""
    return x.to(torch.float16).to(torch.float32)


def _to_bf16_to_fp32(x: torch.Tensor) -> torch.Tensor:
    """近似 BF16 GM 读写精度损失。"""
    return x.to(LOW_DTYPE).to(torch.float32)


def _reference_lower_matmul(
    left: torch.Tensor,
    right: torch.Tensor,
    cumulative_decay: torch.Tensor,
    *,
    diagonal: int,
    cast_operand=None,
) -> torch.Tensor:
    """Compute causal QK/KKT tiles with 16-token relative-decay references."""
    result = torch.zeros(*left.shape[:-1], right.shape[-2], dtype=torch.float32, device=left.device)
    cast = (lambda x: x) if cast_operand is None else cast_operand
    for row_start in range(0, CHUNK_SIZE, REFERENCE_BLOCK_SIZE):
        row_end = row_start + REFERENCE_BLOCK_SIZE
        reference = cumulative_decay[..., slice(row_start, row_start + 1), :]
        left_rows = cast(
            left[..., row_start:row_end, :] * (cumulative_decay[..., row_start:row_end, :] - reference).exp(),
        )
        for col_start in range(0, row_end, REFERENCE_BLOCK_SIZE):
            col_end = col_start + REFERENCE_BLOCK_SIZE
            right_cols = cast(
                right[..., col_start:col_end, :] * (reference - cumulative_decay[..., col_start:col_end, :]).exp(),
            )
            result[..., row_start:row_end, col_start:col_end] = (
                left_rows @ right_cols.transpose(-1, -2)
            )

    return result * torch.tril(
        torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float32, device=left.device), diagonal=diagonal,
    )


def _npu_neumann_inverse_from_negative_t(negative_t: torch.Tensor) -> torch.Tensor:
    """从 NPU ``-T`` 交接重建驻留逆矩阵（packed64：16→32→64 块下三角组合）。"""
    inverse = torch.zeros_like(negative_t, dtype=torch.float32)
    for block_idx in range(0, CHUNK_SIZE, 16):
        block_slice = slice(block_idx, block_idx + 16)
        block = _to_fp16_to_fp32(negative_t[block_slice, block_slice])
        diag_inv = torch.eye(16, dtype=torch.float32, device=negative_t.device) + block
        diag_power2 = block @ block
        diag_inv = diag_inv + _to_fp16_to_fp32(diag_inv) @ _to_fp16_to_fp32(diag_power2)
        diag_power4 = _to_fp16_to_fp32(diag_power2) @ _to_fp16_to_fp32(diag_power2)
        diag_inv = diag_inv + _to_fp16_to_fp32(diag_inv) @ _to_fp16_to_fp32(diag_power4)
        diag_power8 = _to_fp16_to_fp32(diag_power4) @ _to_fp16_to_fp32(diag_power4)
        diag_inv = diag_inv + _to_fp16_to_fp32(diag_inv) @ _to_fp16_to_fp32(diag_power8)
        inverse[block_slice, block_slice] = diag_inv

    for block_idx in range(0, CHUNK_SIZE, 32):
        even = slice(block_idx, block_idx + 16)
        odd = slice(block_idx + 16, block_idx + 32)
        odd_lower_even = _to_fp16_to_fp32(negative_t[odd, even])
        lower_tmp = odd_lower_even @ _to_fp16_to_fp32(inverse[even, even])
        inverse[odd, even] = _to_fp16_to_fp32(inverse[odd, odd]) @ _to_fp16_to_fp32(lower_tmp)

    even32 = slice(0, 32)
    odd32 = slice(32, 64)
    odd32_lower_even32 = _to_fp16_to_fp32(negative_t[odd32, even32])
    lower32_tmp = odd32_lower_even32 @ _to_fp16_to_fp32(inverse[even32, even32])
    inverse[odd32, even32] = _to_fp16_to_fp32(inverse[odd32, odd32]) @ _to_fp16_to_fp32(lower32_tmp)
    return _to_bf16_to_fp32(inverse)


def _stage1_cpu(k, v, q, beta_brc, g, *, scale_value) -> _StageOneGolden:
    """Stage1 CPU 公式（全 fp32）：生成不依赖状态的 chunk 工作区。"""
    if v.shape[1] != k.shape[1]:  # GQA repeat 到 Nv
        repeat = v.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        q = q.repeat_interleave(repeat, dim=1)
    k_chunks = _chunk_major(k).to(torch.float32)
    v_chunks = _chunk_major(v).to(torch.float32)
    q_chunks = _chunk_major(q).to(torch.float32)
    beta = _chunk_major(beta_brc).to(torch.float32)
    g_i = _chunk_major(g).to(torch.float32)

    cumulative_decay = torch.cumsum(g_i, dim=-2)
    gamma = torch.exp(cumulative_decay)
    k_decayed = gamma * k_chunks
    k_decayed_beta = beta * k_decayed
    q_decayed = gamma * q_chunks * scale_value

    t_matrix = _reference_lower_matmul(beta * k_chunks, k_chunks, cumulative_decay, diagonal=-1)

    identity = torch.eye(CHUNK_SIZE, dtype=torch.float32, device=k.device)
    diag_beta = torch.diag_embed(beta.squeeze(-1))
    inverse = torch.linalg.solve_triangular(identity + t_matrix, diag_beta, upper=False, unitriangular=True)
    u_pre = inverse @ v_chunks
    w_matrix = inverse @ k_decayed
    mqk = _reference_lower_matmul(q_chunks * scale_value, k_chunks, cumulative_decay, diagonal=0)
    gamma_c = gamma[..., -1, :, None].contiguous()
    k_restored = k_chunks * (cumulative_decay[..., -1:, :] - cumulative_decay).exp()
    return _StageOneGolden(
        gamma, k_decayed_beta, k_decayed, q_decayed, t_matrix, u_pre, w_matrix, mqk, k_restored, gamma_c,
    )


def _stage1_npu(k, v, q, beta_brc, g, *, scale_value) -> _StageOneGolden:
    """Stage1 NPU-oriented golden：显式 batch/head/chunk 循环 + bf16/fp16 交接。"""
    if v.shape[1] != k.shape[1]:
        rep = v.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        q = q.repeat_interleave(rep, dim=1)
    k_chunks = _to_bf16_to_fp32(_chunk_major(k))
    v_chunks = _chunk_major(v).to(torch.float32)
    q_chunks = _to_bf16_to_fp32(_chunk_major(q))
    beta = _chunk_major(beta_brc).to(torch.float32)
    g_i = _chunk_major(g).to(torch.float32)

    batch, heads, chunks, _, dim = k_chunks.shape
    gamma = torch.empty_like(k_chunks)
    k_decayed_beta = torch.empty_like(k_chunks)
    k_decayed = torch.empty_like(k_chunks)
    q_decayed = torch.empty_like(k_chunks)
    t_output = torch.empty(batch, heads, chunks, CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float32, device=k.device)
    u_pre = torch.empty_like(k_chunks)
    w_matrix = torch.empty_like(k_chunks)
    mqk = torch.empty_like(t_output)
    k_restored = torch.empty_like(k_chunks)
    gamma_c_output = torch.empty(batch, heads, chunks, dim, 1, dtype=torch.float32, device=k.device)
    for b in range(batch):
        for h in range(heads):
            for c in range(chunks):
                k_chunk = k_chunks[b, h, c]
                v_chunk = v_chunks[b, h, c]
                q_chunk = q_chunks[b, h, c]
                beta_c = beta[b, h, c]
                g_c = g_i[b, h, c]

                g_cumsum = torch.cumsum(g_c, dim=0)  # AIV 全程 fp32
                gamma_c = torch.exp(g_cumsum)
                k_decayed_prepare = gamma_c * k_chunk
                k_decayed_chunk = _to_bf16_to_fp32(k_decayed_prepare)
                q_decayed_chunk = _to_bf16_to_fp32(gamma_c * q_chunk * scale_value)

                t_matrix = (
                    _reference_lower_matmul(k_chunk, k_chunk, g_cumsum, diagonal=-1, cast_operand=_to_bf16_to_fp32)
                    * beta_c
                )
                mqk_chunk = _to_bf16_to_fp32(
                    _reference_lower_matmul(
                        q_chunk * scale_value, k_chunk, g_cumsum, diagonal=0, cast_operand=_to_bf16_to_fp32,
                    ),
                )

                negative_t = _to_fp16_to_fp32(-t_matrix)
                resident_inv = _npu_neumann_inverse_from_negative_t(negative_t)

                k_decayed_beta_chunk = _to_bf16_to_fp32(beta_c * k_decayed_prepare)
                v_beta = _to_bf16_to_fp32(beta_c * v_chunk)
                u_pre_chunk = _to_bf16_to_fp32(_to_bf16_to_fp32(resident_inv) @ _to_bf16_to_fp32(v_beta))
                workspace_w = _to_bf16_to_fp32(_to_bf16_to_fp32(resident_inv) @ _to_bf16_to_fp32(-k_decayed_beta_chunk))
                gamma_c_last = gamma_c[-1].reshape(k_chunk.shape[-1], 1)
                k_restored_chunk = _to_bf16_to_fp32(k_chunk * (g_cumsum[-1:] - g_cumsum).exp())

                gamma[b, h, c] = gamma_c
                k_decayed[b, h, c] = k_decayed_chunk
                k_decayed_beta[b, h, c] = k_decayed_beta_chunk
                q_decayed[b, h, c] = q_decayed_chunk
                t_output[b, h, c] = t_matrix
                u_pre[b, h, c] = u_pre_chunk
                w_matrix[b, h, c] = -workspace_w
                mqk[b, h, c] = mqk_chunk
                gamma_c_output[b, h, c] = gamma_c_last
                k_restored[b, h, c] = k_restored_chunk
    return _StageOneGolden(
        gamma, k_decayed_beta, k_decayed, q_decayed, t_output, u_pre, w_matrix, mqk, k_restored, gamma_c_output,
    )


def _stage2_cpu(stage1: _StageOneGolden, initial_state) -> _StageTwoGolden:
    """Stage2 CPU 公式（全 fp32）：用 stage1 工作区 + 初始 state 递推。"""
    _, _, chunks, _, _ = stage1.u_pre.shape
    u_output = torch.zeros_like(stage1.u_pre)
    o_state = torch.zeros_like(stage1.u_pre)
    state = initial_state.to(torch.float32).clone()
    u_pre = stage1.u_pre.to(torch.float32)
    w_matrix = stage1.w.to(torch.float32)
    q_decayed = stage1.q_decayed.to(torch.float32)
    k_restored = stage1.k_restored.to(torch.float32)
    gamma_c = stage1.gamma_c.to(torch.float32)
    for c in range(chunks):
        u_chunk = u_pre[:, :, c] - w_matrix[:, :, c] @ state
        o_chunk = q_decayed[:, :, c] @ state
        state = (
            gamma_c[:, :, c] * state + k_restored[:, :, c].transpose(-1, -2) @ u_chunk
        )
        u_output[:, :, c] = u_chunk
        o_state[:, :, c] = o_chunk
    return _StageTwoGolden(u=u_output, o_state=o_state, s=state)


def _stage2_npu(stage1: _StageOneGolden, initial_state) -> _StageTwoGolden:
    """Stage2 NPU-oriented golden：显式 chunk 递推 + bf16 交接。"""
    batch, heads, chunks, _, _ = stage1.u_pre.shape
    u_output = torch.zeros_like(stage1.u_pre, dtype=torch.float32)
    o_state = torch.zeros_like(stage1.u_pre, dtype=torch.float32)
    u_pre = _to_bf16_to_fp32(stage1.u_pre)
    w_negative = _to_bf16_to_fp32(-stage1.w)  # workspace 存负 W
    q_decayed = _to_bf16_to_fp32(stage1.q_decayed)
    k_restored = _to_bf16_to_fp32(stage1.k_restored)
    gamma_c = stage1.gamma_c.to(torch.float32)
    state = initial_state.to(torch.float32).clone()
    for b in range(batch):
        for h in range(heads):
            state_bh = state[b, h]
            for c in range(chunks):
                w_s = _to_bf16_to_fp32(w_negative[b, h, c]) @ _to_bf16_to_fp32(state_bh)
                u_chunk = _to_bf16_to_fp32(u_pre[b, h, c] + w_s)
                o_chunk = _to_bf16_to_fp32(q_decayed[b, h, c]) @ _to_bf16_to_fp32(state_bh)
                kr_u = _to_bf16_to_fp32(k_restored[b, h, c].transpose(-1, -2)) @ _to_bf16_to_fp32(u_chunk)
                # Current DSL retains the recurrent accumulator in FP32.
                scaled_state = gamma_c[b, h, c] * state_bh
                state_bh = scaled_state + kr_u
                u_output[b, h, c] = u_chunk
                o_state[b, h, c] = o_chunk
            state[b, h] = state_bh
    return _StageTwoGolden(u=u_output, o_state=o_state, s=state)


def _stage3_cpu(stage1: _StageOneGolden, stage2: _StageTwoGolden) -> _StageThreeGolden:
    """Stage3 CPU 公式：O = O_state + Mqk @ U。"""
    output = stage2.o_state.to(torch.float32) + stage1.mqk.to(torch.float32) @ stage2.u.to(torch.float32)
    batch, heads, chunks, chunk, dim = output.shape
    return _StageThreeGolden(out=output.reshape(batch, heads, chunks * chunk, dim).contiguous())


def _stage3_npu(stage1: _StageOneGolden, stage2: _StageTwoGolden) -> _StageThreeGolden:
    """Stage3 NPU-oriented golden：显式 chunk 合成 + bf16 交接。"""
    batch, heads, chunks, chunk, dim = stage2.o_state.shape
    output = torch.empty_like(stage2.o_state, dtype=torch.float32)
    # Q@state and Mqk@U accumulate into one FP32 L0C tile.
    o_state = stage2.o_state.float()
    mqk = _to_bf16_to_fp32(stage1.mqk)
    u_output = _to_bf16_to_fp32(stage2.u)
    for b in range(batch):
        for h in range(heads):
            for c in range(chunks):
                mqk_u = _to_bf16_to_fp32(mqk[b, h, c]) @ _to_bf16_to_fp32(u_output[b, h, c])
                output[b, h, c] = _to_bf16_to_fp32(o_state[b, h, c] + mqk_u)
    return _StageThreeGolden(out=output.reshape(batch, heads, chunks * chunk, dim).contiguous())


def _chain_cpu(inputs: KdaInputs) -> _ChainGolden:
    s1 = _stage1_cpu(inputs.k, inputs.v, inputs.q, inputs.beta_brc, inputs.g, scale_value=inputs.scale_value)
    s2 = _stage2_cpu(s1, inputs.initial_state)
    s3 = _stage3_cpu(s1, s2)
    return _ChainGolden(stage1=s1, stage2=s2, stage3=s3)


def _chain_npu(inputs: KdaInputs) -> _ChainGolden:
    s1 = _stage1_npu(inputs.k, inputs.v, inputs.q, inputs.beta_brc, inputs.g, scale_value=inputs.scale_value)
    s2 = _stage2_npu(s1, inputs.initial_state)
    s3 = _stage3_npu(s1, s2)
    return _ChainGolden(stage1=s1, stage2=s2, stage3=s3)


def compute_golden(name: str, inputs: KdaInputs):
    """Apply activation/normalization, then pad with an identity recurrence tail."""
    tokens = inputs.q.shape[-2]
    alpha = torch.exp(inputs.a_log).view(1, -1, 1, 1)
    bias = inputs.dt_bias.view(1, -1, 1, inputs.dt_bias.shape[-1])
    activated = dataclasses.replace(
        inputs, q=inputs.q.float() / torch.sqrt(inputs.q.float().square().sum(-1, keepdim=True) + 1e-6),
        k=inputs.k.float() / torch.sqrt(inputs.k.float().square().sum(-1, keepdim=True) + 1e-6),
        g=inputs.lower_bound * torch.sigmoid(alpha * (inputs.g.float() + bias)),
        beta_brc=torch.sigmoid(inputs.beta_brc.float()),
    )
    # Zero activated decay means exp(g)=1; beta=0 means no state update.
    padding = (-tokens) % CHUNK_SIZE
    if padding:
        activated = dataclasses.replace(
            activated,
            **{
                name: F.pad(getattr(activated, name), (0, 0, 0, padding))
                for name in ("q", "k", "v", "g", "beta_brc")
            },
        )
    chain = _chain_cpu(activated) if name == "cpu_chunk" else _chain_npu(activated)
    return chain.stage3.out[:, :, :tokens].contiguous(), chain.stage2.s.transpose(-1, -2).contiguous()


pytestmark = pytest.mark.npu
_LOWER_BOUND = float(os.environ.get("KDA_LOWER_BOUND") or -5.0)
_TOL = float(os.environ.get("KDA_TOL", "5e-3"))
_DYNAMIC_KDA = load_sample("flash_kda/flash_kda.py")
_DYNAMIC_METADATA = load_sample("flash_kda/flash_kda_metadata.py")


def require_npu():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")


@dataclasses.dataclass(frozen=True)
class _DynamicPrecisionCase:
    layout: str
    has_cu: bool
    lengths: tuple[int, ...]
    nv: int
    nk: int
    seed: int
    qk_scale: float = 1.0

    @property
    def case_id(self):
        lengths = "-".join(str(length) for length in self.lengths)
        suffix = "-cu" if self.has_cu else ""
        scale_label = f"_qks{self.qk_scale:g}" if self.qk_scale != 1.0 else ""
        return f"{self.layout.lower()}{suffix}_n{self.nv}k{self.nk}_s{lengths}_seed{self.seed}{scale_label}"


_DYNAMIC_FAST_CASES = (
    _DynamicPrecisionCase("BNSD", False, (65,), 1, 1, 101),
    _DynamicPrecisionCase("BSND", False, (65,), 1, 1, 102),
    _DynamicPrecisionCase("TND", True, (65,), 1, 1, 105),
    _DynamicPrecisionCase("BNSD", False, (65,), 16, 1, 201),
    _DynamicPrecisionCase("BSND", False, (65,), 16, 1, 202),
    _DynamicPrecisionCase("TND", True, (65,), 16, 1, 205),
    _DynamicPrecisionCase("BNSD", False, (1, 1), 3, 1, 301, qk_scale=1e-4),
    _DynamicPrecisionCase("BSND", False, (16, 16), 6, 2, 302, qk_scale=64.0),
    _DynamicPrecisionCase("TND", True, (1, 63, 65), 3, 1, 303),
    _DynamicPrecisionCase("BNSD", False, (63,), 96, 96, 304),
    _DynamicPrecisionCase("BSND", False, (64,), 96, 96, 305),
    _DynamicPrecisionCase("TND", True, (65,), 96, 96, 306),
    _DynamicPrecisionCase("BNSD", False, (127,), 6, 3, 307),
    _DynamicPrecisionCase("BSND", False, (128,), 6, 3, 308),
    _DynamicPrecisionCase("TND", True, (129, 64, 1), 6, 2, 309),
    _DynamicPrecisionCase("TND", True, (511, 513), 3, 1, 310),
    _DynamicPrecisionCase("BNSD", False, (513,), 3, 1, 311),
    _DynamicPrecisionCase("BSND", False, (513,), 3, 1, 312),
)


def _make_dynamic_case(case):
    inputs = make_inputs_generalized(
        len(case.lengths), case.nv, max(case.lengths), 128, nk=case.nk, seed=case.seed, lower_bound=_LOWER_BOUND,
    )
    inputs.q.mul_(case.qk_scale)
    inputs.k.mul_(case.qk_scale)
    outputs = {name: [] for name in ("cpu_chunk", "npu_chunk")}
    states = {name: [] for name in outputs}
    for batch_idx, length in enumerate(case.lengths):
        batch_slice = slice(batch_idx, batch_idx + 1)
        one = dataclasses.replace(
            inputs, k=inputs.k[batch_slice, :, :length], v=inputs.v[batch_slice, :, :length],
            q=inputs.q[batch_slice, :, :length], beta_brc=inputs.beta_brc[batch_slice, :, :length],
            g=inputs.g[batch_slice, :, :length], initial_state=inputs.initial_state[batch_slice],
        )
        for name in outputs:
            output, state = compute_golden(name, one)
            outputs[name].append(output)
            states[name].append(state)
    return (
        inputs,
        outputs,
        {name: torch.cat(values, dim=0) for name, values in states.items()},
    )


def _dynamic_layout_inputs(inputs, case):
    beta = inputs.beta_brc.squeeze(-1)
    if case.layout == "BNSD":
        return inputs.q, inputs.k, inputs.v, inputs.g, beta
    if case.layout == "BSND":
        return tuple(tensor.transpose(1, 2).contiguous() for tensor in (inputs.q, inputs.k, inputs.v, inputs.g, beta))

    def packed(tensor):
        return torch.cat(
            [tensor[b, :, :length].transpose(0, 1) for b, length in enumerate(case.lengths)], dim=0,
        ).contiguous()

    return tuple(packed(tensor) for tensor in (inputs.q, inputs.k, inputs.v, inputs.g, beta))


def _dynamic_valid_outputs(output, case):
    if case.layout == "TND":
        result = []
        start = 0
        for length in case.lengths:
            end = start + length
            result.append(output[start:end].transpose(0, 1).unsqueeze(0))
            start += length
        return result
    if case.layout == "BSND":
        output = output.transpose(1, 2)
    return [
        output[slice(batch_idx, batch_idx + 1), :, :length]
        for batch_idx, length in enumerate(case.lengths)
    ]


def _assert_dynamic_precision_case(case, *, template="auto"):
    require_npu()
    inputs, golden_outputs, golden_state = _make_dynamic_case(case)
    q, k, v, g, beta = (tensor.npu() for tensor in _dynamic_layout_inputs(inputs, case))
    cu_seqlens = None
    if case.has_cu:
        cu_seqlens = torch.tensor((0, *torch.tensor(case.lengths).cumsum(0).tolist()), dtype=torch.int32).npu()
    initial_state = inputs.initial_state.transpose(-1, -2).contiguous().npu()
    initial_state_before = initial_state.clone()
    metadata = _DYNAMIC_METADATA.flash_kda_metadata(
        q, v, initial_state, case.layout, cu_seqlens, template=template,
    )
    output, state = _DYNAMIC_KDA.flash_kda(
        q, k, v, g, beta, inputs.scale_value, initial_state, inputs.a_log.npu(), inputs.dt_bias.npu(),
        inputs.lower_bound, case.layout, metadata,
    )
    torch.npu.synchronize()

    actual_state = state.cpu().float()
    assert torch.isfinite(actual_state).all(), f"{case.case_id}: final_state contains inf/nan"
    torch.testing.assert_close(initial_state.cpu().float(), initial_state_before.cpu().float(), atol=0.0, rtol=0.0)
    actual_outputs = [actual.cpu().float() for actual in _dynamic_valid_outputs(output, case)]
    for name in ("cpu_chunk", "npu_chunk"):
        expected_state = golden_state[name].float()
        assert torch.isfinite(expected_state).all(), f"{case.case_id}: {name} state is not finite"
        torch.testing.assert_close(
            actual_state, expected_state, atol=_TOL, rtol=_TOL, msg=lambda msg: f"DSL vs {name} state: {msg}",
        )
        output_max = 0.0
        for actual, expected in zip(actual_outputs, golden_outputs[name]):
            assert torch.isfinite(actual).all(), f"{case.case_id}: DSL output is not finite"
            assert torch.isfinite(expected).all(), f"{case.case_id}: {name} output is not finite"
            torch.testing.assert_close(
                actual, expected.float(), atol=_TOL, rtol=_TOL, msg=lambda msg: f"DSL vs {name} output: {msg}",
            )
            output_max = max(output_max, (actual - expected).abs().amax().item())
        state_max = (actual_state - expected_state).abs().amax().item()
        print(f"{case.case_id} vs {name}: O max_abs={output_max:.3e} state max_abs={state_max:.3e}")
    torch.testing.assert_close(
        golden_state["npu_chunk"], golden_state["cpu_chunk"], atol=_TOL, rtol=_TOL,
        msg=lambda msg: f"npu_chunk vs cpu_chunk state: {msg}",
    )
    for simulated, fp32 in zip(golden_outputs["npu_chunk"], golden_outputs["cpu_chunk"]):
        torch.testing.assert_close(
            simulated, fp32, atol=_TOL, rtol=_TOL, msg=lambda msg: f"npu_chunk vs cpu_chunk output: {msg}",
        )


@pytest.mark.parametrize("template", ("auto", "staged", "persistent"))
@pytest.mark.parametrize("case", _DYNAMIC_FAST_CASES, ids=lambda case: case.case_id)
def test_flash_kda_dynamic_precision(case, template):
    _assert_dynamic_precision_case(case, template=template)


def teardown_module():
    _DYNAMIC_KDA.clear_caches()
    _DYNAMIC_METADATA.clear_metadata_caches()
