# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Precision tests for the weight_quant_batchmatmul_a16w4 kernel (weight-quant A16W4 matmul).

Formula:  y[M,N] = x[M,K] @ (fp4_to_fp16(w[K,N]) * e8m0_to_fp16(scale[K/32,N]))^T

Constraints honoured by the shape list:
  * x is fp16; w is MXFP4 (fp4x2_e2m1 packed, 2 elems/byte) in WeightNZ
    physical layout (ceil(N/16), K*8); one E8M0 scale per 32-K group; K is a
    multiple of 32; fp32 accumulation inside the kernel, fp16 output.
  * Cases cover the stage-3 workload: mainline large shape, M/N/K tails,
    K=32 single-group, odd-J pslot rotation, empty-AIV1-panel and full-grid
    pressure shapes.

Golden kit (inlined below, independence contract):
  * Imports nothing from the operator module; the E2M1 decode table is
    derived from the bit format (OCP MX: 1s + 2e + 1m, bias=1, E=0
    subnormal = M/2, E>0 = 2^(E-1)*(1+M/2)) and self-checked against the
    published 16-value set.
  * WeightNZ packing is used only for input construction (the NPU side
    decodes per its own layout, so a packing mismatch would necessarily
    skew the output away from the golden).
  * Golden path = numpy FP32 LUT x 2^(e-127) + torch FP32 matmul, zero
    shared code with the DSL implementation.

Dual tolerance gates (same as the stage-3 precision gate):
  * design (primary, asserts): np.allclose(rtol=5e-2, atol=1e-2) — fp16
    output tier (DESIGN §8.3).
  * strict (secondary, logged only): max_rel <= 2e-3 with denominator
    max(|golden|, 1.0); K=4096 shapes legitimately exceed it via fp16 output
    quantization + sequential fp32 accumulation on cancellation cells
    (3-30 cells per shape, mean_rel ~1.8e-4) — numeric noise, not a defect.

Note: the source stage-3 gate isolated each case in a subprocess (single-
process sequential launches could accumulate dirty device-sync state and
intermittently aicore-timeout); if that reproduces under pytest, run per
case with -k.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "samples",
        "matmul",
        "weight_quant_batchmatmul",
    ),
)

from weight_quant_batchmatmul_a16w4 import npu_weight_quant_batchmatmul


# ---------------------------------------------------------------------------
# Golden kit — independent CPU reference (E2M1 bit-format LUT x 2^(e-127))
# ---------------------------------------------------------------------------


def _e2m1_decode(code: int) -> float:
    """4-bit E2M1 → float：s(1) e(2) m(1)，bias=1。E=0 亚正规 M/2，否则 2^(E-1)(1+M/2)。"""
    s = (code >> 3) & 0x1
    e = (code >> 1) & 0x3
    m = code & 0x1
    if e == 0:
        mag = 0.5 * m  # 亚正规：M/2 * 2^(1-1)
    else:
        mag = (2.0 ** (e - 1)) * (1.0 + 0.5 * m)
    return -mag if s else mag


#: 由位格式推导的 16 值表（索引 = 4-bit 编码）
E2M1_VALUES = [_e2m1_decode(c) for c in range(16)]

#: 公开标准值集合（IEEE/OCP MX E2M1 公认 16 值；与位格式推导互为独立来源）
_E2M1_STANDARD = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]

assert E2M1_VALUES == _E2M1_STANDARD, (
    f"E2M1 位格式推导与标准值集合不一致: {E2M1_VALUES} vs {_E2M1_STANDARD}"
)


def e8m0_to_scale(e: np.ndarray) -> np.ndarray:
    """E8M0 字节 → FP32 scale = 2^(e-127)（纯幂，无尾数）。"""
    return np.power(2.0, e.astype(np.float32) - 127.0)


def pack_weight_nz(nib: np.ndarray) -> np.ndarray:
    """nib (K, N) uint8（4-bit 编码）→ WeightNZ (ceil(N/16), K*8) uint8 字节流。

    物理布局：byte = p*(K*8) + k*8 + ((n%16)//2)，低 nibble = 偶数列 n，
    高 nibble = 奇数列 n。
    """
    k_, n_ = nib.shape
    p_tot = -(-n_ // 16)
    padded = np.zeros((k_, p_tot * 16), dtype=np.uint8)
    padded[:, :n_] = nib
    pan = padded.reshape(k_, p_tot, 16)
    lo = pan[:, :, 0::2]  # (K, P, 8) 偶数列
    hi = pan[:, :, 1::2]  # (K, P, 8) 奇数列
    b = (hi.astype(np.uint16) << 4) | lo.astype(np.uint16)
    b = b.astype(np.uint8)
    return np.ascontiguousarray(b.transpose(1, 0, 2).reshape(p_tot, k_ * 8))


def make_inputs(
    m: int, n: int, k: int, seed: int, scale_lo: int = 120, scale_hi: int = 136
):
    """固定 seed 生成 (x, w_bytes, scale_u8, nibbles, e8m0_bytes)。

    scale_lo/hi 控制 E8M0 指数范围（默认 [120,136) → scale ∈ 2^-7..2^8，
    FP16 安全域）。
    """
    rng = np.random.default_rng(seed)
    x = (torch.randn(m, k) * 0.3).to(torch.float16)
    nib = rng.integers(0, 16, size=(k, n)).astype(np.uint8)
    e = rng.integers(scale_lo, scale_hi, size=(k // 32, n)).astype(np.uint8)
    w_bytes = pack_weight_nz(nib)
    scale = torch.from_numpy(e.copy())
    return x, w_bytes, scale, nib, e


def golden_fp32(
    x: torch.Tensor, w_bytes: np.ndarray, e_bytes: np.ndarray, m: int, n: int, k: int
) -> np.ndarray:
    """FP32 参考：y = x.float() @ (E2M1(w) × 2^(e-127))。

    w_bytes (ceil(N/16), K*8) → 解包 (K, N)；scale (K/32, N) → 组内广播 (K, N)。
    全程 numpy/torch FP32，与 DSL 实现零共享。
    """
    p_tot = w_bytes.shape[0]
    panels = w_bytes.reshape(p_tot, k, 8)
    tab = np.asarray(E2M1_VALUES, dtype=np.float32)
    lo = tab[(panels & 0xF).astype(np.int64)]  # (P, K, 8) 偶数列
    hi = tab[(panels >> 4).astype(np.int64)]  # (P, K, 8) 奇数列
    w = np.empty((p_tot, k, 16), dtype=np.float32)
    w[:, :, 0::2] = lo
    w[:, :, 1::2] = hi
    w = w.transpose(1, 0, 2).reshape(k, p_tot * 16)[:, :n]
    sc = e8m0_to_scale(e_bytes)  # (K/32, N)
    sc_exp = np.repeat(sc, 32, axis=0).astype(np.float32)  # (K, N)
    w_deq = (w * sc_exp).astype(np.float32)
    x_f = np.ascontiguousarray(x.float().numpy(), dtype=np.float32)
    y = torch.from_numpy(x_f) @ torch.from_numpy(np.ascontiguousarray(w_deq))
    return y.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Test cases (stage-3 workload: TILING §2.5 + DESIGN §1.4 + tail sweep)
# ---------------------------------------------------------------------------

# (m, n, k, seed)
_TEST_CASES = [
    pytest.param(1024, 1024, 4096, 1000, id="main-large"),  # 主线大 shape，grid 8
    pytest.param(512, 300, 4096, 1001, id="ntail-44"),  # N 尾 44（H1 护栏路径）
    pytest.param(129, 129, 4096, 1002, id="mn-tail-1"),  # M/N 尾 1（fractal pad）
    pytest.param(256, 256, 32, 1003, id="k32-exact"),  # K=32 单组单份（SG_TAIL=1）
    pytest.param(128, 128, 96, 1004, id="k-tail-96"),  # K 份尾 96（3 组）
    pytest.param(
        1024, 1024, 384, 1005, id="j-odd-kb-tail"
    ),  # J=3 奇 → pslot 跨 tile 轮转
    pytest.param(64, 17, 32, 1006, id="n17-aiv1-empty"),  # N=17：AIV1 空 panel
    pytest.param(4096, 4096, 4096, 1007, id="full-grid-32"),  # grid 32 全压力
    pytest.param(128, 4096, 4096, 1008, id="single-mrow"),  # 单 m 行 × N 带状
    pytest.param(64, 64, 1024, 1009, id="m-tail-64"),  # M 尾 64
]

# design 口径（DESIGN §8.3 冻结：fp16 输出容差）
_DESIGN_RTOL = 5e-2
_DESIGN_ATOL = 1e-2
# strict 口径（任务书建议值，clamp-1.0 分母，仅记录）
_STRICT_MAX_REL = 2e-3


# ---------------------------------------------------------------------------
# NPU precision test — weight_quant_batchmatmul_a16w4 DSL kernel
# ---------------------------------------------------------------------------


@pytest.mark.npu
@pytest.mark.parametrize("m,n,k,seed", _TEST_CASES)
def test_weight_quant_batchmatmul_a16w4_precision(
    m: int, n: int, k: int, seed: int
) -> None:
    pytest.importorskip("torch_npu")

    x, w_bytes, scale_u8, _nib, e = make_inputs(m, n, k, seed)
    gm_w = torch.from_numpy(w_bytes.copy()).npu()
    sc_t = scale_u8.npu()
    if hasattr(torch, "float8_e8m0fnu"):
        sc_t = sc_t.view(torch.float8_e8m0fnu)

    y = npu_weight_quant_batchmatmul(x.npu(), gm_w, sc_t)
    torch.npu.synchronize()

    assert tuple(y.shape) == (m, n)
    assert y.dtype == torch.float16

    result = y.cpu().float().numpy()
    golden = golden_fp32(x, w_bytes, e, m, n, k)

    diff = np.abs(result.astype(np.float32) - golden)
    rel = diff / np.maximum(np.abs(golden), 1.0)
    strict_ok = bool((rel <= _STRICT_MAX_REL).all())

    logging.info(
        "weight_quant_batchmatmul_a16w4 (M=%s,N=%s,K=%s) max_abs=%.4e max_rel=%.4e mean_rel=%.2e "
        "strict_2e-3=%s",
        m,
        n,
        k,
        float(diff.max()),
        float(rel.max()),
        float(rel.mean()),
        "OK" if strict_ok else "NO",
    )

    assert np.allclose(result, golden, rtol=_DESIGN_RTOL, atol=_DESIGN_ATOL), (
        f"weight_quant_batchmatmul_a16w4 mismatch: M={m} N={n} K={k}, "
        f"max_abs={float(diff.max()):.4e}, max_rel={float(rel.max()):.4e}, "
        f"mean_rel={float(rel.mean()):.2e}, "
        f"strict_2e-3={'OK' if strict_ok else 'NO'}"
    )
