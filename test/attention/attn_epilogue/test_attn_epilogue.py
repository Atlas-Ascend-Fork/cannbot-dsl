# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bounded CI coverage for precision, FP4 weight layout and output metadata.

The broader token sweep, intermediate-byte checks, graph stress and profiler
utilities are maintained outside the repository under local validation/.
"""

# fmt: off

from functools import lru_cache

import pytest
import torch
from _samples_path import load_sample


torch_npu = pytest.importorskip("torch_npu")
op = load_sample("attention/attn_epilogue/attn_epilogue.py")


@lru_cache(maxsize=3)
def _weights(ng):
    """Quantize one deterministic ND weight pair per local group count."""
    rng = torch.Generator().manual_seed(2026)
    wa = (torch.randn(ng, 1024, 4096, generator=rng) * 0.3).bfloat16().npu()
    wb = (torch.randn(5120, ng * 1024, generator=rng) * 0.2).bfloat16().npu()
    qa, sa = torch_npu.npu_dynamic_mx_quant(wa.reshape(-1, 4096), dst_type=292)
    qb, sb = torch_npu.npu_dynamic_mx_quant(wb, dst_type=292)
    e8 = torch.float8_e8m0fnu
    return (
        qa.reshape(ng, 1024, 4096),
        qb,
        sa.view(e8).reshape(ng, 1024, 64, 2),
        sb.view(e8),
    )


def _inputs(t, ng):
    """Build BF16 activations and nonzero-position interleaved RoPE tables."""
    rng = torch.Generator().manual_seed(2027)
    x = (torch.randn(t, ng * 8, 512, generator=rng) * 0.3).bfloat16()
    angles = torch.outer(
        torch.arange(t).float() + 3, 10000.0 ** (-torch.arange(0, 64, 2).float() / 64)
    )
    return (
        x,
        angles.sin().repeat_interleave(2, -1),
        angles.cos().repeat_interleave(2, -1),
    )


def _native(x, sin, cos, weights):
    """Compute inverse RoPE and both native MX projections with BF16 MM1 rounding."""
    wa, wb, sa, sb = weights
    t, ng = x.shape[0], wa.shape[0]
    rotated = x.float()
    even, odd = rotated[..., 448::2].clone(), rotated[..., 449::2].clone()
    c, s = cos[:, None, ::2], sin[:, None, ::2]
    rotated[..., 448::2] = even * c + odd * s
    rotated[..., 449::2] = odd * c - even * s
    aq, asc = torch_npu.npu_dynamic_mx_quant(
        rotated.bfloat16().npu().reshape(t * ng, 4096), dst_type=292
    )
    e8 = torch.float8_e8m0fnu
    y = torch_npu.npu_transpose_quant_batchmatmul(
        aq.reshape(t, ng, 4096),
        wa,
        15,
        x1_scale=asc.view(e8).reshape(t, ng, 64, 2),
        x2_scale=sa,
        group_sizes=[1, 1, 32],
        perm_x1=[1, 0, 2],
        perm_x2=[0, 2, 1],
        perm_y=[1, 0, 2],
    ).reshape(t, ng * 1024)
    yq, ys = torch_npu.npu_dynamic_mx_quant(y, dst_type=292)
    return torch_npu.npu_quant_matmul(
        yq,
        wb.t(),
        sb.transpose(0, 1),
        pertoken_scale=ys.view(e8),
        output_dtype=torch.bfloat16,
        group_sizes=[1, 1, 32],
    )


def _as_nz(weights):
    """Prepack FP8 bytes into standard NZ and transpose their ND scale axes."""
    wa, wb, sa, sb = weights
    e8 = torch.float8_e8m0fnu
    packed = []
    for weight in (wa, wb):
        kn = weight.view(torch.uint8).cpu().transpose(-2, -1).contiguous().npu()
        packed.append(torch_npu.npu_format_cast(kn.view(torch.float8_e4m3fn), 29))
    sa = sa.view(torch.uint8).cpu().permute(0, 2, 1, 3).contiguous().npu().view(e8)
    sb = sb.view(torch.uint8).cpu().permute(1, 0, 2).contiguous().npu().view(e8)
    return *packed, sa, sb


@pytest.mark.npu
@pytest.mark.parametrize(
    "t,ng,nz",
    [
        pytest.param(1, 1, False, id="ng1_nd_t1"),
        pytest.param(16, 1, True, id="ng1_nz_t16"),
        pytest.param(17, 2, False, id="ng2_nd_tail"),
        pytest.param(1, 2, True, id="ng2_nz_t1"),
        pytest.param(72, 8, False, id="ng8_nd_t72"),
        pytest.param(17, 8, True, id="ng8_nz_tail"),
    ],
)
def test_attn_epilogue(t, ng, nz):
    """Check each group/layout once; NG=1 NZ also covers both graph interfaces."""
    torch.npu.set_device(0)
    x, sin, cos = _inputs(t, ng)
    weights = _weights(ng)
    expected = _native(x, sin, cos, weights)
    args = (x.npu(), *(_as_nz(weights) if nz else weights), sin.npu(), cos.npu())
    actual = op.attn_epilogue(*args)
    torch.testing.assert_close(
        actual.view(torch.int16).cpu(), expected.view(torch.int16).cpu(), rtol=0, atol=0
    )
    if ng == 1 and nz:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = op.attn_epilogue(*args)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            captured.view(torch.int16).cpu(),
            expected.view(torch.int16).cpu(),
            rtol=0,
            atol=0,
        )
        try:
            compiled = torch.compile(
                op.attn_epilogue,
                backend="npugraph_ex",
                fullgraph=True,
                dynamic=False,
                options={"clone_input": False},
            )
            for _ in range(3):
                result = compiled(*args)
                torch.testing.assert_close(
                    result.view(torch.int16).cpu(),
                    expected.view(torch.int16).cpu(),
                    rtol=0,
                    atol=0,
                )
        except Exception as exc:
            if "ProviderCallable cannot execute during DSL/MLIR tracing" in str(exc):
                pytest.skip(
                    "current cannbot-dsl ProviderCallable is not torch.compile-traceable"
                )
            raise


@pytest.mark.npu
@pytest.mark.parametrize("t,ng", [(17, 2), (257, 1)])
def test_fp4_woa_nz(t, ng):
    """Check the official FP4 NZ conversion and reject mismatched weight layouts."""
    torch.npu.set_device(0)
    x, sin, cos = _inputs(t, ng)
    _, wb, sa, sb = _weights(ng)
    rng = torch.Generator().manual_seed(2028)
    raw = torch.randint(256, (ng, 1024, 2048), dtype=torch.uint8, generator=rng)
    codebook = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ], dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8)
    decoded = torch.stack((codebook[(raw & 15).long()], codebook[(raw >> 4).long()]), dim=-1)
    woa_fp8 = decoded.reshape(ng, 1024, 4096).npu().view(torch.float8_e4m3fn)
    expected = _native(x, sin, cos, (woa_fp8, wb, sa, sb)).view(torch.int16).cpu()

    raw_npu = raw.npu()
    woa_nz = torch_npu.npu_format_cast(
        raw_npu, torch_npu.Format.FRACTAL_NZ,
        customize_dtype=torch.float8_e4m3fn, input_dtype=torch.float4_e2m1fn_x2,
    )
    assert woa_nz.shape == (ng, 1024, 2048) and woa_nz.dtype == torch.uint8
    assert torch_npu.get_npu_format(woa_nz) in (50, "FRACTAL_NZ_C0_16")
    _, wb_nz, _, sb_nz = _as_nz(_weights(ng))
    x, sin, cos = x.npu(), sin.npu(), cos.npu()
    actual = torch.ops.cannbotdsl.attn_epilogue(x, woa_nz, wb_nz, sa, sb_nz, sin, cos)
    torch.testing.assert_close(actual.view(torch.int16).cpu(), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="WOA and WOB must both use ND or both use NZ"):
        op.attn_epilogue(x, woa_nz, wb, sa, sb, sin, cos)
    with pytest.raises(ValueError, match="FP4 WOA requires uint8 FRACTAL_NZ_C0_16"):
        op.attn_epilogue(x, raw_npu, wb_nz, sa, sb_nz, sin, cos)


@pytest.mark.parametrize("ng,nz", [(8, False), (2, True)])
def test_meta(ng, nz):
    """Check output metadata using logical ND/NZ shapes without NPU access."""
    t = 17
    dtype, e8 = torch.float8_e4m3fn, torch.float8_e8m0fnu
    shapes = [
        ((t, ng * 8, 512), torch.bfloat16),
        ((ng, 4096, 1024) if nz else (ng, 1024, 4096), dtype),
        ((ng * 1024, 5120) if nz else (5120, ng * 1024), dtype),
        ((ng, 64, 1024, 2) if nz else (ng, 1024, 64, 2), e8),
        ((ng * 16, 5120, 2) if nz else (5120, ng * 16, 2), e8),
        ((t, 64), torch.float32),
        ((t, 64), torch.float32),
    ]
    args = [torch.empty(shape, dtype=dt, device="meta") for shape, dt in shapes]
    result = torch.ops.cannbotdsl.attn_epilogue(*args)
    assert result.shape == (t, 5120) and result.dtype == torch.bfloat16
