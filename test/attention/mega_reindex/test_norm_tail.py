# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Isolated RMSNorm tail regression; does not extend the public MX shape ABI."""

import sys
from pathlib import Path

import cannbotdsl as cb
import pytest
import torch
from cannbotdsl import dtypes as dt
from cannbotdsl import simt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _samples_path import load_sample

_front = load_sample("attention/mega_reindex/reindex_front.py")
_runtime = load_sample("attention/mega_reindex/reindex_runtime.py")


class NormProbe:
    def __init__(self, dim):
        self.dim = dim
        self.norm = _front.NormPhase(1, 1e-20, 1 / 64, 1)
        # Exercise only the normalization helper; MX quantization has its own ABI.
        self.norm.norm_dim = dim
        self.norm.norm_chunks = dim // _front.NORM_VECTOR_LANES
        self.norm.norm_tail = dim % _front.NORM_VECTOR_LANES
        self.padded = (
            (dim + _front.NORM_VECTOR_LANES - 1)
            // _front.NORM_VECTOR_LANES
            * _front.NORM_VECTOR_LANES
        )

    @cb.kernel
    def kernel(self, x: cb.Tensor, gamma: cb.Tensor, out: cb.Tensor):
        xx = _runtime.Buffer(cb.MemLoc.UB, (1, self.padded), dt.bfloat16)
        gg = _runtime.Buffer(cb.MemLoc.UB, (1, self.padded), dt.bfloat16)
        yy = _runtime.Buffer(cb.MemLoc.UB, (1, self.padded), dt.bfloat16)
        tmp = _runtime.Buffer(cb.MemLoc.UB, (_front.NORM_VECTOR_LANES,), dt.float32)
        # Poison unused lanes so accidental participation changes the answer.
        with cb.vf(mode="simt", thread=64):
            tid = dt.int32(simt.thread_idx()[0])
            for i in range(tid, self.padded, 64):
                xx[0, i] = dt.bfloat16(10000.0)
                gg[0, i] = dt.bfloat16(10000.0)
        cb.mem_copy(xx.view(self.padded)[: self.dim,], x)
        cb.mem_copy(gg.view(self.padded)[: self.dim,], gamma)
        self.norm._normalize(xx, gg, yy, tmp)
        cb.mem_copy(out[: self.dim,], yy.view(self.padded)[: self.dim,])

    @cb.host
    def run(self, x, gamma, out):
        self.kernel[1](x, gamma, out)


@pytest.mark.npu
@pytest.mark.parametrize("dim", [31, 64, 65, 1279, 1280])
def test_rms_norm_tail(dim):
    op = NormProbe(dim)
    program = cb.compile(
        op.run,
        cb.TensorSpec((dim,), dt.bfloat16),
        cb.TensorSpec((dim,), dt.bfloat16),
        cb.TensorSpec((dim + 16,), dt.bfloat16),
    )
    torch.manual_seed(dim)
    gamma = (torch.rand(dim) + 0.5).bfloat16()
    scrub = torch.ones(200_000_000, device="npu", dtype=torch.float32)
    for x in (torch.ones(dim, dtype=torch.bfloat16), torch.randn(dim).bfloat16()):
        out = torch.full((dim + 16,), -1234.0, device="npu", dtype=torch.bfloat16)
        torch.sum(scrub)
        torch.npu.synchronize()
        program(x.npu(), gamma.npu(), out)
        torch.npu.synchronize()
        expected = (
            x.float()
            * torch.sqrt(1 / (x.float().square().mean() + 1e-20))
            * gamma.float()
        ).bfloat16()
        torch.testing.assert_close(out[:dim].cpu(), expected, rtol=0.008, atol=1e-4)
        assert torch.equal(
            out[dim:].cpu(), torch.full((16,), -1234.0, dtype=torch.bfloat16)
        )
