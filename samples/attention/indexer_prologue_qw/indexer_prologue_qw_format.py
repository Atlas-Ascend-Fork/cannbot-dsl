# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ND to FRACTAL_NZ conversion for static weights. Not part of the kernel."""

from __future__ import annotations

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

# torch_npu.Format.FRACTAL_NZ. Must match FRACTAL_NZ in indexer_prologue_qw.py.
FRACTAL_NZ = 29


def to_nz(weight: torch.Tensor) -> torch.Tensor:
    """Convert one ND device weight to FRACTAL_NZ.

    ``wqb`` and ``weight_w`` are static, so convert them once at load time.
    Tests and callers that still hold ND weights use this. Do not call it per step.
    """
    if torch_npu is None:
        raise ImportError("torch_npu is required to convert weights to FRACTAL_NZ")
    if weight.device.type != "npu":
        raise ValueError("to_nz requires an NPU tensor")
    return torch_npu.npu_format_cast(weight, FRACTAL_NZ)
