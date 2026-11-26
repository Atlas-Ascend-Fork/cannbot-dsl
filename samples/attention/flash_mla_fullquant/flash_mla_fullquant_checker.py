# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host-side scale contract for the flash_mla_fullquant samples.

The kernel folds ``softmax_scale * k_descale`` into one FP32 constant and
also uses it to scale the fully-masked-row sentinel.  A non-positive or
non-finite composite makes the sentinel and the row maximum collapse to the
same value, which would erase valid rows, so it is rejected at the entry.

The per-row factor ``q_descale * (softmax_scale * k_descale)`` is checked in
FP32 because that is what the kernel materializes.  A product that overflows
to inf turns zero logits into NaN, and one that underflows to 0 collapses the
sentinel, so both are rejected instead of silently changing the result.
"""

import math

import torch

DEFAULT_SOFTMAX_SCALE = 1.0 / math.sqrt(512.0)


def _check(condition, error, message):
    if not condition:
        raise error(message)


def validate_scales(q_descale, k_descale, softmax_scale):
    """Validate the scales that fold into the FP32 softmax constant.

    Returns ``(softmax_scale, k_descale, softmax_scale * k_descale)``.
    """
    _check(
        q_descale.dtype == torch.float32,
        TypeError,
        f"q_descale must be float32, got {q_descale.dtype}",
    )
    _check(
        bool(torch.isfinite(q_descale).all()),
        ValueError,
        "q_descale must be finite",
    )
    _check(
        bool((q_descale > 0).all()),
        ValueError,
        "q_descale must be positive",
    )
    _check(
        k_descale.dtype == torch.float32 and tuple(k_descale.shape) == (1,),
        TypeError,
        f"k_descale must be float32 (1,), got {k_descale.dtype} "
        f"{tuple(k_descale.shape)}",
    )
    kd = float(k_descale.detach().cpu().reshape(()))
    _check(math.isfinite(kd), ValueError, f"k_descale must be finite, got {kd}")
    _check(kd > 0.0, ValueError, f"k_descale must be positive, got {kd}")

    ss = DEFAULT_SOFTMAX_SCALE if softmax_scale is None else float(softmax_scale)
    _check(
        math.isfinite(ss),
        ValueError,
        f"softmax_scale must be finite, got {ss}",
    )
    qk_scale = ss * kd
    _check(
        math.isfinite(qk_scale) and qk_scale > 0.0,
        ValueError,
        f"softmax_scale * k_descale must be finite and positive, got {qk_scale!r}",
    )
    _check(
        qk_scale <= float(torch.finfo(torch.float32).max),
        ValueError,
        f"softmax_scale * k_descale exceeds float32 range, got {qk_scale!r}",
    )
    qk_scale_f32 = torch.tensor(qk_scale, dtype=torch.float32)
    _check(
        float(qk_scale_f32) > 0.0,
        ValueError,
        f"softmax_scale * k_descale underflows float32, got {qk_scale!r}",
    )
    row_factor = q_descale.detach().cpu() * qk_scale_f32
    out_of_range = ~(torch.isfinite(row_factor) & (row_factor > 0.0))
    _check(
        not bool(out_of_range.any()),
        ValueError,
        "q_descale * (softmax_scale * k_descale) must be finite and positive "
        f"in float32, got {int(out_of_range.sum())} out-of-range row(s)",
    )
    return ss, kd, qk_scale
