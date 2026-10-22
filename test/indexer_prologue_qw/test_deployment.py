# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""NPU precision for the three kept deployment shapes.

Split-K covers T 72 and T 2048. The smallest split-T deployment shape is T 4096.
Larger T values check every row on the CPU golden and dominate the runtime.
"""

from __future__ import annotations

import os

import pytest
import torch

from ipqw_verify import origins


def _kept(phase: str) -> tuple[int, ...]:
    kept = (72,) if phase == "decode" else (2048, 4096)
    forced = os.environ.get("IPQW_TEST_T")
    if not forced:
        return kept
    want = {int(v) for v in forced.replace(" ", "").split(",") if v}
    return tuple(t for t in kept if t in want)


@pytest.fixture(autouse=True)
def _require_npu():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("no NPU visible")
    # CI can report an available NPU without binding an ACL context to the
    # pytest thread.  Bind it before the first tensor ``.npu()`` conversion
    # rather than relying on LazySetDevice during a test body.
    torch.npu.set_device(0)


def _check_shape(t: int) -> None:
    from ipqw_verify import check_shape
    check_shape(t)


@pytest.mark.npu
@pytest.mark.parametrize(
    "t", _kept("decode"), ids=lambda t: f"T{t}_{'_'.join(origins('decode', t))}"
)
def test_decode_shapes(t: int) -> None:
    _check_shape(t)


@pytest.mark.npu
@pytest.mark.parametrize(
    "t", _kept("prefill"), ids=lambda t: f"T{t}_{'_'.join(origins('prefill', t))}"
)
def test_prefill_shapes(t: int) -> None:
    """T 2048 is split-K. T 4096 is the smallest split-T deployment shape."""
    _check_shape(t)
