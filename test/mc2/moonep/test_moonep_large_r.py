# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""R>64 布局与块划分静态回归（不占卡）。

本机 8 卡无法实跑 R>64，这里守住两条可静态验证的不变量：
- 布局定宽区域不越界：到达槽区 [0, MAX_R*NOTIFY_STRIDE) ⊂ [0, HEAP_BASE)，
  plan 交换区上界随 MAX_R 放大；
- 头部屏障写者/读块封顶常驻核数后，轮转分配仍覆盖全部 peer（R 与块数解耦）。
"""

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "samples", "mc2", "moonep"
    ),
)

try:
    import torch_npu  # noqa: F401

    import reduce_grad as rg
    from moonep_layout import (
        HEAP_BASE,
        MAX_R,
        NOTIFY_STRIDE,
        PLAN_REGION_END,
    )
    from prefetch_weight import _push_layout

    _HAS_DSL_IMPL = True
except Exception:  # CI 无 NPU / 无 DSL 时收集不报错
    _HAS_DSL_IMPL = False

pytestmark = pytest.mark.skipif(not _HAS_DSL_IMPL, reason="需 cannbotdsl/torch_npu")


def _covered(n_rem, n_blk, per_blk):
    """块 b 分管 peer b, b+n_blk, ... 的并集（升序）。"""
    return sorted(
        {
            b + k * n_blk
            for b in range(n_blk)
            for k in range(per_blk)
            if b + k * n_blk < n_rem
        }
    )


def test_layout_covers_max_rank():
    """到达槽区与 plan 交换区在 MAX_R 下不重叠、不越界。"""
    assert MAX_R >= 128
    assert HEAP_BASE >= MAX_R * NOTIFY_STRIDE
    assert HEAP_BASE % 32 == 0
    assert PLAN_REGION_END > HEAP_BASE


def test_peer_block_split_covers_all_peers():
    """屏障写者/读块轮转覆盖全部 peer，且块数不超过常驻核数。"""
    for rank_size in (2, 8, 33, 64, 65, 66, MAX_R):
        n_rem = rank_size - 1
        n_writer, per_writer, n_read, per_read = rg.peer_block_split(rank_size, 64)
        assert n_writer == min(n_rem, 64)
        assert _covered(n_rem, n_writer, per_writer) == list(range(n_rem))
        assert n_read == n_writer
        assert _covered(n_rem, n_read, per_read) == list(range(n_rem))
        # reserve_merge：读块让出给归并时仍须覆盖全部 peer
        n_writer2, per_writer2, n_read2, per_read2 = rg.peer_block_split(
            rank_size, 64, reserve_merge=True
        )
        assert n_writer2 == n_writer
        assert 0 < n_read2 <= 64
        assert _covered(n_rem, n_read2, per_read2) == list(range(n_rem))


def test_prefetch_push_layout_covers_all_peers():
    """prefetch 的 clamp+轮转在 MAX_R 下同样覆盖全部 peer。"""
    for rank_size in (2, 8, 65, MAX_R):
        nblk, per_blk, n_slices = _push_layout(rank_size)
        n_rem = rank_size - 1
        assert nblk <= 64
        if n_slices > 1:
            # 行分片模式：peer = blk % n_rem、slice = blk // n_rem
            assert nblk == n_rem * n_slices
            assert n_slices <= 8
            assert {blk % n_rem for blk in range(nblk)} == set(range(n_rem))
        else:
            assert _covered(n_rem, nblk, per_blk) == list(range(n_rem))


def test_kernel_read_split_at_max_rank():
    """R=MAX_R：直累 kernel 每块分管多个 peer。"""

    acc = rg._UbmemReduceAccKernel(
        MAX_R,
        epn=1,
        sizes=(64, 64, 64),
        elem_bytes=(2, 2, 2),
        ngroups=2,
        copy_off=0,
        chunks=(64, 64, 64),
        blocks=64,
        grad_bf16=True,
    )
    assert acc._n_writer == min(MAX_R - 1, 64)
    assert acc._n_read == acc._blocks

    assert acc._per_read > 1
    assert _covered(MAX_R - 1, acc._n_read, acc._per_read) == list(range(MAX_R - 1))
