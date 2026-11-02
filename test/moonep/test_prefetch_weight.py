# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""prefetch_weight 单独验收：Buffer 构造 API 与端到端数据正确性。

运行（4P）：
    python -m torch.distributed.run --nproc_per_node=4 -m pytest \\
        test/moonep/test_prefetch_weight.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "samples", "moonep")
)

import prefetch_weight as pw  # noqa: E402
from cannbotdsl.tensor import Layout  # noqa: E402
from moonep_context import create_moonep_context  # noqa: E402
from prefetch_weight import create_weight_view, prefetch_weight  # noqa: E402
from test_moonep_ep import (  # noqa: E402
    _H,
    _PrefetchCase,
    _fill_weight_real,
    _init_ep_group,
    _requires_dsl,
    _verify_prefetch_view,
)


@pytest.mark.npu
@_requires_dsl
def test_moonep_prefetch_weight_dsl_api(monkeypatch):
    """prefetch_weight 在 layout 版 DSL（0.8）下的端到端验收。

    绕过 moonep_plan：create_weight_view / normalize_group_list 只对
    context.last_plan_shape 做非空守卫，check_pending_overflow 在
    pending_overflow 为 None 时直接返回。这里手工补上该守卫字段并手工构造
    experts_to_copy，直接驱动 prefetch；真实 plan→prefetch 链路由
    test_moonep_ep.py 的用例覆盖（需 moonep_plan 迁移到 0.8 后才能运行）。
    """
    rank, world_size = _init_ep_group()
    num_tokens, top_k, epn = 32, 8, 4
    w_elems = _H * 17  # 非默认行宽，确保重新 trace，不复用已编译 program

    ctx = create_moonep_context()
    ctx.last_plan_shape = (num_tokens, top_k)  # 绕开 plan 的调用顺序守卫

    # 每行至多一个 home rank，空槽 -1（与 plan 输出格式一致）
    etc_cpu = torch.full((world_size, epn), -1, dtype=torch.int32)
    for dst in range(1, world_size):
        etc_cpu[dst] = torch.arange(epn, dtype=torch.int32)
    etc = etc_cpu.npu()

    reals = [_fill_weight_real(rank, epn, w_elems, g) for g in range(3)]
    views = [create_weight_view(ctx, epn, w_elems, group=g, groups=3) for g in range(3)]
    for view in views:
        view.fill_(-1.0)

    # 记录真实 Buffer 构造参数：证明确实走了 0.8 的 (mem_loc, dtype, layout) 形态
    seen = []
    real_buffer = pw.Buffer

    def _spy(location, dtype, layout, *args, **kwargs):
        seen.append((dtype, layout))
        return real_buffer(location, dtype, layout, *args, **kwargs)

    monkeypatch.setattr(pw, "Buffer", _spy)

    prefetch_weight(ctx, etc, reals, views, epn)
    torch.npu.synchronize()

    assert seen, "prefetch kernel 未重新 trace，Buffer 构造未被观测到"
    assert all(
        not isinstance(dtype, tuple) and isinstance(layout, Layout)
        for dtype, layout in seen
    ), f"0.8 Buffer 应收到 (dtype, Layout)，实得 {seen[:3]}"

    case = _PrefetchCase(rank, epn, w_elems)
    for g, view in enumerate(views):
        w_cpu = view.cpu()
        _verify_prefetch_view(w_cpu, etc_cpu, case, g=g, tag=f"dsl-api g{g} ")
        for i in range(epn):
            exp = float((rank * 1000 + rank * epn + i) * (g + 1))
            assert torch.equal(
                w_cpu[i], torch.full((w_elems,), exp, dtype=torch.bfloat16)
            ), f"第 {g} 组 home 行 {i} 未被同步写入"
