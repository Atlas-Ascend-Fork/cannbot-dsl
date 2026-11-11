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
        test/mc2/moonep/test_prefetch_weight.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "samples", "mc2", "moonep"
    ),
)

# 无 NPU 环境收集期兼容：守卫导入 + pytestmark skipif（裸 import
# prefetch_weight → moonep_context → torch_npu 链在无 NPU 环境收集期失败）
try:
    import prefetch_weight as pw  # noqa: E402
    from cannbotdsl.tensor import Layout  # noqa: E402
    from moonep_context import create_moonep_context  # noqa: E402
    from prefetch_weight import prefetch_weight  # noqa: E402

    _HAS_DSL_IMPL = True
except ImportError:
    _HAS_DSL_IMPL = False

pytestmark = pytest.mark.skipif(
    not _HAS_DSL_IMPL, reason="requires NPU environment (cannbotdsl/torch_npu/HCCL)"
)
from prefetch_utils import (  # noqa: E402
    _H,
    _PrefetchCase,
    _fill_weight_real,
    _init_ep_group,
    _verify_prefetch_view,
)


@pytest.mark.npu
def test_moonep_prefetch_weight_dsl_api(moonep_groups, monkeypatch):
    """prefetch_weight 在 layout 版 DSL（0.8）下的端到端验收。

    显式设置 context.last_plan_shape 并构造 experts_to_copy，独立验证
    权重预取。
    """
    rank, world_size = _init_ep_group()
    num_tokens, top_k, epn = 32, 8, 4
    w_elems = _H * 17  # 非默认行宽，确保重新 trace，不复用已编译 program

    # 独立子组建 ctx：裸 WORLD 会与本 session 其它文件已建的 EB 通信器
    # 撞车（对称堆互踩守卫拒绝）
    ctx = create_moonep_context(group=moonep_groups[0])
    ctx.last_plan_shape = (num_tokens, top_k)  # 设置窗口布局元数据

    # 每行至多一个 home rank，空槽填 -1
    etc_cpu = torch.full((world_size, epn), -1, dtype=torch.int32)
    for dst in range(1, world_size):
        etc_cpu[dst] = torch.arange(epn, dtype=torch.int32)
    etc = etc_cpu.npu()

    reals = [_fill_weight_real(rank, epn, w_elems, g) for g in range(3)]

    # 记录真实 Buffer 构造参数：证明确实走了 0.8 的 (mem_loc, dtype, layout) 形态
    seen = []
    real_buffer = pw.Buffer

    def _spy(location, dtype, layout, *args, **kwargs):
        seen.append((dtype, layout))
        return real_buffer(location, dtype, layout, *args, **kwargs)

    monkeypatch.setattr(pw, "Buffer", _spy)

    # 视图随调用按各组行宽创建并返回（统一 (views, event) 返回）
    views_w, views_s = prefetch_weight(ctx, etc, reals, epn)
    # 空槽哨兵（库内 fill 已移除——bcdf4d8）：fill → barrier → 复推
    from conftest import fill_and_repush

    fill_and_repush(ctx, etc, (views_w, views_s), epn)

    assert seen, "prefetch kernel 未重新 trace，Buffer 构造未被观测到"
    assert all(
        not isinstance(dtype, tuple) and isinstance(layout, Layout)
        for dtype, layout in seen
    ), f"0.8 Buffer 应收到 (dtype, Layout)，实得 {seen[:3]}"

    case = _PrefetchCase(rank, epn, w_elems)
    for g, view in enumerate(views_w):
        w_cpu = view.cpu()
        _verify_prefetch_view(w_cpu, etc_cpu, case, g=g, tag=f"dsl-api g{g} ")
        for i in range(epn):
            exp = float((rank * 1000 + rank * epn + i) * (g + 1))
            assert torch.equal(
                w_cpu[i], torch.full((w_elems,), exp, dtype=torch.bfloat16)
            ), f"第 {g} 组 home 行 {i} 未被同步写入"
