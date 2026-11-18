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

    # 独立子组建 ctx（workspace+连接按测试隔离；连接 tag 进程序列唯一化）
    # workspace 按形状精算（分配即所需——context 缺省仅 plan 交换区下界）
    from moonep_buffer import _workspace_need

    ctx = create_moonep_context(
        group=moonep_groups[0],
        workspace_bytes=_workspace_need(
            world_size, epn, w_elems, 1, torch.bfloat16, torch.bfloat16
        ),
    )
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


@pytest.mark.npu
def test_moonep_prefetch_weight_fp16_ring(moonep_groups):
    """2 组 fp16 + 环形清单：视图 dtype 跟随输入，命中槽与 home 行逐元素相等。"""
    rank, world_size = _init_ep_group()
    epn, w_elems = 1, 17
    ctx = create_moonep_context(group=moonep_groups[0], workspace_bytes=64 << 20)
    ctx.last_plan_shape = (32, 8)

    etc_cpu = torch.full((world_size, epn), -1, dtype=torch.int32)
    for dst in range(world_size):
        home = (dst + 1) % world_size
        etc_cpu[dst] = torch.arange(home * epn, (home + 1) * epn, dtype=torch.int32)
    etc = etc_cpu.npu()

    reals = []
    for g in range(2):
        w = torch.empty(epn, w_elems, dtype=torch.float16)
        for i in range(epn):
            w[i] = float((rank * 100 + rank * epn + i) * (g + 1))
        reals.append(w.npu())

    views_w, views_s = prefetch_weight(ctx, etc, reals, epn)
    assert views_s == ()
    from conftest import fill_and_repush

    fill_and_repush(ctx, etc, (views_w, views_s), epn)

    for g, view in enumerate(views_w):
        assert view.dtype == torch.float16
        view_cpu = view.cpu()
        assert torch.equal(view_cpu[:epn], reals[g].cpu())
        for j in range(epn):
            e = int(etc_cpu[rank][j])
            if e < 0:
                assert bool((view_cpu[epn + j] == -1.0).all())
                continue
            home = e // epn
            exp = torch.full(
                (w_elems,),
                float((home * 100 + e) * (g + 1)),
                dtype=torch.float16,
            )
            assert torch.equal(view_cpu[epn + j], exp)


def test_prefetch_weight_fp16_contract():
    """fp16 白名单、组前缀整除校验。"""
    epn = 3
    w = torch.zeros(epn, 7, dtype=torch.float16)
    flat, sizes, dtype = pw._normalize_groups((w, w), epn)
    assert sizes == (7, 7) and dtype is torch.float16
    assert tuple(flat[0].shape) == (epn, 7)

    from moonep_layout import group_slot_rows

    assert group_slot_rows(4, (200, 100)) == ((0, 16), (12, 24))
    with pytest.raises(ValueError, match="无法与前缀"):
        group_slot_rows(4, (200, 128))

    with pytest.raises(ValueError, match="epn 必须为正"):
        pw._normalize_groups((w, w), 0)
    with pytest.raises(TypeError, match="torch.Tensor"):
        pw._normalize_groups((w, None), epn)

    s = torch.zeros(epn, 32, dtype=torch.float8_e8m0fnu)
    with pytest.raises(ValueError, match="仅支持 fp8 权重"):
        pw._normalize_scales((s, s), epn, 2, torch.float16)
