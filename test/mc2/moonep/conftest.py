# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Distributed-test session cleanup + 测试统一组创建 fixture。

Flushes test output and forwards the pytest exit status during process shutdown."""

import pytest  # noqa: E402
import torch  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "full: 全量矩阵用例（默认门禁不跑，--full 启用）"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--full", action="store_true", help="跑全量矩阵（含 full 标记用例）"
    )


# ---------------------------------------------------------------------------
# 测试统一组创建——session 级 fixture
#
# pytest 规范位置：fixture 自动发现、免 import、测试作用域明确；
# 每 rank 的 pytest 进程独立实例化、用例顺序全 rank 一致。
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def moonep_groups(request):
    """创建/复用测试用 HCCL 子组 + EB 组（C5+C6 适配，session 级）。

    首次请求建组（进程级缓存——同 session 的多个测试共享同一组，
    免重复建组开销）；返回 ``(moonep_group, ep_group)`` 二元组。
    """
    import torch.distributed as dist

    from prefetch_utils import _init_ep_group  # noqa: E402

    _init_ep_group()  # torchrun bootstrap（幂等；skip 语义与用例内一致）
    world = dist.get_world_size()

    from torch_npu._C._distributed_c10d import ProcessGroupHCCL

    # moonep 组：数据面走私有 workspace，不占通信域 HCCL 外部 buffer——
    # 注入引擎下限（2 MB，覆盖 comm 控制面/建连握手）封顶测试进程显存
    # （组显存配置由建组方即本测试自理；不注入即用 HCCL 缺省配置）
    opts = ProcessGroupHCCL.Options()
    opts.hccl_config = {"hccl_buffer_size": 2}
    group = dist.new_group(
        backend="hccl",
        ranks=list(range(world)),
        use_local_synchronization=True,
        pg_options=opts,
    )

    # EB 组：EB 数据面自建注册内存（不占 HCCL 外部 buffer）；建组建议值
    # 直取 EB 包自身的静态方法（当前为默认值）。纯 DSL 验证环境（venv 未装
    # cann_ops_transformer——cp312 无配套 wheel）降级按下限注入（EB 后端
    # 用例不在该环境运行——DSL 侧用例零影响）
    try:
        from cann_ops_transformer.ops import ElasticBuffer

        eb_mb = ElasticBuffer.get_moe_ep_ccl_buffer_size(
            world,
            32,
            32,
            4 * world,
            8,
        )
    except (ImportError, ModuleNotFoundError):
        eb_mb = 2
    ep_opts = ProcessGroupHCCL.Options()
    ep_opts.hccl_config = {"hccl_buffer_size": eb_mb}
    ep_group = dist.new_group(
        backend="hccl",
        ranks=list(range(world)),
        use_local_synchronization=True,
        pg_options=ep_opts,
    )
    return (group, ep_group)


# ---------------------------------------------------------------------------
# 空槽哨兵统一填充（测试侧——库内不做创建期 fill）
#
# 语义：空预取槽（etc=-1）确定性对拍契约——副本行填已知哨兵后，
# 测试可断言"命中槽被推送正确写入、空槽未被越界写入"。
# 用法：prefetch 返回 views 后、下一轮 push 前调用。
# ---------------------------------------------------------------------------


def fill_empty_slots(epn, views):
    """副本行哨兵填充：1B dtype 按 view(uint8).fill_(255)、多字节浮点 fill_(-1)。

    仅填副本段 v[epn:]（home 行为真实参数，不覆写）。
    ``views`` 为 prefetch 返回的 ``(weights, scales)`` 裸二元组
    （scales 未量化时为 ``()``）。
    需 buf.barrier() 保证全组 fill 先于下一轮 push（多轮场景由调用方负责）。
    """
    weights, scales = views
    for v in (*weights, *scales):
        if v.dtype.itemsize == 1:
            v[epn:].view(torch.uint8).fill_(255)
        else:
            v[epn:].fill_(-1)


def fill_and_repush(ctx, etc, views, epn, *, slot=0, slots=1):
    """空槽哨兵协议（模块级 API 版）：fill → sym_barrier → home 行切片复推。

    复推重写命中槽、空槽保持哨兵（home 行切片 data_ptr 同址 ⇒ sync 零拷贝
    跳过）；scale 组同规则随行。sym_barrier 必要性：复推写对端副本槽，
    须保证对端 fill 已完成。
    """
    from moonep_context import sym_barrier
    from prefetch_weight import prefetch_weight

    fill_empty_slots(epn, views)
    sym_barrier(ctx)
    weights, scales = views
    prefetch_weight(
        ctx,
        etc,
        [v[:epn] for v in weights],
        epn,
        local_scales=([s[:epn] for s in scales] if scales else None),
        slot=slot,
        slots=slots,
    )
    torch.npu.synchronize()
