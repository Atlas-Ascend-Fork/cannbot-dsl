# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Standalone weight-prefetch test data and verification helpers."""

import os
from typing import NamedTuple

import pytest
import torch
import torch.distributed as dist

_H = 32


def _init_ep_group():
    """torchrun 拉起时初始化 HCCL 组；单进程直跑则 skip。"""
    pytest.importorskip("torch_npu")

    if not dist.is_initialized():
        if "MASTER_ADDR" not in os.environ or "RANK" not in os.environ:
            pytest.skip("需 torchrun --nproc_per_node=N（N>=2）拉起")
        dist.init_process_group(backend="hccl")
    world = dist.get_world_size()
    if world < 2:
        pytest.skip("EP 验收需 >=2 卡")
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    return rank, world


def _assert_one_source_etc(etc_cpu, epn):
    """专家清单要求每个 destination rank 的副本槽至多来自一个 home rank。"""
    for rank, row in enumerate(etc_cpu.tolist()):
        homes = {int(e) // epn for e in row if int(e) >= 0}
        assert len(homes) <= 1, (
            f"etc[{rank}] 同时来自多个 home rank：{sorted(homes)}（一卡一源契约被破坏）"
        )


def _fill_weight_real(rank, epn, w_elems, g=0):
    """权重编码值张量：行 i = (rank*1000 + rank*epn + i) * (g+1)。"""
    w_real = torch.empty(epn, w_elems, dtype=torch.bfloat16)
    for i in range(epn):
        w_real[i] = float((rank * 1000 + rank * epn + i) * (g + 1))
    return w_real.npu()


class _PrefetchCase(NamedTuple):
    """prefetch 校验上下文：rank / epn / 行宽。"""

    rank: int
    epn: int
    w_elems: int


def _verify_prefetch_view(w_cpu, etc_cpu, case, g=0, tag=""):
    """校验单个权重视图：副本槽 == home 发布值 (home*1000+e)*(g+1)；空槽保持 -1。"""
    _assert_one_source_etc(etc_cpu, case.epn)
    rank, epn, w_elems = case.rank, case.epn, case.w_elems
    for j in range(epn):
        e = int(etc_cpu[rank][j])
        if e < 0:
            # 未承载副本的预取槽：prefetch 不写（必须保留 -1 初值不被踩踏）
            assert bool((w_cpu[epn + j] == -1.0).all()), f"{tag}空预取槽 {j} 被意外写入"
            continue
        home = e // epn
        exp_val = float((home * 1000 + e) * (g + 1))
        got = w_cpu[epn + j].float()
        if not torch.allclose(
            got, torch.full((w_elems,), exp_val), atol=1e-2, rtol=1e-2
        ):
            raise AssertionError(
                f"{tag}预取槽 {j}（专家 {e}，home {home}）期望 {exp_val}，"
                f"实得 {got[0].item():.1f}；本 rank etc={etc_cpu[rank].tolist()}；"
                f"全表 etc={etc_cpu.tolist()}"
            )
