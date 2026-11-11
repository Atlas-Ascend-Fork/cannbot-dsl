# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MoonEP 网络拓扑选择：多层 HCCL rank graph 的 per-peer 链路解析补丁。

本补丁 hook 的是 ``cannbotdsl.ops.distributed._hccl.get_links``——该函数
由 DSL 的 host 侧建链路径（``_build_hccl_buffer_ctx`` → SymBuffer/传输
上下文创建）在**算子首调/编译期**主动调用；补丁在首层 links 请求时按
HCCL 层序为该 (src, dst) 选首个含 URMA 协议的层。即：拓扑选择完全由
host 侧的建链时机驱动，设备侧 kernel 不感知层选择。永久修复属上游
cann/cannbot-dsl#106（DSL 原生支持层序选择后本补丁下线）。PR #366
同步版（32 卡验证）。moonep_context 导入本模块即打补丁（幂等）。
"""

from cannbotdsl.ops.distributed.context import hccl


def _patch_rank_graph_layer_selection():
    """为多层 HCCL rank graph 建 per-peer 链路解析（32 卡已验证，PR #366 同步）。

    HCCL rank graph 在 16-rank 单机返回三层 [0, 1, 2]：层 0 只含同
    supernode（0-7 / 8-15）内的链路，跨组对（如 8->0）为空；DSL 的
    `_build_hccl_buffer_ctx` 盲取 `layers[0]` → acquire_channel 报
    "No link with protocol 4 found"。实测层 1 对组内/跨组对均含
    protocol 4（URMA）与 6 链路。

    950 组网中 net layer 表示一个“可连通域”，Full Mesh / Clos / nD-Mesh
    可能分别出现在不同层，且不保证存在一个对本 rank 全 peer 都有链路
    的层。把某个全覆盖层挪到首位仍然只能覆盖这类特殊拓扑，不能覆盖
    异构/多级组网下的 per-peer 差异。

    补丁改为：当 DSL 请求首层 links 时，按 HCCL 返回的层序为该
    (src, dst) 选择第一个包含 URMA 协议的层，并立即返回该层的 links；
    非首层请求保持原语义。`create_moonep_context()` 使用默认
    `CommEngine()`，其 `protocol` 为 `Protocol.URMA`，对应
    `hccl.COMM_PROTOCOL_UBC_CTP`。`acquire_channel` 只按目标协议在
    links 中筛选 endpoint，且 `layer_id` 本身不进入 C API，因此返回
    首个包含目标协议的层即可让它选中第一层可达且协议匹配的链路。
    8 卡（单层全互联）与原行为等价；16 卡跨 supernode 自动命中层 1；
    无全局全覆盖层的任意组网也能按 peer 建链。

    上游 issue：cann/cannbot-dsl#106
    """
    from cannbotdsl.ops.distributed import _hccl

    if getattr(_hccl.get_links, "_moonep_layer_patch", False):
        return
    _orig_layers = _hccl.get_rank_graph_layers
    _orig_links = _hccl.get_links
    _layers_cache = {}

    def cached_layers(comm):
        if comm not in _layers_cache:
            _layers_cache[comm] = _orig_layers(comm)
        return _layers_cache[comm]

    def links_with_layer_fallback(comm, layer_id, src_rank, dst_rank):
        layers = cached_layers(comm)
        return _select_rank_graph_links_by_protocol(
            comm,
            layer_id,
            src_rank,
            dst_rank,
            layers,
            _orig_links,
            hccl.COMM_PROTOCOL_UBC_CTP,
        )

    links_with_layer_fallback._moonep_layer_patch = True
    _hccl.get_links = links_with_layer_fallback


def _select_rank_graph_links_by_protocol(
    comm,
    layer_id,
    src_rank,
    dst_rank,
    layers,
    get_links,
    target_protocol,
):
    """按 HCCL 层序选择首个包含目标协议的层；仅对首层请求启用多层回退。"""
    if src_rank == dst_rank or len(layers) <= 1 or layer_id != layers[0]:
        return get_links(comm, layer_id, src_rank, dst_rank)

    for candidate_layer in layers:
        try:
            links, _ = get_links(comm, candidate_layer, src_rank, dst_rank)
        except Exception:
            continue
        if any(link.linkAttr.linkProtocol == target_protocol for link in links):
            return links, len(links)
    return [], 0


_patch_rank_graph_layer_selection()  # 模块加载即打补丁（幂等守卫）
