# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys

import pytest

pytest.importorskip("cannbotdsl")

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "samples", "mc2", "moonep"
    ),
)

# 无 NPU 环境收集期兼容：守卫导入 + pytestmark skipif（importorskip
# 对 torch_npu 加载失败的嵌套 ImportError 不总是生效）
try:
    from moonep_context import (  # noqa: E402
        _select_rank_graph_links_by_protocol,
    )

    _HAS_NPU_ENV = True
except ImportError:
    _HAS_NPU_ENV = False

pytestmark = pytest.mark.skipif(
    not _HAS_NPU_ENV, reason="requires NPU environment (torch_npu/HCCL)"
)


class _Link:
    def __init__(self, protocol):
        self.linkAttr = type("LinkAttr", (), {"linkProtocol": protocol})()


def _fake_get_links(links_by_layer):
    def get_links(comm, layer_id, src_rank, dst_rank):
        key = (layer_id, src_rank, dst_rank)
        if key not in links_by_layer:
            raise RuntimeError(f"unexpected link query: {key}")
        return links_by_layer[key], len(links_by_layer[key])

    return get_links


def test_first_layer_request_selects_first_layer_with_target_protocol():
    links = {
        (0, 8, 0): [],
        (1, 8, 0): [_Link(4), _Link(6)],
        (2, 8, 0): [_Link(9)],
        (0, 8, 9): [_Link(4), _Link(6)],
        (1, 8, 9): [_Link(4), _Link(6)],
        (2, 8, 9): [_Link(9)],
    }
    get_links = _fake_get_links(links)
    layers = [0, 1, 2]

    merged, count = _select_rank_graph_links_by_protocol(
        object(), layers[0], 8, 0, layers, get_links, 4
    )
    assert [x.linkAttr.linkProtocol for x in merged] == [4, 6]
    assert count == 2


def test_same_supernode_peer_keeps_first_layer_links():
    links = {
        (0, 8, 9): [_Link(4), _Link(6)],
        (1, 8, 9): [_Link(4), _Link(6)],
        (2, 8, 9): [_Link(9)],
    }
    get_links = _fake_get_links(links)
    layers = [0, 1, 2]

    merged, count = _select_rank_graph_links_by_protocol(
        object(), layers[0], 8, 9, layers, get_links, 4
    )
    assert [x.linkAttr.linkProtocol for x in merged] == [4, 6]
    assert count == 2


def test_first_layer_with_only_non_target_protocol_is_skipped():
    links = {
        (0, 8, 0): [_Link(6)],
        (1, 8, 0): [_Link(4), _Link(6)],
        (2, 8, 0): [_Link(9)],
    }
    get_links = _fake_get_links(links)
    layers = [0, 1, 2]

    merged, count = _select_rank_graph_links_by_protocol(
        object(), layers[0], 8, 0, layers, get_links, 4
    )
    assert [x.linkAttr.linkProtocol for x in merged] == [4, 6]
    assert count == 2


def test_target_protocol_can_be_found_in_fifth_layer():
    layers = [0, 1, 2, 3, 4]
    links = {
        (layer, 8, 0): [] if layer < 4 else [_Link(4), _Link(6)] for layer in layers
    }
    get_links = _fake_get_links(links)

    merged, count = _select_rank_graph_links_by_protocol(
        object(), layers[0], 8, 0, layers, get_links, 4
    )
    assert [x.linkAttr.linkProtocol for x in merged] == [4, 6]
    assert count == 2


def test_explicit_non_first_layer_keeps_original_semantics():
    links = {(2, 8, 0): [_Link(9)]}
    get_links = _fake_get_links(links)
    layers = [0, 1, 2]

    merged, count = _select_rank_graph_links_by_protocol(
        object(), 2, 8, 0, layers, get_links, 4
    )
    assert [x.linkAttr.linkProtocol for x in merged] == [9]
    assert count == 1
