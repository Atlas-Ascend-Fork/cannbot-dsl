# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Query epilogue geometry must remain valid independently of core quota."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _samples_path import load_sample

query = load_sample("attention/mega_reindex/reindex_query.py")
mx = load_sample("attention/mega_reindex/reindex_mx.py")


@pytest.mark.parametrize("cores", [2, 8, 16, 28, 32, 64])
def test_fixed_query_tile_accepts_core_profiles(monkeypatch, cores):
    # Reproduce the old 28-core selector mismatch without a physical NPU.
    monkeypatch.setattr(mx, "AIC_NUM", cores)
    tile = query.get_query_tile_config(cores)
    phase = query.TailQueryPhase(24, 4096, 1280, tile, cores)
    assert (phase.base_m, phase.base_n, phase.base_k) == (16, 128, 256)
    assert phase.core_count == cores
    assert (phase.block_scheduler.m_tiles, phase.block_scheduler.n_tiles) == (2, 32)
    assert phase.k_l0_tiles == 5
    assert tile.full_load and tile.k_l1 == tile.scale_k_l1 == 1280
    assert tile.used_core_num == min(cores, 32)
    assert (tile.m_base_tail_split_count, tile.n_base_tail_split_count) == (1, 1)


@pytest.mark.parametrize("cores", [0, -1])
def test_invalid_query_quota(cores):
    with pytest.raises(ValueError):
        query.get_query_tile_config(cores)
