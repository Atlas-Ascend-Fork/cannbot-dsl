# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch

from _samples_path import load_sample

_sample = load_sample("attention/stem_indexer/stem_indexer_metadata.py")
AIC_CORE_NUM = _sample.AIC_CORE_NUM
AIV_CORE_NUM = _sample.AIV_CORE_NUM
HEAD_METADATA_STRIDE = _sample.HEAD_METADATA_STRIDE
FA_METADATA_STRIDE = _sample.FA_METADATA_STRIDE
stem_indexer_metadata = _sample.stem_indexer_metadata


@pytest.mark.npu
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "q_lengths,kv_lengths,num_heads_q,num_heads_kv",
    [
        ([65536], [65536], 32, 8),
        ([65536], [262144], 32, 4),
        ([0], [0], 32, 8),
        ([129], [0], 32, 4),
        ([1], [65536], 64, 2),
        ([40 * 128], [512 * 128], 32, 8),
        ([0, 1, 129, 8193], [1024, 65536, 513, 131071], 64, 4),
        ([65536] * 4, [262144] * 4, 32, 4),
        ([128, 256, 512], [128, 256, 512], 32, 2),
    ],
)
def test_stem_indexer_metadata(
    q_lengths, kv_lengths, num_heads_q, num_heads_kv, causal
):
    pytest.importorskip("torch_npu")
    q = torch.tensor(q_lengths, dtype=torch.int32, device="npu")
    k = torch.tensor(kv_lengths, dtype=torch.int32, device="npu")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        cube, _ = _sample.get_effective_core_counts(stream)
        metadata = stem_indexer_metadata(q, k, num_heads_q, num_heads_kv, causal=causal)
    stream.synchronize()
    assert metadata.dtype == torch.int32
    batch = len(q_lengths)
    elements = (
        1 + batch * num_heads_kv * (AIC_CORE_NUM + AIV_CORE_NUM)
    ) * HEAD_METADATA_STRIDE
    assert metadata.numel() == (elements + 4095) // 4096 * 4096
    data = metadata.cpu()
    sections = int(data[0])
    assert 0 < sections <= batch * num_heads_kv
    # SI只定义section数量，不使用FA的FD标志及动态分块header字段。
    assert not torch.count_nonzero(data[1:HEAD_METADATA_STRIDE])
    end = HEAD_METADATA_STRIDE + sections * AIC_CORE_NUM * FA_METADATA_STRIDE
    slots = data[HEAD_METADATA_STRIDE:end].reshape(
        sections, AIC_CORE_NUM, FA_METADATA_STRIDE
    )
    previous = (0, 0)
    for section in slots:
        assert not torch.count_nonzero(section[cube:])
        for slot in section[:cube]:
            if not torch.count_nonzero(slot):
                continue
            start, finish = (int(slot[0]), int(slot[1])), (int(slot[3]), int(slot[4]))
            assert start == previous
            assert start <= finish <= (batch * num_heads_kv, 0)
            # SI不拆S2，其起止字段及其余保留字段应为零。
            assert int(slot[2]) == 0 and not torch.count_nonzero(slot[5:])
            previous = finish
    assert previous == (batch * num_heads_kv, 0)
    assert not torch.count_nonzero(data[end:])


@pytest.mark.npu
@pytest.mark.parametrize("causal", [False, True])
def test_equivalent_block_lengths(causal):
    pytest.importorskip("torch_npu")
    # metadata按128-token块分核，落在同一块内的长度应产生相同结果。
    q = torch.tensor([1, 129, 513], dtype=torch.int32, device="npu")
    k = torch.tensor([257, 1025, 8193], dtype=torch.int32, device="npu")
    actual = stem_indexer_metadata(q, k, 32, 4, causal=causal)
    expected = stem_indexer_metadata(
        (q + 127) // 128 * 128, (k + 127) // 128 * 128, 32, 4, causal=causal
    )
    torch.npu.synchronize()
    assert torch.equal(actual.cpu(), expected.cpu())


@pytest.mark.npu
def test_split_kv_sections():
    pytest.importorskip("torch_npu")
    # 单个BN的数据量超过L2预算，触发多section；仅构造长度，不分配Q/K。
    lengths = torch.tensor([1048576] * 4, dtype=torch.int32, device="npu")
    metadata = stem_indexer_metadata(lengths, lengths, 32, 4)
    assert int(metadata[0].cpu()) > 1


@pytest.mark.npu
def test_explicit_block_dim():
    pytest.importorskip("torch_npu")
    cube, _ = _sample.get_effective_core_counts(torch.npu.current_stream())
    q = torch.tensor([65536], dtype=torch.int32, device="npu")
    actual = stem_indexer_metadata(q, q, 64, 8)
    expected = stem_indexer_metadata(q, q, 64, 8, block_dim=cube)
    torch.npu.synchronize()
    assert torch.equal(actual.cpu(), expected.cpu())
    with pytest.raises(ValueError):
        stem_indexer_metadata(q, q, 64, 8, block_dim=cube + 1)


@pytest.mark.npu
def test_input_validation():
    pytest.importorskip("torch_npu")
    q = torch.tensor([128], dtype=torch.int32, device="npu")
    with pytest.raises(ValueError):
        stem_indexer_metadata(q, q, 31, 8)
    with pytest.raises(ValueError):
        stem_indexer_metadata(q.to(torch.int64), q, 32, 8)
