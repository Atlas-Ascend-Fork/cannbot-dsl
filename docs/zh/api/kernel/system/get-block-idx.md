---
title: get_block_idx
api_name: get_block_idx
category: system
api_group: kernel
layer: system
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `get_block_idx`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

获取当前任务块（block）在本次 Kernel 启动任务中的索引。常用于多核任务的数据切分和输出定位。

在 Mix 1:2 模式下，两个 Vector Core 的 block 索引相同；如需区分，可使用 `get_block_idx() * get_subblock_dim() + get_subblock_id()` 获取 Vector Core 的逻辑索引。

## 函数原型

```python
def get_block_idx() -> Int64: ...
```

## 参数说明

无。

## 返回值说明

返回 `[0, get_block_num())` 范围内的 block 索引。
## 流水类型

`PIPE_S`

## 约束说明

无。

## 调用示例

将代码保存为`get_block_idx.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.scalar import vec_store_bypass

@kernel
def _get_block_idx_kernel(output):
    index = get_block_idx()
    vec_store_bypass(output.ptr(index), index)

@host
def run(output):
    _get_block_idx_kernel[4](output)

def main():
    output = torch.full((4,), -1, dtype=torch.int64, device="npu:0")
    run(output)
    torch.npu.synchronize()

    assert output.cpu().tolist() == [0, 1, 2, 3]
    print("get_block_idx example passed")
    print(f"block_indices={output.cpu().tolist()}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
get_block_idx example passed
block_indices=[0, 1, 2, 3]
```
