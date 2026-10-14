---
title: get_block_num
api_name: get_block_num
category: system
api_group: kernel
layer: system
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `get_block_num`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

获取本次 Kernel 启动配置的任务块（block）数量。

## 函数原型

```python
def get_block_num() -> Int64: ...
```

## 参数说明

无。

## 返回值说明

返回本次启动 Kernel 时指定的 block 数量。
## 流水类型

`PIPE_S`

## 约束说明

无。

## 调用示例

将代码保存为`get_block_num.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.scalar import vec_store_bypass

@kernel
def _get_block_num_kernel(output):
    vec_store_bypass(output.ptr(get_block_idx()), get_block_num())

@host
def run(output):
    _get_block_num_kernel[4](output)

def main():
    output = torch.empty(4, dtype=torch.int64, device="npu:0")
    run(output)
    torch.npu.synchronize()

    assert output.cpu().tolist() == [4, 4, 4, 4]
    print("get_block_num example passed")
    print(f"block_num={output.cpu().tolist()[0]}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
get_block_num example passed
block_num=4
```
