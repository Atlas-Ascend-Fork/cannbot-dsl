---
title: clz
api_name: clz
category: scalar_compute
api_group: kernel
layer: scalar
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `clz`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

计算一个`dtypes.uint64`类型整数在二进制表示下的前导零个数，即从二进制最高位开始，到第一个出现二进制1为止，中间连续的0的数量。

## 函数原型

```python
def clz(value) -> Int64: ...
```

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `value` | 输入 | 待统计的64位无符号整数。取值范围为[0, 2^64-1]。 |

## 返回值说明

返回`value`的前导零个数，取值范围为[0, 64]。当`value`为0时，返回64。

## 约束说明

本接口运行在标量流水（`PIPE_S`）上，同一流水内的数据依赖由指令执行顺序保证，无需额外同步。

## 调用示例

将代码保存为`clz.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import clz, vec_store_bypass

@kernel
def _clz_kernel(dst):
    vec_store_bypass(dst.ptr(0), clz(1))

@host
def run(dst):
    _clz_kernel[1](dst)

def main():
    dst = torch.zeros((1,), dtype=torch.int64, device="npu:0")

    run(dst)
    torch.npu.synchronize()

    result = int(dst.cpu()[0])
    assert result == 63, result
    print("clz example passed")
    print(f"clz(1) = {result}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
clz example passed
clz(1) = 63
```
