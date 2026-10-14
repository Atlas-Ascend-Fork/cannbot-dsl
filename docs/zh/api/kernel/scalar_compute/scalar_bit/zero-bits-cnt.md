---
title: zero_bits_cnt
api_name: zero_bits_cnt
category: scalar_compute
api_group: kernel
layer: scalar
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `zero_bits_cnt`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

获取输入`dtypes.uint64`数据的二进制表示中值为0的位数。

## 函数原型

```python
def zero_bits_cnt(value) -> Int64: ...
```

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `value` | 输入 | 待统计的64位无符号整数，取值范围为[0, 2^64-1]。 |

## 返回值说明

`dtypes.int64`类型，表示`value`二进制表示中值为0的位数。取值范围为[0, 64]。

## 约束说明

本接口运行在标量流水（`PIPE_S`）上，同一流水内的数据依赖由指令执行顺序保证，无需额外同步。

## 调用示例

将代码保存为`zero_bits_cnt.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import vec_store_bypass, zero_bits_cnt

@kernel
def _zero_bits_cnt_kernel(dst):
    vec_store_bypass(dst.ptr(0), zero_bits_cnt(15))

@host
def run(dst):
    _zero_bits_cnt_kernel[1](dst)

def main():
    dst = torch.zeros((1,), dtype=torch.int64, device="npu:0")

    run(dst)
    torch.npu.synchronize()

    result = int(dst.cpu()[0])
    assert result == 60, result
    print("zero_bits_cnt example passed")
    print(f"zero_bits_cnt(15) = {result}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
zero_bits_cnt example passed
zero_bits_cnt(15) = 60
```
