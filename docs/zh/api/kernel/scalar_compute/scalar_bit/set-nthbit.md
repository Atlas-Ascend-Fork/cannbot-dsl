---
title: set_nthbit
api_name: set_nthbit
category: scalar_compute
api_group: kernel
layer: scalar
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `set_nthbit`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

将`dtypes.uint64`数据的指定二进制位置为1，其余位保持不变。返回修改后的`dtypes.uint64`整数。

## 函数原型

```python
def set_nthbit(bits, idx) -> UInt64: ...
```

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `bits` | 输入 | 待置位的64位无符号整数，取值范围为[0, 2^64-1]。执行此计算后输入数据不变。 |
| `idx` | 输入 | 位索引，指定需要设置为1的二进制位的位置。实际位索引按`(idx % 64 + 64) % 64`计算，索引范围为[0, 63]。 |

## 返回值说明

返回执行计算得到的结果，数据类型为`dtypes.uint64`。

## 约束说明

当参数`idx > 63`或者`idx < 0`时，位索引的计算逻辑为`(idx % 64 + 64) % 64`。

## 调用示例

将代码保存为`set_nthbit.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import set_nthbit, vec_store_bypass

@kernel
def _set_nthbit_kernel(dst):
    vec_store_bypass(dst.ptr(0), set_nthbit(8, 1))

@host
def run(dst):
    _set_nthbit_kernel[1](dst)

def main():
    dst = torch.zeros((1,), dtype=torch.int64, device="npu:0").view(torch.uint64)

    run(dst)
    torch.npu.synchronize()

    result = int(dst.cpu().view(torch.int64)[0])
    assert result == 10, result
    print("set_nthbit example passed")
    print(f"set_nthbit(8, 1) = {result}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
set_nthbit example passed
set_nthbit(8, 1) = 10
```
