---
title: clear_nthbit
api_name: clear_nthbit
category: scalar_compute
api_group: kernel
layer: scalar
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `clear_nthbit`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

对一个64位无符号整数执行位清除操作，将`bits`中第`idx`位清零，其余位保持不变。返回修改后的64位无符号整数。

本接口为标量位运算，运行在标量流水上，AIC与AIV均可调用，行为一致。

## 函数原型

```python
def clear_nthbit(bits, idx) -> UInt64: ...
```

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `bits` | 输入 | 64位无符号整数。取值范围为[0，2^64−1]。 |
| `idx` | 输入 | 位索引。实际位索引按`(idx % 64 + 64) % 64`计算，范围为[0，63]。 |

## 返回值说明

返回位清零后的结果（`dtypes.uint64`类型）。
## 流水类型

`PIPE_S`

## 约束说明

当`idx`大于63或小于0时，实际清除的位索引按`(idx % 64 + 64) % 64`计算。

## 调用示例

将代码保存为`clear_nthbit.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import clear_nthbit, vec_store_bypass

@kernel
def _clear_nthbit_kernel(dst):
    vec_store_bypass(dst.ptr(0), clear_nthbit(15, 1))

@host
def run(dst):
    _clear_nthbit_kernel[1](dst)

def main():
    dst = torch.zeros((1,), dtype=torch.int64, device="npu:0").view(torch.uint64)

    run(dst)
    torch.npu.synchronize()

    result = int(dst.cpu().view(torch.int64)[0])
    assert result == 13, result
    print("clear_nthbit example passed")
    print(f"clear_nthbit(15, 1) = {result}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
clear_nthbit example passed
clear_nthbit(15, 1) = 13
```
