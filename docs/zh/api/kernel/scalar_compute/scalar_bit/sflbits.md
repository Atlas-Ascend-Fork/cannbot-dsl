---
title: sflbits
api_name: sflbits
category: scalar_compute
api_group: kernel
layer: scalar
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `sflbits`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

对一个64位有符号整数，取其最高位（即符号位）作为参考位，统计从次高位（即最高数值位）开始与符号位相同的连续比特位的个数。

例：`0x0f00000000000000`。

符号位为0，从最高数值位开始往后（不包含符号位）与符号位相同的连续比特数为3，故返回3。

特例：当输入为全0或全1时，接口返回-1。

## 函数原型

```python
def sflbits(value) -> Int64: ...
```

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `value` | 输入 | 待统计的64位有符号整数，取值范围为[-2^63, 2^63-1]。 |

## 返回值说明

返回从最高数值位开始和符号位相同的连续比特位的个数，数据类型为`dtypes.int64`。

当输入是-1（比特位全1）或者0（比特位全0）时，返回-1；其余输入返回值的范围是[0, 62]。

## 约束说明

接口仅支持`dtypes.int64`，传入其他位宽的值会按`dtypes.int64`隐式转换后参与位运算，结果以`dtypes.int64`返回。

## 调用示例

将代码保存为`sflbits.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import sflbits, vec_store_bypass

@kernel
def _sflbits_kernel(dst):
    vec_store_bypass(dst.ptr(0), sflbits(0x0F00000000000000))

@host
def run(dst):
    _sflbits_kernel[1](dst)

def main():
    dst = torch.zeros((1,), dtype=torch.int64, device="npu:0")

    run(dst)
    torch.npu.synchronize()

    result = int(dst.cpu()[0])
    assert result == 3, result
    print("sflbits example passed")
    print(f"sflbits(0x0f00000000000000) = {result}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
sflbits example passed
sflbits(0x0f00000000000000) = 3
```
