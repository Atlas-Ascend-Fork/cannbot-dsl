---
title: cast
api_name: cast
category: scalar_compute
api_group: kernel
layer: scalar
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `cast`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

将`dtypes.float32`类型的标量转换为`dtypes.int32`类型并返回。

关于舍入模式和饱和/非饱和模式的详细说明，请参见[舍入模式与饱和模式](./roundingmode.md)。

## 函数原型

```python
def cast(value, *, dtype, rounding: RoundingMode | str=RoundingMode.RN) -> Int32: ...
```

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `value` | 输入 | 源操作数（标量），数据类型为`dtypes.float32`。 |
| `dtype` | 输入 | 转换后的数据类型，仅支持`dtypes.int32`。 |
| `rounding` | 输入 | 舍入模式，取值为`RoundingMode.RN`、`RoundingMode.RNA`、`RoundingMode.RD`或`RoundingMode.RU`，默认值为`RoundingMode.RN`。取值含义参见[舍入模式与饱和模式](./roundingmode.md)。 |

## 返回值说明

返回`value`精度转换成`dtypes.int32`的结果。

## 约束说明

- 输入值超出`dtypes.int32`的表示范围[−2^31, 2^31-1]时，饱和模式下，结果钳位到`dtypes.int32`的最大值或最小值；非饱和模式下，结果截断为低32位。
- 输入为nan或inf时返回0。
- 当前仅支持`dtypes.float32`转换为`dtypes.int32`；`dtype`传入其他数据类型时抛出`TypeError`。
- 本接口运行在标量流水（`PIPE_S`）上，同一流水内的数据依赖由指令执行顺序保证，无需额外同步。

## 调用示例

将代码保存为`cast.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import dtypes, host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import cast, vec_store_bypass

@kernel
def _cast_kernel(dst):
    vec_store_bypass(dst.ptr(0), cast(2.7, dtype=dtypes.int32))

@host
def run(dst):
    _cast_kernel[1](dst)

def main():
    dst = torch.zeros((1,), dtype=torch.int32, device="npu:0")

    run(dst)
    torch.npu.synchronize()

    result = int(dst.cpu()[0])
    assert result == 3, result
    print("cast example passed")
    print(f"cast(2.7) = {result}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
cast example passed
cast(2.7) = 3
```
