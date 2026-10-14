---
title: RoundingMode
api_name: RoundingMode
category: scalar_compute
api_group: kernel
layer: scalar
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `RoundingMode`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

`RoundingMode` 是标量转换的舍入模式枚举，用于 `scalar.cast` 的 `rounding` 参数。成员为 `RN`（RINT，四舍六入五成双舍入）、`RNA`（ROUND，四舍五入舍入）、`RD`（FLOOR，向负无穷方向舍入）、`RU`（CEIL，向正无穷方向舍入）、`RZ`（TRUNC，向零方向舍入）、`RO`（ODD，最近邻奇数舍入）、`RH`（HYBRID，混合舍入模式）与 `NA`（由实现按约定选择）。

寄存器级转换使用 `reg.RoundingMode`，两者是相互独立的枚举。

## 函数原型

```python
# 枚举成员
RoundingMode.RN  # 四舍六入五成双舍入（RINT）
RoundingMode.RNA  # 四舍五入舍入（ROUND）
RoundingMode.RD  # 向负无穷方向舍入（FLOOR）
RoundingMode.RU  # 向正无穷方向舍入（CEIL）
RoundingMode.RZ  # 向零方向舍入（TRUNC）
RoundingMode.RO  # 最近邻奇数舍入（ODD）
RoundingMode.RH  # 混合舍入模式（HYBRID）
RoundingMode.NA  # 由实现按约定选择
```

## 参数说明

无参数。

## 返回值说明

不适用（枚举说明页）。

## 约束说明

- `scalar.cast` 的 `rounding` 默认值为 `RoundingMode.RN`。
- `rounding` 也可传入匹配的小写字符串（`"rn"`、`"rna"`、`"rd"`、`"ru"`）。
- 本枚举仅用于标量转换；寄存器级转换使用 `reg.RoundingMode`。

## 调用示例

以下示例以显式 `RoundingMode.RD` 调用 `scalar.cast`，并在Host侧校验舍入结果。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import dtypes, host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import RoundingMode, cast, vec_store_bypass

@kernel
def _rounding_mode_kernel(dst):
    vec_store_bypass(
        dst.ptr(0),
        cast(2.7, dtype=dtypes.int32, rounding=RoundingMode.RD),
    )

@host
def run(dst):
    _rounding_mode_kernel[1](dst)

def main():
    dst = torch.zeros((1,), dtype=torch.int32, device="npu:0")

    run(dst)
    torch.npu.synchronize()

    result = int(dst.cpu()[0])
    assert result == 2, result
    print("RoundingMode example passed")
    print(f"cast(2.7, RD) = {result}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
RoundingMode example passed
cast(2.7, RD) = 2
```
