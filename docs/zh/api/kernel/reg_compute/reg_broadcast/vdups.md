---
title: vdups
api_name: vdups
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vdups`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

将标量`value`广播到返回值中。传入`mask`时，`value`写入被`mask`筛选的位置，`mode`取值为`'zeroing'`（默认值）时未被`mask`筛选的位置置0，取值为`'merging'`时这些位置保留`merge`中的值；未传入`mask`时，`value`写入返回值的全部元素位置。

本接口仅在AIV上生效。

## 函数原型

```python
def vdups(value, dtype, *, mask: 'Mask | None'=None, mode: str='zeroing', merge: 'RawVReg | None'=None) -> RawVReg: ...
```

**支持的数据类型：**

dtype支持的数据类型为`dtypes.int8`、`dtypes.uint8`、`dtypes.float8_e8m0`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `value` | 输入 | 源操作数（标量），作为待广播的数据。返回值的数据类型由`dtype`指定。 |
| `dtype` | 输入 | 返回值的数据类型，取值范围见函数原型中支持的数据类型。 |
| `mask` | 输入 | 目的操作数元素操作的有效指示（掩码寄存器）。`mask`筛选的元素在返回值中填充为`value`，未筛选的元素按`mode`的取值处理。 |
| `mode` | 输入 | 掩码模式，取值为`'zeroing'`或`'merging'`，默认值为`'zeroing'`。取值为`'zeroing'`时，返回值中未被`mask`筛选的位置置0；取值为`'merging'`时采用合并模式，返回值中未被`mask`筛选的元素保留`merge`的值，此时必须同时传入`mask`和`merge`。 |
| `merge` | 输入 | 合并模式下保留原值的矢量数据寄存器，仅在`mode`取值为`'merging'`时使用，数据类型须与`dtype`一致，默认值为`None`。返回值中未被`mask`筛选的元素保留该寄存器的值。 |

## 返回值说明

- 返回广播结果，返回类型为与`dtype`对应的矢量数据寄存器。

## 约束说明

- 本接口仅在AIV上生效。
- 本接口需在VF作用域内调用。
- 同一寄存器的数据依赖由硬件保序，无需额外插入同步指令。本接口与前后Reg数据搬运接口之间，如果不同寄存器访问同一Unified Buffer（UB）地址且存在写后读或写后写依赖，需要调用[vmem_bar](../reg_sync/vmem-bar.md)进行同步。
- 使用`mask`前，需要通过掩码设置或搬入接口完成初始化；未初始化的掩码寄存器内容不确定。

## 调用示例

将代码保存为`vdups.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, dtypes, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vdups, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vdups_kernel(dst):
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)
    res = out.produce()
    with vf(mode="simd"):
        acc = vdups(3.5, dtypes.float32)
        vstore(res, 0, acc, full_mask())

    mem_copy(dst, out.consume())

@host
def run(dst):
    _vdups_kernel[1](dst)

def main():
    dst = torch.empty((64,), dtype=torch.float32, device="npu:0")

    run(dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.full((64,), 3.5))
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
first=3.5000, last=3.5000
```
