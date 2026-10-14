---
title: varange
api_name: varange
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `varange`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

以传入的`start`为起始值，生成递增/递减的索引，将生成的索引作为返回值返回，`Vector Length (VL)`表示矢量数据寄存器的位宽，`VL_T`表示该寄存器可存储的元素数量。算法逻辑表示如下：

```text
// 递增
{start, start + 1, start + 2, ... start + VL_T - 2, start + VL_T - 1}
// 递减
{start + VL_T - 1, start + VL_T - 2, start + VL_T - 3, ... start + 1, start}
```

以`dtypes.int16`数据类型，起始值`start=10`为例：
递增索引为{10, 11, 12, 13, ... 135, 136, 137}，递减索引为{137, 136, 135, 134, ... 12, 11, 10}。

本接口仅在AIV上生效。

## 函数原型

```python
def varange(start, dtype, *, order_mode: 'str | OrderMode'=OrderMode.INCREASED) -> RawVReg: ...
```

**支持的数据类型：**

`dtypes.int8`、`dtypes.int16`、`dtypes.float16`、`dtypes.int32`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `start` | 输入 | 源操作数（标量），数据类型须与`dtype`一致。作为递增序列的起点时，序列第0个元素等于`start`，后续元素按1递增；作为递减序列的起点时，序列第0个元素等于`start + VL_T - 1`，后续元素按1递减。取值范围为该dtype的可表示范围。 |
| `dtype` | 输入 | 生成索引的元素类型。 |
| `order_mode` | 输入 | 生成递增或递减索引，取值为`cb.reg.OrderMode.INCREASED`（递增，默认值）或`cb.reg.OrderMode.DECREASED`（递减）。 |

## 返回值说明

- 返回生成的索引，返回值类型与`dtype`指定的数据类型一致。

## 约束说明

- 整型dtype（`dtypes.int8`、`dtypes.int16`、`dtypes.int32`）结果在超出该dtype可表示范围时回绕（wrap-around），不触发异常。例如`dtypes.int8`取`start=127`时，序列前128个元素依次为127、−128、−127、…、-2；`start=−128`时，序列前128个元素依次为−128、−127、…、−1。

## 调用示例

将代码保存为`varange.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, dtypes, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, varange, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _varange_kernel(dst):
    out = Channel(MemLoc.UB, shape=(64,), dtype=dst.dtype, depth=1)

    res = out.produce()
    with vf(mode="simd"):
        mask = full_mask()
        vstore(res, 0, varange(5, dtypes.int32), mask)

    mem_copy(dst, out.consume())

@host
def run(dst):
    _varange_kernel[1](dst)

def main():
    dst = torch.empty((64,), dtype=torch.int32, device="npu:0")

    run(dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.arange(5, 69, dtype=torch.int32))
    print("varange example passed")
    print(f"first={int(dst.cpu()[0])}, last={int(dst.cpu()[-1])}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
varange example passed
first=5, last=68
```
