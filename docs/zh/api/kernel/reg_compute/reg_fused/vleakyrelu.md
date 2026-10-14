---
title: vleakyrelu
api_name: vleakyrelu
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vleakyrelu`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

根据掩码对源矢量数据寄存器中的元素执行Leaky ReLU操作，并将结果作为返回值返回。源操作数中大于0的元素直接写入输出结果，小于或等于0的元素乘以标量`slope`后写入输出结果。掩码对应位置为1的元素参与计算，为0的元素在输出结果中置零。计算公式如下：

$$
dst_i = \begin{cases} src_i & src_i > 0 \\ src_i \times \alpha & src_i \leq 0 \end{cases}
$$

本接口仅在AIV上生效。

## 函数原型

```python
def vleakyrelu(src: RawVReg, slope, *, mask: Mask) -> RawVReg: ...
```

**支持的数据类型：**

`dtypes.float16`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `slope` | 输入 | 负半轴斜率（标量）。数据类型须与`src`一致。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。`mask`未筛选的元素在输出中置零。需通过掩码设置接口预先赋值后再传入。 |

## 返回值说明

对于返回值类型接口，返回保存Leaky ReLU计算结果的矢量数据寄存器，数据类型与`src`保持一致。

## 约束说明

### 通用约束

- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- 掩码位为0的元素位置不参与运算，输出结果对应位置写0。

### 计算约束

- 源矢量数据寄存器中数据为负零（-0）时按负数处理，乘法满足IEEE 754浮点乘法规则。

## 调用示例

将代码保存为`vleakyrelu.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vleakyrelu, vload, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vleakyrelu_kernel(src, dst):
    buf = Channel(MemLoc.UB, (64,), src.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    mem_copy(buf.produce(), src)

    in0 = buf.consume()
    res = out.produce()
    with vf(mode="simd"):
        mask = full_mask()
        vstore(res, 0, vleakyrelu(vload(in0, 0), 0.1, mask=mask), mask)

    mem_copy(dst, out.consume())

@host
def run(src, dst):
    _vleakyrelu_kernel[1](src, dst)

def main():
    src = torch.arange(64, dtype=torch.float32, device="npu:0") - 32.0
    dst = torch.empty_like(src)

    run(src, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.nn.functional.leaky_relu(src, 0.1).cpu())
    print("vleakyrelu example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vleakyrelu example passed
first=-3.2000, last=31.0000
```
