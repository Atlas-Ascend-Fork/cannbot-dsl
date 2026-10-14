---
title: vadd
api_name: vadd
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vadd`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

该接口根据`mask`，对源操作数`lhs`、`rhs`进行按元素求和操作，将结果作为返回值返回。

**无进位输出加法**：单目的操作数，矢量数据寄存器写入加法和，计算公式如下：

$$
dst_i = lhs_i + rhs_i
$$

本接口仅在AIV上生效。

## 函数原型

```python
def vadd(lhs: RawVReg, rhs: RawVReg, *, mask: Mask) -> RawVReg: ...
```

**支持的数据类型：**

`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 参数说明

**表 1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `lhs` | 输入 | 源操作数（矢量数据寄存器）。 |
| `rhs` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

## 返回值说明

- 返回无进位加法结果，返回值类型与源操作数类型一致。

## 约束说明

### 通用约束

- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。

### 无进位输出加法约束

- 整型dtype加法溢出时按环绕（wrap-around）策略处理：有符号类型溢出回绕到对应dtype的最小值或最大值（MAX+1->MIN，MIN-1->MAX），无符号类型溢出回绕到0（UMAX+1->0）。
- 浮点dtype（`dtypes.float16`、`dtypes.bfloat16`、`dtypes.float32`）加法按IEEE 754浮点加法语义执行，溢出与无效输入（如nan、inf）的处理遵循浮点运算规则。

## 调用示例

将代码保存为`vadd.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vadd, vload, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vadd_kernel(src0, src1, dst):
    buf0 = Channel(MemLoc.UB, (64,), src0.dtype, depth=1)
    buf1 = Channel(MemLoc.UB, (64,), src1.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    mem_copy(buf0.produce(), src0)
    mem_copy(buf1.produce(), src1)

    in0 = buf0.consume()
    in1 = buf1.consume()
    res = out.produce()
    with vf(mode="simd"):
        mask = full_mask()
        acc = vadd(vload(in0, 0), vload(in1, 0), mask=mask)
        vstore(res, 0, acc, mask)

    mem_copy(dst, out.consume())

@host
def run(src0, src1, dst):
    _vadd_kernel[1](src0, src1, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    src1 = torch.arange(64, dtype=torch.float32, device="npu:0") + 1.0
    dst = torch.empty_like(src0)

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 + src1).cpu())
    print("vadd example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vadd example passed
first=1.0000, last=127.0000
```
