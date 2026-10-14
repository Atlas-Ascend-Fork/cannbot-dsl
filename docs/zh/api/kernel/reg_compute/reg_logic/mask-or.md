---
title: mask_or
api_name: mask_or
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `mask_or`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

根据`exec_mask`对源操作数`lhs`、`rhs`执行按位或（|）操作，将结果作为返回值返回。

- 掩码寄存器按位或：对两个掩码寄存器执行按位或（|），结果为掩码寄存器。

计算公式如下：

$$
dst_i = lhs_i \,|\, rhs_i
$$

本接口仅在AIV上生效。

## 函数原型

```python
def mask_or(lhs: Mask, rhs: Mask, *, exec_mask: Mask) -> Mask: ...
```

**支持的数据类型：**

掩码寄存器不区分元素数据类型，仅由元素位宽`elem_bits`标识；参与计算的所有掩码寄存器必须具有相同的`elem_bits`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `lhs` | 输入 | 源操作数（掩码寄存器）。 |
| `rhs` | 输入 | 源操作数（掩码寄存器）。 |
| `exec_mask` | 输入 | 源操作数掩码（掩码寄存器），指示在计算过程中哪些bit有效。 |

## 返回值说明

- 通过函数返回值返回结果的函数原型返回按位或结果，返回类型与`lhs`、`rhs`的数据类型一致。

## 约束说明

- `exec_mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- 参与计算的元素个数由矢量长度（VL）决定：
    - 掩码寄存器按位或中比特个数 = VL。
- `exec_mask`比特位为0时，计算结果对应比特位写0。
- `lhs`、`rhs`与`exec_mask`的`elem_bits`必须一致，否则会报错。

## 调用示例

将代码保存为`mask_or.py`后，可通过`python`命令运行。

```bash
python mask_or.py
```

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, dtypes, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import create_mask, full_mask, mask_or, update_mask, vdups, vselect, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _mask_or_kernel(dst):
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    res = out.produce()
    with vf(mode="simd"):
        full = full_mask()
        m1, _ = update_mask(48, elem_bits=32)
        m = mask_or(m1, create_mask(pattern="vl16"), exec_mask=full)
        one = vdups(1.0, dtypes.float32)
        zero = vdups(0.0, dtypes.float32)
        vstore(res, 0, vselect(one, zero, cond_mask=m), full)

    mem_copy(dst, out.consume())

@host
def run(dst):
    _mask_or_kernel[1](dst)

def main():
    dst = torch.zeros(64, dtype=torch.float32, device="npu:0")

    run(dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.where(torch.arange(64) < 48, 1.0, 0.0))
    print("mask_or example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
mask_or example passed
first=1.0000, last=0.0000
```
