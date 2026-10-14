---
title: mask_xor
api_name: mask_xor
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `mask_xor`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

根据`exec_mask`对两个源掩码寄存器`lhs`和`rhs`进行按位异或（^）操作，将结果作为返回值返回。未被`exec_mask`筛选的位置被置为0。

本接口仅在AIV上生效。

## 函数原型

```python
def mask_xor(lhs: Mask, rhs: Mask, *, exec_mask: Mask) -> Mask: ...
```

**支持的数据类型：**

掩码寄存器不区分元素数据类型，仅由元素位宽`elem_bits`标识；参与计算的所有掩码寄存器必须具有相同的`elem_bits`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `lhs` | 输入 | 源掩码寄存器，参与按位异或的操作数。 |
| `rhs` | 输入 | 源掩码寄存器，参与按位异或的操作数。 |
| `exec_mask` | 输入 | 源掩码寄存器，用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。`exec_mask`未筛选的元素在输出中置零。 |

## 返回值说明

- 返回保存按位异或结果的掩码寄存器，数据类型与`lhs`保持一致。

## 约束说明

- `exec_mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- `exec_mask`比特位为0时，计算结果对应比特位写0。
- `lhs`、`rhs`与`exec_mask`的`elem_bits`必须一致，否则会报错。
- 未被`exec_mask`筛选的位置在返回值中置为0。

## 调用示例

将代码保存为`mask_xor.py`后，可通过`python`命令运行。

```bash
python mask_xor.py
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
from cannbotdsl.ops.reg import create_mask, full_mask, mask_xor, vdups, vselect, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _mask_xor_kernel(dst):
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    res = out.produce()
    with vf(mode="simd"):
        full = full_mask()
        m = mask_xor(create_mask(pattern="m3"), create_mask(pattern="h"),
                            exec_mask=full)
        one = vdups(1.0, dtypes.float32)
        zero = vdups(0.0, dtypes.float32)
        vstore(res, 0, vselect(one, zero, cond_mask=m), full)

    mem_copy(dst, out.consume())

@host
def run(dst):
    _mask_xor_kernel[1](dst)

def main():
    dst = torch.zeros(64, dtype=torch.float32, device="npu:0")
    idx = torch.arange(64)

    run(dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.where((idx % 3 == 0) ^ (idx < 32), 1.0, 0.0))
    print("mask_xor example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
mask_xor example passed
first=0.0000, last=1.0000
```
