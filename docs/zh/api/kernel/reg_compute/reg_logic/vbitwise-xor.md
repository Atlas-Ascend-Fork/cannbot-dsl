---
title: vbitwise_xor
api_name: vbitwise_xor
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vbitwise_xor`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

根据`mask`对源矢量数据寄存器`lhs`和`rhs`进行按位异或（^）操作，将结果作为返回值返回。未被`mask`筛选的位置被置为0。

本接口仅在AIV上生效。

## 函数原型

```python
def vbitwise_xor(lhs: RawVReg, rhs: RawVReg, *, mask: Mask) -> RawVReg: ...
```

**支持的数据类型：**

`dtype`支持的数据类型：`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.int32`、`dtypes.uint32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `lhs` | 输入 | 源矢量数据寄存器，参与按位异或的操作数。 |
| `rhs` | 输入 | 源矢量数据寄存器，参与按位异或的操作数。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。`mask`未筛选的元素在输出中置零。 |

## 返回值说明

- 返回保存按位异或结果的矢量数据寄存器，数据类型与`lhs`保持一致。

## 约束说明

- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- `mask`比特位为0时，计算结果对应比特位写0。
- 未被`mask`筛选的位置在返回值中置为0。

## 调用示例

将代码保存为`vbitwise_xor.py`后，可通过`python`命令运行。

```bash
python vbitwise_xor.py
```

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vbitwise_xor, vload, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vbitwise_xor_kernel(src0, src1, dst):
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
        xor_val = vbitwise_xor(vload(in0, 0), vload(in1, 0), mask=mask)
        vstore(res, 0, xor_val, mask)

    mem_copy(dst, out.consume())

@host
def run(src0, src1, dst):
    _vbitwise_xor_kernel[1](src0, src1, dst)

def main():
    src0 = torch.arange(64, dtype=torch.int32, device="npu:0") + 5
    src1 = torch.arange(64, dtype=torch.int32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 ^ src1).cpu())
    print("vbitwise_xor example passed")
    print(f"first={int(dst.cpu()[0])}, last={int(dst.cpu()[-1])}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vbitwise_xor example passed
first=5, last=123
```
