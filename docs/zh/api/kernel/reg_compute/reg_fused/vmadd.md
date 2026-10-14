---
title: vmadd
api_name: vmadd
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vmadd`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

根据`mask`对源操作数`lhs`、`rhs`按元素相乘后，与加数操作数`addend`中的对应元素相加，将结果作为返回值返回。计算公式如下：

$$
lhs_i = lhs_i \times rhs_i + addend_i
$$

本接口仅在AIV上生效。

## 函数原型

```python
def vmadd(lhs: RawVReg, rhs: RawVReg, addend: RawVReg, *, mask: Mask) -> RawVReg: ...
```

**支持的数据类型：**

`dtypes.float16`、`dtypes.bfloat16`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `lhs` | 输入/输出 | 源操作数0（矢量数据寄存器）。计算结果会写入该寄存器。 |
| `rhs` | 输入 | 源操作数1（矢量数据寄存器）。 |
| `addend` | 输入 | 加数操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 掩码寄存器，用于控制各元素是否参与计算。`mask`中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。 |

## 返回值说明

- 返回计算结果，类型为矢量数据寄存器，与`lhs`、`rhs`、`addend`的数据类型一致。

## 约束说明

- `lhs`、`rhs`和`addend`的数据类型需要保持一致。
- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- `mask`掩码位为0时，结果对应元素置0。

## 调用示例

将代码保存为`vmadd.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload, vmadd, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vmadd_kernel(src0, src1, src2, dst):
    buf0 = Channel(MemLoc.UB, (64,), src0.dtype, depth=1)
    buf1 = Channel(MemLoc.UB, (64,), src1.dtype, depth=1)
    buf2 = Channel(MemLoc.UB, (64,), src2.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    mem_copy(buf0.produce(), src0)
    mem_copy(buf1.produce(), src1)
    mem_copy(buf2.produce(), src2)

    in0 = buf0.consume()
    in1 = buf1.consume()
    in2 = buf2.consume()
    res = out.produce()
    with vf(mode="simd"):
        mask = full_mask()
        r = vmadd(vload(in0, 0), vload(in1, 0), vload(in2, 0), mask=mask)
        vstore(res, 0, r, mask)

    mem_copy(dst, out.consume())

@host
def run(src0, src1, src2, dst):
    _vmadd_kernel[1](src0, src1, src2, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0") / 8.0
    src1 = torch.arange(64, dtype=torch.float32, device="npu:0") / 16.0
    src2 = torch.arange(64, dtype=torch.float32, device="npu:0") / 32.0
    dst = torch.empty_like(src0)

    run(src0, src1, src2, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 * src1 + src2).cpu())
    print("vmadd example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vmadd example passed
first=0.0000, last=32.9766
```
