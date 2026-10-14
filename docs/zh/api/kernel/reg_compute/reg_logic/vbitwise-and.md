---
title: vbitwise_and
api_name: vbitwise_and
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vbitwise_and`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

根据`mask`对源操作数`lhs`、`rhs`执行按位与（&）操作，将结果作为返回值返回。

- 矢量数据寄存器按位与：对两个矢量数据寄存器按`dtype`位宽执行按位与（&），结果为矢量数据寄存器。

计算公式如下：

$$
dst_i = lhs_i \& rhs_i
$$

本接口仅在AIV上生效。

## 函数原型

```python
def vbitwise_and(lhs: RawVReg, rhs: RawVReg, *, mask: Mask) -> RawVReg: ...
```

**支持的数据类型：**

`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.int32`、`dtypes.uint32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `lhs` | 输入 | 源操作数（矢量数据寄存器）。 |
| `rhs` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），对应位置为1时参与计算，为0时不参与计算。`mask`未筛选的元素在输出中置零。 |

## 返回值说明

- 通过函数返回值返回结果的函数原型返回按位与结果，返回类型与`lhs`、`rhs`的数据类型一致。

## 约束说明

- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- 参与计算的元素个数由矢量长度（VL）决定：矢量数据寄存器按位与中元素个数 = VL ÷ sizeof(dtype)。
- `mask`比特位为0时，计算结果对应比特位写0。

## 调用示例

将代码保存为`vbitwise_and.py`后，可通过`python`命令运行。

```bash
python vbitwise_and.py
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
from cannbotdsl.ops.reg import full_mask, vbitwise_and, vload, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vbitwise_and_kernel(src0, src1, dst):
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
        and_val = vbitwise_and(vload(in0, 0), vload(in1, 0), mask=mask)
        vstore(res, 0, and_val, mask)

    mem_copy(dst, out.consume())

@host
def run(src0, src1, dst):
    _vbitwise_and_kernel[1](src0, src1, dst)

def main():
    src0 = torch.arange(64, dtype=torch.int32, device="npu:0") + 5
    src1 = torch.arange(64, dtype=torch.int32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 & src1).cpu())
    print("vbitwise_and example passed")
    print(f"first={int(dst.cpu()[0])}, last={int(dst.cpu()[-1])}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vbitwise_and example passed
first=0, last=4
```
