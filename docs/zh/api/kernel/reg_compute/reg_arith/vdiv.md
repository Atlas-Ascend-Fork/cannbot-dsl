---
title: vdiv
api_name: vdiv
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vdiv`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

该接口根据`mask`，对源操作数`lhs`、`rhs`进行按元素求商操作，将结果作为返回值返回。

计算公式如下：

$$
dst_i = lhs_i \div rhs_i
$$

本接口仅在AIV上生效。
## 函数原型

```python
def vdiv(lhs: RawVReg, rhs: RawVReg, *, mask: Mask, precision_mode: bool=False) -> RawVReg: ...
```

**支持的数据类型：**

`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `lhs` | 输入 | 源操作数（矢量数据寄存器）。 |
| `rhs` | 输入 | 源操作数（矢量数据寄存器）。 |
| `precision_mode` | 输入 | 精度模式开关。为`True`时使用计算结果与标准`IEEE 754`二进制浮点除法完全一致（0 ULP）的实现，仅支持`dtypes.float32`；默认为`False`，使用硬件除法实现，结果可能存在1 ULP精度差异。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

## 返回值说明

- 返回逐元素除法结果，返回值类型与源操作数类型一致。

## 约束说明

- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- 除零错误需由调用者避免；浮点dtype除零时结果为`IEEE 754`特殊值（`±inf`或`nan`）。
- `dtypes.float16`、`dtypes.float32`数据类型的计算结果与标准`IEEE 754`二进制浮点除法相比可能存在1 ULP精度差异。

## 调用示例

将代码保存为`vdiv.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vdiv, vload, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vdiv_kernel(src0, src1, dst):
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
        vstore(res, 0, vdiv(vload(in0, 0), vload(in1, 0), mask=mask), mask)

    mem_copy(dst, out.consume())

@host
def run(src0, src1, dst):
    _vdiv_kernel[1](src0, src1, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0") + 1.0
    src1 = torch.arange(64, dtype=torch.float32, device="npu:0") + 2.0
    dst = torch.zeros_like(src0)

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 / src1).cpu())
    print("vdiv example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vdiv example passed
first=0.5000, last=0.9846
```
