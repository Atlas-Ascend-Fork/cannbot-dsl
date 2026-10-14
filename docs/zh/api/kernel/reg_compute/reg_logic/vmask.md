---
title: vmask
api_name: vmask
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vmask`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

将`src`中被`mask`筛选的有效元素复制到返回值对应位置；返回值的寄存器由本接口分配、未经初始化，因此未被`mask`筛选的位置值未定义。

本接口仅在AIV上生效。

## 函数原型

```python
def vmask(src: RawVReg, *, mask: Mask) -> RawVReg: ...
```

**支持的数据类型：**

`dtype`支持的数据类型：`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `src` | 输入 | 源矢量数据寄存器，`dtype`须与返回的矢量数据寄存器完全一致。 |
| `mask` | 输入 | 源掩码寄存器，用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。`mask`未筛选的元素在输出中值未定义。 |

## 返回值说明

- 返回保存掩码复制结果的矢量数据寄存器，数据类型与`src`保持一致。

## 约束说明

- 本接口仅在AIV上生效。
- 本接口需在VF作用域内调用。
- 未被`mask`筛选的位置值未定义：返回值对应的寄存器由本接口分配、未经初始化，调用方不应依赖未被`mask`筛选的位置的值。
- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- `mask`的掩码位宽须与`src`的元素位宽一致（由掩码设置接口按`src`的元素位宽创建），否则有效元素位置错误。

## 调用示例

将代码保存为`vmask.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import create_mask, full_mask, vload, vmask, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vmask_kernel(src, dst):
    buf = Channel(MemLoc.UB, (64,), src.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    mem_copy(buf.produce(), src)

    in0 = buf.consume()
    res = out.produce()
    with vf(mode="simd"):
        full = full_mask()
        half = create_mask(pattern="vl32")
        vstore(res, 0, vmask(vload(in0, 0), mask=half), full)

    mem_copy(dst, out.consume())

@host
def run(src, dst):
    _vmask_kernel[1](src, dst)

def main():
    src = torch.arange(64, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src)

    run(src, dst)
    torch.npu.synchronize()

    # mask 未筛选的位置值未定义，仅校验已筛选的前32个元素。
    torch.testing.assert_close(dst.cpu()[:32], src.cpu()[:32])
    print("vmask example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, lane31={float(dst.cpu()[31]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vmask example passed
first=0.0000, lane31=31.0000
```
