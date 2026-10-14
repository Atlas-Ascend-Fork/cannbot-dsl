---
title: vneg
api_name: vneg
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vneg`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

根据`mask`对源操作数`src`按元素取相反数，并将结果作为返回值返回。

计算公式如下：

$$
dst_i = -src_i
$$

本接口仅在AIV上生效。

## 函数原型

```python
def vneg(src: RawVReg, *, mask: Mask) -> RawVReg: ...
```

**支持的数据类型：**

`dtypes.int8`、`dtypes.int16`、`dtypes.float16`、`dtypes.int32`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。mask中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。 |

## 返回值说明

- 返回取反结果，类型为矢量数据寄存器，与`src`的数据类型一致。

## 约束说明

- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- 本接口需在VF作用域内调用，`src`为矢量数据寄存器。
- `mask`比特位为0时，计算结果对应比特位写0。
- 整数输入为有符号整数类型的最小负值时，结果保留原值不变。
- 浮点输入通过翻转符号位实现取反：正数变为对应负数，负数变为对应正数，+0.0变为-0.0，-0.0变为+0.0，+inf变为-inf，-inf变为+inf。
- 浮点输入nan，取反结果为nan。

## 调用示例

将代码保存为`vneg.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload, vneg, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vneg_kernel(src, dst):
    buf = Channel(MemLoc.UB, (64,), src.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    mem_copy(buf.produce(), src)

    in0 = buf.consume()
    res = out.produce()
    with vf(mode="simd"):
        mask = full_mask()
        vstore(res, 0, vneg(vload(in0, 0), mask=mask), mask)

    mem_copy(dst, out.consume())

@host
def run(src, dst):
    _vneg_kernel[1](src, dst)

def main():
    src = torch.arange(64, dtype=torch.float32, device="npu:0") - 32.0
    dst = torch.zeros_like(src)

    run(src, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (-src).cpu())
    print("vneg example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vneg example passed
first=32.0000, last=-31.0000
```
