---
title: vcompress
api_name: vcompress
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vcompress`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

将`src`中被`mask`选择的有效元素作为返回值返回。有效元素在计算结果中从低到高连续排列，剩余位置元素置为0。

本接口不会将有效数据大小保存至AR寄存器。如果需要筛选有效元素并将其连续搬出至Unified Buffer（UB），请参考[`vsqueeze_and_storeunalign`](./vsqueeze-and-storeunalign.md)。

本接口仅在AIV上生效。

## 函数原型

```python
def vcompress(src: RawVReg, *, mask: Mask) -> RawVReg: ...
```

**支持的数据类型：**

`dtypes.int8`、`dtypes.uint8`、`dtypes.hifloat8`、`dtypes.float8_e8m0`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。mask中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。 |

## 返回值说明

- 返回按掩码压缩后的结果，类型为矢量数据寄存器，与`src`的数据类型一致。

## 约束说明

- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- 本接口需在VF作用域内调用，`src`为矢量数据寄存器。
- `mask`比特位为1的`src`元素按原顺序紧凑排列到计算结果低位；`mask`比特位为0的`src`元素不参与压缩，计算结果中压缩结果之后的剩余高位统一写0。

## 调用示例

将代码保存为`vcompress.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import create_mask, full_mask, vcompress, vload, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vcompress_kernel(src0, dst):
    buf = Channel(MemLoc.UB, (64,), src0.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    mem_copy(buf.produce(), src0)

    in0 = buf.consume()
    res = out.produce()
    with vf(mode="simd"):
        mask = create_mask(pattern="vl32", elem_bits=32)
        vstore(res, 0, vcompress(vload(in0, 0), mask=mask), full_mask())

    mem_copy(dst, out.consume())

@host
def run(src0, dst):
    _vcompress_kernel[1](src0, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, dst)
    torch.npu.synchronize()

    expected = torch.zeros(64)
    expected[:32] = src0.cpu()[:32]
    torch.testing.assert_close(dst.cpu(), expected)
    print("vcompress example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[31]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vcompress example passed
first=0.0000, last=31.0000
```
