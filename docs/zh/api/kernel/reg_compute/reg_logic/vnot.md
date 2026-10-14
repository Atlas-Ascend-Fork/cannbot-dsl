---
title: vnot
api_name: vnot
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vnot`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

根据`mask`对源操作数`src`执行按位取反操作，将结果作为返回值返回。

- 矢量数据寄存器按位取反：对矢量数据寄存器执行按`dtype`位宽按位取反，结果为矢量数据寄存器。

计算公式如下：

$$
dst_i = \sim src_i
$$

本接口仅在AIV上生效。

## 函数原型

```python
def vnot(src: RawVReg, *, mask: Mask) -> RawVReg: ...
```

**支持的数据类型：**

`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器）。<br>&bull;源操作数为矢量数据寄存器时，对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

## 返回值说明

- 通过函数返回值返回结果的函数原型返回按位取反结果，返回类型与`src`的数据类型一致。

## 约束说明

- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- 参与计算的元素个数由矢量长度（VL）决定：矢量数据寄存器按位取反中元素个数 = VL ÷ sizeof(dtype)。
- `mask`比特位为0时，计算结果对应比特位写0。

## 调用示例

将代码保存为`vnot.py`后，可通过`python`命令运行。

```bash
python vnot.py
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
from cannbotdsl.ops.reg import full_mask, vload, vnot, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vnot_kernel(src, dst):
    buf = Channel(MemLoc.UB, (64,), src.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    mem_copy(buf.produce(), src)

    in0 = buf.consume()
    res = out.produce()
    with vf(mode="simd"):
        mask = full_mask()
        vstore(res, 0, vnot(vload(in0, 0), mask=mask), mask)

    mem_copy(dst, out.consume())

@host
def run(src, dst):
    _vnot_kernel[1](src, dst)

def main():
    src = torch.arange(64, dtype=torch.int32, device="npu:0")
    dst = torch.empty_like(src)

    run(src, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (~src).cpu())
    print("vnot example passed")
    print(f"first={int(dst.cpu()[0])}, last={int(dst.cpu()[-1])}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vnot example passed
first=-1, last=-64
```
