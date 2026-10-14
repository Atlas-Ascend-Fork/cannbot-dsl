---
title: vexp
api_name: vexp
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vexp`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

该接口根据`mask`，对源操作数`src`进行按元素求自然指数操作，将结果作为返回值返回。

计算公式如下：
$$
dst_i = e^{src_i}
$$

本接口仅在AIV上生效。
## 函数原型

```python
def vexp(src: RawVReg, *, mask: Mask) -> RawVReg: ...
```

**支持的数据类型：**

`dtypes.float16`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

## 返回值说明

- 返回逐元素指数计算结果，返回值类型与源操作数类型一致。

## 约束说明

- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。

## 调用示例

将代码保存为`vexp.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vexp, vload, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vexp_kernel(src, dst):
    buf = Channel(MemLoc.UB, (64,), src.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    mem_copy(buf.produce(), src)

    in0 = buf.consume()
    res = out.produce()
    with vf(mode="simd"):
        mask = full_mask()
        vstore(res, 0, vexp(vload(in0, 0), mask=mask), mask)

    mem_copy(dst, out.consume())

@host
def run(src, dst):
    _vexp_kernel[1](src, dst)

def main():
    src = torch.arange(64, dtype=torch.float32, device="npu:0") / 16.0
    dst = torch.zeros_like(src)

    run(src, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.exp(src).cpu(), rtol=1e-5, atol=1e-5)
    print("vexp example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vexp example passed
first=1.0000, last=51.2902
```
