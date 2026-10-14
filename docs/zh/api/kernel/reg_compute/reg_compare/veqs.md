---
title: veqs
api_name: veqs
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `veqs`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

根据`mask`将源操作数`src`按元素与标量`scalar`进行二进制比较，若$src_i = scalar$则计算结果对应位为1，否则为0。计算结果作为返回值返回，每个元素的比较结果占一个bit。计算公式如下：

$$
dst_i =
\begin{cases}
1, & src_i = scalar \\
0, & src_i \neq scalar \\
\end{cases}
$$

本接口仅在AIV上生效。

## 函数原型

```python
def veqs(src: RawVReg, scalar, *, mask: Mask) -> Mask: ...
```

**支持的数据类型：**

`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `scalar` | 输入 | 源操作数（标量）。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。mask中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。 |

## 返回值说明

- 返回比较结果，类型为掩码寄存器。

## 约束说明

- 本接口需在VF作用域内调用，`src`为矢量数据寄存器。
- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- `mask`比特位为0时，计算结果对应比特位写0。
- 浮点比较时，+0.0与-0.0视为相等。
- 浮点比较输入含nan时，计算结果对应比特位写0。

## 调用示例

将代码保存为`veqs.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, dtypes, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vdups, veqs, vload, vselect, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _veqs_kernel(src, dst):
    buf = Channel(MemLoc.UB, (64,), src.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)
    mem_copy(buf.produce(), src)

    in0, res = buf.consume(), out.produce()
    with vf(mode="simd"):
        mask = full_mask()
        pred = veqs(vload(in0, 0), 3.0, mask=mask)
        r = vselect(vdups(1.0, dtypes.float32),
                           vdups(0.0, dtypes.float32), cond_mask=pred)
        vstore(res, 0, r, mask)
    mem_copy(dst, out.consume())

@host
def run(src, dst):
    _veqs_kernel[1](src, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0") % 8.0
    dst = torch.empty_like(src0)

    run(src0, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 == 3.0).float().cpu())
    print(f"count={int(dst.sum().item())}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
count=8
```
