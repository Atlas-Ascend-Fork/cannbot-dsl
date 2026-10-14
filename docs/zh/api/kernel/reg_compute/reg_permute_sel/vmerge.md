---
title: vmerge
api_name: vmerge
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vmerge`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

根据`mask`从源操作数`new`、`old`中选择元素，得到计算结果。选择的规则为：当`mask`的比特位为1时，从`new`中选取对应位置的数；当`mask`的比特位为0时，从`old`中选取对应位置的数。计算公式如下：

$$
dst_i =
\begin{cases}
 new_i, & mask_i = 1 \\
 old_i, & mask_i = 0 \\
\end{cases}
$$

本接口仅在AIV上生效。

## 函数原型

```python
def vmerge(mask: Mask, new: RawVReg, old: RawVReg) -> RawVReg: ...
```

**支持的数据类型：**

`dtypes.bool_`、`dtypes.int8`、`dtypes.uint8`、`dtypes.hifloat8`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `mask` | 输入 | 源操作数掩码（掩码寄存器）。指定选择`new`或`old`为有效数据。`mask`的比特位为1时，选取`new`；`mask`的比特位为0时，选取`old`。 |
| `new` | 输入 | 源操作数（矢量数据寄存器）。`mask`的比特位为1时选取该操作数对应位置的元素。 |
| `old` | 输入 | 源操作数（矢量数据寄存器）。`mask`的比特位为0时选取该操作数对应位置的元素。 |

## 返回值说明

- 返回选择结果，类型为矢量数据寄存器，与`new`、`old`的数据类型一致。

## 约束说明

- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- 本接口需在VF作用域内调用。
- `new`和`old`的数据类型需要保持一致。

## 调用示例

将代码保存为`vmerge.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vgt, vload, vmerge, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vmerge_kernel(src0, src1, dst):
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
        a = vload(in0, 0)
        b = vload(in1, 0)
        vstore(res, 0, vmerge(vgt(a, b, mask=mask), a, b), mask)

    mem_copy(dst, out.consume())

@host
def run(src0, src1, dst):
    _vmerge_kernel[1](src0, src1, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    src1 = torch.full((64,), 32.0, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.where(src0.cpu() > src1.cpu(), src0.cpu(), src1.cpu()))
    print("vmerge example passed")
    print(f"first={float(dst.cpu()[0]):.1f}, last={float(dst.cpu()[-1]):.1f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vmerge example passed
first=32.0, last=63.0
```
