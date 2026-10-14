---
title: vshl
api_name: vshl
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vshl`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

根据掩码对源操作数矢量数据寄存器中的元素执行左移，左移位数由标量`shift_bits`指定，并将结果作为返回值返回。掩码对应位置为1的元素参与计算，为0的元素在输出结果中置零。计算公式如下：

$$
dst_i = src_i \ll shift\_bits
$$

根据源操作数的数据类型，左移操作分为以下两种情况：

- **无符号整数：执行逻辑左移。** 逻辑左移会将二进制数整体向左移动指定的位数，最高位被丢弃，最低位用0填充。例如，二进制数1010101010101010（`dtypes.uint16`类型）逻辑左移1位后，结果为0101010101010100。
- **有符号整数：执行算术左移。** 算术左移会将二进制数整体向左移动指定的位数，最高位被丢弃（符号位会被丢弃），最低位用0填充。例如，二进制数1010101010101010（`dtypes.int16`类型）算术左移1位后，结果为0101010101010100；算术左移3位后，结果为0101010101010000。

本接口仅在AIV上生效。

## 函数原型

```python
def vshl(src: RawVReg, shift_bits: int, *, mask: Mask) -> RawVReg: ...
```

**支持的数据类型：**

`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.int32`、`dtypes.uint32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `shift_bits` | 输入 | 位移量（标量），不支持设置为负数。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示参与计算的元素。对应位置为1时参与计算，为0时不参与计算且输出结果对应元素置零。需通过掩码设置接口预先赋值后再传入。 |

## 返回值说明

- 返回保存左移结果的矢量数据寄存器，数据类型与`src`保持一致。

## 约束说明

### 通用约束

- 本接口需在VF作用域内调用。
- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- 掩码位为0的元素位置不参与左移运算，输出结果对应位置写0。

### 位移约束

- `shift_bits`不支持设置为负数，负数行为未定义。
- 当`shift_bits`大于`src`的数据类型位宽时，输出结果中的有效元素置零。

## 调用示例

将代码保存为`vshl.py`后，可通过`python`命令运行。

```bash
python vshl.py
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
from cannbotdsl.ops.reg import full_mask, vload, vshl, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vshl_kernel(src, dst):
    buf = Channel(MemLoc.UB, (64,), src.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    mem_copy(buf.produce(), src)

    in0 = buf.consume()
    res = out.produce()
    with vf(mode="simd"):
        mask = full_mask()
        vstore(res, 0, vshl(vload(in0, 0), 2, mask=mask), mask)

    mem_copy(dst, out.consume())

@host
def run(src, dst):
    _vshl_kernel[1](src, dst)

def main():
    src = torch.arange(64, dtype=torch.int32, device="npu:0")
    dst = torch.empty_like(src)

    run(src, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src << 2).cpu())
    print("vshl example passed")
    print(f"first={int(dst.cpu()[0])}, last={int(dst.cpu()[-1])}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vshl example passed
first=0, last=252
```
