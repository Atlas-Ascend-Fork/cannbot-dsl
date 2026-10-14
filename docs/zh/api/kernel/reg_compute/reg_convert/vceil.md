---
title: vceil
api_name: vceil
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vceil`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

将`src`中的浮点数元素按照CEIL（向正无穷方向舍入）舍入模式舍入到整数值，结果仍保持原浮点数据类型。接口通过函数返回值返回结果，直接返回舍入结果。舍入规则等价于C标准库`ceil`语义：向正无穷方向舍入，即取大于或等于输入值的最小整数，正数向数值增大的方向取整（如1.3 -> 2.0），负数向0方向取整（如-1.7 -> -1.0），整数值保持不变。未被`mask`筛选的元素置零。

关于舍入模式的详细说明，请参见[舍入模式与饱和模式](./roundingmode.md)。

本接口仅在AIV上生效。

## 函数原型

```python
def vceil(src: RawVReg, *, mask: Mask) -> RawVReg: ...
```

**支持的数据类型：**

`<dtype>`取值为：`dtypes.float16`、`dtypes.bfloat16`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。`mask`未筛选的元素在输出中置零。 |

## 返回值说明

- 返回`dtypes.float16`、`dtypes.bfloat16`或`dtypes.float32`类型的矢量数据寄存器，保存舍入结果，数据类型与`src`一致。

## 约束说明

- 本接口需在VF作用域内调用。
- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- `dtypes.float16`和`dtypes.bfloat16`支持饱和模式。`dtypes.float32`类型只支持不饱和模式。
- `mask`掩码位为0时，结果对应元素置0。

## 调用示例

将代码保存为`vceil.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vceil, vload, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vceil_kernel(src, dst):
    buf = Channel(MemLoc.UB, (64,), src.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    mem_copy(buf.produce(), src)

    in0 = buf.consume()
    res = out.produce()
    with vf(mode="simd"):
        mask = full_mask()
        vstore(res, 0, vceil(vload(in0, 0), mask=mask), mask)

    mem_copy(dst, out.consume())

@host
def run(src, dst):
    _vceil_kernel[1](src, dst)

def main():
    src = (torch.arange(64, dtype=torch.float32, device="npu:0") - 32.0) / 3.0
    dst = torch.empty_like(src)

    run(src, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.ceil(src).cpu())
    print("vceil example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vceil example passed
first=-10.0000, last=11.0000
```
