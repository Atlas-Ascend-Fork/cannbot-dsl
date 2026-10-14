---
title: vgt
api_name: vgt
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vgt`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

gt表示大于（greater than），该接口用于逐元素比较两个源操作数大小，将比较结果（$lhs_i > rhs_i$）作为返回值返回。如果比较结果为真，则对应比特位为1，否则为0。

计算公式如下：

$$
dst_i = (lhs_i > rhs_i)
$$

本接口仅在AIV上生效。

## 函数原型

```python
def vgt(lhs: RawVReg, rhs: RawVReg, *, mask: Mask) -> Mask: ...
```

**支持的数据类型：**

`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `lhs` | 输入 | 源操作数（矢量数据寄存器）。 |
| `rhs` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。mask中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。 |

## 返回值说明

- 返回比较结果，类型为掩码寄存器。

## 约束说明

- 本接口需在VF作用域内调用，`lhs`、`rhs`为矢量数据寄存器。
- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- `mask`比特位为0时，计算结果对应比特位写0。
- 浮点比较时，+0.0与-0.0视为相等。
- 浮点比较输入含nan时，计算结果对应比特位写0。

## 调用示例

将代码保存为`vgt.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, dtypes, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vdups, vgt, vload, vselect, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vgt_kernel(src0, src1, dst):
    buf0 = Channel(MemLoc.UB, (64,), src0.dtype, depth=1)
    buf1 = Channel(MemLoc.UB, (64,), src1.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)
    mem_copy(buf0.produce(), src0)
    mem_copy(buf1.produce(), src1)

    in0, in1, res = buf0.consume(), buf1.consume(), out.produce()
    with vf(mode="simd"):
        mask = full_mask()
        pred = vgt(vload(in0, 0), vload(in1, 0), mask=mask)
        r = vselect(vdups(1.0, dtypes.float32),
                           vdups(0.0, dtypes.float32), cond_mask=pred)
        vstore(res, 0, r, mask)
    mem_copy(dst, out.consume())

@host
def run(src0, src1, dst):
    _vgt_kernel[1](src0, src1, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    src1 = torch.flip(src0, [0])
    dst = torch.empty_like(src0)

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 > src1).float().cpu())
    print(f"count={int(dst.sum().item())}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
count=32
```
