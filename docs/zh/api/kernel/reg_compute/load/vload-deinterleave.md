---
title: vload_deinterleave
api_name: vload_deinterleave
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vload_deinterleave`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

从Unified Buffer（UB）中32字节对齐的起始地址读取2×VL长度数据，将偶数下标元素和奇数下标元素分别写入两个矢量数据寄存器，结果作为返回值返回。本接口提供两种功能模式：

- **对齐搬入模式**：将UB源地址的数据解交织后搬入到两个矢量数据寄存器，由用户自行更新源地址。
- **立即数偏移搬入模式**：从相对源起始地址偏移指定距离的位置搬入数据。本接口不会自动更新源地址。

本接口仅在AIV上生效。

## 函数原型

```python
def vload_deinterleave(tensor, offset=0, *, width=None, post_update: bool=False) -> tuple[RawVReg, RawVReg]: ...
```

**支持的数据类型：**

dtype支持的数据类型为`int4b_t`、`dtypes.int8`、`dtypes.uint8`、`dtypes.fp4x2_e2m1`、`dtypes.fp4x2_e1m2`、`dtypes.hifloat8`、`dtypes.float8_e8m0`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。当dtype为`int4b_t`时，返回的两个矢量数据寄存器的实际类型均为`dtypes.int4x2`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `tensor` | 输入 | 源UB地址，实际读取地址必须按32字节对齐。 |
| `offset` | 输入 | 相对`tensor`起始地址的偏移，单位为元素个数。 |
| `width` | 输入 | 解交织的数据位宽，取值为`'b8'`、`'b16'`或`'b32'`，默认值为`None`。取值为`None`时按`tensor`的元素数据类型解交织；取值为其他值时，返回值中偶数下标元素和奇数下标元素的数据类型为该位宽对应的无符号整数类型。 |
| `post_update` | 输入 | 是否采用Post Update模式。取值为`True`或`False`，默认值为`False`。取值为`True`时，接口采用Post Update模式，搬入完成后自动更新源地址指针，`offset`为Post Update步长，单位为由`width`确定的元素个数。 |

## 返回值说明

返回解交织拆分后的两个矢量数据寄存器，分别对应偶数下标元素和奇数下标元素，数据类型与`tensor`保持一致。

## 约束说明

### 通用约束

- 本接口仅在AIV上生效。
- 本接口需在VF作用域内调用。
- 各功能模式下的实际读取地址必须按32字节对齐，且实际读取范围必须在UB地址空间内且不越界，否则会报错。
- UB容量上限为256KB，用户可用容量随编译选项与编程场景变化（默认预留6KB SIMD VF栈 + 2KB 框架预留，可用248KB；SIMD+SIMT混编时再划分32KB～128KB作Data Cache，可用容量进一步减少）。UB地址偏移后不可超过实际可用容量，否则会报错。
- 如果本指令与其他指令存在UB地址重叠，需要插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化，防止出现异常数据。

## 调用示例

将代码保存为`vload_deinterleave.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload_deinterleave, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vload_deinterleave_kernel(src0, dst0, dst1):
    buf = Channel(MemLoc.UB, (128,), src0.dtype, depth=1)
    out0 = Channel(MemLoc.UB, (64,), dst0.dtype, depth=1)
    out1 = Channel(MemLoc.UB, (64,), dst1.dtype, depth=1)

    mem_copy(buf.produce(), src0)

    in0 = buf.consume()
    res0 = out0.produce()
    res1 = out1.produce()
    with vf(mode="simd"):
        mask = full_mask()
        even, odd = vload_deinterleave(in0, 0)
        vstore(res0, 0, even, mask)
        vstore(res1, 0, odd, mask)

    mem_copy(dst0, out0.consume())
    mem_copy(dst1, out1.consume())

@host
def run(src0, dst0, dst1):
    _vload_deinterleave_kernel[1](src0, dst0, dst1)

def main():
    src0 = torch.arange(128, dtype=torch.float32, device="npu:0")
    dst0 = torch.empty((64,), dtype=torch.float32, device="npu:0")
    dst1 = torch.empty((64,), dtype=torch.float32, device="npu:0")

    run(src0, dst0, dst1)
    torch.npu.synchronize()

    torch.testing.assert_close(dst0.cpu(), src0.cpu()[0::2])
    torch.testing.assert_close(dst1.cpu(), src0.cpu()[1::2])
    print("vload_deinterleave example passed")
    print(f"even_first={float(dst0.cpu()[0]):.4f}, odd_first={float(dst1.cpu()[0]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vload_deinterleave example passed
even_first=0.0000, odd_first=1.0000
```
