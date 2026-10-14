---
title: vload_strided
api_name: vload_strided
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vload_strided`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

从Unified Buffer（UB）中32字节对齐的起始地址非连续搬入8个`DataBlock`，并通过函数返回值返回保存搬入结果的矢量数据寄存器。每个`DataBlock`的数据量为32字节，支持配置相邻数据块之间的地址步长和本次搬入的起始读取位置。

本接口与[`vload`](vload.md)的非连续对齐搬入模式功能相同，区别在于本接口通过函数返回值返回结果。

本接口仅在AIV上生效。

## 函数原型

```python
def vload_strided(tensor, offset, mask: Mask, *, block_stride, repeat_stride=0) -> RawVReg: ...
```

**支持的数据类型：**

dtype支持的数据类型为`int4b_t`、`dtypes.int8`、`dtypes.uint8`、`dtypes.fp4x2_e2m1`、`dtypes.fp4x2_e1m2`、`dtypes.hifloat8`、`dtypes.float8_e8m0`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`、`dtypes.int64`。当dtype为`int4b_t`时，返回的矢量数据寄存器的实际类型为`dtypes.int4x2`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `tensor` | 输入 | 源UB地址，实际读取地址必须按32字节对齐。 |
| `offset` | 输入 | 相对`tensor`起始地址的偏移，单位为元素个数。 |
| `mask` | 输入 | 掩码寄存器，用于指示在计算过程中哪些元素参与计算。该接口以`DataBlock`为数据搬运单元。<br>&bull; 当`DataBlock`中的任意一个元素被`mask`筛选成有效元素时，该`DataBlock`中所有数据都会搬入至矢量数据寄存器。<br>&bull; 当`DataBlock`中所有元素都被`mask`筛选成无效元素时，该`DataBlock`中的数据不会搬入到矢量数据寄存器，对应位置的元素设置为0，即使UB越界也不会报错。 |
| `block_stride` | 输入 | 源操作数相邻`DataBlock`之间起始地址的步长，单位为32字节。 |
| `repeat_stride` | 输入 | 本次搬入的起始读取地址相对`tensor`的偏移，单位为32字节。实际起始读取地址为`tensor`偏移`repeat_stride × 32`字节。 |

## 返回值说明

返回保存非连续对齐搬入结果的矢量数据寄存器，数据类型与`tensor`保持一致。

## 约束说明

### 通用约束

- 本接口仅在AIV上生效。
- 本接口需在VF作用域内调用。
- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- 实际读取地址必须按32字节对齐，且有效`DataBlock`的读取范围必须在UB地址空间内且不越界，否则会报错。
- 当一个`DataBlock`中的元素全部被`mask`设置为无效时，该`DataBlock`即使越界也不会报错。
- UB容量上限为256KB，用户可用容量随编译选项与编程场景变化（默认预留6KB SIMD VF栈 + 2KB 框架预留，可用248KB；SIMD+SIMT混编时再划分32KB～128KB作Data Cache，可用容量进一步减少）。UB地址偏移后不可超过实际可用容量，否则会报错。
- 如果本指令与其他指令存在UB地址重叠，需要插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化，防止出现异常数据。

### 矢量数据寄存器非连续对齐搬入模式

- 当dtype为`dtypes.int64`时，掩码以连续8个bit为一组，仅每组最低位的bit有效，用于控制对应的一个b64元素。

## 调用示例

将代码保存为`vload_strided.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload_strided, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vload_strided_kernel(src0, dst):
    buf = Channel(MemLoc.UB, (128,), src0.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    mem_copy(buf.produce(), src0)

    in0 = buf.consume()
    res = out.produce()
    with vf(mode="simd"):
        mask = full_mask()
        r = vload_strided(in0, 0, mask, block_stride=2, repeat_stride=1)
        vstore(res, 0, r, mask)

    mem_copy(dst, out.consume())

@host
def run(src0, dst):
    _vload_strided_kernel[1](src0, dst)

def main():
    src0 = torch.arange(128, dtype=torch.float32, device="npu:0")
    dst = torch.empty((64,), dtype=torch.float32, device="npu:0")

    run(src0, dst)
    torch.npu.synchronize()

    # 每个DataBlock为8个元素，搬入步长为2、起始偏移为1个DataBlock，即取第1、3、…、15个DataBlock
    torch.testing.assert_close(dst.cpu(), src0.cpu().view(16, 8)[1::2].reshape(-1))
    print("vload_strided example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vload_strided example passed
first=8.0000, last=127.0000
```
