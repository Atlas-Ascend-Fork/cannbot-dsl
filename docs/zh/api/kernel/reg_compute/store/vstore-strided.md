---
title: vstore_strided
api_name: vstore_strided
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vstore_strided`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

将矢量数据寄存器中的数据非连续对齐搬出到Unified Buffer（UB）。单条指令搬运8个`DataBlock`，一个`DataBlock`数据量为32字节，通过`block_stride`和`repeat_stride`配置相邻`DataBlock`的步长和相对目的起始地址的偏移。搬出过程中数据格式与内容保持不变。目的操作数为UB地址。

本接口仅在AIV上生效。

## 函数原型

```python
def vstore_strided(tensor, offset, value: RawVReg, mask: Mask, *, block_stride, repeat_stride=0) -> None: ...
```

**支持的数据类型：**

dtype支持的数据类型为`dtypes.int8`、`dtypes.uint8`、`dtypes.hifloat8`、`dtypes.float8_e8m0`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `tensor` | 输出 | 目的操作数（矢量）的起始地址。起始地址需32字节对齐。 |
| `offset` | 输入 | 相对`tensor`起始地址的偏移量，类型为`dtypes.int32`，单位为元素。 |
| `value` | 输入 | 源操作数（矢量数据寄存器）。dtype须与`tensor`一致。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示参与搬出的元素。对应位置为1时参与搬出，为0时不参与搬出。 |
| `block_stride` | 输入 | 目的操作数相邻DataBlock之间起始地址的步长，类型为`dtypes.uint16`，单位为32字节。 |
| `repeat_stride` | 输入 | 目的操作数起始地址偏移，类型为`dtypes.uint16`，单位为32字节。 |

## 返回值说明

- 无返回值。

## 约束说明

### 通用约束

- 本接口仅在AIV上生效。
- 本接口需在VF作用域内调用。
- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- `tensor`起始地址需32字节对齐，否则会报错。
- UB容量上限为256KB，用户可用容量随编译选项与编程场景变化（默认预留6KB SIMD VF栈+2KB 框架预留，可用248KB；SIMD+SIMT混编时再划分32KB~128KB作Data Cache，可用容量进一步减少）。目的操作数地址偏移后不可超过实际可用容量，否则会报错。
- 如果本指令与其他指令存在UB地址重叠，需要插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化，防止出现异常数据。

### 矢量数据寄存器非连续对齐搬出模式

- `mask`比特位为0的位置采用保持模式：`tensor`对应位置保持原值不变，不写入`value`数据。
- 通过`offset`、`block_stride`、`repeat_stride`等参数偏移后的实际访问地址落在UB地址范围内，且实际访问地址仍需32字节对齐，否则会报错。

## 调用示例

将代码保存为`vstore_strided.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload, vstore_strided
from cannbotdsl.tensor import MemLoc

@kernel
def _vstore_strided_kernel(src0, dst):
    buf = Channel(MemLoc.UB, (128,), src0.dtype, depth=1)
    out = Channel(MemLoc.UB, (128,), dst.dtype, depth=1)

    mem_copy(buf.produce(), src0)

    in0 = buf.consume()
    res = out.produce()
    with vf(mode="simd"):
        mask = full_mask()
        lo = vload(in0, 0)
        hi = vload(in0, 64)
        vstore_strided(res, 0, lo, mask, block_stride=2, repeat_stride=0)
        vstore_strided(res, 0, hi, mask, block_stride=2, repeat_stride=1)

    mem_copy(dst, out.consume())

@host
def run(src0, dst):
    _vstore_strided_kernel[1](src0, dst)

def main():
    src0 = torch.arange(128, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, dst)
    torch.npu.synchronize()

    # 搬出步长为2：寄存器lo写入第0、2、…、14个DataBlock，寄存器hi写入第1、3、…、15个DataBlock
    rows = src0.cpu().view(16, 8)
    torch.testing.assert_close(dst.cpu().view(16, 8), torch.stack([rows[:8], rows[8:]], dim=1).reshape(16, 8))
    print("vstore_strided example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vstore_strided example passed
first=0.0000, last=127.0000
```
