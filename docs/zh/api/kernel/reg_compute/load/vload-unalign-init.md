---
title: vload_unalign_init
api_name: vload_unalign_init
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vload_unalign_init`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

本接口根据Unified Buffer（UB）中按dtype对齐的起始地址，初始化`非对齐寄存器`，作为后续非对齐搬入的前置缓存。该阶段仅完成非对齐寄存器初始化，不会将数据搬入矢量数据寄存器。

设本次初始化对应的有效起始字节地址为`src_start_addr`，将`src_start_addr`向低地址方向对齐到32字节边界，得到`aligned_src_start_addr`。本接口将UB中字节地址范围`[aligned_src_start_addr, aligned_src_start_addr + 32)`的32字节数据搬入非对齐寄存器。

本接口输出的非对齐寄存器作为`vload_unalign`的输入非对齐寄存器。

本接口提供一种功能模式：

- **源地址预处理模式**：以UB源地址作为起始地址，为`vload_unalign`的连续非对齐搬入模式准备前置数据缓存。

本接口仅在AIV上生效。

## 函数原型

```python
def vload_unalign_init(tensor, offset=0) -> None: ...
```

**支持的数据类型：**

dtype支持的数据类型为`int4b_t`、`dtypes.int8`、`dtypes.uint8`、`dtypes.fp4x2_e2m1`、`dtypes.fp4x2_e1m2`、`dtypes.hifloat8`、`dtypes.float8_e8m0`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`、`dtypes.int64`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `tensor` | 输入 | 源UB地址。`tensor`与`offset`共同确定的实际访问地址必须按dtype对齐。 |
| `offset` | 输入 | 相对`tensor`起始地址的偏移，单位为元素个数。 |

## 返回值说明

- 无返回值。

## 约束说明

### 通用约束

- 本接口仅在AIV上生效。
- 本接口需在VF作用域内调用。
- 实际访问地址在接口内部向低地址方向对齐到32字节边界后读取数据。传入的`tensor`或`tensor`与`offset`确定的地址必须按dtype对齐，对齐后的32字节读取范围必须在UB地址空间内，否则会报错。
- UB容量上限为256KB，用户可用容量随编译选项与编程场景变化（默认预留6KB SIMD VF栈 + 2KB 框架预留，可用248KB；SIMD+SIMT混编时再划分32KB～128KB作Data Cache，可用容量进一步减少）。UB地址偏移后不可超过实际可用容量，否则会报错。
- 如果本指令与其他指令存在UB地址重叠，必须插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化，防止出现异常数据。

### 源地址预处理模式

- 本接口和紧随其后的搬入必须使用相同的`tensor`地址。
- 与`vload_unalign`配合使用时，每次搬入前都必须调用本接口重新初始化。
- 与`vload_unalign`的连续非对齐搬入模式配合使用时，只有继续使用上次调用自动更新后的`tensor`时才可以复用；如果另外修改了`tensor`，必须重新初始化。

## 调用示例

将代码保存为`vload_unalign_init.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload_unalign, vload_unalign_init, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vload_unalign_init_kernel(src0, dst):
    buf = Channel(MemLoc.UB, (65,), src0.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    mem_copy(buf.produce(), src0)

    in0 = buf.consume()
    res = out.produce()
    with vf(mode="simd"):
        mask = full_mask()
        vload_unalign_init(in0, 1)
        vstore(res, 0, vload_unalign(in0, 1), mask)

    mem_copy(dst, out.consume())

@host
def run(src0, dst):
    _vload_unalign_init_kernel[1](src0, dst)

def main():
    src0 = torch.arange(65, dtype=torch.float32, device="npu:0")
    dst = torch.empty((64,), dtype=torch.float32, device="npu:0")

    run(src0, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), src0.cpu()[1:])
    print("vload_unalign_init example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vload_unalign_init example passed
first=1.0000, last=64.0000
```
