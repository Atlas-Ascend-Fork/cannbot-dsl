---
title: vload_unalign
api_name: vload_unalign
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vload_unalign`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

从Unified Buffer（UB）中按dtype对齐的起始地址读取VL长度连续数据，并搬入矢量数据寄存器。本接口将`vload_unalign_init`中的非对齐寄存器（32字节）中缓存的前置数据与从UB读取的后续数据拼接，得到VL长度数据并搬入矢量数据寄存器。连续搬入时，需要在每次调用前手动更新源地址。

设本次实际读取的起始字节地址为`src_start_addr`，结束字节地址为`src_end_addr`，其中`src_end_addr = src_start_addr + VL`；将`src_start_addr`向低地址方向对齐到32字节边界，得到`aligned_src_start_addr`。`vload_unalign_init`将字节地址范围`[aligned_src_start_addr, aligned_src_start_addr + 32)`的数据缓存到非对齐寄存器，本接口将该缓存与从UB读取的后续数据拼接，得到字节地址范围`[src_start_addr, src_end_addr)`的数据。

本接口仅在AIV上生效。

## 函数原型

```python
def vload_unalign(tensor, offset=0) -> RawVReg: ...
```

**支持的数据类型：**

dtype支持的数据类型为`int4b_t`、`dtypes.int8`、`dtypes.uint8`、`dtypes.fp4x2_e2m1`、`dtypes.fp4x2_e1m2`、`dtypes.hifloat8`、`dtypes.float8_e8m0`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`、`dtypes.int64`。当dtype为`int4b_t`时，返回的矢量数据寄存器的实际类型为`dtypes.int4x2`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `tensor` | 输入 | 本次搬入的有效起始UB地址，必须按dtype对齐。本接口不会更新该地址。 |
| `offset` | 输入 | 相对`tensor`起始地址的偏移，单位为元素个数。 |

## 返回值说明

返回保存非对齐搬入结果的矢量数据寄存器，数据类型与`tensor`保持一致。

## 约束说明

- 本接口仅在AIV上生效。
- 本接口需在VF作用域内调用。
- `tensor`起始地址必须按dtype对齐，且实际访问范围必须在UB地址空间内，否则会报错。
- UB容量上限为256KB，用户可用容量随编译选项与编程场景变化（默认预留6KB SIMD VF栈 + 2KB 框架预留，可用248KB；SIMD+SIMT混编时再划分32KB～128KB作Data Cache，可用容量进一步减少）。UB地址偏移后不可超过实际可用容量，否则会报错。
- 如果本指令与其他指令存在UB地址重叠，必须插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化，防止出现异常数据。
- 每次搬入前必须调用`vload_unalign_init`中的源地址预处理模式进行初始化，且初始化接口和本接口必须使用相同的`tensor`。

## 调用示例

将代码保存为`vload_unalign.py`后，可通过`python`命令运行。

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
def _vload_unalign_kernel(src0, dst):
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
    _vload_unalign_kernel[1](src0, dst)

def main():
    src0 = torch.arange(65, dtype=torch.float32, device="npu:0")
    dst = torch.empty((64,), dtype=torch.float32, device="npu:0")

    run(src0, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), src0.cpu()[1:])
    print("vload_unalign example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vload_unalign example passed
first=1.0000, last=64.0000
```
