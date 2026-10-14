---
title: vstore_unalign_post
api_name: vstore_unalign_post
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vstore_unalign_post`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

将非对齐寄存器中暂存的尾块写入Unified Buffer（UB），用于连续非对齐搬运收尾。本接口提供以下一种模式：

- **立即数偏移搬出模式**：通过立即数指定相对目的起始地址的偏移，单位为元素，用户可选择手动更新偏移或更新目的地址，用于配合前序非对齐搬出接口`vstore_unalign`收尾。

本接口仅在AIV上生效。

## 函数原型

```python
def vstore_unalign_post(tensor, offset, ureg: StoreUnalignReg) -> None: ...
```

**支持的数据类型：**

`dtype`支持的数据类型为`int4b_t`、`dtypes.int8`、`dtypes.uint8`、`dtypes.fp4x2_e2m1`、`dtypes.fp4x2_e1m2`、`dtypes.hifloat8`、`dtypes.float8_e8m0`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`、`dtypes.int64`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `tensor` | 输出 | 目的操作数在UB中的基地址。地址无需32字节对齐，但须按`sizeof(dtype)`字节对齐。 |
| `offset` | 输入 | 目的操作数结束地址的偏移，类型为`dtypes.int32`，单位为元素。 |
| `ureg` | 输入 | 非对齐寄存器，类型为`StoreUnalignReg`，长度为32字节。须与前序主搬出接口使用同一个寄存器。 |

## 返回值说明

- 无返回值。

## 约束说明

### 通用约束

- 本接口仅在AIV上生效。
- 本接口需在VF作用域内调用。
- 需要保证目的操作数的地址加上`offset`对应的偏移地址，访问范围须位于实际可用UB范围内。
- 该接口中的目的地址不需要32B对齐，但数据类型为`dtype`的`tensor`需要`sizeof(dtype)`字节对齐。
- UB容量上限为256KB，用户可用容量随编译选项与编程场景变化（默认预留6KB SIMD VF栈+2KB 框架预留，可用248KB；SIMD+SIMT混编时再划分32KB~128KB作Data Cache，可用容量进一步减少）。目的操作数地址不可超过实际可用容量，否则会报错。
- 如果本指令与其他指令存在UB地址重叠，需要插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化，防止出现异常数据。

### 立即数偏移搬出模式

- 与`vstore_unalign`配套使用，需要复用同一个`ureg`。
- 需要保证目的地址加上偏移后的地址，等于`vstore_unalign`搬运的结束地址，否则无法正确收尾。

## 调用示例

将代码保存为`vstore_unalign_post.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import (
    create_mask,
    vload,
    vsqueeze_and_storeunalign,
    vsqueeze_and_storeunalign_init,
    vstore_unalign_begin,
    vstore_unalign_post,
)
from cannbotdsl.tensor import MemLoc

@kernel
def _vstore_unalign_post_kernel(src0, dst):
    buf = Channel(MemLoc.UB, (64,), src0.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    mem_copy(buf.produce(), src0)

    in0 = buf.consume()
    res = out.produce()
    with vf(mode="simd"):
        mask = create_mask(pattern="vl32", elem_bits=32)
        ureg = vstore_unalign_begin(res)
        sq = vsqueeze_and_storeunalign_init(vload(in0, 0), mask=mask)
        vsqueeze_and_storeunalign(res, 0, sq, ureg)
        vstore_unalign_post(res, 0, ureg)

    mem_copy(dst, out.consume())

@host
def run(src0, dst):
    _vstore_unalign_post_kernel[1](src0, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, dst)
    torch.npu.synchronize()

    assert bool((dst.cpu()[:32] == src0.cpu()[:32]).all()), f"tail mismatch: {dst.cpu()[:4].tolist()}"
    print("vstore_unalign_post example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[31]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vstore_unalign_post example passed
first=0.0000, last=31.0000
```
