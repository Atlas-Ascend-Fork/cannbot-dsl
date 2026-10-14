---
title: vsqueeze_and_storeunalign_finalize
api_name: vsqueeze_and_storeunalign_finalize
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vsqueeze_and_storeunalign_finalize`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

Reg计算数据搬运接口，用于结束一组[`vsqueeze_and_storeunalign`](./vsqueeze-and-storeunalign.md)连续搬出操作，将暂存在非对齐寄存器`ureg`中的尾块写入Unified Buffer（UB）。

本接口仅在AIV上生效。

## 函数原型

```python
def vsqueeze_and_storeunalign_finalize(tensor, offset, ureg: StoreUnalignReg) -> None: ...
```

**支持的数据类型：**

`dtypes.int4x2`、`dtypes.int8`、`dtypes.uint8`、`dtypes.fp4x2_e2m1`、`dtypes.fp4x2_e1m2`、`dtypes.float8_e8m0`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `tensor` | 输出 | 目的操作数，Unified Buffer（UB）中的目的缓冲，需与同组`vsqueeze_and_storeunalign`调用中的`tensor`保持一致。 |
| `offset` | 输入 | 相对于`tensor`的元素偏移，需与同组`vsqueeze_and_storeunalign`调用中的`offset`保持一致。 |
| `ureg` | 输入 | 非对齐寄存器，长度为32B，保存待写入UB的尾块；需与同组`vsqueeze_and_storeunalign`调用中的`ureg`保持一致。 |

## 返回值说明

- 无返回值。
## 流水类型

`PIPE_V`

## 约束说明

### 通用约束

- UB容量上限为256KB，用户可用容量随编译选项与编程场景变化。目的操作数的尾块地址范围不可超过实际可用容量，否则会触发写越界异常。
- `tensor`的地址无需32B对齐，但需按`sizeof(dtype)`字节对齐。
- 如果本接口与其他指令存在UB地址重叠，需要插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化。

### 指令约束

- 本接口需在VF作用域内调用。
- 调用本接口前，需先调用一次或多次`vsqueeze_and_storeunalign`；每组连续搬出操作仅在最后一次主接口调用后执行一次本接口。

## 调用示例

将代码保存为`vsqueeze_and_storeunalign_finalize.py`后，可通过`python`命令运行。

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
    vsqueeze_and_storeunalign_finalize,
    vsqueeze_and_storeunalign_init,
    vstore_unalign_begin,
)
from cannbotdsl.tensor import MemLoc

@kernel
def _vsqueeze_and_storeunalign_finalize_kernel(src0, dst):
    buf = Channel(MemLoc.UB, (64,), src0.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    mem_copy(buf.produce(), src0)

    in0 = buf.consume()
    res = out.produce()
    with vf(mode="simd"):
        mask = create_mask(pattern="m3", elem_bits=32)
        sq = vsqueeze_and_storeunalign_init(vload(in0, 0), mask=mask)
        ureg = vstore_unalign_begin(res)
        vsqueeze_and_storeunalign(res, 0, sq, ureg)
        vsqueeze_and_storeunalign_finalize(res, 0, ureg)

    mem_copy(dst, out.consume())

@host
def run(src0, dst):
    _vsqueeze_and_storeunalign_finalize_kernel[1](src0, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu()[:22], src0.cpu()[0::3])
    print("vsqueeze_and_storeunalign_finalize example passed")
    print(f"first={float(dst.cpu()[0]):.1f}, last={float(dst.cpu()[21]):.1f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vsqueeze_and_storeunalign_finalize example passed
first=0.0, last=63.0
```
