---
title: vmask_store
api_name: vmask_store
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vmask_store`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

将掩码寄存器中的数据以对齐的方式搬出到Unified Buffer（UB），单次搬出量为掩码寄存器长度（32字节），不支持配置掩码。搬运过程中数据格式与内容保持不变。目的操作数为UB地址。

## 函数原型

```python
def vmask_store(tensor, offset, mask: Mask, *, dist='norm') -> None: ...
```

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `tensor` | 输出 | 目的操作数在UB中的基地址，起始地址需32字节对齐。 |
| `offset` | 输入 | 相对`tensor`起始地址的偏移量，类型为`dtypes.int32`，单位为元素个数。 |
| `mask` | 输入 | 源操作数（掩码寄存器）。 |
| `dist` | 输入 | 掩码搬出模式。取值为`'norm'`（连续对齐搬出）或`'pack'`（每间隔1 bit丢弃1 bit，将保留的bit连续压缩存储）。 |

## 返回值说明

- 无返回值。

## 约束说明

- 本接口仅在AIV上生效。
- 本接口需在VF作用域内调用。
- `tensor`起始地址需32字节对齐，否则会报错。
- UB容量上限为256KB，用户可用容量随编译选项与编程场景变化（默认预留6KB SIMD VF栈+2KB 框架预留，可用248KB；SIMD+SIMT混编时再划分32KB~128KB作Data Cache，可用容量进一步减少）。目的操作数地址偏移后不可超过实际可用容量，否则会报错。
- 如果本指令与其他指令存在UB地址重叠，需要插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化，防止出现异常数据。
- `offset`的单位为元素个数。通过`offset`参数偏移后的实际访问地址需落在UB地址范围内，且仍需32字节对齐，否则会报错。

## 调用示例

将代码保存为`vmask_store.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import create_mask, vmask_store
from cannbotdsl.tensor import MemLoc

@kernel
def _vmask_store_kernel(dst):
    out = Channel(MemLoc.UB, (8,), dst.dtype, depth=1)

    res = out.produce()
    with vf(mode="simd"):
        mask = create_mask(pattern="vl32", elem_bits=32)
        vmask_store(res, 0, mask)

    mem_copy(dst, out.consume())

@host
def run(dst):
    _vmask_store_kernel[1](dst)

def main():
    dst = torch.zeros((8,), dtype=torch.int32, device="npu:0")

    run(dst)
    torch.npu.synchronize()

    words = [int(w) & 0xFFFFFFFF for w in dst.cpu().tolist()]
    assert sum(w.bit_count() for w in words) == 32, f"unexpected mask layout: {words}"
    print("vmask_store example passed")
    print(f"word0={words[0]:#010x}, set_bits={sum(w.bit_count() for w in words)}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vmask_store example passed
word0=0x11111111, set_bits=32
```
