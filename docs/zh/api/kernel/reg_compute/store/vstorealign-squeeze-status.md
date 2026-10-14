---
title: vstorealign_squeeze_status
api_name: vstorealign_squeeze_status
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vstorealign_squeeze_status`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

将`vsqueeze_and_storeunalign`操作后保存在AR特殊寄存器中的有效数据长度写入Unified Buffer（UB）。调用无`offset`参数的重载时，写入`tensor`；调用带`offset`参数的重载时，写入`tensor`加元素偏移`offset`的位置。

本接口仅在AIV上生效。

## 函数原型

```python
def vstorealign_squeeze_status(tensor, offset=0, *, post_update: bool=False) -> None: ...
```

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `tensor` | 输出 | 目的操作数在UB上的基地址。无`offset`参数时，实际写入地址为`tensor`；带`offset`参数时，实际写入地址为`tensor`加上`offset`个元素。 |
| `offset` | 输入 | 仅带`offset`参数的重载使用。相对`tensor`基地址的偏移量，类型为`dtypes.int32`，单位为元素；无`offset`参数时，默认值为`0`。 |
| `post_update` | 输入 | 是否采用Post Update模式。取值为`True`或`False`，默认值为`False`。取值为`True`时，接口采用Post Update模式，搬出完成后自动更新目的地址指针，`offset`为目的地址更新量，类型为`dtypes.int32`，单位为元素。接口执行后，目的地址偏移`offset × 4`字节。 |

## 返回值说明

- 无返回值。

## 约束说明

- 本接口仅在AIV上生效。
- 本接口需在VF作用域内调用。
- 实际目的地址需4字节对齐，且不得超出实际可用UB范围；无`offset`参数时实际目的地址为`tensor`，带`offset`参数时为`tensor`加`offset`个元素。
- 调用本接口前，需使用[`vsqueeze_and_storeunalign`](../reg_permute_sel/vsqueeze-and-storeunalign.md)完成squeeze操作；写入的值是其当前累计的有效数据字节数。
- 如果本接口与其他指令存在UB地址重叠，需要插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化。

## 调用示例

将代码保存为`vstorealign_squeeze_status.py`后，可通过`python`命令运行。

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
    vstorealign_squeeze_status,
)
from cannbotdsl.tensor import MemLoc

@kernel
def _vstorealign_squeeze_status_kernel(src0, dst, status):
    buf = Channel(MemLoc.UB, (64,), src0.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)
    st = Channel(MemLoc.UB, (1,), status.dtype, depth=1)

    mem_copy(buf.produce(), src0)

    in0 = buf.consume()
    res = out.produce()
    res2 = st.produce()
    with vf(mode="simd"):
        mask = create_mask(pattern="vl32", elem_bits=32)
        ureg = vstore_unalign_begin(res)
        sq = vsqueeze_and_storeunalign_init(vload(in0, 0), mask=mask)
        vsqueeze_and_storeunalign(res, 0, sq, ureg)
        vstorealign_squeeze_status(res2, 0)

    mem_copy(dst, out.consume())
    mem_copy(status, st.consume())

@host
def run(src0, dst, status):
    _vstorealign_squeeze_status_kernel[1](src0, dst, status)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)
    status = torch.zeros((1,), dtype=torch.int32, device="npu:0")

    run(src0, dst, status)
    torch.npu.synchronize()

    assert int(status.cpu()[0]) == 128, f"squeeze status is not 128 bytes: {int(status.cpu()[0])}"
    print("vstorealign_squeeze_status example passed")
    print(f"status={int(status.cpu()[0])}, first={float(dst.cpu()[0]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vstorealign_squeeze_status example passed
status=128, first=0.0000
```
