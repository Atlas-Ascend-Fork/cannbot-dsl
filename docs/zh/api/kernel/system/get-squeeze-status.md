---
title: get_squeeze_status
api_name: get_squeeze_status
category: system
api_group: kernel
layer: system
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `get_squeeze_status`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

读取 `vsqueeze_and_storeunalign()` 操作后保存在 AR 特殊寄存器中的有效数据长度，即已连续搬出的有效数据字节数。

## 函数原型

```python
def get_squeeze_status() -> Int64: ...
```

## 参数说明

无。

## 返回值说明

返回 squeeze 有效数据长度，单位为字节。
## 流水类型

`PIPE_S`

## 约束说明

- 调用本接口前，需先调用 [`vstore_unalign_begin`](../reg_compute/reg_permute_sel/vstore-unalign-begin.md) 清空 AR 特殊寄存器，再调用 [`vsqueeze_and_storeunalign_init`](../reg_compute/reg_permute_sel/vsqueeze-and-storeunalign-init.md) 标记压缩点并调用 [`vsqueeze_and_storeunalign`](../reg_compute/reg_permute_sel/vsqueeze-and-storeunalign.md) 完成数据筛选；否则返回值没有确定含义。

## 调用示例

以下示例在统一缓冲区（Unified Buffer，UB）中压缩存储 4 个 `uint32` 元素，再读取并打印有效数据长度。

```python
import torch
import torch_npu  # noqa: F401

from cannbotdsl import UB, dtypes, host, make_buffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_squeeze_status
from cannbotdsl.ops.reg import (
    create_mask,
    vdups,
    vsqueeze_and_storeunalign,
    vsqueeze_and_storeunalign_finalize,
    vsqueeze_and_storeunalign_init,
    vstore_unalign_begin,
)

@jit
def compact_store(ub_output):
    with vf(mode="simd"):
        mask = create_mask(pattern="vl4", elem_bits=32)
        source = vdups(7, dtypes.uint32, mask=mask)
        cursor = vstore_unalign_begin(ub_output)
        squeezed = vsqueeze_and_storeunalign_init(source, mask=mask)
        vsqueeze_and_storeunalign(ub_output, 0, squeezed, cursor)
        vsqueeze_and_storeunalign_finalize(ub_output, 0, cursor)

@kernel
def _kernel(status):
    ub_space = UB.view(262144)
    ub_output = make_buffer(
        ub_space[slice(0, 256),], dtype=dtypes.uint32
    )
    compact_store(ub_output)
    status[0] = get_squeeze_status()

@host
def run(status):
    _kernel[1](status)

def main():
    status = torch.empty(1, dtype=torch.int64, device="npu")
    run(status)
    torch.npu.synchronize()
    valid_bytes = int(status.cpu()[0])
    assert valid_bytes == 16
    print(f"squeeze_valid_bytes: {valid_bytes}")
    print("get_squeeze_status example passed")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
squeeze_valid_bytes: 16
get_squeeze_status example passed
```
