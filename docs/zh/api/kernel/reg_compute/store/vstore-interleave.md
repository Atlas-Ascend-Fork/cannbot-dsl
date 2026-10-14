---
title: vstore_interleave
api_name: vstore_interleave
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vstore_interleave`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

将两个矢量数据寄存器中的数据按元素顺序交织后连续对齐搬出到Unified Buffer（UB）。单次搬出量为`2 × VL`（共512字节），不支持配置掩码。搬运过程中数据格式与内容保持不变。

本接口提供两种参数列表不同的功能模式：

- **对齐搬出模式**：偏移固定为0，将`src0`和`src1`中的数据交织搬出到UB起始地址，由用户自行更新目的地址。
- **立即数偏移搬出模式**：通过`dtypes.int32`类型的`offset`指定相对目的起始地址的偏移，用户可选择更新偏移或更新目的地址。

本接口仅在AIV上生效。

## 函数原型

```python
def vstore_interleave(tensor, offset, src0: RawVReg, src1: RawVReg, *, width=None) -> None: ...
```

**支持的数据类型：**

`dtype`支持的数据类型为`int4b_t`、`dtypes.int8`、`dtypes.uint8`、`dtypes.fp4x2_e1m2`、`dtypes.fp4x2_e2m1`、`dtypes.hifloat8`、`dtypes.float8_e4m3fn`、`dtypes.float8_e5m2`、`dtypes.float8_e8m0`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `tensor` | 输出 | 目的操作数（矢量）的起始地址。起始地址需32字节对齐。 |
| `offset` | 输入 | 相对`tensor`起始地址的偏移量，类型为`dtypes.int32`，单位为元素。 |
| `src0` | 输入 | 源操作数0（矢量数据寄存器）。`dtype`须与`tensor`一致。 |
| `src1` | 输入 | 源操作数1（矢量数据寄存器）。`dtype`须与`tensor`一致。 |
| `width` | 输入 | 交织的数据位宽，取值为`'b8'`、`'b16'`或`'b32'`，须与`src0`的数据类型位宽一致，默认值为`None`。取值为`None`时按`src0`的数据类型位宽交织。 |

## 返回值说明

- 无返回值。

## 约束说明

### 通用约束

- 本接口仅在AIV上生效。
- 本接口需在VF作用域内调用。
- `tensor`起始地址需32字节对齐，否则会报错。
- UB容量上限为256KB，用户可用容量随编译选项与编程场景变化（默认预留6KB SIMD VF栈+2KB 框架预留，可用248KB；SIMD+SIMT混编时再划分32KB~128KB作Data Cache，可用容量进一步减少）。目的操作数地址偏移后不可超过实际可用容量，否则会报错。
- 通过`offset`参数偏移后的实际访问地址需落在UB地址范围内，且实际访问地址仍需32字节对齐，否则会报错。
- 如果本指令与其他指令存在UB地址重叠，需要插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化，防止出现异常数据。

## 调用示例

将代码保存为`vstore_interleave.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import vload, vstore_interleave
from cannbotdsl.tensor import MemLoc

@kernel
def _vstore_interleave_kernel(src0, src1, dst):
    buf0 = Channel(MemLoc.UB, (64,), src0.dtype, depth=1)
    buf1 = Channel(MemLoc.UB, (64,), src1.dtype, depth=1)
    out = Channel(MemLoc.UB, (128,), dst.dtype, depth=1)

    mem_copy(buf0.produce(), src0)
    mem_copy(buf1.produce(), src1)

    in0 = buf0.consume()
    in1 = buf1.consume()
    res = out.produce()
    with vf(mode="simd"):
        vstore_interleave(res, 0, vload(in0, 0), vload(in1, 0))

    mem_copy(dst, out.consume())

@host
def run(src0, src1, dst):
    _vstore_interleave_kernel[1](src0, src1, dst)

def main():
    src0 = torch.arange(64, dtype=torch.int32).to(torch.float16).to("npu:0")
    src1 = src0 + 100.0
    dst = torch.empty((128,), dtype=torch.float16, device="npu:0")
    expected = torch.zeros(128, dtype=torch.float16)
    expected[0::2] = src0.cpu()
    expected[1::2] = src1.cpu()

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), expected)
    print("vstore_interleave example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vstore_interleave example passed
first=0.0000, last=163.0000
```
