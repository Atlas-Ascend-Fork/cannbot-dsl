---
title: vcast
api_name: vcast
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vcast`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

将源操作数矢量数据寄存器中的元素转换为`dtype`指定的数据类型，并支持多种舍入模式与饱和模式。

关于舍入模式和饱和/非饱和模式的详细说明，请参见[舍入模式与饱和模式](./roundingmode.md)。

当源操作数与目的操作数类型位宽比为2:1时（例如`dtypes.float32`转换为`dtypes.float16`），写入数据时需要将一个`VL`大小的数据分为两部分，根据位置选择标签选择写入目的操作数索引为偶数的位置或奇数的位置。

```text
# 位置判断仅用于说明接口语义，实际位置参数为编译期常量。
# VL表示矢量数据寄存器位宽，取值256字节。
# 以下以位宽比2:1的dtypes.float32转换为dtypes.float16为例。
offset 取 0 表示写入偶数索引位置，取 1 表示写入奇数索引位置。
对于 i in range(VL // 4)：
    将目的操作数[2 * i]置0
    将目的操作数[2 * i + 1]置0
    如果 mask[i] 为真：
        目的操作数[2 * i + offset] 写为 src[i] 由 dtypes.float32 转换为 dtypes.float16 的结果（RN舍入）
```

本接口仅在AIV上生效。

## 函数原型

```python
def vcast(src: RawVReg, dtype, *, mask: Mask, rounding=RoundingMode.NA, saturate=False, reg_layout=RegLayout.UNKNOWN) -> RawVReg: ...
```

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `dtype` | 输入 | 转换后的元素数据类型，即返回值的数据类型。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。`mask`未筛选的元素在输出中置零。 |
| `rounding` | 输入 | 舍入模式，默认值为`RoundingMode.NA`。取值为`RoundingMode.NA`时，由接口按约定选择：位宽扩展的转换不进行舍入，位宽收窄的转换自动选择`RoundingMode.RN`；其他取值的含义参见[舍入模式与饱和模式](./roundingmode.md)。 |
| `saturate` | 输入 | 是否采用饱和模式。取值为`True`或`False`，默认值为`False`。取值为`True`时采用饱和模式，源数据超出目的数据类型表示范围时，结果限制在目的数据类型的可表示范围内；取值为`False`时采用非饱和模式，整型转换时结果被截断、保留最低有效位（LSB），浮点转换超出目的数据类型表示范围时返回对应符号的`inf`。饱和模式仅适用于位宽收窄的转换。 |
| `reg_layout` | 输入 | 位置选择标签，取值为`RegLayout.UNKNOWN`、`RegLayout.ZERO`、`RegLayout.ONE`、`RegLayout.TWO`或`RegLayout.THREE`，默认值为`RegLayout.UNKNOWN`。取值为`RegLayout.ZERO`时，选择将结果写入目的操作数索引为偶数的位置，其他位置清零；取值为`RegLayout.ONE`时，选择将结果写入目的操作数索引为奇数的位置，其他位置清零；取值为`RegLayout.TWO`、`RegLayout.THREE`时，从源操作数中离散选取每隔4个存储单元的位置读取数据，分别对应索引2、6、10、…和索引3、7、11、…。 |

## 返回值说明

- 返回计算结果，返回值类型与`dtype`一致。

## 约束说明

- 位置选择标签参数仅能使用编译期常量，编译器据此在编译期分发至对应的重载。
- 位置选择标签选择目的操作数的写入地址，其他位置清零。
- `src`的数据类型需要与函数原型匹配。
- `mask`掩码位为0时，结果对应元素置0。
- 结果写入奇数索引位置时，偶数索引位置置零；结果写入偶数索引位置时，奇数索引位置置零。
- 位置选择标签仅适用于位宽比不为1的转换：源与目的操作数位宽比为2时只能取`RegLayout.UNKNOWN`、`RegLayout.ZERO`或`RegLayout.ONE`，取`RegLayout.TWO`、`RegLayout.THREE`时源与目的操作数的位宽比须为4。

## 调用示例

将代码保存为`vcast.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, dtypes, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import PackMode, create_mask, vcast, vload, vstore_pack
from cannbotdsl.tensor import MemLoc

@kernel
def _vcast_kernel(src, dst):
    buf = Channel(MemLoc.UB, (64,), src.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    mem_copy(buf.produce(), src)

    in0 = buf.consume()
    res = out.produce()
    with vf(mode="simd"):
        mask = create_mask(pattern="all", elem_bits=32)
        h = vcast(vload(in0, 0), dtypes.float16, mask=mask)
        vstore_pack(res, 0, h, mask, pack_mode=PackMode.B32_TO_B16)

    mem_copy(dst, out.consume())

@host
def run(src, dst):
    _vcast_kernel[1](src, dst)

def main():
    src = (torch.arange(64, dtype=torch.float32, device="npu:0") - 32.0) / 8.0
    dst = torch.empty((64,), dtype=torch.float16, device="npu:0")

    run(src, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), src.cpu().half())
    print("vcast example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vcast example passed
first=-4.0000, last=3.8750
```
