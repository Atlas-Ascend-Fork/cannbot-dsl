---
title: vreduce_max_datablock
api_name: vreduce_max_datablock
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vreduce_max_datablock`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

根据`mask`对每个`DataBlock`（32B）内的元素求最大值，得到各`DataBlock`的归约结果。结果依次保存在返回值的低位。计算公式如下：

$$
dst_k = \max\{src_i \mid kB \le i < (k + 1)B,\ mask_i = 1\}
$$

其中，$k$表示`DataBlock`的索引，$B$为一个`DataBlock`内的元素个数。

本接口仅在AIV上生效。

## 函数原型

```python
def vreduce_max_datablock(src: RawVReg, *, mask: Mask) -> RawVReg: ...
```

**支持的数据类型：**

`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 掩码寄存器，用于控制各元素是否参与归约。 |

## 返回值说明

- 通过函数返回值返回结果：返回归约结果，类型为矢量数据寄存器，与`src`的数据类型一致。

## 约束说明

- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- 比较时遵循$max(-0, +0) = +0$。
- 返回值与`src`的数据类型需要保持一致。
- 每个`DataBlock`内的最大值连续写入返回值的前8个元素，这8个元素为有效输出，返回值中的其他元素置0。
- 未被`mask`选中的元素被视为对应数据类型的最小值，浮点数类型的最小值为`-inf`。如果一个`DataBlock`中的所有元素均未被`mask`选中，则将该最小值写入返回值的对应位置。
- 仅输出最大值，不输出索引。

## 调用示例

将代码保存为`vreduce_max_datablock.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload, vreduce_max_datablock, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vreduce_max_datablock_kernel(src0, dst):
    buf = Channel(MemLoc.UB, (64,), src0.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)
    mem_copy(buf.produce(), src0)

    in0 = buf.consume()
    res = out.produce()
    with vf(mode="simd"):
        full = full_mask()
        acc = vreduce_max_datablock(vload(in0, 0), mask=full)
        vstore(res, 0, acc, full)

    mem_copy(dst, out.consume())

@host
def run(src0, dst):
    _vreduce_max_datablock_kernel[1](src0, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, dst)
    torch.npu.synchronize()

    expected = torch.zeros(64)
    expected[:8] = src0.cpu().view(8, 8).max(dim=1).values
    torch.testing.assert_close(dst.cpu(), expected)
    print(f"first={float(dst.cpu()[0]):.4f}, block7={float(dst.cpu()[7]):.4f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
first=7.0000, block7=63.0000
```
