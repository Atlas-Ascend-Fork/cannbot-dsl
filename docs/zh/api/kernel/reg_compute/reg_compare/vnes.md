---
title: vnes
api_name: vnes
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vnes`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

根据`mask`将源操作数`src`按元素与标量`scalar`进行比较，若$src_i \neq scalar$，则对应结果位为1，否则为0。每个元素的比较结果占一个比特。计算公式如下：

$$
dst_i =
\begin{cases}
 1, & src_i \neq scalar \\
 0, & src_i = scalar \\
\end{cases}
$$

本接口仅在AIV上生效。

## 函数原型

```python
def vnes(src: RawVReg, scalar, *, mask: Mask) -> Mask: ...
```

**支持的数据类型：**

`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `scalar` | 输入 | 源操作数（标量）。 |
| `mask` | 输入 | 掩码寄存器，用于控制各元素是否参与比较。`mask`中与元素对应的比特位为1时，该元素参与比较；为0时，该元素不参与比较。 |

## 返回值说明

- 返回比较结果，类型为掩码寄存器。`mask`掩码位为0的元素在返回值中对应比特位置0。

## 约束说明

- `mask`需通过掩码设置接口预先赋值后再传入，未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
- `src`与`scalar`的数据类型需要保持一致。
- `mask`掩码位为0时，结果对应比特位置0。
- 浮点比较输入包含nan时，结果对应比特位写1。

## 调用示例

将代码保存为`vnes.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, dtypes, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vdups, vload, vnes, vselect, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vnes_kernel(src, dst):
    buf = Channel(MemLoc.UB, (64,), src.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)
    mem_copy(buf.produce(), src)

    in0, res = buf.consume(), out.produce()
    with vf(mode="simd"):
        mask = full_mask()
        pred = vnes(vload(in0, 0), 32.0, mask=mask)
        r = vselect(vdups(1.0, dtypes.float32),
                           vdups(0.0, dtypes.float32), cond_mask=pred)
        vstore(res, 0, r, mask)
    mem_copy(dst, out.consume())

@host
def run(src, dst):
    _vnes_kernel[1](src, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 != 32.0).float().cpu())
    print(f"count={int(dst.sum().item())}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
count=63
```
