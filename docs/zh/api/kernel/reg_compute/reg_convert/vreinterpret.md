---
title: vreinterpret
api_name: vreinterpret
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vreinterpret`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

将寄存器按等宽 dtype 重新解释比特模式：纯编译期类型重标注，零开销、不发射指令、不改变任何比特。

## 函数原型

```python
def vreinterpret(src: RawVReg, dtype) -> RawVReg: ...
```

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `dtype` | 输入 | 目标dtype，位宽须与源一致。 |

## 返回值说明

返回重新解释后的矢量数据寄存器，数据类型与`dtype`一致。

## 约束说明

- 源与目标dtype位宽必须一致，否则编译期报错。
- 本接口不发射指令，为纯编译期类型重标注。
- 本接口需在VF作用域内调用。
- 本接口为纯向量计算；`mem_copy`、同步等非纯操作不得置于同一 `vf` 作用域内。
- 本接口仅在AIV上生效。

## 调用示例

将代码保存为`vreinterpret.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, dtypes, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload, vreinterpret, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vreinterpret_kernel(src, dst):
    buf = Channel(MemLoc.UB, (64,), src.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    mem_copy(buf.produce(), src)

    in0 = buf.consume()
    res = out.produce()
    with vf(mode="simd"):
        mask = full_mask()
        r = vreinterpret(vload(in0, 0), dtypes.float32)
        vstore(res, 0, r, mask)

    mem_copy(dst, out.consume())

@host
def run(src, dst):
    _vreinterpret_kernel[1](src, dst)

def main():
    src = torch.arange(1, 65, dtype=torch.int32, device="npu:0")
    dst = torch.empty(64, dtype=torch.float32, device="npu:0")

    run(src, dst)
    torch.npu.synchronize()

    bits = dst.cpu().view(torch.int32)
    torch.testing.assert_close(bits, src.cpu())
    print("vreinterpret example passed")
    print(f"first={int(bits[0])}, last={int(bits[-1])}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vreinterpret example passed
first=1, last=64
```
