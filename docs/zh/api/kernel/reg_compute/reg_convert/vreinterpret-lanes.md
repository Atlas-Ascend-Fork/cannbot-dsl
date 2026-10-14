---
title: vreinterpret_lanes
api_name: vreinterpret_lanes
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vreinterpret_lanes`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

将寄存器的 256 字节载荷按不同 lane 宽度重新解释（如 b32 → 256 个 b8 lane），纯编译期重标注，不改变任何比特。

## 函数原型

```python
def vreinterpret_lanes(src: RawVReg, dtype) -> RawVReg: ...
```

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `dtype` | 输入 | 目标lane dtype。 |

## 返回值说明

返回重新解释后的矢量数据寄存器，lane 数按新宽度换算。

## 约束说明

- 载荷总字节数保持 256B，仅 lane 划分变化。
- 本接口需在VF作用域内调用。
- 本接口为纯向量计算；`mem_copy`、同步等非纯操作不得置于同一 `vf` 作用域内。
- 本接口仅在AIV上生效。

## 调用示例

将代码保存为`vreinterpret_lanes.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, dtypes, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import create_mask, vload, vreinterpret_lanes, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vreinterpret_lanes_kernel(src, dst):
    buf = Channel(MemLoc.UB, (64,), src.dtype, depth=1)
    out = Channel(MemLoc.UB, (256,), dst.dtype, depth=1)

    mem_copy(buf.produce(), src)

    in0 = buf.consume()
    res = out.produce()
    with vf(mode="simd"):
        mask = create_mask(pattern="all", elem_bits=8)
        r = vreinterpret_lanes(vload(in0, 0), dtypes.uint8)
        vstore(res, 0, r, mask)

    mem_copy(dst, out.consume())

@host
def run(src, dst):
    _vreinterpret_lanes_kernel[1](src, dst)

def main():
    src = torch.arange(1, 65, dtype=torch.int32, device="npu:0")
    dst = torch.empty(256, dtype=torch.uint8, device="npu:0")

    run(src, dst)
    torch.npu.synchronize()

    bytes_out = dst.cpu()
    torch.testing.assert_close(bytes_out, src.cpu().view(torch.uint8))
    print("vreinterpret_lanes example passed")
    print(f"first={int(bytes_out[0])}, last={int(bytes_out[-1])}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vreinterpret_lanes example passed
first=1, last=0
```
