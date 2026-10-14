---
title: set_hf32_round_mode
api_name: set_hf32_round_mode
category: cube_compute
api_group: kernel
layer: cube
call_context: device
execution_unit: cube
status: experimental
since: 待追溯
---

# `set_hf32_round_mode`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

本接口用于设置HF32模式舍入方式，使用该接口前需要先使用`enable_hf32`开启HF32模式。

## 函数原型

```python
def set_hf32_round_mode(mode: HF32RoundingMode) -> None: ...
```

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `mode` | 输入 | HF32舍入模式控制入参，`HF32RoundingMode`枚举类型，支持如下2种枚举值：<br>&nbsp;&nbsp;&bull; NEAREST_AWAY：FP32将以向最接近的值舍入，平局时远离零的方式舍入为HF32。<br>&nbsp;&nbsp;&bull; NEAREST_EVEN：FP32将以向最接近的值舍入，平局时向偶数舍入的方式舍入为HF32。 |

## 返回值说明

- 无返回值。
## 流水类型

`PIPE_S`

## 约束说明

- 开启HF32模式后，若不调用本接口，则默认使用`NEAREST_EVEN`代表的舍入模式。
- 本接口需在矩阵乘加指令（`matmul`）执行前调用，以此来确保模式配置在矩阵乘加计算过程中生效。
- 本接口配置的舍入模式仅在HF32模式开启期间生效，需先调用`enable_hf32`开启HF32模式再调用本接口，否则会报错。
- 舍入模式配置后会持续生效，HF32模式关闭后再次开启仍将沿用上次的舍入模式配置，如需切换舍入模式，请重新调用本接口。

## 调用示例

将代码保存为`set_hf32_round_mode.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, dtypes, host, make_copy_engine, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.cube import HF32RoundingMode, enable_hf32, set_hf32_round_mode
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.tensor import MemLoc

# HF32 的尾数为 10 bit，1.0 与 1 + 2^-10 的中点即 1 + 2^-11，恰好落在舍入的平局点上。
TIE = 1.0 + 0.5 / 1024.0

@kernel
def _set_hf32_round_mode_kernel(a, b, c):
    l1a = Channel(MemLoc.L1, shape=(16, 16), dtype=dtypes.float32, depth=1)
    l1b = Channel(MemLoc.L1, shape=(16, 16), dtype=dtypes.float32, depth=1)
    l0a = Channel(MemLoc.L0A, shape=(16, 16), dtype=dtypes.float32, depth=1)
    l0b = Channel(MemLoc.L0B, shape=(16, 16), dtype=dtypes.float32, depth=1)
    l0c = Channel(MemLoc.L0C, shape=(16, 16), dtype=dtypes.float32, depth=1)
    mem_copy(l1a.produce(), a, engine=make_copy_engine(format_transform="nd2nz"))
    mem_copy(l1b.produce(), b, engine=make_copy_engine(format_transform="nd2nz"))
    enable_hf32()
    set_hf32_round_mode(HF32RoundingMode.NEAREST_AWAY)
    mem_copy(l0a.produce(), l1a.consume())
    mem_copy(l0b.produce(), l1b.consume())
    matmul(l0c.produce(), l0a.consume(), l0b.consume(), init=True)
    mem_copy(c, l0c.consume())

@host
def run(a, b, c):
    _set_hf32_round_mode_kernel[1](a, b, c)

def main():
    a = torch.eye(16, dtype=torch.float32, device="npu:0") * TIE
    b = torch.eye(16, dtype=torch.float32, device="npu:0")
    c = torch.empty((16, 16), dtype=torch.float32, device="npu:0")
    run(a, b, c)
    torch.npu.synchronize()

    # NEAREST_AWAY将平局值TIE舍入到远离零的一侧，即1 + 2^-10；若使用NEAREST_EVEN则舍入为1.0。
    torch.testing.assert_close(c.cpu(), torch.eye(16, dtype=torch.float32) * (1.0 + 1.0 / 1024.0))
    print(f"tie={a.cpu()[0, 0].item():.8f} -> hf32={c.cpu()[0, 0].item():.8f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
tie=1.00048828 -> hf32=1.00097656
```
