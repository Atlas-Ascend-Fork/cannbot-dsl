---
title: set_fp32_mode
api_name: set_fp32_mode
category: cube_compute
api_group: kernel
layer: cube
call_context: device
execution_unit: cube
status: experimental
since: 待追溯
---

# `set_fp32_mode`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

本接口用于设置Mmad计算关闭HF32模式，其作用与`enable_hf32`相反，两个接口不同时生效。关闭HF32模式后，L0A Buffer与L0B Buffer中的`dtypes.float32`数据在参与Mmad计算之前不做舍入处理。

本接口为矩阵计算相关配置接口，仅在AIC上生效。

## 函数原型

```python
def set_fp32_mode() -> None: ...
```

## 参数说明

无参数。

## 返回值说明

- 无返回值。
## 流水类型

`PIPE_S`

## 约束说明

- 本接口非AIC调用直接返回。
- 本接口需在矩阵乘加指令（`matmul`）执行前调用，以此来确保模式配置在矩阵乘加计算过程中生效。
- 与`enable_hf32`作用相反，二者不同时生效。
- FP32模式启用后会持续生效，不会自动关闭。后续矩阵乘加指令若不显式重新配置，将沿用当前模式。如需开启HF32模式，请重新调用`enable_hf32`接口。

## 调用示例

将代码保存为`set_fp32_mode.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, dtypes, host, make_copy_engine, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.cube import enable_hf32, set_fp32_mode
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.tensor import MemLoc

# HF32 的尾数为 10 bit，1.0 与 1 + 2^-10 的中点即 1 + 2^-11，恰好落在舍入的平局点上。
TIE = 1.0 + 0.5 / 1024.0

@kernel
def _set_fp32_mode_kernel(a, b, c):
    l1a = Channel(MemLoc.L1, shape=(16, 16), dtype=dtypes.float32, depth=1)
    l1b = Channel(MemLoc.L1, shape=(16, 16), dtype=dtypes.float32, depth=1)
    l0a = Channel(MemLoc.L0A, shape=(16, 16), dtype=dtypes.float32, depth=1)
    l0b = Channel(MemLoc.L0B, shape=(16, 16), dtype=dtypes.float32, depth=1)
    l0c = Channel(MemLoc.L0C, shape=(16, 16), dtype=dtypes.float32, depth=1)
    mem_copy(l1a.produce(), a, engine=make_copy_engine(format_transform="nd2nz"))
    mem_copy(l1b.produce(), b, engine=make_copy_engine(format_transform="nd2nz"))
    enable_hf32()
    set_fp32_mode()
    mem_copy(l0a.produce(), l1a.consume())
    mem_copy(l0b.produce(), l1b.consume())
    matmul(l0c.produce(), l0a.consume(), l0b.consume(), init=True)
    mem_copy(c, l0c.consume())

@host
def run(a, b, c):
    _set_fp32_mode_kernel[1](a, b, c)

def main():
    a = torch.eye(16, dtype=torch.float32, device="npu:0") * TIE
    b = torch.eye(16, dtype=torch.float32, device="npu:0")
    c = torch.empty((16, 16), dtype=torch.float32, device="npu:0")
    run(a, b, c)
    torch.npu.synchronize()

    # 先开启HF32模式再关闭，L0A Buffer/L0B Buffer中的float数据不再做舍入处理，
    # 结果即为TIE本身；若关闭未生效，TIE将被舍入为1.0。
    torch.testing.assert_close(c.cpu(), torch.eye(16, dtype=torch.float32) * TIE)
    print(f"tie={a.cpu()[0, 0].item():.8f} -> fp32={c.cpu()[0, 0].item():.8f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
tie=1.00048828 -> fp32=1.00048828
```
