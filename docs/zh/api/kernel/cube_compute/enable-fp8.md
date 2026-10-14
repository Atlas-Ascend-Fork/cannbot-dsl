---
title: enable_fp8
api_name: enable_fp8
category: cube_compute
api_group: kernel
layer: cube
call_context: device
execution_unit: cube
status: experimental
since: 待追溯
---

# `enable_fp8`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

本接口用于设置Mmad计算开启FP8模式，其作用与`enable_hif8`相反，两个接口不同时生效。开启FP8模式后，当矩阵乘加指令的左矩阵A和右矩阵B均以`dtypes.float8_e4m3fn`作为输入数据类型时，L0A Buffer和L0B Buffer中的数据在参与矩阵乘法运算前不会转换为`dtypes.hifloat8`，而是直接以`dtypes.float8_e4m3fn`类型参与计算。

本接口为矩阵计算相关配置接口，仅在AIC上生效。

## 函数原型

```python
def enable_fp8() -> None: ...
```

## 参数说明

无参数。

## 返回值说明

- 无返回值。
## 流水类型

`PIPE_S`

## 约束说明

- 本接口需在矩阵乘加指令（`matmul`）执行前调用，确保模式配置在矩阵乘加结果生成前生效。
- FP8模式启用后会持续生效，后续矩阵乘加指令若不显式重新配置，将沿用当前模式。如需开启HiF8模式，请重新调用`enable_hif8`接口。
- 本接口仅对矩阵乘加输入数据类型为`dtypes.float8_e4m3fn`×`dtypes.float8_e4m3fn`的场景生效，其他FP8数据类型组合（`dtypes.float8_e4m3fn`×`dtypes.float8_e5m2`、`dtypes.float8_e5m2`×`dtypes.float8_e4m3fn`、`dtypes.float8_e5m2`×`dtypes.float8_e5m2`）不支持FP8与HiF8模式选择，调用本接口不产生实际作用。

## 调用示例

将代码保存为`enable_fp8.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, dtypes, host, make_copy_engine, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.cube import enable_fp8
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.tensor import MemLoc

@kernel
def _enable_fp8_kernel(a, b, c):
    l1a = Channel(MemLoc.L1, shape=(16, 16), dtype=dtypes.float8_e4m3fn, depth=1)
    l1b = Channel(MemLoc.L1, shape=(16, 16), dtype=dtypes.float8_e4m3fn, depth=1)
    l0a = Channel(MemLoc.L0A, shape=(16, 16), dtype=dtypes.float8_e4m3fn, depth=1)
    l0b = Channel(MemLoc.L0B, shape=(16, 16), dtype=dtypes.float8_e4m3fn, depth=1)
    l0c = Channel(MemLoc.L0C, shape=(16, 16), dtype=dtypes.float32, depth=1)
    mem_copy(l1a.produce(), a, engine=make_copy_engine(format_transform="nd2nz"))
    mem_copy(l1b.produce(), b, engine=make_copy_engine(format_transform="nd2nz"))
    enable_fp8()
    mem_copy(l0a.produce(), l1a.consume())
    mem_copy(l0b.produce(), l1b.consume())
    matmul(l0c.produce(), l0a.consume(), l0b.consume(), init=True)
    mem_copy(c, l0c.consume())

@host
def run(a, b, c):
    _enable_fp8_kernel[1](a, b, c)

def main():
    # 18.0 对应的fp8_e4m3fn编码为0x59，该编码在HiF8下的数值为5 / 256。
    a = (torch.eye(16, dtype=torch.float32) * 18.0).to(torch.float8_e4m3fn).npu()
    b = a.clone()
    c = torch.empty((16, 16), dtype=torch.float32, device="npu:0")
    run(a, b, c)
    torch.npu.synchronize()

    # 启用FP8模式后，L0A Buffer/L0B Buffer中的fp8_e4m3fn数据不转换为hifloat8，直接参与Mmad计算：
    # 乘加结果为18.0^2 = 324.0；HiF8模式下则为(5 / 256)^2 = 25 / 65536。
    torch.testing.assert_close(c.cpu(), torch.eye(16, dtype=torch.float32) * 324.0)
    print(f"fp8={a.cpu()[0, 0].float().item():.1f} -> matmul={c.cpu()[0, 0].item():.8f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
fp8=18.0 -> matmul=324.00000000
```
