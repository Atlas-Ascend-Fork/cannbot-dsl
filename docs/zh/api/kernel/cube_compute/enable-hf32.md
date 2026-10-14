---
title: enable_hf32
api_name: enable_hf32
category: cube_compute
api_group: kernel
layer: cube
call_context: device
execution_unit: cube
status: experimental
since: 待追溯
---

# `enable_hf32`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

本接口用于设置Mmad计算开启HF32模式，开启该模式后，Mmad计算FP32数据的性能将得到提升，但会带来一定的精度损失。

是否开启HF32对Mmad计算理论性能的影响见下表：

**表1** HF32对Mmad计算理论性能的影响（`NPU架构版本3510`）

| 接口 | 左矩阵A | 右矩阵B | $cube_m$ | $cube_n$ | $cube_k$ | $k_0$ |
| --- | --- | --- | --- | --- | --- | --- |
| `matmul`（不开启HF32） | `dtypes.float32` | `dtypes.float32` | 16 | 16 | 1 | 8 |
| `matmul`（开启HF32） | `dtypes.float32` | `dtypes.float32` | 16 | 16 | 8 | 8 |

性能计算公式如下：

$$
\begin{gathered}
{ceil_m} = \left\lceil \frac{m}{16} \right\rceil \times 16 \\[12pt]
{ceil_n} = \left\lceil \frac{n}{16} \right\rceil \times 16 \\[12pt]
{ceil_k} = \left\lceil \frac{k}{k_0} \right\rceil \times k_0 \\[16pt]
\text{cube利用率} =
\frac{ (m \times n \times k) / ({cube_m} \times {cube_n} \times {cube_k}) }
{ \Delta t + ({ceil_m} \times {ceil_n} \times {ceil_k}) / ({cube_m} \times {cube_n} \times {cube_k}) }
\end{gathered}
$$

关键变量及常量说明：

- $m, n, k$：mmad入参实际计算的大小。
- $ceil_m, ceil_n, ceil_k$：$m, n, k$ 根据分形大小向上对齐后的值。
- $cube_m, cube_n, cube_k$：硬件真实并行度（单位：elements/cycle）。
- $k_0$：L0 Buffer上最小分形K方向大小。
- $\Delta t$：头开销cycle数。

开启HF32模式后，L0A Buffer/L0B Buffer中的FP32数据将在参与Mmad计算之前被舍入为HF32格式，舍入模式使用`set_hf32_round_mode`接口配置，中间计算使用HF32格式，最终的运算结果仍以FP32格式输出，以保证后续处理的兼容性。

FP32与HF32格式的精度对比如下图所示：

**图1** FP32与HF32格式精度示意图（`NPU架构版本3510`）

![FP32与HF32格式精度示意图（NPU架构版本3510）](../figures/mmad_hf32_950.png)

## 函数原型

```python
def enable_hf32() -> None: ...
```

## 参数说明

无参数。

## 返回值说明

- 无返回值。
## 流水类型

`PIPE_S`

## 约束说明

- 本接口需在矩阵乘加指令（`matmul`）执行前调用，以此来确保模式配置在矩阵乘加计算过程中生效。
- HF32模式启用后会持续生效，不会自动关闭。后续矩阵乘加指令若不显式重新配置，将沿用当前模式。如需关闭HF32模式，请重新调用`set_fp32_mode`接口。
- HF32模式启用后，FP32转换为HF32格式的舍入模式由配套的`set_hf32_round_mode`接口配置，须在本接口之后调用该配套接口配置舍入模式，否则舍入模式沿用上次配置或默认值。
- HF32模式仅对输入矩阵数据类型为`dtypes.float32`的矩阵乘加运算场景生效。

## 调用示例

将代码保存为`enable_hf32.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, dtypes, host, make_copy_engine, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.cube import enable_hf32
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.tensor import MemLoc

# HF32 的尾数为 10 bit，1.0 与 1 + 2^-10 的中点即 1 + 2^-11，恰好落在舍入的平局点上。
TIE = 1.0 + 0.5 / 1024.0

@kernel
def _enable_hf32_kernel(a, b, c):
    l1a = Channel(MemLoc.L1, shape=(16, 16), dtype=dtypes.float32, depth=1)
    l1b = Channel(MemLoc.L1, shape=(16, 16), dtype=dtypes.float32, depth=1)
    l0a = Channel(MemLoc.L0A, shape=(16, 16), dtype=dtypes.float32, depth=1)
    l0b = Channel(MemLoc.L0B, shape=(16, 16), dtype=dtypes.float32, depth=1)
    l0c = Channel(MemLoc.L0C, shape=(16, 16), dtype=dtypes.float32, depth=1)
    mem_copy(l1a.produce(), a, engine=make_copy_engine(format_transform="nd2nz"))
    mem_copy(l1b.produce(), b, engine=make_copy_engine(format_transform="nd2nz"))
    enable_hf32()
    mem_copy(l0a.produce(), l1a.consume())
    mem_copy(l0b.produce(), l1b.consume())
    matmul(l0c.produce(), l0a.consume(), l0b.consume(), init=True)
    mem_copy(c, l0c.consume())

@host
def run(a, b, c):
    _enable_hf32_kernel[1](a, b, c)

def main():
    a = torch.eye(16, dtype=torch.float32, device="npu:0") * TIE
    b = torch.eye(16, dtype=torch.float32, device="npu:0")
    c = torch.empty((16, 16), dtype=torch.float32, device="npu:0")
    run(a, b, c)
    torch.npu.synchronize()

    # 开启HF32模式后，L0A Buffer/L0B Buffer中的float数据先舍入为HF32格式再参与Mmad计算：
    # 未调用set_hf32_round_mode时默认使用NEAREST_EVEN，TIE被舍入为1.0；
    # 未开启HF32模式时不做舍入处理，结果即为TIE本身。
    torch.testing.assert_close(c.cpu(), torch.eye(16, dtype=torch.float32))
    print(f"tie={a.cpu()[0, 0].item():.8f} -> hf32={c.cpu()[0, 0].item():.8f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
tie=1.00048828 -> hf32=1.00000000
```
