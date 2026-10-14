---
title: matmul
api_name: matmul
category: cube_compute
api_group: kernel
layer: cube
call_context: device
execution_unit: cube
status: experimental
since: 待追溯
---

# `matmul`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

本接口是面向昇腾AI芯片的矩阵乘加（Mmad）核心计算接口，专为高性能算子开发设计，封装了昇腾NPU硬件的矩阵乘加计算能力，广泛用于神经网络层（如全连接层、卷积层）、数值计算类算子的开发。其计算公式如下：

$$
C_{M \times N} = A_{M \times K} \times B_{K \times N} + C_{M \times N}
$$

其中，A、B、C分别为左、右、结果矩阵，C矩阵可以通过配置本接口的参数，初始化为全0矩阵、L0C Buffer中的矩阵或Bias矩阵，各矩阵的信息说明见下表：

**表** 矩阵信息说明（NPU架构版本3510）

| 矩阵 | 存储位置 | 形状（行数×列数） | 数据格式 | 分形大小（行数×列数） |
| --- | --- | --- | --- | --- |
| A | L0A Buffer | M×K | Nz | 16×K0 |
| B | L0B Buffer | K×N | Zn | K0×16 |
| C | L0C Buffer | M×N | Nz | 16×16 |
| Bias（用于C矩阵初始化） | BiasTable Buffer | 1×N，使用时通过广播复制M行来初始化C矩阵 | ND | - |

表格中K0的取值为`32B / sizeof(dtype)`，`dtype`为矩阵的数据类型。

本接口为矩阵计算接口，仅在AIC上生效。

## 函数原型

```python
def matmul(dst: Tensor, lhs: Tensor, rhs: Tensor, *, bias: Tensor | None=None, init=True, unit_flag: int=0, disable_gemv: bool=True) -> None: ...
```

**支持的数据类型：**

`lhs`、`rhs`、`dst`、`bias`的数据类型取值组合见下表：

**表** 支持的数据类型组合（NPU架构版本3510）

| lhs | rhs | dst | bias |
| --- | --- | --- | --- |
| `dtypes.int8` | `dtypes.int8` | `dtypes.int32` | `dtypes.int32` |
| `dtypes.hifloat8` | `dtypes.hifloat8` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.float8_e5m2` | `dtypes.float8_e5m2` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.float8_e5m2` | `dtypes.float8_e4m3fn` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.float8_e4m3fn` | `dtypes.float8_e5m2` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.float8_e4m3fn` | `dtypes.float8_e4m3fn` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.float16` | `dtypes.float16` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.bfloat16` | `dtypes.bfloat16` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.float32` | `dtypes.float32` | `dtypes.float32` | `dtypes.float32` |

## 参数说明

**表** 参数说明（NPU架构版本3510）

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `dst` | 输出 | 目的操作数，结果矩阵C在L0C Buffer中的起始地址，需按照1024字节对齐。数据类型取值请参见[支持的数据类型组合](#函数原型)。 |
| `lhs` | 输入 | 源操作数，左矩阵A在L0A Buffer中的起始地址，需按照512字节对齐。数据类型取值请参见[支持的数据类型组合](#函数原型)。 |
| `rhs` | 输入 | 源操作数，右矩阵B在L0B Buffer中的起始地址，需按照512字节对齐。数据类型取值请参见[支持的数据类型组合](#函数原型)。 |
| `bias` | 输入 | Bias矩阵在BiasTable Buffer中的起始地址，需按照64字节对齐，可以为非零地址。<br>Bias的数据类型需与C矩阵的数据类型保持一致，调用本接口前，需将Bias数据通过`mem_copy`接口从L1 Buffer搬运到该地址，Bias占用长度为`N × dst的数据类型字节宽度`向上补齐到64字节。 |
| `init` | 输入 | 配置是否将矩阵C的初始值设置为0。<br>&nbsp;&nbsp;&bull; true：将矩阵C的初始值设置为0。<br>&nbsp;&nbsp;&bull; false：不执行清零操作，矩阵C的初始值来源于L0C Buffer或BiasTable Buffer。 |
| `unit_flag` | 输入 | 用于控制矩阵乘加指令与矩阵搬出指令的细粒度并行，开启UnitFlag后，硬件每计算完一个分形，计算结果就会被搬出。取值说明如下：<br>&nbsp;&nbsp;&bull; `0`（DISABLE）：不开启UnitFlag。<br>&nbsp;&nbsp;&bull; `2`（ENABLE_KEEP）：开启UnitFlag，硬件执行完指令后不改变单元标志位。<br>&nbsp;&nbsp;&bull; `3`（ENABLE_UPDATE）：开启UnitFlag，硬件执行完指令后改变单元标志位。<br>矩阵乘加指令与对应的矩阵搬出指令必须都开启或都不开启UnitFlag，开启后指令之间无需再插入同步指令。 |
| `disable_gemv` | 输入 | M为1时，配置是否关闭GEMV模式。<br>&nbsp;&nbsp;&bull; false：开启GEMV模式。<br>&nbsp;&nbsp;&bull; true：关闭GEMV模式。<br>M不为1时，该参数不生效。 |

## 返回值说明

- 无返回值。
## 流水类型

`PIPE_M`

## 约束说明

- 本接口仅在AIC上生效。
- `M`、`K`、`N`中的任意一个值为0时，接口将被视为NOP（空操作）。

- 内存使用约束说明：
  - 针对Ascend 950PR/Ascend 950DT:

      - L0C Buffer大小为256KB，L0A Buffer和L0B Buffer大小均为64KB。BiasTable Buffer大小为4KB。矩阵的起始地址和占用空间不能超出对应Buffer的范围。
      - 各矩阵的起始地址需满足[参数说明](#参数说明)中的对齐要求。
      - 申请矩阵存储空间时，需使用按照分形大小补齐后的数值进行申请：M、N分别向上补齐到16的倍数，K向上补齐到K0的倍数，K0的取值为`32B / sizeof(dtype)`，`dtype`为矩阵的数据类型。`lhs`、`rhs`、`dst`的shape即为矩阵的有效M、K、N值，补齐部分为无效数据，不参与结果矩阵有效区域的计算。
      - M、K、N的单位为元素，取值范围均为[0, 4095]。
      - 当M为1且`disable_gemv`为false时，将开启GEMV模式。此时从L0A Buffer读取矩阵A时按照ND格式读取，矩阵A需按照ND格式排布，起始地址仍需按照512字节对齐。

- 同步约束说明：

  在NPU架构版本3510上，将结果累加到同一块L0C Buffer的相邻两次矩阵乘加指令之间无需额外插入同步指令。如需显式同步Cube流水线，可调用`cube_sync_pipe`并将入参`pipe`设置为`PIPE.M`。

- UnitFlag约束说明：

  - 开启UnitFlag时，矩阵乘加指令与对应矩阵搬出指令需同时开启UnitFlag。当希望同一块L0C Buffer内存空间能持续只被多条矩阵乘加指令或多条矩阵搬出指令操作时，除最后一条外的指令需将`unit_flag`设置为`2`（ENABLE_KEEP），维持被操作内存空间的持续占用状态，最后一条指令设置为`3`（ENABLE_UPDATE），解除被占用状态。
  - 开启UnitFlag时，矩阵计算方向需与矩阵搬出读取顺序保持一致。矩阵搬出指令开启Nz2ND随路格式转换，或未进行随路格式转换但开启B8/B4量化并触发Channel Merge功能时，调用`cube.set_mmad_direction("n")`；其他场景调用`cube.set_mmad_direction("m")`。
  - 开启UnitFlag时，建议矩阵乘加的计算数据量与矩阵搬出的数据量保持一致。两者不一致可能导致执行异常。

- 特殊值/边界值约束说明：

  注意，应避免nan输入，否则可能会产生执行报错；整数类型仅支持饱和模式。

## 调用示例

将代码保存为`matmul.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import Channel, dtypes, host, make_copy_engine, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.tensor import MemLoc

@kernel
def _matmul_kernel(a, b, c):
    l1a = Channel(MemLoc.L1, shape=(16, 16), dtype=dtypes.float16, depth=1)
    l1b = Channel(MemLoc.L1, shape=(16, 16), dtype=dtypes.float16, depth=1)
    l0a = Channel(MemLoc.L0A, shape=(16, 16), dtype=dtypes.float16, depth=1)
    l0b = Channel(MemLoc.L0B, shape=(16, 16), dtype=dtypes.float16, depth=1)
    l0c = Channel(MemLoc.L0C, shape=(16, 16), dtype=dtypes.float32, depth=1)
    mem_copy(l1a.produce(), a, engine=make_copy_engine(format_transform="nd2nz"))
    mem_copy(l1b.produce(), b, engine=make_copy_engine(format_transform="nd2nz"))
    mem_copy(l0a.produce(), l1a.consume())
    mem_copy(l0b.produce(), l1b.consume())
    matmul(l0c.produce(), l0a.consume(), l0b.consume(), init=True)
    mem_copy(c, l0c.consume())

@host
def run(a, b, c):
    _matmul_kernel[1](a, b, c)

def main():
    a = torch.arange(256, dtype=torch.float16, device="npu:0").reshape(16, 16) / 16.0
    b = a.clone()
    c = torch.empty((16, 16), dtype=torch.float32, device="npu:0")
    run(a, b, c)
    torch.npu.synchronize()

    torch.testing.assert_close(c.cpu(), a.cpu().float() @ b.cpu().float().T, rtol=1e-2, atol=1e-2)
    print(f"first={c.cpu()[0, 0].item():.3f}, last={c.cpu()[-1, -1].item():.3f}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
first=4.844, last=3829.844
```
