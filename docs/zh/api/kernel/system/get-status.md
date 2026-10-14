---
title: get_status
api_name: get_status
category: system
api_group: kernel
layer: system
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `get_status`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

读取当前核上下文中 STATUS 特殊寄存器的 64bit 整体值并按 `int64` 返回。STATUS 是 64bit 状态寄存器，记录核运行过程中各类告警事件的发生情况，各状态位一旦因告警行为置位即保持，直到复位。返回值中的标志位可用于检查数值过大（溢出）、数值过小（下溢）以及非数（NaN）或无穷（INF）输入。

## 函数原型

```python
def get_status() -> Int64: ...
```

## 参数说明

无。

## 返回值说明

返回状态寄存器的原始值。各标志位含义如下。

| bit | 含义 |
| --- | --- |
| 5 | 浮点运算溢出；`int16`、`int32` 向量（SIMD）算术运算溢出也会置位。 |
| 6 | 浮点运算下溢，结果浮点数小于非规格化数能表示的最小值，此时结果为 0。 |
| 7 | 浮点数转换为无符号整数时，输入值为负数。 |
| 8 | 从 L0C 搬运到 UB 时发生溢出，例如 `float32` 转 `float16`、`int32` 转 `float16`。 |
| 9 | 从 L0C 搬运到 UB 时，转换结果的绝对值过小，发生下溢，例如 `float32` 转 `float16`。 |
| 10 | 矩阵（Cube）累加运算溢出（可能是 `float32`、`float16`、`int32`）。 |
| 11 | 矩阵（Cube）累加运算下溢（可能是 `float32`、`float16`）。 |
| 13 | 标量指令输入为 NaN/INF。 |
| 14 | 向量指令输入为 NaN/INF。 |
| 15 | 矩阵（Cube）指令输入为 NaN/INF。 |
| 61 | 数据搬运指令输入为 NaN/INF。 |
| 未列出的 bit | 保留位，取值没有定义。 |
## 流水类型

`PIPE_S`

## 约束说明

- 本接口为只读查询接口，仅读取状态寄存器当前值，可在 Cube Core（AIC）与 Vector Core（AIV）上调用。

## 调用示例

```python
import torch
import torch_npu  # noqa: F401

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_status

@kernel
def _kernel(output):
    output[0] = get_status()

@host
def run(output):
    _kernel[1](output)

def main():
    output = torch.empty(1, dtype=torch.int64, device="npu")
    run(output)
    torch.npu.synchronize()
    status = int(output.cpu()[0])
    assert isinstance(status, int)
    print(f"status: 0x{status & ((1 << 64) - 1):016x}")
    print("get_status example passed")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
status: 0x<16 位十六进制数>
get_status example passed
```
