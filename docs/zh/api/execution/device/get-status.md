# get_status
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

读取当前核上下文中STATUS特殊寄存器的64bit整体值并按 `int64` 返回。STATUS是64bit状态寄存器，记录核运行过程中各类告警事件的发生情况，各状态位一旦因告警行为置位即保持，直到复位。返回值中的标志位可用于检查数值过大（溢出）、数值过小（下溢）以及非数（NaN）或无穷（INF）输入。

## 函数原型

```python
get_status() -> Int64
```

## 参数说明

无。

## 返回值说明

返回状态寄存器的原始值，返回类型为 `Int64`。各标志位含义如下。

| bit | 含义 |
| --- | --- |
| 5 | 浮点运算溢出；`int16`、`int32` 向量（SIMD）算术运算溢出也会置位。 |
| 6 | 浮点运算下溢，结果浮点数小于非规格化数能表示的最小值，此时结果为0。 |
| 7 | 浮点数转换为无符号整数时，输入值为负数。 |
| 8 | 从L0C搬运到UB时发生溢出，例如 `float32` 转 `float16`、`int32` 转 `float16`。 |
| 9 | 从L0C搬运到UB时，转换结果的绝对值过小，发生下溢，例如 `float32` 转 `float16`。 |
| 10 | 矩阵（Cube）累加运算溢出（可能是 `float32`、`float16`、`int32`）。 |
| 11 | 矩阵（Cube）累加运算下溢（可能是 `float32`、`float16`）。 |
| 13 | 标量指令输入为NaN/INF。 |
| 14 | 向量指令输入为NaN/INF。 |
| 15 | 矩阵（Cube）指令输入为NaN/INF。 |
| 61 | 数据搬运指令输入为NaN/INF。 |
| 未列出的bit | 保留位，取值没有定义。 |

## 约束说明

无。

## 调用示例

以下示例读取STATUS寄存器，并以十六进制格式打印结果。

```python
import torch
import torch_npu

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
    print(f"status: 0x{status & ((1 << 64) - 1):016x}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
status: 0x<16位十六进制数>
```
