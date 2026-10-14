---
title: ceil_div
api_name: ceil_div
category: types-and-views
api_group: kernel
layer: frontend
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `ceil_div(...)`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明
将 shape 尾部各维分别除以对应的 tile 大小并向上取整，常用于计算各维的迭代次数。

## 函数原型

```python
ceil_div(input: Shape, tiler: Tiler) -> Shape
```

## 参数说明

| 参数 | 类型 | 必选 | 默认值 | 详细说明 |
| --- | --- | --- | --- | --- |
| `input` | `Shape` | 是 | 无 | 被除数，可为标量或多维结构。未与 tiler 配对的前缀维保持不变。 |
| `tiler` | `Tiler` | 是 | 无 | 除数，与 input 尾部相同数量的维度配对。对于每对维度，`a` 表示 `input` 中的维度长度，`b` 表示与之配对的 `tiler` 切块长度，向上取整结果为 `(a + b - 1) // b`。 |

## 返回值说明

返回保持 input 嵌套结构的 Shape；全静态输入在追踪期折叠为 Python 值，动态输入
在 Kernel 执行时计算。

## 约束说明

`tiler` 的维数可以少于或等于 `input`。计算时，`tiler` 从右侧与 `input` 的相同数量维度逐项配对并执行向上取整除法；`input` 中未配对的前部维度保持不变。`tiler` 的维数不能超过 `input`。每个 tile 大小都必须为正数。

## 调用示例

```python
import torch
from torch import as_tensor as from_torch_npu

import cannbotdsl as cbd


@cbd.kernel
def ceil_div_kernel(output: cbd.Tensor):
    result = cbd.ceil_div((10, 6, 7), (3, 4))
    output[0] = result[0]
    output[1] = result[1]
    output[2] = result[2]


@cbd.host
def run(output):
    ceil_div_kernel[1](output)


if __name__ == "__main__":
    output = torch.empty(3, dtype=torch.int64, device="npu")
    run(from_torch_npu(output))
    torch.npu.synchronize()
    print(output.cpu().tolist())
```

### 示例输入

```text
input=(10,6,7), tiler=(3,4)
```

### 预期输出

```text
[10, 2, 2]
```
