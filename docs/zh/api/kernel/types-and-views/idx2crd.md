---
title: idx2crd
api_name: idx2crd
category: types-and-views
api_group: kernel
layer: frontend
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `idx2crd(...)`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

根据 `shape`，把从 0 开始的线性元素索引转换为多维坐标。元素按照行主序编号，
即最后一维的坐标最先连续变化。

例如，`shape=(5, 4)` 表示 5 行、每行 4 个元素。线性索引 `11` 位于第 2 行、
第 3 列，因此转换结果为 `(2, 3)`。

## 函数原型

```python
idx2crd(idx, shape: Shape) -> Coord
```

## 参数说明

| 参数 | 类型 | 必选 | 默认值 | 详细说明 |
| --- | --- | --- | --- | --- |
| `idx` | Python `int` 或 Kernel 内的整数值 | 是 | 无 | 待转换的线性元素索引，从 0 开始计数。Kernel 内的整数值可以来自 Kernel 标量参数、循环索引或整数计算结果。 |
| `shape` | `Shape` | 是 | 无 | 用于确定各维坐标范围的形状。普通 tuple 返回普通多维坐标；嵌套 tuple 返回相同嵌套结构的坐标。shape 中的维度可以是 Python `int`，也可以是 Kernel 内已有的整数值。 |

## 返回值说明

返回 `idx` 在 `shape` 中对应的多维坐标。对于普通多维
`shape=(s0, s1, ..., sn)`，第 `i` 维坐标按以下方式计算：

```text
coord[i] = (idx // 后续各维长度的乘积) % shape[i]
```

最后一维之后没有其他维度，其“后续各维长度的乘积”按 1 计算。例如
`idx=11`、`shape=(5, 4)` 时：

```text
coord[0] = (11 // 4) % 5 = 2
coord[1] = (11 // 1) % 4 = 3
```

因此返回 `(2, 3)`。

返回坐标的结构与 `shape` 一致。例如：

| `shape` | `idx` | 返回值 |
| --- | --- | --- |
| `8` | `3` | `3` |
| `(5, 4)` | `11` | `(2, 3)` |
| `((2, 3), 4)` | `11` | `((0, 2), 3)` |

当 `idx` 和 `shape` 都由 Python `int` 组成时，返回 Python `int` 或 tuple，并在
编译阶段完成计算。只要其中包含 Kernel 内的整数值，就返回可在 Kernel 中继续用于
索引、计算或拆分的坐标值。

## 约束说明

- `shape` 中的每个维度长度必须是正整数。
- 为得到 `shape` 范围内有效且唯一的坐标，`idx` 应满足
  `0 <= idx < shape 各维长度的乘积`；该函数不负责检查 Tensor 是否越界。
- 计算时会先按从左到右的顺序展开嵌套 shape，按行主序计算各维坐标，再将结果恢复
  为与 shape 相同的嵌套结构。
- Kernel 内的动态值必须是整数值，例如 Kernel 标量参数、循环索引或整数计算结果。

## 调用示例

```python
import torch
from torch import as_tensor as from_torch_npu

import cannbotdsl as cbd


@cbd.kernel
def idx2crd_kernel(output: cbd.Tensor):
    coord = cbd.idx2crd(11, (5, 4))
    output[0] = coord[0]
    output[1] = coord[1]


@cbd.host
def run(output):
    idx2crd_kernel[1](output)


if __name__ == "__main__":
    output = torch.empty(2, dtype=torch.int64, device="npu")
    run(from_torch_npu(output))
    torch.npu.synchronize()
    print(output.cpu().tolist())
```

### 示例输入

```text
idx=11, shape=(5,4)
```

### 预期输出

```text
[2, 3]
```
