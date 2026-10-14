---
title: 值选择
api_name: value selection
category: control-flow
api_group: kernel
layer: frontend
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# 值选择

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

CANNBotDSL 支持使用条件表达式以及 Python 内建 `min()`、`max()` 产生结果值。条件
表达式根据判断条件选择两个候选结果之一；`min()`、`max()` 则根据候选值的大小返回
最小值或最大值。

## 条件表达式

### 语法形式

```python
result = x if predicate else y
```

`predicate` 为真时结果为 `x`，否则结果为 `y`。

### 参数说明

`predicate` 支持以下形式：

| 可选形式 | 判断方式 |
| --- | --- |
| Python `bool` | 在生成设备代码时直接确定条件是否成立。 |
| 运行时布尔值 | 直接作为运行时条件。 |
| 运行时整数 | 非 `0` 为真，`0` 为假。 |
| 运行时浮点数 | 非 `0.0` 为真，`0.0` 和 `NaN` 为假。 |

`x` 和 `y` 是两个必选的候选结果：

| 结果形式 | 要求 |
| --- | --- |
| 标量 | 类型必须兼容；一侧为 Python 数值字面量时，会转换为与另一侧 DSL 标量兼容的类型。 |
| tuple/list/dict | 容器类型、嵌套方式和元素数量必须一致，dict 的键也必须一致。 |
| Tensor | 可以在两个已经存在且类型和布局兼容的 Tensor 之间选择；选择不会复制 Tensor 数据。 |

### 返回值说明

返回命中表达式产生的值。运行时条件下，两侧结果必须具有相同结构和兼容类型。

## `min()` / `max()`

### 函数原型

```python
min(value1, value2, *values)
min(iterable)

max(value1, value2, *values)
max(iterable)
```

### 参数说明

`min()`、`max()` 支持以下两种调用形式：

| 调用形式 | 参数 | 类型 | 数量 | 说明 |
| --- | --- | --- | --- | --- |
| 传入多个候选值 | `value1`、`value2`、`*values` | Python 数值或 DSL 标量 | 至少 2 个 | `value1`、`value2` 是前两个候选值，之后可以继续传入任意数量的候选值。 |
| 传入一个可迭代对象 | `iterable` | 非空的固定 tuple、list 等可迭代对象 | 只传入 1 个对象 | 该对象中的每个元素都是一个候选值，例如 `min((a, b, c))` 包含 3 个候选值。 |

### 返回值说明

`min()` 返回候选值中的最小值，`max()` 返回候选值中的最大值。包含运行时值时，
返回类型是各操作数协调后的 DSL 标量类型。

## 约束说明

- 条件表达式的两个候选表达式都会参与编译，必须都是合法、可追踪的表达式。
- 不能在条件表达式的候选表达式中创建新的 Tensor，并将其作为结果带出表达式。
- `min()`、`max()` 的运行时形式只支持整数、index 或浮点标量，不接受运行时布尔值。
- `min()`、`max()` 的操作数必须能够协调为同一数值类型；有符号和无符号运行时整数
  不能隐式混用。

## 调用示例

```python
import cannbotdsl as cbd
from cannbotdsl.tensor import Tensor
import torch
import torch_npu  # noqa: F401


@cbd.kernel
def selection_kernel(x: Tensor, out: Tensor, flag):
    out[0] = x[0] if flag != 0 else x[1]
    out[1] = min(x[0], x[1])
    out[2] = max(x[0], x[1])


class SelectionOp:
    @cbd.host
    def run(self, x, out, flag: cbd.dtypes.int64):
        selection_kernel[1](x, out, flag)


def main():
    x = torch.tensor([3, 9], dtype=torch.int64, device="npu:0")
    for flag, expected in ((1, [3, 3, 9]), (0, [9, 3, 9])):
        out = torch.zeros(3, dtype=torch.int64, device="npu:0")
        SelectionOp().run(x, out, flag)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), torch.tensor(expected, dtype=torch.int64))
        print(f"flag={flag} -> {out.cpu().tolist()}")


if __name__ == "__main__":
    main()
```

### 示例输入

```text
x = [3, 9]
flag 依次为 1 和 0。
```

### 预期输出

```text
flag=1 -> [3, 3, 9]
flag=0 -> [9, 3, 9]
```
