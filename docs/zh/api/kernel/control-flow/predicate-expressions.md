---
title: 判断条件
api_name: predicate expressions
category: control-flow
api_group: kernel
layer: frontend
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# 判断条件

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

判断条件表示真或假，可供 `if`、`while` 和条件表达式使用。CANNBotDSL 支持使用
`and`、`or`、`not`、链式比较以及 Python 内建 `any()`、`all()` 产生或组合条件。

## `and` / `or` / `not`

```python
p = a and b
q = a or b
r = not a
```

`and` 在左侧为真时继续计算右侧；`or` 在左侧为假时继续计算右侧；`not` 对一个条件
取反。

| 操作数形式 | 判断方式 |
| --- | --- |
| Python `bool` | 直接使用其布尔值。 |
| 运行时布尔值 | 直接作为条件。 |
| 运行时整数 | 非 `0` 为真，`0` 为假。 |
| 运行时浮点数 | 非 `0.0` 为真，`0.0` 和 `NaN` 为假。 |

全部操作数为 Python 值时返回 Python `bool`；包含运行时值时返回运行时布尔值。

## 链式比较

```python
result = a < b <= c < d
```

链式比较支持 `<`、`<=`、`>`、`>=`、`==`、`!=`。例如 `a < b < c` 等价于
`a < b and b < c`，但中间操作数 `b` 只计算一次。在 `a < x[0] < b` 中，框架只
读取一次 `x[0]`，再用同一个结果完成前后两次比较。

参与比较的操作数可以是 Python 数值或 DSL 标量。相邻操作数必须能够比较；整数与
浮点数混合时按 DSL 数值提升规则处理。

## `any()` / `all()`

```python
any(iterable)
all(iterable)
```

`any()` 在任一条件为真时返回真，等价于 `or` 链；`all()` 仅在全部条件为真时返回
真，等价于 `and` 链。`iterable` 必须是固定 tuple 或 list，元素可以是 Python 值
或 DSL 标量条件。

全部元素为 Python 值时返回 Python `bool`；包含运行时值时返回运行时布尔值。

## 约束说明

- 逻辑操作数只能使用 Python 布尔值或可转换为条件的 DSL 标量。
- 链式比较至少包含两个操作数，相邻操作数必须能够协调为可比较的类型。
- 动态 `any()`、`all()` 只支持固定 tuple 或 list。每个元素可以是普通 Python 值，
  或 DSL 运行时布尔、整数、浮点标量；Python 值按 `bool(value)` 判断，运行时布尔值
  直接作为条件，运行时整数和浮点数按是否非零判断。不支持生成器、运行时可迭代
  对象或非标量运行时值。

## 调用示例

```python
import cannbotdsl as cbd
from cannbotdsl.tensor import Tensor
import torch
import torch_npu  # noqa: F401


@cbd.kernel
def predicate_kernel(out: Tensor, a, b):
    both = a != 0 and b != 0
    either = a != 0 or b != 0
    first_is_zero = not (a != 0)

    out[0] = 1 if both else 0
    out[1] = 1 if either else 0
    out[2] = 1 if first_is_zero else 0
    out[3] = 1 if 0 < a < 10 else 0
    out[4] = 1 if any((a != 0, b != 0)) else 0
    out[5] = 1 if all((a != 0, b != 0)) else 0


class PredicateOp:
    @cbd.host
    def run(self, out, a: cbd.dtypes.int64, b: cbd.dtypes.int64):
        predicate_kernel[1](out, a, b)


def main():
    cases = (
        (1, 0, [0, 1, 0, 1, 1, 0]),
        (5, 2, [1, 1, 0, 1, 1, 1]),
        (0, 0, [0, 0, 1, 0, 0, 0]),
    )
    for a, b, expected in cases:
        out = torch.zeros(6, dtype=torch.int64, device="npu:0")
        PredicateOp().run(out, a, b)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), torch.tensor(expected, dtype=torch.int64))
        print(f"a={a}, b={b} -> {out.cpu().tolist()}")


if __name__ == "__main__":
    main()
```

### 示例输入

```text
(a, b) 依次为 (1, 0)、(5, 2)、(0, 0)。
输出依次表示 and、or、not、0 < a < 10、any、all 的判断结果。
```

### 预期输出

```text
a=1, b=0 -> [0, 1, 0, 1, 1, 0]
a=5, b=2 -> [1, 1, 0, 1, 1, 1]
a=0, b=0 -> [0, 0, 1, 0, 0, 0]
```
