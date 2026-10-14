---
title: for
api_name: for, range, range_constexpr
category: control-flow
api_group: kernel
layer: frontend
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# for

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

`for` 按给定范围或固定序列重复执行循环体。迭代范围可以使用 Python 内建
`builtins.range()`、CANNBotDSL 提供的 `cannbotdsl.range()` 或
`range_constexpr()`，也可以使用固定 tuple/list。

## 语法形式

```python
for i in iterable:
    ...

for i in iterable:
    ...
else:
    ...
```

## 迭代范围

### `builtins.range()`

```python
builtins.range(stop)
builtins.range(start, stop)
builtins.range(start, stop, step)
```

在 `for` 中使用时建立设备执行的计数循环，不提供循环展开参数。

### `cannbotdsl.range()`

```python
cannbotdsl.range(stop)
cannbotdsl.range(start, stop)
cannbotdsl.range(start, stop, step)
```

| 参数 | 类型 | 必选 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `start` | Python `int` 或运行时整数 | 否 | `0` | 循环起点。 |
| `stop` | Python `int` 或运行时整数 | 是 | 无 | 循环终点，不包含该值。 |
| `step` | Python `int` 或运行时整数 | 否 | `1` | 每次迭代的递增量，必须为正值。 |

`start`、`stop` 和 `step` 可以是 Python 整数或 Kernel 运行时整数。多个运行时整数
必须同为有符号或同为无符号，位宽不同时提升为最宽类型。

### `cannbotdsl.range(..., unroll=N)`

```python
cannbotdsl.range(stop, *, unroll=N)
cannbotdsl.range(start, stop, *, unroll=N)
cannbotdsl.range(start, stop, step, *, unroll=N)
```

`start`、`stop` 和 `step` 的含义与 `cannbotdsl.range()` 相同。`unroll` 为必选的
关键字参数，`N` 必须是编译时已知的正 Python `int`。

`unroll=N` 向后续编译阶段提供正整数展开因子 `N`，用于按每组 `N` 次迭代展开
循环体，由后续编译阶段根据循环边界和目标能力处理。

### `cannbotdsl.range(..., unroll_full=True)`

```python
cannbotdsl.range(stop, *, unroll_full=True)
cannbotdsl.range(start, stop, *, unroll_full=True)
cannbotdsl.range(start, stop, step, *, unroll_full=True)
```

`start`、`stop` 和 `step` 的含义与 `cannbotdsl.range()` 相同。`unroll_full` 为必选的
关键字参数，必须显式设为 Python `True`，并且不能同时指定 `unroll`。

`unroll_full=True` 向后续编译阶段提供完全展开设置。后续编译阶段根据循环边界和
目标能力处理该设置；文档不承诺最终代码中一定完全消除循环。

### `range_constexpr()`

```python
range_constexpr(stop)
range_constexpr(start, stop)
range_constexpr(start, stop, step)
```

`range_constexpr()` 接受一至三个编译时 Python 整数，范围规则与 Python `range`
相同。包含它的 `for` 在生成设备代码时展开循环体，不保留设备循环。

展开次数会直接增加生成代码的体积，适合较小且固定的迭代范围。

### 固定 tuple/list

直接迭代固定 tuple/list 时，前端按固定元素展开循环。元素为 tuple/list 时可以使用
Python 的循环目标解包写法：

```python
for left, right in ((1, 2), (3, 4)):
    ...
```

## `for ... else`

```python
for i in builtins.range(stop):
    ...
else:
    ...
```

当前设备循环不支持 `break`，因此 `for ... else` 不能表达 Python 中“通过 `break`
跳过 `else`”的用法。设备循环结束后会继续执行 `else` 语句块。

## 约束说明

- `cannbotdsl.range()` 是供 CANNBotDSL 前端识别的 `for` 循环范围标记，只能直接用于
  `@host`、`@jit` 或 `@kernel` 函数中的 `for ... in cannbotdsl.range(...)`。它不会像
  `builtins.range()` 一样返回可由普通 Python 代码遍历的 `range` 对象。
- `unroll` 与 `unroll_full=True` 不能同时使用。
- 设备循环体不能使用 `break`、`continue`、`return` 或 `raise` 提前结束或跳出循环。
- 设备循环体内不支持推导式、嵌套函数或类、`lambda`、`try`、`yield`、异步语法、
  `del` 和 `match`。
- `range_constexpr()` 和固定 tuple/list 在生成设备代码时展开，`break`、`continue`
  按编译时执行的 Python 循环语义处理。

## 调用示例

```python
import builtins
import cannbotdsl as cbd
from cannbotdsl.tensor import Tensor
import torch
import torch_npu  # noqa: F401


@cbd.kernel
def for_kernel(out: Tensor):
    total = 0
    for i in builtins.range(4):
        total = total + i
    out[0] = total

    total = 0
    for i in cbd.range(4):
        total = total + i
    out[1] = total

    total = 0
    for i in cbd.range(8, unroll=4):
        total = total + i
    out[2] = total

    total = 0
    for i in cbd.range(4, unroll_full=True):
        total = total + i
    out[3] = total

    total = 0
    for i in cbd.range_constexpr(4):
        total = total + i
    out[4] = total

    pair_total = 0
    for left, right in ((1, 2), (3, 4)):
        pair_total = pair_total + left + right
    out[5] = pair_total

    total = 0
    for i in builtins.range(4):
        total = total + i
    else:
        total = total + 100
    out[6] = total


@cbd.host
def run(out):
    for_kernel[1](out)


out = torch.zeros(7, dtype=torch.int64, device="npu:0")
run(out)
torch.npu.synchronize()
print(out.cpu().tolist())
```

### 示例输入

```text
builtins.range(4)：0、1、2、3
cannbotdsl.range(4)：0、1、2、3
cannbotdsl.range(8, unroll=4)：0 至 7
cannbotdsl.range(4, unroll_full=True)：0、1、2、3
range_constexpr(4)：0、1、2、3
固定 tuple：元素为 (1, 2) 和 (3, 4)
for ... else：循环范围为 0、1、2、3
```

### 预期输出

```text
[6, 6, 28, 6, 6, 10, 106]
```
