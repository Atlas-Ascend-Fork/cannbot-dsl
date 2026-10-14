---
title: 控制流
api_name: control flow
category: control-flow
api_group: kernel
layer: frontend
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# 控制流

## 简介

CANNBotDSL 支持在 `@jit` 和 `@kernel` 函数中使用部分 Python 控制流语法，并提供
`cannbotdsl.range()`、`range_constexpr()`、`const_expr()` 和
`target_version()` 等 DSL API。编译函数时，CANNBotDSL 会识别这些语法和调用，
根据条件或迭代过程能否在编译时确定，选择在生成设备代码时直接处理，或者保留为
设备执行时的控制流。

| 来源 | 支持的写法 |
| --- | --- |
| Python 语法 | `if` / `elif` / `else`、条件表达式、`and` / `or` / `not`、链式比较、`for`、`while`、`for ... else`、`while ... else` |
| Python 内建函数 | `min()`、`max()`、`any()`、`all()`、`builtins.range()` |
| CANNBotDSL API | `cannbotdsl.range()`、`range_constexpr()`、`const_expr()`、`target_version()` |

## 工作方式

CANNBotDSL 在编译 `@jit`、`@kernel` 函数时读取函数的 Python 语法结构，并识别其中
受支持的分支、循环、逻辑表达式和函数调用。

```text
Python 控制流或表达式
          │
          ▼
CANNBotDSL 前端识别
          │
          ├─ 条件或迭代过程能在编译时确定
          │       └─ 生成设备代码时选择、计算或展开
          │
          └─ 依赖 Kernel 运行时数据
                  └─ 保留为设备执行时的分支或循环
```

条件结果或迭代过程能够在编译时确定时，前端在生成设备代码的过程中完成条件选择、
表达式计算或循环展开。条件或循环边界依赖 Tensor 读取值、Kernel 标量形参等运行时
数据时，设备执行 Kernel 时再完成判断或迭代。

在普通 Python 代码中，上述 Python 语法和内建函数仍由 Python 解释器按普通 Python
规则执行。

## 编译与执行

仅包含普通 Python 值的条件或表达式在生成设备代码时按 Python 规则求值；包含 Tensor
读取值、Kernel 标量形参等运行时数据时，由设备执行 Kernel 时完成判断或计算。

| 控制流写法 | 设备执行时处理 | 生成设备代码时处理 |
| --- | :---: | :---: |
| `if const_expr(...)` | ✗ | ✓ |
| `if target_version(...)` | ✗ | ✓ |
| `if` 使用运行时条件 | ✓ | ✗ |
| `while const_expr(...)` | ✗ | ✓ |
| `while target_version(...)` | ✗ | ✓ |
| `while` 使用运行时条件 | ✓ | ✗ |
| `for i in range_constexpr(...)` | ✗ | ✓ |
| `for i in 固定 tuple/list` | ✗ | ✓ |
| `for i in builtins.range(...)` | ✓ | ✗ |
| `for i in cannbotdsl.range(...)` | ✓ | ✗ |
| `for i in cannbotdsl.range(..., unroll=N)` | ✓ | ✗ |
| `for i in cannbotdsl.range(..., unroll_full=True)` | ✓ | ✗ |

“生成设备代码时处理”表示条件选择或循环展开在编译过程中完成，设备执行时不再保留
对应判断或循环。“设备执行时处理”表示前端保留相应控制流，由设备根据运行时条件
或循环边界执行。

`unroll` 和 `unroll_full` 是编译时确定的循环展开设置，但它们作用于设备循环，不会
改变 `cannbotdsl.range()` 作为设备循环的基本性质。

## 公共约束

CANNBotDSL 控制流语义只在 `@jit`、`@kernel` 及其追踪调用范围内生效，并存在如下约束：

| 约束类别 | 核心要求 | 适用结构 |
| --- | --- | --- |
| 路径完整性 | 控制流结束后使用的变量，必须在所有可能路径上都有确定值。 | `if` |
| 状态结构一致性 | 跨分支或跨迭代传递的变量，数据结构必须一致。 | `if`、`for`、`while` |
| 状态类型兼容性 | 同一变量在不同分支或迭代中的数值类型必须能够统一。 | `if`、`for`、`while` |
| 作用域与资源生命周期 | 控制流内部创建的 Tensor 等资源不能直接带到外部；循环变量也不能直接带出循环。 | `if`、`for`、`while` |
| 提前退出限制 | 不在编译期展开的控制流不能通过 `break`、`continue`、`return`、`raise` 跳出。 | `if`、`for`、`while` |
| 编译期与运行时边界 | 运行时数据不能传给只在编译期处理的接口。 | 编译期控制、循环范围 |

## 文档一览

| 文档 | 内容 |
| --- | --- |
| [判断条件](./predicate-expressions.md) | 说明如何通过 `and`、`or`、`not`、链式比较、`any()` 和 `all()` 产生或组合判断条件。 |
| [值选择](./value-selection.md) | 说明条件表达式以及 `min()`、`max()` 如何产生结果值。 |
| [`if`](./if.md) | 说明 `if`、`elif`、`else` 如何根据条件选择执行路径。 |
| [`for`](./for.md) | 说明 `for`、`for ... else`、各种 range 写法、固定序列迭代和循环展开设置。 |
| [`while`](./while.md) | 说明 `while`、`while ... else` 如何根据条件决定是否继续循环。 |
| [编译期控制](./compile-time-control.md) | 说明 `const_expr()` 和 `target_version()` 如何在生成设备代码时确定控制流。 |
