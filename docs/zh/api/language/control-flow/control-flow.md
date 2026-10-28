# 控制流

## 简介

CANNBotDSL支持在`@host`、`@jit`和`@kernel`函数中使用部分Python控制流语法，并提供
`cannbotdsl.range()`、`range_constexpr()`、`const_expr()`和
`target_version()`等DSL API，用于条件判断、循环和结果选择。
需要按条件选择语句块时使用`if`，按范围或固定序列迭代时使用`for`，根据条件决定是否继续循环时使用`while`。

| 来源 | 支持的写法 |
| --- | --- |
| Python语法 | `if` / `elif` / `else`、条件表达式、`and` / `or` / `not`、链式比较、`for`、`while`、`for ... else`、`while ... else` |
| Python内建函数 | `min()`、`max()`、`any()`、`all()`、`builtins.range()` |
| CANNBotDSL API | `cannbotdsl.range()`、`range_constexpr()`、`const_expr()`、`target_version()` |

## 工作方式

CANNBotDSL在编译`@host`、`@jit`、`@kernel`函数时读取函数的Python语法结构，并识别其中
受支持的分支、循环、逻辑表达式和函数调用，并根据具体写法及是否依赖运行时数据，
在编译期或设备执行时处理。

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

显式编译期条件和固定序列迭代可以在生成设备代码的过程中完成条件选择、表达式计算
或循环展开。普通`range()`和依赖Tensor读取值、Kernel标量形参等运行时数据的
控制流，会保留为设备执行时的判断或循环；普通`range()`即使使用编译期已知的边界，也按设备循环处理。

在普通Python代码中，上述Python语法和内建函数仍由Python解释器按普通Python
规则执行。

## 编译与执行

仅包含普通Python值的条件或表达式在生成设备代码时按Python规则求值；包含Tensor
读取值、Kernel标量形参等运行时数据时，由设备执行Kernel时完成判断或计算。

| 控制流写法 | 设备执行时处理 | 生成设备代码时处理 |
| --- | :---: | :---: |
| `if const_expr(...)` | ✗ | ✓ |
| `if target_version(...)` | ✗ | ✓ |
| `if`使用运行时条件 | ✓ | ✗ |
| `while const_expr(...)` | ✗ | ✓ |
| `while target_version(...)` | ✗ | ✓ |
| `while`使用运行时条件 | ✓ | ✗ |
| `for i in range_constexpr(...)` | ✗ | ✓ |
| `for i in 固定 tuple/list` | ✗ | ✓ |
| `for i in builtins.range(...)` | ✓ | ✗ |
| `for i in cannbotdsl.range(...)` | ✓ | ✗ |
| `for i in cannbotdsl.range(..., unroll=N)` | ✓ | ✗ |
| `for i in cannbotdsl.range(..., unroll_full=True)` | ✓ | ✗ |

“生成设备代码时处理”表示条件选择或循环展开在编译过程中完成，设备执行时不再保留
对应判断或循环。“设备执行时处理”表示前端保留相应控制流，由设备根据运行时条件
或循环边界执行。

`unroll`和`unroll_full`在编译期确定，用于设置设备循环的展开方式，不会将其改为
`range_constexpr()`式的编译期迭代。

## 公共约束

CANNBotDSL控制流语义在`@host`、`@jit`、`@kernel`及其追踪调用范围内生效，并存在如下约束：

| 约束类别 | 核心要求 | 适用结构 |
| --- | --- | --- |
| 路径完整性 | 在`if`结束后使用变量时，该变量必须在所有可能路径上都有确定值。 | `if` |
| 状态结构一致性 | 变量跨分支或跨迭代传递时，其数据结构必须一致。 | `if`、`for`、`while` |
| 状态类型兼容性 | 同一变量在不同分支或迭代中的数值类型必须能够统一。 | `if`、`for`、`while` |
| 作用域与资源生命周期 | 运行时控制流内新建的局部Tensor和for循环变量不能带出对应作用域；嵌套的编译期控制流同样受外层运行时作用域约束。 | `if`、`for`、`while` |
| 提前退出限制 | 不在编译期展开的控制流不能通过`break`、`continue`、`return`、`raise`跳出。 | `if`、`for`、`while` |
| 编译期与运行时边界 | 运行时数据不能传给只在编译期处理的接口。 | 编译期控制、循环范围 |

## 文档一览

| 文档 | 内容 |
| --- | --- |
| [判断条件](./predicate-expressions.md) | 说明如何通过`and`、`or`、`not`、链式比较、`any()`和`all()`产生或组合判断条件。 |
| [值选择](./value-selection.md) | 说明条件表达式以及`min()`、`max()`如何产生结果值。 |
| [`if`](./if.md) | 说明`if`、`elif`、`else`如何根据条件选择执行路径。 |
| [`for`](./for.md) | 说明`for`、`for ... else`、各种range写法、固定序列迭代和循环展开设置。 |
| [`while`](./while.md) | 说明`while`、`while ... else`如何根据条件决定是否继续循环。 |
| [编译期控制](./compile-time-control.md) | 通过[`const_expr`](./compile-time-control/const-expr.md)和[`target_version`](./compile-time-control/target-version.md)在编译期选择分支或控制循环。 |
