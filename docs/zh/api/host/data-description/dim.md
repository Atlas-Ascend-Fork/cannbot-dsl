---
title: Dim
api_name: Dim
category: data-description
api_group: host
layer: frontend
call_context: host
execution_unit: none
status: experimental
since: 待追溯
---

# `Dim(...)`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

`Dim` 是 CANNBotDSL 提供的 Host 侧动态维度类型，也是提前编译规格中动态长度的
占位符。在 `TensorSpec` 或 `TensorListSpec` 中无法提前确定某个长度时，可以使用
`Dim` 为这个长度命名，并声明运行时允许的最小值、最大值和整除倍数。

`Dim` 可以通过 `+`、`-`、`*`、`//` 与 Python 整数、其他 `Dim` 或已有表达式组成
维度表达式。执行编译结果时，框架先根据传入的数据绑定各个 `Dim` 占位符，再按照
表达式中的运算关系依次计算最终值。

## 函数原型

```python
Dim(
    name: str,
    min: int = 1,
    max: int | None = None,
    multiple_of: int = 1,
) -> Dim
```

## 参数说明

| 参数 | 类型 | 必选 | 默认值 | 详细说明 |
| --- | --- | --- | --- | --- |
| `name` | `str` | 是 | 无 | 动态维度的名称。同一次 `cannbotdsl.compile(...)` 调用中，`name` 相同的 `Dim` 表示同一个动态长度。 |
| `min` | Python `int` | 否 | `1` | 允许运行时实际长度取得的最小值，包含该值；必须大于或等于 0。 |
| `max` | Python `int` 或 `None` | 否 | `None` | 允许运行时实际长度取得的最大值，包含该值；`None` 表示不声明最大值，显式指定时必须大于或等于 `min`。 |
| `multiple_of` | Python `int` | 否 | `1` | 声明运行时实际长度必须是该值的整数倍；必须大于或等于 1。 |

## 返回值说明

返回不可变的 `Dim` 对象。

## 约束说明

- 同一次 `cannbotdsl.compile(...)` 调用中，`name` 相同的 `Dim` 必须声明完全相同的
  `min`、`max` 和 `multiple_of`，否则会产生约束冲突。
- `+`、`-`、`*` 的另一侧可以是 Python `int`、`Dim` 或维度表达式；`//` 的除数必须
  是正 Python `int`。
- 维度表达式不是新的 `Dim`，不单独具有 `name`、`min`、`max` 或 `multiple_of`。
- 使用 `M * 2` 等维度表达式时，同一次 `cannbotdsl.compile(...)` 的 shape、stride
  或 TensorList length 中必须至少有一个位置直接使用 `M`。框架从该位置取得 `M`
  的实际值，不会根据表达式结果反向求解 `M`。

