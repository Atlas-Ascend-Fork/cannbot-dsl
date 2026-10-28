# 判断条件

判断条件表示真或假，可供`if`、`while`和条件表达式使用。CANNBotDSL支持使用`and`、`or`、`not`、链式比较以及Python内建`any()`、`all()`产生或组合条件。

## 文档一览

| 文档 | 说明 |
| --- | --- |
| [and](./predicate-expressions/and.md) | 逻辑与及短路规则。 |
| [or](./predicate-expressions/or.md) | 逻辑或及短路规则。 |
| [not](./predicate-expressions/not.md) | 真假取反。 |
| [链式比较](./predicate-expressions/chained-comparisons.md) | 连续比较多个操作数。 |
| [any](./predicate-expressions/any.md) | 判断是否存在真值。 |
| [all](./predicate-expressions/all.md) | 判断是否全部为真值。 |

## 真假判断规则

| 操作数形式 | 判断方式 |
| --- | --- |
| Python值 | 按Python的`bool(value)`规则判断，不限于`bool`类型。 |
| 运行时布尔值 | 直接作为条件。 |
| 运行时整数 | 非`0`为真，`0`为假。 |
| 运行时浮点数 | `+0.0`和`-0.0`为假，其他值（包括`NaN`）为真。 |
