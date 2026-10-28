# Tile / Tiler 简介

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

Tile 和 Tiler 是切块规格的公开类型别名。

## 原型定义

```python
Tile = Union[int, _ScalarValue, Tuple["Tile", ...]]
Tiler = Union[Shape, Tile]
```

`_ScalarValue` 表示 Kernel 运行时整数标量，例如 Kernel 运行时传入的维度值或 Kernel 内的整数计算结果。

| 形式 | 示例 | 含义 |
| --- | --- | --- |
| Python 规格 | `(16, 32)` | 直接给出各维切块长度。 |
| 动态规格 | `(m, 32)` | `m` 是追踪中的运行时整数标量。 |
| Tile DSL 值 | `cannbotdsl.make_tiler((16, 32), alignment=16)` | 记录切块结构及可选对齐条件。 |

## 约束说明

无

## 数据类型使用范围

| 使用形式 | 普通 Python | `@host` | `@host` 内调用的 `@jit` | `@kernel` | `@kernel` 内调用的 `@jit` |
| --- | --- | --- | --- | --- | --- |
| 静态切块规格：Python `int` 及其嵌套 tuple | ✓ | ✓ | ✓ | ✓ | ✓ |
| 包含追踪中运行时整数的切块规格 | ✗ | ✓ | ✓ | ✓ | ✓ |

✓ 表示支持对应用法；✗ 表示不支持对应用法。
