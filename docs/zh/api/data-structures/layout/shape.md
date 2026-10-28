# Shape

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`Shape` 描述一维或多维对象的**逻辑坐标范围**。

## 原型定义

```python
Shape = Union[int, _ScalarValue, Tuple["Shape", ...]]
```

`_ScalarValue` 表示 Kernel 运行时整数标量，例如 Kernel 运行时传入的维度值或 Kernel 内的整数计算结果。

| 使用场景 | 写法 | 含义 |
| --- | --- | --- |
| 一维 Shape | `128` 或 `(128,)` | 表示长度为 128 的一维坐标域。 |
| 普通多维 Shape | `(4, 8)` | 第 0 维长度为 4，第 1 维长度为 8，逻辑元素数量为 32。 |
| 含动态维度的 Shape | `(m, 8)` | `m` 来自 Kernel 标量参数或整数计算结果，该维长度在 Kernel 执行时确定。 |

## 约束说明

无

## 数据类型使用范围

| Shape 形式 | 普通 Python | `@host` | `@host` 内调用的 `@jit` | `@kernel` | `@kernel` 内调用的 `@jit` |
| --- | --- | --- | --- | --- | --- |
| 静态 Shape：仅包含 Python `int` 和 tuple | ✓ | ✓ | ✓ | ✓ | ✓ |
| 动态 Shape：包含运行时整数值 | ✗ | ✓ | ✓ | ✓ | ✓ |

✓ 表示可以在对应位置构造和使用该形式；✗ 表示该位置不存在这种运行时值。
