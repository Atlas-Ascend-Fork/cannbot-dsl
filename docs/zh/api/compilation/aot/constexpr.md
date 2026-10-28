# Constexpr
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 简介

`Constexpr[T]` 是用于函数参数注解的类型标记，表示该参数的值需要在编译或追踪时确定。调用函数时传入的是 `T` 类型的实际值，不需要构造 `Constexpr` 对象。

## 定义

```python
Constexpr[T]
```

`T` 表示参数的实际 Python 类型，例如 `int` 或 `float`。`Constexpr` 只用于注解，不能直接调用或实例化。

## 使用示例

下面的示例将 `tile` 声明为编译时常量，并在调用时传入普通整数：

```python
import cannbotdsl as cbd


@cbd.host
def run(x: cbd.Tensor, tile: cbd.Constexpr[int]):
    ...


run(x, 64)
```
