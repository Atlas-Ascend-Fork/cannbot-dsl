# Kernel 启动配置

## 功能说明

Kernel由`@host`函数通过方括号语法启动，方括号中的整数表示启动的block数量。纯SIMT Kernel的启动方式见[SIMT Kernel启动配置](../simt/launch.md)。

## 调用形式

```python
import cannbotdsl as cbd


@cbd.host
def run(x, y, out):
    add_kernel[8](x, y, out)
```

上例中，`x`、`y`、`out` 为核函数的实参。方括号内的 `block` 指定核函数本次启动的逻辑核数量，`[8]` 表示启动 8 个逻辑核。纯 Vector 或纯 Cube 场景下，该值分别表示启动的 Vector Core 或 Cube Core 实例数。在 Mix 1:2 模式下，该值表示启动的核组合数，每个组合包含 1 个 Cube Core 和 2 个 Vector Core。

`block` 支持 Python `int` 或非布尔 DSL 整型标量，取值须大于 0。省略方括号或传入 `None` 时，默认启动 1 个逻辑核。

## 使用位置

- 启动语句须位于`@host`函数体中。例如，上面的`add_kernel[8](x, y, out)`写在`run`内部；不能将这条启动语句移到`@jit`辅助函数中，再由`run`调用该辅助函数。
- 不能在普通 Python 调用层直接调用 `@kernel`，也不能在 `@jit` 或其他设备函数中启动 Kernel。

## 相关接口

- [`@kernel`](./kernel.md)：定义 Kernel。
- [`@host`](./host.md)：提供 Kernel 启动入口。
- [矢量函数作用域](../compute/vector/overview.md)：在 Kernel 中选择 SIMD 或 SIMT 计算模式。
