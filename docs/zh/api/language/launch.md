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

上例启动8个block，`x`、`y`、`out`为核函数的实参。省略方括号时使用默认配置。

## 使用位置

- 启动语句须位于`@host`函数体中。例如，上面的`add_kernel[8](x, y, out)`写在`run`内部；不能将这条启动语句移到`@jit`辅助函数中，再由`run`调用该辅助函数。
- 不能在普通 Python 调用层直接调用 `@kernel`，也不能在 `@jit` 或其他设备函数中启动 Kernel。

## 相关接口

- [`@kernel`](./kernel.md)：定义 Kernel。
- [`@host`](./host.md)：提供 Kernel 启动入口。
- [矢量函数作用域](../compute/vector/overview.md)：在 Kernel 中选择 SIMD 或 SIMT 计算模式。
