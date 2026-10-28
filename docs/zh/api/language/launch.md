# Kernel 启动配置

## 功能说明

Kernel 由 `@host` 函数通过方括号语法启动。方括号中的配置决定启动的 block 数量；配置为 `dim3` 时，还可以表达纯 SIMT Kernel 的 block 和 thread 布局。

## 调用形式

### SIMD 或混合 Kernel

```python
@cb.host
def run(x, y, out):
    add_kernel[8](x, y, out)
```

整数配置表示 block 数量。省略方括号时使用默认配置。

### 纯 SIMT Kernel

```python
@cb.host
def run(x, y, out):
    simt_kernel[cb.dim3(2), cb.dim3(128)](x, y, out)
```

纯 SIMT 形式使用两个 `dim3` 值，分别表示 block 配置和 thread 配置；第三个位置可以传入动态 UB 缓冲区大小：

```python
simt_kernel[cb.dim3(2), cb.dim3(128), dyn_ub_buf](x, y, out)
```

## `dim3`

`dim3(x, y=1, z=1)` 创建三维启动配置。三个轴都必须是正的 32 位无符号整数；thread 配置的体积不能超过 1024。`dim3` 只用于需要显式 thread 布局的纯 SIMT 启动形式。

## 使用位置

- 启动语句必须直接写在 `@host` 函数体中。
- 不能在普通 Python 调用层直接调用 `@kernel`，也不能在 `@jit` 或其他设备函数中启动 Kernel。
- 纯 SIMD 和 SIMD/SIMT 混合 Kernel 使用整数 block 配置；纯 SIMT Kernel 使用两个 `dim3` 配置。

## 相关接口

- [`@kernel`](./decorators.md#kernel)：定义 Kernel。
- [`@host`](./decorators.md#host)：提供 Kernel 启动入口。
- [矢量函数作用域](../compute/vector/overview.md)：在 Kernel 中选择 SIMD 或 SIMT 计算模式。
