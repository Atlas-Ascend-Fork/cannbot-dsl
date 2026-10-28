# Buffer 简介

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

Buffer 用于分配片上存储，调用后直接返回 Tensor。逻辑布局指定数据的形状和访问跨度，物理布局确定实际预留空间。

## 原型定义

```python
Buffer(mem_loc, dtype, layout, *, physical_layout=None, data_format=None) -> Tensor
```


## 数据类型使用范围

| 使用形式 | 普通 Python | `@host` | `@host` 内调用的 `@jit` | `@kernel` | `@kernel` 内调用的 `@jit` |
| --- | --- | --- | --- | --- | --- |
| 调用 `Buffer(...)` 分配片上存储，返回 Tensor | ✗ | ✗ | ✗ | ✓ | ✓ |
| 调用 `make_buffer(t)` 绑定已有 Tensor | ✗ | ✗ | ✗ | ✓ | ✓ |

✓ 表示支持对应用法；✗ 表示不支持对应用法。
