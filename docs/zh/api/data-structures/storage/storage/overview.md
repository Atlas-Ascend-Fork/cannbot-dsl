# Storage 简介

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

Storage 是已有片上存储的句柄，记录容量、对齐要求和默认 Tensor 视图。

## 原型定义

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `size_bytes` | `int` | 句柄记录的存储字节容量。 |
| `alignment` | `int` | 存储起点的字节对齐要求。 |
| `memloc` | `MemLoc` | 存储所在层级。 |

## 使用方式

假设 `t` 为 Buffer 返回的 Tensor，`storage = t.bare()` 获取其存储句柄；`storage.view()` 返回默认视图，`storage.view(layout, dtype=...)` 按指定布局绑定新视图。

## 约束说明

- 不能直接调用 `Storage(...)` 构造句柄。

## 数据类型使用范围

| 使用形式 | 普通 Python | `@host` | `@host` 内调用的 `@jit` | `@kernel` | `@kernel` 内调用的 `@jit` |
| --- | --- | --- | --- | --- | --- |
| 调用 `Tensor.bare()` 获取 Storage | ✗ | ✗ | ✗ | ✓ | ✓ |
| 调用 `next_write(bare=True)` 或 `next_read(bare=True)` 获取 Storage | ✗ | ✗ | ✗ | ✓ | ✓ |
| 调用 `Storage.view(...)` 绑定 Tensor 视图 | ✗ | ✗ | ✗ | ✓ | ✓ |

✓ 表示支持对应用法；✗ 表示不支持对应用法。
