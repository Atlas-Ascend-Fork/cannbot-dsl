# Layout 简介

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

Layout 描述形状信息和元素跨度。

## 原型定义

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `shape` | `Shape` | 该布局的形状。 |
| `stride` | `Stride` | 普通布局中，各轴坐标增加 1 对应的元素偏移增量。 |
| `physical_layout` | `Layout` | 从 Tensor 的逻辑布局获取对应的物理布局。 |

## 约束说明

- Tensor 的逻辑 Layout 的 `physical_layout` 属性返回对应的物理 Layout；Tensor 的物理 Layout 及 `make_layout()` 创建的 Layout，读取该属性均返回自身。

- `shape` 和 `stride` 的 tuple 嵌套结构必须一致，各维长度与对应跨度一一匹配。

## 数据类型使用范围

| 使用形式 | 普通 Python | `@host` | `@host` 内调用的 `@jit` | `@kernel` | `@kernel` 内调用的 `@jit` |
| --- | --- | --- | --- | --- | --- |
| 将 `Layout` 用作类型注解 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 创建或操作 Layout 值 | ✗ | ✓ | ✓ | ✓ | ✓ |

✓ 表示支持对应用法；✗ 表示不支持对应用法。
