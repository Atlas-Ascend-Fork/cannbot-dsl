---
title: Layout
api_name: Layout
category: types-and-views
api_group: kernel
layer: frontend
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# Layout

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 简介

`Layout` 是由 Shape 和 Stride 组成的类型，用于把 Coord 指定的多维坐标
映射为线性元素偏移。

## 定义

### 属性

| 属性 | 类型 | 详细说明 |
| --- | --- | --- |
| `shape` | `Shape` | Layout 的逻辑坐标域。 |
| `stride` | `Stride` | 各维坐标变化对应的线性元素偏移变化量，tuple 层次与 shape 一致。 |


### 坐标映射关系

对于普通多维 Layout，`shape[i]` 限定第 `i` 维坐标的范围：
`0 <= coord[i] < shape[i]`。`stride[i]` 表示该维坐标增加 1 时，线性元素偏移
增加多少。将 Coord 各维的坐标值乘以对应的 Stride，再相加，就得到该坐标相对于
Layout 起点的元素偏移：

```text
元素偏移 = coord[0] * stride[0]
         + coord[1] * stride[1]
         + ...
```

### 嵌套属性层次结构

Layout 可以使用嵌套 tuple 保留维度层次。外层 tuple 中可以单独选择的每一项称为
一个 `mode`。以 `shape=((2, 3), 4)` 为例：

```text
顶层 mode 0：shape=(2, 3)，包含 2 * 3 = 6 个坐标组合
顶层 mode 1：shape=4
全部最终维度：2、3、4
```

配套的 Shape、Stride 和 Coord 可以写成：

| 项目 | 写法 | 含义 |
| --- | --- | --- |
| Shape | `((2, 3), 4)` | 顶层 mode 0 包含长度为 2、3 的两个子维度，顶层 mode 1 的长度为 4。 |
| Stride | `((12, 4), 1)` | 三个最终维度的元素偏移变化量依次为 12、4、1。 |
| Coord | `((1, 2), 3)` | 三个最终维度的坐标依次为 1、2、3。 |

该坐标对应的元素偏移为：

```text
1 * 12 + 2 * 4 + 3 * 1 = 23
```

### 约束

- 不支持直接调用 `Layout(...)`，也不能通过继承定义 Layout 子类。
- 嵌套 Layout 的 Shape 和 Stride 必须具有相同的 tuple 层次。

#### 数据类型使用范围

| 使用形式 | 普通 Python | `@host` | `@host` 内调用的 `@jit` | `@kernel` | `@kernel` 内调用的 `@jit` |
| --- | --- | --- | --- | --- | --- |
| 将 `Layout` 用作类型注解 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 创建或操作 Layout 值 | ✗ | ✓ | ✓ | ✓ | ✓ |

✓ 表示支持对应用法；✗ 表示不支持对应用法。
