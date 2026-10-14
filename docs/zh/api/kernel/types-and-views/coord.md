---
title: Coord
api_name: Coord
category: types-and-views
api_group: kernel
layer: frontend
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# Coord

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 简介

`Coord` 表示一维或多维坐标域中的一个位置。Shape 定义坐标域的范围，Coord 从中
选出一个位置。每一维坐标都从 `0` 开始。

## 定义

| 使用场景 | 写法 | 含义 |
| --- | --- | --- |
| 一维坐标 | `3` 或 `i` | `3` 表示坐标值为 3，即从 0 开始的第 4 个位置；也可以由 Kernel 整数值 `i` 指定位置。 |
| 普通多维坐标 | `(2, 3)` | 第 0 维坐标为 2，第 1 维坐标为 3。 |
| 含动态值的多维坐标 | `(row, col)` | `row`、`col` 可以来自 Kernel 标量参数、循环索引或整数计算结果。 |

### 约束

- Coord 必须为目标接口要求的每个维度提供坐标值。
- 静态坐标应位于 Shape 定义的范围内。

#### 数据类型使用范围

| Coord 形式 | 普通 Python | `@host` | `@host` 内调用的 `@jit` | `@kernel` | `@kernel` 内调用的 `@jit` |
| --- | --- | --- | --- | --- | --- |
| 静态 Coord：仅包含 Python `int`、`None` 和 tuple | ✓ | ✓ | ✓ | ✓ | ✓ |
| 动态 Coord：包含运行时整数值 | ✗ | ✓ | ✓ | ✓ | ✓ |

✓ 表示可以在对应位置构造和使用该形式；✗ 表示该位置不存在这种运行时值。
