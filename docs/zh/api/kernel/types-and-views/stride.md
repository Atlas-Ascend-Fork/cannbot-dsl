---
title: Stride
api_name: Stride
category: types-and-views
api_group: kernel
layer: frontend
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# Stride

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 简介

`Stride` 描述各维坐标变化与线性元素偏移变化之间的关系。某一维坐标增加 1 时，
该维对应的 Stride 值表示线性元素偏移增加多少。

## 定义

| 使用场景 | 写法 | 含义 |
| --- | --- | --- |
| 一维 Stride | `1` 或 `(1,)` | 一维坐标增加 1 时，线性元素偏移增加 1。 |
| 普通多维 Stride | `(8, 1)` | 第 0 维坐标增加 1 时偏移增加 8，第 1 维坐标增加 1 时偏移增加 1。 |
| 含动态值的 Stride | `(row_stride, 1)` | `row_stride` 来自 Kernel 标量参数或整数计算结果，具体值在 Kernel 执行时确定。 |

### 约束

#### 数据类型使用范围

| Stride 形式 | 普通 Python | `@host` | `@host` 内调用的 `@jit` | `@kernel` | `@kernel` 内调用的 `@jit` |
| --- | --- | --- | --- | --- | --- |
| 静态 Stride：仅包含 Python `int` 和 tuple | ✓ | ✓ | ✓ | ✓ | ✓ |
| 动态 Stride：包含运行时整数值 | ✗ | ✓ | ✓ | ✓ | ✓ |

✓ 表示可以在对应位置构造和使用该形式；✗ 表示该位置不存在这种运行时值。
