---
title: Tiler
api_name: Tiler
category: types-and-views
api_group: kernel
layer: frontend
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# Tiler

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 简介

`Tiler` 是描述一次切块规格的 DSL 类型，用于说明切块在各维覆盖的元素数量，以及
这些维度可选的对齐粒度。

## 定义

一个 Tiler 包含以下两部分信息：

| 组成 | 含义 |
| --- | --- |
| 切块长度 | 描述一次切块在各维覆盖的元素数量，例如 `(16, 32)` 表示两个维度分别覆盖 16 和 32 个元素。长度可以在编译期确定，也可以由 Kernel 标量形参或整数计算结果在运行时确定。 |
| `alignment` | 声明各维切块长度满足的对齐粒度。`alignment` 必须是编译期已确定的静态值，不能使用 Kernel 运行时整数值。`None` 表示不额外声明；alignment 只记录对齐条件，不会自动放大、补齐或修改切块长度。 |

Tiler 可以采用以下形式：

| 形式 | 示例 | 说明 |
| --- | --- | --- |
| 一维 Tiler | `64` 或 `(64,)` | 一次切块覆盖 64 个元素。 |
| 多维 Tiler | `(16, 32)` | 各维分别覆盖 16 和 32 个元素。 |
| 动态 Tiler | `(block_m, 32)` | `block_m` 在 Kernel 运行时确定。 |
| 嵌套 Tiler | `((m, n), k)` | 保留接口要求的嵌套分组结构。 |

### 约束

- 编译期确定的切块长度必须为正 Python `int`；运行时长度必须能转换为 Kernel
  运行时使用的 64 位整数。

#### 数据类型使用范围

| Tiler 形式 | 普通 Python | `@host` | `@host` 内调用的 `@jit` | `@kernel` | `@kernel` 内调用的 `@jit` |
| --- | --- | --- | --- | --- | --- |
| 静态 Tiler：仅包含 Python `int` 和 tuple | ✓ | ✓ | ✓ | ✓ | ✓ |
| 动态 Tiler：包含运行时整数值 | ✗ | ✓ | ✓ | ✓ | ✓ |
| `make_tiler()` 创建的切块描述值 | ✗ | ✓ | ✓ | ✓ | ✓ |

✓ 表示可以在对应位置构造和使用该形式；✗ 表示不支持在对应位置创建或操作该形式。
