---
title: 类型与相关接口
api_name: types and related interfaces
category: types-and-views
api_group: kernel
layer: frontend
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# 类型与相关接口

## 简介

CANNBotDSL 使用 Shape、Stride、Layout、Tiler 和 Coord 描述多维数据的形状、元素跨度、
布局、切块规格与坐标；Tensor 将实际存储位置与 Layout 组合为 Kernel 中可访问的多维
数据对象。

Buffer、Channel 和 DelayLineGroup 用于组织 Kernel 中的片上存储或流水状态。相关接口
用于根据已有视图建立资源、构造切块规格、换算坐标，以及计算维度或切块数量。

## 文档一览

| 文档 | 内容 |
| --- | --- |
| [Shape 类型](./shape.md) | 描述多维数据各维度的长度及其嵌套结构。 |
| [Stride 类型](./stride.md) | 描述各维坐标增加 1 时对应的元素跨度。 |
| [Layout 类型](./layout.md) | 组合 Shape 与 Stride，描述多维坐标到线性元素偏移的映射。 |
| [Tiler 类型](./tiler.md) | 描述各维的切块长度。 |
| [Coord 类型](./coord.md) | 描述 Layout 或 Tensor 中的位置坐标。 |
| [Tensor 类型](./tensor.md) | 描述实际存储位置、元素类型以及数据的逻辑布局和物理布局。 |
| [Buffer 类型](./buffer.md) | 通过片上存储分配创建 Tensor。 |
| [Channel 类型](./channel.md) | 管理按序循环使用的片上存储槽位及其生产、消费关系。 |
| [DelayLineGroup 类型](./delay-line-group.md) | 管理多字段值在流水阶段之间的延迟传递。 |
| [`make_tiler(...)`](./make_tiler.md) | 根据切块长度和对齐粒度创建切块描述。 |
| [`make_buffer(...)`](./make_buffer.md) | 将显式 UB 视图绑定为 Buffer。 |
| [`make_channel(...)`](./make_channel.md) | 将一组显式 UB 视图绑定为 Channel 槽位。 |
| [`channel_rewind()`](./channel_rewind.md) | 为后续 Channel 建立新的片上资源规划边界。 |
| [`idx2crd(...)`](./idx2crd.md) | 将线性元素编号转换为多维坐标。 |
| [`ceil_div(...)`](./ceil_div.md) | 计算整数或多维结构的向上取整除法。 |

