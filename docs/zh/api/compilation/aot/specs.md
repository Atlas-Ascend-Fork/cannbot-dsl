# AOT 参数规格

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 简介

AOT 参数规格描述提前编译时需要确定的形状、列表长度、结构参数和编译期常量。它们用于描述编译输入，不是设备侧 Tensor，也不是运行时缓冲区。

## 定义

| 类型 | 用途 |
| --- | --- |
| [Dim](../host/specs/dim.md) | 描述动态维度或动态列表长度的范围。 |
| [TensorSpec](../host/specs/tensor_spec.md) | 描述单个 Tensor 的形状、数据类型和布局规格。 |
| [TensorListSpec](../host/specs/tensor_list_spec.md) | 描述 Tensor 列表的长度和元素规格。 |
| [StructSpec](../host/specs/struct_spec.md) | 描述结构化参数的字段规格。 |
| [Constexpr](./constexpr.md) | 在参数注解中标记需要在编译时确定的值。 |

参数规格可以作为 `cannbotdsl.aot.compile` 的输入。结构参数使用 [StructSchema](../host/specs/struct_schema.md) 描述字段定义；具体用法以对应类型页面为准。
