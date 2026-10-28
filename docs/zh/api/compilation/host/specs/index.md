# Host 参数规格

本目录介绍参数与数据描述接口，用于声明程序参数的数据类型、形状、布局及动态维度等信息，供提前编译使用。

## 接口一览

| 接口 | 说明 |
| --- | --- |
| [Dim](./dim.md) | 为Tensor shape、stride或Tensor列表长度声明动态值及其范围约束。 |
| [TensorSpec](./tensor_spec.md) | 提前编译单个Tensor参数时，告诉编译器它的shape、dtype、stride和存储格式。 |
| [TensorListSpec](./tensor_list_spec.md) | 提前编译Tensor列表参数时，告诉编译器列表长度，以及每个Tensor的类型和形状规则。 |
| [StructSpec](./struct_spec.md) | 描述提前编译时结构体参数的字段规格。 |
| [StructSchema](./struct_schema.md) | 定义结构体名称、字段和字段类型。 |
