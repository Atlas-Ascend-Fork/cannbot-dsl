# Host 编译支持

本目录介绍提前编译所需的参数规格，以及查询 NPU 平台信息和片上存储容量的接口。

## 接口分类

| 分类 | 主要内容 | 接口 |
| --- | --- | --- |
| [参数与数据描述](./specs/index.md) | 描述动态维度、Tensor参数、Tensor列表参数和结构体参数。 | [Dim](./specs/dim.md)、[TensorSpec](./specs/tensor_spec.md)、[TensorListSpec](./specs/tensor_list_spec.md)、[StructSpec](./specs/struct_spec.md)、[StructSchema](./specs/struct_schema.md) |
| [平台信息](./platform/index.md) | 查询NPU型号、可用核数和片上存储容量。 | [get_platform_info](./platform/get-platform-info.md)、[get_mem_size](./platform/get-mem-size.md) |
