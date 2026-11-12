# Tensor

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

Tensor用于表示算子处理的多维数据，包含形状、跨度、数据类型、存储位置和布局等属性。

## 属性

| 属性 | 类型 | 简要说明 |
| --- | --- | --- |
| [`shape`](./properties/shape-and-rank.md) | `Shape` | 逻辑形状，即各维的长度。 |
| [`rank`](./properties/shape-and-rank.md) | `int` | 逻辑形状的顶层维数。 |
| [`stride`](./properties/stride.md) | `Stride` | 各维的元素跨度。 |
| [`dtype`](./properties/dtype.md) | `DType` | 单个元素的数据类型。 |
| [`memloc`](./properties/memloc.md) | `MemLoc` | 数据所在的存储位置。 |
| [`layout`](./properties/layouts.md) | `Layout` | 计算代码使用的逻辑布局。 |
| [`physical_layout`](./properties/layouts.md) | `Layout` | 数据实际存储使用的物理布局。 |
| [`data_format`](./properties/data-format.md) | `str` | 数据排布格式，返回`"nd"`、`"nz"`或`"zn"`。 |

索引与切片规则见[Indexing](./indexing.md)。

## Tensor视图

`reshape`、`flatten`和`permute`等方法返回与原Tensor共享存储的视图，只改变查看数据的形状或轴顺序，不复制数据。例如，将形状为`(2, 3)`的Tensor展平为`(6,)`后，两者仍访问同一份数据；通过任一视图修改元素，另一个视图也能看到修改后的值。

`reshape`、`flatten`和`unflatten`返回视图的`stride`、`data_format`读取规则见[stride](./properties/stride.md#形状变换后的跨度)。

## 搬运中的数据处理

[relu](./relu.md)、[convert](./convert.md)和[pad](./pad.md)用于为`mem_copy`指定ReLU、类型转换与量化、填充等随路处理。这些处理在`mem_copy`搬运时执行，配合方式见[mem_copy的随路功能](../../data-movement/mem-copy.md#随路功能)。
