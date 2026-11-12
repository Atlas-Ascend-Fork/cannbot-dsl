# data_format

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 属性说明

`t.data_format`返回当前Tensor的数据排布格式名称，类型为`str`，取值为小写字符串。

| 取值 | 含义 |
| --- | --- |
| `"nd"` | 普通N维格式，元素位置由布局中的形状和跨度描述。 |
| `"nz"` | NZ分形格式。矩阵分为多个小块，块间按列主序排列，块内按行主序排列。 |
| `"zn"` | ZN分形格式。矩阵分为多个小块，块间按行主序排列，块内按列主序排列。 |

`data_format`给出格式名称，具体形状和跨度由[physical_layout](./layouts.md)描述。分形块的大小与元素类型及存储位置有关；例如，L1中的`float16` NZ Tensor采用16行、16列的分形块。

对`reshape`、`flatten`和`unflatten`的返回视图，`data_format`与`stride`的读取条件相同，具体判定规则和对照示例见[stride](./stride.md#形状变换后的跨度)。

## 示例

以下为Kernel函数体中的代码片段，`cbd`为通过`import cannbotdsl as cbd`导入的模块别名。

```python
ordinary = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((4, 8)))
ordinary_format = ordinary.data_format  # "nd"

matrix = cbd.Buffer(cbd.MemLoc.L1, cbd.dtypes.float16, cbd.make_layout((32, 64)), data_format="nz")
matrix_format = matrix.data_format  # "nz"

transposed = matrix.permute(1, 0)
transposed_format = transposed.data_format  # "zn"
```

`transposed`的逻辑形状为`(64, 32)`，与`matrix`共享存储。格式名称按各自的逻辑坐标描述：原Tensor为NZ，交换两个矩阵维度后的视图为ZN。
