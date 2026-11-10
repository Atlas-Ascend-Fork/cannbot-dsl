# make_conv2d_spec

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

创建一个对卷积规则的描述 `Conv2dSpec`，包含输入特征图形状、权重形状以及步长、填充大小、窗口内相邻采样点间隔 和分组数，并据此计算特征图形状等搬运所需的信息，本身不搬运数据，也不执行卷积计算。
通过 `spec.tile(...)` 生成的 [Conv2dTile](conv2d-tile.md) 将各卷积搬运阶段关联起来，再配合 `matmul` 完成计算。

## 函数原型

```python
make_conv2d_spec(
    *,
    dtype: DType,
    input_shape: Sequence[int],
    filter_shape: Sequence[int],
    stride: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0, 0, 0),
    dilation: Sequence[int] = (1, 1),
    groups: int = 1,
    data_format: str = "nchw",
    filter_format: str = "nchw",
) -> Conv2dSpec
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| dtype | 输入 | 指定输入特征图和卷积权重的数据类型。支持 `dtypes.float16`、`dtypes.bfloat16`。必填，无默认值。 |
| input_shape | 输入 | 输入特征图形状，按 `(N, C, H, W)` 表示批次大小、输入通道数、高和宽；每个输入通道对应一个 H×W 数据平面。须为包含 4 个编译期正整数的 tuple/list，其中 H、W 不超过 65535。必填，无默认值。 |
| filter_shape | 输入 | 卷积权重形状，按 `(O, C/groups, Kh, Kw)` 表示输出通道数、每组输入通道数、卷积核高和宽。每个输出通道使用所在组的全部输入通道，在各通道上进行 Kh×Kw 加权求和，生成一层输出特征图。须为包含 4 个编译期正整数的 tuple/list，第二项须等于输入通道数除以 `groups`；Kh、Kw 为 1～511，且 Kh×Kw 不超过 65535。必填，无默认值。 |
| stride | 输入 | 卷积窗口在高、宽方向每次移动的距离，顺序为 `(stride_h, stride_w)`；`(1, 1)` 表示每次移动 1 格。须为包含 2 个编译期整数的 tuple/list，每项为 1～63。默认 `(1, 1)`。 |
| padding | 输入 | 输入特征图四周的补零宽度，顺序为 `(上, 下, 左, 右)`；仅描述填充规则，不立即修改输入数据。须为包含 4 个编译期整数的 tuple/list，每项为 0～255。默认 `(0, 0, 0, 0)`，即不补零。 |
| dilation | 输入 | 卷积窗口内部相邻采样点在高、宽方向的距离，顺序为 `(dilation_h, dilation_w)`；`(1, 1)` 表示连续取点，`(2, 2)` 表示隔 1 格取点，此时 3×3 卷积核覆盖 5×5 区域，仍只使用 9 个权重。须为包含 2 个编译期整数的 tuple/list，每项为 1～255。默认 `(1, 1)`。 |
| groups | 输入 | 将输入、输出通道分别等分为若干组，各组独立卷积；每个输出通道只使用所在组的输入通道。例如输入通道数为 16、`groups=2` 时，每组使用 8 个输入通道。须为编译期正整数，输入通道数 C 和输出通道数 O 均须能被其整除。默认 `1`，即不分组。 |
| data_format | 输入 | 指定输入特征图的维度顺序。类型为 `str`，仅支持 `"nchw"`，不区分大小写。默认 `"nchw"`。 |
| filter_format | 输入 | 指定卷积权重的维度顺序。类型为 `str`，仅支持 `"nchw"`，不区分大小写；此处表示权重按 OIHW 排列，即输出通道、每组输入通道、卷积核高、卷积核宽。默认 `"nchw"`。 |

## 返回值说明

返回 `Conv2dSpec`，保存卷积配置及推导结果，可读取以下属性：

| 属性 | 含义 |
| --- | --- |
| `dtype`、`input_shape`、`filter_shape`、`stride`、`padding`、`dilation`、`groups`、`data_format`、`filter_format` | 对应创建时的配置；形状、步长、填充和膨胀率保存为 tuple，格式字符串保存为小写。 |
| `batch_size` | 输入批次大小。 |
| `input_channels`、`input_height`、`input_width` | 输入通道数、输入高、输入宽。 |
| `output_channels` | 输出通道数。 |
| `filter_input_channels` | 权重中每组输入通道数。 |
| `kernel_height`、`kernel_width` | 卷积核高、宽。 |
| `input_channels_per_group`、`output_channels_per_group` | 每组输入、输出通道数，分别为对应总通道数除以 `groups`。 |
| `effective_kernel_height`、`effective_kernel_width` | 考虑膨胀率后的卷积核覆盖高、宽，分别为 `dilation_h * (Kh - 1) + 1`、`dilation_w * (Kw - 1) + 1`。 |
| `output_height`、`output_width` | 输出特征图的高、宽，由输入大小、两侧填充、卷积核覆盖范围和步长推导。 |
| `output_shape` | NCHW 输出形状 `(batch_size, output_channels, output_height, output_width)`。 |
| `m_extent` | 输出空间位置总数，即 `output_height * output_width`。 |

对象提供以下方法：

- [Conv2dSpec.tile_count](make-conv2d-spec/tile-count.md)：查询三个方向的切块数量。
- [Conv2dSpec.fmap_l1_shape](make-conv2d-spec/fmap-l1-shape.md)：查询特征图 L1 缓冲区容量形状。
- [Conv2dSpec.tile](make-conv2d-spec/tile.md)：生成当前卷积块的描述。

## 约束说明

卷积配置推导出的输出高、宽须为正数。

## 调用示例

参见 [conv2d_load_fmap 调用示例](../../data-movement/convolution/conv2d-load-fmap.md#调用示例)：先创建 `Conv2dSpec`，再生成 `Conv2dTile`，将输入特征图从 GM 加载到 L1。本接口为卷积搬运提供配置，本身不执行计算。
