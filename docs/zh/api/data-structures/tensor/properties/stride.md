# stride

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 属性说明

`t.stride`表示各维坐标增加1时的元素偏移量，类型为`Stride`，与`t.shape`具有相同的结构。单位为元素；例如，`float32` Tensor的跨度8对应32字节的偏移。

以形状为`(4, 8)`的普通ND Tensor为例：

| `t.stride` | 含义 |
| --- | --- |
| `(8, 1)` | 相邻两行的起点相隔8个元素，同一行的相邻元素相隔1个元素。 |
| `(16, 1)` | 相邻两行的起点相隔16个元素，每行使用其中8个元素。 |
| `(0, 1)` | 各行访问同一组元素，例如由单行Tensor扩展得到的视图。 |

NZ、ZN分形Tensor不支持读取`t.stride`或`t.layout.stride`；应通过[`t.physical_layout.stride`](./layouts.md)读取物理布局的跨度。

### 形状变换后的跨度

以下规则用于判断`reshape`、`flatten`和`unflatten`返回视图是否支持读取`stride`和`data_format`。`flatten`的起止轴相同时返回原Tensor，属性保持不变；合并多个轴时按下表判断。

| 源Tensor及目标形状 | `stride` | `data_format` |
| --- | --- | --- |
| 源Tensor为NZ、ZN格式，或源视图本身不支持读取`data_format`。 | 不支持读取。 | 不支持读取。 |
| 源Tensor为ND格式，除首维外的各维长度及全部跨度均为整数常量，且满足下述行主序连续条件。目标形状可含运行时长度。 | 支持读取，返回变换后各维的跨度。 | 返回`"nd"`。 |
| 不满足上一行的ND Tensor，源形状、目标形状及源跨度均为正整数常量。 | 只在下述轴拆分、合并条件成立时支持读取，否则不支持读取。 | 同一条件成立时返回`"nd"`，否则不支持读取。 |
| 其余ND输入。 | 不支持读取。 | 不支持读取。 |

**行主序连续条件**：末轴跨度为`1`，其他轴满足`stride[i] = shape[i + 1] * stride[i + 1]`。例如，形状为`(2, 3)`、跨度为`(3, 1)`时满足条件。形状为`(n, 3)`且只有`n`在运行时确定时，跨度`(3, 1)`也满足条件。

**轴拆分、合并条件**：忽略长度为`1`的轴后，相邻轴只有满足`stride[i] = shape[i + 1] * stride[i + 1]`才能合并。新形状可在这些轴组成的区间内拆分或合并，不能跨越不满足该等式的边界。`unflatten`只拆分原有轴，满足此条件。

以下示例的形状和跨度均为整数常量，源Tensor均为ND格式：

| 源形状 | 源跨度 | 操作 | 返回视图的属性 |
| --- | --- | --- | --- |
| `(2, 3)` | `(3, 1)` | `flatten()` | `stride=(1,)`，`data_format="nd"`。 |
| `(2, 3)` | `(6, 2)` | `flatten()` | `stride=(2,)`，`data_format="nd"`。 |
| `(2, 3)` | `(8, 1)` | `flatten()`或`reshape(3, 2)` | 两个属性均不支持读取。 |
| `(2, 3)` | `(8, 1)` | `reshape(2, 1, 3)` | `stride=(8, 3, 1)`，`data_format="nd"`。 |
| `(2, 3, 4)` | `(100, 4, 1)` | `flatten(1, 2)` | `stride=(100, 1)`，`data_format="nd"`；第一维的存储间隔不影响后两维合并。 |
| `(2, 3, 4)` | `(100, 4, 1)` | `flatten(0, 1)` | 两个属性均不支持读取；合并范围跨越第一、二维之间的存储间隔。 |

以形状为`(2, 3)`、跨度为`(8, 1)`的Tensor为例，两行数据在存储中分别位于元素偏移`0、1、2`和`8、9、10`处，中间有空隙。调用`flatten()`后，这六个元素按原顺序组成一维视图，但相邻元素的存储间隔依次为`1、1、6、1、1`，无法用一个固定跨度表示。

该结果不支持读取`stride`和`data_format`，原数据的存储布局仍可通过`physical_layout`查看。即使调用`unflatten(0, (2, 3))`恢复原来的二维形状，也不能读取这两个属性。

对于连续ND Tensor，例如形状为`(2, 3)`、跨度为`(3, 1)`的Tensor，展平后的跨度则为`(1,)`。

## 示例

以下为Kernel函数体中的代码片段，`cbd`为通过`import cannbotdsl as cbd`导入的模块别名。

```python
contiguous = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((4, 8)))
contiguous_stride = contiguous.stride  # (8, 1)

spaced = cbd.Buffer(
    cbd.MemLoc.UB,
    cbd.dtypes.float32,
    cbd.make_layout((4, 8), stride=(16, 1)),
)
spaced_stride = spaced.stride  # (16, 1)

row = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((1, 8)))
expanded = row.expand(4, 8)
expanded_stride = expanded.stride  # (0, 1)
```

`expanded`与`row`共享存储，第一维跨度为0，表示各行访问同一组元素。扩展操作说明见[expand](../expand.md)。

以下代码对应前文行间有空隙的Tensor：

```python
t = cbd.Buffer(
    cbd.MemLoc.UB,
    cbd.dtypes.float32,
    cbd.make_layout((2, 3), stride=(8, 1)),
)
flat = t.flatten()
logical_shape = flat.shape                  # (6,)
physical_shape = flat.physical_layout.shape  # (2, 3)
physical_stride = flat.physical_layout.stride  # (8, 1)
# 此时读取flat.stride或flat.data_format会报错。
```
