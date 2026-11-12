# Indexing

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

Tensor索引用于选择共享存储的视图，或访问指定位置的单个元素。以下以Kernel函数中的Tensor为例，`cbd`为通过`import cannbotdsl as cbd`导入的模块别名。

## 获取Tensor视图

索引中使用`None`或切片区间时，返回与原Tensor共享存储的视图，可作为[mem_copy](../../data-movement/mem-copy.md)等接口的操作数。

例如，`src`为GM中形状`(64, 128)`的二维`float32` Tensor，`dst`为UB中形状`(16, 32)`的`float32` Tensor。以下Kernel函数体片段选择一个矩形区域并搬运到`dst`：

```python
tile = src[16:32, 32:64]
cbd.mem_copy(dst, tile)
```

切片本身不搬运数据。通过视图写入数据时，会修改原Tensor对应位置的数据。

以`t.shape=(4, 8)`为例：

| 写法 | 选取范围 | 返回结果 |
| --- | --- | --- |
| `t[1:3, 2:6]` | 下标为1、2的行，以及下标为2、3、4、5的列。 | 形状为`(2, 4)`的Tensor视图。 |
| `row = t[1, None]` | 第二行的全部元素。 | 形状为`(1, 8)`的Tensor视图，可通过`row[0, j]`访问该行元素。 |
| `t[None, None]` | 保留两个维度的全部范围。 | 覆盖原范围的Tensor视图。 |

按块大小和块坐标选取数据时，也可以使用[tile_slice](../../compute/tensor/tile-slice.md)。

### 约束说明

- 切片边界必须非负，步长必须大于`0`。
- 对静态维度使用编译期已知的整数切片边界时，超出末端的边界会裁剪到该维度长度。
- 切片`start:stop`中，`start`为起始下标，`stop`为不包含在选取范围内的结束下标。使用运行时整数值时，必须满足`0 ≤ start ≤ stop ≤ 对应维度长度`，不会自动裁剪。
- 保留完整维度使用`None`；完整的`:`不作为保留维度的标记。

## 单元素访问

完整的整数索引用于读取或写入一个标量，例如读取任务元数据，或在SIMT中访问当前线程处理的元素。

### 坐标

坐标（`Coord`）用于指定元素位置。例如，形状为`(4, 8)`时，坐标`(2, 3)`表示行下标为2、列下标为3的元素，下标从0开始。嵌套形状的坐标结构须与形状一致。

### 读写示例

以GM中的一维元数据Tensor`metadata`为例，以下Kernel函数体片段读取第一个任务参数：

```python
task_parameter = metadata[0]
```

| 写法 | 含义 |
| --- | --- |
| `t[i]` | 读取一维Tensor下标为`i`的元素。 |
| `t[i, j]` | 读取二维Tensor行下标为`i`、列下标为`j`的元素。 |
| `t[i, j] = value` | 将标量`value`写入二维Tensor的指定位置。 |

SIMT中根据线程坐标访问元素的示例见[thread_idx](../../simt/thread/thread-idx.md)。SIMD的向量读写参见[vload](../../compute/vector/load/vload.md)和[vstore](../../compute/vector/store/vstore.md)。

### 约束说明

- 每个维度都须给出整数坐标；嵌套布局的坐标结构须与逻辑形状一致。坐标可以是编译期已知的整数或运行时整数值，且必须位于对应维度的范围内。
- 写入值可以是符合目标元素类型取值范围的数值常量，或与目标元素类型一致的运行时标量；不同类型须先显式转换。
- 普通单元素赋值支持GM和UB；SSBUF的单元素访问用于AIC与AIV之间的标量通信。UB的单元素赋值不能写在SIMD VF中。
- 赋值必须使用完整整数坐标，不支持通过切片或`None`批量赋值。
