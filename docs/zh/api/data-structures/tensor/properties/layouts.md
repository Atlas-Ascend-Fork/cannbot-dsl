# layout / physical_layout

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 属性说明

`layout`描述计算代码使用的逻辑布局，`physical_layout`描述同一份数据在存储空间中的实际形状和跨度。两者的值均为`Layout`对象。

| 查看内容 | 逻辑布局 | 物理布局 |
| --- | --- | --- |
| 完整布局 | `t.layout` | `t.physical_layout` |
| 形状 | `t.layout.shape`，也可通过`t.shape`读取 | `t.physical_layout.shape` |
| 元素跨度 | 支持逐维跨度时读取`t.layout.stride`或`t.stride` | `t.physical_layout.stride` |

普通ND Tensor的两种布局可以相同。采用NZ、ZN分形格式时，计算代码仍按逻辑维度选取数据，物理布局则记录分形块的排列方式；此时不能用`t.stride`读取物理跨度。

使用[make_layout](../make_layout.md)创建布局；使用[from_tensor](../from_tensor.md)为共享存储的新视图指定逻辑布局和物理布局。

## 示例

以下为Kernel函数体中的代码片段，`cbd`为通过`import cannbotdsl as cbd`导入的模块别名。

### ND格式示例

```python
t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((4, 8)))
logical_shape = t.layout.shape            # (4, 8)
physical_shape = t.physical_layout.shape  # (4, 8)
physical_stride = t.physical_layout.stride  # (8, 1)
```

### NZ格式示例

```python
t = cbd.Buffer(cbd.MemLoc.L1, cbd.dtypes.float16, cbd.make_layout((32, 64)), data_format="nz")
logical_shape = t.layout.shape            # (32, 64)
physical_shape = t.physical_layout.shape  # (4, 2, 16, 16)
physical_stride = t.physical_layout.stride  # (512, 256, 16, 1)
```

| 属性值 | 含义 |
| --- | --- |
| 逻辑形状`(32, 64)` | 计算代码按32行、64列选取数据，`t.rank`为2。 |
| 物理形状`(4, 2, 16, 16)` | 数据分为4列、2行分形块，每块包含16行、16列元素。 |
| 物理跨度`(512, 256, 16, 1)` | 依次表示相邻分形块列、分形块行、块内行和块内列的元素间隔。 |

格式名称及含义见[data_format](./data-format.md)。
