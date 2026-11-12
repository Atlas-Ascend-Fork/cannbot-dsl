# dtype

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 属性说明

`t.dtype`表示Tensor单个元素的数据类型，值为`DType`对象。例如，`cbd.dtypes.float32`表示32位浮点类型。完整取值及含义见[数据类型](../../data-types.md)。

## 示例

以下示例读取一个Tensor的数据类型，再用该类型创建临时Tensor。以下为Kernel函数体中的代码片段，`cbd`为通过`import cannbotdsl as cbd`导入的模块别名。

```python
t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((4, 8)))
element_type = t.dtype  # cbd.dtypes.float32

temp = cbd.Buffer(cbd.MemLoc.UB, element_type, cbd.make_layout((4, 8)))
temp_type = temp.dtype  # cbd.dtypes.float32
```
