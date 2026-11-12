# shape / rank

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 属性说明

`shape`表示Tensor各维的长度，`rank`表示逻辑形状的顶层维数。

| 读取方式 | 类型 | 含义 |
| --- | --- | --- |
| `t.shape` | `Shape` | 各维长度组成的形状，等价于`t.layout.shape`。可以包含嵌套元组。 |
| `t.rank` | `int` | 逻辑形状的顶层维数。例如，形状为`(4, 8)`时，维数为2。 |

## 示例

以下为Kernel函数体中的代码片段，`cbd`为通过`import cannbotdsl as cbd`导入的模块别名。

### 普通形状

```python
t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((4, 8)))
shape = t.shape       # (4, 8)
rows = t.shape[0]     # 4
columns = t.shape[1]  # 8
rank = t.rank         # 2
```

### 嵌套形状

```python
t = cbd.Buffer(
    cbd.MemLoc.UB,
    cbd.dtypes.float32,
    cbd.make_layout(((4, 2), 8), stride=((16, 8), 1)),
)
shape = t.shape  # ((4, 2), 8)
rank = t.rank    # 2
```

嵌套形状`((4, 2), 8)`的顶层有两项，因此`rank`为2；第一项`(4, 2)`描述第一个维度的多级结构。
