# memloc

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 属性说明

`t.memloc`表示Tensor的数据所在的存储位置，值为`MemLoc`枚举项。

| 取值 | 含义 |
| --- | --- |
| `cbd.MemLoc.GM` | 全局内存，用于存放输入、输出等数据。 |
| `cbd.MemLoc.UB` | 统一缓冲区，用于向量和标量计算的输入、输出及中间数据。 |
| `cbd.MemLoc.L1` | L1缓冲区，用于缓存矩阵计算等操作的数据。 |
| `cbd.MemLoc.L0A` | 矩阵计算的左操作数缓冲区。 |
| `cbd.MemLoc.L0B` | 矩阵计算的右操作数缓冲区。 |
| `cbd.MemLoc.L0C` | 矩阵计算的累加结果缓冲区。 |
| `cbd.MemLoc.L0A_MX` | MX矩阵计算中左操作数的缩放因子存储空间。 |
| `cbd.MemLoc.L0B_MX` | MX矩阵计算中右操作数的缩放因子存储空间。 |
| `cbd.MemLoc.BIAS` | 矩阵计算的偏置缓冲区。 |
| `cbd.MemLoc.RegFile` | 寄存器存储空间。 |
| `cbd.MemLoc.SSBUF` | 核间通信存储空间，用于在AIC与AIV之间传递控制信息、地址或少量标量数据。 |
| `cbd.MemLoc.FBUF` | 量化参数缓冲区，用于存放缩放系数等量化参数。 |
| `cbd.MemLoc.FBUF_ACT` | 激活系数缓冲区，用于存放激活计算使用的系数表。 |

各片上空间的视图获取方式见[片上存储空间](../../storage/buffer/memory-spaces.md)。

## 示例

以下为Kernel函数体中的代码片段，`cbd`为通过`import cannbotdsl as cbd`导入的模块别名。

```python
t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((4, 8)))
location = t.memloc  # cbd.MemLoc.UB
```
