# simt.RoundingMode

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`RoundingMode`是舍入模式枚举，用于`simt.cast`、`simt.vcast`和`simt.add`的`rounding`参数。

`simt.RoundingMode`与`scalar.RoundingMode`、`reg.RoundingMode`使用同一枚举；各接口支持的成员及实际舍入行为分别由对应接口规定。

## 函数原型

```python
class RoundingMode(IntEnum):
    RN = 0
    RNA = 1
    RD = 2
    RU = 3
    RZ = 4
    RO = 5
    NA = 6
    RH = 7
```

## 参数说明

枚举成员说明如下。

| 枚举成员 | 舍入模式 | 说明 |
| --- | --- | --- |
| `RN` | `CAST_RINT` | 舍入到最近的可表示值，处于中点时选择最低有效位为偶数的结果。 |
| `RNA` | `CAST_ROUND` | 舍入到最近的可表示值，处于中点时向远离零的方向舍入。 |
| `RD` | `CAST_FLOOR` | 向负无穷方向舍入。 |
| `RU` | `CAST_CEIL` | 向正无穷方向舍入。 |
| `RZ` | `CAST_TRUNC` | 向零方向舍入。 |
| `RO` | `CAST_ODD` | 奇数舍入；发生精度损失时，使保留结果的最低有效位为1。 |
| `RH` | `CAST_HYBRID` | 混合舍入，用于支持该模式的`hifloat8`转换。 |
| `NA` | 未指定模式 | 共享枚举中的成员，当前`simt.cast`、`simt.vcast`、`simt.add`均不接受。 |

## 返回值说明

无

## 约束说明

无
