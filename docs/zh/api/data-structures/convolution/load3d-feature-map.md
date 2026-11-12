# Load3DFeatureMap

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

配置[load3d](../../data-movement/convolution/load3d.md)处理的单个样本特征图的高度、宽度、通道数及四周填充量。填充值由[Load3DPadding](load3d-padding.md)指定，填充在`load3d`搬运时执行。

## 函数原型

```python
Load3DFeatureMap(
    *, height: int, width: int, channels: int,
    pad_top: int = 0, pad_bottom: int = 0,
    pad_left: int = 0, pad_right: int = 0,
) -> Load3DFeatureMap
```

整数参数也可传入核函数中计算得到的整数值，取值范围见参数说明。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| height | 输入 | 行输入特征图高度，1～32767，不含 padding。 |
| width | 输入 | 列输入特征图宽度，1～32767，不含 padding。 |
| channels | 输入 | 源操作数的通道数，取值范围：channelSize∈[0, 65535]。<br> 对于half，channelSize除16的余数应当为0，4或8。<br>对于int8_t和uint8_t，channelSize除32的余数应当为0，4，8或16。<br>对于int4b_t，channelSize为8，16，32，N*64，N*64+8，N*64+16，N*64+32，N为正整数。 |
| pad_top | 输入 | 上边界补值行数，0～255。 |
| pad_bottom | 输入 | 下边界补值行数，0～255。 |
| pad_left | 输入 | 左边界补值列数，0～255。 |
| pad_right | 输入 | 右边界补值列数，0～255。 |

## 返回值说明

返回不可变的 `Load3DFeatureMap` 配置对象，作为 `load3d(feature=...)` 的参数。

## 约束说明

`height`、`width`和`channels`须与L1源特征图的实际尺寸一致。与`filter`、`stride`、`dilation`的组合要求见[load3d](../../data-movement/convolution/load3d.md#约束说明)。

## 调用示例

见 [load3d 调用示例](../../data-movement/convolution/load3d.md#调用示例)。
