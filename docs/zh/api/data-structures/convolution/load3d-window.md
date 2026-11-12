# Load3DWindow

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

选择 [load3d](../../data-movement/convolution/load3d.md) 目标`Tensor`的写入位置和大小，起点和大小均针对转置之前的矩阵。

## 函数原型

```python
Load3DWindow(
    *, m_extent: int, k_extent: int,
    m_start: int = 0, k_start: int = 0,
) -> Load3DWindow
```

整数参数也可传入核函数中计算得到的整数值，取值范围见参数说明。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| m_extent | 输入 | 窗口的输出空间位置数，0～65535；0 表示空搬运。8/16 位类型在非末尾 M 块中须为 16 的倍数。 |
| k_extent | 输入 | 窗口的卷积展开元素数，0～65535；0 表示空搬运。未触及完整 K 的最后一个分形块时须为 C0 的倍数。 |
| m_start | 输入 | M 轴起点，0～32767，单位为展开矩阵元素位置，不是输入特征图行号。 |
| k_start | 输入 | K 轴起点，0～65535，须为 C0 的倍数。C0 为一个 32B 通道块包含的元素数。 |

## 返回值说明

返回不可变的 `Load3DWindow` 配置对象。

## 约束说明

窗口的起点和大小按[load3d](../../data-movement/convolution/load3d.md)展开后的矩阵计算。启用转置时，目标Tensor的M、K两维须分别容纳窗口的K、M两维。

## 调用示例

见 [load3d 调用示例](../../data-movement/convolution/load3d.md#调用示例)。
