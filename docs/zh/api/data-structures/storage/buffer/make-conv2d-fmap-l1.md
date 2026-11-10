# make_conv2d_fmap_l1

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

为一个输入特征图块分配 L1 缓冲区，缓冲区大小由 [spec.fmap_l1_shape](../../convolution/make-conv2d-spec/fmap-l1-shape.md) 计算，容量推导逻辑及示例见该方法说明。

缓冲区由 [conv2d_load_fmap](../../../data-movement/convolution/conv2d-load-fmap.md) 写入，供 [conv2d_load_im2col](../../../data-movement/convolution/conv2d-load-im2col.md) 读取；本接口只分配空间，不加载数据。

## 函数原型

```python
make_conv2d_fmap_l1(dtype: DType, spec: Conv2dSpec, *, tile: Sequence[int]) -> Buffer
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| dtype | 输入 | 指定 L1 缓冲区的元素类型。<br>必填，无默认值。<br>支持 `dtypes.float16`、`dtypes.bfloat16`，须与 `spec.dtype` 相同；两种类型每个元素均占 2B。 |
| spec | 输入 | 卷积配置，用于推导输出宽度、输入窗口大小及缓冲区容量。<br>必填，无默认值。<br>须为 [make_conv2d_spec](../../convolution/make-conv2d-spec.md) 创建的 `Conv2dSpec` 对象。 |
| tile | 输入 | 指定一次卷积计算要处理多少个输出位置、多少个输入通道，格式为 `(BM, BC)`：BM 为每块处理的输出通道数，BC 为每块处理的输入通道数（分组内）。<br>必填，无默认值，须以关键字传入。<br>支持包含 2 个编译期整数的 tuple/list。BM 为正的 16 的整数倍且不超过 65535；BC 为 16 或 32；Kh×Kw×BC 不超过 32767，最大输入窗口高度乘输入宽度不超过 65535。 |

## 返回值说明

返回位于 L1 的一维 `Buffer`，元素类型为 `dtype`，形状为 `spec.fmap_l1_shape(tile=tile)`，以 `data_format="nd"` 声明原始存储空间。后续 `conv2d_load_fmap` 将数据按 NC1HWC0 排布写入；ND 声明本身不执行格式转换。

## 约束说明

缓冲区须在前一次读取结束后才能复用；多级流水应使用独立槽位管理读写依赖。

## 调用示例

参见 [conv2d_load_fmap 调用示例](../../../data-movement/convolution/conv2d-load-fmap.md#调用示例)：使用 `make_conv2d_fmap_l1` 分配 L1 缓冲区，再调用 `conv2d_load_fmap` 加载特征图，并读回校验 NC1HWC0 排布。
