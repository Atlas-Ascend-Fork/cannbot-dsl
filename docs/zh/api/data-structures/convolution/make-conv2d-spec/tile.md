# Conv2dSpec.tile

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据块大小和块编号，确定本次处理的输出位置、输出通道和输入通道，并推导需要加载的输入特征图区域。返回 [Conv2dTile](../conv2d-tile.md)，供后续卷积搬运接口使用；本方法不搬运数据，也不执行计算。

## 函数原型

```python
spec.tile(
    *,
    tile: Sequence[int],
    coord: Sequence[int | ScalarValue],
    batch: int | ScalarValue = 0,
    group: int | ScalarValue = 0,
) -> Conv2dTile
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| tile | 输入 | 每块大小 `(BM, BN, BC)`：BM 为每块处理的输出空间位置数，BN 为输出通道数，BC 为输入通道数。输出高、宽按行展开后构成空间位置，例如 4×4 输出有 16 个位置；输入、输出通道均在当前分组内切分。这里的 N 表示输出通道方向，与 NCHW 中的批次 N 不同。<br>必填，无默认值。<br>支持包含 3 个编译期整数的 tuple/list。BM、BN 须为正的 16 的整数倍且不超过 65535；BC 为 16 或 32；Kh×Kw×BC 不超过 32767。最大输入窗口高度乘输入宽度不得超过 65535。 |
| coord | 输入 | 当前块编号 `(m_idx, n_idx, ci_idx)`，依次选择第几个输出空间位置块、输出通道块和输入通道块。编号从 0 开始，不是元素起始下标。例如 `(0, 0, 0)` 选择三个方向的第一块。<br>必填，无默认值。<br>支持包含 3 个整数或整数 DSL 标量的 tuple/list；各项须大于等于 0，且小于 [tile_count](tile-count.md) 返回的对应块数。 |
| batch | 输入 | 选择当前处理的输入样本，编号从 0 开始。<br>默认 `0`，即第一个样本。<br>支持整数或整数 DSL 标量，范围为 `[0, spec.batch_size)`。 |
| group | 输入 | 选择当前处理的卷积分组，输入、输出通道均在该组内切块，组编号从 0 开始。<br>默认 `0`，即第一个分组。<br>支持整数或整数 DSL 标量，范围为 `[0, spec.groups)`。 |

## 返回值说明

返回 [Conv2dTile](../conv2d-tile.md)，包含当前块的起点、尾块裁剪后的有效长度、输入窗口以及用于后续搬运和矩阵计算的视图方法。

## 约束说明

同一块的特征图加载、权重加载、窗口展开和输出写回须使用相互匹配的切块信息；运行时传入的编号也须在有效范围内。

## 调用示例

参见 [conv2d_load_fmap 调用示例](../../../data-movement/convolution/conv2d-load-fmap.md#调用示例)。
