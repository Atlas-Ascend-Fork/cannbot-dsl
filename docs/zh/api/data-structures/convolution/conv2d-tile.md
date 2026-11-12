# Conv2dTile

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

为当前卷积块取得特征图、权重和计算结果的 Tensor 视图。先用 [Conv2dSpec.tile](make-conv2d-spec/tile.md) 选定切块，再调用以下方法，把已有缓冲区的有效范围交给搬运或矩阵乘接口。

## 函数原型

```python
tile.l0a_view(slot: Tensor) -> Tensor
tile.l1b_view(slot: Tensor) -> Tensor
tile.l0b_view(slot: Tensor) -> Tensor
tile.l0c_view(slot: Tensor) -> Tensor
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| slot | 输入 | 对应存储位置的已分配缓冲区。必填，无默认值。类型为 `Tensor`。 |

`slot`的容量和布局须满足下表。

| 方法 | 缓冲区容量及布局 |
| --- | --- |
| `l0a_view` | L0A Buffer，按 `(BM, Kh×Kw×BC)` 预留。 |
| `l1b_view` | L1 Buffer，使用 ZN 视图，按 `(BN, Kh×Kw×BC)` 预留。 |
| `l0b_view` | L0B Buffer，按 `(BN, Kh×Kw×BC)` 分配后交换逻辑轴。 |
| `l0c_view` | L0C Buffer，按 `(BM, BN)` 预留。 |

## 返回值说明

返回与 `slot` 共享存储的 Tensor 视图。创建视图时无需分配新缓冲区。

## 约束说明

特征图加载、权重加载、矩阵乘与输出写回须使用匹配的切块信息。缓冲区容量须覆盖补齐后的物理访问范围。

## 调用示例

见 [conv2d_store_output 调用示例](../../data-movement/convolution/conv2d-store-output.md#调用示例)，其中使用四个视图方法完成单块二维卷积。
