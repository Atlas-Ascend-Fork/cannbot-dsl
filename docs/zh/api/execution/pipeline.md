# PIPE

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 简介

`PIPE` 是 CANNBot-DSL 用于表示硬件流水的枚举类型。它作为部分设备操作和同步接口的参数，用来指定需要等待、通知或维护顺序的流水。流水枚举本身不负责执行同步；是否需要选择某个值以及可选值范围，以调用它的具体接口说明为准。

## 定义

| 枚举项 | 数值 | 说明 |
| --- | ---: | --- |
| `P_NULL` | 0 | 空流水标识。 |
| `S` | 1 | 标量流水。 |
| `V` | 2 | 矢量流水。 |
| `M` | 3 | 矩阵流水。 |
| `MTE1` | 4 | MTE1 流水。 |
| `MTE2` | 5 | MTE2 流水。 |
| `MTE3` | 6 | MTE3 流水。 |
| `MTE4` | 7 | MTE4 流水。 |
| `MTE5` | 8 | MTE5 流水。 |
| `V2` | 9 | V2 流水。 |
| `ALL` | 10 | 全部流水。 |
| `FIXPIPE` | 11 | FIXPIPE 流水。 |
| `VLD` | 12 | 矢量加载流水。 |
| `VST` | 13 | 矢量存储流水。 |
| `LD` | 14 | 加载流水。 |
| `ST` | 15 | 存储流水。 |
| `LD_ST` | 16 | 加载和存储流水。 |
| `VEXE` | 17 | 矢量执行流水。 |
| `AIV_MTE2` | 18 | AIV 上的 MTE2 流水。 |
| `AIC_MTE3` | 19 | AIC 上的 MTE3 流水。 |
| `AIC_MTE1` | 20 | AIC 上的 MTE1 流水。 |

## 使用说明

- `PIPE` 从 `cannbotdsl` 顶层导入：`from cannbotdsl import PIPE`。
- 只在具体接口的参数要求中选择流水类型；不要把 `PIPE` 当作独立的同步原语调用。
- `RingBuffer` 是片上存储数据结构，负责槽位和游标管理，不属于 `PIPE` 的枚举项。需要了解槽位容器时，请参见 [RingBuffer](../data-structures/storage/ring-buffer/index.md)。
