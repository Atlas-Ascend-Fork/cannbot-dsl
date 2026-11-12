# make_ring_buffer

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

用传入的多个静态片上 Tensor 建立循环槽位，复用这些 Tensor 的底层存储。槽位顺序与传入序列一致。

## 函数原型

```python
make_ring_buffer(tensors) -> RingBuffer
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `tensors` | 输入 | 槽位 Tensor 序列。必填，为非空 `list[Tensor]` 或 `tuple[Tensor, ...]`。各 Tensor 必须来自[片上存储空间](../buffer/memory-spaces.md)，具有相同存储层级、元素类型、形状和逻辑/物理布局；形状及物理跨度须为静态正整数。 |

## 返回值说明

返回 RingBuffer，`depth` 为传入 Tensor 的数量；槽位保留各自的地址，并采用输入 Tensor 的元素类型和布局。

公开属性参见 [RingBuffer 简介](./overview.md)。

## 约束说明

- 支持 `MemLoc.UB`、`MemLoc.L1`、`MemLoc.L0A`、`MemLoc.L0B`、`MemLoc.L0C`、`MemLoc.BIAS`。
- 各 Tensor 必须尚未绑定为其他 Buffer 或 RingBuffer。
- 各槽位物理范围和起点须满足所在存储层级的容量及对齐要求。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    root = cbd.UB.view(262144)
    a = root[0:128,].view(cbd.dtypes.int64).view(16)
    b = root[128:256,].view(cbd.dtypes.int64).view(16)
    ring = cbd.make_ring_buffer((a, b))
    w = ring.next_write()
    w[0] = 7
    r = ring.next_read()
    y[0] = ring.depth
    y[1] = r[0]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(2, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print("depth:", y[0].item())
print("读取值:", y[1].item())
```

输入：

```text
UB 区域=[0, 128)、[128, 256) 字节，dtype=int64
向第一个槽位首元素写入 7
```

输出：

```text
depth: 2
读取值: 7
```
