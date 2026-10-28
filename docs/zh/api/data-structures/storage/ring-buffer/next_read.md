# RingBuffer.next_read

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

返回当前读游标对应槽位的 Tensor 或 Storage，再将该读游标推进一个位置；经过 `self.depth` 个槽位后回到第一个槽位。

## 函数原型

```python
next_read(self, *, bare=False) -> Tensor | Storage
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `bare` | 输入 | 返回形式，类型为 Python `bool`，默认 `False`。`False` 返回采用声明布局的 Tensor；`True` 返回 [Storage](../storage/overview.md)，可通过其 `view()` 绑定布局。 |

## 返回值说明

返回选中槽位的 Tensor 或 Storage。Tensor 的元素类型及布局取自 RingBuffer 的声明，Storage 保存该槽位的容量和默认视图。

## 约束说明

无

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    ring = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.int64, cbd.make_layout((4,)), depth=2)
    w0 = ring.next_write()
    w0[0] = 11
    w1 = ring.next_write()
    w1[0] = 22
    r0 = ring.next_read()
    r1 = ring.next_read()
    r2 = ring.next_read()
    y[0] = r0[0]
    y[1] = r1[0]
    y[2] = r2[0]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(3, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print(y.cpu().tolist())
```

输入：

```text
depth=2, 单槽 shape=(4,), dtype=int64
第 0、1 槽首元素分别写入 11、22，连续选槽三次
```

输出：

```text
[11, 22, 11]
```
