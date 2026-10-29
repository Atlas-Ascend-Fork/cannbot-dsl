# Add

## 产品支持情况

- Ascend 950：支持，使用 SIMT 实现。

## 功能说明

`add` 对两个形状相同的 tensor 逐元素相加，计算 `y[i] = x1[i] + x2[i]`，返回独立分配的输出。实现位于 [`add.py`](add.py)。

输入须为同一 NPU 上、shape 完全相同的非空连续 tensor，支持零维 tensor。不支持广播、非连续布局、空张量、原位计算、`out`、`alpha` 或自动求导。

| 输入类型 | 输出类型 |
| --- | --- |
| 两输入均为 float32 | float32 |
| 两输入均为 int32 | int32 |

两输入必须类型相同，不进行类型提升。float32 使用单精度浮点加法；int32 使用 32 位整数加法，溢出按位宽回绕。

## 函数原型

```python
add(x1, x2, *, core_num=None, thread_num=None)
```

`core_num` 为可选 Vector Core 数上限，默认按平台与工作量选择。`thread_num` 支持 32、64、128、256、512、1024，默认 1024。

## 实现说明

所有合法输入统一使用 AddDmaKernel，按 **GM → UB → SIMT 相加 → GM** 执行，不按元素数量切换内核。

Tiling 按平台核数、线程数、UB 容量及各核工作量选择核数和 tile。各核区间按 32 字节单位分配，
最后一核与最后一块按真实有效长度裁剪。三条 UB RingBuffer 各使用两个槽位，组织预取、计算、搬出与排空。
标量、小于 32 字节的输入和连续存储偏移视图均使用同一搬运流程。

## 调用示例

```python
import torch
import torch_npu
from samples.add.add import add

x1 = torch.randn(2, 4, 128, dtype=torch.float32).npu()
x2 = torch.randn_like(x1)
y = add(x1, x2)
```
