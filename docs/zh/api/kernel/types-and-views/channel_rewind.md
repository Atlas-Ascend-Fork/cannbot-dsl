---
title: channel_rewind
api_name: channel_rewind
category: types-and-views
api_group: kernel
layer: frontend
call_context: device
execution_unit: compile-time
status: experimental
since: 待追溯
---

# `channel_rewind()`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

`channel_rewind()` 在当前 Kernel 中插入一个片上内存重新规划边界。调用后，框架会
从该边界开始为新声明的 Channel 重新安排片上内存；边界之前各 Channel 使用过的
内存位置可以再次分配给新的 Channel。

从资源规划角度看，调用 `channel_rewind()` 后，边界之前创建的 Channel 生命周期
已经结束，不应再调用其 `produce()` 或 `consume()`。这些 Channel 原来使用的片上
内存可能已经分配给新的 Channel；继续访问旧 Channel 会与新 Channel 访问同一段
内存，导致数据覆盖或内存踩踏。

`channel_rewind()` 只声明片上内存可以重新规划，不会生成设备侧同步指令，也不会
等待边界之前发起的搬运或计算完成。调用前必须由开发者完成必要的同步，确保旧
Channel 相关操作已经结束。

## 函数原型

```python
channel_rewind(reset_sync_id: bool = True) -> None
```

## 参数说明

1. `reset_sync_id=True`：在重新规划片上内存的同时，将该位置作为新的同步 ID
   分配阶段。边界前已经完成同步流程的 ID 可以重新分配给后续 Channel；该参数
   只影响同步 ID 的分配，不会执行同步操作。
2. `reset_sync_id=False`：只重新规划片上内存，不额外开启同步 ID 分配阶段。边界
   前后的 Channel 仍按同一阶段分配同步 ID，不能依赖本次 `rewind` 释放同步 ID。

## 返回值说明

无返回值。

## 约束说明

- `reset_sync_id` 必须是编译期确定的 Python `bool`。`0`、`1`、运行时布尔值及其他
  类型均不支持。
- 启用 `dsl.UB.view()` 的显式 UB 空间模式后不支持调用 `channel_rewind()`。
- 该接口必须在 `@cannbotdsl.kernel` 的设备代码中调用，也可以写在由该 Kernel
  调用的 `@cannbotdsl.jit` 辅助函数中；不能作为普通 Host 侧 Python 函数单独执行。

## 调用示例

```python
import torch
from torch import as_tensor as from_torch_npu

import cannbotdsl as cbd
from cannbotdsl import dtypes


@cbd.kernel
def reuse_kernel(x0: cbd.Tensor, x1: cbd.Tensor, y0: cbd.Tensor, y1: cbd.Tensor):
    first = cbd.Channel(
        cbd.MemLoc.UB,
        (4,),
        dtypes.float32,
        depth=1,
    )
    cbd.mem_copy(first.produce(), x0)
    cbd.mem_copy(y0, first.consume())

    # 先完成第一阶段，再声明地址复用边界。
    cbd.vec_sync_all()
    cbd.channel_rewind(reset_sync_id=False)

    # 第二阶段重新规划第一阶段已经不再使用的 UB 空间。
    second = cbd.Channel(
        cbd.MemLoc.UB,
        (4,),
        dtypes.float32,
        depth=1,
    )
    cbd.mem_copy(second.produce(), x1)
    cbd.mem_copy(y1, second.consume())


@cbd.host
def run(x0, x1, y0, y1):
    reuse_kernel[1](x0, x1, y0, y1)


if __name__ == "__main__":
    x0 = torch.tensor([1.0, 2.0, 3.0, 4.0], device="npu")
    x1 = torch.tensor([5.0, 6.0, 7.0, 8.0], device="npu")
    y0 = torch.empty_like(x0)
    y1 = torch.empty_like(x1)
    run(
        from_torch_npu(x0),
        from_torch_npu(x1),
        from_torch_npu(y0),
        from_torch_npu(y1),
    )
    torch.npu.synchronize()
    torch.testing.assert_close(y0, x0)
    torch.testing.assert_close(y1, x1)
    print("y0:", y0.cpu().tolist())
    print("y1:", y1.cpu().tolist())
```

### 示例输入

```text
x0 = [1.0, 2.0, 3.0, 4.0]
x1 = [5.0, 6.0, 7.0, 8.0]
第一阶段 Channel：MemLoc.UB，depth=1
第二阶段 Channel：MemLoc.UB，depth=1
```

### 预期输出

```text
y0: [1.0, 2.0, 3.0, 4.0]
y1: [5.0, 6.0, 7.0, 8.0]
```

第一阶段完成并同步后，第二阶段通过 `channel_rewind()` 重新规划可用的 UB 空间；
两个阶段的输出分别保留各自输入数据。
