---
title: global_sync_all
api_name: global_sync_all
category: resource-management
api_group: kernel
layer: sync
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# `global_sync_all`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

`global_sync_all` 是核间同步的易用性接口，一次调用完成本次 kernel 启动内全部核之间的同步：
本次启动的全部核都执行到本接口后，各核上位于本接口之后的指令才继续执行。

在同时包含 Cube 计算和 Vector 计算的核函数（Mix 算子）上，本接口一次调用完成四次核间同步，
使全部 AIC 与全部 AIV 完成全核同步：

1. 模式 2（单个 AI Core 内，AIC 与全部 AIV 之间的同步）：本 AI Core 内的全部 AIV 发送通知，
   AIC 阻塞等待通知到达。
2. 模式 0（同步等待全部 AIC 执行结束）：组间同步，同步所有的 AIC 核，
   直到所有的 AIC 核都执行到本接口。
3. 模式 0（同步等待全部 AIV 执行结束）：组间同步，同步所有的 AIV 核，
   直到所有的 AIV 核都执行到本接口。
4. 模式 2（单个 AI Core 内，AIC 与全部 AIV 之间的同步）：本 AI Core 的 AIC 发送通知，
   全部 AIV 阻塞等待通知到达。

在只包含一类核的核函数（纯 Cube 算子和纯 Vector 算子）上，参与同步的核只有一类，
本接口仅执行该类核的全核同步，即组间同步（模式 0）。

使用本接口不需要关心本次启动各核之间的对应关系，`flag_ids` 保持默认取值即可满足通常用法。
四种核间同步模式的定义请参考[系统同步能力概述](/zh/api/kernel/resource-management/system-sync-overview)。

## 函数原型

```python
def global_sync_all(
    *,
    flag_ids: tuple[int, int, int] | None = None,
) -> None: ...
```

## 参数说明

| 参数 | 输入/输出 | 类型 | 必选 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `flag_ids` | 输入 | `tuple[int, int, int]` | 否 | `(0, 1, 2)` | 依次用于第 1 次同步（模式 2）、第 2 次和第 3 次同步（模式 0）、第 4 次同步（模式 2）的三个核间同步标记，须互不相同且均在 [0, 15] 内，否则抛出 `ValueError`；每个取值各自拥有独立的 4 位计数器。仅支持关键字传参。 |

## 返回值说明

无返回值。

## 约束说明

- 必须由本次启动的全部核执行。
- 连续多次调用本接口时可直接使用默认取值：默认取值在每次同步完成之后即可再次使用。
  与手写的 `*_sync_block_arrive`/`*_sync_block_wait` 同时使用时，应显式传入一组与之错开的取值，
  避免两者的 `flag_id` 冲突。
- 每个计数器最多连续累加 15 次（此时计数器的值为 15），必须保证计数器的值不超过 15，否则触发异常。
- 本接口的流水类型为 `PIPE_S`。
- 模式 2 属于 Cube 和 Vector 混合场景，只在本 AI Core 的 AIC 与 AIV 之间成立；只包含一类核时，
  本接口仅执行该类核之间的全核同步（模式 0）。

## 调用示例

以下示例为只包含一类核的场景（纯 Vector 核函数），启动整核数 28 个 block：
阶段一每个 block 把自己的分片写入 GM workspace，随后调用 `global_sync_all` 完成一次全核同步，
阶段二读取相邻 block 在阶段一写入的分片并写回自己的输出分片，
最后在 Host 侧校验输出与输入的分片循环移位逐元素相等。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.sync import global_sync_all
from cannbotdsl.tensor import MemLoc, Tensor, tile_slice

# 模式 0 的参与者是本次启动的全部核，示例按整核数启动。
_BLOCKS = 28
# 每个 block 处理的分片大小。
_ELEMENTS = 128
_SHAPE = (1, _ELEMENTS)
_GRID = (_BLOCKS, _ELEMENTS)

@kernel
def _grid_barrier_kernel(source: Tensor, workspace: Tensor, destination: Tensor):
    ub = Buffer(MemLoc.UB, _SHAPE, dtypes.float32)
    block = get_block_idx()

    # 阶段一：本 block 把自己的分片从输入写进 GM workspace。
    mem_copy(ub, tile_slice(source, _SHAPE, (block, 0)))
    mem_copy(tile_slice(workspace, _SHAPE, (block, 0)), ub)

    # 全核同步：本次启动的全部核都完成阶段一之后，才允许任一个核进入阶段二。
    global_sync_all()

    # 阶段二：读取相邻 block 在阶段一写入的分片，写进自己的输出分片。
    mem_copy(ub, tile_slice(workspace, _SHAPE, ((block + 1) % _BLOCKS, 0)))
    mem_copy(tile_slice(destination, _SHAPE, (block, 0)), ub)

@host
def grid_barrier(source, workspace, destination):
    _grid_barrier_kernel[_BLOCKS](source, workspace, destination)

def main():
    source_cpu = torch.arange(_BLOCKS * _ELEMENTS, dtype=torch.float32).reshape(_GRID)
    expected = source_cpu.roll(-1, dims=0)
    source = source_cpu.npu()
    workspace = torch.full(_GRID, -1.0, dtype=torch.float32).npu()
    destination = torch.full(_GRID, -1.0, dtype=torch.float32).npu()
    grid_barrier(source, workspace, destination)
    torch.npu.synchronize()
    torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)
    print("global_sync_all example passed")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
global_sync_all example passed
```
