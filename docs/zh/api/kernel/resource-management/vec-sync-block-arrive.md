---
title: vec_sync_block_arrive
api_name: vec_sync_block_arrive
category: resource-management
api_group: kernel
layer: sync
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vec_sync_block_arrive`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

本接口与 `vec_sync_block_wait` 配对使用，实现核间同步。
根据 `mode` 的取值，分别对应[四种核间同步模式](/zh/api/kernel/resource-management/system-sync-overview)中的模式 0、模式 1 和模式 2，
核间同步实现的原理如下。

- 模式 0，组间同步，即不同 group 之间所有 AIV 之间的同步：

    - 全部 AIV 之间的同步场景：所有参与同步的 AIV 都执行 `vec_sync_block_arrive` 后向调度模块发送通知，
      接着调度模块将各 AIV 对应 `flag_id` 的计数器增加 1。
      各 AIV 上配对的 `vec_sync_block_wait` 检测到对应 `flag_id` 的计数器非 0 后解除阻塞，并将计数器减 1。

- 模式 1，同一个 AI Core 内全部 AIV 之间的同步：

    - 同一 AI Core 内所有 AIV（subblock）都执行 `vec_sync_block_arrive` 后向调度模块发送通知，
      接着调度模块将各 AIV（subblock）对应 `flag_id` 的计数器增加 1。
      各 AIV（subblock）上配对的 `vec_sync_block_wait` 检测到对应 `flag_id` 的计数器非 0 后解除阻塞，并将计数器减 1。

- 模式 2，单个 AI Core 内 AIC 与全部 AIV 之间的同步：

    - AIC 等待全部 AIV 的场景：该 AI Core 内全部 AIV 都执行 `vec_sync_block_arrive` 后向调度模块发送通知，
      接着调度模块将 AIC 对应 `flag_id` 的计数器增加 1。
      AIC 上配对的 `cube_sync_block_wait` 检测到对应 `flag_id` 的计数器非 0 后解除阻塞，并将计数器减 1。

    - 全部 AIV 等待 AIC 的场景：该 AI Core 内 AIC 执行 `cube_sync_block_arrive` 后向调度模块发送通知，
      接着调度模块将该 AI Core 内全部 AIV 各自对应 `flag_id` 的计数器增加 1。
      各 AIV 上配对的 `vec_sync_block_wait` 检测到对应 `flag_id` 的计数器非 0 后解除阻塞，并将计数器减 1。

各模式具体的执行原理，请参考[关键特性说明](/zh/api/kernel/resource-management/key-features)中的代码片段及配套时序图。

## 函数原型

```python
def vec_sync_block_arrive(
    pipe: PIPE,
    flag_id: int,
    mode: int = 2,
) -> None: ...
```

## 参数说明

| 参数 | 输入/输出 | 类型 | 必选 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `pipe` | 输入 | `PIPE` | 是 | 无 | 标识在哪条流水的前序指令完成后才允许向调度模块发送通知。AIV 支持的取值为 `PIPE_MTE2`、`PIPE_MTE3`、`PIPE_V`，配对等待侧还额外支持 `PIPE_S`。 |
| `flag_id` | 输入 | `int` | 是 | 无 | 核间同步的标记，用于标识同一组同步信号。每个 `flag_id` 各自拥有独立的 4 位计数器，取值范围为 [0, 15]，且须与配对等待接口的 `flag_id` 相同。 |
| `mode` | 输入 | `int` | 否 | `2` | 指定本次同步的参与者范围：`0` 为模式 0（组间同步），`1` 为模式 1（同一 AI Core 内全部 AIV），`2` 为模式 2（同一 AI Core 内 AIC 与全部 AIV）。须与配对等待接口的 `mode` 相同。 |

## 返回值说明

无返回值。

## 约束说明

- 用户需要确保配套使用（`flag_id` 必须完全一致，`mode` 必须相同）`vec_sync_block_arrive` 和 `vec_sync_block_wait`，
  否则会出现未定义行为；模式 2 下 AIC 侧为 `cube_sync_block_wait`。
- 每个计数器最多连续累加 15 次（此时计数器的值为 15），必须保证计数器的值不超过 15，否则触发异常。
- 本接口不阻塞 `pipe` 流水中的后续指令。
- `flag_id` 取值范围为 [0, 15]。为常量时越界会抛出 `ValueError`；为动态值时不做取值校验，
  超出范围值会被按位宽截断处理为低 4 位（例如，`flag_id=16` 时，截取后为 0，与 `flag_id=0` 共用同一个计数器）。
- 本接口的流水类型为 `PIPE_S`。
- 纯 Vector 场景下支持模式 0 和模式 1。模式 2 属于 Cube 和 Vector 混合场景，
  调用本接口的核函数需要同时包含 Cube 计算与 Vector 计算（Mix 算子）。

## 调用示例

以下示例启动整核数 28 个 block（每个 block 含 2 个 AIV），每个 AIV 先把自己的输入分片写进 GM workspace，
再由 `vec_sync_block_arrive`/`vec_sync_block_wait` 以模式 1 完成一次同一 AI Core 内两个 AIV 之间的同步，
最后读取另一个 AIV 写入的分片并写回自己的输出分片。本示例使用 Mix 核函数，AIC 侧只做一次数据搬运，
校验在 AIV 侧完成：输出为输入在同一 AI Core 内两个 AIV 之间的交换结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx, get_subblock_id
from cannbotdsl.ops.sync import PIPE, vec_sync_block_arrive, vec_sync_block_wait
from cannbotdsl.tensor import MemLoc, Tensor, tile_slice

# 模式 1 的参与者是同一 AI Core 内的全部 AIV，示例按整核数启动，每个 block 含 2 个 AIV。
_BLOCKS = 28
# 每个 AIV 处理的分片大小。
_ELEMENTS = 8
_SHAPE = (1, _ELEMENTS)
_GRID = (_BLOCKS * 2, _ELEMENTS)

@kernel
def _aiv_subblock_barrier_kernel(source: Tensor, workspace: Tensor, destination: Tensor):
    ub = Buffer(MemLoc.UB, _SHAPE, dtypes.float32)
    l1 = Buffer(MemLoc.L1, _SHAPE, dtypes.float32)

    # AIC 侧只做一次数据搬运，本示例的同步与校验都在 AIV 之间进行。
    mem_copy(l1, tile_slice(source, _SHAPE, (get_block_idx(), 0)))

    # 每个 block 的 2 个 AIV 各有自己的槽位，读取对方槽位时按位次取反。
    own = get_block_idx() * 2 + get_subblock_id()
    peer = get_block_idx() * 2 + (1 - get_subblock_id())

    # 阶段一：本 AIV 把自己的输入分片写进 GM workspace 的槽位。
    mem_copy(ub, tile_slice(source, _SHAPE, (own, 0)))
    mem_copy(tile_slice(workspace, _SHAPE, (own, 0)), ub)

    # 模式 1：同一 AI Core 内的两个 AIV 都写完自己的槽位之后，才允许读取对方的槽位。
    vec_sync_block_arrive(PIPE.MTE3, 0, mode=1)
    vec_sync_block_wait(PIPE.S, 0, mode=1)

    # 阶段二：读对方槽位的数据，写进自己的输出分片。
    mem_copy(ub, tile_slice(workspace, _SHAPE, (peer, 0)))
    mem_copy(tile_slice(destination, _SHAPE, (own, 0)), ub)

@host
def aiv_subblock_barrier(source, workspace, destination):
    _aiv_subblock_barrier_kernel[_BLOCKS](source, workspace, destination)

def main():
    source_cpu = torch.arange(_GRID[0] * _GRID[1], dtype=torch.float32).reshape(_GRID)
    expected = source_cpu.view(_BLOCKS, 2, _ELEMENTS).flip(1).reshape(_GRID)
    source = source_cpu.npu()
    workspace = torch.full(_GRID, -1.0, dtype=torch.float32).npu()
    destination = torch.full(_GRID, -1.0, dtype=torch.float32).npu()
    aiv_subblock_barrier(source, workspace, destination)
    torch.npu.synchronize()
    torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)
    print("vec-sync-block-arrive example passed")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vec-sync-block-arrive example passed
```
