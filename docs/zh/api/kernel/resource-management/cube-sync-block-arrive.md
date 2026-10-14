---
title: cube_sync_block_arrive
api_name: cube_sync_block_arrive
category: resource-management
api_group: kernel
layer: sync
call_context: device
execution_unit: cube
status: experimental
since: 待追溯
---

# `cube_sync_block_arrive`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

本接口与 `cube_sync_block_wait` 配对使用，实现核间同步。
根据 `mode` 的取值，分别对应[四种核间同步模式](/zh/api/kernel/resource-management/system-sync-overview)中的模式 0 和模式 2：
`mode` 为 `2` 时实现单个 AI Core 内 AIC 与全部 AIV 之间的同步，`mode` 为 `0` 时实现不同 AI Core 之间全部 AIC 的同步。
核间同步实现的原理如下。

- 模式 2，单个 AI Core 内 AIC 与全部 AIV 之间的同步：

    - 全部 AIV 等待单个 AIC 的场景：单个 AI Core 内 AIC 执行 `cube_sync_block_arrive` 后向调度模块发送通知，
      接着调度模块将该 AI Core 内全部 AIV 各自对应 `flag_id` 的计数器增加 1。
      各 AIV 上配对的 `vec_sync_block_wait` 检测到对应 `flag_id` 的计数器非 0 后解除阻塞，并将计数器减 1。

    - 单个 AIC 等待全部 AIV 的场景：该 AI Core 内全部 AIV 都执行 `vec_sync_block_arrive` 后向调度模块发送通知，
      接着调度模块将 AIC 对应 `flag_id` 的计数器增加 1。
      AIC 上配对的 `cube_sync_block_wait` 检测到对应 `flag_id` 的计数器非 0 后解除阻塞，并将计数器减 1。

- 模式 0，不同 AI Core 之间全部 AIC 之间的同步：

    - 全部 AIC 之间的同步场景：所有参与同步的 AIC 都执行 `cube_sync_block_arrive` 后向调度模块发送通知，
      接着调度模块将各 AIC 对应 `flag_id` 的计数器增加 1。
      各 AIC 上配对的 `cube_sync_block_wait` 检测到对应 `flag_id` 的计数器非 0 后解除阻塞，并将计数器减 1。

各模式具体的执行原理，请参考[关键特性说明](/zh/api/kernel/resource-management/key-features)中的代码片段及配套时序图。

## 函数原型

```python
def cube_sync_block_arrive(
    pipe: PIPE,
    flag_id: int,
    mode: int = 2,
) -> None: ...
```

## 参数说明

| 参数 | 输入/输出 | 类型 | 必选 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `pipe` | 输入 | `PIPE` | 是 | 无 | 标识在哪条流水的前序指令完成后才允许向调度模块发送通知。AIC 支持的取值为 `PIPE_M`、`PIPE_MTE1`、`PIPE_MTE2`、`PIPE_FIXPIPE`，配对等待侧还额外支持 `PIPE_S`。 |
| `flag_id` | 输入 | `int` | 是 | 无 | 核间同步的标记，用于标识同一组同步信号。每个 `flag_id` 各自拥有独立的 4 位计数器，取值范围为 [0, 15]，且须与配对等待接口的 `flag_id` 相同。 |
| `mode` | 输入 | `int` | 否 | `2` | 指定本次同步的参与者范围：`0` 为模式 0（组间同步），`2` 为模式 2（单个 AI Core 内 AIC 与全部 AIV 之间的同步）；模式 1 的两个参与者都是 AIV，AIC 侧不适用。须与配对等待接口的 `mode` 相同。 |

## 返回值说明

无返回值。

## 约束说明

- 用户需要确保配套使用（`flag_id` 必须完全一致，`mode` 必须相同）`cube_sync_block_arrive` 和 `cube_sync_block_wait`，
  否则会出现未定义行为；模式 2 下 AIV 侧为 `vec_sync_block_wait`。
- 每个计数器最多连续累加 15 次（此时计数器的值为 15），必须保证计数器的值不超过 15，否则触发异常。
- 本接口不阻塞 `pipe` 流水中的后续指令。
- `flag_id` 取值范围为 [0, 15]。为常量时越界会抛出 `ValueError`；为动态值时不做取值校验，
  超出范围值会被按位宽截断处理为低 4 位（例如，`flag_id=16` 时，截取后为 0，与 `flag_id=0` 共用同一个计数器）。
- 本接口的流水类型为 `PIPE_S`。
- 纯 Cube 场景下支持模式 0。模式 2 属于 Cube 和 Vector 混合场景，
  调用本接口的核函数需要同时包含 Cube 计算与 Vector 计算（Mix 算子）。

## 调用示例

以下示例启动整核数 28 个 block，每个 AIC 把自己负责的一对 16×16 的 A/B 分片搬入 L1、L0A/L0B 并算出
C 分片，随后由 `cube_sync_block_arrive`/`cube_sync_block_wait` 完成一次全部 AIC 的全核同步，
最后把 C 分片写回 GM。最后在 Host 侧校验输出与 `A @ B^T` 逐元素相等。

```python
import torch
import torch_npu

from cannbotdsl import dtypes, host
from cannbotdsl.channel import Channel
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.matmul import matmul as dsl_matmul
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy
from cannbotdsl.ops.sync import PIPE, cube_sync_block_arrive, cube_sync_block_wait
from cannbotdsl.tensor import MemLoc, Tensor, tile_slice

# 模式 0 的参与者是本次启动的全部 AIC，示例按整核数启动。
_BLOCKS = 28
# 每个 block 负责一对 16×16 的 A/B 分片。
_TILE = 16

@kernel
def _grid_barrier_kernel(source_a: Tensor, source_b: Tensor, destination: Tensor):
    l1_a = Channel(MemLoc.L1, (_TILE, _TILE), dtypes.float16, depth=1)
    l1_b = Channel(MemLoc.L1, (_TILE, _TILE), dtypes.float16, depth=1)
    l0a = Channel(MemLoc.L0A, (_TILE, _TILE), dtypes.float16, depth=1)
    l0b = Channel(MemLoc.L0B, (_TILE, _TILE), dtypes.float16, depth=1)
    l0c = Channel(MemLoc.L0C, (_TILE, _TILE), dtypes.float32, depth=1)
    nd2nz = make_copy_engine(format_transform="nd2nz")
    block = get_block_idx()

    # 阶段一：本 AIC 把自己的 A/B 分片搬进 L1，再经 L0A/L0B 算出 C 分片。
    mem_copy(l1_a.produce(), tile_slice(source_a, (_TILE, _TILE), (block, 0)), engine=nd2nz)
    l1_a_tensor = l1_a.consume()
    mem_copy(l1_b.produce(), tile_slice(source_b, (_TILE, _TILE), (block, 0)), engine=nd2nz)
    l1_b_tensor = l1_b.consume()
    mem_copy(l0a.produce(), l1_a_tensor)
    l0a_tensor = l0a.consume()
    mem_copy(l0b.produce(), l1_b_tensor)
    l0b_tensor = l0b.consume()
    l0c_acc = l0c.produce()
    dsl_matmul(l0c_acc, l0a_tensor, l0b_tensor, init=True)

    # 全核同步：本次启动的全部 AIC 都完成阶段一之后，才允许任一个 AIC 进入阶段二。
    cube_sync_block_arrive(PIPE.FIXPIPE, 0, mode=0)
    cube_sync_block_wait(PIPE.S, 0, mode=0)

    # 阶段二：把 C 分片写回 GM。
    mem_copy(tile_slice(destination, (_TILE, _TILE), (block, 0)), l0c_acc)

@host
def grid_barrier(source_a, source_b, destination):
    _grid_barrier_kernel[_BLOCKS](source_a, source_b, destination)

def main():
    rows = _BLOCKS * _TILE
    a_cpu = torch.randn(rows, _TILE, dtype=torch.float16)
    b_cpu = torch.randn(rows, _TILE, dtype=torch.float16)
    expected = torch.matmul(a_cpu.float().view(_BLOCKS, _TILE, _TILE),
                            b_cpu.float().view(_BLOCKS, _TILE, _TILE).transpose(1, 2))
    source_a, source_b = a_cpu.npu(), b_cpu.npu()
    destination = torch.zeros(rows, _TILE, dtype=torch.float32).npu()
    grid_barrier(source_a, source_b, destination)
    torch.npu.synchronize()
    torch.testing.assert_close(destination.cpu().view(_BLOCKS, _TILE, _TILE),
                               expected, rtol=1e-3, atol=1e-3)
    print("cube-sync-block-arrive example passed")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
cube-sync-block-arrive example passed
```
