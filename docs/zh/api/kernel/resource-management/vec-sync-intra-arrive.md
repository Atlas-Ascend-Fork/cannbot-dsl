---
title: vec_sync_intra_arrive
api_name: vec_sync_intra_arrive
category: resource-management
api_group: kernel
layer: sync
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vec_sync_intra_arrive`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

本接口与 `cube_sync_intra_wait` 配对使用，
实现[四种核间同步模式](/zh/api/kernel/resource-management/system-sync-overview)中的模式 4，
即单个 AI Core 内 AIC 与单个 AIV 之间的同步，核间同步实现的原理如下。

- AIC 等待单个 AIV 的场景：单个 AI Core 内单个 AIV 执行 `vec_sync_intra_arrive` 后向调度模块发送通知，
  接着调度模块将 AIC 对应 `sync_id` 的计数器增加 1。
  AIC 上配对的 `cube_sync_intra_wait` 检测到对应 `sync_id` 的计数器非 0 后解除阻塞，并将计数器减 1。

- 单个 AIV 等待 AIC 的场景：单个 AI Core 内 AIC 执行 `cube_sync_intra_arrive` 后向调度模块发送通知，
  接着调度模块将该 AI Core 内单个 AIV 对应 `sync_id` 的计数器增加 1。
  该 AIV 上配对的 `vec_sync_intra_wait` 检测到对应 `sync_id` 的计数器非 0 后解除阻塞，并将计数器减 1。

该同步模式的具体执行原理，请参考[关键特性说明](/zh/api/kernel/resource-management/key-features)中的代码片段及配套时序图。

## 函数原型

```python
def vec_sync_intra_arrive(
    pipe: PIPE,
    sync_id: int,
) -> None: ...
```

## 参数说明

| 参数 | 输入/输出 | 类型 | 必选 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `pipe` | 输入 | `PIPE` | 是 | 无 | 标识在哪条流水的前序指令完成后才允许向调度模块发送通知。AIV 支持的取值为 `PIPE_S`、`PIPE_MTE2`、`PIPE_MTE3`、`PIPE_V`。 |
| `sync_id` | 输入 | `int` | 是 | 无 | 核间同步的标记，用于标识同一组同步信号。每个 `sync_id` 各自拥有独立的 4 位计数器。一个 AI Core 由 1 个 AIC 与 2 个 AIV 构成，AIC 侧拥有 32 个 `sync_id`（0~31），每个 AIV 侧各拥有 16 个 `sync_id`（0~15）。同步时 `vec_sync_intra_arrive` 与 `cube_sync_intra_wait` 的 `sync_id` 对应关系请参考[约束说明](#约束说明)。 |

## 返回值说明

无返回值。

## 约束说明

- 同一 `sync_id` 不得与 `Channel` 复用。
- 与其它核间同步模式不同，本接口与 `cube_sync_intra_wait` 配对使用时，**不要求** `arrive` 与 `wait` 传入的 `sync_id` 相同，而是要求二者符合如下跨核 ID 映射关系，否则会出现未定义行为。

    - AIV0 调用的 `vec_sync_intra_arrive` 取值为 `0`~`15` 时，分别与 AIC 调用的 `cube_sync_intra_wait` 取值 `0`~`15` 对应。
    - AIV1 调用的 `vec_sync_intra_arrive` 取值为 `0`~`15` 时，分别与 AIC 调用的 `cube_sync_intra_wait` 取值 `16`~`31` 对应。

- 每个计数器最多连续累加 15 次（此时计数器的值为 15），必须保证计数器的值不超过 15，否则触发异常。
- 本接口不阻塞 `pipe` 流水中的后续指令。
- `sync_id` 取值范围为 [0, 15]，超出范围值会被按位宽截断处理为低 4 位（例如，`sync_id=16` 时，截取后为 0）。
- 本接口的流水类型为 `PIPE_S`。

## 调用示例

以下示例在同一个 Mix 核函数中完成一次两个 AIV 与 AIC 之间的数据往返：两个 AIV 各自把输入分片写进 GM workspace，再由 `vec_sync_intra_arrive` 在本地 `sync_id` `0` 上通知 AIC（AIV0 与 AIV1 的本地 `sync_id` `0` 分别对应 AIC 侧的 `sync_id` `0` 与 `16`）；AIC 用 `cube_sync_intra_wait` 在 `sync_id` `0` 与 `16` 上分别等待 AIV0 与 AIV1，把 AIV0 的分片读入 L1 后，再由 `cube_sync_intra_arrive` 分别通知两个 AIV ；两个 AIV 在 `vec_sync_intra_wait` 之后交叉读取对方写进 workspace 的分片并写回各自的输出分片。最后在 Host 侧校验输出与输入按 AIV 位次翻转后逐元素相等。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.sync import (PIPE, cube_sync_intra_arrive, cube_sync_intra_wait,
                                 vec_sync_intra_arrive, vec_sync_intra_wait)
from cannbotdsl.tensor import MemLoc, Tensor

_ELEMENTS = 8
_SHAPE = (1, _ELEMENTS)

@kernel
def _intra_round_trip_kernel(source: Tensor, workspace: Tensor, destination: Tensor):
    ub = Buffer(MemLoc.UB, _SHAPE, dtypes.float32)
    l1 = Buffer(MemLoc.L1, _SHAPE, dtypes.float32)

    # 两个 AIV 各自把输入写进 GM workspace，再在本地的 sync_id 0 上通知 AIC。
    mem_copy(ub, source)
    mem_copy(workspace, ub)
    vec_sync_intra_arrive(PIPE.MTE3, 0)

    # AIC 用 sync_id 0 等 AIV0、用 sync_id 16 等 AIV1，然后读走 workspace。
    cube_sync_intra_wait(PIPE.S, 0)
    cube_sync_intra_wait(PIPE.S, 16)
    mem_copy(l1, workspace)

    # AIC 分别通知两个 AIV，AIV 在本地的 sync_id 0 上等待。
    cube_sync_intra_arrive(PIPE.MTE2, 0)
    cube_sync_intra_arrive(PIPE.MTE2, 16)
    vec_sync_intra_wait(PIPE.S, 0)

    mem_copy(destination, ub)

@host
def intra_round_trip(source, workspace, destination):
    _intra_round_trip_kernel[1](source, workspace, destination)

def main():
    source_cpu = torch.arange(_ELEMENTS, dtype=torch.float32).reshape(_SHAPE)
    source = source_cpu.npu()
    workspace = torch.full(_SHAPE, -1.0, dtype=torch.float32).npu()
    destination = torch.zeros(_SHAPE, dtype=torch.float32).npu()
    intra_round_trip(source, workspace, destination)
    torch.npu.synchronize()
    torch.testing.assert_close(destination.cpu(), source_cpu, rtol=0, atol=0)
    print("vec-sync-intra-arrive example passed")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vec-sync-intra-arrive example passed
```
