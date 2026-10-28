# cube_sync_intra_wait
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

本接口等待同一group内指定AIV的同步通知，在对应计数器非0后解除指定`pipe`后续指令的阻塞，并消费一次通知。同步过程见[关键特性说明](./key-features.md)。

## 函数原型

```python
cube_sync_intra_wait(
    pipe: PIPE,
    sync_id,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `pipe` | 输入 | `[PIPE](../../pipeline.md)`类型。标识阻塞哪条流水上后续指令，直到对应`sync_id`的计数器非0。AIC支持的取值为`PIPE.S`、`PIPE.M`、`PIPE.MTE1`、`PIPE.MTE2`、`PIPE.FIXPIPE`。 |
| `sync_id` | 输入 | 可传入整型常量或运行时整型标量表达式。核间同步的标记，用于标识同一组同步信号。每个`sync_id`各自拥有独立的4位计数器。一个AI Core由1个AIC与2个AIV构成，AIC侧拥有32个`sync_id`（0~31），每个AIV侧各拥有16个`sync_id`（0~15）。同步时`vec_sync_intra_arrive`与`cube_sync_intra_wait`的`sync_id`对应关系请参考[约束说明](#约束说明)。与框架自动生成的跨核同步混用时，编译期必须能够确定该参数的取值范围，否则编译报错；不要求该参数必须为常量。 |

## 返回值说明

无返回值。

## 约束说明

- 本接口须与`vec_sync_intra_arrive`配对使用。AIV向AIC发送通知时，AIV0的本地`sync_id` 0～15对应AIC的`sync_id` 0～15，AIV1的本地`sync_id` 0～15对应AIC的`sync_id` 16～31；每次发送须对应目标接收核的一次等待。

## 调用示例

以下示例在同一个Mix核函数中完成一次AIC与两个AIV之间的同步握手：两个AIV各自把输入分片写进GM workspace，再由`vec_sync_intra_arrive`在本地`sync_id` `0`上通知AIC（AIV0与AIV1的本地`sync_id` `0`分别对应AIC侧的`sync_id` `0`与`16`）；AIC用`cube_sync_intra_wait`在`sync_id` `0`与`16`上分别等待AIV0与AIV1，把AIV0的分片读入L1后，再由`cube_sync_intra_arrive`分别通知两个AIV；两个AIV在`vec_sync_intra_wait`之后交叉读取对方写进workspace的分片并写回各自的输出分片。最后在Host侧校验输出交换了两个AIV对应的输入分片。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_subblock_id
from cannbotdsl.ops.sync import (PIPE, cube_sync_intra_arrive, cube_sync_intra_wait,
                                 vec_sync_intra_arrive, vec_sync_intra_wait)
from cannbotdsl.tensor import MemLoc, Tensor, make_layout, tile_slice

_ELEMENTS = 128
_SHAPE = (1, _ELEMENTS)
_GRID = (2, _ELEMENTS)

@kernel
def _intra_round_trip_kernel(source: Tensor, workspace: Tensor, destination: Tensor):
    ub = Buffer(MemLoc.UB, dtypes.float32, make_layout(_SHAPE))
    l1 = Buffer(MemLoc.L1, dtypes.float32, make_layout(_SHAPE), data_format="nd")

    own = get_subblock_id()
    peer = 1 - own

    # 两个 AIV 写入各自的 workspace 分片，再在本地 sync_id 0 上通知 AIC。
    mem_copy(ub, tile_slice(source, _SHAPE, (own, 0)))
    mem_copy(tile_slice(workspace, _SHAPE, (own, 0)), ub)
    vec_sync_intra_arrive(PIPE.MTE3, 0)

    # AIC 用 sync_id 0 等 AIV0、用 sync_id 16 等 AIV1，然后读取 AIV0 的 workspace 分片。
    cube_sync_intra_wait(PIPE.S, 0)
    cube_sync_intra_wait(PIPE.S, 16)
    mem_copy(l1, tile_slice(workspace, _SHAPE, (0, 0)))

    # AIC 分别通知两个 AIV，AIV 在本地的 sync_id 0 上等待。
    cube_sync_intra_arrive(PIPE.MTE2, 0)
    cube_sync_intra_arrive(PIPE.MTE2, 16)
    vec_sync_intra_wait(PIPE.S, 0)

    # 等待 AIC 回应后，读取另一个 AIV 的分片。
    mem_copy(ub, tile_slice(workspace, _SHAPE, (peer, 0)))
    mem_copy(tile_slice(destination, _SHAPE, (own, 0)), ub)

@host
def intra_round_trip(source, workspace, destination):
    _intra_round_trip_kernel[1](source, workspace, destination)

def main():
    source_cpu = torch.arange(2 * _ELEMENTS, dtype=torch.float32).reshape(_GRID)
    expected = source_cpu.flip(0)
    source = source_cpu.npu()
    workspace = torch.full(_GRID, -1.0, dtype=torch.float32).npu()
    destination = torch.zeros(_GRID, dtype=torch.float32).npu()
    intra_round_trip(source, workspace, destination)
    torch.npu.synchronize()
    torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)
    print("cube-sync-intra-wait example passed")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
cube-sync-intra-wait example passed
```
