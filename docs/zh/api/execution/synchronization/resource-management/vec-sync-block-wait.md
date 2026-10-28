# vec_sync_block_wait
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

本接口等待核间同步通知，在对应`flag_id`的计数器非0后解除指定`pipe`后续指令的阻塞，并消费一次通知。各模式的计数与阻塞过程见[关键特性说明](./key-features.md)。

## 函数原型

```python
vec_sync_block_wait(
    pipe: PIPE,
    flag_id,
    mode: int = 2,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `pipe` | 输入 | `[PIPE](../../pipeline.md)`类型。标识阻塞哪条流水上后续指令，直到对应`flag_id`的计数器非0。AIV支持的取值为`PIPE.S`、`PIPE.MTE2`、`PIPE.MTE3`、`PIPE.V`。 |
| `flag_id` | 输入 | 可传入整型常量或运行时整型标量表达式。核间同步的标记，用于标识同一组同步信号。每个`flag_id`各自拥有独立的4位计数器，取值范围为[0, 15]，且须与配对发送接口的`flag_id`相同。 |
| `mode` | 输入 | 默认值为`2`。指定本次同步的参与者范围：`0`为模式0（组间同步），`1`为模式1（同一个AI Core内全部AIV之间的同步），`2`为模式2（单个AI Core内AIC与全部AIV之间的同步）。须与配对发送接口的`mode`相同。使用非默认模式时，必须显式指定该参数，否则可能导致同步操作被编译器移除。纯Vector算子支持模式0和模式1。模式2要求核函数同时包含Cube计算与Vector计算（Mix算子）。 |

## 返回值说明

无返回值。

## 约束说明

- 模式0下，每个AIV的本接口调用须与一次`vec_sync_block_arrive`配对，全部参与AIV均须发送对应通知。
- 模式1下，同一group内全部AIV均须调用`vec_sync_block_arrive`，与各AIV上的本接口调用配对。
- 模式2下，同一group内各AIV的一次本接口调用，须对应AIC的一次`cube_sync_block_arrive`。

## 调用示例

以下示例按满足目标设备单波驻留条件的block数启动（每个block含2个AIV），每个AIV先把自己的输入分片写进GM workspace，
再由`vec_sync_block_arrive`/`vec_sync_block_wait`以模式1完成一次同一AI Core内两个AIV之间的同步，
最后读取另一个AIV写入的分片并写回自己的输出分片。本示例使用Mix核函数，AIC侧只做一次数据搬运，
校验在Host侧完成：输出为输入在同一AI Core内两个AIV之间的交换结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx, get_subblock_id
from cannbotdsl.ops.sync import PIPE, vec_sync_block_arrive, vec_sync_block_wait
from cannbotdsl.tensor import MemLoc, Tensor, make_layout, tile_slice

# 模式 1 的参与者是同一 AI Core 内的全部 AIV，每个 block 含 2 个 AIV。
_BLOCKS = 28
# 每个 AIV 处理的分片大小。
_ELEMENTS = 8
_SHAPE = (1, _ELEMENTS)
_GRID = (_BLOCKS * 2, _ELEMENTS)

@kernel
def _aiv_subblock_barrier_kernel(source: Tensor, workspace: Tensor, destination: Tensor):
    ub = Buffer(MemLoc.UB, dtypes.float32, make_layout(_SHAPE))
    l1 = Buffer(MemLoc.L1, dtypes.float32, make_layout(_SHAPE), data_format="nd")

    # AIC 侧只做一次数据搬运，本示例的同步在 AIV 之间进行。
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
    print("vec-sync-block-wait example passed")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vec-sync-block-wait example passed
```
