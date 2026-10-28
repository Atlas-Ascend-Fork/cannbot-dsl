# cube_sync_block_wait
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

本接口等待核间同步通知，在对应`flag_id`的计数器非0后解除指定`pipe`后续指令的阻塞，并消费一次通知。各模式的计数与阻塞过程见[关键特性说明](./key-features.md)。

## 函数原型

```python
cube_sync_block_wait(
    pipe: PIPE,
    flag_id,
    mode: int = 2,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `pipe` | 输入 | `[PIPE](../../pipeline.md)`类型。标识阻塞哪条流水上后续指令，直到对应`flag_id`的计数器非0。AIC支持的取值为`PIPE.S`、`PIPE.M`、`PIPE.MTE1`、`PIPE.MTE2`、`PIPE.FIXPIPE`。 |
| `flag_id` | 输入 | 可传入整型常量或运行时整型标量表达式。核间同步的标记，用于标识同一组同步信号。每个`flag_id`各自拥有独立的4位计数器，取值范围为[0, 15]，且须与配对发送接口的`flag_id`相同。 |
| `mode` | 输入 | 默认值为`2`。指定本次同步的参与者范围：`0`为模式0（组间同步），`2`为模式2（单个AI Core内AIC与全部AIV之间的同步）；模式1的两个参与者都是AIV，AIC侧不适用。须与配对发送接口的`mode`相同。使用非默认模式时，必须显式指定该参数，否则可能导致同步操作被编译器移除。纯Cube算子支持模式0。模式2要求核函数同时包含Cube计算与Vector计算（Mix算子）。 |

## 返回值说明

无返回值。

## 约束说明

- 模式0下，每个AIC的本接口调用须与一次`cube_sync_block_arrive`配对，全部参与AIC均须发送对应通知。
- 模式2下，AIC的一次本接口调用须对应同一group内各AIV的一次`vec_sync_block_arrive`，不能只由部分AIV发送通知。

## 调用示例

以下示例按满足目标设备单波驻留条件的block数启动，每个AIC把自己负责的一对16×16的A/B分片搬入L1、L0A/L0B并算出
C分片，随后由`cube_sync_block_arrive`/`cube_sync_block_wait`完成一次全部AIC的全核同步，
最后把C分片写回GM。最后在Host侧校验各分片输出与`A @ B^T`在指定误差范围内一致。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.matmul import matmul as dsl_matmul
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.sync import PIPE, cube_sync_all, cube_sync_block_arrive, cube_sync_block_wait
from cannbotdsl.tensor import MemLoc, Tensor, make_layout, tile_slice

# 模式 0 的参与者是本次启动的全部 AIC，所有参与 block 需要同时驻留执行。
_BLOCKS = 28
# 每个 block 负责一对 16×16 的 A/B 分片。
_TILE = 16

@kernel
def _grid_barrier_kernel(source_a: Tensor, source_b: Tensor, destination: Tensor):
    l1_a = Buffer(MemLoc.L1, dtypes.float16, make_layout((_TILE, _TILE)), data_format="nz")
    l1_b = Buffer(MemLoc.L1, dtypes.float16, make_layout((_TILE, _TILE)), data_format="nz")
    l0a = Buffer(MemLoc.L0A, dtypes.float16, make_layout((_TILE, _TILE)), data_format="nz")
    l0b = Buffer(MemLoc.L0B, dtypes.float16, make_layout((_TILE, _TILE)), data_format="nz")
    l0c = Buffer(MemLoc.L0C, dtypes.float32, make_layout((_TILE, _TILE)), data_format="nz")
    block = get_block_idx()

    # 阶段一：本 AIC 把自己的 A/B 分片搬进 L1，再经 L0A/L0B 算出 C 分片。
    mem_copy(l1_a, tile_slice(source_a, (_TILE, _TILE), (block, 0)))
    mem_copy(l1_b, tile_slice(source_b, (_TILE, _TILE), (block, 0)))
    mem_copy(l0a, l1_a)
    mem_copy(l0b, l1_b)
    l0c_acc = dsl_matmul(l0c, l0a, l0b, init=True)

    # arrive 只等待指定流水，先确保矩阵计算及其他前序流水操作全部完成。
    cube_sync_all()

    # 全核同步：本次启动的全部 AIC 都完成阶段一之后，才允许任一个 AIC 进入阶段二。
    cube_sync_block_arrive(PIPE.FIXPIPE, 0, mode=0)
    cube_sync_block_wait(PIPE.S, 0, mode=0)

    # 阶段二：把 C 分片写回 GM。
    mem_copy(tile_slice(destination, (_TILE, _TILE), (block, 0)), l0c_acc)

@host
def grid_barrier(source_a, source_b, destination):
    _grid_barrier_kernel[_BLOCKS](source_a, source_b, destination)

def main():
    torch.manual_seed(0)
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
    print("cube-sync-block-wait example passed")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
cube-sync-block-wait example passed
```
