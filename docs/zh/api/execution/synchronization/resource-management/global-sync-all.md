# global_sync_all
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

本接口先等待各执行核的前序流水操作完成，再完成本次Kernel启动内全部参与核之间的同步。全部参与核到达同步点后，各核才继续执行后续指令；Mix算子同步AIC与AIV，纯Cube或纯Vector算子同步对应核类型。各模式的组成见[系统同步能力概述](./system-sync-overview.md)。

## 函数原型

```python
global_sync_all(
    *,
    flag_ids = None,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `flag_ids` | 输入 | `list[int]`、`tuple[int, int, int]`或`None`，默认值为`None`。未指定时使用`(0, 1, 2)`。依次用于AIV向AIC通知（模式2）、AIC和AIV各自的全核同步（模式0）、AIC向AIV通知（模式2）的三个核间同步标记，须互不相同且均在[0, 15] 内，否则抛出`ValueError`；每个取值各自拥有独立的4位计数器。仅支持关键字传参。每次屏障完成后可复用同一组取值，连续调用时无需更换默认值。与手工block同步协议重叠时，须选用与该协议在同一计数器命名空间内不冲突的三个ID；默认值冲突时须显式指定。 |

## 返回值说明

无返回值。

## 约束说明

- 必须由本次启动的全部参与核执行，各参与核的调用次数必须一致，不能仅在部分参与核执行的分支中调用。

## 调用示例

以下示例为只包含一类核的场景（纯Vector核函数），按满足目标设备单波驻留条件的block数启动：
阶段一每个block把自己的分片写入GM workspace，随后调用`global_sync_all`完成一次全核同步，
阶段二读取相邻block在阶段一写入的分片并写回自己的输出分片，
最后在Host侧校验输出与输入的分片循环移位逐元素相等。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.sync import global_sync_all
from cannbotdsl.tensor import MemLoc, Tensor, make_layout, tile_slice

# 模式 0 的参与者是本次启动的全部核，所有参与 block 需要同时驻留执行。
_BLOCKS = 28
# 每个 block 处理的分片大小。
_ELEMENTS = 128
_SHAPE = (1, _ELEMENTS)
_GRID = (_BLOCKS, _ELEMENTS)

@kernel
def _grid_barrier_kernel(source: Tensor, workspace: Tensor, destination: Tensor):
    ub = Buffer(MemLoc.UB, dtypes.float32, make_layout(_SHAPE))
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
