# get_block_idx
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

获取当前执行Kernel的逻辑核索引。

## 函数原型

```python
get_block_idx() -> Int64
```

## 参数说明

无。

## 返回值说明

返回当前逻辑核的索引，取值范围为 `[0, get_block_num())`，类型为 `Int64`。逻辑核数见 [get_block_num](./get-block-num.md)。

在Mix 1:2模式下，同一逻辑核组合中的两个Vector Core返回相同的block索引。如需区分这两个子核，可结合 [get_subblock_id](./get-subblock-id.md)；如需得到跨逻辑核组合的Vector Core逻辑索引，可使用以下表达式，其中子核数量由 [get_subblock_dim](./get-subblock-dim.md) 获取：

```python
get_block_idx() * get_subblock_dim() + get_subblock_id()
```

## 约束说明

无。

## 调用示例

以下示例将Kernel启动的逻辑核数设为4，分别读取各逻辑核的索引，并在Host侧打印结果。

```python
import torch
import torch_npu

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.scalar import vec_store_bypass

@kernel
def _get_block_idx_kernel(output):
    index = get_block_idx()
    vec_store_bypass(output.ptr(index), index)

@host
def run(output):
    _get_block_idx_kernel[4](output)

def main():
    output = torch.full((4,), -1, dtype=torch.int64, device="npu:0")
    run(output)
    torch.npu.synchronize()

    print(f"block_indices={output.cpu().tolist()}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
block_indices=[0, 1, 2, 3]
```
