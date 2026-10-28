# get_subblock_id
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

获取当前逻辑AI Core上Cube Core（AIC）或Vector Core（AIV）的逻辑索引。

## 函数原型

```python
get_subblock_id() -> Int64
```

## 参数说明

无。

## 返回值说明

返回当前逻辑AI Core上Cube Core（AIC）或Vector Core（AIV）的逻辑索引，返回值始终小于 [`get_subblock_dim()`](./get-subblock-dim.md) 的返回值，返回类型为 `Int64`。

## 约束说明

无。

## 调用示例

以下示例按子核索引写入输出，并在Host侧打印已执行子核的索引。

```python
import torch
import torch_npu

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_subblock_id

@kernel
def _kernel(output):
    subblock_id = get_subblock_id()
    output[subblock_id] = subblock_id

@host
def run(output):
    _kernel[1](output)

def main():
    output = torch.full((2,), -1, dtype=torch.int64, device="npu")
    run(output)
    torch.npu.synchronize()
    result = output.cpu()
    subblock_ids = [index for index, value in enumerate(result.tolist()) if value >= 0]
    print(f"subblock_ids: {subblock_ids}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
subblock_ids: <[0] 或 [0, 1]>
```
