# get_subblock_dim
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

获取当前配置中一个逻辑AI Core上Cube Core（AIC）或Vector Core（AIV）的数量。

## 函数原型

```python
get_subblock_dim() -> Int64
```

## 参数说明

无。

## 返回值说明

Cube、Vector和Mix 1:1模式返回 `1`，返回类型为 `Int64`。Mix 1:2模式下，Cube侧返回 `1`，Vector侧返回 `2`。

## 约束说明

无。

## 调用示例

以下示例读取各子核的子核数量，并在Host侧打印结果。

```python
import torch
import torch_npu

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_subblock_dim, get_subblock_id

@kernel
def _kernel(output):
    output[get_subblock_id()] = get_subblock_dim()

@host
def run(output):
    _kernel[1](output)

def main():
    output = torch.full((2,), -1, dtype=torch.int64, device="npu")
    run(output)
    torch.npu.synchronize()
    values = [value for value in output.cpu().tolist() if value >= 0]
    print(f"subblock_dims: {sorted(set(values))}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
subblock_dims: <[1]、[2] 或 [1, 2]>
```
