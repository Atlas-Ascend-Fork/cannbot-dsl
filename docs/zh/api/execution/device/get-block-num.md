# get_block_num
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

获取本次Kernel启动配置的逻辑核数。

## 函数原型

```python
get_block_num() -> Int64
```

## 参数说明

无。

## 返回值说明

返回本次Kernel启动时指定的逻辑核数，类型为 `Int64`。

在Mix 1:2模式下，该值表示逻辑核组合的数量，每个组合包含一个Cube Core和两个Vector Core。

## 约束说明

无。

## 调用示例

以下示例将Kernel启动的逻辑核数设为4，读取并打印该数量。

```python
import torch
import torch_npu

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.scalar import vec_store_bypass

@kernel
def _get_block_num_kernel(output):
    vec_store_bypass(output.ptr(get_block_idx()), get_block_num())

@host
def run(output):
    _get_block_num_kernel[4](output)

def main():
    output = torch.empty(4, dtype=torch.int64, device="npu:0")
    run(output)
    torch.npu.synchronize()

    print(f"block_num={output.cpu().tolist()[0]}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
block_num=4
```
