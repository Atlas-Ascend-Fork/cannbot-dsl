# get_core_id
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

获取当前AI Core的物理编号。

## 函数原型

```python
get_core_id() -> Int64
```

## 参数说明

无。

## 返回值说明

返回当前AI Core的编号，返回类型为 `Int64`。

## 约束说明

无。

## 调用示例

以下示例读取并打印当前物理核编号。

```python
import torch
import torch_npu

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_core_id

@kernel
def _kernel(output):
    output[0] = get_core_id()

@host
def run(output):
    _kernel[1](output)

def main():
    output = torch.empty(1, dtype=torch.int64, device="npu")
    run(output)
    torch.npu.synchronize()
    core_id = int(output.cpu()[0])
    print(f"core_id: {core_id}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
core_id: <非负整数>
```
