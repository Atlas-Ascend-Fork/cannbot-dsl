# dci
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

使本核数据缓存（DCache）中的所有缓存行失效，并丢弃尚未写回GM的修改。

## 函数原型

```python
dci() -> None
```

## 参数说明

无

## 返回值说明

无

## 约束说明

无

## 调用示例

以下示例调用`dci`后写入输出，并在Host侧校验写入结果。

```python
import torch
import torch_npu  # noqa: F401

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.sync import dci

@kernel
def _kernel(output):
    # 先使DCache失效，再写入输出；此例仅演示接口调用。
    dci()
    output[0] = 1

@host
def run(output):
    _kernel[1](output)

def main():
    output = torch.zeros(1, dtype=torch.int32, device="npu")
    run(output)
    torch.npu.synchronize()
    value = int(output.cpu()[0])
    assert value == 1
    print(f"output[0]: {value}")
    print("dci example passed")

if __name__ == "__main__":
    main()
```

预期结果：

```text
output[0]: 1
dci example passed
```
