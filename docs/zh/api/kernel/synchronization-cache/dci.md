---
title: dci
api_name: dci
category: synchronization-cache
api_group: kernel
layer: cache-control
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `dci`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

使本核数据缓存（DCache）中的所有缓存行失效；尚未写回 GM 的修改不会被保留，而是直接丢弃。

## 函数原型

```python
def dci() -> None: ...
```

## 参数说明

无。

## 返回值说明

无。
## 流水类型

`PIPE_S`

## 约束说明

- CANNBot DSL 会在失效操作前自动等待已有内存访问完成。
- 仅当本核缓存中尚未写回的修改可以丢弃时使用。需要保留这些修改时，应先使用 `dcci_single()` 或 `dcci_entire_out()` 将其写回 GM。

## 调用示例

```python
import torch
import torch_npu  # noqa: F401

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.sync import dci

@kernel
def _kernel(output):
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

### 预期结果

```text
output[0]: 1
dci example passed
```
