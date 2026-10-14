---
title: dcci_entire_out
api_name: dcci_entire_out
category: synchronization-cache
api_group: kernel
layer: cache-control
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `dcci_entire_out`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

将本核数据缓存（DCache）中所有与 GM 地址相关的有效缓存行写回 GM，并使其失效。适用于在多处标量 GM 写入后一次性同步修改。

## 函数原型

```python
def dcci_entire_out() -> None: ...
```

## 参数说明

无。

## 返回值说明

无。
## 流水类型

`PIPE_S`

## 约束说明

无。

## 调用示例

```python
import torch
import torch_npu  # noqa: F401

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.sync import dcci_entire_out

@kernel
def _kernel(output):
    output[0] = 7
    output[32] = 9
    dcci_entire_out()

@host
def run(output):
    _kernel[1](output)

def main():
    output = torch.zeros(64, dtype=torch.int32, device="npu")
    run(output)
    torch.npu.synchronize()
    values = output.cpu()[[0, 32]].tolist()
    assert values == [7, 9]
    print(f"output values: {values}")
    print("dcci_entire_out example passed")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
output values: [7, 9]
dcci_entire_out example passed
```
