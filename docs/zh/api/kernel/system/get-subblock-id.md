---
title: get_subblock_id
api_name: get_subblock_id
category: system
api_group: kernel
layer: system
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `get_subblock_id`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

获取当前逻辑 AI Core 上 Cube Core（AIC）或 Vector Core（AIV）的逻辑索引。

## 函数原型

```python
def get_subblock_id() -> Int64: ...
```

## 参数说明

无。

## 返回值说明

返回当前逻辑 AI Core 上 Cube Core（AIC）或 Vector Core（AIV）的逻辑索引，返回值始终小于 [`get_subblock_dim()`](./get-subblock-dim) 的返回值。
## 流水类型

`PIPE_S`

## 约束说明

- 本接口为只读查询接口，不修改任何寄存器或存储状态。

## 调用示例

```python
import torch
import torch_npu  # noqa: F401

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
    assert result[0].item() == 0
    assert result[1].item() in (-1, 1)
    subblock_ids = [index for index, value in enumerate(result.tolist()) if value >= 0]
    print(f"subblock_ids: {subblock_ids}")
    print("get_subblock_id example passed")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
subblock_ids: <[0] 或 [0, 1]>
get_subblock_id example passed
```
