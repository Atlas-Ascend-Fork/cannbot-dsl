---
title: get_core_id
api_name: get_core_id
category: system
api_group: kernel
layer: system
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `get_core_id`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

获取当前 AI Core 的编号。按任务切分数据时，应使用 [`get_block_idx()`](./get-block-idx)。

## 函数原型

```python
def get_core_id() -> Int64: ...
```

## 参数说明

无。

## 返回值说明

返回当前 AI Core 的编号。
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
    assert core_id >= 0
    print(f"core_id: {core_id}")
    print("get_core_id example passed")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
core_id: <非负整数>
get_core_id example passed
```
