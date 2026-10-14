---
title: dcci_entire_atomic
api_name: dcci_entire_atomic
category: synchronization-cache
api_group: kernel
layer: cache-control
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `dcci_entire_atomic`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

将本核数据缓存（DCache）中由标量原子操作访问的 GM 缓存行写回 GM，并使其失效；其他执行核随后可以读取到原子操作更新后的值。

## 函数原型

```python
def dcci_entire_atomic() -> None: ...
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
from cannbotdsl.ops.scalar import atomic_add
from cannbotdsl.ops.sync import dcci_entire_atomic

@kernel
def _kernel(counter):
    atomic_add(counter.ptr(0), 1)
    dcci_entire_atomic()

@host
def run(counter):
    _kernel[1](counter)

def main():
    counter = torch.zeros(1, dtype=torch.int32, device="npu")
    run(counter)
    torch.npu.synchronize()
    value = int(counter.cpu()[0])
    assert value == 1
    print(f"counter: {value}")
    print("dcci_entire_atomic example passed")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
counter: 1
dcci_entire_atomic example passed
```
