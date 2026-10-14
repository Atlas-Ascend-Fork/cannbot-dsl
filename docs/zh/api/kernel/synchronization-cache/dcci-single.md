---
title: dcci_single
api_name: dcci_single
category: synchronization-cache
api_group: kernel
layer: cache-control
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `dcci_single`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

每个执行核内部都有一个用于临时保存全局内存（GM）数据的数据缓存（Data Cache，DCache）。`dcci_single` 将指定 GM 地址所在缓存行中的修改写回 GM，并使该缓存行失效；后续标量读取该地址时会重新从 GM 加载数据。

## 函数原型

```python
def dcci_single(point) -> None: ...
```

## 参数说明

| 参数 | 输入/输出 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- | --- |
| `point` | 输入 | GM Tensor 元素 | 是 | 必须直接使用 `tensor[index]` 表达式指定目标地址。地址无需按缓存行边界对齐。 |

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
from cannbotdsl.ops.sync import dcci_single

@kernel
def _kernel(output):
    output[0] = 7
    dcci_single(output[0])

@host
def run(output):
    _kernel[1](output)

def main():
    output = torch.zeros(32, dtype=torch.int32, device="npu")
    run(output)
    torch.npu.synchronize()
    value = int(output.cpu()[0])
    assert value == 7
    print(f"output[0]: {value}")
    print("dcci_single example passed")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
output[0]: 7
dcci_single example passed
```
