---
title: get_subblock_dim
api_name: get_subblock_dim
category: system
api_group: kernel
layer: system
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `get_subblock_dim`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

获取当前配置中一个逻辑 AI Core 上 Cube Core（AIC）或 Vector Core（AIV）的数量。

## 函数原型

```python
def get_subblock_dim() -> Int64: ...
```

## 参数说明

无。

## 返回值说明

Cube、Vector 和 Mix 1:1 模式返回 `1`。Mix 1:2 模式下，Cube 侧返回 `1`，Vector 侧返回 `2`。
## 流水类型

`PIPE_S`

## 约束说明

- 本接口读取的子核数量保存在只读特殊寄存器中，由系统控制器（System Controller，SC）在核函数启动前配置，核函数运行期间不可修改，连续两次调用返回值相同。
- 本接口为只读查询接口，不修改任何寄存器或存储状态。

## 调用示例

```python
import torch
import torch_npu  # noqa: F401

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
    assert values and all(value in (1, 2) for value in values)
    print(f"subblock_dims: {sorted(set(values))}")
    print("get_subblock_dim example passed")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
subblock_dims: <[1]、[2] 或 [1, 2]>
get_subblock_dim example passed
```
