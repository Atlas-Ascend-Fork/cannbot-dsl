---
title: get_vf_len
api_name: get_vf_len
category: system
api_group: kernel
layer: system
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `get_vf_len`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

获取矢量数据寄存器位宽 VL（Vector Length）的大小。

## 函数原型

```python
def get_vf_len() -> Int64: ...
```

## 参数说明

无。

## 返回值说明

返回 VL 的大小，单位为字节。Ascend 950PR/Ascend 950DT 的矢量数据寄存器位宽为 256 字节，本接口返回 `256`。
## 流水类型

`PIPE_S`

## 约束说明

- 本接口不触发硬件指令，不依赖任何特殊寄存器或前置配置接口，可在 Cube Core（AIC）与 Vector Core（AIV）上调用。
- 返回值反映当前芯片的矢量数据寄存器位宽，同一核函数内多次调用返回值相同。

## 调用示例

```python
import torch
import torch_npu  # noqa: F401

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_vf_len

@kernel
def _kernel(output):
    output[0] = get_vf_len()

@host
def run(output):
    _kernel[1](output)

def main():
    output = torch.empty(1, dtype=torch.int64, device="npu")
    run(output)
    torch.npu.synchronize()
    vf_len = int(output.cpu()[0])
    assert vf_len == 256
    print(f"vf_len: {vf_len}")
    print("get_vf_len example passed")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
vf_len: 256
get_vf_len example passed
```
