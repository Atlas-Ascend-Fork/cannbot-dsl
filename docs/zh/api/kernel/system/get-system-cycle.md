---
title: get_system_cycle
api_name: get_system_cycle
category: system
api_group: kernel
layer: system
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `get_system_cycle`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

读取当前系统 cycle 计数器，返回 64 位整数类型的当前计数值。该计数器只读，复位值为 `0`，并随系统时钟持续递增，用于记录硬件运行以来累计的 cycle 数。该接口常用于性能统计、耗时测量和执行时序判断，在 AIC 和 AIV 上均可调用，返回值含义相同。

## 函数原型

```python
def get_system_cycle() -> Int64: ...
```

## 参数说明

无。

## 返回值说明

返回当前系统时钟周期计数。
## 流水类型

`PIPE_S`

## 约束说明

- 读取 cycle 计数与其他计算、搬运流水相互独立。测量这些流水上的操作时，需要先使用对应同步接口确保被测操作完成。
- 针对 Ascend 950PR/Ascend 950DT，若换算成时间需要按照 1 GHz 的频率，时间单位为 us，换算公式为：`time = (cycle数 / 1000) us`。

## 调用示例

```python
import torch
import torch_npu  # noqa: F401

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_system_cycle

@kernel
def _kernel(output):
    start = get_system_cycle()
    end = get_system_cycle()
    output[0] = end - start

@host
def run(output):
    _kernel[1](output)

def main():
    output = torch.empty(1, dtype=torch.int64, device="npu")
    run(output)
    torch.npu.synchronize()
    elapsed_cycles = int(output.cpu()[0])
    assert elapsed_cycles > 0
    print(f"elapsed_cycles: {elapsed_cycles}")
    print("get_system_cycle example passed")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
elapsed_cycles: <正整数>
get_system_cycle example passed
```
