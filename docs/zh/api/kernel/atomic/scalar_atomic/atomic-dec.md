---
title: atomic_dec
api_name: atomic_dec
category: atomic
api_group: kernel
layer: scalar
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `atomic_dec`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

对Global Memory中`ptr`指向的一个`dtypes.uint32`或`dtypes.uint64`元素执行原子递减操作。读取该元素的旧值`old_value`，按照以下公式计算并写回新值`new_value`：

$$
new\_value = (old\_value == 0 || old\_value > value) ? value : (old\_value - 1)
$$

接口返回`old_value`。整个读取、判断和写回过程为原子操作。

## 函数原型

```python
def atomic_dec(ptr, value) -> ScalarValue: ...
```

**支持的数据类型：**

dtype支持的数据类型为`dtypes.uint32`、`dtypes.uint64`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `ptr` | 输入 | Global Memory的地址。 |
| `value` | 输入 | 源操作数。 |

## 返回值说明

`ptr`地址中计算前的原始数据。
## 流水类型

`PIPE_S`

## 约束说明

- 在开启编译器自动同步功能的前提下，编译器能够自动在PIPE_MTE2/PIPE_MTE3与PIPE_S之间插入同步。但是，`atomic_dec`为标量计算，在读写GM时如果与搬运单元（MTE2/MTE3）存在数据依赖，编译器却无法自动插入同步，开发者需要根据实际情况手动插入同步。
- Scalar原子操作会绕过DCache，需要调用[`dcci_single`](/zh/api/kernel/synchronization-cache/dcci-single)接口确保GM与DCache的一致性。

## 调用示例

将代码保存为`atomic_dec.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import dtypes, host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import atomic_dec

@kernel
def _atomic_dec_kernel(acc):
    # 4 个 block 各自把 acc[0] 原子递减 1；value 为上界（10），6 - 4 = 2，未回绕
    atomic_dec(acc.ptr(0), dtypes.uint32(10))

@host
def run(acc):
    _atomic_dec_kernel[4](acc)

def main():
    acc = torch.full((1,), 6, dtype=torch.int32, device="npu:0").view(torch.uint32)
    run(acc)
    torch.npu.synchronize()

    assert acc.cpu().tolist() == [2], acc.cpu().tolist()
    print(f"atomic_dec example passed, acc={acc.cpu().tolist()}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
atomic_dec example passed, acc=[2]
```
