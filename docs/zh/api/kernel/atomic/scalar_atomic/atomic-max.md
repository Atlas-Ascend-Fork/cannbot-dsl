---
title: atomic_max
api_name: atomic_max
category: atomic
api_group: kernel
layer: scalar
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `atomic_max`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

对Global Memory中`ptr`指向的单个元素执行原子取大操作：读取该地址中的旧值`old_value`，比较`old_value`与输入标量值`value`，将较大值`new_value`写回该地址，并返回`old_value`。整个读取、比较和写回过程为原子操作。

计算公式如下：

$$
new\_value = Max(old\_value, value)
$$

## 函数原型

```python
def atomic_max(ptr, value) -> ScalarValue: ...
```

**支持的数据类型：**

`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`、`dtypes.int64`、`dtypes.uint64`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `ptr` | 输入/输出 | Global Memory的地址。 |
| `value` | 输入 | 标量值，数据类型与`ptr`指向元素类型一致。 |

## 返回值说明

返回`ptr`地址中计算前的原始数据`old_value`。

## 约束说明

- `ptr`必须落在Global Memory地址空间。
- `ptr`需按`sizeof(dtype)`字节对齐。
- 本接口不受[`mem_copy`](/zh/api/kernel/data-movement/mem-copy)的`atomic_add`等数据搬运随路原子设置影响，也不会影响后续搬运指令。
- 对同一`ptr`的并发调用以原子方式完成“读取、比较、写回”，不会丢失更新；最终写入结果为各参与值中的最大值。
- 本接口运行在标量流水（`PIPE_S`）上，同一标量流水内的数据依赖由指令执行顺序保证。若本接口与`PIPE_MTE2`或`PIPE_MTE3`上的数据搬运指令访问同一GM地址，且执行顺序影响结果，编译器无法自动完成跨流水同步，调用方需按实际依赖插入`vec_sync_pipe`，或配合使用`vec_sync_notify`与`vec_sync_wait`保证执行顺序。
- 本接口访问GM时绕过DCache，不维护缓存一致性。若其他核或其他通路通过缓存访问同一GM地址，调用方需使用[`dcci_single`](/zh/api/kernel/synchronization-cache/dcci-single)清理或失效对应Cache Line，并保证相关访存操作的执行顺序和数据可见性。
- `dtypes.float32`类型下接口的性能与接口返回值是否被使用有关，具体情况如下：
    - 当使用接口返回值时，接口采用软仿实现，`dtypes.float32`类型下接口的性能低于其它数据类型。
    - 当不使用接口返回值时，`dtypes.float32`类型下接口的性能与其它数据类型一致。

## 调用示例

将代码保存为`atomic_max.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import dtypes, host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.scalar import atomic_max

@kernel
def _atomic_max_kernel(acc):
    # 4 个 block 分别以自己序号 + 1（1~4）参与原子取大
    atomic_max(acc.ptr(0), dtypes.int32(get_block_idx() + 1))

@host
def run(acc):
    _atomic_max_kernel[4](acc)

def main():
    acc = torch.zeros(1, dtype=torch.int32, device="npu:0")
    run(acc)
    torch.npu.synchronize()

    assert acc.cpu().tolist() == [4], acc.cpu().tolist()
    print(f"atomic_max example passed, acc={acc.cpu().tolist()}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
atomic_max example passed, acc=[4]
```
