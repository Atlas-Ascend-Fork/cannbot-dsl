---
title: load_bypass
api_name: load_bypass
category: scalar_compute
api_group: kernel
layer: scalar
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `load_bypass`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

不经过DCache直接从GM地址读取整型数据。
当多核操作GM地址且数据无法对齐到Cache Line时，经过DCache读写可能引入Cache Line粒度的数据覆盖。此时，可使用本接口绕过DCache读取GM数据。

## 函数原型

```python
def load_bypass(ptr) -> ScalarValue: ...
```

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `ptr` | 输入 | 源GM地址。支持的数据类型为`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.int64`、`dtypes.uint64`。 |

## 返回值说明

从GM读取的数据，返回值的数据类型与`ptr`指向的数据类型一致。
## 流水类型

`PIPE_S`

## 约束说明

仅支持整型数据，不支持浮点类型。

## 调用示例

将代码保存为`load_bypass.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import load_bypass, vec_store_bypass

@kernel
def _load_bypass_kernel(src, dst):
    vec_store_bypass(dst.ptr(0), load_bypass(src.ptr(0)))

@host
def run(src, dst):
    _load_bypass_kernel[1](src, dst)

def main():
    src = torch.full((1,), 42, dtype=torch.int32, device="npu:0")
    dst = torch.zeros((1,), dtype=torch.int32, device="npu:0")

    run(src, dst)
    torch.npu.synchronize()

    result = int(dst.cpu()[0])
    assert result == 42, result
    print("load_bypass example passed")
    print(f"loaded={result}")

if __name__ == "__main__":
    main()
```

### 预期结果

```text
load_bypass example passed
loaded=42
```
