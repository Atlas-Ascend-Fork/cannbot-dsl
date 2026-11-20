# get_squeeze_status
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

获取squeeze存储操作产生的有效数据长度。

## 函数原型

```python
get_squeeze_status() -> Int64
```

## 参数说明

无。

## 返回值说明

返回squeeze存储操作产生的有效数据字节数，类型为 `Int64`。

## 约束说明

- 调用本接口前，需先调用 [`vsqueeze_and_storeunalign_init`](../../compute/vector/reg_permute_sel/vsqueeze-and-storeunalign-init.md) 清空AR特殊寄存器，再调用 [`vsqueeze_and_storeunalign`](../../compute/vector/reg_permute_sel/vsqueeze-and-storeunalign.md) 完成数据筛选；否则返回值没有确定含义。

## 调用示例

以下示例在统一缓冲区（Unified Buffer，UB）中压缩存储4个 `uint32` 元素，再读取并打印有效数据长度。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout
from cannbotdsl.tensor import MemLoc
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_squeeze_status
from cannbotdsl.ops.reg import (
    create_mask,
    vdups,
    vsqueeze_and_storeunalign,
    vsqueeze_and_storeunalign_finalize,
    vsqueeze_and_storeunalign_init,
)

@jit
def compact_store(ub_output):
    with vf(mode="simd"):
        mask = create_mask(pattern="vl4", elem_bits=32)
        source = vdups(7, dtypes.uint32, mask=mask)
        cursor = vsqueeze_and_storeunalign_init(ub_output)

        vsqueeze_and_storeunalign(ub_output, 0, source, cursor, mask=mask)
        vsqueeze_and_storeunalign_finalize(ub_output, 0, cursor)

@kernel
def _kernel(status):
    ub_output = Buffer(MemLoc.UB, dtypes.uint32, make_layout((64,)))
    compact_store(ub_output)
    status[0] = get_squeeze_status()

@host
def run(status):
    _kernel[1](status)

def main():
    status = torch.empty(1, dtype=torch.int64, device="npu")
    run(status)
    torch.npu.synchronize()
    valid_bytes = int(status.cpu()[0])
    assert valid_bytes == 16, valid_bytes
    print(f"squeeze_valid_bytes: {valid_bytes}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
squeeze_valid_bytes: 16
```
