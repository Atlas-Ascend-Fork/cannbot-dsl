# vpack
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将源操作数中的元素选取低8位（b16）、低16位（b32）后压缩写入返回值中：`part`取值为`'lower'`（默认值）时写入返回值的低半部分，取值为`'highest'`时写入返回值的高半部分。

![pack high draft示意图](../../../figures/pack_high_draft.svg)

图中为`part='highest'`时的压缩结果。

## 函数原型

```python
vpack(src: RawVReg, dtype, *, part: str = 'lower') -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `dtype` | 输入 | 返回值的数据类型，支持的类型对参见本节的对应关系表。 |
| `part` | 输入 | 指定压缩结果写入返回值的半部分，取值为`'lower'`（低半部分）或`'highest'`（高半部分），默认为`'lower'`。 |

**支持的元素数据类型：**

dtype_src与dtype_dst支持的数据类型对如下：

| dtype_src | dtype_dst |
|---|---|
| `dtypes.uint16` | `dtypes.uint8` |
| `dtypes.int16` | `dtypes.uint8` |
| `dtypes.uint32` | `dtypes.uint16` |
| `dtypes.int32` | `dtypes.uint16` |

## 返回值说明

返回保存压缩结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型由`dtype`指定。

## 约束说明

- 本接口需在VF作用域内调用。
- `src`与返回值均为矢量数据寄存器。

## 调用示例

以下示例演示`vpack`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import create_mask, vload, vpack, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vpack_kernel(src0, dst):
    buf = Buffer(MemLoc.UB, src0.dtype, make_layout((128,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((128,)))

    mem_copy(buf, src0)

    in0 = buf
    res = out
    with vf(mode="simd"):
        mask = create_mask(pattern="vl128", elem_bits=8)
        vstore(res, 0, vpack(vload(in0, 0), dtypes.uint8), mask)

    mem_copy(dst, out)

@host
def run(src0, dst):
    _vpack_kernel[1](src0, dst)

def main():
    src0 = torch.arange(128, dtype=torch.int32).to(torch.uint16).to("npu:0")
    dst = torch.empty((128,), dtype=torch.uint8, device="npu:0")

    run(src0, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu().to(torch.int32), src0.cpu().to(torch.int32))
    print("vpack example passed")
    print(f"first={int(dst.cpu()[0])}, last={int(dst.cpu()[-1])}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vpack example passed
first=0, last=127
```
