# vdup
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将源操作数`src`中由`position`指定的元素广播到返回值中被`mask`筛选的位置。`mask`仅用于筛选返回值中的元素位置，不影响从`src`中读取的元素；`mode`取值为`'zeroing'`（默认值）时返回值中未被`mask`筛选的位置置零，取值为`'merging'`时这些位置保留`merge`中的值。

## 函数原型

```python
vdup(src: RawVReg, *, mask: Mask, position: str = 'lowest', mode: str = 'zeroing', merge: 'RawVReg | None' = None) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器），其最低位元素作为待广播的数据。数据类型需要与返回值保持一致。 |
| `mask` | 输入 | 源操作数元素操作的有效指示（掩码寄存器）。`mask`筛选的元素在返回值中填充为`src`中由`position`指定的元素，未筛选的元素按`mode`的取值处理。 |
| `position` | 输入 | 广播数据的来源位置，取值为`'lowest'`或`'highest'`，默认值为`'lowest'`。取值为`'lowest'`时，待广播的数据为`src`的最低位元素（即下标为0的元素）；取值为`'highest'`时，待广播的数据为`src`的最高位元素（即下标最大的元素）。 |
| `mode` | 输入 | 掩码模式，取值为`'zeroing'`或`'merging'`，默认值为`'zeroing'`。取值为`'zeroing'`时，返回值中未被`mask`筛选的位置置零；取值为`'merging'`时采用合并模式，返回值中未被`mask`筛选的元素保留`merge`的值。 |
| `merge` | 输入 | 合并模式下保留原值的矢量数据寄存器，仅在`mode`取值为`'merging'`时使用，数据类型须与`src`一致，默认值为`None`。返回值中未被`mask`筛选的元素保留该寄存器的值。 |

**支持的元素数据类型：**

`dtypes.int8`、`dtypes.uint8`、`dtypes.float8_e8m0`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 返回值说明

返回保存广播结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`src`、`merge`一致。

## 约束说明

- 本接口需在VF作用域内调用。
- 同一寄存器的数据依赖由硬件保序，无需额外插入同步指令。本接口与前后Reg 数据搬运接口之间，如果不同寄存器访问同一Unified Buffer（UB）地址且存在写后读或写后写依赖，需要调用[vmem_bar](../reg_sync/vmem-bar.md)进行同步。
- `mask`仅筛选返回值中写入广播值的位置，不筛选`src`中的元素。无论`mask`的最低位是否有效，待广播的数据均为`src`中由`position`指定的元素。

## 调用示例

以下示例演示`vdup`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vdup, vload, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vdup_kernel(src0, dst):
    buf = Buffer(MemLoc.UB, src0.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))
    mem_copy(buf, src0)

    in0 = buf
    res = out
    with vf(mode="simd"):
        full = full_mask()
        acc = vdup(vload(in0, 0), mask=full)
        vstore(res, 0, acc, full)

    mem_copy(dst, out)

@host
def run(src0, dst):
    _vdup_kernel[1](src0, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0") + 1.0
    dst = torch.empty_like(src0)

    run(src0, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.full((64,), 1.0))
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
first=1.0000, last=1.0000
```
