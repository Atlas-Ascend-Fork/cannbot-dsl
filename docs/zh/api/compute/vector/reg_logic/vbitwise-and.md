# vbitwise_and
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据`mask`对源操作数`lhs`、`rhs`执行按位与（&）操作，将结果作为返回值返回。

- 矢量数据寄存器按位与：对两个矢量数据寄存器按`dtype`位宽执行按位与（&），结果为矢量数据寄存器。

计算公式如下：

$$
dst_i = lhs_i \& rhs_i
$$

## 函数原型

```python
vbitwise_and(lhs: RawVReg, rhs: RawVReg, *, mask: Mask) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `lhs` | 输入 | 源操作数（矢量数据寄存器）。 |
| `rhs` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），对应位置为1时参与计算，为0时不参与计算。`mask`未筛选的元素在输出中置零。 |

**支持的元素数据类型：**

`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.int32`、`dtypes.uint32`。

## 返回值说明

返回保存按位与结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`lhs`、`rhs`一致。

## 约束说明

- 本接口需在`vf(mode="simd")`作用域内调用。

- 参与计算的元素个数由矢量长度（VL）决定：矢量数据寄存器按位与中元素个数 = VL ÷ sizeof(dtype)。
- `mask`比特位为0时，计算结果对应比特位写0。

## 调用示例

以下示例对两个矢量数据寄存器执行按位与操作，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vbitwise_and, vload, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vbitwise_and_kernel(src0, src1, dst):
    buf0 = Buffer(MemLoc.UB, src0.dtype, make_layout((64,)))
    buf1 = Buffer(MemLoc.UB, src1.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf0, src0)
    mem_copy(buf1, src1)

    in0 = buf0
    in1 = buf1
    res = out
    with vf(mode="simd"):
        mask = full_mask()
        and_val = vbitwise_and(vload(in0, 0), vload(in1, 0), mask=mask)
        vstore(res, 0, and_val, mask)

    mem_copy(dst, out)

@host
def run(src0, src1, dst):
    _vbitwise_and_kernel[1](src0, src1, dst)

def main():
    src0 = torch.arange(64, dtype=torch.int32, device="npu:0") + 5
    src1 = torch.arange(64, dtype=torch.int32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 & src1).cpu())
    print("vbitwise_and example passed")
    print(f"first={int(dst.cpu()[0])}, last={int(dst.cpu()[-1])}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vbitwise_and example passed
first=0, last=4
```
