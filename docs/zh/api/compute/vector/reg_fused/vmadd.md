# vmadd
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据`mask`对源操作数`lhs`、`rhs`按元素相乘后，与加数操作数`addend`中的对应元素相加，将结果作为返回值返回。计算公式如下：

$$
\mathrm{result}_i = lhs_i \times rhs_i + addend_i
$$

其中，`result`表示返回值，输入寄存器`lhs`不被原地修改。

## 函数原型

```python
vmadd(lhs: RawVReg, rhs: RawVReg, addend: RawVReg, *, mask: Mask) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `lhs` | 输入 | 源操作数0，参与乘法计算；计算结果通过返回值返回。 |
| `rhs` | 输入 | 源操作数1（矢量数据寄存器）。 |
| `addend` | 输入 | 加数操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 掩码寄存器，用于控制各元素是否参与计算。`mask`中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。<br>`mask`掩码位为0时，结果对应元素置0。 |

**支持的元素数据类型：**

`dtypes.float16`、`dtypes.bfloat16`、`dtypes.float32`。

## 返回值说明

返回保存乘加结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`lhs`、`rhs`、`addend`一致。

## 约束说明

- 本接口需在`vf(mode="simd")`作用域内调用。

- `lhs`、`rhs`和`addend`的数据类型需要保持一致。

## 调用示例

以下示例演示`vmadd`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload, vmadd, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vmadd_kernel(src0, src1, src2, dst):
    buf0 = Buffer(MemLoc.UB, src0.dtype, make_layout((64,)))
    buf1 = Buffer(MemLoc.UB, src1.dtype, make_layout((64,)))
    buf2 = Buffer(MemLoc.UB, src2.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf0, src0)
    mem_copy(buf1, src1)
    mem_copy(buf2, src2)

    in0 = buf0
    in1 = buf1
    in2 = buf2
    res = out
    with vf(mode="simd"):
        mask = full_mask()
        r = vmadd(vload(in0, 0), vload(in1, 0), vload(in2, 0), mask=mask)
        vstore(res, 0, r, mask)

    mem_copy(dst, out)

@host
def run(src0, src1, src2, dst):
    _vmadd_kernel[1](src0, src1, src2, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0") / 8.0
    src1 = torch.arange(64, dtype=torch.float32, device="npu:0") / 16.0
    src2 = torch.arange(64, dtype=torch.float32, device="npu:0") / 32.0
    dst = torch.empty_like(src0)

    run(src0, src1, src2, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 * src1 + src2).cpu())
    print("vmadd example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vmadd example passed
first=0.0000, last=32.9766
```
