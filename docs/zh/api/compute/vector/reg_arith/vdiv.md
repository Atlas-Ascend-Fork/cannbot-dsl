# vdiv
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

该接口根据`mask`，对源操作数`lhs`、`rhs`进行按元素求商操作，将结果作为返回值返回。

计算公式如下：

$$
dst_i = lhs_i \div rhs_i
$$

## 函数原型

```python
vdiv(lhs: RawVReg, rhs: RawVReg, *, mask: Mask, precision_mode: bool = False) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `lhs` | 输入 | 源操作数（矢量数据寄存器）。 |
| `rhs` | 输入 | 源操作数（矢量数据寄存器）。 |
| `precision_mode` | 输入 | 精度模式开关。为`True`时使用计算结果与标准`IEEE 754`二进制浮点除法完全一致（0 ULP）的实现，仅支持`dtypes.float32`；默认为`False`，使用硬件除法实现，结果可能存在1 ULP精度差异。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

**支持的元素数据类型：**

`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 返回值说明

返回保存逐元素除法结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`lhs`、`rhs`一致。

## 约束说明

- 本接口需在`vf(mode="simd")`作用域内调用。

- 除零错误需由调用者避免；浮点dtype除零时结果为`IEEE 754`特殊值（`±inf`或`nan`）。
- `dtypes.float16`、`dtypes.float32`数据类型的计算结果与标准`IEEE 754`二进制浮点除法相比可能存在1 ULP精度差异。

## 调用示例

以下示例演示`vdiv`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vdiv, vload, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vdiv_kernel(src0, src1, dst):
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
        vstore(res, 0, vdiv(vload(in0, 0), vload(in1, 0), mask=mask), mask)

    mem_copy(dst, out)

@host
def run(src0, src1, dst):
    _vdiv_kernel[1](src0, src1, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0") + 1.0
    src1 = torch.arange(64, dtype=torch.float32, device="npu:0") + 2.0
    dst = torch.zeros_like(src0)

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 / src1).cpu())
    print("vdiv example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vdiv example passed
first=0.5000, last=0.9846
```
