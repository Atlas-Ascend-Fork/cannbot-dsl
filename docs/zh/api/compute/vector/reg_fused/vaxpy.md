# vaxpy
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据`mask`将源操作数`src`中的每个元素与标量`scalar`求积后，与输入累加器`acc`中的对应元素相加，将结果作为返回值返回。其中，`acc`作为输入累加器参与计算。计算公式如下：

$$
\mathrm{result}_i = src_i \times scalar + acc_i
$$

其中，`result`表示返回值，输入寄存器`acc`不被原地修改。

## 函数原型

```python
vaxpy(acc: RawVReg, src: RawVReg, scalar, *, mask: Mask) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `acc` | 输入 | 输入累加器，提供加法所需的原始值；计算结果通过返回值返回。 |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `scalar` | 输入 | 源操作数（标量）。 |
| `mask` | 输入 | 掩码寄存器，用于控制各元素是否参与计算。`mask`中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。<br>`mask`掩码位为0时，结果对应元素置0。 |

**支持的元素数据类型：**

`dtypes.float16`、`dtypes.float32`。

## 返回值说明

返回保存乘加结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`acc`、`src`一致。

## 约束说明

- 本接口需在`vf(mode="simd")`作用域内调用。

## 调用示例

以下示例演示`vaxpy`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vaxpy, vload, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vaxpy_kernel(src0, src1, dst):
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
        r = vaxpy(vload(in0, 0), vload(in1, 0), 2.0, mask=mask)
        vstore(res, 0, r, mask)

    mem_copy(dst, out)

@host
def run(src0, src1, dst):
    _vaxpy_kernel[1](src0, src1, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0") / 8.0
    src1 = torch.arange(64, dtype=torch.float32, device="npu:0") / 16.0
    dst = torch.empty_like(src0)

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 + src1 * 2.0).cpu())
    print("vaxpy example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vaxpy example passed
first=0.0000, last=15.7500
```
