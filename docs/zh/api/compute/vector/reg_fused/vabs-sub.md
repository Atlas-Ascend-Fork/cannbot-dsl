# vabs_sub
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据`mask`对源操作数`lhs`、`rhs`按元素相减后取绝对值，得到计算结果。计算公式如下：

$$
dst_i = \lvert lhs_i - rhs_i \rvert
$$

## 函数原型

```python
vabs_sub(lhs: RawVReg, rhs: RawVReg, *, mask: Mask) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `lhs` | 输入 | 源操作数0（矢量数据寄存器）。 |
| `rhs` | 输入 | 源操作数1（矢量数据寄存器）。 |
| `mask` | 输入 | 掩码寄存器，用于控制各元素是否参与计算。`mask`中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。<br>`mask`掩码位为0时，结果对应元素置0。 |

**支持的元素数据类型：**

`dtypes.float16`、`dtypes.float32`。

## 返回值说明

返回保存差值绝对值计算结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`lhs`、`rhs`一致。

## 约束说明

- 本接口需在`vf(mode="simd")`作用域内调用。

- `lhs`和`rhs`的数据类型需要保持一致。

## 调用示例

以下示例演示`vabs_sub`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vabs_sub, vload, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vabs_sub_kernel(src0, src1, dst):
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
        r = vabs_sub(vload(in0, 0), vload(in1, 0), mask=mask)
        vstore(res, 0, r, mask)

    mem_copy(dst, out)

@host
def run(src0, src1, dst):
    _vabs_sub_kernel[1](src0, src1, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0") / 8.0
    src1 = torch.arange(64, dtype=torch.float32, device="npu:0") / 4.0
    dst = torch.empty_like(src0)

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 - src1).abs().cpu())
    print("vabs_sub example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vabs_sub example passed
first=0.0000, last=7.8750
```
