# vexp_sub
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据`mask`将`lhs`与`rhs`按元素相减，并计算以差值为指数的自然指数函数，将结果作为返回值返回。

需要处理混合精度输入或选择FP16的奇数源位置时，请使用[`vcast_exp_sub`](./vcast-exp-sub.md)。本接口不提供奇偶位置选择参数。

- `vexp_sub`用于`dtypes.float32`类型输入。计算公式如下：

  $$
  result_i = e^{lhs_i - rhs_i}
  $$

- 源操作数为`dtypes.float16`类型时，固定读取源操作数中偶数索引的元素，将其转换为`dtypes.float32`类型后计算；计算公式如下：

  $$
  result_i = e^{float(lhs_{2i}) - float(rhs_{2i})}
  $$

  返回值类型为`dtypes.float32`，元素个数为源操作数元素个数的一半。

## 函数原型

```python
vexp_sub(lhs: RawVReg, rhs: RawVReg, *, mask: Mask) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `lhs` | 输入 | 源操作数0（矢量数据寄存器）。 |
| `rhs` | 输入 | 源操作数1（矢量数据寄存器）。 |
| `mask` | 输入 | 掩码寄存器，用于控制各元素是否参与计算。`mask`中与输出元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。<br>`mask`掩码位为0时，结果对应元素置0。 |

**支持的元素数据类型：**

源操作数`lhs`、`rhs`的元素数据类型均支持`dtypes.float16`、`dtypes.float32`，且二者必须相同。

## 返回值说明

返回保存计算结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型固定为`dtypes.float32`。

## 约束说明

- 本接口需在`vf(mode="simd")`作用域内调用。
- 只有当输出数据类型位宽大于输入时，计算时才会有精度提升。

## 调用示例

以下示例演示输入数据类型为`float32`时`vexp_sub`接口的调用方式，并在Host侧校验计算结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vexp_sub, vload, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vexp_sub_kernel(src0, src1, dst):
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
        r = vexp_sub(vload(in0, 0), vload(in1, 0), mask=mask)
        vstore(res, 0, r, mask)

    mem_copy(dst, out)

@host
def run(src0, src1, dst):
    _vexp_sub_kernel[1](src0, src1, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0") / 16.0
    src1 = torch.arange(64, dtype=torch.float32, device="npu:0") / 32.0
    dst = torch.empty(64, dtype=torch.float32, device="npu:0")

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.exp(src0 - src1).cpu())
    print("vexp_sub example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vexp_sub example passed
first=1.0000, last=7.1617
```

以下示例演示输入数据类型为`float16`时`vexp_sub`接口的调用方式，并在Host侧校验计算结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import create_mask, vexp_sub, vload, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def half_exp_sub(src0, src1, dst):
    left = Buffer(MemLoc.UB, dtypes.float16, make_layout((128,)))
    right = Buffer(MemLoc.UB, dtypes.float16, make_layout((128,)))
    out = Buffer(MemLoc.UB, dtypes.float32, make_layout((64,)))
    mem_copy(left, src0)
    mem_copy(right, src1)
    with vf(mode="simd"):
        a = vload(left, 0)
        b = vload(right, 0)
        output_mask = create_mask(elem_bits=32)
        result = vexp_sub(a, b, mask=output_mask)
        vstore(out, 0, result, output_mask)
    mem_copy(dst, out)

@host
def run_half(src0, src1, dst):
    half_exp_sub[1](src0, src1, dst)

def main():
    a = torch.linspace(-0.75, 0.75, 128, dtype=torch.float16, device="npu:0")
    b = torch.linspace(0.5, -0.25, 128, dtype=torch.float16, device="npu:0")
    dst = torch.empty(64, dtype=torch.float32, device="npu:0")
    run_half(a, b, dst)
    torch.npu.synchronize()
    expected = torch.exp(a.cpu().float()[::2] - b.cpu().float()[::2])
    torch.testing.assert_close(dst.cpu(), expected, atol=2e-4, rtol=2e-3)
    print("vexp_sub fp16 fixed-even example passed")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vexp_sub fp16 fixed-even example passed
```
