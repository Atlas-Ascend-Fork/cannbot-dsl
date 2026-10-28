# vsqrt
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

该接口根据`mask`，对源操作数`src`进行按元素求开方操作，将结果作为返回值返回。

计算公式如下：

$$
dst_i = \sqrt{src_i}
$$

## 函数原型

```python
vsqrt(src: RawVReg, *, mask: Mask) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

**支持的元素数据类型：**

`dtypes.float16`、`dtypes.float32`。

## 返回值说明

返回保存逐元素平方根计算结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`src`一致。

## 约束说明

- 本接口需在`vf(mode="simd")`作用域内调用。

- 输入为负数时结果为nan。
- 输入为-0时结果为-0。

## 调用示例

以下示例演示`vsqrt`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload, vsqrt, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vsqrt_kernel(src, dst):
    buf = Buffer(MemLoc.UB, src.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf, src)

    in0 = buf
    res = out
    with vf(mode="simd"):
        mask = full_mask()
        vstore(res, 0, vsqrt(vload(in0, 0), mask=mask), mask)

    mem_copy(dst, out)

@host
def run(src, dst):
    _vsqrt_kernel[1](src, dst)

def main():
    src = torch.arange(1, 65, dtype=torch.float32, device="npu:0")
    dst = torch.zeros_like(src)

    run(src, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.sqrt(src).cpu())
    print("vsqrt example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vsqrt example passed
first=1.0000, last=8.0000
```
