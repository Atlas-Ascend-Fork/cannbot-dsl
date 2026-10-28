# mask_not
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据`exec_mask`对源操作数`src`执行按位取反操作，将结果作为返回值返回。

- 掩码寄存器按位取反：对掩码寄存器按位取反，结果为掩码寄存器。

计算公式如下：

$$
dst_i = \sim src_i
$$

## 函数原型

```python
mask_not(src: Mask, *, exec_mask: Mask) -> Mask
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（掩码寄存器）。 |
| `exec_mask` | 输入 | 源操作数掩码（掩码寄存器），指示在计算过程中哪些bit有效。 |

**支持的元素数据类型：**

掩码寄存器不区分元素数据类型，仅由元素位宽`elem_bits`标识；参与计算的所有掩码寄存器必须具有相同的`elem_bits`。

## 返回值说明

返回保存按位取反结果的掩码寄存器值，类型为`Mask`，掩码粒度与`src`一致。

## 约束说明

- 本接口需在`vf(mode="simd")`作用域内调用。

- 参与计算的元素个数由矢量长度（VL）决定：
    - 掩码寄存器按位取反中比特个数 = VL。
- `exec_mask`比特位为0时，计算结果对应比特位写0。
- `src`与`exec_mask`的`elem_bits`必须一致，否则会报错。

## 调用示例

以下示例对掩码执行按位取反操作，并在Host侧校验选择结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import create_mask, full_mask, mask_not, vdups, vselect, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _mask_not_kernel(dst):
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    res = out
    with vf(mode="simd"):
        full = full_mask()
        m = mask_not(create_mask(pattern="vl16"), exec_mask=full)
        one = vdups(1.0, dtypes.float32)
        zero = vdups(0.0, dtypes.float32)
        vstore(res, 0, vselect(one, zero, cond_mask=m), full)

    mem_copy(dst, out)

@host
def run(dst):
    _mask_not_kernel[1](dst)

def main():
    dst = torch.zeros(64, dtype=torch.float32, device="npu:0")

    run(dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.where(torch.arange(64) < 16, 0.0, 1.0))
    print("mask_not example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
mask_not example passed
first=0.0000, last=1.0000
```
