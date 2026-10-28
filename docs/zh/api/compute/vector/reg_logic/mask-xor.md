# mask_xor
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据`exec_mask`对两个源掩码寄存器`lhs`和`rhs`进行按位异或（^）操作，将结果作为返回值返回。未被`exec_mask`筛选的位置被置为0。

## 函数原型

```python
mask_xor(lhs: Mask, rhs: Mask, *, exec_mask: Mask) -> Mask
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `lhs` | 输入 | 源掩码寄存器，参与按位异或的操作数。 |
| `rhs` | 输入 | 源掩码寄存器，参与按位异或的操作数。 |
| `exec_mask` | 输入 | 源掩码寄存器，用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。`exec_mask`未筛选的元素在输出中置零。 |

**支持的元素数据类型：**

掩码寄存器不区分元素数据类型，仅由元素位宽`elem_bits`标识；参与计算的所有掩码寄存器必须具有相同的`elem_bits`。

## 返回值说明

返回保存按位异或结果的掩码寄存器值，类型为`Mask`，掩码粒度与`lhs`、`rhs`一致。

## 约束说明

- 本接口需在`vf(mode="simd")`作用域内调用。

- `exec_mask`比特位为0时，计算结果对应比特位写0。
- `lhs`、`rhs`与`exec_mask`的`elem_bits`必须一致，否则会报错。
- 未被`exec_mask`筛选的位置在返回值中置为0。

## 调用示例

以下示例对两个掩码执行按位异或操作，并在Host侧校验选择结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import create_mask, full_mask, mask_xor, vdups, vselect, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _mask_xor_kernel(dst):
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    res = out
    with vf(mode="simd"):
        full = full_mask()
        m = mask_xor(create_mask(pattern="m3"), create_mask(pattern="h"),
                            exec_mask=full)
        one = vdups(1.0, dtypes.float32)
        zero = vdups(0.0, dtypes.float32)
        vstore(res, 0, vselect(one, zero, cond_mask=m), full)

    mem_copy(dst, out)

@host
def run(dst):
    _mask_xor_kernel[1](dst)

def main():
    dst = torch.zeros(64, dtype=torch.float32, device="npu:0")
    idx = torch.arange(64)

    run(dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.where((idx % 3 == 0) ^ (idx < 32), 1.0, 0.0))
    print("mask_xor example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
mask_xor example passed
first=0.0000, last=1.0000
```
