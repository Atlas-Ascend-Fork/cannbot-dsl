# vmerge
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据`mask`从源操作数`new`、`old`中选择元素，得到计算结果。选择的规则为：当`mask`的比特位为1时，从`new`中选取对应位置的数；当`mask`的比特位为0时，从`old`中选取对应位置的数。计算公式如下：

$$
dst_i =
\begin{cases}
 new_i, & mask_i = 1 \\
 old_i, & mask_i = 0 \\
\end{cases}
$$

## 函数原型

```python
vmerge(mask: Mask, new: RawVReg, old: RawVReg) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `mask` | 输入 | 源操作数掩码（掩码寄存器）。指定选择`new`或`old`为有效数据。`mask`的比特位为1时，选取`new`；`mask`的比特位为0时，选取`old`。 |
| `new` | 输入 | 源操作数（矢量数据寄存器）。`mask`的比特位为1时选取该操作数对应位置的元素。 |
| `old` | 输入 | 源操作数（矢量数据寄存器）。`mask`的比特位为0时选取该操作数对应位置的元素。 |

**支持的元素数据类型：**

`dtypes.bool`、`dtypes.int8`、`dtypes.uint8`、`dtypes.hifloat8`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 返回值说明

返回保存选择结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`new`、`old`一致。

## 约束说明

- 本接口需在VF作用域内调用。
- `new`和`old`的数据类型需要保持一致。

## 调用示例

以下示例演示`vmerge`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vgt, vload, vmerge, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vmerge_kernel(src0, src1, dst):
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
        a = vload(in0, 0)
        b = vload(in1, 0)
        vstore(res, 0, vmerge(vgt(a, b, mask=mask), a, b), mask)

    mem_copy(dst, out)

@host
def run(src0, src1, dst):
    _vmerge_kernel[1](src0, src1, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    src1 = torch.full((64,), 32.0, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.where(src0.cpu() > src1.cpu(), src0.cpu(), src1.cpu()))
    print("vmerge example passed")
    print(f"first={float(dst.cpu()[0]):.1f}, last={float(dst.cpu()[-1]):.1f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vmerge example passed
first=32.0, last=63.0
```
