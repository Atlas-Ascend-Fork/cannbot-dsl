# vneg
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据`mask`对源操作数`src`按元素取相反数，并将结果作为返回值返回。

计算公式如下：

$$
dst_i = -src_i
$$

## 函数原型

```python
vneg(src: RawVReg, *, mask: Mask) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。mask中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。 |

**支持的元素数据类型：**

`dtypes.int8`、`dtypes.int16`、`dtypes.float16`、`dtypes.int32`、`dtypes.float32`。

## 返回值说明

返回保存逐元素取反结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`src`一致。

## 约束说明

- 本接口需在VF作用域内调用，`src`为矢量数据寄存器。
- `mask`比特位为0时，计算结果对应比特位写0。
- 整数输入为有符号整数类型的最小负值时，结果保留原值不变。
- 浮点输入通过翻转符号位实现取反：正数变为对应负数，负数变为对应正数，+0.0变为-0.0，-0.0变为+0.0，+inf变为-inf，-inf变为+inf。
- 浮点输入nan，取反结果为nan。

## 调用示例

以下示例演示`vneg`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload, vneg, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vneg_kernel(src, dst):
    buf = Buffer(MemLoc.UB, src.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf, src)

    in0 = buf
    res = out
    with vf(mode="simd"):
        mask = full_mask()
        vstore(res, 0, vneg(vload(in0, 0), mask=mask), mask)

    mem_copy(dst, out)

@host
def run(src, dst):
    _vneg_kernel[1](src, dst)

def main():
    src = torch.arange(64, dtype=torch.float32, device="npu:0") - 32.0
    dst = torch.zeros_like(src)

    run(src, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (-src).cpu())
    print("vneg example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vneg example passed
first=32.0000, last=-31.0000
```
