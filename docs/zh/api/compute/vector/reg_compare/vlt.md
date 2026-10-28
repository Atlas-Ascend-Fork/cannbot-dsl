# vlt
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

lt表示小于（less than），该接口用于逐元素比较两个源操作数大小，将比较结果（$lhs_i < rhs_i$）作为返回值返回。如果比较结果为真，则对应比特位为1，否则为0。

计算公式如下：

$$
dst_i = (lhs_i < rhs_i)
$$

## 函数原型

```python
vlt(lhs: RawVReg, rhs: RawVReg, *, mask: Mask) -> Mask
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `lhs` | 输入 | 源操作数（矢量数据寄存器）。 |
| `rhs` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。mask中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。 |

**支持的元素数据类型：**

`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 返回值说明

返回保存逐元素比较结果的掩码寄存器值，类型为`Mask`。

## 约束说明

- 本接口需在VF作用域内调用，`lhs`、`rhs`为矢量数据寄存器。
- `mask`比特位为0时，计算结果对应比特位写0。
- 浮点比较时，+0.0与-0.0视为相等。
- 浮点比较输入含nan时，计算结果对应比特位写0。

## 调用示例

以下示例演示`vlt`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vdups, vload, vlt, vselect, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vlt_kernel(src0, src1, dst):
    buf0 = Buffer(MemLoc.UB, src0.dtype, make_layout((64,)))
    buf1 = Buffer(MemLoc.UB, src1.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))
    mem_copy(buf0, src0)
    mem_copy(buf1, src1)

    in0, in1, res = buf0, buf1, out
    with vf(mode="simd"):
        mask = full_mask()
        pred = vlt(vload(in0, 0), vload(in1, 0), mask=mask)
        r = vselect(vdups(1.0, dtypes.float32),
                           vdups(0.0, dtypes.float32), cond_mask=pred)
        vstore(res, 0, r, mask)
    mem_copy(dst, out)

@host
def run(src0, src1, dst):
    _vlt_kernel[1](src0, src1, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    src1 = torch.flip(src0, [0])
    dst = torch.empty_like(src0)

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 < src1).float().cpu())
    print(f"count={int(dst.sum().item())}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
count=32
```
