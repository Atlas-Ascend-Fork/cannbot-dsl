# vsub
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

该接口根据`mask`，对源操作数`lhs`、`rhs`进行按元素求差操作，将结果作为返回值返回。

**无借位输出减法**：单目的操作数，矢量数据寄存器写入减法结果，计算公式如下：

$$
dst_i = lhs_i - rhs_i
$$

## 函数原型

```python
vsub(lhs: RawVReg, rhs: RawVReg, *, mask: Mask) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `lhs` | 输入 | 源操作数（矢量数据寄存器）。 |
| `rhs` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

**支持的元素数据类型：**

`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 返回值说明

返回保存逐元素减法结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`lhs`、`rhs`一致。

## 约束说明

- 本接口需在`vf(mode="simd")`作用域内调用。

### 无借位输出减法约束

- 整型dtype减法溢出时按环绕（wrap-around）策略处理：有符号类型溢出回绕到对应dtype的最小值或最大值（MIN-1->MAX，MAX+1->MIN），无符号类型下溢回绕到对应dtype的最大值（0-1->UMAX）。
- 浮点dtype（`dtypes.float16`、`dtypes.bfloat16`、`dtypes.float32`）减法按IEEE 754浮点减法语义执行，溢出与无效输入（如nan、inf）的处理遵循浮点运算规则。

## 调用示例

以下示例演示`vsub`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload, vstore, vsub
from cannbotdsl.tensor import MemLoc

@kernel
def _vsub_kernel(src0, src1, dst):
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
        vstore(res, 0, vsub(vload(in0, 0), vload(in1, 0), mask=mask), mask)

    mem_copy(dst, out)

@host
def run(src0, src1, dst):
    _vsub_kernel[1](src0, src1, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0") + 10.0
    src1 = torch.arange(64, dtype=torch.float32, device="npu:0")
    dst = torch.zeros_like(src0)

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 - src1).cpu())
    print("vsub example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vsub example passed
first=10.0000, last=10.0000
```
