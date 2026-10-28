# vleakyrelu
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据掩码对源矢量数据寄存器中的元素执行Leaky ReLU操作，并将结果作为返回值返回。源操作数中大于0的元素直接写入输出结果，小于或等于0的元素乘以标量`slope`后写入输出结果。掩码对应位置为1的元素参与计算，为0的元素在输出结果中置零。计算公式如下：

$$
dst_i = \begin{cases} src_i & src_i > 0 \\ src_i \times \mathrm{slope} & src_i \leq 0 \end{cases}
$$

## 函数原型

```python
vleakyrelu(src: RawVReg, slope, *, mask: Mask) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `slope` | 输入 | 负半轴斜率（标量）。数据类型须与`src`一致。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。`mask`未筛选的元素在输出中置零。 |

**支持的元素数据类型：**

源操作数`src`的元素数据类型支持`dtypes.float16`、`dtypes.float32`。标量`slope`的数据类型须与`src`的元素数据类型一致。

## 返回值说明

返回保存Leaky ReLU计算结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`src`一致。

## 约束说明

- 本接口需在`vf(mode="simd")`作用域内调用。

### 通用约束

- 掩码位为0的元素位置不参与运算，输出结果对应位置写0。

### 计算约束

- 源矢量数据寄存器中数据为负零（-0）时按负数处理，乘法满足IEEE 754浮点乘法规则。

## 调用示例

以下示例演示`vleakyrelu`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vleakyrelu, vload, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vleakyrelu_kernel(src, dst):
    buf = Buffer(MemLoc.UB, src.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf, src)

    in0 = buf
    res = out
    with vf(mode="simd"):
        mask = full_mask()
        vstore(res, 0, vleakyrelu(vload(in0, 0), 0.1, mask=mask), mask)

    mem_copy(dst, out)

@host
def run(src, dst):
    _vleakyrelu_kernel[1](src, dst)

def main():
    src = torch.arange(64, dtype=torch.float32, device="npu:0") - 32.0
    dst = torch.empty_like(src)

    run(src, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.nn.functional.leaky_relu(src, 0.1).cpu())
    print("vleakyrelu example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vleakyrelu example passed
first=-3.2000, last=31.0000
```
