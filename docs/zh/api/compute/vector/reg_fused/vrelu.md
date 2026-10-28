# vrelu
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据`mask`对源操作数`src`按元素执行ReLU操作，得到计算结果。计算公式如下：

$$
dst_i = (src_i >= 0) ? src_i : 0
$$

## 函数原型

```python
vrelu(src: RawVReg, *, mask: Mask) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 掩码寄存器，用于控制各元素是否参与计算。`mask`中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。<br>`mask`掩码位为0时，结果对应元素置0。 |

**支持的元素数据类型：**

`dtypes.float16`、`dtypes.int32`、`dtypes.float32`。

## 返回值说明

返回保存ReLU计算结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`src`一致。`mask`掩码位为0的元素在返回值中置0。

## 约束说明

- 本接口需在`vf(mode="simd")`作用域内调用。

## 调用示例

以下示例演示`vrelu`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload, vrelu, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vrelu_kernel(src, dst):
    buf = Buffer(MemLoc.UB, src.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf, src)

    in0 = buf
    res = out
    with vf(mode="simd"):
        mask = full_mask()
        vstore(res, 0, vrelu(vload(in0, 0), mask=mask), mask)

    mem_copy(dst, out)

@host
def run(src, dst):
    _vrelu_kernel[1](src, dst)

def main():
    src = torch.arange(64, dtype=torch.float32, device="npu:0") - 32.0
    dst = torch.empty_like(src)

    run(src, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.relu(src).cpu())
    print("vrelu example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vrelu example passed
first=0.0000, last=31.0000
```
