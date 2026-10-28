# vnes
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据`mask`将源操作数`src`按元素与标量`scalar`进行比较，若$src_i \neq scalar$，则对应结果位为1，否则为0。每个元素的比较结果占一个比特。计算公式如下：

$$
dst_i =
\begin{cases}
 1, & src_i \neq scalar \\
 0, & src_i = scalar \\
\end{cases}
$$

## 函数原型

```python
vnes(src: RawVReg, scalar, *, mask: Mask) -> Mask
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `scalar` | 输入 | 源操作数（标量）。 |
| `mask` | 输入 | 掩码寄存器，用于控制各元素是否参与比较。`mask`中与元素对应的比特位为1时，该元素参与比较；为0时，该元素不参与比较。<br>`mask`掩码位为0时，结果对应比特位置0。 |

**支持的元素数据类型：**

`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 返回值说明

返回保存逐元素比较结果的掩码寄存器值，类型为`Mask`。`mask`掩码位为0的元素在返回值中对应比特位置0。

## 约束说明

- 本接口需在`vf(mode="simd")`作用域内调用。

- `src`与`scalar`的数据类型需要保持一致。
- 浮点比较输入包含nan时，结果对应比特位写1。

## 调用示例

以下示例演示`vnes`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vdups, vload, vnes, vselect, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vnes_kernel(src, dst):
    buf = Buffer(MemLoc.UB, src.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))
    mem_copy(buf, src)

    in0, res = buf, out
    with vf(mode="simd"):
        mask = full_mask()
        pred = vnes(vload(in0, 0), 32.0, mask=mask)
        r = vselect(vdups(1.0, dtypes.float32),
                           vdups(0.0, dtypes.float32), cond_mask=pred)
        vstore(res, 0, r, mask)
    mem_copy(dst, out)

@host
def run(src, dst):
    _vnes_kernel[1](src, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 != 32.0).float().cpu())
    print(f"count={int(dst.sum().item())}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
count=63
```
