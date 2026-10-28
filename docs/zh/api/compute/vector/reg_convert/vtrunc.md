# vtrunc
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将`src`中的浮点数元素按照TRUNC（向零方向截断）截断到整数值，结果仍保持原浮点数据类型。接口通过函数返回值返回结果。截断规则等价于C标准库`trunc`语义：向零方向舍入，即丢弃小数部分取整，正数向0方向取整（如1.7 -> 1.0），负数向0方向取整（如-1.7 -> -1.0），整数值保持不变。未被`mask`筛选的元素置零。

关于舍入模式的详细说明，请参见[舍入模式与饱和模式](./roundingmode.md)。

计算公式如下：

$$
dst[i] = trunc(src[i])
$$

## 函数原型

```python
vtrunc(src: RawVReg, *, mask: Mask) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源矢量数据寄存器，提供待截断的浮点元素。 |
| `mask` | 输入 | 源掩码寄存器，用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。`mask`未筛选的元素在输出中置零。 |

**支持的元素数据类型：**

源操作数`src`的元素数据类型支持`dtypes.float16`、`dtypes.bfloat16`、`dtypes.float32`。

## 返回值说明

返回保存向零截断结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`src`一致，仍为浮点类型。

## 约束说明

- 本接口需在VF作用域内调用，`src`为矢量数据寄存器，`mask`为掩码寄存器。
- `dtypes.float32`只支持不饱和模式。

## 调用示例

以下示例演示`vtrunc`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload, vstore, vtrunc
from cannbotdsl.tensor import MemLoc

@kernel
def _vtrunc_kernel(src, dst):
    buf = Buffer(MemLoc.UB, src.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf, src)

    in0 = buf
    res = out
    with vf(mode="simd"):
        mask = full_mask()
        vstore(res, 0, vtrunc(vload(in0, 0), mask=mask), mask)

    mem_copy(dst, out)

@host
def run(src, dst):
    _vtrunc_kernel[1](src, dst)

def main():
    src = (torch.arange(64, dtype=torch.float32, device="npu:0") - 32.0) / 3.0
    dst = torch.empty_like(src)

    run(src, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.trunc(src).cpu())
    print("vtrunc example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vtrunc example passed
first=-10.0000, last=10.0000
```
