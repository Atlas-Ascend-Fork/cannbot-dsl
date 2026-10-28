# vmask
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将`src`中被`mask`筛选的有效元素复制到返回值对应位置；返回值的寄存器由本接口分配、未经初始化，因此未被`mask`筛选的位置值未定义。

## 函数原型

```python
vmask(src: RawVReg, *, mask: Mask) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源矢量数据寄存器，`dtype`须与返回的矢量数据寄存器完全一致。 |
| `mask` | 输入 | 源掩码寄存器，用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。`mask`未筛选的元素在输出中值未定义。 |

**支持的元素数据类型：**

`dtype`支持的数据类型：`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 返回值说明

返回保存掩码复制结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`src`一致。

## 约束说明

- 本接口需在VF作用域内调用。
- 未被`mask`筛选的位置值未定义：返回值对应的寄存器由本接口分配、未经初始化，调用方不应依赖未被`mask`筛选的位置的值。
- `mask`的掩码位宽须与`src`的元素位宽一致（由掩码设置接口按`src`的元素位宽创建），否则有效元素位置错误。

## 调用示例

以下示例演示`vmask`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import create_mask, full_mask, vload, vmask, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vmask_kernel(src, dst):
    buf = Buffer(MemLoc.UB, src.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf, src)

    in0 = buf
    res = out
    with vf(mode="simd"):
        full = full_mask()
        half = create_mask(pattern="vl32")
        vstore(res, 0, vmask(vload(in0, 0), mask=half), full)

    mem_copy(dst, out)

@host
def run(src, dst):
    _vmask_kernel[1](src, dst)

def main():
    src = torch.arange(64, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src)

    run(src, dst)
    torch.npu.synchronize()

    # mask 未筛选的位置值未定义，仅校验已筛选的前32个元素。
    torch.testing.assert_close(dst.cpu()[:32], src.cpu()[:32])
    print("vmask example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, lane31={float(dst.cpu()[31]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vmask example passed
first=0.0000, lane31=31.0000
```
