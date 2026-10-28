# vsqueeze_and_storeunalign_init
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

标记一组连续搬出操作的压缩点。将`src`中`mask`比特位为1的元素压缩后作为返回值返回，返回值需与`vsqueeze_and_storeunalign`一一对应使用，由`vsqueeze_and_storeunalign`搬出到UB。本接口自身不读写AR特殊寄存器。

AR寄存器用于配合`vsqueeze_and_storeunalign`及`vsqueeze_and_storeunalign_finalize`使用：调用`vsqueeze_and_storeunalign`后，有效元素的总字节数会被存入AR寄存器用于接口内自动地址偏移。在一组连续搬出操作开始前，需调用`vstore_unalign_begin`将AR寄存器清零。

## 函数原型

```python
vsqueeze_and_storeunalign_init(src: RawVReg, *, mask: Mask) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。mask中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。 |

## 返回值说明

返回与`src`及`mask`关联、供配套压缩搬出接口使用的输入值，类型为`RawVReg`，其中元素的数据类型与`src`一致。该值须按约束说明用于后续压缩搬出，不能作为独立压缩结果任意使用。

## 约束说明

- 本接口需在VF作用域内调用。
- 每组连续搬出操作开始前，需调用一次本接口，再调用`vsqueeze_and_storeunalign`。本接口的返回值只能被`vsqueeze_and_storeunalign`使用一次，否则会报错。
- 开始新一组操作前，需先调用`vsqueeze_and_storeunalign_finalize`完成上一组操作，避免上一组暂存在非对齐寄存器中的尾块数据丢失。
- 本接口执行后，首次调用`vsqueeze_and_storeunalign`时使用的非对齐寄存器无需预先初始化。

## 调用示例

以下示例演示`vsqueeze_and_storeunalign_init`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import (
    create_mask,
    vload,
    vsqueeze_and_storeunalign,
    vsqueeze_and_storeunalign_finalize,
    vsqueeze_and_storeunalign_init,
    vstore_unalign_begin,
)
from cannbotdsl.tensor import MemLoc

@kernel
def _vsqueeze_and_storeunalign_init_kernel(src0, dst):
    buf = Buffer(MemLoc.UB, src0.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf, src0)

    in0 = buf
    res = out
    with vf(mode="simd"):
        mask = create_mask(pattern="m3", elem_bits=32)
        sq = vsqueeze_and_storeunalign_init(vload(in0, 0), mask=mask)
        ureg = vstore_unalign_begin(res)
        vsqueeze_and_storeunalign(res, 0, sq, ureg)
        vsqueeze_and_storeunalign_finalize(res, 0, ureg)

    mem_copy(dst, out)

@host
def run(src0, dst):
    _vsqueeze_and_storeunalign_init_kernel[1](src0, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu()[:22], src0.cpu()[0::3])
    print("vsqueeze_and_storeunalign_init example passed")
    print(f"first={float(dst.cpu()[0]):.1f}, last={float(dst.cpu()[21]):.1f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vsqueeze_and_storeunalign_init example passed
first=0.0, last=63.0
```
