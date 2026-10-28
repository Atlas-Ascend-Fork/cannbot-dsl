# vstore_unalign_begin
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

开启一组连续搬出操作，对AR特殊寄存器进行清零操作，并返回非对齐寄存器`ureg`。AR寄存器用于配合[`vsqueeze_and_storeunalign`](./vsqueeze-and-storeunalign.md)及[`vsqueeze_and_storeunalign_finalize`](./vsqueeze-and-storeunalign-finalize.md)使用：当调用`vsqueeze_and_storeunalign`后，有效元素的总字节数会被存入AR寄存器用于接口内自动地址偏移。在首次调用`vsqueeze_and_storeunalign`之前，需调用本接口将AR寄存器清零。

## 函数原型

```python
vstore_unalign_begin(tensor, *, no_clear_ar: bool = False) -> StoreUnalignReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `tensor` | 输出 | 位于Unified Buffer（UB）中的目的张量，后续搬出操作将结果写入该张量。 |
| `no_clear_ar` | 输入 | 是否跳过AR寄存器清零：取值为`False`时本接口将AR寄存器清零；取值为`True`时不执行清零，本次搬出操作在上一组操作剩余的AR偏移基础上继续累加，默认为`False`。 |

## 返回值说明

- 返回非对齐搬出状态句柄`ureg`，类型为`StoreUnalignReg`，用于后续`vsqueeze_and_storeunalign`和`vsqueeze_and_storeunalign_finalize`调用。

## 约束说明

- 每组连续搬出操作开始前，需调用一次本接口，再调用`vsqueeze_and_storeunalign`。如果在一组连续搬出过程中再次调用本接口，AR寄存器记录的字节偏移会被重置，后续数据可能覆盖已经写入的结果。
- 开始新一组操作前，需先调用`vsqueeze_and_storeunalign_finalize`完成上一组操作，避免上一组暂存在非对齐寄存器中的尾块数据丢失。
- 本接口执行后，首次调用`vsqueeze_and_storeunalign`时使用的非对齐寄存器无需预先初始化。
- 本接口需在VF作用域内调用。

## 调用示例

以下示例演示`vstore_unalign_begin`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import (
    create_mask,
    vload,
    vsqueeze_and_storeunalign_finalize,
    vsqueeze_and_storeunalign_init,
    vstore_unalign,
    vstore_unalign_begin,
)
from cannbotdsl.tensor import MemLoc

@kernel
def _vstore_unalign_begin_kernel(src0, dst):
    buf = Buffer(MemLoc.UB, src0.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf, src0)

    in0 = buf
    res = out
    with vf(mode="simd"):
        mask = create_mask(pattern="m3", elem_bits=32)
        sq = vsqueeze_and_storeunalign_init(vload(in0, 0), mask=mask)
        ureg = vstore_unalign_begin(res)
        vstore_unalign(res, 0, sq, ureg)
        vsqueeze_and_storeunalign_finalize(res, 0, ureg)

    mem_copy(dst, out)

@host
def run(src0, dst):
    _vstore_unalign_begin_kernel[1](src0, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu()[:22], src0.cpu()[0::3])
    print("vstore_unalign_begin example passed")
    print(f"first={float(dst.cpu()[0]):.1f}, last={float(dst.cpu()[21]):.1f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vstore_unalign_begin example passed
first=0.0, last=63.0
```
