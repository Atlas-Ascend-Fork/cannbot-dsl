# vstorealign_squeeze_status
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将`vsqueeze_and_storeunalign`操作后保存在AR特殊寄存器中的当前累计有效数据字节数，以一个32位状态值写入Unified Buffer（UB）。本接口只有一个函数原型，通过`post_update`选择以下两种模式：

- **普通偏移模式（`post_update=False`）**：写入`tensor[offset]`，`offset=0`表示写入张量起点，不自动推进地址。
- **Post Update模式（`post_update=True`）**：先写入当前游标位置，再将游标推进`offset`个32位元素。同一目的张量的游标在VF入口从张量起点初始化，并在该VF内的后续调用和运行时循环迭代中持续推进；不会修改Python张量对象本身。

## 函数原型

```python
vstorealign_squeeze_status(tensor, offset = 0, *, post_update: bool = False) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `tensor` | 输出 | UB中的目的张量，元素位宽必须为32位，用于接收AR中的状态值。普通模式从指定偏移写入，Post Update模式的游标从该张量起点开始。 |
| `offset` | 输入 | 普通模式下为相对`tensor`起点的访问偏移，Post Update模式下为每次写入后的游标推进量。可传入整数常量或整型DSL标量，单位均为32位元素，默认值为`0`。一个元素对应4字节，不是字节偏移。 |
| `post_update` | 输入 | 是否在写入后自动推进内部目的地址游标，取值为`True`或`False`，默认值为`False`。取值为`True`时每次写入后推进`offset × 4`字节；`offset=0`时不推进。 |

## 返回值说明

无返回值。结果写入`tensor`。

## 约束说明

- 本接口需在VF作用域内调用。
- 两种模式下每次实际写入的4字节范围都必须位于实际可用UB范围内，且实际写入地址需4字节对齐；Post Update模式按当前游标判断范围，而不是将`offset`当作本次起始偏移。
- 调用本接口前，需使用[`vsqueeze_and_storeunalign`](../reg_permute_sel/vsqueeze-and-storeunalign.md)完成squeeze操作；写入的值是其当前累计的有效数据字节数。
- 如果本接口与其他指令存在UB地址重叠，需要插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化。

## 调用示例

以下示例演示`vstorealign_squeeze_status`的调用，并在Host侧校验结果。

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
    vsqueeze_and_storeunalign_init,
    vstore_unalign_begin,
    vstorealign_squeeze_status,
)
from cannbotdsl.tensor import MemLoc

@kernel
def _vstorealign_squeeze_status_kernel(src0, dst, status):
    buf = Buffer(MemLoc.UB, src0.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))
    st = Buffer(MemLoc.UB, status.dtype, make_layout((1,)))

    mem_copy(buf, src0)

    in0 = buf
    res = out
    res2 = st
    with vf(mode="simd"):
        mask = create_mask(pattern="vl32", elem_bits=32)
        ureg = vstore_unalign_begin(res)
        sq = vsqueeze_and_storeunalign_init(vload(in0, 0), mask=mask)
        vsqueeze_and_storeunalign(res, 0, sq, ureg)
        vstorealign_squeeze_status(res2, 0)

    mem_copy(dst, out)
    mem_copy(status, st)

@host
def run(src0, dst, status):
    _vstorealign_squeeze_status_kernel[1](src0, dst, status)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)
    status = torch.zeros((1,), dtype=torch.int32, device="npu:0")

    run(src0, dst, status)
    torch.npu.synchronize()

    assert int(status.cpu()[0]) == 128, f"squeeze status is not 128 bytes: {int(status.cpu()[0])}"
    print("vstorealign_squeeze_status example passed")
    print(f"status={int(status.cpu()[0])}, first={float(dst.cpu()[0]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vstorealign_squeeze_status example passed
status=128, first=0.0000
```
