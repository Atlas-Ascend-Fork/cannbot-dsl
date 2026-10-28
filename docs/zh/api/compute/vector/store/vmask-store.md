# vmask_store
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将掩码寄存器中的数据以对齐的方式搬出到Unified Buffer（UB），单次搬出量为掩码寄存器长度（32字节），不支持配置掩码。搬运过程中数据格式与内容保持不变。目的操作数为UB地址。

## 函数原型

```python
vmask_store(tensor, offset, mask: Mask, *, dist = 'norm') -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `tensor` | 输出 | 目的操作数在UB中的基地址，起始地址需32字节对齐。 |
| `offset` | 输入 | 相对`tensor`起始地址的偏移量，类型为`dtypes.int32`，单位为元素个数。<br>`offset`的单位为元素个数。通过`offset`参数偏移后的实际访问地址需落在UB地址范围内，且仍需32字节对齐，否则会报错。 |
| `mask` | 输入 | 源操作数（掩码寄存器）。 |
| `dist` | 输入 | 掩码搬出模式。取值为`'norm'`（连续对齐搬出）或`'pack'`（每间隔1 bit丢弃1 bit，将保留的bit连续压缩存储）。默认值为`'norm'`。 |

## 返回值说明

无返回值。结果写入`tensor`。

## 约束说明

- 本接口需在VF作用域内调用。
- UB容量上限为256KB，用户可用容量随编译选项与编程场景变化（默认预留6KB SIMD VF栈+2KB框架预留，可用248KB；SIMD+SIMT混编时再划分32KB~128KB作Data Cache，可用容量进一步减少）。目的操作数地址偏移后不可超过实际可用容量，否则会报错。
- 如果本指令与其他指令存在UB地址重叠，需要插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化，防止出现异常数据。

## 调用示例

以下示例演示`vmask_store`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import create_mask, vmask_store
from cannbotdsl.tensor import MemLoc

@kernel
def _vmask_store_kernel(dst):
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((8,)))

    res = out
    with vf(mode="simd"):
        mask = create_mask(pattern="vl32", elem_bits=32)
        vmask_store(res, 0, mask)

    mem_copy(dst, out)

@host
def run(dst):
    _vmask_store_kernel[1](dst)

def main():
    dst = torch.zeros((8,), dtype=torch.int32, device="npu:0")

    run(dst)
    torch.npu.synchronize()

    words = [int(w) & 0xFFFFFFFF for w in dst.cpu().tolist()]
    assert sum(w.bit_count() for w in words) == 32, f"unexpected mask layout: {words}"
    print("vmask_store example passed")
    print(f"word0={words[0]:#010x}, set_bits={sum(w.bit_count() for w in words)}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vmask_store example passed
word0=0x11111111, set_bits=32
```
