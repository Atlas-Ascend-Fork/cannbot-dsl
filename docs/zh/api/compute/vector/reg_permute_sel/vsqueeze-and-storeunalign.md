# vsqueeze_and_storeunalign
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

Reg 计算数据搬运接口。将压缩后的源操作数`value`中的元素连续搬出到Unified Buffer（UB）的`tensor`中。压缩后的元素按其在源操作数中的顺序从低位开始连续排列，`value`中的剩余元素置0。

本接口使用AR特殊寄存器记录当前连续搬出的有效数据总字节数。首次调用本接口前，需调用[`vsqueeze_and_storeunalign_init`](./vsqueeze-and-storeunalign-init.md)标记压缩点，由[`vstore_unalign_begin`](./vstore-unalign-begin.md)清零AR寄存器。接口执行前，AR寄存器中的值表示相对于`tensor`的写入偏移，单位为字节；接口执行后，本次选中元素的总字节数会累加到AR寄存器中。连续调用本接口并保持`tensor`和`ureg`不变，可将多组筛选结果连续写入UB，无需手动更新地址。AR寄存器中的值可通过[`get_squeeze_status`](../../../execution/device/get-squeeze-status.md)接口获取。

接口将可完整组成32B的主块写入UB，未满32B的尾块暂存在非对齐寄存器`ureg`中。完成一次或多次连续搬出后，需调用[`vsqueeze_and_storeunalign_finalize`](./vsqueeze-and-storeunalign-finalize.md)将尾块写入UB。

## 函数原型

```python
vsqueeze_and_storeunalign(tensor, offset, value: RawVReg, ureg: StoreUnalignReg) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `tensor` | 输出 | 位于Unified Buffer（UB）中的目的张量，用于接收连续搬出的有效元素。<br>`tensor`的地址无需32B对齐，但需按`sizeof(dtype)`字节对齐。 |
| `offset` | 输入 | 相对于`tensor`的元素偏移。 |
| `value` | 输入 | 源操作数（矢量数据寄存器），取`vsqueeze_and_storeunalign_init`返回的压缩结果。 |
| `ureg` | 输入 | 非对齐寄存器，由`vstore_unalign_begin`返回。 |

**支持的元素数据类型：**

`dtypes.int4x2`、`dtypes.int8`、`dtypes.uint8`、`dtypes.fp4x2_e2m1`、`dtypes.fp4x2_e1m2`、`dtypes.float8_e8m0`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 返回值说明

无返回值。结果写入`tensor`。

## 约束说明

- 本接口运行在`[PIPE](../../../execution/pipeline.md).V`流水上。

### 通用约束

- UB容量上限为256KB，用户可用容量随编译选项与编程场景变化。`tensor`加AR寄存器记录的累计字节数不可超过实际可用容量，否则会触发写越界异常。
- 如果本接口与其他指令存在UB地址重叠，需要插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化。

### 指令约束

- 本接口需在VF作用域内调用。
- 每组连续搬出操作开始前，需调用一次`vsqueeze_and_storeunalign_init`。连续调用本接口时，需保持`tensor`和`ureg`不变；最后一次调用后，需调用一次`vsqueeze_and_storeunalign_finalize`完成收尾。

## 调用示例

以下示例演示`vsqueeze_and_storeunalign`的调用，并在Host侧校验结果。

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
def _vsqueeze_and_storeunalign_kernel(src0, dst):
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
    _vsqueeze_and_storeunalign_kernel[1](src0, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu()[:22], src0.cpu()[0::3])
    print("vsqueeze_and_storeunalign example passed")
    print(f"first={float(dst.cpu()[0]):.1f}, last={float(dst.cpu()[21]):.1f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vsqueeze_and_storeunalign example passed
first=0.0, last=63.0
```
