# vstore_mask
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将矢量数据寄存器中的数据以对齐的方式搬出到Unified Buffer（UB）。单次搬出量为VL（256字节），支持配置掩码用于指示参与搬出的元素，掩码为1的元素写入目的地址，掩码为0的元素保留目的地址原数据。搬运过程中数据格式与内容保持不变。目的操作数为UB地址。掩码寄存器中的数据搬出由[`vmask_store`](./vmask-store.md)提供。

矢量数据寄存器搬出场景提供以下两种模式：

- **连续对齐搬出模式**：将矢量数据寄存器中的数据搬出到UB目的地址，由用户自行更新目的地址。
- **立即数偏移搬出模式**：通过立即数指定相对目的起始地址的偏移，单位为元素，用户可选择更新偏移或更新目的地址。

非连续对齐搬出`DataBlock`时，请使用[`vstore_strided`](./vstore-strided.md)。

## 函数原型

```python
vstore_mask(tensor, offset, value: RawVReg, mask: Mask, *, merge_mode: 'str | MergeMode' = MergeMode.ZEROING) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `tensor` | 输出 | 目的操作数（矢量）的起始地址。起始地址需32字节对齐。 |
| `offset` | 输入 | 相对`tensor`起始地址的偏移量，类型为`dtypes.int32`，单位为元素。<br>通过`offset`参数偏移后的实际访问地址落在UB地址范围内，且实际访问地址仍需32字节对齐，否则会报错。 |
| `value` | 输入 | 源操作数（矢量数据寄存器）。dtype须与`tensor`一致。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示参与搬出的元素。对应位置为1时参与搬出，为0时不参与搬出。 |
| `merge_mode` | 输入 | 掩码合并模式，类型为`MergeMode`或对应的字符串，默认值为`MergeMode.ZEROING`。本搬出接口对未选中的目的地址均保留原值，此参数不改变这一行为。 |

**支持的元素数据类型：**

dtype支持的数据类型为`dtypes.int4x2`、`dtypes.int8`、`dtypes.uint8`、`dtypes.fp4x2_e1m2`、`dtypes.fp4x2_e2m1`、`dtypes.hifloat8`、`dtypes.float8_e8m0`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 返回值说明

无返回值。结果写入`tensor`。

## 约束说明

### 通用约束

- 本接口需在VF作用域内调用。
- UB容量上限为256KB，用户可用容量随编译选项与编程场景变化（默认预留6KB SIMD VF栈+2KB框架预留，可用248KB；SIMD+SIMT混编时再划分32KB~128KB作Data Cache，可用容量进一步减少）。目的操作数地址偏移后不可超过实际可用容量，否则会报错。
- 如果本指令与其他指令存在UB地址重叠，需要插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化，防止出现异常数据。

### 矢量数据寄存器搬出场景

- `mask`比特位为0的位置采用保持模式：`tensor`对应位置保持原值不变，不写入`value`数据。

## 调用示例

以下示例演示`vstore_mask`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import create_mask, full_mask, vdups, vload, vmem_bar, vstore, vstore_mask
from cannbotdsl.tensor import MemLoc

@kernel
def _vstore_mask_kernel(src0, dst):
    buf = Buffer(MemLoc.UB, src0.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf, src0)

    in0 = buf
    res = out
    with vf(mode="simd"):
        full = full_mask()
        part = create_mask(pattern="vl32", elem_bits=32)
        vstore(res, 0, vdups(0.0, dtypes.float32), full)
        vmem_bar()
        vstore_mask(res, 0, vload(in0, 0), part)

    mem_copy(dst, out)

@host
def run(src0, dst):
    _vstore_mask_kernel[1](src0, dst)

def main():
    src0 = torch.arange(1, 65, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)
    expected = src0.cpu().clone()
    expected[32:] = 0.0

    run(src0, dst)
    torch.npu.synchronize()

    assert bool((dst.cpu() == expected).all()), f"masked store mismatch: {dst.cpu()[:4].tolist()}"
    print("vstore_mask example passed")
    print(f"kept={int((dst.cpu() != 0).sum())}, last_kept={float(dst.cpu()[31]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vstore_mask example passed
kept=32, last_kept=32.0000
```
