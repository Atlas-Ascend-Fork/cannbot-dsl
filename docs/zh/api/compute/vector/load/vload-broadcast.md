# vload_broadcast
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

从Unified Buffer（UB）读取单个元素、8个源元素或一个完整32B DataBlock，按`mode`指定方式广播到矢量寄存器。支持普通元素偏移及Post Update地址自更新。不同模式的读取量、数据排列和对齐要求见[mode枚举与模式选择](#mode枚举与模式选择)。

- **对齐搬入模式**：将UB源地址的数据搬入到矢量数据寄存器，由用户自行更新源地址。
- **立即数偏移搬入模式**：从相对源起始地址偏移指定距离的位置搬入数据。本接口不会自动更新源地址。

**图1** 广播搬入矢量数据寄存器

![vload broadcast示意图](../../../figures/vload_broadcast.svg)

## 函数原型

```python
vload_broadcast(tensor, offset = 0, *, mode = None, width = None, post_update: bool = False) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `tensor` | 输入 | 源UB地址，实际读取地址必须按dtype对齐。 |
| `offset` | 输入 | 可选，相对`tensor`起始地址的偏移，单位为元素个数。默认值为`0`。 |
| `mode` | 输入 | 可选，支持`None`、`brc`、`elem2datablock`、`datablock`和`block`，也接受对应的`BrcMode`枚举。`elem2datablock`仅接受16/32位源；`block`接受8/16/32位源并保留源dtype。 |
| `width` | 输入 | 可选，仅全lane广播使用`b8`或`b16`，要求源为32位整数；返回`uint32`寄存器。`block`不接受`width`。 |
| `post_update` | 输入 | 可选，默认`False`，`offset`是源元素偏移。`True`时每次读后推进内部指针；全lane显式`width`模式按所选位宽计步长，其他模式按源dtype计。全lane及整块广播步长保持32B对齐；元素到块广播的16/32位源分别保持16B/32B对齐。 |

### mode枚举与模式选择

`mode`接受下表中的取值或对应的`BrcMode`枚举。

| mode取值 | BrcMode枚举 | 数据排列 | ASC对应能力 |
| --- | --- | --- | --- |
| `None` / `"brc"` | `BrcMode.FULL` | 单个源元素重复到整个矢量数据寄存器 | `asc_loadalign_brc` / `asc_loadalign_brc_elem` |
| `"elem2datablock"` | `BrcMode.ELEM2DATABLOCK` | 读取8个源元素，每个元素分别重复填满一个32字节DataBlock | `asc_loadalign_brc_elem2datablock` |
| `"datablock"` | `BrcMode.DATABLOCK` | 读取8个源元素，每个元素分别重复填满一个32字节DataBlock | `asc_loadalign_brc_elem2datablock` |
| `"block"` | `BrcMode.BLOCK` | 读取一个完整32字节DataBlock，将整个块重复8次 | `asc_loadalign_brc_datablock` |

全lane普通广播的实际地址按源元素对齐；整块广播按32字节对齐；元素到块广播的16/32位源分别按16字节/32字节对齐。上述模式均支持`post_update=True`。

**支持的元素数据类型：**

dtype支持的数据类型为`dtypes.int4x2`、`dtypes.int8`、`dtypes.uint8`、`dtypes.fp4x2_e2m1`、`dtypes.fp4x2_e1m2`、`dtypes.hifloat8`、`dtypes.float8_e8m0`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 返回值说明

返回保存广播搬入结果的矢量数据寄存器值，类型为`RawVReg`。返回元素类型按`mode`区分：

- `None`、`"brc"`或`BrcMode.FULL`：`width=None`时元素类型与`tensor`一致，`width="b8"`或`"b16"`时为`dtypes.uint32`。
- `"elem2datablock"`、`"datablock"`、`BrcMode.ELEM2DATABLOCK`或`BrcMode.DATABLOCK`：16位源元素对应`dtypes.uint16`，32位源元素对应`dtypes.uint32`。
- `"block"`或`BrcMode.BLOCK`：元素类型与`tensor`一致。例如，`dtypes.float32`源返回`dtypes.float32`寄存器，不改为`dtypes.uint32`。

## 约束说明

### 通用约束

- 本接口需在VF作用域内调用。
- 各功能模式下的实际读取地址必须按dtype对齐，且实际读取范围必须在UB地址空间内且不越界，否则会报错。
- UB容量上限为256KB，用户可用容量随编译选项与编程场景变化（默认预留6KB SIMD VF栈 + 2KB框架预留，可用248KB；SIMD+SIMT混编时再划分32KB～128KB作Data Cache，可用容量进一步减少）。UB地址偏移后不可超过实际可用容量，否则会报错。
- 如果本指令与其他指令存在UB地址重叠，需要插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化，防止出现异常数据。

## 调用示例

以下示例演示`vload_broadcast`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload_broadcast, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vload_broadcast_kernel(src0, dst):
    buf = Buffer(MemLoc.UB, src0.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf, src0)

    in0 = buf
    res = out
    with vf(mode="simd"):
        mask = full_mask()
        vstore(res, 0, vload_broadcast(in0, 0), mask)

    mem_copy(dst, out)

@host
def run(src0, dst):
    _vload_broadcast_kernel[1](src0, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, dst)
    torch.npu.synchronize()

    # mode取默认值，将首个元素广播到整个矢量数据寄存器
    torch.testing.assert_close(dst.cpu(), src0.cpu()[0].expand(64))
    print("vload_broadcast example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vload_broadcast example passed
first=0.0000, last=0.0000
```
