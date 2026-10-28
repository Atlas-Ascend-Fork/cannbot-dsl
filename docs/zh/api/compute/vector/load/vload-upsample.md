# vload_upsample
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

从Unified Buffer（UB）中满足对应对齐要求的起始地址读取连续数据并进行2倍上采样，结果通过函数返回值返回。从UB中32字节对齐的起始地址读取VL/2长度数据，将每个源元素重复两次后返回VL长度数据。

本接口提供两种功能模式：

- **对齐搬入模式**：将UB源地址的数据上采样后搬入到矢量数据寄存器或掩码寄存器，由用户自行更新源地址。
- **立即数偏移搬入模式**：从相对源起始地址偏移指定距离的位置搬入数据。本接口不会自动更新源地址。

对齐搬入模式中，通过函数返回值返回掩码寄存器时，请使用[`vmask_load`](vmask-load.md)。

**图1** 上采样搬入掩码寄存器

![vload upsample示意图](../../../figures/vload_upsample.svg)

## 函数原型

```python
vload_upsample(tensor, offset = 0, *, post_update: bool = False) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `tensor` | 输入 | 源UB地址，实际读取地址必须按32字节对齐。 |
| `offset` | 输入 | 相对`tensor`起始地址的偏移，单位为元素个数。默认值为`0`。 |
| `post_update` | 输入 | 是否采用Post Update模式。取值为`True`或`False`，默认值为`False`。取值为`True`时，接口采用Post Update模式，搬入完成后自动更新源地址指针，`offset`为Post Update步长，单位为元素个数，源地址增加`offset × sizeof(dtype)`字节。 |

**支持的元素数据类型：**

dtype支持的数据类型为`dtypes.int4x2`、`dtypes.int8`、`dtypes.uint8`、`dtypes.fp4x2_e2m1`、`dtypes.fp4x2_e1m2`、`dtypes.hifloat8`、`dtypes.float8_e8m0`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`。

## 返回值说明

返回保存上采样搬入结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`tensor`一致。

## 约束说明

### 通用约束

- 本接口需在VF作用域内调用。
- 实际读取地址必须按32字节对齐，实际读取范围必须在UB地址空间内且不越界，否则会报错。
- UB容量上限为256KB，用户可用容量随编译选项与编程场景变化（默认预留6KB SIMD VF栈 + 2KB框架预留，可用248KB；SIMD+SIMT混编时再划分32KB～128KB作Data Cache，可用容量进一步减少）。UB地址偏移后不可超过实际可用容量，否则会报错。
- 如果本指令与其他指令存在UB地址重叠，需要插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化，防止出现异常数据。

## 调用示例

以下示例演示`vload_upsample`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload_upsample, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vload_upsample_kernel(src0, dst):
    buf = Buffer(MemLoc.UB, src0.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((128,)))

    mem_copy(buf, src0)

    in0 = buf
    res = out
    with vf(mode="simd"):
        mask = full_mask(elem_bits=16)
        vstore(res, 0, vload_upsample(in0, 0), mask)

    mem_copy(dst, out)

@host
def run(src0, dst):
    _vload_upsample_kernel[1](src0, dst)

def main():
    src0 = torch.arange(64, dtype=torch.int32).to(torch.float16).to("npu:0")
    dst = torch.empty((128,), dtype=torch.float16, device="npu:0")

    run(src0, dst)
    torch.npu.synchronize()

    assert bool((dst.cpu() == src0.cpu().repeat_interleave(2)).all()), f"upsample mismatch: {dst.cpu()[:4].tolist()}"
    print("vload_upsample example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vload_upsample example passed
first=0.0000, last=63.0000
```
