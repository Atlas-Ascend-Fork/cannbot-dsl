# vload_unalign
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

从Unified Buffer（UB）中按dtype对齐的起始地址读取VL长度连续数据，并搬入矢量数据寄存器。本接口将`vload_unalign_init`中的非对齐寄存器（32字节）中缓存的前置数据与从UB读取的后续数据拼接，得到VL长度数据并搬入矢量数据寄存器。连续搬入时，需要在每次调用前手动更新源地址。

设本次实际读取的起始字节地址为`src_start_addr`，结束字节地址为`src_end_addr`，其中`src_end_addr = src_start_addr + VL`；将`src_start_addr`向低地址方向对齐到32字节边界，得到`aligned_src_start_addr`。`vload_unalign_init`将字节地址范围`[aligned_src_start_addr, aligned_src_start_addr + 32)`的数据缓存到非对齐寄存器，本接口将该缓存与从UB读取的后续数据拼接，得到字节地址范围`[src_start_addr, src_end_addr)`的数据。

## 函数原型

```python
vload_unalign(tensor, offset = 0) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `tensor` | 输入 | 本次搬入的有效起始UB地址，必须按dtype对齐。本接口不会更新该地址。<br>`tensor`起始地址必须按dtype对齐，且实际访问范围必须在UB地址空间内，否则会报错。 |
| `offset` | 输入 | 相对`tensor`起始地址的偏移，单位为元素个数。默认值为`0`。 |

**支持的元素数据类型：**

dtype支持的数据类型为`dtypes.int4x2`、`dtypes.int8`、`dtypes.uint8`、`dtypes.fp4x2_e2m1`、`dtypes.fp4x2_e1m2`、`dtypes.hifloat8`、`dtypes.float8_e8m0`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`、`dtypes.int64`。

## 返回值说明

返回保存非对齐搬入结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`tensor`一致。

## 约束说明

- 本接口需在VF作用域内调用。
- UB容量上限为256KB，用户可用容量随编译选项与编程场景变化（默认预留6KB SIMD VF栈 + 2KB框架预留，可用248KB；SIMD+SIMT混编时再划分32KB～128KB作Data Cache，可用容量进一步减少）。UB地址偏移后不可超过实际可用容量，否则会报错。
- 如果本指令与其他指令存在UB地址重叠，必须插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化，防止出现异常数据。
- 每次搬入前必须调用`vload_unalign_init`中的源地址预处理模式进行初始化，且初始化接口和本接口必须使用相同的`tensor`。

## 调用示例

以下示例演示`vload_unalign`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload_unalign, vload_unalign_init, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vload_unalign_kernel(src0, dst):
    buf = Buffer(MemLoc.UB, src0.dtype, make_layout((65,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf, src0)

    in0 = buf
    res = out
    with vf(mode="simd"):
        mask = full_mask()
        vload_unalign_init(in0, 1)
        vstore(res, 0, vload_unalign(in0, 1), mask)

    mem_copy(dst, out)

@host
def run(src0, dst):
    _vload_unalign_kernel[1](src0, dst)

def main():
    src0 = torch.arange(65, dtype=torch.float32, device="npu:0")
    dst = torch.empty((64,), dtype=torch.float32, device="npu:0")

    run(src0, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), src0.cpu()[1:])
    print("vload_unalign example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vload_unalign example passed
first=1.0000, last=64.0000
```
