# vstore_interleave
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将两个矢量数据寄存器中的数据按元素顺序交织后连续对齐搬出到Unified Buffer（UB）。单次搬出量为`2 × VL`（共512字节），不支持配置掩码。搬运过程中数据格式与内容保持不变。

通过`offset`指定相对目的起始地址的元素偏移，`offset=0`时从起始地址搬出。多次调用时由用户自行更新偏移或目的张量视图。

## 函数原型

```python
vstore_interleave(tensor, offset, src0: RawVReg, src1: RawVReg, *, width = None) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `tensor` | 输出 | 目的操作数（矢量）的起始地址。起始地址需32字节对齐。 |
| `offset` | 输入 | 相对`tensor`起点的访问偏移，单位为`tensor`的元素个数，必选。可传入整数常量或整型DSL标量，不能传入地址寄存器。<br>偏移后的实际访问地址需32字节对齐，且整个搬出范围须落在UB有效范围内。 |
| `src0` | 输入 | 源操作数0（矢量数据寄存器）。`dtype`须与`tensor`一致。 |
| `src1` | 输入 | 源操作数1（矢量数据寄存器）。`dtype`须与`tensor`一致。 |
| `width` | 输入 | 交织的数据位宽，取值为`'b8'`、`'b16'`或`'b32'`，须与`src0`的数据类型位宽一致，默认值为`None`。取值为`None`时按`src0`的数据类型位宽交织。 |

**支持的元素数据类型：**

`dtype`支持的数据类型为`dtypes.int4x2`、`dtypes.int8`、`dtypes.uint8`、`dtypes.fp4x2_e1m2`、`dtypes.fp4x2_e2m1`、`dtypes.hifloat8`、`dtypes.float8_e4m3fn`、`dtypes.float8_e5m2`、`dtypes.float8_e8m0`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`。

## 返回值说明

无返回值。结果写入`tensor`。

## 约束说明

### 通用约束

- 本接口需在VF作用域内调用。
- UB容量上限为256KB，用户可用容量随编译选项与编程场景变化（默认预留6KB SIMD VF栈+2KB框架预留，可用248KB；SIMD+SIMT混编时再划分32KB~128KB作Data Cache，可用容量进一步减少）。目的操作数地址偏移后不可超过实际可用容量，否则会报错。
- 如果本指令与其他指令存在数据依赖，需要插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化，防止出现异常数据。

## 调用示例

以下示例演示`vstore_interleave`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import vload, vstore_interleave
from cannbotdsl.tensor import MemLoc

@kernel
def _vstore_interleave_kernel(src0, src1, dst):
    buf0 = Buffer(MemLoc.UB, src0.dtype, make_layout((64,)))
    buf1 = Buffer(MemLoc.UB, src1.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((128,)))

    mem_copy(buf0, src0)
    mem_copy(buf1, src1)

    in0 = buf0
    in1 = buf1
    res = out
    with vf(mode="simd"):
        vstore_interleave(res, 0, vload(in0, 0), vload(in1, 0))

    mem_copy(dst, out)

@host
def run(src0, src1, dst):
    _vstore_interleave_kernel[1](src0, src1, dst)

def main():
    src0 = torch.arange(64, dtype=torch.int32).to(torch.float16).to("npu:0")
    src1 = src0 + 100.0
    dst = torch.empty((128,), dtype=torch.float16, device="npu:0")
    expected = torch.zeros(128, dtype=torch.float16)
    expected[0::2] = src0.cpu()
    expected[1::2] = src1.cpu()

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), expected)
    print("vstore_interleave example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vstore_interleave example passed
first=0.0000, last=163.0000
```
