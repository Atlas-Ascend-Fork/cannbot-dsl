# mask_counter
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据元素个数`logical`创建计数器，用于记录剩余待处理的元素个数。将该计数器传入`update_mask`，生成对应的掩码，并自动将计数器减去当前向量处理单元的元素个数。支持b8、b16、b32三种位宽模式，由于VL=256B，各模式的向量处理单元元素个数为：

- b8模式：每次处理256个元素，用于8 bit数据类型的矢量计算。
- b16模式：每次处理128个元素，用于16 bit数据类型的矢量计算。
- b32模式：每次处理64个元素，用于32 bit数据类型的矢量计算。

## 函数原型

```python
mask_counter(logical) -> MaskCounter
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `logical` | 输入 | 逻辑元素总数，类型为`Int32`。 |

## 返回值说明

返回计数器，类型为`MaskCounter`。

## 约束说明

- 本接口需在VF作用域内调用。

## 调用示例

以下示例演示`mask_counter`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, mask_counter, update_mask, vdups, vload, vselect, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _mask_counter_kernel(src0, dst):
    buf = Buffer(MemLoc.UB, src0.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))
    mem_copy(buf, src0)

    in0 = buf
    res = out
    with vf(mode="simd"):
        counter = mask_counter(20)
        m, _ = update_mask(counter, elem_bits=32)
        zero = vdups(0.0, dtypes.float32)
        acc = vselect(vload(in0, 0), zero, cond_mask=m)
        vstore(res, 0, acc, full_mask())

    mem_copy(dst, out)

@host
def run(src0, dst):
    _mask_counter_kernel[1](src0, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, dst)
    torch.npu.synchronize()

    expected = torch.cat([src0.cpu()[:20], torch.zeros(44)])
    torch.testing.assert_close(dst.cpu(), expected)
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
first=0.0000, last=0.0000
```
