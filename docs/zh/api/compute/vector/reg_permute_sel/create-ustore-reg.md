# create_ustore_reg

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

声明独立的非对齐写入尾部寄存器，设备代码中对应 vector_store_unalign。首次写入前无需赋初值。

## 函数原型

```python
create_ustore_reg() -> UStoreReg
```

## 参数说明

无参数。

## 返回值说明

返回 UStoreReg，供普通非对齐写入和刷新共用。

## 约束说明

- 在 SIMD VF 作用域内使用。
- 连续写入和刷新复用同一尾部状态。访问范围须在可用 UB 内。

## 调用示例

以下示例可直接执行，检查有效输出与未写入区域。

```python
import torch
import torch_npu
from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg
from cannbotdsl.tensor import MemLoc

@kernel
def copy_kernel(src, dst):
    ub_src = Buffer(MemLoc.UB, src.dtype, make_layout((64,)))
    ub_dst = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))
    mem_copy(ub_src, src)
    mem_copy(ub_dst, dst)
    with vf(mode="simd"):
        tail = reg.create_ustore_reg()
        value = reg.vload(ub_src, 0)
        reg.vstore_unalign(ub_dst, value, tail, count=13)
        reg.vstore_unalign_post(ub_dst, 13, tail)
    mem_copy(dst, ub_dst)

@host
def run(src, dst):
    copy_kernel[1](src, dst)

src = torch.arange(64, dtype=torch.float32, device="npu:0")
dst = torch.full_like(src, -1)
run(src, dst)
torch.npu.synchronize()
torch.testing.assert_close(dst.cpu()[:13], src.cpu()[:13], rtol=0, atol=0)
torch.testing.assert_close(dst.cpu()[13:], torch.full((51,), -1.0), rtol=0, atol=0)
print("ordinary unaligned store example passed")
```

预期输出 `ordinary unaligned store example passed`。
