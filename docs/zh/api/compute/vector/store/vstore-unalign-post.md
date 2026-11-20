# vstore_unalign_post

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将普通非对齐写入的尾部刷新到 UB。默认 tensor 的起点加 offset 应等于写入结束地址。post_update=True 时 offset 相对于同一状态中的当前指针，通常传 0。对应 asc_storeunalign_post / asc_storeunalign_post_postupdate。

## 函数原型

```python
vstore_unalign_post(tensor, offset, ureg: UStoreReg, *, post_update: bool = False) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| tensor | 输出 | 配套写入的 UB 目标。自动推进模式使用相同视图。 |
| offset | 输入 | 刷新地址的元素偏移。自动推进模式下须在 int32 范围内。 |
| ureg | 输入 | 配套写入使用的 UStoreReg。 |
| post_update | 输入 | 与配套写入模式一致，默认 False。 |

## 返回值说明

无返回值。

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
