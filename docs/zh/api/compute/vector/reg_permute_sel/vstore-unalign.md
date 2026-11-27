# vstore_unalign

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将源向量的前 count 个元素从目标视图起点写入 UB，尾部暂存在 ureg 中。默认由调用方通过目标视图推进地址。post_update=True 时首次以视图起点初始化指针，此后每次自动推进 count 个元素，同一序列使用同一视图和模式。对应 asc_storeunalign / asc_storeunalign_postupdate。

## 函数原型

```python
vstore_unalign(tensor, src: RawVReg, ureg: UStoreReg, *, count, post_update: bool = False) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| tensor | 输出 | UB 目标视图，按元素自然对齐，类型与 src 一致。 |
| src | 输入 | 原始源向量。 |
| ureg | 输入/输出 | create_ustore_reg 返回的尾部状态。 |
| count | 输入 | 本次元素数，范围 0 到向量容量，动态范围由调用方保证。 |
| post_update | 输入 | 可选，是否在写入后自动推进目的指针，默认值为`False`。 |

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
