# Reg 数据搬出

Reg 数据搬出接口用于将矢量数据寄存器、掩码寄存器或非对齐寄存器中的数据写入Unified Buffer（UB）。接口在VF作用域内使用，仅在AIV上生效。

## 接口概览

### 矢量数据寄存器搬出

#### 对齐搬出

对齐搬出接口的目的地址对齐要求随接口而异，并非都要求32字节对齐。使用带偏移的接口时，偏移后的实际访问地址也须满足表中的对齐要求。

**表1** 矢量数据寄存器对齐搬出接口

| 接口名称 | 模式 | 功能简述 | 目的地址对齐要求 |
| --- | --- | --- | --- |
| [`vstore`](vstore.md) | 连续对齐搬出 | 按`mask`将一个VL（256字节）矢量数据寄存器的数据连续写入UB。 | 32字节对齐。 |
| [`vstore`](vstore.md) | 立即数偏移搬出 | 按`mask`将一个VL的数据写入`tensor + offset`，`offset`单位为元素。 | 32字节对齐。 |
| [`vstore`](vstore.md)（`post_update=True`） | 立即数偏移Post Update搬出 | 按`mask`连续搬出一个VL，并按元素偏移自动更新目的地址。 | 32字节对齐。 |
| [`vstore_strided`](vstore-strided.md) | 非连续对齐搬出 | 按`block_stride`和`repeat_stride`配置8个DataBlock的写入位置。 | 32字节对齐。 |
| [`vstore_first`](vstore-first.md) | 首元素搬出 | 将矢量数据寄存器的首个元素写入UB。 | `dtype`对齐。 |
| [`vstore_first`](vstore-first.md) | 首元素立即数偏移搬出 | 将首个元素写入`tensor + offset`，`offset`单位为元素。 | `dtype`对齐。 |
| [`vstore_first`](vstore-first.md)（`post_update=True`） | 首元素Post Update搬出 | 搬出首个元素后，按元素偏移自动更新目的地址。 | `dtype`对齐。 |
| [`vstore_interleave`](vstore-interleave.md) | 交织连续搬出 | 将两个矢量数据寄存器按元素交织后搬出，单次搬出量为`2 x VL`（512字节）。 | 32字节对齐。 |
| [`vstore_interleave`](vstore-interleave.md) | 交织立即数偏移搬出 | 将两个寄存器交织后写入`tensor + offset`，`offset`单位为元素。 | 32字节对齐。 |
| [`vstore_pack`](vstore-pack.md)（`PackMode.PACK_HALF`） | 压缩连续搬出 | 按`mask`将有效元素的低半部分bit压缩后写入UB。 | 32字节对齐。 |
| [`vstore_pack`](vstore-pack.md)（`PackMode.PACK_HALF`） | 压缩立即数偏移搬出 | 将压缩结果写入`tensor + offset`，`offset`单位为元素。 | 32字节对齐。 |
| [`vstore_pack`](vstore-pack.md)（`PackMode.PACK_HALF`、`post_update=True`） | 压缩Post Update搬出 | 压缩搬出后，按元素偏移自动更新目的地址。 | 32字节对齐。 |
| [`vstore_pack`](vstore-pack.md)（`PackMode.PACK_QUARTER`） | Quarter Pack连续搬出 | 按`mask`将有效32位元素的低8bit压缩后写入UB。 | 32字节对齐。 |
| [`vstore_pack`](vstore-pack.md)（`PackMode.PACK_QUARTER`） | Quarter Pack立即数偏移搬出 | 将Quarter Pack压缩结果写入`tensor + offset`，`offset`单位为元素。 | 32字节对齐。 |
| [`vstore_pack`](vstore-pack.md)（`PackMode.PACK_QUARTER`、`post_update=True`） | Quarter Pack Post Update搬出 | Quarter Pack压缩搬出后，按元素偏移自动更新目的地址。 | 32字节对齐。 |

#### 非对齐搬出

非对齐搬出接口通过非对齐寄存器暂存尾块，通过后处理接口搬出暂存的尾块。连续调用时必须复用同一个非对齐寄存器；后处理接口也必须与前序主搬出接口复用该寄存器。

**表2** 矢量数据寄存器非对齐搬出接口

| 接口名称 | 模式 | 功能简述 | 目的地址对齐要求 | 配套使用的后处理接口 |
| --- | --- | --- | --- | --- |
| [`vstore_unalign`](../reg_permute_sel/vstore-unalign.md) | 连续搬出 | 将搬出数据的主块写入UB，尾块暂存至非对齐寄存器。目的地址由用户手动更新。 | `dtype`对齐，无需32字节对齐。 | [`vstore_unalign_post`](vstore-unalign-post.md) |

### 掩码寄存器搬出

#### 对齐搬出

掩码寄存器对齐搬出不支持配置额外掩码。直接搬出时写出32字节掩码数据；压缩搬出时每间隔1bit丢弃1bit，并将保留bit连续写入UB。

**表3** 掩码寄存器对齐搬出接口

| 接口名称 | 模式 | 功能简述 | 目的地址对齐要求 |
| --- | --- | --- | --- |
| [`vmask_store`](vmask-store.md)（`dist='norm'`） | 连续对齐搬出 | 直接将32字节掩码寄存器数据写入UB。 | 32字节对齐。 |
| [`vmask_store`](vmask-store.md)（`dist='pack'`） | 压缩连续搬出 | 每间隔1bit丢弃1bit，将保留bit压缩后写入UB。 | 16字节对齐。 |

## 关键特性

### 对齐连续搬出方式对比

以下提供三种地址偏移方式，每种方式均以两个长度为1024、数据类型为`dtypes.float16`的输入进行逐元素相加，并将结果连续写入UB的计算过程为例。单次迭代处理128个元素，共迭代8次，最终结果均为`dst[i] = src0[i] + src1[i]`。

**表4** 对齐连续搬出方式对比

| 场景 | 搬出方式 | 目的地址维护方式 | 适用场景 |
| --- | --- | --- | --- |
| 场景1 | 用户手动偏移地址搬出 | 调用方自行计算每次搬出的目的地址。 | 目的地址偏移规则简单，且需要由用户显式控制地址。 |
| 场景2 | 通过接口立即数偏移搬出 | 将相对目的基地址的固定元素偏移作为`offset`参数传入。 | 偏移量在编译期确定，且无需修改目的地址指针。 |
| 场景3 | Post Update搬出 | `post_update=True`时接口在每次搬出后自动更新目的地址，`offset`作为地址偏移量。 | 连续搬出且无需用户手动维护目的地址。 |

## 调用示例

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vadd, vload, vstore
from cannbotdsl.tensor import MemLoc

N = 1024
VL = 128  # 单次搬出128个元素

@kernel
def _store_offset_kernel(src0, src1, dst_imm, dst_post):
    buf0 = Buffer(MemLoc.UB, src0.dtype, make_layout((N,)))
    buf1 = Buffer(MemLoc.UB, src1.dtype, make_layout((N,)))
    out0 = Buffer(MemLoc.UB, dst_imm.dtype, make_layout((N,)))
    out1 = Buffer(MemLoc.UB, dst_post.dtype, make_layout((N,)))

    mem_copy(buf0, src0)
    mem_copy(buf1, src1)

    a = buf0
    b = buf1
    r0 = out0
    r1 = out1
    with vf(mode="simd"):
        mask = full_mask(elem_bits=16)
        for i in range(N // VL):
            acc = vadd(vload(a, i * VL), vload(b, i * VL), mask=mask)
            # 立即数偏移搬出：offset 相对目的基地址，单位为元素
            vstore(r0, i * VL, acc, mask)
            # Post Update搬出：offset 为游标步长，目的地址由硬件自动推进
            vstore(r1, VL, acc, mask, post_update=True)

    mem_copy(dst_imm, out0)
    mem_copy(dst_post, out1)

@host
def run(src0, src1, dst_imm, dst_post):
    _store_offset_kernel[1](src0, src1, dst_imm, dst_post)

def main():
    src0 = torch.arange(N, dtype=torch.float16, device="npu:0")
    src1 = torch.full((N,), 1000.0, dtype=torch.float16, device="npu:0")
    dst_imm = torch.full((N,), -1.0, dtype=torch.float16, device="npu:0")
    dst_post = torch.full((N,), -1.0, dtype=torch.float16, device="npu:0")

    run(src0, src1, dst_imm, dst_post)
    torch.npu.synchronize()

    expected = (src0.cpu() + src1.cpu()).half()
    assert bool((dst_imm.cpu() == expected).all()), "立即数偏移搬出结果不符"
    assert bool((dst_post.cpu() == expected).all()), "Post Update搬出结果不符"
    print("store offset example passed")
    print(f"imm_last={float(dst_imm.cpu()[-1]):.1f}, post_last={float(dst_post.cpu()[-1]):.1f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
store offset example passed
imm_last=2023.0, post_last=2023.0
```
