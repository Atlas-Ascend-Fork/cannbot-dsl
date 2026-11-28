# vmask_load
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

从Unified Buffer（UB）搬入掩码，普通、上采样、下采样模式分别读取32B、16B、64B，返回32B掩码寄存器。支持手动元素偏移和Post Update地址自更新。

本接口返回32字节掩码，源必须是32位整数存储。`vload`返回256字节矢量数据，两者搬运量不同。

**图1** 连续搬入掩码寄存器

![vmask load示意图](../../../figures/vmask_load.svg)

## 函数原型

```python
vmask_load(tensor, offset = 0, *, elem_bits: int = 32, dist = 'norm', post_update: bool = False) -> Mask
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `tensor` | 输入 | 32位有符号或无符号整数UB源。实际读取量和地址对齐要求由`dist`决定。 |
| `offset` | 输入 | 可选，单位32位源元素。普通模式为本次读取偏移；Post Update模式为读后步长。 |
| `post_update` | 输入 | 默认`False`；`True`开启掩码搬入地址自更新。 |
| `elem_bits` | 输入 | 可选，掩码元素位宽，支持`8`、`16`、`32`，对应b8、b16、b32模式，默认值为`32`。 |
| `dist` | 输入 | 可选，搬入模式，取值为`'norm'`、`'upsample'`或`'downsample'`，默认值为`'norm'`。取值为`'norm'`时搬入VL/8长度数据；取值为`'upsample'`时读取VL/16长度数据；取值为`'downsample'`时读取VL/4长度数据。 |

| dist | 实际读取量 | 地址对齐 |
| --- | --- | --- |
| `norm` | 32B | 32B |
| `upsample` | 16B，逐bit重复两次 | 16B |
| `downsample` | 64B，保留偶数bit | 32B |

`post_update=True`时首次从`tensor`指定的源UB地址开始搬入，每次搬入后，源地址指针增加`offset`个32位源元素。ASC掩码重载的步长单位是字节，框架自动乘4；调用方不要再次换算。静态步长须为正数、转换后的字节步长可用int32表示且保持所选模式的对齐要求。

## 返回值说明

返回掩码寄存器值，类型为`Mask`。

## 约束说明

- 本接口需在VF作用域内调用。
- 如果本指令与其他指令存在UB地址重叠，需要插入同步指令[`vmem_bar`](../reg_sync/vmem-bar.md)，保证多个指令串行化。

## 调用示例

以下示例演示`vmask_load`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import (
    create_mask,
    full_mask,
    vdups,
    vload,
    vmask_load,
    vmask_store,
    vmem_bar,
    vselect,
    vstore,
)
from cannbotdsl.tensor import MemLoc

@kernel
def _vmask_load_kernel(src0, dst):
    buf = Buffer(MemLoc.UB, src0.dtype, make_layout((64,)))
    tmp = Buffer(MemLoc.UB, dtypes.int32, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf, src0)

    in0 = buf
    saved = tmp
    reloaded = tmp
    res = out
    with vf(mode="simd"):
        mask = full_mask()
        saved_mask = create_mask(pattern="vl32", elem_bits=32)
        vmask_store(saved, 0, saved_mask)
        vmem_bar()
        kept = vmask_load(reloaded, 0)
        vstore(res, 0, vselect(vload(in0, 0),
                                            vdups(0.0, dtypes.float32),
                                            cond_mask=kept), mask)

    mem_copy(dst, out)

@host
def run(src0, dst):
    _vmask_load_kernel[1](src0, dst)

def main():
    src0 = torch.arange(1, 65, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)
    expected = src0.cpu().clone()
    expected[32:] = 0.0

    run(src0, dst)
    torch.npu.synchronize()

    assert bool((dst.cpu() == expected).all()), f"mask mismatch: {dst.cpu()[:4].tolist()}"
    print("vmask_load example passed")
    print(f"kept={int((dst.cpu() != 0).sum())}, last_kept={float(dst.cpu()[31]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vmask_load example passed
kept=32, last_kept=32.0000
```
