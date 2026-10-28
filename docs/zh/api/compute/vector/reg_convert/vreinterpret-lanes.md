# vreinterpret_lanes
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将寄存器的256字节载荷按不同lane宽度重新解释（如b32 → 256个b8 lane），纯编译期重标注，不改变任何比特。

## 函数原型

```python
vreinterpret_lanes(src: RawVReg, dtype) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `dtype` | 输入 | 指定返回寄存器值的元素数据类型，允许改变元素位宽和对应的元素划分。 |

## 返回值说明

返回按`dtype`指定的元素数据类型重新划分后的寄存器值，类型为`RawVReg`。总载荷保持256字节，原始比特不变，元素个数随目标元素位宽变化。

## 约束说明

- 载荷总字节数保持256B，仅lane划分变化。
- 本接口需在VF作用域内调用。
- 本接口为纯向量计算；`mem_copy`、同步等非纯操作不得置于同一`vf`作用域内。

## 调用示例

以下示例演示`vreinterpret_lanes`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import create_mask, vload, vreinterpret_lanes, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vreinterpret_lanes_kernel(src, dst):
    buf = Buffer(MemLoc.UB, src.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((256,)))

    mem_copy(buf, src)

    in0 = buf
    res = out
    with vf(mode="simd"):
        mask = create_mask(pattern="all", elem_bits=8)
        r = vreinterpret_lanes(vload(in0, 0), dtypes.uint8)
        vstore(res, 0, r, mask)

    mem_copy(dst, out)

@host
def run(src, dst):
    _vreinterpret_lanes_kernel[1](src, dst)

def main():
    src = torch.arange(1, 65, dtype=torch.int32, device="npu:0")
    dst = torch.empty(256, dtype=torch.uint8, device="npu:0")

    run(src, dst)
    torch.npu.synchronize()

    bytes_out = dst.cpu()
    torch.testing.assert_close(bytes_out, src.cpu().view(torch.uint8))
    print("vreinterpret_lanes example passed")
    print(f"first={int(bytes_out[0])}, last={int(bytes_out[-1])}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vreinterpret_lanes example passed
first=1, last=0
```
