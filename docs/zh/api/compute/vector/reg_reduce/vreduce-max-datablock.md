# vreduce_max_datablock
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据`mask`对每个`DataBlock`（32B）内的元素求最大值，得到各`DataBlock`的归约结果。结果依次保存在返回值的低位。计算公式如下：

$$
dst_k = \max\{src_i \mid kB \le i < (k + 1)B,\ mask_i = 1\}
$$

其中，$k$表示`DataBlock`的索引，$B$为一个`DataBlock`内的元素个数。

## 函数原型

```python
vreduce_max_datablock(src: RawVReg, *, mask: Mask) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 掩码寄存器，用于控制各元素是否参与归约。 |

**支持的元素数据类型：**

`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 返回值说明

返回保存各数据块的最大值归约结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`src`一致。

## 约束说明

- 本接口需在`vf(mode="simd")`作用域内调用。

- 比较时遵循$max(-0, +0) = +0$。
- 返回值与`src`的数据类型需要保持一致。
- 每个`DataBlock`内的最大值连续写入返回值的前8个元素，这8个元素为有效输出，返回值中的其他元素置0。
- 未被`mask`选中的元素被视为对应数据类型的最小值，浮点数类型的最小值为`-inf`。如果一个`DataBlock`中的所有元素均未被`mask`选中，则将该最小值写入返回值的对应位置。
- 仅输出最大值，不输出索引。

## 调用示例

以下示例演示`vreduce_max_datablock`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload, vreduce_max_datablock, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vreduce_max_datablock_kernel(src0, dst):
    buf = Buffer(MemLoc.UB, src0.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))
    mem_copy(buf, src0)

    in0 = buf
    res = out
    with vf(mode="simd"):
        full = full_mask()
        acc = vreduce_max_datablock(vload(in0, 0), mask=full)
        vstore(res, 0, acc, full)

    mem_copy(dst, out)

@host
def run(src0, dst):
    _vreduce_max_datablock_kernel[1](src0, dst)

def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src0)

    run(src0, dst)
    torch.npu.synchronize()

    expected = torch.zeros(64)
    expected[:8] = src0.cpu().view(8, 8).max(dim=1).values
    torch.testing.assert_close(dst.cpu(), expected)
    print(f"first={float(dst.cpu()[0]):.4f}, block7={float(dst.cpu()[7]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
first=7.0000, block7=63.0000
```
