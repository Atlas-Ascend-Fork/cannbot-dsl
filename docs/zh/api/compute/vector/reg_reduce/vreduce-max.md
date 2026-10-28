# vreduce_max
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据`mask`对源操作数`src`进行归约最大值操作，得到归约结果。结果保存在返回值的第0个元素，最大值在`src`中的索引原始位模式保存在返回值的第1个元素，返回值中的其他元素置0。如果存在多个最大值，则保留最小的索引。计算公式如下：

$$
\begin{aligned}
dst_0 &= \max\{src_i \mid mask_i = 1\} \\
dst_1 &= \operatorname{argmax}\{src_i \mid mask_i = 1\}
\end{aligned}
$$

### 关键特性

**规约产生值+索引两个结果，索引值需要强制类型转换**：

返回值中第1个元素保存的是索引的原始位模式，其存储的数据类型与`src`一致：若`src`为`dtypes.float16`，该元素按16位位模式存储索引；若`src`为`dtypes.float32`，该元素按32位位模式存储索引。读取索引时需要按对应位宽的无符号整数解释该位模式，不能按浮点数直接读取。

## 函数原型

```python
vreduce_max(src: RawVReg, *, mask: Mask) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `mask` | 输入 | 掩码寄存器，用于控制各元素是否参与归约。 |

**支持的元素数据类型：**

`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 返回值说明

返回保存最大值归约结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`src`一致。

## 约束说明

- 本接口需在`vf(mode="simd")`作用域内调用。

- 未被`mask`选中的元素被视为对应数据类型的最小值，浮点数类型的最小值为`-inf`。如果`src`中的所有元素均未被`mask`选中，则将该最小值写入返回值的第0个元素，并将其余元素置0。
- 比较时遵循$max(-0, +0) = +0$。
- 如果输入数据中存在nan，则将nan写入返回值的第0个元素，并将第一个nan的索引写入返回值的第1个元素。

## 调用示例

以下示例演示`vreduce_max`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload, vreduce_max, vstore_first
from cannbotdsl.tensor import MemLoc

@kernel
def _vreduce_max_kernel(src0, dst):
    buf = Buffer(MemLoc.UB, src0.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((1,)))
    mem_copy(buf, src0)

    in0 = buf
    res = out
    with vf(mode="simd"):
        acc = vreduce_max(vload(in0, 0), mask=full_mask())
        vstore_first(res, 0, acc)

    mem_copy(dst, out)

@host
def run(src0, dst):
    _vreduce_max_kernel[1](src0, dst)

def main():
    src0 = torch.arange(1, 65, dtype=torch.float32, device="npu:0")
    dst = torch.empty((1,), dtype=torch.float32, device="npu:0")

    run(src0, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), src0.max().cpu().reshape(1))
    print(f"value={float(dst.cpu()[0]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
value=64.0000
```
