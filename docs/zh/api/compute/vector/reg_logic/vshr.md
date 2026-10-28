# vshr
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据掩码对源操作数矢量数据寄存器中的元素执行右移，右移位数由标量`shift_bits`指定，并将结果作为返回值返回。掩码对应位置为1的元素参与计算，为0的元素在输出结果中置零。计算公式如下：

$$
dst_i = src_i \gg shift\_bits
$$

根据源操作数的数据类型，右移操作分为以下两种情况：

- **无符号整数：执行逻辑右移。** 逻辑右移会将二进制数整体向右移动指定的位数，最低位被丢弃，最高位用0填充。例如，二进制数1010101010101010（`dtypes.uint16`类型）逻辑右移1位后，结果为0101010101010101。
- **有符号整数：执行算术右移。** 算术右移会将二进制数整体向右移动指定的位数，最低位被丢弃，最高位复制符号位。例如，二进制数1010101010101010（`dtypes.int16`类型）算术右移1位后，结果为1101010101010101；算术右移3位后，结果为1111010101010101。

## 函数原型

```python
vshr(src: RawVReg, shift_bits: int, *, mask: Mask) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（矢量数据寄存器）。 |
| `shift_bits` | 输入 | 位移量（标量），不支持设置为负数。<br>`shift_bits`不支持设置为负数，负数行为未定义。 |
| `mask` | 输入 | 源操作数掩码（掩码寄存器），用于指示参与计算的元素。对应位置为1时参与计算，为0时不参与计算且输出结果对应元素置零。 |

**支持的元素数据类型：**

`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.int32`、`dtypes.uint32`。

## 返回值说明

返回保存右移结果的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型与`src`一致。

## 约束说明

### 通用约束

- 本接口需在VF作用域内调用。
- 掩码位为0的元素位置不参与右移运算，输出结果对应位置写0。

### 位移约束

- 对于无符号整数，当`shift_bits`大于`src`的数据类型位宽时，输出结果中的有效元素置零。
- 对于有符号整数，当`shift_bits`大于`src`的数据类型位宽时，`src`中的元素小于0则输出结果对应位置写-1，`src`中的元素大于或等于0则输出结果对应位置写0。

## 调用示例

以下示例将矢量数据寄存器中的元素右移，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vload, vshr, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vshr_kernel(src, dst):
    buf = Buffer(MemLoc.UB, src.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf, src)

    in0 = buf
    res = out
    with vf(mode="simd"):
        mask = full_mask()
        vstore(res, 0, vshr(vload(in0, 0), 2, mask=mask), mask)

    mem_copy(dst, out)

@host
def run(src, dst):
    _vshr_kernel[1](src, dst)

def main():
    src = torch.arange(64, dtype=torch.int32, device="npu:0")
    dst = torch.empty_like(src)

    run(src, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src >> 2).cpu())
    print("vshr example passed")
    print(f"first={int(dst.cpu()[0])}, last={int(dst.cpu()[-1])}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vshr example passed
first=0, last=15
```
