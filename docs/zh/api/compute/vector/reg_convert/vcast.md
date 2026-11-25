# vcast
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将源操作数矢量数据寄存器中的元素转换为`dtype`指定的数据类型，并支持多种舍入模式与饱和模式。

关于舍入模式和饱和/非饱和模式的详细说明，请参见[舍入模式与饱和模式](../../roundingmode.md)。

当源元素位宽大于目标元素位宽时，`reg_layout`用于选择转换结果在目的寄存器中的写入位置。例如，`float32`转`float16`时，设置`reg_layout=RegLayout.ZERO`将结果写入目的偶数索引位置，设置`reg_layout=RegLayout.ONE`将结果写入目的奇数索引位置。`float32`转`float8_e4m3fn`时，`RegLayout.ZERO`、`RegLayout.ONE`、`RegLayout.TWO`、`RegLayout.THREE`分别选择目的索引为`4i`、`4i+1`、`4i+2`、`4i+3`的位置，其中`i`为从0开始的整数。未选中的目的位置清零。

当源元素位宽小于目标元素位宽时，`reg_layout`用于选择参与转换的源元素位置。例如，`float16`转`float32`时，设置`reg_layout=RegLayout.ZERO`读取源偶数索引的元素，设置`reg_layout=RegLayout.ONE`读取源奇数索引的元素。`int8`转`int32`时，`RegLayout.ZERO`、`RegLayout.ONE`、`RegLayout.TWO`、`RegLayout.THREE`分别选择源索引为`4i`、`4i+1`、`4i+2`、`4i+3`的元素，其中`i`为从0开始的整数。各类型转换的位置含义和可用枚举值见下文。

## 函数原型

```python
vcast(
    src: RawVReg,
    dtype,
    *,
    mask: Mask,
    rounding=RoundingMode.NA,
    saturate=False,
    reg_layout=RegLayout.UNKNOWN,
) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 保存待转换元素的矢量数据寄存器，不会被本接口原地修改。支持的源元素类型见[类型组合表](#约束说明)。 |
| `dtype` | 输入 | 目标元素数据类型，例如`dtypes.float32`。支持的源、目标类型组合见[约束说明](#约束说明)。 |
| `mask` | 输入 | 指定参与转换的元素，掩码位为1时参与转换，为0时对应结果位置清零。 |
| `rounding` | 输入 | 舍入模式，使用公共[RoundingMode](../../roundingmode.md#舍入模式说明)或对应的字符串写法。可通过`from cannbotdsl import RoundingMode`或`from cannbotdsl.types import RoundingMode`导入。默认值为`RoundingMode.NA`；各转换可用的模式子集见[约束说明](#约束说明)。 |
| `saturate` | 输入 | 是否启用饱和模式：`True`表示将超出范围的结果限制在目标类型的可表示范围内，`False`表示采用非饱和模式。各类转换的溢出处理见[饱和模式与非饱和模式说明](../../roundingmode.md#饱和模式与非饱和模式说明)，允许启用饱和模式的类型组合见[约束说明](#约束说明)。默认值为`False`。 |
| `reg_layout` | 输入 | 位置选择标签，类型为`RegLayout`，必须为编译期常量，不接受原始整数或字符串。默认值为`RegLayout.UNKNOWN`；枚举成员、位置含义及使用场景见[RegLayout枚举与位置选择](#reglayout枚举与位置选择)，可用成员还受[约束说明](#约束说明)限制。 |

`rounding=RoundingMode.NA`表示不显式指定舍入模式：位宽收窄、等位宽转换以及浮点转整数时默认选择`RN`，可精确表示的位宽扩展转换不进行舍入。整数之间的转换不进行数值舍入。

### RegLayout枚举与位置选择

可通过`from cannbotdsl.ops.reg import RegLayout`导入，也可通过`cannbotdsl.reg.RegLayout`或`cannbotdsl.RegLayout`访问。调用时必须传枚举成员，例如`reg_layout=RegLayout.ONE`，不能传`1`或`"ONE"`。

| 枚举成员 | 位宽比为1:2或2:1时 | 位宽比为1:4或4:1时 |
| --- | --- | --- |
| `RegLayout.UNKNOWN` | 不显式指定位置，使用该转换的默认布局 | 不显式指定位置，使用该转换的默认布局 |
| `RegLayout.ZERO` | 选择偶数索引：0、2、4、…… | 选择第0组位置 |
| `RegLayout.ONE` | 选择奇数索引：1、3、5、…… | 选择第1组位置 |
| `RegLayout.TWO` | 不允许 | 选择第2组位置 |
| `RegLayout.THREE` | 不允许 | 选择第3组位置 |

位宽比始终按“源元素位宽:目标元素位宽”表示。相同位宽只能使用`UNKNOWN`，完整类型组合见[约束说明](#约束说明)。`UNKNOWN`不表示自动读取所有源元素或将结果连续排列。例如`float16`转`float32`的默认布局与`ZERO`一样，只读取每个32位数据槽中的低16位，即普通连续搬入后的偶数索引元素。

四组位置是交错分组，不是四段连续数据。涉及`fp4x2_e2m1`、`fp4x2_e1m2`或`int4x2`的转换时，还需考虑每字节打包两个4位元素的布局。例如，`fp4x2_e2m1`转`bfloat16`时，`ZERO`、`ONE`、`TWO`、`THREE`分别选择8位载体索引为`4i`、`4i+1`、`4i+2`、`4i+3`的位置，每个载体包含两个FP4元素。4位源元素的扩展转换使用8位掩码，不能按普通字节元素直接推断掩码粒度。

位置选择与舍入模式相互独立：`RegLayout.ONE`表示奇数索引位置，不等于`RoundingMode.RO`的向奇数舍入。显式选择源位置的FP16转FP32使用16位源元素粒度的掩码，写回FP32使用32位掩码；默认布局配合`vload_unpack`时，可使用32位掩码。掩码的元素位宽要求见[约束说明](#约束说明)。

## 返回值说明

返回保存转换结果的矢量数据寄存器，类型为`RawVReg`，其中元素的数据类型为`dtype`。结果按`reg_layout`指定的位置组织，未参与转换或未被布局选中的结果位置清零。

## 约束说明

- 本接口必须在`vf(mode="simd")`作用域内调用，不能直接在核函数中调用。
- 源元素类型与`dtype`必须不同。支持的类型组合如下，表中未列出的组合不可使用；“允许饱和的目标类型”列为可设置`saturate=True`的子集，其余组合必须使用`False`。

| 源元素类型 | 可选的`dtype` | 允许饱和的目标类型 |
| --- | --- | --- |
| `dtypes.float32` | `dtypes.float16`、`dtypes.bfloat16`、`dtypes.int16`、`dtypes.int32`、`dtypes.int64`、`dtypes.float8_e4m3fn`、`dtypes.float8_e5m2`、`dtypes.hifloat8` | 本行全部目标类型 |
| `dtypes.float16` | `dtypes.float32`、`dtypes.bfloat16`、`dtypes.int8`、`dtypes.int16`、`dtypes.int32`、`dtypes.uint8`、`dtypes.int4x2`、`dtypes.hifloat8` | `dtypes.int8`、`dtypes.int16`、`dtypes.uint8`、`dtypes.int4x2`、`dtypes.hifloat8` |
| `dtypes.bfloat16` | `dtypes.float32`、`dtypes.float16`、`dtypes.int32`、`dtypes.fp4x2_e2m1`、`dtypes.fp4x2_e1m2`、`dtypes.float8_e8m0` | `dtypes.float16`、`dtypes.int32` |
| `dtypes.int8` | `dtypes.float16`、`dtypes.int16`、`dtypes.int32` | 无 |
| `dtypes.uint8` | `dtypes.float16`、`dtypes.uint16`、`dtypes.uint32` | 无 |
| `dtypes.int16` | `dtypes.float16`、`dtypes.float32`、`dtypes.int32`、`dtypes.uint8`、`dtypes.uint32` | `dtypes.uint8` |
| `dtypes.uint16` | `dtypes.uint8`、`dtypes.uint32` | `dtypes.uint8` |
| `dtypes.int32` | `dtypes.float32`、`dtypes.int16`、`dtypes.int64`、`dtypes.uint8`、`dtypes.uint16` | `dtypes.int16`、`dtypes.uint8`、`dtypes.uint16` |
| `dtypes.uint32` | `dtypes.int16`、`dtypes.uint8`、`dtypes.uint16` | 本行全部目标类型 |
| `dtypes.int64` | `dtypes.float32`、`dtypes.int32` | `dtypes.int32` |
| `dtypes.float8_e4m3fn`、`dtypes.float8_e5m2` | `dtypes.float32` | 无 |
| `dtypes.hifloat8` | `dtypes.float16`、`dtypes.float32` | 无 |
| `dtypes.fp4x2_e2m1`、`dtypes.fp4x2_e1m2` | `dtypes.bfloat16` | 无 |
| `dtypes.int4x2` | `dtypes.float16`、`dtypes.bfloat16`、`dtypes.int16` | 无 |

- 源与目标元素位宽相同时，`reg_layout`只能取`RegLayout.UNKNOWN`；位宽比为2:1或1:2时，只能取`UNKNOWN`、`ZERO`或`ONE`；位宽比为4:1或1:4时，可取全部五个枚举值。各枚举的位置含义见[RegLayout枚举与位置选择](#reglayout枚举与位置选择)。
- 将`dtypes.float16`转换为`dtypes.float32`且`reg_layout`为`RegLayout.UNKNOWN`时，每个32位数据槽仅取低16位进行转换。使用普通`vload`连续搬入时，转换读取下标为0、2、4等的源元素；若需按原顺序转换连续元素，可先使用`vload_unpack`和`UnpackMode.UNPACK2`搬入。若需分别处理奇偶元素，则显式使用`ZERO`和`ONE`各转换一次，源寄存器本身不会被修改。
- `RO`仅用于`float32`转`float16`。`float32`与`float8_e4m3fn`、`float8_e5m2`互转，以及`bfloat16`转`float8_e8m0`时，只能指定`NA`或`RN`。
- `float16`、`float32`转`hifloat8`时，只能指定`NA`、`RN`或`RNA`；反向转换只能指定`NA`或`RN`。`float16`、`bfloat16`转`float32`，以及`fp4x2_e2m1`、`fp4x2_e1m2`转`bfloat16`时，也只能指定`NA`或`RN`。
- 非全真掩码的元素位宽必须与转换方式匹配：4位源元素扩展转换使用8位掩码；涉及`float8_e4m3fn`、`float8_e5m2`、`hifloat8`的转换，以及显式选择源位置的扩展转换，使用源元素位宽；其余转换使用源与目标元素位宽中的较大值。通过`create_mask(pattern="all", ...)`或`full_mask`创建的全真掩码不受此位宽匹配限制。

## 调用示例

### FP32转FP16并连续写回

以下示例将64个`float32`元素转换为`float16`，通过`vstore_pack`连续写回，并在Host侧与PyTorch的转换结果进行校验。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import PackMode, create_mask, vcast, vload, vstore_pack
from cannbotdsl.tensor import MemLoc

@kernel
def _vcast_kernel(src, dst):
    buf = Buffer(MemLoc.UB, src.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf, src)

    in0 = buf
    res = out
    with vf(mode="simd"):
        mask = create_mask(pattern="all", elem_bits=32)
        converted = vcast(vload(in0, 0), dtypes.float16, mask=mask)
        vstore_pack(res, 0, converted, mask, pack_mode=PackMode.PACK_HALF)

    mem_copy(dst, out)

@host
def run(src, dst):
    _vcast_kernel[1](src, dst)

def main():
    src = (torch.arange(64, dtype=torch.float32, device="npu:0") - 32.0) / 8.0
    dst = torch.empty((64,), dtype=torch.float16, device="npu:0")

    run(src, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), src.cpu().half(), rtol=0, atol=0)
    print("vcast example passed")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vcast example passed
```

### FP16转FP32并保持连续顺序

以下示例使用`vload_unpack`解包搬入64个`float16`元素，将其转换为`float32`并按原顺序写回。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import UnpackMode, create_mask, vcast, vload_unpack, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vcast_fp16_to_fp32_kernel(src, dst):
    ub_fp16 = Buffer(MemLoc.UB, dtypes.float16, make_layout((64,)))
    ub_fp32 = Buffer(MemLoc.UB, dtypes.float32, make_layout((64,)))
    mem_copy(ub_fp16, src)

    with vf(mode="simd"):
        mask = create_mask(pattern="all", elem_bits=32)
        unpacked = vload_unpack(ub_fp16, 0, unpack_mode=UnpackMode.UNPACK2)
        converted = vcast(unpacked, dtypes.float32, mask=mask)
        vstore(ub_fp32, 0, converted, mask)

    mem_copy(dst, ub_fp32)

@host
def run(src, dst):
    _vcast_fp16_to_fp32_kernel[1](src, dst)

def main():
    src = torch.arange(64, dtype=torch.float16, device="npu:0")
    dst = torch.empty((64,), dtype=torch.float32, device="npu:0")
    run(src, dst)
    torch.npu.synchronize()
    torch.testing.assert_close(dst.cpu(), src.cpu().float(), rtol=0, atol=0)
    print("vcast FP16 to FP32 example passed")

if __name__ == "__main__":
    main()
```

预期结果：

```text
vcast FP16 to FP32 example passed
```

### FP16转FP32时分别选择奇偶源位置

以下示例普通搬入128个连续FP16元素，用同一个源寄存器分别生成64个偶数索引结果和64个奇数索引结果。两次转换都不修改源寄存器，合起来覆盖全部128个源元素。这里展示分组结果，不自动将两组交错还原为原顺序。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import RegLayout, create_mask, vcast, vload, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _vcast_fp16_parts_kernel(src, dst_even, dst_odd):
    ub_fp16 = Buffer(MemLoc.UB, dtypes.float16, make_layout((128,)))
    ub_even = Buffer(MemLoc.UB, dtypes.float32, make_layout((64,)))
    ub_odd = Buffer(MemLoc.UB, dtypes.float32, make_layout((64,)))
    mem_copy(ub_fp16, src)

    with vf(mode="simd"):
        source_mask = create_mask(pattern="all", elem_bits=16)
        output_mask = create_mask(pattern="all", elem_bits=32)
        value = vload(ub_fp16, 0)
        even = vcast(value, dtypes.float32, mask=source_mask, reg_layout=RegLayout.ZERO)
        odd = vcast(value, dtypes.float32, mask=source_mask, reg_layout=RegLayout.ONE)
        vstore(ub_even, 0, even, output_mask)
        vstore(ub_odd, 0, odd, output_mask)

    mem_copy(dst_even, ub_even)
    mem_copy(dst_odd, ub_odd)

@host
def run(src, dst_even, dst_odd):
    _vcast_fp16_parts_kernel[1](src, dst_even, dst_odd)

def main():
    src = torch.arange(128, dtype=torch.float16, device="npu:0")
    dst_even = torch.empty((64,), dtype=torch.float32, device="npu:0")
    dst_odd = torch.empty_like(dst_even)
    run(src, dst_even, dst_odd)
    torch.npu.synchronize()
    torch.testing.assert_close(dst_even.cpu(), src.cpu()[::2].float(), rtol=0, atol=0)
    torch.testing.assert_close(dst_odd.cpu(), src.cpu()[1::2].float(), rtol=0, atol=0)
    print("vcast RegLayout ZERO/ONE example passed")

if __name__ == "__main__":
    main()
```

预期结果：`dst_even`为0、2、4、……、126，`dst_odd`为1、3、5、……、127。

```text
vcast RegLayout ZERO/ONE example passed
```
