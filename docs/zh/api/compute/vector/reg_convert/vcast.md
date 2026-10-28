# vcast
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将源操作数矢量数据寄存器中的元素转换为`dtype`指定的数据类型，并支持多种舍入模式与饱和模式。

关于舍入模式和饱和/非饱和模式的详细说明，请参见[舍入模式与饱和模式](./roundingmode.md)。

当源操作数与目的操作数类型位宽比为2:1时（例如`dtypes.float32`转换为`dtypes.float16`），写入数据时需要将一个`VL`大小的数据分为两部分，根据位置选择标签选择写入目的操作数索引为偶数的位置或奇数的位置。

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
| `dtype` | 输入 | 目标元素数据类型，使用`dtypes`中的类型描述符。可选值及其对应的源类型在[类型组合表](#约束说明)中逐项列出，不表示任意源类型与目标类型均可组合。 |
| `mask` | 输入 | 指定参与转换的元素，掩码位为1时参与转换，为0时对应结果位置清零。 |
| `rounding` | 输入 | 舍入模式，使用寄存器级 [RoundingMode](./roundingmode.md#舍入模式说明)或对应的字符串写法。默认值为`RoundingMode.NA`；各转换可用的模式子集见[约束说明](#约束说明)。 |
| `saturate` | 输入 | 是否启用饱和模式：`True`表示将超出范围的结果限制在目标类型的可表示范围内，`False`表示采用非饱和模式。各类转换的溢出处理见[饱和模式与非饱和模式说明](./roundingmode.md#饱和模式与非饱和模式说明)，允许启用饱和模式的类型组合见[约束说明](#约束说明)。默认值为`False`。 |
| `reg_layout` | 输入 | 位置选择标签，类型为`RegLayout`，必须为编译期常量，不接受原始整数或字符串。默认值为`RegLayout.UNKNOWN`；可用成员还取决于源、目标位宽比，见[约束说明](#约束说明)。 |

`rounding=RoundingMode.NA`表示不显式指定舍入模式：位宽收窄、等位宽转换以及浮点转整数时默认选择`RN`，可精确表示的位宽扩展转换不进行舍入。整数之间的转换不进行数值舍入。

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

- 源与目标元素位宽相同时，`reg_layout`只能取`RegLayout.UNKNOWN`；位宽比为2:1或1:2时，只能取`UNKNOWN`、`ZERO`或`ONE`；位宽比为4:1或1:4时，可取全部五个枚举值。
- `RO`仅用于`float32`转`float16`。`float32`与`float8_e4m3fn`、`float8_e5m2`互转，以及`bfloat16`转`float8_e8m0`时，只能指定`NA`或`RN`。
- `float16`、`float32`转`hifloat8`时，只能指定`NA`、`RN`或`RNA`；反向转换只能指定`NA`或`RN`。`float16`、`bfloat16`转`float32`，以及`fp4x2_e2m1`、`fp4x2_e1m2`转`bfloat16`时，也只能指定`NA`或`RN`。
- 非全真掩码的元素位宽必须与转换方式匹配：4位源元素扩展转换使用8位掩码；涉及`float8_e4m3fn`、`float8_e5m2`、`hifloat8`的转换，以及显式选择源位置的扩展转换，使用源元素位宽；其余转换使用源与目标元素位宽中的较大值。通过`create_mask(pattern="all", ...)`或`full_mask`创建的全真掩码不受此位宽匹配限制。

## 调用示例

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
