# simt.cast

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将标量`value`转换为`dtype`指定的数据类型。支持的类型组合、舍入模式和饱和选项见[约束说明](#约束说明)。

## 函数原型

```python
simt.cast(value, *, dtype, rounding=RoundingMode.RN, saturate=False) -> 目标类型标量
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | 带明确数据类型的标量；本页源类型范围为`int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`int64`、`uint64`、`float16`、`bfloat16`、`float32`或`bool`。 |
| dtype | 输入 | 目标标量数据类型；可选`int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`int64`、`uint64`、`float16`、`bfloat16`、`float32`、`bool`。必须与源类型不同，合法组合见约束说明。 |
| rounding | 输入 | [RoundingMode](./roundingmode.md)枚举或名称字符串，默认`RoundingMode.RN`。各类型组合的可选值见[约束说明](#约束说明)。 |
| saturate | 输入 | 是否启用饱和转换，类型为`bool`，默认`False`。各类型组合的支持情况和`CTRL[60]`设置要求见[约束说明](#约束说明)。 |

## 返回值说明

返回dtype的新值，不修改输入。

## 约束说明

源和目标均须为标量，且数据类型不同。下表列出支持的转换组合。表中的舍入模式均为[RoundingMode](./roundingmode.md)成员。

| 源类型 | 目标类型 | `rounding` | `saturate` |
| --- | --- | --- | --- |
| `float32` | `float16` | `RN`、`RNA`、`RD`、`RU`、`RZ`、`RO`。 | `False`或`True`；启用时须满足下文`CTRL[60]`设置要求。 |
| `float32` | `bfloat16` | `RN`、`RNA`、`RD`、`RU`、`RZ`。 | `False`或`True`；启用时须满足下文`CTRL[60]`设置要求。 |
| `float16`、`bfloat16` | `float32` | 不生效。 | 不生效。 |
| `float16` | `bfloat16` | `RN`、`RNA`、`RD`、`RU`、`RZ`。 | 仅`False`。 |
| `bfloat16` | `float16` | `RN`、`RNA`、`RD`、`RU`、`RZ`。 | `False`或`True`；启用时须满足下文`CTRL[60]`设置要求。 |
| `int16`、`uint16` | `float16`、`bfloat16` | `RN`、`RD`、`RU`、`RZ`。 | 仅`False`。 |
| `int32`、`uint32` | `float16` | `RN`、`RNA`、`RD`、`RU`、`RZ`。 | `False`或`True`；启用时须满足下文`CTRL[60]`设置要求。 |
| `int32`、`uint32` | `bfloat16` | `RN`、`RNA`、`RD`、`RU`、`RZ`。 | 仅`False`。 |
| `int64`、`uint64` | `float16`、`bfloat16` | `RN`、`RNA`、`RD`、`RU`、`RZ`。 | 仅`False`。 |
| `float16`、`bfloat16` | `int8`、`uint8` | 固定向零舍入，`rounding`参数不生效。 | 不生效。 |
| `float16`、`bfloat16` | `int16`、`uint16` | `RN`、`RD`、`RU`、`RZ`。 | 不生效。 |
| `float16`、`bfloat16` | `int32`、`uint32`、`int64`、`uint64` | `RN`、`RNA`、`RD`、`RU`、`RZ`。 | 不生效。 |
| `int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`int64`、`uint64`、`float32`、`bool` | 同左，且与源类型不同 | 不生效，按普通标量数值转换处理；浮点转整数时，输入须为有限值且结果在目标范围内。 | 不生效。 |

- 不支持表外类型组合。`int8`、`uint8`转换为`float16`或`bfloat16`时，须先转换为`int32`。
- 表中标明支持`saturate=True`的转换受`CTRL[60]`控制：SIMD与SIMT混合执行时，须在进入SIMT函数前，在SIMD侧将`CTRL[60]`置为`0`；纯SIMT Kernel中不能通过该参数启用这些转换的饱和模式。寄存器设置接口见[set_ctrl](../../execution/device/spr/set-ctrl.md)。
- 此处的饱和转换将超范围结果限制为目标类型的有限值，不是将结果限制到`[0, 1]`。

## 调用示例

将输入转换为目标数据类型，并打印结果。

```python
import torch
import torch_npu
from cannbotdsl import simt, dtypes, dim3
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def demo(src: Tensor, dst: Tensor):
    with vf(mode='simt'):
        x = simt.cast(src[0], dtype=dtypes.float16)
        dst[0] = simt.cast(x, dtype=dtypes.float32)


@host
def run(src, dst):
    demo[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([1.5, 2.5, 3.5, 4.5], device='npu', dtype=torch.float32)
    dst = torch.zeros(2, device='npu', dtype=torch.float32)
    run(src, dst)
    torch.npu.synchronize()
    assert dst.cpu().tolist() == [1.5, 0.0]
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[1.5, 0.0]
```
