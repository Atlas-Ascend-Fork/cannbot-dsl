# simt.cast

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将标量`value`转换为`dtype`指定的数据类型。舍入与饱和效果取决于具体类型和参数组合。

## 函数原型

```python
simt.cast(value, *, dtype, rounding=RoundingMode.RN, saturate=False) -> 目标类型标量
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | 带明确数据类型的标量；本页源类型范围为`int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`int64`、`uint64`、`float16`、`bfloat16`、`float32`或`bool`。 |
| dtype | 输入 | 目标标量类型描述符；可选`int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`int64`、`uint64`、`float16`、`bfloat16`、`float32`、`bool`。必须与源类型不同，合法组合见约束说明。 |
| rounding | 输入 | [RoundingMode](./roundingmode.md) 枚举或名称字符串，默认`RoundingMode.RN`；是否生效取决于源、目标类型。 |
| saturate | 输入 | Python bool，默认False，是否请求对应转换的饱和形式。 |

## 返回值说明

返回dtype的新值，不修改输入。

## 约束说明

源和目标必须都是标量，且数据类型不同。下表中的整数类型指`int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`int64`、`uint64`。

| 源类型 | 目标类型 | 额外限制 |
| --- | --- | --- |
| `float32` | `float16`、`bfloat16` | 舍入与饱和选项取决于目标类型；不要将所有舍入枚举理解为每种组合均可用。 |
| `float16`、`bfloat16` | `float32` | `rounding`、`saturate`不生效。 |
| `float16` | `bfloat16` | 使用`saturate=False`。 |
| `bfloat16` | `float16` | 舍入与饱和选项取决于目标半精度类型。 |
| 上述整数类型 | `float16`、`bfloat16` | 可用舍入选项取决于整数位宽及目标浮点类型。 |
| `float16`、`bfloat16` | 上述整数类型 | 转为8位整数时固定向零舍入；整数输出不应用`saturate`。 |
| 上述整数类型、`float32`、`bool` | 上述整数类型、`float32`、`bool`，且与源类型不同 | 按普通标量数值转换处理，不能依赖`rounding`或`saturate`改变行为。浮点转整数时应保证输入有限且结果在目标范围内。 |

- 表外组合不属于本页明确列出的转换范围；不能因为源、目标都在类型列表中就任意组合。
- 使用默认舍入模式`RN`；需要其他舍入模式时，必须满足对应类型组合的设备支持条件。
- `saturate=True`不是通用的`[0, 1]`截断；半精度饱和转换还依赖设备配置，仅设置该参数不保证生效。

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
