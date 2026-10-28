# cast
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对`dtypes.float32`类型的标量`value`按`rounding`指定的舍入模式进行转换，返回`dtypes.int32`类型的结果。

## 函数原型

```python
cast(value, *, dtype, rounding: RoundingMode | str = RoundingMode.RN) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `value` | 输入 | 待转换的标量，数值类型为`dtypes.float32`。 |
| `dtype` | 输入 | 目标数值类型，当前仅支持`dtypes.int32`。 |
| `rounding` | 输入 | 舍入模式，使用[RoundingMode](./roundingmode.md)或对应的小写字符串。默认值为`RoundingMode.RN`；支持`RN`、`RNA`、`RD`、`RU`，或`"rn"`、`"rna"`、`"rd"`、`"ru"`。 |

## 返回值说明

返回将`value`按指定舍入模式转换后的结果，类型为`ScalarValue`，数值类型为`dtypes.int32`。

## 约束说明

- **执行流水**：本接口运行在标量流水（`[PIPE](../../../execution/pipeline.md).S`）上。同一标量流水内的标量数据依赖由指令执行顺序保证，无需额外同步。
- **输入范围**：`value`须为有限数值，且舍入后的结果须落在`dtypes.int32`的可表示范围$[-2^{31}, 2^{31}-1]$内。
- **目标类型**：`dtype`仅支持`dtypes.int32`，传入其他类型时抛出`TypeError`。
- **舍入模式**：使用标量转换的`RoundingMode`或其对应字符串，不能传入寄存器级舍入枚举。模式定义见[RoundingMode](./roundingmode.md)。

## 调用示例

以下示例演示`cast`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import dtypes, host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import cast, vec_store_bypass

@kernel
def _cast_kernel(dst):
    vec_store_bypass(dst.ptr(0), cast(2.7, dtype=dtypes.int32))

@host
def run(dst):
    _cast_kernel[1](dst)

def main():
    dst = torch.zeros((1,), dtype=torch.int32, device="npu:0")

    run(dst)
    torch.npu.synchronize()

    result = int(dst.cpu()[0])
    assert result == 3, result
    print("cast example passed")
    print(f"cast(2.7) = {result}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
cast example passed
cast(2.7) = 3
```
