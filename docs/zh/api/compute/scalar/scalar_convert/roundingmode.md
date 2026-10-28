# RoundingMode
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 简介

`RoundingMode`是标量转换的舍入模式枚举，用于`scalar.cast`的`rounding`参数，决定转换时如何选择结果值。枚举成员的存在不表示每个转换接口都接受该成员，`scalar.cast`的可用子集见[约束说明](#约束说明)。

寄存器级转换使用`reg.RoundingMode`，两者是相互独立的枚举。

## 定义

```python
from cannbotdsl.ops.scalar import RoundingMode

RoundingMode.RN  # 四舍六入五成双舍入（RINT）
RoundingMode.RNA  # 四舍五入舍入（ROUND）
RoundingMode.RD  # 向负无穷方向舍入（FLOOR）
RoundingMode.RU  # 向正无穷方向舍入（CEIL）
RoundingMode.RZ  # 向零方向舍入（TRUNC）
RoundingMode.RO  # 奇数舍入（ODD）
RoundingMode.RH  # 混合舍入模式（HYBRID）
RoundingMode.NA  # 由实现按约定选择
```

## 约束说明

- `scalar.cast`的`rounding`默认值为`RoundingMode.RN`。
- `scalar.cast`当前只接受本枚举的`RN`、`RNA`、`RD`、`RU`子集，不能因为其他成员存在就将其传给该接口。
- `rounding`也可传入匹配的小写字符串（`"rn"`、`"rna"`、`"rd"`、`"ru"`）。
- 本枚举仅用于标量转换；寄存器级转换使用`reg.RoundingMode`。

## 调用示例

以下示例以显式`RoundingMode.RD`调用`scalar.cast`，并在Host侧校验舍入结果。

```python
import torch
import torch_npu

from cannbotdsl import dtypes, host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import RoundingMode, cast, vec_store_bypass

@kernel
def _rounding_mode_kernel(dst):
    vec_store_bypass(
        dst.ptr(0),
        cast(2.7, dtype=dtypes.int32, rounding=RoundingMode.RD),
    )

@host
def run(dst):
    _rounding_mode_kernel[1](dst)

def main():
    dst = torch.zeros((1,), dtype=torch.int32, device="npu:0")

    run(dst)
    torch.npu.synchronize()

    result = int(dst.cpu()[0])
    assert result == 2, result
    print("RoundingMode example passed")
    print(f"cast(2.7, RD) = {result}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
RoundingMode example passed
cast(2.7, RD) = 2
```
