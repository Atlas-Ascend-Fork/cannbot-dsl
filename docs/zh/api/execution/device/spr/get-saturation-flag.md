# get_saturation_flag

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

按饱和模式类型读取当前是否开启饱和，并返回布尔类型的结果。

## 函数原型

```python
get_saturation_flag(saturation_mode: SaturationMode) -> bool
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `saturation_mode` | 输入 | 编译期枚举`SaturationMode`，取值为`FLOAT`、`FLOAT8`、`INT`或`CAST`。枚举及各模式说明见[set_saturation_flag](set-saturation-flag.md#参数说明)。 |

## 返回值说明

`True`表示当前该模式下饱和已开启；`False`表示未开启（对`INT`为截断；对`FLOAT`、`FLOAT8`和`CAST`为非饱和或截断）。

## 约束说明

- 本接口运行在标量流水[PIPE](../../pipeline.md).S上。
- 返回值是API语义层结果：`True`表示该模式下饱和已开启，`False`表示未开启（对`INT`为截断；对`FLOAT`、`FLOAT8`和`CAST`为非饱和或截断）。各模式的语义差异见[set_saturation_flag参数说明](set-saturation-flag.md#参数说明)。
- 对于`CAST`模式，即使本接口返回`True`，也需同时满足饱和策略为`SaturationStrategy.USE_GLOBAL`，CAST饱和配置才会在精度转换中生效。详见[set_saturation_strategy](set-saturation-strategy.md)。

## 调用示例

以下示例连续读取CAST饱和开关，由子核0记录结果，并在Host侧校验两次读值一致。

```python
import torch
import torch_npu

from cannbotdsl import dtypes, host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import (
    SaturationMode,
    get_saturation_flag,
    get_subblock_id,
)

@kernel
def _get_saturation_flag_kernel(output):
    if get_subblock_id() == 0:
        first = get_saturation_flag(SaturationMode.CAST)
        second = get_saturation_flag(SaturationMode.CAST)
        output[0] = dtypes.int64(first)
        output[1] = dtypes.int64(second)

@host
def run(output):
    _get_saturation_flag_kernel[1](output)

def main():
    output = torch.empty((2,), dtype=torch.int64, device="npu:0")
    run(output)
    torch.npu.synchronize()
    first, second = output.cpu().tolist()
    assert first in (0, 1) and first == second
    print("get_saturation_flag example passed")

if __name__ == "__main__":
    main()
```

预期结果：

```text
get_saturation_flag example passed
```
