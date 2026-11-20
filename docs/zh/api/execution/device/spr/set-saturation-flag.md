# set_saturation_flag

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

按饱和模式类型设置对应饱和开关。`enable_sat`表示是否开启饱和。

## 函数原型

```python
set_saturation_flag(saturation_mode: SaturationMode, enable_sat: bool | Boolean) -> None
```

其中`Boolean`表示设备侧布尔值，例如`get_saturation_flag`的结果或设备比较表达式。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `saturation_mode` | 输入 | 编译期枚举`SaturationMode`，定义及取值见下表。 |
| `enable_sat` | 输入 | Python布尔字面量或设备侧`Boolean`值。`True`开启相应饱和开关，`False`关闭。可直接使用`get_saturation_flag`保存的值恢复配置。 |

`SaturationMode`可从`cannbotdsl.ops.arch`或`cannbotdsl.ops`导入，其取值与Ascend C的`asc_saturation_mode`对应：

| `saturation_mode` | `enable_sat=True` | `enable_sat=False` | 说明 |
| --- | --- | --- | --- |
| `FLOAT` | 饱和 | 非饱和 | 浮点计算及部分浮点精度转换场景的饱和控制。浮点计算支持`float16`；浮点精度转换支持`hifloat8`、`float8_e8m0`、`float8_e5m2`、`float8_e4m3fn`、`float16`和`bfloat16`。 |
| `FLOAT8` | 饱和 | 非饱和 | `float8_e8m0`、`float8_e5m2`、`float8_e4m3fn`相关场景的饱和控制。 |
| `INT` | 饱和 | 截断 | 整数计算溢出时的饱和/截断控制。 |
| `CAST` | 饱和 | 截断 | 浮点转整数或整数转整数时的精度转换饱和控制。仅在饱和策略为`SaturationStrategy.USE_GLOBAL`时生效。 |

一般饱和与非饱和行为见[舍入模式与饱和模式](../../../compute/roundingmode.md)，实际效果还取决于类型组合、指令支持和饱和策略。

## 返回值说明

无返回值。

## 约束说明

- 本接口运行在标量流水[PIPE](../../pipeline.md).S上。
- `enable_sat`是API语义参数：同一布尔值在不同模式下的关闭语义不同（`INT`为截断，`FLOAT`、`FLOAT8`和`CAST`为非饱和或截断），请以[参数说明](#参数说明)中的模式表为准。
- 使用全局浮点饱和配置时，需要通过[set_saturation_strategy](set-saturation-strategy.md)将策略设置为`SaturationStrategy.USE_GLOBAL`；对于`SaturationMode.FLOAT8`，还需先对`SaturationMode.FLOAT`开启饱和模式。
- 当`saturation_mode`为`SaturationMode.CAST`时，还需将饱和策略设置为`SaturationStrategy.USE_GLOBAL`，CAST全局饱和配置才会生效。可通过[set_saturation_strategy](set-saturation-strategy.md)配置。
- 本接口仅修改对应模式的饱和配置，不影响其他模式。

## 调用示例

以下示例由子核0关闭、开启整数计算饱和并回读，随后恢复原状态；Host同时校验其他CTRL位未被修改。

```python
import torch
import torch_npu

from cannbotdsl import dtypes, host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import (
    SaturationMode,
    get_ctrl,
    get_saturation_flag,
    get_subblock_id,
    set_saturation_flag,
)

@kernel
def _set_saturation_flag_kernel(output):
    if get_subblock_id() == 0:
        saved = get_saturation_flag(SaturationMode.INT)
        before = get_ctrl()
        set_saturation_flag(SaturationMode.INT, False)
        disabled = get_ctrl()
        flag_disabled = get_saturation_flag(SaturationMode.INT)
        set_saturation_flag(SaturationMode.INT, True)
        enabled = get_ctrl()
        flag_enabled = get_saturation_flag(SaturationMode.INT)
        set_saturation_flag(SaturationMode.INT, saved)
        restored = get_ctrl()
        flag_restored = get_saturation_flag(SaturationMode.INT)
        output[0] = before
        output[1] = disabled
        output[2] = enabled
        output[3] = restored
        output[4] = dtypes.int64(flag_disabled)
        output[5] = dtypes.int64(flag_enabled)
        output[6] = dtypes.int64(saved)
        output[7] = dtypes.int64(flag_restored)

@host
def run(output):
    _set_saturation_flag_kernel[1](output)

def main():
    output = torch.empty((8,), dtype=torch.int64, device="npu:0")
    run(output)
    torch.npu.synchronize()
    before, disabled, enabled, restored, off, on, saved, after = output.cpu().tolist()
    bit = 1 << 53
    assert disabled == (before & ~bit)
    assert enabled == (before | bit)
    assert restored == before
    assert (off, on, after) == (0, 1, saved)
    print("set_saturation_flag example passed")

if __name__ == "__main__":
    main()
```

预期结果：

```text
set_saturation_flag example passed
```
