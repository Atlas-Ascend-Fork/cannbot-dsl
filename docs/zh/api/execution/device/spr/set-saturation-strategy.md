# set_saturation_strategy

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

设置饱和控制策略，选择使用单指令/API配置或全局饱和配置。

## 函数原型

```python
set_saturation_strategy(strategy: SaturationStrategy | Int64) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `strategy` | 输入 | `SaturationStrategy`枚举，或承载合法策略值0/1的设备侧`Int64`，例如`get_saturation_strategy()`返回的值。设备动态值需由调用者保证为0或1。 |

`SaturationStrategy`可从`cannbotdsl.ops.arch`或`cannbotdsl.ops`导入，其取值与Ascend C的`asc_override_strategy`对应：

| 枚举 | 枚举值 | CTRL[60] | 含义 |
| --- | --- | --- | --- |
| `USE_API` | 0 | 0 | 相应转换使用单指令/API路径的饱和配置，例如支持该参数的`reg.vcast(..., saturate=...)`。 |
| `USE_GLOBAL` | 1 | 1 | 相应转换使用全局饱和配置，相关开关通过[set_saturation_flag](set-saturation-flag.md)设置。 |

## 返回值说明

无返回值。

## 约束说明

- 本接口运行在标量流水[PIPE](../../pipeline.md).S上。
- 当策略为`SaturationStrategy.USE_API`时，全局CAST饱和配置不生效；若需要CAST全局饱和，必须先或同时将策略设置为`SaturationStrategy.USE_GLOBAL`。
- 切换策略不会自动修改各饱和模式的开关。模式说明见[set_saturation_flag](set-saturation-flag.md#参数说明)。

## 调用示例

以下示例由子核0切换为API策略、全局策略，并恢复原策略。Host侧校验策略回读和无关CTRL位保留。实际使用时，可在切换与恢复之间执行需要对应策略的计算。

```python
import torch
import torch_npu

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import (
    SaturationStrategy,
    get_ctrl,
    get_saturation_strategy,
    get_subblock_id,
    set_saturation_strategy,
)

@kernel
def _set_saturation_strategy_kernel(output):
    if get_subblock_id() == 0:
        saved = get_saturation_strategy()
        before = get_ctrl()
        set_saturation_strategy(SaturationStrategy.USE_API)
        api_ctrl = get_ctrl()
        api_strategy = get_saturation_strategy()
        set_saturation_strategy(SaturationStrategy.USE_GLOBAL)
        global_ctrl = get_ctrl()
        global_strategy = get_saturation_strategy()
        set_saturation_strategy(saved)
        restored_ctrl = get_ctrl()
        restored_strategy = get_saturation_strategy()
        output[0] = before
        output[1] = api_ctrl
        output[2] = global_ctrl
        output[3] = restored_ctrl
        output[4] = saved
        output[5] = api_strategy
        output[6] = global_strategy
        output[7] = restored_strategy

@host
def run(output):
    _set_saturation_strategy_kernel[1](output)

def main():
    output = torch.empty((8,), dtype=torch.int64, device="npu:0")
    run(output)
    torch.npu.synchronize()
    before, api, glob, restored, saved, api_mode, global_mode, after = output.cpu().tolist()
    bit = 1 << 60
    assert api == (before & ~bit)
    assert glob == (before | bit)
    assert restored == before
    assert (api_mode, global_mode, after) == (0, 1, saved)
    print("set_saturation_strategy example passed")

if __name__ == "__main__":
    main()
```

预期结果：

```text
set_saturation_strategy example passed
```
