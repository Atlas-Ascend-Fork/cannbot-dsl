# get_saturation_strategy

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

读取当前饱和控制策略。

## 函数原型

```python
get_saturation_strategy() -> int
```

## 参数说明

无。

## 返回值说明

当前饱和控制策略，取值如下表所示。

| 返回值 | 对应枚举 | 含义 |
| --- | --- | --- |
| 0 | `SaturationStrategy.USE_API` | 对于提供单指令饱和配置的相应转换，使用API路径的配置。 |
| 1 | `SaturationStrategy.USE_GLOBAL` | 对于相应转换，使用全局饱和配置。 |

## 约束说明

- 本接口运行在标量流水[PIPE](../../pipeline.md).S上。
- `USE_API`不是关闭所有全局CTRL控制位的总开关；具体运算与模式的关系见[set_saturation_strategy](set-saturation-strategy.md#约束说明)。

## 调用示例

以下示例连续读取两次策略，由子核0记录结果，并在Host侧检查返回值的范围和一致性。

```python
import torch
import torch_npu

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_saturation_strategy, get_subblock_id

@kernel
def _get_saturation_strategy_kernel(output):
    if get_subblock_id() == 0:
        output[0] = get_saturation_strategy()
        output[1] = get_saturation_strategy()

@host
def run(output):
    _get_saturation_strategy_kernel[1](output)

def main():
    output = torch.empty((2,), dtype=torch.int64, device="npu:0")
    run(output)
    torch.npu.synchronize()
    first, second = output.cpu().tolist()
    assert first in (0, 1) and first == second
    print("get_saturation_strategy example passed")

if __name__ == "__main__":
    main()
```

预期结果：

```text
get_saturation_strategy example passed
```
