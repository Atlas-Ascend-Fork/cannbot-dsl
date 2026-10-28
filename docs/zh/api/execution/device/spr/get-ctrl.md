# get_ctrl
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

读取CTRL寄存器（控制寄存器）特定比特位上的值。

## 函数原型

```python
get_ctrl() -> int
```

## 参数说明

无。

## 返回值说明

返回CTRL寄存器的值，设备侧类型为`Int64`。各比特位含义参见[`set_ctrl`参数说明](set-ctrl.md#参数说明)。

## 约束说明

- 本接口运行在标量流水`[PIPE](../../pipeline.md).S`上，只能在`@kernel`函数体或由其展开的设备侧辅助函数中调用，不能在Host或`vf(...)`作用域内直接调用。

## 调用示例

以下示例在不修改CTRL的情况下连续读取两次，由子核0写入结果，并在Host侧校验读值一致，不假定CTRL的具体初值。

```python
import torch
import torch_npu

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_ctrl, get_subblock_id

@kernel
def _get_ctrl_kernel(output):
    # 仅由子核0记录，避免多个Vector子核写入相同的输出位置。
    if get_subblock_id() == 0:
        first = get_ctrl()
        second = get_ctrl()
        output[0] = first
        output[1] = second

@host
def run(output):
    _get_ctrl_kernel[1](output)

def main():
    output = torch.empty((2,), dtype=torch.int64, device="npu:0")
    run(output)
    torch.npu.synchronize()

    first, second = output.cpu().tolist()
    assert first == second, (first, second)
    print("get_ctrl example passed")

if __name__ == "__main__":
    main()
```

预期结果：

```text
get_ctrl example passed
```
