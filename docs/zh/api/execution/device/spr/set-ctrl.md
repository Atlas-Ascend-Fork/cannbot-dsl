# set_ctrl
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对CTRL寄存器（控制寄存器）的特定比特位进行设置。

## 函数原型

```python
set_ctrl(value: int) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `value` | 输入 | 待写入CTRL的完整64位配置，公开类型标注为`int`。通常传入`get_ctrl()`保存后仅修改目标位得到的设备侧64位整型值。Python整数字面量需能表示为有符号64位整数，范围为[-2^63，2^63−1]；负值按其64位比特模式写入，不表示只写低位或执行数值饱和转换。 |

**常用CTRL寄存器比特位：**

| CTRL比特位 | 功能 |
| --- | --- |
| CTRL[48] | 控制浮点计算及部分浮点精度转换的饱和模式。0为饱和模式，INF输出限制为相应符号的最大有限值，NaN输出为0；1为非饱和模式，保留INF/NaN输出。浮点计算支持`float16`；浮点精度转换支持`hifloat8`、`float8_e8m0`、`float8_e5m2`、`float8_e4m3fn`、`float16`和`bfloat16`。 |
| CTRL[49] | 控制标量单元访问Unified Buffer（UB）时是否经过DCache。0表示直接从UB读取；1表示从DCache读取。对UB进行大量读取且UB中数据不被Vector单元写入时，可选择开启该模式。 |
| CTRL[50] | 控制`float8_e8m0`、`float8_e5m2`、`float8_e4m3fn`相关场景的饱和模式。0为饱和模式；1为非饱和模式。 |
| CTRL[53] | 控制整数计算指令的饱和模式。0为截断模式，溢出时保留目标位宽内的低位；1为饱和模式，将溢出值限制在目标类型的可表示范围内。 |
| CTRL[56] | 控制MASK模式。0为按位掩码模式，每位指示一个元素是否参与计算；1为数量模式，MASK[31:0]指定实际操作的元素个数。 |
| CTRL[59] | 控制浮点转整数或整数转整数时的转换饱和模式，仅在CTRL[60]为1时生效。0为饱和模式；1为截断模式，溢出时保留目标位宽内的低位。 |
| CTRL[60] | 选择饱和控制策略。0使用单指令/API路径的饱和配置；1使用全局CTRL饱和配置。 |

## 返回值说明

无返回值。

## 约束说明

- 本接口运行在标量流水`[PIPE](../../pipeline.md).S`上，只能在`@kernel`函数体或由其展开的设备侧辅助函数中调用，不能在Host或`vf(...)`作用域内直接调用。
- 使用前需要先用[`get_ctrl()`](get-ctrl.md)获取当前值，仅对需要变更的bit位进行修改。

## 调用示例

以下示例由子核0保存CTRL，仅翻转整数计算饱和位CTRL[53]，读取写入后的值并恢复原值，再在Host侧校验目标位变化、其他位不变和恢复成功。

```python
import torch
import torch_npu

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_ctrl, get_subblock_id, set_ctrl

@kernel
def _set_ctrl_kernel(output):
    # 不假定初始模式；仅修改目标位，且在本Kernel内显式恢复。
    if get_subblock_id() == 0:
        saved = get_ctrl()
        changed = saved ^ (1 << 53)
        set_ctrl(changed)
        observed = get_ctrl()
        set_ctrl(saved)
        restored = get_ctrl()
        # 恢复完成后记录，避免示例将临时状态带入其他计算。
        output[0] = saved
        output[1] = observed
        output[2] = restored

@host
def run(output):
    _set_ctrl_kernel[1](output)

def main():
    output = torch.empty((3,), dtype=torch.int64, device="npu:0")
    run(output)
    torch.npu.synchronize()

    saved, observed, restored = output.cpu().tolist()
    bits64 = (1 << 64) - 1
    assert ((saved ^ observed) & bits64) == (1 << 53)
    assert restored == saved, (saved, restored)
    print("set_ctrl example passed")

if __name__ == "__main__":
    main()
```

预期结果：

```text
set_ctrl example passed
```
