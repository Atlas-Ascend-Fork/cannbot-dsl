# sflbits
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

统计`dtypes.int64`类型的标量`value`的二进制表示中，从次高位向最低位连续与符号位相同的位数，并返回统计结果。符号位为最高位，统计时不包含符号位。

## 函数原型

```python
sflbits(value) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `value` | 输入 | 待统计的64位有符号整数，数值类型为`dtypes.int64`，取值范围为$[-2^{63}, 2^{63}-1]$。 |

## 返回值说明

返回从次高位开始连续与符号位相同的位数，类型为`ScalarValue`，数值类型为`dtypes.int64`。`value`为 -1（所有位均为1）或0（所有位均为0）时返回 -1；其余输入的返回值范围为`[0, 62]`。

## 约束说明

- **执行流水**：本接口运行在标量流水（`[PIPE](../../../execution/pipeline.md).S`）上。同一标量流水内的标量数据依赖由指令执行顺序保证，无需额外同步。
- **数据类型**：本接口按`dtypes.int64`执行计算。其他位宽的输入会先隐式转换为`dtypes.int64`，结果以`dtypes.int64`返回。

## 调用示例

以下示例演示`sflbits`的调用，并在Host侧校验结果。输入`0x0f00000000000000`的符号位为0，从次高位开始连续有3位为0，因此返回3。

```python
import torch
import torch_npu

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import sflbits, vec_store_bypass

@kernel
def _sflbits_kernel(dst):
    vec_store_bypass(dst.ptr(0), sflbits(0x0F00000000000000))

@host
def run(dst):
    _sflbits_kernel[1](dst)

def main():
    dst = torch.zeros((1,), dtype=torch.int64, device="npu:0")

    run(dst)
    torch.npu.synchronize()

    result = int(dst.cpu()[0])
    assert result == 3, result
    print("sflbits example passed")
    print(f"sflbits(0x0f00000000000000) = {result}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
sflbits example passed
sflbits(0x0f00000000000000) = 3
```
