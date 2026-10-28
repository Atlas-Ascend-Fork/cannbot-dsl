# clear_nthbit
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对`dtypes.uint64`类型的标量`bits`执行位清除操作：将第`idx`位清零，其余位保持不变，并返回计算结果。

## 函数原型

```python
clear_nthbit(bits, idx) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `bits` | 输入 | 待操作的64位无符号整数，数值类型为`dtypes.uint64`，取值范围为$[0, 2^{64}-1]$。 |
| `idx` | 输入 | 待操作的位索引，指定需要清零的二进制位，最低位编号为0。实际位索引按`(idx % 64 + 64) % 64`计算，范围为`[0, 63]`。 |

## 返回值说明

返回指定位清零后的结果，类型为`ScalarValue`，数值类型为`dtypes.uint64`。输入`bits`保持不变。

## 约束说明

- **执行流水**：本接口运行在标量流水（`[PIPE](../../../execution/pipeline.md).S`）上。同一标量流水内的标量数据依赖由指令执行顺序保证，无需额外同步。
- **执行侧**：AIC与AIV均可调用，行为一致。

## 调用示例

以下示例演示`clear_nthbit`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import clear_nthbit, vec_store_bypass

@kernel
def _clear_nthbit_kernel(dst):
    vec_store_bypass(dst.ptr(0), clear_nthbit(15, 1))

@host
def run(dst):
    _clear_nthbit_kernel[1](dst)

def main():
    dst = torch.zeros((1,), dtype=torch.int64, device="npu:0").view(torch.uint64)

    run(dst)
    torch.npu.synchronize()

    result = int(dst.cpu().view(torch.int64)[0])
    assert result == 13, result
    print("clear_nthbit example passed")
    print(f"clear_nthbit(15, 1) = {result}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
clear_nthbit example passed
clear_nthbit(15, 1) = 13
```
