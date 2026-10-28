# popc

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

统计`dtypes.uint64`类型的标量`value`的二进制表示中为1的位数，并返回统计结果。

## 函数原型

```python
popc(value) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `value` | 输入 | 待统计的64位无符号整数，数值类型为`dtypes.uint64`，取值范围为$[0, 2^{64}-1]$。 |

## 返回值说明

返回`value`的二进制表示中为1的位数，类型为`ScalarValue`，数值类型为`dtypes.int64`，取值范围为`[0, 64]`。

## 约束说明

- **执行流水**：本接口运行在标量流水（`[PIPE](../../../execution/pipeline.md).S`）上。同一标量流水内的标量数据依赖由指令执行顺序保证，无需额外同步。

## 调用示例

以下示例演示`popc`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import popc, vec_store_bypass

@kernel
def _popc_kernel(dst):
    vec_store_bypass(dst.ptr(0), popc(15))

@host
def run(dst):
    _popc_kernel[1](dst)

def main():
    dst = torch.zeros((1,), dtype=torch.int64, device="npu:0")

    run(dst)
    torch.npu.synchronize()

    result = int(dst.cpu()[0])
    assert result == 4, result
    print("popc example passed")
    print(f"popc(15) = {result}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
popc example passed
popc(15) = 4
```
