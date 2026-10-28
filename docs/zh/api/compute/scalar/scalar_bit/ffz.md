# ffz
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

在`dtypes.uint64`类型的标量`value`的二进制表示中，从最低位向最高位查找第一个为0的位，返回其位索引；未找到时返回 -1。

## 函数原型

```python
ffz(value) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `value` | 输入 | 待查找的64位无符号整数，数值类型为`dtypes.uint64`，取值范围为$[0, 2^{64}-1]$。最低位编号为0，最高位编号为63。 |

## 返回值说明

返回从最低位向最高位查找到的第一个为0的位索引，类型为`ScalarValue`，数值类型为`dtypes.int64`。位索引范围为`[0, 63]`；未找到为0的位时返回 -1。

## 约束说明

- **执行流水**：本接口运行在标量流水（`[PIPE](../../../execution/pipeline.md).S`）上。同一标量流水内的标量数据依赖由指令执行顺序保证，无需额外同步。
- **返回值使用**：将返回值用作位索引前，需先判断其是否为 -1。

## 调用示例

以下示例演示`ffz`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import ffz, vec_store_bypass

@kernel
def _ffz_kernel(dst):
    vec_store_bypass(dst.ptr(0), ffz(3))

@host
def run(dst):
    _ffz_kernel[1](dst)

def main():
    dst = torch.zeros((1,), dtype=torch.int64, device="npu:0")

    run(dst)
    torch.npu.synchronize()

    result = int(dst.cpu()[0])
    assert result == 2, result
    print("ffz example passed")
    print(f"ffz(3) = {result}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
ffz example passed
ffz(3) = 2
```
