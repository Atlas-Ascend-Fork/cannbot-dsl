# load_bypass
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

从`ptr`指向的GM元素读取整型标量值并返回，读取过程绕过DCache。

## 函数原型

```python
load_bypass(ptr) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `ptr` | 输入 | 指向GM中待读取元素的指针，类型为`Pointer`。支持的元素数值类型见下文；本接口不修改指针值或所指元素。 |

**支持的元素数据类型：**

`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.int64`、`dtypes.uint64`。

## 返回值说明

返回从`ptr`指向的GM元素读取到的标量值，类型为`ScalarValue`，数值类型与该元素的数值类型一致。

## 约束说明

- **执行流水**：本接口运行在标量流水（`[PIPE](../../../execution/pipeline.md).S`）上。同一标量流水内的标量数据依赖由指令执行顺序保证。
- **地址要求**：`ptr`须指向GM可访问地址空间内的元素。
- **数据类型**：仅支持整型数据，不支持浮点类型。
- **访存同步**：读取前须保证对源地址的前序写入已完成。本接口不替代跨流水同步或核间同步。
- **缓存一致性**：本接口读取GM时绕过DCache，不维护缓存一致性。与经过DCache的访问混用时，调用方需按实际访问关系处理缓存写回或失效，并保证相关访存的执行顺序。缓存管理说明见[缓存控制接口汇总](../../../execution/synchronization/cache/overview.md#缓存控制接口汇总)。

## 调用示例

以下示例演示`load_bypass`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import load_bypass, vec_store_bypass

@kernel
def _load_bypass_kernel(src, dst):
    vec_store_bypass(dst.ptr(0), load_bypass(src.ptr(0)))

@host
def run(src, dst):
    _load_bypass_kernel[1](src, dst)

def main():
    src = torch.full((1,), 42, dtype=torch.int32, device="npu:0")
    dst = torch.zeros((1,), dtype=torch.int32, device="npu:0")

    run(src, dst)
    torch.npu.synchronize()

    result = int(dst.cpu()[0])
    assert result == 42, result
    print("load_bypass example passed")
    print(f"loaded={result}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
load_bypass example passed
loaded=42
```
