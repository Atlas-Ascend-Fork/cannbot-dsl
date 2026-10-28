# atomic_exch
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对Global Memory中`ptr`指向的单个元素执行原子交换操作。读取该元素的旧值`old_value`，将`value`写回该地址以替换旧值，并返回`old_value`。整个读取和写回过程为原子操作。

## 函数原型

```python
atomic_exch(ptr, value) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `ptr` | 输入/输出 | Global Memory的地址，类型为`Pointer`，需按所指元素的字节数对齐。 |
| `value` | 输入 | 用于替换目标元素的新值，数值类型须与`ptr`所指元素类型一致。 |

**支持的元素数据类型：**

dtype支持的数据类型为`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`、`dtypes.int64`、`dtypes.uint64`。

## 返回值说明

`ptr`地址中计算前的原始数据，类型为`ScalarValue`，数值类型与`ptr`所指元素类型一致。

## 约束说明

- 在开启编译器自动同步功能的前提下，编译器能够自动在[PIPE](../../../execution/pipeline.md).MTE2/PIPE.MTE3与PIPE.S之间插入同步。但是，`atomic_exch`为标量计算，在读写GM时如果与搬运单元（MTE2/MTE3）存在数据依赖，编译器却无法自动插入同步，开发者需要根据实际情况手动插入同步。
- Scalar原子操作会绕过DCache，需要调用[`dcci_single`](../../../execution/synchronization/cache/dcci-single.md)接口确保GM与DCache的一致性。


## 调用示例

以下示例演示`atomic_exch`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import dtypes, host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import atomic_exch

@kernel
def _atomic_exch_kernel(acc):
    # 用 7 替换 acc[0] 的旧值，返回值即被替换掉的旧值
    atomic_exch(acc.ptr(0), dtypes.int32(7))

@host
def run(acc):
    _atomic_exch_kernel[1](acc)

def main():
    acc = torch.zeros(1, dtype=torch.int32, device="npu:0")
    run(acc)
    torch.npu.synchronize()

    assert acc.cpu().tolist() == [7], acc.cpu().tolist()
    print(f"atomic_exch example passed, acc={acc.cpu().tolist()}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
atomic_exch example passed, acc=[7]
```
