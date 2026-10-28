# atomic_add
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对Global Memory中`ptr`指向的单个元素执行原子加操作。读取该元素的旧值`old_value`，将`old_value + value`写回该地址，并返回`old_value`。整个读取、计算和写回过程为原子操作。

## 函数原型

```python
atomic_add(ptr, value) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `ptr` | 输入/输出 | Global Memory的地址，类型为`Pointer`，需按所指元素的字节数对齐。 |
| `value` | 输入 | 加数，数值类型须与`ptr`所指元素类型一致。 |

**支持的元素数据类型：**

dtype支持的数据类型为`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`、`dtypes.int64`、`dtypes.uint64`。

## 返回值说明

`ptr`地址中计算前的原始数据，类型为`ScalarValue`，数值类型与`ptr`所指元素类型一致。

## 约束说明

- 在开启编译器自动同步功能的前提下，编译器能够自动在[PIPE](../../../execution/pipeline.md).MTE2/PIPE.MTE3与PIPE.S之间插入同步。但是，`atomic_add`为标量计算，在读写GM时如果与搬运单元（MTE2/MTE3）存在数据依赖，编译器却无法自动插入同步，开发者需要根据实际情况手动插入同步。
- Scalar原子操作会绕过DCache，需要调用[`dcci_single`](../../../execution/synchronization/cache/dcci-single.md)接口确保GM与DCache的一致性。
- `dtypes.float32`类型下接口的性能与接口返回值是否被使用有关，具体情况如下：
    - 当使用接口返回值时，接口采用软仿实现，`dtypes.float32`类型下接口的性能低于其它数据类型。
    - 当不使用接口返回值时，`dtypes.float32`类型下接口的性能与其它数据类型一致。


## 调用示例

以下示例演示`atomic_add`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import dtypes, host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.scalar import atomic_add

@kernel
def _atomic_add_kernel(acc):
    # 4 个 block 各自把 acc[0] 原子加 1，累加结果与执行顺序无关
    atomic_add(acc.ptr(0), dtypes.int32(1))

@host
def run(acc):
    _atomic_add_kernel[4](acc)

def main():
    acc = torch.zeros(1, dtype=torch.int32, device="npu:0")
    run(acc)
    torch.npu.synchronize()

    assert acc.cpu().tolist() == [4], acc.cpu().tolist()
    print(f"atomic_add example passed, acc={acc.cpu().tolist()}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
atomic_add example passed, acc=[4]
```
