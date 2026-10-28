# atomic_min
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对Global Memory中`ptr`指向的单个元素执行原子取小操作：读取该地址中的旧值`old_value`，比较`old_value`与输入标量值`value`，将较小值`new_value`写回该地址，并返回`old_value`。整个读取、比较和写回过程为原子操作。

计算公式如下：

$$
new\_value = Min(old\_value, value)
$$

## 函数原型

```python
atomic_min(ptr, value) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `ptr` | 输入/输出 | Global Memory的地址，类型为`Pointer`，需按所指元素的字节数对齐。 |
| `value` | 输入 | 参与取小比较的候选值，数值类型须与`ptr`所指元素类型一致。 |

**支持的元素数据类型：**

`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`、`dtypes.int64`、`dtypes.uint64`。

## 返回值说明

`ptr`地址中计算前的原始数据，类型为`ScalarValue`，数值类型与`ptr`所指元素类型一致。

## 约束说明

- 本接口不受数据搬运随路原子设置影响，也不会影响后续搬运指令。
- 本接口运行在标量流水（`[PIPE](../../../execution/pipeline.md).S`）上，同一标量流水内的数据依赖由指令执行顺序保证。若本接口与`PIPE.MTE2`或`PIPE.MTE3`上的数据搬运指令访问同一GM地址，且执行顺序影响结果，编译器无法自动完成跨流水同步，调用方需按实际依赖插入`vec_sync_pipe`，或配合使用`vec_sync_notify`与`vec_sync_wait`保证执行顺序。
- 本接口访问GM时绕过DCache，不维护缓存一致性。若其他核或其他通路通过缓存访问同一GM地址，调用方需使用[`dcci_single`](../../../execution/synchronization/cache/dcci-single.md)清理或失效对应Cache Line，并保证相关访存操作的执行顺序和数据可见性。
- `dtypes.float32`类型下接口的性能与接口返回值是否被使用有关，具体情况如下：
    - 当使用接口返回值时，接口采用软仿实现，`dtypes.float32`类型下接口的性能低于其它数据类型。
    - 当不使用接口返回值时，`dtypes.float32`类型下接口的性能与其它数据类型一致。


## 调用示例

以下示例演示`atomic_min`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import dtypes, host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.scalar import atomic_min

@kernel
def _atomic_min_kernel(acc):
    # 4 个 block 分别以自己序号 + 1（1~4）参与原子取小
    atomic_min(acc.ptr(0), dtypes.int32(get_block_idx() + 1))

@host
def run(acc):
    _atomic_min_kernel[4](acc)

def main():
    acc = torch.full((1,), 100, dtype=torch.int32, device="npu:0")
    run(acc)
    torch.npu.synchronize()

    assert acc.cpu().tolist() == [1], acc.cpu().tolist()
    print(f"atomic_min example passed, acc={acc.cpu().tolist()}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
atomic_min example passed, acc=[1]
```
