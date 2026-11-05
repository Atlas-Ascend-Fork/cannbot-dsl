# simt.vatomic_cas

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对Unified Buffer或Global Memory上`ptr`的数值进行原子比较赋值操作，如果`ptr`上的数值等于指定数值`compare`，则对`ptr`赋值为指定数值`value`，否则`ptr`的数值不变。

## 函数原型

```python
simt.vatomic_cas(
    ptr,
    compare: float16x2 | bfloat16x2,
    value: float16x2 | bfloat16x2,
) -> float16x2 | bfloat16x2
```

其中，`float16x2`、`bfloat16x2`为DSL数据类型名，分别对应`dtypes.float16x2`、`dtypes.bfloat16x2`。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ptr | 输入/输出 | Unified Buffer或Global Memory的地址。通过`tensor.ptr(...)`获取有类型指针，指向两个连续有效元素；指针元素类型须与`value`的分量类型一致。 |
| compare | 输入 | 源操作数，做比较的值。必须是与`value`相同数据类型的双分量短向量。 |
| value | 输入 | 源操作数，用于赋值的值。必须是双分量短向量，不接受普通标量或Python数值字面量；可通过 [simt.make_short_vector](../conversion/make-short-vector.md) 构造。 |

不同数据类型支持的内存范围说明如下：

| 参数数据类型 | 支持的内存空间 |
| --- | --- |
| `float16x2`、`bfloat16x2` | Unified Buffer、Global Memory |

标量类型使用 [simt.atomic_cas](./atomic-cas.md)。

## 返回值说明

Unified Buffer或Global Memory上的初始数据。

返回与`value`相同数据类型的双分量短向量；更新结果写回`ptr`指向的两个元素。

## 约束说明

无

## 调用示例

调用`simt.vatomic_cas`更新数据，并打印更新前后的值。

```python
import torch
import torch_npu
from cannbotdsl import simt, dtypes, dim3
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def demo(src: Tensor, dst: Tensor, old: Tensor):
    with vf(mode='simt'):
        value = simt.make_short_vector(src[0], src[1])
        compare = simt.make_short_vector(src[2], src[3])
        previous = simt.vatomic_cas(dst.ptr(0), compare, value)
        old[0] = previous.x
        old[1] = previous.y


@host
def run(src, dst, old):
    demo[dim3(1), dim3(1)](src, dst, old)


def main():
    src = torch.tensor([3, 6, 10, 20], device='npu', dtype=torch.float16)
    dst = torch.tensor([10, 20], device='npu', dtype=torch.float16)
    old = torch.zeros(2, device='npu', dtype=torch.float16)
    run(src, dst, old)
    torch.npu.synchronize()
    assert dst.cpu().tolist() == [3, 6]
    assert old.cpu().tolist() == [10, 20]
    print('更新后：', dst.cpu().tolist())
    print('更新前：', old.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
更新后： [3.0, 6.0]
更新前： [10.0, 20.0]
```
