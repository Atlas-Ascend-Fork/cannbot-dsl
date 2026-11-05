# simt.atomic_exch

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对Unified Buffer或Global Memory地址做原子赋值操作，即将指定数据赋值到Unified Buffer或Global Memory地址中。

## 函数原型

```python
simt.atomic_exch(
    ptr,
    value: ScalarValue | int | float,
) -> ScalarValue
```

其中，`ScalarValue`从`cannbotdsl.types`导入。支持的数据类型及匹配要求见参数说明。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ptr | 输入/输出 | Unified Buffer或Global Memory的地址。通过`tensor.ptr(...)`获取有类型指针，不能传入整数地址；访问位置必须在有效范围内。 |
| value | 输入 | 源操作数，用于赋值的值。有类型的标量须与`ptr`的元素类型一致；也支持能按该元素类型构造的Python数值字面量。 |

不同数据类型支持的内存范围说明如下：

| 参数数据类型 | 支持的内存空间 |
| --- | --- |
| `int32`、`uint32`、`float32` | Unified Buffer、Global Memory |
| `int64`、`uint64` | Global Memory |

双分量类型`float16x2`、`bfloat16x2`使用 [simt.vatomic_exch](./vatomic-exch.md)。

## 返回值说明

Unified Buffer或Global Memory上的初始数据。

返回`ScalarValue`，数据类型与`ptr`的元素类型相同；更新结果写回`ptr`。

## 约束说明

无

## 调用示例

调用`simt.atomic_exch`更新数据，并打印更新前后的值。

```python
import torch
import torch_npu
from cannbotdsl import simt, dtypes, dim3
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def demo(dst: Tensor, old: Tensor):
    with vf(mode='simt'):
        old[0] = simt.atomic_exch(dst.ptr(0), 3)


@host
def run(dst, old):
    demo[dim3(1), dim3(1)](dst, old)


def main():
    dst = torch.tensor([10], device='npu', dtype=torch.int32)
    old = torch.zeros(1, device='npu', dtype=torch.int32)
    run(dst, old)
    torch.npu.synchronize()
    assert dst.cpu().tolist() == [3]
    assert old.cpu().tolist() == [10]
    print('更新后：', dst.cpu().tolist())
    print('更新前：', old.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
更新后： [3]
更新前： [10]
```
