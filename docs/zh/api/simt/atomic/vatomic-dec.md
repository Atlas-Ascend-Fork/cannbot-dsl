# simt.vatomic_dec

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

逐分量读取`ptr`指向的整数：若原值为0或大于`value`的对应分量，则写入该分量值；否则将原值减1。每个分量的读取、判断和写回为原子操作。

## 函数原型

```python
simt.vatomic_dec(ptr, value: ShortVector) -> ShortVector
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ptr | 输入/输出 | 待更新数据的起始地址，通过`tensor.ptr(...)`获取；指针元素类型须与`value`的分量类型相同。 |
| value | 输入 | 指定各元素计数上限的整数短向量，包含2、3或4个分量，可通过[simt.make_short_vector](../conversion/make-short-vector.md)构造。 |

支持的分量类型及内存空间如下：

| 分量数据类型 | 支持的内存空间 |
| --- | --- |
| `uint32` | Unified Buffer（UB）、Global Memory（GM） |
| `uint64` | Global Memory（GM） |

例如，`dtypes.uint32x2`表示包含两个`uint32`分量的短向量。

## 返回值说明

返回与`value`类型相同的短向量，各分量为对应内存元素更新前的值。更新后的值写回`ptr`指向的连续元素。

## 约束说明

- 从`ptr`开始须有足够的连续有效元素，元素数量不小于`value`的分量数；地址按元素大小对齐。
- 每个分量独立保证原子性，整组分量不保证同时更新。多个线程并发操作时，返回的各分量不保证来自同一时刻。

## 调用示例

对两个整数执行逐分量原子递减，打印更新前后的值。

```python
import torch
import torch_npu
from cannbotdsl import simt, dim3
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def demo(src: Tensor, dst: Tensor, old: Tensor):
    with vf(mode='simt'):
        value = simt.make_short_vector(src[0], src[1])
        previous = simt.vatomic_dec(dst.ptr(0), value)
        old[0] = previous.x
        old[1] = previous.y


@host
def run(src, dst, old):
    demo[dim3(1), dim3(1)](src, dst, old)


def main():
    src = torch.tensor([3, 3], dtype=torch.uint32).to('npu')
    dst = torch.tensor([0, 2], dtype=torch.uint32).to('npu')
    old = torch.tensor([0, 0], dtype=torch.uint32).to('npu')
    run(src, dst, old)
    torch.npu.synchronize()
    assert dst.cpu().tolist() == [3, 1]
    assert old.cpu().tolist() == [0, 2]
    print('更新后：', dst.cpu().tolist())
    print('更新前：', old.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
更新后： [3, 1]
更新前： [0, 2]
```
