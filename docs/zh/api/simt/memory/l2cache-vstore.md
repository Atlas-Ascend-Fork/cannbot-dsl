# simt.l2cache_vstore

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将value写入ptr指向的GM地址，并缓存到L2 Cache，不填入Data Cache。

## 函数原型

```python
simt.l2cache_vstore(ptr, value) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ptr | 输出 | 由`tensor.ptr(...)`获取的GM指针，元素类型为`int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`int64`、`uint64`、`float16`、`bfloat16`、`float32`，必须与短向量分量类型相同。 |
| value | 输入 | 待写入的短向量类型：整数`int8/uint8`、`int16/uint16`、`int32/uint32`、`int64/uint64`的2或4分量类型，`float32x2`、`float32x4`、`float16x2`、`bfloat16x2`。 |

## 返回值说明

无返回值。结果写入ptr指向的内存。

## 约束说明

- UB和通用地址空间指针被拒绝。
- ptr元素必须与短向量分量一致，所有分量须连续、在界内，起始地址按向量ABI对齐。调用者需保证元素类型与边界正确。
- 同一地址的数据已存入读取核的Data Cache后，其他核或原子操作更新GM不会使该缓存行自动失效。读取核再次通过Data Cache访问前，须先使对应缓存行失效，参见[缓存控制](../../execution/synchronization/cache/overview.md)。线程间通过完成标志交接数据时，还须满足[内存可见性与顺序要求](../synchronization/threadfence.md)。

## 调用示例

将短向量的分量连续写入全局内存，并打印结果。

```python
import torch
import torch_npu
from cannbotdsl import simt, dtypes, dim3
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def demo(src: Tensor, dst: Tensor):
    with vf(mode='simt'):
        x = simt.make_short_vector(src[0], src[1])
        simt.l2cache_vstore(dst.ptr(0), x)


@host
def run(src, dst):
    demo[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([3, 7], device='npu', dtype=torch.float16)
    dst = torch.zeros(2, device='npu', dtype=torch.float16)
    run(src, dst)
    torch.npu.synchronize()
    assert dst.cpu().tolist() == [3, 7]
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[3.0, 7.0]
```
