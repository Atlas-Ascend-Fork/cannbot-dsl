# simt.l2cache_load

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

经L2 Cache读取ptr指向的数据；未命中时从GM获取并缓存到L2 Cache，不填入Data Cache。

## 函数原型

```python
simt.l2cache_load(ptr) -> 标量
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ptr | 输入 | 由`tensor.ptr(...)`获取的GM指针，元素类型为`int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`int64`、`uint64`、`float16`、`bfloat16`、`float32`。 |

## 返回值说明

返回与ptr元素同类型的标量，不修改目标内存。

## 约束说明

- UB和通用地址空间指针被拒绝。
- ptr地址按元素类型自然对齐并处于有效内存范围。
- 同一地址的数据已存入读取核的Data Cache后，其他核或原子操作更新GM不会使该缓存行自动失效。读取核再次通过Data Cache访问前，须先使对应缓存行失效，参见[缓存控制](../../execution/synchronization/cache/overview.md)。线程间通过完成标志交接数据时，还须满足[内存可见性与顺序要求](../synchronization/threadfence.md)。

## 调用示例

经L2 Cache读取一个元素，并打印结果。

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
        dst[0] = simt.l2cache_load(src.ptr(0))


@host
def run(src, dst):
    demo[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([3, 7], device='npu', dtype=torch.int32)
    dst = torch.zeros(2, device='npu', dtype=torch.int32)
    run(src, dst)
    torch.npu.synchronize()
    assert dst.cpu().tolist() == [3, 0]
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[3, 0]
```
