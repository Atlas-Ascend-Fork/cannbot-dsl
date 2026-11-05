# simt.nop

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

插入一次空操作，不计算也不读写用户数据。它不提供线程同步或内存顺序保证。

## 函数原型

```python
simt.nop() -> None
```

## 参数说明

无

## 返回值说明

无

## 约束说明

- 不是同步、内存栅栏或缓存刷新，不得用它代替正确的线程间同步。

## 调用示例

在线程计算过程中插入空操作，并打印计算结果。

```python
import torch
import torch_npu
from cannbotdsl import dim3, host, simt
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def example_kernel(src: Tensor, dst: Tensor):
    with vf(mode='simt'):
        i = simt.thread_idx()[0]
        value = src[i] + 1
        simt.nop()
        dst[i] = value


@host
def run(src, dst):
    example_kernel[dim3(1), dim3(4)](src, dst)


def main():
    src = torch.arange(4, dtype=torch.int32, device='npu')
    dst = torch.empty((4,), dtype=torch.int32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    assert torch.equal(dst.cpu(), torch.arange(1, 5, dtype=torch.int32))
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[1, 2, 3, 4]
```
