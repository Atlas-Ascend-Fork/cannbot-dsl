# simt.dcci_entire_out

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

维护当前核中面向全局内存（GM）的整个Data Cache缓存范围，使后续读取能够获取已更新的数据。

![缓存维护与数据读取顺序](figures/cache-read-order.svg)

先通过同步协议确认外部更新已完成，再维护读取方缓存并读取。下面的调用示例只演示接口用法，不包含跨核通知流程。

## 函数原型

```python
simt.dcci_entire_out() -> None
```

## 参数说明

无

## 返回值说明

无

## 约束说明

- 本接口刷新当前核面向GM读取的整个数据缓存，不用于UB缓存维护。
- DCCI处理缓存一致性，不是线程屏障，也不代替内存栅栏或原子操作。外部写入方必须先按同步协议完成更新，再由读取方维护缓存并读取；不能仅刷新缓存就假定其他线程已完成。
- 缓存维护作用于当前核，不负责其他核的脏数据写回。

## 调用示例

刷新数据缓存后读取输入值，并将结果写入输出Tensor。

```python
import torch
import torch_npu
from cannbotdsl import simt, dim3
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def demo(src: Tensor, dst: Tensor):
    with vf(mode='simt'):
        simt.dcci_entire_out()
        dst[0] = src[0]


@host
def run(src, dst):
    demo[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([17], device='npu', dtype=torch.int32)
    dst = torch.zeros(1, device='npu', dtype=torch.int32)
    run(src, dst)
    torch.npu.synchronize()
    assert dst.cpu().tolist() == [17]
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[17]
```
