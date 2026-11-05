# simt.lanemask_le

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

生成32位Lane位图：Lane编号小于或等于当前线程Lane编号的位置置1，其余位为0。第j位对应Lane j，不依赖该Lane是否活跃。

Warp是协同执行的一组线程，当前每个Warp包含32个线程；Lane表示线程在Warp中的位置，编号为0～31。

## 函数原型

```python
simt.lanemask_le() -> int32
```

## 参数说明

无

## 返回值说明

返回dtypes.int32标量表达式；读取32位位图时需按位模式解释，最高位置1的位图存入int32 Tensor后会显示为负数。

## 约束说明

- 只按Lane编号生成位图，不筛选当前分支的活跃线程；活跃集合需另外查询activemask。
- 位图不等于单个布尔值；完整0xffffffff位图存入int32 Tensor的数值为 -1。

## 调用示例

在一个Warp中调用`simt.lanemask_le`，校验结果并打印前8个线程的输出。

```python
import torch
import torch_npu
from cannbotdsl import dim3, host, simt
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def example_kernel(dst: Tensor):
    with vf(mode='simt'):
        i = simt.thread_idx()[0]
        dst[i] = simt.lanemask_le()


@host
def run(dst):
    example_kernel[dim3(1), dim3(32)](dst)


def main():
    dst = torch.empty((32,), dtype=torch.int32, device='npu')
    run(dst)
    torch.npu.synchronize()
    masks = [sum((1 << j for j in range(32) if j <= i)) for i in range(32)]
    expected = torch.tensor([m if m < 1 << 31 else m - (1 << 32) for m in masks], dtype=torch.int32)
    assert torch.equal(dst.cpu(), expected)
    print(dst.cpu().tolist()[:8])


if __name__ == "__main__":
    main()
```

预期结果：

```text
[1, 3, 7, 15, 31, 63, 127, 255]
```
