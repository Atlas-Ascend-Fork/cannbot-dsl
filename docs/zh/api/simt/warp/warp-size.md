# simt.warp_size

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

获取一个线程束（Warp）包含的线程数量，当前为32。Warp是协同执行的一组线程。

## 函数原型

```python
simt.warp_size() -> int32
```

## 参数说明

无

## 返回值说明

返回dtypes.int32标量表达式，当前值为32。

## 约束说明

- 该值不是线程块总线程数，也不是当前活跃线程数，不能代替activemask。

## 调用示例

读取并打印一个Warp中的线程数量。

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
        dst[i] = simt.warp_size()


@host
def run(dst):
    example_kernel[dim3(1), dim3(32)](dst)


def main():
    dst = torch.empty((32,), dtype=torch.int32, device='npu')
    run(dst)
    torch.npu.synchronize()
    assert torch.equal(dst.cpu(), torch.full((32,), 32, dtype=torch.int32))
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
```
