# simt.lane_idx

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

获取当前线程在所属线程束（Warp）中的Lane编号。每个Warp包含32个线程，Lane表示线程在Warp中的位置，编号为0～31，不同Warp中的编号会重复。

## 函数原型

```python
simt.lane_idx() -> int32
```

## 参数说明

无

## 返回值说明

返回dtypes.int32标量表达式，表示Warp内的Lane编号。

## 约束说明

- Lane编号不是线程块全局线程索引，分配输出位置时仍应使用thread_idx或网格线性索引。

## 调用示例

在两个Warp中调用`simt.lane_idx`，校验结果并打印前8个线程的输出。

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
        dst[i] = simt.lane_idx()


@host
def run(dst):
    example_kernel[dim3(1), dim3(64)](dst)


def main():
    dst = torch.empty((64,), dtype=torch.int32, device='npu')
    run(dst)
    torch.npu.synchronize()
    expected = torch.tensor([i % 32 for i in range(64)], dtype=torch.int32)
    assert torch.equal(dst.cpu(), expected)
    print(dst.cpu().tolist()[:8])


if __name__ == "__main__":
    main()
```

预期结果：

```text
[0, 1, 2, 3, 4, 5, 6, 7]
```
