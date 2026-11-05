# simt.activemask

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

查看Warp内所有线程是否为活跃状态。

返回一个32bit的位图，若Warp内某个线程是活跃（已结束线程是非活跃状态）的，则返回值中与线程LaneId对应的bit位为1，否则为0。Warp内所有活跃线程返回相同的结果。

## 函数原型

```python
simt.activemask() -> int32
```

## 参数说明

无

## 返回值说明

32bit的位图：若Warp内某个线程是活跃的，则返回值中与线程LaneId对应的bit位为1，否则为0。

读取32位位图时需按位模式解释，最高位置1的位图存入int32 Tensor后会显示为负数。

## 约束说明

无

## 调用示例

在一个Warp中调用`simt.activemask`，校验结果并打印前8个线程的输出。

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
        dst[i] = simt.activemask()


@host
def run(dst):
    example_kernel[dim3(1), dim3(32)](dst)


def main():
    dst = torch.empty((32,), dtype=torch.int32, device='npu')
    run(dst)
    torch.npu.synchronize()
    assert torch.equal(dst.cpu(), torch.full((32,), -1, dtype=torch.int32))
    print(dst.cpu().tolist()[:8])


if __name__ == "__main__":
    main()
```

预期结果：

```text
[-1, -1, -1, -1, -1, -1, -1, -1]
```
