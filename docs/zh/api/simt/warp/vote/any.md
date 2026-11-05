# simt.any

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

判断是否有活跃线程的输入不为0。

当Warp内所有活跃线程执行本接口后，对所有活跃线程的输入操作数x进行判断，所有活跃线程的x均为0，返回0，否则返回1。Warp内所有活跃线程返回相同的结果。

## 函数原型

```python
simt.any(x: bool | int32 | uint32 | int) -> int32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 当前线程参与投票的值，支持`bool`、`int32`或`uint32`标量，也接受可表示为`int32`的Python整数。布尔值直接表示条件是否成立；整数非零表示条件成立，零表示不成立。 |

## 返回值说明

当Warp内所有活跃线程的输入均为0，返回0，否则返回1。

## 约束说明

无

## 调用示例

在一个Warp中调用`simt.any`，校验结果并打印前8个线程的输出。

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
        dst[i] = simt.any(src[i])


@host
def run(src, dst):
    example_kernel[dim3(1), dim3(32)](src, dst)


def main():
    src = torch.arange(32, dtype=torch.int32, device='npu')
    dst = torch.empty((32,), dtype=torch.int32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    assert torch.equal(dst.cpu(), torch.full((32,), 1, dtype=torch.int32))
    print(dst.cpu().tolist()[:8])


if __name__ == "__main__":
    main()
```

预期结果：

```text
[1, 1, 1, 1, 1, 1, 1, 1]
```
