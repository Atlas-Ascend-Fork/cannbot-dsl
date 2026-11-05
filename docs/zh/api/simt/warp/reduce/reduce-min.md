# simt.reduce_min

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对Warp内所有活跃线程输入value求最小值。Warp内所有活跃线程返回相同的结果。

## 函数原型

```python
simt.reduce_min(value: int32 | float16 | float32) -> int32 | float16 | float32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | 源操作数。 |

## 返回值说明

Warp内所有活跃线程输入value的最小值。

## 约束说明

无

## 调用示例

在一个Warp中调用`simt.reduce_min`，校验结果并打印前8个线程的输出。

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
        dst[i] = simt.reduce_min(src[i])


@host
def run(src, dst):
    example_kernel[dim3(1), dim3(32)](src, dst)


def main():
    src = torch.arange(32, dtype=torch.int32, device='npu')
    dst = torch.empty((32,), dtype=torch.int32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    assert torch.equal(dst.cpu(), torch.full((32,), 0, dtype=torch.int32))
    print(dst.cpu().tolist()[:8])


if __name__ == "__main__":
    main()
```

预期结果：

```text
[0, 0, 0, 0, 0, 0, 0, 0]
```
