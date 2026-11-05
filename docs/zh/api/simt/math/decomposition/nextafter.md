# simt.nextafter

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

返回从lhs沿rhs方向的下一个float32可表示数；lhs=rhs时返回rhs。

## 函数原型

```python
simt.nextafter(lhs, rhs) -> float32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| lhs | 输入 | 起始值，`float32`标量。 |
| rhs | 输入 | 目标方向值，`float32`标量，与`lhs`的数据类型相同。 |

## 返回值说明

返回`float32`类型的浮点数。

任一输入为`NaN`时返回`NaN`。两输入相等时返回`rhs`，包括正负零组合；从最大有限值向同号无穷值步进时返回该无穷值，从无穷值向有限值步进时返回同号最大有限值。

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.nextafter`，校验并打印结果。

```python
import torch
import torch_npu
from cannbotdsl import simt, dtypes, dim3
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def compute(src: Tensor, dst: Tensor):
    with vf(mode='simt'):
        dst[0] = simt.nextafter(src[0], src[1])


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([1.0, 2.0], dtype=torch.float32, device='npu')
    dst = torch.empty((1,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([1.0000001192092896], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[1.0000001192092896]
```
