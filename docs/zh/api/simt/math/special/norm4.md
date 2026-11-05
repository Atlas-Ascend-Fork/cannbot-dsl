# simt.norm4

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

计算四个输入分量的欧氏范数。

$$
r=\sqrt{a^2+b^2+c^2+d^2}
$$

## 函数原型

```python
simt.norm4(a, b, c, d) -> float32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| a | 输入 | 第1维分量，`float32`标量。 |
| b | 输入 | 第2维分量，`float32`标量，与`a`的数据类型相同。 |
| c | 输入 | 第3维分量，`float32`标量，与`a`的数据类型相同。 |
| d | 输入 | 第4维分量，`float32`标量，与`a`的数据类型相同。 |

## 返回值说明

返回`float32`类型的浮点数。

| 输入情况 | 返回值 |
| --- | --- |
| 所有输入分量均为零 | `+0` |
| 任一输入分量为无穷值，即使其他分量为`NaN` | `+inf` |
| 没有无穷值，但含`NaN` | `NaN` |

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.norm4`，校验并打印结果。

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
        dst[0] = simt.norm4(src[0], src[1], src[2], src[3])


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([3.0, 4.0, 0.0, 0.0], dtype=torch.float32, device='npu')
    dst = torch.empty((1,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([5.0], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0.0002, atol=2e-05)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[5.0]
```
