# simt.ldexp

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

按指定的二进制指数缩放`x`。

$$
r=x\,2^{\mathrm{exp}}
$$

## 函数原型

```python
simt.ldexp(x, exp) -> tuple[float32, int32]
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | `float32`标量。 |
| exp | 输入 | 以2为底的缩放指数，int32；不是自然指数。 |

## 返回值说明

返回`(result, exp)`，分别为缩放结果和输入指数的副本，数据类型依次为`float32`、`int32`。

| 输入或结果情况 | 缩放结果 |
| --- | --- |
| `x`为`±0`或`±inf` | 保留`x`的原值 |
| `x`为`NaN` | `NaN` |
| 缩放结果超出`float32`范围 | 可能溢出 |

## 约束说明

- `exp`必须是显式的`int32`标量，不能直接使用未指定类型的Python整数字面量。

## 调用示例

使用一组固定输入调用`simt.ldexp`，校验并打印结果。

```python
import torch
import torch_npu
from cannbotdsl import simt, dtypes, dim3
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def compute(src: Tensor, dst: Tensor, orders: Tensor):
    with vf(mode='simt'):
        a, b = simt.ldexp(src[0], orders[0])
        dst[0] = a
        dst[1] = simt.cast(b, dtype=dtypes.float32)


@host
def run(src: Tensor, dst: Tensor, orders: Tensor):
    compute[dim3(1), dim3(1)](src, dst, orders)


def main():
    src = torch.tensor([1.5], dtype=torch.float32, device='npu')
    dst = torch.empty((2,), dtype=torch.float32, device='npu')
    orders = torch.tensor([3], dtype=torch.int32, device='npu')
    run(src, dst, orders)
    torch.npu.synchronize()
    expected = torch.tensor([12.0, 3], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0.0002, atol=2e-05)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[12.0, 3.0]
```
