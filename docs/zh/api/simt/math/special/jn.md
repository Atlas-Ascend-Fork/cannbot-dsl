# simt.jn

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

计算`n`阶第一类贝塞尔函数。

$$
r=J_n(x)
$$

## 函数原型

```python
simt.jn(n, x) -> float32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| n | 输入 | 贝塞尔阶数；类型为`int32`。 |
| x | 输入 | float32自变量。 |

## 返回值说明

返回`float32`类型的浮点数。

## 约束说明

- `n`的范围为0≤n≤128。
- 每个逻辑块的线程数不超过256。

## 调用示例

传入整数阶数，计算一阶贝塞尔函数值。

```python
import torch
import torch_npu
from cannbotdsl import simt, dim3
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def compute(src: Tensor, dst: Tensor, orders: Tensor):
    with vf(mode='simt'):
        dst[0] = simt.jn(orders[0], src[0])


@host
def run(src: Tensor, dst: Tensor, orders: Tensor):
    compute[dim3(1), dim3(1)](src, dst, orders)


def main():
    src = torch.tensor([1.0], dtype=torch.float32, device='npu')
    dst = torch.empty((1,), dtype=torch.float32, device='npu')
    orders = torch.tensor([1], dtype=torch.int32, device='npu')
    run(src, dst, orders)
    torch.npu.synchronize()
    torch.testing.assert_close(dst.cpu(), torch.special.bessel_j1(src.cpu()), rtol=1e-4, atol=1e-5)
    print([round(v, 6) for v in dst.cpu().tolist()])


if __name__ == "__main__":
    main()
```

预期结果：

```text
[0.440051]
```
