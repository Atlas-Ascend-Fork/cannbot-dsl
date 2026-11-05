# simt.scalbln

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

按指定的二进制指数缩放输入。

$$
r=\mathrm{lhs}\,2^{\mathrm{rhs}}
$$

## 函数原型

```python
simt.scalbln(lhs, rhs) -> float32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| lhs | 输入 | float32被缩放值。 |
| rhs | 输入 | int64的二进制指数。 |

## 返回值说明

返回`float32`类型的浮点数。

## 约束说明

无

## 调用示例

将1.5乘以2的3次方，得到12。

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
        dst[0] = simt.scalbln(src[0], orders[0])


@host
def run(src: Tensor, dst: Tensor, orders: Tensor):
    compute[dim3(1), dim3(1)](src, dst, orders)


def main():
    src = torch.tensor([1.5], dtype=torch.float32, device='npu')
    dst = torch.empty((1,), dtype=torch.float32, device='npu')
    orders = torch.tensor([3], dtype=torch.int64, device='npu')
    run(src, dst, orders)
    torch.npu.synchronize()
    torch.testing.assert_close(dst.cpu(), torch.tensor([12.0]), rtol=1e-4, atol=1e-5)
    print([round(v, 6) for v in dst.cpu().tolist()])


if __name__ == "__main__":
    main()
```

预期结果：

```text
[12.0]
```
