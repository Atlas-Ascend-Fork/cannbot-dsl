# simt.pow

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

计算以`lhs`为底、`rhs`为指数的幂。

## 函数原型

```python
simt.pow(lhs, rhs) -> float32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| lhs | 输入 | 底数，float32标量。 |
| rhs | 输入 | 指数，float32标量；数据类型与`lhs`相同。 |

## 返回值说明

返回计算结果，数据类型为`float32`，不修改输入。

| 输入情况 | 返回值 |
| --- | --- |
| `rhs`为`±0`或`lhs`为`1`，包括另一输入为`NaN` | `1` |
| 不属于上一行且含`NaN`的输入组合 | `NaN` |
| 底数为负的有限值，指数为非整数 | `NaN` |
| 底数为`-0`，指数为负奇整数 | `-inf` |
| 底数为零，指数为负数，且不属于上一行 | `+inf` |

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.pow`，校验并打印结果。

```python
import torch
import torch_npu
from cannbotdsl import simt, dim3
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def example_kernel(src: Tensor, dst: Tensor):
    with vf(mode='simt'):
        lhs = src[0]
        rhs = src[1]
        result = simt.pow(lhs, rhs)
        dst[0] = result


@host
def run(src: Tensor, dst: Tensor):
    example_kernel[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([2, 3], dtype=torch.float32, device='npu')
    dst = torch.empty((1,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([8], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=1e-05, atol=1e-06)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[8.0]
```
