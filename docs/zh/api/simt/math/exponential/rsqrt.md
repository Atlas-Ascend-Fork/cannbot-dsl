# simt.rsqrt

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

计算输入的平方根的倒数。

## 函数原型

```python
simt.rsqrt(x) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 输入值，float16、bfloat16或float32标量。 |

## 返回值说明

返回计算结果，数据类型与`x`相同，不修改输入。

| 输入情况 | 返回值 |
| --- | --- |
| `+0` | `+inf` |
| `-0` | `-inf` |
| `+inf` | `+0` |
| 负数（包括`-inf`） | `NaN` |
| `NaN` | `NaN` |

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.rsqrt`，校验并打印结果。

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
        x = src[0]
        result = simt.rsqrt(x)
        dst[0] = result


@host
def run(src: Tensor, dst: Tensor):
    example_kernel[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([4], dtype=torch.float32, device='npu')
    dst = torch.empty((1,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([0.5], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=1e-05, atol=1e-06)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[0.5]
```
