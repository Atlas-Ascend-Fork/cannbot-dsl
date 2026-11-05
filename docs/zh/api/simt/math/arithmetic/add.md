# simt.add

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

计算两个输入标量之和。

## 函数原型

```python
simt.add(lhs, rhs, rounding=simt.RoundingMode.RN) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| lhs | 输入 | 左加数，类型为`float16`、`bfloat16`或`float32`。 |
| rhs | 输入 | 右加数，数据类型必须与`lhs`相同。 |
| rounding | 输入 | 舍入模式，类型为 [RoundingMode](../../conversion/roundingmode.md) 或名称字符串，默认`simt.RoundingMode.RN`。 |

## 返回值说明

返回两个输入的和，数据类型与输入相同。不修改输入。

| 输入情况 | 返回值 |
| --- | --- |
| 有限值与无穷值相加 | 与无穷值同号的无穷值 |
| 两个同号无穷值相加 | 同号无穷值 |
| 两个异号无穷值相加 | `NaN` |
| 任一输入为`NaN` | `NaN` |
| `+0`与`+0`相加 | `+0` |
| `-0`与`-0`相加 | `-0` |
| `+0`与`-0`相加 | `+0` |
| 两个非零有限值互为相反数 | `+0` |

## 约束说明

- `float16`和`bfloat16`遵循`CAST_RINT`模式舍入；对于所有支持的数据类型，`rounding`均不能改变实际舍入行为。

## 调用示例

使用一组固定输入调用`simt.add`，校验并打印结果。

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
        result = simt.add(lhs, rhs)
        dst[0] = result


@host
def run(src: Tensor, dst: Tensor):
    example_kernel[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([2, 3], dtype=torch.float32, device='npu')
    dst = torch.empty((1,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([5], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=1e-05, atol=1e-06)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[5.0]
```
