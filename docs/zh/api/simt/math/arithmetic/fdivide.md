# simt.fdivide

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

计算被除数除以除数的浮点商。

## 函数原型

```python
simt.fdivide(lhs, rhs) -> float32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| lhs | 输入 | 被除数，float32标量。 |
| rhs | 输入 | 除数，float32标量；数据类型与`lhs`相同。 |

## 返回值说明

返回计算结果，数据类型为`float32`，不修改输入。

| 输入情况 | 返回值 |
| --- | --- |
| 零除以零，或无穷值除以无穷值 | `NaN` |
| 任一输入为`NaN` | `NaN` |
| 非零有限值除以零 | 无穷值 |
| 无穷值除以有限值 | 无穷值 |
| 有限值除以无穷值 | 零 |

非`NaN`结果在两输入同号时为正、异号时为负。

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.fdivide`，校验并打印结果。

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
        result = simt.fdivide(lhs, rhs)
        dst[0] = result


@host
def run(src: Tensor, dst: Tensor):
    example_kernel[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([6, 2], dtype=torch.float32, device='npu')
    dst = torch.empty((1,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([3], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=1e-05, atol=1e-06)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[3.0]
```
