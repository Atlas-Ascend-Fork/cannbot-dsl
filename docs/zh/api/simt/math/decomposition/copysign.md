# simt.copysign

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

返回大小取自`lhs`、符号位取自`rhs`的数值，包含`rhs`的负零符号。

$$
|r|=|\mathrm{lhs}|,\quad \operatorname{signbit}(r)=\operatorname{signbit}(\mathrm{rhs})
$$

## 函数原型

```python
simt.copysign(lhs, rhs) -> float32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| lhs | 输入 | 提供绝对值大小的输入，`float32`标量。 |
| rhs | 输入 | 提供符号位的输入，`float32`标量，与`lhs`的数据类型相同。 |

## 返回值说明

返回`float32`类型的浮点数。

`lhs`为零或无穷值时，保留其大小并采用`rhs`的符号位；`lhs`为`NaN`时结果仍为`NaN`。

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.copysign`，校验并打印结果。

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
        dst[0] = simt.copysign(src[0], src[1])


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([3.0, -1.0], dtype=torch.float32, device='npu')
    dst = torch.empty((1,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([-3.0], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0.0002, atol=2e-05)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[-3.0]
```
