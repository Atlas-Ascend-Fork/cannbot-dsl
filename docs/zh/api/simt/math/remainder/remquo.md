# simt.remquo

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

返回余数和商的低位信息`(r, qbits)`；`q`为`x / y`舍入到最近整数所得的值，恰好处于中点时取偶数，`qbits`包含商的符号及低位信息。

$$
r=x-qy,\quad q=\operatorname{rint}\left(\frac{x}{y}\right)
$$

## 函数原型

```python
simt.remquo(x, y) -> tuple[float32, int32]
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 被除数，`float32`标量。 |
| y | 输入 | 除数，`float32`标量，与`x`的数据类型相同。 |

## 返回值说明

返回`(remainder, quotient_bits)`，分别为余数和带符号的商低位信息，数据类型依次为`float32`、`int32`。

| 输入或结果情况 | 余数 |
| --- | --- |
| `x`为有限值，`y`为无穷值 | `x` |
| 余数为零 | 与`x`同号的零 |
| `x`为无穷值，或`y`为零，或任一输入为`NaN` | `NaN`，此时不应使用商的低位信息 |

## 约束说明

- 第二个返回值只包含商的符号及低位信息，不保证是完整商。

## 调用示例

使用一组固定输入调用`simt.remquo`，校验并打印结果。

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
        a, b = simt.remquo(src[0], src[1])
        dst[0] = a
        dst[1] = simt.cast(b, dtype=dtypes.float32)


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([3.0, 2.0], dtype=torch.float32, device='npu')
    dst = torch.empty((2,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([-1.0, 2], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0.0002, atol=2e-05)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[-1.0, 2.0]
```
