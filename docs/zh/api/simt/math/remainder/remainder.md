# simt.remainder

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将商舍入到最近整数后计算余数；恰好处于中点时取偶数，与向零截断的`fmod`不同。

$$
r=\mathrm{lhs}-q\,\mathrm{rhs},\quad q=\operatorname{rint}\left(\frac{\mathrm{lhs}}{\mathrm{rhs}}\right)
$$

## 函数原型

```python
simt.remainder(lhs, rhs) -> float32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| lhs | 输入 | 被除数，`float32`标量。 |
| rhs | 输入 | 除数，`float32`标量，与`lhs`的数据类型相同。 |

## 返回值说明

返回`float32`类型的浮点数。

| 输入情况 | 返回值 |
| --- | --- |
| `lhs`为有限值，`rhs`为无穷值 | `lhs` |
| 余数为零 | 与`lhs`同号的零 |
| `lhs`为无穷值，或`rhs`为零 | `NaN` |
| 任一输入为`NaN` | `NaN` |

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.remainder`，校验并打印结果。

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
        dst[0] = simt.remainder(src[0], src[1])


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([7.0, 2.0], dtype=torch.float32, device='npu')
    dst = torch.empty((1,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([-1.0], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0.0002, atol=2e-05)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[-1.0]
```
