# simt.frexp

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将有限非零的`x`分解为尾数`m`和二进制指数`e`，返回`(m, e)`。

$$
x=m\,2^e,\quad 0.5\le |m|<1
$$

## 函数原型

```python
simt.frexp(x) -> tuple[float32, int32]
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | `float32`标量。 |

## 返回值说明

返回`(fraction, exponent)`，分别为尾数和二进制指数，数据类型依次为`float32`、`int32`。

| 输入情况 | 尾数 | 二进制指数 |
| --- | --- | --- |
| `±0` | 同号零 | `0` |
| `±inf`或`NaN` | 保留原特殊值 | 不具有通常的分解含义 |

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.frexp`，校验并打印结果。

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
        a, b = simt.frexp(src[0])
        dst[0] = a
        dst[1] = simt.cast(b, dtype=dtypes.float32)


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([8.0], dtype=torch.float32, device='npu')
    dst = torch.empty((2,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([0.5, 4], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0.0002, atol=2e-05)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[0.5, 4.0]
```
