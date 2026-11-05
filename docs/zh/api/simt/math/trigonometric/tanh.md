# simt.tanh

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

返回双曲正切函数值。

$$
r=\tanh(x)
$$

## 函数原型

```python
simt.tanh(x) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | `float16`、`bfloat16`、`float32`标量。 |

## 返回值说明

返回计算结果，数据类型与`x`相同。

| 输入情况 | 返回值 |
| --- | --- |
| `±0` | 同号零 |
| `±inf` | 同号的`1` |
| `NaN` | `NaN` |

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.tanh`，校验并打印结果。

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
        dst[0] = simt.tanh(src[0])


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([0.5], dtype=torch.float32, device='npu')
    dst = torch.empty((1,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([0.46211715726001], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0.0002, atol=2e-05)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[0.46211719512939453]
```
