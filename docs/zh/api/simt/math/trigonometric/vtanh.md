# simt.vtanh

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对双分量短向量`x`的每个分量独立计算双曲正切函数值。

$$
r_i=\tanh(x_i),\quad i\in\{0,1\}
$$

## 函数原型

```python
simt.vtanh(x) -> dtypes.float16x2 | dtypes.bfloat16x2
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 输入短向量，类型为`float16x2`（包含两个`float16`分量）或`bfloat16x2`（包含两个`bfloat16`分量）。 |

## 返回值说明

返回与x同类型的短向量。通过`.x`、`.y`分别访问两个结果分量。

下表按对应分量分别说明结果。

| 输入分量情况 | 返回分量 |
| --- | --- |
| `±0` | 同号零 |
| `±inf` | 同号的`1` |
| `NaN` | `NaN` |

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.vtanh`，校验并打印结果。

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
        x = simt.make_short_vector(src[0], src[1])
        y = simt.vtanh(x)
        dst[0] = y.x
        dst[1] = y.y


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([0.5, 1.0], dtype=torch.float16, device='npu')
    dst = torch.empty((2,), dtype=torch.float16, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([0.46211716, 0.76159416], dtype=torch.float16)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0.003, atol=0.003)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[0.462158203125, 0.76171875]
```
