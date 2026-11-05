# simt.sincospi

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

同时返回正弦值和余弦值。

$$
r=(\sin(\pi x),\cos(\pi x))
$$

## 函数原型

```python
simt.sincospi(x) -> tuple[float32, float32]
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 角度，`float32`标量。 |

## 返回值说明

返回`(sin_value, cos_value)`，依次为`sin(πx)`和`cos(πx)`，两项的数据类型均为`float32`。

| 输入情况 | 正弦值 | 余弦值 |
| --- | --- | --- |
| `±0` | 同号零 | `1` |
| `±inf`或`NaN` | `NaN` | `NaN` |

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.sincospi`，校验并打印结果。

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
        a, b = simt.sincospi(src[0])
        dst[0] = a
        dst[1] = b


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([0.0], dtype=torch.float32, device='npu')
    dst = torch.empty((2,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([0.0, 1.0], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0.0002, atol=2e-05)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[0.0, 1.0]
```
