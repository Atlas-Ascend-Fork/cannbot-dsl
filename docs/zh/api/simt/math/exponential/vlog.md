# simt.vlog

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

逐分量计算输入的自然对数。

## 函数原型

```python
simt.vlog(x) -> dtypes.float16x2 | dtypes.bfloat16x2
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 输入短向量，类型为`float16x2`（包含两个`float16`分量）或`bfloat16x2`（包含两个`bfloat16`分量）。 |

## 返回值说明

返回计算得到的双分量短向量，可通过`.x`、`.y`分别访问，数据类型与`x`相同；不修改输入。

下表按对应分量分别说明结果。

| 输入分量情况 | 返回分量 |
| --- | --- |
| `±0` | `-inf` |
| `+inf` | `+inf` |
| 负数（包括`-inf`） | `NaN` |
| `NaN` | `NaN` |

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.vlog`，校验并打印结果。

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
        x = simt.make_short_vector(src[0], src[1])
        result = simt.vlog(x)
        dst[0] = result.x
        dst[1] = result.y


@host
def run(src: Tensor, dst: Tensor):
    example_kernel[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([2, 4], dtype=torch.float16, device='npu')
    dst = torch.empty((2,), dtype=torch.float16, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([0.6931471805599453, 1.3862943611198906], dtype=torch.float16)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0.003, atol=0.003)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[0.693359375, 1.38671875]
```
