# simt.vfloor

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对双分量短向量x的每个分量向负无穷舍入，取小于或等于x的最大整数值，两个分量互不影响。

## 函数原型

```python
simt.vfloor(x) -> dtypes.float16x2 | dtypes.bfloat16x2
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 输入短向量，类型为`float16x2`（包含两个`float16`分量）或`bfloat16x2`（包含两个`bfloat16`分量）。 |

## 返回值说明

返回与x相同dtype的短向量，分量为取整后的浮点数；不会返回整数向量。通过`.x`、`.y`分别访问两个结果分量。

下表按对应分量分别说明结果。

| 输入分量或结果情况 | 返回分量 |
| --- | --- |
| `NaN` | `NaN` |
| `±inf` | 保留原值 |
| 取整结果为零 | `+0` |

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.vfloor`，校验并打印结果。

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
        x0 = simt.make_short_vector(src[0], src[1])
        y0 = simt.vfloor(x0)
        dst[0] = y0.x
        dst[1] = y0.y
        x1 = simt.make_short_vector(src[2], src[3])
        y1 = simt.vfloor(x1)
        dst[2] = y1.x
        dst[3] = y1.y
        x2 = simt.make_short_vector(src[4], src[5])
        y2 = simt.vfloor(x2)
        dst[4] = y2.x
        dst[5] = y2.y


@host
def run(src, dst):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([2.5, 3.5, -2.5, -3.5, 1.25, -1.25], dtype=torch.float16, device='npu')
    dst = torch.empty((6,), dtype=torch.float16, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([2, 3, -3, -4, 1, -2], dtype=torch.float16)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[2.0, 3.0, -3.0, -4.0, 1.0, -2.0]
```
