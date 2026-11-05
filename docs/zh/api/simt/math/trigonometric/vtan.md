# simt.vtan

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对双分量短向量`x`的每个分量独立计算正切函数值。

$$
r_i=\tan(x_i),\quad i\in\{0,1\}
$$

## 函数原型

```python
simt.vtan(x) -> dtypes.float16x2 | dtypes.bfloat16x2
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 输入短向量，类型为`float16x2`（包含两个`float16`分量）或`bfloat16x2`（包含两个`bfloat16`分量）。 |

## 返回值说明

返回与x同类型的短向量。通过`.x`、`.y`分别访问两个结果分量。

按`float32`精度计算各分量的正切值，返回结果的数据类型与输入相同。

| 输入分量情况 | 返回分量 |
| --- | --- |
| `±0` | 同号零 |
| `±inf` | `NaN` |
| `NaN` | `NaN` |

## 约束说明

- 避开 π/2+kπ 附近的极点。

## 调用示例

将两个输入值组成短向量，逐分量计算正切。

```python
import torch
import torch_npu
from cannbotdsl import simt, dim3
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def compute(src: Tensor, dst: Tensor):
    with vf(mode='simt'):
        v = simt.make_short_vector(src[0], src[1])
        r = simt.vtan(v)
        dst[0] = r.x
        dst[1] = r.y


@host
def run(src, dst):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([0.0, 0.5], dtype=torch.float16, device='npu')
    dst = torch.empty(2, dtype=torch.float16, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    torch.testing.assert_close(dst.cpu(), torch.tan(src.cpu()), rtol=1e-3, atol=1e-3)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[0.0, 0.54638671875]
```
