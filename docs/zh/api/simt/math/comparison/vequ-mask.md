# simt.vequ_mask

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对双分量短向量x、y的对应分量判断`x[i] == y[i]`。将两个分量的比较结果合并为一个整数掩码。

## 函数原型

```python
simt.vequ_mask(x, y) -> int32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 左侧短向量，类型为`float16x2`（包含两个`float16`分量）或`bfloat16x2`（包含两个`bfloat16`分量）。 |
| y | 输入 | 右侧短向量，数据类型必须与`x`相同。 |

## 返回值说明

返回`int32`类型的整数掩码，表示两个对应分量的比较结果，取值如下表。

| `.x`分量的比较结果 | `.y`分量的比较结果 | 返回值 |
| --- | --- | ---: |
| `False` | `False` | `0` |
| `True` | `False` | `65535` |
| `False` | `True` | `-65536` |
| `True` | `True` | `-1` |

正零与负零比较相等。

对应输入分量中任一个为`NaN`时，该组分量的比较结果为`True`。

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.vequ_mask`，校验并打印结果。

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
        y0 = simt.make_short_vector(src[2], src[3])
        z0 = simt.vequ_mask(x0, y0)
        dst[0] = z0
        x1 = simt.make_short_vector(src[4], src[5])
        y1 = simt.make_short_vector(src[6], src[7])
        z1 = simt.vequ_mask(x1, y1)
        dst[1] = z1


@host
def run(src, dst):
    compute[dim3(1), dim3(1)](src, dst)
def main():
    src = torch.tensor([1, 3, 1, 2, 3, 1, 2, 1], dtype=torch.float16, device='npu')
    dst = torch.empty((2,), dtype=torch.int32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    actual = dst.cpu().tolist()
    assert actual == [65535, -65536], actual
    print(actual)


if __name__ == "__main__":
    main()
```

预期结果：

```text
[65535, -65536]
```
