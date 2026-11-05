# simt.vbeq

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对双分量短向量x、y的对应分量判断`x[i] == y[i]`。将两项条件取逻辑与。

## 函数原型

```python
simt.vbeq(x, y) -> bool
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 左侧短向量，类型为`float16x2`（包含两个`float16`分量）或`bfloat16x2`（包含两个`bfloat16`分量）。 |
| y | 输入 | 右侧短向量，数据类型必须与`x`相同。 |

## 返回值说明

返回布尔值，两个对应分量的比较条件都成立时为`True`，否则为`False`。

正零与负零比较相等。

任一输入分量为`NaN`时，返回`False`。

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.vbeq`，将布尔结果转换为整数后校验并打印：1表示真，0表示假。

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
        z0 = simt.vbeq(x0, y0)
        dst[0] = simt.cast(z0, dtype=dtypes.int32)
        x1 = simt.make_short_vector(src[4], src[5])
        y1 = simt.make_short_vector(src[6], src[7])
        z1 = simt.vbeq(x1, y1)
        dst[1] = simt.cast(z1, dtype=dtypes.int32)


@host
def run(src, dst):
    compute[dim3(1), dim3(1)](src, dst)
def main():
    src = torch.tensor([1, 3, 1, 2, 3, 1, 2, 1], dtype=torch.float16, device='npu')
    dst = torch.empty((2,), dtype=torch.int32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([False, False], dtype=torch.int32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[0, 0]
```
