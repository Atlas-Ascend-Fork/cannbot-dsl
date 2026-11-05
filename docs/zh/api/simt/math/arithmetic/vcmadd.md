# simt.vcmadd

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将双分量视为复数执行乘加，第一分量表示实部，第二分量表示虚部。

$$
\begin{aligned}
r_0&=x_0y_0-x_1y_1+z_0,\\
r_1&=x_0y_1+x_1y_0+z_1.
\end{aligned}
$$

实部和虚部采用分步融合乘加，每一步遵循`CAST_RINT`模式舍入。

## 函数原型

```python
simt.vcmadd(x, y, z) -> dtypes.float16x2 | dtypes.bfloat16x2
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 左乘复数短向量，类型为`float16x2`（包含两个`float16`分量）或`bfloat16x2`（包含两个`bfloat16`分量）。 |
| y | 输入 | 右乘复数短向量，数据类型必须与`x`相同。 |
| z | 输入 | 相加的复数短向量，数据类型必须与`x`相同。 |

## 返回值说明

返回计算得到的双分量短向量，可通过`.x`、`.y`分别访问，数据类型与`x`相同；不修改输入。

计算包含实部与虚部的交叉乘积；结果可能与高精度计算后只舍入一次的结果不同。

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.vcmadd`，校验并打印结果。

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
        y = simt.make_short_vector(src[2], src[3])
        z = simt.make_short_vector(src[4], src[5])
        result = simt.vcmadd(x, y, z)
        dst[0] = result.x
        dst[1] = result.y


@host
def run(src: Tensor, dst: Tensor):
    example_kernel[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float16, device='npu')
    dst = torch.empty((2,), dtype=torch.float16, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([0, 16], dtype=torch.float16)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0.003, atol=0.003)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[0.0, 16.0]
```
