# simt.fma

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将前两个输入相乘后加上第三个输入，中间乘积不单独舍入，最终结果只舍入一次。

$$
r=x\times y+z
$$

`float16`、`bfloat16`的计算结果遵循`CAST_RINT`模式舍入。

## 函数原型

```python
simt.fma(x, y, z) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 左乘数，float16、bfloat16或float32标量。 |
| y | 输入 | 右乘数，float16、bfloat16或float32标量；数据类型与`x`相同。 |
| z | 输入 | 加数，float16、bfloat16或float32标量；数据类型与`x`相同。 |

## 返回值说明

返回计算结果，数据类型与`x`相同，不修改输入。

| 输入情况 | 返回值 |
| --- | --- |
| 任一输入为`NaN` | `NaN` |
| 乘数为零与无穷值的组合 | `NaN` |
| 乘积为无穷值，且加数为异号无穷值 | `NaN` |
| 乘积为无穷值，加数为有限值或同号无穷值 | 与乘积同号的无穷值 |
| 有限乘积与无穷值相加 | 与加数同号的无穷值 |

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.fma`，校验并打印结果。

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
        x = src[0]
        y = src[1]
        z = src[2]
        result = simt.fma(x, y, z)
        dst[0] = result


@host
def run(src: Tensor, dst: Tensor):
    example_kernel[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([2, 3, 1], dtype=torch.float32, device='npu')
    dst = torch.empty((1,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([7], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=1e-05, atol=1e-06)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[7.0]
```
