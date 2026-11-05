# simt.fma_relu

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

执行融合乘加，并将负结果截断为零。

$$
r=\max(x\times y+z,0)
$$

`float16`、`bfloat16`的计算结果遵循`CAST_RINT`模式舍入。

## 函数原型

```python
simt.fma_relu(x, y, z) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 左乘数，float16或bfloat16标量。 |
| y | 输入 | 右乘数，float16或bfloat16标量；数据类型与`x`相同。 |
| z | 输入 | 加数，float16或bfloat16标量；数据类型与`x`相同。 |

## 返回值说明

返回计算结果，数据类型与`x`相同，不修改输入。

| 输入情况 | 返回值 |
| --- | --- |
| 任一输入为`NaN` | `NaN` |
| 乘数为零与无穷值的组合 | `NaN` |
| 无穷乘积与异号无穷加数相加 | `NaN` |
| 融合乘加结果为正无穷 | `+inf` |
| 融合乘加结果为负数（包括负无穷） | `0` |

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.fma_relu`，校验并打印结果。

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
        result = simt.fma_relu(x, y, z)
        dst[0] = result


@host
def run(src: Tensor, dst: Tensor):
    example_kernel[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([-2, 3, 1], dtype=torch.float16, device='npu')
    dst = torch.empty((1,), dtype=torch.float16, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([0], dtype=torch.float16)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0.003, atol=0.003)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[0.0]
```
