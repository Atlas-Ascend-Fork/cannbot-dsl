# simt.sad

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

计算`x`与`y`之差的绝对值，再加上`z`。

$$
r=|x-y|+z
$$

## 函数原型

```python
simt.sad(x, y, z) -> int32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | `int32`整数标量。 |
| y | 输入 | `int32`整数标量，与`x`的数据类型相同。 |
| z | 输入 | `int32`整数标量，与`x`的数据类型相同。 |

## 返回值说明

返回计算结果，数据类型为`int32`。

## 约束说明

- 差值的绝对值与累加结果均应在`int32`可表示范围内。

## 调用示例

使用一组固定输入调用`simt.sad`，校验并打印结果。

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
        dst[0] = simt.sad(src[0], src[1], src[2])


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([7, 2, 3], dtype=torch.int32, device='npu')
    dst = torch.empty((1,), dtype=torch.int32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([8], dtype=torch.int32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[8]
```
