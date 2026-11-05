# simt.hadd

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

计算整数半加，向下取整，并避免中间求和溢出。

$$
r=\left\lfloor\frac{\mathrm{lhs}+\mathrm{rhs}}{2}\right\rfloor
$$

## 函数原型

```python
simt.hadd(lhs, rhs) -> int32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| lhs | 输入 | `int32`整数标量。 |
| rhs | 输入 | `int32`整数标量，与`lhs`的数据类型相同。 |

## 返回值说明

返回计算结果，数据类型为`int32`。

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.hadd`，校验并打印结果。

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
        dst[0] = simt.hadd(src[0], src[1])


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([2, 3], dtype=torch.int32, device='npu')
    dst = torch.empty((1,), dtype=torch.int32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([2], dtype=torch.int32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[2]
```
