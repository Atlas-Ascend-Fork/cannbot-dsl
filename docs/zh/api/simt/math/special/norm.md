# simt.norm

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

读取`ptr`起始的`n`个元素，计算这些元素的欧氏范数。

$$
r=\sqrt{\sum_{i=0}^{n-1}\mathrm{ptr}[i]^2}
$$

## 函数原型

```python
simt.norm(n, ptr=None) -> float32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| n | 输入 | 元素个数，类型为`int32`。 |
| ptr | 输入 | 通过`tensor.ptr(...)`获取的`float32`指针；需显式提供。 |

## 返回值说明

返回`float32`类型的浮点数。

| 输入情况 | 返回值 |
| --- | --- |
| 所有输入分量均为零 | `+0` |
| 任一输入分量为无穷值，即使其他分量为`NaN` | `+inf` |
| 没有无穷值，但含`NaN` | `NaN` |

## 约束说明

- `n`必须为正数；`ptr`指向连续且至少包含`n`个`float32`元素的可读存储，所读元素必须已完成写入。
- 当数组元素全部为非正规数时，结果可能为`NaN`。

## 调用示例

使用一组固定输入调用`simt.norm`，校验并打印结果。

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
        dst[0] = simt.norm(2, src.ptr((0,)))


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([3.0, 4.0], dtype=torch.float32, device='npu')
    dst = torch.empty((1,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([5.0], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0.0002, atol=2e-05)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[5.0]
```
