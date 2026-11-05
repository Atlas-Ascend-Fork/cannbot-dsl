# simt.ceil

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对x向正无穷舍入，取大于或等于x的最小整数值。取整后仍返回浮点值，不执行整数类型转换。

## 函数原型

```python
simt.ceil(x) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 待舍入的浮点标量。输入可为float16、bfloat16、float32标量。 |

## 返回值说明

返回计算结果，数据类型与输入相同。

| 输入或结果情况 | 返回值 |
| --- | --- |
| `NaN` | `NaN` |
| `±inf` | 保留原值 |
| 取整结果为零 | `+0` |

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.ceil`，校验并打印结果。

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
        dst[0] = simt.ceil(src[0])
        dst[1] = simt.ceil(src[1])
        dst[2] = simt.ceil(src[2])
        dst[3] = simt.ceil(src[3])
        dst[4] = simt.ceil(src[4])
        dst[5] = simt.ceil(src[5])


@host
def run(src, dst):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([2.5, 3.5, -2.5, -3.5, 1.25, -1.25], dtype=torch.float32, device='npu')
    dst = torch.empty((6,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([3, 4, -2, -3, 2, -1], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[3.0, 4.0, -2.0, -3.0, 2.0, -1.0]
```
