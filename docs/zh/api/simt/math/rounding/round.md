# simt.round

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对x舍入到最近整数，恰好处于中点时向远离零的方向取整。取整后仍返回浮点值，不执行整数类型转换。

## 函数原型

```python
simt.round(x) -> float32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 待舍入的`float32`标量。 |

## 返回值说明

返回计算结果，数据类型为`float32`。

| 输入或结果情况 | 返回值 |
| --- | --- |
| `NaN` | `NaN` |
| `±inf` | 保留原值 |
| 取整结果为零 | `+0` |

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.round`，校验并打印结果。

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
        dst[0] = simt.round(src[0])
        dst[1] = simt.round(src[1])
        dst[2] = simt.round(src[2])
        dst[3] = simt.round(src[3])
        dst[4] = simt.round(src[4])
        dst[5] = simt.round(src[5])


@host
def run(src, dst):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([2.5, 3.5, -2.5, -3.5, 1.25, -1.25], dtype=torch.float32, device='npu')
    dst = torch.empty((6,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([3, 4, -3, -4, 1, -1], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[3.0, 4.0, -3.0, -4.0, 1.0, -1.0]
```
