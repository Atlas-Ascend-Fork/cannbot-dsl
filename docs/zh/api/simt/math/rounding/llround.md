# simt.llround

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对x舍入到最近整数，中点向远离零的方向取整。

## 函数原型

```python
simt.llround(x) -> int64
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 待舍入的`float32`标量。 |

## 返回值说明

返回`int64`类型的整数。

## 约束说明

- 只对有限输入且舍入结果落在int64可表示范围内使用；不依赖NaN、无穷或越界输入的整数结果。
- float32输入本身的精度已决定可区分的数值；int64返回类型不能恢复输入丢失的低位。

## 调用示例

使用一组固定输入调用`simt.llround`，校验并打印结果。

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
        dst[0] = simt.llround(src[0])
        dst[1] = simt.llround(src[1])
        dst[2] = simt.llround(src[2])
        dst[3] = simt.llround(src[3])
        dst[4] = simt.llround(src[4])
        dst[5] = simt.llround(src[5])


@host
def run(src, dst):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([2.5, 3.5, -2.5, -3.5, 1.25, -1.25], dtype=torch.float32, device='npu')
    dst = torch.empty((6,), dtype=torch.int64, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([3, 4, -3, -4, 1, -1], dtype=torch.int64)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[3, 4, -3, -4, 1, -1]
```
