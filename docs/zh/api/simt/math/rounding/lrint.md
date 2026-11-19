# simt.lrint

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

对x舍入到最近整数，中点取偶数。

## 函数原型

```python
simt.lrint(x) -> int64
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 待舍入的 `float16`、`bfloat16` 或 `float32` 标量。 |

## 返回值说明

返回 `int64` 类型的整数。`NaN` 转换为 `0`；正、负无穷以及越界值分别饱和到 `int64` 的最大值、最小值。

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.lrint`，校验并打印结果。

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
        dst[0] = simt.lrint(src[0])
        dst[1] = simt.lrint(src[1])
        dst[2] = simt.lrint(src[2])
        dst[3] = simt.lrint(src[3])
        dst[4] = simt.lrint(src[4])
        dst[5] = simt.lrint(src[5])


@host
def run(src, dst):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([2.5, 3.5, -2.5, -3.5, 1.25, -1.25], dtype=torch.float16, device='npu')
    dst = torch.empty((6,), dtype=torch.int64, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([2, 4, -2, -4, 1, -1], dtype=torch.int64)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[2, 4, -2, -4, 1, -1]
```
