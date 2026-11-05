# simt.clz

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

返回x的32位表示中从最高位起连续零位的个数；x=0时为32。

## 函数原型

```python
simt.clz(x) -> int32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | `int32`整数标量。 |

## 返回值说明

返回计算结果，数据类型为`int32`。

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.clz`，校验并打印结果。

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
        dst[0] = simt.clz(src[0])


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([1], dtype=torch.int32, device='npu')
    dst = torch.empty((1,), dtype=torch.int32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([31], dtype=torch.int32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[31]
```
