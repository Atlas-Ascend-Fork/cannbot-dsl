# simt.mul_cast

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

先按lhs、rhs的整数运算类型求乘积，再转换为dtype指定的公开标量类型。

## 函数原型

```python
simt.mul_cast(lhs, rhs, *, dtype) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| lhs | 输入 | `int32`整数标量。 |
| rhs | 输入 | `int32`整数标量，与`lhs`的数据类型相同。 |
| dtype | 输入 | 必填的目标标量数据类型：`int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`int64`、`uint64`、`float16`、`bfloat16`、`float32`或`bool`；例如`dtypes.float32`。 |

## 返回值说明

返回`ScalarValue`，数据类型由`dtype`指定。

## 约束说明

- 先乘法、后转换。即使目标为`int64`，也不能避免两个`int32`输入相乘时的中间结果溢出。

## 调用示例

使用一组固定输入调用`simt.mul_cast`，校验并打印结果。

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
        dst[0] = simt.mul_cast(src[0], src[1], dtype=dtypes.float32)


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([6, 7], dtype=torch.int32, device='npu')
    dst = torch.empty((1,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([42.0], dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[42.0]
```
