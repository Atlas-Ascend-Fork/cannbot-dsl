# simt.mulhi

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

返回lhs、rhs的全精度整数乘积的高半部分：32位输入取高32位，64位输入取高64位。

## 函数原型

```python
simt.mulhi(lhs, rhs) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| lhs | 输入 | `int32`或`int64`整数标量。 |
| rhs | 输入 | `int32`或`int64`整数标量，与`lhs`的数据类型相同。 |

## 返回值说明

返回计算结果，数据类型与输入相同。

## 约束说明

- 有符号乘积按算术高位解释。
- 输入须具有明确的整数数据类型。

## 调用示例

使用一组固定输入调用`simt.mulhi`，校验并打印结果。

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
        dst[0] = simt.mulhi(src[0], src[1])


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([65536, 65536], dtype=torch.int32, device='npu')
    dst = torch.empty((1,), dtype=torch.int32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([1], dtype=torch.int32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[1]
```
