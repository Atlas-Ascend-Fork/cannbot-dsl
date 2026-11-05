# simt.byte_perm

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将x作为低4字节、y作为高4字节组成8字节序列，按selector的四个低半字节索引分别选出结果的四个字节。

## 函数原型

```python
simt.byte_perm(x, y, selector) -> ScalarValue
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | `int32`或`uint32`整数标量。 |
| y | 输入 | `int32`或`uint32`整数标量，与`x`的数据类型相同。 |
| selector | 输入 | `int32`或`uint32`标量，与`x`、`y`同类型；低四个半字节分别指定四个结果字节的来源。 |

## 返回值说明

返回计算结果，数据类型与输入相同。

## 约束说明

- 按`uint32`位模式处理；`selector`的每个选择索引仅取0～7，索引0对应`x`的最低字节；其他控制位不属于本页用法。

## 调用示例

使用一组固定输入调用`simt.byte_perm`，校验并打印结果。

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
        dst[0] = simt.byte_perm(src[0], src[1], src[2])


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([67305985, 134678021, 12816], dtype=torch.int32, device='npu')
    dst = torch.empty((1,), dtype=torch.int32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([67305985], dtype=torch.int32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[67305985]
```
