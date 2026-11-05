# simt.eq

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

判断标量`lhs == rhs`是否成立，结果为一个布尔值。

## 函数原型

```python
simt.eq(lhs, rhs) -> bool
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| lhs | 输入 | 左操作数，`float16`、`bfloat16`、`float32`浮点标量，或`int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`int64`、`uint64`整数标量。 |
| rhs | 输入 | 右操作数，数据类型必须与`lhs`相同，包括整数的位宽和有无符号属性。 |

## 返回值说明

返回布尔值，相等时为`True`，否则为`False`。浮点输入任一为`NaN`时返回`False`。正零与负零相等。

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.eq`，将布尔结果转换为整数后校验并打印：1表示真，0表示假。

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
        dst[0] = simt.cast(simt.eq(src[0], src[1]), dtype=dtypes.int32)
        dst[1] = simt.cast(simt.eq(src[2], src[3]), dtype=dtypes.int32)
        dst[2] = simt.cast(simt.eq(src[4], src[5]), dtype=dtypes.int32)
        dst[3] = simt.cast(simt.eq(src[6], src[7]), dtype=dtypes.int32)


@host
def run(src, dst):
    compute[dim3(1), dim3(1)](src, dst)
def main():
    src = torch.tensor([1, 2, 2, 1, 1, 1, float('nan'), 1], dtype=torch.float32, device='npu')
    dst = torch.empty((4,), dtype=torch.int32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([False, False, True, False], dtype=torch.int32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[0, 0, 1, 0]
```
