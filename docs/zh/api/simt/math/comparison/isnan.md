# simt.isnan

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

判断标量x是否为NaN。

## 函数原型

```python
simt.isnan(x) -> bool
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | float16、bfloat16或float32浮点标量。 |

## 返回值说明

返回布尔值，输入为`NaN`时为`True`，否则为`False`。

## 约束说明

无

## 调用示例

使用一组固定输入调用`simt.isnan`，将布尔结果转换为整数后校验并打印：1表示真，0表示假。

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
        dst[0] = simt.cast(simt.isnan(src[0]), dtype=dtypes.int32)
        dst[1] = simt.cast(simt.isnan(src[1]), dtype=dtypes.int32)
        dst[2] = simt.cast(simt.isnan(src[2]), dtype=dtypes.int32)
        dst[3] = simt.cast(simt.isnan(src[3]), dtype=dtypes.int32)
        dst[4] = simt.cast(simt.isnan(src[4]), dtype=dtypes.int32)


@host
def run(src, dst):
    compute[dim3(1), dim3(1)](src, dst)
def main():
    src = torch.tensor([0.0, -2.0, float('inf'), -float('inf'), float('nan')], dtype=torch.float32, device='npu')
    dst = torch.empty((5,), dtype=torch.int32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([False, False, False, False, True], dtype=torch.int32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[0, 0, 0, 0, 1]
```
