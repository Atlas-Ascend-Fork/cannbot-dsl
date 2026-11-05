# simt.ilogb

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

返回非零输入`x`的二进制指数。

$$
r=\left\lfloor\log_2|x|\right\rfloor
$$

## 函数原型

```python
simt.ilogb(x) -> int32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | `float32`标量。 |

## 返回值说明

返回`int32`类型的整数。

## 约束说明

- 只对有限、非零、正规float32使用普通整数指数语义；零、NaN、无穷的特殊整数值不作为可移植结果。

## 调用示例

使用一组固定输入调用`simt.ilogb`，校验并打印结果。

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
        dst[0] = simt.ilogb(src[0])


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([8.0], dtype=torch.float32, device='npu')
    dst = torch.empty((1,), dtype=torch.int32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([3], dtype=torch.int32)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[3]
```
