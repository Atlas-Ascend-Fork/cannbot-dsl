# simt.nan

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

构造一个`float32`类型的`NaN`值。

## 函数原型

```python
simt.nan(tag) -> float32
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| tag | 输入 | NaN载荷字符串，例如 "0"；不接受运行时张量或整数。 |

## 返回值说明

返回`float32`类型的`NaN`。

## 约束说明

- tag必须为Python编译期字符串；非字符串会抛出TypeError。
- 不保证字符串对应的NaN载荷位模式；判断结果应使用`isnan`，不要使用相等比较。

## 调用示例

使用一组固定输入调用`simt.nan`，校验并打印结果。

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
        dst[0] = simt.nan('0')


@host
def run(src: Tensor, dst: Tensor):
    compute[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([0.0], dtype=torch.float32, device='npu')
    dst = torch.empty((1,), dtype=torch.float32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    assert torch.isnan(dst.cpu()).all().item()
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[nan]
```
