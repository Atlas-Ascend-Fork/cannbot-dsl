# max

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

比较多个数值，或一组数值，返回其中的最大值。

## 函数原型

传入至少两个候选值时：

```python
max(value1, value2, /, *values)
```

传入一个非空元组或列表等序列时：

```python
max(iterable, /)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `value1` | 输入 | 第一个候选值，必选，支持数值常量或运行时整数、浮点标量。 |
| `value2` | 输入 | 第二个候选值，必选，支持数值常量或运行时整数、浮点标量。 |
| `*values` | 输入 | 后续候选值，可省略，数量不限，支持数值常量或运行时整数、浮点标量。 |
| `iterable` | 输入 | 一组候选值，例如元组`(a, b, c)`或列表`[a, b, c]`。其中至少包含一个元素；比较Tensor中的数据时，应先取出元素，例如`max((x[0], x[1]))`。 |

## 返回值说明

返回候选值中的最大值，数据类型由参与比较的候选值确定。

## 约束说明

- `x`为一维整数或浮点Tensor时，可通过`max(x[0], x[1])`比较其中两个元素。
- 参与比较的运行时值不能混用整数和浮点数。比较浮点数时，两侧须使用相同的数据类型，例如均为`float32`。

## 调用示例

以下示例使用输入`[3, 9]`，并在Host侧校验结果。

```python
import cannbotdsl as cbd
from cannbotdsl.tensor import Tensor
import torch
import torch_npu  # noqa: F401


@cbd.kernel
def max_kernel(x: Tensor, out: Tensor):
    # 分别验证直接传入多个值和传入一组值这两种调用形式。
    out[0] = max(x[0], x[1])
    out[1] = max((x[0], x[1]))


@cbd.host
def run(x: Tensor, out: Tensor):
    max_kernel[1](x, out)


def main():
    x = torch.tensor([3, 9], dtype=torch.int64, device="npu:0")
    out = torch.zeros(2, dtype=torch.int64, device="npu:0")
    run(x, out)
    torch.npu.synchronize()
    expected = torch.tensor([9, 9], dtype=torch.int64)
    torch.testing.assert_close(out.cpu(), expected)
    print("output:", out.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
output: [9, 9]
```
