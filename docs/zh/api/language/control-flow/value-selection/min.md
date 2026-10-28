# min

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

从多个候选值或一个可迭代对象中返回最小值。

## 函数原型

传入至少两个候选值时：

```python
min(value1: Any, value2: Any, /, *values: Any) -> Any
```

传入一个非空、固定的可迭代对象时：

```python
min(iterable: Iterable[Any], /) -> Any
```

`Any`、`Iterable`来自`typing`。以上为Python内建函数在DSL中支持的位置参数调用形式，候选值类型见参数说明。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `value1` | 输入 | 第一个候选值，必选，支持Python数值或DSL标量。 |
| `value2` | 输入 | 第二个候选值，必选，支持Python数值或DSL标量。 |
| `*values` | 输入 | 后续候选值，可省略，数量不限，支持Python数值或DSL标量。 |
| `iterable` | 输入 | 单对象形式：传入一个非空、固定的tuple、list等可迭代对象，其中每个元素都是候选值。 |

## 返回值说明

`min()`返回候选值中的最小值。包含运行时值时，
返回类型为各候选值统一后的DSL标量类型。

## 约束说明

- `min()`的运行时形式只支持整数、index或浮点标量，不接受运行时布尔值。
- `min()`参与运行时比较的候选值须具有兼容的数值类型；有符号和无符号运行时整数不能隐式混用。
- `min()`包含运行时DSL值时，不支持关键字参数。

## 调用示例

以下示例使用输入`[3, 9]`，并在Host侧校验结果。

```python
import cannbotdsl as cbd
from cannbotdsl.tensor import Tensor
import torch
import torch_npu  # noqa: F401


@cbd.kernel
def min_kernel(x: Tensor, out: Tensor):
    # 分别验证多参数和可迭代对象两种调用形式。
    out[0] = min(x[0], x[1])
    out[1] = min((x[0], x[1]))


@cbd.host
def run(x: Tensor, out: Tensor):
    min_kernel[1](x, out)


def main():
    x = torch.tensor([3, 9], dtype=torch.int64, device="npu:0")
    out = torch.zeros(2, dtype=torch.int64, device="npu:0")
    run(x, out)
    torch.npu.synchronize()
    expected = torch.tensor([3, 3], dtype=torch.int64)
    torch.testing.assert_close(out.cpu(), expected)
    print("output:", out.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
output: [3, 3]
```
