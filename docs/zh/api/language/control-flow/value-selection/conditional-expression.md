# 条件表达式

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据判断条件选择两个候选结果之一。

## 语法形式

```python
result = x if predicate else y
```

`predicate`为真时结果为`x`，否则结果为`y`。

## 参数说明

`predicate`支持单个布尔值、整数、浮点数或比较结果。例如，`input`为一维Tensor时，可用`input[0] > 0`作为条件。

`x`和`y`分别是条件为真和为假时返回的结果。

## 返回值说明

返回条件选中的`x`或`y`。选择Tensor时，结果与被选Tensor共享数据。

## 约束说明

- 当`flag`取决于Tensor数据时，`x if flag else y`在编译时会处理`x`和`y`两侧表达式，执行时只计算选中的一侧。两侧均须能通过编译，不能利用`flag`避开无法编译的表达式。
- 在运行时条件表达式中创建并返回Tensor视图时，两侧视图的类型和布局必须兼容。
- 返回元组、列表或字典时，两侧须使用相同的容器类型、元素数量和嵌套结构。例如，两侧都返回列表时，列表长度须相同。字典的键及顺序也须相同。

## 调用示例

以下示例使用输入`[3, 9]`，并在Host侧校验结果。

```python
import cannbotdsl as cbd
from cannbotdsl.tensor import Tensor
import torch
import torch_npu  # noqa: F401


@cbd.kernel
def conditional_expression_kernel(x: Tensor, out: Tensor, flag):
    # 根据运行时flag选择输入元素。
    out[0] = x[0] if flag != 0 else x[1]


class ConditionalExpressionOp:
    @cbd.host
    def run(self, x, out, flag: cbd.dtypes.int64):
        conditional_expression_kernel[1](x, out, flag)


def main():
    x = torch.tensor([3, 9], dtype=torch.int64, device="npu:0")
    for flag, expected in ((1, [3]), (0, [9])):
        out = torch.zeros(1, dtype=torch.int64, device="npu:0")
        ConditionalExpressionOp().run(x, out, flag)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), torch.tensor(expected, dtype=torch.int64))
        print(f"flag={flag} -> {out.cpu().tolist()}")


if __name__ == "__main__":
    main()
```

预期结果：

```text
flag=1 -> [3]
flag=0 -> [9]
```
