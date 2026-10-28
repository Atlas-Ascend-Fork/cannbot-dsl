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

`predicate`支持以下形式：

| 可选形式 | 判断方式 |
| --- | --- |
| 编译期Python值 | 按Python真值规则判断，支持`bool`、整数、浮点数、容器等可转换为`bool`的值。 |
| 运行时布尔值 | 直接作为运行时条件。 |
| 运行时整数 | 非`0`为真，`0`为假。 |
| 运行时浮点数 | `+0.0`和`-0.0`为假，其他值（包括`NaN`）为真。 |

`x`和`y`是两个必选的候选结果：

| 结果形式 | 要求 |
| --- | --- |
| 标量 | 类型必须兼容；一侧为Python数值字面量时，会转换为与另一侧DSL标量兼容的类型。 |
| tuple/list/dict | 容器类型、嵌套方式和元素数量必须一致，dict的键也必须一致。 |
| Tensor | 两侧Tensor的类型和布局必须兼容；选择不会复制Tensor数据。 |

## 返回值说明

返回命中表达式产生的值。运行时条件下，两侧结果必须具有相同结构和兼容类型。

## 约束说明

- 对运行时条件，两个候选表达式都会参与追踪和编译，必须都是合法、可追踪的表达式。
  对Python静态条件，前端按Python语义只求值并追踪选中的候选表达式。
- 在运行时条件表达式中创建并返回Tensor视图时，两侧视图的类型和布局必须兼容。

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
