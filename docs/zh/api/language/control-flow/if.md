# if / elif / else

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`if`、`elif`、`else`根据条件选择语句块，用于在不同输入或运行模式下执行不同操作。多个条件按书写顺序判断，执行第一个成立的分支；均不成立时执行可选的`else`。

## 语法形式

```python
if test:
    ...
elif test2:
    ...
else:
    ...
```

## 条件说明

| 条件形式 | 判断方式 |
| --- | --- |
| 编译期Python值 | 按Python真值规则判断，在生成设备代码时确定条件。 |
| `const_expr(...)`、`target_version(...)` | 作为显式编译期条件处理。 |
| 运行时布尔值 | 直接作为运行时条件。 |
| 运行时整数 | 非`0`为真，`0`为假。 |
| 运行时浮点数 | `+0.0`和`-0.0`为假，其他值（包括`NaN`）为真。 |

## 约束说明

- 运行时分支内不支持`return`、`raise`；`break`、`continue`仅可用于其中嵌套的编译期循环。
- 运行时分支内不支持嵌套函数、嵌套类、`lambda`、推导式、`try`/`except`/`finally`、`yield`、`yield from`、`await`、`async for`、`async with`、`del`和`match`；可以使用`assert`和`pass`。
- 运行时分支内可以声明`global`或`nonlocal`，但不能给相应变量赋值。
- 普通`if`即使在编译时恰好得到Python布尔值，仍按运行时分支规则检查。需要明确
  使用编译期分支时，完整条件必须是`const_expr(...)`或`target_version(...)`。
- 运行时条件的各分支都会参与追踪和编译，不能依赖未命中分支跳过编译检查。
  如果条件是Python静态值，前端按Python语义只追踪命中的分支；需要显式表达
  编译期条件时可使用`const_expr(...)`或`target_version(...)`。

## 调用示例

以下示例按mode选择三个分支，并在Host侧校验各分支计算结果。

```python
import cannbotdsl as cbd
from cannbotdsl.tensor import Tensor
import torch
import torch_npu  # noqa: F401


@cbd.kernel
def classify_kernel(x: Tensor, out: Tensor, mode):
    # 根据mode选择分支，计算并写入结果。
    acc = x[0]
    if mode == 0:
        acc = acc + 1
    elif mode == 1:
        acc = acc * 2
    else:
        acc = acc + 3
    out[0] = acc


class ClassifyOp:
    @cbd.host
    def run(self, x, out, mode: cbd.dtypes.int64):
        classify_kernel[1](x, out, mode)


def main():
    x = torch.tensor([10], dtype=torch.int64, device="npu:0")
    for mode, expected in ((0, 11), (1, 20), (2, 13)):
        out = torch.zeros(1, dtype=torch.int64, device="npu:0")
        ClassifyOp().run(x, out, mode)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), torch.tensor([expected], dtype=torch.int64))
        print(f"mode={mode} -> {expected}")


if __name__ == "__main__":
    main()
```

示例输入：

```text
x[0] = 10
mode依次为0、1、2。
```

预期结果：

```text
mode=0 -> 11
mode=1 -> 20
mode=2 -> 13
```
