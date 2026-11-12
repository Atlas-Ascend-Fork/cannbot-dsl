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

条件支持单个布尔值、整数、浮点数或比较结果。例如，`x`为一维Tensor时，可以用`if x[0] > 0:`根据第一个元素的值选择分支。

## 约束说明

变量初始化、类型、作用域及提前退出的错误示例见[控制流约束](./control-flow.md#约束)。

- 普通`if`、`elif`、`else`分支中不支持`return`、`raise`；`break`、`continue`仅可用于其中嵌套的编译期循环。
- 普通分支中不支持嵌套函数、嵌套类、`lambda`、推导式、`try`/`except`/`finally`、`yield`、`yield from`、`await`、`async for`、`async with`、`del`和`match`；可以使用`assert`和`pass`。

`True`虽然是编译时已知的常量，`if True:`仍受上述语法限制，不能在分支中使用`return`。需要在编译期分支中提前返回时，使用`if cbd.const_expr(True):`，其中`cbd`为`cannbotdsl`的导入别名。用法见[const_expr](./compile-time-control/const-expr.md)。

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
