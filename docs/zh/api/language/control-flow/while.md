# while

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`while`在条件为真时重复执行循环体，适合迭代次数需要由循环中的计算结果决定的场景。
条件可以在编译期确定，也可以依赖Kernel运行时数据。

## 语法形式

```python
while condition:
    body
```

```python
while condition:
    body
else:
    else_body
```

## 条件说明

`condition`支持单个布尔值、整数、浮点数或比较结果，例如`x[0] > 0`、`i < n`。

需要在编译期展开循环时，使用`while cbd.const_expr(condition):`。[const_expr](./compile-time-control/const-expr.md)由CANNBot-DSL提供，调用前先通过`import cannbotdsl as cbd`导入。每次判断循环条件时，条件值都须能在编译期确定。

## `while ... else`

设备运行时`while`不支持`break`，因此这类`while ... else`等价于先执行循环，
再执行`else`语句块。编译期展开的`while`保留Python循环语义，可以使用`break`
并在发生`break`时跳过`else`。

## 约束说明

变量初始化、类型、作用域及提前退出的错误示例见[控制流约束](./control-flow.md#约束)。

- 不在编译期展开的循环体不能使用`break`、`continue`、`return`或`raise`提前结束或跳出循环。
- 动态`while`循环体内新创建的Tensor等资源不能直接在循环外使用；需要在循环外继续使用时，应在循环前创建或取得该资源。
- 只能在`@host`、`@jit`或`@kernel`修饰的上下文中使用。

## 调用示例

以下示例计算循环求和及`while ... else`的结果，并在Host侧校验。

```python
import torch
import torch_npu  # noqa: F401

import cannbotdsl as cbd


@cbd.kernel
def while_kernel(out: cbd.Tensor):
    # 累加0到3，并与包含else语句块的循环比较。
    total = 0
    i = 0
    while i < 4:
        total = total + i
        i = i + 1
    out[0] = total

    total_with_else = 0
    j = 0
    while j < 4:
        total_with_else = total_with_else + j
        j = j + 1
    else:
        total_with_else = total_with_else + 100
    out[1] = total_with_else


@cbd.host
def run(out):
    while_kernel[1](out)


def main():
    out = torch.zeros((2,), dtype=torch.int64, device="npu")
    run(out)
    torch.npu.synchronize()
    assert out.cpu().tolist() == [6, 106]
    print(out.cpu().tolist())


if __name__ == "__main__":
    main()
```

示例输入：

```text
循环范围：0、1、2、3
```

预期结果：

```text
[6, 106]
```
