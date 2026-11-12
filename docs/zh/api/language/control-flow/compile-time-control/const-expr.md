# const_expr
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`const_expr`由CANNBot-DSL提供。通过`import cannbotdsl as cbd`导入后，使用`cbd.const_expr(value)`调用。

`const_expr()`用于标记编译时可确定的条件，例如功能开关或静态参数。在编译期决定执行哪个分支或是否继续`while`循环，无需在设备运行时判断该条件。

## 函数原型

```python
const_expr(value) -> bool
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `value` | 输入 | 必选，无默认值。必须是编译期已知且可转换为`bool`的Python值，不能依赖Tensor读取值（如通过`x[0]`从Tensor中读出的元素值）或其他运行时结果（如由Tensor读取值计算得到的`x[0] + 1`、`x[0] > 0`）。 |

## 返回值说明

返回Python `bool`，值为`bool(value)`。当作为`if`或`while`的完整条件时，条件判断在编译期完成，只选择命中的代码路径。

## 约束说明

将运行时数据用作编译期条件的错误示例见[控制流约束](../control-flow.md#编译期条件依赖运行时数据)。

## 调用示例

以下示例使用固定功能开关选择分支，并在Host侧校验输出。

```python
import torch
import torch_npu  # noqa: F401

import cannbotdsl as cbd


USE_OFFSET = True


@cbd.kernel
def const_expr_kernel(out: cbd.Tensor):
    # 根据编译期功能开关选择分支。
    if cbd.const_expr(USE_OFFSET):
        out[0] = 1
    else:
        out[0] = 0


@cbd.host
def run(out):
    const_expr_kernel[1](out)


def main():
    out = torch.zeros((1,), dtype=torch.int32, device="npu")
    run(out)
    torch.npu.synchronize()
    assert out.cpu().tolist() == [1]
    print("output:", out.cpu().tolist())


if __name__ == "__main__":
    main()
```

当`USE_OFFSET = True`时，预期结果：

```text
output: [1]
```
