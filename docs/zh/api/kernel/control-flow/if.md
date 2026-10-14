---
title: if / elif / else
api_name: if/elif/else
category: control-flow
api_group: kernel
layer: frontend
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# if / elif / else

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

`if` / `elif` / `else` 根据条件选择对应的语句块执行。

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
| Python `bool` | 在生成设备代码时直接确定条件。 |
| `const_expr(...)`、`target_version(...)` | 作为显式编译期条件处理。 |
| 运行时布尔值 | 直接作为运行时条件。 |
| 运行时整数 | 非 `0` 为真，`0` 为假。 |
| 运行时浮点数 | 非 `0.0` 为真，`0.0` 和 `NaN` 为假。 |

## 约束说明

- 运行时分支不支持使用 `break`、`continue`、`return` 或 `raise` 提前退出。
- 运行时分支内不支持嵌套函数、嵌套类、`lambda`、推导式、
  `try`/`except`/`finally`、`yield`、`yield from`、`await`、`async for`、
  `async with`、`del` 和 `match`。
- `assert` 与 `pass` 可以在运行时分支内使用。
- 运行时分支内允许声明 `global` 或 `nonlocal`，但不允许对相应变量赋值。
- 普通 `if` 即使在编译时恰好得到 Python 布尔值，仍按运行时分支规则检查。需要明确
  使用编译期分支时，完整条件必须是 `const_expr(...)` 或 `target_version(...)`。
- 各分支都会参与编译，不能依赖未命中分支跳过编译检查。

## 调用示例

```python
import cannbotdsl as cbd
from cannbotdsl.tensor import Tensor
import torch
import torch_npu  # noqa: F401


@cbd.kernel
def classify_kernel(x: Tensor, out: Tensor, mode):
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

### 示例输入

```text
x[0] = 10
mode 依次为 0、1、2。
```

### 预期输出

```text
mode=0 -> 11
mode=1 -> 20
mode=2 -> 13
```
