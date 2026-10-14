---
title: while
api_name: while
category: control-flow
api_group: kernel
layer: frontend
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# while

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

`while` 在条件为真时重复执行循环体，适合迭代次数需要由循环中的计算结果决定的场景。
条件可以在编译期确定，也可以依赖 Kernel 运行时数据。

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

`condition` 可以是 Python `bool`、`const_expr(...)`、`target_version(...)`，或由
Tensor 读取值、Kernel 标量参数等运行时数据构成的标量条件。整数按非 0 判断，浮点数
按非 0.0 判断。

## `while ... else`

当前 DSL 不支持在这些循环中使用 `break`，因此 `while ... else` 当前等价于先执行
循环，再执行 `else` 语句块。

## 约束说明

- 不在编译期展开的循环体不能使用 `break`、`continue`、`return` 或 `raise` 提前结束或跳出循环。
- 不支持从循环体中带出新创建的 Tensor 等资源；需要在循环内写入 Tensor 时，应在循环外先取得该 Tensor。
- 只能在 `@jit` 或 `@kernel` 修饰的上下文中使用。

## 调用示例

```python
import torch

import cannbotdsl as cbd


@cbd.kernel
def while_kernel(out: cbd.Tensor):
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


if __name__ == "__main__":
    out = torch.zeros((2,), dtype=torch.int64, device="npu")
    run(out)
    torch.npu.synchronize()
    print(out.cpu().tolist())
```

### 输入

```text
循环范围：0、1、2、3
```

### 输出

```text
[6, 106]
```

