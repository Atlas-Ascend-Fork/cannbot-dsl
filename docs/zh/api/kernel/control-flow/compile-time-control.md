---
title: 编译期控制
api_name: const_expr, target_version
category: control-flow
api_group: kernel
layer: frontend
call_context: device
execution_unit: compile-time
status: experimental
since: 待追溯
---

# 编译期控制

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

`const_expr()` 和 `target_version()` 将编译时已经确定的 Python 值标记为编译期条件。
它们作为 `if` 或 `while` 的完整条件时，编译器只保留命中的代码路径。

| 接口 | 适用场景 |
| --- | --- |
| `const_expr()` | 标记普通编译期条件，例如功能开关、静态参数或由 Python 常量计算出的条件。 |
| `target_version()` | 专门标记调用方已经计算好的目标版本判断结果；该接口本身不读取或比较硬件版本。 |

## `const_expr()`

### 函数原型

```python
const_expr(value) -> bool
```

### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 简要说明 |
| --- | --- | --- | --- | --- |
| `value` | 可转换为 `bool` 的编译期 Python 值 | 是 | 无 | 需要标记为编译期条件的值。 |

### 返回值说明

返回 `bool(value)`。当它作为 `if` 或 `while` 的完整条件时，条件判断在编译期完成。

## `target_version()`

### 函数原型

```python
target_version(value) -> bool
```

### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 简要说明 |
| --- | --- | --- | --- | --- |
| `value` | 可转换为 `bool` 的编译期 Python 值 | 是 | 无 | 调用方已经计算好的目标版本判断结果。 |

### 返回值说明

返回 `bool(value)`。当它作为 `if` 或 `while` 的完整条件时，条件判断在编译期完成。

## 约束说明

- `target_version()` 不查询当前硬件版本，版本信息及比较逻辑必须由调用方提供。
- 两个接口的 `value` 都必须在编译期确定，不能依赖 Tensor 读取值、Kernel 标量参数或其他运行时计算结果。
- 两个接口都只能在 `@jit` 或 `@kernel` 修饰的上下文中作为控制流条件使用。

## 调用示例

```python
import torch

import cannbotdsl as cbd


USE_OFFSET = True
TARGET_ARCH = cbd.get_platform_info().npu_arch
IS_DAV_3510 = TARGET_ARCH == "dav-3510"


@cbd.kernel
def compile_time_kernel(out: cbd.Tensor):
    if cbd.const_expr(USE_OFFSET):
        out[0] = 1
    else:
        out[0] = 0

    if cbd.target_version(IS_DAV_3510):
        out[1] = 950
    else:
        out[1] = 0


@cbd.host
def run(out):
    compile_time_kernel[1](out)


if __name__ == "__main__":
    out = torch.zeros((2,), dtype=torch.int32, device="npu")
    run(out)
    torch.npu.synchronize()
    print("target:", TARGET_ARCH)
    print("output:", out.cpu().tolist())
```

### 输入

```text
USE_OFFSET = True
TARGET_ARCH = dav-3510
```

### 输出

```text
target: dav-3510
output: [1, 950]
```
