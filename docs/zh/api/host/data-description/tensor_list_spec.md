---
title: TensorListSpec
api_name: TensorListSpec
category: data-description
api_group: host
layer: frontend
call_context: host
execution_unit: none
status: experimental
since: 待追溯
---

# `TensorListSpec(...)`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

`TensorListSpec` 是一组真实 Tensor 的**元信息占位描述**。它不保存列表和 Tensor
数据，而是使用 `element` 描述每个 Tensor 共同遵循的元信息模板，并使用 `length`
描述 Tensor 的数量，供 `cannbotdsl.compile(...)` 在提前编译 `@host` 函数时确定
张量列表形参的接口。

`TensorListSpec` 本身不是 Python `list` 或 `tuple`，也不是设备侧 `TensorList`，不能
通过下标取得真实 Tensor，更不能读取、写入、搬运或计算 Tensor 数据。

| 使用阶段 | 应传入的对象 | 作用 |
| --- | --- | --- |
| 提前编译 | `TensorListSpec` | 占位描述列表长度以及每个 Tensor 应满足的元信息。 |
| 执行编译结果 | 由真实 NPU Tensor 组成的 `list` 或 `tuple` | 提供实际 Tensor 数据，并接受编译接口的约束检查。 |

## 函数原型

```python
TensorListSpec(
    element: TensorSpec,
    length: int | Dim,
) -> TensorListSpec
```

## 参数说明

| 参数 | 类型 | 必选 | 默认值 | 简要说明 |
| --- | --- | --- | --- | --- |
| `element` | `TensorSpec` | 是 | 无 | 描述列表中每个真实 Tensor 共用的元信息模板；其 `storage_format` 必须为 `"nd"`。 |
| `length` | `int` 或 `Dim` | 是 | 无 | 指定固定或动态的 Tensor 数量。`int` 必须在 signed int64 范围内且大于或等于 1，不接受 `bool`；`Dim.min` 必须大于或等于 1，实际长度须满足该 `Dim` 的 `max` 和 `multiple_of` 约束。 |

## 返回值说明

返回 `TensorListSpec` 对象，包含 `element` 和 `length` 两个只读属性。

## 约束说明

- 对应的 `@host` 形参必须显式注解为 `TensorList`，不能省略或误写成
  `list[Tensor]`、`tuple[Tensor, ...]`、`Tensor`。
- `TensorListSpec` 只能传给 `cannbotdsl.compile(...)`；直接调用 `@host` 函数必须传真实 Tensor 的
  list/tuple。

## 调用示例

```python
import cannbotdsl as cbd
import torch
import torch_npu  # noqa: F401


@cbd.kernel
def copy_first_kernel(xs: cbd.TensorList, y: cbd.Tensor):
    first = xs[0]
    for i in range(first.shape[0]):
        y[i] = first[i]


@cbd.host
def copy_first(xs: cbd.TensorList, y: cbd.Tensor):
    copy_first_kernel[1](xs, y)


element_spec = cbd.TensorSpec((8,), cbd.dtypes.float16)
list_spec = cbd.TensorListSpec(element_spec, length=3)
output_spec = cbd.TensorSpec((8,), cbd.dtypes.float16)

# AOT 编译阶段传入 TensorListSpec。
program = cbd.compile(copy_first, list_spec, output_spec)

# 执行阶段传入真实 Tensor 的 list。
xs = [
    torch.arange(8, dtype=torch.float16, device="npu:0"),
    torch.full((8,), 10, dtype=torch.float16, device="npu:0"),
    torch.full((8,), 20, dtype=torch.float16, device="npu:0"),
]
y = torch.zeros(8, dtype=torch.float16, device="npu:0")
program(xs, y)
torch.npu.synchronize()
print("output:", y.cpu().tolist())
```

### 示例输入

```text
编译输入：
  xs: TensorListSpec(
        element=TensorSpec(shape=(8,), dtype=float16, stride=(1,), storage_format="nd"),
        length=3,
      )
  y: TensorSpec(shape=(8,), dtype=float16, stride=(1,), storage_format="nd")

运行输入：
  xs[0] = [0, 1, 2, 3, 4, 5, 6, 7]
  xs[1] = [10, 10, 10, 10, 10, 10, 10, 10]
  xs[2] = [20, 20, 20, 20, 20, 20, 20, 20]
  y 初始为全 0
```

### 预期输出

```text
output: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
```

运行时 `xs` 绑定为长度为 3 的 TensorList。Kernel 读取 `xs[0]` 并将其复制到
`y`，因此输出内容与列表中的第一个 Tensor 相同。
