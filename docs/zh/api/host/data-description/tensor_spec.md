---
title: TensorSpec
api_name: TensorSpec
category: data-description
api_group: host
layer: frontend
call_context: host
execution_unit: none
status: experimental
since: 待追溯
---

# `TensorSpec(...)`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

`TensorSpec` 是真实 Tensor 的**元信息占位描述**。它不保存 Tensor 数据，而是记录
真实 Tensor 应满足的 shape、dtype、stride 和存储格式，供
`cannbotdsl.compile(...)` 在提前编译 `@host` 函数时确定张量形参的接口。

它只提供编译所需的形状、类型和布局信息，不提供
真实存储和数据访问能力。但 `TensorSpec` 不是 `Tensor` 的子类，也不模拟 Tensor 的
索引接口；不能使用它读取、写入、搬运或计算数据。

| 使用阶段 | 应传入的对象 | 作用 |
| --- | --- | --- |
| 提前编译 | `TensorSpec` | 占位描述真实 Tensor 应满足的元信息。 |
| 执行编译结果 | 真实 NPU Tensor | 提供实际存储和数据，并接受编译接口的约束检查。 |

## 函数原型

```python
TensorSpec(
    shape: Iterable[int | Dim],
    dtype: DType,
    *,
    stride: Iterable[int | Dim] | None = None,
    storage_format: str = "nd",
) -> TensorSpec
```

## 参数说明

| 参数 | 类型 | 必选 | 默认值 | 简要说明 |
| --- | --- | --- | --- | --- |
| `storage_format` | `str` | 否 | `"nd"` | 描述真实 Tensor 的存储格式。 |
| `shape` | 可迭代对象 | 是 | 无 | 描述真实 Tensor 的逻辑形状。 |
| `dtype` | `DType` | 是 | 无 | 描述真实 Tensor 的元素类型。 |
| `stride` | 可迭代对象或 `None` | 否 | `None` | 描述真实 Tensor 各维的元素跨度；省略时自动推导。 |

### `storage_format` 参数可选值

| 可选值 | 含义 |
| --- | --- |
| `"nd"` | 真实 NPU Tensor 使用 ND 存储格式。 |
| `"nz"` | 真实 NPU Tensor 使用 NZ 分形存储格式。 |

### `shape` 参数可选值

| `storage_format` | 可选形式 | 约束 |
| --- | --- | --- |
| `"nd"` | 由 Python `int`、`Dim` 或 `Dim` 运算结果组成的可迭代对象 | 允许任意 rank 和空 shape `()`；静态维度必须大于或等于 0。实际执行能力仍取决于 Host ABI 和设备后端。 |
| `"nz"` | 由 Python `int`、`Dim` 或 `Dim` 运算结果组成的可迭代对象 | rank 必须为 2 或 3；静态维度必须大于 0，直接使用的 `Dim` 必须满足 `min >= 1`。 |

### `dtype` 参数可选值

| `storage_format` | 可选值 |
| --- | --- |
| `"nd"` | 浮点类型：`dtypes.float16`、`dtypes.float32`、`dtypes.float64`、`dtypes.bfloat16`、`dtypes.float8_e4m3fn`、`dtypes.float8_e5m2`、`dtypes.float8_e8m0`、`dtypes.hifloat8`、`dtypes.fp4x2_e2m1`<br>有符号整数：`dtypes.int8`、`dtypes.int16`、`dtypes.int32`、`dtypes.int64`<br>无符号整数：`dtypes.uint8`、`dtypes.uint16`、`dtypes.uint32`、`dtypes.uint64`<br>布尔类型：`dtypes.bool`；`dtypes.bool_` 是同一类型的别名 |
| `"nz"` | 浮点类型：`dtypes.float16`、`dtypes.bfloat16`、`dtypes.float32`<br>有符号整数：`dtypes.int8`<br>无符号整数：`dtypes.uint8` |

### `stride` 参数可选值

stride 的项数必须与 shape 的项数相同，单位是元素。

| `storage_format` | 可选形式 | 约束 |
| --- | --- | --- |
| `"nd"` | `None` | 根据 shape 推导行主序紧凑 stride。 |
| `"nd"` | 由 Python `int`、`Dim` 或 `Dim` 运算结果组成的可迭代对象 | 显式指定各维的元素跨度，项数必须与 shape 相同。 |
| `"nz"` | `None` | 根据 shape 推导 NZ 要求的紧凑行主序 stride。 |
| `"nz"` | 可迭代对象 | 显式值必须与根据 shape 推导出的紧凑 stride 完全一致，不支持非紧凑布局。 |

## 返回值说明

返回不可变的 `TensorSpec` 元信息占位对象，包含 `shape`、`dtype`、`stride` 和
`storage_format` 只读属性。

## 约束说明

- 对应函数形参可以无注解，也可以注解为 `Tensor`。

## 调用示例

```python
import cannbotdsl as cbd
import torch
import torch_npu  # noqa: F401


@cbd.kernel
def copy_kernel(x: cbd.Tensor, y: cbd.Tensor):
    for i in range(x.shape[0]):
        y[i] = x[i]


@cbd.host
def copy(x: cbd.Tensor, y: cbd.Tensor):
    copy_kernel[1](x, y)


M = cbd.Dim("M", min=8, max=1024, multiple_of=8)
tensor_spec = cbd.TensorSpec((M,), cbd.dtypes.float32)

# AOT 编译阶段传入规格对象。
program = cbd.compile(copy, tensor_spec, tensor_spec)

# 执行阶段传入满足规格的真实 NPU Tensor。
x = torch.arange(16, dtype=torch.float32, device="npu:0")
y = torch.zeros(16, dtype=torch.float32, device="npu:0")
program(x, y)
torch.npu.synchronize()
torch.testing.assert_close(y.cpu(), torch.arange(16, dtype=torch.float32))
print("output:", y.cpu().tolist())
```

### 示例输入

```text
编译输入：
  x: TensorSpec(shape=(M,), dtype=float32, stride=(1,), storage_format="nd")
  y: TensorSpec(shape=(M,), dtype=float32, stride=(1,), storage_format="nd")
  M: min=8, max=1024, multiple_of=8

运行输入：
  x: NPU Tensor(shape=(16,), dtype=float32)，内容 [0.0, 1.0, ..., 15.0]
  y: NPU Tensor(shape=(16,), dtype=float32)，初始内容全为 0
```

### 预期输出

```text
output: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
```
