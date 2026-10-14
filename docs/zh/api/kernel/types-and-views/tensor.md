---
title: Tensor
api_name: Tensor
category: types-and-views
api_group: kernel
layer: frontend
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# Tensor

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 简介

`Tensor` 是**实际存储位置与 Layout 的组合**，既可以指向 GM，也可以指向 UB、L1、
L0A、L0B、L0C、BIAS 等片上存储。实际存储位置确定数据所在的存储层级及其起始位置，
dtype 确定元素类型；Layout 则通过 Shape 和 Stride 定义
多维坐标与元素偏移之间的映射。Tensor 将“数据实际存在哪里”和“计算代码如何访问
数据”统一在同一个对象中。

Tensor 同时包含逻辑布局和物理布局。逻辑布局描述计算代码可见的 Shape、Stride 和
坐标范围；物理布局描述数据在对应存储空间中的实际 Shape、Stride 和排布格式。两者
描述的是同一份存储，不代表存在两份数据：ND 场景下二者可以一致，NZ、ZN 等场景下
物理布局可能采用分形排布，而计算代码仍按逻辑布局访问。

Tensor 可以通过下标读取或写入元素，并通过视图共享相同的底层数据。新 Tensor 可以
通过 `permute()`、`view()`、`reinterpret()` 视图接口基于源 Tensor 派生。视图接口
通过改变坐标范围和偏移规则重新解释数据在内存空间的访问方式，但不会自动搬运、重排
或进行数据类型转换。

## 定义

### 属性

Tensor 当前包含以下只读属性。暂未开放的属性仅用于框架内部实现，不作为用户接口：

| 属性 | 类型 | 简要说明 |
| --- | --- | --- |
| `shape` | `Shape` | 当前逻辑 Layout 的坐标范围，等价于 `t.layout.shape`。 |
| `stride` | `Stride` | 各维坐标增加 1 时线性元素偏移的变化量，等价于 `t.layout.stride`，单位为元素。 |
| `dtype` | `DType` | Tensor 中单个元素的存储类型。 |
| `memloc` | `MemLoc` | Tensor 指向的设备存储层级。 |
| `layout` | `Layout` | 由逻辑 Shape 和逻辑 Stride 组成的坐标映射。 |
| `physical_layout` | `Layout` | 数据实际存储使用的物理 Layout。 |
| `physical_shape` | `Shape` | `physical_layout.shape` 的快捷属性。 |
| `physical_stride` | `Stride` | `physical_layout.stride` 的快捷属性，单位为元素。 |
| `logical_format` | `str` | 逻辑 Layout 使用的格式。 |
| `physical_format` | `str` | 数据实际存储使用的格式。 |
| `data_format` | `str` | `physical_format` 的别名。 |
| `rank` | `int` | 逻辑 Shape 的顶层维数。 |

#### 逻辑布局与物理布局属性

同一个 Tensor 可以分别查看逻辑布局和物理布局：

| 查看内容 | 逻辑布局 | 物理布局 |
| --- | --- | --- |
| 完整布局 | `t.layout` | `t.physical_layout` |
| 形状 | `t.shape`，即 `t.layout.shape` | `t.physical_shape`，即 `t.physical_layout.shape` |
| 元素跨度 | `t.stride`，即 `t.layout.stride` | `t.physical_stride`，即 `t.physical_layout.stride` |
| 格式 | `t.logical_format` | `t.physical_format`；`t.data_format` 是它的别名 |

计算代码使用逻辑坐标访问 Tensor；物理属性描述数据在存储中的形状和元素跨度。

### 约束

#### 数据类型使用范围

| 使用形式 | 普通 Python | `@host` | `@host` 内调用的 `@jit` | `@kernel` | `@kernel` 内调用的 `@jit` |
| --- | --- | --- | --- | --- | --- |
| 将 `Tensor` 用作类型注解 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 接收或传递 Tensor 值 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 读取、写入或计算 Tensor 数据 | ✗ | ✗ | ✗ | ✓ | ✓ |

✓ 表示支持对应用法；✗ 表示不支持对应用法。

## 方法

### `permute()`

#### 功能说明

按照 `dims` 指定的顺序重新排列 GM Tensor 的全部轴，返回与输入 Tensor 共享存储
的新视图。该接口只重排逻辑和物理 Shape/Stride 元数据，不移动数据。

#### 函数原型

```python
permute(input: Tensor, dims: tuple[int, ...] | list[int]) -> Tensor
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 详细说明 |
| --- | --- | --- | --- | --- |
| `input` | `Tensor` | 是 | 无 | 待重排轴顺序的 GM Tensor，必须采用 Identity/ND 布局映射。 |
| `dims` | `tuple[int, ...]` 或 `list[int]` | 是 | 无 | 按输出轴顺序填写输入轴编号。例如 `(1, 2, 0)` 表示输出第 0、1、2 轴分别来自输入第 1、2、0 轴。每个输入轴必须且只能出现一次，负轴按 Python 轴编号规则处理。 |

#### 返回值说明

返回与 `input` 共享 Pointer、dtype 和存储地址的 Tensor 视图。

#### 约束说明

- 仅支持 GM Tensor。

#### 调用示例

```python
import torch

import cannbotdsl as cbd


@cbd.kernel
def permute_kernel(x: cbd.Tensor, y: cbd.Tensor):
    view = cbd.permute(x, (1, 0))
    for i in range(3):
        for j in range(2):
            y[i, j] = view[i, j]


@cbd.jit
def run(x, y):
    permute_kernel[1](x, y)


if __name__ == "__main__":
    x = torch.arange(6, dtype=torch.int64, device="npu").reshape(2, 3)
    y = torch.empty((3, 2), dtype=torch.int64, device="npu")
    run(x, y)
    torch.npu.synchronize()
    print(y.cpu().tolist())
```

输入与输出：

```text
输入：[[0, 1, 2], [3, 4, 5]]，dims=(1, 0)
输出：[[0, 3], [1, 4], [2, 5]]
```

### `view()`

#### 功能说明

返回与源 Tensor 共享同一存储的新 Tensor。`view()` 只改变这份数据的解释方式，
不会申请新存储、复制、搬运或重排数据。通过新视图写入数据时，修改的是源 Tensor
指向的同一存储。

`view()` 提供两种互斥的视图变换：

| 变换方式 | 作用 |
| --- | --- |
| Shape view | 保留 dtype，使用新的逻辑 Shape 访问同一段存储。 |
| dtype view | 保留存储中的原始字节，使用新的 dtype 解释这些字节。 |

`view()` 只改变 Tensor 的逻辑访问方式，不会改变已有数据在内存中的实际排列。
`dtype view` 也不执行数值类型转换；需要转换数值时应使用 `cast`。

#### 函数原型

```python
t.view(d0, d1, ...) -> Tensor
t.view((d0, d1, ...)) -> Tensor
t.view(size=(d0, d1, ...)) -> Tensor
t.view(dtype=target_dtype) -> Tensor
t.view(target_dtype) -> Tensor
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 详细说明 |
| --- | --- | --- | --- | --- |
| `*shape` | 一个或多个维度值，或包含全部维度的 tuple/list | 三选一 | `()` | 使用位置参数指定目标逻辑 Shape。 |
| `size` | tuple、list 或 `None` | 三选一 | `None` | 使用关键字指定目标逻辑 Shape。 |
| `dtype` | `DType` 或 `None` | 三选一 | `None` | 使用目标元素类型重新解释原有字节。 |

三种形式互斥，每次调用必须且只能选择一种。

##### Shape 可选值

`t.view(size=(2, 8))` 与 `t.view(2, 8)` 含义相同。

| 调用形式 | 说明 |
| --- | --- |
| `t.view(2, 8)` | 逐维指定目标 Shape。 |
| `t.view((2, 8))`、`t.view([2, 8])` | 使用一个 tuple/list 指定全部维度。 |
| `t.view(rows, 8)` | 使用 Kernel 运行时整数指定动态维度。 |
| `t.view(-1, 8)` | 根据源 Tensor 的逻辑元素数量和其他维度推导 `-1` 所在维度。最多可以包含一个 `-1`；不能与 Kernel 运行时动态维度同时使用；其他目标维度必须是编译期可确定的正整数；源 Tensor 的逻辑元素数量必须能被目标 Shape 中非 `-1` 维度的乘积整除。 |

##### `dtype` 可选值

`t.view(dtype=target_dtype)` 和 `t.view(target_dtype)` 含义相同。不支持 `bool` 类型，且
源 Tensor 的末维 Stride 必须为 1。

目标 dtype 的位宽发生变化时，输出 Tensor 的逻辑 Shape 的最后一维长度按位宽比例调整，
Shape 的其他维度保持不变；逻辑 Stride 的非末维按相同比例调整，末维 Stride 固定为 1。
目标 dtype 比源 dtype 更宽时，源末维长度和其他维度的 Stride 必须能被位宽比整除。

#### 返回值说明

返回与源 Tensor 共享存储的新 Tensor 视图。

#### 约束说明

- 仅支持逻辑格式和物理格式均为 ND 的 Tensor。
- 逻辑和物理 Layout 必须是非嵌套布局，两者 rank 相同且大于 0。
- 支持 GM、UB、L1、L0A、L0B、L0C、BIAS、SSBUF 和 FBUF。

#### 调用示例

```python
import torch

import cannbotdsl as cbd


@cbd.kernel
def view_kernel(x: cbd.Tensor, bits: cbd.Tensor, y: cbd.Tensor, decoded: cbd.Tensor):
    shape_view = x.view(2, 8)
    for i in range(2):
        for j in range(8):
            y[i, j] = shape_view[i, j]

    dtype_view = bits.view(dtype=cbd.dtypes.float32)
    for i in range(2):
        decoded[i] = dtype_view[i]


@cbd.jit
def run(x, bits, y, decoded):
    view_kernel[1](x, bits, y, decoded)


if __name__ == "__main__":
    x = torch.arange(16, dtype=torch.float32, device="npu")
    bits = torch.tensor([1065353216, 1073741824], dtype=torch.int32, device="npu")
    y = torch.empty((2, 8), dtype=torch.float32, device="npu")
    decoded = torch.empty(2, dtype=torch.float32, device="npu")
    run(x, bits, y, decoded)
    torch.npu.synchronize()
    print(y.cpu().tolist())
    print(decoded.cpu().tolist())
```

输入与输出：

```text
输入 x：[0.0, 1.0, ..., 15.0]
输入 bits：[1065353216, 1073741824]
输出 y：[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
         [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]
输出 decoded：[1.0, 2.0]
```

### `reinterpret()`

#### 功能说明

为已有 Tensor 创建共享存储的别名视图，可以改变元素类型、逻辑 Shape、Stride 和
起始位置。`reinterpret()` 不复制、重排或执行数据类型转换。

#### 函数原型

```python
t.reinterpret(
    dtype=None,
    shape=None,
    *,
    stride=None,
    offset: int = 0,
) -> Tensor
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 详细说明 |
| --- | --- | --- | --- | --- |
| `dtype` | `DType` 或 `None` | 否 | `None` | 指定目标元素类型；`None` 表示保留源 dtype。 |
| `shape` | `int`、Kernel 运行时 64 位整数、`tuple`、`list`、`Tiler` 或 `None` | 否 | `None` | 指定目标逻辑访问窗口；`None` 表示保留源 Shape 和 Stride。 |
| `stride` | `list[int]`、`tuple[int, ...]` 或 `None` | 否 | `None` | 指定目标视图各维的元素偏移变化量，单位为目标 dtype 元素。 |
| `offset` | 非负 Python `int` 或 Kernel 中的 64 位整数值 | 否 | `0` | 指定新视图起点相对于当前输入 Tensor 起点的字节偏移，不修改输入 Tensor。 |

##### `dtype` 可选值

不支持 `dtypes.bool`。`dtypes.fp4x2_e2m1`、`dtypes.fp4x2_e1m2` 和
`dtypes.int4x2` 可用于打包 4 位数据的同存储视图；目标 shape、stride、offset 和
存储范围仍需满足后续约束。

##### `shape` 可选值

| 可选值 | 含义 |
| --- | --- |
| Python `int` 或 Kernel 运行时 64 位整数 | 指定一维目标 Shape。 |
| `tuple` 或 `list` | 指定多维目标 Shape；每个维度可以是 Python `int` 或 Kernel 运行时 64 位整数。 |
| `make_tiler(...)` 的返回值 | 使用已经创建的 Tiler 指定目标 Shape 和对齐信息。 |

改变 dtype 时，目标 Shape 和 Stride 必须完全静态。

##### `stride` 可选值

显式指定 Shape 且省略 Stride 时，由框架为目标视图生成连续 Stride。显式 Stride
必须是由非负 Python `int` 组成的 list/tuple，目标 Stride 的 rank 必须与目标 Shape
的 rank 相同。可以只传入 Stride 而不传入 Shape；此时目标 Shape 保持为源 Shape，
显式传入的 Stride 作为目标 Stride。

##### `offset` 可选值

| 可选值 | 约束 |
| --- | --- |
| 非负 Python `int` | 不超过 `INT64_MAX`，偏移后满足地址对齐和范围约束。 |
| Kernel 中的 64 位整数值 | 必须显式指定 Shape 并保持源 dtype；调用方保证执行时偏移为正、对齐且不越界。 |

`offset` 的单位始终是字节，偏移基线是调用 `reinterpret()` 的当前 Tensor。

Tensor 没有公开的地址偏移属性，Tensor 的地址偏移由框架内部参数记录。

#### 返回值说明

返回与源 Tensor 共享存储的新 Tensor 视图。源 Tensor 的起始位置和其他属性保持
不变。

#### 约束说明

- 新视图的完整访问范围必须位于源对象的可用存储范围内。
- dtype 改变时，Shape、Stride 和 Offset 必须在编译时确定，并满足目标 dtype 的
  字节数和地址对齐要求。
- `offset` 为 Kernel 运行时 64 位整数时，必须显式传入 `shape`，并且目标 dtype 必须
  与源 Tensor 相同。

#### 调用示例

```python
import torch

import cannbotdsl as cbd
from cannbotdsl import dtypes


@cbd.kernel
def reinterpret_kernel(x: cbd.Tensor, y: cbd.Tensor):
    tmp = cbd.Buffer(cbd.MemLoc.UB, (4,), dtypes.float32)
    view = tmp.reinterpret(dtype=dtypes.float16, shape=(8,))
    cbd.mem_copy(view, x)
    cbd.mem_copy(y, view)


@cbd.jit
def run(x, y):
    reinterpret_kernel[1](x, y)


if __name__ == "__main__":
    x = torch.arange(8, dtype=torch.float16, device="npu")
    y = torch.empty_like(x)
    run(x, y)
    torch.npu.synchronize()
    print(y.cpu().tolist())
```

输入与输出：

```text
输入：[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
源视图：shape=(4,)，dtype=float32，共 16 字节
目标视图：shape=(8,)，dtype=float16，共 16 字节
输出：[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
```
