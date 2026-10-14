---
title: Buffer
api_name: Buffer
category: types-and-views
api_group: kernel
layer: frontend
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# Buffer

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 简介

创建片上缓存 Buffer 的接口。

`Buffer(...)` 根据 `mem_loc`、`dtype`、shape 和布局参数申请存储，并返回访问该
存储的 `Tensor`。需要自行安排 UB 地址时，也可以先通过 `dsl.UB.view()` 划分空间，
再用 `make_buffer()` 将子视图绑定为 Buffer。

一个 Buffer 只对应一块存储，不负责多块存储的循环轮转，也不提供生产者与消费者
之间的同步。

## 定义

### 构造函数

### 函数原型

```python
Buffer(
    mem_loc: MemLoc,
    shape: Shape | None = None,
    dtype: DType | None = None,
    *,  # 后续参数必须通过参数名传入
    capacity: Shape | None = None,
    layout: Layout | None = None,
    physical_layout: Layout | None = None,
    stride: tuple[int, ...] | None = None,
    data_format: str | None = None,
    n1_pad: int = 0,
) -> Tensor
```

构造函数支持两种互斥的布局声明形式：

```python
# shape 形式：根据 shape、capacity、stride 和格式推导布局
Buffer(
    mem_loc,
    shape,
    dtype,
    capacity=capacity,
    stride=stride,
    data_format=data_format,
    n1_pad=n1_pad,
)

# layout 形式：直接提供逻辑布局和物理布局
Buffer(
    mem_loc,
    dtype=dtype,
    capacity=capacity,
    layout=layout,
    physical_layout=physical_layout,
    data_format=data_format,
    n1_pad=n1_pad,
)
```

### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 简要说明 |
| --- | --- | --- | --- | --- |
| `mem_loc` | `MemLoc` | 是 | 无 | 指定存储所在的片上存储层级。 |
| `shape` | `Shape` 或 `None` | shape 形式必选 | `None` | 指定返回 Tensor 对计算可见的逻辑形状。 |
| `dtype` | `DType` 或 `None` | 是 | `None` | 指定元素的存储类型。 |
| `capacity` | `Shape` 或 `None` | 条件必选 | `None` | 指定实际预留存储的静态容量边界。 |
| `layout` | `Layout` 或 `None` | layout 形式必选 | `None` | 直接指定逻辑 shape 和逻辑 stride。 |
| `physical_layout` | `Layout` 或 `None` | layout 形式必选 | `None` | 直接指定物理 shape 和物理 stride。 |
| `stride` | `tuple[int, ...]` 或 `None` | 否 | `None` | 指定逻辑坐标各维增加 1 时的元素跨度。 |
| `data_format` | `str` 或 `None` | 否 | `None` | 指定 ND、NZ 或 ZN 物理格式。 |
| `n1_pad` | `int` | 否 | `0` | 指定 NZ 布局中相邻 N1 切片之间额外保留的元素间隔。 |

#### `mem_loc` 参数可选值

`mem_loc` 必须使用 `MemLoc` 枚举。

| 可选值 | 物理位置 |
| --- | --- |
| `MemLoc.UB` | AI Core 内 Vector 计算单元使用的 Unified Buffer（UB）。 |
| `MemLoc.L1` | AI Core 内 Cube 数据通路的 L1 Buffer。 |
| `MemLoc.L0A` | Cube 计算单元左输入端的 L0A Buffer。 |
| `MemLoc.L0B` | Cube 计算单元右输入端的 L0B Buffer。 |
| `MemLoc.L0C` | Cube 计算单元保存累加值和矩阵计算结果的 L0C Buffer。 |
| `MemLoc.BIAS` | Cube 计算单元存放 Bias 数据的专用 Buffer。 |
| `MemLoc.SSBUF` | Scalar 计算单元使用的共享 Buffer。 |
| `MemLoc.FBUF` | Fixpipe 数据通路使用的 Buffer。 |

#### `shape` 参数可选值

`shape` 描述返回 Tensor 对计算可见的逻辑形状。使用 shape 形式时，它必须是
非空、非嵌套 tuple。

| 维度形式 | 含义 | 约束 |
| --- | --- | --- |
| 正 Python `int` | 编译时已经确定的静态维度 | 必须大于 0 |
| Kernel 运行时整数 | 由 Kernel 标量参数或运行时计算结果确定的动态维度 | 必须提供同 rank 的静态 `capacity`；运行时值必须为正且不能超过 capacity |

静态维和动态维可以出现在同一个 shape 中。NZ/ZN 要求 shape 和 capacity 的 rank
为 2 或 3；BIAS 要求 rank 为 1。使用 layout 形式时不能再传 `shape`。

#### `dtype` 参数可选值

`dtype` 不能为 `None`。当前可传入以下 `dtypes.*` 描述符：

| 类别 | 可选值 |
| --- | --- |
| 浮点标量类型 | `dtypes.bfloat16`、`dtypes.float16`、`dtypes.float32`、`dtypes.float64`、`dtypes.float8_e4m3fn`、`dtypes.float8_e5m2`、`dtypes.float8_e8m0`、`dtypes.hifloat8` |
| 4 位打包类型 | `dtypes.fp4x2_e1m2`、`dtypes.fp4x2_e2m1`、`dtypes.int4x2` |
| 有符号整数类型 | `dtypes.int8`、`dtypes.int16`、`dtypes.int32`、`dtypes.int64` |
| 无符号整数类型 | `dtypes.uint8`、`dtypes.uint16`、`dtypes.uint32`、`dtypes.uint64` |

Buffer 构造成功只表示能够声明相应存储。具体计算或搬运是否支持该 dtype，仍由
所调用的算子、存储层级和目标产品决定。`MemLoc.BIAS` 只接受 32 位元素。

#### `capacity` 参数可选值

`capacity` 指定实际预留存储的静态容量边界。

| 可选形式 | 使用条件 | 含义 |
| --- | --- | --- |
| `None` | shape 完全静态，且不使用 layout 形式 | 自动令 `capacity = shape` |
| `tuple[int, ...]` | 静态或动态 shape 均可；layout 形式必须使用 | 显式指定每一维的静态容量上限 |

显式 capacity 必须是非空、非嵌套的正 Python int tuple，rank 与逻辑 shape 一致。
shape 中每个静态维必须满足 `shape[i] <= capacity[i]`；动态维的运行时值也不能超过
对应 capacity。实际存储按 capacity 规划。

#### `layout` 参数可选值

| 可选值 | 含义 | 约束 |
| --- | --- | --- |
| `None` | 使用 shape 形式构造 | 必须提供 `shape` |
| `Layout` | 直接使用 `layout.shape` 和 `layout.stride` 作为逻辑布局 | 必须同时提供 `physical_layout` 和静态 `capacity`；不能再提供 `shape` 或 `stride` |

#### `physical_layout` 参数可选值

| 可选值 | 含义 | 约束 |
| --- | --- | --- |
| `None` | shape 形式下由 Buffer 根据 capacity、dtype 和格式推导 | 使用 layout 形式时不允许省略 |
| `Layout` | 直接指定返回 Tensor 的物理 shape 和物理 stride | 只能与 `layout` 一起使用 |

`physical_layout` 描述实际物理排布，但不会替代用于分配边界检查的静态
`capacity`。

#### `stride` 参数可选值

`stride` 的单位是元素，不是字节。

| 可选值 | 含义 | 约束 |
| --- | --- | --- |
| `None` | 根据 `capacity` 推导行主序连续 stride | 默认值 |
| `tuple[int, ...]` | 显式指定各维的元素跨度 | 必须与 capacity 同 rank；每项必须是大于 0 的 Python `int` |

Buffer 不接受运行时 stride；使用 layout 形式时也不能传 `stride`。

#### `data_format` 参数可选值

| 可选值 | 含义 |
| --- | --- |
| `None` | UB、BIAS、SSBUF、FBUF 使用 ND；L1、L0A、L0B、L0C 使用 NZ |
| `"nd"` | 普通多维布局，不进行 NZ/ZN 分形映射 |
| `"nz"` | 使用 NZ 分形物理布局 |
| `"zn"` | 使用 ZN 分形物理布局 |

字符串区分大小写。BIAS、SSBUF 和 FBUF 只支持 ND。

#### `n1_pad` 参数可选值

| 可选值 | 使用条件 | 含义 |
| --- | --- | --- |
| `0` | 所有合法格式 | 不增加 N1 切片间隔 |
| 正 Python `int` | `data_format="nz"` | 增大相邻 N1 切片之间的元素间隔 |

NZ 将逻辑 N 维按 `C0` 个元素分组，得到 `N1=ceil(N/C0)`。`n1_pad` 表示相邻两个
N1 切片之间额外保留的元素间隔：

```text
单个 N1 切片的有效元素数 = M1 * 16 * C0
相邻 N1 切片的物理跨度   = M1 * 16 * C0 + n1_pad
```

`n1_pad` 不能用于 ND 或 ZN，也不会改变逻辑 shape。它会改变 NZ 的物理 stride
和实际存储占用。

### 返回值说明

返回位于 `mem_loc` 指定存储层级的 `Tensor`。

当 shape 小于 capacity 或包含动态维时，返回 Tensor 的逻辑访问范围由 shape
决定，实际预留存储仍由静态 capacity 决定。

### 约束说明

- 调用 `dsl.UB.view()` 后，当前 Kernel 的 UB Buffer 必须通过 `make_buffer()`
  创建。

#### 数据类型使用范围

| 使用形式 | 普通 Python | `@host` | `@host` 内调用的 `@jit` | `@kernel` | `@kernel` 内调用的 `@jit` |
| --- | --- | --- | --- | --- | --- |
| 使用 `Buffer(...)` 分配片上存储 | ✗ | ✗ | ✗ | ✓ | ✓ |

✓ 表示支持对应用法；✗ 表示不支持对应用法。

### 调用示例

```python
import torch
from torch import as_tensor as from_torch_npu

import cannbotdsl as cbd
from cannbotdsl import dtypes


@cbd.kernel
def buffer_copy_kernel(x: cbd.Tensor, y: cbd.Tensor):
    tmp = cbd.Buffer(
        cbd.MemLoc.UB,
        shape=(4,),
        dtype=dtypes.float32,
    )
    cbd.mem_copy(tmp, x)
    cbd.mem_copy(y, tmp)


@cbd.jit
def run(x, y):
    buffer_copy_kernel[1](x, y)


if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="npu")
    y = torch.empty_like(x)
    run(from_torch_npu(x), from_torch_npu(y))
    torch.npu.synchronize()
    print("input :", x.cpu().tolist())
    print("output:", y.cpu().tolist())
```

输入：

```text
[1.0, 2.0, 3.0, 4.0]
```

输出：

```text
input : [1.0, 2.0, 3.0, 4.0]
output: [1.0, 2.0, 3.0, 4.0]
```
