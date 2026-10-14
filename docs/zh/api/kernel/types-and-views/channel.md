---
title: Channel
api_name: Channel
category: types-and-views
api_group: kernel
layer: frontend
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# Channel

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 简介

Channel 是 Kernel 内管理片上存储资源的容器。这些片上存储资源由 Channel 自动
按固定顺序循环使用。容器中每块独立的存储区域称为一个**槽位**，`depth` 表示
槽位数量；所有槽位具有相同的 shape、dtype 和存储布局。使用时，Channel 将当前
槽位以 Tensor 的形式交给生产者或消费者。

除槽位存储外，Channel 容器还维护槽位顺序以及写端、读端各自的推进位置，使计算
代码不需要自行管理 Buffer 数组、槽位下标和循环回绕逻辑。普通构造方式会申请一段
连续的片上存储并划分为等大的槽位；显式 UB 构造方式则将用户已经划分好的静态 UB
视图依次装入 Channel，作为容器中的多个槽位。

## 定义

### 构造函数

`Channel(...)` 创建包含 `depth` 个同规格槽位的 Channel。构造函数确定每个槽位的
逻辑布局、静态容量、元素类型、存储层级和物理格式，并为整组槽位建立共享的资源
声明。普通构造时，框架在 `mem_loc` 对应的片上存储空间中申请一段连续地址，并按
顺序划分为 `depth` 个大小相同的槽位。

#### 函数原型

```python
Channel(
    mem_loc: MemLoc,
    shape: Shape | None = None,
    dtype: DType | None = None,
    *,
    depth: int,
    capacity: Shape | None = None,
    kind: ChannelKind = ChannelKind.SameCore,
    stride: tuple[int, ...] | None = None,
    data_format: str | None = None,
    n1_pad: int = 0,
) -> Channel
```

`shape` 和 `dtype` 在 Python 签名中具有 `None` 默认值，但二者在语义上都是必选
参数；任一缺失都会报错。

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 简要说明 |
| --- | --- | --- | --- | --- |
| `mem_loc` | `MemLoc` | 是 | 无 | 指定所有槽位所在的片上存储层级。 |
| `shape` | `Shape` 或 `None` | 是 | `None` | 指定单个槽位对计算可见的逻辑形状。 |
| `dtype` | `DType` 或 `None` | 是 | `None` | 指定单个槽位的元素存储类型。 |
| `depth` | `int` | 是 | 无 | 指定可循环复用的槽位数量。 |
| `capacity` | `Shape` 或 `None` | 否 | `None` | 指定单个槽位实际预留的静态容量边界。 |
| `kind` | `ChannelKind` | 否 | `ChannelKind.SameCore` | 指定生产者与消费者之间的同步范围。 |
| `stride` | `tuple[int, ...]` 或 `None` | 否 | `None` | 指定单个槽位逻辑布局中各维坐标增加 1 时的元素跨度。 |
| `data_format` | `str` 或 `None` | 否 | `None` | 指定槽位的物理存储格式。 |
| `n1_pad` | `int` | 否 | `0` | 指定 NZ 物理布局中相邻 N1 切片之间额外保留的元素间隔。 |

##### `mem_loc` 参数可选值

`mem_loc` 必须使用 `MemLoc` 枚举。

| 可选值 | 物理位置 |
| --- | --- |
| `MemLoc.UB` | AI Core 内 Vector 计算单元使用的 Unified Buffer（UB）。 |
| `MemLoc.L1` | AI Core 内 Cube 数据通路的 L1 Buffer。 |
| `MemLoc.L0A` | Cube 计算单元左输入端的 L0A Buffer。 |
| `MemLoc.L0B` | Cube 计算单元右输入端的 L0B Buffer。 |
| `MemLoc.L0C` | Cube 计算单元保存累加值和矩阵计算结果的 L0C Buffer。 |
| `MemLoc.BIAS` | Cube 计算单元存放 Bias 数据的专用 Buffer。 |

##### `kind` 参数可选值

| 可选值 | 含义 |
| --- | --- |
| `ChannelKind.SameCore` | 生产者和消费者位于同一个核内；这是默认值。 |
| `ChannelKind.CrossCore` | 生产者和消费者位于不同核，用于跨核交接数据。 |

##### `depth` 参数可选值

`depth` 是可循环复用的槽位数量，也决定最多能够同时保留多少批尚未被后续阶段覆盖
的数据。

| `kind` | depth 可选范围 | 说明 |
| --- | --- | --- |
| `ChannelKind.SameCore` | 正 Python `int`，`depth >= 1` | 实际可用深度还受片上存储容量限制。 |
| `ChannelKind.CrossCore` | 正 Python `int`，`1 <= depth <= 8` | 跨核 Channel 的固定上限为 8。 |

##### `dtype` 参数可选值

`dtype` 必须使用规范的 `dtypes.*` 存储类型描述符。

| 类型类别 | 可选值 |
| --- | --- |
| 常用浮点类型 | `dtypes.float16`、`dtypes.bfloat16`、`dtypes.float32`、`dtypes.float64` |
| 低位浮点类型 | `dtypes.float8_e4m3fn`、`dtypes.float8_e5m2`、`dtypes.float8_e8m0`、`dtypes.hifloat8`、`dtypes.fp4x2_e2m1`、`dtypes.fp4x2_e1m2` |
| 有符号整数 | `dtypes.int8`、`dtypes.int16`、`dtypes.int32`、`dtypes.int64` |
| 无符号整数 | `dtypes.uint8`、`dtypes.uint16`、`dtypes.uint32`、`dtypes.uint64` |
| 打包整数 | `dtypes.int4x2` |

`MemLoc.BIAS` 只接受 32 位元素。

##### `data_format` 参数可选值

| 可选值 | 物理含义 |
| --- | --- |
| `None` | UB、BIAS 使用 `"nd"`；L1、L0A、L0B、L0C 使用 `"nz"`。 |
| `"nd"` | 普通多维布局，不执行 NZ/ZN 分形映射。 |
| `"nz"` | 使用 NZ 分形物理布局。 |
| `"zn"` | 使用 ZN 分形物理布局。 |

字符串区分大小写；`MemLoc.BIAS` 只支持 `"nd"`。

##### `shape` 参数可选值

`shape` 描述单个槽位对计算可见的逻辑形状。维度可以是正 Python `int`，也可以是
由 Kernel 标量参数或运行时计算结果确定的 64 位整数；静态维和动态维可以混用。

| `data_format` 格式 | 支持的 rank | 静态 shape | 动态 shape | 主要要求 |
| --- | --- | --- | --- | --- |
| ND | rank ≥ 1 | 支持 | 支持 | 动态 shape 必须提供同 rank 的静态 capacity。 |
| NZ | rank 2 或 rank 3 | 支持 | 支持 | 物理布局按静态 capacity 推导；`stride` 必须为 `None`。 |
| ZN | rank 2 或 rank 3 | 支持 | 支持 | 物理布局按静态 capacity 推导；`stride` 必须为 `None`。 |
| BIAS（ND） | rank 1 | 支持 | 支持 | 动态 shape 必须提供静态 rank-1 capacity。 |

动态维的运行时值必须大于 0，并且不能超过对应的 capacity。

##### `capacity` 参数可选值

`capacity` 描述每个槽位实际预留的静态容量边界。

| 可选形式 | 使用条件 | 含义 |
| --- | --- | --- |
| `None` | shape 的所有维度都是静态正整数 | 自动令 `capacity = shape`。 |
| `tuple[int, ...]` | 静态或动态 shape 均可 | 显式指定每一维的静态容量上限。 |

显式 capacity 必须是非空、非嵌套的正 Python int tuple，rank 与 shape 完全相同。
shape 的每个静态维必须满足 `shape[i] <= capacity[i]`；动态 shape 必须显式提供
capacity，并由调用方保证运行时值不越界。`shape != capacity` 是合法的，物理存储
仍按 capacity 规划。

##### `stride` 参数可选值

`stride` 的单位是元素，不是字节。

| 可选形式 | 约束 | 含义 |
| --- | --- | --- |
| `None` | 所有格式均可；NZ/ZN 必须使用 | 按 capacity 推导行主序连续逻辑 stride。 |
| 静态 stride | 仅用于 ND；rank 等于 capacity rank；每项是 `[0, INT64_MAX]` 内的 Python `int` | 显式指定各维的元素跨度。 |
| 运行期 stride | 仅用于 ND；rank 等于 capacity rank；shape 与 capacity 必须是相同静态形状 | tuple 中可使用规范的运行期 i64 整数。 |

省略 stride 时按 capacity 推导。

##### `n1_pad` 参数可选值

NZ 将逻辑 N 维按 `C0` 个元素分组，得到 `N1=ceil(N/C0)`。`n1_pad` 表示相邻两个
N1 切片之间额外保留的元素间隔：

```text
单个 N1 切片的有效元素数 = M1 * 16 * C0
相邻 N1 切片的物理跨度   = M1 * 16 * C0 + n1_pad
```

| 可选值 | 使用条件 | 含义 |
| --- | --- | --- |
| `0` | 所有合法格式 | 不增加 N1 切片间隔。 |
| 正 Python `int` | `data_format="nz"`，且 shape 为 rank 2 或 3 | 将相邻 N1 切片的物理跨度增加指定数量的元素。 |

非零 `n1_pad` 不能用于 ND、ZN 或 BIAS。它不改变逻辑 shape 和 physical shape，
但会增大 NZ 的 N1 轴物理步长和片上内存占用；预留间隔不会由构造函数自动写零。

#### 返回值说明

返回 Channel 资源对象。

#### 约束说明

- Channel 不能通过继承定义子类。
- `shape` 和 `dtype` 都必须提供。
- 调用 `dsl.UB.view()` 后，普通 UB Channel 不再允许创建。

##### 数据类型使用范围

| 使用形式 | 普通 Python | `@host` | `@host` 内调用的 `@jit` | `@kernel` | `@kernel` 内调用的 `@jit` |
| --- | --- | --- | --- | --- | --- |
| 创建 Channel | ✗ | ✗ | ✗ | ✓ | ✓ |
| 调用 `produce()`、`consume()` 推进 Channel | ✗ | ✗ | ✗ | ✓ | ✓ |

✓ 表示支持对应用法；✗ 表示不支持对应用法。

## 方法

### `produce()`

选择写入端的下一个槽位，立即推进独立的写游标，并返回该槽位对应的 `Tensor`。
调用本身不访问或初始化槽位数据，不等待消费者，也不执行数据复制。

#### 函数原型

```python
channel.produce() -> Tensor
```

#### 参数说明

无参数。

#### 返回值说明

返回当前写入位置对应的槽位 Tensor。返回值是槽位存储的普通别名，不是数据快照，
可继续传给 `mem_copy()`、计算接口或 Tensor 访问 API。

#### 约束说明

- `produce()` 只选择槽位并推进写游标，不负责等待空槽或检查消费者是否完成。
- 不要在 VF 区域内调用 `produce()`；应在进入 VF 前选择槽位，再将返回的 Tensor
  传入 VF 区域。

### `consume()`

选择读取端的下一个槽位，立即推进独立的读游标，并返回该槽位对应的 `Tensor`。
调用本身不读取槽位数据，不等待生产者，也不执行数据复制。

#### 函数原型

```python
channel.consume() -> Tensor
```

#### 参数说明

无参数。

#### 返回值说明

返回当前读取位置对应的槽位 Tensor。返回值是槽位存储的普通别名，可继续传给
`mem_copy()`、计算接口或 Tensor 访问 API。

#### 约束说明

- `consume()` 只选择槽位并推进读游标，不负责检查槽位是否已经写入数据。
- 不要在 VF 区域内调用 `consume()`；应在进入 VF 前选择槽位，再将返回的 Tensor
  传入 VF 区域。

### 调用示例

```python
import torch
from torch import as_tensor as from_torch_npu

import cannbotdsl as cbd
from cannbotdsl import dtypes


@cbd.kernel
def channel_copy_kernel(x: cbd.Tensor, y: cbd.Tensor):
    ch = cbd.Channel(cbd.MemLoc.UB, (4,), dtypes.float32, depth=2)

    write_slot = ch.produce()
    cbd.mem_copy(write_slot, x)

    read_slot = ch.consume()
    cbd.mem_copy(y, read_slot)


@cbd.jit
def run(x, y):
    channel_copy_kernel[1](x, y)


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
x = [1.0, 2.0, 3.0, 4.0]
Channel: mem_loc=UB, shape=(4,), dtype=float32, depth=2
```

输出：

```text
input : [1.0, 2.0, 3.0, 4.0]
output: [1.0, 2.0, 3.0, 4.0]
```
