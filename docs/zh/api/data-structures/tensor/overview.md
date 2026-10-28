# Tensor 简介

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

Tensor 是**实际存储位置、元素类型与布局的组合**，将“数据实际存在哪里”和“计算代码如何访问数据”组合在同一个对象中。

Tensor 同时包含逻辑布局和物理布局。逻辑布局描述计算代码可见的形状、元素跨度和坐标范围；物理布局描述数据在对应存储空间中的实际形状、元素跨度和排布格式。

两者描述的是**同一份存储**，不代表存在两份数据。ND 场景下二者可以一致；NZ、ZN 等场景下，物理布局可能采用分形排布，而计算代码仍按逻辑坐标访问数据。

## 原型定义

以下以 Tensor 对象 `t` 为例说明其属性和使用方式。

| 属性 | 类型 | 简要说明 |
| --- | --- | --- |
| `shape` | [`Shape`](../layout/shape.md) | 当前逻辑 Layout 的坐标范围，等价于 `t.layout.shape`。 |
| `stride` | [`Stride`](../layout/stride.md) | 各维坐标增加 1 时线性元素偏移的变化量，等价于 `t.layout.stride`，单位为元素。 |
| `dtype` | `DType` | Tensor 中单个元素的存储类型。 |
| `memloc` | `MemLoc` | Tensor 指向的设备存储层级。 |
| `layout` | [`Layout`](../layout/overview.md) | 由逻辑 Shape 和逻辑 Stride 组成的坐标映射。 |
| `physical_layout` | [`Layout`](../layout/overview.md) | 数据实际存储使用的物理 Layout。 |
| `physical_shape` | [`Shape`](../layout/shape.md) | `physical_layout.shape` 的快捷属性。 |
| `physical_stride` | [`Stride`](../layout/stride.md) | `physical_layout.stride` 的快捷属性，单位为元素。 |
| `logical_format` | `str` | 逻辑 Layout 使用的格式。 |
| `physical_format` | `str` | 数据实际存储使用的格式。 |
| `data_format` | `str` | `physical_format` 的别名。 |
| `rank` | `int` | 逻辑 Shape 的顶层维数。 |

同一个 Tensor 可以分别查看逻辑布局和物理布局：

| 查看内容 | 逻辑布局 | 物理布局 |
| --- | --- | --- |
| 完整布局 | `t.layout` | `t.physical_layout` |
| 形状 | `t.shape`，即 `t.layout.shape` | `t.physical_shape`，即 `t.physical_layout.shape` |
| 元素跨度 | `t.stride`，即 `t.layout.stride` | `t.physical_stride`，即 `t.physical_layout.stride` |
| 格式 | `t.logical_format` | `t.physical_format`；`t.data_format` 是它的别名 |

访问 `t[i, j]` 时，`i`、`j` 对应 `t.shape` 中的逻辑轴；数据实际如何存放，由 `t.physical_layout` 描述。

不同布局下，属性的读取范围如下：

| 布局 | `t.stride` | `t.physical_layout`、`t.physical_shape`、`t.physical_stride`、`t.physical_format`、`t.data_format` |
| --- | --- | --- |
| 普通 ND 布局 | 可读取 | 可读取 |
| NZ / ZN 布局 | 不支持读取 | 可读取 |

## 使用方式

### 通过 `[]` 读取和写入元素

`coord` 表示元素的完整坐标，`value` 表示待写入的标量。`t[coord]` 使用完整坐标读取一个元素，返回标量；`t[coord] = value` 将标量写入该位置，修改实际数据。

| 写法 | 含义 |
| --- | --- |
| `t[i]` | 读取一维 Tensor 下标为 `i` 的元素。 |
| `t[i, j]` | 读取二维 Tensor 第 `i` 行、第 `j` 列的元素。 |
| `t[i, j] = value` | 将 `value` 写入二维 Tensor 的指定位置。 |

每个轴都必须给出整数坐标；嵌套布局的坐标 tuple 结构须与逻辑形状一致。坐标可以是 Python 整数或运行时整数值，且必须位于对应轴的范围内。写入值可以是符合目标元素类型取值范围的 Python 标量，或与目标元素类型一致的 Kernel 运行时标量；不同类型须先显式转换。

Tensor 单点读写支持 GM、UB 和 SSBUF，其中 SSBUF 须通过 `vec` 或 `cube` 指定执行核，UB 在 SIMD VF 内支持单点读取但不支持单点赋值。

对于存储位置为 `MemLoc.SSBUF` 的 Tensor `t`，使用方式如下：

| 写法 | 含义 |
| --- | --- |
| `t.vec[coord]` | 在 AIV 核读取指定坐标的元素，返回标量。 |
| `t.vec[coord] = value` | 在 AIV 核将 `value` 写入指定坐标。 |
| `t.cube[coord]` | 在 AIC 核读取指定坐标的元素，返回标量。 |
| `t.cube[coord] = value` | 在 AIC 核将 `value` 写入指定坐标。 |



### 通过 `[]` 派生子视图

索引中使用 `None` 或切片区间时，返回与原 Tensor 共享存储的子视图。

以 `t.shape=(4, 8)` 的二维 Tensor `t` 为例：

| 写法 | 选取范围 | 返回结果 |
| --- | --- | --- |
| `t[1, None]` | 第二行的全部元素。 | 保留两个轴的 Tensor 视图；令 `row = t[1, None]`，可用 `row[0, j]` 访问该行元素。 |
| `t[1:3, 2:6]` | 下标为 1、2 的行，以及下标为 2、3、4、5 的列。 | 形状为 `(2, 4)` 的 Tensor 视图。 |
| `t[None, None]` | 保留两个轴的全部范围。 | 覆盖原范围的 Tensor 视图。 |

子视图保留原元素类型，并按选取位置调整数据访问的起点和范围。它与原 Tensor 指向同一份存储，通过子视图写入元素也会修改原数据。整数坐标与 `None` 组合时，选定的位置仍保留为长度为 1 的轴。

使用时还需满足以下条件：

- 切片步长只能省略或为 `1`，边界必须非负，选取结果不能为空。
- 对静态轴使用 Python 整数切片边界时，超出轴末端的边界会裁剪到该轴长度。
- 切片 `start:stop` 中，`start` 为起始下标，`stop` 为不包含在选取范围内的结束下标。使用运行时整数值时，必须满足 `0 ≤ start < stop ≤ 对应轴长度`，不会自动裁剪。
- 保留完整轴使用 `None`；完整的 `:` 不作为保留轴标记。
- 赋值必须使用完整整数坐标，不支持通过切片或 `None` 批量赋值。
- 对 NZ / ZN Tensor 选取子区域时，若子区域起点使原 NZ / ZN 排布无法直接表示其访问方式，返回组合视图；从已有组合视图选取子区域也会保留组合视图。此时返回 Tensor 的 `stride`、`physical_format` 和 `data_format` 不支持读取，读取会报错；`physical_layout`、`physical_shape` 和 `physical_stride` 仍可读取。

## 数据类型使用范围

| 使用形式 | 普通 Python | `@host` | `@host` 内调用的 `@jit` | `@kernel` | `@kernel` 内调用的 `@jit` |
| --- | --- | --- | --- | --- | --- |
| 将 `Tensor` 用作类型注解 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 接收或传递 Tensor 值 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 读取、写入或计算 Tensor 数据 | ✗ | ✗ | ✗ | ✓ | ✓ |

✓ 表示支持对应用法；✗ 表示不支持对应用法。
