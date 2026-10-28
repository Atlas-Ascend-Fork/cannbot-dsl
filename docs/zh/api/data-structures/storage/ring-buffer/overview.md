# RingBuffer 简介

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

RingBuffer 是在 Kernel 中管理多块片上存储的容器。每块独立存储称为一个**槽位**，`depth` 表示槽位数量；各槽位采用相同的容量和声明规格，带默认视图时具有相同的形状、dtype 和布局。槽位按固定顺序循环使用，选槽后返回访问该槽位的 Tensor，或通过 `bare=True` 返回 Storage 存储句柄。

除槽位存储外，RingBuffer 还维护互相独立的写游标 head 和读游标 tail，计算代码因此无需自行维护 Buffer 数组、槽位下标和回绕逻辑。每次执行声明时，两个游标均从 0 开始：

- `next_write()` 选择当前写槽位，随后将写游标推进一个位置。
- `next_read()` 选择当前读槽位，随后将读游标推进一个位置。
- 游标到达 `depth` 后回绕到 0；一端推进不会改变另一端的位置。

普通构造分别申请 `depth` 块槽位存储，不应假设整组槽位地址连续。需要显式组织已划分的 UB 槽位时，可使用 `cannbotdsl.make_ring_buffer()` 绑定已有视图。

## 原型定义

| 属性 | 类型 | 简要说明 |
| --- | --- | --- |
| `dtype` | `DType` | 单槽位默认视图的存储元素类型。 |
| `memloc` | `MemLoc` | 各槽位所在的片上存储层级。 |
| `shape` | [`Shape`](../../layout/shape.md) | 单槽位默认视图的逻辑形状。 |
| `stride` | [`Stride`](../../layout/stride.md) | 单槽位默认视图的逻辑元素跨度，单位为元素。 |
| `layout` | [`Layout`](../../layout/overview.md) | 单槽位默认视图的逻辑坐标映射。 |
| `physical_layout` | [`Layout`](../../layout/overview.md) | 单槽位默认视图的物理布局。 |
| `physical_shape` | [`Shape`](../../layout/shape.md) | physical_layout.shape 的快捷属性。 |
| `physical_stride` | [`Stride`](../../layout/stride.md) | physical_layout.stride 的快捷属性，单位为元素。 |
| `layout_map_kind` | `LayoutMapKind` | 默认视图使用的逻辑到物理布局映射类别。 |
| `logical_format` | `str` | 默认视图的逻辑格式，当前为 nd。 |
| `physical_format` | `str` | 默认视图的物理格式，例如 nd、nz 或 zn。 |
| `data_format` | `str` | physical_format 的别名。 |
| `depth` | `int` | 循环槽位数量，不表示已写入或可读取的数据数量。 |

“默认视图”指创建 RingBuffer 时由 `dtype`、`layout`、`physical_layout` 和 `data_format` 确定的 Tensor 视图。通过 `bare=True` 取出槽位的 [Storage](../storage/overview.md) 后，可调用 [Storage.view()](../storage/view.md) 绑定其他布局。

## 数据类型使用范围

| 使用形式 | 普通 Python | `@host` | `@host` 内调用的 `@jit` | `@kernel` | `@kernel` 内调用的 `@jit` |
| --- | --- | --- | --- | --- | --- |
| 构造 RingBuffer，或通过 make_ring_buffer 绑定槽位 | ✗ | ✗ | ✗ | ✓ | ✓ |
| 调用 next_write / next_read 选择并推进槽位 | ✗ | ✗ | ✗ | ✓ | ✓ |
| 读取已有 RingBuffer 的槽位深度及默认视图属性 | ✗ | ✗ | ✗ | ✓ | ✓ |

✓ 表示支持对应用法；✗ 表示不支持对应用法。
