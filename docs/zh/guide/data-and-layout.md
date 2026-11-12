# 数据、布局与切块

算子代码的一半工作量是「把对的数据送到对的地方」。这一节讲 CANNBot-DSL 怎么描述数据、怎么切数据，以及切的时候要注意什么。

## 五个描述类型 + 一个数据类型

| 类型 | 描述什么 |
| --- | --- |
| `Shape` | 多维数据各维度的长度及其嵌套结构 |
| `Stride` | 各维坐标增加 1 时对应的元素跨度（单位是**元素**，不是字节） |
| `Layout` | 组合 Shape 与 Stride，描述多维坐标到线性元素偏移的映射 |
| `Tiler` | 各维的切块长度 |
| `Coord` | Layout 或 Tensor 中的位置坐标 |
| **`Tensor`** | **实际存储位置 + Layout**，Kernel 中真正可访问的多维数据对象 |

前五个是纯描述，`Tensor` 是唯一能读写的东西。参考文档见[类型与相关接口](../api/data-structures/index.md)。

::: info 这五个类型的可操作面比看起来小
列出它们是为了让你能读懂 `Tensor` 的属性和接口签名，**不是说你会频繁地直接构造和运算它们**。实际能用的操作只有这几个：

| 接口 | 作用 | 用得多吗 |
| --- | --- | --- |
| [`tile_slice`](../api/compute/tensor/tile-slice.md) | 按 tile 坐标取视图 | **最主要的工具** |
| [`make_tiler`](../api/data-structures/tiler/make_tiler.md) | 创建带对齐声明的 `Tiler` | 动态 tile 时 |
| [`ceil_div`](../api/compute/tensor/ceil_div.md) | 向上取整除法 | 常用 |
| [`idx2crd`](../api/compute/tensor/idx2crd.md) | 线性编号转多维坐标 | 分核算 tile 坐标时 |
| `crd2idx` | 多维坐标转线性编号 | `idx2crd` 的逆 |
| `size` | 算 Shape / Layout 的元素总数 | 偶尔 |
| `make_layout` / `make_tensor` | 手工构造 Layout / Tensor | 很少，特殊布局才用 |
| `Tile` / `PartitionTiler` | 切块描述的两个辅助类型 | 很少 |
| `view` / `from_tensor` / `permute` | 三种视图接口，见下文 | 视情况 |

CANNBot-DSL **不提供 Layout 代数**（没有 `composition`、`logical_divide`、`complement` 这类运算）。上面这些是纯粹的构造与换算工具，不能用来组合出更复杂的布局运算。切块一律通过 `tile_slice` 加 `Tiler` 表达，布局转换一律通过 `mem_copy` 的随路转换完成。
:::

## Tensor：存储位置 + 布局

`Tensor` 把「数据实际存在哪里」和「计算代码如何访问数据」统一在同一个对象中。它既可以指向 GM，也可以指向 UB、L1、L0A、L0B、L0C、BIAS 等片上存储。

常用只读属性：

| 属性 | 含义 |
| --- | --- |
| `t.shape` / `t.stride` | 逻辑 Shape / 逻辑 Stride |
| `t.dtype` | 元素存储类型 |
| `t.memloc` | 所在存储层级 |
| `t.layout` | 逻辑 Layout |
| `t.physical_layout` | 物理布局 |
| `t.data_format` | 数据存储格式 |
| `t.rank` | 逻辑 Shape 的顶层维数 |

### 逻辑布局与物理布局

这是 CANNBot-DSL 里最容易搞混的一对概念。

**同一个 Tensor 同时带有逻辑布局和物理布局，它们描述的是同一份存储，不代表存在两份数据。**

- **逻辑布局**：计算代码可见的 Shape、Stride 和坐标范围。你写 `t[i, j]` 用的就是逻辑坐标。
- **物理布局**：数据在对应存储空间中的实际 Shape、Stride 和排布格式。

ND 场景下二者可以一致。NZ、ZN 等分形场景下物理布局采用分形排布，但**计算代码仍按逻辑布局访问**：

```python
# 声明一个 L1 上的 RingBuffer，逻辑上是 (base_m, k_l1) 的二维矩阵
l1_a = RingBuffer(
    MemLoc.L1, t.a_dtype, make_layout((t.base_m, t.k_l1)), depth=4
)
# 物理上按 NZ 分形存放（L1 的 data_format 默认就是 "nz"）
l1_a_tensor = l1_a.next_write()

# 切 tile 时写的还是逻辑坐标
l1_a_slice = tile_slice(l1_a_tensor, (t.base_m, t.base_k), (0, k_l0_idx))
```

你不需要手算分形偏移，但需要知道**分形对齐约束**（见下文「切块的对齐要求」）。

## 三种视图接口

视图接口通过改变坐标范围和偏移规则重新解释数据的访问方式，**不会自动搬运、重排或进行数据类型转换**。

### `view()`：换形状或换 dtype

```python
shape_view = x.view(2, 8)                      # 换逻辑 Shape，保留 dtype
shape_view = x.view((2, 8))                    # 等价写法
shape_view = x.view(-1, 8)                     # 最多一个 -1，自动推导
dtype_view = bits.view(dtype=cbd.dtypes.float32)  # 保留字节，换 dtype 解释
```

形状位置参数和`dtype=`互斥，每次调用只能选一种。

约束：仅支持逻辑格式和物理格式**均为 ND** 的 Tensor；`dtype view` 要求源 Tensor 的末维 Stride 为 1，且不支持 `bool`。dtype view **不执行数值转换**，需要转换数值请用 `vcast` / `cast`。

### `from_tensor()`：改类型、布局、物理步长、起点

```python
tmp = cbd.Buffer(cbd.MemLoc.UB, dtypes.float32, cbd.make_layout((4,)))
view = cbd.from_tensor(
    tmp, dtype=dtypes.float16, layout=cbd.make_layout((8,))
)  # 16 字节换一种解释
```

```python
cbd.from_tensor(
    t, dtype=None, layout=None,
    physical_layout=None, data_format=None, offset=0,
)
```

`offset` 的单位始终是**字节**，基线是传入的源Tensor。`layout` 描述逻辑形状和元素步长，`physical_layout` 描述包含 padding 的物理布局。布局可以包含运行时尺寸或步长；重绑定不搬动数据，也不做通用容量检查，调用方保证访问范围有效。

### `permute()`：重排轴顺序

```python
view = x.permute(1, 0)
```

只重排逻辑和物理 Shape/Stride 元数据，不移动数据。它可用于 GM 以及搬运、计算接口支持的片上 Tensor；具体组合是否可由硬件执行，由后续 `mem_copy` 或计算接口检查。

::: warning 视图不等于搬运
三个接口都不动数据。要真正改变物理排布（ND→NZ、转置加载），需要通过 `mem_copy` 的源、目的布局和轴变换视图表达；不再使用 `make_copy_engine` 或 `transpose=True`。
:::

## `tile_slice`：切块的主力接口

```python
def tile_slice(input: Tensor, tiler: Tiler, coord: Coord) -> Tensor
```

它在 Tensor 上按 **tile 坐标**创建视图，只创建别名，不复制数据。

**最关键的一点：`coord` 是 tile 序号，不是元素下标。**

```python
# tiler=(64, 128)、coord=(2, 1)
#   → 从元素坐标 (128, 128) 开始取一个 64 × 128 的 tile
tile = tile_slice(gm_x, (64, 128), (2, 1))
```

其他要点：

- 输入维度为 N、`tiler` 维度为 K 时，从输入**最后 K 个逻辑维度**取 tile，返回包含这 K 个维度的视图。
- 最后一个 tile 不完整时，返回视图的实际 shape 会自动缩小到剩余范围（静态非整除尾块自动裁剪）。
- 返回视图**保留输入对应维度的 stride**，不会把非连续数据自动压紧。
- 动态坐标和超大 tile 不提供通用的运行时越界保护，调用者要保证 `coord` 在有效范围内。

### 两级切块：`matmul` 的做法

`matmul` 样例用两层 `tile_slice` 完成 GM→L1→L0 的两级切分：

```python
# 第一级：从 GM 取一个 L1 块，格式由目的布局决定
gm_a_tile = tile_slice(gm_a, (t.base_m, t.k_l1), (m_idx, k_l1_idx))
mem_copy(
    l1_a.next_write(), gm_a_tile,
    options=copy_options(l2_cache_ctl=...),
)
l1_a_tensor = l1_a.next_read()

# 第二级：从 L1 块里取一个 L0 块
for k_l0_idx in range(k_l0_per_l1):
    l1_a_slice = tile_slice(l1_a_tensor, (t.base_m, t.base_k), (0, k_l0_idx))
    mem_copy(l0a.next_write(), l1_a_slice)
```

注意第二级的 `coord=(0, k_l0_idx)`：第 0 维取 0 表示覆盖整个 M 轴，第 1 维按 K 方向滚动。

### `make_tiler`：动态 tile 与对齐

tile 大小需要运行期确定，或者需要声明对齐粒度时，用 [`make_tiler`](../api/data-structures/tiler/make_tiler.md) 创建 `Tiler` 再传给 `tile_slice`：

```python
tiler = cbd.make_tiler(..., alignment=...)
view = cbd.tile_slice(gm_x, tiler, coord)
```

动态 tile 当前仅支持 Identity/ND 的 GM Tensor。

## 切块的对齐要求

对 NZ/ZN 等映射布局切分时，内部 tile 边界必须满足对应布局的分块对齐：

| 布局 | 最后两个逻辑维度的对齐粒度 |
| --- | --- |
| NZ（代码中的 ND2NZ） | 分别按 `16` 和 `32 / 元素字节数` |
| ZN（代码中的 DN2NZ） | 交换上述两个维度的对齐粒度 |

以 `coord=0` 覆盖完整逻辑轴时，不要求该轴的 `tiler` 本身是分块大小的整数倍。

这就是为什么矩阵乘的 tiling 代码里充满了这类对齐：

```python
self.base_m = min(self.BASIC_BLOCK_256, ceil_align(self.m, self.BASIC_BLOCK_16))
self.base_k = self.BASIC_BLOCK_K_128B // self.a_dtype_size   # 128 字节 / 元素字节数
```

矢量侧的对齐常数则来自矢量长度 `VL`（样例中取 64）：

```python
VL = 64
self._num_seg = math.ceil(self._num_col / VL)
self._w = self._num_seg * VL        # 对齐到 VL 的行宽
```

## 辅助接口

| 接口 | 作用 |
| --- | --- |
| [`idx2crd(...)`](../api/compute/tensor/idx2crd.md) | 将线性元素编号转换为多维坐标 |
| [`ceil_div(...)`](../api/compute/tensor/ceil_div.md) | 整数或多维结构的向上取整除法 |
| [`make_tiler(...)`](../api/data-structures/tiler/make_tiler.md) | 根据切块长度和对齐粒度创建切块描述 |

## 索引与元素级读写

`Tensor` 支持下标读写：

```python
@cbd.kernel
def k(x: cbd.Tensor, y: cbd.Tensor):
    for i in range(3):
        for j in range(2):
            y[i, j] = x[j, i]
```

这条路径走的是 Scalar 流水，**只适合搬少量标量**（元数据、标志位、小规模索引表）。批量数据必须走 `mem_copy` + 矢量/矩阵计算接口，否则性能会差几个数量级。

## 动态 shape：逻辑布局与物理布局

片上资源的逻辑布局可以包含运行期维度，但物理分配必须在编译期确定。动态逻辑布局可以搭配静态 `physical_layout`：

```python
ch = RingBuffer(
    MemLoc.UB,
    dtypes.float32,
    make_layout((rows, 128)),
    depth=2,
    physical_layout=make_layout((MAX_ROWS, 128)),
)
```

- `layout` 描述当前逻辑范围；`physical_layout` 决定每槽实际预留的存储。
- 运行时实际范围不得超过物理布局可承载的范围；`reinterpret` 和普通视图不做通用容量或越界检查。
- 需要预留更大空间时，也可以先声明一维 `Buffer` / `RingBuffer`，再通过 `view` 或 `reinterpret` 派生所需 Tensor。

::: danger 静态形状用错会静默算错，不会报错
这是动态 shape 机制最危险的一面。看这个序列：

```python
spec = cbd.TensorSpec((1024,), cbd.dtypes.float16)    # 形状写死成 1024
program = cbd.compile(Add().run, spec, spec, spec)

program(x1024, y1024, out1024)    # ✓ 正常
program(x2048, y2048, out2048)    # ✗ 不报错，但只算前 1024 个元素
```

编译产物是按 1024 专门化的，循环次数、tile 划分、尾块处理全部固化成了常量。传入更长的张量时，**框架不会检查，设备侧也不会越界检查**——后面那一半数据根本没被处理，输出里是未初始化的内容。

两种正确做法：

```python
# 做法一：把变化的维度声明成 Dim，一份产物覆盖一个范围
M = cbd.Dim("M", min=8, max=4096, multiple_of=8)
program = cbd.compile(Add().run, cbd.TensorSpec((M,), dtype), ...)

# 做法二：直接传真实 Tensor，让框架按实际 shape 专门化（开发阶段推荐）
Add().run(x, y, out)
```

同理，Kernel 内部的静态物理布局只限定分配，不替调用方检查运行时 View 是否越界；实际范围由调用方保证。
:::

这个机制让一份编译产物能处理一个范围内的不同 shape，配合 Host 侧的 [`Dim`](../api/compilation/host/specs/dim.md) 使用，详见 [JIT 参数与编译缓存](./jit-arguments.md)。

## 速查

| 你想做的事 | 用什么 |
| --- | --- |
| 从大 Tensor 取第 (i, j) 块 | `tile_slice(t, tiler, (i, j))` |
| 改变逻辑形状，不动数据 | `t.view(...)` |
| 按另一种 dtype 解释同一段字节 | `t.view(dtype=...)` 或 `cbd.from_tensor(t, dtype=..., layout=...)` |
| 在片上存储的某个字节偏移处开一个新视图 | `cbd.from_tensor(t, offset=bytes)` |
| 交换 Tensor 的轴顺序 | `t.permute(...)` |
| ND 数据变成 Cube 要的 NZ | 声明 NZ 目的 Tensor，再 `mem_copy(dst, src)` |
| 矩阵转置加载进 L0A/L0B | `mem_copy(l0a, l1_slice.permute(...))` |
| 真正做数值类型转换 | `vcast`（矢量）/ `cast`（标量） |

## 下一步

[片上存储与流水](./onchip-memory.md)：数据切好了，怎么让它在片上流起来。
