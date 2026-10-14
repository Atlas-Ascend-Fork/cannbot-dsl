# 片上存储、流水与 GM 协作

片上存储用得好不好，基本决定了算子性能。CANNBot-DSL 给了两个抽象：`Buffer` 和 `Channel`。这一节讲它们的区别、怎么用 `Channel` 搭流水线、框架替你做了什么没替你做什么，以及片上装不下时怎么用 GM 上的 workspace 做多核协作。

## 先算预算

写 Channel 之前先算清楚 UB 够不够。三个数要记住：

| 量 | 值 |
| --- | --- |
| UB 物理地址空间 | **256 KB，每个 AIV**（不是每个 AI Core） |
| **`get_mem_size("ub")` 返回** | **253952 字节 = 248 KB**，顶部 8 KB 预留（6 KB VF 溢出 + 2 KB native dump）**已经扣掉** |
| 混合 SIMD + SIMT 时 | SIMT Data Cache 再占一块，按最紧的一侧算 |

::: danger 不要在 `get_mem_size("ub")` 上再减一次预留
`get_mem_size("ub")` 给的已经是**可用总量**，不是 262144。直接拿它当分子：

```python
UB_SIZE = get_mem_size("ub")       # 253952
budget  = UB_SIZE - MY_MARGIN      # ✓ 只减自己的对齐余量
budget  = UB_SIZE - 8 * 1024       # ✗ 白少 8 KB
```
:::

一律用 `get_mem_size("ub")` 查，不要硬编码 `248 * 1024`——预留量会随版本变化，接口不会。预算的推导方式是「可用总量 ÷ 每处理一行需要多少字节」，**所有缓冲都要算进去**，包括 double buffer 的倍数和临时量：

```python
ub_factor = (UB_SIZE - RETAINED_SIZE_1K - nca * gamma_bytes) // (
    nca * elem_bytes * MULTI_FACTOR_2 * DOUBLE_BUFFER_NUM   # x、y 双缓冲
    + FLOAT_BYTE_SIZE * (DOUBLE_BUFFER_NUM + X_REDUCE_TMP_NUM)
    + self._row_stride * FLOAT_BYTE_SIZE
    + FLOAT_BYTE_SIZE * 2
)
self._row_factor = max(int(ub_factor), 1)
```

## `Buffer` 还是 `Channel`

| | `Buffer` | `Channel` |
| --- | --- | --- |
| 对应存储 | 一块 | `depth` 块等大槽位 |
| 轮转 | 无 | 按固定顺序自动循环复用 |
| 生产者/消费者同步 | 不提供 | `produce()` / `consume()` 本身不等待；lowering 按配对插入同步 |
| 返回值 | `Tensor` | Channel 对象，用 `produce()` / `consume()` 取 `Tensor` |
| 典型用途 | 临时 scratch、归约中间量、常驻表 | 搬运与计算之间的数据交接、double buffer、跨核交接 |

一句话判断：**只要一块片上 scratch 就用 `Buffer`；要轮换多个槽做流水就用 `Channel`。**

两者的地址都由内存规划自动分配。需要固定地址或在不同阶段共享同一段存储时，可以先用 `dsl.UB.view()` 划分空间，再用 [`make_buffer()`](/zh/api/kernel/types-and-views/make_buffer.html) / [`make_channel()`](/zh/api/kernel/types-and-views/make_channel.html) 把已有视图绑定成 `Buffer` / `Channel`。

::: warning
调用 `dsl.UB.view()` 之后，当前 Kernel 的 UB `Buffer` 必须通过 `make_buffer()` 创建，普通 UB `Channel` 也不再允许创建。两种风格不能混用。
:::

## `Buffer`

```python
tmp = Buffer(MemLoc.UB, shape=(rf, row_stride), dtype=dtypes.float32)
```

构造函数返回的直接就是 `Tensor`，可以立刻用于 `mem_copy` 和计算接口。

两种互斥的布局声明形式：

```python
# shape 形式：由 shape / capacity / stride / 格式推导布局
Buffer(mem_loc, shape, dtype, capacity=..., stride=..., data_format=..., n1_pad=...)

# layout 形式：直接给逻辑布局和物理布局
Buffer(mem_loc, dtype=..., capacity=..., layout=..., physical_layout=..., ...)
```

`data_format` 省略时的默认值：UB、BIAS、SSBUF、FBUF 为 ND；L1、L0A、L0B、L0C 为 NZ。详见 [Buffer](/zh/api/kernel/types-and-views/buffer.html)。

## `Channel`

### 心智模型

`Channel` 是一个**环形的槽位容器**：

```text
depth = 3 的 Channel

        ┌────────┬────────┬────────┐
        │ slot 0 │ slot 1 │ slot 2 │  ← 片上连续地址，等大
        └────────┴────────┴────────┘
             ▲        ▲
    写游标 ──┘        └── 读游标      两个游标各自独立推进

    produce() → 返回写游标当前槽位的 Tensor，并推进写游标
    consume() → 返回读游标当前槽位的 Tensor，并推进读游标
```

- 所有槽位有相同的 shape、dtype 和存储布局。
- `produce()` / `consume()` **只选槽位并推进游标**，它们本身不访问数据、不复制数据。
- 真正的等待由 lowering 根据 `produce` / `consume` 的配对插入。选槽这一步不会阻塞：写满了不会在 `produce()` 里等空槽，读的时候也不会在 `consume()` 里检查数据是否已到。
- VF 内部的 UB 重叠或跨流水依赖仍要按接口约束自己插 [`vmem_bar`](/zh/api/kernel/reg_compute/reg_sync/vmem-bar.html)，`Channel` 不会替你做这件事。

### 构造参数

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

`shape` 和 `dtype` 在签名里有 `None` 默认值，但语义上都是必选的。

| 参数 | 说明 |
| --- | --- |
| `depth` | 槽位数量，也决定最多能同时保留多少批尚未被覆盖的数据。`SameCore` 时 `depth >= 1`（还受片上容量限制），`CrossCore` 时 `1 <= depth <= 8` |
| `kind` | `ChannelKind.SameCore`（默认，同核）或 `ChannelKind.CrossCore`（跨核交接） |
| `capacity` | 动态 shape 必须显式提供；物理存储按 capacity 规划 |
| `data_format` | `None` 时 UB/BIAS 用 `"nd"`，L1/L0A/L0B/L0C 用 `"nz"` |
| `n1_pad` | 仅 NZ 可用。给相邻 N1 切片额外 stride，**不会自动补零**。用途是对齐或拉开相邻 N1，不是「打开即可消除 bank 冲突」的开关 |

完整参数见 [Channel](/zh/api/kernel/types-and-views/channel.html)。

### 最简用法：depth=1

```python
@cbd.kernel
def copy_kernel(src: cbd.Tensor, dst: cbd.Tensor):
    tmp = cbd.Channel(cbd.MemLoc.UB, shape=(16,), dtype=src.dtype, depth=1)

    slot = tmp.produce()
    cbd.mem_copy(slot, src)       # GM → UB
    slot = tmp.consume()
    cbd.mem_copy(dst, slot)       # UB → GM
```

### double buffer：depth=2 + 预取

把 `depth` 提到 2，并在使用当前数据之前就发出下一块的搬运，搬运与计算就能重叠：

```python
# 取自 samples/rms_norm/rms_norm.py 的 ColSplitKernel
mem_copy(self._x_db_ch.produce(), tile_slice(gm_x, (1, ct), (gm_row, 0)))   # 预取第 0 块
for t in range(nt):
    nxt = t + 1
    if nxt < nt:
        mem_copy(self._x_db_ch.produce(),
                 tile_slice(gm_x, (1, ct), (gm_row, nxt)))                  # 预取下一块
    x = self._x_db_ch.consume()                                             # 消费当前块
    self._compute_x_squared_sum(x, self._tile_ss_buf, t)
```

这是最常见的流水写法：**循环开始前预取一拍，循环内先发下一拍的搬运、再消费当前拍。**

预取不是「先搬几块存货，用完又回到等搬运」。每一轮都会**新发一次搬运、消费一次更早发出的搬运**，在途的块数大致保持为 `depth - 1`。它改变的是时间怎么叠，不是搬运变快：

```text
串行（depth=1）     拷 算 拷 算 拷 算          总时间 ≈ N × (T拷 + T算)
双缓冲（depth=2）   拷
                    拷 算
                       拷 算
                          拷 算               稳态 ≈ N × max(T拷, T算)
```

如果搬运一直比计算慢，稳态仍由搬运决定，计算会被藏进搬运时间里。预取的意义是：**不再每块都付一遍「先搬完再算」**，而不是让慢的那一侧变快。

### 多级流水：depth=4

`depth=2` 只能再挂 1 次在途搬运，藏得住大约「一轮循环」的延迟。一次 `mem_copy` 从发到完成如果跨过好几轮计算（或多级通路要同时在飞：GM→L1、L1→L0、算、搬出），槽位不够就会在 `consume` 时出现空档，DMA 也会一段忙一段闲。加深 depth 是为了**多挂几笔在途搬运，把这条延迟链填满**。

它同样不能突破带宽上限：搬运已经 100% 打满时，再加深只会多占片上内存。所以 `matmul` 样例按 L1 容量在 2 和 4 之间选，不是越大越好：

```python
self.l1_buffer_num = (
    self.DB_SIZE                      # 2
    if (a_l1_4buf + b_l1_4buf + bias_4buf) > self.L1_SIZE
    else self.BASIC_L1_BUFFER_NUM     # 4
)
...
l1_a = Channel(MemLoc.L1, shape=(t.base_m, t.k_l1),
               dtype=t.a_dtype, depth=t.l1_buffer_num)
```

::: warning depth 不等于流水
`depth` 提供的是**容量和在途重叠能力**，它不会替你重排跨迭代的发射顺序。真正的流水仍由你写出来的程序顺序决定：预取写在前面，消费写在后面。
:::

### 跨核 Channel

生产者和消费者在不同核时（典型是 AIC 算矩阵乘、AIV 接着做后处理），只需要加一个 `kind`：

```python
p_l1 = Channel(MemLoc.L1, (CHUNK, CHUNK), dtypes.float16,
               depth=2, kind=ChannelKind.CrossCore)
```

**写法与同核 Channel 完全一致**，仍然是 `produce()` / `consume()`。纯 AIC、纯 AIV、Mix 三种 Kernel 的 Channel 写法统一，不需要切换编译模式。

### `channel_rewind`：阶段之间复用片上资源

一个大算子常常分成几个阶段，各阶段需要的片上缓冲完全不同。`channel_rewind()` 为后续 Channel 建立新的片上资源规划边界，让后面的阶段可以复用前面阶段已经用完的地址：

```python
from cannbotdsl.ops.sync import channel_rewind

# 阶段 1 的 Channel ...
channel_rewind()                      # 默认 reset_sync_id=True
# 阶段 2 的 Channel 从新的规划边界开始分配
```

签名是 `channel_rewind(reset_sync_id=True)`：`reset_sync_id` 控制是否**同时**重置同步资源计数器，**默认为 `True`**。需要跨融合阶段继续复用同步资源时，要**显式传** `False`：

```python
channel_rewind(reset_sync_id=False)   # 只重置地址规划，保留同步资源计数器
```

::: warning 默认会把同步资源一起重置
读到 `reset_sync_id` 这个参数名，容易以为「不传就不重置」。实际相反：**不传就会重置**。融合算子里如果后续阶段还要接着用前面分配的同步资源，必须显式传 `False`，否则症状是同步 ID 冲撞导致的偶发数据错。
:::

详见 [`channel_rewind`](/zh/api/kernel/types-and-views/channel_rewind.html)。

### Channel 的使用约束

- 创建 Channel 和调用 `produce()` / `consume()` **只能在 `@kernel` 或 `@kernel` 内调用的 `@jit` 中**。
- **不要在 VF 区域内调用 `produce()` / `consume()`**：应在进入 `with vf(...)` 之前选好槽位，再把返回的 Tensor 传进去。
- `produce()` 不等待空槽，`consume()` 不检查槽位是否已写入——这些由框架根据配对关系生成的同步负责，但前提是你的 `produce` / `consume` 配对是正确的。
- Channel 不能通过继承定义子类。
- Channel 不能直接 `cbd.print`，要先取出 Tensor。

## `DelayLineGroup`：软件流水的标量延迟线

深度流水下会遇到一个问题：第 i 拍发出的搬运，到第 i+2 拍才被消费，这时你需要知道「当时那一拍对应的是哪个 tile」。手写就要维护一堆下标数组和回绕逻辑。

`DelayLineGroup` 就是为此准备的：它管理多组**命名标量**的历史值，所有字段共享同一个推进位置。

```python
from cannbotdsl.types.delay_line import DelayLineGroup

dl = DelayLineGroup(4, "m", "n", "t", "last")     # depth=4，四个字段

for i in range(total):
    dl.push(m=m_idx, n=n_idx, t=t_idx, last=is_last)

    # 读两拍之前写入的全部字段
    snap = dl.tap(2)
    use(snap.m, snap.n, snap.t, snap.last)

    # 或者按字段分别指定滞后量
    m_at_1 = dl.m.tap(1)
    t_at_3 = dl.t.tap(3)

    dl.advance()                                   # 每轮迭代末尾调用一次
```

要点：

- `depth >= 2`，并且必须**大于实际使用的最大 `lag`**。要读两拍前的值，`depth` 至少为 3。
- 每个字段只保存整数标量，统一按 `int64` 存放。
- `push()` 必须提供全部字段；`advance()` 每轮只调用一次。
- 主循环和流水排空阶段必须采用一致的推进规则，遗漏或重复调用 `advance()` 会让字段值与流水阶段错位。

完整说明见 [DelayLineGroup](/zh/api/kernel/types-and-views/delay-line-group.html)。

## `mem_copy`：搬运的统一入口

```python
mem_copy(dst, src, *, engine=None, transpose=None, deq_scale=None,
         dst_subblock=None, unit_flag=0, pad_value=None,
         left_padding=None, right_padding=None, mx_scale=None,
         part_id=None, actual=None, l2_cache_ctl=0, atomic_add=False, axis=None)
```

除 `dst` 和 `src` 外，其他参数都必须用关键字传入。使用 Channel 时，`dst` 要通过 `produce()` 取、`src` 要通过 `consume()` 取。

### `engine=None` 的默认行为

不指定 `engine` 时，接口会根据源、目标的存储位置和布局自动选择：

- 一般情况下执行不改变逻辑布局的普通搬运。
- GM→L1 时，如果目标 Tensor 的布局要求 ND→NZ 或 DN→NZ，**自动执行对应的格式转换**。
- GM→UB 使用 padding，或源 Tensor 是适合 NDDMA 的非连续视图时，自动使用 NDDMA。

但 `engine=None` **不会**自动启用转置、分区、ReLU、布局折叠或块映射。需要这些行为时必须用 [`make_copy_engine`](/zh/api/kernel/data-movement/make-copy-engine.html) 显式配置（L1→L0A/L0B 的转置也可以直接写 `transpose=True`）。

### 几个高价值选项

| 选项 | 作用 | 典型场景 |
| --- | --- | --- |
| `engine=make_copy_engine(format_transform="nd2nz")` | 随路 ND→NZ | GM/UB → L1 |
| `transpose=True` | 转置加载 | L1 → L0A/L0B |
| `deq_scale=...` | Fixpipe 随路反量化 | L0C 搬出 |
| `relu=True`（在 engine 上） | 随路 ReLU | L0C → GM/UB/L1 |
| `atomic_add=True` | 写 GM 时原子累加 | Split-K、多核归约 |
| `l2_cache_ctl=0/1/2/4` | L2 Cache 策略 | 大矩阵乘按复用情况开关 |
| `split_axis` + `part_id` | 分区搬运 | 两个 AIV 各搬一半 |
| `dst` 传序列 | 一个 GM 源广播到多个 UB/L1 目标 | 常量表广播 |
| `src` 传序列 | 多个同存储源拼接到一个目标 | 聚合输入 |

::: tip 随路即免费
随路转换（格式转换、量化、ReLU）由搬运单元在搬运过程中完成，不额外占用计算单元。能用随路做的事就不要单独写一遍计算。
:::

### 多源 / 多目标的主要约束

- 不能同时传入源序列和目标序列。
- 多源：必须同一存储空间、同 dtype 的 ND Tensor，仅支持 `axis=0`，不能与分区、scale、padding 或原子累加组合。
- 多目标：源必须是一个 ND GM Tensor，目标必须全部位于 UB 或全部位于 L1，并且 shape/dtype/布局相同、地址等间隔排列。

完整约束见 [`mem_copy`](/zh/api/kernel/data-movement/mem-copy.html)。

## UB bank 冲突

UB 不是一块平坦的 SRAM。NPU ARCH 3510 上它分成 **8 个 bank group，每组 2 个 bank，每 bank 16 KB**。访问规则是：

**每周期，一个 bank group 支持两次读，或者一次读加一次写 —— 但不支持两次写。**

于是有两类冲突：

| 冲突 | 触发条件 | 后果 |
| --- | --- | --- |
| 读读冲突 | 两次读落在**同一个 bank**，或超过两次读落在**同一个 bank group** | 串行化，矢量流水出现空档 |
| 写写冲突 | 同一个 bank group 上同周期两次写 | 必然串行 |

实践上的处理办法，按优先级：

1. **调整 tile 的行宽（stride）**，让并发访问的地址落在不同 bank group 上。16 KB 是一个 bank 的跨度，行宽接近 16 KB 整数倍时最容易撞。
2. **错开并发访问的起始偏移**，而不是让多个缓冲从同一个相对位置开始。
3. **NZ 布局下用 `n1_pad`** 给相邻 N1 切片加额外 stride，把它们拉开。

::: warning `n1_pad` 不是「打开即可消除冲突」的开关
它只是给相邻 N1 切片增加 stride，**不会自动补零**，也不保证解决冲突。用它之前要先知道自己的访问模式撞在哪里，否则只是白占片上空间。
:::

## GM 协作：workspace

有些算法的中间结果片上根本放不下，或者需要跨核交换：Stream-K 的部分和、FlashDecode 的分片输出、多核归约的中间量。这时要用 **workspace** —— 一块算子自用的 GM 暂存区。

### workspace 就是一个普通 torch 张量

它不是 DSL 的特殊概念，没有对应的 `MemLoc`。它在[第 3 层的 torch 接口](/zh/guide/programming-model/torch-interop)里分配，作为普通参数传给 `@host`：

```python
def matmul_streamk(a, b):
    c = torch.empty(m, n, dtype=a.dtype, device=a.device)
    ws_rows = tiling.sk_tiles * tiling.sk_splits        # 容量由 tiling 推导
    workspace = torch.empty(ws_rows, n, dtype=torch.float32, device=a.device)
    MatmulStreamK(tiling).run(a, b, c, workspace)
    return c
```

Kernel 里它就是一个 GM Tensor，用 `tile_slice` 取视图、用 `mem_copy` 读写，和输入输出没有区别。

### 五条规矩

| 项 | 规矩 |
| --- | --- |
| 容量 | **由 tiling 推导**，不要给一个「够大的」固定值，那会白占显存 |
| dtype | 通常是 **fp32**，因为存的是待归约的部分和，精度不能降 |
| 清零 | 每个核会完整写自己那段 → 不用清；用 `atomic_add` 累加 → **必须先清零** |
| 复用 | 允许调用方传入自己的 workspace 以跨调用复用，但要校验（`flash_attn` 的 `validate_workspace()` 检查 ndim、宽度、行数下界、dtype、device、连续性） |
| 同步 | **阶段之间的屏障必须自己写**，见下 |

### workspace 的同步不会自动生成

这是最容易漏的一条。Channel 的自动同步只管片上槽位，**workspace 是普通 GM 张量，框架不知道它的生产消费关系**：

```text
阶段 A：多个核各算一段，部分和写入 workspace
   ↓  ← 这道屏障必须显式写：模式 0 的核间同步，或 global_sync_all()
阶段 B：某些核从 workspace 读回，归约成最终结果
```

漏了这道屏障的症状是**偶发错误**：核多、数据大时才出现，`block_dim=1` 时永远正常。排查方法见[同步、Cache 与跨核交接](/zh/guide/programming-model/synchronization)。

另外，如果阶段 B 通过 Scalar 读 workspace，还要考虑 DCache 一致性；多核写入未对齐到 Cache Line 时必须用 `*_bypass` 接口。

### 什么时候用 workspace，什么时候用原子累加

| 情况 | 选择 |
| --- | --- |
| 部分和需要高精度累加，或需要复杂归约 | workspace + 显式归约阶段 |
| 只是简单求和，且精度允许 | `mem_copy(gm, ub, atomic_add=True)`，省掉一个阶段和一次屏障 |
| 归约结果还要参与后续计算 | workspace（原子累加完成时间不确定） |

`matmul_streamk` 用 workspace 是因为它要在 AIV 上做 fp32 归约再 cast 回输出 dtype；纯粹的多核求和用 `atomic_add` 更简单。

## 一个完整的流水骨架

把本节内容串起来，一个典型的 Vector 算子骨架是这样的：

```python
@kernel
class MyKernel:
    def __init__(self, n, dtype):
        # —— 编译期：算 tile 大小和 UB 预算 ——
        self._n = int(n)
        self._tile = compute_tile_from_ub_budget(n, dtype)

        # —— 声明片上资源（地址由框架规划）——
        self._x_ch = Channel(MemLoc.UB, (self._tile,), dtype, depth=2)
        self._y_ch = Channel(MemLoc.UB, (self._tile,), dtype, depth=2)
        self._tmp = Buffer(MemLoc.UB, (VL,), dtypes.float32)

    def __call__(self, gm_x: Tensor, gm_y: Tensor):
        bi, bn = get_block_idx(), get_block_num()
        n_tiles = ceil_div(self._n, self._tile)

        # —— 预取第一拍 ——
        first = bi
        if first < n_tiles:
            mem_copy(self._x_ch.produce(), tile_slice(gm_x, (self._tile,), (first,)))

        for t in range(bi, n_tiles, bn):
            # —— 发下一拍搬运 ——
            nxt = t + bn
            if nxt < n_tiles:
                mem_copy(self._x_ch.produce(),
                         tile_slice(gm_x, (self._tile,), (nxt,)))

            # —— 消费当前拍 ——
            x = self._x_ch.consume()
            out = self._y_ch.produce()
            self._compute(x, out)                 # @jit 方法，内部是 with vf(...)
            mem_copy(tile_slice(gm_y, (self._tile,), (t,)), self._y_ch.consume())

    @jit
    def _compute(self, x, out):
        with vf(mode="simd"):
            ...
```

## 下一步

[矢量寄存器与 lane 模型](/zh/guide/programming-model/vector-registers)：数据到 UB 了，再往里一层是寄存器。
