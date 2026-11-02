# 片上存储、流水与 GM 协作

片上存储用得好不好，基本决定了算子性能。CANNBot-DSL 给了两个抽象：`Buffer` 和 `RingBuffer`。这一节讲它们的区别、怎么用 `RingBuffer` 搭流水线、框架替你做了什么没替你做什么，以及片上装不下时怎么用 GM 上的 workspace 做多核协作。

## 先算预算

写 RingBuffer 之前先算清楚 UB 够不够。三个数要记住：

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

## `Buffer` 还是 `RingBuffer`

| | `Buffer` | `RingBuffer` |
| --- | --- | --- |
| 对应存储 | 一块 | `depth` 块等大槽位 |
| 轮转 | 无 | 写、读两个游标分别按固定顺序循环 |
| 返回值 | 直接返回 `Tensor` | 资源对象；`next_write()` / `next_read()` 返回选中槽的 `Tensor` |
| 数据同步 | 编译器根据实际 Tensor 访问生成 | 同样根据实际 Tensor 访问生成，不按选槽次数配对 |
| 典型用途 | 临时 scratch、归约中间量、常驻表 | 搬运与计算之间的数据交接、double buffer、跨核交接 |

一句话判断：**只要一块片上 scratch 就用 `Buffer`；要轮换多个槽做流水就用 `RingBuffer`。**

自动声明的地址由内存规划分配。需要固定地址、guard 或阶段间共享区域时，先通过 `dsl.UB.view()` 等完整空间根划分手动视图，再用 [`make_buffer()`](../api/data-structures/storage/buffer/make_buffer.md) / `make_ring_buffer()` 绑定资源。

::: warning 自动分配与手动绑定不能在同一空间混用
模式按 **Kernel × 存储空间** 检查。调用 `dsl.UB.view()` 后，当前 Kernel 的 UB 资源必须来自手动视图；其他存储空间可以独立选择自动或手动方式。含手动空间根的 Kernel 禁止使用 `onchip_rewind()`。
:::

## `Buffer`

```python
tmp = Buffer(MemLoc.UB, dtypes.float32, make_layout((rf, row_stride)))
```

构造函数直接返回 `Tensor`，可以立刻用于 `mem_copy` 和计算接口。当前签名为：

```python
Buffer(mem_loc, dtype, layout, *, physical_layout=None, data_format=None)
```

- `layout` 必须使用 `make_layout(shape, stride=...)` 创建；旧的 `shape`、`capacity`、`size_bytes`、`stride`、`n1_pad`、`view`、`alignment` 构造参数已删除。
- 分配大小由物理布局推导。需要预留更大空间时，先声明一维 Tensor，再通过 `view` 或 `reinterpret` 派生所需视图。
- `physical_layout` 描述 padding 和物理步长；动态逻辑布局可以搭配静态物理布局，但完全动态的物理分配不支持。
- `data_format` 省略时，UB、BIAS、SSBUF、FBUF 为 ND；L1、L0A、L0B、L0C 为 NZ。

## `RingBuffer`

### 心智模型

`RingBuffer` 是一个**带两条独立选槽序列的环形槽位容器**：

```text
depth = 3 的 RingBuffer

        ┌────────┬────────┬────────┐
        │ slot 0 │ slot 1 │ slot 2 │
        └────────┴────────┴────────┘
             ▲        ▲
    写游标 ──┘        └── 读游标      两个游标各自独立推进

    next_write() → 返回写游标当前槽位的 Tensor，再推进写游标
    next_read()  → 返回读游标当前槽位的 Tensor，再推进读游标
```

- 所有槽位有相同的 dtype、逻辑布局和物理布局。
- 每次声明动态执行时，两个游标都从槽 0 开始；`depth=1` 时始终选择唯一槽。
- 选槽只返回 Tensor 并推进对应游标，**不读写数据、不等待、不检查满空，也不要求两个游标一一配对**。需要再次访问同一槽时，保存并复用返回的 Tensor，不要额外推进游标。
- 返回值固定指向所选槽，不是数据快照。环绕覆盖该槽后，旧 Tensor 及其视图别名会看到新内容。
- 默认 View 始终使用声明布局，不会跟随“最近一次写入”的实际尾块变化。需要其他范围或布局时，对选中的 Tensor 显式切片或调用 `reinterpret`。
- 编译器根据 Tensor 的实际访问和资源身份生成同步。VF 内部的 UB 重叠或跨流水依赖仍要按接口约束自己插 [`vmem_bar`](../api/compute/vector/reg_sync/vmem-bar.md)。

### 构造参数

```python
RingBuffer(
    mem_loc,
    dtype,
    layout,
    *,
    depth,
    physical_layout=None,
    data_format=None,
)
```

| 参数 | 说明 |
| --- | --- |
| `depth` | 必填的正编译期 Python 整数，表示物理槽数量；没有跨核专用的 8 槽上限 |
| `layout` | 每个槽的默认逻辑布局，用 `make_layout` 创建；属性描述一个槽，不包含环形维度 |
| `physical_layout` | 可选物理布局，决定每槽分配大小，描述 padding 和步长 |
| `data_format` | 仅支持 `nd` / `nz`；UB 默认 `nd`，L1/L0 默认 `nz` |

跨核交接不需要 `kind` 或 `RingBufferKind`：编译器根据实际 Tensor 访问推导跨核属性。旧的 `capacity`、`shape`、`stride`、`n1_pad`、`addr` 等构造参数同样已删除。

### 最简用法：depth=1

```python
@cbd.kernel
def copy_kernel(src: cbd.Tensor, dst: cbd.Tensor):
    tmp = cbd.RingBuffer(
        cbd.MemLoc.UB, src.dtype, cbd.make_layout((16,)), depth=1
    )

    slot = tmp.next_write()
    cbd.mem_copy(slot, src)       # GM → UB
    slot = tmp.next_read()
    cbd.mem_copy(dst, slot)       # UB → GM
```

### double buffer：depth=2 + 预取

把 `depth` 提到 2，并在使用当前数据之前就发出下一块的搬运，搬运与计算就能重叠：

```python
mem_copy(self._x_db_rb.next_write(),
         tile_slice(gm_x, (1, ct), (gm_row, 0)))       # 预取第 0 块
for t in range(nt):
    nxt = t + 1
    if nxt < nt:
        mem_copy(self._x_db_rb.next_write(),
                 tile_slice(gm_x, (1, ct), (gm_row, nxt)))  # 预取下一块
    x = self._x_db_rb.next_read()                            # 使用当前块
    self._compute_x_squared_sum(x, self._tile_ss_buf, t)
```

这是最常见的流水写法：**循环开始前预取一拍，循环内先发下一拍的搬运、再使用当前拍。**

预取不是「先搬几块存货，用完又回到等搬运」。每一轮都会**新发一次搬运、使用一次更早发出的搬运**，在途的块数大致保持为 `depth - 1`。它改变的是时间怎么叠，不是搬运变快：

```text
串行（depth=1）     拷 算 拷 算 拷 算          总时间 ≈ N × (T拷 + T算)
双缓冲（depth=2）   拷
                    拷 算
                       拷 算
                          拷 算               稳态 ≈ N × max(T拷, T算)
```

如果搬运一直比计算慢，稳态仍由搬运决定，计算会被藏进搬运时间里。预取的意义是：**不再每块都付一遍「先搬完再算」**，而不是让慢的那一侧变快。

### 多级流水：depth=4

`depth=2` 通常只能保留 1 次预取在途，藏得住大约「一轮循环」的延迟。一次 `mem_copy` 从发到完成如果跨过好几轮计算，或多级通路要同时在飞（GM→L1、L1→L0、算、搬出），就需要更多独立槽位来保留尚未处理的数据。加深 `depth` 是为了**多保留几笔在途搬运，把这条延迟链填满**。

它同样不能突破带宽上限：搬运已经 100% 打满时，再加深只会多占片上内存。所以 `matmul` 样例按 L1 容量在 2 和 4 之间选，不是越大越好：

```python
self.l1_buffer_num = (
    self.DB_SIZE                      # 2
    if (a_l1_4buf + b_l1_4buf + bias_4buf) > self.L1_SIZE
    else self.BASIC_L1_BUFFER_NUM     # 4
)
...
l1_a = RingBuffer(
    MemLoc.L1,
    t.a_dtype,
    make_layout((t.base_m, t.k_l1)),
    depth=t.l1_buffer_num,
)
```

::: warning depth 不等于流水
`depth` 提供的是**容量和在途重叠能力**，不会替你重排跨迭代的发射顺序，也不存在通用的 `depth = lag + 1` 规则。真正的流水仍由你写出来的预取、主循环和收尾顺序决定。
:::

### 跨核 RingBuffer

生产者和消费者在不同核时（典型是 AIC 算矩阵乘、AIV 接着做后处理），仍然使用普通声明：

```python
p_l1 = RingBuffer(
    MemLoc.L1, dtypes.float16, make_layout((CHUNK, CHUNK)), depth=2
)
```

**写法与同核完全一致**，仍然是 `next_write()` / `next_read()`。编译器根据实际 Tensor 访问识别跨核交接并生成协议；不要手写 `*_sync_*` 加 GM 中转。

### `onchip_rewind`：声明资源分配边界

自动分配场景下，一个大算子可以分成多个阶段。`onchip_rewind()` 为后续资源声明建立地址和 Buffer-ID 分配边界：

```python
from cannbotdsl import onchip_rewind

# 阶段 1 的自动 Buffer / RingBuffer ...
onchip_rewind()                      # 默认 reset_sync_id=True
# 阶段 2 从新的分配边界开始
```

签名是 `onchip_rewind(reset_sync_id=True)`。`reset_sync_id=True` 同时重置同步计数器分配位置；传 `False` 可保留原计数器分配阶段：

```python
onchip_rewind(reset_sync_id=False)
```

::: warning 它不是屏障，也不是游标重置
`onchip_rewind()` **不插入硬件完成屏障、不释放仍存活的资源、不重置任何 RingBuffer 的写读游标**。跨边界复用必须已有运行时互斥或完成证明；含手动空间根的 Kernel 禁止调用。
:::

### RingBuffer 的使用约束

- 创建 `RingBuffer` 和调用 `next_write()` / `next_read()` **只能在 `@kernel` 或 `@kernel` 内调用的 `@jit` 中**。
- **不要在 VF 区域内选槽**：应在进入 `with vf(...)` 之前选好槽位，再把返回的 Tensor 传进去。
- 选槽没有满空或配对检查；算法必须保证生产、使用和环绕覆盖的顺序正确。
- 绑定 View 不分配新存储，也不推进游标。读端需要尾块布局时，保存生产者的 View，或显式构造相同布局。
- `RingBuffer` 不能通过继承定义子类，也不能直接 `cbd.print`；先取出 Tensor。

## `Array`：保存软件流水的标量历史

深度流水下，第 i 拍发出的搬运可能到第 i+2 拍才使用，这时需要保存「那一拍对应哪个 tile」。旧的 `DelayLine` / `DelayLineGroup` 已删除，当前用公开的 `Array(dtype, length, *, init=None)` 显式保存元数据。

```python
from cannbotdsl import Array, dtypes

# 在动态工作循环外分配，覆盖所有使用位置。
task_tiles = Array(dtypes.int64, 4)
task_kv_tiles = Array(dtypes.int64, 4)
tick = 0
issued = 0

for i in range(total):
    write_slot = tick % 4
    task_tiles[write_slot] = dtypes.int64(m_idx)
    task_kv_tiles[write_slot] = dtypes.int64(n_idx)
    issued += 1

    # 读取两拍前的元数据；预热条件保证历史已经写入。
    if tick >= 2 and tick - 2 < issued:
        read_slot = (tick - 2) % 4
        use(task_tiles[read_slot], task_kv_tiles[read_slot])

    tick += 1
```

要点：

- 数组长度必须覆盖仍在使用的历史范围；读取下标和有效性条件由你维护，动态下标不做边界检查。
- 无 `init` 的数组不会默认清零，首次读取必须由预热条件保护；写入要求严格的同类型，窄整数显式转换为 `int64`。
- 主循环和收尾采用一致的时钟规则。收尾不再产生新任务时冻结 `issued`，继续按原协议推进 `tick`。
- `Array` 与 `RingBuffer` 的游标是两套独立状态；数组只保存标量历史，不替代数据同步，也不能在 VF/SIMT 中捕获或访问。

## `mem_copy`：搬运的统一入口

```python
mem_copy(dst, src, options=None) -> None
```

`options` 由 [copy_options](../api/data-movement/copy-options.md) 创建。`mem_copy` 返回 `None`：先显式绑定目的 Tensor 的布局和实际范围，再执行搬运，后续操作继续使用这个 Tensor 或其派生 View。使用 `RingBuffer` 时，`dst` 和 `src` 都是已经选中的 Tensor。

### 默认搬运方式

- 一般情况下执行不改变已声明 Tensor 解释的搬运；写入不会反向修改其他 View 的布局。
- GM→L1 时，如果目标 Tensor 的布局要求 ND→NZ 或 DN→NZ，**自动执行对应的格式转换**。
- GM→UB 默认使用普通 DMA，**不会自动选择 NDDMA**；需要 NDDMA 时显式传入 `copy_options(engine="nddma")`。
- 转置、分区和逻辑数据变换由 Tensor 视图或 `pad` / `convert` / `relu` 等表达式描述，不使用旧的 `format_transform`、`transpose` 等 `mem_copy` 关键字。

### 常用写法

| 需求 | 当前写法 | 典型场景 |
| --- | --- | --- |
| ND→NZ | 声明 NZ 目的 Tensor，直接 `mem_copy(dst, src)` | GM/UB → L1 |
| 转置加载 | `mem_copy(l0, l1.permute(...))` | L1 → L0A/L0B |
| Fixpipe 数值转换和 scale | `mem_copy(dst, convert(src, dtype=..., scale=...))` | L0C 搬出 |
| 随路 ReLU | `mem_copy(dst, relu(src))` | L0C → GM/UB/L1 |
| 原子累加 | `mem_copy(gm, src, options=copy_options(atomic_add=True))` | Split-K、多核归约 |
| L2 Cache 策略 | `options=copy_options(l2_cache_ctl=...)` | 大矩阵乘按复用情况开关 |
| MX scale | `options=copy_options(mx_scale=l1_scale)` | L1 → L0A/L0B |
| 分区搬运 | 对源或目的调用 `.partition(...)` | 两个 AIV 各处理一部分 |
| NDDMA | `options=copy_options(engine="nddma")` | GM → UB 多维非连续搬运 |
| 逻辑 padding | `mem_copy(dst, pad(src, ...), options=...)` | 显式扩展输入逻辑域 |
| `dst` 传序列 | 一个 GM 源广播到多个 UB/L1 目标 | 常量表广播 |
| `src` 传序列 | 多个同存储源拼接到一个目标 | 聚合输入 |

::: tip 随路处理不占用计算单元
格式转换、量化、ReLU 由搬运单元在搬运过程中完成。能用随路做的事就不要单独写一遍计算。
:::

### 多源 / 多目标的主要约束

- 不能同时传入源序列和目标序列。
- 多源：源必须位于同一存储空间，具有相同 rank 和可证明相等的内部范围与地址间隔，按轴 0 拼接；不能与分区、scale、padding 或原子累加组合。
- 多目标：源必须是一个 ND GM Tensor，目标必须全部位于 UB 或全部位于 L1，形状、dtype 和布局一致，地址按固定间隔排列。

完整约束见 [`mem_copy`](../api/data-movement/mem-copy.md)。

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
3. **NZ 布局下通过 `physical_layout` 增加 N1 方向的物理步长**，把相邻分形拉开。

```python
padded = RingBuffer(
    MemLoc.L1,
    dtypes.float16,
    make_layout((64, 128)),
    depth=2,
    physical_layout=make_layout(
        (8, 4, 16, 16), stride=(1056, 256, 16, 1)
    ),
)
```

::: warning 物理 padding 不是「打开即可消除冲突」的开关
增加物理步长只会改变地址分布，**不会自动补零**，也不保证解决冲突。使用前先确认访问模式撞在哪里，并确保逻辑 View 与实际物理布局匹配，否则只是白占片上空间。旧的 `n1_pad` 构造参数已删除。
:::

## GM 协作：workspace

有些算法的中间结果片上根本放不下，或者需要跨核交换：Stream-K 的部分和、FlashDecode 的分片输出、多核归约的中间量。这时要用 **workspace** —— 一块算子自用的 GM 暂存区。

### workspace 就是一个普通 torch 张量

它不是 DSL 的特殊概念，没有对应的 `MemLoc`。它在[第 3 层的 torch 接口](./torch-interop.md)里分配，作为普通参数传给 `@host`：

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

这是最容易漏的一条。`RingBuffer` 的自动数据同步只管片上资源，**workspace 是普通 GM 张量，框架不知道它的阶段间生产消费关系**：

```text
阶段 A：多个核各算一段，部分和写入 workspace
   ↓  ← 这道屏障必须显式写：模式 0 的核间同步，或 global_sync_all()
阶段 B：某些核从 workspace 读回，归约成最终结果
```

漏了这道屏障的症状是**偶发错误**：核多、数据大时才出现，`block_dim=1` 时永远正常。排查方法见[同步、Cache 与跨核交接](./synchronization.md)。

另外，如果阶段 B 通过 Scalar 读 workspace，还要考虑 DCache 一致性；多核写入未对齐到 Cache Line 时必须用 `*_bypass` 接口。

### 什么时候用 workspace，什么时候用原子累加

| 情况 | 选择 |
| --- | --- |
| 部分和需要高精度累加，或需要复杂归约 | workspace + 显式归约阶段 |
| 只是简单求和，且精度允许 | `mem_copy(gm, ub, options=copy_options(atomic_add=True))`，省掉一个阶段和一次屏障 |
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
        self._x_rb = RingBuffer(
            MemLoc.UB, dtype, make_layout((self._tile,)), depth=2
        )
        self._y_rb = RingBuffer(
            MemLoc.UB, dtype, make_layout((self._tile,)), depth=2
        )
        self._tmp = Buffer(MemLoc.UB, dtypes.float32, make_layout((VL,)))

    def __call__(self, gm_x: Tensor, gm_y: Tensor):
        bi, bn = get_block_idx(), get_block_num()
        n_tiles = ceil_div(self._n, self._tile)

        # —— 预取第一拍 ——
        first = bi
        if first < n_tiles:
            mem_copy(
                self._x_rb.next_write(),
                tile_slice(gm_x, (self._tile,), (first,)),
            )

        for t in range(bi, n_tiles, bn):
            # —— 发下一拍搬运 ——
            nxt = t + bn
            if nxt < n_tiles:
                mem_copy(
                    self._x_rb.next_write(),
                    tile_slice(gm_x, (self._tile,), (nxt,)),
                )

            # —— 使用当前拍 ——
            x = self._x_rb.next_read()
            out = self._y_rb.next_write()
            self._compute(x, out)                 # @jit 方法，内部是 with vf(...)
            mem_copy(
                tile_slice(gm_y, (self._tile,), (t,)),
                self._y_rb.next_read(),
            )

    @jit
    def _compute(self, x, out):
        with vf(mode="simd"):
            ...
```

## 下一步

[矢量寄存器与 lane 模型](./vector-registers.md)：数据到 UB 了，再往里一层是寄存器。
