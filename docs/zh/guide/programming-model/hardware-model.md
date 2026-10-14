# 硬件与执行模型

写 CANNBot-DSL 不需要懂昇腾硬件的全部细节，但需要懂两件事：**谁在算**、**数据放在哪**。本节只讲这两件事，并把它们对应到 DSL 里的具体接口。同步单独成章，见[同步、Cache 与跨核交接](/zh/guide/programming-model/synchronization)。

本节描述的是 NPU ARCH 3510（Ascend 950PR / Ascend 950DT）。

## 规格速查

[高性能算子编写指南](/zh/guide/programming-model/performance)要求你先「算理论上限」。要算，就得有数。这张表是那些数。

| 维度 | Ascend 950PR | Ascend 950DT |
| --- | --- | --- |
| Cube Core 数（分档出货） | 32 或 28 | 36 / 32 / 28 |
| Vector Core 数 | Cube 的 2 倍 | Cube 的 2 倍 |
| 片上内存 | HiBL 1.0，128 GB（分档 128 / 112） | HiZQ 2.0，144 GB（分档 144 / 96） |
| 内存带宽 | 1.6 TB/s（分档 1.6 / 1.4） | 4 TB/s |
| 对外互联带宽（芯片级峰值） | 2 TB/s | 2 TB/s |

各精度算力（TFLOPS，INT8 为 TOPS；**Cube + Vector 合计 / 仅 Cube**）：

| 精度 | 950PR（32 核档） | 950DT（36 核档） |
| --- | --- | --- |
| MXFP4 | 1784 / 1730 | 2007 / 1946 |
| HiF8 · MXFP8 · FP8 · INT8 | 919 / 865 | 1034 / 973 |
| BF16 · FP16 | 486 / 432 | 547 / 486 |
| TF32（HF32 档） | 243 / 216 | 273 / 243 |

同频下的倍数关系很干净：**HiF8 / MXFP8 / FP8 是 FP16 的 2 倍，MXFP4 是 4 倍。** 这是降精度的收益上限，见[数据类型与量化](/zh/guide/programming-model/data-types)。

::: danger 两个很容易算错的地方
**一、对外宣传的「1 PFLOPS / 2 PFLOPS」是含 Vector 的合计值。** 判断一个 cube-bound 算子的上限时要用「仅 Cube」那一列。差多少取决于精度，**不是一个固定的小数字**：

| 精度 | 合计 / 仅 Cube（950PR 32 核） | 用合计值会把目标定高 |
| --- | --- | --- |
| MXFP4 | 1784 / 1730 | 3.1% |
| HiF8 · MXFP8 · FP8 | 919 / 865 | 6.2% |
| **BF16 · FP16** | 486 / 432 | **12.5%** |
| TF32 | 243 / 216 | **12.5%** |

也就是说，最常用的 BF16 / FP16 恰好是偏差最大的一档。原因是 Vector 侧的 FP16/BF16 算力是 54 TFLOPS，占 Cube 的 1/8；而 MXFP4 上 Vector 不翻倍，占比就被摊薄了。

**二、核数和容量是分档出货的。** 同一型号有多个档位，所以**一定要用 `get_platform_info()` 和 `get_mem_size()` 查，不要硬编码**。例如 950PR 的加速卡形态 Atlas 350 用的是 28 核、112 GB 档，MXFP4 算力 1561 TFLOPS，和 32 核档差 12.5%。
:::

::: info 关于主频与互联带宽
官方从未公布 AI Core 主频，所有吞吐对比都以「同频」为前提，所以不要试图从主频反推单核吞吐。另外 2 TB/s 是**芯片级聚合带宽**；实际一张 PCIe 加速卡的卡间带宽要低一个量级（双卡配置约 424 GB/s 双向），做通信重叠的 tiling 时别用芯片级数字。
:::

## 谁在算：AIC 与 AIV

一颗昇腾 NPU 上有多个 AI Core。AI Core 内部有两类计算单元：

| 单元 | 全称 | 擅长 | 对应 DSL 接口 |
| --- | --- | --- | --- |
| **AIC** | AI Cube | 矩阵乘加（MMAD） | [`matmul`](/zh/api/kernel/cube_compute/matmul.html) |
| **AIV** | AI Vector | 逐元素/归约类矢量运算 | [Reg 矢量计算](/zh/api/kernel/reg_compute/)（`vadd`、`vreduce_sum` 等） |

此外每个核还有标量单元（Scalar），负责地址计算、循环控制和轻量位运算，对应[标量计算](/zh/api/kernel/scalar_compute/)接口。

AIV 上的矢量计算有**两种编程模式**，都用 `with vf(...)` 圈定：`mode="simd"`（默认，寄存器级 SIMD）和 `mode="simt"`（线程级）。本章和后面大部分内容讲的是 SIMD；SIMT 见[三类计算单元](/zh/guide/programming-model/compute#simt-模式)。按官方架构白皮书的定位，混合编程「以 SIMD 为主、SIMT 为辅」——规则访存的 element-wise 计算用 SIMD，不规则控制流和 gather/scatter 用 SIMT。

### block 与 subblock

AIC/AIV 按 **group** 划分。**只有 Mix 算子**才会在一个 group 里同时出现 block（主核，AIC）和 subblock（从核，AIV）。纯 Cube、纯 Vector 算子没有 subblock：启动的每个 block 就是一颗 AIC 或一颗 AIV。

一次 Kernel 启动对应一组 group：

- `block_dim`（就是 `kernel[block_dim](...)` 方括号里的数）决定启动多少个 group。
- [`get_block_idx()`](/zh/api/kernel/system/get-block-idx.html) 返回当前 group 编号，[`get_block_num()`](/zh/api/kernel/system/get-block-num.html) 返回总数。
- [`get_subblock_id()`](/zh/api/kernel/system/get-subblock-id.html) 在 AIV 上从 0 开始编号，在 AIC 上取值为 0。

算子按计算特征分三类，group 配置也随之不同：

| 算子类型 | 执行核 | block / subblock | 1：N |
| --- | --- | --- | --- |
| Cube 算子 | 仅 AIC | AIC 为 block，无 subblock | 不涉及 |
| Vector 算子 | 仅 AIV | AIV 为 block，无 subblock | 不涉及 |
| Mix 算子 | AIC 与 AIV | AIC 为 block，AIV 为 subblock | 1：1 |
| Mix 算子 | AIC 与 AIV | AIC 为 block，AIV 为 subblock | 1：2 |

1：2 配置下，一个 AIC 配两个 AIV，所以样例里常见这样的写法把逻辑 AIV 编号算出来：

```python
aiv = get_block_idx() * 2 + get_subblock_id()
```

### 多核切分的基本写法

几乎所有算子的第一步都是把输出切成 tile，再把 tile 分给各个 block：

```python
bi = get_block_idx()
bn = get_block_num()

for tile_idx in range(bi, total_tiles, bn):
    ...  # 本核负责的 tile
```

这是最简单的轮转分配。`matmul` 样例在此基础上做了滑动窗口 + 行反转，让相邻核共享 B 矩阵的同一行，提升 L2 命中率；`matmul_streamk` 进一步做了 DP + Split-K 混合调度。这些都属于[高性能算子编写指南](/zh/guide/programming-model/performance)的内容。

## 数据放在哪：存储层级

这是编程模型里最重要的一张表。

| 层级 | `MemLoc` | 容量 | 位置 | 典型用途 | 默认格式 |
| --- | --- | --- | --- | --- | --- |
| GM | — | 128 / 144 GB | 片外全局内存 | Kernel 的输入输出、workspace | ND |
| L2 Cache | — | 最大 128 MB | 多核共享缓存 | 自动生效，可用 `l2_cache_ctl` 调策略 | — |
| UB | `MemLoc.UB` | **256 KB / AIV** | AI Core 内，Vector 使用 | 矢量计算的数据暂存 | ND |
| L1 | `MemLoc.L1` | **512 KB / AIC** | AI Core 内，Cube 数据通路 | 缓存矩阵计算输入 | NZ |
| L0A | `MemLoc.L0A` | 64 KB | Cube 左输入端 | 矩阵乘的左矩阵 | NZ |
| L0B | `MemLoc.L0B` | 64 KB | Cube 右输入端 | 矩阵乘的右矩阵 | NZ |
| L0C | `MemLoc.L0C` | 256 KB | Cube 输出端 | 矩阵计算结果与累加初值 | NZ |
| BIAS | `MemLoc.BIAS` | 4 KB | Cube 专用 | Bias（偏置）数据，只接受 32 位元素 | ND |
| FBUF | `MemLoc.FBUF` | 4 KB | Fixpipe 数据通路 | 量化参数等 | ND |
| SSBUF | `MemLoc.SSBUF` | — | Scalar 共享 Buffer | AIC 与 AIV 之间交换少量标量 | ND |

**上表这 8 个是 `Buffer` / `Channel` 可以申请的全部位置。** `MemLoc` 枚举本身还有 `GM`、`RegFile`、`L0A_MX`、`L0B_MX` 四个成员（共 12 个），但它们不用于片上申请：GM 由 Kernel 形参带入，`RegFile` 是 VF 内的寄存器，`L0A_MX` / `L0B_MX` 是 MX scale 的从属区（地址由数据地址推导）。L2 不在枚举里。

::: danger 「每 AI Core」和「每 AIV」不是一回事
一个 AI Core 含 **1 个 AIC 和 2 个 AIV**。所以：

- **UB 是 256 KB，这是「每个 AIV」的量。** 换算成「每个 AI Core」是 512 KB —— 官方白皮书按 AI Core 口径写（「Unified Buffer 512KB per AI Core」），架构规格按 AIV 口径写，两者都对，但差一倍。
- **L1 是 512 KB，这是「每个 AIC」的量**，同一个 AI Core 里的两个 AIV 共享它。

算 UB 预算时用 **256 KB** 这个量级。按 512 KB 去推 tile，会直接撑爆。
:::

::: danger `get_mem_size("ub")` 返回的是 248 KB，不是 256 KB —— 不要再减一次
这是 UB 预算最容易算错的地方。实际数值是：

| 口径 | 字节 | 说明 |
| --- | --- | --- |
| UB 物理地址空间 | 262144（256 KB） | 每个 AIV |
| **`get_mem_size("ub")` 返回** | **253952（248 KB）** | **顶部 8 KB 预留已经扣掉** |
| 其中 VF 寄存器溢出区 | 6 KB | 活跃矢量寄存器超限时溢出到这里 |
| 其中 native dump 区 | 2 KB | `dump_tensor` / `dump_reg` 用 |

所以 UB 预算的正确写法是 **直接拿 `get_mem_size("ub")` 当可用总量**，再减掉你自己的对齐余量：

```python
UB_SIZE = get_mem_size("ub")      # 253952，不是 262144
budget  = UB_SIZE - MY_MARGIN     # ✓
budget  = UB_SIZE - 8 * 1024      # ✗ 预留已经扣过了，再减就白白少 8 KB
```

样例里那个 `248 * 1024` 正是这个数，但**仍然不要硬编码**：预留量会随版本变化，`get_mem_size` 不会。
:::

::: warning SIMT 模式会再吃掉一块 UB
`vf(mode="simt")` 下会从 UB 划出一段作为 SIMT Data Cache，**这部分不会反映在 `get_mem_size("ub")` 里，要自己留**。实践中常见口径是 SIMD 按 248 KB、开 SIMT 后按 216 KB 算。混合使用两种模式时 UB 预算按最紧的那一侧算。

SIMT 的寄存器配额还与线程数挂钩（线程数越多、每线程寄存器越少），见[三类计算单元 · SIMT 模式](/zh/guide/programming-model/compute#simt-模式)。
:::

::: tip
片上 Buffer 容量在 Host 侧用 [`get_mem_size`](/zh/api/host/platform-profiling/get-mem-size.html) 查询，参数取 `"ub"`、`"l1"`、`"l0a"`、`"l0b"`、`"l0c"`、`"bt"`（BiasTable）、`"fb0"`（Fixpipe Buffer），大小写不敏感，返回字节数。核数用 [`get_platform_info`](/zh/api/host/platform-profiling/get-platform-info.html) 查询。能查到的一律不要硬编码。

**查不到的有两个：L2 和 SSBUF。** 传 `"l2"` / `"ssbuf"` 会抛 `ValueError`，不是返回 0。样例里 L2 仍写经验值 `128 * 1024 * 1024`。

dav-3510 上的典型返回值：

| 参数 | 字节 | |
| --- | --- | --- |
| `"ub"` | 253952 | 248 KB |
| `"l1"` | 524288 | 512 KB |
| `"l0a"` / `"l0b"` | 65536 | 64 KB |
| `"l0c"` | 262144 | 256 KB |
| `"bt"` / `"fb0"` | 4096 | 4 KB |

```python
L1_SIZE = get_mem_size("l1")
L0A_SIZE = get_mem_size("l0a")
L0C_SIZE = get_mem_size("l0c")
AIC_NUM = get_platform_info().cube_core_num
```
:::

### 两条数据通路

存储层级不是随便互通的，`mem_copy` 支持的方向是固定的。实际上只有两条主干通路：

```text
Cube 通路（矩阵计算）
   GM ──nd2nz──► L1 ──────► L0A ─┐
                              MMAD ──► L0C ──► GM / UB / L1
   GM ──nd2nz──► L1 ──────► L0B ─┘       （可随路量化 / ReLU / NZ2ND）
                 L1 ──────► BIAS

Vector 通路（矢量计算）
   GM ────────► UB ──vload──► 寄存器 ──计算──► 寄存器 ──vstore──► UB ────► GM
                                                              （可原子累加）

两条通路之间
   L1 ──► UB      L0C ──► UB      UB ──► L1
```

完整的方向与可选行为见 [`mem_copy` 的支持的搬运方向](/zh/api/kernel/data-movement/mem-copy.html)。记住两件事就够用了：

1. **矩阵乘的输入必须经过 L1 中转**，不能从 GM 直接进 L0A/L0B。
2. **矢量计算的真正操作数是寄存器，不是 UB**。UB 只是寄存器和 GM 之间的中转站。

### 为什么 Cube 需要分形格式

矩阵计算单元采用分块计算逻辑，例如可并行处理 16×16 的矩阵分片。在传统行主序或列主序布局中，读取一个 16×16 的矩形块需要访问多个不连续的内存地址逐行「收集」数据，访存效率很低。

分形存储格式通过数据重排，让每个 16×16 计算块在物理内存中连续存放，硬件一次连续读取即可装载整块数据。

命名上采用「大 Y 小 x」的记法：

- **大 Y（Z/N）**：分形矩阵之间的排列顺序（Z 为行主序，N 为列主序）。
- **小 x（z/n）**：分形矩阵内部元素的排列顺序（z 为行主序，n 为列主序）。

在 DSL 里你一般不需要手写分形排布，而是：

- 声明 L1/L0 的 `Channel` 时用默认的 `data_format="nz"`；
- 从 GM 搬进 L1 时用 `make_copy_engine(format_transform="nd2nz")` 做随路转换；
- 从 L0C 搬到 ND 的 GM/UB 时，可不写 `engine`：接口按源、目标布局选择搬运方式，官方示例和 `matmul` 样例都是直接 `mem_copy(gm, l0c)`。需要显式 NZ2ND、量化或 ReLU 时再用 `make_copy_engine`。

```python
nd2nz = make_copy_engine(format_transform="nd2nz")
mem_copy(l1_a.produce(), gm_a_tile, engine=nd2nz)
```

需要理解分形细节的场景（手写非常规排布、对齐调优）见[背景与核心概念](/zh/api/kernel/cube_compute/fractal-intro.html)和[关键分形格式](/zh/api/kernel/cube_compute/fractal-formats.html)。

### 切块时的对齐要求

`tile_slice` 对 NZ/ZN 布局切分时，内部 tile 边界必须满足分块对齐：

- **NZ**（代码中的 ND2NZ）：最后两个逻辑维度分别按 `16` 和 `32 / 元素字节数` 对齐。
- **ZN**（代码中的 DN2NZ）：交换上述两个维度的对齐粒度。
- 以 `coord=0` 覆盖完整逻辑轴时，不要求该轴的 `tiler` 本身是分块大小的整数倍。

这解释了为什么 `matmul` 的 tiling 里到处是 `ceil_align(x, 16)` 和 `BASIC_BLOCK_K_256B // dtype_size`。

## 谁跟谁同步

同步在本章只给一个结论，细节单独成章。

**结论是：核内「搬运与计算之间」的等待由框架按 `Channel` 的 `produce()` / `consume()` 配对自动插入，其余四类同步要你自己写。**

| 场景 | 谁负责 |
| --- | --- |
| 核内搬运与计算之间 | **框架**（`Channel`） |
| 核内跨流水、VF 内的 UB 重叠 | 你（`vmem_bar`、pipe 级同步） |
| 核间（AIC 之间、AIV 之间、AIC 与 AIV 之间） | 你（四种模式） |
| 跨核数据交接 | **框架**，但要你声明 `ChannelKind.CrossCore` |
| 多核经 Scalar 读写同一块 GM | 你（DCache 维护或 bypass） |

完整说明见[同步、Cache 与跨核交接](/zh/guide/programming-model/synchronization)。

## 从上一代过来最容易踩的五条

本章描述的是 3510。从 NPU ARCH 2201（Atlas A2 / A3）迁移时，下面五条会直接让代码编译不过或行为改变：

| 项 | 2201 | 3510 |
| --- | --- | --- |
| 矢量操作数来源 | UB（MemBase） | **寄存器（RegBase）**——整套矢量代码要重写 |
| `GM → L0A/L0B` | 支持 | **取消**，必须经 L1 两跳 |
| `L1 → GM` | 支持 | **取消** |
| 4:2 结构化稀疏、int4 矩阵乘 | 支持 | **硬件取消**，没有替代接口 |
| AIC↔AIV 标量交接 | 经 GM 往返 | **SSBUF**，不用再绕 GM |

容量、分形、同步模式、低精度格式等**完整的跨代对照表见[附录第一节](/zh/guide/programming-model/appendix#跨代变更速查-npu-arch-2201-3510)**，本章不再重复一遍。

## 硬件概念到 DSL 接口的速查

| 你想做的事 | 接口 |
| --- | --- |
| 知道自己是第几个核 | `get_block_idx()` / `get_block_num()` |
| 知道自己是哪个 AIV | `get_subblock_id()` / `get_subblock_dim()`（AIC 上分别为 0 和 1，AIV 上为 0/1 和 2） |
| 知道当前编译的是 AIC 还是 AIV 侧 | `is_aic()` / `is_aiv()`，**编译期常量；MIX 编译下两者都为 false** |
| 申请一块片上存储 | `Buffer(MemLoc.UB, shape, dtype)` |
| 申请一组可轮转的片上槽位 | `Channel(MemLoc.UB, shape, dtype, depth=N)` |
| 在存储层级之间搬数据 | `mem_copy(dst, src, engine=...)` |
| 配置格式转换 / 分区 / 转置 | `make_copy_engine(...)` |
| 取一个 tile 的视图 | `tile_slice(tensor, tiler, coord)` |
| 做矩阵乘 | `matmul(l0c, l0a, l0b, init=...)` |
| 做矢量计算（SIMD） | `with vf(mode="simd"):` 内的 `v*` 接口 |
| 做矢量计算（SIMT） | 混合模式用 `vf(mode="simt", thread=N)`，纯 SIMT 用 `vf(mode="simt")`；作用域内调用 `cbd.simt.*`，见[两种编程模型](/zh/guide/programming-model/compute#两种-simt-编程模型) |
| 核间同步 | `*_sync_block_*` / `*_sync_intra_*` / `global_sync_all()` |
| 查硬件参数 | `get_platform_info()` / `get_mem_size()` |
| 查当前周期数（性能分析） | `get_system_cycle()` |

## 下一步

硬件模型就绪后，下一个要理解的是「我写的 Python 什么时候执行」：[代码生成](/zh/guide/programming-model/code-generation)。
