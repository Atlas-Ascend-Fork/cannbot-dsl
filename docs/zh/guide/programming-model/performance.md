# 高性能算子编写指南

能跑通和跑得快之间通常差 3～10 倍。这一节给出一套可执行的优化顺序：先判断瓶颈类型，再按五个层次依次检查。

## 第一步：判断瓶颈类型

别一上来就调 tile 大小。先用纸笔算清楚这个算子**理论上**受什么限制。

| 类型 | 判断方法 | 优化重点 |
| --- | --- | --- |
| **访存受限** | 计算访存比低（elementwise、归一化、小 batch GEMV） | 搬运效率、流水重叠、L2 复用 |
| **Cube 受限** | 大矩阵乘，MAC 利用率是上限 | tile 大小、L0C 复用、K 方向累加、UnitFlag |
| **Vector 受限** | 长依赖链的逐元素计算、复杂归约 | VF 内保持寄存器驻留、指令双发、融合接口 |
| **调度受限** | tile 数不能被核数整除、长尾 | 分核策略、Stream-K、滑动窗口 |
| **同步受限** | 频繁的核间同步、跨核交接 | 减少同步次数、加深 Channel、拆分 VF |

算一下**计算访存比**（FLOP / Byte）：向量加是 1/12（每搬 12 字节做 1 次加法），矩阵乘是 O(tile 边长)。前者注定访存受限，再怎么调计算都没用。

### roofline：把「理论上限」算成一个数

需要两个硬件峰值，取自[硬件与执行模型](/zh/guide/programming-model/hardware-model)的规格表。以 950PR 的 32 核档为例：

| 量 | 值 |
| --- | --- |
| 内存带宽 \(B\) | 1.6 TB/s |
| Cube 算力 \(P\)（BF16，**仅 Cube**） | 432 TFLOPS |

拐点（ridge point）是两者的比值：

$$
I^{*} = \frac{P}{B} = \frac{432 \times 10^{12}}{1.6 \times 10^{12}} = 270 \ \text{FLOP/Byte}
$$

判据就一句：**算子的计算访存比 \(I\) 小于 270 就是访存受限，大于就是计算受限。**

- 向量加 \(I = 1/12\)，远小于 270 → 访存受限，上限是 \(B\)。1 GB 的数据搬运至少要 \(1/1600\) 秒 ≈ 625 µs。
- \(M=N=K=4096\) 的 BF16 矩阵乘：\(2MNK = 1.37 \times 10^{11}\) FLOP，访存 \(3 \times 4096^2 \times 2 = 100\) MB，\(I \approx 1365\) → 计算受限，上限是 \(P\)，理论耗时 \(1.37\times10^{11} / 4.32\times10^{14} \approx 318\) µs。

::: danger 三个容易把上限算错的地方
**一、用「仅 Cube」那一列。** 对外宣传的 1 PFLOPS / 2 PFLOPS 是 Cube + Vector 的合计值。拿它当 cube-bound 算子的上限会把目标定高多少，**取决于精度**：MXFP4 是 3.1%，HiF8/MXFP8/FP8 是 6.2%，而 **BF16/FP16 和 TF32 都是 12.5%**。换句话说，最常用的那档偏差最大。各档数字见[硬件与执行模型](/zh/guide/programming-model/hardware-model#规格速查)。

**二、核数和容量是分档的。** 950PR 有 32 核和 28 核两档，950DT 有 36 / 32 / 28 三档，最高档与最低档差 22%。一定要用 `get_platform_info().cube_core_num` 查，别拿白皮书的最高档去算。

**三、降精度会改变拐点。** 换到 MXFP4 后 \(P\) 变成 4 倍，\(I^{*}\) 也跟着变成 4 倍 —— 一个原本计算受限的算子可能因此变成访存受限，优化重点随之改变。
:::

::: info 为什么不用主频推算
官方从未公布 AI Core 主频，白皮书的所有吞吐对比都以「同频」为前提。所以不要试图从主频反推单核吞吐，直接用上面的整芯片峰值除以核数即可。
:::

### 达成率怎么看

$$
\text{达成率} = \frac{\text{理论耗时}}{\text{实测 Task Duration}}
$$

| 达成率 | 判断 |
| --- | --- |
| > 80% | 基本到顶，继续调的收益有限 |
| 50% ～ 80% | 还有空间，按下面五个层次逐层查 |
| < 50% | 大概率有结构性问题（分核不均、流水没重叠、tile 太小），不要先去抠指令 |

## 层次 1：多核切分与负载均衡

### 基本轮转

```python
bi, bn = get_block_idx(), get_block_num()
for tile_idx in range(bi, total_tiles, bn):
    ...
```

### 别启动用不上的核

`total_tiles < block_dim` 时，多余的核只会空转并拖慢启动。Host 侧把 `block_dim` 收敛到实际需要的数量：

```python
m_core = ceil_div(self.m, self.base_m)
n_core = ceil_div(self.n, self.base_n)
self.used_core_num = min(m_core * n_core, self.AIC_NUM)
```

### 滑动窗口：让相邻核共享数据

朴素的行优先遍历下，相邻核处理的 tile 在 N 方向相邻，共享的是 A 矩阵的同一行。`matmul` 样例用**滑动窗口 + 行反转**改善 L2 命中：

```python
WINDOW_LEN = 4
...
row_idx = tile_idx // n_tiles // main_window
if row_idx < main_row:
    m_idx = row_idx * main_window + tile_idx % main_window
    n_idx = (tile_idx // main_window) % n_tiles
else:
    ...
# 奇数行反向遍历 N，使换行时的下一个 tile 与上一个相邻
n_idx = (n_tiles - 1 - n_idx) if (row_idx % 2 != 0) else n_idx
```

### Stream-K：解决长尾

当 tile 数不能被核数整除时，最后一轮只有少数核在干活。Stream-K 的思路是把尾部 tile 按 K 维切开，让更多核参与：

- **DP tiles**（能被核数整除的部分）：各核独立计算完整 K，结果直接写回。
- **SK tiles**（尾部）：按 K 维 split，多核各算一段 K，部分和写入 workspace GM。
- **AIV reduce**：AIV 从 workspace 读各段部分和，fp32 累加后 cast 回输出 dtype。

`samples/matmul/matmul/matmul_streamk.py` 是完整实现。在小 MN、大 K 场景（MN tile 数 ≤ 16、K ≥ 8192）效果最明显。

### 负载均衡率

`matmul` 的 tiling 里有一个 `_get_balance_rate_with_tail`，量化「有效计算量 / 实际占用的核时间」。搜索 tile 大小时把它和 cube-bound 指标一起作为目标函数。这是一个值得借鉴的方法论：**把调度质量变成一个可以被搜索的数字。**

## 层次 2：Tiling

tile 大小是最重要的单个参数。它同时受四个约束：

```text
┌─ L0C 容量 ──── base_m × base_n × 4 字节 × l0c_db ≤ L0C_SIZE
├─ L0A 容量 ──── base_m × base_k × dtype_size × 2 ≤ L0A_SIZE
├─ L1 容量 ───── (base_m + base_n) × k_l1 × dtype_size × l1_buffer_num ≤ L1_SIZE
└─ 分形对齐 ──── M、N 对齐 16；K 对齐 K0 = 32B / dtype_size
```

### 在 Host 侧推导，不在 Kernel 里

把 tiling 写成一个独立的 Host 侧类，输出一组纯数值，再喂给 `@kernel` 类的 `__init__`。`MatmulTiling` 的流程是：

```text
_reset_base()      设定默认 base_m / base_n / base_k
      ↓
_rebalance_block() 按 cube-bound 边界 + 负载均衡率搜索最优 base_m × base_n
      ↓
_cal_l1_tiling()   在 L1 预算内决定 step_ka / step_kb
      ↓
_finalize()        算出 m_l1 / n_l1 / k_l1、used_core_num、l1_buffer_num、l0c_db
```

### 从硬件查容量，不要硬编码

```python
L1_SIZE = get_mem_size("l1")
L0A_SIZE = get_mem_size("l0a")
L0C_SIZE = get_mem_size("l0c")
AIC_NUM = get_platform_info().cube_core_num
```

### Vector 侧的 UB 预算

矢量算子的 tile 大小由 UB 预算倒推：分子是可用 UB 总量，分母是「每处理一行需要多少字节」，商就是一次能处理多少行。**所有缓冲都要算进去**，包括 double buffer 的倍数和临时量。

完整公式与 `rms_norm` 的写法见[片上存储、流水与 GM 协作 · 先算预算](/zh/guide/programming-model/onchip-memory#先算预算)，这里只强调两条最容易错的：

- `UB_SIZE` 必须用 `get_mem_size("ub")` 查，**它返回的 253952（248 KB）已经扣掉了顶部 8 KB 预留**，不要再减一次。
- 分母漏算临时量比分母算大更危险：算大了只是 tile 偏小，漏算了会在设备上溢出。

### 准备多套实现

同一个算子在不同 shape 下最优策略可能完全不同。`rms_norm` 准备了两套 Kernel：

```python
if const_expr(self._is_row_full_load(num_col, self.dtype)):
    op = RowFullLoadKernel(num_col, self.dtype)   # UB 放得下一整行
else:
    op = ColSplitKernel(num_col, self.dtype)      # 按列切分，两趟遍历
```

`RowFullLoadKernel` 内部还按列数分了四个归约分支（单 VL / 双 VL / 折半归约 / 两级折半）。

## 层次 3：流水并行

### double buffer 是起点

Channel `depth=2` + 循环外预取一拍，见[写出第一个算子 Step 3](/zh/guide/programming-model/first-operator#step-3-加上-double-buffer)。

### 按预算自适应加深

```python
self.l1_buffer_num = (
    self.DB_SIZE                       # 2
    if (a_l1_4buf + b_l1_4buf + bias_4buf) > self.L1_SIZE
    else self.BASIC_L1_BUFFER_NUM      # 4
)
self.l0c_db = (
    self.DB_SIZE
    if self.base_m * self.base_n * 4 * self.DB_SIZE <= self.L0C_SIZE
    else 1
)
```

### depth 不等于流水

再强调一次：`depth` 只提供容量和在途重叠能力。**真正的流水由程序顺序决定。** 深度流水时用 [`DelayLineGroup`](/zh/api/kernel/types-and-views/delay-line-group.html) 记住每一拍的 tile 坐标。

### UnitFlag：乘加与搬出的细粒度并行

开启后硬件每算完一个分形就搬出结果，不用等整块算完：

```python
matmul(l0c, l0a, l0b, init=..., unit_flag=2)   # 中间指令：ENABLE_KEEP
matmul(l0c, l0a, l0b, init=..., unit_flag=3)   # 最后一条：ENABLE_UPDATE
```

乘加指令与对应的搬出指令必须**同时**开启或同时不开启，开启后两者之间无需再插同步。方向必须一致：搬出开启 NZ2ND 随路转换或 B8/B4 量化触发 Channel Merge 时用 `cube.set_mmad_direction("n")`，其他用 `"m"`。

### 减少同步次数

- VF 内包含同步时，切分循环把同步外提。
- 跨核交接优先用 `ChannelKind.CrossCore` 的 Channel，而不是手写 `*_sync_*` + GM 中转。
- `global_sync_all()` 是全核屏障，代价不低，不要放在内层循环里。

## 层次 4：搬运效率

### 用随路转换，别单独算一遍

搬运单元可以在搬运过程中顺便做格式转换、量化、激活，**不占用计算单元**：

| 想做的事 | 随路写法 |
| --- | --- |
| ND → NZ | `engine=make_copy_engine(format_transform="nd2nz")` |
| 转置加载进 L0A/L0B | `mem_copy(l0a, l1_slice, transpose=True)` |
| L0C 搬出时反量化 | `mem_copy(dst, l0c, deq_scale=...)` |
| L0C 搬出时 ReLU | `engine=make_copy_engine(relu=True)` |
| 搬出时原子累加 | `mem_copy(gm, ub, atomic_add=True)` |
| NZ → ND | `engine=make_copy_engine(format_transform="nz2nd")` |

### L2 Cache 策略

大矩阵乘下，不被复用的矩阵占着 L2 反而会挤掉被复用的矩阵。`matmul` 按复用情况自适应开关：

```python
# A 在 base_n 覆盖整个 N 时不被复用 → 关掉它的 L2
left_not_l2_cache = (
    self.base_n >= self.n and n_cnt <= 1
    and inner_a * self.a_dtype_size % self.ALIGN_128 == 0 and flag_a
)
...
mem_copy(l1_a.produce(), gm_a_tile, engine=nd2nz_engine_a,
         l2_cache_ctl=self.l2_cache_ctl_a)
```

`l2_cache_ctl` 取值 0、1、2、4，仅支持 GM→UB、UB→GM、GM→L1 和 L0C→GM。

### 对齐

- **128 字节**：`matmul` 的 L2 策略判断用 `% 128 == 0` 作为「搬运友好」的条件。
- **32 字节**：`vload` / `vstore` 要求 UB 起始地址 32 字节对齐；非对齐要用 `vload_unalign` / `vstore_unalign_post`，有额外代价。
- **分形对齐**：NZ 最后两维按 16 和 `32 / 元素字节数` 对齐。
- **Cube 地址对齐**：L0C 1024 字节、L0A/L0B 512 字节、BiasTable 64 字节。

对不齐的时候优先改 tile 大小去迁就对齐，而不是用非对齐接口硬扛。

### NDDMA 与 padding

GM→UB 的多维非连续搬运用 NDDMA（1～5 维，源和目标维度相同，dtype 一致）。需要边界填充时直接在搬运时做：

```python
engine = make_copy_engine(kind="nddma", padding_mode="constant")
mem_copy(slot, src, engine=engine,
         left_padding=(8,), right_padding=(8,), pad_value=-1.0)
```

比「搬进来再写零」少一趟。

### 多源 / 多目标搬运

- 一个 ND GM 源广播到多个 UB 或 L1 目标：`dst` 传序列。
- 多个同存储源按顺序拼进一个目标：`src` 传序列，仅支持 `axis=0`。

这两种模式能把多次 `mem_copy` 合并成一次，减少指令数和同步点。

### 分区搬运

1:2 配置下让两个 AIV 各搬一半：

```python
engine = make_copy_engine(split_axis=0)
mem_copy(dst, src, engine=engine, part_id=get_subblock_id())
```

## 层次 5：计算效率

### 让数据留在寄存器里

Reg 编程模型的全部价值就在这里。一条计算链上的中间结果不要往 UB 写：

```python
# ✗ 每步都落 UB
with vf(mode="simd"):
    t1 = vmul(vload(x, 0), vload(y, 0), mask=m)
    vstore(tmp, 0, t1, m)
with vf(mode="simd"):
    t2 = vadd(vload(tmp, 0), vload(z, 0), mask=m)
    vstore(out, 0, t2, m)

# ✓ 全程寄存器
with vf(mode="simd"):
    t1 = vmul(vload(x, 0), vload(y, 0), mask=m)
    t2 = vadd(t1, vload(z, 0), mask=m)
    vstore(out, 0, t2, m)
```

### 用融合接口

| 想算 | 别写 | 用 |
| --- | --- | --- |
| `a * b + c` | `vadd(vmul(a,b), c)` | `vmadd` |
| `alpha * x + y` | `vadd(vmuls(x,alpha), y)` | `vaxpy` |
| `exp(x - m)` | `vexp(vsub(x, m))` | `vexp_sub` |
| `cast(x); exp(x - m)` | 两步 | `vcast_exp_sub` |
| `|a - b|` | `vabs(vsub(a,b))` | `vabs_sub` |
| ReLU / LeakyReLU / PReLU | 比较 + 选择 | `vrelu` / `vleakyrelu` / `vprelu` |

### 搬入搬出时顺便换类型

bf16/fp16 数据按 fp32 精度计算时，用 `vload_unpack` 在搬入时展开、`vstore_pack` 在搬出时压回，比「搬入 → `vcast` → 算 → `vcast` → 搬出」少两步：

```python
xu = vload_unpack(x_ch, off, unpack_mode=UnpackMode.B16_TO_B32)
x = vcast(xu, dtypes.float32, mask=full)
...
vstore_pack(y_buf, off, vcast(yval, out_dtype, mask=full, rounding=RoundingMode.RN),
            mask, pack_mode=PackMode.B32_TO_B16)
```

### 满足 Hardware Loop 规范

VF 内的循环满足规范才会被优化成硬件循环：从 0 起步长 1、循环内无运行时跳转、边界不可变。硬件侧计数器是 `uint16_t`，Python 里不用也不该手写这个类型。**循环体里的 `if`/`else` 和三元表达式会阻碍 Hardware Loop 生成**——能用 `const_expr` 消掉的就消掉，不能消的尽量提到循环外。

### 用 `unroll` 提升指令双发

依赖链过长导致指令无法并发时：

```python
for i in cannbotdsl.range(n, unroll=4):
    ...
```

但 VF 不是越长越好：指令过多会触发 ICache Miss，这时反而要切分循环、把中间结果搬出 UB 来减少依赖。

### 降精度

| 手段 | 接口 |
| --- | --- |
| fp32 → HF32 | `enable_hf32`、`set_hf32_round_mode`、`set_fp32_mode` |
| FP8 矩阵乘 | `enable_fp8` |
| HiFloat8 | `enable_hif8` |
| MX 块量化（MXFP8 / MXFP4） | `mem_copy(..., mx_scale=...)` + 对应 engine |

仓库里 `matmul/quant_matmul` 覆盖了 MXFP8 / MXFP4 全量化、per-tensor HiFloat8 / INT8 / FP8，以及 MXA8W4 混合量化三条路径。

### 别拿 SIMT 当优化手段

`vf(mode="simt")` 是为**编程便利**准备的，不是为性能准备的。SIMD 侧有双发 ALU 和乱序执行，单位周期吞吐更高；SIMT 的收益是让不规则控制流和 gather/scatter 好写。

两个具体的性能陷阱：

- **SIMT 会从 UB 划走一块 Data Cache**，挤压你的 tile 预算。
- **线程数越多，每线程寄存器越少**（总寄存器文件容量固定）。计算密集的 SIMT kernel 在大线程数下容易溢出到栈，而栈在 Global Memory 上——一次溢出就把寄存器驻留的收益抹平了。遇到这种情况要**降线程数**，不是加线程数。

所以优化顺序是：先用 SIMD 写，撞到控制流复杂度墙了，再把**那一段** VF 换成 SIMT。见[三类计算单元 · SIMT 模式](/zh/guide/programming-model/compute#simt-模式)。

## 常见反模式

| 反模式 | 后果 | 改法 |
| --- | --- | --- |
| 用 Tensor 下标循环搬批量数据 | 走标量流水，慢几个数量级 | `mem_copy` + 矢量接口 |
| 每个 tile 一次 `mem_copy` 且 tile 很小 | 搬运指令数爆炸，带宽打不满 | 加大 tile，或合并成多源/多目标搬运 |
| `depth=1` 的 Channel 做主循环缓冲 | 搬运和计算完全串行 | `depth>=2` + 循环外预取 |
| 内层循环里写运行时 `if` | 阻碍 Hardware Loop 和指令双发 | 提到循环外，或用 `const_expr` / mask 替代 |
| 中间结果反复写回 UB | 浪费 UB 带宽 | 整条链留在 VF 内的寄存器上 |
| 在 VF 内调 `produce()` / `consume()` | 不支持 | 进 VF 前取好 Tensor |
| 硬编码 UB / L1 容量 | 换硬件就错 | `get_mem_size()` |
| `block_dim` 固定为最大核数 | tile 少时空转 | `min(需要的核数, 可用核数)` |
| 先搬运再单独写一遍量化/ReLU | 多一趟计算 | 用随路转换 |
| 全展开大循环 | 编译慢、ICache Miss | 只对小循环用 `range_constexpr` |

## 调优流程

```text
① 写出正确版本，跑通精度测试
      ↓
② 算理论上限（带宽 or MAC），确定瓶颈类型
      ↓
③ 用 msprof 采集 Task Duration，和理论上限比
      ↓
④ 按层次 1→5 的顺序检查，每次只改一个变量
      ↓
⑤ 每次改动后重跑精度测试
      ↓
⑥ 达标后把 tiling 参数化，覆盖 shape 范围做扫描
```

第 ⑥ 步很重要：在一两个 shape 上调出来的参数，换个 shape 往往就不是最优。把 tiling 写成能自动推导的函数（而不是写死的常量），再用一批代表性 shape 做扫描验证。

### 测量规范

同一份代码测两次差 20% 很常见，所以测量方式必须固定下来，否则调优就是在噪声里找信号：

| 项 | 做法 |
| --- | --- |
| 预热 | 先调用 5～10 次，确保 JIT 编译完成、频率爬升到位 |
| 采样次数 | 100 次以上；小算子取 1000 次 |
| 计时方式 | 优先 msprof 的 Task Duration；Python 侧计时必须把 `torch.npu.synchronize()` 放在循环**外** |
| 取值 | **取最小值**，不是平均值 —— 最小值最接近无干扰时的真实耗时 |
| 去离群 | 丢掉明显偏大的样本（通常是被调度打断） |
| 关闭调试 | 测量前必须去掉所有 `cbd.print`，它会触发 Host 回读与同步 |
| 环境 | 固定设备、固定并发，避免与其他进程抢核 |

仓库样例的口径是「msprof 采集 Task Duration，每 shape 重复 10 次取 min」，对比性能时按同一口径才有意义。本节是测量规范的权威出处，[调试与精度验证](/zh/guide/programming-model/debugging#性能分析)一章只给简版。

## 参考样例

| 想学什么 | 看哪个 |
| --- | --- |
| Host 侧 tiling 的完整方法论 | `samples/matmul/matmul/matmul.py` 的 `MatmulTiling` |
| 分核调度与长尾处理 | `samples/matmul/matmul/matmul_streamk.py` |
| UB 预算推导 + 多实现分支 | `samples/rms_norm/rms_norm.py` |
| Cube + Vector 混合流水 | `samples/flash_attn/`、`samples/flash_mla_with_kvcache/` |
| AI CPU 调度 + 稀疏访问 | `samples/qwen_sparse_attn/`、`samples/qsa_indexer/` |
| 量化矩阵乘 | `samples/matmul/quant_matmul/` |
| 深度软件流水 + 跨核 Channel | `samples/flash_kda/`、`samples/mixed_quant_sparse_flash_mla/` |

各样例目录下的 README 带有性能对比图和测试条件，是判断「我的算子算不算快」的参照。

## 下一步

[调试与精度验证](/zh/guide/programming-model/debugging)。
