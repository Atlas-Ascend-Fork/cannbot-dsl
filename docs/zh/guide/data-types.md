# 数据类型与量化

Ascend 950 这一代最大的变化不在存储层级，而在**数值格式**：Cube 原生支持 HiF8 与 MX 微缩放格式，低精度矩阵乘的吞吐是 FP16 的 2 到 4 倍。仓库里一半样例是量化算子，所以这一章要讲清三件事：**有哪些格式**、**scale 存在哪里**、**谁负责把 scale 搬进去**。

::: info 本章的核实口径
格式层面的事实（位宽、块长、分形、对齐）来自昇腾官方架构资料，可以照用。**接口层面的类型拼写**（即 `cbd.dtypes` 下的具体名字）请以 [API 接口清单](../api/language/decorators.md)为准——本章在表格里同时给出 Ascend C 的类型名，便于交叉对照。
:::

## 为什么单独一章

前面几章把数据当成「有 dtype 的字节」。到了量化算子，这个抽象不够用了：

- 一个 MXFP8 张量**不是一个张量**，是「数据张量 + scale 张量」两份存储，它们的形状不同、分形不同、对齐要求也不同。
- scale 不是你算完再乘上去的后处理，而是**由搬运单元在 GM→L1→L0 的路上和数据一起搬进去**，由 Cube 在乘加时直接消费。
- 同样叫「8 位浮点」，HiF8 和 FP8 的编码方式、是否需要 scale、能不能混用，全都不一样。

搞错任何一条，结果都是「能跑、数值错」。

## 数值格式总览

| 格式 | 位宽 | Ascend C 类型名 | 需要 scale | Cube 支持 | 典型用途 |
| --- | --- | --- | :---: | :---: | --- |
| FP32 | 32 | `float` | — | ✓ | 累加、归约、参考实现 |
| HF32（TF32 档） | 32 存储 / 降精度计算 | — | — | ✓ | 用精度换 FP32 矩阵乘吞吐 |
| BF16 | 16 | `bfloat16_t` | — | ✓ | 训练与推理的主力 |
| FP16 | 16 | `half` | — | ✓ | 推理主力 |
| INT8 | 8 | `int8_t` | per-tensor / per-channel | ✓ | 传统量化 |
| **HiF8** | 8 | `hifloat8_t` | **不需要** | ✓ | 接近 FP16 精度的 8 位浮点 |
| FP8 E4M3 | 8 | `fp8_e4m3fn_t` | per-tensor 或 MX | ✓ | 标准 FP8 |
| FP8 E5M2 | 8 | `fp8_e5m2_t` | per-tensor 或 MX | ✓ | 标准 FP8，动态范围更大 |
| **MXFP8** | 8 + 8/32 | 上两者 + `fp8_e8m0_t` | **每 32 元素一个** | ✓ | 块量化，内存减半 |
| **MXFP4** | 4（双元素打包）+ 8/32 | `fp4x2_e2m1_t` / `fp4x2_e1m2_t` + `fp8_e8m0_t` | **每 32 元素一个** | ✓ | 块量化，吞吐 4× FP16 |
| E8M0 | 8 | `fp8_e8m0_t` | — | 作为 scale | MX 的缩放因子，纯指数 |

吞吐关系（同频）：**HiF8 / MXFP8 / FP8 约为 FP16 的 2 倍，MXFP4 约为 4 倍。** 这是选型的首要依据。

::: tip 先看能不能降精度，再看怎么调 tile
一个 cube-bound 的矩阵乘，从 BF16 换到 MXFP4 的收益上限是 4 倍；把 tile 从次优调到最优，收益通常在 10%～30%。顺序别搞反。
:::

## HiF8：不需要 scale 的 8 位浮点

HiF8 是华为自研格式，设计目标是「保持 FP8 的效率，精度接近 FP16」。它和 FP8 E4M3/E5M2 的根本区别是：**指数位宽不固定**。

每个 HiF8 数值的高位是一段变长的**前缀（dot field）**，由它决定剩下的位怎么在指数和尾数之间分配：

| dot field | 指数位宽 | 尾数位宽 |
| --- | :---: | :---: |
| `0000` | 0（非规格化） | 3 |
| `0001` | 0 | 3 |
| `001` | 1 | 3 |
| `01` | 2 | 3 |
| `10` | 3 | 2 |
| `11` | 4 | 1 |

含义是：**绝对值小的数给足尾数位，绝对值大的数牺牲尾数换指数范围。** 配合特殊的非规格化设计，HiF8 的指数范围达到 38 个二进制档（FP16 是 40 档，FP8 E4M3 只有 18 档）。

特殊值只有四个：

| 值 | 编码 |
| --- | --- |
| 零 | `0000_0000` |
| NaN | `1000_0000` |
| +∞ | `0110_1111` |
| −∞ | `1110_1111` |

::: warning HiF8 只有一个零
没有负零。依赖符号零区分分支的算法需要改写。
:::

对写算子的人来说，HiF8 的实际意义只有两句：

1. **它是标量类型，不带任何额外元数据。** 不需要 scale 张量，不需要额外搬运，`matmul` 直接吃。
2. **A 和 B 必须同为 HiF8**，不能一边 HiF8 一边 FP8。而 `fp8_e4m3fn` 与 `fp8_e5m2` 之间是可以混的。

开启方式见 [`enable_hif8`](../api/compute/matrix/enable-hif8.md)。

## MX 微缩放：块量化及其布局

MXFP8 / MXFP4 遵循 OCP Microscaling 规范，核心是**沿 K 轴每 32 个元素共享一个缩放因子**：

$$
C_{M \times N} = (s_A \otimes A)_{M \times K} \times (s_B \otimes B)_{K \times N} + \text{Bias}
$$

- 数据块长固定为 **32**。
- scale 类型固定为 `fp8_e8m0_t`：**8 位纯指数，无符号位、无尾数**，相当于 bfloat16 砍掉符号和尾数。所以缩放只能是 2 的整数次幂。
- FP4 元素**两个打包进一个字节**（类型名里的 `x2` 就是这个意思），因此内层轴长度必须是偶数。

### scale 的逻辑形状

按块长 32 推导，`scaleA` 是 \(M \times K/32\)，`scaleB` 是 \(K/32 \times N\)。但**实际接口上的形状不是这个**，原因在分形：

- `scaleA` 的分形是 **16×2** 个 `fp8_e8m0_t`，也就是一个分形横跨 **两个** 32 元素块。
- `scaleB` 的分形是 **2×16**。

于是 K 方向的有效对齐粒度变成 **64**，scale 的 ND 形状写成：

```text
scaleA : [M, Ceil(K / 64), 2]
scaleB : [Ceil(K / 64), N, 2]
```

末维那个 `2` 就是「相邻两个 scale 组成一对」。仓库 `quant_matmul` 的公开接口正是这个形状。

::: danger K 不是 64 的倍数时要自己补零
当 A 与 B 都在 GM 上、且 `Ceil(K, 32)` 为**奇数**时，最后一个 scale 对只有一半是有效数据。硬件不会替你处理，**调用方必须把 scale 张量在 K 方向补零到偶数个块**。漏掉这一步的症状是最后一个 K 块的结果偏大或变成随机值。
:::

### scale 在片上的位置与分形布局

scale 和数据走**同一条两跳通路**，但在 L1 上的布局要求不同：

```text
GM ──mem_copy(engine=mx_scale_*)──► L1 ──mem_copy──► L0A/L0B 的 MX scale 区
     （数据走 nd2nz，scale 走 mx_scale_*）
```

- L1 上 `scaleA` 要求 **Zz** 布局（外层按行、内层按行），`scaleB` 要求 **Nn** 布局（外层按列、内层按列）。原因是 MX 加载指令对 A 按行读、对 B 按列读。
- 数据以 512 字节分形搬运，scale 以 **32 字节分形**搬运。
- L0 侧 scale 的地址由数据地址自动推导，不需要你单独规划。

### 怎么把 scale 搬进去

两个入口，配合使用：

| 接口 | 作用 |
| --- | --- |
| [`copy_options(mx_scale=l1_scale)`](../api/data-movement/copy-options.md) | 为 L1→L0A/L0B 搬运指定配套的 L1 缩放因子 Tensor；布局及形状要求见参数说明 |
| `mem_copy(l0_data, l1_data, options=options)` | 使用上述配置，将数据及配套缩放因子加载到目标 |

::: warning 接口版本说明
下文的 `format_transform`、`make_copy_engine` 和 `mem_copy(..., mx_scale=...)` 为历史写法，尚未迁移，不适用于当前源码。当前配置入口及可执行示例以 [copy_options](../api/data-movement/copy-options.md) 为准；不能将旧的 `format_transform` 参数直接传给 `copy_options`。
:::

`format_transform` 的 MX 取值共六个：

| 取值 | 含义 |
| --- | --- |
| `"mx_scale_a"` | A 侧 scale，按默认布局 |
| `"mx_scale_b"` | B 侧 scale，按默认布局 |
| `"mx_scale_and"` | A 侧 scale，源为 ND |
| `"mx_scale_adn"` | A 侧 scale，源为 DN |
| `"mx_scale_bnd"` | B 侧 scale，源为 ND |
| `"mx_scale_bdn"` | B 侧 scale，源为 DN |

`mem_copy` 的方向表里也写明了这两处：**GM→L1 支持 MX scale 格式转换**，**L1→L0A/L0B 支持 MX scale 加载**。

```python
nd2nz = make_copy_engine(format_transform="nd2nz")
sa_eng = make_copy_engine(format_transform="mx_scale_and")

# 数据与 scale 分别搬入 L1
mem_copy(l1_a.produce(), gm_a_tile, engine=nd2nz)
mem_copy(l1_sa.produce(), gm_sa_tile, engine=sa_eng)

# 搬入 L0A 时把两者关联
mem_copy(l0a.produce(), l1_a.consume(), mx_scale=l1_sa.consume())
```

::: info 待核实
上面的代码是按 `mem_copy` 与 `make_copy_engine` 的参数语义组织的示意，用于说明**两类搬运的配对关系**。落到具体算子时，参数的确切组合请以 `samples/matmul/quant_matmul/quant_batch_matmul_mx.py` 为准。
:::

## 混合量化：MXA8W4

`quant_matmul` 里的 MXA8W4 是一个值得单独看的模式：**激活用 MXFP8，权重用 MXFP4**。

硬件的 MX 矩阵乘要求两侧元素位宽一致，所以这条路径的做法是：

1. AIV 先做一段 prologue，把 FP4 权重**纯位置换**成 FP8（不是数值转换，只是重排位域）。
2. 位置换带来的 \(2^{-6}\) 因子由 Fixpipe 的 `deq_scale=64` 和 Host 侧的 `bias/64` 两处联合补偿。
3. 转换结果经 **`ChannelKind.CrossCore` 通道**交给 AIC 做 MX matmul。

这是一个「用 Vector 补硬件能力缺口、用跨核 Channel 交接」的典型结构，详见[融合算子的设计方法论](./fusion-design.md)。

## 随路量化与反量化

搬运单元（Fixpipe）可以在 L0C 搬出的路上顺手完成量化、反量化和激活，**不占用计算单元**：

| 想做的事 | 写法 |
| --- | --- |
| L0C 搬出时反量化 | `mem_copy(dst, l0c, deq_scale=...)` |
| L0C 搬出时 ReLU | `engine=make_copy_engine(relu=True)` |
| 搬出时转 ND | `engine=make_copy_engine(format_transform="nz2nd")` |

L0C→UB 的写回还可以把 FP32/INT32 直接量化成 BF16/FP16/FP8/INT8。能在搬运里做的事，不要单独写一遍矢量计算。

## 精度相关的三个坑

### 1. 3510 硬件不实现非规格化数，而且默认 flush-to-zero

中间结果落进非规格化区间时，设备上得到 0，CPU golden 得到一个极小的非零值。症状是**个别元素误差极大**，而不是整体误差偏大——很容易被误判成搬运或掩码问题。

开关是毕昇的 `--cce-ftz`，默认 `true`。完整说明见[编译选项与产物观察](./compiler-options.md)，排查步骤见[调试与精度验证](./debugging.md#精度不对时的四个嫌疑人)。

### 2. 累加顺序

K 方向切块累加的顺序决定了浮点舍入的累积方式。设备上按 tile 顺序累加，golden 如果用 `torch.matmul` 一次算完，两者的误差来源不同。写 golden 时**用 fp32 累加**，并按算子实际的容差标准比对，不要用默认容差。

### 3. MX 的 scale 粒度本身就是误差来源

每 32 个元素共享一个 2 的幂次 scale，这是有损的。MXFP4 的参考实现必须按同样的块长和同样的 scale 取整方式计算，否则比对的是两种不同的量化方案，差异与算子实现无关。

## 选型决策

| 场景 | 建议 |
| --- | --- |
| 先把算子跑对 | FP32 或 BF16，不碰量化 |
| 精度敏感但要吞吐 | HiF8（不需要 scale，改动最小） |
| 标准 FP8 生态对齐 | FP8 E4M3 / E5M2 + per-tensor scale |
| 内存与带宽是瓶颈 | MXFP8 |
| 追求极限吞吐 | MXFP4，必要时走 MXA8W4 混合 |
| FP32 算子想提速又不想改数据类型 | 开 HF32 |

## 速查

| 你想做的事 | 用什么 |
| --- | --- |
| 开 HiF8 矩阵乘 | [`enable_hif8`](../api/compute/matrix/enable-hif8.md) |
| 开 FP8 矩阵乘 | [`enable_fp8`](../api/compute/matrix/enable-fp8.md) |
| 用精度换 FP32 吞吐 | [`enable_hf32`](../api/compute/matrix/enable-hf32.md)、[`set_hf32_round_mode`](../api/compute/matrix/set-hf32-round-mode.md) |
| 搬 MX scale | `make_copy_engine(format_transform="mx_scale_*")` + `mem_copy(..., mx_scale=...)` |
| 搬出时反量化 | `mem_copy(dst, l0c, deq_scale=...)` |
| 矢量侧转换数值类型 | `vcast`（可配 `RoundingMode`） |
| 标量侧转换数值类型 | `cast` |
| 按另一种 dtype 重新解释字节 | `t.view(dtype=...)` / `t.reinterpret(...)`，**不做数值转换** |

## 下一步

[同步、Cache 与跨核交接](./synchronization.md)：量化 prologue 常常跨核，交接怎么写。
