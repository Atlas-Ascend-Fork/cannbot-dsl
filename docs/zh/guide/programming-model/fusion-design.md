# 融合算子的设计方法论

前面几章讲的是「怎么写」。这一章讲「写什么」——拿到一个数学式子，怎么把它拆成能在 AI Core 上跑的阶段。仓库里一半以上的样例是 attention 变体，它们的难点都不在接口，而在这一步。

## 从数学式到阶段图

任何融合算子的设计都是回答三个问题，顺序不能换：

```text
① 这个式子里有哪些计算？分别该交给谁？
      Cube（矩阵乘） / Vector（逐元素、归约） / Scalar（地址、少量元数据）
                              ↓
② 中间结果有多大？片上放得下吗？
      放得下 → 一趟算完
      放不下 → 切分 + 在线归约 / 状态递推 / 两趟遍历
                              ↓
③ 阶段之间在哪里交接？
      同核内 → Channel
      AIC ↔ AIV → CrossCore Channel
      跨核归约 → workspace GM + 核间同步
```

第 ② 步是真正的设计决策点，也是本章的重点。

## 第一步：阶段划分

以 attention 为例：

$$
O = \mathrm{softmax}\!\left(\frac{QK^{\mathrm{T}}}{\sqrt{d}}\right) V
$$

按计算特征拆开：

| 阶段 | 计算 | 执行单元 | 中间结果 |
| --- | --- | --- | --- |
| QK | \(S = QK^{\mathrm{T}}\) | Cube | \(S\)，大小 \(S_1 \times S_2\) |
| softmax | 行最大、减最大、exp、行求和、归一化 | Vector | 行统计量 + \(P\) |
| PV | \(O = PV\) | Cube | \(O\)，大小 \(S_1 \times d\) |

划分原则：

- **矩阵乘一定走 Cube**，哪怕它很小。GEMV（\(M = 1\)）也有专门模式（`disable_gemv=False`）。
- **逐元素、归约、激活一定走 Vector**，而且尽量把整条链留在寄存器里，详见[矢量寄存器与 lane 模型](/zh/guide/programming-model/vector-registers)。
- **Cube 与 Vector 相邻时考虑拆成 Mix 算子**，让两者流水重叠，而不是在一个核上串行。

## 第二步：片上放不下怎么办

这是全部复杂度的来源。\(S = QK^{\mathrm{T}}\) 的大小是 \(S_1 \times S_2\)，序列一长就远超 UB。三种标准解法：

### 解法 A：两趟遍历（最简单）

第一趟只算统计量，第二趟再算结果。`rms_norm` 的 `ColSplitKernel` 就是这样：

```text
第一趟：沿归一化轴扫一遍，累加 x² → 得到 rstd
第二趟：再扫一遍，y = x · rstd · γ
```

代价是**数据读两遍**，带宽翻倍。只在「统计量必须先算完」且数据量不大时用。

### 解法 B：在线归约（attention 的标准做法）

不等全部数据到齐就边扫边修正统计量。softmax 的在线形式是这样推出来的：

朴素 softmax 需要先知道全局最大值 \(m\)：

$$
P_j = \frac{e^{S_j - m}}{\sum_k e^{S_k - m}}
$$

按 KV 分块扫描时，处理到第 \(i\) 块才知道「到目前为止的最大值」\(m_i\)。当新块带来更大的最大值时，**此前累积的分母和输出都要按比例修正**：

$$
m_i = \max(m_{i-1},\ \tilde m_i), \qquad
\alpha = e^{m_{i-1} - m_i}
$$

$$
\ell_i = \alpha \cdot \ell_{i-1} + \sum e^{\tilde S - m_i}, \qquad
O_i = \alpha \cdot O_{i-1} + \tilde P V_i
$$

\(\alpha\) 是修正因子。每块只需维护三个状态：**行最大 \(m\)、行分母 \(\ell\)、累积输出 \(O\)**。片上开销从 \(O(S_2)\) 降到 \(O(1)\)。

落到代码上，这三个状态是常驻的 `Buffer`（不是 `Channel`，因为它们不轮转）：

```python
self.softmax_max_bufs = [Buffer(MemLoc.UB, (tile_m, 1), dtypes.float32) for _ in range(...)]
self.run_max = Buffer(MemLoc.UB, (tile_m, 1), dtypes.float32)
self.run_den = Buffer(MemLoc.UB, (tile_m, 1), dtypes.float32)
self.old_scale_buf = Buffer(MemLoc.UB, (tile_m, 1), dtypes.float32)
```

每块更新时用 `vload_broadcast` 把行标量广播到全 lane、`vstore_first` 写回单个行标量——这是「按行的标量运算」在 Reg 模型里的标准写法。

::: tip 在线归约的判据
当「归约结果需要回头修正已算出的部分」时就用在线形式。softmax 是最典型的；`vreduce_max` 之后再 `vexp_sub` 的朴素写法只适用于一整行能装进片上的情况。
:::

### 解法 C：状态递推（线性注意力类）

KDA、Mamba 这类算子的结构是「块间递推一个状态矩阵」：

```text
chunk 0 → state_0 ──┐
chunk 1 ← state_0   │  每个 chunk 读入上一个 chunk 的状态，
       → state_1 ───┤  算完输出并更新状态
chunk 2 ← state_1   │
       → state_2 ───┘
```

`flash_kda` 按 **64 个 token 一个 chunk** 切分，chunk 间递推状态，返回 `(out, final_state)`。

和在线归约的区别：在线归约的状态是**几个标量**，状态递推的状态是**一个矩阵**（KDA 里是 `[N, 128, 128]`）。所以：

- 状态通常放 L1 或 GM，不是 UB。
- chunk 之间有**真实的串行依赖**，不能靠加核来并行；并行度来自 batch 和 head 维。
- 尾块（序列长度不是 chunk 的整数倍）要单独处理。

## 第三步：交接点怎么选

拆好阶段之后，每个边界都要决定交接方式：

| 边界 | 方式 | 注意 |
| --- | --- | --- |
| 同核内，搬运 ↔ 计算 | `Channel` | 同步自动；`depth >= 2` 才有流水 |
| 同核内，阶段 A ↔ 阶段 B | `Buffer` 或 `Channel` | 阶段间复用片上地址用 `channel_rewind()` |
| AIC ↔ AIV | `Channel(kind=CrossCore)` | `depth <= 8`；别手写同步加 GM 中转 |
| 多核归约 | workspace GM + 核间同步 | 这道屏障**不会**自动生成 |
| 少量标量跨核 | `MemLoc.SSBUF` | 上一代要绕 GM，3510 不用了 |

选交接粒度的权衡：

- **交接太细**：同步次数多，Channel 槽位小，流水填不满。
- **交接太粗**：片上放不下，或者下游要等很久才开始干活。

实践上的起点是**让交接块的大小等于下游一次计算的输入量**，然后按 profiling 结果调。

## 第四步：调度计划外置

当「每个核该干多少活」依赖运行时数据（变长序列、稀疏候选数）时，不要让 AI Core 自己算，交给 AI CPU 先生成一份计划。见 [AI CPU 与调度计划](/zh/guide/programming-model/aicpu)。

判据：**依赖张量内容 + 结果要被所有核共享**，两条同时成立才值得外置。

## 第五步：长尾与负载均衡

阶段都定了之后，最后一个问题是分核。两种典型情况：

**tile 数不能被核数整除** → Stream-K：把尾部 tile 按 K 维切开，让更多核参与，部分和写 workspace 再归约。见[高性能算子编写指南](/zh/guide/programming-model/performance)。

**每个核的活不一样多**（变长序列） → 让 AI CPU 按实际长度做前缀和分配，而不是按 batch 平均分。

## 案例对照

把仓库样例按本章的三个决策点归类，便于找参照：

| 样例 | 阶段划分 | 片上不够的解法 | 交接方式 |
| --- | --- | --- | --- |
| `rms_norm` | Vector 两段 | UB 放得下整行就一趟；否则列切分**两趟** | 同核 `Channel` |
| `matmul` | 纯 Cube | K 向切块 + L0C 累加 | 同核 `Channel`，两级 tile |
| `matmul_streamk` | Cube + AIV 归约 | DP / SK 混合调度 | **workspace GM** + 核间同步 |
| `flash_attn` | Cube(QK) → Vector(softmax) → Cube(PV) | **在线归约** | `CrossCore Channel`，三级流水 |
| `flash_kda` | Vector 预处理 + Chunk 递推 | **状态递推**（64 token / chunk） | AI CPU 调度 + 跨核 Channel |
| `quant_matmul` MXA8W4 | AIV 权重 prologue → AIC MX matmul | K 状态全部窗化 | `CrossCore Channel` |
| `qwen_sparse_attn` | AI CPU 筛块 → Cube + Vector | 块稀疏，只访问候选块 | AI CPU metadata + 跨核 |
| `mixed_quant_sparse_flash_mla` | 多路量化 KV + MLA | 在线归约 + FlashDecode 分片 | AI CPU 计划 + workspace |

## 一个可照搬的推导流程

```text
① 写出数学式，标出每一项的形状
② 按计算特征给每一项指派执行单元（Cube / Vector / Scalar）
③ 算最大的中间结果有多大，和 UB / L1 容量比
      放得下 → 一趟
      放不下 → 问：归约结果需要回头修正吗？
                 需要 → 在线归约
                 不需要但有块间依赖 → 状态递推
                 都不是 → 两趟遍历
④ 在每个阶段边界上选交接方式，定交接块大小
⑤ 问：分核计划依赖运行时数据吗？依赖 → AI CPU 外置
⑥ 问：tile 数能被核数整除吗？不能 → Stream-K 或按前缀和分配
⑦ 写出正确版本，跑精度测试
⑧ 按《高性能算子编写指南》的五个层次调优
```

## 下一步

动手部分到此为止。接下来是工程化：[JIT 参数与编译缓存](/zh/guide/programming-model/jit-arguments)——什么时候会重新编译。

想直接跳到调优，看[高性能算子编写指南](/zh/guide/programming-model/performance)。
