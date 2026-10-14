# 同步、Cache 与跨核交接

CANNBot-DSL 的一个核心主张是「同步样板由框架生成」。这句话成立的范围需要讲清楚——**框架替你管的是核内搬运与计算之间的配对，其余四类同步仍然要你自己写。** 这一章把全部同步场景列齐，并给出「框架自动」与「手工」的分界线。

## 五类场景，按复杂度递增

| 场景 | 谁负责 | 入口 |
| --- | --- | --- |
| 1. 核内搬运与计算之间 | **框架**，按 `produce` / `consume` 配对插入 | `Channel` |
| 2. 核内跨流水、VF 内的 UB 重叠 | 你 | `vmem_bar`、pipe 级同步 |
| 3. 核间（AIC 之间、AIV 之间、AIC 与 AIV 之间） | 你 | `*_sync_block_*` / `*_sync_intra_*` / `global_sync_all` |
| 4. 跨核的数据交接 | **框架**，但要你声明 | `ChannelKind.CrossCore` |
| 5. 多核经 Scalar 读写同一块 GM | 你 | `dcci_*` / `dci` / `*_bypass` |

前两类在一个核内，后三类跨核。绝大多数算子只碰 1 和 4；Mix 算子和多核归约会碰到 3 和 5。

## 场景 1：核内搬运与计算，框架自动

同一个核内，「搬运完成了吗」「上一轮算完了可以覆盖了吗」这类等待，由 lowering 根据 `Channel` 的 `produce()` / `consume()` **配对关系**插入。

关键在于理解这两个调用本身做什么：

```text
produce()  →  返回写游标当前槽位的 Tensor，推进写游标。不等待空槽。
consume()  →  返回读游标当前槽位的 Tensor，推进读游标。不检查数据是否已到。
```

**它们只选槽、不阻塞。** 真正的等待是编译器按「第 n 次 produce 对应第 n 次 consume」这个配对关系生成的。

::: danger 配对写错不会在 produce / consume 上报错
因为这两个调用本身不做检查，配对错误（多一次 produce、漏一次 consume、两个 Channel 的调用交叉错位）会表现成**数据内容不对**，而不是挂死或报错。排查时先数一遍每个 Channel 的 produce / consume 次数是否相等、顺序是否一一对应。
:::

## 场景 2：核内跨流水与 VF 内的 UB 重叠

一个核内有多条流水线并行推进：

| 流水 | 做什么 |
| --- | --- |
| `PIPE_S` | 标量：地址计算、循环控制 |
| `PIPE_V` | 矢量计算（VF 区域） |
| `PIPE_M` | Cube 矩阵乘 |
| `PIPE_MTE1` | L1 → L0A / L0B |
| `PIPE_MTE2` | GM → L1 / UB |
| `PIPE_MTE3` | UB → GM |
| `PIPE_FIX` | Fixpipe，L0C 搬出 |

`Channel` 覆盖的是「搬运流水与计算流水之间的生产消费关系」。它**不覆盖**：

- **VF 区域内的 UB 地址重叠。** 同一个 VF 里先 `vstore` 到某个 UB 地址、再 `vload` 同一地址，需要按接口约束自己插 [`vmem_bar`](/zh/api/kernel/reg_compute/reg_sync/vmem-bar.html)。
- **需要精确控制某条流水的等待点。** 深度流水的 Cube 算子会用 pipe 级同步显式等某条搬运流水，例如 `samples/flash_attn` 里的写法：

```python
from cannbotdsl import PIPE
from cannbotdsl.ops.sync import cube_sync_pipe, vec_sync_pipe

cube_sync_pipe(PIPE.MTE2)    # 等 GM → L1 这条流水
cube_sync_pipe(PIPE.MTE1)    # 等 L1 → L0 这条流水
```

AIV 侧对应的是 `vec_sync_pipe(...)`。两者都在 `cannbotdsl.ops.sync` 里，`PIPE` 也在包顶层导出。

::: info 包内可用 · 文档站未收录
官方 API 文档站没有 `PIPE` 和 `*_sync_pipe` 的独立接口页，但包里（`cannbotdsl/ops/sync.py`）枚举是完整的：

`P_NULL`、`S`、`V`、`V2`、`M`、`MTE1`、`MTE2`、`MTE3`、`MTE4`、`MTE5`、`FIXPIPE`、`VLD`、`VST`、`LD`、`ST`、`LD_ST`、`VEXE`、`AIV_MTE2`、`AIC_MTE3`、`AIC_MTE1`、`ALL`

常用的就是上面那张流水表里的几个，其余是内部或特殊通路用的。

即便如此仍然建议：能用 `Channel`（含 `CrossCore`）表达的依赖就用 Channel，那条路径有文档保证，也不用自己数配对。
:::

::: warning 一个 Channel 管不到的完成屏障
同一个 Buffer 上**连续两次 MTE2 纯写**（包括 `fill_l1` / GM→L1，以及零次循环前后的 GM→UB），如果要求后一次覆盖前一次，必须显式调用 `cube_sync_pipe(PIPE.MTE2)` 或 `vec_sync_pipe(PIPE.MTE2)`。

同一 pipe 上的 Buffer 锁**不能**代替这个完成屏障——锁保证的是互斥，不是「上一笔搬运已落地」。
:::

### 矩阵乘的两个免同步特例

这两条能省掉不少同步指令，值得记住：

1. **累加到同一块 L0C 的相邻两次 `matmul` 之间不需要插同步。**
2. **开启 UnitFlag 后，乘加指令与对应的搬出指令之间不需要插同步**——但两者必须同时开启，且 `set_mmad_direction` 的方向要与搬出的读取顺序一致。

## 场景 3：核间同步

当一个核的计算依赖其他核的结果时需要显式核间同步。典型场景：多个 AIV 各算一部分、用原子累加写进 GM，AIC 必须等全部 AIV 累加完成后才能读。

| 模式 | 范围 | 接口 |
| --- | --- | --- |
| 模式 0 | 全部 AIC 之间，或全部 AIV 之间 | `cube_sync_block_arrive` / `wait`，`vec_sync_block_arrive` / `wait` |
| 模式 1 | 单个 AI Core 内，两个 AIV 之间 | `vec_sync_block_arrive` / `wait` |
| 模式 2 | 单个 AI Core 内，AIC 与**全部** AIV 之间（1:2） | `cube_sync_block_*` 与 `vec_sync_block_*` 配对 |
| 模式 4 | 单个 AI Core 内，AIC 与**单个** AIV 之间（1:1） | `cube_sync_intra_*` / `vec_sync_intra_*` |

::: tip 没有模式 3
编号从 2 跳到 4 不是漏写。NPU ARCH 3510 的跨核同步就只有 0、1、2、4 四种模式（见官方《NPU 架构版本 3510》的「核间同步」一节）。

模式 4 是这一代新增的：模式 2 下 AIC 只能等「两个 AIV 都到」，模式 4 允许 AIV0 和 AIV1 **各自单独**触发 AIC 的等待，粒度更细。
:::

::: warning `sync_intra` 的计数器有上限
每个 `sync_intra` 计数器最多连续累加 **15** 次，消费前不得溢出或复用。另外手工协议**不能复用仍在活动的 Channel 私有 `sync_id`**——Channel 的同步资源由 lowering 管理，混用会互相踩。

AIC 要分别通知或等待两个 AIV 时，对 `b` 和 `b + 16` 各调用一次；语义是集合汇合时直接用模式 2。
:::

命名规律：`*_arrive` 是生产者（发通知），`*_wait` 是消费者（阻塞等待）。两者必须成对出现在对应的核上。

需要粗粒度「全核都到这里」的屏障时，直接用易用性接口：

```python
global_sync_all()     # 依次完成模式 2、模式 0、模式 0、模式 2 四次同步
```

它一次调用解决问题，但代价不低——**不要放在内层循环里**。

另有 `vec_sync_all()` 用于 AIV 侧的整体同步，同样见于 `flash_attn`（参见上面的待核实说明）。

完整说明见[系统同步能力概述](/zh/api/kernel/resource-management/system-sync-overview.html)与[核间同步能力概述](/zh/api/kernel/resource-management/inter-core-sync-overview.html)。

## 场景 4：跨核数据交接

生产者和消费者不在同一个核时（典型是 AIC 算完矩阵乘、AIV 接着做 softmax 或量化），**不要手写 `*_sync_*` 加 GM 中转**。把 `Channel` 声明成跨核的即可：

```python
p_l1 = Channel(MemLoc.L1, (128, 128), dtypes.float16,
               depth=2, kind=ChannelKind.CrossCore)
```

要点：

- **写法与同核 Channel 完全一致**，仍然是 `produce()` / `consume()`，同步仍由 lowering 插入。
- `depth` 上限为 **8**（同核 Channel 只受片上容量限制）。
- 纯 AIC、纯 AIV、Mix 三种 Kernel 的 Channel 写法统一，不需要切换编译模式。

`ChannelKind` 只有两个取值：`SameCore`（默认）和 `CrossCore`。

### SSBUF：标量侧的跨核共享

数据交接走 Channel，但 AIC 与 AIV 之间**少量标量**的交换走 `MemLoc.SSBUF`。它是 3510 新增的核内共享 Buffer，两侧的 Scalar 单元都能访问。

上一代（NPU ARCH 2201）没有这块存储，AIC 与 AIV 的标量交接必须经过 **Global Memory** 往返。所以从 Ascend C 迁移过来时，凡是「为了传几个标量而绕一趟 GM」的代码都应该改用 SSBUF。

## 场景 5：Cache 一致性

Scalar 单元访问 GM 会经过 DCache，而 **DCache 是多核独立缓存**。一个核通过 Scalar 写 GM、另一个核读同一地址时，可能读到过期数据。

两类处理方式：

| 方式 | 接口 | 适用 |
| --- | --- | --- |
| 按需清理 / 失效 Cache 行 | [`dcci_single`](/zh/api/kernel/synchronization-cache/dcci-single.html)、[`dcci_entire_out`](/zh/api/kernel/synchronization-cache/dcci-entire-out.html)、[`dcci_entire_atomic`](/zh/api/kernel/synchronization-cache/dcci-entire-atomic.html)、[`dci`](/zh/api/kernel/synchronization-cache/dci.html) | 地址可控、需要保留 Cache 收益 |
| 绕过 DCache 直读直写 | [`load_bypass`](/zh/api/kernel/scalar_compute/scalar_load/load-bypass.html)、[`vec_store_bypass`](/zh/api/kernel/scalar_compute/scalar_store/vec-store-bypass.html)、[`cube_store_bypass`](/zh/api/kernel/scalar_compute/scalar_store/cube-store-bypass.html) | 多核操作的 GM 地址无法对齐到 Cache Line |

::: danger 未对齐到 Cache Line 的多核写入必须 bypass
经过 DCache 的读写以 Cache Line 为粒度。多个核写同一条 Cache Line 的不同部分时，会发生**整行回写互相覆盖**，表现为结果随机。这种情况下 bypass 不是优化，是正确性要求。
:::

L2 Cache 是多核共享的，并且跨 die 一致性由硬件维护、对软件透明，不存在这个问题；ICache 只读，也不存在。详见[系统缓存概述](/zh/api/kernel/synchronization-cache/)。

## workspace GM 的同步要自己写

这是场景 1 和场景 3 的结合处，也是最容易漏的地方。官方同步章的论断是：**片上同步由 Channel 自动生成，GM workspace 的同步需要手写。**

Stream-K、FlashDecode 这类算法的结构是：

```text
阶段 A：多个核各算一段，部分和写入 workspace GM
   ↓  ← 这里必须有核间同步
阶段 B：某些核从 workspace 读回，归约成最终结果
```

中间那道屏障**不会**由 Channel 生成，因为 workspace 是普通 GM 张量，不是 Channel 槽位。必须用模式 0 的核间同步或 `global_sync_all()` 显式隔开。workspace 本身的分配与管理见[片上存储、流水与 GM 协作](/zh/guide/programming-model/onchip-memory)。

## 排查顺序

多核偶发错误最难查。按这个顺序试，每一步都能排除一大类原因：

1. **把 `block_dim` 降到 1。** 还错 → 不是并发问题，去查分核逻辑或尾块。
2. **在可疑的写入之后加 `global_sync_all()`。** 变正确 → 确实缺同步，再往回收细粒度。
3. **数每个 Channel 的 produce / consume 次数。** 不相等或顺序错位 → 场景 1 的配对问题。
4. **检查 VF 内是否有 UB 地址重叠。** 有 → 补 `vmem_bar`。
5. **检查 Scalar 对 GM 的读写。** 需要 `dcci_*`，或改用 `*_bypass`。
6. **检查多核写入的 GM 地址是否对齐到 Cache Line。** 不对齐 → 必须 bypass。

## 速查

| 你想做的事 | 用什么 |
| --- | --- |
| 核内搬运与计算的同步 | 什么都不用写，`Channel` 负责 |
| AIC 算完交给 AIV | `Channel(..., kind=ChannelKind.CrossCore)` |
| 传几个标量给对侧核 | `MemLoc.SSBUF` |
| VF 内 UB 地址重叠 | `vmem_bar` |
| 等某条搬运流水 | `cube_sync_pipe(PIPE.*)`（见待核实说明） |
| 全部 AIC 或全部 AIV 之间 | 模式 0 的 `*_sync_block_arrive` / `wait` |
| 一个核内 AIC 与单个 AIV | 模式 4 的 `*_sync_intra_*` |
| 全核粗粒度屏障 | `global_sync_all()`，别放内层循环 |
| workspace GM 的阶段隔离 | 手写核间同步，Channel 不管 |
| Scalar 写 GM 后让别的核看到 | `dcci_*` / `dci`，或 `*_bypass` |

## 下一步

机制部分到此为止，接下来动手：[写出第一个算子](/zh/guide/programming-model/first-operator)。
