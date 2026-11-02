# 同步、Cache 与跨核交接

CANNBot-DSL 的一个核心主张是「同步样板由框架生成」。这句话成立的范围需要讲清楚——**框架根据 `Buffer` / `RingBuffer` 对应 Tensor 的实际访问生成数据同步；需要精确控制的流水屏障、核间屏障和 GM Cache 一致性仍要你自己写。** 这一章把全部同步场景列齐，并给出「框架自动」与「手工」的分界线。

## 五类场景，按复杂度递增

| 场景 | 谁负责 | 入口 |
| --- | --- | --- |
| 1. 核内搬运与计算之间 | **框架**，按 Tensor 实际访问插入 | `Buffer` / `RingBuffer` |
| 2. 核内跨流水、VF 内的 UB 重叠 | 你 | `vmem_bar`、pipe 级同步 |
| 3. 核间（AIC 之间、AIV 之间、AIC 与 AIV 之间） | 你 | `*_sync_block_*` / `*_sync_intra_*` / `global_sync_all` |
| 4. 跨核的数据交接 | **框架**，跨核属性按实际访问推导 | `RingBuffer` |
| 5. 多核经 Scalar 读写同一块 GM | 你 | `dcci_*` / `dci` / `*_bypass` |

前两类在一个核内，后三类跨核。绝大多数算子只碰 1 和 4；Mix 算子和多核归约会碰到 3 和 5。

## 场景 1：核内搬运与计算，框架自动

同一个核内，「搬运完成了吗」「上一轮算完了可以覆盖了吗」这类等待，由编译器根据 Tensor 的实际读写、资源身份和控制流插入。

关键在于理解 `RingBuffer` 的两个选槽调用本身做什么：

```text
next_write()  →  返回写游标当前槽位的 Tensor，推进写游标。不等待空槽。
next_read()   →  返回读游标当前槽位的 Tensor，推进读游标。不检查数据是否已到。
```

**它们只选槽、不阻塞，也不是一一配对的生产消费令牌。** 两条游标独立推进；需要再次读取或更新同一槽时，保存并复用返回的 Tensor 即可。编译器分析的是这些 Tensor 上发生的实际访问，而不是「第 n 次 `next_write` 必须对应第 n 次 `next_read`」。

```python
value = staging.next_read()
use(value)
use_again(value)  # 复用同一 Tensor，不推进读游标
```

::: danger 选槽没有满空或配对检查
`next_write()` 不检查槽位是否可覆盖，`next_read()` 不检查数据是否已经生产，head/tail 差值也不表示有效元素数。调用顺序或环绕覆盖安排错误会表现为数据错乱。排查时应核对每个 RingBuffer 的选槽顺序、保存的 Tensor 及其实际访问，而不是机械地凑平两个方法的调用次数。
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

自动数据同步**不覆盖**：

- **VF 区域内的 UB 地址重叠。** 同一个 VF 里先 `vstore` 到某个 UB 地址、再 `vload` 同一地址，需要按接口约束自己插 [`vmem_bar`](../api/compute/vector/reg_sync/vmem-bar.md)。
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

常用的就是上面那张流水表里的几个，其余是内部或特殊通路用的。能用 `Buffer` / `RingBuffer` 的实际 Tensor 访问表达的普通数据依赖，优先让框架生成同步；手工屏障留给需要精确控制的场景。

同一模块里还有 `cube_sync_notify` / `cube_sync_wait` / `vec_sync_notify` / `vec_sync_wait`（跨 pipe 事件，见下一小节）和 `data_barrier(scope="gm")`（用 DSB_DDR 排空本核的 GM 访问；它**不**完成其他执行流水，也**不**失效 DCache，在 MIX 下每个走到的核都会执行）。`data_barrier` 在包顶层导出。
:::

::: warning 自动数据同步覆盖不到的完成屏障
同一个 Buffer 上**连续两次 MTE2 纯写**（包括 `fill_l1` / GM→L1，以及零次循环前后的 GM→UB），如果要求后一次覆盖前一次，必须显式调用 `cube_sync_pipe(PIPE.MTE2)` 或 `vec_sync_pipe(PIPE.MTE2)`。

同一 pipe 上的 Buffer 锁**不能**代替这个完成屏障——锁保证的是互斥，不是「上一笔搬运已落地」。
:::

### 比 `*_sync_pipe` 更细的一档：跨 pipe notify / wait

`*_sync_pipe(PIPE.X)` 是「等 X 这条流水上**所有**已发出的操作」，粒度很粗。要表达「A 流水的**某一笔**操作完成后 B 流水才能继续」，用事件化的 notify / wait：

```python
from cannbotdsl.ops.sync import cube_sync_notify, cube_sync_wait

# FIXPIPE 把 L0C 排空到 GM 之后，MTE2 才能把同一块 GM 读回 L1
cube_sync_notify(PIPE.FIXPIPE, PIPE.MTE2, event_id=0)   # 生产者 pipe 发信号
cube_sync_wait(PIPE.FIXPIPE, PIPE.MTE2, event_id=0)     # 消费者 pipe 阻塞等待
```

AIV 侧对应 `vec_sync_notify` / `vec_sync_wait`，签名一致。

::: danger notify 和 wait 的参数顺序**相同**，不要对调
这是最容易写错的地方。按 AscendC 约定，**两者都传 `(src_pipe=生产者, dst_pipe=消费者)`**：

```python
cube_sync_notify(producer, consumer, event_id)
cube_sync_wait(producer, consumer, event_id)    # ✓ 和 notify 完全一样的顺序
cube_sync_wait(consumer, producer, event_id)    # ✗ 不要因为「wait 在消费者侧」就调换
```

虽然 `wait` 阻塞的是消费者 pipe，但参数顺序描述的是**数据流方向**，不是调用位置。
:::

这四个接口在 `cannbotdsl.ops.sync` 里，**不在包顶层导出**，要从子模块导入。AscendC 不会自动跨 pipe 串行化，所以「FIXPIPE 写完 GM，MTE2 再读同一块」这类同 pipe 之外的内存依赖必须自己表达。

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

::: danger `mode` 的默认值是 2，不是 0
`*_sync_block_arrive` / `*_sync_block_wait` 的完整签名是：

```python
cube_sync_block_arrive(pipe, flag_id, mode=2)
vec_sync_block_arrive(pipe, flag_id, mode=2)
```

**默认是模式 2（核内 AIC↔全部 AIV），不是模式 0。** 想要上表第一行那个「全部 AIC 之间 / 全部 AIV 之间」的全核屏障，必须**显式写 `mode=0`**：

```python
vec_sync_block_arrive(PIPE.MTE3, 0)            # ✗ 第二个位置参数是 flag_id，mode 仍是 2
vec_sync_block_arrive(PIPE.MTE3, 0, mode=0)    # ✓ flag_id=0，模式 0
```

漏掉 `mode=0` 的后果是屏障范围悄悄缩小到单个 AI Core 内——`block_dim=1` 时完全正常，核一多就偶发错误。

`mode` 只对 `arrive` 生效：`wait` 侧的 `mode` 不下发到设备（codegen 发出不带 mode 的 `asc_sync_block_wait`），参与者集合由配对的 `arrive` 决定。但它仍要写对——`split-mix-kernel` 要靠它区分「全核握手（模式 0，在纯核 Kernel 上保留）」和「核内握手（模式 2，在纯核 Kernel 上被裁掉）」。
:::

::: danger `flag_id` 必须 ≤ 15，超了会被静默截断
codegen 把 `flag_id` 打进 FFTS 配置的一个 **4 位**字段（`flagId & 0xf`），**16 会变成 0、17 会变成 1**，不报错。包里为此加了常量上限 `MAX_SYNC_BLOCK_FLAG_ID = 15` 和入参检查（`cannbotdsl/ops/sync.py`），但只拦得住编译期常量——**运行时 SSA 的 flag id 拦不住**，由你保证。

另外 `global_sync_all()` 内部占用 flag id **0、1、2**。手写协议与它共存时请避开这三个，或用 `global_sync_all(flag_ids=...)` 显式改掉。
:::

::: tip 没有模式 3
编号从 2 跳到 4 不是漏写。NPU ARCH 3510 的跨核同步就只有 0、1、2、4 四种模式（见官方《NPU 架构版本 3510》的「核间同步」一节）。

模式 4 是这一代新增的：模式 2 下 AIC 只能等「两个 AIV 都到」，模式 4 允许 AIV0 和 AIV1 **各自单独**触发 AIC 的等待，粒度更细。
:::

::: warning `sync_intra` 的计数器有上限
每个 `sync_intra` 计数器最多连续累加 **15** 次，消费前不得溢出或复用。另外手工协议**不能复用框架已分配的私有同步 ID**——`Buffer` / `RingBuffer` 的同步资源由编译器管理，混用会互相踩。

AIC 要分别通知或等待两个 AIV 时，对 `b` 和 `b + 16` 各调用一次；语义是集合汇合时直接用模式 2。
:::

命名规律：`*_arrive` 是生产者（发通知），`*_wait` 是消费者（阻塞等待）。两者必须成对出现在对应的核上。

需要粗粒度「全核都到这里」的屏障时，直接用易用性接口：

```python
global_sync_all()     # 依次完成模式 2、模式 0、模式 0、模式 2 四次同步
```

它一次调用解决问题，但代价不低——**不要放在内层循环里**。

另有 `vec_sync_all()` 用于 AIV 侧的整体同步，同样见于 `flash_attn`（参见上面的待核实说明）。

完整说明见[系统同步能力概述](../api/execution/synchronization/resource-management/system-sync-overview.md)与[核间同步能力概述](../api/execution/synchronization/resource-management/inter-core-sync-overview.md)。

## 场景 4：跨核数据交接

生产者和消费者不在同一个核时（典型是 AIC 算完矩阵乘、AIV 接着做 softmax 或量化），**不要手写 `*_sync_*` 加 GM 中转**。使用普通 `RingBuffer`，让编译器根据实际 Tensor 访问识别跨核交接：

```python
p_l1 = RingBuffer(
    MemLoc.L1, dtypes.float16, make_layout((128, 128)), depth=2
)
```

要点：

- **写法与同核完全一致**，仍然是 `next_write()` / `next_read()`，同步由编译器按实际访问生成。
- 不需要 `ChannelKind`、`RingBufferKind` 或 `kind` 参数；这些入口已删除。
- `depth` 没有跨核专用的 8 槽上限，容量和硬件资源由编译器检查。
- 纯 AIC、纯 AIV、Mix 三种 Kernel 的写法统一，不需要切换编译模式。

### SSBUF：标量侧的跨核共享

数据交接走 `RingBuffer`，但 AIC 与 AIV 之间**少量标量**的交换走 `MemLoc.SSBUF`。它是 3510 新增的核内共享 Buffer，两侧的 Scalar 单元都能访问。

上一代（NPU ARCH 2201）没有这块存储，AIC 与 AIV 的标量交接必须经过 **Global Memory** 往返。所以从 Ascend C 迁移过来时，凡是「为了传几个标量而绕一趟 GM」的代码都应该改用 SSBUF。

## 场景 5：Cache 一致性

::: warning 不自动插入 dcci
框架不会因为检测到 Scalar 直接读写 GM，就自动插入 `dcci_*`、`dci`
或其他 Cache 一致性指令。直接 GM 访问的 Cache 清理、失效和可见性由
Kernel 作者显式负责；依赖隐式插入 `dcci` 的写法暂不支持。
:::

Scalar 单元访问 GM 会经过 DCache，而 **DCache 是多核独立缓存**。一个核通过 Scalar 写 GM、另一个核读同一地址时，可能读到过期数据。框架不会自动推断这种跨核数据依赖，也不会自动插入 `dcci`。需要保证可见性时，必须显式调用 Cache 操作，并自行保证执行顺序和核间同步。

两类处理方式：

| 方式 | 接口 | 适用 |
| --- | --- | --- |
| 按需清理 / 失效 Cache 行 | [`dcci_single`](../api/execution/synchronization/cache/dcci-single.md)、[`dcci_entire_out`](../api/execution/synchronization/cache/dcci-entire-out.md)、[`dcci_entire_atomic`](../api/execution/synchronization/cache/dcci-entire-atomic.md)、[`dci`](../api/execution/synchronization/cache/dci.md) | 地址可控、需要保留 Cache 收益 |
| 绕过 DCache 直读直写 | [`load_bypass`](../api/compute/scalar/scalar_load/load-bypass.md)、[`vec_store_bypass`](../api/compute/scalar/scalar_store/vec-store-bypass.md)、[`cube_store_bypass`](../api/compute/scalar/scalar_store/cube-store-bypass.md) | 多核操作的 GM 地址无法对齐到 Cache Line |

::: danger 未对齐到 Cache Line 的多核写入必须 bypass
经过 DCache 的读写以 Cache Line 为粒度。多个核写同一条 Cache Line 的不同部分时，会发生**整行回写互相覆盖**，表现为结果随机。这种情况下 bypass 不是优化，是正确性要求。
:::

L2 Cache 是多核共享的，并且跨 die 一致性由硬件维护、对软件透明，不存在这个问题；ICache 只读，也不存在。详见[系统缓存概述](../api/execution/synchronization/cache/overview.md)。

## workspace GM 的同步要自己写

这是场景 1 和场景 3 的结合处，也是最容易漏的地方：**片上 Buffer / RingBuffer 的数据同步可以自动生成，GM workspace 的阶段屏障需要手写。**

Stream-K、FlashDecode 这类算法的结构是：

```text
阶段 A：多个核各算一段，部分和写入 workspace GM
   ↓  ← 这里必须有核间同步
阶段 B：某些核从 workspace 读回，归约成最终结果
```

中间那道屏障**不会**由 `RingBuffer` 生成，因为 workspace 是普通 GM 张量，不是片上槽位。必须用模式 0 的核间同步或 `global_sync_all()` 显式隔开。workspace 本身的分配与管理见[片上存储、流水与 GM 协作](./onchip-memory.md)。

## 排查顺序

多核偶发错误最难查。按这个顺序试，每一步都能排除一大类原因：

1. **把 `block_dim` 降到 1。** 还错 → 不是并发问题，去查分核逻辑或尾块。
2. **在可疑的写入之后加 `global_sync_all()`。** 变正确 → 确实缺同步，再往回收细粒度。
3. **核对每个 RingBuffer 的选槽顺序和保存的 Tensor。** 是否提前环绕覆盖、读错槽，或尾块读端没有沿用生产者的 View。
4. **检查 VF 内是否有 UB 地址重叠。** 有 → 补 `vmem_bar`。
5. **检查 Scalar 对 GM 的读写。** 需要 `dcci_*`，或改用 `*_bypass`。
6. **检查多核写入的 GM 地址是否对齐到 Cache Line。** 不对齐 → 必须 bypass。

## 速查

| 你想做的事 | 用什么 |
| --- | --- |
| 核内搬运与计算的同步 | `Buffer` / `RingBuffer`，框架按实际 Tensor 访问生成 |
| AIC 算完交给 AIV | 普通 `RingBuffer`，跨核属性自动推导 |
| 传几个标量给对侧核 | `MemLoc.SSBUF` |
| VF 内 UB 地址重叠 | `vmem_bar` |
| 等某条搬运流水（整条） | `cube_sync_pipe(PIPE.*)` / `vec_sync_pipe(PIPE.*)` |
| 等另一条 pipe 的**某一笔**操作 | `*_sync_notify` / `*_sync_wait`，两侧参数顺序相同 |
| 排空本核的 GM 访问 | `data_barrier()` |
| 全部 AIC 或全部 AIV 之间 | `*_sync_block_arrive` / `wait`，**必须显式写 `mode=0`**（默认是 2） |
| 一个核内 AIC 与单个 AIV | 模式 4 的 `*_sync_intra_*` |
| 全核粗粒度屏障 | `global_sync_all()`，别放内层循环；它占用 flag id 0/1/2 |
| workspace GM 的阶段隔离 | 手写核间同步，`RingBuffer` 不管 |
| Scalar 写 GM 后让别的核看到 | `dcci_*` / `dci`，或 `*_bypass` |

## 下一步

机制部分到此为止，接下来动手：[写出第一个算子](./first-operator.md)。
