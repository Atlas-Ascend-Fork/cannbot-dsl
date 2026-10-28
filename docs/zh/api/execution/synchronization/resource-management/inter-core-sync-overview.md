# 核间同步能力概述

核间同步的使用场景通常是：一个核计算所依赖的数据，需要等待其他相关核的计算全部完成后才能获取。

以图 1 为例，AIC 需要依赖 AIV 的计算结果。由于整体矢量较大，必须拆分为多个部分，
由每个 AIV 分别完成部分计算；每个 AIV 将其部分计算结果通过原子累加写入 GM。
AIC 必须读取所有 AIV 均完成累加后的最终结果，因此需要通过核间同步来保证 AIC 读取结果时的时序正确性。

**图 1**  核间同步业务场景示例图（AIC 与 AIV 的比例为 1：2）

![核间同步业务场景示例图](../../../figures/inter_core_sync_scenario_example.svg)

## 核的层级结构

AIC/AIV 按 group 划分。一个 group 内包含 1 个 block 和 N 个 subblock（N≥1），
其中 block 表示“主核”，每个 subblock 表示一个“从核”，如图 2 所示。

**图 2**  block 和 subblock 之间关系（灰色部分表示一个 group，即 1 个 block 和 N 个 subblock）

![block 和 subblock 之间关系](../../../figures/block_subblock_relationship_3510.svg)

一次 block 调度对应一个 group：`block_dim` 决定启动多少个 group，
`block_idx` 标识当前 group。Mix 算子的一个 group 包含 1 个 block（AIC）和 N 个 subblock（AIV），
`get_subblock_id()` 在 AIV 上从 0 开始编号，在 AIC 上取值为 0。

## 支持的核间同步场景

算子按计算特征可分为三类：Cube 算子（矩阵计算）、Vector 算子（矢量计算）和 Mix 算子（同时包含矩阵和矢量计算）。
如表 1 所示，算子类型决定了其所需的核间同步方式和 group 配置。
其中 Mix 算子同时包含 AIC 与 AIV 两段执行代码，group 配置可以是 1：1（1 个 AIC 配 1 个 AIV）
或 1：2（1 个 AIC 配 2 个 AIV）；Cube 算子和 Vector 算子为单类核执行，不涉及 block/subblock 划分。

**表 1**  group 配置

| 算子类型 | 执行核 | block/subblock | 1：N |
| --- | --- | --- | --- |
| Cube 算子 | 仅 AIC | AIC 为 block，无 subblock | 不涉及 |
| Vector 算子 | 仅 AIV | AIV 为 block，无 subblock | 不涉及 |
| Mix 算子 | AIC 与 AIV | AIC 为 block，AIV 为 subblock | 1：1 |
| Mix 算子 | AIC 与 AIV | AIC 为 block，AIV 为 subblock | 1：2 |

表 2 总结了核间同步接口及其支持的同步场景。

**表 2**  支持的核间同步场景

| 接口名称 | 同步范围 | 对应模式 | 功能简述 |
| --- | --- | --- | --- |
| [`cube_sync_block_arrive`](./cube-sync-block-arrive.md)<br>[`cube_sync_block_wait`](./cube-sync-block-wait.md)<br>[`vec_sync_block_arrive`](./vec-sync-block-arrive.md)<br>[`vec_sync_block_wait`](./vec-sync-block-wait.md) | 组间同步，即不同 group 之间所有 block（所有 AIC 或所有 AIV）之间的同步。 | 模式 0（同步等待全部 AIC 或全部 AIV 执行结束） | `*_sync_block_arrive` 为生产者，负责发送通知；<br>`*_sync_block_wait` 为消费者，负责阻塞等待通知到达。 |
| [`vec_sync_block_arrive`](./vec-sync-block-arrive.md)<br>[`vec_sync_block_wait`](./vec-sync-block-wait.md) | 组内 subblock 间的同步，即同一 group 内不同 subblock（AIV）之间的同步。 | 模式 1（单个 AI Core 内，全部 AIV 之间的同步） | `vec_sync_block_arrive` 为生产者，负责发送通知；<br>`vec_sync_block_wait` 为消费者，负责阻塞等待通知到达。 |
| [`cube_sync_block_arrive`](./cube-sync-block-arrive.md)<br>[`cube_sync_block_wait`](./cube-sync-block-wait.md)<br>[`vec_sync_block_arrive`](./vec-sync-block-arrive.md)<br>[`vec_sync_block_wait`](./vec-sync-block-wait.md) | 同一 group 内 block（AIC）与所有 subblock（AIV）之间的同步。 | 模式 2（单个 AI Core 内，AIC 与全部 AIV 之间的同步） | `*_sync_block_arrive` 为生产者，负责发送通知；<br>`*_sync_block_wait` 为消费者，负责阻塞等待通知到达。 |
| [`cube_sync_intra_arrive`](./cube-sync-intra-arrive.md)<br>[`cube_sync_intra_wait`](./cube-sync-intra-wait.md)<br>[`vec_sync_intra_arrive`](./vec-sync-intra-arrive.md)<br>[`vec_sync_intra_wait`](./vec-sync-intra-wait.md) | 同一 group 内 block（AIC）与单个 subblock（AIV）之间的同步。 | 模式 4（单个 AI Core 内，AIC 核与单个 AIV 之间同步） | `*_sync_intra_arrive` 为生产者，负责发送通知；<br>`*_sync_intra_wait` 为消费者，负责阻塞等待通知到达。 |

四种模式的计数与阻塞过程见[关键特性说明](./key-features.md)。
各接口的配对要求与约束条件见对应接口页的「约束说明」。
