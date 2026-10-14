# 系统同步能力概述

在编程中，同步是指协调多个执行单元（如线程、进程等）对共享资源的访问顺序和时机来确保程序的正确性。
如果没有同步来严格控制执行顺序，则会出现结果不一致、行为不可预测等多种问题。

核间同步是指在一次 kernel 启动内，协调 AIC 与 AIV 之间、以及不同 AI Core 之间的执行顺序，
通过四种同步控制模式实现。图 1 展示了四种同步控制模式，其中 group 配置以 1：2
（即每个 block 由 1 个 AIC 与 2 个 AIV 构成，详见[核间同步能力概述](/zh/api/kernel/resource-management/inter-core-sync-overview)）为例，
各模式的功能描述如下。

**图 1**  四种核间同步控制模式示意图

![四种核间同步控制模式示意图](/figures/3510_sync_control_mode_diagram.png)

- 模式 0：AI Core 核间的同步控制。对于 AIC 全核场景，同步所有的 AIC 核，直到所有的 AIC 核都执行到
  `cube_sync_block_arrive` 时，`cube_sync_block_wait` 后续的指令才会执行；对于 AIV 全核场景，
  同步所有的 AIV 核，直到所有的 AIV 核都执行到 `vec_sync_block_arrive` 时，
  `vec_sync_block_wait` 后续的指令才会执行。
- 模式 1：AI Core 内部，AIV 核之间的同步控制。如果两个 AIV 核都运行了 `vec_sync_block_arrive`，
  `vec_sync_block_wait` 后续的指令才会执行。
- 模式 2：AI Core 内部，AIC 与 AIV 之间的同步控制。在 AIC 核执行 `cube_sync_block_arrive` 之后，
  两个 AIV 上 `vec_sync_block_wait` 后续的指令才会继续执行；两个 AIV 都执行 `vec_sync_block_arrive` 后，
  AIC 上 `cube_sync_block_wait` 后续的指令才能执行。
- 模式 4：AI Core 内部，AIC 与单个 AIV 之间的同步控制。在单个 AIV 核执行 `vec_sync_intra_arrive` 之后，
  AIC 上 `cube_sync_intra_wait` 后续的指令才会继续执行；AIC 执行 `cube_sync_intra_arrive` 后，
  单个 AIV 上 `vec_sync_intra_wait` 后续的指令才能执行。AIV0 与 AIV1 可单独触发 AIC 等待。
  本模式由 `*_sync_intra_*` 接口实现。

**表 1**  核间同步接口

| 接口名称 | 同步范围 | 对应模式 | 功能简述 |
| --- | --- | --- | --- |
| [`cube_sync_block_arrive`](/zh/api/kernel/resource-management/cube-sync-block-arrive)<br>[`cube_sync_block_wait`](/zh/api/kernel/resource-management/cube-sync-block-wait)<br>[`vec_sync_block_arrive`](/zh/api/kernel/resource-management/vec-sync-block-arrive)<br>[`vec_sync_block_wait`](/zh/api/kernel/resource-management/vec-sync-block-wait) | 组间同步，即不同 group 之间所有 block（所有 AIC 或所有 AIV）之间的同步。 | 模式 0（同步等待全部 AIC 或全部 AIV 执行结束） | `*_sync_block_arrive` 为生产者，负责发送通知；<br>`*_sync_block_wait` 为消费者，负责阻塞等待通知到达。 |
| [`vec_sync_block_arrive`](/zh/api/kernel/resource-management/vec-sync-block-arrive)<br>[`vec_sync_block_wait`](/zh/api/kernel/resource-management/vec-sync-block-wait) | 组内 subblock 间的同步，即同一 group 内不同 subblock（AIV）之间的同步。 | 模式 1（单个 AI Core 内，全部 AIV 之间的同步） | `vec_sync_block_arrive` 为生产者，负责发送通知；<br>`vec_sync_block_wait` 为消费者，负责阻塞等待通知到达。 |
| [`cube_sync_block_arrive`](/zh/api/kernel/resource-management/cube-sync-block-arrive)<br>[`cube_sync_block_wait`](/zh/api/kernel/resource-management/cube-sync-block-wait)<br>[`vec_sync_block_arrive`](/zh/api/kernel/resource-management/vec-sync-block-arrive)<br>[`vec_sync_block_wait`](/zh/api/kernel/resource-management/vec-sync-block-wait) | 同一 group 内 block（AIC）与所有 subblock（AIV）之间的同步。 | 模式 2（单个 AI Core 内，AIC 与全部 AIV 之间的同步） | `*_sync_block_arrive` 为生产者，负责发送通知；<br>`*_sync_block_wait` 为消费者，负责阻塞等待通知到达。 |
| [`cube_sync_intra_arrive`](/zh/api/kernel/resource-management/cube-sync-intra-arrive)<br>[`cube_sync_intra_wait`](/zh/api/kernel/resource-management/cube-sync-intra-wait)<br>[`vec_sync_intra_arrive`](/zh/api/kernel/resource-management/vec-sync-intra-arrive)<br>[`vec_sync_intra_wait`](/zh/api/kernel/resource-management/vec-sync-intra-wait) | 同一 group 内 block（AIC）与单个 subblock（AIV）之间的同步。 | 模式 4（单个 AI Core 内，AIC 核与单个 AIV 之间同步） | `*_sync_intra_arrive` 为生产者，负责发送通知；<br>`*_sync_intra_wait` 为消费者，负责阻塞等待通知到达。 |
| [`global_sync_all`](/zh/api/kernel/resource-management/global-sync-all) | 本次 kernel 启动内全部 AI Core 之间的同步。 | 模式 0 + 模式 2 | 核间同步易用性接口，一次调用依次完成模式 2、模式 0、模式 0、模式 2 四次同步，完成全部 AIC 与全部 AIV 的全核同步。 |

各模式具体的计数与阻塞过程，请参考[关键特性说明](/zh/api/kernel/resource-management/key-features)。
