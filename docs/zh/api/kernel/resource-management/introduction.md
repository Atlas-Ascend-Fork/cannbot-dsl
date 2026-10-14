# Kernel API：同步管理


同步管理 API 用于协调 AIC 与 AIV 之间、以及不同 AI Core 之间的执行顺序。

## 适用范围

片上内存（L1、UB、L0 等）的同步由 `Channel` 自动生成，框架自动管理；
GM 等 workspace 上的同步可以使用本章节开放的同步接口。

## 能力概述

- [系统同步能力概述](/zh/api/kernel/resource-management/system-sync-overview)：AI Core 同步的分类、四种核间同步控制模式，以及核间同步接口总表。
- [核间同步能力概述](/zh/api/kernel/resource-management/inter-core-sync-overview)：核间同步的使用场景、核的层级结构、支持的核间同步场景与配对约束。
- [关键特性说明](/zh/api/kernel/resource-management/key-features)：四种核间同步控制模式各自的计数与阻塞过程。

## 接口

| 接口 | 所在核 | 角色 | 对应模式 |
| --- | --- | --- | --- |
| [`cube_sync_block_arrive`](/zh/api/kernel/resource-management/cube-sync-block-arrive) | AIC | 生产者，负责发送通知 | 模式 0、模式 2 |
| [`cube_sync_block_wait`](/zh/api/kernel/resource-management/cube-sync-block-wait) | AIC | 消费者，负责阻塞等待通知到达 | 模式 0、模式 2 |
| [`cube_sync_intra_arrive`](/zh/api/kernel/resource-management/cube-sync-intra-arrive) | AIC | 生产者，负责向单个 AIV 发送通知 | 模式 4 |
| [`cube_sync_intra_wait`](/zh/api/kernel/resource-management/cube-sync-intra-wait) | AIC | 消费者，负责阻塞等待单个 AIV 的通知到达 | 模式 4 |
| [`global_sync_all`](/zh/api/kernel/resource-management/global-sync-all) | AIC 与 AIV | 核间同步易用性接口，一次调用完成本次 kernel 启动内全部 AI Core 之间的同步 | 模式 0 + 模式 2 |
| [`vec_sync_block_arrive`](/zh/api/kernel/resource-management/vec-sync-block-arrive) | AIV | 生产者，负责发送通知 | 模式 0、模式 1、模式 2 |
| [`vec_sync_block_wait`](/zh/api/kernel/resource-management/vec-sync-block-wait) | AIV | 消费者，负责阻塞等待通知到达 | 模式 0、模式 1、模式 2 |
| [`vec_sync_intra_arrive`](/zh/api/kernel/resource-management/vec-sync-intra-arrive) | AIV | 生产者，负责向 AIC 发送通知 | 模式 4 |
| [`vec_sync_intra_wait`](/zh/api/kernel/resource-management/vec-sync-intra-wait) | AIV | 消费者，负责阻塞等待 AIC 的通知到达 | 模式 4 |

<p class="doc-footnote">本分类下接口的功能描述与术语参考 Ascend C 的 C API 文档。</p>
