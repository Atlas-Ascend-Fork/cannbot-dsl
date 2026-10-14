# 关键特性说明

本章节配合时序图介绍四种核间同步控制模式各自实现的原理。

下述同步特性均以如下场景配置为例：group 配置为 1：2（即每个 block 由 1 个 AIC 与 2 个 AIV 构成），
kernel 启动 2 个 block，即共启动 2 个 AIC 和 4 个 AIV。
为便于描述，将 2 个 AIC 分别编号为 AIC0、AIC1；block 0 中的 2 个 AIV 分别编号为 AIV0-0、AIV0-1，
block 1 中的 2 个 AIV 分别编号为 AIV1-0、AIV1-1，如图 1 所示。
**各核中与 `flag_id` 或 `sync_id` 对应的计数器初始值均为 0。**

**图 1**  同步控制模式示意图

![同步控制模式示意图](/figures/3510_sync_control_mode_diagram.png)

## 多 AI Core 中 AIC 或者 AIV 全核同步（模式 0）

多 AI Core 中 AIC 或者 AIV 全核同步需要配套使用模式 0 的 `cube_sync_block_arrive`/`cube_sync_block_wait`
（AIC 侧）或 `vec_sync_block_arrive`/`vec_sync_block_wait`（AIV 侧）接口，两侧的 `mode` 均取 `0`。
该场景可细分为两类：

- 2 个 AI Core 中的 2 个 AIC 进行全核同步。

    当 2 个 AIC 都执行完 `cube_sync_block_arrive` 时，每个 AIC 对应 `flag_id` 的计数器增加 1。
    若 AIC 该 `flag_id` 的计数器为非 0，被 `cube_sync_block_wait` 所阻塞的指定流水中的指令才会执行下去，
    该 `flag_id` 的计数器减去 1。

    对于每个 AIC，`cube_sync_block_arrive` 和 `cube_sync_block_wait` 都必须配对使用。

- 2 个 AI Core 中的 4 个 AIV 进行全核同步。

    当 4 个 AIV 都执行完 `vec_sync_block_arrive` 时，每个 AIV 对应 `flag_id` 的计数器增加 1。
    若 AIV 该 `flag_id` 的计数器为非 0，被 `vec_sync_block_wait` 所阻塞的指定流水中的指令才会执行下去，
    该 `flag_id` 的计数器减去 1。

    对于每个 AIV，`vec_sync_block_arrive` 和 `vec_sync_block_wait` 都必须配对使用。

以图 2 为例，演示 2 个 AI Core 中的 2 个 AIC（AIC0、AIC1）进行全核同步，代码片段如下：

```python
from cannbotdsl.ops.sync import PIPE, cube_sync_block_arrive, cube_sync_block_wait


@kernel
def all_aic_sync_kernel(source: Tensor, destination: Tensor):
    # 前序计算与写入……

    # 待前置 PIPE_FIXPIPE 流水任务完成后，通知调度模块。
    cube_sync_block_arrive(PIPE.FIXPIPE, 0, mode=0)
    # 直到所有 AIC 都完成前置 PIPE_FIXPIPE 流水，调度模块更新 flag_id=0 的计数器后，
    # 解除 PIPE_S 流水后续指令的阻塞。
    cube_sync_block_wait(PIPE.S, 0, mode=0)

    # 后续计算与写入……
```

AIC0 中在执行 `cube_sync_block_wait` 后，此时 AIC0 `flag_id=0` 的计数器为 0，
后续 `PIPE_S` 流水中的指令被阻塞，需要等到 2 个 AIC 核均执行完 `cube_sync_block_arrive`。

AIC1 的 `cube_sync_block_arrive` 执行完后，此时调度模块感知到 2 个 AIC 均已执行完
`cube_sync_block_arrive`，因此所有 AIC 各自的 `flag_id=0` 的计数器值增加为 1。
AIC0 和 AIC1 检测到各自对应的 `flag_id=0` 的计数器变为 1，都解除 `PIPE_S` 流水后续指令的阻塞，
继续执行后续指令，并且将计数器值减去 1。

**图 2**  模式 0：多 AI Core 中 AIC 全核同步

![模式 0：多 AI Core 中 AIC 全核同步](/figures/inter_core_aic_all_sync.png)

## 单个 AI Core 中 AIV 全核同步（模式 1）

该场景需要配套使用模式 1 的 `vec_sync_block_arrive` 和 `vec_sync_block_wait` 接口，
两侧的 `mode` 均取 `1`。

1 个 AI Core 中的 2 个 AIV 进行全核同步。当 2 个 AIV 全部都执行完 `vec_sync_block_arrive` 时，
每个 AIV 对应 `flag_id` 的计数器增加 1。若 AIV 该 `flag_id` 的计数器为非 0，
被 `vec_sync_block_wait` 所阻塞的指定流水中的指令才会执行下去，该 `flag_id` 的计数器减去 1。
对于每个 AIV，`vec_sync_block_arrive` 和 `vec_sync_block_wait` 都必须配对使用。

> [!CAUTION]注意
> 该模式中，不同 AI Core 的 AIV 核之间不会互相影响同步。

以图 3 为例，演示 block 0 中的 2 个 AIV（AIV0-0、AIV0-1）进行全核同步，代码片段如下：

```python
from cannbotdsl.ops.sync import PIPE, vec_sync_block_arrive, vec_sync_block_wait


@kernel
def aiv_subblock_sync_kernel(source: Tensor, destination: Tensor):
    # 前序计算与写入……

    # 待前置 PIPE_MTE2 流水任务完成后，通知调度模块。两个 AIV 都必须执行。
    vec_sync_block_arrive(PIPE.MTE2, 0, mode=1)
    # 直到该 AI Core 中的所有 AIV 都完成前置 PIPE_MTE2 流水，
    # 调度模块更新 flag_id=0 的计数器后，解除 PIPE_S 流水后续指令的阻塞。
    vec_sync_block_wait(PIPE.S, 0, mode=1)

    # 后续计算与写入……
```

AIV0-0 中在执行 `vec_sync_block_wait` 后，此时 AIV0-0 `flag_id=0` 的计数器为 0，
后续 `PIPE_S` 流水中的指令被阻塞，需要等到 2 个 AIV 均执行完 `vec_sync_block_arrive`。

AIV0-1 的 `vec_sync_block_arrive` 执行完后，此时调度模块感知到 2 个 AIV 均已执行完
`vec_sync_block_arrive`，因此将 AIV0-0 和 AIV0-1 各自的 `flag_id=0` 的计数器值增加为 1。
AIV0-0 和 AIV0-1 检测到各自对应的 `flag_id=0` 的计数器变为 1，都解除 `PIPE_S` 流水后续指令的阻塞，
继续执行后续指令，并且将计数器值减去 1。

**图 3**  模式 1：单个 AI Core 中 AIV 全核同步

![模式 1：单个 AI Core 中 AIV 全核同步](/figures/single_core_aiv_all_sync.png)

## 单个 AI Core 中 AIC 与 AIV 全核同步（模式 2）

单个 AI Core 中 AIC 与 AIV 全核同步需要配套使用模式 2 的 `cube_sync_block_arrive`/`cube_sync_block_wait`
和 `vec_sync_block_arrive`/`vec_sync_block_wait` 接口，两侧的 `mode` 均取 `2`。
该场景可细分为两类：

- 1 个 AI Core 中的 AIC 执行 `cube_sync_block_arrive`，对应的 2 个 AIV 都执行 `vec_sync_block_wait`。

    当 AIC 执行完 `cube_sync_block_arrive` 时，2 个 AIV 对应 `flag_id` 的计数器增加 1。
    若 AIV 该 `flag_id` 的计数器为非 0，则解除 AIV 由 `pipe` 参数指定的流水后续指令的阻塞，
    后续指令继续发射，该 `flag_id` 的计数器减去 1。

    总计 AIC 1 次调用 `cube_sync_block_arrive`，对应 2 个 AIV 各 1 次 `vec_sync_block_wait` 才算配对使用。

- 1 个 AI Core 中的 2 个 AIV 都执行 `vec_sync_block_arrive`，对应的 AIC 执行 `cube_sync_block_wait`。

    当 2 个 AIV 全都执行完 `vec_sync_block_arrive` 时，AIC 对应 `flag_id` 的计数器增加 1。
    若 AIC 该 `flag_id` 的计数器为非 0，则解除 AIC 由 `pipe` 参数指定的流水后续指令的阻塞，
    后续指令继续发射，该 `flag_id` 的计数器减去 1。

    总计 2 个 AIV 各 1 次 `vec_sync_block_arrive`，对应 AIC 1 次调用 `cube_sync_block_wait` 才算配对使用。

以图 4 为例，两个方向合在一个 Mix 核函数中演示（AIV 发起 `vec_sync_block_arrive`），代码片段如下：

```python
from cannbotdsl.ops.sync import (
    PIPE,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    vec_sync_block_arrive,
    vec_sync_block_wait,
)


@kernel
def block_sync_kernel(source: Tensor, destination: Tensor):
    # AIV 侧：前序计算与写入……

    # 每个 AIV 都执行 arrive，AIC 等待。
    vec_sync_block_arrive(PIPE.MTE2, 0, mode=2)
    cube_sync_block_wait(PIPE.S, 0, mode=2)

    # AIC 侧：前序计算与写入……

    # AIC 执行 arrive，每个 AIV 都等待。
    cube_sync_block_arrive(PIPE.MTE2, 1, mode=2)
    vec_sync_block_wait(PIPE.S, 1, mode=2)

    # 后续计算与写入……
```

以 block 0 中的 AIC0 与 AIV0-0、AIV0-1 为例（AIV 发起 `vec_sync_block_arrive`）：

AIC0 中在执行 `cube_sync_block_wait` 后，此时 AIC0 `flag_id=0` 的计数器为 0，
后续 `PIPE_S` 流水中的指令被阻塞，需要等到 2 个 AIV 均执行完 `vec_sync_block_arrive`。

当 AIV0-1 的 `vec_sync_block_arrive` 在 Vector 指令执行完后，前置 `PIPE_MTE2` 的指令全部完成，
`vec_sync_block_arrive` 执行完成，但是此时 AIV0-0 未执行 `vec_sync_block_arrive`。
因此 AIC0 `flag_id=0` 的计数器还是为 0，AIC 的 `PIPE_S` 流水中的指令依然被阻塞。

当 AIV0-0 的前置指令全部执行完毕后，`vec_sync_block_arrive` 生效。此时调度模块感知到 2 个 AIV
均已执行完 `vec_sync_block_arrive`，因此将 AIC0 `flag_id=0` 的计数器值增加为 1。
AIC0 检测到对应的 `flag_id=0` 的计数器变为 1，则 AIC0 核解除 `PIPE_S` 流水后续指令的阻塞，
继续执行后续指令，并且将计数器值减去 1。

**图 4**  模式 2：单个 AI Core 中 AIC 与 AIV 全核同步（AIV 进行 `vec_sync_block_arrive`）

![模式 2：单个 AI Core 中 AIC 与 AIV 全核同步（AIV 进行 vec_sync_block_arrive）](/figures/single_core_aic_aiv_sync_aiv_arrive.png)


## 单个 AI Core 中 AIC 与单个 AIV 同步（模式 4）

单个 AI Core 中 AIC 与单个 AIV 同步需要配套使用模式 4 的 `cube_sync_intra_arrive`/`cube_sync_intra_wait`
和 `vec_sync_intra_arrive`/`vec_sync_intra_wait` 接口。该场景可细分为两类：

- 1 个 AI Core 中的 AIC 执行 `cube_sync_intra_arrive`，对应的 AIV 执行 `vec_sync_intra_wait`。

    当 AIC 执行完 `cube_sync_intra_arrive` 时，AIV 对应 `sync_id` 的计数器增加 1。
    若 AIV 该 `sync_id` 的计数器为非 0，则解除 AIV 由 `pipe` 参数指定的流水的阻塞，
    后续指令继续发射，该 `sync_id` 的计数器减去 1。

    总计 AIC 1 次调用 `cube_sync_intra_arrive`，单个 AIV 完成一次 `vec_sync_intra_wait` 才算配对使用。

- 1 个 AI Core 中的 1 个 AIV 执行 `vec_sync_intra_arrive`，对应的 AIC 执行 `cube_sync_intra_wait`。

    当 AIV 执行完 `vec_sync_intra_arrive` 时，AIC 对应 `sync_id` 的计数器增加 1。
    若 AIC 该 `sync_id` 的计数器为非 0，则解除 AIC 由 `pipe` 参数指定的流水的阻塞，
    后续指令继续发射，该 `sync_id` 的计数器减去 1。

    总计单个 AIV 1 次 `vec_sync_intra_arrive`，对应 AIC 1 次调用 `cube_sync_intra_wait` 才算配对使用。

以图 5 为例，演示 block 0 中的 AIV0-1 与 AIC0 进行同步（AIV0-1 发起 `vec_sync_intra_arrive`），
两个方向合在一起演示，代码片段如下：

```python
from cannbotdsl.ops.sync import (
    PIPE,
    cube_sync_intra_arrive,
    cube_sync_intra_wait,
    vec_sync_intra_arrive,
    vec_sync_intra_wait,
)


@kernel
def intra_sync_kernel(source: Tensor, destination: Tensor):
    # AIV 侧：前序计算与写入……

    # AIV0-1 上的 sync_id 0~15 对应 AIC0 上的 sync_id 16~31。
    vec_sync_intra_arrive(PIPE.MTE3, 0)
    cube_sync_intra_wait(PIPE.S, 0)
    cube_sync_intra_wait(PIPE.S, 16)

    # AIC 侧：前序计算与写入……

    # AIC 分两次发起，分别通知 AIV0 与 AIV1；两个 AIV 都在本地 sync_id 上等待。
    cube_sync_intra_arrive(PIPE.MTE2, 0)
    cube_sync_intra_arrive(PIPE.MTE2, 16)
    vec_sync_intra_wait(PIPE.S, 0)

    # 后续计算与写入……
```

以 AIV0-1 向 AIC0 发起 `vec_sync_intra_arrive` 为例：

AIC0 中在执行 `cube_sync_intra_wait` 后，此时 AIC0 `sync_id=16` 的计数器为 0，
后续由 `pipe` 参数指定的流水（此处为 `PIPE_S`）的指令全部被阻塞，
需要等到 1 个 AIV 执行完 `vec_sync_intra_arrive`。

- AIV0-0 不需要执行 `vec_sync_intra_arrive`，也不需要参与本次同步。
- AIV0-1 的前置 `PIPE_MTE2` 指令全部执行完毕后，`vec_sync_intra_arrive` 生效。
  此时调度模块感知 1 个 AIV 已执行完 `vec_sync_intra_arrive`，因此将 AIC0 `sync_id=16`
  的计数器值增加为 1。AIC0 检测到对应的 `sync_id=16` 的计数器变为 1，
  则 AIC0 核解除由 `pipe` 参数指定的流水后续指令的阻塞，继续执行后续指令，并且将计数器值减去 1。

**图 5**  模式 4：单个 AI Core 中 AIC 与单个 AIV 同步（AIV 进行 `vec_sync_intra_arrive`）

![模式 4：单个 AI Core 中 AIC 与单个 AIV 同步（AIV 进行 vec_sync_intra_arrive）](/figures/single_ai_core_aic_single_aiv_sync.png)
