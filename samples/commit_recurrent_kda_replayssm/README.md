# commit_recurrent_kda_replayssm

依赖版本：CANNBot-DSL 0.8.0。

## 产品支持情况

- Ascend 950：支持。
- 控核：公开参数支持 8、24、28、32 AIC 等效配额；内核实际启动不超过 `2*block_num` 个 AIV。

## 功能说明

`commit_recurrent_kda_replayssm` 根据 Mega ReplaySSM 保存的 replay record，
将每个 batch 已接受的 token 依次提交到递归状态。实现位于
[`commit_recurrent_kda_replayssm.py`](commit_recurrent_kda_replayssm.py)。

### 计算过程

对 batch `b` 的前 `num_accepted_tokens[b]` 个 token，算子按顺序执行：

$$
S \leftarrow S \odot decay + U K^{\top}.
$$

算子原地更新 `recurrent_state`，不修改 replay record。

一个动态 compiled callable 覆盖所有正数 `B`、`S`、`N` 和四种控核配置。模块同时提供
native binary 注册和 `torch.compile` 图模式注册。

## 函数原型

```python
commit_recurrent_kda_replayssm.commit_recurrent_kda_replayssm.commit_recurrent_kda_replayssm(
    recurrent_state: torch.Tensor,
    replay_u: torch.Tensor,
    replay_k: torch.Tensor,
    replay_decay: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    block_num: Optional[int] = None,
) -> None
```

模块路径为
`commit_recurrent_kda_replayssm.commit_recurrent_kda_replayssm`。
除 `block_num` 外，其余参数均为必选。
动态图直接调用上面的 Python 函数；图模式仍调用同一函数，Dynamo 会将其替换为
`torch.ops.cannbotdsl_commit_recurrent_kda_replayssm.commit_recurrent_kda_replayssm`。
native binary 注册名为 `commit_recurrent_kda_replayssm`。

## 参数说明

`B`、`S`、`N` 为正数运行时维度，`D = 128`。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `recurrent_state` | Tensor | 必选 | 输入 checkpoint；算子原地提交已接受 token。 | float32 | `[B, N, D, D]` |
| `replay_u` | Tensor | 必选 | ReplaySSM 保存的 U。 | float16 | `[B, S, N, D]` |
| `replay_k` | Tensor | 必选 | ReplaySSM 保存的 K。 | float16 | `[B, S, N, D]` |
| `replay_decay` | Tensor | 必选 | ReplaySSM 保存的 decay。 | float32 | `[B, S, N, D]` |
| `num_accepted_tokens` | Tensor | 必选 | 每个 batch 提交的 token 数。 | int32 | `[B]` |
| `block_num` | int | 可选 | AIC 配额；可取 `8`、`24`、`28`、`32`，默认自动选择。 | - | - |

## 返回值说明

算子不返回张量。计算结果写入 `recurrent_state`。

## 约束说明

- `B`、`S`、`N` 只要求为正数，`D = 128` 固定；不设置人为的 batch、序列或 head 上限。
- `num_accepted_tokens` 的每个元素必须位于 `[0, S]`。
- 所有输入张量必须连续、位于同一 NPU，并满足参数表中的 dtype 和 shape。
- `block_num=None` 时，算子读取当前 stream 的有效 AIC 配额，并选择不超过该配额的最大支持值。
- 显式 `block_num` 不得超过设备当前 stream 的有效 AIC 配额。
- Commit 内核最多启动 `2 * block_num` 个 AIV；实际数量由 `B` 和任务数确定。
- 算子只原地修改 `recurrent_state`。
- accepted 数值不拷回 host 校验，调用方必须保证每个值位于 `[0,S]`。
- 一个 compiled callable 复用运行时 B、S、N 和四种控核配置；相同编译缓存目录下这些形状共用一个 `.so`。

### 状态与调度语义

对每个 batch，只按 token 顺序处理 `[0,num_accepted_tokens[b])`。每个 token
对全部 head 执行 `state = state * decay + U K^T`；
`num_accepted_tokens[b]=0` 时该 batch 的 checkpoint 保持不变。Replay record
和 accepted 计数只读。

内核按 batch/head 拆分 state 行，并根据工作量选择 32、64 或 128 行任务。公开的
AIC 等效配额映射到 AIV launch grid，实际 AIV 数量按任务数收缩或饱和在
`2 * block_num`。一个动态 B 二进制覆盖全部分支，不分配 GM workspace。

## 确定性计算

当前样例未提供逐位确定性保证，精度测试按数值容差校验。

## 调用示例

```python
import torch
import torch_npu

from commit_recurrent_kda_replayssm.commit_recurrent_kda_replayssm import (
    commit_recurrent_kda_replayssm,
)

B, S, N, D = 16, 8, 6, 128
recurrent_state = torch.zeros(B, N, D, D, dtype=torch.float32).npu()
replay_u = torch.randn(B, S, N, D, dtype=torch.float16).npu()
replay_k = torch.randn(B, S, N, D, dtype=torch.float16).npu()
replay_decay = torch.rand(B, S, N, D, dtype=torch.float32).npu()
num_accepted_tokens = torch.full((B,), 4, dtype=torch.int32).npu()

commit_recurrent_kda_replayssm(
    recurrent_state,
    replay_u,
    replay_k,
    replay_decay,
    num_accepted_tokens,
    block_num=32,
)
```

## 精度测试

在仓库根目录运行独立测试：

```bash
python -m pytest -q \
  test/commit_recurrent_kda_replayssm/test_commit_recurrent_kda_replayssm.py
```

测试使用 CPU FP32 递推作为 golden，覆盖 `B=1/7/16/32/256`、`N=1/6/32/96`、
`S=1/4/8/16`、`num_accepted_tokens=0/1/S`、混合 accepted token 数、
`block_num=8/24/28/32`、动态图接口和 `torch.compile(fullgraph=True)` 注册。
State 使用数值容差校验，replay record 和 accepted 计数保持不变。

## 性能

以下结果于 2026-10-08 在当前 Ascend NPU 上使用 CANN 9.2、CANNBot-DSL
和 msprof 采集。shape 为 `B=16, N=6, S=8, D=128`，accepted 为 4；每个
控核配置预热 10 次后连续采样 30 次，表中数据按要求取 `Task Duration` 最小值。

| B | N | S | accepted tokens | `block_num` | 实际 AIV | Task Duration 最小值 (us) |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 6 | 8 | 4 | 32 | 64 | 7.087 |
| 16 | 6 | 8 | 4 | 28 | 56 | 7.639 |
| 16 | 6 | 8 | 4 | 24 | 48 | 8.122 |
| 16 | 6 | 8 | 4 | 8 | 16 | 16.955 |
