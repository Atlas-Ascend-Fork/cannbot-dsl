# mega_recurrent_kda

`mega_recurrent_kda` 融合投影、causal Conv1D/SiLU、KDA 递推、RMSNorm、输出门控及输出投影。
它读取 PA checkpoint，原地更新 Conv 池和三个 replay 池；接受候选前缀后，由 Commit 更新 checkpoint。

## 维度定义

| 符号 | 含义 |
|---|---|
| `B` | 本次调用的逻辑 batch 数，由输入或 `state_indices` 长度确定。 |
| `block_num` | PA 池的块数量，由张量第 0 轴确定；不是函数入参，也不是计算核数。 |
| `N` | 当前 TP 的 head 数。 |
| `D` | head 维度，当前实现固定为 `D=128`。 |
| `H` | hidden 维度，当前实现固定为 `H=7168`。 |
| `P` | 投影通道数，`P=N*D`。 |
| `seq_len` | BSH 输入本轮的实际序列长度。 |
| `total_tokens` | TH 输入的总 token 数。 |
| `S` | replay 池每个块的 token 容量，至少覆盖本次最长序列；可大于 `seq_len`。 |
| `L` | Conv 池每个块的缓存行数，至少覆盖本次最长序列长度加 2。 |

| TP | N |
|---:|---:|
| 4 | 24 |
| 8 | 12 |
| 16 | 6 |

| 文件 | 职责 |
|---|---|
| [mega_recurrent_kda.py](mega_recurrent_kda.py) | 公开接口、校验、分块几何、TP 调度、同步编排、Kernel 下发与导出。 |
| [mega_recurrent_kda_common.py](mega_recurrent_kda_common.py) | PA 池视图、块和 head 寻址、通用投影与搬运。 |
| [mega_recurrent_kda_frontend.py](mega_recurrent_kda_frontend.py) | 主投影、辅助投影、卷积以及补偿投影。 |
| [mega_recurrent_kda_backend.py](mega_recurrent_kda_backend.py) | KDA 递推、记录发布、输出门控与输出投影。 |

核数配额、任务数、预取 token 和同步标志采用各自的语义常量，彼此不混用。
等长 4 或 8，且 `B=TP*model_batch`、`model_batch` 在 1～32 内时，选择对应优化路径。其他合法形状使用变长路径。支持的有效计算核配额为 8、16、24、28、32 AIC，AIV 至少为 AIC 的两倍。

## 流水协议与 TP 几何

模型 batch 为 `b` 时，算子 batch `B=b*TP`，head 数 `N=96/TP`，投影行数
`R=B*seq_len`，投影宽度 `P=N*D`。递推的 `B*N=96*b` 不随 TP 变化，
而投影形状、权重大小和可驻留数据量随 TP 变化。核数配额还决定每波任务数和同步参与者。

| 层次 | 协议 | 当前实现 |
|---|---|---|
| 前端 | 分阶段任务池 | 通用 S4 先执行 QKV/Z 与 Conv，再执行辅助投影。`_S4StagedPipeline` 管理阶段间同步，投影和辅助阶段的资源由各 TP 提供。 |
| 前端 | 主辅投影分核 | TP8 `S4ParallelSchedule` 在一个行块内划分 QKV/Z 与 Beta、Fa/Fb 任务。 |
| 前端 | 窗口流水与 Z 后移 | `TP4DelayedProjectionSchedule`、TP8 `_HeadStreamSchedule`、TP16 `_ResidentDelayedZSchedule` 在各自窗口内重叠 Z 与递推。 |
| 后端 | 完整行块发布 | 通用 S4 `S4BackendSchedule` 按完整行块发布；TP16 的 S4 输出阶段可驻留权重。 |
| 后端 | head 分组发布 | TP4/TP8 `_HeadPanelBackend` 和 TP8 `_HeadStreamSchedule` 分组交付结果，输出投影跨组保留累加器。 |

后端的递推算术包含 Q/K 归一化、衰减与 Beta 激活、delta 计算和状态更新。
S4 顶层阶段编排由统一入口文件维护，各 TP 沿用原有的分块、配额和特殊路径。

TP8 S4 在 32 AIC、模型 batch 8 或 16 时启用跨项预取、128 行输出投影发布，
以及辅助投影 M64 和任务轮转。其余形状沿用各自的调度。

## 函数原型

```python
def mega_recurrent_kda(
    hidden_states: torch.Tensor,
    qkv_projection_weight: torch.Tensor,
    decay_projection_a_weight: torch.Tensor,
    decay_projection_b_weight: torch.Tensor,
    beta_projection_weight: torch.Tensor,
    output_gate_projection_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
    output_projection_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv_state: torch.Tensor,
    checkpoint_state: torch.Tensor,
    replay_u: torch.Tensor,
    replay_k: torch.Tensor,
    replay_decay: torch.Tensor,
    state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    rms_norm_eps: float = 1e-5,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

`A_log` 是规定的接口参数名，作为命名规范的例外，在 Python 原型、Torch schema 和调用中保持原样。

接口不接收 `block_num` 或 `max_batch`。计算核配额自动读取当前 stream；需要限核时，由调用方设置 stream limit。

## 参数与存储布局

| 参数 | 类型/格式 | shape | 语义 |
|---|---|---|---|
| `hidden_states` | BF16 / ND | `[B,seq_len,H]` 或 `[total_tokens,H]` | BSH 或 TH 输入。 |
| `qkv_projection_weight` | BF16 / NZ | `[3*P,H]` | Q、K、V 投影权重。 |
| `decay_projection_a_weight` | BF16 / NZ | `[D,H]` | 第一级 decay 投影。 |
| `decay_projection_b_weight` | BF16 / NZ | `[P,D]` | 第二级 decay 投影。 |
| `beta_projection_weight` | BF16 / NZ | `[N,H]` | beta 投影。 |
| `output_gate_projection_weight` | BF16 / NZ | `[P,H]` | 输出门控投影。 |
| `output_norm_weight` | FP32 / ND | `[D]` | 输出 RMSNorm 权重。 |
| `output_projection_weight` | BF16 / NZ | `[H,P]` | 输出投影。 |
| `conv1d_weight` | BF16 / ND | `[4,3*P]`、`[3,N,4,D]` 或 `[3,4,D]` | Conv 权重，卷积核长度为 4。 |
| `conv_state` | BF16 / ND | `[block_num,L,3*P]` | PA Conv 池，原地更新。 |
| `checkpoint_state` | FP32 / ND | `[block_num,N,D,D]` | PA checkpoint 池，只读。 |
| `replay_u` | FP32 / ND | `[block_num,N,S,D]` | PA U 记录池，原地更新。 |
| `replay_k` | FP16 / ND | `[block_num,N,S,D]` | PA 归一化 K 记录池，原地更新。 |
| `replay_decay` | FP32 / ND | `[block_num,N,S,D]` | PA decay 记录池，原地更新。 |
| `state_indices` | int32 / ND | `[B]` | 每个逻辑 batch 在五个池中共用的块号。 |
| `conv_num_accepted_tokens` | int32 / ND | `[B]` | 选择 Conv 历史窗口的计数。 |
| `A_log` | FP32 / ND | `[N]` | log decay 参数。 |
| `dt_bias` | FP32 / ND | `[N,D]` | decay bias。 |
| `scale` | FP32 scalar | — | 有限 query scale，常用 `D**-0.5`。 |
| `lower_bound` | FP32 scalar | — | decay gate 下界，在 `[-5,0]` 内。 |
| `rms_norm_eps` | FP32 scalar | — | 输出 RMSNorm epsilon，默认 `1e-5`。 |
| `cu_seqlens` | int32/int64 / ND | `[B+1]` 或 None | TH 的序列前缀和；BSH 必须为 None。 |

## PA 寻址与原位语义

五个池的块数一致，三个 replay 池的完整 shape 一致。
五个池均允许第 0 轴非连续，其余轴必须连续；每个池的 `stride(0)` 分别读取和传递，彼此可以不同。
第 0 轴步长必须不小于单个块的元素数，不允许块之间重叠。

逻辑 batch `b` 访问 `slot=state_indices[b]`。Conv、checkpoint 和 U/K/decay 都使用此物理块号。块号在 `[0,block_num)` 内，同一次调用互不重复，块 0 是普通有效块。

记录池中的元素偏移以各自 dtype 的元素为单位：

```text
slot * replay.stride(0) + (head * S + token) * D + channel
```

Mega 只写所选记录块的有效 token 前缀。未选块、PA 间隙、有效序列以外的记录尾部保持不变。
checkpoint 始终只读。Conv 历史起点为 `max(conv_num_accepted_tokens[b]-1,0)`，读取 3 行；
计数必须在 `[0,L-2]` 内。所选历史的最后两行写回 Conv 块的前两行，本轮 raw QKV 写入随后各行。

## 输入约束

- 所有 Tensor 在同一 NPU，使用普通 Tensor；Parameter 应先 `.detach()`。
- 六个投影权重预先转换为 NZ；其余输入使用 ND。五个 PA 池以外的 Tensor 必须连续。
- TH 前缀和首项为 0、末项为 `total_tokens`，严格递增；每段长度为正。
- `S` 覆盖本次最长序列，`L` 至少为本次最长序列长度加 2。
- writable 池不得与其他输入的有效存储区重叠，也不得设置 `requires_grad=True`。
- `scale`、`lower_bound`、`rms_norm_eps` 必须可表示为有限 FP32；epsilon 必须为正的正规 FP32 数。
- 接口用于推理；共享调用锁和完成事件保护工作区及池的复用。

## 图模式说明

本文的图捕获指 `torch_npu.npu.NPUGraph` / `torch_npu.npu.graph` 使用的 ACL Graph 流捕获机制。
捕获期间跳过设备张量的输入值校验，以及包装层的事件等待和事件记录。静态参数校验保留，块号可用重复的 0 占位，重放前须写入合法块号。
普通调用（包括捕获前的预热）继续执行原有值校验。编译和工作区预热须在捕获前完成，
调用方负责图与共享工作区的生命周期和串行使用。通用路径的 BSH 输入在设备上生成等距序列前缀。TH 输入仍读取 `cu_seqlens` 到主机以选择执行路径，
因此 TH 输入的捕获不在支持范围内。

代码中的 `torch.library` 注册和 `substitute_in_graph` 用于 PyTorch 算子分发、Meta 推导以及
TorchDynamo 计算图中的函数替换；这些注册和捕获期间的校验跳过不代替实际硬件上的捕获与重放验收。

## 调用示例

下例从仓库根目录将 `samples` 加入 `PYTHONPATH` 后运行。示例展示各池不同的第 0 轴步长，投影权重和其他参数按上表准备。

```python
from attention.mega_recurrent_kda.mega_recurrent_kda import mega_recurrent_kda

B, block_num, N, D = 2, 6, 12, 128
seq_len, S, L = 4, 8, 10
P = N * D
device = hidden_states.device

conv_state = torch.zeros(2*block_num, L, 3*P, dtype=torch.bfloat16, device=device)[::2]
checkpoint_state = torch.zeros(3*block_num, N, D, D, dtype=torch.float32, device=device)[::3]
replay_u = torch.zeros(2*block_num, N, S, D, dtype=torch.float32, device=device)[::2]
replay_k = torch.zeros(3*block_num, N, S, D, dtype=torch.float16, device=device)[::3]
replay_decay = torch.zeros(4*block_num, N, S, D, dtype=torch.float32, device=device)[::4]
state_indices = torch.tensor([5, 0], dtype=torch.int32, device=device)

out, conv_state, replay_u, replay_k, replay_decay = mega_recurrent_kda(
    hidden_states=hidden_states,
    qkv_projection_weight=qkv_projection_weight,
    decay_projection_a_weight=decay_projection_a_weight,
    decay_projection_b_weight=decay_projection_b_weight,
    beta_projection_weight=beta_projection_weight,
    output_gate_projection_weight=output_gate_projection_weight,
    output_norm_weight=output_norm_weight,
    output_projection_weight=output_projection_weight,
    conv1d_weight=conv1d_weight,
    conv_state=conv_state,
    checkpoint_state=checkpoint_state,
    replay_u=replay_u,
    replay_k=replay_k,
    replay_decay=replay_decay,
    state_indices=state_indices,
    conv_num_accepted_tokens=conv_num_accepted_tokens,
    A_log=A_log,
    dt_bias=dt_bias,
    scale=D**-0.5,
    lower_bound=-5.0,
)
# 第一个请求的记录写入块 5，第二个请求的记录写入块 0。
```

返回的后四项就是同名输入对象。Commit 应使用本轮生成的记录及相同 `state_indices`。
Native 仅保留统一导出入口 `mega_recurrent_kda`，按 TP16、TP8、TP4 顺序导出；
每个 TP 分别导出 S8、S4 和变长三个变体，合计九个编译入口，每个仅导出一次。
打包时包含本体、common、frontend 和 backend 四个模块，支持包内导入及平铺源码加载。各 TP 的 Kernel profile 标识保持不变。原生工作区、PA 步长等参数属于内部 ABI，不属于公开接口。

### 入口职责

`InputValidation` 负责参数和存储约束校验，`KDAExecutionGuard.serialized` 负责调用串行化、前次异步事件等待和对象引用保留。Mega 与 Commit 继续共享同一把锁和待完成事件状态。本体保留各 TP 的调度与缓冲区分配，并按 TP 隔离工作区和编译缓存。
