# commit_recurrent_kda

`commit_recurrent_kda` 按每个请求接受的 token 前缀，将 PA replay 记录提交到对应的 PA checkpoint 块。
对每个接受的 token，按原顺序执行 `state = state * decay + outer(U,K)`。
只原地更新 checkpoint，所有记录和控制张量保持只读。

## 维度定义

| 符号 | 含义 |
|---|---|
| `B` | 本次逻辑 batch 数，由 `state_indices` 长度确定。 |
| `block_num` | PA 池的块数量，由第 0 轴确定；不是函数入参，也不是计算核数。 |
| `N` | head 数；与 Mega 配合时采用相同 N。 |
| `D` | head 维度，当前实现固定为 `D=128`。 |
| `S` | replay 池每块的 token 容量。 |

## 函数原型

```python
def commit_recurrent_kda(
    checkpoint_state: torch.Tensor,
    replay_u: torch.Tensor,
    replay_k: torch.Tensor,
    replay_decay: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state_indices: torch.Tensor,
) -> None:
    ...
```

公开接口只有六个 Tensor，不接收 `block_num` 或 `max_batch`。
计算核配额自动选择当前 stream 支持的 8、24、28、32 AIC 配额之一；需要限核时由调用方设置 stream limit。

## 参数与存储布局

| 参数 | 类型/格式 | shape | 语义 |
|---|---|---|---|
| `checkpoint_state` | FP32 / ND | `[block_num,N,D,D]` | PA checkpoint 池，原地更新。 |
| `replay_u` | FP32 / ND | `[block_num,N,S,D]` | PA U 记录池，只读。 |
| `replay_k` | FP16 / ND | `[block_num,N,S,D]` | PA 归一化 K 记录池，只读。 |
| `replay_decay` | FP32 / ND | `[block_num,N,S,D]` | PA decay 记录池，只读。 |
| `num_accepted_tokens` | int32 / ND | `[B]` | 每个请求接受的前缀长度。 |
| `state_indices` | int32 / ND | `[B]` | 每个请求在 checkpoint 和三个记录池中共用的 PA 块号。 |

四个池的块数一致，三个记录池的完整 shape 一致。
每个池都允许第 0 轴非连续，其余轴必须连续；各自的 `stride(0)` 独立传入内核，可以彼此不同。
第 0 轴步长不得小于一个块的元素数，块之间不能重叠。

请求 `b` 读取 `replay_u[state_indices[b]]`、`replay_k[state_indices[b]]`、
`replay_decay[state_indices[b]]`，并更新 `checkpoint_state[state_indices[b]]`。

## 约束与原位语义

- 所有输入为同一 NPU 上的普通 Tensor，使用 ND；Parameter 先 `.detach()`。
- 两个控制张量连续；池只允许第 0 轴非连续。
- 块号范围为 `[0,block_num)`，同次调用互不重复；块 0 是普通有效块。
- 接受计数在 `[0,S]` 内，且不得超过该请求本轮实际生成的有效记录长度；后者由调用方保证。
- replay 必须来自对应 checkpoint 的同一轮生成，不能使用过期或尚未写入的记录。
- 每个请求按 token 顺序提交前缀，接受计数为 0 时，该 checkpoint 块完全不变。
- 未选 checkpoint 块、PA 间隙、全部 replay 和控制张量保持不变。
- writable checkpoint 不得与其他输入的有效存储区重叠，也不得设置 `requires_grad=True`。
- 保留原有行分块、状态预取和记录队列调度；接口用于推理。

## 图模式说明

本文的图捕获指 `torch_npu.npu.NPUGraph` / `torch_npu.npu.graph` 使用的 ACL Graph 流捕获机制。
捕获期间跳过设备张量的输入值校验，以及包装层的事件等待和事件记录；静态参数校验保留。
普通调用（包括捕获前的预热）继续执行原有值校验。编译须在捕获前完成，
调用方负责图及输入输出资源的生命周期和串行使用；内核对索引的处理保持不变。

代码中的 `torch.library` 注册和 `substitute_in_graph` 用于 PyTorch 算子分发、Meta 推导以及
TorchDynamo 计算图中的函数替换；这些注册和捕获期间的校验跳过不代替实际硬件上的捕获与重放验收。

## 调用示例

从仓库根目录将 `samples` 加入 `PYTHONPATH` 后运行。

```python
from attention.commit_recurrent_kda.commit_recurrent_kda import commit_recurrent_kda

# 池来自同一轮 Mega，shape 及步长满足上面的约束。
state_indices = torch.tensor([5, 0], dtype=torch.int32, device=checkpoint_state.device)
num_accepted_tokens = torch.tensor([4, 0], dtype=torch.int32, device=checkpoint_state.device)

commit_recurrent_kda(
    checkpoint_state,
    replay_u,
    replay_k,
    replay_decay,
    num_accepted_tokens,
    state_indices,
)
# 读取记录块 5 并更新 checkpoint 块 5；checkpoint 块 0 不变。
```

返回 `None`。Python、Torch schema 和 Meta 使用相同的六 Tensor 签名。
Native 导出名保持 `commit_recurrent_kda`，Torch 入口为
`torch.ops.cannbotdsl_commit_recurrent_kda.commit_recurrent_kda`。

`InputValidation` 仅负责参数和存储约束校验；`KDAExecutionGuard.serialized` 负责调用同步和异步资源生命周期保护，与 Mega 共享锁和待完成事件状态。
