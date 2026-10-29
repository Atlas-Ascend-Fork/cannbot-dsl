# fused_recurrent_kda_snapshot

依赖版本：CANNBot-DSL 0.8.0。

基于 CANNBot-DSL 的 Kimi Delta Attention decode 融合算子，面向 Ascend 950。实现位于 [`fused_recurrent_kda_snapshot.py`](fused_recurrent_kda_snapshot.py)，公开 Python 函数名为 `fused_recurrent_kda`。算子从 state pool 读取初态，计算本轮全部候选 token，并为每个 token 保存完整递归状态快照。

## 功能说明

输入 Q/K/V 已完成投影与卷积。算子融合 Q/K 行级 L2 normalize、gate 与 beta 激活、逐 token KDA 递推和状态快照写回。对一个 batch、一个 head，token 从 0 编号，令 $Q_t,K_t,V_t\in\mathbb{R}^{D}$ 为第 $t$ 个 token 的原始输入，$S_t\in\mathbb{R}^{D_v\times D_k}$ 为处理该 token 前的递归状态，$D_k=D_v=D=128$，则

$$
q_t=\frac{Q_t}{\sqrt{\sum_{j=1}^{D}Q_{t,j}^{2}+10^{-6}}},\qquad
k_t=\frac{K_t}{\sqrt{\sum_{j=1}^{D}K_{t,j}^{2}+10^{-6}}}.
$$

$$
\beta_t=\sigma(\beta_t^{\mathrm{raw}}),\qquad
\ell_t=\operatorname{lower\_bound}\cdot
\sigma\!\left(e^{A_{\log}}(g_t+\operatorname{dt\_bias})\right),\qquad
d_t=e^{\ell_t}.
$$

$$
\begin{aligned}
\bar S_t&=S_t\operatorname{diag}(d_t),\\
u_t&=\beta_t\left(V_t-\bar S_t k_t\right),\\
S_{t+1}&=\bar S_t+u_t k_t^{\top},\\
o_t&=\operatorname{scale}\cdot S_{t+1}q_t.
\end{aligned}
$$

其中 $\sigma(x)=(1+e^{-x})^{-1}$。gate 和 decay 按 key 通道计算，$\beta_t$ 是每个 token、每个 head 的标量。Q/K 归一化、激活和工作状态递推使用 FP32；输出转为 BF16，每个快照转为 `state` 的 dtype。BF16 快照写回不会截断本轮后续 token 使用的 FP32 工作状态。

### 状态槽位与接受计数

以下索引均从 0 开始。令 $I=$ `ssm_state_indices`、$a_b=$ `num_accepted_tokens[b]`，$p_b$ 为 batch $b$ 的 token 起点：BSND/BNSD 下 $p_b=bS$，TND 下 $p_b=$ `cu_seqlens[b]`。本轮递推初态为

$$
S_0^{(b)}=\operatorname{state}\!\left[I\!\left[p_b+\max(a_b-1,0)\right]\right].
$$

第 $t$ 个候选计算后，$S_{t+1}^{(b)}$ 写入槽位 $I[p_b+t]$。`num_accepted_tokens` 只选择前一轮快照，不截断本轮输出；本轮仍计算全部候选并覆盖对应槽位。计数为 0 或 1 时均从该序列第一个 token 的槽位读取，因此 0 不会自动恢复一个独立的未接受状态。

不传 `ssm_state_indices` 时，使用从 0 到 token 总数减一的连续槽位；不传 `num_accepted_tokens` 时，每个 batch 默认取 1。调用前须将所选初态写入对应槽位。下一轮调用前，根据接受结果更新计数或槽位映射。

## 快速开始

在仓库根目录将 `samples` 加入 `PYTHONPATH` 后运行以下示例。`state` 由调用方分配并原地更新；`out` 由算子内部新建。

```bash
export PYTHONPATH="$PWD/samples${PYTHONPATH:+:$PYTHONPATH}"
```

```python
import torch
import torch_npu

from fused_recurrent_kda_snapshot.fused_recurrent_kda_snapshot import (
    fused_recurrent_kda,
)

B, S, N, D = 2, 8, 6, 128
q = torch.randn(B, S, N, D, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
# 每个 token 依次存放全部 Q heads、全部 K heads、全部 V heads。
mixedqkv = torch.cat((q, k, v), dim=2).flatten(2).contiguous()
g = torch.randn_like(q)
beta = torch.randn(B, S, N, 1, device="npu", dtype=torch.bfloat16)
A_log = torch.zeros(N, device="npu", dtype=torch.float32)
dt_bias = torch.zeros(N, D, device="npu", dtype=torch.float32)
state = torch.zeros(B * S, N, D, D, device="npu", dtype=torch.float32)
ssm_state_indices = torch.arange(B * S, device="npu", dtype=torch.int32)
num_accepted_tokens = torch.ones(B, device="npu", dtype=torch.int32)

updated_state, out = fused_recurrent_kda(
    mixedqkv, state, beta, g, None, A_log, dt_bias,
    layout_qkv="BSND",
    ssm_state_indices=ssm_state_indices,
    num_accepted_tokens=num_accepted_tokens,
)
# updated_state 就是原地更新后的 state；out 的 shape 为 [B, S, N, D]。
```

## 函数原型

```python
fused_recurrent_kda(
    mixedqkv: torch.Tensor,
    state: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float | None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = -5.0,
    layout_qkv: str = "BSND",
    *,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

`scale` 必须传入，可用 `None` 选择默认缩放值；`*` 后的参数只能通过关键字传入。

## 参数说明

`B` 是 batch 数，`S` 是稠密布局中每条序列的长度，`T` 是 TND 的 packed token 总数，`N` 是 Q/K/V 共用的头数，`D=128`，`P=N*D`。`C` 是 state pool 的槽位数，`L_b` 是 batch $b$ 的有效长度。令 `L=B*S`（BSND/BNSD）或 `L=T`（TND）。

| 参数 | BSND shape | 类型 | 必选 / 默认值 | 说明 |
|:---|:---|:---|:---|:---|
| `mixedqkv` | `[B,S,3*P]` | BF16 | 必选 | 按完整 Q、K、V 三段打包；输入已完成投影与卷积。 |
| `state` | `[C,N,D,D]` | BF16 / FP32 | 必选 | 初态与逐 token 快照共用的池，原地更新；最后两维依次为 Dv、Dk。 |
| `beta` | `[B,S,N,1]` | BF16 | 必选 | 原始写入 logits，算子内执行 sigmoid；末维 1 必须保留。 |
| `g` | `[B,S,N,D]` | BF16 | 必选 | 原始 gate，由 `A_log`、`dt_bias`、`lower_bound` 激活。 |
| `scale` | 标量 | float / None | 必选 | Query 缩放值；`None` 使用 $1/\sqrt{D}$。 |
| `A_log` | `[N]` | FP32 | 必选 | 每个 head 的 log 时间尺度。 |
| `dt_bias` | `[N,D]` | FP32 | 必选 | 每个 head、每个 key 通道的 gate 偏置。 |
| `lower_bound` | 标量 | float | `-5.0` | gate 下界，范围 `[-5,0]`。 |
| `layout_qkv` | 标量 | str | `"BSND"` | `"BSND"`、`"BNSD"` 或 `"TND"`。 |
| `cu_seqlens` | `[B+1]` | int32 / None | `None`；TND 必传 | packed token 前缀和；BSND/BNSD 不使用它划分序列。 |
| `ssm_state_indices` | `[L]` | int32 / None | `None` | 每个 token 对应的状态槽位；默认连续映射。 |
| `num_accepted_tokens` | `[B]` | int32 / None | `None` | 前一轮接受数，用于选择输入快照；默认每个 batch 为 1。 |

### 布局说明

| 张量 | BSND | BNSD | TND |
|:---|:---|:---|:---|
| `mixedqkv` | `[B,S,3*N*D]` | `[B,3*N,S,D]` | `[T,3*N*D]` |
| `g`、`out` | `[B,S,N,D]` | `[B,N,S,D]` | `[T,N,D]` |
| `beta` | `[B,S,N,1]` | `[B,N,S,1]` | `[T,N,1]` |
| `ssm_state_indices` | `[B*S]` | `[B*S]` | `[T]` |

BNSD 的 head 维按 Q、K、V 各 `N` 个 head 排列；BSND/TND 在每个 token 内将这三段展平。三种布局的状态池均为 `[C,N,D,D]`。稠密布局的槽位索引始终按 batch、token 展平，与 head 维的物理位置无关。

TND 的 `cu_seqlens` 为 `[B+1]` 前缀和，$L_b=$ `cu_seqlens[b+1] - cu_seqlens[b]`；各序列独立选择初态并递推。

## 返回值说明

Python 函数返回 `(state, out)`。

| 返回值 | Shape | 类型 | 说明 |
|:---|:---|:---|:---|
| `state` | `[C,N,D,D]` | 与传入 `state` 相同 | 与输入是同一张量，映射槽位已写入本轮逐 token 快照；其余槽位保持不变。 |
| `out` | 与 `g` 相同 | BF16 | 算子内部新建的全部候选 token 输出，布局与 `g` 相同。 |

模块还提供同签名的 `fused_recurrent_kda_functional`，同样原地更新并返回 `state`；名称中的 functional 不表示复制状态池。

## 约束说明

- 支持 Ascend 950；`B>0`、序列长度为正，`1<=N<=96`，`Dk=Dv=128`。Q/K/V 头数必须相同。
- 所有公开输入张量必须连续并位于同一 NPU，dtype 和 shape 须满足上表。
- BSND/BNSD 要求 `C>=B*S`；TND 要求 `C>=T`。`ssm_state_indices` 的每项须位于 `[0,C)`，同一次调用中使用互不重复的槽位，避免快照覆盖或跨序列读写冲突。
- `num_accepted_tokens[b]` 须位于 `[0,L_b]`。它用于选择历史快照，不表示本轮只计算这些 token。
- TND 的 `cu_seqlens` 必须从 0 严格递增到 `T`，每条序列非空。索引、接受数和 NPU 前缀和的数值不拷回 host 校验，由调用方保证合法。
- `lower_bound` 必须位于 `[-5,0]`。当前接口没有 ReplaySSM 的 `block_num` 参数，由算子根据工作量和当前 stream 的 AIV 配额调度。

## 精度测试

在仓库根目录运行，测试需要 Ascend NPU：

```bash
python -m pytest -q test/fused_recurrent_kda_snapshot/test_fused_recurrent_kda_snapshot.py
```

当前回归包含 BSND、BNSD 和 TND 三个精度场景，使用 FP32 state，覆盖非连续槽位映射和接受数 0、4、8。输出与更新后的状态池均与独立 PyTorch golden 对照，容差为 `atol=rtol=5e-3`。

## 性能

测试使用 CANN 9.2、Release 构建和 msprof `Task Duration` 采集。固定形状为 `B=16,N=6,S=8,D=128`，采用 BSND、FP32 state 和 `accepted=4`；先运行 10 次 warmup，再保留 100 个 Snapshot 任务。

| Task Duration 中位数 (us) | 最小值 (us) | 平均值 (us) | P90 (us) | 最大值 (us) |
|--:|--:|--:|--:|--:|
| 18.0585 | 16.758 | 17.9963 | 18.702 | 19.258 |
