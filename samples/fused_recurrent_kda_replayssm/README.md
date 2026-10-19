# fused_recurrent_kda_replayssm

基于 CANNBot-DSL 的独立 ReplaySSM verify 算子，面向 Ascend 950。实现位于 [`fused_recurrent_kda_replayssm.py`](fused_recurrent_kda_replayssm.py)。算子从只读 FP32 checkpoint 计算全部候选 token，返回 KDA 输出，并写出后续 Commit 使用的 replay record。

## 功能说明

输入 Q/K/V 已完成投影与卷积。算子融合 Q/K 行级 L2 normalize、gate 与 beta 激活、FP32 状态递推和 replay record 写出。对一个 batch、一个 head，token 从 0 编号，令 $Q_t,K_t,V_t\in\mathbb{R}^{D}$ 为第 $t$ 个 token 的原始输入，$S_t\in\mathbb{R}^{D_v\times D_k}$ 为处理该 token 前的递归状态，$D_k=D_v=D=128$，则

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

其中 $\sigma(x)=(1+e^{-x})^{-1}$，$S_0=$ `recurrent_state[b]`。gate 和 decay 按 key 通道计算，$\beta_t$ 是每个 token、每个 head 的标量。递推使用 FP32，输出转换为 BF16。记录保存为

$$
\operatorname{replay\_u}_t=\operatorname{FP16}(u_t),\qquad
\operatorname{replay\_k}_t=\operatorname{FP16}(k_t),\qquad
\operatorname{replay\_decay}_t=\operatorname{FP32}(d_t).
$$

保存的 K 是归一化后的 key，decay 是激活后的指数值，U 是已乘 beta 的修正向量。FP16 U/K 只用于导出记录，本轮后续 token 的递推继续使用 FP32 工作状态。

### Verify 与 Commit

Verify 不修改 `recurrent_state`，也不接收 accepted 参数。接受候选后，调用独立的 [`commit_recurrent_kda_replayssm`](../commit_recurrent_kda_replayssm/README.md)，从同一个 checkpoint 按顺序提交接受前缀。对每个 batch 的前 $a_b$ 个 token，Commit 使用保存的记录执行

$$
S\leftarrow S\operatorname{diag}(\operatorname{replay\_decay}_t)
+\operatorname{replay\_u}_t\operatorname{replay\_k}_t^{\top},\qquad
t=0,\ldots,a_b-1.
$$

$a_b=0$ 时 checkpoint 保持不变。复用记录缓冲区前，须先完成上一轮接受前缀的 Commit；下一次 Verify 会覆盖本轮有效 token 对应的记录。Commit 使用经过 FP16 舍入的 U/K，结果按数值容差与 Verify 的 FP32 工作状态对照。

## 快速开始

在仓库根目录将 `samples` 加入 `PYTHONPATH` 后运行以下示例。三个 replay record 张量必须由调用方预先分配，并通过关键字传入；`out` 由算子内部新建。

```bash
export PYTHONPATH="$PWD/samples${PYTHONPATH:+:$PYTHONPATH}"
```

```python
import torch
import torch_npu

from fused_recurrent_kda_replayssm.fused_recurrent_kda_replayssm import (
    fused_recurrent_kda_replayssm,
)
from commit_recurrent_kda_replayssm.commit_recurrent_kda_replayssm import (
    commit_recurrent_kda_replayssm,
)

B, S, N, D = 2, 8, 6, 128
q = torch.randn(B, S, N, D, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
mixedqkv = torch.cat((q, k, v), dim=2).flatten(2).contiguous()
g = torch.randn_like(q)
beta = torch.randn(B, S, N, 1, device="npu", dtype=torch.bfloat16)
A_log = torch.zeros(N, device="npu", dtype=torch.float32)
dt_bias = torch.zeros(N, D, device="npu", dtype=torch.float32)
checkpoint = torch.zeros(B, N, D, D, device="npu", dtype=torch.float32)
replay_u = torch.empty(B, S, N, D, device="npu", dtype=torch.float16)
replay_k = torch.empty_like(replay_u)
replay_decay = torch.empty(B, S, N, D, device="npu", dtype=torch.float32)

out = fused_recurrent_kda_replayssm(
    mixedqkv, checkpoint, beta, g, None, A_log, dt_bias,
    layout_qkv="BSND",
    replay_u=replay_u, replay_k=replay_k, replay_decay=replay_decay,
)
# Verify 后 checkpoint 不变；out 的 shape 为 [B, S, N, D]。
accepted_tokens = torch.tensor([4, 0], device="npu", dtype=torch.int32)
commit_recurrent_kda_replayssm(
    checkpoint, replay_u, replay_k, replay_decay, accepted_tokens,
)
# Commit 原地更新 checkpoint：第一个 batch 提交 4 个 token，第二个保持不变。
```

## 函数原型

```python
fused_recurrent_kda_replayssm(
    mixedqkv: torch.Tensor,
    recurrent_state: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float | None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = -5.0,
    layout_qkv: str = "BSND",
    *,
    replay_u: torch.Tensor,
    replay_k: torch.Tensor,
    replay_decay: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    block_num: int | None = None,
) -> torch.Tensor
```

`scale` 必须传入，可用 `None` 选择默认缩放值；`*` 后的参数只能通过关键字传入。动态图和 `torch.compile` 图模式使用同一 Python 函数，图模式注册名为 `torch.ops.cannbotdsl_fused_recurrent_kda_replayssm.fused_recurrent_kda_replayssm`。

## 参数说明

`B` 是 batch 数，`S` 是稠密布局中每条序列的长度；TND 下，`S` 表示每个 batch 的 replay record 容量。`T` 是 TND 的 packed token 总数，`N` 是 Q/K/V 共用的头数，`D=128`，`P=N*D`。

| 参数 | BSND shape | 类型 | 必选 / 默认值 | 说明 |
|:---|:---|:---|:---|:---|
| `mixedqkv` | `[B,S,3*P]` | BF16 | 必选 | 按完整 Q、K、V 三段打包；输入已完成投影与卷积。 |
| `recurrent_state` | `[B,N,D,D]` | FP32 | 必选 | 只读 checkpoint，最后两维依次为 Dv、Dk。 |
| `beta` | `[B,S,N,1]` | BF16 | 必选 | 原始写入 logits，算子内执行 sigmoid；末维 1 必须保留。 |
| `g` | `[B,S,N,D]` | BF16 | 必选 | 原始 gate，由 `A_log`、`dt_bias`、`lower_bound` 激活。 |
| `scale` | 标量 | float / None | 必选 | 有限的 Query 缩放值；`None` 使用 $1/\sqrt{D}$。 |
| `A_log` | `[N]` | FP32 | 必选 | 每个 head 的 log 时间尺度。 |
| `dt_bias` | `[N,D]` | FP32 | 必选 | 每个 head、每个 key 通道的 gate 偏置。 |
| `lower_bound` | 标量 | float | `-5.0` | gate 下界，范围 `[-5,0]`。 |
| `layout_qkv` | 标量 | str | `"BSND"` | `"BSND"`、`"BNSD"` 或 `"TND"`。 |
| `replay_u` | `[B,S,N,D]` | FP16 | 必选，关键字参数 | 调用方分配，原地写入已乘 beta 的修正向量 U。 |
| `replay_k` | `[B,S,N,D]` | FP16 | 必选，关键字参数 | 调用方分配，原地写入归一化后的 K。 |
| `replay_decay` | `[B,S,N,D]` | FP32 | 必选，关键字参数 | 调用方分配，原地写入激活后的 decay 指数值。 |
| `cu_seqlens` | `[B+1]` | int32 / None | `None`；TND 必传 | packed token 前缀和；BSND/BNSD 不使用它划分序列。 |
| `block_num` | 标量 | int / None | `None` | AIC 等效配额，可取 8、24、28、32；默认按当前 stream 自动选择。 |

### 布局与记录容量

| 张量 | BSND | BNSD | TND |
|:---|:---|:---|:---|
| `mixedqkv` | `[B,S,3*N*D]` | `[B,3*N,S,D]` | `[T,3*N*D]` |
| `g`、`out` | `[B,S,N,D]` | `[B,N,S,D]` | `[T,N,D]` |
| `beta` | `[B,S,N,1]` | `[B,N,S,1]` | `[T,N,1]` |
| `replay_u`、`replay_k`、`replay_decay` | `[B,S,N,D]` | `[B,S,N,D]` | `[B,S,N,D]` |

BNSD 的 head 维按 Q、K、V 各 `N` 个 head 排列；BSND/TND 在每个 token 内将这三段展平。Replay record 始终按 batch、token、head、channel 存储，不随输入布局改变。

BSND/BNSD 通常按 `[B,S,N,D]` 分配记录，每个 batch 写入 `[0,S)`；也可预留更大的第二维，未写入的后缀保持原值。TND 的 $L_b=$ `cu_seqlens[b+1] - cu_seqlens[b]`，要求 `1<=L_b<=S`；packed token `cu_seqlens[b]+t` 的记录写入 `[b,t]`，各 batch 从记录位置 0 开始。记录的 `[L_b,S)` 后缀保持原值。

## 返回值说明

Verify 返回一个 `out` 张量，三个 replay record 通过传入的缓冲区原地输出，不在返回值中重复返回。

| 输出 | Shape | 类型 | 说明 |
|:---|:---|:---|:---|
| `out` | 与 `g` 相同 | BF16 | 算子内部新建的全部候选 token 输出，布局与 `g` 相同。 |
| `replay_u`、`replay_k` | `[B,S,N,D]` | FP16 | 原地输出，供 Commit 使用；仅本轮有效 token 的记录被覆盖。 |
| `replay_decay` | `[B,S,N,D]` | FP32 | 原地输出，供 Commit 使用；未使用的后缀保持不变。 |

`recurrent_state` 在 Verify 中始终只读；只有后续 Commit 才会将接受结果原地写入 checkpoint。

## 约束说明

- 支持 Ascend 950；`B>0`、序列长度为正，`1<=N<=96`，`Dk=Dv=128`。Q/K/V 头数必须相同。
- 所有张量必须连续并位于同一 NPU，dtype 和 shape 须满足上表。三个 replay record 使用独立缓冲区，容量和 shape 必须一致。
- BSND/BNSD 的物理 batch 数必须与 `recurrent_state` 一致；TND 的 `cu_seqlens` 必须为 int32 `[B+1]`，从 0 严格递增到 `T`，每条序列的长度不得超过 `S`。NPU 前缀和的数值由调用方保证合法。
- `scale` 必须有限，`lower_bound` 必须位于 `[-5,0]`。序列长度按实际输入与记录容量确定，不固定为 8。
- `block_num=None` 时选择当前 stream 配额内最大的支持值；显式值只能取 8、24、28、32，且不得超过有效配额。内核最多启动 `2*block_num` 个 AIV，实际数量按工作量收缩。
- Verify 总是计算全部候选 token。Commit 的 `num_accepted_tokens[b]` 须位于 `[0,L_b]`，不能提交未写入的记录后缀；该计数仅传给 Commit。
- 一个动态 compiled callable 复用输入布局、shape 与启动配额。

## 精度测试

在仓库根目录运行，测试需要 Ascend NPU：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python -m pytest -q test/fused_recurrent_kda_replayssm/test_fused_recurrent_kda_replayssm.py
```

3 个精度 case 覆盖 BSND/BNSD/TND、32/64/128 行分块、跨 wave 调度、
变长序列、不同 accepted 前缀及记录后缀保留；另有 `B=1,N=1,S=1,D=128`
的最小全量精度 case。
独立 CPU FP32 reference 检查输出、U/K/decay 记录和 Commit 后的状态，精度容差
为 `atol=rtol=0.005`；只读 checkpoint、accepted=0 和未使用的记录后缀使用零容差。
