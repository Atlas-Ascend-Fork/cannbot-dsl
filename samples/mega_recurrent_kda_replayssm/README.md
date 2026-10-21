# mega_recurrent_kda_replayssm

## 产品支持情况

- Ascend 950。
- TP4、TP8、TP16；模型 batch 为 1～32，`S=8`、`H=7168`、head dimension 为 128。
- AIC 核数：支持 8、16、24、28、32；同时检查物理 AIC/AIV 核数和当前 stream 的有效配额，选择安全范围内的最大支持核数；至少需要 8 AIC 和 16 AIV。

## 功能说明

`mega_recurrent_kda_replayssm` 执行 ReplaySSM verify，将 QKV、decay、beta、
output-gate 投影，causal Conv1D + SiLU，recurrent KDA、RMSNorm、output
gating 和输出投影融合为一次 mixed AIC/AIV kernel 调用。实现位于
[`mega_recurrent_kda_replayssm.py`](mega_recurrent_kda_replayssm.py)。

算子读取 FP32 checkpoint `recurrent_state`，计算全部 8 个候选 token，
并写出 Commit 所需的 replay record：

- `replay_u`：FP16 `[B, 8, N, 128]`；
- `replay_k`：FP16 `[B, 8, N, 128]`；
- `replay_decay`：FP32 `[B, 8, N, 128]`。

算子不修改 `recurrent_state`。Conv1D 使用独立的 `conv_state_indices` 和
`conv_num_accepted_tokens`，并原地更新 `conv_state`。

全部实现位于同一个算子文件，公开接口包含 21 个参数，核数由设备配额决定。一个 compiled callable 覆盖 TP4/8/16 和 8/16/24/28/32 AIC；模块同时提供 native binary 注册和 `torch.compile` 图模式注册。

### TP 与 batch 的对应关系

`N=96/TP`，`D=128`，`P=N*D`。实际输入为 `[B,8,7168]`，其中 `B=model_batch*TP`。模型 batch 和算子输入的第一维 B 含义不同：

| TP | N | P | 模型 batch | 算子 B |
|---:|---:|---:|:---|:---|
| 4 | 24 | 3072 | 1～32 | 4、8、…、128 |
| 8 | 12 | 1536 | 1～32 | 8、16、…、256 |
| 16 | 6 | 768 | 1～32 | 16、32、…、512 |

`rms_norm_eps` 控制输出 RMSNorm，作为运行时 FP32 scalar 传入；Q/K 归一化固定使用 `1e-6`。

### 计算过程

算子按以下顺序执行：

1. 将 hidden states 投影为 Q/K/V、decay、beta 和 output gate。
2. 对 Q/K/V 执行 causal Conv1D 和 SiLU，并归一化 Q/K。
3. 从 `recurrent_state` 计算 8 个候选 token，写出 U/K/decay replay record。
4. 对 KDA 输出执行 RMSNorm、sigmoid gating 和输出投影。

## 函数原型

```python
mega_recurrent_kda_replayssm(
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
    recurrent_state: torch.Tensor,
    replay_u: torch.Tensor,
    replay_k: torch.Tensor,
    replay_decay: torch.Tensor,
    conv_state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    rms_norm_eps: float = 1e-5) -> torch.Tensor
```

模块路径为 `mega_recurrent_kda_replayssm.mega_recurrent_kda_replayssm`。
除 `rms_norm_eps` 外，其余参数均为必选。
动态图直接调用上面的 Python 函数；图模式仍调用同一函数，Dynamo 会将其替换为
`torch.ops.cannbotdsl_mega_recurrent_kda_replayssm.mega_recurrent_kda_replayssm`。
native binary 注册名为 `mega_recurrent_kda_replayssm`。

## 参数说明

输入布局固定为 BSH。下表的 B、N、P 由上方 TP 对应关系决定，`S=8`、`H=7168`、`D=128`。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型/存储 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `hidden_states` | Tensor | 必选 | 输入 hidden states。 | bfloat16 / ND | `[B, S, H]` |
| `qkv_projection_weight` | Tensor | 必选 | Q/K/V 投影，输出顺序为 Q、K、V。 | bfloat16 / NZ | `[3*P, H]` |
| `decay_projection_a_weight` | Tensor | 必选 | decay 第一级投影。 | bfloat16 / NZ | `[D, H]` |
| `decay_projection_b_weight` | Tensor | 必选 | decay 第二级投影。 | bfloat16 / NZ | `[P, D]` |
| `beta_projection_weight` | Tensor | 必选 | 每个 head 的 beta 投影。 | bfloat16 / NZ | `[N, H]` |
| `output_gate_projection_weight` | Tensor | 必选 | output-gate 投影。 | bfloat16 / NZ | `[P, H]` |
| `output_norm_weight` | Tensor | 必选 | 各 head 共享的 RMSNorm 权重。 | float32 / ND | `[D]` |
| `output_projection_weight` | Tensor | 必选 | 输出投影。 | bfloat16 / NZ | `[H, P]` |
| `conv1d_weight` | Tensor | 必选 | Q/K/V Conv1D 权重；推荐首个 shape。 | bfloat16 / ND | `[3, N, 4, D]`；兼容 `[4, 3*P]`、`[3, 4, D]` |
| `conv_state` | Tensor | 必选 | Conv state 池，原地更新。 | bfloat16 / ND | `[C, L, 3*P]` |
| `recurrent_state` | Tensor | 必选 | Verify 使用的只读 checkpoint。 | float32 / ND | `[B, N, D, D]` |
| `replay_u` | Tensor | 必选 | U replay record 输出。 | float16 / ND | `[B, S, N, D]` |
| `replay_k` | Tensor | 必选 | K replay record 输出。 | float16 / ND | `[B, S, N, D]` |
| `replay_decay` | Tensor | 必选 | decay replay record 输出。 | float32 / ND | `[B, S, N, D]` |
| `conv_state_indices` | Tensor | 必选 | 每个 batch 的 Conv state 槽位。 | int32 / ND | `[B]` |
| `conv_num_accepted_tokens` | Tensor | 必选 | 选择 Conv 历史起点的 accepted 计数。 | int32 / ND | `[B]` |
| `A_log` | Tensor | 必选 | 每个 head 的 log decay 参数。 | float32 / ND | `[N]` |
| `dt_bias` | Tensor | 必选 | 每个 head、每个通道的 decay bias。 | float32 / ND | `[N, D]` |
| `scale` | float | 必选 | 有限的 query scale。 | float32 | - |
| `lower_bound` | float | 必选 | decay gate 下界，范围为 `[-5, 0]`。 | float32 | - |
| `rms_norm_eps` | float | 可选 | 有限且大于零的 RMSNorm epsilon，默认值为 `1e-5`。 | float32 | - |

## 返回值说明

返回连续的 BF16 Tensor，shape 为 `[B,S,H]`，位于输入设备。`conv_state`、`replay_u`、`replay_k` 和 `replay_decay` 同时原地更新，`recurrent_state` 保持不变。

## 约束说明

- `TP` 为 4、8 或 16，`B / TP` 为 1～32 的整数，`N = 96 / TP`。固定 `S = 8`、`H = 7168`、`D = 128`。
- 所有 Tensor 必须连续并位于同一 NPU。
- 六个投影权重必须使用 `FRACTAL_NZ`，逻辑 shape 保持输出通道在前。
  必须传 allocation-root Tensor，不能传 view；`torch.nn.Parameter` 先调用
  `.detach()`。
- beta 权重必须对完整逻辑 `[N,H]` allocation root 做 NZ 转换，保留物理 padding。
- `conv1d_weight` 和 `output_norm_weight` 使用 ND。兼容 Conv 布局会在每次调用时转换，推荐预先准备 `[3,N,4,D]`。
- `conv_state` 要求 `L >= S + 2`。`conv_state_indices` 的每个元素必须位于
  `[0, C)`，同一次调用中的非零槽位不能重复。
- `conv_num_accepted_tokens` 的每个元素必须位于 `[0, 8]`。
- `scale`、`lower_bound`、`rms_norm_eps` 均作为运行时 FP32 scalar 传入；
  修改这些值不会重新编译。`scale` 必须有限，`lower_bound` 必须位于
  `[-5,0]`，`rms_norm_eps` 必须有限且大于零。
- 算子计算全部 8 个 proposal token，并覆盖全部 replay record；accepted 结果由后续 Commit 接口消费，不截断 verify 输出。
- 算子自动读取当前 stream 的有效 AIC 配额，选择不超过该配额的最大支持值（8/16/24/28/32）；配额低于 8 时拒绝运行。
- 索引和 accepted 数值不拷回 host 校验，调用方必须满足上述范围。
- 每个设备保留一个 compiled callable，TP、batch 和配额切换不触发重新编译。workspace 按设备、batch 和 head 数缓存，重叠调用须由调用方串行化。

### 状态语义

Conv 索引范围为 `[0,C)`。槽 0 是不可写的空槽，对应 Conv 输出为零；同一次
调用中的非零索引不能重复。历史起点为
`max(conv_num_accepted_tokens[b]-1,0)`，要求 `L>=10` 且起点加 3 不超过 L。
非空槽的最后两行所选历史写入 `[0:2]`，本轮原始 QKV 投影写入 `[2:10]`。

`recurrent_state[b]` 是 batch `b` 的只读 FP32 checkpoint。Verify 从该
checkpoint 计算全部 8 个候选 token，但不原地提交 state。每个 token 的
FP16 U、FP16 归一化 K 和 FP32 激活后 decay 分别写入 `replay_u`、
`replay_k` 和 `replay_decay`，由独立的
[`commit_recurrent_kda_replayssm`](../commit_recurrent_kda_replayssm/README.md)
按接受前缀提交。

### 调度与 workspace

当前 stream 和物理设备共同限制可用的 AIC/AIV 数，28 AIC 机器不会请求 32 AIC。
TP 由权重和 `A_log` 的 head 维度确定，不增加核数或 TP 接口参数。
TP16 在各核数下保留专用投影与递推重叠调度，32 AIC 的单组路径使用 K128 常驻输出权重。
TP4/8 主投影按行数、head 数和核数选择 N256 或 N384 tile。32 AIC 下，TP4/8 的
model batch 至少为 8 且行数整除 128 时，使用 M128/N384 主投影波次。
回放记录在归一化 K、激活 decay 和计算 U 后立即写出，FP32 递推保持不变。

GM workspace 的 7 个 BF16 区域宽度为
`(P,P,P,P,P,ceil_div(N,16)*16,128)`，另有 FP32 `[128,128]` norm scratch。
Replay U/K/decay 是调用方提供的独立输出，不属于 workspace。

## 确定性计算

当前样例未提供逐位确定性保证，精度测试按数值容差校验。

## 调用示例

在仓库根目录执行时设置 `PYTHONPATH=samples`，或将 samples 加入 Python 模块搜索路径。以下变量按参数表预先准备，六个投影权重在初始化时转为 NZ，后续直接复用：

```python
import torch
import torch_npu

from mega_recurrent_kda_replayssm.mega_recurrent_kda_replayssm import mega_recurrent_kda_replayssm


def to_nz(weight):
    value = weight.detach().to(device="npu", dtype=torch.bfloat16).contiguous()
    return torch_npu.npu_format_cast(value, torch_npu.Format.FRACTAL_NZ)


qkv_weight = to_nz(qkv_weight)
decay_a_weight = to_nz(decay_a_weight)
decay_b_weight = to_nz(decay_b_weight)
beta_weight = to_nz(beta_weight)
output_gate_weight = to_nz(output_gate_weight)
output_projection_weight = to_nz(output_projection_weight)

out = mega_recurrent_kda_replayssm(hidden_states, qkv_weight, decay_a_weight, decay_b_weight, beta_weight,
    output_gate_weight, output_norm_weight, output_projection_weight, conv1d_weight, conv_state, recurrent_state,
    replay_u, replay_k, replay_decay, conv_state_indices, conv_num_accepted_tokens,
    A_log, dt_bias, 128**-0.5, -5.0, 1e-5)
```

## 编译与精度测试

需要可用的 CANN、CANNBot-DSL、PyTorch 和 torch_npu 环境。仓库根目录可运行：

```bash
# 提前编译动态入口；常规算子首次调用也会触发编译
PYTHONPATH=samples python -c 'from mega_recurrent_kda_replayssm.mega_recurrent_kda_replayssm import export_mega_recurrent_kda_replayssm; export_mega_recurrent_kda_replayssm()'

# 设备精度与白盒检查
python -m pytest -q -m npu test/mega_recurrent_kda_replayssm/test_mega_recurrent_kda_replayssm.py
```

CPU FP32 golden 算法直接定义在此测试脚本中，不依赖其他算子的测试文件。
测试覆盖 TP4/8/16、典型短长 batch、当前设备支持的 8/16/24/28/32 配额，
并检查输出、Conv state、U/K/decay 记录、checkpoint 只读和 accepted 0～8 的 commit 状态。
数值容差为 `atol=rtol=0.005`；只读 checkpoint、空 Conv 槽和未选中槽使用零容差。

## 性能

以下为 Ascend 950、Channel API 的 DSL 0.7.1、CANN 9.2.0-beta.2 实测均值，单位 μs。
统计 Verify mixed kernel 的设备执行时间，Commit 单独验证；每点预热 3 次、采样 20 次，并清理 224 MiB L2。
共 48 个典型精度/性能点通过；NPU 白盒 21 项通过，真实 28 AIC / 56 AIV 配额下另有 6 项及 3 个最小 batch 用例通过。

| TP | model batch | 8 AIC | 16 AIC | 24 AIC | 32 AIC |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 299.725 | 185.283 | 185.874 | 137.835 |
| 4 | 2 | 377.234 | 229.044 | 217.237 | 158.785 |
| 4 | 8 | 1060.117 | 569.852 | 457.342 | 326.546 |
| 8 | 1 | 216.237 | 139.829 | 133.311 | 106.101 |
| 8 | 2 | 325.206 | 205.328 | 184.187 | 137.797 |
| 8 | 8 | 1053.266 | 566.658 | 425.238 | 293.842 |
| 16 | 1 | 163.434 | 119.479 | 84.141 | 74.458 |
| 16 | 8 | 986.518 | 492.964 | 365.058 | 278.766 |
| 16 | 16 | 1942.248 | 945.279 | 673.912 | 502.966 |
| 16 | 32 | 3870.474 | 1859.382 | 1279.752 | 961.440 |

32 AIC 下另测的较大 batch：

| TP | model batch 16 | model batch 32 |
|---:|---:|---:|
| 4 | 543.017 | 1049.055 |
| 8 | 504.679 | 985.747 |

本环境同一内容键的正常冷编译用时 10790.004 秒；上表仅统计设备执行时间。
