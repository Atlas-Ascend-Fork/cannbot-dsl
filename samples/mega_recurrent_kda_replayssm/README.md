# mega_recurrent_kda_replayssm

依赖版本：CANNBot-DSL 0.8.0。

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

### Workspace

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
PYTHONPATH=samples python -c 'from mega_recurrent_kda_replayssm.mega_recurrent_kda_replayssm import export_mega_recurrent_kda_replayssm; export_mega_recurrent_kda_replayssm()'
python -m pytest -q test/mega_recurrent_kda_replayssm/test_mega_recurrent_kda_replayssm.py -m npu
```

测试脚本内联独立 CPU golden，作为 NPU 输出和状态的参考，不依赖其他算子的测试文件。覆盖 TP4/8/16、当前硬件与 stream 支持的 8/16/24/28/32 配额、短 batch、尾块及重复调用。切换 stream 前显式等待输入与参考状态初始化完成。数值容差为 `atol=rtol=0.005`，不可变槽位使用零容差。

ReplaySSM 检查 Conv、U/K/decay、checkpoint 不变性，以及 accepted 0～8 的 Commit 状态。

## 性能

测试设备为 Ascend 950，CANN 9.2.0-beta.2。

性能测试覆盖 TP4/8/16 × 模型 batch 1/8/16/32 × 8/16/24/32 AIC，共 48 个代表点，精度检查全部通过。表中核数为 AIC 数，对应 AIV 配额为其两倍；32 AIC 列使用 32 AIC / 64 AIV。

下表为代表点的设备耗时均值，单位 μs。每点采集 20 次样本，每次恢复初始状态并驱逐 224 MiB L2；状态恢复、清 L2 和采集导出不计入设备耗时。

| TP | 模型 batch | 算子 batch | 8 AIC avg μs | 16 AIC avg μs | 24 AIC avg μs | 32 AIC avg μs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 4 | 259.710 | 166.754 | 173.526 | 127.117 |
| 4 | 8 | 32 | 1056.668 | 567.641 | 454.515 | 324.819 |
| 4 | 16 | 64 | 2048.092 | 1052.527 | 805.304 | 508.952 |
| 4 | 32 | 128 | 4030.142 | 2039.602 | 1489.796 | 961.111 |
| 8 | 1 | 8 | 194.294 | 129.681 | 123.526 | 96.394 |
| 8 | 8 | 64 | 1050.179 | 563.454 | 422.861 | 281.624 |
| 8 | 16 | 128 | 2040.717 | 1068.667 | 777.891 | 496.500 |
| 8 | 32 | 256 | 4040.991 | 2054.159 | 1426.657 | 951.494 |
| 16 | 1 | 16 | 163.504 | 119.357 | 83.624 | 73.583 |
| 16 | 8 | 128 | 984.716 | 491.637 | 362.898 | 278.304 |
| 16 | 16 | 256 | 1938.819 | 938.666 | 668.475 | 501.879 |
| 16 | 32 | 512 | 3861.376 | 1851.845 | 1274.259 | 941.099 |

表中仅统计 Verify mixed kernel；后续 Replay Commit 的执行成本另计。
