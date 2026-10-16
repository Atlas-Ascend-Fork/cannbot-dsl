# mega_recurrent_kda

## 产品支持情况

- Ascend 950：支持。
- AIC 核数：支持 8、16、24、28、32；自动选择不超过当前 stream 有效配额的最大支持值。

## 功能说明

`mega_recurrent_kda` 面向多 batch decode，将 QKV、decay、beta、output-gate 投影，causal Conv1D + SiLU，Q/K 归一化，recurrent KDA，RMSNorm，output gating 和输出投影融合为一次 mixed AIC/AIV kernel 调用。实现位于 [`mega_recurrent_kda.py`](mega_recurrent_kda.py)。

本版本支持 `S=8` decode 场景。一个 compiled callable 支持运行时 B，以及 8/16/24/28/32 AIC；host 通过 `get_platform_info(stream=当前 stream)` 读取有效 AIC 配额，选择不超过该配额的最大支持核数，kernel 再通过 `get_block_num()` 选择核数调度。

### 计算过程

算子按以下顺序执行：

1. 将 hidden states 投影为 Q/K/V、decay、beta 和 output gate。
2. 对 Q/K/V 执行 causal Conv1D 和 SiLU，并归一化 Q/K。
3. 按 head 递推 KDA state，对输出执行 RMSNorm 和 sigmoid gating。
4. 将六个 head 的结果拼接并投影回隐藏维度。

`rms_norm_eps` 控制第 3 步的输出 RMSNorm，作为运行时 FP32 scalar 传入。Q/K 归一化固定使用 `1e-6`。

## 函数原型

```python
mega_recurrent_kda.mega_recurrent_kda.mega_recurrent_kda(
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
    conv_state_indices: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    rms_norm_eps: float = 1e-5,
) -> torch.Tensor
```

模块路径为 `mega_recurrent_kda.mega_recurrent_kda`。除 `rms_norm_eps` 外均为必选。

## 参数说明

`B` 为 batch 数，`S` 为 proposal token 数，`H=7168`，`N=6`，`D=128`，`P=N*D=768`。输入布局固定为 BSH。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型/存储 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `hidden_states` | Tensor | 必选 | 输入 hidden states。 | bfloat16 / ND | `[B,S,H]` |
| `qkv_projection_weight` | Tensor | 必选 | Q/K/V 投影，输出通道顺序为 Q、K、V。 | bfloat16 / NZ | `[3*P,H]` |
| `decay_projection_a_weight` | Tensor | 必选 | decay 第一级投影。 | bfloat16 / NZ | `[D,H]` |
| `decay_projection_b_weight` | Tensor | 必选 | decay 第二级投影。 | bfloat16 / NZ | `[P,D]` |
| `beta_projection_weight` | Tensor | 必选 | 每个 head 的 beta 投影。 | bfloat16 / NZ | `[N,H]` |
| `output_gate_projection_weight` | Tensor | 必选 | output-gate 投影。 | bfloat16 / NZ | `[P,H]` |
| `output_norm_weight` | Tensor | 必选 | 各 head 共享的 RMSNorm 权重。 | float32 / ND | `[D]` |
| `output_projection_weight` | Tensor | 必选 | 将 gated head 输出投影回隐藏维度。 | bfloat16 / NZ | `[H,P]` |
| `conv1d_weight` | Tensor | 必选 | Q/K/V Conv1D 权重；推荐使用首个 shape。 | bfloat16 / ND | `[3,N,4,D]`；兼容 `[4,3*P]`、`[3,4,D]` |
| `conv_state` | Tensor | 必选 | Conv state 池，原地更新。 | bfloat16 / ND | `[C,L,3*P]` |
| `recurrent_state` | Tensor | 必选 | recurrent state 池，原地更新。 | float32 / ND | `[R,N,D,D]` |
| `conv_state_indices` | Tensor | 必选 | 每个 batch 的 Conv state 槽位。 | int32 / ND | `[B]` |
| `ssm_state_indices` | Tensor | 必选 | batch-major token 对应的 recurrent state 槽位。 | int32 / ND | `[B*S]` |
| `conv_num_accepted_tokens` | Tensor | 必选 | 选择 Conv 历史起点的 accepted 计数。 | int32 / ND | `[B]` |
| `num_accepted_tokens` | Tensor | 必选 | 选择初始 recurrent state 的 accepted 计数。 | int32 / ND | `[B]` |
| `a_log` | Tensor | 必选 | 每个 head 的 log decay 参数。 | float32 / ND | `[N]` |
| `dt_bias` | Tensor | 必选 | 每个 head、每个通道的 decay bias。 | float32 / ND | `[N,D]` |
| `scale` | float | 必选 | 有限的 query scale，常用值为 `128 ** -0.5`。 | float32 | - |
| `lower_bound` | float | 必选 | decay gate 下界，范围为 `[-5,0]`。 | float32 | - |
| `rms_norm_eps` | float | 可选 | 有限且大于零的输出 RMSNorm epsilon，默认值为 `1e-5`。 | float32 | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `out` | Tensor | 必选 | 输出 hidden states，连续且位于输入设备。 | bfloat16 | `[B,S,H]` |

算子还会原地更新 `conv_state` 和 `recurrent_state`。

## 约束说明

- 仅支持 BSH：`B=16,32,…,512`，`S=8`，`H=7168`，`N=6`，`D=128`。
- 所有 Tensor 必须连续并位于同一 NPU；启动核数从输入设备当前 stream 的有效 AIC 配额读取，选择不超过该配额的最大支持核数。
- 六个投影权重必须使用 `FRACTAL_NZ`，逻辑 shape 保持输出通道在前。必须传 allocation-root Tensor，不能传 view；`torch.nn.Parameter` 先调用 `.detach()`。
- beta 权重必须对完整逻辑 `[6,H]` allocation root 做 NZ 转换，保留物理 padding。
- `conv1d_weight` 和 `output_norm_weight` 保持 ND。兼容 Conv 布局会在每次调用时转换，推荐预先准备 `[3,N,4,D]`。
- `scale`、`lower_bound`、`rms_norm_eps` 均作为运行时 FP32 scalar 传入；修改这些值不会重新编译。
- 算子计算全部 S 个 proposal token；accepted 计数只选择初始历史和 state，不截断输出。
- 一个 compiled callable 复用动态 B、8/16/24/28/32 AIC 和 state 池大小。workspace 尺寸或设备变化时重新分配；并发调用须由调用方串行化。

### 状态池语义

Conv 索引范围为 `[0,C)`。槽 0 是不可写的空槽，对应 Conv 输出为零；同一次调用中的非零索引不能重复。历史起点为 `max(conv_num_accepted_tokens[b]-1,0)`，要求 `L>=S+2` 且起点加 3 不超过 L。

Recurrent 索引范围为 `[0,R)`，槽 0 有效，且 `R>=B*S`。初始状态来自 `ssm_state_indices[b*S+max(num_accepted_tokens[b]-1,0)]`；每个 token 的新状态写入 `ssm_state_indices[b*S+t]`。不同 batch 使用的 recurrent 槽不能重叠。

### 调度与 workspace

运行时只有两个 workload 分支：

- 单 group：`B=16` 且 AIC 为 24/28/32。三种核数共用两轮 recurrent 与逐轮 OProj 主路径；24 AIC 仅在该分支内部采用 18 个 QKV core 加 6 个 auxiliary core 的 frontend 映射。8/16 AIC 的 `B=16` 继续走全局任务池，避免单 group 栅栏开销。
- 全局自适应任务池：全部 8/16 AIC case，以及 24/28/32 AIC 的 `B>=32`。frontend 根据任务量在 panel 粒度和 M128/N384 粒度之间动态选择；backend 每 slot 完成最多 `2*AIC` 个 batch/head recurrent 任务，累计每完成 48 个任务便发布一个 128 行 gated chunk 并启动 OProj。

不同 batch 大小复用同一 compiled callable；运行时 B 和 `get_block_num()` 选择对应调度。

| AIC 核数 | OProj N tile | OProj KS | 每核 tile 数上限 | tile stride | L0C depth | resident input | resident weight |
|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 8 | 128 | 64 | 7 | 1 | 2 | 是 | 否 |
| 16 | 256 | 64 | 2 | 16 | 2 | 否 | 否 |
| 24 | 256 | 64 | 2 | 24 | 2 | 否 | 否 |
| 28 | 256 | 64 | 1 | 1 | 1 | 否 | 是 |
| 32 | 256 | 64 | 1 | 1 | 1 | 否 | 是 |

GM workspace 按 `B*S` 分配 7 个 BF16 区域，宽度为 `(768,768,768,768,768,16,128)`，另有 FP32 `[16*S,128]` norm scratch。

## 确定性计算

当前样例未提供逐位确定性保证，精度测试按数值容差校验。

## 调用示例

六个投影权重在模型初始化时转换一次，后续调用直接复用：

```python
import torch
import torch_npu

from mega_recurrent_kda.mega_recurrent_kda import mega_recurrent_kda

def to_nz(weight):
    return torch_npu.npu_format_cast(
        weight.detach().to(device="npu", dtype=torch.bfloat16).contiguous(),
        torch_npu.Format.FRACTAL_NZ,
    )

qkv_weight = to_nz(qkv_weight)
decay_a_weight = to_nz(decay_a_weight)
decay_b_weight = to_nz(decay_b_weight)
beta_weight = to_nz(beta_weight)
output_gate_weight = to_nz(output_gate_weight)
output_projection_weight = to_nz(output_projection_weight)

out = mega_recurrent_kda(
    hidden_states,
    qkv_weight, decay_a_weight, decay_b_weight, beta_weight,
    output_gate_weight, output_norm_weight, output_projection_weight,
    conv1d_weight, conv_state, recurrent_state,
    conv_state_indices, ssm_state_indices,
    conv_num_accepted_tokens, num_accepted_tokens,
    a_log, dt_bias,
    128 ** -0.5, -5.0, 1e-5,
)
```

## 精度测试

在仓库根目录执行单文件测试。测试内联独立 PyTorch FP32 golden，直接校验输出、Conv state 和每个 recurrent state snapshot，不依赖其他 golden 文件：

```bash
python -m pytest -q test/mega_recurrent_kda/test_mega_recurrent_kda.py
python -m pytest -q test/mega_recurrent_kda/test_mega_recurrent_kda.py --whitebox
```

输出和 state snapshot 使用 `atol=rtol=0.005`；应保持不变的槽位使用零容差。8/24 AIC 精度验证另外以相同输入逐项对比 microbatch 的输出、Conv state 和 recurrent state，覆盖 B16/B64；最大绝对误差不超过 `1.907e-6`（输出）和 `3.987e-8`（recurrent state），Conv state 逐位一致。

## 性能

以下结果于 2026-09-17 使用 Ascend950PR_9599、CANN 9.2.0 和 CANNBot-DSL 采集。固定 `S=8`、`rms_norm_eps=1e-5`、NZ 权重和 seed 123；每个配置先执行 10 次 warmup，再保留 20 次逐次清 L2 的连续 sample，表中为 Task Duration 中位数。

| B | 8 AIC (us) | 24 AIC (us) | 28 AIC (us) | 32 AIC (us) |
|---:|---:|---:|---:|---:|
| 16 | 158.944 | 79.017 | 76.136 | 76.687 |
| 32 | - | 149.534 | 123.800 | 120.870 |
| 48 | - | 167.280 | - | 154.899 |
| 64 | 518.759 | 247.880 | 228.880 | 199.003 |
| 128 | 1003.562 | 512.858 | - | - |
| 256 | 1996.899 | 1039.709 | 953.063 | 916.243 |
