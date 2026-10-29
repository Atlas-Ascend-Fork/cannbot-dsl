# mega_recurrent_kda

依赖版本：CANNBot-DSL 0.8.0。

## 产品支持情况

- Ascend 950。
- TP4、TP8、TP16；模型 batch 为 1～32，`S=8`、`H=7168`、head dimension 为 128。
- 支持的 AIC 配额为 8、16、24、28、32，对应 AIV 配额至少为 AIC 的两倍。

## 功能说明

`mega_recurrent_kda` 将 QKV、decay、beta、output-gate 投影、causal Conv1D + SiLU、Q/K 归一化、recurrent KDA、RMSNorm、output gating 和输出投影融合为一次 mixed AIC/AIV kernel 调用。

全部实现位于 [`mega_recurrent_kda.py`](mega_recurrent_kda.py)。公开接口包含 20 个参数，核数由设备配额决定。算子读取输入设备当前 stream 的有效 AIC 配额，超过 32 时按 32 启动；截断后的配额须属于上述支持集合，AIV 配额也必须满足要求。

### TP 与 batch 的对应关系

`N=96/TP`，`D=128`，`P=N*D`。本样例的实际输入为 `[B,8,7168]`，其中 `B=model_batch*TP`。模型 batch 和算子输入的第一维 B 含义不同：

| TP | N | P | 模型 batch | 算子 B |
|---:|---:|---:|:---|:---|
| 4 | 24 | 3072 | 1～32 | 4、8、…、128 |
| 8 | 12 | 1536 | 1～32 | 8、16、…、256 |
| 16 | 6 | 768 | 1～32 | 16、32、…、512 |

`rms_norm_eps` 控制输出 RMSNorm，作为运行时 FP32 scalar 传入；Q/K 归一化固定使用 `1e-6`。

## 函数原型

```python
mega_recurrent_kda(
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
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    rms_norm_eps: float = 1e-5) -> torch.Tensor
```

模块路径为 `mega_recurrent_kda.mega_recurrent_kda`。除 `rms_norm_eps` 外均为必选参数。

## 参数说明

输入布局固定为 BSH。下表的 B、N、P 由上方 TP 对应关系决定，`S=8`、`H=7168`、`D=128`。

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
| `A_log` | Tensor | 必选 | 每个 head 的 log decay 参数。 | float32 / ND | `[N]` |
| `dt_bias` | Tensor | 必选 | 每个 head、每个通道的 decay bias。 | float32 / ND | `[N,D]` |
| `scale` | float | 必选 | 有限的 query scale，常用值为 `128 ** -0.5`。 | float32 | - |
| `lower_bound` | float | 必选 | decay gate 下界，范围为 `[-5,0]`。 | float32 | - |
| `rms_norm_eps` | float | 可选 | 有限且大于零的输出 RMSNorm epsilon，默认值为 `1e-5`。 | float32 | - |

## 返回值说明

返回连续的 BF16 Tensor，shape 为 `[B,S,H]`，位于输入设备。`conv_state` 和 `recurrent_state` 同时原地更新。

## 约束说明

- 所有 Tensor 必须连续并位于同一 NPU，B 满足对应 TP 的范围和整除约束。
- 六个投影权重使用 `FRACTAL_NZ`，逻辑 shape 保持输出通道在前。传入 allocation-root Tensor，不能传 view；`torch.nn.Parameter` 先调用 `.detach()`。
- beta 权重对完整 `[N,H]` allocation root 做 NZ 转换，保留物理 padding。
- `conv1d_weight` 和 `output_norm_weight` 保持 ND。兼容 Conv 布局每次调用都会转换，推荐预先准备 `[3,N,4,D]`。
- `scale` 有限；`lower_bound` 属于 `[-5,0]`；`rms_norm_eps` 有限且大于零。这三个 scalar 均以运行时 FP32 值传入。
- 计算全部 S 个 proposal token；accepted 计数选择初始历史和 state，不截断输出。
- 一个设备上复用已编译的 callable。workspace 按设备和 shape 缓存；共享 workspace 的调用必须由调用方串行化。

### 状态池语义

Conv 索引范围为 `[0,C)`。槽 0 是不可写的空槽，对应 Conv 输出为零；同一次调用中的非零索引不能重复。历史起点为 `max(conv_num_accepted_tokens[b]-1,0)`，要求 `L>=S+2` 且起点加 3 不超过 L。每个非空槽的 `[0:2]` 写入选中历史的最后两行，`[2:S+2]` 写入当前 raw QKV；其他行和未选槽保持不变。

Recurrent 索引范围为 `[0,R)`，槽 0 有效，且 `R>=B*S`。初始状态来自 `ssm_state_indices[b*S+max(num_accepted_tokens[b]-1,0)]`；每个 token 的新状态写入 `ssm_state_indices[b*S+t]`。不同 batch 使用的 recurrent 槽不能重叠。

### Workspace

GM workspace 包含 7 个 BF16 区域，行数均为 `B*S`，宽度为 `(P,P,P,P,P,16*ceil(N/16),128)`，另有 FP32 `[128,128]` norm scratch。

为在同一编译契约内处理三种 NZ 权重布局，内部保留未使用权重槽的存储描述；完整 TP sweep 约保留 370 MiB 的未使用槽存储。当前 TP 的实际权重直接传入，不复制到这些槽中。调用方应在设备内存预算中计入这部分缓存。

## 调用示例

在仓库根目录执行时设置 `PYTHONPATH=samples`，或将 samples 加入 Python 模块搜索路径。以下变量按参数表预先准备，六个投影权重在初始化时转为 NZ，后续直接复用：

```python
import torch
import torch_npu

from mega_recurrent_kda.mega_recurrent_kda import mega_recurrent_kda


def to_nz(weight):
    value = weight.detach().to(device="npu", dtype=torch.bfloat16).contiguous()
    return torch_npu.npu_format_cast(value, torch_npu.Format.FRACTAL_NZ)


qkv_weight = to_nz(qkv_weight)
decay_a_weight = to_nz(decay_a_weight)
decay_b_weight = to_nz(decay_b_weight)
beta_weight = to_nz(beta_weight)
output_gate_weight = to_nz(output_gate_weight)
output_projection_weight = to_nz(output_projection_weight)

out = mega_recurrent_kda(hidden_states, qkv_weight, decay_a_weight, decay_b_weight, beta_weight,
    output_gate_weight, output_norm_weight, output_projection_weight, conv1d_weight, conv_state, recurrent_state,
    conv_state_indices, ssm_state_indices, conv_num_accepted_tokens, num_accepted_tokens,
    A_log, dt_bias, 128**-0.5, -5.0, 1e-5)
```

需要指定有效核配额时，由调用方设置当前 stream，再调用同一入口；算子接口保持不变：

```python
stream = torch.npu.Stream()
torch.npu.synchronize()
torch.npu.set_stream_limit(stream, cube_num=16, vector_num=32)
try:
    with torch.npu.stream(stream):
        out = mega_recurrent_kda(*arguments, 128**-0.5, -5.0, 1e-5)
        torch.npu.synchronize()
finally:
    torch.npu.synchronize()
    torch.npu.reset_stream_limit(stream)
```

示例中的 `arguments` 为前 17 个 Tensor 参数。切换 stream 前必须保证输入已就绪；读取结果前等待计算完成。

## 编译与精度测试

需要可用的 CANN、CANNBot-DSL、PyTorch 和 torch_npu 环境。仓库根目录可运行：

```bash
PYTHONPATH=samples python -c 'from mega_recurrent_kda.mega_recurrent_kda import export_mega_recurrent_kda; export_mega_recurrent_kda()'
python -m pytest -q test/mega_recurrent_kda/test_mega_recurrent_kda.py -m npu
```

测试脚本内联独立 CPU golden，作为 NPU 输出和状态的参考，不依赖其他算子的测试文件。覆盖 TP4/8/16、当前硬件与 stream 支持的 8/16/24/28/32 配额、短 batch、尾块及重复调用。切换 stream 前显式等待输入与参考状态初始化完成。数值容差为 `atol=rtol=0.005`，不可变槽位使用零容差。

Snapshot 检查全部 8 个 token 的 FP32 状态快照及完整 Conv 状态池，包含 TP8 模型 batch 25 的回归用例。

## 性能

测试设备为 Ascend 950，CANN 9.2.0-beta.2。

性能测试覆盖 TP4/8/16 × 模型 batch 1/8/16/32 × 8/16/24/32 AIC，共 48 个代表点，精度检查全部通过。表中核数为 AIC 数，对应 AIV 配额为其两倍；32 AIC 列使用 32 AIC / 64 AIV。

下表为代表点的设备耗时均值，单位 μs。每点采集 20 次样本，每次恢复初始状态并驱逐 224 MiB L2；状态恢复、清 L2 和采集导出不计入设备耗时。

| TP | 模型 batch | 算子 batch | 8 AIC avg μs | 16 AIC avg μs | 24 AIC avg μs | 32 AIC avg μs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 4 | 269.818 | 180.236 | 186.214 | 141.687 |
| 4 | 8 | 32 | 1037.511 | 567.784 | 476.270 | 380.544 |
| 4 | 16 | 64 | 2009.339 | 1055.065 | 842.428 | 636.223 |
| 4 | 32 | 128 | 3963.959 | 2041.346 | 1550.625 | 1236.537 |
| 8 | 1 | 8 | 194.358 | 133.502 | 131.743 | 104.715 |
| 8 | 8 | 64 | 1035.579 | 563.970 | 438.012 | 336.539 |
| 8 | 16 | 128 | 2000.528 | 1059.746 | 808.080 | 596.761 |
| 8 | 32 | 256 | 3972.262 | 2045.795 | 1498.241 | 1176.206 |
| 16 | 1 | 16 | 208.798 | 160.513 | 148.304 | 75.407 |
| 16 | 8 | 128 | 1039.502 | 568.232 | 440.519 | 321.586 |
| 16 | 16 | 256 | 2053.105 | 1065.384 | 819.283 | 599.755 |
| 16 | 32 | 512 | 4096.337 | 2101.541 | 1560.862 | 1168.986 |
