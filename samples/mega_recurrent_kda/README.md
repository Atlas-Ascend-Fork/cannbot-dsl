# mega_recurrent_kda

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

### 调度与 workspace

TP4/8/16 共用一个算子文件和动态编译入口。32 AIC 使用已有的内部 TP 调度；其他支持配额根据行数、head 数和核数，在固定的 M/N tile 模板中按最忙核工作量统一选型。N384 仅用于完整行块，N256 支持尾块；没有按逐 shape 实测结果索引的调度表。

输出投影按 TP、核数和发布块大小选择权重驻留、输入复用或流式分块路径。TP16 的对应路径可以驻留输出权重分块，TP4/TP8 的当前路径采用其他复用策略；是否能进一步优化需要结合 L1 预算和流水验证。

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
# 提前编译动态入口；常规算子首次调用也会触发编译
PYTHONPATH=samples python -c 'from mega_recurrent_kda.mega_recurrent_kda import export_mega_recurrent_kda; export_mega_recurrent_kda()'

# 无设备接口检查
python -m pytest -q test/mega_recurrent_kda/test_mega_recurrent_kda.py -m 'not npu'

# 设备精度与白盒检查
python -m pytest -q test/mega_recurrent_kda/test_mega_recurrent_kda.py -m npu
```

测试文件内联独立 CPU golden，校验最终输出、内部 workspace、完整 Conv/recurrent 状态池、不可变空槽和重复调用，并覆盖 TP4/TP8 的短 batch 与尾块。数值比较使用 `atol=rtol=0.005`，不可变槽位使用零容差。当前不提供逐位确定性保证。

## 性能

以下为 2026-10-17 在 dav-3510 上的代表性测量记录，对应算子文件 SHA256 为 `e842d24ff76498763de1b69966ae45952e9a96b2e4523469e7b594a825e25e33`。完整测试范围为 TP4/8/16 × 模型 batch 1～32 × 8/16/24/32 AIC，共 384 组。

每组预热 10 次，采集 20 次样本；每次性能采样恢复状态池初态并清理 224 MiB L2，仅统计目标融合内核。表中为算术平均耗时，单位 us，未计状态恢复、清 L2 和 Profiler 导出开销。28 AIC 未纳入本轮性能表。未测小算子基线，不报告相对小算子加速比。

| TP | 模型 batch | 算子 B | 8 AIC avg us | 16 AIC avg us | 24 AIC avg us | 32 AIC avg us |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 4 | 297.406 | 183.969 | 183.724 | 137.848 |
| 4 | 8 | 32 | 1041.499 | 566.466 | 470.679 | 399.642 |
| 4 | 16 | 64 | 2014.204 | 1049.237 | 828.500 | 666.410 |
| 4 | 32 | 128 | 3963.621 | 2033.745 | 1528.974 | 1294.218 |
| 8 | 1 | 8 | 213.786 | 139.306 | 132.772 | 108.397 |
| 8 | 8 | 64 | 1036.876 | 564.738 | 436.069 | 410.589 |
| 8 | 16 | 128 | 2008.373 | 1062.068 | 800.319 | 646.212 |
| 8 | 32 | 256 | 3985.566 | 2056.201 | 1492.666 | 1261.817 |
| 16 | 1 | 16 | 210.033 | 161.007 | 148.721 | 73.787 |
| 16 | 8 | 128 | 1042.872 | 571.432 | 442.226 | 318.546 |
| 16 | 16 | 256 | 2055.269 | 1069.722 | 811.641 | 594.935 |
| 16 | 32 | 512 | 4096.800 | 2109.517 | 1555.538 | 1164.725 |

性能值对应上述源码和测量条件。当前源码的 TP4/TP8 主投影在 32 AIC、模型 batch >= 8 且总行数 `B*S` 能被 128 整除时使用 M128/N384 分块；该分块配置尚无本环境下对应的性能数据，上表不覆盖此配置。
