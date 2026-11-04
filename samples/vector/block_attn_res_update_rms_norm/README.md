# block_attn_res_update_rms_norm

## 产品支持情况

- Ascend 950：支持。

## 功能说明

`block_attn_res_update_rms_norm` 基于 CANNBot-DSL 实现 Block Attention Residual Update 与 RMSNorm 的融合计算，实现位于 [block_attn_res_update_rms_norm.py](block_attn_res_update_rms_norm.py)。

算子先将本层增量 `delta` 加到当前残差块 `partial_block`，再根据 `pseudo_query` 计算当前块的分数，与已有块的 online-softmax 累积结果合并，得到 `h`。最后对 `h` 沿隐藏维做 RMSNorm，并乘以缩放权重 `gamma`，返回 `y`。

只有 `partial_block` 原地更新；`numerator`、`logit_max`、`exp_sum` 等其他输入均不修改。中间结果 `h` 保留在片上，不单独写回显存。

### 计算过程

以下计算对每个 token 独立进行，`sum` 和 `mean` 均沿隐藏维 `D` 归约：

```text
p = partial_block + float32(delta)
score = sum(p * pseudo_query) / sqrt(mean(p * p) + score_eps)
m = max(logit_max, score)
a = exp(logit_max - m)
b = exp(score - m)
inv_denom = 1 / (exp_sum * a + b)
alpha = a * inv_denom
beta = b * inv_denom
h = bfloat16(numerator * alpha + p * beta)
rstd = 1 / sqrt(mean(float32(h) ** 2) + norm_eps)
y = bfloat16(float32(h) * rstd * float32(gamma))
```

上述公式描述数学计算过程，除显式的 BF16 转换外，中间计算使用 FP32。`partial_block` 更新为 `p`。RMSNorm 使用的是已经舍入到 BF16 的 `h`，不是舍入前的 FP32 加权和。

`numerator` 是已有块的加权累积分子，不是归一化后的平均值；它与 `logit_max`、`exp_sum` 共同表示已有块的 online-softmax 状态。`score_eps` 用于计算块分数时的 RMS 归一化，`norm_eps` 用于输出 `h` 的 RMSNorm，两者相互独立。

## 函数原型

```python
def block_attn_res_update_rms_norm(
    partial_block: torch.Tensor,
    delta: torch.Tensor,
    pseudo_query: torch.Tensor,
    numerator: torch.Tensor,
    logit_max: torch.Tensor,
    exp_sum: torch.Tensor,
    gamma: torch.Tensor,
    score_eps: float = 1e-6,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    ...
```

## 参数说明

`T` 为 token 数，`D` 为隐藏维大小，当前固定为 `7168`。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `partial_block` | Tensor | 必选 | 当前残差块；原地加上 `delta`，以 FP32 保存更新结果。 | float32 | `[T, D]` |
| `delta` | Tensor | 必选 | 本层向当前残差块累加的增量。 | bfloat16 | `[T, D]` |
| `pseudo_query` | Tensor | 必选 | 计算当前块分数的查询向量，所有 token 共用。 | float32 | `[D]` |
| `numerator` | Tensor | 必选 | 已有块以 `logit_max` 为基准累积的加权分子，只读。 | float32 | `[T, D]` |
| `logit_max` | Tensor | 必选 | 每个 token 已有块分数的最大值，只读。 | float32 | `[T]` |
| `exp_sum` | Tensor | 必选 | 每个 token 已有块以 `logit_max` 为基准累积的指数和，只读。 | float32 | `[T]` |
| `gamma` | Tensor | 必选 | 输出 RMSNorm 的缩放权重，由调用方传入，例如网络中对应 RMSNorm 层的可学习权重。 | bfloat16 | `[D]` |
| `score_eps` | float | 可选 | 计算块分数时的数值稳定项，默认 `1e-6`。 | - | - |
| `norm_eps` | float | 可选 | 输出 RMSNorm 的数值稳定项，默认 `1e-6`。 | - | - |

## 返回值说明

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|
| `y` | Tensor | RMSNorm 输出，位于输入所在设备。 | bfloat16 | `[T, D]` |

函数仅返回 `y`，不返回 `h` 或 `rstd`。更新后的残差块通过原输入 `partial_block` 获取。

## 约束说明

- `T >= 0`，`D = 7168`；`T` 不限于精度测试列出的取值。
- 所有输入张量必须连续、位于同一 NPU，并满足参数表中的 shape 和 dtype。
- `delta` 和 `gamma` 仅支持 bfloat16，不会自动转换 FP16/FP32 输入。
- `score_eps` 和 `norm_eps` 必须为有限正数。
- `T = 0` 时返回 `[0, 7168]` 的 BF16 空张量，不启动设备内核。
- 调用方应避免输入张量之间的存储别名；重复调用时，`partial_block` 会在上一次更新的基础上继续累加。

## 确定性计算

当前样例未提供逐位确定性保证，现有精度测试按数值容差校验。

## 调用示例

先按照[仓库 README](../../../README.md)安装依赖并加载 CANN 环境。当前主线使用 OpKit 导出接口，还需安装与 CANNBot-DSL 配套的 OpKit。在仓库根目录启动 Python 后执行以下代码，调用样例源码中的接口：

```python
import torch
import torch_npu

from samples.vector.block_attn_res_update_rms_norm.block_attn_res_update_rms_norm import (
    block_attn_res_update_rms_norm,
)

T, D = 32, 7168
device = "npu:0"
partial_block = torch.randn(T, D, device=device, dtype=torch.float32)
delta = torch.randn(T, D, device=device, dtype=torch.bfloat16)
pseudo_query = torch.randn(D, device=device, dtype=torch.float32)
numerator = torch.randn(T, D, device=device, dtype=torch.float32)
logit_max = torch.zeros(T, device=device, dtype=torch.float32)
exp_sum = torch.ones(T, device=device, dtype=torch.float32)
gamma = torch.ones(D, device=device, dtype=torch.bfloat16)

y = block_attn_res_update_rms_norm(
    partial_block, delta, pseudo_query, numerator,
    logit_max, exp_sum, gamma,
    score_eps=1e-6, norm_eps=1e-6,
)
torch.npu.synchronize()
# y 是 RMSNorm 输出，partial_block 已原地更新。
```

## 精度测试

在仓库根目录执行[测试脚本](../../../test/vector/block_attn_res_update_rms_norm/test_block_attn_res_update_rms_norm.py)：

```bash
DEVICE_ID=0 python3 -m pytest \
    test/vector/block_attn_res_update_rms_norm/test_block_attn_res_update_rms_norm.py -v -s
```

测试共 10 个用例：7 个网络 shape（`D = 7168`，`T = 1, 2, 4, 8, 32, 254, 512`），以及相消、score/ExpSub FTZ、倒数 FTZ 三个数值回归用例。另在网络用例内检查空输入、动态 T 和非法 D/dtype 的拒绝行为。

测试使用内置的 Torch / NumPy CPU golden。Update 部分对齐 AscendC 配套 `spec.py` 的计算顺序，随后对舍入到 BF16 的 `h` 做 FP32 RMSNorm；同时校验更新后的 `partial_block` 和最终输出 `y`。

精度标准为 `np.isclose(rtol=0.001, atol=0)`，不满足条件的元素比例不超过 `0.001`，非有限值判为失败；另外检查输出 shape/dtype/device 和只读输入不变。

## 性能

以下为 2026-09-20 的上板测量结果：

- 环境：Ascend 950（64 个 Vector 核），CANN 9.2.0，CANNBot-DSL，PyTorch / torch_npu 2.10。
- 执行方式：使用本实现导出的 native 二进制，固定 `D = 7168`、BF16 `delta/gamma`、两个 epsilon 均为 `1e-6`。
- 每个 shape 测量 3 轮，每轮预热 50 次、采样 200 次，共 600 个有效样本。复用输入缓冲区，不清空 L2；每次调用前恢复 `partial_block`。
- 指标为 msprof 采集的设备端内核 `TaskDuration`，单位为微秒；不包含预热、输入恢复、精度校验或 Host 调用开销，不代表端到端延迟。

| T | p10（μs） | p50（μs） | p90（μs） |
|---:|---:|---:|---:|
| 1 | 3.037 | 3.129 | 3.194 |
| 2 | 3.179 | 3.223 | 3.294 |
| 4 | 3.702 | 3.767 | 3.848 |
| 8 | 3.783 | 3.859 | 3.961 |
| 32 | 4.151 | 4.431 | 4.690 |
| 64 | 4.932 | 5.303 | 5.763 |
| 254 | 8.951 | 9.328 | 9.812 |
| 512 | 14.306 | 14.712 | 15.269 |
