# flash_kda_cp_preprocess

## 产品支持情况

- Ascend 950：支持。

## 功能说明

`flash_kda_cp_preprocess` 基于 CANNBot-DSL 实现 Kimi Delta Attention 的 CP prefill 前半段，在一个 AICore kernel 中完成块内矩阵计算和 rank 内递归摘要，返回八个张量，供调用方执行跨 rank prefix scan，再交给 `flash_kda_cp_main`。实现位于 [`flash_kda_cp_preprocess.py`](flash_kda_cp_preprocess.py)。

调用时先生成调度 metadata，再执行计算：

1. [`flash_kda_cp_preprocess_metadata`](flash_kda_cp_preprocess_metadata.py)：根据输入形状、布局和当前 AICore 配额生成调度信息。
2. `flash_kda_cp_preprocess`：消费 metadata，启动 AICore 算子，执行块内计算与 rank 内状态摘要。

### 计算过程

先对输入进行归一化与门控激活：

$$
Q=\operatorname{L2Norm}(Q_{\mathrm{raw}}),\qquad
K=\operatorname{L2Norm}(K_{\mathrm{raw}}),
$$

$$
g=\mathrm{lower\_bound}\cdot
\operatorname{sigmoid}\left(\exp(\mathrm{A\_log})\cdot(g_{\mathrm{raw}}+\mathrm{dt\_bias})\right),
\qquad
\beta=\operatorname{sigmoid}(\beta_{\mathrm{raw}}).
$$

下标 raw 表示接口传入的原始张量。`L2Norm` 沿最后一维归一化，epsilon 为 `1e-6`；后续公式中的 $Q$、$K$、$g$、$\beta$ 均为上述处理后的结果。

令块内累积对数门控 $G_t=\sum_{i\le t}g_i$，衰减 $\gamma_t=\exp(G_t)$（$i,t$ 为块内 token 下标，逐 Key 通道），并定义：

$$
\tilde K=\gamma\odot K,\qquad
\bar K=\gamma^{-1}\odot K,\qquad
\tilde Q=\operatorname{scale}\cdot(\gamma\odot Q),
\qquad
K^r=e^{G_{\mathrm{end}}-G}\odot K,
$$

其中 $G_{\mathrm{end}}$ 为块末累积对数门控。序列按每块 64 个 token 切分，每个 chunk 内先计算严格下三角矩阵及其逆：

$$
T=\operatorname{stril}\!\left(\operatorname{diag}(\beta)\tilde K\bar K^{\top}\right),
\qquad A^{-1}=(I+T)^{-1},
$$

$$
U=A^{-1}\operatorname{diag}(\beta)V,
\qquad W=A^{-1}\operatorname{diag}(\beta)\tilde K,
\qquad M_{qk}=\operatorname{tril}\!\left(\tilde Q\bar K^{\top}\right).
$$

输出张量与上式的对应关系：`qg` $=\tilde Q$、`kg` $=K^r$、`u` $=U$、`w` $=-W$（已含负号，供 `flash_kda_cp_main` 直接相加）、`mqk` $=M_{qk}$（含对角下三角）、`decay` $_c=e^{G_{\mathrm{end}}}$（块末逐通道衰减）。

第二阶段将 rank 内全部有效 chunk 合成为仿射状态摘要。状态按 `[Value 通道, Key 通道]` 存储，与 `flash_kda_cp_main` 一致。令 $H_c$ 为第 $c$ 块后的状态，$L$ 为当前逻辑序列实际的有效 chunk 数；递推为 $H_c=H_{c-1}P_c+b_c$，其中：

$$
P_c=\operatorname{diag}(\mathrm{decay}_c)-W_c^{\top}K_c^r,
\qquad
b_c=U_c^{\top}K_c^r,
$$

则返回值：

$$
\mathrm{transition}=P_1 P_2\cdots P_L,
\qquad
\mathrm{state\_ext}=\sum_{c=1}^{L}b_c\Big(\prod_{j=c+1}^{L}P_j\Big).
$$

乘积按 chunk 顺序从左到右排列，空乘积为单位矩阵。本 rank 的摘要表示 $H_{\mathrm{out}}=H_{\mathrm{in}}\,\mathrm{transition}+\mathrm{state\_ext}$。调用方对各 rank 的摘要执行外部 CP exclusive prefix scan，得到当前 rank 之前的合成摘要 $(\Phi_{<r},\Psi_{<r})$，再由全局初始状态计算 `local_initial_state` $=H_{\mathrm{global}}\Phi_{<r}+\Psi_{<r}$。首个 rank 的前缀为单位变换和零加项；将该初始状态连同预处理结果交给 `flash_kda_cp_main`。

## 函数原型

先调用同目录 metadata 脚本的 `flash_kda_cp_preprocess_metadata(q, v, layout_qkv, cu_seqlens)`，再把结果作为 `metadata` 传给主接口。

```python
flash_kda_cp_preprocess.flash_kda_cp_preprocess.flash_kda_cp_preprocess(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    scale: float, A_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: float,
    layout_qkv: str, metadata: torch.Tensor, cu_seqlens: torch.Tensor | None = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

模块路径为 `flash_kda_cp_preprocess.flash_kda_cp_preprocess`。

## 参数说明

`B` 为逻辑序列数，`Nqk` 为 Query/Key 头数，`Nv` 为 Value 头数，`S` 为 BNSD/BSND 每条序列长度，`T` 为 TND 总 token 数，`D = 128` 为统一头宽，`M` 为 metadata 算子返回的容量。chunk 容量 `C` 在 BNSD/BSND 下为 `ceil(S / 64) + B - 1`，在 TND 下为 `ceil(T / 64) + B - 1`；metadata 按每条逻辑序列实际的 `ceil(length_i / 64)` 调度。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `q` | Tensor | 必选 | 原始 Query；内核按行归一化。 | bfloat16 | BNSD：`[B, Nqk, S, D]`；其他布局见下表 |
| `k` | Tensor | 必选 | 原始 Key；shape 和布局与 `q` 相同，内核按行归一化。 | bfloat16 | 与 `q` 相同 |
| `v` | Tensor | 必选 | Value。 | bfloat16 | BNSD：`[B, Nv, S, D]`；其他布局见下表 |
| `g` | Tensor | 必选 | 原始逐通道门控输入，激活公式见计算过程。 | bfloat16 | 与 `v` 相同 |
| `beta` | Tensor | 必选 | 原始逐 token 写入 logits；内核执行 sigmoid。 | bfloat16 | BNSD：`[B, Nv, S]`；其他布局见下表 |
| `scale` | float | 必选 | Query-key 缩放系数，常用 `D ** -0.5`。 | float32 | - |
| `A_log` | Tensor | 必选 | 每个 Value 头的 log 时间尺度。 | float32 | `[Nv]` |
| `dt_bias` | Tensor | 必选 | 每个 Value 头、每个 Key 通道的门控偏置。 | float32 | `[Nv, D]` |
| `lower_bound` | float | 必选 | 门控下界，取值范围为 `[-5, 0]`。 | float32 | - |
| `layout_qkv` | str | 必选 | 输入布局，取值为 `"BNSD"`、`"BSND"` 或 `"TND"`。 | - | - |
| `metadata` | Tensor | 必选 | 配套 metadata 算子返回的调度张量。 | int32 | `[M]`，容量 `M` 由 metadata 算子确定 |
| `cu_seqlens` | Tensor | TND 必选，其他布局不传 | TND 每个逻辑序列的 token 边界，首项为 0，末项为 `T`。 | int32 | `[B+1]` |

各布局对应的输入 shape：

| `layout_qkv` | `q`、`k` | `v`、`g` | `beta` |
|:---|:---|:---|:---|
| `BNSD` | `[B, Nqk, S, D]` | `[B, Nv, S, D]` | `[B, Nv, S]` |
| `BSND` | `[B, S, Nqk, D]` | `[B, S, Nv, D]` | `[B, S, Nv]` |
| `TND` | `[T, Nqk, D]` | `[T, Nv, D]` | `[T, Nv]` |

### 递推精度

rank 内摘要始终使用 FP32 状态和 FP32 MMAD：`H_c=H_{c-1}P_c+b_c` 的
`transition`、`state_ext` 以及逐 chunk 的局部矩阵乘均以 FP32 累积。输入和公共中间
张量仍按接口使用 BF16；实现不再提供 BF16 递推分支，也不再分配 BF16 staging 缓存。

### metadata 入参

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `q` | Tensor | 必选 | 与主算子相同的 Query；仅读取 shape、布局和设备信息。 | bfloat16 | 与主算子 `q` 相同 |
| `v` | Tensor | 必选 | 与主算子相同的 Value；仅读取 shape、布局和设备信息。 | bfloat16 | 与主算子 `v` 相同 |
| `layout_qkv` | str | 必选 | 与主算子一致的布局。 | - | - |
| `cu_seqlens` | Tensor | TND 必选 | 与主算子一致的 TND 序列边界。 | int32 | `[B+1]` |

metadata 算子只读取 `q/v` 的属性，不读取它们的数值；TND 下会读取 `cu_seqlens` 的边界值来划分逻辑序列。

### metadata 复用

`flash_kda_cp_preprocess` 不生成 metadata。当 `B`、`Nv`、稠密长度 `S`、TND 总长度 `T` 或 `cu_seqlens` 边界、布局、当前 stream 的有效 AICore 配额或内置调度 cost 公式版本变化时，必须重新生成 metadata；`q/v` 的数值不影响调度。整网可在 forward 开始时生成一次 metadata，并在调度配置相同的层间复用。

metadata 记录每条逻辑序列实际消费的 chunk 数；中间量的统一容量可用同一模块导出的 `chunk_capacity_upper_bound` 计算：BNSD/BSND 传入 `(B, S)`，TND 传入 `(B, T)`，结果分别为上述布局对应的 `C`。

`flash_kda_cp_preprocess_metadata` 与 `flash_kda_cp_main_metadata` 共享同一个 FlashKDA stage12w metadata ABI 和 AICPU 编译对象；两者返回的 metadata 可直接分别供对应主算子消费。

## 返回值说明

返回顺序为 `(transition, state_ext, decay, qg, kg, u, w, mqk)`。`qg`、`kg`、`u`、`w` 对 BNSD/BSND 输入统一使用 BNSD 存储；TND 输入输出 `[T, ...]`。chunk 级量按统一容量分配，均位于输入所在设备。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `transition` | Tensor | 必选 | rank 局部状态仿射变换，见计算过程。 | float32 | `[B, Nv, D, D]` |
| `state_ext` | Tensor | 必选 | rank 局部状态仿射加项，见计算过程。 | float32 | `[B, Nv, D, D]` |
| `decay` | Tensor | 必选 | 每块末端逐通道衰减；`C` 是上述统一容量，metadata 中每条序列只消费自己的有效 chunk。 | float32 | `[B, Nv, C, D]` |
| `qg` | Tensor | 必选 | 衰减并缩放后的 Query。 | bfloat16 | BNSD/BSND 输入：`[B, Nv, S, D]`；TND 输入：`[T, Nv, D]` |
| `kg` | Tensor | 必选 | 相对块末衰减恢复的 Key。 | bfloat16 | BNSD/BSND 输入：`[B, Nv, S, D]`；TND 输入：`[T, Nv, D]` |
| `u` | Tensor | 必选 | 块内 Value 贡献。 | bfloat16 | BNSD/BSND 输入：`[B, Nv, S, D]`；TND 输入：`[T, Nv, D]` |
| `w` | Tensor | 必选 | 块内 Key 贡献，已含负号。 | bfloat16 | BNSD/BSND 输入：`[B, Nv, S, D]`；TND 输入：`[T, Nv, D]` |
| `mqk` | Tensor | 必选 | 块内含对角下三角 QK 矩阵；尾部容量槽保持未消费。 | bfloat16 | `[B, Nv, C, 64, 64]` |

## 约束说明

- 支持 BNSD、BSND 定长输入和多序列 TND 输入，`S/T > 0`，每个序列尾块可不足 64；chunk 中间量按上述布局对应的 `C` 申请。
- TND 的 `cu_seqlens` 形状为 `[B+1]`，边界单调递增，首项为 0、末项为 `T`；metadata、Stage1 和 summary 在一次下发中按逻辑序列调度。
- 当前接口用于 CP prefill 前半段计算，要求 `D = 128`，`B >= 1`。
- 支持 GQA，要求 `1 <= Nqk <= Nv`、`Nv % Nqk == 0`；metadata 调度要求单轮任务数不超过 861。
- 所有张量位于同一 NPU，并满足参数表中的 dtype；`q/k/v/g` 最后一维步长为 1，`A_log`、`dt_bias`、`metadata` 和 `cu_seqlens` 必须连续。生成 metadata 时，`q/v` 也必须连续。
- 主算子与 metadata 使用同一个当前 stream 的有效核数；内核中的全栅栏同步依赖此约定。

## 调用示例

```python
import math
import torch
import torch_npu
from flash_kda_cp_preprocess.flash_kda_cp_preprocess import flash_kda_cp_preprocess
from flash_kda_cp_preprocess.flash_kda_cp_preprocess_metadata import flash_kda_cp_preprocess_metadata

B, Nqk, Nv, S, D = 1, 1, 1, 1024, 128
q = torch.randn(B, Nqk, S, D).bfloat16().npu()
k = torch.randn_like(q)
v = torch.randn(B, Nv, S, D).bfloat16().npu()
g = torch.randn_like(v)
beta = torch.randn(B, Nv, S).bfloat16().npu()
A_log = torch.zeros(Nv, dtype=torch.float32).npu()
dt_bias = torch.zeros(Nv, D, dtype=torch.float32).npu()

metadata = flash_kda_cp_preprocess_metadata(q, v, "BNSD")
prepared = flash_kda_cp_preprocess(q, k, v, g, beta, 1 / math.sqrt(D),
    A_log, dt_bias, -5.0, "BNSD", metadata)
```

`prepared[0]` 和 `prepared[1]` 供调用方进行跨 rank prefix scan；其余结果传给 `flash_kda_cp_main`。

## 精度测试

在仓库根目录执行单文件测试。测试文件内置独立 FP32 分块递推 golden，
对八个输出逐张量按 `atol=0.005, rtol=0.005` 比对，并检查 99.9 分位归一化误差。
精度测试保留 BNSD、BSND、TND 三个白盒场景和一个最小全量场景；另覆盖最小
metadata 调度容量：

```bash
python -m pytest -q test/flash_kda_cp_preprocess/test_flash_kda_cp_preprocess.py
```

## 性能

以下结果于 2026-10-13 在 Ascend 950、CANN 9.2.0 上用 `msprof --task-time=l1` 采集，形状为 `B=1, Nqk=32, Nv=32, S=8192, D=128`，覆盖 BNSD、BSND、TND 三种布局；rank 内摘要使用 FP32 递推。

metadata 在采样循环前生成一次，`Task Duration` 只统计 AICore kernel `flash_kda_cp_preprocess_kernel`。每种布局预热 2 次后连续采样 12 次，表中为 12 次采样的中位数。

| Layout | Task Duration 中位数 (us) |
|:---|--:|
| BNSD | 2842.1 |
| BSND | 2960.0 |
| TND | 2949.8 |
