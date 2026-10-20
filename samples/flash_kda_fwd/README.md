# FlashKDA Fwd

基于 CANNBot-DSL 的 Kimi Delta Attention 训练前向算子，面向 Ascend 950。公开入口位于 [`flash_kda_fwd.py`](flash_kda_fwd.py)，配套 AICPU 调度位于 [`flash_kda_fwd_metadata.py`](flash_kda_fwd_metadata.py)。先生成 metadata，再计算序列输出、状态以及反向所需的缓存。

## 功能说明

计算流程与 [FlashKDA](../flash_kda/README.md) 一致：按需融合 Q/K 行级 L2 normalize、gate 激活和 beta sigmoid，将序列按 64 个 token 切分，在 chunk 内构造因果 QK 矩阵及下三角逆矩阵，再跨 chunk 递推 FP32 状态。训练接口额外返回可配置的反向缓存。

以下 $Q,K$ 表示按开关完成归一化后的输入，$g,\beta$ 表示完成对应激活后的值。每个 chunk 独立累加对数衰减，令

$$
G_t=\sum_{i=0}^{t}g_i,\quad \gamma_t=\exp(G_t),\quad
\tilde K=\gamma\odot K,\quad \bar K=\gamma^{-1}\odot K,\quad
\tilde Q=\operatorname{scale}\cdot(\gamma\odot Q).
$$

与推理版本一致，块内求解为

$$
T_c=\operatorname{stril}\!\left(\operatorname{diag}(\beta)\tilde K\bar K^\top\right),\quad
A_c=(I+T_c)^{-1},\quad
U=A_c\operatorname{diag}(\beta)V,\quad
W=A_c\operatorname{diag}(\beta)\tilde K,
$$

$$
M_{qk}=\operatorname{tril}(\tilde Q\bar K^\top),\quad
K_{\mathrm{restored},t}=\exp(G_{\mathrm{last}}-G_t)\odot K_t.
$$

$\operatorname{stril}$ 取严格下三角，$\operatorname{tril}$ 保留下三角及对角。令 chunk 入口状态 $H_c$ 的维度为 $D_v\times D_k$，与公开状态张量一致，$H_0=$ `initial_state`，则

$$
V_{\mathrm{new}}=U-WH_c^\top,\qquad
O_c=\tilde QH_c^\top+M_{qk}V_{\mathrm{new}},
$$

$$
H_{c+1}=H_c\operatorname{diag}(\gamma_{\mathrm{last}})
       +V_{\mathrm{new}}^\top K_{\mathrm{restored}}.
$$

训练缓存 `Aqk`、`Akk`、`w`、`u`、`v_new` 分别对应 $M_{qk}$、$A_c$、$W$、$U$、$V_{\mathrm{new}}$；`qg` 对应未乘 scale 的 $\gamma\odot Q$，`kg` 对应 $K_{\mathrm{restored}}$，`g_cumsum` 对应 $G$，`h` 保存各 chunk 的入口状态 $H_c$。缓存是否返回由接口开关决定，存储类型见返回值说明；状态使用 FP32 递推。

- [`flash_kda_fwd_staged.py`](flash_kda_fwd_staged.py)：按 chunk group 分阶段调度，通过 workspace 交换阶段结果。
- [`flash_kda_fwd_persistent.py`](flash_kda_fwd_persistent.py)：一个核连续处理序列头的 chunk，保持递推状态驻留。

metadata 默认 `template="auto"`；`template="staged"` 和 `template="persistent"` 可分别固定使用对应模板。两个模板共用公开接口和缓存布局。

persistent 使用连续 ND 状态合并、recurrent L0A/L0B 双缓冲和已加载状态半块复用；状态仍以 FP32 递推，完整保存训练缓存。

## 快速开始

在仓库根目录设置 `export PYTHONPATH="$PWD/samples:$PYTHONPATH"` 后运行：

```python
import math
import torch
import torch_npu

from flash_kda_fwd.flash_kda_fwd import flash_kda_fwd
from flash_kda_fwd.flash_kda_fwd_metadata import flash_kda_fwd_metadata

B, Nqk, Nv, S, D = 1, 3, 3, 8192, 128
q = torch.randn(B, Nqk, S, D, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(B, Nv, S, D, device="npu", dtype=torch.bfloat16)
g = torch.randn_like(v)
beta = torch.randn(B, Nv, S, device="npu", dtype=torch.bfloat16)
initial_state = torch.zeros(B, Nv, D, D, device="npu", dtype=torch.float32)
A_log = torch.zeros(Nv, device="npu", dtype=torch.float32)
dt_bias = torch.zeros(Nv, D, device="npu", dtype=torch.float32)

metadata = flash_kda_fwd_metadata(q, v, initial_state, "BNSD")
results = flash_kda_fwd(
    q, k, v, g, beta, 1 / math.sqrt(D), initial_state,
    A_log, dt_bias, -5.0, "BNSD", metadata,
    disable_recompute=True, return_intermediate_state=True, output_final_state=True,
)
(out, final_state, q_norm, k_norm, q_rstd, k_rstd, beta_activated, Aqk,
 Akk, w, u, qg, kg, v_new, g_cumsum, h) = results
```

## 参数说明

`B` 为逻辑序列数，`Nqk`、`Nv` 为 Q/K、Value 头数，`S` 为存储序列长度，`T` 为 packed token 总数，`Dk=Dv=D=128`。

| 参数 | BNSD shape / 取值 | 类型 | 说明 |
|:---|:---|:---|:---|
| `q`, `k` | `[B,Nqk,S,D]` | BF16 | Query/Key；可在内核归一化，也可由调用方预处理。 |
| `v` | `[B,Nv,S,D]` | BF16 | Value。 |
| `g` | 与 `v` 相同 | BF16 / FP32 | 原始 gate 或预处理的对数衰减；FP32 输入会转换为 BF16。 |
| `beta` | `[B,Nv,S]` | BF16 / FP32 | 原始 logits 或预处理 beta；FP32 输入会转换为 BF16。 |
| `scale` | 默认 `D ** -0.5` | float / None | 传 None 时按头维度计算缩放。 |
| `initial_state` | `[B,Nv,Dv,Dk]` | FP32 | 必传；无历史状态时传全零张量。 |
| `A_log` | `[Nv]` | FP32 / None | 内核激活 gate 时必传；外部激活时可为 None。 |
| `dt_bias` | `[Nv,Dk]` | FP32 / None | gate 偏置；None 按零处理。 |
| `lower_bound` | `[-5,0]` 或 None | float / None | 内核激活 gate 时，数值表示有界 sigmoid 下界；None 使用 softplus 门控。 |
| `layout_qkv` | `"BNSD"` / `"BSND"` / `"TND"` | str | 应显式指定。 |
| `metadata` | `[M]` | int32 | 必传；由配套 metadata 接口生成。 |
| `use_qk_l2norm_in_kernel` | 默认 True | bool | False 时使用已归一化的 Q/K，不返回 norm 缓存。 |
| `use_gate_in_kernel` | 默认 True | bool | False 时 g 已激活，忽略 `A_log`、`dt_bias`。 |
| `use_beta_sigmoid_in_kernel` | 默认 True | bool | False 时直接使用 beta。 |
| `disable_recompute` | 默认 True | bool | 返回 W/U/qg/kg/v_new/g_cumsum 及满足条件的 norm、beta 缓存。 |
| `return_intermediate_state` | 默认 False | bool | 控制是否返回各 chunk 入口状态 h。 |
| `allow_neg_eigval` | 默认 False | bool | True 时 sigmoid beta 乘 2；要求开启 beta sigmoid。 |
| `output_final_state` | 默认 False | bool | 控制是否返回最终状态。 |

BSND 的 `q/k` 为 `[B,S,Nqk,D]`，`v/g` 为 `[B,S,Nv,D]`，`beta` 为 `[B,S,Nv]`。TND 分别为 `[T,Nqk,D]`、`[T,Nv,D]`、`[T,Nv]`。

metadata 接口为 `flash_kda_fwd_metadata(q, v, initial_state, layout_qkv, cu_seqlens=None, *, template="auto")`。TND 必须传连续 int32 `[B+1]` 前缀和 `cu_seqlens`；BNSD/BSND 可传它指定每条有效长度，不传时均为 S。metadata 只处理形状、边界和调度，不读取 Q/V/state 数值；同版本、同形状、同布局、同有效长度和同设备有效核数时可复用。

## 返回值说明

返回 `(out, final_state, q_norm, k_norm, q_rstd, k_rstd, beta_activated, Aqk, Akk, w, u, qg, kg, v_new, g_cumsum, h)`。可选返回项由开关决定；非 None 的张量均位于输入所在设备。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `out` | Tensor | 必选 | 序列输出，布局与 `v` 相同；metadata 标记的有效序列范围之外，输出值未定义。 | bfloat16 | 与 `v` 相同 |
| `final_state` | Tensor 或 None | 可选 | `output_final_state=True` 时返回每条序列的最终递归状态；不原地修改 `initial_state`。 | float32 | `[B, Nv, Dv, Dk]` |
| `q_norm`、`k_norm` | Tensor 或 None | 可选 | 归一化后的 Query/Key；需同时开启 qk L2Norm 和 `disable_recompute`。BSND 输入也按 BNSD 返回，TND 保持 TND。 | bfloat16 | BNSD：`[B, Nqk, S, D]`；TND：`[T, Nqk, D]` |
| `q_rstd`、`k_rstd` | Tensor 或 None | 可选 | Query/Key 的逆范数；需同时开启 qk L2Norm 和 `disable_recompute`。 | float32 | 对应 q/k 去掉最后一维 |
| `beta_activated` | Tensor 或 None | 可选 | sigmoid 后的 beta；需同时开启 beta 激活和 `disable_recompute`。BSND 输入按 BNSD 返回，TND 保持 TND。 | float32 | BNSD：`[B, Nv, S]`；TND：`[T, Nv]` |
| `Aqk` | Tensor | 必选 | 带 decay 和 scale 的 chunk-major 因果 QK 矩阵。 | bfloat16 | BNSD/BSND：`[B, Nv, num_chunks, 64, 64]`，`num_chunks=ceil(S/64)`；TND：`[num_chunks, Nv, 64, 64]`，`num_chunks=ceil(T/64)+B` |
| `Akk` | Tensor | 必选 | chunk-major 的 $(I+T)^{-1}$。 | bfloat16 | 同 `Aqk` |
| `w`、`u` | Tensor 或 None | 可选 | 状态修正权重和修正前 value；`disable_recompute=True` 时返回。BSND 输入按 BNSD 返回，TND 保持 TND。 | bfloat16 | BNSD/BSND：`[B, Nv, S, D]`；TND：`[T, Nv, D]` |
| `qg`、`kg` | Tensor 或 None | 可选 | 带累计衰减的 Q 和从 chunk 末端恢复的 K；`disable_recompute=True` 时返回。 | bfloat16 | 同 `w`、`u` |
| `v_new` | Tensor 或 None | 可选 | 扣除入口状态预测后的 value；`disable_recompute=True` 时返回。 | bfloat16 | 同 `w`、`u` |
| `g_cumsum` | Tensor 或 None | 可选 | 按真实序列长度保存累计对数衰减；仅 `disable_recompute=True` 时返回，公共 dtype 保持 FP32。 | float32 | BNSD/BSND：`[B, Nv, S, Dk]`；TND：`[T, Nv, Dk]` |
| `h` | Tensor 或 None | 可选 | 各 chunk 入口状态；仅 `return_intermediate_state=True` 时返回。 | float32 | BNSD/BSND：`[B, Nv, num_chunks, Dv, Dk]`，`num_chunks=ceil(S/64)`；TND：`[num_chunks, Nv, Dv, Dk]`，`num_chunks=ceil(T/64)+B` |

状态递推、`initial_state`、`final_state` 和保存的 `h` 均使用 FP32。

内部输出存储按 64 个 token 对齐，返回时裁剪为逻辑 shape；非对齐场景下 `out` 可能是不连续的视图。矩阵缓存按逻辑 chunk 保存，有效因果上三角和尾块未使用列为 0，`Akk` 有效对角为 1。

## 约束说明

- 支持 Ascend 950，`Dk=Dv=128`，`B>=1`，序列长度大于 0；支持 BNSD、BSND、TND 和非对齐尾块。
- 支持 GQA，要求 `1 <= Nqk <= Nv <= 96` 且 `Nv % Nqk == 0`。
- 公开输入必须连续且位于同一设备；状态、`A_log`、`dt_bias` 使用 FP32，Q/K/V 使用 BF16。
- TND 前缀和首项为 0、末项为 T，严格递增；BNSD/BSND 的有效长度必须在 `[1,S]` 内。
- metadata 必须来自配套训练模块；不能复用推理模块或其他 ABI 版本的 metadata。
- TND 的 chunk 缓存按容量 `ceil(T/64)+B` 分配，仅每条序列实际使用的 chunk 有效。

## 精度测试

在仓库根目录运行，需要 Ascend NPU：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -q test/flash_kda_fwd/test_flash_kda_fwd.py
```

共 6 个精度用例，覆盖 BNSD、BSND、TND 下的 staged 和 persistent，以及 GQA、非对齐尾块和变长序列。输出、状态和反向缓存与内联 PyTorch FP32 golden、BF16/FP16 交接模拟比较，默认容差为 `atol=rtol=5e-3`。仅在推理与训练 metadata ABI 相同时，测试才额外执行推理结果交叉验证。

## 性能测试

2026-10-19 在 PR 服务器上实测：Ascend 950，`arena_old`，cannbotdsl 0.7.1、PyTorch 2.10.0、CANN 9.2.0，32 个 AIC。

`B=1`、`Nqk=Nv=96`、`S=16384`、`Dk=Dv=128`；Q/K/V/g/beta 为 BF16，初始状态及门控参数为 FP32，metadata 使用 `template="auto"`。`lower_bound=-5.0`，初态、`A_log` 和 `dt_bias` 均为 0。校验后预热 3 次，采样 20 次，不额外清理缓存。下表为设备 kernel Task Duration 的中位数，不包含 metadata 生成、编译和 host 分配时间。

训练前向开启 `disable_recompute=True`、`return_intermediate_state=True`、`output_final_state=True`，保存完整反向缓存；Q/K 归一化、gate 激活、beta sigmoid 均在内核完成，`allow_neg_eigval=False`。

| 布局 | Q/K/V shape | 实际模板 | 内核耗时（μs） |
|:---|:---|:---|---:|
| BNSD | `[1, 96, 16384, 128]` | persistent | 6059.995 |
| BSND | `[1, 16384, 96, 128]` | persistent | 6443.095 |
| TND | `[16384, 96, 128]` | persistent | 6437.970 |
