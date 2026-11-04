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

metadata 根据布局、有效长度和调度配置自动选择 staged/persistent 模板。两个模板共用公开接口和缓存布局。

## 快速开始

在仓库根目录设置 `export PYTHONPATH="$PWD/samples/attention:$PYTHONPATH"` 后运行：

```python
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

metadata = flash_kda_fwd_metadata(batch_size=B, nheads_qk=Nqk, nheads_v=Nv, seqlens=S, enable_grad=True, layout="BNSD")
results = flash_kda_fwd(
    q, k, v, g, beta, metadata, initial_state, A_log, dt_bias,
    layout="BNSD", output_final_state=True,
)
(out, final_state, chunk_states, Aqk, Akk, w, u, qg, kg, v_new,
 g_cumsum, q_norm, q_rstd, k_norm, k_rstd, beta_sigmoid) = results
```

## 参数说明

`B` 为逻辑序列数，`Nqk`、`Nv` 为 Q/K、Value 头数，`S` 为存储序列长度，`T` 为 packed token 总数，`Dk=Dv=D=128`。

| 参数 | BNSD shape / 取值 | 类型 | 说明 |
|:---|:---|:---|:---|
| `q`, `k` | `[B,Nqk,S,D]` | BF16 | Query/Key；可在内核归一化，也可由调用方预处理。 |
| `v` | `[B,Nv,S,D]` | BF16 | Value。 |
| `g` | 与 `v` 相同 | BF16 / FP32 | 原始 gate 或预处理的对数衰减；FP32 输入会转换为 BF16。 |
| `beta` | `[B,Nv,S]` | BF16 / FP32 | 原始 logits 或预处理 beta；核内激活使用 FP32 操作数。 |
| `metadata` | `[M]` | int32 | 必传；由配套 metadata 接口生成。 |
| `initial_state` | `[B,Nv,Dv,Dk]` | FP32 | 可为 None，此时使用全零状态；已传入的状态必须符合规格。 |
| `A_log` | `[Nv]` | FP32 / None | 内核激活 gate 时必传；外部激活时可为 None。 |
| `dt_bias` | `[Nv,Dk]` | FP32 / None | gate 偏置；None 按零处理。 |
| `cu_seqlens` | `[B+1]` | int32 / None | 有效长度前缀和；TND 必传，首项为 0、末项为 T。 |
| `seqused` | `[B]` | int32 / None | 每条序列的有效长度；同时提供 cu 时必须与相邻差值一致。 |
| `scale` | 默认 `D ** -0.5` | float / None | 传 None 时按头维度计算缩放。 |
| `output_final_state` | 默认 False | bool | 控制是否返回最终状态。 |
| `return_intermediate_states` | 默认 False | bool | 仅推理且 True 时返回各 chunk 入口状态。 |
| `use_qk_l2norm_in_kernel` | 默认 True | bool | True 执行 L2Norm；训练时返回 norm/rstd，与 disable_recompute 无关。 |
| `use_gate_in_kernel` | 默认 True | bool | False 时 g 已激活，忽略 `A_log`、`dt_bias`、`lower_bound`。 |
| `use_beta_sigmoid_in_kernel` | 默认 True | bool | True 执行 sigmoid；训练时返回 beta_sigmoid，与 disable_recompute 无关。 |
| `allow_neg_eigval` | 默认 False | bool | True 时 sigmoid beta 乘 2；要求开启 beta sigmoid。 |
| `disable_recompute` | 默认 True | bool | 仅训练且 True 时返回 w/u/qg/kg/v_new/g_cumsum。 |
| `lower_bound` | 默认 None，数值范围 `[-5,0)` | int / float / None | gate 开启时 None 使用 -exp(A_log) × softplus(g + dt_bias)，有限数值使用有界 sigmoid；关闭时忽略。 |
| `layout` | `"BNSD"` / `"BSND"` / `"TND"` | str | 默认 `TND`。 |

BSND 的 `q/k` 为 `[B,S,Nqk,D]`，`v/g` 为 `[B,S,Nv,D]`，`beta` 为 `[B,S,Nv]`。TND 分别为 `[T,Nqk,D]`、`[T,Nv,D]`、`[T,Nv]`。

metadata 接口按详设接收尺寸属性，不接收 q/v/initial_state：

```python
def flash_kda_fwd_metadata(
    cu_seqlens=None, seqused=None, *,
    batch_size, nheads_qk, nheads_v, seqlens,
    headdim_qk=128, headdim_v=128, enable_grad=False, layout="TND",
): ...
```

调用时必须显式提供 `seqlens`：BNSD/BSND 为存储长度 S，TND 为总长度 T。TND 必须传连续 INT32 `[B+1]` 的 `cu_seqlens`；padded 布局可用 cu 或 `seqused` 指定有效长度，不传时均为 S。长度张量和生成的 metadata 位于当前 NPU。`enable_grad=False` 用于推理，训练使用 True；切换模式或形状后重新生成 metadata。真实前向会阻塞回读 metadata 和给定的长度张量，校验本次调用的序列边界与调度；Meta/FakeTensor 只执行静态检查。尺寸、长度、连续性和 GQA 等非法参数抛出 `ValueError`。

主算子的第六个必填输入为 `metadata`，可按位置或关键字传入；从 `scale` 开始的属性按关键字传入；`scale` 的签名默认值为 `128 ** -0.5`，显式传 None 时仍使用该缩放值。

## 返回值说明

运行模式只由 `torch.is_inference_mode_enabled()` 判断；`torch.no_grad()` 不等同推理模式。每次返回固定 16 项，未生效位置为 None：

```python
(out, final_state, chunk_states, Aqk, Akk, w, u, qg, kg, v_new,
 g_cumsum, q_norm, q_rstd, k_norm, k_rstd, beta_sigmoid)
```

`q_rstd` 沿用详细设计的拼写，表示 Q 的逆范数。内部变量名为 q_rstd。

推理时融合计算仍按开关执行，Aqk/Akk 和所有训练中间变量为 None。内部计算仍需的状态及 scratch 不随公开返回关闭而删除。

令 padded 的 `J=ceil(S/64)`，TND 的容量 `J=ceil(T/64)+B`：

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `out` | Tensor | 必选 | 序列输出，布局与 `v` 相同；metadata 标记的有效序列范围之外，输出值未定义。 | bfloat16 | 与 `v` 相同 |
| `final_state` | Tensor 或 None | 可选 | `output_final_state=True` 时返回每条序列的最终递归状态，训练和推理均生效；不原地修改 `initial_state`。 | float32 | `[B,Nv,Dv,Dk]` |
| `chunk_states` | Tensor 或 None | 可选 | 各 chunk 入口状态；仅推理且 `return_intermediate_states=True` 时返回，训练时为 None。 | float32 | BNSD/BSND：`[B,J,Nv,Dv,Dk]`；TND：`[J,Nv,Dv,Dk]` |
| `Aqk` | Tensor 或 None | 可选 | 带 decay 和 scale 的 chunk-major 因果 QK 矩阵；训练时返回，不受 `disable_recompute` 控制，推理时为 None。 | bfloat16 | BNSD/BSND：`[B,J,Nv,64,64]`；TND：`[J,Nv,64,64]` |
| `Akk` | Tensor 或 None | 可选 | chunk-major 的 $(I+T)^{-1}$；训练时返回，不受 `disable_recompute` 控制，推理时为 None。 | bfloat16 | 同 `Aqk` |
| `w`、`u` | Tensor 或 None | 可选 | 状态修正权重和修正前 value；仅训练且 `disable_recompute=True` 时返回。 | bfloat16 | BSND：`[B,S,Nv,D]`；BNSD：`[B,Nv,S,D]`；TND：`[T,Nv,D]` |
| `qg`、`kg` | Tensor 或 None | 可选 | 带累计衰减的 Q 和从 chunk 末端恢复的 K；仅训练且 `disable_recompute=True` 时返回。 | bfloat16 | 同 `w`、`u` |
| `v_new` | Tensor 或 None | 可选 | 扣除入口状态预测后的 value；仅训练且 `disable_recompute=True` 时返回。 | bfloat16 | 同 `w`、`u` |
| `g_cumsum` | Tensor 或 None | 可选 | 按有效序列分 chunk 保存累计对数衰减；仅训练且 `disable_recompute=True` 时返回。内部累计和公开缓存均使用 FP32。 | float32 | BNSD/BSND：`[B,J,Nv,64,Dk]`；TND：`[J,Nv,64,Dk]` |
| `q_norm`、`k_norm` | Tensor 或 None | 可选 | 归一化后的 Query/Key；仅训练且 `use_qk_l2norm_in_kernel=True` 时返回，不受 `disable_recompute` 控制。BSND 输入也按 BNSD 返回，TND 保持 TND。 | bfloat16 | BNSD/BSND：`[B,Nqk,S,D]`；TND：`[T,Nqk,D]` |
| `q_rstd`、`k_rstd` | Tensor 或 None | 可选 | Query/Key 的逆范数；与对应 norm 缓存使用相同返回条件。 | float32 | 对应 norm 去掉最后一维 |
| `beta_sigmoid` | Tensor 或 None | 可选 | 核内 sigmoid 后的 beta；仅训练且 `use_beta_sigmoid_in_kernel=True` 时返回，不受 `disable_recompute` 控制。`allow_neg_eigval=True` 时包含乘 2 后的实际计算值。 | float32 | BNSD/BSND：`[B,S,Nv]`；TND：`[T,Nv]` |

非 None 的张量均位于输入所在设备。状态递推、`initial_state`、`final_state` 和保存的 `chunk_states` 均使用 FP32。

TND 的 chunk 缓存按 `token_start//64 + logical_batch` 定位各序列首槽，容量中可能有未使用槽。只消费有效序列和有效 chunk 范围；out 的无效尾区未定义。内部 out 按 64 对齐，公开 out 在非对齐场景可能为不连续视图。

## 约束说明

- 支持 Ascend 950，`Dk=Dv=128`，`B>=1`，序列长度大于 0；支持 BNSD、BSND、TND 和非对齐尾块。
- 当前实现支持 GQA，要求 `1 <= Nqk <= Nv <= 96` 且 `Nv % Nqk == 0`。
- 公开输入必须连续且位于同一设备；状态、`A_log`、`dt_bias` 使用 FP32，Q/K/V 使用 BF16。
- TND 前缀和首项为 0、末项为 T，严格递增；BNSD/BSND 的有效长度必须在 `[1,S]` 内。
- metadata 必须来自配套训练模块；不能复用推理模块或其他 ABI 版本的 metadata。
- TND 的 chunk 缓存按容量 `ceil(T/64)+B` 分配，仅每条序列实际使用的 chunk 有效。

自定义参数拦截统一抛出 `ValueError`。检查参与计算的输入类型、shape、dtype、设备和连续性；七个开关严格为 bool，`allow_neg_eigval=True` 要求开启 beta sigmoid。`scale` 可为 None 或有限数值，不接受 bool。关闭输出不免除计算输入检查。

Host 检查长度 Tensor 规格，长度内容合法性由设备 metadata 生产路径处理；主入口读取长度并核对 ABI4 metadata 的当前调用一致性，在启动前检查内部 buffer 的规格及可访问范围。底层设备、分配、编译和执行异常保留原诊断。

## 精度测试

在仓库根目录运行：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -q test/attention/flash_kda_fwd/
```

默认容差为 `atol=rtol=5e-3`。

## 性能测试

以下保留原版本的测量记录，不代表本轮修改后重新测量的结果。

测量环境：Ascend 950 PR，cannbotdsl 0.7.1、PyTorch 2.10.0、CANN 9.2.0，32 个 AIC。

`B=1`、`Nqk=Nv=96`、`S=16384`、`Dk=Dv=128`，BF16 输入、FP32 状态，`template="auto"`。

训练前向开启 `disable_recompute=True`、`return_intermediate_state=True`、`output_final_state=True`；Q/K 归一化、gate 激活和 beta sigmoid 均在内核完成，`allow_neg_eigval=False`。

每轮预热 50 次、采样 40 次，共 3 轮。下表为 120 次设备 kernel Task Duration 的中位数；未额外清理缓存，不包含 metadata 生成、编译和 host 分配时间。

| 布局 | Q/K/V shape | 延迟（μs） |
|:---|:---|---:|
| BNSD | `[1, 96, 16384, 128]` | 6114.947 |
| BSND | `[1, 16384, 96, 128]` | 6451.615 |
| TND | `[16384, 96, 128]` | 6435.956 |
