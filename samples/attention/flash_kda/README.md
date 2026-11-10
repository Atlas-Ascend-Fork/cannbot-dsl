# FlashKDA

依赖版本：CANNBot-DSL 0.8.0，面向 Ascend 950。

基于 CANNBot-DSL 的 Kimi Delta Attention prefill 融合算子，训练和推理共用实现：

- `flash_kda` 是算子公开入口，固定返回 `(out, final_state, chunk_states)`。
- `flash_kda_metadata` 是公开的调度元数据生成接口。调用方应令 `enable_grad` 与主算子的运行模式一致：推理为 False，训练为 True。
- 运行模式由 `torch.is_inference_mode_enabled()` 决定，`torch.no_grad()` 不会切换到推理分支。当前支持训练前向，尚未实现自动反向。

实现文件均位于本目录：

- [`flash_kda.py`](flash_kda.py)：公开算子入口及内部前向实现。
- [`flash_kda_metadata.py`](flash_kda_metadata.py)：Host 模板选择、AICPU metadata 生成与解码。
- [`flash_kda_staged.py`](flash_kda_staged.py)：按 chunk group 分阶段调度，通过 workspace 交换阶段结果。
- [`flash_kda_persistent.py`](flash_kda_persistent.py)：一个核连续处理序列头的 chunk，使递推状态驻留。
- [`flash_kda_validation.py`](flash_kda_validation.py)：主入口、metadata 与 Meta/FakeTensor 共用的参数和调度校验。

## 功能说明

按需融合 Q/K 行级 L2 normalize、gate 激活和 beta sigmoid，将序列按 64 个 token 切分，在 chunk 内构造因果 QK 矩阵及下三角逆矩阵，再跨 chunk 递推 FP32 状态。支持非对齐尾块和 GQA。训练模式按开关生成反向所需的缓存。

以下 $Q,K$ 表示按开关完成归一化后的输入，$g,\beta$ 表示完成对应激活后的值。每个 chunk 独立累加对数衰减，令

$$
G_t=\sum_{i=0}^{t}g_i,\quad \gamma_t=\exp(G_t),\quad
\tilde K=\gamma\odot K,\quad \bar K=\gamma^{-1}\odot K,\quad
\tilde Q=\operatorname{scale}\cdot(\gamma\odot Q).
$$

块内求解为

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

内部训练缓存 `Aqk`、`Akk`、`w`、`u`、`v_new` 分别对应 $M_{qk}$、$A_c$、$W$、$U$、$V_{\mathrm{new}}$；`qg` 对应未乘 scale 的 $\gamma\odot Q$，`kg` 对应 $K_{\mathrm{restored}}$，`g_cumsum` 对应 $G$。公开的 `chunk_states` 保存各 chunk 的入口状态 $H_c$，仅在推理模式按开关返回。训练缓存由实现内部管理，不通过公开接口返回；公开返回项见返回值说明。状态使用 FP32 递推。

metadata 的 Host 入口根据布局、有效长度、`enable_grad` 和调度配置，沿用原成本模型自动选择 staged/persistent 模板及分组上限；AICPU 根据选择结果生成调度表。两个模板共用公开接口和缓存布局。公开 metadata 接口不接受 `template`；定向模板测试使用内部构建函数，见同仓库测试用例。

## 快速开始

在仓库根目录设置 `export PYTHONPATH="$PWD/samples/attention:$PYTHONPATH"` 后运行。下面使用 BNSD 布局、完整序列长度和默认缩放值：

```python
import torch
import torch_npu

from flash_kda.flash_kda import flash_kda
from flash_kda.flash_kda_metadata import flash_kda_metadata

B, Nqk, Nv, S, D = 1, 1, 2, 65, 128
q = torch.randn(B, Nqk, S, D, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(B, Nv, S, D, device="npu", dtype=torch.bfloat16)
g = torch.randn_like(v)
beta = torch.randn(B, Nv, S, device="npu", dtype=torch.float32)
initial_state = torch.zeros(B, Nv, D, D, device="npu", dtype=torch.float32)
A_log = torch.zeros(Nv, device="npu", dtype=torch.float32)
dt_bias = torch.zeros(Nv, D, device="npu", dtype=torch.float32)

# Inference: explicitly enter inference_mode and generate inference metadata.
with torch.inference_mode():
    metadata = flash_kda_metadata(
        batch_size=B, nheads_qk=Nqk, nheads_v=Nv, seqlens=S,
        enable_grad=False, layout="BNSD",
    )
    out, final_state, chunk_states = flash_kda(
        q, k, v, g, beta, metadata, initial_state, A_log, dt_bias,
        layout="BNSD", output_final_state=True,
        return_intermediate_states=True,
    )

# Training: use the same flash_kda entry outside inference_mode.
metadata = flash_kda_metadata(
    batch_size=B, nheads_qk=Nqk, nheads_v=Nv, seqlens=S,
    enable_grad=True, layout="BNSD",
)
out, final_state, chunk_states = flash_kda(
    q, k, v, g, beta, metadata, initial_state, A_log, dt_bias,
    layout="BNSD", output_final_state=True,
)
assert chunk_states is None
```

## 参数说明

`flash_kda` 的公开签名：

```python
def flash_kda(
    q, k, v, g, beta, metadata,
    initial_state=None, A_log=None, dt_bias=None,
    cu_seqlens=None, seqused=None, *,
    scale=128 ** -0.5,
    output_final_state=False,
    return_intermediate_states=False,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    allow_neg_eigval=False,
    disable_recompute=True,
    lower_bound=None,
    layout="TND",
): ...
```

`B` 为逻辑序列数，`Nqk`、`Nv` 为 Q/K、Value 头数，`S` 为存储序列长度，`T` 为 packed token 总数，`Dk=Dv=D=128`。

| 参数 | BNSD shape / 取值 | 类型 | 必选/可选 | 说明 |
|:---|:---|:---|:---|:---|
| `q`, `k` | `[B,Nqk,S,D]` | BF16 | 必选 | Query/Key；可在内核归一化，也可由调用方预处理。 |
| `v` | `[B,Nv,S,D]` | BF16 | 必选 | Value。 |
| `g` | 与 `v` 相同 | BF16 / FP32 | 必选 | 原始 gate 或预处理的对数衰减；FP32 输入会转换为 BF16。 |
| `beta` | `[B,Nv,S]` | BF16 / FP32 | 必选 | 原始 logits 或预处理 beta；核内激活使用 FP32 操作数。 |
| `metadata` | `[M]` | int32 | 必选 | 由 `flash_kda_metadata` 生成的调度元数据。 |
| `initial_state` | `[B,Nv,Dv,Dk]` | FP32 | 可选 | 可为 None，此时使用全零状态；已传入的状态必须符合规格。 |
| `A_log` | `[Nv]` | FP32 / None | 条件必选 | 内核激活 gate 时必传；外部激活时可为 None。 |
| `dt_bias` | `[Nv,Dk]` | FP32 / None | 可选 | gate 偏置；None 按零处理。 |
| `cu_seqlens` | `[B+1]` | int32 / None | 条件必选 | 有效长度前缀和；TND 必传，首项为 0、末项为 T。 |
| `seqused` | `[B]` | int32 / None | 可选 | 每条序列的有效长度；同时提供 cu 时必须与相邻差值一致。 |
| `scale` | 默认 `D ** -0.5` | float / None | 可选 | 传 None 时按头维度计算缩放。 |
| `output_final_state` | 默认 False | bool | 可选 | 控制是否返回最终状态。 |
| `return_intermediate_states` | 默认 False | bool | 可选 | 仅推理且 True 时返回各 chunk 入口状态。 |
| `use_qk_l2norm_in_kernel` | 默认 True | bool | 可选 | True 执行 L2Norm；训练时由内部实现保存 norm/rstd。 |
| `use_gate_in_kernel` | 默认 True | bool | 可选 | False 时 g 已激活，忽略 `A_log`、`dt_bias`、`lower_bound`。 |
| `use_beta_sigmoid_in_kernel` | 默认 True | bool | 可选 | True 执行 sigmoid；训练时由内部实现保存激活结果。 |
| `allow_neg_eigval` | 默认 False | bool | 可选 | True 时 sigmoid beta 乘 2；要求开启 beta sigmoid。 |
| `disable_recompute` | 默认 True | bool | 可选 | 控制训练前向的中间缓存保存；True 时保留用于避免重计算的内部缓存，不改变公开返回项。 |
| `lower_bound` | 默认 None，数值范围 `[-5,0)` | int / float / None | 可选 | gate 开启时 None 使用 -exp(A_log) × softplus(g + dt_bias)，有限数值使用有界 sigmoid；关闭时忽略。 |
| `layout` | `"BNSD"` / `"BSND"` / `"TND"` | str | 可选 | 默认 `TND`。 |

BSND 的 `q/k` 为 `[B,S,Nqk,D]`，`v/g` 为 `[B,S,Nv,D]`，`beta` 为 `[B,S,Nv]`。TND 分别为 `[T,Nqk,D]`、`[T,Nv,D]`、`[T,Nv]`。

`flash_kda_metadata` 接收尺寸属性及可选的长度张量，用于生成传给 `flash_kda` 的调度元数据。公开签名：

```python
def flash_kda_metadata(
    cu_seqlens=None, seqused=None, *,
    batch_size, nheads_qk, nheads_v, seqlens,
    headdim_qk=128, headdim_v=128, enable_grad=False, layout="TND",
): ...
```

| 参数 | shape / 取值 | 类型 | 必选/可选 | 说明 |
|:---|:---|:---|:---|:---|
| `cu_seqlens` | `[B+1]` / None | int32 Tensor / None | 条件必选 | TND 布局必传；有效长度前缀和，首项为 0、末项为 T。BNSD/BSND 可省略。张量须连续且位于当前 NPU。 |
| `seqused` | `[B]` / None | int32 Tensor / None | 可选 | 每条序列的有效长度；同时提供 `cu_seqlens` 时必须与相邻差值一致。张量须连续且位于当前 NPU。 |
| `batch_size` | `B >= 1` | int | 必选（关键字） | 逻辑序列数。 |
| `nheads_qk` | `1 <= Nqk <= Nv` | int | 必选（关键字） | Query/Key 头数。 |
| `nheads_v` | `Nv <= 96`，`Nv % Nqk == 0` | int | 必选（关键字） | Value 头数。 |
| `seqlens` | 正整数 | int | 必选（关键字） | BNSD/BSND 的存储长度 S，或 TND 的总 token 数 T。 |
| `headdim_qk` | 默认 128，当前仅支持 128 | int | 可选 | Query/Key 头维度。 |
| `headdim_v` | 默认 128，当前仅支持 128 | int | 可选 | Value 头维度。 |
| `enable_grad` | 默认 False | bool | 可选 | 推理使用 False，训练前向使用 True；与主算子运行模式保持一致。 |
| `layout` | `"BNSD"` / `"BSND"` / `"TND"` | str | 可选 | 默认 `"TND"`，与主算子的布局保持一致。 |

调用时必须显式提供 `seqlens`：BNSD/BSND 为存储长度 S，TND 为总长度 T。TND 必须传连续 INT32 `[B+1]` 的 `cu_seqlens`；padded 布局可用 cu 或 `seqused` 指定有效长度，不传时均为 S。

给定的长度张量必须位于当前 NPU，生成的 metadata 也位于当前 NPU。metadata 和主算子必须使用一致的长度参数；主算子不会只凭 metadata 替代 `cu_seqlens` / `seqused` 入参。

`enable_grad=False` 用于推理，训练使用 True。主算子不会从 `enable_grad` 切换运行模式；切换模式、设备、设备有效核数、形状、布局或有效长度后重新生成 metadata。相同调度配置下可以复用 metadata。真实前向会阻塞回读 metadata 和给定的长度张量，校验本次调用的序列边界与调度；Meta/FakeTensor 只执行静态检查。尺寸、长度、连续性和 GQA 等非法参数抛出 `ValueError`。

主算子的第六个必填输入为 `metadata`，可按位置或关键字传入；从 `scale` 开始的属性按关键字传入；`scale` 的签名默认值为 `128 ** -0.5`，显式传 None 时仍使用该缩放值。

## 返回值说明

`flash_kda` 固定返回以下三项，未启用的可选输出为 None：

```python
(out, final_state, chunk_states)
```

令 BNSD/BSND 的 `J=ceil(S/64)`，TND 的容量 `J=ceil(T/64)+B`：

| 参数名 | 参数类型 | 必选/可选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `out` | Tensor | 必选 | 序列输出，布局与 `v` 相同；metadata 标记的有效序列范围之外，输出值未定义。 | bfloat16 | 与 `v` 相同 |
| `final_state` | Tensor 或 None | 可选 | `output_final_state=True` 时返回每条序列的最终递归状态，训练和推理均生效；不原地修改 `initial_state`。 | float32 | `[B,Nv,Dv,Dk]` |
| `chunk_states` | Tensor 或 None | 可选 | 各 chunk 入口状态；仅推理且 `return_intermediate_states=True` 时返回，训练时为 None。 | float32 | BNSD/BSND：`[B,J,Nv,Dv,Dk]`；TND：`[J,Nv,Dv,Dk]` |

非 None 的张量均位于输入所在设备。状态递推、`initial_state`、`final_state` 和保存的 `chunk_states` 均使用 FP32。训练中间缓存由内部实现管理，不属于公开返回项。

TND 的 chunk 状态按 `token_start//64 + logical_batch` 定位各序列首槽，容量中可能有未使用槽。只消费有效序列和有效 chunk 范围；out 的无效尾区未定义。内部 out 按 64 对齐，公开 out 在非对齐场景可能为不连续视图。

`flash_kda_metadata` 返回当前 NPU 上的一维 int32 Tensor `metadata`，形状为 `[M]`，其中 M 由调度容量决定。将其作为 `flash_kda` 的第六个输入传入。

## 约束说明

- 支持 Ascend 950，`Dk=Dv=128`，`B>=1`，序列长度大于 0；支持 BNSD、BSND、TND 和非对齐尾块。
- 当前实现支持 GQA，要求 `1 <= Nqk <= Nv <= 96` 且 `Nv % Nqk == 0`。
- 参与计算和校验的公开输入必须连续；不校验输入之间的设备一致性。状态、`A_log`、`dt_bias` 使用 FP32，Q/K/V 使用 BF16。
- TND 前缀和首项为 0、末项为 T，严格递增；BNSD/BSND 的有效长度必须在 `[1,S]` 内。
- metadata 必须由本目录的配套模块生成，并与当前布局、形状、长度、运行模式和设备有效核数匹配。应使用 `flash_kda_metadata` 生成；合并前的旧推理 metadata 或其他 ABI 版本的 metadata 不可混用。
- TND 的 chunk 缓存按容量 `ceil(T/64)+B` 分配，仅每条序列实际使用的 chunk 有效。

自定义参数拦截统一抛出 `ValueError`。检查参与计算的输入类型、shape、dtype、设备和连续性；七个开关严格为 bool，`allow_neg_eigval=True` 要求开启 beta sigmoid。`scale` 可为 None 或有限数值，不接受 bool。关闭输出不免除计算输入检查。

Host 检查长度张量的规格和内容，metadata 设备内核也会报告非法长度；主入口读取长度并核对 ABI4 metadata 的当前调用一致性，在启动前检查内部 buffer 的规格及可访问范围。底层设备、分配、编译和执行异常保留原诊断。

## 验证与测试

在仓库根目录运行 `flash_kda` 和 `flash_kda_metadata` 的测试：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -q \
    test/attention/flash_kda/
```

默认精度容差为 `atol=rtol=5e-3`。`test/attention/flash_kda/test_flash_kda.py` 直接调用 Python 入口，检查返回契约、Meta 输出和 NPU 前向精度。

Eager 前向读取并校验长度和 metadata。使用 NPU 图时，先在图外生成 metadata，用同一配置预热，再直接捕获公开入口。捕获使用生成时已校验的 Host 调度信息。

```python
metadata = flash_kda_metadata(**metadata_args)
flash_kda(**inputs, metadata=metadata)
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    outputs = flash_kda(**inputs, metadata=metadata)

inputs["v"].copy_(next_v)
graph.replay()
```

从生成 metadata 到最后一次重放，保留原始 metadata 和长度张量，保持其内容、形状和调度配置固定。改变序列边界时重新生成 metadata 并捕获。

## 历史性能记录

以下保留合并前推理与训练版本的测量记录，尚未对本次统一实现重新测量。两组记录的依赖版本和缓存处理方式不同。

### 推理版本

测量环境：Ascend 950 PR，cannbotdsl 0.8.0、PyTorch 2.10.0、CANN 9.2.1，32 个 AIC。

`B=1`、`Nqk=Nv=96`、`S=16384`、`Dk=Dv=128`，BF16 输入、FP32 状态，使用自动模板路由。

每轮预热 50 次、采样 40 次，共 3 轮。下表为 120 次设备 kernel Task Duration 的中位数。每次计时前使用 1 GiB 数据进行 L2 软件驱逐并同步；清 L2、metadata 生成、编译和 host 分配时间不计入延迟。

| 布局 | Q/K/V shape | 延迟（μs） |
|:---|:---|---:|
| BNSD | `[1, 96, 16384, 128]` | 2455.312 |
| BSND | `[1, 16384, 96, 128]` | 2487.249 |
| TND | `[16384, 96, 128]` | 2484.931 |

### 训练前向版本

测量环境：Ascend 950 PR，cannbotdsl 0.7.1、PyTorch 2.10.0、CANN 9.2.0，32 个 AIC。

`B=1`、`Nqk=Nv=96`、`S=16384`、`Dk=Dv=128`，BF16 输入、FP32 状态，使用自动模板路由。

原记录的训练前向开启 `disable_recompute=True`、`output_final_state=True`，Q/K 归一化、gate 激活和 beta sigmoid 均在内核完成，`allow_neg_eigval=False`。旧记录中的中间状态开关现统一命名为 `return_intermediate_states`；当前接口仅在推理模式下公开返回 `chunk_states`。

每轮预热 50 次、采样 40 次，共 3 轮。下表为 120 次设备 kernel Task Duration 的中位数；未额外清理缓存，不包含 metadata 生成、编译和 host 分配时间。

| 布局 | Q/K/V shape | 延迟（μs） |
|:---|:---|---:|
| BNSD | `[1, 96, 16384, 128]` | 6114.947 |
| BSND | `[1, 16384, 96, 128]` | 6451.615 |
| TND | `[16384, 96, 128]` | 6435.956 |
