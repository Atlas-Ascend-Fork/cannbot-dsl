# MlaPrologV4

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |
| <term>Kirin X90处理器系列产品</term> | × |
| <term>Kirin 9030处理器系列产品</term> | × |

使用 CANNBot-DSL 0.8.2，目标架构为 `dav-3510`；已验证环境为 CANN 9.2.1。

## 功能说明

- 用于 Multi-Head Latent Attention（MLA）推理前处理，接收 BF16 激活与权重，完成投影、均方根归一化（RMSNorm）、旋转位置编码（RoPE）、Q 输出及 KV Cache 原地更新；Q/Cache 可按模式量化。
- 令 `Lkv=512`，计算依赖为：

$$
R=\operatorname{RMSNorm}(x\cdot wqa^T,\gamma_{qa})\cdot\texttt{qr\_scale},\qquad
U=\operatorname{reshape}_{T,N,Dn+Dr}(R\cdot wqb^T),
$$
$$
Q_f=\operatorname{Concat}(U_{:Dn}\cdot wkb,\operatorname{RoPE}(U_{Dn:})),
$$
$$
V=x\cdot wkva^T,\qquad
KV=\operatorname{RMSNorm}(V_{:512},\gamma_{kva})\cdot\texttt{kvr\_scale},\qquad
KR=\operatorname{RoPE}(V_{512:}).
$$

其中 $\operatorname{RMSNorm}(v,\gamma)=v\gamma/\sqrt{\operatorname{mean}(v^2)+\texttt{norm\_eps}}$。
公式表示数学依赖，矩阵乘使用 FP32 累加并保留实现中的 BF16 中间转换；RoPE 以 FP32 计算后转 BF16。
`kvr_scale` 只作用于 KV，不作用于 KR。最终 Q 和 Cache 按下述模式编码；可选 `qr` 为 QA 归一化并缩放后的 BF16 结果。

## 参数说明

以下为公开函数 `mla_prolog_v2` 的参数。`T` 为 token 数，`N` 为 head 数，`B` 和 `S` 分别为 Cache 页数和页大小，`Dcache` 由输出模式确定。
HiF8 输出使用 UINT8 保存原始浮点编码，不表示整数数值量化。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| `x` | 输入 | 输入隐状态，Shape 为 `(T,dim)`。 | BFLOAT16 | ND |
| `wqa` | 输入 | QA 下投影权重，ND Shape 为 `(q_lora,dim)`。 | BFLOAT16 | ND、FRACTAL_NZ |
| `wqb` | 输入 | Q 上投影权重，ND Shape 为 `(N*(Dn+Dr),q_lora)`。 | BFLOAT16 | ND、FRACTAL_NZ |
| `wkva` | 输入 | KVA 投影权重，ND Shape 为 `(512+Dr,dim)`。 | BFLOAT16 | ND、FRACTAL_NZ |
| `wkb` | 输入 | 每头 Q-NoPE 吸收权重，Shape 为 `(N,Dn,512)`，不参与 Cache 生成。 | BFLOAT16 | ND |
| `norm_weight_qa` | 输入 | QA RMSNorm 的 gamma，Shape 为 `(q_lora,)`。 | FLOAT | ND |
| `norm_weight_kva` | 输入 | KV RMSNorm 的 gamma，Shape 为 `(512,)`。 | FLOAT | ND |
| `kv_cache` | 输入/输出 | 原地更新的 Cache，公开 Shape 为 `(B,S,1,Dcache)`。 | BFLOAT16、FLOAT8_E4M3FN、UINT8（HiF8） | ND 载体 |
| `cache_index` | 输入 | 每个 token 的线性 Cache 槽位，Shape 为 `(T,)`。负值跳过写入。 | INT64 | ND |
| `quant_scale_kv` | 可选输入 | Cache 量化乘子，Shape 为 `(1,)`，默认 `None`；模式 2/3 必填，模式 0/1 必须为 `None`。 | FLOAT | ND |
| `rope_sin` | 可选输入 | RoPE 正弦参数，Shape 为 `(T,Dr)`，默认 `None`；启用 RoPE 时必填。 | BFLOAT16 | ND |
| `rope_cos` | 可选输入 | RoPE 余弦参数，与 `rope_sin` 成对提供，默认 `None`。 | BFLOAT16 | ND |
| `norm_eps` | 可选属性 | RMSNorm 的 epsilon，默认 `1e-5`。 | FLOAT | - |
| `qkv_quant_mode` | 可选属性 | Q/Cache 输出模式，使用 `QkvQuantMode`，默认 `A16C16`。 | 枚举 | - |
| `cache_mode` | 可选属性 | Cache 地址映射，使用 `CacheMode`，默认 `PA_BSND`。 | 枚举 | - |
| `qr_scale` | 可选属性 | QA RMSNorm 后的乘子，默认 `1.0`，同时影响 Q 和 qr。 | FLOAT | - |
| `kvr_scale` | 可选属性 | KV RMSNorm 后的乘子，默认 `1.0`，不缩放 KR。 | FLOAT | - |
| `rope_mode` | 可选属性 | 使用 `RopeMode`，默认 `ROPE_INTERLEAVE_HALF`。 | 枚举 | - |
| `return_qr` | 可选属性 | 是否返回 qr，默认 `False`。 | BOOL | - |
| `q` | 输出 | 编码后的 Q，Shape 为 `(T,N,512+Dr)`。 | BFLOAT16、FLOAT8_E4M3FN、UINT8（HiF8） | ND |
| `descale_q` | 输出 | Q 反量化乘子；模式 2 为 `(T,N,1)`，非空模式 3 为 `(1,)`，其余为 `(0,)`。 | FLOAT | ND |
| `qr` | 输出 | QA 归一化并缩放后的结果；启用时为 `(T,q_lora)`，否则为 `(0,)`。 | BFLOAT16 | ND |

对逻辑权重 `W[O,K]`，ND 传入 `(O,K)`；NZ 传入转置后的 `(K,O)` 并转换为真实 format 29，物理 Shape 为 `(O/16,K/16,16,16)`。
三路投影权重须统一格式，普通四维 ND reshape 不能代替 NZ。

### 输出与 RoPE 模式

下表数值用于说明枚举编码，公开调用使用本函数上的 `QkvQuantMode` 成员。

| QkvQuantMode 成员 | 编码 | Q | Cache |
| --- | ---: | --- | --- |
| `A16C16` | 0 | BF16 | BF16 的 `[KV,KR]`，`Dcache=512+Dr`。 |
| `A16C8_KV_NOPE_FP8_E4M3_PER_TOKEN_GROUP_GROUP_SIZE_128_ROPE_BF16_SCALE_FLOAT_KV_CACHE_LAYOUT_N_NOPE_ROPE_SCALE` | 1 | BF16 | KV 每 token 每 128 值量化为 E4M3FN，KR 为 BF16，附带 4 个 FLOAT descale，`Dcache=512+2*Dr+16`。 |
| `A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR` | 2 | 每 token/head 动态量化的 E4M3FN | KV/KR 使用同一个输入量化乘子，E4M3FN，`Dcache=512+Dr`。 |
| `A8C8_QKV_HIFLOAT8_PER_TENSOR` | 3 | 全 T、全 head 共用 descale 的 HiF8 | KV/KR 使用同一个输入量化乘子，HiF8，`Dcache=512+Dr`。 |

Q 的动态 descale 为 `max(abs(Q))/448`（模式 2）或 `/32768`（模式 3），反量化为 `Decode(q)*descale_q`；全零 Q 输出零编码和零 descale。
模式 1 的 KV descale 为 `max(max(abs(KV_group)),1e-4)/448`；Cache 依次存放 KV 编码、KR 的 BF16 字节和 4 个 FLOAT descale，整体使用 E4M3FN Tensor 承载。
模式 2/3 的 Cache 编码为 `Quantize([KV,KR]*quant_scale_kv)`；乘子大于 0 时反量化需除以该乘子。

`RopeMode.NO_ROPE=0` 不旋转；`ROPE_INTERLEAVE_HALF=1` 按偶/奇通道配对；`ROPE_HALF=2` 按前后半区配对。两种旋转模式的输出均按前后半区排列，sin/cos 按 head 广播。

## 约束说明

- `T>=0`；`dim` 为 1024～8192 内的 256 倍数；`q_lora` 为 640、1024、1536 或 2048；`N` 为 1～128；`(Dn,Dr)` 为 `(128,64)`、`(192,64)` 或 `(256,0)`。
- `B>0`；`S` 为 16～1024 内的 16 倍数。模式 1 只支持 PA_BSND；其余模式支持 PA_BSND 和 PA_NZ。PA_NZ 的 Cache 物理布局为 `(B,Dcache/C0,S,C0)`，`C0=32/元素字节数`，与权重 FRACTAL_NZ 不同。
- 输入位于同一 NPU。除 Cache 外均须连续；Cache 允许非零 storage offset 和首轴非连续，其元素 stride 须为 `(stride0,Dcache,Dcache,1)`，且 `stride0>=S*Dcache`，页间空隙保持不变。
- 调用方保证非负 `cache_index` 在 `[0,B*S)` 内且唯一；算子不回读索引数值。负索引仅跳过 Cache 写入，不跳过 Q/qr 计算或全局量化统计。
- `norm_eps` 转 FP32 后须有限且大于 0；`qr_scale/kvr_scale` 须为有限 FP32，允许 0 和负数；三个属性均拒绝 bool。模式 2/3 的 `quant_scale_kv` 须有限且非负，Eager 会回读检查，图模式由调用方在图外保证其值合法。
- 公开入口接受本函数上的枚举成员，也可通过 `QkvQuantMode(1)` 等方式构造，不接受裸整数、字符串或 bool。`Dr=0` 必须指定 `NO_ROPE`；禁用 RoPE 时 sin/cos 可同时省略，传入时仍检查 Shape 和类型。
- `T=0` 返回对应空输出，不启动 Kernel、不更新 Cache。返回值固定为 `(q,descale_q,qr)`，不另返回 Cache；不支持自动求导和功能化 Cache。
- 输出及临时空间由内部按需分配，无需外部 workspace 接口；临时空间按阶段复用，大 T 按容量分块，模式 3 保持全 T、全 head 的全局 Q scale。非空调用按当前 Stream 的可用 MIX 核资源执行。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| PyTorch API | [非量化 MLA 测试](../../../test/attention/mla_prolog_v4/test_mla_prolog_v4.py) | 通过 [mla_prolog_v2](mla_prolog_v4.py) 调用，支持 Eager、ACLGraph 和 `torch.compile` 的 `npugraph_ex` 后端。 |

安装包含该算子的 wheel 后调用：

```python
from ops.mla_prolog_v4 import mla_prolog_v2 as prolog

attrs = dict(
    qkv_quant_mode=prolog.QkvQuantMode.A16C16,
    cache_mode=prolog.CacheMode.PA_BSND,
)
# inputs 按参数表准备。
q, descale_q, qr = prolog(**inputs, **attrs)
```

源码运行时将本目录加入 `PYTHONPATH`，从 `mla_prolog_v4` 模块导入 `mla_prolog_v2`。
ACLGraph/NPUGraph-EX 使用同一公开入口；在图外调用一次完成值检查和编译准备，准备调用会更新 Cache。
捕获及回放期间保持 Shape、dtype、format 和地址不变；输入内容更新后保证索引/scale 合法并建立流依赖。
NPUGraph-EX 使用 `fullgraph=True,dynamic=False,options={"force_eager":False,"clone_input":False}`，图回放不重新执行 Python 值检查。

在已激活 CANN 和 DSL 0.8.2 的环境中，从仓库根目录编包：

```bash
python packaging/native/build.py --operator mla_prolog_v4 --target dav-3510
```

Native 包须覆盖实际调用的编译键；匹配包内索引时直接加载二进制，不重建 IR。
其他编译条件相同，8191、8192、8193 等不同 T 复用 Native；改变 NZ `(dim,q_lora)` 仍可能需要不同二进制。
联合编包、安装及 Native 模式设置见[Native 打包说明](../../../packaging/native/README.md)，其中 DSL 环境使用本算子要求的 0.8.2。
