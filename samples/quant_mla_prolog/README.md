# quant_mla_prolog

## 支持规格

| 参数 | 范围 |
| --- | --- |
| CANNBot-DSL | 0.8.0 |
| CANN | 9.2.1 |
| dim | 1024～8192，256 对齐 |
| q_lora | 640、1024、1536、2048 |
| kv_lora | 512 |
| N | 1～128 |
| (Dn,Dr) | (128,64)、(192,64)、(256,0) |
| 输入量化 | MXFP8 E4M3FN + E8M0 descale，或 HiF8 + FP32 descale |
| 权重 | 三路投影权重统一 ND 或统一真实 NZ；wkb 为 ND BF16 |
| 输出模式 | 0/1/2/3；mode 1 仅 PA_BSND，其余支持 PA_BSND/PA_NZ |
| Cache | block_number>0；block_size=16～1024，16 对齐；负 index 跳过写入；调用方保证非负 index 在容量内且唯一 |
| 空输入 | T=0 无 Kernel 启动；返回固定四个 Tensor 槽位 |

## 算子接口

`quant_mla_prolog` 接收量化激活、量化投影权重及 descale，支持非量化或量化的 Q/Cache 输出。
以下为公开 Python 接口，枚举通过同一个 `qmlaprolog` 模块访问。

```python
import quant_mla_prolog as qmlaprolog


def quant_mla_prolog(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    quant_scale_kv=None,
    rope_sin=None,
    rope_cos=None,
    *,
    norm_eps=1e-5,
    w_qr_quant_mode,
    qkv_quant_mode,
    cache_mode,
    qr_scale=1.0,
    kvr_scale=1.0,
    rope_mode=qmlaprolog.RopeMode.INTERLEAVE_HALF,
    return_qr=False,
):
    ...  # 固定返回 (q, descale_q, qr, descale_qr)，原地更新 kv_cache。
```

### 输入Tensor

`T>=0`；其他维度范围见“支持规格”。所有输入位于同一 NPU，满足连续性要求。
HiF8 使用 uint8/int8 原始编码载体，不表示整数数值量化。

| 参数 | Torch逻辑Shape | Torch dtype | Format与用途 |
| --- | --- | --- | --- |
| `x` | `(T,dim)` | MXFP8：E4M3FN；HiF8：uint8/int8 载体 | ND，输入隐状态 |
| `wqa` | `(q_lora,dim)` | 与输入量化模式对应 | ND 或真实 FRACTAL_NZ，QA 投影 |
| `wqb` | `(N*(Dn+Dr),q_lora)` | 与输入量化模式对应 | ND 或真实 FRACTAL_NZ，Q 上投影 |
| `wkva` | `(512+Dr,dim)` | 与输入量化模式对应 | ND 或真实 FRACTAL_NZ，KVA 投影 |
| `wkb` | `(N,Dn,512)` | BF16 | ND，将 Q-Nope 吸收到 512 维 latent 空间 |
| `descale_x` | MXFP8：`(T,dim/32)`；HiF8：`(1,)` | E8M0 / FP32 | ND，激活反量化乘子 |
| `descale_wqa` | 见下方权重 descale 规则，`O=q_lora,K=dim` | E8M0 / FP32 | ND 载体，QA 权重反量化乘子 |
| `descale_wqb` | 同规则，`O=N*(Dn+Dr),K=q_lora` | E8M0 / FP32 | ND 载体，QB 权重反量化乘子 |
| `descale_wkva` | 同规则，`O=512+Dr,K=dim` | E8M0 / FP32 | ND 载体，KVA 权重反量化乘子 |
| `norm_weight_qa` | `(q_lora,)` | FP32 | ND，QA RMSNorm gamma |
| `norm_weight_kva` | `(512,)` | FP32 | ND，KV RMSNorm gamma |
| **`kv_cache`（原地输入输出）** | mode 0/2/3：`(block_number,block_size,1,512+Dr)`；mode 1：`(block_number,block_size,1,512+2*Dr+16)` | mode 0：BF16；mode 1/2：E4M3FN；mode 3：uint8/int8 载体 | ND 载体，内部按 PA_BSND/PA_NZ 寻址；mode 1 仅 PA_BSND |
| `cache_index` | `(T,)` | INT64 | ND，token 写入的线性 Cache 槽位；数值合法性由调用方保证 |
| `quant_scale_kv` | `(1,)` | FP32 | ND，mode 2/3 必填，有限且 ≥0，不支持负数；其余必须为 `None` |
| `rope_sin` / `rope_cos` | `(T,Dr)` | BF16 | ND，启用 RoPE 时必填，成对提供 |

**三路权重必须统一 ND（format 2）或统一真实 NZ（format 29）。**
真实 NZ 的 Torch 逻辑 Shape 仍是上表二维 Shape；每个 `(O,K)` 权重的物理分块为
`(O/32,K/16,16,32)`。普通 ND reshape 成四维不合法。

MXFP8 权重 descale 的 ND Shape 为 `(O,K/64,2)`，对应 NZ 权重时为
`(O/16,K/64,16,2)`；两者的 descale Tensor 自身均为 ND。
HiF8 权重 descale 为 `(1,O)` FP32，不随权重 ND/NZ 改变。

权重 FRACTAL_NZ 与 Cache PA_NZ 不可互相替代。Cache PA_NZ 按
`[block_number,D/C0,block_size,C0]` 存放，`C0=32/element_size`，公开 Tensor 仍为 BSND 外形。

### 属性与枚举

| 属性 | 类型与合法值 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `norm_eps` | 转 FP32 后有限且严格大于 0，拒绝 bool | `1e-5` | QA/KV 两路 RMSNorm 共用 epsilon |
| `w_qr_quant_mode` | `qmlaprolog.WQrQuantMode` 或合法整数编码 | 必填 | 输入、投影权重和 QR 的量化模式 |
| `qkv_quant_mode` | `qmlaprolog.QkvQuantMode` 或合法整数编码 | 必填 | 最终 Q/Cache 量化模式 |
| `cache_mode` | `qmlaprolog.CacheMode` 或合法整数编码 | 必填 | Cache 物理地址映射 |
| `qr_scale` | 转 FP32 后有限，允许 0 和负数，拒绝 bool | `1.0` | QA RMSNorm 后、QR 量化前缩放 |
| `kvr_scale` | 转 FP32 后有限，允许 0 和负数，拒绝 bool | `1.0` | KV RMSNorm 后缩放，**不缩放 KR** |
| `rope_mode` | `qmlaprolog.RopeMode` 或合法整数编码 | `INTERLEAVE_HALF` | RoPE 变换方式 |
| `return_qr` | 严格 bool | `False` | 是否写出 QR 及其 descale，不改变主 Q 路径 |

三个浮点属性在 Torch 分派前检查，均拒绝 bool、NaN、Inf 及 FP32 溢出；仅 `norm_eps` 额外拒绝 0、负数及 FP32 下溢为 0 的值。
`qr_scale/kvr_scale` 不传使用 1.0，显式传入按该值相乘：1 不改变结果，0 将对应有限值置零，负数改变符号。

| 枚举成员 | 内部编码 | 语义 |
| --- | ---: | --- |
| `WQrQuantMode.MXFP8` | 1 | E4M3FN 数据，K 轴每 32 值共享 E8M0 descale |
| `WQrQuantMode.HIF8` | 2 | HiF8 数据；激活 per-tensor、权重 per-channel FP32 descale |
| `QkvQuantMode.NO_QUANT` | 0 | Q、KV、KR 均为 BF16 |
| `QkvQuantMode.KV_FP8_PER_TOKEN_GROUP` | 1 | Q 为 BF16；KV 按 token/group 动态量化，KR 为 BF16 |
| `QkvQuantMode.Q_KV_FP8` | 2 | Q 按 token/head 动态量化；KV/KR 共用输入量化乘子 |
| `QkvQuantMode.Q_KV_HIF8` | 3 | Q 全局 per-tensor 动态量化；KV/KR 共用输入量化乘子 |
| `CacheMode.PA_BSND` / `PA_NZ` | 0 / 1 | 两种 Cache 物理地址映射 |
| `RopeMode.NONE` | 0 | 不旋转，Dr=64 时位置编码部分直通 |
| `RopeMode.INTERLEAVE_HALF` | 1 | 交错偶/奇通道取对，输出前后半区 |
| `RopeMode.HALF` | 2 | 前后半区取对，输出前后半区 |

合法 `(qkv_quant_mode,w_qr_quant_mode)` 为 `(0,1)`、`(0,2)`、`(1,1)`、`(2,1)`、`(3,2)`。
枚举参数也接受对应合法整数，不接受字符串或 bool。`Dr=0` 必须显式指定 `RopeMode.NONE`。
禁用 RoPE 时 sin/cos 可同时为 `None`；传入时仍须成对满足 Shape/dtype，但不参与计算。

### 输出与Cache原地更新

返回值始终为四个 Tensor，不返回独立 Cache Tensor：

| 输出/更新项 | Shape | Torch dtype | 语义 |
| --- | --- | --- | --- |
| `q`，槽位0 | `(T,N,512+Dr)` | mode 0/1：BF16；mode 2：E4M3FN；mode 3：uint8 | `[Q-Nope(512),Q-RoPE(Dr)]` |
| `descale_q`，槽位1 | mode 2：`(T,N,1)`；T>0 且 mode 3：`(1,)`；其余：`(0,)` | FP32 | Q 反量化乘子 |
| `qr`，槽位2 | true：`(T,q_lora)`；false：`(0,)` | MXFP8：E4M3FN；HiF8：uint8 | QA RMSNorm、缩放后的量化结果 |
| `descale_qr`，槽位3 | QR 启用时 MXFP8：`(T,q_lora/32)`；HiF8：T>0 为 `(1,)`，T=0 为 `(0,)`；QR 禁用时 `(0,)` | MXFP8：E8M0；HiF8：FP32 | QR 反量化乘子 |
| `kv_cache`，输入原地更新 | 与输入完全一致 | 与输入完全一致 | 命中位置更新；未命中字节、data pointer、storage identity 不变 |

mode 1 的 payload 为 `[KV FP8字节, KR BF16字节, 4个FP32 descale字节]`，
Dr=64 为 656 字节，Dr=0 为 528 字节；KR 和 descale 部分不能按 FP8 数值解释。
mode 2/3 的 `quant_scale_kv` 是量化乘子：`payload=Quantize(value*quant_scale_kv)`，
系数 >0 时可用 `value=Decode(payload)/quant_scale_kv` 反量化；系数为 0 时，有限输入量化为零，无法恢复原值，测试直接比较量化输出。Q/QR 的 descale 是输出反量化乘子。

### 计算与调用约束

- QA/KVA/WQB 使用量化投影、FP32 累加及 FP32 输出。QR 内部量化始终参与 WQB；`return_qr` 仅控制公开输出。
- Q-Nope 在 WKB 前转 BF16，WKB FP32 累加后输出 BF16；RoPE FP32 计算后转 BF16。KV 归一化与 `kvr_scale` 不作用于 KR。
- 调用方须保证非负 `cache_index` 小于 `block_number*block_size` 且唯一，算子仅检查其 shape、dtype、device 和布局，不回读或校验索引数值。所有负 INT64 值仅跳过 Cache 写入，token 仍参与 Q/QR 和全局量化统计。
- **T=0 仍校验接口元信息，在 Host 返回**；有效输出保留零长度维度，全局 HiF8 descale 为 `(0,)`。不回读设备值、不查询 Stream、不分配 Kernel workspace、不编译或启动 Kernel，Cache 不变。
- 非空调用必须成功查询当前 Stream 核数配额；接口缺失、失败、非法值明确报错，零配额拒绝启动，不回退整卡核数。
- Eager、Native 和图执行只校验元信息，不回读设备内容。索引数值合法性由调用方保证；非空输入启用 `quant_scale_kv` 时，调用方须先调用 `quant_mla_prolog_precheck` 检查其值，更新 scale 后须重新预检。更新输入后须建立流依赖。公开执行不以 Torch 小算子补算、Cache clone/copy-back 或隐式格式转换修复输入；不支持自动求导或功能化 Cache。

## 计算公式

以下描述数学依赖和量化轴，不以实数等价替代实际 Cast、累加顺序和特殊值规则。
令 `Lq=q_lora`、`Lkv=512`，`Dn` 是 WKB 的归约维，`N` 是 head 数。
权重公开 Shape 按 `(输出维, 输入维)` 排列，WKB 为 `(N,Dn,512)`。

### 1. 量化投影与 RMSNorm

记投影为 $P(A,W;s_A,s_W)$。MXFP8 按 K 轴每 32 个值共享 E8M0 descale：

$$
P_{t,o}=\sum_g s_A[t,g]s_W[o,g]
  \sum_{k=32g}^{32g+31}\operatorname{decode}_{E4M3}(A_{t,k})
                              \operatorname{decode}_{E4M3}(W_{o,k}).
$$

这一路使用原生 MX 随路 scale。HiF8 则先对解码值完成 FP32 累加，再依次乘
权重 descale 和激活 descale，不是 MX 分组随路量化：

$$
C_0=\operatorname{MatMul}_{FP32}(\operatorname{decode}_{HiF8}(A),
                                   \operatorname{decode}_{HiF8}(W)^T),\quad
C_1=\operatorname{FP32}(C_0s_W),\quad P=\operatorname{FP32}(C_1s_A).
$$

归一化沿每行完整特征维进行，Gamma 在 FP32 域相乘：

$$
\operatorname{RMSNorm}(v,\gamma)_j=
 \frac{v_j\gamma_j}{\sqrt{\frac{1}{d}\sum_{i=0}^{d-1}v_i^2+\texttt{norm\_eps}}}.
$$

### 2. Q 与 QR 路径

$$
A=P(x,WQA),\quad R=\operatorname{RMSNorm}(A,\gamma_{QA})\cdot\texttt{qr\_scale},
\qquad (QR,s_{QR})=\operatorname{DynamicQuant}(R),
$$
$$
U=\operatorname{reshape}_{T,N,Dn+Dr}(P(QR,WQB;s_{QR},s_{WQB})),
$$
$$
Q_{NoPE}[t,n,:]=\operatorname{BF16}\!\left(
 \operatorname{MatMul}_{FP32}(\operatorname{BF16}(U[t,n,:Dn]),WKB[n,:,:])\right),
$$
$$
Q_{RoPE}=\operatorname{BF16}(\operatorname{RoPE}(U[:,:,Dn:])),\qquad
Q_f=\operatorname{concat}(Q_{NoPE},Q_{RoPE}).
$$

QR 的 MX 动态量化沿每行每 32 个元素计算 scale，保留本实现的局部 BF16 预舍入及
SAT 语义；HiF8 QR 沿整个 `(T,Lq)` 张量计算一个 FP32 descale。`return_qr=False`
仅隐藏 QR 输出，不移除内部 QR 计算。

| QKV 模式 | 公开 Q | `descale_q` |
|---|---|---|
| 0 / 1 | `Q_f`，BF16 | 空 Tensor |
| 2 | `E4M3(Q_f / s)`，每个 `(token,head)` 的 `s=max(abs(Q_f))/448` | FP32 `(T,N,1)` |
| 3 | `HiF8(Q_f / s)`，整个 Q 的 `s=max(abs(Q_f))/32768` | FP32 `(1,)` |

动态量化全零范围输出零编码；NaN、Inf、饱和、次正规及舍入边界遵循本算子的数值合同，
不能直接用未处理 `s=0` 的除法替代实现。

### 3. K/V、RoPE 与 Cache

$$
V=P(x,WKVA),\quad KV=\operatorname{RMSNorm}(V[:,:512],\gamma_{KVA})
 \cdot\texttt{kvr\_scale},\quad KR=\operatorname{BF16}_{SAT}(\operatorname{RoPE}(V[:,512:])).
$$

KR 不参加 KV 的 RMSNorm。模式 0 写入 BF16 的 `[KV,KR]`；模式 1 分四组、每组
128 个 KV 元素动态量化，$s_g=\max(\max|KV_g|,10^{-4})/448$，写入
`[KV的512字节E4M3编码, KR的BF16字节, 4个FP32 scale字节]`。
模式 2 / 3 分别写入 `E4M3([KV,KR] * quant_scale_kv)` /
`HiF8([KV,KR] * quant_scale_kv)`。这里 `quant_scale_kv` 是输入的**量化乘数**，
不是输出 descale，不能与 Q/QR 的除数混用。

令 $z$ 为 RoPE 输入，$h=Dr/2$，$c,s$ 为对应 token 的 cos/sin：

- `NONE`：不旋转。
- `HALF`：$y=c\odot z+s\odot[-z_{h:},z_{:h}]$。
- `INTERLEAVE_HALF`：$y_i=c_i z_{2i}-s_i z_{2i+1}$，
  $y_{i+h}=s_{i+h}z_{2i}+c_{i+h}z_{2i+1}$，$0\le i<h$。

RoPE 的 cos/sin 按 head 广播。Cache 写入由 `cache_index[t]` 决定；负数跳过，
非负位置的 block/offset 分别为整除/取模 `block_size`。PA_BSND 和 PA_NZ
仅改变物理寻址；未命中位置保持原字节，不 clone Cache，也不 copy-back。
MX 投影结果中有限非零且绝对值小于 $2^{-126}$ 的次正规数置为正零；正常值、已有正负零、NaN 和 Inf 保持对应数值语义。E4M3FN 饱和转换将 NaN 转为零；HiF8 使用 SAT + ties-away，有限溢出及 Inf 饱和、NaN 转为零。BF16、FP8 转换点属于数值契约，Golden 不模拟硬件归约树。

以上全局 scale 公式用于非空输入；T=0 时跳过归约，HiF8 scale 返回 `(0,)`。

## 统一调用接口与执行方式

整网统一调用 `qmlaprolog.quant_mla_prolog(...)`，模块内部完成 Torch/Fake 注册及图分派。
执行方式、Native 加载配置、输入量化模式和 QKV 输出模式分别配置。

| 执行方式 | 调用入口 | 当前配置 |
| --- | --- | --- |
| Eager | `qmlaprolog.quant_mla_prolog` | 检查元信息及设备值，有 D2H 同步成本 |
| Native require 部署 | 同一公开函数 | 设置 `CANNBOTDSL_NATIVE_BINARY_MODE=require`；缺少变体应报错，可配合 Eager/图模式 |
| ACLGraph | 捕获预检后的同一公开函数 | 自动选择异步图入口；捕获/回放外保证设备值合法 |
| NPUGraph-EX | `torch.compile(qmlaprolog.quant_mla_prolog, backend="npugraph_ex", ...)` | 固定 Shape/地址，关闭输入 staging |

Torch namespace 为 `cannbotdsl_quant_mla_prolog`，schema 使用整数序列化枚举，
`Tensor(a!) kv_cache` 声明原地更新。Native 导出名为 `quant_mla_prolog`。
低层 `quant_mla_prolog_native` 是显式异步直调入口，调用方负责设备值预检；该名称不代表二进制安装成功。

### 公共输入准备（以下示例均复用）

从仓库根目录运行，先设置 `export PYTHONPATH="$PWD/samples/quant_mla_prolog${PYTHONPATH:+:$PYTHONPATH}"`。
准备匹配的 CANN、CANNBotDSL、Torch 和 torch_npu。设备初始化、输入构造和 NZ 转换在算子调用/捕获外完成。
示例使用 MXFP8 输入、mode 0、PA_BSND 和 T=3，四个输出槽位始终固定。

```python
import torch
import torch_npu
import quant_mla_prolog as qmlaprolog

torch.npu.set_device(0)
device = torch.device("npu:0")
T, dim, q_lora, N, Dn, Dr = 3, 1024, 1024, 2, 128, 64
generator = torch.Generator().manual_seed(17)


def fp8(shape):
    values = torch.randn(shape, generator=generator) * 0.1
    return values.to(torch.float8_e4m3fn).to(device)


def e8m0(shape):
    return torch.full(shape, 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).to(device)


inputs = dict(
    x=fp8((T, dim)),
    wqa=fp8((q_lora, dim)),
    wqb=fp8((N * (Dn + Dr), q_lora)),
    wkva=fp8((512 + Dr, dim)),
    wkb=(torch.randn(N, Dn, 512, generator=generator) * 0.02).bfloat16().to(device),
    descale_x=e8m0((T, dim // 32)),
    descale_wqa=e8m0((q_lora, dim // 64, 2)),
    descale_wqb=e8m0((N * (Dn + Dr), q_lora // 64, 2)),
    descale_wkva=e8m0((512 + Dr, dim // 64, 2)),
    norm_weight_qa=torch.ones(q_lora, dtype=torch.float32).to(device),
    norm_weight_kva=torch.ones(512, dtype=torch.float32).to(device),
    kv_cache=torch.zeros((2, 128, 1, 512 + Dr), dtype=torch.bfloat16).to(device),
    cache_index=torch.arange(T, dtype=torch.int64).to(device),
    quant_scale_kv=None,
    rope_sin=torch.zeros((T, Dr), dtype=torch.bfloat16).to(device),
    rope_cos=torch.ones((T, Dr), dtype=torch.bfloat16).to(device),
)
attrs = dict(
    norm_eps=1e-5,
    w_qr_quant_mode=qmlaprolog.WQrQuantMode.MXFP8,
    qkv_quant_mode=qmlaprolog.QkvQuantMode.NO_QUANT,
    cache_mode=qmlaprolog.CacheMode.PA_BSND,
    qr_scale=1.0,
    kvr_scale=1.0,
    rope_mode=qmlaprolog.RopeMode.INTERLEAVE_HALF,
    return_qr=False,
)
```

真实 NZ 接入需一起转换三路投影权重，并按“输入Tensor”规则排列对应 descale；不能仅 reshape 权重。

### Eager 调用

```python
qmlaprolog.quant_mla_prolog_precheck(**inputs, **attrs)
q, descale_q, qr, descale_qr = qmlaprolog.quant_mla_prolog(**inputs, **attrs)
torch.npu.synchronize()
assert q.shape == (T, N, 512 + Dr)
assert descale_q.shape == qr.shape == descale_qr.shape == (0,)
```

Cache 原地更新，不另返回。`quant_mla_prolog_precheck` 接收相同输入和属性，返回 `None`；不回读或校验 `cache_index` 数值。非空输入时仅回读启用的 `quant_scale_kv`，拒绝非有限值和负数，不启动融合 Kernel，也不修改 Cache。计算入口要求索引和 scale 已合法，不重复回读；T=0 或未启用 `quant_scale_kv` 时，预检仅校验元信息。

### Native require 部署（调用函数不变）

生产文件保留 `@cannbotdsl.aot.export("quant_mla_prolog")`，导出与运行共用完整融合程序和配置构造。
通过框架 AOT 构建接口收集该模块，不另维护算子专用打包脚本：

```python
from cannbotdsl.aot import build

build(
    collectors="samples/quant_mla_prolog",
    pattern="quant_mla_prolog.py",
    target="dav-3510",
    output="samples/quant_mla_prolog/_native",
    operators=["quant_mla_prolog"],
)
```

构建输出须为新目录，编译产物不提交。将生产文件与同级 `_native/` 一起部署；同名 Python 包也可将二进制置于包内 `_native`。
在新进程导入前设置：

```bash
export CANNBOTDSL_NATIVE_BINARY_MODE=require
export CANNBOTDSL_NATIVE_BINARY_REPORT=/absolute/results/native-report.json
export PYTHONPATH="/absolute/deployment-directory${PYTHONPATH:+:$PYTHONPATH}"
```

```python
import quant_mla_prolog as qmlaprolog

qmlaprolog.quant_mla_prolog_precheck(**inputs, **attrs)
q, descale_q, qr, descale_qr = qmlaprolog.quant_mla_prolog(**inputs, **attrs)
torch.npu.synchronize()
```

`require` 保留元信息校验，设备值仍由显式预检检查。`CANNBOTDSL_QMP_PROFILES` 可指定 JSON profile 文件，字段为
`T, dim, q_lora, N, Dn, Dr, w_qr_quant_mode, qkv_quant_mode, cache_mode, rope_mode, weight_format`；
不接受外部核数或 tiling 参数，未指定时使用源码中的默认合法组合。
默认导出不等于任意 Shape 全覆盖，部署前须核对实际规格，未命中时应报错。
构建、安装、二进制命中和图执行需分别验证；文档示例不代表当前提交完成了新一轮 wheel 验收。

### ACLGraph 捕获和回放

```python
# 图外预检并完成编译/准备，这次调用也会更新 Cache。
qmlaprolog.quant_mla_prolog_precheck(**inputs, **attrs)
prepared = qmlaprolog.quant_mla_prolog(**inputs, **attrs)
torch.npu.synchronize()
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    graph_outputs = qmlaprolog.quant_mla_prolog(**inputs, **attrs)
graph.replay()
torch.npu.synchronize()
q, descale_q, qr, descale_qr = graph_outputs
```

回放期间保持输入、Cache、输出和 graph 存活，地址及 Shape/dtype/format 不变。
输入内容可在图外更新；调用方须始终保证 index 合法，`quant_scale_kv` 改变后必须重新预检。更新输入后须建立流依赖，不能沿用首次合法性结论。
捕获及 replay 不回读设备值，模块不缓存“曾预检通过”状态。准备调用会写 Cache，若需要恢复初始数据应在捕获外完成。
更换 Shape、模式或地址时重新捕获；T=0 的空图不能证明非空图模式通过。

### NPUGraph-EX 调用

```python
qmlaprolog.quant_mla_prolog_precheck(**inputs, **attrs)
qmlaprolog.quant_mla_prolog(**inputs, **attrs)
torch.npu.synchronize()
compiled = torch.compile(
    qmlaprolog.quant_mla_prolog,
    backend="npugraph_ex",
    fullgraph=True,
    dynamic=False,
    options={"force_eager": False, "clone_input": False},
)
q, descale_q, qr, descale_qr = compiled(**inputs, **attrs)
torch.npu.synchronize()
```

也可编译 `forward` 中调用公开函数的整网 Module。固定权重建议注册为模型 buffer，所有输入地址及存储保持稳定。
`clone_input=False` 关闭后端输入 staging；调用方保证 index 合法，`quant_scale_kv` 更新后仍须图外预检。
仅描述固定 Shape、固定地址、无梯度推理契约，不宣称任意后端或动态 Shape 兼容。
后端不可用、图编译失败应报错，不将 Eager 或普通 ACLGraph 回退当作 EX 成功。

## 测试

```bash
python -m pytest -v test/quant_mla_prolog/test_quant_mla_prolog.py
```

仅收集以下 4 个 Case，每例调用目标算子一次：

| Case | 输入 / QKV 模式 | T / q_lora / N | 权重 / Cache | 分支 |
| --- | --- | --- | --- | --- |
| MXFP8 BF16 输出 | 1 / 0 | 1 / 1024 / 4 | ND / PA_BSND | 小 T head 交接，RoPE mode 1 |
| MXFP8 分组 Cache | 1 / 1 | 16 / 1536 / 4 | NZ / PA_BSND | 小 T head 交接，RoPE mode 1 |
| MXFP8 量化输出 | 1 / 2 | 33 / 2048 / 8 | NZ / PA_NZ | T=33 扩展 head 交接，RoPE mode 1 |
| HiF8 量化输出 | 2 / 3 | 3 / 1024 / 2 | ND / PA_NZ | HiF8 路径，RoPE mode 2 |

前三例在公开调用选取编译配置时断言实际启用了 head 交接，避免仅凭 Shape 推断分支覆盖。
测试的模块 fixture 另检查 `return_qr` 非 bool 值在 Torch 分派前被拒绝，并验证默认 False 和显式 True/False 的传递；不增加 Case 数。

每例检查四输出 Shape/dtype/ND Format、Cache 存储身份、未命中区域字节及数值。CPU Golden 和比较器位于同一个测试文件。Q 三方使用提供的 `bm_output` / `bm_output_gold` 独立计算结果，不使用普通 Golden 的量化前中间值。当前接入范围为 MXFP8 mode 2、rope_mode=1、Dr=64，qr_scale/kvr_scale 按传入值计算；其他范围保留二方验收。Cache、scale、QR 及 E8M0 检查不能由 Q 三方豁免。原始高精度分支的 QA descale dtype 分支行为保持不变，并在诊断中明确记录。

Q 和 KV Cache 各自为同一种 dtype 时，先比较合并后的完整输出，两方或三方任一通过即接受；合并两方、三方均失败后，再拆分 NoPE/KV 与 RoPE/KR。拆分后每个分支均须两方或三方至少一个通过，任一分支两者均失败则该输出失败。混合 dtype 的内容从一开始分别比较，不因使用同一种字节载体而合并。当前 mode 0/1 的 Q 为 BF16，mode 2 为 E4M3FN，mode 3 为 HiF8；mode 1 Cache 中的 FP8 KV、BF16 KR 和 FP32 scale 继续独立检查。其余输出比较规则及阈值不变，三方 warning 视为接受。

这 4 例是代表性回归，不等于全量模式和 Shape 覆盖，也不是性能采集脚本。历史全量 Case、性能原始数据和专项脚本保留在先前提交及独立实验目录。
