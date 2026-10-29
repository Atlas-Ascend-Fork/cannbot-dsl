# mla_prolog：单文件 CANNBot-DSL 实现

生产源码只有 [mla_prolog.py](mla_prolog.py)。非空计算为一个融合 MIX Kernel。

## 支持规格

| 参数 | 范围 |
| --- | --- |
| dim | 1024～8192，256 对齐 |
| q_lora | 640、1024、1536、2048 |
| kv_lora | 512 |
| N | 1～128 |
| (Dn,Dr) | (128,64)、(192,64)、(256,0) |
| 权重 | 三路投影权重统一 ND 或统一真实 NZ；wkb 为 ND |
| 输出模式 | 0/1/2/3；mode1 仅 PA_BSND，其余支持 PA_BSND/PA_NZ |
| Cache | `16 ≤ block_size ≤ 1024`，16 对齐；从 `kv_cache.shape[1]` 获取。负 index 跳过写入，重复有效 index 非法 |
| 空输入 | T=0 无主 Kernel 启动；返回固定三个 Tensor 槽位 |
| 核资源 | 按当前 stream 的可用 MIX 资源选择 `min(AIC数, AIV数 // 2)`；四种输出模式均不额外限制为32/36核，mode3不按 `T*N` 限制启动核数 |

核数不属于公开接口参数。调用方调整 stream 核资源配额后，后续调用重新读取配额；
没有可用 MIX 资源时抛出 `ValueError`，不启动 Kernel。

## 算子接口

`mla_prolog` 接收BF16激活和权重，支持非量化或量化的Q/Cache输出；不接收量化权重。
以下为公开Python接口，枚举通过同一个 `mlaprolog` 模块访问。

```python
import mla_prolog as mlaprolog

def mla_prolog(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    quant_scale_kv=None,
    rope_sin=None,
    rope_cos=None,
    *,
    norm_eps=1e-5,
    qkv_quant_mode=mlaprolog.QkvQuantMode.NO_QUANT,
    cache_mode=mlaprolog.CacheMode.PA_BSND,
    qr_scale=1.0,
    kvr_scale=1.0,
    rope_mode=mlaprolog.RopeMode.INTERLEAVE_HALF,
    return_qr=False,
):
    ...  # 固定返回 (q, descale_q, qr)，原地更新 kv_cache。
```

### 输入Tensor

`T>=0`，`block_number>0`；其他维度范围见“支持规格”。所有输入位于同一NPU，
满足连续性要求，不支持以非连续视图或隐式复制修复布局。

| 参数 | Torch逻辑Shape | Torch dtype | Format与用途 |
| --- | --- | --- | --- |
| `x` | `(T,dim)` | `torch.bfloat16` | ND，输入隐状态 |
| `wqa` | `(q_lora,dim)` | `torch.bfloat16` | ND或真实FRACTAL_NZ，QA投影 |
| `wqb` | `(N*(Dn+Dr),q_lora)` | `torch.bfloat16` | ND或真实FRACTAL_NZ，Q上投影 |
| `wkva` | `(512+Dr,dim)` | `torch.bfloat16` | ND或真实FRACTAL_NZ，KVA投影 |
| `wkb` | `(N,Dn,512)` | `torch.bfloat16` | ND，将Q-Nope吸收到512维latent空间，不用于生成Cache |
| `norm_weight_qa` | `(q_lora,)` | `torch.float32` | ND，QA RMSNorm gamma |
| `norm_weight_kva` | `(512,)` | `torch.float32` | ND，Ckv RMSNorm gamma |
| **`kv_cache`（原地输入输出）** | mode 0/2/3：`(block_number,block_size,1,512+Dr)`；mode 1：`(block_number,block_size,1,512+2*Dr+16)` | mode 0：`torch.bfloat16`；mode 1/2：`torch.float8_e4m3fn`；mode 3：`torch.uint8`承载HiF8字节 | `block_size` 为 16～1024 内的 16 的倍数。外部Tensor为ND载体，内部按 `cache_mode` 的PA_BSND/PA_NZ地址映射；mode 1仅PA_BSND |
| `cache_index` | `(T,)` | `torch.int64` | ND，token写入的线性Cache槽位；负值跳过写入 |
| `quant_scale_kv` | `(1,)` | `torch.float32` | ND，mode 2/3必填且有限正数；mode 0/1必须为 `None` |
| `rope_sin` | `(T,Dr)` | `torch.bfloat16` | ND，启用RoPE时必填 |
| `rope_cos` | `(T,Dr)` | `torch.bfloat16` | ND，与sin成对提供 |

**三路权重必须统一ND（format 2）或统一真实NZ（format 29）。**
真实NZ的Torch逻辑Shape仍是上表二维Shape；物理分块分别为
`(q_lora/16,dim/16,16,16)`、`(N*(Dn+Dr)/16,q_lora/16,16,16)`、
`((512+Dr)/16,dim/16,16,16)`。普通ND reshape成四维不合法。
权重FRACTAL_NZ与Cache的PA_NZ不是同一种格式，不可互相替代。

### 属性与枚举

| 属性 | 类型与合法值 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `norm_eps` | 浮点属性，内核类型FP32；须有限且严格大于0，拒绝bool | `1e-5` | QA和Ckv两路RMSNorm共用epsilon |
| `qkv_quant_mode` | `mlaprolog.QkvQuantMode` | `NO_QUANT` | 最终Q及Cache的量化模式 |
| `cache_mode` | `mlaprolog.CacheMode` | `PA_BSND` | Cache内部地址映射 |
| `qr_scale` | 转FP32后有限，允许负数和0，拒绝bool | `1.0` | QA RMSNorm后、WQB前缩放CQ；同时影响Q-Nope、Q-RoPE和qr |
| `kvr_scale` | 转FP32后有限，允许负数和0，拒绝bool | `1.0` | Ckv RMSNorm后缩放Ckv，**不缩放KR** |
| `rope_mode` | `mlaprolog.RopeMode` | `INTERLEAVE_HALF` | RoPE变换方式 |
| `return_qr` | 严格bool | `False` | 是否写出QA归一化并缩放后的qr，不改变主Q路径 |

公开 Python 入口在进入 Torch dispatcher 前拒绝三个浮点属性的 bool 和非数值类型。
`norm_eps` 通过 Python 数值属性传入，例如 `norm_eps=1e-5`，不是 Tensor 参数；内核接收 FP32 标量。
省略时使用 `1e-5`。Host 拒绝非有限值，以及超出 FP32 有限范围或转换为 FP32 后变为0的值，确保内核实际接收的 epsilon 有限且严格大于0。

| 枚举成员 | 内部编码 | 语义 |
| --- | ---: | --- |
| `QkvQuantMode.NO_QUANT` | 0 | Q、KV、KR均为BF16 |
| `QkvQuantMode.KV_FP8_PER_TOKEN_GROUP` | 1 | Q为BF16；KV按token/group动态量化为E4M3FN，group_size=128；KR为BF16 |
| `QkvQuantMode.Q_KV_FP8` | 2 | Q按token/head动态量化为E4M3FN；KV/KR共用输入per-tensor量化乘子 |
| `QkvQuantMode.Q_KV_HIF8` | 3 | Q跨全部token/head做全局per-tensor动态量化；KV/KR共用输入量化乘子，编码均为HiF8 |
| `CacheMode.PA_BSND` | 0 | 按block、token、head、dimension存放 |
| `CacheMode.PA_NZ` | 1 | 按本算子的PA_NZ分块地址映射存放；mode 1不支持 |
| `RopeMode.NONE` | 0 | 不做RoPE，Dr=64时Q-Rope与KR直通 |
| `RopeMode.INTERLEAVE_HALF` | 1 | 从交错偶/奇通道取对，输出为前后半区 |
| `RopeMode.HALF` | 2 | 从前后半区取对，输出仍为前后半区 |

公开Python接口只接收对应枚举成员，不接收上表的裸整数、字符串或bool。
`Dr=0` 必须显式指定 `RopeMode.NONE`；`Dr=64` 可使用三种RoPE模式。
未启用RoPE时sin/cos可同时为 `None`；若传入仍须成对满足上表Shape和dtype，但不参与计算。

### 输出与Cache原地更新

返回值始终为三个Tensor，不返回独立Cache Tensor：

| 输出/更新项 | Shape | Torch dtype | 语义 |
| --- | --- | --- | --- |
| `q`，槽位0 | `(T,N,512+Dr)` | mode 0/1：BF16；mode 2：E4M3FN；mode 3：uint8承载HiF8 | `[Q-Nope(512),Q-RoPE(Dr)]` |
| `descale_q`，槽位1 | mode 2：`(T,N,1)`，包括T=0；T>0且mode 3：`(1,)`；其余：`(0,)` | FP32 | Q反量化乘子，`q_real=Decode(q)*descale_q`；mode 3空输入不产生全局scale |
| `qr`，槽位2 | true：`(T,q_lora)`；false：`(0,)` | BF16 | QA RMSNorm后乘 `qr_scale` 的结果 |
| `kv_cache`，输入原地更新 | 与输入Shape完全一致 | 与输入dtype完全一致 | 仅更新非负index命中位置，未命中区域逐字节不变；data pointer和storage identity不变 |

mode 1每个token的payload按 `[KV FP8字节, KR BF16字节, 4个FP32 descale字节]` 拼接：
`Dr=64` 为656字节，`Dr=0` 为528字节。虽然载体dtype为E4M3FN，KR和descale部分不能按FP8数值解释。
mode 2/3的 `quant_scale_kv` 是**量化乘子**：`payload=Quantize(value*quant_scale_kv)`，
读取时 `value=Decode(payload)/quant_scale_kv`，KV与KR共用此乘子；它不是Q输出的descale。
HiF8的uint8仅承载编码，不表示UINT8数值量化。

### 计算与调用约束

- QA：`cq=RMSNorm(x@wqa.T)*qr_scale`，再经WQB投影；Q-Nope经WKB吸收，Q-Rope按RoPE模式处理，拼接后按输出模式量化。
- KVA：将 `x@wkva.T` 拆为Ckv和KR；仅Ckv执行RMSNorm及 `kvr_scale`，KR执行可选RoPE；之后量化并写Cache。
- `cache_index>=0` 的项必须小于 `block_number*block_size` 且互不重复；所有INT64负值均跳过Cache写入，允许重复负值。跳过写入的token仍参与Q、qr及Q动态量化计算。
- **T=0仍执行接口校验，在Host返回，不分配Kernel workspace、不启动DSL Kernel**；q为 `(0,N,512+Dr)`；mode 2的descale_q为 `(0,N,1)`，mode 0/1/3为 `(0,)`；qr按 `return_qr` 为 `(0,q_lora)` 或 `(0,)`。Cache的地址、storage identity、Shape、dtype及全部原始字节不变。
- 元信息和属性错误在调用层报错；Eager额外回读index/scale值进行预检。图捕获与replay不做值回读，必须在图外保证合法性，详见后续图模式示例。
- 不通过Cache clone、copy-back、TensorMove或隐式Format转换修复不合法输入；公开输出不支持自动求导。

## 计算公式

算子完成MLA前处理：计算Query的非位置编码部分与位置编码部分，生成KV/KR并原地更新Cache，
可选返回QA归一化结果。以下按[RMSNorm、Query与Key的公式说明方式](https://gitcode.com/cann/ops-transformer/blob/master/attention/mla_prolog_v3/README.md)组织，数学定义以本算子接口为准。

记 $x_t=\texttt{x}[t]$，$W_{qa},W_{qb},W_{kva},W_{kb}$ 分别对应四个同名权重输入。
$\gamma_q,\gamma_k$ 分别对应 `norm_weight_qa`、`norm_weight_kva`，
$\epsilon=\texttt{norm\_eps}$，$\alpha_q=\texttt{qr\_scale}$，$\alpha_{kv}=\texttt{kvr\_scale}$。
ND与NZ仅存储格式不同，数学定义相同。

下面逐项列出完整数学计算；公式中的变量是数学量，不是新增接口参数或Kernel节点。
$0\le t<T$、$0\le n<N$，$X=\texttt{x}\in\mathbb R^{T\times dim}$。
数学公式不指定硬件归约顺序。

### 1. 输入与QA下采样（第一个Matmul）

输入与 `wqa` 相乘，得到长度为 $q\_lora$ 的下采样结果：

$$
A^Q=XW_{qa}^{T}\in\mathbb R^{T\times q\_lora}.
$$

$$
a^Q_t=x_tW_{qa}^{T},\qquad
a^Q_{t,j}=\sum_{i=0}^{dim-1}x_{t,i}W_{qa,j,i},\quad 0\le j<q\_lora.
$$

这里 $a^Q_t=A^Q[t]$。权重的逻辑Shape为 $(q\_lora,dim)$，因此公式使用转置；
ND/NZ均执行这一投影，RMSNorm处理的是投影结果，不是原始输入 $x_t$。

### 2. QA平方均值、RMSNorm与尺度矫正

先沿 $q\_lora$ 维计算平方均值，加 `norm_eps` 后开方：

$$
m^Q_t=\frac{1}{q\_lora}\sum_{j=0}^{q\_lora-1}(a^Q_{t,j})^2,
\qquad r^Q_t=\sqrt{m^Q_t+\epsilon}.
$$

投影结果除以均方根，乘 `norm_weight_qa`，最后乘 `qr_scale`：

$$
\overline c^Q_{t,j}=\frac{a^Q_{t,j}}{r^Q_t}\gamma_{q,j},
\qquad c^Q_{t,j}=\alpha_q\overline c^Q_{t,j}.
$$

这就是 $c^Q_t=\alpha_q\operatorname{RMSNorm}(x_tW_{qa}^{T},\gamma_q,\epsilon)$ 的逐项展开。
其中 $\alpha_q$ 作用于后续所有Query分支的公共输入，不只作用于Q-RoPE。

### 3. QB上投影与head拆分

与 `wqb` 相乘，将最后一维按 $N\times(Dn+Dr)$ 解释：

$$
u^Q_{t,n,j}=\sum_{i=0}^{q\_lora-1}
c^Q_{t,i}W_{qb,n(Dn+Dr)+j,i},\qquad 0\le j<Dn+Dr.
$$

每个head的前 $Dn$ 维用于WKB吸收，后 $Dr$ 维用于RoPE：

$$
u^N_{t,n}=u^Q_{t,n,0:Dn},\qquad
u^R_{t,n}=u^Q_{t,n,Dn:Dn+Dr}.
$$

### 4. WKB吸收、Query RoPE与拼接

每个head使用对应的 `wkb[n]`，将非位置编码部分投影到512维：

$$
q^N_{t,n}=u^N_{t,n}W_{kb,n},\qquad
q^N_{t,n,k}=\sum_{j=0}^{Dn-1}u^N_{t,n,j}W_{kb,n,j,k},\quad 0\le k<512.
$$

位置编码部分执行所选RoPE，再与非位置编码部分拼接：

$$
q^R_{t,n}=\operatorname{RoPE}_t(u^R_{t,n}),\qquad
Q_{t,n}=\operatorname{Concat}(q^N_{t,n},q^R_{t,n})\in\mathbb R^{512+Dr}.
$$

这里 $Q$ 是输出编码前的Query；`wkb` 不参与Cache生成。

### 5. KVA下采样、拆分、归一化与Key RoPE

输入与 `wkva` 相乘，并拆出512维KV latent与 $Dr$ 维KR：

$$
a^{KV}_t=x_tW_{kva}^{T},\qquad
a^{KV}_{t,j}=\sum_{i=0}^{dim-1}x_{t,i}W_{kva,j,i},\quad 0\le j<512+Dr.
$$

$$
a^C_t=a^{KV}_{t,0:512},\qquad a^R_t=a^{KV}_{t,512:512+Dr}.
$$

仅KV latent沿512维计算平方均值，加epsilon、开方：

$$
m^{KV}_t=\frac{1}{512}\sum_{j=0}^{511}(a^C_{t,j})^2,
\qquad r^{KV}_t=\sqrt{m^{KV}_t+\epsilon}.
$$

归一化后乘 `norm_weight_kva`，再乘 `kvr_scale`；KR不参与该归一化或缩放：

$$
\overline c^{KV}_{t,j}=\frac{a^C_{t,j}}{r^{KV}_t}\gamma_{k,j},\qquad
k^C_{t,j}=\alpha_{kv}\overline c^{KV}_{t,j},\quad 0\le j<512.
$$

$$
k^R_t=\operatorname{RoPE}_t(a^R_t),\qquad
K_t=\operatorname{Concat}(k^C_t,k^R_t)\in\mathbb R^{512+Dr}.
$$

这里 $K$ 是Cache编码前的逻辑值。$\alpha_{kv}$ **只缩放KV latent，不缩放KR**；
两路RMSNorm共用 `norm_eps`，但分别使用各自的gamma。

### 6. RoPE计算公式

令 $H=Dr/2$，$s=\texttt{rope\_sin}[t]$，$c=\texttt{rope\_cos}[t]$，$0\le j<H$。
对输入向量 $z$，两种旋转模式分别为：

$$
\operatorname{RoPE}_{\mathrm{interleave\text{-}half}}(z)
=\left[(z_{2j}c_j-z_{2j+1}s_j)_{j=0}^{H-1},\;
(z_{2j+1}c_{H+j}+z_{2j}s_{H+j})_{j=0}^{H-1}\right].
$$

$$
\operatorname{RoPE}_{\mathrm{half}}(z)
=\left[(z_jc_j-z_{H+j}s_j)_{j=0}^{H-1},\;
(z_{H+j}c_{H+j}+z_js_{H+j})_{j=0}^{H-1}\right].
$$

`rope_mode=NONE` 时 $\operatorname{RoPE}(z)=z$；$Dr=0$ 时没有位置编码子段，且必须使用NONE。

Q的所有head与KR使用同一token的sin/cos。两种模式都包含逐元素cos乘法、sin乘法和加减运算，
但取对方式不同；不对传入的sin/cos额外计算三角函数。

### 7. 可选qr输出公式

`return_qr=True` 时，输出QA的归一化及尺度矫正结果：

$$
\texttt{qr}_{t}=\operatorname{BF16}(c^Q_t).
$$

`return_qr=False` 时返回BF16空Tensor；不改变主Query中的 $\alpha_q$ 计算。

### 8. Q、Cache与descale的输出计算

以下 $Q,K$ 为上述量化前的最终逻辑值，不表示Kernel中间Buffer。
$F8$、$H8$ 分别表示饱和E4M3FN、HiF8编码，最大有限值分别为448、32768；
F8采用最近偶数舍入，H8采用最近值且中点远离零。$B16$ 表示BF16转换。

对有限值 $v$，饱和与舍入可写为：

$$
\operatorname{clip}_{M}(v)=\min(M,\max(-M,v)),\qquad
F8(v)=\operatorname{Cast}^{\mathrm{RNE}}_{\mathrm{E4M3FN}}(\operatorname{clip}_{448}(v)),\qquad
H8(v)=\operatorname{Cast}^{\mathrm{RNA}}_{\mathrm{HiF8}}(\operatorname{clip}_{32768}(v)).
$$

Cast舍入到目标浮点格式的可表示值，不是先取整为整数；RNA表示中点远离零。

**mode 0：Q与Cache均非量化。**

$$
\texttt{q}=B16(Q),\qquad P_t=B16(K_t),\qquad
\texttt{descale\_q}=\operatorname{Empty}_{FP32}(0).
$$

**mode 1：Q非量化，KV按token/group量化，KR非量化。** 每组128个元素，$g=0,1,2,3$：

$$
A^{KV}_{t,g}=\max_{0\le j<128}|k^C_{t,128g+j}|,\qquad
d^{KV}_{t,g}=\frac{\max(A^{KV}_{t,g},10^{-4})}{448}.
$$

$$
\widehat{k}^{C}_{t,128g+j}=F8(k^C_{t,128g+j}/d^{KV}_{t,g}),\qquad
\widehat{k}^{R}_t=B16(k^R_t).
$$

$$
\texttt{q}=B16(Q),\qquad
\texttt{descale\_q}=\operatorname{Empty}_{FP32}(0),\qquad
P_t=\operatorname{ConcatBytes}
\left(\widehat{k}^{C}_t,\widehat{k}^{R}_t,FP32(d^{KV}_{t,0:4})\right).
$$

这里 $P_t$ 为待写入Cache的payload，字节长度为 $512+2Dr+4\times4$，
即Dr=64时656字节、Dr=0时528字节；只支持PA_BSND。

**mode 2：Q按token/head动态量化，KV与KR共用静态量化乘子。**

$$
A^Q_{t,n}=\max_{0\le k<512+Dr}|Q_{t,n,k}|,\qquad
d^Q_{t,n}=A^Q_{t,n}/448.
$$

$$
\texttt{q}_{t,n,k}=\begin{cases}
0,&A^Q_{t,n}=0,\\
F8(Q_{t,n,k}/d^Q_{t,n}),&A^Q_{t,n}>0,
\end{cases}\qquad
\texttt{descale\_q}_{t,n,0}=d^Q_{t,n}.
$$

$$
s_{KV}=\texttt{quant\_scale\_kv}[0],\qquad
P_t=\operatorname{Concat}\left(F8(k^C_t s_{KV}),F8(k^R_t s_{KV})\right).
$$

每个token/head独立统计整个512+Dr维Query，`descale_q`为 $(T,N,1)$；不能只统计Q-Nope。

**mode 3：Q跨全部token/head动态量化，KV与KR共用静态量化乘子。**

$$
A^Q=\max_{t,n,k}|Q_{t,n,k}|,\qquad d^Q=A^Q/32768.
$$

$$
\texttt{q}=\begin{cases}0,&A^Q=0,\\H8(Q/d^Q),&A^Q>0,\end{cases}
\qquad \texttt{descale\_q}_0=d^Q.
$$

$$
s_{KV}=\texttt{quant\_scale\_kv}[0],\qquad
P_t=\operatorname{Concat}\left(H8(k^C_t s_{KV}),H8(k^R_t s_{KV})\right).
$$

所有token/head共用一个Q的descale，Shape为 $(1,)$。HiF8以uint8保存编码，不按UINT8数值量化。
mode 2/3全零分支返回正零及descale=0，不执行除零；负Cache index的token仍参与Q统计。

量化Q的反量化关系如下，仅用于解释输出数值，不是额外的Kernel计算：

$$
Q_{\mathrm{dequant}}=\operatorname{Decode}(\texttt{q})\cdot\texttt{descale\_q}.
$$

mode 1的KV按对应group乘 $d^{KV}_{t,g}$ 反量化；mode 2/3的KV与KR按
$\operatorname{Decode}(P_t)/s_{KV}$ 反量化。Q的descale是输出乘子，Cache的scale是输入量化乘子，方向不同。

### 9. Cache原地写入与最终返回

令 $s_t=\texttt{cache\_index}[t]$，$B_s=\texttt{kv\_cache.shape[1]}$。对非负有效index，计算页号和页内位置：

$$
b_t=\left\lfloor\frac{s_t}{B_s}\right\rfloor,\qquad r_t=s_t\bmod B_s.
$$

用 $C[b,r]$ 表示PA_BSND或PA_NZ布局下的逻辑payload槽位，更新规则为：

$$
C_{\mathrm{out}}[b,r]=
\begin{cases}
P_t,&\exists t:s_t\ge0,\ (b,r)=(b_t,r_t),\\
C_{\mathrm{in}}[b,r],&\text{其他槽位}.
\end{cases}
$$

非负有效index必须唯一；负index不计算写入地址、不写Cache，但不跳过Q和qr计算。
PA_NZ仅改变物理存放顺序，不改变 $P_t$ 的逻辑内容。更新发生在同一个 `kv_cache` 存储中，
不新建Cache输出，不修改未命中字节。

最终固定返回：

$$
\left(\texttt{q},\ \texttt{descale\_q},\ \texttt{qr}\right),\qquad
\texttt{qr}=\begin{cases}B16(c^Q),&\texttt{return\_qr=True},\\
\operatorname{Empty}_{BF16}(0),&\texttt{return\_qr=False}.\end{cases}
$$

T=0时不执行以上投影、RMSNorm、缩放、RoPE、量化归约和Cache更新，在Host返回，不启动DSL Kernel：

$$
\operatorname{shape}(\texttt{q})=(0,N,512+Dr),\qquad
\operatorname{shape}(\texttt{descale\_q})=
\begin{cases}(0,N,1),&\texttt{qkv\_quant\_mode}=2,\\(0,),&\texttt{qkv\_quant\_mode}\in\{0,1,3\},\end{cases}
$$

$$
\operatorname{shape}(\texttt{qr})=
\begin{cases}(0,q\_lora),&\texttt{return\_qr=True},\\(0,),&\texttt{return\_qr=False}.\end{cases}
$$

空输出dtype仍按接口模式确定，$C_{\mathrm{out}}=C_{\mathrm{in}}$，全部Cache原始字节保持不变。

## 统一调用接口与执行方式

整网统一调用 `mlaprolog.mla_prolog(...)`，不需要按执行方式切换函数。
执行方式、Native二进制加载配置和 `qkv_quant_mode=0/1/2/3` 是三个独立概念。

| 执行方式 | 调用入口 | 当前配置 |
| --- | --- | --- |
| Eager | `mlaprolog.mla_prolog` | 检查元信息及设备值，有 D2H 同步成本 |
| Native require 部署 | `mlaprolog.mla_prolog` | 设置 `CANNBOTDSL_NATIVE_BINARY_MODE=require`，缺少编译变体应报错，不允许 JIT 回退；可配合Eager或图模式使用 |
| ACLGraph | 捕获预检后的 `mlaprolog.mla_prolog` | 自动选择异步图入口；源码和已安装 Native require 均可使用 |
| NPUGraph-EX | `torch.compile(mlaprolog.mla_prolog, backend="npugraph_ex", ...)` | 内部自动选择可入图的原地算子；设备值须在捕获/回放外预检 |

`torch.ops.cannbot_dsl_mla_prolog.mla_prolog` 的 schema 使用整数序列化三个枚举，
`Tensor(a!) kv_cache` 声明原地更新。公开 Python 接口使用 `CacheMode`、
`QkvQuantMode`、`RopeMode`，不是裸整数或字符串。

**公开 `mla_prolog` 在图外预检，手工捕获内自动走异步图入口。**
`index/quant_scale_kv` 的设备值检查包含 CPU 回读，不能放进捕获；低层带校验的
`torch.ops.cannbot_dsl_mla_prolog.mla_prolog` 仍拒绝直接捕获，整网应使用公开 Python 入口。
Fake 注册只描述输出，不提供自动求导。EX 内部注册使用相同参数、固定三个输出及
`Tensor(a!) kv_cache`，不引入自定义 clone/copy-back。图模式调用方必须在捕获前
检查值，并在每次回放前维持合法输入；图 replay 不重新执行 Python 值检查。
HiF8 Q/Cache 使用 uint8 原始字节载体，DSL 内是真正 HiF8 转换。

整网只需要一个算子模块 import。注册、整数 schema、编译态分派均由模块内部处理。
`torch_mla_prolog` 是公开函数 `mla_prolog` 的兼容别名，不是另一套实现。
Eager、Native require部署、ACLGraph、NPUGraph-EX 均使用此公开函数。
底层异步直调由内部执行链和测试工具使用，不作为整网接入示例。

### 公共输入准备（以下示例均复用）

从仓库根目录运行，先设置 `export PYTHONPATH="$PWD/samples/mla_prolog${PYTHONPATH:+:$PYTHONPATH}"`。设备初始化、数据构造和真实 NZ 转换均在算子调用/图捕获之外。
示例使用 mode 0、PA_BSND、T=3，三个输出槽位始终固定。

```python
import torch
import torch_npu
import mla_prolog as mlaprolog

torch.npu.set_device(0)  # ASCEND_RT_VISIBLE_DEVICES 映射后的逻辑卡号
device = torch.device("npu:0")
T, dim, q_lora, N, Dn, Dr = 3, 1024, 640, 2, 128, 64
generator = torch.Generator().manual_seed(17)

def bf16(shape, scale):
    return (torch.randn(shape, generator=generator) * scale).bfloat16().to(device)

index_cpu = torch.arange(T, dtype=torch.int64)
inputs = dict(
    x=bf16((T, dim), 0.25),
    wqa=bf16((q_lora, dim), dim ** -0.5),
    wqb=bf16((N * (Dn + Dr), q_lora), q_lora ** -0.5),
    wkva=bf16((512 + Dr, dim), dim ** -0.5),
    wkb=bf16((N, Dn, 512), Dn ** -0.5),
    norm_weight_qa=torch.ones(q_lora, dtype=torch.float32).to(device),
    norm_weight_kva=torch.ones(512, dtype=torch.float32).to(device),
    kv_cache=torch.zeros((2, 128, 1, 512 + Dr), dtype=torch.bfloat16).to(device),
    cache_index=index_cpu.to(device), quant_scale_kv=None,
    rope_sin=torch.zeros((T, Dr), dtype=torch.bfloat16).to(device),
    rope_cos=torch.ones((T, Dr), dtype=torch.bfloat16).to(device),
)
attrs = dict(
    norm_eps=1e-6, qkv_quant_mode=mlaprolog.QkvQuantMode.NO_QUANT,
    cache_mode=mlaprolog.CacheMode.PA_BSND, qr_scale=1.0, kvr_scale=1.0,
    rope_mode=mlaprolog.RopeMode.INTERLEAVE_HALF, return_qr=False,
)
# 如需真实 NZ，三路必须一起转换；不能仅 reshape 成四维。
# for name in ("wqa", "wqb", "wkva"):
#     inputs[name] = torch_npu.npu_format_cast(inputs[name], 29)
```

### Eager 调用

```python
q, descale_q, qr = mlaprolog.mla_prolog(**inputs, **attrs)
torch.npu.synchronize()
assert q.shape == (T, N, 512 + Dr)
assert descale_q.shape == qr.shape == (0,)
```

Cache 是原地输入输出，不另返回 Cache。Eager 检查 scale 有限且大于零、有效 index
不越界且不重复；所有负 index 跳过 Cache 写入。错误在主 Kernel 启动前抛出。

### Native require 部署（调用函数不变）

Native 使用源码中的 `@export("mla_prolog")` 和框架的 AOT 构建接口，
不另外维护算子专用打包脚本。下例在已加载 CANN 9.2、激活 0.8.0 环境后从仓库根目录执行：

```python
from cannbotdsl.aot import build

build(
    collectors="samples/mla_prolog",
    pattern="mla_prolog.py",
    target="dav-3510",
    output="samples/mla_prolog/_native",
    operators=["mla_prolog"],
)
```

构建输出目录须为新的目标目录；`_native` 编译产物不提交到源码仓库。
将 `mla_prolog.py` 与同级 `_native/` 一起部署。源码导入时自动注册同级 Native 资源，
在新进程导入前设置：

```bash
export CANNBOTDSL_NATIVE_BINARY_MODE=require
export CANNBOTDSL_NATIVE_BINARY_REPORT=/absolute/results/native-report.json
export PYTHONPATH="/absolute/deployment-directory${PYTHONPATH:+:$PYTHONPATH}"
```

```python
import mla_prolog as mlaprolog

q, descale_q, qr = mlaprolog.mla_prolog(**inputs, **attrs)
torch.npu.synchronize()
```

`require` 只控制预编译加载，不关闭 Eager 校验，也不改变调用函数。
构建清单不是任意 Shape 的全规格包；必须包含线上需要的 T 分桶、mode、RoPE、QR、
WQB 列分块和 NZ 静态 Shape。可在构建前设置
`MLA_PROLOG_NATIVE_NZ_PROFILES='[[1024,640,2,128,64]]'` 指定 NZ 的
`[dim,q_lora,N,Dn,Dr]` 列表；ND 导出与该列表独立。
未命中时 `require` 必须报错，不得把 JIT 回退误认为 Native 命中。
ACLGraph 和 NPUGraph-EX 复用后续示例，仍须图外预检。

### ACLGraph 捕获和回放

```python
# 图外调用：完整预检并编译/准备，也会更新命中 Cache。
prepared = mlaprolog.mla_prolog(**inputs, **attrs)
torch.npu.synchronize()
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    graph_outputs = mlaprolog.mla_prolog(**inputs, **attrs)  # 自动走异步图路径
graph.replay()
torch.npu.synchronize()
q, descale_q, qr = graph_outputs
```

回放期间保持输入 Tensor、Cache、输出和 graph 存活，地址及 shape/dtype/format 不变。
可以在回放外更新同一地址的输入内容；更新后必须重新保证 index/scale 合法，并做好流同步。
捕获内及 replay 不检查设备值，模块不缓存“曾预检通过”的状态，也不自动证明调用方已预检。
调用方必须履行图外预检合同；不要把未知是否合法的 index/scale 直接交给捕获或回放。
更换 Shape、模式或地址时重新捕获。捕获与 replay 内不做 Cache clone/copy-back/TensorMove。
上面的准备调用会更新命中 Cache，若测试需要初始 sentinel，应在捕获外恢复。
`T=0` 不启动主 Kernel，不用空图来证明非空图模式已通过。

### NPUGraph-EX 调用

```python
# 首次 Eager 调用在图外预检并准备编译产物；会更新命中 Cache。
mlaprolog.mla_prolog(**inputs, **attrs)
torch.npu.synchronize()
compiled = torch.compile(
    mlaprolog.mla_prolog,
    backend="npugraph_ex",
    fullgraph=True,
    dynamic=False,
    options={"force_eager": False, "clone_input": False},
)
q, descale_q, qr = compiled(**inputs, **attrs)
torch.npu.synchronize()
```

也可以编译 `forward` 中调用 `mlaprolog.mla_prolog` 的整网 Module。
固定权重建议注册为模型 buffer；所有输入地址及存储保持稳定。
`clone_input=False` 避免后端输入 staging 的 TensorMove；不能省略后再引用无额外搬运结论。
新数据在图外写入原地址；尤其 index/scale 改变后必须重新预检，不能沿用首次合法性结论。
只保证本节固定 Shape、固定地址、无梯度推理契约，不宣称动态 Shape 或任意后端兼容。
后端不可用、图编译失败应报错，不回退为 Eager 或普通 ACLGraph 的“EX 成功”。
