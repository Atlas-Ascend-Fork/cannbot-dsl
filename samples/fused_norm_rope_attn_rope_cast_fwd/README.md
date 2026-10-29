# fused_norm_rope_attn_rope_cast_fwd

基于 CANNBot-DSL 的 Prefill 融合算子，将可选的 Q RMSNorm、Q RoPE、Sparse Attention、O 共轭 RoPE 和 MXFP8 量化（含 SF 打包）组合为一次接口调用。

## 算子介绍

计算流程如下：

```text
BF16 Q → Q RoPE → Sparse Attention → O 共轭 RoPE → MXFP8 输出与打包 scale
                        ↑
            可选的逐 head Q RMSNorm 系数
```

- **Q RMSNorm**：从旋转前的 Q 计算 `rsqrt(mean(Q²) + rms_norm_eps)`，将系数乘到 Attention score 上。关闭时不计算该系数。
- **RoPE**：Q 的最后 64 维进行相邻偶奇配对旋转；Attention 输出的最后 64 维进行反向旋转。
- **Sparse Attention**：每个 Query 根据 `indices` 选择 KV，K 和 V 共用同一张量。QK、Softmax 状态和输出累加使用 FP32；Softmax 指数权重转 BF16 后参与 PV，归一化分母保持 FP32。
- **MXFP8 量化**：FP32 Attention 输出经反向旋转后，每 32 个元素共用一个 scale，量化为 E4M3FN；每 8 个 head 组成一个输出组。

| 特性与约束 | 说明 |
| :--- | :--- |
| 支持架构 | NPU ARCH 3510（Ascend 950） |
| 计算场景 | Prefill |
| 输入类型 | Q/KV 为 BF16；索引和位置为 INT32 |
| Attention 维度 | Q head 数为 64 或 128；KV head 数为 1；Q/K/V head dim 均为 512 |
| RoPE | interleave 形式，旋转最后 64 维 |
| 稀疏索引 | 支持非连续、重复和非法索引；支持每个 Query 单独指定有效长度 |
| 量化格式 | E4M3FN 数据、UE8M0 scale，每个 INT32 打包 4 个 scale byte |

实现见 [fused_norm_rope_attn_rope_cast_fwd.py](fused_norm_rope_attn_rope_cast_fwd.py)。

## 快速开始

按照 [仓库 README](../../README.md) 安装依赖并加载 CANN 环境，在仓库根目录运行以下代码：

```python
import torch
import torch_npu

from samples.fused_norm_rope_attn_rope_cast_fwd.fused_norm_rope_attn_rope_cast_fwd import fused_norm_rope_attn_rope_cast_fwd

torch.npu.set_device(0)
S, T, H, K = 4, 32, 64, 32
device = "npu:0"

# Q 需要转换为接口约定的 16 维块交错布局。
logical_q = (torch.randn(S, H, 512, device=device) * 0.1).bfloat16()
q = logical_q.reshape(S, H, 32, 16).permute(0, 2, 1, 3).contiguous().reshape(S, H, 512)
kv = (torch.randn(T, 1, 512, device=device) * 0.1).bfloat16()
indices = torch.randint(T, (S, 1, K), dtype=torch.int32, device=device)
positions = torch.arange(S, dtype=torch.int32, device=device)
theta = torch.randn(S, 32, device=device)
cache = torch.cat((theta.cos(), theta.sin()), dim=-1)

# 全部 18 个参数均须按位置传入；不用 sink/有效长度时传 None。
out_fp8, out_sf, max_logits, lse = fused_norm_rope_attn_rope_cast_fwd(
    q,
    kv,
    indices,
    512**-0.5,  # sm_scale
    512,  # d_v
    None,  # attn_sink
    None,  # topk_length：使用全部 K 个索引
    True,  # enable_q_norm
    1e-6,  # rms_norm_eps
    positions,  # token_positions
    False,  # is_rope_neox_style
    64,  # rope_dim
    cache,  # cos_sin_cache
    H // 8,  # n_wv_group
    32,  # num_per_channels
    True,  # use_tma_aligned_col_major_sf
    True,  # round_sf
    True,  # use_packed_ue8m0
)
torch.npu.synchronize()
print(out_fp8.shape, out_sf.shape, max_logits.shape, lse.shape)
```

## 接口说明

函数名为 `fused_norm_rope_attn_rope_cast_fwd`。参数顺序与上面的示例一致，共 18 个参数，全部为位置参数，没有默认值。

记 `S` 为 Query token 数，`T` 为 KV token 数，`H` 为 Query head 数，`K` 为每个 Query 的索引容量，`P` 为位置表长度。

| 参数 | 类型 / shape | 含义或要求 |
| :--- | :--- | :--- |
| `q` | BF16 `[S,H,512]` | 16 维块交错布局，构造方法见示例 |
| `kv` | BF16 `[T,1,512]` | K 和 V 共用的输入 |
| `indices` | INT32 `[S,1,K]` | KV token 索引 |
| `sm_scale` | float | 有限的 score 缩放系数，支持正数、零和负数 |
| `d_v` | int | 固定传 `512` |
| `attn_sink` | FP32 `[H]` 或 `None` | 只影响 Attention 输出的分母 |
| `topk_length` | INT32 `[S]` 或 `None` | 每行使用的索引前缀长度；`None` 表示全部 K 项 |
| `enable_q_norm` | bool | 是否开启 Q RMSNorm |
| `rms_norm_eps` | float | RMSNorm epsilon，要求有限且非负 |
| `token_positions` | INT32 `[S]` | Query 位置，取值须在 `[0,P)` 内，可重复 |
| `is_rope_neox_style` | bool | 固定传 `False` |
| `rope_dim` | int | 固定传 `64` |
| `cos_sin_cache` | FP32 `[P,64]` | 前 32 项为 cos，后 32 项为 sin |
| `n_wv_group` | int | 固定传 `H // 8` |
| `num_per_channels` | int | 固定传 `32` |
| `use_tma_aligned_col_major_sf` | bool | 固定传 `True` |
| `round_sf` | bool | 固定传 `True` |
| `use_packed_ue8m0` | bool | 固定传 `True` |

### 输入布局与索引规则

Q 的逻辑元素 `(s,h,d)` 在该 token 内的物理偏移为 `(d // 16) * H * 16 + h * 16 + d % 16`。普通连续的 `[S,H,512]` 不能直接作为 Q 使用；模型已提供上述布局时可直接传入。

所有输入张量必须位于同一 NPU。Q 的 head/dim stride 为 `(512,1)`；KV 和 indices 的最后一维连续，三者均支持行间 padding 和 storage offset。位置、cos/sin 表、sink 和有效长度张量须连续。

每行只使用 `indices[s,0,:topk_length[s]]`；调用方须保证 `0 <= topk_length[s] <= K`。负数及大于等于 T 的索引被掩码，重复索引重复参与计算。K 不要求 64 对齐，也支持大于 640。

### 返回值

记 `G=H//8`，`A=ceil(S/4)*4`，返回顺序如下：

| 返回值 | shape | dtype / 布局 |
| :--- | :--- | :--- |
| `out_fp8` | `[S,G,4096]` | `torch.float8_e4m3fn`，连续 |
| `out_sf` | `[S,G,32]` | `torch.int32`，元素 stride 为 `(1,32*A,A)` |
| `max_logits` | `[S,H]` | FP32，连续，不包含 sink |
| `lse` | `[S,H]` | FP32，连续，不包含 sink |

FP8 每组内按 `[16 个维度块, 8 个 head, 32 个元素]` 排列，不能直接 reshape 为逻辑 `[S,H,512]`。SF 与该布局对应，每个 INT32 按小端顺序打包 4 个 UE8M0 byte。

无有效 KV 的行输出为零，`max_logits=-inf`、`lse=+inf`；每个打包 SF 为 `0x69696969`。输入保持只读，每次调用独立分配输出；`S=0` 时返回空输出。

### 片上内存

UB 容量通过 `dsl.get_mem_size("ub")` 获取，表示编译目标的每核容量上限。接口返回非正数或布局超出容量时直接报错。Attention 使用四组 UINT8 有效性标记保存索引掩码；原始索引保持 INT32，Gather 地址和跨度保持 UINT64。

| 阶段 | UB 布局占用 |
| :--- | ---: |
| Attention，无 sink | 251920 字节 |
| Attention，有 sink | 252048 字节 |
| 输出后处理双缓冲 | 172032 字节 |

上述布局均位于当前目标接口提供的 253952 字节（248 KiB）容量内。

## 精度测试

测试文件为 [test_fused_norm_rope_attn_rope_cast_fwd.py](../../test/fused_norm_rope_attn_rope_cast_fwd/test_fused_norm_rope_attn_rope_cast_fwd.py)。在仓库根目录运行：

```bash
python -m pytest test/fused_norm_rope_attn_rope_cast_fwd -v
```

当前保留 **1 个小 shape 用例**：`S=4、T=32、H=64、K=32、D=512`，开启 Q RMSNorm 和 sink。用例包含不同有效长度、空行、重复和非法索引，检查四项输出的精度与布局、输入只读及重复调用一致性。

Q/KV 使用 `randn * 0.1` 生成，其中 0.1 控制输入数值幅度，不是算子参数。CPU golden 按 64-token 分块，并模拟概率权重转 BF16、FP32 PV 累加和 online Softmax 的状态更新。

实际 FP8 数据与 SF 反量化后，与 golden 量化前的 FP32 输出比较；统计量直接与 golden 比较：

| 检查项 | 绝对误差门限 | 相对误差门限 | cosine 差异门限 |
| :--- | :--- | :--- | :--- |
| 反量化输出 | `1.1e-3` | `1.01/8` | `1e-3` |
| max_logits / lse | `1e-5` | `4.01/65536` | `1e-7` |

每个有限元素须满足绝对误差或相对误差门限之一，不允许超差点；相对误差为 `abs(actual-reference)/(abs(reference)+1e-6)`。同时检查整体 cosine 差异及非有限值是否一致。

该用例用于小规模功能回归，不代表所有 shape 和输入幅度均满足门限。CPU 数学运算也不是 NPU 指令的逐位仿真。

## 性能与运行说明

当前仅对 `S=T=8192、H=64、K=128/640` 且 `sm_scale>0` 的场景进行了性能优化和泛化验证，执行方式如下：

| 场景 | 执行方式 |
| :--- | :--- |
| 上述优化规格 | 两个 kernel：Q Norm / Q RoPE / Attention → FP32 工作区 → O 共轭 RoPE / MXFP8 / SF 打包 |
| 其他支持规格 | 一个 AIC/AIV MIX kernel 完成全部计算 |

两阶段路径使用 `[8192,64,512]` 的 FP32 工作区，大小为 **1 GiB**；评估设备耗时须包含两个 kernel 及工作区读写。路径仅根据形状和标量参数选择，小 shape 精度测试不衡量该路径的性能。

核数由平台接口实时查询。算子在当前 NPU stream 上运行；首次调用会编译，后续相同规格复用编译缓存。图捕获前需预热相同规格，跨 stream 的输入依赖由调用方建立。

## 许可证

本样例使用仓库的 [CANN Open Software License Agreement Version 2.0](../../LICENSE)。
