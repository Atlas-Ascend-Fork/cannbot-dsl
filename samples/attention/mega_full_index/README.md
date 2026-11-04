# mega_full_index

基于 CANNBotDSL 实现的 DeepSeek V4.1 第 20 层 Indexer 融合算子，面向 Ascend 950
（ARCH 3510）。一次 MIX kernel 启动完成 Indexer Prologue QW、Indexer Prologue K
和 Quant Lightning Indexer（QLI）：生成量化 Q 和 head 权重，更新 K cache，
计算 token 分数并输出 Top-512 索引及候选块。入口为 `mega_full_index()`，设备 profiling 名称同为 `mega_full_index`。

## 产品支持情况

- Ascend 950：支持。

## 算子介绍

### 模型规格与支持范围

| 项目 | 约定 |
| :--- | :--- |
| Batch / 序列长度 | `B=1`，物理 query 行数 `T=S=6`、`8` 或 `16` |
| 模型维度 | hidden size 为 5120，Q LoRA rank 为 1280，K latent 为 512 |
| Indexer heads | 32 个 Q head，共享一个 K head，每个 head 为 128 维 |
| RoPE | Q/K 最后 64 维，相邻偶奇维配对 |
| KV 长度 | 128-token 分页；逻辑 KV 长度支持 `1 ~ 1048576`，物理容量按 128 向上取整 |
| 量化 | MXFP4，E2M1 数据与 E8M0 scale，每 32 维共享一个 scale |
| 稀疏输出 | Top-512 token；最多 2048 个候选，每个候选覆盖 8 token |
| 核数调度 | 运行时查询当前 stream 的 AIC/AIV 资源；已验证 28/32 AIC |
| Metadata | 仅保留输入槽位，不读取内容，无需调用 QLI Metadata 算子 |

`T` 表示输入的物理行数，`Qv=seqused_q[0]` 表示有效 query 数，
`L=seqused_k[0]` 表示包含本块新 token 的有效 KV 长度。
有效 query 位于前 `Qv` 行，其余行输出无效索引。`block_table` 的宽度决定逻辑容量，
有效长度不得超过该容量；物理 cache 页数可以与逻辑页数不同。

### 计算流程

令 `BF16()` 表示舍入为 BF16，`Quant4()` 表示 block-32 MXFP4 量化：

```text
Q0 = MXMatmul(qr, wqb.T, descale_qr, descale_wqb)     # FP32
Q  = Quant4(RoPE_adjacent(Q0))                       # FP32 RoPE → MXFP4
W  = (x @ ww.T) / 64                                # FP32 输出

Z  = BF16(latent @ wk.T)                            # FP32 累加 → BF16
N  = BF16(Z.float() * rsqrt(mean(Z.float()²) + eps) * norm_weight)
K  = Quant4(BF16(RoPE_adjacent(N)))                  # 写入分页 cache
```

Q/K 的 RoPE 只作用于尾部 64 维，按相邻元素计算：

```text
even' = even * cos - odd * sin
odd'  = odd  * cos + even * sin
```

调用方提供按绝对位置准备的 sin/cos；相邻两列存放同一角度的系数。
Q 的 RoPE 在 FP32 中计算；K 的 Norm、RoPE 分别在输出处舍入为 BF16。
Norm 沿每个 token 的 128 维计算均方值，epsilon 固定为 `1e-20`。

FP4 采用 **nearest-even**：精确中点选择偶数编码，保留负零。
每组 `amax` 的下限为 `6 × 2^-126`，scale 向上取二次幂，
E2M1 数据每字节打包两维，低 nibble 对应偶数维。

Q 使用相邻维 RoPE，与 rotate-half 的配对方式不同；切换实现时应核对数值语义。

QLI 对解量化后的 Q/K 计算点积，点积输出舍入为 BF16。
head 权重先转为 BF16，各 head 的乘法和累加也分别舍入为 BF16：

```text
dot[h, j] = BF16(Q[h] @ K[j].T)
product   = BF16(ReLU(dot[h, j]) * BF16(W[h]))
score[j]  = BF16(score[j] + product)       # 按 head 0…31 顺序累加
```

第 `i` 个有效 query 的可见 token 数为：

```text
visible(i) = clamp(L - Qv + i + 1, 0, L)
```

不可见位置按负无穷参与选择。输出分数最高的最多 512 个 token，再按 token
位置升序发布，不足部分填 `-1`。候选以每 8-token 块的最大分数选择，始终包含
最新可见块；可见块不超过 2048 时输出全部块，超过时执行 Top-2048 选择。
候选索引是逻辑块号，不承诺按分数或位置排序。

融合边界不包含 Compressor、QR 生成与 FP8 量化、权重 NZ 准备或 RoPE 系数生成，
这些数据由网络上游提供。本算子不提供反向计算。

## Python 接口

```python
from samples.attention.mega_full_index.mega_full_index import (
    mega_full_index,
    prepare_mega_full_index,
)

def mega_full_index(
    x, qr, wqb, ww, descale_qr, descale_wqb, q_rope_sin, q_rope_cos,
    latent, wk, norm_weight, k_rope_sin, k_rope_cos,
    k_cache, k_scale_cache, cache_index,
    cu_seqlens_q=None, cu_seqlens_k=None,
    seqused_q=None, seqused_k=None, cmp_residual_k=None,
    block_table=None, output_idx_offset=None, metadata=None,
    *,
    softmax_scale=1.0 / 64.0,
    norm_eps=1e-20,
    storage_mode=1,
    topk=512,
    quant_mode=1,
    max_seqlen_q=-1,
    mask_mode=3,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="PA_BBND",
    return_value=False,
    candidate_topk_blocks=2048,
    candidate_block_size=8,
    q=None,
    descale_q=None,
    w=None,
): ...
```

`mega_full_index()` 准备工作区并执行，返回 8 项 Tensor。
`prepare_mega_full_index()` 的参数顺序相同，但 `softmax_scale`、`norm_eps` 是必填关键字参数；
返回持有输入、输出和工作区的准备对象，调用其 `run()` 执行并返回相同顺序的输出。
重复调用时建议复用准备对象，避免反复分配工作区。

### 输入 Tensor

共有 24 个外部 Tensor 槽位，可选槽位传 `None`。所有非空 Tensor 位于同一 NPU，
除三项 NZ 权重和明确允许页间 padding 的 cache 外，使用连续 ND 布局。
E8M0 以 UINT8 或等价 E8M0 字节载体传入。

| # | 参数 | shape | dtype / format | 含义与要求 |
| ---: | :--- | :--- | :--- | :--- |
| 1 | `x` | `[T,5120]` | BF16 / ND | W 投影输入 |
| 2 | `qr` | `[T,1280]` | FP8 E4M3FN 或 UINT8 / ND | 已量化的 Q LoRA 输入 |
| 3 | `wqb` | `[4096,1280]` | FP8 E4M3FN 或 UINT8 / FRACTAL_NZ | Q 投影权重，4096=32×128 |
| 4 | `ww` | `[32,5120]` | BF16 / FRACTAL_NZ | head 权重投影矩阵 |
| 5 | `descale_qr` | `[T,20,2]` | E8M0 字节 / ND | `qr` 的 block-32 scale |
| 6 | `descale_wqb` | `[4096,20,2]` | E8M0 字节 / ND | `wqb` 的 block-32 scale |
| 7 | `q_rope_sin` | `[T,64]` | FP32 / ND | Q 尾部 RoPE 正弦系数 |
| 8 | `q_rope_cos` | `[T,64]` | FP32 / ND | Q 尾部 RoPE 余弦系数 |
| 9 | `latent` | `[T,512]` | BF16 / ND | Compressor 提供的 K latent |
| 10 | `wk` | `[128,512]` | BF16 / FRACTAL_NZ | K 投影权重 |
| 11 | `norm_weight` | `[128]` | FP32 / ND | K RMSNorm 权重 |
| 12 | `k_rope_sin` | `[T,64]` | FP32 / ND | K 尾部 RoPE 正弦系数 |
| 13 | `k_rope_cos` | `[T,64]` | FP32 / ND | K 尾部 RoPE 余弦系数 |
| 14 | `k_cache` | mode 1：`[P,16,544]`；mode 0：`[P,128,1,64]` | UINT8 / ND | 含历史 K，原地更新；`P` 为物理页数 |
| 15 | `k_scale_cache` | mode 0：`[P,128,1,4]` | UINT8 / ND 或 `None` | mode 0 必须提供；mode 1 必须为 `None` |
| 16 | `cache_index` | `[T]` | INT64 / ND | 物理 token slot；`-1` 跳过写入 |
| 17 | `cu_seqlens_q` | `[2]` | INT32 / ND | 必须提供，B=1 时为 `[0,T]` |
| 18 | `cu_seqlens_k` | — | `None` | 当前 PA 场景传 `None` |
| 19 | `seqused_q` | `[1]` | INT32 / ND | 必须提供，有效 query 数 `Qv` |
| 20 | `seqused_k` | `[1]` | INT32 / ND | 必须提供，有效 KV 长度 `L` |
| 21 | `cmp_residual_k` | — | `None` | 当前仅支持 ratio 1 |
| 22 | `block_table` | `[1,max_pages]` | INT32 / ND | 必须提供，逻辑页到物理页的映射 |
| 23 | `output_idx_offset` | `[T,1]` | INT32 / ND 或 `None` | 给有效 TopK token 索引加偏移；不改变 candidate 块号 |
| 24 | `metadata` | `[1024]` | INT32 / ND 或 `None` | 连续占位 Tensor，不读取其内容 |

权重应在加载时准备为 FRACTAL_NZ，包装层不做隐式量化、format cast 或输入复制。
`metadata=None` 时准备层分配占位 Tensor，不启动 Metadata 计算。

### 属性与预分配输出

| 属性 | 默认值 | 当前支持 |
| :--- | :--- | :--- |
| `softmax_scale` | `1/64` | 固定为 `1/64` |
| `norm_eps` | `1e-20` | 固定为 `1e-20` |
| `storage_mode` | `1` | `0`：分离 cache；`1`：combined cache |
| `topk` | `512` | 固定为 512 |
| `quant_mode` | `1` | MXFP4 |
| `mask_mode` | `3` | right-bottom causal |
| `cmp_ratio` | `1` | 不压缩 |
| `layout_q` / `layout_k` | `"TND"` / `"PA_BBND"` | 固定为上述布局 |
| `max_seqlen_q` | `-1` | 通常保留默认值，有效行数由 `seqused_q` 给出 |
| `return_value` | `False` | 是否返回 TopK 对应分数 |
| `candidate_topk_blocks` | `2048` | 固定为 2048 |
| `candidate_block_size` | `8` | 固定为 8 |

`q`、`descale_q`、`w` 默认为 `None`，可传入与对应输出相同 shape、dtype 的
连续 NPU Tensor。传入后直接写入该 buffer，无需预清零，返回值保留其对象别名。

### 返回值

```text
(q, descale_q, w, k_cache,
 indices, values, candidate_indices, candidate_length)
```

| 输出 | shape | dtype | 说明 |
| :--- | :--- | :--- | :--- |
| `q` | `[T,32,64]` | UINT8 | packed E2M1 Q |
| `descale_q` | `[T,32,2,2]` | UINT8 | 每个 head 的 4 个 E8M0 scale |
| `w` | `[T,32]` | FP32 | 已乘 `1/64` 的 head 权重 |
| `k_cache` | 与输入相同 | UINT8 | 原地更新，返回输入的同一 Tensor 对象 |
| `indices` | `[T,1,512]` | INT32 | TopK token 索引，按位置升序，不足填 `-1` |
| `values` | `[T,1,512]` 或 `[0]` | BF16 | `return_value=True` 时返回对应分数，无效项为负无穷；否则为空 |
| `candidate_indices` | `[T,1,2048]` | INT32 | 逻辑 8-token 块号，有效前缀外填 `-1` |
| `candidate_length` | `[T,1]` | INT32 | 各 query 的有效候选数 |

mode 0 的 `k_scale_cache` 原地更新，不增加返回项。
非有效 query 行的索引填 `-1`，候选长度为 0。

## 分页 Cache 格式

### Mode 1：combined cache，默认

每页 128 个 token，共 16 个组；每组 8 个 token，占 544 字节：

```text
k_cache[P,16,544]

每组 [0:512]   = token 0…7 的 packed E2M1 数据，每 token 64 字节
     [512:544] = token 0…7 的 E8M0 scale，每 token 4 字节
```

每组内先存全部 token 的数据，再存全部 scale，不能按单 token 的 68 字节记录交错。
页内连续，要求 `stride(2)=1`、`stride(1)=544`，页 stride 至少为 8704 字节；
支持页间 padding。

### Mode 0：分离 cache

```text
k_cache[P,128,1,64]        # packed E2M1
k_scale_cache[P,128,1,4]   # E8M0 scale
```

页内 token 行连续，两个 cache 的页数和设备相同。
数据 cache 的 stride 为 `(page_stride,64,64,1)`，scale cache 为
`(scale_page_stride,4,4,1)`；页 stride 分别至少为 8192、512 字节，允许页间 padding。

### 寻址与更新语义

`block_table[0,logical_page]` 给出物理页号，活动页号须位于 `[0,P)`。
`cache_index` 使用物理 token slot，而不是逻辑 token 索引：

```text
physical_page = cache_index // 128
page_offset   = cache_index % 128
```

mode 1 的组号为 `page_offset//8`，组内第 `r=page_offset%8` 个 token 的
data 偏移为 `r*64`，scale 偏移为 `512+r*4`。

`cache_index=-1` 不写入；多个输入行写入同一 slot 时，按输入顺序最后一行生效，
其余 cache 字节保持原值。

上述页号与 slot 的越界属于调用方契约：`mega_full_index()` 与
`prepare_mega_full_index()` 只校验 shape/dtype/stride 等元信息，不读取张量内容，
因此不会引入 Device-to-Host 同步，也不影响 Graph 捕获。需要兜底时在首次接入
调用一次 `check_page_bounds(k_cache, block_table, cache_index)`，
它会在越界时直接报错（该函数自身会触发一次 D2H，不应放进热路径）。调用方应在执行前准备对应格式的历史 cache，
包装层不在两种格式之间自动迁移。mode 1 采用候选消费层要求的 combined PA 字节布局，
不以此代替下游算子的串接验证。

## 快速开始

依赖 CANNBotDSL 0.8、配套 PyTorch/torch_npu 和 CANN，环境配置见
[仓库 README](../../README.md)。加载 CANN 环境后，从仓库根目录执行：

```bash
export PYTHONPATH="$PWD:$PWD/test:$PYTHONPATH"
export CANNBOTDSL_CACHE_DIR="$PWD/.cache/cannbotdsl"
```

以下示例复用随仓库测试工具生成的合成输入，运行 `B=1,S=6,KV=8192`、mode 1。
输入工具只用于演示；实际网络应提供自己的投影权重、FP8 QR、latent、位置参数和历史 cache。

```python
import torch
import torch_npu
from samples.attention.mega_full_index.mega_full_index import (
    mega_full_index,
    prepare_mega_full_index,
)
from mega_full_index.test_mega_full_index import make_inputs

inputs = make_inputs(seq=6, kv=8192)

# 直接调用：内部准备工作区，执行一次融合 kernel。
outputs = mega_full_index(**inputs, storage_mode=1)
q, descale_q, w, cache, indices, values, candidates, candidate_length = outputs
assert cache is inputs["k_cache"]

# 重复调用：复用输入引用、工作区和输出 buffer。
prepared = prepare_mega_full_index(
    **inputs,
    softmax_scale=1.0 / 64.0,
    norm_eps=1e-20,
    storage_mode=1,
)
outputs = prepared.run()  # 首次执行相应配置时编译
outputs = prepared.run()
torch.npu.synchronize()
print(outputs[4].shape)   # torch.Size([6, 1, 512])
```

如需预分配 QW 输出，在准备时传入 `q`、`descale_q`、`w` buffer：
shape 分别为 `[T,32,64]`、`[T,32,2,2]`、`[T,32]`，
dtype 分别为 UINT8、UINT8、FP32。

mode 0 调用时，先准备分离的 data/scale cache，再传 `storage_mode=0`；
其余输入和输出顺序相同。

## 确定性计算

同一输入重复执行、以及 Graph 捕获后重放，公开输出逐项一致（`repeat` 与 `graph`
用例覆盖）。候选块在分数并列时的取舍由实现固定，不依赖执行顺序。

## 精度测试

测试集中在
[test_mega_full_index.py](../../../test/attention/mega_full_index/test_mega_full_index.py)，
包含输入构造、独立 CPU golden、阶段调试、完整回归和 profiling 入口。

CI 固定跑 9 条设备用例；判据自检（`pack_cache`/`unpack_cache` 字节契约、
E2M1 中点与负零、候选块选择参考、分块打分参考、公开签名、任务切分覆盖）
由 session 级 fixture `reference_selfcheck` 跑一次，不占设备、不单独计入用例数。

### 执行命令

```bash
# 9 条设备用例；核数自动获取。
python -m pytest -q test/attention/mega_full_index/test_mega_full_index.py

# 样例导入检查。
python -c "from samples.attention.mega_full_index.mega_full_index import mega_full_index, prepare_mega_full_index"

# 单例重现：使用当前平台可用核数，无需传入核数。
python test/attention/mega_full_index/test_mega_full_index.py \
  --worker --regression-case s8-kv8192-m1-values

# 显式调试入口，不属于回归用例。
python test/attention/mega_full_index/test_mega_full_index.py --worker --stage fp4
python test/attention/mega_full_index/test_mega_full_index.py \
  --worker --stage qw --seq 8 --kv 8192
```

`--stage k`、`--stage qli` 可独立调试相应阶段；它们同样查询当前可用核数。
FP4 专项入口检查中点/邻点、负零和最小 scale。

CPU golden 独立实现计算和选择逻辑。K 投影显式执行
`latent.float() @ wk.float().T`，结果转 BF16 后再转 FP32 进入 Norm；
长 KV 的 QK 分块计算以限制 CPU 内存。Q/scale、K cache/scale 和逐 token BF16
分数逐项精确比较；W 使用 `rtol=0.02, atol=0.02`。TopK、values 和填充精确检查；
候选检查块最大分数、数量、唯一性、最新可见块和选择阈值，阈值并列允许合法候选集合。
测试按上述数值边界验证，不等同于与原模型全精度计算逐层对照。

## 性能

`--output` 需为新的或空目录；同一入口还支持 `--mode pmu` 与 `--mode trace`
（外部 PipeTimeline 采集入口）。

```bash
python test/attention/mega_full_index/test_mega_full_index.py --profile \
  --seq 8 --kv 8192 --storage-mode 1 --mode latency --samples 31 \
  --output .cache/l20_indexer/profile_s8
```

采样口径：B=1、S=8、KV=8192（物理容量 8192 = 64 页 × 128 token）、
storage_mode=1（combined cache）、`return_value=False`（仅索引输出）、
32 AIC（`mix_blocks=32`）、CANNBotDSL 0.8.0、CANN 9.2 weekly；Graph 重放 31 次，
每次调用前驱逐 128 MiB L2；统计 `kernel_details.csv` 中 `mega_full_index`
的设备 `Duration(us)`，不含编译、CPU golden 与主机提交间隔，单位 µs：

| 轮次 | min | p50 | p95 |
| --: | --: | --: | --: |
| 1 | 29.386 | 30.650 | 31.374 |
| 2 | 30.746 | 31.430 | 32.200 |
| 3 | 28.429 | 29.217 | 30.130 |

同一 shape 连续三轮的 p50 为 29.2 / 30.7 / 31.4 µs，更早单轮落在 27.4 ~ 32.8 µs，
属同一波动区间：机器为多人共享，单轮差值不作为优化结论，跨版本对比需同机、
同口径、多轮取中位数。本表只覆盖 S8/KV8192 这一个 shape。
