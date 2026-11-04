# IndexerPrologueK

`IndexerPrologueK` 是基于 CANNBot-DSL 实现的 Indexer Key 前处理算子。它在 NPU 上依次完成matmul、RMSNorm、RoPE、MXFP4 量化和分页 cache 写入。

## 计算流程

```text
latent[T,H]
  -> BF16 GEMM: latent @ wk.T
  -> FP32 RMSNorm * norm_weight，结果舍入为 BF16
  -> 对最后 Dr 维执行相邻元素配对的 RoPE，结果舍入为 BF16
  -> 每 32 个元素量化为 signed E2M1，并生成一个 UE8M0 scale
  -> 根据 cache_index[T] 写入分页 cache
```

实现由两个同流 kernel 组成：

1. Cube kernel：完成 BF16 投影。
2. Vector kernel：融合完成 RMSNorm、RoPE、MXFP4 打包和 cache 散写。

两个 kernel 编译为一个按「模型规格 (H, D, Dr) + 存储模式」缓存的 AOT pipeline。`T` 与整个 cache 几何（block 数、slot 数、第 0/1 维 stride）是动态 `Dim` 契约轴：同一模型规格下任意 `T`/cache 形状复用同一份二进制，仅更换模型规格或存储模式时触发重新编译。

## Python 接口

```python
from indexer_prologue_k import indexer_prologue_k

result = indexer_prologue_k(
    latent,
    wk,
    norm_weight,
    rope_sin,
    rope_cos,
    k_cache,
    cache_index=cache_index,
    storage_mode=storage_mode,
    norm_eps=norm_eps,
    combined_block_size=-1,
)
```

算子原地更新 `k_cache`，并返回同一个 Tensor。`k_scale_cache` 是可选参数，默认值为 `None`。`cache_index`、`storage_mode` 和 `norm_eps` 是必填的 keyword-only 参数。

当 `cache_index[t] == -1` 时，对应 token 不写入 cache。若多个 token 指向同一位置，则按输入顺序写入，最后一个 token 的结果生效。

## 输入与布局

| 参数 | dtype / format | 形状与含义 |
|---|---|---|
| `latent` | BF16 / ND | `[T,H]`，输入 hidden states |
| `wk` | BF16 / FRACTAL_NZ | `[D,H]`，投影权重 |
| `norm_weight` | FP32 / ND | `[D]`，RMSNorm 权重 |
| `rope_sin` | FP32 / ND | `[T,Dr]`，RoPE 正弦系数 |
| `rope_cos` | FP32 / ND | `[T,Dr]`，RoPE 余弦系数 |
| `k_cache` | UINT8 / ND | 量化后的 E2M1 数据；具体形状由存储模式决定 |
| `k_scale_cache` | UINT8 / ND 或 `None` | mode 0 下可选的独立 UE8M0 scale cache |
| `cache_index` | INT64 / ND | `[T]`，展平后的 cache slot，`-1` 表示跳过 |
| `storage_mode` | `int` | `0`：数据与 scale 分离；`1`：combined cache |
| `norm_eps` | `float` | RMSNorm epsilon |
| `combined_block_size` | `int` | mode 1 的组合大小 `G`；mode 0 下忽略 |

除 `wk` 外，其余 Tensor 均使用 ND 格式。Python 包装层不重复校验输入 dtype，调用方应遵守上述接口契约。

规格约束：

- `T > 0`。
- `H` 为 16 的倍数，以满足 FRACTAL_NZ 布局要求。
- `D` 为 32 的倍数，以满足 MXFP4 block-32 量化要求。
- `0 < Dr <= D`，且 `Dr` 为偶数。
- 所有输入 Tensor 位于同一 NPU 设备。
- `k_cache` 以及 mode 0 的 `k_scale_cache` 支持第 0 维非连续 view；其余维度保持标准 ND 行布局。

## Cache 存储模式

### Mode 0：数据与 scale 分离

```text
k_cache:       [block_num, block_size, 1, D/2]
k_scale_cache: [block_num, block_size, 1, D/32] 或 None
```

`k_cache` 保存 E2M1 数据。传入 `k_scale_cache` 时，UE8M0 scale 写入独立 cache；传入 `None` 时仅写入 E2M1 数据，不会访问 scale cache。

### Mode 1：Combined cache

```text
G = combined_block_size
k_cache: [block_num, block_size/G, 1, G*(D/2 + D/32)]
```

每个 combined row 容纳 `G` 个 token：前半部分连续存放 `G` 份 E2M1 数据，后半部分连续存放对应的 `G` 份 UE8M0 scale。`combined_block_size` 必须为正数，且 cache 的第二维需要按 `block_size/G` 构造。

## 实现要点

- Cube/Vector 两个 kernel 的 launch 核数在每次调用时通过 `get_effective_core_counts()` 查询并作为运行时标量传入：以 `torch.npu` 设备属性为基线，叠加 cannbotdsl `get_platform_info` 的流控核配额；查询失败直接报错，不会回退到固定核数。
- 后处理仅将 RoPE 覆盖的尾部提升到 FP32，量化阶段直接读取 RoPE 后的 BF16 缓冲。
- 打包结果使用 32-byte 对齐的临时行，避免短 scale 和非对齐 combined row 写入覆盖相邻数据。
- 对齐的 E2M1 数据段使用 GM→UB→GM 搬运，非对齐规格使用安全路径。
- UE8M0 scale 使用精确长度的短 DMA。
- mode 1 按 combined row 分配 AIV；同一目标行始终由一个核按输入顺序处理，以保证重复 index 的最后写入语义。
- Cache 保持四维 ABI，并通过 TensorSpec stride 完成寻址；第 0/1 维 stride 是动态契约轴，不同 stride 的 cache view 不需要整理、复制，也不会触发单独编译。

## 测试

运行全部测试：

```bash
pytest test/attention/indexer_prologue_k/test_indexer_prologue_k.py -v
```

测试覆盖公开接口、CANN IR 编译、基础端到端计算、重复 index 语义以及随机规格泛化。泛化规格包括：

- 不同的 `T/H/D/Dr`；
- mode 0 有/无 `k_scale_cache`；
- mode 1 的 `G=2/4/8/16`；
- 不同分页形状和 `cache_index=-1`；
- 大规格 `H=7168,D=512,Dr=510`。

## 性能采集

性能脚本提供 `decode`、`prefill`、`combined` 和 `wide` 四个典型场景。脚本会先完成编译和预热，再执行指定次数的测量调用：

```bash
msprof --aic-metrics=PipeUtilization --task-time=on --ai-core=on \
  python test/attention/indexer_prologue_k/profile_indexer_prologue_k.py \
  --case combined --repeat 20
```

首次新 shape 的 AOT 编译时间不应计入 kernel 性能。分析结果时可重点关注各 kernel 的 `Task Duration` 和 PipeUtilization。
