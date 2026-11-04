# MegaReindex

## 算子说明

`mega_reindex` 是面向 Ascend 950、基于 CANNBotDSL 的 DeepSeek V4.1 候选消费层 Indexer 单 kernel 实现。算子把 `QuantX → WQ_A → RMSNorm/QuantQr → WQ_B/RoPE/FP4 Quant → 候选约束 QK/Head Reduce → TopK` 合并为一次 NPU kernel launch，输出稀疏注意力使用的压缩 KV 索引。

模型配置中 L24、L28、L32、L36 具有相同的 Indexer 结构：`hidden_size=5120`、`q_lora_rank=1280`、`index_n_heads=32`、`index_head_dim=128`、`compress_ratio=1`，并共同消费 L20 发布的候选块和 Index K。因此四层可复用同一算子程序，只需传入各层自己的激活和权重。算子不接收或硬编码 layer id。

`attn_norm`、候选源层的候选生成、Index K 的生成与缓存更新不在本算子内。当前性能与迁移回归覆盖 L24、B=1、S=8、kv_s=8192；L28/L32/L36 已确认接口与结构兼容，仍应使用真实层权重做整网验收。

## 实现逻辑

一次 MIX launch 内按全核同步点连接以下阶段，启动核数使用输入 device 当前 stream 的可用配额：

1. **QuantX**：将 BF16 `x[T,5120]` 转为 WQ_A 使用的 MXFP8 数据和 scale。
2. **投影与候选并行**：Cube 核区域并行执行 WQ_A 和 `weights_proj`；Vector 同时把每行候选块展开成紧凑块级 mask，并计算 causal 可见长度。
3. **RMSNorm + QuantQr**：WQ_A 结果经过 RMSNorm，使用 `norm_weight`（RMSNorm gamma）并量化为 WQ_B 输入。
4. **WQ_B + RoPE + FP4 Quant**：产生 32 个 128 维 query head，对最后 64 维执行相邻元素配对 RoPE，再生成仓内原生 packed MXFP4 query 和 E8M0 scale。
5. **候选约束 Score**：以 256 个 K token 为 N tile 执行 MXFP4 QK；L0C 结果转 BF16 后直接交给 Vector，依次执行 ReLU、逐 head BF16 融合乘加、候选/causal mask，输出 BF16 score。
6. **精确 TopK**：两遍 BF16 key 直方图寻找阈值，UB 内压缩阈值同分索引；同分优先较小位置，最终按位置升序输出，空槽写 `-1`。

阶段间使用 `global_sync_all()` 保证依赖，并使用 `onchip_rewind()` 重置片上分配；workspace 只承载必须跨阶段保存的数据，不引入额外算子 launch。

## Python 接口

```python
from attention.mega_reindex.mega_reindex import (
    MegaReindexTailWorkspace,
    mega_reindex,
    prepare_mega_reindex,
)
```

主入口：

```python
mega_reindex(
    x, wqa, descale_wqa, ww, norm_weight,
    wqb, descale_wqb, rope, key=None, descale_key=None,
    *, front_workspace=None, tail_workspace=None,
    norm_eps=1e-20, softmax_scale=1/64, topk=512,
    candidate_block_indices, candidate_block_length,
    cu_seqlens_q, cu_seqlens_k=None, seqused_q, seqused_k,
    candidate_block_size=8, mask_mode=3, cmp_ratio=1,
    layout_k="TND", block_table=None, k=None, descale_k=None,
) -> torch.Tensor
```

`prepare_mega_reindex(...)` 接口相同，返回 `(program, args, front_workspace, tail_workspace)`，用于预编译、图捕获或由上层显式管理 launch。

## 输入参数

令 `B=seqused_q.numel()`，`T=B*S`，`kv_s=key.shape[0]//B`。

| 参数 | dtype | shape / 布局 | 说明 |
| --- | --- | --- | --- |
| `x` | BF16 | `[T,5120]` ND | 当前候选消费层输入激活。 |
| `wqa` | FP8 E4M3/INT8/UINT8 byte view | `[5120,1280]` FRACTAL_NZ | WQ_A 权重；须在计时前完成 NZ 预处理。 |
| `descale_wqa` | E8M0 | `[80,1280,2]` ND | WQ_A 成对 E8M0 scale，K 方向每 64 元素一组。 |
| `ww` | BF16 | `[32,5120]` FRACTAL_NZ | `weights_proj` 权重，不含 `softmax_scale`。 |
| `norm_weight` | BF16 | `[1280]` | WQ_A 后 RMSNorm 的 gamma 参数。 |
| `wqb` | FP8 E4M3/INT8/UINT8 byte view | `[4096,1280]` FRACTAL_NZ | WQ_B 权重，输出 32×128 query。 |
| `descale_wqb` | E8M0 | `[4096,20,2]` ND | WQ_B 成对 E8M0 scale，K 方向每 64 元素一组。 |
| `rope` | FP32 | `[T,32,2]` | 每 token 的 32 个相邻旋转维度对；所有 head 共享。`rope[t,pair,0]` 为 cos，`rope[t,pair,1]` 为 sin，作用于最后 64 维。 |
| `key` | UINT8 | `[B*kv_s,1,64]` | 仓内 packed MXFP4 E2M1 Index K；每个 128 维 key 占 64 byte。 |
| `descale_key` | E8M0 | `[B*kv_s,1,2,2]` | Index K 的 E8M0 scale，共四个 32 元素分组。 |
| `candidate_block_indices` | INT32 | `[T,1,2048]` | 每行有效前缀内的 batch-local 候选块编号；块编号须唯一。 |
| `candidate_block_length` | INT32 | `[T,1]` | 每行候选块有效数量，范围 `[0,2048]`。 |
| `cu_seqlens_q` | INT32 | `[B+1]` | 等容量 Q batch 边界，要求 `cu_seqlens_q[b]=b*S`。 |
| `cu_seqlens_k` | INT32 | `[B+1]` | 等容量 K batch 边界，要求 `cu_seqlens_k[b]=b*kv_s`。 |
| `seqused_q` | INT32 | `[B]` | 每个 batch 实际有效 query 数，范围 `[0,S]`。 |
| `seqused_k` | INT32 | `[B]` | 每个 batch 实际有效 key 数，范围 `[0,kv_s]`。 |
| `front_workspace` | workspace/None | `FrontWorkspace(T,device)` | 可选前段 workspace；重复调用应复用。 |
| `tail_workspace` | workspace/None | `MegaReindexTailWorkspace(T,kv_s,device,topk)` | 可选紧凑 workspace；重复调用应复用。 |
| `norm_eps` | Python float | 正有限值 | RMSNorm epsilon，模型默认 `1e-20`。 |
| `softmax_scale` | Python float | 有限值 | `weights_proj` 缩放，模型默认 `1/64`。 |
| `topk` | Python int | 64 的倍数 | 要返回的索引数；要求 `64<=topk<=kv_s<=16384`。 |
| `candidate_block_size` | Python int | 固定 8 | 每个候选块覆盖 8 个压缩位置。 |
| `mask_mode` | Python int | 固定 3 | 候选块、实际长度和 causal 联合约束模式。 |
| `cmp_ratio` | Python int | 固定 1 | L24/L28/L32/L36 的压缩比例。 |

额外 shape 约束：`T>0`，`B>0`，`T%B==0`，paired-token score 模板要求正偶数 `S=T/B`。所有 Tensor 必须在同一 NPU device 上，stride 和 NZ storage format 必须满足运行时检查。

## 输出参数

返回 INT32 `indices[T,topk]`：

- 有效位置是 batch-local 压缩 KV 索引，并按位置升序排列；
- BF16 score 相同时优先选择较小索引；
- 不可见、非候选或有效位置不足时用 `-1` 补齐；
- 同一 workspace 不可被多个并发 stream 同时使用。

## 调用样例

```python
import torch
from attention.mega_reindex.mega_reindex import FrontWorkspace
from attention.mega_reindex.mega_reindex import (
    MegaReindexTailWorkspace,
    mega_reindex,
)

B, S, kv_s, topk = 1, 8, 8192, 512
T = B * S
front_ws = FrontWorkspace(T, x.device)
tail_ws = MegaReindexTailWorkspace(T, kv_s, x.device, topk=topk)

# wqa、ww、wqb 的 NZ 预处理、编译和 workspace 分配均应在计时前完成。
indices = mega_reindex(
    x, wqa, descale_wqa, ww, norm_weight,
    wqb, descale_wqb, rope, key, descale_key,
    front_workspace=front_ws, tail_workspace=tail_ws,
    norm_eps=1e-20, softmax_scale=1/64, topk=topk,
    candidate_block_indices=candidate_block_indices,
    candidate_block_length=candidate_block_length,
    cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
    seqused_q=seqused_q, seqused_k=seqused_k,
)
torch.npu.synchronize()
assert indices.shape == (T, topk)
assert indices.dtype == torch.int32
```

完整可复用调用函数见仓库 `test/attention/mega_reindex/example.py`。

## 性能优化策略

- **单 kernel 融合**：六个计算阶段只产生一次 launch，消除小算子调度和中间 GM 往返。
- **Cube/Vector 并行**：WQ_A 和 `weights_proj` 使用不同 Cube 核区域；候选处理同时在 Vector 执行，在依赖屏障前完成。
- **NZ 权重直入**：WQ_A、WQ_B、`weights_proj` 在流程外预转 FRACTAL_NZ，运行时不计格式转换成本，提高 L1/L0B 搬运效率。
- **Tile 数据流**：WQ_A 的输出 tile 由 `get_qbmm_mx_tile_config` 根据矩阵尺寸、转置配置和模板使用的设备资源信息选择，N 不固定为 128；MXFP4 QK 当前固定使用 N=256 key tile，其 Cube 结果经 L0C→UB 直接交给 Vector 做 ReLU、权重融合和 mask。
- **双缓冲**：QK 的 K、scale、L0B、L0C 和 UB handoff 使用适合当前 tile 的流水深度，覆盖搬运与计算。
- **Packed MXFP4 原生计算**：Q/K 保持仓内 E2M1 packed 数据及 E8M0 scale，避免先完整解码到 GM。
- **保持数值边界**：QK 从 FP32 L0C 转 BF16；ReLU 与逐 head 融合乘加按 BF16 语义执行，与 L24 参考路径一致。
- **块级候选 mask**：每 8 token 保存 1 byte。B=1、S=8、kv_s=8192 时 mask 从 65536 byte 降到 8192 byte，并减少清零和 scatter 数量。
- **128-lane Vector score**：每个 Vector 分段处理 128 lane，降低循环和指令开销。
- **精确稳定 TopK**：两遍 BF16 直方图和 UB 同分压缩替代通用排序与额外 kernel，并保留严格 tie-break。
- **编译与 workspace 缓存**：缓存键包含 T、B、kv_s、topk、标量和 Tensor spec；同 shape 四层复用程序。workspace 在整网初始化时分配。

WQ_B/Query 使用固定的 `M16/N128/K256` tile，保持现有 RoPE/FP4 后处理布局。
每个 M tile 的 32 个 N tile 按当前可用 Cube 核数循环分配；28 核时，
0～3 号核各执行两个 N tile，4～27 号核各执行一个。该规则不依赖
通用 MX 模板按核数自动选择的 baseN。WQ_A 仍使用通用模板选择 tile。

## 环境与使用

本目录包含全部算子源码。生产入口仅依赖 Python 标准库、PyTorch、
torch_npu 和 CANNBotDSL；不再导入其他 samples 目录或其他仓库。
reindex_front.py 提供前段量化、投影、归一化、FrontWorkspace 和静态权重预处理；
reindex_query.py 提供 WQ_B/RoPE/FP4 Query 生成；
reindex_tail.py 提供候选与 PA 准备、score 和稳定 TopK；
reindex_runtime.py 提供参数校验、运行时适配及内联 MX scale/FP8/FP4 公共函数；
reindex_matmul.py 与 reindex_mx.py 保留独立的底层矩阵乘模板。
这些 phase 在主 kernel 内展开，运行时仍为一次 launch。

在已安装 CANN、torch/torch_npu、CANNBotDSL 的环境中，从默认名称
`cannbot-dsl` 的克隆目录上一级进入仓库，并将 `samples` 加入模块搜索路径：

    cd cannbot-dsl
    PYTHONPATH="$PWD/samples${PYTHONPATH:+:$PYTHONPATH}" python3.12 -c "from attention.mega_reindex.mega_reindex import mega_reindex"

算子以源码方式使用，首次调用按 shape 编译，后续复用进程内编译缓存。
算子入口与 `FrontWorkspace` 从主源码模块导入；`prepare_weight_nz` 位于
`reindex_front.py`，`prepare_wqa_nz` 和 `prepare_wqb_nz` 位于 `reindex_front.py`。
示例和测试使用仓库统一的 `_samples_path.load_sample` 加载对应源码文件。


## PA 分页缓存（PA_BBND）

新增 layout_k="PA_BBND" 和 block_table，兼容仓内
quant_sparse_lightning_indexer_dsl 的 PA 缓存字节布局。
key/descale_key 的旧位置参数继续支持；也可使用仓内命名 k/descale_k
作为关键字别名，两种名字不得同时提供同一输入。

| 参数 | TND | PA_BBND |
| --- | --- | --- |
| k / key | UINT8 [B*kv_s,1,64] | UINT8 [P,page_size/8,544] |
| descale_k / descale_key | UINT8/E8M0 [B*kv_s,1,2,2] | 必须 None，scale 内嵌在 k |
| block_table | 必须 None | 必填 INT32 [B,max_pages] |
| cu_seqlens_k | 必填 | 必须 None |
| seqused_k | INT32 [B] | INT32 [B]，实际有效 token 数 |
| workspace kv_s | TND 每 batch 容量 | max_pages*page_size |

page_size 为 16 的倍数，范围 [16,1024]。
PA 的逻辑容量 max_pages*page_size 仍须满足 topk<=容量<=16384。
每个 8-token 块的前 512 字节保存 K，后 32 字节保存 scale。
K 的页内两轴须连续；物理页之间允许 padding。
block_table 必须连续，活动页表项由调用方保证为 [0,P) 内有效物理页号；
有效长度以外的整页不读取页表，可用 -1 占位。
seqused_k 的值须位于 [0,逻辑容量]。
候选块号仍是 batch-local 逻辑 8-token 块号；输出是逻辑 token 索引，
不会返回物理页号。Q 的等容量 B*S 与正偶数 S 约束保持不变。

数据路径：
L20 packed PA cache -> 本 kernel Vector 按页表读取 -> UB 拆分 K/scale
-> 临时 GM -> 现有 MXFP4 QK -> BF16 score -> TopK。
PA 读取与 WQ_A/weights_proj Cube 投影并行，复用已有全核屏障，
无需额外分页解包 kernel。首版按实际有效页读取，不按候选稀疏程度跳页，
且暂存整个逻辑 K/scale，尚不是 PA 直接到 L1 的零暂存方案。

临时 workspace 增量为 68*B*逻辑容量 字节及 4*(B+1) 字节内部前缀。
B=1、容量8192时为544 KiB加8字节。首次 prepare 分配，重复调用复用；
同一 workspace 不可同时用于并发 stream。
原地更新 page table、缓存内容或 seqused_k 后，下一次调用读取新值，
不需要重新编译。上游 L20 写入须通过同 stream 顺序或 event 保证可见。

调用示意（其余参数沿用原接口）：

    indices = mega_reindex(
        x, wqa, descale_wqa, ww, norm_weight, wqb, descale_wqb, rope,
        k=l20_packed_cache, descale_k=None,
        layout_k="PA_BBND", block_table=block_table,
        candidate_block_indices=candidate_block_indices,
        candidate_block_length=candidate_block_length,
        cu_seqlens_q=cu_seqlens_q, seqused_q=seqused_q, seqused_k=seqused_k,
        front_workspace=front_ws, tail_workspace=tail_ws,
    )

TND 默认路径继续使用原编译模板；PA 的 page size、capacity 和 stride
属于编译签名。本扩展不增加 QSLI metadata 输入，不改变最终 [T,topk] 输出。


测试入口位于仓库 `test/attention/mega_reindex`，不随算子包发布。
回归命令（从仓库根目录运行）：

    python3.12 -m pytest -m npu test/attention/mega_reindex

    python3.12 test/attention/mega_reindex/smoke.py

    python3.12 test/attention/mega_reindex/verify_pa.py
    python3.12 test/attention/mega_reindex/verify_pa_batch.py

测试自行生成输入并将 PA 寻址结果与同字节 TND 路径比较；不依赖其他仓库、
真实模型文件或外部测试数据。第一条覆盖四种页大小、物理页 padding、
乱序页表、短长度、空候选、无效 query、页表更新，仅检查精度一致性；
第二条覆盖 B=2、S=6、多 batch 独立物理页。

每次 prepare 和返回程序的 Python 调用都按输入 device 的当前 stream
查询资源，以 `min(Cube, Vector // 2)` 确定启动核数。两路投影至少需要 2 Cube / 4 Vector。
投影分区、数据遍历和同步使用同一配额，编译缓存包含核数；切换 stream 配额后自动选择对应程序。
Graph 需先在目标核配额下预热，再显式使用该 stream 捕获；核配额改变后重新捕获。
候选阶段在整批元数据不超过本机 `get_mem_size("ub") / 64` 时保留原有
整批加载；超过该预算时，只加载当前 batch 的两个 Q/K 边界和两个有效长度，
共 24 字节逻辑 INT32 元数据，避免 UB 分配随总 batch 数无限增长。

scale 参数使用 `torch.float8_e8m0fnu`，权重预处理函数直接返回 E8M0 scale。
为兼容原始 patch，同时接受同字节编码的 UINT8 输入；内部仅使用 dtype view，
不做数值转换，不改变量化编码、形状和计算公式。

NPU CI 通过既有 `test_deployment.py` 调用 smoke、PA 和多 batch 脚本，
并验证非恒等 RoPE、不同 stream 核配额和 Graph 重放的结果一致性。


### 性能复测的参考快照

`benchmark_migration.py` 需要与待测脚本输入完全一致的参考快照；仓库
不附带 `baseline.pt`。先在参考实现对应的 DSL 环境中运行
`verify_migration.py capture`，再在待测环境中显式传入 `--reference`：

    python3.12 test/attention/mega_reindex/verify_migration.py capture \
      /path/to/reference_setup.py /path/to/baseline.pt
    python3.12 test/attention/mega_reindex/benchmark_migration.py \
      /path/to/candidate_setup.py /path/to/profile_output \
      --reference /path/to/baseline.pt --samples 30

两个 setup 脚本应定义同一组确定性输入及 `program`、`args`、
`candidate_block_length`、`num_blocks`、`front_workspace`、
`tail_workspace`。参考快照由原实现独立进程生成，不能用待测结果充当
参考。若未提供文件或快照缺少 `full/tail/indices`、
`full/tail/score`，性能脚本会在启动 NPU 工作前报错。
