---
pageClass: examples-index-page
---

# 样例导航

本页按场景汇总仓库 README 中的算子样例，包含同目录下的不同实现及配套 metadata 调度算子。每个链接进入对应样例目录，输入约束、依赖、运行方式和验证结果以该目录的 README 为准。

| 场景 | 样例 | 接口 | 说明 |
| --- | --- | --- | --- |
| 矩阵计算 | [matmul/matmul](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/matmul/matmul) | `matmul()` | A16W16 矩阵乘，`C[M,N] = A[M,K] × B[N,K]^T` |
| 矩阵计算 | [matmul/matmul](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/matmul/matmul) | `matmul_streamk()` | 同目录的 Stream-K（DPSK）实现，DP + SK 混合调度，适合小 M/N、大 K |
| 矩阵计算 | [matmul/batch_matmul](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/matmul/batch_matmul) | `batch_matmul()` | 批量矩阵乘，batch 维右对齐广播，支持 rank 2~6 |
| 矩阵计算 | [matmul/quant_matmul](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/matmul/quant_matmul) | `npu_quant_matmul()` | MXFP8 / MXFP4 全量化矩阵乘（`quant_batch_matmul_mx.py`） |
| 矩阵计算 | [matmul/quant_matmul](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/matmul/quant_matmul) | `npu_quant_matmul()` | per-tensor（TT）量化矩阵乘，支持 HiFloat8 / INT8 / FP8（`quant_batch_matmul_hif8_tt.py`） |
| 矩阵计算 | [matmul/quant_matmul](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/matmul/quant_matmul) | `matmul_mix_quant()` | MXA8W4：MXFP8 激活 × MXFP4 权重（`quant_batch_matmul_mxa8w4.py`） |
| 矩阵计算 | [grouped_matmul](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/grouped_matmul) | `group_matmul()` | 分组矩阵乘，支持 M 轴 / K 轴分组 |
| 注意力与调度 | [flash_attn](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/flash_attn) | `flash_attn()`、`flash_attn_metadata()` | Flash Attention，支持 GQA、变长序列与分页 KV Cache |
| 注意力与调度 | [flash_attn_fp8_fullquant](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/flash_attn_fp8_fullquant) | `flash_attn_fp8_fullquant()` | FP8 全量化 Attention，支持 GQA、分页 KV Cache 与 causal mask |
| 注意力与调度 | [sparse_flash_attention](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/sparse_flash_attention) | `sparse_flash_attention()` | Sparse Flash Attention（SFA），按 `sparse_indices` 选取参与计算的 KV token |
| 注意力与调度 | [qwen_sparse_attn](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/qwen_sparse_attn) | `qwen_sparse_attn()` | 固定 `block_size=128` 的分块稀疏注意力，含 AICPU metadata 规划 |
| 注意力与调度 | [flash_mla_with_kvcache](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/flash_mla_with_kvcache) | `flash_mla_with_kvcache()`、`flash_mla_with_kvcache_metadata()` | Flash MLA 推理，Q 为 TND、KV 走分页 cache |
| 注意力与调度 | [quant_block_sparse_attn](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/quant_block_sparse_attn) | `quant_block_sparse_attn()`、`quant_block_sparse_attn_metadata()` | FP8 量化块稀疏注意力，面向分页 prefill |
| 注意力与调度 | [mixed_quant_sparse_flash_mla](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/mixed_quant_sparse_flash_mla) | `mixed_quant_sparse_flash_mla()` | 混合量化稀疏 MLA：BF16 Q + FP8 原始 KV + 可选 FP4 压缩 KV |
| 注意力与调度 | [mixed_quant_sparse_flash_mla_metadata](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/mixed_quant_sparse_flash_mla_metadata) | `mixed_quant_sparse_flash_mla_metadata()` | MQSMLA 的 AICPU 分核调度，输出整行 / FlashDecode 分片计划 |
| 注意力与调度 | [flash_kda](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/flash_kda) | `flash_kda()`、`flash_kda_metadata()` | Kimi Delta Attention prefill 融合算子，含 AICPU 调度 metadata |
| 注意力前处理 | [attn_prologue](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/attn_prologue) | `attn_prologue()` | MXFP8 的 QA/KV 投影、RMSNorm、QR 量化、QB 投影、RoPE 与 KV cache 写回 |
| 稀疏索引与调度 | [indexer_prologue_qw](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/indexer_prologue_qw) | `indexer_prologue_qw()` | MXFP8 Q GEMM、尾部 RoPE、MXFP4 量化，以及 BF16 W GEMM |
| 稀疏索引与调度 | [indexer_prologue_k](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/indexer_prologue_k) | `indexer_prologue_k()` | Indexer K 路前处理：BF16 投影、RMSNorm、RoPE、MXFP4 量化与分页 cache 写入 |
| 稀疏索引与调度 | [qsa_indexer](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/qsa_indexer) | `qsa_indexer()`、`qsa_indexer_metadata()` | 压缩 Key 稀疏索引，选高分压缩块并展开为 token 索引 |
| 稀疏索引与调度 | [stem_indexer](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/stem_indexer) | `stem_indexer()`、`stem_indexer_metadata()` | 块级特征打分 + 动态 TopK 选择 Key Block |
| 稀疏索引与调度 | [quant_lightning_indexer_dsl](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/quant_lightning_indexer_dsl) | `quant_lightning_indexer()` | MXFP4 Lightning Indexer（QLI），遍历上下文选 TopK token |
| 稀疏索引与调度 | [quant_lightning_indexer_metadata_dsl](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/quant_lightning_indexer_metadata_dsl) | `quant_lightning_indexer_metadata()` | QLI 配套的 AICPU 调度算子 |
| 稀疏索引与调度 | [quant_sparse_lightning_indexer_dsl](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/quant_sparse_lightning_indexer_dsl) | `quant_sparse_lightning_indexer()` | MXFP4 Sparse Lightning Indexer（QSLI），仅访问候选 Key 块 |
| 稀疏索引与调度 | [quant_sparse_lightning_indexer_metadata_dsl](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/quant_sparse_lightning_indexer_metadata_dsl) | `quant_sparse_lightning_indexer_metadata()` | QSLI 配套的 AICPU 调度算子 |
| KV Cache | [kv_compress_epilog](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/kv_compress_epilog) | `kv_compress_epilog()` | KV Cache 量化压缩与按槽位原地更新 |
| 归一化与门控 | [rms_norm](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/rms_norm) | `rms_norm()` | RMSNorm，`y = x · rstd · γ` |
| 归一化与门控 | [engram_gate](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/engram_gate) | `engram_gate()` | Engram 残差门：双路 RMS、加权点积与 signed-sqrt sigmoid 门控 |
| 点云 | [pointnet_sa](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/pointnet_sa) | `pointnet_sa()` | PointNet++ SA 层的 shared MLP + max-pool |
| 卷积 | [voxel_conv](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/voxel_conv) | `voxel_conv()` | VoxelNet Convolutional Middle Layers |

## 阅读样例

建议依次查看目录 README、实现文件和对应测试。README 描述样例用途和约束，实现文件展示主要逻辑，测试文件给出输入构造与结果验证方式。

- 首次使用请先完成[环境准备](/zh/install/)。
- 验证结果请参照[运行测试](/zh/project/examples/testing)。
- 理解实现原理可结合[编程模型](/zh/guide/programming-model/overview)和[API 文档](/zh/api/)。
