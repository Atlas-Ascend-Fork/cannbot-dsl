# 算子样例

算子实现和使用说明位于 `samples/<类别>/<算子>/`，配套测试位于
`test/<类别>/<算子>/`。各样例 README 介绍接口、参数、运行环境和测试方法。

## 分类与算子

### attention

Attention、MLA、KDA，以及专属索引、前后处理和 KV Cache 算子。

- [attn_epilogue](attention/attn_epilogue/README.md)
- [commit_recurrent_kda_replayssm](attention/commit_recurrent_kda_replayssm/README.md)
- [flash_attn](attention/flash_attn/README.md)
- [flash_attn_fp8_fullquant](attention/flash_attn_fp8_fullquant/README.md)
- [flash_mla_with_kvcache](attention/flash_mla_with_kvcache/README.md)
- [flash_kda](attention/flash_kda/README.md)
- [flash_kda_cp_main](attention/flash_kda_cp_main/README.md)
- [flash_kda_cp_preprocess](attention/flash_kda_cp_preprocess/README.md)
- [flash_kda_fwd](attention/flash_kda_fwd/README.md)
- [fused_norm_rope_attn_rope_cast_fwd](attention/fused_norm_rope_attn_rope_cast_fwd/README.md)
- [fused_recurrent_kda_replayssm](attention/fused_recurrent_kda_replayssm/README.md)
- [fused_recurrent_kda_snapshot](attention/fused_recurrent_kda_snapshot/README.md)
- [indexer_prologue_k](attention/indexer_prologue_k/README.md)
- [indexer_prologue_qw](attention/indexer_prologue_qw/README.md)
- [mega_compressor](attention/mega_compressor/README.md)
- [mega_mla](attention/mega_mla/README.md)
- [mega_full_index](attention/mega_full_index/README.md)
- [mega_recurrent_kda](attention/mega_recurrent_kda/README.md)
- [mega_recurrent_kda_replayssm](attention/mega_recurrent_kda_replayssm/README.md)
- [mixed_quant_sparse_flash_mla](attention/mixed_quant_sparse_flash_mla/README.md)
- [mla_prolog_v4](attention/mla_prolog_v4/README.md)
- [qsa_indexer](attention/qsa_indexer/README.md)
- [quant_block_sparse_attn](attention/quant_block_sparse_attn/README.md)
- [quant_lightning_indexer](attention/quant_lightning_indexer/README.md)
- [quant_lightning_indexer_metadata](attention/quant_lightning_indexer_metadata/README.md)
- [quant_mla_prolog](attention/quant_mla_prolog/README.md)
- [quant_sparse_lightning_indexer](attention/quant_sparse_lightning_indexer/README.md)
- [quant_sparse_lightning_indexer_metadata](attention/quant_sparse_lightning_indexer_metadata/README.md)
- [qwen_sparse_attn](attention/qwen_sparse_attn/README.md)
- [sparse_flash_mla](attention/sparse_flash_mla/README.md)
- [sparse_flash_mla_metadata](attention/sparse_flash_mla_metadata/README.md)
- [stem_indexer](attention/stem_indexer/README.md)

### matmul

矩阵乘、批量矩阵乘、分组矩阵乘及量化和融合变体。

- [batch_matmul](matmul/batch_matmul/README.md)
- [dual_level_quant_matmul](matmul/dual_level_quant_matmul/README.md)
- [fused_matmul](matmul/fused_matmul/README.md)
- [grouped_matmul](matmul/grouped_matmul/README.md)
- [matmul](matmul/matmul/README.md)
- [quant_matmul](matmul/quant_matmul/README.md)
- [weight_quant_batchmatmul](matmul/weight_quant_batchmatmul/README.md)

### vector

逐元素、归一化、门控、通用索引和量化算子。

- [add](vector/add/README.md)
- [block_attn_res_update_rms_norm](vector/block_attn_res_update_rms_norm/README.md)
- [dynamic_mx_quant_sr](vector/dynamic_mx_quant_sr/README.md)
- [engram_gate](vector/engram_gate/README.md)
- [gather_v2](vector/gather_v2/README.md)
- [scatter_elements](vector/scatter_elements/README.md)

### mc2

跨卡通信与通信计算融合，包括专家权重预取。

- [moonep](mc2/moonep/README.md)

## 运行样例

以 FlashKDA 为例，在仓库根目录执行：

```bash
export PYTHONPATH="$PWD/samples/attention${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pytest test/attention/flash_kda/test_flash_kda.py -v
```

源码与测试路径见 [算子清单](../catalog/README.md)，Native 编译与 wheel 打包见
[Native 打包](../packaging/native/README.md)。
