# 网络算子分组

Native 构建按 [network.json](network.json) 中的算子 ID 选择分组成员。
构建命令见 [Native 编译与打包](../README.md#构建)。

## DS41

- `mixed_quant_sparse_flash_mla`
- `quant_lightning_indexer`
- `quant_sparse_lightning_indexer`
- `indexer_prologue_qw`
- `indexer_prologue_k`
- `engram_gate`
- `attn_epilogue`
- `mega_mla`
- `mega_full_index`
- `mega_compressor`
- `mega_reindex`
- `attn_prologue`

## K3

- `block_attn_res_update_rms_norm`
- `flash_kda`
- `flash_kda_fwd`
- `flash_kda_cp_preprocess`
- `flash_kda_cp_main`
- `fused_recurrent_kda_snapshot`
- `fused_recurrent_kda_replayssm`
- `mega_recurrent_kda`
- `commit_recurrent_kda`
- `mla_prolog_v4`
- `quant_mla_prolog`
