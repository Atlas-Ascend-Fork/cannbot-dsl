# 算子样例

CANNBot-DSL 的算子源码和各样例说明保存在仓库根目录的 `samples/` 中。

请前往[算子样例目录](https://gitcode.com/cann/cannbot-dsl/tree/master/samples)查看样例源码、接口说明和运行方式。

- [mega_mla](https://gitcode.com/cann/cannbot-dsl/blob/1a141a09bd0cd280a65f5ff5231b177315f699d6/samples/attention/mega_mla/README.md)：融合 Q/KV 投影、RMSNorm、RoPE、window cache 更新及双路稀疏 attention。

- [mega_reindex](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/attention/mega_reindex)：DeepSeek V4.1 候选消费层 Indexer，融合量化、投影、RMSNorm、RoPE、候选打分与 TopK；支持 TND 与 PA 分页 KV 两种布局，以及 ds41 Native 打包。
