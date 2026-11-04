# Stem Indexer Metadata

通过NPU的AICPU执行SectionStreamK分核，生成StemIndexer使用的int32 metadata。
按M64/N256分块分配任务，不拆分S2方向的计算。

## 接口

```python
from stem_indexer_metadata import stem_indexer_metadata

metadata = stem_indexer_metadata(
    q_seq_lens, kv_seq_lens, 32, 8,
    causal=True, stem_block_size=128, window_size=4, dim_qkflat=2048,
)
```

- 两个长度输入为当前NPU上的连续一维int32 Tensor，单位是token；batch数相同，范围为1～65536，长度值必须非负。
- `q_heads`为`kv_heads`的正整数倍。固定tile为M64/N256，特征维和stem参数用于SI分核代价计算。
- 默认查询当前设备/流的有效核数；可传`block_dim`限制参与分核的AIC数量，使其与主Kernel一致，不允许超过有效核数。物理metadata容量上限为36个AIC槽、72个AIV保留槽。
- 输出容量为`align_up((1+B*Nkv*(36+72))*16,4096)`个int32。头部第0项为section数；其余头部字段为零，每个AIC槽前6项为起止BN/M/S2坐标；S2起止为零，保留区和padding清零。
- 每个section按96MiB L2预算划分；因果范围扣除固定尾窗口，只分配完整M任务，不进行split-KV。
- 第一次调用编译AICPU产物，之后复用进程内缓存。同当前流异步启动；函数不把长度拷回CPU、不在Host计算分核。
- 先调用本算子，再通过`stem_indexer(..., metadata=metadata)`执行主算子并获取索引及有效长度。主算子必须显式接收metadata；两次调用的长度、head、causal、窗口和核数配置应一致。

## 测试

```bash
pytest test/attention/stem_indexer/test_stem_indexer_metadata.py -v
```

在AICPU上运行并与预期metadata逐元素比较，覆盖因果/非因果、空长度、decode、跨head尾块、变长batch、多section和非默认流；不执行SI主算子。
