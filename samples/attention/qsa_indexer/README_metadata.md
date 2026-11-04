# QSA Indexer Metadata

通过 NPU 的 AICPU 执行 whole-K 分核，生成 QSA Indexer 主算子使用的 INT32 metadata。
每 32 个 Query 形成一个任务，单个任务覆盖其完整 K，不沿 Key 方向跨核拆分。

## 接口

```python
from qsa_indexer_metadata import qsa_indexer_metadata

metadata = qsa_indexer_metadata(
    actual_seq,
    query_positions,
    block_table,
    compressed_page_count=compressed_k.shape[0],
)
```

- `actual_seq` 为当前 NPU 上的连续一维 INT32 Tensor，shape 为 `[B+1]`，
  保存 packed Query 的 batch 边界并包含前导 0。
- `query_positions` 为当前 NPU 上的连续一维 INT32 Tensor，shape 为 `[T]`，
  保存每个 Query 的 batch 内零基位置。
- `block_table` 为当前 NPU 上的连续二维 INT32 Tensor，shape 为 `[B,max_pages]`，
  保存逻辑页到物理页的映射；每个页表项覆盖 256 个压缩 Key。
- `compressed_page_count` 必须是非负 Python `int`，通常传入 `compressed_k.shape[0]`，
  用于检查物理页号。
- `block_dim` 可选；默认查询当前设备和流的有效 Cube 核数。显式值必须位于
  `[1,有效 Cube 核数]`，且不超过 Metadata 物理布局支持的 36 个 AIC 槽。
- Batch 范围为 1～65536；三个 Tensor 必须位于当前同一 NPU、连续且 dtype 正确。
- 输出容量为 `align_up((1+B*36)*16,4096)` 个 INT32。头部第 0 项保存 section 数，
  其余头部字段清零；每个 section 固定保留 36 个核槽，每槽 16 项。
- 每个有效核槽的前 6 项为
  `[start_request, start_m, 0, end_request, end_m, 0]`，范围采用左闭右开坐标；
  `m` 是 batch 内的 Q32 任务编号。inactive core 写入该 section 的排他终点，
  保留字段和对齐 padding 清零。
- 分核代价包含 M128/N256 scoring、TopK4096 trunk、最多 512 个历史候选、
  score 读回和尾部补零。每个 section 按总代价将连续 Q32 任务分配给参与核，
  不拆分单个任务的 K。
- section 按固定 96 MiB 工作集预算在 batch 边界划分；主算子依据 Metadata 范围以及设备侧的
  `actual_seq`、`query_positions` 恢复 Query 起点、实际行数和可见压缩块数。
- 第一次调用编译 AICPU 产物，之后复用进程内缓存。算子在当前 NPU 流上异步启动，
  不把输入复制回 CPU，也不在 Host 计算分核或执行 `torch.npu.synchronize()`。
- 先调用本算子，再通过 `qsa_indexer(..., metadata=metadata)` 执行主算子。
  两次调用应使用相同 `block_dim`，并在同一 NPU 流上依次提交；跨流调用时由调用方建立依赖。

## 测试

```bash
pytest test/attention/qsa_indexer/test_qsa_indexer_metadata.py -v
```

测试在 AICPU 上运行，覆盖单/多 section、零长度 batch、不同 `block_dim`、
非默认流、Metadata 范围连续覆盖和输入校验；不执行 QSA Indexer 主算子。
