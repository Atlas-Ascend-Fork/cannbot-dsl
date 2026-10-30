# quant_sparse_lightning_indexer_metadata

## 产品支持情况

- Ascend 950PR / Ascend 950DT：支持。

## 功能说明

`quant_sparse_lightning_indexer_metadata` 是 [QSLI](../quant_sparse_lightning_indexer/README.md) 配套的独立 AICPU 调度算子。它根据每个 Query 的候选长度、序列长度和静态配置，生成 LI 计算范围及 LD 跨核 TopK 归并关系，不读取候选块索引或 Q/K 特征，不计算分数与 TopK。

每个 Query 独立调度。根据 mask 和压缩参数判断 Query 是否存在可见 Key 后，将有效候选按每 64 个块划分为一个 tile，即每 tile 最多 `64 × 8 = 512` 个候选 token。LI 记录每核在 `(batch, query_row, candidate_tile)` 坐标空间中的起止范围；同一 Query 的候选 tile 被拆到多个核时，LD 记录分片数、临时结果位置和输出行，供主算子归并局部 TopK。工作量来自候选有效前缀，而非完整 S2 扫描。

候选长度为零或 Query 无可见 Key 时保留必要的输出初始化调度。实现位于 [quant_sparse_lightning_indexer_metadata.py](quant_sparse_lightning_indexer_metadata.py)。

## 函数原型

```python
def quant_sparse_lightning_indexer_metadata(
    candidate_block_length,
    cu_seqlens_q=None, cu_seqlens_k=None,
    seqused_q=None, seqused_k=None, cmp_residual_k=None,
    *, batch_size=None, max_seqlen_q=-1, max_seqlen_k=-1,
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, candidate_block_size,
    mask_mode=0, cmp_ratio=1, layout_q="TND", layout_k="TND",
) -> torch.Tensor:
    ...
```

模块路径为 `quant_sparse_lightning_indexer_metadata.quant_sparse_lightning_indexer_metadata`。

## 参数说明

`B` 为 batch 数，`T1` 为 Query 存储 token 总数。下表描述与 QSLI 配套使用的接口。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `candidate_block_length` | Tensor | 必选 | 每个 Query 的有效候选块数量；PA 也接受 `[T1]`。 | int32 | `[T1, 1]` |
| `cu_seqlens_q / cu_seqlens_k` | Tensor | 可选 | 带前导零的 token 前缀和，TND 必须同时提供。 | int32 | `[B + 1]` |
| `seqused_q / seqused_k` | Tensor | 可选 | 各 batch 的实际有效长度。 | int32 | `[B]` |
| `cmp_residual_k` | Tensor | 可选 | causal 压缩场景的长度残差。 | int32 | `[B]` |
| `batch_size` | int | 可选 | 默认 `None`；PA 可显式传入并须与长度输入一致；TND 保持 `None`。 | - | - |
| `max_seqlen_q / max_seqlen_k` | int | 可选 | 最大单 batch 长度，默认 `-1`；K 长度是逻辑上下文上界，不是候选 token 数。 | - | - |
| `num_heads_q / num_heads_k / head_dim` | int | 必选 | 固定为 `32 / 1 / 128`。 | - | - |
| `topk / quant_mode / candidate_block_size` | int | 必选 | 与主算子一致：TopK `[1,8192]`、量化模式 `1`、块大小 `8`。 | - | - |
| `mask_mode / cmp_ratio` | int | 可选 | 默认 `0 / 1`；支持 mask `0/3`，压缩比例 `[1,128]`。 | - | - |
| `layout_q / layout_k` | str | 可选 | 默认 `TND / TND`；Q 为 TND，K 为 TND 或 PA_BBND。 | - | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `metadata` | Tensor | 必选 | 当前 NPU 上连续的 LI/LD 调度信息。 | int32 | `[1024]` |

每条记录有 8 个字段：第 0–31 条为 LI，第 32–35 条保留，第 36 条起为 LD/FD 归并记录，未使用区域清零。记录号从 0 开始；该布局是 QSLI 私有 ABI，不应手工修改，也不能使用 QLI metadata 替代。

## 约束说明

- 候选长度范围为 `[0,2048]`，表示候选数组的有效前缀，不是 token 数或 PA 页数。
- 各 Tensor 为同一 NPU 上的连续 int32 数据；序列长度非负，前缀和从零开始，有效长度不超过输入容量。
- PA 多 batch 需要 `cu_seqlens_q`，并提供 `max_seqlen_k` 或 `seqused_k`；TND 需要 Q/K 前缀和。
- 仅当 `mask_mode=3` 且 `cmp_ratio!=1` 时提供 `cmp_residual_k`，其他情况传 `None`。
- 配套主算子使用 32 个 Cube worker。候选长度、序列长度、布局、TopK、mask 和压缩参数必须一致。
- metadata 不读取候选块索引，不能替代调用方对候选索引合法性、唯一性和页表的保证；逐 token 的 causal 过滤由主算子执行。
- 优先加载相邻 `_aicpu/qsli_metadata_kernel.so`；未预打包时首次调用编译，后续复用进程内缓存。在当前 NPU stream 发射，内部 scratch 记录流生命周期。

### metadata 复用

按同一 stream 顺序先生成 metadata，再调用 QSLI。候选长度、Q/K 有效长度、布局、TopK、mask、压缩参数和设备配置不变时可以复用；仅候选块号或 Q/K/权重数值变化不影响任务划分。每次调用仍须保证输入合法；调度输入变化后必须重新生成。

## 调用示例

配置 CANN 环境及仓库 `samples` 导入路径后执行：

```python
import torch
import torch_npu

from quant_sparse_lightning_indexer_metadata.quant_sparse_lightning_indexer_metadata import (
    quant_sparse_lightning_indexer_metadata,
)

device = "npu:0"
torch.npu.set_device(device)
candidate_length = torch.full((6, 1), 128, dtype=torch.int32, device=device)
cu_q = torch.tensor([0, 6], dtype=torch.int32, device=device)
used_q = torch.tensor([6], dtype=torch.int32, device=device)
used_k = torch.tensor([1024], dtype=torch.int32, device=device)
metadata = quant_sparse_lightning_indexer_metadata(
    candidate_length, cu_seqlens_q=cu_q, seqused_q=used_q, seqused_k=used_k,
    max_seqlen_q=6, max_seqlen_k=1024,
    num_heads_q=32, num_heads_k=1, head_dim=128,
    topk=512, quant_mode=1, candidate_block_size=8,
    mask_mode=0, cmp_ratio=1, layout_q="TND", layout_k="PA_BBND",
)
torch.npu.synchronize()
assert metadata.shape == (1024,) and metadata.dtype == torch.int32
```

主算子接收 `quant_sparse_lightning_indexer(..., metadata=metadata)`，完整调用见 [QSLI 快速开始](../quant_sparse_lightning_indexer/README.md#快速开始)。

## 精度测试

在仓库根目录执行独立 metadata 测试，不加载 QSLI 主计算算子：

```bash
python -m pytest -q test/quant_sparse_lightning_indexer_metadata/test_quant_sparse_lightning_indexer_metadata.py
```

共 6 项测试：1 项 CPU 参考校验、4 项 NPU 调度边界、1 项 NPU 典型调度用例。覆盖 PA/TND、变长、causal、跨核 LD 和零工作量；独立检查逻辑候选 tile 无遗漏或重复、LD 分片数与 Query 行覆盖、工作区不重叠和保留区清零。典型配置为 B=12、S1=6、Nq/Nk=32/1、D=128、S2=128K、TopK=512、候选块数量/大小=2048/8、mask_mode=3、cmp_ratio=2。典型用例标记 `slow`，默认执行；最终 TopK 数值由主算子测试验证。
