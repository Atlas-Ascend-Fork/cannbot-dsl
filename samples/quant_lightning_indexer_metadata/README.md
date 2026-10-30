# quant_lightning_indexer_metadata

## 产品支持情况

- Ascend 950PR / Ascend 950DT：支持。

## 功能说明

`quant_lightning_indexer_metadata` 是 [QLI](../quant_lightning_indexer/README.md) 配套的独立 AICPU 调度算子。它根据有效序列长度、mask、压缩比例和 TopK 生成 LI 计算范围及 LD 跨核归并信息，不读取 Q/K 特征，不计算分数或 TopK。

Query 按最多 6 行分组，Key 按 256-token tile 划分。causal 场景根据有效长度、`cmp_ratio` 和 `cmp_residual_k` 推导右下对齐的可见范围，再估算各 Query group 的计算量。LI 记录每核在 `(batch, query_group, s2_tile)` 坐标空间中的连续起止范围；同一 Query group 被拆到多个核时，LD 记录分片数、临时结果位置和输出行范围，供主算子归并局部 TopK。

零工作量保留必要的输出初始化调度。实现位于 [quant_lightning_indexer_metadata.py](quant_lightning_indexer_metadata.py)。

## 函数原型

```python
def quant_lightning_indexer_metadata(
    cu_seqlens_q=None, cu_seqlens_k=None,
    seqused_q=None, seqused_k=None, cmp_residual_k=None,
    *, batch_size=None, max_seqlen_q=-1, max_seqlen_k=-1,
    num_heads_q, num_heads_k, head_dim, topk,
    mask_mode=0, cmp_ratio=1, layout_q="TND", layout_k="TND",
    candidate_topk_blocks=-1, candidate_block_size=-1,
) -> torch.Tensor:
    ...
```

模块路径为 `quant_lightning_indexer_metadata.quant_lightning_indexer_metadata`。

## 参数说明

`B` 为 batch 数，`T1` 为 Query 存储 token 总数。下表描述与 QLI 配套使用的接口。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `cu_seqlens_q / cu_seqlens_k` | Tensor | 可选 | 带前导零的 token 前缀和；TND 用于定位序列。 | int32 | `[B + 1]` |
| `seqused_q / seqused_k` | Tensor | 可选 | 各 batch 的实际有效长度，提供时优先使用。 | int32 | `[B]` |
| `cmp_residual_k` | Tensor | 可选 | causal 压缩场景的长度残差。 | int32 | `[B]` |
| `batch_size` | int | 可选 | 默认 `None`；PA 可显式传入并须与长度输入一致；TND 保持 `None`。 | - | - |
| `max_seqlen_q / max_seqlen_k` | int | 可选 | 最大单 batch 长度，默认 `-1`；需有足够的序列输入推导缺省长度。 | - | - |
| `num_heads_q / num_heads_k / head_dim` | int | 必选 | 固定为 `32 / 1 / 128`，D 为逻辑维度。 | - | - |
| `topk` | int | 必选 | 与主算子一致，范围 `[1,8192]`。 | - | - |
| `mask_mode / cmp_ratio` | int | 可选 | 默认 `0 / 1`；支持 mask `0/3`，压缩比例 `[1,128]`。 | - | - |
| `layout_q / layout_k` | str | 可选 | 默认 `TND / TND`；Q 为 TND，K 为 TND 或 PA_BBND。 | - | - |
| `candidate_topk_blocks / candidate_block_size` | int | 可选 | 默认 `-1 / -1`，不输出候选；开启候选时与主算子使用相同正容量及块大小 `8`。 | - | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `metadata` | Tensor | 必选 | 当前 NPU 上连续的 LI/LD 调度信息。 | int32 | `[1024]` |

每条记录有 8 个字段：第 0–31 条为 LI，第 32–35 条保留，第 36 条起为 LD/FD 归并记录，未使用区域清零。记录号从 0 开始；调用方应将输出原样传给主算子，不手工构造或修改私有布局。

## 约束说明

- 长度 Tensor 为同一 NPU 上连续的一维 int32 数据；有效长度非负且不超过输入容量，前缀和从零开始。
- PA 多 batch 必须提供 `cu_seqlens_q`，并提供 `max_seqlen_k` 或 `seqused_k`；TND 多 batch 还需 `cu_seqlens_k`。
- 仅当 `mask_mode=3` 且 `cmp_ratio!=1` 时提供 `cmp_residual_k`，其他情况传 `None`。
- `cu_seqlens_q` 表示存储分段，`seqused_q` 可小于分段长度；超出有效 Query 长度的行由主算子输出无效项。
- 配套主算子使用 32 个 Cube worker。布局、长度、TopK、mask、压缩参数及候选配置必须一致；这里的调用契约不表示 Host 会检查所有非法设备数据。
- 优先加载相邻 `_aicpu/metadata_kernel.so`；未预打包时首次调用编译，后续复用进程内缓存。在当前 NPU stream 发射，内部 scratch 记录流生命周期。

### metadata 复用

按同一 stream 顺序先生成 metadata，再调用 QLI。调度输入和设备配置不变时可以复用；仅 Q/K/权重数值变化不影响任务划分。长度或静态属性变化后必须重新生成，不能使用 QSLI metadata 替代。

## 调用示例

配置 CANN 环境及仓库 `samples` 导入路径后执行：

```python
import torch
import torch_npu

from quant_lightning_indexer_metadata.quant_lightning_indexer_metadata import (
    quant_lightning_indexer_metadata,
)

device = "npu:0"
torch.npu.set_device(device)
cu_q = torch.tensor([0, 6], dtype=torch.int32, device=device)
used_q = torch.tensor([6], dtype=torch.int32, device=device)
used_k = torch.tensor([1024], dtype=torch.int32, device=device)
metadata = quant_lightning_indexer_metadata(
    cu_seqlens_q=cu_q, seqused_q=used_q, seqused_k=used_k,
    max_seqlen_q=6, max_seqlen_k=1024,
    num_heads_q=32, num_heads_k=1, head_dim=128, topk=512,
    mask_mode=0, cmp_ratio=1, layout_q="TND", layout_k="PA_BBND",
)
torch.npu.synchronize()
assert metadata.shape == (1024,) and metadata.dtype == torch.int32
```

主算子接收 `quant_lightning_indexer(..., metadata=metadata)`，完整调用见 [QLI 快速开始](../quant_lightning_indexer/README.md#快速开始)。

## 精度测试

在仓库根目录执行独立 metadata 测试，不加载 QLI 主计算算子：

```bash
python -m pytest -q test/quant_lightning_indexer_metadata/test_quant_lightning_indexer_metadata.py
```

共 7 项测试：1 项 CPU 参考校验、4 项 NPU 调度边界、2 项 NPU 典型调度用例。覆盖 PA/TND、变长尾部、causal、跨核 LD 和零工作量；独立枚举任务，检查每个 S2 tile 恰好出现一次、LD 分片数与 Query 行覆盖、工作区不重叠及保留区清零。典型配置为 B=12、S1=6、Nq/Nk=32/1、D=128、S2=64K/128K、TopK=512、mask_mode=3、cmp_ratio=2。典型用例标记 `slow`，默认执行；最终 TopK 数值由主算子测试验证。
