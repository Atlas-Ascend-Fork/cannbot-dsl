# Sparse Flash MLA Metadata

通过 NPU 的 AICPU 执行 Sparse Flash MLA 分核，生成主算子通用分支使用的
`int32[1024]` metadata。公开参数与 ops-transformer 的
`SparseFlashMlaMetadata` 对齐，当前目标仅支持 Ascend 950/A5。

## 算子介绍

metadata 算子根据 Query/KV 的 layout、有效长度、head 配置、mask、窗口和压缩参数，
将连续 `(batch × KV head, query group)` 范围分配给当前设备的有效 AIC 核。主算子
的通用分支读取每个核槽的起止坐标，只处理分配给自己的 Query 范围。
高性能分支不消费 metadata；公共 SMLA 接口不会隐式调用本算子。

当前实现与配套 SMLA Kernel 保持一致：

- 支持 SWA、HCA、CSA、ORI_SPARSE、ORI_CMP_SPARSE 所需的 FA 分核；
- FlashDecode 路径关闭；
- batch consistency 路径关闭，`is_batch_consistency=True` 会报错；
- 在当前 NPU 流异步启动，不将输入复制回 CPU。

## 接口

```python
def sparse_flash_mla_metadata(
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    *,
    num_heads_q,
    num_heads_kv=1,
    head_dim=512,
    batch_size=0,
    max_seqlen_q=0,
    max_seqlen_ori_kv=0,
    max_seqlen_cmp_kv=0,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="PA_BBND",
    has_ori_kv=True,
    has_cmp_kv=True,
    soc_version="Ascend950",
    aic_core_num=None,
    aiv_core_num=None,
    is_batch_consistency=False,
    device=None,
):
    ...
```

### 输入要求

| 参数 | 说明 |
| :--- | :--- |
| `cu_seqlens_q` | TND Query 的物理前缀和，连续一维 NPU INT32，包含前导 0 |
| `cu_seqlens_ori_kv` | TND 原始 KV 的物理前缀和 |
| `cu_seqlens_cmp_kv` | TND 压缩 KV 的物理前缀和 |
| `seqused_q` | 每 batch 实际 Query 长度；提供时用于确定 batch 数 |
| `seqused_ori_kv` | 每 batch 实际原始 KV 长度；PA/PA_BBND 应提供 |
| `seqused_cmp_kv` | 每 batch 实际压缩 KV 长度 |
| `cmp_residual_kv` | 压缩 causal 模式的 residual 长度 |
| `ori_topk_length` | 原始 KV 稀疏 TopK 有效长度 |
| `cmp_topk_length` | 压缩 KV 稀疏 TopK 有效长度 |
| `num_heads_q/kv` | 均为正数，且 `num_heads_q` 必须是 `num_heads_kv` 的整数倍 |
| `head_dim` | 配套 SMLA Kernel 固定为 512 |
| `batch_size` | 无 `seqused_q/cu_seqlens_q` 时使用；必须非负 |
| `max_seqlen_*` | 对应输入的最大有效长度 |
| `ori_topk/cmp_topk` | 对应原始/压缩稀疏索引的容量 K；非稀疏分支设为 0 |
| `cmp_ratio` | 压缩比，必须为正数 |
| `ori_mask_mode` | 0=全注意力，3=right-down causal，4=滑动窗口 |
| `cmp_mask_mode` | 0=全注意力，3=right-down causal |
| `layout_q/layout_kv` | 与主算子调用保持一致；支持 BSND、TND、PA/PA_BBND 等枚举 |
| `has_ori_kv/has_cmp_kv` | 标识对应 KV 输入是否存在 |
| `aic_core_num/aiv_core_num` | 缺省读取当前设备核数；显式值不得超过 36/72 |
| `device` | 可选 NPU device id；缺省为当前 device |

所有非空 Tensor 输入必须满足：

- 位于指定的当前 NPU；
- dtype 为 `torch.int32`；
- 一维且连续。

batch 数按以下优先级确定：`seqused_q.numel()` →
`cu_seqlens_q.numel() - 1` → `batch_size`。

有 CMP 时，调用方应保证有效长度满足
`ORI长度 = cmp_ratio × CMP长度 + residual`；无 residual 时取 residual=0。
该关系是输入语义要求，不代表本接口已逐元素验证。
本算子自身的参数校验范围不等于主算子的支持范围；配套主算子的 head 数、
数据类型、layout 等限制以 SMLA README 为准。

### 输出 ABI

输出为当前 NPU 上连续的 INT32 `[1024]`：

```text
FA 区      36 × 9 = 324 int32
FD 区      72 × 8 = 576 int32
对齐填充              124 int32
总计                 1024 int32
```

FA 每核槽记录 enable、BN2/M/S2 起点、BN2/M/S2 终点和 FlashDecode workspace
信息。有效核槽按核号组成连续前缀，其 `(BN2,M)` 范围左闭右开、首尾相接并完整覆盖
任务域；inactive 槽清零。当前 FlashDecode 和 batch consistency 均关闭，因此 FD 区、
S2 分片字段及末尾 padding 保持为零。

36 AIC / 72 AIV 是 metadata 二进制 ABI 的固定容量上限。实际参与核数从当前设备读取，
不能用实际核数改变槽位步长或输出总长度。

## 快速开始

```python
import torch
import torch_npu

from sparse_flash_mla_metadata import sparse_flash_mla_metadata

torch.npu.set_device(0)
cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32).npu()
seqused_ori_kv = torch.tensor([1024], dtype=torch.int32).npu()

metadata = sparse_flash_mla_metadata(
    cu_seqlens_q=cu_seqlens_q,
    seqused_ori_kv=seqused_ori_kv,
    num_heads_q=64,
    num_heads_kv=1,
    head_dim=512,
    batch_size=1,
    max_seqlen_q=1,
    max_seqlen_ori_kv=1024,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=0,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="PA_BBND",
    has_ori_kv=True,
    has_cmp_kv=False,
)

# metadata: torch.int32，[1024]，位于当前 NPU
```

调用完成后，将结果通过 `sparse_flash_mla(..., metadata=metadata)` 传给主算子通用分支。
两次调用的 layout、长度、mask、窗口、head 数、压缩比和 KV 存在性必须一致。

## 测试

当前仓库使用 `test/sparse_flash_mla/test_sparse_flash_mla.py` 进行集成测试，先调用本算子生成
metadata，再运行 SMLA 并与 PyTorch reference 比较：

```bash
pytest test/sparse_flash_mla/test_sparse_flash_mla.py -v
```

该测试覆盖 metadata 与主算子的集成调用。
当前脚本只执行 SWA/HCA/CSA，用例参数表中的 ORI_SPARSE/ORI_CMP_SPARSE 会被跳过。
