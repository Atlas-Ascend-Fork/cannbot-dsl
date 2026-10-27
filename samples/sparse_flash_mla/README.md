# Sparse Flash MLA

基于 CANNBotDSL 实现的非量化 Sparse Flash MLA（SMLA）算子，面向 Ascend 950
（NPU ARCH 3510）。公共接口 `sparse_flash_mla()` 根据输入自动选择 N64 稀疏分支
或通用分支，完成 QK、在线 Softmax、PV 累加及输出写回。

如需复现，请联系马亚飞（gitcode账号：mayafei）获取最新cannbot-dsl whl包。

## 算子介绍

SMLA 对 Query 与原始 KV、可选压缩 KV 执行融合 Attention。
`sinks` 为每个 Query head 提供 FP32 Softmax sink。

| 模式 | 输入组合 |
| :--- | :--- |
| SWA | 无原始 KV 稀疏索引，且无 CMP KV |
| HCA | 无原始 KV 稀疏索引，有 CMP KV，无 CMP 稀疏索引 |
| CSA | 无原始 KV 稀疏索引，有 CMP KV 和 CMP 稀疏索引 |
| ORI_SPARSE | 有原始 KV 稀疏索引，无 CMP KV |
| ORI_CMP_SPARSE | 有原始 KV 稀疏索引、CMP KV 和 CMP 稀疏索引 |

### 特性与约束

| 项目 | 通用分支 | N64 稀疏分支 |
| :--- | :--- | :--- |
| 数据类型 | FP16/BF16 | FP16/BF16，Q/KV dtype 一致 |
| Query head | 1～128 | 1～64 |
| KV head / head dim | N2=1 / D=512 | N2=1 / D=512 |
| Query layout | BSND、TND | BSND、TND |
| KV layout | BSND、TND、PA/PA_BBND | 与 Q 相同，或 PA/PA_BBND |
| 模式 | SWA、HCA，以及 N1>64 的稀疏模式 | ORI_SPARSE、ORI_CMP_SPARSE、CSA |
| metadata | 必填，INT32 `[1024]` | 必填，INT32 `[1024]` |
| sinks | 必填，FP32 `[N1]` | 可选，提供时为同设备连续 FP32 `[N1]` |
| LSE | 不支持，须设置为 False | 支持 |

公共接口已取消高性能分支路由。所有满足接口约束的 `1 <= N1 <= 64` 稀疏输入，
包括原高性能 mask 00 用例，均进入 `sparse_flash_mla_sparse_n64_kernel.py`。
SWA、HCA 和 `N1>64` 稀疏输入使用通用分支。各分支仍执行自己的完整输入校验。

### Metadata 调用关系

两个分支均使用独立 AICPU 算子 `sparse_flash_mla_metadata()` 生成的 INT32 `[1024]`
metadata。公共接口不会隐式生成 metadata。metadata 参数应与主算子的 layout、长度、
mask、窗口、head 数和压缩配置一致，并在同一 NPU 流上依次提交；跨流依赖由调用方建立。

## 快速开始

确保已安装 CANNBotDSL、PyTorch 和 torch_npu，并加载实际 CANN 环境。
在仓库根目录设置：

```bash
source /path/to/Ascend/cann/set_env.sh
export PYTHONPATH="$PWD/samples/sparse_flash_mla:$PWD/samples/sparse_flash_mla_metadata:$PYTHONPATH"
```

下面通过公共接口执行 N64 稀疏 ORI_CMP_SPARSE：TND 单段、BF16、mask 00、
`cmp_ratio=4`、无 residual，满足 `ORI长度 = cmp_ratio × CMP长度`。
此输入先生成 metadata；无压缩残留时不需要显式传入 residual 计数。

```python
import math
import torch
import torch_npu

from sparse_flash_mla import sparse_flash_mla
from sparse_flash_mla_metadata import sparse_flash_mla_metadata

torch.npu.set_device(0)
torch.manual_seed(0)

T, N1, D = 7, 32, 512
cmp_ratio, cmp_length = 4, 256
ori_length = cmp_ratio * cmp_length
K = 128
device = "npu:0"

q = torch.randn(T, N1, D, dtype=torch.bfloat16, device=device)
ori_kv = torch.randn(ori_length, 1, D, dtype=q.dtype, device=device)
cmp_kv = torch.randn(cmp_length, 1, D, dtype=q.dtype, device=device)
ori_indices = torch.arange(K, dtype=torch.int32, device=device).view(1, 1, K).repeat(T, 1, 1)
cmp_indices = torch.arange(K, dtype=torch.int32, device=device).view(1, 1, K).repeat(T, 1, 1)
ori_topk_length = torch.full((T,), K, dtype=torch.int32, device=device)
cmp_topk_length = torch.full((T,), K, dtype=torch.int32, device=device)
sinks = torch.zeros(N1, dtype=torch.float32, device=device)

cu_q = torch.tensor([0, T], dtype=torch.int32, device=device)
cu_ori = torch.tensor([0, ori_length], dtype=torch.int32, device=device)
cu_cmp = torch.tensor([0, cmp_length], dtype=torch.int32, device=device)
metadata = sparse_flash_mla_metadata(
    cu_seqlens_q=cu_q, cu_seqlens_ori_kv=cu_ori, cu_seqlens_cmp_kv=cu_cmp,
    num_heads_q=N1, num_heads_kv=1, head_dim=D,
    batch_size=1, max_seqlen_q=T,
    max_seqlen_ori_kv=ori_length, max_seqlen_cmp_kv=cmp_length,
    ori_topk=K, cmp_topk=K, cmp_ratio=cmp_ratio,
    ori_topk_length=ori_topk_length, cmp_topk_length=cmp_topk_length,
    ori_mask_mode=0, cmp_mask_mode=0,
    layout_q="TND", layout_kv="TND", has_ori_kv=True, has_cmp_kv=True,
)

out, lse = sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    ori_sparse_indices=ori_indices,
    cmp_sparse_indices=cmp_indices,
    ori_topk_length=ori_topk_length,
    cmp_topk_length=cmp_topk_length,
    sinks=sinks,
    metadata=metadata,
    cu_seqlens_q=cu_q,
    cu_seqlens_ori_kv=cu_ori,
    cu_seqlens_cmp_kv=cu_cmp,
    softmax_scale=1.0 / math.sqrt(D),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    layout_q="TND",
    layout_kv="TND",
    return_softmax_lse=True,
)
torch.npu.synchronize()

# out: BF16 [T,N1,512]
# lse: FP32 [1,T,N1]
```

## 接口

```python
def sparse_flash_mla(
    query_gm,
    *,
    ori_kv,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False,
):
    ...
```

### 输入与输出

| 参数/返回值 | 说明 |
| :--- | :--- |
| `query_gm` | Query，最后两维为 `[N1,512]` |
| `ori_kv/cmp_kv` | 原始/可选压缩 KV，最后两维为 `[1,512]`，dtype 与 Query 一致 |
| `ori_sparse_indices/cmp_sparse_indices` | 原始/压缩 KV 稀疏索引，INT32，最后一维为正数 K |
| `*_block_table` | 通用 PA/PA_BBND 页表，INT32 `[B,max_pages]` |
| `cu_seqlens_*` | TND 路径的物理前缀和，含前导 0 |
| `seqused_*` | 每 batch 的 Query、ORI/CMP KV 有效长度 |
| `cmp_residual_kv` | 每 batch 的压缩 residual 长度；压缩 causal 且 ratio≠1 时按接口约束提供 |
| `*_topk_length` | 每 Query 有效稀疏长度；mask=0 的对应稀疏分支必须提供 |
| `sinks` | 每个 Query head 的 FP32 Softmax sink，shape `[N1]`；N64 稀疏分支支持 `None` |
| `metadata` | 两个分支均需 AICPU metadata，INT32 `[1024]` |
| `softmax_scale` | 通常为 `1/sqrt(512)`；要求有限实数 |
| `cmp_ratio` | 压缩比，范围 1～128；无 CMP 时必须为 1 |
| `ori_mask_mode` | 0=全注意力，3=right-down causal，4=滑动窗口 |
| `cmp_mask_mode` | 0=全注意力，3=right-down causal；无 CMP 时必须为 0 |
| `ori_win_left/right` | 仅 ORI mask=4 使用；其他模式为 -1 |
| `layout_q/layout_kv` | Query 为 BSND/TND，KV 为 BSND/TND/PA/PA_BBND；分支限制见上文 |
| `topk_value_mode` | 固定为 1 |
| 返回 `(out,lse)` | out 与 Query shape/dtype 相同；N64 稀疏分支开启 LSE 时，TND 返回 FP32 `[1,T,N1]`，BSND 返回 FP32 `[B,1,S,N1]`，否则为空 FP32 张量 `(0,)` |

### 一致性要求

1. 所有参与计算的张量位于同一 NPU。非 PA 模式下 Q/KV layout 一致。
2. 通用 TND 路径提供对应 `cu_seqlens_*`；通用 PA/PA_BBND 提供页表与有效长度。
3. metadata 参数与主算子的 layout、长度、mask、窗口、head 数和压缩配置一致。
4. 有 CMP 时，调用方应保证有效长度满足
   `ORI长度 = cmp_ratio × CMP长度 + residual`。无 residual 时取 residual=0；
   这是输入语义要求，不代表 host 已读取张量内容并完整验证该等式。
5. 有效 top-k 前缀中的稀疏索引必须指向有效 KV 行，top-k 长度不得超过对应索引容量。
   `-1` sentinel 仅允许放在有效前缀之外，不能计入有效 top-k 长度。
   host 不逐元素验证索引及长度内容，调用方负责保证合法。
6. 无 CMP 的调用不传 CMP 索引、长度、页表、前缀和、有效长度或 residual 等附属参数。
