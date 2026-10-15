# Sparse Flash MLA

基于 CANNBotDSL 实现的非量化 Sparse Flash MLA（SMLA）算子，面向 Ascend 950
（NPU ARCH 3510）。公共接口 `sparse_flash_mla()` 根据输入自动选择高性能分支
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

| 项目 | 通用分支 | 高性能分支 |
| :--- | :--- | :--- |
| 数据类型 | FP16/BF16，Q、KV、输出一致 | 连续 BF16 Q/KV |
| Query head | 1～128 | 1～64 |
| KV head / head dim | N2=1 / D=512 | N2=1 / D=512 |
| Query layout | BSND、TND | BSND、TND，与 KV layout 相同 |
| KV layout | BSND、TND、PA/PA_BBND，page size 1～1024 | 非分页 BSND/TND，按完整 KV 存储处理 |
| batch / Query | 支持多 batch | BSND 限 B=1；TND 按单段处理；Query 非空 |
| ORI 稀疏索引 | 可选 | 必填 |
| CMP | 可选，支持稠密或稀疏 CMP | 可选；有 CMP 时必须提供 CMP 稀疏索引 |
| metadata | 必填，INT32 `[1024]` | 不依赖 metadata |
| sinks | 必填，FP32 `[N1]` | 必填，连续 FP32 `[N1]` |
| LSE | 不支持，须设置为 False | 支持 |

高性能分支还要求：

- 所有参与计算的张量位于同一 NPU，索引和长度为连续 INT32。
- 不使用页表和 `seqused_*`。BSND 不传 `cu_seqlens_*`；TND 调用方须保证
  数据属于单段完整存储，不能把多 batch 拼接数据当作单段输入。
- 常规模式为 ORI/CMP mask 00，窗口 `(-1,-1)`。必须提供 ORI top-k 长度，
  存在 CMP 时也必须提供 CMP top-k 长度。
- TND 索引形状为 `[T,K]` 或 `[T,1,K]`，长度为 `[T]` 或 `[T,1]`；
  BSND 索引为 `[1,S,K]` 或 `[1,S,1,K]`，长度为 `[1,S]` 或 `[1,S,1]`。
- 另支持受限 legacy 模式：ORI mask=4，有 CMP 时 CMP mask=3，无 CMP 时为 0，
  窗口固定 `(127,0)`，不传 top-k 长度。ORI 稀疏容量至少为 128 且为 128 的倍数，
  CMP 稀疏容量也为 128 的倍数，并需满足完整因果覆盖条件。
- `softmax_scale` 为有限正数，`topk_value_mode=1`。

不满足高性能条件时，公共接口转入通用分支，输入仍需满足通用分支要求，
包括提供匹配的 metadata。回退不保证任意输入都能执行；请求 LSE 后若回退通用分支，
会因通用分支不支持 LSE 而报错。

### Metadata 调用关系

通用分支使用独立 AICPU 算子 `sparse_flash_mla_metadata()` 生成的
INT32 `[1024]` metadata。公共接口不会隐式生成 metadata。
metadata 与通用主 Kernel 应在同一 NPU 流上依次提交；跨流调用由调用方建立依赖。
高性能分支使用自身的任务划分，不消费 metadata。

## 快速开始

确保已安装 CANNBotDSL、PyTorch 和 torch_npu，并加载实际 CANN 环境。
在仓库根目录设置：

```bash
source /path/to/Ascend/cann/set_env.sh
export PYTHONPATH="$PWD/samples/sparse_flash_mla:$PWD/samples/sparse_flash_mla_metadata:$PYTHONPATH"
```

下面通过公共接口执行高性能 ORI_CMP_SPARSE：TND 单段、BF16、mask 00、
`cmp_ratio=4`、无 residual，满足 `ORI长度 = cmp_ratio × CMP长度`。
此输入不需要 metadata，也不需要显式传入 residual 计数。

```python
import math
import torch
import torch_npu

from sparse_flash_mla import sparse_flash_mla

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

out, lse = sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    ori_sparse_indices=ori_indices,
    cmp_sparse_indices=cmp_indices,
    ori_topk_length=ori_topk_length,
    cmp_topk_length=cmp_topk_length,
    sinks=sinks,
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
# lse: FP32 [T,N1]
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
| `cu_seqlens_*` | 通用 TND 路径的物理前缀和，含前导 0 |
| `seqused_*` | 通用分支每 batch 有效长度 |
| `cmp_residual_kv` | residual 长度；通用压缩 causal 且 ratio≠1 时必填。高性能不强制提供；显式提供时检查为同设备连续 INT32 `[1]`，其数值不参与高性能内核计算 |
| `*_topk_length` | 每 Query 有效稀疏长度；mask=0 的对应稀疏分支必须提供 |
| `sinks` | 每个 Query head 的 FP32 Softmax sink，shape `[N1]` |
| `metadata` | 通用分支所需 AICPU metadata，INT32 `[1024]` |
| `softmax_scale` | 通常为 `1/sqrt(512)`；高性能要求有限正数，通用要求有限数 |
| `cmp_ratio` | 压缩比，范围 1～128；无 CMP 时必须为 1 |
| `ori_mask_mode` | 0=全注意力，3=right-down causal，4=滑动窗口；高性能范围见上文 |
| `cmp_mask_mode` | 0=全注意力，3=right-down causal；无 CMP 时必须为 0 |
| `ori_win_left/right` | 仅 ORI mask=4 使用；其他模式为 -1 |
| `layout_q/layout_kv` | Query 为 BSND/TND，KV 为 BSND/TND/PA/PA_BBND；分支限制见上文 |
| `topk_value_mode` | 固定为 1 |
| 返回 `(out,lse)` | out 与 Query shape/dtype 相同；高性能开启 LSE 时返回 shape 为 `query.shape[:-1]` 的 FP32 张量，否则为空 FP32 张量 |

### 一致性要求

1. 所有参与计算的张量位于同一 NPU。非 PA 模式下 Q/KV layout 一致。
2. 通用 TND 路径提供对应 `cu_seqlens_*`；通用 PA/PA_BBND 提供页表与有效长度。
3. 通用 metadata 参数与主算子的 layout、长度、mask、窗口、head 数和压缩配置一致。
4. 有 CMP 时，调用方应保证有效长度满足
   `ORI长度 = cmp_ratio × CMP长度 + residual`。无 residual 时取 residual=0；
   这是输入语义要求，不代表 host 已读取张量内容并完整验证该等式。
5. 稀疏索引指向有效 KV 行，top-k 长度不得超过对应索引容量。
   高性能 host 不逐元素验证索引及长度内容，调用方负责保证合法。
6. 无 CMP 的高性能调用不传 CMP 索引、长度、页表、前缀和、有效长度或 residual 等附属参数。

## 测试

当前仓库集成测试位于 `test/sparse_flash_mla/test_sparse_flash_mla.py`，
用例定义在同目录的 `sparse_flash_mla_paramset.py`。默认执行 `ENABLED_PARAMS`，
也可通过 `SMLA_CASES` 从 `TEST_PARAMS` 选择用例：

```bash
pytest test/sparse_flash_mla/test_sparse_flash_mla.py -v

SMLA_CASES=hca_tnd,csa_decode \
pytest test/sparse_flash_mla/test_sparse_flash_mla.py -v
```

该测试先生成 AICPU metadata，再调用公共 SMLA 接口并与 CPU reference 比较，
同时覆盖 metadata 与主算子的集成调用。
当前测试脚本仍只执行 SWA/HCA/CSA，跳过 ORI_SPARSE/ORI_CMP_SPARSE；
这是测试覆盖限制，不代表算子不支持原始 KV 稀疏输入。
