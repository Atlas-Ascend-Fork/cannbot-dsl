# Quant Sparse Lightning Indexer

基于 CANNBotDSL 实现的 MXFP4 Sparse Lightning Indexer（QSLI），仅访问每个
Query 指定的候选 Key 块，计算加权相关性并选择 TopK token，面向 Ascend NPU。

## 算子介绍

支持 Ascend 950PR / Ascend 950DT。

QSLI 用于稀疏 Attention 的 token 索引选择。与遍历上下文的 QLI 不同，
它只读取候选逻辑块覆盖的 Key；候选可由 QLI 输出，也可由调用方提供。
输入 Q/K 已完成 MXFP4 量化，算子不执行输入量化、Key 压缩、Softmax 或 Attention 的 Value 加权和。

候选集合与计算公式：

对 Query token $t$，记候选数组的有效前缀为 $c_{t,0},\ldots,c_{t,C_t-1}$，
其中 $C_t=$ `candidate_block_length[t]`，每个候选块包含 8 个逻辑 token。
当前 Query 的计算集合为：

$$
\Omega_t=\left(\bigcup_{p=0}^{C_t-1}\{8c_{t,p},\ldots,8c_{t,p}+7\}\right)
\cap\mathcal V_t.
$$

$\mathcal V_t$ 同时约束序列有效长度和 causal 可见范围，候选前缀外的索引不参与计算。
令 $Q^{(4)},K^{(4)}$ 为解码后的 FP4 E2M1 数值，$S^Q,S^K$ 为每 32 个 D 轴元素一组的 E8M0 scale，
则 $g=32,D=128$ 时：

$$
R_{t,h,j}=\sum_{a=0}^{3}S^Q_{t,h,a}S^K_{j,a}
\sum_{d=32a}^{32a+31}Q^{(4)}_{t,h,d}K^{(4)}_{j,d},
\qquad
s_{t,j}=\sum_{h=0}^{31}w_{t,h}\operatorname{ReLU}(R_{t,h,j}),
$$

$$
I_t=\operatorname{TopKIndices}_{j\in\Omega_t}(s_{t,j}),
\qquad V_t=s_{t,I_t}.
$$

主要计算过程为：

1. 从候选有效前缀得到逻辑 8-token 块号，结合 PA 页表或 TND 序列起点定位 K 与 scale。
2. 搬入候选 K/scale，计算分组反量化 QK 相关性，应用 ReLU。
3. 乘逐 Query/head 权重，并沿 32 个 Query head 归约，得到候选 token 的分数。
4. 去除超出有效长度或 causal 范围的 token，选择局部 TopK；跨核分片存在时归并到最终 TopK。
5. 将选择位置映射回原逻辑 token 索引，应用可选索引偏移，并按 `return_value` 返回分数。

QK 交接为 BF16，权重由 FP32 转为 BF16，逐 head 融合乘加按 BF16 舍入。
权重可为负，最终分数也可为负。无效项补 `indices=-1, values=-inf`，
有效索引表示原逻辑 token，而不是候选数组中的位置。

压缩 causal 与空行：

设 batch 的有效 Q/K 长度为 $L_Q,L_K$，Query 行号 $t$ 从 0 开始，
$r=$ `cmp_ratio`，$e=$ `cmp_residual_k`（$r=1$ 时取 0）。
`mask_mode=0` 时所有有效 K 可见；`mask_mode=3` 时：

$$
L_t=\min\left(L_K,\max\left(0,
\left\lfloor\frac{rL_K+e-L_Q+t+1}{r}\right\rfloor\right)\right),
\qquad \mathcal V_t=\{j\mid0\le j<L_t\}.
$$

当 $t\ge L_Q$、候选长度为 0 或交集为空时，输出整行无效项。
TopK 不足时填充尾部；`output_idx_offset` 只作用于有效索引。
[QSLI Metadata](../quant_sparse_lightning_indexer_metadata/README.md) 在 AICPU 上根据候选长度和序列参数生成 LI/LD 调度信息，
主 kernel 按该信息执行候选搬运、计算与归并。

| 特性与约束 | 说明 |
| :--- | :--- |
| 量化 | `quant_mode=1`；MXFP4 E2M1、E8M0 scale |
| 固定轴 | `Nq=32`、`Nk=1`、逻辑 `D=128`，打包末轴 64 |
| Layout | Q 为 `TND`；K 为 `PA_BBND` 或 `TND` |
| 候选容量 | 每行固定容量 2048，实际有效前缀长度为 `[0,2048]` |
| 候选块 | `candidate_block_size=8`，索引为逻辑 8-token 块号，不是 PA 页号 |
| PA 页大小 | 16 的倍数，范围 `(0,1024]`；典型验证为 64/128 |
| TopK / Mask | `topk` 范围 `[1,8192]`；`mask_mode=0/3`；`cmp_ratio` 范围 `[1,128]` |
| Metadata | 同设备连续 `int32[1024]`，必须显式提供 |

实现见 [quant_sparse_lightning_indexer.py](quant_sparse_lightning_indexer.py)。

PA Key 打包格式：

与 QLI 分开传 K、scale 不同，QSLI 的 PA 输入将每个 8-token 块打包为 544 字节：

```text
k[P, page_size/8, 544]
  [0:512]   8 个 token 的 MXFP4 K，每个 token 64 字节
  [512:544] 8 个 token 的 E8M0 scale，每个 token 4 字节
```

PA 路径不能另外传 `descale_k`。TND 路径则使用独立的
`k[T2,1,64]` 和 `descale_k[T2,1,2,2]`，且不接收 `block_table`。

## 快速开始

使用 CANN 9.2.0、CANNBotDSL 0.7.0、PyTorch 与 torch_npu；
PyTorch 需提供 `float8_e8m0fnu`，设备需支持 MXFP4。在仓库根目录执行：

```bash
source ${install_path}/ascend-toolkit/set_env.sh
export PYTHONPATH="$PWD/samples/quant_sparse_lightning_indexer:$PWD/samples/quant_sparse_lightning_indexer_metadata:$PYTHONPATH"
```

以下示例演示 PA 数据打包和完整调用；随机字节只用于接口演示，不代替模型量化或精度基准。

```python
import torch
import torch_npu
from quant_sparse_lightning_indexer import quant_sparse_lightning_indexer
from quant_sparse_lightning_indexer_metadata import quant_sparse_lightning_indexer_metadata

device = "npu:0"
torch.npu.set_device(device)
query_rows, key_tokens, page_size, topk = 6, 1024, 128, 512
pages = key_tokens // page_size
q = torch.randint(0, 256, (query_rows, 32, 64), dtype=torch.uint8).to(device)
w = torch.ones((query_rows, 32), dtype=torch.float32, device=device)
descale_q = torch.full((query_rows, 32, 2, 2), 127, dtype=torch.uint8, device=device)
key_bytes = torch.randint(0, 256, (pages, page_size, 1, 64), dtype=torch.uint8)
scale_bytes = torch.full((pages, page_size, 1, 2, 2), 127, dtype=torch.uint8)
k = torch.cat((key_bytes.reshape(pages, page_size // 8, 512),
               scale_bytes.reshape(pages, page_size // 8, 32)), dim=-1).to(device)
candidate = torch.full((query_rows, 1, 2048), -1, dtype=torch.int32, device=device)
valid_blocks = key_tokens // 8
candidate[:, 0, :valid_blocks] = torch.arange(valid_blocks, dtype=torch.int32, device=device)
candidate_length = torch.full((query_rows, 1), valid_blocks, dtype=torch.int32, device=device)
cu_q = torch.tensor([0, query_rows], dtype=torch.int32, device=device)
used_q = torch.tensor([query_rows], dtype=torch.int32, device=device)
used_k = torch.tensor([key_tokens], dtype=torch.int32, device=device)
block_table = torch.arange(pages, dtype=torch.int32, device=device).view(1, pages)

metadata = quant_sparse_lightning_indexer_metadata(
    candidate_length, cu_seqlens_q=cu_q, seqused_q=used_q, seqused_k=used_k,
    max_seqlen_q=query_rows, max_seqlen_k=key_tokens,
    num_heads_q=32, num_heads_k=1, head_dim=128,
    topk=topk, quant_mode=1, candidate_block_size=8,
    layout_q="TND", layout_k="PA_BBND",
)
indices, values = quant_sparse_lightning_indexer(
    q, k, w, descale_q, candidate, candidate_length,
    cu_seqlens_q=cu_q, seqused_q=used_q, seqused_k=used_k,
    block_table=block_table, metadata=metadata,
    topk=topk, candidate_block_size=8, quant_mode=1, max_seqlen_q=query_rows,
    layout_q="TND", layout_k="PA_BBND", return_value=True,
)
torch.npu.synchronize()
assert indices.shape == values.shape == (query_rows, 1, topk)
```

`quant_sparse_lightning_indexer()` 关键参数：

| 参数 | 类型 / shape | 说明 |
| :--- | :--- | :--- |
| `q` | UINT8 `[T1,32,64]` | 打包 MXFP4 Query |
| `k` | UINT8；PA `[P,page_size/8,544]`，TND `[T2,1,64]` | PA 内嵌 K scale，TND 单独提供 scale |
| `w` | FP32 `[T1,32]` | 逐 Query/head 权重 |
| `descale_q` | UINT8 或 E8M0 `[T1,32,2,2]` | Query 的 MX scale |
| `descale_k` | UINT8 或 E8M0 `[T2,1,2,2]` | 仅 TND 路径必填；PA 必须为 `None` |
| `candidate_block_indices` | INT32 `[T1,1,2048]` | 有效前缀内为合法、不重复的逻辑候选块号；后缀忽略 |
| `candidate_block_length` | INT32 `[T1,1]` | 每行候选块数；PA 也接受等元素数的一维长度输入 |
| `cu_seqlens_q / cu_seqlens_k` | INT32 `[B+1]` | 带前导零的 token 前缀和；TND 需同时提供 Q/K 前缀和 |
| `seqused_q / seqused_k` | INT32 `[B]` 或 `None` | 每个 batch 的实际可用 token 数 |
| `cmp_residual_k` | INT32 `[B]` 或 `None` | 仅 causal 压缩，即 `mask_mode=3` 且 `cmp_ratio!=1` 时提供 |
| `block_table` | INT32 `[B,max_pages]` 或 `None` | PA 逻辑页到物理页映射；PA 多 batch 必填，TND 禁用 |
| `output_idx_offset` | INT32 `[T1,1]` 或 `None` | 为每行有效输出索引添加偏移 |
| `metadata` | INT32 `[1024]` | 必须与当前候选长度、序列长度和静态配置一致 |
| `max_seqlen_q` | int，默认 `-1` | 最大单 batch Q 长度；已知时应显式提供 |
| `return_value` | bool，默认 `False` | False 时返回空 values Tensor |
| 返回 `indices / values` | INT32 / BF16 `[T1,1,topk]` | 有效 token 的逻辑索引及对应分数；无效项为 `indices=-1, values=-inf` |

TopK 不保证按分数排序；索引不是候选数组内的位置，也不是物理 PA 地址。
长度、候选前缀和页表的取值应由调用方保证合法，不能仅依赖 Host 形状检查。
各输入放在同一 NPU，常规输入应连续；K 及 TND descale_k 可在第 0 轴带 padding，内部轴须连续且不重叠。
改变候选长度或序列参数后，应重新生成 metadata；不要复用另一批输入的调度结果。

## 精度测试

测试代码位于 [test_quant_sparse_lightning_indexer.py](../../test/quant_sparse_lightning_indexer/test_quant_sparse_lightning_indexer.py)。
在仓库根目录执行，MXFP4 输入、候选前缀、页表和 PyTorch CPU reference 均由测试生成：

```bash
python -m pytest -q test/quant_sparse_lightning_indexer/test_quant_sparse_lightning_indexer.py
# 只运行典型长序列用例
python -m pytest -q test/quant_sparse_lightning_indexer/test_quant_sparse_lightning_indexer.py -k typical_case
```

NPU 用例覆盖 PA128、PA64、TND、causal 压缩、零候选、尾块、索引偏移及返回分数开关，
并严格检查不足 TopK 和空行的无效分数为 `-inf`，以 TopK=16 回归 LD 非整块尾部处理。
独立 CPU reference 解码 MXFP4，并按 BF16 QK/权重、逐 head 的 BF16 融合乘加和 BF16 输出计算分数；
仅允许候选前缀和 causal 范围内的 token 入选，检查数量、唯一性及 TopK 边界（`rtol=1/128, atol=2.5e-5`）。
测试文件中的 `test_typical_case` 使用以下参数：

| B | S1 | Nq / Nk | D | S2 | K 布局 / 页大小 | TopK | 候选块数量 / 大小 | mask_mode / cmp_ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 12 | 6 | 32 / 1 | 128 | 131072 | PA_BBND / 128 | 512 | 2048 / 8 | 3 / 2 |

各 batch 使用独立物理页，页表和候选块顺序随机排列，权重含正负值。
比较全部 Query 行，并分别调用 `return_value=True/False`。
典型用例标记为 `slow`，默认参与测试；快速回归可加 `-m 'not slow'`。
共 12 项测试：1 项 CPU reference 检查、10 项边界回归、1 项典型精度测试。

## 性能数据

以下为已有 msprof 主 kernel 平均耗时（μs）：

| Case | B | S1 | Nq | Nk | D | S2 | PA页大小 | TopK | 候选块数量 | 候选块大小 | mask_mode | cmp_ratio | 平均耗时（μs） |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QSLI_128K | 12 | 6 | 32 | 1 | 128 | 131072 | 128 | 512 | 2048 | 8 | 3 | 2 | 90.265 |

`return_value=False`。耗时统计主 kernel，不含 metadata 生成和 Python 调用时间。
