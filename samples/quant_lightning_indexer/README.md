# Quant Lightning Indexer

基于 CANNBotDSL 实现的 MXFP4 Lightning Indexer（QLI），对 Query 与可见 Key
计算加权相关性并选择 TopK token，面向 Ascend NPU。可同时输出供
[QSLI](../quant_sparse_lightning_indexer/README.md) 使用的候选块。

## 算子介绍

支持 Ascend 950PR / Ascend 950DT。

QLI 用于推理场景中稀疏 Attention 的前处理：对当前 Query 与上下文 Key 计算相关性，
选出关键 token 的索引，供后续稀疏 Attention 使用。输入 Q/K 已完成 MXFP4 量化，
本算子消费量化数据及其反量化 scale，不负责生成量化数据，也不计算 Softmax 或 Attention 的 Value 加权和。

计算公式：

对一个 batch 内的 Query token $t$，设 Query 头数 $g=32$、Key 头数为 1、逻辑维度 $D=128$。
令 $Q^{(4)}_{t,h,d}$、$K^{(4)}_{j,d}$ 为解码后的 FP4 E2M1 数值，
$S^Q_{t,h,a}$、$S^K_{j,a}$ 为第 $a$ 个 32 元素分组的 E8M0 scale，则：

$$
R_{t,h,j}=\sum_{a=0}^{3}S^Q_{t,h,a}S^K_{j,a}
\sum_{d=32a}^{32a+31}Q^{(4)}_{t,h,d}K^{(4)}_{j,d},
\qquad
s_{t,j}=\sum_{h=0}^{31}w_{t,h}\operatorname{ReLU}(R_{t,h,j}),
$$

$$
I_t=\operatorname{TopKIndices}_{j\in\mathcal V_t}(s_{t,j}),
\qquad V_t=s_{t,I_t}.
$$

$\mathcal V_t$ 是当前 Query 可见的逻辑 Key token 集合；$V_t$ 仅在 `return_value=True` 时返回。
MXFP4 的 scale 沿 D 轴分组，不能将一个 token 的全部 scale 简化为一次标量外积。
每两个 FP4 元素打包为一个 UINT8，低四位对应前一个逻辑元素；128 个逻辑元素占 64 字节，
每个 token/head 的四个 scale 按 `[D/64, 2] = [2,2]` 存储。

主要计算过程为：

1. 根据序列前缀和及 PA 页表定位当前 Query 和上下文 Key；TND 按连续逻辑序列定位。
2. 计算分组 scale 参与反量化的 QK 相关性，得到各 Query head 对各 Key token 的分数。
3. 对相关性应用 ReLU，再乘逐 head 权重，沿 head 轴归约为每个 Key token 的一个分数。
4. 根据有效长度和 causal 配置限定可见 token，在可见集合中选取 TopK，返回逻辑索引及可选分数。

实现中 QK 交接为 BF16，FP32 权重转为 BF16，逐 head 融合乘加的结果按 BF16 舍入。
权重可为负，因此归约后的有效分数也可能为负；无效项不能用 0 代替负无穷参与选择。

可见范围与输出：

设当前 batch 的有效 Q/K 长度为 $L_Q,L_K$，Query 行号 $t$ 从 0 开始，
$r=$ `cmp_ratio`，$e=$ `cmp_residual_k`（$r=1$ 时取 0）。
`mask_mode=0` 时可见长度为 $L_K$；`mask_mode=3` 为右下对齐 causal，其可见长度为：

$$
L_t=\min\left(L_K,\max\left(0,
\left\lfloor\frac{rL_K+e-L_Q+t+1}{r}\right\rfloor\right)\right),
\qquad \mathcal V_t=\{j\mid 0\le j<L_t\}.
$$

当 $t\ge L_Q$ 时该行全部无效。`cmp_ratio` 描述已压缩 Key 的序列对应关系，本算子不执行 Key 压缩。
有效 token 不足 TopK 时，剩余输出填 `indices=-1, values=-inf`；
`output_idx_offset` 只加到有效索引，索引相对于当前 batch 的逻辑 K 序列。

候选块与分核：

可选候选输出将可见 token 按 8-token 逻辑块分组，以块内最大分数选择候选块，
用于后续 QSLI；有效长度尾部不足 8 个 token 的块优先保留。
`candidate_length` 表示候选数组的有效前缀长度，候选块号与主输出 token 索引是不同单位。
关闭候选分支时，两个候选输出为空 Tensor。

[QLI Metadata](../quant_lightning_indexer_metadata/README.md) 在 AICPU 上生成分核边界。
主 kernel 完成 QK、权重归约和分片 TopK；同一 Query 被多个核处理时，再在融合 kernel 内归并各分片结果。
本 DSL 接口使用 `quant_mode=1` 表示 MXFP4，编号不等同于原生 QuantLightningIndexerV2 接口的量化模式枚举。

| 特性与约束 | 说明 |
| :--- | :--- |
| 量化 | `quant_mode=1`；MXFP4 E2M1，2 个逻辑元素打包为 1 字节；scale 为 E8M0 字节 |
| 固定轴 | `Nq=32`、`Nk=1`、逻辑 `D=128`，打包后末轴为 64 |
| Layout | Q 为 `TND`；K 为 `PA_BBND` 或 `TND` |
| PA 页大小 | 16 的倍数，范围 `(0,1024]`；典型验证为 64/128 |
| TopK | `1 <= topk <= 8192` |
| Mask | `mask_mode=0` 无 mask；`3` 为右下对齐 causal；`cmp_ratio` 范围 `[1,128]` |
| 候选输出 | 关闭时两参数均为 `-1`；开启时 `candidate_topk_blocks>0`、`candidate_block_size=8` |
| Metadata | 必须显式传入同设备、连续的 `int32[1024]` |

实现见 [quant_lightning_indexer.py](quant_lightning_indexer.py)。

## 快速开始

使用 CANN 9.2.0、CANNBotDSL 0.7.0、PyTorch 与 torch_npu；
PyTorch 需提供 `float8_e8m0fnu` 类型，设备需支持 MXFP4。
在仓库根目录设置环境和导入路径：

```bash
source ${install_path}/ascend-toolkit/set_env.sh
export PYTHONPATH="$PWD/samples/quant_lightning_indexer:$PWD/samples/quant_lightning_indexer_metadata:$PYTHONPATH"
```

下面构造一组有效打包格式的 PA 输入，仅演示接口，不代替真实数据的量化过程或精度测试：

```python
import torch
import torch_npu
from quant_lightning_indexer import quant_lightning_indexer
from quant_lightning_indexer_metadata import quant_lightning_indexer_metadata

device = "npu:0"
torch.npu.set_device(device)
query_rows, key_tokens, page_size, topk = 6, 1024, 128, 512
pages = key_tokens // page_size
q = torch.randint(0, 256, (query_rows, 32, 64), dtype=torch.uint8).to(device)
k = torch.randint(0, 256, (pages, page_size, 1, 64), dtype=torch.uint8).to(device)
w = torch.ones((query_rows, 32), dtype=torch.float32, device=device)
descale_q = torch.full((query_rows, 32, 2, 2), 127, dtype=torch.uint8, device=device)
descale_k = torch.full((pages, page_size, 1, 2, 2), 127, dtype=torch.uint8, device=device)
cu_q = torch.tensor([0, query_rows], dtype=torch.int32, device=device)
used_q = torch.tensor([query_rows], dtype=torch.int32, device=device)
used_k = torch.tensor([key_tokens], dtype=torch.int32, device=device)
block_table = torch.arange(pages, dtype=torch.int32, device=device).view(1, pages)

metadata = quant_lightning_indexer_metadata(
    cu_seqlens_q=cu_q, seqused_q=used_q, seqused_k=used_k,
    max_seqlen_q=query_rows, max_seqlen_k=key_tokens,
    num_heads_q=32, num_heads_k=1, head_dim=128, topk=topk,
    layout_q="TND", layout_k="PA_BBND",
)
indices, values, candidates, candidate_length = quant_lightning_indexer(
    q, k, w, descale_q, descale_k,
    cu_seqlens_q=cu_q, seqused_q=used_q, seqused_k=used_k,
    block_table=block_table, metadata=metadata,
    topk=topk, quant_mode=1, max_seqlen_q=query_rows,
    layout_q="TND", layout_k="PA_BBND", return_value=True,
)
torch.npu.synchronize()
assert indices.shape == values.shape == (query_rows, 1, topk)
assert candidates.numel() == candidate_length.numel() == 0
```

`quant_lightning_indexer()` 关键参数：

`T1`、`T2` 分别表示拼接后的 Q、K token 数，`B` 为 batch 数，`P` 为物理页数。

| 参数 | 类型 / shape | 说明 |
| :--- | :--- | :--- |
| `q` | UINT8 `[T1,32,64]` | 连续的打包 MXFP4 Query |
| `k` | UINT8；PA `[P,page_size,1,64]`，TND `[T2,1,64]` | 打包 MXFP4 Key |
| `w` | FP32 `[T1,32]` | 逐 Query/head 权重 |
| `descale_q` | UINT8 `[T1,32,2,2]` | 每 32 个逻辑元素一个 E8M0 scale |
| `descale_k` | UINT8；PA `[P,page_size,1,2,2]`，TND `[T2,1,2,2]` | 与 K 对应的 E8M0 scale |
| `cu_seqlens_q / cu_seqlens_k` | INT32 `[B+1]` | 带前导零的 token 前缀和；K 前缀和用于 TND |
| `seqused_q / seqused_k` | INT32 `[B]` | 每个 batch 的实际可用 token 数，不是前缀和 |
| `cmp_residual_k` | INT32 `[B]` 或 `None` | 仅在 `mask_mode=3` 且 `cmp_ratio!=1` 时提供压缩残差 |
| `block_table` | INT32 `[B,max_pages]` 或 `None` | PA 逻辑页到物理页映射；PA 多 batch 必填，TND 必须为 `None` |
| `output_idx_offset` | INT32 `[T1,1]` 或 `None` | 为各行有效输出索引添加偏移 |
| `metadata` | INT32 `[1024]` | 配套 AICPU 算子生成；长度和静态参数必须与本次调用一致 |
| `max_seqlen_q` | int，默认 `-1` | 最大单 batch Q 长度；已知时应显式提供 |
| `return_value` | bool，默认 `False` | 是否返回对应 BF16 分数；为 False 时返回空 values Tensor |
| 返回 `indices / values` | INT32 / BF16 `[T1,1,topk]` | 索引与分数对应；无效项为 `indices=-1, values=-inf` |
| 返回 `candidates / candidate_length` | INT32 `[T1,1,C]` / `[T1,1]` | `C=candidate_topk_blocks`；关闭时均为空 Tensor |

索引相对于各 batch 的逻辑 K 序列，不是物理 PA 地址；TopK 不保证按分数排序或同分顺序。
候选输出覆盖与 QSLI 配套的 `candidate_topk_blocks=2048`、`candidate_block_size=8`，
另以容量 16 的 PA64 causal 用例回归 LD 归并不足 64 项的尾部处理。
各 Tensor 应位于同一 NPU，常规输入使用连续存储；K 与 descale_k 允许第 0 轴带 padding，内部轴必须连续且各行不重叠。
TND 多 batch 需同时提供 Q/K 前缀和；metadata 与主算子必须采用相同布局、长度、TopK、mask、压缩和候选配置。

## 精度测试

测试代码位于 [test_quant_lightning_indexer.py](../../test/quant_lightning_indexer/test_quant_lightning_indexer.py)。
在仓库根目录执行，输入和 PyTorch CPU reference 均由测试生成：

```bash
python -m pytest -q test/quant_lightning_indexer/test_quant_lightning_indexer.py
# 只运行典型长序列用例
python -m pytest -q test/quant_lightning_indexer/test_quant_lightning_indexer.py -k typical_case
```

NPU 用例覆盖 PA128、PA64、TND、变长、causal 压缩、候选开关、索引偏移及返回分数开关，
并严格检查不足 TopK 和空行的无效分数为 `-inf`。
分数按 MXFP4 解码、BF16 QK/权重舍入、逐 head 的 BF16 融合乘加和 BF16 输出独立计算；
检查合法索引、有效数量、唯一性及 TopK 边界（分数 `rtol=1/128, atol=2.5e-5`），不要求同分索引顺序。
测试文件中的 `test_typical_case` 包含以下参数：

| B | S1 | Nq / Nk | D | S2 | K 布局 / 页大小 | TopK | mask_mode / cmp_ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 12 | 6 | 32 / 1 | 128 | 65536 | PA_BBND / 128 | 512 | 3 / 2 |
| 12 | 6 | 32 / 1 | 128 | 131072 | PA_BBND / 128 | 512 | 3 / 2 |

各 batch 使用独立物理页，页表随机排列，权重含正负值；候选输出关闭。
比较所有 Query 行，并分别调用 `return_value=True/False`。
典型用例标记为 `slow`，默认参与测试；仅运行快速回归时可加 `-m 'not slow'`。
共 15 项测试：3 项 CPU reference 检查、10 项边界回归、2 项典型精度测试。

## 性能数据

以下为已有 msprof 主 kernel 平均耗时（μs）：

| Case | B | S1 | Nq | Nk | D | S2 | PA页大小 | TopK | mask_mode | cmp_ratio | 平均耗时（μs） |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QLI_64K | 12 | 6 | 32 | 1 | 128 | 65536 | 128 | 512 | 3 | 2 | 65.4048 |
| QLI_128K | 12 | 6 | 32 | 1 | 128 | 131072 | 128 | 512 | 3 | 2 | 127.586 |

候选输出关闭，`return_value=False`。耗时统计主 kernel，不含 metadata 生成和 Python 调用时间。
