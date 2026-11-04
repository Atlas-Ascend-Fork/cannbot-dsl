# Quant Block Sparse Attention

基于 CANNBotDSL 实现的 FP8 量化块稀疏注意力算子，面向 Ascend NPU 的 Paged Attention Prefill 场景。

## 算子介绍

对第 $i$ 个 Query token 的有效稀疏 KV 集合 $\Omega_i$，计算公式为：

$$
\hat Q_i = Q_i^{fp8} \cdot q\_descale_i,\quad
\hat K_j = K_j^{fp8} \cdot k\_descale_j,\quad
\hat V_j = V_j^{fp8} \cdot v\_descale_{head(j)}
$$

$$
O_i = \sum_{j \in \Omega_i}
softmax_{j \in \Omega_i}
\left(\hat Q_i \hat K_j^T \cdot softmax\_scale + mask_{ij}\right)
\hat V_j
$$

| 特性与约束 | 说明 |
| :--- | :--- |
| quant_mode | 当前支持 1 |
| 数据类型 | Q/K/V：FP8 E4M3FN；输出：BF16；LSE：FP32 |
| Layout | Q：TND、NTD；KV：PA_BNBD；输出：TND |
| 稀疏块 | Q/KV block size 固定为 128 |
| mask_mode | 0=无 mask，3=right-context causal |
| GQA | 支持；`N1 % N2 == 0`，且 `1 <= N1/N2 <= 16` |
| D | 128 |
| 支持架构 | NPU ARCH 3510（Ascend 950PR / Ascend 950DT） |

Kernel 将两个 128-token 稀疏 KV 块组合为一个 256-wide 宏块，依次完成 QK、反量化、mask、在线 Softmax、P 量化和 PV。奇数块、KV 尾块及空稀疏任务均有独立边界处理。

主算子实现详见 `quant_block_sparse_attn.py`。`quant_block_sparse_attn_metadata.py`
使用 AICPU 读取稀疏工作量并生成调度 metadata；Host 仅负责静态参数
校验、输出分配和异步发射，不读取输入 Tensor 内容。

## 快速开始

```bash
source /home/<user>/Ascend/cann-9.2.0/set_env.sh
python -m pip install /path/to/cannbotdsl-*.whl
```

下面示例从测试 bundle 读取一组已经量化并包含稀疏索引和
页表的输入，再由 AICPU 在当前 NPU stream 上生成 metadata：

```python
import torch
import torch_npu

from quant_block_sparse_attn.quant_block_sparse_attn import (
    quant_block_sparse_attn,
)
from quant_block_sparse_attn.quant_block_sparse_attn_metadata import (
    quant_block_sparse_attn_metadata,
)

device = "npu:0"
bundle = torch.load("/path/to/qbsa_bundle.pt", map_location="cpu", weights_only=False)
inputs = {
    name: value.to(device) if isinstance(value, torch.Tensor) else value
    for name, value in bundle["inputs"].items()
}
inputs.pop("metadata", None)  # 按当前输入重新生成 metadata
inputs["metadata"] = quant_block_sparse_attn_metadata(
    inputs["sparse_seq_len"],
    inputs["sparse_seq_len"].shape[1],
    inputs["key"].shape[1],
    inputs["query"].shape[-1],
    cu_seqlens_q=inputs["cu_seqlens_q"],
    seqused_kv=inputs["seqused_kv"],
    sparse_block_size_q=inputs["sparse_q_block_size"],
    sparse_block_size_k=inputs["sparse_kv_block_size"],
    quant_mode=inputs["quant_mode"],
    mask_mode=inputs["mask_mode"],
    layout_q=inputs["layout_q"],
    layout_kv=inputs["layout_kv"],
    layout_sparse_indices=inputs["layout_sparse_indices"],
)

attention_out, softmax_lse = quant_block_sparse_attn(**inputs)
torch.npu.synchronize()
```

`quant_block_sparse_attn()` 关键参数：

| 参数 | 默认值 | 说明 |
| :--- | :---: | :--- |
| `query` | — | FP8；TND `[T,N1,128]` 或 NTD `[N1,T,128]` |
| `key` | — | FP8 PA 页，shape `[P,N2,128,128]` |
| `value` | — | FP8 PA 页，shape 同 key |
| `q_descale` | — | FP32；TND `[T,N1]` 或 NTD `[N1,T]`，逐 token/head 反量化系数 |
| `k_descale` | — | FP32 `[P,N2,128,1]`，逐 token/head 反量化系数 |
| `v_descale` | — | FP32 `[N2]`，逐 KV head 反量化系数 |
| `p_scale` | — | FP32 `[1]` 或 `None`，Probability 的正有限量化尺度 |
| `sparse_indices` | — | INT32 `[B,N1,max_Qb,Kb]`，每个 Q 块选择的逻辑 KV 块 |
| `sparse_seq_len` | — | INT32 `[B,N1,max_Qb]`，各稀疏列表的有效长度 |
| `atten_mask` | — | mask_mode=3 时为 UINT8 `[2048,2048]`；mask_mode=0 时传 `None` |
| `softmax_scale` | — | 有限标量，通常为 `1/sqrt(128)` |
| `sparse_q_block_size` | — | 固定为 128 |
| `sparse_kv_block_size` | — | 固定为 128 |
| `cu_seqlens_q` | `None` | INT32 `[B+1]`，带前导 0，并以 T 结束 |
| `seqused_kv` | `None` | INT32 `[B]`，每个 batch 的实际 KV token 数 |
| `block_table` | `None` | INT32 `[B,L]`，逻辑 KV 块到物理 PA 页的映射 |
| `metadata` | `None` | AICPU metadata 算子生成的 NPU INT32 一维张量，描述 AIC 任务范围 |
| `layout_q` | `TND` | `TND` 或 `NTD` |
| `layout_kv` | `PA_BNBD` | 当前仅支持 `PA_BNBD` |
| `layout_sparse_indices` | `B_N_Qb_Kb` | 当前仅支持 `B_N_Qb_Kb` |
| `layout_out` | `TND` | 当前仅支持 `TND` |
| `quant_mode` | 1 | 当前仅支持 Mode1 |
| `mask_mode` | 3 | 0=无 mask，3=right-context causal |
| `return_softmax_lse` | False | 是否返回 FP32 LSE `[N1,T]` |
| 返回值 | — | `(attention_out, softmax_lse)`；输出为 BF16 `[T,N1,128]` |

### 稀疏索引与 PA 页表

`sparse_indices[b, n, qb]` 保存当前 Q 块要访问的逻辑 KV 块编号，只有前 `sparse_seq_len[b, n, qb]` 项有效；逻辑块再通过 `block_table[b]` 映射到 `key` 和 `value` 的物理 PA 页。

每个任务的有效逻辑索引不能重复。`seqused_kv` 可以产生一个不足 128 token 的末尾块，填充位置不会参与计算。空稀疏任务的输出为 0，LSE 为 `-FLT_MAX`。

metadata 按 AIC 划分 `(batch, query_head, query_block)` 任务。调用方必须在
主算子前调用 `quant_block_sparse_attn_metadata()`；两者在同一 stream 上按序
异步执行，不需要在 Host 侧读回、修改或缓存 metadata 内容。

## 精度测试

测试脚本位于 `test/attention/quant_block_sparse_attn/test_quant_block_sparse_attn.py`，需在 NPU 环境下运行：

```bash
QBSA_TEST_DEVICE=npu:0 PYTHONPATH=samples/attention python -m pytest \
  test/attention/quant_block_sparse_attn/test_quant_block_sparse_attn.py -v -s
```

当前默认启用 4 个代表性泛化 case，覆盖 FP8 基础精度、TND/NTD、多 batch、KV 尾块和随机稀疏 causal 场景。测试会优先读取已有 input/golden bundle；可通过 `QBSA_GENERALIZED_BUNDLE_DIR` 指定泛化 bundle 目录，通过 `QBSA_PREFILL_BUNDLE_MANIFEST` 指定 Prefill bundle manifest。

精度验收允许最多 0.5% 的元素超出逐元素阈值：

| 输出 | rtol | atol |
| :--- | ---: | ---: |
| Attention | 0.0078125 | 0.0001 |
| LSE | 0.005 | 0.000025 |

## 性能对比

![QuantBlockSparseAttn vs AscendC 性能对比](../../../figures/quant_block_sparse_attn.png)

QuantBlockSparseAttn (CANNBotDSL) 与 AscendC 实现在 3 个 8K 和 3 个 16K Prefill case 上的性能对比。数据使用 Level1 profiler 采集，预热 2 次、采集 5 次，并取最小 Task Duration；柱顶倍率为 `AscendC / CANNBotDSL`，大于 1 表示 DSL 更快。
