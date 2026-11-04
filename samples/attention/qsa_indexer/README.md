# QSA Indexer

基于 CANNBotDSL 实现的稀疏 Attention 索引算子，面向 Ascend NPU。
接收预处理后的 Query 和按 4:1 压缩的 Key Cache，选择高分压缩块，
并展开为原始 token 索引，供后续稀疏 Attention 使用。

## 算子介绍

对 Query 行 $t$ 和压缩块 $j$，分数为：

$$
s_{t,j}=\operatorname{BF16}\left(\sum_{h=0}^{3}
\operatorname{ReLU}\left(Q_{t,h}\cdot K_j\right)\right)
$$

点积使用 BF16 输入、FP32 累加；逐 head 执行 ReLU 后求和，再窄化为 BF16。
分数编码为 UINT16 后执行 TopK。令 $p_t=\mathrm{query\_positions}[t]$，
每行可见的完整压缩块数、入选块数和因果尾部长度分别为：

$$
v_t=\left\lfloor\frac{p_t+1}{4}\right\rfloor,\qquad
k_t=\min(v_t,512),\qquad r_t=(p_t+1)\bmod4
$$

从 `[0,v_t)` 中选择 $k_t$ 个高分块，每个块 $j$ 展开为 `[4j,4j+1,4j+2,4j+3]`，
随后追加尾部 `[4v_t,...,4v_t+r_t-1]`。输出有效长度为 $L_t=4k_t+r_t\le2051$。

| 特性与约束 | 说明 |
| :--- | :--- |
| 数据类型 | Q/K 为 BF16，排序分数为 UINT16，输出索引及长度为 INT32 |
| Query | `[T,4,128]`，packed Query，$0\le T\le2^{31}-1$ |
| 压缩 Key | `[num_blocks,256,1,128]`，每页 256 个压缩块，每块对应 4 个原始 token |
| Batch | $1\le B\le65536$ |
| 页容量 | `num_blocks` 和 `max_pages` 均在 $[0,2^{32}-1]$ 内，实际规模受设备内存限制 |
| 可见范围 | 按 `query_positions` 执行 causal 裁剪 |
| TopK | 每行最多选择 512 个压缩块，压缩比例及 TopK 上限不可配置 |
| 输入布局 | 稠密 `torch.strided` Tensor，不支持稀疏或嵌套 Tensor；Q/K 自动连续化 |
| 输出 | `[T,2052]`，前 2051 列存索引，第 2051 列存有效长度，未使用索引填 `-1` |
| 支持架构 | NPU ARCH 3510（Ascend 950PR / Ascend 950DT） |

实现详见 `qsa_indexer.py`。调度信息由独立的
[QSA Indexer Metadata](README_metadata.md) 算子生成。

## 快速开始

安装 CANNBotDSL、PyTorch 和 torch_npu 后，在仓库根目录执行：

```bash
source /path/to/Ascend/cann/set_env.sh
export PYTHONPATH="$PWD/samples/attention/qsa_indexer${PYTHONPATH:+:$PYTHONPATH}"
```

```python
import torch
import torch_npu
from qsa_indexer import qsa_indexer
from qsa_indexer_metadata import qsa_indexer_metadata

torch.npu.set_device(0)
T, compressed_length = 32, 16 * 1024
num_blocks = compressed_length // 256

q = torch.randn(T, 4, 128, dtype=torch.bfloat16).npu()
compressed_k = torch.randn(num_blocks, 256, 1, 128, dtype=torch.bfloat16).npu()
block_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, -1).npu()
actual_seq = torch.tensor([0, T], dtype=torch.int32).npu()
query_positions = torch.arange(64 * 1024 - T, 64 * 1024, dtype=torch.int32).npu()

metadata = qsa_indexer_metadata(
    actual_seq, query_positions, block_table,
    compressed_page_count=compressed_k.shape[0],
)
indices = qsa_indexer(
    q, compressed_k, block_table, actual_seq, query_positions,
    metadata=metadata,
)
torch.npu.synchronize()

# indices: [32,2052]，INT32，位于 NPU。
# indices[:,2051]：每行有效 token 索引数量。
```

`qsa_indexer()` 关键参数：

| 参数 | 默认值 | 说明 |
| :--- | :---: | :--- |
| `q` | — | BF16 Query，shape `[T,4,128]` |
| `compressed_k` | — | BF16 压缩 Key Cache，shape `[num_blocks,256,1,128]`，与 Q 位于同一 NPU |
| `block_table` | — | INT32 页表，shape `[B,max_pages]`；逻辑块 `j` 的物理页为 `block_table[b,j//256]`，页内偏移为 `j%256` |
| `actual_seq` | — | INT32 Query 前缀和，shape `[B+1]`，首项为 0、末项为 T，单调非降 |
| `query_positions` | — | INT32 向量，shape `[T]`；每个 Query 在所属请求原始序列中的非负、零基 token 位置 |
| `metadata` | `None` | 必须显式传入 Metadata 算子输出；为 Q 所在设备上的连续 INT32，容量与 batch 对应 |
| `block_dim` | `None` | 默认获取当前设备/流的有效 Cube 核数；显式值须为非布尔 Python `int`，在 `[1,有效 Cube 核数]` 内，Metadata 布局上限为 36 |
| 返回值 `indices` | — | INT32 `[T,2052]`；每行前 `indices[t,2051]` 个 token 索引有效 |

调用要求：

- 所有输入必须是 Tensor，shape、dtype、布局及容量须满足上表。主算子允许
  `block_table`、`actual_seq`、`query_positions` 来自 CPU 或 Q 所在的同一 NPU，
  并将其搬到目标设备、连续化；生成 Metadata 时，这三个输入须已在当前 NPU 上连续存储。
- 有效页号必须在 `[0,num_blocks)` 内，页表容量须覆盖该请求全部可见压缩块。
  Metadata 在 AICPU 上检查序列偏移、非负位置、有效页号及可见 K 容量；
  主算子只检查 Host 可见的张量属性，不将设备数据读回 Host 校验。
  未访问的页表槽不检查页号；零页容量仅适用于没有完整可见压缩 K 的请求。
- 先调用 Metadata，再显式传给主算子；使用同一批调度输入和相同 `block_dim`，
  生成后不得修改相关调度输入。两个算子在同一 NPU 流上依次提交，跨流时由调用方建立依赖。
  入口均保持异步，不主动执行 `torch.npu.synchronize()`。
- 首次调用编译，后续相同配置复用编译缓存。输入、输出及工作区需满足设备可用内存容量。
  当前提供 Python 直接调用接口，不支持 GE 图模式。
- 输出为切片视图，行跨度为 2056 个 INT32 元素；按 Tensor stride 访问，
  需要连续存储时调用 `indices.contiguous()`。TopK 不保证内部排序或同分元素顺序。

## 精度测试

测试脚本位于 `test/attention/qsa_indexer/test_qsa_indexer.py`，需在 NPU 环境下运行。

```bash
pytest test/attention/qsa_indexer/test_qsa_indexer.py -v
```

覆盖论文中的四个 Prefill Case，Query 长度固定为 16K：

| Case | 原始 Context | Query 长度 | 压缩 K 长度 |
| :--- | ---: | ---: | ---: |
| P64K | 64K | 16K | 16K |
| P128K | 128K | 16K | 32K |
| P256K | 256K | 16K | 64K |
| P512K | 512K | 16K | 128K |

全部 Query 行检查有效长度、因果范围、索引唯一性、四 token 展开、尾部拼接和 `-1` 填充；
另抽取 8 行与 PyTorch golden 比较 TopK 入选集合。

精度标准：按 BF16 分数比较 TopK 边界，绝对误差不超过 `2.5e-5` 或相对误差
不超过 `1e-3` 的边界差异可接受；超出容差的入选差异比例不超过 `5e-3`。

## 性能测试

**采集方法**：使用 msprof，预热 1 次，计时 10 次，报告主算子平均耗时；不包含编译和 Metadata。

**测试结果**

选用论文中的 4 个典型 Prefill 场景 P64K、P128K、P256K、P512K，
并在此基础上泛化了 4 个性能 case，将 Query 长度和压缩 K 长度同时扩展为
32K、64K、128K、256K，共测试 8 个 case，覆盖不同序列长度下的主算子耗时和 MFU。

| Case | Q 长度 | 压缩 K 长度 | 平均耗时（μs） | MFU |
| :--- | ---: | ---: | ---: | ---: |
| P64K | 16K | 16K | 1086.480 | 41.70% |
| P128K | 16K | 32K | 2248.249 | 43.18% |
| P256K | 16K | 64K | 4837.608 | 41.48% |
| P512K | 16K | 128K | 9799.667 | 41.61% |
| Q32K_K32K | 32K | 32K | 4029.729 | 44.97% |
| Q64K_K64K | 64K | 64K | 15626.641 | 46.39% |
| Q128K_K128K | 128K | 128K | 70156.731 | 41.33% |
| Q256K_K256K | 256K | 256K | 297539.188 | 38.98% |
