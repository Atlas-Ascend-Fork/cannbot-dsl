# Mixed Quant Sparse Flash MLA

基于 CANNBotDSL 实现的混合量化稀疏 MLA（MQSMLA）算子，面向 Ascend 950
（NPU ARCH 3510）。AICore 主 Kernel 消费独立 AICPU 算子生成的
`int32[1024]` metadata，输出由调用方预分配。

## 算子介绍

对 BF16 Query、FP8 原始 KV 和可选 FP4 压缩 KV 执行稀疏 Attention，融合
反量化、QK、在线 Softmax、PV 与输出写回。`sinks` 提供每个 Query head 的
Softmax sink；支持整行分核与 metadata 指定的 FlashDecode 分片归约。
支持以下两种输入组合：

| 模式               | 输入组合                                    | 说明                                                   |
| :----------------- | :------------------------------------------ | :----------------------------------------------------- |
| 原始 KV 稀疏注意力 | 原始 KV、原始稀疏索引和原始页表             | 按稀疏索引选择 FP8 原始 KV                             |
| 混合 KV 稀疏注意力 | 同时提供原始侧与压缩侧的 KV、稀疏索引和页表 | 对选中的 FP8 原始 KV 与 FP4 压缩 KV 联合计算 Attention |

实现详见 [`mixed_quant_sparse_flash_mla.py`](mixed_quant_sparse_flash_mla.py)。

### 特性与约束

| 特性与约束   | 说明                                                |
| :----------- | :-------------------------------------------------- |
| Query / 输出 | BF16，TND`[T1,64,512]`                            |
| Head 配置    | Query head 为 64，KV head 为 1，head dim 为 512     |
| 原始 KV      | 必填，UINT8`[pages,page_size,1,544]`，FP8 E4M3FN group32 |
| 压缩 KV      | 可选，UINT8`[pages,page_size,1,320]`，FP4 E2M1 group16 |
| KV 字节布局  | `nope[448] -> rope[64] -> scale`                  |
| KV layout    | PA_BBND，page size 为 1～1024                       |
| 稀疏索引     | INT32`[T1,1,K]`，有效列对应合法逻辑 KV 索引       |
| TopK 长度    | 可选 INT32`[T1,1]`；省略时使用全部 K 列           |
| quant_mode   | 固定为 1                                            |
| metadata     | 必填，INT32`[1024]`                               |
| LSE          | 启用时为 FP32`[1,T1,64]`；否则为 FP32 `[0]`     |
| 支持架构     | NPU ARCH 3510（Ascend 950）                         |

每个 KV token 的 512 个量化值按 `nope[448] -> rope[64]` 排列，随后存放
BF16 scale。FP8 使用 512 个数据字节和 32 个 scale 字节；FP4 使用 256 个
数据字节和 64 个 scale 字节，偶数位置的值放在低 4 bit。

对每个 Query head，将两侧有效稀疏列对应的反量化 KV 合并为 $X$，同一 $X$
同时用作 Key 和 Value。令 $z_j=\text{softmax_scale}\,q^T X_j$，则

$$
Z=\exp(\mathrm{sinks}_h)+\sum_j\exp(z_j),\qquad
O=\frac{\sum_j\exp(z_j)X_j}{Z},\qquad
\mathrm{LSE}=\log Z.
$$

sink 只参与 Softmax 分母，不贡献 Value；两侧有效长度均为零时，输出为零，
LSE 等于该 head 的 sink。原始 KV、索引、页表仍必须提供，不支持省略原始侧的纯压缩接口。

### Metadata 调用关系

```text
mixed_quant_sparse_flash_mla_metadata (AICPU)
                    │
                    ▼
           int32[1024] metadata
                    │
                    ▼
       mixed_quant_sparse_flash_mla (AICore)
```

metadata 实现在同目录
[`mixed_quant_sparse_flash_mla_metadata`](mixed_quant_sparse_flash_mla_metadata.py)。
调用方先生成 metadata，再在同一 NPU 流调用主算子；跨流调用由调用方建立依赖。
主算子不会隐式生成 metadata。

## 快速开始

确保已安装 CANNBotDSL 0.7.0、PyTorch 和 torch_npu。
在仓库根目录执行以下命令，将示例路径替换为自己的 Python 和 CANN 安装位置，
再加入两个 sample 目录。设备可见性按实际环境配置；示例使用当前可见的逻辑设备 0：

```bash
export PYTHON=/path/to/python3.12
export CANN_ENV=/path/to/Ascend/cann/set_env.sh
source "$CANN_ENV"
export CANNBOTDSL_NATIVE_BINARY_MODE=off
export PYTHONPATH="$PWD/samples/mixed_quant_sparse_flash_mla:$PYTHONPATH"
```

用 `"$PYTHON"` 执行下面的 Python 示例。示例执行单 Query、仅原始 KV 的稀疏 Attention。KV 使用全零字节编码，
用于演示输入组织、metadata 生成和输出分配；输出应为零。
实际使用时，应传入按上述布局编码的量化 KV，不能直接将浮点 KV 转为 UINT8。

```python
import torch
import torch_npu

from mixed_quant_sparse_flash_mla import mixed_quant_sparse_flash_mla
from mixed_quant_sparse_flash_mla_metadata import mixed_quant_sparse_flash_mla_metadata

torch.npu.set_device(0)
q = torch.zeros((1, 64, 512), dtype=torch.bfloat16, device="npu")
ori_kv = torch.zeros((1, 128, 1, 544), dtype=torch.uint8, device="npu")
block_table = torch.zeros((1, 1), dtype=torch.int32, device="npu")
indices = torch.arange(128, dtype=torch.int32, device="npu").reshape(1, 1, 128)
cu_q = torch.tensor([0, 1], dtype=torch.int32, device="npu")
ori_length = torch.full((1, 1), 128, dtype=torch.int32, device="npu")
cmp_length = torch.zeros((1, 1), dtype=torch.int32, device="npu")
sinks = torch.zeros(64, dtype=torch.float32, device="npu")

metadata = mixed_quant_sparse_flash_mla_metadata(
    ori_length, cmp_length,
    cu_seqlens_q=cu_q,
    num_heads_q=64, num_heads_kv=1, head_dim=512, quant_mode=1,
    has_cmp_kv=False,
)
out = torch.empty_like(q)
lse = torch.empty(0, dtype=torch.float32, device="npu")
mixed_quant_sparse_flash_mla(
    q, ori_kv=ori_kv, ori_sparse_indices=indices,
    ori_block_table=block_table, cu_seqlens_q=cu_q,
    ori_topk_length=ori_length, sinks=sinks, metadata=metadata,
    quant_mode=1, out=out, lse=lse,
)
torch.npu.synchronize()
assert torch.count_nonzero(out).item() == 0
```

启用 LSE 时，分配 `torch.empty((1, T1, 64), dtype=torch.float32, device="npu")`
作为 `lse`，并在主算子调用中设置 `return_softmax_lse=True`。

使用混合 KV 时，还需提供 `cmp_kv`、`cmp_sparse_indices` 和 `cmp_block_table`，
将每行有效压缩 TopK 长度传给两个算子，并在 metadata 调用中设置 `has_cmp_kv=True`。

### 安装包调用

源码示例与测试直接加载本仓库 sample。需要部署安装包时，使用共享
[Native 构建工程](../../net/native_package/README.md)，注册名为 `mqsmla`，所属组为
`ds41`，同时打包主算子和 metadata 模块：

```bash
export PYTHON=/path/to/python3.12
export CANN_ENV=/path/to/Ascend/cann/set_env.sh
source "$CANN_ENV"
bash net/native_package/run-build.sh --operator mqsmla
"$PYTHON" -m pip install --no-deps --force-reinstall \
  net/native_package/output/operators/mqsmla/wheels/cannbot_arena_net-0.3.0-cp312-cp312-linux_x86_64.whl
export CANNBOTDSL_NATIVE_BINARY_MODE=require
mkdir -p .build/mqsmla-readme
export CANNBOTDSL_NATIVE_BINARY_REPORT="$PWD/.build/mqsmla-readme/native.json"
```

新发行包为 `cannbot-arena-net`，不能与同样提供 `ops` 的旧包
`cannbot-arena-net-ops` 共存。安装包调用时，将上述示例中的两个 import 替换为：

```python
from ops.mixed_quant_sparse_flash_mla import mixed_quant_sparse_flash_mla
from ops.mixed_quant_sparse_flash_mla_metadata import mixed_quant_sparse_flash_mla_metadata
```

AICore 通过包内注册的 Native 目录查找二进制；`require` 模式不允许 JIT 回退，
可检查报告确认命中。metadata 仍使用独立的 AICPU 编译和加载路径。

## 接口

```python
def mixed_quant_sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    softmax_scale=None,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=False,
    out,
    lse,
):
    ...
```

### 输入与输出

| 参数/返回值                         | 说明                                                         |
| :---------------------------------- | :----------------------------------------------------------- |
| `q`                               | BF16 Query`[T1,64,512]`                                    |
| `ori_kv/cmp_kv`                   | 原始/压缩 KV 字节池；压缩侧三个 KV、索引、页表输入须同时提供 |
| `*_sparse_indices`                | 对应 KV 池的 INT32`[T1,1,K]` 稀疏索引                      |
| `*_block_table`                   | INT32`[B,max_pages]` 逻辑页到物理页映射                    |
| `cu_seqlens_q`                    | 必填 INT32`[B+1]`，含前导 0，末值为 T1                     |
| `seqused_q`                       | 可选 INT32`[B]` 兼容参数；约定等于 Query 前缀和跨度，当前仅校验类型/形状                   |
| `seqused_ori_kv/seqused_cmp_kv`   | 可选 INT32`[B]` 兼容参数；当前仅校验类型/形状，不参与索引过滤或长度裁剪                               |
| `ori_topk_length/cmp_topk_length` | 对应每 Query 的有效稀疏列数；须与 metadata 规划一致          |
| `sinks`                           | 必填 FP32`[64]`，每个 Query head 的 Softmax sink           |
| `metadata`                        | 配套 AICPU 算子的输出                                        |
| `softmax_scale`                   | 缺省为`1/sqrt(512)`                                        |
| `out/lse`                         | 调用方预分配的连续输出 Tensor，shape/dtype 见上表            |
| 返回值                              | `None`；结果原地写入 `out` 和启用的 `lse`              |

### 一致性要求

1. 输入、metadata 和输出应位于同一 NPU；除 KV 的页间 padding 外，各 Tensor 应连续。
2. 两次调用的 Query 划分、有效 TopK 长度、head 配置及压缩 KV 存在性必须一致。
3. metadata 接口要求两个 TopK 长度 Tensor；主算子省略长度时，规划长度应填满对应 K。
   无压缩 KV 时，metadata 的压缩 TopK 长度填 0，并设置 `has_cmp_kv=False`。
4. `T1 >= 1`、`B >= 1`、两侧索引的 `K >= 1`。`cu_seqlens_q` 从 0 开始、非递减且末值为 T1，
   允许单个 batch 的 Query 长度为零。每行有效 TopK 长度须在 `[0,K]` 内。
5. KV 允许页间 padding，页内维度必须连续；页表须有 B 行，映射至合法物理页。
   有效稀疏列必须是对应 batch 的合法逻辑 KV 索引；长度之外的列不参与计算。
   主接口不读取张量值进行边界校验，合法前缀和、长度和地址由调用方保证。
6. 不提供压缩 KV 时，主算子的 `cmp_topk_length` 须省略；metadata 的压缩长度仍须提供全零 Tensor。
7. `seqused_*` 不参与当前 AICore 计算，实际 batch 划分由 `cu_seqlens_q` 决定，
   有效稀疏列由 `*_topk_length`（或省略时的 K）决定。
8. 源码模式（`CANNBOTDSL_NATIVE_BINARY_MODE=off`）首次调用触发编译，后续复用进程内缓存。

metadata 接口自身允许省略 `cu_seqlens_q`，此时按单 batch `[0,T1]` 规划；
主算子始终要求显式传入它。两阶段调用建议传入同一个前缀和 Tensor，完整 metadata
参数见 [配套 metadata 接口](mixed_quant_sparse_flash_mla_metadata.py)。

## 精度测试

主算子测试与独立 CPU golden 均位于
[`test_mixed_quant_sparse_flash_mla.py`](../../test/mixed_quant_sparse_flash_mla/test_mixed_quant_sparse_flash_mla.py)，
用例定义见
[`mixed_quant_sparse_flash_mla_paramset.py`](../../test/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla_paramset.py)。
测试使用固定随机种子在本地生成输入，先调用 AICPU metadata，再调用当前仓库的
AICore 源码，与 CPU golden 比较，不依赖外部仓库或保存张量。
输出 golden 采用 128 列分块在线 Softmax、BF16 概率及 FP32 累加；LSE 另用
FP64 logits 和 `logsumexp` 计算，独立验证 sink 与归一化语义。

按「快速开始」配置自己的 Python、CANN 环境和设备后，在仓库根目录执行以下命令。
测试使用当前可见的逻辑设备 0；需已安装 CANNBotDSL 0.7.0 和 pytest。

```bash
export PYTHON=/path/to/python3.12
export CANN_ENV=/path/to/Ascend/cann/set_env.sh
source "$CANN_ENV"

export CANNBOTDSL_NATIVE_BINARY_MODE=off

# 默认回归：4 个主算子精度用例 + 8 个 metadata 检查（3 个结构用例、5 个非法参数用例）
"$PYTHON" -m pytest test/mixed_quant_sparse_flash_mla test/mixed_quant_sparse_flash_mla_metadata -v

# 全部主算子用例及 metadata 检查
MQSMLA_CASES=all "$PYTHON" -m pytest test/mixed_quant_sparse_flash_mla test/mixed_quant_sparse_flash_mla_metadata -v

# 指定主算子用例，仍执行 metadata 检查
MQSMLA_CASES=multirow_42,cmp_only_rows "$PYTHON" -m pytest test/mixed_quant_sparse_flash_mla test/mixed_quant_sparse_flash_mla_metadata -v
```

默认主算子用例为 `mixed_decode`、`ori_only`、`empty_and_tail` 和 `padded_pages`。
当前全部 13 个主算子用例进一步覆盖极小 TopK 与尾块、仅压缩侧有有效 KV 的行、
省略 TopK 长度、多行调度及 FlashDecode。metadata 结构检查验证调度信息，
最终数值正确性由主算子的输出检查确认。

BF16 输出按 `rtol=0.0078125, atol=0.0001` 逐元素比较，要求合格率至少为
99.5%，并应用比较器的最大相对误差检查。启用 LSE 时采用 `rtol=atol=1e-5`；
空行输出必须严格为零，空行 LSE 必须严格等于对应 head 的 `sinks`。

上述测试命令设置 `CANNBOTDSL_NATIVE_BINARY_MODE=off`，验证当前源码；安装 wheel 的
Native 命中验证需独立执行。单独运行 metadata 测试可使用：

```bash
"$PYTHON" -m pytest -q \
  test/mixed_quant_sparse_flash_mla_metadata/test_mixed_quant_sparse_flash_mla_metadata.py
```

## 性能

使用 Ascend950DT_9582、CANN 9.2.0、CANNBotDSL / OpKit 0.7.0 和 msprof。
运行时查询为 32 Cube / 64 Vector 核，profiler 实际 `Block Num=32`。

固定场景如下：B 为 batch 数，S1 为每个 batch 的 Query 长度，S2/S2C 为原始/压缩
KV 长度，K1/K2 为对应稀疏列数。`T1 = B × S1`，两侧 page size 均为 128：

| 场景 | B | S1 | S2 | S2C | K1 | K2 | T1 |
|:---|--:|--:|--:|--:|--:|--:|--:|
| decode-1 | 32 | 1 | 32768 | 16384 | 128 | 512 | 32 |
| decode-6 | 32 | 6 | 32768 | 16384 | 128 | 512 | 192 |
| decode-6-long-s2 | 12 | 6 | 131072 | 65536 | 128 | 512 | 72 |

每轮每例预热 1 次后调用 10 次，仅统计 `MIX_AIC` 类型的 `MqsmlaKernel`，
不计入 AICPU metadata。每轮去掉下标 0、1，合并同版本两轮的 18 个样本；
若极差不小于 2 µs，逐次剔除距中位数最远的样本，直至极差小于 2 µs。

| 场景 | Task Duration (µs) | aiv_vec_time (µs) | aic_mac_time (µs) | 保留样本数 | Task Duration 极差 (µs) |
|:---|--:|--:|--:|--:|--:|
| decode-1 | 17.2022 | 7.8134 | 7.3315 | 17 | 1.843 |
| decode-6 | 63.9355 | 46.0693 | 42.1595 | 17 | 1.746 |
| decode-6-long-s2 | 31.9597 | 18.0892 | 17.0784 | 16 | 1.972 |
