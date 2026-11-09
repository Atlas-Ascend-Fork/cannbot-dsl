# mega_compressor

基于 CANNBotDSL 的压缩算子，融合 KV/gate 投影、softmax pooling、RMSNorm 和 ring-cache 写回。

## 算子介绍

对每个完整压缩组，沿 token 维度计算：

```text
kv = x @ wkv.T
score = x @ wgate.T
pooled = sum(softmax(score) * kv)
output = RMSNorm(pooled, norm_weight, norm_eps)
```

| 特性与约束 | 说明 |
| :--- | :--- |
| 优化 shape | H=5120、D=512、B=1/4、cmp_ratio=1/2 |
| Decode / MTP | S=1/6/8，验收 start_pos=8192 |
| Prefill | S=8192，验收 start_pos=0 |
| 数据类型 | 快路径使用 BF16；投影和 state_cache 使用 FP32 |
| Layout | BSH `[B,S,H]`、压紧 TH `[T,H]` |
| state_cache | FP32 `[blocks,slots,1024]`，按 block table 原地更新 |
| 支持架构 | NPU ARCH 3510；Decode 按当前 stream 的 Mix 核容量分派，专用 prefill 需要 32 个 Mix block |

实现全部位于 `mega_compressor.py`。Decode 使用独占 partial 和固定顺序归约，不使用 atomicAdd；prefill 也使用固定归约顺序。

普通入口的输入整理、设备权重打包和缺省元数据初始化均使用 DSL。通用 Decode 直接读取有效输入行；流式路径在同一个 Kernel 内准备尾块并直接写最终输出，不通过 PyTorch 补零或复制输出。传入连续的 NPU 元数据可以复用已有 Tensor；图模式仍在捕获前完成准备。

当前实现面向新版 CANNBotDSL 0.8 的显式 Layout 接口：
`Buffer(mem_loc, dtype, make_layout(shape))` 和
`RingBuffer(mem_loc, dtype, make_layout(shape), depth=...)`，
`reinterpret` 通过 `layout=make_layout(shape)` 声明视图。
使用 `RingBuffer.next_write/next_read`，拷贝格式由 Tensor layout 推导，
L0C 到 UB 的双 Vector 分片通过 `partition(dim=0)` 表达。
Cache 行显式 reshape 为二维，部分输出拷贝使用与目标大小一致的源视图。
旧版 `0.8.0+g905abe8` 的 shape 参数接口不再适用；更换 wheel 后需重新编译、预热并捕获图。

## 快速开始

在仓库根目录运行，先加载 CANN 环境并导入 `torch_npu`：

```python
import torch
import torch_npu
from samples.attention.mega_compressor.mega_compressor import mega_compressor, prepare_mega_compressor

B, S, H, D = 1, 8, 5120, 512
x = torch.randn(B, S, H, dtype=torch.bfloat16, device="npu")
wkv = torch.randn(D, H, dtype=x.dtype, device=x.device)
wgate = torch.randn_like(wkv)
gamma = torch.ones(D, dtype=x.dtype, device=x.device)
state = torch.zeros(B, 8, 2 * D, device=x.device)
blocks = torch.arange(B, dtype=torch.int32, device=x.device)
starts = torch.full((B,), 8192, dtype=torch.int32, device=x.device)

output = mega_compressor(
    x, wkv, wgate, state, cmp_ratio=2,
    state_block_table=blocks, start_pos=starts,
    norm_weight=gamma, norm_eps=1e-6,
)
```

重复执行完整 BF16 BSH 输入可使用 `prepare_mega_compressor(...)`：参数同上，但 `gamma` 作为第五个位置参数。准备完成后调用返回对象即可；输入、起点和 cache 可原地更新，形状、权重和 block 映射保持固定。更新权重后重新准备。

Host 构造时查询当前 stream 的核资源，Kernel 从实际 launch 读取 block 数并循环处理逻辑任务。修改设备或 stream 核配额后重新构造 Host／prepared 对象并重新捕获图。

| 参数 | 说明 |
| :--- | :--- |
| `norm_weight` | 可选 `[512]`；不传时只返回 pooling，传入时融合 RMSNorm |
| `norm_eps` | 默认 `1e-20`；整网传入模型实际 epsilon |
| `start_pos` | INT32 `[B]`，随推理更新；prepared prefill 起点须按 ratio 对齐 |
| `seqused` | 可选 INT32 `[B]`，每个请求的有效 token 数 |
| `cu_seqlens` | TH 必传 INT32 `[B+1]`，带前导 0，末项为 T |
| `backend` | `auto` 自动选择；`legacy` 通用调度；`pr221` 强制 BSH 专用路径 |

每个请求的有效输出数为 `(start_pos + seqused) // ratio - start_pos // ratio`。BSH 输出容量为 `[B,ceil(S/ratio),512]`；TH 有效行连续排列在开头。**S=1、ratio=2、偶数起点没有有效输出，只更新 cache。** Cache 保存未归一化的 KV/gate；ratio=2 的奇数起点会读取上一 token 的历史。

## 整网接入

单文件包使用 `from mega_compressor.mega_compressor import ...`，`mega_compressor`、`prepare_mega_compressor`、`DecodePlan`、`pack_weights`、`THDecodeHost`、`compile_args` 均从同一文件导入。原 `fused*`、`legacy_execution` 子模块不再保留。

图模式的 TH 调用使用 `THDecodeHost`，在设备端读取实时元数据；模型快路径条件包含 `total in (batch, batch * 6, batch * 8)`。工作区通过 `DecodePlan().workspace_shape(total)` 分配。编译与权重打包在图捕获前完成，升级后重新预热、捕获图。普通 `mega_compressor` 入口包含 host 校验，不用于 decode 图内调用。

确定性以相同硬件、软件、shape、输入及初始 cache 为前提，不承诺跨编译器版本逐位一致。不同请求须使用独立 cache block，同一工作区/cache 不可重叠写入。

## 精度测试

```bash
pytest test/attention/mega_compressor/test_mega_compressor.py -v
```

覆盖 B1/B4、S1/S6/S8/8192、ratio=1/2，对比 CPU golden，检查输出、完整 cache、未写位置、TH ragged/奇数起点及图重放确定性。性能对照工具与原始记录单独维护，不随算子提交。

2026-10-28 在 Ascend 950DT 单卡、CANN `9.2.0-beta.2`、Python 3.12.15、
`torch_npu 2.9.0.post8` 下，使用 `dsl-whl.tar` 中的
`cannbot_dsl-0.8.0-cp312-cp312-manylinux_2_28_x86_64.whl` 验证：28 个 NPU 用例全部通过，
28 个验收配置及 4 个 legacy／通用配置全部编译通过。
该 wheel 的 SHA256 为 `f9481b882b7290956cee09dfc3014ad1c62bba77c802288271ecfa0cf5359f99`。

2026-11-03 使用同一 wheel 和 CANN 环境完成检视修复回归：41 个 NPU 用例全部通过。
在原有 28 个用例之外，新增验证 FP16、通用路径奇数起点、默认／CPU 元数据、权重打包布局、
BSH／TH 非连续输入，以及 stream 限制为 28 个 Cube／56 个 Vector 时的专用和通用 Decode。
输出、完整 cache 和未写位置均与 CPU golden 对照，原有图重放确定性检查继续通过。

## Native 打包

`mega_compressor` 已登记到算子清单 `catalog/operators.json` 和网络清单
`net/ds41/network.json`，由 `scripts/ci/sync_catalog.py` 生成
`packaging/native/operator_groups.toml` 的 `ds41` 组。
打包器通过 `@export("mega_compressor")` 收集真实的权重打包、BSH Decode、TH Decode 和 Prefill 主计算，使用 CANNBotDSL 0.8，导出阶段不分配 NPU Tensor、不执行 Kernel，也不查询当前卡的核配额。

```bash
python packaging/native/build.py --operator mega_compressor
# 构建整组时使用 --group ds41
```

缺省导出 BF16、B1/B4、R2、slots=8、cache_blocks=B、norm_eps=1e-6：
BSH/TH Decode 的 S1/S6/S8 分别生成 28/32 Mix block 变体；BSH S8192 Prefill 生成 32 Mix block 变体。
NPU 元数据须连续且已显式提供；TH 总行数为 B*S，cu/used/start/table 内容可在图中动态变化。

其他配置可将 `CANNBOTDSL_DS41_PROFILES` 指向 JSON 文件，在 `mega_compressor` 数组中设置 `layout`、`batch`、`seq_len`、`ratio`、`slots`、`cache_blocks`、`norm_eps`、`cube_cores`。未导出的配置在 Native `prefer` 模式下回退 JIT，在 `require` 模式下报错。FP16 通用路径、非连续输入整理、缺省/CPU 元数据初始化目前不属于该 Native 导出目录。

2026-11-03 已完成默认 28 个 Native 变体的实际编译与 wheel 构建；另用 5 个含额外 cache block 的配置验证安装后的 wheel。启用 Native `require` 后，BSH Decode、TH 动态元数据/图重放和 Prefill 在 28/32 核配置下的对应输出与 cache 检查通过，记录到 11 次 Native 命中、0 次回退。
