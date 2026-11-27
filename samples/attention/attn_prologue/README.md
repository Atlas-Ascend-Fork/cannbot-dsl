# attn_prologue

基于 CANNBotDSL 的 MXFP8 attention prologue 融合算子，面向 Ascend 950。
在一个 MIX kernel 中完成 QA/KV 投影、RMSNorm、QR 动态量化、QB 投影、尾部 RoPE 和 KV cache 写回。
固定生产维度为 `(H,R,D,Dr)=(5120,1280,512,64)`，Q head 数 `N` 支持 16 和 64，支持 decode / prefill 的 `T=batch*sequence` 泛化。

## 功能说明

输入沿 Q 和 KV 两条支路计算。以下 `Deq` 表示结合 E8M0 scale 的逻辑反量化，
`Quant32` 表示沿特征轴每 32 个元素共享 scale 的动态量化，`@` 表示矩阵乘：

```text
X = Deq(x, descale_x)
qa = X @ Deq(wqa, descale_wqa).T                         # [T,R]
kv = X @ Deq(wkv, descale_wkv).T                         # [T,D]

qr_bf16 = BF16(RMSNorm(qa, norm_weight_qr))              # [T,R] 公开输出
qr_fp8, descale_qr = Quant32(qr_bf16)                    # 内部，供 QB
q_proj = Deq(qr_fp8, descale_qr) @ Deq(wqb, descale_wqb).T    # [T,N*D]
q = BF16(TailRoPE(reshape(q_proj, [T,N,D])))

kv_rope = TailRoPE(RMSNorm(kv, norm_weight_kv))
kv_payload, kv_scale = QuantKV32(kv_rope)              # 每 32 元素一个 BF16 scale
kv_cache[cache_index[t] // BS, cache_index[t] % BS, 0, :] =
    concat(ByteView_uint8(kv_payload[t]), ByteView_uint8(kv_scale[t]))
```

三个矩阵乘由 Cube 执行，归约、归一化、量化和 RoPE 由 Vector 执行。
设备矩阵乘直接消费 FP8 payload 和 scale，使用 FP32 中间累加。
QB 的输入是量化后的 QR；KV 使用每 32 元素一个 BF16 scale，payload 与 scale 一起写入 cache。
算子返回 `(q, qr, descale_qr)`，后续 attention 计算由其他算子完成。

### RMSNorm 与动态量化

RMSNorm 对每个 token 的完整特征行独立计算，不减均值。QA 的行宽为 R，KV 的行宽为 D：

```text
RMSNorm(v, gamma)[j] = v[j] * gamma[j] / sqrt(mean(v^2) + norm_eps)
```

QR（以及网络侧的 BF16 x）每 32 个元素组成一组，采用与 HF `round_scale` 一致的规则：

```text
amax = max(abs(group))
a = max(amax, 1e-4)
s0 = a * (1/448)
scale = ceil_pow2(s0)                 # 不小于 s0 的最小 2 的幂
payload = E4M3FN(clamp(group / scale, -448, 448))
scale 的编码字节 = exponent(scale) + 127
```

`ceil_pow2` 由 FP32 指数/尾数位实现（向上取整），不依赖 log2 边界。全零组因
`a` 的下限而 scale 非零、payload 全 0。E4M3FN 转换采用最近偶数舍入。
相邻两个 scale 成对存储，因此每行 R 个元素对应 `[R/64,2]` 的 scale。

### KV cache 量化与行布局

KV 的每个 token 向量沿 D 轴每 32 个元素一组（D=512 共 16 组），每组使用独立 BF16 scale：

```text
a = max(amax, 1e-4)                          # amax=max(abs(group))，FP32
s0 = a * (1/448)                             # FP32
s_bf16 = ceil_pow2(s0)                       # 不小于 s0 的最小 2 的幂，BF16 存储
s_used = bf16_to_fp32(s_bf16)
payload = E4M3FN(clamp(group / s_used, -448, 448))
```

`ceil_pow2` 由 FP32 指数/尾数位实现（向上取整），不依赖 log2 边界。全零组的
`amax` 被下限抬到 `1e-4`，因此 scale 不为 0、payload 全 0；这是网络原有
`round_scale=True` 口径。

cache 每行固定 544 字节：

```text
字节 0..511    : E4M3FN payload，按 group=32 顺序
字节 512..543  : 16 个 BF16 scale（小端 2 字节），按 group 顺序
```

RoPE 之前量化第 0..13 组；第 14、15 组（元素 448..511）在尾部 RoPE 之后量化。

### 尾部 RoPE

Q 的每个 head 和 KV 向量均保留前 `D-Dr` 维，仅变换最后 Dr 维。位置表为全宽
`[T,Dr]`（输入可为 `[T,1,1,Dr]`），按相邻对 `(2i, 2i+1)` 取 cos/sin，输出保持
interleave 排列：

```text
out[2*i]   = z[2*i] * cos[t,2*i]   - z[2*i+1] * sin[t,2*i]
out[2*i+1] = z[2*i+1] * cos[t,2*i+1] + z[2*i] * sin[t,2*i+1]
```

网络直接传入正常 sin 表，不需要调用方预先加负号；同一 token 的位置表在全部 Q head
上广播。滑窗层使用 `cos_sin["win"]`，压缩注意力层使用 `cos_sin["comp"]`。

## 函数原型

```python
from samples.attention.attn_prologue.attn_prologue import (
    attn_prologue, prepare_attn_prologue, to_nz,
)

q, qr, descale_qr = attn_prologue(
    x, wqa, wqb, wkv,
    descale_x, descale_wqa, descale_wqb, descale_wkv,
    norm_weight_qr, norm_weight_kv,
    rope_sin, rope_cos, cache_index, kv_cache,
    norm_eps=1e-6,
)
```

`to_nz(weight)` 将 NPU 上的权重转换为 NZ。
`prepare_attn_prologue(...)` 接收相同参数，返回可复用的 `PreparedAttnPrologue`；
调用其 `.run()` 执行计算并返回上述三个张量。`attn_prologue(...)` 等价于 prepare 后立即 run。

## 参数与约束

`T=batch*sequence` 为展平后的 token 数，`H=5120`、`R=1280`、`D=512`、`Dr=64`，
Q head 数 `N` 仅支持 16 和 64，由 `wqb.shape[0] // D` 推导。
`P` 是 cache 页数，`BS` 是每页的 token slot 数，两者由调用方指定。
下表均为逻辑形状；NZ 只改变物理布局，不改变权重的逻辑矩阵方向。

| 输入 | 形状 | dtype / 布局 | 含义 |
|---|---|---|---|
| `x` | `[T,H]` | E4M3FN / ND | token 输入特征 |
| `wqa` | `[R,H]` | E4M3FN / NZ | Q 第一次投影权重 |
| `wqb` | `[N*D,R]` | E4M3FN / NZ | Q 第二次投影权重 |
| `wkv` | `[D,H]` | E4M3FN / NZ | KV 投影权重 |
| `descale_x` | `[T,H/64,2]` | E8M0 / ND | 输入的分组 scale |
| `descale_wqa` | `[R,H/64,2]` | E8M0 / ND | QA 权重的分组 scale |
| `descale_wqb` | `[N*D,R/64,2]` | E8M0 / ND | QB 权重的分组 scale |
| `descale_wkv` | `[D,H/64,2]` | E8M0 / ND | KV 权重的分组 scale |
| `norm_weight_qr` | `[R]` | FP32 / ND | QR 的 RMSNorm 权重 |
| `norm_weight_kv` | `[D]` | FP32 / ND | KV 的 RMSNorm 权重 |
| `rope_sin`、`rope_cos` | 各 `[T,Dr]` | FP32 / ND | 每个 token 的位置变换系数 |
| `cache_index` | `[T]` | INT64 / ND | 要更新的扁平 cache slot |
| `kv_cache` | `[P,BS,1,544]` | UINT8 / ND | 原地更新的缓存，行宽 544 |
| `norm_eps` | 标量 | host 实数 | 有限正数，默认 `1e-6`，两条支路共用 |

E4M3FN 和 E8M0 分别为 `torch.float8_e4m3fn`、`torch.float8_e8m0fnu`。
输入和权重沿最后一维每 32 个 payload 共享一个 scale；有限 E8M0 编码字节 b 对应 `2^(b-127)`，
其中 `b=0..254`。旧 `int8`/`uint8` 编码载体应通过 `.view(...)` 恢复 FP8 dtype，不能用数值转换替代字节重解释。

| 输出 | 形状 | dtype | 含义 |
|---|---|---|---|
| `q` | `[T,N,D]` | `torch.bfloat16` | QB 投影并完成尾部 RoPE 的最终 Q |
| `qr` | `[T,R]` | `torch.bfloat16` | RMSNorm 后的 BF16 结果（量化前，供下游算子） |
| `descale_qr` | `[T,R/64,2]` | `torch.int8` | 内部 MXFP8 量化的 E8M0 编码字节 |
| 原地更新 `kv_cache` | `[P,BS,1,544]` | `torch.uint8` | payload 512B + 16×BF16 scale，仅改写命中的 slot |

`descale_qr` 使用 int8 字节载体，与输入 scale 的 E8M0 dtype 不同。公开的 `qr` 是
RMSNorm 后的 BF16 值；QB 仍消费内部 E4M3FN payload 与 `descale_qr`，但量化结果不再作为
公开 QR 输出。
KV 不额外返回 scale，也不返回独立的 K、V 张量。

- 所有输入张量须逻辑连续且位于同一 NPU；调用前设置对应的当前设备。ND 对应 format 0 或 2，NZ 对应 format 29。
- H、R、D 须为 64 的倍数；N 必须为 16 或 64；Dr 为正数且是 16 的倍数，满足 `Dr <= 64`、`Dr <= D/2`。完整验收范围为上述固定生产维度的 T 泛化。
- `kv_cache` 最后一维必须等于 `D + (D/32)*2`（D=512 时为 544）；未命中的 slot 必须逐字节保持不变。
- `x[t]`、位置表和 `cache_index[t]` 须对应同一个 token；算子不从 T 推断位置或所属序列。
- 调用方须保证 `0 <= cache_index[t] < P*BS`，且一次调用内没有重复 slot。host 侧只校验元数据，不读取设备索引值；未命中的 slot 保持原始字节不变。
- 公共入口自动选择分块并查询平台核数，只接受 `norm_eps` 这一可选关键字参数。

## 实现说明

### host 侧准备与编译复用

`prepare_attn_prologue` 按层校验标量、张量公共元数据、投影/scale/RMSNorm/RoPE/cache 各模块及 ND/NZ 布局，
随后规划分块、分配输出与工作区、补齐物理尾行。分块器继续检查 L1/L0/UB 容量与 NZ 网格。
L1 复用路径还将 QB 权重预打包为输出特征宽 256、归约宽 320 的面板，包含一次设备与 host 侧之间的数据往返。
这些准备操作不计入 kernel 耗时。

`.run()` 首次获取编译产物，后续直接复用。同进程中，相同分块与张量契约
（shape、stride、dtype、device、ND/NZ 布局）可共享产物，LRU 缓存最多保留 32 项。
`norm_eps` 作为运行时 FP32 参数，其数值变化不触发重新编译。使用 `@host` 启动入口和
`cannbotdsl.compile()`；kernel 辅助函数使用 `@jit`，寄存器计算使用 `vf(mode="simd")`。
设备实现由 `_Launcher` 在编译前根据 tiling 绑定入口类。L1 复用使用独立入口；NonL1Reuse 按 A 阶段的
Branch/Generic 编排和 B 阶段的 Resident/Streaming 缓冲策略组合成四个单继承入口类。路径无关的量化、
RoPE 等原语位于公共基类，device 方法不再包含 `l1_reuse` 或 `b_resident` 分支。

prepared 对象复用自己的输入参数、工作区和输出；重复 `.run()` 会覆盖同一组输出与目标 cache slot。
需要保存旧结果时应另行复制。准备阶段可能复制输入或预打包权重，因此输入或权重变化后应重新 prepare。
编译缓存只缓存产物，不缓存输入或工作区。

### 分块与数据复用

核数通过平台接口查询，分块不依赖具体机器型号。规划器优先尝试 L1 复用，条件不满足或容量不足时回退通用路径：

| 路径 | 选择条件 | 分块与复用方式 |
|---|---|---|
| L1 复用 | 固定生产维度、`1 <= T <= 256`，QB 为完整且无额外填充、偏移为 0 的 NZ 存储，并通过容量检查 | 使用 8 个 K 分区；token 块按 16 行对齐，候选不超过 96 行。QA/KV 共享驻留 X 与 L0A，A1/B 复用 L1/L0 通道，QB 使用预打包面板和权重预取。 |
| 通用 split-K | `T <= 256` 且未采用 L1 复用 | 根据并行度与容量选择 K 分区数，包含 4 分区优化；各 Cube 输出 FP32 部分和，由 Vector 合并。 |
| 通用 split-T | `T > 256` | 主要沿 token 轴分配任务；大 prefill 在容量允许时采用单个 K 分区，减少工作区与归约开销。 |

通用路径中的 `a1_window` 在容量允许时将更宽的 X/scale 窗口驻留 L1。
`b_resident` 按 token 块复用 QR/scale 和 RoPE 表，跨 head 使用双槽权重预取。
所有候选均检查容量、对齐和 NZ 网格；内部补齐行只服务物理搬运，公开输出仍为 T 行。

### Cube / Vector 流水与同步

通用路径根据 QA/KV 任务量决定是否拆分队列：拆分后增加的 Cube 任务波数不超过共享队列的 5% 时，
先发布 QA，使 KV 的 Cube 投影与 QR 的 Vector 归一化、量化重叠；否则使用 QA/KV 共享队列。
QR 就绪后 Cube 进入 QB，Vector 将 KV 后处理穿插在 QB 结果的消费之间，并在末尾补齐剩余 KV 任务。
L1 复用路径同样在 QR 发布后允许 QB 与 KV 后处理重叠。

同步按数据依赖设置：Cube 发布投影工作区，Vector 发布 QR/scale，使用生产者流水排空、同类核汇合和定向通知保证可见性。
Channel 管理 L1/L0/UB 缓冲槽的生产与消费。RoPE 尾块拼装后保留 `vmem_bar("vst_vld")`，
确保同一 VF 内先写入 UB、再读取的顺序。实际重叠程度取决于 shape。

QA/KV 部分和通过 FP32 GM 工作区传递。令 `work_rows=ceil(T/bm_a1)*bm_a1`，则：

```text
ws_qa.shape = [work_rows * split_k, R]
ws_kv.shape = [work_rows * split_k, D]
workspace_bytes = 4 * work_rows * split_k * (R + D)
```

`prepared.workspace_bytes` 只统计这两个交换区，不包含输出、补齐缓冲和预打包权重。
通用 QA split-K 使用补偿求和，降低抵消误差对 RMSNorm 和 FP8 量化的影响；L1 复用路径使用顺序求和。

### 代码定位

| 模块 | 主要实现 |
|---|---|
| 分块规划与路径选择 | [`AttnPrologueTiling.plan`](attn_prologue_tiling.py)、[`select_attn_prologue_kernel`](attn_prologue_kernel.py) |
| 设备入口类 | [`L1ReuseAttnPrologueKernel`](attn_prologue_kernel_l1_reuse.py)；[NonL1Reuse](attn_prologue_kernel_non_l1_reuse.py) 的 `Branch/Generic × BResident/BStreaming` 四种组合 |
| QA/KV 投影 | 非驻留路径 [`_a1_tile`](attn_prologue_kernel_non_l1_reuse.py) → `_mm_a`；L1 复用路径 [`_resident_a_project`](attn_prologue_kernel_l1_reuse.py) |
| QR 部分和归约 | 非驻留路径 `_acc_partial_qa`；L1 复用路径 [`_acc_all_partials`](attn_prologue_kernel_common.py) |
| QR RMSNorm 与动态量化 | [`_quantize_qr` 和 `_pack_scale`](attn_prologue_kernel_common.py) |
| QB 投影与 Q 写回 | 非驻留路径 [`_stage_b`](attn_prologue_kernel_non_l1_reuse.py)；L1 复用路径 [`_resident_b_cube`](attn_prologue_kernel_l1_reuse.py) |
| KV 归约、后处理与写回 | 非驻留路径 `_service_kv`；L1 复用路径 [`_reduce_normalize_a`](attn_prologue_kernel_l1_reuse.py)；公共 [`_finish_kv_row`](attn_prologue_kernel_common.py) |
| 尾部 RoPE 与跨阶段发布 | 公共 [`_rope_pair` 和 `_publish_qr`](attn_prologue_kernel_common.py)；非驻留 [`_publish_a1`](attn_prologue_kernel_non_l1_reuse.py) |
| 公共入口与编译缓存 | [`prepare_attn_prologue` 和 `_Launcher.compile_cached`](attn_prologue.py) |

## 网络侧封装（替换现有 prolog）

`NetworkAttnPrologue` 对现有 `Attention.compute_win_cache(self, x, kv_cache, attn_metadata)`
提供等价入口，网络侧只替换调用：

```python
net = NetworkAttnPrologue(
    wqa, wkv, wqb, descale_wqa, descale_wkv, descale_wqb,
    norm_weight_qr, norm_weight_kv,
    attention_type="win", weight_nz=True, norm_eps=1e-6,
)
win_kv, qr, q = net.compute_win_cache(
    x_bf16, kv_cache, attn_metadata,
    attention_type="win", norm_eps=1e-20,
)
```

- `kv_cache` 可以是 `KVCache` 对象，封装层取 `kv_cache.win_kv`。
- `attention_type` 必须按当前层显式传入（`"win"` 取 `cos_sin["win"]`，否则取 `"comp"`）；
  `norm_eps` 使用网络值（通常 `1e-20`），不要沿用默认。
- `cos_sin[key]` 是 `(cos, sin)` 元组；封装层按 `rope_sin=sin, rope_cos=cos` 传给 kernel。

- `x`：BF16 `[T,H]`；每步由封装量化为 MXFP8（group=32），不融合进主 kernel。
- 权重：逻辑 `[K,N]` E4M3，weight_scale `[K/64,N,2]` uint8 E8M0；构造时识别 ND/NZ，
  一次性转置为内部 `[N,K]` 并预打包；gamma 为 BF16，构造时转 FP32。
- `attn_metadata["cos_sin"]`：`{"win": (cos, sin), "comp": (cos, sin)}`，均为
  FP32 `[T,1,1,Dr]`；`attention_type=="win"` 取 `"win"`，否则取 `"comp"`。
- `attn_metadata["cache_slot_mapping"]["win_kv"]`：INT64 展平索引，允许 `-1`/越界；
  无效 slot 只跳过 cache 写回，Q/QR 仍计算。
- 返回 `(win_kv, qr, q)`；`win_kv` 是原地更新的 `kv_cache`。

prepared 对象可通过 `update_inputs(x, descale_x, rope_sin, rope_cos, cache_index, norm_eps)`
每步刷新；它直接改写 kernel 读取的补齐缓冲，避免图回放沿用 capture 时的数据。

## 调用示例

安装与当前 Python 版本匹配的 CANNBotDSL 包，并加载 CANN、PyTorch 与 torch_npu 环境。
在仓库根目录运行以下示例；输入构造复用测试工具，生成 `batch=12, sequence=6, T=72` 的数据：

```python
import dataclasses
import sys
import torch
import torch_npu
from samples.attention.attn_prologue.attn_prologue import prepare_attn_prologue, to_nz

sys.path.insert(0, "test/attention/attn_prologue")
from test_attn_prologue import Case, SharedWeights, make_case_inputs, device_input

torch.npu.set_device(0)
inputs = make_case_inputs(Case("decode", 12, 6), SharedWeights())
values = {
    field.name: device_input(field.name, getattr(inputs, field.name), "npu:0", to_nz)
    for field in dataclasses.fields(inputs)
}
prepared = prepare_attn_prologue(**values, norm_eps=1e-6)
q, qr, descale_qr = prepared.run()
torch.npu.synchronize()
# q 为最终 Q；qr 和 descale_qr 为中间 MXFP8 结果；values["kv_cache"] 已被更新。
```

## 精度验证

测试实现集中在 [test_attn_prologue.py](../../../test/attention/attn_prologue/test_attn_prologue.py)：
输入构造、独立 CPU golden、FP8 判据、FP64 复核、pytest 用例及批量命令行入口均在此文件。
pytest 直接调用算子，批量运行复用相同的完整输出校验逻辑；每个用例独立生成输入和 cache，同一批次共享权重。
CPU 测试与 `--list` 不加载 NPU 运行时。

在仓库根目录运行：

```bash
# CPU：12 个用例覆盖 FP8 判据，以及 Q/QR/scale/KV/未命中 cache 的损坏检测。
bash scripts/ci/run_tests.sh --ops=attn_prologue --mode=cpu
# CI 冒烟：4 个正向用例，覆盖 T=1/72 decode 与 T=1024/2048 prefill。
bash scripts/ci/run_tests.sh --ops=attn_prologue --mode=npu
# 或直接调用 pytest；非默认设备可设置 ATTN_PROLOGUE_TEST_DEVICE=npu:1。
python3 -m pytest test/attention/attn_prologue -m npu -v
# 查看标准矩阵（不需要 NPU）。
python3 test/attention/attn_prologue/test_attn_prologue.py --list
# 单个 decode 用例。
python3 test/attention/attn_prologue/test_attn_prologue.py --case-id decode-b12-s6-t72 --device npu:0 --output results/attn_prologue_smoke.jsonl
# 完整的 12 个 decode 和 20 个 prefill 用例。
python3 test/attention/attn_prologue/test_attn_prologue.py --device npu:0 --output results/attn_prologue_accuracy.jsonl
```

标准矩阵为 decode 的 `batch∈{1,4,8,12,16,32}`、`sequence∈{1,6}`，以及 prefill 的
`batch∈{1,4,8,16,32}`、`sequence∈{1024,2048,4096,8192}`，最大 T 为 262144。
runner 使用当前平台核数，不自动限制为 32 核。

Q 须全部有限，且 `abs(got-ref) > 0.02 + 0.02*abs(ref)` 的元素比例不超过 0.1%。
QR、scale 和 KV 与 FP32 参考存在字节差异时，使用独立 FP64 参考进行整行值域复核，最终硬失配须为 0；
未命中的 cache slot 须逐字节保持不变。

源仓库在 2026-09-29 使用 Ascend950PR、32 Cube / 64 Vector完成以下验证（非本次迁移的重跑范围）：
32/32 标准用例、14 个边界用例、35 项输入契约检查全部通过。
另通过 12 次缓冲区预填异常值后的重复运行检查，用于确认计算不依赖缓冲区初始值；
编译缓存复用和 3 个运行时 `norm_eps` 值也通过验证。
Q 最大超差比例为 0.009741%，QR/KV 值域硬失配和未命中 cache 改写均为 0。

## 性能

以下为上述 32 Cube / 64 Vector 环境的实测记录。
每次调用前使用 256 MiB FP16 缓冲区执行 ArgMax 读操作驱逐 L2，并同步；每例采集 10 次目标 kernel 设备耗时，取中位数。
编译、输入准备和驱逐操作均排除在计时之外。`torch.npu.empty_cache()` 不能替代 L2 驱逐。

| T | 耗时中位数（µs） | 极差 / 中位数 |
|---:|---:|---:|
| 1 | 40.61 | 16.68% |
| 72 | 45.45 | 8.94% |
| 192 | 59.05 | 6.55% |
| 1024 | 241.05 | 1.80% |
| 2048 | 445.29 | 1.23% |
| 4096 | 882.45 | 1.06% |
| 262144 | 53604.47 | 1.66% |

采集时未锁定频率；T=1 在复测与独立采集中仍有较大波动，仅作参考。
这些数据为当前版本绝对耗时，不用于推断相对历史版本的稳定收益。
以上为源仓库历史性能记录，不代表本次 CI 验证进行了性能复测。
