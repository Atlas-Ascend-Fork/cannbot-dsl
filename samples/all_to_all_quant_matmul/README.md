# AllToAll + QuantMatmul（MC2 通算融合）

基于 CANNBot-DSL 实现的 TP/EP 推理"激活 AllToAll + 本地 MX 量化 MatMul"
MC2 通算融合算子，面向 Ascend 950（NPU ARCH 3510）。

## 算子介绍

- 接口功能：完成多卡 AllToAll 通信与 MX 量化 MatMul 计算的融合，**先通信后计算**。单 MIX Kernel 内 AIV 子核后台推送本卡激活分片、AIC 子核流水计算，以计算掩盖通信；每卡独立输出全量 `(M, N)`。
- 计算公式：每卡持有 K 分片激活 `a[M, Ka]`（`Ka = K / rankSize`）与全量权重 `b[K, N]`；MX 量化场景 `scaleA` shape 为 `(M, ceil(Ka/64), 2)`、`scaleB` 为 `(K/64, N, 2)`，可选一维 fp32 `bias`：

$$
commOut = AlltoAll(a.view(rankSize, M, Ka)) \\
permutedA = commOut.permute(1, 0, 2).view(M, rankSize \cdot Ka) \\
commScale = AlltoAll(scaleA.view(rankSize, M, \lceil Ka/64 \rceil, 2)) \\
permutedScale = commScale.permute(1, 0, 2, 3).view(M, \lceil Ka/64 \rceil \cdot rankSize, 2) \\
Y = \sum_{0}^{\lceil K/64 \rceil} (permutedA \,@\, b) \cdot (permutedScale \cdot scaleB) + bias
$$

### 特性与约束

| 特性与约束 | 说明 |
| :--- | :--- |
| 数据类型 | A/B 量化 dtype（`a_dtype`/`b_dtype`）：`e4m3` / `e5m2`（MXFP8）或 `fp4`（MXFP4，2 值/字节打包），以 uint8 字节流进出；Scale 恒为 E8M0（uint8）；bias 为一维 fp32 `(N,)`（可选）；输出 `out_dtype`：bf16 / fp16 / fp32 |
| **B 的 ZN 预排布契约** | `b` 的宿主侧字节必须预先按 ZN 分形排布（参见 `test/all_to_all_quant_matmul/test_all_to_all_quant_matmul.py` 的 `prearrange_b_zn`），算子内部按 ZN 视图解读、不做格式搬运；列宽补齐到整 tile 值 `n_padded = ceil(n/base_n)·base_n`（`[n, n_padded)` 列为零——identity 引擎的 zn→zn 拷贝要求源/目的分形对齐宽度相等，b 侧不允许尾块钳位），AOT spec 形状为 `(K, n_padded)` |
| Shape 约束 | **`K` 必须是 `rank_size·base_k` 的倍数**，且 `base_k` 为 64、`base_n` 为 16 的整数倍（K 尾块不支持，host 校验不满足即报错；真实网络形状天然满足）；M/N 任意值（M 尾块：越过 m 的 m tile 由 kernel 整块跳过、a/scale_a 按 `m_padded = ceil(m/base_m)·base_m` 行补零分配，N 尾块经整 tile 补齐视图支持） |
| epoch 上限 | 全量物化 inbox，`head_cnt = ceil(m/head_m) ≤ 14`（更大 m 需 inbox 槽位复用，暂不支持） |
| 通信资源 | `required_comm_buffer_bytes()` 给出 inbox 字节数（urma 为解析上界；ub 为 SELF_REGISTERED 注册下限，窗口容量不足时报错——受全量物化 inbox 总量 ≈ `R·M·(ka+scale_ka)` 限制，与 head_m 无关）；`used_core_num ≥ rank_size` |
| 多核并行 | 单 MIX Kernel（`KERNEL_TYPE_MIX_AIC_1_2`），1 AIC + 2 AIV subcore；输出 tile 按 block cyclic 分核 |
| 支持架构 | Ascend 950（NPU ARCH 3510）；CANN 9.2.0-beta.2 / 9.3.0；cannbot-dsl 0.7.0 |
| 范围外 | HCCL 集合通信版变体、DFX dump、ACLNN 封装、动态 shape |

### 能力状态

当前支持：顺序计算（rank 趟 `r = 0..R-1`）、M/N 任意值尾块（N 含非
16 倍数）、一维 fp32 bias、TransB 用户朝向输入（b `(N,K)` + scale_b
`(N,G,2)` 由测试 harness 侧归一到 kernel ABI，核内契约不变）、
e4m3/e5m2/fp4 量化 dtype、bf16/fp16/fp32 输出。暂不支持：local-first /
L0C 累加 / GM 累加等计算模式、TransA、FP8×FP4 混合、
head_cnt > 14（inbox 槽位复用）。

## 接口

```python
class AllToAllQuantMatmul:
    def __init__(self, m, k, n, rank_size, comm_mode, base_m=128,
                 base_n=128, base_k=64, head_m=None, used_core_num=8,
                 has_bias=False, out_dtype="bf16", a_dtype="e4m3",
                 b_dtype="e4m3")
    def run(self, data_ctx, barrier_ctx, a_gm, scale_a_gm, b_gm,
            scale_b_gm, y_gm, bias_gm)     # @host 入口
    def aot_compile_specs(self)         # AOT 参数规格表
    def required_comm_buffer_bytes(self)
```

| 参数 | 说明 |
| :--- | :--- |
| `m, k, n` | 全局矩阵维度；`k` 必须是 `rank_size·base_k` 的倍数 |
| `rank_size` | 卡数（R） |
| `comm_mode` | 数据面通信模式（默认 `urma`）：`urma` = put/flush 传输流；`ub` = AIV 直写对端 UBMEM 窗口（数据面需 UBMEM 自注册窗口 ≥ `required_comm_buffer_bytes()`） |
| `base_m/base_n/base_k` | 基本块（默认 128/128/64，测试用 256/256/128）；`base_k` 须为 64、`base_n` 须为 16 整数倍 |
| `head_m` | 每个 epoch 的通信 M 跨度；默认 `max(⌈used_core_num/n_tiles⌉, ⌈m/(4·base_m)⌉)·base_m` |
| `used_core_num` | launch 核数，需 ≥ `rank_size` |
| `has_bias` | bias 编译期开关；`True` 时 AOT 契约含 `(n_padded,)` fp32（`[n, n_padded)` 补零），`False` 时为长度 0 占位（调用方传 `torch.empty(0)`） |
| `out_dtype` | 输出 dtype：`bf16` / `fp16` / `fp32`（默认 `bf16`） |
| `a_dtype` / `b_dtype` | 量化 dtype：`e4m3` / `e5m2` / `fp4`（默认 `e4m3`）；`fp4` 须双方同为 `fp4` 且 `n` 为偶数，此时 b 的 GM 契约为 ND 打包字节 `(K, n_padded/2)` 而非 ZN 预排布 |
| `data_ctx` / `barrier_ctx` | `StructRef[CommContext]`：数据窗口（A/scale_a inbox 与 ack 槽位宿主：urma 为 HCCL 窗口、ub 为自注册 UBMEM 窗口）/ 同步窗口（UBMEM，launch 计数器宿主）；均经测试侧 `get_comm_context` 创建 |

`aot_compile_specs()` 契约：`StructRef[CommContext] ×2` + uint8 字节流 ×4
（a/scale_a/b/scale_b）+ `out_dtype` 输出 `y` + fp32 `bias`。尾块安全
契约（对齐形状下与紧凑形状完全一致）：a 为 `(m_padded, ka)`、scale_a 为
`(m_padded, scale_ka)`（`[m, m_padded)` 行补零——核内本地视图按 head_m 谎报
行数，尾 epoch 的越界读须落在补零区）；fp8 的 b 为 ZN 预排
`(k, n_padded)`、fp4 的 b 为 ND 打包 `(k, n_padded/2)`，scale_b 为
`(k/64, n_padded, 2)` 列内配对，bias 为 `(n_padded,)` 或长度 0 占位；
y 保持 `(m, n)`（写出端由 tile_slice 尾块钳位裁剪）。

## 快速开始

多卡场景通过仓级多卡测试基建驱动（编译一次、late-acquire 执行）：

```python
import sys
sys.path.insert(0, "samples/all_to_all_quant_matmul")

import cannbotdsl
from all_to_all_quant_matmul import AllToAllQuantMatmul

M, K, N, R = 2048, 3584, 4096, 2
op = AllToAllQuantMatmul(M, K, N, R, "urma", base_m=256, base_n=256,
                         base_k=128, used_core_num=28, has_bias=True)
main = cannbotdsl.compile(op.run, *op.aot_compile_specs())   # rank 中立编译，无设备
# 每卡进程内：创建 data_ctx/barrier_ctx 与 NPU tensor 后调用
# main(data_ctx, barrier_ctx, a, sa, b_view, sb, y, bias)
# 无 bias 编译（has_bias=False）时 bias 传 torch.empty(0, dtype=float32)
```

## 测试

测试代码位于 `test/all_to_all_quant_matmul/`：多卡编排 runner 与七钩子
实现自含于 `test_all_to_all_quant_matmul.py`；性能基线驱动为
`test_all_to_all_quant_matmul_perf.py`（cases.csv 首列带名的行即性能
用例，`--cases` 按名选择）。

```bash
# L0 无卡单测（AST 校验 + codegen 断言）
pytest test/all_to_all_quant_matmul/test_all_to_all_quant_matmul.py -v

# L1 多卡精度（2 卡；4 卡改 --devices 0,1,2,3；--comm-mode 默认 urma）
python3 test/all_to_all_quant_matmul/test_all_to_all_quant_matmul.py \
    --csv test/all_to_all_quant_matmul/cases.csv --devices 0,1 --comm-mode urma

# 两阶段：无卡编译 → 卡空闲时执行
python3 test/all_to_all_quant_matmul/test_all_to_all_quant_matmul.py \
    --csv test/all_to_all_quant_matmul/cases.csv --stage compile \
    --comm-mode urma --bundle-root ./bundles
python3 test/all_to_all_quant_matmul/test_all_to_all_quant_matmul.py \
    --stage execute --bundle-root ./bundles --devices 0,1 --comm-mode urma \
    --wait-devices 3600

# 连续 launch 一致性（--launches > 1）
python3 test/all_to_all_quant_matmul/test_all_to_all_quant_matmul.py \
    --csv test/all_to_all_quant_matmul/cases.csv --devices 0,1 --comm-mode ub \
    --launches 2

# 性能基线（精度门禁 + 10 轮计时取后 5 平均，每 case 独立进程）
python3 test/all_to_all_quant_matmul/test_all_to_all_quant_matmul_perf.py \
    --csv test/all_to_all_quant_matmul/cases.csv --devices 0,1,2,3 \
    --comm-mode urma --cases case1
```

当前精度容差为临时值 `{y: (1e-2, 1e-2)}`（rtol, atol）
