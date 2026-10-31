# Dynamic MX Quant SR

基于 CANNBot-DSL 实现的 MXFP8 随机舍入量化算子，支持 FP32、BF16 输入，
面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT）。

## 算子介绍

沿指定量化轴，每 32 个元素共享一个 E8M0 scale，将输入量化为 E4M3FN 或 E5M2。
记分组内最大绝对值为 `amax`，普通有限值分组的计算过程为：

$$
a_g = \max_{i\in g}|x_i|,\qquad
s_g = \operatorname{Scale}_{\mathrm{scale\_alg}}(a_g),\qquad
y_i = \operatorname{SR}_{\mathrm{FP8}}\!\left(\operatorname{Normalize}(x_i,s_g)\right).
$$

随机舍入使用 16 位随机字段和舍弃部分的最高 16 位，有限值结果采用饱和处理。
最后一组不足 32 个元素时只处理有效输入。

| 特性与约束 | 说明 |
| :--- | :--- |
| 输入类型 | `torch.float32`、`torch.bfloat16` |
| 输入布局 | 连续、非空、至少一维的 NPU Tensor；支持非零 storage offset |
| 量化轴 | 任意合法正轴或负轴；轴长 K 为 16 的倍数，FP32 要求 K≥32 |
| 分组大小 | 固定为 32 |
| 输出类型 | E4M3FN 或 E5M2，shape 与输入一致 |
| Scale 类型 | `torch.float8_e8m0fnu`，每两个 block 的 scale 组成一对 |
| Scale 算法 | `scale_alg=0/1`，默认 1 |
| 随机状态 | 从输入设备的框架 generator 预留，支持保存状态后重放 |
| 支持架构 | NPU ARCH 3510，使用 AIV |

### Scale 与数值语义

| 算法 | 规则 |
| :--- | :--- |
| `scale_alg=0` | 按 `amax` 的指数向下取共享指数；普通有限组为 `floor(log2(amax))-8`（E4M3FN）或 `floor(log2(amax))-15`（E5M2），并处理指数下界 |
| `scale_alg=1` | 对 `amax × FP32(1/MAX_FINITE)` 的 FP32 乘积向上取二进制幂，包含乘积下溢处理 |

`max_low_bound` 仅对算法 1 的非全零组生效，此时使用 `max(amax, max_low_bound)`。
E8M0 有限 scale 字节 `s` 表示 `2^(s-127)`，`0xFF` 表示无效 scale。
算法 0 算出的 scale 字节为 0 时，归一化乘数取 0；算法 1 的非零有限组在 scale 字节为 0 时使用 `2^127`。

FP32 输入使用 FP32 归一化乘积。BF16 的参考语义是归一化后按 RN 舍入到 BF16；
实现按数值范围选择等价的编码路径，最终输出与独立 CPU 参考逐字节比较。

| 分组情况 | 输出约定 |
| :--- | :--- |
| block32 内存在 Inf 或 NaN | scale 为 `0xFF`，该 block 的有效 FP8 元素均为 `0x7F` |
| 全零 block32 | scale 为 0，元素保持零的符号 |
| 最后一个 block 不成对 | 补齐的 scale 字节为 0 |

实现见 [dynamic_mx_quant_sr.py](dynamic_mx_quant_sr.py)。Host 根据量化轴、后续维度宽度、
dtype 和工作量选择尾轴、stride2、wide 或转置布局；设备侧通过 SIMD 完成 scale、归一化和 FP8 打包，
按路径使用 SIMD 或 SIMD/SIMT 组合执行完整十轮 Philox。核数由输入设备当前流的 `vector_core_num` 查询，
UB tiling 按实际容量选择，并为 SIMT 预留 40 KiB。

## 快速开始

按照 [仓库 README](../../README.md) 安装依赖并加载 CANN 环境，本算子使用以下配套版本：

| 依赖 | 版本或要求 |
| :--- | :--- |
| Python | 3.12 |
| PyTorch / torch_npu | `2.10.0+cpu` / `2.10.0`；提供 E4M3FN、E5M2 和 E8M0 类型 |
| CANN | `9.2.0`，目标架构 `dav-3510` |
| CANNBot-DSL | `0.8.0` |
| 原生桥接构建 | C++ 编译器和 Ninja，均在 `PATH` 中 |

配套 wheel 为 `cannbot_dsl-0.8.0-cp312-cp312-manylinux_2_28_x86_64.whl`，
SHA256 为 `f9481b882b7290956cee09dfc3014ad1c62bba77c802288271ecfa0cf5359f99`。
算子实现及框架 RNG 桥接均位于 [dynamic_mx_quant_sr.py](dynamic_mx_quant_sr.py)。
首次调用会按需编译内嵌的原生桥接，生成文件保存在 PyTorch 扩展缓存目录；Graph 捕获前需完成预热。

在仓库根目录执行：

```python
import torch
import torch_npu

from samples.dynamic_mx_quant_sr.dynamic_mx_quant_sr import dynamic_mx_quant_sr

# 尾轴量化
x = torch.randn((8, 4096), dtype=torch.float32, device="npu")
y, scale = dynamic_mx_quant_sr(x, dst_type=24, scale_alg=1)
assert y.shape == (8, 4096)
assert scale.shape == (8, 64, 2)

# 非尾轴量化
v = torch.randn((2, 128, 64), dtype=torch.bfloat16, device="npu")
vq, vs = dynamic_mx_quant_sr(v, axis=-2, dst_type=23, scale_alg=0)
assert vs.shape == (2, 2, 64, 2)
```

首次调用构建 RNG 桥接并编译所需内核，相同编译特化的后续调用复用缓存。
同一布局和优化类别内支持动态 shape；dtype、目标格式、scale 算法、下界、UB 容量、
布局、优化类别和运行时编译身份共同决定编译特化。

编译期间保持设备、工具链和 codegen 配置稳定。已加载 SDK 的文件指纹在进程内复用，
`clear_caches()` 同时清除此指纹和 Program 缓存；原地编辑 SDK 源码或头文件后应先清缓存。
替换已加载的原生库需重启进程。设备、编译选项和工具链身份仍在每次公开调用时捕获。

## 接口说明

```python
def dynamic_mx_quant_sr(
    input,
    *,
    axis=-1,
    round_mode="stochastic",
    dst_type=24,
    block_size=32,
    scale_alg=1,
    dst_type_max=0.0,
    max_low_bound=0.0,
):
    ...
```

除 `input` 外，所有参数均通过关键字传入。

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `input` | 必填 | 连续、非空的 NPU FP32/BF16 Tensor |
| `axis` | `-1` | 合法的正轴或负轴 |
| `round_mode` | `"stochastic"` | 随机舍入 |
| `dst_type` | `24` | 24=E4M3FN，23=E5M2 |
| `block_size` | `32` | 固定为 32 |
| `scale_alg` | `1` | 0 或 1 |
| `dst_type_max` | `0.0` | 固定为 0 |
| `max_low_bound` | `0.0` | 有限非负数，按 FP32 舍入；算法 0 只接受 0，转换后须为有限值 |

整数属性要求 Python `int`，拒绝 bool；`dst_type_max` 和 `max_low_bound` 接受 Python `int/float`，拒绝 bool。
输入保持只读，每次调用分配独立输出。

### 返回值

返回 `(y, mx_scale)`，两者均为与输入同设备的连续 Tensor：

| 返回值 | Shape | dtype |
| :--- | :--- | :--- |
| `y` | 与输入相同 | `torch.float8_e4m3fn` 或 `torch.float8_e5m2` |
| `mx_scale` | 将输入 shape 的量化轴长度 K 替换成 `ceil(K/64)`，再在末尾追加长度 2 | `torch.float8_e8m0fnu` |

### 随机状态与 Graph

每次调用从输入设备的框架 generator 预留随机区间。令 `G=numel/16`，
预留量为 `4*ceil((G+63)/16384)` 个 32 位随机字，包含未使用的末尾 SIMD padding。
64 位 seed 和 offset 完整传递，offset 四对齐，`offset/4` 是 Philox block 位置。

先生成输入，再保存和恢复输入设备的框架 RNG 状态：

```python
with torch.npu.device(x.device):
    torch.npu.manual_seed(1234)
saved = torch.npu.get_rng_state(x.device)
y0, s0 = dynamic_mx_quant_sr(x)
y1, s1 = dynamic_mx_quant_sr(x)
torch.npu.set_rng_state(saved, x.device)
y0_replay, s0_replay = dynamic_mx_quant_sr(x)
```

相同输入和 RNG 状态产生相同输出；连续调用使用不同随机区间，输出字节可能相同。
标准单流 NPU Graph 捕获前需按相同规格预热，使桥接初始化和编译在图外完成；支持顺序 replay 和状态恢复。
不支持 secondary-stream capture。`torch.compile`/torchair 及多 Host 线程并发 eager/replay 不在验证范围内。
eager 状态预留检查 `offset + increment` 的 64 位溢出；Graph replay 不检查全局 offset 回绕。

### 随机字段映射

随机地址按量化轴移到末尾后的逻辑元素位置计算，独立于 tile 大小和实际 AIV 数。
设逻辑位置为 `i`，每个随机字提供 `F` 个字段：

```text
g       = i // (4 * F)
stream  = g % 16384
draw    = g // 16384
q       = offset // 4 + draw
counter = (0, stream, low32(q), high32(q))
key     = (low32(seed), high32(seed))
words   = Philox4x32_10(counter, key)
word    = words[(i // F) % 4]
```

通用路径 `F=4`。令 `H=word>>16`、`L=word&0xFFFF`、`R=rotate_left32(word,8)`，字段依次为
`[H, R & 0xFFFF, L, R >> 16]`。

以下路径使用 `F=8`：

- FP32：尾轴编译布局、K 为 256 的倍数且 `numel >= 2*1024*1024`。将 `word` 拆为从低到高的字节 `b0/b1/b2/b3`，
  按 `[(0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(3,0),(0,3)]` 选择字节对，低字节在前，组成 16 位字段。
- BF16：K 为 256 的倍数，采用尾轴编译布局或 width16 直通布局。令 `f=i%8`，目标尾数位数 `M` 为 3（E4M3FN）或 2（E5M2），
  字段为 `rotate_right32(word, ((f//2)*8 + (1+M)*(f%2) + 32-9-M) % 32) & 0xFFFF`。

尾轴编译布局包括原始尾轴及非尾轴转置后的布局，字段之间具有确定性关联，并非相互独立的随机样本。
各路径均按上述统一规则预留状态，独立 CPU 实现见
[测试脚本](../../test/dynamic_mx_quant_sr/test_dynamic_mx_quant_sr.py) 中的 `dynamic_mx_quant_golden`。

## 精度测试

测试文件为 [test_dynamic_mx_quant_sr.py](../../test/dynamic_mx_quant_sr/test_dynamic_mx_quant_sr.py)，
脚本内的独立 CPU golden 逐字节比较 FP8 和 E8M0 输出。
覆盖两种输入类型、两种目标格式、两种 scale 算法、特殊值、尾块、存储偏移、尾轴与非尾轴、
动态二进制复用，以及随机状态推进、重放和 Graph。单元素测试包含 BF16 全编码与 FP32 指数边界。

按照快速开始安装配套依赖并加载 CANN 环境，再安装 `numpy`、`ml_dtypes` 和 `pytest`。
测试模块显式导入 `torch_npu`，不依赖后端自动加载。Host 用例同样需要可导入的 `torch_npu`，
但不初始化 NPU 设备。在仓库根目录执行：

```bash
# 全部用例，需要单张支持架构的 NPU 和 AscendC 工具链
python -m pytest test/dynamic_mx_quant_sr/test_dynamic_mx_quant_sr.py -v

# 仅 Host 用例
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest test/dynamic_mx_quant_sr/test_dynamic_mx_quant_sr.py -m "not npu and not ascendc_toolchain" -q

# 仅 NPU 用例
python -m pytest test/dynamic_mx_quant_sr/test_dynamic_mx_quant_sr.py -m npu -q
```

使用仓库统一 CI 时，将 `SR_PYTHON` 设为已安装配套 wheel、测试依赖及 `pytest-xdist` 的 Python 3.12 解释器：

```bash
PYTHON="$SR_PYTHON" CI_TEST_WORKERS=16 bash scripts/ci/run_tests.sh --ops=dynamic_mx_quant_sr --mode=ci
```

## 性能数据

### 测试环境与计时口径

设备为 Ascend950DT_9574（56 个可用 AIV），CANN `9.2.0`、CANNBot-DSL `0.8.0`，
Python `3.12.14`、PyTorch `2.10.0+cpu`、torch_npu `2.10.0`、ml_dtypes `0.6.0`。
输入使用 CPU `torch.randn * 7` 后转换到源 dtype，输入 seed 为 `20260920`，框架 RNG seed 为 `0`。
各配置独立串行运行，复用输入，不显式清 L2。

### 公开入口主核耗时

固定 `dst_type=24`（E4M3FN）、`scale_alg=1`、`axis=1`、`block_size=32`、`max_low_bound=0`。
公开入口预热 20 次、采样 120 次，随机状态逐次推进；表中用例的 FP8 与 E8M0 输出均通过独立 CPU golden 校验。
msprof 使用 `--task-time=on --ai-core=on --aic-metrics=PipeUtilization`，统计量化主核的 `Task Duration`，
包含核内完整 Philox，排除 Host、编译、分配、框架状态预留和布局转换耗时。单位为 **µs**。

| 路径 | 输入类型 | Shape | AIV 数 | 最小值（µs） | 中位数（µs） |
| --- | --- | --- | ---: | ---: | ---: |
| WIDE8 | FP32 | 32×512×8 | 56 | 3.743 | 4.262 |
| WIDE16 | BF16 | 32×512×16 | 56 | 10.157 | 10.765 |
| WIDE32 | FP32 | 32×512×32 | 56 | 9.843 | 10.311 |
| WIDE64 | BF16 | 32×512×64 | 56 | 11.794 | 12.257 |
| 尾轴相邻面板 | FP32 | 3640×256 | 56 | 5.026 | 5.593 |

复现 WIDE8 配置：

```bash
python test/dynamic_mx_quant_sr/test_dynamic_mx_quant_sr.py \
    --shape 32x512x8 --src fp32 --dst e4m3fn --scale-alg 1 --axis 1 \
    --warmup 20 --repeat 120 --rng-overheads \
    --output-dir artifacts/sr-performance --profile
```

性能入口默认统计预分配缓冲和显式随机状态的重放主核，排除编译、分配和布局转换。
`--rng-overheads` 另测默认公开调用的量化主核、Host 发起和同步耗时；上表使用公开调用的量化主核数据。
性能入口在正常退出和异常退出时均恢复调用者的当前设备与所选设备的 generator 状态。

### 公开入口端到端耗时

关闭 msprof，固定 E4M3FN、`scale_alg=1`、`axis=-1`、`max_low_bound=0`，预热 20 次、采样 240 次。
每次调用后同步，计时包含公开接口的 Host 处理、输出分配、框架状态预留、设备执行和同步等待，
输入构造与预热期间的首次编译、SDK 指纹采集和 RNG 桥接构建在采样区间外。单位为 **ms**。

| 输入类型 | Shape | 最小值（ms） | 中位数（ms） |
| --- | --- | ---: | ---: |
| FP32 | 8×4096 | 1.234 | 1.258 |
| BF16 | 8×4096 | 1.249 | 1.285 |
| BF16 | 4096×8192 | 1.257 | 1.276 |

性能数据对应上述输入、环境和计时配置；设备主核耗时与公开入口端到端延迟分别统计。
