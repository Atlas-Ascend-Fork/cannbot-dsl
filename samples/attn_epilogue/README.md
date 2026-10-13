# attn_epilogue

基于 CANNBot-DSL 的 attention 输出投影融合算子，将 inverse RoPE、两次 OCP MXFP8
激活量化和两级矩阵乘融合，面向 Ascend 950PR/950DT（ARCH 3510）。

## 算子介绍

提供单卡完整投影接口，中间工作区由算子内部管理：

| 接口 | 实现文件 | 使用场景 |
| :--- | :--- | :--- |
| `attn_epilogue` | [attn_epilogue.py](attn_epilogue.py) | 单卡完整计算，不做 TP 切分，不需要通信参数 |

固定网络维度为 `NG=8`、`F=4096`、`O_LORA=1024`、`DIM=5120`。
每个 token 有 64 个 attention head，每个 head 为 512 维；连续 8 个 head 合为一组，
共 8 组。令 `Q` 表示沿最后一维、每 32 个元素一组的 OCP MXFP8 量化：

```text
A[t, g, :] = BF16(inverse_rope(O[t, g, :]))
Y[t, g, :] = BF16(MXMatmul(Q(A[t, g, :]), WoA[g].T))
Out[t, :]  = BF16(MXMatmul(Q(concat_groups(Y[t, :, :])), WoB.T))
```

`WoA` 的逻辑 shape 为 `[8, 1024, 4096]`，`WoB` 为 `[5120, 8192]`。
两级 matmul 的权重均由调用方提前量化，接口不接收 BF16 权重。

inverse RoPE 仅作用于每个 head 的最后 64 维，前 448 维原样保留。
尾部按相邻偶数/奇数元素配对，计算 `even' = even*cos + odd*sin`、
`odd' = odd*cos - even*sin`，在 FP32 中计算后舍入为 BF16。

不包含 attention 主体、bias、残差加法或归一化；当前提供前向接口，不提供反向实现。

## 接口与参数

### 不做 TP 切分：attn_epilogue

```python
from samples.attn_epilogue.attn_epilogue import attn_epilogue

o_proj = attn_epilogue(
    o, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos,
)
```

此接口在输入所在设备上执行完整投影，支持任意 `T >= 1`（受设备内存限制），返回 BF16 `[T,5120]`。

### 参数说明

`T` 是输入的有效 token 数，固定 `N=64, D=512, ng=8, o_lora=1024, dim=5120`。
所有张量均须连续且位于同一 NPU。

| 参数 | shape | dtype / 类型 | 布局与说明 |
| :--- | :--- | :--- | :--- |
| `o` | `[T,64,512]` | `torch.bfloat16` | ND，token/head/dim 顺序 |
| `woa` | `[8,1024,4096]` | `torch.float8_e4m3fn` | ND，量化后的第一级权重，不转置 |
| `wob` | `[5120,8192]` | `torch.float8_e4m3fn` | 本版要求 ND，不转置；暂不支持 FRACTAL_NZ |
| `descale_woa` | `[8,1024,64,2]` | `torch.float8_e8m0fnu` | ND，第一级权重的 E8M0 scale |
| `descale_wob` | `[5120,128,2]` | `torch.float8_e8m0fnu` | ND，第二级权重的 E8M0 scale；不支持 NZ scale |
| `rope_sin`、`rope_cos` | `[T,64]` | `torch.float32` | ND，当前 token 的位置表，按相邻元素重复 `[s0,s0,s1,s1,...]`，cos 同理 |
| 返回值 `o_proj` | `[T,5120]` | `torch.bfloat16` | ND，输入 token 的投影结果，无对外 padding |

E8M0 scale 的最后两维将每 32 个元素对应的 scale 打包为 `[K/64, 2]`。
若权重文件以 uint8 保存 scale，使用 `.view(torch.float8_e8m0fnu)` 恢复类型，
不能用数值转换代替位模式解释。接口不接受 BF16 原始权重，也不在 forward 中重新量化权重。

### 使用约束

- 支持任意 `T >= 1`（受设备内存限制），权重及 scale 须为连续 ND 布局，不支持 NZ 输入。
- 返回结果拥有独立的输出存储。普通执行时，内部工作区按设备、当前 stream 和 T 缓存复用；
  跨 stream 的输入输出依赖由网络建立。
- 图捕获前先预热相同规格。图内工作区使用图内存池，与普通执行和其他图隔离。
- 上游输出如为 `[head,T,dim]`，应先 permute 为 `[T,head,dim]` 并连续化，不能直接 reshape。
- 首次调用包含编译和分配开销，应先预热。

## 快速开始

由平台预置 CANN、PyTorch、torch_npu 和 DSL wheel，见[依赖检查脚本](../../install_deps.sh)。
本实现使用 `@host`、独立 `cannbotdsl.compile` 及显式片上存储绑定；已验证版本为
`cannbotdsl 0.7.0+g8f313d6.0.7.x`，旧 wheel 可能缺少这些接口。

```bash
source ${install_path}/ascend-toolkit/set_env.sh
```

下面的完整示例调用 `attn_epilogue`。权重量化仅在初始化时执行；实际网络可直接加载量化权重与 scale。

安装依赖并加载 CANN 环境，将以下代码保存到仓库根目录的 `attn_epilogue_example.py`。
示例的随机权重、输入和位置表在网络中应替换为实际数据。

```python
import torch
import torch_npu

from samples.attn_epilogue.attn_epilogue import attn_epilogue


def main():
    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    T = 72

    torch.manual_seed(2026)
    wa = (torch.randn(8, 1024, 4096) * 0.3).to(torch.bfloat16).to(device)
    wb = (torch.randn(5120, 8192) * 0.2).to(torch.bfloat16).to(device)
    woa, sa = torch_npu.npu_dynamic_mx_quant(wa.reshape(8192, 4096), dst_type=292)
    wob, sb = torch_npu.npu_dynamic_mx_quant(wb, dst_type=292)
    woa = woa.reshape(8, 1024, 4096)
    descale_woa = sa.view(torch.float8_e8m0fnu).reshape(8, 1024, 64, 2)
    descale_wob = sb.view(torch.float8_e8m0fnu)

    torch.manual_seed(2027)
    o = (torch.randn(T, 64, 512) * 0.3).to(torch.bfloat16).to(device)
    positions = torch.arange(T, dtype=torch.float32)
    inv_freq = 10000.0 ** (-torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos().repeat_interleave(2, dim=-1).to(device)
    sin = angles.sin().repeat_interleave(2, dim=-1).to(device)

    o_proj = attn_epilogue(o, woa, wob, descale_woa, descale_wob, sin, cos)
    torch.npu.synchronize()
    print(f"{tuple(o_proj.shape)}, {o_proj.dtype}")


if __name__ == "__main__":
    main()
```

从仓库根目录执行，选择可用设备：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python3 attn_epilogue_example.py
```

## 精度验证

从仓库根目录执行。默认先做接口检查和一个单卡代表用例：

```bash
python3 -m pytest -q test/attn_epilogue -m 'not npu'
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python3 -m pytest -q 'test/attn_epilogue/test_attn_epilogue.py::test_public_tp1[72]'
```

执行完整单卡回归：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python3 -m pytest -q test/attn_epilogue/test_attn_epilogue.py
```

参考结果由 torch_npu 原生量化和矩阵乘构造，使用 `rtol=0, atol=0` 比较，另检查工作区复用、
输出独立性及图重放。

### 验证结果

当前测试见 [test_attn_epilogue.py](../../test/attn_epilogue/test_attn_epilogue.py)，
覆盖接口契约、T 分块边界及 padding、零值与极小值输入，并检查第一级激活量化和
中间 Y 量化的 payload/scale。最终 BF16 输出使用 `rtol=0, atol=0` 与原生参考比较。
测试还检查同一输入的输出一致性、交替输入的图重放、工作区复用及返回结果的独立存储。

单卡验证环境为 Ascend 950DT，32 Cube / 64 Vector，`cannbotdsl 0.7.0`、
`torch 2.12.0+cpu`、`torch_npu 2.12.0`，加载 CANN 9.2.0 环境。
融合算子与小算子使用相同输入和量化权重，独立进程间核对输入哈希一致。

| 验证项 | 覆盖范围 | 结果 |
| :--- | :--- | :--- |
| 最终输出与小算子参考对比 | `T=1,72,128,256` | BF16 逐位一致，最大绝对误差 0，差异位模式数 0 |
| 扩展 T 范围（超过 256） | `T=257,512,1000,1024,2048,4096,8192` | 每例 10 次预热 + 16 次图捕获调用 + 6 次重放，输出均与参考逐位一致 |
| 同输入普通执行 | 每个 T 检查 2 次，共 8 次 | 输出逐位一致 |
| 同输入图重放 | 每个 T 检查 4 次，共 16 次 | 输出逐位一致 |
| 交替输入图重放 | 每个 T 检查 4 次，共 16 次 | 均与各自参考逐位一致，无前次输入残留 |

以上是已测用例的结果，不表示穷举了所有 `T` 或所有输入值；扩展 T 的验证使用
`python test/attn_epilogue/test_attn_epilogue.py --tokens <T>`，口径与上表一致。

## 性能验证

单卡入口可用现有测试脚本采集，无需额外 benchmark 实现：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 msprof --ai-core=off --l2=off --task-time=on --application='python3 test/attn_epilogue/test_attn_epilogue.py --mode public --tokens 72' --output=profile_attn_epilogue
```

脚本先预热并校验精度，随后每次调用前使用 256 MiB FP32 张量的 ReduceSum 驱逐 L2。
统计图重放中最后 96 次 `attn_epilogue` 对应融合 kernel 的 `Task Duration(us)` 中位数，
不包含 ReduceSum、编译、分配或 Python 调度耗时。性能比较必须使用相同输入、核配额、
设备与清 L2 条件；此脚本测量融合算子，不是小算子拼接基线。

### 性能结果

下表与上述四个 T 的精度回归使用同一环境。每种实现、每个 T 采集 32 次，
取设备耗时中位数；融合与小算子分别采集 profiling，每次调用前均用 256 MiB FP32
缓冲区的 ReduceSum 驱逐 L2，并通过流水记录核对同流驱逐操作。
权重量化在初始化时完成，不计入两侧耗时。

小算子基线依次执行 inverse RoPE、`npu_dynamic_mx_quant`、
`npu_transpose_quant_batchmatmul`、第二次 `npu_dynamic_mx_quant` 和 `npu_quant_matmul`。
基线耗时按每次调用对应的五个 kernel 的 `Task Duration(us)` 求和，再取中位数；
融合耗时为 `attn_epilogue` 对应单个融合 kernel 的耗时中位数。两侧均不包含清 L2、编译、
内存分配与 Python 调度耗时，不应将该指标理解为端到端请求时延。

| T | 小算子拼接（µs） | 融合算子（µs） | 耗时降低 | 加速比 |
|---:|---:|---:|---:|---:|
| 1 | 42.065 | 34.064 | 19.02% | 1.235× |
| 72 | 53.905 | 44.677 | 17.12% | 1.207× |
| 128 | 63.078 | 52.330 | 17.04% | 1.205× |
| 256 | 84.233 | 75.684 | 10.15% | 1.113× |

耗时降低为 `(小算子耗时 - 融合耗时) / 小算子耗时`，加速比为
`小算子耗时 / 融合耗时`。实际性能随软件版本、设备核配额和运行状态变化，
应在目标环境按相同清 L2 条件重新测量。

上面的仓内命令采集 96 次融合 kernel，而上述对比每例采集 32 次；
仓内命令不包含小算子基线，不能仅凭该命令复现对比表的基线列。

## 安装包调用

源码示例与测试直接加载本仓库 sample。需要部署安装包时，使用共享
[Native 构建工程](../../net/native_package/README.md)，注册名为 `attn_epilogue`，所属组为
`ds41`：

```bash
export PYTHON=/path/to/python3.12
export CANN_ENV=/path/to/Ascend/cann/set_env.sh
source "$CANN_ENV"
export CANNBOTDSL_WHEEL=/path/to/cannbotdsl-0.7.0-cp312-cp312-manylinux_2_28_x86_64.whl
bash net/native_package/run-build.sh --operator attn_epilogue
"$PYTHON" -m pip install --no-deps --force-reinstall \
  net/native_package/output/operators/attn_epilogue/wheels/*.whl
```

`T` 参与编译（tiling 与算子形状随 `T` 变化），默认按 28/32 核配额导出
`T=1,72,128,256,512,1024,2048,4096,8192` 共 18 个配置；未被枚举的 `T` 会回退 JIT。
可通过 `CANNBOTDSL_DS41_PROFILES` 的 `attn_epilogue` 键覆盖默认集合：

```json
{"attn_epilogue": [{"block_num": 32, "tokens": [72, 256, 1024]}]}
```

安装后可设置 `CANNBOTDSL_NATIVE_BINARY_MODE=require` 验证二进制覆盖是否完整，
未命中时立即报错；`CANNBOTDSL_NATIVE_BINARY_REPORT=/path/to/native-report.json`
可保存命中记录。
